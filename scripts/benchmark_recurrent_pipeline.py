#!/usr/bin/env python3
"""Benchmark the real recurrent rollout-and-PPO pipeline.

Unlike ``benchmark_recurrent_batching.py``, this benchmark constructs simulator
worlds, collects policy-induced trajectories, merges worker evidence, computes
GAE, and performs PPO updates.  It is intended to choose a rollout-worker
topology before a large campaign, not to produce or promote a policy artifact.

The only successful stdout payload is one compact canonical JSON object.
Failures write to stderr and return non-zero without emitting a partial report.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

import torch

from evolution_sim.config.schema import SignalConfig
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_SEED_REGISTRY,
)
from evolution_sim.mind.recurrent_actor_critic import (
    GENOME_CONDITIONING_ACTOR_FILM_V1,
    GENOME_CONDITIONING_DISABLED,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.recurrent_experiment import (
    MAX_RECURRENT_ROLLOUT_WORKERS,
    OPEN_ECOLOGY_PHASE_A,
    RECURRENT_TRAINING_SCENARIOS,
    OpenEcologySignalTreatment,
    RecurrentExperimentRunner,
    build_open_ecology_training_schedule,
    build_recurrent_training_schedule,
)
from evolution_sim.mind.recurrent_genome_population import (
    RecurrentGenomePopulationMode,
)
from evolution_sim.mind.recurrent_policy import recurrent_model_state_sha256
from evolution_sim.mind.recurrent_ppo import RecurrentPPOConfig


REPORT_SCHEMA_VERSION = "mind_recurrent_end_to_end_pipeline_benchmark_v1"
MEASURED_OPERATION = "simulation_rollout_merge_gae_and_ppo_update"
INPUT_CONTRACTS = ("base", "tokenized")
GENOME_CONDITIONING_MODES = (
    GENOME_CONDITIONING_DISABLED,
    GENOME_CONDITIONING_ACTOR_FILM_V1,
)
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class PipelineBenchmarkConfigurationError(ValueError):
    """Raised when a benchmark request is internally inconsistent."""


class PipelineBenchmarkExecutionError(RuntimeError):
    """Raised when a valid benchmark cannot produce complete evidence."""


@dataclass(frozen=True, slots=True)
class PipelineBenchmarkProtocol:
    """Complete configuration for a worker-scaling benchmark."""

    worker_counts: tuple[int, ...]
    repeats: int
    updates: int
    worlds_per_update: int
    rollout_ticks: int
    scenarios: tuple[str, ...]
    device: str
    input_contract: str
    genome_conditioning: str
    genome_population_mode: str
    genome_stream_seed: int | None
    encoder_size: int
    hidden_size: int
    recurrent_layers: int
    learner_seed: int
    update_epochs: int
    sequence_minibatch_size: int
    tbptt_steps: int
    burn_in_steps: int
    preregistered_gate_member: bool = False

    def validate(self) -> None:
        _positive_integer_sequence(
            self.worker_counts,
            field="worker_counts",
            maximum=MAX_RECURRENT_ROLLOUT_WORKERS,
        )
        for field, value in (
            ("repeats", self.repeats),
            ("updates", self.updates),
            ("worlds_per_update", self.worlds_per_update),
            ("rollout_ticks", self.rollout_ticks),
            ("encoder_size", self.encoder_size),
            ("hidden_size", self.hidden_size),
            ("recurrent_layers", self.recurrent_layers),
            ("learner_seed", self.learner_seed),
            ("update_epochs", self.update_epochs),
            ("sequence_minibatch_size", self.sequence_minibatch_size),
            ("tbptt_steps", self.tbptt_steps),
        ):
            _positive_integer(value, field=field)
        if (
            isinstance(self.burn_in_steps, bool)
            or not isinstance(self.burn_in_steps, int)
            or self.burn_in_steps < 0
        ):
            raise PipelineBenchmarkConfigurationError(
                "burn_in_steps must be a non-negative integer"
            )
        if self.burn_in_steps >= self.tbptt_steps:
            raise PipelineBenchmarkConfigurationError(
                "burn_in_steps must be smaller than tbptt_steps"
            )
        if type(self.preregistered_gate_member) is not bool:
            raise PipelineBenchmarkConfigurationError(
                "preregistered_gate_member must be an exact boolean"
            )
        if not self.scenarios or len(self.scenarios) != len(set(self.scenarios)):
            raise PipelineBenchmarkConfigurationError(
                "scenarios must be non-empty and unique"
            )
        unknown_scenarios = sorted(
            set(self.scenarios) - set(RECURRENT_TRAINING_SCENARIOS)
        )
        if unknown_scenarios:
            raise PipelineBenchmarkConfigurationError(
                f"unsupported scenarios: {unknown_scenarios}"
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise PipelineBenchmarkConfigurationError(
                "device must be a non-empty string"
            )
        if self.input_contract not in INPUT_CONTRACTS:
            raise PipelineBenchmarkConfigurationError(
                f"input_contract must be one of {list(INPUT_CONTRACTS)}"
            )
        if self.genome_conditioning not in GENOME_CONDITIONING_MODES:
            raise PipelineBenchmarkConfigurationError(
                "genome_conditioning is unsupported"
            )
        try:
            population_mode = RecurrentGenomePopulationMode(self.genome_population_mode)
        except ValueError as error:
            raise PipelineBenchmarkConfigurationError(
                "genome_population_mode is unsupported"
            ) from error
        if self.genome_conditioning == GENOME_CONDITIONING_DISABLED:
            if (
                population_mode is not RecurrentGenomePopulationMode.DISABLED
                or self.genome_stream_seed is not None
            ):
                raise PipelineBenchmarkConfigurationError(
                    "disabled conditioning requires a disabled genome population "
                    "and no genome stream seed"
                )
        else:
            if population_mode not in {
                RecurrentGenomePopulationMode.HERITABLE,
                RecurrentGenomePopulationMode.ZERO_ALL,
            }:
                raise PipelineBenchmarkConfigurationError(
                    "actor_film_v1 requires heritable or zero_all genomes"
                )
            if (
                isinstance(self.genome_stream_seed, bool)
                or not isinstance(self.genome_stream_seed, int)
                or not 0 <= self.genome_stream_seed <= 2**64 - 1
            ):
                raise PipelineBenchmarkConfigurationError(
                    "active genome conditioning requires an unsigned 64-bit "
                    "genome stream seed"
                )
        if self.input_contract == "tokenized":
            if self.scenarios != ("broad",):
                raise PipelineBenchmarkConfigurationError(
                    "tokenized benchmark is the canonical broad-only open-ecology "
                    "treatment"
                )
            if self.genome_conditioning != GENOME_CONDITIONING_ACTOR_FILM_V1:
                raise PipelineBenchmarkConfigurationError(
                    "open-ecology benchmark requires actor_film_v1"
                )
            if (
                self.learner_seed
                not in OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"]
            ):
                raise PipelineBenchmarkConfigurationError(
                    "open-ecology learner seed is not registered"
                )
            if (
                self.genome_stream_seed
                not in OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"]
            ):
                raise PipelineBenchmarkConfigurationError(
                    "open-ecology genome stream seed is not registered"
                )
            if self.updates * self.worlds_per_update > len(
                OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_train"]
            ):
                raise PipelineBenchmarkConfigurationError(
                    "open-ecology benchmark exceeds the non-reused training seed "
                    "registry"
                )
        if self.preregistered_gate_member:
            mismatches = _preregistered_gate_member_mismatches(self)
            if mismatches:
                raise PipelineBenchmarkConfigurationError(
                    "preregistered gate member shape mismatch: "
                    + ", ".join(mismatches)
                )


def _preregistered_gate_member_mismatches(
    protocol: PipelineBenchmarkProtocol,
) -> tuple[str, ...]:
    expected: tuple[tuple[str, object, object], ...] = (
        ("worker_counts", protocol.worker_counts, (1, 2, 4, 8, 16)),
        ("updates", protocol.updates, 1),
        ("worlds_per_update", protocol.worlds_per_update, 16),
        ("rollout_ticks", protocol.rollout_ticks, 128),
        ("scenarios", protocol.scenarios, ("broad",)),
        ("input_contract", protocol.input_contract, "tokenized"),
        (
            "genome_conditioning",
            protocol.genome_conditioning,
            GENOME_CONDITIONING_ACTOR_FILM_V1,
        ),
        ("encoder_size", protocol.encoder_size, 256),
        ("hidden_size", protocol.hidden_size, 256),
        ("recurrent_layers", protocol.recurrent_layers, 1),
        ("update_epochs", protocol.update_epochs, 4),
        ("sequence_minibatch_size", protocol.sequence_minibatch_size, 16),
        ("tbptt_steps", protocol.tbptt_steps, 128),
        ("burn_in_steps", protocol.burn_in_steps, 16),
    )
    mismatches = [
        name for name, observed, required in expected if observed != required
    ]
    if protocol.genome_population_mode not in {
        RecurrentGenomePopulationMode.HERITABLE.value,
        RecurrentGenomePopulationMode.ZERO_ALL.value,
    }:
        mismatches.append("genome_population_mode")
    return tuple(mismatches)


def run_benchmark(
    protocol: PipelineBenchmarkProtocol,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Run every worker/repeat case and return complete JSON-safe evidence."""

    protocol.validate()
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    source = _source_provenance(root)
    device = _resolve_device(protocol.device)
    if protocol.preregistered_gate_member:
        if source["dirty"]:
            raise PipelineBenchmarkExecutionError(
                "preregistered benchmark member requires a clean exact source tree"
            )
        if device.type != "cuda":
            raise PipelineBenchmarkExecutionError(
                "preregistered benchmark member must execute on CUDA"
            )
    runtime = _runtime_provenance(device)
    signal_config = (
        OpenEcologySignalTreatment().as_signal_config()
        if protocol.input_contract == "tokenized"
        else SignalConfig()
    )
    model_config = RecurrentActorCriticConfig.for_signal_config(
        signal_config,
        encoder_size=protocol.encoder_size,
        hidden_size=protocol.hidden_size,
        recurrent_layers=protocol.recurrent_layers,
        genome_conditioning_mode=protocol.genome_conditioning,
    )
    ppo_config = (
        RecurrentPPOConfig(
            learning_rate=0.0002,
            adam_epsilon=1.0e-8,
            gamma=0.997,
            gae_lambda=0.97,
            policy_clip_range=0.15,
            value_clip_range=0.20,
            value_loss_coefficient=0.50,
            entropy_coefficient=0.02,
            update_epochs=protocol.update_epochs,
            sequence_minibatch_size=protocol.sequence_minibatch_size,
            tbptt_steps=protocol.tbptt_steps,
            burn_in_steps=protocol.burn_in_steps,
            max_gradient_norm=0.50,
            normalize_advantages=True,
            target_kl=0.02,
            learner_seed=protocol.learner_seed,
            world_balanced_loss=True,
        )
        if protocol.input_contract == "tokenized"
        else RecurrentPPOConfig(
            learner_seed=protocol.learner_seed,
            update_epochs=protocol.update_epochs,
            sequence_minibatch_size=protocol.sequence_minibatch_size,
            tbptt_steps=protocol.tbptt_steps,
            burn_in_steps=protocol.burn_in_steps,
        )
    )
    schedule = (
        build_open_ecology_training_schedule(
            training_phase=OPEN_ECOLOGY_PHASE_A,
            update_count=protocol.updates,
            worlds_per_update=protocol.worlds_per_update,
            rollout_ticks=protocol.rollout_ticks,
            learner_seed=protocol.learner_seed,
            genome_stream_seed=protocol.genome_stream_seed,
            genome_population_mode=protocol.genome_population_mode,
        )
        if protocol.input_contract == "tokenized"
        else build_recurrent_training_schedule(
            update_count=protocol.updates,
            worlds_per_update=protocol.worlds_per_update,
            rollout_ticks=protocol.rollout_ticks,
            scenarios=protocol.scenarios,
            genome_stream_seed=protocol.genome_stream_seed,
            genome_population_mode=protocol.genome_population_mode,
        )
    )

    cases: list[dict[str, object]] = []
    expected_model_digest: str | None = None
    expected_semantic_digest: str | None = None
    for worker_count in protocol.worker_counts:
        samples: list[dict[str, object]] = []
        for repeat_index in range(protocol.repeats):
            runner = RecurrentExperimentRunner(
                learner_seed=protocol.learner_seed,
                device=device,
                model_config=model_config,
                ppo_config=ppo_config,
                rollout_workers=worker_count,
            )
            _synchronize(device)
            started_ns = time.perf_counter_ns()
            result = runner.run(schedule)
            _synchronize(device)
            elapsed_ns = time.perf_counter_ns() - started_ns
            if elapsed_ns <= 0:
                raise PipelineBenchmarkExecutionError(
                    "monotonic timer returned a non-positive duration"
                )
            model_digest = recurrent_model_state_sha256(runner.model)
            semantic_digest = stable_payload_digest(
                {
                    "model_state_sha256": model_digest,
                    "total_worlds": result.total_worlds,
                    "total_transitions": result.total_transitions,
                    "updates": [
                        {
                            "update_index": update.update_index,
                            "tasks": [asdict(task) for task in update.tasks],
                            "rollout": asdict(update.rollout),
                            "optimizer": asdict(update.optimizer),
                        }
                        for update in result.updates
                    ],
                }
            )
            if expected_model_digest is None:
                expected_model_digest = model_digest
                expected_semantic_digest = semantic_digest
            elif (
                model_digest != expected_model_digest
                or semantic_digest != expected_semantic_digest
            ):
                raise PipelineBenchmarkExecutionError(
                    "worker or repeat configuration changed deterministic training "
                    "evidence"
                )
            elapsed_seconds = elapsed_ns / 1_000_000_000.0
            samples.append(
                {
                    "repeat_index": repeat_index,
                    "elapsed_ns": elapsed_ns,
                    "worlds_per_second": result.total_worlds / elapsed_seconds,
                    "transitions_per_second": (
                        result.total_transitions / elapsed_seconds
                    ),
                    "total_worlds": result.total_worlds,
                    "total_transitions": result.total_transitions,
                    "final_model_state_sha256": model_digest,
                    "semantic_evidence_sha256": semantic_digest,
                }
            )
            del runner, result
        cases.append(_case_payload(worker_count=worker_count, samples=samples))
    baseline_elapsed_ns = float(cases[0]["elapsed_ns"]["median"])  # type: ignore[index]
    baseline_workers = int(cases[0]["rollout_workers"])
    for case in cases:
        case_elapsed_ns = float(case["elapsed_ns"]["median"])  # type: ignore[index]
        speedup = baseline_elapsed_ns / case_elapsed_ns
        worker_count = int(case["rollout_workers"])
        case["speedup_vs_first_case"] = speedup
        case["parallel_efficiency_vs_first_case"] = speedup / (
            worker_count / baseline_workers
        )

    final_source = _source_provenance(root)
    if final_source != source:
        raise PipelineBenchmarkExecutionError(
            "source provenance changed while the benchmark was running"
        )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": {
            "measured_operation": MEASURED_OPERATION,
            "world_construction_included": True,
            "world_ticks_included": True,
            "observation_encoding_included": True,
            "policy_action_sampling_included": True,
            "ordered_worker_merge_included": True,
            "gae_included": True,
            "ppo_update_included": True,
            "artifact_serialization_included": False,
            "evaluation_included": False,
            "policy_promotion_authorized": False,
        },
        "source": source,
        "runtime": runtime,
        "protocol": {
            **asdict(protocol),
            "device_resolved": str(device),
            "model_config": asdict(model_config),
            "ppo_config": asdict(ppo_config),
            "scheduled_worlds": protocol.updates * protocol.worlds_per_update,
            "scheduled_world_ticks": (
                protocol.updates
                * protocol.worlds_per_update
                * (protocol.rollout_ticks + 1)
            ),
            "schedule_contract": (
                "canonical_open_ecology_phase_a_broad_treatment_v1"
                if protocol.input_contract == "tokenized"
                else "legacy_recurrent_training_schedule"
            ),
            "training_phase": (
                OPEN_ECOLOGY_PHASE_A
                if protocol.input_contract == "tokenized"
                else None
            ),
            "timing_boundary": "runner.run_only",
            "fresh_identically_seeded_runner_per_sample": True,
        },
        "determinism": {
            "cross_worker_model_state_match": True,
            "cross_worker_semantic_evidence_match": True,
            "final_model_state_sha256": expected_model_digest,
            "semantic_evidence_sha256": expected_semantic_digest,
        },
        "preregistered_gate": {
            "member_shape_valid": not _preregistered_gate_member_mismatches(protocol),
            "member_claimed": protocol.preregistered_gate_member,
            "population_mode": protocol.genome_population_mode,
            "required_population_modes": [
                RecurrentGenomePopulationMode.HERITABLE.value,
                RecurrentGenomePopulationMode.ZERO_ALL.value,
            ],
            "complete_pair_required": True,
            "complete_gate_claimed": False,
        },
        "cases": cases,
    }
    json.dumps(report, allow_nan=False)
    return report


def _case_payload(
    *,
    worker_count: int,
    samples: Sequence[dict[str, object]],
) -> dict[str, object]:
    elapsed = [int(sample["elapsed_ns"]) for sample in samples]
    worlds_per_second = [float(sample["worlds_per_second"]) for sample in samples]
    transitions_per_second = [
        float(sample["transitions_per_second"]) for sample in samples
    ]
    return {
        "rollout_workers": worker_count,
        "samples": list(samples),
        "elapsed_ns": _distribution(elapsed),
        "worlds_per_second": _distribution(worlds_per_second),
        "transitions_per_second": _distribution(transitions_per_second),
    }


def _distribution(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        raise PipelineBenchmarkExecutionError("timing distribution cannot be empty")
    parsed = [float(value) for value in values]
    return {
        "minimum": min(parsed),
        "median": float(statistics.median(parsed)),
        "maximum": max(parsed),
        "mean": statistics.fmean(parsed),
    }


def canonical_json(report: dict[str, object]) -> str:
    return json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark complete recurrent simulator rollout, merge, GAE, and PPO "
            "updates across worker counts."
        )
    )
    parser.add_argument("--worker-counts", required=True)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--updates", required=True, type=int)
    parser.add_argument("--worlds-per-update", required=True, type=int)
    parser.add_argument("--rollout-ticks", required=True, type=int)
    parser.add_argument("--scenarios", default="broad")
    parser.add_argument("--device", required=True)
    parser.add_argument("--input-contract", choices=INPUT_CONTRACTS, default="base")
    parser.add_argument(
        "--genome-conditioning",
        choices=GENOME_CONDITIONING_MODES,
        default=GENOME_CONDITIONING_DISABLED,
    )
    parser.add_argument(
        "--genome-population-mode",
        choices=tuple(mode.value for mode in RecurrentGenomePopulationMode),
        default=RecurrentGenomePopulationMode.DISABLED.value,
    )
    parser.add_argument("--genome-stream-seed", type=int)
    parser.add_argument("--encoder-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--recurrent-layers", type=int, default=1)
    parser.add_argument(
        "--learner-seed",
        type=int,
        default=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
    )
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--sequence-minibatch-size", type=int, default=64)
    parser.add_argument("--tbptt-steps", type=int, default=32)
    parser.add_argument("--burn-in-steps", type=int, default=8)
    parser.add_argument("--preregistered-gate-member", action="store_true")
    return parser


def protocol_from_args(args: argparse.Namespace) -> PipelineBenchmarkProtocol:
    protocol = PipelineBenchmarkProtocol(
        worker_counts=_parse_positive_integer_csv(
            args.worker_counts,
            field="worker_counts",
        ),
        repeats=args.repeats,
        updates=args.updates,
        worlds_per_update=args.worlds_per_update,
        rollout_ticks=args.rollout_ticks,
        scenarios=tuple(
            part.strip() for part in args.scenarios.split(",") if part.strip()
        ),
        device=args.device,
        input_contract=args.input_contract,
        genome_conditioning=args.genome_conditioning,
        genome_population_mode=args.genome_population_mode,
        genome_stream_seed=args.genome_stream_seed,
        encoder_size=args.encoder_size,
        hidden_size=args.hidden_size,
        recurrent_layers=args.recurrent_layers,
        learner_seed=args.learner_seed,
        update_epochs=args.update_epochs,
        sequence_minibatch_size=args.sequence_minibatch_size,
        tbptt_steps=args.tbptt_steps,
        burn_in_steps=args.burn_in_steps,
        preregistered_gate_member=args.preregistered_gate_member,
    )
    protocol.validate()
    return protocol


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_benchmark(protocol_from_args(args))
        payload = canonical_json(report)
    except PipelineBenchmarkConfigurationError as error:
        print(f"benchmark configuration rejected: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(
            "benchmark failed closed without a report: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(payload)
    return 0


def _resolve_device(specification: str) -> torch.device:
    normalized = specification.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            normalized = "cuda"
        elif torch.backends.mps.is_available():
            normalized = "mps"
        else:
            normalized = "cpu"
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as error:
        raise PipelineBenchmarkConfigurationError(
            f"invalid torch device {specification!r}"
        ) from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise PipelineBenchmarkConfigurationError(
                "requested CUDA device is unavailable"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise PipelineBenchmarkConfigurationError(
                "requested CUDA device index is unavailable"
            )
        device = torch.device("cuda", index)
    elif device.type == "mps":
        if device.index is not None or not torch.backends.mps.is_available():
            raise PipelineBenchmarkConfigurationError(
                "requested MPS device is unavailable"
            )
    elif device.type != "cpu" or device.index is not None:
        raise PipelineBenchmarkConfigurationError(
            "device must resolve to cpu, mps, or cuda"
        )
    return device


def _runtime_provenance(device: torch.device) -> dict[str, object]:
    cuda: dict[str, object] | None = None
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        cuda = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(index)),
        }
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda": cuda,
    }


def _source_provenance(repository_root: Path) -> dict[str, object]:
    try:
        root = Path(
            _git_output(repository_root, "rev-parse", "--show-toplevel")
        ).resolve()
        commit_sha = _git_output(repository_root, "rev-parse", "HEAD")
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise PipelineBenchmarkExecutionError(
            "exact Git source provenance could not be captured"
        ) from error
    if root != repository_root.resolve():
        raise PipelineBenchmarkExecutionError(
            "repository_root is not the Git top level"
        )
    if _GIT_COMMIT_PATTERN.fullmatch(commit_sha) is None:
        raise PipelineBenchmarkExecutionError(
            "Git HEAD is not an exact 40-character SHA"
        )
    return {
        "commit_sha": commit_sha,
        "dirty": bool(status),
        "capture_contract": "git_head_plus_porcelain_dirty_flag_v1",
    }


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _parse_positive_integer_csv(value: str, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value.strip():
        raise PipelineBenchmarkConfigurationError(f"{field} cannot be empty")
    try:
        parsed = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise PipelineBenchmarkConfigurationError(
            f"{field} must contain only integers"
        ) from error
    _positive_integer_sequence(parsed, field=field)
    return parsed


def _positive_integer_sequence(
    values: Sequence[int],
    *,
    field: str,
    maximum: int | None = None,
) -> None:
    if not values or len(values) != len(set(values)):
        raise PipelineBenchmarkConfigurationError(
            f"{field} must be non-empty and unique"
        )
    for value in values:
        _positive_integer(value, field=field)
        if maximum is not None and value > maximum:
            raise PipelineBenchmarkConfigurationError(
                f"{field} values must not exceed {maximum}"
            )


def _positive_integer(value: object, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineBenchmarkConfigurationError(
            f"{field} must contain positive integers"
        )


if __name__ == "__main__":
    raise SystemExit(main())
