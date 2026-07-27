#!/usr/bin/env python3
"""Measure batched recurrent-model inference without running or mutating a sim.

This benchmark deliberately measures only
``PublicRecurrentActorCritic.forward_sequence`` with one time step per call.
It does not construct worlds, advance agents, sample actions, update recurrent
state stores, train a policy, or estimate end-to-end simulator throughput.

Example:

    PYTHONPATH=python .venv/bin/python scripts/benchmark_recurrent_batching.py \
      --widths 128,256,512 \
      --batch-sizes 1,8,32,128,512 \
      --device cuda:0 \
      --dtype float32 \
      --input-contract tokenized \
      --genome-conditioning actor_film_v1 \
      --warmup 20 \
      --repeats 100

The only successful stdout payload is one compact canonical JSON object.
Configuration and runtime failures write to stderr and return non-zero without
emitting a partial report.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any

import torch

from evolution_sim.config.schema import SignalConfig
from evolution_sim.mind.recurrent_actor_critic import (
    ACTION_COUNT,
    GENOME_CONDITIONING_ACTOR_FILM_V1,
    GENOME_CONDITIONING_DISABLED,
    PREVIOUS_PUBLIC_FEEDBACK_SIZE,
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.recurrent_genome import (
    RECURRENT_CONTROLLER_GENOME_SIZE,
    RECURRENT_CONTROLLER_MAX_HIDDEN_DIMENSION,
)


REPORT_SCHEMA_VERSION = "mind_recurrent_batch_saturation_benchmark_v1"
MEASURED_OPERATION = "PublicRecurrentActorCritic.forward_sequence"
TIME_STEPS_PER_CALL = 1
INPUT_CONTRACTS = ("base", "tokenized")
GENOME_CONDITIONING_MODES = (
    GENOME_CONDITIONING_DISABLED,
    GENOME_CONDITIONING_ACTOR_FILM_V1,
)
DTYPE_NAMES = ("float16", "bfloat16", "float32", "float64")
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class BenchmarkConfigurationError(ValueError):
    """Raised when a benchmark request cannot be executed as specified."""


class BenchmarkExecutionError(RuntimeError):
    """Raised when a valid request cannot complete without partial evidence."""


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    """Complete, explicit protocol for one benchmark report."""

    widths: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    device: str
    dtype: str
    warmup: int
    repeats: int
    input_contract: str
    genome_conditioning: str
    initialization_seed: int = 1729

    def validate(self) -> None:
        _validate_positive_integer_sequence(
            self.widths,
            field="widths",
            maximum=RECURRENT_CONTROLLER_MAX_HIDDEN_DIMENSION,
        )
        _validate_positive_integer_sequence(
            self.batch_sizes,
            field="batch_sizes",
        )
        if not isinstance(self.device, str) or not self.device.strip():
            raise BenchmarkConfigurationError("device must be a non-empty string")
        if self.dtype not in DTYPE_NAMES:
            raise BenchmarkConfigurationError(
                f"dtype must be one of {list(DTYPE_NAMES)}"
            )
        if (
            isinstance(self.warmup, bool)
            or not isinstance(self.warmup, int)
            or self.warmup < 0
        ):
            raise BenchmarkConfigurationError("warmup must be a non-negative integer")
        if (
            isinstance(self.repeats, bool)
            or not isinstance(self.repeats, int)
            or self.repeats <= 0
        ):
            raise BenchmarkConfigurationError("repeats must be a positive integer")
        if self.input_contract not in INPUT_CONTRACTS:
            raise BenchmarkConfigurationError(
                f"input_contract must be one of {list(INPUT_CONTRACTS)}"
            )
        if self.genome_conditioning not in GENOME_CONDITIONING_MODES:
            raise BenchmarkConfigurationError(
                f"genome_conditioning must be one of {list(GENOME_CONDITIONING_MODES)}"
            )
        if (
            isinstance(self.initialization_seed, bool)
            or not isinstance(self.initialization_seed, int)
            or self.initialization_seed < 0
            or self.initialization_seed > (2**63 - 1)
        ):
            raise BenchmarkConfigurationError(
                "initialization_seed must be in [0, 2**63 - 1]"
            )


def run_benchmark(
    protocol: BenchmarkProtocol,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Run a complete benchmark in memory and return a JSON-safe report."""

    protocol.validate()
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    source = _source_provenance(root)
    device = _resolve_device(protocol.device)
    dtype = _DTYPES[protocol.dtype]
    runtime = _runtime_provenance(
        requested_device=protocol.device,
        resolved_device=device,
        dtype_name=protocol.dtype,
    )

    measurements: list[dict[str, object]] = []
    for width in protocol.widths:
        for batch_size in protocol.batch_sizes:
            measurements.append(
                _benchmark_case(
                    protocol,
                    width=width,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                )
            )

    final_source = _source_provenance(root)
    if final_source["commit_sha"] != source["commit_sha"]:
        raise BenchmarkExecutionError(
            "source commit changed while the benchmark was running"
        )

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": {
            "measured_operation": MEASURED_OPERATION,
            "time_steps_per_call": TIME_STEPS_PER_CALL,
            "row_definition": "one_time_step_for_one_batch_member",
            "model_forward_only": True,
            "end_to_end_simulator_throughput_measured": False,
            "end_to_end_simulator_claim_authorized": False,
            "excluded_work": [
                "world_construction",
                "world_step",
                "observation_construction",
                "action_sampling",
                "action_resolution",
                "recurrent_state_store_commit",
                "training",
                "checkpoint_io",
            ],
        },
        "source": source,
        "runtime": runtime,
        "protocol": {
            "widths": list(protocol.widths),
            "width_policy": "encoder_size_equals_hidden_size",
            "batch_sizes": list(protocol.batch_sizes),
            "device_requested": protocol.device,
            "device_resolved": str(device),
            "dtype": protocol.dtype,
            "warmup_calls_per_case": protocol.warmup,
            "measured_calls_per_case": protocol.repeats,
            "time_steps_per_call": TIME_STEPS_PER_CALL,
            "input_contract": protocol.input_contract,
            "genome_conditioning": protocol.genome_conditioning,
            "initialization_seed": protocol.initialization_seed,
            "inference_mode": True,
            "synchronization": _synchronization_name(device),
        },
        "measurements": measurements,
    }
    _assert_json_safe(report)
    return report


def canonical_json(report: dict[str, object]) -> str:
    """Serialize a complete report without NaN, whitespace, or key-order drift."""

    return json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one-step PublicRecurrentActorCritic.forward_sequence "
            "batch saturation. This is not an end-to-end simulator benchmark."
        )
    )
    parser.add_argument(
        "--widths",
        required=True,
        help=(
            "Comma-separated positive widths. Each width is used for both the "
            "encoder and recurrent hidden state."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        required=True,
        help="Comma-separated positive batch sizes.",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Explicit torch device such as cpu, mps, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        required=True,
        choices=DTYPE_NAMES,
        help="Model and floating-input dtype.",
    )
    parser.add_argument(
        "--warmup",
        required=True,
        type=int,
        help="Untimed warmup calls per width and batch-size case.",
    )
    parser.add_argument(
        "--repeats",
        required=True,
        type=int,
        help="Timed calls per width and batch-size case.",
    )
    parser.add_argument(
        "--input-contract",
        default="base",
        choices=INPUT_CONTRACTS,
        help="Public input contract to instantiate.",
    )
    parser.add_argument(
        "--genome-conditioning",
        default=GENOME_CONDITIONING_DISABLED,
        choices=GENOME_CONDITIONING_MODES,
        help=(
            "Disabled or actor_film_v1. The FiLM mode receives an explicit "
            "bounded genome tensor for every row."
        ),
    )
    parser.add_argument(
        "--initialization-seed",
        default=1729,
        type=int,
        help="Model and synthetic-input initialization seed.",
    )
    return parser


def protocol_from_args(args: argparse.Namespace) -> BenchmarkProtocol:
    protocol = BenchmarkProtocol(
        widths=_parse_positive_integer_csv(args.widths, field="widths"),
        batch_sizes=_parse_positive_integer_csv(
            args.batch_sizes,
            field="batch_sizes",
        ),
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        input_contract=args.input_contract,
        genome_conditioning=args.genome_conditioning,
        initialization_seed=args.initialization_seed,
    )
    protocol.validate()
    return protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        protocol = protocol_from_args(args)
        report = run_benchmark(protocol)
        payload = canonical_json(report)
    except BenchmarkConfigurationError as error:
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


def _benchmark_case(
    protocol: BenchmarkProtocol,
    *,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    signal_config = SignalConfig(
        communication_signal_emission_enabled=(protocol.input_contract == "tokenized")
    )
    try:
        model_config = RecurrentActorCriticConfig.for_signal_config(
            signal_config,
            encoder_size=width,
            hidden_size=width,
            genome_conditioning_mode=protocol.genome_conditioning,
        )
        model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=protocol.initialization_seed,
        )
        model = model.to(device=device, dtype=dtype).eval()
        inputs = _case_inputs(
            model,
            protocol=protocol,
            width=width,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
    except (RuntimeError, ValueError) as error:
        raise BenchmarkConfigurationError(
            _case_failure_prefix(width, batch_size, device, dtype)
            + f" could not be constructed: {error}"
        ) from error

    def forward_once() -> None:
        result = model.forward_sequence(
            inputs["observations"],
            inputs["action_masks"],
            inputs["previous_feedback"],
            genome_values=inputs["genome_values"],
            initial_state=inputs["initial_state"],
            episode_starts=inputs["episode_starts"],
        )
        del result

    try:
        with torch.inference_mode():
            for _ in range(protocol.warmup):
                forward_once()
                _synchronize(device)
            _synchronize(device)
            _reset_peak_memory(device)
            latency_samples_ns: list[int] = []
            for _ in range(protocol.repeats):
                _synchronize(device)
                started_ns = time.perf_counter_ns()
                forward_once()
                _synchronize(device)
                elapsed_ns = time.perf_counter_ns() - started_ns
                if elapsed_ns <= 0:
                    raise BenchmarkExecutionError(
                        "monotonic timer returned a non-positive duration"
                    )
                latency_samples_ns.append(elapsed_ns)
            peak_memory = _peak_memory_payload(device)
    except (RuntimeError, ValueError) as error:
        raise BenchmarkConfigurationError(
            _case_failure_prefix(width, batch_size, device, dtype)
            + f" is unsupported by this runtime: {error}"
        ) from error

    latency = _latency_distribution(latency_samples_ns)
    rows_per_second = _throughput_distribution(
        batch_size=batch_size,
        latency_samples_ns=latency_samples_ns,
    )
    return {
        "encoder_size": width,
        "hidden_size": width,
        "batch_size": batch_size,
        "time_steps": TIME_STEPS_PER_CALL,
        "rows_per_call": batch_size * TIME_STEPS_PER_CALL,
        "public_input_schema_version": model_config.public_input_schema_version,
        "public_input_size": model_config.public_input_size,
        "action_count": ACTION_COUNT,
        "previous_feedback_size": PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        "genome_conditioning_mode": model_config.genome_conditioning_mode,
        "genome_values_supplied": inputs["genome_values"] is not None,
        "genome_size": (
            RECURRENT_CONTROLLER_GENOME_SIZE
            if inputs["genome_values"] is not None
            else None
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "latency_ns": latency,
        "rows_per_second": rows_per_second,
        "peak_memory": peak_memory,
    }


def _case_inputs(
    model: PublicRecurrentActorCritic,
    *,
    protocol: BenchmarkProtocol,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | None]:
    case_seed = (protocol.initialization_seed + width * 1_000_003 + batch_size * 97) % (
        2**63
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case_seed)
    observations = (
        torch.rand(
            (
                TIME_STEPS_PER_CALL,
                batch_size,
                model.config.public_input_size,
            ),
            generator=generator,
            dtype=torch.float32,
        )
        .mul_(2.0)
        .sub_(1.0)
        .to(device=device, dtype=dtype)
    )
    action_masks = torch.ones(
        TIME_STEPS_PER_CALL,
        batch_size,
        ACTION_COUNT,
        dtype=torch.bool,
        device=device,
    )
    previous_feedback = torch.zeros(
        TIME_STEPS_PER_CALL,
        batch_size,
        PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        dtype=dtype,
        device=device,
    )
    episode_starts = torch.zeros(
        TIME_STEPS_PER_CALL,
        batch_size,
        dtype=torch.bool,
        device=device,
    )
    genome_values: torch.Tensor | None = None
    if protocol.genome_conditioning == GENOME_CONDITIONING_ACTOR_FILM_V1:
        genome_values = (
            torch.rand(
                (
                    TIME_STEPS_PER_CALL,
                    batch_size,
                    RECURRENT_CONTROLLER_GENOME_SIZE,
                ),
                generator=generator,
                dtype=torch.float32,
            )
            .mul_(0.7)
            .sub_(0.35)
            .to(device=device, dtype=dtype)
        )
    return {
        "observations": observations,
        "action_masks": action_masks,
        "previous_feedback": previous_feedback,
        "episode_starts": episode_starts,
        "genome_values": genome_values,
        "initial_state": model.initial_state(batch_size),
    }


def _source_provenance(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve()
    try:
        resolved_root = Path(
            _git_output(root, "rev-parse", "--show-toplevel")
        ).resolve()
        commit_sha = _git_output(root, "rev-parse", "HEAD")
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkExecutionError(
            "exact Git source provenance could not be captured"
        ) from error
    if resolved_root != root:
        raise BenchmarkExecutionError(
            f"repository_root is not the Git top level: {root}"
        )
    if _GIT_COMMIT_PATTERN.fullmatch(commit_sha) is None:
        raise BenchmarkExecutionError("Git HEAD is not an exact 40-character SHA")
    return {
        "commit_sha": commit_sha,
        "dirty": bool(status),
        "capture_contract": "git_head_plus_porcelain_dirty_flag_v1",
    }


def _runtime_provenance(
    *,
    requested_device: str,
    resolved_device: torch.device,
    dtype_name: str,
) -> dict[str, object]:
    cuda_device: dict[str, object] | None = None
    cuda_driver_version: str | None = None
    if resolved_device.type == "cuda":
        index = _cuda_index(resolved_device)
        properties = torch.cuda.get_device_properties(index)
        cuda_device = {
            "index": index,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
        }
        cuda_driver_version = _nvidia_driver_version(index)

    deterministic_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else None
    )
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
        },
        "torch": {
            "version": str(torch.__version__),
            "git_version": getattr(torch.version, "git_version", None),
            "debug_build": getattr(torch.version, "debug", None),
            "thread_count": torch.get_num_threads(),
            "interop_thread_count": torch.get_num_interop_threads(),
        },
        "device": {
            "requested": requested_device,
            "resolved": str(resolved_device),
            "type": resolved_device.type,
            "dtype": dtype_name,
            "name": _device_name(resolved_device),
        },
        "cuda": {
            "torch_build_version": torch.version.cuda,
            "runtime_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "selected_device": cuda_device,
            "driver_version": cuda_driver_version,
        },
        "cudnn": {
            "available": torch.backends.cudnn.is_available(),
            "enabled": torch.backends.cudnn.enabled,
            "version": torch.backends.cudnn.version(),
            "deterministic": torch.backends.cudnn.deterministic,
            "benchmark": torch.backends.cudnn.benchmark,
        },
        "mps": {
            "built": torch.backends.mps.is_built(),
            "available": torch.backends.mps.is_available(),
        },
        "determinism": {
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_algorithms_warn_only": deterministic_warn_only,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "tf32": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


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
        requested = torch.device(normalized)
    except (RuntimeError, ValueError) as error:
        raise BenchmarkConfigurationError(
            f"invalid torch device {specification!r}"
        ) from error
    if requested.type not in {"cpu", "cuda", "mps"}:
        raise BenchmarkConfigurationError("device type must be cpu, cuda, or mps")
    if requested.type == "cpu":
        if requested.index is not None:
            raise BenchmarkConfigurationError("indexed CPU devices are unsupported")
        return requested
    if requested.type == "mps":
        if requested.index is not None:
            raise BenchmarkConfigurationError("indexed MPS devices are unsupported")
        if not torch.backends.mps.is_available():
            raise BenchmarkConfigurationError("requested MPS device is unavailable")
        return requested
    if not torch.cuda.is_available():
        raise BenchmarkConfigurationError("requested CUDA device is unavailable")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise BenchmarkConfigurationError(
            f"CUDA device index {index} is outside the available range"
        )
    resolved = torch.device("cuda", index)
    try:
        torch.empty(1, device=resolved)
        torch.cuda.synchronize(resolved)
    except RuntimeError as error:
        raise BenchmarkConfigurationError(
            f"CUDA device {resolved} cannot execute torch operations"
        ) from error
    return resolved


def _nvidia_driver_version(index: int) -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
                f"--id={index}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkExecutionError("CUDA driver provenance query failed") from error
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise BenchmarkExecutionError(
            "CUDA driver provenance did not resolve to one version"
        )
    version = next(iter(versions))
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version) is None:
        raise BenchmarkExecutionError("CUDA driver version is malformed")
    return version


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _latency_distribution(samples_ns: Sequence[int]) -> dict[str, object]:
    if not samples_ns or any(
        isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0
        for sample in samples_ns
    ):
        raise BenchmarkExecutionError(
            "latency samples must be non-empty positive integer nanoseconds"
        )
    sorted_samples = sorted(samples_ns)
    return {
        "samples": list(samples_ns),
        "minimum": sorted_samples[0],
        "median": statistics.median(sorted_samples),
        "p95": _percentile(sorted_samples, 0.95),
        "maximum": sorted_samples[-1],
        "mean": statistics.fmean(sorted_samples),
    }


def _throughput_distribution(
    *,
    batch_size: int,
    latency_samples_ns: Sequence[int],
) -> dict[str, float]:
    sample_rates = [
        batch_size * 1_000_000_000.0 / latency_ns for latency_ns in latency_samples_ns
    ]
    sorted_rates = sorted(sample_rates)
    return {
        "minimum": sorted_rates[0],
        "median": float(statistics.median(sorted_rates)),
        "maximum": sorted_rates[-1],
        "mean": statistics.fmean(sorted_rates),
    }


def _percentile(sorted_values: Sequence[int], quantile: float) -> float:
    if not sorted_values:
        raise BenchmarkExecutionError("percentile input cannot be empty")
    if not 0.0 <= quantile <= 1.0:
        raise BenchmarkExecutionError("percentile quantile must be in [0, 1]")
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * (position - lower_index)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_payload(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {
            "available": False,
            "metric": None,
            "peak_vram_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "available": True,
        "metric": "torch_cuda_allocator_max_memory_allocated",
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _synchronization_name(device: torch.device) -> str:
    if device.type == "cuda":
        return "torch.cuda.synchronize_before_and_after_each_timed_call"
    if device.type == "mps":
        return "torch.mps.synchronize_before_and_after_each_timed_call"
    return "cpu_operations_are_synchronous"


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(_cuda_index(device))
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine() or "cpu"


def _cuda_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise BenchmarkExecutionError("CUDA index requested for a non-CUDA device")
    return torch.cuda.current_device() if device.index is None else device.index


def _case_failure_prefix(
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    return f"width={width}, batch_size={batch_size}, device={device}, dtype={dtype}"


def _parse_positive_integer_csv(value: str, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkConfigurationError(f"{field} cannot be empty")
    pieces = value.split(",")
    if any(not piece.strip() for piece in pieces):
        raise BenchmarkConfigurationError(
            f"{field} must be a comma-separated list without empty entries"
        )
    try:
        parsed = tuple(int(piece.strip()) for piece in pieces)
    except ValueError as error:
        raise BenchmarkConfigurationError(
            f"{field} must contain only integers"
        ) from error
    _validate_positive_integer_sequence(parsed, field=field)
    return parsed


def _validate_positive_integer_sequence(
    values: tuple[int, ...],
    *,
    field: str,
    maximum: int | None = None,
) -> None:
    if not isinstance(values, tuple) or not values:
        raise BenchmarkConfigurationError(f"{field} must be a non-empty tuple")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise BenchmarkConfigurationError(
            f"{field} must contain only positive integers"
        )
    if len(values) != len(set(values)):
        raise BenchmarkConfigurationError(f"{field} cannot contain duplicates")
    if maximum is not None and any(value > maximum for value in values):
        raise BenchmarkConfigurationError(f"{field} cannot exceed {maximum}")


def _assert_json_safe(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise BenchmarkExecutionError("benchmark report is not strict JSON") from error


if __name__ == "__main__":
    raise SystemExit(main())
