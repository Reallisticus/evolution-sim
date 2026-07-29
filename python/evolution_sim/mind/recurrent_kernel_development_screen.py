"""Bounded, non-authoritative screen for recurrent inference kernels.

This module deliberately reuses the numerical and semantic comparators from
the sealed Phase-A D04 implementation without producing a D04 report or
writing into an authority namespace.  Candidate kernels are injected only
inside one child process and are always restored before that process exits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import secrets
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from evolution_sim.mind import recurrent_actor_critic
from evolution_sim.mind import open_ecology_phase_a_behavioral_evidence
from evolution_sim.mind import recurrent_experiment
from evolution_sim.mind import recurrent_rollout
from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
    _d4_collector_equivalence,
    _d4_collector_path_facts,
    _d4_phase_a_collection_inputs,
    _isolated_torch_process_state,
    _run_d4_equivalence,
    _validated_d4_bucket_matrix,
    _validated_d4_timed_samples,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
    RECURRENT_NUMERIC_KERNEL_VERSION,
)
from evolution_sim.mind.recurrent_experiment import (
    _collect_recurrent_rollout_task,
)
from evolution_sim.mind.recurrent_rollout import (
    RecurrentFixedBatchRuntimeContract,
)


RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SCHEMA_VERSION = (
    "mind_v3_recurrent_kernel_development_screen_v1"
)
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES = (
    "per_row_bmm_manual_gru_baseline_v1",
    "native_linear_manual_gru_v1",
    "native_linear_fused_gru_speed_ceiling_v1",
    "fixed_row_tile_4_linear_manual_gru_v1",
)
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE = {
    "worlds": 1,
    "rollout_ticks": 128,
    "initial_agents": 64,
}
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS = 5
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_WARMUP_PAIRS = 2
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_NUMERIC_HEADROOM = 0.75
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP = 0.03
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_PAIRED_WINS = 4
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MAXIMUM_RSS_RATIO = 1.10
_SOURCE_SHA_LENGTH = 40
_TILE_ROWS = 4
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_DEVELOPMENT_SCREEN_ROOT = (
    _REPOSITORY_ROOT / "output" / "open-ecology" / "development-screens"
)
_DEVELOPMENT_SCREEN_ROOT_ENV = "EVOLUTION_SIM_OPEN_ECOLOGY_DEVELOPMENT_SCREEN_ROOT"
_CHILD_TIMEOUT_SECONDS = 30 * 60
_CANDIDATE_SELECTABLE = {
    "per_row_bmm_manual_gru_baseline_v1": False,
    "native_linear_manual_gru_v1": True,
    "native_linear_fused_gru_speed_ceiling_v1": False,
    "fixed_row_tile_4_linear_manual_gru_v1": False,
}
_CANDIDATE_IMPLEMENTATION = {
    "per_row_bmm_manual_gru_baseline_v1": {
        "linear_forward": "per_row_bmm",
        "gru_forward": "manual_equations",
        "purpose": "frozen_timing_and_semantic_baseline",
    },
    "native_linear_manual_gru_v1": {
        "linear_forward": "torch_functional_linear",
        "gru_forward": "manual_equations",
        "purpose": "selectable_kernel_candidate",
    },
    "native_linear_fused_gru_speed_ceiling_v1": {
        "linear_forward": "torch_functional_linear",
        "gru_forward": "torch_native_fused_gru",
        "purpose": "nonselectable_speed_ceiling",
    },
    "fixed_row_tile_4_linear_manual_gru_v1": {
        "linear_forward": "fixed_four_row_torch_functional_linear_tiles",
        "gru_forward": "manual_equations",
        "purpose": "nonselectable_architecture_probe",
    },
}
_CORE_COMPARISON_COUNT = 1 * 128 * 64
_BUCKET_COMPARISON_COUNT = 1224
_HIDDEN_SIZE = 256


class RecurrentKernelDevelopmentScreenError(ValueError):
    """Raised when the development-only screen contract fails closed."""


def _validated_linear_inputs(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None,
) -> None:
    if inputs.ndim < 1 or weight.ndim != 2:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate linear inputs and weight must be ranked"
        )
    if inputs.shape[-1] != weight.shape[1]:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate linear input width does not match the weight matrix"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate linear bias width is malformed"
        )


def _native_linear(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    _validated_linear_inputs(inputs, weight, bias)
    return F.linear(inputs, weight, bias)


def _fixed_row_tile_4_linear(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Use one invariant dense row shape for scalar and batched projections.

    The final partial tile is zero padded and sliced.  This is a screening
    implementation only; a selected successor would need a versioned custom
    autograd implementation and full gradient proof before production use.
    """

    _validated_linear_inputs(inputs, weight, bias)
    leading_shape = inputs.shape[:-1]
    flattened = inputs.reshape(-1, inputs.shape[-1])
    projected_tiles: list[Tensor] = []
    for start in range(0, flattened.shape[0], _TILE_ROWS):
        active = flattened[start : start + _TILE_ROWS]
        if active.shape[0] < _TILE_ROWS:
            padded = active.new_zeros((_TILE_ROWS, active.shape[1]))
            padded[: active.shape[0]] = active
        else:
            padded = active
        projected_tiles.append(F.linear(padded, weight, bias)[: active.shape[0]])
    projected = torch.cat(projected_tiles, dim=0)
    return projected.reshape(*leading_shape, weight.shape[0])


@contextmanager
def recurrent_kernel_candidate(candidate: str) -> Iterator[None]:
    """Inject exactly one candidate into this process and restore it."""

    if candidate not in RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES:
        raise RecurrentKernelDevelopmentScreenError(
            f"unknown recurrent kernel candidate {candidate!r}"
        )
    original_linear = recurrent_actor_critic._backend_stable_linear
    original_gru_forward = recurrent_actor_critic.BackendStableGRU.forward
    try:
        if candidate == "per_row_bmm_manual_gru_baseline_v1":
            pass
        elif candidate == "native_linear_manual_gru_v1":
            recurrent_actor_critic._backend_stable_linear = _native_linear
        elif candidate == "native_linear_fused_gru_speed_ceiling_v1":
            recurrent_actor_critic._backend_stable_linear = _native_linear
            recurrent_actor_critic.BackendStableGRU.forward = nn.GRU.forward
        elif candidate == "fixed_row_tile_4_linear_manual_gru_v1":
            recurrent_actor_critic._backend_stable_linear = _fixed_row_tile_4_linear
        yield
    finally:
        recurrent_actor_critic._backend_stable_linear = original_linear
        recurrent_actor_critic.BackendStableGRU.forward = original_gru_forward


def _git_source_state(expected_source_sha: str) -> dict[str, object]:
    if (
        not isinstance(expected_source_sha, str)
        or len(expected_source_sha) != _SOURCE_SHA_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_source_sha)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "expected source SHA must be a lowercase 40-character commit"
        )
    head = subprocess.run(
        ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(_REPOSITORY_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    top_level = subprocess.run(
        [
            "git",
            "-C",
            str(_REPOSITORY_ROOT),
            "rev-parse",
            "--show-toplevel",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_source_sha:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen source SHA does not match the expected commit"
        )
    if status:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen requires a completely clean source tree"
        )
    if Path(top_level).resolve() != _REPOSITORY_ROOT:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen repository root binding drifted"
        )
    imported_modules = {
        "recurrent_actor_critic": recurrent_actor_critic,
        "recurrent_experiment": recurrent_experiment,
        "recurrent_rollout": recurrent_rollout,
        "phase_a_behavioral_evidence": (open_ecology_phase_a_behavioral_evidence),
    }
    imported_paths: dict[str, str] = {}
    for name, module in imported_modules.items():
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RecurrentKernelDevelopmentScreenError(
                f"imported module {name!r} has no source path"
            )
        resolved = Path(module_file).resolve()
        if not resolved.is_relative_to(_REPOSITORY_ROOT):
            raise RecurrentKernelDevelopmentScreenError(
                f"imported module {name!r} is outside the exact checkout"
            )
        imported_paths[name] = str(resolved.relative_to(_REPOSITORY_ROOT))
    return {
        "repository_root": str(_REPOSITORY_ROOT),
        "expected_source_sha": expected_source_sha,
        "observed_source_sha": head,
        "source_clean_including_untracked": True,
        "imported_module_paths": imported_paths,
    }


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _collector_once(
    *,
    mode: str,
    model: recurrent_actor_critic.PublicRecurrentActorCritic,
    task: Any,
) -> tuple[SimpleNamespace, dict[str, object], int]:
    if mode == "scalar":
        fixed_batch_contract = None
    elif mode == "batched":
        fixed_batch_contract = RecurrentFixedBatchRuntimeContract.open_ecology()
    else:
        raise RecurrentKernelDevelopmentScreenError(f"unknown collector mode {mode!r}")
    started = time.perf_counter_ns()
    steps, summary = _collect_recurrent_rollout_task(
        model,
        task,
        feed_forward_history_ablation=False,
        fixed_batch_contract=fixed_batch_contract,
    )
    elapsed_ns = time.perf_counter_ns() - started
    view = SimpleNamespace(steps=steps)
    facts = _d4_collector_path_facts(view)
    if not isinstance(summary.get("terminal_bootstrap"), Mapping):
        raise RecurrentKernelDevelopmentScreenError(
            "collector terminal bootstrap summary is malformed"
        )
    return view, facts, elapsed_ns


def _run_timing_pairs(
    *,
    model: recurrent_actor_critic.PublicRecurrentActorCritic,
    task: Any,
    warmup_pairs: int,
    timing_repeats: int,
) -> tuple[dict[str, object], dict[str, SimpleNamespace]]:
    for warmup in range(warmup_pairs):
        order = ("scalar", "batched") if warmup % 2 == 0 else ("batched", "scalar")
        for mode in order:
            _collector_once(mode=mode, model=model, task=task)

    samples: dict[str, list[dict[str, object]]] = {
        "scalar": [],
        "batched": [],
    }
    first_views: dict[str, SimpleNamespace] = {}
    timing_order: list[list[str]] = []
    paired_wins = 0
    for repeat in range(timing_repeats):
        order = ("scalar", "batched") if repeat % 2 == 0 else ("batched", "scalar")
        timing_order.append(list(order))
        pair_elapsed: dict[str, int] = {}
        for order_index, mode in enumerate(order):
            view, facts, elapsed_ns = _collector_once(
                mode=mode,
                model=model,
                task=task,
            )
            first_views.setdefault(mode, view)
            pair_elapsed[mode] = elapsed_ns
            samples[mode].append(
                {
                    "repeat_index": repeat,
                    "timing_order_index": order_index,
                    "mode": mode,
                    "elapsed_ns": elapsed_ns,
                    "semantic_sha256": facts["semantic_sha256"],
                    "collector_path": facts,
                }
            )
        paired_wins += int(pair_elapsed["batched"] < pair_elapsed["scalar"])

    scalar_elapsed = [int(sample["elapsed_ns"]) for sample in samples["scalar"]]
    batched_elapsed = [int(sample["elapsed_ns"]) for sample in samples["batched"]]
    scalar_median = int(statistics.median(scalar_elapsed))
    batched_median = int(statistics.median(batched_elapsed))
    return (
        {
            "warmup_pairs": warmup_pairs,
            "repeat_count": timing_repeats,
            "timing_order": timing_order,
            "scalar": {
                "elapsed_ns": scalar_elapsed,
                "median_elapsed_ns": scalar_median,
                "semantic_sha256": [
                    sample["semantic_sha256"] for sample in samples["scalar"]
                ],
                "samples": samples["scalar"],
            },
            "batched": {
                "elapsed_ns": batched_elapsed,
                "median_elapsed_ns": batched_median,
                "semantic_sha256": [
                    sample["semantic_sha256"] for sample in samples["batched"]
                ],
                "samples": samples["batched"],
            },
            "paired_batched_wins": paired_wins,
            "own_path_speedup": 1.0 - (batched_median / scalar_median),
        },
        first_views,
    )


def run_candidate_screen(
    *,
    candidate: str,
    expected_source_sha: str,
    child_nonce: str,
    timing_repeats: int = RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS,
    warmup_pairs: int = RECURRENT_KERNEL_DEVELOPMENT_SCREEN_WARMUP_PAIRS,
) -> dict[str, object]:
    """Run one candidate in the current, dedicated child process."""

    if timing_repeats < 1 or warmup_pairs < 0:
        raise RecurrentKernelDevelopmentScreenError(
            "screen timing and warmup counts are invalid"
        )
    if (
        not isinstance(child_nonce, str)
        or len(child_nonce) != 32
        or any(character not in "0123456789abcdef" for character in child_nonce)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "child nonce must be a lowercase 128-bit hex token"
        )
    source_state = _git_source_state(expected_source_sha)
    started = time.perf_counter_ns()
    with (
        recurrent_kernel_candidate(candidate),
        _isolated_torch_process_state(num_threads=1),
    ):
        equivalence = _run_d4_equivalence(
            device="cpu",
            shape=RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE,
        )
        model, tasks, proof_seed_contract = _d4_phase_a_collection_inputs(
            shape=RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE
        )
        timing, first_views = _run_timing_pairs(
            model=model,
            task=tasks[0],
            warmup_pairs=warmup_pairs,
            timing_repeats=timing_repeats,
        )
        collector_equivalence = _d4_collector_equivalence(
            first_views["scalar"],
            first_views["batched"],
        )
    final_source_state = _git_source_state(expected_source_sha)
    if final_source_state != source_state:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate source state changed during execution"
        )
    result: dict[str, object] = {
        "candidate": candidate,
        "status": "completed",
        "child_nonce": child_nonce,
        "development_only": True,
        "launch_authorized": False,
        "authority_evidence_eligible": False,
        "scientific_result": False,
        "selectable_for_implementation": _CANDIDATE_SELECTABLE[candidate],
        "candidate_implementation": _CANDIDATE_IMPLEMENTATION[candidate],
        "source_state": final_source_state,
        "shape": dict(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE),
        "model_contract_version": RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
        "source_numeric_kernel_version": RECURRENT_NUMERIC_KERNEL_VERSION,
        "proof_seed_contract": proof_seed_contract,
        "equivalence": equivalence,
        "collector_equivalence": collector_equivalence,
        "timing": timing,
        "peak_rss_bytes": _peak_rss_bytes(),
        "elapsed_ns": time.perf_counter_ns() - started,
        "runtime": {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_num_threads": 1,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }
    result["child_exact_digest"] = stable_payload_digest(result)
    return result


def _strict_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentKernelDevelopmentScreenError(f"{field} must be positive")
    return value


def _strict_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecurrentKernelDevelopmentScreenError(f"{field} is not SHA256")
    return value


def _finite_nonnegative(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} must be finite and nonnegative"
        )
    return float(value)


def _validate_candidate_identity(candidate: Mapping[str, object]) -> str:
    name = candidate.get("candidate")
    if not isinstance(name, str) or name not in _CANDIDATE_IMPLEMENTATION:
        raise RecurrentKernelDevelopmentScreenError("candidate identity is unknown")
    if (
        candidate.get("status") != "completed"
        or candidate.get("development_only") is not True
        or candidate.get("launch_authorized") is not False
        or candidate.get("authority_evidence_eligible") is not False
        or candidate.get("scientific_result") is not False
        or candidate.get("selectable_for_implementation")
        is not _CANDIDATE_SELECTABLE[name]
        or candidate.get("candidate_implementation") != _CANDIDATE_IMPLEMENTATION[name]
        or candidate.get("shape") != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE
        or candidate.get("model_contract_version")
        != RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION
        or candidate.get("source_numeric_kernel_version")
        != RECURRENT_NUMERIC_KERNEL_VERSION
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate lifecycle, implementation, or source contract drifted"
        )
    nonce = candidate.get("child_nonce")
    if (
        not isinstance(nonce, str)
        or len(nonce) != 32
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise RecurrentKernelDevelopmentScreenError("candidate child nonce drifted")
    digest = candidate.get("child_exact_digest")
    payload = dict(candidate)
    payload.pop("child_exact_digest", None)
    if _strict_sha256(digest, field="child_exact_digest") != stable_payload_digest(
        payload
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate child digest does not match its payload"
        )
    source = candidate.get("source_state")
    if (
        not isinstance(source, Mapping)
        or source.get("expected_source_sha") != source.get("observed_source_sha")
        or source.get("source_clean_including_untracked") is not True
        or not isinstance(source.get("imported_module_paths"), Mapping)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate exact-source binding is malformed"
        )
    runtime = candidate.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("pythonhashseed") != "0"
        or runtime.get("torch_num_threads") != 1
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate deterministic runtime binding drifted"
        )
    _strict_positive_int(candidate.get("peak_rss_bytes"), field="peak RSS")
    _strict_positive_int(candidate.get("elapsed_ns"), field="candidate elapsed time")
    return name


def _validate_equivalence(
    candidate: Mapping[str, object],
    *,
    enforce_numeric_headroom: bool,
) -> tuple[float, dict[str, object]]:
    equivalence = candidate.get("equivalence")
    if not isinstance(equivalence, Mapping):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate core equivalence is missing"
        )
    if (
        equivalence.get("proof_seed_contract") != candidate.get("proof_seed_contract")
        or equivalence.get("numeric_contract")
        != {"relative_tolerance": 1e-5, "absolute_tolerance": 1e-6}
        or equivalence.get("comparison_count") != _CORE_COMPARISON_COUNT
        or equivalence.get("reference_comparison_count") != _CORE_COMPARISON_COUNT
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate core comparison coverage drifted"
        )
    mismatch_fields = (
        "action_mismatch_count",
        "scalar_reference_action_mismatch_count",
        "batched_reference_action_mismatch_count",
    )
    if any(equivalence.get(field) != 0 for field in mismatch_fields):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate core action semantics differ"
        )
    ratio_fields = (
        "max_logit_tolerance_ratio",
        "max_value_tolerance_ratio",
        "max_hidden_tolerance_ratio",
        "max_scalar_reference_logit_tolerance_ratio",
        "max_scalar_reference_value_tolerance_ratio",
        "max_scalar_reference_hidden_tolerance_ratio",
        "max_batched_reference_logit_tolerance_ratio",
        "max_batched_reference_value_tolerance_ratio",
        "max_batched_reference_hidden_tolerance_ratio",
    )
    ratios = [
        _finite_nonnegative(equivalence.get(field), field=field)
        for field in ratio_fields
    ]
    for field in (
        "max_abs_logit_error",
        "max_abs_value_error",
        "max_abs_hidden_error",
    ):
        _finite_nonnegative(equivalence.get(field), field=field)
    semantic_digests = [
        _strict_sha256(equivalence.get(field), field=field)
        for field in (
            "scalar_semantic_sha256",
            "batched_semantic_sha256",
            "reference_semantic_sha256",
        )
    ]
    if len(set(semantic_digests)) != 1:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate free-recurrence semantics differ"
        )
    try:
        bucket = _validated_d4_bucket_matrix(equivalence.get("bucket_matrix"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RecurrentKernelDevelopmentScreenError(
            f"candidate bucket matrix failed exact validation: {exc}"
        ) from exc
    if (
        bucket["comparison_count"] != _BUCKET_COMPARISON_COUNT
        or len(bucket["cases"]) != 13
        or any(bucket[field] != 0 for field in mismatch_fields)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate bucket coverage or semantics drifted"
        )
    bucket_semantics = [
        _strict_sha256(bucket.get(field), field=f"bucket.{field}")
        for field in (
            "scalar_semantic_sha256",
            "batched_semantic_sha256",
            "reference_semantic_sha256",
        )
    ]
    if len(set(bucket_semantics)) != 1:
        raise RecurrentKernelDevelopmentScreenError("candidate bucket semantics differ")
    bucket_ratios = [
        _finite_nonnegative(value, field=f"bucket.{key}")
        for key, value in bucket.items()
        if key.endswith("tolerance_ratio")
    ]
    max_ratio = max((*ratios, *bucket_ratios))
    if (
        enforce_numeric_headroom
        and max_ratio > RECURRENT_KERNEL_DEVELOPMENT_SCREEN_NUMERIC_HEADROOM
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "numeric headroom exceeded: "
            f"{max_ratio:.9f} > "
            f"{RECURRENT_KERNEL_DEVELOPMENT_SCREEN_NUMERIC_HEADROOM:.2f}"
        )
    return max_ratio, bucket


def _validate_collector_and_timing(
    candidate: Mapping[str, object],
    *,
    enforce_numeric_headroom: bool,
) -> None:
    collector = candidate.get("collector_equivalence")
    timing = candidate.get("timing")
    if not isinstance(collector, Mapping) or not isinstance(timing, Mapping):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate collector or timing evidence is missing"
        )
    transition_count = _strict_positive_int(
        collector.get("transition_count"), field="collector transition count"
    )
    if (
        collector.get("batched_transition_count") != transition_count
        or collector.get("paired_transition_count") != transition_count
        or collector.get("numeric_transition_comparison_count") != transition_count
        or collector.get("hidden_component_comparison_count")
        != transition_count * _HIDDEN_SIZE
        or not isinstance(collector.get("bootstrap_value_comparison_count"), int)
        or int(collector["bootstrap_value_comparison_count"]) <= 0
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate collector comparison coverage drifted"
        )
    for field in (
        "semantic_mismatch_count",
        "identity_mismatch_count",
        "hidden_shape_mismatch_count",
        "bootstrap_none_mismatch_count",
    ):
        if collector.get(field) != 0:
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate collector {field} is nonzero"
            )
    for field in (
        "max_abs_input_hidden_error",
        "max_abs_logprob_error",
        "max_abs_entropy_error",
        "max_abs_value_error",
        "max_abs_bootstrap_value_error",
    ):
        _finite_nonnegative(collector.get(field), field=f"collector.{field}")
    collector_ratios = [
        _finite_nonnegative(collector.get(field), field=f"collector.{field}")
        for field in (
            "max_input_hidden_tolerance_ratio",
            "max_logprob_tolerance_ratio",
            "max_entropy_tolerance_ratio",
            "max_value_tolerance_ratio",
            "max_bootstrap_value_tolerance_ratio",
        )
    ]
    if (
        enforce_numeric_headroom
        and max(collector_ratios) > RECURRENT_KERNEL_DEVELOPMENT_SCREEN_NUMERIC_HEADROOM
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "collector numeric headroom exceeded"
        )
    _strict_sha256(
        collector.get("ordered_merge_semantic_sha256"),
        field="ordered collector semantic digest",
    )

    if (
        timing.get("warmup_pairs") != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_WARMUP_PAIRS
        or timing.get("repeat_count")
        != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS
    ):
        raise RecurrentKernelDevelopmentScreenError("candidate timing count drifted")
    expected_order = [
        ["scalar", "batched"] if repeat % 2 == 0 else ["batched", "scalar"]
        for repeat in range(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS)
    ]
    if timing.get("timing_order") != expected_order:
        raise RecurrentKernelDevelopmentScreenError("candidate timing order drifted")
    all_semantics: list[str] = []
    elapsed_by_mode: dict[str, list[int]] = {}
    validated_paths_by_mode: dict[str, list[dict[str, object]]] = {}
    for mode in ("scalar", "batched"):
        path = timing.get(mode)
        if not isinstance(path, Mapping):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} timing path is malformed"
            )
        elapsed = path.get("elapsed_ns")
        semantics = path.get("semantic_sha256")
        if (
            not isinstance(elapsed, list)
            or len(elapsed) != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in elapsed
            )
            or not isinstance(semantics, list)
            or len(semantics) != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS
        ):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} timing samples are malformed"
            )
        semantic_values = [
            _strict_sha256(value, field=f"{mode} semantic digest")
            for value in semantics
        ]
        if path.get("median_elapsed_ns") != int(statistics.median(elapsed)):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} timing median drifted"
            )
        try:
            validated_samples = _validated_d4_timed_samples(
                path.get("samples"),
                mode=mode,
                elapsed_ns=elapsed,
                semantic_sha256=semantic_values,
                timing_order=expected_order,
                expected_transition_count=transition_count,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} timed sample validation failed: {exc}"
            ) from exc
        if any(
            int(sample["collector_path"]["bootstrap_value_count"]) <= 0
            for sample in validated_samples
        ):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} bootstrap coverage is empty"
            )
        validated_paths = [
            dict(sample["collector_path"]) for sample in validated_samples
        ]
        if any(path != validated_paths[0] for path in validated_paths[1:]):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} collector path changed across samples"
            )
        if any(
            int(path["bootstrap_value_count"])
            != int(collector["bootstrap_value_comparison_count"])
            for path in validated_paths
        ):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} bootstrap comparison coverage drifted"
            )
        all_semantics.extend(semantic_values)
        elapsed_by_mode[mode] = elapsed
        validated_paths_by_mode[mode] = validated_paths
    if len(set(all_semantics)) != 1:
        raise RecurrentKernelDevelopmentScreenError(
            "collector semantic digests differ across timing samples"
        )
    common_semantic_digest = all_semantics[0]
    if collector.get("ordered_merge_semantic_sha256") != common_semantic_digest or any(
        path["semantic_sha256"] != common_semantic_digest
        for paths in validated_paths_by_mode.values()
        for path in paths
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "collector equivalence is not bound to the timed semantic digest"
        )
    paired_wins = sum(
        int(batched < scalar)
        for scalar, batched in zip(
            elapsed_by_mode["scalar"],
            elapsed_by_mode["batched"],
            strict=True,
        )
    )
    if timing.get("paired_batched_wins") != paired_wins:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate paired timing wins drifted"
        )
    scalar_median = int(statistics.median(elapsed_by_mode["scalar"]))
    batched_median = int(statistics.median(elapsed_by_mode["batched"]))
    expected_speedup = 1.0 - (batched_median / scalar_median)
    observed_speedup = timing.get("own_path_speedup")
    if (
        isinstance(observed_speedup, bool)
        or not isinstance(observed_speedup, (int, float))
        or not math.isfinite(float(observed_speedup))
        or not math.isclose(
            float(observed_speedup),
            expected_speedup,
            rel_tol=1e-15,
            abs_tol=0.0,
        )
    ):
        raise RecurrentKernelDevelopmentScreenError("candidate timing speedup drifted")


def _candidate_integrity_gate(
    candidate: Mapping[str, object],
    *,
    enforce_numeric_headroom: bool = True,
) -> dict[str, object]:
    try:
        name = _validate_candidate_identity(candidate)
        max_ratio, _bucket = _validate_equivalence(
            candidate,
            enforce_numeric_headroom=enforce_numeric_headroom,
        )
        _validate_collector_and_timing(
            candidate,
            enforce_numeric_headroom=enforce_numeric_headroom,
        )
    except (KeyError, TypeError, RecurrentKernelDevelopmentScreenError) as exc:
        return {
            "passed": False,
            "reasons": [str(exc)],
            "max_numeric_tolerance_ratio": None,
        }
    return {
        "passed": True,
        "reasons": [],
        "max_numeric_tolerance_ratio": max_ratio,
        "selectable_for_implementation": _CANDIDATE_SELECTABLE[name],
    }


def _candidate_semantic_fingerprint(
    candidate: Mapping[str, object],
) -> dict[str, object] | None:
    equivalence = candidate.get("equivalence")
    timing = candidate.get("timing")
    if not isinstance(equivalence, Mapping) or not isinstance(timing, Mapping):
        return None
    bucket_matrix = equivalence.get("bucket_matrix")
    scalar_timing = timing.get("scalar")
    if not isinstance(bucket_matrix, Mapping) or not isinstance(scalar_timing, Mapping):
        return None
    scalar_samples = scalar_timing.get("semantic_sha256")
    if (
        not isinstance(scalar_samples, Sequence)
        or isinstance(scalar_samples, (str, bytes, bytearray))
        or not scalar_samples
    ):
        return None
    return {
        "free_recurrence_scalar_semantic_sha256": equivalence.get(
            "scalar_semantic_sha256"
        ),
        "bucket_matrix_scalar_semantic_sha256": bucket_matrix.get(
            "scalar_semantic_sha256"
        ),
        "collector_scalar_semantic_sha256": scalar_samples[0],
    }


def select_development_candidate(
    candidate_results: Sequence[Mapping[str, object]],
    *,
    baseline_confirmation: Mapping[str, object] | None,
) -> dict[str, object]:
    """Apply preregistered integrity, absolute-speed, and RSS gates."""

    observed_names = [str(result.get("candidate")) for result in candidate_results]
    if observed_names != list(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES):
        return {
            "selected_candidate": None,
            "selection_authorized": False,
            "reason": "candidate order, identity, or coverage drifted",
            "candidate_gates": {},
        }
    by_name = {str(result.get("candidate")): result for result in candidate_results}
    baseline = by_name.get(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0])
    if not isinstance(baseline, Mapping) or not isinstance(
        baseline_confirmation, Mapping
    ):
        return {
            "selected_candidate": None,
            "selection_authorized": False,
            "reason": "bracketing baselines did not complete",
            "candidate_gates": {},
        }
    baseline_gate = _candidate_integrity_gate(
        baseline,
        enforce_numeric_headroom=False,
    )
    confirmation_gate = _candidate_integrity_gate(
        baseline_confirmation,
        enforce_numeric_headroom=False,
    )
    if (
        not baseline_gate["passed"]
        or not confirmation_gate["passed"]
        or baseline_confirmation.get("candidate")
        != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0]
    ):
        return {
            "selected_candidate": None,
            "selection_authorized": False,
            "reason": "bracketing baseline integrity failed",
            "baseline_gate": baseline_gate,
            "baseline_confirmation_gate": confirmation_gate,
            "candidate_gates": {},
        }
    baseline_timing = baseline.get("timing")
    confirmation_timing = baseline_confirmation.get("timing")
    if (
        not isinstance(baseline_timing, Mapping)
        or not isinstance(baseline_timing.get("scalar"), Mapping)
        or not isinstance(confirmation_timing, Mapping)
        or not isinstance(confirmation_timing.get("scalar"), Mapping)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "completed baseline timing is malformed"
        )
    baseline_scalar_samples = (
        int(baseline_timing["scalar"]["median_elapsed_ns"]),  # type: ignore[index]
        int(confirmation_timing["scalar"]["median_elapsed_ns"]),  # type: ignore[index]
    )
    baseline_scalar_ns = min(baseline_scalar_samples)
    baseline_rss_samples = (
        int(baseline["peak_rss_bytes"]),
        int(baseline_confirmation["peak_rss_bytes"]),
    )
    baseline_rss = min(baseline_rss_samples)
    baseline_semantics = _candidate_semantic_fingerprint(baseline)
    confirmation_semantics = _candidate_semantic_fingerprint(baseline_confirmation)
    if (
        baseline_semantics != confirmation_semantics
        or baseline_semantics is None
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in baseline_semantics.values()
        )
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "bracketing baseline semantic fingerprints differ"
        )
    gates: dict[str, object] = {}
    eligible: list[tuple[int, str]] = []
    for name in RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[1:]:
        result = by_name.get(name)
        if not isinstance(result, Mapping):
            gates[name] = {
                "passed": False,
                "reasons": ["candidate result is missing"],
            }
            continue
        integrity = _candidate_integrity_gate(result)
        reasons = list(integrity["reasons"])
        if _candidate_semantic_fingerprint(result) != baseline_semantics:
            reasons.append(
                "candidate behavior semantics differ from the frozen baseline"
            )
        timing = result.get("timing")
        if result.get("status") == "completed" and isinstance(timing, Mapping):
            scalar = timing.get("scalar")
            batched = timing.get("batched")
            if isinstance(scalar, Mapping) and isinstance(batched, Mapping):
                candidate_scalar_ns = int(scalar["median_elapsed_ns"])
                candidate_batched_ns = int(batched["median_elapsed_ns"])
                own_speedup = 1.0 - (candidate_batched_ns / candidate_scalar_ns)
                absolute_speedup = 1.0 - (candidate_batched_ns / baseline_scalar_ns)
                paired_wins = int(timing.get("paired_batched_wins", 0))
                rss_ratio = int(result["peak_rss_bytes"]) / baseline_rss
                if own_speedup < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP:
                    reasons.append("candidate batching speedup is below 3%")
                if (
                    absolute_speedup
                    < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP
                ):
                    reasons.append(
                        "candidate batched path does not beat the frozen "
                        "baseline scalar by 3%"
                    )
                if (
                    paired_wins
                    < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_PAIRED_WINS
                ):
                    reasons.append("candidate won fewer than 4/5 timing pairs")
                if rss_ratio > RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MAXIMUM_RSS_RATIO:
                    reasons.append("candidate peak RSS exceeds 1.10x baseline")
                gate = {
                    **integrity,
                    "passed": not reasons,
                    "reasons": reasons,
                    "own_path_speedup": own_speedup,
                    "absolute_speedup_over_baseline_scalar": absolute_speedup,
                    "paired_batched_wins": paired_wins,
                    "peak_rss_ratio": rss_ratio,
                    "selectable_for_implementation": _CANDIDATE_SELECTABLE[name],
                }
                if not _CANDIDATE_SELECTABLE[name]:
                    reasons.append(
                        "candidate lane is a nonselectable control or ceiling"
                    )
                    gate["passed"] = False
                    gate["reasons"] = reasons
                gates[name] = gate
                if not reasons and _CANDIDATE_SELECTABLE[name]:
                    eligible.append((candidate_batched_ns, name))
                continue
        gates[name] = {
            **integrity,
            "passed": False,
            "reasons": reasons or ["candidate timing is incomplete"],
        }
    selected = min(eligible)[1] if eligible else None
    return {
        "selected_candidate": selected,
        "selection_authorized": selected is not None,
        "reason": (
            "one candidate passed every preregistered development gate"
            if selected is not None
            else "no candidate passed every preregistered development gate"
        ),
        "baseline_scalar_median_elapsed_ns": baseline_scalar_ns,
        "bracketing_baseline_scalar_median_elapsed_ns": list(baseline_scalar_samples),
        "baseline_peak_rss_bytes": baseline_rss,
        "bracketing_baseline_peak_rss_bytes": list(baseline_rss_samples),
        "baseline_gate": baseline_gate,
        "baseline_confirmation_gate": confirmation_gate,
        "candidate_gates": gates,
        "fresh_exact_sha_d04_required_before_launch": True,
    }


def _child_command(
    *,
    candidate: str,
    expected_source_sha: str,
    child_nonce: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "evolution_sim.cli.recurrent_kernel_development_screen",
        "--development-run",
        "--expected-source-sha",
        expected_source_sha,
        "--child-candidate",
        candidate,
        "--child-nonce",
        child_nonce,
    ]


def _run_child_candidate(
    *,
    candidate: str,
    expected_source_sha: str,
) -> dict[str, object]:
    child_nonce = secrets.token_hex(16)
    try:
        completed = subprocess.run(
            _child_command(
                candidate=candidate,
                expected_source_sha=expected_source_sha,
                child_nonce=child_nonce,
            ),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "0"},
            timeout=_CHILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return {
            "candidate": candidate,
            "status": "failed",
            "failure": "child_timeout",
            "timeout_seconds": _CHILD_TIMEOUT_SECONDS,
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            "stderr_tail": stderr[-4000:],
        }
    if completed.returncode != 0:
        return {
            "candidate": candidate,
            "status": "failed",
            "failure": "child_nonzero_exit",
            "returncode": completed.returncode,
            "stderr_sha256": hashlib.sha256(
                completed.stderr.encode("utf-8")
            ).hexdigest(),
            "stderr_tail": completed.stderr[-4000:],
        }
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RecurrentKernelDevelopmentScreenError(
            f"candidate {candidate!r} emitted malformed child JSON"
        ) from exc
    if (
        not isinstance(parsed, dict)
        or parsed.get("candidate") != candidate
        or parsed.get("child_nonce") != child_nonce
        or not isinstance(parsed.get("source_state"), Mapping)
        or parsed["source_state"].get("observed_source_sha") != expected_source_sha
    ):
        raise RecurrentKernelDevelopmentScreenError(
            f"candidate {candidate!r} child identity or source binding drifted"
        )
    return parsed


def run_development_screen(
    *,
    expected_source_sha: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run the fixed candidate set in isolated children and select at most one."""

    source_state = _git_source_state(expected_source_sha)
    emit = progress if progress is not None else (lambda _message: None)
    results: list[dict[str, object]] = []
    for candidate in RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES:
        emit(f"recurrent_kernel_screen_start candidate={candidate}")
        result = _run_child_candidate(
            candidate=candidate,
            expected_source_sha=expected_source_sha,
        )
        results.append(result)
        emit(
            "recurrent_kernel_screen_finish "
            f"candidate={candidate} status={result['status']}"
        )
    baseline_name = RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0]
    emit(f"recurrent_kernel_screen_start candidate={baseline_name} confirmation=true")
    baseline_confirmation = _run_child_candidate(
        candidate=baseline_name,
        expected_source_sha=expected_source_sha,
    )
    emit(
        "recurrent_kernel_screen_finish "
        f"candidate={baseline_name} confirmation=true "
        f"status={baseline_confirmation['status']}"
    )
    final_source_state = _git_source_state(expected_source_sha)
    if final_source_state != source_state:
        raise RecurrentKernelDevelopmentScreenError(
            "parent source state changed during the development screen"
        )
    selection = select_development_candidate(
        results,
        baseline_confirmation=baseline_confirmation,
    )
    report: dict[str, object] = {
        "schema_version": RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SCHEMA_VERSION,
        "development_only": True,
        "launch_authorized": False,
        "authority_evidence_eligible": False,
        "scientific_result": False,
        "training_run": False,
        "training_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "source_state": final_source_state,
        "screen_contract": {
            "candidate_order": list(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES),
            "shape": dict(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE),
            "timing_repeats": (RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS),
            "warmup_pairs": RECURRENT_KERNEL_DEVELOPMENT_SCREEN_WARMUP_PAIRS,
            "numeric_headroom": (RECURRENT_KERNEL_DEVELOPMENT_SCREEN_NUMERIC_HEADROOM),
            "minimum_own_and_absolute_speedup": (
                RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP
            ),
            "minimum_paired_wins": (
                RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_PAIRED_WINS
            ),
            "maximum_peak_rss_ratio": (
                RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MAXIMUM_RSS_RATIO
            ),
            "does_not_write_d04_authority": True,
            "does_not_authorize_phase_a": True,
        },
        "candidate_results": results,
        "baseline_confirmation": baseline_confirmation,
        "selection": selection,
    }
    report["exact_digest"] = stable_payload_digest(report)
    return report


def validate_development_screen_report_path(path: Path) -> Path:
    if path.suffix != ".json":
        raise RecurrentKernelDevelopmentScreenError(
            "development screen report must use a .json path"
        )
    resolved = path.resolve()
    roots = [_LOCAL_DEVELOPMENT_SCREEN_ROOT.resolve()]
    configured_root = os.environ.get(_DEVELOPMENT_SCREEN_ROOT_ENV)
    if configured_root:
        roots.append(Path(configured_root).resolve())
    for root in roots:
        lowered_parts = tuple(part.lower() for part in root.parts)
        if (
            not root.name.endswith("development-screens")
            or any(
                marker in part
                for marker in ("authority", "qualification")
                for part in lowered_parts
            )
            or any(part == "runs" or part.endswith("-runs") for part in lowered_parts)
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "configured development screen root overlaps a forbidden "
                "authority, qualification, or run namespace"
            )
    if not any(resolved.is_relative_to(root) for root in roots):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen report is outside every approved "
            "development-screens root"
        )
    if resolved.exists():
        raise RecurrentKernelDevelopmentScreenError(
            "development screen report already exists; clobbering is forbidden"
        )
    return resolved


def write_development_screen_report(
    path: Path,
    report: Mapping[str, object],
) -> None:
    resolved = validate_development_screen_report_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(dict(report), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved.parent,
            prefix=f".{resolved.name}.pending-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, resolved)
        directory_fd = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen report already exists; clobbering is forbidden"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
