"""Bounded, non-authoritative screen for recurrent inference runtime lanes.

This module deliberately reuses the numerical and semantic comparators from
the sealed Phase-A D04 implementation without producing a D04 report or
writing into an authority namespace. Candidate kernel/projection combinations
are injected only inside one child process and are always restored before that
process exits.
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
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from evolution_sim.env.runtime import observations
from evolution_sim.mind import recurrent_actor_critic
from evolution_sim.mind import open_ecology_phase_a_behavioral_evidence
from evolution_sim.mind import policy_inputs
from evolution_sim.mind import recurrent_experiment
from evolution_sim.mind import recurrent_rollout
from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
    _d4_collector_equivalence,
    _d4_collector_path_facts,
    _d4_phase_a_collection_inputs,
    _d4_proof_seed_contract,
    _isolated_torch_process_state,
    _run_d4_equivalence,
    _validated_d4_bucket_matrix,
    _validated_d4_collector_path,
    _validated_d4_timed_samples,
)
from evolution_sim.mind.policy_inputs import (
    ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION,
    ecological_policy_values_from_decoded,
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
    "mind_v3_recurrent_adapter_development_screen_v3"
)
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES = (
    "per_row_bmm_legacy_adapter_baseline_v2",
    "per_row_bmm_fast_adapter_candidate_v2",
    "fixed_row_tile_4_fast_adapter_control_v2",
    "fixed_row_tile_4_legacy_adapter_control_v2",
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
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_DEVELOPMENT_SCREEN_ROOT = (
    _REPOSITORY_ROOT / "output" / "open-ecology" / "development-screens"
)
_DEVELOPMENT_SCREEN_ROOT_ENV = "EVOLUTION_SIM_OPEN_ECOLOGY_DEVELOPMENT_SCREEN_ROOT"
_CHILD_TIMEOUT_SECONDS = 30 * 60
_LEGACY_OBSERVATION_PROJECTION_VERSION = (
    "quantize_dequantize_then_diagnostic_filter_v1"
)
_LEGACY_TENSOR_BRIDGE_DISPATCH_VERSION = (
    "torch_tensor_device_dtype_layout_legacy_v0"
)
_LEGACY_INFERENCE_CONTEXT_VERSION = "torch_no_grad_v0"
_LEGACY_OUTPUT_MATERIALIZATION_VERSION = (
    "detach_cpu_explicit_float_item_tolist_v0"
)
_ADAPTER_RUNTIME_PROFILE_FIELDS = frozenset(
    {
        "implementation",
        "tensor_bridge_dispatch_version",
        "tensor_bridge_actual_branch",
        "tensor_bridge_probe_branches",
        "inference_context_version",
        "output_materialization_version",
        "observation_projection_version",
    }
)
_SOURCE_BOUND_MODULE_PATHS = MappingProxyType(
    {
        "observations": "python/evolution_sim/env/runtime/observations.py",
        "policy_inputs": "python/evolution_sim/mind/policy_inputs.py",
        "recurrent_actor_critic": (
            "python/evolution_sim/mind/recurrent_actor_critic.py"
        ),
        "recurrent_experiment": "python/evolution_sim/mind/recurrent_experiment.py",
        "recurrent_rollout": "python/evolution_sim/mind/recurrent_rollout.py",
        "phase_a_behavioral_evidence": (
            "python/evolution_sim/mind/"
            "open_ecology_phase_a_behavioral_evidence.py"
        ),
    }
)
_SOURCE_BOUND_MODULE_NAMES = frozenset(
    {
        "observations",
        "policy_inputs",
        "recurrent_actor_critic",
        "recurrent_experiment",
        "recurrent_rollout",
        "phase_a_behavioral_evidence",
    }
)
_CANDIDATE_SELECTABLE = {
    "per_row_bmm_legacy_adapter_baseline_v2": False,
    "per_row_bmm_fast_adapter_candidate_v2": True,
    "fixed_row_tile_4_fast_adapter_control_v2": False,
    "fixed_row_tile_4_legacy_adapter_control_v2": False,
}
_CANDIDATE_IMPLEMENTATION = {
    "per_row_bmm_legacy_adapter_baseline_v2": {
        "linear_forward": "per_row_bmm",
        "gru_forward": "manual_equations",
        "observation_projection": _LEGACY_OBSERVATION_PROJECTION_VERSION,
        "tensor_bridge_dispatch": _LEGACY_TENSOR_BRIDGE_DISPATCH_VERSION,
        "tensor_bridge_actual_branch": (
            "torch_tensor_device_dtype_layout_legacy_v0"
        ),
        "inference_context": _LEGACY_INFERENCE_CONTEXT_VERSION,
        "output_materialization": _LEGACY_OUTPUT_MATERIALIZATION_VERSION,
        "purpose": "bracketing_timing_and_semantic_baseline",
    },
    "per_row_bmm_fast_adapter_candidate_v2": {
        "linear_forward": "per_row_bmm",
        "gru_forward": "manual_equations",
        "observation_projection": ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION,
        "tensor_bridge_dispatch": (
            recurrent_rollout.RECURRENT_TENSOR_BRIDGE_DISPATCH_VERSION
        ),
        "tensor_bridge_actual_branch": (
            "numpy_from_numpy_cpu_float32_bool_native_c_v1"
        ),
        "inference_context": (
            recurrent_rollout.RECURRENT_INFERENCE_CONTEXT_VERSION
        ),
        "output_materialization": (
            recurrent_rollout.RECURRENT_OUTPUT_MATERIALIZATION_VERSION
        ),
        "purpose": "selectable_rollout_adapter_candidate",
    },
    "fixed_row_tile_4_fast_adapter_control_v2": {
        "linear_forward": "fixed_four_row_torch_functional_linear_tiles",
        "gru_forward": "manual_equations",
        "observation_projection": ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION,
        "tensor_bridge_dispatch": (
            recurrent_rollout.RECURRENT_TENSOR_BRIDGE_DISPATCH_VERSION
        ),
        "tensor_bridge_actual_branch": (
            "numpy_from_numpy_cpu_float32_bool_native_c_v1"
        ),
        "inference_context": (
            recurrent_rollout.RECURRENT_INFERENCE_CONTEXT_VERSION
        ),
        "output_materialization": (
            recurrent_rollout.RECURRENT_OUTPUT_MATERIALIZATION_VERSION
        ),
        "purpose": "nonselectable_kernel_speed_ceiling_control",
    },
    "fixed_row_tile_4_legacy_adapter_control_v2": {
        "linear_forward": "fixed_four_row_torch_functional_linear_tiles",
        "gru_forward": "manual_equations",
        "observation_projection": _LEGACY_OBSERVATION_PROJECTION_VERSION,
        "tensor_bridge_dispatch": _LEGACY_TENSOR_BRIDGE_DISPATCH_VERSION,
        "tensor_bridge_actual_branch": (
            "torch_tensor_device_dtype_layout_legacy_v0"
        ),
        "inference_context": _LEGACY_INFERENCE_CONTEXT_VERSION,
        "output_materialization": _LEGACY_OUTPUT_MATERIALIZATION_VERSION,
        "purpose": "nonselectable_kernel_and_legacy_adapter_control",
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


def _per_row_bmm_linear(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    _validated_linear_inputs(inputs, weight, bias)
    return recurrent_actor_critic._per_row_bmm_linear_forward(
        inputs,
        weight,
        bias,
    )


def _fixed_row_tile_4_linear(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Use the versioned production four-row forward inside the screen."""

    _validated_linear_inputs(inputs, weight, bias)
    return recurrent_actor_critic._fixed_row_tile_4_linear_forward(
        inputs,
        weight,
        bias,
    )


def _legacy_observation_projection(
    observation: dict[str, object],
) -> tuple[float, ...]:
    _, _, _, source_values = observations._validated_observation_input_values(
        observation
    )
    values = observations._dequantize_observation_input_values(
        observations._quantize_observation_input_values(source_values)
    )
    return ecological_policy_values_from_decoded(
        values,
        source_vector_size=len(values),
    )


def _legacy_inference_context(torch_module: Any) -> Any:
    return torch_module.no_grad()


def _legacy_materialized_float(value: object) -> float:
    return float(value)


def _legacy_materialized_float_tuple(
    values: Sequence[object],
) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _legacy_initial_hidden(
    core: recurrent_rollout.TorchRecurrentPolicyCore,
) -> tuple[float, ...]:
    state = core._model.initial_state(1)
    return tuple(
        float(value) for value in state.detach().cpu().reshape(-1).tolist()
    )


def _legacy_forward_step(
    core: recurrent_rollout.TorchRecurrentPolicyCore,
    observation: Sequence[float],
    current_action_mask: Sequence[bool],
    previous_feedback: recurrent_rollout.PreviousPublicFeedback,
    hidden: Sequence[float],
    *,
    genome_values: Sequence[float] | None = None,
) -> recurrent_rollout.RecurrentCoreOutput:
    torch_module = core._torch
    reference = next(core._model.parameters())
    observations = torch_module.tensor(
        tuple(observation),
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(1, 1, core.public_input_size)
    action_masks = torch_module.tensor(
        tuple(current_action_mask),
        device=reference.device,
        dtype=torch_module.bool,
    ).reshape(1, 1, len(recurrent_rollout.RECURRENT_ROLLOUT_ACTIONS))
    feedback = torch_module.tensor(
        previous_feedback.vector(),
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(1, 1, recurrent_rollout.RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE)
    state = torch_module.tensor(
        tuple(hidden),
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(core._layers, 1, core._layer_hidden_size)
    genome_tensor = (
        None
        if genome_values is None
        else torch_module.tensor(
            tuple(genome_values),
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(1, 1, recurrent_rollout.RECURRENT_CONTROLLER_GENOME_SIZE)
    )
    with torch_module.no_grad():
        output = core._model.forward_sequence(
            observations,
            action_masks,
            feedback,
            genome_values=genome_tensor,
            initial_state=state,
        )
    return recurrent_rollout.RecurrentCoreOutput(
        logits=tuple(
            float(value)
            for value in output.raw_logits[0, 0].detach().cpu().tolist()
        ),
        value=float(output.values[0, 0].detach().cpu().item()),
        next_hidden=tuple(
            float(value)
            for value in output.final_state.detach().cpu().reshape(-1).tolist()
        ),
    )


def _legacy_forward_fixed_batch(
    core: recurrent_rollout.TorchRecurrentPolicyCore,
    observations: Sequence[Sequence[float]],
    current_action_masks: Sequence[Sequence[bool]],
    previous_feedback: Sequence[recurrent_rollout.PreviousPublicFeedback],
    hidden: Sequence[Sequence[float]],
    *,
    batch_capacity: int,
    execution_batch_rows: int | None = None,
    genome_values: Sequence[Sequence[float]] | None = None,
) -> tuple[recurrent_rollout.RecurrentCoreOutput, ...]:
    active_rows = len(observations)
    selected_execution_rows = recurrent_rollout._fixed_batch_execution_rows(
        active_rows=active_rows,
        batch_capacity=batch_capacity,
    )
    if execution_batch_rows is None:
        execution_batch_rows = selected_execution_rows
    elif (
        isinstance(execution_batch_rows, bool)
        or not isinstance(execution_batch_rows, int)
        or execution_batch_rows != selected_execution_rows
    ):
        raise recurrent_rollout.RecurrentRolloutError(
            "fixed recurrent batch execution rows drifted from the frozen bucket "
            "contract"
        )
    if not (
        len(current_action_masks)
        == len(previous_feedback)
        == len(hidden)
        == active_rows
    ):
        raise recurrent_rollout.RecurrentRolloutError(
            "fixed recurrent batch input row counts disagree"
        )
    conditioned = (
        core.genome_conditioning_mode
        == recurrent_rollout.RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
    )
    if conditioned != (genome_values is not None):
        raise recurrent_rollout.RecurrentRolloutError(
            "fixed recurrent batch genome rows disagree with core conditioning"
        )
    if genome_values is not None and len(genome_values) != active_rows:
        raise recurrent_rollout.RecurrentRolloutError(
            "fixed recurrent batch genome row count disagrees"
        )

    torch_module = core._torch
    reference = next(core._model.parameters())
    padded_observations = torch_module.zeros(
        (1, execution_batch_rows, core.public_input_size),
        device=reference.device,
        dtype=reference.dtype,
    )
    padded_action_masks = torch_module.zeros(
        (
            1,
            execution_batch_rows,
            len(recurrent_rollout.RECURRENT_ROLLOUT_ACTIONS),
        ),
        device=reference.device,
        dtype=torch_module.bool,
    )
    padded_feedback = torch_module.zeros(
        (
            1,
            execution_batch_rows,
            recurrent_rollout.RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE,
        ),
        device=reference.device,
        dtype=reference.dtype,
    )
    padded_state = torch_module.zeros(
        (core._layers, execution_batch_rows, core._layer_hidden_size),
        device=reference.device,
        dtype=reference.dtype,
    )
    padded_action_masks[
        0,
        active_rows:,
        recurrent_rollout.RECURRENT_ROLLOUT_ACTIONS.index("stay"),
    ] = True
    padded_observations[0, :active_rows] = torch_module.tensor(
        tuple(tuple(row) for row in observations),
        device=reference.device,
        dtype=reference.dtype,
    )
    padded_action_masks[0, :active_rows] = torch_module.tensor(
        tuple(tuple(row) for row in current_action_masks),
        device=reference.device,
        dtype=torch_module.bool,
    )
    padded_feedback[0, :active_rows] = torch_module.tensor(
        tuple(feedback.vector() for feedback in previous_feedback),
        device=reference.device,
        dtype=reference.dtype,
    )
    hidden_tensor = torch_module.tensor(
        tuple(tuple(row) for row in hidden),
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(active_rows, core._layers, core._layer_hidden_size)
    padded_state[:, :active_rows] = hidden_tensor.permute(1, 0, 2)

    padded_genomes = None
    if genome_values is not None:
        padded_genomes = torch_module.zeros(
            (
                1,
                execution_batch_rows,
                recurrent_rollout.RECURRENT_CONTROLLER_GENOME_SIZE,
            ),
            device=reference.device,
            dtype=reference.dtype,
        )
        padded_genomes[0, :active_rows] = torch_module.tensor(
            tuple(tuple(row) for row in genome_values),
            device=reference.device,
            dtype=reference.dtype,
        )
    with torch_module.no_grad():
        output = core._model.forward_sequence(
            padded_observations,
            padded_action_masks,
            padded_feedback,
            genome_values=padded_genomes,
            initial_state=padded_state,
        )
    raw_logits = output.raw_logits[0, :active_rows].detach().cpu().tolist()
    values = output.values[0, :active_rows].detach().cpu().tolist()
    final_state = (
        output.final_state[:, :active_rows, :]
        .permute(1, 0, 2)
        .reshape(active_rows, -1)
        .detach()
        .cpu()
        .tolist()
    )
    return tuple(
        recurrent_rollout.RecurrentCoreOutput(
            logits=tuple(float(value) for value in raw_logits[row]),
            value=float(values[row]),
            next_hidden=tuple(float(value) for value in final_state[row]),
        )
        for row in range(active_rows)
    )


@contextmanager
def recurrent_kernel_candidate(candidate: str) -> Iterator[None]:
    """Inject exactly one combined kernel/projection lane and restore it."""

    if candidate not in RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES:
        raise RecurrentKernelDevelopmentScreenError(
            f"unknown recurrent kernel candidate {candidate!r}"
        )
    original_linear_forward = (
        recurrent_actor_critic._backend_stable_linear_forward
    )
    original_projection = recurrent_rollout.ecological_policy_values_from_observation
    original_projection_version = (
        recurrent_rollout.ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION
    )
    original_tensor_bridge_version = (
        recurrent_rollout.RECURRENT_TENSOR_BRIDGE_DISPATCH_VERSION
    )
    original_inference_context_version = (
        recurrent_rollout.RECURRENT_INFERENCE_CONTEXT_VERSION
    )
    original_output_materialization_version = (
        recurrent_rollout.RECURRENT_OUTPUT_MATERIALIZATION_VERSION
    )
    original_inference_context = recurrent_rollout._recurrent_inference_context
    original_materialized_float = recurrent_rollout._materialized_float
    original_materialized_float_tuple = (
        recurrent_rollout._materialized_float_tuple
    )
    original_initial_hidden = recurrent_rollout.TorchRecurrentPolicyCore.initial_hidden
    original_forward_step = recurrent_rollout.TorchRecurrentPolicyCore.forward_step
    original_forward_fixed_batch = (
        recurrent_rollout.TorchRecurrentPolicyCore.forward_fixed_batch
    )
    try:
        if candidate == "per_row_bmm_legacy_adapter_baseline_v2":
            recurrent_actor_critic._backend_stable_linear_forward = (
                recurrent_actor_critic._per_row_bmm_linear_forward
            )
            recurrent_rollout.ecological_policy_values_from_observation = (
                _legacy_observation_projection
            )
            recurrent_rollout.ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION = (
                _LEGACY_OBSERVATION_PROJECTION_VERSION
            )
            recurrent_rollout.RECURRENT_TENSOR_BRIDGE_DISPATCH_VERSION = (
                _LEGACY_TENSOR_BRIDGE_DISPATCH_VERSION
            )
            recurrent_rollout.RECURRENT_INFERENCE_CONTEXT_VERSION = (
                _LEGACY_INFERENCE_CONTEXT_VERSION
            )
            recurrent_rollout.RECURRENT_OUTPUT_MATERIALIZATION_VERSION = (
                _LEGACY_OUTPUT_MATERIALIZATION_VERSION
            )
            recurrent_rollout._recurrent_inference_context = (
                _legacy_inference_context
            )
            recurrent_rollout._materialized_float = _legacy_materialized_float
            recurrent_rollout._materialized_float_tuple = (
                _legacy_materialized_float_tuple
            )
            recurrent_rollout.TorchRecurrentPolicyCore.initial_hidden = (
                _legacy_initial_hidden
            )
            recurrent_rollout.TorchRecurrentPolicyCore.forward_step = (
                _legacy_forward_step
            )
            recurrent_rollout.TorchRecurrentPolicyCore.forward_fixed_batch = (
                _legacy_forward_fixed_batch
            )
        elif candidate == "per_row_bmm_fast_adapter_candidate_v2":
            pass
        elif candidate == "fixed_row_tile_4_fast_adapter_control_v2":
            recurrent_actor_critic._backend_stable_linear_forward = (
                recurrent_actor_critic._fixed_row_tile_4_linear_forward
            )
        elif candidate == "fixed_row_tile_4_legacy_adapter_control_v2":
            recurrent_actor_critic._backend_stable_linear_forward = (
                recurrent_actor_critic._fixed_row_tile_4_linear_forward
            )
            recurrent_rollout.ecological_policy_values_from_observation = (
                _legacy_observation_projection
            )
            recurrent_rollout.ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION = (
                _LEGACY_OBSERVATION_PROJECTION_VERSION
            )
            recurrent_rollout.RECURRENT_TENSOR_BRIDGE_DISPATCH_VERSION = (
                _LEGACY_TENSOR_BRIDGE_DISPATCH_VERSION
            )
            recurrent_rollout.RECURRENT_INFERENCE_CONTEXT_VERSION = (
                _LEGACY_INFERENCE_CONTEXT_VERSION
            )
            recurrent_rollout.RECURRENT_OUTPUT_MATERIALIZATION_VERSION = (
                _LEGACY_OUTPUT_MATERIALIZATION_VERSION
            )
            recurrent_rollout._recurrent_inference_context = (
                _legacy_inference_context
            )
            recurrent_rollout._materialized_float = _legacy_materialized_float
            recurrent_rollout._materialized_float_tuple = (
                _legacy_materialized_float_tuple
            )
            recurrent_rollout.TorchRecurrentPolicyCore.initial_hidden = (
                _legacy_initial_hidden
            )
            recurrent_rollout.TorchRecurrentPolicyCore.forward_step = (
                _legacy_forward_step
            )
            recurrent_rollout.TorchRecurrentPolicyCore.forward_fixed_batch = (
                _legacy_forward_fixed_batch
            )
        yield
    finally:
        recurrent_actor_critic._backend_stable_linear_forward = (
            original_linear_forward
        )
        recurrent_rollout.ecological_policy_values_from_observation = (
            original_projection
        )
        recurrent_rollout.ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION = (
            original_projection_version
        )
        recurrent_rollout.RECURRENT_TENSOR_BRIDGE_DISPATCH_VERSION = (
            original_tensor_bridge_version
        )
        recurrent_rollout.RECURRENT_INFERENCE_CONTEXT_VERSION = (
            original_inference_context_version
        )
        recurrent_rollout.RECURRENT_OUTPUT_MATERIALIZATION_VERSION = (
            original_output_materialization_version
        )
        recurrent_rollout._recurrent_inference_context = (
            original_inference_context
        )
        recurrent_rollout._materialized_float = original_materialized_float
        recurrent_rollout._materialized_float_tuple = (
            original_materialized_float_tuple
        )
        recurrent_rollout.TorchRecurrentPolicyCore.initial_hidden = (
            original_initial_hidden
        )
        recurrent_rollout.TorchRecurrentPolicyCore.forward_step = (
            original_forward_step
        )
        recurrent_rollout.TorchRecurrentPolicyCore.forward_fixed_batch = (
            original_forward_fixed_batch
        )


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
        "observations": observations,
        "policy_inputs": policy_inputs,
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
    if imported_paths != _SOURCE_BOUND_MODULE_PATHS:
        raise RecurrentKernelDevelopmentScreenError(
            "imported modules drifted from the frozen exact-source paths"
        )
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


def _configure_dedicated_child_torch_runtime() -> None:
    """Bind the development child to the Phase-A worker thread topology."""

    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RecurrentKernelDevelopmentScreenError(
                "dedicated child could not bind Torch inter-op threads to one"
            ) from exc
    if torch.get_num_interop_threads() != 1:
        raise RecurrentKernelDevelopmentScreenError(
            "dedicated child Torch inter-op thread binding drifted"
        )


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
    _configure_dedicated_child_torch_runtime()
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
        fixed_batch_runtime_binding = recurrent_rollout._fixed_batch_runtime_binding(
            RecurrentFixedBatchRuntimeContract.open_ecology(),
            core=recurrent_rollout.TorchRecurrentPolicyCore(model),
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
        runtime_digest = str(fixed_batch_runtime_binding["exact_digest"])
        for mode, expected_runtime_digests in (
            ("scalar", []),
            ("batched", [runtime_digest]),
        ):
            samples = timing[mode]["samples"]  # type: ignore[index]
            if any(
                sample["collector_path"]["fixed_batch_runtime_sha256"]
                != expected_runtime_digests
                for sample in samples
            ):
                raise RecurrentKernelDevelopmentScreenError(
                    f"{mode} timing samples are not bound to the observed runtime"
                )
        with recurrent_kernel_candidate(
            "per_row_bmm_legacy_adapter_baseline_v2"
        ):
            legacy_runtime_binding = recurrent_rollout._fixed_batch_runtime_binding(
                RecurrentFixedBatchRuntimeContract.open_ecology(),
                core=recurrent_rollout.TorchRecurrentPolicyCore(model),
            )
            legacy_scalar_view, legacy_scalar_facts, _ = _collector_once(
                mode="scalar",
                model=model,
                task=tasks[0],
            )
            legacy_batched_view, legacy_batched_facts, _ = _collector_once(
                mode="batched",
                model=model,
                task=tasks[0],
            )
            legacy_runtime_digest = str(legacy_runtime_binding["exact_digest"])
            if (
                legacy_scalar_facts["fixed_batch_runtime_sha256"] != []
                or legacy_batched_facts["fixed_batch_runtime_sha256"]
                != [legacy_runtime_digest]
            ):
                raise RecurrentKernelDevelopmentScreenError(
                    "legacy oracle collection is not bound to its runtime"
                )
        legacy_runtime_equivalence = {
            "scalar": _d4_collector_equivalence(
                legacy_scalar_view,
                first_views["scalar"],
            ),
            "batched": _d4_collector_equivalence(
                legacy_batched_view,
                first_views["batched"],
            ),
        }
        observed_thread_runtime = {
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        }
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
        "fixed_batch_runtime_binding": fixed_batch_runtime_binding,
        "legacy_fixed_batch_runtime_binding": legacy_runtime_binding,
        "legacy_runtime_paths": {
            "scalar": legacy_scalar_facts,
            "batched": legacy_batched_facts,
        },
        "legacy_runtime_equivalence": legacy_runtime_equivalence,
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
            **observed_thread_runtime,
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


def _validated_adapter_runtime_binding(
    value: object,
    *,
    implementation: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentKernelDevelopmentScreenError(f"{field} is missing")
    try:
        validated = recurrent_rollout._validated_fixed_batch_runtime_binding(
            value
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} is malformed: {exc}"
        ) from exc
    observed = validated["observed_runtime"]
    expected_actual_branch = implementation["tensor_bridge_actual_branch"]
    if (
        observed.get("implementation")
        != "PublicRecurrentActorCritic.forward_sequence_time1_bounded_bucket_batch_v3"
        or observed.get("tensor_bridge_dispatch_version")
        != implementation["tensor_bridge_dispatch"]
        or observed.get("tensor_bridge_actual_branch") != expected_actual_branch
        or observed.get("tensor_bridge_probe_branches")
        != {"float32": expected_actual_branch, "bool": expected_actual_branch}
        or observed.get("inference_context_version")
        != implementation["inference_context"]
        or observed.get("output_materialization_version")
        != implementation["output_materialization"]
        or observed.get("observation_projection_version")
        != implementation["observation_projection"]
        or observed.get("native_byte_order") not in {"little", "big"}
        or observed.get("device_type") != "cpu"
        or observed.get("device_index") is not None
        or observed.get("dtype") != "torch.float32"
        or observed.get("torch_num_threads") != 1
        or observed.get("torch_num_interop_threads") != 1
        or (
            expected_actual_branch
            == "numpy_from_numpy_cpu_float32_bool_native_c_v1"
            and not isinstance(observed.get("numpy_version"), str)
        )
    ):
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} adapter provenance drifted"
        )
    return validated


def _normalized_common_runtime_binding(
    value: object,
    *,
    field: str,
) -> dict[str, object]:
    """Remove only preregistered adapter identity from a validated binding."""

    if not isinstance(value, Mapping):
        raise RecurrentKernelDevelopmentScreenError(f"{field} is missing")
    try:
        validated = recurrent_rollout._validated_fixed_batch_runtime_binding(
            value
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} is malformed: {exc}"
        ) from exc
    observed = validated.get("observed_runtime")
    if (
        not isinstance(observed, Mapping)
        or not _ADAPTER_RUNTIME_PROFILE_FIELDS.issubset(observed)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} adapter profile fields are incomplete"
        )
    normalized = {
        key: value
        for key, value in validated.items()
        if key != "exact_digest"
    }
    normalized["observed_runtime"] = {
        key: value
        for key, value in observed.items()
        if key not in _ADAPTER_RUNTIME_PROFILE_FIELDS
    }
    return normalized


def _validated_source_state(value: object, *, field: str) -> dict[str, object]:
    expected_keys = {
        "repository_root",
        "expected_source_sha",
        "observed_source_sha",
        "source_clean_including_untracked",
        "imported_module_paths",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} source schema drifted"
        )
    expected_sha = value.get("expected_source_sha")
    observed_sha = value.get("observed_source_sha")
    imported_paths = value.get("imported_module_paths")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != _SOURCE_SHA_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or observed_sha != expected_sha
        or value.get("source_clean_including_untracked") is not True
        or not isinstance(value.get("repository_root"), str)
        or Path(str(value["repository_root"])).resolve() != _REPOSITORY_ROOT
        or not isinstance(imported_paths, Mapping)
        or imported_paths != _SOURCE_BOUND_MODULE_PATHS
    ):
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} exact-source binding is malformed"
        )
    return dict(value)


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
    _validated_source_state(
        candidate.get("source_state"),
        field="candidate",
    )
    candidate_runtime_binding = _validated_adapter_runtime_binding(
        candidate.get("fixed_batch_runtime_binding"),
        implementation=_CANDIDATE_IMPLEMENTATION[name],
        field="candidate fixed-batch runtime binding",
    )
    legacy_runtime_binding = _validated_adapter_runtime_binding(
        candidate.get("legacy_fixed_batch_runtime_binding"),
        implementation=_CANDIDATE_IMPLEMENTATION[
            "per_row_bmm_legacy_adapter_baseline_v2"
        ],
        field="legacy oracle fixed-batch runtime binding",
    )
    runtime = candidate.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("pythonhashseed") != "0"
        or runtime.get("torch_num_threads") != 1
        or runtime.get("torch_num_interop_threads") != 1
        or candidate_runtime_binding["observed_runtime"].get("torch_version")
        != runtime.get("torch_version")
        or legacy_runtime_binding["observed_runtime"].get("torch_version")
        != runtime.get("torch_version")
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
    expected_proof_seed_contract = _d4_proof_seed_contract(
        shape=RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE
    )
    if (
        candidate.get("proof_seed_contract") != expected_proof_seed_contract
        or equivalence.get("proof_seed_contract") != expected_proof_seed_contract
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
    if max_ratio > 1.0:
        raise RecurrentKernelDevelopmentScreenError(
            "declared D04 numeric tolerance exceeded: "
            f"{max_ratio:.9f} > 1.0"
        )
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
) -> float:
    collector = candidate.get("collector_equivalence")
    timing = candidate.get("timing")
    if not isinstance(collector, Mapping) or not isinstance(timing, Mapping):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate collector or timing evidence is missing"
        )
    runtime_binding = candidate.get("fixed_batch_runtime_binding")
    if not isinstance(runtime_binding, Mapping):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate runtime binding is missing from timing evidence"
        )
    runtime_digest = _strict_sha256(
        runtime_binding.get("exact_digest"),
        field="fixed-batch runtime digest",
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
    if max(collector_ratios) > 1.0:
        raise RecurrentKernelDevelopmentScreenError(
            "collector declared D04 numeric tolerance exceeded"
        )
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
        expected_runtime_digests = [] if mode == "scalar" else [runtime_digest]
        if any(
            path["fixed_batch_runtime_sha256"] != expected_runtime_digests
            for path in validated_paths
        ):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} runtime binding differs across timed evidence"
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
    legacy_binding = candidate.get("legacy_fixed_batch_runtime_binding")
    legacy_paths = candidate.get("legacy_runtime_paths")
    if not isinstance(legacy_binding, Mapping) or not isinstance(
        legacy_paths,
        Mapping,
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate legacy runtime path binding is missing"
        )
    legacy_runtime_digest = _strict_sha256(
        legacy_binding.get("exact_digest"),
        field="legacy fixed-batch runtime digest",
    )
    if set(legacy_paths) != {"scalar", "batched"}:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate legacy runtime path coverage drifted"
        )
    for mode in ("scalar", "batched"):
        try:
            legacy_path = _validated_d4_collector_path(
                legacy_paths.get(mode),
                mode=mode,
                expected_semantic_sha256=all_semantics[0],
                expected_transition_count=transition_count,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} legacy runtime path is malformed: {exc}"
            ) from exc
        expected_runtime_digests = (
            [] if mode == "scalar" else [legacy_runtime_digest]
        )
        if (
            legacy_path["fixed_batch_runtime_sha256"]
            != expected_runtime_digests
            or int(legacy_path["bootstrap_value_count"]) <= 0
        ):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} legacy runtime path is not evidence-bound"
            )
    legacy_equivalence = candidate.get("legacy_runtime_equivalence")
    if not isinstance(legacy_equivalence, Mapping) or set(
        legacy_equivalence
    ) != {"scalar", "batched"}:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate legacy-runtime equivalence is missing"
        )
    legacy_ratios: list[float] = []
    for mode in ("scalar", "batched"):
        comparison = legacy_equivalence.get(mode)
        if not isinstance(comparison, Mapping):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} legacy-runtime comparison is malformed"
            )
        if (
            comparison.get("transition_count") != transition_count
            or comparison.get("batched_transition_count") != transition_count
            or comparison.get("paired_transition_count") != transition_count
            or comparison.get("numeric_transition_comparison_count")
            != transition_count
            or comparison.get("hidden_component_comparison_count")
            != transition_count * _HIDDEN_SIZE
            or not isinstance(
                comparison.get("bootstrap_value_comparison_count"),
                int,
            )
            or int(comparison["bootstrap_value_comparison_count"]) <= 0
            or any(
                comparison.get(field) != 0
                for field in (
                    "semantic_mismatch_count",
                    "identity_mismatch_count",
                    "hidden_shape_mismatch_count",
                    "bootstrap_none_mismatch_count",
                )
            )
        ):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} differs structurally from the legacy runtime"
            )
        for field in (
            "max_abs_input_hidden_error",
            "max_abs_logprob_error",
            "max_abs_entropy_error",
            "max_abs_value_error",
            "max_abs_bootstrap_value_error",
        ):
            _finite_nonnegative(
                comparison.get(field),
                field=f"legacy_runtime.{mode}.{field}",
            )
        mode_ratios = [
            _finite_nonnegative(
                comparison.get(field),
                field=f"legacy_runtime.{mode}.{field}",
            )
            for field in (
                "max_input_hidden_tolerance_ratio",
                "max_logprob_tolerance_ratio",
                "max_entropy_tolerance_ratio",
                "max_value_tolerance_ratio",
                "max_bootstrap_value_tolerance_ratio",
            )
        ]
        if max(mode_ratios) > 1.0:
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} exceeds legacy-runtime numeric tolerance"
            )
        candidate_semantics = timing[mode]["semantic_sha256"]  # type: ignore[index]
        if (
            comparison.get("ordered_merge_semantic_sha256")
            != candidate_semantics[0]
        ):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} legacy comparison is not bound to timed semantics"
            )
        legacy_ratios.extend(mode_ratios)
    max_collector_ratio = max((*collector_ratios, *legacy_ratios))
    if (
        enforce_numeric_headroom
        and max_collector_ratio
        > RECURRENT_KERNEL_DEVELOPMENT_SCREEN_NUMERIC_HEADROOM
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "legacy-runtime numeric headroom exceeded"
        )
    return max_collector_ratio


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
        max_collector_ratio = _validate_collector_and_timing(
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
        "max_numeric_tolerance_ratio": max(max_ratio, max_collector_ratio),
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
    all_results = [*candidate_results, baseline_confirmation]
    nonces = [result.get("child_nonce") for result in all_results]
    completed_results = [
        result for result in all_results if result.get("status") == "completed"
    ]
    source_states = [result.get("source_state") for result in completed_results]
    runtimes = [result.get("runtime") for result in completed_results]
    try:
        normalized_runtime_bindings = [
            _normalized_common_runtime_binding(
                result.get("fixed_batch_runtime_binding"),
                field="completed child fixed-batch runtime binding",
            )
            for result in completed_results
        ]
        legacy_runtime_digests = [
            _strict_sha256(
                recurrent_rollout._validated_fixed_batch_runtime_binding(
                    result["legacy_fixed_batch_runtime_binding"]  # type: ignore[arg-type]
                ).get("exact_digest"),
                field="completed child legacy fixed-batch runtime digest",
            )
            for result in completed_results
        ]
    except (KeyError, TypeError, ValueError, RecurrentKernelDevelopmentScreenError):
        normalized_runtime_bindings = []
        legacy_runtime_digests = []
    if (
        any(not isinstance(nonce, str) for nonce in nonces)
        or len(set(nonces)) != len(nonces)
        or not source_states
        or any(source != source_states[0] for source in source_states[1:])
        or any(runtime != runtimes[0] for runtime in runtimes[1:])
        or not normalized_runtime_bindings
        or any(
            binding != normalized_runtime_bindings[0]
            for binding in normalized_runtime_bindings[1:]
        )
        or not legacy_runtime_digests
        or len(set(legacy_runtime_digests)) != 1
    ):
        return {
            "selected_candidate": None,
            "selection_authorized": False,
            "reason": (
                "child nonce, exact-source, or normalized runtime isolation drifted"
            ),
            "baseline_gate": baseline_gate,
            "baseline_confirmation_gate": confirmation_gate,
            "candidate_gates": {},
        }
    baseline_timing = baseline.get("timing")
    confirmation_timing = baseline_confirmation.get("timing")
    if (
        not isinstance(baseline_timing, Mapping)
        or not isinstance(baseline_timing.get("scalar"), Mapping)
        or not isinstance(baseline_timing.get("batched"), Mapping)
        or not isinstance(confirmation_timing, Mapping)
        or not isinstance(confirmation_timing.get("scalar"), Mapping)
        or not isinstance(confirmation_timing.get("batched"), Mapping)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "completed baseline timing is malformed"
        )
    baseline_scalar_samples = (
        int(baseline_timing["scalar"]["median_elapsed_ns"]),  # type: ignore[index]
        int(confirmation_timing["scalar"]["median_elapsed_ns"]),  # type: ignore[index]
    )
    baseline_scalar_ns = min(baseline_scalar_samples)
    baseline_batched_samples = (
        int(baseline_timing["batched"]["median_elapsed_ns"]),  # type: ignore[index]
        int(confirmation_timing["batched"]["median_elapsed_ns"]),  # type: ignore[index]
    )
    baseline_batched_ns = min(baseline_batched_samples)
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
    control_names = (
        "fixed_row_tile_4_fast_adapter_control_v2",
        "fixed_row_tile_4_legacy_adapter_control_v2",
    )
    control_batched_ns: dict[str, int] = {}
    controls_valid = True
    for name in control_names:
        result = by_name.get(name)
        if not isinstance(result, Mapping):
            gates[name] = {
                "evidence_valid": False,
                "selection_eligible": False,
                "reasons": ["candidate result is missing"],
            }
            controls_valid = False
            continue
        integrity = _candidate_integrity_gate(
            result,
            enforce_numeric_headroom=False,
        )
        reasons = list(integrity["reasons"])
        if _candidate_semantic_fingerprint(result) != baseline_semantics:
            reasons.append(
                "candidate behavior semantics differ from the frozen baseline"
            )
        timing = result.get("timing")
        if (
            result.get("status") != "completed"
            or not isinstance(timing, Mapping)
            or not isinstance(timing.get("scalar"), Mapping)
            or not isinstance(timing.get("batched"), Mapping)
        ):
            reasons.append("control timing is incomplete")
        if reasons:
            controls_valid = False
            gates[name] = {
                **integrity,
                "evidence_valid": False,
                "selection_eligible": False,
                "reasons": reasons,
            }
            continue
        scalar_ns = int(timing["scalar"]["median_elapsed_ns"])  # type: ignore[index]
        batched_ns = int(timing["batched"]["median_elapsed_ns"])  # type: ignore[index]
        control_batched_ns[name] = batched_ns
        gates[name] = {
            **integrity,
            "evidence_valid": True,
            "selection_eligible": False,
            "reasons": [],
            "own_path_speedup": 1.0 - (batched_ns / scalar_ns),
            "batched_median_elapsed_ns": batched_ns,
            "nonselectable_kernel_control": True,
        }

    selectable_name = "per_row_bmm_fast_adapter_candidate_v2"
    selectable = by_name.get(selectable_name)
    candidate_reasons: list[str] = []
    if not isinstance(selectable, Mapping):
        candidate_integrity: dict[str, object] = {
            "passed": False,
            "reasons": ["selectable candidate result is missing"],
        }
        candidate_reasons.extend(candidate_integrity["reasons"])  # type: ignore[arg-type]
    else:
        candidate_integrity = _candidate_integrity_gate(selectable)
        candidate_reasons.extend(candidate_integrity["reasons"])  # type: ignore[arg-type]
        if _candidate_semantic_fingerprint(selectable) != baseline_semantics:
            candidate_reasons.append(
                "candidate behavior semantics differ from the frozen baseline"
            )
    candidate_timing = (
        selectable.get("timing") if isinstance(selectable, Mapping) else None
    )
    candidate_batched_ns: int | None = None
    candidate_metrics: dict[str, object] = {}
    if (
        not isinstance(candidate_timing, Mapping)
        or not isinstance(candidate_timing.get("scalar"), Mapping)
        or not isinstance(candidate_timing.get("batched"), Mapping)
        or not isinstance(selectable, Mapping)
        or selectable.get("status") != "completed"
    ):
        candidate_reasons.append("selectable candidate timing is incomplete")
    else:
        candidate_scalar_ns = int(
            candidate_timing["scalar"]["median_elapsed_ns"]  # type: ignore[index]
        )
        candidate_batched_ns = int(
            candidate_timing["batched"]["median_elapsed_ns"]  # type: ignore[index]
        )
        own_speedup = 1.0 - (candidate_batched_ns / candidate_scalar_ns)
        scalar_baseline_speedup = 1.0 - (
            candidate_batched_ns / baseline_scalar_ns
        )
        batched_baseline_speedup = 1.0 - (
            candidate_batched_ns / baseline_batched_ns
        )
        paired_wins = int(candidate_timing.get("paired_batched_wins", 0))
        rss_ratio = int(selectable["peak_rss_bytes"]) / baseline_rss
        if own_speedup < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP:
            candidate_reasons.append("candidate batching speedup is below 3%")
        if (
            scalar_baseline_speedup
            < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP
        ):
            candidate_reasons.append(
                "candidate batched path does not beat legacy scalar by 3%"
            )
        if (
            batched_baseline_speedup
            < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP
        ):
            candidate_reasons.append(
                "candidate batched path does not beat legacy batched by 3%"
            )
        if paired_wins < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_PAIRED_WINS:
            candidate_reasons.append("candidate won fewer than 4/5 timing pairs")
        if rss_ratio > RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MAXIMUM_RSS_RATIO:
            candidate_reasons.append("candidate peak RSS exceeds 1.10x baseline")
        candidate_metrics = {
            "own_path_speedup": own_speedup,
            "absolute_speedup_over_legacy_scalar": scalar_baseline_speedup,
            "absolute_speedup_over_legacy_batched": batched_baseline_speedup,
            "paired_batched_wins": paired_wins,
            "peak_rss_ratio": rss_ratio,
            "batched_median_elapsed_ns": candidate_batched_ns,
            "nonselectable_control_batched_median_elapsed_ns": dict(
                control_batched_ns
            ),
        }
    candidate_evidence_valid = bool(
        isinstance(selectable, Mapping)
        and candidate_integrity.get("passed")
        and _candidate_semantic_fingerprint(selectable) == baseline_semantics
    )
    gates[selectable_name] = {
        **candidate_integrity,
        **candidate_metrics,
        "evidence_valid": candidate_evidence_valid,
        "selection_eligible": not candidate_reasons,
        "reasons": candidate_reasons,
    }
    selected = selectable_name if not candidate_reasons else None
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
        "baseline_batched_median_elapsed_ns": baseline_batched_ns,
        "bracketing_baseline_batched_median_elapsed_ns": list(
            baseline_batched_samples
        ),
        "baseline_peak_rss_bytes": baseline_rss,
        "bracketing_baseline_peak_rss_bytes": list(baseline_rss_samples),
        "baseline_gate": baseline_gate,
        "baseline_confirmation_gate": confirmation_gate,
        "candidate_gates": gates,
        "all_kernel_controls_evidence_valid": controls_valid,
        "fresh_exact_sha_d04_required_before_launch": True,
        "fresh_source_bound_authority_amendment_required_before_d04": True,
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
            "child_nonce": child_nonce,
            "failure": "child_timeout",
            "timeout_seconds": _CHILD_TIMEOUT_SECONDS,
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            "stderr_tail": stderr[-4000:],
        }
    if completed.returncode != 0:
        return {
            "candidate": candidate,
            "status": "failed",
            "child_nonce": child_nonce,
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


def _development_screen_contract() -> dict[str, object]:
    return {
        "candidate_order": list(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES),
        "selectable_rollout_adapter_candidate": (
            "per_row_bmm_fast_adapter_candidate_v2"
        ),
        "nonselectable_kernel_controls": [
            "fixed_row_tile_4_fast_adapter_control_v2",
            "fixed_row_tile_4_legacy_adapter_control_v2",
        ],
        "source_default_numeric_kernel_version": (
            RECURRENT_NUMERIC_KERNEL_VERSION
        ),
        "source_default_observation_projection_version": (
            ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION
        ),
        "source_bound_module_paths": dict(_SOURCE_BOUND_MODULE_PATHS),
        "proof_seed_contract": _d4_proof_seed_contract(
            shape=RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE
        ),
        "fixed_batch_runtime_binding_required": True,
        "execution_device_type": "cpu",
        "execution_device_index": None,
        "execution_dtype": "torch.float32",
        "torch_thread_topology": {"intra_op": 1, "inter_op": 1},
        "common_runtime_comparison_excludes_only": sorted(
            _ADAPTER_RUNTIME_PROFILE_FIELDS
        ),
        "identical_legacy_runtime_binding_digest_required": True,
        "shape": dict(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE),
        "timing_repeats": RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS,
        "warmup_pairs": RECURRENT_KERNEL_DEVELOPMENT_SCREEN_WARMUP_PAIRS,
        "numeric_headroom": RECURRENT_KERNEL_DEVELOPMENT_SCREEN_NUMERIC_HEADROOM,
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
        "fresh_source_bound_authority_amendment_required_before_d04": True,
    }


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
        "screen_contract": _development_screen_contract(),
        "candidate_results": results,
        "baseline_confirmation": baseline_confirmation,
        "selection": selection,
    }
    report["exact_digest"] = stable_payload_digest(report)
    return report


def validate_development_screen_report(
    report: Mapping[str, object],
) -> None:
    """Reconstruct the non-authoritative report before atomic publication."""

    expected_keys = {
        "schema_version",
        "development_only",
        "launch_authorized",
        "authority_evidence_eligible",
        "scientific_result",
        "training_run",
        "training_artifact_created",
        "runtime_action_selection_changed",
        "promotion_authorized",
        "source_state",
        "screen_contract",
        "candidate_results",
        "baseline_confirmation",
        "selection",
        "exact_digest",
    }
    if set(report) != expected_keys:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen report root schema drifted"
        )
    if (
        report.get("schema_version")
        != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SCHEMA_VERSION
        or report.get("development_only") is not True
        or report.get("launch_authorized") is not False
        or report.get("authority_evidence_eligible") is not False
        or report.get("scientific_result") is not False
        or report.get("training_run") is not False
        or report.get("training_artifact_created") is not False
        or report.get("runtime_action_selection_changed") is not False
        or report.get("promotion_authorized") is not False
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen lifecycle flags drifted"
        )
    source = _validated_source_state(
        report.get("source_state"),
        field="development screen",
    )
    contract = report.get("screen_contract")
    if contract != _development_screen_contract():
        raise RecurrentKernelDevelopmentScreenError(
            "development screen declared contract drifted"
        )
    results = report.get("candidate_results")
    confirmation = report.get("baseline_confirmation")
    if (
        not isinstance(results, Sequence)
        or isinstance(results, (str, bytes, bytearray))
        or len(results) != len(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES)
        or any(not isinstance(result, Mapping) for result in results)
        or not isinstance(confirmation, Mapping)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen child evidence is malformed"
        )
    typed_results = [result for result in results if isinstance(result, Mapping)]
    completed_children = [
        result
        for result in (*typed_results, confirmation)
        if result.get("status") == "completed"
    ]
    if any(result.get("source_state") != source for result in completed_children):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen parent and child source bindings differ"
        )
    recomputed_selection = select_development_candidate(
        typed_results,
        baseline_confirmation=confirmation,
    )
    if report.get("selection") != recomputed_selection:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen selection does not reconstruct"
        )
    digest = report.get("exact_digest")
    payload = dict(report)
    payload.pop("exact_digest", None)
    if _strict_sha256(digest, field="report exact digest") != stable_payload_digest(
        payload
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen report digest does not reconstruct"
        )


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
    validate_development_screen_report(report)
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
