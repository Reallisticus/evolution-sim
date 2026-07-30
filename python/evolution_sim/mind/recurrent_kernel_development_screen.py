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
import selectors
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
from evolution_sim.mind import recurrent_policy
from evolution_sim.mind import recurrent_ppo
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
from evolution_sim.mind.recurrent_accuracy_screen import (
    RecurrentKernelDevelopmentScreenError,
    process_start_identity as _process_start_identity,
)
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
    "mind_v3_recurrent_adapter_accuracy_development_screen_v4"
)
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES = (
    "per_row_bmm_legacy_adapter_baseline_v2",
    "per_row_bmm_fast_adapter_control_v3",
    "fixed_row_tile_4_legacy_adapter_control_v3",
    "fixed_row_tile_4_fast_adapter_candidate_v3",
)
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE = {
    "worlds": 1,
    "rollout_ticks": 128,
    "initial_agents": 64,
}
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS = 5
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_WARMUP_PAIRS = 2
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP = 0.03
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_PAIRED_WINS = 4
RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MAXIMUM_RSS_RATIO = 1.10
_SOURCE_SHA_LENGTH = 40
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_DEVELOPMENT_SCREEN_ROOT = (
    _REPOSITORY_ROOT / "output" / "open-ecology" / "development-screens"
)
_DEVELOPMENT_SCREEN_ROOT_ENV = "EVOLUTION_SIM_OPEN_ECOLOGY_DEVELOPMENT_SCREEN_ROOT"
_CANONICAL_DEVELOPMENT_SCREEN_REPORT_NAME = (
    "recurrent-kernel-development-screen.json"
)
_CANONICAL_HOLDOUT_RETIREMENT_MARKER_NAME = (
    "recurrent-kernel-development-screen.holdout-retired.json"
)
_CANONICAL_SOURCE_RETIREMENT_MARKER_NAME = (
    "recurrent-kernel-development-screen.source-retired.json"
)
_CHILD_TIMEOUT_SECONDS = 30 * 60
_INTERACTIVE_FRAME_MAX_BYTES = 64 * 1024 * 1024
_INTERACTIVE_TRAILING_STDOUT_MAX_BYTES = 64 * 1024
_INTERACTIVE_IO_CHUNK_BYTES = 64 * 1024
_CHILD_PROTOCOL_VERSION = (
    "recurrent_accuracy_interactive_child_protocol_v1"
)
_PREHOLDOUT_EVIDENCE_VERSION = (
    "recurrent_accuracy_preholdout_evidence_bundle_v1"
)
_NONSELECTABLE_ACCURACY_SENTINEL_VERSION = (
    "recurrent_accuracy_nonselectable_holdout_sentinel_v1"
)
_COMPLETED_CHILD_COMMON_KEYS = frozenset(
    {
        "candidate",
        "status",
        "child_nonce",
        "development_only",
        "launch_authorized",
        "authority_evidence_eligible",
        "scientific_result",
        "selectable_for_implementation",
        "candidate_implementation",
        "source_state",
        "process_binding",
        "accuracy_preregistration_digest",
        "shape",
        "model_contract_version",
        "source_numeric_kernel_version",
        "proof_seed_contract",
        "fixed_batch_runtime_binding",
        "legacy_fixed_batch_runtime_binding",
        "legacy_runtime_paths",
        "legacy_runtime_equivalence",
        "equivalence",
        "collector_equivalence",
        "timing",
        "peak_rss_bytes",
        "elapsed_ns",
        "runtime",
        "accuracy_evidence",
    }
)
_CONTROL_CHILD_KEYS = _COMPLETED_CHILD_COMMON_KEYS | {
    "child_exact_digest"
}
_SELECTABLE_PREHOLDOUT_CHILD_KEYS = _COMPLETED_CHILD_COMMON_KEYS | {
    "preholdout_exact_digest"
}
_SELECTABLE_COMPLETED_CHILD_KEYS = _SELECTABLE_PREHOLDOUT_CHILD_KEYS | {
    "holdout_authorization_digest",
    "child_exact_digest",
}
_CHILD_RUNTIME_KEYS = frozenset(
    {
        "python_version",
        "torch_version",
        "torch_cuda_available",
        "platform",
        "machine",
        "torch_num_threads",
        "torch_num_interop_threads",
        "pythonhashseed",
    }
)
_SELECTABLE_PREHOLDOUT_ACCURACY_KEYS = frozenset(
    {
        "applicable",
        "candidate_locked_before_holdout",
        "holdout_access_count",
        "holdout_consumed",
        "public_forward",
        "gradients",
        "holdout_forward",
        "holdout_receipt",
        "preregistration_digest",
        "preholdout_candidate_exact_digest",
    }
)
_SELECTABLE_COMPLETED_ACCURACY_KEYS = (
    _SELECTABLE_PREHOLDOUT_ACCURACY_KEYS
    | {
        "preholdout_evidence_digest",
        "retirement_marker_path",
        "retirement_marker_sha256",
    }
)
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
        "recurrent_policy": "python/evolution_sim/mind/recurrent_policy.py",
        "recurrent_ppo": "python/evolution_sim/mind/recurrent_ppo.py",
        "recurrent_accuracy_screen": (
            "python/evolution_sim/mind/recurrent_accuracy_screen.py"
        ),
        "phase_a_behavioral_evidence": (
            "python/evolution_sim/mind/"
            "open_ecology_phase_a_behavioral_evidence.py"
        ),
        "recurrent_kernel_development_screen": (
            "python/evolution_sim/mind/recurrent_kernel_development_screen.py"
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
        "recurrent_policy",
        "recurrent_ppo",
        "recurrent_accuracy_screen",
        "recurrent_kernel_development_screen",
        "phase_a_behavioral_evidence",
    }
)
_CANDIDATE_SELECTABLE = {
    "per_row_bmm_legacy_adapter_baseline_v2": False,
    "per_row_bmm_fast_adapter_control_v3": False,
    "fixed_row_tile_4_fast_adapter_candidate_v3": True,
    "fixed_row_tile_4_legacy_adapter_control_v3": False,
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
    "per_row_bmm_fast_adapter_control_v3": {
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
        "purpose": "nonselectable_fast_adapter_timing_and_semantic_control",
    },
    "fixed_row_tile_4_fast_adapter_candidate_v3": {
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
        "purpose": "sole_selectable_shared_trainable_forward_candidate",
    },
    "fixed_row_tile_4_legacy_adapter_control_v3": {
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
_ACCURACY_METHOD_DOCUMENT_PATH = (
    "docs/research/open-ecology-recurrent-accuracy-screen-method-v1.md"
)
_D04_ORACLE_RELATIVE_TOLERANCE = 1e-5
_D04_ORACLE_ABSOLUTE_TOLERANCE = 1e-6


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
        elif candidate == "per_row_bmm_fast_adapter_control_v3":
            pass
        elif candidate == "fixed_row_tile_4_fast_adapter_candidate_v3":
            recurrent_actor_critic._backend_stable_linear_forward = (
                recurrent_actor_critic._fixed_row_tile_4_linear_forward
            )
        elif candidate == "fixed_row_tile_4_legacy_adapter_control_v3":
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


def _source_artifact_sha256(relative_path: str) -> str:
    path = _REPOSITORY_ROOT / relative_path
    resolved = path.resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or not resolved.is_relative_to(_REPOSITORY_ROOT)
        or resolved != path
    ):
        raise RecurrentKernelDevelopmentScreenError(
            f"source artifact {relative_path!r} is missing, linked, or outside checkout"
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bound_module_sha256() -> dict[str, str]:
    return {
        name: _source_artifact_sha256(relative_path)
        for name, relative_path in _SOURCE_BOUND_MODULE_PATHS.items()
    }


def _source_bound_path_sha256() -> dict[str, str]:
    return {
        relative_path: _source_artifact_sha256(relative_path)
        for relative_path in _SOURCE_BOUND_MODULE_PATHS.values()
    }


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
    symbolic_head = subprocess.run(
        [
            "git",
            "-C",
            str(_REPOSITORY_ROOT),
            "symbolic-ref",
            "-q",
            "--short",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if head != expected_source_sha:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen source SHA does not match the expected commit"
        )
    if status:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen requires a completely clean source tree"
        )
    if symbolic_head.returncode != 1 or symbolic_head.stdout.strip():
        raise RecurrentKernelDevelopmentScreenError(
            "development screen requires a detached exact-source checkout"
        )
    if Path(top_level).resolve() != _REPOSITORY_ROOT:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen repository root binding drifted"
        )
    from evolution_sim.mind import recurrent_accuracy_screen

    screen_module = sys.modules.get(__name__)
    if screen_module is None:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen source module is not imported"
        )
    imported_modules = {
        "observations": observations,
        "policy_inputs": policy_inputs,
        "recurrent_actor_critic": recurrent_actor_critic,
        "recurrent_experiment": recurrent_experiment,
        "recurrent_rollout": recurrent_rollout,
        "recurrent_policy": recurrent_policy,
        "recurrent_ppo": recurrent_ppo,
        "recurrent_accuracy_screen": recurrent_accuracy_screen,
        "phase_a_behavioral_evidence": (open_ecology_phase_a_behavioral_evidence),
        "recurrent_kernel_development_screen": screen_module,
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
    imported_module_sha256 = _source_bound_module_sha256()
    module_byte_sha256 = _source_bound_path_sha256()
    source_module_bundle_sha256 = stable_payload_digest(
        module_byte_sha256
    )
    return {
        "repository_root": str(_REPOSITORY_ROOT),
        "expected_source_sha": expected_source_sha,
        "observed_source_sha": head,
        "source_clean_including_untracked": True,
        "detached_head": True,
        "imported_module_paths": imported_paths,
        "imported_module_sha256": imported_module_sha256,
        "module_byte_sha256": module_byte_sha256,
        "source_module_bundle_sha256": source_module_bundle_sha256,
        "accuracy_method_document_path": _ACCURACY_METHOD_DOCUMENT_PATH,
        "accuracy_method_document_sha256": _source_artifact_sha256(
            _ACCURACY_METHOD_DOCUMENT_PATH
        ),
    }


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _accuracy_preregistration_for_source(
    source_state: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct the prospective accuracy preregistration from source."""

    from evolution_sim.mind import recurrent_accuracy_screen

    observed_sha = source_state.get("observed_source_sha")
    if not isinstance(observed_sha, str):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy preregistration source SHA is missing"
        )
    preregistration = recurrent_accuracy_screen.build_accuracy_preregistration(
        source_sha=observed_sha,
        checkout_root=str(_REPOSITORY_ROOT),
        source_module_digests=_source_bound_path_sha256(),
        required_source_paths=tuple(_SOURCE_BOUND_MODULE_PATHS.values()),
        method_document_path=_ACCURACY_METHOD_DOCUMENT_PATH,
        method_document_sha256=_source_artifact_sha256(
            _ACCURACY_METHOD_DOCUMENT_PATH
        ),
    )
    recurrent_accuracy_screen.validate_accuracy_preregistration(
        preregistration
    )
    return preregistration


def _nonselectable_accuracy_sentinel(
    *,
    preregistration_digest: str,
) -> dict[str, object]:
    """Return the exact holdout-denial record required for every control."""

    sentinel: dict[str, object] = {
        "schema_version": _NONSELECTABLE_ACCURACY_SENTINEL_VERSION,
        "applicable": False,
        "candidate_locked_before_holdout": False,
        "holdout_access_count": 0,
        "holdout_consumed": False,
        "public_forward": None,
        "gradients": None,
        "holdout_forward": None,
        "holdout_receipt": None,
        "preregistration_digest": preregistration_digest,
        "reason": "nonselectable_child_forbidden_from_accuracy_holdout",
    }
    sentinel["exact_digest"] = stable_payload_digest(sentinel)
    return sentinel


def _validate_nonselectable_accuracy_sentinel(
    value: object,
    *,
    preregistration_digest: str,
) -> dict[str, object]:
    expected = _nonselectable_accuracy_sentinel(
        preregistration_digest=preregistration_digest,
    )
    if value != expected:
        raise RecurrentKernelDevelopmentScreenError(
            "nonselectable child accuracy or holdout sentinel drifted"
        )
    return expected


def _candidate_process_binding(
    *,
    parent_process_id: int,
    parent_process_start_identity: str,
    child_nonce: str,
    report_path: str,
) -> dict[str, object]:
    """Bind one child to its live parent, nonce, and no-clobber report."""

    if (
        isinstance(parent_process_id, bool)
        or not isinstance(parent_process_id, int)
        or parent_process_id <= 0
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "parent process ID must be positive"
        )
    if _process_start_identity(parent_process_id) != parent_process_start_identity:
        raise RecurrentKernelDevelopmentScreenError(
            "parent process start identity drifted"
        )
    resolved_report = validate_development_screen_report_path(
        Path(report_path),
        allow_source_retirement_marker=True,
    )
    binding = {
        "parent_process_id": parent_process_id,
        "parent_process_start_identity": parent_process_start_identity,
        "child_process_id": os.getpid(),
        "child_process_start_identity": _process_start_identity(os.getpid()),
        "child_nonce": child_nonce,
        "report_path": str(resolved_report),
    }
    binding["exact_digest"] = stable_payload_digest(binding)
    return binding


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
    report_path: str,
    accuracy_preregistration_digest: str,
    parent_process_id: int,
    parent_process_start_identity: str,
    timing_repeats: int = RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS,
    warmup_pairs: int = RECURRENT_KERNEL_DEVELOPMENT_SCREEN_WARMUP_PAIRS,
) -> dict[str, object]:
    """Run one candidate in the current, dedicated child process."""

    if (
        isinstance(timing_repeats, bool)
        or not isinstance(timing_repeats, int)
        or timing_repeats
        != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS
        or isinstance(warmup_pairs, bool)
        or not isinstance(warmup_pairs, int)
        or warmup_pairs
        != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_WARMUP_PAIRS
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "screen timing and warmup counts differ from the frozen contract"
        )
    if (
        not isinstance(child_nonce, str)
        or len(child_nonce) != 64
        or any(character not in "0123456789abcdef" for character in child_nonce)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "child nonce must be a lowercase 256-bit hex token"
        )
    source_state = _git_source_state(expected_source_sha)
    accuracy_preregistration = _accuracy_preregistration_for_source(
        source_state
    )
    if (
        _strict_sha256(
            accuracy_preregistration_digest,
            field="accuracy preregistration digest",
        )
        != accuracy_preregistration["exact_digest"]
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "child accuracy preregistration differs from its parent"
        )
    process_binding = _candidate_process_binding(
        parent_process_id=parent_process_id,
        parent_process_start_identity=parent_process_start_identity,
        child_nonce=child_nonce,
        report_path=report_path,
    )
    _configure_dedicated_child_torch_runtime()
    started = time.perf_counter_ns()
    public_forward: dict[str, object] | None = None
    gradients: dict[str, object] | None = None
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
        if _CANDIDATE_SELECTABLE[candidate]:
            from evolution_sim.mind import recurrent_accuracy_screen

            public_forward = (
                recurrent_accuracy_screen.evaluate_public_forward_corpus(
                    recurrent_actor_critic._backend_stable_linear
                )
            )
            gradients = recurrent_accuracy_screen.evaluate_gradient_probe(
                recurrent_actor_critic._backend_stable_linear
            )
    final_source_state = _git_source_state(expected_source_sha)
    if final_source_state != source_state:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate source state changed during execution"
        )
    result: dict[str, object] = {
        "candidate": candidate,
        "status": (
            "preholdout_completed"
            if _CANDIDATE_SELECTABLE[candidate]
            else "completed"
        ),
        "child_nonce": child_nonce,
        "development_only": True,
        "launch_authorized": False,
        "authority_evidence_eligible": False,
        "scientific_result": False,
        "selectable_for_implementation": _CANDIDATE_SELECTABLE[candidate],
        "candidate_implementation": _CANDIDATE_IMPLEMENTATION[candidate],
        "source_state": final_source_state,
        "process_binding": process_binding,
        "accuracy_preregistration_digest": (
            accuracy_preregistration_digest
        ),
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
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            **observed_thread_runtime,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }
    if _CANDIDATE_SELECTABLE[candidate]:
        if public_forward is None or gradients is None:
            raise RecurrentKernelDevelopmentScreenError(
                "selectable candidate public accuracy evidence is missing"
            )
        accuracy_evidence: dict[str, object] = {
            "applicable": True,
            "candidate_locked_before_holdout": True,
            "holdout_access_count": 0,
            "holdout_consumed": False,
            "public_forward": public_forward,
            "gradients": gradients,
            "holdout_forward": None,
            "holdout_receipt": None,
            "preregistration_digest": accuracy_preregistration_digest,
        }
        accuracy_evidence["preholdout_candidate_exact_digest"] = (
            stable_payload_digest(accuracy_evidence)
        )
        result["accuracy_evidence"] = accuracy_evidence
        result["preholdout_exact_digest"] = stable_payload_digest(result)
    else:
        result["accuracy_evidence"] = _nonselectable_accuracy_sentinel(
            preregistration_digest=accuracy_preregistration_digest
        )
        result["child_exact_digest"] = stable_payload_digest(result)
    return result


def complete_candidate_holdout_screen(
    preholdout_result: Mapping[str, object],
    *,
    authorization: Mapping[str, object],
) -> dict[str, object]:
    """Consume the parent-issued capability after every control is sealed."""

    from evolution_sim.mind import recurrent_accuracy_screen

    gate = _candidate_preholdout_integrity_gate(
        preholdout_result,
        reexecute_public_evidence=False,
    )
    if gate.get("passed") is not True:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate preholdout evidence failed before capability use: "
            + "; ".join(str(reason) for reason in gate.get("reasons", ()))
        )
    expected_authorization_keys = {
        "protocol_version",
        "decision",
        "capability_token",
        "source_sha",
        "source_module_bundle_sha256",
        "method_document_sha256",
        "report_path",
        "accuracy_preregistration_digest",
        "preholdout_evidence_digest",
        "closing_baseline_evidence_digest",
        "parent_process_id",
        "parent_process_start_identity",
        "child_process_id",
        "child_process_start_identity",
        "child_nonce",
        "capability_commitment_sha256",
        "source_retirement_marker_path",
        "source_retirement_marker_sha256",
        "retirement_marker_path",
        "retirement_marker_sha256",
        "exact_digest",
    }
    if set(authorization) != expected_authorization_keys:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout authorization schema drifted"
        )
    authorization_payload = dict(authorization)
    authorization_digest = authorization_payload.pop("exact_digest", None)
    if (
        authorization.get("protocol_version") != _CHILD_PROTOCOL_VERSION
        or authorization.get("decision") != "consume_holdout"
        or _strict_sha256(
            authorization_digest,
            field="holdout authorization digest",
        )
        != stable_payload_digest(authorization_payload)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout authorization is malformed"
        )
    process_binding = preholdout_result.get("process_binding")
    source_state = preholdout_result.get("source_state")
    if not isinstance(process_binding, Mapping) or not isinstance(
        source_state,
        Mapping,
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate source or process binding is missing"
        )
    bound_fields = (
        "parent_process_id",
        "parent_process_start_identity",
        "child_process_id",
        "child_process_start_identity",
        "child_nonce",
        "report_path",
    )
    if any(
        authorization.get(field) != process_binding.get(field)
        for field in bound_fields
    ) or (
        authorization.get("source_sha")
        != source_state.get("observed_source_sha")
        or authorization.get("source_module_bundle_sha256")
        != source_state.get("source_module_bundle_sha256")
        or authorization.get("method_document_sha256")
        != source_state.get("accuracy_method_document_sha256")
        or authorization.get("accuracy_preregistration_digest")
        != preholdout_result.get("accuracy_preregistration_digest")
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout authorization changed a source, process, or report binding"
        )
    marker_path = Path(str(authorization["retirement_marker_path"]))
    marker_sha256 = _strict_sha256(
        authorization.get("retirement_marker_sha256"),
        field="retirement marker SHA256",
    )
    if (
        not marker_path.is_absolute()
        or not marker_path.is_file()
        or marker_path.parent
        != Path(str(process_binding["report_path"])).parent
        or hashlib.sha256(marker_path.read_bytes()).hexdigest()
        != marker_sha256
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout retirement marker is absent or changed"
        )
    capability_token = _strict_sha256(
        authorization.get("capability_token"),
        field="holdout capability token",
    )
    capability_commitment = _strict_sha256(
        authorization.get("capability_commitment_sha256"),
        field="holdout capability commitment",
    )
    if (
        hashlib.sha256(bytes.fromhex(capability_token)).hexdigest()
        != capability_commitment
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout capability does not open the precommitted value"
        )
    source_marker_path = Path(
        str(authorization["source_retirement_marker_path"])
    )
    source_marker_sha256 = _strict_sha256(
        authorization.get("source_retirement_marker_sha256"),
        field="source retirement marker SHA256",
    )
    if (
        not source_marker_path.is_absolute()
        or not source_marker_path.is_file()
        or source_marker_path.parent != marker_path.parent
        or hashlib.sha256(source_marker_path.read_bytes()).hexdigest()
        != source_marker_sha256
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "source retirement marker is absent or changed"
        )
    capability = recurrent_accuracy_screen.issue_holdout_capability(
        preimage=capability_token,
        capability_commitment_sha256=capability_commitment,
        preholdout_evidence_digest=str(
            authorization["preholdout_evidence_digest"]
        ),
        source_retirement_marker_sha256=source_marker_sha256,
        holdout_retirement_marker_sha256=marker_sha256,
        candidate_identity=(
            "fixed_row_tile_4_fast_adapter_candidate_v3"
        ),
        closing_baseline_validated=True,
        source_retirement_marker_exists=source_marker_path.is_file(),
        holdout_retirement_marker_exists=marker_path.is_file(),
    )
    seeds = recurrent_accuracy_screen.sealed_accuracy_seed_contract()
    authority = recurrent_accuracy_screen.HoldoutSeedAuthority(
        seeds=seeds,
        capability=capability,
        source_sha=str(source_state["observed_source_sha"]),
        source_clean_detached=(
            source_state.get("source_clean_including_untracked") is True
            and source_state.get("detached_head") is True
        ),
        checkout_root=str(source_state["repository_root"]),
        required_source_paths=tuple(
            _SOURCE_BOUND_MODULE_PATHS.values()
        ),
        source_module_digests=_source_bound_path_sha256(),
        method_document_path=_ACCURACY_METHOD_DOCUMENT_PATH,
        method_document_digest=str(
            source_state["accuracy_method_document_sha256"]
        ),
        parent_process_id=int(process_binding["parent_process_id"]),
        parent_process_start_identity=str(
            process_binding["parent_process_start_identity"]
        ),
        child_process_id=int(process_binding["child_process_id"]),
        child_process_start_identity=str(
            process_binding["child_process_start_identity"]
        ),
        child_nonce=str(process_binding["child_nonce"]),
        report_path=str(process_binding["report_path"]),
        preregistration_digest=str(
            preholdout_result["accuracy_preregistration_digest"]
        ),
        candidate_identity=(
            "fixed_row_tile_4_fast_adapter_candidate_v3"
        ),
        preholdout_evidence_digest=str(
            authorization["preholdout_evidence_digest"]
        ),
        source_retirement_marker_path=str(source_marker_path),
        source_retirement_marker_sha256=source_marker_sha256,
        holdout_retirement_marker_path=str(marker_path),
        holdout_retirement_marker_sha256=marker_sha256,
        capability_commitment_sha256=capability_commitment,
        closing_baseline_evidence_digest=str(
            authorization["closing_baseline_evidence_digest"]
        ),
    )
    with (
        recurrent_kernel_candidate(
            "fixed_row_tile_4_fast_adapter_candidate_v3"
        ),
        _isolated_torch_process_state(num_threads=1),
    ):
        holdout = (
            recurrent_accuracy_screen.evaluate_holdout_forward_corpus(
                recurrent_actor_critic._backend_stable_linear,
                authority=authority,
                capability=capability,
            )
        )
    final_source_state = _git_source_state(
        str(source_state["observed_source_sha"])
    )
    if final_source_state != source_state:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate source changed during holdout materialization"
        )
    result = dict(preholdout_result)
    result["status"] = "completed"
    accuracy_evidence = dict(result["accuracy_evidence"])  # type: ignore[arg-type]
    accuracy_evidence.update(
        {
            "holdout_access_count": 1,
            "holdout_consumed": True,
            "holdout_forward": holdout,
            "holdout_receipt": holdout["holdout_receipt"],
            "preholdout_evidence_digest": authorization[
                "preholdout_evidence_digest"
            ],
            "retirement_marker_path": str(marker_path),
            "retirement_marker_sha256": marker_sha256,
        }
    )
    result["accuracy_evidence"] = accuracy_evidence
    result["holdout_authorization_digest"] = authorization["exact_digest"]
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
        or any(
            isinstance(observed.get(field), bool)
            or not isinstance(observed.get(field), int)
            or observed.get(field) != 1
            for field in (
                "torch_num_threads",
                "torch_num_interop_threads",
            )
        )
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


def _validated_child_runtime(
    value: object,
    *,
    candidate_runtime_binding: Mapping[str, object],
    legacy_runtime_binding: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _CHILD_RUNTIME_KEYS:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate deterministic runtime schema drifted"
        )
    if any(
        not isinstance(value.get(field), str) or not value.get(field)
        for field in (
            "python_version",
            "torch_version",
            "platform",
            "machine",
        )
    ) or (
        value.get("pythonhashseed") != "0"
        or any(
            isinstance(value.get(field), bool)
            or not isinstance(value.get(field), int)
            or value.get(field) != 1
            for field in (
                "torch_num_threads",
                "torch_num_interop_threads",
            )
        )
        or value.get("torch_cuda_available") is not False
        or candidate_runtime_binding["observed_runtime"].get(
            "torch_version"
        )
        != value.get("torch_version")
        or legacy_runtime_binding["observed_runtime"].get("torch_version")
        != value.get("torch_version")
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate deterministic runtime binding drifted"
        )
    return dict(value)


def _validated_source_state(value: object, *, field: str) -> dict[str, object]:
    expected_keys = {
        "repository_root",
        "expected_source_sha",
        "observed_source_sha",
        "source_clean_including_untracked",
        "detached_head",
        "imported_module_paths",
        "imported_module_sha256",
        "module_byte_sha256",
        "source_module_bundle_sha256",
        "accuracy_method_document_path",
        "accuracy_method_document_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} source schema drifted"
        )
    expected_sha = value.get("expected_source_sha")
    observed_sha = value.get("observed_source_sha")
    imported_paths = value.get("imported_module_paths")
    imported_module_sha256 = value.get("imported_module_sha256")
    module_byte_sha256 = value.get("module_byte_sha256")
    expected_module_sha256 = _source_bound_module_sha256()
    expected_path_sha256 = _source_bound_path_sha256()
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != _SOURCE_SHA_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or observed_sha != expected_sha
        or value.get("source_clean_including_untracked") is not True
        or value.get("detached_head") is not True
        or not isinstance(value.get("repository_root"), str)
        or Path(str(value["repository_root"])).resolve() != _REPOSITORY_ROOT
        or not isinstance(imported_paths, Mapping)
        or imported_paths != _SOURCE_BOUND_MODULE_PATHS
        or not isinstance(imported_module_sha256, Mapping)
        or imported_module_sha256 != expected_module_sha256
        or not isinstance(module_byte_sha256, Mapping)
        or module_byte_sha256 != expected_path_sha256
        or value.get("source_module_bundle_sha256")
        != stable_payload_digest(expected_path_sha256)
        or value.get("accuracy_method_document_path")
        != _ACCURACY_METHOD_DOCUMENT_PATH
        or value.get("accuracy_method_document_sha256")
        != _source_artifact_sha256(_ACCURACY_METHOD_DOCUMENT_PATH)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} exact-source binding is malformed"
        )
    return dict(value)


def _validated_candidate_process_binding(
    value: object,
    *,
    child_nonce: str,
) -> dict[str, object]:
    expected_keys = {
        "parent_process_id",
        "parent_process_start_identity",
        "child_process_id",
        "child_process_start_identity",
        "child_nonce",
        "report_path",
        "exact_digest",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate process binding schema drifted"
        )
    payload = dict(value)
    exact_digest = payload.pop("exact_digest", None)
    for field in ("parent_process_id", "child_process_id"):
        _strict_positive_int(value.get(field), field=field)
    if (
        value.get("child_nonce") != child_nonce
        or not isinstance(value.get("parent_process_start_identity"), str)
        or not value.get("parent_process_start_identity")
        or not isinstance(value.get("child_process_start_identity"), str)
        or not value.get("child_process_start_identity")
        or not isinstance(value.get("report_path"), str)
        or not Path(str(value["report_path"])).is_absolute()
        or Path(str(value["report_path"])).name
        != _CANONICAL_DEVELOPMENT_SCREEN_REPORT_NAME
        or _strict_sha256(
            exact_digest,
            field="candidate process binding digest",
        )
        != stable_payload_digest(payload)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate process binding is malformed"
        )
    return dict(value)


def _validate_candidate_identity(
    candidate: Mapping[str, object],
    *,
    expected_status: str = "completed",
    digest_field: str = "child_exact_digest",
) -> str:
    name = candidate.get("candidate")
    if not isinstance(name, str) or name not in _CANDIDATE_IMPLEMENTATION:
        raise RecurrentKernelDevelopmentScreenError("candidate identity is unknown")
    if expected_status == "preholdout_completed":
        expected_keys = _SELECTABLE_PREHOLDOUT_CHILD_KEYS
    elif _CANDIDATE_SELECTABLE[name]:
        expected_keys = _SELECTABLE_COMPLETED_CHILD_KEYS
    else:
        expected_keys = _CONTROL_CHILD_KEYS
    if set(candidate) != expected_keys:
        raise RecurrentKernelDevelopmentScreenError(
            "candidate child envelope schema drifted"
        )
    if (
        expected_status == "completed"
        and _CANDIDATE_SELECTABLE[name]
    ):
        _strict_sha256(
            candidate.get("holdout_authorization_digest"),
            field="holdout authorization digest",
        )
    shape = candidate.get("shape")
    if (
        not isinstance(shape, Mapping)
        or set(shape) != set(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE)
        or any(
            isinstance(shape.get(field), bool)
            or not isinstance(shape.get(field), int)
            or shape.get(field)
            != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE[field]
            for field in RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SHAPE
        )
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate screen shape drifted"
        )
    if (
        candidate.get("status") != expected_status
        or candidate.get("development_only") is not True
        or candidate.get("launch_authorized") is not False
        or candidate.get("authority_evidence_eligible") is not False
        or candidate.get("scientific_result") is not False
        or candidate.get("selectable_for_implementation")
        is not _CANDIDATE_SELECTABLE[name]
        or candidate.get("candidate_implementation") != _CANDIDATE_IMPLEMENTATION[name]
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
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise RecurrentKernelDevelopmentScreenError("candidate child nonce drifted")
    digest = candidate.get(digest_field)
    payload = dict(candidate)
    payload.pop(digest_field, None)
    if _strict_sha256(digest, field=digest_field) != stable_payload_digest(
        payload
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate child digest does not match its payload"
        )
    _validated_source_state(
        candidate.get("source_state"),
        field="candidate",
    )
    _validated_candidate_process_binding(
        candidate.get("process_binding"),
        child_nonce=nonce,
    )
    _strict_sha256(
        candidate.get("accuracy_preregistration_digest"),
        field="candidate accuracy preregistration digest",
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
    _validated_child_runtime(
        candidate.get("runtime"),
        candidate_runtime_binding=candidate_runtime_binding,
        legacy_runtime_binding=legacy_runtime_binding,
    )
    _strict_positive_int(candidate.get("peak_rss_bytes"), field="peak RSS")
    _strict_positive_int(candidate.get("elapsed_ns"), field="candidate elapsed time")
    return name


def _validate_equivalence(
    candidate: Mapping[str, object],
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
    return max_ratio, bucket


def _validate_collector_and_timing(
    candidate: Mapping[str, object],
) -> tuple[float, float]:
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
    bootstrap_comparison_count = _strict_positive_int(
        collector.get("bootstrap_value_comparison_count"),
        field="collector bootstrap comparison count",
    )
    if (
        collector.get("batched_transition_count") != transition_count
        or collector.get("paired_transition_count") != transition_count
        or collector.get("numeric_transition_comparison_count") != transition_count
        or collector.get("hidden_component_comparison_count")
        != transition_count * _HIDDEN_SIZE
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
            != bootstrap_comparison_count
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
        _strict_positive_int(
            comparison.get("bootstrap_value_comparison_count"),
            field=(
                f"legacy_runtime.{mode}.bootstrap_value_comparison_count"
            ),
        )
        if (
            comparison.get("transition_count") != transition_count
            or comparison.get("batched_transition_count") != transition_count
            or comparison.get("paired_transition_count") != transition_count
            or comparison.get("numeric_transition_comparison_count")
            != transition_count
            or comparison.get("hidden_component_comparison_count")
            != transition_count * _HIDDEN_SIZE
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
        candidate_semantics = timing[mode]["semantic_sha256"]  # type: ignore[index]
        if (
            comparison.get("ordered_merge_semantic_sha256")
            != candidate_semantics[0]
        ):
            raise RecurrentKernelDevelopmentScreenError(
                f"candidate {mode} legacy comparison is not bound to timed semantics"
            )
        legacy_ratios.extend(mode_ratios)
    return max(collector_ratios), max(legacy_ratios)


def _preholdout_accuracy_payload(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "applicable": True,
        "candidate_locked_before_holdout": True,
        "holdout_access_count": 0,
        "holdout_consumed": False,
        "public_forward": evidence.get("public_forward"),
        "gradients": evidence.get("gradients"),
        "holdout_forward": None,
        "holdout_receipt": None,
        "preregistration_digest": evidence.get(
            "preregistration_digest"
        ),
    }


def _validate_selectable_preholdout_accuracy(
    candidate: Mapping[str, object],
    *,
    reexecute_public_evidence: bool,
    expect_holdout_consumed: bool,
) -> dict[str, object]:
    """Validate all candidate evidence that must precede holdout admission."""

    from evolution_sim.mind import recurrent_accuracy_screen

    evidence = candidate.get("accuracy_evidence")
    if not isinstance(evidence, Mapping):
        raise RecurrentKernelDevelopmentScreenError(
            "selectable candidate accuracy evidence is missing"
        )
    expected_evidence_keys = (
        _SELECTABLE_COMPLETED_ACCURACY_KEYS
        if expect_holdout_consumed
        else _SELECTABLE_PREHOLDOUT_ACCURACY_KEYS
    )
    if set(evidence) != expected_evidence_keys:
        raise RecurrentKernelDevelopmentScreenError(
            "selectable candidate accuracy evidence schema drifted"
        )
    preregistration_digest = _strict_sha256(
        candidate.get("accuracy_preregistration_digest"),
        field="candidate accuracy preregistration digest",
    )
    preholdout = _preholdout_accuracy_payload(evidence)
    holdout_access_count = evidence.get("holdout_access_count")
    if (
        evidence.get("applicable") is not True
        or evidence.get("candidate_locked_before_holdout") is not True
        or isinstance(holdout_access_count, bool)
        or not isinstance(holdout_access_count, int)
        or holdout_access_count != int(expect_holdout_consumed)
        or evidence.get("holdout_consumed")
        is not expect_holdout_consumed
        or (
            not expect_holdout_consumed
            and (
                evidence.get("holdout_forward") is not None
                or evidence.get("holdout_receipt") is not None
            )
        )
        or evidence.get("preregistration_digest")
        != preregistration_digest
        or _strict_sha256(
            evidence.get("preholdout_candidate_exact_digest"),
            field="candidate preholdout accuracy digest",
        )
        != stable_payload_digest(preholdout)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "selectable candidate preholdout accuracy binding drifted"
        )
    public_forward = evidence.get("public_forward")
    gradients = evidence.get("gradients")
    if not isinstance(public_forward, dict) or not isinstance(
        gradients,
        dict,
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "selectable candidate public or gradient evidence is missing"
        )
    if reexecute_public_evidence:
        with (
            recurrent_kernel_candidate(
                "fixed_row_tile_4_fast_adapter_candidate_v3"
            ),
            _isolated_torch_process_state(num_threads=1),
        ):
            recurrent_accuracy_screen.validate_public_forward_report(
                public_forward,
                candidate_forward=(
                    recurrent_actor_critic._backend_stable_linear
                ),
            )
            recurrent_accuracy_screen.validate_gradient_probe_report(
                gradients,
                candidate_forward=(
                    recurrent_actor_critic._backend_stable_linear
                ),
            )
    public_aggregate = public_forward.get("aggregate")
    gradient_aggregate = gradients.get("aggregate")
    if (
        not isinstance(public_aggregate, Mapping)
        or public_aggregate.get("forward_gate_passed") is not True
        or not isinstance(gradient_aggregate, Mapping)
        or gradient_aggregate.get("gradient_gate_passed") is not True
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate public accuracy or gradient gate failed"
        )
    return dict(evidence)


def _validate_selectable_accuracy_evidence(
    candidate: Mapping[str, object],
    *,
    reexecute_public_evidence: bool,
    allow_synthetic_holdout: bool = False,
) -> dict[str, object]:
    """Validate public evidence plus one offline-only consumed holdout."""

    from evolution_sim.mind import recurrent_accuracy_screen

    evidence = _validate_selectable_preholdout_accuracy(
        candidate,
        reexecute_public_evidence=reexecute_public_evidence,
        expect_holdout_consumed=True,
    )
    if (
        evidence.get("holdout_access_count") != 1
        or evidence.get("holdout_consumed") is not True
        or not isinstance(evidence.get("holdout_forward"), dict)
        or evidence.get("holdout_receipt")
        != evidence["holdout_forward"].get("holdout_receipt")
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "selectable candidate holdout lifecycle drifted"
        )
    receipt = evidence["holdout_receipt"]
    if (
        not isinstance(receipt, Mapping)
        or evidence.get("preholdout_evidence_digest")
        != receipt.get("preholdout_evidence_digest")
        or evidence.get("retirement_marker_path")
        != receipt.get("holdout_retirement_marker_path")
        or evidence.get("retirement_marker_sha256")
        != receipt.get("holdout_retirement_marker_sha256")
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "selectable candidate holdout retirement binding drifted"
        )
    preregistration = _accuracy_preregistration_for_source(
        candidate["source_state"]  # type: ignore[arg-type]
    )
    contract = preregistration.get("corpus_contract")
    if not isinstance(contract, Mapping):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy preregistration corpus contract is missing"
        )
    accuracy_seeds = (
        recurrent_accuracy_screen.sealed_accuracy_seed_contract()
    )
    tables = contract.get("tables")
    if not isinstance(tables, Mapping) or not isinstance(
        tables.get("holdout"),
        Sequence,
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy preregistration holdout cases are missing"
        )
    recurrent_accuracy_screen.validate_holdout_report_offline(
        evidence["holdout_forward"],
        expected_seed_contract=accuracy_seeds,
        expected_case_contracts=tables["holdout"],
        allow_synthetic_test=allow_synthetic_holdout,
    )
    if not allow_synthetic_holdout:
        recurrent_accuracy_screen.validate_deciding_holdout_report(
            evidence["holdout_forward"],
            expected_seed_contract=accuracy_seeds,
        )
    holdout_aggregate = evidence["holdout_forward"].get("aggregate")
    if (
        not isinstance(holdout_aggregate, Mapping)
        or holdout_aggregate.get("forward_gate_passed") is not True
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate holdout accuracy gate failed"
        )
    return evidence


def _candidate_preholdout_integrity_gate(
    candidate: Mapping[str, object],
    *,
    reexecute_public_evidence: bool,
) -> dict[str, object]:
    try:
        name = _validate_candidate_identity(
            candidate,
            expected_status="preholdout_completed",
            digest_field="preholdout_exact_digest",
        )
        if name != "fixed_row_tile_4_fast_adapter_candidate_v3":
            raise RecurrentKernelDevelopmentScreenError(
                "preholdout child is not the sole selectable candidate"
            )
        max_ratio, _bucket = _validate_equivalence(candidate)
        max_collector_ratio, max_legacy_rounding_ratio = (
            _validate_collector_and_timing(candidate)
        )
        _validate_selectable_preholdout_accuracy(
            candidate,
            reexecute_public_evidence=reexecute_public_evidence,
            expect_holdout_consumed=False,
        )
    except (
        KeyError,
        TypeError,
        RecurrentKernelDevelopmentScreenError,
    ) as exc:
        return {
            "passed": False,
            "reasons": [str(exc)],
            "max_numeric_tolerance_ratio": None,
        }
    return {
        "passed": True,
        "reasons": [],
        "max_numeric_tolerance_ratio": max(
            max_ratio,
            max_collector_ratio,
        ),
        "legacy_rounding_max_tolerance_ratio_report_only": (
            max_legacy_rounding_ratio
        ),
    }


def _candidate_integrity_gate(
    candidate: Mapping[str, object],
) -> dict[str, object]:
    try:
        name = _validate_candidate_identity(candidate)
        max_ratio, _bucket = _validate_equivalence(candidate)
        max_collector_ratio, max_legacy_rounding_ratio = (
            _validate_collector_and_timing(candidate)
        )
        preregistration_digest = _strict_sha256(
            candidate.get("accuracy_preregistration_digest"),
            field="candidate accuracy preregistration digest",
        )
        if _CANDIDATE_SELECTABLE[name]:
            _validate_selectable_accuracy_evidence(
                candidate,
                reexecute_public_evidence=False,
            )
        else:
            _validate_nonselectable_accuracy_sentinel(
                candidate.get("accuracy_evidence"),
                preregistration_digest=preregistration_digest,
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
        "legacy_rounding_max_tolerance_ratio_report_only": (
            max_legacy_rounding_ratio
        ),
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
    baseline_gate = _candidate_integrity_gate(baseline)
    confirmation_gate = _candidate_integrity_gate(baseline_confirmation)
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
        "per_row_bmm_fast_adapter_control_v3",
        "fixed_row_tile_4_legacy_adapter_control_v3",
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
        integrity = _candidate_integrity_gate(result)
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

    selectable_name = "fixed_row_tile_4_fast_adapter_candidate_v3"
    selectable = by_name.get(selectable_name)
    candidate_reasons: list[str] = []
    if not controls_valid:
        candidate_reasons.append(
            "one or more preregistered nonselectable controls failed"
        )
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


def _write_source_retirement_marker(
    *,
    report_path: Path,
    source_state: Mapping[str, object],
    accuracy_preregistration_digest: str,
    parent_process_id: int,
    parent_process_start_identity: str,
    capability_commitment_sha256: str,
) -> tuple[Path, dict[str, object], str]:
    """Permanently retire the exact source before the first child starts."""

    from evolution_sim.mind import recurrent_accuracy_screen

    resolved_report = validate_development_screen_report_path(report_path)
    marker_path = (
        resolved_report.parent
        / _CANONICAL_SOURCE_RETIREMENT_MARKER_NAME
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker = recurrent_accuracy_screen.source_retirement_marker(
        source_sha=str(source_state.get("observed_source_sha", "")),
        source_module_digest=stable_payload_digest(
            _source_bound_path_sha256()
        ),
        method_document_digest=str(
            source_state.get("accuracy_method_document_sha256", "")
        ),
        preregistration_digest=accuracy_preregistration_digest,
        report_path=str(resolved_report),
        parent_process_id=parent_process_id,
        parent_process_start_identity=(
            parent_process_start_identity
        ),
        capability_commitment_sha256=(
            capability_commitment_sha256
        ),
    )
    serialized = json.dumps(
        marker,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=marker_path.parent,
            prefix=f".{marker_path.name}.pending-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        reopened = json.loads(temporary_path.read_text(encoding="utf-8"))
        if reopened != marker:
            raise RecurrentKernelDevelopmentScreenError(
                "reopened source retirement marker differs from memory"
            )
        os.link(temporary_path, marker_path)
        directory_fd = os.open(marker_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise RecurrentKernelDevelopmentScreenError(
            "source retirement marker already exists"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return marker_path, marker, hashlib.sha256(serialized).hexdigest()


def _holdout_retirement_marker_path(report_path: Path) -> Path:
    resolved_report = validate_development_screen_report_path(
        report_path,
        allow_source_retirement_marker=True,
    )
    marker = (
        resolved_report.parent
        / _CANONICAL_HOLDOUT_RETIREMENT_MARKER_NAME
    )
    if marker.exists():
        raise RecurrentKernelDevelopmentScreenError(
            "holdout retirement marker already exists; this exact-source "
            "screen is permanently retired"
        )
    return marker


def _write_holdout_retirement_marker(
    *,
    report_path: Path,
    preholdout_evidence_digest: str,
    candidate_process_binding: Mapping[str, object],
    source_retirement_marker_path: Path,
    source_retirement_marker_sha256: str,
    capability_commitment_sha256: str,
    baseline_confirmation_digest: str,
) -> tuple[Path, dict[str, object], str]:
    """Atomically retire the exact source before any holdout is revealed."""

    from evolution_sim.mind import recurrent_accuracy_screen

    if (
        not source_retirement_marker_path.is_file()
        or source_retirement_marker_path.parent != report_path.parent
        or hashlib.sha256(
            source_retirement_marker_path.read_bytes()
        ).hexdigest()
        != source_retirement_marker_sha256
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "source retirement marker is absent or changed"
        )
    marker_path = _holdout_retirement_marker_path(report_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    source_marker = json.loads(
        source_retirement_marker_path.read_text(encoding="utf-8")
    )
    if not isinstance(source_marker, dict):
        raise RecurrentKernelDevelopmentScreenError(
            "source retirement marker is malformed"
        )
    marker = recurrent_accuracy_screen.holdout_retirement_marker(
        source_retirement_marker_sha256=(
            source_retirement_marker_sha256
        ),
        source_retirement_marker_exact_digest=str(
            source_marker.get("exact_digest", "")
        ),
        preholdout_evidence_digest=preholdout_evidence_digest,
        closing_baseline_evidence_digest=(
            baseline_confirmation_digest
        ),
        parent_process_id=int(
            candidate_process_binding["parent_process_id"]
        ),
        parent_process_start_identity=str(
            candidate_process_binding[
                "parent_process_start_identity"
            ]
        ),
        child_process_id=int(
            candidate_process_binding["child_process_id"]
        ),
        child_process_start_identity=str(
            candidate_process_binding[
                "child_process_start_identity"
            ]
        ),
        child_nonce=str(candidate_process_binding["child_nonce"]),
        capability_commitment_sha256=(
            capability_commitment_sha256
        ),
    )
    serialized = json.dumps(
        marker,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=marker_path.parent,
            prefix=f".{marker_path.name}.pending-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        reopened = json.loads(temporary_path.read_text(encoding="utf-8"))
        if reopened != marker:
            raise RecurrentKernelDevelopmentScreenError(
                "reopened holdout retirement marker differs from memory"
            )
        reopened_payload = dict(reopened)
        reopened_digest = reopened_payload.pop("exact_digest", None)
        if reopened_digest != stable_payload_digest(reopened_payload):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout retirement marker digest does not reconstruct"
            )
        os.link(temporary_path, marker_path)
        directory_fd = os.open(marker_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout retirement marker already exists"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return marker_path, marker, hashlib.sha256(serialized).hexdigest()


def _failed_child_result(**fields: object) -> dict[str, object]:
    failed = dict(fields)
    failed["child_exact_digest"] = stable_payload_digest(failed)
    return failed


def _validate_failed_child_envelope(
    child: Mapping[str, object],
    *,
    expected_candidate: str,
    allow_preholdout_abort: bool,
) -> dict[str, object]:
    base_keys = {
        "candidate",
        "status",
        "child_nonce",
        "failure",
        "stderr_sha256",
        "stderr_tail",
        "child_exact_digest",
    }
    failure = child.get("failure")
    if failure == "child_timeout":
        expected_keys = base_keys | {"timeout_seconds"}
    elif failure in {"child_nonzero_exit", "child_nonempty_stderr"}:
        expected_keys = base_keys | {"returncode"}
    elif failure == "holdout_admission_denied" and allow_preholdout_abort:
        expected_keys = base_keys | {
            "reason",
            "preholdout_result",
            "returncode",
        }
    else:
        raise RecurrentKernelDevelopmentScreenError(
            "failed child stage is unknown"
        )
    if (
        set(child) != expected_keys
        or child.get("candidate") != expected_candidate
        or child.get("status") != "failed"
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "failed child envelope schema drifted"
        )
    child_nonce = child.get("child_nonce")
    if (
        not isinstance(child_nonce, str)
        or len(child_nonce) != 64
        or any(
            character not in "0123456789abcdef"
            for character in child_nonce
        )
        or not isinstance(child.get("stderr_tail"), str)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "failed child nonce or stderr binding drifted"
        )
    _strict_sha256(
        child.get("stderr_sha256"),
        field="failed child stderr SHA256",
    )
    digest = child.get("child_exact_digest")
    payload = dict(child)
    payload.pop("child_exact_digest", None)
    if _strict_sha256(
        digest,
        field="failed child exact digest",
    ) != stable_payload_digest(payload):
        raise RecurrentKernelDevelopmentScreenError(
            "failed child digest does not reconstruct"
        )
    if failure == "child_timeout":
        if child.get("timeout_seconds") != _CHILD_TIMEOUT_SECONDS:
            raise RecurrentKernelDevelopmentScreenError(
                "failed child timeout contract drifted"
            )
    else:
        returncode = child.get("returncode")
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise RecurrentKernelDevelopmentScreenError(
                "failed child return code is malformed"
            )
        if failure == "child_nonzero_exit" and returncode == 0:
            raise RecurrentKernelDevelopmentScreenError(
                "nonzero-exit failure recorded a zero return code"
            )
        if (
            failure == "child_nonempty_stderr"
            and (
                returncode != 0
                or child.get("stderr_tail") == ""
            )
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "stderr failure record is inconsistent"
            )
        if failure == "holdout_admission_denied":
            if (
                returncode != 0
                or child.get("stderr_tail") != ""
                or child.get("stderr_sha256")
                != hashlib.sha256(b"").hexdigest()
                or not isinstance(child.get("reason"), str)
                or not child.get("reason")
                or not isinstance(
                    child.get("preholdout_result"),
                    Mapping,
                )
            ):
                raise RecurrentKernelDevelopmentScreenError(
                    "clean holdout-abort record is malformed"
                )
    return dict(child)


def _run_child_candidate(
    *,
    candidate: str,
    expected_source_sha: str,
    report_path: str,
    accuracy_preregistration_digest: str,
    parent_process_id: int,
    parent_process_start_identity: str,
    child_command_factory: Callable[
        [str, str, str, str, str, int, str],
        Sequence[str],
    ],
) -> dict[str, object]:
    child_nonce = secrets.token_hex(32)
    child_command = child_command_factory(
        candidate,
        expected_source_sha,
        child_nonce,
        report_path,
        accuracy_preregistration_digest,
        parent_process_id,
        parent_process_start_identity,
    )
    if (
        not isinstance(child_command, Sequence)
        or isinstance(child_command, (str, bytes, bytearray))
        or not child_command
        or any(not isinstance(part, str) or not part for part in child_command)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen child command is malformed"
        )
    try:
        completed = subprocess.run(
            list(child_command),
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
        return _failed_child_result(
            candidate=candidate,
            status="failed",
            child_nonce=child_nonce,
            failure="child_timeout",
            timeout_seconds=_CHILD_TIMEOUT_SECONDS,
            stderr_sha256=hashlib.sha256(
                stderr.encode("utf-8")
            ).hexdigest(),
            stderr_tail=stderr[-4000:],
        )
    if completed.returncode != 0:
        return _failed_child_result(
            candidate=candidate,
            status="failed",
            child_nonce=child_nonce,
            failure="child_nonzero_exit",
            returncode=completed.returncode,
            stderr_sha256=hashlib.sha256(
                completed.stderr.encode("utf-8")
            ).hexdigest(),
            stderr_tail=completed.stderr[-4000:],
        )
    if completed.stderr:
        return _failed_child_result(
            candidate=candidate,
            status="failed",
            child_nonce=child_nonce,
            failure="child_nonempty_stderr",
            returncode=completed.returncode,
            stderr_sha256=hashlib.sha256(
                completed.stderr.encode("utf-8")
            ).hexdigest(),
            stderr_tail=completed.stderr[-4000:],
        )
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


def _read_interactive_child_frame(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    if process.stdout is None:
        raise RecurrentKernelDevelopmentScreenError(
            "interactive candidate stdout pipe is missing"
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "interactive candidate frame timeout must be positive"
        )
    descriptor = process.stdout.fileno()
    deadline = time.monotonic() + float(timeout_seconds)
    framed_bytes = bytearray()

    def parsed_frame() -> dict[str, object]:
        try:
            line = bytes(framed_bytes).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RecurrentKernelDevelopmentScreenError(
                "interactive candidate emitted non-UTF-8 framed JSON"
            ) from exc
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecurrentKernelDevelopmentScreenError(
                "interactive candidate emitted malformed framed JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise RecurrentKernelDevelopmentScreenError(
                "interactive candidate frame must be an object"
            )
        return parsed

    selector = selectors.DefaultSelector()
    original_blocking = os.get_blocking(descriptor)
    try:
        os.set_blocking(descriptor, False)
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            reached_eof = False
            while True:
                try:
                    chunk = os.read(
                        descriptor,
                        _INTERACTIVE_IO_CHUNK_BYTES,
                    )
                except BlockingIOError:
                    break
                except InterruptedError:
                    continue
                if not chunk:
                    reached_eof = True
                    break
                newline_offset = chunk.find(b"\n")
                if newline_offset >= 0:
                    framed_bytes.extend(chunk[:newline_offset])
                    if (
                        len(framed_bytes) > _INTERACTIVE_FRAME_MAX_BYTES
                    ):
                        raise RecurrentKernelDevelopmentScreenError(
                            "interactive candidate frame exceeded the "
                            "maximum byte count"
                        )
                    if newline_offset + 1 != len(chunk):
                        raise RecurrentKernelDevelopmentScreenError(
                            "interactive candidate emitted extra bytes after "
                            "its framed JSON"
                        )
                    while True:
                        try:
                            extra = os.read(descriptor, 1)
                        except BlockingIOError:
                            extra = b""
                        except InterruptedError:
                            continue
                        break
                    if extra:
                        raise RecurrentKernelDevelopmentScreenError(
                            "interactive candidate emitted extra bytes after "
                            "its framed JSON"
                        )
                    return parsed_frame()
                framed_bytes.extend(chunk)
                if len(framed_bytes) > _INTERACTIVE_FRAME_MAX_BYTES:
                    raise RecurrentKernelDevelopmentScreenError(
                        "interactive candidate frame exceeded the maximum "
                        "byte count"
                    )
            if reached_eof:
                if framed_bytes:
                    return parsed_frame()
                raise RecurrentKernelDevelopmentScreenError(
                    "interactive candidate exited before publishing its frame"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise RecurrentKernelDevelopmentScreenError(
                    "interactive candidate frame timed out"
                )
    finally:
        selector.close()
        try:
            os.set_blocking(descriptor, original_blocking)
        except OSError:
            pass


def _wait_for_interactive_child_exit(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float,
) -> tuple[int, bytes]:
    """Drain bounded trailing stdout while waiting, preventing pipe deadlock."""

    if process.stdout is None:
        raise RecurrentKernelDevelopmentScreenError(
            "interactive candidate stdout pipe is missing"
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "interactive candidate exit timeout must be positive"
        )
    descriptor = process.stdout.fileno()
    deadline = time.monotonic() + float(timeout_seconds)
    trailing_stdout = bytearray()
    reached_eof = False
    selector = selectors.DefaultSelector()
    original_blocking = os.get_blocking(descriptor)
    try:
        os.set_blocking(descriptor, False)
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            while not reached_eof:
                remaining_capacity = (
                    _INTERACTIVE_TRAILING_STDOUT_MAX_BYTES
                    + 1
                    - len(trailing_stdout)
                )
                if remaining_capacity <= 0:
                    raise RecurrentKernelDevelopmentScreenError(
                        "interactive candidate trailing stdout exceeded the "
                        "maximum byte count"
                    )
                try:
                    chunk = os.read(
                        descriptor,
                        min(
                            _INTERACTIVE_IO_CHUNK_BYTES,
                            remaining_capacity,
                        ),
                    )
                except BlockingIOError:
                    break
                except InterruptedError:
                    continue
                if not chunk:
                    reached_eof = True
                    break
                trailing_stdout.extend(chunk)
                if (
                    len(trailing_stdout)
                    > _INTERACTIVE_TRAILING_STDOUT_MAX_BYTES
                ):
                    raise RecurrentKernelDevelopmentScreenError(
                        "interactive candidate trailing stdout exceeded the "
                        "maximum byte count"
                    )
            returncode = process.poll()
            if reached_eof and returncode is not None:
                return returncode, bytes(trailing_stdout)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RecurrentKernelDevelopmentScreenError(
                    "interactive candidate did not exit before its deadline"
                )
            if reached_eof:
                try:
                    returncode = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    raise RecurrentKernelDevelopmentScreenError(
                        "interactive candidate did not exit before its "
                        "deadline"
                    ) from exc
                return returncode, bytes(trailing_stdout)
            if not selector.select(remaining):
                raise RecurrentKernelDevelopmentScreenError(
                    "interactive candidate did not exit before its deadline"
                )
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        selector.close()
        try:
            os.set_blocking(descriptor, original_blocking)
        except OSError:
            pass


def _cleanup_interactive_candidate(child: SimpleNamespace) -> None:
    process = child.process
    if process.poll() is None:
        process.kill()
        process.wait()
    for stream_name in ("stdin", "stdout"):
        stream = getattr(process, stream_name, None)
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass
    if not child.stderr_handle.closed:
        child.stderr_handle.close()


def _start_interactive_candidate(
    *,
    expected_source_sha: str,
    report_path: str,
    accuracy_preregistration_digest: str,
    parent_process_id: int,
    parent_process_start_identity: str,
    child_command_factory: Callable[
        [str, str, str, str, str, int, str],
        Sequence[str],
    ],
) -> SimpleNamespace:
    candidate = "fixed_row_tile_4_fast_adapter_candidate_v3"
    child_nonce = secrets.token_hex(32)
    command = child_command_factory(
        candidate,
        expected_source_sha,
        child_nonce,
        report_path,
        accuracy_preregistration_digest,
        parent_process_id,
        parent_process_start_identity,
    )
    if (
        not isinstance(command, Sequence)
        or isinstance(command, (str, bytes, bytearray))
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "interactive candidate child command is malformed"
        )
    stderr_handle = tempfile.TemporaryFile(
        mode="w+",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_handle,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    try:
        frame = _read_interactive_child_frame(
            process,
            timeout_seconds=_CHILD_TIMEOUT_SECONDS,
        )
        if (
            frame.get("protocol_version") != _CHILD_PROTOCOL_VERSION
            or frame.get("frame") != "preholdout"
            or not isinstance(frame.get("result"), Mapping)
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "interactive candidate preholdout frame drifted"
            )
        result = dict(frame["result"])  # type: ignore[arg-type]
        process_binding = result.get("process_binding")
        if (
            result.get("candidate") != candidate
            or result.get("child_nonce") != child_nonce
            or not isinstance(result.get("source_state"), Mapping)
            or result["source_state"].get("observed_source_sha")
            != expected_source_sha
            or not isinstance(process_binding, Mapping)
            or process_binding.get("child_process_id") != process.pid
            or process_binding.get("child_process_start_identity")
            != _process_start_identity(process.pid)
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "interactive candidate process, source, or nonce drifted"
            )
        return SimpleNamespace(
            process=process,
            stderr_handle=stderr_handle,
            child_nonce=child_nonce,
            preholdout_result=result,
        )
    except BaseException:
        _cleanup_interactive_candidate(
            SimpleNamespace(
                process=process,
                stderr_handle=stderr_handle,
            )
        )
        raise


def _finish_interactive_candidate(
    child: SimpleNamespace,
    *,
    authorization: Mapping[str, object],
) -> dict[str, object]:
    process = child.process
    deadline = time.monotonic() + _CHILD_TIMEOUT_SECONDS
    try:
        if process.stdin is None:
            raise RecurrentKernelDevelopmentScreenError(
                "interactive candidate stdin pipe is missing"
            )
        process.stdin.write(
            json.dumps(
                dict(authorization),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        process.stdin.flush()
        frame = _read_interactive_child_frame(
            process,
            timeout_seconds=max(
                deadline - time.monotonic(),
                sys.float_info.epsilon,
            ),
        )
        process.stdin.close()
        returncode, trailing_stdout = _wait_for_interactive_child_exit(
            process,
            timeout_seconds=max(
                deadline - time.monotonic(),
                sys.float_info.epsilon,
            ),
        )
        child.stderr_handle.seek(0)
        stderr = child.stderr_handle.read()
        if returncode != 0 or stderr or trailing_stdout:
            raise RecurrentKernelDevelopmentScreenError(
                "interactive candidate finalization emitted an error, "
                "stderr, or trailing stdout"
            )
        if (
            frame.get("protocol_version") != _CHILD_PROTOCOL_VERSION
            or frame.get("frame") != "completed"
            or not isinstance(frame.get("result"), Mapping)
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "interactive candidate final frame drifted"
            )
        result = dict(frame["result"])  # type: ignore[arg-type]
        if (
            result.get("candidate")
            != "fixed_row_tile_4_fast_adapter_candidate_v3"
            or result.get("child_nonce") != child.child_nonce
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "interactive candidate final identity drifted"
            )
        return result
    finally:
        _cleanup_interactive_candidate(child)


def _abort_interactive_candidate(
    child: SimpleNamespace,
    *,
    reason: str,
) -> dict[str, object]:
    process = child.process
    try:
        if process.stdin is not None:
            process.stdin.write(
                json.dumps(
                    {
                        "protocol_version": _CHILD_PROTOCOL_VERSION,
                        "decision": "abort",
                        "reason": reason,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            process.stdin.flush()
            process.stdin.close()
        returncode, trailing_stdout = _wait_for_interactive_child_exit(
            process,
            timeout_seconds=30,
        )
        child.stderr_handle.seek(0)
        stderr = child.stderr_handle.read()
        if returncode != 0 or stderr or trailing_stdout:
            raise RecurrentKernelDevelopmentScreenError(
                "interactive candidate abort emitted an error, stderr, or "
                "trailing stdout"
            )
        return _failed_child_result(
            candidate="fixed_row_tile_4_fast_adapter_candidate_v3",
            status="failed",
            child_nonce=child.child_nonce,
            failure="holdout_admission_denied",
            reason=reason,
            preholdout_result=child.preholdout_result,
            returncode=returncode,
            stderr_sha256=hashlib.sha256(b"").hexdigest(),
            stderr_tail="",
        )
    finally:
        _cleanup_interactive_candidate(child)


def _preholdout_admission_bundle(
    *,
    controls: Sequence[Mapping[str, object]],
    candidate: Mapping[str, object],
    baseline_confirmation: Mapping[str, object],
    source_state: Mapping[str, object],
    report_path: str,
    accuracy_preregistration_digest: str,
    parent_process_id: int,
    parent_process_start_identity: str,
    source_retirement_marker_path: Path,
    source_retirement_marker_sha256: str,
    capability_commitment_sha256: str,
) -> tuple[dict[str, object], bool, list[str]]:
    """Reconstruct every deciding pre-holdout dependency."""

    expected_controls = list(
        RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[:3]
    )
    reasons: list[str] = []
    if [control.get("candidate") for control in controls] != expected_controls:
        reasons.append("preholdout control order or identity drifted")
    control_gates = {
        str(control.get("candidate")): _candidate_integrity_gate(control)
        for control in controls
    }
    for name, gate in control_gates.items():
        if gate.get("passed") is not True:
            reasons.append(
                f"{name} integrity failed: "
                + "; ".join(str(value) for value in gate.get("reasons", ()))
            )
    confirmation_gate = _candidate_integrity_gate(
        baseline_confirmation
    )
    if confirmation_gate.get("passed") is not True:
        reasons.append(
            "closing baseline integrity failed: "
            + "; ".join(
                str(value)
                for value in confirmation_gate.get("reasons", ())
            )
        )
    candidate_gate = _candidate_preholdout_integrity_gate(
        candidate,
        reexecute_public_evidence=True,
    )
    if candidate_gate.get("passed") is not True:
        reasons.append(
            "candidate preholdout integrity failed: "
            + "; ".join(
                str(value) for value in candidate_gate.get("reasons", ())
            )
        )
    all_children = [*controls, candidate, baseline_confirmation]
    nonces = [child.get("child_nonce") for child in all_children]
    child_sources = [child.get("source_state") for child in all_children]
    child_runtimes = [child.get("runtime") for child in all_children]
    try:
        common_bindings = [
            _normalized_common_runtime_binding(
                child.get("fixed_batch_runtime_binding"),
                field="preholdout child fixed-batch runtime binding",
            )
            for child in all_children
        ]
        legacy_binding_digests = [
            _strict_sha256(
                child["legacy_fixed_batch_runtime_binding"].get(  # type: ignore[union-attr]
                    "exact_digest"
                ),
                field="preholdout legacy runtime digest",
            )
            for child in all_children
        ]
    except (
        KeyError,
        TypeError,
        RecurrentKernelDevelopmentScreenError,
    ) as exc:
        reasons.append(f"preholdout runtime reconstruction failed: {exc}")
        common_bindings = []
        legacy_binding_digests = []
    if (
        len(set(nonces)) != len(nonces)
        or any(not isinstance(nonce, str) for nonce in nonces)
        or any(child_source != source_state for child_source in child_sources)
        or not child_runtimes
        or any(runtime != child_runtimes[0] for runtime in child_runtimes[1:])
        or not common_bindings
        or any(
            binding != common_bindings[0]
            for binding in common_bindings[1:]
        )
        or not legacy_binding_digests
        or len(set(legacy_binding_digests)) != 1
    ):
        reasons.append(
            "preholdout nonce, source, or normalized runtime drifted"
        )
    baseline = controls[0] if controls else {}
    baseline_semantics = _candidate_semantic_fingerprint(baseline)
    if (
        baseline_semantics is None
        or _candidate_semantic_fingerprint(baseline_confirmation)
        != baseline_semantics
        or any(
            _candidate_semantic_fingerprint(control)
            != baseline_semantics
            for control in controls[1:]
        )
        or _candidate_semantic_fingerprint(candidate)
        != baseline_semantics
    ):
        reasons.append("preholdout behavior semantic fingerprints drifted")
    try:
        baseline_timing = baseline["timing"]  # type: ignore[index]
        confirmation_timing = baseline_confirmation["timing"]
        candidate_timing = candidate["timing"]
        baseline_scalar = min(
            int(baseline_timing["scalar"]["median_elapsed_ns"]),  # type: ignore[index]
            int(confirmation_timing["scalar"]["median_elapsed_ns"]),  # type: ignore[index]
        )
        baseline_batched = min(
            int(baseline_timing["batched"]["median_elapsed_ns"]),  # type: ignore[index]
            int(confirmation_timing["batched"]["median_elapsed_ns"]),  # type: ignore[index]
        )
        baseline_rss = min(
            int(baseline["peak_rss_bytes"]),  # type: ignore[index]
            int(baseline_confirmation["peak_rss_bytes"]),
        )
        candidate_scalar = int(
            candidate_timing["scalar"]["median_elapsed_ns"]  # type: ignore[index]
        )
        candidate_batched = int(
            candidate_timing["batched"]["median_elapsed_ns"]  # type: ignore[index]
        )
        speed_metrics = {
            "own_path_speedup": 1.0
            - candidate_batched / candidate_scalar,
            "absolute_speedup_over_legacy_scalar": 1.0
            - candidate_batched / baseline_scalar,
            "absolute_speedup_over_legacy_batched": 1.0
            - candidate_batched / baseline_batched,
            "paired_batched_wins": int(
                candidate_timing["paired_batched_wins"]  # type: ignore[index]
            ),
            "peak_rss_ratio": int(candidate["peak_rss_bytes"])
            / baseline_rss,
        }
        if (
            speed_metrics["own_path_speedup"]
            < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP
            or speed_metrics["absolute_speedup_over_legacy_scalar"]
            < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP
            or speed_metrics["absolute_speedup_over_legacy_batched"]
            < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_SPEEDUP
            or speed_metrics["paired_batched_wins"]
            < RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MINIMUM_PAIRED_WINS
            or speed_metrics["peak_rss_ratio"]
            > RECURRENT_KERNEL_DEVELOPMENT_SCREEN_MAXIMUM_RSS_RATIO
        ):
            reasons.append("preholdout speed, paired-win, or RSS gate failed")
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        speed_metrics = {}
        reasons.append(f"preholdout speed reconstruction failed: {exc}")
    bundle: dict[str, object] = {
        "schema_version": _PREHOLDOUT_EVIDENCE_VERSION,
        "source_sha": source_state.get("observed_source_sha"),
        "source_module_bundle_sha256": source_state.get(
            "source_module_bundle_sha256"
        ),
        "method_document_sha256": source_state.get(
            "accuracy_method_document_sha256"
        ),
        "report_path": report_path,
        "accuracy_preregistration_digest": (
            accuracy_preregistration_digest
        ),
        "parent_process_id": parent_process_id,
        "parent_process_start_identity": (
            parent_process_start_identity
        ),
        "candidate_process_binding": candidate.get("process_binding"),
        "ordered_control_digests": [
            control.get("child_exact_digest") for control in controls
        ],
        "candidate_preholdout_digest": candidate.get(
            "preholdout_exact_digest"
        ),
        "baseline_confirmation_digest": baseline_confirmation.get(
            "child_exact_digest"
        ),
        "source_retirement_marker_path": str(
            source_retirement_marker_path
        ),
        "source_retirement_marker_sha256": (
            source_retirement_marker_sha256
        ),
        "capability_commitment_sha256": (
            capability_commitment_sha256
        ),
        "control_gates": control_gates,
        "candidate_gate": candidate_gate,
        "baseline_confirmation_gate": confirmation_gate,
        "speed_metrics": speed_metrics,
        "admission_authorized": not reasons,
        "reasons": reasons,
    }
    bundle["exact_digest"] = stable_payload_digest(bundle)
    return bundle, not reasons, reasons


def _development_screen_contract() -> dict[str, object]:
    return {
        "candidate_order": list(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES),
        "sole_selectable_shared_trainable_forward_candidate": (
            "fixed_row_tile_4_fast_adapter_candidate_v3"
        ),
        "nonselectable_kernel_controls": [
            "per_row_bmm_fast_adapter_control_v3",
            "fixed_row_tile_4_legacy_adapter_control_v3",
        ],
        "source_default_numeric_kernel_version": (
            RECURRENT_NUMERIC_KERNEL_VERSION
        ),
        "source_default_observation_projection_version": (
            ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION
        ),
        "source_bound_module_paths": dict(_SOURCE_BOUND_MODULE_PATHS),
        "accuracy_method_document_path": _ACCURACY_METHOD_DOCUMENT_PATH,
        "interactive_child_protocol_version": _CHILD_PROTOCOL_VERSION,
        "preholdout_evidence_version": _PREHOLDOUT_EVIDENCE_VERSION,
        "source_retirement_marker_before_child_1": True,
        "holdout_retirement_marker_before_capability_send": True,
        "capability_precommitted_before_child_1": True,
        "capability_issued_only_after_closing_baseline": True,
        "raw_capability_persisted": False,
        "nonselectable_accuracy_sentinel_version": (
            _NONSELECTABLE_ACCURACY_SENTINEL_VERSION
        ),
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
        "d04_numeric_contract": {
            "relative_tolerance": _D04_ORACLE_RELATIVE_TOLERANCE,
            "absolute_tolerance": _D04_ORACLE_ABSOLUTE_TOLERANCE,
            "maximum_tolerance_ratio": 1.0,
        },
        "legacy_rounding_distance_is_selection_gate": False,
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
    report_path: Path,
    child_command_factory: Callable[
        [str, str, str, str, str, int, str],
        Sequence[str],
    ],
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run the fixed candidate set in isolated children and select at most one."""

    from evolution_sim.mind import recurrent_accuracy_screen

    resolved_report = validate_development_screen_report_path(report_path)
    source_state = _git_source_state(expected_source_sha)
    accuracy_preregistration = _accuracy_preregistration_for_source(
        source_state
    )
    accuracy_preregistration_digest = _strict_sha256(
        accuracy_preregistration.get("exact_digest"),
        field="accuracy preregistration digest",
    )
    parent_process_id = os.getpid()
    parent_process_start_identity = _process_start_identity(parent_process_id)
    private_capability_preimage = (
        recurrent_accuracy_screen.generate_holdout_capability_preimage()
    )
    capability_commitment_sha256 = (
        recurrent_accuracy_screen.holdout_capability_commitment(
            private_capability_preimage
        )
    )
    (
        source_retirement_marker_path,
        source_retirement_marker,
        source_retirement_marker_sha256,
    ) = _write_source_retirement_marker(
        report_path=resolved_report,
        source_state=source_state,
        accuracy_preregistration_digest=(
            accuracy_preregistration_digest
        ),
        parent_process_id=parent_process_id,
        parent_process_start_identity=(
            parent_process_start_identity
        ),
        capability_commitment_sha256=(
            capability_commitment_sha256
        ),
    )
    emit = progress if progress is not None else (lambda _message: None)
    results: list[dict[str, object]] = []
    for candidate in RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[:3]:
        emit(f"recurrent_kernel_screen_start candidate={candidate}")
        result = _run_child_candidate(
            candidate=candidate,
            expected_source_sha=expected_source_sha,
            report_path=str(resolved_report),
            accuracy_preregistration_digest=(
                accuracy_preregistration_digest
            ),
            parent_process_id=parent_process_id,
            parent_process_start_identity=(
                parent_process_start_identity
            ),
            child_command_factory=child_command_factory,
        )
        results.append(result)
        emit(
            "recurrent_kernel_screen_finish "
            f"candidate={candidate} status={result['status']}"
        )
    selectable_name = "fixed_row_tile_4_fast_adapter_candidate_v3"
    emit(f"recurrent_kernel_screen_start candidate={selectable_name}")
    interactive_child = _start_interactive_candidate(
        expected_source_sha=expected_source_sha,
        report_path=str(resolved_report),
        accuracy_preregistration_digest=(
            accuracy_preregistration_digest
        ),
        parent_process_id=parent_process_id,
        parent_process_start_identity=parent_process_start_identity,
        child_command_factory=child_command_factory,
    )
    candidate_preholdout = interactive_child.preholdout_result
    baseline_name = RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0]
    try:
        emit(
            f"recurrent_kernel_screen_start candidate={baseline_name} "
            "confirmation=true"
        )
        baseline_confirmation = _run_child_candidate(
            candidate=baseline_name,
            expected_source_sha=expected_source_sha,
            report_path=str(resolved_report),
            accuracy_preregistration_digest=(
                accuracy_preregistration_digest
            ),
            parent_process_id=parent_process_id,
            parent_process_start_identity=(
                parent_process_start_identity
            ),
            child_command_factory=child_command_factory,
        )
        emit(
            "recurrent_kernel_screen_finish "
            f"candidate={baseline_name} confirmation=true "
            f"status={baseline_confirmation['status']}"
        )
        (
            preholdout_evidence,
            holdout_admission_authorized,
            admission_reasons,
        ) = _preholdout_admission_bundle(
            controls=results,
            candidate=candidate_preholdout,
            baseline_confirmation=baseline_confirmation,
            source_state=source_state,
            report_path=str(resolved_report),
            accuracy_preregistration_digest=(
                accuracy_preregistration_digest
            ),
            parent_process_id=parent_process_id,
            parent_process_start_identity=(
                parent_process_start_identity
            ),
            source_retirement_marker_path=(
                source_retirement_marker_path
            ),
            source_retirement_marker_sha256=(
                source_retirement_marker_sha256
            ),
            capability_commitment_sha256=(
                capability_commitment_sha256
            ),
        )
        preholdout_evidence_digest = _strict_sha256(
            preholdout_evidence.get("exact_digest"),
            field="preholdout evidence digest",
        )
        candidate_process_binding = candidate_preholdout.get(
            "process_binding"
        )
        if not isinstance(candidate_process_binding, Mapping):
            raise RecurrentKernelDevelopmentScreenError(
                "candidate process binding is absent before retirement"
            )
        baseline_confirmation_digest = _strict_sha256(
            baseline_confirmation.get("child_exact_digest"),
            field="baseline confirmation digest",
        )
        holdout_retirement_marker_path: Path | None = None
        holdout_retirement_marker: dict[str, object] | None = None
        holdout_retirement_marker_sha256: str | None = None
        if holdout_admission_authorized:
            (
                holdout_retirement_marker_path,
                holdout_retirement_marker,
                holdout_retirement_marker_sha256,
            ) = _write_holdout_retirement_marker(
                report_path=resolved_report,
                preholdout_evidence_digest=preholdout_evidence_digest,
                candidate_process_binding=candidate_process_binding,
                source_retirement_marker_path=(
                    source_retirement_marker_path
                ),
                source_retirement_marker_sha256=(
                    source_retirement_marker_sha256
                ),
                capability_commitment_sha256=(
                    capability_commitment_sha256
                ),
                baseline_confirmation_digest=(
                    baseline_confirmation_digest
                ),
            )
            capability = recurrent_accuracy_screen.issue_holdout_capability(
                preimage=private_capability_preimage,
                capability_commitment_sha256=(
                    capability_commitment_sha256
                ),
                preholdout_evidence_digest=(
                    preholdout_evidence_digest
                ),
                source_retirement_marker_sha256=(
                    source_retirement_marker_sha256
                ),
                holdout_retirement_marker_sha256=(
                    holdout_retirement_marker_sha256
                ),
                candidate_identity=selectable_name,
                closing_baseline_validated=True,
                source_retirement_marker_exists=(
                    source_retirement_marker_path.is_file()
                ),
                holdout_retirement_marker_exists=(
                    holdout_retirement_marker_path.is_file()
                ),
            )
            if (
                recurrent_accuracy_screen.holdout_capability_commitment(
                    capability.token
                )
                != capability_commitment_sha256
            ):
                raise RecurrentKernelDevelopmentScreenError(
                    "late-issued holdout capability does not open the "
                    "pre-child commitment"
                )
            authorization: dict[str, object] = {
                "protocol_version": _CHILD_PROTOCOL_VERSION,
                "decision": "consume_holdout",
                "capability_token": capability.token,
                "capability_commitment_sha256": (
                    capability_commitment_sha256
                ),
                "source_sha": source_state["observed_source_sha"],
                "source_module_bundle_sha256": source_state[
                    "source_module_bundle_sha256"
                ],
                "method_document_sha256": source_state[
                    "accuracy_method_document_sha256"
                ],
                "report_path": str(resolved_report),
                "accuracy_preregistration_digest": (
                    accuracy_preregistration_digest
                ),
                "preholdout_evidence_digest": (
                    preholdout_evidence_digest
                ),
                "closing_baseline_evidence_digest": (
                    baseline_confirmation_digest
                ),
                "parent_process_id": candidate_process_binding[
                    "parent_process_id"
                ],
                "parent_process_start_identity": (
                    candidate_process_binding[
                        "parent_process_start_identity"
                    ]
                ),
                "child_process_id": candidate_process_binding[
                    "child_process_id"
                ],
                "child_process_start_identity": (
                    candidate_process_binding[
                        "child_process_start_identity"
                    ]
                ),
                "child_nonce": candidate_process_binding[
                    "child_nonce"
                ],
                "source_retirement_marker_path": str(
                    source_retirement_marker_path
                ),
                "source_retirement_marker_sha256": (
                    source_retirement_marker_sha256
                ),
                "retirement_marker_path": str(
                    holdout_retirement_marker_path
                ),
                "retirement_marker_sha256": (
                    holdout_retirement_marker_sha256
                ),
            }
            authorization["exact_digest"] = stable_payload_digest(
                authorization
            )
            candidate_result = _finish_interactive_candidate(
                interactive_child,
                authorization=authorization,
            )
        else:
            candidate_result = _abort_interactive_candidate(
                interactive_child,
                reason="; ".join(admission_reasons),
            )
    except BaseException:
        if interactive_child.process.poll() is None:
            _abort_interactive_candidate(
                interactive_child,
                reason="parent_failed_before_holdout_completion",
            )
        raise
    results.append(candidate_result)
    emit(
        "recurrent_kernel_screen_finish "
        f"candidate={selectable_name} status={candidate_result['status']}"
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
    candidate_accuracy: Mapping[str, object] = {}
    if candidate_result.get("status") == "completed":
        completed_accuracy = candidate_result.get("accuracy_evidence")
        if not isinstance(completed_accuracy, Mapping):
            raise RecurrentKernelDevelopmentScreenError(
                "completed candidate accuracy evidence is absent"
            )
        candidate_accuracy = completed_accuracy
    lifecycle = {
        "qualification_run": False,
        "training_run": False,
        "training_artifact_created": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "phase_a_output_created": False,
        "source_attempt_retired": True,
        "holdout_attempt_retired": (
            holdout_retirement_marker is not None
        ),
    }
    report: dict[str, object] = {
        "schema_version": RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SCHEMA_VERSION,
        "status": "completed",
        "development_only": True,
        "launch_authorized": False,
        "authority_evidence_eligible": False,
        "scientific_result": False,
        "source": final_source_state,
        "contract": {
            "screen": _development_screen_contract(),
            "accuracy_preregistration": accuracy_preregistration,
            "preholdout_evidence": preholdout_evidence,
            "source_retirement_marker": source_retirement_marker,
            "source_retirement_marker_path": str(
                source_retirement_marker_path
            ),
            "source_retirement_marker_sha256": (
                source_retirement_marker_sha256
            ),
            "holdout_retirement_marker": (
                holdout_retirement_marker
            ),
            "holdout_retirement_marker_path": (
                None
                if holdout_retirement_marker_path is None
                else str(holdout_retirement_marker_path)
            ),
            "holdout_retirement_marker_sha256": (
                holdout_retirement_marker_sha256
            ),
            "capability_commitment_sha256": (
                capability_commitment_sha256
            ),
        },
        "runtime": {
            "common_child_runtime": (
                results[0].get("runtime") if results else None
            ),
            "child_count": 5,
            "isolated_children": True,
            "candidate_interactive_holdout_admission": True,
        },
        "children": results,
        "baseline_confirmation": baseline_confirmation,
        "corpus": accuracy_preregistration.get("corpus_contract"),
        "accuracy": {
            "public_forward": candidate_accuracy.get(
                "public_forward"
            ),
            "holdout_forward": candidate_accuracy.get(
                "holdout_forward"
            ),
        },
        "gradients": candidate_accuracy.get("gradients"),
        "holdout_receipt": candidate_accuracy.get("holdout_receipt"),
        "selection": selection,
        "lifecycle": lifecycle,
    }
    report["exact_digest"] = stable_payload_digest(report)
    return report


def validate_development_screen_report(
    report: Mapping[str, object],
) -> None:
    """Reconstruct the non-authoritative report before atomic publication."""

    from evolution_sim.mind import recurrent_accuracy_screen

    expected_keys = {
        "schema_version",
        "status",
        "development_only",
        "launch_authorized",
        "authority_evidence_eligible",
        "scientific_result",
        "source",
        "contract",
        "runtime",
        "children",
        "baseline_confirmation",
        "corpus",
        "accuracy",
        "gradients",
        "holdout_receipt",
        "selection",
        "lifecycle",
        "exact_digest",
    }
    if set(report) != expected_keys:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen report root schema drifted"
        )
    if (
        report.get("schema_version")
        != RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SCHEMA_VERSION
        or report.get("status") != "completed"
        or report.get("development_only") is not True
        or report.get("launch_authorized") is not False
        or report.get("authority_evidence_eligible") is not False
        or report.get("scientific_result") is not False
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen lifecycle flags drifted"
        )
    lifecycle = report.get("lifecycle")
    expected_lifecycle_keys = {
        "qualification_run",
        "training_run",
        "training_artifact_created",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "promotion_authorized",
        "phase_a_output_created",
        "source_attempt_retired",
        "holdout_attempt_retired",
    }
    if (
        not isinstance(lifecycle, Mapping)
        or set(lifecycle) != expected_lifecycle_keys
        or any(
            lifecycle.get(field) is not False
            for field in (
                "qualification_run",
                "training_run",
                "training_artifact_created",
                "runtime_artifact_created",
                "runtime_action_selection_changed",
                "promotion_authorized",
                "phase_a_output_created",
            )
        )
        or lifecycle.get("source_attempt_retired") is not True
        or not isinstance(lifecycle.get("holdout_attempt_retired"), bool)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen nested lifecycle drifted"
        )
    source = _validated_source_state(
        report.get("source"),
        field="development screen",
    )
    contract = report.get("contract")
    expected_contract_keys = {
        "screen",
        "accuracy_preregistration",
        "preholdout_evidence",
        "source_retirement_marker",
        "source_retirement_marker_path",
        "source_retirement_marker_sha256",
        "holdout_retirement_marker",
        "holdout_retirement_marker_path",
        "holdout_retirement_marker_sha256",
        "capability_commitment_sha256",
    }
    if (
        not isinstance(contract, Mapping)
        or set(contract) != expected_contract_keys
        or contract.get("screen") != _development_screen_contract()
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen declared contract drifted"
        )
    preregistration = contract.get("accuracy_preregistration")
    if not isinstance(preregistration, dict):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen accuracy preregistration is missing"
        )
    recurrent_accuracy_screen.validate_accuracy_preregistration(
        preregistration
    )
    if preregistration != _accuracy_preregistration_for_source(source):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen preregistration does not match source"
        )
    if report.get("corpus") != preregistration.get("corpus_contract"):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen corpus differs from preregistration"
        )
    source_marker = contract.get("source_retirement_marker")
    source_marker_path = contract.get("source_retirement_marker_path")
    source_marker_sha256 = _strict_sha256(
        contract.get("source_retirement_marker_sha256"),
        field="source retirement marker SHA256",
    )
    if (
        not isinstance(source_marker, Mapping)
        or not isinstance(source_marker_path, str)
        or not Path(source_marker_path).is_file()
        or hashlib.sha256(Path(source_marker_path).read_bytes()).hexdigest()
        != source_marker_sha256
        or json.loads(Path(source_marker_path).read_text(encoding="utf-8"))
        != source_marker
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen source retirement marker drifted"
        )
    source_marker_parent_id = _strict_positive_int(
        source_marker.get("parent_process_id"),
        field="source retirement parent process ID",
    )
    source_marker_parent_start = source_marker.get(
        "parent_process_start_identity"
    )
    source_marker_report_path = source_marker.get("report_path")
    if (
        source_marker.get("source_retired") is not True
        or source_marker.get("no_retry_for_exact_source") is not True
        or not isinstance(source_marker_parent_start, str)
        or not source_marker_parent_start
        or not isinstance(source_marker_report_path, str)
        or not Path(source_marker_report_path).is_absolute()
        or Path(source_marker_report_path).name
        != _CANONICAL_DEVELOPMENT_SCREEN_REPORT_NAME
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "source retirement process or report binding drifted"
        )
    expected_source_marker = recurrent_accuracy_screen.source_retirement_marker(
        source_sha=str(source["observed_source_sha"]),
        source_module_digest=stable_payload_digest(
            _source_bound_path_sha256()
        ),
        method_document_digest=str(
            source["accuracy_method_document_sha256"]
        ),
        preregistration_digest=str(preregistration["exact_digest"]),
        report_path=str(
            Path(source_marker_path).parent
            / _CANONICAL_DEVELOPMENT_SCREEN_REPORT_NAME
        ),
        parent_process_id=source_marker_parent_id,
        parent_process_start_identity=source_marker_parent_start,
        capability_commitment_sha256=str(
            contract["capability_commitment_sha256"]
        ),
    )
    if source_marker != expected_source_marker:
        raise RecurrentKernelDevelopmentScreenError(
            "source retirement marker does not reconstruct"
        )
    results = report.get("children")
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
    if [
        result.get("candidate") for result in typed_results
    ] != list(RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen child order or identity drifted"
        )
    for expected_name, child in zip(
        RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[:3],
        typed_results[:3],
        strict=True,
    ):
        if child.get("status") == "completed":
            _validate_candidate_identity(child)
        elif child.get("status") == "failed":
            _validate_failed_child_envelope(
                child,
                expected_candidate=expected_name,
                allow_preholdout_abort=False,
            )
        else:
            raise RecurrentKernelDevelopmentScreenError(
                "control child status drifted"
            )
    if confirmation.get("candidate") != (
        RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0]
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "closing baseline identity drifted"
        )
    if confirmation.get("status") == "completed":
        _validate_candidate_identity(confirmation)
    elif confirmation.get("status") == "failed":
        _validate_failed_child_envelope(
            confirmation,
            expected_candidate=(
                RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0]
            ),
            allow_preholdout_abort=False,
        )
    else:
        raise RecurrentKernelDevelopmentScreenError(
            "closing baseline status drifted"
        )
    completed_children = [
        result
        for result in (*typed_results, confirmation)
        if result.get("status") == "completed"
    ]
    if any(result.get("source_state") != source for result in completed_children):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen parent and child source bindings differ"
        )
    candidate_result = typed_results[-1]
    if candidate_result.get("status") == "completed":
        _validate_candidate_identity(candidate_result)
        candidate_preholdout = dict(candidate_result)
        candidate_preholdout.pop("child_exact_digest", None)
        candidate_preholdout.pop("holdout_authorization_digest", None)
        candidate_preholdout["status"] = "preholdout_completed"
        final_accuracy = candidate_preholdout.get("accuracy_evidence")
        if not isinstance(final_accuracy, Mapping):
            raise RecurrentKernelDevelopmentScreenError(
                "completed candidate accuracy evidence is malformed"
            )
        candidate_preholdout["accuracy_evidence"] = {
            **_preholdout_accuracy_payload(final_accuracy),
            "preholdout_candidate_exact_digest": final_accuracy.get(
                "preholdout_candidate_exact_digest"
            ),
        }
        _validate_selectable_accuracy_evidence(
            candidate_result,
            reexecute_public_evidence=False,
        )
    elif (
        candidate_result.get("status") == "failed"
        and isinstance(candidate_result.get("preholdout_result"), Mapping)
    ):
        _validate_failed_child_envelope(
            candidate_result,
            expected_candidate=(
                "fixed_row_tile_4_fast_adapter_candidate_v3"
            ),
            allow_preholdout_abort=True,
        )
        candidate_preholdout = dict(
            candidate_result["preholdout_result"]  # type: ignore[arg-type]
        )
        _validate_candidate_identity(
            candidate_preholdout,
            expected_status="preholdout_completed",
            digest_field="preholdout_exact_digest",
        )
    else:
        raise RecurrentKernelDevelopmentScreenError(
            "selectable candidate lacks reconstructible preholdout evidence"
        )
    preholdout = contract.get("preholdout_evidence")
    if not isinstance(preholdout, Mapping):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen preholdout evidence is missing"
        )
    (
        reconstructed_preholdout,
        admission_authorized,
        admission_reasons,
    ) = _preholdout_admission_bundle(
        controls=typed_results[:3],
        candidate=candidate_preholdout,
        baseline_confirmation=confirmation,
        source_state=source,
        report_path=str(source_marker["report_path"]),
        accuracy_preregistration_digest=str(
            preregistration["exact_digest"]
        ),
        parent_process_id=int(source_marker["parent_process_id"]),
        parent_process_start_identity=source_marker_parent_start,
        source_retirement_marker_path=Path(source_marker_path),
        source_retirement_marker_sha256=source_marker_sha256,
        capability_commitment_sha256=str(
            contract["capability_commitment_sha256"]
        ),
    )
    if preholdout != reconstructed_preholdout:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen preholdout evidence does not reconstruct"
        )
    if admission_authorized:
        if candidate_result.get("status") != "completed":
            raise RecurrentKernelDevelopmentScreenError(
                "authorized holdout admission lacks a completed candidate "
                "terminal state"
            )
    elif (
        candidate_result.get("status") != "failed"
        or candidate_result.get("failure") != "holdout_admission_denied"
        or candidate_result.get("reason") != "; ".join(admission_reasons)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "denied holdout admission lacks its exact failed candidate "
            "terminal state"
        )
    holdout_marker = contract.get("holdout_retirement_marker")
    holdout_marker_path = contract.get("holdout_retirement_marker_path")
    holdout_marker_sha256 = contract.get(
        "holdout_retirement_marker_sha256"
    )
    if admission_authorized:
        if (
            not isinstance(holdout_marker, Mapping)
            or not isinstance(holdout_marker_path, str)
            or not Path(holdout_marker_path).is_file()
            or _strict_sha256(
                holdout_marker_sha256,
                field="holdout retirement marker SHA256",
            )
            != hashlib.sha256(
                Path(holdout_marker_path).read_bytes()
            ).hexdigest()
            or json.loads(
                Path(holdout_marker_path).read_text(encoding="utf-8")
            )
            != holdout_marker
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout retirement marker is absent or changed"
            )
        holdout_marker_parent_id = _strict_positive_int(
            holdout_marker.get("parent_process_id"),
            field="holdout retirement parent process ID",
        )
        holdout_marker_child_id = _strict_positive_int(
            holdout_marker.get("child_process_id"),
            field="holdout retirement child process ID",
        )
        holdout_marker_parent_start = holdout_marker.get(
            "parent_process_start_identity"
        )
        holdout_marker_child_start = holdout_marker.get(
            "child_process_start_identity"
        )
        holdout_marker_child_nonce = _strict_sha256(
            holdout_marker.get("child_nonce"),
            field="holdout retirement child nonce",
        )
        _strict_sha256(
            holdout_marker.get("exact_digest"),
            field="holdout retirement exact digest",
        )
        if (
            holdout_marker.get("holdout_retired") is not True
            or holdout_marker.get("no_retry_for_exact_source") is not True
            or not isinstance(holdout_marker_parent_start, str)
            or not holdout_marker_parent_start
            or not isinstance(holdout_marker_child_start, str)
            or not holdout_marker_child_start
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout retirement process or lifecycle binding drifted"
            )
        candidate_process_binding = candidate_preholdout.get(
            "process_binding"
        )
        if not isinstance(candidate_process_binding, Mapping):
            raise RecurrentKernelDevelopmentScreenError(
                "candidate process binding is malformed"
            )
        candidate_parent_id = _strict_positive_int(
            candidate_process_binding.get("parent_process_id"),
            field="candidate parent process ID",
        )
        candidate_child_id = _strict_positive_int(
            candidate_process_binding.get("child_process_id"),
            field="candidate child process ID",
        )
        candidate_parent_start = candidate_process_binding.get(
            "parent_process_start_identity"
        )
        candidate_child_start = candidate_process_binding.get(
            "child_process_start_identity"
        )
        candidate_child_nonce = _strict_sha256(
            candidate_preholdout.get("child_nonce"),
            field="candidate child nonce",
        )
        if (
            not isinstance(candidate_parent_start, str)
            or not candidate_parent_start
            or not isinstance(candidate_child_start, str)
            or not candidate_child_start
            or holdout_marker_parent_id != candidate_parent_id
            or holdout_marker_child_id != candidate_child_id
            or holdout_marker_parent_start != candidate_parent_start
            or holdout_marker_child_start != candidate_child_start
            or holdout_marker_child_nonce != candidate_child_nonce
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout retirement marker does not bind the candidate process"
            )
        expected_holdout_marker = (
            recurrent_accuracy_screen.holdout_retirement_marker(
                source_retirement_marker_sha256=source_marker_sha256,
                source_retirement_marker_exact_digest=str(
                    source_marker["exact_digest"]
                ),
                preholdout_evidence_digest=str(
                    preholdout["exact_digest"]
                ),
                closing_baseline_evidence_digest=str(
                    confirmation["child_exact_digest"]
                ),
                parent_process_id=holdout_marker_parent_id,
                parent_process_start_identity=holdout_marker_parent_start,
                child_process_id=holdout_marker_child_id,
                child_process_start_identity=holdout_marker_child_start,
                child_nonce=holdout_marker_child_nonce,
                capability_commitment_sha256=str(
                    contract["capability_commitment_sha256"]
                ),
            )
        )
        if holdout_marker != expected_holdout_marker:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout retirement marker does not reconstruct"
            )
        if lifecycle.get("holdout_attempt_retired") is not True:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout lifecycle does not acknowledge its marker"
            )
    elif (
        holdout_marker is not None
        or holdout_marker_path is not None
        or holdout_marker_sha256 is not None
        or lifecycle.get("holdout_attempt_retired") is not False
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "negative preholdout screen falsely claims holdout retirement"
        )
    recomputed_selection = select_development_candidate(
        typed_results,
        baseline_confirmation=confirmation,
    )
    if report.get("selection") != recomputed_selection:
        raise RecurrentKernelDevelopmentScreenError(
            "development screen selection does not reconstruct"
        )
    if candidate_result.get("status") == "completed":
        candidate_accuracy = candidate_result.get("accuracy_evidence")
        if not isinstance(candidate_accuracy, Mapping):
            raise RecurrentKernelDevelopmentScreenError(
                "candidate accuracy evidence is malformed"
            )
        expected_accuracy = {
            "public_forward": candidate_accuracy.get("public_forward"),
            "holdout_forward": candidate_accuracy.get("holdout_forward"),
        }
        expected_gradients = candidate_accuracy.get("gradients")
        expected_holdout_receipt = candidate_accuracy.get(
            "holdout_receipt"
        )
    else:
        expected_accuracy = {
            "public_forward": None,
            "holdout_forward": None,
        }
        expected_gradients = None
        expected_holdout_receipt = None
    if (
        report.get("accuracy") != expected_accuracy
        or report.get("gradients") != expected_gradients
        or report.get("holdout_receipt")
        != expected_holdout_receipt
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "root accuracy evidence differs from the candidate child"
        )
    runtime = report.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "common_child_runtime",
            "child_count",
            "isolated_children",
            "candidate_interactive_holdout_admission",
        }
        or runtime.get("child_count") != 5
        or runtime.get("isolated_children") is not True
        or runtime.get("candidate_interactive_holdout_admission") is not True
        or runtime.get("common_child_runtime")
        != typed_results[0].get("runtime")
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen root runtime binding drifted"
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


def validate_development_screen_report_path(
    path: Path,
    *,
    allow_source_retirement_marker: bool = False,
    allow_retirement_marker: bool = False,
) -> Path:
    if (
        path.suffix != ".json"
        or path.name != _CANONICAL_DEVELOPMENT_SCREEN_REPORT_NAME
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "development screen report must use the one canonical no-clobber "
            f"name {_CANONICAL_DEVELOPMENT_SCREEN_REPORT_NAME!r}"
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
    marker = (
        resolved.parent
        / _CANONICAL_HOLDOUT_RETIREMENT_MARKER_NAME
    )
    if marker.exists() and not allow_retirement_marker:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout retirement marker already exists; this exact-source "
            "screen cannot be rerun"
        )
    source_marker = (
        resolved.parent
        / _CANONICAL_SOURCE_RETIREMENT_MARKER_NAME
    )
    if source_marker.exists() and not allow_source_retirement_marker:
        raise RecurrentKernelDevelopmentScreenError(
            "source retirement marker already exists; this exact-source "
            "screen cannot be rerun"
        )
    return resolved


def write_development_screen_report(
    path: Path,
    report: Mapping[str, object],
) -> None:
    validate_development_screen_report(report)
    resolved = validate_development_screen_report_path(
        path,
        allow_source_retirement_marker=True,
        allow_retirement_marker=True,
    )
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
        reopened = json.loads(temporary_path.read_text(encoding="utf-8"))
        if reopened != dict(report):
            raise RecurrentKernelDevelopmentScreenError(
                "reopened pending development report differs from memory"
            )
        validate_development_screen_report(reopened)
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
