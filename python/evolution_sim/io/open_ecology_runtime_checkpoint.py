from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from functools import reduce
from operator import mul
from random import Random
import re
from typing import Mapping

import torch
from torch import Tensor

from evolution_sim.env.fields import EnvironmentFieldMaps
from evolution_sim.env.runtime.bootstrap import StaticTopology
from evolution_sim.env.runtime.reproduction import ReproductiveGroupRecord
from evolution_sim.env.runtime.signals import (
    CommunicationReceiverProjection,
    SignalEmission,
    SignalFieldState,
)
from evolution_sim.env.runtime.state import (
    Agent,
    BioticFieldState,
    CarcassDeposit,
    FreshKillDeposit,
    Tile,
    TrophicProfile,
)
from evolution_sim.env.world import SimulationWorld
from evolution_sim.genome.schema import Genome, ReproductiveGenome
from evolution_sim.io.open_ecology_checkpoint import VersionedCheckpointState
from evolution_sim.io.open_ecology_rotating_writer import (
    validate_open_ecology_evidence_continuation_state,
)
from evolution_sim.mind.recurrent_actor_critic import PreviousPublicFeedbackInput
from evolution_sim.mind.recurrent_genome import RECURRENT_CONTROLLER_GENOME_SIZE
from evolution_sim.mind.recurrent_genome_population import (
    RECURRENT_GENOME_MIND_METADATA_SCHEMA_VERSION,
    RecurrentGenomePopulationManager,
)
from evolution_sim.mind.recurrent_policy import (
    RECURRENT_PUBLIC_HISTORY_PREFIX_SCHEMA_VERSION,
    DeterministicPublicRecurrentPolicy,
    recurrent_model_state_sha256,
    validate_public_recurrent_history_prefix,
)


OPEN_ECOLOGY_RUNTIME_BINDING_SCHEMA_VERSION = (
    "open_ecology_runtime_checkpoint_binding_v2"
)
OPEN_ECOLOGY_WORLD_STATE_SCHEMA_VERSION = "open_ecology_simulation_world_state_v1"
OPEN_ECOLOGY_ENVIRONMENT_RNG_SCHEMA_VERSION = "open_ecology_python_random_state_v1"
OPEN_ECOLOGY_RECURRENT_POLICY_SCHEMA_VERSION = (
    "open_ecology_frozen_recurrent_policy_state_v1"
)
OPEN_ECOLOGY_PUBLIC_FEEDBACK_SCHEMA_VERSION = (
    "open_ecology_public_feedback_history_state_v1"
)
OPEN_ECOLOGY_SAMPLING_RNG_SCHEMA_VERSION = "open_ecology_torch_generator_state_v1"
OPEN_ECOLOGY_GENOME_POPULATION_SCHEMA_VERSION = (
    "open_ecology_recurrent_genome_population_adapter_v1"
)
OPEN_ECOLOGY_EVIDENCE_WRITER_ADAPTER_SCHEMA_VERSION = (
    "open_ecology_evidence_writer_continuation_adapter_v1"
)
OPEN_ECOLOGY_RUNTIME_DEFAULT_MAX_STATE_BYTES = 128 * 1024 * 1024
OPEN_ECOLOGY_RUNTIME_DEFAULT_MAX_HISTORY_RECORDS = 1_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_WORLD_EXCLUDED_ATTRIBUTES = frozenset({"config", "rng", "policy", "trajectory_sink"})
_PERSISTENT_RUNNER_STATE_ATTRIBUTE = "_open_ecology_persistent_runner_state"
_PERSISTENT_RUNNER_STATE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_persistent_island_runner_state_v1"
)
_PERSISTENT_RUNNER_STATE_KEYS = {
    "schema_version",
    "task_sha256",
    "model_state_sha256",
    "observed_tick",
    "completed_world_tick",
    "extinct",
    "extinction_tick",
    "episode_reset_count",
    "world_replacement_count",
    "interval",
    "summaries",
    "milestones",
    "event_counts",
    "writer_initial_event_count",
    "next_event_index",
    "last_evidence_tick",
    "evidence_continuation_sha256",
    "evidence_finished",
    "evidence_aborted",
    "runner_state_sha256",
}
_ALLOWED_DATACLASS_TYPES = {
    f"{value.__module__}.{value.__qualname__}": value
    for value in (
        Agent,
        BioticFieldState,
        CarcassDeposit,
        CommunicationReceiverProjection,
        EnvironmentFieldMaps,
        FreshKillDeposit,
        Genome,
        ReproductiveGenome,
        ReproductiveGroupRecord,
        SignalEmission,
        SignalFieldState,
        StaticTopology,
        Tile,
        TrophicProfile,
    )
}


class OpenEcologyRuntimeCheckpointError(ValueError):
    """Raised when a real runtime checkpoint cannot be captured or restored."""


@dataclass(frozen=True, slots=True)
class RuntimeCheckpointBinding:
    """Exact source, campaign contracts, and island boundary for all components."""

    source_git_sha: str
    source_manifest_sha256: str
    config_contract_sha256: str
    seed_contract_sha256: str
    run_generation_id: str
    island_id: str
    generation_index: int
    completed_tick: int

    def __post_init__(self) -> None:
        _git_sha(self.source_git_sha, field="source_git_sha")
        _sha256(self.source_manifest_sha256, field="source_manifest_sha256")
        _sha256(self.config_contract_sha256, field="config_contract_sha256")
        _sha256(self.seed_contract_sha256, field="seed_contract_sha256")
        _identifier(self.run_generation_id, field="run_generation_id")
        _identifier(self.island_id, field="island_id")
        _nonnegative_int(self.generation_index, field="generation_index")
        _nonnegative_int(self.completed_tick, field="completed_tick")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPEN_ECOLOGY_RUNTIME_BINDING_SCHEMA_VERSION,
            "source_git_sha": self.source_git_sha,
            "source_manifest_sha256": self.source_manifest_sha256,
            "config_contract_sha256": self.config_contract_sha256,
            "seed_contract_sha256": self.seed_contract_sha256,
            "run_generation_id": self.run_generation_id,
            "island_id": self.island_id,
            "generation_index": self.generation_index,
            "completed_tick": self.completed_tick,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RuntimeCheckpointBinding:
        _exact_keys(
            value,
            {
                "schema_version",
                "source_git_sha",
                "source_manifest_sha256",
                "config_contract_sha256",
                "seed_contract_sha256",
                "run_generation_id",
                "island_id",
                "generation_index",
                "completed_tick",
            },
            field="runtime checkpoint binding",
        )
        if value["schema_version"] != OPEN_ECOLOGY_RUNTIME_BINDING_SCHEMA_VERSION:
            raise OpenEcologyRuntimeCheckpointError(
                "runtime checkpoint binding schema version is stale"
            )
        return cls(
            source_git_sha=value["source_git_sha"],  # type: ignore[arg-type]
            source_manifest_sha256=value["source_manifest_sha256"],  # type: ignore[arg-type]
            config_contract_sha256=value["config_contract_sha256"],  # type: ignore[arg-type]
            seed_contract_sha256=value["seed_contract_sha256"],  # type: ignore[arg-type]
            run_generation_id=value["run_generation_id"],  # type: ignore[arg-type]
            island_id=value["island_id"],  # type: ignore[arg-type]
            generation_index=value["generation_index"],  # type: ignore[arg-type]
            completed_tick=value["completed_tick"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RuntimeCheckpointComponents:
    world_state: VersionedCheckpointState
    environment_rng_state: VersionedCheckpointState
    recurrent_policy_state: VersionedCheckpointState
    public_feedback_history: VersionedCheckpointState
    sampling_rng_state: VersionedCheckpointState
    genome_population_snapshot: VersionedCheckpointState
    evidence_writer_continuation_state: VersionedCheckpointState


@dataclass(frozen=True, slots=True)
class ExtractedRuntimeEvidenceContinuation:
    """Strictly decoded adapter binding and rotating-writer continuation."""

    binding: RuntimeCheckpointBinding
    continuation_state: dict[str, object]


@dataclass(frozen=True, slots=True)
class _DecodedRecurrentPolicyState:
    hidden_by_agent: dict[int, Tensor]
    decision_index: int
    genome_population_pre_founder_state_sha256: str
    last_world_genome_provenance: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class _DecodedFeedbackHistoryState:
    feedback_by_agent: dict[int, PreviousPublicFeedbackInput]
    history_by_agent: dict[int, list[dict[str, object]]]


def extract_open_ecology_runtime_evidence_continuation(
    *,
    adapter_schema_version: object,
    payload: Mapping[str, object],
) -> ExtractedRuntimeEvidenceContinuation:
    """Strictly extract a rotating-writer state from its runtime adapter."""

    if adapter_schema_version != OPEN_ECOLOGY_EVIDENCE_WRITER_ADAPTER_SCHEMA_VERSION:
        raise OpenEcologyRuntimeCheckpointError(
            "evidence writer continuation adapter schema version is stale"
        )
    if not isinstance(payload, Mapping):
        raise OpenEcologyRuntimeCheckpointError(
            "evidence writer continuation adapter payload must be an object"
        )
    _exact_keys(
        payload,
        {"binding", "state"},
        field="evidence writer continuation adapter payload",
    )
    raw_binding = payload["binding"]
    if not isinstance(raw_binding, Mapping):
        raise OpenEcologyRuntimeCheckpointError(
            "evidence writer continuation adapter binding must be an object"
        )
    binding = RuntimeCheckpointBinding.from_dict(raw_binding)
    state = payload["state"]
    if not isinstance(state, Mapping):
        raise OpenEcologyRuntimeCheckpointError(
            "evidence writer continuation adapter state must be an object"
        )
    _exact_keys(
        state,
        {"continuation_state"},
        field="evidence writer continuation adapter state",
    )
    continuation = state["continuation_state"]
    if not isinstance(continuation, Mapping):
        raise OpenEcologyRuntimeCheckpointError(
            "evidence writer continuation must be an object"
        )
    return ExtractedRuntimeEvidenceContinuation(
        binding=binding,
        continuation_state=_validated_evidence_continuation(
            continuation,
            binding=binding,
            allow_lagging_tick=True,
        ),
    )


def capture_open_ecology_runtime_checkpoint(
    world: SimulationWorld,
    policy: DeterministicPublicRecurrentPolicy,
    *,
    binding: RuntimeCheckpointBinding,
    evidence_writer_continuation_state: Mapping[str, object],
    max_state_bytes: int = OPEN_ECOLOGY_RUNTIME_DEFAULT_MAX_STATE_BYTES,
    max_history_records: int = OPEN_ECOLOGY_RUNTIME_DEFAULT_MAX_HISTORY_RECORDS,
) -> RuntimeCheckpointComponents:
    """Capture all seven real runtime components at a finalized tick boundary."""

    _validate_runtime_pair(world, policy, binding=binding)
    if policy._pending_by_agent:
        raise OpenEcologyRuntimeCheckpointError(
            "checkpoint requires a finalized tick boundary with no pending decisions"
        )
    if policy._reset_recurrent_state_each_decision and policy._feedback_by_agent:
        raise OpenEcologyRuntimeCheckpointError(
            "full-reset recurrent policy checkpoint cannot retain previous feedback"
        )
    if world.trajectory_sink is not None:
        raise OpenEcologyRuntimeCheckpointError(
            "checkpoint does not serialize an open trajectory sink"
        )
    retained_buffers = {
        "events": world.events,
        "viewer_frames": world.viewer_frames,
        "trajectory_records": world.trajectory_records,
        "policy_decision_diagnostics_records": (
            world.policy_decision_diagnostics_records
        ),
        "policy_update_trace_records": world.policy_update_trace_records,
        "policy_update_trace_records_by_trajectory_record": (
            world.policy_update_trace_records_by_trajectory_record
        ),
    }
    nonempty_buffers = sorted(
        name for name, records in retained_buffers.items() if records
    )
    if world.record_events or world.retain_trajectory_records or nonempty_buffers:
        raise OpenEcologyRuntimeCheckpointError(
            "persistent checkpoint requires bounded no-retention runtime mode; "
            f"nonempty retained buffers: {nonempty_buffers}"
        )
    byte_ceiling = _positive_int(max_state_bytes, field="max_state_bytes")
    history_ceiling = _nonnegative_int(
        max_history_records,
        field="max_history_records",
    )
    binding_payload = binding.to_dict()
    world_attributes = {
        key: value
        for key, value in world.__dict__.items()
        if key not in _WORLD_EXCLUDED_ATTRIBUTES
    }
    if _PERSISTENT_RUNNER_STATE_ATTRIBUTE in world_attributes:
        world_attributes[_PERSISTENT_RUNNER_STATE_ATTRIBUTE] = (
            _validated_persistent_runner_state(
                world_attributes[_PERSISTENT_RUNNER_STATE_ATTRIBUTE],
                policy=policy,
                binding=binding,
            )
        )
    world_payload = _component_payload(
        binding_payload,
        {
            "world_config_sha256": _digest(world.config.to_dict()),
            "attribute_names": sorted(world_attributes),
            "attributes": _encode_value(world_attributes),
        },
    )
    _bounded_payload(world_payload, byte_ceiling, field="world_state")

    rng_version, rng_values, gauss_next = world.rng.getstate()
    environment_rng_payload = _component_payload(
        binding_payload,
        {
            "python_random_state_version": rng_version,
            "internal_state": list(rng_values),
            "gauss_next": gauss_next,
        },
    )

    hidden_states = [
        {
            "agent_id": agent_id,
            "tensor": _encode_float32_tensor(tensor),
        }
        for agent_id, tensor in sorted(policy._state_by_agent.items())
    ]
    recurrent_policy_payload = _component_payload(
        binding_payload,
        {
            "artifact_digest": policy._artifact_digest,
            "capture_public_history": policy._capture_public_history,
            "decision_index": policy._decision_index,
            "genome_conditioning_mode": policy._genome_conditioning_mode,
            "genome_population_pre_founder_state_sha256": (
                policy._genome_population_pre_founder_state_sha256
            ),
            "hidden_states": hidden_states,
            "last_world_genome_provenance": _json_clone(
                policy._last_world_genome_provenance
            ),
            "model_state_sha256": policy._model_state_sha256,
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "reset_recurrent_state_each_decision": (
                policy._reset_recurrent_state_each_decision
            ),
            "selection": policy._selection,
        },
    )
    _bounded_payload(recurrent_policy_payload, byte_ceiling, field="recurrent_policy")

    history_count = sum(
        len(records) for records in policy._public_history_by_agent.values()
    )
    if history_count > history_ceiling:
        raise OpenEcologyRuntimeCheckpointError(
            "public recurrent history exceeds max_history_records "
            f"({history_count} > {history_ceiling})"
        )
    feedback_payload = _component_payload(
        binding_payload,
        {
            "feedback_by_agent": [
                {
                    "agent_id": agent_id,
                    "requested_action_id": feedback.requested_action_id,
                    "resolved_action_id": feedback.resolved_action_id,
                    "resolution_action_valid": feedback.resolution_action_valid,
                    "moved": feedback.moved,
                    "reward_total": feedback.reward_total,
                }
                for agent_id, feedback in sorted(policy._feedback_by_agent.items())
            ],
            "history_by_agent": [
                {"agent_id": agent_id, "records": _json_clone(records)}
                for agent_id, records in sorted(policy._public_history_by_agent.items())
            ],
            "history_record_count": history_count,
        },
    )
    _bounded_payload(feedback_payload, byte_ceiling, field="public_feedback_history")

    generator = policy._sampling_generator
    if generator is None:
        sampling_state: dict[str, object] = {
            "enabled": False,
            "sampling_seed": None,
            "state_hex": None,
            "state_sha256": None,
        }
    else:
        raw_state = bytes(generator.get_state().cpu().tolist())
        sampling_state = {
            "enabled": True,
            "sampling_seed": policy._sampling_seed,
            "state_hex": raw_state.hex(),
            "state_sha256": hashlib.sha256(raw_state).hexdigest(),
        }
    sampling_rng_payload = _component_payload(binding_payload, sampling_state)

    manager = policy._genome_population_manager
    if manager is None:
        raise OpenEcologyRuntimeCheckpointError(
            "H/Z/R checkpoint requires an active recurrent genome population"
        )
    pre_founder_sha256 = _sha256(
        policy._genome_population_pre_founder_state_sha256,
        field="genome_population_pre_founder_state_sha256",
    )
    capture_recurrent = _DecodedRecurrentPolicyState(
        hidden_by_agent=policy._state_by_agent,
        decision_index=_nonnegative_int(
            policy._decision_index,
            field="decision_index",
        ),
        genome_population_pre_founder_state_sha256=pre_founder_sha256,
        last_world_genome_provenance=policy._last_world_genome_provenance,
    )
    capture_feedback = _DecodedFeedbackHistoryState(
        feedback_by_agent=policy._feedback_by_agent,
        history_by_agent=policy._public_history_by_agent,
    )
    _validate_fresh_genome_population_binding(
        policy,
        restored_manager=manager,
        restored_pre_founder_sha256=pre_founder_sha256,
    )
    _validate_decoded_runtime_state(
        world,
        policy,
        binding=binding,
        attributes=world_attributes,
        recurrent=capture_recurrent,
        feedback=capture_feedback,
        manager=manager,
    )
    genome_payload = _component_payload(
        binding_payload,
        {"snapshot": manager.snapshot_artifact()},
    )
    _bounded_payload(genome_payload, byte_ceiling, field="genome_population")

    validated_continuation = _validated_evidence_continuation(
        evidence_writer_continuation_state,
        binding=binding,
        allow_lagging_tick=(_PERSISTENT_RUNNER_STATE_ATTRIBUTE in world_attributes),
    )
    _validate_persistent_runner_continuation_binding(
        world_attributes,
        continuation=validated_continuation,
    )
    writer_payload = _component_payload(
        binding_payload,
        {"continuation_state": validated_continuation},
    )
    return RuntimeCheckpointComponents(
        world_state=VersionedCheckpointState(
            OPEN_ECOLOGY_WORLD_STATE_SCHEMA_VERSION,
            world_payload,
        ),
        environment_rng_state=VersionedCheckpointState(
            OPEN_ECOLOGY_ENVIRONMENT_RNG_SCHEMA_VERSION,
            environment_rng_payload,
        ),
        recurrent_policy_state=VersionedCheckpointState(
            OPEN_ECOLOGY_RECURRENT_POLICY_SCHEMA_VERSION,
            recurrent_policy_payload,
        ),
        public_feedback_history=VersionedCheckpointState(
            OPEN_ECOLOGY_PUBLIC_FEEDBACK_SCHEMA_VERSION,
            feedback_payload,
        ),
        sampling_rng_state=VersionedCheckpointState(
            OPEN_ECOLOGY_SAMPLING_RNG_SCHEMA_VERSION,
            sampling_rng_payload,
        ),
        genome_population_snapshot=VersionedCheckpointState(
            OPEN_ECOLOGY_GENOME_POPULATION_SCHEMA_VERSION,
            genome_payload,
        ),
        evidence_writer_continuation_state=VersionedCheckpointState(
            OPEN_ECOLOGY_EVIDENCE_WRITER_ADAPTER_SCHEMA_VERSION,
            writer_payload,
        ),
    )


def restore_open_ecology_runtime_checkpoint(
    world: SimulationWorld,
    policy: DeterministicPublicRecurrentPolicy,
    *,
    binding: RuntimeCheckpointBinding,
    components: RuntimeCheckpointComponents,
) -> dict[str, object]:
    """Atomically restore a fresh real world/policy pair and writer continuation."""

    _validate_runtime_pair(world, policy, binding=binding, require_tick=False)
    expected_versions = {
        "world_state": OPEN_ECOLOGY_WORLD_STATE_SCHEMA_VERSION,
        "environment_rng_state": OPEN_ECOLOGY_ENVIRONMENT_RNG_SCHEMA_VERSION,
        "recurrent_policy_state": OPEN_ECOLOGY_RECURRENT_POLICY_SCHEMA_VERSION,
        "public_feedback_history": OPEN_ECOLOGY_PUBLIC_FEEDBACK_SCHEMA_VERSION,
        "sampling_rng_state": OPEN_ECOLOGY_SAMPLING_RNG_SCHEMA_VERSION,
        "genome_population_snapshot": OPEN_ECOLOGY_GENOME_POPULATION_SCHEMA_VERSION,
        "evidence_writer_continuation_state": (
            OPEN_ECOLOGY_EVIDENCE_WRITER_ADAPTER_SCHEMA_VERSION
        ),
    }
    states = {name: getattr(components, name) for name in expected_versions}
    for name, expected_version in expected_versions.items():
        state = states[name]
        if state.schema_version != expected_version:
            raise OpenEcologyRuntimeCheckpointError(
                f"{name} adapter schema version is stale"
            )
        _validate_component_binding(state.payload, binding=binding, field=name)

    world_state = _component_state(states["world_state"].payload, field="world_state")
    _exact_keys(
        world_state,
        {"world_config_sha256", "attribute_names", "attributes"},
        field="world_state.state",
    )
    if world_state["world_config_sha256"] != _digest(world.config.to_dict()):
        raise OpenEcologyRuntimeCheckpointError(
            "fresh world config does not match checkpoint world config"
        )
    expected_attribute_names = sorted(
        key for key in world.__dict__ if key not in _WORLD_EXCLUDED_ATTRIBUTES
    )
    if world_state["attribute_names"] != expected_attribute_names:
        raise OpenEcologyRuntimeCheckpointError(
            "SimulationWorld runtime attribute surface drifted"
        )
    decoded_attributes = _decode_value(world_state["attributes"])
    if not isinstance(decoded_attributes, dict):
        raise OpenEcologyRuntimeCheckpointError(
            "decoded SimulationWorld attributes are not a dictionary"
        )
    if sorted(decoded_attributes) != expected_attribute_names:
        raise OpenEcologyRuntimeCheckpointError(
            "decoded SimulationWorld attribute set drifted"
        )

    rng_state = _component_state(
        states["environment_rng_state"].payload,
        field="environment_rng_state",
    )
    _exact_keys(
        rng_state,
        {"python_random_state_version", "internal_state", "gauss_next"},
        field="environment_rng_state.state",
    )
    restored_rng = Random()
    try:
        restored_rng.setstate(
            (
                _nonnegative_int(
                    rng_state["python_random_state_version"],
                    field="python_random_state_version",
                ),
                tuple(
                    _nonnegative_int(value, field="internal_state item")
                    for value in _list(
                        rng_state["internal_state"], field="internal_state"
                    )
                ),
                _optional_finite_float(rng_state["gauss_next"], field="gauss_next"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise OpenEcologyRuntimeCheckpointError(
            f"environment RNG state is invalid: {exc}"
        ) from exc

    recurrent_state = _component_state(
        states["recurrent_policy_state"].payload,
        field="recurrent_policy_state",
    )
    decoded_recurrent = _decode_recurrent_policy_state(policy, recurrent_state)
    feedback_state = _component_state(
        states["public_feedback_history"].payload,
        field="public_feedback_history",
    )
    decoded_feedback = _decode_feedback_history(policy, feedback_state)
    sampling_state = _component_state(
        states["sampling_rng_state"].payload,
        field="sampling_rng_state",
    )
    decoded_sampling_generator = _decode_sampling_generator(policy, sampling_state)
    genome_state = _component_state(
        states["genome_population_snapshot"].payload,
        field="genome_population_snapshot",
    )
    _exact_keys(genome_state, {"snapshot"}, field="genome_population.state")
    snapshot = genome_state["snapshot"]
    if not isinstance(snapshot, Mapping):
        raise OpenEcologyRuntimeCheckpointError(
            "genome population snapshot must be an object"
        )
    try:
        manager = RecurrentGenomePopulationManager.from_snapshot_artifact(snapshot)
    except ValueError as exc:
        raise OpenEcologyRuntimeCheckpointError(
            f"genome population snapshot is invalid: {exc}"
        ) from exc
    if manager.mode.value not in {"heritable", "zero_all"}:
        raise OpenEcologyRuntimeCheckpointError(
            "restored genome population is not an H/Z/R population"
        )
    _validate_fresh_genome_population_binding(
        policy,
        restored_manager=manager,
        restored_pre_founder_sha256=(
            decoded_recurrent.genome_population_pre_founder_state_sha256
        ),
    )
    _validate_decoded_runtime_state(
        world,
        policy,
        binding=binding,
        attributes=decoded_attributes,
        recurrent=decoded_recurrent,
        feedback=decoded_feedback,
        manager=manager,
    )

    writer_state = _component_state(
        states["evidence_writer_continuation_state"].payload,
        field="evidence_writer_continuation_state",
    )
    _exact_keys(
        writer_state,
        {"continuation_state"},
        field="evidence_writer_continuation_state.state",
    )
    continuation = writer_state["continuation_state"]
    if not isinstance(continuation, Mapping):
        raise OpenEcologyRuntimeCheckpointError(
            "evidence writer continuation must be an object"
        )
    validated_continuation = _validated_evidence_continuation(
        continuation,
        binding=binding,
        allow_lagging_tick=(_PERSISTENT_RUNNER_STATE_ATTRIBUTE in decoded_attributes),
    )
    _validate_persistent_runner_continuation_binding(
        decoded_attributes,
        continuation=validated_continuation,
    )

    # All decoding, dependency validation, and cross-component validation is
    # complete. These ownership transfers cannot call external validators.
    policy._state_by_agent = decoded_recurrent.hidden_by_agent
    policy._decision_index = decoded_recurrent.decision_index
    policy._genome_population_pre_founder_state_sha256 = (
        decoded_recurrent.genome_population_pre_founder_state_sha256
    )
    policy._last_world_genome_provenance = (
        decoded_recurrent.last_world_genome_provenance
    )
    policy._feedback_by_agent = decoded_feedback.feedback_by_agent
    policy._public_history_by_agent = decoded_feedback.history_by_agent
    policy._sampling_generator = decoded_sampling_generator
    policy._genome_population_manager = manager
    world.__dict__.update(decoded_attributes)
    world.rng = restored_rng
    world.policy = policy
    world.trajectory_sink = None
    return validated_continuation


def _decode_recurrent_policy_state(
    policy: DeterministicPublicRecurrentPolicy,
    state: Mapping[str, object],
) -> _DecodedRecurrentPolicyState:
    expected_keys = {
        "artifact_digest",
        "capture_public_history",
        "decision_index",
        "genome_conditioning_mode",
        "genome_population_pre_founder_state_sha256",
        "hidden_states",
        "last_world_genome_provenance",
        "model_state_sha256",
        "policy_id",
        "policy_version",
        "reset_recurrent_state_each_decision",
        "selection",
    }
    _exact_keys(state, expected_keys, field="recurrent_policy_state.state")
    expected_values = {
        "artifact_digest": policy._artifact_digest,
        "capture_public_history": policy._capture_public_history,
        "genome_conditioning_mode": policy._genome_conditioning_mode,
        "model_state_sha256": recurrent_model_state_sha256(policy.model),
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "reset_recurrent_state_each_decision": (
            policy._reset_recurrent_state_each_decision
        ),
        "selection": policy._selection,
    }
    for key, expected in expected_values.items():
        if state[key] != expected:
            raise OpenEcologyRuntimeCheckpointError(
                f"fresh recurrent policy {key} does not match checkpoint"
            )
    if policy._pending_by_agent:
        raise OpenEcologyRuntimeCheckpointError(
            "fresh recurrent policy unexpectedly has pending decisions"
        )
    hidden_states = _list(state["hidden_states"], field="hidden_states")
    restored_hidden: dict[int, Tensor] = {}
    previous_agent_id = 0
    for index, entry in enumerate(hidden_states):
        if not isinstance(entry, Mapping):
            raise OpenEcologyRuntimeCheckpointError(
                f"hidden_states[{index}] must be an object"
            )
        _exact_keys(entry, {"agent_id", "tensor"}, field=f"hidden_states[{index}]")
        agent_id = _positive_int(entry["agent_id"], field="hidden state agent_id")
        if agent_id <= previous_agent_id:
            raise OpenEcologyRuntimeCheckpointError(
                "hidden states must use strictly increasing agent ids"
            )
        previous_agent_id = agent_id
        tensor_payload = entry["tensor"]
        if not isinstance(tensor_payload, Mapping):
            raise OpenEcologyRuntimeCheckpointError("hidden tensor must be an object")
        tensor = _decode_float32_tensor(tensor_payload)
        expected_shape = tuple(policy.model.initial_state(1).shape)
        if tuple(tensor.shape) != expected_shape:
            raise OpenEcologyRuntimeCheckpointError(
                "hidden tensor shape does not match frozen model"
            )
        restored_hidden[agent_id] = tensor
    if policy._reset_recurrent_state_each_decision and restored_hidden:
        raise OpenEcologyRuntimeCheckpointError(
            "reset-each-decision policy checkpoint cannot retain hidden states"
        )
    decision_index = _nonnegative_int(
        state["decision_index"],
        field="decision_index",
    )
    prefounder = state["genome_population_pre_founder_state_sha256"]
    prefounder = _sha256(
        prefounder,
        field="genome_population_pre_founder_state_sha256",
    )
    provenance = state["last_world_genome_provenance"]
    if provenance is not None and not isinstance(provenance, dict):
        raise OpenEcologyRuntimeCheckpointError(
            "last_world_genome_provenance must be an object or null"
        )
    cloned_provenance = _json_clone(provenance)
    if cloned_provenance is not None and not isinstance(cloned_provenance, dict):
        raise OpenEcologyRuntimeCheckpointError(
            "last_world_genome_provenance must be an object or null"
        )
    return _DecodedRecurrentPolicyState(
        hidden_by_agent=restored_hidden,
        decision_index=decision_index,
        genome_population_pre_founder_state_sha256=prefounder,
        last_world_genome_provenance=cloned_provenance,
    )


def _decode_feedback_history(
    policy: DeterministicPublicRecurrentPolicy,
    state: Mapping[str, object],
) -> _DecodedFeedbackHistoryState:
    _exact_keys(
        state,
        {"feedback_by_agent", "history_by_agent", "history_record_count"},
        field="public_feedback_history.state",
    )
    restored_feedback: dict[int, PreviousPublicFeedbackInput] = {}
    previous_agent_id = 0
    for index, entry in enumerate(
        _list(state["feedback_by_agent"], field="feedback_by_agent")
    ):
        if not isinstance(entry, Mapping):
            raise OpenEcologyRuntimeCheckpointError(
                f"feedback_by_agent[{index}] must be an object"
            )
        _exact_keys(
            entry,
            {
                "agent_id",
                "requested_action_id",
                "resolved_action_id",
                "resolution_action_valid",
                "moved",
                "reward_total",
            },
            field=f"feedback_by_agent[{index}]",
        )
        agent_id = _positive_int(entry["agent_id"], field="feedback agent_id")
        if agent_id <= previous_agent_id:
            raise OpenEcologyRuntimeCheckpointError(
                "feedback must use strictly increasing agent ids"
            )
        previous_agent_id = agent_id
        try:
            restored_feedback[agent_id] = PreviousPublicFeedbackInput(
                requested_action_id=entry["requested_action_id"],  # type: ignore[arg-type]
                resolved_action_id=entry["resolved_action_id"],  # type: ignore[arg-type]
                resolution_action_valid=entry["resolution_action_valid"],  # type: ignore[arg-type]
                moved=entry["moved"],  # type: ignore[arg-type]
                reward_total=_finite_float(
                    entry["reward_total"],
                    field="reward_total",
                ),
            )
        except ValueError as exc:
            raise OpenEcologyRuntimeCheckpointError(
                f"feedback for agent {agent_id} is invalid: {exc}"
            ) from exc
    restored_history: dict[int, list[dict[str, object]]] = {}
    previous_agent_id = 0
    observed_count = 0
    for index, entry in enumerate(
        _list(state["history_by_agent"], field="history_by_agent")
    ):
        if not isinstance(entry, Mapping):
            raise OpenEcologyRuntimeCheckpointError(
                f"history_by_agent[{index}] must be an object"
            )
        _exact_keys(entry, {"agent_id", "records"}, field=f"history_by_agent[{index}]")
        agent_id = _positive_int(entry["agent_id"], field="history agent_id")
        if agent_id <= previous_agent_id:
            raise OpenEcologyRuntimeCheckpointError(
                "history must use strictly increasing agent ids"
            )
        previous_agent_id = agent_id
        records = _json_clone(_list(entry["records"], field="history records"))
        if not isinstance(records, list):
            raise OpenEcologyRuntimeCheckpointError(
                "public history records must be a list"
            )
        if not all(isinstance(record, dict) for record in records):
            raise OpenEcologyRuntimeCheckpointError(
                "public history records must be objects"
            )
        try:
            validate_public_recurrent_history_prefix(
                {
                    "schema_version": (RECURRENT_PUBLIC_HISTORY_PREFIX_SCHEMA_VERSION),
                    "record_count": len(records),
                    "records": records,
                },
                expected_public_input_schema_version=(
                    policy.model.config.public_input_schema_version
                ),
                expected_public_input_size=policy.model.config.public_input_size,
            )
        except ValueError as exc:
            raise OpenEcologyRuntimeCheckpointError(
                f"public history for agent {agent_id} is invalid: {exc}"
            ) from exc
        restored_history[agent_id] = records  # type: ignore[assignment]
        observed_count += len(records)
    if observed_count != _nonnegative_int(
        state["history_record_count"],
        field="history_record_count",
    ):
        raise OpenEcologyRuntimeCheckpointError(
            "public history record count does not match records"
        )
    if not policy._capture_public_history and restored_history:
        raise OpenEcologyRuntimeCheckpointError(
            "history exists for a policy configured not to capture it"
        )
    if policy._reset_recurrent_state_each_decision and restored_feedback:
        raise OpenEcologyRuntimeCheckpointError(
            "full-reset recurrent policy checkpoint cannot restore previous feedback"
        )
    return _DecodedFeedbackHistoryState(
        feedback_by_agent=restored_feedback,
        history_by_agent=restored_history,
    )


def _decode_sampling_generator(
    policy: DeterministicPublicRecurrentPolicy,
    state: Mapping[str, object],
) -> torch.Generator | None:
    _exact_keys(
        state,
        {"enabled", "sampling_seed", "state_hex", "state_sha256"},
        field="sampling_rng_state.state",
    )
    enabled = state["enabled"]
    if type(enabled) is not bool:
        raise OpenEcologyRuntimeCheckpointError(
            "sampling RNG enabled must be an exact boolean"
        )
    if enabled != (policy._sampling_generator is not None):
        raise OpenEcologyRuntimeCheckpointError(
            "fresh policy sampling mode does not match checkpoint"
        )
    if state["sampling_seed"] != policy._sampling_seed:
        raise OpenEcologyRuntimeCheckpointError(
            "fresh policy sampling seed does not match checkpoint"
        )
    if not enabled:
        if state["state_hex"] is not None or state["state_sha256"] is not None:
            raise OpenEcologyRuntimeCheckpointError(
                "disabled sampling RNG must not contain state"
            )
        return None
    raw_hex = state["state_hex"]
    if not isinstance(raw_hex, str) or not raw_hex or len(raw_hex) % 2:
        raise OpenEcologyRuntimeCheckpointError("sampling RNG state_hex is invalid")
    try:
        raw_state = bytes.fromhex(raw_hex)
    except ValueError as exc:
        raise OpenEcologyRuntimeCheckpointError(
            "sampling RNG state_hex is invalid"
        ) from exc
    expected_sha256 = _sha256(state["state_sha256"], field="sampling state SHA256")
    if hashlib.sha256(raw_state).hexdigest() != expected_sha256:
        raise OpenEcologyRuntimeCheckpointError("sampling RNG state SHA256 mismatch")
    fresh_generator = policy._sampling_generator
    assert fresh_generator is not None
    try:
        restored_generator = torch.Generator(device=fresh_generator.device)
        restored_generator.set_state(torch.tensor(list(raw_state), dtype=torch.uint8))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OpenEcologyRuntimeCheckpointError(
            f"sampling RNG state is invalid: {exc}"
        ) from exc
    return restored_generator


def _validate_fresh_genome_population_binding(
    policy: DeterministicPublicRecurrentPolicy,
    *,
    restored_manager: RecurrentGenomePopulationManager,
    restored_pre_founder_sha256: str,
) -> None:
    fresh_manager = policy._genome_population_manager
    if fresh_manager is None:
        raise OpenEcologyRuntimeCheckpointError(
            "fresh policy has no active recurrent genome population"
        )
    comparisons = {
        "mode": (fresh_manager.mode, restored_manager.mode),
        "genome_stream_seed": (
            fresh_manager.genome_stream_seed,
            restored_manager.genome_stream_seed,
        ),
        "world_identity": (
            fresh_manager.world_identity,
            restored_manager.world_identity,
        ),
        "binding_sha256": (
            fresh_manager.binding_sha256,
            restored_manager.binding_sha256,
        ),
        "mutation": (fresh_manager.mutation, restored_manager.mutation),
    }
    for field, (fresh_value, restored_value) in comparisons.items():
        if fresh_value != restored_value:
            raise OpenEcologyRuntimeCheckpointError(
                f"fresh recurrent genome population {field} does not match checkpoint"
            )
    fresh_pre_founder_sha256 = policy._genome_population_pre_founder_state_sha256
    if (
        fresh_pre_founder_sha256 != fresh_manager.empty_state_sha256
        or restored_pre_founder_sha256 != restored_manager.empty_state_sha256
        or restored_pre_founder_sha256 != fresh_pre_founder_sha256
    ):
        raise OpenEcologyRuntimeCheckpointError(
            "fresh recurrent genome population pre-founder binding does not "
            "match checkpoint"
        )


def _validate_decoded_runtime_state(
    fresh_world: SimulationWorld,
    policy: DeterministicPublicRecurrentPolicy,
    *,
    binding: RuntimeCheckpointBinding,
    attributes: Mapping[str, object],
    recurrent: _DecodedRecurrentPolicyState,
    feedback: _DecodedFeedbackHistoryState,
    manager: RecurrentGenomePopulationManager,
) -> None:
    tick = _nonnegative_int(attributes.get("tick"), field="restored world tick")
    if tick != binding.completed_tick:
        raise OpenEcologyRuntimeCheckpointError(
            "restored world tick does not match checkpoint binding"
        )
    if (
        attributes.get("_policy_id") != policy.policy_id
        or attributes.get("_policy_version") != policy.policy_version
    ):
        raise OpenEcologyRuntimeCheckpointError(
            "restored world policy identity does not match the fresh policy"
        )
    if _PERSISTENT_RUNNER_STATE_ATTRIBUTE in attributes:
        _validated_persistent_runner_state(
            attributes[_PERSISTENT_RUNNER_STATE_ATTRIBUTE],
            policy=policy,
            binding=binding,
        )

    raw_agents = attributes.get("agents")
    if not isinstance(raw_agents, dict):
        raise OpenEcologyRuntimeCheckpointError(
            "restored world agents must be a dictionary"
        )
    agents: dict[int, Agent] = {}
    for raw_agent_id, raw_agent in raw_agents.items():
        agent_id = _positive_int(raw_agent_id, field="world agent id")
        if not isinstance(raw_agent, Agent):
            raise OpenEcologyRuntimeCheckpointError(
                f"restored world agent {agent_id} has an invalid type"
            )
        if agent_id != raw_agent.agent_id:
            raise OpenEcologyRuntimeCheckpointError(
                "world agent dictionary keys do not match agent identities"
            )
        if type(raw_agent.alive) is not bool:
            raise OpenEcologyRuntimeCheckpointError(
                f"restored world agent {agent_id} alive flag is invalid"
            )
        agents[agent_id] = raw_agent

    raw_grid = attributes.get("grid")
    if (
        not isinstance(raw_grid, list)
        or len(raw_grid) != fresh_world.config.height
        or any(
            not isinstance(row, list) or len(row) != fresh_world.config.width
            for row in raw_grid
        )
    ):
        raise OpenEcologyRuntimeCheckpointError(
            "restored world grid dimensions do not match world config"
        )
    occupied: dict[int, tuple[int, int]] = {}
    for y, row in enumerate(raw_grid):
        assert isinstance(row, list)
        for x, tile in enumerate(row):
            if not isinstance(tile, Tile):
                raise OpenEcologyRuntimeCheckpointError(
                    "restored world grid contains a non-Tile value"
                )
            occupant_id = tile.occupant_id
            if occupant_id is None:
                continue
            occupant_id = _positive_int(occupant_id, field="grid occupant id")
            if occupant_id in occupied:
                raise OpenEcologyRuntimeCheckpointError(
                    "one world agent occupies more than one tile"
                )
            occupied[occupant_id] = (x, y)

    live_agent_ids = {agent_id for agent_id, agent in agents.items() if agent.alive}
    if set(occupied) != live_agent_ids:
        raise OpenEcologyRuntimeCheckpointError(
            "grid occupants do not exactly match live world agents"
        )
    if any(
        occupied[agent_id] != (agents[agent_id].x, agents[agent_id].y)
        for agent_id in live_agent_ids
    ):
        raise OpenEcologyRuntimeCheckpointError(
            "grid occupancy coordinates do not match live agent coordinates"
        )
    next_agent_id = _positive_int(
        attributes.get("next_agent_id"),
        field="world next_agent_id",
    )
    if agents and next_agent_id <= max(agents):
        raise OpenEcologyRuntimeCheckpointError(
            "world next_agent_id is not above all existing agent ids"
        )

    agent_ids = set(agents)
    for label, state_agent_ids in (
        ("recurrent hidden state", set(recurrent.hidden_by_agent)),
        ("public feedback", set(feedback.feedback_by_agent)),
        ("public history", set(feedback.history_by_agent)),
    ):
        if state_agent_ids - agent_ids:
            raise OpenEcologyRuntimeCheckpointError(
                f"{label} refers to an unknown world agent"
            )
    genome_agent_ids = set(manager.registered_agent_ids)
    if genome_agent_ids != live_agent_ids:
        raise OpenEcologyRuntimeCheckpointError(
            "genome population does not exactly match live world agents"
        )
    for agent_id in sorted(live_agent_ids):
        metadata = agents[agent_id].mind_inheritance_metadata
        if not isinstance(metadata, dict):
            raise OpenEcologyRuntimeCheckpointError(
                f"live agent {agent_id} mind inheritance metadata is invalid"
            )
        _validate_plain_json_value(
            metadata,
            field=f"agent {agent_id} mind inheritance metadata",
        )
        _exact_keys(
            metadata,
            {
                "schema_version",
                "inherited_state",
                "state_size",
                "population_mode",
                "population_binding_sha256",
                "inheritance_kind",
                "genome_sha256",
                "parent_genome_sha256s",
            },
            field=f"agent {agent_id} mind inheritance metadata",
        )
        expected_metadata = {
            "schema_version": RECURRENT_GENOME_MIND_METADATA_SCHEMA_VERSION,
            "inherited_state": True,
            "state_size": RECURRENT_CONTROLLER_GENOME_SIZE,
            "population_mode": manager.mode.value,
            "population_binding_sha256": manager.binding_sha256,
            "genome_sha256": manager.genome_sha256_for_agent(agent_id),
        }
        if any(
            metadata.get(field) != expected
            for field, expected in expected_metadata.items()
        ):
            raise OpenEcologyRuntimeCheckpointError(
                f"live agent {agent_id} mind inheritance metadata does not "
                "match the genome population"
            )
        inheritance_kind = metadata["inheritance_kind"]
        parent_genome_sha256s = metadata["parent_genome_sha256s"]
        expected_parent_count = {
            "founder": 0,
            "asexual": 1,
            "two_parent": 2,
        }.get(inheritance_kind)
        if (
            expected_parent_count is None
            or not isinstance(parent_genome_sha256s, list)
            or len(parent_genome_sha256s) != expected_parent_count
            or any(
                _SHA256_RE.fullmatch(value) is None
                for value in parent_genome_sha256s
                if isinstance(value, str)
            )
            or not all(isinstance(value, str) for value in parent_genome_sha256s)
        ):
            raise OpenEcologyRuntimeCheckpointError(
                f"live agent {agent_id} mind inheritance lineage is invalid"
            )

    retained_buffer_names = (
        "events",
        "viewer_frames",
        "trajectory_records",
        "policy_decision_diagnostics_records",
        "policy_update_trace_records",
        "policy_update_trace_records_by_trajectory_record",
    )
    nonempty_buffers = sorted(
        name for name in retained_buffer_names if attributes.get(name)
    )
    if (
        attributes.get("record_events") is not False
        or attributes.get("retain_trajectory_records") is not False
        or nonempty_buffers
    ):
        raise OpenEcologyRuntimeCheckpointError(
            "restored persistent runtime violates bounded no-retention mode; "
            f"nonempty retained buffers: {nonempty_buffers}"
        )


def _validated_persistent_runner_state(
    value: object,
    *,
    policy: DeterministicPublicRecurrentPolicy,
    binding: RuntimeCheckpointBinding,
) -> dict[str, object]:
    _validate_plain_json_value(
        value,
        field=_PERSISTENT_RUNNER_STATE_ATTRIBUTE,
    )
    cloned = _json_clone(value)
    if not isinstance(cloned, dict):
        raise OpenEcologyRuntimeCheckpointError(
            f"{_PERSISTENT_RUNNER_STATE_ATTRIBUTE} must be an object"
        )
    _exact_keys(
        cloned,
        _PERSISTENT_RUNNER_STATE_KEYS,
        field=_PERSISTENT_RUNNER_STATE_ATTRIBUTE,
    )
    if cloned["schema_version"] != _PERSISTENT_RUNNER_STATE_SCHEMA_VERSION:
        raise OpenEcologyRuntimeCheckpointError(
            "persistent runner-state schema version is stale"
        )
    _sha256(cloned["task_sha256"], field="runner_state.task_sha256")
    if cloned["model_state_sha256"] != recurrent_model_state_sha256(policy.model):
        raise OpenEcologyRuntimeCheckpointError(
            "persistent runner-state model binding does not match the fresh policy"
        )
    _sha256(
        cloned["evidence_continuation_sha256"],
        field="runner_state.evidence_continuation_sha256",
    )
    supplied_digest = _sha256(
        cloned["runner_state_sha256"],
        field="runner_state.runner_state_sha256",
    )
    if supplied_digest != _digest(
        {key: item for key, item in cloned.items() if key != "runner_state_sha256"}
    ):
        raise OpenEcologyRuntimeCheckpointError(
            "persistent runner-state digest mismatch"
        )

    observed_tick = _nonnegative_int(
        cloned["observed_tick"],
        field="runner_state.observed_tick",
    )
    completed_tick = _nonnegative_int(
        cloned["completed_world_tick"],
        field="runner_state.completed_world_tick",
    )
    genesis_boundary = observed_tick == 0 and completed_tick == 0
    completed_boundary = observed_tick > 0 and observed_tick == completed_tick + 1
    if completed_tick != binding.completed_tick or not (
        genesis_boundary or completed_boundary
    ):
        raise OpenEcologyRuntimeCheckpointError(
            "persistent runner-state tick boundary does not match checkpoint"
        )
    for field_name in (
        "episode_reset_count",
        "world_replacement_count",
        "writer_initial_event_count",
        "next_event_index",
    ):
        _nonnegative_int(cloned[field_name], field=f"runner_state.{field_name}")
    if cloned["next_event_index"] < cloned["writer_initial_event_count"]:
        raise OpenEcologyRuntimeCheckpointError(
            "persistent runner-state event index precedes its initial count"
        )
    for field_name in ("extinct", "evidence_finished", "evidence_aborted"):
        if type(cloned[field_name]) is not bool:
            raise OpenEcologyRuntimeCheckpointError(
                f"runner_state.{field_name} must be an exact boolean"
            )
    extinction_tick = cloned["extinction_tick"]
    if extinction_tick is not None:
        _nonnegative_int(extinction_tick, field="runner_state.extinction_tick")
    if (cloned["extinct"] and extinction_tick is None) or (
        not cloned["extinct"] and extinction_tick is not None
    ):
        raise OpenEcologyRuntimeCheckpointError(
            "persistent runner-state extinction flag and tick disagree"
        )
    last_evidence_tick = cloned["last_evidence_tick"]
    if last_evidence_tick is not None:
        _nonnegative_int(
            last_evidence_tick,
            field="runner_state.last_evidence_tick",
        )
    if cloned["evidence_finished"] or cloned["evidence_aborted"]:
        raise OpenEcologyRuntimeCheckpointError(
            "persistent runner-state checkpoint cannot own a closed evidence stream"
        )
    expected_container_types = {
        "interval": dict,
        "summaries": list,
        "milestones": list,
        "event_counts": dict,
    }
    for field_name, expected_type in expected_container_types.items():
        if type(cloned[field_name]) is not expected_type:
            raise OpenEcologyRuntimeCheckpointError(
                f"runner_state.{field_name} has an invalid JSON container type"
            )
    return cloned


def _validate_persistent_runner_continuation_binding(
    attributes: Mapping[str, object],
    *,
    continuation: Mapping[str, object],
) -> None:
    value = attributes.get(_PERSISTENT_RUNNER_STATE_ATTRIBUTE)
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise OpenEcologyRuntimeCheckpointError(
            f"{_PERSISTENT_RUNNER_STATE_ATTRIBUTE} must be an object"
        )
    comparisons = {
        "evidence_continuation_sha256": continuation["state_sha256"],
        "next_event_index": continuation["next_event_index"],
        "last_evidence_tick": continuation["last_tick"],
    }
    if any(value.get(field) != expected for field, expected in comparisons.items()):
        raise OpenEcologyRuntimeCheckpointError(
            "persistent runner-state does not match the evidence continuation"
        )


def _validate_plain_json_value(
    value: object,
    *,
    field: str,
    depth: int = 0,
) -> None:
    if depth > 128:
        raise OpenEcologyRuntimeCheckpointError(f"{field} nesting is too deep")
    if value is None or type(value) in {bool, str, int}:
        return
    if type(value) is float:
        _finite_float(value, field=field)
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_plain_json_value(
                item,
                field=f"{field}[{index}]",
                depth=depth + 1,
            )
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise OpenEcologyRuntimeCheckpointError(
                    f"{field} contains a non-string JSON object key"
                )
            _validate_plain_json_value(
                item,
                field=f"{field}.{key}",
                depth=depth + 1,
            )
        return
    raise OpenEcologyRuntimeCheckpointError(
        f"{field} contains non-JSON value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _validated_evidence_continuation(
    value: Mapping[str, object],
    *,
    binding: RuntimeCheckpointBinding,
    allow_lagging_tick: bool = False,
) -> dict[str, object]:
    try:
        continuation = validate_open_ecology_evidence_continuation_state(value)
    except ValueError as exc:
        raise OpenEcologyRuntimeCheckpointError(
            f"evidence writer continuation is invalid: {exc}"
        ) from exc
    last_tick = continuation["last_tick"]
    tick_mismatch = (
        last_tick != binding.completed_tick
        if not allow_lagging_tick
        else last_tick is not None and last_tick > binding.completed_tick
    )
    if tick_mismatch:
        detail = (
            "is outside the checkpoint boundary"
            if allow_lagging_tick
            else "does not match the checkpoint completed tick"
        )
        raise OpenEcologyRuntimeCheckpointError(
            f"evidence writer continuation last_tick {detail}"
        )
    return continuation


def _validate_runtime_pair(
    world: SimulationWorld,
    policy: DeterministicPublicRecurrentPolicy,
    *,
    binding: RuntimeCheckpointBinding,
    require_tick: bool = True,
) -> None:
    if not isinstance(world, SimulationWorld):
        raise TypeError("world must be a SimulationWorld")
    if not isinstance(policy, DeterministicPublicRecurrentPolicy):
        raise TypeError("policy must be a DeterministicPublicRecurrentPolicy")
    if world.policy is not policy:
        raise OpenEcologyRuntimeCheckpointError(
            "world and recurrent policy are not the same runtime pair"
        )
    if world._policy_id != policy.policy_id:
        raise OpenEcologyRuntimeCheckpointError("world policy identity drifted")
    if world._policy_version != policy.policy_version:
        raise OpenEcologyRuntimeCheckpointError("world policy version drifted")
    if require_tick and world.tick != binding.completed_tick:
        raise OpenEcologyRuntimeCheckpointError(
            "world tick does not match checkpoint binding"
        )
    if policy._model_state_sha256 != recurrent_model_state_sha256(policy.model):
        raise OpenEcologyRuntimeCheckpointError(
            "frozen recurrent model state changed before checkpoint"
        )


def _component_payload(
    binding: Mapping[str, object],
    state: Mapping[str, object],
) -> dict[str, object]:
    return {"binding": dict(binding), "state": dict(state)}


def _validate_component_binding(
    payload: Mapping[str, object],
    *,
    binding: RuntimeCheckpointBinding,
    field: str,
) -> None:
    _exact_keys(payload, {"binding", "state"}, field=field)
    if payload["binding"] != binding.to_dict():
        raise OpenEcologyRuntimeCheckpointError(
            f"{field} is not bound to the requested runtime boundary"
        )
    if not isinstance(payload["state"], Mapping):
        raise OpenEcologyRuntimeCheckpointError(f"{field}.state must be an object")


def _component_state(
    payload: Mapping[str, object],
    *,
    field: str,
) -> Mapping[str, object]:
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise OpenEcologyRuntimeCheckpointError(f"{field}.state must be an object")
    return state


def _encode_float32_tensor(tensor: Tensor) -> dict[str, object]:
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise OpenEcologyRuntimeCheckpointError(
            "recurrent hidden tensors must be CPU float32"
        )
    contiguous = tensor.detach().contiguous()
    if not bool(torch.isfinite(contiguous).all().item()):
        raise OpenEcologyRuntimeCheckpointError(
            "recurrent hidden tensor contains a non-finite value"
        )
    raw = contiguous.numpy().tobytes(order="C")
    return {
        "dtype": "float32",
        "shape": list(contiguous.shape),
        "raw_hex": raw.hex(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _decode_float32_tensor(payload: Mapping[str, object]) -> Tensor:
    _exact_keys(payload, {"dtype", "shape", "raw_hex", "raw_sha256"}, field="tensor")
    if payload["dtype"] != "float32":
        raise OpenEcologyRuntimeCheckpointError("hidden tensor dtype is unsupported")
    shape = [
        _nonnegative_int(value, field="tensor shape")
        for value in _list(payload["shape"], field="tensor shape")
    ]
    if not shape or any(value == 0 for value in shape):
        raise OpenEcologyRuntimeCheckpointError("hidden tensor shape is empty")
    raw_hex = payload["raw_hex"]
    if not isinstance(raw_hex, str) or len(raw_hex) % 2:
        raise OpenEcologyRuntimeCheckpointError("hidden tensor raw_hex is invalid")
    try:
        raw = bytes.fromhex(raw_hex)
    except ValueError as exc:
        raise OpenEcologyRuntimeCheckpointError(
            "hidden tensor raw_hex is invalid"
        ) from exc
    expected_bytes = reduce(mul, shape, 1) * 4
    if len(raw) != expected_bytes:
        raise OpenEcologyRuntimeCheckpointError(
            "hidden tensor byte count does not match shape"
        )
    if hashlib.sha256(raw).hexdigest() != _sha256(
        payload["raw_sha256"],
        field="tensor raw_sha256",
    ):
        raise OpenEcologyRuntimeCheckpointError("hidden tensor SHA256 mismatch")
    tensor = (
        torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(shape)
    )
    if not bool(torch.isfinite(tensor).all().item()):
        raise OpenEcologyRuntimeCheckpointError(
            "restored hidden tensor contains a non-finite value"
        )
    return tensor


def _encode_value(value: object, *, depth: int = 0) -> object:
    if depth > 128:
        raise OpenEcologyRuntimeCheckpointError("runtime state nesting is too deep")
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise OpenEcologyRuntimeCheckpointError(
                "runtime state contains a non-finite float"
            )
        return value
    if is_dataclass(value) and not isinstance(value, type):
        class_name = f"{type(value).__module__}.{type(value).__qualname__}"
        if class_name not in _ALLOWED_DATACLASS_TYPES:
            raise OpenEcologyRuntimeCheckpointError(
                f"runtime state dataclass {class_name!r} is not allowed"
            )
        return {
            "__runtime_type__": "dataclass",
            "class": class_name,
            "fields": {
                field.name: _encode_value(getattr(value, field.name), depth=depth + 1)
                for field in fields(value)
            },
        }
    if isinstance(value, list):
        return {
            "__runtime_type__": "list",
            "items": [_encode_value(item, depth=depth + 1) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "__runtime_type__": "tuple",
            "items": [_encode_value(item, depth=depth + 1) for item in value],
        }
    if isinstance(value, set):
        items = [_encode_value(item, depth=depth + 1) for item in value]
        items.sort(key=_canonical_text)
        return {"__runtime_type__": "set", "items": items}
    if isinstance(value, frozenset):
        items = [_encode_value(item, depth=depth + 1) for item in value]
        items.sort(key=_canonical_text)
        return {"__runtime_type__": "frozenset", "items": items}
    if isinstance(value, Mapping):
        items = [
            [
                _encode_value(key, depth=depth + 1),
                _encode_value(item, depth=depth + 1),
            ]
            for key, item in value.items()
        ]
        items.sort(key=lambda item: _canonical_text(item[0]))
        return {"__runtime_type__": "dict", "items": items}
    raise OpenEcologyRuntimeCheckpointError(
        f"runtime state type {type(value).__module__}.{type(value).__qualname__} "
        "is not JSON-adaptable"
    )


def _decode_value(value: object, *, depth: int = 0) -> object:
    if depth > 128:
        raise OpenEcologyRuntimeCheckpointError("runtime state nesting is too deep")
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        return _finite_float(value, field="runtime state float")
    if not isinstance(value, Mapping):
        raise OpenEcologyRuntimeCheckpointError(
            "encoded runtime state must use tagged objects"
        )
    tag = value.get("__runtime_type__")
    if tag == "dataclass":
        _exact_keys(
            value,
            {"__runtime_type__", "class", "fields"},
            field="encoded dataclass",
        )
        class_name = value["class"]
        if (
            not isinstance(class_name, str)
            or class_name not in _ALLOWED_DATACLASS_TYPES
        ):
            raise OpenEcologyRuntimeCheckpointError(
                "encoded runtime state names an unknown dataclass"
            )
        raw_fields = value["fields"]
        if not isinstance(raw_fields, Mapping):
            raise OpenEcologyRuntimeCheckpointError(
                "encoded dataclass fields must be an object"
            )
        cls = _ALLOWED_DATACLASS_TYPES[class_name]
        expected_fields = {field.name for field in fields(cls)}
        _exact_keys(raw_fields, expected_fields, field=f"{class_name} fields")
        kwargs = {
            name: _decode_value(item, depth=depth + 1)
            for name, item in raw_fields.items()
        }
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise OpenEcologyRuntimeCheckpointError(
                f"encoded dataclass {class_name} is invalid: {exc}"
            ) from exc
    if tag in {"list", "tuple", "set", "frozenset"}:
        _exact_keys(value, {"__runtime_type__", "items"}, field=f"encoded {tag}")
        items = [
            _decode_value(item, depth=depth + 1)
            for item in _list(value["items"], field=f"encoded {tag} items")
        ]
        if tag == "list":
            return items
        if tag == "tuple":
            return tuple(items)
        if tag == "set":
            return set(items)
        return frozenset(items)
    if tag == "dict":
        _exact_keys(value, {"__runtime_type__", "items"}, field="encoded dict")
        restored: dict[object, object] = {}
        for index, pair in enumerate(_list(value["items"], field="encoded dict items")):
            if not isinstance(pair, list) or len(pair) != 2:
                raise OpenEcologyRuntimeCheckpointError(
                    f"encoded dict item {index} is not a key/value pair"
                )
            key = _decode_value(pair[0], depth=depth + 1)
            try:
                if key in restored:
                    raise OpenEcologyRuntimeCheckpointError(
                        "encoded dict contains a duplicate key"
                    )
                restored[key] = _decode_value(pair[1], depth=depth + 1)
            except TypeError as exc:
                raise OpenEcologyRuntimeCheckpointError(
                    "encoded dict key is not hashable"
                ) from exc
        return restored
    raise OpenEcologyRuntimeCheckpointError("encoded runtime state tag is unknown")


def _bounded_payload(payload: object, ceiling: int, *, field: str) -> None:
    size = len(_canonical_text(payload).encode("utf-8"))
    if size > ceiling:
        raise OpenEcologyRuntimeCheckpointError(
            f"{field} exceeds max_state_bytes ({size} > {ceiling})"
        )


def _canonical_text(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise OpenEcologyRuntimeCheckpointError(
            f"runtime state is not canonical JSON: {exc}"
        ) from exc


def _json_clone(value: object) -> object:
    if value is None:
        return None
    try:
        return json.loads(_canonical_text(value))
    except json.JSONDecodeError as exc:  # pragma: no cover - canonical output
        raise OpenEcologyRuntimeCheckpointError(
            f"runtime state JSON clone failed: {exc}"
        ) from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_text(value).encode("utf-8")).hexdigest()


def _exact_keys(
    value: Mapping[object, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    if set(value) != expected:
        raise OpenEcologyRuntimeCheckpointError(f"{field} field set drifted")


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise OpenEcologyRuntimeCheckpointError(f"{field} must be a list")
    return value


def _identifier(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
        or len(value) > 256
    ):
        raise OpenEcologyRuntimeCheckpointError(f"{field} is not a valid identifier")
    return value


def _git_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise OpenEcologyRuntimeCheckpointError(f"{field} must be a lowercase git SHA")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OpenEcologyRuntimeCheckpointError(f"{field} must be a lowercase SHA256")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologyRuntimeCheckpointError(
            f"{field} must be a nonnegative integer"
        )
    return value


def _positive_int(value: object, *, field: str) -> int:
    parsed = _nonnegative_int(value, field=field)
    if parsed == 0:
        raise OpenEcologyRuntimeCheckpointError(f"{field} must be a positive integer")
    return parsed


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenEcologyRuntimeCheckpointError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise OpenEcologyRuntimeCheckpointError(f"{field} must be finite")
    return parsed


def _optional_finite_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field=field)
