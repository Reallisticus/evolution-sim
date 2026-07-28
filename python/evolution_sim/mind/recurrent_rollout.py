from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
    encode_observation_input,
)
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.env.runtime.trajectory import (
    REWARD_COMPONENT_BOUNDS,
    REWARD_SCHEMA_VERSION,
    REWARD_TOTAL_BOUNDS,
)
from evolution_sim.env.runtime.ticks import deterministic_agent_turn_order
from evolution_sim.env.runtime.state import empty_mind_inheritance_metadata
from evolution_sim.mind.policy_inputs import (
    ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ecological_policy_input_values,
)
from evolution_sim.mind.recurrent_genome import (
    RECURRENT_CONTROLLER_GENOME_SIZE,
    RecurrentControllerGenome,
    RecurrentGenomeError,
    zero_recurrent_genome,
)
from evolution_sim.mind.recurrent_genome_population import (
    RecurrentGenomePopulationError,
    RecurrentGenomePopulationManager,
    RecurrentGenomePopulationMode,
    recurrent_genome_stream_binding_sha256,
)


RECURRENT_ROLLOUT_POLICY_ID = "mind_v3_recurrent_on_policy"
RECURRENT_ROLLOUT_POLICY_VERSION = "mind_v3_recurrent_on_policy_v1"
RECURRENT_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION = "mind_v3_recurrent_rollout_decision_v2"
RECURRENT_GENOME_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION = (
    "mind_v3_recurrent_rollout_decision_genome_v1"
)
RECURRENT_ROLLOUT_ACTION_SOURCE = "learned_recurrent_on_policy"
RECURRENT_ROLLOUT_ACTIONS: tuple[str, ...] = tuple(ACTION_NAMES)
RECURRENT_REWARD_COMPONENTS: tuple[str, ...] = tuple(REWARD_COMPONENT_BOUNDS)
RECURRENT_PUBLIC_FEEDBACK_SCHEMA_VERSION = "mind_previous_public_feedback_v1"
RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE = len(RECURRENT_ROLLOUT_ACTIONS) * 2 + 3
RECURRENT_LEARNED_INPUT_VECTOR_SIZE = (
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE
    + len(RECURRENT_ROLLOUT_ACTIONS)
    + RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE
)
RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-ippo|policy-action-sampling-v1|2026-07-21"
)
MAX_RECURRENT_POLICY_SAMPLING_SEED = 2**63 - 1
MAX_RECURRENT_GENOME_STREAM_SEED = 2**64 - 1
_MIN_DERIVED_POLICY_SAMPLING_SEED = 2**31
_FEEDBACK_REWARD_SCALE = max(abs(bound) for bound in REWARD_TOTAL_BOUNDS)
RECURRENT_GENOME_CONDITIONING_DISABLED = "disabled"
RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1 = "actor_film_v1"
RECURRENT_FIXED_BATCH_RUNTIME_SCHEMA_VERSION = (
    "mind_v3_recurrent_intra_world_fixed_batch_runtime_v1"
)
RECURRENT_FIXED_BATCH_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION = (
    "mind_v3_recurrent_rollout_decision_fixed_batch_v1"
)
RECURRENT_GENOME_FIXED_BATCH_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION = (
    "mind_v3_recurrent_rollout_decision_genome_fixed_batch_v1"
)
RECURRENT_FIXED_BATCH_ROW_ORDER_POLICY = (
    "tick_start_seed_tick_agent_hash_permutation_v1"
)
RECURRENT_FIXED_BATCH_PADDING_POLICY = (
    "right_pad_zero_observation_zero_feedback_zero_hidden_stay_mask_v1"
)
RECURRENT_FIXED_BATCH_EXECUTION_SCOPE = "single_world_intra_tick_pure_forward_v1"
RECURRENT_FIXED_BATCH_RELEASE_STATUS = (
    "experimental_opt_in_repeatability_and_measured_topology_gate_required_v1"
)
RECURRENT_ACTION_FREE_BOOTSTRAP_SCHEMA_VERSION = (
    "mind_v3_recurrent_action_free_bootstrap_v1"
)
OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY = 320


class RecurrentRolloutError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RecurrentFixedBatchRuntimeContract:
    """Versioned deterministic contract for one world's staged model forward.

    The contract deliberately excludes action sampling and simulator mutation.
    It batches only the pure recurrent forward pass over immutable tick-start
    policy inputs. Sampling, recurrent-state commitment, and action resolution
    remain in the scalar hash-permuted turn loop.
    """

    batch_capacity: int
    schema_version: str = RECURRENT_FIXED_BATCH_RUNTIME_SCHEMA_VERSION
    row_order_policy: str = RECURRENT_FIXED_BATCH_ROW_ORDER_POLICY
    padding_policy: str = RECURRENT_FIXED_BATCH_PADDING_POLICY
    execution_scope: str = RECURRENT_FIXED_BATCH_EXECUTION_SCOPE

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_capacity, bool)
            or not isinstance(self.batch_capacity, int)
            or self.batch_capacity <= 0
        ):
            raise RecurrentRolloutError(
                "fixed recurrent batch capacity must be a positive integer"
            )
        expected = {
            "schema_version": RECURRENT_FIXED_BATCH_RUNTIME_SCHEMA_VERSION,
            "row_order_policy": RECURRENT_FIXED_BATCH_ROW_ORDER_POLICY,
            "padding_policy": RECURRENT_FIXED_BATCH_PADDING_POLICY,
            "execution_scope": RECURRENT_FIXED_BATCH_EXECUTION_SCOPE,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise RecurrentRolloutError(
                    f"fixed recurrent batch {field_name} contract drifted"
                )

    @classmethod
    def open_ecology(cls) -> RecurrentFixedBatchRuntimeContract:
        return cls(batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY)

    def as_contract(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "batch_capacity": self.batch_capacity,
            "row_order_policy": self.row_order_policy,
            "padding_policy": self.padding_policy,
            "execution_scope": self.execution_scope,
            "release_status": RECURRENT_FIXED_BATCH_RELEASE_STATUS,
            "default_enabled": False,
            "authoritative_launch_authorized": False,
            "tick_start_observations": True,
            "tick_start_action_masks": True,
            "sequential_policy_sampling": True,
            "sequential_action_resolution": True,
            "passive_staged_rows_discarded_without_sampling": True,
            "passive_staged_rows_discarded_without_hidden_commit": True,
            "cross_world_batching": False,
        }
        payload["contract_sha256"] = _stable_payload_sha256(payload)
        return payload


def derive_recurrent_policy_sampling_seed(*, task_identity: str) -> int:
    """Derive a deterministic signed-63-bit action-sampling seed.

    The namespace and caller-supplied task identity are the complete derivation
    material. Environment seeds are deliberately absent. Derived seeds live
    above the simulator's signed-31-bit seed range, making the two seed axes
    disjoint by construction rather than by chance.
    """

    if (
        not isinstance(task_identity, str)
        or not task_identity
        or task_identity != task_identity.strip()
    ):
        raise RecurrentRolloutError(
            "policy sampling task_identity must be non-empty and trimmed"
        )
    material = (f"{RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE}|{task_identity}").encode(
        "utf-8"
    )
    digest_value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    derived_span = (
        MAX_RECURRENT_POLICY_SAMPLING_SEED - _MIN_DERIVED_POLICY_SAMPLING_SEED + 1
    )
    return _MIN_DERIVED_POLICY_SAMPLING_SEED + (digest_value % derived_span)


@dataclass(frozen=True, slots=True)
class RecurrentCoreOutput:
    """Detached inference output for one public observation.

    The core owns representation learning. The collector owns the stable action
    mask and seeded sampling, so a training rollout cannot silently fall back to
    a heuristic or an invalid action.
    """

    logits: tuple[float, ...]
    value: float
    next_hidden: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PreviousPublicFeedback:
    """Previous finalized same-agent feedback visible to the recurrent core."""

    requested_action_index: int | None
    resolved_action_index: int | None
    resolution_action_valid: bool
    moved: bool
    reward_total: float

    def __post_init__(self) -> None:
        for field_name in ("requested_action_index", "resolved_action_index"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= len(RECURRENT_ROLLOUT_ACTIONS)
            ):
                raise RecurrentRolloutError(f"{field_name} is out of range")
        if (self.requested_action_index is None) != (
            self.resolved_action_index is None
        ):
            raise RecurrentRolloutError(
                "requested and resolved feedback actions must both be present "
                "or both be absent"
            )
        if type(self.resolution_action_valid) is not bool:
            raise RecurrentRolloutError("resolution_action_valid must be a boolean")
        if type(self.moved) is not bool:
            raise RecurrentRolloutError("moved must be a boolean")
        reward_total = _finite_float(self.reward_total, field="feedback reward_total")
        if not REWARD_TOTAL_BOUNDS[0] <= reward_total <= REWARD_TOTAL_BOUNDS[1]:
            raise RecurrentRolloutError(
                "feedback reward_total is outside the public reward contract"
            )
        if self.requested_action_index is None and (
            self.resolution_action_valid or self.moved or reward_total != 0.0
        ):
            raise RecurrentRolloutError("birth/reset feedback must be entirely zero")
        if self.moved and not self.resolution_action_valid:
            raise RecurrentRolloutError(
                "moved feedback requires a resolution-valid action"
            )

    @classmethod
    def zero(cls) -> PreviousPublicFeedback:
        return cls(
            requested_action_index=None,
            resolved_action_index=None,
            resolution_action_valid=False,
            moved=False,
            reward_total=0.0,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> PreviousPublicFeedback:
        requested_action = _action_name(
            record.get("requested_action"),
            field="requested_action",
        )
        resolved_action = _action_name(
            record.get("resolved_action"),
            field="resolved_action",
        )
        return cls(
            requested_action_index=RECURRENT_ROLLOUT_ACTIONS.index(requested_action),
            resolved_action_index=RECURRENT_ROLLOUT_ACTIONS.index(resolved_action),
            resolution_action_valid=_strict_bool(
                record.get("resolution_action_valid"),
                field="resolution_action_valid",
            ),
            moved=_strict_bool(record.get("moved"), field="moved"),
            reward_total=_bounded_reward_total(record),
        )

    @property
    def available(self) -> bool:
        return self.requested_action_index is not None

    def vector(self) -> tuple[float, ...]:
        requested = [0.0] * len(RECURRENT_ROLLOUT_ACTIONS)
        resolved = [0.0] * len(RECURRENT_ROLLOUT_ACTIONS)
        if self.requested_action_index is not None:
            requested[self.requested_action_index] = 1.0
        if self.resolved_action_index is not None:
            resolved[self.resolved_action_index] = 1.0
        values = (
            *requested,
            *resolved,
            float(self.resolution_action_valid),
            float(self.moved),
            self.reward_total / _FEEDBACK_REWARD_SCALE,
        )
        if len(values) != RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE:
            raise AssertionError("previous public feedback vector size drifted")
        return values


class RecurrentPolicyCore(Protocol):
    hidden_size: int
    public_input_schema_version: str
    public_input_size: int
    learned_input_size: int
    genome_conditioning_mode: str

    def initial_hidden(self) -> Sequence[float]: ...

    def forward_step(
        self,
        observation: Sequence[float],
        current_action_mask: Sequence[bool],
        previous_feedback: PreviousPublicFeedback,
        hidden: Sequence[float],
        *,
        genome_values: Sequence[float] | None = None,
    ) -> RecurrentCoreOutput: ...

    def forward_fixed_batch(
        self,
        observations: Sequence[Sequence[float]],
        current_action_masks: Sequence[Sequence[bool]],
        previous_feedback: Sequence[PreviousPublicFeedback],
        hidden: Sequence[Sequence[float]],
        *,
        batch_capacity: int,
        genome_values: Sequence[Sequence[float]] | None = None,
    ) -> Sequence[RecurrentCoreOutput]: ...

    def fixed_batch_runtime_metadata(self) -> Mapping[str, object]: ...


class TorchRecurrentPolicyCore:
    """No-grad adapter from the shared torch model to the rollout protocol."""

    def __init__(self, model: Any) -> None:
        try:
            import torch
            from evolution_sim.mind.recurrent_actor_critic import (
                GENOME_CONDITIONING_ACTOR_FILM_V1,
                GENOME_CONDITIONING_DISABLED,
                PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION,
                PREVIOUS_PUBLIC_FEEDBACK_SIZE,
                PublicRecurrentActorCritic,
            )
        except ModuleNotFoundError as exc:
            raise RecurrentRolloutError(
                "TorchRecurrentPolicyCore requires the optional Mind ML stack"
            ) from exc
        if not isinstance(model, PublicRecurrentActorCritic):
            raise RecurrentRolloutError("model must be a PublicRecurrentActorCritic")
        if PREVIOUS_PUBLIC_FEEDBACK_SIZE != RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE:
            raise RecurrentRolloutError(
                "torch and rollout previous-feedback contracts disagree"
            )
        if (
            PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION
            != RECURRENT_PUBLIC_FEEDBACK_SCHEMA_VERSION
        ):
            raise RecurrentRolloutError(
                "torch and rollout previous-feedback schema versions disagree"
            )
        expected_learned_input_size = (
            model.config.public_input_size
            + len(RECURRENT_ROLLOUT_ACTIONS)
            + RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE
        )
        if model.config.learned_encoder_input_size != expected_learned_input_size:
            raise RecurrentRolloutError(
                "torch and rollout learned input sizes disagree"
            )
        self._torch = torch
        self._model = model
        self._model.eval()
        self.public_input_schema_version = model.config.public_input_schema_version
        self.public_input_size = model.config.public_input_size
        self.learned_input_size = model.config.learned_encoder_input_size
        self.genome_conditioning_mode = model.config.genome_conditioning_mode
        if self.genome_conditioning_mode not in {
            GENOME_CONDITIONING_DISABLED,
            GENOME_CONDITIONING_ACTOR_FILM_V1,
        }:
            raise RecurrentRolloutError(
                "torch model genome conditioning mode is unsupported"
            )
        self._layers = int(model.config.recurrent_layers)
        self._layer_hidden_size = int(model.config.hidden_size)
        self.hidden_size = self._layers * self._layer_hidden_size

    def initial_hidden(self) -> tuple[float, ...]:
        state = self._model.initial_state(1)
        return tuple(
            float(value) for value in state.detach().cpu().reshape(-1).tolist()
        )

    def forward_step(
        self,
        observation: Sequence[float],
        current_action_mask: Sequence[bool],
        previous_feedback: PreviousPublicFeedback,
        hidden: Sequence[float],
        *,
        genome_values: Sequence[float] | None = None,
    ) -> RecurrentCoreOutput:
        torch = self._torch
        reference = next(self._model.parameters())
        observations = torch.tensor(
            tuple(observation),
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(1, 1, self.public_input_size)
        action_masks = torch.tensor(
            tuple(current_action_mask),
            device=reference.device,
            dtype=torch.bool,
        ).reshape(1, 1, len(RECURRENT_ROLLOUT_ACTIONS))
        feedback = torch.tensor(
            previous_feedback.vector(),
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(1, 1, RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE)
        state = torch.tensor(
            tuple(hidden),
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(self._layers, 1, self._layer_hidden_size)
        genome_tensor = (
            None
            if genome_values is None
            else torch.tensor(
                tuple(genome_values),
                device=reference.device,
                dtype=reference.dtype,
            ).reshape(1, 1, RECURRENT_CONTROLLER_GENOME_SIZE)
        )
        with torch.no_grad():
            output = self._model.forward_sequence(
                observations,
                action_masks,
                feedback,
                genome_values=genome_tensor,
                initial_state=state,
            )
        return RecurrentCoreOutput(
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

    def forward_fixed_batch(
        self,
        observations: Sequence[Sequence[float]],
        current_action_masks: Sequence[Sequence[bool]],
        previous_feedback: Sequence[PreviousPublicFeedback],
        hidden: Sequence[Sequence[float]],
        *,
        batch_capacity: int,
        genome_values: Sequence[Sequence[float]] | None = None,
    ) -> tuple[RecurrentCoreOutput, ...]:
        """Run one padded time-major forward without sampling or state commit."""

        if (
            isinstance(batch_capacity, bool)
            or not isinstance(batch_capacity, int)
            or batch_capacity <= 0
        ):
            raise RecurrentRolloutError(
                "fixed recurrent batch capacity must be a positive integer"
            )
        active_rows = len(observations)
        if active_rows <= 0 or active_rows > batch_capacity:
            raise RecurrentRolloutError(
                "fixed recurrent batch active rows must be within capacity"
            )
        if not (
            len(current_action_masks)
            == len(previous_feedback)
            == len(hidden)
            == active_rows
        ):
            raise RecurrentRolloutError(
                "fixed recurrent batch input row counts disagree"
            )
        conditioned = (
            self.genome_conditioning_mode == RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
        )
        if conditioned != (genome_values is not None):
            raise RecurrentRolloutError(
                "fixed recurrent batch genome rows disagree with core conditioning"
            )
        if genome_values is not None and len(genome_values) != active_rows:
            raise RecurrentRolloutError(
                "fixed recurrent batch genome row count disagrees"
            )

        torch = self._torch
        reference = next(self._model.parameters())
        padded_observations = torch.zeros(
            (1, batch_capacity, self.public_input_size),
            device=reference.device,
            dtype=reference.dtype,
        )
        padded_action_masks = torch.zeros(
            (1, batch_capacity, len(RECURRENT_ROLLOUT_ACTIONS)),
            device=reference.device,
            dtype=torch.bool,
        )
        padded_feedback = torch.zeros(
            (1, batch_capacity, RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE),
            device=reference.device,
            dtype=reference.dtype,
        )
        padded_state = torch.zeros(
            (self._layers, batch_capacity, self._layer_hidden_size),
            device=reference.device,
            dtype=reference.dtype,
        )
        padded_action_masks[
            0,
            active_rows:,
            RECURRENT_ROLLOUT_ACTIONS.index("stay"),
        ] = True
        padded_observations[0, :active_rows] = torch.tensor(
            tuple(tuple(row) for row in observations),
            device=reference.device,
            dtype=reference.dtype,
        )
        padded_action_masks[0, :active_rows] = torch.tensor(
            tuple(tuple(row) for row in current_action_masks),
            device=reference.device,
            dtype=torch.bool,
        )
        padded_feedback[0, :active_rows] = torch.tensor(
            tuple(feedback.vector() for feedback in previous_feedback),
            device=reference.device,
            dtype=reference.dtype,
        )
        hidden_tensor = torch.tensor(
            tuple(tuple(row) for row in hidden),
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(active_rows, self._layers, self._layer_hidden_size)
        padded_state[:, :active_rows] = hidden_tensor.permute(1, 0, 2)

        padded_genomes = None
        if genome_values is not None:
            padded_genomes = torch.zeros(
                (1, batch_capacity, RECURRENT_CONTROLLER_GENOME_SIZE),
                device=reference.device,
                dtype=reference.dtype,
            )
            padded_genomes[0, :active_rows] = torch.tensor(
                tuple(tuple(row) for row in genome_values),
                device=reference.device,
                dtype=reference.dtype,
            )
        with torch.no_grad():
            output = self._model.forward_sequence(
                padded_observations,
                padded_action_masks,
                padded_feedback,
                genome_values=padded_genomes,
                initial_state=padded_state,
            )
        final_state = output.final_state.detach().cpu()
        raw_logits = output.raw_logits[0].detach().cpu()
        values = output.values[0].detach().cpu()
        return tuple(
            RecurrentCoreOutput(
                logits=tuple(float(value) for value in raw_logits[row].tolist()),
                value=float(values[row].item()),
                next_hidden=tuple(
                    float(value)
                    for value in final_state[:, row, :].reshape(-1).tolist()
                ),
            )
            for row in range(active_rows)
        )

    def fixed_batch_runtime_metadata(self) -> dict[str, object]:
        """Observed numerical/runtime binding for same-contract reproduction."""

        torch = self._torch
        reference = next(self._model.parameters())
        cuda_version = getattr(torch.version, "cuda", None)
        cudnn_version = (
            torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
        )
        cuda_matmul = getattr(torch.backends, "cuda", None)
        matmul_backend = (
            getattr(cuda_matmul, "matmul", None) if cuda_matmul is not None else None
        )
        cpu_backend = getattr(torch.backends, "cpu", None)
        cpu_capability = (
            cpu_backend.get_cpu_capability()
            if cpu_backend is not None
            and callable(getattr(cpu_backend, "get_cpu_capability", None))
            else None
        )
        parallel_backend = _torch_parallel_backend(torch)
        build_config = str(torch.__config__.show())
        cpu_architecture = platform.machine().strip()
        if not cpu_architecture or not build_config:
            raise RecurrentRolloutError(
                "CPU architecture and Torch build identity are required for "
                "fixed-batch provenance"
            )
        return {
            "implementation": (
                "PublicRecurrentActorCritic.forward_sequence_time1_batch_v1"
            ),
            "device_type": reference.device.type,
            "device_index": reference.device.index,
            "dtype": str(reference.dtype),
            "torch_version": str(torch.__version__),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "cpu_architecture": cpu_architecture,
            "cpu_capability": (None if cpu_capability is None else str(cpu_capability)),
            "aten_parallel_backend": parallel_backend,
            "torch_build_config_sha256": hashlib.sha256(
                build_config.encode("utf-8")
            ).hexdigest(),
            "cuda_version": None if cuda_version is None else str(cuda_version),
            "cudnn_version": cudnn_version,
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(
                getattr(
                    getattr(torch.backends, "cudnn", object()), "deterministic", False
                )
            ),
            "cudnn_benchmark": bool(
                getattr(getattr(torch.backends, "cudnn", object()), "benchmark", False)
            ),
            "cuda_matmul_allow_tf32": (
                None
                if matmul_backend is None
                else bool(getattr(matmul_backend, "allow_tf32", False))
            ),
            "cudnn_allow_tf32": (
                None
                if not hasattr(torch.backends, "cudnn")
                else bool(getattr(torch.backends.cudnn, "allow_tf32", False))
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }


@dataclass(frozen=True, slots=True)
class StagedRecurrentDecision:
    tick: int
    turn_rank: int
    active_rows: int
    observation_snapshot: dict[str, object]
    policy_input: tuple[float, ...]
    previous_feedback: PreviousPublicFeedback
    action_mask: tuple[bool, ...]
    hidden: tuple[float, ...]
    output: RecurrentCoreOutput
    genome_values: tuple[float, ...] | None
    genome_sha256: str | None
    genome_stream_seed: int | None


@dataclass(frozen=True, slots=True)
class PendingDecision:
    world_id: str
    environment_seed: int
    policy_sampling_seed: int
    tick_phase: str
    agent_id: int
    decision_index: int
    observation: tuple[float, ...]
    previous_feedback: PreviousPublicFeedback
    action_mask: tuple[bool, ...]
    action_index: int
    requested_action: str
    logprob: float
    entropy: float
    value: float
    hidden: tuple[float, ...]
    next_hidden: tuple[float, ...]
    genome_values: tuple[float, ...] | None = None
    genome_sha256: str | None = None
    genome_stream_seed: int | None = None
    fixed_batch_runtime_sha256: str | None = None
    fixed_batch_turn_rank: int | None = None
    fixed_batch_active_rows: int | None = None
    fixed_batch_capacity: int | None = None

    def __post_init__(self) -> None:
        genome_values, genome_sha256, genome_stream_seed = (
            _validated_rollout_genome_fields(
                genome_values=self.genome_values,
                genome_sha256=self.genome_sha256,
                genome_stream_seed=self.genome_stream_seed,
                field="pending decision",
            )
        )
        object.__setattr__(self, "genome_values", genome_values)
        object.__setattr__(self, "genome_sha256", genome_sha256)
        object.__setattr__(self, "genome_stream_seed", genome_stream_seed)
        _validate_fixed_batch_decision_fields(
            runtime_sha256=self.fixed_batch_runtime_sha256,
            turn_rank=self.fixed_batch_turn_rank,
            active_rows=self.fixed_batch_active_rows,
            capacity=self.fixed_batch_capacity,
            field="pending decision",
        )


@dataclass(frozen=True, slots=True)
class RecurrentRolloutStep:
    """One policy decision aligned to its finalized simulator transition."""

    world_id: str
    world_seed: int
    tick: int
    agent_id: int
    decision_index: int
    observation: tuple[float, ...]
    previous_feedback: PreviousPublicFeedback
    action_mask: tuple[bool, ...]
    hidden: tuple[float, ...]
    action_index: int
    requested_action: str
    logprob: float
    entropy: float
    value: float
    reward: float
    reward_components: dict[str, float]
    resolved_action_index: int
    resolved_action: str
    resolution_action_mask: tuple[bool, ...]
    action_valid: bool
    resolution_action_valid: bool
    moved: bool
    outcome: dict[str, object]
    terminated: bool = False
    truncated: bool = False
    bootstrap_value: float | None = None
    passive_terminal_reward: float = 0.0
    passive_terminal_reward_components: dict[str, float] = field(default_factory=dict)
    passive_terminal_tick: int | None = None
    environment_seed: int | None = None
    policy_sampling_seed: int | None = None
    genome_values: tuple[float, ...] | None = None
    genome_sha256: str | None = None
    genome_stream_seed: int | None = None
    fixed_batch_runtime_sha256: str | None = None
    fixed_batch_turn_rank: int | None = None
    fixed_batch_active_rows: int | None = None
    fixed_batch_capacity: int | None = None

    def __post_init__(self) -> None:
        world_seed = _strict_int(self.world_seed, field="world_seed")
        environment_seed = (
            world_seed
            if self.environment_seed is None
            else _strict_int(self.environment_seed, field="environment_seed")
        )
        if environment_seed != world_seed:
            raise RecurrentRolloutError(
                "environment_seed must match the legacy world_seed alias"
            )
        policy_sampling_seed = self.policy_sampling_seed
        if policy_sampling_seed is None:
            policy_sampling_seed = derive_recurrent_policy_sampling_seed(
                task_identity=f"legacy-rollout-step:{self.world_id}"
            )
        policy_sampling_seed = _policy_sampling_seed(
            policy_sampling_seed,
            field="policy_sampling_seed",
        )
        object.__setattr__(self, "world_seed", world_seed)
        object.__setattr__(self, "environment_seed", environment_seed)
        object.__setattr__(self, "policy_sampling_seed", policy_sampling_seed)
        genome_values, genome_sha256, genome_stream_seed = (
            _validated_rollout_genome_fields(
                genome_values=self.genome_values,
                genome_sha256=self.genome_sha256,
                genome_stream_seed=self.genome_stream_seed,
                field="rollout step",
            )
        )
        object.__setattr__(self, "genome_values", genome_values)
        object.__setattr__(self, "genome_sha256", genome_sha256)
        object.__setattr__(self, "genome_stream_seed", genome_stream_seed)
        _validate_fixed_batch_decision_fields(
            runtime_sha256=self.fixed_batch_runtime_sha256,
            turn_rank=self.fixed_batch_turn_rank,
            active_rows=self.fixed_batch_active_rows,
            capacity=self.fixed_batch_capacity,
            field="rollout step",
        )
        reward, reward_components = _validated_reward_values(
            self.reward,
            self.reward_components,
            field="rollout reward",
        )
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "reward_components", reward_components)
        if self.passive_terminal_tick is None:
            if self.passive_terminal_reward != 0.0:
                raise RecurrentRolloutError(
                    "passive terminal reward requires a passive terminal tick"
                )
            if self.passive_terminal_reward_components:
                raise RecurrentRolloutError(
                    "passive terminal reward components require a passive terminal tick"
                )
        else:
            passive_reward, passive_components = _validated_reward_values(
                self.passive_terminal_reward,
                self.passive_terminal_reward_components,
                field="passive terminal reward",
            )
            object.__setattr__(self, "passive_terminal_reward", passive_reward)
            object.__setattr__(
                self,
                "passive_terminal_reward_components",
                passive_components,
            )


@dataclass(frozen=True, slots=True)
class RecurrentAdvantageRow:
    step: RecurrentRolloutStep
    advantage: float
    return_target: float


class RecurrentRolloutBuffer:
    """Sequence-preserving rollout storage with agent-local GAE boundaries."""

    def __init__(self) -> None:
        self._steps: list[RecurrentRolloutStep] = []
        self._indices_by_agent: dict[tuple[str, int], list[int]] = {}
        self._world_ids: set[str] = set()
        self._world_seed_provenance: dict[str, dict[str, object] | None] = {}

    @property
    def steps(self) -> tuple[RecurrentRolloutStep, ...]:
        return tuple(self._steps)

    @property
    def world_seed_provenance(self) -> dict[str, dict[str, object]]:
        return {
            world_id: copy.deepcopy(provenance)
            for world_id, provenance in self._world_seed_provenance.items()
            if provenance is not None
        }

    def register_world(
        self,
        world_id: str,
        *,
        genome_world_identity: str | None = None,
        environment_seed: int | None = None,
        policy_sampling_seed: int | None = None,
        genome_conditioning_mode: str | None = None,
        genome_population_mode: RecurrentGenomePopulationMode | str | None = None,
        genome_stream_seed: int | None = None,
        genome_population_binding_sha256: str | None = None,
        genome_population_pre_founder_state_sha256: str | None = None,
        fixed_batch_runtime: Mapping[str, object] | None = None,
    ) -> None:
        if not world_id:
            raise RecurrentRolloutError("world_id must not be empty")
        if world_id in self._world_ids:
            raise RecurrentRolloutError(f"world_id already collected: {world_id!r}")
        if (environment_seed is None) != (policy_sampling_seed is None):
            raise RecurrentRolloutError(
                "world seed provenance requires both environment and policy seeds"
            )
        genome_provenance = (
            genome_conditioning_mode,
            genome_population_mode,
            genome_stream_seed,
            genome_population_binding_sha256,
            genome_population_pre_founder_state_sha256,
        )
        if any(value is not None for value in genome_provenance) and not all(
            value is not None for value in genome_provenance
        ):
            raise RecurrentRolloutError(
                "world genome provenance requires conditioning mode, population "
                "mode, stream seed, binding SHA256, and pre-founder state SHA256"
            )
        provenance: dict[str, object] | None = None
        if environment_seed is not None and policy_sampling_seed is not None:
            provenance = {
                "environment_seed": _strict_int(
                    environment_seed,
                    field="environment_seed",
                ),
                "policy_sampling_seed": _policy_sampling_seed(
                    policy_sampling_seed,
                    field="policy_sampling_seed",
                ),
            }
            if genome_conditioning_mode is not None:
                resolved_genome_world_identity = (
                    world_id if genome_world_identity is None else genome_world_identity
                )
                if (
                    not isinstance(resolved_genome_world_identity, str)
                    or not resolved_genome_world_identity
                    or resolved_genome_world_identity
                    != resolved_genome_world_identity.strip()
                ):
                    raise RecurrentRolloutError(
                        "genome_world_identity must be non-empty and trimmed"
                    )
                parsed_conditioning_mode = _genome_conditioning_mode(
                    genome_conditioning_mode
                )
                if (
                    parsed_conditioning_mode
                    != RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
                ):
                    raise RecurrentRolloutError(
                        "registered genome provenance requires actor_film_v1 "
                        "conditioning"
                    )
                parsed_population_mode = _active_genome_population_mode(
                    genome_population_mode
                )
                parsed_stream_seed = _genome_stream_seed(
                    genome_stream_seed,
                    field="genome_stream_seed",
                )
                parsed_binding_sha256 = _sha256(
                    genome_population_binding_sha256,
                    field="genome_population_binding_sha256",
                )
                try:
                    expected_binding_sha256 = recurrent_genome_stream_binding_sha256(
                        genome_stream_seed=parsed_stream_seed,
                        world_identity=resolved_genome_world_identity,
                    )
                except RecurrentGenomePopulationError as exc:
                    raise RecurrentRolloutError(
                        f"world genome provenance binding is invalid: {exc}"
                    ) from exc
                if parsed_binding_sha256 != expected_binding_sha256:
                    raise RecurrentRolloutError(
                        "world genome provenance binding SHA256 does not match "
                        "its stream seed and world identity"
                    )
                provenance.update(
                    {
                        **(
                            {"genome_world_identity": (resolved_genome_world_identity)}
                            if resolved_genome_world_identity != world_id
                            else {}
                        ),
                        "genome_conditioning_mode": parsed_conditioning_mode,
                        "genome_population_mode": parsed_population_mode.value,
                        "genome_stream_seed": parsed_stream_seed,
                        "genome_population_binding_sha256": parsed_binding_sha256,
                        "genome_population_pre_founder_state_sha256": _sha256(
                            genome_population_pre_founder_state_sha256,
                            field="genome_population_pre_founder_state_sha256",
                        ),
                    }
                )
            if fixed_batch_runtime is not None:
                provenance["fixed_batch_runtime"] = (
                    _validated_fixed_batch_runtime_binding(fixed_batch_runtime)
                )
        elif any(value is not None for value in genome_provenance):
            raise RecurrentRolloutError(
                "world genome provenance requires world seed provenance"
            )
        elif fixed_batch_runtime is not None:
            raise RecurrentRolloutError(
                "fixed batch runtime provenance requires world seed provenance"
            )
        self._world_ids.add(world_id)
        self._world_seed_provenance[world_id] = provenance

    def finalize_world_genome_provenance(
        self,
        world_id: str,
        *,
        final_state_sha256: str,
        reset_state_sha256: str,
    ) -> None:
        provenance = self._world_seed_provenance.get(world_id)
        if provenance is None:
            raise RecurrentRolloutError(
                f"world has no registered seed provenance: {world_id!r}"
            )
        if (
            provenance.get("genome_conditioning_mode")
            != RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
        ):
            raise RecurrentRolloutError(
                "cannot finalize genome provenance for an unconditioned world"
            )
        if "genome_population_final_state_sha256" in provenance:
            raise RecurrentRolloutError("world genome provenance was already finalized")
        parsed_final_state_sha256 = _sha256(
            final_state_sha256,
            field="genome_population_final_state_sha256",
        )
        parsed_reset_state_sha256 = _sha256(
            reset_state_sha256,
            field="genome_population_reset_state_sha256",
        )
        if parsed_reset_state_sha256 != provenance.get(
            "genome_population_pre_founder_state_sha256"
        ):
            raise RecurrentRolloutError(
                "world genome reset state SHA256 must match its pre-founder empty state"
            )
        provenance.update(
            {
                "genome_population_final_state_sha256": (parsed_final_state_sha256),
                "genome_population_reset_state_sha256": (parsed_reset_state_sha256),
            }
        )

    def append(self, step: RecurrentRolloutStep) -> None:
        if step.world_id not in self._world_ids:
            raise RecurrentRolloutError(
                f"rollout world was not registered: {step.world_id!r}"
            )
        observed_provenance = {
            "environment_seed": int(step.environment_seed),
            "policy_sampling_seed": int(step.policy_sampling_seed),
        }
        registered_provenance = self._world_seed_provenance[step.world_id]
        if registered_provenance is None:
            self._world_seed_provenance[step.world_id] = observed_provenance
            registered_provenance = observed_provenance
        elif any(
            observed_provenance[key] != registered_provenance.get(key)
            for key in observed_provenance
        ):
            raise RecurrentRolloutError(
                "rollout step seed provenance does not match its registered world"
            )
        registered_conditioning_mode = registered_provenance.get(
            "genome_conditioning_mode"
        )
        fixed_batch_runtime = registered_provenance.get("fixed_batch_runtime")
        if fixed_batch_runtime is None:
            if step.fixed_batch_runtime_sha256 is not None:
                raise RecurrentRolloutError(
                    "fixed-batch rollout step requires registered runtime provenance"
                )
        else:
            if not isinstance(fixed_batch_runtime, Mapping):
                raise RecurrentRolloutError(
                    "registered fixed batch runtime provenance is invalid"
                )
            if step.fixed_batch_runtime_sha256 != fixed_batch_runtime.get(
                "exact_digest"
            ) or step.fixed_batch_capacity != fixed_batch_runtime.get("batch_capacity"):
                raise RecurrentRolloutError(
                    "fixed-batch rollout step disagrees with registered runtime"
                )
        if step.genome_values is None:
            if registered_conditioning_mode is not None:
                raise RecurrentRolloutError(
                    "unconditioned rollout step cannot enter a genome-conditioned world"
                )
        else:
            if (
                registered_conditioning_mode
                != RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
            ):
                raise RecurrentRolloutError(
                    "genome-conditioned rollout step requires matching registered "
                    "world provenance"
                )
            if step.genome_stream_seed != registered_provenance.get(
                "genome_stream_seed"
            ):
                raise RecurrentRolloutError(
                    "rollout step genome stream seed does not match its registered "
                    "world"
                )
            if (
                registered_provenance.get("genome_population_mode")
                == RecurrentGenomePopulationMode.ZERO_ALL.value
                and step.genome_values != zero_recurrent_genome().values
            ):
                raise RecurrentRolloutError(
                    "zero_all rollout world contains a nonzero controller genome"
                )
        key = (step.world_id, step.agent_id)
        indices = self._indices_by_agent.setdefault(key, [])
        if indices:
            previous = self._steps[indices[-1]]
            if previous.terminated or previous.truncated:
                raise RecurrentRolloutError(
                    "cannot append after an agent episode boundary: "
                    f"world={step.world_id!r} agent={step.agent_id}"
                )
            if step.tick <= previous.tick:
                raise RecurrentRolloutError(
                    "agent rollout ticks must increase strictly: "
                    f"previous={previous.tick} current={step.tick}"
                )
        if step.terminated and step.truncated:
            raise RecurrentRolloutError("a rollout step cannot terminate and truncate")
        indices.append(len(self._steps))
        self._steps.append(step)

    def mark_passive_terminal(
        self,
        *,
        world_id: str,
        agent_id: int,
        tick: int,
        reward: float,
        reward_components: Mapping[str, object],
    ) -> bool:
        """Attach a no-action terminal transition without fabricating an action.

        An agent can be killed before its turn. The simulator then emits a
        passive terminal trajectory record, but there is no policy log-prob for
        that record. Its reward is kept separately and discounted after the
        agent's last real action during GAE.
        """

        key = (world_id, agent_id)
        indices = self._indices_by_agent.get(key)
        if not indices:
            return False
        index = indices[-1]
        step = self._steps[index]
        if step.terminated or step.truncated:
            raise RecurrentRolloutError(
                "passive terminal arrived after a closed agent rollout"
            )
        if tick != step.tick + 1:
            raise RecurrentRolloutError(
                "passive terminal tick must immediately follow the last policy "
                f"decision: previous={step.tick} terminal={tick}"
            )
        reward_total, components = _validated_reward_values(
            reward,
            reward_components,
            field="passive terminal reward",
        )
        self._steps[index] = replace(
            step,
            terminated=True,
            passive_terminal_reward=reward_total,
            passive_terminal_reward_components=components,
            passive_terminal_tick=int(tick),
        )
        return True

    def mark_truncated(
        self,
        *,
        world_id: str,
        agent_id: int,
        bootstrap_value: float,
    ) -> bool:
        return self.mark_truncated_batch(
            world_id=world_id,
            bootstrap_values_by_agent={agent_id: bootstrap_value},
        )[agent_id]

    def mark_truncated_batch(
        self,
        *,
        world_id: str,
        bootstrap_values_by_agent: Mapping[int, object],
    ) -> dict[int, bool]:
        """Validate every bootstrap row before committing any truncation."""

        planned: list[tuple[int, RecurrentRolloutStep]] = []
        target_eligibility: dict[int, bool] = {}
        for raw_agent_id, raw_bootstrap_value in bootstrap_values_by_agent.items():
            agent_id = _strict_int(
                raw_agent_id,
                field="bootstrap agent_id",
            )
            key = (world_id, agent_id)
            indices = self._indices_by_agent.get(key)
            if not indices:
                target_eligibility[agent_id] = False
                continue
            index = indices[-1]
            step = self._steps[index]
            if step.terminated:
                target_eligibility[agent_id] = False
                continue
            if step.truncated:
                raise RecurrentRolloutError("agent rollout was already truncated")
            planned.append(
                (
                    index,
                    replace(
                        step,
                        truncated=True,
                        bootstrap_value=_finite_float(
                            raw_bootstrap_value,
                            field="bootstrap value",
                        ),
                    ),
                )
            )
            target_eligibility[agent_id] = True
        for index, replacement in planned:
            self._steps[index] = replacement
        return target_eligibility

    def validate_world_closed(self, world_id: str) -> None:
        open_agents = [
            agent_id
            for (
                candidate_world_id,
                agent_id,
            ), indices in self._indices_by_agent.items()
            if candidate_world_id == world_id
            and indices
            and not (
                self._steps[indices[-1]].terminated
                or self._steps[indices[-1]].truncated
            )
        ]
        if open_agents:
            raise RecurrentRolloutError(
                "world rollout has unclosed agent sequences; finalize the "
                "action-free policy-visible tick-start bootstrap: "
                f"{sorted(open_agents)}"
            )

    def sequences(self) -> tuple[tuple[RecurrentRolloutStep, ...], ...]:
        ordered = sorted(
            self._indices_by_agent.values(),
            key=lambda indices: indices[0],
        )
        return tuple(
            tuple(self._steps[index] for index in indices) for indices in ordered
        )

    def compute_gae(
        self,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[RecurrentAdvantageRow, ...]:
        gamma = _unit_interval(gamma, field="gamma", allow_zero=True)
        gae_lambda = _unit_interval(
            gae_lambda,
            field="gae_lambda",
            allow_zero=True,
        )
        advantages: dict[int, float] = {}
        returns: dict[int, float] = {}
        for indices in self._indices_by_agent.values():
            if not indices:
                continue
            last = self._steps[indices[-1]]
            if not (last.terminated or last.truncated):
                raise RecurrentRolloutError(
                    "cannot compute GAE for an open agent sequence"
                )
            next_advantage = 0.0
            for position in range(len(indices) - 1, -1, -1):
                index = indices[position]
                step = self._steps[index]
                is_last = position == len(indices) - 1
                if is_last and step.truncated:
                    if step.bootstrap_value is None:
                        raise RecurrentRolloutError(
                            "truncated rollout is missing a bootstrap value"
                        )
                    next_value = step.bootstrap_value
                    continuation = 1.0
                elif is_last:
                    next_value = 0.0
                    continuation = 0.0
                else:
                    next_value = self._steps[indices[position + 1]].value
                    continuation = 1.0

                effective_reward = step.reward
                if is_last and step.passive_terminal_tick is not None:
                    effective_reward += gamma * step.passive_terminal_reward
                delta = (
                    effective_reward + gamma * continuation * next_value - step.value
                )
                advantage = delta + gamma * gae_lambda * continuation * next_advantage
                advantages[index] = advantage
                returns[index] = advantage + step.value
                next_advantage = advantage

        return tuple(
            RecurrentAdvantageRow(
                step=step,
                advantage=advantages[index],
                return_target=returns[index],
            )
            for index, step in enumerate(self._steps)
        )


class RecurrentOnPolicyCollector:
    """Policy-induced recurrent rollout collector for shared-policy PPO/IPPO.

    `SimulationWorld` finalizes transitions only after the complete multi-agent
    tick. `observe_transition` is therefore the alignment boundary: requested
    action, old log-prob/value, public input, reward, and resolved outcome are
    joined only there.

    For an unbiased finite-horizon truncation, run exactly `rollout_ticks`
    scored ticks, prepare the next tick's policy-visible state with
    `SimulationWorld.prepare_policy_visible_tick_start`, then call
    `finalize_action_free_bootstrap`. No bootstrap action is sampled, committed,
    or resolved.
    """

    policy_id = RECURRENT_ROLLOUT_POLICY_ID
    policy_version = RECURRENT_ROLLOUT_POLICY_VERSION

    def __init__(
        self,
        core: RecurrentPolicyCore,
        *,
        buffer: RecurrentRolloutBuffer | None = None,
        reset_recurrent_state_each_decision: bool = False,
        fixed_batch_contract: RecurrentFixedBatchRuntimeContract | None = None,
    ) -> None:
        if isinstance(core.hidden_size, bool) or int(core.hidden_size) <= 0:
            raise RecurrentRolloutError("core.hidden_size must be positive")
        if type(reset_recurrent_state_each_decision) is not bool:
            raise RecurrentRolloutError(
                "reset_recurrent_state_each_decision must be an exact boolean"
            )
        self._core = core
        self._genome_conditioning_mode = _genome_conditioning_mode(
            getattr(
                core,
                "genome_conditioning_mode",
                RECURRENT_GENOME_CONDITIONING_DISABLED,
            )
        )
        self._public_input_schema_version = getattr(
            core,
            "public_input_schema_version",
            ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        )
        self._public_input_size = getattr(
            core,
            "public_input_size",
            ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        )
        self._learned_input_size = getattr(
            core,
            "learned_input_size",
            (
                self._public_input_size
                + len(RECURRENT_ROLLOUT_ACTIONS)
                + RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE
            ),
        )
        if self._public_input_schema_version not in {
            ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
            TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        }:
            raise RecurrentRolloutError("core public input schema is unsupported")
        if (
            isinstance(self._public_input_size, bool)
            or not isinstance(self._public_input_size, int)
            or self._public_input_size <= 0
        ):
            raise RecurrentRolloutError("core public input size must be positive")
        expected_learned_input_size = (
            self._public_input_size
            + len(RECURRENT_ROLLOUT_ACTIONS)
            + RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE
        )
        if self._learned_input_size != expected_learned_input_size:
            raise RecurrentRolloutError(
                "core learned input size does not match its public input contract"
            )
        self.buffer = buffer if buffer is not None else RecurrentRolloutBuffer()
        self._reset_recurrent_state_each_decision = reset_recurrent_state_each_decision
        if fixed_batch_contract is not None and not isinstance(
            fixed_batch_contract,
            RecurrentFixedBatchRuntimeContract,
        ):
            raise RecurrentRolloutError(
                "fixed_batch_contract must be a RecurrentFixedBatchRuntimeContract"
            )
        self._fixed_batch_contract = fixed_batch_contract
        self._fixed_batch_runtime = (
            None
            if fixed_batch_contract is None
            else _fixed_batch_runtime_binding(
                fixed_batch_contract,
                core=core,
            )
        )
        self._active_world_id: str | None = None
        self._genome_world_identity: str | None = None
        self._environment_seed = 0
        self._policy_sampling_seed = 0
        self._rollout_ticks = 0
        self._bootstrap_phase = False
        self._rng = random.Random()
        self._decision_index = 0
        self._hidden_by_agent: dict[int, tuple[float, ...]] = {}
        self._feedback_by_agent: dict[int, PreviousPublicFeedback] = {}
        self._pending_by_agent: dict[int, PendingDecision] = {}
        self._genome_population_manager: RecurrentGenomePopulationManager | None = None
        self._staged_tick: int | None = None
        self._staged_by_agent: dict[int, StagedRecurrentDecision] = {}
        self._bootstrap_value_rows: dict[int, dict[str, object]] = {}

    @property
    def last_bootstrap_evidence(self) -> dict[str, object]:
        """Return canonical evidence for the active world's finite boundary."""

        self._require_active_world()
        return copy.deepcopy(self._build_bootstrap_evidence())

    @property
    def genome_conditioning_mode(self) -> str:
        return self._genome_conditioning_mode

    def start_world(
        self,
        *,
        world_id: str,
        genome_world_identity: str | None = None,
        rollout_ticks: int,
        environment_seed: int | None = None,
        policy_sampling_seed: int | None = None,
        seed: int | None = None,
        genome_stream_seed: int | None = None,
        genome_population_mode: RecurrentGenomePopulationMode | str | None = None,
    ) -> None:
        if self._active_world_id is not None:
            raise RecurrentRolloutError(
                "finish the active world before starting another"
            )
        self._validate_core_conditioning_mode_binding()
        if environment_seed is None:
            environment_seed = seed
        elif seed is not None and seed != environment_seed:
            raise RecurrentRolloutError(
                "environment_seed and legacy seed alias disagree"
            )
        if isinstance(environment_seed, bool) or not isinstance(
            environment_seed,
            int,
        ):
            raise RecurrentRolloutError("environment_seed must be an integer")
        if policy_sampling_seed is None:
            policy_sampling_seed = derive_recurrent_policy_sampling_seed(
                task_identity=f"legacy-world-id:{world_id}"
            )
        policy_sampling_seed = _policy_sampling_seed(
            policy_sampling_seed,
            field="policy_sampling_seed",
        )
        if isinstance(rollout_ticks, bool) or int(rollout_ticks) <= 0:
            raise RecurrentRolloutError("rollout_ticks must be positive")
        genome_population_manager: RecurrentGenomePopulationManager | None = None
        resolved_genome_world_identity = (
            world_id if genome_world_identity is None else genome_world_identity
        )
        if (
            not isinstance(resolved_genome_world_identity, str)
            or not resolved_genome_world_identity
            or resolved_genome_world_identity != resolved_genome_world_identity.strip()
        ):
            raise RecurrentRolloutError(
                "genome_world_identity must be non-empty and trimmed"
            )
        if self._genome_conditioning_mode == RECURRENT_GENOME_CONDITIONING_DISABLED:
            if genome_stream_seed is not None:
                raise RecurrentRolloutError(
                    "disabled recurrent core cannot accept a genome stream seed"
                )
            if (
                genome_population_mode is not None
                and genome_population_mode != RecurrentGenomePopulationMode.DISABLED
                and genome_population_mode
                != RecurrentGenomePopulationMode.DISABLED.value
            ):
                raise RecurrentRolloutError(
                    "disabled recurrent core cannot use an active genome population"
                )
        else:
            parsed_genome_stream_seed = _genome_stream_seed(
                genome_stream_seed,
                field="genome_stream_seed",
            )
            parsed_population_mode = _active_genome_population_mode(
                genome_population_mode
            )
            try:
                genome_population_manager = RecurrentGenomePopulationManager(
                    genome_stream_seed=parsed_genome_stream_seed,
                    world_identity=resolved_genome_world_identity,
                    mode=parsed_population_mode,
                )
            except RecurrentGenomePopulationError as exc:
                raise RecurrentRolloutError(
                    f"recurrent genome population setup failed: {exc}"
                ) from exc
        genome_provenance: dict[str, object] = {}
        if genome_population_manager is not None:
            genome_provenance = {
                "genome_conditioning_mode": self._genome_conditioning_mode,
                "genome_population_mode": genome_population_manager.mode,
                "genome_stream_seed": genome_population_manager.genome_stream_seed,
                "genome_population_binding_sha256": (
                    genome_population_manager.binding_sha256
                ),
                "genome_population_pre_founder_state_sha256": (
                    genome_population_manager.state_sha256
                ),
            }
        self.buffer.register_world(
            world_id,
            genome_world_identity=(
                resolved_genome_world_identity
                if resolved_genome_world_identity != world_id
                else None
            ),
            environment_seed=environment_seed,
            policy_sampling_seed=policy_sampling_seed,
            fixed_batch_runtime=self._fixed_batch_runtime,
            **genome_provenance,
        )
        self._active_world_id = world_id
        self._genome_world_identity = resolved_genome_world_identity
        self._environment_seed = int(environment_seed)
        self._policy_sampling_seed = policy_sampling_seed
        self._rollout_ticks = int(rollout_ticks)
        self._bootstrap_phase = False
        self._rng = random.Random(policy_sampling_seed)
        self._decision_index = 0
        self._hidden_by_agent.clear()
        self._feedback_by_agent.clear()
        self._pending_by_agent.clear()
        self._staged_tick = None
        self._staged_by_agent.clear()
        self._bootstrap_value_rows.clear()
        self._genome_population_manager = genome_population_manager

    def contextual_founder_metadata(
        self,
        *,
        agent_id: int,
        trophic_role: object | None = None,
        meat_mode: object | None = None,
    ) -> dict[str, object]:
        del trophic_role, meat_mode
        return self.founder_metadata(agent_id=agent_id)

    def founder_metadata(self, *, agent_id: int) -> dict[str, object]:
        self._require_active_world()
        if self._genome_conditioning_mode == RECURRENT_GENOME_CONDITIONING_DISABLED:
            return empty_mind_inheritance_metadata()
        self._require_zero_newborn_runtime_state(agent_id)
        manager = self._require_genome_population_manager()
        try:
            return manager.founder_metadata(agent_id=agent_id)
        except RecurrentGenomePopulationError as exc:
            raise RecurrentRolloutError(
                f"recurrent founder genome registration failed: {exc}"
            ) from exc

    def child_metadata(
        self,
        *,
        child_agent_id: int,
        primary_parent_id: int,
        secondary_parent_id: int | None,
    ) -> dict[str, object]:
        self._require_active_world()
        if self._genome_conditioning_mode == RECURRENT_GENOME_CONDITIONING_DISABLED:
            return empty_mind_inheritance_metadata()
        self._require_zero_newborn_runtime_state(child_agent_id)
        manager = self._require_genome_population_manager()
        try:
            return manager.child_metadata(
                child_agent_id=child_agent_id,
                primary_parent_id=primary_parent_id,
                secondary_parent_id=secondary_parent_id,
            )
        except RecurrentGenomePopulationError as exc:
            raise RecurrentRolloutError(
                f"recurrent child genome registration failed: {exc}"
            ) from exc

    def finish_world(self) -> None:
        world_id = self._require_active_world()
        if self._staged_tick is not None or self._staged_by_agent:
            raise RecurrentRolloutError(
                "world finished with an unclosed fixed recurrent batch tick"
            )
        if self._pending_by_agent:
            raise RecurrentRolloutError(
                "world finished with decisions that have no finalized transition"
            )
        self.buffer.validate_world_closed(world_id)
        manager = self._genome_population_manager
        if manager is not None:
            final_state_sha256 = manager.state_sha256
            self.buffer.finalize_world_genome_provenance(
                world_id,
                final_state_sha256=final_state_sha256,
                reset_state_sha256=manager.empty_state_sha256,
            )
            manager.reset()
            if manager.state_sha256 != manager.empty_state_sha256:
                raise RecurrentRolloutError(
                    "recurrent genome population reset state digest mismatch"
                )
        self._hidden_by_agent.clear()
        self._feedback_by_agent.clear()
        self._pending_by_agent.clear()
        self._staged_tick = None
        self._staged_by_agent.clear()
        self._genome_population_manager = None
        self._active_world_id = None
        self._genome_world_identity = None
        self._environment_seed = 0
        self._policy_sampling_seed = 0
        self._bootstrap_phase = False

    def finalize_action_free_bootstrap(
        self,
        *,
        tick: int,
        ordered_agent_ids: Sequence[int],
        observations_by_agent: Mapping[int, Mapping[str, object]],
    ) -> dict[str, object]:
        """Evaluate V(s_T) for every living agent without sampling an action.

        Living agents with no scored decision (for example, reproduction on the
        final scored tick) remain bound in the evidence but cannot enter PPO
        target arrays.
        """

        world_id = self._require_active_world()
        self._validate_core_conditioning_mode_binding()
        tick = _strict_int(tick, field="action-free bootstrap tick")
        if tick != self._rollout_ticks:
            raise RecurrentRolloutError(
                "action-free bootstrap tick must equal rollout_ticks"
            )
        if self._pending_by_agent:
            raise RecurrentRolloutError(
                "action-free bootstrap cannot run over pending decisions"
            )
        ordered_ids = tuple(
            _strict_int(agent_id, field="action-free bootstrap agent_id")
            for agent_id in ordered_agent_ids
        )
        if len(set(ordered_ids)) != len(ordered_ids):
            raise RecurrentRolloutError(
                "action-free bootstrap agent order must be unique"
            )
        if set(observations_by_agent) != set(ordered_ids):
            raise RecurrentRolloutError(
                "action-free bootstrap observations must exactly cover living agents"
            )
        if not self._bootstrap_phase:
            if ordered_ids:
                raise RecurrentRolloutError(
                    "action-free bootstrap requires all scored ticks to be finalized"
                )
            self.buffer.validate_world_closed(world_id)
        if ordered_ids:
            expected_order = deterministic_agent_turn_order(
                ordered_ids,
                seed=self._environment_seed,
                tick=tick,
            )
            if ordered_ids != expected_order:
                raise RecurrentRolloutError(
                    "action-free bootstrap rows are not in deterministic turn order"
                )
        elif self._staged_tick is not None or self._staged_by_agent:
            raise RecurrentRolloutError(
                "empty action-free bootstrap contains staged recurrent rows"
            )

        state_before = self._action_free_policy_state()
        value_rows: dict[int, dict[str, object]] = {}
        prepared_value_rows: dict[
            int,
            tuple[tuple[float, ...], RecurrentCoreOutput, str | None],
        ] = {}
        try:
            if (
                ordered_ids
                and self._fixed_batch_contract is not None
                and self._staged_tick is None
                and not self._staged_by_agent
            ):
                self.stage_tick_start_batch(
                    tick=tick,
                    ordered_agent_ids=ordered_ids,
                    observations_by_agent=observations_by_agent,
                )
            for agent_id in ordered_ids:
                observation = observations_by_agent[agent_id]
                if self._fixed_batch_contract is None:
                    (
                        policy_input,
                        mask,
                        hidden,
                        previous_feedback,
                        genome_values,
                        genome_sha256,
                        genome_stream_seed,
                    ) = self._decision_inputs(
                        observation,
                        _mapping(
                            observation.get("action_mask"),
                            field="action_mask",
                        ),
                        expected_agent_id=agent_id,
                    )
                    if (
                        self._genome_conditioning_mode
                        == RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
                    ):
                        raw_output = self._core.forward_step(
                            policy_input,
                            mask,
                            previous_feedback,
                            hidden,
                            genome_values=genome_values,
                        )
                    else:
                        raw_output = self._core.forward_step(
                            policy_input,
                            mask,
                            previous_feedback,
                            hidden,
                        )
                    output = self._validated_core_output(raw_output)
                else:
                    if self._staged_tick != tick:
                        raise RecurrentRolloutError(
                            "action-free bootstrap fixed batch tick drifted"
                        )
                    staged = self._consume_staged_decision(
                        observation=observation,
                        action_mask=_mapping(
                            observation.get("action_mask"),
                            field="action_mask",
                        ),
                        expected_agent_id=agent_id,
                        missing_error=(
                            f"agent {agent_id} has no staged bootstrap value"
                        ),
                        observation_drift_error=(
                            "action-free bootstrap staged observation drifted"
                        ),
                        state_drift_error=(
                            "action-free bootstrap staged inputs drifted"
                        ),
                    )
                    policy_input = staged.policy_input
                    mask = staged.action_mask
                    hidden = staged.hidden
                    previous_feedback = staged.previous_feedback
                    genome_values = staged.genome_values
                    genome_sha256 = staged.genome_sha256
                    output = staged.output
                prepared_value_rows[agent_id] = (
                    policy_input,
                    output,
                    genome_sha256,
                )
            target_eligibility = self.buffer.mark_truncated_batch(
                world_id=world_id,
                bootstrap_values_by_agent={
                    agent_id: output.value
                    for agent_id, (
                        _policy_input,
                        output,
                        _genome_sha256,
                    ) in prepared_value_rows.items()
                },
            )
            for agent_id, (
                policy_input,
                output,
                genome_sha256,
            ) in prepared_value_rows.items():
                value_rows[agent_id] = _bootstrap_value_evidence_row(
                    agent_id=agent_id,
                    policy_input=policy_input,
                    value=output.value,
                    target_eligible=target_eligibility[agent_id],
                    genome_sha256=genome_sha256,
                )
            if self._staged_by_agent:
                raise RecurrentRolloutError(
                    "action-free bootstrap left staged recurrent rows"
                )
            self._staged_tick = None
            self._bootstrap_value_rows = value_rows
            state_after = self._action_free_policy_state()
            if state_after != state_before:
                raise RecurrentRolloutError(
                    "action-free bootstrap mutated policy, RNG, or genome state"
                )
            return copy.deepcopy(self._build_bootstrap_evidence())
        except Exception:
            self._staged_tick = None
            self._staged_by_agent.clear()
            raise

    def stage_tick_start_batch(
        self,
        *,
        tick: int,
        ordered_agent_ids: Sequence[int],
        observations_by_agent: Mapping[int, Mapping[str, object]],
    ) -> None:
        """Stage pure recurrent outputs in the simulator's exact turn order."""

        self._require_active_world()
        self._validate_core_conditioning_mode_binding()
        contract = self._fixed_batch_contract
        if contract is None:
            return
        if self._staged_tick is not None or self._staged_by_agent:
            raise RecurrentRolloutError(
                "previous fixed recurrent batch tick was not reconciled"
            )
        tick = _strict_int(tick, field="fixed batch tick")
        if not 0 <= tick <= self._rollout_ticks:
            raise RecurrentRolloutError(
                "fixed recurrent batch tick is outside rollout/bootstrap horizon"
            )
        ordered_ids = tuple(
            _strict_int(agent_id, field="fixed batch agent_id")
            for agent_id in ordered_agent_ids
        )
        if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
            raise RecurrentRolloutError(
                "fixed recurrent batch turn order must be non-empty and unique"
            )
        expected_order = deterministic_agent_turn_order(
            ordered_ids,
            seed=self._environment_seed,
            tick=tick,
        )
        if ordered_ids != expected_order:
            raise RecurrentRolloutError(
                "fixed recurrent batch rows are not in deterministic hash turn order"
            )
        if len(ordered_ids) > contract.batch_capacity:
            raise RecurrentRolloutError(
                "fixed recurrent batch active rows exceed configured capacity"
            )
        if set(observations_by_agent) != set(ordered_ids):
            raise RecurrentRolloutError(
                "fixed recurrent batch observations do not exactly cover turn order"
            )
        if self._pending_by_agent:
            raise RecurrentRolloutError(
                "fixed recurrent batch cannot stage over unfinalized decisions"
            )

        policy_inputs: list[tuple[float, ...]] = []
        observation_snapshots: list[dict[str, object]] = []
        masks: list[tuple[bool, ...]] = []
        feedback_rows: list[PreviousPublicFeedback] = []
        hidden_rows: list[tuple[float, ...]] = []
        genome_rows: list[tuple[float, ...]] = []
        genome_bindings: list[tuple[str | None, int | None]] = []
        for agent_id in ordered_ids:
            # Bind the encoded input and drift guard to the same immutable
            # public tick-start value, even for a hostile mutable Mapping.
            observation = copy.deepcopy(dict(observations_by_agent[agent_id]))
            (
                policy_input,
                mask,
                hidden,
                previous_feedback,
                genome_values,
                genome_sha256,
                genome_stream_seed,
            ) = self._decision_inputs(
                observation,
                _mapping(observation.get("action_mask"), field="action_mask"),
                expected_agent_id=agent_id,
            )
            observation_snapshots.append(observation)
            policy_inputs.append(policy_input)
            masks.append(mask)
            feedback_rows.append(previous_feedback)
            hidden_rows.append(hidden)
            if genome_values is not None:
                genome_rows.append(genome_values)
            genome_bindings.append((genome_sha256, genome_stream_seed))

        forward_batch = getattr(self._core, "forward_fixed_batch", None)
        if not callable(forward_batch):
            raise RecurrentRolloutError(
                "fixed recurrent batch core has no forward_fixed_batch implementation"
            )
        raw_outputs = forward_batch(
            tuple(policy_inputs),
            tuple(masks),
            tuple(feedback_rows),
            tuple(hidden_rows),
            batch_capacity=contract.batch_capacity,
            genome_values=(
                tuple(genome_rows)
                if self._genome_conditioning_mode
                == RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
                else None
            ),
        )
        if not isinstance(raw_outputs, Sequence) or isinstance(
            raw_outputs,
            (str, bytes),
        ):
            raise RecurrentRolloutError(
                "fixed recurrent batch core output must be an ordered sequence"
            )
        if len(raw_outputs) != len(ordered_ids):
            raise RecurrentRolloutError(
                "fixed recurrent batch core output row count drifted"
            )
        staged: dict[int, StagedRecurrentDecision] = {}
        for turn_rank, (agent_id, raw_output) in enumerate(
            zip(ordered_ids, raw_outputs, strict=True)
        ):
            if not isinstance(raw_output, RecurrentCoreOutput):
                raise RecurrentRolloutError(
                    "fixed recurrent batch core emitted an invalid row"
                )
            genome_sha256, genome_stream_seed = genome_bindings[turn_rank]
            staged[agent_id] = StagedRecurrentDecision(
                tick=tick,
                turn_rank=turn_rank,
                active_rows=len(ordered_ids),
                observation_snapshot=observation_snapshots[turn_rank],
                policy_input=policy_inputs[turn_rank],
                previous_feedback=feedback_rows[turn_rank],
                action_mask=masks[turn_rank],
                hidden=hidden_rows[turn_rank],
                output=self._validated_core_output(raw_output),
                genome_values=(genome_rows[turn_rank] if genome_rows else None),
                genome_sha256=genome_sha256,
                genome_stream_seed=genome_stream_seed,
            )
        self._staged_tick = tick
        self._staged_by_agent = staged

    def reconcile_live_agent_ids(
        self,
        *,
        live_agent_ids: Sequence[int],
    ) -> None:
        self._require_active_world()
        self._validate_core_conditioning_mode_binding()
        self._reconcile_staged_tick(live_agent_ids=live_agent_ids)
        if self._genome_conditioning_mode == RECURRENT_GENOME_CONDITIONING_DISABLED:
            return
        manager = self._require_genome_population_manager()
        try:
            planned_dead_agent_ids = manager.reconciliation_dead_agent_ids(
                live_agent_ids
            )
        except RecurrentGenomePopulationError as exc:
            raise RecurrentRolloutError(
                f"recurrent genome live-agent reconciliation failed: {exc}"
            ) from exc
        pending_dead_agent_ids = tuple(
            agent_id
            for agent_id in planned_dead_agent_ids
            if agent_id in self._pending_by_agent
        )
        if pending_dead_agent_ids:
            raise RecurrentRolloutError(
                "recurrent genome reconciliation found dead agents with "
                f"unfinalized decisions: {list(pending_dead_agent_ids)}"
            )
        discarded_agent_ids = manager.reconcile_live_agent_ids(live_agent_ids)
        if discarded_agent_ids != planned_dead_agent_ids:
            raise RecurrentRolloutError(
                "recurrent genome reconciliation plan changed before apply"
            )
        for agent_id in discarded_agent_ids:
            self._hidden_by_agent.pop(agent_id, None)
            self._feedback_by_agent.pop(agent_id, None)

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        world_id = self._require_active_world()
        self._validate_core_conditioning_mode_binding()
        agent_id = _agent_id_from_observation(observation)
        if agent_id in self._pending_by_agent:
            raise RecurrentRolloutError(
                f"agent {agent_id} has an unfinalized previous decision"
            )
        staged = None
        if self._fixed_batch_contract is not None:
            if self._staged_tick is None:
                raise RecurrentRolloutError(
                    "fixed recurrent batch decision has no staged tick-start output"
                )
            staged = self._consume_staged_decision(
                observation=observation,
                action_mask=action_mask,
                expected_agent_id=agent_id,
                missing_error=(f"agent {agent_id} has no staged fixed-batch output"),
                observation_drift_error=(
                    "fixed recurrent batch staged observation drifted before sampling"
                ),
                state_drift_error=(
                    "fixed recurrent batch staged inputs drifted before sampling"
                ),
            )
            policy_input = staged.policy_input
            mask = staged.action_mask
            hidden = staged.hidden
            previous_feedback = staged.previous_feedback
            genome_values = staged.genome_values
            genome_sha256 = staged.genome_sha256
            genome_stream_seed = staged.genome_stream_seed
            output = staged.output
        else:
            (
                policy_input,
                mask,
                hidden,
                previous_feedback,
                genome_values,
                genome_sha256,
                genome_stream_seed,
            ) = self._decision_inputs(
                observation,
                action_mask,
                expected_agent_id=agent_id,
            )
            if (
                self._genome_conditioning_mode
                == RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
            ):
                raw_output = self._core.forward_step(
                    policy_input,
                    mask,
                    previous_feedback,
                    hidden,
                    genome_values=genome_values,
                )
            else:
                raw_output = self._core.forward_step(
                    policy_input,
                    mask,
                    previous_feedback,
                    hidden,
                )
            output = self._validated_core_output(raw_output)
        action_index, logprob, entropy = _sample_masked_action(
            output.logits,
            mask,
            rng=self._rng,
        )
        requested_action = RECURRENT_ROLLOUT_ACTIONS[action_index]
        tick_phase = "bootstrap" if self._bootstrap_phase else "rollout"
        decision_index = self._decision_index
        self._decision_index += 1
        pending = PendingDecision(
            world_id=world_id,
            environment_seed=self._environment_seed,
            policy_sampling_seed=self._policy_sampling_seed,
            tick_phase=tick_phase,
            agent_id=agent_id,
            decision_index=decision_index,
            observation=tuple(policy_input),
            previous_feedback=previous_feedback,
            action_mask=mask,
            action_index=action_index,
            requested_action=requested_action,
            logprob=logprob,
            entropy=entropy,
            value=output.value,
            hidden=hidden,
            next_hidden=output.next_hidden,
            genome_values=genome_values,
            genome_sha256=genome_sha256,
            genome_stream_seed=genome_stream_seed,
            fixed_batch_runtime_sha256=(
                None
                if self._fixed_batch_runtime is None
                else str(self._fixed_batch_runtime["exact_digest"])
            ),
            fixed_batch_turn_rank=(None if staged is None else staged.turn_rank),
            fixed_batch_active_rows=(None if staged is None else staged.active_rows),
            fixed_batch_capacity=(
                None
                if self._fixed_batch_contract is None
                else self._fixed_batch_contract.batch_capacity
            ),
        )
        self._pending_by_agent[agent_id] = pending
        if not self._reset_recurrent_state_each_decision:
            self._hidden_by_agent[agent_id] = output.next_hidden
        diagnostics: dict[str, object] = {
            "schema_version": (
                _rollout_diagnostic_schema(
                    genome_conditioned=genome_values is not None,
                    fixed_batch=staged is not None,
                )
            ),
            "world_id": world_id,
            "environment_seed": self._environment_seed,
            "policy_sampling_seed": self._policy_sampling_seed,
            "policy_sampling_seed_namespace": (
                RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE
            ),
            "decision_index": decision_index,
            "agent_id": agent_id,
            "phase": tick_phase,
            "policy_input_schema_version": self._public_input_schema_version,
            "policy_input_size": self._public_input_size,
            "learned_input_size": self._learned_input_size,
            "previous_feedback_schema_version": (
                RECURRENT_PUBLIC_FEEDBACK_SCHEMA_VERSION
            ),
            "previous_feedback_available": previous_feedback.available,
            "recurrent_state_reset_each_decision": (
                self._reset_recurrent_state_each_decision
            ),
            "action_index": action_index,
            "logprob": round(logprob, 12),
            "value": round(output.value, 12),
        }
        if genome_values is not None:
            manager = self._require_genome_population_manager()
            diagnostics.update(
                {
                    "genome_conditioning_mode": self._genome_conditioning_mode,
                    "genome_population_mode": manager.mode.value,
                    "genome_population_binding_sha256": manager.binding_sha256,
                    "genome_sha256": genome_sha256,
                    "genome_stream_seed": genome_stream_seed,
                }
            )
        if staged is not None:
            if self._fixed_batch_runtime is None or self._fixed_batch_contract is None:
                raise AssertionError("fixed batch runtime binding disappeared")
            diagnostics["fixed_batch"] = {
                "runtime_schema_version": (
                    RECURRENT_FIXED_BATCH_RUNTIME_SCHEMA_VERSION
                ),
                "runtime_exact_digest": self._fixed_batch_runtime["exact_digest"],
                "batch_capacity": self._fixed_batch_contract.batch_capacity,
                "active_rows": staged.active_rows,
                "padding_rows": (
                    self._fixed_batch_contract.batch_capacity - staged.active_rows
                ),
                "turn_rank": staged.turn_rank,
                "row_order_policy": RECURRENT_FIXED_BATCH_ROW_ORDER_POLICY,
                "sequential_sampling": True,
                "sequential_resolution": True,
            }
        return ActionDecision(
            requested_action=requested_action,
            source=RECURRENT_ROLLOUT_ACTION_SOURCE,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            diagnostics=diagnostics,
        )

    def observe_transition(
        self,
        record: dict[str, object],
    ) -> dict[str, object] | None:
        world_id = self._require_active_world()
        agent_id = _record_agent_id(record)
        tick = _record_tick(record)
        pending = self._pending_by_agent.pop(agent_id, None)

        if pending is None:
            self._observe_passive_record(
                record,
                world_id=world_id,
                agent_id=agent_id,
                tick=tick,
            )
            if tick == self._rollout_ticks - 1:
                self._bootstrap_phase = True
            return None

        self._validate_alignment(record, pending=pending, tick=tick)
        if pending.tick_phase == "bootstrap":
            target_eligible = self.buffer.mark_truncated(
                world_id=world_id,
                agent_id=agent_id,
                bootstrap_value=pending.value,
            )
            self._bootstrap_value_rows[agent_id] = _bootstrap_value_evidence_row(
                agent_id=agent_id,
                policy_input=pending.observation,
                value=pending.value,
                target_eligible=target_eligible,
                genome_sha256=pending.genome_sha256,
            )
            if _record_terminated(record):
                self._discard_terminal_genome(agent_id)
            self._hidden_by_agent.pop(agent_id, None)
            self._feedback_by_agent.pop(agent_id, None)
            update_trace: dict[str, object] = {
                "schema_version": _rollout_diagnostic_schema(
                    genome_conditioned=pending.genome_values is not None,
                    fixed_batch=pending.fixed_batch_runtime_sha256 is not None,
                ),
                "decision_index": pending.decision_index,
                "phase": "bootstrap",
                "collected": False,
                "environment_seed": pending.environment_seed,
                "policy_sampling_seed": pending.policy_sampling_seed,
            }
            if pending.genome_values is not None:
                update_trace.update(
                    {
                        "genome_conditioning_mode": (self._genome_conditioning_mode),
                        "genome_sha256": pending.genome_sha256,
                        "genome_stream_seed": pending.genome_stream_seed,
                    }
                )
            if pending.fixed_batch_runtime_sha256 is not None:
                update_trace["fixed_batch_runtime_sha256"] = (
                    pending.fixed_batch_runtime_sha256
                )
            return update_trace

        if tick >= self._rollout_ticks:
            raise RecurrentRolloutError(
                "rollout decision finalized beyond configured rollout horizon"
            )
        reward, reward_components = _reward_payload(record)
        terminated = _record_terminated(record)
        outcome = _mapping(record.get("outcome"), field="outcome")
        resolved_action = _action_name(
            record.get("resolved_action"),
            field="resolved_action",
        )
        resolution_action_mask = _action_mask_tuple(
            _mapping(
                record.get("resolution_action_mask"),
                field="resolution_action_mask",
            )
        )
        step = RecurrentRolloutStep(
            world_id=world_id,
            world_seed=pending.environment_seed,
            tick=tick,
            agent_id=agent_id,
            decision_index=pending.decision_index,
            observation=pending.observation,
            previous_feedback=pending.previous_feedback,
            action_mask=pending.action_mask,
            hidden=pending.hidden,
            action_index=pending.action_index,
            requested_action=pending.requested_action,
            logprob=pending.logprob,
            entropy=pending.entropy,
            value=pending.value,
            reward=reward,
            reward_components=reward_components,
            resolved_action_index=RECURRENT_ROLLOUT_ACTIONS.index(resolved_action),
            resolved_action=resolved_action,
            resolution_action_mask=resolution_action_mask,
            action_valid=_strict_bool(record.get("action_valid"), field="action_valid"),
            resolution_action_valid=_strict_bool(
                record.get("resolution_action_valid"),
                field="resolution_action_valid",
            ),
            moved=_strict_bool(record.get("moved"), field="moved"),
            outcome=copy.deepcopy(dict(outcome)),
            terminated=terminated,
            environment_seed=pending.environment_seed,
            policy_sampling_seed=pending.policy_sampling_seed,
            genome_values=pending.genome_values,
            genome_sha256=pending.genome_sha256,
            genome_stream_seed=pending.genome_stream_seed,
            fixed_batch_runtime_sha256=pending.fixed_batch_runtime_sha256,
            fixed_batch_turn_rank=pending.fixed_batch_turn_rank,
            fixed_batch_active_rows=pending.fixed_batch_active_rows,
            fixed_batch_capacity=pending.fixed_batch_capacity,
        )
        self.buffer.append(step)
        if terminated:
            self._discard_terminal_genome(agent_id)
            self._hidden_by_agent.pop(agent_id, None)
            self._feedback_by_agent.pop(agent_id, None)
        elif not self._reset_recurrent_state_each_decision:
            self._feedback_by_agent[agent_id] = PreviousPublicFeedback.from_record(
                record
            )
        if tick == self._rollout_ticks - 1:
            self._bootstrap_phase = True
        update_trace = {
            "schema_version": _rollout_diagnostic_schema(
                genome_conditioned=pending.genome_values is not None,
                fixed_batch=pending.fixed_batch_runtime_sha256 is not None,
            ),
            "decision_index": pending.decision_index,
            "phase": "rollout",
            "collected": True,
            "terminated": terminated,
            "environment_seed": pending.environment_seed,
            "policy_sampling_seed": pending.policy_sampling_seed,
        }
        if pending.genome_values is not None:
            update_trace.update(
                {
                    "genome_conditioning_mode": self._genome_conditioning_mode,
                    "genome_sha256": pending.genome_sha256,
                    "genome_stream_seed": pending.genome_stream_seed,
                }
            )
        if pending.fixed_batch_runtime_sha256 is not None:
            update_trace["fixed_batch_runtime_sha256"] = (
                pending.fixed_batch_runtime_sha256
            )
        return update_trace

    def _observe_passive_record(
        self,
        record: Mapping[str, object],
        *,
        world_id: str,
        agent_id: int,
        tick: int,
    ) -> None:
        if str(record.get("action_source")) != "passive":
            raise RecurrentRolloutError(
                "transition has no pending policy decision and is not passive"
            )
        if not _record_terminated(record):
            raise RecurrentRolloutError("passive transition must terminate the agent")

        if tick > self._rollout_ticks:
            raise RecurrentRolloutError(
                "passive terminal arrived beyond the one-tick bootstrap boundary"
            )
        self._require_live_genome_if_conditioned(agent_id)
        reward, reward_components = _reward_payload(record)
        self.buffer.mark_passive_terminal(
            world_id=world_id,
            agent_id=agent_id,
            tick=tick,
            reward=reward,
            reward_components=reward_components,
        )
        self._discard_terminal_genome(agent_id)
        self._hidden_by_agent.pop(agent_id, None)
        self._feedback_by_agent.pop(agent_id, None)

    def _validate_alignment(
        self,
        record: Mapping[str, object],
        *,
        pending: PendingDecision,
        tick: int,
    ) -> None:
        diagnostics = _mapping(
            record.get("policy_decision_diagnostics"),
            field="policy_decision_diagnostics",
        )
        self._validate_pending_genome(pending)
        expected_diagnostic_schema = _rollout_diagnostic_schema(
            genome_conditioned=pending.genome_values is not None,
            fixed_batch=pending.fixed_batch_runtime_sha256 is not None,
        )
        if diagnostics.get("schema_version") != expected_diagnostic_schema:
            raise RecurrentRolloutError(
                "transition decision diagnostic schema mismatch"
            )
        if diagnostics.get("world_id") != pending.world_id:
            raise RecurrentRolloutError("transition world_id does not match decision")
        if diagnostics.get("environment_seed") != pending.environment_seed:
            raise RecurrentRolloutError(
                "transition environment_seed does not match decision"
            )
        if diagnostics.get("policy_sampling_seed") != pending.policy_sampling_seed:
            raise RecurrentRolloutError(
                "transition policy_sampling_seed does not match decision"
            )
        if (
            diagnostics.get("policy_sampling_seed_namespace")
            != RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE
        ):
            raise RecurrentRolloutError(
                "transition policy sampling namespace does not match collector"
            )
        if diagnostics.get("decision_index") != pending.decision_index:
            raise RecurrentRolloutError("transition decision_index does not match")
        if diagnostics.get("phase") != pending.tick_phase:
            raise RecurrentRolloutError("transition phase does not match decision")
        if pending.fixed_batch_runtime_sha256 is not None:
            fixed_batch = _mapping(
                diagnostics.get("fixed_batch"),
                field="fixed_batch",
            )
            expected_fixed_batch = {
                "runtime_schema_version": (
                    RECURRENT_FIXED_BATCH_RUNTIME_SCHEMA_VERSION
                ),
                "runtime_exact_digest": pending.fixed_batch_runtime_sha256,
                "batch_capacity": pending.fixed_batch_capacity,
                "active_rows": pending.fixed_batch_active_rows,
                "padding_rows": (
                    int(pending.fixed_batch_capacity)
                    - int(pending.fixed_batch_active_rows)
                ),
                "turn_rank": pending.fixed_batch_turn_rank,
                "row_order_policy": RECURRENT_FIXED_BATCH_ROW_ORDER_POLICY,
                "sequential_sampling": True,
                "sequential_resolution": True,
            }
            if dict(fixed_batch) != expected_fixed_batch:
                raise RecurrentRolloutError(
                    "transition fixed-batch diagnostics do not match decision"
                )
        elif "fixed_batch" in diagnostics:
            raise RecurrentRolloutError(
                "scalar transition unexpectedly contains fixed-batch diagnostics"
            )
        if pending.genome_values is not None:
            manager = self._require_genome_population_manager()
            expected_genome_diagnostics = {
                "genome_conditioning_mode": self._genome_conditioning_mode,
                "genome_population_mode": manager.mode.value,
                "genome_population_binding_sha256": manager.binding_sha256,
                "genome_sha256": pending.genome_sha256,
                "genome_stream_seed": pending.genome_stream_seed,
            }
            for key, expected_value in expected_genome_diagnostics.items():
                if diagnostics.get(key) != expected_value:
                    raise RecurrentRolloutError(
                        f"transition {key} does not match decision"
                    )
        if record.get("policy_id") != self.policy_id:
            raise RecurrentRolloutError("transition policy_id does not match collector")
        if record.get("policy_version") != self.policy_version:
            raise RecurrentRolloutError(
                "transition policy_version does not match collector"
            )
        if record.get("action_source") != RECURRENT_ROLLOUT_ACTION_SOURCE:
            raise RecurrentRolloutError(
                "transition action source does not match collector"
            )
        if record.get("requested_action") != pending.requested_action:
            raise RecurrentRolloutError("transition requested action does not match")
        if (
            _action_mask_tuple(_mapping(record.get("action_mask"), field="action_mask"))
            != pending.action_mask
        ):
            raise RecurrentRolloutError(
                "transition action mask does not match decision"
            )
        expected_bootstrap_tick = self._rollout_ticks
        if pending.tick_phase == "bootstrap" and tick != expected_bootstrap_tick:
            raise RecurrentRolloutError(
                "bootstrap decision finalized at unexpected tick: "
                f"expected={expected_bootstrap_tick} actual={tick}"
            )

    def _decision_inputs(
        self,
        observation: Mapping[str, object],
        action_mask: Mapping[str, object],
        *,
        expected_agent_id: int,
    ) -> tuple[
        tuple[float, ...],
        tuple[bool, ...],
        tuple[float, ...],
        PreviousPublicFeedback,
        tuple[float, ...] | None,
        str | None,
        int | None,
    ]:
        observed_agent_id = _agent_id_from_observation(observation)
        if observed_agent_id != expected_agent_id:
            raise RecurrentRolloutError(
                "fixed recurrent batch observation agent identity drifted"
            )
        policy_input = ecological_policy_input_values(
            encode_observation_input(dict(observation))
        )
        observed_schema_version = (
            TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
            if observation.get("schema_version")
            == TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION
            else ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
        )
        if observed_schema_version != self._public_input_schema_version:
            raise RecurrentRolloutError(
                "ecological policy input schema does not match the recurrent core"
            )
        if len(policy_input) != self._public_input_size:
            raise RecurrentRolloutError(
                "ecological policy input size does not match the recurrent core: "
                f"{len(policy_input)} != {self._public_input_size}"
            )
        mask = _action_mask_tuple(action_mask)
        (
            hidden,
            previous_feedback,
            genome_values,
            genome_sha256,
            genome_stream_seed,
        ) = self._decision_state_inputs(expected_agent_id)
        return (
            tuple(policy_input),
            mask,
            hidden,
            previous_feedback,
            genome_values,
            genome_sha256,
            genome_stream_seed,
        )

    def _decision_state_inputs(
        self,
        expected_agent_id: int,
    ) -> tuple[
        tuple[float, ...],
        PreviousPublicFeedback,
        tuple[float, ...] | None,
        str | None,
        int | None,
    ]:
        hidden = (
            None
            if self._reset_recurrent_state_each_decision
            else self._hidden_by_agent.get(expected_agent_id)
        )
        if hidden is None:
            hidden = self._initial_hidden()
        previous_feedback = (
            PreviousPublicFeedback.zero()
            if self._reset_recurrent_state_each_decision
            else self._feedback_by_agent.get(
                expected_agent_id,
                PreviousPublicFeedback.zero(),
            )
        )
        genome_values: tuple[float, ...] | None = None
        genome_sha256: str | None = None
        genome_stream_seed: int | None = None
        if (
            self._genome_conditioning_mode
            == RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
        ):
            manager = self._require_genome_population_manager()
            try:
                binding = manager.genome_binding_for_agent(expected_agent_id)
            except RecurrentGenomePopulationError as exc:
                raise RecurrentRolloutError(
                    f"recurrent genome lookup failed before decision: {exc}"
                ) from exc
            genome_values = binding.genome.values
            genome_sha256 = binding.genome_sha256
            genome_stream_seed = manager.genome_stream_seed
        return (
            hidden,
            previous_feedback,
            genome_values,
            genome_sha256,
            genome_stream_seed,
        )

    def _consume_staged_decision(
        self,
        *,
        observation: Mapping[str, object],
        action_mask: Mapping[str, object],
        expected_agent_id: int,
        missing_error: str,
        observation_drift_error: str,
        state_drift_error: str,
    ) -> StagedRecurrentDecision:
        staged = self._staged_by_agent.get(expected_agent_id)
        if staged is None:
            raise RecurrentRolloutError(missing_error)
        if dict(observation) != staged.observation_snapshot:
            raise RecurrentRolloutError(observation_drift_error)
        mask = _action_mask_tuple(action_mask)
        (
            hidden,
            previous_feedback,
            genome_values,
            genome_sha256,
            genome_stream_seed,
        ) = self._decision_state_inputs(expected_agent_id)
        if (
            mask,
            hidden,
            previous_feedback,
            genome_values,
            genome_sha256,
            genome_stream_seed,
        ) != (
            staged.action_mask,
            staged.hidden,
            staged.previous_feedback,
            staged.genome_values,
            staged.genome_sha256,
            staged.genome_stream_seed,
        ):
            raise RecurrentRolloutError(state_drift_error)
        consumed = self._staged_by_agent.pop(expected_agent_id)
        if consumed is not staged:
            raise AssertionError("staged recurrent decision changed before consume")
        return staged

    def _reconcile_staged_tick(
        self,
        *,
        live_agent_ids: Sequence[int],
    ) -> None:
        if self._fixed_batch_contract is None:
            if self._staged_tick is not None or self._staged_by_agent:
                raise RecurrentRolloutError(
                    "scalar recurrent collector contains fixed-batch state"
                )
            return
        if self._staged_tick is None:
            raise RecurrentRolloutError(
                "fixed recurrent batch tick ended without a staged forward"
            )
        live_ids = {
            _strict_int(agent_id, field="live agent_id") for agent_id in live_agent_ids
        }
        live_unconsumed = sorted(live_ids.intersection(self._staged_by_agent))
        if live_unconsumed:
            raise RecurrentRolloutError(
                "live agents reached fixed-batch reconciliation without sampling: "
                f"{live_unconsumed}"
            )
        self._staged_by_agent.clear()
        self._staged_tick = None

    def _build_bootstrap_evidence(self) -> dict[str, object]:
        values = [
            copy.deepcopy(self._bootstrap_value_rows[agent_id])
            for agent_id in sorted(self._bootstrap_value_rows)
        ]
        zero_decision_alive_agent_ids = [
            int(row["agent_id"]) for row in values if row["target_eligible"] is False
        ]
        payload: dict[str, object] = {
            "schema_version": RECURRENT_ACTION_FREE_BOOTSTRAP_SCHEMA_VERSION,
            "world_id": self._require_active_world(),
            "tick": self._rollout_ticks,
            "boundary": (
                "exact_policy_visible_tick_start_before_action_v1"
                if self._bootstrap_phase
                else "terminal_before_horizon_no_bootstrap_v1"
            ),
            "action_sampled": False,
            "action_committed": False,
            "action_resolved": False,
            "alive_agent_count": len(values),
            "target_eligible_agent_count": (
                len(values) - len(zero_decision_alive_agent_ids)
            ),
            "zero_decision_alive_agent_count": len(zero_decision_alive_agent_ids),
            "zero_decision_alive_agent_ids": zero_decision_alive_agent_ids,
            "values": values,
        }
        payload["exact_digest"] = _stable_payload_sha256(payload)
        return payload

    def _action_free_policy_state(self) -> dict[str, object]:
        manager = self._genome_population_manager
        return {
            "decision_index": self._decision_index,
            "sampling_rng_state": copy.deepcopy(self._rng.getstate()),
            "hidden_by_agent": copy.deepcopy(self._hidden_by_agent),
            "feedback_by_agent": copy.deepcopy(self._feedback_by_agent),
            "pending_by_agent": copy.deepcopy(self._pending_by_agent),
            "genome_population_state_sha256": (
                None if manager is None else manager.state_sha256
            ),
        }

    def _initial_hidden(self) -> tuple[float, ...]:
        hidden = tuple(
            _finite_float(value, field="initial hidden state")
            for value in self._core.initial_hidden()
        )
        if len(hidden) != int(self._core.hidden_size):
            raise RecurrentRolloutError(
                "initial hidden state length does not match core.hidden_size"
            )
        return hidden

    def _validated_core_output(
        self,
        output: RecurrentCoreOutput,
    ) -> RecurrentCoreOutput:
        logits = tuple(
            _finite_float(value, field="action logit") for value in output.logits
        )
        if len(logits) != len(RECURRENT_ROLLOUT_ACTIONS):
            raise RecurrentRolloutError(
                "core must emit one logit per stable action: "
                f"expected={len(RECURRENT_ROLLOUT_ACTIONS)} actual={len(logits)}"
            )
        next_hidden = tuple(
            _finite_float(value, field="next hidden state")
            for value in output.next_hidden
        )
        if len(next_hidden) != int(self._core.hidden_size):
            raise RecurrentRolloutError(
                "next hidden state length does not match core.hidden_size"
            )
        return RecurrentCoreOutput(
            logits=logits,
            value=_finite_float(output.value, field="value estimate"),
            next_hidden=next_hidden,
        )

    def _require_genome_population_manager(
        self,
    ) -> RecurrentGenomePopulationManager:
        self._validate_core_conditioning_mode_binding()
        if (
            self._genome_conditioning_mode
            != RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
        ):
            raise RecurrentRolloutError(
                "genome population is unavailable for a disabled recurrent core"
            )
        manager = self._genome_population_manager
        if manager is None:
            raise RecurrentRolloutError(
                "actor_film_v1 collection has no active genome population"
            )
        if manager.mode not in {
            RecurrentGenomePopulationMode.HERITABLE,
            RecurrentGenomePopulationMode.ZERO_ALL,
        }:
            raise RecurrentRolloutError(
                "actor_film_v1 collection has an incompatible population mode"
            )
        if (
            self._genome_world_identity is None
            or manager.world_identity != self._genome_world_identity
        ):
            raise RecurrentRolloutError(
                "recurrent genome population world binding does not match collector"
            )
        return manager

    def _validate_core_conditioning_mode_binding(self) -> None:
        observed_mode = _genome_conditioning_mode(
            getattr(
                self._core,
                "genome_conditioning_mode",
                RECURRENT_GENOME_CONDITIONING_DISABLED,
            )
        )
        if observed_mode != self._genome_conditioning_mode:
            raise RecurrentRolloutError(
                "recurrent core genome conditioning mode changed after collector "
                "construction"
            )

    def _require_zero_newborn_runtime_state(self, agent_id: int) -> None:
        if agent_id in self._hidden_by_agent:
            raise RecurrentRolloutError(
                f"newborn agent {agent_id} already has recurrent hidden state"
            )
        if agent_id in self._feedback_by_agent:
            raise RecurrentRolloutError(
                f"newborn agent {agent_id} already has previous public feedback"
            )
        if agent_id in self._pending_by_agent:
            raise RecurrentRolloutError(
                f"newborn agent {agent_id} already has a pending decision"
            )

    def _validate_pending_genome(self, pending: PendingDecision) -> None:
        if self._genome_conditioning_mode == RECURRENT_GENOME_CONDITIONING_DISABLED:
            if pending.genome_values is not None:
                raise RecurrentRolloutError(
                    "disabled recurrent core received genome-conditioned decision"
                )
            return
        if pending.genome_values is None:
            raise RecurrentRolloutError(
                "actor_film_v1 decision is missing its controller genome"
            )
        manager = self._require_genome_population_manager()
        if pending.genome_stream_seed != manager.genome_stream_seed:
            raise RecurrentRolloutError(
                "pending controller genome stream seed does not match active world"
            )
        try:
            binding = manager.genome_binding_for_agent(pending.agent_id)
        except RecurrentGenomePopulationError as exc:
            raise RecurrentRolloutError(
                f"pending controller genome ownership failed: {exc}"
            ) from exc
        if (
            pending.genome_values != binding.genome.values
            or pending.genome_sha256 != binding.genome_sha256
        ):
            raise RecurrentRolloutError(
                "pending controller genome does not match active population state"
            )

    def _discard_terminal_genome(self, agent_id: int) -> None:
        if self._genome_conditioning_mode == RECURRENT_GENOME_CONDITIONING_DISABLED:
            return
        manager = self._require_live_genome_if_conditioned(agent_id)
        if manager is None:
            raise RecurrentRolloutError(
                "conditioned terminal cleanup has no genome population"
            )
        try:
            discarded = manager.discard_agent(agent_id)
        except RecurrentGenomePopulationError as exc:
            raise RecurrentRolloutError(
                f"terminal controller genome cleanup failed: {exc}"
            ) from exc
        if not discarded:
            raise RecurrentRolloutError(
                f"terminal controller genome cleanup missed agent {agent_id}"
            )

    def _require_live_genome_if_conditioned(
        self,
        agent_id: int,
    ) -> RecurrentGenomePopulationManager | None:
        if self._genome_conditioning_mode == RECURRENT_GENOME_CONDITIONING_DISABLED:
            return None
        manager = self._require_genome_population_manager()
        try:
            manager.genome_for_agent(agent_id)
        except RecurrentGenomePopulationError as exc:
            raise RecurrentRolloutError(
                f"terminal controller genome cleanup failed: {exc}"
            ) from exc
        return manager

    def _require_active_world(self) -> str:
        if self._active_world_id is None:
            raise RecurrentRolloutError("start_world must be called before collection")
        return self._active_world_id


def _fixed_batch_runtime_binding(
    contract: RecurrentFixedBatchRuntimeContract,
    *,
    core: RecurrentPolicyCore,
) -> dict[str, object]:
    forward_batch = getattr(core, "forward_fixed_batch", None)
    runtime_metadata = getattr(core, "fixed_batch_runtime_metadata", None)
    if not callable(forward_batch) or not callable(runtime_metadata):
        raise RecurrentRolloutError(
            "fixed recurrent batch core must implement batched forward and runtime "
            "metadata"
        )
    observed_runtime = runtime_metadata()
    if not isinstance(observed_runtime, Mapping) or not observed_runtime:
        raise RecurrentRolloutError(
            "fixed recurrent batch runtime metadata must be a non-empty mapping"
        )
    try:
        canonical_runtime = json.loads(
            json.dumps(
                dict(observed_runtime),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RecurrentRolloutError(
            "fixed recurrent batch runtime metadata must be canonical JSON"
        ) from exc
    if not isinstance(canonical_runtime, dict) or any(
        not isinstance(key, str) or not key for key in canonical_runtime
    ):
        raise RecurrentRolloutError(
            "fixed recurrent batch runtime metadata keys must be non-empty strings"
        )
    payload: dict[str, object] = {
        "schema_version": RECURRENT_FIXED_BATCH_RUNTIME_SCHEMA_VERSION,
        "batch_capacity": contract.batch_capacity,
        "contract": contract.as_contract(),
        "observed_runtime": canonical_runtime,
    }
    payload["exact_digest"] = _stable_payload_sha256(payload)
    return payload


def _validated_fixed_batch_runtime_binding(
    value: Mapping[str, object],
) -> dict[str, object]:
    if set(value) != {
        "schema_version",
        "batch_capacity",
        "contract",
        "observed_runtime",
        "exact_digest",
    }:
        raise RecurrentRolloutError(
            "fixed recurrent batch runtime fields do not match the exact contract"
        )
    if value.get("schema_version") != RECURRENT_FIXED_BATCH_RUNTIME_SCHEMA_VERSION:
        raise RecurrentRolloutError(
            "fixed recurrent batch runtime schema version drifted"
        )
    capacity = value.get("batch_capacity")
    contract = RecurrentFixedBatchRuntimeContract(
        batch_capacity=_strict_int(capacity, field="fixed batch capacity")
    )
    if value.get("contract") != contract.as_contract():
        raise RecurrentRolloutError("fixed recurrent batch embedded contract drifted")
    observed_runtime = value.get("observed_runtime")
    if not isinstance(observed_runtime, Mapping) or not observed_runtime:
        raise RecurrentRolloutError(
            "fixed recurrent batch observed runtime must be a non-empty mapping"
        )
    payload = {
        "schema_version": value["schema_version"],
        "batch_capacity": capacity,
        "contract": copy.deepcopy(dict(value["contract"])),  # type: ignore[arg-type]
        "observed_runtime": copy.deepcopy(dict(observed_runtime)),
    }
    expected_digest = _stable_payload_sha256(payload)
    if value.get("exact_digest") != expected_digest:
        raise RecurrentRolloutError(
            "fixed recurrent batch runtime exact digest mismatched"
        )
    payload["exact_digest"] = expected_digest
    return payload


def _validate_fixed_batch_decision_fields(
    *,
    runtime_sha256: object,
    turn_rank: object,
    active_rows: object,
    capacity: object,
    field: str,
) -> None:
    values = (runtime_sha256, turn_rank, active_rows, capacity)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise RecurrentRolloutError(
            f"{field} fixed-batch fields must be all present or all absent"
        )
    _sha256(runtime_sha256, field=f"{field} fixed_batch_runtime_sha256")
    parsed_turn_rank = _strict_int(
        turn_rank,
        field=f"{field} fixed_batch_turn_rank",
    )
    parsed_active_rows = _strict_int(
        active_rows,
        field=f"{field} fixed_batch_active_rows",
    )
    parsed_capacity = _strict_int(
        capacity,
        field=f"{field} fixed_batch_capacity",
    )
    if (
        parsed_capacity <= 0
        or not 0 < parsed_active_rows <= parsed_capacity
        or not 0 <= parsed_turn_rank < parsed_active_rows
    ):
        raise RecurrentRolloutError(f"{field} fixed-batch row bounds are invalid")


def _rollout_diagnostic_schema(
    *,
    genome_conditioned: bool,
    fixed_batch: bool,
) -> str:
    if fixed_batch:
        return (
            RECURRENT_GENOME_FIXED_BATCH_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION
            if genome_conditioned
            else RECURRENT_FIXED_BATCH_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION
        )
    return (
        RECURRENT_GENOME_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION
        if genome_conditioned
        else RECURRENT_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION
    )


def _torch_parallel_backend(torch_module: Any) -> str:
    parallel_info = str(torch_module.__config__.parallel_info())
    prefix = "ATen parallel backend:"
    for raw_line in parallel_info.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            backend = line.removeprefix(prefix).strip()
            if backend:
                return backend
    raise RecurrentRolloutError(
        "Torch parallel backend is unavailable for fixed-batch provenance"
    )


def _stable_payload_sha256(value: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecurrentRolloutError(
            "fixed recurrent batch payload is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _bootstrap_value_evidence_row(
    *,
    agent_id: int,
    policy_input: Sequence[float],
    value: float,
    target_eligible: bool,
    genome_sha256: str | None,
) -> dict[str, object]:
    if type(target_eligible) is not bool:
        raise RecurrentRolloutError(
            "bootstrap target eligibility must be an exact boolean"
        )
    canonical_input = tuple(
        _finite_float(component, field="bootstrap policy input")
        for component in policy_input
    )
    row: dict[str, object] = {
        "agent_id": _strict_int(agent_id, field="bootstrap agent_id"),
        "policy_input_sha256": hashlib.sha256(
            json.dumps(
                canonical_input,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "value": round(
            _finite_float(value, field="bootstrap value"),
            12,
        ),
        "target_eligible": target_eligible,
        "genome_sha256": (
            None
            if genome_sha256 is None
            else _sha256(genome_sha256, field="bootstrap genome_sha256")
        ),
    }
    row["exact_digest"] = _stable_payload_sha256(row)
    return row


def _validated_rollout_genome_fields(
    *,
    genome_values: tuple[float, ...] | None,
    genome_sha256: str | None,
    genome_stream_seed: int | None,
    field: str,
) -> tuple[tuple[float, ...] | None, str | None, int | None]:
    presence = (
        genome_values is not None,
        genome_sha256 is not None,
        genome_stream_seed is not None,
    )
    if not any(presence):
        return None, None, None
    if not all(presence):
        raise RecurrentRolloutError(
            f"{field} controller genome values, SHA256, and stream seed "
            "must be present together"
        )
    try:
        genome = RecurrentControllerGenome(values=genome_values)
    except RecurrentGenomeError as exc:
        raise RecurrentRolloutError(
            f"{field} controller genome is invalid: {exc}"
        ) from exc
    parsed_sha256 = _sha256(genome_sha256, field=f"{field} genome_sha256")
    if parsed_sha256 != genome.sha256:
        raise RecurrentRolloutError(
            f"{field} controller genome SHA256 does not match its values"
        )
    parsed_stream_seed = _genome_stream_seed(
        genome_stream_seed,
        field=f"{field} genome_stream_seed",
    )
    return genome.values, parsed_sha256, parsed_stream_seed


def _genome_conditioning_mode(value: object) -> str:
    if not isinstance(value, str) or value not in {
        RECURRENT_GENOME_CONDITIONING_DISABLED,
        RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1,
    }:
        raise RecurrentRolloutError(
            "core genome conditioning mode must be exactly disabled or actor_film_v1"
        )
    return str(value)


def _active_genome_population_mode(
    value: object,
) -> RecurrentGenomePopulationMode:
    try:
        mode = (
            value
            if isinstance(value, RecurrentGenomePopulationMode)
            else RecurrentGenomePopulationMode(value)
        )
    except (TypeError, ValueError) as exc:
        raise RecurrentRolloutError(
            "actor_film_v1 requires genome population mode heritable or zero_all"
        ) from exc
    if mode not in {
        RecurrentGenomePopulationMode.HERITABLE,
        RecurrentGenomePopulationMode.ZERO_ALL,
    }:
        raise RecurrentRolloutError(
            "actor_film_v1 requires genome population mode heritable or zero_all"
        )
    return mode


def _genome_stream_seed(value: object, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_RECURRENT_GENOME_STREAM_SEED
    ):
        raise RecurrentRolloutError(f"{field} must be an unsigned 64-bit integer")
    return value


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecurrentRolloutError(f"{field} must be a lowercase SHA256")
    return value


def _sample_masked_action(
    logits: Sequence[float],
    mask: Sequence[bool],
    *,
    rng: random.Random,
) -> tuple[int, float, float]:
    valid_indices = [index for index, valid in enumerate(mask) if valid]
    if not valid_indices:
        raise RecurrentRolloutError("action mask contains no valid action")
    max_logit = max(logits[index] for index in valid_indices)
    weights = [math.exp(logits[index] - max_logit) for index in valid_indices]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise RecurrentRolloutError("masked action distribution is not finite")
    threshold = rng.random() * total
    cumulative = 0.0
    selected_offset = len(valid_indices) - 1
    for offset, weight in enumerate(weights):
        cumulative += weight
        if threshold < cumulative:
            selected_offset = offset
            break
    probabilities = [weight / total for weight in weights]
    probability = probabilities[selected_offset]
    logprob = math.log(probability)
    entropy = -sum(
        candidate * math.log(candidate)
        for candidate in probabilities
        if candidate > 0.0
    )
    return valid_indices[selected_offset], logprob, entropy


def _action_mask_tuple(action_mask: Mapping[str, object]) -> tuple[bool, ...]:
    if set(action_mask) != set(RECURRENT_ROLLOUT_ACTIONS):
        missing = sorted(set(RECURRENT_ROLLOUT_ACTIONS) - set(action_mask))
        extra = sorted(set(action_mask) - set(RECURRENT_ROLLOUT_ACTIONS))
        raise RecurrentRolloutError(
            f"action mask keys drifted; missing={missing} extra={extra}"
        )
    return tuple(
        _strict_bool(action_mask[action], field=f"action_mask.{action}")
        for action in RECURRENT_ROLLOUT_ACTIONS
    )


def _agent_id_from_observation(observation: Mapping[str, object]) -> int:
    metadata = _mapping(observation.get("metadata"), field="observation.metadata")
    return _strict_int(metadata.get("agent_id"), field="observation.metadata.agent_id")


def _record_agent_id(record: Mapping[str, object]) -> int:
    return _strict_int(record.get("agent_id"), field="record.agent_id")


def _record_tick(record: Mapping[str, object]) -> int:
    tick = _strict_int(record.get("tick"), field="record.tick")
    if tick < 0:
        raise RecurrentRolloutError("record.tick must not be negative")
    return tick


def _record_terminated(record: Mapping[str, object]) -> bool:
    outcome = _mapping(record.get("outcome"), field="outcome")
    died = _strict_bool(outcome.get("died"), field="outcome.died")
    after = _mapping(record.get("after"), field="after")
    alive = _strict_bool(after.get("alive"), field="after.alive")
    if died == alive:
        raise RecurrentRolloutError("outcome.died and after.alive disagree")
    return died


def _reward_total(record: Mapping[str, object]) -> float:
    reward_total, _components = _reward_payload(record)
    return reward_total


def _reward_payload(
    record: Mapping[str, object],
) -> tuple[float, dict[str, float]]:
    reward = _mapping(record.get("reward"), field="reward")
    if reward.get("schema_version") != REWARD_SCHEMA_VERSION:
        raise RecurrentRolloutError(
            "reward.schema_version does not match the canonical reward contract"
        )
    return _validated_reward_values(
        reward.get("total"),
        _mapping(reward.get("components"), field="reward.components"),
        field="reward",
    )


def _validated_reward_values(
    total: object,
    components: Mapping[str, object],
    *,
    field: str,
) -> tuple[float, dict[str, float]]:
    observed_keys = set(components)
    expected_keys = set(RECURRENT_REWARD_COMPONENTS)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise RecurrentRolloutError(
            f"{field} component keys drifted; missing={missing} extra={extra}"
        )
    parsed: dict[str, float] = {}
    for name in RECURRENT_REWARD_COMPONENTS:
        value = _finite_float(
            components[name],
            field=f"{field}.components.{name}",
        )
        lower, upper = REWARD_COMPONENT_BOUNDS[name]
        if not lower <= value <= upper:
            raise RecurrentRolloutError(
                f"{field}.components.{name} is outside canonical bounds "
                f"[{lower}, {upper}]"
            )
        parsed[name] = value

    reward_total = _finite_float(total, field=f"{field}.total")
    if not REWARD_TOTAL_BOUNDS[0] <= reward_total <= REWARD_TOTAL_BOUNDS[1]:
        raise RecurrentRolloutError(
            f"{field}.total is outside the public reward contract bounds"
        )
    expected_total = round(math.fsum(parsed.values()), 4)
    if not math.isclose(reward_total, expected_total, rel_tol=0.0, abs_tol=1.0e-9):
        raise RecurrentRolloutError(
            f"{field}.total does not equal its canonical rounded component sum: "
            f"{reward_total} != {expected_total}"
        )
    return reward_total, parsed


def _bounded_reward_total(record: Mapping[str, object]) -> float:
    reward_total = _reward_total(record)
    if not REWARD_TOTAL_BOUNDS[0] <= reward_total <= REWARD_TOTAL_BOUNDS[1]:
        raise RecurrentRolloutError(
            "reward.total is outside the public reward contract bounds"
        )
    return reward_total


def _action_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value not in RECURRENT_ROLLOUT_ACTIONS:
        raise RecurrentRolloutError(f"{field} is not a stable action")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentRolloutError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise RecurrentRolloutError(f"{field} keys must be strings")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise RecurrentRolloutError(f"{field} must be a boolean")
    return value


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecurrentRolloutError(f"{field} must be an integer")
    return int(value)


def _policy_sampling_seed(value: object, *, field: str) -> int:
    parsed = _strict_int(value, field=field)
    if parsed < 1 or parsed > MAX_RECURRENT_POLICY_SAMPLING_SEED:
        raise RecurrentRolloutError(
            f"{field} must be in [1, {MAX_RECURRENT_POLICY_SAMPLING_SEED}]"
        )
    return parsed


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentRolloutError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecurrentRolloutError(f"{field} must be finite")
    return parsed


def _unit_interval(
    value: object,
    *,
    field: str,
    allow_zero: bool,
) -> float:
    parsed = _finite_float(value, field=field)
    lower_ok = parsed >= 0.0 if allow_zero else parsed > 0.0
    if not lower_ok or parsed > 1.0:
        bracket = "[0, 1]" if allow_zero else "(0, 1]"
        raise RecurrentRolloutError(f"{field} must be in {bracket}")
    return parsed
