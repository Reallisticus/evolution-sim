from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from evolution_sim.env.runtime.action_contract import (
    ACTION_MASK_CONTRACT_VERSION,
    ACTION_NAMES,
)
from evolution_sim.env.runtime.observations import encode_observation_input
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.env.runtime.state import empty_mind_inheritance_metadata
from evolution_sim.mind.recurrent_actor_critic import (
    ACTION_COUNT,
    ActionSelection,
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    GENOME_CONDITIONING_ACTOR_FILM_V1,
    GENOME_CONDITIONING_DISABLED,
    PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION,
    PREVIOUS_PUBLIC_FEEDBACK_SIZE,
    PreviousPublicFeedbackInput,
    PublicRecurrentActorCritic,
    RecurrentContextError,
    public_policy_tensor_from_observation_input,
    previous_public_feedback_tensor,
    strict_action_mask_tensor,
    validate_previous_feedback_tensor,
)
from evolution_sim.mind.policy_inputs import (
    ECOLOGICAL_POLICY_INPUT_POLICY,
    ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ecological_policy_input_payload,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_genome_population import (
    RecurrentGenomePopulationError,
    RecurrentGenomePopulationManager,
    RecurrentGenomePopulationMode,
)
from evolution_sim.mind.recurrent_rollout import (
    RECURRENT_ROLLOUT_ACTION_SOURCE,
)


PUBLIC_RECURRENT_POLICY_ID = "mind_v3_public_recurrent_actor_critic"
PUBLIC_RECURRENT_POLICY_VERSION = "mind_v3_public_recurrent_actor_critic_v1"
PUBLIC_RECURRENT_DIAGNOSTIC_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_actor_critic_decision_v3"
)
PUBLIC_RECURRENT_GENOME_DIAGNOSTIC_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_actor_critic_decision_genome_v1"
)
PUBLIC_RECURRENT_DISTRIBUTION_DIAGNOSTIC_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_masked_distribution_diagnostics_v1"
)
PUBLIC_RECURRENT_ARGMAX_SELECTION = "deterministic_masked_argmax"
PUBLIC_RECURRENT_SAMPLED_SELECTION = "replay_deterministic_masked_sampling"
RECURRENT_COUNTERFACTUAL_ACTION_SOURCE = (
    "recurrent_counterfactual_diagnostics_intervention"
)
RECURRENT_DIAGNOSTIC_CHECKPOINT_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_diagnostic_checkpoint_v2"
)
RECURRENT_PUBLIC_HISTORY_PREFIX_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_history_prefix_v1"
)
RECURRENT_GENOME_WORLD_PROVENANCE_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_genome_world_provenance_v1"
)


class RecurrentPolicyAdapterError(ValueError):
    pass


def frozen_cpu_model_copy(
    model: PublicRecurrentActorCritic,
) -> PublicRecurrentActorCritic:
    """Return an inference-only snapshot for a complete rollout phase."""

    if not isinstance(model, PublicRecurrentActorCritic):
        raise TypeError("model must be a PublicRecurrentActorCritic")
    snapshot = copy.deepcopy(model).to(device="cpu", dtype=torch.float32)
    snapshot.eval()
    for parameter in snapshot.parameters():
        parameter.requires_grad_(False)
    return snapshot


def recurrent_model_state_sha256(model: PublicRecurrentActorCritic) -> str:
    """Digest every named model tensor without serializing executable objects."""

    if not isinstance(model, PublicRecurrentActorCritic):
        raise TypeError("model must be a PublicRecurrentActorCritic")
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, Tensor):
            raise RecurrentPolicyAdapterError(
                f"model state value {name!r} is not a tensor"
            )
        contiguous = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(contiguous).all().item()):
            raise RecurrentPolicyAdapterError(
                f"model state tensor {name!r} is non-finite"
            )
        raw = contiguous.numpy().tobytes(order="C")
        descriptor = (
            f"{name}|{contiguous.dtype}|{tuple(contiguous.shape)}|{len(raw)}"
        ).encode("utf-8")
        _update_length_prefixed_digest(digest, descriptor)
        _update_length_prefixed_digest(digest, raw)
    return digest.hexdigest()


def validate_public_recurrent_history_prefix(
    prefix: Mapping[str, object],
    *,
    expected_public_input_schema_version: str = (
        ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
    ),
    expected_public_input_size: int = ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
) -> None:
    """Validate a JSON-safe, actor-public recurrent reconstruction prefix."""

    if not isinstance(prefix, Mapping):
        raise RecurrentPolicyAdapterError("public history prefix must be a mapping")
    if set(prefix) != {"schema_version", "record_count", "records"}:
        raise RecurrentPolicyAdapterError("public history prefix field set drifted")
    if prefix.get("schema_version") != RECURRENT_PUBLIC_HISTORY_PREFIX_SCHEMA_VERSION:
        raise RecurrentPolicyAdapterError("public history prefix schema drifted")
    records = prefix.get("records")
    if not isinstance(records, list):
        raise RecurrentPolicyAdapterError(
            "public history prefix records must be a list"
        )
    record_count = prefix.get("record_count")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 0
        or record_count != len(records)
    ):
        raise RecurrentPolicyAdapterError(
            "public history prefix record count does not match records"
        )
    for index, value in enumerate(records):
        if not isinstance(value, Mapping):
            raise RecurrentPolicyAdapterError(
                f"public history record {index} must be a mapping"
            )
        expected_keys = {
            "public_observation",
            "public_action_mask",
            "previous_public_feedback",
            "episode_start",
            "recurrent_state_reset_before_decision",
        }
        if set(value) != expected_keys:
            raise RecurrentPolicyAdapterError(
                f"public history record {index} field set drifted"
            )
        episode_start = _exact_bool(
            value.get("episode_start"),
            field=f"public history record {index} episode_start",
        )
        if episode_start is not (index == 0):
            raise RecurrentPolicyAdapterError(
                "public history prefix must start at the focal agent episode boundary"
            )
        reset_before = _exact_bool(
            value.get("recurrent_state_reset_before_decision"),
            field=(
                f"public history record {index} recurrent_state_reset_before_decision"
            ),
        )
        if episode_start and not reset_before:
            raise RecurrentPolicyAdapterError(
                "episode-start history record must reset recurrent state"
            )
        _validate_public_observation_payload(
            value.get("public_observation"),
            field=f"public history record {index} observation",
            expected_schema_version=expected_public_input_schema_version,
            expected_size=expected_public_input_size,
        )
        _validate_public_action_mask_payload(
            value.get("public_action_mask"),
            field=f"public history record {index} action mask",
        )
        feedback_values = _validate_public_feedback_payload(
            value.get("previous_public_feedback"),
            field=f"public history record {index} previous feedback",
        )
        if episode_start and any(item != 0.0 for item in feedback_values):
            raise RecurrentPolicyAdapterError(
                "episode-start history feedback must be all zero"
            )


def reconstruct_current_model_hidden_from_public_prefix(
    model: PublicRecurrentActorCritic,
    prefix: Mapping[str, object],
) -> Tensor:
    """Recompute hidden state using only public history and current parameters.

    No stored source hidden tensor is accepted by this API.  This makes it safe
    for post-update auxiliary learning: representation changes are replayed
    through the public prefix instead of silently consuming stale state.
    """

    if not isinstance(model, PublicRecurrentActorCritic):
        raise TypeError("model must be a PublicRecurrentActorCritic")
    validate_public_recurrent_history_prefix(
        prefix,
        expected_public_input_schema_version=(model.config.public_input_schema_version),
        expected_public_input_size=model.config.public_input_size,
    )
    reference = next(model.parameters())
    state = model.initial_state(1)
    records = prefix["records"]
    if not isinstance(records, list):
        raise AssertionError("validated public history records changed type")
    with torch.no_grad():
        for value in records:
            if not isinstance(value, Mapping):
                raise AssertionError("validated public history record changed type")
            if value["recurrent_state_reset_before_decision"] is True:
                state = model.initial_state(1)
            observation = _public_observation_tensor_from_payload(
                value["public_observation"],
                device=reference.device,
                dtype=reference.dtype,
                expected_schema_version=model.config.public_input_schema_version,
                expected_size=model.config.public_input_size,
            )
            action_mask = _public_action_mask_tensor_from_payload(
                value["public_action_mask"],
                device=reference.device,
            )
            feedback = _public_feedback_tensor_from_payload(
                value["previous_public_feedback"],
                device=reference.device,
                dtype=reference.dtype,
            )
            output = model.forward_sequence(
                observation.reshape(1, 1, -1),
                action_mask.reshape(1, 1, -1),
                feedback.reshape(1, 1, -1),
                initial_state=state,
            )
            state = output.final_state
    return state.detach().clone()


def verified_source_recurrent_state_for_exact_artifact(
    model: PublicRecurrentActorCritic,
    source_context: Mapping[str, object],
    *,
    artifact_digest: str,
) -> Tensor:
    """Load source hidden evidence only under exact artifact and model identity."""

    if not isinstance(model, PublicRecurrentActorCritic):
        raise TypeError("model must be a PublicRecurrentActorCritic")
    if not isinstance(source_context, Mapping):
        raise RecurrentPolicyAdapterError("source recurrent context must be a mapping")
    expected_artifact = source_context.get("source_artifact_digest")
    if not isinstance(artifact_digest, str) or not artifact_digest:
        raise RecurrentPolicyAdapterError("artifact_digest must be non-empty")
    if artifact_digest != expected_artifact:
        raise RecurrentPolicyAdapterError(
            "stored source hidden requires an exact artifact digest match"
        )
    expected_model_digest = source_context.get("source_model_state_sha256")
    observed_model_digest = recurrent_model_state_sha256(model)
    if observed_model_digest != expected_model_digest:
        raise RecurrentPolicyAdapterError(
            "stored source hidden requires an exact model-state digest match"
        )
    if source_context.get("source_artifact_match_required") is not True:
        raise RecurrentPolicyAdapterError(
            "stored source hidden context did not require exact artifact matching"
        )
    if source_context.get("stored_state_usage") != (
        "exact_source_artifact_verification_only"
    ):
        raise RecurrentPolicyAdapterError("stored source hidden usage is not safe")
    shape = source_context.get("source_recurrent_state_shape")
    expected_shape = [
        model.config.recurrent_layers,
        1,
        model.config.hidden_size,
    ]
    if shape != expected_shape:
        raise RecurrentPolicyAdapterError(
            "stored source hidden shape does not match the exact source model"
        )
    values = source_context.get("source_recurrent_state")
    try:
        state = torch.tensor(
            values,
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise RecurrentPolicyAdapterError(
            "stored source hidden values are not a valid tensor"
        ) from error
    if list(state.shape) != expected_shape or not bool(torch.isfinite(state).all()):
        raise RecurrentPolicyAdapterError("stored source hidden tensor is invalid")
    if _tensor_sha256(state) != source_context.get("source_recurrent_state_sha256"):
        raise RecurrentPolicyAdapterError("stored source hidden digest mismatch")
    return state.detach().clone()


@dataclass(frozen=True, slots=True)
class _PendingInferenceDecision:
    agent_id: int
    decision_index: int
    requested_action: str
    action_source: str
    genome_values: tuple[float, ...] | None = None
    genome_sha256: str | None = None
    genome_stream_seed: int | None = None
    genome_population_mode: str | None = None
    genome_population_binding_sha256: str | None = None


class DeterministicPublicRecurrentPolicy:
    """Frozen, no-fallback recurrent actor for evaluation and replay."""

    policy_id = PUBLIC_RECURRENT_POLICY_ID

    def __init__(
        self,
        model: PublicRecurrentActorCritic,
        *,
        artifact_digest: str,
        copy_to_cpu: bool = True,
        reset_recurrent_state_each_decision: bool = False,
        sampling_seed: int | None = None,
        capture_public_history: bool = False,
    ) -> None:
        if not artifact_digest or not artifact_digest.strip():
            raise RecurrentPolicyAdapterError("artifact_digest must be non-empty")
        if type(reset_recurrent_state_each_decision) is not bool:
            raise RecurrentPolicyAdapterError(
                "reset_recurrent_state_each_decision must be an exact boolean"
            )
        if sampling_seed is not None and (
            isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, int)
            or sampling_seed < 0
            or sampling_seed > 2**63 - 1
        ):
            raise RecurrentPolicyAdapterError(
                "sampling_seed must be an integer in [0, 2**63 - 1]"
            )
        if type(capture_public_history) is not bool:
            raise RecurrentPolicyAdapterError(
                "capture_public_history must be an exact boolean"
            )
        if copy_to_cpu:
            model = frozen_cpu_model_copy(model)
        elif not isinstance(model, PublicRecurrentActorCritic):
            raise TypeError("model must be a PublicRecurrentActorCritic")
        self.model = model
        self.model.eval()
        selection = (
            PUBLIC_RECURRENT_ARGMAX_SELECTION
            if sampling_seed is None
            else PUBLIC_RECURRENT_SAMPLED_SELECTION
        )
        self.policy_version = (
            f"{PUBLIC_RECURRENT_POLICY_VERSION}+{artifact_digest.strip()[:16]}"
            f"+{selection}"
        )
        self._artifact_digest = artifact_digest.strip()
        self._selection = selection
        self._sampling_seed = sampling_seed
        self._capture_public_history = capture_public_history
        self._genome_conditioning_mode = model.config.genome_conditioning_mode
        if self._genome_conditioning_mode not in {
            GENOME_CONDITIONING_DISABLED,
            GENOME_CONDITIONING_ACTOR_FILM_V1,
        }:
            raise RecurrentPolicyAdapterError(
                "model genome conditioning mode is unsupported"
            )
        self._sampling_generator: torch.Generator | None = None
        if sampling_seed is not None:
            reference = next(self.model.parameters())
            self._sampling_generator = torch.Generator(device=reference.device)
            self._sampling_generator.manual_seed(sampling_seed)
        self._reset_recurrent_state_each_decision = reset_recurrent_state_each_decision
        self._state_by_agent: dict[int, Tensor] = {}
        self._feedback_by_agent: dict[int, PreviousPublicFeedbackInput] = {}
        self._public_history_by_agent: dict[int, list[dict[str, object]]] = {}
        self._pending_by_agent: dict[int, _PendingInferenceDecision] = {}
        self._genome_population_manager: RecurrentGenomePopulationManager | None = None
        self._genome_population_pre_founder_state_sha256: str | None = None
        self._last_world_genome_provenance: dict[str, object] | None = None
        self._decision_index = 0
        self._model_state_sha256 = recurrent_model_state_sha256(self.model)
        self._parameter_versions = tuple(
            parameter._version for parameter in self.model.parameters()
        )
        self._buffer_versions = tuple(
            (name, buffer._version) for name, buffer in self.model.named_buffers()
        )

    @property
    def genome_conditioning_mode(self) -> str:
        return self._genome_conditioning_mode

    @property
    def last_world_genome_provenance(self) -> dict[str, object] | None:
        return copy.deepcopy(self._last_world_genome_provenance)

    def start_world(
        self,
        *,
        world_identity: str,
        genome_stream_seed: int,
        genome_population_mode: RecurrentGenomePopulationMode | str,
    ) -> None:
        """Bind one conditioned evaluator to an exact inherited population."""

        self._assert_model_unchanged()
        if self._genome_conditioning_mode == GENOME_CONDITIONING_DISABLED:
            raise RecurrentPolicyAdapterError(
                "disabled recurrent policy does not accept a genome population"
            )
        if self._genome_population_manager is not None:
            raise RecurrentPolicyAdapterError(
                "reset the active conditioned world before starting another"
            )
        if (
            self._state_by_agent
            or self._feedback_by_agent
            or self._public_history_by_agent
            or self._pending_by_agent
            or self._decision_index != 0
        ):
            raise RecurrentPolicyAdapterError(
                "conditioned world binding requires clean reusable policy state"
            )
        try:
            mode = (
                genome_population_mode
                if isinstance(
                    genome_population_mode,
                    RecurrentGenomePopulationMode,
                )
                else RecurrentGenomePopulationMode(genome_population_mode)
            )
        except (TypeError, ValueError) as exc:
            raise RecurrentPolicyAdapterError(
                "conditioned recurrent policy requires heritable or zero_all "
                "population mode"
            ) from exc
        if mode not in {
            RecurrentGenomePopulationMode.HERITABLE,
            RecurrentGenomePopulationMode.ZERO_ALL,
        }:
            raise RecurrentPolicyAdapterError(
                "conditioned recurrent policy requires heritable or zero_all "
                "population mode"
            )
        try:
            manager = RecurrentGenomePopulationManager(
                genome_stream_seed=genome_stream_seed,
                world_identity=world_identity,
                mode=mode,
            )
        except RecurrentGenomePopulationError as exc:
            raise RecurrentPolicyAdapterError(
                f"recurrent genome population setup failed: {exc}"
            ) from exc
        self._genome_population_manager = manager
        self._genome_population_pre_founder_state_sha256 = manager.state_sha256
        self._last_world_genome_provenance = None

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
        if self._genome_conditioning_mode == GENOME_CONDITIONING_DISABLED:
            return empty_mind_inheritance_metadata()
        self._require_zero_newborn_runtime_state(agent_id)
        manager = self._require_genome_population_manager()
        try:
            return manager.founder_metadata(agent_id=agent_id)
        except RecurrentGenomePopulationError as exc:
            raise RecurrentPolicyAdapterError(
                f"recurrent founder genome registration failed: {exc}"
            ) from exc

    def child_metadata(
        self,
        *,
        child_agent_id: int,
        primary_parent_id: int,
        secondary_parent_id: int | None,
    ) -> dict[str, object]:
        if self._genome_conditioning_mode == GENOME_CONDITIONING_DISABLED:
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
            raise RecurrentPolicyAdapterError(
                f"recurrent child genome registration failed: {exc}"
            ) from exc

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        return self._decide(
            observation,
            action_mask,
            action_override=None,
            intervention_id=None,
        )

    def decide_with_action_override(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
        *,
        requested_action: str,
        intervention_id: str,
    ) -> ActionDecision:
        """Run one diagnostics-only recurrent decision with a forced action.

        The learned actor is still evaluated and the ordinary recurrent and
        sampling states still advance.  Only the requested action returned to
        the simulator is replaced.  ``SimulationWorld`` calls :meth:`decide`,
        so this seam cannot alter ordinary runtime selection unless a separate
        diagnostics harness invokes it explicitly.
        """

        if requested_action not in ACTION_NAMES:
            raise RecurrentPolicyAdapterError(
                "counterfactual requested_action is not a stable action"
            )
        if not intervention_id or intervention_id != intervention_id.strip():
            raise RecurrentPolicyAdapterError(
                "counterfactual intervention_id must be non-empty and trimmed"
            )
        if action_mask.get(requested_action) is not True:
            raise RecurrentPolicyAdapterError(
                "counterfactual requested_action is not valid in the current mask"
            )
        return self._decide(
            observation,
            action_mask,
            action_override=requested_action,
            intervention_id=intervention_id,
        )

    def diagnostics_checkpoint_state(self, *, agent_id: int) -> dict[str, object]:
        """Return controller-derived state for an offline exact-branch checkpoint.

        The recurrent tensor and previous feedback are derived exclusively from
        prior public policy inputs.  They are optimizer/provenance context, not
        additional environment features.  Checkpoints are only valid between
        finalized decisions, so a pending transition fails closed.
        """

        self._assert_model_unchanged()
        resolved_agent_id = _positive_int(agent_id, field="checkpoint agent_id")
        if self._genome_conditioning_mode == GENOME_CONDITIONING_ACTOR_FILM_V1:
            raise RecurrentPolicyAdapterError(
                "conditioned diagnostic checkpoint requires a complete validated "
                "recurrent-genome population snapshot; this checkpoint schema "
                "cannot represent one"
            )
        if not self._capture_public_history:
            raise RecurrentPolicyAdapterError(
                "diagnostic checkpoint requires capture_public_history=True"
            )
        if resolved_agent_id in self._pending_by_agent:
            raise RecurrentPolicyAdapterError(
                "cannot checkpoint an agent with an unfinalized decision"
            )
        feedback = self._feedback_by_agent.get(
            resolved_agent_id,
            PreviousPublicFeedbackInput.zero(),
        )
        stored_state = self._state_by_agent.get(resolved_agent_id)
        recurrent_state = (
            stored_state.detach().clone()
            if stored_state is not None
            else self.model.initial_state(1)
        )
        sampling_state = (
            self._sampling_generator.get_state()
            if self._sampling_generator is not None
            else None
        )
        history_prefix = {
            "schema_version": RECURRENT_PUBLIC_HISTORY_PREFIX_SCHEMA_VERSION,
            "record_count": len(
                self._public_history_by_agent.get(resolved_agent_id, ())
            ),
            "records": copy.deepcopy(
                list(self._public_history_by_agent.get(resolved_agent_id, ()))
            ),
        }
        validate_public_recurrent_history_prefix(
            history_prefix,
            expected_public_input_schema_version=(
                self.model.config.public_input_schema_version
            ),
            expected_public_input_size=self.model.config.public_input_size,
        )
        return {
            "schema_version": RECURRENT_DIAGNOSTIC_CHECKPOINT_SCHEMA_VERSION,
            "artifact_digest": self._artifact_digest,
            "model_state_sha256": self._model_state_sha256,
            "action_selection": self._selection,
            "policy_sampling_seed": self._sampling_seed,
            "decision_index": self._decision_index,
            "agent_id": resolved_agent_id,
            "recurrent_state_initialized": stored_state is not None,
            "recurrent_state_shape": list(recurrent_state.shape),
            "recurrent_state": recurrent_state.detach().cpu().tolist(),
            "recurrent_state_sha256": _tensor_sha256(recurrent_state),
            "previous_public_feedback_available": (
                feedback.requested_action_id is not None
            ),
            "previous_public_feedback": list(feedback.values()),
            "public_history_prefix": history_prefix,
            "public_history_prefix_sha256": stable_payload_digest(history_prefix),
            "sampling_state_sha256": (
                _tensor_sha256(sampling_state) if sampling_state is not None else None
            ),
            "derived_from_public_history": True,
            "current_model_state_reconstruction": (
                "replay_public_history_prefix_from_episode_boundary"
            ),
            "private_world_state_included": False,
            "runtime_environment_input": False,
            "diagnostics_only": True,
        }

    def _decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
        *,
        action_override: str | None,
        intervention_id: str | None,
    ) -> ActionDecision:
        self._assert_model_unchanged()
        agent_id = _agent_id_from_observation(observation)
        if agent_id in self._pending_by_agent:
            raise RecurrentPolicyAdapterError(
                f"agent {agent_id} has an unfinalized prior decision"
            )
        reference = next(self.model.parameters())
        encoded_observation = encode_observation_input(observation)
        public_observation = public_policy_tensor_from_observation_input(
            encoded_observation,
            expected_schema_version=self.model.config.public_input_schema_version,
            expected_size=self.model.config.public_input_size,
            device=reference.device,
            dtype=reference.dtype,
        )
        public_mask = strict_action_mask_tensor(
            action_mask,
            device=reference.device,
        )
        feedback = self._feedback_by_agent.get(
            agent_id,
            PreviousPublicFeedbackInput.zero(),
        )
        feedback_tensor = previous_public_feedback_tensor(
            feedback,
            device=reference.device,
            dtype=reference.dtype,
        )
        recurrent_state = (
            None
            if self._reset_recurrent_state_each_decision
            else self._state_by_agent.get(agent_id)
        )
        if recurrent_state is None:
            recurrent_state = self.model.initial_state(1)
        genome_values: tuple[float, ...] | None = None
        genome_sha256: str | None = None
        genome_stream_seed: int | None = None
        genome_population_mode: str | None = None
        genome_population_binding_sha256: str | None = None
        genome_tensor: Tensor | None = None
        if self._genome_conditioning_mode == GENOME_CONDITIONING_ACTOR_FILM_V1:
            manager = self._require_genome_population_manager()
            try:
                binding = manager.genome_binding_for_agent(agent_id)
            except RecurrentGenomePopulationError as exc:
                raise RecurrentPolicyAdapterError(
                    f"recurrent genome lookup failed before decision: {exc}"
                ) from exc
            genome_values = binding.genome.values
            genome_sha256 = binding.genome_sha256
            genome_stream_seed = manager.genome_stream_seed
            genome_population_mode = manager.mode.value
            genome_population_binding_sha256 = manager.binding_sha256
            genome_tensor = torch.tensor(
                genome_values,
                device=reference.device,
                dtype=reference.dtype,
            )
        with torch.no_grad():
            if genome_tensor is None:
                selection = self.model.act(
                    public_observation,
                    public_mask,
                    feedback_tensor,
                    recurrent_state=recurrent_state,
                    deterministic=self._sampling_generator is None,
                    generator=self._sampling_generator,
                )
            else:
                selection = self.model.act(
                    public_observation,
                    public_mask,
                    feedback_tensor,
                    genome_values=genome_tensor,
                    recurrent_state=recurrent_state,
                    deterministic=self._sampling_generator is None,
                    generator=self._sampling_generator,
                )
        natural_action_index = int(selection.actions[0].item())
        natural_requested_action = ACTION_NAMES[natural_action_index]
        if not action_mask[natural_requested_action]:
            raise AssertionError("masked recurrent policy selected an invalid action")
        learned_distribution = _learned_masked_distribution_diagnostics(
            selection=selection,
            public_mask=public_mask,
            selected_action_index=natural_action_index,
        )
        requested_action = action_override or natural_requested_action
        action_source = (
            RECURRENT_COUNTERFACTUAL_ACTION_SOURCE
            if action_override is not None
            else RECURRENT_ROLLOUT_ACTION_SOURCE
        )
        decision_index = self._decision_index
        self._decision_index += 1
        if self._capture_public_history:
            history = self._public_history_by_agent.setdefault(agent_id, [])
            episode_start = len(history) == 0
            history.append(
                {
                    "public_observation": ecological_policy_input_payload(
                        encoded_observation
                    ),
                    "public_action_mask": {
                        "schema_version": ACTION_MASK_CONTRACT_VERSION,
                        "action_order": list(ACTION_NAMES),
                        "values": [
                            bool(public_mask[index].item())
                            for index in range(ACTION_COUNT)
                        ],
                    },
                    "previous_public_feedback": {
                        "schema_version": PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION,
                        "shape": [PREVIOUS_PUBLIC_FEEDBACK_SIZE],
                        "values": list(feedback.values()),
                    },
                    "episode_start": episode_start,
                    "recurrent_state_reset_before_decision": (
                        episode_start or self._reset_recurrent_state_each_decision
                    ),
                }
            )
        if not self._reset_recurrent_state_each_decision:
            self._state_by_agent[agent_id] = selection.next_state.detach().clone()
        self._pending_by_agent[agent_id] = _PendingInferenceDecision(
            agent_id=agent_id,
            decision_index=decision_index,
            requested_action=requested_action,
            action_source=action_source,
            genome_values=genome_values,
            genome_sha256=genome_sha256,
            genome_stream_seed=genome_stream_seed,
            genome_population_mode=genome_population_mode,
            genome_population_binding_sha256=(genome_population_binding_sha256),
        )
        diagnostics: dict[str, object] = {
            "schema_version": (
                PUBLIC_RECURRENT_GENOME_DIAGNOSTIC_SCHEMA_VERSION
                if genome_values is not None
                else PUBLIC_RECURRENT_DIAGNOSTIC_SCHEMA_VERSION
            ),
            "artifact_digest": self._artifact_digest,
            "model_state_sha256": self._model_state_sha256,
            "decision_index": decision_index,
            "agent_id": agent_id,
            "policy_input_schema_version": (
                self.model.config.public_input_schema_version
            ),
            "policy_input_size": self.model.config.public_input_size,
            "learned_input_size": self.model.config.learned_encoder_input_size,
            "previous_feedback_available": (feedback.requested_action_id is not None),
            "recurrent_state_reset_each_decision": (
                self._reset_recurrent_state_each_decision
            ),
            "action_index": natural_action_index,
            "action_selection": self._selection,
            "sampling_seed": self._sampling_seed,
            "value": round(float(selection.values[0].item()), 12),
            "learned_masked_distribution": learned_distribution,
        }
        if genome_values is not None:
            diagnostics.update(
                {
                    "genome_conditioning_mode": self._genome_conditioning_mode,
                    "genome_population_mode": genome_population_mode,
                    "genome_population_binding_sha256": (
                        genome_population_binding_sha256
                    ),
                    "genome_sha256": genome_sha256,
                    "genome_stream_seed": genome_stream_seed,
                }
            )
        if action_override is not None:
            diagnostics.update(
                {
                    "diagnostics_only": True,
                    "counterfactual_intervention": True,
                    "runtime_action_selection_integration": False,
                    "intervention_id": intervention_id,
                    "natural_action_index": natural_action_index,
                    "natural_requested_action": natural_requested_action,
                    "forced_action_index": ACTION_NAMES.index(requested_action),
                    "forced_requested_action": requested_action,
                    "natural_action_replaced": (
                        requested_action != natural_requested_action
                    ),
                }
            )
        return ActionDecision(
            requested_action=requested_action,
            source=action_source,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            diagnostics=diagnostics,
        )

    def observe_transition(
        self,
        record: dict[str, object],
    ) -> dict[str, object] | None:
        self._assert_model_unchanged()
        agent_id = _record_agent_id(record)
        pending = self._pending_by_agent.get(agent_id)
        action_source = record.get("action_source")
        terminated = _record_terminated(record)
        if pending is None:
            if action_source != "passive" or not terminated:
                raise RecurrentPolicyAdapterError(
                    "transition has no pending recurrent decision and is not passive"
                )
            self._require_live_genome_if_conditioned(agent_id)
        else:
            self._validate_pending_transition(record, pending=pending)

        feedback = (
            _feedback_from_record(record)
            if pending is not None and not terminated
            else None
        )
        if pending is not None:
            removed = self._pending_by_agent.pop(agent_id, None)
            if removed is not pending:
                raise AssertionError("validated pending recurrent decision changed")

        if terminated:
            self._discard_terminal_genome(agent_id)
            self._state_by_agent.pop(agent_id, None)
            self._feedback_by_agent.pop(agent_id, None)
            self._public_history_by_agent.pop(agent_id, None)
        elif pending is not None:
            assert feedback is not None
            self._feedback_by_agent[agent_id] = feedback
        return None

    def reconcile_live_agent_ids(
        self,
        *,
        live_agent_ids: Sequence[int],
    ) -> None:
        """Drop same-tick dead genome owners after trajectory finalization."""

        self._assert_model_unchanged()
        if self._genome_conditioning_mode == GENOME_CONDITIONING_DISABLED:
            return
        manager = self._require_genome_population_manager()
        try:
            planned_dead_agent_ids = manager.reconciliation_dead_agent_ids(
                live_agent_ids
            )
        except RecurrentGenomePopulationError as exc:
            raise RecurrentPolicyAdapterError(
                f"recurrent genome live-agent reconciliation failed: {exc}"
            ) from exc
        pending_dead_agent_ids = tuple(
            agent_id
            for agent_id in planned_dead_agent_ids
            if agent_id in self._pending_by_agent
        )
        if pending_dead_agent_ids:
            raise RecurrentPolicyAdapterError(
                "recurrent genome reconciliation found dead agents with "
                f"unfinalized decisions: {list(pending_dead_agent_ids)}"
            )
        discarded_agent_ids = manager.reconcile_live_agent_ids(live_agent_ids)
        if discarded_agent_ids != planned_dead_agent_ids:
            raise RecurrentPolicyAdapterError(
                "recurrent genome reconciliation plan changed before apply"
            )
        for agent_id in discarded_agent_ids:
            self._state_by_agent.pop(agent_id, None)
            self._feedback_by_agent.pop(agent_id, None)
            self._public_history_by_agent.pop(agent_id, None)

    def reset_world(self) -> None:
        self._assert_model_unchanged()
        if self._pending_by_agent:
            raise RecurrentPolicyAdapterError(
                "cannot reset a world with unfinalized decisions"
            )
        manager = self._genome_population_manager
        if manager is not None:
            pre_founder_state_sha256 = self._genome_population_pre_founder_state_sha256
            if pre_founder_state_sha256 is None:
                raise RecurrentPolicyAdapterError(
                    "conditioned world is missing pre-founder population provenance"
                )
            final_state_sha256 = manager.state_sha256
            expected_reset_state_sha256 = manager.empty_state_sha256
            if expected_reset_state_sha256 != pre_founder_state_sha256:
                raise RecurrentPolicyAdapterError(
                    "recurrent genome population pre-founder state digest drifted"
                )
            provenance: dict[str, object] = {
                "schema_version": RECURRENT_GENOME_WORLD_PROVENANCE_SCHEMA_VERSION,
                "genome_conditioning_mode": self._genome_conditioning_mode,
                "genome_population_mode": manager.mode.value,
                "action_selection": self._selection,
                "policy_sampling_seed": self._sampling_seed,
                "world_identity": manager.world_identity,
                "genome_stream_seed": manager.genome_stream_seed,
                "genome_population_binding_sha256": manager.binding_sha256,
                "genome_population_pre_founder_state_sha256": (
                    pre_founder_state_sha256
                ),
                "genome_population_final_state_sha256": final_state_sha256,
                "genome_population_reset_state_sha256": (expected_reset_state_sha256),
            }
            provenance["provenance_sha256"] = stable_payload_digest(provenance)
            manager.reset()
            if manager.state_sha256 != expected_reset_state_sha256:
                raise RecurrentPolicyAdapterError(
                    "recurrent genome population reset state digest mismatch"
                )
            self._last_world_genome_provenance = provenance
            if self._sampling_generator is not None:
                assert self._sampling_seed is not None
                self._sampling_generator.manual_seed(self._sampling_seed)
        self._state_by_agent.clear()
        self._feedback_by_agent.clear()
        self._public_history_by_agent.clear()
        self._genome_population_manager = None
        self._genome_population_pre_founder_state_sha256 = None
        self._decision_index = 0

    def _assert_model_unchanged(self) -> None:
        if self.model.config.genome_conditioning_mode != self._genome_conditioning_mode:
            raise RecurrentPolicyAdapterError(
                "frozen evaluation model genome conditioning mode changed"
            )
        observed = tuple(parameter._version for parameter in self.model.parameters())
        if observed != self._parameter_versions:
            raise RecurrentPolicyAdapterError(
                "frozen evaluation model parameters changed"
            )
        observed_buffer_versions = tuple(
            (name, buffer._version) for name, buffer in self.model.named_buffers()
        )
        if observed_buffer_versions != self._buffer_versions:
            raise RecurrentPolicyAdapterError("frozen evaluation model buffers changed")

    def _require_genome_population_manager(
        self,
    ) -> RecurrentGenomePopulationManager:
        if self._genome_conditioning_mode != GENOME_CONDITIONING_ACTOR_FILM_V1:
            raise RecurrentPolicyAdapterError(
                "genome population is unavailable for a disabled recurrent policy"
            )
        manager = self._genome_population_manager
        if manager is None:
            raise RecurrentPolicyAdapterError(
                "actor_film_v1 policy requires start_world before "
                "SimulationWorld construction"
            )
        if manager.mode not in {
            RecurrentGenomePopulationMode.HERITABLE,
            RecurrentGenomePopulationMode.ZERO_ALL,
        }:
            raise RecurrentPolicyAdapterError(
                "actor_film_v1 policy has an incompatible genome population mode"
            )
        return manager

    def _require_zero_newborn_runtime_state(self, agent_id: int) -> None:
        resolved_agent_id = _positive_int(agent_id, field="newborn agent_id")
        if resolved_agent_id in self._state_by_agent:
            raise RecurrentPolicyAdapterError(
                f"newborn agent {resolved_agent_id} already has recurrent state"
            )
        if resolved_agent_id in self._feedback_by_agent:
            raise RecurrentPolicyAdapterError(
                f"newborn agent {resolved_agent_id} already has public feedback"
            )
        if resolved_agent_id in self._public_history_by_agent:
            raise RecurrentPolicyAdapterError(
                f"newborn agent {resolved_agent_id} already has public history"
            )
        if resolved_agent_id in self._pending_by_agent:
            raise RecurrentPolicyAdapterError(
                f"newborn agent {resolved_agent_id} already has a pending decision"
            )

    def _validate_pending_transition(
        self,
        record: Mapping[str, object],
        *,
        pending: _PendingInferenceDecision,
    ) -> None:
        if _record_agent_id(record) != pending.agent_id:
            raise RecurrentPolicyAdapterError("transition agent_id mismatch")
        if record.get("policy_id") != self.policy_id:
            raise RecurrentPolicyAdapterError("transition policy_id mismatch")
        if record.get("policy_version") != self.policy_version:
            raise RecurrentPolicyAdapterError("transition policy_version mismatch")
        if record.get("action_source") != pending.action_source:
            raise RecurrentPolicyAdapterError("transition action source mismatch")
        if record.get("requested_action") != pending.requested_action:
            raise RecurrentPolicyAdapterError("transition requested action mismatch")
        self._validate_pending_genome(pending)
        if pending.genome_values is None:
            return
        diagnostics = record.get("policy_decision_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise RecurrentPolicyAdapterError(
                "conditioned transition requires decision diagnostics"
            )
        expected_genome_diagnostics = {
            "schema_version": PUBLIC_RECURRENT_GENOME_DIAGNOSTIC_SCHEMA_VERSION,
            "artifact_digest": self._artifact_digest,
            "model_state_sha256": self._model_state_sha256,
            "decision_index": pending.decision_index,
            "agent_id": pending.agent_id,
            "genome_conditioning_mode": self._genome_conditioning_mode,
            "genome_population_mode": pending.genome_population_mode,
            "genome_population_binding_sha256": (
                pending.genome_population_binding_sha256
            ),
            "genome_sha256": pending.genome_sha256,
            "genome_stream_seed": pending.genome_stream_seed,
        }
        for key, expected_value in expected_genome_diagnostics.items():
            if diagnostics.get(key) != expected_value:
                raise RecurrentPolicyAdapterError(
                    f"transition {key} does not match conditioned decision"
                )

    def _validate_pending_genome(
        self,
        pending: _PendingInferenceDecision,
    ) -> None:
        if self._genome_conditioning_mode == GENOME_CONDITIONING_DISABLED:
            if any(
                value is not None
                for value in (
                    pending.genome_values,
                    pending.genome_sha256,
                    pending.genome_stream_seed,
                    pending.genome_population_mode,
                    pending.genome_population_binding_sha256,
                )
            ):
                raise RecurrentPolicyAdapterError(
                    "disabled recurrent policy received a conditioned decision"
                )
            return
        if any(
            value is None
            for value in (
                pending.genome_values,
                pending.genome_sha256,
                pending.genome_stream_seed,
                pending.genome_population_mode,
                pending.genome_population_binding_sha256,
            )
        ):
            raise RecurrentPolicyAdapterError(
                "actor_film_v1 decision is missing controller-genome ownership"
            )
        manager = self._require_genome_population_manager()
        if pending.genome_stream_seed != manager.genome_stream_seed:
            raise RecurrentPolicyAdapterError(
                "pending controller genome stream seed does not match active world"
            )
        if pending.genome_population_mode != manager.mode.value:
            raise RecurrentPolicyAdapterError(
                "pending controller genome population mode does not match active world"
            )
        if pending.genome_population_binding_sha256 != manager.binding_sha256:
            raise RecurrentPolicyAdapterError(
                "pending controller genome population binding does not match "
                "active world"
            )
        try:
            binding = manager.genome_binding_for_agent(pending.agent_id)
        except RecurrentGenomePopulationError as exc:
            raise RecurrentPolicyAdapterError(
                f"pending controller genome ownership failed: {exc}"
            ) from exc
        if (
            pending.genome_values != binding.genome.values
            or pending.genome_sha256 != binding.genome_sha256
        ):
            raise RecurrentPolicyAdapterError(
                "pending controller genome does not match active population state"
            )

    def _require_live_genome_if_conditioned(
        self,
        agent_id: int,
    ) -> RecurrentGenomePopulationManager | None:
        if self._genome_conditioning_mode == GENOME_CONDITIONING_DISABLED:
            return None
        manager = self._require_genome_population_manager()
        try:
            manager.genome_binding_for_agent(agent_id)
        except RecurrentGenomePopulationError as exc:
            raise RecurrentPolicyAdapterError(
                f"terminal controller genome cleanup failed: {exc}"
            ) from exc
        return manager

    def _discard_terminal_genome(self, agent_id: int) -> None:
        manager = self._require_live_genome_if_conditioned(agent_id)
        if manager is None:
            return
        try:
            discarded = manager.discard_agent(agent_id)
        except RecurrentGenomePopulationError as exc:
            raise RecurrentPolicyAdapterError(
                f"terminal controller genome cleanup failed: {exc}"
            ) from exc
        if not discarded:
            raise RecurrentPolicyAdapterError(
                f"terminal controller genome cleanup missed agent {agent_id}"
            )


def validate_public_recurrent_distribution_diagnostics(
    value: Mapping[str, object],
    *,
    action_mask: Mapping[str, object] | None = None,
) -> None:
    """Validate the learned masked distribution exposed by a decision."""

    if not isinstance(value, Mapping):
        raise RecurrentPolicyAdapterError(
            "learned masked distribution diagnostics must be a mapping"
        )
    expected_keys = {
        "schema_version",
        "valid_action_count",
        "entropy",
        "normalized_entropy",
        "normalized_entropy_eligible",
        "selected_action",
        "selected_action_probability",
        "top_action",
        "top_action_probability",
        "top_two_probability_margin",
        "eat_probability",
        "masked_logits",
        "probabilities",
    }
    if set(value) != expected_keys:
        raise RecurrentPolicyAdapterError(
            "learned masked distribution diagnostics field set drifted"
        )
    if value.get("schema_version") != (
        PUBLIC_RECURRENT_DISTRIBUTION_DIAGNOSTIC_SCHEMA_VERSION
    ):
        raise RecurrentPolicyAdapterError(
            "learned masked distribution diagnostics schema drifted"
        )
    logits = value.get("masked_logits")
    probabilities = value.get("probabilities")
    if not isinstance(logits, Mapping) or set(logits) != set(ACTION_NAMES):
        raise RecurrentPolicyAdapterError(
            "learned masked logits must cover the stable action ordering"
        )
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(
        ACTION_NAMES
    ):
        raise RecurrentPolicyAdapterError(
            "learned probabilities must cover the stable action ordering"
        )
    parsed_mask: dict[str, bool]
    if action_mask is None:
        parsed_mask = {action: logits[action] is not None for action in ACTION_NAMES}
    else:
        if set(action_mask) != set(ACTION_NAMES):
            raise RecurrentPolicyAdapterError(
                "distribution validation action mask is incomplete"
            )
        parsed_mask = {
            action: _exact_bool(
                action_mask[action],
                field=f"distribution action mask {action}",
            )
            for action in ACTION_NAMES
        }
    valid_actions = [action for action in ACTION_NAMES if parsed_mask[action]]
    if not valid_actions:
        raise RecurrentPolicyAdapterError(
            "learned masked distribution has no valid actions"
        )
    valid_action_count = _positive_int(
        value.get("valid_action_count"),
        field="learned masked distribution valid-action count",
    )
    if valid_action_count != len(valid_actions):
        raise RecurrentPolicyAdapterError(
            "learned masked distribution valid-action count drifted"
        )
    parsed_probabilities: dict[str, float] = {}
    parsed_valid_logits: dict[str, float] = {}
    for action in ACTION_NAMES:
        probability = _finite_float(
            probabilities[action],
            field=f"learned probability {action}",
        )
        if probability < 0.0 or probability > 1.0:
            raise RecurrentPolicyAdapterError(
                f"learned probability {action} must be in [0, 1]"
            )
        parsed_probabilities[action] = probability
        if parsed_mask[action]:
            parsed_valid_logits[action] = _finite_float(
                logits[action],
                field=f"learned masked logit {action}",
            )
        elif logits[action] is not None or probability != 0.0:
            raise RecurrentPolicyAdapterError(
                "invalid action must have null masked logit and zero probability"
            )
    if not math.isclose(
        math.fsum(parsed_probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=2.0e-7,
    ):
        raise RecurrentPolicyAdapterError(
            "learned masked probabilities do not sum to one"
        )
    maximum_logit = max(parsed_valid_logits.values())
    softmax_weights = {
        action: math.exp(parsed_valid_logits[action] - maximum_logit)
        for action in valid_actions
    }
    softmax_denominator = math.fsum(softmax_weights.values())
    for action in valid_actions:
        _require_close(
            parsed_probabilities[action],
            softmax_weights[action] / softmax_denominator,
            field=f"learned probability {action} from masked logits",
            tolerance=2.0e-7,
        )
    selected_action = _action_name(
        value.get("selected_action"),
        field="learned selected action",
    )
    if not parsed_mask[selected_action]:
        raise RecurrentPolicyAdapterError("learned selected action is masked out")
    _require_close(
        value.get("selected_action_probability"),
        parsed_probabilities[selected_action],
        field="learned selected action probability",
    )
    ranked_actions = sorted(
        valid_actions,
        key=lambda action: (-parsed_probabilities[action], ACTION_NAMES.index(action)),
    )
    if value.get("top_action") != ranked_actions[0]:
        raise RecurrentPolicyAdapterError("learned top action drifted")
    _require_close(
        value.get("top_action_probability"),
        parsed_probabilities[ranked_actions[0]],
        field="learned top action probability",
    )
    expected_margin = (
        parsed_probabilities[ranked_actions[0]]
        - parsed_probabilities[ranked_actions[1]]
        if len(ranked_actions) > 1
        else None
    )
    if expected_margin is None:
        if value.get("top_two_probability_margin") is not None:
            raise RecurrentPolicyAdapterError(
                "one-action distribution cannot have a top-two margin"
            )
    else:
        _require_close(
            value.get("top_two_probability_margin"),
            expected_margin,
            field="learned top-two probability margin",
        )
    _require_close(
        value.get("eat_probability"),
        parsed_probabilities["eat"],
        field="learned eat probability",
    )
    entropy = _finite_float(value.get("entropy"), field="learned entropy")
    if entropy < 0.0 or entropy > math.log(len(valid_actions)) + 1.0e-6:
        raise RecurrentPolicyAdapterError("learned entropy is outside mask bounds")
    expected_entropy = -math.fsum(
        probability * math.log(probability)
        for action in valid_actions
        for probability in (parsed_probabilities[action],)
        if probability > 0.0
    )
    _require_close(
        entropy,
        expected_entropy,
        field="learned entropy from probabilities",
        tolerance=2.0e-7,
    )
    expected_eligible = len(valid_actions) > 1
    if value.get("normalized_entropy_eligible") is not expected_eligible:
        raise RecurrentPolicyAdapterError(
            "learned normalized-entropy eligibility drifted"
        )
    if expected_eligible:
        _require_close(
            value.get("normalized_entropy"),
            entropy / math.log(len(valid_actions)),
            field="learned normalized entropy",
            tolerance=2.0e-9,
        )
    elif value.get("normalized_entropy") is not None:
        raise RecurrentPolicyAdapterError(
            "one-action distribution must exclude normalized entropy"
        )


def _learned_masked_distribution_diagnostics(
    *,
    selection: ActionSelection,
    public_mask: Tensor,
    selected_action_index: int,
) -> dict[str, object]:
    logits = selection.masked_logits[0].detach().cpu()
    probabilities = torch.softmax(logits, dim=-1)
    mask_values = public_mask.detach().cpu().tolist()
    valid_indices = [index for index, valid in enumerate(mask_values) if bool(valid)]
    if not valid_indices:
        raise AssertionError("validated recurrent mask became empty")
    probability_total = math.fsum(
        float(probabilities[index].item()) for index in valid_indices
    )
    if not math.isfinite(probability_total) or probability_total <= 0.0:
        raise AssertionError("learned recurrent probabilities are non-finite")
    serialized_probabilities = [0.0 for _ in ACTION_NAMES]
    for index in valid_indices:
        serialized_probabilities[index] = round(
            float(probabilities[index].item()) / probability_total,
            12,
        )
    residual = 1.0 - math.fsum(serialized_probabilities)
    last_valid_index = valid_indices[-1]
    serialized_probabilities[last_valid_index] = round(
        serialized_probabilities[last_valid_index] + residual,
        12,
    )
    ranked_indices = sorted(
        valid_indices,
        key=lambda index: (-serialized_probabilities[index], index),
    )
    entropy = -math.fsum(
        probability * math.log(probability)
        for index in valid_indices
        for probability in (serialized_probabilities[index],)
        if probability > 0.0
    )
    normalized_entropy = (
        entropy / math.log(len(valid_indices)) if len(valid_indices) > 1 else None
    )
    result: dict[str, object] = {
        "schema_version": (PUBLIC_RECURRENT_DISTRIBUTION_DIAGNOSTIC_SCHEMA_VERSION),
        "valid_action_count": len(valid_indices),
        "entropy": round(entropy, 12),
        "normalized_entropy": (
            round(normalized_entropy, 12) if normalized_entropy is not None else None
        ),
        "normalized_entropy_eligible": len(valid_indices) > 1,
        "selected_action": ACTION_NAMES[selected_action_index],
        "selected_action_probability": serialized_probabilities[selected_action_index],
        "top_action": ACTION_NAMES[ranked_indices[0]],
        "top_action_probability": serialized_probabilities[ranked_indices[0]],
        "top_two_probability_margin": (
            round(
                serialized_probabilities[ranked_indices[0]]
                - serialized_probabilities[ranked_indices[1]],
                12,
            )
            if len(ranked_indices) > 1
            else None
        ),
        "eat_probability": serialized_probabilities[ACTION_NAMES.index("eat")],
        "masked_logits": {
            action: (
                round(float(logits[index].item()), 12)
                if bool(mask_values[index])
                else None
            )
            for index, action in enumerate(ACTION_NAMES)
        },
        "probabilities": {
            action: serialized_probabilities[index]
            for index, action in enumerate(ACTION_NAMES)
        },
    }
    validate_public_recurrent_distribution_diagnostics(
        result,
        action_mask={
            action: bool(mask_values[index])
            for index, action in enumerate(ACTION_NAMES)
        },
    )
    return result


def _validate_public_observation_payload(
    value: object,
    *,
    field: str,
    expected_schema_version: str = ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    expected_size: int = ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise RecurrentPolicyAdapterError(f"{field} must be a mapping")
    if set(value) != {"schema_version", "policy", "values", "shape"}:
        raise RecurrentPolicyAdapterError(f"{field} field set drifted")
    if expected_schema_version not in {
        ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    }:
        raise RecurrentPolicyAdapterError(f"{field} expected schema is unsupported")
    if value.get("schema_version") != expected_schema_version:
        raise RecurrentPolicyAdapterError(f"{field} schema drifted")
    if value.get("policy") != ECOLOGICAL_POLICY_INPUT_POLICY:
        raise RecurrentPolicyAdapterError(f"{field} policy drifted")
    _validate_exact_vector_shape(
        value.get("shape"),
        expected_size=expected_size,
        field=f"{field} shape",
    )
    values = value.get("values")
    if not isinstance(values, list) or len(values) != expected_size:
        raise RecurrentPolicyAdapterError(f"{field} values have the wrong size")
    parsed = tuple(_finite_float(item, field=f"{field} value") for item in values)
    if any(item < -1.0 or item > 1.0 for item in parsed):
        raise RecurrentPolicyAdapterError(f"{field} values must be in [-1, 1]")
    return parsed


def _validate_public_action_mask_payload(
    value: object,
    *,
    field: str,
) -> tuple[bool, ...]:
    if not isinstance(value, Mapping):
        raise RecurrentPolicyAdapterError(f"{field} must be a mapping")
    if set(value) != {"schema_version", "action_order", "values"}:
        raise RecurrentPolicyAdapterError(f"{field} field set drifted")
    if value.get("schema_version") != ACTION_MASK_CONTRACT_VERSION:
        raise RecurrentPolicyAdapterError(f"{field} schema drifted")
    if value.get("action_order") != list(ACTION_NAMES):
        raise RecurrentPolicyAdapterError(f"{field} action ordering drifted")
    values = value.get("values")
    if not isinstance(values, list) or len(values) != ACTION_COUNT:
        raise RecurrentPolicyAdapterError(f"{field} values have the wrong size")
    parsed = tuple(_exact_bool(item, field=f"{field} value") for item in values)
    if not any(parsed):
        raise RecurrentPolicyAdapterError(f"{field} cannot be empty")
    return parsed


def _validate_public_feedback_payload(
    value: object,
    *,
    field: str,
) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise RecurrentPolicyAdapterError(f"{field} must be a mapping")
    if set(value) != {"schema_version", "shape", "values"}:
        raise RecurrentPolicyAdapterError(f"{field} field set drifted")
    if value.get("schema_version") != PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION:
        raise RecurrentPolicyAdapterError(f"{field} schema drifted")
    _validate_exact_vector_shape(
        value.get("shape"),
        expected_size=PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        field=f"{field} shape",
    )
    values = value.get("values")
    if not isinstance(values, list) or len(values) != PREVIOUS_PUBLIC_FEEDBACK_SIZE:
        raise RecurrentPolicyAdapterError(f"{field} values have the wrong size")
    parsed = tuple(_finite_float(item, field=f"{field} value") for item in values)
    if any(item < -1.0 or item > 1.0 for item in parsed):
        raise RecurrentPolicyAdapterError(f"{field} values must be in [-1, 1]")
    try:
        validate_previous_feedback_tensor(
            torch.tensor(parsed, dtype=torch.float64),
            leading_shape=(),
        )
    except RecurrentContextError as error:
        raise RecurrentPolicyAdapterError(
            f"{field} violates the previous-feedback semantic contract"
        ) from error
    return parsed


def _public_observation_tensor_from_payload(
    value: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
    expected_schema_version: str = ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    expected_size: int = ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
) -> Tensor:
    return torch.tensor(
        _validate_public_observation_payload(
            value,
            field="public observation",
            expected_schema_version=expected_schema_version,
            expected_size=expected_size,
        ),
        device=device,
        dtype=dtype,
    )


def _public_action_mask_tensor_from_payload(
    value: object,
    *,
    device: torch.device,
) -> Tensor:
    return torch.tensor(
        _validate_public_action_mask_payload(value, field="public action mask"),
        device=device,
        dtype=torch.bool,
    )


def _public_feedback_tensor_from_payload(
    value: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return torch.tensor(
        _validate_public_feedback_payload(value, field="previous public feedback"),
        device=device,
        dtype=dtype,
    )


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    raw_bytes = bytes(contiguous.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(raw_bytes).hexdigest()


def _update_length_prefixed_digest(
    digest: "hashlib._Hash",
    value: bytes,
) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _require_close(
    value: object,
    expected: float,
    *,
    field: str,
    tolerance: float = 1.0e-9,
) -> None:
    parsed = _finite_float(value, field=field)
    if not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RecurrentPolicyAdapterError(f"{field} drifted")


def _feedback_from_record(
    record: Mapping[str, object],
) -> PreviousPublicFeedbackInput:
    requested = _action_name(record.get("requested_action"), field="requested_action")
    resolved = _action_name(record.get("resolved_action"), field="resolved_action")
    return PreviousPublicFeedbackInput(
        requested_action_id=ACTION_NAMES.index(requested),
        resolved_action_id=ACTION_NAMES.index(resolved),
        resolution_action_valid=_exact_bool(
            record.get("resolution_action_valid"),
            field="resolution_action_valid",
        ),
        moved=_exact_bool(record.get("moved"), field="moved"),
        reward_total=_reward_total(record),
    )


def _agent_id_from_observation(observation: Mapping[str, object]) -> int:
    metadata = observation.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RecurrentPolicyAdapterError("observation metadata must be a mapping")
    return _positive_int(metadata.get("agent_id"), field="observation agent_id")


def _record_agent_id(record: Mapping[str, object]) -> int:
    return _positive_int(record.get("agent_id"), field="transition agent_id")


def _record_terminated(record: Mapping[str, object]) -> bool:
    after = record.get("after")
    if not isinstance(after, Mapping):
        raise RecurrentPolicyAdapterError("transition after state must be a mapping")
    alive = _exact_bool(after.get("alive"), field="after.alive")
    return not alive


def _reward_total(record: Mapping[str, object]) -> float:
    reward = record.get("reward")
    if not isinstance(reward, Mapping):
        raise RecurrentPolicyAdapterError("transition reward must be a mapping")
    return _finite_float(reward.get("total"), field="reward.total")


def _action_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value not in ACTION_NAMES:
        raise RecurrentPolicyAdapterError(f"{field} is not a stable action")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentPolicyAdapterError(f"{field} must be a positive integer")
    return value


def _exact_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise RecurrentPolicyAdapterError(f"{field} must be an exact boolean")
    return value


def _validate_exact_vector_shape(
    value: object,
    *,
    expected_size: int,
    field: str,
) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or isinstance(value[0], bool)
        or not isinstance(value[0], int)
        or value[0] != expected_size
    ):
        raise RecurrentPolicyAdapterError(
            f"{field} must contain exactly one integer size"
        )


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentPolicyAdapterError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecurrentPolicyAdapterError(f"{field} must be finite")
    return parsed


__all__ = [
    "DeterministicPublicRecurrentPolicy",
    "PUBLIC_RECURRENT_ARGMAX_SELECTION",
    "PUBLIC_RECURRENT_DIAGNOSTIC_SCHEMA_VERSION",
    "PUBLIC_RECURRENT_DISTRIBUTION_DIAGNOSTIC_SCHEMA_VERSION",
    "PUBLIC_RECURRENT_GENOME_DIAGNOSTIC_SCHEMA_VERSION",
    "PUBLIC_RECURRENT_POLICY_ID",
    "PUBLIC_RECURRENT_POLICY_VERSION",
    "PUBLIC_RECURRENT_SAMPLED_SELECTION",
    "RECURRENT_GENOME_WORLD_PROVENANCE_SCHEMA_VERSION",
    "RECURRENT_PUBLIC_HISTORY_PREFIX_SCHEMA_VERSION",
    "RecurrentPolicyAdapterError",
    "frozen_cpu_model_copy",
    "reconstruct_current_model_hidden_from_public_prefix",
    "recurrent_model_state_sha256",
    "validate_public_recurrent_distribution_diagnostics",
    "validate_public_recurrent_history_prefix",
    "verified_source_recurrent_state_for_exact_artifact",
]
