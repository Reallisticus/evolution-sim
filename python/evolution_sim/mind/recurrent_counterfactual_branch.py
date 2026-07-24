from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import math
import random

from torch import Generator, Tensor, float32, float64, tensor, uint8

from evolution_sim.config import WorldConfig
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
import evolution_sim.env.runtime.observations as runtime_observations
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.runtime.trajectory import REWARD_TOTAL_BOUNDS
from evolution_sim.env.world import SimulationWorld
from evolution_sim.mind.evaluation_harness import (
    CONTROLLED_FIXTURE_NAMES,
    _fixture_world,
)
from evolution_sim.mind.policy_inputs import (
    ECOLOGICAL_POLICY_INPUT_POLICY,
    ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ecological_policy_input_payload,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION,
    PREVIOUS_PUBLIC_FEEDBACK_SIZE,
    PublicRecurrentActorCritic,
    RecurrentContextError,
    validate_previous_feedback_tensor,
)
from evolution_sim.mind.recurrent_policy import (
    PUBLIC_RECURRENT_ARGMAX_SELECTION,
    PUBLIC_RECURRENT_SAMPLED_SELECTION,
    RECURRENT_COUNTERFACTUAL_ACTION_SOURCE,
    DeterministicPublicRecurrentPolicy,
    RecurrentPolicyAdapterError,
    reconstruct_current_model_hidden_from_public_prefix,
    recurrent_model_state_sha256,
    validate_public_recurrent_distribution_diagnostics,
    validate_public_recurrent_history_prefix,
    verified_source_recurrent_state_for_exact_artifact,
)
from evolution_sim.mind.recurrent_rollout import RECURRENT_ROLLOUT_ACTION_SOURCE
from evolution_sim.mind.recurrent_seed_registry import (
    RECURRENT_SEED_REGISTRY,
    SCALE_DEVELOPMENT_SEED_REGISTRY,
    SCALE_DEVELOPMENT_V2_SEED_REGISTRY,
)


RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION = (
    "mind_v3_recurrent_counterfactual_branch_row_v4"
)
RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION = (
    "mind_v3_recurrent_counterfactual_boundary_source_branch_row_v5"
)
RECURRENT_COUNTERFACTUAL_BRANCH_PROTOCOL_VERSION = (
    "mind_v3_recurrent_exact_all_valid_action_branch_protocol_v4"
)
RECURRENT_COUNTERFACTUAL_AGGREGATE_SCHEMA_VERSION = (
    "mind_v3_recurrent_counterfactual_multi_tape_aggregate_v2"
)
RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE = (
    "mind_v3_recurrent_counterfactual_continuation_environment_tape_seed_v2"
)
RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE = (
    "mind_v3_recurrent_counterfactual_continuation_policy_tape_seed_v2"
)
RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY = (
    "after_focal_natural_policy_draw_before_action_resolution_v1"
)
RECURRENT_COUNTERFACTUAL_TAPE_CHUNK_SCHEMA_VERSION = (
    "mind_v3_recurrent_counterfactual_tape_chunk_v1"
)
_DECISION_BOUNDARY_PROVENANCE_DIGEST_FIELDS = (
    "pre_boundary_environment_rng_state_sha256",
    "pre_boundary_policy_sampling_state_sha256",
    "post_boundary_environment_rng_state_sha256",
    "post_boundary_policy_sampling_state_sha256",
)
_DECISION_BOUNDARY_PROVENANCE_BOOLEAN_FIELDS = (
    "boundary_reached",
    "source_natural_action_match",
    "fixed_source_prefix_verified",
)
_COMPACT_FIRST_TRANSITION_FIELDS = {
    "requested_action",
    "resolved_action",
    "observation_action_valid",
    "resolution_action_valid",
    "same_tick_resolution_mask_drift",
    "expected_same_tick_occupancy_drift",
    "invalid_reason",
    "moved",
    "reward_total",
    "reproduced",
    "died",
    "after_alive",
    "after_energy_ratio",
    "after_hydration_ratio",
    "after_health_ratio",
}
_BRANCH_ROW_FIELDS = {
    "schema_version",
    "contract",
    "trainable_public_context",
    "optimizer_context",
    "labels",
    "metadata",
    "component_digests",
    "training_ran",
    "training_artifact_created",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "promotion_authorized",
    "exact_digest",
}
_BRANCH_LABEL_FIELDS = {
    "source_requested_action",
    "source_policy_value",
    "source_behavior_distribution",
    "baseline",
    "action_outcomes",
    "all_tick_start_mask_valid_actions_evaluated",
    "valid_action_count",
    "baseline_source_action_behavior_match",
}
_COMPACT_BRANCH_BASELINE_FIELDS = {
    "first_transition",
    "focal_transition_count",
    "focal_policy_decision_count",
    "focal_passive_transition_count",
    "focal_discounted_return",
    "focal_terminal",
    "population_alive",
    "births_during_horizon",
    "deaths_during_horizon",
    "behavior_digest",
    "evidence_digest",
}
_COMPACT_BRANCH_ACTION_FIELDS = {
    *_COMPACT_BRANCH_BASELINE_FIELDS,
    "action",
    "natural_requested_action",
    "intervention_count",
    "paired_vs_baseline",
    "unsupported_requested_action_count",
    "heuristic_action_source_count",
    "unexpected_action_source_count",
    "replay_verified",
    "replay_evidence_digest",
}
_COMPACT_MULTI_TAPE_COMMON_OUTCOME_FIELDS = {
    "natural_requested_action",
    "focal_discounted_return",
    "focal_terminal_alive",
    "population_alive",
    "births_during_horizon",
    "deaths_during_horizon",
    "first_transition",
    "unsupported_requested_action_count",
    "heuristic_action_source_count",
    "unexpected_action_source_count",
    "behavior_digest",
    "evidence_digest",
    "replay_verified",
    "replay_evidence_digest",
    "continuation_rng_retape_boundary",
    *_DECISION_BOUNDARY_PROVENANCE_DIGEST_FIELDS,
    *_DECISION_BOUNDARY_PROVENANCE_BOOLEAN_FIELDS,
}
_COMPACT_MULTI_TAPE_BASELINE_FIELDS = _COMPACT_MULTI_TAPE_COMMON_OUTCOME_FIELDS
_COMPACT_MULTI_TAPE_ACTION_FIELDS = {
    *_COMPACT_MULTI_TAPE_COMMON_OUTCOME_FIELDS,
    "action",
    "paired_vs_baseline",
}
_MULTI_TAPE_PROVENANCE_FIELDS = {
    "tape_index",
    "environment_sampling_identity",
    "environment_sampling_seed",
    "policy_sampling_identity",
    "policy_sampling_seed",
    "continuation_rng_retape_boundary",
    *_DECISION_BOUNDARY_PROVENANCE_DIGEST_FIELDS,
    *_DECISION_BOUNDARY_PROVENANCE_BOOLEAN_FIELDS,
    "baseline",
    "action_outcomes",
}
_SAMPLE_STATISTIC_FIELDS = {
    "mean",
    "sample_variance",
    "standard_error",
}
_AGGREGATE_METRIC_FIELDS = {
    "focal_discounted_return",
    "focal_terminal_alive",
    "population_alive",
    "births_during_horizon",
    "deaths_during_horizon",
}
_MULTI_TAPE_ACTION_AGGREGATE_FIELDS = {
    "action",
    "tape_count",
    "outcome_statistics",
    "paired_delta_statistics",
    "focal_survival_probability",
    "uncertainty_penalized_score",
}
_MULTI_TAPE_AGGREGATE_FIELDS = {
    "tape_count",
    "uncertainty_penalty",
    "uncertainty_penalized_score_policy",
    "baseline_outcome_statistics",
    "baseline_focal_survival_probability",
    "action_outcomes",
}
MAX_RECURRENT_COUNTERFACTUAL_TAPE_SEED = 2**63 - 1
RECURRENT_COUNTERFACTUAL_TRAINING_USE = (
    "post_update_supervised_auxiliary_policy_improvement_only"
)
RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO = "broad"
RECURRENT_COUNTERFACTUAL_LEGACY_SEED_ROLES = ("train", "curriculum")
RECURRENT_COUNTERFACTUAL_SCALE_V1_SEED_ROLES = (
    "scale_train",
    "scale_curriculum",
)
RECURRENT_COUNTERFACTUAL_SCALE_V2_SEED_ROLES = (
    "scale_v2_train",
    "scale_v2_curriculum",
)
RECURRENT_COUNTERFACTUAL_SCALE_SEED_ROLES = (
    *RECURRENT_COUNTERFACTUAL_SCALE_V1_SEED_ROLES,
    *RECURRENT_COUNTERFACTUAL_SCALE_V2_SEED_ROLES,
)
RECURRENT_COUNTERFACTUAL_SEED_ROLES = (
    *RECURRENT_COUNTERFACTUAL_LEGACY_SEED_ROLES,
    *RECURRENT_COUNTERFACTUAL_SCALE_SEED_ROLES,
)
RECURRENT_COUNTERFACTUAL_SCENARIOS = (
    RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO,
    *CONTROLLED_FIXTURE_NAMES,
)
_TRAINABLE_PUBLIC_CONTEXT_KEYS = {
    "public_history_prefix",
    "current_public_observation",
    "current_public_action_mask",
    "previous_public_feedback",
}
_OPTIMIZER_CONTEXT_KEYS = {
    "source_artifact_digest",
    "source_model_state_sha256",
    "source_recurrent_state",
    "source_recurrent_state_shape",
    "source_recurrent_state_sha256",
    "source_public_history_prefix_sha256",
    "derived_from_public_history",
    "source_artifact_match_required",
    "stored_state_usage",
    "current_model_state_policy",
    "runtime_environment_input",
}
_CONTINUATION_PROVENANCE_KEYS = {
    "continuation_policy_artifact_digest",
    "continuation_policy_model_state_sha256",
    "continuation_policy_action_selection",
    "continuation_policy_sampling_seed",
    "continuation_checkpoint_sampling_state_sha256",
    "branch_horizon_ticks",
    "exact_replay_repeat_count_per_action",
    "repeat_indices",
    "repeat_policy_sampling_seeds",
    "repeat_checkpoint_sampling_state_sha256",
    "common_checkpoint_across_actions",
    "source_checkpoint_reused_by_deepcopy",
}
_METADATA_KEYS = {
    "seed_role",
    "environment_seed",
    "scenario",
    "branch_tick",
    "horizon_ticks",
    "focal_agent_id",
    "gamma",
    "source_artifact_digest",
    "source_model_state_sha256",
    "policy_sampling_seed",
    "source_action_selection",
    "source_decision_index",
    "source_record_digest",
    "source_decision_diagnostics_digest",
    "source_observation_digest",
    "source_action_mask_digest",
    "source_sampling_state_sha256",
    "source_public_history_prefix_sha256",
    "branch_identity_digest",
    "branch_protocol_digest",
    "continuation_provenance",
    "continuation_provenance_digest",
    "baseline_behavior_digest",
    "baseline_evidence_digest",
    "private_checkpoint_serialized",
}
_BOUNDARY_SOURCE_METADATA_KEYS = {
    *_METADATA_KEYS,
    "source_pre_boundary_environment_rng_state_sha256",
    "source_pre_boundary_policy_sampling_state_sha256",
    "source_boundary_identity_sha256",
}
_FORBIDDEN_TRAINABLE_KEY_TOKENS = (
    "seed",
    "fixture",
    "scenario",
    "tick",
    "agent",
    "path",
    "digest",
    "provenance",
    "private",
    "outcome",
    "target",
    "label",
    "metadata",
    "split",
)
_PAIRED_DELTA_FIELDS = (
    "focal_discounted_return_delta",
    "focal_terminal_alive_delta",
    "population_alive_delta",
    "births_during_horizon_delta",
    "deaths_during_horizon_delta",
)
_SOURCE_DECISION_PREFIX_PROJECTION_FIELDS = (
    "schema_version",
    "artifact_digest",
    "model_state_sha256",
    "decision_index",
    "agent_id",
    "previous_feedback_available",
    "recurrent_state_reset_each_decision",
    "action_index",
    "action_selection",
    "sampling_seed",
    "value",
    "learned_masked_distribution",
)


class RecurrentCounterfactualBranchError(ValueError):
    """Raised when an exact recurrent counterfactual contract fails closed."""


def derive_recurrent_counterfactual_tape_seed(
    *,
    namespace: str,
    identity: str,
) -> int:
    """Derive one independent continuation-tape seed from explicit provenance."""

    if namespace not in {
        RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE,
        RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE,
    }:
        raise RecurrentCounterfactualBranchError(
            "counterfactual continuation tape seed namespace is unsupported"
        )
    resolved_identity = _nonempty_string(identity, field="tape seed identity")
    payload = f"{namespace}|{resolved_identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        MAX_RECURRENT_COUNTERFACTUAL_TAPE_SEED
    )


class OneShotRecurrentCounterfactualPolicy:
    """Diagnostics-only focal intervention over a frozen recurrent delegate."""

    def __init__(
        self,
        *,
        delegate: DeterministicPublicRecurrentPolicy,
        target_agent_id: int,
        forced_action: str,
        expected_observation_digest: str,
        expected_action_mask_digest: str,
        intervention_id: str,
    ) -> None:
        if not isinstance(delegate, DeterministicPublicRecurrentPolicy):
            raise TypeError("delegate must be a DeterministicPublicRecurrentPolicy")
        self.delegate = delegate
        self.target_agent_id = _positive_int(
            target_agent_id,
            field="target_agent_id",
        )
        if forced_action not in ACTION_NAMES:
            raise RecurrentCounterfactualBranchError(
                "forced_action is not a stable action"
            )
        self.forced_action = forced_action
        self.expected_observation_digest = _nonempty_string(
            expected_observation_digest,
            field="expected_observation_digest",
        )
        self.expected_action_mask_digest = _nonempty_string(
            expected_action_mask_digest,
            field="expected_action_mask_digest",
        )
        self.intervention_id = _nonempty_string(
            intervention_id,
            field="intervention_id",
        )
        self.used = False
        self.intervention_count = 0
        self.natural_requested_action: str | None = None
        self.intervention_diagnostics: dict[str, object] | None = None

    @property
    def policy_id(self) -> str:
        return self.delegate.policy_id

    @property
    def policy_version(self) -> str:
        return self.delegate.policy_version

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        agent_id = _observation_agent_id(observation)
        if self.used or agent_id != self.target_agent_id:
            return self.delegate.decide(observation, action_mask)

        observation_digest = runtime_observations.observation_digest(observation)
        if observation_digest != self.expected_observation_digest:
            raise RecurrentCounterfactualBranchError(
                "focal observation digest does not match the materialized state"
            )
        complete_mask = _complete_action_mask(action_mask)
        if stable_payload_digest(complete_mask) != self.expected_action_mask_digest:
            raise RecurrentCounterfactualBranchError(
                "focal action-mask digest does not match the materialized state"
            )
        if complete_mask[self.forced_action] is not True:
            raise RecurrentCounterfactualBranchError(
                "forced action is not valid in the materialized tick-start mask"
            )

        decision = self.delegate.decide_with_action_override(
            observation,
            action_mask,
            requested_action=self.forced_action,
            intervention_id=self.intervention_id,
        )
        diagnostics = _mapping(
            decision.diagnostics,
            field="counterfactual decision diagnostics",
        )
        natural_action = diagnostics.get("natural_requested_action")
        if not isinstance(natural_action, str) or natural_action not in ACTION_NAMES:
            raise RecurrentCounterfactualBranchError(
                "counterfactual decision did not record its natural action"
            )
        if decision.requested_action != self.forced_action:
            raise RecurrentCounterfactualBranchError(
                "counterfactual decision did not return the forced action"
            )
        self.used = True
        self.intervention_count += 1
        self.natural_requested_action = natural_action
        self.intervention_diagnostics = dict(diagnostics)
        return decision

    def observe_transition(
        self,
        record: dict[str, object],
    ) -> dict[str, object] | None:
        return self.delegate.observe_transition(record)

    def reset_world(self) -> None:
        self.delegate.reset_world()
        self.used = False
        self.intervention_count = 0
        self.natural_requested_action = None
        self.intervention_diagnostics = None


class _DecisionBoundaryRetapedRecurrentCounterfactualPolicy:
    """Retape continuation RNG only after the exact focal natural draw.

    This diagnostics-only wrapper preserves the source checkpoint, all
    pre-focal decisions, and the focal learner draw.  It replaces future
    environment and policy randomness at the causal decision boundary, before
    Foundation resolves either the natural baseline action or a forced action.
    """

    def __init__(
        self,
        *,
        delegate: DeterministicPublicRecurrentPolicy,
        environment_rng: random.Random,
        target_agent_id: int,
        source_action: str,
        forced_action: str | None,
        expected_observation_digest: str,
        expected_action_mask_digest: str,
        expected_source_decision_projection_sha256: str,
        intervention_id: str,
        environment_sampling_seed: int,
        policy_sampling_seed: int,
    ) -> None:
        if not isinstance(delegate, DeterministicPublicRecurrentPolicy):
            raise TypeError("delegate must be a DeterministicPublicRecurrentPolicy")
        if not isinstance(environment_rng, random.Random):
            raise TypeError("environment_rng must be random.Random")
        self.delegate = delegate
        self.environment_rng = environment_rng
        self.target_agent_id = _positive_int(
            target_agent_id,
            field="target_agent_id",
        )
        self.source_action = _stable_action(
            source_action,
            field="source_action",
        )
        if forced_action is not None:
            _stable_action(forced_action, field="forced_action")
        self.forced_action = forced_action
        self.expected_observation_digest = _nonempty_string(
            expected_observation_digest,
            field="expected_observation_digest",
        )
        self.expected_action_mask_digest = _nonempty_string(
            expected_action_mask_digest,
            field="expected_action_mask_digest",
        )
        if not _valid_sha256(expected_source_decision_projection_sha256):
            raise RecurrentCounterfactualBranchError(
                "expected source decision projection digest is malformed"
            )
        self.expected_source_decision_projection_sha256 = (
            expected_source_decision_projection_sha256
        )
        self.intervention_id = _nonempty_string(
            intervention_id,
            field="intervention_id",
        )
        self.environment_sampling_seed = _required_sampling_seed(
            environment_sampling_seed,
            field="environment_sampling_seed",
        )
        self.policy_sampling_seed = _required_sampling_seed(
            policy_sampling_seed,
            field="policy_sampling_seed",
        )
        if self.environment_sampling_seed == self.policy_sampling_seed:
            raise RecurrentCounterfactualBranchError(
                "environment and policy continuation seeds must differ"
            )
        if delegate._sampling_generator is None:  # noqa: SLF001 - diagnostics seam
            raise RecurrentCounterfactualBranchError(
                "decision-boundary retaping requires sampled recurrent policy state"
            )
        self.used = False
        self.intervention_count = 0
        self.natural_requested_action: str | None = None
        self.boundary_provenance: dict[str, object] | None = None

    @property
    def policy_id(self) -> str:
        return self.delegate.policy_id

    @property
    def policy_version(self) -> str:
        return self.delegate.policy_version

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        agent_id = _observation_agent_id(observation)
        if self.used or agent_id != self.target_agent_id:
            return self.delegate.decide(observation, action_mask)

        observation_digest = runtime_observations.observation_digest(observation)
        if observation_digest != self.expected_observation_digest:
            raise RecurrentCounterfactualBranchError(
                "focal observation digest does not match the materialized state"
            )
        complete_mask = _complete_action_mask(action_mask)
        if stable_payload_digest(complete_mask) != self.expected_action_mask_digest:
            raise RecurrentCounterfactualBranchError(
                "focal action-mask digest does not match the materialized state"
            )
        if (
            self.forced_action is not None
            and complete_mask[self.forced_action] is not True
        ):
            raise RecurrentCounterfactualBranchError(
                "forced action is not valid in the materialized tick-start mask"
            )

        if self.forced_action is None:
            decision = self.delegate.decide(observation, action_mask)
            natural_action = _stable_action(
                decision.requested_action,
                field="baseline natural action",
            )
        else:
            decision = self.delegate.decide_with_action_override(
                observation,
                action_mask,
                requested_action=self.forced_action,
                intervention_id=self.intervention_id,
            )
            diagnostics = _mapping(
                decision.diagnostics,
                field="counterfactual decision diagnostics",
            )
            natural_action = _stable_action(
                diagnostics.get("natural_requested_action"),
                field="counterfactual natural action",
            )
            if decision.requested_action != self.forced_action:
                raise RecurrentCounterfactualBranchError(
                    "counterfactual decision did not return the forced action"
                )
        if natural_action != self.source_action:
            raise RecurrentCounterfactualBranchError(
                "branch learner natural action drifted from the source decision"
            )
        diagnostics = _mapping(
            decision.diagnostics,
            field="decision-boundary learner diagnostics",
        )
        if _source_decision_prefix_projection_sha256(diagnostics) != (
            self.expected_source_decision_projection_sha256
        ):
            raise RecurrentCounterfactualBranchError(
                "branch learner decision projection drifted from the fixed "
                "source prefix"
            )

        pre_environment_digest = stable_payload_digest(
            self.environment_rng.getstate()
        )
        pre_policy_digest = _policy_sampling_state_sha256(self.delegate)
        self.environment_rng.seed(self.environment_sampling_seed)
        sampling_generator = self.delegate._sampling_generator  # noqa: SLF001
        if sampling_generator is None:
            raise RecurrentCounterfactualBranchError(
                "decision-boundary policy sampling generator disappeared"
            )
        sampling_generator.manual_seed(self.policy_sampling_seed)
        self.delegate._sampling_seed = self.policy_sampling_seed  # noqa: SLF001
        post_environment_digest = stable_payload_digest(
            self.environment_rng.getstate()
        )
        post_policy_digest = _policy_sampling_state_sha256(self.delegate)
        if post_environment_digest != _seeded_environment_rng_state_sha256(
            self.environment_sampling_seed
        ):
            raise RecurrentCounterfactualBranchError(
                "environment continuation seed did not produce its canonical state"
            )
        if post_policy_digest != _seeded_policy_sampling_state_sha256(
            self.policy_sampling_seed
        ):
            raise RecurrentCounterfactualBranchError(
                "policy continuation seed did not produce its canonical CPU state"
            )
        self.boundary_provenance = {
            "continuation_rng_retape_boundary": (
                RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY
            ),
            "pre_boundary_environment_rng_state_sha256": pre_environment_digest,
            "pre_boundary_policy_sampling_state_sha256": pre_policy_digest,
            "post_boundary_environment_rng_state_sha256": post_environment_digest,
            "post_boundary_policy_sampling_state_sha256": post_policy_digest,
            "boundary_reached": True,
            "source_natural_action_match": True,
            "fixed_source_prefix_verified": True,
        }
        self.used = True
        self.natural_requested_action = natural_action
        if self.forced_action is not None:
            self.intervention_count += 1
        return decision

    def observe_transition(
        self,
        record: dict[str, object],
    ) -> dict[str, object] | None:
        return self.delegate.observe_transition(record)

    def reset_world(self) -> None:
        self.delegate.reset_world()
        self.used = False
        self.intervention_count = 0
        self.natural_requested_action = None
        self.boundary_provenance = None


def build_recurrent_counterfactual_branch_row(
    model: PublicRecurrentActorCritic,
    *,
    artifact_digest: str,
    seed_role: str,
    environment_seed: int,
    scenario: str,
    branch_tick: int,
    horizon_ticks: int = 32,
    focal_agent_id: int | None = None,
    policy_sampling_seed: int | None = None,
    gamma: float = 0.99,
    verify_replay: bool = True,
) -> dict[str, object]:
    """Query every valid focal action at one real recurrent-policy state.

    The function trains nothing and changes no default policy.  It materializes
    a pre-tick world checkpoint reached by the supplied frozen learner, runs an
    unforced baseline, and compares it with one exact intervention per action
    in the focal tick-start public mask.
    """

    if not isinstance(model, PublicRecurrentActorCritic):
        raise TypeError("model must be a PublicRecurrentActorCritic")
    resolved_artifact_digest = _nonempty_string(
        artifact_digest,
        field="artifact_digest",
    )
    resolved_seed_role = _validated_seed_role(seed_role, environment_seed)
    resolved_scenario = _validated_scenario(scenario)
    resolved_branch_tick = _nonnegative_int(branch_tick, field="branch_tick")
    resolved_horizon = _positive_int(horizon_ticks, field="horizon_ticks")
    resolved_gamma = _unit_interval(gamma, field="gamma")
    resolved_policy_sampling_seed = _optional_sampling_seed(
        policy_sampling_seed,
        field="policy_sampling_seed",
    )
    if focal_agent_id is not None:
        focal_agent_id = _positive_int(focal_agent_id, field="focal_agent_id")
    if type(verify_replay) is not bool:
        raise RecurrentCounterfactualBranchError(
            "verify_replay must be an exact boolean"
        )

    total_ticks = resolved_branch_tick + resolved_horizon
    policy = DeterministicPublicRecurrentPolicy(
        model,
        artifact_digest=resolved_artifact_digest,
        sampling_seed=resolved_policy_sampling_seed,
        capture_public_history=True,
    )
    source_world = _world_for_scenario(
        scenario=resolved_scenario,
        environment_seed=environment_seed,
        ticks=total_ticks,
        policy=policy,
    )
    _configure_manual_world(source_world)
    for tick in range(resolved_branch_tick):
        if not source_world.alive_agents():
            raise RecurrentCounterfactualBranchError(
                "source world terminated before the requested branch tick"
            )
        source_world.tick = tick
        source_world._run_tick()
    if not source_world.alive_agents():
        raise RecurrentCounterfactualBranchError(
            "source world has no learner decision at the requested branch tick"
        )

    checkpoint_world = deepcopy(source_world)
    source_diagnostics_start = len(source_world.policy_decision_diagnostics_records)
    source_world.tick = resolved_branch_tick
    source_world._run_tick()
    source_records = tuple(source_world.tick_trajectory_records)
    source_diagnostics = tuple(
        source_world.policy_decision_diagnostics_records[source_diagnostics_start:]
    )
    source_record, source_decision_diagnostics = _select_focal_source_record(
        source_records,
        source_diagnostics,
        focal_agent_id=focal_agent_id,
    )
    resolved_focal_agent_id = _positive_int(
        source_record.get("agent_id"),
        field="source focal agent_id",
    )
    source_action = _stable_action(
        source_record.get("requested_action"),
        field="source requested_action",
    )
    source_mask = _complete_action_mask(source_record.get("action_mask"))
    valid_actions = tuple(
        action for action in ACTION_NAMES if source_mask[action] is True
    )
    if source_action not in valid_actions:
        raise RecurrentCounterfactualBranchError(
            "source learner requested an action outside its tick-start mask"
        )
    source_observation_digest = _nonempty_string(
        source_record.get("observation_digest"),
        field="source observation_digest",
    )
    source_action_mask_digest = stable_payload_digest(source_mask)
    checkpoint_policy = checkpoint_world.policy
    if not isinstance(checkpoint_policy, DeterministicPublicRecurrentPolicy):
        raise RecurrentCounterfactualBranchError(
            "materialized checkpoint does not own the recurrent policy"
        )
    checkpoint_context = checkpoint_policy.diagnostics_checkpoint_state(
        agent_id=resolved_focal_agent_id,
    )
    source_model_state_sha256 = recurrent_model_state_sha256(model)
    if checkpoint_context.get("artifact_digest") != resolved_artifact_digest:
        raise RecurrentCounterfactualBranchError(
            "checkpoint source artifact digest does not match the requested artifact"
        )
    if checkpoint_context.get("model_state_sha256") != source_model_state_sha256:
        raise RecurrentCounterfactualBranchError(
            "checkpoint source model digest does not match the supplied model"
        )
    public_history_prefix = deepcopy(checkpoint_context.get("public_history_prefix"))
    if not isinstance(public_history_prefix, Mapping):
        raise RecurrentCounterfactualBranchError(
            "checkpoint did not provide a public recurrent history prefix"
        )
    try:
        validate_public_recurrent_history_prefix(public_history_prefix)
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "checkpoint public recurrent history prefix is invalid"
        ) from error
    public_history_prefix_sha256 = stable_payload_digest(public_history_prefix)
    if (
        checkpoint_context.get("public_history_prefix_sha256")
        != public_history_prefix_sha256
    ):
        raise RecurrentCounterfactualBranchError(
            "checkpoint public recurrent history prefix digest mismatch"
        )
    branch_id = (
        f"recurrent-counterfactual-{resolved_seed_role}-{resolved_scenario}-"
        f"seed-{environment_seed}-tick-{resolved_branch_tick}-"
        f"agent-{resolved_focal_agent_id}"
    )

    baseline = _execute_continuation(
        checkpoint_world,
        branch_id=branch_id,
        branch_tick=resolved_branch_tick,
        horizon_ticks=resolved_horizon,
        focal_agent_id=resolved_focal_agent_id,
        source_action=source_action,
        expected_observation_digest=source_observation_digest,
        expected_action_mask_digest=source_action_mask_digest,
        forced_action=None,
        gamma=resolved_gamma,
    )
    action_runs: list[dict[str, object]] = []
    for action in valid_actions:
        run = _execute_continuation(
            checkpoint_world,
            branch_id=branch_id,
            branch_tick=resolved_branch_tick,
            horizon_ticks=resolved_horizon,
            focal_agent_id=resolved_focal_agent_id,
            source_action=source_action,
            expected_observation_digest=source_observation_digest,
            expected_action_mask_digest=source_action_mask_digest,
            forced_action=action,
            gamma=resolved_gamma,
        )
        replay_verified: bool | None = None
        replay_digest: str | None = None
        if verify_replay:
            replay = _execute_continuation(
                checkpoint_world,
                branch_id=branch_id,
                branch_tick=resolved_branch_tick,
                horizon_ticks=resolved_horizon,
                focal_agent_id=resolved_focal_agent_id,
                source_action=source_action,
                expected_observation_digest=source_observation_digest,
                expected_action_mask_digest=source_action_mask_digest,
                forced_action=action,
                gamma=resolved_gamma,
            )
            replay_digest = str(replay["evidence_digest"])
            replay_verified = (
                replay_digest == run["evidence_digest"]
                and replay["behavior_digest"] == run["behavior_digest"]
            )
            if not replay_verified:
                raise RecurrentCounterfactualBranchError(
                    f"counterfactual replay mismatch for action {action!r}"
                )
        run["replay_verified"] = replay_verified
        run["replay_evidence_digest"] = replay_digest
        action_runs.append(run)

    natural_run = next(
        (run for run in action_runs if run["forced_action"] == source_action),
        None,
    )
    if natural_run is None:
        raise RecurrentCounterfactualBranchError(
            "source action was not included in all-valid-action enumeration"
        )
    baseline_behavior_match = (
        natural_run["behavior_digest"] == baseline["behavior_digest"]
    )
    if not baseline_behavior_match:
        raise RecurrentCounterfactualBranchError(
            "source-action intervention did not reproduce baseline behavior"
        )

    source_behavior_distribution = deepcopy(
        source_decision_diagnostics.get("learned_masked_distribution")
    )
    if not isinstance(source_behavior_distribution, Mapping):
        raise RecurrentCounterfactualBranchError(
            "source decision lacks learned masked distribution diagnostics"
        )
    try:
        validate_public_recurrent_distribution_diagnostics(
            source_behavior_distribution,
            action_mask=source_mask,
        )
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "source learned masked distribution diagnostics are invalid"
        ) from error

    trainable_public_context = {
        "public_history_prefix": public_history_prefix,
        "current_public_observation": ecological_policy_input_payload(
            _mapping(
                source_record.get("observation_input"),
                field="source observation_input",
            )
        ),
        "current_public_action_mask": source_mask,
        "previous_public_feedback": {
            "schema_version": PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION,
            "shape": [PREVIOUS_PUBLIC_FEEDBACK_SIZE],
            "values": list(checkpoint_context["previous_public_feedback"]),
        },
    }
    optimizer_context = {
        "source_artifact_digest": resolved_artifact_digest,
        "source_model_state_sha256": source_model_state_sha256,
        "source_recurrent_state": deepcopy(checkpoint_context["recurrent_state"]),
        "source_recurrent_state_shape": list(
            checkpoint_context["recurrent_state_shape"]
        ),
        "source_recurrent_state_sha256": checkpoint_context["recurrent_state_sha256"],
        "source_public_history_prefix_sha256": public_history_prefix_sha256,
        "derived_from_public_history": True,
        "source_artifact_match_required": True,
        "stored_state_usage": "exact_source_artifact_verification_only",
        "current_model_state_policy": (
            "reconstruct_from_trainable_public_history_prefix"
        ),
        "runtime_environment_input": False,
    }
    labels = {
        "source_requested_action": source_action,
        "source_policy_value": _finite_float(
            source_decision_diagnostics.get("value"),
            field="source policy value",
        ),
        "source_behavior_distribution": dict(source_behavior_distribution),
        "baseline": _compact_baseline_label(baseline),
        "action_outcomes": [
            _compact_action_label(run, baseline=baseline) for run in action_runs
        ],
        "all_tick_start_mask_valid_actions_evaluated": True,
        "valid_action_count": len(valid_actions),
        "baseline_source_action_behavior_match": baseline_behavior_match,
    }
    contract = _counterfactual_contract(
        verify_replay=verify_replay,
        seed_role=resolved_seed_role,
    )
    protocol_digest = stable_payload_digest(contract)
    source_record_digest = stable_payload_digest(source_record)
    source_diagnostics_digest = stable_payload_digest(source_decision_diagnostics)
    repeat_count = 2 if verify_replay else 1
    continuation_provenance = {
        "continuation_policy_artifact_digest": resolved_artifact_digest,
        "continuation_policy_model_state_sha256": source_model_state_sha256,
        "continuation_policy_action_selection": checkpoint_context["action_selection"],
        "continuation_policy_sampling_seed": resolved_policy_sampling_seed,
        "continuation_checkpoint_sampling_state_sha256": checkpoint_context[
            "sampling_state_sha256"
        ],
        "branch_horizon_ticks": resolved_horizon,
        "exact_replay_repeat_count_per_action": repeat_count,
        "repeat_indices": list(range(repeat_count)),
        "repeat_policy_sampling_seeds": [
            resolved_policy_sampling_seed for _ in range(repeat_count)
        ],
        "repeat_checkpoint_sampling_state_sha256": [
            checkpoint_context["sampling_state_sha256"] for _ in range(repeat_count)
        ],
        "common_checkpoint_across_actions": True,
        "source_checkpoint_reused_by_deepcopy": True,
    }
    continuation_provenance_digest = stable_payload_digest(continuation_provenance)
    branch_identity_digest = stable_payload_digest(
        {
            "protocol_digest": protocol_digest,
            "artifact_digest": resolved_artifact_digest,
            "model_state_sha256": source_model_state_sha256,
            "seed_role": resolved_seed_role,
            "environment_seed": environment_seed,
            "policy_sampling_seed": resolved_policy_sampling_seed,
            "scenario": resolved_scenario,
            "branch_tick": resolved_branch_tick,
            "horizon_ticks": resolved_horizon,
            "exact_replay_repeat_count_per_action": repeat_count,
            "focal_agent_id": resolved_focal_agent_id,
            "source_record_digest": source_record_digest,
            "source_recurrent_state_sha256": checkpoint_context[
                "recurrent_state_sha256"
            ],
            "source_sampling_state_sha256": checkpoint_context["sampling_state_sha256"],
            "source_public_history_prefix_sha256": (public_history_prefix_sha256),
            "continuation_provenance_digest": (continuation_provenance_digest),
        }
    )
    metadata = {
        "seed_role": resolved_seed_role,
        "environment_seed": environment_seed,
        "scenario": resolved_scenario,
        "branch_tick": resolved_branch_tick,
        "horizon_ticks": resolved_horizon,
        "focal_agent_id": resolved_focal_agent_id,
        "gamma": resolved_gamma,
        "source_artifact_digest": resolved_artifact_digest,
        "source_model_state_sha256": source_model_state_sha256,
        "policy_sampling_seed": resolved_policy_sampling_seed,
        "source_action_selection": checkpoint_context["action_selection"],
        "source_decision_index": source_decision_diagnostics.get("decision_index"),
        "source_record_digest": source_record_digest,
        "source_decision_diagnostics_digest": source_diagnostics_digest,
        "source_observation_digest": source_observation_digest,
        "source_action_mask_digest": source_action_mask_digest,
        "source_sampling_state_sha256": checkpoint_context["sampling_state_sha256"],
        "source_public_history_prefix_sha256": public_history_prefix_sha256,
        "branch_identity_digest": branch_identity_digest,
        "branch_protocol_digest": protocol_digest,
        "continuation_provenance": continuation_provenance,
        "continuation_provenance_digest": continuation_provenance_digest,
        "baseline_behavior_digest": baseline["behavior_digest"],
        "baseline_evidence_digest": baseline["evidence_digest"],
        "private_checkpoint_serialized": False,
    }
    row: dict[str, object] = {
        "schema_version": RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION,
        "contract": contract,
        "trainable_public_context": trainable_public_context,
        "optimizer_context": optimizer_context,
        "labels": labels,
        "metadata": metadata,
        "component_digests": {
            "contract": protocol_digest,
            "trainable_public_context": stable_payload_digest(trainable_public_context),
            "optimizer_context": stable_payload_digest(optimizer_context),
            "labels": stable_payload_digest(labels),
            "metadata": stable_payload_digest(metadata),
        },
        "training_ran": False,
        "training_artifact_created": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
    }
    row["exact_digest"] = stable_payload_digest(row)
    validate_recurrent_counterfactual_branch_row(row)
    return row


def build_recurrent_counterfactual_nested_horizon_materialization(
    model: PublicRecurrentActorCritic,
    *,
    artifact_digest: str,
    seed_role: str,
    environment_seed: int,
    scenario: str,
    branch_tick_candidates: Sequence[int],
    horizons: Sequence[int],
    source_policy_sampling_seed: int,
    branch_selection_seed: int,
    branch_tick_stratum_index: int | None = None,
    gamma: float = 0.99,
    continuation_tape_count: int = 1,
    continuation_tape_identity: str | None = None,
    terminal_target_world_tick: int | None = None,
    uncertainty_penalty: float = 0.0,
) -> dict[str, object]:
    return _build_recurrent_counterfactual_nested_horizon_materialization(
        model,
        artifact_digest=artifact_digest,
        seed_role=seed_role,
        environment_seed=environment_seed,
        scenario=scenario,
        branch_tick_candidates=branch_tick_candidates,
        horizons=horizons,
        source_policy_sampling_seed=source_policy_sampling_seed,
        branch_selection_seed=branch_selection_seed,
        branch_tick_stratum_index=branch_tick_stratum_index,
        gamma=gamma,
        continuation_tape_count=continuation_tape_count,
        continuation_tape_identity=continuation_tape_identity,
        terminal_target_world_tick=terminal_target_world_tick,
        uncertainty_penalty=uncertainty_penalty,
        continuation_tape_indices=None,
        defer_multi_tape_assembly=False,
    )


def build_recurrent_counterfactual_nested_horizon_tape_chunk_materialization(
    model: PublicRecurrentActorCritic,
    *,
    artifact_digest: str,
    seed_role: str,
    environment_seed: int,
    scenario: str,
    branch_tick_candidates: Sequence[int],
    horizons: Sequence[int],
    source_policy_sampling_seed: int,
    branch_selection_seed: int,
    continuation_tape_count: int,
    continuation_tape_identity: str,
    continuation_tape_indices: Sequence[int],
    branch_tick_stratum_index: int | None = None,
    gamma: float = 0.99,
    terminal_target_world_tick: int | None = None,
    uncertainty_penalty: float = 0.0,
) -> dict[str, object]:
    """Materialize an ephemeral, ordered subset of continuation tapes.

    Tape chunks are process-transfer work products only. They contain no
    serialized private checkpoint and cannot be consumed as aggregate evidence
    until :func:`merge_recurrent_counterfactual_nested_horizon_tape_chunks`
    proves complete, canonical tape coverage and assembles the normal evidence
    rows.
    """

    return _build_recurrent_counterfactual_nested_horizon_materialization(
        model,
        artifact_digest=artifact_digest,
        seed_role=seed_role,
        environment_seed=environment_seed,
        scenario=scenario,
        branch_tick_candidates=branch_tick_candidates,
        horizons=horizons,
        source_policy_sampling_seed=source_policy_sampling_seed,
        branch_selection_seed=branch_selection_seed,
        branch_tick_stratum_index=branch_tick_stratum_index,
        gamma=gamma,
        continuation_tape_count=continuation_tape_count,
        continuation_tape_identity=continuation_tape_identity,
        terminal_target_world_tick=terminal_target_world_tick,
        uncertainty_penalty=uncertainty_penalty,
        continuation_tape_indices=continuation_tape_indices,
        defer_multi_tape_assembly=True,
    )


def _build_recurrent_counterfactual_nested_horizon_materialization(
    model: PublicRecurrentActorCritic,
    *,
    artifact_digest: str,
    seed_role: str,
    environment_seed: int,
    scenario: str,
    branch_tick_candidates: Sequence[int],
    horizons: Sequence[int],
    source_policy_sampling_seed: int,
    branch_selection_seed: int,
    branch_tick_stratum_index: int | None,
    gamma: float,
    continuation_tape_count: int,
    continuation_tape_identity: str | None,
    terminal_target_world_tick: int | None,
    uncertainty_penalty: float,
    continuation_tape_indices: Sequence[int] | None,
    defer_multi_tape_assembly: bool,
) -> dict[str, object]:
    """Materialize one source checkpoint and derive exact nested horizon rows.

    This is an engineering primitive for post-PPO diagnostics/auxiliary data.
    It trains nothing, serializes no world checkpoint, and changes no runtime
    policy. Every horizon is a proven prefix of the same max-horizon baseline,
    forced-action continuation, and exact replay.
    """

    if not isinstance(model, PublicRecurrentActorCritic):
        raise TypeError("model must be a PublicRecurrentActorCritic")
    resolved_artifact_digest = _nonempty_string(
        artifact_digest,
        field="artifact_digest",
    )
    resolved_seed_role = _validated_seed_role(seed_role, environment_seed)
    resolved_scenario = _validated_scenario(scenario)
    resolved_branch_ticks = _validated_branch_tick_candidates(branch_tick_candidates)
    resolved_horizons = _validated_nested_horizons(horizons)
    resolved_source_sampling_seed = _required_sampling_seed(
        source_policy_sampling_seed,
        field="source_policy_sampling_seed",
    )
    resolved_selection_seed = _required_sampling_seed(
        branch_selection_seed,
        field="branch_selection_seed",
    )
    resolved_stratum_index = (
        None
        if branch_tick_stratum_index is None
        else _nonnegative_int(
            branch_tick_stratum_index,
            field="branch_tick_stratum_index",
        )
    )
    if resolved_stratum_index is not None and resolved_stratum_index >= len(
        resolved_branch_ticks
    ):
        raise RecurrentCounterfactualBranchError(
            "branch_tick_stratum_index must index branch_tick_candidates"
        )
    resolved_tape_count = _positive_int(
        continuation_tape_count,
        field="continuation_tape_count",
    )
    resolved_tape_indices = (
        None
        if continuation_tape_indices is None
        else _validated_continuation_tape_indices(
            continuation_tape_indices,
            tape_count=resolved_tape_count,
        )
    )
    if defer_multi_tape_assembly != (resolved_tape_indices is not None):
        raise RecurrentCounterfactualBranchError(
            "deferred multi-tape assembly requires explicit tape indices"
        )
    resolved_tape_identity = (
        None
        if continuation_tape_identity is None
        else _nonempty_string(
            continuation_tape_identity,
            field="continuation_tape_identity",
        )
    )
    resolved_terminal_target = (
        None
        if terminal_target_world_tick is None
        else _positive_int(
            terminal_target_world_tick,
            field="terminal_target_world_tick",
        )
    )
    resolved_uncertainty_penalty = _finite_float(
        uncertainty_penalty,
        field="uncertainty_penalty",
    )
    if resolved_uncertainty_penalty < 0.0:
        raise RecurrentCounterfactualBranchError(
            "uncertainty_penalty must be non-negative"
        )
    upgraded_evidence_requested = (
        resolved_tape_count != 1
        or resolved_tape_identity is not None
        or resolved_terminal_target is not None
        or resolved_uncertainty_penalty != 0.0
        or resolved_stratum_index is not None
    )
    if upgraded_evidence_requested and resolved_tape_identity is None:
        raise RecurrentCounterfactualBranchError(
            "multi-tape or terminal aggregate evidence requires "
            "continuation_tape_identity"
        )
    if defer_multi_tape_assembly and resolved_tape_identity is None:
        raise RecurrentCounterfactualBranchError(
            "tape chunks require continuation_tape_identity"
        )
    if not upgraded_evidence_requested and resolved_tape_count != 1:
        raise AssertionError("unreachable continuation tape configuration")
    if (
        len(
            {
                environment_seed,
                resolved_source_sampling_seed,
                resolved_selection_seed,
            }
        )
        != 3
    ):
        raise RecurrentCounterfactualBranchError(
            "environment, source-policy, and branch-selection seeds must differ"
        )
    resolved_gamma = _unit_interval(gamma, field="gamma")
    maximum_branch_tick = max(resolved_branch_ticks)
    maximum_horizon = resolved_horizons[-1]
    if resolved_terminal_target is not None:
        overrun_ticks = tuple(
            tick
            for tick in resolved_branch_ticks
            if tick + maximum_horizon > resolved_terminal_target
        )
        if overrun_ticks:
            raise RecurrentCounterfactualBranchError(
                "relative counterfactual horizons must not run beyond the "
                "absolute terminal target; offending branch ticks: "
                f"{list(overrun_ticks)}"
            )
    source_tick_limit = maximum_branch_tick + maximum_horizon
    if resolved_terminal_target is not None:
        source_tick_limit = max(source_tick_limit, resolved_terminal_target)
    source_model_state_sha256 = recurrent_model_state_sha256(model)

    policy = DeterministicPublicRecurrentPolicy(
        model,
        artifact_digest=resolved_artifact_digest,
        sampling_seed=resolved_source_sampling_seed,
        capture_public_history=True,
    )
    source_world = _world_for_scenario(
        scenario=resolved_scenario,
        environment_seed=environment_seed,
        ticks=source_tick_limit,
        policy=policy,
    )
    _configure_manual_world(source_world)
    candidate_set = set(resolved_branch_ticks)
    materialized_candidates: dict[
        int,
        tuple[
            SimulationWorld,
            tuple[dict[str, object], ...],
            tuple[dict[str, object] | None, ...],
        ],
    ] = {}
    source_tick_executions = 0
    for tick in range(maximum_branch_tick + 1):
        if not source_world.alive_agents():
            break
        checkpoint = deepcopy(source_world) if tick in candidate_set else None
        diagnostics_start = len(source_world.policy_decision_diagnostics_records)
        source_world.tick = tick
        source_world._run_tick()
        source_tick_executions += 1
        if checkpoint is None:
            continue
        tick_records = tuple(
            dict(record) for record in source_world.tick_trajectory_records
        )
        tick_diagnostics = _copy_aligned_source_decision_diagnostics(
            tick_records,
            source_world.policy_decision_diagnostics_records[diagnostics_start:],
        )
        eligible = _eligible_focal_source_records(
            tick_records,
            tick_diagnostics,
        )
        if eligible:
            materialized_candidates[tick] = (
                checkpoint,
                tick_records,
                tick_diagnostics,
            )

    eligible_branch_ticks = tuple(
        tick for tick in resolved_branch_ticks if tick in materialized_candidates
    )
    if not eligible_branch_ticks:
        raise RecurrentCounterfactualBranchError(
            "no requested branch tick contained an eligible learner decision"
        )
    selection_rng = random.Random(resolved_selection_seed)
    stratum_traversal: tuple[int, ...] | None = None
    if resolved_stratum_index is not None:
        stratum_traversal = tuple(
            resolved_branch_ticks[
                (resolved_stratum_index + offset) % len(resolved_branch_ticks)
            ]
            for offset in range(len(resolved_branch_ticks))
        )
        selected_tick = next(
            tick for tick in stratum_traversal if tick in materialized_candidates
        )
        selected_eligible_tick_index = eligible_branch_ticks.index(selected_tick)
    else:
        selected_eligible_tick_index = (
            selection_rng.randrange(len(eligible_branch_ticks))
            if upgraded_evidence_requested
            else 0
        )
        selected_tick = eligible_branch_ticks[selected_eligible_tick_index]
    fallback_index = resolved_branch_ticks.index(selected_tick)
    if (
        resolved_terminal_target is not None
        and resolved_terminal_target <= selected_tick
    ):
        raise RecurrentCounterfactualBranchError(
            "terminal_target_world_tick must be after the selected branch tick"
        )
    checkpoint_world, source_records, source_diagnostics = materialized_candidates[
        selected_tick
    ]
    eligible = _eligible_focal_source_records(source_records, source_diagnostics)
    candidate_payload = [
        _focal_candidate_payload(record, diagnostics)
        for record, diagnostics in eligible
    ]
    candidate_list_digest = stable_payload_digest(candidate_payload)
    selected_candidate_index = selection_rng.randrange(len(eligible))
    source_record, source_decision_diagnostics = eligible[selected_candidate_index]
    resolved_focal_agent_id = _positive_int(
        source_record.get("agent_id"),
        field="source focal agent_id",
    )
    source_action = _stable_action(
        source_record.get("requested_action"),
        field="source requested_action",
    )
    source_mask = _complete_action_mask(source_record.get("action_mask"))
    valid_actions = tuple(
        action for action in ACTION_NAMES if source_mask[action] is True
    )
    if len(valid_actions) < 2:
        raise RecurrentCounterfactualBranchError(
            "selected nested branch state must expose at least two valid actions"
        )
    if source_action not in valid_actions:
        raise RecurrentCounterfactualBranchError(
            "source learner requested an action outside its tick-start mask"
        )
    source_observation_digest = _nonempty_string(
        source_record.get("observation_digest"),
        field="source observation_digest",
    )
    source_action_mask_digest = stable_payload_digest(source_mask)
    checkpoint_policy = checkpoint_world.policy
    if not isinstance(checkpoint_policy, DeterministicPublicRecurrentPolicy):
        raise RecurrentCounterfactualBranchError(
            "materialized checkpoint does not own the recurrent policy"
        )
    checkpoint_context = checkpoint_policy.diagnostics_checkpoint_state(
        agent_id=resolved_focal_agent_id,
    )
    if checkpoint_context.get("artifact_digest") != resolved_artifact_digest:
        raise RecurrentCounterfactualBranchError(
            "checkpoint source artifact digest does not match the requested artifact"
        )
    if checkpoint_context.get("model_state_sha256") != source_model_state_sha256:
        raise RecurrentCounterfactualBranchError(
            "checkpoint source model digest does not match the supplied model"
        )
    public_history_prefix = deepcopy(checkpoint_context.get("public_history_prefix"))
    if not isinstance(public_history_prefix, Mapping):
        raise RecurrentCounterfactualBranchError(
            "checkpoint did not provide a public recurrent history prefix"
        )
    try:
        validate_public_recurrent_history_prefix(public_history_prefix)
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "checkpoint public recurrent history prefix is invalid"
        ) from error
    public_history_prefix_sha256 = stable_payload_digest(public_history_prefix)
    if (
        checkpoint_context.get("public_history_prefix_sha256")
        != public_history_prefix_sha256
    ):
        raise RecurrentCounterfactualBranchError(
            "checkpoint public recurrent history prefix digest mismatch"
        )
    source_behavior_distribution = deepcopy(
        source_decision_diagnostics.get("learned_masked_distribution")
    )
    if not isinstance(source_behavior_distribution, Mapping):
        raise RecurrentCounterfactualBranchError(
            "source decision lacks learned masked distribution diagnostics"
        )
    try:
        validate_public_recurrent_distribution_diagnostics(
            source_behavior_distribution,
            action_mask=source_mask,
        )
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "source learned masked distribution diagnostics are invalid"
        ) from error

    branch_id = (
        f"recurrent-counterfactual-{resolved_seed_role}-{resolved_scenario}-"
        f"seed-{environment_seed}-tick-{selected_tick}-"
        f"agent-{resolved_focal_agent_id}"
    )
    baseline_max = _execute_continuation(
        checkpoint_world,
        branch_id=branch_id,
        branch_tick=selected_tick,
        horizon_ticks=maximum_horizon,
        focal_agent_id=resolved_focal_agent_id,
        source_action=source_action,
        expected_observation_digest=source_observation_digest,
        expected_action_mask_digest=source_action_mask_digest,
        forced_action=None,
        gamma=resolved_gamma,
        prefix_horizons=resolved_horizons,
    )
    baseline_by_horizon = _nested_runs_by_horizon(
        baseline_max,
        horizons=resolved_horizons,
        field="baseline",
    )
    action_runs_by_horizon: dict[int, list[dict[str, object]]] = {
        horizon: [] for horizon in resolved_horizons
    }
    replay_runs_by_action_horizon: dict[
        str,
        dict[int, dict[str, object]],
    ] = {}
    action_max_runs: list[dict[str, object]] = []
    replay_max_runs: list[dict[str, object]] = []
    for action in valid_actions:
        run_max = _execute_continuation(
            checkpoint_world,
            branch_id=branch_id,
            branch_tick=selected_tick,
            horizon_ticks=maximum_horizon,
            focal_agent_id=resolved_focal_agent_id,
            source_action=source_action,
            expected_observation_digest=source_observation_digest,
            expected_action_mask_digest=source_action_mask_digest,
            forced_action=action,
            gamma=resolved_gamma,
            prefix_horizons=resolved_horizons,
        )
        replay_max = _execute_continuation(
            checkpoint_world,
            branch_id=branch_id,
            branch_tick=selected_tick,
            horizon_ticks=maximum_horizon,
            focal_agent_id=resolved_focal_agent_id,
            source_action=source_action,
            expected_observation_digest=source_observation_digest,
            expected_action_mask_digest=source_action_mask_digest,
            forced_action=action,
            gamma=resolved_gamma,
            prefix_horizons=resolved_horizons,
        )
        runs = _nested_runs_by_horizon(
            run_max,
            horizons=resolved_horizons,
            field=f"action {action}",
        )
        replays = _nested_runs_by_horizon(
            replay_max,
            horizons=resolved_horizons,
            field=f"action replay {action}",
        )
        replay_runs_by_action_horizon[action] = replays
        for horizon in resolved_horizons:
            run = runs[horizon]
            replay = replays[horizon]
            replay_verified = (
                replay["evidence_digest"] == run["evidence_digest"]
                and replay["behavior_digest"] == run["behavior_digest"]
            )
            if not replay_verified:
                raise RecurrentCounterfactualBranchError(
                    f"nested counterfactual replay mismatch for {action!r} "
                    f"at horizon {horizon}"
                )
            run["replay_verified"] = True
            run["replay_evidence_digest"] = replay["evidence_digest"]
            action_runs_by_horizon[horizon].append(run)
        action_max_runs.append(run_max)
        replay_max_runs.append(replay_max)

    rows: list[dict[str, object]] = []
    for horizon in resolved_horizons:
        baseline = baseline_by_horizon[horizon]
        action_runs = action_runs_by_horizon[horizon]
        natural = next(
            run for run in action_runs if run["forced_action"] == source_action
        )
        if natural["behavior_digest"] != baseline["behavior_digest"]:
            raise RecurrentCounterfactualBranchError(
                "nested source-action branch did not reproduce its baseline"
            )
        rows.append(
            _assemble_nested_branch_row(
                artifact_digest=resolved_artifact_digest,
                seed_role=resolved_seed_role,
                environment_seed=environment_seed,
                scenario=resolved_scenario,
                branch_tick=selected_tick,
                horizon_ticks=horizon,
                gamma=resolved_gamma,
                policy_sampling_seed=resolved_source_sampling_seed,
                focal_agent_id=resolved_focal_agent_id,
                source_action=source_action,
                source_mask=source_mask,
                valid_actions=valid_actions,
                source_record=source_record,
                source_decision_diagnostics=source_decision_diagnostics,
                source_behavior_distribution=source_behavior_distribution,
                checkpoint_context=checkpoint_context,
                source_model_state_sha256=source_model_state_sha256,
                public_history_prefix=public_history_prefix,
                public_history_prefix_sha256=public_history_prefix_sha256,
                source_observation_digest=source_observation_digest,
                source_action_mask_digest=source_action_mask_digest,
                baseline=baseline,
                action_runs=action_runs,
            )
        )

    prefix_proof = _nested_prefix_proof(
        horizons=resolved_horizons,
        baseline_by_horizon=baseline_by_horizon,
        action_runs_by_horizon=action_runs_by_horizon,
        replay_runs_by_action_horizon=replay_runs_by_action_horizon,
        valid_actions=valid_actions,
    )
    actual_continuation_tick_count = sum(
        int(run["executed_tick_count"])
        for run in (baseline_max, *action_max_runs, *replay_max_runs)
    )
    max_horizon_continuation_count = 1 + (2 * len(valid_actions))
    upgraded_payload: dict[str, object] = {}
    if resolved_tape_identity is not None:
        if defer_multi_tape_assembly:
            if resolved_tape_indices is None:
                raise AssertionError("deferred tape indices were not resolved")
            tape_chunk, boundary_bound_rows = _materialize_multi_tape_evidence_chunk(
                checkpoint_world,
                legacy_rows=rows,
                tape_count=resolved_tape_count,
                tape_indices=resolved_tape_indices,
                tape_identity=resolved_tape_identity,
                terminal_target_world_tick=resolved_terminal_target,
                branch_id=branch_id,
                branch_tick=selected_tick,
                horizons=resolved_horizons,
                focal_agent_id=resolved_focal_agent_id,
                source_action=source_action,
                valid_actions=valid_actions,
                expected_observation_digest=source_observation_digest,
                expected_action_mask_digest=source_action_mask_digest,
                expected_source_decision_projection_sha256=(
                    _source_decision_prefix_projection_sha256(
                        source_decision_diagnostics
                    )
                ),
                gamma=resolved_gamma,
                excluded_seeds=(
                    environment_seed,
                    resolved_source_sampling_seed,
                    resolved_selection_seed,
                ),
            )
            rows = list(boundary_bound_rows)
            upgraded_payload = {"multi_tape_chunk": tape_chunk}
        else:
            (
                aggregate_rows,
                terminal_target,
                multi_tape_compute,
                boundary_bound_rows,
            ) = _materialize_multi_tape_aggregate_evidence(
                checkpoint_world,
                legacy_rows=rows,
                tape_count=resolved_tape_count,
                tape_identity=resolved_tape_identity,
                terminal_target_world_tick=resolved_terminal_target,
                uncertainty_penalty=resolved_uncertainty_penalty,
                branch_id=branch_id,
                branch_tick=selected_tick,
                horizons=resolved_horizons,
                focal_agent_id=resolved_focal_agent_id,
                source_action=source_action,
                valid_actions=valid_actions,
                expected_observation_digest=source_observation_digest,
                expected_action_mask_digest=source_action_mask_digest,
                expected_source_decision_projection_sha256=(
                    _source_decision_prefix_projection_sha256(
                        source_decision_diagnostics
                    )
                ),
                gamma=resolved_gamma,
                excluded_seeds=(
                    environment_seed,
                    resolved_source_sampling_seed,
                    resolved_selection_seed,
                ),
            )
            rows = list(boundary_bound_rows)
            upgraded_payload = {
                "aggregate_rows": aggregate_rows,
                "terminal_target": terminal_target,
                "multi_tape_compute": multi_tape_compute,
            }
    selection_payload = {
        "requested_branch_ticks": list(resolved_branch_ticks),
        "selected_branch_tick": selected_tick,
        "fallback_index": fallback_index,
        "eligible_candidate_count": len(eligible),
        "candidate_list_digest": candidate_list_digest,
        "selected_candidate_index": selected_candidate_index,
        "selected_focal_agent_id": resolved_focal_agent_id,
        "selection_policy": (
            "deterministic_stratified_tick_rotation_with_eligible_fallback_then_"
            "uniform_current_tick_learner_decision"
            if resolved_stratum_index is not None
            else (
                "stratified_uniform_seeded_tick_then_uniform_current_tick_"
                "learner_decision"
                if upgraded_evidence_requested
                else "uniform_seeded_over_current_tick_multi_action_learner_decisions"
            )
        ),
        "outcome_or_future_data_used": False,
    }
    if upgraded_evidence_requested:
        selection_payload.update(
            {
                "eligible_branch_ticks": list(eligible_branch_ticks),
                "eligible_branch_tick_count": len(eligible_branch_ticks),
                "selected_eligible_tick_index": selected_eligible_tick_index,
            }
        )
    if resolved_stratum_index is not None:
        selection_payload.update(
            {
                "requested_branch_tick_stratum_index": resolved_stratum_index,
                "branch_tick_strata_traversal": list(stratum_traversal or ()),
                "selected_branch_tick_stratum_index": resolved_branch_ticks.index(
                    selected_tick
                ),
            }
        )
    return {
        "rows": tuple(rows),
        "selection": selection_payload,
        "prefix_proof": prefix_proof,
        "compute": {
            "source_tick_executions": source_tick_executions,
            "selected_continuation_checkpoint_count": 1,
            "valid_action_count": len(valid_actions),
            "horizon_count": len(resolved_horizons),
            "action_horizon_target_count": (
                len(valid_actions) * len(resolved_horizons)
            ),
            "max_horizon_continuation_count": (max_horizon_continuation_count),
            "exact_replay_continuation_count": len(valid_actions),
            "maximum_horizon_ticks": maximum_horizon,
            "actual_continuation_tick_count": actual_continuation_tick_count,
            "maximum_continuation_tick_budget": (
                max_horizon_continuation_count * maximum_horizon
            ),
            "independent_horizon_tick_budget": (
                max_horizon_continuation_count * sum(resolved_horizons)
            ),
            "nested_tick_budget_saved": (
                max_horizon_continuation_count
                * (sum(resolved_horizons) - maximum_horizon)
            ),
        },
        "source_model_state_sha256": source_model_state_sha256,
        "source_artifact_digest": resolved_artifact_digest,
        "environment_seed": environment_seed,
        "source_policy_sampling_seed": resolved_source_sampling_seed,
        "branch_selection_seed": resolved_selection_seed,
        "horizons": resolved_horizons,
        "valid_actions": valid_actions,
        "private_checkpoint_serialized": False,
        "training_ran": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        **upgraded_payload,
    }


def merge_recurrent_counterfactual_nested_horizon_tape_chunks(
    materializations: Sequence[Mapping[str, object]],
    *,
    continuation_tape_count: int,
    continuation_tape_identity: str,
    terminal_target_world_tick: int | None,
    uncertainty_penalty: float,
) -> dict[str, object]:
    """Merge ephemeral tape chunks into the canonical scientific payload."""

    if not materializations:
        raise RecurrentCounterfactualBranchError(
            "multi-tape merge requires at least one materialization"
        )
    first = deepcopy(dict(materializations[0]))
    raw_first_chunk = first.pop("multi_tape_chunk", None)
    if raw_first_chunk is None:
        raise RecurrentCounterfactualBranchError(
            "multi-tape materialization is missing its tape chunk"
        )
    chunks: list[Mapping[str, object]] = [
        _mapping(raw_first_chunk, field="multi-tape chunk")
    ]
    for raw_materialization in materializations[1:]:
        candidate = deepcopy(dict(raw_materialization))
        raw_chunk = candidate.pop("multi_tape_chunk", None)
        if raw_chunk is None:
            raise RecurrentCounterfactualBranchError(
                "multi-tape materialization is missing its tape chunk"
            )
        if candidate != first:
            raise RecurrentCounterfactualBranchError(
                "multi-tape chunks did not reproduce one exact source materialization"
            )
        chunks.append(_mapping(raw_chunk, field="multi-tape chunk"))
    if any(
        field in first
        for field in ("aggregate_rows", "terminal_target", "multi_tape_compute")
    ):
        raise RecurrentCounterfactualBranchError(
            "deferred tape chunks cannot contain assembled aggregate evidence"
        )
    rows = first.get("rows")
    selection = _mapping(first.get("selection"), field="chunk selection")
    horizons = first.get("horizons")
    valid_actions = first.get("valid_actions")
    if (
        not isinstance(rows, tuple)
        or not isinstance(horizons, tuple)
        or not isinstance(valid_actions, tuple)
    ):
        raise RecurrentCounterfactualBranchError(
            "tape chunk source tuple fields drifted"
        )
    branch_tick = _nonnegative_int(
        selection.get("selected_branch_tick"),
        field="selected branch tick",
    )
    excluded_seeds = (
        _required_sampling_seed(
            first.get("environment_seed"),
            field="source environment seed",
        ),
        _required_sampling_seed(
            first.get("source_policy_sampling_seed"),
            field="source policy sampling seed",
        ),
        _required_sampling_seed(
            first.get("branch_selection_seed"),
            field="branch selection seed",
        ),
    )
    aggregate_rows, terminal_target, multi_tape_compute = (
        _assemble_multi_tape_aggregate_evidence(
            source_rows=rows,
            chunks=chunks,
            tape_count=continuation_tape_count,
            tape_identity=continuation_tape_identity,
            terminal_target_world_tick=terminal_target_world_tick,
            uncertainty_penalty=uncertainty_penalty,
            branch_tick=branch_tick,
            horizons=horizons,
            valid_actions=valid_actions,
            excluded_seeds=excluded_seeds,
        )
    )
    first.update(
        {
            "aggregate_rows": aggregate_rows,
            "terminal_target": terminal_target,
            "multi_tape_compute": multi_tape_compute,
        }
    )
    return first


def _materialize_multi_tape_evidence_chunk(
    checkpoint_world: SimulationWorld,
    *,
    legacy_rows: Sequence[Mapping[str, object]],
    tape_count: int,
    tape_indices: Sequence[int],
    tape_identity: str,
    terminal_target_world_tick: int | None,
    branch_id: str,
    branch_tick: int,
    horizons: Sequence[int],
    focal_agent_id: int,
    source_action: str,
    valid_actions: Sequence[str],
    expected_observation_digest: str,
    expected_action_mask_digest: str,
    expected_source_decision_projection_sha256: str,
    gamma: float,
    excluded_seeds: Sequence[int],
) -> tuple[
    dict[str, object],
    tuple[dict[str, object], ...],
]:
    """Execute an ordered subset of independent retaped checkpoints."""

    if not legacy_rows:
        raise RecurrentCounterfactualBranchError(
            "multi-tape evidence requires legacy source rows"
        )
    resolved_tape_indices = _validated_continuation_tape_indices(
        tape_indices,
        tape_count=tape_count,
    )
    terminal_horizon = (
        None
        if terminal_target_world_tick is None
        else terminal_target_world_tick - branch_tick
    )
    if terminal_horizon is not None and terminal_horizon <= 0:
        raise RecurrentCounterfactualBranchError(
            "absolute terminal target must follow the branch state"
        )
    execution_horizons = tuple(
        sorted(
            {
                *(_positive_int(item, field="aggregate horizon") for item in horizons),
                *(() if terminal_horizon is None else (terminal_horizon,)),
            }
        )
    )
    maximum_horizon = execution_horizons[-1]
    tape_entries_by_horizon: dict[int, list[dict[str, object]]] = {
        horizon: [] for horizon in execution_horizons
    }
    actual_continuation_tick_count = 0
    all_environment_seeds: list[int] = []
    all_policy_seeds: list[int] = []
    excluded_seed_set = set(excluded_seeds)

    for tape_index in resolved_tape_indices:
        environment_identity = f"{tape_identity}:environment:{tape_index}"
        policy_identity = f"{tape_identity}:policy:{tape_index}"
        environment_sampling_seed = derive_recurrent_counterfactual_tape_seed(
            namespace=(
                RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE
            ),
            identity=environment_identity,
        )
        policy_sampling_seed = derive_recurrent_counterfactual_tape_seed(
            namespace=RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE,
            identity=policy_identity,
        )
        if (
            environment_sampling_seed == policy_sampling_seed
            or environment_sampling_seed in excluded_seed_set
            or policy_sampling_seed in excluded_seed_set
            or environment_sampling_seed in all_environment_seeds
            or policy_sampling_seed in all_policy_seeds
            or environment_sampling_seed in all_policy_seeds
            or policy_sampling_seed in all_environment_seeds
        ):
            raise RecurrentCounterfactualBranchError(
                "continuation tape seeds must be globally unique and source-disjoint"
            )
        all_environment_seeds.append(environment_sampling_seed)
        all_policy_seeds.append(policy_sampling_seed)
        tape_branch_id = f"{branch_id}:tape:{tape_index}"
        baseline_max = _execute_continuation(
            checkpoint_world,
            branch_id=tape_branch_id,
            branch_tick=branch_tick,
            horizon_ticks=maximum_horizon,
            focal_agent_id=focal_agent_id,
            source_action=source_action,
            expected_observation_digest=expected_observation_digest,
            expected_action_mask_digest=expected_action_mask_digest,
            expected_source_decision_projection_sha256=(
                expected_source_decision_projection_sha256
            ),
            forced_action=None,
            gamma=gamma,
            prefix_horizons=execution_horizons,
            boundary_environment_sampling_seed=environment_sampling_seed,
            boundary_policy_sampling_seed=policy_sampling_seed,
        )
        baseline_replay_max = _execute_continuation(
            checkpoint_world,
            branch_id=tape_branch_id,
            branch_tick=branch_tick,
            horizon_ticks=maximum_horizon,
            focal_agent_id=focal_agent_id,
            source_action=source_action,
            expected_observation_digest=expected_observation_digest,
            expected_action_mask_digest=expected_action_mask_digest,
            expected_source_decision_projection_sha256=(
                expected_source_decision_projection_sha256
            ),
            forced_action=None,
            gamma=gamma,
            prefix_horizons=execution_horizons,
            boundary_environment_sampling_seed=environment_sampling_seed,
            boundary_policy_sampling_seed=policy_sampling_seed,
        )
        tape_natural_action = _stable_action(
            baseline_max.get("natural_requested_action"),
            field="multi-tape baseline natural action",
        )
        baseline_by_horizon = _nested_runs_by_horizon(
            baseline_max,
            horizons=execution_horizons,
            field=f"tape {tape_index} baseline",
        )
        baseline_replay_by_horizon = _nested_runs_by_horizon(
            baseline_replay_max,
            horizons=execution_horizons,
            field=f"tape {tape_index} baseline replay",
        )
        action_runs_by_horizon: dict[int, list[dict[str, object]]] = {
            horizon: [] for horizon in execution_horizons
        }
        action_replays_by_horizon: dict[int, list[dict[str, object]]] = {
            horizon: [] for horizon in execution_horizons
        }
        action_max_runs: list[dict[str, object]] = []
        action_replay_max_runs: list[dict[str, object]] = []
        for action in valid_actions:
            action_max = _execute_continuation(
                checkpoint_world,
                branch_id=tape_branch_id,
                branch_tick=branch_tick,
                horizon_ticks=maximum_horizon,
                focal_agent_id=focal_agent_id,
                source_action=tape_natural_action,
                expected_observation_digest=expected_observation_digest,
                expected_action_mask_digest=expected_action_mask_digest,
                expected_source_decision_projection_sha256=(
                    expected_source_decision_projection_sha256
                ),
                forced_action=action,
                gamma=gamma,
                prefix_horizons=execution_horizons,
                boundary_environment_sampling_seed=environment_sampling_seed,
                boundary_policy_sampling_seed=policy_sampling_seed,
            )
            action_replay_max = _execute_continuation(
                checkpoint_world,
                branch_id=tape_branch_id,
                branch_tick=branch_tick,
                horizon_ticks=maximum_horizon,
                focal_agent_id=focal_agent_id,
                source_action=tape_natural_action,
                expected_observation_digest=expected_observation_digest,
                expected_action_mask_digest=expected_action_mask_digest,
                expected_source_decision_projection_sha256=(
                    expected_source_decision_projection_sha256
                ),
                forced_action=action,
                gamma=gamma,
                prefix_horizons=execution_horizons,
                boundary_environment_sampling_seed=environment_sampling_seed,
                boundary_policy_sampling_seed=policy_sampling_seed,
            )
            runs = _nested_runs_by_horizon(
                action_max,
                horizons=execution_horizons,
                field=f"tape {tape_index} action {action}",
            )
            replays = _nested_runs_by_horizon(
                action_replay_max,
                horizons=execution_horizons,
                field=f"tape {tape_index} action replay {action}",
            )
            for horizon in execution_horizons:
                action_runs_by_horizon[horizon].append(runs[horizon])
                action_replays_by_horizon[horizon].append(replays[horizon])
            action_max_runs.append(action_max)
            action_replay_max_runs.append(action_replay_max)

        for horizon in execution_horizons:
            baseline = baseline_by_horizon[horizon]
            baseline_replay = baseline_replay_by_horizon[horizon]
            if baseline.get("behavior_digest") != baseline_replay.get(
                "behavior_digest"
            ) or baseline.get("evidence_digest") != baseline_replay.get(
                "evidence_digest"
            ) or _decision_boundary_provenance(
                baseline
            ) != _decision_boundary_provenance(baseline_replay):
                raise RecurrentCounterfactualBranchError(
                    f"multi-tape baseline replay mismatch at tape {tape_index} "
                    f"horizon {horizon}"
                )
            action_entries: list[dict[str, object]] = []
            for action, run, replay in zip(
                valid_actions,
                action_runs_by_horizon[horizon],
                action_replays_by_horizon[horizon],
                strict=True,
            ):
                if run.get("behavior_digest") != replay.get(
                    "behavior_digest"
                ) or run.get("evidence_digest") != replay.get(
                    "evidence_digest"
                ) or _decision_boundary_provenance(
                    run
                ) != _decision_boundary_provenance(replay):
                    raise RecurrentCounterfactualBranchError(
                        f"multi-tape action replay mismatch for {action!r} "
                        f"at tape {tape_index} horizon {horizon}"
                    )
                if (
                    action == tape_natural_action
                    and run.get("behavior_digest") != baseline.get("behavior_digest")
                ):
                    raise RecurrentCounterfactualBranchError(
                        "multi-tape source-action negative control did not "
                        "reproduce baseline behavior"
                    )
                action_entries.append(
                    _compact_multi_tape_action_evidence(
                        run,
                        replay=replay,
                        baseline=baseline,
                    )
                )
            baseline_entry = _compact_multi_tape_baseline_evidence(
                baseline,
                replay=baseline_replay,
            )
            boundary_provenance = _decision_boundary_provenance(baseline)
            for run in (baseline_entry, *action_entries):
                if any(
                    run[field] != boundary_provenance[field]
                    for field in _DECISION_BOUNDARY_PROVENANCE_DIGEST_FIELDS
                ):
                    raise RecurrentCounterfactualBranchError(
                        "paired actions within a tape did not share boundary RNG state"
                    )
            tape_entries_by_horizon[horizon].append(
                {
                    "tape_index": tape_index,
                    "environment_sampling_identity": environment_identity,
                    "environment_sampling_seed": environment_sampling_seed,
                    "policy_sampling_identity": policy_identity,
                    "policy_sampling_seed": policy_sampling_seed,
                    **boundary_provenance,
                    "baseline": baseline_entry,
                    "action_outcomes": action_entries,
                }
            )
        actual_continuation_tick_count += sum(
            int(run["executed_tick_count"])
            for run in (
                baseline_max,
                baseline_replay_max,
                *action_max_runs,
                *action_replay_max_runs,
            )
        )

    first_tape = tape_entries_by_horizon[execution_horizons[0]][0]
    source_boundary = _decision_boundary_provenance(first_tape)
    source_pre_environment_digest = source_boundary[
        "pre_boundary_environment_rng_state_sha256"
    ]
    source_pre_policy_digest = source_boundary[
        "pre_boundary_policy_sampling_state_sha256"
    ]
    for horizon_entries in tape_entries_by_horizon.values():
        for tape in horizon_entries:
            boundary = _decision_boundary_provenance(tape)
            if (
                boundary["pre_boundary_environment_rng_state_sha256"]
                != source_pre_environment_digest
                or boundary["pre_boundary_policy_sampling_state_sha256"]
                != source_pre_policy_digest
            ):
                raise RecurrentCounterfactualBranchError(
                    "multi-tape generation did not preserve one source boundary"
                )
    boundary_bound_rows = _bind_branch_rows_to_source_boundary(
        legacy_rows,
        pre_boundary_environment_rng_state_sha256=source_pre_environment_digest,
        pre_boundary_policy_sampling_state_sha256=source_pre_policy_digest,
    )
    chunk_without_digest: dict[str, object] = {
        "schema_version": RECURRENT_COUNTERFACTUAL_TAPE_CHUNK_SCHEMA_VERSION,
        "tape_identity": tape_identity,
        "continuation_tape_count": tape_count,
        "tape_indices": resolved_tape_indices,
        "execution_horizons": execution_horizons,
        "horizon_tape_provenance": tuple(
            {
                "horizon_ticks": horizon,
                "tape_provenance": tuple(tape_entries_by_horizon[horizon]),
            }
            for horizon in execution_horizons
        ),
        "actual_continuation_tick_count": actual_continuation_tick_count,
        "maximum_horizon_ticks": maximum_horizon,
        "source_pre_boundary_environment_rng_state_sha256": (
            source_pre_environment_digest
        ),
        "source_pre_boundary_policy_sampling_state_sha256": (
            source_pre_policy_digest
        ),
        "private_checkpoint_serialized": False,
    }
    chunk = {
        **chunk_without_digest,
        "exact_digest": stable_payload_digest(chunk_without_digest),
    }
    _validate_multi_tape_evidence_chunk(chunk)
    return chunk, boundary_bound_rows


def _materialize_multi_tape_aggregate_evidence(
    checkpoint_world: SimulationWorld,
    *,
    legacy_rows: Sequence[Mapping[str, object]],
    tape_count: int,
    tape_identity: str,
    terminal_target_world_tick: int | None,
    uncertainty_penalty: float,
    branch_id: str,
    branch_tick: int,
    horizons: Sequence[int],
    focal_agent_id: int,
    source_action: str,
    valid_actions: Sequence[str],
    expected_observation_digest: str,
    expected_action_mask_digest: str,
    expected_source_decision_projection_sha256: str,
    gamma: float,
    excluded_seeds: Sequence[int],
) -> tuple[
    tuple[dict[str, object], ...],
    dict[str, object] | None,
    dict[str, int],
    tuple[dict[str, object], ...],
]:
    chunk, boundary_bound_rows = _materialize_multi_tape_evidence_chunk(
        checkpoint_world,
        legacy_rows=legacy_rows,
        tape_count=tape_count,
        tape_indices=tuple(range(tape_count)),
        tape_identity=tape_identity,
        terminal_target_world_tick=terminal_target_world_tick,
        branch_id=branch_id,
        branch_tick=branch_tick,
        horizons=horizons,
        focal_agent_id=focal_agent_id,
        source_action=source_action,
        valid_actions=valid_actions,
        expected_observation_digest=expected_observation_digest,
        expected_action_mask_digest=expected_action_mask_digest,
        expected_source_decision_projection_sha256=(
            expected_source_decision_projection_sha256
        ),
        gamma=gamma,
        excluded_seeds=excluded_seeds,
    )
    aggregate_rows, terminal_target, multi_tape_compute = (
        _assemble_multi_tape_aggregate_evidence(
            source_rows=boundary_bound_rows,
            chunks=(chunk,),
            tape_count=tape_count,
            tape_identity=tape_identity,
            terminal_target_world_tick=terminal_target_world_tick,
            uncertainty_penalty=uncertainty_penalty,
            branch_tick=branch_tick,
            horizons=horizons,
            valid_actions=valid_actions,
            excluded_seeds=excluded_seeds,
        )
    )
    return (
        aggregate_rows,
        terminal_target,
        multi_tape_compute,
        boundary_bound_rows,
    )


def _assemble_multi_tape_aggregate_evidence(
    *,
    source_rows: Sequence[Mapping[str, object]],
    chunks: Sequence[Mapping[str, object]],
    tape_count: int,
    tape_identity: str,
    terminal_target_world_tick: int | None,
    uncertainty_penalty: float,
    branch_tick: int,
    horizons: Sequence[int],
    valid_actions: Sequence[str],
    excluded_seeds: Sequence[int],
) -> tuple[
    tuple[dict[str, object], ...],
    dict[str, object] | None,
    dict[str, int],
]:
    if not source_rows:
        raise RecurrentCounterfactualBranchError(
            "multi-tape assembly requires source rows"
        )
    resolved_tape_count = _positive_int(tape_count, field="continuation_tape_count")
    resolved_tape_identity = _nonempty_string(
        tape_identity,
        field="continuation_tape_identity",
    )
    resolved_horizons = _validated_nested_horizons(horizons)
    resolved_terminal_target = (
        None
        if terminal_target_world_tick is None
        else _positive_int(
            terminal_target_world_tick,
            field="terminal_target_world_tick",
        )
    )
    terminal_horizon = (
        None
        if resolved_terminal_target is None
        else resolved_terminal_target - branch_tick
    )
    if terminal_horizon is not None and terminal_horizon <= 0:
        raise RecurrentCounterfactualBranchError(
            "absolute terminal target must follow the branch state"
        )
    execution_horizons = tuple(
        sorted(
            {
                *resolved_horizons,
                *(() if terminal_horizon is None else (terminal_horizon,)),
            }
        )
    )
    maximum_horizon = execution_horizons[-1]
    tape_entries_by_horizon: dict[int, list[dict[str, object]]] = {
        horizon: [] for horizon in execution_horizons
    }
    observed_indices: list[int] = []
    actual_continuation_tick_count = 0
    source_pre_environment_digest: str | None = None
    source_pre_policy_digest: str | None = None
    for raw_chunk in chunks:
        chunk = _validate_multi_tape_evidence_chunk(raw_chunk)
        if (
            chunk["tape_identity"] != resolved_tape_identity
            or chunk["continuation_tape_count"] != resolved_tape_count
            or tuple(chunk["execution_horizons"]) != execution_horizons
            or chunk["maximum_horizon_ticks"] != maximum_horizon
        ):
            raise RecurrentCounterfactualBranchError(
                "multi-tape chunk assembly contract drifted"
            )
        chunk_pre_environment = chunk[
            "source_pre_boundary_environment_rng_state_sha256"
        ]
        chunk_pre_policy = chunk[
            "source_pre_boundary_policy_sampling_state_sha256"
        ]
        if source_pre_environment_digest is None:
            source_pre_environment_digest = str(chunk_pre_environment)
            source_pre_policy_digest = str(chunk_pre_policy)
        elif (
            chunk_pre_environment != source_pre_environment_digest
            or chunk_pre_policy != source_pre_policy_digest
        ):
            raise RecurrentCounterfactualBranchError(
                "multi-tape chunks do not share one source boundary"
            )
        observed_indices.extend(int(index) for index in chunk["tape_indices"])
        actual_continuation_tick_count += int(
            chunk["actual_continuation_tick_count"]
        )
        for horizon_payload in chunk["horizon_tape_provenance"]:
            horizon = int(horizon_payload["horizon_ticks"])
            tape_entries_by_horizon[horizon].extend(
                deepcopy(list(horizon_payload["tape_provenance"]))
            )
    if tuple(sorted(observed_indices)) != tuple(range(resolved_tape_count)):
        raise RecurrentCounterfactualBranchError(
            "multi-tape chunks must cover every tape index exactly once"
        )
    for horizon in execution_horizons:
        tape_entries_by_horizon[horizon].sort(
            key=lambda entry: int(entry["tape_index"])
        )
        if tuple(
            int(entry["tape_index"])
            for entry in tape_entries_by_horizon[horizon]
        ) != tuple(range(resolved_tape_count)):
            raise RecurrentCounterfactualBranchError(
                "multi-tape chunk horizon coverage drifted"
            )
    first_source_row = source_rows[0]
    aggregate_rows = tuple(
        _assemble_multi_tape_aggregate_row(
            source_row=first_source_row,
            target_kind="relative_horizon",
            branch_tick=branch_tick,
            horizon_ticks=horizon,
            terminal_target_world_tick=resolved_terminal_target,
            tape_identity=resolved_tape_identity,
            tape_provenance=tape_entries_by_horizon[horizon],
            uncertainty_penalty=uncertainty_penalty,
            excluded_seeds=excluded_seeds,
        )
        for horizon in resolved_horizons
    )
    terminal_target = (
        None
        if terminal_horizon is None
        else _assemble_multi_tape_aggregate_row(
            source_row=first_source_row,
            target_kind="absolute_terminal_world_tick",
            branch_tick=branch_tick,
            horizon_ticks=terminal_horizon,
            terminal_target_world_tick=resolved_terminal_target,
            tape_identity=resolved_tape_identity,
            tape_provenance=tape_entries_by_horizon[terminal_horizon],
            uncertainty_penalty=uncertainty_penalty,
            excluded_seeds=excluded_seeds,
        )
    )
    continuation_count_per_tape = 2 + (2 * len(valid_actions))
    maximum_budget = (
        resolved_tape_count * continuation_count_per_tape * maximum_horizon
    )
    if actual_continuation_tick_count > maximum_budget:
        raise RecurrentCounterfactualBranchError(
            "multi-tape continuation execution exceeded its maximum budget"
        )
    return (
        aggregate_rows,
        terminal_target,
        {
            "continuation_tape_count": resolved_tape_count,
            "continuation_count_per_tape": continuation_count_per_tape,
            "max_horizon_continuation_count": (
                resolved_tape_count * continuation_count_per_tape
            ),
            "exact_replay_continuation_count": (
                resolved_tape_count * (1 + len(valid_actions))
            ),
            "maximum_horizon_ticks": maximum_horizon,
            "actual_continuation_tick_count": actual_continuation_tick_count,
            "maximum_continuation_tick_budget": maximum_budget,
        },
    )


def _validate_multi_tape_evidence_chunk(
    value: Mapping[str, object],
) -> dict[str, object]:
    chunk = dict(_mapping(value, field="multi-tape evidence chunk"))
    expected_fields = {
        "schema_version",
        "tape_identity",
        "continuation_tape_count",
        "tape_indices",
        "execution_horizons",
        "horizon_tape_provenance",
        "actual_continuation_tick_count",
        "maximum_horizon_ticks",
        "source_pre_boundary_environment_rng_state_sha256",
        "source_pre_boundary_policy_sampling_state_sha256",
        "private_checkpoint_serialized",
        "exact_digest",
    }
    if set(chunk) != expected_fields:
        raise RecurrentCounterfactualBranchError(
            "multi-tape evidence chunk field set drifted"
        )
    if (
        chunk.get("schema_version")
        != RECURRENT_COUNTERFACTUAL_TAPE_CHUNK_SCHEMA_VERSION
    ):
        raise RecurrentCounterfactualBranchError(
            "multi-tape evidence chunk schema drifted"
        )
    tape_identity = _nonempty_string(
        chunk.get("tape_identity"),
        field="chunk tape identity",
    )
    tape_count = _positive_int(
        chunk.get("continuation_tape_count"),
        field="chunk continuation tape count",
    )
    raw_indices = chunk.get("tape_indices")
    if not isinstance(raw_indices, tuple):
        raise RecurrentCounterfactualBranchError(
            "chunk tape indices must be a tuple"
        )
    tape_indices = _validated_continuation_tape_indices(
        raw_indices,
        tape_count=tape_count,
    )
    raw_horizons = chunk.get("execution_horizons")
    if not isinstance(raw_horizons, tuple):
        raise RecurrentCounterfactualBranchError(
            "chunk execution horizons must be a tuple"
        )
    execution_horizons = _validated_nested_horizons(raw_horizons)
    if _positive_int(
        chunk.get("maximum_horizon_ticks"),
        field="chunk maximum horizon",
    ) != execution_horizons[-1]:
        raise RecurrentCounterfactualBranchError(
            "chunk maximum horizon drifted"
        )
    _positive_int(
        chunk.get("actual_continuation_tick_count"),
        field="chunk actual continuation tick count",
    )
    pre_environment = chunk.get(
        "source_pre_boundary_environment_rng_state_sha256"
    )
    pre_policy = chunk.get("source_pre_boundary_policy_sampling_state_sha256")
    if not _valid_sha256(pre_environment) or not _valid_sha256(pre_policy):
        raise RecurrentCounterfactualBranchError(
            "chunk source boundary RNG-state digest is malformed"
        )
    if chunk.get("private_checkpoint_serialized") is not False:
        raise RecurrentCounterfactualBranchError(
            "multi-tape evidence chunks cannot serialize private checkpoints"
        )
    raw_horizon_payloads = chunk.get("horizon_tape_provenance")
    if not isinstance(raw_horizon_payloads, tuple) or len(
        raw_horizon_payloads
    ) != len(execution_horizons):
        raise RecurrentCounterfactualBranchError(
            "chunk horizon provenance count drifted"
        )
    for expected_horizon, raw_horizon_payload in zip(
        execution_horizons,
        raw_horizon_payloads,
        strict=True,
    ):
        horizon_payload = _mapping(
            raw_horizon_payload,
            field="chunk horizon provenance",
        )
        if set(horizon_payload) != {"horizon_ticks", "tape_provenance"}:
            raise RecurrentCounterfactualBranchError(
                "chunk horizon provenance field set drifted"
            )
        if horizon_payload.get("horizon_ticks") != expected_horizon:
            raise RecurrentCounterfactualBranchError(
                "chunk horizon provenance order drifted"
            )
        tape_provenance = horizon_payload.get("tape_provenance")
        if not isinstance(tape_provenance, tuple) or len(
            tape_provenance
        ) != len(tape_indices):
            raise RecurrentCounterfactualBranchError(
                "chunk tape provenance count drifted"
            )
        for expected_index, raw_tape in zip(
            tape_indices,
            tape_provenance,
            strict=True,
        ):
            tape = _mapping(raw_tape, field="chunk tape provenance")
            if set(tape) != _MULTI_TAPE_PROVENANCE_FIELDS:
                raise RecurrentCounterfactualBranchError(
                    "chunk tape provenance field set drifted"
                )
            if tape.get("tape_index") != expected_index:
                raise RecurrentCounterfactualBranchError(
                    "chunk tape provenance index drifted"
                )
            environment_identity = f"{tape_identity}:environment:{expected_index}"
            policy_identity = f"{tape_identity}:policy:{expected_index}"
            if (
                tape.get("environment_sampling_identity") != environment_identity
                or tape.get("policy_sampling_identity") != policy_identity
                or tape.get("environment_sampling_seed")
                != derive_recurrent_counterfactual_tape_seed(
                    namespace=(
                        RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE
                    ),
                    identity=environment_identity,
                )
                or tape.get("policy_sampling_seed")
                != derive_recurrent_counterfactual_tape_seed(
                    namespace=(
                        RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE
                    ),
                    identity=policy_identity,
                )
            ):
                raise RecurrentCounterfactualBranchError(
                    "chunk tape seed provenance drifted"
                )
            boundary = _decision_boundary_provenance(tape)
            if (
                boundary["pre_boundary_environment_rng_state_sha256"]
                != pre_environment
                or boundary["pre_boundary_policy_sampling_state_sha256"]
                != pre_policy
            ):
                raise RecurrentCounterfactualBranchError(
                    "chunk tape source boundary drifted"
                )
    exact_digest = chunk.pop("exact_digest")
    if not _valid_sha256(exact_digest) or exact_digest != stable_payload_digest(
        chunk
    ):
        raise RecurrentCounterfactualBranchError(
            "multi-tape evidence chunk exact digest mismatch"
        )
    chunk["exact_digest"] = exact_digest
    return chunk


def _compact_multi_tape_baseline_evidence(
    run: Mapping[str, object],
    *,
    replay: Mapping[str, object],
) -> dict[str, object]:
    return {
        "natural_requested_action": _stable_action(
            run.get("natural_requested_action"),
            field="multi-tape baseline action",
        ),
        **_compact_multi_tape_outcome_metrics(run),
        "first_transition": deepcopy(run.get("first_transition")),
        "unsupported_requested_action_count": run.get(
            "unsupported_requested_action_count"
        ),
        "heuristic_action_source_count": run.get("heuristic_action_source_count"),
        "unexpected_action_source_count": run.get("unexpected_action_source_count"),
        "behavior_digest": run.get("behavior_digest"),
        "evidence_digest": run.get("evidence_digest"),
        "replay_verified": True,
        "replay_evidence_digest": replay.get("evidence_digest"),
        **_decision_boundary_provenance(run),
    }


def _compact_multi_tape_action_evidence(
    run: Mapping[str, object],
    *,
    replay: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    action = _stable_action(run.get("forced_action"), field="multi-tape action")
    return {
        "action": action,
        "natural_requested_action": _stable_action(
            run.get("natural_requested_action"),
            field="multi-tape natural action",
        ),
        **_compact_multi_tape_outcome_metrics(run),
        "first_transition": deepcopy(run.get("first_transition")),
        "paired_vs_baseline": _paired_delta_payload(run, baseline=baseline),
        "unsupported_requested_action_count": run.get(
            "unsupported_requested_action_count"
        ),
        "heuristic_action_source_count": run.get("heuristic_action_source_count"),
        "unexpected_action_source_count": run.get("unexpected_action_source_count"),
        "behavior_digest": run.get("behavior_digest"),
        "evidence_digest": run.get("evidence_digest"),
        "replay_verified": True,
        "replay_evidence_digest": replay.get("evidence_digest"),
        **_decision_boundary_provenance(run),
    }


def _compact_multi_tape_outcome_metrics(
    run: Mapping[str, object],
) -> dict[str, object]:
    terminal = _mapping(run.get("focal_terminal"), field="multi-tape terminal")
    alive = terminal.get("alive")
    if type(alive) is not bool:
        raise RecurrentCounterfactualBranchError(
            "multi-tape focal terminal alive must be an exact boolean"
        )
    return {
        "focal_discounted_return": _finite_float(
            run.get("focal_discounted_return"),
            field="multi-tape focal discounted return",
        ),
        "focal_terminal_alive": alive,
        "population_alive": _nonnegative_int(
            run.get("population_alive"),
            field="multi-tape population alive",
        ),
        "births_during_horizon": _nonnegative_int(
            run.get("births_during_horizon"),
            field="multi-tape births",
        ),
        "deaths_during_horizon": _nonnegative_int(
            run.get("deaths_during_horizon"),
            field="multi-tape deaths",
        ),
    }


def _paired_delta_payload(
    run: Mapping[str, object],
    *,
    baseline: Mapping[str, object],
) -> dict[str, object]:
    run_terminal = _mapping(run.get("focal_terminal"), field="run terminal")
    baseline_terminal = _mapping(
        baseline.get("focal_terminal"),
        field="baseline terminal",
    )
    return {
        "focal_discounted_return_delta": _round(
            float(run["focal_discounted_return"])
            - float(baseline["focal_discounted_return"])
        ),
        "focal_terminal_alive_delta": int(run_terminal.get("alive") is True)
        - int(baseline_terminal.get("alive") is True),
        "population_alive_delta": int(run["population_alive"])
        - int(baseline["population_alive"]),
        "births_during_horizon_delta": int(run["births_during_horizon"])
        - int(baseline["births_during_horizon"]),
        "deaths_during_horizon_delta": int(run["deaths_during_horizon"])
        - int(baseline["deaths_during_horizon"]),
    }


def _source_boundary_identity_payload(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Project the public/digest-only source facts that identify one boundary."""

    return {
        "seed_role": metadata.get("seed_role"),
        "environment_seed": metadata.get("environment_seed"),
        "scenario": metadata.get("scenario"),
        "branch_tick": metadata.get("branch_tick"),
        "focal_agent_id": metadata.get("focal_agent_id"),
        "source_artifact_digest": metadata.get("source_artifact_digest"),
        "source_model_state_sha256": metadata.get("source_model_state_sha256"),
        "policy_sampling_seed": metadata.get("policy_sampling_seed"),
        "source_decision_index": metadata.get("source_decision_index"),
        "source_record_digest": metadata.get("source_record_digest"),
        "source_decision_diagnostics_digest": metadata.get(
            "source_decision_diagnostics_digest"
        ),
        "source_observation_digest": metadata.get("source_observation_digest"),
        "source_action_mask_digest": metadata.get("source_action_mask_digest"),
        "source_sampling_state_sha256": metadata.get(
            "source_sampling_state_sha256"
        ),
        "source_public_history_prefix_sha256": metadata.get(
            "source_public_history_prefix_sha256"
        ),
        "source_pre_boundary_environment_rng_state_sha256": metadata.get(
            "source_pre_boundary_environment_rng_state_sha256"
        ),
        "source_pre_boundary_policy_sampling_state_sha256": metadata.get(
            "source_pre_boundary_policy_sampling_state_sha256"
        ),
    }


def _bind_branch_rows_to_source_boundary(
    rows: Sequence[Mapping[str, object]],
    *,
    pre_boundary_environment_rng_state_sha256: object,
    pre_boundary_policy_sampling_state_sha256: object,
) -> tuple[dict[str, object], ...]:
    """Commit digest-only decision-boundary evidence into each source row."""

    if not rows:
        raise RecurrentCounterfactualBranchError(
            "source-boundary binding requires at least one branch row"
        )
    if not _valid_sha256(pre_boundary_environment_rng_state_sha256):
        raise RecurrentCounterfactualBranchError(
            "source pre-boundary environment RNG-state digest is malformed"
        )
    if not _valid_sha256(pre_boundary_policy_sampling_state_sha256):
        raise RecurrentCounterfactualBranchError(
            "source pre-boundary policy RNG-state digest is malformed"
        )
    bound_rows: list[dict[str, object]] = []
    for source_row in rows:
        validate_recurrent_counterfactual_branch_row(source_row)
        row = deepcopy(dict(source_row))
        metadata = dict(_mapping(row.get("metadata"), field="source metadata"))
        metadata.update(
            {
                "source_pre_boundary_environment_rng_state_sha256": (
                    pre_boundary_environment_rng_state_sha256
                ),
                "source_pre_boundary_policy_sampling_state_sha256": (
                    pre_boundary_policy_sampling_state_sha256
                ),
            }
        )
        metadata["source_boundary_identity_sha256"] = stable_payload_digest(
            _source_boundary_identity_payload(metadata)
        )
        row["schema_version"] = (
            RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION
        )
        row["metadata"] = metadata
        components = dict(
            _mapping(row.get("component_digests"), field="source component digests")
        )
        components["metadata"] = stable_payload_digest(metadata)
        row["component_digests"] = components
        row.pop("exact_digest", None)
        row["exact_digest"] = stable_payload_digest(row)
        validate_recurrent_counterfactual_branch_row(row)
        bound_rows.append(row)
    return tuple(bound_rows)


def _assemble_multi_tape_aggregate_row(
    *,
    source_row: Mapping[str, object],
    target_kind: str,
    branch_tick: int,
    horizon_ticks: int,
    terminal_target_world_tick: int | None,
    tape_identity: str,
    tape_provenance: Sequence[Mapping[str, object]],
    uncertainty_penalty: float,
    excluded_seeds: Sequence[int],
) -> dict[str, object]:
    validate_recurrent_counterfactual_branch_row(source_row)
    if len(excluded_seeds) != 3:
        raise RecurrentCounterfactualBranchError(
            "aggregate source seed exclusion must contain environment, policy, "
            "and branch-selection seeds"
        )
    source_environment_seed = _required_sampling_seed(
        excluded_seeds[0],
        field="aggregate source environment seed",
    )
    source_policy_sampling_seed = _required_sampling_seed(
        excluded_seeds[1],
        field="aggregate source policy sampling seed",
    )
    source_branch_selection_seed = _required_sampling_seed(
        excluded_seeds[2],
        field="aggregate source branch-selection seed",
    )
    source_metadata = _mapping(source_row.get("metadata"), field="source metadata")
    source_labels = _mapping(source_row.get("labels"), field="source labels")
    trainable_public_context = deepcopy(
        dict(
            _mapping(
                source_row.get("trainable_public_context"),
                field="source trainable public context",
            )
        )
    )
    optimizer_context = deepcopy(
        dict(
            _mapping(
                source_row.get("optimizer_context"),
                field="source optimizer context",
            )
        )
    )
    source_checkpoint_identity = {
        "source_branch_row_exact_digest": source_row.get("exact_digest"),
        "seed_role": source_metadata.get("seed_role"),
        "environment_seed": source_metadata.get("environment_seed"),
        "branch_selection_seed": source_branch_selection_seed,
        "scenario": source_metadata.get("scenario"),
        "branch_tick": branch_tick,
        "focal_agent_id": source_metadata.get("focal_agent_id"),
        "source_record_digest": source_metadata.get("source_record_digest"),
        "source_observation_digest": source_metadata.get("source_observation_digest"),
        "source_action_mask_digest": source_metadata.get("source_action_mask_digest"),
        "source_public_history_prefix_sha256": source_metadata.get(
            "source_public_history_prefix_sha256"
        ),
        "source_pre_boundary_environment_rng_state_sha256": source_metadata.get(
            "source_pre_boundary_environment_rng_state_sha256"
        ),
        "source_pre_boundary_policy_sampling_state_sha256": source_metadata.get(
            "source_pre_boundary_policy_sampling_state_sha256"
        ),
        "source_boundary_identity_sha256": source_metadata.get(
            "source_boundary_identity_sha256"
        ),
    }
    source_identity = {
        "source_branch_row_exact_digest": source_row.get("exact_digest"),
        "source_model_state_sha256": source_metadata.get("source_model_state_sha256"),
        "source_artifact_digest": source_metadata.get("source_artifact_digest"),
        "source_environment_seed": source_environment_seed,
        "source_policy_sampling_seed": source_policy_sampling_seed,
        "source_branch_selection_seed": source_branch_selection_seed,
        "source_checkpoint_identity_sha256": stable_payload_digest(
            source_checkpoint_identity
        ),
        "source_pre_boundary_environment_rng_state_sha256": source_metadata.get(
            "source_pre_boundary_environment_rng_state_sha256"
        ),
        "source_pre_boundary_policy_sampling_state_sha256": source_metadata.get(
            "source_pre_boundary_policy_sampling_state_sha256"
        ),
        "source_boundary_identity_sha256": source_metadata.get(
            "source_boundary_identity_sha256"
        ),
        "trainable_public_context_sha256": stable_payload_digest(
            trainable_public_context
        ),
    }
    if (
        source_metadata.get("environment_seed") != source_environment_seed
        or source_metadata.get("policy_sampling_seed")
        != source_policy_sampling_seed
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate excluded source seeds drifted from source metadata"
        )
    source_behavior = {
        "source_requested_action": source_labels.get("source_requested_action"),
        "current_public_action_mask": deepcopy(
            trainable_public_context["current_public_action_mask"]
        ),
        "source_behavior_distribution": deepcopy(
            source_labels.get("source_behavior_distribution")
        ),
    }
    tape_contract = {
        "tape_identity": tape_identity,
        "tape_count": len(tape_provenance),
        "environment_seed_namespace": (
            RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE
        ),
        "policy_seed_namespace": (
            RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE
        ),
        "excluded_source_seeds": list(excluded_seeds),
        "same_exact_source_checkpoint": True,
        "continuation_rng_retape_boundary": (
            RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY
        ),
        "fixed_source_prefix_through_focal_natural_draw": True,
        "horizon_includes_branch_tick": True,
        "same_tick_post_focal_consequences_governed_by_tape": True,
        "policy_sampling_generator_device_type": "cpu",
        "common_boundary_rng_state_across_actions_within_tape": True,
        "paired_baseline_and_forced_actions_share_tape": True,
        "exact_replay_required_for_baseline_and_actions": True,
        "event_aligned_common_random_numbers": False,
        "sequential_rng_event_reassignment_after_divergence_possible": True,
        "private_world_checkpoint_serialized": False,
    }
    copied_tapes = deepcopy([dict(tape) for tape in tape_provenance])
    aggregate = _aggregate_multi_tape_outcomes(
        copied_tapes,
        uncertainty_penalty=uncertainty_penalty,
    )
    target = {
        "kind": target_kind,
        "branch_tick": branch_tick,
        "horizon_ticks": horizon_ticks,
        "target_world_tick": branch_tick + horizon_ticks,
        "absolute_terminal_target_world_tick": terminal_target_world_tick,
        "is_absolute_terminal_target": (target_kind == "absolute_terminal_world_tick"),
    }
    components = {
        "target": stable_payload_digest(target),
        "trainable_public_context": stable_payload_digest(trainable_public_context),
        "optimizer_context": stable_payload_digest(optimizer_context),
        "source_identity": stable_payload_digest(source_identity),
        "source_behavior": stable_payload_digest(source_behavior),
        "tape_contract": stable_payload_digest(tape_contract),
        "tape_provenance": stable_payload_digest(copied_tapes),
        "aggregate": stable_payload_digest(aggregate),
    }
    row: dict[str, object] = {
        "schema_version": RECURRENT_COUNTERFACTUAL_AGGREGATE_SCHEMA_VERSION,
        "target": target,
        "trainable_public_context": trainable_public_context,
        "optimizer_context": optimizer_context,
        "source_identity": source_identity,
        "source_behavior": source_behavior,
        "tape_contract": tape_contract,
        "tape_provenance": copied_tapes,
        "aggregate": aggregate,
        "component_digests": components,
        "training_ran": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
    }
    row["exact_digest"] = stable_payload_digest(row)
    validate_recurrent_counterfactual_aggregate_row(row)
    return row


def _aggregate_multi_tape_outcomes(
    tape_provenance: Sequence[Mapping[str, object]],
    *,
    uncertainty_penalty: float,
) -> dict[str, object]:
    tape_count = len(tape_provenance)
    if tape_count <= 0:
        raise RecurrentCounterfactualBranchError(
            "multi-tape aggregate requires at least one tape"
        )
    baseline_values = {
        field: [
            _multi_tape_metric(
                _mapping(tape.get("baseline"), field="tape baseline"),
                field=field,
            )
            for tape in tape_provenance
        ]
        for field in (
            "focal_discounted_return",
            "focal_terminal_alive",
            "population_alive",
            "births_during_horizon",
            "deaths_during_horizon",
        )
    }
    first_actions = _mapping(
        tape_provenance[0],
        field="first tape",
    ).get("action_outcomes")
    if not isinstance(first_actions, list) or not first_actions:
        raise RecurrentCounterfactualBranchError(
            "multi-tape action outcomes cannot be empty"
        )
    actions = [
        _stable_action(
            _mapping(item, field="first tape action").get("action"),
            field="first tape action",
        )
        for item in first_actions
    ]
    action_aggregates: list[dict[str, object]] = []
    for action in actions:
        outcomes = [
            next(
                _mapping(item, field="tape action outcome")
                for item in _action_outcome_sequence(tape)
                if _mapping(item, field="tape action outcome").get("action") == action
            )
            for tape in tape_provenance
        ]
        outcome_statistics = {
            field: _sample_statistics(
                [_multi_tape_metric(outcome, field=field) for outcome in outcomes]
            )
            for field in (
                "focal_discounted_return",
                "focal_terminal_alive",
                "population_alive",
                "births_during_horizon",
                "deaths_during_horizon",
            )
        }
        paired_statistics = {
            field: _sample_statistics(
                [
                    _finite_float(
                        _mapping(
                            outcome.get("paired_vs_baseline"),
                            field="paired outcome",
                        ).get(field),
                        field=field,
                    )
                    for outcome in outcomes
                ]
            )
            for field in _PAIRED_DELTA_FIELDS
        }
        return_delta = paired_statistics["focal_discounted_return_delta"]
        action_aggregates.append(
            {
                "action": action,
                "tape_count": tape_count,
                "outcome_statistics": outcome_statistics,
                "paired_delta_statistics": paired_statistics,
                "focal_survival_probability": outcome_statistics[
                    "focal_terminal_alive"
                ]["mean"],
                "uncertainty_penalized_score": _round(
                    float(return_delta["mean"])
                    - uncertainty_penalty * float(return_delta["standard_error"])
                ),
            }
        )
    return {
        "tape_count": tape_count,
        "uncertainty_penalty": uncertainty_penalty,
        "uncertainty_penalized_score_policy": (
            "paired_focal_discounted_return_delta_mean_minus_coefficient_times_standard_error"
        ),
        "baseline_outcome_statistics": {
            field: _sample_statistics(values)
            for field, values in baseline_values.items()
        },
        "baseline_focal_survival_probability": _sample_statistics(
            baseline_values["focal_terminal_alive"]
        )["mean"],
        "action_outcomes": action_aggregates,
    }


def _action_outcome_sequence(tape: Mapping[str, object]) -> list[object]:
    outcomes = tape.get("action_outcomes")
    if not isinstance(outcomes, list):
        raise RecurrentCounterfactualBranchError("tape action outcomes must be a list")
    return outcomes


def _multi_tape_metric(outcome: Mapping[str, object], *, field: str) -> float:
    value = outcome.get(field)
    if field == "focal_terminal_alive":
        if type(value) is not bool:
            raise RecurrentCounterfactualBranchError(
                "focal terminal alive must be an exact boolean"
            )
        return float(value)
    return _finite_float(value, field=field)


def _sample_statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise RecurrentCounterfactualBranchError(
            "sample statistics require at least one value"
        )
    parsed = tuple(_finite_float(value, field="sample value") for value in values)
    mean = math.fsum(parsed) / len(parsed)
    sample_variance = (
        0.0
        if len(parsed) == 1
        else math.fsum((value - mean) ** 2 for value in parsed) / (len(parsed) - 1)
    )
    standard_error = math.sqrt(sample_variance / len(parsed))
    return {
        "mean": _round(mean),
        "sample_variance": _round(sample_variance),
        "standard_error": _round(standard_error),
    }


def _assemble_nested_branch_row(
    *,
    artifact_digest: str,
    seed_role: str,
    environment_seed: int,
    scenario: str,
    branch_tick: int,
    horizon_ticks: int,
    gamma: float,
    policy_sampling_seed: int,
    focal_agent_id: int,
    source_action: str,
    source_mask: Mapping[str, bool],
    valid_actions: Sequence[str],
    source_record: Mapping[str, object],
    source_decision_diagnostics: Mapping[str, object],
    source_behavior_distribution: Mapping[str, object],
    checkpoint_context: Mapping[str, object],
    source_model_state_sha256: str,
    public_history_prefix: Mapping[str, object],
    public_history_prefix_sha256: str,
    source_observation_digest: str,
    source_action_mask_digest: str,
    baseline: Mapping[str, object],
    action_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    trainable_public_context = {
        "public_history_prefix": deepcopy(dict(public_history_prefix)),
        "current_public_observation": ecological_policy_input_payload(
            _mapping(
                source_record.get("observation_input"),
                field="source observation_input",
            )
        ),
        "current_public_action_mask": dict(source_mask),
        "previous_public_feedback": {
            "schema_version": PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION,
            "shape": [PREVIOUS_PUBLIC_FEEDBACK_SIZE],
            "values": list(checkpoint_context["previous_public_feedback"]),
        },
    }
    optimizer_context = {
        "source_artifact_digest": artifact_digest,
        "source_model_state_sha256": source_model_state_sha256,
        "source_recurrent_state": deepcopy(checkpoint_context["recurrent_state"]),
        "source_recurrent_state_shape": list(
            checkpoint_context["recurrent_state_shape"]
        ),
        "source_recurrent_state_sha256": checkpoint_context["recurrent_state_sha256"],
        "source_public_history_prefix_sha256": public_history_prefix_sha256,
        "derived_from_public_history": True,
        "source_artifact_match_required": True,
        "stored_state_usage": "exact_source_artifact_verification_only",
        "current_model_state_policy": (
            "reconstruct_from_trainable_public_history_prefix"
        ),
        "runtime_environment_input": False,
    }
    labels = {
        "source_requested_action": source_action,
        "source_policy_value": _finite_float(
            source_decision_diagnostics.get("value"),
            field="source policy value",
        ),
        "source_behavior_distribution": dict(source_behavior_distribution),
        "baseline": _compact_baseline_label(baseline),
        "action_outcomes": [
            _compact_action_label(run, baseline=baseline) for run in action_runs
        ],
        "all_tick_start_mask_valid_actions_evaluated": True,
        "valid_action_count": len(valid_actions),
        "baseline_source_action_behavior_match": True,
    }
    contract = _counterfactual_contract(
        verify_replay=True,
        seed_role=seed_role,
    )
    protocol_digest = stable_payload_digest(contract)
    source_record_digest = stable_payload_digest(source_record)
    source_diagnostics_digest = stable_payload_digest(source_decision_diagnostics)
    continuation_provenance = {
        "continuation_policy_artifact_digest": artifact_digest,
        "continuation_policy_model_state_sha256": source_model_state_sha256,
        "continuation_policy_action_selection": checkpoint_context["action_selection"],
        "continuation_policy_sampling_seed": policy_sampling_seed,
        "continuation_checkpoint_sampling_state_sha256": checkpoint_context[
            "sampling_state_sha256"
        ],
        "branch_horizon_ticks": horizon_ticks,
        "exact_replay_repeat_count_per_action": 2,
        "repeat_indices": [0, 1],
        "repeat_policy_sampling_seeds": [
            policy_sampling_seed,
            policy_sampling_seed,
        ],
        "repeat_checkpoint_sampling_state_sha256": [
            checkpoint_context["sampling_state_sha256"],
            checkpoint_context["sampling_state_sha256"],
        ],
        "common_checkpoint_across_actions": True,
        "source_checkpoint_reused_by_deepcopy": True,
    }
    continuation_provenance_digest = stable_payload_digest(continuation_provenance)
    branch_identity_digest = stable_payload_digest(
        {
            "protocol_digest": protocol_digest,
            "artifact_digest": artifact_digest,
            "model_state_sha256": source_model_state_sha256,
            "seed_role": seed_role,
            "environment_seed": environment_seed,
            "policy_sampling_seed": policy_sampling_seed,
            "scenario": scenario,
            "branch_tick": branch_tick,
            "horizon_ticks": horizon_ticks,
            "exact_replay_repeat_count_per_action": 2,
            "focal_agent_id": focal_agent_id,
            "source_record_digest": source_record_digest,
            "source_recurrent_state_sha256": checkpoint_context[
                "recurrent_state_sha256"
            ],
            "source_sampling_state_sha256": checkpoint_context["sampling_state_sha256"],
            "source_public_history_prefix_sha256": (public_history_prefix_sha256),
            "continuation_provenance_digest": (continuation_provenance_digest),
        }
    )
    metadata = {
        "seed_role": seed_role,
        "environment_seed": environment_seed,
        "scenario": scenario,
        "branch_tick": branch_tick,
        "horizon_ticks": horizon_ticks,
        "focal_agent_id": focal_agent_id,
        "gamma": gamma,
        "source_artifact_digest": artifact_digest,
        "source_model_state_sha256": source_model_state_sha256,
        "policy_sampling_seed": policy_sampling_seed,
        "source_action_selection": checkpoint_context["action_selection"],
        "source_decision_index": source_decision_diagnostics.get("decision_index"),
        "source_record_digest": source_record_digest,
        "source_decision_diagnostics_digest": source_diagnostics_digest,
        "source_observation_digest": source_observation_digest,
        "source_action_mask_digest": source_action_mask_digest,
        "source_sampling_state_sha256": checkpoint_context["sampling_state_sha256"],
        "source_public_history_prefix_sha256": public_history_prefix_sha256,
        "branch_identity_digest": branch_identity_digest,
        "branch_protocol_digest": protocol_digest,
        "continuation_provenance": continuation_provenance,
        "continuation_provenance_digest": continuation_provenance_digest,
        "baseline_behavior_digest": baseline["behavior_digest"],
        "baseline_evidence_digest": baseline["evidence_digest"],
        "private_checkpoint_serialized": False,
    }
    row: dict[str, object] = {
        "schema_version": RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION,
        "contract": contract,
        "trainable_public_context": trainable_public_context,
        "optimizer_context": optimizer_context,
        "labels": labels,
        "metadata": metadata,
        "component_digests": {
            "contract": protocol_digest,
            "trainable_public_context": stable_payload_digest(trainable_public_context),
            "optimizer_context": stable_payload_digest(optimizer_context),
            "labels": stable_payload_digest(labels),
            "metadata": stable_payload_digest(metadata),
        },
        "training_ran": False,
        "training_artifact_created": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
    }
    row["exact_digest"] = stable_payload_digest(row)
    validate_recurrent_counterfactual_branch_row(row)
    return row


def _validated_branch_tick_candidates(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecurrentCounterfactualBranchError(
            "branch tick candidates must be an ordered sequence"
        )
    parsed = tuple(
        _nonnegative_int(item, field="branch tick candidate") for item in value
    )
    if not parsed or len(set(parsed)) != len(parsed):
        raise RecurrentCounterfactualBranchError(
            "branch tick candidates must be non-empty and unique"
        )
    return parsed


def _validated_nested_horizons(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecurrentCounterfactualBranchError(
            "nested horizons must be an ordered sequence"
        )
    parsed = tuple(_positive_int(item, field="nested horizon") for item in value)
    if len(parsed) < 2 or tuple(sorted(set(parsed))) != parsed:
        raise RecurrentCounterfactualBranchError(
            "nested horizons must contain at least two unique increasing values"
        )
    return parsed


def _required_sampling_seed(value: object, *, field: str) -> int:
    parsed = _optional_sampling_seed(value, field=field)
    if parsed is None:
        raise RecurrentCounterfactualBranchError(f"{field} is required")
    return parsed


def _eligible_focal_source_records(
    records: Sequence[Mapping[str, object]],
    diagnostics: Sequence[object],
) -> tuple[tuple[dict[str, object], dict[str, object]], ...]:
    if len(records) != len(diagnostics):
        raise RecurrentCounterfactualBranchError(
            "source records and decision diagnostics are not aligned"
        )
    candidates: list[tuple[int, int, dict[str, object], dict[str, object]]] = []
    for index, (record, raw_diagnostics) in enumerate(
        zip(records, diagnostics, strict=True)
    ):
        action_source = record.get("action_source")
        if action_source == "passive":
            if raw_diagnostics is not None:
                raise RecurrentCounterfactualBranchError(
                    "passive source decision diagnostics must be absent"
                )
            continue
        if action_source != RECURRENT_ROLLOUT_ACTION_SOURCE:
            raise RecurrentCounterfactualBranchError(
                "source action source is not canonical recurrent or passive"
            )
        agent_id = _positive_int(record.get("agent_id"), field="source agent_id")
        action_mask = _complete_action_mask(record.get("action_mask"))
        if sum(action_mask.values()) < 2:
            continue
        parsed_diagnostics = dict(
            _mapping(raw_diagnostics, field="source decision diagnostics")
        )
        if parsed_diagnostics.get("agent_id") != agent_id:
            raise RecurrentCounterfactualBranchError(
                "source diagnostic agent identity is not aligned"
            )
        candidates.append((agent_id, index, dict(record), parsed_diagnostics))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return tuple((record, diagnostic) for _, _, record, diagnostic in candidates)


def _copy_aligned_source_decision_diagnostics(
    records: Sequence[Mapping[str, object]],
    diagnostics: Sequence[object],
) -> tuple[dict[str, object] | None, ...]:
    if len(records) != len(diagnostics):
        raise RecurrentCounterfactualBranchError(
            "source records and decision diagnostics are not aligned"
        )
    copied: list[dict[str, object] | None] = []
    for record, diagnostic in zip(records, diagnostics, strict=True):
        action_source = record.get("action_source")
        if action_source == "passive":
            if diagnostic is not None:
                raise RecurrentCounterfactualBranchError(
                    "passive source decision diagnostics must be absent"
                )
            copied.append(None)
            continue
        if action_source != RECURRENT_ROLLOUT_ACTION_SOURCE:
            raise RecurrentCounterfactualBranchError(
                "source action source is not canonical recurrent or passive"
            )
        copied.append(dict(_mapping(diagnostic, field="source decision diagnostics")))
    return tuple(copied)


def _focal_candidate_payload(
    record: Mapping[str, object],
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    mask = _complete_action_mask(record.get("action_mask"))
    return {
        "agent_id": _positive_int(record.get("agent_id"), field="candidate agent_id"),
        "decision_index": _nonnegative_int(
            diagnostics.get("decision_index"),
            field="candidate decision_index",
        ),
        "observation_digest": _nonempty_string(
            record.get("observation_digest"),
            field="candidate observation digest",
        ),
        "action_mask_digest": stable_payload_digest(mask),
        "valid_action_count": sum(mask.values()),
    }


def _nested_runs_by_horizon(
    maximum_run: Mapping[str, object],
    *,
    horizons: Sequence[int],
    field: str,
) -> dict[int, dict[str, object]]:
    raw = _mapping(
        maximum_run.get("nested_prefix_runs"),
        field=f"{field} nested prefix runs",
    )
    expected_keys = {str(horizon) for horizon in horizons}
    if set(raw) != expected_keys:
        raise RecurrentCounterfactualBranchError(
            f"{field} nested prefix horizon set drifted"
        )
    return {
        horizon: deepcopy(
            dict(
                _mapping(
                    raw[str(horizon)],
                    field=f"{field} horizon {horizon}",
                )
            )
        )
        for horizon in horizons
    }


def _nested_prefix_proof(
    *,
    horizons: Sequence[int],
    baseline_by_horizon: Mapping[int, Mapping[str, object]],
    action_runs_by_horizon: Mapping[int, Sequence[Mapping[str, object]]],
    replay_runs_by_action_horizon: Mapping[
        str,
        Mapping[int, Mapping[str, object]],
    ],
    valid_actions: Sequence[str],
) -> dict[str, object]:
    streams: dict[str, object] = {
        "baseline": _nested_prefix_stream_proof(
            baseline_by_horizon,
            horizons=horizons,
            field="baseline",
        )
    }
    for action_index, action in enumerate(valid_actions):
        action_runs = {
            horizon: action_runs_by_horizon[horizon][action_index]
            for horizon in horizons
        }
        streams[f"action:{action}"] = _nested_prefix_stream_proof(
            action_runs,
            horizons=horizons,
            field=f"action {action}",
        )
        streams[f"replay:{action}"] = _nested_prefix_stream_proof(
            replay_runs_by_action_horizon[action],
            horizons=horizons,
            field=f"replay {action}",
        )
    return {
        "schema_version": "mind_v3_recurrent_nested_prefix_proof_v1",
        "horizons": list(horizons),
        "stream_count": len(streams),
        "streams": streams,
        "all_shorter_streams_are_exact_max_stream_prefixes": True,
        "raw_private_world_state_serialized": False,
    }


def _nested_prefix_stream_proof(
    runs: Mapping[int, Mapping[str, object]],
    *,
    horizons: Sequence[int],
    field: str,
) -> dict[str, object]:
    sequence_fields = (
        "causal_record_digest_sequence",
        "policy_record_digest_sequence",
        "diagnostics_digest_sequence",
    )
    maximum = runs[horizons[-1]]
    proof_by_horizon: dict[str, object] = {}
    for horizon in horizons:
        run = runs[horizon]
        horizon_proof: dict[str, object] = {
            "behavior_digest": run.get("behavior_digest"),
            "evidence_digest": run.get("evidence_digest"),
            "executed_tick_count": run.get("executed_tick_count"),
        }
        for sequence_field in sequence_fields:
            observed = run.get(sequence_field)
            maximum_sequence = maximum.get(sequence_field)
            if not isinstance(observed, tuple) or not isinstance(
                maximum_sequence,
                tuple,
            ):
                raise RecurrentCounterfactualBranchError(
                    f"{field} prefix digest sequence is malformed"
                )
            if observed != maximum_sequence[: len(observed)]:
                raise RecurrentCounterfactualBranchError(
                    f"{field} horizon {horizon} is not an exact maximum prefix"
                )
            horizon_proof[sequence_field] = list(observed)
        proof_by_horizon[str(horizon)] = horizon_proof
    return {
        "horizons": proof_by_horizon,
        "exact_prefix": True,
    }


def reconstruct_current_model_hidden_from_branch_row(
    model: PublicRecurrentActorCritic,
    row: Mapping[str, object],
) -> Tensor:
    """Recompute a branch hidden state from actor-public history only."""

    validate_recurrent_counterfactual_branch_row(row)
    context = _mapping(
        row.get("trainable_public_context"),
        field="trainable_public_context",
    )
    prefix = _mapping(
        context.get("public_history_prefix"),
        field="public_history_prefix",
    )
    try:
        return reconstruct_current_model_hidden_from_public_prefix(model, prefix)
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "current model could not reconstruct branch recurrent state"
        ) from error


def verified_source_recurrent_state_from_branch_row(
    model: PublicRecurrentActorCritic,
    row: Mapping[str, object],
    *,
    artifact_digest: str,
) -> Tensor:
    """Read stored source hidden evidence only after exact identity checks."""

    validate_recurrent_counterfactual_branch_row(row)
    optimizer_context = _mapping(
        row.get("optimizer_context"),
        field="optimizer_context",
    )
    try:
        stored_state = verified_source_recurrent_state_for_exact_artifact(
            model,
            optimizer_context,
            artifact_digest=artifact_digest,
        )
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "stored source hidden failed exact artifact verification"
        ) from error
    context = _mapping(
        row.get("trainable_public_context"),
        field="trainable_public_context",
    )
    prefix = _mapping(
        context.get("public_history_prefix"),
        field="public_history_prefix",
    )
    try:
        reconstructed_state = reconstruct_current_model_hidden_from_public_prefix(
            model,
            prefix,
        )
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "stored source hidden public-history reconstruction failed"
        ) from error
    if not stored_state.equal(reconstructed_state):
        raise RecurrentCounterfactualBranchError(
            "stored source hidden does not exactly match its public-history "
            "reconstruction"
        )
    return stored_state


def validate_recurrent_counterfactual_branch_row(
    row: Mapping[str, object],
) -> None:
    if not isinstance(row, Mapping):
        raise RecurrentCounterfactualBranchError("branch row must be a mapping")
    if set(row) != _BRANCH_ROW_FIELDS:
        raise RecurrentCounterfactualBranchError("branch row field set drifted")
    schema_version = row.get("schema_version")
    if schema_version not in {
        RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION,
        RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION,
    }:
        raise RecurrentCounterfactualBranchError("branch row schema version drifted")
    context = _mapping(
        row.get("trainable_public_context"),
        field="trainable_public_context",
    )
    if set(context) != _TRAINABLE_PUBLIC_CONTEXT_KEYS:
        raise RecurrentCounterfactualBranchError(
            "trainable public context field set drifted"
        )
    forbidden = _forbidden_key_paths(context)
    if forbidden:
        raise RecurrentCounterfactualBranchError(
            f"trainable public context contains forbidden keys: {forbidden[:8]}"
        )
    public_history_prefix = _mapping(
        context.get("public_history_prefix"),
        field="public_history_prefix",
    )
    try:
        validate_public_recurrent_history_prefix(public_history_prefix)
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "public recurrent history prefix is invalid"
        ) from error
    public_history_prefix_sha256 = stable_payload_digest(public_history_prefix)
    observation = _mapping(
        context.get("current_public_observation"),
        field="current_public_observation",
    )
    _validate_current_public_observation(observation)
    mask = _complete_action_mask(context.get("current_public_action_mask"))
    feedback = _mapping(
        context.get("previous_public_feedback"),
        field="previous_public_feedback",
    )
    _validate_previous_public_feedback(feedback)
    _validate_feedback_history_alignment(
        public_history_prefix,
        feedback,
        field="previous public feedback",
    )

    optimizer_context = _mapping(
        row.get("optimizer_context"),
        field="optimizer_context",
    )
    if set(optimizer_context) != _OPTIMIZER_CONTEXT_KEYS:
        raise RecurrentCounterfactualBranchError("optimizer context field set drifted")
    if optimizer_context.get("derived_from_public_history") is not True:
        raise RecurrentCounterfactualBranchError(
            "optimizer recurrent state must be public-history-derived"
        )
    if optimizer_context.get("runtime_environment_input") is not False:
        raise RecurrentCounterfactualBranchError(
            "optimizer context cannot become an environment input"
        )
    if optimizer_context.get("source_artifact_match_required") is not True:
        raise RecurrentCounterfactualBranchError(
            "optimizer recurrent state must require its source artifact"
        )
    if not _nonempty_string(
        optimizer_context.get("source_artifact_digest"),
        field="optimizer source artifact digest",
    ):
        raise AssertionError("validated source artifact digest became empty")
    if not _valid_sha256(optimizer_context.get("source_model_state_sha256")):
        raise RecurrentCounterfactualBranchError(
            "optimizer source model-state digest is malformed"
        )
    if optimizer_context.get("source_public_history_prefix_sha256") != (
        public_history_prefix_sha256
    ):
        raise RecurrentCounterfactualBranchError(
            "optimizer public-history prefix digest mismatch"
        )
    if optimizer_context.get("stored_state_usage") != (
        "exact_source_artifact_verification_only"
    ):
        raise RecurrentCounterfactualBranchError(
            "stored source recurrent state usage is unsafe"
        )
    if optimizer_context.get("current_model_state_policy") != (
        "reconstruct_from_trainable_public_history_prefix"
    ):
        raise RecurrentCounterfactualBranchError(
            "current-model recurrent state must be reconstructed from public history"
        )
    state_shape = optimizer_context.get("source_recurrent_state_shape")
    if (
        not isinstance(state_shape, list)
        or len(state_shape) != 3
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in state_shape
        )
    ):
        raise RecurrentCounterfactualBranchError(
            "source recurrent state shape must have three positive dimensions"
        )
    canonical_state = _validated_float32_tensor_tree(
        optimizer_context.get("source_recurrent_state"),
        expected_shape=state_shape,
        field="source recurrent state",
    )
    if not _valid_sha256(optimizer_context.get("source_recurrent_state_sha256")):
        raise RecurrentCounterfactualBranchError(
            "source recurrent state digest is malformed"
        )
    canonical_state_bytes = bytes(
        canonical_state.contiguous().view(uint8).reshape(-1).tolist()
    )
    if optimizer_context.get("source_recurrent_state_sha256") != hashlib.sha256(
        canonical_state_bytes
    ).hexdigest():
        raise RecurrentCounterfactualBranchError(
            "source recurrent state digest does not match its stored values"
        )
    metadata_for_limits = _mapping(row.get("metadata"), field="metadata")
    maximum_focal_transition_count = _positive_int(
        metadata_for_limits.get("horizon_ticks"),
        field="metadata horizon_ticks",
    )

    labels = _mapping(row.get("labels"), field="labels")
    if set(labels) != _BRANCH_LABEL_FIELDS:
        raise RecurrentCounterfactualBranchError("branch labels field set drifted")
    _finite_float(
        labels.get("source_policy_value"),
        field="labels source policy value",
    )
    source_action = _stable_action(
        labels.get("source_requested_action"),
        field="labels.source_requested_action",
    )
    source_behavior_distribution = _mapping(
        labels.get("source_behavior_distribution"),
        field="labels.source_behavior_distribution",
    )
    try:
        validate_public_recurrent_distribution_diagnostics(
            source_behavior_distribution,
            action_mask=mask,
        )
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "source behavior distribution diagnostics are invalid"
        ) from error
    if source_behavior_distribution.get("selected_action") != source_action:
        raise RecurrentCounterfactualBranchError(
            "source behavior distribution selected action drifted"
        )
    outcomes = labels.get("action_outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise RecurrentCounterfactualBranchError("action outcomes cannot be empty")
    baseline = _mapping(labels.get("baseline"), field="labels.baseline")
    if set(baseline) != _COMPACT_BRANCH_BASELINE_FIELDS:
        raise RecurrentCounterfactualBranchError(
            "branch baseline label field set drifted"
        )
    _validate_compact_branch_outcome(
        baseline,
        field="branch baseline",
        maximum_transition_count=maximum_focal_transition_count,
    )
    baseline_first_transition = _mapping(
        baseline.get("first_transition"),
        field="branch baseline first transition",
    )
    if (
        baseline_first_transition.get("requested_action") != source_action
        or baseline_first_transition.get("observation_action_valid") is not True
    ):
        raise RecurrentCounterfactualBranchError(
            "branch baseline did not execute its public-mask-valid source action"
        )
    expected_actions = [action for action in ACTION_NAMES if mask[action] is True]
    observed_actions = [
        _stable_action(
            _mapping(outcome, field="action outcome").get("action"),
            field="action outcome action",
        )
        for outcome in outcomes
    ]
    if observed_actions != expected_actions:
        raise RecurrentCounterfactualBranchError(
            "action outcomes do not cover every valid action exactly once"
        )
    if source_action not in observed_actions:
        raise RecurrentCounterfactualBranchError(
            "source action is absent from action outcomes"
        )
    if labels.get("all_tick_start_mask_valid_actions_evaluated") is not True:
        raise RecurrentCounterfactualBranchError(
            "all-valid-action coverage proof is absent"
        )
    if _positive_int(
        labels.get("valid_action_count"),
        field="labels valid action count",
    ) != len(expected_actions):
        raise RecurrentCounterfactualBranchError(
            "valid action count does not match the public mask"
        )
    if labels.get("baseline_source_action_behavior_match") is not True:
        raise RecurrentCounterfactualBranchError(
            "source-action branch did not reproduce the baseline"
        )
    contract = _mapping(row.get("contract"), field="contract")
    if contract.get("historical_row_training_use") != (
        RECURRENT_COUNTERFACTUAL_TRAINING_USE
    ):
        raise RecurrentCounterfactualBranchError(
            "counterfactual row is not restricted to supervised auxiliary use"
        )
    if contract.get("ppo_ratio_data_use_forbidden") is not True:
        raise RecurrentCounterfactualBranchError(
            "historical counterfactual rows must be forbidden as PPO ratio data"
        )
    if contract.get("current_model_recurrent_state_policy") != (
        "recompute_from_public_history_prefix_never_consume_stale_hidden"
    ):
        raise RecurrentCounterfactualBranchError(
            "counterfactual current-model recurrent-state policy drifted"
        )
    verify_replay = contract.get("exact_replay_required") is True
    for outcome in outcomes:
        parsed = _mapping(outcome, field="action outcome")
        if set(parsed) != _COMPACT_BRANCH_ACTION_FIELDS:
            raise RecurrentCounterfactualBranchError(
                "branch action label field set drifted"
            )
        action = _stable_action(parsed.get("action"), field="action outcome action")
        if parsed.get("natural_requested_action") != source_action:
            raise RecurrentCounterfactualBranchError(
                "action branch natural selection drifted from the source action"
            )
        first_transition = _mapping(
            parsed.get("first_transition"),
            field="action outcome first transition",
        )
        _validate_compact_branch_outcome(
            parsed,
            field=f"branch action {action}",
            maximum_transition_count=maximum_focal_transition_count,
        )
        paired = _mapping(
            parsed.get("paired_vs_baseline"),
            field="branch paired delta",
        )
        _validate_paired_delta_payload(paired, field="branch paired delta")
        parsed_terminal = _mapping(
            parsed.get("focal_terminal"),
            field="branch action focal terminal",
        )
        baseline_terminal = _mapping(
            baseline.get("focal_terminal"),
            field="branch baseline focal terminal",
        )
        expected_paired = {
            "focal_discounted_return_delta": _round(
                float(parsed["focal_discounted_return"])
                - float(baseline["focal_discounted_return"])
            ),
            "focal_terminal_alive_delta": int(parsed_terminal["alive"] is True)
            - int(baseline_terminal["alive"] is True),
            "population_alive_delta": int(parsed["population_alive"])
            - int(baseline["population_alive"]),
            "births_during_horizon_delta": int(parsed["births_during_horizon"])
            - int(baseline["births_during_horizon"]),
            "deaths_during_horizon_delta": int(parsed["deaths_during_horizon"])
            - int(baseline["deaths_during_horizon"]),
        }
        if dict(paired) != expected_paired:
            raise RecurrentCounterfactualBranchError(
                "branch paired delta does not match its baseline"
            )
        if first_transition.get("requested_action") != action:
            raise RecurrentCounterfactualBranchError(
                "action branch did not request its enumerated action"
            )
        if first_transition.get("observation_action_valid") is not True:
            raise RecurrentCounterfactualBranchError(
                "action branch was not valid in the tick-start public mask"
            )
        if _positive_int(
            parsed.get("intervention_count"),
            field="action outcome intervention count",
        ) != 1:
            raise RecurrentCounterfactualBranchError(
                "every action outcome must contain exactly one intervention"
            )
        for field in (
            "heuristic_action_source_count",
            "unsupported_requested_action_count",
            "unexpected_action_source_count",
        ):
            if _nonnegative_int(
                parsed.get(field),
                field=f"action outcome {field}",
            ) != 0:
                raise RecurrentCounterfactualBranchError(
                    "counterfactual continuation used an unsupported action source"
                )
        focal_transition_count = _positive_int(
            parsed.get("focal_transition_count"),
            field="action outcome focal transition count",
        )
        focal_policy_decision_count = _positive_int(
            parsed.get("focal_policy_decision_count"),
            field="action outcome focal policy decision count",
        )
        focal_passive_transition_count = _nonnegative_int(
            parsed.get("focal_passive_transition_count"),
            field="action outcome focal passive transition count",
        )
        if focal_transition_count != (
            focal_policy_decision_count + focal_passive_transition_count
        ):
            raise RecurrentCounterfactualBranchError(
                "focal action and passive transition counts are inconsistent"
            )
        if verify_replay and parsed.get("replay_verified") is not True:
            raise RecurrentCounterfactualBranchError(
                "counterfactual action lacks exact replay verification"
            )
        if not _valid_sha256(parsed.get("behavior_digest")) or not _valid_sha256(
            parsed.get("evidence_digest")
        ):
            raise RecurrentCounterfactualBranchError(
                "counterfactual action digest is malformed"
            )
        if verify_replay and (
            parsed.get("replay_evidence_digest") != parsed.get("evidence_digest")
        ):
            raise RecurrentCounterfactualBranchError(
                "counterfactual replay evidence digest does not match"
            )
    source_outcome = next(
        _mapping(outcome, field="source action outcome")
        for outcome in outcomes
        if _mapping(outcome, field="action outcome").get("action") == source_action
    )
    negative_control_fields = (
        "first_transition",
        "focal_transition_count",
        "focal_policy_decision_count",
        "focal_passive_transition_count",
        "focal_discounted_return",
        "focal_terminal",
        "population_alive",
        "births_during_horizon",
        "deaths_during_horizon",
        "behavior_digest",
    )
    if any(
        source_outcome.get(field) != baseline.get(field)
        for field in negative_control_fields
    ) or dict(
        _mapping(
            source_outcome.get("paired_vs_baseline"),
            field="source action paired delta",
        )
    ) != {
        "focal_discounted_return_delta": 0.0,
        "focal_terminal_alive_delta": 0,
        "population_alive_delta": 0,
        "births_during_horizon_delta": 0,
        "deaths_during_horizon_delta": 0,
    }:
        raise RecurrentCounterfactualBranchError(
            "source-action negative control does not exactly match the baseline"
        )
    if contract.get("focal_return_record_policy") != (
        "all_focal_trajectory_records_including_passive_terminal"
    ):
        raise RecurrentCounterfactualBranchError(
            "counterfactual focal-return record policy drifted"
        )

    metadata = metadata_for_limits
    expected_metadata_keys = (
        _BOUNDARY_SOURCE_METADATA_KEYS
        if schema_version
        == RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION
        else _METADATA_KEYS
    )
    if set(metadata) != expected_metadata_keys:
        raise RecurrentCounterfactualBranchError("metadata field set drifted")
    role = str(metadata.get("seed_role", ""))
    seed = _positive_int(metadata.get("environment_seed"), field="environment_seed")
    _validated_seed_role(role, seed)
    expected_contract = _counterfactual_contract(
        verify_replay=verify_replay,
        seed_role=role,
    )
    if stable_payload_digest(contract) != stable_payload_digest(expected_contract):
        raise RecurrentCounterfactualBranchError(
            "counterfactual branch protocol contract drifted"
        )
    expected_allowed_roles = _counterfactual_seed_role_cohort(role)
    if contract.get("allowed_seed_roles") != list(expected_allowed_roles):
        raise RecurrentCounterfactualBranchError(
            "counterfactual allowed seed-role cohort drifted"
        )
    if metadata.get("private_checkpoint_serialized") is not False:
        raise RecurrentCounterfactualBranchError(
            "private world checkpoint cannot be serialized in a branch row"
        )
    if schema_version == RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION:
        if not _valid_sha256(
            metadata.get("source_pre_boundary_environment_rng_state_sha256")
        ) or not _valid_sha256(
            metadata.get("source_pre_boundary_policy_sampling_state_sha256")
        ):
            raise RecurrentCounterfactualBranchError(
                "boundary-bound source row has malformed pre-boundary RNG evidence"
            )
        if metadata.get("source_boundary_identity_sha256") != stable_payload_digest(
            _source_boundary_identity_payload(metadata)
        ):
            raise RecurrentCounterfactualBranchError(
                "boundary-bound source row identity digest drifted"
            )
    if metadata.get("scenario") not in RECURRENT_COUNTERFACTUAL_SCENARIOS:
        raise RecurrentCounterfactualBranchError("metadata scenario is unsupported")
    branch_tick = _nonnegative_int(
        metadata.get("branch_tick"),
        field="metadata branch_tick",
    )
    focal_agent_id = _positive_int(
        metadata.get("focal_agent_id"),
        field="metadata focal_agent_id",
    )
    _unit_interval(metadata.get("gamma"), field="metadata gamma")
    _nonnegative_int(
        metadata.get("source_decision_index"),
        field="metadata source_decision_index",
    )
    for digest_field in (
        "source_record_digest",
        "source_decision_diagnostics_digest",
        "source_observation_digest",
        "source_action_mask_digest",
    ):
        if not _valid_sha256(metadata.get(digest_field)):
            raise RecurrentCounterfactualBranchError(
                f"metadata {digest_field} is malformed"
            )
    if metadata.get("source_action_mask_digest") != stable_payload_digest(mask):
        raise RecurrentCounterfactualBranchError(
            "metadata source action-mask digest drifted from public context"
        )
    if not _valid_sha256(metadata.get("branch_protocol_digest")):
        raise RecurrentCounterfactualBranchError("branch protocol digest is malformed")
    if metadata.get("source_artifact_digest") != optimizer_context.get(
        "source_artifact_digest"
    ):
        raise RecurrentCounterfactualBranchError(
            "metadata and optimizer source artifact digests disagree"
        )
    if metadata.get("source_model_state_sha256") != optimizer_context.get(
        "source_model_state_sha256"
    ):
        raise RecurrentCounterfactualBranchError(
            "metadata and optimizer source model digests disagree"
        )
    if metadata.get("source_public_history_prefix_sha256") != (
        public_history_prefix_sha256
    ):
        raise RecurrentCounterfactualBranchError(
            "metadata public-history prefix digest mismatch"
        )
    policy_sampling_seed = _optional_sampling_seed(
        metadata.get("policy_sampling_seed"),
        field="metadata policy_sampling_seed",
    )
    expected_action_selection = (
        PUBLIC_RECURRENT_ARGMAX_SELECTION
        if policy_sampling_seed is None
        else PUBLIC_RECURRENT_SAMPLED_SELECTION
    )
    if metadata.get("source_action_selection") != expected_action_selection:
        raise RecurrentCounterfactualBranchError(
            "metadata source action-selection contract drifted"
        )
    source_sampling_state_sha256 = metadata.get("source_sampling_state_sha256")
    if (
        policy_sampling_seed is None
        and source_sampling_state_sha256 is not None
    ) or (
        policy_sampling_seed is not None
        and not _valid_sha256(source_sampling_state_sha256)
    ):
        raise RecurrentCounterfactualBranchError(
            "metadata source sampling-state digest drifted"
        )
    horizon_ticks = _positive_int(
        metadata.get("horizon_ticks"),
        field="metadata horizon_ticks",
    )
    continuation = _mapping(
        metadata.get("continuation_provenance"),
        field="metadata continuation_provenance",
    )
    if set(continuation) != _CONTINUATION_PROVENANCE_KEYS:
        raise RecurrentCounterfactualBranchError(
            "continuation provenance field set drifted"
        )
    continuation_digest = stable_payload_digest(continuation)
    if metadata.get("continuation_provenance_digest") != continuation_digest:
        raise RecurrentCounterfactualBranchError(
            "continuation provenance digest mismatch"
        )
    if continuation.get("continuation_policy_artifact_digest") != (
        metadata.get("source_artifact_digest")
    ):
        raise RecurrentCounterfactualBranchError(
            "continuation policy artifact digest drifted"
        )
    if continuation.get("continuation_policy_model_state_sha256") != (
        metadata.get("source_model_state_sha256")
    ):
        raise RecurrentCounterfactualBranchError(
            "continuation policy model digest drifted"
        )
    continuation_policy_sampling_seed = _optional_sampling_seed(
        continuation.get("continuation_policy_sampling_seed"),
        field="continuation policy sampling seed",
    )
    if continuation_policy_sampling_seed != policy_sampling_seed:
        raise RecurrentCounterfactualBranchError(
            "continuation policy sampling seed drifted"
        )
    if continuation.get("continuation_policy_action_selection") != (
        expected_action_selection
    ):
        raise RecurrentCounterfactualBranchError(
            "continuation policy action-selection contract drifted"
        )
    continuation_horizon_ticks = _positive_int(
        continuation.get("branch_horizon_ticks"),
        field="continuation branch horizon",
    )
    if continuation_horizon_ticks != horizon_ticks:
        raise RecurrentCounterfactualBranchError("continuation branch horizon drifted")
    repeat_count = _positive_int(
        continuation.get("exact_replay_repeat_count_per_action"),
        field="continuation exact replay repeat count",
    )
    expected_repeat_count = 2 if contract.get("exact_replay_required") is True else 1
    if repeat_count != expected_repeat_count:
        raise RecurrentCounterfactualBranchError(
            "continuation exact replay repeat count drifted"
        )
    repeat_indices = continuation.get("repeat_indices")
    if not isinstance(repeat_indices, list) or [
        _nonnegative_int(index, field="continuation repeat index")
        for index in repeat_indices
    ] != list(range(repeat_count)):
        raise RecurrentCounterfactualBranchError("continuation repeat indices drifted")
    repeat_policy_sampling_seeds = continuation.get(
        "repeat_policy_sampling_seeds"
    )
    if not isinstance(repeat_policy_sampling_seeds, list) or [
        _optional_sampling_seed(
            seed,
            field="continuation repeat policy sampling seed",
        )
        for seed in repeat_policy_sampling_seeds
    ] != [policy_sampling_seed for _ in range(repeat_count)]:
        raise RecurrentCounterfactualBranchError(
            "continuation repeat policy-sampling seeds drifted"
        )
    checkpoint_sampling_digest = metadata.get("source_sampling_state_sha256")
    if continuation.get("continuation_checkpoint_sampling_state_sha256") != (
        checkpoint_sampling_digest
    ):
        raise RecurrentCounterfactualBranchError(
            "continuation checkpoint sampling-state digest drifted"
        )
    if continuation.get("repeat_checkpoint_sampling_state_sha256") != [
        checkpoint_sampling_digest for _ in range(repeat_count)
    ]:
        raise RecurrentCounterfactualBranchError(
            "continuation repeat sampling-state provenance drifted"
        )
    for field in (
        "common_checkpoint_across_actions",
        "source_checkpoint_reused_by_deepcopy",
    ):
        if continuation.get(field) is not True:
            raise RecurrentCounterfactualBranchError(
                f"continuation provenance field {field!r} must be true"
            )
    if (
        metadata.get("baseline_behavior_digest") != baseline.get("behavior_digest")
        or metadata.get("baseline_evidence_digest") != baseline.get("evidence_digest")
    ):
        raise RecurrentCounterfactualBranchError(
            "metadata baseline evidence digest drifted from labels"
        )
    expected_branch_identity = stable_payload_digest(
        {
            "protocol_digest": metadata.get("branch_protocol_digest"),
            "artifact_digest": metadata.get("source_artifact_digest"),
            "model_state_sha256": metadata.get("source_model_state_sha256"),
            "seed_role": role,
            "environment_seed": seed,
            "policy_sampling_seed": policy_sampling_seed,
            "scenario": metadata.get("scenario"),
            "branch_tick": branch_tick,
            "horizon_ticks": horizon_ticks,
            "exact_replay_repeat_count_per_action": repeat_count,
            "focal_agent_id": focal_agent_id,
            "source_record_digest": metadata.get("source_record_digest"),
            "source_recurrent_state_sha256": optimizer_context.get(
                "source_recurrent_state_sha256"
            ),
            "source_sampling_state_sha256": source_sampling_state_sha256,
            "source_public_history_prefix_sha256": public_history_prefix_sha256,
            "continuation_provenance_digest": continuation_digest,
        }
    )
    if metadata.get("branch_identity_digest") != expected_branch_identity:
        raise RecurrentCounterfactualBranchError("branch identity digest drifted")
    components = _mapping(row.get("component_digests"), field="component_digests")
    expected_components = {
        "contract": stable_payload_digest(contract),
        "trainable_public_context": stable_payload_digest(context),
        "optimizer_context": stable_payload_digest(optimizer_context),
        "labels": stable_payload_digest(labels),
        "metadata": stable_payload_digest(metadata),
    }
    if dict(components) != expected_components:
        raise RecurrentCounterfactualBranchError("component digest mismatch")
    if metadata.get("branch_protocol_digest") != expected_components["contract"]:
        raise RecurrentCounterfactualBranchError(
            "metadata branch protocol digest does not match the contract"
        )
    exact_digest = row.get("exact_digest")
    without_exact = dict(row)
    without_exact.pop("exact_digest", None)
    if exact_digest != stable_payload_digest(without_exact):
        raise RecurrentCounterfactualBranchError("branch row exact digest mismatch")
    for flag in (
        "training_ran",
        "training_artifact_created",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "promotion_authorized",
    ):
        if row.get(flag) is not False:
            raise RecurrentCounterfactualBranchError(
                f"diagnostics-only lifecycle flag {flag!r} must remain false"
            )


def validate_recurrent_counterfactual_aggregate_row(
    row: Mapping[str, object],
) -> None:
    """Validate compact multi-tape evidence without treating it as PPO data."""

    if not isinstance(row, Mapping):
        raise RecurrentCounterfactualBranchError(
            "counterfactual aggregate row must be a mapping"
        )
    expected_row_fields = {
        "schema_version",
        "target",
        "trainable_public_context",
        "optimizer_context",
        "source_identity",
        "source_behavior",
        "tape_contract",
        "tape_provenance",
        "aggregate",
        "component_digests",
        "training_ran",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "promotion_authorized",
        "exact_digest",
    }
    if set(row) != expected_row_fields:
        raise RecurrentCounterfactualBranchError(
            "counterfactual aggregate field set drifted"
        )
    if row.get("schema_version") != RECURRENT_COUNTERFACTUAL_AGGREGATE_SCHEMA_VERSION:
        raise RecurrentCounterfactualBranchError(
            "counterfactual aggregate schema version drifted"
        )
    target = _mapping(row.get("target"), field="aggregate target")
    if set(target) != {
        "kind",
        "branch_tick",
        "horizon_ticks",
        "target_world_tick",
        "absolute_terminal_target_world_tick",
        "is_absolute_terminal_target",
    }:
        raise RecurrentCounterfactualBranchError("aggregate target field set drifted")
    target_kind = target.get("kind")
    if target_kind not in {"relative_horizon", "absolute_terminal_world_tick"}:
        raise RecurrentCounterfactualBranchError("aggregate target kind drifted")
    branch_tick = _nonnegative_int(target.get("branch_tick"), field="branch tick")
    horizon_ticks = _positive_int(target.get("horizon_ticks"), field="horizon")
    target_world_tick = _positive_int(
        target.get("target_world_tick"),
        field="aggregate target world tick",
    )
    if target_world_tick != branch_tick + horizon_ticks:
        raise RecurrentCounterfactualBranchError("aggregate target tick drifted")
    terminal_target = target.get("absolute_terminal_target_world_tick")
    if terminal_target is not None:
        terminal_target = _positive_int(terminal_target, field="terminal target tick")
        if target_world_tick > terminal_target:
            raise RecurrentCounterfactualBranchError(
                "aggregate target cannot extend beyond its absolute terminal tick"
            )
    is_terminal = target.get("is_absolute_terminal_target")
    if type(is_terminal) is not bool or is_terminal != (
        target_kind == "absolute_terminal_world_tick"
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate absolute terminal flag drifted"
        )
    if is_terminal and target_world_tick != terminal_target:
        raise RecurrentCounterfactualBranchError(
            "absolute terminal aggregate does not end at its target tick"
        )

    context = _mapping(
        row.get("trainable_public_context"),
        field="aggregate trainable public context",
    )
    if set(context) != _TRAINABLE_PUBLIC_CONTEXT_KEYS:
        raise RecurrentCounterfactualBranchError(
            "aggregate trainable public context field set drifted"
        )
    forbidden = _forbidden_key_paths(context)
    if forbidden:
        raise RecurrentCounterfactualBranchError(
            f"aggregate trainable public context contains forbidden keys: "
            f"{forbidden[:8]}"
        )
    prefix = _mapping(
        context.get("public_history_prefix"),
        field="aggregate public history prefix",
    )
    try:
        validate_public_recurrent_history_prefix(prefix)
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "aggregate public recurrent history prefix is invalid"
        ) from error
    observation = _mapping(
        context.get("current_public_observation"),
        field="aggregate public observation",
    )
    _validate_current_public_observation(observation)
    mask = _complete_action_mask(context.get("current_public_action_mask"))
    feedback = _mapping(
        context.get("previous_public_feedback"),
        field="aggregate previous public feedback",
    )
    _validate_previous_public_feedback(feedback)
    _validate_feedback_history_alignment(
        prefix,
        feedback,
        field="aggregate previous public feedback",
    )

    optimizer = _mapping(
        row.get("optimizer_context"),
        field="aggregate optimizer context",
    )
    if set(optimizer) != _OPTIMIZER_CONTEXT_KEYS:
        raise RecurrentCounterfactualBranchError(
            "aggregate optimizer context field set drifted"
        )
    if (
        optimizer.get("derived_from_public_history") is not True
        or optimizer.get("source_artifact_match_required") is not True
        or optimizer.get("runtime_environment_input") is not False
        or optimizer.get("stored_state_usage")
        != "exact_source_artifact_verification_only"
        or optimizer.get("current_model_state_policy")
        != "reconstruct_from_trainable_public_history_prefix"
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate optimizer context safety contract drifted"
        )
    if optimizer.get("source_public_history_prefix_sha256") != (
        stable_payload_digest(prefix)
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate optimizer public-history digest drifted"
        )
    _nonempty_string(
        optimizer.get("source_artifact_digest"),
        field="aggregate optimizer source artifact digest",
    )
    if not _valid_sha256(optimizer.get("source_model_state_sha256")):
        raise RecurrentCounterfactualBranchError(
            "aggregate optimizer source model-state digest is malformed"
        )
    optimizer_state_shape = optimizer.get("source_recurrent_state_shape")
    if (
        not isinstance(optimizer_state_shape, list)
        or len(optimizer_state_shape) != 3
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in optimizer_state_shape
        )
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate optimizer recurrent state shape drifted"
        )
    optimizer_state = _validated_float32_tensor_tree(
        optimizer.get("source_recurrent_state"),
        expected_shape=optimizer_state_shape,
        field="aggregate optimizer recurrent state",
    )
    optimizer_state_sha256 = hashlib.sha256(
        bytes(optimizer_state.contiguous().view(uint8).reshape(-1).tolist())
    ).hexdigest()
    if optimizer.get("source_recurrent_state_sha256") != optimizer_state_sha256:
        raise RecurrentCounterfactualBranchError(
            "aggregate optimizer recurrent state digest drifted"
        )

    source_identity = _mapping(
        row.get("source_identity"),
        field="aggregate source identity",
    )
    if set(source_identity) != {
        "source_branch_row_exact_digest",
        "source_model_state_sha256",
        "source_artifact_digest",
        "source_environment_seed",
        "source_policy_sampling_seed",
        "source_branch_selection_seed",
        "source_checkpoint_identity_sha256",
        "source_pre_boundary_environment_rng_state_sha256",
        "source_pre_boundary_policy_sampling_state_sha256",
        "source_boundary_identity_sha256",
        "trainable_public_context_sha256",
    }:
        raise RecurrentCounterfactualBranchError(
            "aggregate source identity field set drifted"
        )
    if not _valid_sha256(source_identity.get("source_model_state_sha256")):
        raise RecurrentCounterfactualBranchError(
            "aggregate source model-state digest is malformed"
        )
    if not _valid_sha256(source_identity.get("source_branch_row_exact_digest")):
        raise RecurrentCounterfactualBranchError(
            "aggregate source branch-row digest is malformed"
        )
    _nonempty_string(
        source_identity.get("source_artifact_digest"),
        field="aggregate source artifact digest",
    )
    _required_sampling_seed(
        source_identity.get("source_policy_sampling_seed"),
        field="aggregate source policy sampling seed",
    )
    _required_sampling_seed(
        source_identity.get("source_environment_seed"),
        field="aggregate source environment seed",
    )
    _required_sampling_seed(
        source_identity.get("source_branch_selection_seed"),
        field="aggregate source branch-selection seed",
    )
    if (
        source_identity.get("source_model_state_sha256")
        != optimizer.get("source_model_state_sha256")
        or source_identity.get("source_artifact_digest")
        != optimizer.get("source_artifact_digest")
        or source_identity.get("trainable_public_context_sha256")
        != stable_payload_digest(context)
        or not _valid_sha256(source_identity.get("source_checkpoint_identity_sha256"))
        or not _valid_sha256(
            source_identity.get(
                "source_pre_boundary_environment_rng_state_sha256"
            )
        )
        or not _valid_sha256(
            source_identity.get("source_pre_boundary_policy_sampling_state_sha256")
        )
        or not _valid_sha256(source_identity.get("source_boundary_identity_sha256"))
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate source identity does not match its exact public source"
        )

    source_behavior = _mapping(
        row.get("source_behavior"),
        field="aggregate source behavior",
    )
    if set(source_behavior) != {
        "source_requested_action",
        "current_public_action_mask",
        "source_behavior_distribution",
    }:
        raise RecurrentCounterfactualBranchError(
            "aggregate source behavior field set drifted"
        )
    source_action = _stable_action(
        source_behavior.get("source_requested_action"),
        field="aggregate source action",
    )
    if _complete_action_mask(source_behavior.get("current_public_action_mask")) != mask:
        raise RecurrentCounterfactualBranchError(
            "aggregate source action mask drifted from public context"
        )
    behavior_distribution = _mapping(
        source_behavior.get("source_behavior_distribution"),
        field="aggregate source behavior distribution",
    )
    try:
        validate_public_recurrent_distribution_diagnostics(
            behavior_distribution,
            action_mask=mask,
        )
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "aggregate source behavior distribution is invalid"
        ) from error
    if behavior_distribution.get("selected_action") != source_action:
        raise RecurrentCounterfactualBranchError(
            "aggregate source selected action drifted"
        )

    tape_contract = _mapping(
        row.get("tape_contract"),
        field="aggregate tape contract",
    )
    expected_tape_contract_fields = {
        "tape_identity",
        "tape_count",
        "environment_seed_namespace",
        "policy_seed_namespace",
        "excluded_source_seeds",
        "same_exact_source_checkpoint",
        "continuation_rng_retape_boundary",
        "fixed_source_prefix_through_focal_natural_draw",
        "horizon_includes_branch_tick",
        "same_tick_post_focal_consequences_governed_by_tape",
        "policy_sampling_generator_device_type",
        "common_boundary_rng_state_across_actions_within_tape",
        "paired_baseline_and_forced_actions_share_tape",
        "exact_replay_required_for_baseline_and_actions",
        "event_aligned_common_random_numbers",
        "sequential_rng_event_reassignment_after_divergence_possible",
        "private_world_checkpoint_serialized",
    }
    if set(tape_contract) != expected_tape_contract_fields:
        raise RecurrentCounterfactualBranchError(
            "aggregate tape contract field set drifted"
        )
    tape_identity = _nonempty_string(
        tape_contract.get("tape_identity"),
        field="aggregate tape identity",
    )
    tape_count = _positive_int(tape_contract.get("tape_count"), field="tape count")
    if (
        tape_contract.get("environment_seed_namespace")
        != RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE
        or tape_contract.get("policy_seed_namespace")
        != RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate continuation seed namespace drifted"
        )
    if (
        tape_contract.get("continuation_rng_retape_boundary")
        != RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate continuation RNG retape boundary drifted"
        )
    if tape_contract.get("policy_sampling_generator_device_type") != "cpu":
        raise RecurrentCounterfactualBranchError(
            "aggregate continuation policy sampling device drifted"
        )
    for field in (
        "same_exact_source_checkpoint",
        "fixed_source_prefix_through_focal_natural_draw",
        "horizon_includes_branch_tick",
        "same_tick_post_focal_consequences_governed_by_tape",
        "common_boundary_rng_state_across_actions_within_tape",
        "paired_baseline_and_forced_actions_share_tape",
        "exact_replay_required_for_baseline_and_actions",
        "sequential_rng_event_reassignment_after_divergence_possible",
    ):
        if tape_contract.get(field) is not True:
            raise RecurrentCounterfactualBranchError(
                f"aggregate tape contract field {field!r} must be true"
            )
    if (
        tape_contract.get("event_aligned_common_random_numbers") is not False
        or tape_contract.get("private_world_checkpoint_serialized") is not False
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate tape contract overclaims RNG alignment or serializes private state"
        )
    excluded_raw = tape_contract.get("excluded_source_seeds")
    if not isinstance(excluded_raw, list):
        raise RecurrentCounterfactualBranchError(
            "aggregate excluded source seeds must be a list"
        )
    excluded_seeds = {
        _required_sampling_seed(seed, field="excluded source seed")
        for seed in excluded_raw
    }
    expected_excluded_seeds = [
        source_identity["source_environment_seed"],
        source_identity["source_policy_sampling_seed"],
        source_identity["source_branch_selection_seed"],
    ]
    if (
        len(excluded_seeds) != len(excluded_raw)
        or excluded_raw != expected_excluded_seeds
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate excluded source seeds must be unique, ordered, and "
            "source-complete"
        )

    tape_provenance = row.get("tape_provenance")
    if not isinstance(tape_provenance, list) or len(tape_provenance) != tape_count:
        raise RecurrentCounterfactualBranchError(
            "aggregate tape provenance count drifted"
        )
    expected_actions = [action for action in ACTION_NAMES if mask[action] is True]
    observed_tape_seeds: set[int] = set()
    source_pre_boundary_digests: tuple[object, object] | None = None
    observed_post_boundary_environment_digests: set[object] = set()
    observed_post_boundary_policy_digests: set[object] = set()
    for expected_index, raw_tape in enumerate(tape_provenance):
        tape = _mapping(raw_tape, field="aggregate tape provenance")
        if set(tape) != _MULTI_TAPE_PROVENANCE_FIELDS:
            raise RecurrentCounterfactualBranchError(
                "aggregate tape provenance field set drifted"
            )
        if _nonnegative_int(
            tape.get("tape_index"),
            field="aggregate tape index",
        ) != expected_index:
            raise RecurrentCounterfactualBranchError(
                "aggregate tape indices must be contiguous and ordered"
            )
        environment_identity = f"{tape_identity}:environment:{expected_index}"
        policy_identity = f"{tape_identity}:policy:{expected_index}"
        if (
            tape.get("environment_sampling_identity") != environment_identity
            or tape.get("policy_sampling_identity") != policy_identity
        ):
            raise RecurrentCounterfactualBranchError(
                "aggregate tape seed identity drifted"
            )
        environment_seed = _required_sampling_seed(
            tape.get("environment_sampling_seed"),
            field="aggregate environment tape seed",
        )
        policy_seed = _required_sampling_seed(
            tape.get("policy_sampling_seed"),
            field="aggregate policy tape seed",
        )
        if environment_seed != derive_recurrent_counterfactual_tape_seed(
            namespace=(
                RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE
            ),
            identity=environment_identity,
        ) or policy_seed != derive_recurrent_counterfactual_tape_seed(
            namespace=RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE,
            identity=policy_identity,
        ):
            raise RecurrentCounterfactualBranchError(
                "aggregate tape seed does not match its namespaced identity"
            )
        if (
            environment_seed in excluded_seeds
            or policy_seed in excluded_seeds
            or environment_seed in observed_tape_seeds
            or policy_seed in observed_tape_seeds
            or environment_seed == policy_seed
        ):
            raise RecurrentCounterfactualBranchError(
                "aggregate tape seeds are not independent and source-disjoint"
            )
        observed_tape_seeds.update((environment_seed, policy_seed))
        boundary_provenance = _decision_boundary_provenance(tape)
        current_pre_boundary_digests = (
            boundary_provenance["pre_boundary_environment_rng_state_sha256"],
            boundary_provenance["pre_boundary_policy_sampling_state_sha256"],
        )
        if current_pre_boundary_digests != (
            source_identity[
                "source_pre_boundary_environment_rng_state_sha256"
            ],
            source_identity["source_pre_boundary_policy_sampling_state_sha256"],
        ):
            raise RecurrentCounterfactualBranchError(
                "aggregate tape pre-boundary RNG state drifted from its "
                "boundary-bound source row"
            )
        if source_pre_boundary_digests is None:
            source_pre_boundary_digests = current_pre_boundary_digests
        elif current_pre_boundary_digests != source_pre_boundary_digests:
            raise RecurrentCounterfactualBranchError(
                "aggregate tapes did not preserve one fixed source prefix"
            )
        post_environment_digest = boundary_provenance[
            "post_boundary_environment_rng_state_sha256"
        ]
        post_policy_digest = boundary_provenance[
            "post_boundary_policy_sampling_state_sha256"
        ]
        if post_environment_digest != _seeded_environment_rng_state_sha256(
            environment_seed
        ) or post_policy_digest != _seeded_policy_sampling_state_sha256(policy_seed):
            raise RecurrentCounterfactualBranchError(
                "aggregate post-boundary RNG state does not match its tape seed"
            )
        if (
            post_environment_digest in observed_post_boundary_environment_digests
            or post_policy_digest in observed_post_boundary_policy_digests
        ):
            raise RecurrentCounterfactualBranchError(
                "aggregate continuation tapes reused post-boundary RNG state"
            )
        observed_post_boundary_environment_digests.add(post_environment_digest)
        observed_post_boundary_policy_digests.add(post_policy_digest)
        baseline = _mapping(tape.get("baseline"), field="aggregate tape baseline")
        _validate_compact_multi_tape_outcome(
            baseline,
            action=None,
            baseline=None,
            boundary_provenance=boundary_provenance,
            maximum_transition_count=horizon_ticks,
        )
        outcomes = _action_outcome_sequence(tape)
        observed_actions = [
            _stable_action(
                _mapping(outcome, field="aggregate tape action").get("action"),
                field="aggregate tape action",
            )
            for outcome in outcomes
        ]
        if observed_actions != expected_actions:
            raise RecurrentCounterfactualBranchError(
                "aggregate tape action coverage drifted"
            )
        tape_natural_action = _stable_action(
            baseline.get("natural_requested_action"),
            field="aggregate baseline natural action",
        )
        if tape_natural_action != source_action:
            raise RecurrentCounterfactualBranchError(
                "aggregate tape natural action drifted from source behavior"
            )
        for action, outcome in zip(expected_actions, outcomes, strict=True):
            parsed = _mapping(outcome, field="aggregate tape action")
            if parsed.get("natural_requested_action") != tape_natural_action:
                raise RecurrentCounterfactualBranchError(
                    "aggregate forced action did not share its tape baseline draw"
                )
            _validate_compact_multi_tape_outcome(
                parsed,
                action=action,
                baseline=baseline,
                boundary_provenance=boundary_provenance,
                maximum_transition_count=horizon_ticks,
            )
            if action == source_action:
                negative_control_fields = (
                    "natural_requested_action",
                    "focal_discounted_return",
                    "focal_terminal_alive",
                    "population_alive",
                    "births_during_horizon",
                    "deaths_during_horizon",
                    "first_transition",
                    "unsupported_requested_action_count",
                    "heuristic_action_source_count",
                    "unexpected_action_source_count",
                    "behavior_digest",
                )
                paired = _mapping(
                    parsed.get("paired_vs_baseline"),
                    field="aggregate source-action paired delta",
                )
                if any(
                    parsed.get(field) != baseline.get(field)
                    for field in negative_control_fields
                ) or dict(paired) != {
                    "focal_discounted_return_delta": 0.0,
                    "focal_terminal_alive_delta": 0,
                    "population_alive_delta": 0,
                    "births_during_horizon_delta": 0,
                    "deaths_during_horizon_delta": 0,
                }:
                    raise RecurrentCounterfactualBranchError(
                        "aggregate source-action negative control did not "
                        "exactly reproduce its baseline"
                    )

    aggregate = _mapping(row.get("aggregate"), field="aggregate statistics")
    _validate_multi_tape_aggregate_payload(
        aggregate,
        tape_count=tape_count,
        expected_actions=expected_actions,
    )
    uncertainty_penalty = _finite_float(
        aggregate.get("uncertainty_penalty"),
        field="aggregate uncertainty penalty",
    )
    if uncertainty_penalty < 0.0:
        raise RecurrentCounterfactualBranchError(
            "aggregate uncertainty penalty must be non-negative"
        )
    expected_aggregate = _aggregate_multi_tape_outcomes(
        tape_provenance,
        uncertainty_penalty=uncertainty_penalty,
    )
    if dict(aggregate) != expected_aggregate:
        raise RecurrentCounterfactualBranchError(
            "aggregate statistics do not match tape evidence"
        )

    components = _mapping(
        row.get("component_digests"),
        field="aggregate component digests",
    )
    expected_components = {
        "target": stable_payload_digest(target),
        "trainable_public_context": stable_payload_digest(context),
        "optimizer_context": stable_payload_digest(optimizer),
        "source_identity": stable_payload_digest(source_identity),
        "source_behavior": stable_payload_digest(source_behavior),
        "tape_contract": stable_payload_digest(tape_contract),
        "tape_provenance": stable_payload_digest(tape_provenance),
        "aggregate": stable_payload_digest(aggregate),
    }
    if dict(components) != expected_components:
        raise RecurrentCounterfactualBranchError("aggregate component digest mismatch")
    without_exact = dict(row)
    observed_exact = without_exact.pop("exact_digest", None)
    if observed_exact != stable_payload_digest(without_exact):
        raise RecurrentCounterfactualBranchError("aggregate exact digest mismatch")
    for flag in (
        "training_ran",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "promotion_authorized",
    ):
        if row.get(flag) is not False:
            raise RecurrentCounterfactualBranchError(
                f"aggregate lifecycle flag {flag!r} must remain false"
            )


def _validate_compact_multi_tape_outcome(
    outcome: Mapping[str, object],
    *,
    action: str | None,
    baseline: Mapping[str, object] | None,
    boundary_provenance: Mapping[str, object],
    maximum_transition_count: int,
) -> None:
    expected_fields = (
        _COMPACT_MULTI_TAPE_BASELINE_FIELDS
        if action is None
        else _COMPACT_MULTI_TAPE_ACTION_FIELDS
    )
    if set(outcome) != expected_fields:
        raise RecurrentCounterfactualBranchError(
            "aggregate compact outcome field set drifted"
        )
    focal_discounted_return = _finite_float(
        outcome.get("focal_discounted_return"),
        field="aggregate outcome focal_discounted_return",
    )
    _validate_feasible_discounted_return(
        focal_discounted_return,
        maximum_transition_count=maximum_transition_count,
        field="aggregate outcome focal_discounted_return",
    )
    population_alive = _nonnegative_int(
        outcome.get("population_alive"),
        field="aggregate outcome population_alive",
    )
    births_during_horizon = _nonnegative_int(
        outcome.get("births_during_horizon"),
        field="aggregate outcome births_during_horizon",
    )
    deaths_during_horizon = _nonnegative_int(
        outcome.get("deaths_during_horizon"),
        field="aggregate outcome deaths_during_horizon",
    )
    focal_terminal_alive = outcome.get("focal_terminal_alive")
    if type(focal_terminal_alive) is not bool:
        raise RecurrentCounterfactualBranchError(
            "aggregate focal terminal alive must be an exact boolean"
        )
    if focal_terminal_alive and population_alive == 0:
        raise RecurrentCounterfactualBranchError(
            "aggregate outcome cannot report a living focal agent in an "
            "empty population"
        )
    if not focal_terminal_alive and deaths_during_horizon == 0:
        raise RecurrentCounterfactualBranchError(
            "aggregate terminal-dead focal outcome requires at least one "
            "horizon death"
        )
    for field in (
        "unsupported_requested_action_count",
        "heuristic_action_source_count",
        "unexpected_action_source_count",
    ):
        if _nonnegative_int(
            outcome.get(field),
            field=f"aggregate outcome {field}",
        ) != 0:
            raise RecurrentCounterfactualBranchError(
                "aggregate continuation contains an unsupported action source"
            )
    if (
        outcome.get("replay_verified") is not True
        or outcome.get("replay_evidence_digest") != outcome.get("evidence_digest")
        or not _valid_sha256(outcome.get("behavior_digest"))
        or not _valid_sha256(outcome.get("evidence_digest"))
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate continuation lacks exact replay evidence"
        )
    if _decision_boundary_provenance(outcome) != dict(boundary_provenance):
        raise RecurrentCounterfactualBranchError(
            "aggregate continuation did not share its tape boundary RNG state"
        )
    first_transition = _mapping(
        outcome.get("first_transition"),
        field="aggregate first transition",
    )
    _validate_compact_first_transition(first_transition)
    if first_transition.get("reproduced") is True and births_during_horizon == 0:
        raise RecurrentCounterfactualBranchError(
            "aggregate first-transition reproduction requires at least one "
            "horizon birth"
        )
    if first_transition.get("died") is True and focal_terminal_alive:
        raise RecurrentCounterfactualBranchError(
            "aggregate outcome cannot revive a focal agent after "
            "first-transition death"
        )
    if first_transition.get("died") is True and deaths_during_horizon == 0:
        raise RecurrentCounterfactualBranchError(
            "aggregate first-transition death requires at least one horizon death"
        )
    if first_transition.get("observation_action_valid") is not True:
        raise RecurrentCounterfactualBranchError(
            "aggregate first transition was not public-mask valid"
        )
    if action is None:
        natural_action = _stable_action(
            outcome.get("natural_requested_action"),
            field="aggregate baseline natural action",
        )
        if first_transition.get("requested_action") != natural_action:
            raise RecurrentCounterfactualBranchError(
                "aggregate baseline transition drifted from its natural action"
            )
        return
    if (
        outcome.get("action") != action
        or first_transition.get("requested_action") != action
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate forced action identity drifted"
        )
    if baseline is None:
        raise AssertionError("action aggregate validation requires a baseline")
    paired = _mapping(
        outcome.get("paired_vs_baseline"),
        field="aggregate paired delta",
    )
    _validate_paired_delta_payload(paired, field="aggregate paired delta")
    expected_paired = {
        "focal_discounted_return_delta": _round(
            float(outcome["focal_discounted_return"])
            - float(baseline["focal_discounted_return"])
        ),
        "focal_terminal_alive_delta": int(outcome["focal_terminal_alive"] is True)
        - int(baseline["focal_terminal_alive"] is True),
        "population_alive_delta": int(outcome["population_alive"])
        - int(baseline["population_alive"]),
        "births_during_horizon_delta": int(outcome["births_during_horizon"])
        - int(baseline["births_during_horizon"]),
        "deaths_during_horizon_delta": int(outcome["deaths_during_horizon"])
        - int(baseline["deaths_during_horizon"]),
    }
    if dict(paired) != expected_paired:
        raise RecurrentCounterfactualBranchError(
            "aggregate paired delta does not match its tape baseline"
        )


def _validate_paired_delta_payload(
    paired: Mapping[str, object],
    *,
    field: str,
) -> None:
    if set(paired) != set(_PAIRED_DELTA_FIELDS):
        raise RecurrentCounterfactualBranchError(
            f"{field} field set drifted"
        )
    _finite_float(
        paired.get("focal_discounted_return_delta"),
        field=f"{field} focal discounted return delta",
    )
    for count_field in _PAIRED_DELTA_FIELDS[1:]:
        value = paired.get(count_field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RecurrentCounterfactualBranchError(
                f"{field} {count_field} must be an exact integer"
            )


def _validate_multi_tape_aggregate_payload(
    aggregate: Mapping[str, object],
    *,
    tape_count: int,
    expected_actions: Sequence[str],
) -> None:
    if set(aggregate) != _MULTI_TAPE_AGGREGATE_FIELDS:
        raise RecurrentCounterfactualBranchError(
            "aggregate statistics field set drifted"
        )
    if _positive_int(
        aggregate.get("tape_count"),
        field="aggregate statistics tape count",
    ) != tape_count:
        raise RecurrentCounterfactualBranchError(
            "aggregate statistics tape count drifted"
        )
    _finite_float(
        aggregate.get("uncertainty_penalty"),
        field="aggregate statistics uncertainty penalty",
    )
    if aggregate.get("uncertainty_penalized_score_policy") != (
        "paired_focal_discounted_return_delta_mean_minus_coefficient_times_standard_error"
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate uncertainty-penalized score policy drifted"
        )
    baseline_statistics = _mapping(
        aggregate.get("baseline_outcome_statistics"),
        field="aggregate baseline outcome statistics",
    )
    if set(baseline_statistics) != _AGGREGATE_METRIC_FIELDS:
        raise RecurrentCounterfactualBranchError(
            "aggregate baseline metric field set drifted"
        )
    for metric in _AGGREGATE_METRIC_FIELDS:
        _validate_sample_statistics_payload(
            _mapping(
                baseline_statistics.get(metric),
                field=f"aggregate baseline {metric} statistics",
            ),
            field=f"aggregate baseline {metric} statistics",
        )
    _finite_float(
        aggregate.get("baseline_focal_survival_probability"),
        field="aggregate baseline focal survival probability",
    )
    action_outcomes = aggregate.get("action_outcomes")
    if not isinstance(action_outcomes, list):
        raise RecurrentCounterfactualBranchError(
            "aggregate action outcomes must be a list"
        )
    observed_actions: list[str] = []
    for raw_outcome in action_outcomes:
        outcome = _mapping(raw_outcome, field="aggregate action statistics")
        if set(outcome) != _MULTI_TAPE_ACTION_AGGREGATE_FIELDS:
            raise RecurrentCounterfactualBranchError(
                "aggregate action statistics field set drifted"
            )
        action = _stable_action(
            outcome.get("action"),
            field="aggregate action statistics action",
        )
        observed_actions.append(action)
        if _positive_int(
            outcome.get("tape_count"),
            field="aggregate action statistics tape count",
        ) != tape_count:
            raise RecurrentCounterfactualBranchError(
                "aggregate action statistics tape count drifted"
            )
        outcome_statistics = _mapping(
            outcome.get("outcome_statistics"),
            field="aggregate action outcome statistics",
        )
        if set(outcome_statistics) != _AGGREGATE_METRIC_FIELDS:
            raise RecurrentCounterfactualBranchError(
                "aggregate action outcome metric field set drifted"
            )
        for metric in _AGGREGATE_METRIC_FIELDS:
            _validate_sample_statistics_payload(
                _mapping(
                    outcome_statistics.get(metric),
                    field=f"aggregate action {action} {metric} statistics",
                ),
                field=f"aggregate action {action} {metric} statistics",
            )
        paired_statistics = _mapping(
            outcome.get("paired_delta_statistics"),
            field="aggregate paired-delta statistics",
        )
        if set(paired_statistics) != set(_PAIRED_DELTA_FIELDS):
            raise RecurrentCounterfactualBranchError(
                "aggregate paired-delta statistic field set drifted"
            )
        for metric in _PAIRED_DELTA_FIELDS:
            _validate_sample_statistics_payload(
                _mapping(
                    paired_statistics.get(metric),
                    field=f"aggregate paired-delta {metric} statistics",
                ),
                field=f"aggregate paired-delta {metric} statistics",
            )
        _finite_float(
            outcome.get("focal_survival_probability"),
            field=f"aggregate action {action} focal survival probability",
        )
        _finite_float(
            outcome.get("uncertainty_penalized_score"),
            field=f"aggregate action {action} uncertainty-penalized score",
        )
    if observed_actions != list(expected_actions):
        raise RecurrentCounterfactualBranchError(
            "aggregate action statistics do not cover valid actions in stable order"
        )


def _validate_sample_statistics_payload(
    statistics: Mapping[str, object],
    *,
    field: str,
) -> None:
    if set(statistics) != _SAMPLE_STATISTIC_FIELDS:
        raise RecurrentCounterfactualBranchError(
            f"{field} field set drifted"
        )
    for statistic in _SAMPLE_STATISTIC_FIELDS:
        _finite_float(
            statistics.get(statistic),
            field=f"{field} {statistic}",
        )


def _validate_compact_first_transition(
    first_transition: Mapping[str, object],
) -> None:
    if set(first_transition) != _COMPACT_FIRST_TRANSITION_FIELDS:
        raise RecurrentCounterfactualBranchError(
            "aggregate first transition field set drifted"
        )
    requested_action = _stable_action(
        first_transition.get("requested_action"),
        field="aggregate first transition requested action",
    )
    resolved_action = _stable_action(
        first_transition.get("resolved_action"),
        field="aggregate first transition resolved action",
    )
    boolean_fields = (
        "observation_action_valid",
        "resolution_action_valid",
        "same_tick_resolution_mask_drift",
        "expected_same_tick_occupancy_drift",
        "moved",
        "reproduced",
        "died",
        "after_alive",
    )
    for field in boolean_fields:
        if type(first_transition.get(field)) is not bool:
            raise RecurrentCounterfactualBranchError(
                f"aggregate first transition {field!r} must be an exact boolean"
            )
    observation_valid = first_transition["observation_action_valid"]
    resolution_valid = first_transition["resolution_action_valid"]
    same_tick_drift = first_transition["same_tick_resolution_mask_drift"]
    invalid_reason = first_transition.get("invalid_reason")
    if invalid_reason is not None and (
        not isinstance(invalid_reason, str) or not invalid_reason.strip()
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate first transition invalid reason is malformed"
        )
    if same_tick_drift != (observation_valid and not resolution_valid):
        raise RecurrentCounterfactualBranchError(
            "aggregate first transition resolution-mask drift is incoherent"
        )
    expected_occupancy_drift = (
        requested_action.startswith("move_")
        and observation_valid
        and not resolution_valid
        and invalid_reason == "not_in_resolution_action_mask"
    )
    if (
        first_transition["expected_same_tick_occupancy_drift"]
        != expected_occupancy_drift
    ):
        raise RecurrentCounterfactualBranchError(
            "aggregate first transition occupancy-drift classification is incoherent"
        )
    if resolution_valid:
        if invalid_reason is not None:
            raise RecurrentCounterfactualBranchError(
                "valid aggregate first transition cannot carry an invalid reason"
            )
        if resolved_action != requested_action:
            raise RecurrentCounterfactualBranchError(
                "valid aggregate first transition must resolve its requested action"
            )
    elif (
        invalid_reason != "not_in_resolution_action_mask"
        or resolved_action != "stay"
    ):
        raise RecurrentCounterfactualBranchError(
            "resolution-invalid aggregate first transition must fail closed to stay "
            "with the canonical resolution-mask reason"
        )
    expected_moved = resolution_valid and resolved_action.startswith("move_")
    if first_transition["moved"] != expected_moved:
        raise RecurrentCounterfactualBranchError(
            "aggregate first transition movement flags are incoherent"
        )
    if first_transition["died"] != (not first_transition["after_alive"]):
        raise RecurrentCounterfactualBranchError(
            "aggregate first transition death and alive flags are incoherent"
        )
    reward_total = _finite_float(
        first_transition.get("reward_total"),
        field="aggregate first transition reward total",
    )
    if not REWARD_TOTAL_BOUNDS[0] <= reward_total <= REWARD_TOTAL_BOUNDS[1]:
        raise RecurrentCounterfactualBranchError(
            "aggregate first transition reward total is outside canonical bounds"
        )
    for field in (
        "after_energy_ratio",
        "after_hydration_ratio",
        "after_health_ratio",
    ):
        value = first_transition.get(field)
        if first_transition["after_alive"]:
            _nonnegative_float(
                value,
                field=f"aggregate first transition {field}",
            )
            continue
        if value is None:
            continue
        _finite_float(value, field=f"aggregate first transition {field}")


def _validate_compact_branch_outcome(
    outcome: Mapping[str, object],
    *,
    field: str,
    maximum_transition_count: int,
) -> None:
    first_transition = _mapping(
        outcome.get("first_transition"),
        field=f"{field} first transition",
    )
    _validate_compact_first_transition(first_transition)
    transition_count = _positive_int(
        outcome.get("focal_transition_count"),
        field=f"{field} focal transition count",
    )
    policy_count = _positive_int(
        outcome.get("focal_policy_decision_count"),
        field=f"{field} focal policy decision count",
    )
    passive_count = _nonnegative_int(
        outcome.get("focal_passive_transition_count"),
        field=f"{field} focal passive transition count",
    )
    if transition_count != policy_count + passive_count:
        raise RecurrentCounterfactualBranchError(
            f"{field} focal transition counts are inconsistent"
        )
    if transition_count > maximum_transition_count:
        raise RecurrentCounterfactualBranchError(
            f"{field} focal transition count cannot exceed its horizon"
        )
    focal_discounted_return = _finite_float(
        outcome.get("focal_discounted_return"),
        field=f"{field} focal discounted return",
    )
    _validate_feasible_discounted_return(
        focal_discounted_return,
        maximum_transition_count=transition_count,
        field=f"{field} focal discounted return",
    )
    terminal = _mapping(
        outcome.get("focal_terminal"),
        field=f"{field} focal terminal",
    )
    _validate_compact_focal_terminal(terminal, field=f"{field} focal terminal")
    if passive_count > 1:
        raise RecurrentCounterfactualBranchError(
            f"{field} can contain at most one passive terminal transition"
        )
    if passive_count == 1 and terminal.get("alive") is True:
        raise RecurrentCounterfactualBranchError(
            f"{field} passive focal transition requires terminal death"
        )
    population_alive = _nonnegative_int(
        outcome.get("population_alive"),
        field=f"{field} population_alive",
    )
    births_during_horizon = _nonnegative_int(
        outcome.get("births_during_horizon"),
        field=f"{field} births_during_horizon",
    )
    deaths_during_horizon = _nonnegative_int(
        outcome.get("deaths_during_horizon"),
        field=f"{field} deaths_during_horizon",
    )
    if first_transition.get("reproduced") is True and births_during_horizon == 0:
        raise RecurrentCounterfactualBranchError(
            f"{field} first-transition reproduction requires at least one "
            "horizon birth"
        )
    if first_transition.get("died") is True and terminal.get("alive") is True:
        raise RecurrentCounterfactualBranchError(
            f"{field} cannot revive a focal agent after first-transition death"
        )
    if first_transition.get("died") is True:
        if deaths_during_horizon == 0:
            raise RecurrentCounterfactualBranchError(
                f"{field} first-transition death requires at least one horizon death"
            )
        if transition_count != 1 or policy_count != 1 or passive_count != 0:
            raise RecurrentCounterfactualBranchError(
                f"{field} first-transition death must terminate focal transitions"
            )
    if terminal.get("alive") is True and population_alive == 0:
        raise RecurrentCounterfactualBranchError(
            f"{field} cannot report a living focal agent in an empty population"
        )
    if terminal.get("alive") is False and deaths_during_horizon == 0:
        raise RecurrentCounterfactualBranchError(
            f"{field} terminal-dead focal outcome requires at least one "
            "horizon death"
        )
    if not _valid_sha256(outcome.get("behavior_digest")) or not _valid_sha256(
        outcome.get("evidence_digest")
    ):
        raise RecurrentCounterfactualBranchError(
            f"{field} behavior or evidence digest is malformed"
        )


def _validate_compact_focal_terminal(
    terminal: Mapping[str, object],
    *,
    field: str,
) -> None:
    if set(terminal) != {
        "alive",
        "energy_ratio",
        "hydration_ratio",
        "health_ratio",
    }:
        raise RecurrentCounterfactualBranchError(
            f"{field} field set drifted"
        )
    alive = terminal.get("alive")
    if type(alive) is not bool:
        raise RecurrentCounterfactualBranchError(
            f"{field} alive must be an exact boolean"
        )
    ratio_fields = ("energy_ratio", "hydration_ratio", "health_ratio")
    if alive:
        for ratio_field in ratio_fields:
            _nonnegative_float(
                terminal.get(ratio_field),
                field=f"{field} {ratio_field}",
            )
    elif any(terminal.get(ratio_field) is not None for ratio_field in ratio_fields):
        raise RecurrentCounterfactualBranchError(
            f"{field} dead-state ratios must be null"
        )


def _validate_current_public_observation(
    observation: Mapping[str, object],
) -> None:
    if set(observation) != {"schema_version", "policy", "values", "shape"}:
        raise RecurrentCounterfactualBranchError(
            "current public observation field set drifted"
        )
    if (
        observation.get("schema_version") != ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
        or observation.get("policy") != ECOLOGICAL_POLICY_INPUT_POLICY
    ):
        raise RecurrentCounterfactualBranchError(
            "current public observation contract drifted"
        )
    _validate_exact_vector_shape(
        observation.get("shape"),
        expected_size=ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        field="current public observation shape",
    )
    values = observation.get("values")
    if not isinstance(values, list):
        raise RecurrentCounterfactualBranchError(
            "current public observation values must be a list"
        )
    _validate_bounded_numeric_vector(
        values,
        expected_length=ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        field="current public observation",
        minimum=-1.0,
        maximum=1.0,
    )


def _validate_previous_public_feedback(
    feedback: Mapping[str, object],
) -> None:
    if set(feedback) != {"schema_version", "shape", "values"}:
        raise RecurrentCounterfactualBranchError(
            "previous public feedback field set drifted"
        )
    if (
        feedback.get("schema_version") != PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION
    ):
        raise RecurrentCounterfactualBranchError(
            "previous public feedback contract drifted"
        )
    _validate_exact_vector_shape(
        feedback.get("shape"),
        expected_size=PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        field="previous public feedback shape",
    )
    values = feedback.get("values")
    if not isinstance(values, list):
        raise RecurrentCounterfactualBranchError(
            "previous public feedback values must be a list"
        )
    _validate_bounded_numeric_vector(
        values,
        expected_length=PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        field="previous public feedback",
        minimum=-1.0,
        maximum=1.0,
    )
    try:
        validate_previous_feedback_tensor(
            tensor(values, dtype=float64),
            leading_shape=(),
        )
    except RecurrentContextError as error:
        raise RecurrentCounterfactualBranchError(
            "previous public feedback violates its semantic contract"
        ) from error


def _validate_feedback_history_alignment(
    prefix: Mapping[str, object],
    feedback: Mapping[str, object],
    *,
    field: str,
) -> None:
    if prefix.get("record_count") != 0:
        return
    values = feedback.get("values")
    if not isinstance(values, list):
        raise AssertionError("validated previous feedback values became invalid")
    if any(float(value) != 0.0 for value in values):
        raise RecurrentCounterfactualBranchError(
            f"{field} must be all zero when the public history is empty"
        )


def _validate_feasible_discounted_return(
    value: float,
    *,
    maximum_transition_count: int,
    field: str,
) -> None:
    lower = float(REWARD_TOTAL_BOUNDS[0]) * maximum_transition_count
    upper = float(REWARD_TOTAL_BOUNDS[1]) * maximum_transition_count
    tolerance = 1.0e-9 * max(1.0, abs(lower), abs(upper))
    if value < lower - tolerance or value > upper + tolerance:
        raise RecurrentCounterfactualBranchError(
            f"{field} is outside feasible reward bounds"
        )


def _continuation_initial_rng_provenance(
    world: SimulationWorld,
    *,
    focal_agent_id: int,
) -> dict[str, object]:
    policy = world.policy
    if isinstance(
        policy,
        (
            OneShotRecurrentCounterfactualPolicy,
            _DecisionBoundaryRetapedRecurrentCounterfactualPolicy,
        ),
    ):
        policy = policy.delegate
    if not isinstance(policy, DeterministicPublicRecurrentPolicy):
        raise RecurrentCounterfactualBranchError(
            "continuation checkpoint lost its recurrent policy"
        )
    checkpoint = policy.diagnostics_checkpoint_state(agent_id=focal_agent_id)
    return {
        "initial_environment_rng_state_sha256": stable_payload_digest(
            world.rng.getstate()
        ),
        "initial_policy_sampling_state_sha256": checkpoint.get("sampling_state_sha256"),
    }


def _policy_sampling_state_sha256(
    policy: DeterministicPublicRecurrentPolicy,
) -> str:
    sampling_generator = policy._sampling_generator  # noqa: SLF001 - diagnostics seam
    if sampling_generator is None:
        raise RecurrentCounterfactualBranchError(
            "continuation policy lacks a sampling generator"
        )
    state = sampling_generator.get_state().detach().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(state.shape)).encode("ascii"))
    digest.update(bytes(state.tolist()))
    return digest.hexdigest()


def _seeded_environment_rng_state_sha256(seed: int) -> str:
    generator = random.Random()
    generator.seed(seed)
    return stable_payload_digest(generator.getstate())


def _seeded_policy_sampling_state_sha256(seed: int) -> str:
    generator = Generator(device="cpu")
    generator.manual_seed(seed)
    state = generator.get_state().detach().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(state.shape)).encode("ascii"))
    digest.update(bytes(state.tolist()))
    return digest.hexdigest()


def _decision_boundary_provenance(
    value: Mapping[str, object],
) -> dict[str, object]:
    boundary = value.get("continuation_rng_retape_boundary")
    if boundary != RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY:
        raise RecurrentCounterfactualBranchError(
            "continuation RNG retape boundary drifted"
        )
    result: dict[str, object] = {
        "continuation_rng_retape_boundary": boundary,
    }
    for field in _DECISION_BOUNDARY_PROVENANCE_DIGEST_FIELDS:
        digest = value.get(field)
        if not _valid_sha256(digest):
            raise RecurrentCounterfactualBranchError(
                f"decision-boundary provenance field {field!r} is malformed"
            )
        result[field] = digest
    for field in _DECISION_BOUNDARY_PROVENANCE_BOOLEAN_FIELDS:
        if value.get(field) is not True:
            raise RecurrentCounterfactualBranchError(
                f"decision-boundary provenance field {field!r} must be true"
            )
        result[field] = True
    return result


def _source_decision_prefix_projection_sha256(
    diagnostics: Mapping[str, object],
) -> str:
    """Bind the entire learner decision at the fixed causal prefix.

    The override diagnostics add intervention-only fields, so compare the
    common source-decision projection that must be identical for the natural
    baseline and every forced branch before continuation RNG retaping.
    """

    missing = [
        field
        for field in _SOURCE_DECISION_PREFIX_PROJECTION_FIELDS
        if field not in diagnostics
    ]
    if missing:
        raise RecurrentCounterfactualBranchError(
            "source decision projection is missing required diagnostics"
        )
    projection = {
        field: deepcopy(diagnostics[field])
        for field in _SOURCE_DECISION_PREFIX_PROJECTION_FIELDS
    }
    return stable_payload_digest(projection)


def _execute_continuation(
    checkpoint_world: SimulationWorld,
    *,
    branch_id: str,
    branch_tick: int,
    horizon_ticks: int,
    focal_agent_id: int,
    source_action: str,
    expected_observation_digest: str,
    expected_action_mask_digest: str,
    expected_source_decision_projection_sha256: str | None = None,
    forced_action: str | None,
    gamma: float,
    prefix_horizons: Sequence[int] | None = None,
    require_source_action_match: bool = True,
    boundary_environment_sampling_seed: int | None = None,
    boundary_policy_sampling_seed: int | None = None,
) -> dict[str, object]:
    world = deepcopy(checkpoint_world)
    _configure_manual_world(world)
    initial_rng_provenance = _continuation_initial_rng_provenance(
        world,
        focal_agent_id=focal_agent_id,
    )
    record_start = len(world.trajectory_records)
    diagnostics_start = len(world.policy_decision_diagnostics_records)
    wrapper: (
        OneShotRecurrentCounterfactualPolicy
        | _DecisionBoundaryRetapedRecurrentCounterfactualPolicy
        | None
    ) = None
    boundary_retape_requested = (
        boundary_environment_sampling_seed is not None
        or boundary_policy_sampling_seed is not None
    )
    if boundary_retape_requested:
        if (
            boundary_environment_sampling_seed is None
            or boundary_policy_sampling_seed is None
            or expected_source_decision_projection_sha256 is None
        ):
            raise RecurrentCounterfactualBranchError(
                "decision-boundary retaping requires both continuation seeds "
                "and the exact source decision projection"
            )
        if not isinstance(world.policy, DeterministicPublicRecurrentPolicy):
            raise RecurrentCounterfactualBranchError(
                "branch checkpoint lost its recurrent policy"
            )
        wrapper = _DecisionBoundaryRetapedRecurrentCounterfactualPolicy(
            delegate=world.policy,
            environment_rng=world.rng,
            target_agent_id=focal_agent_id,
            source_action=source_action,
            forced_action=forced_action,
            expected_observation_digest=expected_observation_digest,
            expected_action_mask_digest=expected_action_mask_digest,
            expected_source_decision_projection_sha256=(
                expected_source_decision_projection_sha256
            ),
            intervention_id=(
                f"{branch_id}:action:{forced_action or 'natural_baseline'}"
            ),
            environment_sampling_seed=boundary_environment_sampling_seed,
            policy_sampling_seed=boundary_policy_sampling_seed,
        )
        world.policy = wrapper
    elif forced_action is not None:
        if not isinstance(world.policy, DeterministicPublicRecurrentPolicy):
            raise RecurrentCounterfactualBranchError(
                "branch checkpoint lost its recurrent policy"
            )
        wrapper = OneShotRecurrentCounterfactualPolicy(
            delegate=world.policy,
            target_agent_id=focal_agent_id,
            forced_action=forced_action,
            expected_observation_digest=expected_observation_digest,
            expected_action_mask_digest=expected_action_mask_digest,
            intervention_id=f"{branch_id}:action:{forced_action}",
        )
        world.policy = wrapper

    start_births = int(world.births)
    start_deaths = int(world.deaths)
    executed_ticks = 0
    resolved_prefix_horizons = _validated_prefix_horizons(
        prefix_horizons,
        maximum_horizon=horizon_ticks,
    )
    snapshots: dict[int, dict[str, object]] = {}
    for tick in range(branch_tick, branch_tick + horizon_ticks):
        if not world.alive_agents():
            break
        world.tick = tick
        world._run_tick()
        executed_ticks += 1
        if executed_ticks in resolved_prefix_horizons:
            snapshots[executed_ticks] = _continuation_snapshot(
                world,
                record_start=record_start,
                diagnostics_start=diagnostics_start,
                start_births=start_births,
                start_deaths=start_deaths,
                executed_ticks=executed_ticks,
                focal_agent_id=focal_agent_id,
            )
    final_snapshot = _continuation_snapshot(
        world,
        record_start=record_start,
        diagnostics_start=diagnostics_start,
        start_births=start_births,
        start_deaths=start_deaths,
        executed_ticks=executed_ticks,
        focal_agent_id=focal_agent_id,
    )
    for prefix_horizon in resolved_prefix_horizons:
        snapshots.setdefault(prefix_horizon, deepcopy(final_snapshot))

    intervention_count = wrapper.intervention_count if wrapper is not None else 0
    natural_action = (
        wrapper.natural_requested_action if wrapper is not None else source_action
    )
    result = _summarize_continuation_snapshot(
        final_snapshot,
        branch_tick=branch_tick,
        horizon_ticks=horizon_ticks,
        focal_agent_id=focal_agent_id,
        source_action=source_action,
        forced_action=forced_action,
        gamma=gamma,
        intervention_count=intervention_count,
        natural_action=natural_action,
        require_source_action_match=require_source_action_match,
    )
    result.update(initial_rng_provenance)
    boundary_provenance: dict[str, object] | None = None
    if isinstance(
        wrapper,
        _DecisionBoundaryRetapedRecurrentCounterfactualPolicy,
    ):
        if wrapper.used is not True or wrapper.boundary_provenance is None:
            raise RecurrentCounterfactualBranchError(
                "focal decision boundary was not reached exactly once"
            )
        boundary_provenance = _decision_boundary_provenance(
            wrapper.boundary_provenance
        )
        result.update(boundary_provenance)
    if resolved_prefix_horizons:
        result["nested_prefix_runs"] = {
            str(prefix_horizon): _summarize_continuation_snapshot(
                snapshots[prefix_horizon],
                branch_tick=branch_tick,
                horizon_ticks=prefix_horizon,
                focal_agent_id=focal_agent_id,
                source_action=source_action,
                forced_action=forced_action,
                gamma=gamma,
                intervention_count=intervention_count,
                natural_action=natural_action,
                require_source_action_match=require_source_action_match,
            )
            for prefix_horizon in resolved_prefix_horizons
        }
        for nested in result["nested_prefix_runs"].values():
            nested.update(initial_rng_provenance)
            if boundary_provenance is not None:
                nested.update(boundary_provenance)
    return result


def _continuation_snapshot(
    world: SimulationWorld,
    *,
    record_start: int,
    diagnostics_start: int,
    start_births: int,
    start_deaths: int,
    executed_ticks: int,
    focal_agent_id: int,
) -> dict[str, object]:
    return {
        "records": deepcopy(tuple(world.trajectory_records[record_start:])),
        "diagnostics": deepcopy(
            tuple(world.policy_decision_diagnostics_records[diagnostics_start:])
        ),
        "executed_tick_count": executed_ticks,
        "focal_terminal": _focal_terminal(world, focal_agent_id),
        "population_alive": len(world.alive_agents()),
        "births_during_horizon": int(world.births) - start_births,
        "deaths_during_horizon": int(world.deaths) - start_deaths,
    }


def _summarize_continuation_snapshot(
    snapshot: Mapping[str, object],
    *,
    branch_tick: int,
    horizon_ticks: int,
    focal_agent_id: int,
    source_action: str,
    forced_action: str | None,
    gamma: float,
    intervention_count: int,
    natural_action: str | None,
    require_source_action_match: bool,
) -> dict[str, object]:
    raw_records = snapshot.get("records")
    raw_diagnostics = snapshot.get("diagnostics")
    if not isinstance(raw_records, tuple) or not isinstance(raw_diagnostics, tuple):
        raise RecurrentCounterfactualBranchError(
            "continuation snapshot records and diagnostics must be tuples"
        )
    records = tuple(
        _mapping(record, field="continuation record") for record in raw_records
    )
    diagnostics = tuple(raw_diagnostics)
    if len(records) != len(diagnostics):
        raise RecurrentCounterfactualBranchError(
            "trajectory records and decision diagnostics are not aligned"
        )
    first_focal = next(
        (
            record
            for record in records
            if _record_int(record, "tick") == branch_tick
            and _record_int(record, "agent_id") == focal_agent_id
            and record.get("action_source") != "passive"
        ),
        None,
    )
    if first_focal is None:
        raise RecurrentCounterfactualBranchError(
            "focal agent did not act at the materialized branch state"
        )
    expected_first_action = (
        _stable_action(
            first_focal.get("requested_action"),
            field="retaped baseline first requested action",
        )
        if forced_action is None and not require_source_action_match
        else forced_action or source_action
    )
    if first_focal.get("requested_action") != expected_first_action:
        raise RecurrentCounterfactualBranchError(
            "focal continuation requested an unexpected first action"
        )

    if forced_action is None:
        natural_action = _stable_action(
            first_focal.get("requested_action"),
            field="baseline first requested action",
        )
    if require_source_action_match and natural_action != source_action:
        raise RecurrentCounterfactualBranchError(
            "branch learner natural action drifted from the source decision"
        )
    if forced_action is not None and intervention_count != 1:
        raise RecurrentCounterfactualBranchError(
            "counterfactual continuation did not apply exactly one intervention"
        )

    action_source_counts = Counter(
        str(record.get("action_source", "unknown")) for record in records
    )
    unsupported_requested_action_count = sum(
        1
        for record in records
        if record.get("action_source") != "passive"
        and record.get("action_valid") is not True
    )
    if unsupported_requested_action_count:
        raise RecurrentCounterfactualBranchError(
            "continuation contained an unsupported requested action"
        )
    heuristic_action_source_count = sum(
        count for source, count in action_source_counts.items() if "heuristic" in source
    )
    if heuristic_action_source_count:
        raise RecurrentCounterfactualBranchError(
            "continuation contained a heuristic action source"
        )
    allowed_sources = {
        RECURRENT_ROLLOUT_ACTION_SOURCE,
        RECURRENT_COUNTERFACTUAL_ACTION_SOURCE,
        "passive",
    }
    unexpected_action_source_count = sum(
        count
        for source, count in action_source_counts.items()
        if source not in allowed_sources
    )
    if unexpected_action_source_count:
        raise RecurrentCounterfactualBranchError(
            "continuation contained a non-recurrent action source"
        )

    (
        focal_records,
        discounted_return,
        focal_passive_transition_count,
    ) = _discounted_focal_return(
        records,
        focal_agent_id=focal_agent_id,
        gamma=gamma,
    )
    terminal = _mapping(
        snapshot.get("focal_terminal"),
        field="snapshot focal terminal",
    )
    final_metrics = {
        "executed_tick_count": _nonnegative_int(
            snapshot.get("executed_tick_count"),
            field="executed tick count",
        ),
        "focal_transition_count": len(focal_records),
        "focal_policy_decision_count": (
            len(focal_records) - focal_passive_transition_count
        ),
        "focal_passive_transition_count": focal_passive_transition_count,
        "focal_discounted_return": _round(discounted_return),
        "focal_terminal": dict(terminal),
        "population_alive": _nonnegative_int(
            snapshot.get("population_alive"),
            field="population alive",
        ),
        "births_during_horizon": _nonnegative_int(
            snapshot.get("births_during_horizon"),
            field="births during horizon",
        ),
        "deaths_during_horizon": _nonnegative_int(
            snapshot.get("deaths_during_horizon"),
            field="deaths during horizon",
        ),
    }
    behavior_payload = {
        "branch_tick": branch_tick,
        "horizon_ticks": horizon_ticks,
        "focal_agent_id": focal_agent_id,
        "records": [_causal_record_payload(record) for record in records],
        "final_metrics": final_metrics,
    }
    evidence_payload = {
        "behavior": behavior_payload,
        "policy_records": [
            {
                "tick": record.get("tick"),
                "agent_id": record.get("agent_id"),
                "action_source": record.get("action_source"),
                "policy_id": record.get("policy_id"),
                "policy_version": record.get("policy_version"),
                "requested_action": record.get("requested_action"),
            }
            for record in records
        ],
        "decision_diagnostics": list(diagnostics),
        "forced_action": forced_action,
        "intervention_count": intervention_count,
        "natural_requested_action": natural_action,
    }
    causal_digest_sequence = tuple(
        stable_payload_digest(_causal_record_payload(record)) for record in records
    )
    policy_digest_sequence = tuple(
        stable_payload_digest(
            {
                "tick": record.get("tick"),
                "agent_id": record.get("agent_id"),
                "action_source": record.get("action_source"),
                "policy_id": record.get("policy_id"),
                "policy_version": record.get("policy_version"),
                "requested_action": record.get("requested_action"),
            }
        )
        for record in records
    )
    diagnostics_digest_sequence = tuple(
        stable_payload_digest(diagnostic) for diagnostic in diagnostics
    )
    return {
        "forced_action": forced_action,
        "natural_requested_action": natural_action,
        "intervention_count": intervention_count,
        "first_transition": _compact_first_transition(first_focal),
        **final_metrics,
        "unsupported_requested_action_count": unsupported_requested_action_count,
        "heuristic_action_source_count": heuristic_action_source_count,
        "unexpected_action_source_count": unexpected_action_source_count,
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "behavior_digest": stable_payload_digest(behavior_payload),
        "evidence_digest": stable_payload_digest(evidence_payload),
        "causal_record_digest_sequence": causal_digest_sequence,
        "policy_record_digest_sequence": policy_digest_sequence,
        "diagnostics_digest_sequence": diagnostics_digest_sequence,
    }


def _validated_prefix_horizons(
    value: Sequence[int] | None,
    *,
    maximum_horizon: int,
) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecurrentCounterfactualBranchError(
            "prefix horizons must be an ordered sequence"
        )
    parsed = tuple(_positive_int(item, field="prefix horizon") for item in value)
    if not parsed or tuple(sorted(set(parsed))) != parsed:
        raise RecurrentCounterfactualBranchError(
            "prefix horizons must be unique and strictly increasing"
        )
    if parsed[-1] != maximum_horizon:
        raise RecurrentCounterfactualBranchError(
            "prefix horizons must end at the maximum continuation horizon"
        )
    return parsed


def _discounted_focal_return(
    records: Sequence[Mapping[str, object]],
    *,
    focal_agent_id: int,
    gamma: float,
) -> tuple[tuple[Mapping[str, object], ...], float, int]:
    """Return every focal transition, including a passive terminal death.

    Foundation emits a trajectory row when an agent dies before its turn. Its
    synthetic ``stay`` is not a policy action, but its reward is the terminal
    transition reward and therefore remains part of the focal return.
    """

    focal_records = tuple(
        record
        for record in records
        if _record_int(record, "agent_id") == focal_agent_id
    )
    discounted_return = 0.0
    passive_transition_count = 0
    for index, record in enumerate(focal_records):
        reward = _mapping(record.get("reward"), field="focal reward")
        discounted_return += (gamma**index) * _finite_float(
            reward.get("total"),
            field="focal reward total",
        )
        if record.get("action_source") == "passive":
            passive_transition_count += 1
    return focal_records, discounted_return, passive_transition_count


def _compact_action_label(
    run: Mapping[str, object],
    *,
    baseline: Mapping[str, object],
) -> dict[str, object]:
    action = _stable_action(run.get("forced_action"), field="forced action label")
    run_terminal = _mapping(run.get("focal_terminal"), field="focal terminal")
    baseline_terminal = _mapping(
        baseline.get("focal_terminal"),
        field="baseline focal terminal",
    )
    return {
        "action": action,
        "natural_requested_action": run.get("natural_requested_action"),
        "intervention_count": run.get("intervention_count"),
        "first_transition": deepcopy(run.get("first_transition")),
        "focal_transition_count": run.get("focal_transition_count"),
        "focal_policy_decision_count": run.get("focal_policy_decision_count"),
        "focal_passive_transition_count": run.get("focal_passive_transition_count"),
        "focal_discounted_return": run.get("focal_discounted_return"),
        "focal_terminal": deepcopy(run_terminal),
        "population_alive": run.get("population_alive"),
        "births_during_horizon": run.get("births_during_horizon"),
        "deaths_during_horizon": run.get("deaths_during_horizon"),
        "paired_vs_baseline": {
            "focal_discounted_return_delta": _round(
                float(run["focal_discounted_return"])
                - float(baseline["focal_discounted_return"])
            ),
            "focal_terminal_alive_delta": int(run_terminal.get("alive") is True)
            - int(baseline_terminal.get("alive") is True),
            "population_alive_delta": int(run["population_alive"])
            - int(baseline["population_alive"]),
            "births_during_horizon_delta": int(run["births_during_horizon"])
            - int(baseline["births_during_horizon"]),
            "deaths_during_horizon_delta": int(run["deaths_during_horizon"])
            - int(baseline["deaths_during_horizon"]),
        },
        "unsupported_requested_action_count": run.get(
            "unsupported_requested_action_count"
        ),
        "heuristic_action_source_count": run.get("heuristic_action_source_count"),
        "unexpected_action_source_count": run.get("unexpected_action_source_count"),
        "behavior_digest": run.get("behavior_digest"),
        "evidence_digest": run.get("evidence_digest"),
        "replay_verified": run.get("replay_verified"),
        "replay_evidence_digest": run.get("replay_evidence_digest"),
    }


def _compact_baseline_label(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "first_transition": deepcopy(run.get("first_transition")),
        "focal_transition_count": run.get("focal_transition_count"),
        "focal_policy_decision_count": run.get("focal_policy_decision_count"),
        "focal_passive_transition_count": run.get("focal_passive_transition_count"),
        "focal_discounted_return": run.get("focal_discounted_return"),
        "focal_terminal": deepcopy(run.get("focal_terminal")),
        "population_alive": run.get("population_alive"),
        "births_during_horizon": run.get("births_during_horizon"),
        "deaths_during_horizon": run.get("deaths_during_horizon"),
        "behavior_digest": run.get("behavior_digest"),
        "evidence_digest": run.get("evidence_digest"),
    }


def _compact_first_transition(record: Mapping[str, object]) -> dict[str, object]:
    outcome = _mapping(record.get("outcome"), field="first transition outcome")
    reward = _mapping(record.get("reward"), field="first transition reward")
    after = _mapping(record.get("after"), field="first transition after")
    action_valid = record.get("action_valid") is True
    resolution_valid = record.get("resolution_action_valid") is True
    invalid_reason = outcome.get("invalid_reason")
    requested_action = _stable_action(
        record.get("requested_action"),
        field="first transition requested_action",
    )
    return {
        "requested_action": requested_action,
        "resolved_action": _stable_action(
            record.get("resolved_action"),
            field="first transition resolved_action",
        ),
        "observation_action_valid": action_valid,
        "resolution_action_valid": resolution_valid,
        "same_tick_resolution_mask_drift": action_valid and not resolution_valid,
        "expected_same_tick_occupancy_drift": (
            requested_action.startswith("move_")
            and action_valid
            and not resolution_valid
            and invalid_reason == "not_in_resolution_action_mask"
        ),
        "invalid_reason": invalid_reason,
        "moved": record.get("moved") is True,
        "reward_total": _finite_float(
            reward.get("total"),
            field="first transition reward total",
        ),
        "reproduced": outcome.get("reproduced") is True,
        "died": outcome.get("died") is True,
        "after_alive": after.get("alive") is True,
        "after_energy_ratio": _optional_finite_float(after.get("energy_ratio")),
        "after_hydration_ratio": _optional_finite_float(after.get("hydration_ratio")),
        "after_health_ratio": _optional_finite_float(after.get("health_ratio")),
    }


def _causal_record_payload(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "tick": record.get("tick"),
        "agent_id": record.get("agent_id"),
        "requested_action": record.get("requested_action"),
        "resolved_action": record.get("resolved_action"),
        "action_valid": record.get("action_valid"),
        "resolution_action_valid": record.get("resolution_action_valid"),
        "moved": record.get("moved"),
        "before": record.get("before"),
        "after": record.get("after"),
        "outcome": record.get("outcome"),
        "reward": record.get("reward"),
    }


def _select_focal_source_record(
    records: Sequence[Mapping[str, object]],
    diagnostics: Sequence[object],
    *,
    focal_agent_id: int | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if len(records) != len(diagnostics):
        raise RecurrentCounterfactualBranchError(
            "source records and decision diagnostics are not aligned"
        )
    candidates: list[tuple[int, int, dict[str, object], dict[str, object]]] = []
    for index, (record, raw_diagnostics) in enumerate(
        zip(records, diagnostics, strict=True)
    ):
        action_source = record.get("action_source")
        if action_source == "passive":
            if raw_diagnostics is not None:
                raise RecurrentCounterfactualBranchError(
                    "passive source decision diagnostics must be absent"
                )
            continue
        if action_source != RECURRENT_ROLLOUT_ACTION_SOURCE:
            raise RecurrentCounterfactualBranchError(
                "source action source is not canonical recurrent or passive"
            )
        agent_id = _positive_int(record.get("agent_id"), field="source agent_id")
        if focal_agent_id is not None and agent_id != focal_agent_id:
            continue
        action_mask = _complete_action_mask(record.get("action_mask"))
        parsed_diagnostics = _mapping(
            raw_diagnostics,
            field="source decision diagnostics",
        )
        if parsed_diagnostics.get("agent_id") != agent_id:
            raise RecurrentCounterfactualBranchError(
                "source diagnostic agent identity is not aligned"
            )
        candidates.append(
            (
                -sum(action_mask.values()),
                index,
                dict(record),
                dict(parsed_diagnostics),
            )
        )
    if not candidates:
        raise RecurrentCounterfactualBranchError(
            "no recurrent learner decision matched the requested focal state"
        )
    _, _, selected_record, selected_diagnostics = min(candidates)
    return selected_record, selected_diagnostics


def _world_for_scenario(
    *,
    scenario: str,
    environment_seed: int,
    ticks: int,
    policy: DeterministicPublicRecurrentPolicy,
) -> SimulationWorld:
    if scenario == RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO:
        return SimulationWorld(
            WorldConfig(seed=environment_seed, max_ticks=ticks),
            policy=policy,
        )
    return _fixture_world(
        fixture_name=scenario,
        seed=environment_seed,
        ticks=ticks,
        policy=policy,
    )


def _configure_manual_world(world: SimulationWorld) -> None:
    world.mode = RunMode.SUMMARY_ONLY
    world.record_events = False
    world.record_tick_details = True
    world.record_trajectory = True
    world.retain_trajectory_records = True
    world.trajectory_sink = None


def _counterfactual_contract(
    *,
    verify_replay: bool,
    seed_role: str,
) -> dict[str, object]:
    allowed_seed_roles = _counterfactual_seed_role_cohort(seed_role)
    return {
        "schema_version": RECURRENT_COUNTERFACTUAL_BRANCH_PROTOCOL_VERSION,
        "diagnostics_only": True,
        "source_policy_frozen": True,
        "source_state_policy": "real_frozen_recurrent_learner_visited_state",
        "candidate_action_policy": "every_tick_start_public_mask_valid_action",
        "focal_intervention_count_per_action": 1,
        "other_agent_policy": "same_frozen_recurrent_policy_state",
        "natural_actor_evaluated_before_override": True,
        "natural_sampler_draw_consumed_before_override": True,
        "actual_forced_outcome_used_as_next_public_feedback": True,
        "exact_replay_required": verify_replay,
        "baseline_equivalence": "source_action_behavior_digest_exact_match",
        "allowed_seed_roles": list(allowed_seed_roles),
        "trainable_public_context_fields": sorted(_TRAINABLE_PUBLIC_CONTEXT_KEYS),
        "optimizer_context_fields": sorted(_OPTIMIZER_CONTEXT_KEYS),
        "metadata_used_as_actor_input": False,
        "outcome_labels_used_as_actor_input": False,
        "optimizer_context_used_as_runtime_environment_input": False,
        "public_history_prefix_policy": (
            "full_focal_episode_public_actor_input_prefix"
        ),
        "current_model_recurrent_state_policy": (
            "recompute_from_public_history_prefix_never_consume_stale_hidden"
        ),
        "stored_source_recurrent_state_policy": (
            "exact_source_artifact_and_model_verification_only"
        ),
        "historical_row_training_use": RECURRENT_COUNTERFACTUAL_TRAINING_USE,
        "ppo_ratio_data_use_forbidden": True,
        "behavior_masked_distribution_recorded": True,
        "focal_return_record_policy": (
            "all_focal_trajectory_records_including_passive_terminal"
        ),
        "environment_and_policy_sampling_seeds_separate": True,
        "continuation_repeat_seed_provenance_recorded": True,
        "target_estimand": (
            "single_exact_rollout_on_one_deepcopied_sequential_rng_tape"
        ),
        "expected_causal_effect_estimated": False,
        "continuation_outcome_uncertainty_estimated": False,
        "event_aligned_common_random_numbers": False,
        "exact_replay_proves_determinism_not_statistical_uncertainty": True,
        "sequential_rng_event_reassignment_after_branch_divergence_possible": True,
        "private_world_checkpoint_serialized": False,
        "ordinary_decide_calls_override": False,
        "training_ran": False,
        "runtime_integrated": False,
    }


def _focal_terminal(world: SimulationWorld, agent_id: int) -> dict[str, object]:
    agent = world.agents.get(agent_id)
    alive = bool(agent is not None and agent.alive)
    if not alive or agent is None:
        return {
            "alive": False,
            "energy_ratio": None,
            "hydration_ratio": None,
            "health_ratio": None,
        }
    return {
        "alive": True,
        "energy_ratio": _round(agent.energy / max(agent.genome.max_energy, 1e-9)),
        "hydration_ratio": _round(
            agent.hydration / max(agent.genome.max_hydration, 1e-9)
        ),
        "health_ratio": _round(agent.health / max(agent.max_health, 1e-9)),
    }


def _forbidden_key_paths(value: object, *, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            lowered = key_text.lower()
            if any(token in lowered for token in _FORBIDDEN_TRAINABLE_KEY_TOKENS):
                failures.append(path)
            failures.extend(_forbidden_key_paths(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            failures.extend(_forbidden_key_paths(nested, prefix=f"{prefix}[{index}]"))
    return failures


def _validated_seed_role(seed_role: object, environment_seed: object) -> str:
    role = _nonempty_string(seed_role, field="seed_role")
    if role not in RECURRENT_COUNTERFACTUAL_SEED_ROLES:
        raise RecurrentCounterfactualBranchError(
            "counterfactual queries may use only train or curriculum legacy "
            "seeds or scale_train/scale_curriculum development seeds"
        )
    seed = _positive_int(environment_seed, field="environment_seed")
    if role in RECURRENT_COUNTERFACTUAL_SCALE_V2_SEED_ROLES:
        registry = SCALE_DEVELOPMENT_V2_SEED_REGISTRY
    elif role in RECURRENT_COUNTERFACTUAL_SCALE_V1_SEED_ROLES:
        registry = SCALE_DEVELOPMENT_SEED_REGISTRY
    else:
        registry = RECURRENT_SEED_REGISTRY
    if seed not in registry[role]:
        raise RecurrentCounterfactualBranchError(
            f"environment_seed is not registered for seed role {role!r}"
        )
    return role


def _counterfactual_seed_role_cohort(role: str) -> tuple[str, ...]:
    if role in RECURRENT_COUNTERFACTUAL_SCALE_V2_SEED_ROLES:
        return RECURRENT_COUNTERFACTUAL_SCALE_V2_SEED_ROLES
    if role in RECURRENT_COUNTERFACTUAL_SCALE_V1_SEED_ROLES:
        return RECURRENT_COUNTERFACTUAL_SCALE_V1_SEED_ROLES
    if role in RECURRENT_COUNTERFACTUAL_LEGACY_SEED_ROLES:
        return RECURRENT_COUNTERFACTUAL_LEGACY_SEED_ROLES
    raise RecurrentCounterfactualBranchError(
        f"counterfactual seed role {role!r} has no allowed cohort"
    )


def _validated_scenario(value: object) -> str:
    scenario = _nonempty_string(value, field="scenario")
    if scenario not in RECURRENT_COUNTERFACTUAL_SCENARIOS:
        raise RecurrentCounterfactualBranchError(
            f"unsupported recurrent counterfactual scenario: {scenario!r}"
        )
    return scenario


def _optional_sampling_seed(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 2**63 - 1
    ):
        raise RecurrentCounterfactualBranchError(
            f"{field} must be null or an integer in [0, 2**63 - 1]"
        )
    return value


def _complete_action_mask(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise RecurrentCounterfactualBranchError("action mask must be a mapping")
    if set(value) != set(ACTION_NAMES):
        raise RecurrentCounterfactualBranchError(
            "action mask must contain the complete stable action contract"
        )
    mask: dict[str, bool] = {}
    for action in ACTION_NAMES:
        raw = value[action]
        if type(raw) is not bool:
            raise RecurrentCounterfactualBranchError(
                "action mask values must be exact booleans"
            )
        mask[action] = raw
    if not any(mask.values()):
        raise RecurrentCounterfactualBranchError("action mask cannot be empty")
    return mask


def _observation_agent_id(observation: Mapping[str, object]) -> int:
    metadata = _mapping(observation.get("metadata"), field="observation metadata")
    return _positive_int(metadata.get("agent_id"), field="observation agent_id")


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentCounterfactualBranchError(f"{field} must be a mapping")
    return value


def _stable_action(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value not in ACTION_NAMES:
        raise RecurrentCounterfactualBranchError(f"{field} is not a stable action")
    return value


def _record_int(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecurrentCounterfactualBranchError(f"record {field} must be an integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentCounterfactualBranchError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentCounterfactualBranchError(
            f"{field} must be a non-negative integer"
        )
    return value


def _validated_continuation_tape_indices(
    values: Sequence[int],
    *,
    tape_count: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RecurrentCounterfactualBranchError(
            "continuation tape indices must be a sequence"
        )
    resolved = tuple(
        _nonnegative_int(value, field="continuation tape index")
        for value in values
    )
    if not resolved:
        raise RecurrentCounterfactualBranchError(
            "continuation tape indices cannot be empty"
        )
    if resolved != tuple(sorted(set(resolved))):
        raise RecurrentCounterfactualBranchError(
            "continuation tape indices must be unique and ordered"
        )
    if resolved[-1] >= _positive_int(tape_count, field="continuation_tape_count"):
        raise RecurrentCounterfactualBranchError(
            "continuation tape index exceeds the configured tape count"
        )
    return resolved


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecurrentCounterfactualBranchError(
            f"{field} must be a non-empty trimmed string"
        )
    return value


def _unit_interval(value: object, *, field: str) -> float:
    parsed = _finite_float(value, field=field)
    if parsed < 0.0 or parsed > 1.0:
        raise RecurrentCounterfactualBranchError(f"{field} must be in [0, 1]")
    return parsed


def _nonnegative_float(value: object, *, field: str) -> float:
    parsed = _finite_float(value, field=field)
    if parsed < 0.0:
        raise RecurrentCounterfactualBranchError(
            f"{field} must be non-negative"
        )
    return parsed


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentCounterfactualBranchError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecurrentCounterfactualBranchError(f"{field} must be finite")
    return parsed


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field="optional finite value")


def _validate_numeric_vector(
    values: Sequence[object],
    *,
    expected_length: int,
    field: str,
) -> None:
    if len(values) != expected_length:
        raise RecurrentCounterfactualBranchError(
            f"{field} length does not match its contract"
        )
    for value in values:
        _finite_float(value, field=field)


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
        raise RecurrentCounterfactualBranchError(
            f"{field} must contain exactly one integer size"
        )


def _validate_bounded_numeric_vector(
    values: Sequence[object],
    *,
    expected_length: int,
    field: str,
    minimum: float,
    maximum: float,
) -> None:
    _validate_numeric_vector(
        values,
        expected_length=expected_length,
        field=field,
    )
    for value in values:
        parsed = float(value)
        if parsed < minimum or parsed > maximum:
            raise RecurrentCounterfactualBranchError(
                f"{field} values must be in [{minimum}, {maximum}]"
            )


def _flatten_numeric_tree(value: object, *, field: str) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for nested in value:
            flattened.extend(_flatten_numeric_tree(nested, field=field))
        return flattened
    return [_finite_float(value, field=field)]


def _validated_float32_tensor_tree(
    value: object,
    *,
    expected_shape: Sequence[int],
    field: str,
) -> Tensor:
    _flatten_numeric_tree(value, field=field)
    try:
        parsed = tensor(value, dtype=float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise RecurrentCounterfactualBranchError(
            f"{field} is not a rectangular numeric tensor"
        ) from error
    if list(parsed.shape) != list(expected_shape):
        raise RecurrentCounterfactualBranchError(
            f"{field} does not match its declared shape"
        )
    return parsed


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _round(value: float) -> float:
    return round(float(value), 12)


__all__ = [
    "OneShotRecurrentCounterfactualPolicy",
    "RECURRENT_COUNTERFACTUAL_AGGREGATE_SCHEMA_VERSION",
    "RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION",
    "RECURRENT_COUNTERFACTUAL_BRANCH_PROTOCOL_VERSION",
    "RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION",
    "RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO",
    "RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE",
    "RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE",
    "RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY",
    "RECURRENT_COUNTERFACTUAL_TAPE_CHUNK_SCHEMA_VERSION",
    "RECURRENT_COUNTERFACTUAL_LEGACY_SEED_ROLES",
    "RECURRENT_COUNTERFACTUAL_SCALE_SEED_ROLES",
    "RECURRENT_COUNTERFACTUAL_SCALE_V1_SEED_ROLES",
    "RECURRENT_COUNTERFACTUAL_SCALE_V2_SEED_ROLES",
    "RECURRENT_COUNTERFACTUAL_SCENARIOS",
    "RECURRENT_COUNTERFACTUAL_SEED_ROLES",
    "RECURRENT_COUNTERFACTUAL_TRAINING_USE",
    "RecurrentCounterfactualBranchError",
    "build_recurrent_counterfactual_branch_row",
    "build_recurrent_counterfactual_nested_horizon_materialization",
    "build_recurrent_counterfactual_nested_horizon_tape_chunk_materialization",
    "derive_recurrent_counterfactual_tape_seed",
    "merge_recurrent_counterfactual_nested_horizon_tape_chunks",
    "reconstruct_current_model_hidden_from_branch_row",
    "validate_recurrent_counterfactual_aggregate_row",
    "validate_recurrent_counterfactual_branch_row",
    "verified_source_recurrent_state_from_branch_row",
]
