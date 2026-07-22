from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import random

from torch import Tensor

from evolution_sim.config import WorldConfig
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
import evolution_sim.env.runtime.observations as runtime_observations
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.world import SimulationWorld
from evolution_sim.mind.evaluation_harness import (
    CONTROLLED_FIXTURE_NAMES,
    _fixture_world,
)
from evolution_sim.mind.policy_inputs import ecological_policy_input_payload
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION,
    PREVIOUS_PUBLIC_FEEDBACK_SIZE,
    PublicRecurrentActorCritic,
)
from evolution_sim.mind.recurrent_policy import (
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
from evolution_sim.mind.recurrent_seed_registry import RECURRENT_SEED_REGISTRY


RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION = (
    "mind_v3_recurrent_counterfactual_branch_row_v4"
)
RECURRENT_COUNTERFACTUAL_BRANCH_PROTOCOL_VERSION = (
    "mind_v3_recurrent_exact_all_valid_action_branch_protocol_v4"
)
RECURRENT_COUNTERFACTUAL_TRAINING_USE = (
    "post_update_supervised_auxiliary_policy_improvement_only"
)
RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO = "broad"
RECURRENT_COUNTERFACTUAL_SEED_ROLES = ("train", "curriculum")
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


class RecurrentCounterfactualBranchError(ValueError):
    """Raised when an exact recurrent counterfactual contract fails closed."""


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
    contract = _counterfactual_contract(verify_replay=verify_replay)
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
    gamma: float = 0.99,
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
        ticks=maximum_branch_tick + maximum_horizon,
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

    selected_tick: int | None = None
    fallback_index = -1
    for index, tick in enumerate(resolved_branch_ticks):
        if tick in materialized_candidates:
            selected_tick = tick
            fallback_index = index
            break
    if selected_tick is None:
        raise RecurrentCounterfactualBranchError(
            "no requested branch tick contained an eligible learner decision"
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
    selection_rng = random.Random(resolved_selection_seed)
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
    return {
        "rows": tuple(rows),
        "selection": {
            "requested_branch_ticks": list(resolved_branch_ticks),
            "selected_branch_tick": selected_tick,
            "fallback_index": fallback_index,
            "eligible_candidate_count": len(eligible),
            "candidate_list_digest": candidate_list_digest,
            "selected_candidate_index": selected_candidate_index,
            "selected_focal_agent_id": resolved_focal_agent_id,
            "selection_policy": (
                "uniform_seeded_over_current_tick_multi_action_learner_decisions"
            ),
            "outcome_or_future_data_used": False,
        },
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
    contract = _counterfactual_contract(verify_replay=True)
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
        return verified_source_recurrent_state_for_exact_artifact(
            model,
            optimizer_context,
            artifact_digest=artifact_digest,
        )
    except RecurrentPolicyAdapterError as error:
        raise RecurrentCounterfactualBranchError(
            "stored source hidden failed exact artifact verification"
        ) from error


def validate_recurrent_counterfactual_branch_row(
    row: Mapping[str, object],
) -> None:
    if not isinstance(row, Mapping):
        raise RecurrentCounterfactualBranchError("branch row must be a mapping")
    if row.get("schema_version") != RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION:
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
    observation_values = observation.get("values")
    if not isinstance(observation_values, list) or len(observation_values) != (
        ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE
    ):
        raise RecurrentCounterfactualBranchError(
            "current public observation does not contain the safe 541 features"
        )
    _validate_numeric_vector(
        observation_values,
        expected_length=ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        field="current public observation",
    )
    mask = _complete_action_mask(context.get("current_public_action_mask"))
    feedback = _mapping(
        context.get("previous_public_feedback"),
        field="previous_public_feedback",
    )
    feedback_values = feedback.get("values")
    if not isinstance(feedback_values, list) or len(feedback_values) != (
        PREVIOUS_PUBLIC_FEEDBACK_SIZE
    ):
        raise RecurrentCounterfactualBranchError(
            "previous public feedback vector size drifted"
        )
    _validate_numeric_vector(
        feedback_values,
        expected_length=PREVIOUS_PUBLIC_FEEDBACK_SIZE,
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
    state_values = _flatten_numeric_tree(
        optimizer_context.get("source_recurrent_state"),
        field="source recurrent state",
    )
    expected_state_size = math.prod(state_shape)
    if len(state_values) != expected_state_size:
        raise RecurrentCounterfactualBranchError(
            "source recurrent state does not match its declared shape"
        )
    if not _valid_sha256(optimizer_context.get("source_recurrent_state_sha256")):
        raise RecurrentCounterfactualBranchError(
            "source recurrent state digest is malformed"
        )

    labels = _mapping(row.get("labels"), field="labels")
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
    if labels.get("valid_action_count") != len(expected_actions):
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
        action = _stable_action(parsed.get("action"), field="action outcome action")
        if parsed.get("natural_requested_action") != source_action:
            raise RecurrentCounterfactualBranchError(
                "action branch natural selection drifted from the source action"
            )
        first_transition = _mapping(
            parsed.get("first_transition"),
            field="action outcome first transition",
        )
        if first_transition.get("requested_action") != action:
            raise RecurrentCounterfactualBranchError(
                "action branch did not request its enumerated action"
            )
        if first_transition.get("observation_action_valid") is not True:
            raise RecurrentCounterfactualBranchError(
                "action branch was not valid in the tick-start public mask"
            )
        if parsed.get("intervention_count") != 1:
            raise RecurrentCounterfactualBranchError(
                "every action outcome must contain exactly one intervention"
            )
        if parsed.get("heuristic_action_source_count") != 0:
            raise RecurrentCounterfactualBranchError(
                "counterfactual continuation used a heuristic action source"
            )
        if parsed.get("unsupported_requested_action_count") != 0:
            raise RecurrentCounterfactualBranchError(
                "counterfactual continuation requested an unsupported action"
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
    baseline = _mapping(labels.get("baseline"), field="labels.baseline")
    source_outcome = next(
        _mapping(outcome, field="source action outcome")
        for outcome in outcomes
        if _mapping(outcome, field="action outcome").get("action") == source_action
    )
    if source_outcome.get("behavior_digest") != baseline.get("behavior_digest"):
        raise RecurrentCounterfactualBranchError(
            "source-action behavior digest does not match the baseline"
        )
    if contract.get("focal_return_record_policy") != (
        "all_focal_trajectory_records_including_passive_terminal"
    ):
        raise RecurrentCounterfactualBranchError(
            "counterfactual focal-return record policy drifted"
        )

    metadata = _mapping(row.get("metadata"), field="metadata")
    if set(metadata) != _METADATA_KEYS:
        raise RecurrentCounterfactualBranchError("metadata field set drifted")
    role = str(metadata.get("seed_role", ""))
    seed = _positive_int(metadata.get("environment_seed"), field="environment_seed")
    _validated_seed_role(role, seed)
    if metadata.get("private_checkpoint_serialized") is not False:
        raise RecurrentCounterfactualBranchError(
            "private world checkpoint cannot be serialized in a branch row"
        )
    if metadata.get("scenario") not in RECURRENT_COUNTERFACTUAL_SCENARIOS:
        raise RecurrentCounterfactualBranchError("metadata scenario is unsupported")
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
    if continuation.get("continuation_policy_sampling_seed") != (policy_sampling_seed):
        raise RecurrentCounterfactualBranchError(
            "continuation policy sampling seed drifted"
        )
    if continuation.get("branch_horizon_ticks") != horizon_ticks:
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
    if continuation.get("repeat_indices") != list(range(repeat_count)):
        raise RecurrentCounterfactualBranchError("continuation repeat indices drifted")
    if continuation.get("repeat_policy_sampling_seeds") != [
        policy_sampling_seed for _ in range(repeat_count)
    ]:
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
    if not _valid_sha256(metadata.get("branch_identity_digest")):
        raise RecurrentCounterfactualBranchError("branch identity digest is malformed")
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
    forced_action: str | None,
    gamma: float,
    prefix_horizons: Sequence[int] | None = None,
) -> dict[str, object]:
    world = deepcopy(checkpoint_world)
    _configure_manual_world(world)
    record_start = len(world.trajectory_records)
    diagnostics_start = len(world.policy_decision_diagnostics_records)
    wrapper: OneShotRecurrentCounterfactualPolicy | None = None
    if forced_action is not None:
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
    )
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
            )
            for prefix_horizon in resolved_prefix_horizons
        }
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
    expected_first_action = forced_action or source_action
    if first_focal.get("requested_action") != expected_first_action:
        raise RecurrentCounterfactualBranchError(
            "focal continuation requested an unexpected first action"
        )

    if forced_action is None:
        natural_action = _stable_action(
            first_focal.get("requested_action"),
            field="baseline first requested action",
        )
    if natural_action != source_action:
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


def _counterfactual_contract(*, verify_replay: bool) -> dict[str, object]:
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
        "allowed_seed_roles": list(RECURRENT_COUNTERFACTUAL_SEED_ROLES),
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
            "counterfactual queries may use only train or curriculum seeds"
        )
    seed = _positive_int(environment_seed, field="environment_seed")
    if seed not in RECURRENT_SEED_REGISTRY[role]:
        raise RecurrentCounterfactualBranchError(
            f"environment_seed is not registered for seed role {role!r}"
        )
    return role


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


def _flatten_numeric_tree(value: object, *, field: str) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for nested in value:
            flattened.extend(_flatten_numeric_tree(nested, field=field))
        return flattened
    return [_finite_float(value, field=field)]


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
    "RECURRENT_COUNTERFACTUAL_BRANCH_PROTOCOL_VERSION",
    "RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION",
    "RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO",
    "RECURRENT_COUNTERFACTUAL_SCENARIOS",
    "RECURRENT_COUNTERFACTUAL_SEED_ROLES",
    "RECURRENT_COUNTERFACTUAL_TRAINING_USE",
    "RecurrentCounterfactualBranchError",
    "build_recurrent_counterfactual_branch_row",
    "build_recurrent_counterfactual_nested_horizon_materialization",
    "reconstruct_current_model_hidden_from_branch_row",
    "validate_recurrent_counterfactual_branch_row",
    "verified_source_recurrent_state_from_branch_row",
]
