from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evolution_sim.env.runtime.policy import ActionDecision, ObservationHeuristicPolicy
from evolution_sim.mind.artifacts import load_model_artifact
from evolution_sim.mind.feature_policy import feature_keys_from_observation
from evolution_sim.mind.neural import (
    CompiledNeuralActorCriticNetwork,
    NEURAL_ACTOR_PRIOR_POLICIES,
    compile_neural_actor_critic_network,
    is_neural_actor_critic_model_type,
    score_neural_actor_critic,
)

LEARNED_POLICY_ID = "mind_v1_learned_policy"
NEURAL_POLICY_ID = "mind_v2_neural_policy"
HEURISTIC_GUARD_POLICY = "observation_heuristic_safety_floor_v1"
HEURISTIC_DELEGATE_POLICY = "observation_heuristic_confidence_delegate_v1"
VALUE_SUPPORTED_DEVIATION_POLICY = "positive_value_safe_deviation_v1"
NEURAL_SCORE_NORMALIZATION_POLICY = "mask_renormalized_neural_actor_scores_v1"


@dataclass(frozen=True, slots=True)
class LearnedPolicy:
    action_scores: dict[str, float]
    action_score_metadata: dict[str, object] | None = None
    conditional_action_scores: dict[str, dict[str, float]] | None = None
    conditional_action_metadata: dict[str, dict[str, object]] | None = None
    action_value_estimates: dict[str, float] | None = None
    conditional_action_value_estimates: dict[str, dict[str, float]] | None = None
    neural_network: (
        dict[str, object] | CompiledNeuralActorCriticNetwork | None
    ) = None
    policy_id: str = LEARNED_POLICY_ID
    policy_version: str = "mind_v1_learned_policy_v1"
    fallback_action: str = "stay"
    heuristic_guard: bool = False
    heuristic_confidence_threshold: float | None = None
    heuristic_override_min_margin: float | None = None
    heuristic_delegate: bool = False
    heuristic_delegate_max_training_score_margin: float | None = None
    heuristic_safe_local_eat_min_score: float | None = None
    heuristic_safe_local_eat_min_food: float | None = None
    heuristic_safe_local_eat_min_plant_ratio: float | None = None
    heuristic_safe_plant_move_min_score: float | None = None
    heuristic_safe_plant_move_min_strength: float | None = None
    heuristic_safe_plant_move_max_local_food_ratio: float | None = None
    heuristic_safe_plant_move_max_distance: int | None = None
    value_supported_deviation_policy: str | None = None
    value_supported_deviation_min_support: int | None = None
    value_supported_deviation_min_value_margin: float | None = None
    value_supported_deviation_min_learned_value: float | None = None
    value_supported_deviation_min_score_margin: float | None = None
    value_supported_deviation_min_predicted_advantage: float | None = None
    neural_actor_prior_policy: str | None = None
    neural_actor_prior_blend_weight: float | None = None

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        score_match = self._score_match_for(observation, action_mask)
        (
            learned_action,
            learned_score,
            learned_runner_up_score,
            learned_score_margin,
        ) = self._best_scored_action(
            score_match.scores,
            action_mask,
        )
        if self.heuristic_guard:
            heuristic_action = ObservationHeuristicPolicy().decide(
                observation,
                action_mask,
            )
            heuristic_score = float(
                score_match.scores.get(heuristic_action.requested_action, 0.0)
            )
            learned_action_value = _action_value(score_match, learned_action)
            heuristic_action_value = _action_value(
                score_match,
                heuristic_action.requested_action,
            )
            learned_action_predicted_advantage = _predicted_advantage(
                score_match,
                learned_action_value,
            )
            value_supported_deviation_reason = _value_supported_deviation_reason(
                score_match=score_match,
                learned_action=learned_action,
                learned_score_margin=learned_score_margin,
                heuristic_action=heuristic_action.requested_action,
                heuristic_source=heuristic_action.source,
                observation=observation,
                policy=self.value_supported_deviation_policy,
                min_support=self.value_supported_deviation_min_support,
                min_value_margin=(
                    self.value_supported_deviation_min_value_margin
                ),
                min_learned_value=(
                    self.value_supported_deviation_min_learned_value
                ),
                min_score_margin=(
                    self.value_supported_deviation_min_score_margin
                ),
                min_predicted_advantage=(
                    self.value_supported_deviation_min_predicted_advantage
                ),
            )
            heuristic_delegate_reason = _heuristic_delegate_reason(
                score_match=score_match,
                learned_action=learned_action,
                heuristic_action=heuristic_action.requested_action,
                max_training_score_margin=(
                    self.heuristic_delegate_max_training_score_margin
                    if self.heuristic_delegate
                    else None
                ),
            )
            if (
                heuristic_delegate_reason is not None
                and value_supported_deviation_reason is None
            ):
                return self._decision(
                    heuristic_action.requested_action,
                    source=f"{self.policy_id}:{HEURISTIC_DELEGATE_POLICY}",
                    diagnostics=_decision_diagnostics(
                        score_match=score_match,
                        guard_used=False,
                        learned_action=learned_action,
                        learned_score=learned_score,
                        learned_runner_up_score=learned_runner_up_score,
                        learned_score_margin=learned_score_margin,
                        heuristic_action=heuristic_action.requested_action,
                        heuristic_score=heuristic_score,
                        learned_action_value=learned_action_value,
                        heuristic_action_value=heuristic_action_value,
                        learned_action_predicted_advantage=(
                            learned_action_predicted_advantage
                        ),
                        heuristic_delegate_used=True,
                        heuristic_delegate_reason=heuristic_delegate_reason,
                    ),
                )
            safe_deviation_reason = value_supported_deviation_reason
            if safe_deviation_reason is None:
                safe_deviation_reason = _safe_deviation_reason(
                    learned_action=learned_action,
                    learned_score=learned_score,
                    learned_score_margin=learned_score_margin,
                    heuristic_action=heuristic_action.requested_action,
                    observation=observation,
                    local_eat_min_score=self.heuristic_safe_local_eat_min_score,
                    local_eat_min_food=self.heuristic_safe_local_eat_min_food,
                    local_eat_min_plant_ratio=(
                        self.heuristic_safe_local_eat_min_plant_ratio
                    ),
                    plant_move_min_score=self.heuristic_safe_plant_move_min_score,
                    plant_move_min_strength=(
                        self.heuristic_safe_plant_move_min_strength
                    ),
                    plant_move_max_local_food_ratio=(
                        self.heuristic_safe_plant_move_max_local_food_ratio
                    ),
                    plant_move_max_distance=(
                        self.heuristic_safe_plant_move_max_distance
                    ),
                )
            if safe_deviation_reason is None and _guard_should_use_heuristic(
                learned_action=learned_action,
                learned_score=learned_score,
                heuristic_action=heuristic_action.requested_action,
                heuristic_score=heuristic_score,
                heuristic_source=heuristic_action.source,
                observation=observation,
                confidence_threshold=self.heuristic_confidence_threshold,
                override_min_margin=self.heuristic_override_min_margin,
            ):
                return self._decision(
                    heuristic_action.requested_action,
                    source=f"{self.policy_id}:{HEURISTIC_GUARD_POLICY}",
                    diagnostics=_decision_diagnostics(
                        score_match=score_match,
                        guard_used=True,
                        learned_action=learned_action,
                        learned_score=learned_score,
                        learned_runner_up_score=learned_runner_up_score,
                        learned_score_margin=learned_score_margin,
                        heuristic_action=heuristic_action.requested_action,
                        heuristic_score=heuristic_score,
                        learned_action_value=learned_action_value,
                        heuristic_action_value=heuristic_action_value,
                        learned_action_predicted_advantage=(
                            learned_action_predicted_advantage
                        ),
                        heuristic_delegate_used=False,
                        safe_deviation_used=False,
                    ),
                )
            if safe_deviation_reason is not None:
                return self._decision(
                    learned_action,
                    source=self.policy_id,
                    diagnostics=_decision_diagnostics(
                        score_match=score_match,
                        guard_used=False,
                        learned_action=learned_action,
                        learned_score=learned_score,
                        learned_runner_up_score=learned_runner_up_score,
                        learned_score_margin=learned_score_margin,
                        heuristic_action=heuristic_action.requested_action,
                        heuristic_score=heuristic_score,
                        learned_action_value=learned_action_value,
                        heuristic_action_value=heuristic_action_value,
                        learned_action_predicted_advantage=(
                            learned_action_predicted_advantage
                        ),
                        heuristic_delegate_used=False,
                        safe_deviation_used=True,
                        safe_deviation_reason=safe_deviation_reason,
                    ),
                )
        return self._decision(
            learned_action,
            source=self.policy_id,
            diagnostics=_decision_diagnostics(
                score_match=score_match,
                guard_used=False,
                learned_action=learned_action,
                learned_score=learned_score,
                learned_runner_up_score=learned_runner_up_score,
                learned_score_margin=learned_score_margin,
                heuristic_delegate_used=False,
            ),
        )

    def _best_scored_action(
        self,
        action_scores: dict[str, float],
        action_mask: dict[str, bool],
    ) -> tuple[str, float, float, float]:
        best_action = (
            self.fallback_action
            if action_mask.get(self.fallback_action, False)
            else "stay"
        )
        best_score = float("-inf")
        runner_up_score = float("-inf")
        for action in sorted(action_mask):
            if not bool(action_mask[action]):
                continue
            score = float(action_scores.get(action, 0.0))
            if score > best_score:
                runner_up_score = best_score
                best_score = score
                best_action = action
            elif score > runner_up_score:
                runner_up_score = score
        if best_score == float("-inf"):
            best_score = 0.0
        if runner_up_score == float("-inf"):
            runner_up_score = 0.0
        return best_action, best_score, runner_up_score, best_score - runner_up_score

    def _decision(
        self,
        requested_action: str,
        *,
        source: str,
        diagnostics: dict[str, object] | None = None,
    ) -> ActionDecision:
        return ActionDecision(
            requested_action=requested_action,
            source=source,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            diagnostics=diagnostics,
        )

    def _score_match_for(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> _ScoreMatch:
        if self.neural_network is not None:
            scores, action_values, state_value = score_neural_actor_critic(
                network=self.neural_network,
                observation=observation,
            )
            scores = _mask_renormalized_scores(scores, action_mask)
            score_source = "neural_actor_critic"
            feature_key = None
            match_depth = None
            support = _metadata_int(self.action_score_metadata, "record_count")
            prior_match = self._contextual_prior_match_for(
                observation,
                action_mask,
            )
            if (
                self.neural_actor_prior_policy in NEURAL_ACTOR_PRIOR_POLICIES
                and self.neural_actor_prior_blend_weight is not None
                and self.neural_actor_prior_blend_weight > 0.0
            ):
                scores = _blend_action_scores(
                    learned_scores=scores,
                    prior_scores=_mask_renormalized_scores(
                        prior_match.scores,
                        action_mask,
                    ),
                    prior_weight=self.neural_actor_prior_blend_weight,
                )
                score_source = (
                    f"neural_actor_critic:{self.neural_actor_prior_policy}"
                )
                feature_key = prior_match.feature_key
                match_depth = prior_match.match_depth
                support = prior_match.support
            return _ScoreMatch(
                scores=scores,
                score_source=score_source,
                feature_key=feature_key,
                match_depth=match_depth,
                support=support,
                training_score_margin=_live_score_margin(scores, action_mask),
                value_estimates=action_values,
                state_value=state_value,
            )
        return self._contextual_prior_match_for(observation, action_mask)

    def _contextual_prior_match_for(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> _ScoreMatch:
        if self.conditional_action_scores:
            for depth, feature_key in enumerate(
                feature_keys_from_observation(observation, action_mask)
            ):
                scores = self.conditional_action_scores.get(feature_key)
                if scores is not None:
                    metadata = _metadata_for_key(
                        self.conditional_action_metadata,
                        feature_key,
                    )
                    return _ScoreMatch(
                        scores=scores,
                        score_source="conditional",
                        feature_key=feature_key,
                        match_depth=depth,
                        support=_metadata_int(metadata, "record_count"),
                        training_score_margin=_training_score_margin(metadata),
                        value_estimates=_values_for_key(
                            self.conditional_action_value_estimates,
                            feature_key,
                        ),
                        state_value=None,
                    )
        return _ScoreMatch(
            scores=self.action_scores,
            score_source="global",
            feature_key=None,
            match_depth=None,
            support=_metadata_int(self.action_score_metadata, "record_count"),
            training_score_margin=_training_score_margin(
                self.action_score_metadata
            ),
            value_estimates=self.action_value_estimates,
            state_value=None,
        )


def load_learned_policy(
    artifact_path: str | Path,
    *,
    enable_mind: bool = False,
) -> LearnedPolicy:
    artifact = load_model_artifact(artifact_path, enable_mind=enable_mind)
    manifest = artifact["manifest"]
    if not isinstance(manifest, dict):
        raise ValueError("learned policy artifact manifest must be an object")
    model = artifact["model"]
    if not isinstance(model, dict):
        raise ValueError("learned policy artifact model must be an object")
    action_scores = model.get("action_scores")
    if not isinstance(action_scores, dict):
        raise ValueError("learned policy artifact model is missing action_scores")
    fallback_action = model.get("fallback_action", "stay")
    action_score_metadata = model.get("action_score_metadata")
    conditional_action_scores = model.get("conditional_action_scores")
    conditional_action_metadata = model.get("conditional_action_metadata")
    action_value_estimates = model.get("action_value_estimates")
    conditional_action_value_estimates = model.get(
        "conditional_action_value_estimates"
    )
    neural_network = model.get("neural_network")
    heuristic_guard_policy = model.get("heuristic_guard_policy")
    heuristic_confidence_threshold = model.get("heuristic_confidence_threshold")
    heuristic_override_min_margin = model.get("heuristic_override_min_margin")
    heuristic_delegate_policy = model.get("heuristic_delegate_policy")
    heuristic_delegate_max_training_score_margin = model.get(
        "heuristic_delegate_max_training_score_margin"
    )
    heuristic_safe_local_eat_min_score = model.get(
        "heuristic_safe_local_eat_min_score"
    )
    heuristic_safe_local_eat_min_food = model.get(
        "heuristic_safe_local_eat_min_food"
    )
    heuristic_safe_local_eat_min_plant_ratio = model.get(
        "heuristic_safe_local_eat_min_plant_ratio"
    )
    heuristic_safe_plant_move_min_score = model.get(
        "heuristic_safe_plant_move_min_score"
    )
    heuristic_safe_plant_move_min_strength = model.get(
        "heuristic_safe_plant_move_min_strength"
    )
    heuristic_safe_plant_move_max_local_food_ratio = model.get(
        "heuristic_safe_plant_move_max_local_food_ratio"
    )
    heuristic_safe_plant_move_max_distance = model.get(
        "heuristic_safe_plant_move_max_distance"
    )
    value_supported_deviation_policy = model.get("value_supported_deviation_policy")
    value_supported_deviation_min_support = model.get(
        "value_supported_deviation_min_support"
    )
    value_supported_deviation_min_value_margin = model.get(
        "value_supported_deviation_min_value_margin"
    )
    value_supported_deviation_min_learned_value = model.get(
        "value_supported_deviation_min_learned_value"
    )
    value_supported_deviation_min_score_margin = model.get(
        "value_supported_deviation_min_score_margin"
    )
    value_supported_deviation_min_predicted_advantage = model.get(
        "value_supported_deviation_min_predicted_advantage"
    )
    neural_actor_prior_policy = model.get("neural_actor_prior_policy")
    neural_actor_prior_blend_weight = model.get("neural_actor_prior_blend_weight")
    model_type = manifest.get("model_type")
    is_neural_policy = is_neural_actor_critic_model_type(model_type)
    return LearnedPolicy(
        action_scores={
            str(action): float(score)
            for action, score in action_scores.items()
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        },
        action_score_metadata=_parse_score_metadata(action_score_metadata),
        conditional_action_scores=_parse_conditional_action_scores(
            conditional_action_scores
        ),
        conditional_action_metadata=_parse_conditional_action_metadata(
            conditional_action_metadata
        ),
        action_value_estimates=_parse_action_values(action_value_estimates),
        conditional_action_value_estimates=_parse_conditional_action_values(
            conditional_action_value_estimates
        ),
        neural_network=(
            compile_neural_actor_critic_network(neural_network)
            if is_neural_policy and isinstance(neural_network, dict)
            else None
        ),
        policy_id=NEURAL_POLICY_ID if is_neural_policy else LEARNED_POLICY_ID,
        policy_version=(
            "mind_v2_neural_policy_v1"
            if is_neural_policy
            else "mind_v1_learned_policy_v1"
        ),
        fallback_action=str(fallback_action),
        heuristic_guard=heuristic_guard_policy == HEURISTIC_GUARD_POLICY,
        heuristic_confidence_threshold=_optional_float(
            heuristic_confidence_threshold
        ),
        heuristic_override_min_margin=_optional_float(
            heuristic_override_min_margin
        ),
        heuristic_delegate=heuristic_delegate_policy == HEURISTIC_DELEGATE_POLICY,
        heuristic_delegate_max_training_score_margin=_optional_float(
            heuristic_delegate_max_training_score_margin
        ),
        heuristic_safe_local_eat_min_score=_optional_float(
            heuristic_safe_local_eat_min_score
        ),
        heuristic_safe_local_eat_min_food=_optional_float(
            heuristic_safe_local_eat_min_food
        ),
        heuristic_safe_local_eat_min_plant_ratio=_optional_float(
            heuristic_safe_local_eat_min_plant_ratio
        ),
        heuristic_safe_plant_move_min_score=_optional_float(
            heuristic_safe_plant_move_min_score
        ),
        heuristic_safe_plant_move_min_strength=_optional_float(
            heuristic_safe_plant_move_min_strength
        ),
        heuristic_safe_plant_move_max_local_food_ratio=_optional_float(
            heuristic_safe_plant_move_max_local_food_ratio
        ),
        heuristic_safe_plant_move_max_distance=_optional_int(
            heuristic_safe_plant_move_max_distance
        ),
        value_supported_deviation_policy=(
            str(value_supported_deviation_policy)
            if value_supported_deviation_policy is not None
            else None
        ),
        value_supported_deviation_min_support=_optional_int(
            value_supported_deviation_min_support
        ),
        value_supported_deviation_min_value_margin=_optional_float(
            value_supported_deviation_min_value_margin
        ),
        value_supported_deviation_min_learned_value=_optional_float(
            value_supported_deviation_min_learned_value
        ),
        value_supported_deviation_min_score_margin=_optional_float(
            value_supported_deviation_min_score_margin
        ),
        value_supported_deviation_min_predicted_advantage=_optional_float(
            value_supported_deviation_min_predicted_advantage
        ),
        neural_actor_prior_policy=(
            str(neural_actor_prior_policy)
            if neural_actor_prior_policy is not None
            else None
        ),
        neural_actor_prior_blend_weight=_optional_float(
            neural_actor_prior_blend_weight
        ),
    )


def _parse_conditional_action_scores(
    payload: object,
) -> dict[str, dict[str, float]] | None:
    if not isinstance(payload, dict):
        return None
    parsed: dict[str, dict[str, float]] = {}
    for feature_key, scores in payload.items():
        if not isinstance(scores, dict):
            continue
        parsed[str(feature_key)] = {
            str(action): float(score)
            for action, score in scores.items()
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        }
    return parsed


def _parse_action_values(payload: object) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    return {
        str(action): float(value)
        for action, value in payload.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _parse_conditional_action_values(
    payload: object,
) -> dict[str, dict[str, float]] | None:
    if not isinstance(payload, dict):
        return None
    parsed: dict[str, dict[str, float]] = {}
    for feature_key, values in payload.items():
        parsed_values = _parse_action_values(values)
        if parsed_values is not None:
            parsed[str(feature_key)] = parsed_values
    return parsed


@dataclass(frozen=True, slots=True)
class _ScoreMatch:
    scores: dict[str, float]
    score_source: str
    feature_key: str | None
    match_depth: int | None
    support: int | None
    training_score_margin: float | None
    value_estimates: dict[str, float] | None
    state_value: float | None


def _decision_diagnostics(
    *,
    score_match: _ScoreMatch,
    guard_used: bool,
    learned_action: str,
    learned_score: float,
    learned_runner_up_score: float,
    learned_score_margin: float,
    heuristic_action: str | None = None,
    heuristic_score: float | None = None,
    learned_action_value: float | None = None,
    heuristic_action_value: float | None = None,
    learned_action_predicted_advantage: float | None = None,
    heuristic_delegate_used: bool = False,
    heuristic_delegate_reason: str | None = None,
    safe_deviation_used: bool = False,
    safe_deviation_reason: str | None = None,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "guard_used": guard_used,
        "heuristic_delegate_used": heuristic_delegate_used,
        "safe_deviation_used": safe_deviation_used,
        "learned_action": learned_action,
        "learned_score": learned_score,
        "learned_runner_up_score": learned_runner_up_score,
        "learned_score_margin": learned_score_margin,
        "score_source": score_match.score_source,
    }
    if score_match.feature_key is not None:
        diagnostics["score_feature_key"] = score_match.feature_key
    if score_match.match_depth is not None:
        diagnostics["score_match_depth"] = score_match.match_depth
    if score_match.support is not None:
        diagnostics["score_support"] = score_match.support
    if score_match.training_score_margin is not None:
        diagnostics["training_score_margin"] = score_match.training_score_margin
    if score_match.state_value is not None:
        diagnostics["state_value_estimate"] = score_match.state_value
    if heuristic_action is not None:
        diagnostics["heuristic_action"] = heuristic_action
    if heuristic_score is not None:
        diagnostics["heuristic_score"] = heuristic_score
    if learned_action_value is not None:
        diagnostics["learned_action_value"] = learned_action_value
    if heuristic_action_value is not None:
        diagnostics["heuristic_action_value"] = heuristic_action_value
    if learned_action_predicted_advantage is not None:
        diagnostics["learned_action_predicted_advantage"] = (
            learned_action_predicted_advantage
        )
    if heuristic_delegate_reason is not None:
        diagnostics["heuristic_delegate_reason"] = heuristic_delegate_reason
    if safe_deviation_reason is not None:
        diagnostics["safe_deviation_reason"] = safe_deviation_reason
    return diagnostics


def _parse_score_metadata(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    parsed: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, str)):
            parsed[str(key)] = value
    return parsed


def _parse_conditional_action_metadata(
    payload: object,
) -> dict[str, dict[str, object]] | None:
    if not isinstance(payload, dict):
        return None
    parsed: dict[str, dict[str, object]] = {}
    for feature_key, metadata in payload.items():
        parsed_metadata = _parse_score_metadata(metadata)
        if parsed_metadata is not None:
            parsed[str(feature_key)] = parsed_metadata
    return parsed


def _metadata_for_key(
    payload: dict[str, dict[str, object]] | None,
    key: str,
) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    metadata = payload.get(key)
    if isinstance(metadata, dict):
        return metadata
    return None


def _values_for_key(
    payload: dict[str, dict[str, float]] | None,
    key: str,
) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    values = payload.get(key)
    if isinstance(values, dict):
        return values
    return None


def _metadata_int(
    payload: dict[str, object] | None,
    key: str,
) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _metadata_float(
    payload: dict[str, object] | None,
    key: str,
) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _training_score_margin(
    metadata: dict[str, object] | None,
) -> float | None:
    delegate_score_margin = _metadata_float(metadata, "delegate_score_margin")
    if delegate_score_margin is not None:
        return delegate_score_margin
    return _metadata_float(metadata, "score_margin")


def _live_score_margin(
    scores: dict[str, float],
    action_mask: dict[str, bool],
) -> float:
    best_score = float("-inf")
    runner_up_score = float("-inf")
    for action in sorted(action_mask):
        if not bool(action_mask[action]):
            continue
        score = float(scores.get(action, 0.0))
        if score > best_score:
            runner_up_score = best_score
            best_score = score
        elif score > runner_up_score:
            runner_up_score = score
    if best_score == float("-inf"):
        return 0.0
    if runner_up_score == float("-inf"):
        return float(best_score)
    return max(0.0, best_score - runner_up_score)


def _mask_renormalized_scores(
    scores: dict[str, float],
    action_mask: dict[str, bool],
) -> dict[str, float]:
    available_actions = [
        action
        for action, available in action_mask.items()
        if bool(available)
    ]
    total = sum(max(0.0, float(scores.get(action, 0.0))) for action in available_actions)
    if total <= 0.0:
        return dict(scores)
    normalized = {action: 0.0 for action in scores}
    for action in available_actions:
        normalized[action] = max(0.0, float(scores.get(action, 0.0))) / total
    return normalized


def _blend_action_scores(
    *,
    learned_scores: dict[str, float],
    prior_scores: dict[str, float],
    prior_weight: float,
) -> dict[str, float]:
    clamped_prior_weight = max(0.0, min(1.0, prior_weight))
    learned_weight = 1.0 - clamped_prior_weight
    actions = set(learned_scores) | set(prior_scores)
    return {
        action: (
            learned_weight * float(learned_scores.get(action, 0.0))
            + clamped_prior_weight * float(prior_scores.get(action, 0.0))
        )
        for action in actions
    }


def _heuristic_delegate_reason(
    *,
    score_match: _ScoreMatch,
    learned_action: str,
    heuristic_action: str,
    max_training_score_margin: float | None,
) -> str | None:
    if max_training_score_margin is None or learned_action == heuristic_action:
        return None
    training_margin = score_match.training_score_margin
    if training_margin is None or training_margin < max_training_score_margin:
        return "low_confidence_action_prior"
    return None


def _action_value(
    score_match: _ScoreMatch,
    action: str,
) -> float | None:
    if score_match.value_estimates is None:
        return None
    value = score_match.value_estimates.get(action)
    if value is None:
        return None
    return float(value)


def _predicted_advantage(
    score_match: _ScoreMatch,
    learned_action_value: float | None,
) -> float | None:
    if learned_action_value is None or score_match.state_value is None:
        return None
    return learned_action_value - float(score_match.state_value)


def _value_supported_deviation_reason(
    *,
    score_match: _ScoreMatch,
    learned_action: str,
    learned_score_margin: float,
    heuristic_action: str,
    heuristic_source: str,
    observation: dict[str, object],
    policy: str | None,
    min_support: int | None,
    min_value_margin: float | None,
    min_learned_value: float | None,
    min_score_margin: float | None,
    min_predicted_advantage: float | None,
) -> str | None:
    if policy != VALUE_SUPPORTED_DEVIATION_POLICY:
        return None
    if learned_action == heuristic_action:
        return None
    if (
        min_support is None
        or min_value_margin is None
        or min_learned_value is None
        or score_match.support is None
        or score_match.support < min_support
    ):
        return None
    learned_value = _action_value(score_match, learned_action)
    heuristic_value = _action_value(score_match, heuristic_action)
    if learned_value is None or heuristic_value is None:
        return None
    if learned_value < min_learned_value:
        return None
    if learned_value - heuristic_value < min_value_margin:
        return None
    if _value_supported_local_resource_eat_allowed(
        learned_action=learned_action,
        heuristic_action=heuristic_action,
        observation=observation,
    ):
        return "value_supported_local_resource_eat"
    if _value_supported_neural_resource_action_allowed(
        score_match=score_match,
        learned_action=learned_action,
        learned_score_margin=learned_score_margin,
        learned_value=learned_value,
        heuristic_action=heuristic_action,
        heuristic_source=heuristic_source,
        observation=observation,
        min_score_margin=min_score_margin,
        min_predicted_advantage=min_predicted_advantage,
    ):
        return "value_supported_neural_resource_action"
    return None


def _value_supported_local_resource_eat_allowed(
    *,
    learned_action: str,
    heuristic_action: str,
    observation: dict[str, object],
) -> bool:
    if learned_action != "eat":
        return False
    if heuristic_action != "stay":
        return False
    if not _has_safe_deviation_vitals(observation):
        return False
    center = _center_patch_cell(observation.get("local_patch"))
    if not center:
        return False
    if _ratio(center.get("hazard_level"), default=1.0) > 0.4:
        return False
    center_food = _ratio(center.get("food"), default=0.0)
    center_animal_food = max(
        _ratio(center.get("fresh_kill_energy"), default=0.0),
        _ratio(center.get("carcass_energy"), default=0.0),
    )
    meat_mode = _meat_mode(observation)
    if meat_mode in {"hunter", "scavenger"}:
        return center_animal_food > 0.0
    if meat_mode == "mixed" and center_animal_food > 0.0:
        return True
    return center_food > 0.0


def _value_supported_neural_resource_action_allowed(
    *,
    score_match: _ScoreMatch,
    learned_action: str,
    learned_score_margin: float,
    learned_value: float,
    heuristic_action: str,
    heuristic_source: str,
    observation: dict[str, object],
    min_score_margin: float | None,
    min_predicted_advantage: float | None,
) -> bool:
    if score_match.score_source != "neural_actor_critic":
        return False
    if min_score_margin is None or min_predicted_advantage is None:
        return False
    if learned_score_margin < min_score_margin:
        return False
    if score_match.state_value is None:
        return False
    if learned_value - float(score_match.state_value) < min_predicted_advantage:
        return False
    if heuristic_source == "heuristic_observation_conserve":
        return False
    if heuristic_action.startswith("attack_"):
        return False
    if not _has_safe_deviation_vitals(observation):
        return False
    center = _center_patch_cell(observation.get("local_patch"))
    if not center:
        return False
    if _ratio(center.get("hazard_level"), default=1.0) > 0.4:
        return False
    if learned_action != "eat":
        return False
    if heuristic_action == "drink":
        return False
    return _has_local_food_for_diet(observation, center)


def _has_local_food_for_diet(
    observation: dict[str, object],
    center: dict[str, object],
) -> bool:
    center_food = _ratio(center.get("food"), default=0.0)
    center_animal_food = max(
        _ratio(center.get("fresh_kill_energy"), default=0.0),
        _ratio(center.get("carcass_energy"), default=0.0),
    )
    meat_mode = _meat_mode(observation)
    if meat_mode in {"hunter", "scavenger"}:
        return center_animal_food > 0.0
    if meat_mode == "mixed" and center_animal_food > 0.0:
        return True
    return center_food > 0.0


def _guard_should_use_heuristic(
    *,
    learned_action: str,
    learned_score: float,
    heuristic_action: str,
    heuristic_score: float,
    heuristic_source: str,
    observation: dict[str, object],
    confidence_threshold: float | None,
    override_min_margin: float | None,
) -> bool:
    if learned_action == heuristic_action:
        return False
    if heuristic_source == "heuristic_observation_conserve":
        return True
    if confidence_threshold is not None and learned_score < confidence_threshold:
        return True
    if (
        override_min_margin is not None
        and learned_score - heuristic_score < override_min_margin
    ):
        return True
    if heuristic_action != "stay":
        return True
    if learned_action.startswith("attack_"):
        return True
    self_state = observation.get("self")
    if not isinstance(self_state, dict):
        return False
    energy_ratio = _ratio(self_state.get("energy_ratio"), default=1.0)
    hydration_ratio = _ratio(self_state.get("hydration_ratio"), default=1.0)
    health_ratio = _ratio(self_state.get("health_ratio"), default=1.0)
    return (
        (learned_action == "eat" and (energy_ratio >= 0.82 or hydration_ratio < 0.58))
        or (
            learned_action.startswith("move_")
            and (
                energy_ratio < 0.62
                or hydration_ratio < 0.58
                or health_ratio < 0.72
            )
        )
        or hydration_ratio < 0.34
        or health_ratio < 0.42
    )


def _safe_deviation_reason(
    *,
    learned_action: str,
    learned_score: float,
    learned_score_margin: float,
    heuristic_action: str,
    observation: dict[str, object],
    local_eat_min_score: float | None,
    local_eat_min_food: float | None,
    local_eat_min_plant_ratio: float | None,
    plant_move_min_score: float | None,
    plant_move_min_strength: float | None,
    plant_move_max_local_food_ratio: float | None,
    plant_move_max_distance: int | None,
) -> str | None:
    if not _has_safe_deviation_vitals(observation):
        return None
    center = _center_patch_cell(observation.get("local_patch"))
    if not center:
        return None
    hazard_level = _ratio(center.get("hazard_level"), default=1.0)
    if hazard_level > 0.4:
        return None
    if _safe_local_eat_deviation_allowed(
        learned_action=learned_action,
        learned_score=learned_score,
        heuristic_action=heuristic_action,
        observation=observation,
        center=center,
        min_score=local_eat_min_score,
        min_food=local_eat_min_food,
        min_plant_ratio=local_eat_min_plant_ratio,
    ):
        return "local_resource_eat"
    if _safe_plant_move_deviation_allowed(
        learned_action=learned_action,
        learned_score=learned_score,
        learned_score_margin=learned_score_margin,
        heuristic_action=heuristic_action,
        observation=observation,
        center=center,
        min_score=plant_move_min_score,
        min_strength=plant_move_min_strength,
        max_local_food_ratio=plant_move_max_local_food_ratio,
        max_distance=plant_move_max_distance,
    ):
        return "stronger_plant_navigation"
    return None


def _has_safe_deviation_vitals(observation: dict[str, object]) -> bool:
    self_state = observation.get("self")
    if not isinstance(self_state, dict):
        return False
    energy_ratio = _ratio(self_state.get("energy_ratio"), default=0.0)
    hydration_ratio = _ratio(self_state.get("hydration_ratio"), default=0.0)
    health_ratio = _ratio(self_state.get("health_ratio"), default=0.0)
    return energy_ratio >= 0.5 and hydration_ratio >= 0.7 and health_ratio >= 0.75


def _safe_local_eat_deviation_allowed(
    *,
    learned_action: str,
    learned_score: float,
    heuristic_action: str,
    observation: dict[str, object],
    center: dict[str, object],
    min_score: float | None,
    min_food: float | None,
    min_plant_ratio: float | None,
) -> bool:
    if (
        min_score is None
        or min_food is None
        or min_plant_ratio is None
        or learned_score < min_score
    ):
        return False
    if learned_action != "eat" or not heuristic_action.startswith("move_"):
        return False
    center_food = _ratio(center.get("food"), default=0.0)
    has_local_resource = (
        center_food > 0.0
        or _ratio(center.get("fresh_kill_energy"), default=0.0) > 0.0
        or _ratio(center.get("carcass_energy"), default=0.0) > 0.0
    )
    if not has_local_resource:
        return False
    meat_mode = _meat_mode(observation)
    center_animal_food = max(
        _ratio(center.get("fresh_kill_energy"), default=0.0),
        _ratio(center.get("carcass_energy"), default=0.0),
    )
    if meat_mode in {"scavenger", "hunter"}:
        return center_animal_food > 0.0
    if meat_mode == "mixed" and center_animal_food > 0.0:
        return True
    return center_food >= max(
        min_food,
        min_plant_ratio * _navigation_strength(observation, "plant"),
    )


def _safe_plant_move_deviation_allowed(
    *,
    learned_action: str,
    learned_score: float,
    learned_score_margin: float,
    heuristic_action: str,
    observation: dict[str, object],
    center: dict[str, object],
    min_score: float | None,
    min_strength: float | None,
    max_local_food_ratio: float | None,
    max_distance: int | None,
) -> bool:
    if (
        min_score is None
        or min_strength is None
        or max_local_food_ratio is None
        or max_distance is None
    ):
        return False
    if learned_score < min_score or learned_score_margin < 0.0:
        return False
    if heuristic_action != "eat" or not learned_action.startswith("move_"):
        return False
    if _meat_mode(observation) != "none":
        return False
    plant_target = _navigation_target(observation, "plant")
    distance = _navigation_distance(plant_target)
    strength = _ratio(plant_target.get("strength"), default=0.0)
    center_food = _ratio(center.get("food"), default=0.0)
    return (
        0 < distance <= max_distance
        and strength >= min_strength
        and center_food < max_local_food_ratio * strength
    )


def _center_patch_cell(payload: object) -> dict[str, object]:
    if not isinstance(payload, list):
        return {}
    for cell in payload:
        if not isinstance(cell, dict):
            continue
        if cell.get("dx") == 0 and cell.get("dy") == 0:
            return cell
    return {}


def _meat_mode(observation: dict[str, object]) -> str:
    self_state = observation.get("self")
    if not isinstance(self_state, dict):
        return "unknown"
    return str(self_state.get("meat_mode", "unknown"))


def _navigation_target(observation: dict[str, object], key: str) -> dict[str, object]:
    navigation = observation.get("navigation")
    if not isinstance(navigation, dict):
        return {}
    target = navigation.get(key)
    if not isinstance(target, dict):
        return {}
    return target


def _navigation_strength(observation: dict[str, object], key: str) -> float:
    return _ratio(_navigation_target(observation, key).get("strength"), default=0.0)


def _navigation_distance(target: dict[str, object]) -> int:
    value = target.get("distance")
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _ratio(value: object, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
