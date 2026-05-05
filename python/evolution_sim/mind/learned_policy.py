from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evolution_sim.env.runtime.policy import ActionDecision, ObservationHeuristicPolicy
from evolution_sim.mind.artifacts import load_model_artifact
from evolution_sim.mind.feature_policy import feature_keys_from_observation

LEARNED_POLICY_ID = "mind_v1_learned_policy"
HEURISTIC_GUARD_POLICY = "observation_heuristic_safety_floor_v1"


@dataclass(frozen=True, slots=True)
class LearnedPolicy:
    action_scores: dict[str, float]
    action_score_metadata: dict[str, object] | None = None
    conditional_action_scores: dict[str, dict[str, float]] | None = None
    conditional_action_metadata: dict[str, dict[str, object]] | None = None
    policy_id: str = LEARNED_POLICY_ID
    policy_version: str = "mind_v1_learned_policy_v1"
    fallback_action: str = "stay"
    heuristic_guard: bool = False
    heuristic_confidence_threshold: float | None = None
    heuristic_override_min_margin: float | None = None

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
            if _guard_should_use_heuristic(
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
                        training_score_margin=_metadata_float(
                            metadata,
                            "score_margin",
                        ),
                    )
        return _ScoreMatch(
            scores=self.action_scores,
            score_source="global",
            feature_key=None,
            match_depth=None,
            support=_metadata_int(self.action_score_metadata, "record_count"),
            training_score_margin=_metadata_float(
                self.action_score_metadata,
                "score_margin",
            ),
        )


def load_learned_policy(
    artifact_path: str | Path,
    *,
    enable_mind: bool = False,
) -> LearnedPolicy:
    artifact = load_model_artifact(artifact_path, enable_mind=enable_mind)
    model = artifact["model"]
    assert isinstance(model, dict)
    action_scores = model.get("action_scores")
    if not isinstance(action_scores, dict):
        raise ValueError("learned policy artifact model is missing action_scores")
    fallback_action = model.get("fallback_action", "stay")
    action_score_metadata = model.get("action_score_metadata")
    conditional_action_scores = model.get("conditional_action_scores")
    conditional_action_metadata = model.get("conditional_action_metadata")
    heuristic_guard_policy = model.get("heuristic_guard_policy")
    heuristic_confidence_threshold = model.get("heuristic_confidence_threshold")
    heuristic_override_min_margin = model.get("heuristic_override_min_margin")
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
        fallback_action=str(fallback_action),
        heuristic_guard=heuristic_guard_policy == HEURISTIC_GUARD_POLICY,
        heuristic_confidence_threshold=_optional_float(
            heuristic_confidence_threshold
        ),
        heuristic_override_min_margin=_optional_float(
            heuristic_override_min_margin
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


@dataclass(frozen=True, slots=True)
class _ScoreMatch:
    scores: dict[str, float]
    score_source: str
    feature_key: str | None
    match_depth: int | None
    support: int | None
    training_score_margin: float | None


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
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "guard_used": guard_used,
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
    if heuristic_action is not None:
        diagnostics["heuristic_action"] = heuristic_action
    if heuristic_score is not None:
        diagnostics["heuristic_score"] = heuristic_score
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
