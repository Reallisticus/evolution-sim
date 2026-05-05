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
    conditional_action_scores: dict[str, dict[str, float]] | None = None
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
        action_scores = self._scores_for(observation, action_mask)
        learned_action, learned_score = self._best_scored_action(
            action_scores,
            action_mask,
        )
        if self.heuristic_guard:
            heuristic_action = ObservationHeuristicPolicy().decide(
                observation,
                action_mask,
            )
            heuristic_score = float(
                action_scores.get(heuristic_action.requested_action, 0.0)
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
                    diagnostics={
                        "guard_used": True,
                        "learned_action": learned_action,
                        "learned_score": learned_score,
                        "heuristic_action": heuristic_action.requested_action,
                        "heuristic_score": heuristic_score,
                    },
                )
        return self._decision(
            learned_action,
            source=self.policy_id,
            diagnostics={
                "guard_used": False,
                "learned_action": learned_action,
                "learned_score": learned_score,
            },
        )

    def _best_scored_action(
        self,
        action_scores: dict[str, float],
        action_mask: dict[str, bool],
    ) -> tuple[str, float]:
        best_action = self.fallback_action if action_mask.get(self.fallback_action, False) else "stay"
        best_score = float("-inf")
        for action in sorted(action_mask):
            if not bool(action_mask[action]):
                continue
            score = float(action_scores.get(action, 0.0))
            if score > best_score:
                best_score = score
                best_action = action
        return best_action, best_score

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

    def _scores_for(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> dict[str, float]:
        if self.conditional_action_scores:
            for feature_key in feature_keys_from_observation(observation, action_mask):
                scores = self.conditional_action_scores.get(feature_key)
                if scores is not None:
                    return scores
        return self.action_scores


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
    conditional_action_scores = model.get("conditional_action_scores")
    heuristic_guard_policy = model.get("heuristic_guard_policy")
    heuristic_confidence_threshold = model.get("heuristic_confidence_threshold")
    heuristic_override_min_margin = model.get("heuristic_override_min_margin")
    return LearnedPolicy(
        action_scores={
            str(action): float(score)
            for action, score in action_scores.items()
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        },
        conditional_action_scores=_parse_conditional_action_scores(
            conditional_action_scores
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
