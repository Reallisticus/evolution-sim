from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.mind.artifacts import load_model_artifact

LEARNED_POLICY_ID = "mind_v1_learned_policy"


@dataclass(frozen=True, slots=True)
class LearnedPolicy:
    action_scores: dict[str, float]
    policy_id: str = LEARNED_POLICY_ID
    policy_version: str = "mind_v1_learned_policy_v1"
    fallback_action: str = "stay"

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        del observation
        best_action = self.fallback_action if action_mask.get(self.fallback_action, False) else "stay"
        best_score = float("-inf")
        for action in sorted(action_mask):
            if not bool(action_mask[action]):
                continue
            score = float(self.action_scores.get(action, 0.0))
            if score > best_score:
                best_score = score
                best_action = action
        return ActionDecision(
            requested_action=best_action,
            source=self.policy_id,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
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
    return LearnedPolicy(
        action_scores={
            str(action): float(score)
            for action, score in action_scores.items()
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        },
        fallback_action=str(fallback_action),
    )
