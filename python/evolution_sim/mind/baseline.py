from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from evolution_sim.env.runtime.action_contract import ACTION_CONTRACT_VERSION
from evolution_sim.env.runtime.action_space import ACTION_NAMES
from evolution_sim.env.runtime.observations import OBSERVATION_SCHEMA_VERSION
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.trajectory import TRAJECTORY_SCHEMA_VERSION
from evolution_sim.mind.contracts import MIND_MODEL_ARTIFACT_VERSION
from evolution_sim.mind.provenance import validate_dataset_provenance

BEHAVIOR_CLONING_BASELINE_MODEL_TYPE = "global_action_prior_bc_v1"


@dataclass(frozen=True, slots=True)
class BehaviorCloningBaseline:
    action_scores: dict[str, float]
    record_count: int
    provenance: dict[str, object]

    def to_artifact(self) -> dict[str, object]:
        provenance = validate_dataset_provenance(self.provenance)
        return {
            "manifest": {
                "artifact_version": MIND_MODEL_ARTIFACT_VERSION,
                "model_type": BEHAVIOR_CLONING_BASELINE_MODEL_TYPE,
                "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "policy_interface_version": POLICY_INTERFACE_VERSION,
                "action_contract_version": ACTION_CONTRACT_VERSION,
                "trained_record_count": self.record_count,
                "provenance": provenance,
            },
            "model": {
                "action_scores": dict(sorted(self.action_scores.items())),
                "fallback_action": "stay",
            },
        }


def train_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    counts: Counter[str] = Counter()
    record_count = 0
    for record in records:
        record_count += 1
        requested_action = str(record["requested_action"])
        resolved_action = str(record["resolved_action"])
        label = (
            requested_action
            if bool(record.get("resolution_action_valid", False))
            else resolved_action
        )
        counts[label] += 1
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("behavior-cloning baseline requires at least one record")
    action_scores = {
        action: counts.get(action, 0) / total
        for action in ACTION_NAMES
    }
    return BehaviorCloningBaseline(
        action_scores=action_scores,
        record_count=record_count,
        provenance=provenance,
    )
