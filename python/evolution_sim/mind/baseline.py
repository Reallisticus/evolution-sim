from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable

from evolution_sim.env.runtime.action_contract import ACTION_CONTRACT_VERSION
from evolution_sim.env.runtime.action_space import ACTION_NAMES
from evolution_sim.env.runtime.observations import OBSERVATION_SCHEMA_VERSION
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.trajectory import TRAJECTORY_SCHEMA_VERSION
from evolution_sim.mind.contracts import MIND_MODEL_ARTIFACT_VERSION
from evolution_sim.mind.feature_policy import (
    FEATURE_POLICY_VERSION,
    feature_keys_from_record,
)
from evolution_sim.mind.provenance import validate_dataset_provenance

CONTEXTUAL_PRIOR_TRAINER = "contextual-prior"
REWARD_WEIGHTED_CONTEXTUAL_PRIOR_TRAINER = "reward-weighted-contextual-prior"
TRAINER_CHOICES: tuple[str, ...] = (
    CONTEXTUAL_PRIOR_TRAINER,
    REWARD_WEIGHTED_CONTEXTUAL_PRIOR_TRAINER,
)
BEHAVIOR_CLONING_BASELINE_MODEL_TYPE = "guarded_contextual_local_prior_bc_v2"
REWARD_WEIGHTED_BASELINE_MODEL_TYPE = (
    "guarded_reward_weighted_contextual_prior_bc_v1"
)
CONDITIONAL_MIN_RECORDS = 3
CONDITIONAL_SCORE_POLICY = "smoothed_contextual_action_prior_v1"
CONDITIONAL_PRIOR_CORRECTION_EXPONENT = 0.0
CONDITIONAL_SCORE_SMOOTHING_ALPHA = 0.1
UNIFORM_SAMPLE_WEIGHT_POLICY = "uniform_v1"
REWARD_TOTAL_SHIFTED_SAMPLE_WEIGHT_POLICY = "reward_total_shifted_clamp_v1"
UNIFORM_SAMPLE_WEIGHT_BASE = 1.0
UNIFORM_SAMPLE_WEIGHT_MIN = 1.0
REWARD_TOTAL_SAMPLE_WEIGHT_BASE = 1.0
REWARD_TOTAL_SAMPLE_WEIGHT_MIN = 0.05
HEURISTIC_CONFIDENCE_THRESHOLD = 0.5
HEURISTIC_OVERRIDE_MIN_MARGIN = 1.0
HEURISTIC_DELEGATE_POLICY = "observation_heuristic_confidence_delegate_v1"
HEURISTIC_DELEGATE_MAX_TRAINING_SCORE_MARGIN = 0.221
HEURISTIC_SAFE_LOCAL_EAT_MIN_SCORE = None
HEURISTIC_SAFE_LOCAL_EAT_MIN_FOOD = None
HEURISTIC_SAFE_LOCAL_EAT_MIN_PLANT_RATIO = None
HEURISTIC_SAFE_PLANT_MOVE_MIN_SCORE = None
HEURISTIC_SAFE_PLANT_MOVE_MIN_STRENGTH = None
HEURISTIC_SAFE_PLANT_MOVE_MAX_LOCAL_FOOD_RATIO = None
HEURISTIC_SAFE_PLANT_MOVE_MAX_DISTANCE = None
HEURISTIC_GUARD_POLICY = "observation_heuristic_safety_floor_v1"


@dataclass(frozen=True, slots=True)
class BehaviorCloningBaseline:
    action_scores: dict[str, float]
    action_score_metadata: dict[str, object]
    conditional_action_scores: dict[str, dict[str, float]]
    conditional_action_metadata: dict[str, dict[str, object]]
    record_count: int
    provenance: dict[str, object]
    model_type: str = BEHAVIOR_CLONING_BASELINE_MODEL_TYPE
    trainer: str = CONTEXTUAL_PRIOR_TRAINER
    sample_weight_policy: str = UNIFORM_SAMPLE_WEIGHT_POLICY
    sample_weight_base: float = UNIFORM_SAMPLE_WEIGHT_BASE
    sample_weight_min: float = UNIFORM_SAMPLE_WEIGHT_MIN
    sample_weight_total: float = 0.0

    def to_artifact(self) -> dict[str, object]:
        provenance = validate_dataset_provenance(self.provenance)
        return {
            "manifest": {
                "artifact_version": MIND_MODEL_ARTIFACT_VERSION,
                "model_type": self.model_type,
                "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "policy_interface_version": POLICY_INTERFACE_VERSION,
                "action_contract_version": ACTION_CONTRACT_VERSION,
                "trained_record_count": self.record_count,
                "provenance": provenance,
            },
            "model": {
                "action_scores": dict(sorted(self.action_scores.items())),
                "action_score_metadata": dict(
                    sorted(self.action_score_metadata.items())
                ),
                "conditional_action_scores": {
                    key: dict(sorted(scores.items()))
                    for key, scores in sorted(self.conditional_action_scores.items())
                },
                "conditional_action_metadata": {
                    key: dict(sorted(metadata.items()))
                    for key, metadata in sorted(
                        self.conditional_action_metadata.items()
                    )
                },
                "fallback_action": "stay",
                "feature_policy_version": FEATURE_POLICY_VERSION,
                "conditional_min_records": CONDITIONAL_MIN_RECORDS,
                "conditional_score_policy": CONDITIONAL_SCORE_POLICY,
                "conditional_prior_correction_exponent": (
                    CONDITIONAL_PRIOR_CORRECTION_EXPONENT
                ),
                "conditional_score_smoothing_alpha": (
                    CONDITIONAL_SCORE_SMOOTHING_ALPHA
                ),
                "trainer": self.trainer,
                "sample_weight_policy": self.sample_weight_policy,
                "sample_weight_base": self.sample_weight_base,
                "sample_weight_min": self.sample_weight_min,
                "sample_weight_total": round(self.sample_weight_total, 4),
                "heuristic_guard_policy": HEURISTIC_GUARD_POLICY,
                "heuristic_confidence_threshold": HEURISTIC_CONFIDENCE_THRESHOLD,
                "heuristic_override_min_margin": HEURISTIC_OVERRIDE_MIN_MARGIN,
                "heuristic_delegate_policy": HEURISTIC_DELEGATE_POLICY,
                "heuristic_delegate_max_training_score_margin": (
                    HEURISTIC_DELEGATE_MAX_TRAINING_SCORE_MARGIN
                ),
                "heuristic_safe_local_eat_min_score": (
                    HEURISTIC_SAFE_LOCAL_EAT_MIN_SCORE
                ),
                "heuristic_safe_local_eat_min_food": (
                    HEURISTIC_SAFE_LOCAL_EAT_MIN_FOOD
                ),
                "heuristic_safe_local_eat_min_plant_ratio": (
                    HEURISTIC_SAFE_LOCAL_EAT_MIN_PLANT_RATIO
                ),
                "heuristic_safe_plant_move_min_score": (
                    HEURISTIC_SAFE_PLANT_MOVE_MIN_SCORE
                ),
                "heuristic_safe_plant_move_min_strength": (
                    HEURISTIC_SAFE_PLANT_MOVE_MIN_STRENGTH
                ),
                "heuristic_safe_plant_move_max_local_food_ratio": (
                    HEURISTIC_SAFE_PLANT_MOVE_MAX_LOCAL_FOOD_RATIO
                ),
                "heuristic_safe_plant_move_max_distance": (
                    HEURISTIC_SAFE_PLANT_MOVE_MAX_DISTANCE
                ),
            },
        }


def train_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_contextual_prior_baseline(
        records,
        provenance=provenance,
        model_type=BEHAVIOR_CLONING_BASELINE_MODEL_TYPE,
        trainer=CONTEXTUAL_PRIOR_TRAINER,
        sample_weight_policy=UNIFORM_SAMPLE_WEIGHT_POLICY,
        sample_weight_base=UNIFORM_SAMPLE_WEIGHT_BASE,
        sample_weight_min=UNIFORM_SAMPLE_WEIGHT_MIN,
        sample_weight_fn=_uniform_sample_weight,
    )


def train_reward_weighted_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_contextual_prior_baseline(
        records,
        provenance=provenance,
        model_type=REWARD_WEIGHTED_BASELINE_MODEL_TYPE,
        trainer=REWARD_WEIGHTED_CONTEXTUAL_PRIOR_TRAINER,
        sample_weight_policy=REWARD_TOTAL_SHIFTED_SAMPLE_WEIGHT_POLICY,
        sample_weight_base=REWARD_TOTAL_SAMPLE_WEIGHT_BASE,
        sample_weight_min=REWARD_TOTAL_SAMPLE_WEIGHT_MIN,
        sample_weight_fn=_reward_total_shifted_sample_weight,
    )


def train_baseline_with_trainer(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
    trainer: str,
) -> BehaviorCloningBaseline:
    if trainer == CONTEXTUAL_PRIOR_TRAINER:
        return train_behavior_cloning_baseline(records, provenance=provenance)
    if trainer == REWARD_WEIGHTED_CONTEXTUAL_PRIOR_TRAINER:
        return train_reward_weighted_behavior_cloning_baseline(
            records,
            provenance=provenance,
        )
    raise ValueError(f"unsupported Mind baseline trainer: {trainer!r}")


def _train_contextual_prior_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
    model_type: str,
    trainer: str,
    sample_weight_policy: str,
    sample_weight_base: float,
    sample_weight_min: float,
    sample_weight_fn: Callable[[dict[str, object]], float],
) -> BehaviorCloningBaseline:
    support_counts: Counter[str] = Counter()
    weighted_counts: Counter[str] = Counter()
    conditional_support_counts: dict[str, Counter[str]] = {}
    conditional_weighted_counts: dict[str, Counter[str]] = {}
    record_count = 0
    sample_weight_total = 0.0
    for record in records:
        record_count += 1
        requested_action = str(record["requested_action"])
        resolved_action = str(record["resolved_action"])
        label = (
            requested_action
            if bool(record.get("resolution_action_valid", False))
            else resolved_action
        )
        weight = sample_weight_fn(record)
        support_counts[label] += 1
        weighted_counts[label] += weight
        sample_weight_total += weight
        for feature_key in feature_keys_from_record(record):
            conditional_support_counts.setdefault(feature_key, Counter())[label] += 1
            conditional_weighted_counts.setdefault(feature_key, Counter())[label] += (
                weight
            )
    total = sum(support_counts.values())
    if total <= 0:
        raise ValueError("behavior-cloning baseline requires at least one record")
    action_scores = _normalize_scores(weighted_counts)
    action_score_metadata = _score_metadata(
        support_counts,
        scores=action_scores,
        weighted_record_count=sum(weighted_counts.values()),
    )
    conditional_action_scores = {
        key: _normalize_prior_corrected_scores(
            conditional_weighted_counts[key],
            global_counts=weighted_counts,
            alpha=CONDITIONAL_SCORE_SMOOTHING_ALPHA,
            prior_exponent=CONDITIONAL_PRIOR_CORRECTION_EXPONENT,
        )
        for key, action_counts in conditional_support_counts.items()
        if sum(action_counts.values()) >= CONDITIONAL_MIN_RECORDS
    }
    conditional_action_metadata = {
        key: _score_metadata(
            action_counts,
            scores=conditional_action_scores[key],
            weighted_record_count=sum(conditional_weighted_counts[key].values()),
        )
        for key, action_counts in conditional_support_counts.items()
        if key in conditional_action_scores
    }
    return BehaviorCloningBaseline(
        action_scores=action_scores,
        action_score_metadata=action_score_metadata,
        conditional_action_scores=conditional_action_scores,
        conditional_action_metadata=conditional_action_metadata,
        record_count=record_count,
        provenance=provenance,
        model_type=model_type,
        trainer=trainer,
        sample_weight_policy=sample_weight_policy,
        sample_weight_base=sample_weight_base,
        sample_weight_min=sample_weight_min,
        sample_weight_total=sample_weight_total,
    )


def _normalize_scores(counts: Counter[str]) -> dict[str, float]:
    total = sum(float(value) for value in counts.values())
    if total <= 0:
        return {action: 0.0 for action in ACTION_NAMES}
    return {
        action: float(counts.get(action, 0.0)) / total
        for action in ACTION_NAMES
    }


def _normalize_prior_corrected_scores(
    counts: Counter[str],
    *,
    global_counts: Counter[str],
    alpha: float,
    prior_exponent: float,
) -> dict[str, float]:
    total = sum(float(value) for value in counts.values())
    global_total = sum(float(value) for value in global_counts.values())
    if total <= 0 or global_total <= 0:
        return {action: 0.0 for action in ACTION_NAMES}
    action_count = len(ACTION_NAMES)
    raw_scores = {
        action: (float(counts.get(action, 0.0)) + alpha)
        / (total + alpha * action_count)
        for action in ACTION_NAMES
    }
    global_priors = {
        action: (float(global_counts.get(action, 0.0)) + alpha)
        / (global_total + alpha * action_count)
        for action in ACTION_NAMES
    }
    corrected_scores = {
        action: raw_scores[action] / (global_priors[action] ** prior_exponent)
        for action in ACTION_NAMES
    }
    corrected_total = sum(corrected_scores.values())
    if corrected_total <= 0:
        return {action: 0.0 for action in ACTION_NAMES}
    return {
        action: corrected_scores[action] / corrected_total
        for action in ACTION_NAMES
    }


def _score_metadata(
    counts: Counter[str],
    *,
    scores: dict[str, float] | None = None,
    weighted_record_count: float | None = None,
) -> dict[str, object]:
    total = int(sum(counts.values()))
    if total <= 0:
        metadata = {
            "record_count": 0,
            "top_action": "stay",
            "top_score": 0.0,
            "runner_up_action": "stay",
            "runner_up_score": 0.0,
            "score_margin": 0.0,
        }
        if weighted_record_count is not None:
            metadata["weighted_record_count"] = round(weighted_record_count, 4)
        return metadata
    ranking_scores = scores if scores is not None else _normalize_scores(counts)
    ranked = sorted(
        (
            (float(ranking_scores.get(action, 0.0)), action)
            for action in sorted(ACTION_NAMES)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    top_score, top_action = ranked[0]
    runner_up_score, runner_up_action = (
        ranked[1] if len(ranked) > 1 else (0.0, "stay")
    )
    metadata = {
        "record_count": total,
        "top_action": top_action,
        "top_score": top_score,
        "runner_up_action": runner_up_action,
        "runner_up_score": runner_up_score,
        "score_margin": top_score - runner_up_score,
    }
    if weighted_record_count is not None:
        metadata["weighted_record_count"] = round(weighted_record_count, 4)
    return metadata


def _uniform_sample_weight(record: dict[str, object]) -> float:
    return UNIFORM_SAMPLE_WEIGHT_BASE


def _reward_total_shifted_sample_weight(record: dict[str, object]) -> float:
    return max(
        REWARD_TOTAL_SAMPLE_WEIGHT_MIN,
        REWARD_TOTAL_SAMPLE_WEIGHT_BASE + _record_reward_total(record),
    )


def _record_reward_total(record: dict[str, object]) -> float:
    reward = record.get("reward")
    if not isinstance(reward, dict):
        return 0.0
    total = reward.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return 0.0
    reward_total = float(total)
    if not math.isfinite(reward_total):
        return 0.0
    return reward_total
