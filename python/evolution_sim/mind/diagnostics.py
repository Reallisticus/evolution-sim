from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from evolution_sim.env.runtime.action_space import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    MEAT_MODE_VOCAB,
    SELF_INPUT_FIELDS,
    TROPHIC_ROLE_VOCAB,
    decode_observation_input,
)
from evolution_sim.mind.dataset import TrajectoryJsonlDataset
from evolution_sim.mind.feature_policy import feature_keys_from_record
from evolution_sim.mind.learned_policy import HEURISTIC_GUARD_POLICY

SELF_FIELD_INDEX = {field: index for index, field in enumerate(SELF_INPUT_FIELDS)}


def build_artifact_diagnostics(
    artifact: Mapping[str, object],
    datasets: Sequence[TrajectoryJsonlDataset],
) -> dict[str, object]:
    records = [record for dataset in datasets for record in dataset.records]
    model = _mapping(artifact.get("model"))
    action_scores = _score_mapping(model.get("action_scores"))
    conditional_scores = _conditional_score_mapping(
        model.get("conditional_action_scores")
    )

    label_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    match_depth_counts: Counter[str] = Counter()
    correct = 0
    matched_records = 0
    for record in records:
        label = _training_label(record)
        prediction, match_depth = _predict_record_action(
            record,
            action_scores=action_scores,
            conditional_scores=conditional_scores,
        )
        label_counts[label] += 1
        predicted_counts[prediction] += 1
        if prediction == label:
            correct += 1
        if match_depth is None:
            match_depth_counts["fallback"] += 1
        else:
            matched_records += 1
            match_depth_counts[str(match_depth)] += 1

    record_count = len(records)
    return {
        "record_count": record_count,
        "imitation": {
            "top1_correct": correct,
            "top1_accuracy": _rate(correct, record_count),
        },
        "action_distribution": {
            "label_counts": _sorted_counts(label_counts),
            "predicted_counts": _sorted_counts(predicted_counts),
            "prediction_label_tvd": _total_variation_distance(
                label_counts,
                predicted_counts,
            ),
        },
        "contextual_coverage": {
            "conditional_feature_count": len(conditional_scores),
            "matched_records": matched_records,
            "matched_record_rate": _rate(matched_records, record_count),
            "match_depth_counts": _sorted_counts(match_depth_counts),
        },
    }


def build_policy_diagnostics(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    action_source_counts: Counter[str] = Counter()
    guard_intervention_count = 0
    role_stats: dict[str, dict[str, object]] = {}
    mode_stats: dict[str, dict[str, object]] = {}
    for record in records:
        action_source = str(record.get("action_source", "unknown"))
        action_source_counts[action_source] += 1
        guard_used = HEURISTIC_GUARD_POLICY in action_source
        if guard_used:
            guard_intervention_count += 1
        reward = _reward_total(record)
        action = str(record.get("requested_action", "unknown"))
        role = _record_enum_label(record, "trophic_role_code", TROPHIC_ROLE_VOCAB)
        mode = _record_enum_label(record, "meat_mode_code", MEAT_MODE_VOCAB)
        _update_group_stats(
            role_stats,
            role,
            reward=reward,
            action=action,
            guard_used=guard_used,
        )
        _update_group_stats(
            mode_stats,
            mode,
            reward=reward,
            action=action,
            guard_used=guard_used,
        )

    record_count = len(records)
    return {
        "record_count": record_count,
        "action_source_counts": _sorted_counts(action_source_counts),
        "guard_intervention_count": guard_intervention_count,
        "guard_intervention_rate": _rate(guard_intervention_count, record_count),
        "by_trophic_role": _finalize_group_stats(role_stats),
        "by_meat_mode": _finalize_group_stats(mode_stats),
    }


def _predict_record_action(
    record: Mapping[str, object],
    *,
    action_scores: Mapping[str, float],
    conditional_scores: Mapping[str, Mapping[str, float]],
) -> tuple[str, int | None]:
    scores: Mapping[str, float] = action_scores
    match_depth: int | None = None
    for depth, feature_key in enumerate(feature_keys_from_record(record)):
        conditional = conditional_scores.get(feature_key)
        if conditional is not None:
            scores = conditional
            match_depth = depth
            break
    return _best_scored_action(scores, _bool_mapping(record.get("action_mask"))), match_depth


def _best_scored_action(
    scores: Mapping[str, float],
    action_mask: Mapping[str, bool],
) -> str:
    best_action = "stay"
    best_score = float("-inf")
    for action in sorted(action_mask):
        if not action_mask[action]:
            continue
        score = float(scores.get(action, 0.0))
        if score > best_score:
            best_action = action
            best_score = score
    return best_action


def _training_label(record: Mapping[str, object]) -> str:
    if bool(record.get("resolution_action_valid", False)):
        return str(record["requested_action"])
    return str(record["resolved_action"])


def _mapping(payload: object) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    return {}


def _score_mapping(payload: object) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(action): float(score)
        for action, score in payload.items()
        if isinstance(score, (int, float)) and not isinstance(score, bool)
    }


def _conditional_score_mapping(payload: object) -> dict[str, dict[str, float]]:
    if not isinstance(payload, Mapping):
        return {}
    parsed: dict[str, dict[str, float]] = {}
    for feature_key, scores in payload.items():
        parsed[str(feature_key)] = _score_mapping(scores)
    return parsed


def _bool_mapping(payload: object) -> dict[str, bool]:
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): bool(value) for key, value in payload.items()}


def _total_variation_distance(
    left: Counter[str],
    right: Counter[str],
) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 and right_total <= 0:
        return 0.0
    actions = set(ACTION_NAMES) | set(left) | set(right)
    drift = 0.0
    for action in actions:
        left_share = left[action] / left_total if left_total else 0.0
        right_share = right[action] / right_total if right_total else 0.0
        drift += abs(left_share - right_share)
    return round(drift / 2.0, 4)


def _reward_total(record: Mapping[str, object]) -> float:
    reward = record.get("reward")
    if not isinstance(reward, Mapping):
        return 0.0
    total = reward.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return 0.0
    return float(total)


def _record_enum_label(
    record: Mapping[str, object],
    field: str,
    vocab: tuple[str, ...],
) -> str:
    observation_input = record.get("observation_input")
    if not isinstance(observation_input, dict):
        return "unknown"
    try:
        values = decode_observation_input(observation_input)
    except ValueError:
        return "unknown"
    index = SELF_FIELD_INDEX[field]
    if index >= len(values):
        return "unknown"
    return _enum_label(values[index], vocab)


def _enum_label(value: float, vocab: tuple[str, ...]) -> str:
    if not vocab:
        return "unknown"
    index = round(float(value) * (len(vocab) - 1))
    index = max(0, min(len(vocab) - 1, index))
    return vocab[index]


def _update_group_stats(
    groups: dict[str, dict[str, object]],
    group: str,
    *,
    reward: float,
    action: str,
    guard_used: bool,
) -> None:
    stats = groups.setdefault(
        group,
        {
            "record_count": 0,
            "total_reward": 0.0,
            "guard_intervention_count": 0,
            "action_counts": Counter(),
        },
    )
    stats["record_count"] = int(stats["record_count"]) + 1
    stats["total_reward"] = float(stats["total_reward"]) + reward
    if guard_used:
        stats["guard_intervention_count"] = int(stats["guard_intervention_count"]) + 1
    action_counts = stats["action_counts"]
    if isinstance(action_counts, Counter):
        action_counts[action] += 1


def _finalize_group_stats(
    groups: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    finalized: dict[str, dict[str, object]] = {}
    for group, stats in sorted(groups.items()):
        record_count = int(stats["record_count"])
        action_counts = stats["action_counts"]
        if not isinstance(action_counts, Counter):
            action_counts = Counter()
        guard_count = int(stats["guard_intervention_count"])
        finalized[group] = {
            "record_count": record_count,
            "mean_reward": _rate(float(stats["total_reward"]), record_count),
            "guard_intervention_count": guard_count,
            "guard_intervention_rate": _rate(guard_count, record_count),
            "action_counts": _sorted_counts(action_counts),
        }
    return finalized


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / denominator, 4)
