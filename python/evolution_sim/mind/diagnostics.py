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
from evolution_sim.mind.learned_policy import (
    HEURISTIC_DELEGATE_POLICY,
    HEURISTIC_GUARD_POLICY,
    NEURAL_SCORE_NORMALIZATION_POLICY,
)
from evolution_sim.mind.neural import (
    compile_neural_actor_critic_network,
    score_neural_actor_critic_values,
)

SELF_FIELD_INDEX = {field: index for index, field in enumerate(SELF_INPUT_FIELDS)}
POLICY_DIAGNOSTIC_CONTEXT_DEPTH = 2
POLICY_DIAGNOSTIC_TOP_CONTEXT_LIMIT = 12


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
    conditional_metadata = _conditional_metadata_mapping(
        model.get("conditional_action_metadata")
    )

    label_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    true_positive_counts: Counter[str] = Counter()
    confusion_counts: Counter[tuple[str, str]] = Counter()
    match_depth_counts: Counter[str] = Counter()
    support_bucket_counts: Counter[str] = Counter()
    margin_bucket_counts: Counter[str] = Counter()
    label_reward_stats: dict[str, dict[str, object]] = {}
    predicted_reward_stats: dict[str, dict[str, object]] = {}
    score_margin_reward_stats: dict[str, dict[str, object]] = {}
    correct = 0
    matched_records = 0
    for record in records:
        label = _training_label(record)
        prediction, match_depth, prediction_margin = _predict_record_action(
            record,
            action_scores=action_scores,
            conditional_scores=conditional_scores,
        )
        reward = _reward_total(record)
        label_counts[label] += 1
        predicted_counts[prediction] += 1
        confusion_counts[(label, prediction)] += 1
        _update_reward_stats(label_reward_stats, label, reward=reward)
        _update_reward_stats(predicted_reward_stats, prediction, reward=reward)
        _update_reward_stats(
            score_margin_reward_stats,
            _score_margin_bucket(prediction_margin),
            reward=reward,
        )
        if prediction == label:
            correct += 1
            true_positive_counts[label] += 1
        if match_depth is None:
            match_depth_counts["fallback"] += 1
        else:
            matched_records += 1
            match_depth_counts[str(match_depth)] += 1
    for metadata in conditional_metadata.values():
        support_bucket_counts[
            _support_bucket(_metadata_number(metadata, "record_count"))
        ] += 1
        margin_bucket_counts[
            _score_margin_bucket(_metadata_number(metadata, "score_margin"))
        ] += 1

    record_count = len(records)
    diagnostics = {
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
            "confusion": {
                "matrix": _confusion_matrix(confusion_counts),
                "top_misclassifications": _top_misclassifications(
                    confusion_counts
                ),
                "per_action": _per_action_confusion(
                    label_counts,
                    predicted_counts,
                    true_positive_counts,
                ),
            },
        },
        "contextual_coverage": {
            "conditional_feature_count": len(conditional_scores),
            "matched_records": matched_records,
            "matched_record_rate": _rate(matched_records, record_count),
            "match_depth_counts": _sorted_counts(match_depth_counts),
            "support_bucket_counts": _sorted_counts(support_bucket_counts),
            "score_margin_bucket_counts": _sorted_counts(margin_bucket_counts),
        },
        "reward_calibration": {
            "by_label_action": _finalize_reward_stats(label_reward_stats),
            "by_predicted_action": _finalize_reward_stats(predicted_reward_stats),
            "by_score_margin_bucket": _finalize_reward_stats(
                score_margin_reward_stats
            ),
        },
    }
    neural_diagnostics = _build_neural_calibration_diagnostics(model, records)
    if neural_diagnostics is not None:
        diagnostics["neural_calibration"] = neural_diagnostics
    return diagnostics


def _build_neural_calibration_diagnostics(
    model: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    network = model.get("neural_network")
    if not isinstance(network, Mapping):
        return None
    compiled_network = compile_neural_actor_critic_network(network)

    correct = 0
    scored_count = 0
    action_value_abs_error_total = 0.0
    state_value_abs_error_total = 0.0
    action_value_margin_total = 0.0
    predicted_advantage_total = 0.0
    label_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    value_margin_bucket_stats: dict[str, dict[str, object]] = {}
    score_margin_bucket_stats: dict[str, dict[str, object]] = {}
    predicted_advantage_bucket_stats: dict[str, dict[str, object]] = {}
    label_action_stats: dict[str, dict[str, object]] = {}
    predicted_action_stats: dict[str, dict[str, object]] = {}

    for record in records:
        observation_input = record.get("observation_input")
        if not isinstance(observation_input, dict):
            continue
        values = decode_observation_input(observation_input)
        actor_scores, action_values, state_value = score_neural_actor_critic_values(
            network=compiled_network,
            values=values,
        )
        action_mask = _bool_mapping(record.get("action_mask"))
        actor_scores = _mask_renormalized_scores(actor_scores, action_mask)
        prediction, score_margin = _best_scored_action_with_margin(
            actor_scores,
            action_mask,
        )
        label = _training_label(record)
        reward = _reward_total(record)
        label_action_value = float(action_values.get(label, state_value))
        predicted_action_value = float(action_values.get(prediction, state_value))
        action_value_abs_error = abs(label_action_value - reward)
        state_value_abs_error = abs(float(state_value) - reward)
        action_value_margin = _action_value_margin_for_action(
            action_values,
            action_mask,
            prediction,
        )
        predicted_advantage = predicted_action_value - float(state_value)

        scored_count += 1
        label_counts[label] += 1
        predicted_counts[prediction] += 1
        action_value_abs_error_total += action_value_abs_error
        state_value_abs_error_total += state_value_abs_error
        action_value_margin_total += action_value_margin
        predicted_advantage_total += predicted_advantage
        if prediction == label:
            correct += 1

        _update_calibration_group_stats(
            score_margin_bucket_stats,
            _score_margin_bucket(score_margin),
            reward=reward,
            correct=prediction == label,
            action_value_abs_error=action_value_abs_error,
            state_value_abs_error=state_value_abs_error,
        )
        _update_calibration_group_stats(
            value_margin_bucket_stats,
            _signed_value_bucket(action_value_margin),
            reward=reward,
            correct=prediction == label,
            action_value_abs_error=action_value_abs_error,
            state_value_abs_error=state_value_abs_error,
        )
        _update_calibration_group_stats(
            predicted_advantage_bucket_stats,
            _signed_value_bucket(predicted_advantage),
            reward=reward,
            correct=prediction == label,
            action_value_abs_error=action_value_abs_error,
            state_value_abs_error=state_value_abs_error,
        )
        _update_calibration_group_stats(
            label_action_stats,
            label,
            reward=reward,
            correct=prediction == label,
            action_value_abs_error=action_value_abs_error,
            state_value_abs_error=state_value_abs_error,
        )
        _update_calibration_group_stats(
            predicted_action_stats,
            prediction,
            reward=reward,
            correct=prediction == label,
            action_value_abs_error=action_value_abs_error,
            state_value_abs_error=state_value_abs_error,
        )

    return {
        "record_count": scored_count,
        "score_normalization_policy": NEURAL_SCORE_NORMALIZATION_POLICY,
        "actor_top1_accuracy": _rate(correct, scored_count),
        "action_value_mean_abs_error": _rate(
            action_value_abs_error_total,
            scored_count,
        ),
        "state_value_mean_abs_error": _rate(
            state_value_abs_error_total,
            scored_count,
        ),
        "action_value_margin_mean": _rate(action_value_margin_total, scored_count),
        "predicted_advantage_mean": _rate(predicted_advantage_total, scored_count),
        "label_counts": _sorted_counts(label_counts),
        "predicted_counts": _sorted_counts(predicted_counts),
        "by_score_margin_bucket": _finalize_calibration_group_stats(
            score_margin_bucket_stats
        ),
        "by_action_value_margin_bucket": _finalize_calibration_group_stats(
            value_margin_bucket_stats
        ),
        "by_predicted_advantage_bucket": _finalize_calibration_group_stats(
            predicted_advantage_bucket_stats
        ),
        "by_label_action": _finalize_calibration_group_stats(label_action_stats),
        "by_predicted_action": _finalize_calibration_group_stats(
            predicted_action_stats
        ),
    }


def build_policy_diagnostics(
    records: Sequence[Mapping[str, object]],
    *,
    decision_diagnostics: Sequence[Mapping[str, object] | None] | None = None,
) -> dict[str, object]:
    action_source_counts: Counter[str] = Counter()
    guard_intervention_count = 0
    heuristic_delegate_count = 0
    safe_deviation_count = 0
    action_stats: dict[str, dict[str, object]] = {}
    heuristic_delegate_action_stats: dict[str, dict[str, object]] = {}
    safe_deviation_action_stats: dict[str, dict[str, object]] = {}
    context_stats: dict[str, dict[str, object]] = {}
    suppressed_action_stats: dict[str, dict[str, object]] = {}
    delegate_suppressed_action_stats: dict[str, dict[str, object]] = {}
    score_source_stats: dict[str, dict[str, object]] = {}
    support_bucket_stats: dict[str, dict[str, object]] = {}
    score_margin_bucket_stats: dict[str, dict[str, object]] = {}
    role_stats: dict[str, dict[str, object]] = {}
    mode_stats: dict[str, dict[str, object]] = {}
    for index, record in enumerate(records):
        action_source = str(record.get("action_source", "unknown"))
        action_source_counts[action_source] += 1
        decision_diagnostic = _decision_diagnostic_at(decision_diagnostics, index)
        guard_used = HEURISTIC_GUARD_POLICY in action_source
        if guard_used:
            guard_intervention_count += 1
        heuristic_delegate_used = (
            HEURISTIC_DELEGATE_POLICY in action_source
            or _diagnostic_bool(decision_diagnostic, "heuristic_delegate_used")
        )
        if heuristic_delegate_used:
            heuristic_delegate_count += 1
        reward = _reward_total(record)
        action = str(record.get("requested_action", "unknown"))
        if heuristic_delegate_used:
            _update_heuristic_delegate_group_stats(
                heuristic_delegate_action_stats,
                action,
                reward=reward,
            )
        safe_deviation_used = _diagnostic_bool(
            decision_diagnostic,
            "safe_deviation_used",
        )
        if safe_deviation_used:
            safe_deviation_count += 1
            _update_safe_deviation_group_stats(
                safe_deviation_action_stats,
                _diagnostic_label(
                    decision_diagnostic,
                    "learned_action",
                    default=action,
                ),
                reward=reward,
            )
        suppressed_action = _guard_suppressed_action(
            decision_diagnostic,
            guard_used=guard_used,
        )
        delegate_suppressed_action = _delegate_suppressed_action(
            decision_diagnostic,
            heuristic_delegate_used=heuristic_delegate_used,
        )
        score_source = _diagnostic_label(
            decision_diagnostic,
            "score_source",
            default="unknown",
        )
        support_bucket = _support_bucket(
            _diagnostic_number(decision_diagnostic, "score_support")
        )
        score_margin_bucket = _score_margin_bucket(
            _diagnostic_number(decision_diagnostic, "learned_score_margin")
        )
        if suppressed_action is not None:
            _update_group_stats(
                suppressed_action_stats,
                suppressed_action,
                reward=reward,
                action=action,
                guard_used=guard_used,
                heuristic_delegate_used=heuristic_delegate_used,
            )
        if delegate_suppressed_action is not None:
            _update_group_stats(
                delegate_suppressed_action_stats,
                delegate_suppressed_action,
                reward=reward,
                action=action,
                guard_used=guard_used,
                heuristic_delegate_used=heuristic_delegate_used,
            )
        _update_group_stats(
            score_source_stats,
            score_source,
            reward=reward,
            action=action,
            guard_used=guard_used,
            heuristic_delegate_used=heuristic_delegate_used,
        )
        _update_group_stats(
            support_bucket_stats,
            support_bucket,
            reward=reward,
            action=action,
            guard_used=guard_used,
            heuristic_delegate_used=heuristic_delegate_used,
        )
        _update_group_stats(
            score_margin_bucket_stats,
            score_margin_bucket,
            reward=reward,
            action=action,
            guard_used=guard_used,
            heuristic_delegate_used=heuristic_delegate_used,
        )
        _update_group_stats(
            action_stats,
            action,
            reward=reward,
            action=action,
            guard_used=guard_used,
            heuristic_delegate_used=heuristic_delegate_used,
        )
        _update_group_stats(
            context_stats,
            _record_policy_context_key(record),
            reward=reward,
            action=action,
            guard_used=guard_used,
            heuristic_delegate_used=heuristic_delegate_used,
        )
        role = _record_enum_label(record, "trophic_role_code", TROPHIC_ROLE_VOCAB)
        mode = _record_enum_label(record, "meat_mode_code", MEAT_MODE_VOCAB)
        _update_group_stats(
            role_stats,
            role,
            reward=reward,
            action=action,
            guard_used=guard_used,
            heuristic_delegate_used=heuristic_delegate_used,
        )
        _update_group_stats(
            mode_stats,
            mode,
            reward=reward,
            action=action,
            guard_used=guard_used,
            heuristic_delegate_used=heuristic_delegate_used,
        )

    record_count = len(records)
    return {
        "record_count": record_count,
        "action_source_counts": _sorted_counts(action_source_counts),
        "guard_intervention_count": guard_intervention_count,
        "guard_intervention_rate": _rate(guard_intervention_count, record_count),
        "heuristic_delegate_count": heuristic_delegate_count,
        "heuristic_delegate_rate": _rate(heuristic_delegate_count, record_count),
        "heuristic_delegate_by_action": _finalize_heuristic_delegate_group_stats(
            heuristic_delegate_action_stats
        ),
        "safe_deviation_count": safe_deviation_count,
        "safe_deviation_rate": _rate(safe_deviation_count, record_count),
        "safe_deviation_by_action": _finalize_safe_deviation_group_stats(
            safe_deviation_action_stats
        ),
        "guard_intervention_by_action": _finalize_group_stats(action_stats),
        "guard_suppressed_learned_action": _finalize_group_stats(
            suppressed_action_stats
        ),
        "heuristic_delegate_suppressed_learned_action": _finalize_group_stats(
            delegate_suppressed_action_stats
        ),
        "guard_intervention_by_score_source": _finalize_group_stats(
            score_source_stats
        ),
        "heuristic_delegate_by_score_source": _finalize_group_stats(
            score_source_stats
        ),
        "guard_intervention_by_support_bucket": _finalize_group_stats(
            support_bucket_stats
        ),
        "heuristic_delegate_by_support_bucket": _finalize_group_stats(
            support_bucket_stats
        ),
        "guard_intervention_by_score_margin_bucket": _finalize_group_stats(
            score_margin_bucket_stats
        ),
        "heuristic_delegate_by_score_margin_bucket": _finalize_group_stats(
            score_margin_bucket_stats
        ),
        "top_guarded_contexts": _top_guarded_contexts(context_stats),
        "top_delegated_contexts": _top_delegated_contexts(context_stats),
        "by_trophic_role": _finalize_group_stats(role_stats),
        "by_meat_mode": _finalize_group_stats(mode_stats),
    }


def _predict_record_action(
    record: Mapping[str, object],
    *,
    action_scores: Mapping[str, float],
    conditional_scores: Mapping[str, Mapping[str, float]],
) -> tuple[str, int | None, float]:
    scores: Mapping[str, float] = action_scores
    match_depth: int | None = None
    for depth, feature_key in enumerate(feature_keys_from_record(record)):
        conditional = conditional_scores.get(feature_key)
        if conditional is not None:
            scores = conditional
            match_depth = depth
            break
    prediction, score_margin = _best_scored_action_with_margin(
        scores,
        _bool_mapping(record.get("action_mask")),
    )
    return prediction, match_depth, score_margin


def _best_scored_action(
    scores: Mapping[str, float],
    action_mask: Mapping[str, bool],
) -> str:
    prediction, _score_margin = _best_scored_action_with_margin(
        scores,
        action_mask,
    )
    return prediction


def _best_scored_action_with_margin(
    scores: Mapping[str, float],
    action_mask: Mapping[str, bool],
) -> tuple[str, float]:
    best_action = "stay"
    best_score = float("-inf")
    runner_up_score = float("-inf")
    for action in sorted(action_mask):
        if not action_mask[action]:
            continue
        score = float(scores.get(action, 0.0))
        if score > best_score:
            runner_up_score = best_score
            best_action = action
            best_score = score
        elif score > runner_up_score:
            runner_up_score = score
    if best_score == float("-inf"):
        best_score = 0.0
    if runner_up_score == float("-inf"):
        runner_up_score = 0.0
    return best_action, best_score - runner_up_score


def _mask_renormalized_scores(
    scores: Mapping[str, float],
    action_mask: Mapping[str, bool],
) -> dict[str, float]:
    available_actions = [
        action
        for action, available in action_mask.items()
        if bool(available)
    ]
    total = sum(max(0.0, float(scores.get(action, 0.0))) for action in available_actions)
    if total <= 0.0:
        return {str(action): float(score) for action, score in scores.items()}
    normalized = {str(action): 0.0 for action in scores}
    for action in available_actions:
        normalized[str(action)] = max(0.0, float(scores.get(action, 0.0))) / total
    return normalized


def _action_value_margin_for_action(
    action_values: Mapping[str, float],
    action_mask: Mapping[str, bool],
    action: str,
) -> float:
    action_value = float(action_values.get(action, 0.0))
    best_other = float("-inf")
    for candidate in sorted(action_mask):
        if candidate == action or not action_mask[candidate]:
            continue
        best_other = max(best_other, float(action_values.get(candidate, 0.0)))
    if best_other == float("-inf"):
        best_other = 0.0
    return action_value - best_other


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


def _conditional_metadata_mapping(payload: object) -> dict[str, dict[str, object]]:
    if not isinstance(payload, Mapping):
        return {}
    parsed: dict[str, dict[str, object]] = {}
    for feature_key, metadata in payload.items():
        if isinstance(metadata, Mapping):
            parsed[str(feature_key)] = dict(metadata)
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


def _confusion_matrix(
    confusion_counts: Counter[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for label in ACTION_NAMES:
        row = {
            prediction: confusion_counts[(label, prediction)]
            for prediction in ACTION_NAMES
            if confusion_counts[(label, prediction)] > 0
        }
        matrix[label] = row
    return matrix


def _top_misclassifications(
    confusion_counts: Counter[tuple[str, str]],
) -> list[dict[str, object]]:
    confusions: list[dict[str, object]] = []
    for (label, prediction), count in confusion_counts.items():
        if label == prediction or count <= 0:
            continue
        confusions.append(
            {
                "label": label,
                "prediction": prediction,
                "count": count,
            }
        )
    confusions.sort(
        key=lambda item: (
            -int(item["count"]),
            str(item["label"]),
            str(item["prediction"]),
        )
    )
    return confusions[:POLICY_DIAGNOSTIC_TOP_CONTEXT_LIMIT]


def _per_action_confusion(
    label_counts: Counter[str],
    predicted_counts: Counter[str],
    true_positive_counts: Counter[str],
) -> dict[str, dict[str, object]]:
    per_action: dict[str, dict[str, object]] = {}
    for action in ACTION_NAMES:
        label_count = label_counts[action]
        predicted_count = predicted_counts[action]
        true_positive_count = true_positive_counts[action]
        per_action[action] = {
            "label_count": label_count,
            "predicted_count": predicted_count,
            "true_positive_count": true_positive_count,
            "recall": _rate(true_positive_count, label_count),
            "precision": _rate(true_positive_count, predicted_count),
        }
    return per_action


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


def _record_policy_context_key(record: Mapping[str, object]) -> str:
    try:
        feature_keys = feature_keys_from_record(record)
    except ValueError:
        return "unavailable"
    if not feature_keys:
        return "unavailable"
    if len(feature_keys) > POLICY_DIAGNOSTIC_CONTEXT_DEPTH:
        return feature_keys[POLICY_DIAGNOSTIC_CONTEXT_DEPTH]
    return feature_keys[-1]


def _decision_diagnostic_at(
    diagnostics: Sequence[Mapping[str, object] | None] | None,
    index: int,
) -> Mapping[str, object]:
    if diagnostics is None or index >= len(diagnostics):
        return {}
    diagnostic = diagnostics[index]
    if isinstance(diagnostic, Mapping):
        return diagnostic
    return {}


def _guard_suppressed_action(
    diagnostic: Mapping[str, object],
    *,
    guard_used: bool,
) -> str | None:
    if not guard_used:
        return None
    learned_action = diagnostic.get("learned_action")
    if not isinstance(learned_action, str) or not learned_action:
        return None
    return learned_action


def _delegate_suppressed_action(
    diagnostic: Mapping[str, object],
    *,
    heuristic_delegate_used: bool,
) -> str | None:
    if not heuristic_delegate_used:
        return None
    learned_action = diagnostic.get("learned_action")
    if not isinstance(learned_action, str) or not learned_action:
        return None
    return learned_action


def _diagnostic_label(
    diagnostic: Mapping[str, object],
    key: str,
    *,
    default: str,
) -> str:
    value = diagnostic.get(key)
    if isinstance(value, str) and value:
        return value
    return default


def _diagnostic_number(
    diagnostic: Mapping[str, object],
    key: str,
) -> float | None:
    value = diagnostic.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _diagnostic_bool(
    diagnostic: Mapping[str, object],
    key: str,
) -> bool:
    return diagnostic.get(key) is True


def _metadata_number(
    metadata: Mapping[str, object],
    key: str,
) -> float | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _support_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 0:
        return "0"
    if value < 10:
        return "1-9"
    if value < 32:
        return "10-31"
    if value < 100:
        return "32-99"
    return "100+"


def _score_margin_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.1:
        return "0.00-0.09"
    if value < 0.25:
        return "0.10-0.24"
    if value < 0.5:
        return "0.25-0.49"
    if value < 1.0:
        return "0.50-0.99"
    return "1.00+"


def _signed_value_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < -0.1:
        return "<-0.10"
    if value < -0.02:
        return "-0.10--0.02"
    if value < 0.02:
        return "-0.02-0.02"
    if value < 0.1:
        return "0.02-0.09"
    return "0.10+"


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
    heuristic_delegate_used: bool = False,
) -> None:
    stats = groups.setdefault(
        group,
        {
            "record_count": 0,
            "total_reward": 0.0,
            "guard_intervention_count": 0,
            "heuristic_delegate_count": 0,
            "action_counts": Counter(),
        },
    )
    stats["record_count"] = int(stats["record_count"]) + 1
    stats["total_reward"] = float(stats["total_reward"]) + reward
    if guard_used:
        stats["guard_intervention_count"] = int(stats["guard_intervention_count"]) + 1
    if heuristic_delegate_used:
        stats["heuristic_delegate_count"] = (
            int(stats["heuristic_delegate_count"]) + 1
        )
    action_counts = stats["action_counts"]
    if isinstance(action_counts, Counter):
        action_counts[action] += 1


def _update_reward_stats(
    groups: dict[str, dict[str, object]],
    group: str,
    *,
    reward: float,
) -> None:
    stats = groups.setdefault(
        group,
        {
            "record_count": 0,
            "total_reward": 0.0,
        },
    )
    stats["record_count"] = int(stats["record_count"]) + 1
    stats["total_reward"] = float(stats["total_reward"]) + reward


def _update_calibration_group_stats(
    groups: dict[str, dict[str, object]],
    group: str,
    *,
    reward: float,
    correct: bool,
    action_value_abs_error: float,
    state_value_abs_error: float,
) -> None:
    stats = groups.setdefault(
        group,
        {
            "record_count": 0,
            "correct_count": 0,
            "total_reward": 0.0,
            "action_value_abs_error_total": 0.0,
            "state_value_abs_error_total": 0.0,
        },
    )
    stats["record_count"] = int(stats["record_count"]) + 1
    if correct:
        stats["correct_count"] = int(stats["correct_count"]) + 1
    stats["total_reward"] = float(stats["total_reward"]) + reward
    stats["action_value_abs_error_total"] = (
        float(stats["action_value_abs_error_total"]) + action_value_abs_error
    )
    stats["state_value_abs_error_total"] = (
        float(stats["state_value_abs_error_total"]) + state_value_abs_error
    )


def _finalize_calibration_group_stats(
    groups: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    finalized: dict[str, dict[str, object]] = {}
    for group, stats in sorted(groups.items()):
        record_count = int(stats["record_count"])
        finalized[group] = {
            "record_count": record_count,
            "actor_top1_accuracy": _rate(
                int(stats["correct_count"]),
                record_count,
            ),
            "mean_reward": _rate(float(stats["total_reward"]), record_count),
            "action_value_mean_abs_error": _rate(
                float(stats["action_value_abs_error_total"]),
                record_count,
            ),
            "state_value_mean_abs_error": _rate(
                float(stats["state_value_abs_error_total"]),
                record_count,
            ),
        }
    return finalized


def _finalize_reward_stats(
    groups: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    finalized: dict[str, dict[str, object]] = {}
    for group, stats in sorted(groups.items()):
        record_count = int(stats["record_count"])
        finalized[group] = {
            "record_count": record_count,
            "mean_reward": _rate(float(stats["total_reward"]), record_count),
        }
    return finalized


def _update_safe_deviation_group_stats(
    groups: dict[str, dict[str, object]],
    group: str,
    *,
    reward: float,
) -> None:
    stats = groups.setdefault(
        group,
        {
            "record_count": 0,
            "total_reward": 0.0,
            "safe_deviation_count": 0,
        },
    )
    stats["record_count"] = int(stats["record_count"]) + 1
    stats["safe_deviation_count"] = int(stats["safe_deviation_count"]) + 1
    stats["total_reward"] = float(stats["total_reward"]) + reward


def _update_heuristic_delegate_group_stats(
    groups: dict[str, dict[str, object]],
    group: str,
    *,
    reward: float,
) -> None:
    stats = groups.setdefault(
        group,
        {
            "record_count": 0,
            "total_reward": 0.0,
            "heuristic_delegate_count": 0,
        },
    )
    stats["record_count"] = int(stats["record_count"]) + 1
    stats["heuristic_delegate_count"] = (
        int(stats["heuristic_delegate_count"]) + 1
    )
    stats["total_reward"] = float(stats["total_reward"]) + reward


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
        delegate_count = int(stats["heuristic_delegate_count"])
        finalized[group] = {
            "record_count": record_count,
            "mean_reward": _rate(float(stats["total_reward"]), record_count),
            "guard_intervention_count": guard_count,
            "guard_intervention_rate": _rate(guard_count, record_count),
            "heuristic_delegate_count": delegate_count,
            "heuristic_delegate_rate": _rate(delegate_count, record_count),
            "action_counts": _sorted_counts(action_counts),
        }
    return finalized


def _finalize_heuristic_delegate_group_stats(
    groups: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    finalized: dict[str, dict[str, object]] = {}
    for group, stats in sorted(groups.items()):
        record_count = int(stats["record_count"])
        delegate_count = int(stats["heuristic_delegate_count"])
        finalized[group] = {
            "record_count": record_count,
            "mean_reward": _rate(float(stats["total_reward"]), record_count),
            "heuristic_delegate_count": delegate_count,
            "heuristic_delegate_rate": _rate(delegate_count, record_count),
        }
    return finalized


def _finalize_safe_deviation_group_stats(
    groups: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    finalized: dict[str, dict[str, object]] = {}
    for group, stats in sorted(groups.items()):
        record_count = int(stats["record_count"])
        safe_count = int(stats["safe_deviation_count"])
        finalized[group] = {
            "record_count": record_count,
            "mean_reward": _rate(float(stats["total_reward"]), record_count),
            "safe_deviation_count": safe_count,
            "safe_deviation_rate": _rate(safe_count, record_count),
        }
    return finalized


def _top_guarded_contexts(
    groups: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    for feature_key, stats in groups.items():
        record_count = int(stats["record_count"])
        guard_count = int(stats["guard_intervention_count"])
        if guard_count <= 0:
            continue
        action_counts = stats["action_counts"]
        if not isinstance(action_counts, Counter):
            action_counts = Counter()
        contexts.append(
            {
                "feature_key": feature_key,
                "record_count": record_count,
                "mean_reward": _rate(float(stats["total_reward"]), record_count),
                "guard_intervention_count": guard_count,
                "guard_intervention_rate": _rate(guard_count, record_count),
                "action_counts": _sorted_counts(action_counts),
            }
        )
    contexts.sort(
        key=lambda context: (
            -int(context["guard_intervention_count"]),
            -float(context["guard_intervention_rate"]),
            str(context["feature_key"]),
        )
    )
    return contexts[:POLICY_DIAGNOSTIC_TOP_CONTEXT_LIMIT]


def _top_delegated_contexts(
    groups: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    for feature_key, stats in groups.items():
        record_count = int(stats["record_count"])
        delegate_count = int(stats["heuristic_delegate_count"])
        if delegate_count <= 0:
            continue
        action_counts = stats["action_counts"]
        if not isinstance(action_counts, Counter):
            action_counts = Counter()
        contexts.append(
            {
                "feature_key": feature_key,
                "record_count": record_count,
                "mean_reward": _rate(float(stats["total_reward"]), record_count),
                "heuristic_delegate_count": delegate_count,
                "heuristic_delegate_rate": _rate(delegate_count, record_count),
                "action_counts": _sorted_counts(action_counts),
            }
        )
    contexts.sort(
        key=lambda context: (
            -int(context["heuristic_delegate_count"]),
            -float(context["heuristic_delegate_rate"]),
            str(context["feature_key"]),
        )
    )
    return contexts[:POLICY_DIAGNOSTIC_TOP_CONTEXT_LIMIT]


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / denominator, 4)
