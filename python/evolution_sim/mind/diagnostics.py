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
from evolution_sim.env.runtime.trajectory import REWARD_TOTAL_BOUNDS
from evolution_sim.mind.dataset import (
    TrajectoryJsonlDataset,
    build_trajectory_transitions,
    discounted_return_targets,
)
from evolution_sim.mind.feature_policy import feature_keys_from_record
from evolution_sim.mind.learned_policy import (
    HEURISTIC_DELEGATE_POLICY,
    HEURISTIC_GUARD_POLICY,
    NEURAL_SCORE_NORMALIZATION_POLICY,
)
from evolution_sim.mind.neural import (
    TORCH_IQL_DISCOUNT,
    compile_neural_actor_critic_network,
    score_neural_action_viability_components,
    score_neural_actor_critic_values,
    score_neural_viability_components,
)
from evolution_sim.mind.viability import (
    VIABILITY_ACTION_HEAD_POLICY,
    VIABILITY_ACTION_SUPERVISION_POLICY,
    VIABILITY_COMPONENT_NAMES,
    VIABILITY_FLOOR_RISK_RATIO,
    VIABILITY_HEAD_POLICY,
    VIABILITY_HEALTH_FLOOR_RISK_RATIO,
    VIABILITY_REPRODUCTION_DIAGNOSTIC_POLICY,
    VIABILITY_RUNTIME_DECISION_POLICY,
    VIABILITY_SURVIVAL_HORIZON_TICKS,
    VIABILITY_SUPPRESSION_COMPONENT,
    VIABILITY_TARGET_POLICY,
    build_viability_component_targets,
    reproduction_viable,
)

SELF_FIELD_INDEX = {field: index for index, field in enumerate(SELF_INPUT_FIELDS)}
POLICY_DIAGNOSTIC_CONTEXT_DEPTH = 2
POLICY_DIAGNOSTIC_TOP_CONTEXT_LIMIT = 12


def build_artifact_diagnostics(
    artifact: Mapping[str, object],
    datasets: Sequence[TrajectoryJsonlDataset],
) -> dict[str, object]:
    return finalize_artifact_diagnostics_shards(
        artifact,
        [build_artifact_diagnostics_shard_stats(artifact, datasets)],
    )


def build_artifact_diagnostics_shard_stats(
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
    true_positive_counts: Counter[str] = Counter()
    confusion_counts: Counter[tuple[str, str]] = Counter()
    match_depth_counts: Counter[str] = Counter()
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
    return {
        "record_count": len(records),
        "correct": correct,
        "matched_records": matched_records,
        "label_counts": label_counts,
        "predicted_counts": predicted_counts,
        "true_positive_counts": true_positive_counts,
        "confusion_counts": confusion_counts,
        "match_depth_counts": match_depth_counts,
        "label_reward_stats": label_reward_stats,
        "predicted_reward_stats": predicted_reward_stats,
        "score_margin_reward_stats": score_margin_reward_stats,
        "neural_calibration": _build_neural_calibration_stats(model, records),
        "viability_calibration": _build_viability_calibration_stats(
            model,
            records,
        ),
    }


def finalize_artifact_diagnostics_shards(
    artifact: Mapping[str, object],
    shard_stats: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    model = _mapping(artifact.get("model"))
    conditional_scores = _conditional_score_mapping(
        model.get("conditional_action_scores")
    )
    conditional_metadata = _conditional_metadata_mapping(
        model.get("conditional_action_metadata")
    )

    aggregate = _empty_artifact_diagnostics_stats()
    for stats in shard_stats:
        _merge_artifact_diagnostics_stats(aggregate, stats)
    support_bucket_counts: Counter[str] = Counter()
    margin_bucket_counts: Counter[str] = Counter()
    for metadata in conditional_metadata.values():
        support_bucket_counts[
            _support_bucket(_metadata_number(metadata, "record_count"))
        ] += 1
        margin_bucket_counts[
            _score_margin_bucket(_metadata_number(metadata, "score_margin"))
        ] += 1

    record_count = int(aggregate["record_count"])
    correct = int(aggregate["correct"])
    matched_records = int(aggregate["matched_records"])
    label_counts = _counter(aggregate["label_counts"])
    predicted_counts = _counter(aggregate["predicted_counts"])
    true_positive_counts = _counter(aggregate["true_positive_counts"])
    confusion_counts = _tuple_counter(aggregate["confusion_counts"])
    match_depth_counts = _counter(aggregate["match_depth_counts"])
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
            "by_label_action": _finalize_reward_stats(
                _stats_mapping(aggregate["label_reward_stats"])
            ),
            "by_predicted_action": _finalize_reward_stats(
                _stats_mapping(aggregate["predicted_reward_stats"])
            ),
            "by_score_margin_bucket": _finalize_reward_stats(
                _stats_mapping(aggregate["score_margin_reward_stats"])
            ),
        },
    }
    neural_stats = aggregate.get("neural_calibration")
    neural_diagnostics = (
        _finalize_neural_calibration_stats(_stats_mapping(neural_stats))
        if isinstance(neural_stats, Mapping)
        else None
    )
    if neural_diagnostics is not None:
        diagnostics["neural_calibration"] = neural_diagnostics
    viability_stats = aggregate.get("viability_calibration")
    viability_diagnostics = (
        _finalize_viability_calibration_stats(
            model,
            _stats_mapping(viability_stats),
        )
        if isinstance(viability_stats, Mapping)
        else None
    )
    if viability_diagnostics is not None:
        diagnostics["viability_calibration"] = viability_diagnostics
    return diagnostics


def _build_neural_calibration_stats(
    model: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    network = model.get("neural_network")
    if not isinstance(network, Mapping):
        return None
    compiled_network = compile_neural_actor_critic_network(network)
    record_dicts = [dict(record) for record in records]
    transitions = build_trajectory_transitions(record_dicts)
    return_targets = discounted_return_targets(
        transitions,
        discount=TORCH_IQL_DISCOUNT,
    )
    return_targets_by_index = {
        index: target
        for index, target in enumerate(return_targets)
    }

    correct = 0
    scored_count = 0
    action_value_abs_error_total = 0.0
    state_value_abs_error_total = 0.0
    action_value_return_abs_error_total = 0.0
    predicted_action_value_return_abs_error_total = 0.0
    state_value_return_abs_error_total = 0.0
    return_target_total = 0.0
    return_target_min: float | None = None
    return_target_max: float | None = None
    action_value_margin_total = 0.0
    predicted_advantage_total = 0.0
    label_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    value_margin_bucket_stats: dict[str, dict[str, object]] = {}
    score_margin_bucket_stats: dict[str, dict[str, object]] = {}
    predicted_advantage_bucket_stats: dict[str, dict[str, object]] = {}
    label_action_stats: dict[str, dict[str, object]] = {}
    predicted_action_stats: dict[str, dict[str, object]] = {}

    for index, record in enumerate(record_dicts):
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
        return_target = return_targets_by_index.get(index, reward)
        label_action_value = float(action_values.get(label, state_value))
        predicted_action_value = float(action_values.get(prediction, state_value))
        action_value_abs_error = abs(label_action_value - reward)
        state_value_abs_error = abs(float(state_value) - reward)
        action_value_return_abs_error = abs(label_action_value - return_target)
        predicted_action_value_return_abs_error = abs(
            predicted_action_value - return_target
        )
        state_value_return_abs_error = abs(float(state_value) - return_target)
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
        action_value_return_abs_error_total += action_value_return_abs_error
        predicted_action_value_return_abs_error_total += (
            predicted_action_value_return_abs_error
        )
        state_value_return_abs_error_total += state_value_return_abs_error
        return_target_total += return_target
        return_target_min = (
            return_target
            if return_target_min is None
            else min(return_target_min, return_target)
        )
        return_target_max = (
            return_target
            if return_target_max is None
            else max(return_target_max, return_target)
        )
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
            action_value_return_abs_error=action_value_return_abs_error,
            state_value_return_abs_error=state_value_return_abs_error,
        )
        _update_calibration_group_stats(
            value_margin_bucket_stats,
            _signed_value_bucket(action_value_margin),
            reward=reward,
            correct=prediction == label,
            action_value_abs_error=action_value_abs_error,
            state_value_abs_error=state_value_abs_error,
            action_value_return_abs_error=action_value_return_abs_error,
            state_value_return_abs_error=state_value_return_abs_error,
        )
        _update_calibration_group_stats(
            predicted_advantage_bucket_stats,
            _signed_value_bucket(predicted_advantage),
            reward=reward,
            correct=prediction == label,
            action_value_abs_error=action_value_abs_error,
            state_value_abs_error=state_value_abs_error,
            action_value_return_abs_error=action_value_return_abs_error,
            state_value_return_abs_error=state_value_return_abs_error,
        )
        _update_calibration_group_stats(
            label_action_stats,
            label,
            reward=reward,
            correct=prediction == label,
            action_value_abs_error=action_value_abs_error,
            state_value_abs_error=state_value_abs_error,
            action_value_return_abs_error=action_value_return_abs_error,
            state_value_return_abs_error=state_value_return_abs_error,
        )
        _update_calibration_group_stats(
            predicted_action_stats,
            prediction,
            reward=reward,
            correct=prediction == label,
            action_value_abs_error=action_value_abs_error,
            state_value_abs_error=state_value_abs_error,
            action_value_return_abs_error=(
                predicted_action_value_return_abs_error
            ),
            state_value_return_abs_error=state_value_return_abs_error,
        )

    return {
        "record_count": scored_count,
        "correct": correct,
        "action_value_abs_error_total": action_value_abs_error_total,
        "state_value_abs_error_total": state_value_abs_error_total,
        "action_value_return_abs_error_total": action_value_return_abs_error_total,
        "predicted_action_value_return_abs_error_total": (
            predicted_action_value_return_abs_error_total
        ),
        "state_value_return_abs_error_total": state_value_return_abs_error_total,
        "return_target_total": return_target_total,
        "return_target_min": return_target_min,
        "return_target_max": return_target_max,
        "action_value_margin_total": action_value_margin_total,
        "predicted_advantage_total": predicted_advantage_total,
        "label_counts": label_counts,
        "predicted_counts": predicted_counts,
        "value_margin_bucket_stats": value_margin_bucket_stats,
        "score_margin_bucket_stats": score_margin_bucket_stats,
        "predicted_advantage_bucket_stats": predicted_advantage_bucket_stats,
        "label_action_stats": label_action_stats,
        "predicted_action_stats": predicted_action_stats,
    }


def _build_viability_calibration_stats(
    model: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    network = model.get("neural_network")
    if not isinstance(network, Mapping):
        return None
    compiled_network = compile_neural_actor_critic_network(network)
    record_dicts = [dict(record) for record in records]
    component_targets, aggregate_targets, survival_horizons = (
        build_viability_component_targets(record_dicts)
    )
    if not component_targets:
        return None
    risk_scores: list[float] = []
    targets: list[bool] = []
    component_scores: dict[str, list[float]] = {
        component: []
        for component in VIABILITY_COMPONENT_NAMES
    }
    component_targets_by_name: dict[str, list[bool]] = {
        component: []
        for component in VIABILITY_COMPONENT_NAMES
    }
    logged_action_stats: dict[str, dict[str, object]] = {}
    predicted_action_stats: dict[str, dict[str, object]] = {}
    role_stats: dict[str, dict[str, object]] = {}
    mode_stats: dict[str, dict[str, object]] = {}
    risk_bucket_stats: dict[str, dict[str, object]] = {}
    component_stats: dict[str, dict[str, dict[str, object]]] = {
        component: {}
        for component in VIABILITY_COMPONENT_NAMES
    }
    target_counts: Counter[str] = Counter()
    predicted_risk_score_total = 0.0
    reproduction_viable_count = 0
    suppressed_action_score_count = 0

    for record, components, _target in zip(
        record_dicts,
        component_targets,
        aggregate_targets,
        strict=True,
    ):
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
        predicted_action, _score_margin = _best_scored_action_with_margin(
            actor_scores,
            action_mask,
        )
        logged_action = _training_label(record)
        action_viability_scores = score_neural_action_viability_components(
            network=compiled_network,
            values=values,
        )
        state_viability_scores = score_neural_viability_components(
            network=compiled_network,
            values=values,
        )
        if action_viability_scores is not None:
            logged_component_scores = action_viability_scores[logged_action]
            predicted_component_scores = action_viability_scores[predicted_action]
            suppressed_action = _suppressed_learned_action(record, action_mask)
        elif state_viability_scores is not None:
            logged_component_scores = state_viability_scores
            predicted_component_scores = state_viability_scores
            suppressed_action = None
        else:
            logged_component_scores = None
            predicted_component_scores = None
            suppressed_action = None
        observed_components = [
            component
            for component in VIABILITY_COMPONENT_NAMES
            if _viability_component_observed_for_diagnostics(
                component,
                record=record,
                action_viability_scores=action_viability_scores,
                suppressed_action=suppressed_action,
            )
        ]
        if logged_component_scores is None:
            logged_risk_score = _value_to_viability_risk_score(
                float(action_values.get(logged_action, state_value))
            )
        else:
            logged_risk_score = max(
                _action_conditioned_component_score(
                    component,
                    logged_component_scores=logged_component_scores,
                    action_viability_scores=action_viability_scores,
                    suppressed_action=suppressed_action,
                )
                for component in observed_components
            )
        predicted_risk_score = (
            max(
                float(predicted_component_scores[component])
                for component in observed_components
            )
            if predicted_component_scores is not None
            else _value_to_viability_risk_score(
                float(action_values.get(predicted_action, state_value))
            )
        )
        for component, active in components.items():
            if component not in observed_components:
                continue
            if active:
                target_counts[component] += 1
            component_score = (
                _action_conditioned_component_score(
                    component,
                    logged_component_scores=logged_component_scores,
                    action_viability_scores=action_viability_scores,
                    suppressed_action=suppressed_action,
                )
                if logged_component_scores is not None
                else logged_risk_score
            )
            if (
                action_viability_scores is not None
                and component == VIABILITY_SUPPRESSION_COMPONENT
                and suppressed_action is not None
            ):
                suppressed_action_score_count += 1
            component_scores[component].append(component_score)
            component_targets_by_name[component].append(active)
            _update_viability_group_stats(
                component_stats[component],
                _risk_score_bucket(component_score),
                risk_score=component_score,
                target=active,
            )
        if reproduction_viable(record):
            reproduction_viable_count += 1

        target = any(bool(components[component]) for component in observed_components)
        risk_scores.append(logged_risk_score)
        targets.append(target)
        predicted_risk_score_total += predicted_risk_score
        _update_viability_group_stats(
            logged_action_stats,
            logged_action,
            risk_score=logged_risk_score,
            target=target,
        )
        _update_viability_group_stats(
            predicted_action_stats,
            predicted_action,
            risk_score=predicted_risk_score,
            target=target,
        )
        _update_viability_group_stats(
            role_stats,
            _record_enum_label(record, "trophic_role_code", TROPHIC_ROLE_VOCAB),
            risk_score=logged_risk_score,
            target=target,
        )
        _update_viability_group_stats(
            mode_stats,
            _record_enum_label(record, "meat_mode_code", MEAT_MODE_VOCAB),
            risk_score=logged_risk_score,
            target=target,
        )
        _update_viability_group_stats(
            risk_bucket_stats,
            _risk_score_bucket(logged_risk_score),
            risk_score=logged_risk_score,
            target=target,
        )

    return {
        "risk_scores": risk_scores,
        "targets": targets,
        "component_scores": component_scores,
        "component_targets_by_name": component_targets_by_name,
        "logged_action_stats": logged_action_stats,
        "predicted_action_stats": predicted_action_stats,
        "role_stats": role_stats,
        "mode_stats": mode_stats,
        "risk_bucket_stats": risk_bucket_stats,
        "component_stats": component_stats,
        "target_counts": target_counts,
        "predicted_risk_score_total": predicted_risk_score_total,
        "reproduction_viable_count": reproduction_viable_count,
        "suppressed_action_score_count": suppressed_action_score_count,
        "survival_horizons": list(survival_horizons),
    }


def _empty_artifact_diagnostics_stats() -> dict[str, object]:
    return {
        "record_count": 0,
        "correct": 0,
        "matched_records": 0,
        "label_counts": Counter(),
        "predicted_counts": Counter(),
        "true_positive_counts": Counter(),
        "confusion_counts": Counter(),
        "match_depth_counts": Counter(),
        "label_reward_stats": {},
        "predicted_reward_stats": {},
        "score_margin_reward_stats": {},
        "neural_calibration": None,
        "viability_calibration": None,
    }


def _merge_artifact_diagnostics_stats(
    target: dict[str, object],
    source: Mapping[str, object],
) -> None:
    target["record_count"] = int(target["record_count"]) + int(
        source.get("record_count", 0)
    )
    target["correct"] = int(target["correct"]) + int(source.get("correct", 0))
    target["matched_records"] = int(target["matched_records"]) + int(
        source.get("matched_records", 0)
    )
    _counter(target["label_counts"]).update(_counter(source.get("label_counts")))
    _counter(target["predicted_counts"]).update(
        _counter(source.get("predicted_counts"))
    )
    _counter(target["true_positive_counts"]).update(
        _counter(source.get("true_positive_counts"))
    )
    _tuple_counter(target["confusion_counts"]).update(
        _tuple_counter(source.get("confusion_counts"))
    )
    _counter(target["match_depth_counts"]).update(
        _counter(source.get("match_depth_counts"))
    )
    _merge_reward_stats_groups(
        _mutable_stats_mapping(target["label_reward_stats"]),
        _stats_mapping(source.get("label_reward_stats")),
    )
    _merge_reward_stats_groups(
        _mutable_stats_mapping(target["predicted_reward_stats"]),
        _stats_mapping(source.get("predicted_reward_stats")),
    )
    _merge_reward_stats_groups(
        _mutable_stats_mapping(target["score_margin_reward_stats"]),
        _stats_mapping(source.get("score_margin_reward_stats")),
    )
    neural_stats = source.get("neural_calibration")
    if isinstance(neural_stats, Mapping):
        if not isinstance(target.get("neural_calibration"), Mapping):
            target["neural_calibration"] = _empty_neural_calibration_stats()
        _merge_neural_calibration_stats(
            _mutable_stats_mapping(target["neural_calibration"]),
            neural_stats,
        )
    viability_stats = source.get("viability_calibration")
    if isinstance(viability_stats, Mapping):
        if not isinstance(target.get("viability_calibration"), Mapping):
            target["viability_calibration"] = _empty_viability_calibration_stats()
        _merge_viability_calibration_stats(
            _mutable_stats_mapping(target["viability_calibration"]),
            viability_stats,
        )


def _empty_neural_calibration_stats() -> dict[str, object]:
    return {
        "record_count": 0,
        "correct": 0,
        "action_value_abs_error_total": 0.0,
        "state_value_abs_error_total": 0.0,
        "action_value_return_abs_error_total": 0.0,
        "predicted_action_value_return_abs_error_total": 0.0,
        "state_value_return_abs_error_total": 0.0,
        "return_target_total": 0.0,
        "return_target_min": None,
        "return_target_max": None,
        "action_value_margin_total": 0.0,
        "predicted_advantage_total": 0.0,
        "label_counts": Counter(),
        "predicted_counts": Counter(),
        "value_margin_bucket_stats": {},
        "score_margin_bucket_stats": {},
        "predicted_advantage_bucket_stats": {},
        "label_action_stats": {},
        "predicted_action_stats": {},
    }


def _merge_neural_calibration_stats(
    target: dict[str, object],
    source: Mapping[str, object],
) -> None:
    target["record_count"] = int(target["record_count"]) + int(
        source.get("record_count", 0)
    )
    target["correct"] = int(target["correct"]) + int(source.get("correct", 0))
    for field in (
        "action_value_abs_error_total",
        "state_value_abs_error_total",
        "action_value_return_abs_error_total",
        "predicted_action_value_return_abs_error_total",
        "state_value_return_abs_error_total",
        "return_target_total",
        "action_value_margin_total",
        "predicted_advantage_total",
    ):
        target[field] = float(target[field]) + float(source.get(field, 0.0))
    target_min = target.get("return_target_min")
    source_min = source.get("return_target_min")
    if isinstance(source_min, (int, float)) and not isinstance(source_min, bool):
        target["return_target_min"] = (
            float(source_min)
            if target_min is None
            else min(float(target_min), float(source_min))
        )
    target_max = target.get("return_target_max")
    source_max = source.get("return_target_max")
    if isinstance(source_max, (int, float)) and not isinstance(source_max, bool):
        target["return_target_max"] = (
            float(source_max)
            if target_max is None
            else max(float(target_max), float(source_max))
        )
    _counter(target["label_counts"]).update(_counter(source.get("label_counts")))
    _counter(target["predicted_counts"]).update(
        _counter(source.get("predicted_counts"))
    )
    for field in (
        "value_margin_bucket_stats",
        "score_margin_bucket_stats",
        "predicted_advantage_bucket_stats",
        "label_action_stats",
        "predicted_action_stats",
    ):
        _merge_calibration_group_stats(
            _mutable_stats_mapping(target[field]),
            _stats_mapping(source.get(field)),
        )


def _finalize_neural_calibration_stats(
    stats: Mapping[str, object],
) -> dict[str, object]:
    scored_count = int(stats.get("record_count", 0))
    return_target_min = stats.get("return_target_min")
    return_target_max = stats.get("return_target_max")
    return {
        "record_count": scored_count,
        "score_normalization_policy": NEURAL_SCORE_NORMALIZATION_POLICY,
        "critic_return_target_policy": "discounted_return_target_v1",
        "critic_return_discount": TORCH_IQL_DISCOUNT,
        "critic_return_target_mean": _rate(
            float(stats.get("return_target_total", 0.0)),
            scored_count,
        ),
        "critic_return_target_min": (
            round(float(return_target_min), 4)
            if isinstance(return_target_min, (int, float))
            and not isinstance(return_target_min, bool)
            else 0.0
        ),
        "critic_return_target_max": (
            round(float(return_target_max), 4)
            if isinstance(return_target_max, (int, float))
            and not isinstance(return_target_max, bool)
            else 0.0
        ),
        "actor_top1_accuracy": _rate(int(stats.get("correct", 0)), scored_count),
        "action_value_mean_abs_error": _rate(
            float(stats.get("action_value_abs_error_total", 0.0)),
            scored_count,
        ),
        "state_value_mean_abs_error": _rate(
            float(stats.get("state_value_abs_error_total", 0.0)),
            scored_count,
        ),
        "action_value_return_mean_abs_error": _rate(
            float(stats.get("action_value_return_abs_error_total", 0.0)),
            scored_count,
        ),
        "predicted_action_value_return_mean_abs_error": _rate(
            float(
                stats.get("predicted_action_value_return_abs_error_total", 0.0)
            ),
            scored_count,
        ),
        "state_value_return_mean_abs_error": _rate(
            float(stats.get("state_value_return_abs_error_total", 0.0)),
            scored_count,
        ),
        "action_value_margin_mean": _rate(
            float(stats.get("action_value_margin_total", 0.0)),
            scored_count,
        ),
        "predicted_advantage_mean": _rate(
            float(stats.get("predicted_advantage_total", 0.0)),
            scored_count,
        ),
        "label_counts": _sorted_counts(_counter(stats.get("label_counts"))),
        "predicted_counts": _sorted_counts(
            _counter(stats.get("predicted_counts"))
        ),
        "by_score_margin_bucket": _finalize_calibration_group_stats(
            _stats_mapping(stats.get("score_margin_bucket_stats"))
        ),
        "by_action_value_margin_bucket": _finalize_calibration_group_stats(
            _stats_mapping(stats.get("value_margin_bucket_stats"))
        ),
        "by_predicted_advantage_bucket": _finalize_calibration_group_stats(
            _stats_mapping(stats.get("predicted_advantage_bucket_stats"))
        ),
        "by_label_action": _finalize_calibration_group_stats(
            _stats_mapping(stats.get("label_action_stats"))
        ),
        "by_predicted_action": _finalize_calibration_group_stats(
            _stats_mapping(stats.get("predicted_action_stats"))
        ),
    }


def _empty_viability_calibration_stats() -> dict[str, object]:
    return {
        "risk_scores": [],
        "targets": [],
        "component_scores": {
            component: []
            for component in VIABILITY_COMPONENT_NAMES
        },
        "component_targets_by_name": {
            component: []
            for component in VIABILITY_COMPONENT_NAMES
        },
        "logged_action_stats": {},
        "predicted_action_stats": {},
        "role_stats": {},
        "mode_stats": {},
        "risk_bucket_stats": {},
        "component_stats": {
            component: {}
            for component in VIABILITY_COMPONENT_NAMES
        },
        "target_counts": Counter(),
        "predicted_risk_score_total": 0.0,
        "reproduction_viable_count": 0,
        "suppressed_action_score_count": 0,
        "survival_horizons": [],
    }


def _merge_viability_calibration_stats(
    target: dict[str, object],
    source: Mapping[str, object],
) -> None:
    _list(target["risk_scores"]).extend(_float_sequence(source.get("risk_scores")))
    _list(target["targets"]).extend(_bool_sequence(source.get("targets")))
    target_component_scores = _mutable_stats_mapping(target["component_scores"])
    source_component_scores = _stats_mapping(source.get("component_scores"))
    target_component_targets = _mutable_stats_mapping(
        target["component_targets_by_name"]
    )
    source_component_targets = _stats_mapping(
        source.get("component_targets_by_name")
    )
    for component in VIABILITY_COMPONENT_NAMES:
        _list(target_component_scores[component]).extend(
            _float_sequence(source_component_scores.get(component))
        )
        _list(target_component_targets[component]).extend(
            _bool_sequence(source_component_targets.get(component))
        )
    for field in (
        "logged_action_stats",
        "predicted_action_stats",
        "role_stats",
        "mode_stats",
        "risk_bucket_stats",
    ):
        _merge_viability_group_stats(
            _mutable_stats_mapping(target[field]),
            _stats_mapping(source.get(field)),
        )
    target_component_stats = _mutable_stats_mapping(target["component_stats"])
    source_component_stats = _stats_mapping(source.get("component_stats"))
    for component in VIABILITY_COMPONENT_NAMES:
        _merge_viability_group_stats(
            _mutable_stats_mapping(target_component_stats[component]),
            _stats_mapping(source_component_stats.get(component)),
        )
    _counter(target["target_counts"]).update(_counter(source.get("target_counts")))
    target["predicted_risk_score_total"] = float(
        target["predicted_risk_score_total"]
    ) + float(source.get("predicted_risk_score_total", 0.0))
    target["reproduction_viable_count"] = int(
        target["reproduction_viable_count"]
    ) + int(source.get("reproduction_viable_count", 0))
    target["suppressed_action_score_count"] = int(
        target["suppressed_action_score_count"]
    ) + int(source.get("suppressed_action_score_count", 0))
    _list(target["survival_horizons"]).extend(
        _float_sequence(source.get("survival_horizons"))
    )


def _finalize_viability_calibration_stats(
    model: Mapping[str, object],
    stats: Mapping[str, object],
) -> dict[str, object] | None:
    network = model.get("neural_network")
    if not isinstance(network, Mapping):
        return None
    risk_scores = _float_sequence(stats.get("risk_scores"))
    targets = _bool_sequence(stats.get("targets"))
    record_count = len(risk_scores)
    positive_count = sum(1 for target in targets if target)
    brier_total = sum(
        (score - (1.0 if target else 0.0)) ** 2
        for score, target in zip(risk_scores, targets, strict=True)
    )
    component_scores = _stats_mapping(stats.get("component_scores"))
    component_targets_by_name = _stats_mapping(
        stats.get("component_targets_by_name")
    )
    component_stats = _stats_mapping(stats.get("component_stats"))
    component_calibration = {}
    for component in VIABILITY_COMPONENT_NAMES:
        scores = _float_sequence(component_scores.get(component))
        component_targets = _bool_sequence(component_targets_by_name.get(component))
        component_positive_count = sum(1 for target in component_targets if target)
        component_brier_total = sum(
            (score - (1.0 if target else 0.0)) ** 2
            for score, target in zip(scores, component_targets, strict=True)
        )
        component_calibration[component] = {
            "observed_count": len(component_targets),
            "constraint_risk_count": component_positive_count,
            "constraint_risk_rate": _rate(
                component_positive_count,
                len(component_targets),
            ),
            "risk_brier_score": _rate(
                component_brier_total,
                len(component_targets),
            ),
            "risk_auc": _binary_auc(scores, component_targets),
            "by_risk_score_bucket": _finalize_viability_group_stats(
                _stats_mapping(component_stats.get(component))
            ),
        }
    if "action_viability_component_output_weights" in network:
        risk_score_policy = VIABILITY_ACTION_HEAD_POLICY
        viability_head_conditioning = "action"
    elif "viability_component_output_weights" in network:
        risk_score_policy = VIABILITY_HEAD_POLICY
        viability_head_conditioning = "state"
    else:
        risk_score_policy = "normalized_action_value_constraint_risk_proxy_v0"
        viability_head_conditioning = "value_proxy"
    survival_horizons = _float_sequence(stats.get("survival_horizons"))
    reproduction_viable_count = int(stats.get("reproduction_viable_count", 0))
    return {
        "schema_version": "mind_viability_critic_diagnostics_v0",
        "target_policy": VIABILITY_TARGET_POLICY,
        "reproduction_diagnostic_policy": VIABILITY_REPRODUCTION_DIAGNOSTIC_POLICY,
        "component_names": list(VIABILITY_COMPONENT_NAMES),
        "risk_score_policy": risk_score_policy,
        "viability_head_conditioning": viability_head_conditioning,
        "action_conditioned_score_policy": (
            VIABILITY_ACTION_SUPERVISION_POLICY
            if viability_head_conditioning == "action"
            else None
        ),
        "action_conditioned_logged_action_only": (
            False
            if viability_head_conditioning == "action"
            else None
        ),
        "action_conditioned_suppressed_action_score_count": (
            int(stats.get("suppressed_action_score_count", 0))
            if viability_head_conditioning == "action"
            else None
        ),
        "runtime_decision_policy": VIABILITY_RUNTIME_DECISION_POLICY,
        "record_count": record_count,
        "constraint_risk_count": positive_count,
        "constraint_risk_rate": _rate(positive_count, record_count),
        "risk_brier_score": _rate(brier_total, record_count),
        "risk_auc": _binary_auc(risk_scores, targets),
        "mean_logged_action_risk_score": _rate(sum(risk_scores), record_count),
        "mean_predicted_action_risk_score": _rate(
            float(stats.get("predicted_risk_score_total", 0.0)),
            record_count,
        ),
        "survival_horizon_ticks": VIABILITY_SURVIVAL_HORIZON_TICKS,
        "survival_horizon_observed_mean": _rate(
            sum(survival_horizons),
            len(survival_horizons),
        ),
        "floor_risk_ratio": VIABILITY_FLOOR_RISK_RATIO,
        "health_floor_risk_ratio": VIABILITY_HEALTH_FLOOR_RISK_RATIO,
        "target_counts": _sorted_counts(_counter(stats.get("target_counts"))),
        "reproduction_viable_count": reproduction_viable_count,
        "reproduction_viable_rate": _rate(reproduction_viable_count, record_count),
        "by_component": component_calibration,
        "by_risk_score_bucket": _finalize_viability_group_stats(
            _stats_mapping(stats.get("risk_bucket_stats"))
        ),
        "by_logged_action": _finalize_viability_group_stats(
            _stats_mapping(stats.get("logged_action_stats"))
        ),
        "by_predicted_action": _finalize_viability_group_stats(
            _stats_mapping(stats.get("predicted_action_stats"))
        ),
        "by_trophic_role": _finalize_viability_group_stats(
            _stats_mapping(stats.get("role_stats"))
        ),
        "by_meat_mode": _finalize_viability_group_stats(
            _stats_mapping(stats.get("mode_stats"))
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


def _value_to_viability_risk_score(value: float) -> float:
    lower, upper = REWARD_TOTAL_BOUNDS
    if upper <= lower:
        return 0.5
    normalized_value = (float(value) - lower) / (upper - lower)
    return _clamp(1.0 - normalized_value, 0.0, 1.0)


def _viability_component_observed_for_diagnostics(
    component: str,
    *,
    record: Mapping[str, object],
    action_viability_scores: Mapping[str, Mapping[str, float]] | None,
    suppressed_action: str | None,
) -> bool:
    if (
        action_viability_scores is None
        or component != VIABILITY_SUPPRESSION_COMPONENT
    ):
        return True
    return suppressed_action is not None or _learned_policy_action_source(
        record.get("action_source")
    )


def _learned_policy_action_source(action_source: object) -> bool:
    return isinstance(action_source, str) and (
        action_source.startswith("mind_v1_learned_policy")
        or action_source.startswith("mind_v2_neural_policy")
    )


def _action_conditioned_component_score(
    component: str,
    *,
    logged_component_scores: Mapping[str, float],
    action_viability_scores: Mapping[str, Mapping[str, float]] | None,
    suppressed_action: str | None,
) -> float:
    if (
        component == VIABILITY_SUPPRESSION_COMPONENT
        and action_viability_scores is not None
        and suppressed_action is not None
    ):
        return float(action_viability_scores[suppressed_action][component])
    return float(logged_component_scores[component])


def _suppressed_learned_action(
    record: Mapping[str, object],
    action_mask: Mapping[str, bool],
) -> str | None:
    diagnostic = _mapping(record.get("policy_decision_diagnostics"))
    action_source = record.get("action_source")
    fallback_used = (
        (isinstance(action_source, str) and HEURISTIC_GUARD_POLICY in action_source)
        or (isinstance(action_source, str) and HEURISTIC_DELEGATE_POLICY in action_source)
        or _diagnostic_bool(diagnostic, "guard_used")
        or _diagnostic_bool(diagnostic, "heuristic_delegate_used")
    )
    if not fallback_used:
        return None
    learned_action = diagnostic.get("learned_action")
    logged_action = _training_label(record)
    if not isinstance(learned_action, str) or learned_action not in ACTION_NAMES:
        return None
    if learned_action == logged_action:
        return None
    if not bool(action_mask.get(learned_action, False)):
        return None
    return learned_action


def _risk_score_bucket(value: float) -> str:
    if value < 0.2:
        return "0.00-0.19"
    if value < 0.4:
        return "0.20-0.39"
    if value < 0.6:
        return "0.40-0.59"
    if value < 0.8:
        return "0.60-0.79"
    return "0.80-1.00"


def _binary_auc(
    scores: Sequence[float],
    targets: Sequence[bool],
) -> float | None:
    positive_count = sum(1 for target in targets if target)
    negative_count = len(targets) - positive_count
    if positive_count <= 0 or negative_count <= 0:
        return None
    ranked = sorted(
        enumerate(scores),
        key=lambda item: (float(item[1]), item[0]),
    )
    rank_sum_positive = 0.0
    rank = 1
    position = 0
    while position < len(ranked):
        next_position = position + 1
        while (
            next_position < len(ranked)
            and float(ranked[next_position][1]) == float(ranked[position][1])
        ):
            next_position += 1
        average_rank = (rank + rank + (next_position - position) - 1) / 2.0
        for ranked_index in range(position, next_position):
            original_index = int(ranked[ranked_index][0])
            if targets[original_index]:
                rank_sum_positive += average_rank
        rank += next_position - position
        position = next_position
    auc = (
        rank_sum_positive
        - positive_count * (positive_count + 1) / 2.0
    ) / float(positive_count * negative_count)
    return round(float(auc), 4)


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
    action_value_return_abs_error: float,
    state_value_return_abs_error: float,
) -> None:
    stats = groups.setdefault(
        group,
        {
            "record_count": 0,
            "correct_count": 0,
            "total_reward": 0.0,
            "action_value_abs_error_total": 0.0,
            "state_value_abs_error_total": 0.0,
            "action_value_return_abs_error_total": 0.0,
            "state_value_return_abs_error_total": 0.0,
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
    stats["action_value_return_abs_error_total"] = (
        float(stats["action_value_return_abs_error_total"])
        + action_value_return_abs_error
    )
    stats["state_value_return_abs_error_total"] = (
        float(stats["state_value_return_abs_error_total"])
        + state_value_return_abs_error
    )


def _update_viability_group_stats(
    groups: dict[str, dict[str, object]],
    group: str,
    *,
    risk_score: float,
    target: bool,
) -> None:
    stats = groups.setdefault(
        group,
        {
            "record_count": 0,
            "constraint_risk_count": 0,
            "risk_score_total": 0.0,
            "brier_total": 0.0,
        },
    )
    target_value = 1.0 if target else 0.0
    stats["record_count"] = int(stats["record_count"]) + 1
    if target:
        stats["constraint_risk_count"] = int(stats["constraint_risk_count"]) + 1
    stats["risk_score_total"] = float(stats["risk_score_total"]) + risk_score
    stats["brier_total"] = float(stats["brier_total"]) + (
        risk_score - target_value
    ) ** 2


def _merge_reward_stats_groups(
    target: dict[str, dict[str, object]],
    source: Mapping[str, object],
) -> None:
    for group, raw_stats in source.items():
        if not isinstance(raw_stats, Mapping):
            continue
        stats = target.setdefault(
            str(group),
            {
                "record_count": 0,
                "total_reward": 0.0,
            },
        )
        stats["record_count"] = int(stats["record_count"]) + int(
            raw_stats.get("record_count", 0)
        )
        stats["total_reward"] = float(stats["total_reward"]) + float(
            raw_stats.get("total_reward", 0.0)
        )


def _merge_calibration_group_stats(
    target: dict[str, dict[str, object]],
    source: Mapping[str, object],
) -> None:
    for group, raw_stats in source.items():
        if not isinstance(raw_stats, Mapping):
            continue
        stats = target.setdefault(
            str(group),
            {
                "record_count": 0,
                "correct_count": 0,
                "total_reward": 0.0,
                "action_value_abs_error_total": 0.0,
                "state_value_abs_error_total": 0.0,
                "action_value_return_abs_error_total": 0.0,
                "state_value_return_abs_error_total": 0.0,
            },
        )
        stats["record_count"] = int(stats["record_count"]) + int(
            raw_stats.get("record_count", 0)
        )
        stats["correct_count"] = int(stats["correct_count"]) + int(
            raw_stats.get("correct_count", 0)
        )
        for field in (
            "total_reward",
            "action_value_abs_error_total",
            "state_value_abs_error_total",
            "action_value_return_abs_error_total",
            "state_value_return_abs_error_total",
        ):
            stats[field] = float(stats[field]) + float(raw_stats.get(field, 0.0))


def _merge_viability_group_stats(
    target: dict[str, dict[str, object]],
    source: Mapping[str, object],
) -> None:
    for group, raw_stats in source.items():
        if not isinstance(raw_stats, Mapping):
            continue
        stats = target.setdefault(
            str(group),
            {
                "record_count": 0,
                "constraint_risk_count": 0,
                "risk_score_total": 0.0,
                "brier_total": 0.0,
            },
        )
        stats["record_count"] = int(stats["record_count"]) + int(
            raw_stats.get("record_count", 0)
        )
        stats["constraint_risk_count"] = int(
            stats["constraint_risk_count"]
        ) + int(raw_stats.get("constraint_risk_count", 0))
        stats["risk_score_total"] = float(stats["risk_score_total"]) + float(
            raw_stats.get("risk_score_total", 0.0)
        )
        stats["brier_total"] = float(stats["brier_total"]) + float(
            raw_stats.get("brier_total", 0.0)
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
            "action_value_return_mean_abs_error": _rate(
                float(stats["action_value_return_abs_error_total"]),
                record_count,
            ),
            "state_value_return_mean_abs_error": _rate(
                float(stats["state_value_return_abs_error_total"]),
                record_count,
            ),
        }
    return finalized


def _finalize_viability_group_stats(
    groups: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    finalized: dict[str, dict[str, object]] = {}
    for group, stats in sorted(groups.items()):
        record_count = int(stats["record_count"])
        constraint_risk_count = int(stats["constraint_risk_count"])
        finalized[group] = {
            "record_count": record_count,
            "constraint_risk_count": constraint_risk_count,
            "constraint_risk_rate": _rate(
                constraint_risk_count,
                record_count,
            ),
            "mean_risk_score": _rate(
                float(stats["risk_score_total"]),
                record_count,
            ),
            "risk_brier_score": _rate(float(stats["brier_total"]), record_count),
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


def _counter(payload: object) -> Counter[str]:
    if isinstance(payload, Counter):
        return payload
    return Counter()


def _tuple_counter(payload: object) -> Counter[tuple[str, str]]:
    if isinstance(payload, Counter):
        return payload
    return Counter()


def _stats_mapping(payload: object) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    return {}


def _mutable_stats_mapping(payload: object) -> dict[str, object]:
    if isinstance(payload, dict):
        return payload
    raise TypeError("diagnostic stats payload must be a mutable dictionary")


def _list(payload: object) -> list[object]:
    if isinstance(payload, list):
        return payload
    raise TypeError("diagnostic stats payload must be a list")


def _float_sequence(payload: object) -> list[float]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return []
    return [
        float(value)
        for value in payload
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def _bool_sequence(payload: object) -> list[bool]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return []
    return [bool(value) for value in payload]


def _rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / denominator, 4)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
