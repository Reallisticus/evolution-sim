from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.dataset import (
    TRAJECTORY_DATASET_RECORD_INDEX_FIELD,
    TRAJECTORY_EPISODE_ID_FIELD,
    TrajectoryJsonlDataset,
    records_with_trajectory_context,
)
from evolution_sim.mind.rollout_context import RolloutContextConfig
from evolution_sim.mind.rollout_context_audit import (
    DEFAULT_HELDOUT_FRACTION,
    ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID,
    RolloutContextAuditError,
    _assign_split,
    _contextual_rows,
    _evaluate_counts,
    _float,
    _mapping,
    _predict,
    _round,
    _source_summary,
    _split_summary,
    _train_counts,
    load_rollout_context_audit_datasets,
)

MIND_V3_HYDRATION_AFTER_CARRION_AUDIT_SCHEMA_VERSION = (
    "mind_v3_hydration_after_carrion_audit_v1"
)
MIND_V3_HYDRATION_AFTER_CARRION_AUDIT_POLICY = (
    "post_carrion_hydration_intervention_vs_eat_alias_v1"
)
DEFAULT_HYDRATION_THRESHOLDS: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
DEFAULT_MIN_TRUE_DRINK_EAT_RATE_REDUCTION = 0.05
DEFAULT_MAX_EAT_LABEL_DAMAGE_RATE = 0.01
DEFAULT_MIN_CYCLE_ALIAS_SHARE = 0.6


class HydrationAfterCarrionAuditError(ValueError):
    pass


def build_hydration_after_carrion_audit_report(
    records: Sequence[Mapping[str, object]],
    *,
    source_paths: Sequence[str | Path] | None = None,
    heldout_seed_values: Iterable[int] | None = None,
    heldout_source_patterns: Sequence[str] = (),
    heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
    hydration_thresholds: Sequence[float] = DEFAULT_HYDRATION_THRESHOLDS,
    min_true_drink_eat_rate_reduction: float = (
        DEFAULT_MIN_TRUE_DRINK_EAT_RATE_REDUCTION
    ),
    max_eat_label_damage_rate: float = DEFAULT_MAX_EAT_LABEL_DAMAGE_RATE,
    min_cycle_alias_share: float = DEFAULT_MIN_CYCLE_ALIAS_SHARE,
    context_config: RolloutContextConfig | None = None,
) -> dict[str, object]:
    config = context_config or RolloutContextConfig()
    resolved_records = [dict(record) for record in records]
    if not resolved_records:
        raise HydrationAfterCarrionAuditError("audit records must not be empty")
    thresholds = _validated_thresholds(hydration_thresholds)
    if min_true_drink_eat_rate_reduction < 0.0:
        raise HydrationAfterCarrionAuditError(
            "min_true_drink_eat_rate_reduction must be non-negative"
        )
    if max_eat_label_damage_rate < 0.0:
        raise HydrationAfterCarrionAuditError(
            "max_eat_label_damage_rate must be non-negative"
        )
    if min_cycle_alias_share < 0.0 or min_cycle_alias_share > 1.0:
        raise HydrationAfterCarrionAuditError(
            "min_cycle_alias_share must be in [0.0, 1.0]"
        )

    heldout_seeds = (
        {int(seed) for seed in heldout_seed_values}
        if heldout_seed_values is not None
        else set()
    )
    contextual_rows = _contextual_rows(resolved_records, config=config)
    examples = [
        row
        for row in contextual_rows
        if row.label in row.valid_actions and row.action_source != "passive"
    ]
    if len(examples) < 2:
        raise HydrationAfterCarrionAuditError(
            "audit needs at least two policy action rows"
        )

    split = _assign_split(
        examples,
        heldout_seeds=heldout_seeds,
        heldout_source_patterns=heldout_source_patterns,
        heldout_fraction=heldout_fraction,
    )
    train_examples = [example for example in examples if split[example.row_id] == "train"]
    heldout_examples = [
        example for example in examples if split[example.row_id] == "heldout"
    ]
    if not train_examples or not heldout_examples:
        raise HydrationAfterCarrionAuditError(
            "train/held-out split must include at least one row on each side"
        )

    counts = _train_counts(
        train_examples,
        key_policy=ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID,
    )
    baseline_eval = _evaluate_counts(
        heldout_examples,
        counts,
        key_policy=ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID,
    )
    predictions = {
        example.row_id: _predict(
            example,
            counts,
            key_policy=ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID,
        )
        for example in heldout_examples
    }
    focus_examples = [
        example
        for example in heldout_examples
        if _is_post_carrion(example)
        and example.label in ("drink", "eat")
        and "drink" in example.valid_actions
        and "eat" in example.valid_actions
    ]
    residual_drink_eat = [
        example
        for example in focus_examples
        if example.label == "drink" and predictions[example.row_id] == "eat"
    ]
    future_by_row_id = _future_rows_by_row_id(contextual_rows)
    intervention_summaries = [
        _hydration_intervention_summary(
            focus_examples,
            predictions=predictions,
            threshold=threshold,
        )
        for threshold in thresholds
    ]
    best_intervention = _best_intervention(intervention_summaries)
    cycle_alias = _cycle_alias_summary(
        residual_drink_eat,
        future_by_row_id=future_by_row_id,
    )
    assessment = _assessment(
        best_intervention,
        cycle_alias,
        min_true_drink_eat_rate_reduction=min_true_drink_eat_rate_reduction,
        max_eat_label_damage_rate=max_eat_label_damage_rate,
        min_cycle_alias_share=min_cycle_alias_share,
    )
    return {
        "schema_version": MIND_V3_HYDRATION_AFTER_CARRION_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_HYDRATION_AFTER_CARRION_AUDIT_POLICY,
        "source_trajectories": _source_summary(
            resolved_records,
            source_paths=source_paths,
        ),
        "train_heldout_split": _split_summary(
            examples,
            split,
            heldout_seeds=heldout_seeds,
            heldout_source_patterns=heldout_source_patterns,
            heldout_fraction=heldout_fraction,
        ),
        "baseline_model": {
            "model_id": ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID,
            "record_count": baseline_eval["record_count"],
            "accuracy": baseline_eval["accuracy"],
            "top_confusions": baseline_eval["top_confusions"],
        },
        "focus": _focus_summary(focus_examples, predictions),
        "residual_true_drink_predicted_eat": {
            "record_count": len(residual_drink_eat),
            "examples": [_residual_example(example) for example in residual_drink_eat[:12]],
        },
        "hydration_interventions": intervention_summaries,
        "best_hydration_intervention": best_intervention,
        "cycle_alias_assessment": cycle_alias,
        "failure_mode_assessment": assessment,
    }


def build_hydration_after_carrion_audit_report_from_datasets(
    datasets: Sequence[TrajectoryJsonlDataset],
    **kwargs: object,
) -> dict[str, object]:
    records = tuple(records_with_trajectory_context(datasets))
    return build_hydration_after_carrion_audit_report(
        records,
        source_paths=[dataset.path for dataset in datasets],
        **kwargs,
    )


def write_hydration_after_carrion_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def load_hydration_after_carrion_audit_datasets(
    paths: Sequence[str | Path],
) -> tuple[TrajectoryJsonlDataset, ...]:
    try:
        return load_rollout_context_audit_datasets(paths)
    except RolloutContextAuditError as exc:
        raise HydrationAfterCarrionAuditError(str(exc)) from exc


def _focus_summary(
    examples: Sequence[object],
    predictions: Mapping[str, str],
) -> dict[str, object]:
    label_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    for example in examples:
        label_counts[example.label] += 1
        predicted = predictions[example.row_id]
        prediction_counts[predicted] += 1
        pair_counts[f"{example.label}->{predicted}"] += 1
    true_drink_count = label_counts.get("drink", 0)
    drink_predicted_eat_count = pair_counts.get("drink->eat", 0)
    return {
        "policy": "post_carrion_rows_with_drink_and_eat_valid",
        "record_count": len(examples),
        "label_counts": _ordered_counts(label_counts),
        "prediction_counts": _ordered_counts(prediction_counts),
        "label_prediction_counts": dict(sorted(pair_counts.items())),
        "true_drink_predicted_as_eat_count": drink_predicted_eat_count,
        "true_drink_predicted_as_eat_rate": _round(
            drink_predicted_eat_count / float(true_drink_count)
            if true_drink_count
            else 0.0
        ),
    }


def _hydration_intervention_summary(
    examples: Sequence[object],
    *,
    predictions: Mapping[str, str],
    threshold: float,
) -> dict[str, object]:
    true_drink_count = 0
    true_eat_count = 0
    baseline_drink_predicted_eat = 0
    intervention_drink_predicted_eat = 0
    selected_count = 0
    selected_labels: Counter[str] = Counter()
    eat_label_forced_drink_count = 0
    eat_label_changed_from_correct_count = 0
    examples_payload: list[dict[str, object]] = []
    for example in examples:
        baseline = predictions[example.row_id]
        predicted = baseline
        selected = _should_force_drink(example, threshold=threshold)
        if selected:
            selected_count += 1
            selected_labels[example.label] += 1
            predicted = "drink"
            if len(examples_payload) < 8:
                examples_payload.append(_intervention_example(example, baseline))
        if example.label == "drink":
            true_drink_count += 1
            if baseline == "eat":
                baseline_drink_predicted_eat += 1
            if predicted == "eat":
                intervention_drink_predicted_eat += 1
        elif example.label == "eat":
            true_eat_count += 1
            if selected:
                eat_label_forced_drink_count += 1
                if baseline == "eat":
                    eat_label_changed_from_correct_count += 1

    baseline_rate = _rate(baseline_drink_predicted_eat, true_drink_count)
    intervention_rate = _rate(intervention_drink_predicted_eat, true_drink_count)
    return {
        "policy": "force_drink_when_post_carrion_hydration_below_threshold_v1",
        "hydration_threshold": _round(threshold),
        "selected_count": selected_count,
        "selected_label_counts": _ordered_counts(selected_labels),
        "true_drink_count": true_drink_count,
        "baseline_true_drink_predicted_as_eat_count": baseline_drink_predicted_eat,
        "intervention_true_drink_predicted_as_eat_count": (
            intervention_drink_predicted_eat
        ),
        "baseline_true_drink_predicted_as_eat_rate": baseline_rate,
        "intervention_true_drink_predicted_as_eat_rate": intervention_rate,
        "true_drink_predicted_eat_absolute_rate_reduction": _round(
            baseline_rate - intervention_rate
        ),
        "true_eat_count": true_eat_count,
        "eat_label_forced_drink_count": eat_label_forced_drink_count,
        "eat_label_changed_from_correct_count": eat_label_changed_from_correct_count,
        "eat_label_damage_rate": _rate(eat_label_changed_from_correct_count, true_eat_count),
        "examples": examples_payload,
    }


def _cycle_alias_summary(
    residuals: Sequence[object],
    *,
    future_by_row_id: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    current_drink_succeeded = 0
    next_eat_with_gain_1 = 0
    next_eat_with_gain_2 = 0
    next_eat_with_gain_3 = 0
    died_after_current = 0
    examples: list[dict[str, object]] = []
    for example in residuals:
        future = tuple(future_by_row_id.get(example.row_id, ()))
        if _drank(example):
            current_drink_succeeded += 1
        if _died_after(example):
            died_after_current += 1
        if _future_eat_with_gain(future, limit=1):
            next_eat_with_gain_1 += 1
        if _future_eat_with_gain(future, limit=2):
            next_eat_with_gain_2 += 1
        if _future_eat_with_gain(future, limit=3):
            next_eat_with_gain_3 += 1
        if len(examples) < 12:
            examples.append(_cycle_example(example, future[:3]))
    denominator = len(residuals)
    return {
        "policy": "same_agent_future_rows_diagnostic_only_v1",
        "residual_count": denominator,
        "current_drink_succeeded_count": current_drink_succeeded,
        "current_drink_succeeded_share": _rate(current_drink_succeeded, denominator),
        "next_eat_with_resource_gain_within_1_count": next_eat_with_gain_1,
        "next_eat_with_resource_gain_within_1_share": _rate(
            next_eat_with_gain_1,
            denominator,
        ),
        "next_eat_with_resource_gain_within_2_count": next_eat_with_gain_2,
        "next_eat_with_resource_gain_within_2_share": _rate(
            next_eat_with_gain_2,
            denominator,
        ),
        "next_eat_with_resource_gain_within_3_count": next_eat_with_gain_3,
        "next_eat_with_resource_gain_within_3_share": _rate(
            next_eat_with_gain_3,
            denominator,
        ),
        "died_after_current_count": died_after_current,
        "examples": examples,
    }


def _assessment(
    best_intervention: Mapping[str, object],
    cycle_alias: Mapping[str, object],
    *,
    min_true_drink_eat_rate_reduction: float,
    max_eat_label_damage_rate: float,
    min_cycle_alias_share: float,
) -> dict[str, object]:
    reduction = _float(
        best_intervention.get("true_drink_predicted_eat_absolute_rate_reduction")
    )
    damage = _float(best_intervention.get("eat_label_damage_rate"))
    cycle_share = _float(
        cycle_alias.get("next_eat_with_resource_gain_within_2_share")
    )
    hydration_positive = (
        reduction >= min_true_drink_eat_rate_reduction
        and damage <= max_eat_label_damage_rate
    )
    cycle_positive = cycle_share >= min_cycle_alias_share
    blockers: list[dict[str, object]] = []
    if reduction < min_true_drink_eat_rate_reduction:
        blockers.append(
            {
                "reason": "hydration_intervention_true_drink_eat_reduction_below_floor",
                "required_absolute_rate_reduction": _round(
                    min_true_drink_eat_rate_reduction
                ),
                "observed_absolute_rate_reduction": reduction,
            }
        )
    if damage > max_eat_label_damage_rate:
        blockers.append(
            {
                "reason": "hydration_intervention_damages_eat_labels",
                "max_eat_label_damage_rate": _round(max_eat_label_damage_rate),
                "observed_eat_label_damage_rate": damage,
            }
        )
    if cycle_share < min_cycle_alias_share:
        blockers.append(
            {
                "reason": "drink_then_eat_cycle_alias_share_below_floor",
                "required_cycle_alias_share": _round(min_cycle_alias_share),
                "observed_cycle_alias_share": cycle_share,
            }
        )
    return {
        "status": "audit_pass" if hydration_positive or cycle_positive else "audit_fail",
        "materially_supports_hydration_intervention": hydration_positive,
        "materially_supports_temporal_option_alias": cycle_positive,
        "required_true_drink_eat_absolute_rate_reduction": _round(
            min_true_drink_eat_rate_reduction
        ),
        "max_eat_label_damage_rate": _round(max_eat_label_damage_rate),
        "required_cycle_alias_share": _round(min_cycle_alias_share),
        "blockers": blockers,
    }


def _best_intervention(
    interventions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not interventions:
        return {}
    return dict(
        max(
            interventions,
            key=lambda item: (
                _float(item.get("true_drink_predicted_eat_absolute_rate_reduction")),
                -_float(item.get("eat_label_damage_rate")),
                -_float(item.get("hydration_threshold")),
            ),
        )
    )


def _future_rows_by_row_id(
    contextual_rows: Sequence[object],
) -> dict[str, tuple[object, ...]]:
    by_agent: dict[tuple[str, str, int], list[object]] = {}
    for row in contextual_rows:
        record = row.record
        episode_id = str(record.get(TRAJECTORY_EPISODE_ID_FIELD, ""))
        key = (row.source_path, episode_id, _agent_id(record))
        by_agent.setdefault(key, []).append(row)
    result: dict[str, tuple[object, ...]] = {}
    for rows in by_agent.values():
        rows.sort(key=lambda row: _record_index(row.record))
        for index, row in enumerate(rows):
            result[row.row_id] = tuple(rows[index + 1 : index + 4])
    return result


def _should_force_drink(example: object, *, threshold: float) -> bool:
    return (
        _is_post_carrion(example)
        and "drink" in example.valid_actions
        and "eat" in example.valid_actions
        and _before_ratio(example, "hydration_ratio") < threshold
    )


def _is_post_carrion(example: object) -> bool:
    return bool(example.context_snapshot.get("post_carrion_contact", False))


def _before_ratio(example: object, field: str) -> float:
    before = _mapping(example.record.get("before"))
    return _float(before.get(field))


def _drank(example: object) -> bool:
    outcome = _mapping(example.record.get("outcome"))
    drinking = _mapping(outcome.get("drinking"))
    return bool(drinking.get("drank", False))


def _eat_with_gain(example: object) -> bool:
    outcome = _mapping(example.record.get("outcome"))
    feeding = _mapping(outcome.get("feeding"))
    return bool(feeding.get("ate", False)) and _float(outcome.get("resource_gain")) > 0.0


def _future_eat_with_gain(future: Sequence[object], *, limit: int) -> bool:
    return any(row.label == "eat" and _eat_with_gain(row) for row in future[:limit])


def _died_after(example: object) -> bool:
    after = _mapping(example.record.get("after"))
    return after.get("alive") is False


def _residual_example(example: object) -> dict[str, object]:
    return {
        "row_id": example.row_id,
        "source_path": example.source_path,
        "source_seed": example.source_seed,
        "tick": example.record.get("tick"),
        "agent_id": example.record.get("agent_id"),
        "label": example.label,
        "predicted": "eat",
        "energy_ratio": _round(_before_ratio(example, "energy_ratio")),
        "hydration_ratio": _round(_before_ratio(example, "hydration_ratio")),
        "health_ratio": _round(_before_ratio(example, "health_ratio")),
    }


def _intervention_example(example: object, baseline: str) -> dict[str, object]:
    return {
        "row_id": example.row_id,
        "source_path": example.source_path,
        "source_seed": example.source_seed,
        "tick": example.record.get("tick"),
        "agent_id": example.record.get("agent_id"),
        "label": example.label,
        "baseline_predicted": baseline,
        "intervention_predicted": "drink",
        "energy_ratio": _round(_before_ratio(example, "energy_ratio")),
        "hydration_ratio": _round(_before_ratio(example, "hydration_ratio")),
        "health_ratio": _round(_before_ratio(example, "health_ratio")),
    }


def _cycle_example(example: object, future: Sequence[object]) -> dict[str, object]:
    payload = _residual_example(example)
    payload["current_drink_succeeded"] = _drank(example)
    payload["future"] = [
        {
            "tick": row.record.get("tick"),
            "label": row.label,
            "ate_with_resource_gain": _eat_with_gain(row),
            "drank": _drank(row),
        }
        for row in future
    ]
    return payload


def _ordered_counts(counts: Counter[str]) -> dict[str, int]:
    return {action: int(counts.get(action, 0)) for action in ACTION_NAMES}


def _record_index(record: Mapping[str, object]) -> int:
    value = record.get(TRAJECTORY_DATASET_RECORD_INDEX_FIELD)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return int(value)


def _agent_id(record: Mapping[str, object]) -> int:
    value = record.get("agent_id")
    if isinstance(value, bool) or not isinstance(value, int):
        return -1
    return int(value)


def _validated_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    thresholds: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HydrationAfterCarrionAuditError(
                "hydration thresholds must be finite numbers"
            )
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
            raise HydrationAfterCarrionAuditError(
                "hydration thresholds must be in [0.0, 1.0]"
            )
        thresholds.append(parsed)
    if not thresholds:
        raise HydrationAfterCarrionAuditError(
            "at least one hydration threshold is required"
        )
    return tuple(sorted(set(thresholds)))


def _rate(numerator: int, denominator: int) -> float:
    return _round(numerator / float(denominator) if denominator else 0.0)


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
