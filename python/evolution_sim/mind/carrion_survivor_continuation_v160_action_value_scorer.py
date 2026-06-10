from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    _dominant_count_share,
    _float,
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V158_DATASET_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION,
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
    load_v154_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V159_REPORT_PATH,
    EXPECTED_V158_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V159_SCORER_READINESS_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V159_SCORER_READINESS_SCHEMA_VERSION,
    _complete_bool_mask,
    _first_public_action,
    _safe_action_set,
    _top_value_target,
    validate_v158_target_rows,
    v159_target_dataset_leakage_scan,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_action_value_scorer_training_diagnostic_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_action_value_scorer_diagnostic_artifact_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_action_value_scorer_training_v1"
)
EXPECTED_V159_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v159_scorer_readiness_"
    "recommends_separate_opt_in_training_diagnostic_no_training"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v160-carrion-survivor-continuation-action-value-scorer-training.json"
)
DEFAULT_ARTIFACT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v160-carrion-survivor-continuation-action-value-scorer-artifact.json"
)
DEFAULT_MAX_DOMINANT_PREDICTED_ACTION_SHARE = 0.50
DEFAULT_MIN_SAFE_HIT_MARGIN_OVER_BEST_TRIVIAL = 0.05
K_NEIGHBORS = 1


class CarrionSurvivorContinuationV160ActionValueScorerError(ValueError):
    pass


def run_carrion_survivor_continuation_v160_action_value_scorer(
    *,
    v158_dataset_path: str | Path = DEFAULT_V158_DATASET_PATH,
    v159_report_path: str | Path = DEFAULT_V159_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    artifact_output_path: str | Path = DEFAULT_ARTIFACT_OUTPUT_PATH,
    max_dominant_predicted_action_share: float = (
        DEFAULT_MAX_DOMINANT_PREDICTED_ACTION_SHARE
    ),
    min_safe_hit_margin_over_best_trivial: float = (
        DEFAULT_MIN_SAFE_HIT_MARGIN_OVER_BEST_TRIVIAL
    ),
) -> dict[str, object]:
    rows = load_v154_dataset(v158_dataset_path)
    v159_report = load_json_report(v159_report_path)
    source_validation = validate_v160_sources(
        v159_report=v159_report,
        rows=rows,
    )
    row_contract = validate_v158_target_rows(rows)
    leakage_scan = v159_target_dataset_leakage_scan(rows)
    artifact = build_diagnostic_scorer_artifact(
        rows=rows,
        source_validation=source_validation,
        row_contract=row_contract,
        leakage_scan=leakage_scan,
    )
    write_json(artifact_output_path, artifact)
    loo = leave_one_row_out_diagnostics(
        rows=rows,
        max_dominant_predicted_action_share=max_dominant_predicted_action_share,
        min_safe_hit_margin_over_best_trivial=(
            min_safe_hit_margin_over_best_trivial
        ),
    )
    classification = _top_level_classification(
        source_validation=source_validation,
        row_contract=row_contract,
        leakage_scan=leakage_scan,
        loo=loo,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_POLICY,
        "contract": {
            "diagnostics_only": True,
            "serialized_scorer_artifact_created": True,
            "runtime_artifact_created": False,
            "model_artifact_creation_allowed_for_runtime": False,
            "runtime_policy_integration_allowed": False,
            "shadow_live_ab_allowed": False,
            "live_runtime_override_allowed": False,
            "training_scope": "diagnostics_only_leave_one_row_out_public_action_value_scorer",
            "uses_only_public_trainable_features": True,
            "uses_public_action_masks": True,
            "uses_safe_action_set_targets": True,
            "uses_action_value_targets": True,
            "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_future_outcome_as_trainable_input": False,
            "uses_branch_reason_as_trainable_input": False,
            "training_authorized_for_runtime": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
        },
        "inputs": {
            "v158_dataset": str(v158_dataset_path),
            "v159_report": str(v159_report_path),
            "artifact_output": str(artifact_output_path),
            "k_neighbors": K_NEIGHBORS,
            "max_dominant_predicted_action_share": _round(
                max_dominant_predicted_action_share
            ),
            "min_safe_hit_margin_over_best_trivial": _round(
                min_safe_hit_margin_over_best_trivial
            ),
        },
        "source_validation": source_validation,
        "row_contract_validation": row_contract,
        "leakage_scan": leakage_scan,
        "artifact": {
            "path": str(artifact_output_path),
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION
            ),
            "scorer_type": "public_feature_mask_1nn_action_value_scorer_v1",
            "diagnostics_only": True,
            "runtime_artifact": False,
            "artifact_digest": stable_payload_digest(artifact),
        },
        "leave_one_row_out": loo,
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "dataset_digest": stable_payload_digest(rows),
        "artifact_created": True,
        "runtime_artifact_created": False,
        "diagnostic_training_ran": True,
        "training_ran": True,
        "shadow_live_ab_ran": False,
        "training_authorized_for_runtime": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = stable_payload_digest(report)
    write_json(output_path, report)
    return report


def validate_v160_sources(
    *,
    v159_report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    if (
        v159_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V159_SCORER_READINESS_SCHEMA_VERSION
    ):
        failures.append("v159_schema_version_mismatch")
    if (
        v159_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V159_SCORER_READINESS_POLICY
    ):
        failures.append("v159_policy_mismatch")
    observed_v159_classification = _mapping(v159_report.get("classification")).get(
        "primary"
    )
    if observed_v159_classification != EXPECTED_V159_CLASSIFICATION:
        failures.append("v159_unexpected_classification")
    route = _mapping(v159_report.get("route_recommendation"))
    if route.get("future_opt_in_training_diagnostic_recommended") is not True:
        failures.append("v159_training_diagnostic_not_recommended")
    source = _mapping(v159_report.get("source_validation"))
    if source.get("passed") is not True:
        failures.append("v159_source_validation_failed")
    if source.get("expected_v158_classification") != EXPECTED_V158_CLASSIFICATION:
        failures.append("v159_expected_v158_classification_mismatch")
    if _mapping(v159_report.get("row_contract_validation")).get("passed") is not True:
        failures.append("v159_row_contract_validation_failed")
    if _mapping(v159_report.get("leakage_scan")).get("passed") is not True:
        failures.append("v159_leakage_scan_failed")
    exact_digest_validation = exact_digest_validation_report(v159_report)
    if exact_digest_validation.get("passed") is not True:
        failures.append("v159_exact_digest_mismatch")
    dataset_digest = stable_payload_digest(rows)
    if v159_report.get("dataset_digest") != dataset_digest:
        failures.append("v159_dataset_digest_mismatch")
    lifecycle = _v160_source_lifecycle_scan(v159_report)
    if lifecycle.get("passed") is not True:
        failures.append("v159_lifecycle_fields_not_false")
    return {
        "policy": "m3_carrion_survivor_continuation_v160_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v159_classification": EXPECTED_V159_CLASSIFICATION,
        "observed_v159_classification": observed_v159_classification,
        "expected_v158_classification": EXPECTED_V158_CLASSIFICATION,
        "dataset_digest": dataset_digest,
        "v159_reported_dataset_digest": v159_report.get("dataset_digest"),
        "v159_exact_digest_validation": exact_digest_validation,
        "lifecycle_authorization": lifecycle,
        "row_count": len(rows),
    }


def build_diagnostic_scorer_artifact(
    *,
    rows: Sequence[Mapping[str, object]],
    source_validation: Mapping[str, object],
    row_contract: Mapping[str, object],
    leakage_scan: Mapping[str, object],
) -> dict[str, object]:
    vectors, feature_keys, normalization = _feature_vectors(rows)
    training_rows = []
    for row_index, row in enumerate(rows):
        training_rows.append(
            {
                "row_index": row_index,
                "feature_vector": _sparse_vector(vectors[row_index]),
                "public_action_mask": _complete_bool_mask(row.get("public_action_mask")),
                "safe_action_set": _safe_action_set(row),
                "action_value_targets": _artifact_action_value_targets(row),
            }
        )
    artifact = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_POLICY,
        "scorer_type": "public_feature_mask_1nn_action_value_scorer_v1",
        "contract": {
            "diagnostics_only": True,
            "runtime_artifact": False,
            "runtime_policy_integration_allowed": False,
            "training_scope": "diagnostics_only_public_features_action_value_targets",
            "k_neighbors": K_NEIGHBORS,
            "uses_only_public_trainable_features": True,
            "uses_public_action_masks": True,
            "uses_safe_action_set_targets": True,
            "uses_action_value_targets": True,
            "uses_private_world_state_as_trainable_input": False,
            "uses_future_outcome_as_trainable_input": False,
            "uses_branch_reason_as_trainable_input": False,
            "runtime_action_selection_changed": False,
        },
        "source": {
            "dataset_digest": stable_payload_digest(rows),
            "source_validation_passed": source_validation.get("passed") is True,
            "row_contract_validation_passed": row_contract.get("passed") is True,
            "leakage_scan_passed": leakage_scan.get("passed") is True,
            "row_count": len(rows),
        },
        "feature_keys": feature_keys,
        "normalization": normalization,
        "training_rows": training_rows,
    }
    artifact["exact_digest"] = stable_payload_digest(artifact)
    return artifact


def leave_one_row_out_diagnostics(
    *,
    rows: Sequence[Mapping[str, object]],
    max_dominant_predicted_action_share: float = (
        DEFAULT_MAX_DOMINANT_PREDICTED_ACTION_SHARE
    ),
    min_safe_hit_margin_over_best_trivial: float = (
        DEFAULT_MIN_SAFE_HIT_MARGIN_OVER_BEST_TRIVIAL
    ),
) -> dict[str, object]:
    vectors, feature_keys, normalization = _feature_vectors(rows)
    predictions = []
    scorer_counts: Counter[str] = Counter()
    safe_hit_count = 0
    unsupported_prediction_count = 0
    no_prediction_count = 0
    best_fixed_hit_count = 0
    first_public_hit_count = 0
    exact_top_value_hit_count = 0
    for row_index, row in enumerate(rows):
        train_indexes = [index for index in range(len(rows)) if index != row_index]
        scorer_action, source, neighbor_indexes = _predict_1nn_action(
            rows=rows,
            vectors=vectors,
            feature_keys=feature_keys,
            normalization=normalization,
            row_index=row_index,
            train_indexes=train_indexes,
        )
        safe_set = set(_safe_action_set(row))
        public_mask = _complete_bool_mask(row.get("public_action_mask"))
        if scorer_action:
            scorer_counts.update([scorer_action])
            safe_hit_count += int(scorer_action in safe_set)
            unsupported_prediction_count += int(public_mask.get(scorer_action) is not True)
        else:
            no_prediction_count += 1
        best_fixed = _best_fixed_safe_action(rows, train_indexes, public_mask)
        best_fixed_hit_count += int(best_fixed in safe_set)
        first_public = _first_public_action(row)
        first_public_hit_count += int(first_public in safe_set)
        exact_top = str(_top_value_target(row).get("action", ""))
        exact_top_value_hit_count += int(exact_top in safe_set)
        predictions.append(
            {
                "row_index": row_index,
                "predicted_action": scorer_action,
                "prediction_source": source,
                "nearest_neighbor_row_indexes": neighbor_indexes,
                "safe_hit": scorer_action in safe_set if scorer_action else False,
                "unsupported_prediction": (
                    public_mask.get(scorer_action) is not True
                    if scorer_action
                    else False
                ),
                "safe_action_set": sorted(safe_set, key=_action_order),
                "best_fixed_safe_action_baseline": best_fixed,
                "first_public_action_baseline": first_public,
                "v159_exact_top_value_upper_bound_action": exact_top,
            }
        )
    dominant = _dominant_count_share(_positive_counter(scorer_counts))
    row_count = len(rows)
    safe_hit_rate = _safe_rate(safe_hit_count, row_count)
    best_fixed_rate = _safe_rate(best_fixed_hit_count, row_count)
    first_public_rate = _safe_rate(first_public_hit_count, row_count)
    best_trivial_rate = max(best_fixed_rate, first_public_rate)
    exact_top_rate = _safe_rate(exact_top_value_hit_count, row_count)
    margin = _round(safe_hit_rate - best_trivial_rate)
    unsupported_floor_passed = unsupported_prediction_count == 0 and no_prediction_count == 0
    dominance_floor_passed = (
        _float(dominant.get("share")) <= float(max_dominant_predicted_action_share)
        and bool(scorer_counts)
    )
    safe_hit_margin_floor_passed = (
        margin >= float(min_safe_hit_margin_over_best_trivial)
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v160_leave_one_row_out_diagnostics_v1",
        "scorer_type": "public_feature_mask_1nn_action_value_scorer_v1",
        "k_neighbors": K_NEIGHBORS,
        "row_count": row_count,
        "prediction_count": sum(scorer_counts.values()),
        "no_prediction_count": no_prediction_count,
        "unsupported_prediction_count": unsupported_prediction_count,
        "unsupported_prediction_floor_passed": unsupported_floor_passed,
        "predicted_action_counts": dict(sorted(scorer_counts.items())),
        "dominant_predicted_action": dominant.get("key"),
        "dominant_predicted_action_count": dominant.get("count"),
        "dominant_predicted_action_share": dominant.get("share"),
        "max_dominant_predicted_action_share": _round(
            max_dominant_predicted_action_share
        ),
        "dominant_prediction_floor_passed": dominance_floor_passed,
        "safe_hit_count": safe_hit_count,
        "safe_hit_rate": safe_hit_rate,
        "best_fixed_safe_action_baseline_hit_count": best_fixed_hit_count,
        "best_fixed_safe_action_baseline_hit_rate": best_fixed_rate,
        "first_public_action_baseline_hit_count": first_public_hit_count,
        "first_public_action_baseline_hit_rate": first_public_rate,
        "best_trivial_baseline_hit_rate": best_trivial_rate,
        "safe_hit_margin_over_best_trivial": margin,
        "min_safe_hit_margin_over_best_trivial": _round(
            min_safe_hit_margin_over_best_trivial
        ),
        "safe_hit_margin_floor_passed": safe_hit_margin_floor_passed,
        "v159_exact_top_value_upper_bound": {
            "non_runtime": True,
            "target_leaky": True,
            "safe_hit_count": exact_top_value_hit_count,
            "safe_hit_rate": exact_top_rate,
        },
        "floor_passed": (
            unsupported_floor_passed
            and dominance_floor_passed
            and safe_hit_margin_floor_passed
        ),
        "predictions": predictions,
    }


def _top_level_classification(
    *,
    source_validation: Mapping[str, object],
    row_contract: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    loo: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v160_action_value_scorer_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_shadow"
    if row_contract.get("passed") is not True:
        return prefix + "row_contract_invalid_closed_no_shadow"
    if leakage_scan.get("passed") is not True:
        return prefix + "leakage_failed_closed_no_shadow"
    if loo.get("unsupported_prediction_floor_passed") is not True:
        return prefix + "unsupported_predictions_closed_no_shadow"
    if loo.get("dominant_prediction_floor_passed") is not True:
        return prefix + "predicted_action_dominance_closed_no_shadow"
    if loo.get("safe_hit_margin_floor_passed") is not True:
        return prefix + "loo_generalization_failed_closed_archive_expansion"
    return prefix + "diagnostic_scorer_ready_for_future_shadow_eval"


def _route_recommendation(classification: str) -> dict[str, object]:
    shadow_ready = classification.endswith("diagnostic_scorer_ready_for_future_shadow_eval")
    return {
        "policy": "m3_carrion_survivor_continuation_v160_route_recommendation_v1",
        "future_shadow_evaluation_recommended": shadow_ready,
        "recommended_next_route": (
            "separate_opt_in_shadow_scorer_evaluation_without_runtime_override"
            if shadow_ready
            else "expand_carrion_survivor_continuation_archive_before_more_scorer_training"
        ),
        "threshold_tuning_recommended": False,
        "runtime_policy_integration_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "runtime_action_selection_changed": False,
    }


def _feature_vectors(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, float]], list[str], dict[str, dict[str, float]]]:
    raw_vectors = [
        _flatten_public_payload(
            {
                "trainable_public_features": _mapping(
                    row.get("trainable_public_features")
                ),
                "public_action_mask": _complete_bool_mask(row.get("public_action_mask")),
            }
        )
        for row in rows
    ]
    keys = sorted({key for vector in raw_vectors for key in vector})
    mins: dict[str, float] = {}
    ranges: dict[str, float] = {}
    for key in keys:
        values = [float(vector.get(key, 0.0)) for vector in raw_vectors]
        minimum = min(values, default=0.0)
        maximum = max(values, default=0.0)
        mins[key] = _round(minimum)
        ranges[key] = _round(max(maximum - minimum, 1.0))
    normalized = []
    for vector in raw_vectors:
        normalized.append(
            {
                key: _round((float(vector.get(key, 0.0)) - mins[key]) / ranges[key])
                for key in keys
            }
        )
    normalization = {
        key: {"min": mins[key], "range": ranges[key]}
        for key in keys
    }
    return normalized, keys, normalization


def _predict_1nn_action(
    *,
    rows: Sequence[Mapping[str, object]],
    vectors: Sequence[Mapping[str, float]],
    feature_keys: Sequence[str],
    normalization: Mapping[str, Mapping[str, float]],
    row_index: int,
    train_indexes: Sequence[int],
) -> tuple[str, str, list[int]]:
    del normalization
    if not train_indexes:
        return "", "no_training_rows", []
    neighbors = sorted(
        train_indexes,
        key=lambda index: (
            _vector_distance(vectors[row_index], vectors[index], feature_keys),
            index,
        ),
    )[:K_NEIGHBORS]
    public_mask = _complete_bool_mask(rows[row_index].get("public_action_mask"))
    candidates: list[dict[str, object]] = []
    for neighbor_index in neighbors:
        for target in _list_of_mappings(rows[neighbor_index].get("action_value_targets")):
            action = str(target.get("action", ""))
            if (
                action in ACTION_NAMES
                and public_mask.get(action) is True
                and target.get("target_available") is True
                and _finite_number(target.get("value_target"))
            ):
                candidates.append(dict(target))
    if not candidates:
        return "", "nearest_neighbor_no_public_value_candidate", neighbors
    ranked = sorted(
        candidates,
        key=lambda target: (
            -_float(target.get("value_target")),
            target.get("safe_target") is not True,
            target.get("robust_safe_action") is not True,
            _action_order(str(target.get("action", ""))),
        ),
    )
    return str(ranked[0].get("action", "")), "nearest_neighbor_value_target", neighbors


def _best_fixed_safe_action(
    rows: Sequence[Mapping[str, object]],
    train_indexes: Sequence[int],
    public_mask: Mapping[str, bool],
) -> str:
    counts: Counter[str] = Counter()
    for index in train_indexes:
        counts.update(_safe_action_set(rows[index]))
    candidates = [
        action
        for action in ACTION_NAMES
        if public_mask.get(action) is True
    ]
    if not candidates:
        return ""
    return sorted(candidates, key=lambda action: (-int(counts.get(action, 0)), _action_order(action)))[0]


def _artifact_action_value_targets(row: Mapping[str, object]) -> list[dict[str, object]]:
    targets = []
    for target in _list_of_mappings(row.get("action_value_targets")):
        action = str(target.get("action", ""))
        if action not in ACTION_NAMES:
            continue
        targets.append(
            {
                "action": action,
                "public_mask": target.get("public_mask") is True,
                "target_available": target.get("target_available") is True,
                "safe_target": target.get("safe_target") is True,
                "value_target": target.get("value_target")
                if _finite_number(target.get("value_target"))
                else None,
                "robust_safe_action": target.get("robust_safe_action") is True,
            }
        )
    return targets


def _flatten_public_payload(value: object, *, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_public_payload(value[key], prefix=child_prefix))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten_public_payload(item, prefix=child_prefix))
    elif isinstance(value, bool):
        flattened[prefix] = 1.0 if value else 0.0
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            flattened[prefix] = _round(float(value))
    return flattened


def _vector_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
    feature_keys: Sequence[str],
) -> float:
    total = 0.0
    for key in feature_keys:
        delta = float(left.get(key, 0.0)) - float(right.get(key, 0.0))
        total += delta * delta
    return math.sqrt(total)


def _sparse_vector(vector: Mapping[str, float]) -> dict[str, float]:
    return {
        key: _round(value)
        for key, value in sorted(vector.items())
        if abs(float(value)) > 0.0
    }


def _positive_counter(counter: Counter[str]) -> Counter[str]:
    return Counter(
        {
            key: int(value)
            for key, value in counter.items()
            if int(value) > 0 and str(key)
        }
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _safe_rate(numerator: float | int, denominator: float | int) -> float:
    denominator_float = float(denominator)
    if denominator_float <= 0.0:
        return 0.0
    return _round(float(numerator) / denominator_float)


def _v160_source_lifecycle_scan(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    for field in (
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
        "runtime_action_selection_changed",
    ):
        if report.get(field) is not False:
            failures.append(
                {
                    "field": field,
                    "observed": report.get(field),
                    "expected": False,
                }
            )
    contract = _mapping(report.get("contract"))
    for field in (
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
        "runtime_action_selection_changed",
        "shadow_live_ab_allowed",
        "live_runtime_override_allowed",
    ):
        if field in contract and contract.get(field) is not False:
            failures.append(
                {
                    "field": f"contract.{field}",
                    "observed": contract.get(field),
                    "expected": False,
                }
            )
    return {
        "policy": "m3_carrion_survivor_continuation_v160_source_lifecycle_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }
