from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
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
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
    load_v154_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    _complete_bool_mask,
    _safe_action_set,
    _safe_rate,
    _top_value_target,
)
from evolution_sim.mind.carrion_survivor_continuation_v160_action_value_scorer import (
    _feature_vectors,
    _vector_distance,
)
from evolution_sim.mind.carrion_survivor_continuation_v165_preterminal_target_dataset_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v166_source_split_action_value_scorer import (
    DEFAULT_ARTIFACT_OUTPUT_PATH as DEFAULT_V166_ARTIFACT_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V166_REPORT_PATH,
    DEFAULT_V165_DATASET_PATH,
    EXPECTED_V165_DATASET_DIGEST,
    M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION,
    source_split_diagnostics,
    source_split_groups,
    trainable_feature_payloads,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_source_split_failure_autopsy_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_source_split_failure_autopsy_v1"
)
EXPECTED_V166_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v166_source_split_action_value_scorer_"
    "source_split_generalization_failed_closed_archive_source_expansion"
)
V166_SHADOW_READY_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v166_source_split_action_value_scorer_"
    "diagnostic_scorer_ready_for_future_diagnostics_only_shadow_eval"
)
EXPECTED_V166_EXACT_DIGEST = (
    "afbbd681be7279f49b477c55bf85c73f7bff082e9c81f90059c3e0497e70a6fa"
)
EXPECTED_V166_ARTIFACT_DIGEST = (
    "e1b5f8cb9b7ca8e916fe6ff0e162e7d451d9c5c25affb1a2eaf000eafc8a0e7d"
)
EXPECTED_ZERO_SAFE_HIT_PRETERMINAL_SOURCE_SEEDS = (13, 19, 29, 41)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v167-carrion-survivor-continuation-source-split-failure-autopsy.json"
)
TOP_K_VALUES = (1, 3, 5)


class CarrionSurvivorContinuationV167SourceSplitFailureAutopsyError(ValueError):
    pass


def run_carrion_survivor_continuation_v167_source_split_failure_autopsy(
    *,
    v166_report_path: str | Path = DEFAULT_V166_REPORT_PATH,
    v166_artifact_path: str | Path = DEFAULT_V166_ARTIFACT_PATH,
    v165_dataset_path: str | Path = DEFAULT_V165_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v166_exact_digest: str | None = EXPECTED_V166_EXACT_DIGEST,
    expected_v166_artifact_digest: str | None = EXPECTED_V166_ARTIFACT_DIGEST,
    expected_v165_dataset_digest: str | None = EXPECTED_V165_DATASET_DIGEST,
    expected_zero_safe_hit_preterminal_source_seeds: Sequence[int] = (
        EXPECTED_ZERO_SAFE_HIT_PRETERMINAL_SOURCE_SEEDS
    ),
) -> dict[str, object]:
    v166_report = load_json_report(v166_report_path)
    v166_artifact = load_json_report(v166_artifact_path)
    rows = load_v154_dataset(v165_dataset_path)
    source_validation = validate_v167_sources(
        v166_report=v166_report,
        v166_artifact=v166_artifact,
        rows=rows,
        expected_v166_exact_digest=expected_v166_exact_digest,
        expected_v166_artifact_digest=expected_v166_artifact_digest,
        expected_v165_dataset_digest=expected_v165_dataset_digest,
        expected_zero_safe_hit_preterminal_source_seeds=(
            expected_zero_safe_hit_preterminal_source_seeds
        ),
    )
    recomputed = source_split_diagnostics(rows=rows)
    recompute_validation = validate_recomputed_v166_source_split(
        v166_report=v166_report,
        recomputed_source_split=recomputed,
    )
    preterminal = preterminal_failure_autopsy(rows=rows, recomputed_source_split=recomputed)
    failure_summary = summarize_failure_modes(preterminal)
    classification = _classification(
        source_validation=source_validation,
        recompute_validation=recompute_validation,
        observed_v166_classification=str(
            _mapping(v166_report.get("classification")).get("primary") or ""
        ),
        failure_summary=failure_summary,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_POLICY,
        "contract": {
            "diagnostics_only": True,
            "autopsy_only": True,
            "retraining_ran": False,
            "training_ran": False,
            "k_tuning_ran": False,
            "threshold_tuning_ran": False,
            "shadow_eval_ran": False,
            "artifact_created": False,
            "runtime_artifact_created": False,
            "runtime_policy_integration_allowed": False,
            "runtime_action_selection_changed": False,
            "live_ab_allowed": False,
            "promotion_authorized": False,
            "replay_viewer_schema_changed": False,
            "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_future_outcome_as_trainable_input": False,
        },
        "inputs": {
            "v166_report": str(v166_report_path),
            "v166_artifact": str(v166_artifact_path),
            "v165_dataset": str(v165_dataset_path),
            "expected_v166_exact_digest": expected_v166_exact_digest,
            "expected_v166_artifact_digest": expected_v166_artifact_digest,
            "expected_v165_dataset_digest": expected_v165_dataset_digest,
            "expected_zero_safe_hit_preterminal_source_seeds": [
                int(seed) for seed in expected_zero_safe_hit_preterminal_source_seeds
            ],
            "top_k_values": list(TOP_K_VALUES),
        },
        "source_validation": source_validation,
        "recomputed_v166_source_split_validation": recompute_validation,
        "v166_source_split_metrics": _v166_metric_summary(recomputed),
        "preterminal_row_autopsy": preterminal["rows"],
        "failing_source_seed_summaries": preterminal[
            "failing_source_seed_summaries"
        ],
        "confusion_summaries": preterminal["confusion_summaries"],
        "top_k_diagnostics": preterminal["top_k_diagnostics"],
        "failure_mode_summary": failure_summary,
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(
            classification=classification,
            failure_summary=failure_summary,
        ),
        "dataset_digest": stable_payload_digest(rows),
        "trainable_feature_payload_digest": stable_payload_digest(
            trainable_feature_payloads(rows)
        ),
        "artifact_created": False,
        "runtime_artifact_created": False,
        "training_ran": False,
        "retraining_ran": False,
        "k_tuning_ran": False,
        "threshold_tuning_ran": False,
        "shadow_eval_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "runtime_action_selection_changed": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v167_sources(
    *,
    v166_report: Mapping[str, object],
    v166_artifact: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v166_exact_digest: str | None = EXPECTED_V166_EXACT_DIGEST,
    expected_v166_artifact_digest: str | None = EXPECTED_V166_ARTIFACT_DIGEST,
    expected_v165_dataset_digest: str | None = EXPECTED_V165_DATASET_DIGEST,
    expected_zero_safe_hit_preterminal_source_seeds: Sequence[int] = (
        EXPECTED_ZERO_SAFE_HIT_PRETERMINAL_SOURCE_SEEDS
    ),
) -> dict[str, object]:
    failures: list[str] = []
    if (
        v166_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION
    ):
        failures.append("v166_schema_version_mismatch")
    if v166_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_POLICY:
        failures.append("v166_policy_mismatch")
    observed_classification = str(
        _mapping(v166_report.get("classification")).get("primary") or ""
    )
    if observed_classification not in {
        EXPECTED_V166_CLASSIFICATION,
        V166_SHADOW_READY_CLASSIFICATION,
    }:
        failures.append("v166_unexpected_classification")
    exact_digest_validation = exact_digest_validation_report(v166_report)
    if exact_digest_validation.get("passed") is not True:
        failures.append("v166_exact_digest_mismatch")
    observed_exact_digest = str(v166_report.get("exact_digest") or "")
    if expected_v166_exact_digest and observed_exact_digest != expected_v166_exact_digest:
        failures.append("v166_unexpected_exact_digest")
    artifact_validation = validate_v166_artifact(
        v166_report=v166_report,
        v166_artifact=v166_artifact,
        expected_v166_artifact_digest=expected_v166_artifact_digest,
    )
    if artifact_validation.get("passed") is not True:
        failures.append("v166_artifact_validation_failed")
    dataset_digest = stable_payload_digest(rows)
    if expected_v165_dataset_digest and dataset_digest != expected_v165_dataset_digest:
        failures.append("v165_dataset_digest_mismatch")
    if v166_report.get("dataset_digest") != dataset_digest:
        failures.append("v166_report_dataset_digest_mismatch")
    source_split = _mapping(v166_report.get("source_split_diagnostics"))
    observed_zero_seeds = _zero_safe_hit_preterminal_source_seeds(source_split)
    expected_zero_seeds = sorted(
        {int(seed) for seed in expected_zero_safe_hit_preterminal_source_seeds}
    )
    if observed_classification == EXPECTED_V166_CLASSIFICATION:
        if observed_zero_seeds != expected_zero_seeds:
            failures.append("v166_zero_safe_hit_source_seed_set_mismatch")
        if source_split.get("safe_hit_margin_floor_passed") is not False:
            failures.append("v166_safe_hit_margin_floor_not_failed")
    lifecycle = _v166_lifecycle_authorization_scan(v166_report)
    if lifecycle.get("passed") is not True:
        failures.append("v166_lifecycle_authorization_not_false")
    return {
        "policy": "m3_carrion_survivor_continuation_v167_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v166_classification": EXPECTED_V166_CLASSIFICATION,
        "v166_shadow_ready_classification": V166_SHADOW_READY_CLASSIFICATION,
        "observed_v166_classification": observed_classification,
        "expected_v166_exact_digest": expected_v166_exact_digest,
        "observed_v166_exact_digest": observed_exact_digest,
        "v166_exact_digest_validation": exact_digest_validation,
        "artifact_validation": artifact_validation,
        "expected_v165_dataset_digest": expected_v165_dataset_digest,
        "observed_v165_dataset_digest": dataset_digest,
        "v166_reported_dataset_digest": v166_report.get("dataset_digest"),
        "expected_zero_safe_hit_preterminal_source_seeds": expected_zero_seeds,
        "observed_zero_safe_hit_preterminal_source_seeds": observed_zero_seeds,
        "v166_lifecycle_authorization": lifecycle,
        "row_count": len(rows),
    }


def validate_v166_artifact(
    *,
    v166_report: Mapping[str, object],
    v166_artifact: Mapping[str, object],
    expected_v166_artifact_digest: str | None,
) -> dict[str, object]:
    failures: list[str] = []
    if (
        v166_artifact.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION
    ):
        failures.append("v166_artifact_schema_version_mismatch")
    if v166_artifact.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_POLICY:
        failures.append("v166_artifact_policy_mismatch")
    exact_validation = exact_digest_validation_report(v166_artifact)
    if exact_validation.get("passed") is not True:
        failures.append("v166_artifact_exact_digest_mismatch")
    observed_artifact_digest = str(v166_artifact.get("exact_digest") or "")
    reported_artifact_digest = str(
        _mapping(v166_report.get("artifact")).get("artifact_digest") or ""
    )
    if expected_v166_artifact_digest and observed_artifact_digest != expected_v166_artifact_digest:
        failures.append("v166_unexpected_artifact_digest")
    if reported_artifact_digest != observed_artifact_digest:
        failures.append("v166_report_artifact_digest_mismatch")
    contract = _mapping(v166_artifact.get("contract"))
    if contract.get("runtime_artifact") is not False:
        failures.append("v166_artifact_runtime_artifact_not_false")
    return {
        "policy": "m3_carrion_survivor_continuation_v167_artifact_validation_v1",
        "passed": not failures,
        "failures": failures,
        "expected_v166_artifact_digest": expected_v166_artifact_digest,
        "observed_v166_artifact_digest": observed_artifact_digest,
        "reported_v166_artifact_digest": reported_artifact_digest,
        "v166_artifact_exact_digest_validation": exact_validation,
    }


def validate_recomputed_v166_source_split(
    *,
    v166_report: Mapping[str, object],
    recomputed_source_split: Mapping[str, object],
) -> dict[str, object]:
    reported = _mapping(v166_report.get("source_split_diagnostics"))
    reported_payload = _source_split_comparison_payload(reported)
    recomputed_payload = _source_split_comparison_payload(recomputed_source_split)
    reported_digest = stable_payload_digest(reported_payload)
    recomputed_digest = stable_payload_digest(recomputed_payload)
    return {
        "policy": "m3_carrion_survivor_continuation_v167_recomputed_v166_source_split_validation_v1",
        "passed": reported_digest == recomputed_digest,
        "reported_comparison_digest": reported_digest,
        "recomputed_comparison_digest": recomputed_digest,
        "reported_prediction_count": reported.get("prediction_count"),
        "recomputed_prediction_count": recomputed_source_split.get("prediction_count"),
        "reported_safe_hit_margin_over_best_trivial": reported.get(
            "safe_hit_margin_over_best_trivial"
        ),
        "recomputed_safe_hit_margin_over_best_trivial": recomputed_source_split.get(
            "safe_hit_margin_over_best_trivial"
        ),
        "reported_zero_safe_hit_preterminal_source_seeds": (
            _zero_safe_hit_preterminal_source_seeds(reported)
        ),
        "recomputed_zero_safe_hit_preterminal_source_seeds": (
            _zero_safe_hit_preterminal_source_seeds(recomputed_source_split)
        ),
    }


def preterminal_failure_autopsy(
    *,
    rows: Sequence[Mapping[str, object]],
    recomputed_source_split: Mapping[str, object],
) -> dict[str, object]:
    vectors, feature_keys, _normalization = _feature_vectors(rows)
    groups = source_split_groups(rows)
    group_by_row = {int(group["row_index"]): group for group in groups}
    group_row_indexes: dict[str, list[int]] = defaultdict(list)
    for group in groups:
        group_row_indexes[str(group["group_key"])].append(int(group["row_index"]))
    prediction_by_row = {
        _int(prediction.get("row_index")): prediction
        for prediction in _list_of_mappings(recomputed_source_split.get("predictions"))
    }

    row_autopsies: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        if (
            row.get("schema_version")
            != M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ):
            continue
        group = group_by_row[row_index]
        train_indexes = [
            index
            for index in range(len(rows))
            if str(group_by_row[index]["group_key"]) != str(group["group_key"])
        ]
        ranked = _ranked_train_neighbors(
            row_index=row_index,
            train_indexes=train_indexes,
            vectors=vectors,
            feature_keys=feature_keys,
        )
        prediction = prediction_by_row.get(row_index, {})
        predicted_neighbor_index = _first_index(
            prediction.get("nearest_neighbor_row_indexes")
        )
        predicted_distance = _distance_for_neighbor(ranked, predicted_neighbor_index)
        safe_support = _nearest_safe_support_neighbor(
            row=row,
            ranked=ranked,
            rows=rows,
        )
        safe_set = _safe_action_set(row)
        predicted_action = str(prediction.get("predicted_action") or "")
        safe_hit = prediction.get("safe_hit") is True
        mode = _row_failure_mode(
            safe_hit=safe_hit,
            safe_support_available=safe_support.get("row_index") is not None,
        )
        top_k = _top_k_presence(
            ranked=ranked,
            row=row,
            rows=rows,
        )
        row_autopsies.append(
            {
                "row_index": row_index,
                "source_seed": group.get("source_seed"),
                "source_group": group.get("group_key"),
                "safe_action_set": safe_set,
                "robust_winner_action": row.get("robust_winner_action"),
                "target_leaky_exact_top_action": _top_value_target(row).get("action"),
                "predicted_action": predicted_action,
                "safe_hit": safe_hit,
                "failure_mode": mode,
                "nearest_neighbor_row_index": predicted_neighbor_index,
                "nearest_neighbor_source_seed": _source_seed_for_index(
                    group_by_row,
                    predicted_neighbor_index,
                ),
                "nearest_neighbor_source_group": _source_group_for_index(
                    group_by_row,
                    predicted_neighbor_index,
                ),
                "nearest_neighbor_safe_action_set": _safe_action_set(
                    rows[predicted_neighbor_index]
                )
                if predicted_neighbor_index is not None
                else [],
                "distance_to_predicted_neighbor": predicted_distance,
                "nearest_safe_support_neighbor_row_index": safe_support.get("row_index"),
                "nearest_safe_support_source_seed": safe_support.get("source_seed"),
                "nearest_safe_support_source_group": safe_support.get("source_group"),
                "nearest_safe_support_action_overlap": safe_support.get(
                    "safe_action_overlap"
                ),
                "distance_to_nearest_safe_support_neighbor": safe_support.get(
                    "distance"
                ),
                "rank_of_first_safe_support_neighbor": safe_support.get("rank"),
                "distance_gap_predicted_to_nearest_safe_support": _distance_gap(
                    predicted_distance,
                    safe_support.get("distance"),
                ),
                "safe_action_support_available_in_training": _safe_action_support(
                    row=row,
                    train_indexes=train_indexes,
                    rows=rows,
                    group_by_row=group_by_row,
                ),
                "top_k_safe_support": top_k,
            }
        )
    failing_summaries = _failing_source_seed_summaries(row_autopsies)
    return {
        "rows": row_autopsies,
        "failing_source_seed_summaries": failing_summaries,
        "confusion_summaries": _confusion_summaries(row_autopsies),
        "top_k_diagnostics": _top_k_diagnostics(row_autopsies),
    }


def summarize_failure_modes(
    preterminal: Mapping[str, object],
) -> dict[str, object]:
    rows = _list_of_mappings(preterminal.get("rows"))
    failed_rows = [row for row in rows if row.get("safe_hit") is not True]
    mode_counts = Counter(str(row.get("failure_mode") or "") for row in failed_rows)
    support_missing = int(mode_counts.get("action_support_missing", 0))
    aliasing = int(mode_counts.get("feature_neighbor_aliasing", 0))
    if support_missing and aliasing:
        primary = "mixed_source_support_and_feature_aliasing"
    elif support_missing:
        primary = "source_action_support_missing"
    elif aliasing:
        primary = "nearest_neighbor_feature_aliasing"
    else:
        primary = "no_failed_preterminal_rows"
    return {
        "policy": "m3_carrion_survivor_continuation_v167_failure_mode_summary_v1",
        "primary_failure_mode": primary,
        "preterminal_row_count": len(rows),
        "failed_preterminal_row_count": len(failed_rows),
        "safe_hit_preterminal_row_count": len(rows) - len(failed_rows),
        "failure_mode_counts": dict(sorted(mode_counts.items())),
        "source_action_support_missing_row_count": support_missing,
        "feature_neighbor_aliasing_row_count": aliasing,
        "zero_safe_hit_source_seeds": [
            _int(summary.get("source_seed"))
            for summary in _list_of_mappings(
                preterminal.get("failing_source_seed_summaries")
            )
            if _int(summary.get("safe_hit_count")) == 0
        ],
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    recompute_validation: Mapping[str, object],
    observed_v166_classification: str,
    failure_summary: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v167_source_split_failure_autopsy_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_shadow"
    if recompute_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_shadow"
    if observed_v166_classification == V166_SHADOW_READY_CLASSIFICATION:
        return prefix + "unexpected_shadow_ready_input_closed_no_autopsy"
    mode = str(failure_summary.get("primary_failure_mode") or "")
    if mode == "source_action_support_missing":
        return prefix + "source_action_support_missing_closed_source_expansion"
    if mode == "mixed_source_support_and_feature_aliasing":
        return prefix + "mixed_source_support_and_feature_aliasing_closed_source_expansion"
    if mode == "nearest_neighbor_feature_aliasing":
        return prefix + "nearest_neighbor_feature_aliasing_closed_feature_contract_expansion"
    return prefix + "inconclusive_closed_feature_contract_expansion"


def _route_recommendation(
    *,
    classification: str,
    failure_summary: Mapping[str, object],
) -> dict[str, object]:
    del classification
    source_route = str(failure_summary.get("primary_failure_mode") or "") in {
        "source_action_support_missing",
        "mixed_source_support_and_feature_aliasing",
    }
    recommended = (
        "targeted_source_support_expansion"
        if source_route
        else "public_feature_contract_expansion"
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v167_route_recommendation_v1",
        "recommended_next_route": recommended,
        "targeted_source_support_expansion_recommended": source_route,
        "public_feature_contract_expansion_recommended": not source_route,
        "retraining_recommended": False,
        "k_tuning_recommended": False,
        "threshold_tuning_recommended": False,
        "shadow_eval_recommended": False,
        "live_ab_allowed": False,
        "runtime_policy_integration_allowed": False,
        "promotion_authorized": False,
        "runtime_action_selection_changed": False,
    }


def _source_split_comparison_payload(source_split: Mapping[str, object]) -> dict[str, object]:
    predictions = []
    for prediction in _list_of_mappings(source_split.get("predictions")):
        predictions.append(
            {
                "row_index": prediction.get("row_index"),
                "source_group_key": prediction.get("source_group_key"),
                "predicted_action": prediction.get("predicted_action"),
                "nearest_neighbor_row_indexes": prediction.get(
                    "nearest_neighbor_row_indexes"
                ),
                "safe_hit": prediction.get("safe_hit"),
                "unsupported_prediction": prediction.get("unsupported_prediction"),
            }
        )
    return {
        "prediction_count": source_split.get("prediction_count"),
        "no_prediction_count": source_split.get("no_prediction_count"),
        "unsupported_prediction_count": source_split.get("unsupported_prediction_count"),
        "predicted_action_counts": source_split.get("predicted_action_counts"),
        "safe_hit_count": source_split.get("safe_hit_count"),
        "safe_hit_rate": source_split.get("safe_hit_rate"),
        "best_trivial_baseline_hit_rate": source_split.get(
            "best_trivial_baseline_hit_rate"
        ),
        "safe_hit_margin_over_best_trivial": source_split.get(
            "safe_hit_margin_over_best_trivial"
        ),
        "per_source_seed_safe_hit_rates": source_split.get(
            "per_source_seed_safe_hit_rates"
        ),
        "predictions": predictions,
    }


def _v166_metric_summary(source_split: Mapping[str, object]) -> dict[str, object]:
    return {
        "prediction_count": source_split.get("prediction_count"),
        "no_prediction_count": source_split.get("no_prediction_count"),
        "unsupported_prediction_count": source_split.get("unsupported_prediction_count"),
        "predicted_action_counts": source_split.get("predicted_action_counts"),
        "dominant_predicted_action_share": source_split.get(
            "dominant_predicted_action_share"
        ),
        "safe_hit_count": source_split.get("safe_hit_count"),
        "safe_hit_rate": source_split.get("safe_hit_rate"),
        "best_trivial_baseline_hit_rate": source_split.get(
            "best_trivial_baseline_hit_rate"
        ),
        "safe_hit_margin_over_best_trivial": source_split.get(
            "safe_hit_margin_over_best_trivial"
        ),
        "per_source_seed_safe_hit_rates": source_split.get(
            "per_source_seed_safe_hit_rates"
        ),
        "target_leaky_exact_top_upper_bound": source_split.get(
            "target_leaky_exact_top_upper_bound"
        ),
    }


def _ranked_train_neighbors(
    *,
    row_index: int,
    train_indexes: Sequence[int],
    vectors: Sequence[Mapping[str, float]],
    feature_keys: Sequence[str],
) -> list[dict[str, object]]:
    return [
        {"row_index": index, "distance": _round(distance)}
        for distance, index in sorted(
            (
                (
                    _vector_distance(vectors[row_index], vectors[index], feature_keys),
                    index,
                )
                for index in train_indexes
            ),
            key=lambda item: (item[0], item[1]),
        )
    ]


def _nearest_safe_support_neighbor(
    *,
    row: Mapping[str, object],
    ranked: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    safe_set = set(_safe_action_set(row))
    public_mask = _complete_bool_mask(row.get("public_action_mask"))
    groups = source_split_groups(rows)
    group_by_row = {int(group["row_index"]): group for group in groups}
    for rank, neighbor in enumerate(ranked, start=1):
        index = _int(neighbor.get("row_index"))
        overlap = sorted(
            [
                action
                for action in safe_set
                if public_mask.get(action) is True
                and action in _safe_action_set(rows[index])
            ],
            key=_action_order,
        )
        if not overlap:
            continue
        group = group_by_row[index]
        return {
            "row_index": index,
            "source_seed": group.get("source_seed"),
            "source_group": group.get("group_key"),
            "safe_action_overlap": overlap,
            "distance": neighbor.get("distance"),
            "rank": rank,
        }
    return {
        "row_index": None,
        "source_seed": None,
        "source_group": None,
        "safe_action_overlap": [],
        "distance": None,
        "rank": None,
    }


def _safe_action_support(
    *,
    row: Mapping[str, object],
    train_indexes: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    group_by_row: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    support: dict[str, object] = {}
    for action in _safe_action_set(row):
        indexes = [
            index for index in train_indexes if action in _safe_action_set(rows[index])
        ]
        support[action] = {
            "training_row_count": len(indexes),
            "source_groups": sorted(
                {str(group_by_row[index].get("group_key")) for index in indexes}
            ),
            "source_seeds": sorted(
                {
                    _int(group_by_row[index].get("source_seed"))
                    for index in indexes
                    if group_by_row[index].get("source_seed") is not None
                }
            ),
            "row_indexes": indexes,
        }
    return support


def _top_k_presence(
    *,
    ranked: Sequence[Mapping[str, object]],
    row: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    safe_set = set(_safe_action_set(row))
    public_mask = _complete_bool_mask(row.get("public_action_mask"))
    for k in TOP_K_VALUES:
        top = ranked[:k]
        support_indexes = []
        for neighbor in top:
            index = _int(neighbor.get("row_index"))
            if any(
                public_mask.get(action) is True
                and action in _safe_action_set(rows[index])
                for action in safe_set
            ):
                support_indexes.append(index)
        payload[str(k)] = {
            "safe_support_neighbor_in_top_k": bool(support_indexes),
            "safe_support_neighbor_row_indexes": support_indexes,
            "analysis_only_no_k_tuning_recommended": True,
        }
    return payload


def _failing_source_seed_summaries(
    row_autopsies: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_seed: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in row_autopsies:
        by_seed[_int(row.get("source_seed"))].append(row)
    summaries = []
    for seed in sorted(by_seed):
        rows = by_seed[seed]
        safe_hits = sum(1 for row in rows if row.get("safe_hit") is True)
        if safe_hits > 0:
            continue
        predicted = Counter(str(row.get("predicted_action") or "") for row in rows)
        modes = Counter(str(row.get("failure_mode") or "") for row in rows)
        summaries.append(
            {
                "source_seed": seed,
                "row_count": len(rows),
                "safe_hit_count": safe_hits,
                "safe_hit_rate": _safe_rate(safe_hits, len(rows)),
                "predicted_action_counts": dict(sorted(predicted.items())),
                "safe_action_support_available_in_training": (
                    _merge_safe_action_support(rows)
                ),
                "failure_mode_counts": dict(sorted(modes.items())),
                "failure_type": _seed_failure_type(modes),
            }
        )
    return summaries


def _merge_safe_action_support(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    merged: dict[str, dict[str, object]] = {}
    for row in rows:
        for action, support in _mapping(
            row.get("safe_action_support_available_in_training")
        ).items():
            payload = _mapping(support)
            target = merged.setdefault(
                action,
                {
                    "max_training_row_count": 0,
                    "source_groups": set(),
                    "source_seeds": set(),
                    "row_indexes": set(),
                },
            )
            target["max_training_row_count"] = max(
                _int(target["max_training_row_count"]),
                _int(payload.get("training_row_count")),
            )
            target["source_groups"].update(str(group) for group in _list_like(payload.get("source_groups")))
            target["source_seeds"].update(_int(seed) for seed in _list_like(payload.get("source_seeds")))
            target["row_indexes"].update(_int(index) for index in _list_like(payload.get("row_indexes")))
    return {
        action: {
            "max_training_row_count": _int(payload["max_training_row_count"]),
            "source_groups": sorted(payload["source_groups"]),
            "source_seeds": sorted(payload["source_seeds"]),
            "row_indexes": sorted(payload["row_indexes"]),
        }
        for action, payload in sorted(merged.items())
    }


def _seed_failure_type(modes: Counter[str]) -> str:
    support_missing = int(modes.get("action_support_missing", 0)) > 0
    aliasing = int(modes.get("feature_neighbor_aliasing", 0)) > 0
    if support_missing and aliasing:
        return "mixed_source_support_and_feature_aliasing"
    if support_missing:
        return "action_support_missing"
    if aliasing:
        return "feature_neighbor_aliasing"
    return "no_failure_rows"


def _confusion_summaries(
    row_autopsies: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    predicted_vs_safe = Counter()
    failing_seed_to_neighbor_seed = Counter()
    robust_to_predicted = Counter()
    exact_top_to_predicted = Counter()
    for row in row_autopsies:
        predicted = str(row.get("predicted_action") or "")
        safe_key = "|".join(str(action) for action in _list_like(row.get("safe_action_set")))
        predicted_vs_safe.update([f"{predicted}->{safe_key}"])
        robust_to_predicted.update([f"{row.get('robust_winner_action')}->{predicted}"])
        exact_top_to_predicted.update(
            [f"{row.get('target_leaky_exact_top_action')}->{predicted}"]
        )
        if row.get("safe_hit") is not True:
            failing_seed_to_neighbor_seed.update(
                [
                    (
                        f"{row.get('source_seed')}->"
                        f"{row.get('nearest_neighbor_source_seed')}"
                    )
                ]
            )
    return {
        "policy": "m3_carrion_survivor_continuation_v167_confusion_summaries_v1",
        "predicted_action_vs_safe_action_set": dict(sorted(predicted_vs_safe.items())),
        "failing_seed_to_nearest_neighbor_seed": dict(
            sorted(failing_seed_to_neighbor_seed.items())
        ),
        "robust_winner_action_to_predicted_action": dict(
            sorted(robust_to_predicted.items())
        ),
        "target_leaky_exact_top_action_to_predicted_action": dict(
            sorted(exact_top_to_predicted.items())
        ),
    }


def _top_k_diagnostics(
    row_autopsies: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = list(row_autopsies)
    by_k: dict[str, object] = {}
    for k in TOP_K_VALUES:
        key = str(k)
        present = [
            row
            for row in rows
            if _mapping(_mapping(row.get("top_k_safe_support")).get(key)).get(
                "safe_support_neighbor_in_top_k"
            )
            is True
        ]
        failing_present = [
            row for row in present if row.get("safe_hit") is not True
        ]
        by_k[key] = {
            "preterminal_row_count": len(rows),
            "rows_with_safe_support_neighbor_in_top_k": len(present),
            "share_with_safe_support_neighbor_in_top_k": _safe_rate(
                len(present),
                len(rows),
            ),
            "failed_rows_with_safe_support_neighbor_in_top_k": len(failing_present),
            "analysis_only_no_k_tuning_recommended": True,
        }
    return {
        "policy": "m3_carrion_survivor_continuation_v167_top_k_diagnostics_v1",
        "k_values": list(TOP_K_VALUES),
        "top_k": by_k,
        "analysis_only": True,
        "k_tuning_recommended": False,
    }


def _row_failure_mode(*, safe_hit: bool, safe_support_available: bool) -> str:
    if safe_hit:
        return "safe_hit"
    if not safe_support_available:
        return "action_support_missing"
    return "feature_neighbor_aliasing"


def _zero_safe_hit_preterminal_source_seeds(
    source_split: Mapping[str, object],
) -> list[int]:
    rates = _mapping(source_split.get("per_source_seed_safe_hit_rates"))
    zero = []
    for seed, payload in rates.items():
        data = _mapping(payload)
        if data.get("present") is True and _int(data.get("safe_hit_count")) == 0:
            zero.append(_int(seed))
    return sorted(zero)


def _v166_lifecycle_authorization_scan(
    report: Mapping[str, object],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for source, payload in (
        ("v166", report),
        ("v166.contract", _mapping(report.get("contract"))),
        ("v166.route_recommendation", _mapping(report.get("route_recommendation"))),
    ):
        for field in (
            "runtime_artifact_created",
            "runtime_policy_integration_allowed",
            "runtime_override_path_allowed",
            "live_ab_allowed",
            "shadow_live_ab_allowed",
            "live_runtime_override_allowed",
            "promotion_authorized",
            "runtime_promotion_allowed",
            "threshold_tuning_recommended",
            "runtime_action_selection_changed",
        ):
            if field in payload and payload.get(field) is not False:
                failures.append(
                    {
                        "source": source,
                        "field": field,
                        "observed": payload.get(field),
                        "expected": False,
                    }
                )
    return {
        "policy": "m3_carrion_survivor_continuation_v167_v166_lifecycle_authorization_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def _source_seed_for_index(
    group_by_row: Mapping[int, Mapping[str, object]],
    row_index: int | None,
) -> object:
    if row_index is None:
        return None
    return group_by_row[row_index].get("source_seed")


def _source_group_for_index(
    group_by_row: Mapping[int, Mapping[str, object]],
    row_index: int | None,
) -> object:
    if row_index is None:
        return None
    return group_by_row[row_index].get("group_key")


def _distance_for_neighbor(
    ranked: Sequence[Mapping[str, object]],
    row_index: int | None,
) -> object:
    if row_index is None:
        return None
    for neighbor in ranked:
        if _int(neighbor.get("row_index")) == row_index:
            return neighbor.get("distance")
    return None


def _distance_gap(left: object, right: object) -> object:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return _round(float(right) - float(left))


def _first_index(value: object) -> int | None:
    if not isinstance(value, list) or not value:
        return None
    return _int(value[0])


def _list_like(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _json_round_trip_digest(payload: Mapping[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    json_payload = json.loads(
        json.dumps(without_digest, sort_keys=True, allow_nan=False)
    )
    return stable_payload_digest(json_payload)
