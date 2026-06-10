from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import json
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
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    load_jsonl_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V172_REPORT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V172_DATASET_PATH,
    SUPPORT_PROVENANCE_SEEDS,
    trainable_payload_leakage_scan,
    validate_v172_target_rows,
)
from evolution_sim.mind.carrion_survivor_continuation_v173_source_split_scorer import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V173_REPORT_PATH,
    EXPECTED_V172_DATASET_DIGEST,
    M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_SCHEMA_VERSION,
    _complete_action_mask,
    _decode_zlib_int16,
    _even_samples,
    _json_round_trip_digest,
    _mask_signature,
    _normalized_feature_vectors,
    _ordered_action_set,
    _public_feature_vector,
    _public_observation_values,
    _rate,
    _safe_action_set,
    _source_seed,
    _squared_distance,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v174_mechanism_failure_battery_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v174_mechanism_failure_battery_v1"
)
EXPECTED_V173_EXACT_DIGEST = (
    "571534801bff339863214dccbfa24b3d070ca3a4bf004a57b69ddad0686f6f07"
)
EXPECTED_V173_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v173_source_split_scorer_"
    "source_split_set_valued_partial_no_shadow"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v174-carrion-survivor-continuation-mechanism-failure-battery.json"
)
DEFAULT_V175_PLAN_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v175-carrion-survivor-continuation-route-plan.jsonl"
)
K_NEAREST_VALUES = (1, 3, 5, 10, 25, 50)
RADIUS_VALUES = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0)
RANKING_CONFIGS = (
    ("all_train", None),
    ("k_nearest_50", 50),
    ("k_nearest_25", 25),
    ("k_nearest_10", 10),
    ("k_nearest_5", 5),
)
LOCAL_MIN_MARGIN_OVER_TRIVIAL = 0.05
RANKING_MIN_PAIRWISE_ACCURACY = 0.53
RANKING_MIN_MARGIN_OVER_SHUFFLED = 0.05
RANKING_MIN_TOP3_SAFE_HIT_RATE = 0.65
PUBLIC_CONTEXT_MIN_IMPROVEMENT = 0.03
CONFORMAL_TARGET_COVERAGE = 0.9
CONFORMAL_MAX_FULL_SET_SHARE = 0.5


class CarrionSurvivorContinuationV174MechanismFailureBatteryError(ValueError):
    pass


FeatureVectorBuilder = Callable[[Mapping[str, object]], list[float]]


def run_carrion_survivor_continuation_v174_mechanism_failure_battery(
    *,
    v173_report_path: str | Path = DEFAULT_V173_REPORT_PATH,
    v172_report_path: str | Path = DEFAULT_V172_REPORT_PATH,
    v172_dataset_path: str | Path = DEFAULT_V172_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    v175_plan_output_path: str | Path | None = DEFAULT_V175_PLAN_OUTPUT_PATH,
    expected_v173_exact_digest: str | None = EXPECTED_V173_EXACT_DIGEST,
    expected_v173_classification: str = EXPECTED_V173_CLASSIFICATION,
    expected_v172_dataset_digest: str | None = EXPECTED_V172_DATASET_DIGEST,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> dict[str, object]:
    v173_report = load_json_report(v173_report_path)
    v172_report = load_json_report(v172_report_path)
    rows = load_jsonl_dataset(v172_dataset_path)
    source_validation = validate_v174_sources(
        v173_report=v173_report,
        v172_report=v172_report,
        rows=rows,
        expected_v173_exact_digest=expected_v173_exact_digest,
        expected_v173_classification=expected_v173_classification,
        expected_v172_dataset_digest=expected_v172_dataset_digest,
        support_provenance_seeds=support_provenance_seeds,
    )
    leakage_scan = trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in rows]
    )
    row_schema_validation = validate_v172_target_rows(
        rows,
        support_provenance_seeds=support_provenance_seeds,
    )
    if (
        source_validation.get("passed") is True
        and leakage_scan.get("passed") is True
        and row_schema_validation.get("passed") is True
    ):
        context = _split_context(
            rows,
            support_provenance_seeds=support_provenance_seeds,
            feature_vector_builder=_public_feature_vector,
        )
        lane_a = lane_a_v173_score_mechanics_autopsy(rows, context=context)
        lane_b = lane_b_local_support_frontier(rows, context=context)
        lane_c = lane_c_action_ranking_capacity_probe(rows, context=context)
        lane_d = lane_d_conformal_action_conditional_set_calibration(
            rows,
            context=context,
            lane_b=lane_b,
            lane_c=lane_c,
        )
        lane_e = lane_e_public_context_sufficiency_probe(
            rows,
            baseline_context=context,
            lane_b=lane_b,
            lane_c=lane_c,
            support_provenance_seeds=support_provenance_seeds,
        )
    else:
        lane_a = _skipped_lane("lane_a_v173_score_mechanics_autopsy")
        lane_b = _skipped_lane("lane_b_local_support_frontier")
        lane_c = _skipped_lane("lane_c_action_ranking_capacity_probe")
        lane_d = _skipped_lane("lane_d_conformal_action_conditional_set_calibration")
        lane_e = _skipped_lane("lane_e_public_context_sufficiency_probe")
    planner = lane_f_next_route_planner(
        source_validation=source_validation,
        leakage_scan=leakage_scan,
        row_schema_validation=row_schema_validation,
        lane_a=lane_a,
        lane_b=lane_b,
        lane_c=lane_c,
        lane_d=lane_d,
        lane_e=lane_e,
    )
    classification = str(planner.get("classification"))
    plan_output = _maybe_write_v175_plan(
        planner=planner,
        output_path=v175_plan_output_path,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v173_report": str(v173_report_path),
            "v172_report": str(v172_report_path),
            "v172_dataset": str(v172_dataset_path),
            "expected_v173_exact_digest": expected_v173_exact_digest,
            "expected_v173_classification": expected_v173_classification,
            "expected_v172_dataset_digest": expected_v172_dataset_digest,
            "support_provenance_seeds": [
                int(seed) for seed in support_provenance_seeds
            ],
            "k_nearest_values": [int(value) for value in K_NEAREST_VALUES],
            "radius_values": [_round(value) for value in RADIUS_VALUES],
            "diagnostics_only_no_runtime_artifact": True,
        },
        "source_validation": source_validation,
        "leakage_scan": leakage_scan,
        "row_schema_validation": row_schema_validation,
        "lanes": {
            "lane_a_v173_score_mechanics_autopsy": lane_a,
            "lane_b_local_support_frontier": lane_b,
            "lane_c_action_ranking_capacity_probe": lane_c,
            "lane_d_conformal_action_conditional_set_calibration": lane_d,
            "lane_e_public_context_sufficiency_probe": lane_e,
            "lane_f_next_route_planner": planner,
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(planner),
        "v175_plan_output": plan_output,
        "dataset_digest": stable_payload_digest(rows),
        "support_provenance_seed_policy": _support_provenance_seed_policy(
            support_provenance_seeds=support_provenance_seeds,
        ),
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v174_sources(
    *,
    v173_report: Mapping[str, object],
    v172_report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v173_exact_digest: str | None = EXPECTED_V173_EXACT_DIGEST,
    expected_v173_classification: str = EXPECTED_V173_CLASSIFICATION,
    expected_v172_dataset_digest: str | None = EXPECTED_V172_DATASET_DIGEST,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> dict[str, object]:
    failures: list[str] = []
    observed_classification = str(
        _mapping(v173_report.get("classification")).get("primary") or ""
    )
    exact = exact_digest_validation_report(v173_report)
    observed_exact = str(v173_report.get("exact_digest") or "")
    dataset_digest = stable_payload_digest(rows)
    v173_dataset_digest = str(v173_report.get("dataset_digest") or "")
    if (
        v173_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_SCHEMA_VERSION
    ):
        failures.append("v173_schema_version_mismatch")
    if (
        v173_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_POLICY
    ):
        failures.append("v173_policy_mismatch")
    if observed_classification != expected_v173_classification:
        failures.append("v173_unexpected_classification")
    if exact.get("passed") is not True:
        failures.append("v173_exact_digest_mismatch")
    if expected_v173_exact_digest and observed_exact != expected_v173_exact_digest:
        failures.append("v173_unexpected_exact_digest")
    if expected_v172_dataset_digest and dataset_digest != expected_v172_dataset_digest:
        failures.append("v172_unexpected_dataset_digest")
    if v173_dataset_digest and v173_dataset_digest != dataset_digest:
        failures.append("v173_dataset_digest_mismatch")
    if v173_dataset_digest and expected_v172_dataset_digest:
        if v173_dataset_digest != expected_v172_dataset_digest:
            failures.append("v173_unexpected_reported_dataset_digest")
    source = _mapping(v173_report.get("source_validation"))
    if source and source.get("passed") is not True:
        failures.append("v173_source_validation_not_passed")
    split = _mapping(v173_report.get("source_split_evaluation"))
    if split and _int(split.get("row_count")) != len(rows):
        failures.append("v173_split_row_count_mismatch")
    lifecycle = _v173_lifecycle_validation(v173_report)
    if lifecycle.get("passed") is not True:
        failures.append("v173_lifecycle_not_diagnostics_only")
    source_seed_validation = _source_seed_validation(
        rows,
        support_provenance_seeds=support_provenance_seeds,
    )
    if source_seed_validation.get("passed") is not True:
        failures.append("v172_support_provenance_seed_rows_invalid")
    v172_dataset = _mapping(v172_report.get("dataset"))
    v172_reported_dataset_digest = str(v172_dataset.get("dataset_digest") or "")
    if v172_reported_dataset_digest and v172_reported_dataset_digest != dataset_digest:
        failures.append("v172_reported_dataset_digest_mismatch")
    return {
        "policy": "m3_carrion_survivor_continuation_v174_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v173_exact_digest": expected_v173_exact_digest,
        "observed_v173_exact_digest": observed_exact,
        "v173_exact_digest_validation": exact,
        "expected_v173_classification": expected_v173_classification,
        "observed_v173_classification": observed_classification,
        "expected_v172_dataset_digest": expected_v172_dataset_digest,
        "observed_v172_dataset_digest": dataset_digest,
        "v173_reported_dataset_digest": v173_dataset_digest,
        "v172_reported_dataset_digest": v172_reported_dataset_digest,
        "row_count": len(rows),
        "v173_lifecycle_validation": lifecycle,
        "support_provenance_seed_validation": source_seed_validation,
    }


def lane_a_v173_score_mechanics_autopsy(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
) -> dict[str, object]:
    per_row = []
    full_set_count = 0
    positive_equals_public_count = 0
    positive_action_counts: Counter[int] = Counter()
    margins: list[float] = []
    entropies: list[float] = []
    for row_index in _evaluation_indexes(context):
        row = rows[row_index]
        distances = _distances_for_row(context, row_index)
        scores = _support_scores(rows, row_index, distances)
        public_actions = _public_actions_for_row(row)
        positive_actions = _positive_ranked_actions(scores)
        public_width = len(public_actions)
        positive_count = len(positive_actions)
        full_set = set(positive_actions) == set(public_actions)
        if full_set:
            full_set_count += 1
        if positive_count == public_width:
            positive_equals_public_count += 1
        positive_action_counts.update([positive_count])
        top = _top_two_margin(scores)
        margins.append(float(top["top_1_top_2_margin"]))
        entropy = _score_entropy(scores)
        entropies.append(entropy)
        per_row.append(
            {
                "row_index": row_index,
                "left_out_seed": _source_seed(row),
                "row_type": _row_type(row),
                "safe_action_set": _safe_action_set(row),
                "public_mask_width": public_width,
                "positive_action_count": positive_count,
                "score_entropy": _round(entropy),
                "top_1_action": top["top_1_action"],
                "top_2_action": top["top_2_action"],
                "top_1_score": top["top_1_score"],
                "top_2_score": top["top_2_score"],
                "top_1_top_2_margin": top["top_1_top_2_margin"],
                "set_valued_output_equals_full_public_mask": full_set,
            }
        )
    row_count = len(per_row)
    return {
        "policy": "m3_carrion_survivor_continuation_v174_lane_a_v173_score_mechanics_autopsy_v1",
        "row_count": row_count,
        "score_rule_recomputed": "include every public action with score > 0.0",
        "score_entropy_mean": _mean(entropies),
        "top_1_top_2_margin_mean": _mean(margins),
        "positive_action_count_distribution": {
            str(width): int(count)
            for width, count in sorted(positive_action_counts.items())
        },
        "full_public_mask_set_count": int(full_set_count),
        "full_public_mask_set_share": _rate(full_set_count, row_count),
        "positive_count_equals_public_mask_width_count": int(
            positive_equals_public_count
        ),
        "positive_count_equals_public_mask_width_share": _rate(
            positive_equals_public_count,
            row_count,
        ),
        "full_set_positive_rule_proof": {
            "positive_rule_is_score_greater_than_zero": True,
            "uncertainty_calibration_used": False,
            "threshold_other_than_zero_used": False,
            "full_set_count_equals_positive_count_equals_public_width_count": (
                full_set_count == positive_equals_public_count
            ),
            "full_set_behavior_explained_by_positive_score_rule": (
                full_set_count == positive_equals_public_count == row_count
            ),
        },
        "per_row_score_mechanics": per_row,
        "floor_passed": False,
    }


def lane_b_local_support_frontier(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
) -> dict[str, object]:
    entries = []
    for k in K_NEAREST_VALUES:
        entries.append(
            _local_support_entry(
                rows,
                context=context,
                mode="k_nearest",
                label=f"k_nearest_{k}",
                k=int(k),
                radius=None,
            )
        )
    for radius in RADIUS_VALUES:
        entries.append(
            _local_support_entry(
                rows,
                context=context,
                mode="radius",
                label=f"radius_{_radius_label(radius)}",
                k=None,
                radius=float(radius),
            )
        )
    best = _best_local_support_entry(entries)
    return {
        "policy": "m3_carrion_survivor_continuation_v174_lane_b_local_support_frontier_v1",
        "leave_one_support_seed_out": True,
        "k_nearest_values": [int(value) for value in K_NEAREST_VALUES],
        "radius_values": [_round(value) for value in RADIUS_VALUES],
        "frontier_entries": entries,
        "best_frontier_entry": best,
        "local_support_ready_for_shadow_diagnostic_no_runtime": bool(
            best.get("floor_passed") is True
        ),
        "floor_passed": bool(best.get("floor_passed") is True),
    }


def lane_c_action_ranking_capacity_probe(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
) -> dict[str, object]:
    entries = []
    for label, k in RANKING_CONFIGS:
        entries.append(
            _ranking_capacity_entry(
                rows,
                context=context,
                label=label,
                k=k,
                shuffled_targets=False,
            )
        )
    negative_entries = []
    for label, k in RANKING_CONFIGS:
        negative_entries.append(
            _ranking_capacity_entry(
                rows,
                context=context,
                label=f"{label}_shuffled_target_control",
                k=k,
                shuffled_targets=True,
            )
        )
    best = _best_ranking_entry(entries, negative_entries)
    return {
        "policy": "m3_carrion_survivor_continuation_v174_lane_c_action_ranking_capacity_probe_v1",
        "rankq_inspired_diagnostics_only": True,
        "runtime_artifact_created": False,
        "ranking_entries": entries,
        "shuffled_target_negative_control_entries": negative_entries,
        "best_ranking_entry": best,
        "action_ranking_capacity_ready_for_v175_diagnostic_fit_no_runtime": bool(
            best.get("capacity_floor_passed") is True
        ),
        "floor_passed": bool(best.get("capacity_floor_passed") is True),
    }


def lane_d_conformal_action_conditional_set_calibration(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    lane_b: Mapping[str, object],
    lane_c: Mapping[str, object],
) -> dict[str, object]:
    best_local = _mapping(lane_b.get("best_frontier_entry"))
    local_label = str(best_local.get("label") or "k_nearest_25")
    local_k = _int(best_local.get("k"), default=25) if best_local.get("k") else None
    local_radius = (
        _float(best_local.get("radius"))
        if best_local.get("radius") is not None
        else None
    )
    best_ranking = _mapping(lane_c.get("best_ranking_entry"))
    ranking_k = (
        _int(best_ranking.get("k"))
        if best_ranking.get("k") is not None
        else None
    )
    entries = [
        _conformal_entry(
            rows,
            context=context,
            score_source="lane_b_local_support",
            label=local_label,
            target_coverage=CONFORMAL_TARGET_COVERAGE,
            local_k=local_k,
            local_radius=local_radius,
            ranking_k=None,
        ),
        _conformal_entry(
            rows,
            context=context,
            score_source="lane_c_action_ranking",
            label=str(best_ranking.get("label") or "all_train"),
            target_coverage=CONFORMAL_TARGET_COVERAGE,
            local_k=None,
            local_radius=None,
            ranking_k=ranking_k,
        ),
    ]
    best = max(
        entries,
        key=lambda entry: (
            1 if entry.get("floor_passed") is True else 0,
            _float(entry.get("coverage_rate")),
            -_float(entry.get("average_action_conditional_width")),
        ),
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v174_lane_d_conformal_action_conditional_set_calibration_v1",
        "target_coverage": _round(CONFORMAL_TARGET_COVERAGE),
        "calibration_policy": (
            "calibrate candidate-rank widths only on non-heldout train folds, "
            "then evaluate on the left-out support seed"
        ),
        "entries": entries,
        "best_entry": best,
        "safety_only_holds_by_returning_full_masks": any(
            entry.get("safety_only_holds_by_returning_full_masks") is True
            for entry in entries
        ),
        "floor_passed": bool(best.get("floor_passed") is True),
    }


def lane_e_public_context_sufficiency_probe(
    rows: Sequence[Mapping[str, object]],
    *,
    baseline_context: Mapping[str, object],
    lane_b: Mapping[str, object],
    lane_c: Mapping[str, object],
    support_provenance_seeds: Sequence[int],
) -> dict[str, object]:
    baseline_projection = _projection_summary(
        rows,
        context=baseline_context,
        projection_id="v173_static_public_feature_vector",
    )
    projections = [baseline_projection]
    for projection_id, builder in (
        ("decoded_numeric_observation_summary", _decoded_observation_summary_vector),
        ("action_mask_geometry", _action_mask_geometry_vector),
        (
            "decoded_summary_plus_action_mask_geometry",
            _decoded_summary_plus_action_mask_geometry_vector,
        ),
    ):
        context = _split_context(
            rows,
            support_provenance_seeds=support_provenance_seeds,
            feature_vector_builder=builder,
        )
        projections.append(
            _projection_summary(rows, context=context, projection_id=projection_id)
        )
    unavailable = [
        {
            "projection_id": "previous_same_agent_public_observation_delta_window",
            "available_without_schema_change": False,
            "reason": (
                "v172 public rows do not carry previous same-agent public "
                "observation snapshots or deltas"
            ),
        },
        {
            "projection_id": "short_public_trajectory_window",
            "available_without_schema_change": False,
            "reason": (
                "v172 public rows are single-observation target rows; no "
                "public trajectory window is serialized in trainable features"
            ),
        },
    ]
    baseline = _mapping(projections[0])
    improved = []
    for projection in projections[1:]:
        local_gain = _float(projection.get("best_local_singleton_hit_rate")) - _float(
            baseline.get("best_local_singleton_hit_rate")
        )
        ranking_gain = _float(projection.get("ranking_pairwise_accuracy")) - _float(
            baseline.get("ranking_pairwise_accuracy")
        )
        top3_gain = _float(projection.get("ranking_top_3_safe_hit_rate")) - _float(
            baseline.get("ranking_top_3_safe_hit_rate")
        )
        if max(local_gain, ranking_gain, top3_gain) >= PUBLIC_CONTEXT_MIN_IMPROVEMENT:
            improved.append(
                {
                    "projection_id": projection.get("projection_id"),
                    "best_local_singleton_hit_rate_gain": _round(local_gain),
                    "ranking_pairwise_accuracy_gain": _round(ranking_gain),
                    "ranking_top_3_safe_hit_rate_gain": _round(top3_gain),
                }
            )
    return {
        "policy": "m3_carrion_survivor_continuation_v174_lane_e_public_context_sufficiency_probe_v1",
        "forbidden_trainable_fields": [
            "seed",
            "fixture",
            "branch",
            "tick",
            "agent",
            "private_state",
            "outcome",
            "target",
            "runtime_requested_or_resolved_action",
        ],
        "projections": projections,
        "unavailable_public_context_projections": unavailable,
        "any_projection_improved": bool(improved),
        "improved_projection_count": len(improved),
        "improved_projections": improved,
        "min_improvement_required": _round(PUBLIC_CONTEXT_MIN_IMPROVEMENT),
        "floor_passed": bool(improved),
        "lane_b_reference_best_label": _mapping(
            lane_b.get("best_frontier_entry")
        ).get("label"),
        "lane_c_reference_best_label": _mapping(
            lane_c.get("best_ranking_entry")
        ).get("label"),
    }


def lane_f_next_route_planner(
    *,
    source_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    lane_a: Mapping[str, object],
    lane_b: Mapping[str, object],
    lane_c: Mapping[str, object],
    lane_d: Mapping[str, object],
    lane_e: Mapping[str, object],
) -> dict[str, object]:
    if (
        source_validation.get("passed") is not True
        or leakage_scan.get("passed") is not True
        or row_schema_validation.get("passed") is not True
    ):
        classification = "v174_invalid_closed_no_shadow"
        route = "close_invalid_without_shadow"
        plan_rows: list[dict[str, object]] = []
    elif lane_b.get("floor_passed") is True:
        classification = "v174_local_support_ready_for_shadow_diagnostic_no_runtime"
        route = "future_shadow_diagnostic_only_after_explicit_authorization"
        plan_rows = []
    elif lane_c.get("floor_passed") is True:
        classification = (
            "v174_action_ranking_capacity_ready_for_v175_diagnostic_fit_no_runtime"
        )
        route = "v175_diagnostics_only_action_ranking_fit"
        plan_rows = _v175_action_ranking_plan_rows(lane_c=lane_c)
    elif lane_e.get("floor_passed") is True:
        classification = "v174_public_context_expansion_ready_no_runtime"
        route = "v175_public_context_expansion_without_runtime_artifact"
        plan_rows = []
    elif _conformal_safety_only_by_full_masks(lane_d):
        classification = "v174_world_model_or_replay_expansion_plan_ready_no_training"
        route = "v175_exact_branch_replay_or_world_model_transition_diagnostic"
        plan_rows = _v175_replay_or_world_model_plan_rows(
            lane_a=lane_a,
            lane_b=lane_b,
            lane_c=lane_c,
            lane_d=lane_d,
            lane_e=lane_e,
        )
    else:
        classification = "v174_scorer_family_closed_static_public_features_insufficient"
        route = "close_static_public_feature_scorer_family"
        plan_rows = []
    return {
        "policy": "m3_carrion_survivor_continuation_v174_lane_f_next_route_planner_v1",
        "classification": classification,
        "recommended_next_route": route,
        "training_authorized": False,
        "runtime_artifact_authorized": False,
        "runtime_action_selection_change_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
        "replay_expansion_ran": False,
        "world_model_training_ran": False,
        "v175_plan_rows": plan_rows,
        "write_optional_v175_plan_jsonl": bool(
            classification
            == "v174_world_model_or_replay_expansion_plan_ready_no_training"
        ),
    }


def _local_support_entry(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    mode: str,
    label: str,
    k: int | None,
    radius: float | None,
) -> dict[str, object]:
    predictions = []
    for row_index in _evaluation_indexes(context):
        row = rows[row_index]
        distances = _selected_distances(
            _distances_for_row(context, row_index),
            k=k,
            radius=radius,
        )
        scores = _support_scores(rows, row_index, distances)
        ranked = _positive_ranked_actions(scores)
        safe_set = _safe_action_set(row)
        public_actions = _public_actions_for_row(row)
        singleton = ranked[0] if ranked else None
        baselines = _baseline_predictions(
            rows,
            row_index=row_index,
            train_indexes=_train_indexes_for_row(context, row_index),
        )
        predictions.append(
            {
                "row_index": row_index,
                "left_out_seed": _source_seed(row),
                "row_type": _row_type(row),
                "safe_action_set": safe_set,
                "public_actions": public_actions,
                "selected_neighbor_count": len(distances),
                "singleton_action": singleton,
                "top_2_actions": ranked[:2],
                "top_3_actions": ranked[:3],
                "unsupported": bool(
                    singleton is not None and singleton not in set(public_actions)
                ),
                "no_prediction": singleton is None,
                "action_frequency_baselines": baselines["action_frequency"],
                "mask_only_baselines": baselines["mask_only"],
            }
        )
    metrics = _local_prediction_metrics(predictions)
    metrics.update(
        {
            "mode": mode,
            "label": label,
            "k": k,
            "radius": _round(radius) if radius is not None else None,
            "floor_passed": _local_floor_passed(metrics),
        }
    )
    return metrics


def _ranking_capacity_entry(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    label: str,
    k: int | None,
    shuffled_targets: bool,
) -> dict[str, object]:
    pairwise_correct = 0.0
    pairwise_total = 0
    pairwise_ties = 0
    top_hits = Counter()
    top_widths: dict[int, list[int]] = defaultdict(list)
    per_action: dict[str, Counter[str]] = {
        action: Counter() for action in ACTION_NAMES
    }
    predictions = []
    for row_index in _evaluation_indexes(context):
        row = rows[row_index]
        selected = _selected_distances(_distances_for_row(context, row_index), k=k)
        scores = _ranking_scores(
            rows,
            row_index=row_index,
            distances=selected,
            shuffled_targets=shuffled_targets,
        )
        ranked = _ranked_actions(scores)
        safe = set(_safe_action_set(row))
        public_actions = list(scores)
        unsafe = [action for action in public_actions if action not in safe]
        for safe_action in sorted(safe, key=_action_order):
            if safe_action not in scores:
                continue
            per_action[safe_action].update(["safe_occurrence"])
            safe_score = scores[safe_action]
            for unsafe_action in unsafe:
                pairwise_total += 1
                unsafe_score = scores[unsafe_action]
                if safe_score > unsafe_score:
                    pairwise_correct += 1.0
                elif safe_score == unsafe_score:
                    pairwise_correct += 0.5
                    pairwise_ties += 1
                    per_action[safe_action].update(["pairwise_tie"])
                else:
                    per_action[safe_action].update(["pairwise_loss"])
            if safe_action not in ranked[:1]:
                per_action[safe_action].update(["top_1_failure"])
            if safe_action not in ranked[:3]:
                per_action[safe_action].update(["top_3_failure"])
        for width in (1, 2, 3):
            top = ranked[:width]
            top_hits.update([width] if set(top) & safe else [])
            top_widths[width].append(len(top))
        predictions.append(
            {
                "row_index": row_index,
                "left_out_seed": _source_seed(row),
                "safe_action_set": _safe_action_set(row),
                "top_1_actions": ranked[:1],
                "top_2_actions": ranked[:2],
                "top_3_actions": ranked[:3],
            }
        )
    row_count = len(predictions)
    return {
        "label": label,
        "k": k,
        "shuffled_target_negative_control": shuffled_targets,
        "row_count": row_count,
        "pairwise_comparison_count": int(pairwise_total),
        "pairwise_accuracy": _round(pairwise_correct / pairwise_total)
        if pairwise_total
        else 0.0,
        "pairwise_tie_count": int(pairwise_ties),
        "top_1_safe_hit_rate": _rate(int(top_hits.get(1, 0)), row_count),
        "top_2_safe_hit_rate": _rate(int(top_hits.get(2, 0)), row_count),
        "top_3_safe_hit_rate": _rate(int(top_hits.get(3, 0)), row_count),
        "top_1_average_width": _mean(top_widths[1]),
        "top_2_average_width": _mean(top_widths[2]),
        "top_3_average_width": _mean(top_widths[3]),
        "per_action_failures": _per_action_failure_payload(per_action),
        "predictions_sample": predictions[:24],
    }


def _conformal_entry(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    score_source: str,
    label: str,
    target_coverage: float,
    local_k: int | None,
    local_radius: float | None,
    ranking_k: int | None,
) -> dict[str, object]:
    per_seed = {}
    action_misses: dict[str, Counter[str]] = {
        action: Counter() for action in ACTION_NAMES
    }
    total_covered = 0
    total_rows = 0
    widths: list[int] = []
    full_sets = 0
    for heldout_seed in _support_seeds(context):
        train_indexes = [
            index
            for index, seed in enumerate(_row_seeds(context))
            if int(seed) != int(heldout_seed)
        ]
        calibration = _calibration_widths(
            rows,
            context=context,
            train_indexes=train_indexes,
            score_source=score_source,
            target_coverage=target_coverage,
            local_k=local_k,
            local_radius=local_radius,
            ranking_k=ranking_k,
        )
        seed_rows = []
        seed_covered = 0
        seed_widths: list[int] = []
        seed_full_sets = 0
        for row_index in _indexes_by_seed(context).get(int(heldout_seed), []):
            scores = _candidate_scores(
                rows,
                context=context,
                row_index=row_index,
                score_source=score_source,
                local_k=local_k,
                local_radius=local_radius,
                ranking_k=ranking_k,
                support_indexes=train_indexes,
            )
            ranked = _ranked_actions(scores)
            ranks = {action: rank + 1 for rank, action in enumerate(ranked)}
            candidate_set = [
                action
                for action in ranked
                if ranks[action]
                <= _int(
                    _mapping(calibration.get("action_rank_widths")).get(action),
                    default=_int(calibration.get("global_rank_width"), default=1),
                )
            ]
            public = set(_public_actions_for_row(rows[row_index]))
            safe = set(_safe_action_set(rows[row_index]))
            covered = bool(set(candidate_set) & safe)
            full = set(candidate_set) == public
            if covered:
                seed_covered += 1
            if full:
                seed_full_sets += 1
            seed_widths.append(len(candidate_set))
            for safe_action in safe:
                action_misses[safe_action].update(["count"])
                if safe_action not in candidate_set:
                    action_misses[safe_action].update(["miss"])
            seed_rows.append(
                {
                    "row_index": row_index,
                    "covered": covered,
                    "candidate_width": len(candidate_set),
                    "full_public_mask_set": full,
                }
            )
        per_seed[str(heldout_seed)] = {
            "row_count": len(seed_rows),
            "coverage_rate": _rate(seed_covered, len(seed_rows)),
            "average_action_conditional_width": _mean(seed_widths),
            "full_set_share": _rate(seed_full_sets, len(seed_rows)),
            "calibration": calibration,
        }
        total_covered += seed_covered
        total_rows += len(seed_rows)
        widths.extend(seed_widths)
        full_sets += seed_full_sets
    coverage_rate = _rate(total_covered, total_rows)
    full_set_share = _rate(full_sets, total_rows)
    safety_only_by_full_masks = (
        coverage_rate >= _round(target_coverage)
        and full_set_share > CONFORMAL_MAX_FULL_SET_SHARE
    )
    return {
        "score_source": score_source,
        "label": label,
        "target_coverage": _round(target_coverage),
        "coverage_rate": coverage_rate,
        "miss_rate": _round(1.0 - coverage_rate),
        "average_action_conditional_width": _mean(widths),
        "full_set_share": full_set_share,
        "action_conditional_miss_rates": {
            action: {
                "row_count": int(counter.get("count", 0)),
                "miss_count": int(counter.get("miss", 0)),
                "miss_rate": _rate(
                    int(counter.get("miss", 0)),
                    int(counter.get("count", 0)),
                ),
            }
            for action, counter in action_misses.items()
            if int(counter.get("count", 0)) > 0
        },
        "per_seed_coverage": per_seed,
        "safety_only_holds_by_returning_full_masks": safety_only_by_full_masks,
        "floor_passed": (
            coverage_rate >= _round(target_coverage)
            and full_set_share <= CONFORMAL_MAX_FULL_SET_SHARE
        ),
    }


def _projection_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    projection_id: str,
) -> dict[str, object]:
    local_entries = [
        _local_support_entry(
            rows,
            context=context,
            mode="k_nearest",
            label=f"{projection_id}_k_nearest_{k}",
            k=int(k),
            radius=None,
        )
        for k in K_NEAREST_VALUES
    ]
    best_local = _best_local_support_entry(local_entries)
    ranking = _ranking_capacity_entry(
        rows,
        context=context,
        label=f"{projection_id}_all_train",
        k=None,
        shuffled_targets=False,
    )
    return {
        "projection_id": projection_id,
        "available_without_schema_change": True,
        "uses_only_public_rows": True,
        "uses_seed_fixture_branch_tick_agent_private_outcome_fields": False,
        "best_local_label": best_local.get("label"),
        "best_local_singleton_hit_rate": best_local.get("singleton_hit_rate"),
        "best_local_top_3_set_hit_rate": best_local.get("top_3_set_hit_rate"),
        "ranking_pairwise_accuracy": ranking.get("pairwise_accuracy"),
        "ranking_top_1_safe_hit_rate": ranking.get("top_1_safe_hit_rate"),
        "ranking_top_3_safe_hit_rate": ranking.get("top_3_safe_hit_rate"),
    }


def _split_context(
    rows: Sequence[Mapping[str, object]],
    *,
    support_provenance_seeds: Sequence[int],
    feature_vector_builder: FeatureVectorBuilder,
) -> dict[str, object]:
    row_seeds = [_source_seed(row) for row in rows]
    indexes_by_seed: dict[int, list[int]] = defaultdict(list)
    for index, seed in enumerate(row_seeds):
        indexes_by_seed[int(seed)].append(index)
    vectors = _normalized_vectors(rows, feature_vector_builder)
    support_seeds = sorted({int(seed) for seed in support_provenance_seeds})
    distances_by_row: dict[int, list[tuple[float, int]]] = {}
    train_indexes_by_row: dict[int, list[int]] = {}
    for seed in support_seeds:
        train_indexes = [
            index for index, row_seed in enumerate(row_seeds) if int(row_seed) != seed
        ]
        for row_index in indexes_by_seed.get(seed, []):
            distances = sorted(
                (
                    _squared_distance(vectors[row_index], vectors[train_index]),
                    train_index,
                )
                for train_index in train_indexes
            )
            distances_by_row[row_index] = distances
            train_indexes_by_row[row_index] = train_indexes
    return {
        "support_seeds": support_seeds,
        "row_seeds": row_seeds,
        "indexes_by_seed": dict(indexes_by_seed),
        "vectors": vectors,
        "distances_by_row": distances_by_row,
        "train_indexes_by_row": train_indexes_by_row,
        "evaluation_indexes": sorted(distances_by_row),
    }


def _normalized_vectors(
    rows: Sequence[Mapping[str, object]],
    feature_vector_builder: FeatureVectorBuilder,
) -> list[list[float]]:
    if feature_vector_builder is _public_feature_vector:
        return _normalized_feature_vectors(rows)
    raw_vectors = [feature_vector_builder(row) for row in rows]
    width = max((len(vector) for vector in raw_vectors), default=0)
    padded = [vector + [0.0] * (width - len(vector)) for vector in raw_vectors]
    if not padded:
        return []
    mins = [min(vector[index] for vector in padded) for index in range(width)]
    maxs = [max(vector[index] for vector in padded) for index in range(width)]
    normalized = []
    for vector in padded:
        current = []
        for index, value in enumerate(vector):
            span = maxs[index] - mins[index]
            current.append(0.0 if span == 0.0 else (float(value) - mins[index]) / span)
        normalized.append(current)
    return normalized


def _support_scores(
    rows: Sequence[Mapping[str, object]],
    row_index: int,
    distances: Sequence[tuple[float, int]],
) -> dict[str, float]:
    public_mask = _complete_action_mask(_mapping(rows[row_index].get("public_action_mask")))
    scores = {action: 0.0 for action in ACTION_NAMES if public_mask.get(action)}
    for distance, train_index in distances:
        weight = 1.0 / (1.0 + float(distance))
        for action in _safe_action_set(rows[train_index]):
            if action in scores:
                scores[action] += weight
    return {action: _round(value) for action, value in scores.items()}


def _ranking_scores(
    rows: Sequence[Mapping[str, object]],
    *,
    row_index: int,
    distances: Sequence[tuple[float, int]],
    shuffled_targets: bool,
) -> dict[str, float]:
    public_mask = _complete_action_mask(_mapping(rows[row_index].get("public_action_mask")))
    numerators = {action: 0.0 for action in ACTION_NAMES if public_mask.get(action)}
    denominators = {action: 0.0 for action in numerators}
    for distance, train_index in distances:
        source_index = (train_index + 1) % len(rows) if shuffled_targets else train_index
        source_safe = set(_safe_action_set(rows[source_index]))
        train_mask = _complete_action_mask(
            _mapping(rows[train_index].get("public_action_mask"))
        )
        weight = 1.0 / (1.0 + float(distance))
        for action in numerators:
            if train_mask.get(action):
                denominators[action] += weight
                numerators[action] += weight * (1.0 if action in source_safe else 0.0)
    return {
        action: _round(
            numerators[action] / denominators[action]
            if denominators[action] > 0.0
            else 0.0
        )
        for action in numerators
    }


def _candidate_scores(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    row_index: int,
    score_source: str,
    local_k: int | None,
    local_radius: float | None,
    ranking_k: int | None,
    support_indexes: Sequence[int],
) -> dict[str, float]:
    distances = _distances_for_row_against_indexes(
        context,
        row_index=row_index,
        indexes=support_indexes,
    )
    if score_source == "lane_b_local_support":
        selected = _selected_distances(distances, k=local_k, radius=local_radius)
        return _support_scores(rows, row_index, selected)
    selected = _selected_distances(distances, k=ranking_k)
    return _ranking_scores(
        rows,
        row_index=row_index,
        distances=selected,
        shuffled_targets=False,
    )


def _calibration_widths(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    train_indexes: Sequence[int],
    score_source: str,
    target_coverage: float,
    local_k: int | None,
    local_radius: float | None,
    ranking_k: int | None,
) -> dict[str, object]:
    global_required: list[int] = []
    by_action: dict[str, list[int]] = defaultdict(list)
    row_seeds = _row_seeds(context)
    for row_index in train_indexes:
        support_indexes = [
            index
            for index in train_indexes
            if int(row_seeds[index]) != int(row_seeds[row_index])
        ]
        if not support_indexes:
            support_indexes = [index for index in train_indexes if index != row_index]
        scores = _candidate_scores(
            rows,
            context=context,
            row_index=row_index,
            score_source=score_source,
            local_k=local_k,
            local_radius=local_radius,
            ranking_k=ranking_k,
            support_indexes=support_indexes,
        )
        ranked = _ranked_actions(scores)
        ranks = {action: rank + 1 for rank, action in enumerate(ranked)}
        safe_ranks = [
            ranks[action] for action in _safe_action_set(rows[row_index]) if action in ranks
        ]
        if not safe_ranks:
            continue
        global_required.append(min(safe_ranks))
        for action in _safe_action_set(rows[row_index]):
            if action in ranks:
                by_action[action].append(ranks[action])
    global_width = _conformal_quantile_rank(global_required, target_coverage)
    action_widths = {
        action: _conformal_quantile_rank(ranks, target_coverage)
        for action, ranks in sorted(by_action.items(), key=lambda item: _action_order(item[0]))
    }
    for action in ACTION_NAMES:
        action_widths.setdefault(action, global_width)
    return {
        "calibration_row_count": len(global_required),
        "global_rank_width": int(global_width),
        "action_rank_widths": action_widths,
    }


def _local_prediction_metrics(
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    row_count = len(predictions)
    singleton_hits = sum(_hit(prediction, "singleton_action") for prediction in predictions)
    top2_hits = sum(_set_hit(prediction, "top_2_actions") for prediction in predictions)
    top3_hits = sum(_set_hit(prediction, "top_3_actions") for prediction in predictions)
    unique = [item for item in predictions if item.get("row_type") == "unique"]
    tied = [item for item in predictions if item.get("row_type") == "tied"]
    predicted = Counter(
        str(item.get("singleton_action"))
        for item in predictions
        if item.get("singleton_action")
    )
    dominant = _dominant_count_share(predicted)
    baseline = _baseline_metrics_from_predictions(predictions)
    neighbor_counts = [_int(item.get("selected_neighbor_count")) for item in predictions]
    per_seed = _per_seed_prediction_metrics(predictions)
    return {
        "row_count": row_count,
        "singleton_hit_count": int(singleton_hits),
        "singleton_hit_rate": _rate(singleton_hits, row_count),
        "top_2_set_hit_count": int(top2_hits),
        "top_2_set_hit_rate": _rate(top2_hits, row_count),
        "top_3_set_hit_count": int(top3_hits),
        "top_3_set_hit_rate": _rate(top3_hits, row_count),
        "robust_winner_unique_row_count": len(unique),
        "robust_winner_hit_count": int(
            sum(_hit(prediction, "singleton_action") for prediction in unique)
        ),
        "robust_winner_hit_rate": _rate(
            sum(_hit(prediction, "singleton_action") for prediction in unique),
            len(unique),
        ),
        "tied_row_count": len(tied),
        "tied_row_singleton_hit_count": int(
            sum(_hit(prediction, "singleton_action") for prediction in tied)
        ),
        "tied_row_singleton_hit_rate": _rate(
            sum(_hit(prediction, "singleton_action") for prediction in tied),
            len(tied),
        ),
        "tied_row_top_2_set_hit_rate": _rate(
            sum(_set_hit(prediction, "top_2_actions") for prediction in tied),
            len(tied),
        ),
        "tied_row_top_3_set_hit_rate": _rate(
            sum(_set_hit(prediction, "top_3_actions") for prediction in tied),
            len(tied),
        ),
        "dominant_action": dominant.get("key"),
        "dominant_action_count": dominant.get("count"),
        "dominant_action_share": dominant.get("share"),
        "predicted_action_counts": {
            action: int(predicted.get(action, 0))
            for action in ACTION_NAMES
            if int(predicted.get(action, 0)) > 0
        },
        "unsupported_count": int(
            sum(item.get("unsupported") is True for item in predictions)
        ),
        "no_prediction_count": int(
            sum(item.get("no_prediction") is True for item in predictions)
        ),
        "average_selected_neighbor_count": _mean(neighbor_counts),
        "unsupported_row_count": int(
            sum(_int(item.get("selected_neighbor_count")) == 0 for item in predictions)
        ),
        "baselines_at_matching_width": baseline,
        "per_left_out_seed": per_seed,
    }


def _baseline_metrics_from_predictions(
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    row_count = len(predictions)
    result: dict[str, object] = {}
    for baseline_key, prefix in (
        ("action_frequency_baselines", "action_frequency"),
        ("mask_only_baselines", "mask_only"),
    ):
        for width in (1, 2, 3):
            hit_count = 0
            for prediction in predictions:
                baseline = _mapping(prediction.get(baseline_key))
                actions = _list_of_strings(baseline.get(str(width)))
                safe = set(_list_of_strings(prediction.get("safe_action_set")))
                if set(actions) & safe:
                    hit_count += 1
            result[f"{prefix}_width_{width}_hit_count"] = int(hit_count)
            result[f"{prefix}_width_{width}_hit_rate"] = _rate(hit_count, row_count)
    result["best_trivial_width_1_hit_rate"] = max(
        _float(result.get("action_frequency_width_1_hit_rate")),
        _float(result.get("mask_only_width_1_hit_rate")),
    )
    return result


def _per_seed_prediction_metrics(
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    by_seed: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for prediction in predictions:
        by_seed[_int(prediction.get("left_out_seed"))].append(prediction)
    result = {}
    for seed, rows in sorted(by_seed.items()):
        result[str(seed)] = {
            "row_count": len(rows),
            "singleton_hit_rate": _rate(
                sum(_hit(prediction, "singleton_action") for prediction in rows),
                len(rows),
            ),
            "top_2_set_hit_rate": _rate(
                sum(_set_hit(prediction, "top_2_actions") for prediction in rows),
                len(rows),
            ),
            "top_3_set_hit_rate": _rate(
                sum(_set_hit(prediction, "top_3_actions") for prediction in rows),
                len(rows),
            ),
            "unsupported_count": int(
                sum(item.get("unsupported") is True for item in rows)
            ),
            "no_prediction_count": int(
                sum(item.get("no_prediction") is True for item in rows)
            ),
        }
    return result


def _best_local_support_entry(
    entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not entries:
        return {}
    return dict(
        max(
            entries,
            key=lambda entry: (
                _float(entry.get("singleton_hit_rate")),
                _float(entry.get("top_3_set_hit_rate")),
                -_float(entry.get("dominant_action_share")),
                -_float(entry.get("average_selected_neighbor_count")),
                str(entry.get("label")),
            ),
        )
    )


def _best_ranking_entry(
    entries: Sequence[Mapping[str, object]],
    negative_entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    negative_by_k = {
        entry.get("k"): entry
        for entry in negative_entries
    }
    best = dict(
        max(
            entries,
            key=lambda entry: (
                _float(entry.get("pairwise_accuracy")),
                _float(entry.get("top_3_safe_hit_rate")),
                _float(entry.get("top_1_safe_hit_rate")),
                str(entry.get("label")),
            ),
        )
    ) if entries else {}
    negative = _mapping(negative_by_k.get(best.get("k")))
    margin = _float(best.get("pairwise_accuracy")) - _float(
        negative.get("pairwise_accuracy")
    )
    best["shuffled_target_pairwise_accuracy"] = negative.get("pairwise_accuracy")
    best["pairwise_accuracy_margin_over_shuffled_target"] = _round(margin)
    best["min_pairwise_accuracy"] = _round(RANKING_MIN_PAIRWISE_ACCURACY)
    best["min_pairwise_margin_over_shuffled_target"] = _round(
        RANKING_MIN_MARGIN_OVER_SHUFFLED
    )
    best["min_top_3_safe_hit_rate"] = _round(RANKING_MIN_TOP3_SAFE_HIT_RATE)
    best["capacity_floor_passed"] = (
        _float(best.get("pairwise_accuracy")) >= RANKING_MIN_PAIRWISE_ACCURACY
        and margin >= RANKING_MIN_MARGIN_OVER_SHUFFLED
        and _float(best.get("top_3_safe_hit_rate")) >= RANKING_MIN_TOP3_SAFE_HIT_RATE
    )
    return best


def _local_floor_passed(metrics: Mapping[str, object]) -> bool:
    baselines = _mapping(metrics.get("baselines_at_matching_width"))
    margin = _float(metrics.get("singleton_hit_rate")) - _float(
        baselines.get("best_trivial_width_1_hit_rate")
    )
    return (
        margin >= LOCAL_MIN_MARGIN_OVER_TRIVIAL
        and _int(metrics.get("unsupported_count")) == 0
        and _int(metrics.get("no_prediction_count")) == 0
        and _float(metrics.get("dominant_action_share")) <= 0.75
    )


def _baseline_predictions(
    rows: Sequence[Mapping[str, object]],
    *,
    row_index: int,
    train_indexes: Sequence[int],
) -> dict[str, dict[str, list[str]]]:
    public_mask = _complete_action_mask(_mapping(rows[row_index].get("public_action_mask")))
    action_counts: Counter[str] = Counter()
    mask_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    target_signature = _mask_signature(public_mask)
    for train_index in train_indexes:
        safe = _safe_action_set(rows[train_index])
        action_counts.update(safe)
        fallback_counts.update(safe)
        train_mask = _complete_action_mask(_mapping(rows[train_index].get("public_action_mask")))
        if _mask_signature(train_mask) == target_signature:
            mask_counts.update(safe)
    return {
        "action_frequency": {
            str(width): _top_counted_public_actions(action_counts, public_mask, width)
            for width in (1, 2, 3)
        },
        "mask_only": {
            str(width): _top_counted_public_actions(
                mask_counts or fallback_counts,
                public_mask,
                width,
            )
            for width in (1, 2, 3)
        },
    }


def _top_counted_public_actions(
    counts: Counter[str],
    public_mask: Mapping[str, bool],
    width: int,
) -> list[str]:
    candidates = [action for action in ACTION_NAMES if public_mask.get(action)]
    ranked = sorted(
        candidates,
        key=lambda action: (-int(counts.get(action, 0)), _action_order(action)),
    )
    return ranked[:width]


def _decoded_observation_summary_vector(row: Mapping[str, object]) -> list[float]:
    observation = _mapping(
        _mapping(row.get("trainable_public_features")).get("public_observation")
    )
    decoded = []
    if (
        observation.get("storage_encoding") == "zlib_base64_little_endian_int16"
        and isinstance(observation.get("data"), str)
    ):
        decoded = _decode_zlib_int16(str(observation.get("data") or ""))
    if not decoded:
        values = _public_observation_values(observation)
        decoded = [int(round(value * 32768.0)) for value in values]
    if not decoded:
        return [0.0] * 12
    sorted_values = sorted(float(value) for value in decoded)
    total = float(len(sorted_values))
    mean = sum(sorted_values) / total
    variance = sum((value - mean) ** 2 for value in sorted_values) / total
    return [
        len(sorted_values) / 1000.0,
        mean / 32768.0,
        math.sqrt(variance) / 32768.0,
        sorted_values[0] / 32768.0,
        sorted_values[-1] / 32768.0,
        _quantile(sorted_values, 0.25) / 32768.0,
        _quantile(sorted_values, 0.50) / 32768.0,
        _quantile(sorted_values, 0.75) / 32768.0,
        sum(1 for value in sorted_values if value > 0.0) / total,
        sum(1 for value in sorted_values if value == 0.0) / total,
        sum(1 for value in sorted_values if value < 0.0) / total,
        (sorted_values[-1] - sorted_values[0]) / 32768.0,
    ]


def _action_mask_geometry_vector(row: Mapping[str, object]) -> list[float]:
    mask = _complete_action_mask(_mapping(row.get("public_action_mask")))
    actions = [1.0 if mask.get(action) else 0.0 for action in ACTION_NAMES]
    movement = _count_mask_group(mask, ("move_north", "move_south", "move_east", "move_west"))
    resource = _count_mask_group(mask, ("eat", "drink", "mate", "stay"))
    attack = _count_mask_group(mask, ("attack_north", "attack_south", "attack_east", "attack_west"))
    signal = sum(1 for action in ACTION_NAMES if action.startswith("signal_") and mask.get(action))
    width = sum(1 for value in actions if value > 0.0)
    return [
        *actions,
        width / float(len(ACTION_NAMES)),
        movement / 4.0,
        resource / 4.0,
        attack / 4.0,
        signal / 8.0,
    ]


def _decoded_summary_plus_action_mask_geometry_vector(
    row: Mapping[str, object],
) -> list[float]:
    return [
        *_decoded_observation_summary_vector(row),
        *_action_mask_geometry_vector(row),
    ]


def _v173_lifecycle_validation(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    if report.get("diagnostics_only") is not True:
        failures.append({"field": "diagnostics_only", "observed": report.get("diagnostics_only")})
    for field in (
        "training_ran",
        "training_authorized",
        "scorer_retraining_ran",
        "scorer_retraining_authorized",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "runtime_observation_schema_changed",
        "runtime_policy_changed",
        "shadow_eval_ran",
        "live_ab_ran",
        "live_ab_allowed",
        "k_tuning_ran",
        "threshold_tuning_ran",
        "replay_viewer_schema_changed",
        "promotion_authorized",
        "staging_authorized",
        "commit_authorized",
        "reset_authorized",
        "clean_authorized",
    ):
        if field in report and report.get(field) is not False:
            failures.append({"field": field, "observed": report.get(field)})
    return {
        "policy": "m3_carrion_survivor_continuation_v174_v173_lifecycle_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def _source_seed_validation(
    rows: Sequence[Mapping[str, object]],
    *,
    support_provenance_seeds: Sequence[int],
) -> dict[str, object]:
    expected = sorted({int(seed) for seed in support_provenance_seeds})
    counts = Counter(_source_seed(row) for row in rows)
    observed = sorted(seed for seed in counts if seed > 0)
    failures = []
    if observed != expected:
        failures.append("source_seed_set_mismatch")
    for row_index, row in enumerate(rows):
        metadata = _mapping(row.get("metadata"))
        if (
            metadata.get("source_seed_is_support_provenance_not_future_promotion_holdout")
            is not True
        ):
            failures.append(f"row_{row_index}_source_seed_not_support_provenance")
            break
    return {
        "policy": "m3_carrion_survivor_continuation_v174_support_seed_validation_v1",
        "passed": not failures,
        "failures": failures,
        "expected_support_provenance_seeds": expected,
        "observed_source_seeds": observed,
        "row_counts_by_seed": {
            str(seed): int(counts.get(seed, 0)) for seed in expected
        },
    }


def _route_recommendation(planner: Mapping[str, object]) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v174_route_recommendation_v1",
        "recommended_next_route": planner.get("recommended_next_route"),
        "classification": planner.get("classification"),
        "training_authorized": False,
        "runtime_integration_authorized": False,
        "runtime_action_selection_change_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
    }


def _support_provenance_seed_policy(
    *,
    support_provenance_seeds: Sequence[int],
) -> dict[str, object]:
    seeds = sorted({int(seed) for seed in support_provenance_seeds})
    return {
        "policy": "m3_carrion_survivor_continuation_v174_support_provenance_seed_policy_v1",
        "support_provenance_seeds": seeds,
        "support_provenance_seeds_are_future_promotion_heldout": False,
        "future_promotion_heldout_seed_reuse_allowed": False,
        "new_promotion_heldout_broad_seeds_required": True,
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "mechanism_aware_failure_battery": True,
        "lanes": ["A", "B", "C", "D", "E", "F"],
        "training_allowed": False,
        "runtime_artifact_allowed": False,
        "runtime_action_change_allowed": False,
        "shadow_or_live_eval_allowed": False,
        "promotion_allowed": False,
        "gate_relaxation_allowed": False,
        "support_provenance_seed_reuse_as_promotion_heldout_allowed": False,
        "staging_allowed": False,
        "commit_allowed": False,
        "reset_allowed": False,
        "clean_allowed": False,
    }


def _lifecycle_flags() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "artifact_created": False,
        "diagnostic_artifact_created": False,
        "runtime_artifact_created": False,
        "training_ran": False,
        "training_authorized": False,
        "scorer_retraining_ran": False,
        "scorer_retraining_authorized": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": False,
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "k_tuning_ran": False,
        "threshold_tuning_ran": False,
        "replay_expansion_ran": False,
        "world_model_training_ran": False,
        "replay_viewer_schema_changed": False,
        "promotion_authorized": False,
        "staging_authorized": False,
        "commit_authorized": False,
        "reset_authorized": False,
        "clean_authorized": False,
        "non_promoted": True,
    }


def _maybe_write_v175_plan(
    *,
    planner: Mapping[str, object],
    output_path: str | Path | None,
) -> dict[str, object]:
    plan_rows = _list_of_mappings(planner.get("v175_plan_rows"))
    should_write = planner.get("write_optional_v175_plan_jsonl") is True
    if not should_write or output_path is None:
        return {
            "written": False,
            "reason": (
                "planner_did_not_recommend_replay_or_world_model_expansion"
                if not should_write
                else "output_path_disabled"
            ),
            "jsonl_row_count": 0,
        }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in plan_rows) + "\n",
        encoding="utf-8",
    )
    return {
        "written": True,
        "path": str(path),
        "jsonl_row_count": len(plan_rows),
        "plan_digest": stable_payload_digest(plan_rows),
    }


def _v175_action_ranking_plan_rows(
    *,
    lane_c: Mapping[str, object],
) -> list[dict[str, object]]:
    best = _mapping(lane_c.get("best_ranking_entry"))
    return [
        {
            "schema_version": "m3_carrion_survivor_continuation_v175_plan_row_v1",
            "route": "diagnostics_only_action_ranking_fit",
            "reason": "pairwise ranking is above shuffled control but not runtime-ready",
            "best_v174_ranking_label": best.get("label"),
            "pairwise_accuracy": best.get("pairwise_accuracy"),
            "pairwise_accuracy_margin_over_shuffled_target": best.get(
                "pairwise_accuracy_margin_over_shuffled_target"
            ),
            "training_authorized": False,
            "runtime_artifact_authorized": False,
            "shadow_or_live_eval_authorized": False,
            "promotion_authorized": False,
        }
    ]


def _v175_replay_or_world_model_plan_rows(
    *,
    lane_a: Mapping[str, object],
    lane_b: Mapping[str, object],
    lane_c: Mapping[str, object],
    lane_d: Mapping[str, object],
    lane_e: Mapping[str, object],
) -> list[dict[str, object]]:
    best_local = _mapping(lane_b.get("best_frontier_entry"))
    best_ranking = _mapping(lane_c.get("best_ranking_entry"))
    return [
        {
            "schema_version": "m3_carrion_survivor_continuation_v175_plan_row_v1",
            "route": "exact_branch_replay_expansion",
            "priority": 1,
            "reason": (
                "local support and calibrated sets do not produce narrow "
                "safe coverage under static public features"
            ),
            "v174_best_local_label": best_local.get("label"),
            "v174_best_local_singleton_hit_rate": best_local.get("singleton_hit_rate"),
            "training_authorized": False,
            "runtime_artifact_authorized": False,
        },
        {
            "schema_version": "m3_carrion_survivor_continuation_v175_plan_row_v1",
            "route": "world_model_transition_diagnostic",
            "priority": 2,
            "reason": (
                "ranking signal is weak or non-calibrated; diagnose whether "
                "public observation lacks transition-sufficient state"
            ),
            "v174_best_ranking_label": best_ranking.get("label"),
            "v174_best_ranking_pairwise_accuracy": best_ranking.get(
                "pairwise_accuracy"
            ),
            "v174_full_set_share": lane_a.get("full_public_mask_set_share"),
            "v174_conformal_full_mask_failure": lane_d.get(
                "safety_only_holds_by_returning_full_masks"
            ),
            "v174_public_projection_improved": lane_e.get("any_projection_improved"),
            "training_authorized": False,
            "runtime_artifact_authorized": False,
        },
    ]


def _conformal_safety_only_by_full_masks(lane_d: Mapping[str, object]) -> bool:
    return lane_d.get("safety_only_holds_by_returning_full_masks") is True


def _skipped_lane(policy: str) -> dict[str, object]:
    return {
        "policy": f"m3_carrion_survivor_continuation_v174_{policy}_v1",
        "skipped": True,
        "reason": "source_validation_or_row_contract_failed",
        "floor_passed": False,
    }


def _evaluation_indexes(context: Mapping[str, object]) -> list[int]:
    return [_int(index) for index in _list_like(context.get("evaluation_indexes"))]


def _support_seeds(context: Mapping[str, object]) -> list[int]:
    return [_int(seed) for seed in _list_like(context.get("support_seeds"))]


def _row_seeds(context: Mapping[str, object]) -> list[int]:
    return [_int(seed) for seed in _list_like(context.get("row_seeds"))]


def _indexes_by_seed(context: Mapping[str, object]) -> Mapping[int, list[int]]:
    return _mapping(context.get("indexes_by_seed"))  # type: ignore[return-value]


def _distances_for_row(
    context: Mapping[str, object],
    row_index: int,
) -> list[tuple[float, int]]:
    distances = _mapping(context.get("distances_by_row")).get(row_index, [])
    return [
        (float(item[0]), int(item[1]))
        for item in distances
        if isinstance(item, tuple) and len(item) == 2
    ]


def _train_indexes_for_row(
    context: Mapping[str, object],
    row_index: int,
) -> list[int]:
    return [
        _int(index)
        for index in _list_like(
            _mapping(context.get("train_indexes_by_row")).get(row_index)
        )
    ]


def _distances_for_row_against_indexes(
    context: Mapping[str, object],
    *,
    row_index: int,
    indexes: Sequence[int],
) -> list[tuple[float, int]]:
    vectors = _mapping(context).get("vectors")
    if not isinstance(vectors, list):
        return []
    target = vectors[row_index]
    return sorted(
        (_squared_distance(target, vectors[index]), int(index)) for index in indexes
    )


def _selected_distances(
    distances: Sequence[tuple[float, int]],
    *,
    k: int | None = None,
    radius: float | None = None,
) -> list[tuple[float, int]]:
    selected = list(distances)
    if radius is not None:
        selected = [item for item in selected if float(item[0]) <= float(radius)]
    if k is not None:
        selected = selected[: int(k)]
    return selected


def _public_actions_for_row(row: Mapping[str, object]) -> list[str]:
    mask = _complete_action_mask(_mapping(row.get("public_action_mask")))
    return [action for action in ACTION_NAMES if mask.get(action)]


def _positive_ranked_actions(scores: Mapping[str, float]) -> list[str]:
    return sorted(
        [action for action, score in scores.items() if float(score) > 0.0],
        key=lambda action: (-float(scores[action]), _action_order(action)),
    )


def _ranked_actions(scores: Mapping[str, float]) -> list[str]:
    return sorted(
        [action for action in scores],
        key=lambda action: (-float(scores[action]), _action_order(action)),
    )


def _top_two_margin(scores: Mapping[str, float]) -> dict[str, object]:
    ranked = _ranked_actions(scores)
    top_1 = ranked[0] if ranked else None
    top_2 = ranked[1] if len(ranked) > 1 else None
    top_1_score = _round(scores.get(str(top_1), 0.0)) if top_1 else 0.0
    top_2_score = _round(scores.get(str(top_2), 0.0)) if top_2 else 0.0
    return {
        "top_1_action": top_1,
        "top_2_action": top_2,
        "top_1_score": top_1_score,
        "top_2_score": top_2_score,
        "top_1_top_2_margin": _round(top_1_score - top_2_score),
    }


def _score_entropy(scores: Mapping[str, float]) -> float:
    positives = [float(value) for value in scores.values() if float(value) > 0.0]
    total = sum(positives)
    if total <= 0.0 or len(positives) <= 1:
        return 0.0
    return -sum((value / total) * math.log(value / total, 2) for value in positives)


def _row_type(row: Mapping[str, object]) -> str:
    return "unique" if len(_safe_action_set(row)) == 1 else "tied"


def _hit(prediction: Mapping[str, object], key: str) -> bool:
    action = prediction.get(key)
    return isinstance(action, str) and action in set(
        _list_of_strings(prediction.get("safe_action_set"))
    )


def _set_hit(prediction: Mapping[str, object], key: str) -> bool:
    return bool(
        set(_list_of_strings(prediction.get(key)))
        & set(_list_of_strings(prediction.get("safe_action_set")))
    )


def _per_action_failure_payload(
    per_action: Mapping[str, Counter[str]],
) -> dict[str, dict[str, object]]:
    result = {}
    for action in ACTION_NAMES:
        counter = per_action.get(action, Counter())
        count = int(counter.get("safe_occurrence", 0))
        if count <= 0:
            continue
        result[action] = {
            "safe_occurrence_count": count,
            "pairwise_loss_count": int(counter.get("pairwise_loss", 0)),
            "pairwise_tie_count": int(counter.get("pairwise_tie", 0)),
            "top_1_failure_count": int(counter.get("top_1_failure", 0)),
            "top_1_failure_rate": _rate(int(counter.get("top_1_failure", 0)), count),
            "top_3_failure_count": int(counter.get("top_3_failure", 0)),
            "top_3_failure_rate": _rate(int(counter.get("top_3_failure", 0)), count),
        }
    return result


def _conformal_quantile_rank(values: Sequence[int], coverage: float) -> int:
    if not values:
        return len(ACTION_NAMES)
    ordered = sorted(max(1, int(value)) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(float(coverage) * len(ordered)) - 1))
    return int(ordered[index])


def _count_mask_group(mask: Mapping[str, bool], actions: Sequence[str]) -> int:
    return sum(1 for action in actions if mask.get(action))


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = float(q) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _mean(values: Sequence[float] | Sequence[int]) -> float:
    return _round(sum(float(value) for value in values) / len(values)) if values else 0.0


def _list_like(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _list_of_strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _radius_label(radius: float) -> str:
    text = f"{float(radius):.1f}"
    return text.replace(".", "_")
