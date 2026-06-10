from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
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
    _complete_action_mask,
    _json_round_trip_digest,
    _mask_signature,
    _public_feature_vector,
    _rate,
    _safe_action_set,
    _source_seed,
    _squared_distance,
)
from evolution_sim.mind.carrion_survivor_continuation_v174_mechanism_failure_battery import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V174_REPORT_PATH,
    EXPECTED_V173_EXACT_DIGEST,
    M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_SCHEMA_VERSION,
    RANKING_CONFIGS,
    _lifecycle_flags as _v174_lifecycle_flags,
    _normalized_vectors,
    _ranked_actions,
    _selected_distances,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V175_V174_ROUTE_CORRECTION_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v175_v174_route_correction_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V175_V174_ROUTE_CORRECTION_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v175_v174_route_correction_audit_v1"
)
EXPECTED_V174_EXACT_DIGEST = (
    "26b0e68d71b46acd85faeccc3c525d19f096908388ea1ee0dd77a76be1d070bd"
)
EXPECTED_V174_CLASSIFICATION = (
    "v174_action_ranking_capacity_ready_for_v175_diagnostic_fit_no_runtime"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v175-carrion-survivor-continuation-v174-route-correction-audit.json"
)
DEFAULT_V176_PLAN_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v176-carrion-survivor-continuation-replay-or-world-model-plan.jsonl"
)
STRICT_PAIRWISE_ACCURACY_FLOOR = 0.53
STRICT_TOP3_SAFE_HIT_FLOOR = 0.65
STRICT_UNIQUE_PAIRWISE_ACCURACY_FLOOR = 0.53
STRICT_UNIQUE_TOP1_SAFE_HIT_FLOOR = 0.20
STRICT_PAIRWISE_MARGIN_OVER_CLEAN_CONTROL_FLOOR = 0.05


class CarrionSurvivorContinuationV175V174RouteCorrectionAuditError(ValueError):
    pass


def run_carrion_survivor_continuation_v175_v174_route_correction_audit(
    *,
    v174_report_path: str | Path = DEFAULT_V174_REPORT_PATH,
    v173_report_path: str | Path = DEFAULT_V173_REPORT_PATH,
    v172_report_path: str | Path = DEFAULT_V172_REPORT_PATH,
    v172_dataset_path: str | Path = DEFAULT_V172_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    v176_plan_output_path: str | Path | None = DEFAULT_V176_PLAN_OUTPUT_PATH,
    expected_v174_exact_digest: str | None = EXPECTED_V174_EXACT_DIGEST,
    expected_v174_classification: str = EXPECTED_V174_CLASSIFICATION,
    expected_v173_exact_digest: str | None = EXPECTED_V173_EXACT_DIGEST,
    expected_v172_dataset_digest: str | None = EXPECTED_V172_DATASET_DIGEST,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> dict[str, object]:
    v174_report = load_json_report(v174_report_path)
    v173_report = load_json_report(v173_report_path)
    v172_report = load_json_report(v172_report_path)
    rows = load_jsonl_dataset(v172_dataset_path)
    source_validation = validate_v175_sources(
        v174_report=v174_report,
        v173_report=v173_report,
        v172_report=v172_report,
        rows=rows,
        expected_v174_exact_digest=expected_v174_exact_digest,
        expected_v174_classification=expected_v174_classification,
        expected_v173_exact_digest=expected_v173_exact_digest,
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
        context = _ranking_context(
            rows,
            support_provenance_seeds=support_provenance_seeds,
        )
        audited_label = _audited_v174_ranking_label(v174_report)
        lane_a = lane_a_per_seed_lane_c_recompute(
            rows,
            context=context,
            audited_label=audited_label,
        )
        lane_b = lane_b_clean_negative_controls(
            rows,
            context=context,
            audited_label=audited_label,
        )
        lane_c = lane_c_tied_row_inflation_audit(lane_a=lane_a)
    else:
        lane_a = _skipped_lane("lane_a_per_seed_lane_c_recompute")
        lane_b = _skipped_lane("lane_b_clean_negative_controls")
        lane_c = _skipped_lane("lane_c_tied_row_inflation_audit")
    lane_d = lane_d_route_correction(
        source_validation=source_validation,
        leakage_scan=leakage_scan,
        row_schema_validation=row_schema_validation,
        lane_a=lane_a,
        lane_b=lane_b,
        lane_c=lane_c,
    )
    classification = str(lane_d.get("classification"))
    plan_output = _maybe_write_v176_plan(
        lane_d=lane_d,
        output_path=v176_plan_output_path,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V175_V174_ROUTE_CORRECTION_AUDIT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V175_V174_ROUTE_CORRECTION_AUDIT_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v174_report": str(v174_report_path),
            "v173_report": str(v173_report_path),
            "v172_report": str(v172_report_path),
            "v172_dataset": str(v172_dataset_path),
            "expected_v174_exact_digest": expected_v174_exact_digest,
            "expected_v174_classification": expected_v174_classification,
            "expected_v173_exact_digest": expected_v173_exact_digest,
            "expected_v172_dataset_digest": expected_v172_dataset_digest,
            "support_provenance_seeds": [
                int(seed) for seed in support_provenance_seeds
            ],
            "ranking_configs": [
                {"label": str(label), "k": k} for label, k in RANKING_CONFIGS
            ],
        },
        "source_validation": source_validation,
        "leakage_scan": leakage_scan,
        "row_schema_validation": row_schema_validation,
        "lanes": {
            "lane_a_per_seed_lane_c_recompute": lane_a,
            "lane_b_clean_negative_controls": lane_b,
            "lane_c_tied_row_inflation_audit": lane_c,
            "lane_d_route_correction": lane_d,
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(lane_d),
        "v176_plan_output": plan_output,
        "dataset_digest": stable_payload_digest(rows),
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v175_sources(
    *,
    v174_report: Mapping[str, object],
    v173_report: Mapping[str, object],
    v172_report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v174_exact_digest: str | None,
    expected_v174_classification: str,
    expected_v173_exact_digest: str | None,
    expected_v172_dataset_digest: str | None,
    support_provenance_seeds: Sequence[int],
) -> dict[str, object]:
    failures: list[str] = []
    observed_v174_classification = str(
        _mapping(v174_report.get("classification")).get("primary") or ""
    )
    v174_exact = exact_digest_validation_report(v174_report)
    observed_v174_exact = str(v174_report.get("exact_digest") or "")
    if (
        v174_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_SCHEMA_VERSION
    ):
        failures.append("v174_schema_version_mismatch")
    if (
        v174_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_POLICY
    ):
        failures.append("v174_policy_mismatch")
    if observed_v174_classification != expected_v174_classification:
        failures.append("v174_unexpected_classification")
    if v174_exact.get("passed") is not True:
        failures.append("v174_exact_digest_mismatch")
    if expected_v174_exact_digest and observed_v174_exact != expected_v174_exact_digest:
        failures.append("v174_unexpected_exact_digest")
    v173_exact = exact_digest_validation_report(v173_report)
    observed_v173_exact = str(v173_report.get("exact_digest") or "")
    if v173_exact.get("passed") is not True:
        failures.append("v173_exact_digest_mismatch")
    if expected_v173_exact_digest and observed_v173_exact != expected_v173_exact_digest:
        failures.append("v173_unexpected_exact_digest")
    dataset_digest = stable_payload_digest(rows)
    if expected_v172_dataset_digest and dataset_digest != expected_v172_dataset_digest:
        failures.append("v172_unexpected_dataset_digest")
    v174_reported_dataset_digest = str(v174_report.get("dataset_digest") or "")
    if v174_reported_dataset_digest != dataset_digest:
        failures.append("v174_dataset_digest_mismatch")
    v172_reported_dataset_digest = str(
        _mapping(v172_report.get("dataset")).get("dataset_digest") or ""
    )
    if v172_reported_dataset_digest and v172_reported_dataset_digest != dataset_digest:
        failures.append("v172_reported_dataset_digest_mismatch")
    v174_source = _mapping(v174_report.get("source_validation"))
    if v174_source:
        if v174_source.get("observed_v173_exact_digest") != observed_v173_exact:
            failures.append("v174_source_v173_digest_mismatch")
        if v174_source.get("observed_v172_dataset_digest") != dataset_digest:
            failures.append("v174_source_v172_dataset_digest_mismatch")
    v174_lifecycle = _lifecycle_validation(
        v174_report,
        policy="m3_carrion_survivor_continuation_v175_v174_lifecycle_validation_v1",
    )
    if v174_lifecycle.get("passed") is not True:
        failures.append("v174_lifecycle_not_diagnostics_only")
    v173_lifecycle = _lifecycle_validation(
        v173_report,
        policy="m3_carrion_survivor_continuation_v175_v173_lifecycle_validation_v1",
    )
    if v173_lifecycle.get("passed") is not True:
        failures.append("v173_lifecycle_not_diagnostics_only")
    source_seed_validation = _source_seed_validation(
        rows,
        support_provenance_seeds=support_provenance_seeds,
    )
    if source_seed_validation.get("passed") is not True:
        failures.append("v172_support_provenance_seed_rows_invalid")
    return {
        "policy": "m3_carrion_survivor_continuation_v175_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v174_exact_digest": expected_v174_exact_digest,
        "observed_v174_exact_digest": observed_v174_exact,
        "v174_exact_digest_validation": v174_exact,
        "expected_v174_classification": expected_v174_classification,
        "observed_v174_classification": observed_v174_classification,
        "expected_v173_exact_digest": expected_v173_exact_digest,
        "observed_v173_exact_digest": observed_v173_exact,
        "v173_exact_digest_validation": v173_exact,
        "expected_v172_dataset_digest": expected_v172_dataset_digest,
        "observed_v172_dataset_digest": dataset_digest,
        "v174_reported_dataset_digest": v174_reported_dataset_digest,
        "v172_reported_dataset_digest": v172_reported_dataset_digest,
        "v174_lifecycle_validation": v174_lifecycle,
        "v173_lifecycle_validation": v173_lifecycle,
        "support_provenance_seed_validation": source_seed_validation,
    }


def lane_a_per_seed_lane_c_recompute(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    audited_label: str,
) -> dict[str, object]:
    entries = []
    for label, k in RANKING_CONFIGS:
        entry = _ranking_entry(
            rows,
            context=context,
            label=str(label),
            k=k,
            control_maps_by_seed=None,
        )
        entries.append(entry)
    audited = _entry_by_label(entries, audited_label) or _best_aggregate_entry(entries)
    per_seed = _mapping(audited.get("per_seed"))
    failed_seeds = [
        int(seed)
        for seed, payload in per_seed.items()
        if _seed_floor_passed(_mapping(payload)) is not True
    ]
    return {
        "policy": "m3_carrion_survivor_continuation_v175_lane_a_per_seed_lane_c_recompute_v1",
        "audited_v174_ranking_label": audited.get("label"),
        "ranking_entries": entries,
        "audited_entry": audited,
        "per_seed_floor_policy": _per_seed_floor_policy(),
        "failed_support_seeds": failed_seeds,
        "any_support_seed_missed_floor": bool(failed_seeds),
        "seed_41": _mapping(per_seed.get("41")),
        "floor_passed": not failed_seeds,
    }


def lane_b_clean_negative_controls(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    audited_label: str,
) -> dict[str, object]:
    audited_k = _ranking_k_by_label(audited_label)
    candidate = _ranking_entry(
        rows,
        context=context,
        label=audited_label,
        k=audited_k,
        control_maps_by_seed=None,
    )
    controls = []
    for control_name in (
        "seed_stratified_target_permutation",
        "action_mask_preserving_target_permutation",
        "global_label_permutation",
    ):
        maps = _control_maps_by_seed(rows, context=context, control_name=control_name)
        controls.append(
            {
                "control_name": control_name,
                "heldout_label_source_validation": _control_source_validation(
                    context,
                    maps,
                ),
                "metrics": _ranking_entry(
                    rows,
                    context=context,
                    label=f"{audited_label}_{control_name}",
                    k=audited_k,
                    control_maps_by_seed=maps,
                ),
            }
        )
    per_seed_margins = {}
    clean_control_failures = []
    candidate_per_seed = _mapping(candidate.get("per_seed"))
    for seed, candidate_metrics in candidate_per_seed.items():
        candidate_pairwise = _float(_mapping(candidate_metrics).get("pairwise_accuracy"))
        control_scores = [
            _float(
                _mapping(
                    _mapping(_mapping(control.get("metrics")).get("per_seed")).get(seed)
                ).get("pairwise_accuracy")
            )
            for control in controls
        ]
        best_control = max(control_scores, default=0.0)
        margin = candidate_pairwise - best_control
        passed = margin >= STRICT_PAIRWISE_MARGIN_OVER_CLEAN_CONTROL_FLOOR
        per_seed_margins[str(seed)] = {
            "candidate_pairwise_accuracy": _round(candidate_pairwise),
            "best_clean_control_pairwise_accuracy": _round(best_control),
            "pairwise_margin_over_best_clean_control": _round(margin),
            "min_margin_required": _round(
                STRICT_PAIRWISE_MARGIN_OVER_CLEAN_CONTROL_FLOOR
            ),
            "passed": passed,
        }
        if not passed:
            clean_control_failures.append(int(seed))
    validation_passed = all(
        _mapping(control.get("heldout_label_source_validation")).get("passed") is True
        for control in controls
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v175_lane_b_clean_negative_controls_v1",
        "audited_v174_ranking_label": audited_label,
        "candidate_metrics": candidate,
        "controls": controls,
        "per_seed_margins_over_best_clean_control": per_seed_margins,
        "clean_control_failed_support_seeds": clean_control_failures,
        "clean_controls_never_pull_from_heldout_seed": validation_passed,
        "clean_controls_erase_margin": bool(clean_control_failures),
        "floor_passed": validation_passed and not clean_control_failures,
    }


def lane_c_tied_row_inflation_audit(
    *,
    lane_a: Mapping[str, object],
) -> dict[str, object]:
    audited = _mapping(lane_a.get("audited_entry"))
    aggregate = _mapping(audited.get("aggregate"))
    unique = _mapping(aggregate.get("unique_rows"))
    tied = _mapping(aggregate.get("tied_rows"))
    unique_top1 = _float(unique.get("top_1_safe_hit_rate"))
    tied_top1 = _float(tied.get("top_1_safe_hit_rate"))
    unique_pairwise = _float(unique.get("pairwise_accuracy"))
    aggregate_pairwise = _float(aggregate.get("pairwise_accuracy"))
    inflation = tied_top1 - unique_top1
    broad_tied_sets_hide_unique_weakness = (
        unique_pairwise < STRICT_UNIQUE_PAIRWISE_ACCURACY_FLOOR
        or unique_top1 < STRICT_UNIQUE_TOP1_SAFE_HIT_FLOOR
    ) and inflation > 0.25
    return {
        "policy": "m3_carrion_survivor_continuation_v175_lane_c_tied_row_inflation_audit_v1",
        "audited_v174_ranking_label": audited.get("label"),
        "aggregate_pairwise_accuracy": aggregate_pairwise,
        "unique_rows": unique,
        "tied_rows": tied,
        "tied_minus_unique_top_1_safe_hit_rate": _round(inflation),
        "broad_tied_sets_hide_unique_row_weakness": broad_tied_sets_hide_unique_weakness,
        "floor_passed": not broad_tied_sets_hide_unique_weakness,
    }


def lane_d_route_correction(
    *,
    source_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    lane_a: Mapping[str, object],
    lane_b: Mapping[str, object],
    lane_c: Mapping[str, object],
) -> dict[str, object]:
    if (
        source_validation.get("passed") is not True
        or leakage_scan.get("passed") is not True
        or row_schema_validation.get("passed") is not True
    ):
        classification = "v175_invalid_closed_no_shadow"
        recommendation = "close_invalid_without_shadow"
    elif lane_a.get("floor_passed") is not True or lane_c.get("floor_passed") is not True:
        classification = "v175_v174_route_overstated_static_ranking_not_ready"
        recommendation = "v176_exact_branch_replay_or_world_model_transition_diagnostic"
    elif lane_b.get("floor_passed") is not True:
        classification = "v175_v174_negative_control_failure_static_ranking_closed"
        recommendation = "v176_exact_branch_replay_or_world_model_transition_diagnostic"
    else:
        classification = "v175_action_ranking_fit_still_worth_diagnostics_only"
        recommendation = "v176_action_ranking_diagnostics_only_fit"
    plan_rows = (
        _v176_plan_rows(lane_a=lane_a, lane_b=lane_b, lane_c=lane_c)
        if recommendation
        == "v176_exact_branch_replay_or_world_model_transition_diagnostic"
        else []
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v175_lane_d_route_correction_v1",
        "classification": classification,
        "recommended_next_route": recommendation,
        "training_authorized": False,
        "runtime_artifact_authorized": False,
        "runtime_action_selection_change_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
        "replay_expansion_ran": False,
        "world_model_training_ran": False,
        "v176_plan_rows": plan_rows,
        "write_optional_v176_plan_jsonl": bool(plan_rows),
    }


def _ranking_entry(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    label: str,
    k: int | None,
    control_maps_by_seed: Mapping[int, Mapping[int, Mapping[str, object]]] | None,
) -> dict[str, object]:
    per_seed_metrics = {}
    aggregate_accumulator = _new_metric_accumulator()
    for seed in _support_seeds(context):
        seed_accumulator = _new_metric_accumulator()
        for row_index in _indexes_by_seed(context).get(seed, []):
            distances = _selected_distances(_distances_for_row(context, row_index), k=k)
            control_map = (
                _mapping(control_maps_by_seed.get(seed))
                if control_maps_by_seed is not None
                else None
            )
            scores = _ranking_scores(
                rows,
                row_index=row_index,
                distances=distances,
                control_map=control_map,
            )
            _record_ranking_row(
                seed_accumulator,
                rows=rows,
                row_index=row_index,
                scores=scores,
            )
            _record_ranking_row(
                aggregate_accumulator,
                rows=rows,
                row_index=row_index,
                scores=scores,
            )
        per_seed_metrics[str(seed)] = _finalize_metrics(seed_accumulator)
        per_seed_metrics[str(seed)]["floor_passed"] = _seed_floor_passed(
            per_seed_metrics[str(seed)]
        )
    aggregate = _finalize_metrics(aggregate_accumulator)
    return {
        "label": label,
        "k": k,
        "aggregate": aggregate,
        "per_seed": per_seed_metrics,
    }


def _ranking_scores(
    rows: Sequence[Mapping[str, object]],
    *,
    row_index: int,
    distances: Sequence[tuple[float, int]],
    control_map: Mapping[int, Mapping[str, object]] | None,
) -> dict[str, float]:
    public_mask = _complete_action_mask(_mapping(rows[row_index].get("public_action_mask")))
    numerators = {action: 0.0 for action in ACTION_NAMES if public_mask.get(action)}
    denominators = {action: 0.0 for action in numerators}
    for distance, train_index in distances:
        control = _mapping(control_map.get(train_index)) if control_map else {}
        source_safe = (
            set(_list_of_strings(control.get("safe_action_set")))
            if control
            else set(_safe_action_set(rows[train_index]))
        )
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


def _record_ranking_row(
    accumulator: dict[str, object],
    *,
    rows: Sequence[Mapping[str, object]],
    row_index: int,
    scores: Mapping[str, float],
) -> None:
    ranked = _ranked_actions(scores)
    safe = set(_safe_action_set(rows[row_index]))
    unsafe = [action for action in scores if action not in safe]
    row_type = "unique" if len(safe) == 1 else "tied"
    _accumulator_update(accumulator, row_type, safe=safe, ranked=ranked, scores=scores, unsafe=unsafe)
    for safe_action in sorted(safe, key=_action_order):
        if safe_action not in scores:
            continue
        action_failures = _mapping(accumulator.get("action_failures"))
        counter = action_failures.setdefault(safe_action, Counter())
        if isinstance(counter, Counter):
            counter.update(["safe_occurrence"])
            if safe_action not in ranked[:1]:
                counter.update(["top_1_failure"])
            if safe_action not in ranked[:3]:
                counter.update(["top_3_failure"])
            for unsafe_action in unsafe:
                if scores[safe_action] < scores[unsafe_action]:
                    counter.update(["pairwise_loss"])
                elif scores[safe_action] == scores[unsafe_action]:
                    counter.update(["pairwise_tie"])


def _accumulator_update(
    accumulator: dict[str, object],
    row_type: str,
    *,
    safe: set[str],
    ranked: Sequence[str],
    scores: Mapping[str, float],
    unsafe: Sequence[str],
) -> None:
    for key in ("all", row_type):
        bucket = _mapping(accumulator.get(key))
        bucket["row_count"] = _int(bucket.get("row_count")) + 1
        for width in (1, 2, 3):
            if set(ranked[:width]) & safe:
                bucket[f"top_{width}_safe_hit_count"] = (
                    _int(bucket.get(f"top_{width}_safe_hit_count")) + 1
                )
        for safe_action in safe:
            if safe_action not in scores:
                continue
            for unsafe_action in unsafe:
                bucket["pairwise_total"] = _int(bucket.get("pairwise_total")) + 1
                if scores[safe_action] > scores[unsafe_action]:
                    bucket["pairwise_correct"] = _float(
                        bucket.get("pairwise_correct")
                    ) + 1.0
                elif scores[safe_action] == scores[unsafe_action]:
                    bucket["pairwise_correct"] = _float(
                        bucket.get("pairwise_correct")
                    ) + 0.5
                    bucket["pairwise_tie_count"] = (
                        _int(bucket.get("pairwise_tie_count")) + 1
                    )


def _finalize_metrics(accumulator: Mapping[str, object]) -> dict[str, object]:
    all_rows = _finalize_bucket(_mapping(accumulator.get("all")))
    unique_rows = _finalize_bucket(_mapping(accumulator.get("unique")))
    tied_rows = _finalize_bucket(_mapping(accumulator.get("tied")))
    all_rows["unique_rows"] = unique_rows
    all_rows["tied_rows"] = tied_rows
    all_rows["action_failures"] = _action_failure_payload(
        _mapping(accumulator.get("action_failures"))
    )
    return all_rows


def _finalize_bucket(bucket: Mapping[str, object]) -> dict[str, object]:
    row_count = _int(bucket.get("row_count"))
    pairwise_total = _int(bucket.get("pairwise_total"))
    pairwise_correct = _float(bucket.get("pairwise_correct"))
    return {
        "row_count": row_count,
        "pairwise_comparison_count": pairwise_total,
        "pairwise_accuracy": _round(pairwise_correct / pairwise_total)
        if pairwise_total
        else 0.0,
        "pairwise_tie_count": _int(bucket.get("pairwise_tie_count")),
        "top_1_safe_hit_count": _int(bucket.get("top_1_safe_hit_count")),
        "top_1_safe_hit_rate": _rate(_int(bucket.get("top_1_safe_hit_count")), row_count),
        "top_2_safe_hit_count": _int(bucket.get("top_2_safe_hit_count")),
        "top_2_safe_hit_rate": _rate(_int(bucket.get("top_2_safe_hit_count")), row_count),
        "top_3_safe_hit_count": _int(bucket.get("top_3_safe_hit_count")),
        "top_3_safe_hit_rate": _rate(_int(bucket.get("top_3_safe_hit_count")), row_count),
    }


def _new_metric_accumulator() -> dict[str, object]:
    return {
        "all": defaultdict(float),
        "unique": defaultdict(float),
        "tied": defaultdict(float),
        "action_failures": {},
    }


def _seed_floor_passed(metrics: Mapping[str, object]) -> bool:
    unique = _mapping(metrics.get("unique_rows"))
    return (
        _float(metrics.get("pairwise_accuracy")) >= STRICT_PAIRWISE_ACCURACY_FLOOR
        and _float(metrics.get("top_3_safe_hit_rate")) >= STRICT_TOP3_SAFE_HIT_FLOOR
        and _float(unique.get("pairwise_accuracy"))
        >= STRICT_UNIQUE_PAIRWISE_ACCURACY_FLOOR
        and _float(unique.get("top_1_safe_hit_rate"))
        >= STRICT_UNIQUE_TOP1_SAFE_HIT_FLOOR
    )


def _per_seed_floor_policy() -> dict[str, object]:
    return {
        "min_pairwise_accuracy": _round(STRICT_PAIRWISE_ACCURACY_FLOOR),
        "min_top_3_safe_hit_rate": _round(STRICT_TOP3_SAFE_HIT_FLOOR),
        "min_unique_row_pairwise_accuracy": _round(
            STRICT_UNIQUE_PAIRWISE_ACCURACY_FLOOR
        ),
        "min_unique_row_top_1_safe_hit_rate": _round(
            STRICT_UNIQUE_TOP1_SAFE_HIT_FLOOR
        ),
        "all_support_seeds_must_pass": True,
    }


def _control_maps_by_seed(
    rows: Sequence[Mapping[str, object]],
    *,
    context: Mapping[str, object],
    control_name: str,
) -> dict[int, dict[int, dict[str, object]]]:
    result = {}
    row_seeds = _row_seeds(context)
    for heldout_seed in _support_seeds(context):
        train_indexes = [
            index for index, seed in enumerate(row_seeds) if int(seed) != heldout_seed
        ]
        if control_name == "seed_stratified_target_permutation":
            groups = _groups_by_key(train_indexes, key_fn=lambda index: str(row_seeds[index]))
        elif control_name == "action_mask_preserving_target_permutation":
            groups = _groups_by_key(
                train_indexes,
                key_fn=lambda index: "|".join(
                    _mask_signature(
                        _complete_action_mask(
                            _mapping(rows[index].get("public_action_mask"))
                        )
                    )
                ),
            )
        else:
            groups = {"global": list(train_indexes)}
        mapping: dict[int, dict[str, object]] = {}
        for group_key, indexes in sorted(groups.items()):
            ordered = sorted(
                indexes,
                key=lambda index: stable_payload_digest(
                    {
                        "control": control_name,
                        "heldout_seed": heldout_seed,
                        "group_key": group_key,
                        "index": index,
                    }
                ),
            )
            for offset, train_index in enumerate(ordered):
                source_index = ordered[(offset + 1) % len(ordered)]
                mapping[train_index] = {
                    "source_index": source_index,
                    "source_seed": row_seeds[source_index],
                    "safe_action_set": _safe_action_set(rows[source_index]),
                }
        result[heldout_seed] = mapping
    return result


def _control_source_validation(
    context: Mapping[str, object],
    maps_by_seed: Mapping[int, Mapping[int, Mapping[str, object]]],
) -> dict[str, object]:
    failures = []
    for heldout_seed, mapping in sorted(maps_by_seed.items()):
        for train_index, payload in sorted(mapping.items()):
            if _int(_mapping(payload).get("source_seed")) == int(heldout_seed):
                failures.append(
                    {
                        "heldout_seed": int(heldout_seed),
                        "train_index": int(train_index),
                        "source_index": _mapping(payload).get("source_index"),
                    }
                )
                break
    return {
        "policy": "m3_carrion_survivor_continuation_v175_control_source_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:32],
        "support_seeds": _support_seeds(context),
    }


def _groups_by_key(
    indexes: Sequence[int],
    *,
    key_fn: object,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indexes:
        groups[str(key_fn(index))].append(int(index))  # type: ignore[operator]
    return groups


def _ranking_context(
    rows: Sequence[Mapping[str, object]],
    *,
    support_provenance_seeds: Sequence[int],
) -> dict[str, object]:
    row_seeds = [_source_seed(row) for row in rows]
    indexes_by_seed: dict[int, list[int]] = defaultdict(list)
    for index, seed in enumerate(row_seeds):
        indexes_by_seed[int(seed)].append(index)
    vectors = _normalized_vectors(rows, _public_feature_vector)
    support_seeds = sorted({int(seed) for seed in support_provenance_seeds})
    distances_by_row: dict[int, list[tuple[float, int]]] = {}
    for seed in support_seeds:
        train_indexes = [
            index for index, row_seed in enumerate(row_seeds) if int(row_seed) != seed
        ]
        for row_index in indexes_by_seed.get(seed, []):
            distances_by_row[row_index] = sorted(
                (
                    _squared_distance(vectors[row_index], vectors[train_index]),
                    train_index,
                )
                for train_index in train_indexes
            )
    return {
        "support_seeds": support_seeds,
        "row_seeds": row_seeds,
        "indexes_by_seed": dict(indexes_by_seed),
        "distances_by_row": distances_by_row,
    }


def _audited_v174_ranking_label(v174_report: Mapping[str, object]) -> str:
    best = _mapping(
        _mapping(
            _mapping(v174_report.get("lanes")).get(
                "lane_c_action_ranking_capacity_probe"
            )
        ).get("best_ranking_entry")
    )
    return str(best.get("label") or "all_train")


def _ranking_k_by_label(label: str) -> int | None:
    for current_label, k in RANKING_CONFIGS:
        if str(current_label) == str(label):
            return k
    return None


def _entry_by_label(
    entries: Sequence[Mapping[str, object]],
    label: str,
) -> dict[str, object] | None:
    for entry in entries:
        if str(entry.get("label")) == str(label):
            return dict(entry)
    return None


def _best_aggregate_entry(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not entries:
        return {}
    return dict(
        max(
            entries,
            key=lambda entry: (
                _float(_mapping(entry.get("aggregate")).get("pairwise_accuracy")),
                _float(_mapping(entry.get("aggregate")).get("top_3_safe_hit_rate")),
                str(entry.get("label")),
            ),
        )
    )


def _action_failure_payload(
    action_failures: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    result = {}
    for action in ACTION_NAMES:
        counter = action_failures.get(action)
        if not isinstance(counter, Counter):
            continue
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


def _v176_plan_rows(
    *,
    lane_a: Mapping[str, object],
    lane_b: Mapping[str, object],
    lane_c: Mapping[str, object],
) -> list[dict[str, object]]:
    audited = _mapping(lane_a.get("audited_entry"))
    return [
        {
            "schema_version": "m3_carrion_survivor_continuation_v176_plan_row_v1",
            "route": "exact_branch_replay_expansion",
            "priority": 1,
            "reason": (
                "v174 action-ranking route is not per-seed defensible under "
                "strict support-seed floors"
            ),
            "failed_support_seeds": lane_a.get("failed_support_seeds"),
            "audited_v174_ranking_label": audited.get("label"),
            "training_authorized": False,
            "runtime_artifact_authorized": False,
        },
        {
            "schema_version": "m3_carrion_survivor_continuation_v176_plan_row_v1",
            "route": "world_model_transition_diagnostic",
            "priority": 2,
            "reason": (
                "static public single-observation ranking may lack transition "
                "state needed to distinguish safe actions"
            ),
            "clean_control_failed_support_seeds": lane_b.get(
                "clean_control_failed_support_seeds"
            ),
            "tied_row_inflation_detected": lane_c.get(
                "broad_tied_sets_hide_unique_row_weakness"
            ),
            "training_authorized": False,
            "runtime_artifact_authorized": False,
        },
    ]


def _maybe_write_v176_plan(
    *,
    lane_d: Mapping[str, object],
    output_path: str | Path | None,
) -> dict[str, object]:
    plan_rows = _list_of_mappings(lane_d.get("v176_plan_rows"))
    should_write = lane_d.get("write_optional_v176_plan_jsonl") is True
    if not should_write or output_path is None:
        return {
            "written": False,
            "reason": (
                "planner_did_not_recommend_replay_or_world_model_transition_diagnostic"
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


def _route_recommendation(lane_d: Mapping[str, object]) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v175_route_recommendation_v1",
        "recommended_next_route": lane_d.get("recommended_next_route"),
        "classification": lane_d.get("classification"),
        "training_authorized": False,
        "runtime_integration_authorized": False,
        "runtime_action_selection_change_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "v174_route_correction_audit": True,
        "training_allowed": False,
        "fit_allowed": False,
        "runtime_artifact_allowed": False,
        "runtime_action_change_allowed": False,
        "shadow_or_live_eval_allowed": False,
        "promotion_allowed": False,
        "gate_relaxation_allowed": False,
        "staging_allowed": False,
        "commit_allowed": False,
        "reset_allowed": False,
        "clean_allowed": False,
    }


def _lifecycle_flags() -> dict[str, object]:
    flags = dict(_v174_lifecycle_flags())
    flags.update(
        {
            "fit_ran": False,
            "replay_expansion_ran": False,
            "world_model_training_ran": False,
        }
    )
    return flags


def _lifecycle_validation(
    report: Mapping[str, object],
    *,
    policy: str,
) -> dict[str, object]:
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
        "replay_expansion_ran",
        "world_model_training_ran",
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
        "policy": policy,
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
        "policy": "m3_carrion_survivor_continuation_v175_support_seed_validation_v1",
        "passed": not failures,
        "failures": failures,
        "expected_support_provenance_seeds": expected,
        "observed_source_seeds": observed,
        "row_counts_by_seed": {
            str(seed): int(counts.get(seed, 0)) for seed in expected
        },
    }


def _skipped_lane(policy: str) -> dict[str, object]:
    return {
        "policy": f"m3_carrion_survivor_continuation_v175_{policy}_v1",
        "skipped": True,
        "reason": "source_validation_or_row_contract_failed",
        "floor_passed": False,
    }


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


def _list_like(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _list_of_strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
