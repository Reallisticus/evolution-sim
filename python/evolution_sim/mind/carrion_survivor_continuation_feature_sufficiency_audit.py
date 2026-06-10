from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import math
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import decode_observation_input
from evolution_sim.mind.candidate_campaign import (
    _dominant_count_share,
    _float,
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    DEFAULT_DATASET_OUTPUT_PATH as DEFAULT_V154_DATASET_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V154_REPORT_PATH,
    TARGET_CARRION_SEEDS,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V155_REPORT_PATH,
    EXPECTED_V154_SUPPORT_READY_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION,
    load_json_report,
    load_v154_dataset,
    validate_v154_train_eval_inputs,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_planner_distilled import planner_distilled_runtime_row

M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_public_feature_sufficiency_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_public_feature_sufficiency_audit_v1"
)
EXPECTED_V155_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_train_eval_"
    "pretraining_alias_prior_blocked_closed_no_training"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v156-carrion-survivor-continuation-public-feature-sufficiency-audit.json"
)
DEFAULT_MIN_NN_OVER_TRIVIAL_MARGIN = 0.05
DEFAULT_MATERIAL_CONFLICT_ROW_REDUCTION = 0.50
FORBIDDEN_FEATURE_KEY_TOKENS = (
    "seed",
    "fixture",
    "branch",
    "tick",
    "agent",
    "path",
    "digest",
    "provenance",
    "private",
    "world",
    "future",
)
FORBIDDEN_FEATURE_STRING_MARKERS = (
    "carrion_only",
    "m3-carrion-specific-archive",
    "branch-",
    "seed-",
)


class CarrionSurvivorContinuationFeatureSufficiencyError(ValueError):
    pass


def run_carrion_survivor_continuation_feature_sufficiency_audit(
    *,
    v154_report_path: str | Path = DEFAULT_V154_REPORT_PATH,
    v154_dataset_path: str | Path = DEFAULT_V154_DATASET_PATH,
    v155_report_path: str | Path = DEFAULT_V155_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    min_nn_over_trivial_margin: float = DEFAULT_MIN_NN_OVER_TRIVIAL_MARGIN,
    material_conflict_row_reduction: float = DEFAULT_MATERIAL_CONFLICT_ROW_REDUCTION,
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    v154_report = load_json_report(v154_report_path)
    v155_report = load_json_report(v155_report_path)
    dataset_rows = load_v154_dataset(v154_dataset_path)
    audit_rows = attach_public_recent_transition_context(
        dataset_rows,
        branch_results=_list_of_mappings(v154_report.get("continuation_branch_results")),
    )
    source_validation = validate_sources(
        v154_report=v154_report,
        v155_report=v155_report,
        dataset_rows=dataset_rows,
        target_seeds=target_seeds,
    )
    feature_reports: list[dict[str, object]] = []
    candidate_policies = candidate_feature_policies()
    if source_validation.get("passed") is True:
        baseline_conflicting_row_count: int | None = None
        for policy in candidate_policies:
            feature_report = evaluate_feature_policy(
                rows=audit_rows,
                policy=policy,
                target_seeds=target_seeds,
                min_nn_over_trivial_margin=min_nn_over_trivial_margin,
                material_conflict_row_reduction=material_conflict_row_reduction,
                baseline_conflicting_row_count=baseline_conflicting_row_count,
            )
            if baseline_conflicting_row_count is None:
                baseline_conflicting_row_count = _int(
                    feature_report.get("conflicting_row_count")
                )
            feature_reports.append(feature_report)
    support_ready = [
        report
        for report in feature_reports
        if report.get("support_ready_no_training") is True
    ]
    leakage_failures = [
        report
        for report in feature_reports
        if _mapping(report.get("leakage_scan")).get("passed") is not True
    ]
    if source_validation.get("passed") is not True:
        primary = (
            "m3_carrion_survivor_continuation_public_feature_sufficiency_"
            "source_invalid_closed_no_training"
        )
        recommendation = "repair_source_integrity_before_public_feature_audit"
    elif leakage_failures:
        primary = (
            "m3_carrion_survivor_continuation_public_feature_sufficiency_"
            "leakage_failed_closed_no_training"
        )
        recommendation = "do_not_train_forbidden_public_feature_surface"
    elif support_ready:
        primary = (
            "m3_carrion_survivor_continuation_public_feature_sufficiency_"
            "support_ready_no_training"
        )
        recommendation = (
            "review_public_feature_surface_before_any_separate_opt_in_training"
        )
    else:
        primary = (
            "m3_carrion_survivor_continuation_public_feature_sufficiency_"
            "public_feature_surface_insufficient_no_training"
        )
        recommendation = (
            "return_to_archive_generation_with_more_distinct_public_predecision_context"
        )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_POLICY,
        "contract": {
            "diagnostics_only": True,
            "artifact_creation_allowed": False,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
            "shadow_live_ab_allowed": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_seed_fixture_branch_path_digest_provenance_as_trainable_input": False,
            "uses_future_outcome_as_trainable_input": False,
        },
        "inputs": {
            "v154_report": str(v154_report_path),
            "v154_dataset": str(v154_dataset_path),
            "v155_report": str(v155_report_path),
            "target_seeds": [int(seed) for seed in target_seeds],
            "min_nn_over_trivial_margin": _round(min_nn_over_trivial_margin),
            "material_conflict_row_reduction": _round(
                material_conflict_row_reduction
            ),
        },
        "source_validation": source_validation,
        "feature_policy_count": len(feature_reports),
        "feature_policies": feature_reports,
        "best_feature_policy": _best_feature_policy(feature_reports),
        "classification": {"primary": primary, "labels": [primary]},
        "route_recommendation": recommendation,
        "artifact_created": False,
        "training_ran": False,
        "shadow_live_ab_ran": False,
        "training_authorized": False,
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


def validate_sources(
    *,
    v154_report: Mapping[str, object],
    v155_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    failures: list[str] = []
    v154_validation = validate_v154_train_eval_inputs(
        v154_report=v154_report,
        dataset_rows=dataset_rows,
        min_label_count=100,
        target_seeds=target_seeds,
    )
    if v154_validation.get("passed") is not True:
        failures.append("v154_source_validation_failed")
    if (
        v155_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION
    ):
        failures.append("v155_schema_version_mismatch")
    if v155_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY:
        failures.append("v155_policy_mismatch")
    v155_classification = _mapping(v155_report.get("classification")).get("primary")
    if v155_classification != EXPECTED_V155_CLASSIFICATION:
        failures.append("v155_unexpected_classification")
    for field in (
        "diagnostics_only",
        "training_authorized",
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
        "runtime_action_selection_changed",
    ):
        expected = True if field == "diagnostics_only" else False
        if v155_report.get(field) is not expected:
            failures.append(f"v155_{field}_mismatch")
    v155_source = _mapping(v155_report.get("source_validation"))
    if v155_source.get("passed") is not True:
        failures.append("v155_source_validation_failed")
    if v155_source.get("dataset_digest") != v154_validation.get("dataset_digest"):
        failures.append("v155_v154_dataset_digest_mismatch")
    artifact = _mapping(v155_report.get("artifact"))
    training = _mapping(v155_report.get("training"))
    evaluation = _mapping(v155_report.get("evaluation"))
    if artifact.get("created") is not False:
        failures.append("v155_artifact_was_created")
    if training.get("diagnostic_training_ran") is not False:
        failures.append("v155_training_ran")
    if evaluation.get("skipped") is not True:
        failures.append("v155_evaluation_not_skipped")
    return {
        "policy": "m3_carrion_survivor_continuation_feature_sufficiency_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v154_validation": v154_validation,
        "expected_v154_classification": EXPECTED_V154_SUPPORT_READY_CLASSIFICATION,
        "expected_v155_classification": EXPECTED_V155_CLASSIFICATION,
        "observed_v155_classification": v155_classification,
        "dataset_digest": v154_validation.get("dataset_digest"),
        "label_count": v154_validation.get("label_count"),
    }


def candidate_feature_policies() -> list[dict[str, object]]:
    return [
        {
            "policy_id": "current_observation_action_mask_only",
            "description": "current public observation_input and action_mask only",
            "builder": _features_current_observation_mask,
        },
        {
            "policy_id": "current_observation_action_mask_existing_public_prior_context",
            "description": (
                "current public observation_input, action_mask, and any existing "
                "public prior context already present in v154 trainable rows"
            ),
            "builder": _features_existing_prior_context,
        },
        {
            "policy_id": "current_observation_action_mask_public_vital_recent_deltas",
            "description": (
                "current public observation_input/action_mask plus whitelisted "
                "pre-decision public energy/hydration/health recent deltas"
            ),
            "builder": _features_vital_recent_deltas,
        },
        {
            "policy_id": "current_observation_action_mask_public_recent_action_result_history",
            "description": (
                "current public observation_input/action_mask plus whitelisted "
                "prior finalized public action/result history"
            ),
            "builder": _features_recent_action_result_history,
        },
        {
            "policy_id": "public_local_resource_carrion_contact_indicators",
            "description": (
                "decoded public local resource/carrion/contact indicators and "
                "action_mask, without full observation tensor"
            ),
            "builder": _features_local_resource_contact_indicators,
        },
    ]


def attach_public_recent_transition_context(
    rows: Sequence[Mapping[str, object]],
    *,
    branch_results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    public_context_by_branch = {
        str(branch.get("branch_id", "")): _whitelisted_recent_public_transition_from_branch(
            branch
        )
        for branch in branch_results
        if str(branch.get("branch_id", ""))
    }
    audit_rows: list[dict[str, object]] = []
    for row in rows:
        row_payload = dict(row)
        branch_id = str(_mapping(row.get("metadata")).get("branch_id", ""))
        public_transition = public_context_by_branch.get(branch_id, {})
        if public_transition:
            row_payload["_v156_public_recent_transition"] = public_transition
        audit_rows.append(row_payload)
    return audit_rows


def evaluate_feature_policy(
    *,
    rows: Sequence[Mapping[str, object]],
    policy: Mapping[str, object],
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
    min_nn_over_trivial_margin: float = DEFAULT_MIN_NN_OVER_TRIVIAL_MARGIN,
    material_conflict_row_reduction: float = DEFAULT_MATERIAL_CONFLICT_ROW_REDUCTION,
    baseline_conflicting_row_count: int | None = None,
) -> dict[str, object]:
    builder = policy.get("builder")
    if not callable(builder):
        raise CarrionSurvivorContinuationFeatureSufficiencyError(
            "feature policy builder must be callable"
        )
    feature_rows = [
        {
            "row_index": index,
            "seed": _row_seed(row),
            "label_action": _label_action(row),
            "action_mask": _mapping(
                _mapping(_mapping(row.get("trainable")).get("features")).get(
                    "action_mask"
                )
            ),
            "features": builder(row),
        }
        for index, row in enumerate(rows)
    ]
    leakage = feature_leakage_scan(
        [dict(item["features"]) for item in feature_rows if isinstance(item.get("features"), Mapping)]
    )
    alias = alias_report(feature_rows)
    loo = leave_one_seed_out_nn_report(feature_rows, target_seeds=target_seeds)
    action_only = action_only_baseline(feature_rows, target_seeds=target_seeds)
    mask_only = mask_only_baseline(feature_rows, target_seeds=target_seeds)
    trivial_best = max(_float(action_only.get("accuracy")), _float(mask_only.get("accuracy")))
    nn_margin = _round(_float(loo.get("accuracy")) - trivial_best)
    baseline_conflicting_rows = (
        baseline_conflicting_row_count
        if baseline_conflicting_row_count is not None
        else _int(alias.get("conflicting_row_count"))
    )
    material_threshold = math.floor(
        _float(baseline_conflicting_rows)
        * (1.0 - float(material_conflict_row_reduction))
    )
    conflicts_materially_collapsed = (
        _int(alias.get("conflicting_row_count")) <= int(material_threshold)
    )
    support_ready = (
        leakage.get("passed") is True
        and conflicts_materially_collapsed
        and nn_margin >= float(min_nn_over_trivial_margin)
    )
    return {
        "policy_id": policy.get("policy_id"),
        "description": policy.get("description"),
        "row_count": len(feature_rows),
        "exact_feature_group_count": alias.get("exact_feature_group_count"),
        "conflicting_group_count": alias.get("conflicting_group_count"),
        "conflicting_row_count": alias.get("conflicting_row_count"),
        "baseline_conflicting_row_count": int(baseline_conflicting_rows),
        "conflicting_examples": alias.get("conflicting_examples"),
        "leave_one_seed_out_nn_accuracy": loo.get("accuracy"),
        "leave_one_seed_out_nn": loo,
        "action_only_baseline": action_only,
        "mask_only_baseline": mask_only,
        "best_trivial_baseline_accuracy": _round(trivial_best),
        "nearest_neighbor_minus_best_trivial_accuracy": nn_margin,
        "dominant_predicted_action_share": loo.get("dominant_predicted_action_share"),
        "dominant_predicted_action": loo.get("dominant_predicted_action"),
        "leakage_scan": leakage,
        "conflicts_materially_collapsed": conflicts_materially_collapsed,
        "material_conflict_row_threshold": int(material_threshold),
        "min_nn_over_trivial_margin": _round(min_nn_over_trivial_margin),
        "support_ready_no_training": support_ready,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
    }


def alias_report(
    feature_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in feature_rows:
        groups[stable_payload_digest(_mapping(row.get("features")))].append(row)
    conflicts = []
    for digest, items in sorted(groups.items()):
        counts = Counter(str(item.get("label_action", "")) for item in items)
        if len(counts) <= 1:
            continue
        conflicts.append(
            {
                "feature_digest": digest,
                "row_count": len(items),
                "label_action_counts": dict(sorted(counts.items())),
                "row_indexes_sample": [
                    _int(item.get("row_index")) for item in items[:12]
                ],
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_feature_alias_report_v1",
        "exact_feature_group_count": len(groups),
        "conflicting_group_count": len(conflicts),
        "conflicting_row_count": sum(int(item["row_count"]) for item in conflicts),
        "max_rows_per_feature_group": max((len(items) for items in groups.values()), default=0),
        "max_label_action_count_per_feature_group": max(
            (
                len(Counter(str(item.get("label_action", "")) for item in items))
                for items in groups.values()
            ),
            default=0,
        ),
        "conflicting_examples": conflicts[:8],
    }


def leave_one_seed_out_nn_report(
    feature_rows: Sequence[Mapping[str, object]],
    *,
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    vectors = [_feature_vector(_mapping(row.get("features"))) for row in feature_rows]
    per_seed = []
    correct_total = 0
    predicted_total = 0
    predicted_counts: Counter[str] = Counter()
    missing_seeds = []
    for seed in target_seeds:
        heldout_indexes = [
            index for index, row in enumerate(feature_rows) if _int(row.get("seed")) == int(seed)
        ]
        train_indexes = [
            index for index, row in enumerate(feature_rows) if _int(row.get("seed")) != int(seed)
        ]
        if not heldout_indexes:
            missing_seeds.append(int(seed))
            per_seed.append(
                {
                    "seed": int(seed),
                    "heldout_row_count": 0,
                    "predicted_count": 0,
                    "correct_count": 0,
                    "accuracy": 0.0,
                    "missing": True,
                }
            )
            continue
        correct = 0
        seed_predictions: Counter[str] = Counter()
        samples = []
        for heldout_index in heldout_indexes:
            predicted, distance = _nearest_neighbor_prediction(
                heldout_vector=vectors[heldout_index],
                train_indexes=train_indexes,
                feature_rows=feature_rows,
                vectors=vectors,
            )
            label = str(feature_rows[heldout_index].get("label_action", ""))
            seed_predictions.update([predicted])
            predicted_counts.update([predicted])
            if predicted == label:
                correct += 1
            if len(samples) < 8:
                samples.append(
                    {
                        "row_index": _int(feature_rows[heldout_index].get("row_index")),
                        "label_action": label,
                        "predicted_action": predicted,
                        "nearest_distance": _round(distance),
                    }
                )
        correct_total += correct
        predicted_total += len(heldout_indexes)
        per_seed.append(
            {
                "seed": int(seed),
                "train_row_count": len(train_indexes),
                "heldout_row_count": len(heldout_indexes),
                "predicted_count": len(heldout_indexes),
                "correct_count": correct,
                "accuracy": _safe_rate(correct, len(heldout_indexes)),
                "prediction_counts": dict(sorted(seed_predictions.items())),
                "sample_predictions": samples,
                "missing": False,
            }
        )
    dominant = _dominant_count_share(predicted_counts)
    return {
        "policy": "m3_carrion_survivor_continuation_feature_loo_1nn_v1",
        "target_seeds": [int(seed) for seed in target_seeds],
        "missing_target_seeds": missing_seeds,
        "all_target_seeds_evaluated": not missing_seeds,
        "predicted_count": predicted_total,
        "correct_count": correct_total,
        "accuracy": _safe_rate(correct_total, predicted_total),
        "predicted_action_counts": dict(sorted(predicted_counts.items())),
        "dominant_predicted_action": dominant.get("key"),
        "dominant_predicted_action_share": dominant.get("share"),
        "per_seed": per_seed,
    }


def action_only_baseline(
    feature_rows: Sequence[Mapping[str, object]],
    *,
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    return _prior_baseline(
        feature_rows,
        target_seeds=target_seeds,
        grouping_key=None,
        policy="m3_carrion_survivor_continuation_feature_action_only_baseline_v1",
    )


def mask_only_baseline(
    feature_rows: Sequence[Mapping[str, object]],
    *,
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    return _prior_baseline(
        feature_rows,
        target_seeds=target_seeds,
        grouping_key=lambda row: stable_payload_digest(_mapping(row.get("action_mask"))),
        policy="m3_carrion_survivor_continuation_feature_mask_only_baseline_v1",
    )


def feature_leakage_scan(
    feature_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures = []
    for row_index, payload in enumerate(feature_payloads):
        _scan_feature_value(
            value=payload,
            path=(),
            row_index=row_index,
            failures=failures,
        )
    return {
        "policy": "m3_carrion_survivor_continuation_feature_payload_leakage_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
        "forbidden_key_tokens": list(FORBIDDEN_FEATURE_KEY_TOKENS),
        "forbidden_string_markers": list(FORBIDDEN_FEATURE_STRING_MARKERS),
    }


def _features_current_observation_mask(row: Mapping[str, object]) -> dict[str, object]:
    features = _trainable_features(row)
    return {
        "observation_input": deepcopy(_mapping(features.get("observation_input"))),
        "action_mask": _complete_action_mask(_mapping(features.get("action_mask"))),
    }


def _features_existing_prior_context(row: Mapping[str, object]) -> dict[str, object]:
    features = _trainable_features(row)
    return {
        **_features_current_observation_mask(row),
        "prior_public_context": [
            _sanitize_existing_public_context_item(item)
            for item in _list_of_mappings(features.get("prior_public_context"))
        ],
    }


def _features_vital_recent_deltas(row: Mapping[str, object]) -> dict[str, object]:
    recent = _whitelisted_recent_public_transition(row)
    return {
        **_features_current_observation_mask(row),
        "recent_public_vital_delta": {
            "available": bool(recent),
            "energy_delta": _ratio_delta(recent, "energy"),
            "hydration_delta": _ratio_delta(recent, "hydration"),
            "health_delta": _ratio_delta(recent, "health"),
            "energy_after": _optional_float_feature(recent.get("energy_ratio_after")),
            "hydration_after": _optional_float_feature(
                recent.get("hydration_ratio_after")
            ),
            "health_after": _optional_float_feature(recent.get("health_ratio_after")),
        },
    }


def _features_recent_action_result_history(row: Mapping[str, object]) -> dict[str, object]:
    recent = _whitelisted_recent_public_transition(row)
    requested = str(recent.get("requested_action") or "")
    resolved = str(recent.get("resolved_action") or "")
    return {
        **_features_current_observation_mask(row),
        "recent_public_action_result": {
            "available": bool(recent),
            "requested_action": _action_one_hot_payload(requested),
            "resolved_action": _action_one_hot_payload(resolved),
            "moved": bool(recent.get("moved", False)),
            "ate": bool(recent.get("ate", False)),
            "died_after_action": bool(recent.get("died_after_action", False)),
            "food_source_carcass": str(recent.get("food_source") or "") == "carcass",
            "food_source_fresh_kill": str(recent.get("food_source") or "") == "fresh_kill",
            "energy_delta": _ratio_delta(recent, "energy"),
            "hydration_delta": _ratio_delta(recent, "hydration"),
        },
    }


def _features_local_resource_contact_indicators(
    row: Mapping[str, object],
) -> dict[str, object]:
    features = _trainable_features(row)
    runtime_row = planner_distilled_runtime_row(
        observation_input=_mapping(features.get("observation_input")),
        action_mask=_mapping(features.get("action_mask")),
        public_history_trace=[],
    )
    state = _mapping(runtime_row.get("compact_state"))
    self_state = _mapping(state.get("self"))
    center = _mapping(state.get("center"))
    local = _mapping(state.get("local"))
    adjacent = _mapping(state.get("adjacent"))
    adjacent_payload = {
        direction: {
            "water": _optional_float_feature(_mapping(adjacent.get(direction)).get("water")),
            "food": _optional_float_feature(_mapping(adjacent.get(direction)).get("food")),
            "carrion": _cell_carrion(_mapping(adjacent.get(direction))),
            "risk": _cell_risk(_mapping(adjacent.get(direction))),
        }
        for direction in ("north", "south", "east", "west")
    }
    return {
        "action_mask": _complete_action_mask(_mapping(features.get("action_mask"))),
        "public_self_vitals": {
            "energy_ratio": _optional_float_feature(self_state.get("energy_ratio")),
            "hydration_ratio": _optional_float_feature(
                self_state.get("hydration_ratio")
            ),
            "health_ratio": _optional_float_feature(self_state.get("health_ratio")),
        },
        "public_current_cell": {
            "water": _optional_float_feature(center.get("water")),
            "food": _optional_float_feature(center.get("food")),
            "carrion": _cell_carrion(center),
            "risk": _cell_risk(center),
        },
        "public_local_radius": {
            "radius1_water": _optional_float_feature(local.get("radius1_water")),
            "radius1_food": _optional_float_feature(local.get("radius1_food")),
            "radius1_carrion": _optional_float_feature(local.get("radius1_carrion")),
            "radius1_risk": _optional_float_feature(local.get("radius1_risk")),
            "radius2_water": _optional_float_feature(local.get("radius2_water")),
            "radius2_food": _optional_float_feature(local.get("radius2_food")),
            "radius2_carrion": _optional_float_feature(local.get("radius2_carrion")),
            "radius2_risk": _optional_float_feature(local.get("radius2_risk")),
        },
        "public_adjacent_cells": adjacent_payload,
    }


def _prior_baseline(
    feature_rows: Sequence[Mapping[str, object]],
    *,
    target_seeds: Sequence[int],
    grouping_key: Callable[[Mapping[str, object]], str] | None,
    policy: str,
) -> dict[str, object]:
    per_seed = []
    correct_total = 0
    predicted_total = 0
    predicted_counts: Counter[str] = Counter()
    missing_seeds = []
    for seed in target_seeds:
        heldout = [row for row in feature_rows if _int(row.get("seed")) == int(seed)]
        train = [row for row in feature_rows if _int(row.get("seed")) != int(seed)]
        if not heldout:
            missing_seeds.append(int(seed))
            per_seed.append(
                {
                    "seed": int(seed),
                    "heldout_row_count": 0,
                    "correct_count": 0,
                    "accuracy": 0.0,
                    "missing": True,
                }
            )
            continue
        global_counts = Counter(str(row.get("label_action", "")) for row in train)
        fallback = _majority_action(global_counts)
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        if grouping_key is not None:
            for row in train:
                grouped[str(grouping_key(row))].update([str(row.get("label_action", ""))])
        correct = 0
        seed_predictions: Counter[str] = Counter()
        for row in heldout:
            if grouping_key is None:
                predicted = fallback
            else:
                predicted = _majority_action(grouped.get(str(grouping_key(row)), Counter()))
                if not predicted:
                    predicted = fallback
            seed_predictions.update([predicted])
            predicted_counts.update([predicted])
            if predicted == str(row.get("label_action", "")):
                correct += 1
        correct_total += correct
        predicted_total += len(heldout)
        per_seed.append(
            {
                "seed": int(seed),
                "train_row_count": len(train),
                "heldout_row_count": len(heldout),
                "correct_count": correct,
                "accuracy": _safe_rate(correct, len(heldout)),
                "prediction_counts": dict(sorted(seed_predictions.items())),
                "missing": False,
            }
        )
    dominant = _dominant_count_share(predicted_counts)
    return {
        "policy": policy,
        "target_seeds": [int(seed) for seed in target_seeds],
        "missing_target_seeds": missing_seeds,
        "all_target_seeds_evaluated": not missing_seeds,
        "predicted_count": predicted_total,
        "correct_count": correct_total,
        "accuracy": _safe_rate(correct_total, predicted_total),
        "predicted_action_counts": dict(sorted(predicted_counts.items())),
        "dominant_predicted_action": dominant.get("key"),
        "dominant_predicted_action_share": dominant.get("share"),
        "per_seed": per_seed,
    }


def _nearest_neighbor_prediction(
    *,
    heldout_vector: Sequence[float],
    train_indexes: Sequence[int],
    feature_rows: Sequence[Mapping[str, object]],
    vectors: Sequence[Sequence[float]],
) -> tuple[str, float]:
    best: tuple[float, str, int] | None = None
    for index in train_indexes:
        distance = _squared_distance(heldout_vector, vectors[index])
        action = str(feature_rows[index].get("label_action", ""))
        candidate = (distance, action, int(index))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return "", float("inf")
    return best[1], best[0]


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    size = max(len(left), len(right))
    total = 0.0
    for index in range(size):
        a = float(left[index]) if index < len(left) else 0.0
        b = float(right[index]) if index < len(right) else 0.0
        diff = a - b
        total += diff * diff
    return _round(total)


def _feature_vector(payload: Mapping[str, object]) -> tuple[float, ...]:
    values: list[float] = []
    _append_feature_values(payload, values)
    return tuple(_round(value) for value in values)


def _append_feature_values(value: object, values: list[float]) -> None:
    if isinstance(value, Mapping):
        if _looks_like_observation_input(value):
            try:
                values.extend(float(item) for item in decode_observation_input(dict(value)))
            except (TypeError, ValueError):
                pass
            return
        for key in sorted(value):
            _append_feature_values(value[key], values)
        return
    if isinstance(value, list):
        for item in value:
            _append_feature_values(item, values)
        return
    if isinstance(value, bool):
        values.append(1.0 if value else 0.0)
        return
    if isinstance(value, (int, float)):
        number = float(value)
        values.append(number if math.isfinite(number) else 0.0)
        return
    if isinstance(value, str):
        # String categorical values should be one-hot before they reach this point.
        return


def _scan_feature_value(
    *,
    value: object,
    path: Sequence[str],
    row_index: int,
    failures: list[dict[str, object]],
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if any(token in lowered for token in FORBIDDEN_FEATURE_KEY_TOKENS):
                failures.append(
                    {
                        "row_index": row_index,
                        "path": ".".join((*path, key)),
                        "reason": "forbidden_feature_key_token",
                        "key": key,
                    }
                )
            _scan_feature_value(
                value=child,
                path=(*path, key),
                row_index=row_index,
                failures=failures,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_feature_value(
                value=child,
                path=(*path, str(index)),
                row_index=row_index,
                failures=failures,
            )
        return
    if isinstance(value, str):
        if path and path[-1] == "data" and "observation_input" in path:
            return
        lowered = value.lower()
        if any(marker in lowered for marker in FORBIDDEN_FEATURE_STRING_MARKERS):
            failures.append(
                {
                    "row_index": row_index,
                    "path": ".".join(path),
                    "reason": "forbidden_feature_string_marker",
                    "value": value[:128],
                }
            )


def _whitelisted_recent_public_transition(
    row: Mapping[str, object],
) -> dict[str, object]:
    return dict(_mapping(row.get("_v156_public_recent_transition")))


def _whitelisted_recent_public_transition_from_branch(
    branch: Mapping[str, object],
) -> dict[str, object]:
    context = _mapping(branch.get("carrion_archive_context"))
    if not context:
        return {}
    reason = _mapping(context.get("reason_evidence"))
    if not reason:
        return {}
    return {
        "requested_action": _action_or_empty(reason.get("requested_action")),
        "resolved_action": _action_or_empty(reason.get("resolved_action")),
        "moved": bool(reason.get("moved", False)),
        "ate": bool(reason.get("ate", False)),
        "food_source": _food_source_or_empty(reason.get("food_source")),
        "hydration_ratio_before": _optional_float_feature(
            reason.get("hydration_ratio_before")
        ),
        "hydration_ratio_after": _optional_float_feature(
            reason.get("hydration_ratio_after")
        ),
        "energy_ratio_before": _optional_float_feature(reason.get("energy_ratio_before")),
        "energy_ratio_after": _optional_float_feature(reason.get("energy_ratio_after")),
        "health_ratio_before": _optional_float_feature(reason.get("health_ratio_before")),
        "health_ratio_after": _optional_float_feature(reason.get("health_ratio_after")),
        "died_after_action": bool(reason.get("died_after_action", False)),
    }


def _sanitize_existing_public_context_item(item: Mapping[str, object]) -> dict[str, object]:
    requested = _action_or_empty(item.get("requested_action"))
    resolved = _action_or_empty(item.get("resolved_action"))
    return {
        "relative_step_age": _optional_float_feature(item.get("relative_step_age")),
        "requested_action": _action_one_hot_payload(requested),
        "resolved_action": _action_one_hot_payload(resolved),
        "moved": bool(item.get("moved", False)),
        "ate": bool(item.get("ate", False)),
        "drank": bool(item.get("drank", False)),
        "energy_delta": _optional_float_feature(item.get("energy_ratio_delta")),
        "hydration_delta": _optional_float_feature(item.get("hydration_ratio_delta")),
        "health_delta": _optional_float_feature(item.get("health_ratio_delta")),
    }


def _trainable_features(row: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(_mapping(row.get("trainable")).get("features"))


def _complete_action_mask(mask: Mapping[str, object]) -> dict[str, bool]:
    return {action: bool(mask.get(action, False)) for action in ACTION_NAMES}


def _row_seed(row: Mapping[str, object]) -> int:
    return _int(_mapping(row.get("metadata")).get("seed"))


def _label_action(row: Mapping[str, object]) -> str:
    return str(_mapping(_mapping(row.get("trainable")).get("label")).get("action", ""))


def _action_or_empty(value: object) -> str:
    action = str(value or "")
    return action if action in ACTION_NAMES else ""


def _food_source_or_empty(value: object) -> str:
    source = str(value or "")
    return source if source in {"carcass", "fresh_kill", "plant"} else ""


def _action_one_hot_payload(action: str) -> dict[str, bool]:
    return {name: action == name for name in ACTION_NAMES}


def _ratio_delta(source: Mapping[str, object], prefix: str) -> float:
    before = _optional_float_feature(source.get(f"{prefix}_ratio_before"))
    after = _optional_float_feature(source.get(f"{prefix}_ratio_after"))
    return _round(after - before)


def _optional_float_feature(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return _round(number if math.isfinite(number) else 0.0)


def _cell_carrion(cell: Mapping[str, object]) -> float:
    return max(
        _optional_float_feature(cell.get("fresh_kill")),
        _optional_float_feature(cell.get("carcass")),
        _optional_float_feature(cell.get("carrion_signal")),
    )


def _cell_risk(cell: Mapping[str, object]) -> float:
    return max(
        _optional_float_feature(cell.get("hazard_level")),
        _optional_float_feature(cell.get("predator_risk")),
    )


def _looks_like_observation_input(value: Mapping[str, object]) -> bool:
    return (
        value.get("schema_version") == "mind_observation_v3"
        and value.get("storage_encoding") == "zlib_base64_little_endian_int16"
        and isinstance(value.get("data"), str)
    )


def _majority_action(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return sorted(
        counts.items(),
        key=lambda item: (int(item[1]), _action_sort_key(str(item[0]))),
        reverse=True,
    )[0][0]


def _action_sort_key(action: str) -> int:
    try:
        return len(ACTION_NAMES) - ACTION_NAMES.index(action)
    except ValueError:
        return 0


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return _round(float(numerator) / float(denominator))


def _best_feature_policy(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not reports:
        return None
    best = sorted(
        reports,
        key=lambda report: (
            bool(report.get("support_ready_no_training")),
            -_int(report.get("conflicting_row_count")),
            _float(report.get("nearest_neighbor_minus_best_trivial_accuracy")),
            _float(report.get("leave_one_seed_out_nn_accuracy")),
            str(report.get("policy_id", "")),
        ),
        reverse=True,
    )[0]
    return {
        "policy_id": best.get("policy_id"),
        "support_ready_no_training": best.get("support_ready_no_training"),
        "conflicting_row_count": best.get("conflicting_row_count"),
        "leave_one_seed_out_nn_accuracy": best.get("leave_one_seed_out_nn_accuracy"),
        "nearest_neighbor_minus_best_trivial_accuracy": best.get(
            "nearest_neighbor_minus_best_trivial_accuracy"
        ),
    }
