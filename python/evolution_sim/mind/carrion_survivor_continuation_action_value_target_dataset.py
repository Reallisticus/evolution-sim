from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
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
    DEFAULT_MIN_SAFE_RUNS_PER_ACTION,
    DEFAULT_MIN_SAFE_SHARE_PER_ACTION,
    DEFAULT_OUTPUT_PATH as DEFAULT_V157_REPORT_PATH,
    DEFAULT_UNIQUE_WINNER_SCORE_MARGIN,
    EXPECTED_V156_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_AUDIT_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_AUDIT_SCHEMA_VERSION,
    _action_order,
    _branch_proxy_row,
    _group_rows_by_feature_key,
    evaluate_action_value_feature_policy,
    hard_trainable_feature_leakage_scan,
    validate_action_value_audit_sources,
)
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    DEFAULT_DATASET_OUTPUT_PATH as DEFAULT_V154_DATASET_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V154_REPORT_PATH,
    TARGET_CARRION_SEEDS,
)
from evolution_sim.mind.carrion_survivor_continuation_feature_sufficiency_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V156_REPORT_PATH,
    attach_public_recent_transition_context,
    candidate_feature_policies,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V155_REPORT_PATH,
    load_json_report,
    load_v154_dataset,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_set_valued_action_value_target_dataset_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_set_valued_action_value_target_dataset_row_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_set_valued_action_value_target_dataset_v1"
)
EXPECTED_V157_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_public_state_action_value_audit_"
    "set_or_action_value_targets_reduce_conflicts_no_training"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v158-carrion-survivor-continuation-set-valued-action-value-target-dataset.json"
)
DEFAULT_TARGET_DATASET_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v158-carrion-survivor-continuation-set-valued-action-value-target-dataset.jsonl"
)
DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE = 0.50
SOURCE_FALSE_LIFECYCLE_FIELDS = (
    "training_authorized",
    "promotion_authorized",
    "runtime_promotion_allowed",
    "default_runtime_behavior_changed",
    "runtime_action_selection_changed",
)
SOURCE_OPTIONAL_FALSE_LIFECYCLE_FIELDS = (
    "artifact_created",
    "training_ran",
    "shadow_live_ab_ran",
)


class CarrionSurvivorContinuationActionValueTargetDatasetError(ValueError):
    pass


def run_carrion_survivor_continuation_action_value_target_dataset(
    *,
    v154_report_path: str | Path = DEFAULT_V154_REPORT_PATH,
    v154_dataset_path: str | Path = DEFAULT_V154_DATASET_PATH,
    v155_report_path: str | Path = DEFAULT_V155_REPORT_PATH,
    v156_report_path: str | Path = DEFAULT_V156_REPORT_PATH,
    v157_report_path: str | Path = DEFAULT_V157_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    target_dataset_output_path: str | Path = DEFAULT_TARGET_DATASET_OUTPUT_PATH,
    min_safe_runs_per_action: int = DEFAULT_MIN_SAFE_RUNS_PER_ACTION,
    min_safe_share_per_action: float = DEFAULT_MIN_SAFE_SHARE_PER_ACTION,
    unique_winner_score_margin: float = DEFAULT_UNIQUE_WINNER_SCORE_MARGIN,
    max_dominant_safe_action_share: float = DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    v154_report = load_json_report(v154_report_path)
    v155_report = load_json_report(v155_report_path)
    v156_report = load_json_report(v156_report_path)
    v157_report = load_json_report(v157_report_path)
    dataset_rows = load_v154_dataset(v154_dataset_path)
    branch_results = _list_of_mappings(
        v154_report.get("continuation_branch_results")
    )
    source_validation = validate_action_value_target_dataset_sources(
        v154_report=v154_report,
        v155_report=v155_report,
        v156_report=v156_report,
        v157_report=v157_report,
        dataset_rows=dataset_rows,
        branch_results=branch_results,
        target_seeds=target_seeds,
    )
    target_rows: list[dict[str, object]] = []
    selected_policy_report: dict[str, object] = {}
    build_validation: dict[str, object] = _skipped_build_validation(
        "source_validation_failed"
    )
    if source_validation.get("passed") is True:
        selected_policy_report, target_rows, build_validation = (
            build_action_value_target_rows(
                dataset_rows=dataset_rows,
                branch_results=branch_results,
                v157_report=v157_report,
                min_safe_runs_per_action=min_safe_runs_per_action,
                min_safe_share_per_action=min_safe_share_per_action,
                unique_winner_score_margin=unique_winner_score_margin,
            )
        )
    leakage_scan = target_dataset_leakage_scan(target_rows)
    target_summary = summarize_action_value_target_rows(
        target_rows,
        max_dominant_safe_action_share=max_dominant_safe_action_share,
    )
    lifecycle_authorization = source_validation.get("lifecycle_authorization")
    lifecycle_authorization_payload = (
        lifecycle_authorization if isinstance(lifecycle_authorization, Mapping) else {}
    )
    classification = _top_level_classification(
        source_validation=source_validation,
        build_validation=build_validation,
        leakage_scan=leakage_scan,
        target_summary=target_summary,
        lifecycle_authorization=lifecycle_authorization_payload,
    )
    dataset_digest = stable_payload_digest(target_rows)
    _write_jsonl(target_dataset_output_path, target_rows)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_POLICY,
        "contract": {
            "diagnostics_only": True,
            "artifact_creation_allowed": False,
            "model_artifact_creation_allowed": False,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
            "shadow_live_ab_allowed": False,
            "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_future_outcome_as_trainable_input": False,
            "uses_branch_reason_as_trainable_input": False,
        },
        "inputs": {
            "v154_report": str(v154_report_path),
            "v154_dataset": str(v154_dataset_path),
            "v155_report": str(v155_report_path),
            "v156_report": str(v156_report_path),
            "v157_report": str(v157_report_path),
            "target_dataset_output": str(target_dataset_output_path),
            "target_seeds": [int(seed) for seed in target_seeds],
            "min_safe_runs_per_action": int(min_safe_runs_per_action),
            "min_safe_share_per_action": _round(min_safe_share_per_action),
            "unique_winner_score_margin": _round(unique_winner_score_margin),
            "max_dominant_safe_action_share": _round(max_dominant_safe_action_share),
        },
        "source_validation": source_validation,
        "selected_v157_feature_policy": _selected_policy_summary(
            selected_policy_report
        ),
        "target_build_validation": build_validation,
        "leakage_scan": leakage_scan,
        "dataset": {
            "path": str(target_dataset_output_path),
            "row_schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION
            ),
            "group_count": target_summary.get("group_count"),
            "action_value_target_row_count": target_summary.get(
                "action_value_target_row_count"
            ),
            "unique_winner_group_count": target_summary.get(
                "unique_winner_group_count"
            ),
            "multi_action_safe_set_group_count": target_summary.get(
                "multi_action_safe_set_group_count"
            ),
            "unresolved_group_count": target_summary.get("unresolved_group_count"),
            "per_action_support_counts": target_summary.get(
                "per_action_support_counts"
            ),
            "per_action_value_target_counts": target_summary.get(
                "per_action_value_target_counts"
            ),
            "dominant_safe_action": target_summary.get("dominant_safe_action"),
            "dominant_safe_action_count": target_summary.get(
                "dominant_safe_action_count"
            ),
            "dominant_safe_action_share": target_summary.get(
                "dominant_safe_action_share"
            ),
            "action_support_non_collapsed": target_summary.get(
                "action_support_non_collapsed"
            ),
            "dataset_digest": dataset_digest,
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "dataset_created": True,
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


def validate_action_value_target_dataset_sources(
    *,
    v154_report: Mapping[str, object],
    v155_report: Mapping[str, object],
    v156_report: Mapping[str, object],
    v157_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    failures: list[str] = []
    v154_v157_validation = validate_action_value_audit_sources(
        v154_report=v154_report,
        v155_report=v155_report,
        v156_report=v156_report,
        dataset_rows=dataset_rows,
        branch_results=branch_results,
        target_seeds=target_seeds,
    )
    if v154_v157_validation.get("passed") is not True:
        failures.append("v154_v155_v156_source_validation_failed")
    if (
        v157_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_AUDIT_SCHEMA_VERSION
    ):
        failures.append("v157_schema_version_mismatch")
    if v157_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_AUDIT_POLICY:
        failures.append("v157_policy_mismatch")
    observed_v157_classification = _mapping(v157_report.get("classification")).get(
        "primary"
    )
    if observed_v157_classification != EXPECTED_V157_CLASSIFICATION:
        failures.append("v157_unexpected_classification")
    v157_source = _mapping(v157_report.get("source_validation"))
    if v157_source.get("passed") is not True:
        failures.append("v157_source_validation_failed")
    dataset_digest = stable_payload_digest(dataset_rows)
    if v157_source.get("dataset_digest") != dataset_digest:
        failures.append("v157_dataset_digest_mismatch")
    branch_digest = stable_payload_digest(branch_results)
    if v157_source.get("continuation_branch_evidence_digest") != branch_digest:
        failures.append("v157_continuation_branch_evidence_digest_mismatch")
    expected_policy_ids = [
        str(policy.get("policy_id")) for policy in candidate_feature_policies()
    ]
    observed_policy_ids = [
        str(policy.get("policy_id"))
        for policy in _list_of_mappings(v157_report.get("feature_policies"))
    ]
    if observed_policy_ids != expected_policy_ids:
        failures.append("v157_feature_policy_sequence_mismatch")
    best_policy_id = str(_mapping(v157_report.get("best_feature_policy")).get("policy_id", ""))
    if best_policy_id not in expected_policy_ids:
        failures.append("v157_best_feature_policy_unknown")
    selected = _v157_policy_report(v157_report, best_policy_id)
    if selected and _mapping(selected.get("leakage_scan")).get("passed") is not True:
        failures.append("v157_best_feature_policy_leakage_failed")
    lifecycle = lifecycle_authorization_scan(
        {
            "v154": v154_report,
            "v155": v155_report,
            "v156": v156_report,
            "v157": v157_report,
        }
    )
    if lifecycle.get("passed") is not True:
        failures.append("lifecycle_authorization_fields_not_false")
    exact_digest_validation = exact_digest_validation_report(v157_report)
    if exact_digest_validation.get("passed") is not True:
        failures.append("v157_exact_digest_mismatch")
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_target_dataset_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v156_classification": EXPECTED_V156_CLASSIFICATION,
        "expected_v157_classification": EXPECTED_V157_CLASSIFICATION,
        "observed_v157_classification": observed_v157_classification,
        "v154_v155_v156_source_validation": v154_v157_validation,
        "lifecycle_authorization": lifecycle,
        "v157_exact_digest_validation": exact_digest_validation,
        "dataset_digest": dataset_digest,
        "label_count": len(dataset_rows),
        "continuation_branch_result_count": len(branch_results),
        "continuation_branch_evidence_digest": branch_digest,
        "v157_selected_policy_id": best_policy_id,
        "v157_feature_policy_ids": observed_policy_ids,
        "v157_exact_digest": v157_report.get("exact_digest"),
    }


def build_action_value_target_rows(
    *,
    dataset_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    v157_report: Mapping[str, object],
    min_safe_runs_per_action: int = DEFAULT_MIN_SAFE_RUNS_PER_ACTION,
    min_safe_share_per_action: float = DEFAULT_MIN_SAFE_SHARE_PER_ACTION,
    unique_winner_score_margin: float = DEFAULT_UNIQUE_WINNER_SCORE_MARGIN,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    best_policy_id = str(_mapping(v157_report.get("best_feature_policy")).get("policy_id", ""))
    policy = _candidate_policy_by_id(best_policy_id)
    builder = policy.get("builder")
    if not callable(builder):
        raise CarrionSurvivorContinuationActionValueTargetDatasetError(
            f"v157 best feature policy is not buildable: {best_policy_id}"
        )
    audit_rows = attach_public_recent_transition_context(
        dataset_rows,
        branch_results=branch_results,
    )
    branch_rows = attach_public_recent_transition_context(
        [_branch_proxy_row(result) for result in branch_results],
        branch_results=branch_results,
    )
    selected_policy_report = evaluate_action_value_feature_policy(
        rows=audit_rows,
        branch_results=branch_results,
        branch_rows=branch_rows,
        policy=policy,
        min_safe_runs_per_action=min_safe_runs_per_action,
        min_safe_share_per_action=min_safe_share_per_action,
        unique_winner_score_margin=unique_winner_score_margin,
    )
    row_groups = _group_rows_by_feature_key(audit_rows, builder)
    group_reports_by_digest = {
        str(group.get("feature_digest", "")): group
        for group in _list_of_mappings(selected_policy_report.get("group_reports"))
    }
    target_rows: list[dict[str, object]] = []
    missing_group_digests: list[str] = []
    for feature_digest in sorted(row_groups):
        group = group_reports_by_digest.get(feature_digest)
        if group is None:
            missing_group_digests.append(feature_digest)
            continue
        features = deepcopy(_mapping(row_groups[feature_digest].get("features")))
        target_rows.append(
            _build_target_dataset_row(
                features=features,
                group=group,
                feature_policy_id=best_policy_id,
            )
        )
    v157_selected = _v157_policy_report(v157_report, best_policy_id)
    selected_consistency = _selected_policy_consistency_report(
        recomputed=selected_policy_report,
        v157_selected=v157_selected,
    )
    failures = []
    if missing_group_digests:
        failures.append("missing_v157_group_report_for_dataset_feature_state")
    if selected_consistency.get("passed") is not True:
        failures.append("recomputed_selected_policy_differs_from_v157")
    return (
        dict(selected_policy_report),
        target_rows,
        {
            "policy": "m3_carrion_survivor_continuation_action_value_target_dataset_build_validation_v1",
            "passed": not failures,
            "failures": failures,
            "selected_policy_consistency": selected_consistency,
            "v154_exact_public_feature_group_count": len(row_groups),
            "v157_selected_public_state_evidence_group_count": len(
                _list_of_mappings(selected_policy_report.get("group_reports"))
            ),
            "target_dataset_row_count": len(target_rows),
            "missing_group_digest_count": len(missing_group_digests),
        },
    )


def target_dataset_leakage_scan(
    target_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    trainable_payloads = [
        {
            "trainable_public_features": deepcopy(
                _mapping(row.get("trainable_public_features"))
            ),
            "public_action_mask": deepcopy(_mapping(row.get("public_action_mask"))),
        }
        for row in target_rows
    ]
    scan = hard_trainable_feature_leakage_scan(trainable_payloads)
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_target_dataset_leakage_scan_v1",
        "passed": scan.get("passed") is True,
        "failure_count": scan.get("failure_count"),
        "failures": scan.get("failures"),
        "trainable_feature_row_count": len(trainable_payloads),
        "hard_trainable_feature_leakage_scan": scan,
    }


def summarize_action_value_target_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    max_dominant_safe_action_share: float = DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
) -> dict[str, object]:
    classification_counts = Counter(str(row.get("target_classification", "")) for row in rows)
    support_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    for row in rows:
        for action in _safe_action_set(row):
            support_counts.update([action])
        for target in _list_of_mappings(row.get("action_value_targets")):
            if target.get("target_available") is True:
                target_counts.update([str(target.get("action", ""))])
    support_counts = Counter(
        {
            action: int(support_counts.get(action, 0))
            for action in ACTION_NAMES
            if int(support_counts.get(action, 0)) > 0
        }
    )
    target_counts_payload = {
        action: int(target_counts.get(action, 0))
        for action in ACTION_NAMES
        if int(target_counts.get(action, 0)) > 0
    }
    dominant = _dominant_count_share(support_counts)
    unresolved = sum(
        count
        for label, count in classification_counts.items()
        if label not in {"unique_robust_winner", "multi_action_safe_set"}
    )
    action_support_non_collapsed = (
        sum(int(count) for count in support_counts.values()) > 0
        and len([action for action, count in support_counts.items() if count > 0]) >= 2
        and _float(dominant.get("share")) <= float(max_dominant_safe_action_share)
    )
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_target_dataset_summary_v1",
        "group_count": len(rows),
        "action_value_target_row_count": len(rows),
        "unique_winner_group_count": int(classification_counts.get("unique_robust_winner", 0)),
        "multi_action_safe_set_group_count": int(
            classification_counts.get("multi_action_safe_set", 0)
        ),
        "unresolved_group_count": int(unresolved),
        "classification_counts": dict(sorted(classification_counts.items())),
        "per_action_support_counts": dict(sorted(support_counts.items())),
        "per_action_value_target_counts": dict(sorted(target_counts_payload.items())),
        "dominant_safe_action": dominant.get("key"),
        "dominant_safe_action_count": dominant.get("count"),
        "dominant_safe_action_share": dominant.get("share"),
        "max_dominant_safe_action_share": _round(max_dominant_safe_action_share),
        "action_support_non_collapsed": action_support_non_collapsed,
    }


def lifecycle_authorization_scan(
    reports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for source_name, report in reports.items():
        if report.get("diagnostics_only") is not True:
            failures.append(
                {
                    "source": source_name,
                    "field": "diagnostics_only",
                    "observed": report.get("diagnostics_only"),
                    "expected": True,
                }
            )
        for field in SOURCE_FALSE_LIFECYCLE_FIELDS:
            if report.get(field) is not False:
                failures.append(
                    {
                        "source": source_name,
                        "field": field,
                        "observed": report.get(field),
                        "expected": False,
                    }
                )
        for field in SOURCE_OPTIONAL_FALSE_LIFECYCLE_FIELDS:
            if field in report and report.get(field) is not False:
                failures.append(
                    {
                        "source": source_name,
                        "field": field,
                        "observed": report.get(field),
                        "expected": False,
                    }
                )
        contract = _mapping(report.get("contract"))
        for field in SOURCE_FALSE_LIFECYCLE_FIELDS:
            if field in contract and contract.get(field) is not False:
                failures.append(
                    {
                        "source": source_name,
                        "field": f"contract.{field}",
                        "observed": contract.get(field),
                        "expected": False,
                    }
                )
        for field in ("artifact_creation_allowed", "shadow_live_ab_allowed"):
            if field in contract and contract.get(field) is not False:
                failures.append(
                    {
                        "source": source_name,
                        "field": f"contract.{field}",
                        "observed": contract.get(field),
                        "expected": False,
                    }
                )
    v155 = _mapping(reports.get("v155"))
    artifact = _mapping(v155.get("artifact"))
    training = _mapping(v155.get("training"))
    if artifact.get("created") is not False:
        failures.append(
            {
                "source": "v155",
                "field": "artifact.created",
                "observed": artifact.get("created"),
                "expected": False,
            }
        )
    if training.get("diagnostic_training_ran") is not False:
        failures.append(
            {
                "source": "v155",
                "field": "training.diagnostic_training_ran",
                "observed": training.get("diagnostic_training_ran"),
                "expected": False,
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_target_dataset_lifecycle_authorization_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def exact_digest_validation_report(report: Mapping[str, object]) -> dict[str, object]:
    observed = report.get("exact_digest")
    without_digest = dict(report)
    without_digest.pop("exact_digest", None)
    computed = stable_payload_digest(without_digest)
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_target_dataset_exact_digest_validation_v1",
        "passed": observed == computed,
        "observed_exact_digest": observed,
        "computed_exact_digest": computed,
    }


def _build_target_dataset_row(
    *,
    features: Mapping[str, object],
    group: Mapping[str, object],
    feature_policy_id: str,
) -> dict[str, object]:
    action_mask = _complete_action_mask(_mapping(features.get("action_mask")))
    safe_set = _ordered_action_set(_list_of_actions(group.get("safe_action_set")))
    classification = str(group.get("classification", ""))
    row: dict[str, object] = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION
        ),
        "feature_policy_id": feature_policy_id,
        "trainable_public_features": deepcopy(dict(features)),
        "public_action_mask": action_mask,
        "action_value_targets": _action_value_targets(
            action_mask=action_mask,
            action_reports=_list_of_mappings(group.get("action_values")),
            safe_action_set=safe_set,
        ),
        "safe_action_set": safe_set,
        "target_classification": classification,
    }
    robust_winner = str(group.get("robust_winner_action") or "")
    if classification == "unique_robust_winner" and robust_winner in ACTION_NAMES:
        row["robust_winner_action"] = robust_winner
    return row


def _action_value_targets(
    *,
    action_mask: Mapping[str, object],
    action_reports: Sequence[Mapping[str, object]],
    safe_action_set: Sequence[str],
) -> list[dict[str, object]]:
    reports_by_action = {
        str(report.get("action", "")): report for report in action_reports
    }
    safe = set(safe_action_set)
    targets = []
    for action in ACTION_NAMES:
        report = _mapping(reports_by_action.get(action))
        available = bool(report)
        target = {
            "action": action,
            "public_mask": bool(action_mask.get(action, False)),
            "target_available": available,
            "safe_target": action in safe,
            "score_target": _round(_float(report.get("mean_outcome_score")))
            if available
            else None,
            "value_target": _round(_float(report.get("mean_outcome_score")))
            if available
            else None,
            "safe_run_count": _int(report.get("safe_run_count")) if available else 0,
            "safe_share": _round(_float(report.get("safe_share"))) if available else 0.0,
            "continuation_run_count": _int(report.get("continuation_run_count"))
            if available
            else 0,
            "robust_safe_action": bool(report.get("robust_safe_action"))
            if available
            else False,
            "resolved_invalid_risk_run_count": _int(
                report.get("resolved_invalid_risk_run_count")
            )
            if available
            else 0,
            "unsupported_requested_action_total": _int(
                report.get("unsupported_requested_action_total")
            )
            if available
            else 0,
        }
        targets.append(target)
    return targets


def _top_level_classification(
    *,
    source_validation: Mapping[str, object],
    build_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    target_summary: Mapping[str, object],
    lifecycle_authorization: Mapping[str, object],
) -> str:
    prefix = (
        "m3_carrion_survivor_continuation_public_state_action_value_target_dataset_"
    )
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if build_validation.get("passed") is not True:
        return prefix + "build_validation_failed_closed_no_training"
    if leakage_scan.get("passed") is not True:
        return prefix + "leakage_failed_closed_no_training"
    if lifecycle_authorization.get("passed") is not True:
        return prefix + "lifecycle_authorization_failed_closed_no_training"
    if _int(target_summary.get("unresolved_group_count")) != 0:
        return prefix + "unresolved_public_state_groups_closed_no_training"
    if target_summary.get("action_support_non_collapsed") is not True:
        return prefix + "action_support_collapsed_closed_no_training"
    return prefix + "support_ready_no_training"


def _route_recommendation(classification: str) -> dict[str, object]:
    support_ready = classification.endswith("support_ready_no_training")
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_target_dataset_route_recommendation_v1",
        "target_dataset_support_ready": support_ready,
        "recommended_next_route": (
            "review_set_valued_action_value_target_dataset_before_any_separate_opt_in_training"
            if support_ready
            else "keep_target_dataset_diagnostics_closed_until_source_or_support_issue_is_resolved"
        ),
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
    }


def _selected_policy_summary(policy_report: Mapping[str, object]) -> dict[str, object]:
    if not policy_report:
        return {}
    return {
        "policy_id": policy_report.get("policy_id"),
        "exact_public_feature_group_count": policy_report.get(
            "exact_public_feature_group_count"
        ),
        "single_label_conflicting_group_count": policy_report.get(
            "single_label_conflicting_group_count"
        ),
        "single_label_conflicting_row_count": policy_report.get(
            "single_label_conflicting_row_count"
        ),
        "action_value_resolvable_conflicting_group_count": policy_report.get(
            "action_value_resolvable_conflicting_group_count"
        ),
        "action_value_resolvable_conflicting_row_count": policy_report.get(
            "action_value_resolvable_conflicting_row_count"
        ),
        "action_value_unresolved_conflicting_group_count": policy_report.get(
            "action_value_unresolved_conflicting_group_count"
        ),
        "action_value_unresolved_conflicting_row_count": policy_report.get(
            "action_value_unresolved_conflicting_row_count"
        ),
        "classification_counts": policy_report.get("classification_counts"),
    }


def _selected_policy_consistency_report(
    *,
    recomputed: Mapping[str, object],
    v157_selected: Mapping[str, object],
) -> dict[str, object]:
    fields = (
        "policy_id",
        "exact_public_feature_group_count",
        "single_label_conflicting_group_count",
        "single_label_conflicting_row_count",
        "action_value_resolvable_conflicting_group_count",
        "action_value_resolvable_conflicting_row_count",
        "action_value_unresolved_conflicting_group_count",
        "action_value_unresolved_conflicting_row_count",
        "classification_counts",
    )
    mismatches = []
    for field in fields:
        if recomputed.get(field) != v157_selected.get(field):
            mismatches.append(
                {
                    "field": field,
                    "recomputed": recomputed.get(field),
                    "v157": v157_selected.get(field),
                }
            )
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_target_dataset_v157_policy_consistency_v1",
        "passed": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _skipped_build_validation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_target_dataset_build_validation_v1",
        "passed": False,
        "failures": [reason],
        "target_dataset_row_count": 0,
    }


def _candidate_policy_by_id(policy_id: str) -> dict[str, object]:
    for policy in candidate_feature_policies():
        if str(policy.get("policy_id")) == policy_id:
            return policy
    raise CarrionSurvivorContinuationActionValueTargetDatasetError(
        f"unknown v157 feature policy id: {policy_id}"
    )


def _v157_policy_report(
    v157_report: Mapping[str, object],
    policy_id: str,
) -> dict[str, object]:
    for report in _list_of_mappings(v157_report.get("feature_policies")):
        if str(report.get("policy_id")) == policy_id:
            return dict(report)
    return {}


def _complete_action_mask(mask: Mapping[str, object]) -> dict[str, bool]:
    return {action: bool(mask.get(action, False)) for action in ACTION_NAMES}


def _ordered_action_set(actions: Sequence[str]) -> list[str]:
    return sorted(
        {action for action in actions if action in ACTION_NAMES},
        key=_action_order,
    )


def _list_of_actions(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(action) for action in value if str(action) in ACTION_NAMES]


def _safe_action_set(row: Mapping[str, object]) -> list[str]:
    return _ordered_action_set(_list_of_actions(row.get("safe_action_set")))


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    output.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
