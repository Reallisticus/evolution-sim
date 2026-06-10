from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import math
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    _dominant_count_share,
    _float,
    _int,
    _list,
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
from evolution_sim.mind.carrion_survivor_continuation_feature_sufficiency_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V156_REPORT_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_SCHEMA_VERSION,
    attach_public_recent_transition_context,
    candidate_feature_policies,
    feature_leakage_scan,
    validate_sources as validate_feature_sufficiency_sources,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V155_REPORT_PATH,
    load_json_report,
    load_v154_dataset,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_public_state_action_value_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_public_state_action_value_audit_v1"
)
EXPECTED_V156_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_public_feature_sufficiency_"
    "public_feature_surface_insufficient_no_training"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v157-carrion-survivor-continuation-public-state-action-value-audit.json"
)
DEFAULT_MIN_SAFE_RUNS_PER_ACTION = 2
DEFAULT_MIN_SAFE_SHARE_PER_ACTION = 0.50
DEFAULT_UNIQUE_WINNER_SCORE_MARGIN = 0.25
FORBIDDEN_TRAINABLE_FEATURE_KEY_TOKENS = (
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
    "outcome",
    "reason",
)
FORBIDDEN_TRAINABLE_FEATURE_STRING_MARKERS = (
    "carrion_only",
    "m3-carrion-specific-archive",
    "branch-",
    "seed-",
    "carrion_contact",
    "post_carrion_hydration_risk",
    "movement_stall",
    "terminal_extinction",
)
GROUP_CLASSIFICATIONS = (
    "unique_robust_winner",
    "multi_action_safe_set",
    "conflicting_no_public_winner",
    "insufficient_action_coverage",
)


class CarrionSurvivorContinuationActionValueAuditError(ValueError):
    pass


def run_carrion_survivor_continuation_action_value_audit(
    *,
    v154_report_path: str | Path = DEFAULT_V154_REPORT_PATH,
    v154_dataset_path: str | Path = DEFAULT_V154_DATASET_PATH,
    v155_report_path: str | Path = DEFAULT_V155_REPORT_PATH,
    v156_report_path: str | Path = DEFAULT_V156_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    min_safe_runs_per_action: int = DEFAULT_MIN_SAFE_RUNS_PER_ACTION,
    min_safe_share_per_action: float = DEFAULT_MIN_SAFE_SHARE_PER_ACTION,
    unique_winner_score_margin: float = DEFAULT_UNIQUE_WINNER_SCORE_MARGIN,
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    v154_report = load_json_report(v154_report_path)
    v155_report = load_json_report(v155_report_path)
    v156_report = load_json_report(v156_report_path)
    dataset_rows = load_v154_dataset(v154_dataset_path)
    branch_results = _list_of_mappings(
        v154_report.get("continuation_branch_results")
    )
    source_validation = validate_action_value_audit_sources(
        v154_report=v154_report,
        v155_report=v155_report,
        v156_report=v156_report,
        dataset_rows=dataset_rows,
        branch_results=branch_results,
        target_seeds=target_seeds,
    )
    policy_reports: list[dict[str, object]] = []
    if source_validation.get("passed") is True:
        audit_rows = attach_public_recent_transition_context(
            dataset_rows,
            branch_results=branch_results,
        )
        branch_rows = attach_public_recent_transition_context(
            [_branch_proxy_row(result) for result in branch_results],
            branch_results=branch_results,
        )
        for policy in candidate_feature_policies():
            policy_reports.append(
                evaluate_action_value_feature_policy(
                    rows=audit_rows,
                    branch_results=branch_results,
                    branch_rows=branch_rows,
                    policy=policy,
                    min_safe_runs_per_action=min_safe_runs_per_action,
                    min_safe_share_per_action=min_safe_share_per_action,
                    unique_winner_score_margin=unique_winner_score_margin,
                )
            )
    best_policy = _best_policy_report(policy_reports)
    leakage_failures = [
        report
        for report in policy_reports
        if _mapping(report.get("leakage_scan")).get("passed") is not True
    ]
    conflict_reduction = _conflict_reduction_summary(
        policy_reports=policy_reports,
        best_policy=best_policy,
        v155_report=v155_report,
        v156_report=v156_report,
    )
    classification = _top_level_classification(
        source_validation=source_validation,
        leakage_failures=leakage_failures,
        conflict_reduction=conflict_reduction,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_AUDIT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_AUDIT_POLICY,
        "contract": {
            "diagnostics_only": True,
            "artifact_creation_allowed": False,
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
            "target_seeds": [int(seed) for seed in target_seeds],
            "min_safe_runs_per_action": int(min_safe_runs_per_action),
            "min_safe_share_per_action": _round(min_safe_share_per_action),
            "unique_winner_score_margin": _round(unique_winner_score_margin),
        },
        "source_validation": source_validation,
        "feature_policy_count": len(policy_reports),
        "feature_policies": policy_reports,
        "best_feature_policy": best_policy,
        "conflict_reduction": conflict_reduction,
        "group_classification_labels": list(GROUP_CLASSIFICATIONS),
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
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


def validate_action_value_audit_sources(
    *,
    v154_report: Mapping[str, object],
    v155_report: Mapping[str, object],
    v156_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    failures: list[str] = []
    v154_v155_validation = validate_feature_sufficiency_sources(
        v154_report=v154_report,
        v155_report=v155_report,
        dataset_rows=dataset_rows,
        target_seeds=target_seeds,
    )
    if v154_v155_validation.get("passed") is not True:
        failures.append("v154_v155_source_validation_failed")
    if (
        v156_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_SCHEMA_VERSION
    ):
        failures.append("v156_schema_version_mismatch")
    if v156_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_POLICY:
        failures.append("v156_policy_mismatch")
    observed_v156_classification = _mapping(v156_report.get("classification")).get(
        "primary"
    )
    if observed_v156_classification != EXPECTED_V156_CLASSIFICATION:
        failures.append("v156_unexpected_classification")
    for field in (
        "diagnostics_only",
        "training_authorized",
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
        "runtime_action_selection_changed",
    ):
        expected = True if field == "diagnostics_only" else False
        if v156_report.get(field) is not expected:
            failures.append(f"v156_{field}_mismatch")
    if v156_report.get("artifact_created") is not False:
        failures.append("v156_artifact_created_not_false")
    if v156_report.get("training_ran") is not False:
        failures.append("v156_training_ran_not_false")
    if v156_report.get("shadow_live_ab_ran") is not False:
        failures.append("v156_shadow_live_ab_ran_not_false")
    v156_source = _mapping(v156_report.get("source_validation"))
    if v156_source.get("passed") is not True:
        failures.append("v156_source_validation_failed")
    dataset_digest = stable_payload_digest(dataset_rows)
    if v156_source.get("dataset_digest") != dataset_digest:
        failures.append("v156_dataset_digest_mismatch")
    expected_policy_ids = [
        str(policy.get("policy_id")) for policy in candidate_feature_policies()
    ]
    observed_policy_ids = [
        str(report.get("policy_id"))
        for report in _list_of_mappings(v156_report.get("feature_policies"))
    ]
    if observed_policy_ids != expected_policy_ids:
        failures.append("v156_feature_policy_sequence_mismatch")
    branch_result_digest = stable_payload_digest(branch_results)
    expected_branch_result_digest = str(
        v154_report.get("continuation_branch_evidence_digest", "")
    )
    if expected_branch_result_digest and expected_branch_result_digest != branch_result_digest:
        failures.append("v154_continuation_branch_evidence_digest_mismatch")
    if not branch_results:
        failures.append("v154_continuation_branch_results_missing")
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v156_classification": EXPECTED_V156_CLASSIFICATION,
        "observed_v156_classification": observed_v156_classification,
        "v154_v155_source_validation": v154_v155_validation,
        "dataset_digest": dataset_digest,
        "label_count": len(dataset_rows),
        "continuation_branch_result_count": len(branch_results),
        "continuation_branch_evidence_digest": branch_result_digest,
        "v156_exact_digest": v156_report.get("exact_digest"),
    }


def evaluate_action_value_feature_policy(
    *,
    rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    branch_rows: Sequence[Mapping[str, object]],
    policy: Mapping[str, object],
    min_safe_runs_per_action: int = DEFAULT_MIN_SAFE_RUNS_PER_ACTION,
    min_safe_share_per_action: float = DEFAULT_MIN_SAFE_SHARE_PER_ACTION,
    unique_winner_score_margin: float = DEFAULT_UNIQUE_WINNER_SCORE_MARGIN,
) -> dict[str, object]:
    builder = policy.get("builder")
    if not callable(builder):
        raise CarrionSurvivorContinuationActionValueAuditError(
            "feature policy builder must be callable"
        )
    row_groups = _group_rows_by_feature_key(rows, builder)
    branch_groups = _group_branch_results_by_feature_key(
        branch_results=branch_results,
        branch_rows=branch_rows,
        builder=builder,
    )
    feature_payloads = [
        group["features"]
        for group in row_groups.values()
        if isinstance(group.get("features"), Mapping)
    ]
    leakage = hard_trainable_feature_leakage_scan(feature_payloads)
    groups = []
    all_group_digests = sorted(set(row_groups) | set(branch_groups))
    for digest in all_group_digests:
        group = _build_action_value_group_report(
            feature_digest=digest,
            row_group=_mapping(row_groups.get(digest)),
            branch_group=_mapping(branch_groups.get(digest)),
            min_safe_runs_per_action=min_safe_runs_per_action,
            min_safe_share_per_action=min_safe_share_per_action,
            unique_winner_score_margin=unique_winner_score_margin,
        )
        groups.append(group)
    classification_counts = Counter(
        str(group.get("classification")) for group in groups
    )
    single_label_conflicts = [
        group for group in groups if group.get("single_label_conflict") is True
    ]
    resolvable = [
        group
        for group in single_label_conflicts
        if group.get("set_or_action_value_target_resolves_conflict") is True
    ]
    unresolved = [
        group
        for group in single_label_conflicts
        if group.get("set_or_action_value_target_resolves_conflict") is not True
    ]
    conflicting_row_count = sum(_int(group.get("row_count")) for group in single_label_conflicts)
    resolvable_row_count = sum(_int(group.get("row_count")) for group in resolvable)
    unresolved_row_count = sum(_int(group.get("row_count")) for group in unresolved)
    action_value_reduction_share = _safe_rate(
        resolvable_row_count,
        conflicting_row_count,
    )
    return {
        "policy_id": policy.get("policy_id"),
        "description": policy.get("description"),
        "row_count": sum(_int(group.get("row_count")) for group in groups),
        "exact_public_feature_group_count": len(groups),
        "single_label_conflicting_group_count": len(single_label_conflicts),
        "single_label_conflicting_row_count": int(conflicting_row_count),
        "action_value_resolvable_conflicting_group_count": len(resolvable),
        "action_value_resolvable_conflicting_row_count": int(resolvable_row_count),
        "action_value_unresolved_conflicting_group_count": len(unresolved),
        "action_value_unresolved_conflicting_row_count": int(unresolved_row_count),
        "action_value_conflicting_row_reduction_share": action_value_reduction_share,
        "would_reduce_single_label_conflicting_rows": resolvable_row_count > 0,
        "classification_counts": dict(sorted(classification_counts.items())),
        "group_reports": groups,
        "leakage_scan": leakage,
        "min_safe_runs_per_action": int(min_safe_runs_per_action),
        "min_safe_share_per_action": _round(min_safe_share_per_action),
        "unique_winner_score_margin": _round(unique_winner_score_margin),
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
    }


def hard_trainable_feature_leakage_scan(
    feature_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    v156_scan = feature_leakage_scan(feature_payloads)
    hard_failures: list[dict[str, object]] = []
    for row_index, payload in enumerate(feature_payloads):
        _scan_hard_trainable_feature_value(
            value=payload,
            path=(),
            row_index=row_index,
            failures=hard_failures,
        )
    failures = [
        *_list_of_mappings(v156_scan.get("failures")),
        *hard_failures,
    ]
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_hard_trainable_feature_leakage_scan_v1",
        "passed": v156_scan.get("passed") is True and not hard_failures,
        "failure_count": len(failures),
        "failures": failures[:64],
        "v156_feature_leakage_scan": v156_scan,
        "hard_forbidden_key_tokens": list(FORBIDDEN_TRAINABLE_FEATURE_KEY_TOKENS),
        "hard_forbidden_string_markers": list(
            FORBIDDEN_TRAINABLE_FEATURE_STRING_MARKERS
        ),
    }


def _group_rows_by_feature_key(
    rows: Sequence[Mapping[str, object]],
    builder: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for row_index, row in enumerate(rows):
        features = dict(builder(row))
        digest = stable_payload_digest(features)
        group = groups.setdefault(
            digest,
            {
                "features": features,
                "rows": [],
            },
        )
        _list(group["rows"]).append(
            {
                "row_index": row_index,
                "dataset_row_index": _row_index(row, default=row_index),
                "label_action": _label_action(row),
                "seed": _row_seed(row),
            }
        )
    return groups


def _group_branch_results_by_feature_key(
    *,
    branch_results: Sequence[Mapping[str, object]],
    branch_rows: Sequence[Mapping[str, object]],
    builder: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for result_index, result in enumerate(branch_results):
        if result_index >= len(branch_rows):
            break
        features = dict(builder(branch_rows[result_index]))
        digest = stable_payload_digest(features)
        group = groups.setdefault(
            digest,
            {
                "features": features,
                "branch_results": [],
            },
        )
        _list(group["branch_results"]).append(
            {
                "source_branch_result_index": int(result_index),
                "branch_result": result,
            }
        )
    return groups


def _build_action_value_group_report(
    *,
    feature_digest: str,
    row_group: Mapping[str, object],
    branch_group: Mapping[str, object],
    min_safe_runs_per_action: int,
    min_safe_share_per_action: float,
    unique_winner_score_margin: float,
) -> dict[str, object]:
    rows = _list_of_mappings(row_group.get("rows"))
    branch_entries = _list_of_mappings(branch_group.get("branch_results"))
    label_counts = Counter(str(row.get("label_action", "")) for row in rows)
    label_counts.pop("", None)
    action_reports, coverage = _action_level_evidence(
        branch_entries=branch_entries,
        min_safe_runs_per_action=min_safe_runs_per_action,
        min_safe_share_per_action=min_safe_share_per_action,
    )
    classification, winner, safe_set = _classify_action_value_group(
        action_reports=action_reports,
        coverage=coverage,
        unique_winner_score_margin=unique_winner_score_margin,
    )
    single_label_conflict = len(label_counts) > 1
    resolves_conflict = single_label_conflict and classification in {
        "unique_robust_winner",
        "multi_action_safe_set",
    }
    return {
        "feature_digest": feature_digest,
        "row_count": len(rows),
        "branch_result_count": len(branch_entries),
        "label_action_counts": dict(sorted(label_counts.items())),
        "single_label_conflict": single_label_conflict,
        "classification": classification,
        "robust_winner_action": winner,
        "safe_action_set": safe_set,
        "set_or_action_value_target_resolves_conflict": resolves_conflict,
        "coverage": coverage,
        "action_values": action_reports,
        "row_indexes_sample": [
            _int(row.get("dataset_row_index")) for row in rows[:12]
        ],
    }


def _action_level_evidence(
    *,
    branch_entries: Sequence[Mapping[str, object]],
    min_safe_runs_per_action: int,
    min_safe_share_per_action: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    runs_by_action: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    expected_pairs: set[tuple[int, str, int]] = set()
    observed_pairs: set[tuple[int, str, int]] = set()
    expected_actions: set[str] = set()
    for entry in branch_entries:
        result_index = _int(entry.get("source_branch_result_index"))
        result = _mapping(entry.get("branch_result"))
        actions = _valid_public_mask_actions(result)
        indexes = _continuation_indexes(result)
        for action in actions:
            expected_actions.add(action)
            for index in indexes:
                expected_pairs.add((result_index, action, index))
        for run in _list_of_mappings(result.get("continuation_runs")):
            action = str(run.get("forced_action", ""))
            index = _int(run.get("continuation_index"), default=-1)
            if action in ACTION_NAMES:
                runs_by_action[action].append(run)
                observed_pairs.add((result_index, action, index))
    missing_pairs = sorted(expected_pairs - observed_pairs, key=_coverage_pair_sort_key)
    action_reports = [
        _summarize_action_runs(
            action=action,
            runs=runs_by_action.get(action, []),
            min_safe_runs_per_action=min_safe_runs_per_action,
            min_safe_share_per_action=min_safe_share_per_action,
        )
        for action in sorted(expected_actions | set(runs_by_action), key=_action_order)
    ]
    coverage_complete = bool(branch_entries) and bool(expected_actions) and not missing_pairs
    coverage = {
        "policy": "m3_carrion_survivor_continuation_action_value_group_action_coverage_v1",
        "complete": coverage_complete,
        "branch_result_count": len(branch_entries),
        "expected_action_count": len(expected_actions),
        "observed_action_count": len(
            {str(report.get("action")) for report in action_reports}
        ),
        "missing_action_continuation_count": len(missing_pairs),
        "missing_action_continuations_sample": [
            {
                "source_branch_result_index": result_index,
                "action": action,
                "continuation_index": index,
            }
            for result_index, action, index in missing_pairs[:16]
        ],
    }
    return action_reports, coverage


def _summarize_action_runs(
    *,
    action: str,
    runs: Sequence[Mapping[str, object]],
    min_safe_runs_per_action: int,
    min_safe_share_per_action: float,
) -> dict[str, object]:
    flags = [_run_outcome_flags(run) for run in runs]
    run_count = len(flags)
    survival = sum(1 for item in flags if item["survival_improved"])
    hydration = sum(1 for item in flags if item["hydration_recovery"])
    reproduction = sum(1 for item in flags if item["reproduction_readiness"])
    blockers = sum(1 for item in flags if item["blocker_reduction"])
    improved = sum(1 for item in flags if item["any_recovery"])
    resolved_risk = sum(1 for item in flags if item["resolved_invalid_risk"])
    unsupported_run_count = sum(
        1 for item in flags if _int(item["unsupported_requested_action_count"]) > 0
    )
    unsupported_total = sum(
        _int(item["unsupported_requested_action_count"]) for item in flags
    )
    safe = sum(1 for item in flags if item["safe_run"])
    score_sum = sum(_float(item["outcome_score"]) for item in flags)
    max_resolved_invalid_delta = max(
        (_float(item["resolved_invalid_delta"]) for item in flags),
        default=0.0,
    )
    max_unsupported_requested = max(
        (_int(item["unsupported_requested_action_count"]) for item in flags),
        default=0,
    )
    safe_share = _safe_rate(safe, run_count)
    mean_score = _round(score_sum / run_count) if run_count else 0.0
    robust = (
        run_count > 0
        and safe >= int(min_safe_runs_per_action)
        and safe_share >= float(min_safe_share_per_action)
        and resolved_risk == 0
        and unsupported_total == 0
    )
    return {
        "action": action,
        "continuation_run_count": run_count,
        "safe_run_count": safe,
        "safe_share": safe_share,
        "survival_improved_count": survival,
        "hydration_recovery_count": hydration,
        "reproduction_readiness_count": reproduction,
        "blocker_reduction_count": blockers,
        "any_recovery_count": improved,
        "resolved_invalid_risk_run_count": resolved_risk,
        "unsupported_requested_action_run_count": unsupported_run_count,
        "unsupported_requested_action_total": int(unsupported_total),
        "max_unsupported_requested_action_count": int(max_unsupported_requested),
        "max_resolved_invalid_delta": _round(max_resolved_invalid_delta),
        "mean_outcome_score": mean_score,
        "robust_safe_action": robust,
    }


def _run_outcome_flags(run: Mapping[str, object]) -> dict[str, object]:
    improvement = _mapping(run.get("outcome_improvement"))
    baseline = _mapping(run.get("deltas_vs_baseline"))
    first = _mapping(run.get("first_action_outcome"))
    first_outcome = _mapping(first.get("outcome"))
    replay = _mapping(run.get("replay_verification"))
    survival = bool(improvement.get("survival_improved")) or (
        _int(baseline.get("alive_agents")) > 0
        or _int(baseline.get("target_alive")) > 0
        or _int(baseline.get("deaths")) < 0
    )
    hydration = bool(improvement.get("hydration_recovery")) or (
        _optional_float(baseline.get("target_hydration_ratio")) is not None
        and _float(baseline.get("target_hydration_ratio")) > 0.0
    )
    reproduction = bool(improvement.get("reproduction_readiness")) or (
        _int(baseline.get("births")) > 0
        or first_outcome.get("reproduced") is True
        or first_outcome.get("reproduction_ready_after") is True
    )
    blocker = bool(improvement.get("fewer_blockers")) or (
        _int(baseline.get("unsupported_requested_action_count")) < 0
        or _int(baseline.get("unsupported_resolved_action_count")) < 0
        or _int(baseline.get("deaths")) < 0
    )
    resolved_invalid_delta = _optional_float(
        baseline.get("unsupported_resolved_action_count")
    )
    resolved_invalid_risk = (
        (resolved_invalid_delta is not None and resolved_invalid_delta > 0.0)
        or first_outcome.get("resolution_action_valid") is False
    )
    unsupported_requested = _int(run.get("unsupported_requested_action_count"))
    any_recovery = survival or hydration or reproduction or blocker
    outcome_score = (
        int(survival)
        + int(hydration)
        + int(reproduction)
        + int(blocker)
        - int(resolved_invalid_risk)
        - int(unsupported_requested)
    )
    safe = (
        run.get("forced_action_used") is True
        and run.get("forced_action_supported") is True
        and replay.get("verified") is True
        and _int(run.get("heuristic_action_source_count")) == 0
        and unsupported_requested == 0
        and not resolved_invalid_risk
        and any_recovery
    )
    return {
        "survival_improved": survival,
        "hydration_recovery": hydration,
        "reproduction_readiness": reproduction,
        "blocker_reduction": blocker,
        "any_recovery": any_recovery,
        "resolved_invalid_delta": 0.0
        if resolved_invalid_delta is None
        else _round(resolved_invalid_delta),
        "resolved_invalid_risk": resolved_invalid_risk,
        "unsupported_requested_action_count": int(unsupported_requested),
        "outcome_score": _round(outcome_score),
        "safe_run": safe,
    }


def _classify_action_value_group(
    *,
    action_reports: Sequence[Mapping[str, object]],
    coverage: Mapping[str, object],
    unique_winner_score_margin: float,
) -> tuple[str, str | None, list[str]]:
    if coverage.get("complete") is not True:
        return "insufficient_action_coverage", None, []
    robust = [
        report
        for report in action_reports
        if report.get("robust_safe_action") is True
    ]
    if not robust:
        return "conflicting_no_public_winner", None, []
    ranked_all = sorted(action_reports, key=_action_report_rank_key)
    ranked_robust = sorted(robust, key=_action_report_rank_key)
    top = ranked_robust[0]
    top_action = str(top.get("action", ""))
    top_score = _float(top.get("mean_outcome_score"))
    second_score = (
        _float(ranked_all[1].get("mean_outcome_score"))
        if len(ranked_all) > 1
        else -math.inf
    )
    robust_safe_set = [
        str(report.get("action", ""))
        for report in ranked_robust
        if str(report.get("action", ""))
    ]
    if len(robust) == 1:
        if (
            not math.isfinite(second_score)
            or top_score - second_score >= float(unique_winner_score_margin)
            or all(_int(report.get("safe_run_count")) == 0 for report in ranked_all[1:])
        ):
            return "unique_robust_winner", top_action, [top_action]
        return "conflicting_no_public_winner", None, robust_safe_set
    second_robust_score = _float(ranked_robust[1].get("mean_outcome_score"))
    if top_score - second_robust_score >= float(unique_winner_score_margin):
        return "unique_robust_winner", top_action, [top_action]
    return "multi_action_safe_set", None, robust_safe_set


def _conflict_reduction_summary(
    *,
    policy_reports: Sequence[Mapping[str, object]],
    best_policy: Mapping[str, object],
    v155_report: Mapping[str, object],
    v156_report: Mapping[str, object],
) -> dict[str, object]:
    v155_conflicting_rows = _observed_v155_conflicting_row_count(
        v155_report=v155_report,
        v156_report=v156_report,
    )
    best_resolved = _int(best_policy.get("action_value_resolvable_conflicting_row_count"))
    best_unresolved = _int(best_policy.get("action_value_unresolved_conflicting_row_count"))
    best_reduction_share = _safe_rate(best_resolved, v155_conflicting_rows)
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_conflict_reduction_v1",
        "observed_v155_single_label_conflicting_row_count": int(v155_conflicting_rows),
        "best_policy_id": best_policy.get("policy_id"),
        "best_policy_resolvable_conflicting_row_count": int(best_resolved),
        "best_policy_unresolved_conflicting_row_count": int(best_unresolved),
        "best_policy_conflicting_row_reduction_share": best_reduction_share,
        "would_reduce_119_conflicting_rows": (
            v155_conflicting_rows == 119 and best_resolved > 0
        ),
        "would_reduce_single_label_conflicting_rows": best_resolved > 0,
        "would_eliminate_all_single_label_conflicting_rows": (
            v155_conflicting_rows > 0 and best_resolved >= v155_conflicting_rows
        ),
        "policy_reductions": [
            {
                "policy_id": report.get("policy_id"),
                "single_label_conflicting_row_count": report.get(
                    "single_label_conflicting_row_count"
                ),
                "action_value_resolvable_conflicting_row_count": report.get(
                    "action_value_resolvable_conflicting_row_count"
                ),
                "action_value_unresolved_conflicting_row_count": report.get(
                    "action_value_unresolved_conflicting_row_count"
                ),
                "reduction_share": report.get(
                    "action_value_conflicting_row_reduction_share"
                ),
            }
            for report in policy_reports
        ],
    }


def _best_policy_report(
    policy_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not policy_reports:
        return {}
    report = max(
        policy_reports,
        key=lambda item: (
            _int(item.get("action_value_resolvable_conflicting_row_count")),
            -_int(item.get("action_value_unresolved_conflicting_row_count")),
            _float(item.get("action_value_conflicting_row_reduction_share")),
            str(item.get("policy_id", "")),
        ),
    )
    return {
        "policy_id": report.get("policy_id"),
        "single_label_conflicting_group_count": report.get(
            "single_label_conflicting_group_count"
        ),
        "single_label_conflicting_row_count": report.get(
            "single_label_conflicting_row_count"
        ),
        "action_value_resolvable_conflicting_group_count": report.get(
            "action_value_resolvable_conflicting_group_count"
        ),
        "action_value_resolvable_conflicting_row_count": report.get(
            "action_value_resolvable_conflicting_row_count"
        ),
        "action_value_unresolved_conflicting_group_count": report.get(
            "action_value_unresolved_conflicting_group_count"
        ),
        "action_value_unresolved_conflicting_row_count": report.get(
            "action_value_unresolved_conflicting_row_count"
        ),
        "action_value_conflicting_row_reduction_share": report.get(
            "action_value_conflicting_row_reduction_share"
        ),
        "classification_counts": report.get("classification_counts"),
        "leakage_scan": report.get("leakage_scan"),
        "would_reduce_single_label_conflicting_rows": report.get(
            "would_reduce_single_label_conflicting_rows"
        ),
    }


def _top_level_classification(
    *,
    source_validation: Mapping[str, object],
    leakage_failures: Sequence[Mapping[str, object]],
    conflict_reduction: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True:
        return (
            "m3_carrion_survivor_continuation_public_state_action_value_audit_"
            "source_invalid_closed_no_training"
        )
    if leakage_failures:
        return (
            "m3_carrion_survivor_continuation_public_state_action_value_audit_"
            "leakage_failed_closed_no_training"
        )
    if conflict_reduction.get("would_reduce_single_label_conflicting_rows") is True:
        return (
            "m3_carrion_survivor_continuation_public_state_action_value_audit_"
            "set_or_action_value_targets_reduce_conflicts_no_training"
        )
    return (
        "m3_carrion_survivor_continuation_public_state_action_value_audit_"
        "public_state_ambiguity_remains_no_training"
    )


def _route_recommendation(classification: str) -> dict[str, object]:
    reduces = classification.endswith("set_or_action_value_targets_reduce_conflicts_no_training")
    return {
        "policy": "m3_carrion_survivor_continuation_action_value_route_recommendation_v1",
        "single_label_imitation_replacement_worth_review": bool(reduces),
        "recommended_next_route": (
            "review_set_valued_or_action_value_targets_as_a_diagnostics_only_design"
            if reduces
            else "public_state_ambiguity_or_source_issue_remains_closed_no_training"
        ),
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
    }


def _observed_v155_conflicting_row_count(
    *,
    v155_report: Mapping[str, object],
    v156_report: Mapping[str, object],
) -> int:
    pretraining = _mapping(v155_report.get("pretraining_review"))
    alias = _mapping(pretraining.get("nearest_neighbor_alias_collision_audit"))
    count = _int(alias.get("conflicting_exact_feature_row_count"), default=-1)
    if count >= 0:
        return count
    policies = _list_of_mappings(v156_report.get("feature_policies"))
    if policies:
        return _int(policies[0].get("conflicting_row_count"))
    return 0


def _branch_proxy_row(result: Mapping[str, object]) -> dict[str, object]:
    public_features = _mapping(result.get("public_features"))
    return {
        "metadata": {
            "branch_id": result.get("branch_id"),
            "seed": result.get("seed"),
        },
        "trainable": {
            "features": {
                "observation_input": deepcopy(
                    _mapping(public_features.get("observation_input"))
                ),
                "action_mask": deepcopy(_mapping(public_features.get("action_mask"))),
                "prior_public_context": [],
            },
        },
    }


def _scan_hard_trainable_feature_value(
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
            if any(token in lowered for token in FORBIDDEN_TRAINABLE_FEATURE_KEY_TOKENS):
                failures.append(
                    {
                        "row_index": row_index,
                        "path": ".".join((*path, key)),
                        "reason": "forbidden_trainable_feature_key_token",
                        "key": key,
                    }
                )
            _scan_hard_trainable_feature_value(
                value=child,
                path=(*path, key),
                row_index=row_index,
                failures=failures,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_hard_trainable_feature_value(
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
        if any(marker in lowered for marker in FORBIDDEN_TRAINABLE_FEATURE_STRING_MARKERS):
            failures.append(
                {
                    "row_index": row_index,
                    "path": ".".join(path),
                    "reason": "forbidden_trainable_feature_string_marker",
                    "value": value[:128],
                }
            )


def _valid_public_mask_actions(result: Mapping[str, object]) -> list[str]:
    public_features = _mapping(result.get("public_features"))
    mask = _mapping(public_features.get("action_mask"))
    valid = [action for action in ACTION_NAMES if bool(mask.get(action, False))]
    if valid:
        return valid
    return [
        str(action)
        for action in _list(result.get("candidate_actions"))
        if str(action) in ACTION_NAMES
    ]


def _continuation_indexes(result: Mapping[str, object]) -> list[int]:
    indexes = [
        _int(index)
        for index in _list(result.get("continuation_indexes"))
        if _int(index, default=-1) >= 0
    ]
    if indexes:
        return sorted(set(indexes))
    observed = [
        _int(run.get("continuation_index"))
        for run in _list_of_mappings(result.get("continuation_runs"))
        if _int(run.get("continuation_index"), default=-1) >= 0
    ]
    return sorted(set(observed))


def _action_report_rank_key(report: Mapping[str, object]) -> tuple[float, int, float, int]:
    action = str(report.get("action", ""))
    return (
        -_float(report.get("mean_outcome_score")),
        -_int(report.get("safe_run_count")),
        _float(report.get("resolved_invalid_risk_run_count")),
        _action_order(action),
    )


def _coverage_pair_sort_key(item: tuple[int, str, int]) -> tuple[int, int, int]:
    result_index, action, continuation_index = item
    return (int(result_index), _action_order(action), int(continuation_index))


def _action_order(action: str) -> int:
    try:
        return ACTION_NAMES.index(action)
    except ValueError:
        return len(ACTION_NAMES)


def _label_action(row: Mapping[str, object]) -> str:
    return str(_mapping(_mapping(row.get("trainable")).get("label")).get("action", ""))


def _row_seed(row: Mapping[str, object]) -> int:
    return _int(_mapping(row.get("metadata")).get("seed"))


def _row_index(row: Mapping[str, object], *, default: int) -> int:
    metadata = _mapping(row.get("metadata"))
    return _int(metadata.get("row_index"), default=default)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    denominator_float = float(denominator)
    if denominator_float <= 0.0:
        return 0.0
    return _round(float(numerator) / denominator_float)
