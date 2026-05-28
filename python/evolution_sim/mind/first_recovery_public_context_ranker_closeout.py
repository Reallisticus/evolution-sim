from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_candidate_ranker_capacity_audit import (
    _list_like,
)
from evolution_sim.mind.first_recovery_history_refreshed_surface_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V134_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_PROBE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_rollout_history_context_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V133_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _mapping,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_refreshed_candidate_public_feature_surface import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V131_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_refreshed_surface_blocker_slice_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V132_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_SCHEMA_VERSION,
)

MIND_V3_FIRST_RECOVERY_PUBLIC_CONTEXT_RANKER_CLOSEOUT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_public_context_ranker_closeout_v1"
)
MIND_V3_FIRST_RECOVERY_PUBLIC_CONTEXT_RANKER_CLOSEOUT_POLICY = (
    "diagnostics_only_first_recovery_v135_public_context_ranker_closeout_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v135-first-recovery-public-context-ranker-closeout.json"
)

CLASSIFICATION = "first_recovery_public_context_ranker_path_closed_not_promotable"
FINAL_RECOMMENDATION = "stop_first_recovery_public_context_ranker_path"
EXPECTED_V132_RECOMMENDATION = "add_public_rollout_history_context"
EXPECTED_V133_RECOMMENDATION = (
    "public_rollout_history_context_ready_for_refreshed_surface"
)
EXPECTED_V134_CLASSIFICATION = "history_refreshed_surface_blocked_by_signal"


@dataclass(frozen=True, slots=True)
class FirstRecoveryPublicContextRankerCloseoutBuild:
    report: dict[str, object]


def build_first_recovery_public_context_ranker_closeout(
    *,
    v131_report: Mapping[str, object] | None = None,
    v131_report_path: str | Path | None = DEFAULT_V131_REPORT_PATH,
    v132_report: Mapping[str, object] | None = None,
    v132_report_path: str | Path | None = DEFAULT_V132_REPORT_PATH,
    v133_report: Mapping[str, object] | None = None,
    v133_report_path: str | Path | None = DEFAULT_V133_REPORT_PATH,
    v134_report: Mapping[str, object] | None = None,
    v134_report_path: str | Path | None = DEFAULT_V134_REPORT_PATH,
) -> FirstRecoveryPublicContextRankerCloseoutBuild:
    v131_payload, v131_evidence = _resolve_json_report(
        v131_report,
        v131_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION
        ),
    )
    v132_payload, v132_evidence = _resolve_json_report(
        v132_report,
        v132_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_SCHEMA_VERSION
        ),
    )
    v133_payload, v133_evidence = _resolve_json_report(
        v133_report,
        v133_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION
        ),
    )
    v134_payload, v134_evidence = _resolve_json_report(
        v134_report,
        v134_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_PROBE_SCHEMA_VERSION
        ),
    )
    source_reports = {
        "v131_report": v131_evidence,
        "v132_report": v132_evidence,
        "v133_report": v133_evidence,
        "v134_report": v134_evidence,
    }
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v131_report=v131_payload,
        v132_report=v132_payload,
        v133_report=v133_payload,
        v134_report=v134_payload,
    )
    decision_chain = _decision_chain(
        v131_report=v131_payload,
        v132_report=v132_payload,
        v133_report=v133_payload,
        v134_report=v134_payload,
    )
    terminal_signal_summary = _terminal_signal_summary(v134_payload)
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_PUBLIC_CONTEXT_RANKER_CLOSEOUT_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_PUBLIC_CONTEXT_RANKER_CLOSEOUT_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "decision_chain": decision_chain,
        "terminal_signal_summary": terminal_signal_summary,
        "path_closeout_decision": _path_closeout_decision(
            source_integrity=source_integrity,
            terminal_signal_summary=terminal_signal_summary,
        ),
        "pivot_recommendations": _pivot_recommendations(),
        "recommendation": _recommendation(),
        "classification": _classification(),
        "authorization_block": _authorization_block(),
        "non_promoted": True,
    }
    return FirstRecoveryPublicContextRankerCloseoutBuild(report=report)


def write_first_recovery_public_context_ranker_closeout_report(
    build: FirstRecoveryPublicContextRankerCloseoutBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "closeout_report_only": True,
        "first_recovery_feature_probe_created": False,
        "v136_first_recovery_feature_diagnostic_recommended": False,
        "training_executed": False,
        "training_authorized": False,
        "model_artifact_created": False,
        "runtime_loadable_artifact_created": False,
        "shadow_scorer_execution_authorized": False,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "runtime_policy_effect": "none",
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "foundation_effect": "none",
        "claim_causality": False,
    }


def _source_integrity(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    v131_report: Mapping[str, object] | None,
    v132_report: Mapping[str, object] | None,
    v133_report: Mapping[str, object] | None,
    v134_report: Mapping[str, object] | None,
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")

    v131_source = _mapping(_mapping(v131_report or {}).get("source_integrity"))
    v132_source = _mapping(_mapping(v132_report or {}).get("source_integrity"))
    v133_source = _mapping(_mapping(v133_report or {}).get("source_integrity"))
    v134_source = _mapping(_mapping(v134_report or {}).get("source_integrity"))
    v132_recommendation = _mapping(_mapping(v132_report or {}).get("recommendation"))
    v133_recommendation = _mapping(_mapping(v133_report or {}).get("recommendation"))
    v134_classification = _mapping(_mapping(v134_report or {}).get("classification"))
    v134_leakage = _mapping(_mapping(v134_report or {}).get("leakage_audit"))

    if v132_source.get("passed") is not True:
        failures.append("v132_source_integrity_not_passed")
    if v133_source.get("passed") is not True:
        failures.append("v133_source_integrity_not_passed")
    if v134_source.get("passed") is not True:
        failures.append("v134_source_integrity_not_passed")
    if v132_recommendation.get("next_step") != EXPECTED_V132_RECOMMENDATION:
        failures.append("v132_recommendation_contract_mismatch")
    if v133_recommendation.get("next_step") != EXPECTED_V133_RECOMMENDATION:
        failures.append("v133_recommendation_contract_mismatch")
    if v134_classification.get("primary") != EXPECTED_V134_CLASSIFICATION:
        failures.append("v134_not_blocked_by_signal")
    if _number(v134_leakage.get("leakage_count")) != 0:
        failures.append("v134_leakage_nonzero")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v131_source_integrity_passed": v131_source.get("passed"),
        "v132_source_integrity_passed": v132_source.get("passed"),
        "v133_source_integrity_passed": v133_source.get("passed"),
        "v134_source_integrity_passed": v134_source.get("passed"),
        "v132_recommendation": v132_recommendation.get("next_step"),
        "required_v132_recommendation": EXPECTED_V132_RECOMMENDATION,
        "v133_recommendation": v133_recommendation.get("next_step"),
        "required_v133_recommendation": EXPECTED_V133_RECOMMENDATION,
        "v134_classification": v134_classification.get("primary"),
        "required_v134_classification": EXPECTED_V134_CLASSIFICATION,
        "v134_leakage_count": _number(v134_leakage.get("leakage_count")),
    }


def _decision_chain(
    *,
    v131_report: Mapping[str, object] | None,
    v132_report: Mapping[str, object] | None,
    v133_report: Mapping[str, object] | None,
    v134_report: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    return [
        _v131_decision_step(v131_report),
        _v132_decision_step(v132_report),
        _v133_decision_step(v133_report),
        _v134_decision_step(v134_report),
    ]


def _v131_decision_step(report: Mapping[str, object] | None) -> dict[str, object]:
    metric_gate = _mapping(_mapping(report or {}).get("metric_gate"))
    probe = _mapping(_mapping(report or {}).get("probe_comparison"))
    fixture_open = _fixture_open_dominant(report)
    return {
        "version": "v131",
        "conclusion": "blocked_by_heldout_action_only_tie_and_fixture_open_collapse",
        "classification": _classification_primary(report),
        "metric_failures": _list_like(metric_gate.get("failures")),
        "heldout_accuracy": _number(probe.get("heldout_accuracy")),
        "heldout_action_only_accuracy": _number(
            probe.get("heldout_action_only_accuracy")
        ),
        "heldout_action_order_accuracy": _number(
            probe.get("heldout_action_order_accuracy")
        ),
        "fixture_open_dominant_predicted_action": fixture_open,
        "claim_causality": False,
    }


def _v132_decision_step(report: Mapping[str, object] | None) -> dict[str, object]:
    aggregate = _mapping(_mapping(report or {}).get("blocker_class_aggregate"))
    counts = dict(_mapping(aggregate.get("counts_by_failed_blocker_class")))
    dominant = _dominant_count(counts)
    return {
        "version": "v132",
        "conclusion": "fixture_open_action_collapse_identified_as_dominant_failed_blocker",
        "classification": _classification_primary(report),
        "recommendation": _mapping(_mapping(report or {}).get("recommendation")).get(
            "next_step"
        ),
        "failed_blocker_class_counts": counts,
        "dominant_failed_blocker_class": dominant,
        "claim_causality": False,
    }


def _v133_decision_step(report: Mapping[str, object] | None) -> dict[str, object]:
    availability = _mapping(_mapping(report or {}).get("public_history_availability"))
    open_analysis = _mapping(
        _mapping(report or {}).get("open_fixture_collapse_analysis")
    )
    collapse_variance = _mapping(open_analysis.get("collapse_variance"))
    failed_heldout = _mapping(_mapping(report or {}).get("heldout_failure_analysis"))
    failed_heldout_variance = _mapping(failed_heldout.get("failed_heldout_variance"))
    leakage = _mapping(_mapping(report or {}).get("leakage_audit"))
    return {
        "version": "v133",
        "conclusion": "public_history_context_available_clean_and_variable",
        "classification": _classification_primary(report),
        "recommendation": _mapping(_mapping(report or {}).get("recommendation")).get(
            "next_step"
        ),
        "history_row_count": _number(availability.get("history_row_count")),
        "history_available_row_count": _number(
            availability.get("history_available_row_count")
        ),
        "missing_public_history_field_count": _number(
            availability.get("missing_public_history_field_count")
        ),
        "leakage_count": _number(leakage.get("leakage_count")),
        "fixture_open_collapse_row_count": _number(
            open_analysis.get("fixture_open_collapse_row_count")
        ),
        "fixture_open_collapse_history_available_row_count": _number(
            open_analysis.get("collapse_history_available_row_count")
        ),
        "fixture_open_collapse_varying_history_path_count": _number(
            collapse_variance.get("varying_feature_path_count")
        ),
        "failed_heldout_varying_history_path_count": _number(
            failed_heldout_variance.get("varying_feature_path_count")
        ),
        "claim_causality": False,
    }


def _v134_decision_step(report: Mapping[str, object] | None) -> dict[str, object]:
    summary = _terminal_signal_summary(report)
    return {
        "version": "v134",
        "conclusion": "public_history_context_tested_but_path_remained_blocked_by_signal",
        "classification": _classification_primary(report),
        "metric_failures": _list_like(
            _mapping(_mapping(report or {}).get("metric_gate")).get("failures")
        ),
        "heldout_accuracy": summary["heldout_accuracy"],
        "heldout_action_only_accuracy": summary["heldout_action_only_accuracy"],
        "heldout_delta_vs_action_only": summary["heldout_delta_vs_action_only"],
        "heldout_action_order_accuracy": summary["heldout_action_order_accuracy"],
        "fixture_open_dominant_predicted_action": summary[
            "fixture_open_dominant_predicted_action"
        ],
        "seed29_passed": summary["seed29_passed"],
        "seed29_accuracy": summary["seed29_accuracy"],
        "overall_dominant_predicted_action": summary[
            "overall_dominant_predicted_action"
        ],
        "claim_causality": False,
    }


def _terminal_signal_summary(report: Mapping[str, object] | None) -> dict[str, object]:
    probe = _mapping(_mapping(report or {}).get("probe_comparison"))
    fixture_open = _fixture_open_dominant(report)
    seed29 = _mapping(_mapping(report or {}).get("seed29_evaluation"))
    seed29_metrics = _mapping(seed29.get("metrics"))
    action_distribution = _mapping(_mapping(report or {}).get("action_distribution"))
    material = _mapping(_mapping(report or {}).get("material_exact_match_recall"))
    unsupported = _mapping(_mapping(report or {}).get("unsupported_action_audit"))
    leakage = _mapping(_mapping(report or {}).get("leakage_audit"))
    return {
        "v134_classification": _classification_primary(report),
        "heldout_accuracy": _number(probe.get("heldout_accuracy")),
        "heldout_action_only_accuracy": _number(
            probe.get("heldout_action_only_accuracy")
        ),
        "heldout_delta_vs_action_only": _number(
            probe.get("heldout_delta_vs_action_only")
        ),
        "heldout_action_order_accuracy": _number(
            probe.get("heldout_action_order_accuracy")
        ),
        "heldout_delta_vs_action_order": _number(
            probe.get("heldout_delta_vs_action_order")
        ),
        "fixture_open_dominant_predicted_action": fixture_open,
        "fixture_open_collapse_improved_to_threshold": (
            _number(fixture_open.get("share")) is not None
            and _number(fixture_open.get("share")) <= 0.5
        ),
        "seed29_passed": seed29.get("passed"),
        "seed29_accuracy": _number(seed29_metrics.get("accuracy")),
        "overall_dominant_predicted_action": _mapping(
            action_distribution.get("dominant_predicted_action")
        ),
        "unsupported_action_count": _number(
            unsupported.get("unsupported_action_count")
        ),
        "unsupported_action_rate": _number(unsupported.get("unsupported_action_rate")),
        "material_exact_match_recall": _number(
            material.get("repaired_label_material_gain_exact_match_recall")
        ),
        "leakage_count": _number(leakage.get("leakage_count")),
        "decisive_signal_gates_failed": [
            "heldout_signal_not_above_action_only_baseline",
            "seed29_failed",
            "overall_dominant_predicted_action_share_above_threshold",
        ],
        "claim_causality": False,
    }


def _path_closeout_decision(
    *,
    source_integrity: Mapping[str, object],
    terminal_signal_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "closed": True,
        "promotable": False,
        "source_integrity_passed": source_integrity.get("passed"),
        "reason": "v134_history_refreshed_probe_failed_decisive_signal_gates",
        "no_additional_first_recovery_feature_probe": True,
        "terminal_evidence": {
            "heldout_delta_vs_action_only": terminal_signal_summary.get(
                "heldout_delta_vs_action_only"
            ),
            "seed29_passed": terminal_signal_summary.get("seed29_passed"),
            "overall_dominant_predicted_action": terminal_signal_summary.get(
                "overall_dominant_predicted_action"
            ),
        },
        "claim_causality": False,
    }


def _pivot_recommendations() -> dict[str, object]:
    options = [
        "broader_sequence_world_model_controller_path",
        "rollout_level_policy_capacity_beyond_first_recovery_branch_ranking",
        "collect_new_task_family_only_if_tied_to_new_controller_design",
        "package_merge_current_diagnostics_before_more_research_work",
    ]
    return {
        "non_authorizing": True,
        "options": [
            {
                "name": option,
                "authorized_by_this_report": False,
                "runtime_policy_change_authorized": False,
                "training_authorized": False,
                "shadow_scorer_authorized": False,
            }
            for option in options
        ],
    }


def _recommendation() -> dict[str, object]:
    return {
        "next_step": FINAL_RECOMMENDATION,
        "stop_current_diagnostic_batch": True,
        "first_recovery_public_context_ranker_path_promotable": False,
        "no_v136_first_recovery_feature_probe": True,
        "training_executed": False,
        "training_authorized": False,
        "shadow_scorer_execution_authorized": False,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "runtime_policy_change_recommended": False,
        "gate_change_recommended": False,
        "viewer_change_recommended": False,
        "replay_golden_change_recommended": False,
        "foundation_change_recommended": False,
        "claim_causality": False,
    }


def _classification() -> dict[str, object]:
    return {
        "primary": CLASSIFICATION,
        "labels": [CLASSIFICATION],
        "allowed_classifications": [CLASSIFICATION],
    }


def _authorization_block() -> dict[str, object]:
    return {
        "training_authorized": False,
        "training_executed": False,
        "model_artifact_created": False,
        "runtime_loadable_artifact_created": False,
        "shadow_scorer_execution_authorized": False,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "runtime_policy_change_authorized": False,
        "runtime_policy_change_recommended": False,
        "gate_change_authorized": False,
        "viewer_change_authorized": False,
        "replay_golden_change_authorized": False,
        "foundation_change_authorized": False,
        "claim_causality": False,
    }


def _fixture_open_dominant(
    report: Mapping[str, object] | None,
) -> dict[str, object]:
    fixture_open = _mapping(_mapping(report or {}).get("fixture_open_evaluation"))
    groups = _mapping(fixture_open.get("groups"))
    open_group = _mapping(groups.get("open_mind_v3"))
    return dict(_mapping(open_group.get("dominant_predicted_action")))


def _classification_primary(report: Mapping[str, object] | None) -> object:
    return _mapping(_mapping(report or {}).get("classification")).get("primary")


def _dominant_count(counts: Mapping[str, object]) -> dict[str, object]:
    if not counts:
        return {}
    key, value = max(
        ((str(key), _number(value) or 0) for key, value in counts.items()),
        key=lambda item: (item[1], item[0]),
    )
    total = sum(_number(value) or 0 for value in counts.values())
    return {
        "blocker_class": key,
        "count": value,
        "share": value / total if total else 0.0,
        "total": total,
    }


def _number(value: object) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return None
