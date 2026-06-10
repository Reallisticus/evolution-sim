from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_history_refreshed_surface_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V134_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_PROBE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_context_ranker_closeout import (
    CLASSIFICATION as V135_CLOSED_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V135_REPORT_PATH,
    FINAL_RECOMMENDATION as V135_FINAL_RECOMMENDATION,
    MIND_V3_FIRST_RECOVERY_PUBLIC_CONTEXT_RANKER_CLOSEOUT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_rollout_history_context_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V133_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
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
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_CURRENT_ROUTE_DECISION_SCHEMA_VERSION = (
    "mind_v3_v136_current_route_decision_v1"
)
MIND_V3_CURRENT_ROUTE_DECISION_POLICY = (
    "diagnostics_only_mind_v3_v136_current_route_decision_v1"
)

DEFAULT_OUTPUT_PATH = Path("output/mind/mind-v3-v136-current-route-decision.json")

LATEST_EVIDENCE_VERSION = 135
CURRENT_CLOSED_PATH = V135_CLOSED_PATH
NEXT_ALLOWED_RESEARCH_DIRECTION = "rollout_level_sequence_or_world_model_diagnostic"

EXPECTED_V135_TERMINAL_METRICS: dict[str, object] = {
    "heldout_accuracy": 0.2222222222222222,
    "heldout_action_only_accuracy": 0.2222222222222222,
    "heldout_action_order_accuracy": 0.3888888888888889,
    "seed29_accuracy": 0.0,
    "dominant_predicted_action": {
        "action": "move_north",
        "count": 14,
        "share": 0.56,
        "total": 25,
    },
    "leakage_count": 0,
    "unsupported_action_count": 0,
}

CLOSED_FAMILIES = [
    "v61-v63 scalar/current-IQL family",
    "v64-v65 one-row context/hydration probes",
    "v94-v104 runtime-transfer residual/planner failures",
    "v131-v135 first-recovery public-context ranker",
]

NEXT_DISALLOWED_ACTIONS = [
    "scalar IQL tuning",
    "prior-blend tuning",
    "global actor-bias calibration",
    "another v136 first-recovery feature probe",
    "runtime promotion",
]

DISALLOWED_POLICY_INPUTS = [
    "private world state",
    "fixture identity",
    "seed identity",
    "future rows",
    "heuristic recommendation",
]


@dataclass(frozen=True, slots=True)
class MindV3CurrentRouteDecisionBuild:
    report: dict[str, object]


def build_mind_v3_current_route_decision_report(
    *,
    v131_report: Mapping[str, object] | None = None,
    v131_report_path: str | Path | None = DEFAULT_V131_REPORT_PATH,
    v132_report: Mapping[str, object] | None = None,
    v132_report_path: str | Path | None = DEFAULT_V132_REPORT_PATH,
    v133_report: Mapping[str, object] | None = None,
    v133_report_path: str | Path | None = DEFAULT_V133_REPORT_PATH,
    v134_report: Mapping[str, object] | None = None,
    v134_report_path: str | Path | None = DEFAULT_V134_REPORT_PATH,
    v135_report: Mapping[str, object] | None = None,
    v135_report_path: str | Path | None = DEFAULT_V135_REPORT_PATH,
) -> MindV3CurrentRouteDecisionBuild:
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
    v135_payload, v135_evidence = _resolve_json_report(
        v135_report,
        v135_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_PUBLIC_CONTEXT_RANKER_CLOSEOUT_SCHEMA_VERSION
        ),
    )

    source_reports = {
        "v131_report": v131_evidence,
        "v132_report": v132_evidence,
        "v133_report": v133_evidence,
        "v134_report": v134_evidence,
        "v135_report": v135_evidence,
    }
    source_payloads = {
        "v131_report": v131_payload,
        "v132_report": v132_payload,
        "v133_report": v133_payload,
        "v134_report": v134_payload,
        "v135_report": v135_payload,
    }
    v135_terminal_metrics = _v135_terminal_metrics(v135_payload)
    v135_authorization = _v135_authorization(v135_payload)
    source_integrity = _source_integrity(
        source_reports=source_reports,
        source_payloads=source_payloads,
        v135_terminal_metrics=v135_terminal_metrics,
        v135_authorization=v135_authorization,
    )
    report = {
        "schema_version": MIND_V3_CURRENT_ROUTE_DECISION_SCHEMA_VERSION,
        "audit_policy": MIND_V3_CURRENT_ROUTE_DECISION_POLICY,
        "latest_evidence_version": LATEST_EVIDENCE_VERSION,
        "current_closed_path": CURRENT_CLOSED_PATH,
        "diagnostics_only": True,
        "report_only": True,
        "non_promoted": True,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "v106_v135_reconciliation": _v106_v135_reconciliation(),
        "v135_source_integrity_and_terminal_metrics": v135_terminal_metrics,
        "closed_families": list(CLOSED_FAMILIES),
        "closed_family_decisions": _closed_family_decisions(),
        "next_allowed_research_direction": NEXT_ALLOWED_RESEARCH_DIRECTION,
        "next_disallowed_actions": list(NEXT_DISALLOWED_ACTIONS),
        "decision": _decision(source_integrity=source_integrity),
        "authorization_block": _authorization_block(
            v135_authorization=v135_authorization
        ),
        "policy_input_boundary": _policy_input_boundary(),
    }
    report["decision_digest"] = stable_payload_digest(
        {
            "latest_evidence_version": report["latest_evidence_version"],
            "current_closed_path": report["current_closed_path"],
            "source_integrity": report["source_integrity"],
            "v135_source_integrity_and_terminal_metrics": report[
                "v135_source_integrity_and_terminal_metrics"
            ],
            "closed_families": report["closed_families"],
            "next_allowed_research_direction": report[
                "next_allowed_research_direction"
            ],
            "next_disallowed_actions": report["next_disallowed_actions"],
            "authorization_block": report["authorization_block"],
        }
    )
    return MindV3CurrentRouteDecisionBuild(report=report)


def write_mind_v3_current_route_decision_report(
    build: MindV3CurrentRouteDecisionBuild,
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
        "current_route_decision_report_only": True,
        "runtime_policy_effect": "none",
        "trainer_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "viewer_effect": "none",
        "foundation_effect": "none",
        "training_executed": False,
        "training_authorized": False,
        "runtime_policy_change_authorized": False,
        "shadow_scorer_execution_authorized": False,
        "model_artifact_created": False,
        "runtime_loadable_artifact_created": False,
        "v136_first_recovery_feature_probe_created": False,
        "v136_first_recovery_feature_probe_authorized": False,
        "claim_causality": False,
    }


def _source_integrity(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    source_payloads: Mapping[str, Mapping[str, object] | None],
    v135_terminal_metrics: Mapping[str, object],
    v135_authorization: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is not True:
            failures.append(f"{name}_schema_mismatch")

    for name in ("v131_report", "v132_report", "v133_report", "v134_report"):
        source = _mapping(_mapping(source_payloads.get(name)).get("source_integrity"))
        if source.get("passed") is not True:
            failures.append(f"{name}_source_integrity_not_passed")

    v135_payload = source_payloads.get("v135_report")
    v135_source = _mapping(_mapping(v135_payload).get("source_integrity"))
    v135_classification = _mapping(_mapping(v135_payload).get("classification"))
    v135_recommendation = _mapping(_mapping(v135_payload).get("recommendation"))
    v135_path_closeout = _mapping(_mapping(v135_payload).get("path_closeout_decision"))

    if v135_source.get("passed") is not True:
        failures.append("v135_source_integrity_not_passed")
    if v135_classification.get("primary") != CURRENT_CLOSED_PATH:
        failures.append("v135_closed_path_classification_mismatch")
    if v135_recommendation.get("next_step") != V135_FINAL_RECOMMENDATION:
        failures.append("v135_final_recommendation_mismatch")
    if v135_recommendation.get("no_v136_first_recovery_feature_probe") is not True:
        failures.append("v135_does_not_block_v136_feature_probe")
    if v135_path_closeout.get("closed") is not True:
        failures.append("v135_path_not_closed")
    if v135_path_closeout.get("promotable") is not False:
        failures.append("v135_path_promotable")
    if v135_path_closeout.get("no_additional_first_recovery_feature_probe") is not True:
        failures.append("v135_allows_additional_first_recovery_feature_probe")
    if not _v135_terminal_metrics_match_expected(v135_terminal_metrics):
        failures.append("v135_terminal_metrics_mismatch")
    if not _authorization_matches_expected_false(v135_authorization):
        failures.append("v135_authorization_block_mismatch")

    digest_match = _v135_embedded_source_digests_match(
        source_reports=source_reports,
        v135_report=v135_payload,
    )
    if digest_match is False:
        failures.append("v135_embedded_source_digest_mismatch")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "loaded_report_count": sum(
            1 for evidence in source_reports.values() if evidence.get("loaded") is True
        ),
        "required_report_count": len(source_reports),
        "v135_source_integrity_passed": v135_source.get("passed"),
        "v135_closed_path_classification": v135_classification.get("primary"),
        "required_current_closed_path": CURRENT_CLOSED_PATH,
        "v135_final_recommendation": v135_recommendation.get("next_step"),
        "required_v135_final_recommendation": V135_FINAL_RECOMMENDATION,
        "v135_path_closed": v135_path_closeout.get("closed"),
        "v135_path_promotable": v135_path_closeout.get("promotable"),
        "v135_terminal_metrics_match_expected": (
            _v135_terminal_metrics_match_expected(v135_terminal_metrics)
        ),
        "v135_authorization_block_match_expected": (
            _authorization_matches_expected_false(v135_authorization)
        ),
        "v135_embedded_source_digests_match": digest_match,
    }


def _v135_terminal_metrics(
    v135_report: Mapping[str, object] | None,
) -> dict[str, object]:
    terminal = _mapping(_mapping(v135_report).get("terminal_signal_summary"))
    dominant = _mapping(terminal.get("overall_dominant_predicted_action"))
    metrics = {
        "source_integrity_passed": _mapping(
            _mapping(v135_report).get("source_integrity")
        ).get("passed"),
        "classification": _mapping(
            _mapping(v135_report).get("classification")
        ).get("primary"),
        "heldout_accuracy": _number(terminal.get("heldout_accuracy")),
        "heldout_action_only_accuracy": _number(
            terminal.get("heldout_action_only_accuracy")
        ),
        "heldout_action_order_accuracy": _number(
            terminal.get("heldout_action_order_accuracy")
        ),
        "seed29_accuracy": _number(terminal.get("seed29_accuracy")),
        "dominant_predicted_action": dict(dominant),
        "dominant_predicted_action_name": dominant.get("action"),
        "dominant_predicted_action_share": _number(dominant.get("share")),
        "leakage_count": _number(terminal.get("leakage_count")),
        "unsupported_action_count": _number(terminal.get("unsupported_action_count")),
        "expected": EXPECTED_V135_TERMINAL_METRICS,
    }
    metrics["matches_expected"] = _v135_terminal_metrics_match_expected(metrics)
    return metrics


def _v135_authorization(v135_report: Mapping[str, object] | None) -> dict[str, object]:
    auth = _mapping(_mapping(v135_report).get("authorization_block"))
    return {
        "training_authorized": auth.get("training_authorized"),
        "runtime_policy_change_authorized": auth.get(
            "runtime_policy_change_authorized"
        ),
        "shadow_scorer_execution_authorized": auth.get(
            "shadow_scorer_execution_authorized"
        ),
        "downstream_shadow_scorer_allowed": auth.get(
            "downstream_shadow_scorer_allowed"
        ),
        "v113_readiness_rerun_allowed": auth.get("v113_readiness_rerun_allowed"),
        "gate_change_authorized": auth.get("gate_change_authorized"),
        "viewer_change_authorized": auth.get("viewer_change_authorized"),
        "replay_golden_change_authorized": auth.get(
            "replay_golden_change_authorized"
        ),
        "foundation_change_authorized": auth.get("foundation_change_authorized"),
    }


def _v135_terminal_metrics_match_expected(metrics: Mapping[str, object]) -> bool:
    dominant = _mapping(metrics.get("dominant_predicted_action"))
    expected_dominant = _mapping(
        EXPECTED_V135_TERMINAL_METRICS["dominant_predicted_action"]
    )
    return (
        _same_number(
            metrics.get("heldout_accuracy"),
            EXPECTED_V135_TERMINAL_METRICS["heldout_accuracy"],
        )
        and _same_number(
            metrics.get("heldout_action_only_accuracy"),
            EXPECTED_V135_TERMINAL_METRICS["heldout_action_only_accuracy"],
        )
        and _same_number(
            metrics.get("heldout_action_order_accuracy"),
            EXPECTED_V135_TERMINAL_METRICS["heldout_action_order_accuracy"],
        )
        and _same_number(
            metrics.get("seed29_accuracy"),
            EXPECTED_V135_TERMINAL_METRICS["seed29_accuracy"],
        )
        and dominant.get("action") == expected_dominant.get("action")
        and _same_number(dominant.get("count"), expected_dominant.get("count"))
        and _same_number(dominant.get("share"), expected_dominant.get("share"))
        and _same_number(dominant.get("total"), expected_dominant.get("total"))
        and _same_number(
            metrics.get("leakage_count"),
            EXPECTED_V135_TERMINAL_METRICS["leakage_count"],
        )
        and _same_number(
            metrics.get("unsupported_action_count"),
            EXPECTED_V135_TERMINAL_METRICS["unsupported_action_count"],
        )
    )


def _authorization_matches_expected_false(auth: Mapping[str, object]) -> bool:
    required_false = (
        "training_authorized",
        "runtime_policy_change_authorized",
        "shadow_scorer_execution_authorized",
    )
    return all(auth.get(key) is False for key in required_false)


def _v135_embedded_source_digests_match(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    v135_report: Mapping[str, object] | None,
) -> bool | None:
    embedded = _mapping(_mapping(v135_report).get("source_reports"))
    compared = 0
    for name in ("v131_report", "v132_report", "v133_report", "v134_report"):
        observed_sha = source_reports.get(name, {}).get("file_sha256")
        embedded_sha = _mapping(embedded.get(name)).get("file_sha256")
        if observed_sha is None or embedded_sha is None:
            continue
        compared += 1
        if observed_sha != embedded_sha:
            return False
    return True if compared else None


def _v106_v135_reconciliation() -> list[dict[str, object]]:
    return [
        {
            "versions": "v106-v113",
            "family": "first_recovery_public_and_branch_evidence",
            "conclusion": (
                "first-recovery rows and branch archives became constructible, "
                "but public signal and state-action readiness remained blocked"
            ),
            "runtime_policy_change_authorized": False,
            "training_authorized": False,
        },
        {
            "versions": "v114-v124",
            "family": "coverage_label_repair_and_rare_action_archive",
            "conclusion": (
                "coverage, tie repair, and rare-action contracts improved the "
                "diagnostic archive but did not authorize a runtime path"
            ),
            "runtime_policy_change_authorized": False,
            "training_authorized": False,
        },
        {
            "versions": "v125-v130",
            "family": "shadow_scorer_and_candidate_public_feature_surface",
            "conclusion": (
                "shadow scoring and candidate-feature rankers stayed blocked by "
                "metric and missing-signal failures"
            ),
            "runtime_policy_change_authorized": False,
            "training_authorized": False,
        },
        {
            "versions": "v131-v135",
            "family": "first_recovery_public_context_ranker",
            "conclusion": (
                "refreshed public context and public rollout history were tested; "
                "v135 closed the path as not promotable"
            ),
            "runtime_policy_change_authorized": False,
            "training_authorized": False,
        },
    ]


def _closed_family_decisions() -> list[dict[str, object]]:
    return [
        {
            "family": family,
            "closed": True,
            "promotable": False,
            "training_authorized": False,
            "runtime_policy_change_authorized": False,
        }
        for family in CLOSED_FAMILIES
    ]


def _decision(*, source_integrity: Mapping[str, object]) -> dict[str, object]:
    return {
        "current_closed_path": CURRENT_CLOSED_PATH,
        "closed": True,
        "promotable": False,
        "training_authorized": False,
        "runtime_policy_change_authorized": False,
        "shadow_scorer_execution_authorized": False,
        "first_recovery_public_context_ranker_path_open": False,
        "more_first_recovery_ranker_or_feature_probes_authorized": False,
        "source_integrity_passed": source_integrity.get("passed"),
        "reason": (
            "v135 terminal evidence ties heldout action-only accuracy, fails "
            "seed29, and retains dominant move_north action concentration"
        ),
        "next_allowed_research_direction": NEXT_ALLOWED_RESEARCH_DIRECTION,
    }


def _authorization_block(
    *,
    v135_authorization: Mapping[str, object],
) -> dict[str, object]:
    return {
        "closed_path": CURRENT_CLOSED_PATH,
        "training_authorized": False,
        "runtime_policy_change_authorized": False,
        "shadow_scorer_execution_authorized": False,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "runtime_policy_change_recommended": False,
        "gate_change_authorized": False,
        "viewer_change_authorized": False,
        "replay_golden_change_authorized": False,
        "foundation_change_authorized": False,
        "v136_first_recovery_feature_probe_authorized": False,
        "another_first_recovery_ranker_probe_authorized": False,
        "runtime_promotion_authorized": False,
        "v135_authorization_observed": dict(v135_authorization),
        "claim_causality": False,
    }


def _policy_input_boundary() -> dict[str, object]:
    return {
        "disallowed_policy_inputs": list(DISALLOWED_POLICY_INPUTS),
        "private_world_state_allowed": False,
        "fixture_identity_allowed": False,
        "seed_identity_allowed": False,
        "future_rows_allowed": False,
        "heuristic_recommendation_allowed": False,
        "public_rollout_context_allowed_only_for_new_non_ranker_diagnostic": True,
        "claim_causality": False,
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return None


def _same_number(left: object, right: object) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return False
    return abs(float(left_number) - float(right_number)) <= 1e-15
