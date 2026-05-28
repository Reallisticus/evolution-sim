from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_archive_blocker_diagnostic import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V116_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_oracle_tie_break_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V117_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _int,
    _mapping,
    _resolve_archive_rows,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_rare_action_coverage_targeting import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V121_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V119_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V119_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V120_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION,
    _resolve_manifest_rows,
    _trainable_leakage,
)
from evolution_sim.mind.first_recovery_tie_aware_label_repair import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V118_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_SCHEMA_VERSION = (
    "mind_v3_first_recovery_inspector_bundle_v1"
)
MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_POLICY = (
    "diagnostics_only_first_recovery_inspector_bundle_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-first-recovery-inspector-bundle.json"
)

MAX_EXAMPLES = 16
EXPECTED_V121_CLASSIFICATION = "rare_action_coverage_not_available_in_existing_archive"
EXPECTED_V121_RARE_ATTACK_CANDIDATE_COUNTS: dict[str, int] = {
    "attack_east": 0,
    "attack_west": 0,
}
UPSTREAM_NO_AUTHORIZATION_FIELDS: tuple[str, ...] = (
    "v113_readiness_rerun_allowed",
    "downstream_shadow_scorer_allowed",
    "claim_causality",
    "runtime_policy_change_recommended",
    "trained_artifact_change_recommended",
    "gate_change_recommended",
    "observation_field_change_recommended",
    "viewer_change_recommended",
)

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "diagnostics_only_no_runtime_promotion",
    "missing_evidence_inconclusive",
    "first_recovery_inspector_bundle_ready",
    "first_recovery_inspector_source_integrity_failed",
    "readiness_rerun_blocked",
)


@dataclass(frozen=True, slots=True)
class FirstRecoveryInspectorBundleBuild:
    report: dict[str, object]


def build_first_recovery_inspector_bundle(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    archive_rows: Sequence[Mapping[str, object]] | None = None,
    archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    v116_report: Mapping[str, object] | None = None,
    v116_report_path: str | Path | None = DEFAULT_V116_REPORT_PATH,
    v117_report: Mapping[str, object] | None = None,
    v117_report_path: str | Path | None = DEFAULT_V117_REPORT_PATH,
    v118_report: Mapping[str, object] | None = None,
    v118_report_path: str | Path | None = DEFAULT_V118_REPORT_PATH,
    v119_report: Mapping[str, object] | None = None,
    v119_report_path: str | Path | None = DEFAULT_V119_REPORT_PATH,
    manifest_rows: Sequence[Mapping[str, object]] | None = None,
    manifest_path: str | Path | None = DEFAULT_V119_MANIFEST_PATH,
    v120_report: Mapping[str, object] | None = None,
    v120_report_path: str | Path | None = DEFAULT_V120_REPORT_PATH,
    v121_report: Mapping[str, object] | None = None,
    v121_report_path: str | Path | None = DEFAULT_V121_REPORT_PATH,
) -> FirstRecoveryInspectorBundleBuild:
    archive_payload, archive_evidence = _resolve_json_report(
        archive_report,
        archive_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    rows, rows_evidence = _resolve_archive_rows(archive_rows, archive_rows_path)
    v116_payload, v116_evidence = _resolve_json_report(
        v116_report,
        v116_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION,
    )
    v117_payload, v117_evidence = _resolve_json_report(
        v117_report,
        v117_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_SCHEMA_VERSION,
    )
    v118_payload, v118_evidence = _resolve_json_report(
        v118_report,
        v118_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
    )
    v119_payload, v119_evidence = _resolve_json_report(
        v119_report,
        v119_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION
        ),
    )
    manifest, manifest_evidence = _resolve_manifest_rows(manifest_rows, manifest_path)
    v120_payload, v120_evidence = _resolve_json_report(
        v120_report,
        v120_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION
        ),
    )
    v121_payload, v121_evidence = _resolve_json_report(
        v121_report,
        v121_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION
        ),
    )
    source_reports = {
        "v115_archive_report": archive_evidence,
        "v115_archive_rows": rows_evidence,
        "v116_report": v116_evidence,
        "v117_report": v117_evidence,
        "v118_report": v118_evidence,
        "v119_report": v119_evidence,
        "v119_manifest": manifest_evidence,
        "v120_report": v120_evidence,
        "v121_report": v121_evidence,
        "source_report_policy": "inputs_are_read_only_diagnostics_sources",
    }
    reports = {
        "v115": archive_payload or {},
        "v116": v116_payload or {},
        "v117": v117_payload or {},
        "v118": v118_payload or {},
        "v119": v119_payload or {},
        "v120": v120_payload or {},
        "v121": v121_payload or {},
    }
    source_integrity = _source_integrity(
        source_reports=source_reports,
        reports=reports,
        archive_rows=rows,
        manifest_rows=manifest,
    )
    branch_rows = _branch_rows(manifest)
    classification_chain = _classification_chain(reports)
    blocker_summary = _blocker_summary(reports)
    classification = _classification(source_integrity)
    recommendation = _recommendation(classification)
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_POLICY,
        "contract": _contract(),
        "boundary_labels": _boundary_labels(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "classification_chain": classification_chain,
        "blocker_summary": blocker_summary,
        "branch_count": len(branch_rows),
        "branch_rows": branch_rows,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryInspectorBundleBuild(report=report)


def write_first_recovery_inspector_bundle(
    build: FirstRecoveryInspectorBundleBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, sort_keys=True, indent=2, allow_nan=False, fp=handle)
        handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_SCHEMA_VERSION,
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "observation_field_change": False,
        "viewer_effect": "read_only_model_only",
        "archive_replay_executed": False,
        "training_executed": False,
        "readiness_rerun_executed": False,
        "runtime_policy_implemented": False,
        "shadow_scorer_implemented": False,
        "bundle_is_replay_contract": False,
        "bundle_is_training_manifest": False,
        "claim_causality": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "contract_digest": stable_payload_digest(
            {
                "schema_version": MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_SCHEMA_VERSION,
                "audit_policy": MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_POLICY,
            }
        ),
    }


def _boundary_labels() -> dict[str, object]:
    return {
        "audit_metadata_not_trainable_input": True,
        "bundle_is_not_replay_contract": True,
        "bundle_is_not_training_manifest": True,
        "bundle_does_not_authorize_readiness_shadow_runtime_promotion": True,
        "private_world_state_exposed": False,
        "trainable_public_input_contents_exposed": False,
        "labels": [
            "audit_metadata_is_not_trainable_input",
            "not_a_replay_contract",
            "not_a_training_manifest",
            "does_not_authorize_readiness_shadow_runtime_promotion",
            "does_not_expose_private_world_state",
        ],
    }


def _source_integrity(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    reports: Mapping[str, Mapping[str, object]],
    archive_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if not name.startswith("v"):
            continue
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")

    v116_source = _mapping(reports["v116"].get("source_archive_verification"))
    v116_expected = _mapping(v116_source.get("expected_v115_facts"))
    v116_reported_archive_rows = v116_source.get("archive_row_count_report")
    v116_expected_archive_rows = v116_expected.get("archive_row_count")
    v116_branch_result_count = v116_source.get("branch_result_count")
    v119_contract = _mapping(reports["v119"].get("contract_checks"))
    v120_source = _mapping(reports["v120"].get("source_integrity"))
    v121_source = _mapping(reports["v121"].get("source_integrity"))
    v121_classification = _mapping(reports["v121"].get("classification"))
    v121_search = _mapping(reports["v121"].get("candidate_search"))
    manifest_digest = stable_payload_digest(list(manifest_rows))
    manifest_leakage = _trainable_leakage(manifest_rows)
    branch_ids = [
        row.get("branch_id")
        for row in manifest_rows
        if isinstance(row.get("branch_id"), str) and row.get("branch_id")
    ]

    if v116_source.get("verification_passed") is not True:
        failures.append("v116_archive_verification_not_passed")
    if len(archive_rows) == 0:
        failures.append("v115_archive_rows_empty")
    if not _strict_positive_int(v116_reported_archive_rows) or (
        len(archive_rows) != v116_reported_archive_rows
    ):
        failures.append("v115_archive_row_count_mismatch")
    if v116_expected_archive_rows is not None and (
        not _strict_positive_int(v116_expected_archive_rows)
        or len(archive_rows) != v116_expected_archive_rows
    ):
        failures.append("v115_archive_row_count_mismatch")
    if v116_source.get("replay_verified") is not True:
        failures.append("v115_replay_not_verified")
    if not _strict_positive_int(v116_branch_result_count) or (
        v116_branch_result_count != len(set(branch_ids))
    ):
        failures.append("v115_branch_result_count_mismatch")
    if v119_contract.get("passed") is not True:
        failures.append("v119_contract_checks_not_passed")
    if v119_contract.get("total_violation_count") not in (0, None):
        failures.append("v119_contract_violations_nonzero")
    if v120_source.get("passed") is not True:
        failures.append("v120_source_integrity_not_passed")
    if v121_source.get("passed") is not True:
        failures.append("v121_source_integrity_not_passed")
    if v121_classification.get("primary") != EXPECTED_V121_CLASSIFICATION:
        failures.append("v121_classification_unexpected")
    if "failures" not in v121_source or not isinstance(
        v121_source.get("failures"),
        list,
    ):
        failures.append("v121_source_failures_missing_or_malformed")
    elif v121_source.get("failures") != []:
        failures.append("v121_source_failures_not_empty")
    v121_candidate_report = _v121_candidate_count_report(v121_search)
    v121_candidate_counts = _mapping(v121_candidate_report.get("counts"))
    if v121_candidate_report["matches_expected"] is not True:
        failures.append("v121_candidate_counts_unexpected")
    if len(branch_ids) != len(manifest_rows):
        failures.append("manifest_branch_id_missing_or_malformed")
    if len(set(branch_ids)) != len(branch_ids):
        failures.append("manifest_branch_ids_not_unique")
    if manifest_leakage["split_key_leak_count"]:
        failures.append("manifest_split_trainable_leakage")
    if manifest_leakage["forbidden_metadata_key_count"]:
        failures.append("manifest_metadata_trainable_leakage")
    v119_digest = _mapping(reports["v119"].get("manifest")).get("manifest_digest")
    v120_reported_digest = v120_source.get("reported_manifest_digest")
    v120_computed_digest = v120_source.get("computed_manifest_digest")
    v121_digest = v121_source.get("manifest_digest")
    for name, digest in (
        ("v119_manifest_digest", v119_digest),
        ("v120_reported_manifest_digest", v120_reported_digest),
        ("v120_computed_manifest_digest", v120_computed_digest),
        ("v121_manifest_digest", v121_digest),
    ):
        if not _valid_sha256_digest(digest):
            failures.append(f"{name}_missing_or_malformed")
    for name, digest in (
        ("v119_manifest_digest", v119_digest),
        ("v120_reported_manifest_digest", v120_reported_digest),
        ("v120_computed_manifest_digest", v120_computed_digest),
        ("v121_manifest_digest", v121_digest),
    ):
        if _valid_sha256_digest(digest) and digest != manifest_digest:
            failures.append(f"{name}_mismatch")
    failures.extend(_upstream_no_authorization_failures(reports))

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "archive_row_count": len(archive_rows),
        "manifest_row_count": len(manifest_rows),
        "manifest_unique_branch_count": len(set(branch_ids)),
        "manifest_digest": manifest_digest,
        "v119_reported_manifest_digest": v119_digest,
        "v120_reported_manifest_digest": v120_reported_digest,
        "v120_computed_manifest_digest": v120_computed_digest,
        "v121_manifest_digest": v121_digest,
        "manifest_digest_matches_v119": (
            v119_digest == manifest_digest
        ),
        "manifest_digest_matches_v120": (
            v120_reported_digest == manifest_digest
            and v120_computed_digest == manifest_digest
        ),
        "manifest_digest_matches_v121": (
            v121_digest == manifest_digest
        ),
        "v115_archive_verified": v116_source.get("verification_passed") is True,
        "v115_replay_verified": v116_source.get("replay_verified") is True,
        "v115_branch_result_count": v116_source.get("branch_result_count"),
        "v115_archive_row_count": v116_source.get("archive_row_count_report"),
        "v115_archive_expected_row_count": v116_expected_archive_rows,
        "v115_archive_rows_match_v116_report": (
            _strict_positive_int(v116_reported_archive_rows)
            and len(archive_rows) == v116_reported_archive_rows
        ),
        "v115_branch_result_count_matches_manifest": (
            _strict_positive_int(v116_branch_result_count)
            and v116_branch_result_count == len(set(branch_ids))
        ),
        "v119_contract_checks_passed": v119_contract.get("passed"),
        "v119_contract_total_violation_count": v119_contract.get(
            "total_violation_count"
        ),
        "v120_source_integrity_passed": v120_source.get("passed"),
        "v120_source_failures": v120_source.get("failures"),
        "v121_source_integrity_passed": v121_source.get("passed"),
        "v121_source_failures": v121_source.get("failures"),
        "v121_classification_primary": v121_classification.get("primary"),
        "v121_candidate_counts": v121_candidate_counts,
        "v121_candidate_counts_match_expected": v121_candidate_report[
            "matches_expected"
        ],
        "v121_candidate_count_failures": v121_candidate_report["failures"],
        "manifest_trainable_leakage": manifest_leakage,
        "upstream_no_authorization_checks": _upstream_no_authorization_checks(
            reports
        ),
    }


def _strict_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _valid_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _upstream_no_authorization_checks(
    reports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    checks: dict[str, object] = {}
    for version in ("v118", "v119", "v120", "v121"):
        recommendation = _mapping(reports[version].get("recommendation"))
        fields = {
            field: recommendation.get(field)
            for field in UPSTREAM_NO_AUTHORIZATION_FIELDS
        }
        if version == "v121" and "replay_golden_change_recommended" in recommendation:
            fields["replay_golden_change_recommended"] = recommendation.get(
                "replay_golden_change_recommended"
            )
        checks[version] = {
            "fields": fields,
            "passed": all(value is False for value in fields.values()),
        }
    return checks


def _upstream_no_authorization_failures(
    reports: Mapping[str, Mapping[str, object]],
) -> list[str]:
    failures: list[str] = []
    for version, check in _upstream_no_authorization_checks(reports).items():
        fields = _mapping(check).get("fields", {})
        for field, value in _mapping(fields).items():
            if value is not False:
                failures.append(f"{version}_{field}_not_false")
    return failures


def _v121_candidate_count_report(
    candidate_search: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    per_action = _mapping(candidate_search.get("per_action"))
    counts: dict[str, int] = {}
    for action, expected in EXPECTED_V121_RARE_ATTACK_CANDIDATE_COUNTS.items():
        if action not in per_action or not isinstance(per_action.get(action), Mapping):
            failures.append(f"{action}_candidate_payload_missing")
            continue
        payload = _mapping(per_action.get(action))
        value = payload.get("valid_candidate_count")
        if type(value) is not int:
            failures.append(f"{action}_candidate_count_missing_or_malformed")
            continue
        counts[action] = value
        if value != expected:
            failures.append(f"{action}_candidate_count_unexpected")
    extra_actions = sorted(str(action) for action in set(per_action) - set(counts))
    if extra_actions:
        failures.append("extra_candidate_actions_present")
    return {
        "counts": counts,
        "expected_counts": EXPECTED_V121_RARE_ATTACK_CANDIDATE_COUNTS,
        "failures": failures,
        "matches_expected": not failures
        and counts == EXPECTED_V121_RARE_ATTACK_CANDIDATE_COUNTS,
    }


def _classification_chain(
    reports: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    return {
        version: _classification_entry(version, report)
        for version, report in reports.items()
        if version != "v115"
    } | {"v115": _classification_entry("v115", reports["v115"])}


def _classification_entry(
    version: str,
    report: Mapping[str, object],
) -> dict[str, object]:
    classification = _mapping(report.get("classification"))
    recommendation = _mapping(report.get("recommendation"))
    if version == "v115":
        recommendation = _mapping(report.get("research_recommendation"))
    return {
        "schema_version": report.get("schema_version"),
        "primary": classification.get("primary"),
        "labels": list(classification.get("labels", []))
        if isinstance(classification.get("labels"), list)
        else [],
        "missing_evidence": list(classification.get("missing_evidence", []))
        if isinstance(classification.get("missing_evidence"), list)
        else [],
        "next_step": recommendation.get("next_step")
        or recommendation.get("recommendation"),
        "v113_readiness_rerun_allowed": recommendation.get(
            "v113_readiness_rerun_allowed",
            False,
        ),
        "downstream_shadow_scorer_allowed": recommendation.get(
            "downstream_shadow_scorer_allowed",
            False,
        ),
        "claim_causality": recommendation.get("claim_causality", False),
    }


def _blocker_summary(
    reports: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    v115_summary = _mapping(reports["v115"].get("branch_archive_summary"))
    v115_oracle = _mapping(reports["v115"].get("oracle_label_summary"))
    v116_source = _mapping(reports["v116"].get("source_archive_verification"))
    v116_stay = _mapping(reports["v116"].get("stay_dominance_analysis"))
    v117_tie = _mapping(reports["v117"].get("objective_tie_break_analysis"))
    v118_repair = _mapping(reports["v118"].get("tie_aware_label_repair"))
    v119_contract = _mapping(reports["v119"].get("contract_checks"))
    v119_split = _mapping(reports["v119"].get("split_support"))
    v120_scarcity = _mapping(reports["v120"].get("scarcity_analysis"))
    v121_support = _mapping(reports["v121"].get("current_rare_action_support"))
    v121_search = _mapping(reports["v121"].get("candidate_search"))
    best_policy = str(v118_repair.get("best_valid_policy") or "")
    best_policy_report = _mapping(_mapping(v118_repair.get("policies")).get(best_policy))
    return {
        "summary_labels": [
            "v115_archive_verified",
            "v116_stay_dominance_detected",
            "v117_tie_break_artifact_likely",
            "v118_repaired_distribution_available",
            "v119_contract_clean_but_split_support_limited",
            "v120_rare_action_support_limitation",
            "v121_no_recoverable_rare_attack_candidates_in_existing_archive",
        ],
        "v115_archive_verified": {
            "verified": (
                v115_summary.get("replay_verified") is True
                and v116_source.get("verification_passed") is True
            ),
            "selected_targets": v116_source.get("selected_target_count"),
            "skipped_targets": v116_source.get("skipped_target_count"),
            "branch_result_count": v116_source.get("branch_result_count"),
            "archive_row_count": v116_source.get("archive_row_count_report")
            or v115_summary.get("archive_row_count"),
            "heuristic_action_source_count": v116_source.get(
                "heuristic_action_source_count"
            ),
            "resolution_invalid_count": _mapping(
                reports["v115"].get("legality_summary")
            ).get("resolution_invalid_count"),
        },
        "v116_stay_dominance": {
            "dominant_oracle_action": v116_stay.get("dominant_oracle_action")
            or v115_oracle.get("dominant_oracle_action"),
            "dominant_oracle_action_count": v116_stay.get(
                "dominant_oracle_action_count"
            )
            or v115_oracle.get("dominant_oracle_action_count"),
            "dominant_oracle_action_share": v116_stay.get(
                "dominant_oracle_action_share"
            )
            or v115_oracle.get("dominant_oracle_action_share"),
            "objective_scoring_artifact_risk": v116_stay.get(
                "objective_scoring_artifact_risk"
            ),
            "branch_outcome_signal_present": v116_stay.get(
                "branch_outcome_signal_present"
            ),
        },
        "v117_tie_break_artifact": {
            "classification": _mapping(reports["v117"].get("classification")).get(
                "primary"
            ),
            "current_serialized_oracle_stay_count": v117_tie.get(
                "current_serialized_oracle_stay_count"
            ),
            "unique_objective_best_stay_count": _mapping(
                v117_tie.get("unique_objective_best_action_counts")
            ).get("stay"),
            "tie_neutral_stay_counts": _tie_neutral_stay_counts(v117_tie),
        },
        "v118_repaired_distribution": {
            "classification": _mapping(reports["v118"].get("classification")).get(
                "primary"
            ),
            "best_policy": best_policy,
            "repaired_action_counts": best_policy_report.get(
                "repaired_action_counts"
            ),
            "dominant_action": best_policy_report.get("dominant_action"),
            "dominant_action_share": best_policy_report.get("dominant_action_share"),
            "changed_branch_count": best_policy_report.get("changed_branch_count"),
            "unique_best_changed_count": best_policy_report.get(
                "unique_best_changed_count"
            ),
            "resolution_invalid_selected_count": best_policy_report.get(
                "resolution_invalid_selected_count"
            ),
            "objective_equivalence_violation_count": best_policy_report.get(
                "objective_equivalence_violation_count"
            ),
        },
        "v119_contract_clean_split_limited": {
            "classification": _mapping(reports["v119"].get("classification")).get(
                "primary"
            ),
            "contract_passed": v119_contract.get("passed"),
            "total_violation_count": v119_contract.get("total_violation_count"),
            "split_support_adequate": v119_split.get("support_adequate"),
            "split_warnings": list(v119_split.get("warnings", []))
            if isinstance(v119_split.get("warnings"), list)
            else [],
        },
        "v120_rare_action_support_limitation": {
            "classification": _mapping(reports["v120"].get("classification")).get(
                "primary"
            ),
            "all_splits_can_contain_every_action_class": v120_scarcity.get(
                "all_splits_can_contain_every_action_class_by_total_support"
            ),
            "train2_validation1_test1_feasible_by_total_support": (
                v120_scarcity.get(
                    "train2_validation1_test1_feasible_by_total_support"
                )
            ),
            "rare_action_additional_needed": v120_scarcity.get(
                "rare_action_additional_needed_for_train2_validation1_test1"
            ),
        },
        "v121_no_recoverable_rare_attack_candidates": {
            "classification": _mapping(reports["v121"].get("classification")).get(
                "primary"
            ),
            "rare_action_support": v121_support.get("repaired_action_counts"),
            "valid_candidate_counts": _candidate_counts(v121_search),
            "any_candidate_available": v121_search.get("any_candidate_available"),
            "all_required_actions_have_candidate": v121_search.get(
                "all_required_actions_have_candidate"
            ),
        },
    }


def _tie_neutral_stay_counts(
    objective_tie_break_analysis: Mapping[str, object],
) -> dict[str, object]:
    alternatives = _mapping(objective_tie_break_analysis.get("tie_neutral_alternatives"))
    if alternatives:
        return {
            str(name): _mapping(payload).get("stay_count")
            for name, payload in alternatives.items()
        }
    counts = _mapping(objective_tie_break_analysis.get("tie_neutral_stay_counts"))
    if counts:
        return dict(counts)
    action_counts = _mapping(
        objective_tie_break_analysis.get("tie_neutral_action_counts")
    )
    return {
        str(name): _mapping(payload).get("stay")
        for name, payload in action_counts.items()
    }


def _candidate_counts(candidate_search: Mapping[str, object]) -> dict[str, int]:
    return {
        str(action): _int(_mapping(payload).get("valid_candidate_count"))
        for action, payload in _mapping(candidate_search.get("per_action")).items()
    }


def _branch_rows(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for row in sorted(manifest_rows, key=lambda item: str(item.get("branch_id"))):
        branch_id = row.get("branch_id")
        if not isinstance(branch_id, str) or not branch_id:
            continue
        trainable = row.get("trainable_public_input")
        leakage = _trainable_leakage([row])
        rows[branch_id] = {
            "branch_id": branch_id,
            "audit_metadata": _audit_metadata(row),
            "current_oracle_action": row.get("current_oracle_action"),
            "repaired_action": row.get("repaired_action"),
            "legal_tied_candidate_actions": _string_list(
                row.get("legal_tied_candidate_actions")
            ),
            "unique_objective_best": row.get("unique_objective_best") is True,
            "objective_equivalence_verified": (
                row.get("objective_equivalence_verified") is True
            ),
            "selected_resolution_legal": row.get("selected_resolution_legal")
            is True,
            "current_archive_row_id": row.get("current_archive_row_id"),
            "repaired_archive_row_id": row.get("repaired_archive_row_id"),
            "changed": row.get("changed") is True,
            "selected_observation_digest": row.get("selected_observation_digest"),
            "trainable_public_input": {
                "contents_exposed": False,
                "present": isinstance(trainable, Mapping),
                "non_empty_mapping": isinstance(trainable, Mapping) and bool(trainable),
                "clean": (
                    isinstance(trainable, Mapping)
                    and bool(trainable)
                    and leakage["split_key_leak_count"] == 0
                    and leakage["forbidden_metadata_key_count"] == 0
                ),
                "split_key_leak_count": leakage["split_key_leak_count"],
                "forbidden_metadata_key_count": leakage[
                    "forbidden_metadata_key_count"
                ],
            },
        }
    return rows


def _audit_metadata(row: Mapping[str, object]) -> dict[str, object]:
    metadata = _mapping(row.get("non_trainable_audit_metadata"))
    provenance = _mapping(metadata.get("provenance"))
    return {
        "audit_only": True,
        "non_trainable": metadata.get("non_trainable") is True,
        "purpose": metadata.get("purpose") or "audit_only_not_trainable",
        "source_kind": metadata.get("source_kind") or provenance.get("source_kind"),
        "source_path": metadata.get("source_path") or provenance.get("source_path"),
        "seed": metadata.get("seed") if "seed" in metadata else provenance.get("seed"),
        "tick": metadata.get("tick") if "tick" in metadata else provenance.get("tick"),
        "agent_id": metadata.get("agent_id")
        if "agent_id" in metadata
        else provenance.get("agent_id"),
        "record_index": metadata.get("record_index")
        if "record_index" in metadata
        else provenance.get("record_index"),
        "logged_action": provenance.get("logged_action"),
        "branch_state_digest": provenance.get("branch_state_digest"),
        "audit_metadata_trainable": False,
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return sorted(str(item) for item in value)


def _classification(source_integrity: Mapping[str, object]) -> dict[str, object]:
    if source_integrity.get("passed") is True:
        primary = "first_recovery_inspector_bundle_ready"
    elif source_integrity.get("failures"):
        primary = "first_recovery_inspector_source_integrity_failed"
    else:
        primary = "missing_evidence_inconclusive"
    labels = [primary, "diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
    return {
        "primary": primary,
        "labels": labels,
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    if classification.get("primary") == "first_recovery_inspector_bundle_ready":
        next_step = "inspect_v115_v121_first_recovery_chain_read_only"
        summary = (
            "The inspector bundle is ready for read-only review of the v115-v121 "
            "diagnostic chain; it does not authorize training, shadow scoring, "
            "readiness, runtime policy, gates, replay, or golden changes."
        )
    else:
        next_step = "source_integrity_must_pass_before_first_recovery_inspection"
        summary = (
            "The inspector bundle source evidence is incomplete or inconsistent; "
            "keep all downstream authorizing paths blocked."
        )
    return {
        "next_step": next_step,
        "summary": summary,
        "claim_causality": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "viewer_change_recommended": False,
        "replay_golden_change_recommended": False,
    }
