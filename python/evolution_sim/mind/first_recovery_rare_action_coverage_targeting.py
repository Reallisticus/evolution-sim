from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_archive_blocker_diagnostic import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _int,
    _mapping,
    _resolve_archive_rows,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V119_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V119_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V120_REPORT_PATH,
    EXPECTED_REPAIRED_ACTION_COUNTS,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION,
    _resolve_manifest_rows,
    _trainable_leakage,
)
from evolution_sim.mind.first_recovery_tie_aware_label_repair import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V118_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
    _branch_repair_contexts,
)
from evolution_sim.mind.first_recovery_oracle_tie_break_audit import (
    _serialized_objective_key,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION = (
    "mind_v3_first_recovery_rare_action_coverage_targeting_v1"
)
MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_POLICY = (
    "diagnostics_only_first_recovery_v121_rare_action_coverage_targeting_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v121-first-recovery-rare-action-coverage-targeting.json"
)

RARE_ACTION_ADDITIONAL_NEEDED: dict[str, int] = {
    "attack_east": 1,
    "attack_west": 1,
}
RARE_ACTIONS: tuple[str, ...] = tuple(RARE_ACTION_ADDITIONAL_NEEDED)
EXPECTED_V120_CLASSIFICATION = "split_support_feasibility_limited_by_rare_actions"
EXPECTED_V115_ARCHIVE_ROW_COUNT = 530
EXPECTED_V115_ARCHIVE_BRANCH_COUNT = 106
EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION: dict[str, int] = {
    "3": 10,
    "4": 24,
    "5": 32,
    "6": 36,
    "7": 4,
}
REQUIRED_V118_CLASSIFICATION_LABELS = frozenset(
    {
        "diagnostics_only_no_runtime_promotion",
        "readiness_rerun_blocked",
    }
)
MAX_EXAMPLES = 16

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "diagnostics_only_no_runtime_promotion",
    "missing_evidence_inconclusive",
    "rare_action_coverage_source_integrity_failed",
    "rare_action_coverage_candidate_available",
    "rare_action_coverage_not_available_in_existing_archive",
    "readiness_rerun_blocked",
)


@dataclass(frozen=True, slots=True)
class FirstRecoveryRareActionCoverageTargetingBuild:
    report: dict[str, object]


def build_first_recovery_rare_action_coverage_targeting(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    archive_rows: Sequence[Mapping[str, object]] | None = None,
    archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    v118_report: Mapping[str, object] | None = None,
    v118_report_path: str | Path | None = DEFAULT_V118_REPORT_PATH,
    v119_report: Mapping[str, object] | None = None,
    v119_report_path: str | Path | None = DEFAULT_V119_REPORT_PATH,
    manifest_rows: Sequence[Mapping[str, object]] | None = None,
    manifest_path: str | Path | None = DEFAULT_V119_MANIFEST_PATH,
    v120_report: Mapping[str, object] | None = None,
    v120_report_path: str | Path | None = DEFAULT_V120_REPORT_PATH,
) -> FirstRecoveryRareActionCoverageTargetingBuild:
    archive_payload, archive_evidence = _resolve_json_report(
        archive_report,
        archive_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    rows, rows_evidence = _resolve_archive_rows(archive_rows, archive_rows_path)
    v118_payload, v118_evidence = _resolve_json_report(
        v118_report,
        v118_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
    )
    v119_payload, v119_evidence = _resolve_json_report(
        v119_report,
        v119_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
    )
    manifest, manifest_evidence = _resolve_manifest_rows(manifest_rows, manifest_path)
    v120_payload, v120_evidence = _resolve_json_report(
        v120_report,
        v120_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION
        ),
    )
    source_reports = {
        "v115_archive_report": archive_evidence,
        "v115_archive_rows": rows_evidence,
        "v118_report": v118_evidence,
        "v119_report": v119_evidence,
        "v119_manifest": manifest_evidence,
        "v120_report": v120_evidence,
        "source_report_policy": "inputs_are_read_only_diagnostics_sources",
    }
    source_integrity = _source_integrity(
        archive_evidence=archive_evidence,
        archive_rows=rows,
        rows_evidence=rows_evidence,
        v118_report=v118_payload,
        v118_evidence=v118_evidence,
        v119_report=v119_payload,
        v119_evidence=v119_evidence,
        manifest_rows=manifest,
        manifest_evidence=manifest_evidence,
        v120_report=v120_payload,
        v120_evidence=v120_evidence,
    )
    current_support = _current_rare_action_support(manifest)
    candidate_search = _candidate_search(rows, manifest, source_integrity)
    classification = _classification(
        source_integrity=source_integrity,
        candidate_search=candidate_search,
    )
    recommendation = _recommendation(classification=classification)
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "current_rare_action_support": current_support,
        "candidate_search": candidate_search,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryRareActionCoverageTargetingBuild(report=report)


def write_first_recovery_rare_action_coverage_targeting_report(
    build: FirstRecoveryRareActionCoverageTargetingBuild,
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
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION
        ),
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "observation_field_change": False,
        "viewer_effect": "none",
        "archive_replay_executed": False,
        "training_executed": False,
        "readiness_rerun_executed": False,
        "runtime_policy_implemented": False,
        "shadow_scorer_implemented": False,
        "objective_values_changed": False,
        "input_artifacts_mutated": False,
        "claim_causality": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "diagnostic_policy": (
            "read_only_v115_v118_v119_v120_rare_attack_coverage_targeting"
        ),
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION
                ),
                "audit_policy": MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_POLICY,
            }
        ),
    }


def _source_integrity(
    *,
    archive_evidence: Mapping[str, object],
    archive_rows: Sequence[Mapping[str, object]],
    rows_evidence: Mapping[str, object],
    v118_report: Mapping[str, object] | None,
    v118_evidence: Mapping[str, object],
    v119_report: Mapping[str, object] | None,
    v119_evidence: Mapping[str, object],
    manifest_rows: Sequence[Mapping[str, object]],
    manifest_evidence: Mapping[str, object],
    v120_report: Mapping[str, object] | None,
    v120_evidence: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in (
        ("v115_archive_report", archive_evidence),
        ("v115_archive_rows", rows_evidence),
        ("v118_report", v118_evidence),
        ("v119_report", v119_evidence),
        ("v119_manifest", manifest_evidence),
        ("v120_report", v120_evidence),
    ):
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
    for name, evidence in (
        ("v115_archive_report", archive_evidence),
        ("v118_report", v118_evidence),
        ("v119_report", v119_evidence),
        ("v120_report", v120_evidence),
    ):
        if evidence.get("loaded") is True and evidence.get("schema_matches") is not True:
            failures.append(f"{name}_schema_mismatch")

    manifest_digest = stable_payload_digest(list(manifest_rows))
    manifest_branch_ids = {
        row.get("branch_id")
        for row in manifest_rows
        if isinstance(row.get("branch_id"), str) and row.get("branch_id")
    }
    archive_branch_report = _archive_branch_integrity(
        archive_rows,
        manifest_branch_ids=manifest_branch_ids,
    )
    if archive_branch_report["archive_row_count"] == 0:
        failures.append("v115_archive_rows_empty")
    if archive_branch_report["archive_row_count"] != EXPECTED_V115_ARCHIVE_ROW_COUNT:
        failures.append("v115_archive_row_count_mismatch")
    if (
        archive_branch_report["unique_archive_branch_count"]
        != EXPECTED_V115_ARCHIVE_BRANCH_COUNT
    ):
        failures.append("v115_archive_branch_count_mismatch")
    if archive_branch_report["missing_branch_id_row_count"]:
        failures.append("v115_archive_branch_id_missing_or_malformed")
    if archive_branch_report["row_identity_mismatch_count"]:
        failures.append("v115_archive_row_identity_mismatch")
    if archive_branch_report["branch_sets_match_manifest"] is not True:
        failures.append("v115_archive_manifest_branch_mismatch")
    if archive_branch_report["branch_size_distribution"] != (
        EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION
    ):
        failures.append("v115_archive_branch_size_distribution_mismatch")
    if archive_branch_report["branch_row_count_mismatch_count"]:
        failures.append("v115_archive_branch_row_count_mismatch")
    if archive_branch_report["branch_integrity_failures"]:
        failures.append("v115_archive_branch_integrity_failed")

    v118 = _mapping(v118_report or {})
    v118_classification = _mapping(v118.get("classification"))
    v118_recommendation = _mapping(v118.get("recommendation"))
    if v118_classification.get("primary") != "tie_aware_repair_clears_action_collapse":
        failures.append("v118_classification_not_clearing")
    v118_labels = {
        str(label)
        for label in v118_classification.get("labels", [])
        if isinstance(label, str)
    }
    for label in sorted(REQUIRED_V118_CLASSIFICATION_LABELS):
        if label not in v118_labels:
            failures.append(f"v118_classification_missing_{label}")
    for field in (
        "v113_readiness_rerun_allowed",
        "downstream_shadow_scorer_allowed",
        "claim_causality",
        "runtime_policy_change_recommended",
        "trained_artifact_change_recommended",
        "gate_change_recommended",
        "observation_field_change_recommended",
        "viewer_change_recommended",
    ):
        if v118_recommendation.get(field) is not False:
            failures.append(f"v118_{field}_not_false")

    v119 = _mapping(v119_report or {})
    v119_classification = _mapping(v119.get("classification"))
    v119_recommendation = _mapping(v119.get("recommendation"))
    v119_contract_checks = _mapping(v119.get("contract_checks"))
    v119_manifest = _mapping(v119.get("manifest"))
    v119_reported_digest = v119_manifest.get("manifest_digest")
    if v119_classification.get("primary") != "repaired_label_contract_support_limited":
        failures.append("v119_classification_not_support_limited")
    if v119_contract_checks.get("passed") is not True:
        failures.append("v119_contract_checks_not_passed")
    if v119_contract_checks.get("total_violation_count") != 0:
        failures.append("v119_contract_total_violation_count_not_zero")
    if v119_contract_checks.get("integrity_failures") != []:
        failures.append("v119_contract_integrity_failures_not_empty_or_malformed")
    for field in (
        "v113_readiness_rerun_allowed",
        "downstream_shadow_scorer_allowed",
        "claim_causality",
    ):
        if v119_recommendation.get(field) is not False:
            failures.append(f"v119_{field}_not_false")

    manifest_counts = Counter(str(row.get("repaired_action")) for row in manifest_rows)
    manifest_trainable_integrity = _manifest_trainable_integrity(manifest_rows)
    manifest_leakage = _trainable_leakage(manifest_rows)
    v120 = _mapping(v120_report or {})
    v120_source = _mapping(v120.get("source_integrity"))
    v120_recommendation = _mapping(v120.get("recommendation"))
    v120_classification = _mapping(v120.get("classification"))
    v120_rare_needed = _mapping(
        _mapping(v120.get("scarcity_analysis")).get(
            "rare_action_additional_needed_for_train2_validation1_test1"
        )
    )
    v120_leakage = _mapping(v120_source.get("trainable_leakage"))
    v120_manifest_counts = {
        str(action): _int(count)
        for action, count in _mapping(
            v120_source.get("manifest_repaired_action_counts")
        ).items()
    }
    if v120_classification.get("primary") != EXPECTED_V120_CLASSIFICATION:
        failures.append("v120_classification_not_support_limited")
    if v120_source.get("passed") is not True:
        failures.append("v120_source_integrity_not_passed")
    if "failures" not in v120_source:
        failures.append("v120_source_failures_missing")
    elif v120_source.get("failures") != []:
        failures.append("v120_source_failures_not_empty_or_malformed")
    if "trainable_leakage" not in v120_source or not isinstance(
        v120_source.get("trainable_leakage"),
        Mapping,
    ):
        failures.append("v120_trainable_leakage_missing_or_malformed")
    if v120_source.get("manifest_digest_matches") is not True:
        failures.append("v120_manifest_digest_not_verified")
    if v120_source.get("computed_manifest_digest") != manifest_digest:
        failures.append("v120_computed_manifest_digest_mismatch")
    if v120_source.get("reported_manifest_digest") != manifest_digest:
        failures.append("v120_reported_manifest_digest_mismatch")
    if v119_reported_digest != manifest_digest:
        failures.append("v119_manifest_digest_mismatch")
    if dict(manifest_counts) != EXPECTED_REPAIRED_ACTION_COUNTS:
        failures.append("manifest_repaired_action_counts_unexpected")
    if v120_manifest_counts != EXPECTED_REPAIRED_ACTION_COUNTS:
        failures.append("v120_manifest_repaired_action_counts_unexpected")
    if {
        str(action): _int(count) for action, count in v120_rare_needed.items()
    } != RARE_ACTION_ADDITIONAL_NEEDED:
        failures.append("v120_rare_action_additions_unexpected")
    if type(v120_leakage.get("split_key_leak_count")) is not int:
        failures.append("v120_split_trainable_leakage_missing_or_malformed")
    elif v120_leakage.get("split_key_leak_count") != 0:
        failures.append("v120_split_trainable_leakage_nonzero")
    if type(v120_leakage.get("forbidden_metadata_key_count")) is not int:
        failures.append("v120_metadata_trainable_leakage_missing_or_malformed")
    elif v120_leakage.get("forbidden_metadata_key_count") != 0:
        failures.append("v120_metadata_trainable_leakage_nonzero")
    if manifest_leakage["split_key_leak_count"]:
        failures.append("manifest_split_trainable_leakage")
    if manifest_leakage["forbidden_metadata_key_count"]:
        failures.append("manifest_metadata_trainable_leakage")
    if manifest_trainable_integrity["missing_or_malformed_count"]:
        failures.append("manifest_trainable_public_input_missing_or_malformed")
    for field, failure in (
        ("v113_readiness_rerun_allowed", "v120_v113_readiness_not_blocked"),
        ("downstream_shadow_scorer_allowed", "v120_shadow_scorer_not_blocked"),
        ("claim_causality", "v120_claim_causality_not_false"),
        ("runtime_policy_change_recommended", "v120_runtime_policy_change_recommended"),
        (
            "trained_artifact_change_recommended",
            "v120_trained_artifact_change_recommended",
        ),
        ("gate_change_recommended", "v120_gate_change_recommended"),
        (
            "observation_field_change_recommended",
            "v120_observation_field_change_recommended",
        ),
        ("viewer_change_recommended", "v120_viewer_change_recommended"),
    ):
        if v120_recommendation.get(field) is not False:
            failures.append(failure)
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v115_archive_row_count": archive_branch_report["archive_row_count"],
        "v115_archive_unique_branch_count": (
            archive_branch_report["unique_archive_branch_count"]
        ),
        "v115_archive_branch_size_distribution": (
            archive_branch_report["branch_size_distribution"]
        ),
        "v115_expected_branch_size_distribution": (
            EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION
        ),
        "v115_archive_branch_sets_match_manifest": (
            archive_branch_report["branch_sets_match_manifest"]
        ),
        "v115_archive_missing_branch_id_row_count": (
            archive_branch_report["missing_branch_id_row_count"]
        ),
        "v115_archive_missing_branch_id_examples": (
            archive_branch_report["missing_branch_id_examples"]
        ),
        "v115_archive_row_identity_mismatch_count": (
            archive_branch_report["row_identity_mismatch_count"]
        ),
        "v115_archive_row_identity_mismatch_examples": (
            archive_branch_report["row_identity_mismatch_examples"]
        ),
        "v115_archive_branch_row_count_mismatch_count": (
            archive_branch_report["branch_row_count_mismatch_count"]
        ),
        "v115_archive_branch_row_count_mismatch_examples": (
            archive_branch_report["branch_row_count_mismatch_examples"]
        ),
        "v115_archive_branch_integrity_failures": (
            archive_branch_report["branch_integrity_failures"]
        ),
        "v115_archive_branch_integrity_examples": (
            archive_branch_report["branch_integrity_examples"]
        ),
        "v118_classification_primary": v118_classification.get("primary"),
        "v119_classification_primary": v119_classification.get("primary"),
        "v120_classification_primary": v120_classification.get("primary"),
        "v120_source_integrity_passed": v120_source.get("passed"),
        "v120_source_failures": v120_source.get("failures"),
        "manifest_row_count": len(manifest_rows),
        "manifest_digest": manifest_digest,
        "v119_reported_manifest_digest": v119_reported_digest,
        "v120_reported_manifest_digest": v120_source.get("reported_manifest_digest"),
        "v120_computed_manifest_digest": v120_source.get("computed_manifest_digest"),
        "manifest_digest_matches_v119": v119_reported_digest == manifest_digest,
        "manifest_digest_matches_v120": (
            v120_source.get("reported_manifest_digest") == manifest_digest
            and v120_source.get("computed_manifest_digest") == manifest_digest
            and v120_source.get("manifest_digest_matches") is True
        ),
        "manifest_repaired_action_counts": _counter_to_dict(manifest_counts),
        "manifest_trainable_public_input_integrity": manifest_trainable_integrity,
        "v120_manifest_repaired_action_counts": dict(
            sorted(v120_manifest_counts.items())
        ),
        "rare_action_additional_needed": {
            str(action): _int(count) for action, count in v120_rare_needed.items()
        },
        "manifest_trainable_leakage": manifest_leakage,
        "v120_trainable_leakage": {
            "split_key_leak_count": v120_leakage.get("split_key_leak_count"),
            "forbidden_metadata_key_count": v120_leakage.get(
                "forbidden_metadata_key_count"
            ),
        },
        "v113_readiness_rerun_allowed": v120_recommendation.get(
            "v113_readiness_rerun_allowed"
        ),
        "downstream_shadow_scorer_allowed": v120_recommendation.get(
            "downstream_shadow_scorer_allowed"
        ),
        "claim_causality": v120_recommendation.get("claim_causality"),
    }


def _archive_branch_integrity(
    archive_rows: Sequence[Mapping[str, object]],
    *,
    manifest_branch_ids: set[object],
) -> dict[str, object]:
    archive_branch_ids: set[str] = set()
    branch_counts: Counter[str] = Counter()
    missing_branch_id_count = 0
    examples: list[dict[str, object]] = []
    identity_mismatch_count = 0
    identity_examples: list[dict[str, object]] = []
    for index, row in enumerate(archive_rows):
        branch_id = _mapping(row.get("provenance")).get("branch_id")
        archive_row_id = row.get("archive_row_id")
        candidate_action = row.get("candidate_action")
        if isinstance(branch_id, str) and branch_id:
            archive_branch_ids.add(branch_id)
            branch_counts[branch_id] += 1
            expected_prefix = f"{branch_id}::action::"
            identity_failure: str | None = None
            if not isinstance(archive_row_id, str) or not archive_row_id:
                identity_failure = "archive_row_id_missing_or_malformed"
            elif not archive_row_id.startswith(expected_prefix):
                identity_failure = "archive_row_id_branch_prefix_mismatch"
            elif archive_row_id.removeprefix(expected_prefix) != str(candidate_action):
                identity_failure = "archive_row_id_candidate_action_suffix_mismatch"
            if identity_failure is not None:
                identity_mismatch_count += 1
                if len(identity_examples) < MAX_EXAMPLES:
                    identity_examples.append(
                        {
                            "row_index": index,
                            "reason": identity_failure,
                            "archive_row_id": archive_row_id,
                            "provenance_branch_id": branch_id,
                            "candidate_action": candidate_action,
                        }
                    )
            continue
        missing_branch_id_count += 1
        identity_mismatch_count += 1
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "row_index": index,
                    "archive_row_id": row.get("archive_row_id"),
                    "candidate_action": row.get("candidate_action"),
                    "branch_id_repr": repr(branch_id),
                }
            )
        if len(identity_examples) < MAX_EXAMPLES:
            identity_examples.append(
                {
                    "row_index": index,
                    "reason": "provenance_branch_id_missing_or_malformed",
                    "archive_row_id": archive_row_id,
                    "provenance_branch_id": branch_id,
                    "candidate_action": candidate_action,
                }
            )
    allowed_sizes = {int(size) for size in EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION}
    row_count_mismatch_examples: list[dict[str, object]] = []
    row_count_mismatch_count = 0
    for branch_id in sorted(manifest_branch_ids):
        count = branch_counts.get(str(branch_id), 0)
        if count in allowed_sizes:
            continue
        row_count_mismatch_count += 1
        if len(row_count_mismatch_examples) < MAX_EXAMPLES:
            row_count_mismatch_examples.append(
                {
                    "branch_id": branch_id,
                    "archive_row_count": count,
                    "expected_allowed_counts": sorted(allowed_sizes),
                }
            )
    contexts, branch_failures, branch_examples = _branch_repair_contexts(archive_rows)
    return {
        "archive_row_count": len(archive_rows),
        "unique_archive_branch_count": len(archive_branch_ids),
        "manifest_branch_count": len(manifest_branch_ids),
        "branch_sets_match_manifest": archive_branch_ids == manifest_branch_ids,
        "branch_size_distribution": {
            str(size): count
            for size, count in sorted(Counter(branch_counts.values()).items())
        },
        "archive_only_branch_count": len(archive_branch_ids - set(manifest_branch_ids)),
        "manifest_only_branch_count": len(set(manifest_branch_ids) - archive_branch_ids),
        "missing_branch_id_row_count": missing_branch_id_count,
        "missing_branch_id_examples": examples,
        "row_identity_mismatch_count": identity_mismatch_count,
        "row_identity_mismatch_examples": identity_examples,
        "branch_row_count_mismatch_count": row_count_mismatch_count,
        "branch_row_count_mismatch_examples": row_count_mismatch_examples,
        "branch_context_count": len(contexts),
        "branch_integrity_failures": list(branch_failures),
        "branch_integrity_examples": list(branch_examples),
    }


def _manifest_trainable_integrity(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    examples: list[dict[str, object]] = []
    missing_or_malformed_count = 0
    for index, row in enumerate(manifest_rows):
        trainable = row.get("trainable_public_input")
        if isinstance(trainable, Mapping) and bool(trainable):
            continue
        missing_or_malformed_count += 1
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "row_index": index,
                    "branch_id": row.get("branch_id"),
                    "repaired_action": row.get("repaired_action"),
                    "trainable_public_input_type": type(
                        row.get("trainable_public_input")
                    ).__name__,
                }
            )
    return {
        "missing_or_malformed_count": missing_or_malformed_count,
        "examples": examples,
    }


def _current_rare_action_support(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows_by_action: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        action = str(row.get("repaired_action"))
        if action in RARE_ACTIONS:
            rows_by_action[action].append(row)
    per_action = {
        action: {
            "repaired_branch_count": len(rows_by_action[action]),
            "additional_needed_for_train2_validation1_test1": (
                RARE_ACTION_ADDITIONAL_NEEDED[action]
            ),
            "branches": [
                _manifest_support_example(row)
                for row in sorted(
                    rows_by_action[action],
                    key=lambda item: str(item.get("branch_id")),
                )
            ],
        }
        for action in RARE_ACTIONS
    }
    return {
        "rare_actions": list(RARE_ACTIONS),
        "per_action": per_action,
        "repaired_action_counts": {
            action: per_action[action]["repaired_branch_count"]
            for action in RARE_ACTIONS
        },
    }


def _manifest_support_example(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": row.get("branch_id"),
        "changed": row.get("changed"),
        "current_oracle_action": row.get("current_oracle_action"),
        "repaired_action": row.get("repaired_action"),
        "unique_objective_best": row.get("unique_objective_best"),
        "objective_equivalence_verified": row.get("objective_equivalence_verified"),
        "selected_resolution_legal": row.get("selected_resolution_legal"),
        "legal_tied_candidate_actions": list(
            row.get("legal_tied_candidate_actions")
            if isinstance(row.get("legal_tied_candidate_actions"), list)
            else []
        ),
        "current_archive_row_id": row.get("current_archive_row_id"),
        "repaired_archive_row_id": row.get("repaired_archive_row_id"),
    }


def _candidate_search(
    archive_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    source_integrity: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True:
        return {
            "analysis_blocked": True,
            "blocked_reasons": list(source_integrity.get("failures", [])),
            "required_actions": list(RARE_ACTIONS),
            "per_action": {},
            "all_required_actions_have_candidate": False,
            "any_candidate_available": False,
        }
    contexts, integrity_failures, integrity_examples = _branch_repair_contexts(
        archive_rows
    )
    if integrity_failures:
        return {
            "analysis_blocked": True,
            "blocked_reasons": integrity_failures,
            "branch_integrity_examples": integrity_examples,
            "required_actions": list(RARE_ACTIONS),
            "per_action": {},
            "all_required_actions_have_candidate": False,
            "any_candidate_available": False,
        }
    manifest_by_branch = {
        str(row.get("branch_id")): row
        for row in manifest_rows
        if isinstance(row.get("branch_id"), str) and row.get("branch_id")
    }
    per_action = {
        action: _candidate_search_for_action(action, contexts, manifest_by_branch)
        for action in RARE_ACTIONS
    }
    all_required = all(
        _int(report.get("valid_candidate_count")) > 0
        for report in per_action.values()
    )
    any_available = any(
        _int(report.get("valid_candidate_count")) > 0
        for report in per_action.values()
    )
    return {
        "analysis_blocked": False,
        "candidate_policy": {
            "objective_values_changed": False,
            "candidate_scope": (
                "unselected_legal_candidates_tied_on_v117_serialized_objective_key"
            ),
            "unique_objective_best_policy": "reject_changes_to_unique_best_branches",
            "trainable_input_policy": "candidate_trainable_public_input_must_pass_v120_leakage_guard",
            "selection_authorized": False,
        },
        "required_actions": list(RARE_ACTIONS),
        "per_action": per_action,
        "all_required_actions_have_candidate": all_required,
        "any_candidate_available": any_available,
    }


def _candidate_search_for_action(
    action: str,
    contexts: Sequence[object],
    manifest_by_branch: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    rejection_examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    for context in sorted(contexts, key=lambda item: item.branch_id):
        manifest = manifest_by_branch.get(context.branch_id)
        if manifest is None:
            rejection_counts["branch_missing_from_manifest"] += 1
            _add_rejection_example(
                rejection_examples,
                "branch_missing_from_manifest",
                context,
                action,
            )
            continue
        target_rows = [
            row for row in context.rows if str(row.get("candidate_action")) == action
        ]
        if not target_rows:
            rejection_counts["attack_candidate_absent"] += 1
            continue
        if str(manifest.get("repaired_action")) == action:
            rejection_counts["already_selected_by_v118_action_balance"] += 1
            continue
        if context.unique_objective_best:
            rejection_counts["unique_objective_best_change_rejected"] += 1
            _add_rejection_example(
                rejection_examples,
                "unique_objective_best_change_rejected",
                context,
                action,
            )
            continue
        equivalent = [
            row
            for row in target_rows
            if _serialized_objective_key(row) == context.rank1_key
        ]
        if not equivalent:
            rejection_counts["objective_equivalence_missing"] += 1
            _add_rejection_example(
                rejection_examples,
                "objective_equivalence_missing",
                context,
                action,
            )
            continue
        legal = [row for row in equivalent if row.get("resolution_legal") is True]
        if not legal:
            rejection_counts["resolution_invalid"] += 1
            _add_rejection_example(
                rejection_examples,
                "resolution_invalid",
                context,
                action,
            )
            continue
        clean_rows = [
            row for row in legal if _candidate_trainable_rejection(context, row) is None
        ]
        if not clean_rows:
            rejection = _first_candidate_trainable_rejection(context, legal)
            rejection_counts[rejection] += 1
            _add_rejection_example(
                rejection_examples,
                rejection,
                context,
                action,
            )
            continue
        selected = sorted(clean_rows, key=_candidate_row_key)[0]
        candidates.append(_candidate_example(context, selected, manifest, action))
    candidates = sorted(candidates, key=lambda item: str(item["candidate_sort_key"]))
    return {
        "target_action": action,
        "existing_repaired_branch_count": sum(
            1 for row in manifest_by_branch.values()
            if str(row.get("repaired_action")) == action
        ),
        "valid_candidate_count": len(candidates),
        "valid_candidates": [
            _without_sort_key(candidate) for candidate in candidates[:MAX_EXAMPLES]
        ],
        "top_candidate": (
            _without_sort_key(candidates[0]) if candidates else None
        ),
        "rejection_counts": _counter_to_dict(rejection_counts),
        "rejection_examples": {
            key: value for key, value in sorted(rejection_examples.items())
        },
    }


def _candidate_trainable_clean(
    context: object,
    row: Mapping[str, object],
) -> bool:
    return _candidate_trainable_rejection(context, row) is None


def _first_candidate_trainable_rejection(
    context: object,
    rows: Sequence[Mapping[str, object]],
) -> str:
    for row in rows:
        rejection = _candidate_trainable_rejection(context, row)
        if rejection is not None:
            return rejection
    return "trainable_leakage"


def _candidate_trainable_rejection(
    context: object,
    row: Mapping[str, object],
) -> str | None:
    trainable = row.get("trainable_public_input")
    if not isinstance(trainable, Mapping) or not trainable:
        return "trainable_public_input_missing_or_malformed"
    leakage = _trainable_leakage(
        [
            {
                "branch_id": context.branch_id,
                "repaired_action": row.get("candidate_action"),
                "trainable_public_input": row.get("trainable_public_input"),
            }
        ]
    )
    if (
        leakage["split_key_leak_count"] == 0
        and leakage["forbidden_metadata_key_count"] == 0
    ):
        return None
    return "trainable_leakage"


def _candidate_example(
    context: object,
    selected: Mapping[str, object],
    manifest: Mapping[str, object],
    action: str,
) -> dict[str, object]:
    return {
        "candidate_sort_key": _candidate_row_key(selected),
        "branch_id": context.branch_id,
        "target_action": action,
        "candidate_archive_row_id": selected.get("archive_row_id"),
        "current_oracle_action": manifest.get("current_oracle_action"),
        "current_repaired_action": manifest.get("repaired_action"),
        "current_repaired_archive_row_id": manifest.get("repaired_archive_row_id"),
        "candidate_resolution_legal": selected.get("resolution_legal") is True,
        "candidate_objective_equivalence_verified": (
            _serialized_objective_key(selected) == context.rank1_key
        ),
        "candidate_trainable_public_input_clean": _candidate_trainable_clean(
            context,
            selected,
        ),
        "unique_objective_best": context.unique_objective_best,
        "would_change_unique_objective_best": False,
        "objective_values_changed": False,
        "legal_tied_candidate_actions": sorted(
            str(row.get("candidate_action")) for row in context.legal_equivalent
        ),
        "serialized_objective_key": list(context.rank1_key),
    }


def _candidate_row_key(row: Mapping[str, object]) -> str:
    return f"{row.get('candidate_action')}::{row.get('archive_row_id')}"


def _without_sort_key(candidate: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in candidate.items() if key != "candidate_sort_key"}


def _add_rejection_example(
    examples: dict[str, list[dict[str, object]]],
    reason: str,
    context: object,
    action: str,
) -> None:
    if len(examples[reason]) >= MAX_EXAMPLES:
        return
    examples[reason].append(
        {
            "branch_id": context.branch_id,
            "target_action": action,
            "current_oracle_action": context.current.get("candidate_action"),
            "unique_objective_best": context.unique_objective_best,
            "legal_tied_candidate_actions": sorted(
                str(row.get("candidate_action")) for row in context.legal_equivalent
            ),
        }
    )


def _classification(
    *,
    source_integrity: Mapping[str, object],
    candidate_search: Mapping[str, object],
) -> dict[str, object]:
    labels = ["diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
    failures = [str(item) for item in source_integrity.get("failures", [])]
    if source_integrity.get("passed") is not True:
        primary = (
            "missing_evidence_inconclusive"
            if any(failure.startswith("missing_") for failure in failures)
            else "rare_action_coverage_source_integrity_failed"
        )
        labels.insert(0, primary)
    elif candidate_search.get("all_required_actions_have_candidate") is True:
        primary = "rare_action_coverage_candidate_available"
        labels.insert(0, primary)
    else:
        primary = "rare_action_coverage_not_available_in_existing_archive"
        labels.insert(0, primary)
    return {
        "primary": primary,
        "labels": _dedupe_allowed(labels),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    primary = classification.get("primary")
    if primary in (
        "missing_evidence_inconclusive",
        "rare_action_coverage_source_integrity_failed",
    ):
        next_step = "source_integrity_must_pass_before_rare_action_coverage_targeting"
        summary = (
            "Source integrity failed, so rare-action coverage targeting cannot be "
            "trusted from these inputs."
        )
    elif primary == "rare_action_coverage_candidate_available":
        next_step = (
            "propose_diagnostics_only_repair_policy_for_identified_rare_action_candidates"
        )
        summary = (
            "Existing archive evidence contains legal objective-equivalent rare "
            "attack near-misses, but this diagnostic does not authorize relabeling, "
            "shadow scoring, training, or readiness."
        )
    else:
        next_step = "collect_additional_first_recovery_attack_coverage_diagnostics"
        summary = (
            "Existing v115/v119/v120 artifacts cannot fill the attack_east and "
            "attack_west rare-action gap; collect diagnostics-only coverage for "
            "legal objective-equivalent attack opportunities."
        )
    return {
        "next_step": next_step,
        "summary": summary,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "viewer_change_recommended": False,
        "shadow_scorer_change_recommended": False,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "claim_causality": False,
    }


def _dedupe_allowed(labels: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    allowed = set(ALLOWED_CLASSIFICATIONS)
    result: list[str] = []
    for label in labels:
        if label in allowed and label not in seen:
            result.append(label)
            seen.add(label)
    return result
