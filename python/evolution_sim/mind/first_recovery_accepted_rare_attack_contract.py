from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V123_REPORT_PATH,
    DEFAULT_V122_RERUN_CANDIDATES_OUTPUT_PATH as DEFAULT_V122_OVER_V123_CANDIDATES_PATH,
    DEFAULT_V122_RERUN_OUTPUT_PATH as DEFAULT_V122_OVER_V123_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_ACTIVE_COVERAGE_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    trainable_public_input_leakage,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _int,
    _mapping,
    _resolve_archive_rows,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_rare_attack_coverage_collection import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V122_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION,
    TARGET_ACTIONS,
)
from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V119_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V119_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V120_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION,
    STRICT_TEST_MIN,
    STRICT_TRAIN_MIN,
    STRICT_VALIDATION_MIN,
    _evaluate_split_assignment,
    _stratified_three_way_assignments,
    _trainable_leakage,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_accepted_rare_attack_contract_v1"
)
MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_POLICY = (
    "diagnostics_only_first_recovery_v124_accepted_rare_attack_contract_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-contract.json"
)
DEFAULT_MANIFEST_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-manifest.jsonl"
)

EXPECTED_V119_CLASSIFICATION = "repaired_label_contract_support_limited"
EXPECTED_V120_CLASSIFICATION = "split_support_feasibility_limited_by_rare_actions"
EXPECTED_V122_DEFAULT_CLASSIFICATION = "rare_attack_coverage_not_found_within_budget"
EXPECTED_V123_CLASSIFICATION = "active_coverage_rare_attacks_found"
EXPECTED_V122_OVER_V123_CLASSIFICATION = "rare_attack_coverage_candidates_found"
EXPECTED_ACCEPTED_ADDITIONS = {"attack_east": 1, "attack_west": 1}

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "diagnostics_only_no_runtime_promotion",
    "accepted_rare_attack_contract_ready_for_shadow_proposal",
    "accepted_rare_attack_contract_split_support_failed",
    "accepted_rare_attack_contract_source_integrity_failed",
    "readiness_rerun_blocked",
)

AUTHORIZATION_FIELDS: tuple[str, ...] = (
    "v113_readiness_rerun_allowed",
    "downstream_shadow_scorer_allowed",
    "claim_causality",
    "runtime_policy_change_recommended",
    "trained_artifact_change_recommended",
    "gate_change_recommended",
    "viewer_change_recommended",
    "replay_golden_change_recommended",
)

MAX_EXAMPLES = 12


@dataclass(frozen=True, slots=True)
class FirstRecoveryAcceptedRareAttackContractBuild:
    report: dict[str, object]
    manifest_rows: tuple[dict[str, object], ...]


def build_first_recovery_accepted_rare_attack_contract(
    *,
    v119_report: Mapping[str, object] | None = None,
    v119_report_path: str | Path | None = DEFAULT_V119_REPORT_PATH,
    v119_manifest_rows: Sequence[Mapping[str, object]] | None = None,
    v119_manifest_path: str | Path | None = DEFAULT_V119_MANIFEST_PATH,
    v120_report: Mapping[str, object] | None = None,
    v120_report_path: str | Path | None = DEFAULT_V120_REPORT_PATH,
    v122_default_report: Mapping[str, object] | None = None,
    v122_default_report_path: str | Path | None = DEFAULT_V122_REPORT_PATH,
    v123_report: Mapping[str, object] | None = None,
    v123_report_path: str | Path | None = DEFAULT_V123_REPORT_PATH,
    v123_archive_rows: Sequence[Mapping[str, object]] | None = None,
    v123_archive_rows_path: str | Path | None = DEFAULT_V123_ARCHIVE_ROWS_PATH,
    v122_over_v123_report: Mapping[str, object] | None = None,
    v122_over_v123_report_path: str | Path | None = DEFAULT_V122_OVER_V123_REPORT_PATH,
    v122_over_v123_candidate_rows: Sequence[Mapping[str, object]] | None = None,
    v122_over_v123_candidates_path: str | Path | None = (
        DEFAULT_V122_OVER_V123_CANDIDATES_PATH
    ),
) -> FirstRecoveryAcceptedRareAttackContractBuild:
    v119_payload, v119_evidence = _resolve_json_report(
        v119_report,
        v119_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
    )
    v119_rows, v119_manifest_evidence = _resolve_jsonl_rows(
        v119_manifest_rows,
        v119_manifest_path,
        row_kind="v119_manifest",
    )
    v120_payload, v120_evidence = _resolve_json_report(
        v120_report,
        v120_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION
        ),
    )
    v122_default_payload, v122_default_evidence = _resolve_json_report(
        v122_default_report,
        v122_default_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION,
    )
    v123_payload, v123_evidence = _resolve_json_report(
        v123_report,
        v123_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    v123_rows, v123_rows_evidence = _resolve_archive_rows(
        v123_archive_rows,
        v123_archive_rows_path,
    )
    v122_over_payload, v122_over_evidence = _resolve_json_report(
        v122_over_v123_report,
        v122_over_v123_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION,
    )
    candidate_rows, candidate_evidence = _resolve_jsonl_rows(
        v122_over_v123_candidate_rows,
        v122_over_v123_candidates_path,
        row_kind="v122_over_v123_candidates",
    )
    source_reports = {
        "v119_report": v119_evidence,
        "v119_manifest": v119_manifest_evidence,
        "v120_report": v120_evidence,
        "v122_default_report": v122_default_evidence,
        "v123_report": v123_evidence,
        "v123_archive_rows": v123_rows_evidence,
        "v122_over_v123_report": v122_over_evidence,
        "v122_over_v123_candidates": candidate_evidence,
    }
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v119_report=v119_payload,
        v119_rows=v119_rows,
        v120_report=v120_payload,
        v122_default_report=v122_default_payload,
        v123_report=v123_payload,
        v123_rows=v123_rows,
        v122_over_v123_report=v122_over_payload,
        candidate_rows=candidate_rows,
    )
    manifest_rows = (
        _merged_manifest_rows(
            v119_rows=v119_rows,
            v123_rows=v123_rows,
            candidate_rows=candidate_rows,
        )
        if source_integrity["passed"]
        else tuple()
    )
    contract_checks = _contract_checks(
        manifest_rows=manifest_rows,
        v119_rows=v119_rows,
        candidate_rows=candidate_rows,
        source_integrity=source_integrity,
    )
    split_support = _split_support(manifest_rows)
    source_seed_policy = _source_seed_policy(manifest_rows)
    classification = _classification(
        source_integrity=source_integrity,
        contract_checks=contract_checks,
        split_support=split_support,
    )
    recommendation = _recommendation(classification)
    manifest_digest = stable_payload_digest(list(manifest_rows))
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "accepted_candidate_contract": {
            "source": "v122_over_v123_candidates_joined_to_v123_archive_rows",
            "accepted_additions_required": EXPECTED_ACCEPTED_ADDITIONS,
            "accepted_candidate_count": len(candidate_rows)
            if source_integrity.get("passed") is True
            else 0,
            "selection_authorized": False,
            "training_authorized": False,
            "runtime_policy_authorized": False,
        },
        "manifest": {
            "schema_version": (
                MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION
            ),
            "manifest_path": str(DEFAULT_MANIFEST_OUTPUT_PATH),
            "manifest_row_count": len(manifest_rows),
            "manifest_digest": manifest_digest,
            "base_v119_manifest_row_count": len(v119_rows),
            "accepted_v123_candidate_row_count": (
                len(manifest_rows) - len(v119_rows) if manifest_rows else 0
            ),
            "trainable_public_input_field": "trainable_public_input",
            "non_trainable_audit_metadata_field": "non_trainable_audit_metadata",
        },
        "contract_checks": contract_checks,
        "source_seed_policy": source_seed_policy,
        "split_support": split_support,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryAcceptedRareAttackContractBuild(
        report=report,
        manifest_rows=manifest_rows,
    )


def write_first_recovery_accepted_rare_attack_contract_outputs(
    build: FirstRecoveryAcceptedRareAttackContractBuild,
    *,
    output_path: str | Path,
    manifest_output_path: str | Path,
) -> None:
    manifest = Path(manifest_output_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in build.manifest_rows:
            json.dump(row, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
    report = dict(build.report)
    report["manifest"] = {
        **_mapping(report.get("manifest")),
        "manifest_path": str(manifest),
        "manifest_row_count": len(build.manifest_rows),
        "manifest_digest": stable_payload_digest(list(build.manifest_rows)),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION
        ),
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "viewer_effect": "none",
        "objective_values_changed": False,
        "synthetic_labels_created": False,
        "training_executed": False,
        "readiness_rerun_executed": False,
        "shadow_scorer_implemented": False,
        "source_seed_provenance_trainable": False,
        "strict_heldout_generalization_claimed": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "claim_causality": False,
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION
                ),
                "policy": MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_POLICY,
            }
        ),
    }


def _resolve_jsonl_rows(
    rows: Sequence[Mapping[str, object]] | None,
    path: str | Path | None,
    *,
    row_kind: str,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    if rows is not None:
        parsed = tuple(dict(row) for row in rows)
        return parsed, {
            "loaded": True,
            "in_memory": True,
            "path": str(path) if path is not None else None,
            "row_count": len(parsed),
            "row_kind": row_kind,
        }
    if path is None:
        return (), {
            "loaded": False,
            "path": None,
            "row_count": 0,
            "row_kind": row_kind,
            "error": "missing_jsonl_path",
        }
    source = Path(path)
    if not source.exists():
        return (), {
            "loaded": False,
            "path": str(source),
            "row_count": 0,
            "row_kind": row_kind,
            "error": "missing_jsonl_file",
        }
    parsed: list[dict[str, object]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{row_kind} row {line_number} must be an object")
                parsed.append(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return (), {
            "loaded": False,
            "path": str(source),
            "row_count": 0,
            "row_kind": row_kind,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    return tuple(parsed), {
        "loaded": True,
        "path": str(source),
        "row_count": len(parsed),
        "row_kind": row_kind,
    }


def _source_integrity(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    v119_report: Mapping[str, object] | None,
    v119_rows: Sequence[Mapping[str, object]],
    v120_report: Mapping[str, object] | None,
    v122_default_report: Mapping[str, object] | None,
    v123_report: Mapping[str, object] | None,
    v123_rows: Sequence[Mapping[str, object]],
    v122_over_v123_report: Mapping[str, object] | None,
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")

    v119 = _mapping(v119_report or {})
    v120 = _mapping(v120_report or {})
    v122_default = _mapping(v122_default_report or {})
    v123 = _mapping(v123_report or {})
    v122_over = _mapping(v122_over_v123_report or {})
    manifest_digest = stable_payload_digest(list(v119_rows))
    v119_counts = _action_counts(v119_rows)
    v119_branch_ids = _valid_branch_ids(v119_rows)
    candidate_counts = _candidate_action_counts(candidate_rows)
    v123_selected = _selected_v123_archive_rows(v123_rows, candidate_rows)
    v123_identity = _v123_archive_identity(v123_rows)
    join_validation = _candidate_archive_join_validation(
        candidate_rows=candidate_rows,
        selected_rows=v123_selected,
    )
    row_replay_validation = _accepted_candidate_row_replay_validation(
        candidate_rows=candidate_rows,
        selected_rows=v123_selected,
    )

    if _primary(v119) != EXPECTED_V119_CLASSIFICATION:
        failures.append("v119_classification_unexpected")
    v119_contract = _mapping(v119.get("contract_checks"))
    if v119_contract.get("passed") is not True:
        failures.append("v119_contract_checks_not_passed")
    if v119_contract.get("total_violation_count") != 0:
        failures.append("v119_contract_violations_nonzero")
    if v119_contract.get("integrity_failures") != []:
        failures.append("v119_contract_integrity_failures_not_empty_or_malformed")
    v119_manifest = _mapping(v119.get("manifest"))
    if v119_manifest.get("manifest_digest") != manifest_digest:
        failures.append("v119_manifest_digest_mismatch")
    if _int(v119_manifest.get("manifest_row_count")) != len(v119_rows):
        failures.append("v119_manifest_row_count_mismatch")
    if _mapping(v119_contract.get("repaired_action_counts")) and {
        str(action): _int(count)
        for action, count in _mapping(
            v119_contract.get("repaired_action_counts")
        ).items()
    } != dict(v119_counts):
        failures.append("v119_repaired_action_counts_mismatch")

    if _primary(v120) != EXPECTED_V120_CLASSIFICATION:
        failures.append("v120_classification_unexpected")
    v120_source = _mapping(v120.get("source_integrity"))
    if v120_source.get("passed") is not True:
        failures.append("v120_source_integrity_not_passed")
    if v120_source.get("failures") != []:
        failures.append("v120_source_failures_not_empty_or_malformed")
    v120_needed = {
        str(action): _int(count)
        for action, count in _mapping(
            _mapping(v120.get("scarcity_analysis")).get(
                "rare_action_additional_needed_for_train2_validation1_test1"
            )
        ).items()
        if str(action) in TARGET_ACTIONS
    }
    if v120_needed != EXPECTED_ACCEPTED_ADDITIONS:
        failures.append("v120_rare_action_additions_unexpected")

    if _primary(v122_default) != EXPECTED_V122_DEFAULT_CLASSIFICATION:
        failures.append("v122_default_classification_unexpected")
    default_source = _mapping(v122_default.get("source_integrity"))
    if default_source.get("passed") is not True:
        failures.append("v122_default_source_integrity_not_passed")
    default_domain = _mapping(
        _mapping(v122_default.get("candidate_collection")).get(
            "candidate_source_domain"
        )
    )
    if _int(default_domain.get("unmanifested_candidate_branch_count")) != 0:
        failures.append("v122_default_unmanifested_branch_count_not_zero")
    default_labels = _list_like(_mapping(v122_default.get("classification")).get("labels"))
    if "candidate_source_contains_no_unmanifested_branches" not in default_labels:
        failures.append("v122_default_no_unmanifested_label_missing")
    if _target_counts(v122_default.get("found_candidate_counts")) != {
        action: 0 for action in TARGET_ACTIONS
    }:
        failures.append("v122_default_found_candidate_counts_unexpected")

    if _primary(v123) != EXPECTED_V123_CLASSIFICATION:
        failures.append("v123_classification_unexpected")
    if (
        v123.get("active_coverage_schema_version")
        != MIND_V3_FIRST_RECOVERY_ACTIVE_COVERAGE_ARCHIVE_SCHEMA_VERSION
    ):
        failures.append("v123_active_schema_mismatch")
    v123_source = _mapping(v123.get("source_integrity"))
    if v123_source.get("passed") is not True:
        failures.append("v123_source_integrity_not_passed")
    if v123_source.get("failures") != []:
        failures.append("v123_source_failures_not_empty_or_malformed")
    if _int(v123.get("generated_archive_row_count")) != len(v123_rows):
        failures.append("v123_archive_row_count_mismatch")
    if _int(v123.get("generated_branch_count")) != len(_archive_branch_ids(v123_rows)):
        failures.append("v123_generated_branch_count_mismatch")
    if _target_counts(v123.get("accepted_by_v122_candidate_counts")) != (
        EXPECTED_ACCEPTED_ADDITIONS
    ):
        failures.append("v123_accepted_candidate_counts_unexpected")
    replay = _mapping(v123.get("replay_verification"))
    if v123.get("replay_verified") is not True or replay.get("replay_verified") is not True:
        failures.append("v123_replay_not_verified")
    if _int(replay.get("missing_replay_verification_count")) != 0:
        failures.append("v123_replay_verification_missing")
    if _int(replay.get("replay_verification_skipped_count")) != 0:
        failures.append("v123_replay_verification_skipped")
    if _int(replay.get("replay_verification_failure_count")) != 0:
        failures.append("v123_replay_verification_failed")
    if _int(v123.get("heuristic_action_source_count")) != 0:
        failures.append("v123_heuristic_action_source_nonzero")
    if _int(v123.get("strict_seed_leakage_count")) != 0:
        failures.append("v123_strict_seed_leakage_nonzero")
    if v123_identity["identity_failure_count"]:
        failures.append("v123_archive_row_identity_failure")

    v123_leakage = _mapping(v123.get("trainable_leakage"))
    for key in ("split_key_leak_count", "forbidden_metadata_key_count"):
        if _int(v123_leakage.get(key)) != 0:
            failures.append(f"v123_trainable_{key}_nonzero")
    branch_archive_leakage = _mapping(v123_leakage.get("branch_archive_leakage"))
    if _int(branch_archive_leakage.get("leakage_count")) != 0:
        failures.append("v123_branch_archive_trainable_leakage")

    if _primary(v122_over) != EXPECTED_V122_OVER_V123_CLASSIFICATION:
        failures.append("v122_over_v123_classification_unexpected")
    over_source = _mapping(v122_over.get("source_integrity"))
    if over_source.get("passed") is not True:
        failures.append("v122_over_v123_source_integrity_not_passed")
    if _target_counts(v122_over.get("found_candidate_counts")) != (
        EXPECTED_ACCEPTED_ADDITIONS
    ):
        failures.append("v122_over_v123_found_candidate_counts_unexpected")
    if _mapping(v122_over.get("recommendation")).get(
        "would_clear_v120_rare_action_limitation_if_accepted"
    ) is not True:
        failures.append("v122_over_v123_clearance_not_reported")
    if len(candidate_rows) != sum(EXPECTED_ACCEPTED_ADDITIONS.values()):
        failures.append("accepted_candidate_row_count_unexpected")
    if candidate_counts != EXPECTED_ACCEPTED_ADDITIONS:
        failures.append("accepted_candidate_action_counts_unexpected")
    candidate_branch_ids = _valid_branch_ids(candidate_rows)
    if len(candidate_branch_ids) != len(set(candidate_branch_ids)):
        failures.append("accepted_candidate_branch_ids_not_unique")
    if set(candidate_branch_ids) & set(v119_branch_ids):
        failures.append("accepted_candidate_branch_already_in_v119")
    failures.extend(join_validation["failure_labels"])
    failures.extend(row_replay_validation["failure_labels"])
    failures.extend(
        _candidate_contract_failures(
            candidate_rows=candidate_rows,
            selected_rows=v123_selected,
        )
    )

    for version, report in (
        ("v119", v119),
        ("v120", v120),
        ("v122_default", v122_default),
        ("v123", v123),
        ("v122_over_v123", v122_over),
    ):
        failures.extend(_no_authorization_failures(version, report))

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v119_manifest_row_count": len(v119_rows),
        "v119_manifest_digest": manifest_digest,
        "v119_repaired_action_counts": _counter_to_dict(v119_counts),
        "v120_rare_action_additional_needed": v120_needed,
        "v122_default_unmanifested_candidate_branch_count": _int(
            default_domain.get("unmanifested_candidate_branch_count")
        ),
        "v123_archive_row_count": len(v123_rows),
        "v123_archive_branch_count": len(_archive_branch_ids(v123_rows)),
        "v123_archive_identity": v123_identity,
        "v123_accepted_by_v122_candidate_counts": _target_counts(
            v123.get("accepted_by_v122_candidate_counts")
        ),
        "v122_over_v123_candidate_row_count": len(candidate_rows),
        "v122_over_v123_candidate_counts": candidate_counts,
        "selected_v123_archive_row_count": len(v123_selected),
        "accepted_candidate_join_validation": join_validation,
        "accepted_candidate_row_replay_validation": row_replay_validation,
        "authorization_compatibility": {
            "legacy_missing_authorization_fields_normalized_to_false": {
                version: _legacy_missing_authorization_fields(report, version)
                for version, report in (
                    ("v119", v119),
                    ("v120", v120),
                    ("v122_default", v122_default),
                    ("v123", v123),
                    ("v122_over_v123", v122_over),
                )
            }
        },
        "authorization_checks": {
            version: _authorization_fields(version, report)
            for version, report in (
                ("v119", v119),
                ("v120", v120),
                ("v122_default", v122_default),
                ("v123", v123),
                ("v122_over_v123", v122_over),
            )
        },
    }


def _candidate_archive_join_validation(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    selected_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    for index, candidate in enumerate(candidate_rows):
        branch_id = candidate.get("branch_id")
        target_action = candidate.get("target_action")
        candidate_row_id = candidate.get("candidate_archive_row_id")
        current_row_id = candidate.get("current_archive_row_id")
        selected = selected_rows.get(str(candidate_row_id))
        selected_provenance = _mapping(_mapping(selected or {}).get("provenance"))
        expected_prefix = (
            f"{branch_id}::action::" if isinstance(branch_id, str) and branch_id else None
        )

        if not isinstance(branch_id, str) or not branch_id:
            _record_join_failure(
                counts,
                examples,
                "accepted_candidate_branch_id_missing_or_malformed",
                index,
                candidate,
                selected,
            )
        if not isinstance(candidate_row_id, str) or not candidate_row_id:
            _record_join_failure(
                counts,
                examples,
                "accepted_candidate_archive_row_id_missing_or_malformed",
                index,
                candidate,
                selected,
            )
        elif expected_prefix is not None and not candidate_row_id.startswith(
            expected_prefix
        ):
            _record_join_failure(
                counts,
                examples,
                "accepted_candidate_archive_row_id_branch_prefix_mismatch",
                index,
                candidate,
                selected,
            )
        if (
            isinstance(current_row_id, str)
            and current_row_id
            and expected_prefix is not None
            and not current_row_id.startswith(expected_prefix)
        ):
            _record_join_failure(
                counts,
                examples,
                "accepted_candidate_current_archive_row_id_branch_prefix_mismatch",
                index,
                candidate,
                selected,
            )
        if isinstance(candidate_row_id, str) and candidate_row_id:
            suffix = (
                candidate_row_id.rsplit("::action::", 1)[1]
                if "::action::" in candidate_row_id
                else None
            )
            if suffix != target_action:
                _record_join_failure(
                    counts,
                    examples,
                    "accepted_candidate_archive_row_id_action_suffix_mismatch",
                    index,
                    candidate,
                    selected,
                )
        if selected:
            if selected_provenance.get("branch_id") != branch_id:
                _record_join_failure(
                    counts,
                    examples,
                    "accepted_candidate_selected_provenance_branch_mismatch",
                    index,
                    candidate,
                    selected,
                )
            if selected.get("candidate_action") != target_action:
                _record_join_failure(
                    counts,
                    examples,
                    "accepted_candidate_selected_action_mismatch",
                    index,
                    candidate,
                    selected,
                )
            if selected.get("archive_row_id") != candidate_row_id:
                _record_join_failure(
                    counts,
                    examples,
                    "accepted_candidate_selected_archive_row_id_mismatch",
                    index,
                    candidate,
                    selected,
                )
    return {
        "passed": not counts,
        "failure_labels": sorted(counts),
        "mismatch_count": sum(counts.values()),
        "mismatch_counts": _counter_to_dict(counts),
        "examples": examples,
    }


def _accepted_candidate_row_replay_validation(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    selected_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    for index, candidate in enumerate(candidate_rows):
        selected = selected_rows.get(str(candidate.get("candidate_archive_row_id")))
        if not selected:
            continue
        if selected.get("replay_verification_result") is not True:
            _record_join_failure(
                counts,
                examples,
                "accepted_candidate_row_replay_result_not_true",
                index,
                candidate,
                selected,
            )
        if not _valid_sha256_digest(selected.get("replay_verification_digest")):
            _record_join_failure(
                counts,
                examples,
                "accepted_candidate_row_replay_digest_missing_or_malformed",
                index,
                candidate,
                selected,
            )
    return {
        "passed": not counts,
        "failure_labels": sorted(counts),
        "failure_count": sum(counts.values()),
        "failure_counts": _counter_to_dict(counts),
        "examples": examples,
    }


def _record_join_failure(
    counts: Counter[str],
    examples: list[dict[str, object]],
    label: str,
    index: int,
    candidate: Mapping[str, object],
    selected: Mapping[str, object] | None,
) -> None:
    counts[label] += 1
    if len(examples) >= MAX_EXAMPLES:
        return
    selected_payload = _mapping(selected or {})
    examples.append(
        {
            "label": label,
            "candidate_index": index,
            "branch_id": candidate.get("branch_id"),
            "target_action": candidate.get("target_action"),
            "candidate_archive_row_id": candidate.get("candidate_archive_row_id"),
            "current_archive_row_id": candidate.get("current_archive_row_id"),
            "selected_archive_row_id": selected_payload.get("archive_row_id"),
            "selected_provenance_branch_id": _mapping(
                selected_payload.get("provenance")
            ).get("branch_id"),
            "selected_candidate_action": selected_payload.get("candidate_action"),
            "selected_replay_verification_result": selected_payload.get(
                "replay_verification_result"
            ),
            "selected_replay_verification_digest": selected_payload.get(
                "replay_verification_digest"
            ),
        }
    )


def _candidate_contract_failures(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    selected_rows: Mapping[str, Mapping[str, object]],
) -> list[str]:
    failures: list[str] = []
    for candidate in candidate_rows:
        action = str(candidate.get("repaired_action"))
        row_id = candidate.get("candidate_archive_row_id")
        if action not in TARGET_ACTIONS:
            failures.append("accepted_candidate_action_not_target_rare_attack")
        if candidate.get("target_action") != action:
            failures.append("accepted_candidate_target_repaired_action_mismatch")
        if candidate.get("selection_authorized") is not False:
            failures.append("accepted_candidate_selection_authorized_not_false")
        if candidate.get("selected_resolution_legal") is not True:
            failures.append("accepted_candidate_resolution_not_legal")
        if candidate.get("objective_equivalence_verified") is not True:
            failures.append("accepted_candidate_objective_equivalence_not_verified")
        if candidate.get("unique_objective_best") is True and candidate.get("changed") is True:
            failures.append("accepted_candidate_unique_best_changed")
        if candidate.get("trainable_public_input_present") is not True:
            failures.append("accepted_candidate_trainable_input_not_present")
        if candidate.get("trainable_public_input_clean") is not True:
            failures.append("accepted_candidate_trainable_input_not_clean")
        if candidate.get("trainable_public_input_contents_exposed") is not False:
            failures.append("accepted_candidate_trainable_contents_exposed")
        selected = selected_rows.get(str(row_id))
        if not selected:
            failures.append("accepted_candidate_v123_archive_row_missing")
            continue
        if selected.get("candidate_action") != action:
            failures.append("accepted_candidate_v123_action_mismatch")
        if selected.get("resolution_legal") is not True:
            failures.append("accepted_candidate_v123_resolution_not_legal")
        if selected.get("observation_legal") is not True:
            failures.append("accepted_candidate_v123_observation_not_legal")
        trainable = selected.get("trainable_public_input")
        if not isinstance(trainable, Mapping) or not trainable:
            failures.append("accepted_candidate_trainable_input_missing_or_malformed")
            continue
        leakage = _trainable_leakage(
            [
                {
                    "branch_id": candidate.get("branch_id"),
                    "repaired_action": action,
                    "trainable_public_input": trainable,
                }
            ]
        )
        if leakage["split_key_leak_count"] or leakage["forbidden_metadata_key_count"]:
            failures.append("accepted_candidate_trainable_input_leakage")
    return failures


def _merged_manifest_rows(
    *,
    v119_rows: Sequence[Mapping[str, object]],
    v123_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    selected_rows = _selected_v123_archive_rows(v123_rows, candidate_rows)
    merged = [_normalize_v119_manifest_row(row) for row in v119_rows]
    for candidate in sorted(
        candidate_rows,
        key=lambda row: (str(row.get("target_action")), str(row.get("branch_id"))),
    ):
        selected = selected_rows[str(candidate.get("candidate_archive_row_id"))]
        merged.append(_accepted_candidate_manifest_row(candidate, selected))
    return tuple(merged)


def _normalize_v119_manifest_row(row: Mapping[str, object]) -> dict[str, object]:
    payload = dict(row)
    payload["schema_version"] = (
        MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION
    )
    payload["diagnostics_only"] = True
    payload["selection_authorized"] = False
    payload["training_authorized"] = False
    payload["runtime_policy_authorized"] = False
    payload["source_contract"] = "v119_repaired_label_manifest"
    return payload


def _accepted_candidate_manifest_row(
    candidate: Mapping[str, object],
    selected: Mapping[str, object],
) -> dict[str, object]:
    provenance = _mapping(selected.get("provenance"))
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION
        ),
        "branch_id": candidate.get("branch_id"),
        "changed": candidate.get("changed"),
        "current_archive_row_id": candidate.get("current_archive_row_id"),
        "current_oracle_action": candidate.get("current_oracle_action"),
        "legal_tied_candidate_actions": list(
            candidate.get("legal_tied_candidate_actions", [])
        ),
        "non_trainable_audit_metadata": {
            "non_trainable": True,
            "purpose": "audit_only_not_trainable",
            "source_contract": "v123_active_coverage_archive",
            "source_candidate_contract": "v122_over_v123_diagnostics_candidate",
            "source_seed_policy": "source_seed_provenance_audit_only_not_trainable",
            "selection_authorized": False,
            "provenance": dict(provenance),
            "seed": provenance.get("seed"),
            "source_kind": provenance.get("source_kind"),
            "source_path": provenance.get("source_path"),
            "record_index": provenance.get("record_index"),
            "tick": selected.get("tick"),
            "agent_id": provenance.get("agent_id"),
        },
        "objective_equivalence_verified": candidate.get(
            "objective_equivalence_verified"
        ),
        "repaired_action": candidate.get("repaired_action"),
        "repaired_archive_row_id": candidate.get("candidate_archive_row_id"),
        "selected_observation_digest": selected.get("observation_digest"),
        "selected_resolution_legal": candidate.get("selected_resolution_legal"),
        "serialized_objective_key": list(candidate.get("serialized_objective_key", [])),
        "trainable_public_input": selected.get("trainable_public_input"),
        "unique_objective_best": candidate.get("unique_objective_best"),
        "diagnostics_only": True,
        "selection_authorized": False,
        "training_authorized": False,
        "runtime_policy_authorized": False,
    }


def _contract_checks(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    v119_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    source_integrity: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    row_count = len(manifest_rows)
    branch_ids = _valid_branch_ids(manifest_rows)
    counts = _action_counts(manifest_rows)
    v119_counts = _action_counts(v119_rows)
    expected_counts = dict(v119_counts)
    for action, addition in EXPECTED_ACCEPTED_ADDITIONS.items():
        expected_counts[action] = expected_counts.get(action, 0) + addition
    leakage = _trainable_leakage(manifest_rows)
    branch_archive_leakage = trainable_public_input_leakage(
        [
            {
                "archive_row_id": row.get("repaired_archive_row_id"),
                "candidate_action": row.get("repaired_action"),
                "trainable_public_input": row.get("trainable_public_input"),
            }
            for row in manifest_rows
        ]
    )
    if source_integrity.get("passed") is not True:
        failures.append("source_integrity_not_passed")
    if row_count != len(v119_rows) + sum(EXPECTED_ACCEPTED_ADDITIONS.values()):
        failures.append("manifest_row_count_unexpected")
    if len(branch_ids) != row_count:
        failures.append("manifest_branch_ids_missing_or_malformed")
    if len(set(branch_ids)) != len(branch_ids):
        failures.append("manifest_branch_ids_not_unique")
    if dict(counts) != expected_counts:
        failures.append("repaired_action_counts_unexpected")
    if _candidate_action_counts(candidate_rows) != EXPECTED_ACCEPTED_ADDITIONS:
        failures.append("accepted_candidate_action_counts_unexpected")
    if leakage["split_key_leak_count"]:
        failures.append("trainable_split_assignment_leakage")
    if leakage["forbidden_metadata_key_count"]:
        failures.append("trainable_metadata_leakage")
    if _int(branch_archive_leakage.get("leakage_count")) != 0:
        failures.append("branch_archive_trainable_leakage")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "manifest_row_count": row_count,
        "expected_manifest_row_count": len(v119_rows)
        + sum(EXPECTED_ACCEPTED_ADDITIONS.values()),
        "unique_branch_count": len(set(branch_ids)),
        "repaired_action_counts": _counter_to_dict(counts),
        "expected_repaired_action_counts": dict(sorted(expected_counts.items())),
        "base_v119_repaired_action_counts": _counter_to_dict(v119_counts),
        "accepted_candidate_action_counts": _candidate_action_counts(candidate_rows),
        "trainable_leakage": leakage,
        "branch_archive_trainable_leakage": branch_archive_leakage,
        "total_violation_count": len(set(failures)),
    }


def _split_support(manifest_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not manifest_rows:
        return {
            "analysis_blocked": True,
            "blocked_reason": "manifest_unavailable",
            "strict_train_validation_test_support_met": False,
        }
    evaluation = _evaluate_split_assignment(
        manifest_rows,
        assignments=_stratified_three_way_assignments(manifest_rows),
        split_names=("train", "validation", "test"),
        policy_metadata={
            "method": "deterministic_stratified_by_repaired_action",
            "split_assignment_inputs": ["branch_id", "repaired_action"],
            "uses_only_allowed_audit_metadata": True,
            "excluded_from_trainable_public_input": (
                _trainable_leakage(manifest_rows)["split_key_leak_count"] == 0
            ),
            "assignment_rule": (
                "per_action_hash_order_assign_first_to_test_second_to_validation_rest_to_train"
            ),
        },
    )
    splits = _mapping(evaluation.get("splits"))
    strict = bool(evaluation.get("train2_validation1_test1_target_met"))
    return {
        "analysis_blocked": False,
        "policy": evaluation.get("policy"),
        "splits": splits,
        "all_splits_contain_every_action_class": evaluation.get(
            "all_splits_contain_every_action_class"
        ),
        "minimum_per_action_support_by_split": evaluation.get(
            "minimum_per_action_support_by_split"
        ),
        "strict_targets": {
            "train_min": STRICT_TRAIN_MIN,
            "validation_min": STRICT_VALIDATION_MIN,
            "test_min": STRICT_TEST_MIN,
        },
        "strict_train_validation_test_support_met": strict,
    }


def _source_seed_policy(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    trainable_seed_leak_count = 0
    audit_seed_count = 0
    examples: list[dict[str, object]] = []
    for row in manifest_rows:
        metadata = _mapping(row.get("non_trainable_audit_metadata"))
        if "seed" in metadata or "provenance" in metadata:
            audit_seed_count += 1
        trainable = row.get("trainable_public_input")
        if isinstance(trainable, Mapping) and _trainable_contains_key(trainable, "seed"):
            trainable_seed_leak_count += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append(
                    {
                        "branch_id": row.get("branch_id"),
                        "repaired_action": row.get("repaired_action"),
                    }
                )
    return {
        "source_seed_provenance_may_exist_in_audit_metadata": True,
        "source_seed_in_trainable_input": trainable_seed_leak_count > 0,
        "source_seed_trainable_leak_count": trainable_seed_leak_count,
        "audit_metadata_seed_reference_count": audit_seed_count,
        "strict_heldout_generalization_claimed": False,
        "claim_causality": False,
        "examples": examples,
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    contract_checks: Mapping[str, object],
    split_support: Mapping[str, object],
) -> dict[str, object]:
    labels = ["diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
    if (
        source_integrity.get("passed") is not True
        or contract_checks.get("passed") is not True
    ):
        primary = "accepted_rare_attack_contract_source_integrity_failed"
    elif split_support.get("strict_train_validation_test_support_met") is not True:
        primary = "accepted_rare_attack_contract_split_support_failed"
    else:
        primary = "accepted_rare_attack_contract_ready_for_shadow_proposal"
    labels.insert(0, primary)
    return {
        "primary": primary,
        "labels": _dedupe_allowed(labels),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    primary = classification.get("primary")
    if primary == "accepted_rare_attack_contract_ready_for_shadow_proposal":
        next_step = "prepare_separate_diagnostics_only_shadow_scorer_proposal"
        summary = (
            "The accepted rare-attack manifest clears the deterministic split-support "
            "target, but this contract still does not authorize training, shadow "
            "scoring execution, readiness, runtime policy, or promotion."
        )
    elif primary == "accepted_rare_attack_contract_split_support_failed":
        next_step = "collect_additional_first_recovery_rare_action_coverage_diagnostics"
        summary = (
            "The accepted rare-attack rows did not clear deterministic split support."
        )
    else:
        next_step = "source_integrity_must_pass_before_accepting_rare_attack_manifest"
        summary = (
            "Source integrity failed, so the accepted rare-attack contract cannot be "
            "used for any downstream proposal."
        )
    return {
        "next_step": next_step,
        "summary": summary,
        "training_executed": False,
        "trained_artifact_change_recommended": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "runtime_policy_change_recommended": False,
        "gate_change_recommended": False,
        "viewer_change_recommended": False,
        "replay_golden_change_recommended": False,
        "claim_causality": False,
    }


def _selected_v123_archive_rows(
    v123_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    by_id = {
        str(row.get("archive_row_id")): row
        for row in v123_rows
        if isinstance(row.get("archive_row_id"), str)
    }
    selected: dict[str, Mapping[str, object]] = {}
    for candidate in candidate_rows:
        row_id = str(candidate.get("candidate_archive_row_id"))
        if row_id in by_id:
            selected[row_id] = by_id[row_id]
    return selected


def _v123_archive_identity(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        reason = _row_identity_reason(row)
        if reason is None:
            continue
        if len(failures) < MAX_EXAMPLES:
            failures.append(
                {
                    "row_index": index,
                    "archive_row_id": row.get("archive_row_id"),
                    "branch_id": _mapping(row.get("provenance")).get("branch_id"),
                    "candidate_action": row.get("candidate_action"),
                    "reason": reason,
                }
            )
    return {
        "row_count": len(rows),
        "identity_failure_count": sum(
            1 for row in rows if _row_identity_reason(row) is not None
        ),
        "identity_failure_examples": failures,
    }


def _row_identity_reason(row: Mapping[str, object]) -> str | None:
    branch_id = _mapping(row.get("provenance")).get("branch_id")
    row_id = row.get("archive_row_id")
    action = row.get("candidate_action")
    if not isinstance(branch_id, str) or not branch_id:
        return "provenance_branch_id_missing_or_malformed"
    if not isinstance(row_id, str) or not row_id:
        return "archive_row_id_missing_or_malformed"
    prefix = f"{branch_id}::action::"
    if not row_id.startswith(prefix):
        return "archive_row_id_branch_prefix_mismatch"
    if row_id.removeprefix(prefix) != str(action):
        return "archive_row_id_candidate_action_suffix_mismatch"
    return None


def _no_authorization_failures(
    version: str,
    report: Mapping[str, object],
) -> list[str]:
    recommendation = _mapping(report.get("recommendation"))
    failures: list[str] = []
    for field in AUTHORIZATION_FIELDS:
        if field not in recommendation:
            if field == "replay_golden_change_recommended" and version in {
                "v119",
                "v120",
            }:
                continue
            failures.append(f"{version}_{field}_missing")
        elif recommendation.get(field) is not False:
            failures.append(f"{version}_{field}_not_false")
    return failures


def _authorization_fields(
    version: str,
    report: Mapping[str, object],
) -> dict[str, object]:
    recommendation = _mapping(report.get("recommendation"))
    normalized: dict[str, object] = {}
    for field in AUTHORIZATION_FIELDS:
        value = recommendation.get(field)
        if (
            value is None
            and field == "replay_golden_change_recommended"
            and version in {"v119", "v120"}
        ):
            value = False
        normalized[field] = value
    return normalized


def _legacy_missing_authorization_fields(
    report: Mapping[str, object],
    version: str,
) -> list[str]:
    recommendation = _mapping(report.get("recommendation"))
    return [
        field
        for field in AUTHORIZATION_FIELDS
        if field not in recommendation
        and field == "replay_golden_change_recommended"
        and version in {"v119", "v120"}
    ]


def _primary(report: Mapping[str, object]) -> object:
    return _mapping(report.get("classification")).get("primary")


def _action_counts(rows: Sequence[Mapping[str, object]]) -> Counter[str]:
    return Counter(str(row.get("repaired_action")) for row in rows)


def _candidate_action_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        action: sum(1 for row in rows if row.get("repaired_action") == action)
        for action in TARGET_ACTIONS
    }


def _target_counts(payload: object) -> dict[str, int]:
    mapping = _mapping(payload)
    return {action: _int(mapping.get(action)) for action in TARGET_ACTIONS}


def _valid_branch_ids(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return [
        str(row.get("branch_id"))
        for row in rows
        if isinstance(row.get("branch_id"), str) and row.get("branch_id")
    ]


def _archive_branch_ids(rows: Sequence[Mapping[str, object]]) -> set[str]:
    return {
        str(_mapping(row.get("provenance")).get("branch_id"))
        for row in rows
        if isinstance(_mapping(row.get("provenance")).get("branch_id"), str)
        and _mapping(row.get("provenance")).get("branch_id")
    }


def _trainable_contains_key(value: object, key: str) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(item_key) == key or _trainable_contains_key(item, key)
            for item_key, item in value.items()
        )
    if isinstance(value, list):
        return any(_trainable_contains_key(item, key) for item in value)
    return False


def _list_like(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _valid_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _dedupe_allowed(labels: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    allowed = set(ALLOWED_CLASSIFICATIONS)
    result: list[str] = []
    for label in labels:
        if label not in allowed or label in seen:
            continue
        seen.add(label)
        result.append(label)
    return result
