from __future__ import annotations

import hashlib
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
    FORBIDDEN_TRAINABLE_KEYS,
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_oracle_tie_break_audit import (
    _serialized_objective_key,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _int,
    _list,
    _mapping,
    _resolve_archive_rows,
    _resolve_json_report,
    _share,
)
from evolution_sim.mind.first_recovery_tie_aware_label_repair import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V118_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
    _action_balance_selections,
    _branch_repair_contexts,
    _summarize_policy_report,
    build_first_recovery_tie_aware_label_repair,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_repaired_label_contract_audit_v1"
)
MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_POLICY = (
    "diagnostics_only_first_recovery_v119_repaired_label_contract_audit_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v119-first-recovery-repaired-label-contract-audit.json"
)
DEFAULT_MANIFEST_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v119-first-recovery-repaired-label-manifest.jsonl"
)

REPAIRED_POLICY_NAME = "action_balance_resolution_legal"
EXPECTED_V115_BRANCH_COUNT = 106
MIN_SPLIT_ROW_COUNT = 10
MIN_SPLIT_ACTION_SUPPORT = 1
MAX_EXAMPLES = 16

AUDIT_METADATA_TRAINABLE_KEYS = frozenset(
    {
        "audit_metadata",
        "non_trainable_audit_metadata",
        "provenance",
        "source_metadata",
    }
)

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "diagnostics_only_no_runtime_promotion",
    "missing_evidence_inconclusive",
    "source_archive_verification_failed",
    "objective_input_field_integrity",
    "repaired_label_contract_ready_for_shadow_scorer_proposal",
    "repaired_label_contract_support_limited",
    "repaired_label_contract_failed",
    "readiness_rerun_blocked",
)


@dataclass(frozen=True, slots=True)
class FirstRecoveryRepairedLabelContractAuditBuild:
    report: dict[str, object]
    manifest_rows: tuple[dict[str, object], ...]


def build_first_recovery_repaired_label_contract_audit(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    archive_rows: Sequence[Mapping[str, object]] | None = None,
    archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    v118_report: Mapping[str, object] | None = None,
    v118_report_path: str | Path | None = DEFAULT_V118_REPORT_PATH,
    strict_source_verification: bool | None = None,
) -> FirstRecoveryRepairedLabelContractAuditBuild:
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
    strict = (
        _mapping(archive_evidence).get("in_memory") is not True
        if strict_source_verification is None
        else bool(strict_source_verification)
    )
    computed_v118 = build_first_recovery_tie_aware_label_repair(
        archive_report=archive_payload,
        archive_report_path=archive_report_path,
        archive_rows=rows,
        archive_rows_path=archive_rows_path,
        strict_source_verification=strict,
    ).report
    source_reports = _source_reports(
        archive_report=archive_payload,
        archive_evidence=archive_evidence,
        archive_rows=rows,
        rows_evidence=rows_evidence,
        v118_evidence=v118_evidence,
        strict_source_verification=strict,
    )
    source_verification = dict(
        _mapping(computed_v118.get("source_archive_verification"))
    )
    objective_input_integrity = dict(
        _mapping(computed_v118.get("objective_input_integrity"))
    )
    missing_evidence = _missing_evidence(source_reports=source_reports, rows=rows)
    manifest_rows, reconstruction = _repaired_manifest_rows(
        rows,
        source_verification=source_verification,
        objective_input_integrity=objective_input_integrity,
        missing_evidence=missing_evidence,
        strict_source_verification=strict,
    )
    computed_policy = _mapping(reconstruction.get("computed_policy_report"))
    v118_policy = _v118_policy_report(v118_payload)
    contract_checks = _contract_checks(
        manifest_rows=manifest_rows,
        computed_policy_report=computed_policy,
        v118_policy_report=v118_policy,
        strict_source_verification=strict,
    )
    split_support = _split_support_diagnostics(manifest_rows)
    classification = _classification(
        missing_evidence=missing_evidence,
        source_verification=source_verification,
        objective_input_integrity=objective_input_integrity,
        contract_checks=contract_checks,
        split_support=split_support,
        strict_source_verification=strict,
    )
    recommendation = _recommendation(classification=classification)
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_archive_verification": source_verification,
        "objective_input_integrity": objective_input_integrity,
        "v118_reference": _v118_reference(v118_payload),
        "manifest": {
            "schema_version": (
                MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION
            ),
            "manifest_row_count": len(manifest_rows),
            "manifest_digest": stable_payload_digest(list(manifest_rows)),
            "policy_name": REPAIRED_POLICY_NAME,
            "trainable_public_input_field": "trainable_public_input",
            "non_trainable_audit_metadata_field": "non_trainable_audit_metadata",
        },
        "reconstruction": reconstruction,
        "contract_checks": contract_checks,
        "split_support": split_support,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryRepairedLabelContractAuditBuild(
        report=report,
        manifest_rows=tuple(manifest_rows),
    )


def write_first_recovery_repaired_label_contract_audit_outputs(
    build: FirstRecoveryRepairedLabelContractAuditBuild,
    *,
    output_path: str | Path,
    manifest_output_path: str | Path,
) -> None:
    output = Path(output_path)
    manifest = Path(manifest_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in build.manifest_rows:
            json.dump(row, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
    report = dict(build.report)
    report["manifest"] = {
        **_mapping(report.get("manifest")),
        "manifest_path": str(manifest),
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION
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
        "claim_causality": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "diagnostic_policy": (
            "read_only_v115_v118_repaired_label_contract_and_manifest_audit"
        ),
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION
                ),
                "audit_policy": (
                    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_POLICY
                ),
            }
        ),
    }


def _source_reports(
    *,
    archive_report: Mapping[str, object] | None,
    archive_evidence: Mapping[str, object],
    archive_rows: Sequence[Mapping[str, object]],
    rows_evidence: Mapping[str, object],
    v118_evidence: Mapping[str, object],
    strict_source_verification: bool,
) -> dict[str, object]:
    return {
        "archive_report": archive_evidence,
        "archive_rows": rows_evidence,
        "v118_report": v118_evidence,
        "archive_schema_version": _mapping(archive_report or {}).get("schema_version"),
        "archive_row_count": len(archive_rows),
        "strict_source_verification": bool(strict_source_verification),
        "source_report_policy": "inputs_are_read_only_diagnostics_sources",
    }


def _v118_policy_report(v118_report: Mapping[str, object] | None) -> dict[str, object]:
    repair = _mapping(_mapping(v118_report or {}).get("tie_aware_label_repair"))
    policies = _mapping(repair.get("policies"))
    return dict(_mapping(policies.get(REPAIRED_POLICY_NAME)))


def _v118_reference(v118_report: Mapping[str, object] | None) -> dict[str, object]:
    repair = _mapping(_mapping(v118_report or {}).get("tie_aware_label_repair"))
    classification = _mapping(_mapping(v118_report or {}).get("classification"))
    recommendation = _mapping(_mapping(v118_report or {}).get("recommendation"))
    policy = _mapping(_mapping(repair.get("policies")).get(REPAIRED_POLICY_NAME))
    return {
        "schema_version": _mapping(v118_report or {}).get("schema_version"),
        "classification_primary": classification.get("primary"),
        "best_valid_policy": repair.get("best_valid_policy"),
        "best_clearing_policy": repair.get("best_clearing_policy"),
        "policy_name": REPAIRED_POLICY_NAME,
        "policy_repaired_action_counts": policy.get("repaired_action_counts"),
        "policy_dominant_action": policy.get("dominant_action"),
        "policy_dominant_action_share": policy.get("dominant_action_share"),
        "policy_changed_branch_count": policy.get("changed_branch_count"),
        "claim_causality": recommendation.get("claim_causality"),
        "v113_readiness_rerun_allowed": recommendation.get(
            "v113_readiness_rerun_allowed"
        ),
        "downstream_shadow_scorer_allowed": recommendation.get(
            "downstream_shadow_scorer_allowed"
        ),
    }


def _repaired_manifest_rows(
    archive_rows: Sequence[Mapping[str, object]],
    *,
    source_verification: Mapping[str, object],
    objective_input_integrity: Mapping[str, object],
    missing_evidence: Sequence[str],
    strict_source_verification: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    blocked_reasons: list[str] = []
    if missing_evidence:
        blocked_reasons.append("missing_evidence")
    if objective_input_integrity.get("passed") is not True:
        blocked_reasons.append("objective_input_field_integrity")
    if (
        strict_source_verification
        and source_verification.get("verification_passed") is not True
    ):
        blocked_reasons.append("source_archive_verification_failed")
    if blocked_reasons:
        return [], {
            "analysis_blocked": True,
            "blocked_reasons": sorted(set(blocked_reasons)),
            "policy_name": REPAIRED_POLICY_NAME,
        }

    contexts, integrity_failures, integrity_examples = _branch_repair_contexts(
        archive_rows
    )
    if integrity_failures:
        return [], {
            "analysis_blocked": True,
            "blocked_reasons": integrity_failures,
            "branch_integrity_examples": integrity_examples,
            "policy_name": REPAIRED_POLICY_NAME,
        }
    selections = _action_balance_selections(contexts)
    computed_policy = _summarize_policy_report(
        REPAIRED_POLICY_NAME,
        contexts,
        selections,
    )
    manifest_rows: list[dict[str, object]] = []
    for context in sorted(contexts, key=lambda item: item.branch_id):
        selected = selections.get(context.branch_id)
        if selected is None:
            continue
        manifest_rows.append(_manifest_row(context, selected))
    return manifest_rows, {
        "analysis_blocked": False,
        "policy_name": REPAIRED_POLICY_NAME,
        "computed_policy_report": computed_policy,
    }


def _manifest_row(
    context: object,
    selected: Mapping[str, object],
) -> dict[str, object]:
    current = context.current
    current_action = str(current.get("candidate_action"))
    repaired_action = str(selected.get("candidate_action"))
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION
        ),
        "branch_id": context.branch_id,
        "current_oracle_action": current_action,
        "repaired_action": repaired_action,
        "current_archive_row_id": current.get("archive_row_id"),
        "repaired_archive_row_id": selected.get("archive_row_id"),
        "changed": current_action != repaired_action,
        "unique_objective_best": bool(context.unique_objective_best),
        "serialized_objective_key": list(context.rank1_key),
        "legal_tied_candidate_actions": sorted(
            str(row.get("candidate_action")) for row in context.legal_equivalent
        ),
        "trainable_public_input": dict(_mapping(selected.get("trainable_public_input"))),
        "selected_observation_digest": selected.get("observation_digest"),
        "selected_resolution_legal": selected.get("resolution_legal") is True,
        "objective_equivalence_verified": (
            _serialized_objective_key(selected) == context.rank1_key
        ),
        "non_trainable_audit_metadata": {
            "non_trainable": True,
            "purpose": "audit_only_not_trainable",
            "policy_name": REPAIRED_POLICY_NAME,
            "source_kind": selected.get("source_kind"),
            "source_path": selected.get("source_path"),
            "seed": selected.get("seed"),
            "tick": selected.get("tick"),
            "agent_id": selected.get("agent_id"),
            "record_index": selected.get("record_index"),
            "provenance": dict(_mapping(selected.get("provenance"))),
        },
    }


def _contract_checks(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    computed_policy_report: Mapping[str, object],
    v118_policy_report: Mapping[str, object],
    strict_source_verification: bool,
) -> dict[str, object]:
    branch_ids = [str(row.get("branch_id")) for row in manifest_rows]
    repaired_counts = Counter(str(row.get("repaired_action")) for row in manifest_rows)
    v118_counts = {
        str(key): _int(value)
        for key, value in _mapping(
            v118_policy_report.get("repaired_action_counts")
        ).items()
    }
    computed_counts = {
        str(key): _int(value)
        for key, value in _mapping(
            computed_policy_report.get("repaired_action_counts")
        ).items()
    }
    violation_counts = Counter()
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    if len(branch_ids) != len(set(branch_ids)):
        violation_counts["duplicate_branch_id"] += len(branch_ids) - len(set(branch_ids))
    if len(manifest_rows) != len(set(branch_ids)):
        violation_counts["manifest_row_count_mismatch"] += 1
    if strict_source_verification and len(manifest_rows) != EXPECTED_V115_BRANCH_COUNT:
        violation_counts["expected_v115_branch_count_mismatch"] += 1
    if dict(repaired_counts) != v118_counts:
        violation_counts["v118_repaired_action_count_mismatch"] += 1
    if dict(repaired_counts) != computed_counts:
        violation_counts["computed_repaired_action_count_mismatch"] += 1
    for index, row in enumerate(manifest_rows):
        _check_manifest_row(row, index, violation_counts, examples)
    for field in (
        "unique_best_branches_changed_count",
        "resolution_invalid_selected_count",
        "objective_equivalence_violation_count",
        "unselected_branch_count",
    ):
        count = _int(computed_policy_report.get(field))
        if count:
            violation_counts[field] += count
    total = sum(violation_counts.values())
    integrity_failures = [
        field for field, count in sorted(violation_counts.items()) if count
    ]
    return {
        "branch_count": len(set(branch_ids)),
        "expected_branch_count": (
            EXPECTED_V115_BRANCH_COUNT if strict_source_verification else None
        ),
        "manifest_row_count": len(manifest_rows),
        "manifest_row_count_matches_branch_count": (
            len(manifest_rows) == len(set(branch_ids))
        ),
        "repaired_action_counts": _counter_to_dict(repaired_counts),
        "v118_repaired_action_counts": dict(sorted(v118_counts.items())),
        "computed_repaired_action_counts": dict(sorted(computed_counts.items())),
        "repaired_action_counts_match_v118": dict(repaired_counts) == v118_counts,
        "repaired_action_counts_match_computed": (
            dict(repaired_counts) == computed_counts
        ),
        "unique_best_changed_count": _int(
            computed_policy_report.get("unique_best_branches_changed_count")
        ),
        "resolution_invalid_selected_count": _int(
            computed_policy_report.get("resolution_invalid_selected_count")
        ),
        "objective_equivalence_violation_count": _int(
            computed_policy_report.get("objective_equivalence_violation_count")
        ),
        "unselected_branch_count": _int(
            computed_policy_report.get("unselected_branch_count")
        ),
        "trainable_public_input_missing_count": violation_counts[
            "trainable_public_input_missing"
        ],
        "forbidden_trainable_key_count": violation_counts[
            "forbidden_trainable_key"
        ],
        "audit_metadata_separation_violation_count": violation_counts[
            "audit_metadata_separation"
        ],
        "non_trainable_audit_metadata_missing_count": violation_counts[
            "non_trainable_audit_metadata_missing"
        ],
        "violation_counts": dict(sorted(violation_counts.items())),
        "total_violation_count": total,
        "passed": total == 0,
        "integrity_failures": integrity_failures,
        "examples": {key: value for key, value in sorted(examples.items())},
    }


def _check_manifest_row(
    row: Mapping[str, object],
    index: int,
    violation_counts: Counter[str],
    examples: defaultdict[str, list[dict[str, object]]],
) -> None:
    if row.get("unique_objective_best") is True and row.get("changed") is True:
        _add_violation(
            "unique_best_branch_changed",
            row,
            index,
            violation_counts,
            examples,
        )
    if row.get("selected_resolution_legal") is not True:
        _add_violation(
            "resolution_invalid_selected",
            row,
            index,
            violation_counts,
            examples,
        )
    if row.get("objective_equivalence_verified") is not True:
        _add_violation(
            "objective_equivalence_violation",
            row,
            index,
            violation_counts,
            examples,
        )
    trainable = row.get("trainable_public_input")
    if not isinstance(trainable, Mapping) or not trainable:
        _add_violation(
            "trainable_public_input_missing",
            row,
            index,
            violation_counts,
            examples,
        )
    else:
        forbidden_paths = _forbidden_trainable_paths(trainable)
        for path in forbidden_paths:
            _add_violation(
                "forbidden_trainable_key",
                row,
                index,
                violation_counts,
                examples,
                detail={"path": path},
            )
        audit_paths = _audit_metadata_trainable_paths(trainable)
        for path in audit_paths:
            _add_violation(
                "audit_metadata_separation",
                row,
                index,
                violation_counts,
                examples,
                detail={"path": path},
            )
    metadata = row.get("non_trainable_audit_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("non_trainable") is not True:
        _add_violation(
            "non_trainable_audit_metadata_missing",
            row,
            index,
            violation_counts,
            examples,
        )


def _add_violation(
    name: str,
    row: Mapping[str, object],
    index: int,
    violation_counts: Counter[str],
    examples: defaultdict[str, list[dict[str, object]]],
    detail: Mapping[str, object] | None = None,
) -> None:
    violation_counts[name] += 1
    if len(examples[name]) >= MAX_EXAMPLES:
        return
    payload = {
        "row_index": index,
        "branch_id": row.get("branch_id"),
        "current_oracle_action": row.get("current_oracle_action"),
        "repaired_action": row.get("repaired_action"),
    }
    if detail:
        payload.update(detail)
    examples[name].append(payload)


def _forbidden_trainable_paths(
    value: object,
    *,
    prefix: str = "trainable_public_input",
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_string = str(key)
            path = f"{prefix}.{key_string}"
            if key_string in FORBIDDEN_TRAINABLE_KEYS:
                paths.append(path)
            paths.extend(_forbidden_trainable_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_forbidden_trainable_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def _audit_metadata_trainable_paths(
    value: object,
    *,
    prefix: str = "trainable_public_input",
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_string = str(key)
            path = f"{prefix}.{key_string}"
            if key_string in AUDIT_METADATA_TRAINABLE_KEYS:
                paths.append(path)
            paths.extend(_audit_metadata_trainable_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_audit_metadata_trainable_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def _split_support_diagnostics(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    split_payloads = {
        "train": _empty_split_payload(),
        "validation": _empty_split_payload(),
        "test": _empty_split_payload(),
    }
    all_actions = sorted({str(row.get("repaired_action")) for row in manifest_rows})
    overall_counts = Counter(str(row.get("repaired_action")) for row in manifest_rows)
    for row in manifest_rows:
        split = _branch_split(str(row.get("branch_id")))
        payload = split_payloads[split]
        payload["branch_count"] += 1
        payload["repaired_action_counts"][str(row.get("repaired_action"))] += 1
        payload["current_oracle_action_counts"][
            str(row.get("current_oracle_action"))
        ] += 1
        if row.get("changed") is True:
            payload["changed_branch_count"] += 1
            payload["changed_current_action_counts"][
                str(row.get("current_oracle_action"))
            ] += 1
            payload["changed_repaired_action_counts"][
                str(row.get("repaired_action"))
            ] += 1
    split_reports: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    for split, payload in split_payloads.items():
        counts = payload["repaired_action_counts"]
        missing = [action for action in all_actions if counts[action] <= 0]
        minimum = min((counts[action] for action in all_actions), default=0)
        too_small = payload["branch_count"] < MIN_SPLIT_ROW_COUNT
        adequate = not too_small and not missing and minimum >= MIN_SPLIT_ACTION_SUPPORT
        if too_small:
            warnings.append(f"{split}_split_too_small")
        if missing:
            warnings.append(f"{split}_split_missing_action_classes")
        split_reports[split] = {
            "branch_count": payload["branch_count"],
            "repaired_action_counts": _counter_to_dict(counts),
            "current_oracle_action_counts": _counter_to_dict(
                payload["current_oracle_action_counts"]
            ),
            "changed_branch_count": payload["changed_branch_count"],
            "changed_current_action_counts": _counter_to_dict(
                payload["changed_current_action_counts"]
            ),
            "changed_repaired_action_counts": _counter_to_dict(
                payload["changed_repaired_action_counts"]
            ),
            "missing_action_classes": missing,
            "minimum_per_action_support": minimum,
            "too_small_for_shadow_scorer_evaluation": too_small,
            "support_adequate": adequate,
        }
    overall_min = min((overall_counts[action] for action in all_actions), default=0)
    if overall_min < MIN_SPLIT_ACTION_SUPPORT:
        warnings.append("overall_action_support_too_small")
    split_support_adequate = (
        bool(manifest_rows)
        and not warnings
        and all(report["support_adequate"] for report in split_reports.values())
    )
    return {
        "split_policy": {
            "method": "sha256_branch_id_mod_100",
            "train": "bucket < 70",
            "validation": "70 <= bucket < 85",
            "test": "bucket >= 85",
            "minimum_split_row_count": MIN_SPLIT_ROW_COUNT,
            "minimum_per_action_support": MIN_SPLIT_ACTION_SUPPORT,
        },
        "all_repaired_action_classes": all_actions,
        "overall_repaired_action_counts": _counter_to_dict(overall_counts),
        "overall_minimum_per_action_support": overall_min,
        "splits": split_reports,
        "warnings": sorted(set(warnings)),
        "support_adequate": split_support_adequate,
    }


def _empty_split_payload() -> dict[str, object]:
    return {
        "branch_count": 0,
        "repaired_action_counts": Counter(),
        "current_oracle_action_counts": Counter(),
        "changed_branch_count": 0,
        "changed_current_action_counts": Counter(),
        "changed_repaired_action_counts": Counter(),
    }


def _branch_split(branch_id: str) -> str:
    bucket = int(hashlib.sha256(branch_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _classification(
    *,
    missing_evidence: Sequence[str],
    source_verification: Mapping[str, object],
    objective_input_integrity: Mapping[str, object],
    contract_checks: Mapping[str, object],
    split_support: Mapping[str, object],
    strict_source_verification: bool,
) -> dict[str, object]:
    labels = ["diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
    if missing_evidence:
        primary = "repaired_label_contract_failed"
        labels[:0] = ["missing_evidence_inconclusive", primary]
    elif objective_input_integrity.get("passed") is not True:
        primary = "repaired_label_contract_failed"
        labels[:0] = ["objective_input_field_integrity", primary]
    elif (
        strict_source_verification
        and source_verification.get("verification_passed") is not True
    ):
        primary = "repaired_label_contract_failed"
        labels[:0] = ["source_archive_verification_failed", primary]
    elif contract_checks.get("passed") is not True:
        primary = "repaired_label_contract_failed"
        labels.insert(0, primary)
    elif split_support.get("support_adequate") is True:
        primary = "repaired_label_contract_ready_for_shadow_scorer_proposal"
        labels.insert(0, primary)
    else:
        primary = "repaired_label_contract_support_limited"
        labels.insert(0, primary)
    return {
        "primary": primary,
        "labels": _dedupe_allowed(labels),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(
    *,
    classification: Mapping[str, object],
) -> dict[str, object]:
    primary = classification.get("primary")
    return {
        "next_step": (
            "contract_is_clean_but_requires_separate_shadow_scorer_proposal"
            if primary == "repaired_label_contract_ready_for_shadow_scorer_proposal"
            else "keep_repaired_labels_diagnostics_only_until_contract_or_support_blockers_clear"
        ),
        "summary": (
            "The repaired-label contract is clean enough to inform a separate "
            "shadow-scorer proposal, but v119 does not authorize shadow scoring "
            "or readiness."
            if primary == "repaired_label_contract_ready_for_shadow_scorer_proposal"
            else "The repaired-label contract is diagnostics-only and does not "
            "authorize shadow scoring or readiness."
        ),
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "viewer_change_recommended": False,
        "claim_causality": False,
    }


def _missing_evidence(
    *,
    source_reports: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> list[str]:
    missing: list[str] = []
    report_evidence = _mapping(source_reports.get("archive_report"))
    rows_evidence = _mapping(source_reports.get("archive_rows"))
    v118_evidence = _mapping(source_reports.get("v118_report"))
    if report_evidence.get("loaded") is not True:
        missing.append("archive_report")
    elif report_evidence.get("schema_matches") is not True:
        missing.append("archive_report_schema")
    if rows_evidence.get("loaded") is not True or not rows:
        missing.append("archive_rows")
    if v118_evidence.get("loaded") is not True:
        missing.append("v118_report")
    elif v118_evidence.get("schema_matches") is not True:
        missing.append("v118_report_schema")
    return sorted(set(missing))


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
