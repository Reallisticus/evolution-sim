from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _int,
    _mapping,
    _resolve_json_report,
    _share,
)
from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V119_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V119_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
    _branch_split as _v119_branch_split,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION = (
    "mind_v3_first_recovery_repaired_label_split_support_feasibility_v1"
)
MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_POLICY = (
    "diagnostics_only_first_recovery_v120_repaired_label_split_support_feasibility_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v120-first-recovery-repaired-label-split-support-feasibility.json"
)

EXPECTED_REPAIRED_ACTION_COUNTS: dict[str, int] = {
    "attack_east": 3,
    "attack_west": 3,
    "drink": 7,
    "eat": 17,
    "move_east": 16,
    "move_north": 15,
    "move_south": 15,
    "move_west": 14,
    "stay": 16,
}
EXPECTED_MANIFEST_ROW_COUNT = 106
STRICT_TRAIN_MIN = 2
STRICT_VALIDATION_MIN = 1
STRICT_TEST_MIN = 1
SPLIT_ASSIGNMENT_TRAINABLE_KEYS = frozenset(
    {"split", "fold", "branch_split", "split_assignment"}
)
FORBIDDEN_TRAINABLE_METADATA_KEYS = frozenset(
    {
        "agent_id",
        "audit",
        "audit_metadata",
        "branch_id",
        "digest",
        "fixture",
        "fixture_name",
        "non_trainable_audit_metadata",
        "path",
        "private",
        "private_world_state",
        "provenance",
        "record_index",
        "report",
        "report_path",
        "seed",
        "source",
        "source_kind",
        "source_path",
    }
)
FORBIDDEN_TRAINABLE_METADATA_TOKENS = frozenset(
    {
        "agent",
        "audit",
        "branch",
        "digest",
        "fixture",
        "logged",
        "metadata",
        "path",
        "private",
        "provenance",
        "record",
        "report",
        "seed",
        "source",
        "world",
    }
)

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "diagnostics_only_no_runtime_promotion",
    "missing_evidence_inconclusive",
    "repaired_label_source_integrity_failed",
    "split_support_feasibility_limited_by_rare_actions",
    "deterministic_stratified_split_feasible_but_shadow_still_blocked",
    "readiness_rerun_blocked",
)


@dataclass(frozen=True, slots=True)
class FirstRecoveryRepairedLabelSplitSupportFeasibilityBuild:
    report: dict[str, object]


def build_first_recovery_repaired_label_split_support_feasibility(
    *,
    v119_report: Mapping[str, object] | None = None,
    v119_report_path: str | Path | None = DEFAULT_V119_REPORT_PATH,
    manifest_rows: Sequence[Mapping[str, object]] | None = None,
    manifest_path: str | Path | None = DEFAULT_V119_MANIFEST_PATH,
) -> FirstRecoveryRepairedLabelSplitSupportFeasibilityBuild:
    report_payload, report_evidence = _resolve_json_report(
        v119_report,
        v119_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
    )
    rows, manifest_evidence = _resolve_manifest_rows(manifest_rows, manifest_path)
    source_integrity = _source_integrity(
        v119_report=report_payload,
        manifest_rows=rows,
        report_evidence=report_evidence,
        manifest_evidence=manifest_evidence,
    )
    support = _total_support(rows)
    split_policies = _split_policy_evaluations(rows)
    kfold = _stratified_kfold_feasibility(rows)
    scarcity = _scarcity_analysis(support)
    classification = _classification(
        source_integrity=source_integrity,
        split_policies=split_policies,
        scarcity=scarcity,
    )
    recommendation = _recommendation(
        classification=classification,
        scarcity=scarcity,
        split_policies=split_policies,
    )
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION
        ),
        "audit_policy": (
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_POLICY
        ),
        "contract": _contract(),
        "source_reports": {
            "v119_report": report_evidence,
            "v119_manifest": manifest_evidence,
        },
        "source_integrity": source_integrity,
        "total_repaired_label_support": support,
        "split_policy_evaluations": split_policies,
        "stratified_kfold_feasibility": kfold,
        "scarcity_analysis": scarcity,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryRepairedLabelSplitSupportFeasibilityBuild(report=report)


def write_first_recovery_repaired_label_split_support_feasibility_report(
    build: FirstRecoveryRepairedLabelSplitSupportFeasibilityBuild,
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
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION
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
        "claim_causality": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION
                ),
                "audit_policy": (
                    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_POLICY
                ),
            }
        ),
    }


def _resolve_manifest_rows(
    manifest_rows: Sequence[Mapping[str, object]] | None,
    manifest_path: str | Path | None,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    if manifest_rows is not None:
        rows = tuple(dict(row) for row in manifest_rows)
        return rows, {
            "loaded": True,
            "in_memory": True,
            "row_count": len(rows),
            "path": None,
        }
    if manifest_path is None:
        return (), {"loaded": False, "path": None, "row_count": 0}
    path = Path(manifest_path)
    if not path.exists():
        return (), {"loaded": False, "path": str(path), "row_count": 0}
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"manifest row {line_number} must be a JSON object")
            rows.append(payload)
    return tuple(rows), {
        "loaded": True,
        "in_memory": False,
        "path": str(path),
        "row_count": len(rows),
    }


def _source_integrity(
    *,
    v119_report: Mapping[str, object] | None,
    manifest_rows: Sequence[Mapping[str, object]],
    report_evidence: Mapping[str, object],
    manifest_evidence: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    report = _mapping(v119_report or {})
    classification = _mapping(report.get("classification"))
    recommendation = _mapping(report.get("recommendation"))
    contract_checks = _mapping(report.get("contract_checks"))
    manifest = _mapping(report.get("manifest"))
    counts = _action_counts(manifest_rows)
    reported_digest = manifest.get("manifest_digest")
    computed_digest = stable_payload_digest(list(manifest_rows))
    branch_id_report = _branch_id_integrity(manifest_rows)
    branch_ids = list(branch_id_report["valid_branch_ids"])
    contract_counts_payload = contract_checks.get("repaired_action_counts")
    report_counts = {
        str(key): _int(value)
        for key, value in _mapping(contract_counts_payload).items()
    }
    leakage = _trainable_leakage(manifest_rows)
    if report_evidence.get("loaded") is not True:
        failures.append("missing_v119_report")
    elif report_evidence.get("schema_matches") is not True:
        failures.append("v119_report_schema_mismatch")
    if manifest_evidence.get("loaded") is not True:
        failures.append("missing_v119_manifest")
    if "manifest_digest" not in manifest:
        failures.append("manifest_digest_missing")
    elif not _valid_sha256_digest(reported_digest):
        failures.append("manifest_digest_malformed")
    elif reported_digest != computed_digest:
        failures.append("manifest_digest_mismatch")
    if classification.get("primary") != "repaired_label_contract_support_limited":
        failures.append("v119_classification_not_support_limited")
    if len(manifest_rows) != EXPECTED_MANIFEST_ROW_COUNT:
        failures.append("manifest_row_count_not_106")
    if branch_id_report["invalid_branch_id_count"]:
        failures.append("manifest_branch_id_invalid")
    if len(set(branch_ids)) != len(branch_ids):
        failures.append("manifest_branch_ids_not_unique")
    if dict(counts) != EXPECTED_REPAIRED_ACTION_COUNTS:
        failures.append("manifest_repaired_action_counts_unexpected")
    if "repaired_action_counts" not in contract_checks:
        failures.append("contract_repaired_action_counts_missing")
    elif dict(counts) != report_counts:
        failures.append("manifest_counts_do_not_match_v119_report")
    if "manifest_row_count" not in manifest or not _int_field_equals(
        manifest, "manifest_row_count", len(manifest_rows)
    ):
        failures.append("v119_manifest_row_count_mismatch")
    if "manifest_row_count" not in contract_checks or not _int_field_equals(
        contract_checks, "manifest_row_count", len(manifest_rows)
    ):
        failures.append("contract_manifest_row_count_mismatch")
    if "branch_count" not in contract_checks or not _int_field_equals(
        contract_checks, "branch_count", len(set(branch_ids))
    ):
        failures.append("contract_branch_count_mismatch")
    if "passed" not in contract_checks:
        failures.append("contract_checks_passed_missing")
    elif contract_checks.get("passed") is not True:
        failures.append("contract_checks_passed_not_true")
    if "total_violation_count" not in contract_checks:
        failures.append("contract_total_violation_count_missing")
    elif _int(contract_checks.get("total_violation_count")) != 0:
        failures.append("v119_contract_violations_nonzero")
    if "integrity_failures" not in contract_checks:
        failures.append("contract_integrity_failures_missing")
    elif contract_checks.get("integrity_failures") != []:
        failures.append("contract_integrity_failures_not_empty_or_malformed")
    for field in ("failures", "source_integrity_failures"):
        if field in contract_checks and contract_checks.get(field) != []:
            failures.append(f"contract_{field}_not_empty_or_malformed")
    if leakage["split_key_leak_count"]:
        failures.append("trainable_split_assignment_leakage")
    if leakage["forbidden_metadata_key_count"]:
        failures.append("trainable_metadata_leakage")
    if recommendation.get("v113_readiness_rerun_allowed") is not False:
        failures.append("v113_readiness_not_blocked")
    if recommendation.get("downstream_shadow_scorer_allowed") is not False:
        failures.append("shadow_scorer_not_blocked")
    if recommendation.get("claim_causality") is not False:
        failures.append("claim_causality_not_false")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v119_classification_primary": classification.get("primary"),
        "reported_manifest_digest": reported_digest,
        "computed_manifest_digest": computed_digest,
        "manifest_digest_matches": (
            _valid_sha256_digest(reported_digest) and reported_digest == computed_digest
        ),
        "manifest_row_count": len(manifest_rows),
        "unique_branch_id_count": len(set(branch_ids)),
        "invalid_branch_id_count": branch_id_report["invalid_branch_id_count"],
        "manifest_branch_ids_unique": (
            len(set(branch_ids)) == len(branch_ids)
            and branch_id_report["invalid_branch_id_count"] == 0
        ),
        "branch_id_examples": branch_id_report["examples"],
        "manifest_repaired_action_counts": _counter_to_dict(counts),
        "v119_report_repaired_action_counts": dict(sorted(report_counts.items())),
        "contract_total_violation_count": _int(
            contract_checks.get("total_violation_count")
        ),
        "contract_checks_passed": contract_checks.get("passed"),
        "contract_integrity_failures": contract_checks.get("integrity_failures"),
        "v113_readiness_rerun_allowed": recommendation.get(
            "v113_readiness_rerun_allowed"
        ),
        "downstream_shadow_scorer_allowed": recommendation.get(
            "downstream_shadow_scorer_allowed"
        ),
        "claim_causality": recommendation.get("claim_causality"),
        "trainable_leakage": leakage,
    }


def _total_support(manifest_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts = _action_counts(manifest_rows)
    scarce_for_three_way = {
        action: max(0, 3 - count)
        for action, count in sorted(counts.items())
        if count < 3
    }
    scarce_for_strict = {
        action: max(0, _strict_required_total() - count)
        for action, count in sorted(counts.items())
        if count < _strict_required_total()
    }
    return {
        "repaired_action_counts": _counter_to_dict(counts),
        "total_branch_count": sum(counts.values()),
        "minimum_action_support": min(counts.values()) if counts else 0,
        "hard_scarcity": {
            "all_splits_min_1_required_total_per_action": 3,
            "all_splits_min_1_feasible_by_total_support": not scarce_for_three_way,
            "train2_validation1_test1_required_total_per_action": (
                _strict_required_total()
            ),
            "train2_validation1_test1_feasible_by_total_support": (
                not scarce_for_strict
            ),
            "additional_needed_for_all_splits_min_1": scarce_for_three_way,
            "additional_needed_for_train2_validation1_test1": scarce_for_strict,
        },
    }


def _int_field_equals(
    payload: Mapping[str, object],
    field: str,
    expected: int,
) -> bool:
    value = payload.get(field)
    return type(value) is int and value == expected


def _valid_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _branch_id_integrity(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    valid: list[str] = []
    examples: list[dict[str, object]] = []
    invalid_count = 0
    for index, row in enumerate(manifest_rows):
        branch_id = row.get("branch_id")
        if isinstance(branch_id, str) and branch_id:
            valid.append(branch_id)
            continue
        invalid_count += 1
        if len(examples) < 16:
            examples.append(
                {
                    "row_index": index,
                    "branch_id_repr": repr(branch_id),
                    "repaired_action": row.get("repaired_action"),
                }
            )
    return {
        "valid_branch_ids": valid,
        "invalid_branch_id_count": invalid_count,
        "examples": examples,
    }


def _branch_id(row: Mapping[str, object]) -> str | None:
    value = row.get("branch_id")
    return value if isinstance(value, str) and value else None


def _trainable_leakage(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    split_examples: list[dict[str, object]] = []
    metadata_examples: list[dict[str, object]] = []
    split_count = 0
    metadata_count = 0
    for index, row in enumerate(manifest_rows):
        trainable = row.get("trainable_public_input")
        if not isinstance(trainable, Mapping):
            continue
        for path in _forbidden_trainable_paths(
            trainable,
            forbidden_keys=SPLIT_ASSIGNMENT_TRAINABLE_KEYS,
        ):
            split_count += 1
            if len(split_examples) < 16:
                split_examples.append(_leak_example(row, index, path))
        for path in _forbidden_trainable_paths(
            trainable,
            forbidden_keys=FORBIDDEN_TRAINABLE_METADATA_KEYS,
            forbidden_token_families=FORBIDDEN_TRAINABLE_METADATA_TOKENS,
            include_path_digest_suffixes=True,
        ):
            metadata_count += 1
            if len(metadata_examples) < 16:
                metadata_examples.append(_leak_example(row, index, path))
    return {
        "split_key_leak_count": split_count,
        "forbidden_metadata_key_count": metadata_count,
        "split_key_examples": split_examples,
        "forbidden_metadata_key_examples": metadata_examples,
        "split_forbidden_keys": sorted(SPLIT_ASSIGNMENT_TRAINABLE_KEYS),
        "metadata_forbidden_keys": sorted(FORBIDDEN_TRAINABLE_METADATA_KEYS),
        "metadata_forbidden_tokens": sorted(FORBIDDEN_TRAINABLE_METADATA_TOKENS),
    }


def _forbidden_trainable_paths(
    value: object,
    *,
    forbidden_keys: frozenset[str],
    forbidden_token_families: frozenset[str] = frozenset(),
    prefix: str = "trainable_public_input",
    include_path_digest_suffixes: bool = False,
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_string = str(key)
            path = f"{prefix}.{key_string}"
            if (
                key_string in forbidden_keys
                or (
                    forbidden_token_families
                    and bool(
                        set(_key_tokens(key_string)) & set(forbidden_token_families)
                    )
                )
                or (
                    include_path_digest_suffixes
                    and (key_string.endswith("_path") or key_string.endswith("_digest"))
                )
            ):
                paths.append(path)
            paths.extend(
                _forbidden_trainable_paths(
                    item,
                    forbidden_keys=forbidden_keys,
                    forbidden_token_families=forbidden_token_families,
                    prefix=path,
                    include_path_digest_suffixes=include_path_digest_suffixes,
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(
                _forbidden_trainable_paths(
                    item,
                    forbidden_keys=forbidden_keys,
                    forbidden_token_families=forbidden_token_families,
                    prefix=f"{prefix}[{index}]",
                    include_path_digest_suffixes=include_path_digest_suffixes,
                )
            )
    return paths


def _key_tokens(key: str) -> tuple[str, ...]:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return tuple(
        token
        for token in re.split(r"[^A-Za-z0-9]+", camel_split.lower())
        if token
    )


def _leak_example(
    row: Mapping[str, object],
    index: int,
    path: str,
) -> dict[str, object]:
    return {
        "row_index": index,
        "branch_id": row.get("branch_id"),
        "repaired_action": row.get("repaired_action"),
        "path": path,
    }


def _split_policy_evaluations(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "v119_sha256_bucket": _evaluate_split_assignment(
            manifest_rows,
            assignments={
                branch_id: _v119_branch_split(branch_id)
                for row in manifest_rows
                for branch_id in [_branch_id(row)]
                if branch_id is not None
            },
            split_names=("train", "validation", "test"),
            policy_metadata={
                "method": "sha256_branch_id_mod_100",
                "split_assignment_inputs": ["branch_id"],
                "uses_only_allowed_audit_metadata": True,
                "excluded_from_trainable_public_input": _split_excluded_from_trainable(
                    manifest_rows
                ),
                "train": "bucket < 70",
                "validation": "70 <= bucket < 85",
                "test": "bucket >= 85",
            },
        ),
        "stratified_by_repaired_action_min_one_each": _evaluate_split_assignment(
            manifest_rows,
            assignments=_stratified_three_way_assignments(manifest_rows),
            split_names=("train", "validation", "test"),
            policy_metadata={
                "method": "deterministic_stratified_by_repaired_action",
                "split_assignment_inputs": ["branch_id", "repaired_action"],
                "uses_only_allowed_audit_metadata": True,
                "excluded_from_trainable_public_input": _split_excluded_from_trainable(
                    manifest_rows
                ),
                "assignment_rule": (
                    "per_action_hash_order_assign_first_to_test_second_to_validation_rest_to_train"
                ),
            },
        ),
    }


def _stratified_kfold_feasibility(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts = _action_counts(manifest_rows)
    max_complete = min(counts.values()) if counts else 0
    reports: dict[str, object] = {}
    for k in (3, 4):
        reports[f"k_{k}"] = _evaluate_split_assignment(
            manifest_rows,
            assignments=_stratified_kfold_assignments(manifest_rows, k=k),
            split_names=tuple(f"fold_{index}" for index in range(k)),
            policy_metadata={
                "method": "deterministic_stratified_kfold_by_repaired_action",
                "k": k,
                "split_assignment_inputs": ["branch_id", "repaired_action"],
                "uses_only_allowed_audit_metadata": True,
                "excluded_from_trainable_public_input": _split_excluded_from_trainable(
                    manifest_rows
                ),
            },
        )
    return {
        "maximum_complete_action_class_k": max_complete,
        "kfold_reports": reports,
        "k3_every_fold_has_every_action_class": _mapping(reports["k_3"]).get(
            "all_splits_contain_every_action_class"
        ),
        "k4_every_fold_has_every_action_class": _mapping(reports["k_4"]).get(
            "all_splits_contain_every_action_class"
        ),
    }


def _scarcity_analysis(support: Mapping[str, object]) -> dict[str, object]:
    hard = _mapping(support.get("hard_scarcity"))
    strict_needed = {
        str(action): _int(count)
        for action, count in _mapping(
            hard.get("additional_needed_for_train2_validation1_test1")
        ).items()
    }
    return {
        "all_splits_can_contain_every_action_class_by_total_support": hard.get(
            "all_splits_min_1_feasible_by_total_support"
        ),
        "train2_validation1_test1_feasible_by_total_support": hard.get(
            "train2_validation1_test1_feasible_by_total_support"
        ),
        "rare_action_additional_needed_for_train2_validation1_test1": strict_needed,
        "minimum_additional_repaired_branches_for_stricter_target": sum(
            strict_needed.values()
        ),
        "next_diagnostics_only_data_coverage_slice": (
            "target_additional_first_recovery_archive_coverage_for_rare_repaired_actions"
            if strict_needed
            else "separate_shadow_scorer_proposal_required_before_any_training_or_readiness"
        ),
    }


def _evaluate_split_assignment(
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    assignments: Mapping[str, str],
    split_names: Sequence[str],
    policy_metadata: Mapping[str, object],
) -> dict[str, object]:
    all_actions = sorted({str(row.get("repaired_action")) for row in manifest_rows})
    payloads = {
        split: {"branch_count": 0, "repaired_action_counts": Counter()}
        for split in split_names
    }
    for row in manifest_rows:
        branch_id = _branch_id(row)
        if branch_id is None:
            continue
        split = assignments.get(branch_id)
        if split not in payloads:
            continue
        payloads[split]["branch_count"] += 1
        payloads[split]["repaired_action_counts"][str(row.get("repaired_action"))] += 1
    split_reports: dict[str, dict[str, object]] = {}
    for split, payload in payloads.items():
        counts = payload["repaired_action_counts"]
        missing = [action for action in all_actions if counts[action] <= 0]
        minimum = min((counts[action] for action in all_actions), default=0)
        split_reports[split] = {
            "branch_count": payload["branch_count"],
            "repaired_action_counts": _counter_to_dict(counts),
            "missing_action_classes": missing,
            "minimum_per_action_support": minimum,
            "contains_every_action_class": not missing,
        }
    all_contain = bool(split_reports) and all(
        report["contains_every_action_class"] for report in split_reports.values()
    )
    train_report = split_reports.get("train", {})
    validation_report = split_reports.get("validation", {})
    test_report = split_reports.get("test", {})
    strict_target_met = (
        _int(train_report.get("minimum_per_action_support")) >= STRICT_TRAIN_MIN
        and _int(validation_report.get("minimum_per_action_support"))
        >= STRICT_VALIDATION_MIN
        and _int(test_report.get("minimum_per_action_support")) >= STRICT_TEST_MIN
    )
    return {
        "policy": dict(policy_metadata),
        "splits": split_reports,
        "all_splits_contain_every_action_class": all_contain,
        "minimum_per_action_support_by_split": {
            split: report["minimum_per_action_support"]
            for split, report in split_reports.items()
        },
        "train2_validation1_test1_target_met": strict_target_met,
    }


def _stratified_three_way_assignments(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for _action, rows in _rows_by_action(manifest_rows).items():
        ordered = sorted(rows, key=_stable_row_key)
        for index, row in enumerate(ordered):
            branch_id = _branch_id(row)
            if branch_id is None:
                continue
            if len(ordered) == 1:
                split = "train"
            elif len(ordered) == 2:
                split = "validation" if index == 0 else "train"
            else:
                split = "test" if index == 0 else "validation" if index == 1 else "train"
            assignments[branch_id] = split
    return assignments


def _stratified_kfold_assignments(
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for _action, rows in _rows_by_action(manifest_rows).items():
        ordered = sorted(rows, key=_stable_row_key)
        for index, row in enumerate(ordered):
            branch_id = _branch_id(row)
            if branch_id is not None:
                assignments[branch_id] = f"fold_{index % k}"
    return assignments


def _rows_by_action(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        if _branch_id(row) is None:
            continue
        grouped[str(row.get("repaired_action"))].append(row)
    return dict(sorted(grouped.items()))


def _stable_row_key(row: Mapping[str, object]) -> tuple[str, str]:
    branch_id = _branch_id(row) or ""
    digest = hashlib.sha256(branch_id.encode("utf-8")).hexdigest()
    return (digest, branch_id)


def _split_excluded_from_trainable(
    manifest_rows: Sequence[Mapping[str, object]],
) -> bool:
    return not any(
        _trainable_contains_split_key(_mapping(row.get("trainable_public_input")))
        for row in manifest_rows
    )


def _trainable_contains_split_key(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in {"split", "fold", "branch_split", "split_assignment"}:
                return True
            if _trainable_contains_split_key(item):
                return True
    elif isinstance(value, list):
        return any(_trainable_contains_split_key(item) for item in value)
    return False


def _action_counts(manifest_rows: Sequence[Mapping[str, object]]) -> Counter[str]:
    return Counter(str(row.get("repaired_action")) for row in manifest_rows)


def _strict_required_total() -> int:
    return STRICT_TRAIN_MIN + STRICT_VALIDATION_MIN + STRICT_TEST_MIN


def _classification(
    *,
    source_integrity: Mapping[str, object],
    split_policies: Mapping[str, object],
    scarcity: Mapping[str, object],
) -> dict[str, object]:
    labels = ["diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
    if source_integrity.get("passed") is not True:
        primary = (
            "missing_evidence_inconclusive"
            if any(
                str(item).startswith("missing_")
                for item in _list_like(source_integrity.get("failures"))
            )
            else "repaired_label_source_integrity_failed"
        )
        labels.insert(0, primary)
    elif scarcity.get("train2_validation1_test1_feasible_by_total_support") is not True:
        primary = "split_support_feasibility_limited_by_rare_actions"
        labels.insert(0, primary)
    elif _mapping(
        split_policies.get("stratified_by_repaired_action_min_one_each")
    ).get("all_splits_contain_every_action_class") is True:
        primary = "deterministic_stratified_split_feasible_but_shadow_still_blocked"
        labels.insert(0, primary)
    else:
        primary = "split_support_feasibility_limited_by_rare_actions"
        labels.insert(0, primary)
    return {
        "primary": primary,
        "labels": _dedupe_allowed(labels),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(
    *,
    classification: Mapping[str, object],
    scarcity: Mapping[str, object],
    split_policies: Mapping[str, object],
) -> dict[str, object]:
    primary = classification.get("primary")
    if primary in (
        "missing_evidence_inconclusive",
        "repaired_label_source_integrity_failed",
    ):
        return {
            "next_step": "source_integrity_must_pass_before_split_support_feasibility",
            "summary": (
                "Source integrity failed, so split support feasibility cannot be "
                "trusted from these inputs."
            ),
            "stratified_all_actions_each_split": None,
            "rare_action_additional_needed": {},
            "downstream_shadow_scorer_allowed": False,
            "v113_readiness_rerun_allowed": False,
            "runtime_policy_change_recommended": False,
            "trained_artifact_change_recommended": False,
            "gate_change_recommended": False,
            "observation_field_change_recommended": False,
            "viewer_change_recommended": False,
            "claim_causality": False,
        }
    rare_needed = _mapping(
        scarcity.get("rare_action_additional_needed_for_train2_validation1_test1")
    )
    stratified = _mapping(split_policies.get("stratified_by_repaired_action_min_one_each"))
    return {
        "next_step": (
            "run_diagnostics_only_archive_coverage_slice_for_rare_repaired_actions"
            if rare_needed
            else "separate_shadow_scorer_proposal_required_before_any_training_or_readiness"
        ),
        "summary": (
            "A deterministic stratified split can place every repaired action in "
            "all train/validation/test splits, but train>=2 plus validation/test>=1 "
            f"is impossible for rare actions without additional coverage: {dict(rare_needed)}."
            if rare_needed
            else "Deterministic stratified split support is feasible, but this "
            "diagnostic does not authorize shadow scoring or readiness."
        ),
        "stratified_all_actions_each_split": stratified.get(
            "all_splits_contain_every_action_class"
        ),
        "rare_action_additional_needed": dict(rare_needed),
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "viewer_change_recommended": False,
        "claim_causality": False,
    }


def _list_like(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else []


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
