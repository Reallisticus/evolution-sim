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
from evolution_sim.mind.first_recovery_oracle_tie_break_audit import (
    FirstRecoveryOracleTieBreakAuditError,
    _serialized_objective_key,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _int,
    _mapping,
    _resolve_archive_rows,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_rare_action_coverage_targeting import (
    RARE_ACTION_ADDITIONAL_NEEDED,
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
    _branch_repair_contexts,
)
from evolution_sim.mind.first_recovery_rare_action_coverage_targeting import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V121_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION = (
    "mind_v3_first_recovery_rare_attack_coverage_collection_v1"
)
MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_POLICY = (
    "diagnostics_only_first_recovery_v122_rare_attack_coverage_collection_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v122-first-recovery-rare-attack-coverage-collection.json"
)
DEFAULT_CANDIDATES_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v122-first-recovery-rare-attack-candidates.jsonl"
)

TARGET_ACTIONS: tuple[str, ...] = ("attack_east", "attack_west")
EXPECTED_CURRENT_RARE_SUPPORT: dict[str, int] = {
    "attack_east": 3,
    "attack_west": 3,
}
EXPECTED_V121_CLASSIFICATION = "rare_action_coverage_not_available_in_existing_archive"
EXPECTED_V120_CLASSIFICATION = "split_support_feasibility_limited_by_rare_actions"
EXPECTED_V119_CLASSIFICATION = "repaired_label_contract_support_limited"
DEFAULT_MAX_SEEDS = 128
DEFAULT_MAX_BRANCHES = 106
DEFAULT_MAX_CANDIDATES_PER_ACTION = 1
MAX_EXAMPLES = 16

NO_AUTHORIZATION_FIELDS: tuple[str, ...] = (
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
    "rare_attack_coverage_source_integrity_failed",
    "rare_attack_coverage_candidates_found",
    "rare_attack_coverage_partial",
    "rare_attack_coverage_not_found_within_budget",
    "candidate_source_contains_no_unmanifested_branches",
    "readiness_rerun_blocked",
)


@dataclass(frozen=True, slots=True)
class FirstRecoveryRareAttackCoverageCollectionBuild:
    report: dict[str, object]
    candidate_rows: tuple[dict[str, object], ...]


def build_first_recovery_rare_attack_coverage_collection(
    *,
    v119_report: Mapping[str, object] | None = None,
    v119_report_path: str | Path | None = DEFAULT_V119_REPORT_PATH,
    manifest_rows: Sequence[Mapping[str, object]] | None = None,
    manifest_path: str | Path | None = DEFAULT_V119_MANIFEST_PATH,
    v120_report: Mapping[str, object] | None = None,
    v120_report_path: str | Path | None = DEFAULT_V120_REPORT_PATH,
    v121_report: Mapping[str, object] | None = None,
    v121_report_path: str | Path | None = DEFAULT_V121_REPORT_PATH,
    candidate_archive_report: Mapping[str, object] | None = None,
    candidate_archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    candidate_archive_rows: Sequence[Mapping[str, object]] | None = None,
    candidate_archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    target_actions: Sequence[str] = TARGET_ACTIONS,
    max_seeds: int = DEFAULT_MAX_SEEDS,
    seed_allowlist: Sequence[int | str] = (),
    max_branches: int = DEFAULT_MAX_BRANCHES,
    max_candidates_per_action: int = DEFAULT_MAX_CANDIDATES_PER_ACTION,
) -> FirstRecoveryRareAttackCoverageCollectionBuild:
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
    v121_payload, v121_evidence = _resolve_json_report(
        v121_report,
        v121_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION
        ),
    )
    archive_payload, archive_evidence = _resolve_json_report(
        candidate_archive_report,
        candidate_archive_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    archive_rows, archive_rows_evidence = _resolve_archive_rows(
        candidate_archive_rows,
        candidate_archive_rows_path,
    )
    source_reports = {
        "v119_report": v119_evidence,
        "v119_manifest": manifest_evidence,
        "v120_report": v120_evidence,
        "v121_report": v121_evidence,
        "candidate_archive_report": archive_evidence,
        "candidate_archive_rows": archive_rows_evidence,
        "source_report_policy": "inputs_are_read_only_diagnostics_sources",
    }
    config = _search_config(
        target_actions=target_actions,
        max_seeds=max_seeds,
        seed_allowlist=seed_allowlist,
        max_branches=max_branches,
        max_candidates_per_action=max_candidates_per_action,
        candidate_archive_report=archive_payload,
    )
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v119_report=v119_payload,
        manifest_rows=manifest,
        v120_report=v120_payload,
        v121_report=v121_payload,
    )
    collection = _collect_candidates(
        archive_rows=archive_rows,
        manifest_rows=manifest,
        source_integrity=source_integrity,
        search_config=config,
    )
    classification = _classification(
        source_integrity=source_integrity,
        collection=collection,
    )
    recommendation = _recommendation(
        classification=classification,
        collection=collection,
    )
    candidate_rows_out = tuple(collection.get("candidate_manifest_rows", ()))
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "search_config": config,
        "search_budget_used": collection.get("search_budget_used", {}),
        "found_candidate_counts": collection.get("found_candidate_counts", {}),
        "accepted_candidate_examples": collection.get(
            "accepted_candidate_examples",
            {},
        ),
        "rejection_counts": collection.get("rejection_counts", {}),
        "rejection_examples": collection.get("rejection_examples", {}),
        "candidate_collection": {
            key: value
            for key, value in collection.items()
            if key != "candidate_manifest_rows"
        },
        "stricter_split_support_if_accepted": collection.get(
            "stricter_split_support_if_accepted",
            {},
        ),
        "classification": classification,
        "recommendation": recommendation,
        "candidate_manifest": {
            "row_count": len(candidate_rows_out),
            "schema_version": (
                MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION
            ),
            "diagnostics_only": True,
            "not_training_manifest": True,
            "trainable_public_input_contents_exposed": False,
        },
        "non_promoted": True,
    }
    return FirstRecoveryRareAttackCoverageCollectionBuild(
        report=report,
        candidate_rows=candidate_rows_out,
    )


def write_first_recovery_rare_attack_coverage_collection_outputs(
    build: FirstRecoveryRareAttackCoverageCollectionBuild,
    *,
    output_path: str | Path,
    candidates_output_path: str | Path | None = DEFAULT_CANDIDATES_OUTPUT_PATH,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(build.report)
    if candidates_output_path is not None:
        candidates_path = Path(candidates_output_path)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        with candidates_path.open("w", encoding="utf-8") as handle:
            for row in build.candidate_rows:
                json.dump(row, handle, sort_keys=True, allow_nan=False)
                handle.write("\n")
        report["candidate_manifest"] = {
            **_mapping(report.get("candidate_manifest")),
            "path": str(candidates_path),
        }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION
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
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION
                ),
                "audit_policy": (
                    MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_POLICY
                ),
            }
        ),
    }


def _source_integrity(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    v119_report: Mapping[str, object] | None,
    manifest_rows: Sequence[Mapping[str, object]],
    v120_report: Mapping[str, object] | None,
    v121_report: Mapping[str, object] | None,
) -> dict[str, object]:
    failures: list[str] = []
    for name in (
        "v119_report",
        "v119_manifest",
        "v120_report",
        "v121_report",
        "candidate_archive_report",
        "candidate_archive_rows",
    ):
        evidence = _mapping(source_reports.get(name))
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")

    v119 = _mapping(v119_report or {})
    v120 = _mapping(v120_report or {})
    v121 = _mapping(v121_report or {})
    v119_classification = _mapping(v119.get("classification"))
    v120_classification = _mapping(v120.get("classification"))
    v121_classification = _mapping(v121.get("classification"))
    v119_contract = _mapping(v119.get("contract_checks"))
    v119_manifest = _mapping(v119.get("manifest"))
    v120_source = _mapping(v120.get("source_integrity"))
    v120_scarcity = _mapping(v120.get("scarcity_analysis"))
    v121_source = _mapping(v121.get("source_integrity"))
    v121_support = _mapping(v121.get("current_rare_action_support"))
    v121_search = _mapping(v121.get("candidate_search"))
    manifest_digest = stable_payload_digest(list(manifest_rows))

    if v119_classification.get("primary") != EXPECTED_V119_CLASSIFICATION:
        failures.append("v119_classification_unexpected")
    if v119_contract.get("passed") is not True:
        failures.append("v119_contract_checks_not_passed")
    if v119_contract.get("total_violation_count") != 0:
        failures.append("v119_contract_violations_nonzero")
    if v120_classification.get("primary") != EXPECTED_V120_CLASSIFICATION:
        failures.append("v120_classification_unexpected")
    if v120_source.get("passed") is not True:
        failures.append("v120_source_integrity_not_passed")
    if v120_source.get("failures") != []:
        failures.append("v120_source_failures_not_empty_or_malformed")
    if v121_classification.get("primary") != EXPECTED_V121_CLASSIFICATION:
        failures.append("v121_classification_unexpected")
    if v121_source.get("passed") is not True:
        failures.append("v121_source_integrity_not_passed")
    if v121_source.get("failures") != []:
        failures.append("v121_source_failures_not_empty_or_malformed")

    digest_fields = {
        "v119_manifest_digest": v119_manifest.get("manifest_digest"),
        "v120_reported_manifest_digest": v120_source.get("reported_manifest_digest"),
        "v120_computed_manifest_digest": v120_source.get("computed_manifest_digest"),
        "v121_manifest_digest": v121_source.get("manifest_digest"),
    }
    for name, value in digest_fields.items():
        if not _valid_sha256_digest(value):
            failures.append(f"{name}_missing_or_malformed")
        elif value != manifest_digest:
            failures.append(f"{name}_mismatch")

    manifest_rare_counts = {
        action: sum(1 for row in manifest_rows if row.get("repaired_action") == action)
        for action in TARGET_ACTIONS
    }
    if manifest_rare_counts != EXPECTED_CURRENT_RARE_SUPPORT:
        failures.append("current_rare_support_unexpected")
    v121_rare_counts = {
        str(action): _int(count)
        for action, count in _mapping(v121_support.get("repaired_action_counts")).items()
        if str(action) in TARGET_ACTIONS
    }
    if v121_rare_counts != EXPECTED_CURRENT_RARE_SUPPORT:
        failures.append("v121_current_rare_support_unexpected")
    v120_rare_needed = {
        str(action): _int(count)
        for action, count in _mapping(
            v120_scarcity.get(
                "rare_action_additional_needed_for_train2_validation1_test1"
            )
        ).items()
        if str(action) in TARGET_ACTIONS
    }
    if v120_rare_needed != RARE_ACTION_ADDITIONAL_NEEDED:
        failures.append("v120_rare_action_additions_unexpected")
    v121_candidate_counts = _v121_candidate_counts(v121_search)
    if v121_candidate_counts != {action: 0 for action in TARGET_ACTIONS}:
        failures.append("v121_valid_candidates_not_zero")

    for version, report in (
        ("v119", v119),
        ("v120", v120),
        ("v121", v121),
    ):
        failures.extend(_no_authorization_failures(version, report))

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "manifest_row_count": len(manifest_rows),
        "manifest_digest": manifest_digest,
        "digest_fields": digest_fields,
        "manifest_digest_matches_all_sources": all(
            value == manifest_digest for value in digest_fields.values()
        ),
        "current_rare_support": manifest_rare_counts,
        "expected_current_rare_support": EXPECTED_CURRENT_RARE_SUPPORT,
        "v120_rare_action_additional_needed": v120_rare_needed,
        "v121_valid_candidate_counts": v121_candidate_counts,
        "v119_classification_primary": v119_classification.get("primary"),
        "v120_classification_primary": v120_classification.get("primary"),
        "v121_classification_primary": v121_classification.get("primary"),
        "authorization_checks": {
            version: _authorization_fields(report)
            for version, report in (("v119", v119), ("v120", v120), ("v121", v121))
        },
    }


def _search_config(
    *,
    target_actions: Sequence[str],
    max_seeds: int,
    seed_allowlist: Sequence[int | str],
    max_branches: int,
    max_candidates_per_action: int,
    candidate_archive_report: Mapping[str, object] | None,
) -> dict[str, object]:
    actions = tuple(action for action in target_actions if action in TARGET_ACTIONS)
    return {
        "target_actions": list(actions or TARGET_ACTIONS),
        "target_additional_needed": {
            action: RARE_ACTION_ADDITIONAL_NEEDED[action]
            for action in (actions or TARGET_ACTIONS)
        },
        "max_seeds": max(0, int(max_seeds)),
        "seed_allowlist": [str(seed) for seed in seed_allowlist],
        "max_branches": max(0, int(max_branches)),
        "max_candidates_per_action": max(1, int(max_candidates_per_action)),
        "candidate_source": "candidate_archive_rows",
        "candidate_archive_schema_version": _mapping(candidate_archive_report or {}).get(
            "schema_version"
        ),
        "deterministic_order": "seed_then_branch_id_then_action",
        "selection_authorized": False,
    }


def _collect_candidates(
    *,
    archive_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    source_integrity: Mapping[str, object],
    search_config: Mapping[str, object],
) -> dict[str, object]:
    target_actions = tuple(str(action) for action in search_config.get("target_actions", TARGET_ACTIONS))
    max_branches = int(search_config.get("max_branches", DEFAULT_MAX_BRANCHES))
    max_candidates = int(
        search_config.get(
            "max_candidates_per_action",
            DEFAULT_MAX_CANDIDATES_PER_ACTION,
        )
    )
    if source_integrity.get("passed") is not True:
        return _blocked_collection(
            target_actions=target_actions,
            blocked_reasons=source_integrity.get("failures", []),
        )
    identity_report = _candidate_archive_identity(archive_rows)
    try:
        contexts, integrity_failures, integrity_examples = _branch_repair_contexts(
            archive_rows
        )
    except FirstRecoveryOracleTieBreakAuditError as exc:
        return _blocked_collection(
            target_actions=target_actions,
            blocked_reasons=("objective_input_field_integrity",),
            branch_integrity_examples=(
                {
                    "reason": "objective_input_field_integrity",
                    "detail": str(exc),
                },
            ),
            candidate_archive_identity=identity_report,
        )
    if integrity_failures:
        return _blocked_collection(
            target_actions=target_actions,
            blocked_reasons=integrity_failures,
            branch_integrity_examples=integrity_examples,
            candidate_archive_identity=identity_report,
        )
    existing_branch_ids = {
        str(row.get("branch_id"))
        for row in manifest_rows
        if isinstance(row.get("branch_id"), str) and row.get("branch_id")
    }
    source_domain = _candidate_source_domain(
        contexts,
        existing_branch_ids=existing_branch_ids,
    )
    contexts = _budgeted_contexts(contexts, search_config)
    accepted: dict[str, list[dict[str, object]]] = {
        action: [] for action in target_actions
    }
    candidate_manifest_rows: list[dict[str, object]] = []
    rejection_counts: dict[str, Counter[str]] = {
        action: Counter() for action in target_actions
    }
    rejection_examples: dict[str, dict[str, list[dict[str, object]]]] = {
        action: defaultdict(list) for action in target_actions
    }
    branches_evaluated = 0
    for context in contexts:
        if branches_evaluated >= max_branches:
            break
        branches_evaluated += 1
        for action in target_actions:
            if len(accepted[action]) >= max_candidates:
                continue
            candidate, rejection = _candidate_for_context_action(
                context,
                action,
                existing_branch_ids=existing_branch_ids,
            )
            if candidate is None:
                rejection_counts[action][rejection or "not_candidate"] += 1
                _add_rejection_example(
                    rejection_examples[action],
                    rejection or "not_candidate",
                    context,
                    action,
                )
                continue
            accepted[action].append(candidate)
            candidate_manifest_rows.append(candidate)
    found_counts = {action: len(accepted[action]) for action in target_actions}
    found_actions = {
        action: count >= RARE_ACTION_ADDITIONAL_NEEDED.get(action, 1)
        for action, count in found_counts.items()
    }
    updated_support = {
        action: EXPECTED_CURRENT_RARE_SUPPORT.get(action, 0) + found_counts[action]
        for action in target_actions
    }
    support_feasible = all(
        updated_support[action] >= (
            EXPECTED_CURRENT_RARE_SUPPORT[action]
            + RARE_ACTION_ADDITIONAL_NEEDED[action]
        )
        for action in target_actions
    )
    return {
        "analysis_blocked": False,
        "candidate_policy": {
            "target_actions": list(target_actions),
            "objective_values_changed": False,
            "candidate_scope": (
                "new_or_unmanifested_first_recovery_branches_with_target_attack_candidate"
            ),
            "unique_objective_best_policy": (
                "unique_best_target_attack_allowed_only_when_current_rank1_is_target"
            ),
            "trainable_input_policy": "candidate_trainable_public_input_must_pass_v120_leakage_guard",
            "selection_authorized": False,
        },
        "candidate_archive_identity": identity_report,
        "candidate_source_domain": source_domain,
        "search_budget_used": {
            "candidate_archive_row_count": len(archive_rows),
            "candidate_branch_count": source_domain["candidate_branch_count"],
            "budgeted_candidate_branch_count": len(contexts),
            "manifest_branch_count": source_domain["manifest_branch_count"],
            "unmanifested_candidate_branch_count": (
                source_domain["unmanifested_candidate_branch_count"]
            ),
            "already_manifested_branch_count": (
                source_domain["already_manifested_branch_count"]
            ),
            "branches_evaluated": branches_evaluated,
            "max_branches": max_branches,
            "selected_seed_count": len({str(_context_seed(context)) for context in contexts}),
            "selected_seeds": sorted(
                {str(_context_seed(context)) for context in contexts}
            ),
            "budget_exhausted": branches_evaluated >= max_branches
            and not all(found_actions.values()),
        },
        "found_candidate_counts": found_counts,
        "found_target_actions": found_actions,
        "accepted_candidate_examples": {
            action: accepted[action][:MAX_EXAMPLES] for action in target_actions
        },
        "candidate_manifest_rows": candidate_manifest_rows,
        "rejection_counts": {
            action: _counter_to_dict(counter)
            for action, counter in rejection_counts.items()
        },
        "rejection_examples": {
            action: {
                reason: examples
                for reason, examples in sorted(per_action.items())
            }
            for action, per_action in rejection_examples.items()
        },
        "stricter_split_support_if_accepted": {
            "current_rare_support": {
                action: EXPECTED_CURRENT_RARE_SUPPORT[action]
                for action in target_actions
            },
            "accepted_additional_support": found_counts,
            "updated_rare_support": updated_support,
            "required_additional_support": {
                action: RARE_ACTION_ADDITIONAL_NEEDED[action]
                for action in target_actions
            },
            "train2_validation1_test1_would_be_feasible_for_targets": support_feasible,
        },
    }


def _blocked_collection(
    *,
    target_actions: Sequence[str],
    blocked_reasons: Sequence[object],
    branch_integrity_examples: Sequence[Mapping[str, object]] = (),
    candidate_archive_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "analysis_blocked": True,
        "blocked_reasons": sorted(set(str(reason) for reason in blocked_reasons)),
        "branch_integrity_examples": [dict(item) for item in branch_integrity_examples],
        "candidate_archive_identity": dict(candidate_archive_identity or {}),
        "candidate_source_domain": {
            "candidate_branch_count": 0,
            "manifest_branch_count": 0,
            "unmanifested_candidate_branch_count": 0,
            "already_manifested_branch_count": 0,
            "answer": "candidate_source_domain_unavailable",
        },
        "search_budget_used": {
            "candidate_archive_row_count": 0,
            "candidate_branch_count": 0,
            "branches_evaluated": 0,
            "budget_exhausted": False,
        },
        "found_candidate_counts": {action: 0 for action in target_actions},
        "found_target_actions": {action: False for action in target_actions},
        "accepted_candidate_examples": {action: [] for action in target_actions},
        "candidate_manifest_rows": [],
        "rejection_counts": {action: {} for action in target_actions},
        "rejection_examples": {action: {} for action in target_actions},
        "stricter_split_support_if_accepted": {
            "train2_validation1_test1_would_be_feasible_for_targets": False,
        },
    }


def _candidate_for_context_action(
    context: object,
    action: str,
    *,
    existing_branch_ids: set[str],
) -> tuple[dict[str, object] | None, str | None]:
    if context.branch_id in existing_branch_ids:
        return None, "already_in_v119_manifest"
    if _row_identity_reason(context.current) is not None:
        return None, "branch_identity_missing_or_malformed"
    target_rows = [
        row for row in context.rows if str(row.get("candidate_action")) == action
    ]
    if not target_rows:
        return None, "target_action_absent"
    valid_identity_rows = [
        row for row in target_rows
        if _row_identity_reason(row) is None
    ]
    if not valid_identity_rows:
        return None, "branch_identity_missing_or_malformed"
    try:
        equivalent = [
            row for row in valid_identity_rows
            if _serialized_objective_key(row) == context.rank1_key
        ]
    except FirstRecoveryOracleTieBreakAuditError:
        return None, "objective_input_field_integrity"
    if not equivalent:
        return None, "objective_equivalence_violation"
    legal = [row for row in equivalent if row.get("resolution_legal") is True]
    if not legal:
        return None, "resolution_illegal"
    if context.unique_objective_best and str(context.current.get("candidate_action")) != action:
        return None, "unique_objective_best_change_rejected"
    clean = [row for row in legal if _candidate_trainable_clean(context, row)]
    if not clean:
        return None, "trainable_metadata_leakage"
    selected = sorted(clean, key=_candidate_sort_key)[0]
    try:
        return _candidate_manifest_row(context, selected, action), None
    except FirstRecoveryOracleTieBreakAuditError:
        return None, "objective_input_field_integrity"


def _candidate_source_domain(
    contexts: Sequence[object],
    *,
    existing_branch_ids: set[str],
) -> dict[str, object]:
    candidate_branch_ids = {
        str(context.branch_id)
        for context in contexts
        if isinstance(getattr(context, "branch_id", None), str)
        and getattr(context, "branch_id")
    }
    already_manifested = candidate_branch_ids & existing_branch_ids
    unmanifested = candidate_branch_ids - existing_branch_ids
    answer = (
        "candidate_source_contains_no_unmanifested_branches"
        if candidate_branch_ids and not unmanifested
        else "candidate_source_contains_unmanifested_branches"
        if unmanifested
        else "candidate_source_contains_no_branches"
    )
    return {
        "answer": answer,
        "candidate_branch_count": len(candidate_branch_ids),
        "manifest_branch_count": len(existing_branch_ids),
        "unmanifested_candidate_branch_count": len(unmanifested),
        "already_manifested_branch_count": len(already_manifested),
        "already_manifested_branch_examples": sorted(already_manifested)[:MAX_EXAMPLES],
        "unmanifested_candidate_branch_examples": sorted(unmanifested)[:MAX_EXAMPLES],
    }


def _candidate_manifest_row(
    context: object,
    selected: Mapping[str, object],
    action: str,
) -> dict[str, object]:
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION
        ),
        "branch_id": context.branch_id,
        "target_action": action,
        "current_oracle_action": context.current.get("candidate_action"),
        "repaired_action": action,
        "current_archive_row_id": context.current.get("archive_row_id"),
        "candidate_archive_row_id": selected.get("archive_row_id"),
        "changed": str(context.current.get("candidate_action")) != action,
        "unique_objective_best": context.unique_objective_best,
        "objective_equivalence_verified": (
            _serialized_objective_key(selected) == context.rank1_key
        ),
        "selected_resolution_legal": selected.get("resolution_legal") is True,
        "trainable_public_input_present": isinstance(
            selected.get("trainable_public_input"),
            Mapping,
        )
        and bool(selected.get("trainable_public_input")),
        "trainable_public_input_clean": _candidate_trainable_clean(context, selected),
        "trainable_public_input_contents_exposed": False,
        "legal_tied_candidate_actions": sorted(
            str(row.get("candidate_action")) for row in context.legal_equivalent
        ),
        "serialized_objective_key": list(context.rank1_key),
        "diagnostics_only": True,
        "selection_authorized": False,
    }


def _candidate_archive_identity(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures_by_id: dict[str, str] = {}
    examples: list[dict[str, object]] = []
    missing_count = 0
    mismatch_count = 0
    for index, row in enumerate(rows):
        branch_id = _mapping(row.get("provenance")).get("branch_id")
        archive_row_id = row.get("archive_row_id")
        candidate_action = row.get("candidate_action")
        reason = _row_identity_reason(row)
        if reason in (
            "provenance_branch_id_missing_or_malformed",
            "archive_row_id_missing_or_malformed",
        ):
            missing_count += 1
        elif reason is not None:
            mismatch_count += 1
        if reason is None:
            continue
        key = str(archive_row_id) if archive_row_id is not None else f"row-{index}"
        failures_by_id[key] = reason
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "row_index": index,
                    "archive_row_id": archive_row_id,
                    "provenance_branch_id": branch_id,
                    "candidate_action": candidate_action,
                    "reason": reason,
                }
            )
    return {
        "row_count": len(rows),
        "identity_failure_count": len(failures_by_id),
        "missing_or_malformed_identity_count": missing_count,
        "identity_mismatch_count": mismatch_count,
        "identity_failure_examples": examples,
        "row_failures_by_archive_row_id": failures_by_id,
    }


def _row_identity_reason(row: Mapping[str, object]) -> str | None:
    branch_id = _mapping(row.get("provenance")).get("branch_id")
    archive_row_id = row.get("archive_row_id")
    candidate_action = row.get("candidate_action")
    if not isinstance(branch_id, str) or not branch_id:
        return "provenance_branch_id_missing_or_malformed"
    if not isinstance(archive_row_id, str) or not archive_row_id:
        return "archive_row_id_missing_or_malformed"
    expected_prefix = f"{branch_id}::action::"
    if not archive_row_id.startswith(expected_prefix):
        return "archive_row_id_branch_prefix_mismatch"
    if archive_row_id.removeprefix(expected_prefix) != str(candidate_action):
        return "archive_row_id_candidate_action_suffix_mismatch"
    return None


def _budgeted_contexts(
    contexts: Sequence[object],
    search_config: Mapping[str, object],
) -> list[object]:
    explicit_seeds = {
        str(seed) for seed in search_config.get("seed_allowlist", []) if str(seed)
    }
    max_seeds = int(search_config.get("max_seeds", DEFAULT_MAX_SEEDS))
    sorted_contexts = sorted(
        contexts,
        key=lambda context: (str(_context_seed(context)), context.branch_id),
    )
    if explicit_seeds:
        return [
            context for context in sorted_contexts
            if str(_context_seed(context)) in explicit_seeds
        ]
    selected_seeds: set[str] = set()
    result: list[object] = []
    for context in sorted_contexts:
        seed = str(_context_seed(context))
        if seed not in selected_seeds and len(selected_seeds) >= max_seeds:
            continue
        selected_seeds.add(seed)
        result.append(context)
    return result


def _context_seed(context: object) -> object:
    return _mapping(context.current.get("provenance")).get("seed", "unknown")


def _candidate_trainable_clean(context: object, row: Mapping[str, object]) -> bool:
    trainable = row.get("trainable_public_input")
    if not isinstance(trainable, Mapping) or not trainable:
        return False
    leakage = _trainable_leakage(
        [
            {
                "branch_id": context.branch_id,
                "repaired_action": row.get("candidate_action"),
                "trainable_public_input": trainable,
            }
        ]
    )
    return (
        leakage["split_key_leak_count"] == 0
        and leakage["forbidden_metadata_key_count"] == 0
    )


def _candidate_sort_key(row: Mapping[str, object]) -> tuple[str, str]:
    return (str(row.get("candidate_action")), str(row.get("archive_row_id")))


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
    collection: Mapping[str, object],
) -> dict[str, object]:
    labels = ["diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
    if source_integrity.get("passed") is not True:
        failures = [str(item) for item in source_integrity.get("failures", [])]
        primary = (
            "missing_evidence_inconclusive"
            if any(failure.startswith("missing_") for failure in failures)
            else "rare_attack_coverage_source_integrity_failed"
        )
    else:
        found = _mapping(collection.get("found_target_actions"))
        source_domain = _mapping(collection.get("candidate_source_domain"))
        if (
            _int(source_domain.get("candidate_branch_count")) > 0
            and _int(source_domain.get("unmanifested_candidate_branch_count")) == 0
        ):
            labels.append("candidate_source_contains_no_unmanifested_branches")
        found_count = sum(1 for value in found.values() if value is True)
        if found and found_count == len(found):
            primary = "rare_attack_coverage_candidates_found"
        elif found_count > 0:
            primary = "rare_attack_coverage_partial"
        else:
            primary = "rare_attack_coverage_not_found_within_budget"
    labels.insert(0, primary)
    return {
        "primary": primary,
        "labels": _dedupe_allowed(labels),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(
    *,
    classification: Mapping[str, object],
    collection: Mapping[str, object],
) -> dict[str, object]:
    primary = classification.get("primary")
    if primary in (
        "missing_evidence_inconclusive",
        "rare_attack_coverage_source_integrity_failed",
    ):
        next_step = "source_integrity_must_pass_before_rare_attack_collection"
        summary = (
            "Source integrity failed, so rare-attack coverage collection cannot be "
            "trusted from these inputs."
        )
    elif primary == "rare_attack_coverage_candidates_found":
        next_step = "review_diagnostics_only_rare_attack_candidates_before_any_contract_update"
        summary = (
            "The bounded diagnostics harness found enough rare-attack candidates "
            "to clear the v120 target if a future contract update accepts them; "
            "this report does not authorize training, shadow scoring, readiness, "
            "or runtime policy changes."
        )
    elif primary == "rare_attack_coverage_partial":
        next_step = "continue_bounded_diagnostics_only_rare_attack_collection"
        summary = (
            "The bounded diagnostics harness found partial rare-attack coverage, "
            "but not enough to clear the v120 target."
        )
    else:
        source_domain = _mapping(collection.get("candidate_source_domain"))
        if (
            _int(source_domain.get("candidate_branch_count")) > 0
            and _int(source_domain.get("unmanifested_candidate_branch_count")) == 0
        ):
            next_step = "use_active_coverage_archive_with_unmanifested_branches"
            summary = (
                "The candidate source contains no unmanifested branches relative "
                "to v119, so increasing max_branches on this source cannot add "
                "rare-attack support."
            )
        else:
            next_step = "increase_diagnostics_only_first_recovery_attack_coverage_budget_or_close_line"
            summary = (
                "No sufficient rare-attack coverage was found within the configured "
                "bounded diagnostics budget."
            )
    support = _mapping(collection.get("stricter_split_support_if_accepted"))
    return {
        "next_step": next_step,
        "summary": summary,
        "would_clear_v120_rare_action_limitation_if_accepted": support.get(
            "train2_validation1_test1_would_be_feasible_for_targets",
            False,
        ),
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "viewer_change_recommended": False,
        "replay_golden_change_recommended": False,
        "shadow_scorer_change_recommended": False,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "claim_causality": False,
    }


def _no_authorization_failures(
    version: str,
    report: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    recommendation = _mapping(report.get("recommendation"))
    for field in NO_AUTHORIZATION_FIELDS:
        if recommendation.get(field) is not False:
            failures.append(f"{version}_{field}_not_false")
    if version == "v121" and "replay_golden_change_recommended" in recommendation:
        if recommendation.get("replay_golden_change_recommended") is not False:
            failures.append("v121_replay_golden_change_recommended_not_false")
    return failures


def _authorization_fields(report: Mapping[str, object]) -> dict[str, object]:
    recommendation = _mapping(report.get("recommendation"))
    return {field: recommendation.get(field) for field in NO_AUTHORIZATION_FIELDS}


def _v121_candidate_counts(candidate_search: Mapping[str, object]) -> dict[str, int]:
    return {
        action: _int(_mapping(_mapping(candidate_search.get("per_action")).get(action)).get("valid_candidate_count"))
        for action in TARGET_ACTIONS
    }


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
        if label in allowed and label not in seen:
            result.append(label)
            seen.add(label)
    return result
