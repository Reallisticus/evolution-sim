from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V124_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _int,
    _mapping,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    _trainable_leakage,
)
from evolution_sim.mind.first_recovery_shadow_ranker import (
    DOMINANT_SELECTED_ACTION_SHARE_MAX,
    MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PROPOSAL_SCHEMA_VERSION = (
    "mind_v3_first_recovery_shadow_scorer_proposal_v1"
)
MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PROPOSAL_POLICY = (
    "diagnostics_only_first_recovery_v125_shadow_scorer_proposal_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v125-first-recovery-shadow-scorer-proposal.json"
)

EXPECTED_V124_CLASSIFICATION = "accepted_rare_attack_contract_ready_for_shadow_proposal"
EXPECTED_MANIFEST_ROW_COUNT = 108
EXPECTED_REPAIRED_ACTION_COUNTS: dict[str, int] = {
    "attack_east": 4,
    "attack_west": 4,
    "drink": 7,
    "eat": 17,
    "move_east": 16,
    "move_north": 15,
    "move_south": 15,
    "move_west": 14,
    "stay": 16,
}
DOMINANT_ACTION_SHARE_CAP = DOMINANT_SELECTED_ACTION_SHARE_MAX

AUTHORIZATION_FIELDS: tuple[str, ...] = (
    "training_executed",
    "trained_artifact_change_recommended",
    "v113_readiness_rerun_allowed",
    "downstream_shadow_scorer_allowed",
    "runtime_policy_change_recommended",
    "gate_change_recommended",
    "viewer_change_recommended",
    "replay_golden_change_recommended",
    "claim_causality",
)

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "shadow_scorer_proposal_ready_for_review",
    "shadow_scorer_proposal_source_integrity_failed",
    "diagnostics_only_no_runtime_promotion",
    "readiness_rerun_blocked",
)

MAX_EXAMPLES = 16


@dataclass(frozen=True, slots=True)
class FirstRecoveryShadowScorerProposalBuild:
    report: dict[str, object]


def build_first_recovery_shadow_scorer_proposal(
    *,
    v124_report: Mapping[str, object] | None = None,
    v124_report_path: str | Path | None = DEFAULT_V124_REPORT_PATH,
    v124_manifest_rows: Sequence[Mapping[str, object]] | None = None,
    v124_manifest_path: str | Path | None = DEFAULT_V124_MANIFEST_PATH,
) -> FirstRecoveryShadowScorerProposalBuild:
    report_payload, report_evidence = _resolve_json_report(
        v124_report,
        v124_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION,
    )
    manifest_rows, manifest_evidence = _resolve_manifest_rows(
        v124_manifest_rows,
        v124_manifest_path,
    )
    source_integrity = _source_integrity(
        v124_report=report_payload,
        manifest_rows=manifest_rows,
        report_evidence=report_evidence,
        manifest_evidence=manifest_evidence,
    )
    proposal = _proposal(
        v124_report=report_payload or {},
        manifest_rows=manifest_rows,
        source_integrity=source_integrity,
    )
    classification = _classification(source_integrity)
    recommendation = _recommendation(classification)
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PROPOSAL_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PROPOSAL_POLICY,
        "contract": _contract(),
        "source_reports": {
            "v124_report": report_evidence,
            "v124_manifest": manifest_evidence,
        },
        "source_integrity": source_integrity,
        "proposal": proposal,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryShadowScorerProposalBuild(report=report)


def write_first_recovery_shadow_scorer_proposal_report(
    build: FirstRecoveryShadowScorerProposalBuild,
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
        "schema_version": MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PROPOSAL_SCHEMA_VERSION,
        "diagnostics_only": True,
        "proposal_only": True,
        "shadow_scorer_executed": False,
        "training_executed": False,
        "model_artifact_created": False,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "foundation_effect": "none",
        "readiness_rerun_executed": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "claim_causality": False,
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PROPOSAL_SCHEMA_VERSION
                ),
                "policy": MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PROPOSAL_POLICY,
            }
        ),
    }


def _resolve_manifest_rows(
    rows: Sequence[Mapping[str, object]] | None,
    path: str | Path | None,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    if rows is not None:
        parsed = tuple(dict(row) for row in rows)
        return parsed, {
            "loaded": True,
            "in_memory": True,
            "path": str(path) if path is not None else None,
            "row_count": len(parsed),
        }
    if path is None:
        return (), {
            "loaded": False,
            "path": None,
            "row_count": 0,
            "error": "missing_manifest_path",
        }
    source = Path(path)
    if not source.exists():
        return (), {
            "loaded": False,
            "path": str(source),
            "row_count": 0,
            "error": "missing_manifest_file",
        }
    parsed: list[dict[str, object]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"manifest row {line_number} must be an object")
                parsed.append(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return (), {
            "loaded": False,
            "path": str(source),
            "row_count": 0,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    return tuple(parsed), {
        "loaded": True,
        "path": str(source),
        "row_count": len(parsed),
    }


def _source_integrity(
    *,
    v124_report: Mapping[str, object] | None,
    manifest_rows: Sequence[Mapping[str, object]],
    report_evidence: Mapping[str, object],
    manifest_evidence: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    report = _mapping(v124_report or {})
    contract = _mapping(report.get("contract"))
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    contract_checks = _mapping(report.get("contract_checks"))
    split_support = _mapping(report.get("split_support"))
    manifest = _mapping(report.get("manifest"))
    recommendation = _mapping(report.get("recommendation"))
    join_validation = _mapping(source.get("accepted_candidate_join_validation"))
    row_replay = _mapping(source.get("accepted_candidate_row_replay_validation"))
    trainable_leakage = _mapping(contract_checks.get("trainable_leakage"))
    branch_archive_leakage = _mapping(
        contract_checks.get("branch_archive_trainable_leakage")
    )
    source_seed_policy = _mapping(report.get("source_seed_policy"))
    manifest_digest = stable_payload_digest(list(manifest_rows))
    row_counts = _action_counts(manifest_rows)
    branch_ids = _valid_branch_ids(manifest_rows)
    unsupported = _unsupported_selection_report(manifest_rows)
    loaded_leakage = _trainable_leakage(manifest_rows)
    dominant = _dominant_action(row_counts)
    contract_boundary = _v124_contract_boundary_checks(contract)

    if report_evidence.get("loaded") is not True:
        failures.append("missing_v124_report")
    elif report_evidence.get("schema_matches") is not True:
        failures.append("v124_report_schema_mismatch")
    if manifest_evidence.get("loaded") is not True:
        failures.append("missing_v124_manifest")

    if classification.get("primary") != EXPECTED_V124_CLASSIFICATION:
        failures.append("v124_classification_unexpected")
    if source.get("passed") is not True:
        failures.append("v124_source_integrity_not_passed")
    if source.get("failures") != []:
        failures.append("v124_source_failures_not_empty_or_malformed")
    if contract_checks.get("passed") is not True:
        failures.append("v124_contract_checks_not_passed")
    if contract_checks.get("failures") != []:
        failures.append("v124_contract_failures_not_empty_or_malformed")
    failures.extend(contract_boundary["failures"])

    if _int(manifest.get("manifest_row_count")) != EXPECTED_MANIFEST_ROW_COUNT:
        failures.append("v124_report_manifest_row_count_unexpected")
    if len(manifest_rows) != EXPECTED_MANIFEST_ROW_COUNT:
        failures.append("v124_manifest_row_count_unexpected")
    if _int(contract_checks.get("manifest_row_count")) != EXPECTED_MANIFEST_ROW_COUNT:
        failures.append("v124_contract_manifest_row_count_unexpected")
    if _int(contract_checks.get("unique_branch_count")) != EXPECTED_MANIFEST_ROW_COUNT:
        failures.append("v124_unique_branch_count_unexpected")
    if len(branch_ids) != EXPECTED_MANIFEST_ROW_COUNT:
        failures.append("v124_manifest_branch_id_missing_or_malformed")
    if len(set(branch_ids)) != len(branch_ids):
        failures.append("v124_manifest_branch_ids_not_unique")

    if manifest.get("manifest_digest") != manifest_digest:
        failures.append("v124_manifest_digest_mismatch")
    if _dict_of_ints(contract_checks.get("repaired_action_counts")) != (
        EXPECTED_REPAIRED_ACTION_COUNTS
    ):
        failures.append("v124_repaired_action_counts_unexpected")
    if _counter_to_dict(row_counts) != EXPECTED_REPAIRED_ACTION_COUNTS:
        failures.append("v124_manifest_repaired_action_counts_unexpected")

    if split_support.get("strict_train_validation_test_support_met") is not True:
        failures.append("v124_strict_split_support_not_true")
    if split_support.get("all_splits_contain_every_action_class") is not True:
        failures.append("v124_split_missing_action_class")
    if join_validation.get("passed") is not True:
        failures.append("v124_join_validation_not_passed")
    if _int(join_validation.get("mismatch_count")) != 0:
        failures.append("v124_join_validation_mismatch_nonzero")
    if join_validation.get("failure_labels") != []:
        failures.append("v124_join_validation_failures_not_empty_or_malformed")
    if row_replay.get("passed") is not True:
        failures.append("v124_row_replay_validation_not_passed")
    if _int(row_replay.get("failure_count")) != 0:
        failures.append("v124_row_replay_validation_failures_nonzero")
    if row_replay.get("failure_labels") != []:
        failures.append("v124_row_replay_validation_labels_not_empty_or_malformed")

    if _int(trainable_leakage.get("split_key_leak_count")) != 0:
        failures.append("v124_trainable_split_leakage_nonzero")
    if _int(trainable_leakage.get("forbidden_metadata_key_count")) != 0:
        failures.append("v124_trainable_metadata_leakage_nonzero")
    if _int(branch_archive_leakage.get("leak_count")) != 0:
        failures.append("v124_branch_archive_trainable_leakage_nonzero")
    if loaded_leakage["split_key_leak_count"]:
        failures.append("manifest_trainable_split_leakage_detected")
    if loaded_leakage["forbidden_metadata_key_count"]:
        failures.append("manifest_trainable_metadata_leakage_detected")

    if unsupported["unsupported_selection_count"]:
        failures.append("manifest_unsupported_selection_detected")
    if source_seed_policy.get("source_seed_in_trainable_input") is not False:
        failures.append("v124_source_seed_trainable_policy_not_false")
    if _int(source_seed_policy.get("source_seed_trainable_leak_count")) != 0:
        failures.append("v124_source_seed_trainable_leakage_nonzero")
    if source_seed_policy.get("strict_heldout_generalization_claimed") is not False:
        failures.append("v124_strict_heldout_claim_not_false")
    if dominant["share"] > DOMINANT_ACTION_SHARE_CAP:
        failures.append("dominant_action_share_exceeds_cap")

    for field in AUTHORIZATION_FIELDS:
        if recommendation.get(field) not in (False, None):
            failures.append(f"v124_{field}_not_false_or_none")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v124_classification_primary": classification.get("primary"),
        "v124_source_integrity_passed": source.get("passed"),
        "v124_contract_boundary_checks": contract_boundary,
        "v124_contract_checks_passed": contract_checks.get("passed"),
        "manifest_row_count": len(manifest_rows),
        "unique_branch_count": len(set(branch_ids)),
        "manifest_digest": manifest_digest,
        "reported_manifest_digest": manifest.get("manifest_digest"),
        "manifest_digest_matches": manifest.get("manifest_digest") == manifest_digest,
        "repaired_action_counts": _counter_to_dict(row_counts),
        "expected_repaired_action_counts": EXPECTED_REPAIRED_ACTION_COUNTS,
        "dominant_action": dominant,
        "dominant_action_share_cap": DOMINANT_ACTION_SHARE_CAP,
        "strict_split_support": {
            "strict_train_validation_test_support_met": split_support.get(
                "strict_train_validation_test_support_met"
            ),
            "all_splits_contain_every_action_class": split_support.get(
                "all_splits_contain_every_action_class"
            ),
            "minimum_per_action_support_by_split": split_support.get(
                "minimum_per_action_support_by_split"
            ),
        },
        "accepted_candidate_join_validation": dict(join_validation),
        "accepted_candidate_row_replay_validation": dict(row_replay),
        "report_trainable_leakage": {
            "split_key_leak_count": _int(
                trainable_leakage.get("split_key_leak_count")
            ),
            "forbidden_metadata_key_count": _int(
                trainable_leakage.get("forbidden_metadata_key_count")
            ),
            "branch_archive_leak_count": _int(branch_archive_leakage.get("leak_count")),
        },
        "manifest_trainable_leakage": loaded_leakage,
        "unsupported_selection_report": unsupported,
        "authorization_checks": {
            field: recommendation.get(field) for field in AUTHORIZATION_FIELDS
        },
        "source_seed_policy": dict(source_seed_policy),
    }


def _v124_contract_boundary_checks(contract: Mapping[str, object]) -> dict[str, object]:
    failures: list[str] = []
    if contract.get("diagnostics_only") is not True:
        failures.append("v124_contract_diagnostics_only_not_true")
    for field in (
        "training_executed",
        "readiness_rerun_executed",
        "v113_readiness_rerun_allowed",
        "downstream_shadow_scorer_allowed",
        "claim_causality",
    ):
        if contract.get(field) is not False:
            failures.append(f"v124_contract_{field}_not_false")
    for field in (
        "runtime_policy_effect",
        "trained_artifact_effect",
        "gate_effect",
        "viewer_effect",
        "replay_golden_effect",
    ):
        if contract.get(field) != "none":
            failures.append(f"v124_contract_{field}_not_none")
    for field in (
        "shadow_scorer_executed",
        "shadow_scorer_implemented",
        "model_artifact_created",
        "runtime_policy_implemented",
        "archive_replay_executed",
        "source_seed_provenance_trainable",
        "strict_heldout_generalization_claimed",
        "synthetic_labels_created",
        "objective_values_changed",
    ):
        if field in contract and contract.get(field) is not False:
            failures.append(f"v124_contract_{field}_not_false")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "diagnostics_only": contract.get("diagnostics_only"),
        "training_executed": contract.get("training_executed"),
        "readiness_rerun_executed": contract.get("readiness_rerun_executed"),
        "v113_readiness_rerun_allowed": contract.get("v113_readiness_rerun_allowed"),
        "downstream_shadow_scorer_allowed": contract.get(
            "downstream_shadow_scorer_allowed"
        ),
        "claim_causality": contract.get("claim_causality"),
        "effects": {
            field: contract.get(field)
            for field in (
                "runtime_policy_effect",
                "trained_artifact_effect",
                "gate_effect",
                "viewer_effect",
                "replay_golden_effect",
            )
        },
        "optional_no_effect_flags": {
            field: contract.get(field)
            for field in (
                "shadow_scorer_executed",
                "shadow_scorer_implemented",
                "model_artifact_created",
                "runtime_policy_implemented",
                "archive_replay_executed",
                "source_seed_provenance_trainable",
                "strict_heldout_generalization_claimed",
                "synthetic_labels_created",
                "objective_values_changed",
            )
            if field in contract
        },
    }


def _proposal(
    *,
    v124_report: Mapping[str, object],
    manifest_rows: Sequence[Mapping[str, object]],
    source_integrity: Mapping[str, object],
) -> dict[str, object]:
    split_support = _mapping(v124_report.get("split_support"))
    return {
        "proposal_only": True,
        "shadow_scorer_execution_planned": False,
        "shadow_scorer_execution_authorized": False,
        "training_planned": False,
        "model_artifact_planned": False,
        "source_contract": "v124_accepted_rare_attack_manifest",
        "source_requirements": {
            "source_integrity_must_pass_before_execution": True,
            "manifest_digest_pinned": source_integrity.get("manifest_digest"),
            "manifest_row_count": source_integrity.get("manifest_row_count"),
            "unique_branch_count": source_integrity.get("unique_branch_count"),
        },
        "planned_trainable_input_allowlist": _trainable_allowlist(manifest_rows),
        "forbidden_input_list": _forbidden_input_list(),
        "planned_split_policy": {
            "source": "v124.split_support.policy",
            "policy": split_support.get("policy"),
            "splits": split_support.get("splits"),
            "strict_targets": split_support.get("strict_targets"),
            "split_assignment_excluded_from_trainable_public_input": True,
        },
        "planned_scorer_acceptance_metrics": _planned_acceptance_metrics(),
        "planned_blocker_taxonomy": _planned_blocker_taxonomy(),
        "planned_report_paths": {
            "execution_report": (
                "output/mind/mind-v3-v126-first-recovery-shadow-scorer-execution.json"
            ),
            "prediction_rows": (
                "output/mind/mind-v3-v126-first-recovery-shadow-scorer-predictions.jsonl"
            ),
            "no_model_artifact_output": True,
        },
        "explicit_boundary_statement": (
            "v125 is a diagnostics-only design/contract proposal. It does not run "
            "a shadow scorer, train a model, create an artifact, change runtime "
            "policy, authorize readiness, or promote any behavior."
        ),
    }


def _trainable_allowlist(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    top_level = sorted(
        {
            str(key)
            for row in manifest_rows
            for key in _mapping(row.get("trainable_public_input")).keys()
        }
    )
    return {
        "policy": "only_public_candidate_state_fields_from_trainable_public_input",
        "top_level_keys_observed": top_level,
        "required_top_level_keys": [
            "action_mask",
            "candidate_action",
            "candidate_action_index",
            "post_carrion_first_recovery",
            "public_transition_context",
            "schema_version",
            "target_public_state_before",
        ],
        "nested_public_state_keys": [
            "age",
            "alive",
            "energy_ratio",
            "health_ratio",
            "hydration_ratio",
        ],
        "trainable_public_input_contents_embedded_in_v125": False,
    }


def _forbidden_input_list() -> dict[str, object]:
    return {
        "forbidden_families": [
            "seed",
            "fixture_or_source_identity",
            "branch_id",
            "archive_row_id",
            "logged_action_fallback",
            "private_world_state",
            "audit_metadata",
            "future_or_current_outcome_leakage",
        ],
        "explicit_forbidden_keys": [
            "seed",
            "seed_id",
            "source",
            "source_kind",
            "source_path",
            "fixture",
            "fixture_name",
            "branch_id",
            "archive_row_id",
            "current_archive_row_id",
            "repaired_archive_row_id",
            "logged_action",
            "logged_action_fallback",
            "provenance",
            "non_trainable_audit_metadata",
            "private_world_state",
            "simulation_world",
            "world",
            "observation_digest",
            "replay_verification_digest",
            "terminal_alive_delta",
            "birth_delta",
            "death_delta",
            "death_reduction_delta",
            "recovery_vitals_deltas",
            "first_action_outcome",
            "oracle_rank",
            "oracle_best_action",
            "serialized_objective_key",
        ],
    }


def _planned_acceptance_metrics() -> dict[str, object]:
    return {
        "source_integrity": "all_v124_source_and_manifest_checks_must_pass",
        "heldout_current_row_signal": {
            "required": True,
            "criterion": "heldout/current-row signal required before any runtime planning",
            "expected_positive_label": "shadow_ranker_current_row_learns_signal",
        },
        "seed29_evaluation": {
            "required": True,
            "criterion": "seed29 evaluation must pass",
            "expected_positive_label": "shadow_ranker_seed29_passes",
        },
        "fixture_open_generalization": {
            "required": True,
            "criterion": "fixture-open/generalization must pass",
            "expected_positive_label": "shadow_ranker_fixture_open_generalizes",
        },
        "action_distribution": {
            "dominant_selected_action_share_max": DOMINANT_SELECTED_ACTION_SHARE_MAX,
            "criterion": "dominant selected action share <= 0.5",
        },
        "unsupported_action_audit": {
            "unsupported_action_rate_required": 0.0,
            "criterion": "unsupported action rate == 0",
        },
        "leakage_audit": {
            "trainable_leakage_required": 0,
            "criterion": "trainable leakage == 0",
        },
        "material_gain_recall": {
            "minimum_recall": MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
            "criterion": "material-gain recall >= 0.2",
        },
        "split_support": "train_min_2_validation_min_1_test_min_1_per_action",
        "required_execution_outputs": [
            "per_branch_prediction_rows",
            "heldout_signal",
            "seed29_evaluation",
            "fixture_open_evaluation",
            "material_gain_recall",
            "action_distribution",
            "unsupported_action_audit",
            "leakage_audit",
            "per_action_confusion_or_rank_summary",
        ],
        "not_acceptance": [
            "no_training_authorization",
            "no_runtime_policy_authorization",
            "no_v113_readiness_authorization",
            "no_gate_or_replay_golden_change",
        ],
    }


def _planned_blocker_taxonomy() -> list[dict[str, object]]:
    return [
        {
            "label": "source_integrity_failed",
            "meaning": "v124 report or manifest checks fail before execution.",
        },
        {
            "label": "trainable_leakage_detected",
            "meaning": "seed/source/branch/private/audit/outcome metadata enters trainable inputs.",
        },
        {
            "label": "split_support_failed",
            "meaning": "deterministic train/validation/test support no longer covers every action.",
        },
        {
            "label": "action_collapse_detected",
            "meaning": "shadow predictions exceed the dominant-action share cap.",
        },
        {
            "label": "unsupported_selection_detected",
            "meaning": "predicted selections are resolution-illegal or not supported by the repaired-label contract.",
        },
        {
            "label": "heldout_or_fixture_open_regression",
            "meaning": "validation/test or fixture-open audits fail while identity is excluded from inputs.",
        },
        {
            "label": "heldout_signal_missing_or_weak",
            "meaning": "current-row heldout signal is missing or too weak for a shadow-scorer execution proposal.",
        },
        {
            "label": "seed29_failed",
            "meaning": "seed29 grouped holdout evaluation does not pass.",
        },
        {
            "label": "fixture_open_failed",
            "meaning": "fixture-open/generalization evaluation does not pass.",
        },
        {
            "label": "material_gain_recall_below_floor",
            "meaning": "material-gain recall is below the existing shadow-ranker floor.",
        },
        {
            "label": "runtime_or_promotion_boundary_crossed",
            "meaning": "execution attempts to train, promote, change gates, alter replay/goldens, or authorize readiness.",
        },
    ]


def _classification(source_integrity: Mapping[str, object]) -> dict[str, object]:
    primary = (
        "shadow_scorer_proposal_ready_for_review"
        if source_integrity.get("passed") is True
        else "shadow_scorer_proposal_source_integrity_failed"
    )
    return {
        "primary": primary,
        "labels": _dedupe_allowed(
            [primary, "diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
        ),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    ready = classification.get("primary") == "shadow_scorer_proposal_ready_for_review"
    return {
        "next_step": (
            "review_v125_proposal_before_any_shadow_scorer_execution_slice"
            if ready
            else "source_integrity_must_pass_before_shadow_scorer_proposal_review"
        ),
        "summary": (
            "v124 is sufficient to review a future shadow-scorer execution design, "
            "but v125 does not execute or authorize that scorer."
            if ready
            else "v124 source integrity failed, so no shadow-scorer execution design "
            "should proceed from these inputs."
        ),
        "proposal_ready_for_review": ready,
        "shadow_scorer_executed": False,
        "shadow_scorer_execution_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "training_executed": False,
        "trained_artifact_change_recommended": False,
        "model_artifact_created": False,
        "runtime_policy_change_recommended": False,
        "v113_readiness_rerun_allowed": False,
        "gate_change_recommended": False,
        "viewer_change_recommended": False,
        "replay_golden_change_recommended": False,
        "foundation_change_recommended": False,
        "claim_causality": False,
    }


def _unsupported_selection_report(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    examples: list[dict[str, object]] = []
    count = 0
    for index, row in enumerate(manifest_rows):
        reason = None
        if row.get("selected_resolution_legal") is not True:
            reason = "resolution_not_legal"
        elif row.get("objective_equivalence_verified") is not True:
            reason = "objective_equivalence_not_verified"
        elif row.get("unique_objective_best") is True and row.get("changed") is True:
            reason = "unique_objective_best_changed"
        if reason is None:
            continue
        count += 1
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "row_index": index,
                    "branch_id": row.get("branch_id"),
                    "repaired_action": row.get("repaired_action"),
                    "reason": reason,
                }
            )
    return {"unsupported_selection_count": count, "examples": examples}


def _dominant_action(counts: Counter[str]) -> dict[str, object]:
    if not counts:
        return {"action": None, "count": 0, "share": 0.0}
    action, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    total = sum(counts.values())
    return {
        "action": action,
        "count": count,
        "share": count / total if total else 0.0,
        "total": total,
    }


def _action_counts(rows: Sequence[Mapping[str, object]]) -> Counter[str]:
    return Counter(str(row.get("repaired_action")) for row in rows)


def _valid_branch_ids(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return [
        str(row.get("branch_id"))
        for row in rows
        if isinstance(row.get("branch_id"), str) and row.get("branch_id")
    ]


def _dict_of_ints(value: object) -> dict[str, int]:
    return {
        str(key): _int(item)
        for key, item in _mapping(value).items()
    }


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
