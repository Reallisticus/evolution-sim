from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import gzip
import json
from pathlib import Path

from evolution_sim.mind.candidate_campaign import (
    _float,
    _int,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v188_terminal_carrion_survival_support as v188,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V189_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V189_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v189-carrion-survivor-continuation-terminal-survival-support-dataset-audit.json"
)
DEFAULT_V188_REPORT_PATH = v188.DEFAULT_OUTPUT_PATH
EXPECTED_V188_REPORT_EXACT_DIGEST = (
    "9a5a28685fd7adea175b1c8370f7042cdfbd1624b2c9ed3d0e42235b0bd4e5c0"
)
EXPECTED_V187_REPORT_EXACT_DIGEST = v188.EXPECTED_V187_REPORT_EXACT_DIGEST
EXPECTED_V188_REQUIRED_ROUTE = v188.POSITIVE_SUPPORT_ROUTE
EXPECTED_V188_BACKUP = (
    "gdrive:evolution-sim-backups/archives/"
    "20260612T171041Z-v188-terminal-carrion-survival-support.tar.zst"
)
EXPECTED_V188_BACKUP_SHA256 = (
    "67856f7b67c9156ad64c7da66ce5f118b03c1132d30c83b00c097ea0dedf7747"
)

TARGETED_SUPPORT_EXPANSION_ROUTE = (
    "v190_targeted_legal_terminal_survival_support_expansion_before_slice_3_training_no_training"
)
SLICE_3_TRAINING_ROUTE = "v190_transition_row_policy_training_slice_3_opt_in"
STOP_ROUTE = "stop"

DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE = 0.50
DEFAULT_REQUIRED_TARGET_SEED_COVERAGE_FRACTION = 1.0
DEFAULT_EXPECTED_SELECTED_SUPPORT = {
    "seed": 13,
    "branch_id": "carrion-only-seed-13-branch-0-tick-0-agent-9",
    "continuation_script": "conserve_after_carrion",
    "alive_agents": 1,
    "births": 5,
    "unsupported_requested_action_count": 0,
    "unsupported_resolved_action_count": 0,
    "dominant_requested_action": "stay",
    "dominant_requested_action_share": 0.5392,
    "trajectory_path": (
        "output/mind/v188-terminal-carrion-survival-support-trajectories/"
        "branch-carrion-only-seed-13-branch-0-tick-0-agent-9-conserve-after-carrion-120.jsonl.gz"
    ),
}
FORBIDDEN_TRAINABLE_TOKENS = (
    "seed",
    "fixture",
    "branch",
    "path",
    "digest",
    "provenance",
    "private",
    "future",
    "source",
    "trajectory",
    "tick",
    "agent_id",
)
FALSE_LIFECYCLE_FLAGS = (
    "training_ran",
    "training_artifact_created",
    "slice_3_training_consumed",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "promotion_authorized",
    "gate_relaxation_allowed",
)


def run_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit(
    *,
    v188_report_path: str | Path = DEFAULT_V188_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v188_report_exact_digest: str = EXPECTED_V188_REPORT_EXACT_DIGEST,
    required_v188_route: str = EXPECTED_V188_REQUIRED_ROUTE,
    max_dominant_requested_action_share: float = DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE,
    required_target_seed_coverage_fraction: float = (
        DEFAULT_REQUIRED_TARGET_SEED_COVERAGE_FRACTION
    ),
    expected_selected_support: Mapping[str, object] | None = (
        DEFAULT_EXPECTED_SELECTED_SUPPORT
    ),
    v188_report_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    v188_report = (
        dict(v188_report_override)
        if v188_report_override is not None
        else load_json_report(v188_report_path)
    )
    source_validation = validate_v189_v188_source(
        v188_report,
        expected_v188_report_exact_digest=expected_v188_report_exact_digest,
        required_v188_route=required_v188_route,
    )
    historical_dedupe = historical_dedupe_audit()
    support_coverage = support_coverage_audit(
        v188_report,
        required_target_seed_coverage_fraction=required_target_seed_coverage_fraction,
    )
    aggregate_attempts = aggregate_attempted_continuation_audit(v188_report)
    selected_support = selected_support_trajectory_audit(
        v188_report,
        expected_selected_support=expected_selected_support,
    )
    trainable_leakage = trainable_leakage_audit(selected_support)
    route_decision = route_decision_audit(
        source_validation=source_validation,
        historical_dedupe=historical_dedupe,
        support_coverage=support_coverage,
        aggregate_attempts=aggregate_attempts,
        selected_support=selected_support,
        trainable_leakage=trainable_leakage,
        max_dominant_requested_action_share=max_dominant_requested_action_share,
    )
    classification = _classification(
        source_validation=source_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V189_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V189_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_POLICY
        ),
        "contract": _contract(
            expected_v188_report_exact_digest=expected_v188_report_exact_digest,
            required_v188_route=required_v188_route,
            max_dominant_requested_action_share=max_dominant_requested_action_share,
            required_target_seed_coverage_fraction=(
                required_target_seed_coverage_fraction
            ),
        ),
        "inputs": {
            "v188_report": str(v188_report_path),
            "expected_v188_report_exact_digest": expected_v188_report_exact_digest,
            "required_v188_route_path": "route_decision.recommended_next_route",
            "required_v188_route": required_v188_route,
            "v188_backup": EXPECTED_V188_BACKUP,
            "v188_backup_sha256": EXPECTED_V188_BACKUP_SHA256,
            "expected_v187_report_exact_digest": EXPECTED_V187_REPORT_EXACT_DIGEST,
            "output": str(output_path),
        },
        "source_validation": source_validation,
        "historical_dedupe": historical_dedupe,
        "support_coverage_audit": support_coverage,
        "aggregate_attempted_continuation_audit": aggregate_attempts,
        "selected_support_trajectory_audit": selected_support,
        "trainable_leakage_audit": trainable_leakage,
        "dataset_audit": {
            "policy": "m3_carrion_survivor_continuation_v189_dataset_audit_v1",
            "trainable_dataset_created": False,
            "derived_dataset_path": None,
            "selected_support_manifest_audited": True,
            "selected_support_run_count": selected_support.get("support_run_count"),
            "legal_positive_target_seed_count": support_coverage.get(
                "legal_positive_target_seed_count"
            ),
            "target_seed_count": support_coverage.get("target_seed_count"),
            "dataset_ready_for_slice_3_training": route_decision.get(
                "slice_3_training_authorized"
            )
            is True,
            "reason": route_decision.get("primary_blocker"),
        },
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "terminal_survival_support_dataset_audit",
                "historical_dedupe",
                "no_runtime_integration",
                "no_promotion",
            ],
        },
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v189_v188_source(
    v188_report: Mapping[str, object],
    *,
    expected_v188_report_exact_digest: str = EXPECTED_V188_REPORT_EXACT_DIGEST,
    required_v188_route: str = EXPECTED_V188_REQUIRED_ROUTE,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v188_report)
    route = _mapping(v188_report.get("route_decision"))
    source = _mapping(v188_report.get("source_validation"))
    classification = _mapping(v188_report.get("classification"))
    checks = {
        "v188_schema_matches": (
            v188_report.get("schema_version")
            == v188.M3_CARRION_SURVIVOR_CONTINUATION_V188_TERMINAL_CARRION_SURVIVAL_SUPPORT_SCHEMA_VERSION
        ),
        "v188_policy_matches": (
            v188_report.get("policy")
            == v188.M3_CARRION_SURVIVOR_CONTINUATION_V188_TERMINAL_CARRION_SURVIVAL_SUPPORT_POLICY
        ),
        "v188_exact_digest_valid": exact_validation.get("passed") is True,
        "v188_exact_digest_matches_expected": (
            str(v188_report.get("exact_digest") or "")
            == str(expected_v188_report_exact_digest)
        ),
        "v188_route_matches_required": (
            str(route.get("recommended_next_route") or "") == str(required_v188_route)
        ),
        "v188_source_validation_passed": source.get("passed") is True,
        "v188_v187_exact_digest_matches_expected": (
            str(source.get("observed_v187_report_exact_digest") or "")
            == EXPECTED_V187_REPORT_EXACT_DIGEST
        ),
        "v188_positive_support_found": (
            _mapping(v188_report.get("support_result")).get("positive_support_found")
            is True
        ),
        "v188_training_did_not_run": v188_report.get("training_ran") is False,
        "v188_training_artifact_not_created": (
            v188_report.get("training_artifact_created") is False
        ),
        "v188_slice_3_not_consumed": (
            v188_report.get("slice_3_training_consumed") is False
        ),
        "v188_runtime_artifact_not_created": (
            v188_report.get("runtime_artifact_created") is False
        ),
        "v188_runtime_action_selection_unchanged": (
            v188_report.get("runtime_action_selection_changed") is False
        ),
        "v188_promotion_not_authorized": (
            v188_report.get("promotion_authorized") is False
        ),
        "v188_gate_relaxation_not_allowed": (
            v188_report.get("gate_relaxation_allowed") is False
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v189_v188_source_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v188_report_exact_digest": expected_v188_report_exact_digest,
        "observed_v188_report_exact_digest": v188_report.get("exact_digest"),
        "v188_exact_digest_validation": exact_validation,
        "required_v188_route": required_v188_route,
        "observed_v188_route": route.get("recommended_next_route"),
        "observed_v188_classification": classification.get("primary"),
        "observed_v187_report_exact_digest": source.get(
            "observed_v187_report_exact_digest"
        ),
    }


def historical_dedupe_audit() -> dict[str, object]:
    findings = [
        {
            "lane": "v32_mechanical_feasibility",
            "status": "already_proven_do_not_rerun",
            "summary": (
                "Legal policy-visible carrion survival was mechanically proven; "
                "hydration_safe_carrion_cycle kept agents alive on the v32 fixture."
            ),
            "next_use": "treat_as_prior_positive_feasibility_context_only",
        },
        {
            "lane": "v51_multi_seed_scripted_counterfactual_support",
            "status": "already_proven_do_not_rerun",
            "summary": (
                "hydration_safe_carrion_cycle survived all six target fixture "
                "seeds, and v51 recorded 19 survivor runs out of 30 scripts."
            ),
            "next_use": "do_not_reopen_counterfactual_feasibility",
        },
        {
            "lane": "v53_v63_iql_coefficient_prior_action_distribution_loop",
            "status": "closed_non_promotable_do_not_tune_again",
            "summary": (
                "IQL produced nonzero carrion survival in some slices, but "
                "dominant action share and strict per-seed alive/birth regressions "
                "kept scalar coefficient, prior-blend, risk, and global actor-bias "
                "variants non-promotable."
            ),
            "next_use": "do_not_route_to_scalar_iql_or_prior_blend_tuning",
        },
        {
            "lane": "v154_v176_tiny_support_archive_nearest_neighbor_scorer_loop",
            "status": "closed_strategically_saturated_do_not_rerun",
            "summary": (
                "Support existed, but public-feature aliasing, source-split "
                "generalization, and scorer/action-value collapse closed the "
                "tiny archive and nearest-neighbor scorer route."
            ),
            "next_use": "do_not_open_another_micro_archive_scorer_probe",
        },
        {
            "lane": "v181_v182_sparse_imputed_support_autopsy_design",
            "status": "diagnosed_and_hardened_do_not_rediscover",
            "summary": (
                "Sparse imputed transition-value support and broad eat overrides "
                "were already diagnosed, and imputed/low-observed-support "
                "abstention diagnostics were added."
            ),
            "next_use": "treat_as_inherited_support_quality_boundary",
        },
        {
            "lane": "v183_v186_transition_row_support_repair_training",
            "status": "slice_2_spent_do_not_rerun",
            "summary": (
                "v183 expanded exact transition rows, v184/v185 audited and "
                "repaired them, and v186 spent slice 2. The slice failed only on "
                "terminal carrion survival with dominant share under cap and "
                "no broad per-seed alive/birth regressions."
            ),
            "next_use": "do_not_consume_slice_3_without_new_legal_terminal_support",
        },
    ]
    return {
        "policy": "m3_carrion_survivor_continuation_v189_historical_dedupe_v1",
        "passed": True,
        "source_docs": ["AGENTS.md", "docs/mind-v3-autonomous-evolution.md"],
        "finding_count": len(findings),
        "findings": findings,
        "rerun_old_feasibility_counterfactual_iql_or_scorer_loops": False,
        "dedupe_constraint": (
            "v189 must audit v188 terminal-survival support evidence and route "
            "forward; it must not rediscover or rerun these closed lanes."
        ),
    }


def support_coverage_audit(
    v188_report: Mapping[str, object],
    *,
    required_target_seed_coverage_fraction: float,
) -> dict[str, object]:
    support = _mapping(v188_report.get("support_result"))
    terminal_by_seed = {
        str(seed): _int(count)
        for seed, count in _mapping(support.get("terminal_survivors_by_seed")).items()
    }
    attempted_by_seed = {
        str(seed): _int(count)
        for seed, count in _mapping(
            support.get("attempted_terminal_survivors_by_seed")
        ).items()
    }
    target_seed_count = _int(support.get("target_seed_count"))
    positive_count = sum(1 for count in terminal_by_seed.values() if count > 0)
    required_count = (
        target_seed_count
        if required_target_seed_coverage_fraction >= 1.0
        else int(target_seed_count * required_target_seed_coverage_fraction)
    )
    coverage_fraction = (
        positive_count / target_seed_count if target_seed_count > 0 else 0.0
    )
    all_target_seeds = (
        target_seed_count > 0
        and positive_count >= target_seed_count
        and support.get("all_target_seeds_have_positive_support") is True
    )
    failures = []
    if positive_count < required_count:
        failures.append("legal_terminal_support_seed_coverage_below_required")
    if not all_target_seeds:
        failures.append("not_all_target_seeds_have_legal_positive_support")
    return {
        "policy": "m3_carrion_survivor_continuation_v189_support_coverage_audit_v1",
        "passed": not failures,
        "failures": failures,
        "target_seed_count": target_seed_count,
        "legal_positive_target_seed_count": positive_count,
        "required_legal_positive_target_seed_count": required_count,
        "legal_support_coverage_fraction": _round(coverage_fraction),
        "required_target_seed_coverage_fraction": _round(
            required_target_seed_coverage_fraction
        ),
        "terminal_survivors_by_seed": terminal_by_seed,
        "attempted_terminal_survivors_by_seed": attempted_by_seed,
        "all_target_seeds_have_positive_support": all_target_seeds,
        "support_run_count": len(_mappings(support.get("support_runs"))),
    }


def aggregate_attempted_continuation_audit(
    v188_report: Mapping[str, object],
) -> dict[str, object]:
    support_search = _mapping(v188_report.get("support_search"))
    support = _mapping(v188_report.get("support_result"))
    unsupported_requested = _int(
        support_search.get("unsupported_requested_action_count")
    )
    unsupported_resolved = _int(support_search.get("unsupported_resolved_action_count"))
    rejected = unsupported_requested > 0 or unsupported_resolved > 0
    return {
        "policy": "m3_carrion_survivor_continuation_v189_aggregate_attempt_audit_v1",
        "passed": True,
        "aggregate_attempted_continuations_are_training_support": False,
        "aggregate_attempted_continuations_rejected_as_support": rejected,
        "rejection_reason": (
            "aggregate_has_unsupported_actions" if rejected else None
        ),
        "unsupported_requested_action_count": unsupported_requested,
        "unsupported_resolved_action_count": unsupported_resolved,
        "attempted_successful_branch_run_count": _int(
            support.get("attempted_successful_branch_run_count")
        ),
        "attempted_positive_seed_count": _int(
            support.get("attempted_positive_seed_count")
        ),
        "attempted_terminal_survivors_by_seed": _mapping(
            support.get("attempted_terminal_survivors_by_seed")
        ),
        "diagnostic_only": True,
    }


def selected_support_trajectory_audit(
    v188_report: Mapping[str, object],
    *,
    expected_selected_support: Mapping[str, object] | None,
) -> dict[str, object]:
    support = _mapping(v188_report.get("support_result"))
    support_runs = _mappings(support.get("support_runs"))
    manifest_rows = _mappings(support.get("support_trajectory_manifest"))
    run_audits = [_support_run_audit(run, manifest_rows) for run in support_runs]
    expected_check = _expected_selected_support_check(
        support_runs,
        expected_selected_support=expected_selected_support,
    )
    failures: list[str] = []
    if not support_runs:
        failures.append("selected_support_runs_empty")
    if len(manifest_rows) != len(support_runs):
        failures.append("manifest_count_does_not_match_support_runs")
    if len({str(run.get("trajectory_path") or "") for run in support_runs}) != len(
        support_runs
    ):
        failures.append("support_trajectory_paths_not_unique")
    for audit in run_audits:
        if audit.get("passed") is not True:
            failures.append("support_run_audit_failed")
            break
    if expected_check.get("passed") is not True:
        failures.append("expected_selected_support_mismatch")
    total_records = sum(_int(audit.get("record_count")) for audit in run_audits)
    dominant = _dominant_action_share_from_audits(run_audits)
    return {
        "policy": "m3_carrion_survivor_continuation_v189_selected_support_trajectory_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": sorted(set(failures)),
        "support_run_count": len(support_runs),
        "manifest_count": len(manifest_rows),
        "record_count": total_records,
        "run_audits": run_audits,
        "expected_selected_support_check": expected_check,
        "dominant_requested_action": dominant.get("key"),
        "dominant_requested_action_count": dominant.get("count"),
        "dominant_requested_action_share": dominant.get("share"),
        "replay_validity_passed": all(
            audit.get("replay_verified_by_report") is True for audit in run_audits
        ),
        "action_mask_legality_passed": all(
            audit.get("action_mask_legality_passed") is True for audit in run_audits
        ),
        "terminal_facts_match_manifest": all(
            audit.get("terminal_facts_match_manifest") is True for audit in run_audits
        ),
        "path_backed_availability_passed": all(
            audit.get("path_exists") is True for audit in run_audits
        ),
    }


def trainable_leakage_audit(
    selected_support_audit: Mapping[str, object],
) -> dict[str, object]:
    run_audits = _mappings(selected_support_audit.get("run_audits"))
    failures: list[dict[str, object]] = []
    for run_index, audit in enumerate(run_audits):
        for failure in _mappings(audit.get("trainable_leakage_failures")):
            failures.append({"run_index": run_index, **dict(failure)})
    return {
        "policy": "m3_carrion_survivor_continuation_v189_trainable_leakage_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
        "forbidden_trainable_tokens": list(FORBIDDEN_TRAINABLE_TOKENS),
        "candidate_trainable_payload_policy": (
            "current_public_observation_and_current_public_action_mask_only"
        ),
        "metadata_only_fields": [
            "seed",
            "fixture",
            "branch_id",
            "branch_tick",
            "trajectory_path",
            "replay_digest",
            "logical_replay_digest",
            "provenance",
            "terminal_alive_agents",
            "births",
            "future_outcome",
        ],
        "trainable_seed_fixture_branch_path_digest_provenance_private_future_leakage": (
            bool(failures)
        ),
        "trainable_dataset_created": False,
    }


def route_decision_audit(
    *,
    source_validation: Mapping[str, object],
    historical_dedupe: Mapping[str, object],
    support_coverage: Mapping[str, object],
    aggregate_attempts: Mapping[str, object],
    selected_support: Mapping[str, object],
    trainable_leakage: Mapping[str, object],
    max_dominant_requested_action_share: float,
) -> dict[str, object]:
    blockers: list[str] = []
    if source_validation.get("passed") is not True:
        blockers.append("v188_source_validation_failed")
    if historical_dedupe.get("passed") is not True:
        blockers.append("historical_dedupe_failed")
    if selected_support.get("passed") is not True:
        blockers.append("selected_support_trajectory_audit_failed")
    if trainable_leakage.get("passed") is not True:
        blockers.append("trainable_leakage_audit_failed")
    if support_coverage.get("passed") is not True:
        blockers.append(
            "legal_terminal_support_target_seed_coverage_insufficient"
        )
    dominant_share = _float(selected_support.get("dominant_requested_action_share"))
    if dominant_share > float(max_dominant_requested_action_share):
        blockers.append("selected_support_dominant_requested_action_share_above_cap")
    if (
        aggregate_attempts.get("aggregate_attempted_continuations_rejected_as_support")
        is True
    ):
        blockers.append("aggregate_attempted_continuations_rejected_as_support")
    training_authorized = not blockers
    if source_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif training_authorized:
        route = SLICE_3_TRAINING_ROUTE
    else:
        route = TARGETED_SUPPORT_EXPANSION_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v189_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "slice_3_training_authorized": training_authorized,
        "slice_3_training_allowed_for_this_command": False,
        "slice_3_training_consumed": False,
        "runtime_integration_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "training_authorization_scope": (
            "future_explicit_slice_3_route" if training_authorized else "closed"
        ),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "primary_blocker": blockers[0] if blockers else None,
        "legal_positive_target_seed_count": support_coverage.get(
            "legal_positive_target_seed_count"
        ),
        "target_seed_count": support_coverage.get("target_seed_count"),
        "dominant_requested_action": selected_support.get(
            "dominant_requested_action"
        ),
        "dominant_requested_action_share": selected_support.get(
            "dominant_requested_action_share"
        ),
        "max_dominant_requested_action_share": _round(
            max_dominant_requested_action_share
        ),
        "aggregate_unsupported_resolved_action_count": aggregate_attempts.get(
            "unsupported_resolved_action_count"
        ),
        "rationale": _route_rationale(blockers=blockers, route=route),
    }


def _support_run_audit(
    run: Mapping[str, object],
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    path = Path(str(run.get("trajectory_path") or ""))
    matching_manifest = [
        row for row in manifest_rows if str(row.get("path") or "") == str(path)
    ]
    trajectory = _audit_trajectory_path(path, run)
    replay = _mapping(run.get("replay_verification"))
    checks = {
        "path_exists": path.exists(),
        "manifest_row_present": len(matching_manifest) == 1,
        "replay_verified_by_report": (
            run.get("replay_verified") is True
            or replay.get("verified") is True
        ),
        "unsupported_requested_action_count_zero": (
            _int(run.get("unsupported_requested_action_count")) == 0
        ),
        "unsupported_resolved_action_count_zero": (
            _int(run.get("unsupported_resolved_action_count")) == 0
        ),
        "heuristic_action_source_count_zero": (
            _int(run.get("heuristic_action_source_count")) == 0
        ),
        "trajectory_readable": trajectory.get("readable") is True,
        "terminal_facts_match_manifest": (
            trajectory.get("terminal_facts_match_manifest") is True
        ),
        "action_mask_legality_passed": (
            trajectory.get("action_mask_legality_passed") is True
        ),
        "trajectory_invalid_counts_zero": (
            trajectory.get("trajectory_invalid_counts_zero") is True
        ),
        "trainable_leakage_scan_passed": (
            trajectory.get("trainable_leakage_scan_passed") is True
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v189_support_run_audit_v1",
        "passed": not failures,
        "failures": failures,
        "branch_id": run.get("branch_id"),
        "seed": _int(run.get("seed")),
        "fixture": run.get("fixture"),
        "ticks": _int(run.get("ticks")),
        "branch_tick": _int(run.get("branch_tick")),
        "base_script": run.get("base_script"),
        "continuation_script": run.get("continuation_script"),
        "trajectory_path": str(path),
        "path_exists": path.exists(),
        "alive_agents": _int(run.get("alive_agents")),
        "births": _int(run.get("births")),
        "deaths": _int(run.get("deaths")),
        "dominant_requested_action": run.get("dominant_requested_action"),
        "dominant_requested_action_share": run.get(
            "dominant_requested_action_share"
        ),
        "replay_verified_by_report": checks["replay_verified_by_report"],
        "manifest_row": dict(matching_manifest[0]) if matching_manifest else None,
        **trajectory,
    }


def _audit_trajectory_path(
    path: Path,
    run: Mapping[str, object],
) -> dict[str, object]:
    if not path.exists():
        return _unreadable_trajectory_audit("missing_path")
    try:
        rows = list(_iter_jsonl(path))
    except (OSError, EOFError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        payload = _unreadable_trajectory_audit(type(exc).__name__)
        payload["error"] = str(exc)
        return payload
    header = next((row for row in rows if row.get("type") == "header"), None)
    footer = next((row for row in rows if row.get("type") == "footer"), None)
    records = [
        _mapping(row.get("record"))
        for row in rows
        if row.get("type") == "record" and isinstance(row.get("record"), Mapping)
    ]
    legality_failures: list[dict[str, object]] = []
    leakage_failures: list[dict[str, object]] = []
    requested_counts: Counter[str] = Counter()
    action_source_counts: Counter[str] = Counter()
    for record_index, record in enumerate(records):
        requested = str(record.get("requested_action") or "")
        resolved = str(record.get("resolved_action") or requested)
        requested_counts.update([requested])
        action_source_counts.update([str(record.get("action_source") or "")])
        action_mask = _mapping(record.get("action_mask"))
        resolution_mask = _mapping(record.get("resolution_action_mask")) or action_mask
        if record.get("action_valid") is not True:
            legality_failures.append(
                {
                    "record_index": record_index,
                    "reason": "action_valid_false",
                    "requested_action": requested,
                    "resolved_action": resolved,
                }
            )
        if record.get("resolution_action_valid") is False:
            legality_failures.append(
                {
                    "record_index": record_index,
                    "reason": "resolution_action_valid_false",
                    "requested_action": requested,
                    "resolved_action": resolved,
                }
            )
        if requested and action_mask.get(requested) is not True:
            legality_failures.append(
                {
                    "record_index": record_index,
                    "reason": "requested_action_not_in_public_action_mask",
                    "requested_action": requested,
                }
            )
        if resolved and resolution_mask.get(resolved) is not True:
            legality_failures.append(
                {
                    "record_index": record_index,
                    "reason": "resolved_action_not_in_resolution_action_mask",
                    "resolved_action": resolved,
                }
            )
        candidate_payload = {
            "current_public_observation": record.get("observation_input"),
            "current_public_action_mask": record.get("action_mask"),
        }
        _scan_trainable_key_leakage(
            candidate_payload,
            failures=leakage_failures,
            path=("candidate_trainable_payload",),
            record_index=record_index,
        )
    footer_summary = _mapping(footer.get("summary") if footer else None)
    trajectory_summary = _mapping(footer.get("trajectory_summary") if footer else None)
    invalid_counts_zero = (
        _int(trajectory_summary.get("invalid_action_count")) == 0
        and _int(trajectory_summary.get("invalid_observation_action_count")) == 0
        and _int(trajectory_summary.get("invalid_resolution_action_count")) == 0
    )
    terminal_matches = (
        footer is not None
        and _int(footer_summary.get("alive_agents")) == _int(run.get("alive_agents"))
        and _int(footer_summary.get("births")) == _int(run.get("births"))
        and _int(footer_summary.get("deaths")) == _int(run.get("deaths"))
    )
    dominant = _dominant_count_share(requested_counts)
    return {
        "readable": True,
        "header_present": header is not None,
        "footer_present": footer is not None,
        "record_count": len(records),
        "footer_record_count": _int(trajectory_summary.get("record_count")),
        "record_count_matches_footer": (
            _int(trajectory_summary.get("record_count")) == len(records)
        ),
        "footer_alive_agents": _int(footer_summary.get("alive_agents")),
        "footer_births": _int(footer_summary.get("births")),
        "footer_deaths": _int(footer_summary.get("deaths")),
        "terminal_facts_match_manifest": terminal_matches,
        "trajectory_invalid_counts_zero": invalid_counts_zero,
        "action_mask_legality_passed": not legality_failures,
        "action_mask_legality_failure_count": len(legality_failures),
        "action_mask_legality_failures": legality_failures[:32],
        "requested_action_counts": dict(sorted(requested_counts.items())),
        "dominant_requested_action_from_trajectory": dominant.get("key"),
        "dominant_requested_action_share_from_trajectory": dominant.get("share"),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "trainable_leakage_scan_passed": not leakage_failures,
        "trainable_leakage_failure_count": len(leakage_failures),
        "trainable_leakage_failures": leakage_failures[:32],
    }


def _unreadable_trajectory_audit(reason: str) -> dict[str, object]:
    return {
        "readable": False,
        "reason": reason,
        "header_present": False,
        "footer_present": False,
        "record_count": 0,
        "footer_record_count": None,
        "record_count_matches_footer": False,
        "footer_alive_agents": None,
        "footer_births": None,
        "footer_deaths": None,
        "terminal_facts_match_manifest": False,
        "trajectory_invalid_counts_zero": False,
        "action_mask_legality_passed": False,
        "action_mask_legality_failure_count": None,
        "action_mask_legality_failures": [],
        "requested_action_counts": {},
        "dominant_requested_action_from_trajectory": None,
        "dominant_requested_action_share_from_trajectory": None,
        "action_source_counts": {},
        "trainable_leakage_scan_passed": False,
        "trainable_leakage_failure_count": None,
        "trainable_leakage_failures": [],
    }


def _expected_selected_support_check(
    support_runs: Sequence[Mapping[str, object]],
    *,
    expected_selected_support: Mapping[str, object] | None,
) -> dict[str, object]:
    if expected_selected_support is None:
        return {
            "policy": "m3_carrion_survivor_continuation_v189_expected_support_check_v1",
            "passed": True,
            "skipped": True,
            "reason": "no_expected_selected_support_supplied",
        }
    expected = dict(expected_selected_support)
    if len(support_runs) != 1:
        return {
            "policy": "m3_carrion_survivor_continuation_v189_expected_support_check_v1",
            "passed": False,
            "skipped": False,
            "failures": ["selected_support_run_count_not_one"],
            "expected": expected,
            "observed_support_run_count": len(support_runs),
        }
    run = support_runs[0]
    comparisons = {
        key: _expected_value_matches(run.get(key), expected_value)
        for key, expected_value in expected.items()
    }
    failures = [key for key, passed in comparisons.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v189_expected_support_check_v1",
        "passed": not failures,
        "skipped": False,
        "failures": failures,
        "expected": expected,
        "observed": {key: run.get(key) for key in expected},
    }


def _expected_value_matches(observed: object, expected: object) -> bool:
    if isinstance(expected, float):
        return abs(_float(observed) - expected) < 1e-9
    if isinstance(expected, int):
        return _int(observed, default=-999999) == expected
    return str(observed) == str(expected)


def _scan_trainable_key_leakage(
    value: object,
    *,
    failures: list[dict[str, object]],
    path: tuple[str, ...],
    record_index: int,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            lower = key_text.lower()
            for token in FORBIDDEN_TRAINABLE_TOKENS:
                if token in lower:
                    failures.append(
                        {
                            "record_index": record_index,
                            "path": ".".join((*path, key_text)),
                            "reason": "forbidden_trainable_key_token",
                            "token": token,
                        }
                    )
                    break
            _scan_trainable_key_leakage(
                child,
                failures=failures,
                path=(*path, key_text),
                record_index=record_index,
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _scan_trainable_key_leakage(
                child,
                failures=failures,
                path=(*path, str(index)),
                record_index=record_index,
            )


def _iter_jsonl(path: Path) -> Sequence[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _dominant_action_share_from_audits(
    run_audits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    for audit in run_audits:
        counts.update(
            {
                str(action): _int(count)
                for action, count in _mapping(
                    audit.get("requested_action_counts")
                ).items()
            }
        )
    return _dominant_count_share(counts)


def _dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return {"key": key, "count": int(count), "share": _round(count / total)}


def _mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _route_rationale(*, blockers: Sequence[str], route: str) -> str:
    if route == STOP_ROUTE:
        return "Pinned v188 source validation failed; stop without new work."
    if not blockers:
        return (
            "The v188 selected support evidence is path-backed, legal, diverse, "
            "leakage-clean, and covers every target seed, so a future explicit "
            "slice-3 route may be authorized. This command still does not train."
        )
    return (
        "v188 proves one legal terminal-survival continuation, but current clean "
        "support is not sufficient for slice-3 training. Route to targeted legal "
        "terminal-survival support expansion without training."
    )


def _classification(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if route_decision.get("slice_3_training_authorized") is True:
        return prefix + "support_ready_for_future_explicit_slice_3_training"
    return prefix + "support_insufficient_routes_to_targeted_legal_expansion_no_training"


def _contract(
    *,
    expected_v188_report_exact_digest: str,
    required_v188_route: str,
    max_dominant_requested_action_share: float,
    required_target_seed_coverage_fraction: float,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "support_expansion_ran": False,
        "rerun_v32_v51_v53_v63_v154_v176_v181_v186_allowed": False,
        "input_v188_report_exact_digest_pinned": expected_v188_report_exact_digest,
        "required_v188_route": required_v188_route,
        "max_dominant_requested_action_share": _round(
            max_dominant_requested_action_share
        ),
        "required_target_seed_coverage_fraction": _round(
            required_target_seed_coverage_fraction
        ),
        "expected_blocked_route_when_support_insufficient": (
            TARGETED_SUPPORT_EXPANSION_ROUTE
        ),
    }


def _lifecycle_flags() -> dict[str, object]:
    return {
        "training_ran": False,
        "training_artifact_created": False,
        "fit_ran": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": False,
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "gate_relaxation_ran": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "generic_carrion_autopsy_rerun": False,
        "v180_rerun": False,
        "v186_rerun": False,
        "v188_rerun": False,
        "non_promoted": True,
        "diagnostics_only": True,
    }


def _digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)
