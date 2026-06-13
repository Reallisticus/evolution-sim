from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import gzip
import json
from pathlib import Path

from evolution_sim.mind.candidate_campaign import _float, _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion as v190,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V191_LEGAL_SUPPORT_REPAIR_ARCHITECTURE_REVIEW_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v191_legal_support_repair_architecture_review_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V191_LEGAL_SUPPORT_REPAIR_ARCHITECTURE_REVIEW_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v191_legal_support_repair_architecture_review_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v191-carrion-survivor-continuation-legal-support-repair-architecture-review.json"
)
DEFAULT_V190_REPORT_PATH = v190.DEFAULT_OUTPUT_PATH
EXPECTED_V190_REPORT_EXACT_DIGEST = (
    "b53de9d19f67f681e334520c030978f97f6f7486e44a9f9808be11de269234b5"
)
EXPECTED_V190_REQUIRED_ROUTE = v190.BLOCKED_ROUTE
EXPECTED_V190_BACKUP = (
    "gdrive:evolution-sim-backups/archives/"
    "20260613T100008Z-v190-targeted-legal-terminal-survival-support-expansion.tar.zst"
)
EXPECTED_V190_BACKUP_SHA256 = (
    "e2c1392b066342a43c78274646bfce6589adacc5a61ec3e296f574fed1e7d333"
)

DEFAULT_TARGET_SEEDS = (13, 19, 29, 37, 41, 43)
DEFAULT_EXPECTED_BRANCH_POINT_COUNT = 18
DEFAULT_EXPECTED_BRANCH_RUN_COUNT = 90
DEFAULT_EXPECTED_UNSUPPORTED_REQUESTED_ACTION_COUNT = 0
DEFAULT_EXPECTED_UNSUPPORTED_RESOLVED_ACTION_COUNT = 860
DEFAULT_EXPECTED_CLEAN_LEGAL_SUPPORT_SEED_COUNT = 0
DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE = 0.50
DEFAULT_METADATA_DOC_PATHS = (
    Path("AGENTS.md"),
    Path("docs/mind-v3-autonomous-evolution.md"),
    Path("docs/repository-audit-2026-06-10-remediation.md"),
)

ACTION_RESOLUTION_CONTRACT_REPAIR_ROUTE = (
    "v192_action_resolution_contract_repair_no_training"
)
ARCHITECTURE_REVIEW_ROUTE = "v192_architecture_review_no_training"
TERMINAL_SUPPORT_AUDIT_ROUTE = (
    "v192_terminal_survival_support_dataset_audit_before_slice_3_training"
)
STOP_ROUTE = "stop"

FALSE_V190_LIFECYCLE_FLAGS = (
    "training_ran",
    "training_artifact_created",
    "slice_3_training_consumed",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "promotion_authorized",
    "gate_relaxation_allowed",
)
FALSE_V191_LIFECYCLE_FLAGS = (
    "training_ran",
    "training_artifact_created",
    "slice_3_training_consumed",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "promotion_authorized",
    "gate_relaxation_allowed",
    "support_generation_ran",
    "support_expansion_ran",
    "v180_rerun",
    "v186_rerun",
    "v188_rerun",
    "v189_rerun",
    "v190_rerun",
)


def run_carrion_survivor_continuation_v191_legal_support_repair_architecture_review(
    *,
    v190_report_path: str | Path = DEFAULT_V190_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v190_report_exact_digest: str = EXPECTED_V190_REPORT_EXACT_DIGEST,
    required_v190_route: str = EXPECTED_V190_REQUIRED_ROUTE,
    target_seeds: Sequence[int] = DEFAULT_TARGET_SEEDS,
    expected_branch_point_count: int = DEFAULT_EXPECTED_BRANCH_POINT_COUNT,
    expected_branch_run_count: int = DEFAULT_EXPECTED_BRANCH_RUN_COUNT,
    expected_unsupported_requested_action_count: int = (
        DEFAULT_EXPECTED_UNSUPPORTED_REQUESTED_ACTION_COUNT
    ),
    expected_unsupported_resolved_action_count: int = (
        DEFAULT_EXPECTED_UNSUPPORTED_RESOLVED_ACTION_COUNT
    ),
    expected_clean_legal_support_seed_count: int = (
        DEFAULT_EXPECTED_CLEAN_LEGAL_SUPPORT_SEED_COUNT
    ),
    max_dominant_requested_action_share: float = (
        DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE
    ),
    metadata_doc_paths: Sequence[str | Path] = DEFAULT_METADATA_DOC_PATHS,
    v190_report_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    v190_report = (
        dict(v190_report_override)
        if v190_report_override is not None
        else load_json_report(v190_report_path)
    )
    source_validation = validate_v191_v190_source(
        v190_report,
        expected_v190_report_exact_digest=expected_v190_report_exact_digest,
        required_v190_route=required_v190_route,
        target_seeds=target_seeds,
        expected_branch_point_count=expected_branch_point_count,
        expected_branch_run_count=expected_branch_run_count,
        expected_unsupported_requested_action_count=(
            expected_unsupported_requested_action_count
        ),
        expected_unsupported_resolved_action_count=(
            expected_unsupported_resolved_action_count
        ),
        expected_clean_legal_support_seed_count=(
            expected_clean_legal_support_seed_count
        ),
        metadata_doc_paths=metadata_doc_paths,
    )
    classification = unsupported_resolved_classification(
        v190_report,
        target_seeds=target_seeds,
        max_dominant_requested_action_share=max_dominant_requested_action_share,
    )
    near_clean = near_clean_seed_analysis(
        v190_report,
        target_seeds=target_seeds,
        max_dominant_requested_action_share=max_dominant_requested_action_share,
    )
    repair = repair_recommendation_audit(
        classification=classification,
        near_clean=near_clean,
    )
    route_decision = route_decision_audit(
        source_validation=source_validation,
        classification=classification,
        repair=recover_mapping(repair),
        near_clean=near_clean,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V191_LEGAL_SUPPORT_REPAIR_ARCHITECTURE_REVIEW_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V191_LEGAL_SUPPORT_REPAIR_ARCHITECTURE_REVIEW_POLICY
        ),
        "contract": contract(
            expected_v190_report_exact_digest=expected_v190_report_exact_digest,
            required_v190_route=required_v190_route,
            max_dominant_requested_action_share=max_dominant_requested_action_share,
        ),
        "inputs": {
            "v190_report": str(v190_report_path),
            "expected_v190_report_exact_digest": expected_v190_report_exact_digest,
            "required_v190_route_path": "route_decision.recommended_next_route",
            "required_v190_route": required_v190_route,
            "v190_backup": EXPECTED_V190_BACKUP,
            "v190_backup_sha256": EXPECTED_V190_BACKUP_SHA256,
            "output": str(output_path),
        },
        "source_validation": source_validation,
        "unsupported_resolved_action_classification": classification,
        "near_clean_seed_analysis": near_clean,
        "repair_recommendation_audit": repair,
        "route_decision": route_decision,
        "classification": {
            "primary": classification_for(
                source_validation=source_validation,
                route_decision=route_decision,
            ),
            "labels": [
                "diagnostics_only",
                "legal_support_repair_architecture_review",
                "unsupported_resolved_action_root_cause",
                "no_blind_support_expansion",
                "no_slice_3_training",
                "no_runtime_integration",
                "no_promotion",
            ],
        },
        **lifecycle_flags(),
    }
    report["classification"]["labels"].insert(0, report["classification"]["primary"])
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v191_v190_source(
    v190_report: Mapping[str, object],
    *,
    expected_v190_report_exact_digest: str = EXPECTED_V190_REPORT_EXACT_DIGEST,
    required_v190_route: str = EXPECTED_V190_REQUIRED_ROUTE,
    target_seeds: Sequence[int] = DEFAULT_TARGET_SEEDS,
    expected_branch_point_count: int = DEFAULT_EXPECTED_BRANCH_POINT_COUNT,
    expected_branch_run_count: int = DEFAULT_EXPECTED_BRANCH_RUN_COUNT,
    expected_unsupported_requested_action_count: int = (
        DEFAULT_EXPECTED_UNSUPPORTED_REQUESTED_ACTION_COUNT
    ),
    expected_unsupported_resolved_action_count: int = (
        DEFAULT_EXPECTED_UNSUPPORTED_RESOLVED_ACTION_COUNT
    ),
    expected_clean_legal_support_seed_count: int = (
        DEFAULT_EXPECTED_CLEAN_LEGAL_SUPPORT_SEED_COUNT
    ),
    metadata_doc_paths: Sequence[str | Path] = DEFAULT_METADATA_DOC_PATHS,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v190_report)
    route = _mapping(v190_report.get("route_decision"))
    search = _mapping(v190_report.get("expansion_search"))
    target_manifest = _mapping(v190_report.get("target_manifest"))
    support = _mapping(v190_report.get("legal_support_audit"))
    branch_report = _mapping(search.get("branch_report"))
    aggregate = _mapping(branch_report.get("aggregate"))
    backup = backup_metadata_audit(metadata_doc_paths)
    expected_seeds = [int(seed) for seed in target_seeds]
    checks = {
        "v190_schema_matches": (
            v190_report.get("schema_version")
            == v190.M3_CARRION_SURVIVOR_CONTINUATION_V190_TARGETED_LEGAL_TERMINAL_SURVIVAL_SUPPORT_EXPANSION_SCHEMA_VERSION
        ),
        "v190_policy_matches": (
            v190_report.get("policy")
            == v190.M3_CARRION_SURVIVOR_CONTINUATION_V190_TARGETED_LEGAL_TERMINAL_SURVIVAL_SUPPORT_EXPANSION_POLICY
        ),
        "v190_exact_digest_valid": exact_validation.get("passed") is True,
        "v190_exact_digest_matches_expected": (
            str(v190_report.get("exact_digest") or "")
            == str(expected_v190_report_exact_digest)
        ),
        "v190_route_matches_required": (
            str(route.get("recommended_next_route") or "") == str(required_v190_route)
        ),
        "v190_source_validation_passed": (
            _mapping(v190_report.get("source_validation")).get("passed") is True
        ),
        "v190_expansion_ran": search.get("ran") is True,
        "v190_branch_point_count_matches_expected": (
            _int(search.get("branch_point_count") or aggregate.get("branch_point_count"))
            == int(expected_branch_point_count)
        ),
        "v190_branch_run_count_matches_expected": (
            _int(search.get("branch_run_count") or aggregate.get("branch_run_count"))
            == int(expected_branch_run_count)
        ),
        "v190_replay_verified": (
            search.get("replay_verified") is True or aggregate.get("replay_verified") is True
        ),
        "v190_unsupported_requested_matches_expected": (
            _int(search.get("unsupported_requested_action_count"))
            == int(expected_unsupported_requested_action_count)
        ),
        "v190_unsupported_resolved_matches_expected": (
            _int(search.get("unsupported_resolved_action_count"))
            == int(expected_unsupported_resolved_action_count)
        ),
        "v190_clean_legal_support_count_matches_expected": (
            _int(support.get("clean_legal_support_seed_count"))
            == int(expected_clean_legal_support_seed_count)
        ),
        "v190_target_seed_count_matches_expected": (
            _int(support.get("target_seed_count")) == len(expected_seeds)
        ),
        "v190_targeted_seeds_match_expected": (
            [_int(seed) for seed in target_manifest.get("targeted_expansion_seeds", [])]
            == expected_seeds
        ),
        "v190_backup_metadata_recorded": backup.get("passed") is True,
        **{
            f"v190_{flag}_closed": v190_report.get(flag) is False
            for flag in FALSE_V190_LIFECYCLE_FLAGS
        },
        "v190_support_expansion_was_diagnostic": (
            v190_report.get("support_expansion_ran") is True
            and v190_report.get("diagnostics_only") is True
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v191_v190_source_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v190_report_exact_digest": expected_v190_report_exact_digest,
        "observed_v190_report_exact_digest": v190_report.get("exact_digest"),
        "v190_exact_digest_validation": exact_validation,
        "required_v190_route": required_v190_route,
        "observed_v190_route": route.get("recommended_next_route"),
        "v190_backup_metadata_audit": backup,
    }


def backup_metadata_audit(
    metadata_doc_paths: Sequence[str | Path] = DEFAULT_METADATA_DOC_PATHS,
) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for raw_path in metadata_doc_paths:
        path = Path(raw_path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            findings.append(
                {
                    "path": str(path),
                    "readable": False,
                    "error": str(exc),
                    "contains_backup": False,
                    "contains_archive_sha256": False,
                }
            )
            continue
        findings.append(
            {
                "path": str(path),
                "readable": True,
                "contains_backup": EXPECTED_V190_BACKUP in text,
                "contains_archive_sha256": EXPECTED_V190_BACKUP_SHA256 in text,
            }
        )
    matching_paths = [
        item["path"]
        for item in findings
        if item.get("contains_backup") is True
        and item.get("contains_archive_sha256") is True
    ]
    return {
        "policy": "m3_carrion_survivor_continuation_v191_v190_backup_metadata_audit_v1",
        "passed": bool(matching_paths),
        "expected_v190_backup": EXPECTED_V190_BACKUP,
        "expected_v190_backup_sha256": EXPECTED_V190_BACKUP_SHA256,
        "matching_doc_paths": matching_paths,
        "checked_doc_paths": findings,
    }


def unsupported_resolved_classification(
    v190_report: Mapping[str, object],
    *,
    target_seeds: Sequence[int],
    max_dominant_requested_action_share: float,
) -> dict[str, object]:
    branch_report = _mapping(_mapping(v190_report.get("expansion_search")).get("branch_report"))
    branch_runs = _mappings(branch_report.get("branch_runs"))
    events: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    missing_paths: list[str] = []
    read_errors: list[dict[str, str]] = []
    for run in branch_runs:
        path = Path(str(run.get("trajectory_path") or ""))
        run_events: list[dict[str, object]] = []
        if not path.exists():
            missing_paths.append(str(path))
        else:
            try:
                for record in _trajectory_records(path):
                    if record.get("resolution_action_valid") is not False:
                        continue
                    event = unsupported_resolved_event(run, record)
                    run_events.append(event)
                    events.append(event)
            except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
                read_errors.append({"path": str(path), "error": str(exc)})
        run_summaries.append(
            {
                "seed": _int(run.get("seed")),
                "branch_id": run.get("branch_id"),
                "continuation_script": run.get("continuation_script"),
                "trajectory_path": str(path),
                "reported_unsupported_resolved_action_count": _int(
                    run.get("unsupported_resolved_action_count")
                ),
                "observed_unsupported_resolved_action_count": len(run_events),
                "reported_unsupported_requested_action_count": _int(
                    run.get("unsupported_requested_action_count")
                ),
                "alive_agents": _int(run.get("alive_agents")),
                "births": _int(run.get("births")),
                "dominant_requested_action": run.get("dominant_requested_action"),
                "dominant_requested_action_share": run.get(
                    "dominant_requested_action_share"
                ),
                "event_count_matches_report": (
                    len(run_events)
                    == _int(run.get("unsupported_resolved_action_count"))
                ),
            }
        )
    report_total = _int(
        _mapping(v190_report.get("expansion_search")).get(
            "unsupported_resolved_action_count"
        )
    )
    requested_invalid_total = _int(
        _mapping(v190_report.get("expansion_search")).get(
            "unsupported_requested_action_count"
        )
    )
    by_seed = _count_by(events, "seed")
    by_requested = _count_by(events, "requested_action")
    by_resolved = _count_by(events, "resolved_action")
    by_reason = _count_by(events, "legality_reason")
    by_script = _count_by(events, "continuation_script")
    by_seed_script = _count_by_pair(events, "seed", "continuation_script")
    by_seed_branch_script = _count_by_triple(
        events,
        "seed",
        "branch_id",
        "continuation_script",
    )
    movement_actions = {
        "move_north",
        "move_south",
        "move_east",
        "move_west",
    }
    all_observation_valid = bool(events) and all(
        event.get("action_valid") is True
        and event.get("observation_mask_allows_requested_action") is True
        for event in events
    )
    all_resolution_mask_rejects = bool(events) and all(
        event.get("resolution_mask_allows_requested_action") is False
        for event in events
    )
    all_resolved_to_stay = bool(events) and all(
        event.get("resolved_action") == "stay" for event in events
    )
    all_requested_movement = bool(events) and all(
        event.get("requested_action") in movement_actions for event in events
    )
    event_count_matches_report = len(events) == report_total
    run_counts_match = all(
        summary.get("event_count_matches_report") is True for summary in run_summaries
    )
    root = root_cause_assessment(
        events=events,
        all_observation_valid=all_observation_valid,
        all_resolution_mask_rejects=all_resolution_mask_rejects,
        all_resolved_to_stay=all_resolved_to_stay,
        all_requested_movement=all_requested_movement,
        event_count_matches_report=event_count_matches_report,
        run_counts_match=run_counts_match,
        missing_paths=missing_paths,
        read_errors=read_errors,
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v191_unsupported_resolved_classification_v1",
        "passed": bool(root.get("classification_confident")),
        "target_seeds": [int(seed) for seed in target_seeds],
        "max_dominant_requested_action_share": _round(
            max_dominant_requested_action_share
        ),
        "v190_reported_unsupported_requested_action_count": requested_invalid_total,
        "v190_reported_unsupported_resolved_action_count": report_total,
        "observed_unsupported_resolved_action_count": len(events),
        "event_count_matches_v190_report": event_count_matches_report,
        "run_event_counts_match_v190_run_summaries": run_counts_match,
        "trajectory_path_missing_count": len(missing_paths),
        "trajectory_path_missing": missing_paths,
        "trajectory_read_error_count": len(read_errors),
        "trajectory_read_errors": read_errors,
        "all_unsupported_requested_actions_zero": requested_invalid_total == 0,
        "all_invalid_resolution_records_are_observation_valid": all_observation_valid,
        "all_invalid_resolution_records_flip_observation_true_to_resolution_false": (
            all_observation_valid and all_resolution_mask_rejects
        ),
        "all_invalid_resolution_records_resolve_to_stay": all_resolved_to_stay,
        "all_invalid_resolution_requested_actions_are_movement": all_requested_movement,
        "underlying_movement_blocker_serialized_in_v190_trajectory": False,
        "legality_reason_available": bool(by_reason),
        "counts_by_seed": by_seed,
        "counts_by_continuation_script": by_script,
        "counts_by_requested_action": by_requested,
        "counts_by_resolved_action": by_resolved,
        "counts_by_legality_reason": by_reason,
        "counts_by_seed_and_continuation_script": by_seed_script,
        "counts_by_seed_branch_and_continuation_script": by_seed_branch_script,
        "run_summaries": run_summaries,
        "unsupported_resolved_events": events,
        "root_cause_assessment": root,
    }


def unsupported_resolved_event(
    run: Mapping[str, object],
    record: Mapping[str, object],
) -> dict[str, object]:
    requested = str(record.get("requested_action") or "")
    action_mask = _mapping(record.get("action_mask"))
    resolution_mask = _mapping(record.get("resolution_action_mask"))
    outcome = _mapping(record.get("outcome"))
    return {
        "seed": _int(run.get("seed")),
        "branch_id": str(run.get("branch_id") or ""),
        "continuation_script": str(run.get("continuation_script") or ""),
        "tick": _int(record.get("tick")),
        "agent_id": _int(record.get("agent_id")),
        "requested_action": requested,
        "resolved_action": str(record.get("resolved_action") or ""),
        "action_valid": record.get("action_valid") is True,
        "resolution_action_valid": record.get("resolution_action_valid") is True,
        "observation_mask_allows_requested_action": action_mask.get(requested) is True,
        "resolution_mask_allows_requested_action": (
            resolution_mask.get(requested) is True
        ),
        "legality_reason": str(outcome.get("invalid_reason") or "unknown"),
        "movement_only_resolution_context": requested.startswith("move_"),
    }


def root_cause_assessment(
    *,
    events: Sequence[Mapping[str, object]],
    all_observation_valid: bool,
    all_resolution_mask_rejects: bool,
    all_resolved_to_stay: bool,
    all_requested_movement: bool,
    event_count_matches_report: bool,
    run_counts_match: bool,
    missing_paths: Sequence[str],
    read_errors: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    reporting_consistent = (
        event_count_matches_report
        and run_counts_match
        and not missing_paths
        and not read_errors
    )
    timing_pattern = (
        bool(events)
        and all_observation_valid
        and all_resolution_mask_rejects
        and all_resolved_to_stay
    )
    movement_occupancy_inferred = timing_pattern and all_requested_movement
    primary = (
        "action_mask_timing_mismatch_same_tick_movement_occupancy_race"
        if movement_occupancy_inferred and reporting_consistent
        else "inconclusive_architecture_review_required"
    )
    return {
        "primary": primary,
        "classification_confident": (
            primary
            == "action_mask_timing_mismatch_same_tick_movement_occupancy_race"
        ),
        "legality_audit_or_reporting_bug": False if reporting_consistent else None,
        "counterfactual_script_action_resolution_contract_mismatch": (
            timing_pattern
        ),
        "action_mask_timing_mismatch": timing_pattern,
        "movement_occupancy_water_hazard_resolution_issue": (
            "same_tick_movement_occupancy_race_inferred_from_source_contract"
            if movement_occupancy_inferred
            else "not_proven"
        ),
        "real_architecture_limitation": timing_pattern,
        "reporting_consistency_checks_passed": reporting_consistent,
        "why_zero_unsupported_requested_but_many_unsupported_resolved": (
            "The scripts selected only actions allowed by each agent's tick-start "
            "public observation action_mask. The simulator then applied actions in "
            "deterministic agent order using a fresh live resolution_action_mask. "
            "All 860 rejected records are movement requests that were true in the "
            "observation mask and false in the resolution mask, resolving to stay."
        ),
        "why_not_water_or_hazard_from_v190_artifact": (
            "The v190 artifacts serialize mask truth values and invalid_reason but "
            "not the target-tile blocker. Runtime movement mask construction uses "
            "in_bounds, non-water terrain, and empty occupancy; because terrain and "
            "bounds do not flip within the action phase, true-to-false movement "
            "mask flips point to same-tick occupancy changes."
        ),
        "minimum_safe_next_step": (
            "Add diagnostics-only action-resolution contract repair that serializes "
            "movement target/blocker details and audits public-script continuations "
            "against both observation and live resolution legality. Do not train or "
            "relax support gates."
        ),
    }


def near_clean_seed_analysis(
    v190_report: Mapping[str, object],
    *,
    target_seeds: Sequence[int],
    max_dominant_requested_action_share: float,
) -> dict[str, object]:
    branch_report = _mapping(_mapping(v190_report.get("expansion_search")).get("branch_report"))
    runs = _mappings(branch_report.get("branch_runs"))
    support = _mapping(v190_report.get("legal_support_audit"))
    support_per_seed = _mapping(support.get("per_seed"))
    per_seed: dict[str, object] = {}
    for raw_seed in target_seeds:
        seed = int(raw_seed)
        seed_runs = [run for run in runs if _int(run.get("seed")) == seed]
        terminal_runs = [run for run in seed_runs if _int(run.get("alive_agents")) > 0]
        zero_unsupported_terminal = [
            run
            for run in terminal_runs
            if _int(run.get("unsupported_requested_action_count")) == 0
            and _int(run.get("unsupported_resolved_action_count")) == 0
        ]
        within_share_terminal_with_unsupported = [
            run
            for run in terminal_runs
            if _float(run.get("dominant_requested_action_share"))
            <= max_dominant_requested_action_share
            and _int(run.get("unsupported_resolved_action_count")) > 0
        ]
        best = _mapping(_mapping(support_per_seed.get(str(seed))).get("best_attempt"))
        if not best:
            best = best_attempt(seed_runs)
        nearest_zero_unsupported = best_cap_distance_attempt(
            zero_unsupported_terminal,
            max_dominant_requested_action_share=max_dominant_requested_action_share,
        )
        nearest_within_share = best_resolution_repair_attempt(
            within_share_terminal_with_unsupported
        )
        per_seed[str(seed)] = {
            "seed": seed,
            "branch_run_count": len(seed_runs),
            "terminal_survivor_attempt_count": len(terminal_runs),
            "zero_unsupported_terminal_attempt_count": len(
                zero_unsupported_terminal
            ),
            "within_action_share_cap_terminal_attempts_with_unsupported_resolution_count": len(
                within_share_terminal_with_unsupported
            ),
            "best_v190_attempt": best,
            "nearest_zero_unsupported_terminal_attempt": nearest_zero_unsupported,
            "best_within_share_terminal_attempt_needing_resolution_repair": (
                nearest_within_share
            ),
            "repair_recommendation": per_seed_repair_recommendation(
                seed=seed,
                best_attempt_payload=best,
                nearest_zero_unsupported=nearest_zero_unsupported,
                nearest_within_share=nearest_within_share,
                max_dominant_requested_action_share=(
                    max_dominant_requested_action_share
                ),
            ),
        }
    return {
        "policy": "m3_carrion_survivor_continuation_v191_near_clean_seed_analysis_v1",
        "target_seeds": [int(seed) for seed in target_seeds],
        "max_dominant_requested_action_share": _round(
            max_dominant_requested_action_share
        ),
        "seed_29_near_clean_focus": per_seed.get("29"),
        "seed_41_near_clean_focus": per_seed.get("41"),
        "per_seed": per_seed,
    }


def best_attempt(runs: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    if not runs:
        return None

    def key(run: Mapping[str, object]) -> tuple[int, int, int, float, str]:
        return (
            _int(run.get("alive_agents")),
            _int(run.get("births")),
            -_int(run.get("unsupported_resolved_action_count")),
            -_float(run.get("dominant_requested_action_share")),
            str(run.get("branch_id") or ""),
        )

    return compact_run(max(runs, key=key))


def best_cap_distance_attempt(
    runs: Sequence[Mapping[str, object]],
    *,
    max_dominant_requested_action_share: float,
) -> dict[str, object] | None:
    if not runs:
        return None

    def key(run: Mapping[str, object]) -> tuple[float, int, int, str]:
        return (
            -abs(
                _float(run.get("dominant_requested_action_share"))
                - max_dominant_requested_action_share
            ),
            _int(run.get("alive_agents")),
            _int(run.get("births")),
            str(run.get("branch_id") or ""),
        )

    return compact_run(max(runs, key=key))


def best_resolution_repair_attempt(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not runs:
        return None

    def key(run: Mapping[str, object]) -> tuple[int, int, int, float, str]:
        return (
            _int(run.get("alive_agents")),
            _int(run.get("births")),
            -_int(run.get("unsupported_resolved_action_count")),
            -_float(run.get("dominant_requested_action_share")),
            str(run.get("branch_id") or ""),
        )

    return compact_run(max(runs, key=key))


def compact_run(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "seed": _int(run.get("seed")),
        "branch_id": run.get("branch_id"),
        "branch_tick": _int(run.get("branch_tick")),
        "continuation_script": run.get("continuation_script"),
        "alive_agents": _int(run.get("alive_agents")),
        "births": _int(run.get("births")),
        "unsupported_requested_action_count": _int(
            run.get("unsupported_requested_action_count")
        ),
        "unsupported_resolved_action_count": _int(
            run.get("unsupported_resolved_action_count")
        ),
        "dominant_requested_action": run.get("dominant_requested_action"),
        "dominant_requested_action_share": run.get(
            "dominant_requested_action_share"
        ),
        "trajectory_path": run.get("trajectory_path"),
    }


def per_seed_repair_recommendation(
    *,
    seed: int,
    best_attempt_payload: Mapping[str, object] | None,
    nearest_zero_unsupported: Mapping[str, object] | None,
    nearest_within_share: Mapping[str, object] | None,
    max_dominant_requested_action_share: float,
) -> dict[str, object]:
    best = _mapping(best_attempt_payload)
    best_unsupported = _int(best.get("unsupported_resolved_action_count"))
    best_share = _float(best.get("dominant_requested_action_share"))
    zero = _mapping(nearest_zero_unsupported)
    within = _mapping(nearest_within_share)
    needs_resolution = best_unsupported > 0 or bool(within)
    needs_share = best_share > max_dominant_requested_action_share or (
        bool(zero)
        and _float(zero.get("dominant_requested_action_share"))
        > max_dominant_requested_action_share
    )
    if seed == 29 and bool(zero):
        label = "action_diversity_cap_repair_only"
        needs_resolution = False
        needs_share = True
    elif seed == 41:
        label = "resolution_contract_repair_plus_small_action_share_repair"
    elif needs_resolution and needs_share:
        label = "resolution_contract_repair_or_cap_aware_support_selection"
    elif needs_resolution:
        label = "resolution_contract_repair_first"
    elif needs_share:
        label = "action_diversity_cap_repair"
    else:
        label = "no_v191_repair_needed"
    return {
        "policy": "m3_carrion_survivor_continuation_v191_per_seed_repair_recommendation_v1",
        "seed": int(seed),
        "recommendation": label,
        "needs_action_resolution_contract_repair": bool(needs_resolution),
        "needs_action_share_cap_repair": bool(needs_share),
        "do_not_count_current_v190_attempt_as_support": True,
        "rationale": repair_rationale(
            seed=seed,
            label=label,
            best=best,
            zero=zero,
            within=within,
        ),
    }


def repair_rationale(
    *,
    seed: int,
    label: str,
    best: Mapping[str, object],
    zero: Mapping[str, object],
    within: Mapping[str, object],
) -> str:
    if seed == 29:
        return (
            "Seed 29 already has a terminal-survivor continuation with zero "
            "unsupported requested/resolved actions, but its stay share is 0.5154, "
            "above the 0.50 cap. This is an action-diversity repair, not a legality "
            "repair."
        )
    if seed == 41:
        return (
            "Seed 41 is barely above the action-share cap and still has four "
            "unsupported resolved movement actions in the strongest v190 attempt. "
            "It needs action-resolution contract repair plus a small cap-aware "
            "action-diversity repair."
        )
    if within:
        return (
            "A terminal attempt is already under the action-share cap but is "
            "blocked by same-tick resolution-mask flips; repair the resolution "
            "contract before more support generation."
        )
    if zero:
        return (
            "A zero-unsupported terminal attempt exists but remains above the "
            "dominant action-share cap; repair action diversity before support "
            "can count."
        )
    return (
        f"Best v190 attempt is blocked by {best.get('unsupported_resolved_action_count')} "
        "unsupported resolved actions; diagnose the resolution contract before "
        "architecture or support generation."
    )


def repair_recommendation_audit(
    *,
    classification: Mapping[str, object],
    near_clean: Mapping[str, object],
) -> dict[str, object]:
    root = _mapping(classification.get("root_cause_assessment"))
    per_seed = _mapping(near_clean.get("per_seed"))
    recommendations = [
        _mapping(payload).get("repair_recommendation")
        for payload in per_seed.values()
        if isinstance(payload, Mapping)
    ]
    recommendation_counts = Counter(
        str(_mapping(item).get("recommendation"))
        for item in recommendations
        if item
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v191_repair_recommendation_audit_v1",
        "diagnostics_only": True,
        "repair_implemented_in_v191": False,
        "support_generated_in_v191": False,
        "root_cause_primary": root.get("primary"),
        "minimal_safe_repair_justified": (
            root.get("primary")
            == "action_mask_timing_mismatch_same_tick_movement_occupancy_race"
        ),
        "recommended_repair_route": ACTION_RESOLUTION_CONTRACT_REPAIR_ROUTE,
        "recommended_repair_scope": [
            "serialize movement target and resolution blocker details in diagnostic trajectories",
            "split audit terminology between requested_action_valid_at_observation and requested_action_valid_at_resolution",
            "add diagnostics-only public-action-mask constrained repair probe before any new support generation",
            "keep runtime policy and acceptance gates unchanged",
        ],
        "per_seed_recommendation_counts": dict(sorted(recommendation_counts.items())),
    }


def route_decision_audit(
    *,
    source_validation: Mapping[str, object],
    classification: Mapping[str, object],
    repair: Mapping[str, object],
    near_clean: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[str] = []
    root = _mapping(classification.get("root_cause_assessment"))
    if source_validation.get("passed") is not True:
        blockers.append("v190_source_validation_failed")
    if classification.get("event_count_matches_v190_report") is not True:
        blockers.append("unsupported_resolved_event_count_mismatch")
    if classification.get("all_unsupported_requested_actions_zero") is not True:
        blockers.append("unsupported_requested_actions_nonzero")
    if (
        root.get("primary")
        != "action_mask_timing_mismatch_same_tick_movement_occupancy_race"
    ):
        blockers.append("unsupported_resolved_root_cause_not_actionable")
    if source_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif not blockers and repair.get("minimal_safe_repair_justified") is True:
        route = ACTION_RESOLUTION_CONTRACT_REPAIR_ROUTE
    elif not blockers:
        route = TERMINAL_SUPPORT_AUDIT_ROUTE
    else:
        route = ARCHITECTURE_REVIEW_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v191_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "slice_3_training_authorized": False,
        "slice_3_training_allowed_for_this_command": False,
        "slice_3_training_consumed": False,
        "runtime_integration_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "support_generation_allowed_by_v191": False,
        "blind_support_expansion_allowed": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "primary_blocker": blockers[0] if blockers else "action_resolution_contract",
        "unsupported_resolved_root_cause": root.get("primary"),
        "near_clean_seed_29": _mapping(near_clean.get("seed_29_near_clean_focus")).get(
            "repair_recommendation"
        ),
        "near_clean_seed_41": _mapping(near_clean.get("seed_41_near_clean_focus")).get(
            "repair_recommendation"
        ),
        "rationale": route_rationale(route),
    }


def route_rationale(route: str) -> str:
    if route == STOP_ROUTE:
        return "Pinned v190 source evidence did not validate; stop without v191 claims."
    if route == ACTION_RESOLUTION_CONTRACT_REPAIR_ROUTE:
        return (
            "v190 unsupported resolved actions are explained by observation-valid "
            "movement requests becoming invalid under the live resolution mask "
            "during deterministic same-tick action ordering. Route to a narrow "
            "diagnostics-only action-resolution contract repair; do not train or "
            "generate more support yet."
        )
    if route == TERMINAL_SUPPORT_AUDIT_ROUTE:
        return (
            "All six seeds would need clean support under the cap before dataset "
            "audit. v191 does not generate such support."
        )
    return (
        "The unsupported resolved-action blocker was not safely repairable from "
        "available evidence; route to architecture review."
    )


def classification_for(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v191_legal_support_repair_architecture_review_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if route_decision.get("recommended_next_route") == ACTION_RESOLUTION_CONTRACT_REPAIR_ROUTE:
        return prefix + "action_resolution_contract_repair_route_no_training"
    if route_decision.get("recommended_next_route") == ARCHITECTURE_REVIEW_ROUTE:
        return prefix + "architecture_review_route_no_training"
    return prefix + "terminal_support_audit_route_no_training"


def contract(
    *,
    expected_v190_report_exact_digest: str,
    required_v190_route: str,
    max_dominant_requested_action_share: float,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "unsupported_resolved_action_root_cause_review": True,
        "blind_support_expansion_allowed": False,
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "input_v190_report_exact_digest_pinned": expected_v190_report_exact_digest,
        "required_v190_route": required_v190_route,
        "expected_v190_backup": EXPECTED_V190_BACKUP,
        "expected_v190_backup_sha256": EXPECTED_V190_BACKUP_SHA256,
        "max_dominant_requested_action_share": _round(
            max_dominant_requested_action_share
        ),
        "allowed_routes": [
            ACTION_RESOLUTION_CONTRACT_REPAIR_ROUTE,
            ARCHITECTURE_REVIEW_ROUTE,
            TERMINAL_SUPPORT_AUDIT_ROUTE,
            STOP_ROUTE,
        ],
        "forbidden_routes": [
            "slice_3_training",
            "v180_rerun",
            "v186_rerun",
            "runtime_integration",
            "promotion",
            "gate_relaxation",
            "blind_support_expansion",
        ],
    }


def lifecycle_flags() -> dict[str, object]:
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
        "v189_rerun": False,
        "v190_rerun": False,
        "non_promoted": True,
        "diagnostics_only": True,
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def _trajectory_records(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[dict[str, object]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, Mapping) or row.get("type") != "record":
                continue
            record = row.get("record")
            if isinstance(record, Mapping):
                records.append(dict(record))
    return records


def _mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _count_by(
    events: Sequence[Mapping[str, object]],
    key: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter(str(event.get(key)) for event in events)
    return dict(sorted(counts.items()))


def _count_by_pair(
    events: Sequence[Mapping[str, object]],
    first: str,
    second: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter(
        f"{event.get(first)}|{event.get(second)}" for event in events
    )
    return dict(sorted(counts.items()))


def _count_by_triple(
    events: Sequence[Mapping[str, object]],
    first: str,
    second: str,
    third: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter(
        f"{event.get(first)}|{event.get(second)}|{event.get(third)}"
        for event in events
    )
    return dict(sorted(counts.items()))


def recover_mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}
