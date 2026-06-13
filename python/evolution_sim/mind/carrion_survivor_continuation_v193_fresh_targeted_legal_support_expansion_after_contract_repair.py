from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import gzip
import json
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion as v190,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v192_action_resolution_contract_repair as v192,
)
from evolution_sim.mind.candidate_campaign import _float, _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_branch_explore import (
    MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
    build_carrion_branch_explore_report,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V193_FRESH_TARGETED_LEGAL_SUPPORT_EXPANSION_AFTER_CONTRACT_REPAIR_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V193_FRESH_TARGETED_LEGAL_SUPPORT_EXPANSION_AFTER_CONTRACT_REPAIR_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v193-carrion-survivor-continuation-fresh-targeted-legal-support-expansion-after-contract-repair.json"
)
DEFAULT_SUPPORT_TRAJECTORY_DIR = Path(
    "output/mind/v193-fresh-targeted-legal-support-expansion-after-contract-repair-trajectories"
)
DEFAULT_V192_REPORT_PATH = v192.DEFAULT_OUTPUT_PATH

EXPECTED_V192_REPORT_EXACT_DIGEST = (
    "17c49629d1268ff2f15d248465241775b9205b212222cc56a3132a7b9b8dfdae"
)
EXPECTED_V191_REPORT_EXACT_DIGEST = v192.EXPECTED_V191_REPORT_EXACT_DIGEST
EXPECTED_V190_REPORT_EXACT_DIGEST = v192.EXPECTED_V190_REPORT_EXACT_DIGEST
REQUIRED_V192_ROUTE = v192.NEXT_ROUTE_AFTER_REPAIR
EXPECTED_V192_BACKUP = (
    "gdrive:evolution-sim-backups/archives/"
    "20260613T151045Z-v192-action-resolution-contract-repair.tar.zst"
)
EXPECTED_V192_BACKUP_SHA256 = (
    "e5210b816dddfb6e54ad48fadad417f057fa7de93af6981acbec46991f265fac"
)

DEFAULT_TARGET_SEEDS = v192.TARGET_SEEDS
DEFAULT_TICKS = v190.DEFAULT_TICKS
DEFAULT_BASE_SCRIPT = v190.DEFAULT_BASE_SCRIPT
DEFAULT_CONTINUATION_SCRIPTS = v190.DEFAULT_CONTINUATION_SCRIPTS
DEFAULT_MAX_BRANCH_POINTS_PER_SEED = v190.DEFAULT_MAX_BRANCH_POINTS_PER_SEED
DEFAULT_MIN_BRANCH_TICK = v190.DEFAULT_MIN_BRANCH_TICK
DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE = (
    v190.DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE
)

SUCCESS_ROUTE = (
    "v194_repaired_contract_terminal_survival_support_dataset_audit_before_slice_3_training_no_training"
)
BLOCKED_ROUTE = "v194_repaired_contract_cap_support_repair_no_training"
STOP_ROUTE = "stop"

FALSE_V192_LIFECYCLE_FLAGS = (
    "training_ran",
    "training_artifact_created",
    "slice_3_training_consumed",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "promotion_authorized",
    "gate_relaxation_allowed",
    "support_generation_ran",
    "support_expansion_ran",
)


def run_carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair(
    *,
    v192_report_path: str | Path = DEFAULT_V192_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    support_trajectory_dir: str | Path = DEFAULT_SUPPORT_TRAJECTORY_DIR,
    expected_v192_report_exact_digest: str = EXPECTED_V192_REPORT_EXACT_DIGEST,
    expected_v191_report_exact_digest: str = EXPECTED_V191_REPORT_EXACT_DIGEST,
    expected_v190_report_exact_digest: str = EXPECTED_V190_REPORT_EXACT_DIGEST,
    required_v192_route: str = REQUIRED_V192_ROUTE,
    seeds: Sequence[int] = DEFAULT_TARGET_SEEDS,
    ticks: int = DEFAULT_TICKS,
    base_script: str = DEFAULT_BASE_SCRIPT,
    continuation_scripts: Sequence[str] = DEFAULT_CONTINUATION_SCRIPTS,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    min_branch_tick: int = DEFAULT_MIN_BRANCH_TICK,
    max_dominant_requested_action_share: float = (
        DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE
    ),
    verify_replay: bool = True,
    v192_report_override: Mapping[str, object] | None = None,
    branch_report_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    v192_report = (
        dict(v192_report_override)
        if v192_report_override is not None
        else load_json_report(v192_report_path)
    )
    source_validation = validate_v193_v192_source(
        v192_report,
        expected_v192_report_exact_digest=expected_v192_report_exact_digest,
        expected_v191_report_exact_digest=expected_v191_report_exact_digest,
        expected_v190_report_exact_digest=expected_v190_report_exact_digest,
        required_v192_route=required_v192_route,
    )
    search_budget = build_search_budget(
        seeds=seeds,
        ticks=ticks,
        base_script=base_script,
        continuation_scripts=continuation_scripts,
        max_branch_points_per_seed=max_branch_points_per_seed,
        min_branch_tick=min_branch_tick,
        verify_replay=verify_replay,
        support_trajectory_dir=support_trajectory_dir,
    )
    if source_validation.get("passed") is not True:
        branch_report = None
        expansion_search = skipped_expansion_search(
            search_budget,
            reason="source_validation_failed",
        )
    elif branch_report_override is not None:
        branch_report = dict(branch_report_override)
        expansion_search = summarize_expansion_search(branch_report, search_budget)
    else:
        branch_report = build_carrion_branch_explore_report(
            seeds=tuple(_int(seed) for seed in seeds),
            ticks=int(ticks),
            fixture_name="carrion_only",
            base_script=str(base_script),
            continuation_scripts=tuple(str(script) for script in continuation_scripts),
            max_branch_points_per_seed=int(max_branch_points_per_seed),
            min_branch_tick=int(min_branch_tick),
            trajectory_output_dir=Path(support_trajectory_dir),
            verify_replay=bool(verify_replay),
        )
        expansion_search = summarize_expansion_search(branch_report, search_budget)

    support_audit = repaired_contract_support_audit(
        branch_report=branch_report,
        target_seeds=tuple(_int(seed) for seed in seeds),
        max_dominant_requested_action_share=max_dominant_requested_action_share,
        replay_verification_required=verify_replay,
    )
    route_decision = route_decision_audit(
        source_validation=source_validation,
        support_audit=support_audit,
    )
    classification = classification_for(
        source_validation=source_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V193_FRESH_TARGETED_LEGAL_SUPPORT_EXPANSION_AFTER_CONTRACT_REPAIR_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V193_FRESH_TARGETED_LEGAL_SUPPORT_EXPANSION_AFTER_CONTRACT_REPAIR_POLICY
        ),
        "contract": contract(
            expected_v192_report_exact_digest=expected_v192_report_exact_digest,
            expected_v191_report_exact_digest=expected_v191_report_exact_digest,
            expected_v190_report_exact_digest=expected_v190_report_exact_digest,
            required_v192_route=required_v192_route,
            max_dominant_requested_action_share=max_dominant_requested_action_share,
        ),
        "inputs": {
            "v192_report": str(v192_report_path),
            "expected_v192_report_exact_digest": expected_v192_report_exact_digest,
            "expected_v191_report_exact_digest": expected_v191_report_exact_digest,
            "expected_v190_report_exact_digest": expected_v190_report_exact_digest,
            "required_v192_route": required_v192_route,
            "v192_backup": EXPECTED_V192_BACKUP,
            "v192_backup_sha256": EXPECTED_V192_BACKUP_SHA256,
            "output": str(output_path),
            "support_trajectory_dir": str(support_trajectory_dir),
        },
        "source_validation": source_validation,
        "search_budget": search_budget,
        "fresh_expansion_search": expansion_search,
        "repaired_contract_support_audit": support_audit,
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "fresh_targeted_legal_terminal_survival_support_expansion",
                "v192_repaired_support_evidence_contract",
                "no_slice_3_training",
                "no_runtime_integration",
                "no_promotion",
            ],
        },
        **lifecycle_flags(support_expansion_ran=expansion_search.get("ran") is True),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v193_v192_source(
    v192_report: Mapping[str, object],
    *,
    expected_v192_report_exact_digest: str,
    expected_v191_report_exact_digest: str,
    expected_v190_report_exact_digest: str,
    required_v192_route: str,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v192_report)
    source = _mapping(v192_report.get("source_validation"))
    route = _mapping(v192_report.get("route_decision"))
    repair = _mapping(v192_report.get("repair_scope_decision"))
    movement = _mapping(v192_report.get("movement_target_blocker_audit"))
    blocker_counts = _mapping(movement.get("blocker_classification_counts"))
    checks = {
        "v192_schema_matches": (
            v192_report.get("schema_version")
            == v192.M3_CARRION_SURVIVOR_CONTINUATION_V192_ACTION_RESOLUTION_CONTRACT_REPAIR_SCHEMA_VERSION
        ),
        "v192_policy_matches": (
            v192_report.get("policy")
            == v192.M3_CARRION_SURVIVOR_CONTINUATION_V192_ACTION_RESOLUTION_CONTRACT_REPAIR_POLICY
        ),
        "v192_exact_digest_valid": exact_validation.get("passed") is True,
        "v192_exact_digest_matches_expected": (
            v192_report.get("exact_digest") == expected_v192_report_exact_digest
        ),
        "v192_route_matches_required": (
            route.get("recommended_next_route") == required_v192_route
        ),
        "v192_source_validation_passed": source.get("passed") is True,
        "v192_v191_digest_pin_matches_expected": (
            source.get("observed_v191_report_exact_digest")
            == expected_v191_report_exact_digest
        ),
        "v192_v190_digest_pin_matches_expected": (
            source.get("observed_v190_report_exact_digest")
            == expected_v190_report_exact_digest
        ),
        "v192_selected_support_evidence_contract_repair": (
            repair.get("selected_repair_type") == "support_evidence_contract_repair"
            and repair.get("contract_repair_sufficient") is True
        ),
        "v192_occupancy_drift_count_matches_expected": (
            _int(blocker_counts.get("resolution_invalid_same_tick_occupancy_race"))
            == v192.EXPECTED_UNSUPPORTED_RESOLVED_ACTION_COUNT
        ),
        "v192_no_static_or_unexpected_resolution_blockers": (
            _int(movement.get("bounds_blocker_count")) == 0
            and _int(movement.get("water_blocker_count")) == 0
            and _int(movement.get("hazard_blocker_count")) == 0
            and _int(movement.get("depleted_resource_blocker_count")) == 0
            and _int(movement.get("stale_or_illegal_script_mask_count")) == 0
        ),
        "v192_all_resolution_invalid_events_are_same_tick_occupancy_races": (
            movement.get("all_resolution_invalid_events_are_same_tick_occupancy_races")
            is True
        ),
    }
    for flag in FALSE_V192_LIFECYCLE_FLAGS:
        checks[f"v192_{flag}_closed"] = v192_report.get(flag) is False
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v193_v192_source_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v192_report_exact_digest": expected_v192_report_exact_digest,
        "observed_v192_report_exact_digest": v192_report.get("exact_digest"),
        "expected_v191_report_exact_digest": expected_v191_report_exact_digest,
        "observed_v191_report_exact_digest": source.get(
            "observed_v191_report_exact_digest"
        ),
        "expected_v190_report_exact_digest": expected_v190_report_exact_digest,
        "observed_v190_report_exact_digest": source.get(
            "observed_v190_report_exact_digest"
        ),
        "required_v192_route": required_v192_route,
        "observed_v192_route": route.get("recommended_next_route"),
        "v192_exact_digest_validation": exact_validation,
        "checks": checks,
    }


def build_search_budget(
    *,
    seeds: Sequence[int],
    ticks: int,
    base_script: str,
    continuation_scripts: Sequence[str],
    max_branch_points_per_seed: int,
    min_branch_tick: int,
    verify_replay: bool,
    support_trajectory_dir: str | Path,
) -> dict[str, object]:
    seed_values = [_int(seed) for seed in seeds]
    scripts = [str(script) for script in continuation_scripts]
    return {
        "policy": "m3_carrion_survivor_continuation_v193_search_budget_v1",
        "bounded": True,
        "open_ended_sweep": False,
        "fixture": "carrion_only",
        "horizon_ticks": int(ticks),
        "target_seeds": seed_values,
        "target_seed_count": len(seed_values),
        "base_branch_policy": str(base_script),
        "continuation_scripts": scripts,
        "script_count": len(scripts),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "min_branch_tick": int(min_branch_tick),
        "max_branch_points": int(max_branch_points_per_seed) * len(seed_values),
        "max_branch_continuation_runs": (
            int(max_branch_points_per_seed) * len(seed_values) * len(scripts)
        ),
        "branch_engine_schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_engine_policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "verify_replay": bool(verify_replay),
        "support_trajectory_dir": str(support_trajectory_dir),
        "uses_existing_exact_branch_replay_utility": True,
        "training_ran": False,
        "runtime_action_selection_changed": False,
    }


def skipped_expansion_search(
    search_budget: Mapping[str, object],
    *,
    reason: str,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v193_fresh_expansion_search_v1",
        "ran": False,
        "reason": reason,
        "search_budget": dict(search_budget),
        "branch_report_digest": None,
        "branch_report": None,
    }


def summarize_expansion_search(
    branch_report: Mapping[str, object],
    search_budget: Mapping[str, object],
) -> dict[str, object]:
    aggregate = _mapping(branch_report.get("aggregate"))
    return {
        "policy": "m3_carrion_survivor_continuation_v193_fresh_expansion_search_v1",
        "ran": True,
        "search_budget": dict(search_budget),
        "branch_report_digest": stable_payload_digest(branch_report),
        "branch_report_schema_version": branch_report.get("schema_version"),
        "branch_report_policy": branch_report.get("branch_policy"),
        "branch_point_count": _int(aggregate.get("branch_point_count")),
        "branch_run_count": _int(aggregate.get("branch_run_count")),
        "successful_branch_run_count": _int(
            aggregate.get("successful_branch_run_count")
        ),
        "positive_seed_count": _int(aggregate.get("positive_seed_count")),
        "replay_verified": aggregate.get("replay_verified") is True,
        "unsupported_requested_action_count": _int(
            aggregate.get("unsupported_requested_action_count")
        ),
        "raw_unsupported_resolved_action_count": _int(
            aggregate.get("unsupported_resolved_action_count")
        ),
        "attempted_terminal_survivors_by_seed": _mapping(
            aggregate.get("terminal_survivor_count_by_seed")
        ),
        "trajectory_paths": list(aggregate.get("trajectory_paths") or []),
        "branch_report": dict(branch_report),
    }


def repaired_contract_support_audit(
    *,
    branch_report: Mapping[str, object] | None,
    target_seeds: Sequence[int],
    max_dominant_requested_action_share: float,
    replay_verification_required: bool,
) -> dict[str, object]:
    seed_values = [_int(seed) for seed in target_seeds]
    if branch_report is None:
        return {
            "policy": "m3_carrion_survivor_continuation_v193_repaired_contract_support_audit_v1",
            "passed": False,
            "reason": "fresh_expansion_search_not_run",
            "target_seed_count": len(seed_values),
            "clean_support_seed_count": 0,
            "selected_support_runs": [],
            "per_seed": {
                str(seed): {
                    "seed": seed,
                    "clean_support_run_count": 0,
                    "blockers": ["not_searched"],
                    "best_repaired_contract_attempt": None,
                }
                for seed in seed_values
            },
            "blockers": ["fresh_expansion_search_not_run"],
        }

    branch_runs = _mappings(branch_report.get("branch_runs"))
    run_audits = [
        repaired_contract_run_audit(
            run,
            max_dominant_requested_action_share=max_dominant_requested_action_share,
            replay_verification_required=replay_verification_required,
        )
        for run in branch_runs
    ]
    per_seed = {
        str(seed): per_seed_repaired_support_summary(
            seed=seed,
            run_audits=run_audits,
            max_dominant_requested_action_share=max_dominant_requested_action_share,
        )
        for seed in seed_values
    }
    selected_support_runs = [
        summary["best_repaired_contract_attempt"]
        for summary in per_seed.values()
        if isinstance(summary.get("best_repaired_contract_attempt"), Mapping)
        and summary["best_repaired_contract_attempt"].get(
            "counts_as_repaired_contract_support"
        )
        is True
    ]
    selected_diversity = selected_support_set_action_diversity(selected_support_runs)
    missing = [
        seed
        for seed in seed_values
        if _int(_mapping(per_seed[str(seed)]).get("clean_support_run_count")) <= 0
    ]
    blockers: list[str] = []
    if missing:
        blockers.append("not_all_target_seeds_have_repaired_contract_support")
    if _float(selected_diversity.get("dominant_requested_action_share")) > (
        max_dominant_requested_action_share
    ):
        blockers.append("selected_support_set_dominant_requested_action_share_above_cap")
    aggregate_expected_drift = sum(
        _int(audit.get("expected_same_tick_occupancy_drift_count"))
        for audit in run_audits
    )
    aggregate_unexpected_resolution_invalid = sum(
        _int(audit.get("unexpected_resolution_invalid_count")) for audit in run_audits
    )
    aggregate_unsupported_requested = sum(
        _int(audit.get("unsupported_requested_action_count")) for audit in run_audits
    )
    selected_expected_drift = sum(
        _int(run.get("expected_same_tick_occupancy_drift_count"))
        for run in selected_support_runs
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v193_repaired_contract_support_audit_v1",
        "passed": not blockers,
        "blockers": blockers,
        "target_seed_count": len(seed_values),
        "target_seeds": seed_values,
        "clean_support_seed_count": sum(
            1
            for summary in per_seed.values()
            if _int(_mapping(summary).get("clean_support_run_count")) > 0
        ),
        "missing_clean_support_seeds": missing,
        "run_audit_count": len(run_audits),
        "clean_run_count": sum(
            1
            for audit in run_audits
            if audit.get("counts_as_repaired_contract_support") is True
        ),
        "terminal_survivor_attempt_count": sum(
            1 for audit in run_audits if _int(audit.get("alive_agents")) > 0
        ),
        "aggregate_unsupported_requested_action_count": aggregate_unsupported_requested,
        "aggregate_expected_same_tick_occupancy_drift_count": aggregate_expected_drift,
        "aggregate_unexpected_resolution_invalid_count": (
            aggregate_unexpected_resolution_invalid
        ),
        "aggregate_expected_occupancy_drift_counted_as_successful_move": False,
        "selected_support_run_count": len(selected_support_runs),
        "selected_support_runs": selected_support_runs,
        "selected_expected_same_tick_occupancy_drift_count": selected_expected_drift,
        "selected_support_set_action_diversity": selected_diversity,
        "dominant_requested_action": selected_diversity.get(
            "dominant_requested_action"
        ),
        "dominant_requested_action_count": selected_diversity.get(
            "dominant_requested_action_count"
        ),
        "dominant_requested_action_share": selected_diversity.get(
            "dominant_requested_action_share"
        ),
        "max_dominant_requested_action_share": _round(
            max_dominant_requested_action_share
        ),
        "per_seed": per_seed,
        "run_audits": run_audits,
        "future_trainable_payload_policy": (
            "current_public_observation_and_current_public_action_mask_only"
        ),
        "trainable_dataset_created": False,
    }


def repaired_contract_run_audit(
    run: Mapping[str, object],
    *,
    max_dominant_requested_action_share: float,
    replay_verification_required: bool,
) -> dict[str, object]:
    replay = _mapping(run.get("replay_verification"))
    replay_verified = (
        replay.get("verified") is True or run.get("replay_verified") is True
    )
    path = Path(str(run.get("trajectory_path") or ""))
    trajectory_audit = repaired_contract_trajectory_audit(path, run)
    blocker_counts = _mapping(trajectory_audit.get("resolution_invalid_blocker_counts"))
    expected_drift = _int(
        blocker_counts.get("resolution_invalid_same_tick_occupancy_race")
    )
    unexpected_resolution_invalid = _int(
        trajectory_audit.get("unexpected_resolution_invalid_count")
    )
    unsupported_requested = _int(trajectory_audit.get("unsupported_requested_count"))
    checks = {
        "terminal_alive_positive": _int(run.get("alive_agents")) > 0,
        "unsupported_requested_action_count_zero": (
            _int(run.get("unsupported_requested_action_count")) == 0
            and unsupported_requested == 0
        ),
        "expected_same_tick_occupancy_drift_counted_separately": expected_drift
        >= 0,
        "expected_occupancy_drift_did_not_move": (
            _int(trajectory_audit.get("expected_occupancy_drift_moved_count")) == 0
        ),
        "unexpected_resolution_invalid_count_zero": unexpected_resolution_invalid == 0,
        "stale_or_illegal_script_mask_count_zero": (
            _int(blocker_counts.get("stale_or_illegal_script_mask")) == 0
        ),
        "bounds_blocker_count_zero": _int(blocker_counts.get("bounds_blocker")) == 0,
        "water_blocker_count_zero": _int(blocker_counts.get("water_blocker")) == 0,
        "hazard_blocker_count_zero": _int(blocker_counts.get("hazard_blocker")) == 0,
        "depleted_resource_blocker_count_zero": (
            _int(blocker_counts.get("depleted_resource_blocker")) == 0
        ),
        "heuristic_action_source_count_zero": (
            _int(run.get("heuristic_action_source_count")) == 0
        ),
        "dominant_requested_action_share_within_cap": (
            _float(run.get("dominant_requested_action_share"))
            <= max_dominant_requested_action_share
        ),
        "deterministic_replay_verified": (
            replay_verified if replay_verification_required else True
        ),
        "trajectory_path_present": bool(str(run.get("trajectory_path") or "")),
        "trajectory_readable": trajectory_audit.get("readable") is True,
        "terminal_facts_match_manifest": (
            trajectory_audit.get("terminal_facts_match_manifest") is True
        ),
        "record_count_matches_footer": (
            trajectory_audit.get("record_count_matches_footer") is True
        ),
        "action_mask_legality_passed_under_repaired_contract": (
            trajectory_audit.get("action_mask_legality_passed_under_repaired_contract")
            is True
        ),
        "trajectory_invalid_counts_match_repaired_contract": (
            trajectory_audit.get("trajectory_invalid_counts_match_repaired_contract")
            is True
        ),
        "trainable_leakage_scan_passed": (
            trajectory_audit.get("trainable_leakage_scan_passed") is True
        ),
        "trajectory_dominant_requested_action_share_within_cap": (
            _float(
                trajectory_audit.get("dominant_requested_action_share_from_trajectory")
            )
            <= max_dominant_requested_action_share
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v193_repaired_contract_run_audit_v1",
        "counts_as_repaired_contract_support": not failures,
        "failures": failures,
        "seed": _int(run.get("seed")),
        "branch_id": run.get("branch_id"),
        "continuation_script": run.get("continuation_script"),
        "alive_agents": _int(run.get("alive_agents")),
        "births": _int(run.get("births")),
        "deaths": _int(run.get("deaths")),
        "unsupported_requested_action_count": unsupported_requested,
        "raw_unsupported_resolved_action_count": _int(
            run.get("unsupported_resolved_action_count")
        ),
        "expected_same_tick_occupancy_drift_count": expected_drift,
        "unexpected_resolution_invalid_count": unexpected_resolution_invalid,
        "heuristic_action_source_count": _int(run.get("heuristic_action_source_count")),
        "dominant_requested_action": run.get("dominant_requested_action"),
        "dominant_requested_action_share": run.get(
            "dominant_requested_action_share"
        ),
        "replay_verified": replay_verified,
        "trajectory_path": str(path),
        "requested_action_counts": trajectory_audit.get("requested_action_counts", {}),
        "trajectory_audit": trajectory_audit,
        **checks,
    }


def repaired_contract_trajectory_audit(
    path: Path,
    run: Mapping[str, object],
) -> dict[str, object]:
    if not path.exists():
        return unreadable_trajectory_audit("missing_path")
    try:
        rows = list(iter_jsonl(path))
    except (OSError, EOFError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        payload = unreadable_trajectory_audit(type(exc).__name__)
        payload["error"] = str(exc)
        return payload
    header = next((row for row in rows if row.get("type") == "header"), None)
    footer = next((row for row in rows if row.get("type") == "footer"), None)
    records = [
        _mapping(row.get("record"))
        for row in rows
        if row.get("type") == "record" and isinstance(row.get("record"), Mapping)
    ]
    config = _mapping(_mapping(header).get("config"))
    blocker_counts: Counter[str] = Counter()
    expected_drift_events: list[dict[str, object]] = []
    unexpected_events: list[dict[str, object]] = []
    legality_failures: list[dict[str, object]] = []
    leakage_failures: list[dict[str, object]] = []
    requested_counts: Counter[str] = Counter()
    action_source_counts: Counter[str] = Counter()
    unsupported_requested_count = 0
    expected_drift_moved_count = 0
    for record_index, record in enumerate(records):
        requested = str(record.get("requested_action") or "")
        resolved = str(record.get("resolved_action") or requested)
        if requested:
            requested_counts.update([requested])
        action_source_counts.update([str(record.get("action_source") or "")])
        action_mask = _mapping(record.get("action_mask"))
        resolution_mask = _mapping(record.get("resolution_action_mask")) or action_mask
        if record.get("action_valid") is not True:
            unsupported_requested_count += 1
            legality_failures.append(
                {
                    "record_index": record_index,
                    "reason": "action_valid_false",
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
        if record.get("resolution_action_valid") is False:
            event = v192.movement_target_blocker_event(run, record, config=config)
            blocker = str(event.get("blocker_classification") or "")
            blocker_counts.update([blocker])
            if blocker == "resolution_invalid_same_tick_occupancy_race":
                expected_drift_events.append(event)
                if event.get("agent_moved") is True:
                    expected_drift_moved_count += 1
            else:
                unexpected_events.append(event)
                legality_failures.append(
                    {
                        "record_index": record_index,
                        "reason": "unexpected_resolution_invalid",
                        "blocker_classification": blocker,
                        "requested_action": requested,
                        "resolved_action": resolved,
                    }
                )
        elif resolved and resolution_mask.get(resolved) is not True:
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
        scan_trainable_key_leakage(
            candidate_payload,
            failures=leakage_failures,
            path=("candidate_trainable_payload",),
            record_index=record_index,
        )
    footer_summary = _mapping(_mapping(footer).get("summary"))
    trajectory_summary = _mapping(_mapping(footer).get("trajectory_summary"))
    invalid_action_count = _int(trajectory_summary.get("invalid_action_count"))
    invalid_observation_count = _int(
        trajectory_summary.get("invalid_observation_action_count")
    )
    invalid_resolution_count = _int(
        trajectory_summary.get("invalid_resolution_action_count")
    )
    terminal_matches = (
        footer is not None
        and _int(footer_summary.get("alive_agents")) == _int(run.get("alive_agents"))
        and _int(footer_summary.get("births")) == _int(run.get("births"))
        and _int(footer_summary.get("deaths")) == _int(run.get("deaths"))
    )
    dominant = dominant_count_share(requested_counts)
    expected_count = len(expected_drift_events)
    unexpected_count = len(unexpected_events)
    repaired_invalid_counts_match = (
        invalid_action_count == 0
        and invalid_observation_count == unsupported_requested_count
        and invalid_resolution_count == expected_count + unexpected_count
    )
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
        "invalid_action_count": invalid_action_count,
        "invalid_observation_action_count": invalid_observation_count,
        "invalid_resolution_action_count": invalid_resolution_count,
        "trajectory_invalid_counts_match_repaired_contract": (
            repaired_invalid_counts_match
        ),
        "unsupported_requested_count": unsupported_requested_count,
        "expected_same_tick_occupancy_drift_count": expected_count,
        "unexpected_resolution_invalid_count": unexpected_count,
        "expected_occupancy_drift_moved_count": expected_drift_moved_count,
        "resolution_invalid_blocker_counts": dict(sorted(blocker_counts.items())),
        "expected_same_tick_occupancy_drift_events": expected_drift_events[:32],
        "unexpected_resolution_invalid_events": unexpected_events[:32],
        "action_mask_legality_passed_under_repaired_contract": not legality_failures,
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


def unreadable_trajectory_audit(reason: str) -> dict[str, object]:
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
        "invalid_action_count": None,
        "invalid_observation_action_count": None,
        "invalid_resolution_action_count": None,
        "trajectory_invalid_counts_match_repaired_contract": False,
        "unsupported_requested_count": 0,
        "expected_same_tick_occupancy_drift_count": 0,
        "unexpected_resolution_invalid_count": 0,
        "expected_occupancy_drift_moved_count": 0,
        "resolution_invalid_blocker_counts": {},
        "expected_same_tick_occupancy_drift_events": [],
        "unexpected_resolution_invalid_events": [],
        "action_mask_legality_passed_under_repaired_contract": False,
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


def per_seed_repaired_support_summary(
    *,
    seed: int,
    run_audits: Sequence[Mapping[str, object]],
    max_dominant_requested_action_share: float,
) -> dict[str, object]:
    seed_audits = [
        dict(audit) for audit in run_audits if _int(audit.get("seed")) == int(seed)
    ]
    clean = [
        audit
        for audit in seed_audits
        if audit.get("counts_as_repaired_contract_support") is True
    ]
    terminal = [audit for audit in seed_audits if _int(audit.get("alive_agents")) > 0]
    within_share = [
        audit
        for audit in seed_audits
        if _float(audit.get("dominant_requested_action_share"))
        <= max_dominant_requested_action_share
    ]
    blockers: list[str] = []
    if not clean:
        blockers.append("no_repaired_contract_support_run")
    if terminal and not clean:
        blockers.append("terminal_survivors_exist_only_as_rejected_attempts")
    best = best_repaired_contract_attempt(seed_audits)
    return {
        "seed": int(seed),
        "branch_run_count": len(seed_audits),
        "terminal_survivor_attempt_count": len(terminal),
        "within_action_share_cap_attempt_count": len(within_share),
        "clean_support_run_count": len(clean),
        "expected_same_tick_occupancy_drift_count": sum(
            _int(audit.get("expected_same_tick_occupancy_drift_count"))
            for audit in seed_audits
        ),
        "unexpected_resolution_invalid_count": sum(
            _int(audit.get("unexpected_resolution_invalid_count"))
            for audit in seed_audits
        ),
        "blockers": blockers,
        "best_repaired_contract_attempt": best,
    }


def best_repaired_contract_attempt(
    audits: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not audits:
        return None

    def key(audit: Mapping[str, object]) -> tuple[int, int, int, int, int, float, str, str]:
        return (
            1 if audit.get("counts_as_repaired_contract_support") is True else 0,
            _int(audit.get("alive_agents")),
            _int(audit.get("births")),
            -_int(audit.get("unexpected_resolution_invalid_count")),
            -_int(audit.get("unsupported_requested_action_count")),
            -_float(audit.get("dominant_requested_action_share")),
            str(audit.get("branch_id") or ""),
            str(audit.get("continuation_script") or ""),
        )

    selected = max(audits, key=key)
    return {
        "seed": _int(selected.get("seed")),
        "branch_id": selected.get("branch_id"),
        "continuation_script": selected.get("continuation_script"),
        "alive_agents": selected.get("alive_agents"),
        "births": selected.get("births"),
        "unsupported_requested_action_count": selected.get(
            "unsupported_requested_action_count"
        ),
        "expected_same_tick_occupancy_drift_count": selected.get(
            "expected_same_tick_occupancy_drift_count"
        ),
        "unexpected_resolution_invalid_count": selected.get(
            "unexpected_resolution_invalid_count"
        ),
        "dominant_requested_action": selected.get("dominant_requested_action"),
        "dominant_requested_action_share": selected.get(
            "dominant_requested_action_share"
        ),
        "replay_verified": selected.get("replay_verified") is True,
        "trajectory_path": selected.get("trajectory_path"),
        "requested_action_counts": selected.get("requested_action_counts", {}),
        "counts_as_repaired_contract_support": (
            selected.get("counts_as_repaired_contract_support") is True
        ),
        "failures": selected.get("failures"),
    }


def selected_support_set_action_diversity(
    selected_support_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    for run in selected_support_runs:
        counts.update(
            {
                str(action): _int(count)
                for action, count in _mapping(run.get("requested_action_counts")).items()
            }
        )
    dominant = dominant_count_share(counts)
    return {
        "requested_action_counts": dict(sorted(counts.items())),
        "dominant_requested_action": dominant.get("key"),
        "dominant_requested_action_count": dominant.get("count"),
        "dominant_requested_action_share": dominant.get("share"),
    }


def route_decision_audit(
    *,
    source_validation: Mapping[str, object],
    support_audit: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[str] = []
    if source_validation.get("passed") is not True:
        blockers.append("v192_source_validation_failed")
    if support_audit.get("passed") is not True:
        blockers.extend(str(item) for item in support_audit.get("blockers", []))
    if source_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif not blockers:
        route = SUCCESS_ROUTE
    else:
        route = BLOCKED_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v193_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "slice_3_training_authorized": False,
        "slice_3_training_allowed_for_this_command": False,
        "slice_3_training_consumed": False,
        "fresh_dataset_audit_required_before_slice_3_training": (
            route == SUCCESS_ROUTE
        ),
        "cap_or_support_repair_required": route == BLOCKED_ROUTE,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "primary_blocker": blockers[0] if blockers else None,
        "clean_support_seed_count": support_audit.get("clean_support_seed_count"),
        "target_seed_count": support_audit.get("target_seed_count"),
        "dominant_requested_action": support_audit.get("dominant_requested_action"),
        "dominant_requested_action_share": support_audit.get(
            "dominant_requested_action_share"
        ),
        "rationale": route_rationale(route=route, blockers=blockers),
    }


def route_rationale(*, route: str, blockers: Sequence[str]) -> str:
    if route == STOP_ROUTE:
        return "Pinned v192 source evidence did not validate; stop without v193 claims."
    if route == SUCCESS_ROUTE:
        return (
            "Fresh targeted expansion produced repaired-contract legal terminal "
            "survival support for every target seed under the action-share cap. "
            "Route to a fresh dataset audit before any slice-3 training."
        )
    return (
        "Fresh targeted expansion remains partial under the repaired contract or "
        "violates the action-share cap. Slice 3 stays blocked; route to a "
        "no-training cap/support repair."
    )


def classification_for(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = (
        "m3_carrion_survivor_continuation_v193_fresh_targeted_legal_support_"
        "expansion_after_contract_repair_"
    )
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if route_decision.get("recommended_next_route") == SUCCESS_ROUTE:
        return prefix + "all_target_seed_support_routes_to_fresh_dataset_audit_no_training"
    return prefix + "partial_or_cap_blocked_routes_to_no_training_repair"


def contract(
    *,
    expected_v192_report_exact_digest: str,
    expected_v191_report_exact_digest: str,
    expected_v190_report_exact_digest: str,
    required_v192_route: str,
    max_dominant_requested_action_share: float,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "fresh_targeted_support_expansion_only": True,
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "input_v192_report_exact_digest_pinned": expected_v192_report_exact_digest,
        "input_v191_report_exact_digest_pinned": expected_v191_report_exact_digest,
        "input_v190_report_exact_digest_pinned": expected_v190_report_exact_digest,
        "required_v192_route": required_v192_route,
        "expected_v192_backup": EXPECTED_V192_BACKUP,
        "expected_v192_backup_sha256": EXPECTED_V192_BACKUP_SHA256,
        "max_dominant_requested_action_share": _round(
            max_dominant_requested_action_share
        ),
        "repaired_support_evidence_contract": {
            "observation_valid_same_tick_occupancy_drift_is_unsupported_request": False,
            "count_expected_same_tick_occupancy_drift_separately": True,
            "count_expected_same_tick_occupancy_drift_as_successful_move": False,
            "unsupported_requested_actions_block_support": True,
            "stale_or_illegal_script_actions_block_support": True,
            "bounds_water_hazard_depleted_or_unexpected_resolution_invalid_blocks_support": True,
            "dominant_requested_action_share_cap_is_hard": True,
        },
        "success_route": SUCCESS_ROUTE,
        "blocked_route": BLOCKED_ROUTE,
        "forbidden_routes": [
            "slice_3_training",
            "runtime_integration",
            "runtime_action_selection_change",
            "promotion",
            "gate_relaxation",
            "v180_rerun",
            "v186_rerun",
        ],
    }


def lifecycle_flags(*, support_expansion_ran: bool) -> dict[str, object]:
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
        "support_generation_ran": bool(support_expansion_ran),
        "support_expansion_ran": bool(support_expansion_ran),
        "generic_carrion_autopsy_rerun": False,
        "v180_rerun": False,
        "v186_rerun": False,
        "v188_rerun": False,
        "v189_rerun": False,
        "v190_rerun": False,
        "v191_rerun": False,
        "v192_rerun": False,
        "non_promoted": True,
        "diagnostics_only": True,
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def iter_jsonl(path: Path) -> Sequence[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scan_trainable_key_leakage(
    value: object,
    *,
    failures: list[dict[str, object]],
    path: tuple[str, ...],
    record_index: int,
) -> None:
    forbidden_tokens = (
        "seed",
        "fixture",
        "branch",
        "path",
        "digest",
        "lineage",
        "agent_id",
        "species_id",
        "episode",
        "target",
        "oracle",
        "outcome",
        "terminal",
        "births",
        "deaths",
        "alive_agents",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            lower = key_text.lower()
            for token in forbidden_tokens:
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
            scan_trainable_key_leakage(
                child,
                failures=failures,
                path=(*path, key_text),
                record_index=record_index,
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            scan_trainable_key_leakage(
                child,
                failures=failures,
                path=(*path, str(index)),
                record_index=record_index,
            )


def dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return {"key": key, "count": int(count), "share": _round(count / total)}


def _mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]
