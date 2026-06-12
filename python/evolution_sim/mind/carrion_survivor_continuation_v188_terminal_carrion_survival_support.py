from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind.candidate_campaign import _int, _mapping, write_json
from evolution_sim.mind.carrion_branch_explore import (
    DEFAULT_CARRION_BRANCH_BASE_SCRIPT,
    MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
    build_carrion_branch_explore_report,
)
from evolution_sim.mind.carrion_counterfactual import DEFAULT_COUNTERFACTUAL_SCRIPTS
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v187_v186_delta_blocker_review import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V187_REPORT_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_V187_V186_DELTA_BLOCKER_REVIEW_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V187_V186_DELTA_BLOCKER_REVIEW_SCHEMA_VERSION,
    RECOMMENDED_NEXT_ROUTE as REQUIRED_V187_ROUTE,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V188_TERMINAL_CARRION_SURVIVAL_SUPPORT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v188_terminal_carrion_survival_support_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V188_TERMINAL_CARRION_SURVIVAL_SUPPORT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v188_terminal_carrion_survival_support_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v188-carrion-survivor-continuation-terminal-survival-support.json"
)
DEFAULT_SUPPORT_TRAJECTORY_DIR = Path(
    "output/mind/v188-terminal-carrion-survival-support-trajectories"
)

EXPECTED_V187_REPORT_EXACT_DIGEST = (
    "4366edf8b1b53e9974d6548e5fe9876e305693711a1b5d66192ef42f9b6228c6"
)
EXPECTED_V187_BACKUP = (
    "gdrive:evolution-sim-backups/archives/"
    "20260612T161155Z-v187-v186-delta-blocker-review.tar.zst"
)
EXPECTED_V187_BACKUP_SHA256 = (
    "c4705d5a53a5f6e5b47dd2125f4cd92d54fc5afb0cf5911044b05fc3f9dab01f"
)

DEFAULT_CARRION_FIXTURE_SEEDS = (13, 19, 29, 37, 41, 43)
DEFAULT_TICKS = 120
DEFAULT_MAX_BRANCH_POINTS_PER_SEED = 1
DEFAULT_MIN_BRANCH_TICK = 0
DEFAULT_BASE_SCRIPT = DEFAULT_CARRION_BRANCH_BASE_SCRIPT
DEFAULT_CONTINUATION_SCRIPTS = DEFAULT_COUNTERFACTUAL_SCRIPTS

POSITIVE_SUPPORT_ROUTE = (
    "v189_terminal_survival_support_dataset_audit_before_slice_3_training"
)
NO_SUPPORT_ROUTE = "review_terminal_survival_feasibility_or_architecture_branch_no_training"
STOP_ROUTE = "stop"


def run_carrion_survivor_continuation_v188_terminal_carrion_survival_support(
    *,
    v187_report_path: str | Path = DEFAULT_V187_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    support_trajectory_dir: str | Path = DEFAULT_SUPPORT_TRAJECTORY_DIR,
    expected_v187_report_exact_digest: str = EXPECTED_V187_REPORT_EXACT_DIGEST,
    required_v187_route: str = REQUIRED_V187_ROUTE,
    seeds: Sequence[int] = DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = DEFAULT_TICKS,
    base_script: str = DEFAULT_BASE_SCRIPT,
    continuation_scripts: Sequence[str] = DEFAULT_CONTINUATION_SCRIPTS,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    min_branch_tick: int = DEFAULT_MIN_BRANCH_TICK,
    verify_replay: bool = True,
    branch_report_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    v187_report = load_json_report(v187_report_path)
    source_validation = validate_v188_v187_source(
        v187_report,
        expected_v187_report_exact_digest=expected_v187_report_exact_digest,
        required_v187_route=required_v187_route,
    )
    seed_values = tuple(int(seed) for seed in seeds)
    script_values = tuple(str(script) for script in continuation_scripts)
    search_budget = _search_budget(
        seeds=seed_values,
        ticks=ticks,
        base_script=base_script,
        continuation_scripts=script_values,
        max_branch_points_per_seed=max_branch_points_per_seed,
        min_branch_tick=min_branch_tick,
        verify_replay=verify_replay,
        support_trajectory_dir=support_trajectory_dir,
    )
    if source_validation.get("passed") is not True:
        branch_report = None
        support_search = _skipped_search(search_budget, reason="source_validation_failed")
    elif branch_report_override is not None:
        branch_report = dict(branch_report_override)
        support_search = _support_search(branch_report, search_budget)
    else:
        branch_report = build_carrion_branch_explore_report(
            seeds=seed_values,
            ticks=int(ticks),
            fixture_name="carrion_only",
            base_script=str(base_script),
            continuation_scripts=script_values,
            max_branch_points_per_seed=int(max_branch_points_per_seed),
            min_branch_tick=int(min_branch_tick),
            trajectory_output_dir=Path(support_trajectory_dir),
            verify_replay=bool(verify_replay),
        )
        support_search = _support_search(branch_report, search_budget)
    support_result = _support_result(
        branch_report=branch_report,
        seeds=seed_values,
        support_search=support_search,
    )
    route_decision = _route_decision(
        source_validation=source_validation,
        support_result=support_result,
    )
    classification = _classification(
        source_validation=source_validation,
        support_result=support_result,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V188_TERMINAL_CARRION_SURVIVAL_SUPPORT_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V188_TERMINAL_CARRION_SURVIVAL_SUPPORT_POLICY
        ),
        "contract": _contract(
            expected_v187_report_exact_digest=expected_v187_report_exact_digest,
            required_v187_route=required_v187_route,
        ),
        "inputs": {
            "v187_report": str(v187_report_path),
            "expected_v187_report_exact_digest": expected_v187_report_exact_digest,
            "required_v187_route": required_v187_route,
            "v187_route_path": "route_decision.recommended_next_route",
            "v187_backup": EXPECTED_V187_BACKUP,
            "v187_backup_sha256": EXPECTED_V187_BACKUP_SHA256,
            "output": str(output_path),
            "support_trajectory_dir": str(support_trajectory_dir),
        },
        "source_validation": source_validation,
        "search_budget": search_budget,
        "support_search": support_search,
        "support_result": support_result,
        "legality_action_mask_validation": _legality_validation(
            support_search=support_search,
            support_result=support_result,
        ),
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "terminal_carrion_survival_support_generation",
                "no_slice_3_training",
                "no_runtime_integration",
                "no_promotion",
            ],
        },
        **_lifecycle_flags(support_generation_ran=support_search.get("ran") is True),
    }
    report["exact_digest"] = _digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v188_v187_source(
    v187_report: Mapping[str, object],
    *,
    expected_v187_report_exact_digest: str = EXPECTED_V187_REPORT_EXACT_DIGEST,
    required_v187_route: str = REQUIRED_V187_ROUTE,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v187_report)
    route = _mapping(v187_report.get("route_decision"))
    classification = _mapping(v187_report.get("classification"))
    checks = {
        "v187_schema_matches": (
            v187_report.get("schema_version")
            == M3_CARRION_SURVIVOR_CONTINUATION_V187_V186_DELTA_BLOCKER_REVIEW_SCHEMA_VERSION
        ),
        "v187_policy_matches": (
            v187_report.get("policy")
            == M3_CARRION_SURVIVOR_CONTINUATION_V187_V186_DELTA_BLOCKER_REVIEW_POLICY
        ),
        "v187_exact_digest_valid": exact_validation.get("passed") is True,
        "v187_exact_digest_matches_expected": (
            str(v187_report.get("exact_digest") or "")
            == str(expected_v187_report_exact_digest)
        ),
        "v187_route_matches_required": (
            str(route.get("recommended_next_route") or "") == str(required_v187_route)
        ),
        "v187_training_did_not_run": v187_report.get("training_ran") is False,
        "v187_training_artifact_not_created": (
            v187_report.get("training_artifact_created") is False
        ),
        "v187_slice_3_not_consumed": (
            v187_report.get("slice_3_training_consumed") is False
        ),
        "v187_runtime_artifact_not_created": (
            v187_report.get("runtime_artifact_created") is False
        ),
        "v187_runtime_action_selection_unchanged": (
            v187_report.get("runtime_action_selection_changed") is False
        ),
        "v187_promotion_not_authorized": (
            v187_report.get("promotion_authorized") is False
        ),
        "v187_gate_relaxation_not_allowed": (
            v187_report.get("gate_relaxation_allowed") is False
        ),
        "v187_support_expansion_not_ran": (
            v187_report.get("support_expansion_ran") is False
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v188_v187_source_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v187_report_exact_digest": expected_v187_report_exact_digest,
        "observed_v187_report_exact_digest": v187_report.get("exact_digest"),
        "v187_exact_digest_validation": exact_validation,
        "required_v187_route": required_v187_route,
        "observed_v187_route": route.get("recommended_next_route"),
        "observed_v187_classification": classification.get("primary"),
    }


def _search_budget(
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
    return {
        "policy": "m3_carrion_survivor_continuation_v188_search_budget_v1",
        "fixture": "carrion_only",
        "horizon_ticks": int(ticks),
        "seeds": [int(seed) for seed in seeds],
        "seed_count": len(seeds),
        "base_branch_policy": str(base_script),
        "continuation_scripts": [str(script) for script in continuation_scripts],
        "script_count": len(continuation_scripts),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "min_branch_tick": int(min_branch_tick),
        "max_branch_points": int(max_branch_points_per_seed) * len(seeds),
        "max_branch_continuation_runs": (
            int(max_branch_points_per_seed) * len(seeds) * len(continuation_scripts)
        ),
        "branch_engine_schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_engine_policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "verify_replay": bool(verify_replay),
        "support_trajectory_dir": str(support_trajectory_dir),
        "bounded_policy_visible_script_space": True,
        "runtime_policy_installation": False,
    }


def _skipped_search(
    search_budget: Mapping[str, object],
    *,
    reason: str,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v188_support_search_v1",
        "ran": False,
        "reason": reason,
        "search_budget": dict(search_budget),
        "branch_report_digest": None,
        "branch_report": None,
    }


def _support_search(
    branch_report: Mapping[str, object],
    search_budget: Mapping[str, object],
) -> dict[str, object]:
    aggregate = _mapping(branch_report.get("aggregate"))
    return {
        "policy": "m3_carrion_survivor_continuation_v188_support_search_v1",
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
        "unsupported_resolved_action_count": _int(
            aggregate.get("unsupported_resolved_action_count")
        ),
        "terminal_survivor_count_by_seed": _terminal_survivors_by_seed(
            aggregate,
            _int_sequence(_mapping(search_budget).get("seeds")),
        ),
        "trajectory_paths": list(aggregate.get("trajectory_paths") or []),
        "branch_report": dict(branch_report),
    }


def _support_result(
    *,
    branch_report: Mapping[str, object] | None,
    seeds: Sequence[int],
    support_search: Mapping[str, object],
) -> dict[str, object]:
    if support_search.get("ran") is not True or branch_report is None:
        return {
            "policy": "m3_carrion_survivor_continuation_v188_support_result_v1",
            "positive_support_found": False,
            "evidence_type": "not_searched_source_invalid",
            "terminal_survivors_by_seed": {
                str(int(seed)): 0 for seed in seeds
            },
            "support_runs": [],
            "support_trajectory_manifest": [],
            "infeasibility_scope": None,
        }
    aggregate = _mapping(branch_report.get("aggregate"))
    branch_runs = [
        dict(run)
        for run in branch_report.get("branch_runs", [])
        if isinstance(run, Mapping)
    ]
    successful_runs = [
        run
        for run in branch_runs
        if _int(run.get("alive_agents")) > 0
        and _int(run.get("heuristic_action_source_count")) == 0
        and _int(run.get("unsupported_requested_action_count")) == 0
        and _int(run.get("unsupported_resolved_action_count")) == 0
        and (
            run.get("replay_verification") is None
            or _mapping(run.get("replay_verification")).get("verified") is True
        )
    ]
    positive = bool(successful_runs)
    terminal_by_seed = _terminal_survivors_by_successful_runs(successful_runs, seeds)
    attempted_terminal_by_seed = _terminal_survivors_by_seed(aggregate, seeds)
    best_support = _best_successful_run(successful_runs)
    return {
        "policy": "m3_carrion_survivor_continuation_v188_support_result_v1",
        "positive_support_found": positive,
        "evidence_type": (
            "legal_deterministic_branch_continuation"
            if positive
            else "bounded_policy_visible_script_space_infeasibility_proof"
        ),
        "terminal_survivors_by_seed": terminal_by_seed,
        "attempted_terminal_survivors_by_seed": attempted_terminal_by_seed,
        "target_seed_count": len(seeds),
        "positive_seed_count": sum(
            1 for count in terminal_by_seed.values() if int(count) > 0
        ),
        "attempted_positive_seed_count": _int(aggregate.get("positive_seed_count")),
        "successful_branch_run_count": len(successful_runs),
        "attempted_successful_branch_run_count": _int(
            aggregate.get("successful_branch_run_count")
        ),
        "terminal_alive_agent_total": sum(
            _int(run.get("alive_agents")) for run in successful_runs
        ),
        "attempted_terminal_alive_agent_total": _int(
            aggregate.get("terminal_alive_agent_total")
        ),
        "best_support": best_support,
        "attempted_best_branch_run": aggregate.get("best_branch_run"),
        "support_runs": [_support_run_payload(run) for run in successful_runs],
        "support_trajectory_manifest": [
            _support_trajectory_payload(run) for run in successful_runs
        ],
        "all_target_seeds_have_positive_support": all(
            _int(terminal_by_seed.get(str(int(seed)))) > 0 for seed in seeds
        ),
        "fresh_audit_required_before_slice_3_training": positive,
        "infeasibility_scope": None if positive else _infeasibility_scope(
            branch_report=branch_report,
            seeds=seeds,
        ),
    }


def _legality_validation(
    *,
    support_search: Mapping[str, object],
    support_result: Mapping[str, object],
) -> dict[str, object]:
    if support_search.get("ran") is not True:
        return {
            "policy": "m3_carrion_survivor_continuation_v188_action_mask_validation_v1",
            "passed": False,
            "reason": "search_not_run",
            "unsupported_requested_action_count": None,
            "unsupported_resolved_action_count": None,
            "replay_verified": False,
        }
    support_runs = [
        _mapping(run)
        for run in support_result.get("support_runs", [])
        if isinstance(run, Mapping)
    ]
    if support_result.get("positive_support_found") is True:
        unsupported_requested = sum(
            _int(run.get("unsupported_requested_action_count")) for run in support_runs
        )
        unsupported_resolved = sum(
            _int(run.get("unsupported_resolved_action_count")) for run in support_runs
        )
        replay_verified = all(run.get("replay_verified") is True for run in support_runs)
        checks = {
            "selected_support_run_count_positive": bool(support_runs),
            "unsupported_requested_action_count_zero": unsupported_requested == 0,
            "unsupported_resolved_action_count_zero": unsupported_resolved == 0,
            "deterministic_replay_verified": replay_verified,
        }
        failures = [name for name, passed in checks.items() if not passed]
        return {
            "policy": "m3_carrion_survivor_continuation_v188_action_mask_validation_v1",
            "validation_scope": "selected_positive_support_runs",
            **checks,
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
            "unsupported_requested_action_count": unsupported_requested,
            "unsupported_resolved_action_count": unsupported_resolved,
            "aggregate_attempted_unsupported_requested_action_count": _int(
                support_search.get("unsupported_requested_action_count")
            ),
            "aggregate_attempted_unsupported_resolved_action_count": _int(
                support_search.get("unsupported_resolved_action_count")
            ),
            "replay_verified": replay_verified,
            "selected_actions_are_legal_under_action_mask": not failures,
        }
    unsupported_requested = _int(support_search.get("unsupported_requested_action_count"))
    unsupported_resolved = _int(support_search.get("unsupported_resolved_action_count"))
    replay_verified = support_search.get("replay_verified") is True
    checks = {
        "unsupported_requested_action_count_zero": unsupported_requested == 0,
        "unsupported_resolved_action_count_zero": unsupported_resolved == 0,
        "deterministic_replay_verified": replay_verified,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v188_action_mask_validation_v1",
        "validation_scope": "bounded_search_space",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "unsupported_requested_action_count": unsupported_requested,
        "unsupported_resolved_action_count": unsupported_resolved,
        "replay_verified": replay_verified,
        "selected_actions_are_legal_under_action_mask": not failures,
    }


def _route_decision(
    *,
    source_validation: Mapping[str, object],
    support_result: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True:
        selected = STOP_ROUTE
    elif support_result.get("positive_support_found") is True:
        selected = POSITIVE_SUPPORT_ROUTE
    else:
        selected = NO_SUPPORT_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v188_route_decision_v1",
        "recommended_next_route": selected,
        "selected_route": selected,
        "exactly_one_next_route_recommended": True,
        "fresh_audit_required_before_slice_3_training": (
            selected == POSITIVE_SUPPORT_ROUTE
        ),
        "slice_3_training_allowed": False,
        "runtime_integration_allowed": False,
        "promotion_authorized": False,
        "rationale": _route_rationale(
            source_validation=source_validation,
            support_result=support_result,
        ),
    }


def _route_rationale(
    *,
    source_validation: Mapping[str, object],
    support_result: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True:
        return "Pinned v187 digest or route did not validate; stop."
    if support_result.get("positive_support_found") is True:
        return (
            "At least one legal deterministic carrion_only@120 branch "
            "continuation produced terminal alive agents. Route to a fresh "
            "terminal-survival support dataset audit before any slice-3 training."
        )
    return (
        "The bounded policy-visible script and branch space produced no terminal "
        "survivor support; review feasibility scope or branch architecture without "
        "training."
    )


def _classification(
    *,
    source_validation: Mapping[str, object],
    support_result: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v188_terminal_carrion_survival_support_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if route_decision.get("recommended_next_route") == POSITIVE_SUPPORT_ROUTE:
        return prefix + "positive_legal_branch_support_found_no_training"
    if support_result.get("positive_support_found") is False:
        return prefix + "bounded_infeasibility_scope_no_training"
    return prefix + "closed_no_training"


def _contract(
    *,
    expected_v187_report_exact_digest: str,
    required_v187_route: str,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "support_evidence_only": True,
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "v186_rerun_allowed": False,
        "v187_rerun_allowed": False,
        "generic_carrion_autopsy_rerun_allowed": False,
        "scalar_threshold_nearest_neighbor_tuning_allowed": False,
        "input_v187_report_exact_digest_pinned": expected_v187_report_exact_digest,
        "required_v187_route": required_v187_route,
        "positive_support_next_route": POSITIVE_SUPPORT_ROUTE,
        "no_support_next_route": NO_SUPPORT_ROUTE,
    }


def _lifecycle_flags(*, support_generation_ran: bool) -> dict[str, object]:
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
        "support_generation_ran": bool(support_generation_ran),
        "support_expansion_ran": False,
        "generic_carrion_autopsy_rerun": False,
        "v186_rerun": False,
        "v187_rerun": False,
        "non_promoted": True,
    }


def _support_run_payload(run: Mapping[str, object]) -> dict[str, object]:
    replay = _mapping(run.get("replay_verification"))
    return {
        "branch_id": run.get("branch_id"),
        "seed": _int(run.get("seed")),
        "fixture": run.get("fixture"),
        "ticks": _int(run.get("ticks")),
        "branch_tick": _int(run.get("branch_tick")),
        "base_script": run.get("base_script"),
        "continuation_script": run.get("continuation_script"),
        "branch_state_digest": run.get("branch_state_digest"),
        "alive_agents": _int(run.get("alive_agents")),
        "births": _int(run.get("births")),
        "deaths": _int(run.get("deaths")),
        "trajectory_path": run.get("trajectory_path"),
        "heuristic_action_source_count": _int(run.get("heuristic_action_source_count")),
        "unsupported_requested_action_count": _int(
            run.get("unsupported_requested_action_count")
        ),
        "unsupported_resolved_action_count": _int(
            run.get("unsupported_resolved_action_count")
        ),
        "dominant_requested_action": run.get("dominant_requested_action"),
        "dominant_requested_action_share": run.get("dominant_requested_action_share"),
        "replay_verified": replay.get("verified") is True if replay else None,
        "replay_digest": replay.get("expected_digest"),
    }


def _support_trajectory_payload(run: Mapping[str, object]) -> dict[str, object]:
    replay = _mapping(run.get("replay_verification"))
    return {
        "path": run.get("trajectory_path"),
        "seed": _int(run.get("seed")),
        "branch_id": run.get("branch_id"),
        "continuation_script": run.get("continuation_script"),
        "logical_replay_digest": replay.get("expected_digest"),
        "terminal_alive_agents": _int(run.get("alive_agents")),
        "births": _int(run.get("births")),
    }


def _infeasibility_scope(
    *,
    branch_report: Mapping[str, object],
    seeds: Sequence[int],
) -> dict[str, object]:
    contract = _mapping(branch_report.get("contract"))
    aggregate = _mapping(branch_report.get("aggregate"))
    return {
        "policy": "bounded_policy_visible_branch_script_space_v1",
        "fixture": contract.get("fixture_name", "carrion_only"),
        "seeds": [int(seed) for seed in seeds],
        "ticks": contract.get("ticks"),
        "base_script": contract.get("base_script"),
        "continuation_scripts": contract.get("continuation_scripts"),
        "max_branch_points_per_seed": contract.get("max_branch_points_per_seed"),
        "branch_point_count": aggregate.get("branch_point_count"),
        "branch_run_count": aggregate.get("branch_run_count"),
        "unrecoverable_state_summary": aggregate.get("unrecoverable_state_summary"),
        "claim_scope": (
            "No terminal survivor found only within this bounded deterministic "
            "policy-visible script and branch space."
        ),
    }


def _terminal_survivors_by_seed(
    aggregate: Mapping[str, object],
    seeds: Sequence[int],
) -> dict[str, int]:
    observed = _mapping(aggregate.get("terminal_survivor_count_by_seed"))
    return {
        str(int(seed)): _int(observed.get(str(int(seed))))
        for seed in seeds
    }


def _terminal_survivors_by_successful_runs(
    branch_runs: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
) -> dict[str, int]:
    values = {str(int(seed)): 0 for seed in seeds}
    for run in branch_runs:
        seed = str(_int(run.get("seed")))
        if seed in values:
            values[seed] += _int(run.get("alive_agents"))
    return values


def _best_successful_run(
    branch_runs: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not branch_runs:
        return None
    return dict(
        max(
            branch_runs,
            key=lambda run: (
                _int(run.get("alive_agents")),
                _int(run.get("births")),
                -_int(run.get("deaths")),
                str(run.get("continuation_script") or ""),
            ),
        )
    )


def _int_sequence(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(int(item) for item in value)


def _digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)
