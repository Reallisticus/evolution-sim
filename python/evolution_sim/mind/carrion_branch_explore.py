from __future__ import annotations

import gzip
import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.cli.mind_v3_evaluate import (
    _aggregate_runs,
    _dominant_action_summary,
    _fixture_world,
    _heuristic_action_source_count,
    _json_ready,
    _neural_anchor_diagnostics,
    _reproduction_failure_attribution,
    _round,
    _safe_path_part,
    _temporal_readiness_attribution,
    _write_trajectory_records,
)
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.mind.carrion_counterfactual import (
    DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    DEFAULT_COUNTERFACTUAL_SCRIPTS,
    CarrionCounterfactualPolicy,
)
from evolution_sim.mind.horizon_labels import ANIMAL_RESOURCE_FOOD_SOURCES
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION = (
    "mind_v3_carrion_branch_explore_v1"
)
MIND_V3_CARRION_BRANCH_EXPLORE_POLICY = (
    "deterministic_post_contact_branch_explore_v1"
)
DEFAULT_CARRION_BRANCH_BASE_SCRIPT = "hydration_safe_carrion_cycle"
DEFAULT_CARRION_BRANCH_POINTS_PER_SEED = 1


class CarrionBranchExploreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _BranchPoint:
    branch_id: str
    seed: int
    fixture_name: str
    branch_tick: int
    branch_index: int
    base_script: str
    contact: dict[str, object]
    alive_agents_at_branch: int
    births_at_branch: int
    deaths_at_branch: int
    trajectory_record_count_at_branch: int
    branch_state_digest: str
    world: SimulationWorld


def build_carrion_branch_explore_report(
    *,
    seeds: Sequence[int] = DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    ticks: int = DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    fixture_name: str = "carrion_only",
    base_script: str = DEFAULT_CARRION_BRANCH_BASE_SCRIPT,
    continuation_scripts: Sequence[str] = DEFAULT_COUNTERFACTUAL_SCRIPTS,
    max_branch_points_per_seed: int = DEFAULT_CARRION_BRANCH_POINTS_PER_SEED,
    min_branch_tick: int = 0,
    trajectory_output_dir: str | Path | None = None,
    verify_replay: bool = True,
) -> dict[str, object]:
    if fixture_name != "carrion_only":
        raise CarrionBranchExploreError(
            "carrion branch explore currently supports carrion_only only"
        )
    seed_values = _validated_seeds(seeds)
    tick_count = _positive_int(ticks, field="ticks")
    branch_limit = _positive_int(
        max_branch_points_per_seed,
        field="max_branch_points_per_seed",
    )
    min_tick = _nonnegative_int(min_branch_tick, field="min_branch_tick")
    script_values = _validated_scripts(continuation_scripts)
    _validated_scripts((base_script,))
    output_dir = Path(trajectory_output_dir) if trajectory_output_dir else None

    contract = _branch_contract(
        seeds=seed_values,
        ticks=tick_count,
        fixture_name=fixture_name,
        base_script=base_script,
        continuation_scripts=script_values,
        max_branch_points_per_seed=branch_limit,
        min_branch_tick=min_tick,
        verify_replay=verify_replay,
    )
    branch_points: list[_BranchPoint] = []
    branch_runs: list[dict[str, object]] = []
    discovery_reports: list[dict[str, object]] = []
    for seed in seed_values:
        discovered = _discover_branch_points(
            seed=seed,
            ticks=tick_count,
            fixture_name=fixture_name,
            base_script=base_script,
            max_branch_points=branch_limit,
            min_branch_tick=min_tick,
        )
        branch_points.extend(discovered["branch_points"])
        discovery_reports.append(discovered["report"])
        for branch_point in discovered["branch_points"]:
            for continuation_script in script_values:
                branch_runs.append(
                    _run_branch_continuation(
                        branch_point,
                        continuation_script=continuation_script,
                        ticks=tick_count,
                        trajectory_output_dir=output_dir,
                        verify_replay=verify_replay,
                    )
                )

    aggregate = _aggregate_branch_runs(
        branch_points=branch_points,
        branch_runs=branch_runs,
        seeds=seed_values,
    )
    acceptance = _acceptance(
        aggregate,
        seeds=seed_values,
        replay_verification_required=verify_replay,
    )
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "contract": contract,
        "provenance": {
            "branch_contract_digest": stable_payload_digest(contract),
        },
        "scope": {
            "fixture_name": fixture_name,
            "state_restore_available": True,
            "state_restore_policy": (
                "in_process_deepcopy_of_exact_simulator_state_after_contact_tick_v1"
            ),
            "branch_replay_policy": (
                "deterministic_in_process_copied_branch_state_continuation_"
                "replay_v1"
            ),
            "policy_input_policy": (
                "base and continuation scripts use policy-visible observation "
                "self, local_patch, navigation, and action_mask only"
            ),
        },
        "aggregate": aggregate,
        "acceptance": acceptance,
        "discovery": discovery_reports,
        "branch_points": [_branch_point_payload(point) for point in branch_points],
        "branch_runs": branch_runs,
    }


def write_carrion_branch_explore_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _discover_branch_points(
    *,
    seed: int,
    ticks: int,
    fixture_name: str,
    base_script: str,
    max_branch_points: int,
    min_branch_tick: int,
) -> dict[str, object]:
    world = _fixture_world(
        fixture_name=fixture_name,
        seed=seed,
        ticks=ticks,
        policy=CarrionCounterfactualPolicy(base_script),
    )
    _configure_manual_summary_run(world)
    branch_points: list[_BranchPoint] = []
    contact_count = 0
    ticks_executed = 0
    for tick in range(ticks):
        world.tick = tick
        world._run_tick()
        ticks_executed = tick + 1
        contacts = _animal_resource_contact_records(world.tick_trajectory_records)
        contact_count += len(contacts)
        if tick >= min_branch_tick:
            for contact in contacts:
                if len(branch_points) >= max_branch_points:
                    break
                snapshot = deepcopy(world)
                branch_index = len(branch_points)
                branch_id = _branch_id(
                    fixture_name=fixture_name,
                    seed=seed,
                    branch_index=branch_index,
                    tick=tick,
                    contact=contact,
                )
                branch_points.append(
                    _BranchPoint(
                        branch_id=branch_id,
                        seed=seed,
                        fixture_name=fixture_name,
                        branch_tick=tick,
                        branch_index=branch_index,
                        base_script=base_script,
                        contact=contact,
                        alive_agents_at_branch=len(snapshot.alive_agents()),
                        births_at_branch=int(snapshot.births),
                        deaths_at_branch=int(snapshot.deaths),
                        trajectory_record_count_at_branch=len(
                            snapshot.trajectory_records
                        ),
                        branch_state_digest=_branch_state_digest(
                            snapshot,
                            branch_id=branch_id,
                            branch_tick=tick,
                        ),
                        world=snapshot,
                    )
                )
            if len(branch_points) >= max_branch_points:
                break
        if not world.alive_agents():
            break
    return {
        "branch_points": branch_points,
        "report": {
            "seed": int(seed),
            "fixture": fixture_name,
            "base_script": base_script,
            "ticks_requested": int(ticks),
            "ticks_executed_until_discovery_stop": int(ticks_executed),
            "animal_resource_contact_count_seen": int(contact_count),
            "branch_point_count": len(branch_points),
            "branch_ids": [point.branch_id for point in branch_points],
            "terminal_alive_before_branch_stop": len(world.alive_agents()),
            "births_before_branch_stop": int(world.births),
            "deaths_before_branch_stop": int(world.deaths),
        },
    }


def _run_branch_continuation(
    branch_point: _BranchPoint,
    *,
    continuation_script: str,
    ticks: int,
    trajectory_output_dir: Path | None,
    verify_replay: bool,
) -> dict[str, object]:
    run, digest = _execute_branch_continuation(
        branch_point,
        continuation_script=continuation_script,
        ticks=ticks,
        trajectory_output_dir=trajectory_output_dir,
    )
    verification = None
    if verify_replay:
        replay_run, replay_digest = _execute_branch_continuation(
            branch_point,
            continuation_script=continuation_script,
            ticks=ticks,
            trajectory_output_dir=None,
        )
        verification = {
            "verified": replay_digest == digest,
            "expected_digest": digest,
            "actual_digest": replay_digest,
            "replay_alive_agents": int(replay_run["alive_agents"]),
            "replay_births": int(replay_run["births"]),
            "replay_deaths": int(replay_run["deaths"]),
        }
    run["replay_verification"] = verification
    return run


def _execute_branch_continuation(
    branch_point: _BranchPoint,
    *,
    continuation_script: str,
    ticks: int,
    trajectory_output_dir: Path | None,
) -> tuple[dict[str, object], str]:
    world = deepcopy(branch_point.world)
    world.policy = CarrionCounterfactualPolicy(continuation_script)
    _configure_manual_summary_run(world)
    for tick in range(branch_point.branch_tick + 1, ticks):
        world.tick = tick
        world._run_tick()
        if not world.alive_agents():
            break
    trajectory_path = _branch_trajectory_path(
        trajectory_output_dir,
        branch_id=branch_point.branch_id,
        continuation_script=continuation_script,
        ticks=ticks,
    )
    run = _summarize_branch_world(
        world,
        branch_point=branch_point,
        continuation_script=continuation_script,
        ticks=ticks,
        trajectory_output_path=trajectory_path,
    )
    digest = stable_payload_digest(_replay_digest_payload(run))
    return run, digest


def _summarize_branch_world(
    world: SimulationWorld,
    *,
    branch_point: _BranchPoint,
    continuation_script: str,
    ticks: int,
    trajectory_output_path: Path | None,
) -> dict[str, object]:
    summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)
    action_source_counts = Counter(
        str(record.get("action_source", "unknown"))
        for record in world.trajectory_records
    )
    policy_id_counts = Counter(
        str(record.get("policy_id", "unknown"))
        for record in world.trajectory_records
    )
    requested_action_counts = Counter(
        str(record["requested_action"])
        for record in world.trajectory_records
        if isinstance(record.get("requested_action"), str)
    )
    resolved_action_counts = Counter(
        str(record["resolved_action"])
        for record in world.trajectory_records
        if isinstance(record.get("resolved_action"), str)
    )
    dominant_action = _dominant_action_summary(requested_action_counts)
    if trajectory_output_path is not None:
        _write_trajectory_records(
            world=world,
            summary=summary,
            output_path=trajectory_output_path,
            seed=branch_point.seed,
            split_id=(
                "mind_v3_carrion_branch_explore_"
                f"{branch_point.branch_id}_{continuation_script}"
            ),
        )
    heuristic_action_count = _heuristic_action_source_count(action_source_counts)
    run = {
        "branch_id": branch_point.branch_id,
        "seed": branch_point.seed,
        "fixture": branch_point.fixture_name,
        "ticks": int(ticks),
        "branch_tick": branch_point.branch_tick,
        "branch_index": branch_point.branch_index,
        "base_script": branch_point.base_script,
        "continuation_script": continuation_script,
        "branch_state_digest": branch_point.branch_state_digest,
        "contact": dict(branch_point.contact),
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "trophic_role_counts_at_end": _json_ready(
            summary.get("trophic_role_counts_at_end", {})
        ),
        "meat_mode_counts_at_end": _json_ready(
            summary.get("meat_mode_counts_at_end", {})
        ),
        "animal_resource_opportunity_by_meat_mode_end": _json_ready(
            summary.get("animal_resource_opportunity_by_meat_mode_end", {})
        ),
        "diet_by_trophic_role_end": _json_ready(
            summary.get("diet_by_trophic_role_end", {})
        ),
        "diet_by_meat_mode_end": _json_ready(summary.get("diet_by_meat_mode_end", {})),
        "combat_end": _json_ready(summary.get("combat_end", {})),
        "fresh_kill_end": _json_ready(summary.get("fresh_kill_end", {})),
        "carcass_end": _json_ready(summary.get("carcass_end", {})),
        "reproduction_failure_attribution": _reproduction_failure_attribution(
            summary
        ),
        "temporal_readiness_attribution": _temporal_readiness_attribution(
            world.trajectory_records
        ),
        "neural_anchor_diagnostics": _neural_anchor_diagnostics(
            world.policy_decision_diagnostics_records
        ),
        "trajectory_record_count": len(world.trajectory_records),
        "heuristic_action_source_count": heuristic_action_count,
        "zero_heuristic_runtime_actions": heuristic_action_count == 0,
        "unique_requested_actions": len(requested_action_counts),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "dominant_requested_action": dominant_action["action"],
        "dominant_requested_action_count": dominant_action["count"],
        "dominant_requested_action_share": dominant_action["share"],
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
    }
    if trajectory_output_path is not None:
        run["trajectory_path"] = str(trajectory_output_path)
    return run


def _aggregate_branch_runs(
    *,
    branch_points: Sequence[_BranchPoint],
    branch_runs: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
) -> dict[str, object]:
    successful_runs = [
        run
        for run in branch_runs
        if int(run.get("alive_agents", 0)) > 0
        and int(run.get("heuristic_action_source_count", 1)) == 0
    ]
    seeds_with_branch = sorted({int(point.seed) for point in branch_points})
    seeds_with_success = sorted({int(run["seed"]) for run in successful_runs})
    replay_items = [
        run.get("replay_verification")
        for run in branch_runs
        if run.get("replay_verification") is not None
    ]
    replay_verified = bool(replay_items) and all(
        bool(item.get("verified", False))
        for item in replay_items
        if isinstance(item, Mapping)
    )
    return {
        "seed_count": len(seeds),
        "branch_point_count": len(branch_points),
        "branch_run_count": len(branch_runs),
        "seeds_with_branch": seeds_with_branch,
        "seeds_with_branch_count": len(seeds_with_branch),
        "seeds_with_successful_branch": seeds_with_success,
        "positive_seed_count": len(seeds_with_success),
        "successful_branch_run_count": len(successful_runs),
        "terminal_alive_agent_total": sum(
            int(run.get("alive_agents", 0)) for run in successful_runs
        ),
        "max_alive_agents": max(
            (int(run.get("alive_agents", 0)) for run in branch_runs),
            default=0,
        ),
        "max_births": max(
            (int(run.get("births", 0)) for run in branch_runs),
            default=0,
        ),
        "replay_verification_count": len(replay_items),
        "replay_verified": replay_verified,
        "best_branch_run": _best_branch_run(branch_runs),
        "by_continuation_script": _continuation_script_summaries(branch_runs),
        "combined": _aggregate_runs([dict(run) for run in branch_runs])
        if branch_runs
        else {},
    }


def _acceptance(
    aggregate: Mapping[str, object],
    *,
    seeds: Sequence[int],
    replay_verification_required: bool,
) -> dict[str, object]:
    positive_seed_count = int(aggregate.get("positive_seed_count", 0))
    branch_point_count = int(aggregate.get("branch_point_count", 0))
    successful_branch_run_count = int(aggregate.get("successful_branch_run_count", 0))
    replay_verified = bool(aggregate.get("replay_verified", False))
    blockers = []
    if branch_point_count <= 0:
        blockers.append("no_branch_points_found")
    if successful_branch_run_count <= 0:
        blockers.append("no_successful_terminal_survivor_branch")
    if positive_seed_count < len(seeds):
        blockers.append("not_all_target_seeds_have_terminal_survivor_branch")
    if replay_verification_required and not replay_verified:
        blockers.append("branch_replay_not_verified")
    return {
        "diagnostic_acceptance_passed": not blockers,
        "blockers": blockers,
        "requires_positive_terminal_survivor_per_seed": True,
        "requires_zero_heuristic_runtime_actions": True,
        "requires_replay_verification": replay_verification_required,
        "positive_seed_count": positive_seed_count,
        "target_seed_count": len(seeds),
        "successful_branch_run_count": successful_branch_run_count,
    }


def _continuation_script_summaries(
    branch_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in branch_runs:
        grouped.setdefault(str(run.get("continuation_script", "unknown")), []).append(
            dict(run)
        )
    return {
        script: {
            "run_count": len(runs),
            "successful_branch_run_count": sum(
                1
                for run in runs
                if int(run.get("alive_agents", 0)) > 0
                and int(run.get("heuristic_action_source_count", 1)) == 0
            ),
            "aggregate": _aggregate_runs(runs),
        }
        for script, runs in sorted(grouped.items())
    }


def _best_branch_run(
    branch_runs: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not branch_runs:
        return None

    def key(run: Mapping[str, object]) -> tuple[int, int, float, str]:
        return (
            int(run.get("alive_agents", 0)),
            int(run.get("births", 0)),
            -float(run.get("dominant_requested_action_share", 1.0)),
            str(run.get("branch_id", "")),
        )

    selected = max(branch_runs, key=key)
    return {
        "branch_id": str(selected.get("branch_id", "")),
        "seed": int(selected.get("seed", 0)),
        "branch_tick": int(selected.get("branch_tick", 0)),
        "base_script": str(selected.get("base_script", "")),
        "continuation_script": str(selected.get("continuation_script", "")),
        "alive_agents": int(selected.get("alive_agents", 0)),
        "births": int(selected.get("births", 0)),
        "dominant_requested_action": selected.get("dominant_requested_action"),
        "dominant_requested_action_share": selected.get(
            "dominant_requested_action_share",
            0.0,
        ),
        "trajectory_path": selected.get("trajectory_path"),
    }


def _animal_resource_contact_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    contacts = []
    for record in records:
        outcome = record.get("outcome")
        outcome_payload = outcome if isinstance(outcome, Mapping) else {}
        feeding = outcome_payload.get("feeding")
        feeding_payload = feeding if isinstance(feeding, Mapping) else {}
        food_source = feeding_payload.get("food_source")
        if food_source not in ANIMAL_RESOURCE_FOOD_SOURCES:
            continue
        if feeding_payload.get("ate") is False:
            continue
        contacts.append(
            {
                "tick": int(record.get("tick", 0)),
                "agent_id": int(record.get("agent_id", 0)),
                "requested_action": str(record.get("requested_action", "")),
                "resolved_action": str(record.get("resolved_action", "")),
                "food_source": str(food_source),
                "gained_energy": _round(
                    _finite_float(feeding_payload.get("gained_energy"))
                ),
                "consumed": _round(_finite_float(feeding_payload.get("consumed"))),
                "before": _state_excerpt(record.get("before")),
                "after": _state_excerpt(record.get("after")),
            }
        )
    return contacts


def _branch_point_payload(point: _BranchPoint) -> dict[str, object]:
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture_name,
        "branch_tick": point.branch_tick,
        "branch_index": point.branch_index,
        "base_script": point.base_script,
        "contact": dict(point.contact),
        "alive_agents_at_branch": point.alive_agents_at_branch,
        "births_at_branch": point.births_at_branch,
        "deaths_at_branch": point.deaths_at_branch,
        "trajectory_record_count_at_branch": point.trajectory_record_count_at_branch,
        "branch_state_digest": point.branch_state_digest,
    }


def _branch_contract(
    *,
    seeds: Sequence[int],
    ticks: int,
    fixture_name: str,
    base_script: str,
    continuation_scripts: Sequence[str],
    max_branch_points_per_seed: int,
    min_branch_tick: int,
    verify_replay: bool,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "fixture_name": fixture_name,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "base_script": base_script,
        "continuation_scripts": list(continuation_scripts),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "min_branch_tick": int(min_branch_tick),
        "branch_trigger": (
            "first trajectory record at or after min_branch_tick whose outcome "
            "feeding.food_source is carcass or fresh_kill"
        ),
        "branch_timing": "post_tick_after_contact_record_v1",
        "verify_replay": bool(verify_replay),
    }


def _branch_id(
    *,
    fixture_name: str,
    seed: int,
    branch_index: int,
    tick: int,
    contact: Mapping[str, object],
) -> str:
    agent_id = int(contact.get("agent_id", 0))
    return (
        f"{_safe_path_part(fixture_name)}-seed-{int(seed)}-"
        f"branch-{int(branch_index)}-tick-{int(tick)}-agent-{agent_id}"
    )


def _branch_state_digest(
    world: SimulationWorld,
    *,
    branch_id: str,
    branch_tick: int,
) -> str:
    alive = [
        {
            "agent_id": int(agent.agent_id),
            "lineage_id": int(agent.lineage_id),
            "x": int(agent.x),
            "y": int(agent.y),
            "age": int(agent.age),
            "energy": _round(float(agent.energy)),
            "hydration": _round(float(agent.hydration)),
            "health": _round(float(agent.health)),
            "alive": bool(agent.alive),
        }
        for agent in sorted(world.alive_agents(), key=lambda item: item.agent_id)
    ]
    carcass_cells = []
    for y, row in enumerate(world.grid):
        for x, tile in enumerate(row):
            if tile.carcass_deposits:
                carcass_cells.append(
                    {
                        "x": x,
                        "y": y,
                        "energy": _round(
                            sum(
                                float(deposit.energy_remaining)
                                for deposit in tile.carcass_deposits
                            )
                        ),
                        "count": len(tile.carcass_deposits),
                    }
                )
    return stable_payload_digest(
        {
            "branch_id": branch_id,
            "seed": int(world.config.seed),
            "tick": int(branch_tick),
            "births": int(world.births),
            "deaths": int(world.deaths),
            "next_agent_id": int(world.next_agent_id),
            "alive_agents": alive,
            "carcass_cells": carcass_cells,
        }
    )


def _replay_digest_payload(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": run.get("branch_id"),
        "seed": run.get("seed"),
        "fixture": run.get("fixture"),
        "ticks": run.get("ticks"),
        "branch_tick": run.get("branch_tick"),
        "base_script": run.get("base_script"),
        "continuation_script": run.get("continuation_script"),
        "branch_state_digest": run.get("branch_state_digest"),
        "ticks_executed": run.get("ticks_executed"),
        "alive_agents": run.get("alive_agents"),
        "births": run.get("births"),
        "deaths": run.get("deaths"),
        "requested_action_counts": run.get("requested_action_counts"),
        "resolved_action_counts": run.get("resolved_action_counts"),
        "action_source_counts": run.get("action_source_counts"),
        "policy_id_counts": run.get("policy_id_counts"),
    }


def _branch_trajectory_path(
    output_dir: Path | None,
    *,
    branch_id: str,
    continuation_script: str,
    ticks: int,
) -> Path | None:
    if output_dir is None:
        return None
    return output_dir / (
        f"branch-{_safe_path_part(branch_id)}-"
        f"{_safe_path_part(continuation_script)}-{int(ticks)}.jsonl.gz"
    )


def _configure_manual_summary_run(world: SimulationWorld) -> None:
    world.record_events = False
    world.record_tick_details = True
    world.record_trajectory = True
    world.retain_trajectory_records = True
    world.trajectory_sink = None


def _state_excerpt(payload: object) -> dict[str, object]:
    state = payload if isinstance(payload, Mapping) else {}
    return {
        "x": int(state.get("x", 0)) if isinstance(state.get("x", 0), int) else 0,
        "y": int(state.get("y", 0)) if isinstance(state.get("y", 0), int) else 0,
        "energy_ratio": _round(_finite_float(state.get("energy_ratio"))),
        "hydration_ratio": _round(_finite_float(state.get("hydration_ratio"))),
        "health_ratio": _round(_finite_float(state.get("health_ratio"))),
        "alive": bool(state.get("alive", False)),
    }


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return 0.0
    return parsed


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if not values:
        raise CarrionBranchExploreError("at least one seed is required")
    return values


def _validated_scripts(scripts: Sequence[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(str(script) for script in scripts if str(script)))
    if not values:
        raise CarrionBranchExploreError("at least one script is required")
    unsupported = sorted(
        script for script in values if script not in DEFAULT_COUNTERFACTUAL_SCRIPTS
    )
    if unsupported:
        raise CarrionBranchExploreError(
            "unsupported branch script(s): " + ", ".join(unsupported)
        )
    return values


def _positive_int(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise CarrionBranchExploreError(f"{field} must be positive")
    return parsed


def _nonnegative_int(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise CarrionBranchExploreError(f"{field} must be nonnegative")
    return parsed


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
