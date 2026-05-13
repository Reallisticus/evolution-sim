from __future__ import annotations

import gzip
import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.cli.mind_v3_evaluate import (
    _dominant_action_summary,
    _fixture_world,
    _heuristic_action_source_count,
    _json_ready,
    _round,
    _safe_path_part,
)
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.mind.carrion_branch_explore import (
    _branch_state_digest,
    _configure_manual_summary_run,
    _nonnegative_int,
    _positive_int,
    _state_excerpt,
    _validated_seeds,
)
from evolution_sim.mind.carrion_counterfactual import (
    DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    DEFAULT_COUNTERFACTUAL_SCRIPTS,
    CarrionCounterfactualPolicy,
)
from evolution_sim.mind.dataset import (
    TRAJECTORY_DATASET_RECORD_INDEX_FIELD,
    TRAJECTORY_EPISODE_ID_FIELD,
    TRAJECTORY_SOURCE_PATH_FIELD,
)
from evolution_sim.mind.outcome_metrics import (
    aggregate_run_outcome_metrics,
    build_run_outcome_metrics,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.rollout_context import RolloutContextConfig
from evolution_sim.mind.rollout_context_audit import _contextual_rows

MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION = (
    "mind_v3_branch_action_oracle_audit_v1"
)
MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_POLICY = (
    "exact_ambiguous_post_carrion_first_action_oracle_v1"
)
DEFAULT_BRANCH_ACTION_ORACLE_SEEDS: tuple[int, ...] = (37, 41, 43)
DEFAULT_BRANCH_ACTION_ORACLE_BASE_SCRIPT = "carrion_then_water"
DEFAULT_BRANCH_ACTION_ORACLE_CANDIDATE_ACTIONS: tuple[str, ...] = (
    "drink",
    "eat",
    "stay",
)
DEFAULT_BRANCH_ACTION_ORACLE_TARGET_LABELS: tuple[str, ...] = ("drink", "eat")
DEFAULT_BRANCH_ACTION_ORACLE_POINTS_PER_SEED = 4
DEFAULT_TARGET_HORIZON_TRACE_TICKS: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    5,
    8,
    13,
    21,
    34,
    55,
    89,
)


class BranchActionOracleAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _ActionBranchPoint:
    branch_id: str
    seed: int
    fixture_name: str
    branch_tick: int
    branch_index: int
    record_index: int
    agent_id: int
    logged_action: str
    base_script: str
    before: dict[str, object]
    action_mask: dict[str, bool]
    observation_input: dict[str, object]
    observation_digest: str | None
    observation_schema: str | None
    context_snapshot: dict[str, object]
    branch_state_digest: str
    world: SimulationWorld


class _ForcedFirstActionPolicy:
    policy_version = "branch_action_oracle_forced_first_action_v1"

    def __init__(
        self,
        *,
        target_agent_id: int,
        forced_action: str,
        continuation_script: str,
    ):
        self.target_agent_id = int(target_agent_id)
        self.forced_action = str(forced_action)
        self.delegate = CarrionCounterfactualPolicy(continuation_script)
        self.policy_id = (
            f"mind_v3_branch_action_oracle_{self.forced_action}_"
            f"then_{continuation_script}"
        )
        self.used = False

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        metadata = observation.get("metadata")
        agent_id = metadata.get("agent_id") if isinstance(metadata, Mapping) else None
        if (
            not self.used
            and agent_id == self.target_agent_id
            and self.forced_action in ACTION_NAMES
            and bool(action_mask.get(self.forced_action, False))
        ):
            self.used = True
            return ActionDecision(
                requested_action=self.forced_action,
                source=f"branch_oracle_force:{self.forced_action}",
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                diagnostics={
                    "forced_first_action": self.forced_action,
                    "target_agent_id": self.target_agent_id,
                },
            )
        return self.delegate.decide(observation, action_mask)


def build_branch_action_oracle_audit_report(
    *,
    seeds: Sequence[int] = DEFAULT_BRANCH_ACTION_ORACLE_SEEDS,
    ticks: int = DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    fixture_name: str = "carrion_only",
    base_script: str = DEFAULT_BRANCH_ACTION_ORACLE_BASE_SCRIPT,
    continuation_script: str = DEFAULT_BRANCH_ACTION_ORACLE_BASE_SCRIPT,
    candidate_actions: Sequence[str] = DEFAULT_BRANCH_ACTION_ORACLE_CANDIDATE_ACTIONS,
    target_labels: Sequence[str] = DEFAULT_BRANCH_ACTION_ORACLE_TARGET_LABELS,
    max_branch_points_per_seed: int = DEFAULT_BRANCH_ACTION_ORACLE_POINTS_PER_SEED,
    min_branch_tick: int = 0,
    min_oracle_changed_action_count: int = 1,
    min_terminal_alive_gain_total: int = 1,
    verify_replay: bool = True,
) -> dict[str, object]:
    if fixture_name != "carrion_only":
        raise BranchActionOracleAuditError(
            "branch action oracle currently supports carrion_only only"
        )
    seed_values = _validated_seeds(seeds)
    tick_count = _positive_int(ticks, field="ticks")
    branch_limit = _positive_int(
        max_branch_points_per_seed,
        field="max_branch_points_per_seed",
    )
    min_tick = _nonnegative_int(min_branch_tick, field="min_branch_tick")
    actions = _validated_actions(candidate_actions)
    labels = _validated_actions(target_labels)
    _validated_script(base_script)
    _validated_script(continuation_script)
    changed_floor = _nonnegative_int(
        min_oracle_changed_action_count,
        field="min_oracle_changed_action_count",
    )
    alive_gain_floor = _nonnegative_int(
        min_terminal_alive_gain_total,
        field="min_terminal_alive_gain_total",
    )
    contract = _contract(
        seeds=seed_values,
        ticks=tick_count,
        fixture_name=fixture_name,
        base_script=base_script,
        continuation_script=continuation_script,
        candidate_actions=actions,
        target_labels=labels,
        max_branch_points_per_seed=branch_limit,
        min_branch_tick=min_tick,
        min_oracle_changed_action_count=changed_floor,
        min_terminal_alive_gain_total=alive_gain_floor,
        verify_replay=verify_replay,
    )
    branch_points: list[_ActionBranchPoint] = []
    branch_results: list[dict[str, object]] = []
    discovery: list[dict[str, object]] = []
    for seed in seed_values:
        discovered = _discover_branch_points(
            seed=seed,
            ticks=tick_count,
            fixture_name=fixture_name,
            base_script=base_script,
            candidate_actions=actions,
            target_labels=labels,
            max_branch_points=branch_limit,
            min_branch_tick=min_tick,
        )
        points = discovered["branch_points"]
        branch_points.extend(points)
        discovery.append(discovered["report"])
        for point in points:
            branch_results.append(
                _evaluate_branch_point(
                    point,
                    ticks=tick_count,
                    continuation_script=continuation_script,
                    candidate_actions=actions,
                    verify_replay=verify_replay,
                )
            )
    aggregate = _aggregate_results(branch_results)
    acceptance = _acceptance(
        aggregate,
        min_oracle_changed_action_count=changed_floor,
        min_terminal_alive_gain_total=alive_gain_floor,
    )
    return {
        "schema_version": MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_POLICY,
        "contract": contract,
        "provenance": {"contract_digest": stable_payload_digest(contract)},
        "scope": {
            "fixture_name": fixture_name,
            "state_restore_available": True,
            "state_restore_policy": (
                "in_process_deepcopy_of_exact_simulator_state_before_"
                "ambiguous_decision_tick_v1"
            ),
            "policy_input_policy": (
                "forced action is diagnostic-only; continuation script uses "
                "policy-visible observation self, local_patch, navigation, and "
                "action_mask"
            ),
        },
        "aggregate": aggregate,
        "acceptance": acceptance,
        "discovery": discovery,
        "branch_points": [_branch_point_payload(point) for point in branch_points],
        "branch_results": branch_results,
    }


def write_branch_action_oracle_audit_report(
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
    candidate_actions: Sequence[str],
    target_labels: Sequence[str],
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
    snapshots: dict[int, SimulationWorld] = {}
    ticks_executed = 0
    for tick in range(ticks):
        world.tick = tick
        snapshots[tick] = deepcopy(world)
        world._run_tick()
        ticks_executed = tick + 1
        if not world.alive_agents():
            break
    contextual_rows = _contextual_rows(
        _contextual_records(world.trajectory_records, seed=seed),
        config=RolloutContextConfig(),
    )
    branch_points: list[_ActionBranchPoint] = []
    for row in contextual_rows:
        if len(branch_points) >= max_branch_points:
            break
        tick = int(row.record.get("tick", 0))
        if tick < min_branch_tick:
            continue
        if not row.context_snapshot.get("post_carrion_contact"):
            continue
        if row.label not in target_labels:
            continue
        if not all(action in row.valid_actions for action in ("drink", "eat")):
            continue
        valid_candidates = [
            action for action in candidate_actions if action in row.valid_actions
        ]
        if len(valid_candidates) < 2:
            continue
        snapshot = snapshots.get(tick)
        if snapshot is None:
            continue
        branch_index = len(branch_points)
        agent_id = int(row.record.get("agent_id", 0))
        branch_id = _branch_id(
            fixture_name=fixture_name,
            seed=seed,
            branch_index=branch_index,
            tick=tick,
            agent_id=agent_id,
            logged_action=row.label,
        )
        branch_points.append(
            _ActionBranchPoint(
                branch_id=branch_id,
                seed=seed,
                fixture_name=fixture_name,
                branch_tick=tick,
                branch_index=branch_index,
                record_index=int(
                    row.record.get(TRAJECTORY_DATASET_RECORD_INDEX_FIELD, 0)
                ),
                agent_id=agent_id,
                logged_action=row.label,
                base_script=base_script,
                before=_state_excerpt(row.record.get("before")),
                action_mask={
                    action: action in row.valid_actions for action in ACTION_NAMES
                },
                observation_input=_observation_input_payload(
                    row.record.get("observation_input")
                ),
                observation_digest=_optional_string(
                    row.record.get("observation_digest")
                ),
                observation_schema=_optional_string(
                    row.record.get("observation_schema")
                ),
                context_snapshot=dict(row.context_snapshot),
                branch_state_digest=_branch_state_digest(
                    snapshot,
                    branch_id=branch_id,
                    branch_tick=tick,
                ),
                world=snapshot,
            )
        )
    return {
        "branch_points": branch_points,
        "report": {
            "seed": int(seed),
            "fixture": fixture_name,
            "base_script": base_script,
            "ticks_requested": int(ticks),
            "ticks_executed": int(ticks_executed),
            "ambiguous_branch_point_count": len(branch_points),
            "branch_ids": [point.branch_id for point in branch_points],
            "terminal_alive_after_discovery_run": len(world.alive_agents()),
            "births_after_discovery_run": int(world.births),
            "deaths_after_discovery_run": int(world.deaths),
        },
    }


def _evaluate_branch_point(
    point: _ActionBranchPoint,
    *,
    ticks: int,
    continuation_script: str,
    candidate_actions: Sequence[str],
    verify_replay: bool,
) -> dict[str, object]:
    action_runs = [
        _execute_action_branch(
            point,
            forced_action=action,
            continuation_script=continuation_script,
            ticks=ticks,
            verify_replay=verify_replay,
        )
        for action in candidate_actions
        if bool(point.action_mask.get(action, False))
    ]
    logged = next(
        (run for run in action_runs if run["forced_action"] == point.logged_action),
        None,
    )
    best = _best_action_run(action_runs)
    alive_delta = 0
    birth_delta = 0
    target_alive_delta = 0
    if best is not None and logged is not None:
        alive_delta = int(best["alive_agents"]) - int(logged["alive_agents"])
        birth_delta = int(best["births"]) - int(logged["births"])
        target_alive_delta = int(bool(best["target_alive_at_end"])) - int(
            bool(logged["target_alive_at_end"])
        )
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture_name,
        "branch_tick": point.branch_tick,
        "branch_index": point.branch_index,
        "record_index": point.record_index,
        "agent_id": point.agent_id,
        "base_script": point.base_script,
        "continuation_script": continuation_script,
        "logged_action": point.logged_action,
        "before": point.before,
        "context": {
            "post_carrion_contact": bool(
                point.context_snapshot.get("post_carrion_contact", False)
            ),
            "ticks_since_animal_resource_gain": (
                point.context_snapshot.get("ticks_since_animal_resource_gain")
            ),
            "ticks_since_drink": point.context_snapshot.get("ticks_since_drink"),
        },
        "valid_candidate_actions": [
            action for action in candidate_actions if bool(point.action_mask.get(action))
        ],
        "branch_state_digest": point.branch_state_digest,
        "action_runs": action_runs,
        "logged_action_run": _run_excerpt(logged),
        "oracle_best_action_run": _run_excerpt(best),
        "oracle_best_action": best["forced_action"] if best is not None else None,
        "oracle_changed_action": (
            bool(best is not None and best["forced_action"] != point.logged_action)
        ),
        "oracle_alive_delta_vs_logged": alive_delta,
        "oracle_birth_delta_vs_logged": birth_delta,
        "oracle_target_alive_delta_vs_logged": target_alive_delta,
        "material_oracle_gain": (
            alive_delta > 0 or birth_delta > 0 or target_alive_delta > 0
        ),
    }


def _execute_action_branch(
    point: _ActionBranchPoint,
    *,
    forced_action: str,
    continuation_script: str,
    ticks: int,
    verify_replay: bool,
) -> dict[str, object]:
    run, digest = _execute_once(
        point,
        forced_action=forced_action,
        continuation_script=continuation_script,
        ticks=ticks,
    )
    verification = None
    if verify_replay:
        replay, replay_digest = _execute_once(
            point,
            forced_action=forced_action,
            continuation_script=continuation_script,
            ticks=ticks,
        )
        verification = {
            "verified": replay_digest == digest,
            "expected_digest": digest,
            "actual_digest": replay_digest,
            "replay_alive_agents": int(replay["alive_agents"]),
            "replay_births": int(replay["births"]),
            "replay_deaths": int(replay["deaths"]),
        }
    run["replay_verification"] = verification
    return run


def _execute_once(
    point: _ActionBranchPoint,
    *,
    forced_action: str,
    continuation_script: str,
    ticks: int,
) -> tuple[dict[str, object], str]:
    world = deepcopy(point.world)
    policy = _ForcedFirstActionPolicy(
        target_agent_id=point.agent_id,
        forced_action=forced_action,
        continuation_script=continuation_script,
    )
    world.policy = policy
    _configure_manual_summary_run(world)
    population_horizon_trace: list[dict[str, object]] = []
    for tick in range(point.branch_tick, ticks):
        world.tick = tick
        world._run_tick()
        horizon_delta = tick - point.branch_tick
        if horizon_delta in DEFAULT_TARGET_HORIZON_TRACE_TICKS:
            population_horizon_trace.append(
                _population_horizon_snapshot(
                    world,
                    point=point,
                    forced_action=forced_action,
                    horizon_tick_delta=horizon_delta,
                )
            )
        if not world.alive_agents():
            break
    run = _summarize_world(
        world,
        point=point,
        forced_action=forced_action,
        continuation_script=continuation_script,
        forced_action_used=policy.used,
        ticks=ticks,
        population_horizon_trace=population_horizon_trace,
    )
    return run, stable_payload_digest(_digest_payload(run))


def _summarize_world(
    world: SimulationWorld,
    *,
    point: _ActionBranchPoint,
    forced_action: str,
    continuation_script: str,
    forced_action_used: bool,
    ticks: int,
    population_horizon_trace: Sequence[Mapping[str, object]],
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
    target = world.agents.get(point.agent_id)
    target_alive = bool(target is not None and target.alive)
    dominant = _dominant_action_summary(requested_action_counts)
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture_name,
        "ticks": int(ticks),
        "branch_tick": point.branch_tick,
        "agent_id": point.agent_id,
        "logged_action": point.logged_action,
        "forced_action": forced_action,
        "forced_action_used": bool(forced_action_used),
        "continuation_script": continuation_script,
        "branch_state_digest": point.branch_state_digest,
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "target_alive_at_end": target_alive,
        "target_energy_ratio_at_end": (
            _round(world._energy_ratio(target)) if target_alive else None
        ),
        "target_hydration_ratio_at_end": (
            _round(world._hydration_ratio(target)) if target_alive else None
        ),
        "population_horizon_trace": [dict(item) for item in population_horizon_trace],
        "outcome_metrics": build_run_outcome_metrics(
            summary=summary,
            trajectory_records=world.trajectory_records,
        ),
        "first_action_outcome": _first_target_action_outcome(
            world.trajectory_records,
            point=point,
            forced_action=forced_action,
        ),
        "target_horizon_trace": _target_horizon_trace(
            world.trajectory_records,
            point=point,
            forced_action=forced_action,
            horizons=DEFAULT_TARGET_HORIZON_TRACE_TICKS,
        ),
        "trajectory_record_count": len(world.trajectory_records),
        "heuristic_action_source_count": _heuristic_action_source_count(
            action_source_counts
        ),
        "zero_heuristic_runtime_actions": (
            _heuristic_action_source_count(action_source_counts) == 0
        ),
        "unique_requested_actions": len(requested_action_counts),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "dominant_requested_action": dominant["action"],
        "dominant_requested_action_count": dominant["count"],
        "dominant_requested_action_share": dominant["share"],
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
        "summary_excerpt": {
            "trophic_role_counts_at_end": _json_ready(
                summary.get("trophic_role_counts_at_end", {})
            ),
            "meat_mode_counts_at_end": _json_ready(
                summary.get("meat_mode_counts_at_end", {})
            ),
        },
    }


def _aggregate_results(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    oracle_changed = [
        result for result in branch_results if bool(result.get("oracle_changed_action"))
    ]
    material = [
        result for result in branch_results if bool(result.get("material_oracle_gain"))
    ]
    alive_gain_total = sum(
        max(0, int(result.get("oracle_alive_delta_vs_logged", 0)))
        for result in branch_results
    )
    birth_gain_total = sum(
        max(0, int(result.get("oracle_birth_delta_vs_logged", 0)))
        for result in branch_results
    )
    target_alive_gain_total = sum(
        max(0, int(result.get("oracle_target_alive_delta_vs_logged", 0)))
        for result in branch_results
    )
    all_runs = [
        run
        for result in branch_results
        for run in list(result.get("action_runs", []))
        if isinstance(run, Mapping)
    ]
    replay_items = [
        run.get("replay_verification")
        for run in all_runs
        if isinstance(run.get("replay_verification"), Mapping)
    ]
    winner_counts = Counter(
        str(result.get("oracle_best_action"))
        for result in branch_results
        if result.get("oracle_best_action") is not None
    )
    logged_counts = Counter(str(result.get("logged_action")) for result in branch_results)
    heuristic_count = sum(
        int(run.get("heuristic_action_source_count", 0)) for run in all_runs
    )
    return {
        "branch_point_count": len(branch_results),
        "action_run_count": len(all_runs),
        "oracle_changed_action_count": len(oracle_changed),
        "material_oracle_gain_count": len(material),
        "terminal_alive_gain_total_vs_logged": int(alive_gain_total),
        "birth_gain_total_vs_logged": int(birth_gain_total),
        "target_alive_gain_total_vs_logged": int(target_alive_gain_total),
        "oracle_best_action_counts": dict(sorted(winner_counts.items())),
        "logged_action_counts": dict(sorted(logged_counts.items())),
        "heuristic_action_source_count": int(heuristic_count),
        "zero_heuristic_runtime_actions": heuristic_count == 0,
        "replay_verification_count": len(replay_items),
        "replay_verified": bool(replay_items)
        and all(bool(item.get("verified", False)) for item in replay_items),
        "first_material_oracle_gain": _first_material(material),
        "outcome_metrics": aggregate_run_outcome_metrics(all_runs),
    }


def _acceptance(
    aggregate: Mapping[str, object],
    *,
    min_oracle_changed_action_count: int,
    min_terminal_alive_gain_total: int,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    if int(aggregate.get("branch_point_count", 0)) <= 0:
        blockers.append({"reason": "no_ambiguous_post_carrion_branch_points"})
    if (
        int(aggregate.get("oracle_changed_action_count", 0))
        < min_oracle_changed_action_count
    ):
        blockers.append(
            {
                "reason": "oracle_changed_action_count_below_floor",
                "required": int(min_oracle_changed_action_count),
                "observed": int(aggregate.get("oracle_changed_action_count", 0)),
            }
        )
    if (
        int(aggregate.get("terminal_alive_gain_total_vs_logged", 0))
        < min_terminal_alive_gain_total
    ):
        blockers.append(
            {
                "reason": "terminal_alive_gain_total_below_floor",
                "required": int(min_terminal_alive_gain_total),
                "observed": int(aggregate.get("terminal_alive_gain_total_vs_logged", 0)),
            }
        )
    if int(aggregate.get("heuristic_action_source_count", 0)) != 0:
        blockers.append(
            {
                "reason": "heuristic_action_source_count_nonzero",
                "observed": int(aggregate.get("heuristic_action_source_count", 0)),
            }
        )
    if not bool(aggregate.get("replay_verified", False)):
        blockers.append({"reason": "branch_replay_not_verified"})
    return {
        "diagnostic_acceptance_passed": not blockers,
        "materially_supports_branch_action_oracle": not blockers,
        "blockers": blockers,
    }


def _best_action_run(
    runs: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    if not runs:
        return None
    return max(
        runs,
        key=lambda run: (
            int(run.get("alive_agents", 0)),
            int(run.get("births", 0)),
            int(bool(run.get("target_alive_at_end", False))),
            -int(run.get("deaths", 0)),
            -float(run.get("dominant_requested_action_share", 1.0)),
            str(run.get("forced_action", "")),
        ),
    )


def _run_excerpt(run: Mapping[str, object] | None) -> dict[str, object] | None:
    if run is None:
        return None
    return {
        "forced_action": run.get("forced_action"),
        "forced_action_used": bool(run.get("forced_action_used", False)),
        "alive_agents": int(run.get("alive_agents", 0)),
        "births": int(run.get("births", 0)),
        "deaths": int(run.get("deaths", 0)),
        "target_alive_at_end": bool(run.get("target_alive_at_end", False)),
        "dominant_requested_action": run.get("dominant_requested_action"),
        "dominant_requested_action_share": run.get("dominant_requested_action_share"),
        "heuristic_action_source_count": int(run.get("heuristic_action_source_count", 0)),
    }


def _first_material(
    material_results: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not material_results:
        return None
    selected = material_results[0]
    return {
        "branch_id": selected.get("branch_id"),
        "seed": selected.get("seed"),
        "branch_tick": selected.get("branch_tick"),
        "agent_id": selected.get("agent_id"),
        "logged_action": selected.get("logged_action"),
        "oracle_best_action": selected.get("oracle_best_action"),
        "oracle_alive_delta_vs_logged": selected.get("oracle_alive_delta_vs_logged"),
        "oracle_birth_delta_vs_logged": selected.get("oracle_birth_delta_vs_logged"),
        "before": selected.get("before"),
    }


def _contextual_records(
    records: Sequence[Mapping[str, object]],
    *,
    seed: int,
) -> list[dict[str, object]]:
    contextual: list[dict[str, object]] = []
    episode_id = f"branch_action_oracle:seed={int(seed)}"
    source = f"branch-action-oracle-seed-{int(seed)}.jsonl.gz"
    for index, record in enumerate(records):
        row = dict(record)
        row[TRAJECTORY_EPISODE_ID_FIELD] = episode_id
        row[TRAJECTORY_DATASET_RECORD_INDEX_FIELD] = index
        row[TRAJECTORY_SOURCE_PATH_FIELD] = source
        contextual.append(row)
    return contextual


def _branch_id(
    *,
    fixture_name: str,
    seed: int,
    branch_index: int,
    tick: int,
    agent_id: int,
    logged_action: str,
) -> str:
    return (
        f"{_safe_path_part(fixture_name)}-seed-{int(seed)}-"
        f"action-branch-{int(branch_index)}-tick-{int(tick)}-"
        f"agent-{int(agent_id)}-logged-{_safe_path_part(logged_action)}"
    )


def _branch_point_payload(point: _ActionBranchPoint) -> dict[str, object]:
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture_name,
        "branch_tick": point.branch_tick,
        "branch_index": point.branch_index,
        "record_index": point.record_index,
        "agent_id": point.agent_id,
        "logged_action": point.logged_action,
        "base_script": point.base_script,
        "before": point.before,
        "context": {
            "post_carrion_contact": bool(
                point.context_snapshot.get("post_carrion_contact", False)
            ),
            "ticks_since_animal_resource_gain": (
                point.context_snapshot.get("ticks_since_animal_resource_gain")
            ),
            "ticks_since_drink": point.context_snapshot.get("ticks_since_drink"),
        },
        "valid_actions": [
            action for action in ACTION_NAMES if bool(point.action_mask.get(action))
        ],
        "policy_state": {
            "observation_input": point.observation_input,
            "observation_digest": point.observation_digest,
            "observation_schema": point.observation_schema,
            "action_mask": dict(point.action_mask),
        },
        "branch_state_digest": point.branch_state_digest,
    }


def _digest_payload(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": run.get("branch_id"),
        "seed": run.get("seed"),
        "fixture": run.get("fixture"),
        "ticks": run.get("ticks"),
        "branch_tick": run.get("branch_tick"),
        "agent_id": run.get("agent_id"),
        "logged_action": run.get("logged_action"),
        "forced_action": run.get("forced_action"),
        "forced_action_used": run.get("forced_action_used"),
        "continuation_script": run.get("continuation_script"),
        "branch_state_digest": run.get("branch_state_digest"),
        "ticks_executed": run.get("ticks_executed"),
        "alive_agents": run.get("alive_agents"),
        "births": run.get("births"),
        "deaths": run.get("deaths"),
        "target_alive_at_end": run.get("target_alive_at_end"),
        "first_action_outcome": run.get("first_action_outcome"),
        "target_horizon_trace": run.get("target_horizon_trace"),
        "population_horizon_trace": run.get("population_horizon_trace"),
        "outcome_metrics": run.get("outcome_metrics"),
        "requested_action_counts": run.get("requested_action_counts"),
        "action_source_counts": run.get("action_source_counts"),
        "policy_id_counts": run.get("policy_id_counts"),
    }


def _contract(
    *,
    seeds: Sequence[int],
    ticks: int,
    fixture_name: str,
    base_script: str,
    continuation_script: str,
    candidate_actions: Sequence[str],
    target_labels: Sequence[str],
    max_branch_points_per_seed: int,
    min_branch_tick: int,
    min_oracle_changed_action_count: int,
    min_terminal_alive_gain_total: int,
    verify_replay: bool,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_POLICY,
        "fixture_name": fixture_name,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "base_script": base_script,
        "continuation_script": continuation_script,
        "candidate_actions": list(candidate_actions),
        "target_labels": list(target_labels),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "min_branch_tick": int(min_branch_tick),
        "branch_trigger": (
            "post-carrion rows where drink and eat are both legal and logged "
            "action is in target_labels"
        ),
        "branch_timing": "pre_tick_before_ambiguous_decision_record_v1",
        "min_oracle_changed_action_count": int(min_oracle_changed_action_count),
        "min_terminal_alive_gain_total": int(min_terminal_alive_gain_total),
        "verify_replay": bool(verify_replay),
    }


def _validated_actions(actions: Sequence[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(str(action) for action in actions if str(action)))
    if not values:
        raise BranchActionOracleAuditError("at least one action is required")
    unsupported = sorted(action for action in values if action not in ACTION_NAMES)
    if unsupported:
        raise BranchActionOracleAuditError(
            "unsupported action(s): " + ", ".join(unsupported)
        )
    return values


def _validated_script(script: str) -> None:
    if script not in DEFAULT_COUNTERFACTUAL_SCRIPTS:
        raise BranchActionOracleAuditError(f"unsupported script: {script}")


def _observation_input_payload(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_target_action_outcome(
    trajectory_records: Sequence[Mapping[str, object]],
    *,
    point: _ActionBranchPoint,
    forced_action: str,
) -> dict[str, object]:
    record = next(
        (
            item
            for item in trajectory_records
            if _optional_int(item.get("tick")) == point.branch_tick
            and _optional_int(item.get("agent_id")) == point.agent_id
        ),
        None,
    )
    if record is None:
        return {
            "record_found": False,
            "branch_tick": point.branch_tick,
            "agent_id": point.agent_id,
            "forced_action": forced_action,
        }
    return _target_record_outcome(
        record,
        point=point,
        forced_action=forced_action,
        horizon_tick_delta=0,
    )


def _target_horizon_trace(
    trajectory_records: Sequence[Mapping[str, object]],
    *,
    point: _ActionBranchPoint,
    forced_action: str,
    horizons: Sequence[int],
) -> list[dict[str, object]]:
    records_by_tick = {
        _optional_int(record.get("tick")): record
        for record in trajectory_records
        if _optional_int(record.get("agent_id")) == point.agent_id
    }
    trace = []
    for horizon in horizons:
        tick_delta = max(0, int(horizon))
        record = records_by_tick.get(point.branch_tick + tick_delta)
        if record is None:
            trace.append(
                {
                    "record_found": False,
                    "horizon_tick_delta": tick_delta,
                    "tick": point.branch_tick + tick_delta,
                    "agent_id": point.agent_id,
                    "forced_action": forced_action,
                }
            )
            continue
        trace.append(
            _target_record_outcome(
                record,
                point=point,
                forced_action=forced_action,
                horizon_tick_delta=tick_delta,
            )
        )
    return trace


def _target_record_outcome(
    record: Mapping[str, object],
    *,
    point: _ActionBranchPoint,
    forced_action: str,
    horizon_tick_delta: int,
) -> dict[str, object]:
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    outcome = _mapping(record.get("outcome"))
    drinking = _mapping(outcome.get("drinking"))
    feeding = _mapping(outcome.get("feeding"))
    passive = _mapping(outcome.get("passive"))
    return {
        "record_found": True,
        "horizon_tick_delta": int(horizon_tick_delta),
        "tick": _optional_int(record.get("tick")),
        "branch_tick": point.branch_tick,
        "agent_id": point.agent_id,
        "forced_action": forced_action,
        "requested_action": _optional_string(record.get("requested_action")),
        "resolved_action": _optional_string(record.get("resolved_action")),
        "action_valid": bool(record.get("action_valid", False)),
        "resolution_action_valid": bool(
            record.get("resolution_action_valid", False)
        ),
        "moved": bool(record.get("moved", False)),
        "alive_before": bool(before.get("alive", False)),
        "alive_after": bool(after.get("alive", False)),
        "x_delta": _optional_int(after.get("x")) - _optional_int(before.get("x")),
        "y_delta": _optional_int(after.get("y")) - _optional_int(before.get("y")),
        "energy_ratio_before": _optional_float(before.get("energy_ratio")),
        "energy_ratio_after": _optional_float(after.get("energy_ratio")),
        "energy_ratio_delta": _delta(after, before, "energy_ratio"),
        "hydration_ratio_before": _optional_float(before.get("hydration_ratio")),
        "hydration_ratio_after": _optional_float(after.get("hydration_ratio")),
        "hydration_ratio_delta": _delta(after, before, "hydration_ratio"),
        "health_ratio_before": _optional_float(before.get("health_ratio")),
        "health_ratio_after": _optional_float(after.get("health_ratio")),
        "health_ratio_delta": _delta(after, before, "health_ratio"),
        "resource_gain": _optional_float(outcome.get("resource_gain")),
        "drank": bool(drinking.get("drank", False)),
        "ate": bool(feeding.get("ate", False)),
        "died": bool(outcome.get("died", False)),
        "death_cause": _optional_string(passive.get("death_cause")),
        "died_after_action": bool(passive.get("died_after_action", False)),
    }


def _population_horizon_snapshot(
    world: SimulationWorld,
    *,
    point: _ActionBranchPoint,
    forced_action: str,
    horizon_tick_delta: int,
) -> dict[str, object]:
    requested_action_counts = Counter(
        str(record.get("requested_action"))
        for record in world.tick_trajectory_records
        if isinstance(record.get("requested_action"), str)
    )
    dominant = _dominant_action_summary(requested_action_counts)
    target = world.agents.get(point.agent_id)
    target_alive = bool(target is not None and target.alive)
    tick_resource_gain = 0.0
    for record in world.tick_trajectory_records:
        outcome = _mapping(record.get("outcome"))
        value = outcome.get("resource_gain")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            tick_resource_gain += float(value)
    return {
        "horizon_tick_delta": int(horizon_tick_delta),
        "tick": int(world.tick),
        "branch_tick": point.branch_tick,
        "agent_id": point.agent_id,
        "forced_action": forced_action,
        "alive_agents": len(world.alive_agents()),
        "births": int(world.births),
        "deaths": int(world.deaths),
        "target_alive": target_alive,
        "target_energy_ratio": (
            _round(world._energy_ratio(target)) if target_alive else None
        ),
        "target_hydration_ratio": (
            _round(world._hydration_ratio(target)) if target_alive else None
        ),
        "target_health_ratio": (
            _round(world._health_ratio(target)) if target_alive else None
        ),
        "tick_resource_gain": _round(tick_resource_gain),
        "tick_trajectory_record_count": len(world.tick_trajectory_records),
        "tick_requested_action_counts": dict(sorted(requested_action_counts.items())),
        "tick_dominant_requested_action": dominant["action"],
        "tick_dominant_requested_action_share": dominant["share"],
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return _round(float(value))


def _delta(
    after: Mapping[str, object],
    before: Mapping[str, object],
    field: str,
) -> float | None:
    after_value = _optional_float(after.get(field))
    before_value = _optional_float(before.get(field))
    if after_value is None or before_value is None:
        return None
    return _round(after_value - before_value)


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
