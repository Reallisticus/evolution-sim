from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import (
    ACTION_NAMES,
    CORE_ACTIONS,
    MOVEMENT_ACTIONS,
)
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.mind.broad_transfer_residual_audit import (
    V98_STRICT_EXCLUDED_SEEDS,
    _complete_action_mask,
    _project_public_history,
    _public_history_item_from_record,
    _record_consumed_animal_resource,
    _record_drank,
    _record_requested_action,
    _vitals_snapshot,
)
from evolution_sim.mind.branch_action_oracle_audit import (
    DEFAULT_TARGET_HORIZON_TRACE_TICKS,
    _first_target_action_outcome,
    _population_horizon_snapshot,
    _target_horizon_trace,
)
from evolution_sim.mind.branch_mode_objective_audit import _action_option_mode
from evolution_sim.mind.branch_utility_risk_audit import (
    _field_summary,
    _target_local_scalar,
    _utility_comparison,
)
from evolution_sim.mind.carrion_branch_explore import (
    _branch_state_digest,
    _configure_manual_summary_run,
)
from evolution_sim.mind.evaluation_helpers import (
    dominant_action_summary as _dominant_action_summary,
    heuristic_action_source_count as _heuristic_action_source_count,
    json_ready as _json_ready,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_planner_distilled import planner_distilled_runtime_row
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION = (
    "mind_v3_v99_broad_branch_residual_oracle_audit_v1"
)
MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_POLICY = (
    "v99_broad_linear_residual_branch_oracle_v1"
)
V99_DEFAULT_SUPPORT_SEEDS: tuple[int, ...] = (
    2,
    3,
    7,
    11,
    17,
    23,
    31,
    47,
    53,
    59,
)
V99_DEFAULT_TICKS = 120
V99_DEFAULT_BRANCH_POINTS_PER_SEED = 4
V99_MIN_BRANCH_POINTS = 40
V99_MIN_SOURCE_SEEDS = 8
V99_MIN_SAFE_OVERRIDE_COUNT = 8
V99_MIN_SAFE_OVERRIDE_SHARE = 0.20
V99_MAX_DOMINANT_ORACLE_ACTION_SHARE = 0.50
_BRANCH_CANDIDATE_ACTIONS = (*CORE_ACTIONS, *MOVEMENT_ACTIONS)


class BroadBranchResidualOracleAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _BroadBranchCandidate:
    seed: int
    tick: int
    record_index: int
    agent_id: int
    logged_action: str
    before: dict[str, object]
    action_mask: dict[str, bool]
    observation_input: dict[str, object]
    observation_digest: str | None
    observation_schema: str | None
    public_history_trace: tuple[dict[str, object], ...]
    compact_state: dict[str, object]
    legal_actions: tuple[str, ...]
    categories: tuple[str, ...]
    priority: float


@dataclass(frozen=True, slots=True)
class _BroadBranchPoint:
    branch_id: str
    seed: int
    fixture_name: str
    branch_tick: int
    branch_index: int
    record_index: int
    agent_id: int
    logged_action: str
    before: dict[str, object]
    action_mask: dict[str, bool]
    observation_input: dict[str, object]
    observation_digest: str | None
    observation_schema: str | None
    public_history_trace: tuple[dict[str, object], ...]
    compact_state: dict[str, object]
    categories: tuple[str, ...]
    branch_state_digest: str
    world: SimulationWorld


class _ForcedFirstActionThenDelegatePolicy:
    policy_id = "mind_v3_v99_broad_branch_oracle_forced_first_action"
    policy_version = "mind_v3_v99_broad_branch_oracle_forced_first_action_v1"

    def __init__(
        self,
        *,
        target_agent_id: int,
        forced_action: str,
        delegate: object,
    ) -> None:
        self.target_agent_id = int(target_agent_id)
        self.forced_action = str(forced_action)
        self.delegate = delegate
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
                    "diagnostic_only": True,
                    "heuristic_free": True,
                },
            )
        decide = getattr(self.delegate, "decide", None)
        if not callable(decide):
            raise BroadBranchResidualOracleAuditError(
                "branch delegate policy does not implement decide"
            )
        return decide(observation, action_mask)

    def observe_transition(self, record: dict[str, object]) -> dict[str, object] | None:
        observe = getattr(self.delegate, "observe_transition", None)
        if not callable(observe):
            return None
        feedback = dict(record)
        if (
            int(feedback.get("agent_id", -1)) == self.target_agent_id
            and feedback.get("action_source") == f"branch_oracle_force:{self.forced_action}"
        ):
            feedback["policy_id"] = getattr(self.delegate, "policy_id", None)
            feedback["policy_version"] = getattr(self.delegate, "policy_version", None)
        return observe(feedback)


def build_broad_branch_residual_oracle_audit_report(
    *,
    seeds: Sequence[int] = V99_DEFAULT_SUPPORT_SEEDS,
    ticks: int = V99_DEFAULT_TICKS,
    max_branch_points_per_seed: int = V99_DEFAULT_BRANCH_POINTS_PER_SEED,
    min_branch_points: int = V99_MIN_BRANCH_POINTS,
    verify_replay: bool = True,
) -> dict[str, object]:
    seed_values = _validated_non_strict_seeds(seeds)
    tick_count = _positive_int(ticks, field="ticks")
    branch_limit = _positive_int(
        max_branch_points_per_seed,
        field="max_branch_points_per_seed",
    )
    min_points = _positive_int(min_branch_points, field="min_branch_points")
    contract = {
        "schema_version": MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_POLICY,
        "diagnostic_only": True,
        "runtime_policy_trained": False,
        "promotion_run_executed": False,
        "ticks": tick_count,
        "support_seeds": list(seed_values),
        "strict_excluded_seeds": list(V98_STRICT_EXCLUDED_SEEDS),
        "max_branch_points_per_seed": branch_limit,
        "min_branch_points": min_points,
        "candidate_actions": list(_BRANCH_CANDIDATE_ACTIONS),
        "branch_selection_policy": (
            "policy_visible_broad_failure_recovery_balanced_v1"
        ),
        "continuation_policy": "copied_linear_mind_v3_policy_state_v1",
        "replay_verification_required": bool(verify_replay),
        "feature_contract": {
            "uses_policy_visible_observation_input": True,
            "uses_action_mask": True,
            "uses_public_history_trace": True,
            "uses_seed_id_as_runtime_feature": False,
            "uses_fixture_identity": False,
            "uses_branch_id": False,
            "uses_logged_action_as_runtime_fallback": False,
            "uses_private_world_state_as_runtime_input": False,
            "uses_simulator_in_loop_for_diagnostic_labels": True,
        },
        "support_floors": {
            "no_strict_seed_leakage": True,
            "replay_verified": True,
            "heuristic_action_source_count": 0,
            "unsupported_candidate_action_count": 0,
            "branch_point_count": min_points,
            "source_seed_count": V99_MIN_SOURCE_SEEDS,
            "safe_non_logged_override_count": V99_MIN_SAFE_OVERRIDE_COUNT,
            "safe_non_logged_override_share": V99_MIN_SAFE_OVERRIDE_SHARE,
            "dominant_oracle_action_share_max": V99_MAX_DOMINANT_ORACLE_ACTION_SHARE,
            "target_alive_delta_negative_count": 0,
            "mean_target_local_score_delta_gt": 0.0,
            "mean_terminal_alive_delta_min": 0.0,
            "mean_birth_delta_min": 0.0,
        },
    }
    discovery: list[dict[str, object]] = []
    branch_points: list[_BroadBranchPoint] = []
    for seed in seed_values:
        candidates, scan_report = _scan_broad_branch_candidates(
            seed=seed,
            ticks=tick_count,
        )
        selected = select_broad_branch_candidates(
            candidates,
            max_branch_points=branch_limit,
        )
        points = _materialize_branch_points(
            seed=seed,
            ticks=tick_count,
            selected=selected,
        )
        discovery.append(
            {
                **scan_report,
                "selected_branch_point_count": len(points),
                "selected_branch_ids": [point.branch_id for point in points],
                "selected_categories": dict(
                    sorted(Counter(cat for point in points for cat in point.categories).items())
                ),
            }
        )
        branch_points.extend(points)

    branch_results = [
        _evaluate_branch_point(
            point,
            ticks=tick_count,
            verify_replay=verify_replay,
        )
        for point in branch_points
    ]
    aggregate = _aggregate_results(
        branch_points=branch_points,
        branch_results=branch_results,
        seeds=seed_values,
    )
    acceptance = _acceptance(
        aggregate=aggregate,
        floors=contract["support_floors"],
    )
    accepted = bool(acceptance["v99_broad_branch_residual_oracle_accepted"])
    support_probe = {
        "policy": MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_POLICY,
        "accuracy": 1.0 if accepted else 0.0,
        "support_accuracy_floor": 1.0,
        "materially_supports_v100_residual_distillation": accepted,
        "runtime_policy_status": (
            "v100_distillation_allowed"
            if accepted
            else "rejected_no_runtime_policy"
        ),
        "branch_point_count": aggregate["branch_point_count"],
        "safe_non_logged_override_count": aggregate[
            "safe_non_logged_override_count"
        ],
        "safe_non_logged_override_share": aggregate[
            "safe_non_logged_override_share"
        ],
        "mean_target_local_score_delta": aggregate[
            "target_local_score_delta_summary"
        ]["mean"],
        "blocker_count": acceptance["blocker_count"],
    }
    return {
        "schema_version": MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_POLICY,
        "contract": contract,
        "provenance": {
            "contract_digest": stable_payload_digest(contract),
        },
        "discovery": discovery,
        "branch_points": [_branch_point_payload(point) for point in branch_points],
        "branch_results": branch_results,
        "aggregate": aggregate,
        "broad_branch_residual_oracle_support_probe": support_probe,
        "acceptance": acceptance,
        "v99_broad_branch_residual_oracle_accepted": accepted,
        "v100_residual_distillation_allowed": bool(
            acceptance["v100_residual_distillation_allowed"]
        ),
        "blocker_count": int(acceptance["blocker_count"]),
    }


def write_broad_branch_residual_oracle_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def select_broad_branch_candidates(
    candidates: Sequence[_BroadBranchCandidate],
    *,
    max_branch_points: int,
) -> list[_BroadBranchCandidate]:
    limit = _positive_int(max_branch_points, field="max_branch_points")
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.priority,
            item.tick,
            item.agent_id,
            item.logged_action,
            item.record_index,
        ),
    )
    selected: list[_BroadBranchCandidate] = []
    seen: set[tuple[int, int, int]] = set()

    def add(candidate: _BroadBranchCandidate) -> bool:
        key = (candidate.tick, candidate.agent_id, candidate.record_index)
        if key in seen:
            return False
        selected.append(candidate)
        seen.add(key)
        return True

    category_order = (
        "pre_death",
        "recovery",
        "movement",
        "hydration",
        "energy",
        "animal_resource",
        "reproduction_readiness",
    )
    made_progress = True
    while made_progress and len(selected) < limit:
        made_progress = False
        for category in category_order:
            for candidate in ordered:
                if category in candidate.categories and add(candidate):
                    made_progress = True
                    if len(selected) >= limit:
                        return selected
                    break
    for candidate in ordered:
        if add(candidate):
            if len(selected) >= limit:
                break
    return selected


def _scan_broad_branch_candidates(
    *,
    seed: int,
    ticks: int,
) -> tuple[list[_BroadBranchCandidate], dict[str, object]]:
    world = SimulationWorld(
        WorldConfig(seed=seed, max_ticks=ticks),
        policy=MindV3EvolutionPolicy(seed=seed),
    )
    _configure_manual_summary_run(world)
    history_by_agent: dict[int, list[dict[str, object]]] = {}
    ticks_since_resource: dict[int, int | None] = {}
    ticks_since_drink: dict[int, int | None] = {}
    candidates: list[_BroadBranchCandidate] = []
    skipped: Counter[str] = Counter()
    tick_count = 0
    record_index = 0
    for tick in range(ticks):
        world.tick = tick
        world._run_tick()
        tick_count = tick + 1
        for record in world.tick_trajectory_records:
            agent_id = _int(record.get("agent_id"))
            history = _project_public_history(
                history_by_agent.get(agent_id, []),
                current_tick=_int(record.get("tick")),
                current_record_index=record_index,
            )
            candidate, reason = _candidate_from_record(
                record,
                seed=seed,
                record_index=record_index,
                public_history_trace=history,
                ticks=ticks,
            )
            if candidate is not None:
                candidates.append(candidate)
            else:
                skipped.update([reason])
            item = _public_history_item_from_record(
                record,
                record_index=record_index,
                ticks_since_animal_resource_gain=ticks_since_resource.get(agent_id),
                ticks_since_drink=ticks_since_drink.get(agent_id),
            )
            agent_history = history_by_agent.setdefault(agent_id, [])
            agent_history.append(item)
            if len(agent_history) > 8:
                del agent_history[0 : len(agent_history) - 8]
            if _record_consumed_animal_resource(record):
                ticks_since_resource[agent_id] = 0
            else:
                previous = ticks_since_resource.get(agent_id)
                ticks_since_resource[agent_id] = (
                    None if previous is None else previous + 1
                )
            if _record_drank(record):
                ticks_since_drink[agent_id] = 0
            else:
                previous_drink = ticks_since_drink.get(agent_id)
                ticks_since_drink[agent_id] = (
                    None if previous_drink is None else previous_drink + 1
                )
            record_index += 1
        if not world.alive_agents():
            break
    by_category = Counter(cat for candidate in candidates for cat in candidate.categories)
    return candidates, {
        "seed": int(seed),
        "ticks_requested": int(ticks),
        "ticks_executed": tick_count,
        "candidate_count": len(candidates),
        "skipped_counts": dict(sorted(skipped.items())),
        "candidate_categories": dict(sorted(by_category.items())),
        "candidate_logged_actions": dict(
            sorted(Counter(item.logged_action for item in candidates).items())
        ),
    }


def _candidate_from_record(
    record: Mapping[str, object],
    *,
    seed: int,
    record_index: int,
    public_history_trace: Sequence[Mapping[str, object]],
    ticks: int,
) -> tuple[_BroadBranchCandidate | None, str]:
    if str(record.get("action_source", "")).startswith("passive"):
        return None, "passive_record"
    requested = _record_requested_action(record)
    if requested not in _BRANCH_CANDIDATE_ACTIONS:
        return None, "logged_action_not_active"
    if record.get("action_valid") is not True:
        return None, "logged_action_invalid"
    action_mask = _complete_action_mask(_mapping(record.get("action_mask")))
    legal = tuple(
        action
        for action in _BRANCH_CANDIDATE_ACTIONS
        if bool(action_mask.get(action, False))
    )
    if requested not in legal:
        return None, "logged_action_not_legal"
    if len(legal) < 2:
        return None, "insufficient_legal_alternatives"
    observation_input = dict(_mapping(record.get("observation_input")))
    runtime_row = planner_distilled_runtime_row(
        observation_input=observation_input,
        action_mask=action_mask,
        public_history_trace=public_history_trace,
    )
    compact_state = dict(_mapping(runtime_row.get("compact_state")))
    if not compact_state:
        return None, "observation_not_decodable"
    before = _vitals_snapshot(_mapping(record.get("before")))
    categories = _branch_categories(
        before=before,
        compact_state=compact_state,
        legal_actions=legal,
    )
    if not categories:
        return None, "low_value_broad_state"
    priority = _branch_priority(
        tick=_int(record.get("tick")),
        ticks=ticks,
        before=before,
        compact_state=compact_state,
        legal_actions=legal,
        categories=categories,
    )
    return (
        _BroadBranchCandidate(
            seed=int(seed),
            tick=_int(record.get("tick")),
            record_index=int(record_index),
            agent_id=_int(record.get("agent_id")),
            logged_action=requested,
            before=before,
            action_mask=action_mask,
            observation_input=observation_input,
            observation_digest=_optional_string(record.get("observation_digest")),
            observation_schema=_optional_string(record.get("observation_schema")),
            public_history_trace=tuple(dict(item) for item in public_history_trace),
            compact_state=compact_state,
            legal_actions=legal,
            categories=tuple(categories),
            priority=_round(priority),
        ),
        "selected",
    )


def _materialize_branch_points(
    *,
    seed: int,
    ticks: int,
    selected: Sequence[_BroadBranchCandidate],
) -> list[_BroadBranchPoint]:
    selected_by_key = {
        (candidate.tick, candidate.agent_id, candidate.record_index): candidate
        for candidate in selected
    }
    if not selected_by_key:
        return []
    world = SimulationWorld(
        WorldConfig(seed=seed, max_ticks=ticks),
        policy=MindV3EvolutionPolicy(seed=seed),
    )
    _configure_manual_summary_run(world)
    points: list[_BroadBranchPoint] = []
    record_index = 0
    for tick in range(ticks):
        world.tick = tick
        snapshot = deepcopy(world)
        world._run_tick()
        for record in world.tick_trajectory_records:
            key = (tick, _int(record.get("agent_id")), record_index)
            candidate = selected_by_key.get(key)
            if candidate is not None:
                if _record_requested_action(record) != candidate.logged_action:
                    raise BroadBranchResidualOracleAuditError(
                        "deterministic materialization mismatch for selected record"
                    )
                branch_index = len(points)
                branch_id = (
                    f"broad-seed-{int(seed)}-branch-{branch_index}-"
                    f"tick-{tick}-agent-{candidate.agent_id}-"
                    f"logged-{_safe_path_part(candidate.logged_action)}"
                )
                points.append(
                    _BroadBranchPoint(
                        branch_id=branch_id,
                        seed=int(seed),
                        fixture_name="broad",
                        branch_tick=tick,
                        branch_index=branch_index,
                        record_index=record_index,
                        agent_id=candidate.agent_id,
                        logged_action=candidate.logged_action,
                        before=dict(candidate.before),
                        action_mask=dict(candidate.action_mask),
                        observation_input=dict(candidate.observation_input),
                        observation_digest=candidate.observation_digest,
                        observation_schema=candidate.observation_schema,
                        public_history_trace=tuple(
                            dict(item) for item in candidate.public_history_trace
                        ),
                        compact_state=dict(candidate.compact_state),
                        categories=tuple(candidate.categories),
                        branch_state_digest=_branch_state_digest(
                            snapshot,
                            branch_id=branch_id,
                            branch_tick=tick,
                        ),
                        world=snapshot,
                    )
                )
            record_index += 1
        if len(points) >= len(selected_by_key):
            break
        if not world.alive_agents():
            break
    if len(points) != len(selected_by_key):
        raise BroadBranchResidualOracleAuditError(
            "failed to materialize all selected broad branch points"
        )
    return points


def _evaluate_branch_point(
    point: _BroadBranchPoint,
    *,
    ticks: int,
    verify_replay: bool,
) -> dict[str, object]:
    action_runs = [
        _execute_action_branch(
            point,
            forced_action=action,
            ticks=ticks,
            verify_replay=verify_replay,
        )
        for action in _legal_branch_actions(point.action_mask)
    ]
    logged = _action_run(action_runs, point.logged_action)
    if logged is None:
        raise BroadBranchResidualOracleAuditError(
            f"logged action missing from branch runs: {point.logged_action}"
        )
    row = {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "logged_action": point.logged_action,
        "before": point.before,
        "action_mask": point.action_mask,
        "target_local_action": None,
    }
    best = max(
        action_runs,
        key=lambda run: (
            _target_local_scalar(run, row),
            _int(run.get("alive_agents")),
            _int(run.get("births")),
            -_int(run.get("deaths")),
            str(run.get("forced_action", "")),
        ),
    )
    comparison = _utility_comparison(
        row,
        rule=MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_POLICY,
        predicted_action=str(best.get("forced_action", "")),
        predicted_run=best,
        logged_run=logged,
    )
    safe_non_logged = (
        comparison["predicted_action"] != point.logged_action
        and _float(comparison.get("target_local_score_delta")) > 0.0
        and _float(comparison.get("target_alive_delta")) >= 0.0
        and _float(comparison.get("terminal_alive_delta")) >= 0.0
        and _float(comparison.get("birth_delta")) >= 0.0
    )
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture_name,
        "branch_tick": point.branch_tick,
        "branch_index": point.branch_index,
        "record_index": point.record_index,
        "agent_id": point.agent_id,
        "logged_action": point.logged_action,
        "categories": list(point.categories),
        "branch_state_digest": point.branch_state_digest,
        "action_runs": action_runs,
        "logged_action_run": _run_excerpt(logged),
        "target_local_oracle_action_run": _run_excerpt(best),
        "target_local_oracle_action": best.get("forced_action"),
        "target_local_oracle_mode": _action_option_mode(
            str(best.get("forced_action", ""))
        ),
        "oracle_changed_action": bool(best.get("forced_action") != point.logged_action),
        "safe_non_logged_override": bool(safe_non_logged),
        "utility_comparison_vs_logged": comparison,
    }


def _execute_action_branch(
    point: _BroadBranchPoint,
    *,
    forced_action: str,
    ticks: int,
    verify_replay: bool,
) -> dict[str, object]:
    run, digest = _execute_once(point, forced_action=forced_action, ticks=ticks)
    verification = None
    if verify_replay:
        replay, replay_digest = _execute_once(
            point,
            forced_action=forced_action,
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
    point: _BroadBranchPoint,
    *,
    forced_action: str,
    ticks: int,
) -> tuple[dict[str, object], str]:
    world = deepcopy(point.world)
    delegate = world.policy
    policy = _ForcedFirstActionThenDelegatePolicy(
        target_agent_id=point.agent_id,
        forced_action=forced_action,
        delegate=delegate,
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
    run = _summarize_branch_world(
        world,
        point=point,
        forced_action=forced_action,
        ticks=ticks,
        forced_action_used=policy.used,
        population_horizon_trace=population_horizon_trace,
    )
    return run, stable_payload_digest(_digest_payload(run))


def _summarize_branch_world(
    world: SimulationWorld,
    *,
    point: _BroadBranchPoint,
    forced_action: str,
    ticks: int,
    forced_action_used: bool,
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
    first_outcome = _first_target_action_outcome(
        world.trajectory_records,
        point=point,
        forced_action=forced_action,
    )
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
        "forced_action_supported": bool(point.action_mask.get(forced_action, False)),
        "branch_state_digest": point.branch_state_digest,
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "terminal_alive_agents": int(summary["alive_agents"]),
        "target_alive_at_end": target_alive,
        "target_energy_ratio_at_end": (
            _round(world._energy_ratio(target)) if target_alive else None
        ),
        "target_hydration_ratio_at_end": (
            _round(world._hydration_ratio(target)) if target_alive else None
        ),
        "target_health_ratio_at_end": (
            _round(world._health_ratio(target)) if target_alive else None
        ),
        "population_horizon_trace": [dict(item) for item in population_horizon_trace],
        "first_action_outcome": first_outcome,
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
    *,
    branch_points: Sequence[_BroadBranchPoint],
    branch_results: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
) -> dict[str, object]:
    comparisons = [
        _mapping(result.get("utility_comparison_vs_logged"))
        for result in branch_results
    ]
    oracle_actions = [
        str(result.get("target_local_oracle_action", ""))
        for result in branch_results
    ]
    oracle_modes = [_action_option_mode(action) for action in oracle_actions]
    replay_items = [
        run.get("replay_verification")
        for result in branch_results
        for run in _list_of_mappings(result.get("action_runs"))
        if run.get("replay_verification") is not None
    ]
    replay_verified = bool(replay_items) and all(
        bool(_mapping(item).get("verified")) for item in replay_items
    )
    action_runs = [
        run
        for result in branch_results
        for run in _list_of_mappings(result.get("action_runs"))
    ]
    unsupported = sum(
        1
        for run in action_runs
        if run.get("forced_action_used") is not True
        or run.get("forced_action_supported") is not True
        or _mapping(run.get("first_action_outcome")).get("action_valid") is False
    )
    heuristic_count = sum(
        _int(run.get("heuristic_action_source_count")) for run in action_runs
    )
    safe_overrides = [
        result
        for result in branch_results
        if result.get("safe_non_logged_override") is True
    ]
    by_seed: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for comparison in comparisons:
        by_seed[str(comparison.get("seed"))].append(comparison)
    action_counts = Counter(oracle_actions)
    mode_counts = Counter(oracle_modes)
    dominant_action = _dominant_count_share(action_counts)
    dominant_mode = _dominant_count_share(mode_counts)
    target_alive_negative = sum(
        1 for item in comparisons if _float(item.get("target_alive_delta")) < 0.0
    )
    return {
        "seed_count": len(seeds),
        "source_seed_count": len({int(point.seed) for point in branch_points}),
        "source_seeds": sorted({int(point.seed) for point in branch_points}),
        "strict_seed_leak_count": len(
            set(int(point.seed) for point in branch_points)
            & set(V98_STRICT_EXCLUDED_SEEDS)
        ),
        "branch_point_count": len(branch_points),
        "branch_result_count": len(branch_results),
        "action_run_count": len(action_runs),
        "comparison_count": len(comparisons),
        "replay_verification_count": len(replay_items),
        "replay_verified": replay_verified,
        "heuristic_action_source_count": heuristic_count,
        "unsupported_candidate_action_count": unsupported,
        "safe_non_logged_override_count": len(safe_overrides),
        "safe_non_logged_override_share": _safe_rate(
            len(safe_overrides),
            len(branch_results),
        ),
        "oracle_changed_action_count": sum(
            1 for result in branch_results if result.get("oracle_changed_action") is True
        ),
        "oracle_action_counts": dict(sorted(action_counts.items())),
        "oracle_mode_counts": dict(sorted(mode_counts.items())),
        "dominant_oracle_action": dominant_action["key"],
        "dominant_oracle_action_count": dominant_action["count"],
        "dominant_oracle_action_share": dominant_action["share"],
        "dominant_oracle_mode": dominant_mode["key"],
        "dominant_oracle_mode_count": dominant_mode["count"],
        "dominant_oracle_mode_share": dominant_mode["share"],
        "target_alive_delta_negative_count": target_alive_negative,
        "target_local_score_delta_summary": _field_summary(
            comparisons,
            "target_local_score_delta",
        ),
        "terminal_alive_delta_summary": _field_summary(
            comparisons,
            "terminal_alive_delta",
        ),
        "birth_delta_summary": _field_summary(comparisons, "birth_delta"),
        "per_seed": {
            seed: {
                "comparison_count": len(items),
                "target_local_score_delta": _round(
                    _mean([_float(item.get("target_local_score_delta")) for item in items])
                ),
                "terminal_alive_delta": _round(
                    _mean([_float(item.get("terminal_alive_delta")) for item in items])
                ),
                "birth_delta": _round(
                    _mean([_float(item.get("birth_delta")) for item in items])
                ),
                "target_alive_delta_negative_count": sum(
                    1 for item in items if _float(item.get("target_alive_delta")) < 0.0
                ),
            }
            for seed, items in sorted(by_seed.items(), key=lambda item: int(item[0]))
        },
        "worst_examples": sorted(
            [
                {
                    "branch_id": item.get("branch_id"),
                    "seed": item.get("seed"),
                    "logged_action": item.get("logged_action"),
                    "predicted_action": item.get("predicted_action"),
                    "target_local_score_delta": item.get("target_local_score_delta"),
                    "terminal_alive_delta": item.get("terminal_alive_delta"),
                    "birth_delta": item.get("birth_delta"),
                    "target_alive_delta": item.get("target_alive_delta"),
                }
                for item in comparisons
            ],
            key=lambda item: (
                _float(item.get("target_local_score_delta")),
                str(item.get("branch_id", "")),
            ),
        )[:16],
        "safe_override_examples": [
            {
                "branch_id": result.get("branch_id"),
                "seed": result.get("seed"),
                "branch_tick": result.get("branch_tick"),
                "agent_id": result.get("agent_id"),
                "logged_action": result.get("logged_action"),
                "oracle_action": result.get("target_local_oracle_action"),
                "categories": result.get("categories"),
                "target_local_score_delta": _mapping(
                    result.get("utility_comparison_vs_logged")
                ).get("target_local_score_delta"),
                "terminal_alive_delta": _mapping(
                    result.get("utility_comparison_vs_logged")
                ).get("terminal_alive_delta"),
                "birth_delta": _mapping(result.get("utility_comparison_vs_logged")).get(
                    "birth_delta"
                ),
            }
            for result in safe_overrides[:24]
        ],
    }


def _acceptance(
    *,
    aggregate: Mapping[str, object],
    floors: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []

    def block(
        reason: str,
        field: str,
        observed: object,
        required: object,
        comparator: str,
    ) -> None:
        blockers.append(
            {
                "reason": reason,
                "field": field,
                "observed": observed,
                "required": required,
                "comparator": comparator,
            }
        )

    if _int(aggregate.get("strict_seed_leak_count")) != 0:
        block(
            "strict_seed_leakage",
            "strict_seed_leak_count",
            aggregate.get("strict_seed_leak_count"),
            0,
            "eq",
        )
    if aggregate.get("replay_verified") is not True:
        block("replay_not_verified", "replay_verified", False, True, "eq")
    if _int(aggregate.get("heuristic_action_source_count")) != 0:
        block(
            "heuristic_action_source_count_nonzero",
            "heuristic_action_source_count",
            aggregate.get("heuristic_action_source_count"),
            0,
            "eq",
        )
    if _int(aggregate.get("unsupported_candidate_action_count")) != 0:
        block(
            "unsupported_candidate_action_count_nonzero",
            "unsupported_candidate_action_count",
            aggregate.get("unsupported_candidate_action_count"),
            0,
            "eq",
        )
    if _int(aggregate.get("branch_point_count")) < _int(
        floors.get("branch_point_count")
    ):
        block(
            "insufficient_branch_points",
            "branch_point_count",
            aggregate.get("branch_point_count"),
            floors.get("branch_point_count"),
            "ge",
        )
    if _int(aggregate.get("source_seed_count")) < _int(
        floors.get("source_seed_count")
    ):
        block(
            "insufficient_source_seeds",
            "source_seed_count",
            aggregate.get("source_seed_count"),
            floors.get("source_seed_count"),
            "ge",
        )
    if _int(aggregate.get("safe_non_logged_override_count")) < _int(
        floors.get("safe_non_logged_override_count")
    ):
        block(
            "insufficient_safe_non_logged_overrides",
            "safe_non_logged_override_count",
            aggregate.get("safe_non_logged_override_count"),
            floors.get("safe_non_logged_override_count"),
            "ge",
        )
    if _float(aggregate.get("safe_non_logged_override_share")) < _float(
        floors.get("safe_non_logged_override_share")
    ):
        block(
            "safe_override_share_below_floor",
            "safe_non_logged_override_share",
            aggregate.get("safe_non_logged_override_share"),
            floors.get("safe_non_logged_override_share"),
            "ge",
        )
    if _float(aggregate.get("dominant_oracle_action_share")) > _float(
        floors.get("dominant_oracle_action_share_max")
    ):
        block(
            "dominant_oracle_action_share_above_cap",
            "dominant_oracle_action_share",
            aggregate.get("dominant_oracle_action_share"),
            floors.get("dominant_oracle_action_share_max"),
            "le",
        )
    if _int(aggregate.get("target_alive_delta_negative_count")) != 0:
        block(
            "target_alive_delta_negative",
            "target_alive_delta_negative_count",
            aggregate.get("target_alive_delta_negative_count"),
            0,
            "eq",
        )
    if _float(_mapping(aggregate.get("target_local_score_delta_summary")).get("mean")) <= 0.0:
        block(
            "mean_target_local_score_delta_not_positive",
            "target_local_score_delta_summary.mean",
            _mapping(aggregate.get("target_local_score_delta_summary")).get("mean"),
            floors.get("mean_target_local_score_delta_gt"),
            "gt",
        )
    if _float(_mapping(aggregate.get("terminal_alive_delta_summary")).get("mean")) < 0.0:
        block(
            "mean_terminal_alive_delta_negative",
            "terminal_alive_delta_summary.mean",
            _mapping(aggregate.get("terminal_alive_delta_summary")).get("mean"),
            floors.get("mean_terminal_alive_delta_min"),
            "ge",
        )
    if _float(_mapping(aggregate.get("birth_delta_summary")).get("mean")) < 0.0:
        block(
            "mean_birth_delta_negative",
            "birth_delta_summary.mean",
            _mapping(aggregate.get("birth_delta_summary")).get("mean"),
            floors.get("mean_birth_delta_min"),
            "ge",
        )
    accepted = not blockers
    return {
        "v99_broad_branch_residual_oracle_accepted": accepted,
        "v100_residual_distillation_allowed": accepted,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "recommendation": (
            "Proceed to v100 diagnostic distillation of support-gated residual "
            "override labels from broad branch continuation outcomes."
            if accepted
            else "Do not build v100 runtime residual from this diagnostic result."
        ),
    }


def _branch_categories(
    *,
    before: Mapping[str, object],
    compact_state: Mapping[str, object],
    legal_actions: Sequence[str],
) -> list[str]:
    self_state = _mapping(compact_state.get("self"))
    center = _mapping(compact_state.get("center"))
    local = _mapping(compact_state.get("local"))
    navigation = _mapping(compact_state.get("navigation"))
    energy = _float(before.get("energy_ratio"))
    hydration = _float(before.get("hydration_ratio"))
    health = _float(before.get("health_ratio"))
    categories: list[str] = []
    if min(energy, hydration, health) < 0.28:
        categories.append("pre_death")
    if 3.0 - energy - hydration - health > 1.0:
        categories.append("recovery")
    if hydration < 0.55 or bool(legal_actions and "drink" in legal_actions):
        water = _target_distance(navigation, "water")
        if water <= 4.0 or _float(center.get("water")) > 0.0:
            categories.append("hydration")
    if energy < 0.62 or "eat" in legal_actions:
        food_signal = (
            _float(center.get("food"))
            + _float(center.get("vegetation"))
            + _float(center.get("fresh_kill_energy"))
            + _float(center.get("carcass_energy"))
            + _float(center.get("carrion_signal"))
            + _float(local.get("radius1_food"))
            + _float(local.get("radius1_carrion"))
        )
        if food_signal > 0.0 or _target_distance(navigation, "plant") <= 4.0:
            categories.append("energy")
    if _legal_move_count(legal_actions) >= 2:
        categories.append("movement")
    if _float(local.get("radius2_carrion")) > 0.0 or _target_distance(
        navigation,
        "carrion",
    ) <= 6.0:
        categories.append("animal_resource")
    if (
        min(energy, hydration, health) >= 0.60
        and "stay" in legal_actions
        and _float(self_state.get("age_norm")) >= 0.0
    ):
        categories.append("reproduction_readiness")
    return list(dict.fromkeys(categories))


def _branch_priority(
    *,
    tick: int,
    ticks: int,
    before: Mapping[str, object],
    compact_state: Mapping[str, object],
    legal_actions: Sequence[str],
    categories: Sequence[str],
) -> float:
    navigation = _mapping(compact_state.get("navigation"))
    local = _mapping(compact_state.get("local"))
    energy = _float(before.get("energy_ratio"))
    hydration = _float(before.get("hydration_ratio"))
    health = _float(before.get("health_ratio"))
    vital_debt = (1.0 - energy) + (1.0 - hydration) + (1.0 - health)
    late = _safe_rate(tick, max(1, ticks - 1))
    recovery_opportunity = max(
        0.0,
        _float(local.get("radius1_food")),
        _float(local.get("radius1_carrion")),
        1.0 - min(_target_distance(navigation, "water") / 8.0, 1.0),
        1.0 - min(_target_distance(navigation, "plant") / 8.0, 1.0),
    )
    return (
        late * 2.0
        + vital_debt * 1.5
        + recovery_opportunity
        + _legal_move_count(legal_actions) * 0.05
        + len(categories) * 0.1
    )


def _branch_point_payload(point: _BroadBranchPoint) -> dict[str, object]:
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture_name,
        "branch_tick": point.branch_tick,
        "branch_index": point.branch_index,
        "record_index": point.record_index,
        "agent_id": point.agent_id,
        "logged_action": point.logged_action,
        "before": dict(point.before),
        "action_mask": dict(point.action_mask),
        "observation_input": dict(point.observation_input),
        "observation_digest": point.observation_digest,
        "observation_schema": point.observation_schema,
        "public_history_trace": [dict(item) for item in point.public_history_trace],
        "compact_state": dict(point.compact_state),
        "categories": list(point.categories),
        "branch_state_digest": point.branch_state_digest,
    }


def _run_excerpt(run: Mapping[str, object] | None) -> dict[str, object] | None:
    if run is None:
        return None
    return {
        "forced_action": run.get("forced_action"),
        "forced_action_used": run.get("forced_action_used"),
        "forced_action_supported": run.get("forced_action_supported"),
        "alive_agents": run.get("alive_agents"),
        "births": run.get("births"),
        "deaths": run.get("deaths"),
        "target_alive_at_end": run.get("target_alive_at_end"),
        "target_energy_ratio_at_end": run.get("target_energy_ratio_at_end"),
        "target_hydration_ratio_at_end": run.get("target_hydration_ratio_at_end"),
        "target_health_ratio_at_end": run.get("target_health_ratio_at_end"),
        "first_action_outcome": run.get("first_action_outcome"),
        "heuristic_action_source_count": run.get("heuristic_action_source_count"),
        "dominant_requested_action": run.get("dominant_requested_action"),
        "dominant_requested_action_share": run.get("dominant_requested_action_share"),
    }


def _digest_payload(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": run.get("branch_id"),
        "forced_action": run.get("forced_action"),
        "forced_action_used": run.get("forced_action_used"),
        "ticks_executed": run.get("ticks_executed"),
        "alive_agents": run.get("alive_agents"),
        "births": run.get("births"),
        "deaths": run.get("deaths"),
        "target_alive_at_end": run.get("target_alive_at_end"),
        "target_energy_ratio_at_end": run.get("target_energy_ratio_at_end"),
        "target_hydration_ratio_at_end": run.get("target_hydration_ratio_at_end"),
        "target_health_ratio_at_end": run.get("target_health_ratio_at_end"),
        "first_action_outcome": run.get("first_action_outcome"),
        "population_horizon_trace": run.get("population_horizon_trace"),
        "target_horizon_trace": run.get("target_horizon_trace"),
        "requested_action_counts": run.get("requested_action_counts"),
    }


def _legal_branch_actions(action_mask: Mapping[str, object]) -> list[str]:
    return [
        action
        for action in _BRANCH_CANDIDATE_ACTIONS
        if bool(action_mask.get(action, False))
    ]


def _action_run(
    runs: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next(
        (run for run in runs if str(run.get("forced_action", "")) == action),
        None,
    )


def _dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = max(sorted(counts.items()), key=lambda item: (item[1], item[0]))
    return {"key": key, "count": int(count), "share": _round(count / float(total))}


def _target_distance(navigation: Mapping[str, object], target: str) -> float:
    state = _mapping(navigation.get(target))
    value = state.get("distance")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("inf")
    parsed = float(value)
    return parsed if math.isfinite(parsed) else float("inf")


def _legal_move_count(actions: Sequence[str]) -> int:
    return sum(1 for action in actions if str(action).startswith("move_"))


def _validated_non_strict_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not values:
        raise BroadBranchResidualOracleAuditError("at least one seed is required")
    leaked = sorted(set(values) & set(V98_STRICT_EXCLUDED_SEEDS))
    if leaked:
        raise BroadBranchResidualOracleAuditError(
            "v99 support seeds must exclude strict seeds: "
            + ", ".join(str(seed) for seed in leaked)
        )
    return values


def _positive_int(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise BroadBranchResidualOracleAuditError(f"{field} must be positive")
    return parsed


def _safe_path_part(value: object) -> str:
    text = str(value).strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "unknown"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _optional_string(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    denom = float(denominator)
    if denom <= 0.0:
        return 0.0
    return _round(float(numerator) / denom)


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_output(path: Path) -> TextIO:
    return path.open("w", encoding="utf-8")
