from __future__ import annotations

from collections import Counter
from typing import Sequence

from evolution_sim.cli.evaluate import _aggregate_report, _dominant_lineage, _round_float
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.policy import ObservationHeuristicPolicy, Policy
from evolution_sim.env.runtime.trajectory import (
    build_trajectory_summary,
)
from evolution_sim.mind.contracts import mind_v1_data_contract
from evolution_sim.mind.gates import build_mind_v1_gate_report


def evaluate_policy(
    *,
    policy_name: str,
    policy: Policy | None,
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    trajectory_counter: Counter[str] = Counter()
    total_records = 0
    invalid_observation = 0
    invalid_resolution = 0
    total_reward = 0.0
    for seed in seeds:
        world = SimulationWorld(
            WorldConfig(seed=seed, max_ticks=ticks),
            policy=policy or ObservationHeuristicPolicy(),
        )
        result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        summary = result.summary
        trajectory_summary = build_trajectory_summary(
            world.trajectory_records,
            signal_config=world.config.signals,
        )
        run = _policy_run_record(
            seed=seed,
            summary=summary,
            trajectory_summary=trajectory_summary,
        )
        total_records += int(trajectory_summary["record_count"])
        invalid_observation += int(
            trajectory_summary["invalid_observation_action_count"]
        )
        invalid_resolution += int(
            trajectory_summary["invalid_resolution_action_count"]
        )
        total_reward += (
            float(trajectory_summary["mean_reward"])
            * int(trajectory_summary["record_count"])
        )
        for action, count in dict(trajectory_summary["action_counts"]).items():
            trajectory_counter[str(action)] += int(count)
        runs.append(run)

    aggregate = _aggregate_report(runs)
    if total_records:
        invalid_observation_rate = _round_float(invalid_observation / total_records)
        invalid_resolution_rate = _round_float(invalid_resolution / total_records)
        aggregate["trajectory"] = {
            "record_count": total_records,
            "invalid_observation_action_count": invalid_observation,
            "invalid_resolution_action_count": invalid_resolution,
            "invalid_observation_action_rate": invalid_observation_rate,
            "invalid_resolution_action_rate": invalid_resolution_rate,
            "invalid_action_rate": invalid_observation_rate,
            "mean_reward": _round_float(total_reward / total_records),
            "action_counts": {
                action: trajectory_counter[action]
                for action in sorted(trajectory_counter)
            },
        }
    return {
        "policy": policy_name,
        "runs": runs,
        "aggregate": aggregate,
    }


def _policy_run_record(
    *,
    seed: int,
    summary: dict[str, object],
    trajectory_summary: dict[str, object],
) -> dict[str, object]:
    return {
        "seed": seed,
        "run_id": summary["run_id"],
        "summary_schema_version": summary["summary_schema_version"],
        "ticks_executed": summary["ticks_executed"],
        "extinct": summary["extinct"],
        "alive_agents": summary["alive_agents"],
        "births": summary["births"],
        "deaths": summary["deaths"],
        "peak_alive_agents": summary["peak_alive_agents"],
        "max_agents": summary["max_agents"],
        "land_tile_count": summary["land_tile_count"],
        "max_agent_saturation_at_end": summary["max_agent_saturation_at_end"],
        "peak_max_agent_saturation": summary["peak_max_agent_saturation"],
        "carrying_capacity": summary["carrying_capacity"],
        "total_agents_seen": summary["total_agents_seen"],
        "last_birth_tick": summary["last_birth_tick"],
        "dominant_lineage": _dominant_lineage(summary),
        "resource_pressure": summary["resource_pressure"],
        "selection_heredity": summary["selection_heredity"],
        "trophic_lifecycle": summary["trophic_lifecycle"],
        "reproduction": summary["reproduction_end"],
        "trophic": {
            "role_counts": summary["trophic_role_counts_at_end"],
            "meat_mode_counts": summary["meat_mode_counts_at_end"],
            "diet": summary["diet_end"],
            "diet_by_trophic_role": summary["diet_by_trophic_role_end"],
            "diet_by_meat_mode": summary["diet_by_meat_mode_end"],
            "animal_resource_opportunity_by_meat_mode": summary[
                "animal_resource_opportunity_by_meat_mode_end"
            ],
        },
        "combat": summary["combat_end"],
        "fresh_kill": summary["fresh_kill_end"],
        "carrion": summary["carcass_end"],
        "hazard": {
            "counts": summary["hazard_counts_at_end"],
            "hazardous_tiles": summary["hazard_stats_at_end"]["hazardous_tiles"],
            "avg_hazard_level": summary["hazard_stats_at_end"]["avg_hazard_level"],
        },
        "hydrology": {
            "primary_counts": summary["hydrology_primary_counts_at_end"],
            "hard_access_tiles": summary["hydrology_primary_stats_at_end"][
                "hard_access_tiles"
            ],
        },
        "ecology": {
            "counts": summary["ecology_state_counts_at_end"],
            "avg_vegetation": summary["ecology_stats_at_end"]["avg_vegetation"],
            "avg_recovery_debt": summary["ecology_stats_at_end"]["avg_recovery_debt"],
        },
        "trajectory": trajectory_summary,
    }


def compare_heuristic_and_learned(
    *,
    learned_policy: Policy,
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    heuristic = evaluate_policy(
        policy_name=ObservationHeuristicPolicy().policy_id,
        policy=ObservationHeuristicPolicy(),
        seeds=seeds,
        ticks=ticks,
    )
    learned = evaluate_policy(
        policy_name=learned_policy.policy_id,
        policy=learned_policy,
        seeds=seeds,
        ticks=ticks,
    )
    return {
        "protocol": {
            "mind_v1_data_contract": mind_v1_data_contract(),
            "seeds": list(seeds),
            "ticks": ticks,
            "mode": RunMode.SUMMARY_ONLY.value,
        },
        "heuristic": heuristic,
        "learned": learned,
        "comparison": {
            "alive_agents_mean_delta": _metric_mean_delta(
                learned["aggregate"],
                heuristic["aggregate"],
                "alive_agents",
            ),
            "births_mean_delta": _metric_mean_delta(
                learned["aggregate"],
                heuristic["aggregate"],
                "births",
            ),
        },
        "mind_v1_gates": build_mind_v1_gate_report(
            learned,
            baseline_report=heuristic,
        ),
    }


def _metric_mean_delta(
    left: dict[str, object],
    right: dict[str, object],
    field: str,
) -> float | None:
    left_stats = left.get(field)
    right_stats = right.get(field)
    if not isinstance(left_stats, dict) or not isinstance(right_stats, dict):
        return None
    left_mean = left_stats.get("mean")
    right_mean = right_stats.get("mean")
    if not isinstance(left_mean, (int, float)) or not isinstance(right_mean, (int, float)):
        return None
    return _round_float(float(left_mean) - float(right_mean))
