from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Sequence

from evolution_sim.evaluation.reporting import (
    aggregate_report,
    dominant_lineage,
    round_float,
)
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.policy import ObservationHeuristicPolicy, Policy
from evolution_sim.env.runtime.trajectory import (
    build_trajectory_summary,
)
from evolution_sim.mind.contracts import mind_v1_data_contract
from evolution_sim.mind.diagnostics import build_policy_diagnostics
from evolution_sim.mind.gates import (
    build_mind_v1_gate_report,
    normalize_mind_v1_gate_criteria,
)


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
    all_records: list[dict[str, object]] = []
    all_decision_diagnostics: list[dict[str, object] | None] = []
    for seed in seeds:
        world = SimulationWorld(
            WorldConfig(seed=seed, max_ticks=ticks),
            policy=policy or ObservationHeuristicPolicy(),
        )
        result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        summary = result.summary
        all_records.extend(world.trajectory_records)
        all_decision_diagnostics.extend(world.policy_decision_diagnostics_records)
        trajectory_summary = build_trajectory_summary(
            world.trajectory_records,
            signal_config=world.config.signals,
        )
        run = _policy_run_record(
            seed=seed,
            summary=summary,
            trajectory_summary=trajectory_summary,
            policy_diagnostics=build_policy_diagnostics(
                world.trajectory_records,
                decision_diagnostics=world.policy_decision_diagnostics_records,
            ),
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

    aggregate = aggregate_report(runs)
    if total_records:
        invalid_observation_rate = round_float(invalid_observation / total_records)
        invalid_resolution_rate = round_float(invalid_resolution / total_records)
        aggregate["trajectory"] = {
            "record_count": total_records,
            "invalid_observation_action_count": invalid_observation,
            "invalid_resolution_action_count": invalid_resolution,
            "invalid_observation_action_rate": invalid_observation_rate,
            "invalid_resolution_action_rate": invalid_resolution_rate,
            "invalid_action_rate": invalid_observation_rate,
            "mean_reward": round_float(total_reward / total_records),
            "action_counts": {
                action: trajectory_counter[action]
                for action in sorted(trajectory_counter)
            },
        }
    aggregate["policy_diagnostics"] = build_policy_diagnostics(
        all_records,
        decision_diagnostics=all_decision_diagnostics,
    )
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
    policy_diagnostics: dict[str, object],
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
        "dominant_lineage": dominant_lineage(summary),
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
        "policy_diagnostics": policy_diagnostics,
    }


def compare_heuristic_and_learned(
    *,
    learned_policy: Policy,
    seeds: Sequence[int],
    ticks: int,
    gate_criteria: Mapping[str, object] | None = None,
    reference_guard_intervention_rate: float | None = None,
) -> dict[str, object]:
    resolved_gate_criteria = normalize_mind_v1_gate_criteria(gate_criteria)
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
            "gate_criteria": resolved_gate_criteria,
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
            "by_trophic_role": _paired_count_totals_comparison(
                heuristic["aggregate"],
                learned["aggregate"],
                "trophic_role_counts_at_end",
            ),
            "by_meat_mode": _paired_count_totals_comparison(
                heuristic["aggregate"],
                learned["aggregate"],
                "meat_mode_counts_at_end",
            ),
            "policy_diagnostics_by_trophic_role": (
                _paired_policy_diagnostic_group_comparison(
                    heuristic["aggregate"],
                    learned["aggregate"],
                    "by_trophic_role",
                )
            ),
            "policy_diagnostics_by_meat_mode": (
                _paired_policy_diagnostic_group_comparison(
                    heuristic["aggregate"],
                    learned["aggregate"],
                    "by_meat_mode",
                )
            ),
            "per_seed": _paired_seed_comparison(
                heuristic["runs"],
                learned["runs"],
            ),
        },
        "mind_v1_gates": build_mind_v1_gate_report(
            learned,
            baseline_report=heuristic,
            reference_guard_intervention_rate=reference_guard_intervention_rate,
            **resolved_gate_criteria,
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
    return round_float(float(left_mean) - float(right_mean))


def _paired_count_totals_comparison(
    heuristic_aggregate: dict[str, object],
    learned_aggregate: dict[str, object],
    field: str,
) -> dict[str, dict[str, object]]:
    heuristic_total = _numeric_bucket(heuristic_aggregate.get(field), "total")
    learned_total = _numeric_bucket(learned_aggregate.get(field), "total")
    heuristic_mean = _numeric_bucket(heuristic_aggregate.get(field), "per_run_mean")
    learned_mean = _numeric_bucket(learned_aggregate.get(field), "per_run_mean")
    labels = sorted(
        set(heuristic_total)
        | set(learned_total)
        | set(heuristic_mean)
        | set(learned_mean)
    )
    comparison: dict[str, dict[str, object]] = {}
    for label in labels:
        heuristic_total_value = heuristic_total.get(label, 0.0)
        learned_total_value = learned_total.get(label, 0.0)
        heuristic_mean_value = heuristic_mean.get(label, 0.0)
        learned_mean_value = learned_mean.get(label, 0.0)
        comparison[label] = {
            "heuristic_total": _whole_number_if_integral(heuristic_total_value),
            "learned_total": _whole_number_if_integral(learned_total_value),
            "total_delta": _whole_number_if_integral(
                learned_total_value - heuristic_total_value
            ),
            "heuristic_per_run_mean": round_float(heuristic_mean_value),
            "learned_per_run_mean": round_float(learned_mean_value),
            "per_run_mean_delta": round_float(
                learned_mean_value - heuristic_mean_value
            ),
        }
    return comparison


def _paired_policy_diagnostic_group_comparison(
    heuristic_aggregate: dict[str, object],
    learned_aggregate: dict[str, object],
    group_field: str,
) -> dict[str, dict[str, object]]:
    heuristic_groups = _policy_diagnostic_groups(heuristic_aggregate, group_field)
    learned_groups = _policy_diagnostic_groups(learned_aggregate, group_field)
    comparison: dict[str, dict[str, object]] = {}
    for label in sorted(set(heuristic_groups) | set(learned_groups)):
        heuristic_group = heuristic_groups.get(label, {})
        learned_group = learned_groups.get(label, {})
        heuristic_record_count = _group_number(heuristic_group, "record_count", 0.0)
        learned_record_count = _group_number(learned_group, "record_count", 0.0)
        heuristic_guard_count = _group_number(
            heuristic_group,
            "guard_intervention_count",
            0.0,
        )
        learned_guard_count = _group_number(
            learned_group,
            "guard_intervention_count",
            0.0,
        )
        comparison[label] = {
            "heuristic_record_count": _whole_number_if_integral(
                heuristic_record_count
            ),
            "learned_record_count": _whole_number_if_integral(learned_record_count),
            "record_count_delta": _whole_number_if_integral(
                learned_record_count - heuristic_record_count
            ),
            "heuristic_mean_reward": _optional_group_number(
                heuristic_group,
                "mean_reward",
            ),
            "learned_mean_reward": _optional_group_number(
                learned_group,
                "mean_reward",
            ),
            "mean_reward_delta": _optional_group_delta(
                learned_group,
                heuristic_group,
                "mean_reward",
            ),
            "heuristic_guard_intervention_count": _whole_number_if_integral(
                heuristic_guard_count
            ),
            "learned_guard_intervention_count": _whole_number_if_integral(
                learned_guard_count
            ),
            "guard_intervention_count_delta": _whole_number_if_integral(
                learned_guard_count - heuristic_guard_count
            ),
            "heuristic_guard_intervention_rate": _optional_group_number(
                heuristic_group,
                "guard_intervention_rate",
            ),
            "learned_guard_intervention_rate": _optional_group_number(
                learned_group,
                "guard_intervention_rate",
            ),
            "guard_intervention_rate_delta": _optional_group_delta(
                learned_group,
                heuristic_group,
                "guard_intervention_rate",
            ),
        }
    return comparison


def _numeric_bucket(payload: object, bucket: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    values = payload.get(bucket)
    if not isinstance(values, Mapping):
        return {}
    parsed: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        parsed[str(key)] = float(value)
    return parsed


def _policy_diagnostic_groups(
    aggregate: dict[str, object],
    group_field: str,
) -> dict[str, Mapping[str, object]]:
    diagnostics = aggregate.get("policy_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return {}
    groups = diagnostics.get(group_field)
    if not isinstance(groups, Mapping):
        return {}
    return {
        str(label): group
        for label, group in groups.items()
        if isinstance(group, Mapping)
    }


def _group_number(
    group: Mapping[str, object],
    field: str,
    default: float,
) -> float:
    value = group.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _optional_group_number(
    group: Mapping[str, object],
    field: str,
) -> float | None:
    value = group.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round_float(float(value))


def _optional_group_delta(
    left: Mapping[str, object],
    right: Mapping[str, object],
    field: str,
) -> float | None:
    left_value = _optional_group_number(left, field)
    right_value = _optional_group_number(right, field)
    if left_value is None or right_value is None:
        return None
    return round_float(left_value - right_value)


def _whole_number_if_integral(value: float) -> int | float:
    if value.is_integer():
        return int(value)
    return round_float(value)


def _paired_seed_comparison(
    heuristic_runs: Sequence[dict[str, object]],
    learned_runs: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    heuristic_by_seed = _runs_by_seed(heuristic_runs)
    learned_by_seed = _runs_by_seed(learned_runs)
    paired: list[dict[str, object]] = []
    for seed in sorted(set(heuristic_by_seed) | set(learned_by_seed)):
        heuristic = heuristic_by_seed.get(seed, {})
        learned = learned_by_seed.get(seed, {})
        record: dict[str, object] = {"seed": seed}
        for field in ("alive_agents", "births", "deaths"):
            heuristic_value = _run_numeric_value(heuristic, field)
            learned_value = _run_numeric_value(learned, field)
            record[f"heuristic_{field}"] = heuristic_value
            record[f"learned_{field}"] = learned_value
            record[f"{field}_delta"] = (
                None
                if heuristic_value is None or learned_value is None
                else round_float(learned_value - heuristic_value)
            )
        paired.append(record)
    return paired


def _runs_by_seed(
    runs: Sequence[dict[str, object]],
) -> dict[int, dict[str, object]]:
    by_seed: dict[int, dict[str, object]] = {}
    for run in runs:
        seed = run.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        by_seed[seed] = run
    return by_seed


def _run_numeric_value(
    run: dict[str, object],
    field: str,
) -> float | None:
    value = run.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
