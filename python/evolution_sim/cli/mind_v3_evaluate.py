from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Any

from evolution_sim.config import ClimateConfig, ResourceRegrowthConfig, WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime import reproduction as runtime_reproduction
from evolution_sim.env.runtime.state import Agent
from evolution_sim.genome import Genome
from evolution_sim.genome.species import genome_vector
from evolution_sim.mind.evolution import load_mind_v3_founder_template
from evolution_sim.mind.v3_policy import (
    MIND_V3_REPRODUCTION_READINESS_GOALS,
    MindV3EvolutionPolicy,
)

MIND_V3_EVALUATION_SCHEMA_VERSION = (
    "mind_v3_autonomous_evolution_evaluation_v1"
)
MIND_V3_REPRODUCTION_ATTRIBUTION_POLICY = (
    "terminal_reproduction_failure_attribution_v2"
)
MIND_V3_TEMPORAL_READINESS_ATTRIBUTION_POLICY = (
    "trajectory_temporal_reproduction_blocker_attribution_v1"
)
MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY = (
    "mind_v3_controlled_ecology_fixture_suite_v1"
)
MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY = (
    "mind_v3_controlled_fixture_hard_gate_v1"
)
CONTROLLED_FIXTURE_NAMES = (
    "plant_only",
    "carrion_only",
    "prey_rich",
    "mixed_stable",
)
BIOLOGICAL_BLOCKER_REASONS = (
    "age",
    "cooldown",
    "energy",
    "hydration",
    "health",
    "matched_diet",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Mind v3 autonomous evolution against the heuristic "
            "baseline without heuristic fallback inside the v3 policy."
        )
    )
    parser.add_argument(
        "--seeds",
        default="5,13,19,29",
        help="Comma-separated deterministic seed list.",
    )
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument(
        "--founder-template",
        type=Path,
        help=(
            "Optional raw Mind v3 controller metadata or evolution-search "
            "report whose best candidate initializes founders."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-autonomous-evolution-report.json"),
    )
    parser.add_argument(
        "--fixture-suite",
        choices=["none", "basic"],
        default="none",
        help=(
            "Optionally run controlled ecology fixtures through the same "
            "simulator and policies. Defaults to disabled."
        ),
    )
    parser.add_argument(
        "--fixture-seeds",
        help=(
            "Comma-separated deterministic seed list for --fixture-suite. "
            "Defaults to --seeds."
        ),
    )
    parser.add_argument(
        "--fixture-ticks",
        type=int,
        help="Tick horizon for --fixture-suite. Defaults to --ticks.",
    )
    parser.add_argument(
        "--fixture-min-alive",
        type=float,
        default=1.0,
        help="Minimum v3 alive-agent mean required for every enabled fixture.",
    )
    parser.add_argument(
        "--fixture-min-births",
        type=float,
        default=0.0,
        help="Minimum v3 births mean required for every enabled fixture.",
    )
    parser.add_argument(
        "--fixture-min-mixed-stable-births",
        type=float,
        default=0.0,
        help="Minimum v3 births mean required specifically on mixed_stable.",
    )
    parser.add_argument(
        "--fixture-min-energy-viability",
        type=float,
        default=0.0,
        help="Minimum terminal energy viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-hydration-viability",
        type=float,
        default=0.0,
        help="Minimum terminal hydration viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-health-viability",
        type=float,
        default=0.0,
        help="Minimum terminal health viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-matched-diet-viability",
        type=float,
        default=0.0,
        help="Minimum terminal matched-diet viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-biologically-ready",
        type=float,
        default=0.0,
        help=(
            "Minimum biologically-ready terminal agent mean required for "
            "every fixture."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = _parse_seeds(args.seeds)
    founder_template = (
        load_mind_v3_founder_template(args.founder_template)
        if args.founder_template is not None
        else None
    )
    heuristic_runs = [
        _run_once(seed=seed, ticks=args.ticks, policy=None) for seed in seeds
    ]
    mind_v3_runs = [
        _run_once(
            seed=seed,
            ticks=args.ticks,
            policy=_mind_v3_policy(seed=seed, founder_template=founder_template),
        )
        for seed in seeds
    ]
    report = {
        "schema_version": MIND_V3_EVALUATION_SCHEMA_VERSION,
        "policy": {
            "policy_id": "mind_v3_autonomous_evolution_policy",
            "policy_version": "mind_v3_autonomous_evolution_policy_v1",
            "heuristic_free": True,
            "founder_template_source": (
                str(args.founder_template)
                if args.founder_template is not None
                else None
            ),
            "founder_template_count": _founder_template_count(founder_template),
            "founder_template_specialization_profile_counts": (
                _founder_template_specialization_profile_counts(founder_template)
            ),
        },
        "comparison": {
            "heuristic": {
                "runs": heuristic_runs,
                "aggregate": _aggregate_runs(heuristic_runs),
            },
            "mind_v3": {
                "runs": mind_v3_runs,
                "aggregate": _aggregate_runs(mind_v3_runs),
            },
        },
    }
    report["comparison"]["delta"] = _comparison_delta(
        heuristic=report["comparison"]["heuristic"]["aggregate"],
        mind_v3=report["comparison"]["mind_v3"]["aggregate"],
    )
    if args.fixture_suite != "none":
        fixture_seeds = (
            _parse_seeds(args.fixture_seeds)
            if args.fixture_seeds is not None
            else seeds
        )
        fixture_ticks = (
            int(args.fixture_ticks)
            if args.fixture_ticks is not None
            else int(args.ticks)
        )
        fixture_config = mind_v3_fixture_gate_config(
            suite=args.fixture_suite,
            seeds=fixture_seeds,
            ticks=fixture_ticks,
            min_alive=float(args.fixture_min_alive),
            min_births=float(args.fixture_min_births),
            min_mixed_stable_births=float(args.fixture_min_mixed_stable_births),
            min_energy_viability=float(args.fixture_min_energy_viability),
            min_hydration_viability=float(args.fixture_min_hydration_viability),
            min_health_viability=float(args.fixture_min_health_viability),
            min_matched_diet_viability=float(args.fixture_min_matched_diet_viability),
            min_biologically_ready=float(args.fixture_min_biologically_ready),
        )
        report["fixture_suite"] = run_mind_v3_fixture_suite(
            suite=args.fixture_suite,
            seeds=fixture_seeds,
            ticks=fixture_ticks,
            founder_template=founder_template,
        )
        report["fixture_gate"] = mind_v3_fixture_gate_status(
            fixture_suite=report["fixture_suite"],
            fixture_config=fixture_config,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"mind_v3_report={args.output}")
    print(
        "mind_v3_heuristic_action_source_count="
        f"{report['comparison']['mind_v3']['aggregate']['heuristic_action_source_count']}"
    )
    print(
        "mind_v3_alive_agents_mean="
        f"{report['comparison']['mind_v3']['aggregate']['alive_agents_mean']}"
    )
    print(
        "mind_v3_births_mean="
        f"{report['comparison']['mind_v3']['aggregate']['births_mean']}"
    )
    if "fixture_suite" in report:
        print(
            "mind_v3_fixture_count="
            f"{len(report['fixture_suite']['fixtures'])}"
        )
    if "fixture_gate" in report:
        print(f"mind_v3_fixture_gate_passed={report['fixture_gate']['passed']}")


def _mind_v3_policy(
    *,
    seed: int,
    founder_template: dict[str, object] | list[dict[str, object]] | None,
) -> MindV3EvolutionPolicy:
    return MindV3EvolutionPolicy(
        seed=seed,
        founder_template_metadata=founder_template,
    )


def _run_once(
    *,
    seed: int,
    ticks: int,
    policy: object | None,
) -> dict[str, object]:
    world = SimulationWorld(
        WorldConfig(seed=seed, max_ticks=ticks),
        policy=policy,
    )
    return _run_world(world=world, seed=seed, ticks=ticks)


def _run_world(
    *,
    world: SimulationWorld,
    seed: int,
    ticks: int,
) -> dict[str, object]:
    result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
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
    summary = result.summary
    return {
        "seed": seed,
        "ticks": ticks,
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
        "diet_by_meat_mode_end": _json_ready(
            summary.get("diet_by_meat_mode_end", {})
        ),
        "combat_end": _json_ready(summary.get("combat_end", {})),
        "fresh_kill_end": _json_ready(summary.get("fresh_kill_end", {})),
        "carcass_end": _json_ready(summary.get("carcass_end", {})),
        "reproduction_failure_attribution": (
            _reproduction_failure_attribution(summary)
        ),
        "temporal_readiness_attribution": _temporal_readiness_attribution(
            world.trajectory_records
        ),
        "trajectory_record_count": len(world.trajectory_records),
        "heuristic_action_source_count": _heuristic_action_source_count(
            action_source_counts
        ),
        "unique_requested_actions": len(requested_action_counts),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "dominant_requested_action": dominant_action["action"],
        "dominant_requested_action_count": dominant_action["count"],
        "dominant_requested_action_share": dominant_action["share"],
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
    }


def run_mind_v3_fixture_suite(
    *,
    suite: str,
    seeds: list[int],
    ticks: int,
    founder_template: dict[str, object] | list[dict[str, object]] | None,
) -> dict[str, object]:
    fixtures = []
    for fixture_name in _fixture_names(suite):
        heuristic_runs = [
            _run_fixture_once(
                fixture_name=fixture_name,
                seed=seed,
                ticks=ticks,
                policy=None,
            )
            for seed in seeds
        ]
        mind_v3_runs = [
            _run_fixture_once(
                fixture_name=fixture_name,
                seed=seed,
                ticks=ticks,
                policy=_mind_v3_policy(
                    seed=seed,
                    founder_template=founder_template,
                ),
            )
            for seed in seeds
        ]
        heuristic_aggregate = _aggregate_runs(heuristic_runs)
        mind_v3_aggregate = _aggregate_runs(mind_v3_runs)
        fixtures.append(
            {
                "fixture": fixture_name,
                "scenario_config": _fixture_scenario_config(fixture_name),
                "comparison": {
                    "heuristic": {
                        "runs": heuristic_runs,
                        "aggregate": heuristic_aggregate,
                    },
                    "mind_v3": {
                        "runs": mind_v3_runs,
                        "aggregate": mind_v3_aggregate,
                    },
                    "delta": _comparison_delta(
                        heuristic=heuristic_aggregate,
                        mind_v3=mind_v3_aggregate,
                    ),
                },
            }
        )
    return {
        "policy": MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
        "suite": suite,
        "fixture_names": list(_fixture_names(suite)),
        "seeds": list(seeds),
        "ticks": int(ticks),
        "fixtures": fixtures,
    }


def mind_v3_fixture_gate_config(
    *,
    suite: str,
    seeds: list[int],
    ticks: int | None,
    min_alive: float,
    min_births: float,
    min_mixed_stable_births: float,
    min_energy_viability: float,
    min_hydration_viability: float,
    min_health_viability: float,
    min_matched_diet_viability: float,
    min_biologically_ready: float,
) -> dict[str, object]:
    if suite not in {"basic"}:
        raise SystemExit(f"unsupported fixture suite: {suite}")
    if not seeds:
        raise SystemExit("--fixture-seeds must include at least one integer seed")
    if ticks is not None and int(ticks) < 1:
        raise SystemExit("--fixture-ticks must be >= 1")
    floors = {
        "min_alive": min_alive,
        "min_births": min_births,
        "min_mixed_stable_births": min_mixed_stable_births,
        "min_energy_viability": min_energy_viability,
        "min_hydration_viability": min_hydration_viability,
        "min_health_viability": min_health_viability,
        "min_matched_diet_viability": min_matched_diet_viability,
        "min_biologically_ready": min_biologically_ready,
    }
    for name, value in floors.items():
        if float(value) < 0.0:
            raise SystemExit(f"--fixture-{name.removeprefix('min_')} must be >= 0")
    return {
        "policy": MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY,
        "suite": suite,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks) if ticks is not None else None,
        **{name: float(value) for name, value in floors.items()},
    }


def mind_v3_fixture_gate_status(
    *,
    fixture_suite: Mapping[str, object],
    fixture_config: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    per_fixture: dict[str, object] = {}
    fixtures = fixture_suite.get("fixtures")
    fixture_items = fixtures if isinstance(fixtures, list) else []
    for raw_fixture in fixture_items:
        if not isinstance(raw_fixture, Mapping):
            continue
        fixture_name = str(raw_fixture.get("fixture", "unknown"))
        comparison = raw_fixture.get("comparison")
        mind_v3 = comparison.get("mind_v3") if isinstance(comparison, Mapping) else None
        aggregate = mind_v3.get("aggregate") if isinstance(mind_v3, Mapping) else None
        if not isinstance(aggregate, Mapping):
            blockers.append(
                _fixture_gate_blocker(
                    fixture=fixture_name,
                    reason="fixture_mind_v3_aggregate_missing",
                    metric="aggregate",
                    value=0.0,
                    floor=1.0,
                )
            )
            continue
        metrics = _fixture_gate_metrics(aggregate)
        fixture_blockers = _fixture_gate_blockers(
            fixture_name=fixture_name,
            metrics=metrics,
            fixture_config=fixture_config,
        )
        blockers.extend(fixture_blockers)
        per_fixture[fixture_name] = {
            "metrics": metrics,
            "passed": not fixture_blockers,
            "blockers": fixture_blockers,
        }
    if not fixture_items:
        blockers.append(
            _fixture_gate_blocker(
                fixture=None,
                reason="fixture_suite_empty",
                metric="fixture_count",
                value=0.0,
                floor=1.0,
            )
        )
    return {
        "policy": MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY,
        "fixture_suite_policy": MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
        "suite": str(fixture_config["suite"]),
        "seeds": [int(seed) for seed in list(fixture_config["seeds"])],
        "ticks": (
            fixture_config.get("ticks")
            if fixture_config.get("ticks") is not None
            else fixture_suite.get("ticks")
        ),
        "min_alive": float(fixture_config["min_alive"]),
        "min_births": float(fixture_config["min_births"]),
        "min_mixed_stable_births": float(
            fixture_config["min_mixed_stable_births"]
        ),
        "min_energy_viability": float(fixture_config["min_energy_viability"]),
        "min_hydration_viability": float(
            fixture_config["min_hydration_viability"]
        ),
        "min_health_viability": float(fixture_config["min_health_viability"]),
        "min_matched_diet_viability": float(
            fixture_config["min_matched_diet_viability"]
        ),
        "min_biologically_ready": float(
            fixture_config["min_biologically_ready"]
        ),
        "passed": not blockers,
        "blockers": blockers,
        "per_fixture": dict(sorted(per_fixture.items())),
    }


def _fixture_gate_metrics(aggregate: Mapping[str, object]) -> dict[str, float]:
    attribution = aggregate.get("reproduction_failure_attribution")
    if not isinstance(attribution, Mapping):
        attribution = {}
    viability = attribution.get("terminal_viability_shares_mean")
    if not isinstance(viability, Mapping):
        viability = {}
    return {
        "alive_agents_mean": _float_metric(aggregate.get("alive_agents_mean")),
        "births_mean": _float_metric(aggregate.get("births_mean")),
        "energy_viability_share_mean": _float_metric(viability.get("energy")),
        "energy_requirement_satisfaction_mean": _float_metric(
            aggregate.get("terminal_energy_requirement_satisfaction_mean")
        ),
        "balanced_reproduction_readiness_mean": _float_metric(
            aggregate.get("terminal_balanced_reproduction_readiness_mean")
        ),
        "energy_hydration_balance_mean": _float_metric(
            aggregate.get("terminal_energy_hydration_balance_mean")
        ),
        "energy_hydration_gap_abs_mean": _float_metric(
            aggregate.get("terminal_energy_hydration_gap_abs_mean")
        ),
        "hydration_viability_share_mean": _float_metric(
            viability.get("hydration")
        ),
        "health_viability_share_mean": _float_metric(viability.get("health")),
        "matched_diet_viability_share_mean": _float_metric(
            viability.get("matched_diet")
        ),
        "biologically_ready_agents_mean": _float_metric(
            attribution.get("biologically_ready_agents_mean")
        ),
    }


def _fixture_gate_blockers(
    *,
    fixture_name: str,
    metrics: Mapping[str, float],
    fixture_config: Mapping[str, object],
) -> list[dict[str, object]]:
    checks = [
        ("fixture_alive_floor", "alive_agents_mean", "min_alive"),
        ("fixture_birth_floor", "births_mean", "min_births"),
        (
            "fixture_energy_viability_floor",
            "energy_viability_share_mean",
            "min_energy_viability",
        ),
        (
            "fixture_hydration_viability_floor",
            "hydration_viability_share_mean",
            "min_hydration_viability",
        ),
        (
            "fixture_health_viability_floor",
            "health_viability_share_mean",
            "min_health_viability",
        ),
        (
            "fixture_matched_diet_viability_floor",
            "matched_diet_viability_share_mean",
            "min_matched_diet_viability",
        ),
        (
            "fixture_biological_readiness_floor",
            "biologically_ready_agents_mean",
            "min_biologically_ready",
        ),
    ]
    blockers = []
    for reason, metric_name, floor_name in checks:
        value = float(metrics[metric_name])
        floor = float(fixture_config[floor_name])
        if value < floor:
            blockers.append(
                _fixture_gate_blocker(
                    fixture=fixture_name,
                    reason=reason,
                    metric=metric_name,
                    value=value,
                    floor=floor,
                )
            )
    mixed_stable_floor = float(fixture_config["min_mixed_stable_births"])
    if fixture_name == "mixed_stable" and (
        float(metrics["births_mean"]) < mixed_stable_floor
    ):
        blockers.append(
            _fixture_gate_blocker(
                fixture=fixture_name,
                reason="fixture_mixed_stable_birth_floor",
                metric="births_mean",
                value=float(metrics["births_mean"]),
                floor=mixed_stable_floor,
            )
        )
    return blockers


def _fixture_gate_blocker(
    *,
    fixture: str | None,
    reason: str,
    metric: str,
    value: float,
    floor: float,
) -> dict[str, object]:
    return {
        "fixture": fixture,
        "reason": reason,
        "metric": metric,
        "value": _round(value),
        "floor": _round(floor),
    }


def _float_metric(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _run_fixture_once(
    *,
    fixture_name: str,
    seed: int,
    ticks: int,
    policy: object | None,
) -> dict[str, object]:
    world = _fixture_world(
        fixture_name=fixture_name,
        seed=seed,
        ticks=ticks,
        policy=policy,
    )
    run = _run_world(world=world, seed=seed, ticks=ticks)
    run["fixture"] = fixture_name
    run["evaluation_context"] = "controlled_ecology_fixture"
    return run


def _fixture_names(suite: str) -> tuple[str, ...]:
    if suite == "basic":
        return CONTROLLED_FIXTURE_NAMES
    raise SystemExit(f"unsupported fixture suite: {suite}")


def _fixture_world(
    *,
    fixture_name: str,
    seed: int,
    ticks: int,
    policy: object | None,
) -> SimulationWorld:
    config = WorldConfig(
        seed=seed,
        width=18,
        height=12,
        max_ticks=ticks,
        initial_agents=0,
        max_agents=180,
        resources=_fixture_resources(fixture_name),
        climate=ClimateConfig(season_length=60),
    )
    world = SimulationWorld(config, policy=policy)
    genomes = _archetype_genomes()
    scenario = _fixture_scenario_config(fixture_name)
    _configure_uniform_arena(
        world,
        terrain=str(scenario["terrain"]),
        food=float(scenario["food"]),
        vegetation=float(scenario["vegetation"]),
        shelter=float(scenario["shelter"]),
        recovery_debt=float(scenario["recovery_debt"]),
        fertility=float(scenario["fertility"]),
        moisture=float(scenario["moisture"]),
        heat=float(scenario["heat"]),
    )
    _set_water_border(world)
    _seed_archetypes(
        world,
        counts=dict(scenario["initial_archetype_counts"]),
        genomes=genomes,
    )
    for deposit in scenario["carcass_deposits"]:
        x = int(deposit["x"])
        y = int(deposit["y"])
        world._deposit_carcass(
            world.grid[y][x],
            x=x,
            y=y,
            energy=float(deposit["energy"]),
            source_species=deposit.get("source_species"),
            source_agent_id=None,
            cause="controlled_fixture",
            killer_id=None,
        )
    world.reset_derived_caches()
    return world


def _fixture_resources(fixture_name: str) -> ResourceRegrowthConfig:
    if fixture_name == "plant_only":
        return ResourceRegrowthConfig(
            plain_food_rate=0.038,
            forest_food_rate=0.062,
            wetland_food_rate=0.052,
            rocky_food_rate=0.006,
        )
    if fixture_name == "carrion_only":
        return ResourceRegrowthConfig(
            plain_food_rate=0.003,
            forest_food_rate=0.004,
            wetland_food_rate=0.004,
            rocky_food_rate=0.002,
        )
    if fixture_name == "prey_rich":
        return ResourceRegrowthConfig(
            plain_food_rate=0.044,
            forest_food_rate=0.026,
            wetland_food_rate=0.026,
            rocky_food_rate=0.008,
        )
    if fixture_name == "mixed_stable":
        return ResourceRegrowthConfig(
            plain_food_rate=0.024,
            forest_food_rate=0.038,
            wetland_food_rate=0.034,
            rocky_food_rate=0.01,
        )
    raise SystemExit(f"unsupported fixture: {fixture_name}")


def _fixture_scenario_config(fixture_name: str) -> dict[str, object]:
    scenarios: dict[str, dict[str, object]] = {
        "plant_only": {
            "terrain": "forest",
            "food": 0.92,
            "vegetation": 0.88,
            "shelter": 0.42,
            "recovery_debt": 0.02,
            "fertility": 0.88,
            "moisture": 0.82,
            "heat": 0.28,
            "initial_archetype_counts": {
                "herbivore": 4,
                "hunter": 2,
                "scavenger": 2,
                "omnivore": 4,
            },
            "carcass_deposits": [],
        },
        "carrion_only": {
            "terrain": "rocky",
            "food": 0.02,
            "vegetation": 0.06,
            "shelter": 0.18,
            "recovery_debt": 0.08,
            "fertility": 0.16,
            "moisture": 0.58,
            "heat": 0.42,
            "initial_archetype_counts": {
                "herbivore": 2,
                "hunter": 2,
                "scavenger": 5,
                "omnivore": 3,
            },
            "carcass_deposits": [
                {"x": 3, "y": 7, "energy": 1.1, "source_species": 101},
                {"x": 4, "y": 8, "energy": 1.1, "source_species": 101},
                {"x": 5, "y": 7, "energy": 0.9, "source_species": 101},
                {"x": 14, "y": 8, "energy": 0.7, "source_species": 101},
            ],
        },
        "prey_rich": {
            "terrain": "plain",
            "food": 0.58,
            "vegetation": 0.46,
            "shelter": 0.16,
            "recovery_debt": 0.03,
            "fertility": 0.62,
            "moisture": 0.64,
            "heat": 0.34,
            "initial_archetype_counts": {
                "herbivore": 8,
                "hunter": 4,
                "scavenger": 2,
                "omnivore": 2,
            },
            "carcass_deposits": [],
        },
        "mixed_stable": {
            "terrain": "wetland",
            "food": 0.5,
            "vegetation": 0.58,
            "shelter": 0.3,
            "recovery_debt": 0.02,
            "fertility": 0.68,
            "moisture": 0.86,
            "heat": 0.24,
            "initial_archetype_counts": {
                "herbivore": 3,
                "hunter": 3,
                "scavenger": 3,
                "omnivore": 3,
            },
            "carcass_deposits": [
                {"x": 3, "y": 8, "energy": 0.65, "source_species": 101},
                {"x": 14, "y": 8, "energy": 0.65, "source_species": 101},
            ],
        },
    }
    try:
        return _json_ready(scenarios[fixture_name])
    except KeyError as exc:
        raise SystemExit(f"unsupported fixture: {fixture_name}") from exc


def _configure_uniform_arena(
    world: SimulationWorld,
    *,
    terrain: str,
    food: float,
    vegetation: float,
    shelter: float,
    recovery_debt: float,
    fertility: float,
    moisture: float,
    heat: float,
) -> None:
    for row in world.grid:
        for tile in row:
            tile.terrain = terrain
            tile.food = food
            tile.water = 1.0 if terrain == "water" else 0.0
            tile.fertility = fertility
            tile.moisture = moisture
            tile.heat = heat
            tile.vegetation = vegetation
            tile.shelter = shelter
            tile.recovery_debt = recovery_debt
            tile.fresh_kill_deposits = []
            tile.carcass_deposits = []
            tile.occupant_id = None
    world.reset_derived_caches()


def _set_water_border(world: SimulationWorld) -> None:
    for x in range(world.config.width):
        _set_water_tile(world, x=x, y=0)


def _set_water_tile(world: SimulationWorld, *, x: int, y: int) -> None:
    tile = world.grid[y][x]
    tile.terrain = "water"
    tile.food = 0.0
    tile.water = 1.0
    tile.fertility = 0.0
    tile.moisture = 1.0
    tile.heat = 0.28
    tile.vegetation = 0.0
    tile.shelter = 0.0
    tile.recovery_debt = 0.0
    tile.fresh_kill_deposits = []
    tile.carcass_deposits = []
    tile.occupant_id = None


def _archetype_genomes() -> dict[str, Genome]:
    base = Genome.sample_initial(Random(13))
    return {
        "herbivore": replace(
            base,
            attack_power=0.44,
            attack_cost_multiplier=1.24,
            defense_rating=0.76,
            meat_efficiency=0.24,
            food_efficiency=1.58,
            water_efficiency=1.22,
            plant_bias=1.72,
            carrion_bias=0.22,
            live_prey_bias=0.2,
            reproduction_threshold=0.68,
            mutation_scale=0.03,
        ),
        "hunter": replace(
            base,
            attack_power=1.6,
            attack_cost_multiplier=0.76,
            defense_rating=1.18,
            meat_efficiency=1.56,
            food_efficiency=0.5,
            water_efficiency=0.94,
            plant_bias=0.46,
            carrion_bias=0.74,
            live_prey_bias=1.58,
            reproduction_threshold=0.72,
            mutation_scale=0.03,
        ),
        "scavenger": replace(
            base,
            attack_power=0.82,
            attack_cost_multiplier=0.96,
            defense_rating=0.98,
            meat_efficiency=1.5,
            food_efficiency=0.56,
            water_efficiency=1.0,
            plant_bias=0.48,
            carrion_bias=1.6,
            live_prey_bias=0.38,
            reproduction_threshold=0.72,
            mutation_scale=0.03,
        ),
        "omnivore": replace(
            base,
            attack_power=1.0,
            attack_cost_multiplier=0.92,
            defense_rating=0.94,
            meat_efficiency=1.02,
            food_efficiency=1.12,
            water_efficiency=1.06,
            plant_bias=1.06,
            carrion_bias=0.94,
            live_prey_bias=0.9,
            reproduction_threshold=0.69,
            mutation_scale=0.03,
        ),
    }


def _seed_archetypes(
    world: SimulationWorld,
    *,
    counts: dict[str, int],
    genomes: dict[str, Genome],
) -> None:
    placements = {
        "herbivore": (2, 2),
        "hunter": (world.config.width - 5, 2),
        "scavenger": (2, world.config.height - 5),
        "omnivore": (world.config.width - 5, world.config.height - 5),
    }
    lineage_ids = {
        "herbivore": 101,
        "hunter": 202,
        "scavenger": 303,
        "omnivore": 404,
    }
    for archetype, count in sorted(counts.items()):
        start_x, start_y = placements[archetype]
        for offset in range(int(count)):
            x = start_x + (offset % 3)
            y = start_y + (offset // 3)
            _add_fixture_agent(
                world,
                genome=genomes[archetype],
                x=min(max(x, 0), world.config.width - 1),
                y=min(max(y, 1), world.config.height - 1),
                lineage_id=lineage_ids[archetype],
            )
    world.current_species_map = {
        agent.agent_id: agent.lineage_id for agent in world.alive_agents()
    }
    world.agent_last_species_map = world.current_species_map.copy()
    world.reset_derived_caches()


def _add_fixture_agent(
    world: SimulationWorld,
    *,
    genome: Genome,
    x: int,
    y: int,
    lineage_id: int,
    energy_ratio: float = 0.9,
    hydration_ratio: float = 0.9,
    health_ratio: float = 0.96,
    age: int = 32,
) -> int:
    agent_id = world.next_agent_id
    reproductive_state = runtime_reproduction.founder_reproductive_state(lineage_id)
    agent = Agent(
        agent_id=agent_id,
        parent_id=None,
        lineage_id=lineage_id,
        birth_tick=0,
        death_tick=None,
        x=x,
        y=y,
        energy=genome.max_energy * energy_ratio,
        hydration=genome.max_hydration * hydration_ratio,
        health=genome.max_health * health_ratio,
        max_health=genome.max_health,
        injury_load=0.0,
        age=age,
        alive=True,
        last_reproduction_tick=-10_000,
        last_damage_source="none",
        recent_plant_energy=0.0,
        recent_fresh_kill_energy=0.0,
        recent_carcass_energy=0.0,
        genome_vector=genome_vector(genome),
        genome=genome,
        reproductive_group_id=reproductive_state.group_id,
        reproductive_stage=reproductive_state.stage,
        reproductive_expression=reproductive_state.expression,
        mind_inheritance_metadata=world._founder_mind_metadata(
            agent_id,
            genome=genome,
        ),
    )
    world._place_agent(agent)
    runtime_reproduction.register_founder_group(
        world.reproductive_groups,
        agent,
        tick=0,
    )
    world.next_agent_id += 1
    return agent_id


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    alive_values = [int(run["alive_agents"]) for run in runs]
    birth_values = [int(run["births"]) for run in runs]
    death_values = [int(run["deaths"]) for run in runs]
    action_source_counts: Counter[str] = Counter()
    policy_id_counts: Counter[str] = Counter()
    requested_action_counts: Counter[str] = Counter()
    resolved_action_counts: Counter[str] = Counter()
    for run in runs:
        action_source_counts.update(
            {
                str(source): int(count)
                for source, count in dict(run["action_source_counts"]).items()
            }
        )
        policy_id_counts.update(
            {
                str(policy_id): int(count)
                for policy_id, count in dict(run["policy_id_counts"]).items()
            }
        )
        requested_action_counts.update(
            {
                str(action): int(count)
                for action, count in dict(
                    run.get("requested_action_counts", {})
                ).items()
            }
        )
        resolved_action_counts.update(
            {
                str(action): int(count)
                for action, count in dict(
                    run.get("resolved_action_counts", {})
                ).items()
            }
        )
    dominant_action = _dominant_action_summary(requested_action_counts)
    reproduction_attribution = _aggregate_reproduction_failure_attribution(runs)
    temporal_readiness = _aggregate_temporal_readiness_attribution(runs)
    terminal_viability = reproduction_attribution[
        "terminal_viability_shares_mean"
    ]
    return {
        "run_count": len(runs),
        "alive_agents_mean": _mean(alive_values),
        "births_mean": _mean(birth_values),
        "deaths_mean": _mean(death_values),
        "trajectory_record_count": sum(
            int(run["trajectory_record_count"]) for run in runs
        ),
        "heuristic_action_source_count": _heuristic_action_source_count(
            action_source_counts
        ),
        "unique_requested_actions_mean": _mean(
            [int(run.get("unique_requested_actions", 0)) for run in runs]
        ),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "dominant_requested_action": dominant_action["action"],
        "dominant_requested_action_count": dominant_action["count"],
        "dominant_requested_action_share": dominant_action["share"],
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
        "reproduction_ready_agents_mean": reproduction_attribution[
            "ready_agents_mean"
        ],
        "ready_agents_mean": reproduction_attribution["ready_agents_mean"],
        "biologically_reproduction_ready_agents_mean": (
            reproduction_attribution["biologically_ready_agents_mean"]
        ),
        "biologically_ready_agents_mean": reproduction_attribution[
            "biologically_ready_agents_mean"
        ],
        "terminal_ready_share_mean": reproduction_attribution[
            "terminal_ready_share_mean"
        ],
        "terminal_biologically_ready_share_mean": reproduction_attribution[
            "terminal_biologically_ready_share_mean"
        ],
        "terminal_energy_requirement_satisfaction_mean": (
            reproduction_attribution[
                "terminal_energy_requirement_satisfaction_mean"
            ]
        ),
        "terminal_balanced_reproduction_readiness_mean": (
            reproduction_attribution[
                "terminal_balanced_reproduction_readiness_mean"
            ]
        ),
        "terminal_energy_hydration_balance_mean": (
            reproduction_attribution["terminal_energy_hydration_balance_mean"]
        ),
        "terminal_energy_hydration_gap_abs_mean": (
            reproduction_attribution["terminal_energy_hydration_gap_abs_mean"]
        ),
        "terminal_energy_shortfall_share_mean": reproduction_attribution[
            "terminal_energy_shortfall_share_mean"
        ],
        "terminal_energy_shortfall_agents_mean": reproduction_attribution[
            "terminal_energy_shortfall_agents_mean"
        ],
        "terminal_energy_total_mean": reproduction_attribution[
            "terminal_energy_total_mean"
        ],
        "terminal_energy_required_total_mean": reproduction_attribution[
            "terminal_energy_required_total_mean"
        ],
        "terminal_energy_gap_total_mean": reproduction_attribution[
            "terminal_energy_gap_total_mean"
        ],
        "terminal_energy_viability_share_mean": float(
            dict(terminal_viability).get("energy", 0.0)
        ),
        "terminal_hydration_viability_share_mean": float(
            dict(terminal_viability).get("hydration", 0.0)
        ),
        "terminal_health_viability_share_mean": float(
            dict(terminal_viability).get("health", 0.0)
        ),
        "terminal_matched_diet_viability_share_mean": float(
            dict(terminal_viability).get("matched_diet", 0.0)
        ),
        "reproduction_failure_attribution": reproduction_attribution,
        "temporal_readiness_attribution": temporal_readiness,
        "primary_temporal_readiness_blocker": temporal_readiness[
            "primary_temporal_readiness_blocker"
        ],
    }


def _comparison_delta(
    *,
    heuristic: dict[str, object],
    mind_v3: dict[str, object],
) -> dict[str, object]:
    heuristic_attr = dict(heuristic["reproduction_failure_attribution"])
    mind_v3_attr = dict(mind_v3["reproduction_failure_attribution"])
    return {
        "alive_agents_mean": _round(
            float(mind_v3["alive_agents_mean"])
            - float(heuristic["alive_agents_mean"])
        ),
        "births_mean": _round(
            float(mind_v3["births_mean"]) - float(heuristic["births_mean"])
        ),
        "biologically_ready_agents_mean": _round(
            float(mind_v3_attr["biologically_ready_agents_mean"])
            - float(heuristic_attr["biologically_ready_agents_mean"])
        ),
        "ready_agents_mean": _round(
            float(mind_v3_attr["ready_agents_mean"])
            - float(heuristic_attr["ready_agents_mean"])
        ),
    }


def _reproduction_failure_attribution(
    summary: dict[str, object],
) -> dict[str, object]:
    alive = int(summary.get("alive_agents", 0))
    reproduction_end = summary.get("reproduction_end", {})
    if not isinstance(reproduction_end, dict):
        reproduction_end = {}
    blocker_counts = _int_counter(
        reproduction_end.get("biological_blocker_counts", {})
    )
    blocker_shares = {
        reason: _share(blocker_counts.get(reason, 0), alive)
        for reason in BIOLOGICAL_BLOCKER_REASONS
    }
    terminal_viability_shares = {
        reason: _share(max(0, alive - blocker_counts.get(reason, 0)), alive)
        for reason in BIOLOGICAL_BLOCKER_REASONS
    }
    primary_blocker = _dominant_count_key(blocker_counts)
    ready_agents = int(reproduction_end.get("ready_agents", 0))
    biologically_ready_agents = int(
        reproduction_end.get("biologically_ready_agents", 0)
    )
    energy_readiness = _terminal_energy_readiness(reproduction_end)
    energy_requirement_satisfaction = float(
        energy_readiness["energy_requirement_satisfaction"]
    )
    terminal_hydration_viability = float(terminal_viability_shares["hydration"])
    terminal_balanced_readiness = _balanced_readiness_score(
        [
            energy_requirement_satisfaction,
            terminal_hydration_viability,
            float(terminal_viability_shares["health"]),
            float(terminal_viability_shares["matched_diet"]),
        ]
    )
    return {
        "policy": MIND_V3_REPRODUCTION_ATTRIBUTION_POLICY,
        "alive_agents": alive,
        "ready_agents": ready_agents,
        "biologically_ready_agents": biologically_ready_agents,
        "terminal_ready_share": _share(ready_agents, alive),
        "terminal_biologically_ready_share": _share(
            biologically_ready_agents,
            alive,
        ),
        "blocked_by_max_population_agents": int(
            reproduction_end.get("blocked_by_max_population_agents", 0)
        ),
        "blocked_by_local_crowding_agents": int(
            reproduction_end.get("blocked_by_local_crowding_agents", 0)
        ),
        "biological_blocker_counts": dict(sorted(blocker_counts.items())),
        "biological_blocker_share_by_alive": blocker_shares,
        "terminal_viability_shares": terminal_viability_shares,
        "terminal_energy_requirement_satisfaction": (
            energy_requirement_satisfaction
        ),
        "terminal_balanced_reproduction_readiness": terminal_balanced_readiness,
        "terminal_energy_hydration_balance": _round(
            min(energy_requirement_satisfaction, terminal_hydration_viability)
        ),
        "terminal_energy_hydration_gap_abs": _round(
            abs(energy_requirement_satisfaction - terminal_hydration_viability)
        ),
        "terminal_energy_shortfall_share": energy_readiness[
            "energy_shortfall_share"
        ],
        "terminal_energy_shortfall_agents": energy_readiness[
            "energy_shortfall_agents"
        ],
        "terminal_energy_total": energy_readiness["energy_total"],
        "terminal_energy_required_total": energy_readiness[
            "energy_required_total"
        ],
        "terminal_energy_gap_total": energy_readiness["energy_gap_total"],
        "primary_terminal_blocker": primary_blocker,
        "reproductive_stage_counts": _json_ready(
            reproduction_end.get("reproductive_stage_counts", {})
        ),
        "reproductive_expression_counts": _json_ready(
            reproduction_end.get("reproductive_expression_counts", {})
        ),
        "biological_blocker_counts_by_trophic_role": _json_ready(
            reproduction_end.get("biological_blocker_counts_by_trophic_role", {})
        ),
        "biological_blocker_counts_by_meat_mode": _json_ready(
            reproduction_end.get("biological_blocker_counts_by_meat_mode", {})
        ),
        "energy_readiness_by_trophic_role": _json_ready(
            reproduction_end.get("energy_readiness_by_trophic_role", {})
        ),
        "energy_readiness_by_meat_mode": _json_ready(
            reproduction_end.get("energy_readiness_by_meat_mode", {})
        ),
        "by_trophic_role": _json_ready(reproduction_end.get("by_trophic_role", {})),
        "by_meat_mode": _json_ready(reproduction_end.get("by_meat_mode", {})),
    }


def _aggregate_reproduction_failure_attribution(
    runs: list[dict[str, object]],
) -> dict[str, object]:
    attributions = [
        dict(run["reproduction_failure_attribution"])
        for run in runs
        if isinstance(run.get("reproduction_failure_attribution"), dict)
    ]
    alive_total = sum(int(item["alive_agents"]) for item in attributions)
    blocker_counts: Counter[str] = Counter()
    for item in attributions:
        blocker_counts.update(_int_counter(item["biological_blocker_counts"]))
    blocker_share_by_alive = {
        reason: _share(blocker_counts.get(reason, 0), alive_total)
        for reason in BIOLOGICAL_BLOCKER_REASONS
    }
    blocker_share_by_alive_mean = {
        reason: _mean(
            [
                float(
                    dict(item["biological_blocker_share_by_alive"]).get(
                        reason,
                        0.0,
                    )
                )
                for item in attributions
            ]
        )
        for reason in BIOLOGICAL_BLOCKER_REASONS
    }
    terminal_viability_shares_mean = {
        reason: _mean(
            [
                float(dict(item["terminal_viability_shares"]).get(reason, 0.0))
                for item in attributions
            ]
        )
        for reason in BIOLOGICAL_BLOCKER_REASONS
    }
    return {
        "policy": MIND_V3_REPRODUCTION_ATTRIBUTION_POLICY,
        "run_count": len(attributions),
        "alive_agents_total": alive_total,
        "ready_agents_mean": _mean(
            [int(item["ready_agents"]) for item in attributions]
        ),
        "biologically_ready_agents_mean": _mean(
            [int(item["biologically_ready_agents"]) for item in attributions]
        ),
        "terminal_ready_share_mean": _mean(
            [float(item["terminal_ready_share"]) for item in attributions]
        ),
        "terminal_biologically_ready_share_mean": _mean(
            [
                float(item["terminal_biologically_ready_share"])
                for item in attributions
            ]
        ),
        "blocked_by_max_population_agents_mean": _mean(
            [
                int(item["blocked_by_max_population_agents"])
                for item in attributions
            ]
        ),
        "blocked_by_local_crowding_agents_mean": _mean(
            [
                int(item["blocked_by_local_crowding_agents"])
                for item in attributions
            ]
        ),
        "biological_blocker_counts": dict(sorted(blocker_counts.items())),
        "biological_blocker_share_by_alive": blocker_share_by_alive,
        "biological_blocker_share_by_alive_mean": blocker_share_by_alive_mean,
        "terminal_viability_shares_mean": terminal_viability_shares_mean,
        "terminal_energy_requirement_satisfaction_mean": _mean(
            [
                float(
                    item.get("terminal_energy_requirement_satisfaction", 0.0)
                )
                for item in attributions
            ]
        ),
        "terminal_balanced_reproduction_readiness_mean": _mean(
            [
                float(
                    item.get("terminal_balanced_reproduction_readiness", 0.0)
                )
                for item in attributions
            ]
        ),
        "terminal_energy_hydration_balance_mean": _mean(
            [
                float(item.get("terminal_energy_hydration_balance", 0.0))
                for item in attributions
            ]
        ),
        "terminal_energy_hydration_gap_abs_mean": _mean(
            [
                float(item.get("terminal_energy_hydration_gap_abs", 0.0))
                for item in attributions
            ]
        ),
        "terminal_energy_shortfall_share_mean": _mean(
            [
                float(item.get("terminal_energy_shortfall_share", 0.0))
                for item in attributions
            ]
        ),
        "terminal_energy_shortfall_agents_mean": _mean(
            [
                int(item.get("terminal_energy_shortfall_agents", 0))
                for item in attributions
            ]
        ),
        "terminal_energy_total_mean": _mean(
            [
                float(item.get("terminal_energy_total", 0.0))
                for item in attributions
            ]
        ),
        "terminal_energy_required_total_mean": _mean(
            [
                float(item.get("terminal_energy_required_total", 0.0))
                for item in attributions
            ]
        ),
        "terminal_energy_gap_total_mean": _mean(
            [
                float(item.get("terminal_energy_gap_total", 0.0))
                for item in attributions
            ]
        ),
        "primary_terminal_blocker": _dominant_count_key(blocker_counts),
        "biological_blocker_counts_by_trophic_role": _sum_grouped_int_counts(
            attributions,
            "biological_blocker_counts_by_trophic_role",
        ),
        "biological_blocker_counts_by_meat_mode": _sum_grouped_int_counts(
            attributions,
            "biological_blocker_counts_by_meat_mode",
        ),
        "terminal_readiness_by_trophic_role": _sum_grouped_int_counts(
            attributions,
            "by_trophic_role",
        ),
        "terminal_readiness_by_meat_mode": _sum_grouped_int_counts(
            attributions,
            "by_meat_mode",
        ),
    }


def _temporal_readiness_attribution(
    records: list[dict[str, object]],
) -> dict[str, object]:
    blocker_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    alive_record_count = 0
    for record in records:
        after = record.get("after")
        if not isinstance(after, Mapping) or not bool(after.get("alive", False)):
            continue
        alive_record_count += 1
        blockers = _core_readiness_blockers(after)
        blocker_counts.update(blockers)
        primary = _primary_core_readiness_blocker(after)
        if primary is not None:
            primary_counts.update([primary])
    return {
        "policy": MIND_V3_TEMPORAL_READINESS_ATTRIBUTION_POLICY,
        "alive_agent_tick_count": alive_record_count,
        "core_blocker_agent_tick_counts": {
            field: int(blocker_counts[field]) for field in _core_fields()
        },
        "core_blocker_agent_tick_shares": {
            field: _round(int(blocker_counts[field]) / max(1, alive_record_count))
            for field in _core_fields()
        },
        "primary_core_blocker_agent_tick_counts": {
            field: int(primary_counts[field]) for field in _core_fields()
        },
        "primary_core_blocker_agent_tick_shares": {
            field: _round(int(primary_counts[field]) / max(1, alive_record_count))
            for field in _core_fields()
        },
        "primary_temporal_readiness_blocker": _dominant_count_key(primary_counts),
    }


def _aggregate_temporal_readiness_attribution(
    runs: list[dict[str, object]],
) -> dict[str, object]:
    alive_tick_count = 0
    blocker_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    for run in runs:
        attribution = run.get("temporal_readiness_attribution")
        if not isinstance(attribution, Mapping):
            continue
        alive_tick_count += int(attribution.get("alive_agent_tick_count", 0))
        blocker_counts.update(
            _int_counter(attribution.get("core_blocker_agent_tick_counts", {}))
        )
        primary_counts.update(
            _int_counter(
                attribution.get("primary_core_blocker_agent_tick_counts", {})
            )
        )
    return {
        "policy": MIND_V3_TEMPORAL_READINESS_ATTRIBUTION_POLICY,
        "alive_agent_tick_count": alive_tick_count,
        "core_blocker_agent_tick_counts": {
            field: int(blocker_counts[field]) for field in _core_fields()
        },
        "core_blocker_agent_tick_shares": {
            field: _round(int(blocker_counts[field]) / max(1, alive_tick_count))
            for field in _core_fields()
        },
        "primary_core_blocker_agent_tick_counts": {
            field: int(primary_counts[field]) for field in _core_fields()
        },
        "primary_core_blocker_agent_tick_shares": {
            field: _round(int(primary_counts[field]) / max(1, alive_tick_count))
            for field in _core_fields()
        },
        "primary_temporal_readiness_blocker": _dominant_count_key(primary_counts),
    }


def _core_readiness_blockers(after: Mapping[str, object]) -> list[str]:
    blockers: list[str] = []
    for field, goal in MIND_V3_REPRODUCTION_READINESS_GOALS.items():
        value = _float_value(after.get(field))
        if value is not None and value < float(goal):
            blockers.append(_core_blocker_name(field))
    return blockers


def _primary_core_readiness_blocker(after: Mapping[str, object]) -> str | None:
    gaps: dict[str, float] = {}
    for field, goal in MIND_V3_REPRODUCTION_READINESS_GOALS.items():
        value = _float_value(after.get(field))
        if value is None:
            continue
        gap = max(0.0, 1.0 - value / max(float(goal), 1e-9))
        if gap > 0.0:
            gaps[_core_blocker_name(field)] = gap
    if not gaps:
        return None
    return max(gaps, key=lambda field: (gaps[field], field))


def _core_fields() -> tuple[str, ...]:
    return tuple(
        _core_blocker_name(field)
        for field in MIND_V3_REPRODUCTION_READINESS_GOALS
    )


def _core_blocker_name(field: str) -> str:
    return field.removesuffix("_ratio")


def _heuristic_action_source_count(counts: Counter[str]) -> int:
    return sum(count for source, count in counts.items() if "heuristic" in source)


def _dominant_action_summary(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"action": None, "count": 0, "share": 0.0}
    action, count = max(
        sorted(counts.items()),
        key=lambda item: (int(item[1]), str(item[0])),
    )
    return {
        "action": str(action),
        "count": int(count),
        "share": _round(int(count) / float(total)),
    }


def _dominant_count_key(counts: Counter[str]) -> str | None:
    positive = {key: count for key, count in counts.items() if int(count) > 0}
    if not positive:
        return None
    key, _ = max(
        sorted(positive.items()),
        key=lambda item: (int(item[1]), str(item[0])),
    )
    return str(key)


def _int_counter(value: object) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counts
    for key, count in value.items():
        counts[str(key)] += int(count)
    return counts


def _sum_grouped_int_counts(
    attributions: list[dict[str, object]],
    key: str,
) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = {}
    for attribution in attributions:
        raw_groups = attribution.get(key, {})
        if not isinstance(raw_groups, dict):
            continue
        for raw_group, raw_counts in raw_groups.items():
            if not isinstance(raw_counts, dict):
                continue
            group = str(raw_group)
            if group not in grouped:
                grouped[group] = Counter()
            grouped[group].update(_int_counter(raw_counts))
    return {
        group: dict(sorted(counts.items()))
        for group, counts in sorted(grouped.items())
    }


def _terminal_energy_readiness(
    reproduction_end: Mapping[str, object],
) -> dict[str, float | int]:
    raw_by_mode = reproduction_end.get("energy_readiness_by_meat_mode", {})
    by_mode = raw_by_mode if isinstance(raw_by_mode, Mapping) else {}
    alive_agents = 0
    shortfall_agents = 0
    energy_total = 0.0
    required_total = 0.0
    gap_total = 0.0
    for raw_counts in by_mode.values():
        if not isinstance(raw_counts, Mapping):
            continue
        alive_agents += _nonnegative_int(raw_counts.get("alive_agents"))
        shortfall_agents += _nonnegative_int(
            raw_counts.get("energy_shortfall_agents")
        )
        energy_total += _nonnegative_float(raw_counts.get("energy_total"))
        required_total += _nonnegative_float(
            raw_counts.get("energy_required_total")
        )
        gap_total += _nonnegative_float(raw_counts.get("energy_gap_total"))
    return {
        "alive_agents": alive_agents,
        "energy_shortfall_agents": shortfall_agents,
        "energy_total": _round(energy_total),
        "energy_required_total": _round(required_total),
        "energy_gap_total": _round(gap_total),
        "energy_shortfall_share": _share(shortfall_agents, alive_agents),
        "energy_requirement_satisfaction": _round(
            max(0.0, min(1.0, 1.0 - gap_total / required_total))
            if required_total > 0.0
            else 0.0
        ),
    }


def _balanced_readiness_score(values: list[float]) -> float:
    bounded = [max(0.0, min(1.0, float(value))) for value in values]
    if not bounded or any(value <= 0.0 for value in bounded):
        return 0.0
    return _round(len(bounded) / sum(1.0 / value for value in bounded))


def _float_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _nonnegative_float(value: object) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _share(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _round(numerator / float(denominator))


def _founder_template_count(
    template: dict[str, object] | list[dict[str, object]] | None,
) -> int:
    if template is None:
        return 0
    if isinstance(template, list):
        return len(template)
    return 1


def _founder_template_specialization_profile_counts(
    template: dict[str, object] | list[dict[str, object]] | None,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    templates = template if isinstance(template, list) else [template]
    for metadata in templates:
        if not isinstance(metadata, dict):
            continue
        profile = metadata.get("specialization_profile")
        counts.update([str(profile) if isinstance(profile, str) else "unknown"])
    return dict(sorted(counts.items()))


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise SystemExit("--seeds must include at least one integer seed")
    return seeds


def _mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return _round(sum(values) / float(len(values)))


def _round(value: float) -> float:
    return round(float(value), 4)


if __name__ == "__main__":
    main()
