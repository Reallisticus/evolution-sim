from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.io import write_json_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the evolution simulator headlessly.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sim-runs/latest-run.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = WorldConfig(seed=args.seed, max_ticks=args.ticks)
    result = SimulationWorld(config).run()
    replay_path = write_json_replay(result, args.output)

    summary = result.summary
    print(f"replay={replay_path}")
    print(f"run_id={summary['run_id']}")
    print(f"ticks_executed={summary['ticks_executed']}")
    print(f"alive_agents={summary['alive_agents']}")
    print(f"births={summary['births']}")
    print(f"deaths={summary['deaths']}")
    print(f"species_created={summary['species_created']}")
    print(f"alive_species_count={summary['alive_species_count']}")
    print(f"total_agents_seen={summary['total_agents_seen']}")
    print(
        "climate_end="
        f"{summary['disturbance_at_end']}:{summary['disturbance_strength_at_end']}"
    )
    print(
        "land_fields="
        f"fertility_mean={summary['field_stats']['land_fertility']['mean']} "
        f"moisture_mean={summary['field_stats']['land_moisture']['mean']} "
        f"heat_mean={summary['field_stats']['land_heat']['mean']}"
    )
    print(
        "terrain_counts="
        + " ".join(
            f"{terrain}={count}" for terrain, count in summary["terrain_counts"].items()
        )
    )
    print(
        "habitat_end="
        + " ".join(
            f"{state}={count}"
            for state, count in summary["habitat_state_counts_at_end"].items()
        )
    )
    print(
        "hydrology_primary_end="
        + " ".join(
            f"{reason}={count}"
            for reason, count in summary["hydrology_primary_counts_at_end"].items()
        )
    )
    print(
        "hydrology_support_end="
        + " ".join(
            f"{reason}={count}"
            for reason, count in summary["hydrology_support_counts_at_end"].items()
        )
    )
    print(
        "hydrology_primary_stats="
        f"hard_access_tiles={summary['hydrology_primary_stats_at_end']['hard_access_tiles']}"
    )
    print(
        "refuge_end="
        + " ".join(
            f"{reason}={count}" for reason, count in summary["refuge_counts_at_end"].items()
        )
    )
    print(
        "refuge_stats="
        f"avg_refuge_score_forest_tiles={summary['refuge_stats_at_end']['avg_refuge_score_forest_tiles']}"
    )
    print(
        "hazard_end="
        + " ".join(
            f"{hazard_type}={count}"
            for hazard_type, count in summary["hazard_counts_at_end"].items()
        )
    )
    print(
        "hazard_stats="
        f"hazardous_tiles={summary['hazard_stats_at_end']['hazardous_tiles']} "
        f"avg_hazard_level={summary['hazard_stats_at_end']['avg_hazard_level']}"
    )
    print(
        "carcass_end="
        f"carcass_tiles={summary['carcass_stats_at_end']['carcass_tiles']} "
        f"total_carcass_energy={summary['carcass_stats_at_end']['total_carcass_energy']}"
    )
    print(
        "combat_end="
        + " ".join(f"{field}={value}" for field, value in summary["combat_end"].items())
    )
    print(
        "ecology_end="
        + " ".join(
            f"{state}={count}"
            for state, count in summary["ecology_state_counts_at_end"].items()
        )
    )
    print(
        "ecology_stats="
        f"vegetation_mean={summary['ecology_stats_at_end']['avg_vegetation']} "
        f"recovery_mean={summary['ecology_stats_at_end']['avg_recovery_debt']}"
    )


if __name__ == "__main__":
    main()
