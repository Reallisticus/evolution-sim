from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a saved simulation replay.")
    parser.add_argument("replay", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(args.replay.read_text(encoding="utf-8"))
    summary = payload["summary"]

    print(f"run_id={summary['run_id']}")
    print(f"seed={summary['seed']}")
    print(f"ticks_executed={summary['ticks_executed']}")
    print(f"alive_agents={summary['alive_agents']}")
    print(f"peak_alive_agents={summary['peak_alive_agents']}")
    print(f"births={summary['births']}")
    print(f"deaths={summary['deaths']}")
    print(f"extinct={summary['extinct']}")
    print(f"season_at_end={summary['season_at_end']}")
    print(
        "climate_end="
        f"{summary['disturbance_at_end']}:{summary['disturbance_strength_at_end']}"
    )
    print(f"alive_lineages={len(summary['alive_lineages'])}")
    print(f"species_created={summary['species_created']}")
    print(f"alive_species_count={summary['alive_species_count']}")
    print(f"last_birth_tick={summary['last_birth_tick']}")
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

    for entry in summary["top_lineages"][:5]:
        print(
            "top_lineage="
            f"{entry['lineage_id']} total={entry['total_agents']} alive={entry['alive_agents']}"
        )

    for entry in summary["top_species"][:5]:
        print(
            "top_species="
            f"{entry['species_id']} label={entry['label']} peak={entry['peak_members']} "
            f"alive={entry['alive_members']} lineages={','.join(map(str, entry['lineages']))}"
        )


if __name__ == "__main__":
    main()
