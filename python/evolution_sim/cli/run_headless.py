from __future__ import annotations

import argparse
from pathlib import Path
import sys

from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.io import write_json_replay
from evolution_sim.mind.learned_policy import (
    MIND_RUNTIME_MODE_GUARDED,
    MIND_RUNTIME_MODES,
    load_learned_policy,
)
from evolution_sim.mind.evolution import load_mind_v3_founder_template
from evolution_sim.mind.v3_neural import load_mind_v3_neural_artifact
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the evolution simulator headlessly.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic RNG seed.")
    parser.add_argument("--ticks", type=int, default=400, help="Maximum ticks to simulate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sim-runs/latest-run.json"),
        help="Replay JSON destination. Existing files are replaced atomically.",
    )
    parser.add_argument(
        "--mind-artifact",
        type=Path,
        help="Optional Mind model artifact to use as the replay policy.",
    )
    parser.add_argument(
        "--enable-mind",
        action="store_true",
        help="Required with --mind-artifact to enable learned-policy inference.",
    )
    parser.add_argument(
        "--mind-runtime-mode",
        choices=sorted(MIND_RUNTIME_MODES),
        default=MIND_RUNTIME_MODE_GUARDED,
        help=(
            "Mind runtime mode. The default guarded mode keeps heuristic safety "
            "fallback active; autonomous modes are legacy learned-artifact "
            "experiment surfaces."
        ),
    )
    parser.add_argument(
        "--mind-v3-autonomous-evolution",
        action="store_true",
        help=(
            "Use the feature-gated Mind v3 inherited autonomous controller "
            "without heuristic fallback."
        ),
    )
    parser.add_argument(
        "--mind-v3-founder-template",
        type=Path,
        help=(
            "Optional raw Mind v3 controller metadata or evolution-search "
            "report used by --mind-v3-autonomous-evolution."
        ),
    )
    parser.add_argument(
        "--mind-v3-neural-artifact",
        type=Path,
        help=(
            "Optional frozen Mind v3 neural policy artifact used by "
            "--mind-v3-autonomous-evolution."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mind_v3_founder_template is not None and not args.mind_v3_autonomous_evolution:
        raise SystemExit(
            "--mind-v3-founder-template requires --mind-v3-autonomous-evolution"
        )
    if args.mind_v3_neural_artifact is not None and not args.mind_v3_autonomous_evolution:
        raise SystemExit(
            "--mind-v3-neural-artifact requires --mind-v3-autonomous-evolution"
        )
    if args.mind_v3_autonomous_evolution and (
        args.mind_artifact is not None or args.enable_mind
    ):
        raise SystemExit(
            "--mind-v3-autonomous-evolution cannot be combined with "
            "--mind-artifact or --enable-mind"
        )
    if args.enable_mind and args.mind_artifact is None:
        raise SystemExit("--enable-mind requires --mind-artifact")
    if args.mind_artifact is not None and not args.enable_mind:
        raise SystemExit("--mind-artifact requires --enable-mind")
    if args.mind_v3_autonomous_evolution:
        founder_template = (
            load_mind_v3_founder_template(args.mind_v3_founder_template)
            if args.mind_v3_founder_template is not None
            else None
        )
        neural_artifact = (
            load_mind_v3_neural_artifact(args.mind_v3_neural_artifact)
            if args.mind_v3_neural_artifact is not None
            else None
        )
        policy = MindV3EvolutionPolicy(
            seed=args.seed,
            founder_template_metadata=founder_template,
            neural_artifact=neural_artifact,
        )
        mind_runtime_mode = "mind-v3-autonomous-evolution"
    else:
        policy = (
            load_learned_policy(
                args.mind_artifact,
                enable_mind=args.enable_mind,
                runtime_mode=args.mind_runtime_mode,
            )
            if args.mind_artifact is not None
            else None
        )
        mind_runtime_mode = args.mind_runtime_mode
    config = WorldConfig(seed=args.seed, max_ticks=args.ticks)
    result = SimulationWorld(config, policy=policy).run()
    if args.output.exists():
        print(f"warning: overwriting existing replay {args.output}", file=sys.stderr)
    try:
        replay_path = write_json_replay(result, args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"failed to write replay {args.output}: {exc}") from exc

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
    if policy is not None:
        print(f"mind_policy={policy.policy_id}")
        print(f"mind_policy_version={policy.policy_version}")
        print(f"mind_runtime_mode={mind_runtime_mode}")
        if args.mind_v3_founder_template is not None:
            print(f"mind_v3_founder_template={args.mind_v3_founder_template}")
        if args.mind_v3_neural_artifact is not None:
            print(f"mind_v3_neural_artifact={args.mind_v3_neural_artifact}")
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
        f"total_carcass_energy={summary['carcass_stats_at_end']['total_carcass_energy']} "
        f"deposition_events={summary['carcass_end']['deposition_events']} "
        f"energy_deposited={summary['carcass_end']['energy_deposited']} "
        f"consumption_events={summary['carcass_end']['consumption_events']} "
        f"energy_consumed={summary['carcass_end']['energy_consumed']} "
        f"energy_decayed={summary['carcass_end']['energy_decayed']} "
        f"conservation_error={summary['carcass_end']['conservation_error']}"
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
