from __future__ import annotations

import argparse
from pathlib import Path
import sys

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import JsonlTrajectoryWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect trajectory records without building full replay surfaces."
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic RNG seed.")
    parser.add_argument("--ticks", type=int, default=400, help="Maximum ticks to simulate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/trajectories/latest-trajectory.jsonl.gz"),
        help="Trajectory JSONL destination. Use a .gz suffix for gzip compression.",
    )
    parser.add_argument(
        "--split-id",
        default="unsplit",
        help="Dataset split identifier to record in trajectory provenance.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = WorldConfig(seed=args.seed, max_ticks=args.ticks)
    writer = JsonlTrajectoryWriter(
        args.output,
        source_seeds=[args.seed],
        split_id=args.split_id,
    )
    if args.output.exists():
        print(f"warning: overwriting existing trajectory {args.output}", file=sys.stderr)
    try:
        result = SimulationWorld(config).run(
            mode=RunMode.SUMMARY_ONLY,
            trajectory_sink=writer,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"failed to write trajectory {args.output}: {exc}") from exc

    summary = result.summary
    print(f"trajectory={args.output}")
    print(f"run_id={summary['run_id']}")
    print(f"ticks_executed={summary['ticks_executed']}")
    print(f"alive_agents={summary['alive_agents']}")
    print(f"births={summary['births']}")
    print(f"deaths={summary['deaths']}")
    print(f"trajectory_records={writer.record_count}")
    print(f"mean_reward={writer.trajectory_summary['mean_reward']}")


if __name__ == "__main__":
    main()
