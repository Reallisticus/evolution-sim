from __future__ import annotations

import argparse
from pathlib import Path
import sys

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind.learned_policy import load_learned_policy


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
    parser.add_argument(
        "--mind-artifact",
        type=Path,
        help="Optional Mind model artifact to use as the trajectory policy.",
    )
    parser.add_argument(
        "--enable-mind",
        action="store_true",
        help="Required with --mind-artifact to enable learned-policy inference.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.enable_mind and args.mind_artifact is None:
        raise SystemExit("--enable-mind requires --mind-artifact")
    if args.mind_artifact is not None and not args.enable_mind:
        raise SystemExit("--mind-artifact requires --enable-mind")
    policy = (
        load_learned_policy(args.mind_artifact, enable_mind=args.enable_mind)
        if args.mind_artifact is not None
        else None
    )
    config = WorldConfig(seed=args.seed, max_ticks=args.ticks)
    writer = JsonlTrajectoryWriter(
        args.output,
        source_seeds=[args.seed],
        split_id=args.split_id,
    )
    if args.output.exists():
        print(f"warning: overwriting existing trajectory {args.output}", file=sys.stderr)
    try:
        result = SimulationWorld(config, policy=policy).run(
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
    if policy is not None:
        print(f"mind_policy={policy.policy_id}")
        print(f"mind_policy_version={policy.policy_version}")
    print(f"trajectory_records={writer.record_count}")
    print(f"mean_reward={writer.trajectory_summary['mean_reward']}")


if __name__ == "__main__":
    main()
