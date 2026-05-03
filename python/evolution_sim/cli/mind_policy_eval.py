from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolution_sim.cli.evaluate import parse_seed_selection
from evolution_sim.mind.evaluation import compare_heuristic_and_learned
from evolution_sim.mind.learned_policy import load_learned_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare heuristic and learned policies on summary-only seeds."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--enable-mind",
        action="store_true",
        help="Required to run learned-policy inference.",
    )
    parser.add_argument("--seeds", help="Comma-separated seed list.")
    parser.add_argument("--seed", action="append", type=int, help="Add one seed.")
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = load_learned_policy(args.artifact, enable_mind=args.enable_mind)
    seeds = parse_seed_selection(args.seed, args.seeds)
    report = compare_heuristic_and_learned(
        learned_policy=policy,
        seeds=seeds,
        ticks=args.ticks,
    )
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
