from __future__ import annotations

import argparse
import json

from evolution_sim.cli.evaluate import parse_seed_selection
from evolution_sim.mind.splits import deterministic_seed_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation seed splits for Mind experiments."
    )
    parser.add_argument("--seeds", help="Comma-separated seed list.")
    parser.add_argument("--seed", action="append", type=int, help="Add one seed.")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_seed_selection(args.seed, args.seeds)
    print(
        json.dumps(
            deterministic_seed_split(
                seeds,
                validation_fraction=args.validation_fraction,
                split_seed=args.split_seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
