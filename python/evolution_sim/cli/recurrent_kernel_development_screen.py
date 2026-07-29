"""Run the bounded, non-authoritative recurrent-kernel development screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evolution_sim.mind.recurrent_kernel_development_screen import (
    RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES,
    RecurrentKernelDevelopmentScreenError,
    run_candidate_screen,
    run_development_screen,
    validate_development_screen_report_path,
    write_development_screen_report,
)


DEFAULT_REPORT = Path(
    "output/open-ecology/development-screens/recurrent-kernel-development-screen.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Screen a fixed set of recurrent numerical kernels without "
            "producing launch authority or a scientific result."
        )
    )
    parser.add_argument("--development-run", action="store_true")
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--child-candidate",
        choices=RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--child-nonce", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.development_run:
        raise SystemExit("--development-run is required")
    if args.report.suffix != ".json":
        raise SystemExit("--report must use a canonical .json path")
    try:
        if args.child_candidate is not None:
            if args.child_nonce is None:
                raise SystemExit("--child-nonce is required for a child candidate")
            result = run_candidate_screen(
                candidate=args.child_candidate,
                expected_source_sha=args.expected_source_sha,
                child_nonce=args.child_nonce,
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0
        validate_development_screen_report_path(args.report)
        report = run_development_screen(
            expected_source_sha=args.expected_source_sha,
            progress=lambda message: print(message, flush=True),
        )
        write_development_screen_report(args.report, report)
    except RecurrentKernelDevelopmentScreenError as exc:
        raise SystemExit(str(exc)) from exc
    selection = report["selection"]
    assert isinstance(selection, dict)
    print(
        "recurrent_kernel_development_screen_complete "
        f"selected={selection['selected_candidate']} report={args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
