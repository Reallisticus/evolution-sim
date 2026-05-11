from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.fixture_labels import (
    FixtureLabelError,
    build_fixture_label_report,
    load_fixture_report,
    write_fixture_label_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract deterministic fixture blocker labels from Mind v3 "
            "evaluation/search reports."
        )
    )
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
        help="Mind v3 evaluation or search JSON report. Repeat to combine reports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-fixture-labels.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        reports = [load_fixture_report(path) for path in args.report]
        label_report = build_fixture_label_report(
            reports,
            report_paths=args.report,
        )
        write_fixture_label_report(label_report, args.output)
    except (OSError, ValueError, FixtureLabelError) as exc:
        raise SystemExit(f"failed to extract fixture labels: {exc}") from exc

    aggregate = label_report["aggregate"]
    print(f"fixture_labels={args.output}")
    print(f"schema_version={label_report['schema_version']}")
    print(f"report_count={label_report['source']['report_count']}")  # type: ignore[index]
    print(f"label_count={aggregate['label_count']}")  # type: ignore[index]
    print(f"failed_label_count={aggregate['failed_label_count']}")  # type: ignore[index]
    print(f"pressure_total={aggregate['pressure_total']}")  # type: ignore[index]


if __name__ == "__main__":
    main()
