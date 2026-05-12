from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolution_sim.mind.temporal_credit_audit import (
    MIND_V3_TEMPORAL_CREDIT_AUDIT_SCHEMA_VERSION,
    TemporalCreditAuditError,
    build_temporal_credit_audit_report,
    load_temporal_credit_json_report,
    write_temporal_credit_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Mind v3 horizon labels for long-horizon temporal-credit "
            "positive support before training a policy artifact."
        )
    )
    parser.add_argument(
        "--horizon-labels",
        type=Path,
        required=True,
        help="mind_horizon_labels_v1 report to audit.",
    )
    parser.add_argument(
        "--autopsy-report",
        type=Path,
        help="Optional carrion autopsy report to attach as failure context.",
    )
    parser.add_argument(
        "--fixture-labels",
        type=Path,
        help="Optional fixture blocker label report to attach pressure context.",
    )
    parser.add_argument("--primary-horizon", type=int, default=120)
    parser.add_argument("--comparison-horizon", type=int, default=80)
    parser.add_argument("--min-primary-survivor-count", type=int, default=1)
    parser.add_argument(
        "--min-primary-post-contact-survivor-count",
        type=int,
        default=1,
    )
    parser.add_argument("--min-primary-survival-rate", type=float, default=0.0)
    parser.add_argument("--max-primary-censored-share", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-temporal-credit-audit.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        horizon_label_report = load_temporal_credit_json_report(args.horizon_labels)
        autopsy_report = (
            load_temporal_credit_json_report(args.autopsy_report)
            if args.autopsy_report is not None
            else None
        )
        fixture_label_report = (
            load_temporal_credit_json_report(args.fixture_labels)
            if args.fixture_labels is not None
            else None
        )
        report = build_temporal_credit_audit_report(
            horizon_label_report,
            primary_horizon=int(args.primary_horizon),
            comparison_horizon=int(args.comparison_horizon),
            min_primary_survivor_count=int(args.min_primary_survivor_count),
            min_primary_post_contact_survivor_count=int(
                args.min_primary_post_contact_survivor_count
            ),
            min_primary_survival_rate=float(args.min_primary_survival_rate),
            max_primary_censored_share=float(args.max_primary_censored_share),
            autopsy_report=autopsy_report,
            fixture_label_report=fixture_label_report,
            horizon_label_report_path=args.horizon_labels,
            autopsy_report_path=args.autopsy_report,
            fixture_label_report_path=args.fixture_labels,
        )
        write_temporal_credit_audit_report(report, args.output)
    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        TemporalCreditAuditError,
    ) as exc:
        raise SystemExit(f"failed to audit temporal-credit labels: {exc}") from exc

    primary = report["horizons"][report["primary_horizon"]]  # type: ignore[index]
    readiness = report["readiness"]  # type: ignore[index]
    print(f"temporal_credit_audit={args.output}")
    print(f"schema_version={MIND_V3_TEMPORAL_CREDIT_AUDIT_SCHEMA_VERSION}")
    print(f"primary_horizon={report['primary_horizon']}")
    print(f"primary_survivor_count={primary['survivor_count']}")  # type: ignore[index]
    print(
        "primary_post_contact_survivor_count="
        f"{primary['post_contact_survivor_count']}"  # type: ignore[index]
    )
    print(f"ready={readiness['ready']}")  # type: ignore[index]
    print(f"blocker_count={len(readiness['blockers'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
