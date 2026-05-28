from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_sequence_continuation_scorer import (
    MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION,
    BranchSequenceContinuationScorerError,
    build_branch_sequence_continuation_scorer_report,
    load_json_report,
    write_branch_sequence_continuation_scorer_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Mind v3 v94 replay-backed sequence continuation "
            "scorer diagnostic."
        )
    )
    parser.add_argument(
        "--support-branch-action-oracle-labels",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v93-depleted-resource-trap-support-labels.json"
        ),
    )
    parser.add_argument(
        "--strict-branch-action-oracle-labels",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v91-failure-frontier-branch-action-oracle-labels.json"
        ),
    )
    parser.add_argument(
        "--support-branch-action-oracle-audit",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--v93-depleted-resource-trap-audit",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v94-branch-sequence-continuation-scorer.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        support_labels = load_json_report(args.support_branch_action_oracle_labels)
        strict_labels = load_json_report(args.strict_branch_action_oracle_labels)
        support_audit = (
            load_json_report(args.support_branch_action_oracle_audit)
            if args.support_branch_action_oracle_audit is not None
            else None
        )
        v93_audit = (
            load_json_report(args.v93_depleted_resource_trap_audit)
            if args.v93_depleted_resource_trap_audit is not None
            else None
        )
        report = build_branch_sequence_continuation_scorer_report(
            support_branch_action_oracle_labels=support_labels,
            strict_branch_action_oracle_labels=strict_labels,
            support_branch_action_oracle_audit=support_audit,
            v93_depleted_resource_trap_audit=v93_audit,
        )
        write_branch_sequence_continuation_scorer_report(report, args.output)
    except (OSError, ValueError, BranchSequenceContinuationScorerError) as exc:
        raise SystemExit(
            f"failed to build branch sequence continuation scorer: {exc}"
        ) from exc

    coverage = report["coverage"]  # type: ignore[index]
    acceptance = report["acceptance"]  # type: ignore[index]
    best = acceptance["best_rule_for_diagnostics"]  # type: ignore[index]
    print(f"branch_sequence_continuation_scorer={args.output}")
    print(
        "schema_version="
        f"{MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION}"
    )
    print(
        "support_sequence_example_count="
        f"{report['sequence_continuation_dataset']['support_sequence_example_count']}"  # type: ignore[index]
    )
    print(f"strict_eval_label_count={coverage['strict_eval_label_count']}")  # type: ignore[index]
    print(f"best_rule={best.get('rule')}")  # type: ignore[union-attr]
    print(
        "best_rule_mean_target_local_score_delta="
        f"{best.get('mean_target_local_score_delta')}"  # type: ignore[union-attr]
    )
    print(
        "seed_41_tick_113_avoided="
        f"{best.get('seed_41_tick_113_avoided')}"  # type: ignore[union-attr]
    )
    print(
        "seed_41_tick_114_avoided="
        f"{best.get('seed_41_tick_114_avoided')}"  # type: ignore[union-attr]
    )
    print(
        "v94_sequence_continuation_scorer_accepted="
        f"{acceptance['v94_sequence_continuation_scorer_accepted']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(acceptance['strict_blockers'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
