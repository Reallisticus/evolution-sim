from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_branch_constrained_planning_audit
from evolution_sim.mind.branch_constrained_planning_audit import (
    MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION,
    _greedy_constrained_assignment,
    build_branch_constrained_planning_audit_report,
)
from evolution_sim.mind.branch_sequence_continuation_scorer import (
    build_branch_sequence_continuation_scorer_report,
)
from python.tests.test_mind_v3_branch_sequence_continuation_scorer import (
    _labels_with_trace_targets,
)


class MindV3BranchConstrainedPlanningAuditTests(unittest.TestCase):
    def test_branch_constrained_planning_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:branch-constrained-planning-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_branch_constrained_planning_audit"
            ),
        )

    def test_constrained_planning_audit_reports_schema_and_contract(self) -> None:
        strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
        sequence = _sequence_report(strict)

        report = build_branch_constrained_planning_audit_report(
            strict_branch_action_oracle_labels=strict,
            branch_sequence_continuation_scorer_report=sequence,
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION,
        )
        self.assertFalse(report["contract"]["runtime_policy_trained"])
        self.assertFalse(report["contract"]["runtime_ready"])
        self.assertTrue(report["contract"]["diagnostic_only"])
        self.assertTrue(report["contract"]["uses_replay_backed_candidate_outcomes"])
        self.assertFalse(report["contract"]["uses_exact_reexecution"])
        self.assertIn("planner_reports", report)
        self.assertIn("constrained_planning_support_probe", report)
        self.assertFalse(report["acceptance"]["runtime_ready"])

    def test_greedy_constrained_assignment_reduces_dominant_action(self) -> None:
        baseline = [
            _comparison("b1", "eat", 1.0),
            _comparison("b2", "eat", 2.0),
            _comparison("b3", "eat", 3.0),
            _comparison("b4", "drink", 0.0),
        ]
        outcomes = {
            "b1": [_comparison("b1", "eat", 1.0), _comparison("b1", "stay", 0.9)],
            "b2": [_comparison("b2", "eat", 2.0), _comparison("b2", "stay", 2.1)],
            "b3": [_comparison("b3", "eat", 3.0), _comparison("b3", "stay", 2.8)],
            "b4": [_comparison("b4", "drink", 0.0)],
        }

        planned = _greedy_constrained_assignment(
            baseline,
            candidate_outcomes=outcomes,
            max_action_count=2,
        )

        counts = {
            action: sum(1 for item in planned if item["predicted_action"] == action)
            for action in ("eat", "stay", "drink")
        }
        self.assertEqual(counts["eat"], 2)
        self.assertEqual(counts["stay"], 1)
        self.assertTrue(
            all(item["target_alive_delta"] >= 0.0 for item in planned)
        )

    def test_constrained_planning_audit_json_is_deterministic(self) -> None:
        strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
        sequence = _sequence_report(strict)

        first = build_branch_constrained_planning_audit_report(
            strict_branch_action_oracle_labels=strict,
            branch_sequence_continuation_scorer_report=sequence,
        )
        second = build_branch_constrained_planning_audit_report(
            strict_branch_action_oracle_labels=strict,
            branch_sequence_continuation_scorer_report=sequence,
        )

        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_branch_constrained_planning_audit_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
            sequence = _sequence_report(strict)
            strict_path = tmp / "strict.json"
            sequence_path = tmp / "sequence.json"
            output_path = tmp / "planning.json"
            strict_path.write_text(json.dumps(strict), encoding="utf-8")
            sequence_path.write_text(json.dumps(sequence), encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "mind_v3_branch_constrained_planning_audit",
                    "--strict-branch-action-oracle-labels",
                    str(strict_path),
                    "--branch-sequence-continuation-scorer",
                    str(sequence_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_branch_constrained_planning_audit.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(payload["coverage"]["strict_eval_label_count"], 3)


def _sequence_report(strict: dict[str, object]) -> dict[str, object]:
    support = _labels_with_trace_targets(seed_start=101, branch_prefix="support")
    return build_branch_sequence_continuation_scorer_report(
        support_branch_action_oracle_labels=support,
        strict_branch_action_oracle_labels=strict,
    )


def _comparison(branch_id: str, action: str, target_local_delta: float) -> dict[str, object]:
    return {
        "branch_id": branch_id,
        "seed": 13,
        "predicted_action": action,
        "predicted_mode": "exploit_resource" if action == "eat" else "other",
        "target_alive_delta": 0.0,
        "target_local_score_delta": target_local_delta,
        "terminal_alive_delta": 0.0,
        "birth_delta": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
