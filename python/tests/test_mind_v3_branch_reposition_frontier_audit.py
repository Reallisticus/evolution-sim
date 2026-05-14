from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_branch_reposition_frontier_audit
from evolution_sim.mind.branch_action_oracle_labels import (
    build_branch_action_oracle_label_report,
)
from evolution_sim.mind.branch_reposition_frontier_audit import (
    MIND_V3_REPOSITION_FRONTIER_AUDIT_SCHEMA_VERSION,
    V91_REPOSITION_DIRECTION_ACCURACY_FLOOR,
    build_reposition_frontier_audit_report,
)
from python.tests.test_mind_v3_branch_action_oracle_labels import (
    _synthetic_audit_report,
)


class MindV3BranchRepositionFrontierAuditTests(unittest.TestCase):
    def test_branch_reposition_frontier_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:branch-reposition-frontier-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_branch_reposition_frontier_audit"
            ),
        )

    def test_branch_reposition_frontier_audit_reports_decomposition_and_utility(
        self,
    ) -> None:
        audit = _synthetic_audit_report()
        labels = build_branch_action_oracle_label_report(audit)

        report = build_reposition_frontier_audit_report(
            labels,
            source_branch_action_oracle_audit=audit,
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_REPOSITION_FRONTIER_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["contract"]["support_floors"][
                "reposition_exact_direction_accuracy"
            ],
            V91_REPOSITION_DIRECTION_ACCURACY_FLOOR,
        )
        self.assertIn("mode_objective_support", report)
        self.assertIn("reposition_decomposition", report)
        self.assertIn("branch_utility", report)
        decomposition = report["reposition_decomposition"]
        self.assertIn("best_decoder", decomposition)
        self.assertIn("decoders", decomposition)
        utility = report["branch_utility"]
        self.assertIn("target_local_score_delta_summary", utility)
        self.assertFalse(
            report["acceptance"]["v91_reposition_frontier_diagnostic_accepted"]
        )

    def test_branch_reposition_frontier_audit_json_is_deterministic(self) -> None:
        audit = _synthetic_audit_report()
        labels = build_branch_action_oracle_label_report(audit)

        first = build_reposition_frontier_audit_report(
            labels,
            source_branch_action_oracle_audit=audit,
        )
        second = build_reposition_frontier_audit_report(
            labels,
            source_branch_action_oracle_audit=audit,
        )

        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_branch_reposition_frontier_audit_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            labels_path = tmp / "labels.json"
            audit_path = tmp / "audit.json"
            output_path = tmp / "frontier.json"
            audit = _synthetic_audit_report()
            labels_path.write_text(
                json.dumps(build_branch_action_oracle_label_report(audit)),
                encoding="utf-8",
            )
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "mind_v3_branch_reposition_frontier_audit",
                    "--branch-action-oracle-labels",
                    str(labels_path),
                    "--source-branch-action-oracle-audit",
                    str(audit_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_branch_reposition_frontier_audit.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_REPOSITION_FRONTIER_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(payload["coverage"]["label_count"], 3)


if __name__ == "__main__":
    unittest.main()
