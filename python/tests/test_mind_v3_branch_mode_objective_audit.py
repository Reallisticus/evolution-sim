from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_branch_mode_objective_audit
from evolution_sim.mind.branch_action_oracle_labels import (
    build_branch_action_oracle_label_report,
)
from evolution_sim.mind.branch_mode_objective_audit import (
    MIND_V3_BRANCH_MODE_OBJECTIVE_AUDIT_SCHEMA_VERSION,
    OPTION_MODE_MAX_DOMINANT_PREDICTION_SHARE,
    OPTION_MODE_SUPPORT_ACCURACY_FLOOR,
    build_branch_mode_objective_audit_report,
)
from python.tests.test_mind_v3_branch_action_oracle_labels import (
    _synthetic_audit_report,
)


class MindV3BranchModeObjectiveAuditTests(unittest.TestCase):
    def test_branch_mode_objective_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:branch-mode-objective-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_branch_mode_objective_audit"
            ),
        )

    def test_branch_mode_objective_audit_reports_coverage_and_floors(self) -> None:
        labels = build_branch_action_oracle_label_report(_synthetic_audit_report())
        report = build_branch_mode_objective_audit_report(labels)

        self.assertEqual(
            report["schema_version"],
            MIND_V3_BRANCH_MODE_OBJECTIVE_AUDIT_SCHEMA_VERSION,
        )
        coverage = report["coverage"]
        self.assertEqual(coverage["label_count"], 3)
        self.assertEqual(
            coverage["labels_by_oracle_option_mode"],
            {"conserve": 1, "exploit_resource": 1, "recover_hydration": 1},
        )
        self.assertIn("logged_action_to_oracle_action_matrix", coverage)
        self.assertIn("target_delta_summary_vs_logged", coverage)
        support = report["support"]["option_mode"]
        self.assertEqual(
            report["contract"]["support_floors"]["option_mode_accuracy"],
            OPTION_MODE_SUPPORT_ACCURACY_FLOOR,
        )
        self.assertEqual(
            report["contract"]["support_floors"][
                "max_dominant_predicted_option_mode"
            ],
            OPTION_MODE_MAX_DOMINANT_PREDICTION_SHARE,
        )
        self.assertIn("best_mode_accuracy", support)
        self.assertFalse(report["acceptance"]["runtime_training_allowed"])

    def test_branch_mode_objective_audit_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            labels_path = tmp / "labels.json"
            output_path = tmp / "mode-objective.json"
            labels_path.write_text(
                json.dumps(
                    build_branch_action_oracle_label_report(
                        _synthetic_audit_report()
                    )
                ),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "mind_v3_branch_mode_objective_audit",
                    "--branch-action-oracle-labels",
                    str(labels_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_branch_mode_objective_audit.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_BRANCH_MODE_OBJECTIVE_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(payload["coverage"]["label_count"], 3)


if __name__ == "__main__":
    unittest.main()
