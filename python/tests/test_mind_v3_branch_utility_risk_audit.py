from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_branch_utility_risk_audit
from evolution_sim.mind.branch_action_oracle_labels import (
    build_branch_action_oracle_label_report,
)
from evolution_sim.mind.branch_utility_risk_audit import (
    MIND_V3_BRANCH_UTILITY_RISK_AUDIT_SCHEMA_VERSION,
    _catastrophic_class_report,
    _veto_then_max,
    build_branch_utility_risk_audit_report,
)
from python.tests.test_mind_v3_branch_action_oracle_labels import (
    _synthetic_audit_report,
)


class MindV3BranchUtilityRiskAuditTests(unittest.TestCase):
    def test_branch_utility_risk_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:branch-utility-risk-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_branch_utility_risk_audit"
            ),
        )

    def test_branch_utility_risk_audit_reports_schema_and_loo_split(self) -> None:
        labels = build_branch_action_oracle_label_report(_synthetic_audit_report())

        report = build_branch_utility_risk_audit_report(labels)

        self.assertEqual(
            report["schema_version"],
            MIND_V3_BRANCH_UTILITY_RISK_AUDIT_SCHEMA_VERSION,
        )
        self.assertFalse(report["contract"]["runtime_policy_trained"])
        self.assertEqual(
            report["contract"]["split_policy"],
            "leave_one_source_seed_out_v1",
        )
        self.assertFalse(
            report["contract"]["feature_contract"]["uses_fixture_identity"]
        )
        candidate_predictions = report["candidate_utility_predictions"]
        self.assertEqual(candidate_predictions["held_out_seed_leak_count"], 0)
        for item in candidate_predictions["candidate_scores"]:
            held_out_seed = str(item["held_out_seed"])
            self.assertNotIn(held_out_seed, item["neighbor_seed_counts"])
        self.assertIn("decision_rule_reports", report)
        self.assertIn("acceptance", report)

    def test_branch_utility_risk_audit_json_is_deterministic(self) -> None:
        labels = build_branch_action_oracle_label_report(_synthetic_audit_report())

        first = build_branch_utility_risk_audit_report(labels)
        second = build_branch_utility_risk_audit_report(labels)

        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_risk_veto_filters_target_death_before_utility(self) -> None:
        selected = _veto_then_max(
            [
                {
                    "action": "eat",
                    "predicted": {
                        "mean_target_local_utility": 100.0,
                        "target_death_risk": 0.9,
                    },
                },
                {
                    "action": "stay",
                    "predicted": {
                        "mean_target_local_utility": 10.0,
                        "target_death_risk": 0.0,
                    },
                },
            ],
            score_key="mean_target_local_utility",
            risk_keys=("target_death_risk",),
            risk_caps=(0.2,),
        )

        self.assertEqual(selected, "stay")

    def test_catastrophic_reporting_keeps_all_negative_examples(self) -> None:
        comparisons = [
            {
                "branch_id": "seed-41-worst",
                "seed": 41,
                "target_local_score_delta": -1055.0,
                "target_alive_delta": -1.0,
            },
            {
                "branch_id": "seed-13-negative",
                "seed": 13,
                "target_local_score_delta": -1.0,
                "target_alive_delta": 0.0,
            },
        ]

        report = _catastrophic_class_report(comparisons)

        self.assertEqual(report["target_death_negative_count"], 1)
        self.assertEqual(report["catastrophic_score_count"], 1)
        self.assertFalse(report["seed_41_catastrophic_class_avoided"])
        self.assertEqual(
            report["worst_seed_41_examples"][0]["branch_id"],
            "seed-41-worst",
        )

    def test_branch_utility_risk_audit_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            labels_path = tmp / "labels.json"
            output_path = tmp / "utility-risk.json"
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
                    "mind_v3_branch_utility_risk_audit",
                    "--branch-action-oracle-labels",
                    str(labels_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_branch_utility_risk_audit.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_BRANCH_UTILITY_RISK_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(payload["coverage"]["label_count"], 3)


if __name__ == "__main__":
    unittest.main()
