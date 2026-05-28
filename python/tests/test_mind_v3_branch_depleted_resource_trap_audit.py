from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_branch_depleted_resource_trap_audit
from evolution_sim.mind.branch_action_oracle_labels import (
    build_branch_action_oracle_label_report,
)
from evolution_sim.mind.branch_depleted_resource_trap_audit import (
    MIND_V3_DEPLETED_RESOURCE_TRAP_AUDIT_SCHEMA_VERSION,
    _trap_signature,
    build_depleted_resource_trap_audit_report,
)
from python.tests.test_mind_v3_branch_action_oracle_labels import (
    _synthetic_audit_report,
)


class MindV3BranchDepletedResourceTrapAuditTests(unittest.TestCase):
    def test_branch_depleted_resource_trap_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:branch-depleted-resource-trap-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_branch_depleted_resource_trap_audit"
            ),
        )

    def test_trap_signature_detects_adjacent_resource_trap(self) -> None:
        row = {
            "before": {"energy_ratio": 0.1},
            "action_mask": {
                "eat": True,
                "move_south": True,
                "move_east": False,
                "move_west": False,
                "move_north": False,
            },
            "public_history_trace": [{"energy_ratio_delta": -0.04}],
            "compact_state": {
                "self": {"energy_ratio": 0.1},
                "center": {"food": 0.005, "carcass": 0.0, "carrion_signal": 0.0},
                "adjacent": {
                    "south": {
                        "food": 0.13,
                        "carcass": 0.37,
                        "carrion_signal": 0.35,
                    }
                },
            },
        }

        signature = _trap_signature(row)

        self.assertTrue(signature["is_trap"])
        self.assertEqual(signature["best_adjacent_move_action"], "move_south")
        self.assertLess(signature["current_resource"], 0.08)
        self.assertGreater(signature["best_adjacent_resource"], 0.12)

    def test_depleted_resource_trap_audit_reports_schema_and_contract(self) -> None:
        labels = build_branch_action_oracle_label_report(_synthetic_audit_report())

        report = build_depleted_resource_trap_audit_report(
            support_branch_action_oracle_labels=labels,
            strict_branch_action_oracle_labels=labels,
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_DEPLETED_RESOURCE_TRAP_AUDIT_SCHEMA_VERSION,
        )
        self.assertFalse(report["contract"]["runtime_policy_trained"])
        self.assertFalse(
            report["contract"]["feature_contract"]["uses_fixture_identity"]
        )
        self.assertIn("trap_state_discovery", report)
        self.assertIn("decision_rule_reports", report)
        self.assertFalse(
            report["acceptance"]["v93_depleted_resource_trap_diagnostic_accepted"]
        )

    def test_depleted_resource_trap_audit_uses_replay_verified_trap_source_rows(
        self,
    ) -> None:
        audit = _synthetic_audit_report()
        audit["contract"]["branch_selection_policy"] = "depleted_resource_trap_v1"  # type: ignore[index]
        audit["discovery"] = [
            {
                "seed": 37,
                "eligible_row_count": 3,
                "eligible_depleted_resource_trap_row_count": 3,
                "selected_depleted_resource_trap_row_count": 3,
            }
        ]
        labels = build_branch_action_oracle_label_report(audit)

        report = build_depleted_resource_trap_audit_report(
            support_branch_action_oracle_labels=labels,
            strict_branch_action_oracle_labels=labels,
            support_branch_action_oracle_audit=audit,
        )

        coverage = report["coverage"]
        self.assertEqual(coverage["support_label_count"], 3)
        self.assertEqual(coverage["support_trap_row_count"], 3)
        self.assertEqual(coverage["support_signature_trap_row_count"], 0)
        self.assertEqual(
            coverage["support_source_selection"][
                "selected_depleted_resource_trap_row_count"
            ],
            3,
        )

    def test_depleted_resource_trap_audit_json_is_deterministic(self) -> None:
        labels = build_branch_action_oracle_label_report(_synthetic_audit_report())

        first = build_depleted_resource_trap_audit_report(
            support_branch_action_oracle_labels=labels,
            strict_branch_action_oracle_labels=labels,
        )
        second = build_depleted_resource_trap_audit_report(
            support_branch_action_oracle_labels=labels,
            strict_branch_action_oracle_labels=labels,
        )

        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_depleted_resource_trap_audit_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            support_path = tmp / "support.json"
            strict_path = tmp / "strict.json"
            output_path = tmp / "trap.json"
            labels = build_branch_action_oracle_label_report(
                _synthetic_audit_report()
            )
            support_path.write_text(json.dumps(labels), encoding="utf-8")
            strict_labels = copy.deepcopy(labels)
            strict_path.write_text(json.dumps(strict_labels), encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "mind_v3_branch_depleted_resource_trap_audit",
                    "--support-branch-action-oracle-labels",
                    str(support_path),
                    "--strict-branch-action-oracle-labels",
                    str(strict_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_branch_depleted_resource_trap_audit.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_DEPLETED_RESOURCE_TRAP_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(payload["coverage"]["support_label_count"], 3)


if __name__ == "__main__":
    unittest.main()
