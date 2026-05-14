from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import (
    mind_v3_branch_action_oracle_audit,
    mind_v3_carrion_branch_explore,
)
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.branch_action_oracle_audit import (
    MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
    build_branch_action_oracle_audit_report,
)
from evolution_sim.mind.carrion_branch_explore import (
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
    build_carrion_branch_explore_report,
)


class MindV3CarrionBranchExploreTests(unittest.TestCase):
    def test_branch_explore_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-branch-explore"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_branch_explore"
            ),
        )
        self.assertEqual(
            package["scripts"]["sim:mind:v3:branch-action-oracle-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_branch_action_oracle_audit"
            ),
        )

    def test_report_branches_from_post_contact_state_and_verifies_replay(self) -> None:
        report = build_carrion_branch_explore_report(
            seeds=(29,),
            ticks=8,
            continuation_scripts=("hydration_safe_carrion_cycle",),
            verify_replay=True,
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        )
        self.assertTrue(report["scope"]["state_restore_available"])
        self.assertEqual(report["aggregate"]["branch_point_count"], 1)
        self.assertEqual(report["aggregate"]["branch_run_count"], 1)
        self.assertEqual(report["aggregate"]["positive_seed_count"], 1)
        self.assertTrue(report["aggregate"]["replay_verified"])
        outcome = report["aggregate"]["outcome_metrics"]
        self.assertEqual(outcome["terminal_survivor_run_count"], 1)
        self.assertGreater(outcome["total_terminal_alive_agents"], 0)
        self.assertIn("total_scavenger_animal_resource_events", outcome)
        self.assertTrue(report["acceptance"]["diagnostic_acceptance_passed"])
        branch = report["branch_points"][0]
        self.assertEqual(branch["contact"]["food_source"], "carcass")
        run = report["branch_runs"][0]
        self.assertGreater(run["alive_agents"], 0)
        self.assertIn("outcome_metrics", run)
        self.assertIn("scavenging", run["outcome_metrics"])
        self.assertEqual(run["heuristic_action_source_count"], 0)
        self.assertTrue(run["replay_verification"]["verified"])

    def test_branch_explore_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "branch-explore.json"

            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_branch_explore",
                    "--seeds",
                    "29",
                    "--ticks",
                    "8",
                    "--continuation-script",
                    "hydration_safe_carrion_cycle",
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_branch_explore.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        )
        self.assertEqual(payload["aggregate"]["branch_point_count"], 1)
        self.assertTrue(payload["acceptance"]["diagnostic_acceptance_passed"])

    def test_branch_action_oracle_audit_finds_changed_action_gain(self) -> None:
        report = build_branch_action_oracle_audit_report(
            seeds=(37,),
            ticks=70,
            candidate_actions=("drink", "eat", "stay"),
            max_branch_points_per_seed=1,
            verify_replay=True,
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
        )
        self.assertGreaterEqual(report["aggregate"]["branch_point_count"], 1)
        self.assertGreaterEqual(report["aggregate"]["oracle_changed_action_count"], 1)
        self.assertGreaterEqual(
            report["aggregate"]["terminal_alive_gain_total_vs_logged"],
            1,
        )
        self.assertEqual(report["aggregate"]["heuristic_action_source_count"], 0)
        self.assertTrue(report["aggregate"]["replay_verified"])
        self.assertTrue(report["acceptance"]["diagnostic_acceptance_passed"])
        first = report["aggregate"]["first_material_oracle_gain"]
        self.assertIsNotNone(first)
        self.assertNotEqual(first["logged_action"], first["oracle_best_action"])
        action_run = report["branch_results"][0]["action_runs"][0]
        self.assertIn("first_action_outcome", action_run)
        self.assertIn("target_horizon_trace", action_run)
        self.assertIn("population_horizon_trace", action_run)
        self.assertGreaterEqual(len(action_run["target_horizon_trace"]), 1)
        self.assertGreaterEqual(len(action_run["population_horizon_trace"]), 1)
        policy_state = report["branch_points"][0]["policy_state"]
        self.assertEqual(
            policy_state["observation_input"]["schema_version"],
            "mind_observation_v3",
        )
        self.assertEqual(
            set(policy_state["action_mask"]),
            set(ACTION_NAMES),
        )
        self.assertIn("public_history_trace", policy_state)
        self.assertIsInstance(policy_state["public_history_trace"], list)

    def test_branch_action_oracle_audit_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "branch-action-oracle.json"

            with patch(
                "sys.argv",
                [
                    "mind_v3_branch_action_oracle_audit",
                    "--seeds",
                    "37",
                    "--ticks",
                    "30",
                    "--max-branch-points-per-seed",
                    "1",
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_branch_action_oracle_audit.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(payload["aggregate"]["branch_point_count"], 1)
        self.assertIn("acceptance", payload)


if __name__ == "__main__":
    unittest.main()
