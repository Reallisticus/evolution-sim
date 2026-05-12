from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_carrion_branch_explore
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


if __name__ == "__main__":
    unittest.main()
