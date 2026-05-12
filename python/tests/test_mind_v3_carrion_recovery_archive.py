from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_carrion_recovery_archive
from evolution_sim.mind.carrion_branch_explore import (
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_recovery_archive import (
    MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
    build_carrion_recovery_archive_report,
)


class MindV3CarrionRecoveryArchiveTests(unittest.TestCase):
    def test_recovery_archive_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-recovery-archive"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_recovery_archive"
            ),
        )

    def test_archive_keeps_survivor_and_failure_elites_and_exports_dataset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "recovery-dataset.jsonl"

            report = build_carrion_recovery_archive_report(
                branch_report=_synthetic_branch_report(),
                dataset_output_path=dataset_path,
                min_survivor_cells=2,
                min_failure_cells=1,
            )

            lines = dataset_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            report["schema_version"],
            MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
        )
        self.assertTrue(report["acceptance"]["archive_acceptance_passed"])
        self.assertGreaterEqual(report["aggregate"]["survivor_cell_count"], 2)
        self.assertGreaterEqual(report["aggregate"]["failure_cell_count"], 1)
        self.assertGreaterEqual(report["dataset"]["survivor_count"], 2)
        self.assertGreaterEqual(report["dataset"]["failure_count"], 1)
        outcome = report["aggregate"]["outcome_metrics"]
        self.assertEqual(outcome["terminal_survivor_run_count"], 3)
        self.assertEqual(outcome["total_births"], 36)
        self.assertIn("total_scavenger_animal_resource_events", outcome)
        self.assertEqual(len(lines), report["dataset"]["record_count"])
        first_record = json.loads(lines[0])
        self.assertIn("record_id", first_record)
        self.assertIn("outcome_metrics", first_record)
        self.assertEqual(
            first_record["schema_version"],
            "mind_v3_carrion_recovery_dataset_record_v1",
        )

    def test_recovery_archive_cli_writes_report_and_dataset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "archive.json"
            dataset_path = Path(tmpdir) / "dataset.jsonl"

            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_recovery_archive",
                    "--seeds",
                    "29",
                    "--ticks",
                    "8",
                    "--continuation-script",
                    "hydration_safe_carrion_cycle",
                    "--min-survivor-cells",
                    "1",
                    "--min-failure-cells",
                    "0",
                    "--dataset-output",
                    str(dataset_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_recovery_archive.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            dataset_lines = dataset_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
        )
        self.assertTrue(payload["acceptance"]["archive_acceptance_passed"])
        self.assertEqual(len(dataset_lines), payload["dataset"]["record_count"])


def _synthetic_branch_report() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_policy": "deterministic_post_contact_branch_explore_v1",
        "contract": {"ticks": 120, "seeds": [29, 37]},
        "aggregate": {
            "replay_verified": True,
            "branch_run_count": 4,
        },
        "acceptance": {
            "diagnostic_acceptance_passed": True,
        },
        "branch_points": [],
        "branch_runs": [
            _branch_run(
                seed=29,
                script="hydration_safe_carrion_cycle",
                alive=3,
                births=12,
                dominant_action="stay",
                dominant_share=0.24,
            ),
            _branch_run(
                seed=29,
                script="water_first_recovery",
                alive=1,
                births=6,
                dominant_action="drink",
                dominant_share=0.25,
            ),
            _branch_run(
                seed=37,
                script="conserve_after_carrion",
                alive=1,
                births=5,
                dominant_action="stay",
                dominant_share=0.54,
            ),
            _branch_run(
                seed=37,
                script="carrion_then_water",
                alive=0,
                births=13,
                dominant_action="move_north",
                dominant_share=0.20,
            ),
        ],
    }


def _branch_run(
    *,
    seed: int,
    script: str,
    alive: int,
    births: int,
    dominant_action: str,
    dominant_share: float,
) -> dict[str, object]:
    return {
        "branch_id": f"branch-{seed}",
        "seed": seed,
        "fixture": "carrion_only",
        "branch_tick": 0,
        "base_script": "hydration_safe_carrion_cycle",
        "continuation_script": script,
        "alive_agents": alive,
        "births": births,
        "deaths": 12 + births - alive,
        "contact": {
            "food_source": "carcass",
            "gained_energy": 0.1017,
            "after": {
                "energy_ratio": 0.5609,
                "hydration_ratio": 0.9423,
                "health_ratio": 1.0,
            },
        },
        "dominant_requested_action": dominant_action,
        "dominant_requested_action_share": dominant_share,
        "unique_requested_actions": 7,
        "heuristic_action_source_count": 0,
        "zero_heuristic_runtime_actions": True,
        "trajectory_path": f"output/branch-{seed}-{script}.jsonl.gz",
    }


if __name__ == "__main__":
    unittest.main()
