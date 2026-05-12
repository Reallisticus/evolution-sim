from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_carrion_recovery_distill
from evolution_sim.mind.carrion_recovery_archive import (
    build_carrion_recovery_archive_report,
    write_carrion_recovery_archive_report,
)
from evolution_sim.mind.carrion_recovery_distill import (
    MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
    build_carrion_recovery_distillation_report,
)


class MindV3CarrionRecoveryDistillTests(unittest.TestCase):
    def test_recovery_distill_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-recovery-distill"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_recovery_distill"
            ),
        )

    def test_recovery_distill_trains_artifact_and_reports_outcomes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            horizon_path = tmp_path / "horizon-labels.json"
            artifact_path = tmp_path / "artifact.json"
            evaluation_path = tmp_path / "evaluation.json"

            report = build_carrion_recovery_distillation_report(
                archive_report=archive,
                horizons=(1,),
                hidden_units=4,
                eval_seeds=(29,),
                eval_ticks=8,
                fixture_names=("carrion_only",),
                fixture_seeds=(29,),
                fixture_ticks=8,
                horizon_output_path=horizon_path,
                artifact_output_path=artifact_path,
                evaluation_output_path=evaluation_path,
            )

            self.assertEqual(
                report["schema_version"],
                MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
            )
            self.assertTrue(report["acceptance"]["data_path_acceptance_passed"])
            self.assertGreater(report["training"]["selected_trajectory_count"], 0)
            self.assertGreater(report["training"]["trained_record_count"], 0)
            self.assertTrue(horizon_path.exists())
            self.assertTrue(artifact_path.exists())
            self.assertTrue(evaluation_path.exists())
            candidate = report["evaluation"]["open"]["comparison"][
                "mind_v3_recovery_distilled"
            ]["aggregate"]
            self.assertEqual(candidate["heuristic_action_source_count"], 0)
            self.assertIn("outcome_metrics", candidate)
            self.assertIn("total_births", candidate["outcome_metrics"])
            self.assertIn(
                "total_scavenger_carcass_events",
                candidate["outcome_metrics"],
            )

    def test_recovery_distill_cli_writes_report_and_prints_counters(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            archive_path = tmp_path / "archive.json"
            output_path = tmp_path / "distill.json"
            horizon_path = tmp_path / "horizon-labels.json"
            artifact_path = tmp_path / "artifact.json"
            evaluation_path = tmp_path / "evaluation.json"
            write_carrion_recovery_archive_report(archive, archive_path)

            stdout = StringIO()
            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_recovery_distill",
                    "--archive-report",
                    str(archive_path),
                    "--horizons",
                    "1",
                    "--hidden-units",
                    "4",
                    "--eval-seeds",
                    "29",
                    "--eval-ticks",
                    "8",
                    "--fixture-names",
                    "carrion_only",
                    "--fixture-seeds",
                    "29",
                    "--fixture-ticks",
                    "8",
                    "--horizon-output",
                    str(horizon_path),
                    "--artifact-output",
                    str(artifact_path),
                    "--evaluation-output",
                    str(evaluation_path),
                    "--output",
                    str(output_path),
                ],
            ), redirect_stdout(stdout):
                mind_v3_carrion_recovery_distill.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
        )
        self.assertTrue(payload["acceptance"]["data_path_acceptance_passed"])
        self.assertIn("open_candidate_total_births=", stdout.getvalue())
        self.assertIn(
            "open_candidate_vs_linear_alive_agents_mean_delta=",
            stdout.getvalue(),
        )
        self.assertIn(
            "fixture_candidate_carrion_only_total_scavenger_carcass_events=",
            stdout.getvalue(),
        )


def _build_small_recovery_archive(tmp_path: Path) -> dict[str, object]:
    report = build_carrion_recovery_archive_report(
        seeds=(29,),
        ticks=8,
        continuation_scripts=("hydration_safe_carrion_cycle",),
        trajectory_output_dir=tmp_path / "trajectories",
        max_dataset_records_per_class=1,
        min_survivor_cells=1,
        min_failure_cells=0,
    )
    if not report["acceptance"]["archive_acceptance_passed"]:
        raise AssertionError(report["acceptance"])
    return report


if __name__ == "__main__":
    unittest.main()
