from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v178_transition_row_dataset_audit as v178,
    carrion_survivor_continuation_v180_transition_row_policy_training as v180,
)
from evolution_sim.mind.provenance import stable_payload_digest
from tests.test_mind_v3_carrion_survivor_continuation_v178_transition_row_dataset_audit import (
    _digest_without_exact,
    _report_exact_digest,
    _support_ready_rows,
    _write_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV180TransitionRowPolicyTrainingTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v180-transition-row-policy-training"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v180_transition_row_policy_training"
            ),
        )

    def test_valid_pinned_authorization_trains_artifact_without_runtime_promotion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows, v178_report = _authorized_v178_inputs(tmpdir)
            artifact_path = paths["root"] / "v180-artifact.json"
            output_path = paths["root"] / "v180-report.json"

            report = v180.run_carrion_survivor_continuation_v180_transition_row_policy_training(
                authorization_report_path=paths["v178_report"],
                transition_dataset_path=paths["dataset"],
                artifact_output_path=artifact_path,
                output_path=output_path,
                expected_authorization_report_exact_digest=v178_report[
                    "exact_digest"
                ],
                expected_dataset_digest=stable_payload_digest(rows),
                expected_source_producer=v178.V177_SOURCE_PRODUCER,
                run_evaluation=False,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertTrue(report["authorization_report_validation"]["passed"])
        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["training"]["ran"])
        self.assertTrue(report["training"]["passed"])
        self.assertTrue(report["training_ran"])
        self.assertTrue(report["training_artifact_created"])
        self.assertTrue(report["artifact"]["created"])
        self.assertEqual(report["artifact"]["digest"], stable_payload_digest(artifact))
        self.assertEqual(
            artifact["built_from"]["authorization_report_exact_digest"],
            v178_report["exact_digest"],
        )
        self.assertEqual(
            artifact["built_from"]["dataset_digest"],
            stable_payload_digest(rows),
        )
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["shadow_eval_ran"])
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v180_transition_row_policy_"
                "training_trained_shadow_eval_not_run_no_promotion"
            ),
        )
        self.assertEqual(report["exact_digest"], _digest_without_exact(report))

    def test_authorization_report_digest_mismatch_fails_closed_without_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows, _v178_report = _authorized_v178_inputs(tmpdir)
            artifact_path = paths["root"] / "v180-artifact.json"

            report = v180.run_carrion_survivor_continuation_v180_transition_row_policy_training(
                authorization_report_path=paths["v178_report"],
                transition_dataset_path=paths["dataset"],
                artifact_output_path=artifact_path,
                output_path=paths["root"] / "v180-report.json",
                expected_authorization_report_exact_digest="wrong",
                expected_dataset_digest=stable_payload_digest(rows),
                expected_source_producer=v178.V177_SOURCE_PRODUCER,
                run_evaluation=False,
            )

        self.assertFalse(report["authorization_report_validation"]["passed"])
        self.assertIn(
            "expected_exact_digest_matches",
            report["authorization_report_validation"]["failures"],
        )
        self.assertFalse(report["source_validation"]["passed"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["artifact"]["created"])
        self.assertFalse(artifact_path.exists())
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v180_transition_row_policy_"
                "training_source_invalid_closed_no_training"
            ),
        )
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["promotion_authorized"])

    def test_dataset_digest_mismatch_fails_closed_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _rows, v178_report = _authorized_v178_inputs(tmpdir)

            report = v180.run_carrion_survivor_continuation_v180_transition_row_policy_training(
                authorization_report_path=paths["v178_report"],
                transition_dataset_path=paths["dataset"],
                artifact_output_path=paths["root"] / "v180-artifact.json",
                output_path=paths["root"] / "v180-report.json",
                expected_authorization_report_exact_digest=v178_report[
                    "exact_digest"
                ],
                expected_dataset_digest="wrong",
                expected_source_producer=v178.V177_SOURCE_PRODUCER,
                run_evaluation=False,
            )

        self.assertFalse(report["authorization_report_validation"]["passed"])
        self.assertIn(
            "dataset_digest_matches",
            report["authorization_report_validation"]["failures"],
        )
        self.assertFalse(report["source_validation"]["passed"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])

    def test_non_authorizing_v178_report_digest_match_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            input_paths, rows = _write_inputs(tmpdir, rows)
            root = Path(tmpdir)
            v178_report_path = root / "v178-non-authorizing.json"
            v178_report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=input_paths["dataset"],
                v177_report_path=input_paths["v177_report"],
                output_path=v178_report_path,
                expected_dataset_digest=stable_payload_digest(rows),
            )
            artifact_path = root / "v180-artifact.json"

            report = v180.run_carrion_survivor_continuation_v180_transition_row_policy_training(
                authorization_report_path=v178_report_path,
                transition_dataset_path=input_paths["dataset"],
                artifact_output_path=artifact_path,
                output_path=root / "v180-report.json",
                expected_authorization_report_exact_digest=v178_report[
                    "exact_digest"
                ],
                expected_dataset_digest=stable_payload_digest(rows),
                expected_authorization_classification=v178_report["classification"][
                    "primary"
                ],
                expected_source_producer=v178.V177_SOURCE_PRODUCER,
                run_evaluation=False,
            )

        self.assertTrue(report["authorization_report_validation"]["exact_digest_valid"])
        self.assertTrue(
            report["authorization_report_validation"][
                "expected_exact_digest_matches"
            ]
        )
        self.assertFalse(report["authorization_report_validation"]["passed"])
        self.assertIn(
            "training_authorization_field_true",
            report["authorization_report_validation"]["failures"],
        )
        self.assertIn(
            "expected_source_report_exact_digest_provided",
            report["authorization_report_validation"]["failures"],
        )
        self.assertFalse(report["source_validation"]["passed"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["artifact"]["created"])
        self.assertFalse(artifact_path.exists())

    def test_cli_writes_parseable_report_and_prints_route_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows, v178_report = _authorized_v178_inputs(tmpdir)
            artifact_path = paths["root"] / "v180-artifact.json"
            output_path = paths["root"] / "v180-report.json"

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v180_transition_row_policy_training",
                    "--authorization-report",
                    str(paths["v178_report"]),
                    "--transition-dataset",
                    str(paths["dataset"]),
                    "--artifact-output",
                    str(artifact_path),
                    "--output",
                    str(output_path),
                    "--expected-authorization-report-exact-digest",
                    str(v178_report["exact_digest"]),
                    "--expected-dataset-digest",
                    stable_payload_digest(rows),
                    "--expected-source-producer",
                    v178.V177_SOURCE_PRODUCER,
                    "--skip-evaluation",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("training_ran=True", result.stdout)
        self.assertIn("training_artifact_created=True", result.stdout)
        self.assertIn("runtime_artifact_created=False", result.stdout)
        self.assertIn("promotion_authorized=False", result.stdout)
        self.assertTrue(report["training_ran"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])


def _authorized_v178_inputs(
    tmpdir: str,
) -> tuple[dict[str, Path], list[dict[str, object]], dict[str, object]]:
    rows = _support_ready_rows()
    input_paths, rows = _write_inputs(tmpdir, rows)
    root = Path(tmpdir)
    paths = {
        "root": root,
        "v177_report": input_paths["v177_report"],
        "dataset": input_paths["dataset"],
        "v178_report": root / "v178-authorized.json",
    }
    report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
        transition_dataset_path=paths["dataset"],
        v177_report_path=paths["v177_report"],
        output_path=paths["v178_report"],
        expected_v177_report_exact_digest=_report_exact_digest(paths["v177_report"]),
        expected_dataset_digest=stable_payload_digest(rows),
    )
    return paths, rows, report


if __name__ == "__main__":
    unittest.main()
