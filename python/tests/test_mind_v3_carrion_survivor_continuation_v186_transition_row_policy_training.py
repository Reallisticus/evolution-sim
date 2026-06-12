from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v178_transition_row_dataset_audit as v178,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v185_v183_target_resolution_repair as v185,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v186_transition_row_policy_training as v186,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_mind_v3_carrion_survivor_continuation_v178_transition_row_dataset_audit import (
    _digest_without_exact,
    _report_exact_digest,
)
from python.tests.test_mind_v3_carrion_survivor_continuation_v185_v183_target_resolution_repair import (
    _read_jsonl,
    _write_v184_failure_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV186TransitionRowPolicyTrainingTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v186-transition-row-policy-training"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v186_transition_row_policy_training"
            ),
        )

    def test_valid_pinned_v185_audit_trains_slice_2_artifact_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows, audit = _authorized_v185_inputs(tmpdir)
            artifact_path = paths["root"] / "v186-artifact.json"
            report = v186.run_carrion_survivor_continuation_v186_transition_row_policy_training(
                authorization_report_path=paths["v185_audit"],
                transition_dataset_path=paths["repaired_dataset"],
                artifact_output_path=artifact_path,
                output_path=paths["root"] / "v186-report.json",
                expected_authorization_report_exact_digest=str(audit["exact_digest"]),
                expected_dataset_digest=stable_payload_digest(rows),
                run_evaluation=False,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertTrue(report["authorization_report_validation"]["passed"])
        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["training"]["ran"])
        self.assertTrue(report["training"]["passed"])
        self.assertTrue(report["training_ran"])
        self.assertTrue(report["training_artifact_created"])
        self.assertTrue(report["slice_2_training_consumed"])
        self.assertEqual(report["training_slice_budget"]["current_slices_consumed"], 2)
        self.assertEqual(report["artifact"]["digest"], stable_payload_digest(artifact))
        self.assertEqual(
            artifact["artifact_policy"],
            v186.M3_CARRION_SURVIVOR_CONTINUATION_V186_ARTIFACT_POLICY,
        )
        self.assertEqual(
            artifact["built_from"]["source_producer"],
            v178.V185_SOURCE_PRODUCER,
        )
        self.assertEqual(
            artifact["built_from"]["authorization_route"],
            v178.V186_SLICE_2_TRAINING_ROUTE,
        )
        self.assertEqual(artifact["built_from"]["training_slice_index"], 2)
        self.assertFalse(artifact["built_from"]["first_opt_in_training_slice"])
        self.assertTrue(artifact["built_from"]["slice_2_opt_in_training_slice"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["shadow_eval_ran"])
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v186_transition_row_policy_"
                "training_slice_2_trained_shadow_eval_not_run_no_promotion"
            ),
        )
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_authorization_report_digest_mismatch_fails_closed_without_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows, _audit = _authorized_v185_inputs(tmpdir)
            artifact_path = paths["root"] / "v186-artifact.json"
            report = v186.run_carrion_survivor_continuation_v186_transition_row_policy_training(
                authorization_report_path=paths["v185_audit"],
                transition_dataset_path=paths["repaired_dataset"],
                artifact_output_path=artifact_path,
                output_path=paths["root"] / "v186-report.json",
                expected_authorization_report_exact_digest="wrong",
                expected_dataset_digest=stable_payload_digest(rows),
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
        self.assertFalse(report["slice_2_training_consumed"])
        self.assertFalse(report["artifact"]["created"])
        self.assertFalse(artifact_path.exists())

    def test_route_mismatch_fails_closed_even_with_matching_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows, audit = _authorized_v185_inputs(tmpdir)
            bad_audit = dict(audit)
            bad_audit["route_recommendation"] = dict(audit["route_recommendation"])
            bad_audit["route_recommendation"]["recommended_next_route"] = (
                v178.V185_SLICE_2_TRAINING_ROUTE
            )
            bad_audit["exact_digest"] = _digest_without_exact(bad_audit)
            bad_path = paths["root"] / "v185-audit-wrong-route.json"
            bad_path.write_text(json.dumps(bad_audit, sort_keys=True, indent=2) + "\n")
            artifact_path = paths["root"] / "v186-artifact.json"

            report = v186.run_carrion_survivor_continuation_v186_transition_row_policy_training(
                authorization_report_path=bad_path,
                transition_dataset_path=paths["repaired_dataset"],
                artifact_output_path=artifact_path,
                output_path=paths["root"] / "v186-report.json",
                expected_authorization_report_exact_digest=str(bad_audit["exact_digest"]),
                expected_dataset_digest=stable_payload_digest(rows),
                run_evaluation=False,
            )

        self.assertTrue(report["authorization_report_validation"]["exact_digest_valid"])
        self.assertFalse(report["authorization_report_validation"]["passed"])
        self.assertIn(
            "route_matches_v186_slice_2_opt_in",
            report["authorization_report_validation"]["failures"],
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["slice_2_training_consumed"])
        self.assertFalse(artifact_path.exists())

    def test_source_and_classification_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows, audit = _authorized_v185_inputs(tmpdir)
            bad_audit = dict(audit)
            bad_audit["source_validation"] = dict(audit["source_validation"])
            bad_audit["classification"] = dict(audit["classification"])
            bad_audit["source_validation"]["source_producer"] = (
                v178.V183_SOURCE_PRODUCER
            )
            bad_audit["classification"]["primary"] = (
                "m3_carrion_survivor_continuation_v185_repaired_transition_row_"
                "dataset_audit_valid_support_ready_digest_pins_required_no_training"
            )
            bad_audit["exact_digest"] = _digest_without_exact(bad_audit)
            bad_path = paths["root"] / "v185-audit-wrong-source-classification.json"
            bad_path.write_text(json.dumps(bad_audit, sort_keys=True, indent=2) + "\n")
            artifact_path = paths["root"] / "v186-artifact.json"

            report = v186.run_carrion_survivor_continuation_v186_transition_row_policy_training(
                authorization_report_path=bad_path,
                transition_dataset_path=paths["repaired_dataset"],
                artifact_output_path=artifact_path,
                output_path=paths["root"] / "v186-report.json",
                expected_authorization_report_exact_digest=str(bad_audit["exact_digest"]),
                expected_dataset_digest=stable_payload_digest(rows),
                run_evaluation=False,
            )

        self.assertFalse(report["authorization_report_validation"]["passed"])
        self.assertIn(
            "classification_matches_v185_authorized",
            report["authorization_report_validation"]["failures"],
        )
        self.assertIn(
            "source_producer_matches_v185_repair",
            report["authorization_report_validation"]["failures"],
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["slice_2_training_consumed"])
        self.assertFalse(artifact_path.exists())

    def test_cli_writes_parseable_report_and_prints_slice_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows, audit = _authorized_v185_inputs(tmpdir)
            artifact_path = paths["root"] / "v186-artifact.json"
            output_path = paths["root"] / "v186-report.json"

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v186_transition_row_policy_training",
                    "--authorization-report",
                    str(paths["v185_audit"]),
                    "--transition-dataset",
                    str(paths["repaired_dataset"]),
                    "--artifact-output",
                    str(artifact_path),
                    "--output",
                    str(output_path),
                    "--expected-authorization-report-exact-digest",
                    str(audit["exact_digest"]),
                    "--expected-dataset-digest",
                    stable_payload_digest(rows),
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
        self.assertIn("slice_2_training_consumed=True", result.stdout)
        self.assertIn("current_slices_consumed=2", result.stdout)
        self.assertIn(f"required_route={v178.V186_SLICE_2_TRAINING_ROUTE}", result.stdout)
        self.assertTrue(report["training_ran"])
        self.assertTrue(report["slice_2_training_consumed"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])


def _authorized_v185_inputs(
    tmpdir: str,
) -> tuple[dict[str, Path], list[dict[str, object]], dict[str, object]]:
    paths, report_digest, dataset_digest = _write_v184_failure_inputs(tmpdir)
    root = Path(tmpdir)
    paths["root"] = root
    repair = v185.run_carrion_survivor_continuation_v185_v183_target_resolution_repair(
        v184_report_path=paths["v184_report"],
        v183_transition_dataset_path=paths["dataset"],
        output_path=paths["v185_report"],
        repaired_transition_dataset_output_path=paths["repaired_dataset"],
        expected_v184_report_exact_digest=report_digest,
        expected_v183_report_exact_digest=_report_exact_digest(paths["v183_report"]),
        expected_v183_dataset_digest=dataset_digest,
    )
    rows = _read_jsonl(paths["repaired_dataset"])
    audit = v185.run_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit(
        v185_report_path=paths["v185_report"],
        transition_dataset_path=paths["repaired_dataset"],
        output_path=paths["v185_audit"],
        expected_v185_report_exact_digest=str(repair["exact_digest"]),
        expected_dataset_digest=stable_payload_digest(rows),
    )
    return paths, rows, audit


if __name__ == "__main__":
    unittest.main()
