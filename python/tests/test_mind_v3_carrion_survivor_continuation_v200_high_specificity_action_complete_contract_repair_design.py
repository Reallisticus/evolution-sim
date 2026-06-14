from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design as v200,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_mind_v3_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response import (
    _write_valid_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV200RepairDesignTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v200-high-specificity-action-complete-contract-repair-design"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design"
            ),
        )

    def test_source_pin_failure_stops_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, _, artifact_digest = _write_valid_v199_inputs(tmpdir)
            report = v200.run_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design(
                v198_report_path=paths["v198_report"],
                v199_report_path=paths["v199_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v200_report"],
                expected_v198_report_exact_digest=v198_digest,
                expected_v199_report_exact_digest="bad-digest",
                expected_v195_artifact_digest=artifact_digest,
            )

        self.assertFalse(report["source_pin_validation"]["passed"])
        self.assertEqual(report["route_decision"]["selected_route"], v200.STOP_ROUTE)
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_missing_v198_report_fails_closed_to_source_pins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, v199_digest, artifact_digest = _write_valid_v199_inputs(
                tmpdir
            )
            missing_v198 = Path(tmpdir) / "missing-v198-report.json"
            report = v200.run_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design(
                v198_report_path=missing_v198,
                v199_report_path=paths["v199_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v200_report"],
                expected_v198_report_exact_digest=v198_digest,
                expected_v199_report_exact_digest=v199_digest,
                expected_v195_artifact_digest=artifact_digest,
            )

        self.assertFalse(report["source_pin_validation"]["passed"])
        self.assertIn("v198_report_loaded", report["source_pin_validation"]["failures"])
        self.assertEqual(report["route_decision"]["selected_route"], v200.STOP_ROUTE)
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_v198_v197_digest_mismatch_fails_closed_to_source_pins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, v199_digest, artifact_digest = _write_valid_v199_inputs(
                tmpdir,
                v198_observed_v197_digest="wrong-v197-digest",
            )
            report = v200.run_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design(
                v198_report_path=paths["v198_report"],
                v199_report_path=paths["v199_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v200_report"],
                expected_v198_report_exact_digest=v198_digest,
                expected_v199_report_exact_digest=v199_digest,
                expected_v195_artifact_digest=artifact_digest,
            )

        self.assertFalse(report["source_pin_validation"]["passed"])
        self.assertIn(
            "v198_observed_v197_report_digest_matches_expected",
            report["source_pin_validation"]["failures"],
        )
        self.assertEqual(report["route_decision"]["selected_route"], v200.STOP_ROUTE)

    def test_inconsistent_v199_facts_route_to_audit_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, v199_digest, artifact_digest = _write_valid_v199_inputs(
                tmpdir,
                absent_count=53841,
            )
            report = v200.run_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design(
                v198_report_path=paths["v198_report"],
                v199_report_path=paths["v199_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v200_report"],
                expected_v198_report_exact_digest=v198_digest,
                expected_v199_report_exact_digest=v199_digest,
                expected_v195_artifact_digest=artifact_digest,
            )

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertFalse(report["v199_fact_assessment"]["facts_consistent"])
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v200.V199_AUDIT_REPAIR_ROUTE,
        )

    def test_instrumentation_missing_routes_to_instrumentation_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, v199_digest, artifact_digest = _write_valid_v199_inputs(
                tmpdir,
                diagnostics_provenance="skipped",
            )
            report = v200.run_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design(
                v198_report_path=paths["v198_report"],
                v199_report_path=paths["v199_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v200_report"],
                expected_v198_report_exact_digest=v198_digest,
                expected_v199_report_exact_digest=v199_digest,
                expected_v195_artifact_digest=artifact_digest,
            )

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertFalse(report["v199_fact_assessment"]["instrumentation_present"])
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v200.INSTRUMENTATION_REPAIR_ROUTE,
        )

    def test_existing_artifact_action_incomplete_routes_to_source_contract_audit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, v199_digest, artifact_digest = _write_valid_v199_inputs(
                tmpdir
            )
            report = v200.run_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design(
                v198_report_path=paths["v198_report"],
                v199_report_path=paths["v199_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v200_report"],
                expected_v198_report_exact_digest=v198_digest,
                expected_v199_report_exact_digest=v199_digest,
                expected_v195_artifact_digest=artifact_digest,
            )

        self.assertEqual(
            report["v199_fact_assessment"]["primary_blocker"],
            "existing_artifact_action_incomplete_for_current_valid_actions",
        )
        self.assertEqual(
            report["action_complete_repair_design"]["selected_repair_class"],
            "source_contract_audit_for_missing_current_valid_actions",
        )
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v200.SOURCE_CONTRACT_AUDIT_ROUTE,
        )
        self.assertFalse(
            report["route_decision"][
                "future_explicit_slice_4_training_route_authorized"
            ]
        )

    def test_cli_writes_closed_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, v199_digest, artifact_digest = _write_valid_v199_inputs(
                tmpdir
            )
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design",
                    "--v198-report",
                    str(paths["v198_report"]),
                    "--v199-report",
                    str(paths["v199_report"]),
                    "--v195-artifact",
                    str(paths["v195_artifact"]),
                    "--output",
                    str(paths["v200_report"]),
                    "--expected-v198-report-exact-digest",
                    v198_digest,
                    "--expected-v199-report-exact-digest",
                    v199_digest,
                    "--expected-v195-artifact-digest",
                    artifact_digest,
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(paths["v200_report"].read_text(encoding="utf-8"))

        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("slice_4_training_consumed=False", result.stdout)
        self.assertIn("support_expansion_ran=False", result.stdout)
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v200.SOURCE_CONTRACT_AUDIT_ROUTE,
        )
        self.assertTrue(exact_digest_validation_report(report)["passed"])


def _write_valid_v199_inputs(
    tmpdir: str,
    *,
    absent_count: int = v200.EXPECTED_V199_ABSENT_COUNT,
    diagnostics_provenance: str = "real_replay",
    v198_observed_v197_digest: str = v200.EXPECTED_V197_REPORT_EXACT_DIGEST,
) -> tuple[dict[str, Path], str, str, str]:
    paths, _, artifact_digest = _write_valid_inputs(tmpdir)
    v198_report_path = Path(tmpdir) / "v198-report.json"
    v199_report_path = Path(tmpdir) / "v199-report.json"
    v200_report_path = Path(tmpdir) / "v200-report.json"
    present = v200.EXPECTED_V199_PRESENT_COUNT
    incomplete = v200.EXPECTED_V199_PRESENT_INCOMPLETE_COUNT
    v198_report = {
        "schema_version": (
            "m3_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair_report_v1"
        ),
        "policy": (
            "diagnostics_only_m3_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair_v1"
        ),
        "source_pin_validation": {
            "passed": True,
            "observed_v197_report_exact_digest": v198_observed_v197_digest,
            "observed_v197_route": (
                "v198_high_specificity_coverage_or_model_capacity_repair_before_slice_4_training_no_training"
            ),
            "checks": {
                "v197_inherited_v196_digest_matches": True,
                "v197_inherited_v195_report_digest_matches": True,
                "v197_inherited_v195_artifact_digest_matches": True,
            },
        },
    }
    v198_report["exact_digest"] = stable_payload_digest(v198_report)
    v198_report_path.write_text(
        json.dumps(v198_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    v198_digest = str(v198_report["exact_digest"])
    v199_report = {
        "schema_version": (
            "m3_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit_report_v1"
        ),
        "policy": (
            "diagnostics_only_m3_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit_v1"
        ),
        "classification": {"primary": v200.EXPECTED_V199_CLASSIFICATION},
        "source_pin_validation": {
            "passed": True,
            "expected_v198_report_exact_digest": v198_digest,
            "expected_v198_archive_path": v200.EXPECTED_V198_ARCHIVE_PATH,
        },
        "action_complete_audit": {
            "policy": (
                "m3_carrion_survivor_continuation_v199_"
                "high_specificity_action_complete_audit_v1"
            ),
            "ran": True,
            "diagnostics_only": True,
            "diagnostics_provenance": diagnostics_provenance,
            "diagnostics_override_used": False,
            "transition_value_action_override_enabled": False,
            "source_key_specificity_gate_enabled": True,
            "candidate_key_coverage_diagnostics_enabled": True,
            "training_rerun": False,
            "slice_4_training_consumed": False,
            "support_generation_ran": False,
            "support_expansion_ran": False,
            "combined": {
                "high_specificity_candidate_evaluated_count": (
                    v200.EXPECTED_V199_EVALUATED_COUNT
                ),
                "key_absent_count": absent_count,
                "key_present_count": present,
                "present_but_action_incomplete_count": incomplete,
                "present_complete_but_observed_support_floor_failed_count": 0,
                "imputed_valid_action_score_count": 0,
                "candidate_key_action_coverage_count": 215,
                "runtime_action_selection_changed_count": 0,
                "missing_current_valid_action_counts": {
                    "eat": 628,
                    "move_west": 475,
                },
                "by_seed": {
                    "carrion_only:13": {
                        "key_absent_count": 1,
                        "key_present_count": 2,
                        "present_but_action_incomplete_count": 2,
                        "missing_current_valid_action_counts": {"eat": 2},
                    }
                },
            },
        },
        "contract_assessment": {
            "primary_blocker": (
                "high_specificity_action_complete_or_support_floor_gap_confirmed"
            )
        },
        "route_decision": {
            "selected_route": v200.EXPECTED_V199_ROUTE,
            "recommended_next_route": v200.EXPECTED_V199_ROUTE,
        },
        "training_ran": False,
        "fit_ran": False,
        "training_artifact_created": False,
        "slice_4_training_started": False,
        "slice_4_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_integration_ran": False,
        "runtime_action_selection_changed": False,
        "runtime_policy_changed": False,
        "gate_relaxation_ran": False,
        "gate_relaxation_allowed": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "promotion_authorized": False,
        "non_promoted": True,
    }
    v199_report["exact_digest"] = stable_payload_digest(v199_report)
    v199_report_path.write_text(
        json.dumps(v199_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths = dict(paths)
    paths["v198_report"] = v198_report_path
    paths["v199_report"] = v199_report_path
    paths["v200_report"] = v200_report_path
    return paths, v198_digest, str(v199_report["exact_digest"]), artifact_digest


if __name__ == "__main__":
    unittest.main()
