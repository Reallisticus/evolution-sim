from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit as v199,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_mind_v3_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response import (
    _write_valid_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV199ActionCompleteAuditTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v199-high-specificity-action-complete-contract-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit"
            ),
        )

    def test_source_pin_failure_stops_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, artifact_digest = _write_valid_v198_inputs(tmpdir)
            report = v199.run_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit(
                v198_report_path=paths["v198_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v199_report"],
                expected_v198_report_exact_digest="bad-digest",
                expected_v195_artifact_digest=artifact_digest,
                run_diagnostic_replay=False,
            )

        self.assertFalse(report["source_pin_validation"]["passed"])
        self.assertEqual(report["route_decision"]["selected_route"], v199.STOP_ROUTE)
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_skipped_audit_fails_closed_to_instrumentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, artifact_digest = _write_valid_v198_inputs(tmpdir)
            report = v199.run_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit(
                v198_report_path=paths["v198_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v199_report"],
                expected_v198_report_exact_digest=v198_digest,
                expected_v195_artifact_digest=artifact_digest,
                run_diagnostic_replay=False,
            )

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertFalse(report["action_complete_audit"]["ran"])
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v199.INSTRUMENTATION_ROUTE,
        )
        self.assertFalse(
            report["route_decision"][
                "future_explicit_slice_4_training_route_authorized"
            ]
        )

    def test_audit_override_cannot_prove_real_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, artifact_digest = _write_valid_v198_inputs(tmpdir)
            report = v199.run_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit(
                v198_report_path=paths["v198_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v199_report"],
                expected_v198_report_exact_digest=v198_digest,
                expected_v195_artifact_digest=artifact_digest,
                run_diagnostic_replay=False,
                action_audit_override=_real_v199_audit(
                    _combined_audit_payload(
                        candidate_count=100,
                        key_absent_count=0,
                        key_present_count=100,
                        incomplete_count=100,
                    )
                ),
            )

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertFalse(report["contract_assessment"]["audit_real_replay_provenance"])
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v199.INSTRUMENTATION_ROUTE,
        )
        self.assertFalse(report["slice_4_training_consumed"])

    def test_absent_high_specificity_routes_to_public_model_capacity(self) -> None:
        audit = _real_v199_audit(
            _combined_audit_payload(
                candidate_count=100,
                key_absent_count=100,
                key_present_count=0,
                incomplete_count=0,
            )
        )
        assessment = v199.assess_action_complete_contract(audit)
        route = v199.route_decision_for_v199(
            source_validation={"passed": True},
            action_audit=audit,
            contract_assessment=assessment,
        )

        self.assertEqual(
            assessment["primary_blocker"],
            "high_specificity_absent_dominates",
        )
        self.assertEqual(route["selected_route"], v199.PUBLIC_MODEL_CAPACITY_ROUTE)
        self.assertFalse(route["direct_slice_4_training_allowed"])

    def test_present_action_incomplete_routes_to_repair_design(self) -> None:
        audit = _real_v199_audit(
            _combined_audit_payload(
                candidate_count=100,
                key_absent_count=90,
                key_present_count=10,
                incomplete_count=10,
            )
        )
        assessment = v199.assess_action_complete_contract(audit)
        route = v199.route_decision_for_v199(
            source_validation={"passed": True},
            action_audit=audit,
            contract_assessment=assessment,
        )

        self.assertEqual(
            assessment["primary_blocker"],
            "high_specificity_action_complete_or_support_floor_gap_confirmed",
        )
        self.assertEqual(route["selected_route"], v199.ACTION_COMPLETE_REPAIR_ROUTE)
        self.assertFalse(route["slice_4_training_consumed"])

    def test_cli_writes_closed_report_when_audit_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v198_digest, artifact_digest = _write_valid_v198_inputs(tmpdir)
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit",
                    "--v198-report",
                    str(paths["v198_report"]),
                    "--v195-artifact",
                    str(paths["v195_artifact"]),
                    "--output",
                    str(paths["v199_report"]),
                    "--expected-v198-report-exact-digest",
                    v198_digest,
                    "--expected-v195-artifact-digest",
                    artifact_digest,
                    "--skip-diagnostic-replay",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(paths["v199_report"].read_text(encoding="utf-8"))

        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("slice_4_training_consumed=False", result.stdout)
        self.assertIn("support_expansion_ran=False", result.stdout)
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v199.INSTRUMENTATION_ROUTE,
        )
        self.assertTrue(exact_digest_validation_report(report)["passed"])


def _write_valid_v198_inputs(tmpdir: str) -> tuple[dict[str, Path], str, str]:
    paths, _, artifact_digest = _write_valid_inputs(tmpdir)
    v198_report_path = Path(tmpdir) / "v198-report.json"
    v199_report_path = Path(tmpdir) / "v199-report.json"
    v198_report = {
        "schema_version": (
            "m3_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair_report_v1"
        ),
        "policy": (
            "diagnostics_only_m3_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair_v1"
        ),
        "classification": {"primary": v199.EXPECTED_V198_CLASSIFICATION},
        "source_pin_validation": {"passed": True},
        "high_specificity_probe": _v198_real_probe(
            _v198_combined_probe_payload()
        ),
        "coverage_assessment": {
            "primary_blocker": v199.EXPECTED_V198_PRIMARY_BLOCKER,
            "live_high_specificity_candidate_breakdown": {
                "present_but_action_incomplete_count": 20
            },
        },
        "route_decision": {
            "selected_route": v199.EXPECTED_V198_ROUTE,
            "recommended_next_route": v199.EXPECTED_V198_ROUTE,
            "primary_blocker": v199.EXPECTED_V198_PRIMARY_BLOCKER,
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
    v198_report["exact_digest"] = stable_payload_digest(v198_report)
    v198_report_path.write_text(
        json.dumps(v198_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths = dict(paths)
    paths["v198_report"] = v198_report_path
    paths["v199_report"] = v199_report_path
    return paths, str(v198_report["exact_digest"]), artifact_digest


def _v198_combined_probe_payload() -> dict[str, object]:
    return {
        "decision_count": 100,
        "candidate_key_coverage_decision_count": 100,
        "missing_candidate_key_coverage_count": 0,
        "runtime_action_selection_changed_count": 0,
        "high_specificity_any_complete_count": 0,
        "candidate_key_category_breakdown": {
            "self_nav_feature_hit": {
                "present_but_action_incomplete_count": 20,
            }
        },
    }


def _v198_real_probe(combined: dict[str, object]) -> dict[str, object]:
    return {
        "policy": (
            "m3_carrion_survivor_continuation_v198_"
            "high_specificity_coverage_probe_v1"
        ),
        "ran": True,
        "diagnostics_only": True,
        "diagnostics_provenance": "real_replay",
        "diagnostics_override_used": False,
        "real_replay_provenance": True,
        "transition_value_action_override_enabled": False,
        "source_key_specificity_gate_enabled": True,
        "candidate_key_coverage_diagnostics_enabled": True,
        "training_rerun": False,
        "slice_4_training_consumed": False,
        "combined": combined,
    }


def _combined_audit_payload(
    *,
    candidate_count: int,
    key_absent_count: int,
    key_present_count: int,
    incomplete_count: int,
) -> dict[str, object]:
    return {
        "decision_count": 100,
        "candidate_key_coverage_decision_count": 100,
        "missing_candidate_key_coverage_count": 0,
        "runtime_action_selection_changed_count": 0,
        "high_specificity_candidate_evaluated_count": candidate_count,
        "key_absent_count": key_absent_count,
        "key_present_count": key_present_count,
        "present_but_action_incomplete_count": incomplete_count,
        "present_complete_but_observed_support_floor_failed_count": 0,
        "imputed_valid_action_score_count": 0,
        "present_complete_floor_satisfied_but_unclear_best_count": 0,
    }


def _real_v199_audit(combined: dict[str, object]) -> dict[str, object]:
    return {
        "policy": (
            "m3_carrion_survivor_continuation_v199_"
            "high_specificity_action_complete_audit_v1"
        ),
        "ran": True,
        "diagnostics_only": True,
        "diagnostics_provenance": "real_replay",
        "diagnostics_override_used": False,
        "real_replay_provenance": True,
        "transition_value_action_override_enabled": False,
        "source_key_specificity_gate_enabled": True,
        "candidate_key_coverage_diagnostics_enabled": True,
        "training_rerun": False,
        "slice_4_training_consumed": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "combined": combined,
    }


if __name__ == "__main__":
    unittest.main()
