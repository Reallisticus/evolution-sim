from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v197_coverage_abstention_repair_design as v197,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair as v198,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from python.tests.test_mind_v3_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response import (
    _write_valid_inputs,
)
from python.tests.test_mind_v3_carrion_survivor_continuation_v197_coverage_abstention_repair_design import (
    _after_specificity_gate_diagnostics,
    _write_valid_v196_report,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV198HighSpecificityTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v198-high-specificity-coverage-or-model-capacity-repair"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair"
            ),
        )

    def test_skipped_probe_fails_closed_to_instrumentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                paths,
                v197_digest,
                v196_digest,
                v195_report_digest,
                artifact_digest,
            ) = _write_valid_v197_inputs(tmpdir)
            output_path = paths["v198_report"]

            report = v198.run_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair(
                v197_report_path=paths["v197_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=output_path,
                expected_v197_report_exact_digest=v197_digest,
                expected_v196_report_exact_digest=v196_digest,
                expected_v195_report_exact_digest=v195_report_digest,
                expected_v195_artifact_digest=artifact_digest,
                run_diagnostic_replay=False,
            )

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertFalse(report["high_specificity_probe"]["ran"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v198.INSTRUMENTATION_ROUTE,
        )
        self.assertFalse(
            report["route_decision"][
                "future_explicit_slice_4_training_route_authorized"
            ]
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_probe_override_cannot_authorize_future_slice_4(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                paths,
                v197_digest,
                v196_digest,
                v195_report_digest,
                artifact_digest,
            ) = _write_valid_v197_inputs(tmpdir)
            report = v198.run_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair(
                v197_report_path=paths["v197_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v198_report"],
                expected_v197_report_exact_digest=v197_digest,
                expected_v196_report_exact_digest=v196_digest,
                expected_v195_report_exact_digest=v195_report_digest,
                expected_v195_artifact_digest=artifact_digest,
                run_diagnostic_replay=False,
                probe_diagnostics_override=_real_probe(
                    _combined_probe_payload(
                        decision_count=100,
                        high_present_count=50,
                        high_complete_count=50,
                        high_floor_count=50,
                        high_selected_count=50,
                    )
                ),
            )

        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v198.INSTRUMENTATION_ROUTE,
        )
        self.assertFalse(
            report["coverage_assessment"]["probe_real_replay_provenance"]
        )
        self.assertEqual(
            report["coverage_assessment"]["primary_blocker"],
            "probe_not_real_replay",
        )
        self.assertFalse(report["slice_4_training_consumed"])

    def test_absent_high_specificity_public_states_routes_to_model_capacity(
        self,
    ) -> None:
        probe = _real_probe(
            _combined_probe_payload(
                decision_count=100,
                high_present_count=0,
                high_complete_count=0,
                high_floor_count=0,
                high_selected_count=0,
            )
        )
        assessment = v198.assess_high_specificity_coverage(
            artifact_table={"high_specificity_key_count": 12},
            probe=probe,
        )
        route = v198.route_decision_for_v198(
            source_validation={"passed": True},
            probe=probe,
            coverage_assessment=assessment,
        )

        self.assertEqual(
            assessment["primary_blocker"],
            "absent_high_specificity_public_states",
        )
        self.assertEqual(route["selected_route"], v198.PUBLIC_MODEL_CAPACITY_ROUTE)
        self.assertFalse(route["direct_slice_4_training_allowed"])

    def test_low_live_high_specificity_overlap_routes_to_model_capacity(
        self,
    ) -> None:
        probe = _real_probe(
            _combined_probe_payload(
                decision_count=100,
                high_present_count=10,
                high_complete_count=10,
                high_floor_count=10,
                high_selected_count=10,
            )
        )
        assessment = v198.assess_high_specificity_coverage(
            artifact_table={"high_specificity_key_count": 12},
            probe=probe,
        )
        route = v198.route_decision_for_v198(
            source_validation={"passed": True},
            probe=probe,
            coverage_assessment=assessment,
        )

        self.assertEqual(
            assessment["primary_blocker"],
            "high_specificity_live_overlap_below_slice_4_floor",
        )
        self.assertEqual(route["selected_route"], v198.PUBLIC_MODEL_CAPACITY_ROUTE)
        self.assertFalse(route["slice_4_training_consumed"])

    def test_high_specificity_present_but_action_incomplete_routes_to_contract_audit(
        self,
    ) -> None:
        probe = _real_probe(
            _combined_probe_payload(
                decision_count=100,
                high_present_count=20,
                high_complete_count=0,
                high_floor_count=0,
                high_selected_count=0,
                high_incomplete_count=20,
            )
        )
        assessment = v198.assess_high_specificity_coverage(
            artifact_table={"high_specificity_key_count": 12},
            probe=probe,
        )
        route = v198.route_decision_for_v198(
            source_validation={"passed": True},
            probe=probe,
            coverage_assessment=assessment,
        )

        self.assertEqual(
            assessment["primary_blocker"],
            "high_specificity_action_incomplete",
        )
        self.assertEqual(route["selected_route"], v198.ACTION_COMPLETE_ROUTE)
        self.assertFalse(route["slice_4_training_consumed"])

    def test_cli_writes_closed_report_when_probe_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                paths,
                v197_digest,
                v196_digest,
                v195_report_digest,
                artifact_digest,
            ) = _write_valid_v197_inputs(tmpdir)
            output_path = paths["v198_report"]
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair",
                    "--v197-report",
                    str(paths["v197_report"]),
                    "--v195-artifact",
                    str(paths["v195_artifact"]),
                    "--output",
                    str(output_path),
                    "--expected-v197-report-exact-digest",
                    v197_digest,
                    "--expected-v196-report-exact-digest",
                    v196_digest,
                    "--expected-v195-report-exact-digest",
                    v195_report_digest,
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
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("slice_4_training_consumed=False", result.stdout)
        self.assertIn("runtime_action_selection_changed=False", result.stdout)
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v198.INSTRUMENTATION_ROUTE,
        )
        self.assertTrue(exact_digest_validation_report(report)["passed"])


def _write_valid_v197_inputs(
    tmpdir: str,
) -> tuple[dict[str, Path], str, str, str, str]:
    paths, v195_report_digest, artifact_digest = _write_valid_inputs(tmpdir)
    v196_report = _write_valid_v196_report(
        paths=paths,
        expected_report_digest=v195_report_digest,
        expected_artifact_digest=artifact_digest,
    )
    v197_report_path = Path(tmpdir) / "v197-report.json"
    v198_report_path = Path(tmpdir) / "v198-report.json"
    report = v197.run_carrion_survivor_continuation_v197_coverage_abstention_repair_design(
        v196_report_path=paths["v196_report"],
        v195_report_path=paths["v195_report"],
        v195_artifact_path=paths["v195_artifact"],
        output_path=v197_report_path,
        expected_v196_report_exact_digest=str(v196_report["exact_digest"]),
        expected_v195_report_exact_digest=v195_report_digest,
        expected_v195_artifact_digest=artifact_digest,
        run_diagnostic_replay=False,
        specificity_gate_diagnostics_override=(
            _after_specificity_gate_diagnostics(
                low_specificity_count=180,
                miss_count=20,
                exact_high_count=0,
                override_applied_count=0,
            )
        ),
    )
    report["after_specificity_gate"].update(
        {
            "policy": (
                "m3_carrion_survivor_continuation_v197_"
                "specificity_gate_diagnostic_replay_v1"
            ),
            "diagnostics_only": True,
            "diagnostics_provenance": "real_replay",
            "diagnostics_override_used": False,
            "source_key_specificity_gate_enabled": True,
            "training_rerun": False,
            "slice_4_training_consumed": False,
        }
    )
    report["exact_digest"] = v197.digest_without_exact(report)
    v197_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths = dict(paths)
    paths["v197_report"] = v197_report_path
    paths["v198_report"] = v198_report_path
    return (
        paths,
        str(report["exact_digest"]),
        str(v196_report["exact_digest"]),
        v195_report_digest,
        artifact_digest,
    )


def _combined_probe_payload(
    *,
    decision_count: int,
    high_present_count: int,
    high_complete_count: int,
    high_floor_count: int,
    high_selected_count: int,
    high_incomplete_count: int = 0,
) -> dict[str, object]:
    stats = v198._empty_probe_stats()
    stats["decision_count"] = decision_count
    stats["candidate_key_coverage_decision_count"] = decision_count
    stats["high_specificity_any_key_present_count"] = high_present_count
    stats["high_specificity_any_complete_count"] = high_complete_count
    stats["high_specificity_any_observed_support_floor_satisfied_count"] = (
        high_floor_count
    )
    stats["high_specificity_selected_source_count"] = high_selected_count
    stats["candidate_key_category_stats"] = {
        "self_nav_feature_hit": {
            "candidate_key_evaluated_count": decision_count,
            "key_present_count": high_present_count,
            "key_absent_count": decision_count - high_present_count,
            "complete_for_current_valid_actions_count": high_complete_count,
            "present_but_action_incomplete_count": high_incomplete_count,
            "observed_support_floor_satisfied_count": high_floor_count,
            "selected_source_category_count": high_selected_count,
        }
    }
    return v198._finalize_probe_stats(stats)


def _real_probe(combined: dict[str, object]) -> dict[str, object]:
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


if __name__ == "__main__":
    unittest.main()
