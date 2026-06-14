from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response as v196,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v197_coverage_abstention_repair_design as v197,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from python.tests.test_mind_v3_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response import (
    _coverage_collapse_lookup_diagnostics,
    _write_valid_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV197CoverageAbstentionTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v197-coverage-abstention-repair-design"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v197_coverage_abstention_repair_design"
            ),
        )

    def test_specificity_gate_blocks_collapse_but_low_coverage_routes_to_capacity_repair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, artifact_digest = _write_valid_inputs(tmpdir)
            v196_report = _write_valid_v196_report(
                paths=paths,
                expected_report_digest=report_digest,
                expected_artifact_digest=artifact_digest,
            )
            report = v197.run_carrion_survivor_continuation_v197_coverage_abstention_repair_design(
                v196_report_path=paths["v196_report"],
                v195_report_path=paths["v195_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v195_report"].parent / "v197-report.json",
                expected_v196_report_exact_digest=str(v196_report["exact_digest"]),
                expected_v195_report_exact_digest=report_digest,
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

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertTrue(report["failure_fact_validation"]["passed"])
        comparison = report["specificity_gate_comparison"]
        self.assertTrue(comparison["low_specificity_collapse_blocked"])
        self.assertEqual(comparison["low_specificity_rejected_decision_count"], 180)
        self.assertEqual(comparison["exact_or_high_specificity_hit_share"], 0.0)
        self.assertFalse(
            comparison["exact_or_high_specificity_coverage_sufficient_for_slice_4"]
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v197.COVERAGE_MODEL_CAPACITY_ROUTE,
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["gate_relaxation_allowed"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_specificity_gate_override_cannot_route_to_future_slice_4(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, artifact_digest = _write_valid_inputs(tmpdir)
            v196_report = _write_valid_v196_report(
                paths=paths,
                expected_report_digest=report_digest,
                expected_artifact_digest=artifact_digest,
            )
            report = v197.run_carrion_survivor_continuation_v197_coverage_abstention_repair_design(
                v196_report_path=paths["v196_report"],
                v195_report_path=paths["v195_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v195_report"].parent / "v197-report.json",
                expected_v196_report_exact_digest=str(v196_report["exact_digest"]),
                expected_v195_report_exact_digest=report_digest,
                expected_v195_artifact_digest=artifact_digest,
                run_diagnostic_replay=False,
                specificity_gate_diagnostics_override=(
                    _after_specificity_gate_diagnostics(
                        low_specificity_count=130,
                        miss_count=20,
                        exact_high_count=50,
                        override_applied_count=0,
                    )
                ),
            )

        comparison = report["specificity_gate_comparison"]
        self.assertTrue(comparison["low_specificity_collapse_blocked"])
        self.assertTrue(
            comparison["exact_or_high_specificity_coverage_sufficient_for_slice_4"]
        )
        self.assertFalse(comparison["real_replay_provenance_for_slice_4"])
        self.assertFalse(comparison["slice_4_training_justified"])
        self.assertEqual(
            comparison["slice_4_training_justification_blocked_reason"],
            "specificity_gate_diagnostics_not_real_replay",
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v197.COVERAGE_MODEL_CAPACITY_ROUTE,
        )
        self.assertFalse(
            report["route_decision"]["future_explicit_slice_4_training_route_authorized"]
        )
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertFalse(report["runtime_action_selection_changed"])

    def test_real_replay_provenance_can_route_to_future_explicit_slice_4(self) -> None:
        after = _after_specificity_gate_diagnostics(
            low_specificity_count=130,
            miss_count=20,
            exact_high_count=50,
            override_applied_count=0,
        )
        after.update(
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
        comparison = v197.compare_specificity_gate_diagnostics(
            before={
                "combined": {
                    "override_applied_share": 0.782944,
                    "dominant_applied_override_action": "eat",
                    "dominant_applied_override_action_share": 0.939312,
                }
            },
            after=after,
        )
        route = v197.route_decision_for_v197(
            source_validation={"passed": True},
            failure_fact_validation={"passed": True},
            comparison=comparison,
            after_specificity_gate=after,
        )

        self.assertTrue(comparison["real_replay_provenance_for_slice_4"])
        self.assertTrue(comparison["slice_4_training_justified"])
        self.assertEqual(route["recommended_next_route"], v197.FUTURE_SLICE_4_ROUTE)
        self.assertTrue(route["future_explicit_slice_4_training_route_authorized"])
        self.assertFalse(route["slice_4_training_consumed"])

    def test_v196_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, artifact_digest = _write_valid_inputs(tmpdir)
            _write_valid_v196_report(
                paths=paths,
                expected_report_digest=report_digest,
                expected_artifact_digest=artifact_digest,
            )
            report = v197.run_carrion_survivor_continuation_v197_coverage_abstention_repair_design(
                v196_report_path=paths["v196_report"],
                v195_report_path=paths["v195_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v195_report"].parent / "v197-report.json",
                expected_v196_report_exact_digest="wrong",
                expected_v195_report_exact_digest=report_digest,
                expected_v195_artifact_digest=artifact_digest,
                run_diagnostic_replay=False,
            )

        self.assertFalse(report["source_pin_validation"]["passed"])
        self.assertIn(
            "v196_report_exact_digest_matches_expected",
            report["source_pin_validation"]["failures"],
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v197.STOP_ROUTE,
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertFalse(report["runtime_action_selection_changed"])

    def test_cli_writes_report_without_training_when_replay_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, artifact_digest = _write_valid_inputs(tmpdir)
            v196_report = _write_valid_v196_report(
                paths=paths,
                expected_report_digest=report_digest,
                expected_artifact_digest=artifact_digest,
            )
            output_path = paths["v195_report"].parent / "v197-report.json"
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v197_coverage_abstention_repair_design",
                    "--v196-report",
                    str(paths["v196_report"]),
                    "--v195-report",
                    str(paths["v195_report"]),
                    "--v195-artifact",
                    str(paths["v195_artifact"]),
                    "--output",
                    str(output_path),
                    "--expected-v196-report-exact-digest",
                    str(v196_report["exact_digest"]),
                    "--expected-v195-report-exact-digest",
                    report_digest,
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
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v197.INSTRUMENTATION_ROUTE,
        )
        self.assertTrue(exact_digest_validation_report(report)["passed"])


def _write_valid_v196_report(
    *,
    paths: dict[str, Path],
    expected_report_digest: str,
    expected_artifact_digest: str,
) -> dict[str, object]:
    return v196.run_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response(
        v195_report_path=paths["v195_report"],
        v195_artifact_path=paths["v195_artifact"],
        v186_report_path=paths["v186_report"],
        output_path=paths["v196_report"],
        expected_v195_report_exact_digest=expected_report_digest,
        expected_v195_artifact_digest=expected_artifact_digest,
        backup_metadata_override={"passed": True},
        run_diagnostic_replay=False,
        lookup_diagnostics_override=_coverage_collapse_lookup_diagnostics(),
    )


def _after_specificity_gate_diagnostics(
    *,
    low_specificity_count: int,
    miss_count: int,
    exact_high_count: int,
    override_applied_count: int,
) -> dict[str, object]:
    broad = _lookup_payload(
        decision_count=100,
        low_specificity_count=low_specificity_count // 2,
        miss_count=miss_count // 2,
        exact_high_count=exact_high_count // 2,
        override_applied_count=override_applied_count // 2,
    )
    carrion = _lookup_payload(
        decision_count=100,
        low_specificity_count=low_specificity_count - (low_specificity_count // 2),
        miss_count=miss_count - (miss_count // 2),
        exact_high_count=exact_high_count - (exact_high_count // 2),
        override_applied_count=override_applied_count - (override_applied_count // 2),
    )
    combined = v197.combine_lookup_scopes(broad, carrion)
    return {
        "policy": "test_v197_specificity_gate_diagnostics",
        "ran": True,
        "broad": broad,
        "carrion_only": carrion,
        "combined": combined,
    }


def _lookup_payload(
    *,
    decision_count: int,
    low_specificity_count: int,
    miss_count: int,
    exact_high_count: int,
    override_applied_count: int,
) -> dict[str, object]:
    stats = v197._empty_lookup_stats()
    stats["decision_count"] = decision_count
    stats["supported_score_count"] = decision_count - miss_count
    stats["observed_support_floor_satisfied_count"] = decision_count - miss_count
    stats["clear_best_count"] = decision_count - miss_count
    stats["override_applied_count"] = override_applied_count
    stats["runtime_action_selection_changed_count"] = override_applied_count
    stats["missing_supported_score_count"] = miss_count
    stats["low_specificity_rejected_decision_count"] = low_specificity_count
    stats["source_key_specificity_gate_enabled_count"] = decision_count
    stats["source_key_specificity_gate_passed_count"] = exact_high_count
    stats["source_key_specificity_gate_failed_count"] = (
        decision_count - exact_high_count
    )
    stats["heuristic_action_source_count"] = 0
    stats["source_key_category_counts"] = {
        "mask_only_hit": low_specificity_count,
        "miss": miss_count,
        "self_nav_feature_hit": exact_high_count,
    }
    stats["override_rejected_reason_counts"] = {
        v197.LOW_SPECIFICITY_REJECTION_REASON: low_specificity_count,
        "missing_supported_scores_for_valid_actions": miss_count,
    }
    stats["applied_override_action_counts"] = (
        {"drink": override_applied_count} if override_applied_count else {}
    )
    stats["predicted_action_counts"] = {
        "drink": exact_high_count + override_applied_count
    }
    if override_applied_count:
        stats["final_requested_action_counts"] = {
            "stay": decision_count - override_applied_count,
            "drink": override_applied_count,
        }
    else:
        stats["final_requested_action_counts"] = {
            "stay": decision_count // 2,
            "drink": decision_count - (decision_count // 2),
        }
    stats["requested_action_counts"] = dict(stats["final_requested_action_counts"])
    return v197._finalize_lookup_stats(stats)


if __name__ == "__main__":
    unittest.main()
