from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit as v201,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract as v202,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_mind_v3_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit import (
    _write_valid_v201_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV202HarnessContractTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v202-public-masked-model-capacity-harness-contract"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract"
            ),
        )

    def test_valid_lineage_writes_closed_contract_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, digests = _write_valid_v202_inputs(tmpdir)
            report = v202.run_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract(
                v198_report_path=paths["v198_report"],
                v199_report_path=paths["v199_report"],
                v200_report_path=paths["v200_report"],
                v201_report_path=paths["v201_report"],
                output_path=paths["v202_report"],
                expected_v198_report_exact_digest=digests["v198"],
                expected_v199_report_exact_digest=digests["v199"],
                expected_v200_report_exact_digest=digests["v200"],
                expected_v201_report_exact_digest=digests["v201"],
            )

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertTrue(report["v201_fact_assessment"]["passed"])
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v202.HARNESS_SCAFFOLD_ROUTE,
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract_ready_for_scaffold_no_training"
            ),
        )
        self.assertFalse(report["training_started"])
        self.assertFalse(report["training_slice_4_consumed"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["support_generated"])
        self.assertFalse(report["support_expanded"])
        self.assertFalse(report["dataset_mutated"])
        self.assertFalse(report["gate_relaxed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertTrue(report["non_promoted"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_bad_v201_digest_fails_closed_to_source_pin_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, digests = _write_valid_v202_inputs(tmpdir)
            report = v202.run_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract(
                v198_report_path=paths["v198_report"],
                v199_report_path=paths["v199_report"],
                v200_report_path=paths["v200_report"],
                v201_report_path=paths["v201_report"],
                output_path=paths["v202_report"],
                expected_v198_report_exact_digest=digests["v198"],
                expected_v199_report_exact_digest=digests["v199"],
                expected_v200_report_exact_digest=digests["v200"],
                expected_v201_report_exact_digest="bad-v201-digest",
            )

        self.assertFalse(report["source_pin_validation"]["passed"])
        self.assertIn(
            "v201_exact_digest_matches_expected",
            report["source_pin_validation"]["failures"],
        )
        self.assertEqual(report["route_decision"]["selected_route"], v202.STOP_ROUTE)
        self.assertFalse(report["training_started"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_v201_fact_mismatch_routes_to_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, digests = _write_valid_v202_inputs(tmpdir)
            v201_report = json.loads(paths["v201_report"].read_text(encoding="utf-8"))
            counts = v201_report["source_contract_audit"]["counts"]
            counts["lower_specificity_only_missing_action_count"] = 0
            _rewrite_json_with_exact(paths["v201_report"], v201_report)
            digests["v201"] = str(v201_report["exact_digest"])

            report = v202.run_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract(
                v198_report_path=paths["v198_report"],
                v199_report_path=paths["v199_report"],
                v200_report_path=paths["v200_report"],
                v201_report_path=paths["v201_report"],
                output_path=paths["v202_report"],
                expected_v198_report_exact_digest=digests["v198"],
                expected_v199_report_exact_digest=digests["v199"],
                expected_v200_report_exact_digest=digests["v200"],
                expected_v201_report_exact_digest=digests["v201"],
            )

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertFalse(report["v201_fact_assessment"]["passed"])
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v202.V201_RECONCILIATION_ROUTE,
        )

    def test_adversarial_interface_and_leakage_routes(self) -> None:
        source = {"passed": True}
        facts = {"passed": True}
        interface_gap = v202.route_decision_for_v202(
            source_validation=source,
            v201_facts=facts,
            harness_contract=v202.public_masked_model_capacity_harness_contract(
                interface_expressible=False
            ),
        )
        leakage = v202.route_decision_for_v202(
            source_validation=source,
            v201_facts=facts,
            harness_contract=v202.public_masked_model_capacity_harness_contract(
                requires_private_runtime_features=True
            ),
        )

        self.assertEqual(
            interface_gap["selected_route"], v202.INTERFACE_GAP_REPAIR_ROUTE
        )
        self.assertEqual(
            leakage["selected_route"], v202.LEAKAGE_CONTRACT_REPAIR_ROUTE
        )

    def test_cli_writes_closed_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, digests = _write_valid_v202_inputs(tmpdir)
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract",
                    "--v198-report",
                    str(paths["v198_report"]),
                    "--v199-report",
                    str(paths["v199_report"]),
                    "--v200-report",
                    str(paths["v200_report"]),
                    "--v201-report",
                    str(paths["v201_report"]),
                    "--output",
                    str(paths["v202_report"]),
                    "--expected-v198-report-exact-digest",
                    digests["v198"],
                    "--expected-v199-report-exact-digest",
                    digests["v199"],
                    "--expected-v200-report-exact-digest",
                    digests["v200"],
                    "--expected-v201-report-exact-digest",
                    digests["v201"],
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(paths["v202_report"].read_text(encoding="utf-8"))

        self.assertIn("training_started=False", result.stdout)
        self.assertIn("training_slice_4_consumed=False", result.stdout)
        self.assertIn("support_generated=False", result.stdout)
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v202.HARNESS_SCAFFOLD_ROUTE,
        )
        self.assertTrue(exact_digest_validation_report(report)["passed"])


def _write_valid_v202_inputs(tmpdir: str) -> tuple[dict[str, Path], dict[str, str]]:
    paths, digests = _write_valid_v201_inputs(tmpdir)
    paths["v202_report"] = Path(tmpdir) / "v202-report.json"
    v198_report = json.loads(paths["v198_report"].read_text(encoding="utf-8"))
    v198_report["classification"] = {"primary": v202.EXPECTED_V198_CLASSIFICATION}
    v198_report["route_decision"] = {
        "selected_route": v202.EXPECTED_V198_ROUTE,
        "recommended_next_route": v202.EXPECTED_V198_ROUTE,
    }
    _rewrite_json_with_exact(paths["v198_report"], v198_report)
    digests["v198"] = str(v198_report["exact_digest"])
    v200_report = json.loads(paths["v200_report"].read_text(encoding="utf-8"))
    v200_report["source_pin_validation"][
        "observed_v198_report_exact_digest"
    ] = digests["v198"]
    _rewrite_json_with_exact(paths["v200_report"], v200_report)
    digests["v200"] = str(v200_report["exact_digest"])
    v201_report = v201.run_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit(
        v198_report_path=paths["v198_report"],
        v199_report_path=paths["v199_report"],
        v200_report_path=paths["v200_report"],
        v195_artifact_path=paths["v195_artifact"],
        output_path=paths["v201_report"],
        expected_v198_report_exact_digest=digests["v198"],
        expected_v199_report_exact_digest=digests["v199"],
        expected_v200_report_exact_digest=digests["v200"],
        expected_v195_artifact_digest=digests["artifact"],
    )
    audit = v201_report["source_contract_audit"]
    audit["counts"].update(
        {
            "total_missing_current_valid_action_demands": (
                v202.EXPECTED_V201_MISSING_CURRENT_VALID_ACTION_DEMANDS
            ),
            "same_high_specific_action_present_for_missing_demands_count": (
                v202.EXPECTED_V201_SAME_HIGH_SPECIFIC_ACTION_SUPPORT
            ),
            "same_high_specific_below_observed_support_floor_count": (
                v202.EXPECTED_V201_SAME_HIGH_SPECIFIC_BELOW_FLOOR_SUPPORT
            ),
            "lower_specificity_only_missing_action_count": (
                v202.EXPECTED_V201_LOWER_SPECIFICITY_ONLY_DEMANDS
            ),
            "missing_action_absent_at_same_and_known_lower_keys_count": (
                v202.EXPECTED_V201_ABSENT_AT_SAME_AND_KNOWN_LOWER_KEYS
            ),
            "candidate_key_action_coverage_count": (
                v202.EXPECTED_V201_CANDIDATE_KEY_ACTION_COVERAGE_COUNT
            ),
            "artifact_feature_key_count": (
                v202.EXPECTED_V201_ARTIFACT_FEATURE_KEY_COUNT
            ),
        }
    )
    audit["gap_classification"]["lower_specificity_only_missing_action_share"] = (
        v202.EXPECTED_V201_LOWER_SPECIFICITY_ONLY_SHARE
    )
    audit.setdefault("unsafe_non_authorizing_patterns", {})[
        "lower_specificity_graft_or_imputation"
    ] = (
        "non_authorizing unless exact same high-specific public key and current-valid action support is proven"
    )
    _rewrite_json_with_exact(paths["v201_report"], v201_report)
    digests["v201"] = str(v201_report["exact_digest"])
    return paths, digests


def _rewrite_json_with_exact(path: Path, report: dict[str, object]) -> None:
    report.pop("exact_digest", None)
    report["exact_digest"] = stable_payload_digest(report)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
