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
from evolution_sim.mind import (
    carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit as v201,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV201SourceContractAuditTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v201-high-specificity-action-complete-source-contract-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit"
            ),
        )

    def test_source_pin_failure_stops_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, digests = _write_valid_v201_inputs(tmpdir)
            report = v201.run_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit(
                v198_report_path=paths["v198_report"],
                v199_report_path=paths["v199_report"],
                v200_report_path=paths["v200_report"],
                v195_artifact_path=paths["v195_artifact"],
                output_path=paths["v201_report"],
                expected_v198_report_exact_digest=digests["v198"],
                expected_v199_report_exact_digest=digests["v199"],
                expected_v200_report_exact_digest="bad-v200-digest",
                expected_v195_artifact_digest=digests["artifact"],
            )

        self.assertFalse(report["source_pin_validation"]["passed"])
        self.assertIn(
            "v200_report_exact_digest_matches_expected",
            report["source_pin_validation"]["failures"],
        )
        self.assertEqual(report["route_decision"]["selected_route"], v201.STOP_ROUTE)
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_lower_specificity_only_evidence_routes_to_public_capacity_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, digests = _write_valid_v201_inputs(tmpdir)
            report = v201.run_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit(
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

        audit = report["source_contract_audit"]
        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertEqual(
            audit["primary_blocker"],
            "lower_specificity_only_evidence_dominates_missing_actions",
        )
        self.assertEqual(
            audit["counts"]["lower_specificity_only_missing_action_count"], 2
        )
        self.assertEqual(
            audit["counts"][
                "same_high_specific_action_present_for_missing_demands_count"
            ],
            0,
        )
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v201.PUBLIC_MODEL_CAPACITY_ROUTE,
        )
        self.assertFalse(
            report["route_decision"][
                "future_explicit_slice_4_training_route_authorized"
            ]
        )

    def test_absent_same_high_specific_actions_route_to_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, digests = _write_valid_v201_inputs(
                tmpdir,
                include_lower_specificity_support=False,
            )
            report = v201.run_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit(
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

        audit = report["source_contract_audit"]
        self.assertEqual(
            audit["primary_blocker"],
            "same_high_specific_current_valid_action_rows_absent",
        )
        self.assertEqual(
            audit["counts"][
                "missing_action_absent_at_same_and_known_lower_keys_count"
            ],
            2,
        )
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v201.DATASET_CONTRACT_DESIGN_ROUTE,
        )

    def test_missing_artifact_table_routes_to_instrumentation_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, digests = _write_valid_v201_inputs(
                tmpdir,
                include_artifact_table=False,
            )
            report = v201.run_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit(
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

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertFalse(
            report["source_contract_audit"][
                "source_contract_auditable_from_existing_artifacts"
            ]
        )
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v201.INSTRUMENTATION_REPAIR_ROUTE,
        )

    def test_cli_writes_closed_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, digests = _write_valid_v201_inputs(tmpdir)
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit",
                    "--v198-report",
                    str(paths["v198_report"]),
                    "--v199-report",
                    str(paths["v199_report"]),
                    "--v200-report",
                    str(paths["v200_report"]),
                    "--v195-artifact",
                    str(paths["v195_artifact"]),
                    "--output",
                    str(paths["v201_report"]),
                    "--expected-v198-report-exact-digest",
                    digests["v198"],
                    "--expected-v199-report-exact-digest",
                    digests["v199"],
                    "--expected-v200-report-exact-digest",
                    digests["v200"],
                    "--expected-v195-artifact-digest",
                    digests["artifact"],
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(paths["v201_report"].read_text(encoding="utf-8"))

        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("slice_4_training_consumed=False", result.stdout)
        self.assertIn("support_expansion_ran=False", result.stdout)
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v201.PUBLIC_MODEL_CAPACITY_ROUTE,
        )
        self.assertTrue(exact_digest_validation_report(report)["passed"])


def _write_valid_v201_inputs(
    tmpdir: str,
    *,
    include_lower_specificity_support: bool = True,
    include_artifact_table: bool = True,
) -> tuple[dict[str, Path], dict[str, str]]:
    base = Path(tmpdir)
    paths = {
        "v198_report": base / "v198-report.json",
        "v199_report": base / "v199-report.json",
        "v200_report": base / "v200-report.json",
        "v201_report": base / "v201-report.json",
        "v195_artifact": base / "v195-artifact.json",
    }
    candidate_key = "mask=11001010000000000000|nav=water=a,plant=b,carrion=c,prey=d"
    lower_key = "mask=11001010000000000000"
    artifact_table: dict[str, object] = {}
    if include_artifact_table:
        artifact_table = {
            candidate_key: {
                "stay": {"count": 1, "utility_mean": 1.0},
            }
        }
        if include_lower_specificity_support:
            artifact_table[lower_key] = {
                "eat": {"count": 3, "utility_mean": 2.0},
            }
    artifact = {
        "schema_version": "mind_v3_v142_public_transition_value_scorer_v1",
        "artifact_policy": (
            "opt_in_m3_carrion_survivor_continuation_v195_public_repaired_contract_terminal_survival_support_policy_v1"
        ),
        "runtime_action_selection_authorized": False,
        "promotion_authorized": False,
        "utility_tables": {
            "policy": "test_feature_action_utility",
            "feature_action_utility": artifact_table,
        },
    }
    _write_json_with_digest(paths["v195_artifact"], artifact)
    artifact_digest = stable_payload_digest(artifact)

    v198_report = {
        "schema_version": (
            "m3_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair_report_v1"
        ),
        "policy": (
            "diagnostics_only_m3_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair_v1"
        ),
        "source_pin_validation": {
            "passed": True,
            "observed_v197_report_exact_digest": (
                v201.EXPECTED_V197_REPORT_EXACT_DIGEST
            ),
            "observed_v197_route": (
                "v198_high_specificity_coverage_or_model_capacity_repair_before_slice_4_training_no_training"
            ),
            "checks": {
                "v197_inherited_v196_digest_matches": True,
                "v197_inherited_v195_report_digest_matches": True,
                "v197_inherited_v195_artifact_digest_matches": True,
            },
        },
        **_closed_lifecycle(),
    }
    _write_json_with_digest(paths["v198_report"], v198_report)
    v198_digest = str(v198_report["exact_digest"])

    v199_report = {
        "schema_version": (
            "m3_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit_report_v1"
        ),
        "policy": (
            "diagnostics_only_m3_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit_v1"
        ),
        "classification": {"primary": v201.EXPECTED_V199_CLASSIFICATION},
        "source_pin_validation": {
            "passed": True,
            "expected_v198_report_exact_digest": v198_digest,
        },
        "action_complete_audit": {
            "ran": True,
            "diagnostics_provenance": "real_replay",
            "combined": {
                "high_specificity_candidate_evaluated_count": (
                    v201.EXPECTED_HIGH_SPECIFIC_CANDIDATES
                ),
                "key_absent_count": v201.EXPECTED_KEY_ABSENT,
                "key_present_count": v201.EXPECTED_KEY_PRESENT,
                "present_but_action_incomplete_count": (
                    v201.EXPECTED_PRESENT_INCOMPLETE
                ),
                "present_complete_but_observed_support_floor_failed_count": 0,
                "imputed_valid_action_score_count": 0,
                "candidate_key_action_coverage_count": 1,
                "runtime_action_selection_changed_count": 0,
                "missing_current_valid_action_counts": (
                    v201.EXPECTED_MISSING_ACTION_COUNTS
                ),
                "by_scope": {
                    "carrion_only": {
                        "missing_current_valid_action_counts": {"eat": 2}
                    }
                },
                "by_seed": {
                    "carrion_only:13": {
                        "missing_current_valid_action_counts": {"eat": 2}
                    }
                },
                "by_high_specificity_category": {
                    "nav_feature_hit": {
                        "missing_current_valid_action_counts": {"eat": 2}
                    }
                },
                "candidate_key_action_coverage": {
                    candidate_key: {
                        "candidate_key": candidate_key,
                        "candidate_key_category": "nav_feature_hit",
                        "candidate_key_digest": stable_payload_digest(candidate_key),
                        "key_present_count": 2,
                        "present_but_action_incomplete_count": 2,
                        "present_complete_but_observed_support_floor_failed_count": 0,
                        "missing_current_valid_action_counts": {"eat": 2},
                        "below_observed_support_floor_action_counts": {},
                        "scope_counts": {"carrion_only": 2},
                        "seed_counts": {"carrion_only:13": 2},
                        "valid_action_demand_counts": {"eat": 2, "stay": 2},
                    }
                },
            },
        },
        "route_decision": {
            "selected_route": v201.EXPECTED_V199_ROUTE,
            "recommended_next_route": v201.EXPECTED_V199_ROUTE,
        },
        **_closed_lifecycle(),
    }
    _write_json_with_digest(paths["v199_report"], v199_report)
    v199_digest = str(v199_report["exact_digest"])

    v200_report = v200.run_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design(
        v198_report_path=paths["v198_report"],
        v199_report_path=paths["v199_report"],
        v195_artifact_path=paths["v195_artifact"],
        output_path=paths["v200_report"],
        expected_v198_report_exact_digest=v198_digest,
        expected_v199_report_exact_digest=v199_digest,
        expected_v195_artifact_digest=artifact_digest,
    )
    return paths, {
        "v198": v198_digest,
        "v199": v199_digest,
        "v200": str(v200_report["exact_digest"]),
        "artifact": artifact_digest,
    }


def _write_json_with_digest(path: Path, report: dict[str, object]) -> None:
    report["exact_digest"] = stable_payload_digest(report)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _closed_lifecycle() -> dict[str, bool]:
    return {
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


if __name__ == "__main__":
    unittest.main()
