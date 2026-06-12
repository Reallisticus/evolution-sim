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
    carrion_survivor_continuation_v184_v183_transition_row_dataset_audit as v184,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_mind_v3_carrion_survivor_continuation_v178_transition_row_dataset_audit import (
    _report,
    _report_exact_digest,
    _support_ready_rows,
    _write_json,
    _write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV184V183TransitionRowDatasetAuditTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v184-v183-transition-row-dataset-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit"
            ),
        )

    def test_v178_accepts_pinned_v183_source_and_uses_slice_2_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows = _write_v183_inputs(tmpdir, _support_ready_rows())
            dataset_digest = stable_payload_digest(rows)
            report_digest = _report_exact_digest(paths["v183_report"])

            with _patched_v183_digests(report_digest, dataset_digest):
                report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                    transition_dataset_path=paths["dataset"],
                    v177_report_path=paths["v183_report"],
                    output_path=paths["v178_report"],
                    expected_v177_report_exact_digest=report_digest,
                    expected_dataset_digest=dataset_digest,
                )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertEqual(
            report["source_validation"]["source_producer"],
            v178.V183_SOURCE_PRODUCER,
        )
        self.assertTrue(
            report["source_validation"]["v183_expansion_validation"]["passed"]
        )
        self.assertTrue(report["training_authorization"]["authorized"])
        self.assertEqual(
            report["route_recommendation"]["recommended_next_route"],
            v178.V185_SLICE_2_TRAINING_ROUTE,
        )
        self.assertTrue(
            report["route_recommendation"]["slice_2_opt_in_training_route_authorized"]
        )
        self.assertFalse(
            report["route_recommendation"]["first_opt_in_training_slice_authorized"]
        )
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])

    def test_v178_fails_closed_when_v183_source_is_not_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows = _write_v183_inputs(tmpdir, _support_ready_rows())

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v183_report"],
                output_path=paths["v178_report"],
                expected_v177_report_exact_digest=_report_exact_digest(
                    paths["v183_report"]
                ),
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v183_canonical_exact_digest_mismatch",
            report["source_validation"]["failures"],
        )
        self.assertIn(
            "v183_canonical_dataset_digest_mismatch",
            report["source_validation"]["failures"],
        )
        self.assertFalse(report["training_authorization"]["authorized"])
        self.assertFalse(
            report["route_recommendation"]["transition_row_training_authorized"]
        )
        self.assertFalse(report["training_ran"])

    def test_v184_authorizes_only_future_slice_2_route_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows = _write_v183_inputs(tmpdir, _support_ready_rows())
            dataset_digest = stable_payload_digest(rows)
            report_digest = _report_exact_digest(paths["v183_report"])

            with _patched_v183_digests(report_digest, dataset_digest):
                report = v184.run_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit(
                    transition_dataset_path=paths["dataset"],
                    v183_report_path=paths["v183_report"],
                    output_path=paths["v184_report"],
                )

        self.assertEqual(
            report["schema_version"],
            v184.M3_CARRION_SURVIVOR_CONTINUATION_V184_V183_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["policy"],
            v184.M3_CARRION_SURVIVOR_CONTINUATION_V184_V183_TRANSITION_ROW_DATASET_AUDIT_POLICY,
        )
        self.assertEqual(
            report["classification"]["primary"],
            v184.V184_AUTHORIZED_CLASSIFICATION,
        )
        self.assertTrue(report["source_validation"]["passed"])
        self.assertEqual(report["dataset"]["dataset_digest"], dataset_digest)
        self.assertTrue(report["training_authorization"]["authorized"])
        self.assertEqual(
            report["route_recommendation"]["recommended_next_route"],
            v178.V185_SLICE_2_TRAINING_ROUTE,
        )
        self.assertTrue(
            report["route_recommendation"]["slice_2_opt_in_training_route_authorized"]
        )
        self.assertFalse(
            report["route_recommendation"]["first_opt_in_training_slice_authorized"]
        )
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["slice_2_training_consumed"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_v184_cli_writes_fail_closed_report_for_noncanonical_v183_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows = _write_v183_inputs(tmpdir, _support_ready_rows())
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit",
                    "--transition-dataset",
                    str(paths["dataset"]),
                    "--v183-report",
                    str(paths["v183_report"]),
                    "--output",
                    str(paths["v184_report"]),
                    "--expected-v183-report-exact-digest",
                    _report_exact_digest(paths["v183_report"]),
                    "--expected-dataset-digest",
                    stable_payload_digest(rows),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v184_report"].read_text(encoding="utf-8"))

        self.assertIn("source_producer=v183_exact_transition_support_expansion", completed.stdout)
        self.assertIn("source_validation_passed=False", completed.stdout)
        self.assertEqual(
            written["classification"]["primary"],
            v184.V184_SOURCE_INVALID_CLASSIFICATION,
        )
        self.assertFalse(written["training_ran"])
        self.assertFalse(written["slice_2_training_consumed"])
        self.assertFalse(written["runtime_artifact_created"])


def _write_v183_inputs(
    tmpdir: str,
    rows: list[dict[str, object]],
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    root = Path(tmpdir)
    paths = {
        "v183_report": root / "v183.json",
        "dataset": root / "v183-transition-rows.jsonl",
        "v178_report": root / "v178.json",
        "v184_report": root / "v184.json",
    }
    _write_jsonl(paths["dataset"], rows)
    dataset_digest = stable_payload_digest(rows)
    report = _report(
        {
            "schema_version": v178.EXPECTED_V183_SCHEMA_VERSION,
            "policy": v178.EXPECTED_V183_POLICY,
            "classification": {
                "primary": v178.EXPECTED_V183_CLASSIFICATION,
                "labels": [v178.EXPECTED_V183_CLASSIFICATION],
            },
            "contract": {
                "failure_response_to": "v182_imputed_abstention_design",
                "diagnostics_only": True,
                "training_allowed": False,
                "training_ran": False,
                "training_artifact_created": False,
                "slice_2_training_consumed": False,
                "runtime_artifact_allowed": False,
                "runtime_artifact_created": False,
                "runtime_integration_allowed": False,
                "runtime_action_selection_changed": False,
                "default_runtime_behavior_changed": False,
                "promotion_authorized": False,
                "gate_relaxation_allowed": False,
                "observed_support_floor": 2,
                "targets_v182_carrion_observed_support_zero": True,
                "targets_v182_broad_seed_19_regression_states": True,
                "uses_v182_observed_imputed_support_fields": True,
                "legacy_supported_prediction_is_not_strict_observed_support": True,
                "fresh_v178_style_audit_required_before_slice_2_training": True,
            },
            "source_validation": _v183_upstream_source_validation(),
            "branch_materialization": {
                "policy": "m3_carrion_survivor_continuation_v183_exact_branch_materialization_v1",
                "passed": True,
                "selected_branch_point_count": v178.DEFAULT_MIN_BRANCH_COUNT,
                "materialized_branch_point_count": v178.DEFAULT_MIN_BRANCH_COUNT,
                "materialization_failure_count": 0,
                "materialization_failures": [],
                "exact_materialization_proven": True,
            },
            "metrics": {
                "policy": "m3_carrion_survivor_continuation_v177_transition_dataset_metrics_v1",
                "transition_row_count": len(rows),
                "selected_branch_point_count": v178.DEFAULT_MIN_BRANCH_COUNT,
                "materialized_branch_point_count": v178.DEFAULT_MIN_BRANCH_COUNT,
                "replay_verification_enabled": True,
                "all_replays_verified": True,
                "replay_verified_row_count": len(rows),
                "all_forced_actions_used": True,
                "forced_action_used_count": len(rows),
                "compact_transition_support_ready": True,
            },
            "support_summary": {
                "policy": "m3_carrion_survivor_continuation_v179_support_summary_v1",
                "passed": True,
                "failure_count": 0,
                "failures": [],
                "minimums": {
                    "row_count": v178.DEFAULT_MIN_ROW_COUNT,
                    "seed_count": v178.DEFAULT_MIN_SEED_COUNT,
                    "branch_count": v178.DEFAULT_MIN_BRANCH_COUNT,
                    "forced_action_count": v178.DEFAULT_MIN_FORCED_ACTION_COUNT,
                },
                "observed": {
                    "row_count": len(rows),
                    "seed_count": v178.DEFAULT_MIN_SEED_COUNT,
                    "branch_count": v178.DEFAULT_MIN_BRANCH_COUNT,
                    "forced_action_count": v178.DEFAULT_MIN_FORCED_ACTION_COUNT,
                },
                "v178_default_support_thresholds_met": True,
            },
            "route_recommendation": {
                "policy": "m3_carrion_survivor_continuation_v183_route_recommendation_v1",
                "recommended_next_route": (
                    "fresh_v178_style_transition_row_dataset_audit_before_any_slice_2_training"
                ),
                "v178_style_audit_recommended": True,
                "slice_2_training_authorized": False,
                "transition_row_training_authorized": False,
                "runtime_integration_authorized": False,
                "promotion_authorized": False,
            },
            "dataset": {
                "path": str(paths["dataset"]),
                "row_count": len(rows),
                "dataset_digest": dataset_digest,
                "row_schema_version": (
                    "m3_carrion_survivor_continuation_v177_compact_transition_diagnostic_row_v1"
                ),
                "feature_policy_id": "current_forced_next_public_transition_context_v1",
                "source_dataset_digest": "source-dataset-digest",
                "source_dataset_path": "output/mind/v179.jsonl",
            },
            "diagnostics_only": True,
            "training_ran": False,
            "training_artifact_created": False,
            "fit_ran": False,
            "scorer_retraining_ran": False,
            "scorer_retraining_authorized": False,
            "diagnostic_dataset_created": True,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "runtime_observation_schema_changed": False,
            "runtime_policy_changed": False,
            "shadow_eval_ran": False,
            "live_ab_ran": False,
            "live_ab_allowed": False,
            "promotion_authorized": False,
            "gate_relaxation_ran": False,
            "replay_viewer_schema_changed": False,
            "slice_2_training_consumed": False,
            "non_promoted": True,
        }
    )
    _write_json(paths["v183_report"], report)
    return paths, rows


def _v183_upstream_source_validation() -> dict[str, object]:
    digest = "1" * 64
    artifact_digest = "2" * 64
    dataset_digest = "3" * 64
    return {
        "policy": "m3_carrion_survivor_continuation_v183_source_validation_v1",
        "passed": True,
        "failure_count": 0,
        "failures": [],
        "v182_schema_version_matches": True,
        "v182_policy_matches": True,
        "v182_exact_digest_valid": True,
        "v182_exact_digest_matches_expected": True,
        "v182_classification_matches_expected": True,
        "v182_source_validation_passed": True,
        "v182_routes_to_exact_support_expansion": True,
        "v182_training_not_run": True,
        "v182_slice_2_training_not_consumed": True,
        "v182_runtime_action_selection_unchanged": True,
        "v181_exact_digest_valid": True,
        "v181_exact_digest_matches_expected": True,
        "v181_classification_matches_expected": True,
        "v181_training_not_run": True,
        "v180_exact_digest_valid": True,
        "v180_exact_digest_matches_expected": True,
        "v180_classification_matches_expected": True,
        "v180_training_slice_1_ran": True,
        "v180_runtime_action_selection_unchanged": True,
        "v180_promotion_not_authorized": True,
        "v180_artifact_digest_matches_expected": True,
        "v180_artifact_digest_matches_report": True,
        "v179_exact_digest_valid": True,
        "v179_exact_digest_matches_expected": True,
        "v179_classification_matches_expected": True,
        "v179_source_validation_passed": True,
        "v179_support_summary_passed": True,
        "v179_dataset_digest_matches_expected": True,
        "v179_dataset_digest_matches_report": True,
        "v179_dataset_digest_matches_v180_report": True,
        "expected_v182_report_exact_digest": digest,
        "observed_v182_report_exact_digest": digest,
        "expected_v181_report_exact_digest": digest,
        "observed_v181_report_exact_digest": digest,
        "expected_v180_report_exact_digest": digest,
        "observed_v180_report_exact_digest": digest,
        "expected_v180_artifact_digest": artifact_digest,
        "observed_v180_artifact_digest": artifact_digest,
        "expected_v179_report_exact_digest": digest,
        "observed_v179_report_exact_digest": digest,
        "expected_v179_dataset_digest": dataset_digest,
        "observed_v179_dataset_digest": dataset_digest,
    }


class _patched_v183_digests:
    def __init__(self, report_digest: str, dataset_digest: str) -> None:
        self.report_digest = report_digest
        self.dataset_digest = dataset_digest
        self.old_report_digest = v178.EXPECTED_V183_REPORT_EXACT_DIGEST
        self.old_dataset_digest = v178.EXPECTED_V183_DATASET_DIGEST

    def __enter__(self) -> None:
        v178.EXPECTED_V183_REPORT_EXACT_DIGEST = self.report_digest
        v178.EXPECTED_V183_DATASET_DIGEST = self.dataset_digest

    def __exit__(self, *args: object) -> None:
        v178.EXPECTED_V183_REPORT_EXACT_DIGEST = self.old_report_digest
        v178.EXPECTED_V183_DATASET_DIGEST = self.old_dataset_digest


if __name__ == "__main__":
    unittest.main()
