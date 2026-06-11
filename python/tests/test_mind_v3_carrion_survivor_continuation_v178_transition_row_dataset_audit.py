from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zlib

from evolution_sim.env.runtime import observations
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind import (
    carrion_survivor_continuation_v178_transition_row_dataset_audit as v178,
)
from evolution_sim.mind.carrion_survivor_continuation_v177_exact_branch_replay_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV178TransitionRowDatasetAuditTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v178-transition-row-dataset-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v178_transition_row_dataset_audit"
            ),
        )

    def test_support_ready_dataset_authorizes_first_opt_in_training_slice(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            paths, rows = _write_inputs(tmpdir, rows)

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_v177_report_exact_digest=_report_exact_digest(
                    paths["v177_report"]
                ),
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["row_schema_validation"]["passed"])
        self.assertTrue(report["leakage_scan"]["passed"])
        self.assertTrue(report["value_leakage_scan"]["passed"])
        self.assertTrue(report["feature_contract_audit"]["passed"])
        self.assertTrue(report["observation_audit"]["passed"])
        self.assertTrue(report["support_readiness"]["passed"])
        self.assertTrue(report["default_support_readiness"]["passed"])
        self.assertTrue(report["training_authorization"]["authorized"])
        self.assertTrue(
            report["contract"]["next_same_lane_opt_in_training_slice_authorized"]
        )
        self.assertGreaterEqual(
            report["support_readiness"]["observed"]["row_count"],
            v178.DEFAULT_MIN_ROW_COUNT,
        )
        self.assertGreaterEqual(
            report["support_readiness"]["observed"]["seed_count"],
            v178.DEFAULT_MIN_SEED_COUNT,
        )
        self.assertGreaterEqual(
            report["support_readiness"]["observed"]["branch_count"],
            v178.DEFAULT_MIN_BRANCH_COUNT,
        )
        self.assertGreaterEqual(
            report["support_readiness"]["observed"]["forced_action_count"],
            v178.DEFAULT_MIN_FORCED_ACTION_COUNT,
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "valid_support_ready_transition_row_training_authorized"
            ),
        )
        self.assertNotIn("no_training", report["classification"]["primary"])
        self.assertEqual(
            report["route_recommendation"]["recommended_next_route"],
            "v179_transition_row_policy_training_slice_opt_in",
        )
        self.assertNotIn(
            "model_design_audit",
            report["route_recommendation"]["recommended_next_route"],
        )
        self.assertFalse(report["training_authorized"])
        self.assertTrue(
            report["training_authorization"][
                "next_same_lane_opt_in_training_slice_authorized"
            ]
        )
        self.assertTrue(
            report["route_recommendation"]["transition_row_training_authorized"]
        )
        self.assertTrue(
            report["route_recommendation"][
                "first_opt_in_training_slice_authorized"
            ]
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertEqual(report["exact_digest"], _digest_without_exact(report))

    def test_lowered_support_thresholds_do_not_authorize_tiny_dataset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows = _write_inputs(
                tmpdir,
                [
                    _transition_row(
                        branch_id="branch-a",
                        seed=19,
                        forced_action="stay",
                        value_offset=0.01,
                    )
                ],
            )

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_v177_report_exact_digest=_report_exact_digest(
                    paths["v177_report"]
                ),
                expected_dataset_digest=stable_payload_digest(rows),
                min_row_count=1,
                min_seed_count=1,
                min_branch_count=1,
                min_forced_action_count=1,
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["row_schema_validation"]["passed"])
        self.assertTrue(report["support_readiness"]["passed"])
        self.assertTrue(report["support_readiness"]["diagnostic_support_minimums_met"])
        self.assertNotIn(
            "training_scale_capacity_work_unblocked",
            report["support_readiness"],
        )
        self.assertFalse(report["default_support_readiness"]["passed"])
        self.assertFalse(report["training_authorization"]["authorized"])
        self.assertFalse(
            report["training_authorization"]["default_support_readiness_passed"]
        )
        self.assertFalse(
            report["training_authorization"]["support_thresholds_match_defaults"]
        )
        self.assertIn(
            "default_support_readiness_passed",
            report["training_authorization"]["failures"],
        )
        self.assertIn(
            "support_thresholds_match_defaults",
            report["training_authorization"]["failures"],
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "valid_support_limited_expand_before_training"
            ),
        )
        self.assertEqual(
            report["route_recommendation"]["recommended_next_route"],
            "v179_expand_exact_branch_transition_rows_no_training",
        )
        self._assert_training_closed(report)

    def test_support_ready_dataset_requires_source_report_digest_pin(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            paths, rows = _write_inputs(tmpdir, rows)

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertTrue(report["default_support_readiness"]["passed"])
        self.assertFalse(report["training_authorization"]["authorized"])
        self.assertIn(
            "expected_source_report_exact_digest_provided",
            report["training_authorization"]["failures"],
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "valid_support_ready_digest_pins_required_no_training"
            ),
        )
        self.assertEqual(
            report["route_recommendation"]["recommended_next_route"],
            (
                "rerun_v178_with_expected_source_and_dataset_digests_before_"
                "training_authorization"
            ),
        )
        self._assert_training_closed(report)

    def test_nondefault_thresholds_do_not_authorize_default_ready_dataset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            paths, rows = _write_inputs(tmpdir, rows)

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_v177_report_exact_digest=_report_exact_digest(
                    paths["v177_report"]
                ),
                expected_dataset_digest=stable_payload_digest(rows),
                min_row_count=1,
                min_seed_count=1,
                min_branch_count=1,
                min_forced_action_count=1,
            )

        self.assertTrue(report["support_readiness"]["passed"])
        self.assertTrue(report["default_support_readiness"]["passed"])
        self.assertFalse(
            report["training_authorization"]["support_thresholds_match_defaults"]
        )
        self.assertEqual(
            report["training_authorization"]["failures"],
            ["support_thresholds_match_defaults"],
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "valid_support_ready_default_threshold_recheck_required_no_training"
            ),
        )
        self.assertEqual(
            report["route_recommendation"]["recommended_next_route"],
            "rerun_v178_with_default_support_thresholds_before_training_authorization",
        )
        self._assert_training_closed(report)

    def test_default_support_thresholds_keep_tiny_dataset_limited(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows = _write_inputs(
                tmpdir,
                [
                    _transition_row(
                        branch_id="branch-a",
                        seed=19,
                        forced_action="stay",
                        value_offset=0.01,
                    )
                ],
            )

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["row_schema_validation"]["passed"])
        self.assertFalse(report["support_readiness"]["passed"])
        self.assertFalse(report["default_support_readiness"]["passed"])
        self.assertFalse(report["training_authorization"]["authorized"])
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "valid_support_limited_expand_before_training"
            ),
        )
        self.assertEqual(
            report["route_recommendation"]["recommended_next_route"],
            "v179_expand_exact_branch_transition_rows_no_training",
        )
        self._assert_training_closed(report)

    def test_v177_missing_lifecycle_field_fails_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            paths, rows = _write_inputs(tmpdir, rows)
            v177_report = json.loads(paths["v177_report"].read_text(encoding="utf-8"))
            v177_report.pop("training_ran")
            _write_json(paths["v177_report"], _report(v177_report))

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v177_lifecycle_not_diagnostics_only",
            report["source_validation"]["failures"],
        )
        self.assertIn(
            {
                "field": "training_ran",
                "expected": False,
                "observed": None,
            },
            report["source_validation"]["v177_lifecycle_validation"]["failures"],
        )
        self._assert_training_closed(report)

    def test_v177_training_allowed_contract_fails_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            paths, rows = _write_inputs(tmpdir, rows)
            v177_report = json.loads(paths["v177_report"].read_text(encoding="utf-8"))
            v177_report["contract"]["training_allowed"] = True
            _write_json(paths["v177_report"], _report(v177_report))

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v177_lifecycle_not_diagnostics_only",
            report["source_validation"]["failures"],
        )
        self.assertIn(
            {
                "field": "contract.training_allowed",
                "expected": False,
                "observed": True,
            },
            report["source_validation"]["v177_lifecycle_validation"]["failures"],
        )
        self._assert_training_closed(report)

    def test_missing_target_summary_fails_contract_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            for row in rows:
                row.pop("short_horizon_public_outcome_summary")
            paths, rows = _write_inputs(tmpdir, rows)

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(report["target_audit"]["passed"])
        self.assertIn(
            "short_horizon_public_outcome_summary_missing",
            {failure["reason"] for failure in report["target_audit"]["failures"]},
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "dataset_contract_invalid_closed_no_training"
            ),
        )
        self._assert_training_closed(report)

    def test_invalid_forced_action_resolution_fails_contract_without_training(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            summary = rows[0]["short_horizon_public_outcome_summary"]
            self.assertIsInstance(summary, dict)
            summary["current_resolved_action"] = "eat"
            summary["current_action_valid"] = False
            summary["current_resolution_action_valid"] = False
            paths, rows = _write_inputs(tmpdir, rows)

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(report["target_audit"]["passed"])
        reasons = {failure["reason"] for failure in report["target_audit"]["failures"]}
        self.assertIn("current_resolved_action_mismatch", reasons)
        self.assertIn("current_action_valid_not_true", reasons)
        self.assertIn("current_resolution_action_valid_not_true", reasons)
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "dataset_contract_invalid_closed_no_training"
            ),
        )
        self._assert_training_closed(report)

    def test_exact_digest_is_stable_across_repeated_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            paths, rows = _write_inputs(tmpdir, rows)
            second_report_path = paths["report"].with_name("v178-second.json")

            first = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_v177_report_exact_digest=_report_exact_digest(
                    paths["v177_report"]
                ),
                expected_dataset_digest=stable_payload_digest(rows),
            )
            second = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=second_report_path,
                expected_v177_report_exact_digest=_report_exact_digest(
                    paths["v177_report"]
                ),
                expected_dataset_digest=stable_payload_digest(rows),
            )
            first_written = json.loads(paths["report"].read_text(encoding="utf-8"))
            second_written = json.loads(
                second_report_path.read_text(encoding="utf-8")
            )

        self.assertEqual(first["exact_digest"], second["exact_digest"])
        self.assertEqual(first, first_written)
        self.assertEqual(second, second_written)
        self.assertEqual(first_written, second_written)
        self.assertEqual(first["exact_digest"], _digest_without_exact(first))

    def test_duplicate_branch_action_fails_closed_without_training(self) -> None:
        row = _transition_row(
            branch_id="branch-a",
            seed=19,
            forced_action="stay",
            value_offset=0.01,
        )
        duplicate = _transition_row(
            branch_id="branch-a",
            seed=19,
            forced_action="stay",
            value_offset=0.02,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _rows = _write_inputs(tmpdir, [row, duplicate])

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                min_row_count=1,
                min_seed_count=1,
                min_branch_count=1,
                min_forced_action_count=1,
            )

        self.assertFalse(report["identity_audit"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "dataset_contract_invalid_closed_no_training"
            ),
        )
        self.assertFalse(report["training_ran"])
        self._assert_training_closed(report)

    def test_cloned_source_identity_across_branch_ids_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            first_metadata = rows[0]["metadata"]
            self.assertIsInstance(first_metadata, dict)
            for row in rows[1:]:
                metadata = row["metadata"]
                self.assertIsInstance(metadata, dict)
                for field in (
                    "seed",
                    "source_path",
                    "line_number",
                    "branch_tick",
                    "agent_id",
                    "source_record_digest",
                    "materialized_record_digest",
                ):
                    metadata[field] = first_metadata[field]
            paths, rows = _write_inputs(tmpdir, rows)

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_v177_report_exact_digest=_report_exact_digest(
                    paths["v177_report"]
                ),
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(report["identity_audit"]["passed"])
        reasons = {failure["reason"] for failure in report["identity_audit"]["failures"]}
        self.assertIn("source_materialization_identity_reused_across_branches", reasons)
        self.assertIn("source_record_digest_reused_across_branches", reasons)
        self.assertIn("materialized_record_digest_reused_across_branches", reasons)
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "dataset_contract_invalid_closed_no_training"
            ),
        )
        self._assert_training_closed(report)

    def test_branch_id_mixed_source_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            first_metadata = rows[0]["metadata"]
            mixed_metadata = rows[1]["metadata"]
            self.assertIsInstance(first_metadata, dict)
            self.assertIsInstance(mixed_metadata, dict)
            self.assertNotEqual(rows[0]["forced_action"], rows[1]["forced_action"])
            mixed_metadata["branch_id"] = first_metadata["branch_id"]
            paths, rows = _write_inputs(tmpdir, rows)

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_v177_report_exact_digest=_report_exact_digest(
                    paths["v177_report"]
                ),
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(report["identity_audit"]["passed"])
        reasons = {failure["reason"] for failure in report["identity_audit"]["failures"]}
        self.assertIn("branch_id_mixes_source_materialization_identity", reasons)
        self.assertIn("branch_id_mixes_source_record_digest", reasons)
        self.assertIn("branch_id_mixes_materialized_record_digest", reasons)
        self.assertIn("branch_id_mixes_branch_state_digest", reasons)
        self.assertIn("branch_id_mixes_current_state_payload", reasons)
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "dataset_contract_invalid_closed_no_training"
            ),
        )
        self._assert_training_closed(report)

    def test_smuggled_trainable_values_fail_closed_without_training(self) -> None:
        row = _transition_row(
            branch_id="branch-a",
            seed=19,
            forced_action="stay",
            value_offset=0.01,
        )
        previous = row["previous_same_agent_public_context"]
        self.assertIsInstance(previous, dict)
        previous["note"] = "seed:41 reward:1.0"
        row["trainable_public_features"][
            "previous_same_agent_public_context"
        ] = previous
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _rows = _write_inputs(tmpdir, [row])

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                min_row_count=1,
                min_seed_count=1,
                min_branch_count=1,
                min_forced_action_count=1,
            )

        self.assertFalse(report["value_leakage_scan"]["passed"])
        self.assertFalse(report["feature_contract_audit"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "dataset_contract_invalid_closed_no_training"
            ),
        )
        self.assertFalse(report["training_ran"])
        self._assert_training_closed(report)

    def test_unexpected_numeric_trainable_context_value_fails_contract(self) -> None:
        row = _transition_row(
            branch_id="branch-a",
            seed=19,
            forced_action="stay",
            value_offset=0.01,
        )
        previous = row["previous_same_agent_public_context"]
        self.assertIsInstance(previous, dict)
        previous["note"] = 41
        row["trainable_public_features"][
            "previous_same_agent_public_context"
        ] = previous
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _rows = _write_inputs(tmpdir, [row])

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                min_row_count=1,
                min_seed_count=1,
                min_branch_count=1,
                min_forced_action_count=1,
            )

        self.assertTrue(report["leakage_scan"]["passed"])
        self.assertFalse(report["value_leakage_scan"]["passed"])
        self.assertFalse(report["feature_contract_audit"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "dataset_contract_invalid_closed_no_training"
            ),
        )
        self.assertFalse(report["training_ran"])
        self._assert_training_closed(report)

    def test_action_mask_audit_failure_fails_closed_without_training(self) -> None:
        row = _transition_row(
            branch_id="branch-a",
            seed=19,
            forced_action="stay",
            value_offset=0.01,
        )
        row["current_public_action_mask"]["stay"] = False
        row["trainable_public_features"]["current_public_action_mask"] = row[
            "current_public_action_mask"
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _rows = _write_inputs(tmpdir, [row])

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                min_row_count=1,
                min_seed_count=1,
                min_branch_count=1,
                min_forced_action_count=1,
            )

        self.assertFalse(report["action_mask_audit"]["passed"])
        self.assertEqual(
            report["action_mask_audit"]["failures"][0]["reason"],
            "forced_action_not_current_mask_supported",
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "dataset_contract_invalid_closed_no_training"
            ),
        )
        self.assertFalse(report["training_ran"])
        self._assert_training_closed(report)

    def test_observation_decode_failure_fails_closed_without_training(self) -> None:
        row = _transition_row(
            branch_id="branch-a",
            seed=19,
            forced_action="stay",
            value_offset=0.01,
        )
        row["current_public_observation"]["data"] = "not-valid-base64"
        row["trainable_public_features"]["current_public_observation"] = row[
            "current_public_observation"
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _rows = _write_inputs(tmpdir, [row])

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                min_row_count=1,
                min_seed_count=1,
                min_branch_count=1,
                min_forced_action_count=1,
            )

        self.assertFalse(report["observation_audit"]["passed"])
        self.assertEqual(
            report["observation_audit"]["failures"][0]["field"],
            "current_public_observation",
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "dataset_contract_invalid_closed_no_training"
            ),
        )
        self.assertFalse(report["training_ran"])
        self._assert_training_closed(report)

    def test_expected_dataset_digest_mismatch_fails_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _rows = _write_inputs(
                tmpdir,
                [
                    _transition_row(
                        branch_id="branch-a",
                        seed=19,
                        forced_action="stay",
                        value_offset=0.01,
                    )
                ],
            )

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["dataset"],
                v177_report_path=paths["v177_report"],
                output_path=paths["report"],
                expected_dataset_digest="wrong",
                min_row_count=1,
                min_seed_count=1,
                min_branch_count=1,
                min_forced_action_count=1,
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v177_dataset_digest_mismatch",
            report["source_validation"]["failures"],
        )
        self._assert_training_closed(report)
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "source_invalid_closed_no_training"
            ),
        )

    def test_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows = _write_inputs(
                tmpdir,
                [
                    _transition_row(
                        branch_id="branch-a",
                        seed=19,
                        forced_action="stay",
                        value_offset=0.01,
                    ),
                    _transition_row(
                        branch_id="branch-b",
                        seed=41,
                        forced_action="drink",
                        value_offset=0.02,
                    ),
                ],
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v178_transition_row_dataset_audit",
                    "--transition-dataset",
                    str(paths["dataset"]),
                    "--v177-report",
                    str(paths["v177_report"]),
                    "--output",
                    str(paths["report"]),
                    "--expected-dataset-digest",
                    stable_payload_digest(rows),
                    "--min-row-count",
                    "2",
                    "--min-seed-count",
                    "2",
                    "--min-branch-count",
                    "2",
                    "--min-forced-action-count",
                    "2",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v178_transition_row_dataset_audit=",
            completed.stdout,
        )
        self.assertIn("classification=", completed.stdout)
        self.assertIn("value_leakage_scan_passed=True", completed.stdout)
        self.assertIn("feature_contract_audit_passed=True", completed.stdout)
        self.assertIn("training_authorized=False", completed.stdout)
        self.assertIn(
            "training_authorization.next_same_lane_opt_in_training_slice_authorized=False",
            completed.stdout,
        )
        self.assertIn(
            "route_recommendation.transition_row_training_authorized=False",
            completed.stdout,
        )
        self.assertIn(
            "contract.next_same_lane_opt_in_training_slice_authorized=False",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            v178.M3_CARRION_SURVIVOR_CONTINUATION_V178_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(written["dataset"]["row_count"], 2)

    def test_cli_prints_authorized_support_ready_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_ready_rows()
            paths, rows = _write_inputs(tmpdir, rows)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v178_transition_row_dataset_audit",
                    "--transition-dataset",
                    str(paths["dataset"]),
                    "--v177-report",
                    str(paths["v177_report"]),
                    "--output",
                    str(paths["report"]),
                    "--expected-dataset-digest",
                    stable_payload_digest(rows),
                    "--expected-source-report-exact-digest",
                    _report_exact_digest(paths["v177_report"]),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["report"].read_text(encoding="utf-8"))

        self.assertIn(
            "classification="
            "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
            "valid_support_ready_transition_row_training_authorized",
            completed.stdout,
        )
        self.assertIn(
            "recommended_next_route=v179_transition_row_policy_training_slice_opt_in",
            completed.stdout,
        )
        self.assertIn(
            "route_recommendation.transition_row_training_authorized=True",
            completed.stdout,
        )
        self.assertIn(
            "contract.next_same_lane_opt_in_training_slice_authorized=True",
            completed.stdout,
        )
        self.assertIn("training_authorized=False", completed.stdout)
        self.assertIn(
            "training_authorization.next_same_lane_opt_in_training_slice_authorized=True",
            completed.stdout,
        )
        self.assertFalse(written["training_authorized"])
        self.assertTrue(
            written["training_authorization"][
                "next_same_lane_opt_in_training_slice_authorized"
            ]
        )
        self.assertTrue(
            written["route_recommendation"]["transition_row_training_authorized"]
        )

    def _assert_training_closed(self, report: dict[str, object]) -> None:
        self.assertFalse(report["training_authorized"])
        self.assertFalse(
            report["route_recommendation"]["transition_row_training_authorized"]
        )
        self.assertFalse(
            report["route_recommendation"]["first_opt_in_training_slice_authorized"]
        )
        self.assertFalse(
            report["contract"]["next_same_lane_opt_in_training_slice_authorized"]
        )
        self.assertFalse(report["training_authorization"]["authorized"])


def _write_inputs(
    tmpdir: str,
    rows: list[dict[str, object]],
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    root = Path(tmpdir)
    paths = {
        "v177_report": root / "v177.json",
        "dataset": root / "v177-transition-rows.jsonl",
        "report": root / "v178.json",
    }
    _write_jsonl(paths["dataset"], rows)
    dataset_digest = stable_payload_digest(rows)
    v177_report = _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION
            ),
            "policy": (
                M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY
            ),
            "classification": {
                "primary": v178.EXPECTED_V177_CLASSIFICATION,
                "labels": [v178.EXPECTED_V177_CLASSIFICATION],
            },
            "dataset": {
                "row_count": len(rows),
                "dataset_digest": dataset_digest,
            },
            "diagnostics_only": True,
            "contract": {
                "diagnostics_only": True,
                "training_allowed": False,
                "fit_allowed": False,
                "runtime_artifact_allowed": False,
                "runtime_action_change_allowed": False,
                "shadow_or_live_eval_allowed": False,
                "promotion_allowed": False,
                "gate_relaxation_allowed": False,
                "replay_viewer_schema_change_allowed": False,
                "source_identity_metadata_only": True,
                "current_and_next_public_fields_are_dataset_inputs": True,
                "short_horizon_outcomes_are_diagnostic_targets_only": True,
            },
            "training_ran": False,
            "training_authorized": False,
            "fit_ran": False,
            "scorer_retraining_ran": False,
            "scorer_retraining_authorized": False,
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
            "non_promoted": True,
        }
    )
    _write_json(paths["v177_report"], v177_report)
    return paths, rows


def _support_ready_rows() -> list[dict[str, object]]:
    seeds = [101, 103, 107]
    actions = list(ACTION_NAMES[: v178.DEFAULT_MIN_FORCED_ACTION_COUNT])
    return [
        _transition_row(
            branch_id=f"branch-{index:03d}",
            seed=seeds[index % len(seeds)],
            forced_action=actions[index % len(actions)],
            value_offset=0.01 + index * 0.0001,
        )
        for index in range(v178.DEFAULT_MIN_ROW_COUNT)
    ]


def _transition_row(
    *,
    branch_id: str,
    seed: int,
    forced_action: str,
    value_offset: float,
) -> dict[str, object]:
    branch_number = _branch_number(branch_id)
    current_observation = _encoded_observation(value_offset)
    next_observation = _encoded_observation(value_offset + 0.1)
    previous_observation = _encoded_observation(value_offset - 0.1)
    current_mask = {action: True for action in ACTION_NAMES}
    next_mask = {action: True for action in ACTION_NAMES}
    previous_context = {
        "available": True,
        "public_observation": previous_observation,
        "public_action_mask": current_mask,
        "public_action": "eat",
        "moved": False,
    }
    trainable_features = {
        "current_public_observation": current_observation,
        "current_public_action_mask": current_mask,
        "forced_action": forced_action,
        "previous_same_agent_public_context": previous_context,
        "next_public_observation": next_observation,
        "next_public_action_mask": next_mask,
    }
    return {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION
        ),
        "row_origin": "v177_exact_branch_replay_from_v176_shard_plan",
        "feature_policy_id": (
            M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY
        ),
        "trainable_public_features": trainable_features,
        "current_public_observation": current_observation,
        "current_public_action_mask": current_mask,
        "forced_action": forced_action,
        "previous_same_agent_public_context": previous_context,
        "next_public_observation": next_observation,
        "next_public_action_mask": next_mask,
        "next_public_observation_available": True,
        "next_public_action_mask_available": True,
        "transition_done": False,
        "short_horizon_public_outcome_summary": {
            "forced_action_used": True,
            "current_requested_action": forced_action,
            "current_resolved_action": forced_action,
            "current_action_valid": True,
            "current_resolution_action_valid": True,
            "current_moved": forced_action.startswith("move_"),
            "current_reward_total": 0.25,
            "current_resource_gain": 0.5 if forced_action in {"eat", "drink"} else 0.0,
            "target_terminal": {
                "alive": True,
                "energy_ratio": 0.75,
                "health_ratio": 0.8,
                "hydration_ratio": 0.65,
            },
            "alive_agents": 12,
            "births": 2,
            "deaths": 1,
        },
        "replay_verification": {
            "verified": True,
            "expected_digest": f"digest-{branch_id}-{forced_action}",
            "actual_digest": f"digest-{branch_id}-{forced_action}",
        },
        "metadata": {
            "metadata_schema_version": (
                "m3_carrion_survivor_continuation_v177_compact_transition_metadata_v1"
            ),
            "branch_id": branch_id,
            "seed": seed,
            "fixture": "broad",
            "branch_tick": 94 + branch_number,
            "agent_id": 17 + branch_number,
            "source_path": "source.jsonl",
            "line_number": 2 + branch_number,
            "source_row_index": 563 + branch_number,
            "failed_safe_action": "drink",
            "failure_types": ["unique_top_1_miss", "unique_top_3_miss"],
            "source_record_digest": f"source-{branch_id}",
            "materialized_record_digest": f"materialized-{branch_id}",
            "branch_state_digest": f"state-{branch_id}",
            "replay_verification_digest": f"digest-{branch_id}-{forced_action}",
            "source_identity_used_as_trainable_input": False,
            "runtime_requested_or_resolved_action_used_as_trainable_input": False,
            "current_or_future_outcome_used_as_trainable_input": False,
            "diagnostic_target_used_as_trainable_input": False,
        },
    }


def _branch_number(branch_id: str) -> int:
    digits = "".join(char for char in branch_id if char.isdigit())
    if digits:
        return int(digits)
    return sum(ord(char) for char in branch_id) % 1000


def _encoded_observation(offset: float) -> dict[str, object]:
    values = [
        max(-1.0, min(1.0, offset + (index % 7) * 0.001))
        for index in range(observations.OBSERVATION_INPUT_VECTOR_SIZE)
    ]
    packed = struct.pack(
        f"<{len(values)}h",
        *[
            int(round(value * observations.OBSERVATION_QUANTIZATION_SCALE))
            for value in values
        ],
    )
    return {
        "schema_version": observations.OBSERVATION_SCHEMA_VERSION,
        "encoder_version": observations.OBSERVATION_ENCODER_VERSION,
        "decoded_dtype": observations.OBSERVATION_INPUT_DTYPE,
        "storage_dtype": observations.OBSERVATION_STORAGE_DTYPE,
        "storage_encoding": observations.OBSERVATION_STORAGE_ENCODING,
        "shape": [observations.OBSERVATION_INPUT_VECTOR_SIZE],
        "value_range": list(observations.OBSERVATION_INPUT_VALUE_RANGE),
        "data": base64.b64encode(zlib.compress(packed, level=6)).decode("ascii"),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _report_exact_digest(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8")).get("exact_digest") or "")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _report(payload: dict[str, object]) -> dict[str, object]:
    report = dict(payload)
    report["exact_digest"] = _digest_without_exact(report)
    return report


def _digest_without_exact(payload: dict[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(without_digest, sort_keys=True, allow_nan=False))
    )


if __name__ == "__main__":
    unittest.main()
