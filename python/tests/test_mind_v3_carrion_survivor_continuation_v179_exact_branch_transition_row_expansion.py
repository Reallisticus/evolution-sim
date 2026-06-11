from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock
import zlib

from evolution_sim.env.runtime import observations
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind import (
    carrion_survivor_continuation_v178_transition_row_dataset_audit as v178,
    carrion_survivor_continuation_v179_exact_branch_transition_row_expansion as v179,
)
from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY,
)
from evolution_sim.mind.carrion_survivor_continuation_v177_exact_branch_replay_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION,
    _record_materialization_payload,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]
SUPPORT_SEEDS = [5, 13, 19, 29, 37, 41]
PUBLIC_ACTIONS = list(ACTION_NAMES[:6])


class MindV3CarrionSurvivorContinuationV179ExactBranchTransitionRowExpansionTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v179-exact-branch-transition-row-expansion"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion"
            ),
        )

    def test_default_plan_exceeds_v178_support_floors(self) -> None:
        rows = _v172_rows()

        plan_rows, plan = v179.build_v179_plan_rows(
            rows,
            support_provenance_seeds=SUPPORT_SEEDS,
            branches_per_seed=v179.DEFAULT_BRANCHES_PER_SEED,
            max_forced_actions_per_branch=v179.DEFAULT_MAX_FORCED_ACTIONS_PER_BRANCH,
        )

        self.assertTrue(plan["passed"])
        self.assertEqual(plan["selected_plan_row_count"], 30)
        self.assertEqual(plan["selected_seed_count"], 6)
        self.assertGreaterEqual(
            sum(len(row["candidate_forced_actions"]) for row in plan_rows),
            v179.V178_DEFAULT_MIN_ROW_COUNT,
        )
        self.assertGreaterEqual(
            plan["candidate_forced_action_count"],
            v179.V178_DEFAULT_MIN_FORCED_ACTION_COUNT,
        )

    def test_source_digest_mismatch_fails_closed_without_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)

            report = v179.run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                v177_report_path=paths["v177_report"],
                v177_transition_dataset_path=paths["v177_dataset"],
                output_path=paths["report"],
                transition_dataset_output_path=paths["dataset"],
                expected_v172_exact_digest="wrong",
                expected_v172_dataset_digest=stable_payload_digest(
                    payloads["v172_rows"]
                ),
                expected_v172_row_count=len(payloads["v172_rows"]),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v172_unexpected_exact_digest",
            report["source_validation"]["failures"],
        )
        self.assertEqual(report["dataset"]["row_count"], 0)
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_"
                "expansion_source_invalid_closed_no_training"
            ),
        )
        self._assert_training_closed(report)

    def test_deterministic_output_and_lifecycle_flags_stay_no_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            rows = _v178_support_ready_rows()
            tagged_rows = v179._tag_v179_rows(rows)
            first = _run_with_mocked_transition_rows(paths, payloads, rows)
            second = _run_with_mocked_transition_rows(paths, payloads, rows)
            written_rows = [
                json.loads(line)
                for line in paths["dataset"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(first, second)
        self.assertEqual(first["exact_digest"], _digest_without_exact(first))
        self.assertEqual(
            first["dataset"]["dataset_digest"],
            stable_payload_digest(tagged_rows),
        )
        self.assertEqual(written_rows, tagged_rows)
        self.assertEqual(
            first["classification"]["primary"],
            v179.V179_SUPPORT_READY_CLASSIFICATION,
        )
        self.assertEqual(
            first["route_recommendation"]["recommended_next_route"],
            "v178_transition_row_dataset_audit_default_thresholds_no_training",
        )
        self.assertTrue(first["support_summary"]["passed"])
        self.assertFalse(
            first["route_recommendation"]["transition_row_training_authorized"]
        )
        self._assert_training_closed(first)

    def test_no_verify_replay_cannot_be_support_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            rows = _v178_support_ready_rows()

            report = _run_with_mocked_transition_rows(
                paths,
                payloads,
                rows,
                verify_replay=False,
            )

        self.assertTrue(report["support_summary"]["passed"])
        self.assertFalse(report["metrics"]["replay_verification_enabled"])
        self.assertFalse(report["metrics"]["all_replays_verified"])
        self.assertEqual(
            report["metrics"]["v179_replay_verification_failure_reason"],
            "replay_verification_disabled",
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_"
                "expansion_replay_not_deterministic_closed_no_training"
            ),
        )
        self.assertFalse(
            report["route_recommendation"][
                "v178_default_threshold_audit_recommended"
            ]
        )
        self._assert_training_closed(report)

    def test_duplicate_source_branch_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            rows = list(payloads["v172_rows"])
            rows[1] = dict(rows[1])
            rows[1]["metadata"] = dict(rows[1]["metadata"])
            rows[1]["metadata"]["branch_id"] = rows[0]["metadata"]["branch_id"]
            _write_jsonl(paths["v172_dataset"], rows)
            _write_json(
                paths["v172_report"],
                _v172_report(rows, exact_classification=payloads["v172_classification"]),
            )

            report = v179.run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                v177_report_path=paths["v177_report"],
                v177_transition_dataset_path=paths["v177_dataset"],
                output_path=paths["report"],
                transition_dataset_output_path=paths["dataset"],
                expected_v172_exact_digest=None,
                expected_v172_dataset_digest=stable_payload_digest(rows),
                expected_v172_row_count=len(rows),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v172_source_row_integrity_failed",
            report["source_validation"]["failures"],
        )
        self.assertIn(
            "duplicate_branch_id",
            {
                failure["reason"]
                for failure in report["source_validation"]["v172_source_row_integrity"][
                    "failures"
                ]
            },
        )
        self._assert_training_closed(report)

    def test_cloned_source_identity_across_v172_branch_rows_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            rows = list(payloads["v172_rows"])
            first_metadata = dict(rows[0]["metadata"])
            rows[1] = dict(rows[1])
            rows[1]["metadata"] = dict(rows[1]["metadata"])
            for field in (
                "seed",
                "source_path",
                "line_number",
                "branch_tick",
                "agent_id",
                "source_record_digest",
            ):
                rows[1]["metadata"][field] = first_metadata[field]
            _write_jsonl(paths["v172_dataset"], rows)
            _write_json(
                paths["v172_report"],
                _v172_report(rows, exact_classification=payloads["v172_classification"]),
            )

            report = v179.run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                v177_report_path=paths["v177_report"],
                v177_transition_dataset_path=paths["v177_dataset"],
                output_path=paths["report"],
                transition_dataset_output_path=paths["dataset"],
                expected_v172_exact_digest=None,
                expected_v172_dataset_digest=stable_payload_digest(rows),
                expected_v172_row_count=len(rows),
                expected_v177_report_exact_digest=payloads["v177_report"]["exact_digest"],
                expected_v177_dataset_digest=stable_payload_digest(
                    payloads["v177_rows"]
                ),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v172_source_row_integrity_failed",
            report["source_validation"]["failures"],
        )
        reasons = {
            failure["reason"]
            for failure in report["source_validation"]["v172_source_row_integrity"][
                "failures"
            ]
        }
        self.assertIn("source_materialization_identity_reused_across_branches", reasons)
        self.assertIn("source_record_digest_reused_across_branches", reasons)
        self._assert_training_closed(report)

    def test_missing_v172_lifecycle_field_fails_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            v172_report = dict(payloads["v172_report"])
            v172_report.pop("training_ran")
            _write_json(paths["v172_report"], _report(v172_report))

            report = v179.run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                v177_report_path=paths["v177_report"],
                v177_transition_dataset_path=paths["v177_dataset"],
                output_path=paths["report"],
                transition_dataset_output_path=paths["dataset"],
                expected_v172_exact_digest=_digest_without_exact(v172_report),
                expected_v172_dataset_digest=stable_payload_digest(
                    payloads["v172_rows"]
                ),
                expected_v172_row_count=len(payloads["v172_rows"]),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v172_lifecycle_not_diagnostics_only",
            report["source_validation"]["failures"],
        )
        self.assertIn(
            {
                "field": "training_ran",
                "expected": False,
                "observed": None,
            },
            report["source_validation"]["v172_lifecycle_validation"]["failures"],
        )
        self._assert_training_closed(report)

    def test_v177_training_allowed_contract_fails_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            v177_report = dict(payloads["v177_report"])
            v177_report["contract"] = dict(v177_report["contract"])
            v177_report["contract"]["training_allowed"] = True
            _write_json(paths["v177_report"], _report(v177_report))

            report = v179.run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                v177_report_path=paths["v177_report"],
                v177_transition_dataset_path=paths["v177_dataset"],
                output_path=paths["report"],
                transition_dataset_output_path=paths["dataset"],
                expected_v172_exact_digest=payloads["v172_report"]["exact_digest"],
                expected_v172_dataset_digest=stable_payload_digest(
                    payloads["v172_rows"]
                ),
                expected_v172_row_count=len(payloads["v172_rows"]),
                expected_v177_report_exact_digest=_digest_without_exact(v177_report),
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

    def test_v177_expected_exact_digest_mismatch_fails_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)

            report = v179.run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                v177_report_path=paths["v177_report"],
                v177_transition_dataset_path=paths["v177_dataset"],
                output_path=paths["report"],
                transition_dataset_output_path=paths["dataset"],
                expected_v172_exact_digest=payloads["v172_report"]["exact_digest"],
                expected_v172_dataset_digest=stable_payload_digest(
                    payloads["v172_rows"]
                ),
                expected_v172_row_count=len(payloads["v172_rows"]),
                expected_v177_report_exact_digest="wrong",
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v177_unexpected_exact_digest",
            report["source_validation"]["failures"],
        )
        self._assert_training_closed(report)

    def test_stale_source_record_fails_selection_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            _write_source_records(
                paths["source_records"],
                payloads["v172_rows"],
                stale_first_record=True,
            )

            report = v179.run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                v177_report_path=paths["v177_report"],
                v177_transition_dataset_path=paths["v177_dataset"],
                output_path=paths["report"],
                transition_dataset_output_path=paths["dataset"],
                expected_v172_exact_digest=payloads["v172_report"]["exact_digest"],
                expected_v172_dataset_digest=stable_payload_digest(
                    payloads["v172_rows"]
                ),
                expected_v172_row_count=len(payloads["v172_rows"]),
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertFalse(report["selection"]["passed"])
        self.assertEqual(
            report["selection"]["failures"][0]["reason"],
            "source_record_plan_mismatch",
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_"
                "expansion_selection_invalid_closed_no_training"
            ),
        )
        self._assert_training_closed(report)

    def test_source_record_digest_mismatch_fails_selection_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            _write_source_records(
                paths["source_records"],
                payloads["v172_rows"],
                mutate_first_record_digest=True,
            )

            report = v179.run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                v177_report_path=paths["v177_report"],
                v177_transition_dataset_path=paths["v177_dataset"],
                output_path=paths["report"],
                transition_dataset_output_path=paths["dataset"],
                expected_v172_exact_digest=payloads["v172_report"]["exact_digest"],
                expected_v172_dataset_digest=stable_payload_digest(
                    payloads["v172_rows"]
                ),
                expected_v172_row_count=len(payloads["v172_rows"]),
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertFalse(report["selection"]["passed"])
        self.assertEqual(
            report["selection"]["failures"][0]["reason"],
            "source_record_plan_mismatch",
        )
        self.assertIn(
            "source_record_digest",
            {
                mismatch["field"]
                for mismatch in report["selection"]["failures"][0]["mismatches"]
            },
        )
        self._assert_training_closed(report)

    def test_forced_action_not_used_fails_v179_support_ready_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            rows = _v178_support_ready_rows()
            rows[0]["short_horizon_public_outcome_summary"] = dict(
                rows[0]["short_horizon_public_outcome_summary"]
            )
            rows[0]["short_horizon_public_outcome_summary"][
                "forced_action_used"
            ] = False

            report = _run_with_mocked_transition_rows(paths, payloads, rows)

        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_"
                "expansion_forced_action_not_used_closed_no_training"
            ),
        )
        self.assertFalse(report["metrics"]["all_forced_actions_used"])
        self.assertFalse(
            report["route_recommendation"]["v178_default_threshold_audit_recommended"]
        )
        self._assert_training_closed(report)

    def test_trainable_leakage_fails_closed_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            rows = _v178_support_ready_rows()
            rows[0]["trainable_public_features"] = dict(rows[0]["trainable_public_features"])
            rows[0]["trainable_public_features"]["branch_id"] = "leak"

            report = _run_with_mocked_transition_rows(paths, payloads, rows)

        self.assertFalse(report["leakage_scan"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_"
                "expansion_compact_transition_rows_invalid_closed_no_training"
            ),
        )
        self._assert_training_closed(report)

    def test_cli_writes_parseable_report_and_prints_route_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion",
                    "--v172-report",
                    str(paths["v172_report"]),
                    "--v172-dataset",
                    str(paths["v172_dataset"]),
                    "--v177-report",
                    str(paths["v177_report"]),
                    "--v177-transition-dataset",
                    str(paths["v177_dataset"]),
                    "--output",
                    str(paths["report"]),
                    "--transition-dataset-output",
                    str(paths["dataset"]),
                    "--expected-v172-exact-digest",
                    "wrong",
                    "--expected-v172-dataset-digest",
                    stable_payload_digest(payloads["v172_rows"]),
                    "--expected-v172-row-count",
                    str(len(payloads["v172_rows"])),
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
            "carrion_survivor_continuation_v179_exact_branch_transition_row_"
            "expansion_report=",
            completed.stdout,
        )
        self.assertIn("classification=", completed.stdout)
        self.assertIn("recommended_next_route=", completed.stdout)
        self.assertIn(
            "route_recommendation.transition_row_training_authorized=False",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            v179.M3_CARRION_SURVIVOR_CONTINUATION_V179_EXACT_BRANCH_TRANSITION_ROW_EXPANSION_SCHEMA_VERSION,
        )
        self.assertEqual(written["dataset"]["row_count"], 0)

    def test_v178_audits_v179_output_at_default_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = _v178_support_ready_rows()
            dataset = root / "v179-rows.jsonl"
            v179_report = root / "v179.json"
            v178_report = root / "v178.json"
            _write_jsonl(dataset, rows)
            v179_payload = _v179_report(rows, dataset)
            _write_json(v179_report, v179_payload)

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=dataset,
                v177_report_path=v179_report,
                output_path=v178_report,
                expected_v177_report_exact_digest=v179_payload["exact_digest"],
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertEqual(
            report["source_validation"]["source_producer"],
            "v179_exact_branch_transition_row_expansion",
        )
        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["default_support_readiness"]["passed"])
        self.assertTrue(report["training_authorization"]["authorized"])
        self.assertTrue(
            report["training_authorization"][
                "next_same_lane_opt_in_training_slice_authorized"
            ]
        )
        self.assertEqual(
            report["route_recommendation"]["recommended_next_route"],
            "v179_transition_row_policy_training_slice_opt_in",
        )
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["promotion_authorized"])

    def test_v178_rejects_v179_output_without_upstream_v177_pins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = _v178_support_ready_rows()
            dataset = root / "v179-rows.jsonl"
            v179_report = root / "v179.json"
            v178_report = root / "v178.json"
            _write_jsonl(dataset, rows)
            report_payload = _v179_report(rows, dataset)
            source_validation = report_payload["source_validation"]
            self.assertIsInstance(source_validation, dict)
            source_validation["expected_v177_report_exact_digest"] = None
            source_validation["expected_v177_report_exact_digest_provided"] = False
            source_validation["expected_v177_dataset_digest"] = None
            source_validation["expected_v177_dataset_digest_provided"] = False
            source_validation["v177_source_digests_pinned"] = False
            report_payload = _report(report_payload)
            _write_json(v179_report, report_payload)

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=dataset,
                v177_report_path=v179_report,
                output_path=v178_report,
                expected_v177_report_exact_digest=report_payload["exact_digest"],
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertEqual(
            report["source_validation"]["source_producer"],
            "v179_exact_branch_transition_row_expansion",
        )
        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v179_v177_source_digests_not_pinned",
            report["source_validation"]["failures"],
        )
        self.assertFalse(
            report["source_validation"]["v179_replay_validation"]["passed"]
        )
        self.assertFalse(
            report["source_validation"]["v179_replay_validation"][
                "v177_source_digests_pinned"
            ]
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "source_invalid_closed_no_training"
            ),
        )
        self.assertFalse(report["training_authorization"]["authorized"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["promotion_authorized"])

    def test_v178_rejects_v179_output_without_replay_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = _v178_support_ready_rows()
            dataset = root / "v179-rows.jsonl"
            v179_report = root / "v179.json"
            v178_report = root / "v178.json"
            report_payload = _v179_report(rows, dataset)
            report_payload["inputs"]["verify_replay"] = False
            report_payload["metrics"]["replay_verification_enabled"] = False
            report_payload["metrics"]["all_replays_verified"] = False
            _write_jsonl(dataset, rows)
            report_payload = _report(report_payload)
            _write_json(v179_report, report_payload)

            report = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=dataset,
                v177_report_path=v179_report,
                output_path=v178_report,
                expected_v177_report_exact_digest=report_payload["exact_digest"],
                expected_dataset_digest=stable_payload_digest(rows),
            )

        self.assertEqual(
            report["source_validation"]["source_producer"],
            "v179_exact_branch_transition_row_expansion",
        )
        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v179_replay_verification_not_proven",
            report["source_validation"]["failures"],
        )
        self.assertFalse(
            report["source_validation"]["v179_replay_validation"]["passed"]
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
                "source_invalid_closed_no_training"
            ),
        )
        self.assertFalse(report["training_authorization"]["authorized"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["promotion_authorized"])

    def _assert_training_closed(self, report: dict[str, object]) -> None:
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])


def _run_with_mocked_transition_rows(
    paths: dict[str, Path],
    payloads: dict[str, object],
    rows: list[dict[str, object]],
    *,
    verify_replay: bool = True,
) -> dict[str, object]:
    with mock.patch.object(
        v179,
        "build_selected_branch_points",
        return_value=(
            ["selected"] * 30,
            {},
            {
                "passed": True,
                "selected_branch_point_count": 30,
                "plan_row_count": 30,
                "failure_count": 0,
                "failures": [],
            },
        ),
    ), mock.patch.object(
        v179,
        "materialize_selected_branch_points",
        return_value=(
            ["materialized"] * 30,
            {
                "passed": True,
                "selected_branch_point_count": 30,
                "materialized_branch_point_count": 30,
            },
        ),
    ), mock.patch.object(
        v179,
        "build_compact_transition_rows",
        return_value=rows,
    ):
        return v179.run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
            v172_report_path=paths["v172_report"],
            v172_dataset_path=paths["v172_dataset"],
            v177_report_path=paths["v177_report"],
            v177_transition_dataset_path=paths["v177_dataset"],
            output_path=paths["report"],
            transition_dataset_output_path=paths["dataset"],
            expected_v172_exact_digest=payloads["v172_report"]["exact_digest"],
            expected_v172_dataset_digest=stable_payload_digest(payloads["v172_rows"]),
            expected_v172_row_count=len(payloads["v172_rows"]),
            expected_v177_report_exact_digest=payloads["v177_report"]["exact_digest"],
            expected_v177_dataset_digest=stable_payload_digest(payloads["v177_rows"]),
            verify_replay=verify_replay,
        )


def _write_inputs(tmpdir: str) -> tuple[dict[str, Path], dict[str, object]]:
    root = Path(tmpdir)
    paths = {
        "source_records": root / "source.jsonl",
        "v172_report": root / "v172.json",
        "v172_dataset": root / "v172.jsonl",
        "v177_report": root / "v177.json",
        "v177_dataset": root / "v177.jsonl",
        "report": root / "v179.json",
        "dataset": root / "v179.jsonl",
    }
    v172_rows = _v172_rows(source_path=paths["source_records"])
    source_records = _source_records(v172_rows)
    _sync_source_record_digests(v172_rows, source_records)
    v177_rows = [_v177_transition_row()]
    v172_report = _v172_report(v172_rows)
    v177_report = _v177_report(v177_rows)
    _write_jsonl(paths["source_records"], source_records)
    _write_jsonl(paths["v172_dataset"], v172_rows)
    _write_json(paths["v172_report"], v172_report)
    _write_jsonl(paths["v177_dataset"], v177_rows)
    _write_json(paths["v177_report"], v177_report)
    return paths, {
        "v172_rows": v172_rows,
        "v172_report": v172_report,
        "v172_classification": v179.EXPECTED_V172_CLASSIFICATION,
        "v177_rows": v177_rows,
        "v177_report": v177_report,
    }


def _v172_rows(
    *,
    source_path: Path | str = "source.jsonl",
) -> list[dict[str, object]]:
    rows = []
    for index in range(30):
        seed = SUPPORT_SEEDS[index % len(SUPPORT_SEEDS)]
        rows.append(_v172_row(index=index, seed=seed, source_path=source_path))
    return rows


def _v172_row(
    *,
    index: int,
    seed: int,
    source_path: Path | str,
) -> dict[str, object]:
    mask = {action: action in PUBLIC_ACTIONS for action in ACTION_NAMES}
    safe_action = PUBLIC_ACTIONS[index % len(PUBLIC_ACTIONS)]
    targets = []
    for action in ACTION_NAMES:
        public = mask[action]
        safe = action == safe_action
        targets.append(
            {
                "action": action,
                "public_mask": public,
                "target_available": public,
                "replay_verified": public,
                "safe_target": safe,
                "value_target": 1.0 if safe else 0.25 if public else None,
            }
        )
    branch_tick = 90 + index
    return {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ),
        "row_origin": "v172_from_v171_replay_verified_target_row",
        "feature_policy_id": (
            M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY
        ),
        "trainable_public_features": {
            "public_observation": {"schema_version": "synthetic", "values": [index]},
            "action_mask": mask,
        },
        "public_action_mask": mask,
        "action_value_targets": targets,
        "target_best_outcome_action_set": [safe_action],
        "safe_action_set": [safe_action],
        "target_outcome_summary": {"best_outcome_action_set": [safe_action]},
        "target_classification": "unique_replay_verified_best_action",
        "robust_winner_action": safe_action,
        "metadata": {
            "metadata_schema_version": (
                "m3_carrion_survivor_continuation_v172_replay_target_metadata_v1"
            ),
            "branch_id": f"branch-{index:03d}",
            "seed": seed,
            "fixture": "broad",
            "branch_tick": branch_tick,
            "agent_id": 100 + index,
            "source_path": str(source_path),
            "line_number": index + 1,
            "source_row_index": index,
            "branch_state_digest": f"state-{index}",
            "source_record_digest": f"source-{index}",
            "source_seed_is_support_provenance_not_future_promotion_holdout": True,
            "source_identity_used_as_trainable_input": False,
            "runtime_requested_or_resolved_action_used_as_trainable_input": False,
            "future_outcome_used_as_trainable_input": False,
            "target_safe_action_used_as_trainable_input": False,
            "private_state_used_as_trainable_input": False,
        },
    }


def _source_records(
    rows: list[dict[str, object]],
    *,
    stale_first_record: bool = False,
    mutate_first_record_digest: bool = False,
) -> list[dict[str, object]]:
    records = []
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        tick = metadata["branch_tick"] + 1 if index == 0 and stale_first_record else metadata["branch_tick"]
        observation_values = [index + 1000] if index == 0 and mutate_first_record_digest else [index]
        records.append(
            {
                "record": {
                    "tick": tick,
                    "agent_id": metadata["agent_id"],
                    "observation_schema": "mind_observation_v3",
                    "observation_metadata": {"seed": metadata["seed"]},
                    "observation_input": {
                        "schema_version": "synthetic",
                        "values": observation_values,
                    },
                    "observation_digest": (
                        f"obs-mutated-{index}"
                        if index == 0 and mutate_first_record_digest
                        else f"obs-{index}"
                    ),
                    "action_mask": row["public_action_mask"],
                    "requested_action": "stay",
                    "resolved_action": "stay",
                    "moved": False,
                    "before": {"alive": True},
                    "after": {"alive": True},
                }
            }
        )
    return records


def _write_source_records(
    path: Path,
    rows: list[dict[str, object]],
    *,
    stale_first_record: bool = False,
    mutate_first_record_digest: bool = False,
) -> None:
    _write_jsonl(
        path,
        _source_records(
            rows,
            stale_first_record=stale_first_record,
            mutate_first_record_digest=mutate_first_record_digest,
        ),
    )


def _sync_source_record_digests(
    rows: list[dict[str, object]],
    source_records: list[dict[str, object]],
) -> None:
    for row, payload in zip(rows, source_records, strict=True):
        metadata = row["metadata"]
        if isinstance(metadata, dict):
            metadata["source_record_digest"] = stable_payload_digest(
                _record_materialization_payload(payload["record"])
            )


def _v172_report(
    rows: list[dict[str, object]],
    *,
    exact_classification: str = v179.EXPECTED_V172_CLASSIFICATION,
) -> dict[str, object]:
    return _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION
            ),
            "policy": (
                M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_POLICY
            ),
            "classification": {
                "primary": exact_classification,
                "labels": [exact_classification],
            },
            "dataset": {
                "row_count": len(rows),
                "dataset_digest": stable_payload_digest(rows),
            },
            "diagnostics_only": True,
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
            "replay_viewer_schema_changed": False,
            "non_promoted": True,
        }
    )


def _v177_report(rows: list[dict[str, object]]) -> dict[str, object]:
    classification = v179.V177_EXPECTED_CLASSIFICATION
    return _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION
            ),
            "policy": (
                M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY
            ),
            "classification": {"primary": classification, "labels": [classification]},
            "dataset": {
                "row_count": len(rows),
                "dataset_digest": stable_payload_digest(rows),
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
            **v179._lifecycle_flags(),
        }
    )


def _v179_report(
    rows: list[dict[str, object]],
    dataset_path: Path,
) -> dict[str, object]:
    v177_report_digest = "synthetic-v177-report-exact-digest"
    v177_dataset_digest = "synthetic-v177-transition-dataset-digest"
    return _report(
        {
            "schema_version": (
                v179.M3_CARRION_SURVIVOR_CONTINUATION_V179_EXACT_BRANCH_TRANSITION_ROW_EXPANSION_SCHEMA_VERSION
            ),
            "policy": (
                v179.M3_CARRION_SURVIVOR_CONTINUATION_V179_EXACT_BRANCH_TRANSITION_ROW_EXPANSION_POLICY
            ),
            "contract": v179._diagnostics_only_contract(),
            "inputs": {"verify_replay": True},
            "source_validation": {
                "policy": "m3_carrion_survivor_continuation_v179_source_validation_v1",
                "passed": True,
                "failures": [],
                "v177_exact_digest_validation": {
                    "passed": True,
                    "observed": v177_report_digest,
                    "expected": v177_report_digest,
                },
                "expected_v177_report_exact_digest": v177_report_digest,
                "observed_v177_report_exact_digest": v177_report_digest,
                "expected_v177_report_exact_digest_provided": True,
                "expected_v177_dataset_digest": v177_dataset_digest,
                "observed_v177_dataset_digest": v177_dataset_digest,
                "expected_v177_dataset_digest_provided": True,
                "v177_source_digests_pinned": True,
            },
            "branch_materialization": {
                "passed": True,
                "exact_materialization_proven": True,
                "materialized_branch_point_count": len(rows),
            },
            "metrics": {
                "replay_verification_enabled": True,
                "replay_verified_row_count": len(rows),
                "all_replays_verified": True,
            },
            "dataset": {
                "path": str(dataset_path),
                "row_count": len(rows),
                "dataset_digest": stable_payload_digest(rows),
                "row_schema_version": (
                    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION
                ),
                "feature_policy_id": (
                    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY
                ),
            },
            "classification": {
                "primary": v179.V179_SUPPORT_READY_CLASSIFICATION,
                "labels": [v179.V179_SUPPORT_READY_CLASSIFICATION],
            },
            **v179._lifecycle_flags(),
        }
    )


def _v177_transition_row() -> dict[str, object]:
    return _transition_row(
        branch_id="v177-branch",
        seed=41,
        forced_action="stay",
        value_offset=0.01,
        row_origin="v177_exact_branch_replay_from_v176_shard_plan",
    )


def _v178_support_ready_rows() -> list[dict[str, object]]:
    seeds = [101, 103, 107]
    return [
        _transition_row(
            branch_id=f"v179-branch-{index:03d}",
            seed=seeds[index % len(seeds)],
            forced_action=PUBLIC_ACTIONS[index % len(PUBLIC_ACTIONS)],
            value_offset=0.02 + index * 0.0001,
            row_origin="v179_exact_branch_replay_from_v172_support_expansion",
        )
        for index in range(v178.DEFAULT_MIN_ROW_COUNT)
    ]


def _transition_row(
    *,
    branch_id: str,
    seed: int,
    forced_action: str,
    value_offset: float,
    row_origin: str,
) -> dict[str, object]:
    branch_number = _branch_number(branch_id)
    current_observation = _encoded_observation(value_offset)
    next_observation = _encoded_observation(value_offset + 0.1)
    previous_observation = _encoded_observation(value_offset - 0.1)
    current_mask = {action: action in PUBLIC_ACTIONS for action in ACTION_NAMES}
    next_mask = {action: action in PUBLIC_ACTIONS for action in ACTION_NAMES}
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
        "row_origin": row_origin,
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
            "failure_types": ["v179_support_expansion"],
            "source_record_digest": f"source-{branch_id}",
            "materialized_record_digest": f"materialized-{branch_id}",
            "branch_state_digest": f"state-{branch_id}",
            "replay_verification_digest": f"digest-{branch_id}-{forced_action}",
            "source_identity_used_for_exact_materialization_only": True,
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
