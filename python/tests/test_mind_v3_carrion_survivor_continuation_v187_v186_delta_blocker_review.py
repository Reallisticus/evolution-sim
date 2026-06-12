from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v186_transition_row_policy_training as v186,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v187_v186_delta_blocker_review as v187,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV187V186DeltaBlockerReviewTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v187-v186-delta-blocker-review"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v187_v186_delta_blocker_review"
            ),
        )

    def test_valid_v186_delta_recommends_terminal_survival_support_route(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, artifact_digest = _write_valid_v186_inputs(tmpdir)
            report = v187.run_carrion_survivor_continuation_v187_v186_delta_blocker_review(
                v186_report_path=paths["v186_report"],
                v186_artifact_path=paths["v186_artifact"],
                output_path=paths["v187_report"],
                expected_v186_report_exact_digest=report_digest,
                expected_v186_artifact_digest=artifact_digest,
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["delta_validation"]["passed"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v187.RECOMMENDED_NEXT_ROUTE,
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v187_v186_delta_blocker_"
                "review_terminal_survival_support_generation_before_slice_3_"
                "no_training"
            ),
        )
        self.assertEqual(
            report["v186_delta"]["dominant_requested_action_share"], 0.4179
        )
        self.assertEqual(report["v186_delta"]["heuristic_action_source_count"], 0)
        self.assertEqual(report["v186_delta"]["carrion_only_terminal_survivors"], 0)
        self.assertEqual(report["v186_delta"]["carrion_fixture_births_mean"], 2.8333)
        self.assertEqual(report["v186_delta"]["broad_alive_birth_regressions"], [])
        self.assertTrue(
            all(
                finding["status"] == "inherited_not_rediscovered"
                and finding["rerun"] is False
                for finding in report["inherited_prior_findings"]
            )
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["slice_3_training_consumed"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["gate_relaxation_allowed"])
        self.assertFalse(report["v180_rerun"])
        self.assertFalse(report["v186_rerun"])
        self.assertFalse(report["support_expansion_ran"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_v186_report_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _report_digest, artifact_digest = _write_valid_v186_inputs(tmpdir)
            report = v187.run_carrion_survivor_continuation_v187_v186_delta_blocker_review(
                v186_report_path=paths["v186_report"],
                v186_artifact_path=paths["v186_artifact"],
                output_path=paths["v187_report"],
                expected_v186_report_exact_digest="wrong",
                expected_v186_artifact_digest=artifact_digest,
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v186_report_exact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertEqual(report["route_decision"]["recommended_next_route"], "stop")
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_3_training_consumed"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])

    def test_v186_artifact_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, _artifact_digest = _write_valid_v186_inputs(tmpdir)
            report = v187.run_carrion_survivor_continuation_v187_v186_delta_blocker_review(
                v186_report_path=paths["v186_report"],
                v186_artifact_path=paths["v186_artifact"],
                output_path=paths["v187_report"],
                expected_v186_report_exact_digest=report_digest,
                expected_v186_artifact_digest="wrong",
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v186_artifact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertEqual(report["route_decision"]["recommended_next_route"], "stop")
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["slice_3_training_consumed"])

    def test_unexpected_v186_delta_fails_closed_without_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _report_digest, artifact_digest = _write_valid_v186_inputs(tmpdir)
            report_payload = json.loads(paths["v186_report"].read_text(encoding="utf-8"))
            report_payload["acceptance"] = dict(report_payload["acceptance"])
            report_payload["acceptance"]["controlled_fixture"] = dict(
                report_payload["acceptance"]["controlled_fixture"]
            )
            report_payload["acceptance"]["controlled_fixture"][
                "total_terminal_alive_agents"
            ] = 1
            report_payload["acceptance"]["passed"] = True
            report_digest = _write_report(paths["v186_report"], report_payload)

            report = v187.run_carrion_survivor_continuation_v187_v186_delta_blocker_review(
                v186_report_path=paths["v186_report"],
                v186_artifact_path=paths["v186_artifact"],
                output_path=paths["v187_report"],
                expected_v186_report_exact_digest=report_digest,
                expected_v186_artifact_digest=artifact_digest,
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertFalse(report["delta_validation"]["passed"])
        self.assertIn(
            "carrion_only_terminal_survivors_matches_expected",
            report["delta_validation"]["failures"],
        )
        self.assertEqual(report["route_decision"]["recommended_next_route"], "stop")
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v187_v186_delta_blocker_"
                "review_unexpected_delta_closed_no_training"
            ),
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_3_training_consumed"])

    def test_cli_writes_report_and_prints_delta_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, artifact_digest = _write_valid_v186_inputs(tmpdir)

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v187_v186_delta_blocker_review",
                    "--v186-report",
                    str(paths["v186_report"]),
                    "--v186-artifact",
                    str(paths["v186_artifact"]),
                    "--output",
                    str(paths["v187_report"]),
                    "--expected-v186-report-exact-digest",
                    report_digest,
                    "--expected-v186-artifact-digest",
                    artifact_digest,
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(paths["v187_report"].read_text(encoding="utf-8"))

        self.assertIn(f"recommended_next_route={v187.RECOMMENDED_NEXT_ROUTE}", result.stdout)
        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("slice_3_training_consumed=False", result.stdout)
        self.assertIn("runtime_action_selection_changed=False", result.stdout)
        self.assertIn("promotion_authorized=False", result.stdout)
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v187.RECOMMENDED_NEXT_ROUTE,
        )
        self.assertTrue(exact_digest_validation_report(report)["passed"])


def _write_valid_v186_inputs(tmpdir: str) -> tuple[dict[str, Path], str, str]:
    root = Path(tmpdir)
    paths = {
        "v186_report": root / "v186-report.json",
        "v186_artifact": root / "v186-artifact.json",
        "v187_report": root / "v187-report.json",
    }
    artifact = _valid_v186_artifact()
    artifact_digest = stable_payload_digest(artifact)
    _write_json(paths["v186_artifact"], artifact)
    report = _valid_v186_report(artifact_digest)
    report_digest = _write_report(paths["v186_report"], report)
    return paths, report_digest, artifact_digest


def _valid_v186_artifact() -> dict[str, object]:
    return {
        "schema_version": v186.MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
        "artifact_policy": v186.M3_CARRION_SURVIVOR_CONTINUATION_V186_ARTIFACT_POLICY,
        "model_id": "public_transition_value_utility_lookup",
        "runtime_action_selection_authorized": False,
        "promotion_authorized": False,
        "built_from": {
            "source": "v185_repaired_transition_rows",
            "source_producer": v186.EXPECTED_V185_SOURCE_PRODUCER,
            "dataset_digest": v186.EXPECTED_V185_REPAIRED_TRANSITION_DATASET_DIGEST,
            "authorization_report_exact_digest": (
                v186.EXPECTED_V185_REPAIRED_AUDIT_EXACT_DIGEST
            ),
            "authorization_route": v186.EXPECTED_V186_TRAINING_ROUTE,
            "training_row_count": 137,
            "training_slice_index": 2,
            "previous_training_slices_consumed": 1,
            "campaign_slice_cap": 10,
            "first_opt_in_training_slice": False,
            "slice_2_opt_in_training_slice": True,
            "runtime_action_selection_authorized": False,
            "promotion_authorized": False,
        },
    }


def _valid_v186_report(artifact_digest: str) -> dict[str, object]:
    return {
        "schema_version": (
            v186.M3_CARRION_SURVIVOR_CONTINUATION_V186_TRANSITION_ROW_POLICY_TRAINING_SCHEMA_VERSION
        ),
        "policy": v186.M3_CARRION_SURVIVOR_CONTINUATION_V186_TRANSITION_ROW_POLICY_TRAINING_POLICY,
        "contract": {
            "training_route_required": v186.EXPECTED_V186_TRAINING_ROUTE,
            "source_producer_required": v186.EXPECTED_V185_SOURCE_PRODUCER,
            "training_slice_index": 2,
            "first_opt_in_training_slice": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "promotion_authorized": False,
            "gate_relaxation_allowed": False,
        },
        "inputs": {
            "expected_training_route": v186.EXPECTED_V186_TRAINING_ROUTE,
            "expected_source_producer": v186.EXPECTED_V185_SOURCE_PRODUCER,
        },
        "source_validation": {
            "passed": True,
            "required_training_route": v186.EXPECTED_V186_TRAINING_ROUTE,
        },
        "artifact": {
            "created": True,
            "digest": artifact_digest,
            "artifact_policy": (
                v186.M3_CARRION_SURVIVOR_CONTINUATION_V186_ARTIFACT_POLICY
            ),
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "promotion_authorized": False,
            "training_slice_index": 2,
        },
        "training": {
            "ran": True,
            "passed": True,
            "training_slice_index": 2,
            "first_opt_in_training_slice": False,
            "slice_2_opt_in_training_slice": True,
            "authorization_route": v186.EXPECTED_V186_TRAINING_ROUTE,
            "source_producer": v186.EXPECTED_V185_SOURCE_PRODUCER,
        },
        "acceptance": {
            "policy": "m3_carrion_survivor_continuation_v186_acceptance_v1",
            "passed": False,
            "blocker_count": 1,
            "blockers": [
                {
                    "reason": "carrion_only_terminal_survivors_zero",
                    "observed": 0,
                    "required": ">0",
                }
            ],
            "controlled_fixture": {
                "fixture": "carrion_only",
                "total_terminal_alive_agents": 0,
                "terminal_survivor_run_count": 0,
                "alive_agents_mean": 0.0,
                "births_mean": 2.8333,
            },
            "dominant_requested_action_share": 0.4179,
            "heuristic_action_source_count": 0,
            "per_seed_alive_birth_regressions": [],
            "runtime_integration_authorized": False,
            "promotion_evidence": False,
            "shadow_acceptance_only": True,
            "training_slice_index": 2,
        },
        "training_slice_budget": {
            "campaign": "carrion_transition_row_policy",
            "slice_cap": 10,
            "previous_slices_consumed": 1,
            "this_slice_index": 2,
            "this_slice_consumed": True,
            "current_slices_consumed": 2,
        },
        "classification": {
            "primary": v187.EXPECTED_V186_CLASSIFICATION,
            "labels": [v187.EXPECTED_V186_CLASSIFICATION],
        },
        "training_ran": True,
        "training_artifact_created": True,
        "fit_ran": True,
        "slice_2_training_consumed": True,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": True,
        "live_ab_ran": False,
        "promotion_authorized": False,
        "gate_relaxation_ran": False,
        "non_promoted": True,
    }


def _write_report(path: Path, report: dict[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    report["exact_digest"] = stable_payload_digest(payload)
    _write_json(path, report)
    return str(report["exact_digest"])


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
