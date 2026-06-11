from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind import (
    carrion_survivor_continuation_v181_v180_failure_response as v181,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV181V180FailureResponseTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v181-v180-failure-response"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v181_v180_failure_response"
            ),
        )

    def test_valid_v180_failure_report_produces_autopsy_without_training(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir)

            report = v181.run_carrion_survivor_continuation_v181_v180_failure_response(
                v180_report_path=paths["v180_report"],
                v180_artifact_path=paths["artifact"],
                transition_dataset_path=paths["dataset"],
                output_path=paths["root"] / "v181-report.json",
                expected_v180_report_exact_digest=paths["v180_exact_digest"],
                expected_v180_artifact_digest=paths["artifact_digest"],
                expected_dataset_digest=paths["dataset_digest"],
                run_trace_replay=False,
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["diagnostic_trace_replay_ran"])
        self.assertFalse(
            report["failure_mechanism"]["slice_2_training_consumed"]
        )
        self.assertEqual(
            report["classification"]["primary"],
            v181.V181_AUTOPSY_CLASSIFICATION,
        )
        self.assertIn(
            "artifact_utility_table_majority_imputed_actions",
            report["failure_mechanism"]["labels"],
        )
        self.assertIn(
            "broad_overrides_concentrated_on_eat",
            report["failure_mechanism"]["labels"],
        )
        self.assertIn(
            "carrion_fixture_no_complete_valid_action_support",
            report["failure_mechanism"]["labels"],
        )
        self.assertEqual(report["exact_digest"], v181._digest_without_exact(report))

    def test_v180_report_digest_mismatch_fails_closed_without_training(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir)

            report = v181.run_carrion_survivor_continuation_v181_v180_failure_response(
                v180_report_path=paths["v180_report"],
                v180_artifact_path=paths["artifact"],
                transition_dataset_path=paths["dataset"],
                output_path=paths["root"] / "v181-report.json",
                expected_v180_report_exact_digest="wrong",
                expected_v180_artifact_digest=paths["artifact_digest"],
                expected_dataset_digest=paths["dataset_digest"],
                run_trace_replay=False,
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v180_report_exact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertEqual(
            report["classification"]["primary"],
            v181.V181_SOURCE_INVALID_CLASSIFICATION,
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["diagnostic_trace_replay_ran"])
        self.assertFalse(report["promotion_authorized"])

    def test_artifact_digest_mismatch_fails_closed_without_trace_replay(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir)

            report = v181.run_carrion_survivor_continuation_v181_v180_failure_response(
                v180_report_path=paths["v180_report"],
                v180_artifact_path=paths["artifact"],
                transition_dataset_path=paths["dataset"],
                output_path=paths["root"] / "v181-report.json",
                expected_v180_report_exact_digest=paths["v180_exact_digest"],
                expected_v180_artifact_digest="wrong",
                expected_dataset_digest=paths["dataset_digest"],
                run_trace_replay=True,
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "artifact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertFalse(report["diagnostic_trace_replay_ran"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["promotion_authorized"])

    def test_cli_writes_parseable_report_and_prints_route_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir)
            output_path = paths["root"] / "v181-report.json"

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v181_v180_failure_response",
                    "--v180-report",
                    str(paths["v180_report"]),
                    "--v180-artifact",
                    str(paths["artifact"]),
                    "--transition-dataset",
                    str(paths["dataset"]),
                    "--output",
                    str(output_path),
                    "--expected-v180-report-exact-digest",
                    paths["v180_exact_digest"],
                    "--expected-v180-artifact-digest",
                    paths["artifact_digest"],
                    "--expected-dataset-digest",
                    paths["dataset_digest"],
                    "--skip-trace-replay",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("slice_2_training_consumed=False", result.stdout)
        self.assertIn("runtime_artifact_created=False", result.stdout)
        self.assertIn("promotion_authorized=False", result.stdout)
        self.assertEqual(
            report["classification"]["primary"],
            v181.V181_AUTOPSY_CLASSIFICATION,
        )

    def test_trace_aggregate_preserves_weighted_margin_and_utility_ranges(
        self,
    ) -> None:
        aggregate = v181._aggregate_trace_runs(
            [
                {
                    "trace": {
                        "transition_value_decision_count": 2,
                        "supported_score_count": 1,
                        "clear_best_count": 1,
                        "override_applied_count": 1,
                        "runtime_action_selection_changed_count": 1,
                        "missing_supported_score_count": 1,
                        "utility_margin": {
                            "count": 2,
                            "min": 0.1,
                            "max": 0.3,
                            "mean": 0.2,
                        },
                        "selected_utility": {
                            "count": 2,
                            "min": 1.0,
                            "max": 3.0,
                            "mean": 2.0,
                        },
                    }
                },
                {
                    "trace": {
                        "transition_value_decision_count": 1,
                        "supported_score_count": 1,
                        "clear_best_count": 1,
                        "override_applied_count": 1,
                        "runtime_action_selection_changed_count": 1,
                        "utility_margin": {
                            "count": 1,
                            "min": 0.5,
                            "max": 0.5,
                            "mean": 0.5,
                        },
                        "selected_utility": {
                            "count": 1,
                            "min": 4.0,
                            "max": 4.0,
                            "mean": 4.0,
                        },
                    }
                },
            ]
        )

        self.assertEqual(aggregate["utility_margin"]["count"], 3)
        self.assertEqual(aggregate["utility_margin"]["min"], 0.1)
        self.assertEqual(aggregate["utility_margin"]["max"], 0.5)
        self.assertEqual(aggregate["utility_margin"]["mean"], 0.3)
        self.assertEqual(aggregate["selected_utility"]["min"], 1.0)
        self.assertEqual(aggregate["selected_utility"]["max"], 4.0)
        self.assertEqual(aggregate["selected_utility"]["mean"], 2.666667)


def _write_inputs(tmpdir: str) -> dict[str, object]:
    root = Path(tmpdir)
    dataset_rows = [{"row_id": "transition-row-1"}]
    dataset_path = root / "dataset.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dataset_rows),
        encoding="utf-8",
    )
    dataset_digest = stable_payload_digest(dataset_rows)
    artifact = _artifact()
    artifact_path = root / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_digest = stable_payload_digest(artifact)
    report = _v180_failure_report(
        artifact_digest=artifact_digest,
        dataset_digest=dataset_digest,
    )
    report["exact_digest"] = v181._digest_without_exact(report)
    v180_report_path = root / "v180-report.json"
    v180_report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return {
        "root": root,
        "dataset": dataset_path,
        "dataset_digest": dataset_digest,
        "artifact": artifact_path,
        "artifact_digest": artifact_digest,
        "v180_report": v180_report_path,
        "v180_exact_digest": report["exact_digest"],
    }


def _artifact() -> dict[str, object]:
    action_stats = {}
    for action in ACTION_NAMES:
        if action == "eat":
            action_stats[action] = {
                "count": 2,
                "utility_mean": 1.25,
                "component_means": {"target_terminal_alive": 0.5},
            }
        else:
            action_stats[action] = {
                "count": 1,
                "utility_mean": -5.0,
                "component_means": {"imputed_unobserved_action": 1.0},
            }
    return {
        "schema_version": "mind_v3_v142_public_transition_value_scorer_v1",
        "artifact_policy": "opt_in_m3_test_v180_public_transition_row_policy_v1",
        "model_id": "public_transition_value_utility_lookup",
        "diagnostics_only": False,
        "explicit_opt_in_required": True,
        "runtime_action_selection_authorized": False,
        "runtime_policy_change_requires_explicit_flag": True,
        "promotion_authorized": False,
        "built_from": {
            "source": "test",
            "training_row_count": 1,
            "dataset_digest": "filled-by-report",
        },
        "utility_tables": {
            "policy": "test",
            "feature_action_utility": {
                "mask=eat:1|self=test": action_stats,
            },
        },
    }


def _v180_failure_report(
    *,
    artifact_digest: str,
    dataset_digest: str,
) -> dict[str, object]:
    broad_diag = {
        "decision_count": 10,
        "supported_score_count": 2,
        "supported_score_share": 0.2,
        "clear_best_count": 2,
        "override_applied_count": 2,
        "missing_supported_score_count": 8,
        "missing_supported_score_share": 0.8,
        "no_prediction_count": 8,
        "predicted_action_counts": {"eat": 9, "drink": 1},
        "score_source_counts": {
            "feature_action_utility": 2,
            "missing_supported_scores_for_valid_actions": 8,
        },
        "override_rejected_reason_counts": {
            "missing_supported_scores_for_valid_actions": 8,
        },
    }
    carrion_diag = {
        "decision_count": 12,
        "supported_score_count": 0,
        "supported_score_share": 0.0,
        "override_applied_count": 0,
        "missing_supported_score_count": 12,
        "missing_supported_score_share": 1.0,
        "no_prediction_count": 12,
        "predicted_action_counts": {},
        "score_source_counts": {
            "missing_supported_scores_for_valid_actions": 12,
        },
        "override_rejected_reason_counts": {
            "missing_supported_scores_for_valid_actions": 12,
        },
    }
    return {
        "classification": {"primary": v181.EXPECTED_V180_CLASSIFICATION},
        "artifact": {"created": True, "digest": artifact_digest},
        "dataset": {"dataset_digest": dataset_digest, "row_count": 1},
        "source_validation": {"passed": True},
        "authorization_report_validation": {"passed": True},
        "training_ran": True,
        "training_artifact_created": True,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "acceptance": {
            "passed": False,
            "blockers": [
                {"reason": "carrion_only_terminal_survivors_zero"},
                {"reason": "per_seed_alive_or_birth_regression"},
            ],
        },
        "evaluation": {
            "ran": True,
            "broad": {
                "baseline": {
                    "aggregate": {
                        "alive_agents_mean": 13.0,
                        "births_mean": 12.0,
                        "dominant_requested_action_share": 0.41,
                        "heuristic_action_source_count": 0,
                    }
                },
                "candidate": {
                    "aggregate": {
                        "alive_agents_mean": 8.0,
                        "births_mean": 8.0,
                        "dominant_requested_action_share": 0.48,
                        "heuristic_action_source_count": 0,
                        "transition_value_scorer_diagnostics": broad_diag,
                    },
                    "runs": [],
                },
            },
            "controlled_fixture": {
                "baseline": {
                    "aggregate": {
                        "alive_agents_mean": 0.0,
                        "births_mean": 2.0,
                        "dominant_requested_action_share": 0.32,
                        "heuristic_action_source_count": 0,
                        "outcome_metrics": {
                            "total_terminal_alive_agents": 0,
                            "terminal_survivor_run_count": 0,
                        },
                    }
                },
                "candidate": {
                    "aggregate": {
                        "alive_agents_mean": 0.0,
                        "births_mean": 2.0,
                        "dominant_requested_action_share": 0.32,
                        "heuristic_action_source_count": 0,
                        "transition_value_scorer_diagnostics": carrion_diag,
                        "outcome_metrics": {
                            "total_terminal_alive_agents": 0,
                            "terminal_survivor_run_count": 0,
                        },
                    },
                    "runs": [],
                },
            },
            "per_seed_alive_birth_deltas": [
                {
                    "suite": "broad",
                    "seed": 5,
                    "alive_agents_delta": -2,
                    "births_delta": -2,
                },
                {
                    "suite": "carrion_only",
                    "seed": 13,
                    "alive_agents_delta": 0,
                    "births_delta": 0,
                },
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
