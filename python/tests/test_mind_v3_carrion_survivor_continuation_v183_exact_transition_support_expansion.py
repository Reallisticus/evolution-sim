from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind import (
    carrion_survivor_continuation_v181_v180_failure_response as v181,
    carrion_survivor_continuation_v182_imputed_abstention_design as v182,
    carrion_survivor_continuation_v183_exact_transition_support_expansion as v183,
)
from evolution_sim.mind.provenance import stable_payload_digest
from tests.test_mind_v3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion import (
    _v178_support_ready_rows,
)
from tests.test_mind_v3_carrion_survivor_continuation_v181_v180_failure_response import (
    _artifact,
    _v180_failure_report,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV183ExactTransitionSupportExpansionTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v183-exact-transition-support-expansion"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v183_exact_transition_support_expansion"
            ),
        )

    def test_v182_digest_mismatch_fails_closed_without_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir)
            kwargs = _run_kwargs(paths)
            kwargs["expected_v182_report_exact_digest"] = "wrong"

            report = v183.run_carrion_survivor_continuation_v183_exact_transition_support_expansion(
                **kwargs,
                generate_source_trajectories=False,
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v182_exact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertEqual(report["dataset"]["row_count"], 0)
        self.assertEqual(
            report["classification"]["primary"],
            v183.V183_SOURCE_INVALID_CLASSIFICATION,
        )
        self._assert_training_closed(report)

    def test_plan_uses_observed_support_fields_not_supported_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            carrion_path = root / "carrion.jsonl"
            broad_path = root / "broad.jsonl"
            _write_trajectory(
                carrion_path,
                [
                    _record(
                        seed=13,
                        tick=0,
                        agent_id=7,
                        reason="missing_supported_scores_for_valid_actions",
                        source="missing_supported_scores_for_valid_actions",
                        supported_scores=False,
                    )
                ],
            )
            _write_trajectory(
                broad_path,
                [
                    _record(
                        seed=19,
                        tick=1,
                        agent_id=3,
                        reason="low_observed_support_for_valid_actions",
                        source="feature_action_utility",
                        supported_scores=True,
                    )
                ],
            )

            plan_rows, plan = v183.build_v183_plan_rows(
                [
                    {
                        "path": str(carrion_path),
                        "suite": "carrion_observed_support_zero",
                        "fixture": "carrion_only",
                        "seed": 13,
                    },
                    {
                        "path": str(broad_path),
                        "suite": "broad_seed_19_regression",
                        "fixture": "broad",
                        "seed": 19,
                    },
                ],
                carrion_branches_per_seed=1,
                broad_branches_per_seed=1,
                max_forced_actions_per_branch=6,
                ticks=120,
            )

        self.assertTrue(plan["passed"])
        self.assertEqual(plan["selected_plan_row_count"], 2)
        self.assertEqual(
            {row["suite"] for row in plan_rows},
            {"carrion_observed_support_zero", "broad_seed_19_regression"},
        )
        broad = next(row for row in plan_rows if row["suite"] == "broad_seed_19_regression")
        self.assertTrue(
            broad["support_hole_diagnostics"][
                "transition_value_supported_scores_for_all_valid_actions"
            ]
        )
        self.assertFalse(
            broad["support_hole_diagnostics"][
                "transition_value_observed_support_floor_satisfied_for_all_valid_actions"
            ]
        )

    def test_ready_expansion_report_stays_diagnostics_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir)
            rows = _v183_ready_rows()
            plan_rows = [
                {"suite": "carrion_observed_support_zero"},
                {"suite": "broad_seed_19_regression"},
            ]
            with (
                mock.patch.object(
                    v183,
                    "build_v183_plan_rows",
                    return_value=(
                        plan_rows,
                        {
                            "passed": True,
                            "selected_plan_row_count": len(plan_rows),
                            "selected_by_suite": {
                                "carrion_observed_support_zero": 1,
                                "broad_seed_19_regression": 1,
                            },
                        },
                    ),
                ),
                mock.patch.object(
                    v183,
                    "build_selected_branch_points_v183",
                    return_value=(
                        [],
                        {},
                        {
                            "passed": True,
                            "selected_branch_point_count": len(plan_rows),
                        },
                    ),
                ),
                mock.patch.object(
                    v183,
                    "materialize_selected_branch_points_v183",
                    return_value=(
                        [],
                        {
                            "passed": True,
                            "materialized_branch_point_count": len(plan_rows),
                        },
                    ),
                ),
                mock.patch.object(
                    v183,
                    "build_compact_transition_rows_v183",
                    return_value=rows,
                ),
            ):
                report = v183.run_carrion_survivor_continuation_v183_exact_transition_support_expansion(
                    **_run_kwargs(paths),
                    generate_source_trajectories=False,
                    source_trajectory_specs=[
                        {"path": "synthetic", "suite": "carrion_observed_support_zero"},
                    ],
                )

            written_rows = [
                json.loads(line)
                for line in Path(paths["expanded_dataset"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(report["exact_digest"], v183._json_round_trip_digest(report))
            self.assertEqual(report["dataset"]["row_count"], len(rows))
            self.assertEqual(written_rows, v183._tag_v183_rows(rows))
            self.assertEqual(
                report["classification"]["primary"],
                v183.V183_EXPANSION_READY_FOR_V178_AUDIT_CLASSIFICATION,
            )
            self.assertEqual(
                report["route_recommendation"]["recommended_next_route"],
                "fresh_v178_style_transition_row_dataset_audit_before_any_slice_2_training",
            )
            self.assertFalse(report["route_recommendation"]["slice_2_training_authorized"])
            self.assertTrue(
                report["target_support_delta"]["uses_v182_observed_imputed_support_fields"]
            )
            self.assertFalse(
                report["target_support_delta"][
                    "legacy_supported_prediction_treated_as_strict_observed_support"
                ]
            )
            self._assert_training_closed(report)

    def test_cli_writes_report_and_prints_lifecycle_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir)

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v183_exact_transition_support_expansion",
                    "--v182-report",
                    str(paths["v182_report"]),
                    "--v181-report",
                    str(paths["v181_report"]),
                    "--v180-report",
                    str(paths["v180_report"]),
                    "--v180-artifact",
                    str(paths["artifact"]),
                    "--v179-report",
                    str(paths["v179_report"]),
                    "--v179-transition-dataset",
                    str(paths["v179_dataset"]),
                    "--output",
                    str(paths["report"]),
                    "--expanded-transition-dataset-output",
                    str(paths["expanded_dataset"]),
                    "--expected-v182-report-exact-digest",
                    paths["v182_exact_digest"],
                    "--expected-v181-report-exact-digest",
                    paths["v181_exact_digest"],
                    "--expected-v180-report-exact-digest",
                    paths["v180_exact_digest"],
                    "--expected-v180-artifact-digest",
                    paths["artifact_digest"],
                    "--expected-v179-report-exact-digest",
                    paths["v179_exact_digest"],
                    "--expected-v179-dataset-digest",
                    paths["v179_dataset_digest"],
                    "--skip-source-generation",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))

        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("training_artifact_created=False", result.stdout)
        self.assertIn("runtime_artifact_created=False", result.stdout)
        self.assertIn("runtime_action_selection_changed=False", result.stdout)
        self.assertIn("promotion_authorized=False", result.stdout)
        self.assertIn("slice_2_training_consumed=False", result.stdout)
        self.assertTrue(report["source_validation"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            v183.V183_EXPANSION_INSUFFICIENT_CLASSIFICATION,
        )
        self._assert_training_closed(report)

    def _assert_training_closed(self, report: dict[str, object]) -> None:
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["slice_2_training_consumed"])


def _run_kwargs(paths: dict[str, object]) -> dict[str, object]:
    return {
        "v182_report_path": paths["v182_report"],
        "v181_report_path": paths["v181_report"],
        "v180_report_path": paths["v180_report"],
        "v180_artifact_path": paths["artifact"],
        "v179_report_path": paths["v179_report"],
        "v179_transition_dataset_path": paths["v179_dataset"],
        "output_path": paths["report"],
        "expanded_transition_dataset_output_path": paths["expanded_dataset"],
        "expected_v182_report_exact_digest": paths["v182_exact_digest"],
        "expected_v181_report_exact_digest": paths["v181_exact_digest"],
        "expected_v180_report_exact_digest": paths["v180_exact_digest"],
        "expected_v180_artifact_digest": paths["artifact_digest"],
        "expected_v179_report_exact_digest": paths["v179_exact_digest"],
        "expected_v179_dataset_digest": paths["v179_dataset_digest"],
    }


def _write_inputs(tmpdir: str) -> dict[str, object]:
    root = Path(tmpdir)
    v179_rows = _v183_ready_rows()
    v179_dataset = root / "v179-dataset.jsonl"
    _write_jsonl(v179_dataset, v179_rows)
    v179_dataset_digest = stable_payload_digest(v179_rows)
    artifact = _artifact()
    artifact_path = root / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_digest = stable_payload_digest(artifact)
    v179_report = _v179_report(v179_rows, dataset_path=v179_dataset)
    v179_report_path = root / "v179-report.json"
    _write_report(v179_report_path, v179_report)
    v180_report = _v180_failure_report(
        artifact_digest=artifact_digest,
        dataset_digest=v179_dataset_digest,
    )
    v180_report["artifact"]["digest"] = artifact_digest
    v180_report["dataset"]["dataset_digest"] = v179_dataset_digest
    v180_report["exact_digest"] = _digest_without_exact(v180_report)
    v180_report_path = root / "v180-report.json"
    v180_report_path.write_text(
        json.dumps(v180_report, sort_keys=True),
        encoding="utf-8",
    )
    v181_report = _v181_report(v180_report, artifact_digest, v179_dataset_digest)
    v181_report_path = root / "v181-report.json"
    _write_report(v181_report_path, v181_report)
    v182_report = _v182_report(v181_report, v180_report, artifact_digest, v179_dataset_digest)
    v182_report_path = root / "v182-report.json"
    _write_report(v182_report_path, v182_report)
    return {
        "root": root,
        "artifact": artifact_path,
        "artifact_digest": artifact_digest,
        "v179_dataset": v179_dataset,
        "v179_dataset_digest": v179_dataset_digest,
        "v179_report": v179_report_path,
        "v179_exact_digest": v179_report["exact_digest"],
        "v180_report": v180_report_path,
        "v180_exact_digest": v180_report["exact_digest"],
        "v181_report": v181_report_path,
        "v181_exact_digest": v181_report["exact_digest"],
        "v182_report": v182_report_path,
        "v182_exact_digest": v182_report["exact_digest"],
        "report": root / "v183-report.json",
        "expanded_dataset": root / "v183-dataset.jsonl",
    }


def _v183_ready_rows() -> list[dict[str, object]]:
    rows = []
    for index, row in enumerate(_v178_support_ready_rows()):
        payload = json.loads(json.dumps(row, sort_keys=True))
        metadata = dict(payload["metadata"])
        if index % 2 == 0:
            metadata["failure_types"] = ["v183_carrion_observed_support_zero"]
            metadata["fixture"] = "carrion_only"
        else:
            metadata["failure_types"] = [
                "v183_broad_seed_19_regression_state",
                "v183_low_observed_support_for_valid_actions",
            ]
            metadata["fixture"] = "broad"
        payload["metadata"] = metadata
        rows.append(payload)
    return rows


def _v179_report(rows: list[dict[str, object]], *, dataset_path: Path) -> dict[str, object]:
    report = {
        "schema_version": (
            "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion_report_v1"
        ),
        "policy": (
            "diagnostics_only_m3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion_v1"
        ),
        "source_validation": {"passed": True},
        "support_summary": {"passed": True},
        "classification": {"primary": v183.V179_SUPPORT_READY_CLASSIFICATION},
        "dataset": {
            "path": str(dataset_path),
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
    report["exact_digest"] = _digest_without_exact(report)
    return report


def _v181_report(
    v180_report: dict[str, object],
    artifact_digest: str,
    dataset_digest: str,
) -> dict[str, object]:
    report = {
        "classification": {"primary": v181.V181_AUTOPSY_CLASSIFICATION},
        "source_validation": {
            "passed": True,
            "observed_v180_report_exact_digest": v180_report["exact_digest"],
            "observed_v180_artifact_digest": artifact_digest,
            "observed_dataset_digest": dataset_digest,
        },
        "failure_mechanism": {"slice_2_training_consumed": False},
        "training_ran": False,
        "training_artifact_created": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
    }
    report["exact_digest"] = _digest_without_exact(report)
    return report


def _v182_report(
    v181_report: dict[str, object],
    v180_report: dict[str, object],
    artifact_digest: str,
    dataset_digest: str,
) -> dict[str, object]:
    report = {
        "schema_version": (
            v182.M3_CARRION_SURVIVOR_CONTINUATION_V182_IMPUTED_ABSTENTION_SCHEMA_VERSION
        ),
        "policy": v182.M3_CARRION_SURVIVOR_CONTINUATION_V182_IMPUTED_ABSTENTION_POLICY,
        "classification": {"primary": v182.V182_EXACT_SUPPORT_EXPANSION_CLASSIFICATION},
        "source_validation": {
            "passed": True,
            "observed_v181_report_exact_digest": v181_report["exact_digest"],
            "observed_v180_report_exact_digest": v180_report["exact_digest"],
            "observed_v180_artifact_digest": artifact_digest,
            "observed_dataset_digest": dataset_digest,
        },
        "route": {
            "next_route": "exact_transition_support_expansion_before_any_slice_2_training"
        },
        "design_diagnostics": {
            "broad": {
                "transition_value_scorer_diagnostics": {
                    "override_applied_count": 122,
                }
            },
            "controlled_fixture": {
                "transition_value_scorer_diagnostics": {
                    "observed_support_floor_satisfied_count": 0,
                    "decision_count": 2122,
                }
            },
        },
        "shadow_evaluation": {
            "broad": {
                "candidate": {
                    "runs": [
                        {
                            "seed": 19,
                            "transition_value_scorer_diagnostics": {
                                "observed_support_floor_satisfied_count": 18,
                                "decision_count": 2143,
                            },
                        }
                    ]
                }
            },
            "per_seed_alive_birth_deltas": [
                {
                    "suite": "broad",
                    "seed": 19,
                    "alive_agents_delta": -1,
                    "births_delta": -2,
                }
            ],
        },
        "training_ran": False,
        "training_artifact_created": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "slice_2_training_consumed": False,
    }
    report["exact_digest"] = _digest_without_exact(report)
    return report


def _record(
    *,
    seed: int,
    tick: int,
    agent_id: int,
    reason: str,
    source: str,
    supported_scores: bool,
) -> dict[str, object]:
    mask = {action: action in ACTION_NAMES[:6] for action in ACTION_NAMES}
    return {
        "seed": seed,
        "tick": tick,
        "agent_id": agent_id,
        "requested_action": "eat",
        "resolved_action": "eat",
        "observation_input": {"schema_version": "test", "data": [seed, tick, agent_id]},
        "action_mask": mask,
        "policy_decision_diagnostics": {
            "transition_value_score_source": source,
            "transition_value_override_rejected_reason": reason,
            "transition_value_observed_support_floor_satisfied_for_all_valid_actions": False,
            "transition_value_observed_scores_for_all_valid_actions": False,
            "transition_value_supported_scores_for_all_valid_actions": supported_scores,
            "transition_value_valid_action_observed_support_floor": 2,
        },
    }


def _write_trajectory(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"header": True}, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps({"record": record}, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


def _digest_without_exact(report: dict[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


if __name__ == "__main__":
    unittest.main()
