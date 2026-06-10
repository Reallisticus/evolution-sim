from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY,
    SUPPORT_PROVENANCE_SEEDS,
)
from evolution_sim.mind.carrion_survivor_continuation_v174_mechanism_failure_battery import (
    EXPECTED_V173_EXACT_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v175_v174_route_correction_audit import (
    EXPECTED_V174_EXACT_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v176_transition_diagnostic_planner import (
    EXPECTED_V175_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v176_transition_diagnostic_planner,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV176TransitionDiagnosticPlannerTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v176-transition-diagnostic-planner"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v176_transition_diagnostic_planner"
            ),
        )

    def test_v175_digest_mismatch_closes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)

            report = run_carrion_survivor_continuation_v176_transition_diagnostic_planner(
                v175_report_path=paths["v175_report"],
                v176_plan_path=paths["v176_plan"],
                v174_report_path=paths["v174_report"],
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                output_path=paths["v176_report"],
                group_relative_output_path=paths["group_relative"],
                v177_shard_plan_output_path=paths["v177_shards"],
                expected_v175_exact_digest="wrong",
                expected_v176_plan_digest=stable_payload_digest(payloads["plan_rows"]),
                expected_v174_exact_digest=payloads["v174_report"]["exact_digest"],
                expected_v172_dataset_digest=stable_payload_digest(payloads["rows"]),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn("v175_unexpected_exact_digest", report["source_validation"]["failures"])
        self.assertEqual(
            report["classification"]["primary"],
            "v176_invalid_closed_no_training",
        )
        self.assertTrue(report["lanes"]["lane_a_failed_seed_replay_target_map"]["skipped"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["fit_ran"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["promotion_authorized"])

    def test_synthetic_exact_branch_plan_and_group_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)

            report = run_carrion_survivor_continuation_v176_transition_diagnostic_planner(
                v175_report_path=paths["v175_report"],
                v176_plan_path=paths["v176_plan"],
                v174_report_path=paths["v174_report"],
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                output_path=paths["v176_report"],
                group_relative_output_path=paths["group_relative"],
                v177_shard_plan_output_path=paths["v177_shards"],
                expected_v175_exact_digest=payloads["v175_report"]["exact_digest"],
                expected_v176_plan_digest=stable_payload_digest(payloads["plan_rows"]),
                expected_v174_exact_digest=payloads["v174_report"]["exact_digest"],
                expected_v172_dataset_digest=stable_payload_digest(payloads["rows"]),
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertEqual(
            report["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["classification"]["primary"],
            "v176_exact_branch_replay_plan_ready_no_training",
        )
        lane_a = report["lanes"]["lane_a_failed_seed_replay_target_map"]
        self.assertGreater(lane_a["unique_row_failure_counts_by_seed"]["19"], 0)
        self.assertGreater(lane_a["unique_row_failure_counts_by_seed"]["41"], 0)
        lane_b = report["lanes"]["lane_b_existing_transition_evidence_audit"]
        self.assertIn("next_public_observation", lane_b["missing_transition_fields"])
        lane_e = report["lanes"][
            "lane_e_training_free_group_relative_transition_experience_probe"
        ]
        self.assertGreater(lane_e["candidate_transition_experience_row_count"], 0)
        self.assertTrue(report["group_relative_transition_experience_output"]["written"])
        self.assertTrue(report["v177_shard_plan_output"]["written"])
        self.assertEqual(report["exact_digest"], _digest_without_exact(report))

    def test_cli_writes_report(self) -> None:
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
                    "mind_v3_carrion_survivor_continuation_v176_transition_diagnostic_planner",
                    "--v175-report",
                    str(paths["v175_report"]),
                    "--v176-plan",
                    str(paths["v176_plan"]),
                    "--v174-report",
                    str(paths["v174_report"]),
                    "--v172-report",
                    str(paths["v172_report"]),
                    "--v172-dataset",
                    str(paths["v172_dataset"]),
                    "--output",
                    str(paths["v176_report"]),
                    "--group-relative-output",
                    str(paths["group_relative"]),
                    "--v177-shard-plan-output",
                    str(paths["v177_shards"]),
                    "--expected-v175-exact-digest",
                    payloads["v175_report"]["exact_digest"],
                    "--expected-v176-plan-digest",
                    stable_payload_digest(payloads["plan_rows"]),
                    "--expected-v174-exact-digest",
                    payloads["v174_report"]["exact_digest"],
                    "--expected-v172-dataset-digest",
                    stable_payload_digest(payloads["rows"]),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v176_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v176_transition_diagnostic_planner_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_SCHEMA_VERSION,
        )
        self.assertIn("exact_digest", written)


def _write_inputs(tmpdir: str) -> tuple[dict[str, Path], dict[str, object]]:
    root = Path(tmpdir)
    paths = {
        "v172_report": root / "v172.json",
        "v172_dataset": root / "v172.jsonl",
        "v174_report": root / "v174.json",
        "v175_report": root / "v175.json",
        "v176_plan": root / "v176-plan.jsonl",
        "v176_report": root / "v176.json",
        "group_relative": root / "group-relative.jsonl",
        "v177_shards": root / "v177-shards.jsonl",
    }
    rows = []
    row_id = 0
    for seed in SUPPORT_PROVENANCE_SEEDS:
        failed_seed = seed in {19, 41}
        rows.append(
            _v172_row(
                seed=seed,
                row_id=row_id,
                x=0.0,
                safe_action="drink" if failed_seed else "eat",
            )
        )
        row_id += 1
        rows.append(
            _v172_row(
                seed=seed,
                row_id=row_id,
                x=1.0,
                safe_action="eat" if failed_seed else "drink",
            )
        )
        row_id += 1
    _write_jsonl(paths["v172_dataset"], rows)
    dataset_digest = stable_payload_digest(rows)
    v172_report = {"dataset": {"row_count": len(rows), "dataset_digest": dataset_digest}}
    _write_json(paths["v172_report"], v172_report)
    v174_report = _report(
        {
            "schema_version": "m3_carrion_survivor_continuation_v174_mechanism_failure_battery_report_v1",
            "policy": "diagnostics_only_m3_carrion_survivor_continuation_v174_mechanism_failure_battery_v1",
            "classification": {
                "primary": "v174_action_ranking_capacity_ready_for_v175_diagnostic_fit_no_runtime",
                "labels": [
                    "v174_action_ranking_capacity_ready_for_v175_diagnostic_fit_no_runtime"
                ],
            },
            "source_validation": {
                "passed": True,
                "observed_v173_exact_digest": EXPECTED_V173_EXACT_DIGEST,
                "observed_v172_dataset_digest": dataset_digest,
            },
            "dataset_digest": dataset_digest,
            **_lifecycle_flags(),
        }
    )
    _write_json(paths["v174_report"], v174_report)
    plan_rows = [
        {
            "schema_version": "m3_carrion_survivor_continuation_v176_plan_row_v1",
            "route": "exact_branch_replay_expansion",
            "priority": 1,
            "training_authorized": False,
            "runtime_artifact_authorized": False,
        },
        {
            "schema_version": "m3_carrion_survivor_continuation_v176_plan_row_v1",
            "route": "world_model_transition_diagnostic",
            "priority": 2,
            "training_authorized": False,
            "runtime_artifact_authorized": False,
        },
    ]
    _write_jsonl(paths["v176_plan"], plan_rows)
    v175_report = _report(
        {
            "schema_version": "m3_carrion_survivor_continuation_v175_v174_route_correction_audit_report_v1",
            "policy": "diagnostics_only_m3_carrion_survivor_continuation_v175_v174_route_correction_audit_v1",
            "classification": {
                "primary": EXPECTED_V175_CLASSIFICATION,
                "labels": [EXPECTED_V175_CLASSIFICATION],
            },
            "v176_plan_output": {
                "written": True,
                "jsonl_row_count": len(plan_rows),
                "plan_digest": stable_payload_digest(plan_rows),
            },
            "dataset_digest": dataset_digest,
            **_lifecycle_flags(),
        }
    )
    _write_json(paths["v175_report"], v175_report)
    return paths, {
        "rows": rows,
        "plan_rows": plan_rows,
        "v174_report": v174_report,
        "v175_report": v175_report,
    }


def _v172_row(
    *,
    seed: int,
    row_id: int,
    x: float,
    safe_action: str,
) -> dict[str, object]:
    public_actions = ["eat", "drink"]
    mask = {action: action in public_actions for action in ACTION_NAMES}
    targets = [
        {
            "action": action,
            "public_mask": mask[action],
            "target_available": mask[action],
            "replay_verified": mask[action],
            "safe_target": action == safe_action,
            "best_outcome_action": action == safe_action,
            "replay_outcome_summary": (
                [10, 5, 0, 1, 0.9, 0.8, 1.0, 0]
                if action == safe_action
                else [8, 4, -1, 1, 0.4, 0.2, 1.0, 0]
                if mask[action]
                else None
            ),
            "score_target": (
                1.0
                if action == safe_action
                else 0.0
                if mask[action]
                else None
            ),
            "value_target": (
                1.0
                if action == safe_action
                else 0.0
                if mask[action]
                else None
            ),
        }
        for action in ACTION_NAMES
    ]
    return {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ),
        "row_origin": "synthetic_v172_row",
        "feature_policy_id": (
            M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY
        ),
        "trainable_public_features": {
            "public_observation": {"x": x},
            "action_mask": mask,
        },
        "public_action_mask": mask,
        "action_value_targets": targets,
        "target_best_outcome_action_set": [safe_action],
        "safe_action_set": [safe_action],
        "target_outcome_summary": {
            "best_outcome_action_set": [safe_action],
            "best_outcome_action_count": 1,
        },
        "target_classification": "unique_replay_verified_best_action",
        "robust_winner_action": safe_action,
        "metadata": {
            "metadata_schema_version": (
                "m3_carrion_survivor_continuation_v172_replay_target_metadata_v1"
            ),
            "source": "synthetic_v172",
            "seed": seed,
            "branch_id": f"branch-{seed}-{row_id}",
            "branch_tick": 97 + row_id,
            "agent_id": row_id + 100,
            "source_path": f"source-{seed}.jsonl",
            "line_number": row_id + 1,
            "source_record_digest": f"source-digest-{row_id}",
            "materialized_record_digest": f"materialized-digest-{row_id}",
            "branch_state_digest": f"branch-state-{row_id}",
            "source_seed_is_support_provenance_not_future_promotion_holdout": True,
            "source_identity_used_as_trainable_input": False,
            "runtime_requested_or_resolved_action_used_as_trainable_input": False,
            "future_outcome_used_as_trainable_input": False,
            "target_safe_action_used_as_trainable_input": False,
            "private_state_used_as_trainable_input": False,
        },
    }


def _lifecycle_flags() -> dict[str, object]:
    return {
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
        "k_tuning_ran": False,
        "threshold_tuning_ran": False,
        "replay_viewer_schema_changed": False,
        "promotion_authorized": False,
        "staging_authorized": False,
        "commit_authorized": False,
        "reset_authorized": False,
        "clean_authorized": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


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
