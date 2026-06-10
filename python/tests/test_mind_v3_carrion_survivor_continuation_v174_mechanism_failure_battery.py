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
)
from evolution_sim.mind.carrion_survivor_continuation_v173_source_split_scorer import (
    M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v174_mechanism_failure_battery import (
    EXPECTED_V173_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v174_mechanism_failure_battery,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV174MechanismFailureBatteryTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v174-mechanism-failure-battery"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v174_mechanism_failure_battery"
            ),
        )

    def test_v173_digest_mismatch_closes_invalid_without_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)

            report = run_carrion_survivor_continuation_v174_mechanism_failure_battery(
                v173_report_path=paths["v173_report"],
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                output_path=paths["v174_report"],
                v175_plan_output_path=paths["v175_plan"],
                expected_v173_exact_digest="wrong",
                expected_v172_dataset_digest=stable_payload_digest(payloads["rows"]),
                support_provenance_seeds=(5, 13),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v173_unexpected_exact_digest",
            report["source_validation"]["failures"],
        )
        self.assertEqual(
            report["classification"]["primary"],
            "v174_invalid_closed_no_shadow",
        )
        self.assertTrue(
            report["lanes"]["lane_a_v173_score_mechanics_autopsy"]["skipped"]
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["shadow_eval_ran"])
        self.assertFalse(report["promotion_authorized"])

    def test_synthetic_battery_reports_mechanisms_and_local_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)

            report = run_carrion_survivor_continuation_v174_mechanism_failure_battery(
                v173_report_path=paths["v173_report"],
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                output_path=paths["v174_report"],
                v175_plan_output_path=paths["v175_plan"],
                expected_v173_exact_digest=payloads["v173_report"]["exact_digest"],
                expected_v172_dataset_digest=stable_payload_digest(payloads["rows"]),
                support_provenance_seeds=(5, 13),
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertEqual(
            report["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_SCHEMA_VERSION,
        )
        lane_a = report["lanes"]["lane_a_v173_score_mechanics_autopsy"]
        self.assertEqual(lane_a["full_public_mask_set_share"], 1.0)
        self.assertTrue(
            lane_a["full_set_positive_rule_proof"][
                "full_set_behavior_explained_by_positive_score_rule"
            ]
        )
        lane_b = report["lanes"]["lane_b_local_support_frontier"]
        self.assertTrue(lane_b["floor_passed"])
        self.assertEqual(
            lane_b["best_frontier_entry"]["singleton_hit_rate"],
            1.0,
        )
        self.assertEqual(
            report["classification"]["primary"],
            "v174_local_support_ready_for_shadow_diagnostic_no_runtime",
        )
        self.assertFalse(report["v175_plan_output"]["written"])
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
                    "mind_v3_carrion_survivor_continuation_v174_mechanism_failure_battery",
                    "--v173-report",
                    str(paths["v173_report"]),
                    "--v172-report",
                    str(paths["v172_report"]),
                    "--v172-dataset",
                    str(paths["v172_dataset"]),
                    "--output",
                    str(paths["v174_report"]),
                    "--v175-plan-output",
                    str(paths["v175_plan"]),
                    "--expected-v173-exact-digest",
                    payloads["v173_report"]["exact_digest"],
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
            written = json.loads(paths["v174_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v174_mechanism_failure_battery_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V174_MECHANISM_FAILURE_BATTERY_SCHEMA_VERSION,
        )
        self.assertIn("exact_digest", written)


def _write_inputs(tmpdir: str) -> tuple[dict[str, Path], dict[str, object]]:
    root = Path(tmpdir)
    paths = {
        "v172_report": root / "v172.json",
        "v172_dataset": root / "v172.jsonl",
        "v173_report": root / "v173.json",
        "v174_report": root / "v174.json",
        "v175_plan": root / "v175-plan.jsonl",
    }
    rows = [
        _v172_row(seed=5, row_id=0, x=0.0, safe_action="eat"),
        _v172_row(seed=5, row_id=1, x=1.0, safe_action="drink"),
        _v172_row(seed=13, row_id=2, x=0.0, safe_action="eat"),
        _v172_row(seed=13, row_id=3, x=1.0, safe_action="drink"),
    ]
    _write_jsonl(paths["v172_dataset"], rows)
    v172_report = {
        "dataset": {
            "path": str(paths["v172_dataset"]),
            "row_count": len(rows),
            "dataset_digest": stable_payload_digest(rows),
        }
    }
    _write_json(paths["v172_report"], v172_report)
    v173_report = _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_SCHEMA_VERSION
            ),
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_POLICY,
            "classification": {
                "primary": EXPECTED_V173_CLASSIFICATION,
                "labels": [EXPECTED_V173_CLASSIFICATION],
            },
            "source_validation": {"passed": True, "failures": []},
            "source_split_evaluation": {
                "row_count": len(rows),
                "unique_row_count": len(rows),
                "tied_row_count": 0,
            },
            "dataset_digest": stable_payload_digest(rows),
            "diagnostics_only": True,
            "training_ran": False,
            "training_authorized": False,
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
    )
    _write_json(paths["v173_report"], v173_report)
    return paths, {"rows": rows, "v172_report": v172_report, "v173_report": v173_report}


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
            "replay_outcome_summary": [row_id, action] if mask[action] else None,
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
            "branch_tick": 97,
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
