from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from evolution_sim.mind import (
    carrion_survivor_continuation_v177_exact_branch_replay_expansion as v177,
)
from evolution_sim.mind.carrion_survivor_continuation_v176_transition_diagnostic_planner import (
    M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV177ExactBranchReplayExpansionTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v177-exact-branch-replay-expansion"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v177_exact_branch_replay_expansion"
            ),
        )

    def test_v176_digest_mismatch_closes_invalid_without_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)

            report = v177.run_carrion_survivor_continuation_v177_exact_branch_replay_expansion(
                v176_report_path=paths["v176_report"],
                v177_shard_plan_path=paths["plan"],
                output_path=paths["report"],
                transition_dataset_output_path=paths["dataset"],
                expected_v176_exact_digest="wrong",
                expected_v177_shard_plan_digest=stable_payload_digest(
                    payloads["plan_rows"]
                ),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v176_unexpected_exact_digest",
            report["source_validation"]["failures"],
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v177_exact_branch_replay_"
                "expansion_source_invalid_closed_no_training"
            ),
        )
        self.assertEqual(report["dataset"]["row_count"], 0)
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_action_selection_changed"])

    def test_synthetic_compact_transition_report_writes_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            transition_rows = [_transition_row()]
            with mock.patch.object(
                v177,
                "materialize_selected_branch_points",
                return_value=(
                    ["fake-materialized"],
                    {
                        "passed": True,
                        "materialized_branch_point_count": 1,
                        "selected_branch_point_count": 1,
                    },
                ),
            ), mock.patch.object(
                v177,
                "build_compact_transition_rows",
                return_value=transition_rows,
            ):
                report = v177.run_carrion_survivor_continuation_v177_exact_branch_replay_expansion(
                    v176_report_path=paths["v176_report"],
                    v177_shard_plan_path=paths["plan"],
                    output_path=paths["report"],
                    transition_dataset_output_path=paths["dataset"],
                    expected_v176_exact_digest=payloads["v176_report"]["exact_digest"],
                    expected_v177_shard_plan_digest=stable_payload_digest(
                        payloads["plan_rows"]
                    ),
                )
            written_rows = [
                json.loads(line)
                for line in paths["dataset"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["selection"]["passed"])
        self.assertEqual(report["dataset"]["row_count"], 1)
        self.assertEqual(written_rows, transition_rows)
        self.assertTrue(report["leakage_scan"]["passed"])
        self.assertTrue(report["row_schema_validation"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v177_exact_branch_replay_"
                "expansion_compact_transition_rows_ready_no_training"
            ),
        )
        self.assertEqual(report["exact_digest"], _digest_without_exact(report))
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["promotion_authorized"])

    def test_leakage_scan_allows_same_agent_context_but_rejects_agent_id(self) -> None:
        allowed = v177.trainable_payload_leakage_scan(
            [
                {
                    "previous_same_agent_public_context": {
                        "available": True,
                        "public_action": "eat",
                    }
                }
            ]
        )
        rejected = v177.trainable_payload_leakage_scan(
            [
                {
                    "previous_same_agent_public_context": {
                        "agent_id": 17,
                    }
                }
            ]
        )

        self.assertTrue(allowed["passed"])
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["failures"][0]["token"], "agent_id")

    def test_cli_writes_invalid_report_without_replay(self) -> None:
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
                    "mind_v3_carrion_survivor_continuation_v177_exact_branch_replay_expansion",
                    "--v176-report",
                    str(paths["v176_report"]),
                    "--v177-shard-plan",
                    str(paths["plan"]),
                    "--output",
                    str(paths["report"]),
                    "--transition-dataset-output",
                    str(paths["dataset"]),
                    "--expected-v176-exact-digest",
                    "wrong",
                    "--expected-v177-shard-plan-digest",
                    stable_payload_digest(payloads["plan_rows"]),
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
            "carrion_survivor_continuation_v177_exact_branch_replay_expansion_report=",
            completed.stdout,
        )
        self.assertIn("classification=", completed.stdout)
        self.assertEqual(
            written["schema_version"],
            v177.M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION,
        )
        self.assertEqual(written["dataset"]["row_count"], 0)


def _write_inputs(tmpdir: str) -> tuple[dict[str, Path], dict[str, object]]:
    root = Path(tmpdir)
    paths = {
        "source": root / "source.jsonl",
        "v176_report": root / "v176.json",
        "plan": root / "v177-plan.jsonl",
        "report": root / "v177.json",
        "dataset": root / "v177.jsonl",
    }
    _write_jsonl(
        paths["source"],
        [
            {"record": _source_record(tick=93, requested_action="eat")},
            {"record": _source_record(tick=94, requested_action="drink")},
        ],
    )
    plan_rows = [
        {
            "schema_version": (
                "m3_carrion_survivor_continuation_v177_exact_branch_replay_shard_row_v1"
            ),
            "route": "exact_branch_replay_expansion",
            "priority": 1,
            "seed": 41,
            "branch_tick": 94,
            "agent_id": 17,
            "branch_id": "v171-seed-041-source-line-000002-tick-94-agent-17",
            "source_path": str(paths["source"]),
            "line_number": 2,
            "row_index": 563,
            "failed_safe_action": "drink",
            "failure_types": ["unique_top_1_miss"],
            "candidate_forced_actions": ["stay", "drink"],
            "training_authorized": False,
            "runtime_artifact_authorized": False,
        }
    ]
    _write_jsonl(paths["plan"], plan_rows)
    v176_report = _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_SCHEMA_VERSION
            ),
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_POLICY,
            "classification": {
                "primary": v177.EXPECTED_V176_CLASSIFICATION,
                "labels": [v177.EXPECTED_V176_CLASSIFICATION],
            },
            "v177_shard_plan_output": {
                "written": True,
                "plan_digest": stable_payload_digest(plan_rows),
            },
            "diagnostics_only": True,
            "training_ran": False,
            "fit_ran": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "runtime_observation_schema_changed": False,
            "runtime_policy_changed": False,
            "shadow_eval_ran": False,
            "live_ab_ran": False,
            "promotion_authorized": False,
            "replay_viewer_schema_changed": False,
        }
    )
    _write_json(paths["v176_report"], v176_report)
    return paths, {"plan_rows": plan_rows, "v176_report": v176_report}


def _source_record(*, tick: int, requested_action: str) -> dict[str, object]:
    return {
        "tick": tick,
        "agent_id": 17,
        "observation_schema": "mind_observation_v3",
        "observation_metadata": {"seed": 41},
        "observation_input": {"schema_version": "synthetic", "values": [tick]},
        "observation_digest": f"digest-{tick}",
        "action_mask": {
            action: action in {"stay", "drink"} for action in v177.ACTION_NAMES
        },
        "requested_action": requested_action,
        "resolved_action": requested_action,
        "moved": False,
        "before": {"alive": True},
    }


def _transition_row() -> dict[str, object]:
    current_observation = {"schema_version": "synthetic", "values": [94]}
    next_observation = {"schema_version": "synthetic", "values": [95]}
    current_mask = {action: action in {"stay", "drink"} for action in v177.ACTION_NAMES}
    next_mask = {action: action in {"stay", "eat"} for action in v177.ACTION_NAMES}
    previous_context = {
        "available": True,
        "public_observation": {"schema_version": "synthetic", "values": [93]},
        "public_action_mask": current_mask,
        "public_action": "eat",
        "moved": False,
    }
    return {
        "schema_version": (
            v177.M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION
        ),
        "row_origin": "v177_exact_branch_replay_from_v176_shard_plan",
        "feature_policy_id": (
            v177.M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY
        ),
        "trainable_public_features": {
            "current_public_observation": current_observation,
            "current_public_action_mask": current_mask,
            "forced_action": "drink",
            "previous_same_agent_public_context": previous_context,
            "next_public_observation": next_observation,
            "next_public_action_mask": next_mask,
        },
        "current_public_observation": current_observation,
        "current_public_action_mask": current_mask,
        "forced_action": "drink",
        "previous_same_agent_public_context": previous_context,
        "next_public_observation": next_observation,
        "next_public_action_mask": next_mask,
        "next_public_observation_available": True,
        "next_public_action_mask_available": True,
        "transition_done": False,
        "short_horizon_public_outcome_summary": {
            "forced_action_used": True,
            "current_reward_total": 0.1,
        },
        "replay_verification": {
            "verified": True,
            "expected_digest": "digest",
            "actual_digest": "digest",
        },
        "metadata": {
            "metadata_schema_version": (
                "m3_carrion_survivor_continuation_v177_compact_transition_metadata_v1"
            ),
            "branch_id": "branch",
            "seed": 41,
            "fixture": "broad",
            "branch_tick": 94,
            "agent_id": 17,
            "source_path": "source.jsonl",
            "line_number": 2,
            "source_identity_used_as_trainable_input": False,
            "runtime_requested_or_resolved_action_used_as_trainable_input": False,
            "current_or_future_outcome_used_as_trainable_input": False,
            "diagnostic_target_used_as_trainable_input": False,
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
