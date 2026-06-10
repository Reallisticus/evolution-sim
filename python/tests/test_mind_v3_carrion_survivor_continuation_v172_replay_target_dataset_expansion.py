from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_DATASET_FEATURE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    EXPECTED_V171_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v172_replay_target_dataset_expansion,
    trainable_payload_leakage_scan,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV172ReplayTargetDatasetExpansionTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v172-replay-target-dataset-expansion"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v172_replay_target_dataset_expansion"
            ),
        )

    def test_source_digest_mismatch_closes_invalid_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_ready_inputs(tmpdir)

            report = run_carrion_survivor_continuation_v172_replay_target_dataset_expansion(
                v171_report_path=paths["v171_report"],
                v171_dataset_path=paths["v171_dataset"],
                output_path=paths["v172_report"],
                target_dataset_output_path=paths["v172_dataset"],
                expected_v171_exact_digest="wrong",
                expected_v171_dataset_digest=stable_payload_digest(
                    payloads["rows"]
                ),
                expected_v171_replay_verified_count=4,
                expected_v171_dataset_row_count=2,
                expected_v171_average_replay_safe_set_width=1.5,
                v170_best_set_width=3.0,
                support_provenance_seeds=(5, 13),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v171_unexpected_exact_digest",
            report["source_validation"]["failures"],
        )
        self.assertTrue(report["classification"]["primary"].endswith("closed_invalid"))
        self.assertEqual(report["dataset"]["row_count"], 0)
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertEqual(report["exact_digest"], _digest_without_exact(report))

    def test_builds_v172_dataset_with_metadata_only_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_ready_inputs(tmpdir)
            source_report = payloads["report"]
            source_rows = payloads["rows"]

            report = run_carrion_survivor_continuation_v172_replay_target_dataset_expansion(
                v171_report_path=paths["v171_report"],
                v171_dataset_path=paths["v171_dataset"],
                output_path=paths["v172_report"],
                target_dataset_output_path=paths["v172_dataset"],
                expected_v171_exact_digest=str(source_report["exact_digest"]),
                expected_v171_dataset_digest=stable_payload_digest(source_rows),
                expected_v171_replay_verified_count=4,
                expected_v171_dataset_row_count=2,
                expected_v171_average_replay_safe_set_width=1.5,
                v170_best_set_width=3.0,
                support_provenance_seeds=(5, 13),
            )
            rows = _read_jsonl(paths["v172_dataset"])

        self.assertTrue(
            report["classification"]["primary"].endswith(
                "replay_target_dataset_support_ready_for_v173_source_split_scorer_no_training"
            )
        )
        self.assertEqual(report["dataset"]["row_count"], 2)
        self.assertEqual(report["dataset"]["dataset_digest"], stable_payload_digest(rows))
        self.assertTrue(report["leakage_scan"]["passed"])
        self.assertTrue(report["row_schema_validation"]["passed"])
        self.assertEqual(
            report["safe_set_width_distribution"]["width_counts"],
            {"1": 1, "2": 1},
        )
        self.assertEqual(
            report["unique_action_vs_tied_action_row_counts"],
            {
                "unique_action_row_count": 1,
                "tied_action_row_count": 1,
                "unresolved_row_count": 0,
            },
        )
        self.assertEqual(
            report["action_support_distribution"]["per_action_support_counts"],
            {"eat": 1, "move_north": 1, "stay": 1},
        )
        self.assertEqual(
            report["source_split_evaluation_plan_for_v173"][
                "disallowed_future_promotion_heldout_seed_reuse"
            ],
            [5, 13],
        )
        for row in rows:
            trainable_text = json.dumps(
                row["trainable_public_features"],
                sort_keys=True,
            )
            self.assertNotIn("seed", trainable_text)
            self.assertNotIn("branch_id", trainable_text)
            self.assertNotIn("source_path", trainable_text)
            self.assertTrue(
                row["metadata"][
                    "source_seed_is_support_provenance_not_future_promotion_holdout"
                ]
            )
            self.assertFalse(
                row["metadata"][
                    "runtime_requested_or_resolved_action_used_as_trainable_input"
                ]
            )
        self.assertEqual(report["exact_digest"], _digest_without_exact(report))

    def test_trainable_payload_leakage_scan_rejects_provenance_keys(self) -> None:
        scan = trainable_payload_leakage_scan(
            [
                {
                    "public_observation": {"seed_bucket": 5},
                    "action_mask": {"eat": True},
                }
            ]
        )

        self.assertFalse(scan["passed"])
        self.assertEqual(scan["failures"][0]["token"], "seed")

    def test_cli_writes_source_invalid_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_ready_inputs(tmpdir)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v172_replay_target_dataset_expansion",
                    "--v171-report",
                    str(paths["v171_report"]),
                    "--v171-dataset",
                    str(paths["v171_dataset"]),
                    "--output",
                    str(paths["v172_report"]),
                    "--target-dataset-output",
                    str(paths["v172_dataset"]),
                    "--expected-v171-exact-digest",
                    str(payloads["report"]["exact_digest"]),
                    "--expected-v171-dataset-digest",
                    stable_payload_digest(payloads["rows"]),
                    "--expected-v171-classification",
                    EXPECTED_V171_CLASSIFICATION,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v172_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v172_replay_target_dataset_expansion_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION,
        )
        self.assertTrue(written["classification"]["primary"].endswith("closed_invalid"))
        self.assertIn("exact_digest", written)


def _write_ready_inputs(tmpdir: str) -> tuple[dict[str, Path], dict[str, object]]:
    root = Path(tmpdir)
    paths = {
        "v171_report": root / "v171.json",
        "v171_dataset": root / "v171.jsonl",
        "v172_report": root / "v172.json",
        "v172_dataset": root / "v172.jsonl",
    }
    rows = [
        _v171_row(
            seed=5,
            branch_id="branch-5",
            safe_actions=["eat"],
            public_actions=["stay", "eat"],
        ),
        _v171_row(
            seed=13,
            branch_id="branch-13",
            safe_actions=["move_north", "stay"],
            public_actions=["stay", "move_north"],
        ),
    ]
    _write_jsonl(paths["v171_dataset"], rows)
    report = _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION
            ),
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY,
            "mode": "merge",
            "classification": {
                "primary": EXPECTED_V171_CLASSIFICATION,
                "labels": [EXPECTED_V171_CLASSIFICATION],
            },
            "dataset": {
                "path": str(paths["v171_dataset"]),
                "row_count": len(rows),
                "dataset_digest": stable_payload_digest(rows),
                "trainable_payload_policy": (
                    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_DATASET_FEATURE_POLICY
                ),
            },
            "metrics": {
                "replay_verified_run_count": 4,
                "candidate_run_count": 4,
                "all_replays_verified": True,
                "dataset_row_count": len(rows),
                "average_replay_safe_set_width": 1.5,
                "v170_best_set_average_width": 3.0,
                "support_narrows_broad_set_valued_candidates": True,
                "ready_support_criteria_passed": True,
            },
            "partial_shard_status": {
                "partial": False,
                "missing_shard_ids": [],
                "partial_shard_ids": [],
            },
            "route_recommendation": {
                "v172_target_dataset_expansion_recommended": True,
                "training_authorized": False,
                "runtime_integration_authorized": False,
                "shadow_or_live_eval_authorized": False,
                "promotion_authorized": False,
            },
            "leakage_scan": {"passed": True, "failure_count": 0, "failures": []},
            "diagnostics_only": True,
            "training_ran": False,
            "scorer_retraining_ran": False,
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
    _write_report(paths["v171_report"], report)
    return paths, {"rows": rows, "report": report}


def _v171_row(
    *,
    seed: int,
    branch_id: str,
    safe_actions: list[str],
    public_actions: list[str],
) -> dict[str, object]:
    mask = {action: action in public_actions for action in ACTION_NAMES}
    targets = [
        {
            "action": action,
            "public_mask": mask[action],
            "target_available": mask[action],
            "replay_verified": mask[action],
            "safe_target": action in safe_actions,
            "replay_outcome_key": [1, 2, 3] if mask[action] else None,
            "score_target": 1.0 if action in safe_actions else 0.0 if mask[action] else None,
            "value_target": 1.0 if action in safe_actions else 0.0 if mask[action] else None,
        }
        for action in ACTION_NAMES
    ]
    row: dict[str, object] = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_ROW_SCHEMA_VERSION
        ),
        "feature_policy_id": (
            M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_DATASET_FEATURE_POLICY
        ),
        "trainable_public_features": {
            "public_observation": {
                "schema_version": "synthetic_public_observation",
                "energy_ratio": 0.7,
            },
            "action_mask": mask,
        },
        "public_action_mask": mask,
        "action_value_targets": targets,
        "safe_action_set": safe_actions,
        "target_classification": (
            "unique_replay_verified_winner"
            if len(safe_actions) == 1
            else "multi_action_replay_verified_safe_set"
        ),
        "metadata": {
            "metadata_schema_version": (
                "m3_carrion_survivor_continuation_v171_replay_expansion_metadata_v1"
            ),
            "branch_id": branch_id,
            "seed": seed,
            "fixture": "broad",
            "branch_tick": 97,
            "agent_id": seed + 100,
            "source_path": f"source-{seed}.jsonl",
            "line_number": 1,
            "source_record_digest": f"source-digest-{seed}",
            "materialized_record_digest": f"materialized-digest-{seed}",
            "branch_state_digest": f"state-digest-{seed}",
            "replay_digests_by_action": {
                action: f"replay-{branch_id}-{action}"
                for action in public_actions
            },
            "source_identity_used_as_trainable_input": False,
            "runtime_requested_or_resolved_action_used_as_trainable_input": False,
            "future_outcome_used_as_trainable_input": False,
            "target_safe_action_used_as_trainable_input": False,
        },
    }
    if len(safe_actions) == 1:
        row["robust_winner_action"] = safe_actions[0]
    return row


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
