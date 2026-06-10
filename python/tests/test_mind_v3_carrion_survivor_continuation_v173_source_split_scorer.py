from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY,
)
from evolution_sim.mind.carrion_survivor_continuation_v173_source_split_scorer import (
    EXPECTED_V172_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v173_source_split_scorer,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV173SourceSplitScorerTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v173-source-split-scorer"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v173_source_split_scorer"
            ),
        )

    def test_source_digest_mismatch_closes_invalid_no_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_ready_inputs(tmpdir)

            report = run_carrion_survivor_continuation_v173_source_split_scorer(
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                output_path=paths["v173_report"],
                expected_v172_exact_digest="wrong",
                expected_v172_dataset_digest=stable_payload_digest(
                    payloads["rows"]
                ),
                expected_v172_row_count=4,
                expected_v172_unique_row_count=4,
                expected_v172_tied_row_count=0,
                support_provenance_seeds=(5, 13),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v172_unexpected_exact_digest",
            report["source_validation"]["failures"],
        )
        self.assertTrue(
            report["classification"]["primary"].endswith("closed_invalid_no_shadow")
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["shadow_eval_ran"])
        self.assertFalse(report["promotion_authorized"])
        self.assertEqual(report["exact_digest"], _digest_without_exact(report))

    def test_synthetic_source_split_ready_uses_only_public_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_ready_inputs(tmpdir)

            report = run_carrion_survivor_continuation_v173_source_split_scorer(
                v172_report_path=paths["v172_report"],
                v172_dataset_path=paths["v172_dataset"],
                output_path=paths["v173_report"],
                expected_v172_exact_digest=str(payloads["report"]["exact_digest"]),
                expected_v172_dataset_digest=stable_payload_digest(
                    payloads["rows"]
                ),
                expected_v172_row_count=4,
                expected_v172_unique_row_count=4,
                expected_v172_tied_row_count=0,
                support_provenance_seeds=(5, 13),
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["leakage_scan"]["passed"])
        self.assertTrue(report["row_schema_validation"]["passed"])
        self.assertTrue(
            report["classification"]["primary"].endswith(
                "source_split_scorer_ready_for_v174_shadow_eval_no_runtime"
            )
        )
        split = report["source_split_evaluation"]
        singleton = split["singleton_scorer"]
        self.assertEqual(split["unique_row_count"], 4)
        self.assertEqual(split["tied_row_count"], 0)
        self.assertEqual(singleton["safe_hit_rate"], 1.0)
        self.assertGreaterEqual(
            singleton["safe_hit_margin_over_best_trivial"],
            0.05,
        )
        self.assertEqual(singleton["unsupported_prediction_count"], 0)
        self.assertEqual(singleton["no_prediction_count"], 0)
        self.assertTrue(
            singleton["every_left_out_seed_has_nonzero_safe_hit_support"]
        )
        self.assertTrue(
            split["trainable_input_policy"]["uses_only_trainable_public_features"]
        )
        self.assertFalse(
            report["support_provenance_seed_policy"][
                "support_provenance_seeds_are_future_promotion_heldout"
            ]
        )

    def test_cli_writes_report(self) -> None:
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
                    "mind_v3_carrion_survivor_continuation_v173_source_split_scorer",
                    "--v172-report",
                    str(paths["v172_report"]),
                    "--v172-dataset",
                    str(paths["v172_dataset"]),
                    "--output",
                    str(paths["v173_report"]),
                    "--expected-v172-exact-digest",
                    str(payloads["report"]["exact_digest"]),
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
            written = json.loads(paths["v173_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v173_source_split_scorer_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_SCHEMA_VERSION,
        )
        self.assertIn("exact_digest", written)


def _write_ready_inputs(tmpdir: str) -> tuple[dict[str, Path], dict[str, object]]:
    root = Path(tmpdir)
    paths = {
        "v172_report": root / "v172.json",
        "v172_dataset": root / "v172.jsonl",
        "v173_report": root / "v173.json",
    }
    rows = [
        _v172_row(seed=5, row_id=0, x=0.0, safe_action="eat"),
        _v172_row(seed=5, row_id=1, x=1.0, safe_action="move_north"),
        _v172_row(seed=13, row_id=2, x=0.0, safe_action="eat"),
        _v172_row(seed=13, row_id=3, x=1.0, safe_action="move_north"),
    ]
    _write_jsonl(paths["v172_dataset"], rows)
    report = _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION
            ),
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_POLICY,
            "classification": {
                "primary": EXPECTED_V172_CLASSIFICATION,
                "labels": [EXPECTED_V172_CLASSIFICATION],
            },
            "dataset": {
                "path": str(paths["v172_dataset"]),
                "row_count": len(rows),
                "row_schema_version": (
                    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
                ),
                "feature_policy_id": (
                    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY
                ),
                "dataset_digest": stable_payload_digest(rows),
            },
            "leakage_scan": {"passed": True, "failure_count": 0, "failures": []},
            "row_schema_validation": {
                "passed": True,
                "failure_count": 0,
                "failures": [],
                "row_count": len(rows),
                "classification_counts": {
                    "unique_replay_verified_best_action": len(rows)
                },
            },
            "unique_action_vs_tied_action_row_counts": {
                "unique_action_row_count": len(rows),
                "tied_action_row_count": 0,
                "unresolved_row_count": 0,
            },
            "route_recommendation": {
                "recommended_next_route": (
                    "v173_diagnostics_only_source_split_scorer_no_training"
                ),
                "v173_source_split_scorer_recommended": True,
                "training_authorized": False,
                "scorer_retraining_authorized": False,
                "runtime_integration_authorized": False,
                "shadow_or_live_eval_authorized": False,
                "promotion_authorized": False,
            },
            "source_split_evaluation_plan_for_v173": {
                "recommended_next_route": (
                    "v173_diagnostics_only_source_split_scorer_no_training"
                ),
                "source_split_evaluation_required": True,
                "leave_one_support_seed_out_required": True,
                "source_group_key": "metadata.seed",
                "support_provenance_seeds": [5, 13],
                "disallowed_future_promotion_heldout_seed_reuse": [5, 13],
                "training_authorized": False,
                "scorer_retraining_authorized": False,
                "shadow_eval_authorized": False,
                "runtime_integration_authorized": False,
                "promotion_authorized": False,
            },
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
    _write_report(paths["v172_report"], report)
    return paths, {"rows": rows, "report": report}


def _v172_row(
    *,
    seed: int,
    row_id: int,
    x: float,
    safe_action: str,
) -> dict[str, object]:
    public_actions = ["stay", "eat", "move_north"]
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
            "score_target": 1.0 if action == safe_action else 0.0 if mask[action] else None,
            "value_target": 1.0 if action == safe_action else 0.0 if mask[action] else None,
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
