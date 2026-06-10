from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.dataset import TrajectoryJsonlDataset
from evolution_sim.mind.rollout_sequence_strict_seed_support_recheck import (
    MIND_V3_ROLLOUT_SEQUENCE_STRICT_SEED_SUPPORT_RECHECK_SCHEMA_VERSION,
)
from evolution_sim.mind.sequence_history_shadow_scorer import (
    DEFAULT_OUTPUT_PATH,
    MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION,
    build_sequence_history_shadow_scorer_report,
    load_sequence_history_shadow_scorer_artifact,
)

ROOT = Path(__file__).resolve().parents[2]
STRICT_SEEDS = (5, 13, 19, 29, 37, 41)


class MindV3SequenceHistoryShadowScorerTests(unittest.TestCase):
    def test_sequence_history_shadow_scorer_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:sequence-history-shadow-scorer"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_sequence_history_shadow_scorer"
            ),
        )

    def test_synthetic_artifact_loads_and_scores_deterministically(self) -> None:
        report = build_sequence_history_shadow_scorer_report(
            v138_report=_v138_report(),
            v138_report_path=None,
            train_trajectory_datasets=(
                _dataset("train-seed2.jsonl", ["eat", "drink"] * 5, seed=2),
                _dataset("train-seed3.jsonl", ["eat", "drink"] * 5, seed=3),
            ),
            strict_heldout_trajectory_datasets=tuple(
                _dataset(f"strict-seed{seed}.jsonl", ["eat", "drink"] * 5, seed=seed)
                for seed in STRICT_SEEDS
            ),
        ).report

        self.assertEqual(
            report["schema_version"],
            MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION,
        )
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(
            report["artifact_roundtrip"][
                "loaded_artifact_scores_match_pre_serialization"
            ]
        )
        scorer = load_sequence_history_shadow_scorer_artifact(report)
        score = scorer.score(
            sequence_keys=("mask=0000010000000000000|history_any=True",),
            valid_actions=("eat", "drink"),
        )
        self.assertIn(score["predicted_action"], ("eat", "drink"))
        self.assertEqual(set(score["scores"]), {"eat", "drink"})
        self.assertEqual(score["scores"], score["valid_action_scores"])
        self.assertIn("eat", score["raw_train_counts"])
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_authorized"]
        )
        self.assertFalse(
            report["authorization_block"]["shadow_scorer_execution_authorized"]
        )

    def test_fails_closed_when_v138_source_integrity_failed(self) -> None:
        v138 = _v138_report()
        v138["source_integrity"] = {"passed": False}

        report = build_sequence_history_shadow_scorer_report(
            v138_report=v138,
            v138_report_path=None,
            train_trajectory_datasets=(
                _dataset("train-seed2.jsonl", ["eat", "drink"] * 3, seed=2),
            ),
            strict_heldout_trajectory_datasets=tuple(
                _dataset(f"strict-seed{seed}.jsonl", ["eat", "drink"] * 3, seed=seed)
                for seed in STRICT_SEEDS
            ),
        ).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "v138_source_integrity_not_passed",
            report["source_integrity"]["failures"],
        )
        self.assertEqual(
            report["classification"]["primary"],
            "sequence_history_shadow_scorer_source_integrity_failed",
        )

    def test_fails_closed_when_v138_report_is_blocked(self) -> None:
        v138 = _v138_report()
        v138["support_floors"] = {"passed": False}
        v138["classification"] = {
            "primary": "rollout_sequence_strict_seed_support_blocked"
        }

        report = build_sequence_history_shadow_scorer_report(
            v138_report=v138,
            v138_report_path=None,
            train_trajectory_datasets=(
                _dataset("train-seed2.jsonl", ["eat", "drink"] * 3, seed=2),
            ),
            strict_heldout_trajectory_datasets=tuple(
                _dataset(f"strict-seed{seed}.jsonl", ["eat", "drink"] * 3, seed=seed)
                for seed in STRICT_SEEDS
            ),
        ).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "v138_support_floors_not_passed",
            report["source_integrity"]["failures"],
        )
        self.assertIn(
            "v138_classification_not_ready",
            report["source_integrity"]["failures"],
        )

    def test_fails_closed_when_train_source_seed_set_intersects_strict_seed(
        self,
    ) -> None:
        report = build_sequence_history_shadow_scorer_report(
            v138_report=_v138_report(),
            v138_report_path=None,
            train_trajectory_datasets=(
                _dataset("opaque-train-mixed.jsonl", ["eat", "drink"] * 3, seed=(2, 5)),
                _dataset("opaque-train-clean.jsonl", ["eat", "drink"] * 3, seed=2),
            ),
            strict_heldout_trajectory_datasets=tuple(
                _dataset(f"strict-seed{seed}.jsonl", ["eat", "drink"] * 3, seed=seed)
                for seed in STRICT_SEEDS
            ),
        ).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "train_source_seed_set_intersects_strict_seeds",
            report["source_integrity"]["failures"],
        )
        self.assertEqual(
            report["strict_seed_source_integrity"][
                "strict_seed_train_source_leakage_count"
            ],
            1,
        )

    def test_no_data_supported_action_does_not_pick_by_action_order(self) -> None:
        report = build_sequence_history_shadow_scorer_report(
            v138_report=_v138_report(),
            v138_report_path=None,
            train_trajectory_datasets=(
                _dataset("train-seed2.jsonl", ["eat", "drink"] * 3, seed=2),
            ),
            strict_heldout_trajectory_datasets=tuple(
                _dataset(f"strict-seed{seed}.jsonl", ["eat", "drink"] * 3, seed=seed)
                for seed in STRICT_SEEDS
            ),
        ).report
        scorer = load_sequence_history_shadow_scorer_artifact(report)

        score = scorer.score(
            sequence_keys=("no-matching-sequence-key",),
            valid_actions=("stay",),
        )

        self.assertIsNone(score["predicted_action"])
        self.assertEqual(score["score_source"], "no_data_supported_action")
        self.assertFalse(score["supported_prediction"])
        self.assertEqual(score["scores"], {"stay": 0})

    def test_score_payload_respects_valid_action_mask(self) -> None:
        report = build_sequence_history_shadow_scorer_report(
            v138_report=_v138_report(),
            v138_report_path=None,
            train_trajectory_datasets=(
                _dataset("train-seed2.jsonl", ["eat", "drink"] * 3, seed=2),
            ),
            strict_heldout_trajectory_datasets=tuple(
                _dataset(f"strict-seed{seed}.jsonl", ["eat", "drink"] * 3, seed=seed)
                for seed in STRICT_SEEDS
            ),
        ).report
        scorer = load_sequence_history_shadow_scorer_artifact(report)

        score = scorer.score(
            sequence_keys=("no-matching-sequence-key",),
            valid_actions=("drink",),
        )

        self.assertEqual(score["predicted_action"], "drink")
        self.assertEqual(set(score["scores"]), {"drink"})
        self.assertEqual(set(score["valid_action_scores"]), {"drink"})
        self.assertIn("eat", score["raw_train_counts"])
        self.assertNotIn("eat", score["scores"])

    def test_serialized_sequence_keys_have_no_forbidden_feature_tokens(self) -> None:
        report = build_sequence_history_shadow_scorer_report(
            v138_report=_v138_report(),
            v138_report_path=None,
            train_trajectory_datasets=(
                _dataset("train-seed2.jsonl", ["eat", "drink"] * 3, seed=2),
            ),
            strict_heldout_trajectory_datasets=tuple(
                _dataset(f"strict-seed{seed}.jsonl", ["eat", "drink"] * 3, seed=seed)
                for seed in STRICT_SEEDS
            ),
        ).report

        self.assertTrue(report["artifact_feature_leakage_scan"]["passed"])
        forbidden = {
            "seed",
            "path",
            "provenance",
            "fixture",
            "future",
            "private",
            "world",
            "outcome",
        }
        sequence_counts = report["artifact"]["sequence_lookup"]["sequence_counts"]
        self.assertGreater(len(sequence_counts), 0)
        for key in sequence_counts:
            lowered = key.lower()
            self.assertFalse(any(token in lowered for token in forbidden), key)

    def test_no_input_cli_exits_nonzero_without_clobbering_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "v139.json"
            output.write_text("keep-me\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evolution_sim.cli.mind_v3_sequence_history_shadow_scorer",
                    "--output",
                    str(output),
                ],
                check=False,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("at least one --train-trajectory", result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep-me\n")

    def test_real_artifact_metrics_when_outputs_exist(self) -> None:
        v138 = Path(
            "output/mind/mind-v3-v138-rollout-sequence-strict-seed-support-recheck.json"
        )
        train = sorted(
            Path("output/mind/v98-broad-support-trajectories").glob(
                "open-mind-v3-[0-9]*-120.jsonl.gz"
            )
        )
        heldout = sorted(
            Path("output/mind/v138-strict-heldout-trajectories").glob(
                "open-mind-v3-[0-9]*-120.jsonl.gz"
            )
        )
        if not v138.exists() or len(train) != 10 or len(heldout) != 6:
            self.skipTest("local v138 and v98/v138 trajectory artifacts are absent")

        report = build_sequence_history_shadow_scorer_report(
            v138_report_path=v138,
            train_trajectory_paths=train,
            strict_heldout_trajectory_paths=heldout,
        ).report

        self.assertEqual(
            DEFAULT_OUTPUT_PATH.name,
            "mind-v3-v139-sequence-history-shadow-scorer.json",
        )
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["artifact_feature_leakage_scan"]["passed"])
        self.assertTrue(report["support_floors"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            "sequence_history_shadow_scorer_passed_diagnostics_only_ready_for_shadow_runtime_logging",
        )
        self.assertTrue(
            report["artifact_roundtrip"][
                "loaded_artifact_scores_match_pre_serialization"
            ]
        )
        self.assertEqual(report["artifact_roundtrip"]["mismatch_count"], 0)
        self.assertEqual(report["artifact_roundtrip"]["checked_example_count"], 11512)
        metrics = report["strict_heldout_metrics"]
        sequence = metrics["model_summaries"][
            "public_rollout_sequence_history_lookup"
        ]
        self.assertEqual(sequence["heldout_record_count"], 11512)
        self.assertEqual(sequence["accuracy"], 0.372481)
        self.assertEqual(sequence["correct_count"], 4288)
        self.assertEqual(sequence["dominant_predicted_action"]["action"], "stay")
        self.assertEqual(sequence["dominant_predicted_action_share"], 0.309851)
        self.assertEqual(sequence["unsupported_action_count"], 0)
        comparisons = metrics["baseline_comparisons"]
        self.assertEqual(
            comparisons["strict_heldout_accuracy_delta_vs_action_only"],
            0.189889,
        )
        self.assertEqual(
            comparisons["strict_heldout_accuracy_delta_vs_action_order"],
            0.189889,
        )
        self.assertEqual(
            metrics["strict_seed_evaluated_record_counts"],
            {
                "5": 1672,
                "13": 1945,
                "19": 2118,
                "29": 2035,
                "37": 2084,
                "41": 1658,
            },
        )
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_authorized"]
        )
        self.assertFalse(
            report["authorization_block"]["shadow_scorer_execution_authorized"]
        )


def _v138_report() -> dict[str, object]:
    return {
        "schema_version": (
            MIND_V3_ROLLOUT_SEQUENCE_STRICT_SEED_SUPPORT_RECHECK_SCHEMA_VERSION
        ),
        "source_integrity": {"passed": True},
        "support_floors": {"passed": True},
        "classification": {
            "primary": (
                "rollout_sequence_strict_seed_support_ready_for_future_sequence_world_model_scorer_diagnostic"
            )
        },
    }


def _dataset(
    path: str,
    actions: list[str],
    *,
    seed: int | tuple[int, ...],
) -> TrajectoryJsonlDataset:
    records = [_record(index, action) for index, action in enumerate(actions)]
    seeds = [seed] if isinstance(seed, int) else list(seed)
    return TrajectoryJsonlDataset(
        path=Path(path),
        header={},
        records=tuple(records),
        footer={
            "provenance": {
                "source_seeds": seeds,
                "config_digest": "unit-config",
                "contract_digest": "unit-contract",
                "split_id": "unit",
                "trajectory_paths": [path],
                "record_count": len(records),
            }
        },
    )


def _record(tick: int, action: str, *, agent_id: int = 1) -> dict[str, object]:
    mask = {name: False for name in ACTION_NAMES}
    mask["eat"] = True
    mask["drink"] = True
    return {
        "tick": tick,
        "agent_id": agent_id,
        "requested_action": action,
        "resolved_action": action,
        "action_source": "mind_v3",
        "resolution_action_valid": True,
        "action_mask": mask,
        "moved": False,
        "before": {
            "energy_ratio": 0.5,
            "hydration_ratio": 0.5,
            "health_ratio": 1.0,
        },
        "after": {
            "energy_ratio": 0.6 if action == "eat" else 0.5,
            "hydration_ratio": 0.6 if action == "drink" else 0.5,
            "health_ratio": 1.0,
        },
        "outcome": {
            "resource_gain": 0.1 if action == "eat" else 0.0,
            "feeding": {
                "ate": action == "eat",
                "food_source": "plant" if action == "eat" else None,
            },
            "drinking": {"drank": action == "drink"},
        },
    }
