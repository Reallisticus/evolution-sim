from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.current_route_decision import (
    CURRENT_CLOSED_PATH,
    MIND_V3_CURRENT_ROUTE_DECISION_SCHEMA_VERSION,
    NEXT_ALLOWED_RESEARCH_DIRECTION,
)
from evolution_sim.mind.dataset import TrajectoryJsonlDataset
from evolution_sim.mind.rollout_sequence_strict_seed_support_recheck import (
    DEFAULT_OUTPUT_PATH,
    MIND_V3_ROLLOUT_SEQUENCE_STRICT_SEED_SUPPORT_RECHECK_SCHEMA_VERSION,
    build_rollout_sequence_strict_seed_support_recheck_report,
)
from evolution_sim.mind.rollout_sequence_support_audit import (
    MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parents[2]
STRICT_SEEDS = (5, 13, 19, 29, 37, 41)


class MindV3RolloutSequenceStrictSeedSupportRecheckTests(unittest.TestCase):
    def test_strict_seed_recheck_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:rollout-sequence-strict-seed-support-recheck"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_rollout_sequence_strict_seed_support_recheck"
            ),
        )

    def test_synthetic_strict_seeds_are_forced_heldout_and_clear_floors(self) -> None:
        report = build_rollout_sequence_strict_seed_support_recheck_report(
            v137_report=_v137_report(),
            v137_report_path=None,
            v136_report=_v136_report(),
            v136_report_path=None,
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
            MIND_V3_ROLLOUT_SEQUENCE_STRICT_SEED_SUPPORT_RECHECK_SCHEMA_VERSION,
        )
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["support_floors"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            "rollout_sequence_strict_seed_support_ready_for_future_sequence_world_model_scorer_diagnostic",
        )
        split = report["train_heldout_split"]
        self.assertEqual(split["train_source_count"], 2)
        self.assertEqual(split["heldout_source_count"], 6)
        roles = report["strict_seed_source_integrity"]
        self.assertEqual(roles["strict_seeds_present_in_heldout"], list(STRICT_SEEDS))
        self.assertEqual(roles["strict_seeds_present_in_train"], [])
        metrics = report["strict_seed_metrics"]
        self.assertTrue(
            metrics["baseline_comparisons"]["beats_action_only_baseline"]
        )
        self.assertTrue(
            metrics["baseline_comparisons"]["beats_action_order_baseline"]
        )
        self.assertEqual(metrics["unsupported_action_count"], 0)
        for seed in STRICT_SEEDS:
            self.assertGreater(
                metrics["per_strict_seed"][str(seed)]["heldout_record_count"],
                0,
            )
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_authorized"]
        )

    def test_fails_closed_when_v137_is_not_ready(self) -> None:
        v137 = _v137_report()
        v137["classification"] = {"primary": "rollout_sequence_support_blocked"}

        report = build_rollout_sequence_strict_seed_support_recheck_report(
            v137_report=v137,
            v137_report_path=None,
            v136_report=_v136_report(),
            v136_report_path=None,
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
            "v137_classification_not_ready",
            report["source_integrity"]["failures"],
        )
        self.assertEqual(
            report["classification"]["primary"],
            "rollout_sequence_strict_seed_recheck_source_integrity_failed",
        )
        self.assertFalse(report["authorization_block"]["training_authorized"])

    def test_fails_closed_when_strict_seed_appears_in_train(self) -> None:
        report = build_rollout_sequence_strict_seed_support_recheck_report(
            v137_report=_v137_report(),
            v137_report_path=None,
            v136_report=_v136_report(),
            v136_report_path=None,
            train_trajectory_datasets=(
                _dataset("train-seed5.jsonl", ["eat", "drink"] * 3, seed=5),
                _dataset("train-seed2.jsonl", ["eat", "drink"] * 3, seed=2),
            ),
            strict_heldout_trajectory_datasets=tuple(
                _dataset(f"strict-seed{seed}.jsonl", ["eat", "drink"] * 3, seed=seed)
                for seed in STRICT_SEEDS
            ),
        ).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "strict_seeds_not_present_only_in_heldout",
            report["source_integrity"]["failures"],
        )
        roles = report["strict_seed_source_integrity"]
        self.assertEqual(roles["strict_seeds_present_in_train"], [5])
        self.assertEqual(roles["strict_seed_train_source_leakage_count"], 1)

    def test_fails_closed_when_train_source_seed_set_intersects_strict_seed(
        self,
    ) -> None:
        report = build_rollout_sequence_strict_seed_support_recheck_report(
            v137_report=_v137_report(),
            v137_report_path=None,
            v136_report=_v136_report(),
            v136_report_path=None,
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
            "strict_seeds_not_present_only_in_heldout",
            report["source_integrity"]["failures"],
        )
        roles = report["strict_seed_source_integrity"]
        self.assertEqual(roles["strict_seeds_present_in_train"], [5])
        self.assertEqual(roles["strict_seed_train_source_leakage_count"], 1)
        self.assertEqual(
            roles["strict_seed_train_source_leakage_examples"][0]["source_seed_set"],
            [2, 5],
        )

    def test_fails_closed_when_train_and_heldout_source_paths_overlap(self) -> None:
        report = build_rollout_sequence_strict_seed_support_recheck_report(
            v137_report=_v137_report(),
            v137_report_path=None,
            v136_report=_v136_report(),
            v136_report_path=None,
            train_trajectory_datasets=(
                _dataset("shared-nonstrict.jsonl", ["eat", "drink"] * 3, seed=2),
                _dataset("train-clean.jsonl", ["eat", "drink"] * 3, seed=3),
            ),
            strict_heldout_trajectory_datasets=(
                _dataset("shared-nonstrict.jsonl", ["eat", "drink"] * 3, seed=3),
                *(
                    _dataset(
                        f"strict-seed{seed}.jsonl",
                        ["eat", "drink"] * 3,
                        seed=seed,
                    )
                    for seed in STRICT_SEEDS
                ),
            ),
        ).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "train_strict_heldout_source_path_overlap",
            report["source_integrity"]["failures"],
        )
        self.assertEqual(report["source_integrity"]["source_path_overlap_count"], 1)
        self.assertEqual(
            report["source_integrity"]["source_path_overlap_examples"],
            ["shared-nonstrict.jsonl"],
        )
        roles = report["strict_seed_source_integrity"]
        self.assertEqual(roles["source_path_overlap_count"], 1)

    def test_passive_only_strict_seed_source_fails_evaluated_count_floor(self) -> None:
        report = build_rollout_sequence_strict_seed_support_recheck_report(
            v137_report=_v137_report(),
            v137_report_path=None,
            v136_report=_v136_report(),
            v136_report_path=None,
            train_trajectory_datasets=(
                _dataset("train-seed2.jsonl", ["eat", "drink"] * 4, seed=2),
                _dataset("train-seed3.jsonl", ["eat", "drink"] * 4, seed=3),
            ),
            strict_heldout_trajectory_datasets=(
                _dataset("strict-seed5.jsonl", ["eat", "drink"] * 4, seed=5, passive=True),
                *(
                    _dataset(
                        f"strict-seed{seed}.jsonl",
                        ["eat", "drink"] * 4,
                        seed=seed,
                    )
                    for seed in STRICT_SEEDS
                    if seed != 5
                ),
            ),
        ).report

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertFalse(report["support_floors"]["passed"])
        self.assertEqual(
            report["support_floors"]["first_failed_floor"],
            "all_strict_seeds_have_evaluated_records",
        )
        metrics = report["strict_seed_metrics"]
        self.assertEqual(metrics["strict_seed_evaluated_record_counts"]["5"], 0)
        self.assertEqual(metrics["strict_seed_missing_evaluated_record_seeds"], [5])

    def test_opaque_strict_paths_use_footer_provenance_seed(self) -> None:
        report = build_rollout_sequence_strict_seed_support_recheck_report(
            v137_report=_v137_report(),
            v137_report_path=None,
            v136_report=_v136_report(),
            v136_report_path=None,
            train_trajectory_datasets=(
                _dataset("opaque-train-a.jsonl", ["eat", "drink"] * 3, seed=2),
                _dataset("opaque-train-b.jsonl", ["eat", "drink"] * 3, seed=3),
            ),
            strict_heldout_trajectory_datasets=tuple(
                _dataset(f"opaque-heldout-{index}.jsonl", ["eat", "drink"] * 3, seed=seed)
                for index, seed in enumerate(STRICT_SEEDS)
            ),
        ).report

        roles = report["strict_seed_source_integrity"]
        self.assertTrue(roles["strict_seed_presence_only_heldout"])
        self.assertEqual(roles["strict_seeds_present_in_heldout"], list(STRICT_SEEDS))
        for source in roles["strict_heldout_sources"]:
            self.assertEqual(len(source["source_seed_set"]), 1)
        self.assertTrue(report["source_integrity"]["passed"])

    def test_no_input_cli_exits_nonzero_without_clobbering_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "v138.json"
            output.write_text("keep-me\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evolution_sim.cli.mind_v3_rollout_sequence_strict_seed_support_recheck",
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

    def test_real_artifact_strict_seed_metrics_when_outputs_exist(self) -> None:
        v136 = Path("output/mind/mind-v3-v136-current-route-decision.json")
        v137 = Path("output/mind/mind-v3-v137-rollout-sequence-support-audit.json")
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
        if not v136.exists() or not v137.exists() or len(train) != 10 or len(heldout) != 6:
            self.skipTest("local v136/v137 and v98/v138 trajectory artifacts are absent")

        report = build_rollout_sequence_strict_seed_support_recheck_report(
            v137_report_path=v137,
            v136_report_path=v136,
            train_trajectory_paths=train,
            strict_heldout_trajectory_paths=heldout,
        ).report

        self.assertEqual(
            DEFAULT_OUTPUT_PATH.name,
            "mind-v3-v138-rollout-sequence-strict-seed-support-recheck.json",
        )
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["support_floors"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            "rollout_sequence_strict_seed_support_ready_for_future_sequence_world_model_scorer_diagnostic",
        )
        split = report["train_heldout_split"]
        self.assertEqual(split["example_count"], 21420)
        self.assertEqual(split["train_record_count"], 9908)
        self.assertEqual(split["heldout_record_count"], 11512)
        self.assertEqual(split["train_source_count"], 10)
        self.assertEqual(split["heldout_source_count"], 6)
        self.assertEqual(split["configured_heldout_seed_leakage_count"], 0)
        roles = report["strict_seed_source_integrity"]
        self.assertEqual(roles["strict_seeds_present_in_heldout"], list(STRICT_SEEDS))
        self.assertEqual(roles["strict_seeds_present_in_train"], [])
        self.assertEqual(roles["strict_seed_train_source_leakage_count"], 0)
        self.assertEqual(roles["source_path_overlap_count"], 0)
        metrics = report["strict_seed_metrics"]
        aggregate = metrics["aggregate_strict_heldout"]
        self.assertTrue(metrics["all_strict_seeds_have_evaluated_records"])
        self.assertEqual(metrics["strict_seed_missing_evaluated_record_seeds"], [])
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
        self.assertEqual(aggregate["heldout_record_count"], 11512)
        self.assertEqual(aggregate["accuracy"], 0.372481)
        self.assertEqual(aggregate["correct_count"], 4288)
        self.assertEqual(aggregate["dominant_predicted_action"]["action"], "stay")
        self.assertEqual(aggregate["dominant_predicted_action_share"], 0.309851)
        self.assertEqual(aggregate["unsupported_action_count"], 0)
        comparisons = metrics["baseline_comparisons"]
        self.assertEqual(
            comparisons["strict_heldout_accuracy_delta_vs_action_only"],
            0.189889,
        )
        self.assertEqual(
            comparisons["strict_heldout_accuracy_delta_vs_action_order"],
            0.189889,
        )
        expected_per_seed = {
            "5": (1672, 0.417464, 698, 0.169258, 0.248206, 0.30323),
            "13": (1945, 0.320308, 623, 0.183033, 0.137275, 0.347044),
            "19": (2118, 0.348914, 739, 0.20916, 0.139754, 0.339943),
            "29": (2035, 0.451106, 918, 0.218673, 0.232433, 0.342015),
            "37": (2084, 0.341651, 712, 0.145393, 0.196258, 0.367562),
            "41": (1658, 0.360676, 598, 0.164053, 0.196623, 0.282268),
        }
        for seed, expected in expected_per_seed.items():
            seed_metrics = metrics["per_strict_seed"][seed]
            self.assertEqual(seed_metrics["heldout_record_count"], expected[0])
            self.assertEqual(seed_metrics["accuracy"], expected[1])
            self.assertEqual(seed_metrics["correct_count"], expected[2])
            self.assertEqual(seed_metrics["action_only_accuracy"], expected[3])
            self.assertEqual(seed_metrics["action_order_accuracy"], expected[3])
            self.assertEqual(seed_metrics["accuracy_delta_vs_action_only"], expected[4])
            self.assertEqual(seed_metrics["accuracy_delta_vs_action_order"], expected[4])
            self.assertEqual(
                seed_metrics["dominant_predicted_action_share"],
                expected[5],
            )
            self.assertEqual(seed_metrics["unsupported_action_count"], 0)
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_authorized"]
        )


def _v136_report() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CURRENT_ROUTE_DECISION_SCHEMA_VERSION,
        "source_integrity": {"passed": True},
        "next_allowed_research_direction": NEXT_ALLOWED_RESEARCH_DIRECTION,
        "current_closed_path": CURRENT_CLOSED_PATH,
    }


def _v137_report() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION,
        "source_integrity": {"passed": True},
        "classification": {"primary": "rollout_sequence_support_ready_for_future_sequence_world_model_scorer"},
    }


def _dataset(
    path: str,
    actions: list[str],
    *,
    seed: int | tuple[int, ...],
    passive: bool = False,
) -> TrajectoryJsonlDataset:
    records = [
        _record(index, action, passive=passive)
        for index, action in enumerate(actions)
    ]
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


def _record(
    tick: int,
    action: str,
    *,
    agent_id: int = 1,
    passive: bool = False,
) -> dict[str, object]:
    mask = {name: False for name in ACTION_NAMES}
    mask["eat"] = True
    mask["drink"] = True
    return {
        "tick": tick,
        "agent_id": agent_id,
        "requested_action": action,
        "resolved_action": action,
        "action_source": "passive" if passive else "mind_v3",
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
