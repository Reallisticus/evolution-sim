from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind.artifacts import MindArtifactError, write_model_artifact
from evolution_sim.mind.baseline import train_behavior_cloning_baseline
from evolution_sim.mind.contracts import (
    MIND_MODEL_ARTIFACT_VERSION,
    MIND_RUNTIME_ENABLED_DEFAULT,
    MIND_V1_DATA_CONTRACT_VERSION,
    mind_v1_data_contract,
)
from evolution_sim.mind.dataset import TrajectoryDatasetError, load_trajectory_jsonl
from evolution_sim.mind.evaluation import compare_heuristic_and_learned
from evolution_sim.mind.learned_policy import LearnedPolicy, load_learned_policy
from evolution_sim.mind.splits import deterministic_seed_split


class MindV1Tests(unittest.TestCase):
    def _write_tiny_trajectory(self, path: Path) -> None:
        writer = JsonlTrajectoryWriter(path)
        SimulationWorld(WorldConfig(seed=7, max_ticks=2)).run(
            mode=RunMode.SUMMARY_ONLY,
            trajectory_sink=writer,
        )

    def test_mind_v1_contract_declares_current_runtime_schemas_disabled_by_default(self) -> None:
        contract = mind_v1_data_contract()

        self.assertEqual(contract["contract_version"], MIND_V1_DATA_CONTRACT_VERSION)
        self.assertFalse(contract["runtime_enabled_by_default"])
        self.assertFalse(MIND_RUNTIME_ENABLED_DEFAULT)
        self.assertEqual(contract["model_artifact_version"], MIND_MODEL_ARTIFACT_VERSION)
        self.assertIn("trajectory_record_fields", contract)
        json.dumps(contract)

    def test_trajectory_jsonl_loader_validates_header_records_and_footer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(path)

            dataset = load_trajectory_jsonl(path)

        self.assertEqual(dataset.header["type"], "header")
        self.assertGreater(dataset.record_count, 0)
        self.assertEqual(dataset.footer["type"], "footer")
        self.assertEqual(
            dataset.footer["trajectory_summary"]["record_count"],
            dataset.record_count,
        )

    def test_trajectory_jsonl_loader_rejects_stale_footer_schema(self) -> None:
        with TemporaryDirectory() as tmpdir:
            original_path = Path(tmpdir) / "trajectory.jsonl.gz"
            stale_path = Path(tmpdir) / "stale-trajectory.jsonl.gz"
            self._write_tiny_trajectory(original_path)
            with gzip.open(original_path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            rows[-1]["summary"]["summary_schema_version"] = "stale"
            with gzip.open(stale_path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")

            with self.assertRaisesRegex(TrajectoryDatasetError, "summary schema"):
                load_trajectory_jsonl(stale_path)

    def test_deterministic_seed_split_is_stable_and_disjoint(self) -> None:
        first = deterministic_seed_split(
            [5, 4, 3, 2, 1, 5],
            validation_fraction=0.4,
            split_seed=17,
        )
        second = deterministic_seed_split(
            [5, 4, 3, 2, 1, 5],
            validation_fraction=0.4,
            split_seed=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first["train"]) & set(first["validation"]), set())
        self.assertEqual(sorted(first["train"] + first["validation"]), [1, 2, 3, 4, 5])

    def test_behavior_cloning_artifact_requires_explicit_mind_enable_flag(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            artifact_path = Path(tmpdir) / "bc-artifact.json"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(dataset.records)
            write_model_artifact(artifact_path, baseline.to_artifact())

            with self.assertRaises(MindArtifactError):
                load_learned_policy(artifact_path)
            policy = load_learned_policy(artifact_path, enable_mind=True)

        self.assertEqual(policy.policy_id, "mind_v1_learned_policy")

    def test_learned_policy_obeys_action_mask(self) -> None:
        policy = LearnedPolicy(action_scores={"eat": 1.0, "stay": 0.1})

        blocked = policy.decide({}, {"stay": True, "eat": False})
        allowed = policy.decide({}, {"stay": True, "eat": True})

        self.assertEqual(blocked.requested_action, "stay")
        self.assertEqual(allowed.requested_action, "eat")

    def test_policy_evaluation_compares_heuristic_and_learned_on_summary_only_seeds(self) -> None:
        policy = LearnedPolicy(action_scores={"stay": 1.0})

        report = compare_heuristic_and_learned(
            learned_policy=policy,
            seeds=[7],
            ticks=2,
        )

        self.assertEqual(report["protocol"]["mode"], RunMode.SUMMARY_ONLY.value)
        self.assertIn("trajectory", report["learned"]["aggregate"])
        self.assertIn("mind_v1_gates", report)
        self.assertIn(report["mind_v1_gates"]["status"], {"pass", "review", "fail"})
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
