from __future__ import annotations

import gzip
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_train
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind.artifacts import (
    MindArtifactError,
    validate_model_artifact_manifest,
    write_model_artifact,
)
from evolution_sim.mind.baseline import train_behavior_cloning_baseline
from evolution_sim.mind.contracts import (
    MIND_MODEL_ARTIFACT_VERSION,
    MIND_RUNTIME_ENABLED_DEFAULT,
    MIND_V1_DATA_CONTRACT_VERSION,
    mind_v1_data_contract,
)
from evolution_sim.mind.dataset import (
    TrajectoryDatasetError,
    combined_dataset_provenance,
    dataset_provenance,
    load_trajectory_jsonl,
)
from evolution_sim.mind.evaluation import compare_heuristic_and_learned
from evolution_sim.mind.feature_policy import feature_keys_from_observation
from evolution_sim.mind.gates import build_mind_v1_gate_report
from evolution_sim.mind.learned_policy import LearnedPolicy, load_learned_policy
from evolution_sim.mind.splits import deterministic_seed_split


class MindV1Tests(unittest.TestCase):
    def _write_tiny_trajectory(self, path: Path, *, seed: int = 7) -> None:
        writer = JsonlTrajectoryWriter(
            path,
            source_seeds=[seed],
            split_id="tiny_train",
        )
        SimulationWorld(WorldConfig(seed=seed, max_ticks=2)).run(
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
        self.assertEqual(dataset.header["provenance"]["source_seeds"], [7])
        self.assertEqual(dataset.header["provenance"]["split_id"], "tiny_train")
        self.assertGreater(dataset.record_count, 0)
        self.assertEqual(dataset.footer["type"], "footer")
        self.assertEqual(
            dataset.footer["trajectory_summary"]["record_count"],
            dataset.record_count,
        )
        self.assertEqual(
            dataset.footer["provenance"]["record_count"],
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
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            write_model_artifact(artifact_path, baseline.to_artifact())

            with self.assertRaises(MindArtifactError):
                load_learned_policy(artifact_path)
            policy = load_learned_policy(artifact_path, enable_mind=True)

        self.assertEqual(policy.policy_id, "mind_v1_learned_policy")
        self.assertTrue(policy.heuristic_guard)

    def test_behavior_cloning_artifact_manifest_requires_provenance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            artifact = baseline.to_artifact()
            del artifact["manifest"]["provenance"]

            with self.assertRaisesRegex(MindArtifactError, "provenance"):
                validate_model_artifact_manifest(artifact)

    def test_mind_train_cli_writes_valid_bc_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            artifact_path = Path(tmpdir) / "bc-artifact.json"
            self._write_tiny_trajectory(trajectory_path)

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_train",
                        "--trajectory",
                        str(trajectory_path),
                        "--output",
                        str(artifact_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
            ):
                mind_train.main()

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)
            self.assertEqual(
                artifact["manifest"]["provenance"]["record_count"],
                artifact["manifest"]["trained_record_count"],
            )

    def test_mind_train_cli_accepts_seed_bank_trajectories(self) -> None:
        with TemporaryDirectory() as tmpdir:
            first_trajectory_path = Path(tmpdir) / "trajectory-seed7.jsonl.gz"
            second_trajectory_path = Path(tmpdir) / "trajectory-seed8.jsonl.gz"
            artifact_path = Path(tmpdir) / "bc-seed-bank-artifact.json"
            self._write_tiny_trajectory(first_trajectory_path, seed=7)
            self._write_tiny_trajectory(second_trajectory_path, seed=8)

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_train",
                        "--trajectory",
                        str(first_trajectory_path),
                        "--trajectory",
                        str(second_trajectory_path),
                        "--output",
                        str(artifact_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
            ):
                mind_train.main()

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)
            provenance = artifact["manifest"]["provenance"]
            self.assertEqual(provenance["source_seeds"], [7, 8])
            self.assertEqual(provenance["source_dataset_count"], 2)
            self.assertEqual(
                provenance["record_count"],
                artifact["manifest"]["trained_record_count"],
            )
            self.assertEqual(
                provenance["trajectory_paths"],
                [str(first_trajectory_path), str(second_trajectory_path)],
            )
            self.assertGreater(len(artifact["model"]["conditional_action_scores"]), 0)

    def test_combined_dataset_provenance_rejects_mismatched_contracts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            first_trajectory_path = Path(tmpdir) / "trajectory-seed7.jsonl.gz"
            second_trajectory_path = Path(tmpdir) / "trajectory-stale.jsonl.gz"
            self._write_tiny_trajectory(first_trajectory_path, seed=7)
            with gzip.open(first_trajectory_path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            rows[-1]["provenance"]["contract_digest"] = "stale"
            with gzip.open(second_trajectory_path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")

            first_dataset = load_trajectory_jsonl(first_trajectory_path)
            second_dataset = load_trajectory_jsonl(second_trajectory_path)

            with self.assertRaisesRegex(TrajectoryDatasetError, "contract digests"):
                combined_dataset_provenance([first_dataset, second_dataset])

    def test_learned_policy_obeys_action_mask(self) -> None:
        policy = LearnedPolicy(action_scores={"eat": 1.0, "stay": 0.1})

        blocked = policy.decide({}, {"stay": True, "eat": False})
        allowed = policy.decide({}, {"stay": True, "eat": True})

        self.assertEqual(blocked.requested_action, "stay")
        self.assertEqual(allowed.requested_action, "eat")

    def test_learned_policy_uses_contextual_scores_before_global_prior(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.2,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
            },
            "local_patch": [
                {"dx": 0, "dy": 0, "food": 0.2, "fresh_kill_energy": 0.0, "carcass_energy": 0.0}
            ],
            "navigation": {},
        }
        action_mask = {"stay": True, "eat": True}
        contextual_key = feature_keys_from_observation(observation, action_mask)[0]
        policy = LearnedPolicy(
            action_scores={"eat": 1.0, "stay": 0.1},
            conditional_action_scores={contextual_key: {"eat": 0.0, "stay": 1.0}},
        )

        decision = policy.decide(observation, action_mask)

        self.assertEqual(decision.requested_action, "stay")

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
        self.assertIn("heuristic", report)
        self.assertIn("learned", report)
        json.dumps(report)

    def test_mind_gate_uses_policy_visible_invalid_action_rate(self) -> None:
        report = {
            "runs": [
                {
                    "alive_agents": 1,
                    "resource_pressure": {
                        "plant_budget": {"energy_available_at_end": 1.0},
                    },
                    "land_tile_count": 1,
                }
            ],
            "aggregate": {
                "births": {"mean": 0.0},
                "trajectory": {
                    "invalid_action_rate": 0.0,
                    "invalid_observation_action_rate": 0.0,
                    "invalid_resolution_action_rate": 0.5,
                },
            },
        }

        gate = build_mind_v1_gate_report(report)

        self.assertEqual(gate["status"], "pass")


if __name__ == "__main__":
    unittest.main()
