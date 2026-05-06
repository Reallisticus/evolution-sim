from __future__ import annotations

import gzip
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_gate, mind_train
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.policy import ActionDecision
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
from evolution_sim.mind.diagnostics import (
    build_artifact_diagnostics,
    build_policy_diagnostics,
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
        self.assertEqual(policy.heuristic_override_min_margin, 1.0)
        self.assertTrue(policy.heuristic_delegate)
        self.assertEqual(policy.heuristic_delegate_max_training_score_margin, 0.221)
        self.assertIsNone(policy.heuristic_safe_local_eat_min_score)
        self.assertIsNone(policy.heuristic_safe_local_eat_min_food)
        self.assertIsNone(policy.heuristic_safe_local_eat_min_plant_ratio)
        self.assertIsNone(policy.heuristic_safe_plant_move_min_score)
        self.assertIsNone(policy.heuristic_safe_plant_move_min_strength)
        self.assertIsNone(policy.heuristic_safe_plant_move_max_local_food_ratio)
        self.assertIsNone(policy.heuristic_safe_plant_move_max_distance)

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

    def test_behavior_cloning_artifact_rejects_incomplete_action_scores(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            artifact = baseline.to_artifact()
            del artifact["model"]["action_scores"]["eat"]

            with self.assertRaisesRegex(MindArtifactError, "action_scores"):
                validate_model_artifact_manifest(artifact)

    def test_behavior_cloning_artifact_requires_explicit_safe_deviation_fields(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            artifact = baseline.to_artifact()
            del artifact["model"]["heuristic_safe_local_eat_min_score"]

            with self.assertRaisesRegex(
                MindArtifactError,
                "heuristic_safe_local_eat_min_score",
            ):
                validate_model_artifact_manifest(artifact)

    def test_behavior_cloning_artifact_requires_score_policy_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            artifact = baseline.to_artifact()
            del artifact["model"]["conditional_score_policy"]

            with self.assertRaisesRegex(
                MindArtifactError,
                "conditional_score_policy",
            ):
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
            self.assertIn(
                "heuristic_override_min_margin",
                artifact["model"],
            )
            self.assertEqual(artifact["model"]["heuristic_override_min_margin"], 1.0)
            self.assertEqual(
                artifact["model"]["heuristic_delegate_policy"],
                "observation_heuristic_confidence_delegate_v1",
            )
            self.assertEqual(
                artifact["model"]["heuristic_delegate_max_training_score_margin"],
                0.221,
            )
            self.assertEqual(
                artifact["model"]["conditional_score_policy"],
                "smoothed_contextual_action_prior_v1",
            )
            self.assertAlmostEqual(
                artifact["model"]["conditional_prior_correction_exponent"],
                0.0,
            )
            self.assertAlmostEqual(
                artifact["model"]["conditional_score_smoothing_alpha"],
                0.1,
            )
            self.assertIsNone(artifact["model"]["heuristic_safe_local_eat_min_score"])
            self.assertIsNone(artifact["model"]["heuristic_safe_plant_move_min_score"])
            self.assertEqual(
                artifact["model"]["action_score_metadata"]["record_count"],
                artifact["manifest"]["trained_record_count"],
            )
            self.assertEqual(
                artifact["manifest"]["provenance"]["record_count"],
                artifact["manifest"]["trained_record_count"],
            )

    def test_behavior_cloning_artifact_diagnostics_report_imitation_and_coverage(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )

            artifact = baseline.to_artifact()
            diagnostics = build_artifact_diagnostics(artifact, [dataset])

        self.assertIn("action_score_metadata", artifact["model"])
        self.assertIn("conditional_action_metadata", artifact["model"])
        self.assertEqual(
            artifact["model"]["action_score_metadata"]["record_count"],
            dataset.record_count,
        )
        self.assertEqual(diagnostics["record_count"], dataset.record_count)
        self.assertGreaterEqual(diagnostics["imitation"]["top1_accuracy"], 0.0)
        self.assertLessEqual(diagnostics["imitation"]["top1_accuracy"], 1.0)
        self.assertEqual(
            sum(diagnostics["action_distribution"]["label_counts"].values()),
            dataset.record_count,
        )
        self.assertEqual(
            sum(diagnostics["action_distribution"]["predicted_counts"].values()),
            dataset.record_count,
        )
        self.assertGreaterEqual(
            diagnostics["action_distribution"]["prediction_label_tvd"],
            0.0,
        )
        self.assertLessEqual(
            diagnostics["action_distribution"]["prediction_label_tvd"],
            1.0,
        )
        confusion = diagnostics["action_distribution"]["confusion"]
        self.assertIn("matrix", confusion)
        self.assertIn("top_misclassifications", confusion)
        self.assertIn("per_action", confusion)
        self.assertEqual(
            sum(
                sum(row.values())
                for row in confusion["matrix"].values()
            ),
            dataset.record_count,
        )
        self.assertIn("eat", confusion["per_action"])
        self.assertIn("precision", confusion["per_action"]["eat"])
        self.assertIn("recall", confusion["per_action"]["eat"])
        self.assertIn("match_depth_counts", diagnostics["contextual_coverage"])
        self.assertEqual(
            sum(diagnostics["contextual_coverage"]["match_depth_counts"].values()),
            dataset.record_count,
        )
        self.assertIn("support_bucket_counts", diagnostics["contextual_coverage"])
        self.assertIn(
            "score_margin_bucket_counts",
            diagnostics["contextual_coverage"],
        )
        self.assertGreaterEqual(
            diagnostics["contextual_coverage"]["matched_record_rate"],
            0.0,
        )

    def test_mind_gate_cli_writes_seed_bank_report_and_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mind-gate-report.json"
            artifact_path = tmp_path / "mind-gate-artifact.json"
            trajectory_dir = tmp_path / "trajectories"

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_gate",
                        "--train-seed",
                        "7",
                        "--validation-seed",
                        "8",
                        "--ticks",
                        "2",
                        "--trajectory-dir",
                        str(trajectory_dir),
                        "--artifact-output",
                        str(artifact_path),
                        "--output",
                        str(report_path),
                        "--max-alive-agents-mean-regression",
                        "0.5",
                        "--max-births-mean-regression",
                        "0.25",
                        "--max-invalid-action-rate",
                        "0.03",
                        "--min-viable-run-share",
                        "0.5",
                        "--min-births-per-run-mean",
                        "0.0",
                        "--min-plant-energy-available-per-land-tile",
                        "0.0",
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_gate.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)
            self.assertTrue(report["complete"])
            self.assertEqual(report["protocol"]["train_seeds"], [7])
            self.assertEqual(report["protocol"]["validation_seeds"], [8])
            self.assertEqual(
                report["protocol"]["criteria"],
                {
                    "max_alive_agents_mean_regression": 0.5,
                    "max_alive_agents_per_seed_regression": 2.0,
                    "max_births_mean_regression": 0.25,
                    "max_births_per_seed_regression": 1.0,
                    "max_invalid_action_rate": 0.03,
                    "max_guard_intervention_rate": 0.45,
                    "max_guard_intervention_rate_by_group": 0.5,
                    "min_guard_intervention_rate_reduction": 0.0,
                    "min_viable_run_share": 0.5,
                    "min_births_per_run_mean": 0.0,
                    "min_plant_energy_available_per_land_tile": 0.0,
                },
            )
            self.assertEqual(
                report["evaluation"]["protocol"]["gate_criteria"],
                report["protocol"]["criteria"],
            )
            self.assertEqual(report["artifact"]["path"], str(artifact_path))
            self.assertGreater(report["artifact"]["trained_record_count"], 0)
            self.assertEqual(
                report["artifact"]["trained_record_count"],
                artifact["manifest"]["trained_record_count"],
            )
            self.assertEqual(report["artifact"]["conditional_min_records"], 3)
            self.assertEqual(
                report["artifact"]["conditional_score_policy"],
                artifact["model"]["conditional_score_policy"],
            )
            self.assertEqual(
                report["artifact"]["conditional_prior_correction_exponent"],
                artifact["model"]["conditional_prior_correction_exponent"],
            )
            self.assertEqual(
                report["artifact"]["conditional_score_smoothing_alpha"],
                artifact["model"]["conditional_score_smoothing_alpha"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_confidence_threshold"],
                artifact["model"]["heuristic_confidence_threshold"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_override_min_margin"],
                artifact["model"]["heuristic_override_min_margin"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_delegate_policy"],
                artifact["model"]["heuristic_delegate_policy"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_delegate_max_training_score_margin"],
                artifact["model"]["heuristic_delegate_max_training_score_margin"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_safe_local_eat_min_score"],
                artifact["model"]["heuristic_safe_local_eat_min_score"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_safe_plant_move_min_score"],
                artifact["model"]["heuristic_safe_plant_move_min_score"],
            )
            self.assertEqual(
                report["artifact_diagnostics"]["train"]["record_count"],
                report["artifact"]["trained_record_count"],
            )
            self.assertGreater(
                report["artifact_diagnostics"]["held_out"]["record_count"],
                0,
            )
            self.assertIn("imitation", report["artifact_diagnostics"]["train"])
            self.assertIn("imitation", report["artifact_diagnostics"]["held_out"])
            self.assertIn(
                "contextual_coverage",
                report["artifact_diagnostics"]["train"],
            )
            self.assertIn(report["readiness"]["status"], {"pass", "review", "fail"})
            self.assertEqual(len(report["trajectory_collection"]), 1)
            self.assertEqual(len(report["artifact_diagnostic_collection"]), 1)

    def test_mind_gate_rejects_reused_trajectory_seed_mismatch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_dir = tmp_path / "trajectories"
            trajectory_dir.mkdir()
            mislabeled_path = trajectory_dir / "seed8-ticks2.jsonl.gz"
            self._write_tiny_trajectory(mislabeled_path, seed=7)

            with self.assertRaisesRegex(ValueError, "seed 8"):
                mind_gate.run_mind_gate(
                    train_seeds=[8],
                    validation_seeds=[9],
                    ticks=2,
                    validation_ticks=[2],
                    trajectory_dir=trajectory_dir,
                    artifact_output=tmp_path / "artifact.json",
                    report_output=tmp_path / "report.json",
                    reuse_trajectories=True,
                )

    def test_mind_gate_default_validation_seeds_are_held_out_seed_bank(self) -> None:
        self.assertEqual(mind_gate.DEFAULT_VALIDATION_SEEDS, (5, 13, 19, 29))
        self.assertTrue(
            set(mind_gate.DEFAULT_VALIDATION_SEEDS).isdisjoint(
                mind_gate.DEFAULT_TRAIN_SEEDS
            )
        )

    def test_mind_extended_gate_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))
        script = package["scripts"]["sim:mind:gate:extended"]

        self.assertIn("evolution_sim.cli.mind_gate", script)
        self.assertIn("--validation-seeds 5,13,19,29,37,41", script)
        self.assertIn("--validation-ticks 120,180", script)
        self.assertIn("output/mind/mind-v1-gate-extended-report.json", script)

    def test_mind_gate_cli_fail_on_blockers_exits_nonzero(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mind-gate-report.json"
            artifact_path = tmp_path / "mind-gate-artifact.json"

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_gate",
                        "--train-seed",
                        "7",
                        "--validation-seed",
                        "8",
                        "--ticks",
                        "2",
                        "--artifact-output",
                        str(artifact_path),
                        "--output",
                        str(report_path),
                        "--min-viable-run-share",
                        "2.0",
                        "--fail-on-blockers",
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    mind_gate.main()

            self.assertEqual(raised.exception.code, 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["readiness"]["status"], "fail")

    def test_mind_gate_cli_evaluates_validation_tick_matrix(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mind-gate-report.json"
            artifact_path = tmp_path / "mind-gate-artifact.json"

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_gate",
                        "--train-seed",
                        "7",
                        "--validation-seed",
                        "8",
                        "--ticks",
                        "2",
                        "--validation-ticks",
                        "2,3",
                        "--artifact-output",
                        str(artifact_path),
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_gate.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["protocol"]["validation_ticks"], [2, 3])
            self.assertEqual(
                [entry["ticks"] for entry in report["evaluation_matrix"]],
                [2, 3],
            )
            self.assertEqual(report["evaluation"]["protocol"]["ticks"], 2)
            self.assertIn(report["readiness"]["status"], {"pass", "review", "fail"})

    def test_mind_gate_cli_fail_on_review_exits_nonzero(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mind-gate-report.json"
            artifact_path = tmp_path / "mind-gate-artifact.json"

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_gate",
                        "--train-seed",
                        "7",
                        "--validation-seed",
                        "8",
                        "--ticks",
                        "2",
                        "--artifact-output",
                        str(artifact_path),
                        "--output",
                        str(report_path),
                        "--min-births-per-run-mean",
                        "999.0",
                        "--max-guard-intervention-rate",
                        "1.0",
                        "--max-guard-intervention-rate-by-group",
                        "1.0",
                        "--fail-on-review",
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    mind_gate.main()

            self.assertEqual(raised.exception.code, 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["readiness"]["status"], "review")

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

    def test_feature_policy_keeps_no_signal_local_patch_tokens_neutral(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
            },
            "local_patch": [
                {"dx": 0, "dy": 0, "terrain": "plain", "food": 0.0},
                {"dx": 1, "dy": 0, "terrain": "plain", "food": 0.0},
                {"dx": -1, "dy": 0, "terrain": "plain", "food": 0.0},
            ],
            "navigation": {},
        }

        feature_key = feature_keys_from_observation(
            observation,
            {"stay": True, "eat": True},
        )[0]

        self.assertIn("foodc0:waterc0:carrionc0:preyc0:riskc0", feature_key)

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

    def test_learned_policy_reports_score_support_and_margin_diagnostics(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "food": 0.2,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                }
            ],
            "navigation": {},
        }
        action_mask = {"stay": True, "eat": True}
        contextual_key = feature_keys_from_observation(observation, action_mask)[0]
        policy = LearnedPolicy(
            action_scores={"eat": 0.2, "stay": 0.8},
            action_score_metadata={
                "record_count": 100,
                "score_margin": 0.6,
            },
            conditional_action_scores={contextual_key: {"eat": 0.75, "stay": 0.25}},
            conditional_action_metadata={
                contextual_key: {
                    "record_count": 14,
                    "score_margin": 0.5,
                }
            },
        )

        decision = policy.decide(observation, action_mask)

        self.assertEqual(decision.requested_action, "eat")
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertEqual(diagnostics["score_source"], "conditional")
        self.assertEqual(diagnostics["score_feature_key"], contextual_key)
        self.assertEqual(diagnostics["score_match_depth"], 0)
        self.assertEqual(diagnostics["score_support"], 14)
        self.assertAlmostEqual(float(diagnostics["learned_score_margin"]), 0.5)
        self.assertAlmostEqual(float(diagnostics["training_score_margin"]), 0.5)

    def test_heuristic_guard_conserves_low_energy_agent_before_learned_move(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.42,
                "hydration_ratio": 0.7,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        action_mask = {"stay": True, "move_north": True}
        policy = LearnedPolicy(
            action_scores={"move_north": 1.0, "stay": 0.1},
            heuristic_guard=True,
        )

        decision = policy.decide(observation, action_mask)

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)

    def test_heuristic_guard_defers_low_confidence_learned_action(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_north": 0.42, "stay": 0.4},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
        )

        decision = policy.decide(observation, {"stay": True, "move_north": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)

    def test_heuristic_guard_defers_small_margin_learned_disagreement(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_north": 0.58, "stay": 0.52},
            heuristic_guard=True,
            heuristic_override_min_margin=0.1,
        )

        decision = policy.decide(observation, {"stay": True, "move_north": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)

    def test_heuristic_delegate_handles_low_margin_action_prior(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_north": 0.58, "stay": 0.52},
            action_score_metadata={"record_count": 20, "score_margin": 0.1},
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=0.1,
        )

        decision = policy.decide(observation, {"stay": True, "move_north": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn(
            "observation_heuristic_confidence_delegate_v1",
            decision.source,
        )
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        self.assertTrue(decision.diagnostics["heuristic_delegate_used"])
        self.assertEqual(
            decision.diagnostics["heuristic_delegate_reason"],
            "low_confidence_action_prior",
        )

    def test_heuristic_guard_preserves_explicit_conservation(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.7,
                "hydration_ratio": 0.8,
                "health_ratio": 1.0,
                "trophic_role": "carnivore",
                "meat_mode": "hunter",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_north": 1.0, "stay": 0.0},
            heuristic_guard=True,
        )

        decision = policy.decide(observation, {"stay": True, "move_north": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)

    def test_heuristic_guard_allows_safe_local_eat_deviation(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.28,
                    "vegetation": 0.3,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"eat": 0.3, "move_east": 0.2, "stay": 0.0},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            heuristic_safe_local_eat_min_score=0.25,
            heuristic_safe_local_eat_min_food=0.25,
            heuristic_safe_local_eat_min_plant_ratio=0.5,
        )

        decision = policy.decide(
            observation,
            {"eat": True, "move_east": True, "stay": True},
        )

        self.assertEqual(decision.requested_action, "eat")
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        self.assertTrue(decision.diagnostics["safe_deviation_used"])
        self.assertEqual(
            decision.diagnostics["safe_deviation_reason"],
            "local_resource_eat",
        )
        self.assertEqual(decision.diagnostics["heuristic_action"], "move_east")

    def test_heuristic_guard_rejects_inferior_local_eat_deviation(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.28,
                    "vegetation": 0.3,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"eat": 0.3, "move_east": 0.2, "stay": 0.0},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            heuristic_safe_local_eat_min_score=0.25,
            heuristic_safe_local_eat_min_food=0.25,
            heuristic_safe_local_eat_min_plant_ratio=1.0,
        )

        decision = policy.decide(
            observation,
            {"eat": True, "move_east": True, "stay": True},
        )

        self.assertEqual(decision.requested_action, "move_east")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        self.assertFalse(decision.diagnostics["safe_deviation_used"])

    def test_heuristic_guard_rejects_meat_specialist_plant_only_local_eat_deviation(
        self,
    ) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "carnivore",
                "meat_mode": "hunter",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.45,
                    "vegetation": 0.45,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"eat": 0.3, "move_east": 0.2, "stay": 0.0},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            heuristic_safe_local_eat_min_score=0.25,
            heuristic_safe_local_eat_min_food=0.25,
            heuristic_safe_local_eat_min_plant_ratio=0.5,
        )

        decision = policy.decide(
            observation,
            {"eat": True, "move_east": True, "stay": True},
        )

        self.assertNotEqual(decision.requested_action, "eat")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        self.assertFalse(decision.diagnostics["safe_deviation_used"])

    def test_heuristic_guard_allows_safe_plant_move_deviation(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.82,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.25,
                    "vegetation": 0.3,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_east": 0.4, "eat": 0.3, "stay": 0.0},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            heuristic_safe_plant_move_min_score=0.4,
            heuristic_safe_plant_move_min_strength=0.3,
            heuristic_safe_plant_move_max_local_food_ratio=0.9,
            heuristic_safe_plant_move_max_distance=3,
        )

        decision = policy.decide(
            observation,
            {"eat": True, "move_east": True, "stay": True},
        )

        self.assertEqual(decision.requested_action, "move_east")
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        self.assertTrue(decision.diagnostics["safe_deviation_used"])
        self.assertEqual(
            decision.diagnostics["safe_deviation_reason"],
            "stronger_plant_navigation",
        )

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
        self.assertIn("by_trophic_role", report["comparison"])
        self.assertIn("by_meat_mode", report["comparison"])
        self.assertIn(
            "policy_diagnostics_by_trophic_role",
            report["comparison"],
        )
        self.assertIn(
            "policy_diagnostics_by_meat_mode",
            report["comparison"],
        )
        role_comparison = report["comparison"]["by_trophic_role"]
        self.assertGreaterEqual(len(role_comparison), 1)
        role_delta = next(iter(role_comparison.values()))
        self.assertEqual(
            role_delta["total_delta"],
            role_delta["learned_total"] - role_delta["heuristic_total"],
        )
        mode_comparison = report["comparison"]["policy_diagnostics_by_meat_mode"]
        self.assertGreaterEqual(len(mode_comparison), 1)
        mode_delta = next(iter(mode_comparison.values()))
        self.assertIn("mean_reward_delta", mode_delta)
        self.assertIn("guard_intervention_rate_delta", mode_delta)
        learned_diagnostics = report["learned"]["aggregate"]["policy_diagnostics"]
        self.assertIn("guard_intervention_rate", learned_diagnostics)
        self.assertIn("heuristic_delegate_rate", learned_diagnostics)
        self.assertIn("action_source_counts", learned_diagnostics)
        self.assertIn("by_trophic_role", learned_diagnostics)
        self.assertIn("by_meat_mode", learned_diagnostics)
        self.assertIn("heuristic_delegate_by_score_source", learned_diagnostics)
        self.assertIn("heuristic_delegate_by_support_bucket", learned_diagnostics)
        self.assertIn("heuristic_delegate_by_score_margin_bucket", learned_diagnostics)
        self.assertIn("top_delegated_contexts", learned_diagnostics)
        json.dumps(report)

    def test_mind_evaluation_uses_public_reporting_helpers(self) -> None:
        source = Path("python/evolution_sim/mind/evaluation.py").read_text(
            encoding="utf-8",
        )

        self.assertNotIn("from evolution_sim.cli.evaluate import _", source)

    def test_policy_diagnostics_report_guard_actions_and_contexts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        records = [dict(dataset.records[0]), dict(dataset.records[1])]
        records[0]["action_source"] = (
            "mind_v1_learned_policy:observation_heuristic_safety_floor_v1"
        )
        records[1]["action_source"] = "mind_v1_learned_policy"

        diagnostics = build_policy_diagnostics(records)

        guarded_action = str(records[0]["requested_action"])
        self.assertEqual(diagnostics["guard_intervention_count"], 1)
        self.assertIn(guarded_action, diagnostics["guard_intervention_by_action"])
        self.assertEqual(
            diagnostics["guard_intervention_by_action"][guarded_action][
                "guard_intervention_count"
            ],
            1,
        )
        self.assertGreaterEqual(len(diagnostics["top_guarded_contexts"]), 1)
        top_context = diagnostics["top_guarded_contexts"][0]
        self.assertIn("feature_key", top_context)
        self.assertEqual(top_context["guard_intervention_count"], 1)
        self.assertIn("action_counts", top_context)

    def test_policy_diagnostics_report_heuristic_delegate_share(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        record = dict(dataset.records[0])
        record["action_source"] = (
            "mind_v1_learned_policy:"
            "observation_heuristic_confidence_delegate_v1"
        )

        diagnostics = build_policy_diagnostics(
            [record],
            decision_diagnostics=[
                {
                    "heuristic_delegate_used": True,
                    "learned_action": "eat",
                    "heuristic_action": record["requested_action"],
                    "score_source": "conditional",
                    "score_support": 0,
                    "learned_score_margin": 0.12,
                }
            ],
        )

        requested_action = str(record["requested_action"])
        self.assertEqual(diagnostics["guard_intervention_count"], 0)
        self.assertEqual(diagnostics["heuristic_delegate_count"], 1)
        self.assertEqual(diagnostics["heuristic_delegate_rate"], 1.0)
        self.assertEqual(
            diagnostics["heuristic_delegate_by_action"][requested_action][
                "heuristic_delegate_count"
            ],
            1,
        )
        self.assertEqual(
            diagnostics["heuristic_delegate_suppressed_learned_action"]["eat"][
                "heuristic_delegate_count"
            ],
            1,
        )
        self.assertEqual(
            diagnostics["heuristic_delegate_by_score_source"]["conditional"][
                "heuristic_delegate_count"
            ],
            1,
        )
        self.assertEqual(
            diagnostics["heuristic_delegate_by_support_bucket"]["0"][
                "heuristic_delegate_count"
            ],
            1,
        )
        self.assertEqual(
            diagnostics["heuristic_delegate_by_score_margin_bucket"]["0.10-0.24"][
                "heuristic_delegate_count"
            ],
            1,
        )
        self.assertGreaterEqual(len(diagnostics["top_delegated_contexts"]), 1)
        top_context = diagnostics["top_delegated_contexts"][0]
        self.assertIn("feature_key", top_context)
        self.assertEqual(top_context["heuristic_delegate_count"], 1)
        self.assertIn("action_counts", top_context)
        self.assertEqual(
            sum(
                int(group["heuristic_delegate_count"])
                for group in diagnostics["by_trophic_role"].values()
            ),
            1,
        )
        self.assertEqual(
            sum(
                int(group["heuristic_delegate_count"])
                for group in diagnostics["by_meat_mode"].values()
            ),
            1,
        )

    def test_policy_evaluation_reports_guard_suppressed_learned_actions(self) -> None:
        class AlwaysGuardedPolicy:
            policy_id = "mind_v1_learned_policy"
            policy_version = "mind_v1_learned_policy_test"

            def decide(
                self,
                observation: dict[str, object],
                action_mask: dict[str, bool],
            ) -> ActionDecision:
                return ActionDecision(
                    requested_action="stay",
                    source=(
                        "mind_v1_learned_policy:"
                        "observation_heuristic_safety_floor_v1"
                    ),
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    diagnostics={
                        "guard_used": True,
                        "learned_action": "eat",
                        "heuristic_action": "stay",
                        "score_source": "conditional",
                        "score_support": 14,
                        "learned_score_margin": 0.5,
                    },
                )

        report = compare_heuristic_and_learned(
            learned_policy=AlwaysGuardedPolicy(),
            seeds=[7],
            ticks=1,
        )

        diagnostics = report["learned"]["aggregate"]["policy_diagnostics"]
        self.assertIn("guard_suppressed_learned_action", diagnostics)
        self.assertEqual(
            diagnostics["guard_suppressed_learned_action"]["eat"][
                "guard_intervention_count"
            ],
            diagnostics["guard_intervention_count"],
        )
        self.assertEqual(
            diagnostics["guard_intervention_by_score_source"]["conditional"][
                "guard_intervention_count"
            ],
            diagnostics["guard_intervention_count"],
        )
        self.assertEqual(
            diagnostics["guard_intervention_by_support_bucket"]["10-31"][
                "guard_intervention_count"
            ],
            diagnostics["guard_intervention_count"],
        )
        self.assertEqual(
            diagnostics["guard_intervention_by_score_margin_bucket"]["0.50-0.99"][
                "guard_intervention_count"
            ],
            diagnostics["guard_intervention_count"],
        )
        self.assertNotIn(
            "policy_decision_diagnostics",
            report["learned"]["runs"][0]["trajectory"],
        )

    def test_policy_diagnostics_separates_zero_support_bucket(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        record = dict(dataset.records[0])
        record["action_source"] = (
            "mind_v1_learned_policy:observation_heuristic_safety_floor_v1"
        )
        record["policy_decision_diagnostics"] = {
            "guard_used": True,
            "score_support": 0,
        }

        diagnostics = build_policy_diagnostics(
            [record],
            decision_diagnostics=[record["policy_decision_diagnostics"]],
        )

        self.assertIn("0", diagnostics["guard_intervention_by_support_bucket"])
        self.assertNotIn("1-9", diagnostics["guard_intervention_by_support_bucket"])

    def test_policy_diagnostics_reports_safe_deviation_share(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        record = dict(dataset.records[0])
        record["action_source"] = "mind_v1_learned_policy"
        diagnostic = {
            "guard_used": False,
            "safe_deviation_used": True,
            "learned_action": "eat",
        }

        diagnostics = build_policy_diagnostics(
            [record],
            decision_diagnostics=[diagnostic],
        )

        self.assertEqual(diagnostics["safe_deviation_count"], 1)
        self.assertEqual(diagnostics["safe_deviation_rate"], 1.0)
        self.assertEqual(
            diagnostics["safe_deviation_by_action"]["eat"]["safe_deviation_count"],
            1,
        )

    def test_policy_evaluation_reports_paired_seed_deltas(self) -> None:
        policy = LearnedPolicy(action_scores={"stay": 1.0})

        report = compare_heuristic_and_learned(
            learned_policy=policy,
            seeds=[7],
            ticks=2,
        )

        per_seed = report["comparison"]["per_seed"]
        self.assertEqual(len(per_seed), 1)
        self.assertEqual(per_seed[0]["seed"], 7)
        self.assertEqual(
            per_seed[0]["alive_agents_delta"],
            per_seed[0]["learned_alive_agents"] - per_seed[0]["heuristic_alive_agents"],
        )
        self.assertEqual(
            per_seed[0]["births_delta"],
            per_seed[0]["learned_births"] - per_seed[0]["heuristic_births"],
        )

    def test_policy_evaluation_applies_mind_gate_criteria(self) -> None:
        policy = LearnedPolicy(action_scores={"stay": 1.0})

        report = compare_heuristic_and_learned(
            learned_policy=policy,
            seeds=[7],
            ticks=2,
            gate_criteria={"min_viable_run_share": 2.0},
        )

        self.assertEqual(report["protocol"]["gate_criteria"]["min_viable_run_share"], 2.0)
        self.assertEqual(report["mind_v1_gates"]["status"], "fail")
        self.assertEqual(
            report["mind_v1_gates"]["blockers"][0]["field"],
            "alive_agents",
        )

    def test_mind_gate_blocks_per_seed_alive_regression(self) -> None:
        learned_report = {
            "runs": [
                {
                    "seed": 1,
                    "alive_agents": 7,
                    "births": 8,
                    "land_tile_count": 1,
                    "resource_pressure": {
                        "plant_budget": {"energy_available_at_end": 1.0},
                    },
                }
            ],
            "aggregate": {
                "births": {"mean": 8.0},
                "trajectory": {
                    "invalid_observation_action_rate": 0.0,
                },
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.0,
                    "by_trophic_role": {},
                    "by_meat_mode": {},
                },
            },
        }
        heuristic_report = {
            "runs": [{"seed": 1, "alive_agents": 10, "births": 8}],
            "aggregate": {"alive_agents": {"mean": 10.0}, "births": {"mean": 8.0}},
        }

        gate = build_mind_v1_gate_report(
            learned_report,
            baseline_report=heuristic_report,
            max_alive_agents_mean_regression=10.0,
            max_alive_agents_per_seed_regression=2.0,
        )

        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["blockers"][0]["field"],
            "alive_agents.per_seed_delta",
        )

    def test_mind_gate_warns_on_per_seed_birth_regression(self) -> None:
        learned_report = {
            "runs": [
                {
                    "seed": 1,
                    "alive_agents": 10,
                    "births": 5,
                    "land_tile_count": 1,
                    "resource_pressure": {
                        "plant_budget": {"energy_available_at_end": 1.0},
                    },
                }
            ],
            "aggregate": {
                "births": {"mean": 5.0},
                "trajectory": {
                    "invalid_observation_action_rate": 0.0,
                },
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.0,
                    "by_trophic_role": {},
                    "by_meat_mode": {},
                },
            },
        }
        heuristic_report = {
            "runs": [{"seed": 1, "alive_agents": 10, "births": 7}],
            "aggregate": {"alive_agents": {"mean": 10.0}, "births": {"mean": 7.0}},
        }

        gate = build_mind_v1_gate_report(
            learned_report,
            baseline_report=heuristic_report,
            max_births_mean_regression=10.0,
            max_births_per_seed_regression=1.0,
        )

        self.assertEqual(gate["status"], "review")
        self.assertEqual(
            gate["warnings"][0]["field"],
            "births.per_seed_delta",
        )

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
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.0,
                    "by_trophic_role": {},
                    "by_meat_mode": {},
                },
            },
        }

        gate = build_mind_v1_gate_report(report)

        self.assertEqual(gate["status"], "pass")

    def test_mind_gate_blocks_high_guard_intervention_rate(self) -> None:
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
                "births": {"mean": 1.0},
                "trajectory": {"invalid_observation_action_rate": 0.0},
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.75,
                    "by_trophic_role": {
                        "herbivore": {"guard_intervention_rate": 0.75}
                    },
                    "by_meat_mode": {
                        "none": {"guard_intervention_rate": 0.75}
                    },
                },
            },
        }

        gate = build_mind_v1_gate_report(
            report,
            max_guard_intervention_rate=0.5,
            max_guard_intervention_rate_by_group=0.6,
        )

        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["blockers"][0]["field"],
            "policy_diagnostics.guard_intervention_rate",
        )

    def test_mind_gate_blocks_guard_group_cap(self) -> None:
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
                "births": {"mean": 1.0},
                "trajectory": {"invalid_observation_action_rate": 0.0},
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.4,
                    "by_trophic_role": {
                        "carnivore": {"guard_intervention_rate": 0.6}
                    },
                    "by_meat_mode": {
                        "hunter": {"guard_intervention_rate": 0.4}
                    },
                },
            },
        }

        gate = build_mind_v1_gate_report(
            report,
            max_guard_intervention_rate=0.45,
            max_guard_intervention_rate_by_group=0.5,
        )

        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["blockers"][0]["field"],
            "policy_diagnostics.by_trophic_role.carnivore.guard_intervention_rate",
        )

    def test_mind_gate_blocks_insufficient_guard_reduction(self) -> None:
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
                "births": {"mean": 1.0},
                "trajectory": {"invalid_observation_action_rate": 0.0},
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.42,
                    "by_trophic_role": {
                        "herbivore": {"guard_intervention_rate": 0.42}
                    },
                    "by_meat_mode": {
                        "none": {"guard_intervention_rate": 0.42}
                    },
                },
            },
        }

        gate = build_mind_v1_gate_report(
            report,
            max_guard_intervention_rate=1.0,
            max_guard_intervention_rate_by_group=1.0,
            min_guard_intervention_rate_reduction=0.1,
            reference_guard_intervention_rate=0.45,
        )

        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["blockers"][0]["field"],
            "policy_diagnostics.guard_intervention_rate_reduction",
        )


if __name__ == "__main__":
    unittest.main()
