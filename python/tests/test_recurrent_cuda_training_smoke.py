from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.cli.mind_v3_public_recurrent_ippo_cuda_training_smoke import (
        build_parser,
    )
    from evolution_sim.mind.recurrent_counterfactual_comparison import EXACT_ARM
    from evolution_sim.mind.recurrent_cuda_training_smoke import (
        RECURRENT_CUDA_TRAINING_SMOKE_BUNDLE_COUNT,
        RECURRENT_CUDA_TRAINING_SMOKE_CONTRACT_VERSION,
        RECURRENT_CUDA_TRAINING_SMOKE_RUN_ID,
        RECURRENT_CUDA_TRAINING_SMOKE_SCHEMA_VERSION,
        RECURRENT_CUDA_TRAINING_SMOKE_UPDATE_COUNT,
        RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT,
        build_recurrent_cuda_training_smoke_contract,
        validate_recurrent_cuda_training_smoke_report,
    )
    from evolution_sim.mind.provenance import stable_payload_digest
    from evolution_sim.mind.recurrent_runtime_provenance import (
        build_recurrent_scale_runtime_provenance,
    )
    from evolution_sim.mind.recurrent_scale_campaign import (
        RecurrentScaleCampaignError,
        build_recurrent_scale_campaign_preregistration,
        load_strict_json,
        write_atomic_json,
    )
    from evolution_sim.mind.recurrent_scale_execution import (
        build_scale_run_components,
    )
    from evolution_sim.mind.recurrent_seed_registry import (
        SCALE_DEVELOPMENT_SEED_REGISTRY,
        SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
        SCALE_DEVELOPMENT_V2_SEED_REGISTRY,
        SCALE_DEVELOPMENT_V2_SEED_REGISTRY_VERSION,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentCudaTrainingSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preregistration = build_recurrent_scale_campaign_preregistration(
            source_commit="1" * 40,
            source_manifest_sha256="2" * 64,
        )
        self.learner_seed = SCALE_DEVELOPMENT_V2_SEED_REGISTRY[
            "scale_v2_learner"
        ][0]

    def test_smoke_contract_uses_full_exact_algorithm_and_only_bounds_workload(
        self,
    ) -> None:
        model, ppo, counterfactual, schedule = build_scale_run_components(
            self.preregistration,
            learner_seed=self.learner_seed,
            arm=EXACT_ARM,
            counterfactual_workers=8,
        )
        self.assertIsNotNone(counterfactual)
        assert counterfactual is not None

        contract = build_recurrent_cuda_training_smoke_contract(
            self.preregistration,
            learner_seed=self.learner_seed,
            rollout_workers=8,
            counterfactual_workers=8,
            evaluation_workers=8,
        )

        self.assertEqual(contract["arm"], EXACT_ARM)
        self.assertEqual(
            contract["schema_version"],
            RECURRENT_CUDA_TRAINING_SMOKE_CONTRACT_VERSION,
        )
        self.assertEqual(
            contract["purpose"],
            "pre_tmux_scale_v2_cuda_training_and_crash_resume_launch_gate",
        )
        self.assertEqual(
            contract["seed_contract"],
            {
                "registry_version": SCALE_DEVELOPMENT_V2_SEED_REGISTRY_VERSION,
                "registry_sha256": SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
                "learner_role": "scale_v2_learner",
            },
        )
        self.assertEqual(
            RECURRENT_CUDA_TRAINING_SMOKE_RUN_ID,
            "cuda-training-smoke-preflight-v2",
        )
        self.assertEqual(
            RECURRENT_CUDA_TRAINING_SMOKE_SCHEMA_VERSION,
            "mind_v3_public_recurrent_ippo_cuda_training_smoke_v2",
        )
        algorithm = contract["algorithm"]
        self.assertIsInstance(algorithm, dict)
        assert isinstance(algorithm, dict)
        self.assertEqual(algorithm["model"], asdict(model))
        self.assertEqual(algorithm["ppo"], asdict(ppo))
        self.assertEqual(
            algorithm["counterfactual_collection"],
            json.loads(json.dumps(asdict(counterfactual.collection))),
        )
        self.assertEqual(
            algorithm["counterfactual_auxiliary"],
            counterfactual.auxiliary.as_contract(),
        )
        self.assertEqual(
            algorithm["counterfactual_step"],
            counterfactual.step.as_contract(),
        )
        workload = contract["bounded_workload"]
        self.assertIsInstance(workload, dict)
        assert isinstance(workload, dict)
        self.assertEqual(workload["warmup_updates_before_checkpoint"], 1)
        self.assertEqual(
            workload["parity_compared_continuation_updates"],
            RECURRENT_CUDA_TRAINING_SMOKE_UPDATE_COUNT,
        )
        self.assertEqual(workload["updates_on_uninterrupted_path"], 2)
        self.assertEqual(
            workload["worlds_per_update"], RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT
        )
        self.assertEqual(
            workload["counterfactual_bundles_per_update"],
            RECURRENT_CUDA_TRAINING_SMOKE_BUNDLE_COUNT,
        )
        self.assertEqual(workload["rollout_ticks"], schedule[0][0].rollout_ticks)
        self.assertTrue(workload["full_multi_tape_count_preserved"])
        self.assertTrue(workload["full_terminal_target_preserved"])
        self.assertEqual(contract, json.loads(json.dumps(contract)))

    def test_smoke_task_uses_only_scale_training_role_and_lifecycle_is_closed(
        self,
    ) -> None:
        contract = build_recurrent_cuda_training_smoke_contract(
            self.preregistration,
            learner_seed=self.learner_seed,
            rollout_workers=1,
            counterfactual_workers=1,
            evaluation_workers=1,
        )
        workload = contract["bounded_workload"]
        isolation = contract["isolation"]
        assert isinstance(workload, dict)
        assert isinstance(isolation, dict)
        for name in ("warmup_task", "continued_task"):
            task = workload[name]
            assert isinstance(task, dict)
            self.assertIn(
                task["seed_role"],
                {"scale_v2_train", "scale_v2_curriculum"},
            )
            self.assertNotIn(
                task["environment_seed"],
                SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_selection"],
            )
        self.assertEqual(
            isolation,
            {
                "campaign_run_id_used": False,
                "campaign_cell_output_used": False,
                "campaign_slice_consumed": False,
                "validation_seeds_accessed": False,
                "lockbox_seeds_accessed": False,
                "runtime_artifact_created": False,
                "promotion_authorized": False,
            },
        )

    def test_v1_scale_learner_seed_is_rejected_by_v2_smoke_contract(self) -> None:
        stale_learner_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
        self.assertNotIn(
            stale_learner_seed,
            SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_learner"],
        )
        with self.assertRaisesRegex(
            RecurrentScaleCampaignError,
            "scale_v2_learner",
        ):
            build_recurrent_cuda_training_smoke_contract(
                self.preregistration,
                learner_seed=stale_learner_seed,
                rollout_workers=1,
                counterfactual_workers=1,
                evaluation_workers=1,
            )

    def test_cli_requires_pinned_inputs_and_explicit_worker_topology(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(
            [
                "--preregistration",
                "preregistration.json",
                "--expected-preregistration-digest",
                "a" * 64,
                "--runtime-provenance",
                "runtime.json",
                "--output-root",
                "preflight/cuda-training-smoke",
                "--device",
                "cuda:0",
                "--rollout-workers",
                "8",
                "--counterfactual-workers",
                "8",
                "--evaluation-workers",
                "8",
            ]
        )
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.rollout_workers, 8)
        self.assertEqual(args.counterfactual_workers, 8)
        self.assertEqual(args.evaluation_workers, 8)

    def test_smoke_report_survives_atomic_write_load_and_validation(self) -> None:
        contract = build_recurrent_cuda_training_smoke_contract(
            self.preregistration,
            learner_seed=self.learner_seed,
            rollout_workers=1,
            counterfactual_workers=1,
            evaluation_workers=1,
        )
        runtime = build_recurrent_scale_runtime_provenance(
            source_commit="1" * 40,
            source_manifest_sha256="2" * 64,
            preregistration_digest=str(self.preregistration["exact_digest"]),
            repository_clean=True,
            device="cpu",
            rollout_workers=1,
            counterfactual_workers=1,
            evaluation_workers=1,
        )
        runtime["device"] = {
            "type": "cuda",
            "index": 0,
            "name": "synthetic-cuda",
            "compute_capability": [8, 9],
            "total_memory_bytes": 12 * 1024**3,
            "multiprocessor_count": 56,
        }
        runtime["nvidia"] = {
            "required": True,
            "driver_version": "595.58.03",
            "cuda_driver_api_version": "13.2",
            "cuda_driver_api_version_raw": 13020,
            "cuda_runtime_api_version": "13.0",
            "cuda_runtime_api_version_raw": 13000,
        }
        runtime["exact_digest"] = stable_payload_digest(
            {key: value for key, value in runtime.items() if key != "exact_digest"}
        )
        initial_digest = "3" * 64
        checkpoint_digest = "9" * 64
        final_digest = "4" * 64
        evidence_digest = "5" * 64
        report: dict[str, object] = {
            "schema_version": RECURRENT_CUDA_TRAINING_SMOKE_SCHEMA_VERSION,
            "kind": "isolated_cuda_training_preflight",
            "preregistration_digest": self.preregistration["exact_digest"],
            "source": {
                "commit": "1" * 40,
                "manifest_sha256": "2" * 64,
                "clean_tree_verified": True,
            },
            "runtime_provenance": {
                "exact_digest": runtime["exact_digest"],
                "path": "/synthetic/runtime-provenance.json",
                "file_sha256": "6" * 64,
            },
            "contract": contract,
            "device": "cuda:0",
            "checkpoint": {
                "schema_version": "synthetic-checkpoint-v1",
                "checkpoint_sha256": "7" * 64,
                "path": "/synthetic/mid-training-crash-checkpoint.json",
                "file_sha256": "8" * 64,
                "completed_updates": 1,
                "model_loaded_exactly": True,
                "optimizer_moments_nonempty_and_loaded": True,
                "rng_loaded": True,
                "trainer_update_index_restored": 1,
                "auxiliary_update_count_restored": 1,
                "attempted_auxiliary_bundle_ledger_count": 1,
                "attempted_auxiliary_bundle_ledger_restored": True,
                "runtime_provenance_bound": True,
            },
            "training": {
                "real_simulator_interactions": True,
                "warmup_updates_before_checkpoint": 1,
                "parity_compared_continuation_updates": 1,
                "worlds_per_update": 1,
                "warmup_rollout_transitions": 64,
                "continued_rollout_transitions": 64,
                "continued_ppo_parameter_delta_l2": 0.1,
                "counterfactual": {
                    "aggregate_multi_tape": True,
                    "bundle_count": 1,
                    "continuation_tape_count": self.preregistration["counterfactual"][
                        "independent_rng_tapes_per_branch"
                    ],
                    "terminal_target_world_tick": self.preregistration[
                        "counterfactual"
                    ]["absolute_terminal_target_tick"],
                    "auxiliary_step_accepted": True,
                    "optimizer_step_count": 1,
                    "backward_pass_count": 1,
                    "parameter_delta_l2": 0.01,
                    "mean_behavior_kl_old_to_post": 0.001,
                    "max_state_behavior_kl_old_to_post": 0.002,
                },
            },
            "resume_parity": {
                "initial_model_sha256": initial_digest,
                "checkpoint_model_sha256": checkpoint_digest,
                "uninterrupted_final_model_sha256": final_digest,
                "resumed_final_model_sha256": final_digest,
                "model_digest_exact_match": True,
                "uninterrupted_update_evidence_sha256": evidence_digest,
                "resumed_update_evidence_sha256": evidence_digest,
                "update_evidence_exact_match": True,
            },
            "lifecycle": {
                "preflight_only": True,
                "campaign_cell_output_used": False,
                "campaign_slice_consumed": False,
                "validation_seeds_accessed": False,
                "lockbox_seeds_accessed": False,
                "runtime_artifact_created": False,
                "runtime_action_selection_changed": False,
                "promotion_authorized": False,
                "gate_relaxation_authorized": False,
            },
        }
        report["exact_digest"] = stable_payload_digest(report)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            write_atomic_json(path, report)
            loaded = load_strict_json(path)
        self.assertEqual(loaded, report)
        validate_recurrent_cuda_training_smoke_report(
            loaded,
            preregistration=self.preregistration,
            runtime_provenance=runtime,
        )

        stale_schema = copy.deepcopy(report)
        stale_schema["schema_version"] = (
            "mind_v3_public_recurrent_ippo_cuda_training_smoke_v1"
        )
        stale_schema["exact_digest"] = stable_payload_digest(
            {
                key: value
                for key, value in stale_schema.items()
                if key != "exact_digest"
            }
        )
        with self.assertRaisesRegex(
            RecurrentScaleCampaignError,
            "schema drifted",
        ):
            validate_recurrent_cuda_training_smoke_report(
                stale_schema,
                preregistration=self.preregistration,
                runtime_provenance=runtime,
            )

        stale_learner = copy.deepcopy(report)
        stale_contract = stale_learner["contract"]
        assert isinstance(stale_contract, dict)
        stale_contract["learner_seed"] = SCALE_DEVELOPMENT_SEED_REGISTRY[
            "scale_learner"
        ][0]
        stale_contract["exact_digest"] = stable_payload_digest(
            {
                key: value
                for key, value in stale_contract.items()
                if key != "exact_digest"
            }
        )
        stale_learner["exact_digest"] = stable_payload_digest(
            {
                key: value
                for key, value in stale_learner.items()
                if key != "exact_digest"
            }
        )
        with self.assertRaisesRegex(
            RecurrentScaleCampaignError,
            "contract drifted",
        ):
            validate_recurrent_cuda_training_smoke_report(
                stale_learner,
                preregistration=self.preregistration,
                runtime_provenance=runtime,
            )


if __name__ == "__main__":
    unittest.main()
