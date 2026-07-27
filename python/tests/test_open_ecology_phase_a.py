from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import evolution_sim.mind.open_ecology_phase_a as phase_a
from evolution_sim.mind.open_ecology_seed_registry import OPEN_ECOLOGY_SEED_REGISTRY
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    GENOME_CONDITIONING_ACTOR_FILM_V1,
    PublicRecurrentActorCritic,
)
from evolution_sim.mind.recurrent_artifact import (
    build_recurrent_training_crash_checkpoint,
    load_recurrent_training_crash_checkpoint,
    write_recurrent_training_crash_checkpoint,
)
from evolution_sim.mind.recurrent_experiment import OPEN_ECOLOGY_PHASE_A


_SOURCE_COMMIT = "a" * 40
_SOURCE_MANIFEST = "b" * 64


def _distribution(values: list[int | float]) -> dict[str, float]:
    parsed = [float(value) for value in values]
    return {
        "minimum": min(parsed),
        "median": float(median(parsed)),
        "maximum": max(parsed),
        "mean": sum(parsed) / len(parsed),
    }


def _benchmark_report(mode: str) -> dict[str, object]:
    model, ppo, _schedule = phase_a.build_phase_a_run_components(
        cell_id="A0",
        learner_index=0,
    )
    model_sha = "c" * 64
    semantic_sha = "d" * 64
    elapsed = {
        1: 5_000_000_000,
        2: 3_000_000_000,
        4: 2_000_000_000,
        8: 2_500_000_000,
        16: 4_000_000_000,
    }
    cases: list[dict[str, object]] = []
    for workers in phase_a.OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS:
        elapsed_ns = elapsed[workers]
        samples = [
            {
                "repeat_index": repeat_index,
                "elapsed_ns": elapsed_ns,
                "worlds_per_second": 16.0 / (elapsed_ns / 1e9),
                "transitions_per_second": 100.0,
                "total_worlds": 16,
                "total_transitions": 1_000,
                "final_model_state_sha256": model_sha,
                "semantic_evidence_sha256": semantic_sha,
            }
            for repeat_index in range(phase_a.OPEN_ECOLOGY_PHASE_A_BENCHMARK_REPEATS)
        ]
        cases.append(
            {
                "rollout_workers": workers,
                "samples": samples,
                "elapsed_ns": _distribution([elapsed_ns] * len(samples)),
                "worlds_per_second": _distribution(
                    [float(sample["worlds_per_second"]) for sample in samples]
                ),
                "transitions_per_second": _distribution([100.0] * len(samples)),
                "speedup_vs_first_case": 1.0,
                "parallel_efficiency_vs_first_case": 1.0,
            }
        )
    return {
        "schema_version": phase_a.OPEN_ECOLOGY_PHASE_A_BENCHMARK_SCHEMA_VERSION,
        "scope": {
            "measured_operation": "simulation_rollout_merge_gae_and_ppo_update",
            "world_construction_included": True,
            "world_ticks_included": True,
            "observation_encoding_included": True,
            "policy_action_sampling_included": True,
            "ordered_worker_merge_included": True,
            "gae_included": True,
            "ppo_update_included": True,
            "artifact_serialization_included": False,
            "evaluation_included": False,
            "policy_promotion_authorized": False,
        },
        "source": {
            "commit_sha": _SOURCE_COMMIT,
            "dirty": False,
            "capture_contract": "git_head_plus_porcelain_dirty_flag_v1",
        },
        "runtime": {"device": "cuda:0"},
        "protocol": {
            "worker_counts": list(phase_a.OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS),
            "repeats": phase_a.OPEN_ECOLOGY_PHASE_A_BENCHMARK_REPEATS,
            "updates": 1,
            "worlds_per_update": 16,
            "rollout_ticks": 128,
            "scenarios": ["broad"],
            "device": "cuda",
            "input_contract": "tokenized",
            "genome_conditioning": GENOME_CONDITIONING_ACTOR_FILM_V1,
            "genome_population_mode": mode,
            "genome_stream_seed": OPEN_ECOLOGY_SEED_REGISTRY[
                "open_ecology_genome_stream"
            ][0],
            "encoder_size": 256,
            "hidden_size": 256,
            "recurrent_layers": 1,
            "learner_seed": OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
            "update_epochs": 4,
            "sequence_minibatch_size": 16,
            "tbptt_steps": 128,
            "burn_in_steps": 16,
            "preregistered_gate_member": True,
            "device_resolved": "cuda:0",
            "model_config": asdict(model),
            "ppo_config": asdict(ppo),
            "scheduled_worlds": 16,
            "scheduled_world_ticks": 16 * 129,
            "schedule_contract": ("canonical_open_ecology_phase_a_broad_treatment_v1"),
            "training_phase": OPEN_ECOLOGY_PHASE_A,
            "timing_boundary": "runner.run_only",
            "fresh_identically_seeded_runner_per_sample": True,
        },
        "determinism": {
            "cross_worker_model_state_match": True,
            "cross_worker_semantic_evidence_match": True,
            "final_model_state_sha256": model_sha,
            "semantic_evidence_sha256": semantic_sha,
        },
        "preregistered_gate": {
            "member_shape_valid": True,
            "member_claimed": True,
            "population_mode": mode,
            "required_population_modes": ["heritable", "zero_all"],
            "complete_pair_required": True,
            "complete_gate_claimed": False,
        },
        "cases": cases,
    }


def _runtime_contract(*, workers: int) -> dict[str, object]:
    runtime: dict[str, object] = {
        "schema_version": phase_a.OPEN_ECOLOGY_PHASE_A_RUNTIME_SCHEMA_VERSION,
        "python": {"implementation": "CPython", "version": "3.test"},
        "platform": {"system": "Linux", "machine": "x86_64"},
        "torch": {
            "version": torch.__version__,
            "cuda_version": "test",
            "cudnn_version": 1,
            "float32_matmul_precision": "highest",
        },
        "device": {
            "type": "cuda",
            "name": "test",
            "compute_capability": [8, 9],
            "total_memory_bytes": 12 * 1024**3,
        },
        "determinism": {
            "deterministic_algorithms_enabled": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False,
            "cublas_workspace_config": ":4096:8",
            "default_dtype": "torch.float32",
        },
        "rollout_workers": workers,
        "ordered_worker_merge_required": True,
    }
    runtime["exact_digest"] = stable_payload_digest(runtime)
    return runtime


def _campaign() -> dict[str, object]:
    envelope: dict[str, object] = {
        "schema_version": (
            phase_a.OPEN_ECOLOGY_PHASE_A_RESOURCE_ENVELOPE_SCHEMA_VERSION
        ),
        "source_commit": _SOURCE_COMMIT,
        "maximum_wall_seconds": 7 * 24 * 60 * 60,
        "evidence_sha256": "e" * 64,
    }
    envelope["exact_digest"] = stable_payload_digest(envelope)
    gate = phase_a.build_open_ecology_phase_a_throughput_gate(
        source_commit=_SOURCE_COMMIT,
        heritable_report=_benchmark_report("heritable"),
        zero_all_report=_benchmark_report("zero_all"),
        resource_envelope=envelope,
    )
    return phase_a.build_open_ecology_phase_a_preregistration(
        source_commit=_SOURCE_COMMIT,
        source_manifest_sha256=_SOURCE_MANIFEST,
        runtime_contract=_runtime_contract(
            workers=int(gate["selected_rollout_workers"])
        ),
        throughput_gate=gate,
    )


class OpenEcologyPhaseATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.campaign = _campaign()

    def test_schedule_is_exact_fixed_density_and_domain_separated(self) -> None:
        _, _, a0_schedule = phase_a.build_phase_a_run_components(
            cell_id="A0",
            learner_index=0,
        )
        _, _, a3_schedule = phase_a.build_phase_a_run_components(
            cell_id="A3",
            learner_index=0,
        )
        a0 = [task for update in a0_schedule for task in update]
        a3 = [task for update in a3_schedule for task in update]
        self.assertEqual(len(a0_schedule), 8)
        self.assertEqual([len(update) for update in a0_schedule], [16] * 8)
        self.assertEqual({task.open_ecology_training_phase for task in a0}, {"phase_a"})
        self.assertEqual(
            {task.open_ecology_treatment.initial_agents for task in a0},
            {64},
        )
        self.assertEqual(
            [task.environment_seed for task in a0],
            [task.environment_seed for task in a3],
        )
        self.assertTrue(
            all(
                left.policy_sampling_identity != right.policy_sampling_identity
                for left, right in zip(a0, a3, strict=True)
            )
        )

    def test_throughput_gate_and_preregistration_roundtrip_fail_closed(self) -> None:
        phase_a.validate_open_ecology_phase_a_preregistration(self.campaign)
        projection = self.campaign["throughput_gate"]["resource_projection"]
        self.assertEqual(
            projection["training_world_ticks"],
            2_048 * 129 + 4_096 * 257,
        )
        self.assertEqual(
            projection["phase_a_training_world_ticks"],
            2_048 * 129,
        )
        self.assertEqual(
            projection["phase_b_training_world_ticks"],
            4_096 * 257,
        )
        self.assertIs(
            projection["phase_a_training_projection_authoritative"],
            True,
        )
        self.assertIs(
            projection["phase_b_training_projection_authoritative"],
            False,
        )
        self.assertIs(
            projection["phase_b_mixed_density_full_update_benchmark_required"],
            True,
        )
        self.assertEqual(
            projection["phase_a_primary_selection_executions"],
            2_560,
        )
        self.assertEqual(
            projection["phase_a_replay_selection_executions"],
            2_560,
        )
        self.assertEqual(
            projection["phase_a_physical_selection_executions"],
            5_120,
        )
        self.assertEqual(
            projection["phase_b_primary_selection_executions"],
            1_280,
        )
        self.assertEqual(
            projection["phase_b_replay_selection_executions"],
            1_280,
        )
        self.assertEqual(
            projection["phase_b_physical_selection_executions"],
            2_560,
        )
        self.assertEqual(
            projection["selection_world_ticks"],
            5_120 * 512 + 2_560 * 2_000,
        )
        self.assertIs(projection["selection_projection_authoritative"], False)
        self.assertIs(
            projection["long_horizon_selection_benchmark_required"],
            True,
        )
        self.assertIs(projection["inside_recorded_resource_envelope"], False)
        self.assertIs(
            self.campaign["throughput_gate"]["training_topology_gate_passed"],
            True,
        )
        self.assertIs(
            self.campaign["throughput_gate"]["selection_resource_gate_passed"],
            False,
        )
        self.assertIs(self.campaign["throughput_gate"]["gate_passed"], False)
        selection = self.campaign["selection"]
        self.assertEqual(selection["primary_executions_per_artifact"], 160)
        self.assertEqual(
            selection["independent_replay_executions_per_artifact"],
            160,
        )
        self.assertEqual(selection["physical_world_runs_per_artifact"], 320)
        tampered = json.loads(json.dumps(self.campaign))
        tampered["architecture"]["hidden_size"] = 512
        tampered["configuration_sha256"] = stable_payload_digest(
            {
                "architecture": tampered["architecture"],
                "training": tampered["training"],
                "selection": tampered["selection"],
                "seed_contract": tampered["seed_contract"],
                "throughput_gate": tampered["throughput_gate"],
            }
        )
        tampered["exact_digest"] = stable_payload_digest(
            {key: value for key, value in tampered.items() if key != "exact_digest"}
        )
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "architecture",
        ):
            phase_a.validate_open_ecology_phase_a_preregistration(tampered)

    def test_forged_reused_evidence_cannot_authorize_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "proof.json"
            evidence.write_text('{"proved":true}\\n', encoding="utf-8")
            reference = phase_a._file_reference(evidence, base=root)
            source = self.campaign["source"]
            proofs = [
                {
                    "dependency_id": dependency_id,
                    "status": "behaviorally_proved",
                    "source_commit": source["commit"],
                    "source_manifest_sha256": source["manifest_sha256"],
                    "assertions": phase_a._phase_a_readiness_assertions(dependency_id),
                    "evidence": [reference],
                }
                for dependency_id in (
                    phase_a.OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES
                )
            ]
            capacity = 1_000 * 1024**3
            common_gate = {"passed": True, "evidence": [reference]}
            authorization: dict[str, object] = {
                "schema_version": (
                    phase_a.OPEN_ECOLOGY_PHASE_A_LAUNCH_AUTHORIZATION_SCHEMA_VERSION
                ),
                "campaign_digest": self.campaign["exact_digest"],
                "configuration_sha256": self.campaign["configuration_sha256"],
                "source": source,
                "readiness_dependencies": proofs,
                "operational_gates": {
                    "throughput": {
                        "passed": True,
                        "throughput_gate_digest": self.campaign["throughput_gate"][
                            "exact_digest"
                        ],
                        "evidence": [reference],
                    },
                    "storage": {
                        "passed": True,
                        "checked_at_utc": "2026-07-27T00:00:00Z",
                        "google_drive_free_bytes": 400 * 1024**3,
                        "projected_active_storage_bytes": 100 * 1024**3,
                        "target_filesystem_capacity_bytes": capacity,
                        "target_filesystem_free_bytes": 300 * 1024**3,
                        "required_target_free_bytes": 200 * 1024**3,
                        "evidence": [reference],
                    },
                    "output_lock": common_gate,
                    "immutable_uploader": common_gate,
                    "verification_before_prune": common_gate,
                    "terminal_aggregate_validator": common_gate,
                },
                "authorization": {
                    "phase_a_training_authorized": True,
                    "authorization_basis": (
                        "all_10_behavioral_dependencies_plus_operational_gates"
                    ),
                    "authorization_scope": "phase_a_training_only",
                    "phase_b_authorized": False,
                    "runtime_integration_authorized": False,
                    "promotion_authorized": False,
                },
            }
            authorization["exact_digest"] = stable_payload_digest(authorization)
            authorization_path = root / "launch-authorization.json"
            authorization_path.write_text(
                json.dumps(authorization, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "behavioral proof producers",
            ):
                phase_a.validate_open_ecology_phase_a_launch_authorization(
                    authorization,
                    preregistration=self.campaign,
                    authorization_path=authorization_path,
                )
            with (
                patch.object(
                    phase_a,
                    "open_ecology_phase_a_launch_readiness",
                    return_value={"phase_a_training_authorized": True},
                ),
                self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "throughput launch proof failed",
                ),
            ):
                phase_a.validate_open_ecology_phase_a_launch_authorization(
                    authorization,
                    preregistration=self.campaign,
                    authorization_path=authorization_path,
                )

    def test_launch_readiness_truthfully_lists_unimplemented_producers(self) -> None:
        readiness = phase_a.open_ecology_phase_a_launch_readiness()
        self.assertIs(readiness["phase_a_training_authorized"], False)
        self.assertIs(
            readiness["dependency_specific_proof_producers_available"],
            False,
        )
        self.assertEqual(
            readiness["blockers"][:10],
            list(phase_a.OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES),
        )
        self.assertIs(readiness["claim_boundary"]["training_launch"], False)

    def test_scalar_only_selection_cannot_authorize_phase_b(self) -> None:
        reports = []
        for cell_id in phase_a.OPEN_ECOLOGY_PHASE_A_CELL_ORDER:
            for learner_index in range(4):
                reports.append(
                    {
                        "cell_id": cell_id,
                        "learner_index": learner_index,
                        "exact_digest": f"{learner_index + 1:064x}",
                        "gates": {"eligible": True},
                        "metrics": {
                            "median_per_decision_normalized_individual_return": (
                                float(learner_index)
                            ),
                            "heldout_value_rmse": 1.0,
                            "advantage_variance": 1.0,
                        },
                    }
                )
        with (
            patch.object(
                phase_a,
                "validate_open_ecology_phase_a_preregistration",
            ),
            patch.object(
                phase_a,
                "validate_open_ecology_phase_a_learner_evidence",
            ),
        ):
            result = phase_a.preview_open_ecology_phase_a_cell_selection(
                self.campaign,
                reports,
            )
        self.assertIs(result["phase_b_authorized"], False)
        self.assertIsNone(result["selected_cell_id"])
        self.assertEqual(
            result["authorization_blocker"],
            "fresh_exact_cpu_open_ecology_evidence_producer_unavailable",
        )

    def test_checkpoint_roundtrip_has_no_wall_clock_digest_input(self) -> None:
        model_config, _ppo, _schedule = phase_a.build_phase_a_run_components(
            cell_id="A0",
            learner_index=0,
        )
        learner_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0]
        model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=learner_seed,
        )
        state = {
            "phase_a_campaign_digest": "1" * 64,
            "phase_a_run_contract_digest": "2" * 64,
            "phase_a_update_exact_digest": "3" * 64,
            "phase_a_previous_commit_exact_digest": None,
        }
        checkpoint = build_recurrent_training_crash_checkpoint(
            model,
            optimizer_state={},
            rng_state=state,
            optimizer_type="torch.optim.Adam",
            training_config={"exact_digest": "4" * 64},
            seed_registry_digest="5" * 64,
            source_commit="6" * 40,
            source_manifest_sha256="7" * 64,
            learner_seed=learner_seed,
            completed_updates=1,
            run_id="phase-a-roundtrip-test",
        )
        self.assertNotIn("phase_a_elapsed_seconds", state)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            write_recurrent_training_crash_checkpoint(path, checkpoint)
            loaded = load_recurrent_training_crash_checkpoint(path)
            rebuilt = build_recurrent_training_crash_checkpoint(
                loaded.model,
                optimizer_state=loaded.optimizer_state,
                rng_state=loaded.rng_state,
                optimizer_type="torch.optim.Adam",
                training_config={"exact_digest": "4" * 64},
                seed_registry_digest="5" * 64,
                source_commit="6" * 40,
                source_manifest_sha256="7" * 64,
                learner_seed=learner_seed,
                completed_updates=1,
                run_id="phase-a-roundtrip-test",
            )
        self.assertEqual(
            checkpoint["checkpoint_sha256"],
            rebuilt["checkpoint_sha256"],
        )
        self.assertEqual(checkpoint, rebuilt)

    def test_terminal_cannot_replace_the_final_prefix_model(self) -> None:
        model_config, _ppo, _schedule = phase_a.build_phase_a_run_components(
            cell_id="A0",
            learner_index=0,
        )
        learner_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0]
        prefix_model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=learner_seed,
        )
        replacement_model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=learner_seed + 1,
        )
        run_contract = {
            "campaign_digest": "1" * 64,
            "exact_digest": "2" * 64,
            "run_id": "phase-a-a0-learner-0-test",
            "cell_id": "A0",
            "learner_index": 0,
            "learner_seed": learner_seed,
        }
        commit_digests = tuple(f"{index + 1:064x}" for index in range(8))
        terminal: dict[str, object] = {
            "schema_version": phase_a.OPEN_ECOLOGY_PHASE_A_TERMINAL_SCHEMA_VERSION,
            "campaign_digest": run_contract["campaign_digest"],
            "run_contract_digest": run_contract["exact_digest"],
            "run_id": run_contract["run_id"],
            "cell_id": run_contract["cell_id"],
            "learner_index": run_contract["learner_index"],
            "learner_seed": learner_seed,
            "run_contract": {
                "file": {"path": "run-contract.json"},
                "exact_digest": run_contract["exact_digest"],
                "selection_binding_required": True,
            },
            "training": {
                "completed_updates": 8,
                "training_world_count": 128,
                "commit_exact_digests": list(commit_digests),
                "terminal_model_state_sha256": (
                    phase_a.recurrent_model_state_sha256(replacement_model)
                ),
            },
            "artifact": {
                "file": {"path": "terminal-training-artifact.json"},
                "artifact_sha256": "3" * 64,
                "strict_cpu_reconstruction_verified": True,
                "selection_evaluation_pending": True,
            },
            "lifecycle": {
                "development_only": True,
                "training_complete": True,
                "selection_complete": False,
                "cell_selection_authorized": False,
                "runtime_integration_authorized": False,
                "promotion_authorized": False,
                "validation_accessed": False,
                "lockbox_accessed": False,
            },
        }
        terminal["exact_digest"] = stable_payload_digest(terminal)
        prefix = phase_a.PhaseAEvidencePrefix(
            completed_updates=8,
            commit_digests=commit_digests,
            terminal_checkpoint=SimpleNamespace(model=prefix_model),
        )
        with (
            patch.object(
                phase_a,
                "_load_strict_json",
                side_effect=[terminal, run_contract],
            ),
            patch.object(phase_a, "_verify_file_reference"),
            patch.object(
                phase_a,
                "verify_phase_a_evidence_prefix",
                return_value=prefix,
            ),
            self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "final committed checkpoint",
            ),
        ):
            phase_a._verify_phase_a_terminal(
                Path("/tmp/terminal/terminal.json"),
                artifact_path=Path("/tmp/terminal/terminal-training-artifact.json"),
                run_contract_path=Path("/tmp/terminal/run-contract.json"),
                run_contract=run_contract,
                run_directory=Path("/tmp/run"),
                schedule=(),
            )

    def test_selection_request_is_derived_from_verified_terminal_bundle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_directory = root / phase_a.phase_a_run_id(
                cell_id="A0",
                learner_index=0,
            )
            terminal_directory = run_directory / "terminal"
            terminal_directory.mkdir(parents=True)
            terminal_path = terminal_directory / "terminal.json"
            artifact_path = terminal_directory / "terminal-training-artifact.json"
            run_contract_path = terminal_directory / "run-contract.json"
            run_contract = phase_a.build_phase_a_run_contract(
                self.campaign,
                cell_id="A0",
                learner_index=0,
            )
            run_contract_path.write_text(
                json.dumps(run_contract, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifact_path.write_text('{"artifact":"test"}\n', encoding="utf-8")
            commits = [f"{index + 1:064x}" for index in range(8)]
            terminal: dict[str, object] = {
                "campaign_digest": self.campaign["exact_digest"],
                "cell_id": "A0",
                "learner_index": 0,
                "run_contract": {
                    "file": {"path": "run-contract.json"},
                },
                "artifact": {
                    "file": {"path": "terminal-training-artifact.json"},
                    "artifact_sha256": "c" * 64,
                },
                "training": {
                    "commit_exact_digests": commits,
                    "terminal_model_state_sha256": "d" * 64,
                },
                "exact_digest": "e" * 64,
            }
            terminal_path.write_text(
                json.dumps(terminal, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(phase_a, "_require_live_source") as live_source,
                patch.object(
                    phase_a,
                    "_verify_phase_a_terminal",
                    return_value=terminal,
                ) as verifier,
            ):
                request = phase_a.build_verified_phase_a_selection_request(
                    self.campaign,
                    cell_id="A0",
                    learner_index=0,
                    terminal_path=terminal_path,
                    evaluation_workers=3,
                )

            verifier.assert_called_once()
            live_source.assert_called_once_with(self.campaign)
            self.assertEqual(request.artifact_path, artifact_path)
            self.assertEqual(request.run_contract_path, run_contract_path)
            self.assertEqual(
                request.expected_artifact_sha256,
                terminal["artifact"]["artifact_sha256"],
            )
            self.assertEqual(request.evaluation_workers, 3)
            authority = request.training_authority
            self.assertIsNotNone(authority)
            assert authority is not None
            self.assertEqual(
                authority["terminal_exact_digest"],
                terminal["exact_digest"],
            )
            self.assertEqual(
                authority["final_prefix_commit_exact_digest"],
                commits[-1],
            )
            self.assertEqual(
                authority["run_contract_exact_digest"],
                run_contract["exact_digest"],
            )
            self.assertEqual(
                authority["artifact_file_sha256"],
                phase_a._file_sha256(artifact_path),
            )

            tampered = dict(terminal)
            tampered["cell_id"] = "A1"
            terminal_path.write_text(
                json.dumps(tampered, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with (
                self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "identity",
                ),
                patch.object(phase_a, "_require_live_source"),
            ):
                phase_a.build_verified_phase_a_selection_request(
                    self.campaign,
                    cell_id="A0",
                    learner_index=0,
                    terminal_path=terminal_path,
                )


if __name__ == "__main__":
    unittest.main()
