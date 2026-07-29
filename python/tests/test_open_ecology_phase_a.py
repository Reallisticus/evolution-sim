from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
    RECURRENT_NUMERIC_KERNEL_VERSION,
)
from evolution_sim.mind.recurrent_artifact import (
    build_recurrent_training_crash_checkpoint,
    load_recurrent_training_crash_checkpoint,
    write_recurrent_training_crash_checkpoint,
)
from evolution_sim.mind.recurrent_experiment import (
    OPEN_ECOLOGY_PHASE_A,
    RECURRENT_EXPERIMENT_CONTRACT_VERSION,
)


REQUIRES_MIND_ML = True

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
    model, _ppo, _schedule = phase_a.build_phase_a_run_components(
        cell_id="A0",
        learner_index=0,
    )
    benchmark_seeds = OPEN_ECOLOGY_SEED_REGISTRY[
        phase_a.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE
    ]
    ppo = phase_a._phase_a_ppo_config(
        learner_seed=benchmark_seeds[phase_a.OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX]
    )
    environment_seed_indices = list(
        range(phase_a.OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT)
    )
    seed_access = {
        "schema_version": (
            phase_a.OPEN_ECOLOGY_BENCHMARK_SEED_PROVENANCE_SCHEMA_VERSION
        ),
        "seed_registry_contract": {
            "version": phase_a.OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
            "sha256": phase_a.OPEN_ECOLOGY_CANONICAL_SHA256,
        },
        "environment_seed_roles": [phase_a.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE],
        "environment_seeds_by_role": {
            phase_a.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE: [
                benchmark_seeds[index] for index in environment_seed_indices
            ],
        },
        "environment_seed_indices": environment_seed_indices,
        "observed_environment_seed_count": len(environment_seed_indices),
        "canonical_registry_membership_valid": True,
        "registry_ordered_non_reused_range": {
            "offset": 0,
            "count": len(environment_seed_indices),
            "exclusive_stop": len(environment_seed_indices),
        },
        "model_initialization": {
            "seed": benchmark_seeds[phase_a.OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX],
            "registry_role": phase_a.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
            "registry_index": (phase_a.OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX),
        },
        "genome_stream": {
            "seed": benchmark_seeds[
                phase_a.OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
            ],
            "registry_role": phase_a.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
            "registry_index": (phase_a.OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX),
        },
        "genome_population_mode": mode,
        "training_phase": OPEN_ECOLOGY_PHASE_A,
        "initial_agent_density_cycle": [phase_a.OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS],
        "scientific_environment_seed_roles_accessed": [],
        "training_seeds_accessed": False,
        "selection_seeds_accessed": False,
        "validation_seeds_accessed": False,
        "lockbox_seeds_accessed": False,
    }
    collector = {
        "experiment_contract_version": (
            phase_a.RECURRENT_FIXED_BATCH_EXPERIMENT_CONTRACT_VERSION
        ),
        "fixed_batch_enabled": True,
        "fixed_batch_capacity": phase_a.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        "fixed_batch_runtime_contract": phase_a.RecurrentFixedBatchRuntimeContract(
            batch_capacity=phase_a.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        ).as_contract(),
    }
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
                "seed_access_sha256": stable_payload_digest(seed_access),
                "collector_contract_sha256": stable_payload_digest(collector),
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
            "fixed_batch_collection_included": True,
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
        "seed_access": seed_access,
        "collector": collector,
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
            "genome_stream_seed": benchmark_seeds[
                phase_a.OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
            ],
            "encoder_size": 256,
            "hidden_size": 256,
            "recurrent_layers": 1,
            "learner_seed": benchmark_seeds[
                phase_a.OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX
            ],
            "update_epochs": 4,
            "sequence_minibatch_size": 16,
            "tbptt_steps": 128,
            "burn_in_steps": 16,
            "fixed_batch_capacity": (
                phase_a.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
            ),
            "preregistered_gate_member": True,
            "device_resolved": "cuda:0",
            "model_config": asdict(model),
            "ppo_config": asdict(ppo),
            "scheduled_worlds": 16,
            "scheduled_world_ticks": 16 * 129,
            "schedule_contract": (
                "canonical_open_ecology_operational_benchmark_phase_a_v1"
            ),
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
        "fixed_batch_runtime_contract": (
            phase_a.RecurrentFixedBatchRuntimeContract.open_ecology().as_contract()
        ),
    }
    runtime["exact_digest"] = stable_payload_digest(runtime)
    return runtime


def _campaign() -> dict[str, object]:
    envelope = phase_a.build_open_ecology_phase_a_resource_envelope(
        source_commit=_SOURCE_COMMIT,
    )
    gate = phase_a.build_open_ecology_phase_a_throughput_gate(
        source_commit=_SOURCE_COMMIT,
        heritable_report=_benchmark_report("heritable"),
        zero_all_report=_benchmark_report("zero_all"),
        resource_envelope=envelope,
    )
    return phase_a.build_open_ecology_phase_a_preregistration(
        source_commit=_SOURCE_COMMIT,
        source_manifest_sha256=_SOURCE_MANIFEST,
        archive_tool_authority_sha256="f" * 64,
        runtime_contract=_runtime_contract(
            workers=int(gate["selected_rollout_workers"])
        ),
        throughput_gate=gate,
    )


def _write_valid_phase_a_update(
    run_directory: Path,
    *,
    campaign: dict[str, object],
) -> tuple[
    dict[str, object],
    tuple[tuple[phase_a.PhaseAOpenEcologyRolloutTask, ...], ...],
    dict[str, object],
]:
    update_index = 0
    previous_commit_exact_digest = None
    model_config, _ppo, schedule = phase_a.build_phase_a_run_components(
        cell_id="A0",
        learner_index=0,
    )
    run_contract = phase_a.build_phase_a_run_contract(
        campaign,
        cell_id="A0",
        learner_index=0,
    )
    learner_seed = int(run_contract["learner_seed"])
    model = PublicRecurrentActorCritic(
        model_config,
        initialization_seed=learner_seed,
    )
    before = phase_a.recurrent_model_state_sha256(model)
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    after = phase_a.recurrent_model_state_sha256(model)
    update: dict[str, object] = {
        "schema_version": phase_a.OPEN_ECOLOGY_PHASE_A_UPDATE_SCHEMA_VERSION,
        "run_id": run_contract["run_id"],
        "run_contract_digest": run_contract["exact_digest"],
        "update_index": update_index,
        "previous_commit_exact_digest": previous_commit_exact_digest,
        "tasks": [asdict(task) for task in schedule[update_index]],
        "rollout": {},
        "optimizer": {
            "minibatch_count": 1,
            "parameter_delta_l2": 0.001,
            "post_step_kl_audit_count": 1,
            "post_step_kl_rejected_step_count": 0,
            "post_step_kl_rollback_performed": False,
            "post_step_kl_rejection_reason": None,
        },
        "counterfactual_collection": None,
        "counterfactual_auxiliary": None,
        "model_state_sha256_before_update": before,
        "model_state_sha256_after_update": after,
    }
    update["exact_digest"] = stable_payload_digest(update)
    checkpoint = build_recurrent_training_crash_checkpoint(
        model,
        optimizer_state={},
        rng_state={
            "phase_a_campaign_digest": campaign["exact_digest"],
            "phase_a_run_contract_digest": run_contract["exact_digest"],
            "phase_a_update_exact_digest": update["exact_digest"],
            "phase_a_previous_commit_exact_digest": previous_commit_exact_digest,
        },
        optimizer_type="torch.optim.Adam",
        training_config=run_contract,
        seed_registry_digest=phase_a.OPEN_ECOLOGY_CANONICAL_SHA256,
        source_commit=_SOURCE_COMMIT,
        source_manifest_sha256=_SOURCE_MANIFEST,
        learner_seed=learner_seed,
        completed_updates=update_index + 1,
        run_id=str(run_contract["run_id"]),
    )
    commit = phase_a.write_phase_a_update_commit(
        run_directory,
        update_index=update_index,
        update_payload=update,
        checkpoint=checkpoint,
        run_contract=run_contract,
        previous_commit_exact_digest=previous_commit_exact_digest,
    )
    return run_contract, schedule, commit


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
                left.policy_sampling_identity == right.policy_sampling_identity
                for left, right in zip(a0, a3, strict=True)
            )
        )
        self.assertTrue(
            all(
                left.phase_a_genome_world_identity
                == right.phase_a_genome_world_identity
                for left, right in zip(a0, a3, strict=True)
            )
        )
        self.assertTrue(
            all(
                left.task_id != right.task_id
                for left, right in zip(a0, a3, strict=True)
            )
        )

    def test_phase_a_update_transition_rejects_no_learning_but_preserves_mixed_kl(
        self,
    ) -> None:
        before = "1" * 64
        after = "2" * 64
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "accepted PPO minibatch",
        ):
            phase_a._validate_phase_a_update_transition(
                optimizer={
                    "minibatch_count": 0,
                    "parameter_delta_l2": 0.0,
                    "post_step_kl_audit_count": 1,
                    "post_step_kl_rejected_step_count": 1,
                    "post_step_kl_rollback_performed": True,
                    "post_step_kl_rejection_reason": "forward_kl_exceeded",
                },
                model_state_sha256_before_update=before,
                model_state_sha256_after_update=before,
            )
        phase_a._validate_phase_a_update_transition(
            optimizer={
                "minibatch_count": 3,
                "parameter_delta_l2": 0.25,
                "post_step_kl_audit_count": 4,
                "post_step_kl_rejected_step_count": 1,
                "post_step_kl_rollback_performed": True,
                "post_step_kl_rejection_reason": "forward_kl_exceeded",
            },
            model_state_sha256_before_update=before,
            model_state_sha256_after_update=after,
        )
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "accepted plus rejected",
        ):
            phase_a._validate_phase_a_update_transition(
                optimizer={
                    "minibatch_count": 3,
                    "parameter_delta_l2": 0.25,
                    "post_step_kl_audit_count": 5,
                    "post_step_kl_rejected_step_count": 1,
                    "post_step_kl_rollback_performed": True,
                    "post_step_kl_rejection_reason": "forward_kl_exceeded",
                },
                model_state_sha256_before_update=before,
                model_state_sha256_after_update=after,
            )

    def test_resume_recovers_exact_next_pending_update_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            run_contract, schedule, commit = _write_valid_phase_a_update(
                run_directory,
                campaign=self.campaign,
            )
            final = run_directory / "updates" / "update-0000"
            pending = (
                run_directory
                / "updates"
                / ".update-0000.pending-123-0123456789abcdef0123456789abcdef"
            )
            final.rename(pending)

            phase_a._recover_unpublished_update_staging(
                run_directory,
                run_contract=run_contract,
                schedule=schedule,
                resume=True,
            )
            prefix = phase_a.verify_phase_a_evidence_prefix(
                run_directory,
                run_contract=run_contract,
                schedule=schedule,
            )
            self.assertFalse(pending.exists())
            self.assertTrue(final.is_dir())
            self.assertEqual(prefix.completed_updates, 1)
            self.assertEqual(prefix.commit_digests, (commit["exact_digest"],))

            phase_a._recover_unpublished_update_staging(
                run_directory,
                run_contract=run_contract,
                schedule=schedule,
                resume=True,
            )
            self.assertEqual(
                sorted(path.name for path in (run_directory / "updates").iterdir()),
                ["update-0000"],
            )

    def test_pending_update_requires_resume_and_preserves_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            run_contract, schedule, _commit = _write_valid_phase_a_update(
                run_directory,
                campaign=self.campaign,
            )
            final = run_directory / "updates" / "update-0000"
            pending = (
                run_directory
                / "updates"
                / ".update-0000.pending-123-0123456789abcdef0123456789abcdef"
            )
            final.rename(pending)
            before = {
                path.name: path.read_bytes()
                for path in pending.iterdir()
                if path.is_file()
            }

            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "resume was not enabled",
            ):
                phase_a._recover_unpublished_update_staging(
                    run_directory,
                    run_contract=run_contract,
                    schedule=schedule,
                    resume=False,
                )

            self.assertTrue(pending.is_dir())
            self.assertFalse(final.exists())
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in pending.iterdir()
                    if path.is_file()
                },
                before,
            )

    def test_pending_update_rejects_tampering_and_unknown_bytes_without_deleting(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            run_contract, schedule, _commit = _write_valid_phase_a_update(
                run_directory,
                campaign=self.campaign,
            )
            final = run_directory / "updates" / "update-0000"
            pending = (
                run_directory
                / "updates"
                / ".update-0000.pending-123-0123456789abcdef0123456789abcdef"
            )
            final.rename(pending)
            update_path = pending / "update.json"
            update_path.write_bytes(update_path.read_bytes() + b" ")
            tampered = update_path.read_bytes()

            with self.assertRaises(phase_a.OpenEcologyPhaseAError):
                phase_a._recover_unpublished_update_staging(
                    run_directory,
                    run_contract=run_contract,
                    schedule=schedule,
                    resume=True,
                )
            self.assertTrue(pending.is_dir())
            self.assertEqual(update_path.read_bytes(), tampered)
            self.assertFalse(final.exists())

            unknown = run_directory / "updates" / ".update-0000.pending-unknown"
            shutil.copytree(pending, unknown)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "unknown update entry",
            ):
                phase_a._recover_unpublished_update_staging(
                    run_directory,
                    run_contract=run_contract,
                    schedule=schedule,
                    resume=True,
                )
            self.assertTrue(unknown.is_dir())
            self.assertEqual(update_path.read_bytes(), tampered)

    def test_pending_update_source_replacement_never_becomes_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            run_contract, schedule, _commit = _write_valid_phase_a_update(
                run_directory,
                campaign=self.campaign,
            )
            updates = run_directory / "updates"
            final = updates / "update-0000"
            pending = (
                updates / ".update-0000.pending-123-0123456789abcdef0123456789abcdef"
            )
            preserved_original = updates / ".injected-verified-original"
            final.rename(pending)
            original_rename = phase_a._rename_phase_a_name_no_replace

            def substitute_source(
                parent_descriptor: int,
                *,
                source_name: str,
                destination_name: str,
                destination_path: Path,
            ) -> None:
                os.rename(
                    source_name,
                    preserved_original.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                os.mkdir(source_name, mode=0o700, dir_fd=parent_descriptor)
                (updates / source_name / "replacement.bin").write_bytes(
                    b"unverified replacement"
                )
                original_rename(
                    parent_descriptor,
                    source_name=source_name,
                    destination_name=destination_name,
                    destination_path=destination_path,
                )

            with (
                patch.object(
                    phase_a,
                    "_rename_phase_a_name_no_replace",
                    side_effect=substitute_source,
                ),
                self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "identity changed",
                ),
            ):
                phase_a._recover_unpublished_update_staging(
                    run_directory,
                    run_contract=run_contract,
                    schedule=schedule,
                    resume=True,
                )

            self.assertTrue(preserved_original.is_dir())
            self.assertTrue((final / "replacement.bin").is_file())
            self.assertEqual(
                (final / "replacement.bin").read_bytes(),
                b"unverified replacement",
            )

    def test_pending_update_destination_race_preserves_both_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            run_contract, schedule, _commit = _write_valid_phase_a_update(
                run_directory,
                campaign=self.campaign,
            )
            updates = run_directory / "updates"
            final = updates / "update-0000"
            pending = (
                updates / ".update-0000.pending-123-0123456789abcdef0123456789abcdef"
            )
            final.rename(pending)
            original_rename = phase_a._rename_phase_a_name_no_replace

            def race_destination(
                parent_descriptor: int,
                *,
                source_name: str,
                destination_name: str,
                destination_path: Path,
            ) -> None:
                os.mkdir(destination_name, mode=0o700, dir_fd=parent_descriptor)
                (updates / destination_name / "racer.bin").write_bytes(
                    b"racing destination"
                )
                original_rename(
                    parent_descriptor,
                    source_name=source_name,
                    destination_name=destination_name,
                    destination_path=destination_path,
                )

            with (
                patch.object(
                    phase_a,
                    "_rename_phase_a_name_no_replace",
                    side_effect=race_destination,
                ),
                self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "already exists",
                ),
            ):
                phase_a._recover_unpublished_update_staging(
                    run_directory,
                    run_contract=run_contract,
                    schedule=schedule,
                    resume=True,
                )

            self.assertTrue(pending.is_dir())
            self.assertEqual(
                {path.name for path in pending.iterdir()},
                {"update.json", "checkpoint.json", "commit.json"},
            )
            self.assertEqual(
                (final / "racer.bin").read_bytes(),
                b"racing destination",
            )

    def test_resource_envelope_is_reproducible_and_protocol_bound(self) -> None:
        first = phase_a.build_open_ecology_phase_a_resource_envelope(
            source_commit=_SOURCE_COMMIT,
        )
        second = phase_a.build_open_ecology_phase_a_resource_envelope(
            source_commit=_SOURCE_COMMIT,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first["maximum_wall_seconds"],
            phase_a.OPEN_ECOLOGY_PHASE_A_MAXIMUM_WALL_SECONDS,
        )
        self.assertEqual(first["maximum_wall_seconds"], 604_800)
        self.assertEqual(
            first["evidence_sha256"],
            phase_a.OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256,
        )
        phase_a.validate_open_ecology_phase_a_resource_envelope(
            first,
            expected_source_commit=_SOURCE_COMMIT,
        )

        repository_root = Path(phase_a.__file__).resolve().parents[3]
        preregistration_document = (
            repository_root / phase_a.OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH
        )
        self.assertEqual(
            hashlib.sha256(preregistration_document.read_bytes()).hexdigest(),
            phase_a.OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256,
        )

        for field, hostile_value in (
            ("maximum_wall_seconds", 604_801),
            ("evidence_sha256", "e" * 64),
        ):
            with self.subTest(field=field):
                hostile = json.loads(json.dumps(first))
                hostile[field] = hostile_value
                hostile["exact_digest"] = stable_payload_digest(
                    {
                        key: value
                        for key, value in hostile.items()
                        if key != "exact_digest"
                    }
                )
                with self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "prospectively sealed seven-day protocol",
                ):
                    phase_a.validate_open_ecology_phase_a_resource_envelope(
                        hostile,
                        expected_source_commit=_SOURCE_COMMIT,
                    )

    def test_launch_authority_amendment_v3_is_exactly_bound(self) -> None:
        self.assertEqual(
            phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SCHEMA_VERSION,
            "mind_v3_open_ecology_launch_authority_amendment_v3",
        )
        self.assertEqual(
            phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_PATH,
            "docs/research/open-ecology-launch-authority-amendment-v3.md",
        )
        repository_root = Path(phase_a.__file__).resolve().parents[3]
        prior_amendment_document = (
            repository_root
            / phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_PATH
        )
        self.assertEqual(
            phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SCHEMA_VERSION,
            "mind_v3_open_ecology_launch_authority_amendment_v2",
        )
        self.assertEqual(
            phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_PATH,
            "docs/research/open-ecology-launch-authority-amendment-v2.md",
        )
        self.assertEqual(
            hashlib.sha256(prior_amendment_document.read_bytes()).hexdigest(),
            phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SHA256,
        )
        amendment_document = (
            repository_root / phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_PATH
        )
        self.assertEqual(
            hashlib.sha256(amendment_document.read_bytes()).hexdigest(),
            phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SHA256,
        )
        prior_sealed = self.campaign["sealed_document"][
            "prior_launch_authority_amendment"
        ]
        self.assertEqual(
            prior_sealed,
            {
                "schema_version": (
                    phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SCHEMA_VERSION
                ),
                "path": phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_PATH,
                "file_sha256": (
                    phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SHA256
                ),
            },
        )
        sealed = self.campaign["sealed_document"]["launch_authority_amendment"]
        self.assertEqual(
            sealed,
            {
                "schema_version": (
                    phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SCHEMA_VERSION
                ),
                "path": phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_PATH,
                "file_sha256": (
                    phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SHA256
                ),
            },
        )
        self.assertEqual(
            self.campaign["launch_authority"]["schema_version"],
            phase_a.OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SCHEMA_VERSION,
        )
        self.assertEqual(
            self.campaign["architecture"]["model_contract_version"],
            RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
        )
        self.assertEqual(
            self.campaign["architecture"]["numeric_kernel"],
            RECURRENT_NUMERIC_KERNEL_VERSION,
        )

        hostile = json.loads(json.dumps(self.campaign))
        del hostile["sealed_document"]["prior_launch_authority_amendment"]
        hostile["exact_digest"] = stable_payload_digest(
            {
                key: value
                for key, value in hostile.items()
                if key != "exact_digest"
            }
        )
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "sealed Phase A document binding drifted",
        ):
            phase_a.validate_open_ecology_phase_a_preregistration(hostile)
        expected_buckets = list(phase_a.RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS)
        self.assertEqual(
            self.campaign["runtime_contract"]["fixed_batch_runtime_contract"][
                "execution_batch_buckets"
            ],
            expected_buckets,
        )
        self.assertEqual(
            self.campaign["throughput_gate"]["benchmark_contract"][
                "fixed_batch_execution_buckets"
            ],
            expected_buckets,
        )
        self.assertEqual(
            self.campaign["training"]["fixed_batch_execution_buckets"],
            expected_buckets,
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
            4_608,
        )
        self.assertEqual(
            projection["phase_a_replay_selection_executions"],
            4_608,
        )
        self.assertEqual(
            projection["phase_a_physical_selection_executions"],
            9_216,
        )
        self.assertEqual(
            projection["phase_b_primary_selection_executions"],
            2_304,
        )
        self.assertEqual(
            projection["phase_b_replay_selection_executions"],
            2_304,
        )
        self.assertEqual(
            projection["phase_b_physical_selection_executions"],
            4_608,
        )
        self.assertEqual(
            projection["selection_world_ticks"],
            9_216 * 512 + 4_608 * 2_000,
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
        self.assertEqual(selection["trained_policy_executions_per_artifact"], 160)
        self.assertEqual(
            selection["initialized_baseline_executions_per_artifact"],
            128,
        )
        self.assertEqual(selection["primary_executions_per_artifact"], 288)
        self.assertEqual(
            selection["independent_replay_executions_per_artifact"],
            288,
        )
        self.assertEqual(selection["physical_world_runs_per_artifact"], 576)
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
        tampered_kernel = json.loads(json.dumps(self.campaign))
        tampered_kernel["architecture"]["numeric_kernel"] = "unsealed_dense_kernel"
        tampered_kernel["configuration_sha256"] = stable_payload_digest(
            {
                "architecture": tampered_kernel["architecture"],
                "training": tampered_kernel["training"],
                "selection": tampered_kernel["selection"],
                "seed_contract": tampered_kernel["seed_contract"],
                "throughput_gate": tampered_kernel["throughput_gate"],
            }
        )
        tampered_kernel["exact_digest"] = stable_payload_digest(
            {
                key: value
                for key, value in tampered_kernel.items()
                if key != "exact_digest"
            }
        )
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "architecture",
        ):
            phase_a.validate_open_ecology_phase_a_preregistration(tampered_kernel)

    def test_scientific_seed_or_scalar_benchmark_cannot_authorize(self) -> None:
        resource_envelope = phase_a.build_open_ecology_phase_a_resource_envelope(
            source_commit=_SOURCE_COMMIT,
        )
        scientific_seed_report = json.loads(json.dumps(_benchmark_report("heritable")))
        scientific_access = scientific_seed_report["seed_access"]
        training_seeds = list(
            OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_train"][
                : phase_a.OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT
            ]
        )
        scientific_access["environment_seed_roles"] = ["open_ecology_train"]
        scientific_access["environment_seeds_by_role"] = {
            "open_ecology_train": training_seeds
        }
        scientific_access["scientific_environment_seed_roles_accessed"] = [
            "open_ecology_train"
        ]
        scientific_access["training_seeds_accessed"] = True
        scientific_seed_digest = stable_payload_digest(scientific_access)
        for case in scientific_seed_report["cases"]:
            for sample in case["samples"]:
                sample["seed_access_sha256"] = scientific_seed_digest
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "scientific or noncanonical seed role",
        ):
            phase_a.build_open_ecology_phase_a_throughput_gate(
                source_commit=_SOURCE_COMMIT,
                heritable_report=scientific_seed_report,
                zero_all_report=_benchmark_report("zero_all"),
                resource_envelope=resource_envelope,
            )

        scalar_report = json.loads(json.dumps(_benchmark_report("heritable")))
        scalar_report["collector"] = {
            "experiment_contract_version": (RECURRENT_EXPERIMENT_CONTRACT_VERSION),
            "fixed_batch_enabled": False,
            "fixed_batch_capacity": None,
            "fixed_batch_runtime_contract": None,
        }
        scalar_digest = stable_payload_digest(scalar_report["collector"])
        for case in scalar_report["cases"]:
            for sample in case["samples"]:
                sample["collector_contract_sha256"] = scalar_digest
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "canonical fixed-batch collector",
        ):
            phase_a.build_open_ecology_phase_a_throughput_gate(
                source_commit=_SOURCE_COMMIT,
                heritable_report=scalar_report,
                zero_all_report=_benchmark_report("zero_all"),
                resource_envelope=resource_envelope,
            )

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
                "fields differ",
            ):
                phase_a.validate_open_ecology_phase_a_launch_authorization(
                    authorization,
                    preregistration=self.campaign,
                    authorization_path=authorization_path,
                )

    def test_launch_readiness_truthfully_requires_external_evidence(self) -> None:
        readiness = phase_a.open_ecology_phase_a_launch_readiness()
        self.assertIs(readiness["phase_a_training_authorized"], False)
        self.assertIs(
            readiness["dependency_specific_proof_producers_available"],
            True,
        )
        self.assertIs(readiness["authorization_assembler_available"], True)
        self.assertEqual(
            readiness["blockers"],
            ["launch_evidence_index_required"],
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

    def test_authoritative_selection_reopens_and_reexecutes_all_terminals(
        self,
    ) -> None:
        requests: list[SimpleNamespace] = []

        def build_request(
            _preregistration: object,
            *,
            cell_id: str,
            learner_index: int,
            terminal_path: Path,
            evaluation_workers: int,
        ) -> SimpleNamespace:
            self.assertEqual(
                terminal_path.parts[-3:],
                (
                    phase_a.phase_a_run_id(
                        cell_id=cell_id,
                        learner_index=learner_index,
                    ),
                    "terminal",
                    "terminal.json",
                ),
            )
            self.assertEqual(evaluation_workers, 3)
            request = SimpleNamespace(
                cell_id=cell_id,
                learner_index=learner_index,
            )
            requests.append(request)
            return request

        def evaluate(request: SimpleNamespace) -> dict[str, object]:
            ordinal = len(phase_a.OPEN_ECOLOGY_PHASE_A_CELL_ORDER) * int(
                request.learner_index
            ) + phase_a.OPEN_ECOLOGY_PHASE_A_CELL_ORDER.index(request.cell_id)
            return {"exact_digest": f"{ordinal + 1:064x}"}

        def authorize(
            _preregistration: object,
            _report: object,
            *,
            cell_id: str,
            learner_index: int,
            terminal_path: Path,
            evaluation_workers: int,
        ) -> dict[str, object]:
            del terminal_path, evaluation_workers
            ordinal = (
                phase_a.OPEN_ECOLOGY_PHASE_A_CELL_ORDER.index(cell_id)
                * phase_a.OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT
                + learner_index
            )
            return {
                "cell_id": cell_id,
                "learner_index": learner_index,
                "terminal_authority": {
                    "exact_digest": f"{ordinal + 101:064x}",
                },
                "exact_digest": f"{ordinal + 201:064x}",
            }

        with tempfile.TemporaryDirectory() as temporary:
            terminal_root = Path(temporary)
            selection_authorization_path = (
                terminal_root / "selection-authorization.json"
            )
            selection_authorization_path.write_text(
                json.dumps({"exact_digest": "f" * 64}) + "\n",
                encoding="utf-8",
            )
            terminal_paths = tuple(
                terminal_root
                / phase_a.phase_a_run_id(
                    cell_id=cell_id,
                    learner_index=learner_index,
                )
                / "terminal"
                / "terminal.json"
                for cell_id in phase_a.OPEN_ECOLOGY_PHASE_A_CELL_ORDER
                for learner_index in range(phase_a.OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)
            )
            with (
                patch.object(
                    phase_a,
                    "validate_open_ecology_phase_a_preregistration",
                ),
                patch.object(phase_a, "_require_live_source") as live_source,
                patch.object(
                    phase_a,
                    "_canonical_phase_a_terminal_paths",
                    return_value=terminal_paths,
                ),
                patch.object(
                    phase_a,
                    "validate_open_ecology_phase_a_selection_authorization",
                ) as selection_authorization_validator,
                patch.object(
                    phase_a,
                    "build_verified_phase_a_selection_request",
                    side_effect=build_request,
                ) as request_builder,
                patch(
                    "evolution_sim.mind.open_ecology_selection."
                    "evaluate_open_ecology_selection_artifact",
                    side_effect=evaluate,
                ) as evaluator,
                patch.object(
                    phase_a,
                    "authorize_phase_a_terminal_selection_report",
                    side_effect=authorize,
                ) as authorizer,
                patch.object(
                    phase_a,
                    "preview_open_ecology_phase_a_cell_selection",
                    return_value={
                        "provisional_selected_cell_id": "A2",
                        "provisional_selected_contract": {
                            "critic_genome_conditioning": "none",
                            "value_shared_trunk_gradient": "stop",
                        },
                        "cell_summaries": {"A2": {"eligible": True}},
                        "selection_rule": "sealed-test-rule",
                    },
                ) as preview,
            ):
                result = phase_a.authorize_open_ecology_phase_a_cell_selection(
                    self.campaign,
                    terminal_root=temporary,
                    selection_authorization_path=selection_authorization_path,
                    evaluation_workers=3,
                )

        self.assertEqual(len(requests), 16)
        self.assertEqual(request_builder.call_count, 16)
        self.assertEqual(evaluator.call_count, 16)
        self.assertEqual(authorizer.call_count, 16)
        self.assertEqual(preview.call_count, 1)
        self.assertEqual(selection_authorization_validator.call_count, 2)
        self.assertEqual(live_source.call_count, 4)
        self.assertEqual(result["selected_cell_id"], "A2")
        self.assertIs(result["phase_b_authorized"], True)
        self.assertEqual(result["authority"]["terminal_bundle_count"], 16)
        self.assertEqual(
            result["authority"]["selection_authorization_exact_digest"],
            "f" * 64,
        )
        self.assertIs(
            result["authority"]["selection_preflight_completed_before_seed_access"],
            True,
        )
        self.assertIs(
            result["authority"]["caller_supplied_learner_summaries"],
            False,
        )
        self.assertEqual(len(result["learner_evidence"]), 16)
        self.assertIs(
            result["lifecycle"]["runtime_integration_authorized"],
            False,
        )

    def test_authoritative_selection_failure_injections_never_authorize(
        self,
    ) -> None:
        def request(
            _preregistration: object,
            *,
            cell_id: str,
            learner_index: int,
            terminal_path: Path,
            evaluation_workers: int,
        ) -> SimpleNamespace:
            del terminal_path, evaluation_workers
            return SimpleNamespace(
                cell_id=cell_id,
                learner_index=learner_index,
            )

        def report(item: SimpleNamespace) -> dict[str, object]:
            ordinal = (
                phase_a.OPEN_ECOLOGY_PHASE_A_CELL_ORDER.index(item.cell_id)
                * phase_a.OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT
                + item.learner_index
            )
            return {"exact_digest": f"{ordinal + 1:064x}"}

        def evidence(
            _preregistration: object,
            _report: object,
            *,
            cell_id: str,
            learner_index: int,
            terminal_path: Path,
            evaluation_workers: int,
        ) -> dict[str, object]:
            del terminal_path, evaluation_workers
            ordinal = (
                phase_a.OPEN_ECOLOGY_PHASE_A_CELL_ORDER.index(cell_id)
                * phase_a.OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT
                + learner_index
            )
            return {
                "cell_id": cell_id,
                "learner_index": learner_index,
                "terminal_authority": {
                    "exact_digest": f"{ordinal + 101:064x}",
                },
                "exact_digest": f"{ordinal + 201:064x}",
            }

        preview = {
            "provisional_selected_cell_id": "A0",
            "provisional_selected_contract": {
                "critic_genome_conditioning": "none",
                "value_shared_trunk_gradient": "stop",
            },
            "cell_summaries": {"A0": {"eligible": True}},
            "selection_rule": "sealed-test-rule",
        }
        cases = (
            (
                "missing terminal",
                {
                    "request_side_effect": [
                        *(
                            request(
                                None,
                                cell_id="A0",
                                learner_index=index,
                                terminal_path=Path("unused"),
                                evaluation_workers=1,
                            )
                            for index in range(2)
                        ),
                        phase_a.OpenEcologyPhaseAError("missing terminal"),
                    ],
                },
                "missing terminal",
            ),
            (
                "primary reexecution mismatch",
                {
                    "authorize_side_effect": phase_a.OpenEcologyPhaseAError(
                        "selection report changed under exact reexecution"
                    ),
                },
                "changed under exact reexecution",
            ),
            (
                "no eligible cell",
                {
                    "preview_side_effect": phase_a.OpenEcologyPhaseAError(
                        "no Phase A cell is eligible; Phase B remains blocked"
                    ),
                },
                "no Phase A cell is eligible",
            ),
            (
                "final source drift",
                {
                    "source_side_effect": [
                        None,
                        None,
                        phase_a.OpenEcologyPhaseAError(
                            "Phase A runtime source manifest drifted"
                        ),
                    ],
                },
                "source manifest drifted",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            terminal_root = Path(temporary)
            selection_authorization_path = (
                terminal_root / "selection-authorization.json"
            )
            selection_authorization_path.write_text(
                json.dumps({"exact_digest": "f" * 64}) + "\n",
                encoding="utf-8",
            )
            terminal_paths = tuple(
                terminal_root
                / phase_a.phase_a_run_id(
                    cell_id=cell_id,
                    learner_index=learner_index,
                )
                / "terminal"
                / "terminal.json"
                for cell_id in phase_a.OPEN_ECOLOGY_PHASE_A_CELL_ORDER
                for learner_index in range(phase_a.OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)
            )
            for label, overrides, expected in cases:
                with self.subTest(label=label):
                    with (
                        patch.object(
                            phase_a,
                            "validate_open_ecology_phase_a_preregistration",
                        ),
                        patch.object(
                            phase_a,
                            "_require_live_source",
                            side_effect=overrides.get("source_side_effect"),
                        ),
                        patch.object(
                            phase_a,
                            "_canonical_phase_a_terminal_paths",
                            return_value=terminal_paths,
                        ),
                        patch.object(
                            phase_a,
                            "validate_open_ecology_phase_a_selection_authorization",
                        ),
                        patch.object(
                            phase_a,
                            "build_verified_phase_a_selection_request",
                            side_effect=overrides.get(
                                "request_side_effect",
                                request,
                            ),
                        ),
                        patch(
                            "evolution_sim.mind.open_ecology_selection."
                            "evaluate_open_ecology_selection_artifact",
                            side_effect=report,
                        ) as evaluator,
                        patch.object(
                            phase_a,
                            "authorize_phase_a_terminal_selection_report",
                            side_effect=overrides.get(
                                "authorize_side_effect",
                                evidence,
                            ),
                        ),
                        patch.object(
                            phase_a,
                            "preview_open_ecology_phase_a_cell_selection",
                            side_effect=overrides.get("preview_side_effect"),
                            return_value=preview,
                        ),
                        self.assertRaisesRegex(
                            phase_a.OpenEcologyPhaseAError,
                            expected,
                        ),
                    ):
                        phase_a.authorize_open_ecology_phase_a_cell_selection(
                            self.campaign,
                            terminal_root=temporary,
                            selection_authorization_path=(selection_authorization_path),
                        )
                    if label == "missing terminal":
                        self.assertEqual(evaluator.call_count, 0)

    def test_selection_preflight_failures_open_zero_selection_seeds(self) -> None:
        evidence_digests = {
            "causal_evaluator_rejection_battery": "1" * 64,
            "capture_noninterference_reexecution": "2" * 64,
            "phase_a_selection_throughput": "3" * 64,
        }

        def fake_report_loader(
            _base: Path,
            _reference: object,
            *,
            expected_kind: str,
            preregistration: object,
            authorization_time: object,
        ) -> dict[str, object]:
            del preregistration, authorization_time
            return {"exact_digest": evidence_digests[expected_kind]}

        cases = (
            ("missing authorization", "cannot be opened safely"),
            ("authorization symlink", "symbolic link"),
            ("stale authorization", "stale or detached"),
            ("duplicate terminal", "missing, duplicate, or reordered"),
            ("missing terminal input", "cannot be opened safely"),
            ("terminal symlink", "symbolic link"),
            ("replaced terminal input", "stale, replaced, or detached"),
            ("authorization replaced during preflight", "replaced during preflight"),
        )
        for label, expected in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                terminal_root = base / "terminals"
                terminal_root.mkdir()
                requests: list[SimpleNamespace] = []
                for cell_id in phase_a.OPEN_ECOLOGY_PHASE_A_CELL_ORDER:
                    for learner_index in range(
                        phase_a.OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT
                    ):
                        terminal_directory = (
                            terminal_root
                            / phase_a.phase_a_run_id(
                                cell_id=cell_id,
                                learner_index=learner_index,
                            )
                            / "terminal"
                        )
                        terminal_directory.mkdir(parents=True)
                        terminal_path = terminal_directory / "terminal.json"
                        run_contract_path = terminal_directory / "run-contract.json"
                        artifact_path = (
                            terminal_directory / "terminal-training-artifact.json"
                        )
                        terminal_path.write_text(
                            f'{{"terminal":"{cell_id}-{learner_index}"}}\n',
                            encoding="utf-8",
                        )
                        run_contract_path.write_text(
                            f'{{"contract":"{cell_id}-{learner_index}"}}\n',
                            encoding="utf-8",
                        )
                        artifact_path.write_text(
                            f'{{"artifact":"{cell_id}-{learner_index}"}}\n',
                            encoding="utf-8",
                        )
                        requests.append(
                            SimpleNamespace(
                                cell_id=cell_id,
                                learner_index=learner_index,
                                artifact_path=artifact_path,
                                run_contract_path=run_contract_path,
                                training_authority={
                                    "exact_digest": (f"{len(requests) + 101:064x}"),
                                },
                            )
                        )

                with patch(
                    "evolution_sim.mind.open_ecology_selection."
                    "OpenEcologySelectionRequest",
                    SimpleNamespace,
                ):
                    matrix = [
                        phase_a._phase_a_selection_terminal_authority_entry(
                            request,
                            terminal_root=terminal_root,
                        )
                        for request in requests
                    ]
                authorization: dict[str, object] = {
                    "schema_version": (
                        phase_a.OPEN_ECOLOGY_PHASE_A_SELECTION_AUTHORIZATION_SCHEMA_VERSION
                    ),
                    "authorized_at_utc": "2026-07-27T00:00:00Z",
                    "campaign_digest": self.campaign["exact_digest"],
                    "configuration_sha256": self.campaign["configuration_sha256"],
                    "source": self.campaign["source"],
                    "dependency_evidence": [
                        {
                            "evidence_kind": kind,
                            "semantic_report_digest": evidence_digests[kind],
                            "file": {
                                "relative_path": f"{kind}.json",
                                "sha256": "4" * 64,
                                "byte_length": 1,
                            },
                        }
                        for kind in (
                            "causal_evaluator_rejection_battery",
                            "capture_noninterference_reexecution",
                        )
                    ],
                    "selection_throughput_evidence": {
                        "evidence_kind": "phase_a_selection_throughput",
                        "semantic_report_digest": evidence_digests[
                            "phase_a_selection_throughput"
                        ],
                        "file": {
                            "relative_path": "phase-a-selection-throughput.json",
                            "sha256": "5" * 64,
                            "byte_length": 1,
                        },
                    },
                    "terminal_matrix": matrix,
                    "authorization": {
                        "authorization_scope": "phase_a_selection_only",
                        "phase_a_selection_authorized": True,
                        "phase_b_authorized": False,
                        "phase_c_authorized": False,
                        "phase_d_authorized": False,
                        "runtime_integration_authorized": False,
                        "promotion_authorized": False,
                    },
                }

                authorization_path = base / "selection-authorization.json"

                def write_authorization(
                    payload: dict[str, object],
                    *,
                    destination: Path = authorization_path,
                ) -> None:
                    payload.pop("exact_digest", None)
                    payload["exact_digest"] = stable_payload_digest(payload)
                    destination.write_text(
                        json.dumps(payload, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

                write_authorization(authorization)
                if label == "missing authorization":
                    authorization_path.unlink()
                elif label == "authorization symlink":
                    real_path = base / "real-selection-authorization.json"
                    authorization_path.replace(real_path)
                    authorization_path.symlink_to(real_path)
                elif label == "stale authorization":
                    authorization["campaign_digest"] = "9" * 64
                    write_authorization(authorization)
                elif label == "duplicate terminal":
                    matrix[-1] = dict(matrix[0])
                    write_authorization(authorization)
                elif label == "missing terminal input":
                    requests[-1].artifact_path.unlink()
                elif label == "terminal symlink":
                    artifact_path = requests[-1].artifact_path
                    replacement = base / "replacement-artifact.json"
                    replacement.write_text("{}\n", encoding="utf-8")
                    artifact_path.unlink()
                    artifact_path.symlink_to(replacement)
                elif label == "replaced terminal input":
                    requests[-1].artifact_path.write_text(
                        '{"artifact":"replacement"}\n',
                        encoding="utf-8",
                    )

                request_by_identity = {
                    (request.cell_id, request.learner_index): request
                    for request in requests
                }

                def build_request(
                    _preregistration: object,
                    *,
                    cell_id: str,
                    learner_index: int,
                    terminal_path: Path,
                    evaluation_workers: int,
                    _request_by_identity: dict[
                        tuple[str, int], SimpleNamespace
                    ] = request_by_identity,
                ) -> SimpleNamespace:
                    del terminal_path, evaluation_workers
                    return _request_by_identity[(cell_id, learner_index)]

                original_validator = (
                    phase_a.validate_open_ecology_phase_a_selection_authorization
                )
                validation_count = 0

                def validate_and_replace(
                    *args: object,
                    _original_validator: object = original_validator,
                    _label: str = label,
                    _authorization: dict[str, object] = authorization,
                    _write_authorization: object = write_authorization,
                    **kwargs: object,
                ) -> None:
                    nonlocal validation_count
                    _original_validator(*args, **kwargs)  # type: ignore[operator]
                    validation_count += 1
                    if (
                        _label == "authorization replaced during preflight"
                        and validation_count == 1
                    ):
                        _authorization["authorized_at_utc"] = "2026-07-27T00:00:01Z"
                        _write_authorization(_authorization)  # type: ignore[operator]

                with (
                    patch.object(
                        phase_a,
                        "validate_open_ecology_phase_a_preregistration",
                    ),
                    patch.object(phase_a, "_require_live_source"),
                    patch.object(
                        phase_a,
                        "build_verified_phase_a_selection_request",
                        side_effect=build_request,
                    ),
                    patch(
                        "evolution_sim.mind.open_ecology_selection."
                        "OpenEcologySelectionRequest",
                        SimpleNamespace,
                    ),
                    patch(
                        "evolution_sim.mind.open_ecology_phase_a_readiness."
                        "_load_report_reference",
                        side_effect=fake_report_loader,
                    ),
                    patch.object(
                        phase_a,
                        "validate_open_ecology_phase_a_selection_authorization",
                        side_effect=validate_and_replace,
                    ),
                    patch(
                        "evolution_sim.mind.open_ecology_selection."
                        "evaluate_open_ecology_selection_artifact",
                    ) as evaluator,
                    self.assertRaisesRegex(
                        phase_a.OpenEcologyPhaseAError,
                        expected,
                    ),
                ):
                    phase_a.authorize_open_ecology_phase_a_cell_selection(
                        self.campaign,
                        terminal_root=terminal_root,
                        selection_authorization_path=authorization_path,
                    )
                self.assertEqual(evaluator.call_count, 0)

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
                "initial_model_state_sha256": (
                    phase_a.recurrent_model_state_sha256(prefix_model)
                ),
                "terminal_model_state_sha256": (
                    phase_a.recurrent_model_state_sha256(replacement_model)
                ),
                "initial_to_terminal_model_changed": True,
                "cumulative_accepted_ppo_minibatches": 8,
                "cumulative_post_step_kl_rejected_steps": 1,
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
            initial_model_state_sha256=(
                phase_a.recurrent_model_state_sha256(prefix_model)
            ),
            terminal_model_state_sha256=(
                phase_a.recurrent_model_state_sha256(prefix_model)
            ),
            cumulative_accepted_ppo_minibatches=8,
            cumulative_post_step_kl_rejected_steps=1,
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
                "does not match its prefix",
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
                    "initial_model_state_sha256": run_contract[
                        "initial_full_model_sha256"
                    ],
                    "terminal_model_state_sha256": "d" * 64,
                    "initial_to_terminal_model_changed": True,
                    "cumulative_accepted_ppo_minibatches": 8,
                    "cumulative_post_step_kl_rejected_steps": 1,
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
