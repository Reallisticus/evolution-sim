from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
import hashlib
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from evolution_sim.env.world import SimulationWorld
from evolution_sim.env.runtime.signals import CommunicationReceiverProjection
from evolution_sim.io.open_ecology_campaign_storage import (
    CampaignStorageError,
    CampaignStorageLock,
    default_storage_lock_path,
)
from evolution_sim.io.open_ecology_checkpoint import (
    REQUIRED_CHECKPOINT_COMPONENTS,
    load_open_ecology_checkpoint,
)
from evolution_sim.io.open_ecology_rotating_writer import (
    load_open_ecology_evidence_manifest,
)
from evolution_sim.mind.open_ecology_persistent_island import (
    OPEN_ECOLOGY_PARALLEL_CAMPAIGN_ADVANCE_AUTHORIZED,
    OPEN_ECOLOGY_PHASE_D_ARM_ORDER,
    OPEN_ECOLOGY_PHASE_D_TASK_COUNT,
    PersistentArtifactBinding,
    PersistentIslandRunner,
    PersistentIslandTask,
    build_persistent_island_task_matrix,
    prioritized_persistent_island_triplet,
    _verify_live_source_authority,
)
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_CANONICAL_SHA256,
    OPEN_ECOLOGY_SEED_REGISTRY,
)
from evolution_sim.mind.recurrent_actor_critic import (
    GENOME_CONDITIONING_ACTOR_FILM_V1,
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.recurrent_artifact import (
    FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
    build_frozen_recurrent_policy_artifact,
    write_frozen_recurrent_policy_artifact,
)
from evolution_sim.mind.recurrent_experiment import OpenEcologySignalTreatment
from evolution_sim.mind.recurrent_genome import zero_recurrent_genome


REQUIRES_MIND_ML = True
_SOURCE_COMMIT = "a" * 40


class OpenEcologyPersistentIslandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary_directory.name).resolve()
        cls.artifact_path = cls.base / "frozen-policy.json"
        config = RecurrentActorCriticConfig.for_signal_config(
            OpenEcologySignalTreatment().as_signal_config(),
            encoder_size=256,
            hidden_size=256,
            recurrent_layers=1,
            genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
        )
        model = PublicRecurrentActorCritic(
            config,
            initialization_seed=17,
        ).to(device="cpu", dtype=torch.float32)
        cls.artifact = build_frozen_recurrent_policy_artifact(
            model,
            training_config={
                "algorithm": "recurrent_ppo",
                "purpose": "persistent_island_test",
            },
            experiment_config={
                "phase": "phase_b",
                "cell_id": "A0",
            },
            seed_registry_digest=OPEN_ECOLOGY_CANONICAL_SHA256,
            source_commit=_SOURCE_COMMIT,
            source_manifest_sha256="b" * 64,
            data_metadata={
                "policy_induced": True,
                "agent_steps": 4096,
                "digest": "c" * 64,
            },
            run_metadata={
                "run_id": "persistent-island-test",
                "episodes": 32,
            },
            learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
            learner_device="cpu",
            full_world_replay_manifest={
                "schema_version": FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
                "manifest_sha256": "d" * 64,
                "replay_engine_contract_sha256": "e" * 64,
                "environment_seed_registry_sha256": "f" * 64,
                "environment_seed_roles": [
                    "scale_train",
                    "scale_selection",
                ],
                "scenario_names": ["broad"],
                "tick_horizons": [120],
                "world_count": 8,
                "replay_verified_world_count": 8,
                "policy_sampling_stream_count": 4,
                "all_replays_exact": True,
                "verification_runner": "test-full-world-replay",
                "verification_runner_sha256": "1" * 64,
            },
        )
        write_frozen_recurrent_policy_artifact(
            cls.artifact_path,
            cls.artifact,
        )
        cls.artifact_file_sha256 = hashlib.sha256(
            cls.artifact_path.read_bytes()
        ).hexdigest()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def setUp(self) -> None:
        source_authority = patch(
            "evolution_sim.mind.open_ecology_persistent_island."
            "_verify_live_source_authority"
        )
        source_authority.start()
        self.addCleanup(source_authority.stop)
        self.campaign_root = self.base / self.id().rsplit(".", 1)[-1]
        self.campaign_root.mkdir()
        self.tasks = build_persistent_island_task_matrix(
            self._bindings(),
            selected_density=64,
        )

    def test_task_matrix_is_exact_and_binds_all_independent_identities(
        self,
    ) -> None:
        self.assertEqual(len(self.tasks), OPEN_ECOLOGY_PHASE_D_TASK_COUNT)
        triplet = prioritized_persistent_island_triplet(self.tasks)

        self.assertEqual(
            tuple(task.arm for task in triplet),
            OPEN_ECOLOGY_PHASE_D_ARM_ORDER,
        )
        self.assertEqual(
            tuple(task.task_id for task in triplet),
            (
                "phase-d-l00-i00-h",
                "phase-d-l00-i00-z",
                "phase-d-l00-i00-r",
            ),
        )
        self.assertEqual(len({task.task_id for task in self.tasks}), 48)
        self.assertEqual(
            len({task.policy_sampling_identity for task in self.tasks}),
            48,
        )
        self.assertEqual(
            len({task.policy_sampling_seed for task in self.tasks}),
            48,
        )
        self.assertEqual(
            triplet[0].environment_seed,
            OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_island"][0],
        )
        self.assertEqual(
            triplet[0].genome_stream_seed,
            OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][0],
        )

    def test_live_source_authority_fails_closed_on_hostile_source_drift(
        self,
    ) -> None:
        task = prioritized_persistent_island_triplet(self.tasks)[0]
        repository_root = Path(__file__).resolve().parents[2]

        def git_probe(*, head: str = _SOURCE_COMMIT, status: str = ""):
            def run(command: list[str], **_: object) -> SimpleNamespace:
                arguments = tuple(command[3:])
                stdout_by_arguments = {
                    ("rev-parse", "--show-toplevel"): f"{repository_root}\n",
                    ("rev-parse", "HEAD"): f"{head}\n",
                    ("status", "--porcelain=v1", "--untracked-files=all"): status,
                }
                return SimpleNamespace(
                    returncode=0,
                    stdout=stdout_by_arguments[arguments],
                )

            return run

        authority_module = "evolution_sim.mind.open_ecology_persistent_island"

        def matching_manifest(_root: Path) -> dict[str, str]:
            return {"aggregate_sha256": "b" * 64}

        with (
            patch(
                f"{authority_module}.subprocess.run",
                side_effect=git_probe(),
            ),
            patch(
                f"{authority_module}.source_file_hash_manifest",
                new=matching_manifest,
            ),
        ):
            _verify_live_source_authority(task, artifact=self.artifact)

        hostile_cases = (
            ("wrong HEAD", git_probe(head="c" * 40), "b" * 64),
            ("dirty checkout", git_probe(status=" M python/file.py\n"), "b" * 64),
            ("manifest mismatch", git_probe(), "c" * 64),
        )
        for label, run_probe, manifest_sha256 in hostile_cases:

            def selected_manifest(
                _root: Path,
                *,
                selected_sha256: str = manifest_sha256,
            ) -> dict[str, str]:
                return {"aggregate_sha256": selected_sha256}

            with (
                self.subTest(case=label),
                patch(
                    f"{authority_module}.subprocess.run",
                    side_effect=run_probe,
                ),
                patch(
                    f"{authority_module}.source_file_hash_manifest",
                    new=selected_manifest,
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "source|HEAD|clean",
                ),
            ):
                _verify_live_source_authority(task, artifact=self.artifact)

        outside_namespace: dict[str, object] = {}
        exec(
            compile(
                "def outside_manifest(_root):\n"
                "    return {'aggregate_sha256': 'b' * 64}\n",
                "/tmp/outside-open-ecology-source.py",
                "exec",
            ),
            outside_namespace,
        )
        with (
            patch(
                f"{authority_module}.source_file_hash_manifest",
                new=outside_namespace["outside_manifest"],
            ),
            self.assertRaisesRegex(ValueError, "outside exact repository"),
        ):
            _verify_live_source_authority(task, artifact=self.artifact)

    def test_real_h_world_advances_twice_without_reset_and_stays_frozen(
        self,
    ) -> None:
        task = prioritized_persistent_island_triplet(self.tasks)[0]
        runner = self._open(task)
        world_identity = id(runner.world)
        model_sha256 = runner.model_state_sha256
        initial_boundary = runner.quiescent_checkpoint

        self.assertEqual(initial_boundary["observed_tick"], 0)
        self.assertTrue(initial_boundary["archive_safe_while_runner_idle"])
        self.assertFalse(initial_boundary["launch_readiness"])
        self.assertEqual(runner.event_counts, {"lineage": 64})
        self.assertEqual(
            runner.frozen_policy_diagnostics["trainable_parameter_count"],
            0,
        )
        self.assertFalse(runner.frozen_policy_diagnostics["model_training"])

        first = runner.advance_to(1)
        second = runner.advance_to(2)
        boundary = runner.quiescent_checkpoint

        self.assertEqual(first.observed_tick, 1)
        self.assertEqual(second.observed_tick, 2)
        self.assertTrue(first.same_world_instance)
        self.assertTrue(second.same_world_instance)
        self.assertEqual(id(runner.world), world_identity)
        self.assertEqual(runner.model_state_sha256, model_sha256)
        self.assertEqual(boundary["observed_tick"], 2)
        continuation = boundary["evidence_continuation"]
        self.assertIsInstance(continuation, dict)
        self.assertEqual(
            continuation["total_event_count"],
            sum(runner.event_counts.values()),
        )
        runner.abort_evidence()

    def test_storage_lock_blocks_mutation_and_idle_boundary_is_archive_safe(
        self,
    ) -> None:
        task = prioritized_persistent_island_triplet(self.tasks)[0]
        runner = self._open(task)
        lock_path = default_storage_lock_path(self.campaign_root)

        with CampaignStorageLock(
            lock_path,
            campaign_id="persistent-test",
            source_git_sha=_SOURCE_COMMIT,
        ):
            with self.assertRaisesRegex(
                CampaignStorageError,
                "locked by another process",
            ):
                runner.advance_to(1)

        self.assertEqual(runner.observed_tick, 0)
        self.assertEqual(runner.advance_to(1).observed_tick, 1)
        self.assertTrue(runner.quiescent_checkpoint["archive_safe_while_runner_idle"])
        runner.abort_evidence()

    def test_distinct_tasks_share_advance_lock_while_global_scan_fails_closed(
        self,
    ) -> None:
        h_task, z_task, _ = prioritized_persistent_island_triplet(self.tasks)
        h_runner = self._open(h_task)
        z_runner = self._open(z_task)
        tick_arrivals = threading.Barrier(3)
        release_ticks = threading.Event()

        def blocked_tick(_world: SimulationWorld) -> tuple[int, int]:
            tick_arrivals.wait(timeout=30)
            if not release_ticks.wait(timeout=30):
                raise TimeoutError("concurrency test did not release tick workers")
            raise RuntimeError("intentional concurrent tick stop")

        with (
            patch.object(SimulationWorld, "_run_tick", new=blocked_tick),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            h_future = executor.submit(h_runner.advance_to, 1)
            z_future = executor.submit(z_runner.advance_to, 1)
            tick_arrivals.wait(timeout=30)
            self.assertTrue(OPEN_ECOLOGY_PARALLEL_CAMPAIGN_ADVANCE_AUTHORIZED)
            with self.assertRaisesRegex(
                CampaignStorageError,
                "locked by another process",
            ):
                with CampaignStorageLock(
                    default_storage_lock_path(self.campaign_root),
                    campaign_id="persistent-test",
                    source_git_sha=_SOURCE_COMMIT,
                ):
                    self.fail("exclusive scan lock entered during shared advances")
            with self.assertRaisesRegex(
                CampaignStorageError,
                "mutation barrier",
            ):
                h_runner.advance_to(0)
            with self.assertRaisesRegex(
                CampaignStorageError,
                "mutation barrier",
            ):
                _ = h_runner.quiescent_checkpoint
            release_ticks.set()
            with self.assertRaisesRegex(
                RuntimeError,
                "intentional concurrent tick stop",
            ):
                h_future.result(timeout=30)
            with self.assertRaisesRegex(
                RuntimeError,
                "intentional concurrent tick stop",
            ):
                z_future.result(timeout=30)

        with CampaignStorageLock(
            default_storage_lock_path(self.campaign_root),
            campaign_id="persistent-test",
            source_git_sha=_SOURCE_COMMIT,
        ):
            pass
        h_runner.abort_evidence()
        z_runner.abort_evidence()

    def test_checkpoint_publication_boundaries_are_globally_exclusive(
        self,
    ) -> None:
        h_task, z_task, _ = prioritized_persistent_island_triplet(self.tasks)
        h_runner = self._open(h_task)
        z_runner = self._open(z_task)
        tick_arrivals = threading.Barrier(3)
        first_boundary_entered = threading.Event()
        release_boundary = threading.Event()
        overlap_observed = threading.Event()
        counter_lock = threading.Lock()
        active_boundaries = 0
        maximum_active_boundaries = 0

        def empty_tick(_world: SimulationWorld) -> tuple[int, int]:
            tick_arrivals.wait(timeout=30)
            return (0, 0)

        def blocked_checkpoint(_runner: PersistentIslandRunner) -> None:
            nonlocal active_boundaries, maximum_active_boundaries
            with counter_lock:
                active_boundaries += 1
                maximum_active_boundaries = max(
                    maximum_active_boundaries,
                    active_boundaries,
                )
                if active_boundaries > 1:
                    overlap_observed.set()
                first_boundary_entered.set()
            if not release_boundary.wait(timeout=30):
                raise TimeoutError("concurrency test did not release publication")
            with counter_lock:
                active_boundaries -= 1

        with (
            patch.object(SimulationWorld, "_run_tick", new=empty_tick),
            patch.object(
                PersistentIslandRunner,
                "_write_runtime_checkpoint_locked",
                new=blocked_checkpoint,
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            h_future = executor.submit(h_runner.advance_to, 1)
            z_future = executor.submit(z_runner.advance_to, 1)
            tick_arrivals.wait(timeout=30)
            self.assertTrue(first_boundary_entered.wait(timeout=30))
            time.sleep(0.2)
            self.assertFalse(overlap_observed.is_set())
            release_boundary.set()
            self.assertEqual(h_future.result(timeout=30).observed_tick, 1)
            self.assertEqual(z_future.result(timeout=30).observed_tick, 1)

        self.assertEqual(maximum_active_boundaries, 1)
        self.assertEqual(active_boundaries, 0)
        h_runner.abort_evidence()
        z_runner.abort_evidence()

    def test_h_z_and_exact_history_reset_r_are_real_world_treatments(self) -> None:
        h_task, z_task, r_task = prioritized_persistent_island_triplet(self.tasks)
        h_runner = self._open(h_task)
        z_runner = self._open(z_task)
        r_runner = self._open(r_task)
        h_metadata = [
            agent.mind_inheritance_metadata for agent in h_runner.world.alive_agents()
        ]
        z_metadata = [
            agent.mind_inheritance_metadata for agent in z_runner.world.alive_agents()
        ]

        self.assertEqual(h_runner.arm_contract["description"], "heritable_history")
        self.assertEqual(
            z_runner.arm_contract["description"],
            "zero_genome_history",
        )
        self.assertEqual(
            r_runner.arm_contract["description"],
            "heritable_reset_each_decision",
        )
        self.assertTrue(
            all(metadata["population_mode"] == "heritable" for metadata in h_metadata)
        )
        self.assertTrue(
            all(metadata["population_mode"] == "zero_all" for metadata in z_metadata)
        )
        self.assertTrue(
            all(
                metadata["genome_sha256"] == zero_recurrent_genome().sha256
                for metadata in z_metadata
            )
        )
        r_result = r_runner.advance_to(2)
        self.assertEqual(r_result.observed_tick, 2)
        self.assertTrue(r_result.same_world_instance)
        self.assertEqual(
            r_runner.runtime_history_diagnostics,
            {
                "reset_recurrent_state_each_decision": True,
                "recurrent_state_agent_count": 0,
                "previous_public_feedback_agent_count": 0,
                "pending_decision_count": 0,
            },
        )
        h_runner.abort_evidence()
        z_runner.abort_evidence()
        r_runner.abort_evidence()

    def test_natural_extinction_is_terminal_and_never_reseeds(self) -> None:
        task = prioritized_persistent_island_triplet(self.tasks)[0]
        runner = self._open(task)
        world_identity = id(runner.world)
        for agent in runner.world.alive_agents():
            agent.health = 0.0
            agent.energy = -1_000_000.0
            agent.hydration = -1_000_000.0

        terminal = runner.advance_to(1)
        after = runner.advance_to(10_000)

        self.assertTrue(terminal.extinct)
        self.assertEqual(terminal.extinction_tick, 1)
        self.assertEqual(after.observed_tick, 1)
        self.assertEqual(id(runner.world), world_identity)
        self.assertEqual(runner.world.alive_agents(), [])
        self.assertEqual(runner.event_counts["death"], 64)
        self.assertTrue(terminal.summaries[0]["terminal"])
        checkpoint_metadata = runner.latest_runtime_checkpoint
        assert checkpoint_metadata is not None
        aggregate_pins = runner.latest_aggregate_resume_pins
        assert aggregate_pins is not None
        restored = PersistentIslandRunner.restore_from_current(
            task,
            campaign_root=self.campaign_root,
            campaign_id="persistent-test",
            pins=aggregate_pins,
        )
        self.assertEqual(restored.summaries, runner.summaries)
        self.assertEqual(restored.first_milestone, runner.first_milestone)
        self.assertEqual(restored.event_counts, runner.event_counts)
        manifest = runner.finish_evidence()
        self.assertEqual(manifest["status"], "complete")

    def test_disk_checkpoint_fresh_runner_matches_uninterrupted_h_z_r(self) -> None:
        for task in prioritized_persistent_island_triplet(self.tasks):
            with self.subTest(arm=task.arm):
                baseline_root = self.campaign_root / f"baseline-{task.arm}"
                resumed_root = self.campaign_root / f"resumed-{task.arm}"
                baseline_root.mkdir()
                baseline = self._open(
                    task,
                    campaign_root=baseline_root,
                    checkpoint_interval_ticks=2,
                )
                baseline.advance_to(2)
                self.assertIsInstance(
                    baseline.world.cached_signal_state.communication_receiver_projection,
                    CommunicationReceiverProjection,
                )
                checkpoint_metadata = baseline.latest_runtime_checkpoint
                assert checkpoint_metadata is not None
                aggregate_pins = baseline.latest_aggregate_resume_pins
                assert aggregate_pins is not None
                checkpoint_path = Path(str(checkpoint_metadata["path"]))
                checkpoint = load_open_ecology_checkpoint(
                    checkpoint_path,
                    expected_source_git_sha=_SOURCE_COMMIT,
                    require_restartable=True,
                )
                self.assertEqual(
                    set(checkpoint["components"]),
                    set(REQUIRED_CHECKPOINT_COMPONENTS),
                )
                self.assertLessEqual(checkpoint_path.stat().st_size, 256 * 1024 * 1024)

                shutil.copytree(baseline_root, resumed_root)
                resumed = PersistentIslandRunner.restore_from_current(
                    task,
                    campaign_root=resumed_root,
                    campaign_id="persistent-test",
                    pins=aggregate_pins,
                )
                self.assertNotEqual(id(baseline.world), id(resumed.world))
                self.assertEqual(resumed.observed_tick, 2)
                self.assertTrue(
                    resumed.quiescent_checkpoint["restartable_world_checkpoint_present"]
                )
                self.assertTrue(
                    resumed.quiescent_checkpoint["aggregate_current_restartable"]
                )

                baseline_result = baseline.advance_to(5)
                resumed_result = resumed.advance_to(5)
                self.assertEqual(baseline_result, resumed_result)
                self._assert_runtime_equal(baseline, resumed)
                self.assertEqual(baseline.summaries, resumed.summaries)
                self.assertEqual(baseline.first_milestone, resumed.first_milestone)
                self.assertEqual(baseline.event_counts, resumed.event_counts)
                self.assertEqual(
                    baseline._interval.to_dict(),
                    resumed._interval.to_dict(),
                )
                self.assertEqual(
                    baseline.quiescent_checkpoint["evidence_continuation"],
                    resumed.quiescent_checkpoint["evidence_continuation"],
                )
                self.assertEqual(
                    load_open_ecology_evidence_manifest(
                        baseline_root / task.task_id,
                        verify_shards=True,
                    ),
                    load_open_ecology_evidence_manifest(
                        resumed_root / task.task_id,
                        verify_shards=True,
                    ),
                )
                baseline_checkpoint_names = sorted(
                    path.name
                    for path in (baseline_root / "checkpoints" / task.task_id).glob(
                        "checkpoint-observed-*.json"
                    )
                )
                resumed_checkpoint_names = sorted(
                    path.name
                    for path in (resumed_root / "checkpoints" / task.task_id).glob(
                        "checkpoint-observed-*.json"
                    )
                )
                self.assertEqual(
                    baseline_checkpoint_names,
                    [
                        "checkpoint-observed-00000002.json",
                        "checkpoint-observed-00000004.json",
                        "checkpoint-observed-00000005.json",
                    ],
                )
                self.assertEqual(
                    baseline_checkpoint_names,
                    resumed_checkpoint_names,
                )
                baseline.abort_evidence()
                resumed.abort_evidence()
                del baseline, resumed, checkpoint
                gc.collect()
                shutil.rmtree(baseline_root)
                shutil.rmtree(resumed_root)

    def test_third_checkpoint_is_not_pruned_without_verified_archive_receipt(
        self,
    ) -> None:
        task = prioritized_persistent_island_triplet(self.tasks)[0]
        checkpoint_directory = self.campaign_root / "checkpoints" / task.task_id
        checkpoint_directory.mkdir(parents=True)
        older = checkpoint_directory / "checkpoint-observed-00000000.json"
        newer = checkpoint_directory / "checkpoint-observed-00000002.json"
        older.write_bytes(b"hostile-unverified-older-checkpoint\n")
        newer.write_bytes(b"hostile-unverified-newer-checkpoint\n")
        runner = self._open(task, checkpoint_interval_ticks=1)

        runner.advance_to(1)
        latest = runner.latest_runtime_checkpoint
        assert latest is not None
        frontier = runner.campaign_barrier_frontier()
        self.assertEqual(frontier["task_id"], task.task_id)
        self.assertEqual(frontier["observed_tick"], 1)
        self.assertEqual(frontier["aggregate_completed_world_tick"], 0)
        self.assertTrue(frontier["aggregate_current_restartable"])
        self.assertTrue(frontier["aggregate_resume_authorized"])
        hostile_frontier = deepcopy(frontier)
        hostile_frontier["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            CampaignStorageError,
            "frontier digest drifted",
        ):
            with PersistentIslandRunner.campaign_storage_barrier(
                campaign_root=self.campaign_root,
                campaign_id="persistent-test",
                source_git_sha=_SOURCE_COMMIT,
                frontiers=(hostile_frontier,),
            ):
                self.fail("tampered campaign frontier entered storage barrier")
        with self.assertRaisesRegex(
            CampaignStorageError,
            "duplicate task ids",
        ):
            with PersistentIslandRunner.campaign_storage_barrier(
                campaign_root=self.campaign_root,
                campaign_id="persistent-test",
                source_git_sha=_SOURCE_COMMIT,
                frontiers=(frontier, frontier),
            ):
                self.fail("duplicate task frontier entered storage barrier")
        with PersistentIslandRunner.campaign_storage_barrier(
            campaign_root=self.campaign_root,
            campaign_id="persistent-test",
            source_git_sha=_SOURCE_COMMIT,
            frontiers=(frontier,),
        ) as pinned_frontiers:
            self.assertEqual(pinned_frontiers, (frontier,))
            with self.assertRaisesRegex(
                CampaignStorageError,
                "mutation barrier",
            ):
                runner.advance_to(2)
            self.assertEqual(runner.observed_tick, 1)
        profile = latest["profile"]
        assert isinstance(profile, dict)
        component_bytes = profile["component_canonical_envelope_bytes"]
        assert isinstance(component_bytes, dict)
        self.assertEqual(set(component_bytes), set(REQUIRED_CHECKPOINT_COMPONENTS))
        self.assertTrue(all(value > 0 for value in component_bytes.values()))
        self.assertEqual(
            profile["component_canonical_envelope_total_bytes"],
            sum(component_bytes.values()),
        )
        for field_name in (
            "capture_elapsed_ns",
            "write_elapsed_ns",
            "readback_elapsed_ns",
            "aggregate_publish_elapsed_ns",
            "checkpoint_pipeline_elapsed_ns",
        ):
            self.assertGreater(profile[field_name], 0)
        checkpoint_paths = [older, Path(str(latest["path"])), newer]
        self.assertTrue(all(path.is_file() for path in checkpoint_paths))
        self.assertEqual(
            older.read_bytes(),
            b"hostile-unverified-older-checkpoint\n",
        )
        self.assertEqual(
            newer.read_bytes(),
            b"hostile-unverified-newer-checkpoint\n",
        )
        self.assertEqual(
            sorted(path.name for path in checkpoint_paths),
            [
                "checkpoint-observed-00000000.json",
                "checkpoint-observed-00000001.json",
                "checkpoint-observed-00000002.json",
            ],
        )
        manifest_before = load_open_ecology_evidence_manifest(
            self.campaign_root / task.task_id,
            verify_shards=True,
        )
        loose_restore = PersistentIslandRunner.restore_from_checkpoint(
            task,
            checkpoint_path=Path(str(latest["path"])),
            campaign_root=self.campaign_root,
            campaign_id="persistent-test",
        )
        with self.assertRaisesRegex(ValueError, "inspection-only"):
            loose_restore.record_intervention(
                intervention_id="must-not-write",
                intervention_kind="hostile_loose_restore",
                branch_id="untrusted",
            )
        self.assertEqual(
            load_open_ecology_evidence_manifest(
                self.campaign_root / task.task_id,
                verify_shards=True,
            ),
            manifest_before,
        )
        runner.abort_evidence()

    def _open(
        self,
        task: PersistentIslandTask,
        *,
        campaign_root: Path | None = None,
        checkpoint_interval_ticks: int = 5_000,
    ) -> PersistentIslandRunner:
        root = self.campaign_root if campaign_root is None else campaign_root
        return PersistentIslandRunner.open(
            task,
            evidence_directory=root / task.task_id,
            campaign_root=root,
            campaign_id="persistent-test",
            checkpoint_interval_ticks=checkpoint_interval_ticks,
        )

    def _assert_runtime_equal(
        self,
        baseline: PersistentIslandRunner,
        resumed: PersistentIslandRunner,
    ) -> None:
        self.assertEqual(baseline.world.grid, resumed.world.grid)
        self.assertEqual(baseline.world.agents, resumed.world.agents)
        self.assertEqual(
            baseline.world.rng.getstate(),
            resumed.world.rng.getstate(),
        )
        self.assertEqual(
            deepcopy(baseline.world.tick_trajectory_records),
            deepcopy(resumed.world.tick_trajectory_records),
        )
        baseline_policy = baseline.world.policy
        resumed_policy = resumed.world.policy
        self.assertEqual(
            baseline_policy._decision_index,
            resumed_policy._decision_index,
        )
        self.assertEqual(
            baseline_policy._feedback_by_agent,
            resumed_policy._feedback_by_agent,
        )
        self.assertEqual(
            baseline_policy._public_history_by_agent,
            resumed_policy._public_history_by_agent,
        )
        self.assertEqual(
            set(baseline_policy._state_by_agent),
            set(resumed_policy._state_by_agent),
        )
        for agent_id, state in baseline_policy._state_by_agent.items():
            self.assertTrue(
                torch.equal(state, resumed_policy._state_by_agent[agent_id])
            )
        self.assertEqual(
            baseline_policy._sampling_generator.get_state().tolist(),
            resumed_policy._sampling_generator.get_state().tolist(),
        )
        self.assertEqual(
            baseline_policy._genome_population_manager.snapshot_artifact(),
            resumed_policy._genome_population_manager.snapshot_artifact(),
        )

    def _bindings(self) -> tuple[PersistentArtifactBinding, ...]:
        return tuple(
            PersistentArtifactBinding(
                learner_index=index,
                learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][index],
                artifact_path=str(self.artifact_path),
                artifact_sha256=str(self.artifact["artifact_sha256"]),
                artifact_file_sha256=self.artifact_file_sha256,
                source_commit=_SOURCE_COMMIT,
                terminal_authority_sha256="2" * 64,
            )
            for index in range(4)
        )


if __name__ == "__main__":
    unittest.main()
