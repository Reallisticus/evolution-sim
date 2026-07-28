from __future__ import annotations

from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evolution_sim.io.open_ecology_aggregate_commit import (
    OpenEcologyAggregateIdentityPins,
    OpenEcologyAggregateResumePins,
)
from evolution_sim.io.open_ecology_campaign_storage import (
    StorageEntry,
    StorageScan,
    canonical_json_bytes,
    check_campaign_storage,
)
from evolution_sim.mind import open_ecology_campaign_coordinator as coordinator_module
from evolution_sim.mind.open_ecology_campaign_coordinator import (
    OpenEcologyCampaignCoordinator,
    OpenEcologyCampaignCoordinatorError,
    _atomic_create,
    _load_barrier_failure_observation,
    _load_barrier_intent,
    build_static_worker_assignments,
    load_campaign_frontier_receipt,
)
from evolution_sim.mind.open_ecology_persistent_island import (
    OPEN_ECOLOGY_PHASE_D_TASK_COUNT,
    PersistentAdvanceResult,
    PersistentArtifactBinding,
    build_persistent_island_task_matrix,
)
from evolution_sim.mind.open_ecology_process_workers import (
    PersistentProcessWorkerLauncher,
    build_process_worker_slots,
    local_process_worker_host_identity,
    process_worker_assignment_sha256,
)
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_SEED_REGISTRY,
)


REQUIRES_MIND_ML = True
_SOURCE_COMMIT = "a" * 40
_SOURCE_MANIFEST = "b" * 64


def _digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class _FakeRunner:
    def __init__(
        self,
        task,
        *,
        stale: bool = False,
        terminal_tick: int | None = None,
    ) -> None:
        self.task = task
        self._observed_tick = 0
        self._pins = None
        self._stale = stale
        self._terminal_tick = terminal_tick

    @property
    def observed_tick(self) -> int:
        return self._observed_tick

    @property
    def extinct(self) -> bool:
        return (
            self._terminal_tick is not None
            and self._observed_tick >= self._terminal_tick
        )

    @property
    def latest_aggregate_resume_pins(self):
        return self._pins

    @property
    def quiescent_checkpoint(self):
        return {
            "task_id": self.task.task_id,
            "campaign_id": "campaign-test",
            "campaign_root": str(self._campaign_root),
            "source_git_sha": _SOURCE_COMMIT,
            "observed_tick": self._observed_tick,
            "archive_safe_while_runner_idle": True,
            "restartable_world_checkpoint_present": self._pins is not None,
            "aggregate_current_restartable": self._pins is not None,
            "aggregate_resume_authorized": True,
            "boundary_sha256": "f" * 64,
        }

    def bind_campaign_root(self, campaign_root: Path) -> None:
        self._campaign_root = campaign_root

    def advance_to(self, target_tick: int) -> PersistentAdvanceResult:
        if self._terminal_tick is None:
            self._observed_tick = target_tick - 1 if self._stale else target_tick
        else:
            self._observed_tick = min(target_tick, self._terminal_tick)
        digest = hashlib.sha256(self.task.task_id.encode()).hexdigest()
        if self._pins is None or not self.extinct:
            self._pins = OpenEcologyAggregateResumePins(
                identity=OpenEcologyAggregateIdentityPins(
                    source_git_sha=_SOURCE_COMMIT,
                    config_contract_sha256="c" * 64,
                    seed_contract_sha256="d" * 64,
                    run_generation_id=(
                        f"phase-d:{self.task.task_id}:"
                        f"{hashlib.sha256(b'campaign-test').hexdigest()[:16]}"
                    ),
                    island_id=self.task.task_id,
                    simulation_generation_index=0,
                    tick=self._observed_tick - 1,
                ),
                aggregate_generation_index=max(0, target_tick // 5_000 - 1),
                commit_sha256=digest,
                checkpoint_sha256="1" * 64,
                checkpoint_generation_identity_sha256="2" * 64,
                evidence_manifest_sha256="3" * 64,
                evidence_manifest_status="open",
            )
        return PersistentAdvanceResult(
            task_id=self.task.task_id,
            requested_target_tick=target_tick,
            observed_tick=self._observed_tick,
            extinct=self.extinct,
            extinction_tick=self._terminal_tick if self.extinct else None,
            summaries=(),
            milestones=(),
            model_state_sha256="4" * 64,
            same_world_instance=True,
        )

    def campaign_barrier_frontier(self):
        payload = {
            "schema_version": ("mind_v3_open_ecology_campaign_barrier_frontier_v1"),
            "task_id": self.task.task_id,
            "task_sha256": _digest(self.task.to_dict()),
            "campaign_id": "campaign-test",
            "campaign_root": str(self._campaign_root),
            "source_git_sha": _SOURCE_COMMIT,
            "observed_tick": self._observed_tick,
            "extinct": self.extinct,
            "extinction_tick": self._terminal_tick if self.extinct else None,
            "config_contract_sha256": "c" * 64,
            "seed_contract_sha256": "d" * 64,
            "world_config_sha256": self.task.world_config_sha256,
            "aggregate_generation_index": self._pins.aggregate_generation_index,
            "aggregate_completed_world_tick": self._pins.identity.tick,
            "aggregate_commit_sha256": self._pins.commit_sha256,
            "checkpoint_sha256": self._pins.checkpoint_sha256,
            "checkpoint_generation_identity_sha256": (
                self._pins.checkpoint_generation_identity_sha256
            ),
            "evidence_manifest_sha256": self._pins.evidence_manifest_sha256,
            "evidence_manifest_status": "open",
            "evidence_continuation_sha256": "e" * 64,
            "aggregate_current_restartable": True,
            "aggregate_resume_authorized": True,
        }
        payload["frontier_sha256"] = _digest(payload)
        return payload


class _FakeStorageScanner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.generation = 0
        self.calls = 0

    def __call__(self, *_args, **_kwargs) -> StorageScan:
        self.calls += 1
        entries = ()
        total_bytes = 0
        if self.generation:
            entries = (
                StorageEntry(
                    path="hostile-drift.bin",
                    kind="file",
                    mode=0o444,
                    size=1,
                    sha256="9" * 64,
                    device=1,
                    inode=2,
                    mtime_ns=3,
                    link_count=1,
                ),
            )
            total_bytes = 1
        return StorageScan(
            root=self.root,
            entries=entries,
            total_file_bytes=total_bytes,
            free_bytes=500 * 1024**3,
            filesystem_total_bytes=1024 * 1024**3,
        )


class OpenEcologyCampaignCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.campaign_root = self.base / "campaign"
        self.receipt_directory = self.base / "receipts"
        self.campaign_root.mkdir()
        self.tasks = build_persistent_island_task_matrix(
            self._bindings(),
            selected_density=64,
        )
        self.runners = {task.task_id: _FakeRunner(task) for task in self.tasks}
        for runner in self.runners.values():
            runner.bind_campaign_root(self.campaign_root)
        self.scanner = _FakeStorageScanner(self.campaign_root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_immutable_publication_cannot_replace_a_racing_destination(self) -> None:
        destination = self.base / "frontier.json"
        real_rename = coordinator_module._rename_name_no_replace

        def publish_competitor_then_rename(
            parent_descriptor: int,
            source_name: str,
            destination_name: str,
        ) -> None:
            competitor = os.open(
                destination_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o444,
                dir_fd=parent_descriptor,
            )
            try:
                coordinator_module._write_all(competitor, b"competitor\n")
                os.fsync(competitor)
            finally:
                os.close(competitor)
            real_rename(
                parent_descriptor,
                source_name,
                destination_name,
            )

        with patch.object(
            coordinator_module,
            "_rename_name_no_replace",
            side_effect=publish_competitor_then_rename,
        ):
            with self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "refusing to overwrite",
            ):
                _atomic_create(destination, b"authoritative\n")

        self.assertEqual(destination.read_bytes(), b"competitor\n")
        stages = list(self.base.glob(f".{destination.name}.pending-*"))
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].read_bytes(), b"authoritative\n")

    def test_immutable_publication_rejects_a_replaced_stage_and_keeps_competitor(
        self,
    ) -> None:
        destination = self.base / "frontier.json"
        real_rename = coordinator_module._rename_name_no_replace

        def replace_stage_then_publish(
            parent_descriptor: int,
            source_name: str,
            destination_name: str,
        ) -> None:
            os.unlink(source_name, dir_fd=parent_descriptor)
            competitor = os.open(
                source_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o444,
                dir_fd=parent_descriptor,
            )
            try:
                coordinator_module._write_all(
                    competitor,
                    b"stage-competitor\n",
                )
                os.fchmod(competitor, 0o444)
                os.fsync(competitor)
            finally:
                os.close(competitor)
            real_rename(
                parent_descriptor,
                source_name,
                destination_name,
            )

        with patch.object(
            coordinator_module,
            "_rename_name_no_replace",
            side_effect=replace_stage_then_publish,
        ):
            with self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "identity changed",
            ):
                _atomic_create(destination, b"authoritative\n")

        self.assertEqual(destination.read_bytes(), b"stage-competitor\n")

    def test_immutable_publication_detects_post_open_destination_replacement(
        self,
    ) -> None:
        destination = self.base / "frontier.json"
        retained = self.base / "retained-frontier.json"
        real_readback = coordinator_module._read_exact_descriptor
        readback_count = 0

        def replace_destination_during_readback(
            descriptor: int,
            *,
            expected_bytes: int,
        ) -> bytes:
            nonlocal readback_count
            readback_count += 1
            observed = real_readback(
                descriptor,
                expected_bytes=expected_bytes,
            )
            if readback_count == 2:
                os.replace(destination, retained)
                destination.write_bytes(b"post-open-competitor\n")
                destination.chmod(0o444)
            return observed

        with patch.object(
            coordinator_module,
            "_read_exact_descriptor",
            side_effect=replace_destination_during_readback,
        ):
            with self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "changed during publication|changed after readback",
            ):
                _atomic_create(destination, b"authoritative\n")

        self.assertEqual(readback_count, 2)
        self.assertEqual(destination.read_bytes(), b"post-open-competitor\n")
        self.assertEqual(retained.read_bytes(), b"authoritative\n")

    def test_receipt_directory_creation_rejects_a_symlink_ancestor(self) -> None:
        outside = self.base / "outside"
        outside.mkdir()
        hostile = self.base / "receipt-alias"
        hostile.symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "receipt_directory is unsafe.*not a real directory",
        ):
            self._coordinator(receipt_directory=hostile / "nested")

        self.assertEqual(list(outside.iterdir()), [])

    def test_rejected_receipt_directory_inside_campaign_is_never_created(
        self,
    ) -> None:
        rejected = self.campaign_root / "must-not-create-receipts"

        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "outside the scanned active campaign root",
        ):
            self._coordinator(receipt_directory=rejected)

        self.assertFalse(rejected.exists())

    def test_intent_is_not_followed_by_runner_mutation_until_exact_readback(
        self,
    ) -> None:
        coordinator = self._coordinator()
        real_readback = coordinator_module._read_exact_descriptor
        readback_count = 0

        def corrupt_published_readback(
            descriptor: int,
            *,
            expected_bytes: int,
        ) -> bytes:
            nonlocal readback_count
            readback_count += 1
            observed = real_readback(
                descriptor,
                expected_bytes=expected_bytes,
            )
            if readback_count == 2:
                return b"corrupt-readback"
            return observed

        with patch.object(
            coordinator_module,
            "_read_exact_descriptor",
            side_effect=corrupt_published_readback,
        ):
            with self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "bytes changed during publication",
            ):
                coordinator.advance_one_barrier()

        self.assertEqual(readback_count, 2)
        self.assertTrue(
            all(runner.observed_tick == 0 for runner in self.runners.values())
        )

    def test_exact_matrix_static_assignment_and_read_only_status(self) -> None:
        assignments = build_static_worker_assignments(
            self.tasks,
            worker_count=4,
        )
        self.assertEqual(len(assignments), OPEN_ECOLOGY_PHASE_D_TASK_COUNT)
        self.assertEqual(
            tuple(assignment.task_id for assignment in assignments[:3]),
            (
                "phase-d-l00-i00-h",
                "phase-d-l00-i00-z",
                "phase-d-l00-i00-r",
            ),
        )
        self.assertTrue(all(item.qualification_priority for item in assignments[:3]))
        self.assertEqual(
            tuple(item.worker_index for item in assignments[:4]),
            (0, 1, 2, 3),
        )

        coordinator = self._coordinator()
        before = tuple(runner.observed_tick for runner in self.runners.values())
        status = coordinator.status()
        after = tuple(runner.observed_tick for runner in self.runners.values())
        self.assertEqual(before, after)
        self.assertTrue(status["read_only"])
        self.assertFalse(status["pruning_or_deletion_available"])
        self.assertEqual(len(status["tasks"]), 48)

    def test_barrier_writes_complete_external_receipt_and_chains_next(self) -> None:
        coordinator = self._coordinator()
        first = coordinator.advance_one_barrier()

        self.assertEqual(first.frontier_tick, 5_000)
        self.assertEqual(first.path.parent, self.receipt_directory)
        self.assertFalse(first.path.is_relative_to(self.campaign_root))
        self.assertEqual(len(first.resume_pins_by_task()), 3)
        intent_binding = first.payload["barrier_intent"]
        intent_path = Path(intent_binding["path"])
        self.assertTrue(intent_path.is_file())
        self.assertEqual(
            hashlib.sha256(intent_path.read_bytes()).hexdigest(),
            intent_binding["intent_sha256"],
        )
        self.assertEqual(first.payload["barrier_scope"], "qualification_triplet")
        self.assertTrue(first.payload["all_started_tasks_quiescent"])
        self.assertFalse(first.payload["all_tasks_at_common_frontier"])
        self.assertTrue(first.payload["next_interval_authorized"])
        self.assertFalse(first.payload["pruning_or_deletion_performed"])
        self.assertEqual(
            [record["observed_tick"] for record in first.payload["tasks"]],
            [5_000, 5_000, 5_000, *([0] * 45)],
        )
        self.assertEqual(
            load_campaign_frontier_receipt(first.path).receipt_sha256,
            first.receipt_sha256,
        )

        second = coordinator.advance_one_barrier()
        self.assertEqual(
            self.scanner.calls,
            2,
            "an uninterrupted process performs one complete post-barrier scan "
            "per frontier without redundantly rehashing the predecessor",
        )
        self.assertEqual(second.frontier_tick, 10_000)
        self.assertEqual(second.payload["matrix_frontier_tick"], 0)
        self.assertEqual(
            second.payload["previous_frontier_receipt_sha256"],
            first.receipt_sha256,
        )
        self.assertTrue(
            second.payload["qualification"]["milestone_reached_or_terminal"]
        )

        later = coordinator.advance_through(15_000)
        self.assertEqual(
            [(item.payload["barrier_scope"], item.frontier_tick) for item in later],
            [
                ("matrix_catchup", 5_000),
                ("matrix_catchup", 10_000),
                ("full_matrix", 15_000),
            ],
        )
        self.assertTrue(later[1].payload["all_tasks_at_common_frontier"])
        self.assertTrue(later[-1].payload["all_tasks_at_common_frontier"])
        self.assertEqual(len(later[-1].resume_pins_by_task()), 48)
        self.assertEqual(
            [record["observed_tick"] for record in later[-1].payload["tasks"]],
            [15_000] * 48,
        )
        completion = coordinator.advance_through(50_000)
        self.assertEqual(completion[-1].frontier_tick, 50_000)
        self.assertFalse(completion[-1].payload["next_interval_authorized"])
        self.assertTrue(completion[-1].payload["all_tasks_at_common_frontier"])
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "already at",
        ):
            coordinator.advance_one_barrier()

    def test_missing_task_and_source_mismatch_fail_before_advance(self) -> None:
        missing = dict(self.runners)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "runner set.*missing",
        ):
            self._coordinator(runners=missing)

        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "source.*task artifact",
        ):
            self._coordinator(source_git_sha="e" * 40)
        self.assertTrue(
            all(runner.observed_tick == 0 for runner in self.runners.values())
        )

        triplet_only = {
            task.task_id: self.runners[task.task_id] for task in self.tasks[:3]
        }
        qualification = self._coordinator(runners=triplet_only)
        qualification.advance_through(10_000)
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "selected campaign tasks have no runner",
        ):
            qualification.advance_one_barrier()

    def test_production_advance_requires_a_real_health_probe(self) -> None:
        coordinator = OpenEcologyCampaignCoordinator(
            tasks=self.tasks,
            runners=self.runners,
            campaign_root=self.campaign_root,
            receipt_directory=self.receipt_directory,
            campaign_id="campaign-test",
            source_git_sha=_SOURCE_COMMIT,
            source_manifest_sha256=_SOURCE_MANIFEST,
            worker_count=1,
            source_probe=lambda: {
                "repository_root": "/exact/checkout",
                "source_git_sha": _SOURCE_COMMIT,
                "source_manifest_sha256": _SOURCE_MANIFEST,
                "git_clean": True,
            },
            storage_scanner=self.scanner,
            barrier_context_factory=lambda _frontiers: nullcontext(),
        )
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "resource/health probe is required",
        ):
            coordinator.advance_one_barrier()
        self.assertTrue(
            all(runner.observed_tick == 0 for runner in self.runners.values())
        )

        unhealthy = self._coordinator()
        unhealthy._health_probe = lambda phase, frontier: {
            "phase": phase,
            "frontier_tick": frontier,
            "healthy": False,
        }
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "did not affirm",
        ):
            unhealthy.advance_one_barrier()
        self.assertTrue(
            all(runner.observed_tick == 0 for runner in self.runners.values())
        )

        created: list[str] = []

        def deferred_factory(task):
            created.append(task.task_id)
            runner = _FakeRunner(task)
            runner.bind_campaign_root(self.campaign_root)
            return runner

        deferred = self._coordinator(
            runners={},
            pending_runner_factory=deferred_factory,
        )
        deferred._health_probe = lambda phase, frontier: {
            "phase": phase,
            "frontier_tick": frontier,
            "healthy": False,
        }
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "did not affirm",
        ):
            deferred.advance_one_barrier()
        self.assertEqual(created, [])

    def test_stale_task_frontier_does_not_write_storage_receipt(self) -> None:
        hostile_id = self.tasks[2].task_id
        self.runners[hostile_id] = _FakeRunner(self.tasks[2], stale=True)
        self.runners[hostile_id].bind_campaign_root(self.campaign_root)
        coordinator = self._coordinator()

        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "live runner failed|stale",
        ):
            coordinator.advance_one_barrier()
        self.assertEqual(list(self.receipt_directory.glob("step-*.json")), [])
        intents = list(self.receipt_directory.glob("barrier-intent-*.json"))
        self.assertEqual(len(intents), 1)
        self.assertTrue(intents[0].is_file())
        failures = list(self.receipt_directory.glob("barrier-failure-*.json"))
        self.assertEqual(len(failures), 1)
        failure = json.loads(failures[0].read_bytes())
        self.assertFalse(failure["receipt_published"])
        self.assertFalse(failure["new_attempt_authorized"])

    def test_unresolved_barrier_intent_blocks_a_fresh_second_attempt(self) -> None:
        coordinator = self._coordinator()
        priority = {task.task_id for task in self.tasks[:3]}
        coordinator._begin_barrier_intent(
            scope="qualification_triplet",
            target_tick=5_000,
            selected_task_ids=priority,
            allow_existing_unresolved=False,
        )

        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "unresolved barrier intent",
        ):
            coordinator.advance_one_barrier()

        self.assertTrue(
            all(runner.observed_tick == 0 for runner in self.runners.values())
        )
        self.assertFalse(tuple(self.receipt_directory.glob("step-*.json")))

    def test_unresolved_intent_reconciles_mixed_workers_and_binds_failure(
        self,
    ) -> None:
        slots = build_process_worker_slots(
            self.tasks,
            worker_count=1,
            host_identity=local_process_worker_host_identity(),
        )
        launcher_arguments = {
            "tasks": self.tasks,
            "campaign_root": self.campaign_root,
            "campaign_id": "campaign-test",
            "repository_root": Path(__file__).resolve().parents[2],
            "source_git_sha": _SOURCE_COMMIT,
            "source_manifest_sha256": _SOURCE_MANIFEST,
            "slots": slots,
            "assignment_sha256": process_worker_assignment_sha256(
                self.tasks,
                slots,
            ),
            "response_timeout_seconds": 15.0,
            "startup_timeout_seconds": 15.0,
            "shutdown_timeout_seconds": 5.0,
        }
        selected = {task.task_id for task in self.tasks[:3]}
        with PersistentProcessWorkerLauncher(
            **launcher_arguments,
            _test_behaviors={},
        ) as original_launcher:
            original = self._coordinator(
                runners={},
                worker_launcher=original_launcher,
            )
            intent = original._begin_barrier_intent(
                scope="qualification_triplet",
                target_tick=5_000,
                selected_task_ids=selected,
                allow_existing_unresolved=False,
            )
            original._record_barrier_failure(
                barrier_intent=intent,
                selected_task_ids=selected,
                error=RuntimeError("simulated parent loss after mixed publication"),
            )

        failure_path = next(self.receipt_directory.glob("barrier-failure-*.json"))
        failure_before = failure_path.read_bytes()
        intent_before = Path(str(intent["path"])).read_bytes()
        with PersistentProcessWorkerLauncher(
            **launcher_arguments,
            _test_behaviors={0: "reconcile_mixed"},
        ) as recovery_launcher:
            recovered = self._coordinator(
                runners={},
                worker_launcher=recovery_launcher,
            )
            receipt = recovered.advance_one_barrier()

            recovery = receipt.payload["barrier_recovery"]
            self.assertTrue(recovery["original_intent_reused"])
            self.assertFalse(recovery["new_attempt_authorized"])
            self.assertTrue(recovery["failure_journal_preserved"])
            self.assertEqual(
                set(recovery["disposition_by_task"].values()),
                {
                    "already_committed_target",
                    "advanced_from_exact_predecessor",
                    "recovered_original_precurrent_attempt",
                },
            )
            self.assertEqual(failure_path.read_bytes(), failure_before)
            self.assertEqual(Path(str(intent["path"])).read_bytes(), intent_before)
            self.assertEqual(
                len(tuple(self.receipt_directory.glob("barrier-intent-*.json"))),
                1,
            )
            self.assertEqual(receipt.frontier_tick, 5_000)

            reloaded = self._coordinator(
                runners={},
                worker_launcher=recovery_launcher,
                authorized_frontier_tick=5_000,
                previous_frontier_receipt=receipt,
            )
            self.assertEqual(reloaded.authorized_frontier_tick, 5_000)

        failure_path.chmod(0o600)
        failure_payload = json.loads(failure_path.read_bytes())
        failure_payload["error_message"] = "forged"
        failure_path.write_bytes(canonical_json_bytes(failure_payload))
        failure_path.chmod(0o400)
        with PersistentProcessWorkerLauncher(
            **launcher_arguments,
            _test_behaviors={},
        ) as hostile_launcher:
            with self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "failure observation digest",
            ):
                self._coordinator(
                    runners={},
                    worker_launcher=hostile_launcher,
                    authorized_frontier_tick=5_000,
                    previous_frontier_receipt=receipt,
                )

    def test_runner_without_campaign_barrier_frontier_fails_closed(self) -> None:
        runner = self.runners[self.tasks[0].task_id]
        runner.campaign_barrier_frontier = None  # type: ignore[assignment]
        coordinator = self._coordinator()

        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "callable campaign barrier frontier",
        ):
            coordinator.advance_one_barrier()

        self.assertFalse(tuple(self.receipt_directory.glob("step-*.json")))

    def test_every_authoritative_barrier_uses_complete_sha256_scan(self) -> None:
        with patch(
            "evolution_sim.io.open_ecology_campaign_storage._descriptor_disk_usage",
            return_value=(2 * 1024**4, 1024**4),
        ):
            coordinator = self._coordinator(scanner=check_campaign_storage)
            first = coordinator.advance_one_barrier()
            second = coordinator.advance_one_barrier()

            self.assertEqual(
                first.payload["storage"]["validation_mode"],
                "complete_sha256_v1",
            )
            self.assertEqual(
                second.payload["storage"]["validation_mode"],
                "complete_sha256_v1",
            )
            self.assertIsNone(
                second.payload["storage"]["predecessor_entry_manifest_sha256"]
            )
            terminal = coordinator.advance_through(50_000)[-1]
            self.assertEqual(
                terminal.payload["storage"]["validation_mode"],
                "terminal_complete_sha256_v1",
            )
            self.assertIsNone(
                terminal.payload["storage"]["predecessor_entry_manifest_sha256"]
            )

    def test_same_size_restored_mtime_cannot_reuse_barrier_file_hash(self) -> None:
        tracked = self.campaign_root / "same-size.bin"
        tracked.write_bytes(b"before")
        original = tracked.stat()
        with patch(
            "evolution_sim.io.open_ecology_campaign_storage._descriptor_disk_usage",
            return_value=(2 * 1024**4, 1024**4),
        ):
            coordinator = self._coordinator(scanner=check_campaign_storage)
            first = coordinator.advance_one_barrier()
            tracked.write_bytes(b"after!")
            os.utime(
                tracked,
                ns=(original.st_atime_ns, original.st_mtime_ns),
            )
            second = coordinator.advance_one_barrier()

        self.assertNotEqual(
            first.payload["storage"]["entry_manifest_sha256"],
            second.payload["storage"]["entry_manifest_sha256"],
        )

    def test_terminal_task_keeps_earlier_pinned_tick_across_later_frontiers(
        self,
    ) -> None:
        terminal_task = self.tasks[0]
        terminal = _FakeRunner(terminal_task, terminal_tick=100)
        terminal.bind_campaign_root(self.campaign_root)
        self.runners[terminal_task.task_id] = terminal
        coordinator = self._coordinator()

        receipts = coordinator.advance_through(15_000)

        self.assertEqual(terminal.observed_tick, 100)
        self.assertTrue(terminal.extinct)
        records = {
            record["task_id"]: record for record in receipts[-1].payload["tasks"]
        }
        terminal_record = records[terminal_task.task_id]
        self.assertEqual(terminal_record["task_state"], "terminal_extinct")
        self.assertEqual(terminal_record["observed_tick"], 100)
        self.assertEqual(
            terminal_record["aggregate_resume_pins"]["identity"]["tick"],
            99,
        )
        self.assertEqual(terminal_record["expected_task_tick"], 15_000)

    def test_resume_requires_receipt_and_rejects_stale_storage(self) -> None:
        missing_path = self.receipt_directory / "missing.json"
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "missing",
        ):
            load_campaign_frontier_receipt(missing_path)

        coordinator = self._coordinator()
        receipt = coordinator.advance_one_barrier()
        self.assertTrue(receipt.external_sha256_verified)
        symlink = self.receipt_directory / "receipt-link.json"
        symlink.symlink_to(receipt.path)
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "regular file",
        ):
            load_campaign_frontier_receipt(symlink)
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "external SHA256 authority",
        ):
            load_campaign_frontier_receipt(
                receipt.path,
                expected_receipt_sha256="0" * 64,
            )
        hostile_envelope = json.loads(receipt.path.read_bytes())
        hostile_record = hostile_envelope["payload"]["tasks"][0]
        hostile_record["learner_seed"] += 1
        unsigned_record = dict(hostile_record)
        unsigned_record.pop("frontier_record_sha256")
        hostile_record["frontier_record_sha256"] = hashlib.sha256(
            canonical_json_bytes(unsigned_record)
        ).hexdigest()
        hostile_envelope["payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(hostile_envelope["payload"])
        ).hexdigest()
        hostile_path = self.receipt_directory / "hostile-semantic-receipt.json"
        hostile_bytes = canonical_json_bytes(hostile_envelope)
        hostile_path.write_bytes(hostile_bytes)
        hostile = load_campaign_frontier_receipt(
            hostile_path,
            expected_receipt_sha256=hashlib.sha256(hostile_bytes).hexdigest(),
        )
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "task binding learner_seed",
        ):
            self._coordinator(
                authorized_frontier_tick=5_000,
                previous_frontier_receipt=hostile,
            )
        resumed = self._coordinator(
            authorized_frontier_tick=5_000,
            previous_frontier_receipt=receipt,
        )
        resumed.validate_retained_frontier_storage()

        self.scanner.generation = 1
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "storage receipt is stale",
        ):
            resumed.validate_retained_frontier_storage()
        before_ticks = {
            task_id: runner.observed_tick for task_id, runner in self.runners.items()
        }
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "storage receipt is stale",
        ):
            resumed.advance_one_barrier()
        self.assertEqual(
            {task_id: runner.observed_tick for task_id, runner in self.runners.items()},
            before_ticks,
        )

    def test_authority_loaders_reject_path_swap_to_symlink_during_read(
        self,
    ) -> None:
        def assert_swap_rejected(name, loader) -> None:
            target = self.receipt_directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"{}\n")
            target.chmod(0o400)
            retained = target.with_suffix(".retained")
            real_read = os.read
            swapped = False

            def swap_after_read(descriptor, size):
                nonlocal swapped
                chunk = real_read(descriptor, size)
                if not swapped:
                    swapped = True
                    os.replace(target, retained)
                    target.symlink_to(retained)
                return chunk

            with (
                patch(
                    "evolution_sim.mind.open_ecology_campaign_coordinator.os.read",
                    side_effect=swap_after_read,
                ),
                self.assertRaisesRegex(
                    OpenEcologyCampaignCoordinatorError,
                    "path entry changed",
                ),
            ):
                loader(target)

        assert_swap_rejected(
            "swap-receipt.json",
            lambda path: load_campaign_frontier_receipt(path),
        )
        assert_swap_rejected(
            "swap-intent.json",
            _load_barrier_intent,
        )
        assert_swap_rejected(
            "swap-failure.json",
            lambda path: _load_barrier_failure_observation(
                path,
                expected_intent_path=self.receipt_directory / "intent.json",
                expected_intent_sha256="a" * 64,
                campaign_id="campaign-test",
                source_git_sha=_SOURCE_COMMIT,
                matrix_sha256="b" * 64,
            ),
        )

    def test_output_lock_contention_and_parallel_without_runner_contract_fail_closed(
        self,
    ) -> None:
        coordinator = self._coordinator()
        coordinator.advance_one_barrier()
        lock_path = self.receipt_directory / ".open-ecology-coordinator.lock"
        descriptor = os.open(lock_path, os.O_RDWR)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "locked by another process",
            ):
                coordinator.advance_one_barrier()
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        self.assertEqual(
            len(list(self.receipt_directory.glob("step-*.json"))),
            1,
        )

        fresh_base = self.base / "parallel"
        fresh_base.mkdir()
        fresh_receipts = self.base / "parallel-receipts"
        parallel_runners = {task.task_id: _FakeRunner(task) for task in self.tasks}
        for runner in parallel_runners.values():
            runner.bind_campaign_root(fresh_base)
        parallel = self._coordinator(
            campaign_root=fresh_base,
            receipt_directory=fresh_receipts,
            runners=parallel_runners,
            worker_count=2,
            scanner=_FakeStorageScanner(fresh_base),
        )
        expected = (
            "parallel campaign advance is not authorized"
            if not parallel.parallel_execution_authorized
            else "persistent process/host worker launcher"
        )
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            expected,
        ):
            parallel.advance_one_barrier()

    def test_spawned_process_launcher_crosses_qualification_catchup_and_matrix(
        self,
    ) -> None:
        slots = build_process_worker_slots(
            self.tasks,
            worker_count=2,
            host_identity=local_process_worker_host_identity(),
        )
        with PersistentProcessWorkerLauncher(
            tasks=self.tasks,
            campaign_root=self.campaign_root,
            campaign_id="campaign-test",
            repository_root=Path(__file__).resolve().parents[2],
            source_git_sha=_SOURCE_COMMIT,
            source_manifest_sha256=_SOURCE_MANIFEST,
            slots=slots,
            assignment_sha256=process_worker_assignment_sha256(
                self.tasks,
                slots,
            ),
            response_timeout_seconds=15.0,
            startup_timeout_seconds=15.0,
            shutdown_timeout_seconds=5.0,
            _test_behaviors={},
        ) as launcher:
            worker_pids = dict(launcher.worker_pids)
            coordinator = self._coordinator(
                runners={},
                worker_count=2,
                worker_launcher=launcher,
            )
            qualification = coordinator.advance_through(10_000)
            self.assertEqual(
                [
                    (item.payload["barrier_scope"], item.frontier_tick)
                    for item in qualification
                ],
                [
                    ("qualification_triplet", 5_000),
                    ("qualification_triplet", 10_000),
                ],
            )
            self.assertEqual(dict(launcher.worker_pids), worker_pids)
            self.assertEqual(len(qualification[-1].resume_pins_by_task()), 3)

        with PersistentProcessWorkerLauncher(
            tasks=self.tasks,
            campaign_root=self.campaign_root,
            campaign_id="campaign-test",
            repository_root=Path(__file__).resolve().parents[2],
            source_git_sha=_SOURCE_COMMIT,
            source_manifest_sha256=_SOURCE_MANIFEST,
            slots=slots,
            assignment_sha256=process_worker_assignment_sha256(
                self.tasks,
                slots,
            ),
            response_timeout_seconds=15.0,
            startup_timeout_seconds=15.0,
            shutdown_timeout_seconds=5.0,
            _test_behaviors={},
        ) as resumed_launcher:
            resumed = self._coordinator(
                runners={},
                worker_count=2,
                authorized_frontier_tick=10_000,
                previous_frontier_receipt=qualification[-1],
                worker_launcher=resumed_launcher,
            )
            carried_validations: list[tuple[str, ...]] = []

            def validate_protocol_carried_records(records) -> None:
                carried_validations.append(
                    tuple(
                        record["task_id"]
                        for record in records
                        if record["frontier_observation_mode"]
                        == "retained_unmaterialized_worker"
                    )
                )

            # The protocol backend writes bounded synthetic preservation files,
            # not real aggregate generations.  Keep this test on orchestration
            # semantics while asserting that the production carried-CURRENT
            # validation hook runs under every final barrier.
            resumed._validate_carried_frontiers_on_disk = (  # type: ignore[method-assign]
                validate_protocol_carried_records
            )
            receipts = resumed.advance_through(15_000)
            self.assertEqual(
                [
                    (item.payload["barrier_scope"], item.frontier_tick)
                    for item in receipts
                ],
                [
                    ("matrix_catchup", 5_000),
                    ("matrix_catchup", 10_000),
                    ("full_matrix", 15_000),
                ],
            )
            carried = receipts[0].payload["tasks"][:3]
            self.assertEqual(
                [record["frontier_observation_mode"] for record in carried],
                ["retained_unmaterialized_worker"] * 3,
            )
            self.assertEqual(
                [record["observed_tick"] for record in carried],
                [10_000] * 3,
            )
            self.assertTrue(
                all(record["aggregate_resume_pins"] is not None for record in carried)
            )
            self.assertEqual(len(receipts[-1].resume_pins_by_task()), 48)
            self.assertTrue(receipts[-1].payload["all_tasks_at_common_frontier"])
            worker_batch = receipts[-1].payload["process_worker_frontier_batch"]
            self.assertEqual(worker_batch["target_tick"], 15_000)
            self.assertEqual(
                worker_batch["assignment_sha256"],
                resumed.assignment_sha256,
            )
            self.assertEqual(len(worker_batch["result_sha256_by_task"]), 48)
            self.assertEqual(
                {status["worker_pid"] for status in worker_batch["worker_statuses"]},
                set(resumed_launcher.worker_pids.values()),
            )
            self.assertEqual(
                carried_validations,
                [
                    tuple(task.task_id for task in self.tasks[:3]),
                    tuple(task.task_id for task in self.tasks[:3]),
                    (),
                ],
            )

    def test_default_barrier_consumes_real_runner_frontier_contract(self) -> None:
        for task in self.tasks[:3]:
            lock_path = self.campaign_root.parent / (
                f".{self.campaign_root.name}.{task.task_id}.open-ecology-task.lock"
            )
            lock_path.write_bytes(
                canonical_json_bytes(
                    {
                        "campaign_id": "campaign-test",
                        "schema_version": (
                            "mind_v3_open_ecology_task_mutation_lock_v1"
                        ),
                        "source_git_sha": _SOURCE_COMMIT,
                        "task_id": task.task_id,
                    }
                )
            )
            lock_path.chmod(0o600)
        coordinator = self._coordinator(use_default_barrier=True)

        receipt = coordinator.advance_one_barrier()

        self.assertEqual(receipt.frontier_tick, 5_000)
        self.assertEqual(receipt.payload["barrier_scope"], "qualification_triplet")
        self.assertEqual(len(receipt.resume_pins_by_task()), 3)

    def _coordinator(
        self,
        *,
        runners=None,
        campaign_root=None,
        receipt_directory=None,
        source_git_sha: str = _SOURCE_COMMIT,
        worker_count: int = 1,
        authorized_frontier_tick: int = 0,
        previous_frontier_receipt=None,
        scanner=None,
        use_default_barrier: bool = False,
        worker_launcher=None,
        pending_runner_factory=None,
    ) -> OpenEcologyCampaignCoordinator:
        selected_root = campaign_root or self.campaign_root
        selected_receipts = receipt_directory or self.receipt_directory
        selected_scanner = scanner or self.scanner
        arguments = {
            "tasks": self.tasks,
            "runners": self.runners if runners is None else runners,
            "campaign_root": selected_root,
            "receipt_directory": selected_receipts,
            "campaign_id": "campaign-test",
            "source_git_sha": source_git_sha,
            "source_manifest_sha256": _SOURCE_MANIFEST,
            "worker_count": worker_count,
            "authorized_frontier_tick": authorized_frontier_tick,
            "previous_frontier_receipt": previous_frontier_receipt,
            "worker_launcher": worker_launcher,
            "pending_runner_factory": pending_runner_factory,
            "source_probe": lambda: {
                "repository_root": "/exact/checkout",
                "source_git_sha": _SOURCE_COMMIT,
                "source_manifest_sha256": _SOURCE_MANIFEST,
                "git_clean": True,
            },
            "health_probe": lambda phase, frontier: {
                "phase": phase,
                "frontier_tick": frontier,
                "healthy": True,
            },
            "storage_scanner": selected_scanner,
        }
        if not use_default_barrier:
            arguments["barrier_context_factory"] = lambda _frontiers: nullcontext()
        return OpenEcologyCampaignCoordinator(
            **arguments,
        )

    def _bindings(self):
        return tuple(
            PersistentArtifactBinding(
                learner_index=index,
                learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][index],
                artifact_path=str(self.base / f"artifact-{index}.json"),
                artifact_sha256=f"{index + 1:x}" * 64,
                artifact_file_sha256=f"{index + 5:x}" * 64,
                source_commit=_SOURCE_COMMIT,
                terminal_authority_sha256=f"{index + 9:x}" * 64,
            )
            for index in range(4)
        )


if __name__ == "__main__":
    unittest.main()
