"""Fail-closed orchestration for the persistent open-ecology campaign.

The coordinator owns campaign-wide barriers.  Island runners own simulation
state and task-local checkpoint publication.  A scope frontier is released
only after every started task has a restartable aggregate ``CURRENT`` and one
complete active-root storage scan has been bound into an immutable external
receipt.  Qualification receipts name the other 45 tasks as unstarted; they do
not fabricate full-matrix authority.

This module deliberately exposes no deletion, pruning, migration, or tuning
operation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Protocol

from evolution_sim.io.open_ecology_aggregate_commit import (
    OpenEcologyAggregateCommitError,
    OpenEcologyAggregateIdentityPins,
    OpenEcologyAggregateResumePins,
    inspect_current_open_ecology_aggregate_generation,
)
from evolution_sim.io.open_ecology_campaign_storage import (
    CampaignStorageError,
    CampaignStorageLimits,
    CampaignStorageLock,
    StorageScan,
    canonical_json_bytes,
    check_campaign_storage,
    default_storage_lock_path,
    ensure_real_directory_tree,
    _normalize_real_tree_path as _normalize_storage_real_tree_path,
    _open_real_directory_tree as _open_storage_real_directory_tree,
)
from evolution_sim.mind.open_ecology_campaign_contract import (
    OpenEcologyCampaignCoordinatorError,
)
from evolution_sim.mind.open_ecology_persistent_island import (
    OPEN_ECOLOGY_PHASE_D_ARM_ORDER,
    OPEN_ECOLOGY_PHASE_D_ISLAND_COUNT,
    OPEN_ECOLOGY_PHASE_D_LEARNER_COUNT,
    OPEN_ECOLOGY_PHASE_D_TARGET_TICKS,
    OPEN_ECOLOGY_PHASE_D_TASK_COUNT,
    OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS,
    PersistentAdvanceResult,
    PersistentIslandRunner,
    PersistentIslandTask,
    persistent_interval_attempt_authority_payload_sha256,
    persistent_interval_attempt_authority_sha256,
    prioritized_persistent_island_triplet,
)
from evolution_sim.mind.open_ecology_process_workers import (
    OPEN_ECOLOGY_PROCESS_WORKER_LAUNCHER_CONTRACT,
    ProcessWorkerExpectedState,
    ProcessWorkerFrontierBatch,
    ProcessWorkerReconciliationBatch,
)


OPEN_ECOLOGY_CAMPAIGN_COORDINATOR_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_coordinator_v1"
)
OPEN_ECOLOGY_CAMPAIGN_FRONTIER_RECEIPT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_frontier_receipt_v1"
)
OPEN_ECOLOGY_CAMPAIGN_FRONTIER_ENVELOPE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_frontier_receipt_envelope_v1"
)
OPEN_ECOLOGY_CAMPAIGN_STATUS_SCHEMA_VERSION = "mind_v3_open_ecology_campaign_status_v1"
OPEN_ECOLOGY_CAMPAIGN_OUTPUT_LOCK_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_output_lock_v1"
)
OPEN_ECOLOGY_CAMPAIGN_BARRIER_INTENT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_barrier_intent_v1"
)
OPEN_ECOLOGY_CAMPAIGN_BARRIER_INTENT_ENVELOPE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_barrier_intent_envelope_v1"
)
OPEN_ECOLOGY_CAMPAIGN_BARRIER_FAILURE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_barrier_failure_observation_v1"
)
OPEN_ECOLOGY_CAMPAIGN_BARRIER_RECOVERY_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_barrier_recovery_v1"
)
OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK = 10_000
OPEN_ECOLOGY_CAMPAIGN_MAX_WORKERS = OPEN_ECOLOGY_PHASE_D_TASK_COUNT
OPEN_ECOLOGY_CAMPAIGN_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
OPEN_ECOLOGY_CAMPAIGN_MAX_BARRIER_INTENT_BYTES = 4 * 1024 * 1024

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_OUTPUT_LOCK_NAME = ".open-ecology-coordinator.lock"
_OUTPUT_LOCK_MAX_BYTES = 16 * 1024


class CampaignRunner(Protocol):
    task: PersistentIslandTask

    @property
    def observed_tick(self) -> int: ...

    @property
    def extinct(self) -> bool: ...

    @property
    def latest_aggregate_resume_pins(
        self,
    ) -> OpenEcologyAggregateResumePins | None: ...

    @property
    def quiescent_checkpoint(self) -> Mapping[str, object]: ...

    def campaign_barrier_frontier(self) -> Mapping[str, object]: ...

    def advance_to(self, target_tick: int) -> PersistentAdvanceResult: ...


HealthProbe = Callable[[str, int], Mapping[str, object]]
SourceProbe = Callable[[], Mapping[str, object]]
PendingRunnerFactory = Callable[[PersistentIslandTask], CampaignRunner]
StorageScanner = Callable[..., StorageScan]
BarrierContextFactory = Callable[
    [Sequence[Mapping[str, object]]],
    AbstractContextManager[object],
]


class CampaignWorkerLauncher(Protocol):
    """Persistent process/host launcher for static worker queues.

    The launcher owns worker IPC and calls the supplied runner proxies.  A
    production launcher keeps workers alive across barriers or proxies to
    persistent remote workers; an in-process thread pool is not this contract.
    """

    @property
    def execution_contract(self) -> str: ...

    @property
    def assignment_sha256(self) -> str: ...

    @property
    def slots(self) -> Sequence[object]: ...

    def advance_frontier(
        self,
        *,
        selected_task_ids: Sequence[str],
        target_tick: int,
        expected_states: Mapping[str, ProcessWorkerExpectedState],
    ) -> ProcessWorkerFrontierBatch: ...

    def reconcile_frontier(
        self,
        *,
        selected_task_ids: Sequence[str],
        target_tick: int,
        expected_states: Mapping[str, ProcessWorkerExpectedState],
        intent_sha256: str,
        attempt_authority_sha256_by_task: Mapping[str, str],
    ) -> ProcessWorkerReconciliationBatch: ...


@dataclass(frozen=True, slots=True)
class StaticWorkerAssignment:
    """One immutable matrix-to-worker queue position."""

    task_id: str
    worker_index: int
    worker_queue_index: int
    global_queue_index: int
    qualification_priority: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "worker_index": self.worker_index,
            "worker_queue_index": self.worker_queue_index,
            "global_queue_index": self.global_queue_index,
            "qualification_priority": self.qualification_priority,
        }


@dataclass(frozen=True, slots=True)
class CampaignFrontierReceipt:
    """Validated immutable authority for one complete global frontier."""

    path: Path
    payload: Mapping[str, object]
    payload_sha256: str
    receipt_sha256: str
    external_sha256_verified: bool

    @property
    def frontier_tick(self) -> int:
        return _nonnegative_int(
            self.payload.get("frontier_tick"),
            field="frontier receipt frontier_tick",
        )

    @property
    def source_git_sha(self) -> str:
        return _git_sha(
            self.payload.get("source_git_sha"),
            field="frontier receipt source_git_sha",
        )

    @property
    def source_manifest_sha256(self) -> str:
        return _sha256(
            self.payload.get("source_manifest_sha256"),
            field="frontier receipt source_manifest_sha256",
        )

    def resume_pins_by_task(self) -> dict[str, OpenEcologyAggregateResumePins]:
        records = _sequence(self.payload.get("tasks"), field="frontier receipt tasks")
        pins: dict[str, OpenEcologyAggregateResumePins] = {}
        for index, raw_record in enumerate(records):
            record = _mapping(raw_record, field=f"frontier receipt tasks[{index}]")
            task_id = _identifier(
                record.get("task_id"),
                field=f"frontier receipt tasks[{index}].task_id",
            )
            if record.get("task_state") == "pending_unstarted":
                if record.get("aggregate_resume_pins") is not None:
                    raise OpenEcologyCampaignCoordinatorError(
                        "pending task unexpectedly carries aggregate pins"
                    )
                continue
            pins_payload = _mapping(
                record.get("aggregate_resume_pins"),
                field=f"frontier receipt tasks[{index}].aggregate_resume_pins",
            )
            if task_id in pins:
                raise OpenEcologyCampaignCoordinatorError(
                    "frontier receipt repeats a task"
                )
            pins[task_id] = _resume_pins_from_dict(pins_payload)
        return pins


def build_static_worker_assignments(
    tasks: Sequence[PersistentIslandTask],
    *,
    worker_count: int,
) -> tuple[StaticWorkerAssignment, ...]:
    """Freeze a deterministic round-robin queue without changing task order."""

    _validate_exact_task_matrix(tasks)
    parsed_workers = _positive_int(worker_count, field="worker_count")
    if parsed_workers > OPEN_ECOLOGY_CAMPAIGN_MAX_WORKERS:
        raise OpenEcologyCampaignCoordinatorError(
            "worker_count cannot exceed the fixed 48-task matrix"
        )
    priority_ids = {
        task.task_id for task in prioritized_persistent_island_triplet(tasks)
    }
    queue_lengths = [0] * parsed_workers
    assignments: list[StaticWorkerAssignment] = []
    for global_index, task in enumerate(tasks):
        worker_index = global_index % parsed_workers
        assignments.append(
            StaticWorkerAssignment(
                task_id=task.task_id,
                worker_index=worker_index,
                worker_queue_index=queue_lengths[worker_index],
                global_queue_index=global_index,
                qualification_priority=task.task_id in priority_ids,
            )
        )
        queue_lengths[worker_index] += 1
    return tuple(assignments)


class OpenEcologyCampaignCoordinator:
    """Advance all 48 persistent tasks through storage-gated 5k barriers."""

    def __init__(
        self,
        *,
        tasks: Sequence[PersistentIslandTask],
        runners: Mapping[str, CampaignRunner],
        campaign_root: str | Path,
        receipt_directory: str | Path,
        campaign_id: str,
        source_git_sha: str,
        source_manifest_sha256: str,
        worker_count: int,
        authorized_frontier_tick: int = 0,
        previous_frontier_receipt: CampaignFrontierReceipt | None = None,
        health_probe: HealthProbe | None = None,
        source_probe: SourceProbe | None = None,
        worker_launcher: CampaignWorkerLauncher | None = None,
        pending_runner_factory: PendingRunnerFactory | None = None,
        storage_scanner: StorageScanner = check_campaign_storage,
        storage_limits: CampaignStorageLimits | None = None,
        barrier_context_factory: BarrierContextFactory | None = None,
        launch_authority: Mapping[str, object] | None = None,
    ) -> None:
        self.tasks = tuple(tasks)
        _validate_exact_task_matrix(self.tasks)
        self._task_by_id = {task.task_id: task for task in self.tasks}
        surplus = sorted(set(runners) - set(self._task_by_id))
        priority_ids = {
            task.task_id for task in prioritized_persistent_island_triplet(self.tasks)
        }
        missing_priority = sorted(priority_ids - set(runners))
        if surplus or (
            worker_launcher is None
            and pending_runner_factory is None
            and missing_priority
        ):
            missing = sorted(set(self._task_by_id) - set(runners))
            surplus = sorted(set(runners) - set(self._task_by_id))
            raise OpenEcologyCampaignCoordinatorError(
                "campaign runner set omits qualification tasks or has surplus; "
                f"missing={missing}, surplus={surplus}"
            )
        self._runners = dict(runners)
        for task_id, runner in self._runners.items():
            if runner.task != self._task_by_id[task_id]:
                raise OpenEcologyCampaignCoordinatorError(
                    f"runner task binding drifted for {task_id}"
                )

        self.campaign_id = _identifier(campaign_id, field="campaign_id")
        self.source_git_sha = _git_sha(source_git_sha, field="source_git_sha")
        self.source_manifest_sha256 = _sha256(
            source_manifest_sha256,
            field="source_manifest_sha256",
        )
        self._launch_authority = (
            None
            if launch_authority is None
            else dict(
                _mapping(
                    launch_authority,
                    field="launch_authority",
                )
            )
        )
        if self._launch_authority is not None:
            if set(self._launch_authority) != {
                "launch_spec_path",
                "launch_spec_sha256",
            }:
                raise OpenEcologyCampaignCoordinatorError(
                    "launch authority schema is not exact"
                )
            launch_spec_path = self._launch_authority.get("launch_spec_path")
            if (
                not isinstance(launch_spec_path, str)
                or not Path(launch_spec_path).is_absolute()
            ):
                raise OpenEcologyCampaignCoordinatorError(
                    "launch authority path must be absolute"
                )
            _sha256(
                self._launch_authority.get("launch_spec_sha256"),
                field="launch authority SHA256",
            )
        task_sources = {task.artifact.source_commit for task in self.tasks}
        if task_sources != {self.source_git_sha}:
            raise OpenEcologyCampaignCoordinatorError(
                "campaign source does not match every task artifact"
            )
        self.campaign_root = _existing_absolute_directory(
            campaign_root,
            field="campaign_root",
        )
        self.receipt_directory = _prepare_receipt_directory(
            receipt_directory,
            campaign_root=self.campaign_root,
        )
        self.worker_count = _positive_int(worker_count, field="worker_count")
        if self.worker_count > OPEN_ECOLOGY_CAMPAIGN_MAX_WORKERS:
            raise OpenEcologyCampaignCoordinatorError(
                "worker_count cannot exceed the fixed 48-task matrix"
            )
        self.assignments = build_static_worker_assignments(
            self.tasks,
            worker_count=self.worker_count,
        )
        self.matrix_sha256 = _sha256_bytes(
            _canonical_json_bytes([task.to_dict() for task in self.tasks])
        )
        self.assignment_sha256 = _sha256_bytes(
            _canonical_json_bytes(
                [assignment.to_dict() for assignment in self.assignments]
            )
        )
        self._authorized_frontier_tick = _frontier_tick(
            authorized_frontier_tick,
            allow_zero=True,
        )
        self._previous_frontier_receipt = previous_frontier_receipt
        self._qualification_frontier_tick = 0
        self._matrix_frontier_tick = 0
        self._coordination_sequence_index = 0
        if previous_frontier_receipt is None:
            if self._authorized_frontier_tick != 0:
                raise OpenEcologyCampaignCoordinatorError(
                    "a nonzero authorized frontier requires its external receipt"
                )
        else:
            if not previous_frontier_receipt.external_sha256_verified:
                raise OpenEcologyCampaignCoordinatorError(
                    "retained frontier receipt lacks external SHA256 authority"
                )
            self._validate_receipt_identity(previous_frontier_receipt)
            if (
                previous_frontier_receipt.frontier_tick
                != self._authorized_frontier_tick
            ):
                raise OpenEcologyCampaignCoordinatorError(
                    "authorized frontier and retained receipt disagree"
                )
            self._qualification_frontier_tick = _frontier_tick(
                previous_frontier_receipt.payload.get("qualification_frontier_tick"),
                allow_zero=True,
            )
            self._matrix_frontier_tick = _frontier_tick(
                previous_frontier_receipt.payload.get("matrix_frontier_tick"),
                allow_zero=True,
            )
            self._coordination_sequence_index = _positive_int(
                previous_frontier_receipt.payload.get("coordination_sequence_index"),
                field="retained coordination_sequence_index",
            )
            if (
                self._qualification_frontier_tick
                > OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
                or self._matrix_frontier_tick > self._qualification_frontier_tick
            ):
                raise OpenEcologyCampaignCoordinatorError(
                    "retained qualification/matrix frontier state is invalid"
                )
        self._health_probe = health_probe
        if source_probe is None:
            raise OpenEcologyCampaignCoordinatorError(
                "an explicit pinned source probe is required"
            )
        self._source_probe = source_probe
        self._worker_launcher = worker_launcher
        self._pending_runner_factory = pending_runner_factory
        if worker_launcher is not None:
            if (
                worker_launcher.execution_contract
                != OPEN_ECOLOGY_PROCESS_WORKER_LAUNCHER_CONTRACT
            ):
                raise OpenEcologyCampaignCoordinatorError(
                    "worker launcher is not the persistent process/host queue contract"
                )
            if worker_launcher.assignment_sha256 != self.assignment_sha256:
                raise OpenEcologyCampaignCoordinatorError(
                    "worker launcher assignment digest drifted"
                )
            if len(worker_launcher.slots) != self.worker_count:
                raise OpenEcologyCampaignCoordinatorError(
                    "worker launcher slot count does not match worker_count"
                )
        self._storage_scanner = storage_scanner
        self._storage_limits = storage_limits or CampaignStorageLimits()
        self._storage_limits.validate()
        self._barrier_context_factory = (
            barrier_context_factory or self._default_barrier_context
        )
        self._last_worker_batch_summary: Mapping[str, object] | None = None
        self._last_storage_scan: StorageScan | None = None
        # A restart must prove that the retained whole-tree receipt still names
        # the live active root before any worker mutates it.  An uninterrupted
        # coordinator already performs a complete post-frontier scan under the
        # global exclusive barrier, so repeating that same preflight before
        # every later barrier only rehashes an unchanged predecessor tree.
        self._retained_storage_preflight_required = (
            previous_frontier_receipt is not None
        )
        self._validate_runner_positions()

    @property
    def authorized_frontier_tick(self) -> int:
        return self._authorized_frontier_tick

    @property
    def qualification_frontier_tick(self) -> int:
        return self._qualification_frontier_tick

    @property
    def matrix_frontier_tick(self) -> int:
        return self._matrix_frontier_tick

    @property
    def _campaign_stage(self) -> str:
        if self._qualification_frontier_tick < OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK:
            return "qualification"
        if self._matrix_frontier_tick < OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK:
            return "matrix_catchup"
        if self._matrix_frontier_tick < OPEN_ECOLOGY_PHASE_D_TARGET_TICKS:
            return "full_matrix"
        return "complete"

    @property
    def previous_frontier_receipt(self) -> CampaignFrontierReceipt | None:
        return self._previous_frontier_receipt

    @property
    def parallel_execution_authorized(self) -> bool:
        return bool(
            getattr(
                __import__(
                    "evolution_sim.mind.open_ecology_persistent_island",
                    fromlist=["OPEN_ECOLOGY_PARALLEL_CAMPAIGN_ADVANCE_AUTHORIZED"],
                ),
                "OPEN_ECOLOGY_PARALLEL_CAMPAIGN_ADVANCE_AUTHORIZED",
                False,
            )
        )

    def status(self) -> dict[str, object]:
        """Return a read-only status snapshot; never advances or writes."""

        task_status: list[dict[str, object]] = []
        for task in self.tasks:
            runner = self._runners.get(task.task_id)
            pins = None if runner is None else runner.latest_aggregate_resume_pins
            task_status.append(
                {
                    "task_id": task.task_id,
                    "arm": task.arm,
                    "learner_index": task.learner_index,
                    "island_index": task.island_index,
                    "observed_tick": 0 if runner is None else runner.observed_tick,
                    "extinct": False if runner is None else runner.extinct,
                    "task_state": (
                        "pending_unstarted"
                        if runner is None or pins is None
                        else ("terminal_extinct" if runner.extinct else "active")
                    ),
                    "aggregate_current_present": pins is not None,
                    "aggregate_commit_sha256": (
                        None if pins is None else pins.commit_sha256
                    ),
                }
            )
        payload: dict[str, object] = {
            "schema_version": OPEN_ECOLOGY_CAMPAIGN_STATUS_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "campaign_root": str(self.campaign_root),
            "source_git_sha": self.source_git_sha,
            "source_manifest_sha256": self.source_manifest_sha256,
            "launch_authority": (
                None if self._launch_authority is None else dict(self._launch_authority)
            ),
            "matrix_sha256": self.matrix_sha256,
            "assignment_sha256": self.assignment_sha256,
            "worker_count": self.worker_count,
            "parallel_execution_authorized": self.parallel_execution_authorized,
            "worker_execution_contract": (
                "single_process_sequential_v1"
                if self._worker_launcher is None
                else self._worker_launcher.execution_contract
            ),
            "authorized_frontier_tick": self._authorized_frontier_tick,
            "qualification_frontier_tick": self._qualification_frontier_tick,
            "matrix_frontier_tick": self._matrix_frontier_tick,
            "campaign_stage": self._campaign_stage,
            "next_barrier": self._next_step_payload(),
            "previous_frontier_receipt_sha256": (
                None
                if self._previous_frontier_receipt is None
                else self._previous_frontier_receipt.receipt_sha256
            ),
            "qualification_triplet_task_ids": [
                task.task_id
                for task in prioritized_persistent_island_triplet(self.tasks)
            ],
            "tasks": task_status,
            "read_only": True,
            "pruning_or_deletion_available": False,
        }
        payload["status_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
        return payload

    def advance_one_barrier(self) -> CampaignFrontierReceipt:
        """Advance exactly one 5k frontier and authorize no later tick."""

        step = self._next_step()
        if step is None:
            raise OpenEcologyCampaignCoordinatorError(
                "campaign is already at its 50,000-tick frontier"
            )
        scope, target_tick, selected_task_ids = step
        with _CoordinatorOutputLock(
            self.receipt_directory / _OUTPUT_LOCK_NAME,
            campaign_id=self.campaign_id,
            source_git_sha=self.source_git_sha,
            matrix_sha256=self.matrix_sha256,
        ):
            self._validate_runner_positions()
            source_before = self._verified_source_snapshot()
            sequence_index = self._coordination_sequence_index + 1
            intent_path = self.receipt_directory / (
                f"barrier-intent-{sequence_index:04d}-{scope}-{target_tick:08d}.json"
            )
            recovery_mode = intent_path.exists() or intent_path.is_symlink()
            retained_storage_preflight_succeeded = False
            if self._retained_storage_preflight_required and not recovery_mode:
                self._last_storage_scan = self.validate_retained_frontier_storage()
                retained_storage_preflight_succeeded = True
            health_before = self._health_snapshot("before_advance", target_tick)
            barrier_intent = self._begin_barrier_intent(
                scope=scope,
                target_tick=target_tick,
                selected_task_ids=selected_task_ids,
                allow_existing_unresolved=(
                    recovery_mode or retained_storage_preflight_succeeded
                ),
            )
            barrier_recovery: Mapping[str, object] | None = None
            try:
                if recovery_mode:
                    failure_observation = self._ensure_barrier_failure_observation(
                        barrier_intent=barrier_intent,
                        selected_task_ids=selected_task_ids,
                    )
                    barrier_recovery = self._reconcile_runners(
                        target_tick=target_tick,
                        selected_task_ids=selected_task_ids,
                        barrier_intent=barrier_intent,
                        failure_observation=failure_observation,
                    )
                else:
                    self._advance_runners(
                        target_tick,
                        selected_task_ids=selected_task_ids,
                    )
            except BaseException as error:
                self._record_barrier_failure(
                    barrier_intent=barrier_intent,
                    selected_task_ids=selected_task_ids,
                    error=error,
                )
                raise
            frontier_records, runner_frontiers = self._collect_frontier_records(
                scope=scope,
                target_tick=target_tick,
            )
            with self._barrier_context_factory(runner_frontiers):
                # The exclusive storage barrier prevents a new writer from
                # starting after the immediately preceding frontier reread.
                # The runner barrier itself validates and task-locks those
                # exact digest-bound receipts; rereading here would recursively
                # acquire the same non-reentrant task locks.
                self._validate_carried_frontiers_on_disk(frontier_records)
                source_after = self._verified_source_snapshot()
                if source_after != source_before:
                    raise OpenEcologyCampaignCoordinatorError(
                        "source authority changed during the campaign interval"
                    )
                (
                    storage_scan,
                    storage_validation_mode,
                    predecessor_storage_manifest_sha256,
                ) = self._scan_frontier_storage(
                    target_tick=target_tick,
                    force_complete=recovery_mode,
                )
                health_after = self._health_snapshot(
                    "frontier_quiescent",
                    target_tick,
                )
                receipt = self._write_or_validate_frontier_receipt(
                    target_tick=target_tick,
                    scope=scope,
                    frontier_records=frontier_records,
                    source_snapshot=source_after,
                    health_before=health_before,
                    health_after=health_after,
                    storage_scan=storage_scan,
                    storage_validation_mode=storage_validation_mode,
                    predecessor_storage_manifest_sha256=(
                        predecessor_storage_manifest_sha256
                    ),
                    barrier_intent=barrier_intent,
                    barrier_recovery=barrier_recovery,
                )
            if scope == "qualification_triplet":
                self._qualification_frontier_tick = target_tick
            else:
                self._matrix_frontier_tick = target_tick
            self._authorized_frontier_tick = target_tick
            self._coordination_sequence_index += 1
            self._previous_frontier_receipt = receipt
            self._retained_storage_preflight_required = False
            self._last_storage_scan = storage_scan
            return receipt

    def advance_through(self, target_tick: int) -> tuple[CampaignFrontierReceipt, ...]:
        """Advance through one exact 5k-multiple frontier."""

        requested = _frontier_tick(target_tick, allow_zero=False)
        if (
            self._qualification_frontier_tick
            >= OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
            and requested < self._matrix_frontier_tick
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "target frontier cannot precede retained authority"
            )
        receipts: list[CampaignFrontierReceipt] = []
        while not self._requested_frontier_reached(requested):
            receipts.append(self.advance_one_barrier())
        return tuple(receipts)

    def validate_retained_frontier_storage(
        self,
        receipt: CampaignFrontierReceipt | None = None,
    ) -> StorageScan:
        """Freshly prove that retained frontier authority still names this tree."""

        retained = receipt or self._previous_frontier_receipt
        if retained is None:
            raise OpenEcologyCampaignCoordinatorError(
                "no retained frontier receipt is available"
            )
        if not retained.external_sha256_verified:
            raise OpenEcologyCampaignCoordinatorError(
                "retained frontier validation requires external SHA256 authority"
            )
        self._validate_receipt_identity(retained)
        started_task_ids = set(retained.resume_pins_by_task())
        missing_started_views = started_task_ids - set(self._runners)
        if missing_started_views:
            if self._worker_launcher is None:
                raise OpenEcologyCampaignCoordinatorError(
                    "retained frontier started-task views are missing"
                )
            barrier_context: AbstractContextManager[object] = CampaignStorageLock(
                default_storage_lock_path(self.campaign_root),
                campaign_id=self.campaign_id,
                source_git_sha=self.source_git_sha,
            )
        else:
            records, runner_frontiers = self._collect_frontier_records(
                scope=str(retained.payload.get("barrier_scope")),
                target_tick=retained.frontier_tick,
            )
            expected_records = _sequence(
                retained.payload.get("tasks"),
                field="retained frontier tasks",
            )
            if _canonical_json_bytes(records) != _canonical_json_bytes(
                expected_records
            ):
                raise OpenEcologyCampaignCoordinatorError(
                    "retained frontier task pins are stale"
                )
            barrier_context = self._barrier_context_factory(runner_frontiers)
        with barrier_context:
            scan = self._storage_scanner(
                self.campaign_root,
                campaign_id=self.campaign_id,
                source_git_sha=self.source_git_sha,
                limits=self._storage_limits,
            )
            _assert_storage_scan_matches_receipt(scan, retained)
            return scan

    def _validate_carried_frontiers_on_disk(
        self,
        frontier_records: Sequence[Mapping[str, object]],
    ) -> None:
        """Bind every unmaterialized carried row to live CURRENT under the barrier."""

        for raw_record in frontier_records:
            record = _mapping(raw_record, field="frontier record")
            if (
                record.get("frontier_observation_mode")
                != "retained_unmaterialized_worker"
            ):
                continue
            task_id = _identifier(
                record.get("task_id"),
                field="carried frontier task_id",
            )
            task = self._task_by_id.get(task_id)
            if task is None:
                raise OpenEcologyCampaignCoordinatorError(
                    f"carried frontier names unknown task {task_id}"
                )
            pins = _resume_pins_from_dict(
                _mapping(
                    record.get("aggregate_resume_pins"),
                    field=f"{task_id}.carried aggregate_resume_pins",
                )
            )
            try:
                inspect_current_open_ecology_aggregate_generation(
                    self.campaign_root / "aggregates" / task_id,
                    evidence_directory=self.campaign_root / task_id,
                    pins=pins,
                )
            except (OpenEcologyAggregateCommitError, OSError) as error:
                raise OpenEcologyCampaignCoordinatorError(
                    f"carried frontier CURRENT/evidence drifted for {task_id}"
                ) from error

    def _scan_frontier_storage(
        self,
        *,
        target_tick: int,
        force_complete: bool = False,
    ) -> tuple[StorageScan, str, str | None]:
        del force_complete
        scan = self._storage_scanner(
            self.campaign_root,
            campaign_id=self.campaign_id,
            source_git_sha=self.source_git_sha,
            limits=self._storage_limits,
        )
        mode = (
            "terminal_complete_sha256_v1"
            if target_tick == OPEN_ECOLOGY_PHASE_D_TARGET_TICKS
            else "complete_sha256_v1"
        )
        return scan, mode, None

    def _begin_barrier_intent(
        self,
        *,
        scope: str,
        target_tick: int,
        selected_task_ids: set[str],
        allow_existing_unresolved: bool,
    ) -> dict[str, object]:
        sequence_index = self._coordination_sequence_index + 1
        selected = [
            task.task_id for task in self.tasks if task.task_id in selected_task_ids
        ]
        predecessor_states = [
            self._process_worker_expected_state(task_id).to_dict()
            for task_id in selected
        ]
        attempt_authorities: list[dict[str, object]] = []
        for state in predecessor_states:
            task_id = _identifier(
                state.get("task_id"),
                field="barrier intent task_id",
            )
            task = self._task_by_id[task_id]
            pins_payload = state.get("aggregate_resume_pins")
            if pins_payload is None:
                predecessor_commit = "genesis"
                run_generation_id = _persistent_run_generation_id(
                    campaign_id=self.campaign_id,
                    task_id=task_id,
                )
                predecessor_pins = None
            else:
                pins = _mapping(
                    pins_payload,
                    field=f"{state['task_id']}.intent predecessor pins",
                )
                identity = _mapping(
                    pins.get("identity"),
                    field=f"{state['task_id']}.intent predecessor identity",
                )
                predecessor_commit = _sha256(
                    pins.get("commit_sha256"),
                    field=f"{state['task_id']}.intent predecessor commit",
                )
                run_generation_id = _identifier(
                    identity.get("run_generation_id"),
                    field=f"{state['task_id']}.intent run_generation_id",
                )
                predecessor_pins = _resume_pins_from_dict(pins)
            authority = {
                "task_id": task_id,
                "run_generation_id": run_generation_id,
                "predecessor_aggregate_commit_sha256": predecessor_commit,
                "requested_target_tick": target_tick,
            }
            authority["attempt_authority_sha256"] = (
                persistent_interval_attempt_authority_sha256(
                    task,
                    campaign_id=self.campaign_id,
                    predecessor=predecessor_pins,
                    target_tick=target_tick,
                )
            )
            attempt_authorities.append(authority)
        unsigned_payload: dict[str, object] = {
            "schema_version": OPEN_ECOLOGY_CAMPAIGN_BARRIER_INTENT_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "campaign_root": str(self.campaign_root),
            "receipt_directory": str(self.receipt_directory),
            "source_git_sha": self.source_git_sha,
            "source_manifest_sha256": self.source_manifest_sha256,
            "launch_authority": (
                None if self._launch_authority is None else dict(self._launch_authority)
            ),
            "matrix_sha256": self.matrix_sha256,
            "assignment_sha256": self.assignment_sha256,
            "worker_count": self.worker_count,
            "worker_execution_contract": (
                "single_process_sequential_v1"
                if self._worker_launcher is None
                else self._worker_launcher.execution_contract
            ),
            "coordination_sequence_index": sequence_index,
            "barrier_scope": scope,
            "target_tick": target_tick,
            "selected_task_ids": selected,
            "predecessor_states": predecessor_states,
            "attempt_authorities": attempt_authorities,
            "attempt_authorities_sha256": _sha256_bytes(
                _canonical_json_bytes(attempt_authorities)
            ),
            "qualification_frontier_tick_before": (self._qualification_frontier_tick),
            "matrix_frontier_tick_before": self._matrix_frontier_tick,
            "previous_frontier_receipt_sha256": (
                None
                if self._previous_frontier_receipt is None
                else self._previous_frontier_receipt.receipt_sha256
            ),
            "no_task_mutation_precedes_this_intent": True,
            "unresolved_intent_blocks_new_attempt": True,
            "pruning_or_deletion_authorized": False,
        }
        payload = {
            **unsigned_payload,
            "intent_payload_sha256": _sha256_bytes(
                _canonical_json_bytes(unsigned_payload)
            ),
        }
        envelope = {
            "schema_version": (
                OPEN_ECOLOGY_CAMPAIGN_BARRIER_INTENT_ENVELOPE_SCHEMA_VERSION
            ),
            "payload": payload,
            "payload_sha256": _sha256_bytes(_canonical_json_bytes(payload)),
        }
        encoded = _canonical_json_bytes(envelope)
        if len(encoded) > OPEN_ECOLOGY_CAMPAIGN_MAX_BARRIER_INTENT_BYTES:
            raise OpenEcologyCampaignCoordinatorError(
                "barrier intent exceeds its 4-MiB ceiling"
            )
        destination = self.receipt_directory / (
            f"barrier-intent-{sequence_index:04d}-{scope}-{target_tick:08d}.json"
        )
        receipt_destination = self.receipt_directory / (
            f"step-{sequence_index:04d}-{scope}-{target_tick:08d}.json"
        )
        if destination.exists() or destination.is_symlink():
            (
                loaded_payload,
                intent_sha256,
                loaded_payload_sha256,
            ) = _load_barrier_intent(destination)
            if _canonical_json_bytes(loaded_payload) != _canonical_json_bytes(payload):
                raise OpenEcologyCampaignCoordinatorError(
                    "existing barrier intent does not match the exact retry"
                )
            if loaded_payload_sha256 != envelope["payload_sha256"]:
                raise OpenEcologyCampaignCoordinatorError(
                    "existing barrier intent payload authority drifted"
                )
            if receipt_destination.exists() or receipt_destination.is_symlink():
                raise OpenEcologyCampaignCoordinatorError(
                    "barrier intent is already resolved by a frontier receipt; "
                    "resume from that receipt"
                )
            if not allow_existing_unresolved:
                raise OpenEcologyCampaignCoordinatorError(
                    "unresolved barrier intent blocks a new attempt; inspect the "
                    "preserved task attempts before recovery"
                )
        else:
            _atomic_create(destination, encoded)
            intent_sha256 = _sha256_bytes(encoded)
        return {
            "path": str(destination),
            "intent_sha256": intent_sha256,
            "payload_sha256": envelope["payload_sha256"],
            "attempt_authorities_sha256": payload["attempt_authorities_sha256"],
        }

    def _advance_runners(
        self,
        target_tick: int,
        *,
        selected_task_ids: set[str],
    ) -> None:
        if self.worker_count > 1 and not self.parallel_execution_authorized:
            raise OpenEcologyCampaignCoordinatorError(
                "parallel campaign advance is not authorized by the runner "
                "locking contract"
            )
        if self.worker_count > 1 and self._worker_launcher is None:
            raise OpenEcologyCampaignCoordinatorError(
                "parallel campaign advance requires a persistent process/host "
                "worker launcher; in-process threads are not launch authority"
            )
        if self._worker_launcher is None:
            missing = sorted(selected_task_ids - set(self._runners))
            if missing:
                if self._pending_runner_factory is None:
                    raise OpenEcologyCampaignCoordinatorError(
                        "selected campaign tasks have no runner: " + ",".join(missing)
                    )
                for task_id in missing:
                    task = self._task_by_id[task_id]
                    runner = self._pending_runner_factory(task)
                    if (
                        runner.task != task
                        or runner.observed_tick != 0
                        or runner.extinct
                        or (
                            runner.latest_aggregate_resume_pins is not None
                            and not _is_exact_genesis_pins(
                                runner.latest_aggregate_resume_pins,
                                task=task,
                                campaign_id=self.campaign_id,
                                frontier=runner.campaign_barrier_frontier(),
                            )
                        )
                    ):
                        raise OpenEcologyCampaignCoordinatorError(
                            "pending runner factory returned drifted state for "
                            f"{task_id}"
                        )
                    self._runners[task_id] = runner
            for assignment in self.assignments:
                if assignment.task_id in selected_task_ids:
                    self._advance_runner(assignment.task_id, target_tick)
            self._last_worker_batch_summary = None
            return

        selected_in_matrix_order = tuple(
            task.task_id for task in self.tasks if task.task_id in selected_task_ids
        )
        expected_states = {
            task_id: self._process_worker_expected_state(task_id)
            for task_id in selected_in_matrix_order
        }
        batch = self._worker_launcher.advance_frontier(
            selected_task_ids=selected_in_matrix_order,
            target_tick=target_tick,
            expected_states=expected_states,
        )
        if (
            batch.target_tick != target_tick
            or batch.selected_task_ids != selected_in_matrix_order
            or batch.assignment_sha256 != self.assignment_sha256
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "process worker batch identity drifted"
            )
        views = batch.runner_views(self._task_by_id)
        if set(views) != selected_task_ids:
            raise OpenEcologyCampaignCoordinatorError(
                "process worker batch runner views are incomplete"
            )
        self._runners.update(views)
        self._last_worker_batch_summary = _process_worker_batch_summary(batch)
        for task_id in selected_in_matrix_order:
            runner = self._runners[task_id]
            if not runner.extinct and runner.observed_tick != target_tick:
                raise OpenEcologyCampaignCoordinatorError(
                    f"worker launcher left {task_id} before the frontier"
                )
            if runner.extinct and runner.observed_tick > target_tick:
                raise OpenEcologyCampaignCoordinatorError(
                    f"worker launcher moved terminal {task_id} past frontier"
                )

    def _reconcile_runners(
        self,
        *,
        target_tick: int,
        selected_task_ids: set[str],
        barrier_intent: Mapping[str, object],
        failure_observation: Mapping[str, object],
    ) -> dict[str, object]:
        """Resolve only the immutable original attempt through worker authority."""

        if self._worker_launcher is None:
            raise OpenEcologyCampaignCoordinatorError(
                "unresolved barrier intent recovery requires the persistent "
                "process/host worker launcher"
            )
        raw_path = barrier_intent.get("path")
        if not isinstance(raw_path, str):
            raise OpenEcologyCampaignCoordinatorError(
                "barrier recovery intent path is not a string"
            )
        intent_payload, intent_sha256, intent_payload_sha256 = _load_barrier_intent(
            Path(raw_path)
        )
        if (
            barrier_intent.get("intent_sha256") != intent_sha256
            or barrier_intent.get("payload_sha256") != intent_payload_sha256
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "barrier recovery intent binding drifted"
            )
        selected = tuple(
            _identifier(task_id, field="recovery selected task_id")
            for task_id in _sequence(
                intent_payload.get("selected_task_ids"),
                field="recovery selected_task_ids",
            )
        )
        expected_selected = tuple(
            task.task_id for task in self.tasks if task.task_id in selected_task_ids
        )
        if selected != expected_selected:
            raise OpenEcologyCampaignCoordinatorError(
                "barrier recovery selected task matrix drifted"
            )
        expected_states: dict[str, ProcessWorkerExpectedState] = {}
        for task_id, raw_state in zip(
            selected,
            _sequence(
                intent_payload.get("predecessor_states"),
                field="recovery predecessor_states",
            ),
            strict=True,
        ):
            state = _mapping(
                raw_state,
                field=f"{task_id}.recovery predecessor state",
            )
            pins_payload = state.get("aggregate_resume_pins")
            expected_states[task_id] = ProcessWorkerExpectedState(
                task_id=task_id,
                observed_tick=_nonnegative_int(
                    state.get("observed_tick"),
                    field=f"{task_id}.recovery predecessor observed_tick",
                ),
                terminal_extinct=_boolean(
                    state.get("terminal_extinct"),
                    field=f"{task_id}.recovery predecessor terminal_extinct",
                ),
                aggregate_resume_pins=(
                    None
                    if pins_payload is None
                    else _resume_pins_from_dict(
                        _mapping(
                            pins_payload,
                            field=f"{task_id}.recovery predecessor pins",
                        )
                    )
                ),
            )
        attempt_authorities: dict[str, str] = {}
        for task_id, raw_attempt in zip(
            selected,
            _sequence(
                intent_payload.get("attempt_authorities"),
                field="recovery attempt_authorities",
            ),
            strict=True,
        ):
            attempt = _mapping(
                raw_attempt,
                field=f"{task_id}.recovery attempt authority",
            )
            if attempt.get("task_id") != task_id:
                raise OpenEcologyCampaignCoordinatorError(
                    f"barrier recovery attempt task drifted for {task_id}"
                )
            attempt_authorities[task_id] = _sha256(
                attempt.get("attempt_authority_sha256"),
                field=f"{task_id}.recovery attempt authority SHA256",
            )

        batch = self._worker_launcher.reconcile_frontier(
            selected_task_ids=selected,
            target_tick=target_tick,
            expected_states=expected_states,
            intent_sha256=intent_sha256,
            attempt_authority_sha256_by_task=attempt_authorities,
        )
        if (
            batch.target_tick != target_tick
            or batch.selected_task_ids != selected
            or batch.assignment_sha256 != self.assignment_sha256
            or batch.intent_sha256 != intent_sha256
            or dict(batch.attempt_authority_sha256_by_task) != attempt_authorities
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "process worker reconciliation identity drifted"
            )
        views = batch.runner_views(self._task_by_id)
        if set(views) != selected_task_ids:
            raise OpenEcologyCampaignCoordinatorError(
                "process worker reconciliation runner views are incomplete"
            )
        self._runners.update(views)
        self._last_worker_batch_summary = _process_worker_reconciliation_batch_summary(
            batch
        )
        for task_id in selected:
            runner = self._runners[task_id]
            if not runner.extinct and runner.observed_tick != target_tick:
                raise OpenEcologyCampaignCoordinatorError(
                    f"reconciliation left {task_id} before the frontier"
                )
            if runner.extinct and runner.observed_tick > target_tick:
                raise OpenEcologyCampaignCoordinatorError(
                    f"reconciliation moved terminal {task_id} past frontier"
                )

        unsigned: dict[str, object] = {
            "schema_version": OPEN_ECOLOGY_CAMPAIGN_BARRIER_RECOVERY_SCHEMA_VERSION,
            "intent_path": raw_path,
            "intent_sha256": intent_sha256,
            "intent_payload_sha256": intent_payload_sha256,
            "attempt_authorities_sha256": _sha256(
                intent_payload.get("attempt_authorities_sha256"),
                field="recovery attempt_authorities_sha256",
            ),
            "failure_observation": dict(failure_observation),
            "process_reconciliation_sha256": batch.reconciliation_sha256,
            "disposition_by_task": {
                task_id: batch.disposition_by_task[task_id] for task_id in selected
            },
            "original_intent_reused": True,
            "new_attempt_authorized": False,
            "failure_journal_preserved": True,
            "pruning_or_deletion_authorized": False,
        }
        return {
            **unsigned,
            "recovery_sha256": _sha256_bytes(_canonical_json_bytes(unsigned)),
        }

    def _process_worker_expected_state(
        self,
        task_id: str,
    ) -> ProcessWorkerExpectedState:
        runner = self._runners.get(task_id)
        if runner is not None:
            pins = runner.latest_aggregate_resume_pins
            if runner.observed_tick == 0 and pins is not None:
                if not _is_exact_genesis_pins(
                    pins,
                    task=self._task_by_id[task_id],
                    campaign_id=self.campaign_id,
                    frontier=runner.campaign_barrier_frontier(),
                ):
                    raise OpenEcologyCampaignCoordinatorError(
                        f"tick-zero genesis pins drifted for {task_id}"
                    )
                pins = None
            return ProcessWorkerExpectedState(
                task_id=task_id,
                observed_tick=runner.observed_tick,
                terminal_extinct=runner.extinct,
                aggregate_resume_pins=pins,
            )
        if self._previous_frontier_receipt is not None:
            records = _sequence(
                self._previous_frontier_receipt.payload.get("tasks"),
                field="retained frontier tasks",
            )
            for raw_record in records:
                record = _mapping(raw_record, field="retained frontier task")
                if record.get("task_id") != task_id:
                    continue
                if record.get("task_state") == "pending_unstarted":
                    break
                return ProcessWorkerExpectedState(
                    task_id=task_id,
                    observed_tick=_nonnegative_int(
                        record.get("observed_tick"),
                        field=f"{task_id}.retained observed_tick",
                    ),
                    terminal_extinct=_boolean(
                        record.get("terminal_extinct"),
                        field=f"{task_id}.retained terminal_extinct",
                    ),
                    aggregate_resume_pins=_resume_pins_from_dict(
                        _mapping(
                            record.get("aggregate_resume_pins"),
                            field=f"{task_id}.retained aggregate_resume_pins",
                        )
                    ),
                )
        return ProcessWorkerExpectedState(
            task_id=task_id,
            observed_tick=0,
            terminal_extinct=False,
            aggregate_resume_pins=None,
        )

    def _next_step(
        self,
    ) -> tuple[str, int, set[str]] | None:
        priority_ids = {
            task.task_id for task in prioritized_persistent_island_triplet(self.tasks)
        }
        if self._qualification_frontier_tick < OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK:
            return (
                "qualification_triplet",
                self._qualification_frontier_tick
                + OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS,
                priority_ids,
            )
        if self._matrix_frontier_tick < OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK:
            return (
                "matrix_catchup",
                self._matrix_frontier_tick
                + OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS,
                set(self._task_by_id) - priority_ids,
            )
        if self._matrix_frontier_tick < OPEN_ECOLOGY_PHASE_D_TARGET_TICKS:
            return (
                "full_matrix",
                self._matrix_frontier_tick
                + OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS,
                set(self._task_by_id),
            )
        return None

    def _next_step_payload(self) -> dict[str, object] | None:
        step = self._next_step()
        if step is None:
            return None
        scope, target_tick, task_ids = step
        return {
            "barrier_scope": scope,
            "target_tick": target_tick,
            "task_count": len(task_ids),
            "task_ids_sha256": _sha256_bytes(_canonical_json_bytes(sorted(task_ids))),
        }

    def _requested_frontier_reached(self, requested: int) -> bool:
        if requested <= OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK:
            return self._qualification_frontier_tick >= requested
        return (
            self._qualification_frontier_tick
            >= OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
            and self._matrix_frontier_tick >= requested
        )

    def _expected_task_ticks_after_step(
        self,
        *,
        scope: str,
        target_tick: int,
    ) -> dict[str, int]:
        priority_ids = {
            task.task_id for task in prioritized_persistent_island_triplet(self.tasks)
        }
        if scope == "qualification_triplet":
            if target_tick not in (5_000, 10_000):
                raise OpenEcologyCampaignCoordinatorError(
                    "qualification target must be 5,000 or 10,000"
                )
            return {
                task.task_id: (target_tick if task.task_id in priority_ids else 0)
                for task in self.tasks
            }
        if scope == "matrix_catchup":
            if (
                self._qualification_frontier_tick
                != OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
                or target_tick not in (5_000, 10_000)
            ):
                raise OpenEcologyCampaignCoordinatorError(
                    "matrix catchup requires a completed qualification triplet"
                )
            return {
                task.task_id: (
                    OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
                    if task.task_id in priority_ids
                    else target_tick
                )
                for task in self.tasks
            }
        if scope == "full_matrix":
            if (
                self._qualification_frontier_tick
                != OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
                or target_tick <= OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
            ):
                raise OpenEcologyCampaignCoordinatorError(
                    "full matrix frontier cannot precede catchup"
                )
            return {task.task_id: target_tick for task in self.tasks}
        raise OpenEcologyCampaignCoordinatorError("campaign barrier scope is unknown")

    def _advance_runner(self, task_id: str, target_tick: int) -> None:
        runner = self._runners[task_id]
        result = runner.advance_to(target_tick)
        if (
            result.task_id != task_id
            or result.requested_target_tick != target_tick
            or result.observed_tick != runner.observed_tick
            or result.extinct is not runner.extinct
        ):
            raise OpenEcologyCampaignCoordinatorError(
                f"runner advance result drifted for {task_id}"
            )
        if not runner.extinct and runner.observed_tick != target_tick:
            raise OpenEcologyCampaignCoordinatorError(
                f"live runner failed to reach frontier for {task_id}"
            )
        if runner.extinct and runner.observed_tick > target_tick:
            raise OpenEcologyCampaignCoordinatorError(
                f"terminal runner is ahead of frontier for {task_id}"
            )

    def _collect_frontier_records(
        self,
        *,
        scope: str,
        target_tick: int,
    ) -> tuple[list[dict[str, object]], list[Mapping[str, object]]]:
        records: list[dict[str, object]] = []
        runner_frontiers: list[Mapping[str, object]] = []
        expected_ticks = self._expected_task_ticks_after_step(
            scope=scope,
            target_tick=target_tick,
        )
        for task in self.tasks:
            runner = self._runners.get(task.task_id)
            expected_tick = expected_ticks[task.task_id]
            if runner is None:
                if expected_tick > 0:
                    records.append(
                        self._carried_frontier_record(
                            task,
                            expected_tick=expected_tick,
                            barrier_target_tick=target_tick,
                        )
                    )
                else:
                    records.append(_pending_frontier_record(task))
                continue
            if (
                expected_tick == 0
                and runner.latest_aggregate_resume_pins is None
                and runner.observed_tick == 0
            ):
                records.append(_pending_frontier_record(task))
                continue
            runner_frontier = _runner_barrier_frontier(runner)
            runner_frontiers.append(runner_frontier)
            records.append(
                _frontier_record(
                    task,
                    runner,
                    expected_tick=expected_tick,
                    barrier_target_tick=target_tick,
                    campaign_id=self.campaign_id,
                    campaign_root=self.campaign_root,
                    source_git_sha=self.source_git_sha,
                )
            )
        if len(records) != OPEN_ECOLOGY_PHASE_D_TASK_COUNT:
            raise OpenEcologyCampaignCoordinatorError(
                "campaign frontier is missing tasks"
            )
        return records, runner_frontiers

    def _carried_frontier_record(
        self,
        task: PersistentIslandTask,
        *,
        expected_tick: int,
        barrier_target_tick: int,
    ) -> dict[str, object]:
        if self._worker_launcher is None or self._previous_frontier_receipt is None:
            raise OpenEcologyCampaignCoordinatorError(
                f"started task {task.task_id} has no live or retained frontier"
            )
        previous_records = _sequence(
            self._previous_frontier_receipt.payload.get("tasks"),
            field="previous frontier tasks",
        )
        matches = [
            _mapping(record, field="previous frontier task")
            for record in previous_records
            if _mapping(record, field="previous frontier task").get("task_id")
            == task.task_id
        ]
        if len(matches) != 1:
            raise OpenEcologyCampaignCoordinatorError(
                f"retained frontier identity is missing for {task.task_id}"
            )
        previous = dict(matches[0])
        if previous.get("task_state") not in {"active", "terminal_extinct"}:
            raise OpenEcologyCampaignCoordinatorError(
                f"retained started state is unavailable for {task.task_id}"
            )
        pins = _resume_pins_from_dict(
            _mapping(
                previous.get("aggregate_resume_pins"),
                field=f"{task.task_id}.retained aggregate_resume_pins",
            )
        )
        observed_tick = _nonnegative_int(
            previous.get("observed_tick"),
            field=f"{task.task_id}.retained observed_tick",
        )
        terminal = _boolean(
            previous.get("terminal_extinct"),
            field=f"{task.task_id}.retained terminal_extinct",
        )
        if (
            pins.identity.source_git_sha != self.source_git_sha
            or pins.identity.island_id != task.task_id
            or pins.identity.tick != observed_tick - 1
            or (terminal and observed_tick > expected_tick)
            or (not terminal and observed_tick != expected_tick)
        ):
            raise OpenEcologyCampaignCoordinatorError(
                f"retained frontier pins drifted for {task.task_id}"
            )
        previous.pop("frontier_record_sha256", None)
        previous["requested_barrier_target_tick"] = barrier_target_tick
        previous["expected_task_tick"] = expected_tick
        previous["frontier_observation_mode"] = "retained_unmaterialized_worker"
        previous["frontier_record_sha256"] = _sha256_bytes(
            _canonical_json_bytes(previous)
        )
        return previous

    def _write_or_validate_frontier_receipt(
        self,
        *,
        target_tick: int,
        scope: str,
        frontier_records: Sequence[Mapping[str, object]],
        source_snapshot: Mapping[str, object],
        health_before: Mapping[str, object],
        health_after: Mapping[str, object],
        storage_scan: StorageScan,
        storage_validation_mode: str,
        predecessor_storage_manifest_sha256: str | None,
        barrier_intent: Mapping[str, object],
        barrier_recovery: Mapping[str, object] | None,
    ) -> CampaignFrontierReceipt:
        sequence_index = self._coordination_sequence_index + 1
        destination = self.receipt_directory / (
            f"step-{sequence_index:04d}-{scope}-{target_tick:08d}.json"
        )
        if destination.exists():
            raise OpenEcologyCampaignCoordinatorError(
                "immutable frontier receipt already exists; retain its external "
                "SHA256 and resume from it instead of regenerating authority"
            )

        previous_receipt_sha256 = (
            None
            if self._previous_frontier_receipt is None
            else self._previous_frontier_receipt.receipt_sha256
        )
        qualification_ids = [
            task.task_id for task in prioritized_persistent_island_triplet(self.tasks)
        ]
        qualification_frontier_after = (
            target_tick
            if scope == "qualification_triplet"
            else self._qualification_frontier_tick
        )
        matrix_frontier_after = (
            self._matrix_frontier_tick
            if scope == "qualification_triplet"
            else target_tick
        )
        all_tasks_at_common_frontier = scope == "full_matrix" or (
            scope == "matrix_catchup"
            and target_tick == OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
        )
        payload: dict[str, object] = {
            "schema_version": OPEN_ECOLOGY_CAMPAIGN_FRONTIER_RECEIPT_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "campaign_root": str(self.campaign_root),
            "receipt_directory": str(self.receipt_directory),
            "source_git_sha": self.source_git_sha,
            "source_manifest_sha256": self.source_manifest_sha256,
            "launch_authority": (
                None if self._launch_authority is None else dict(self._launch_authority)
            ),
            "source_snapshot": dict(source_snapshot),
            "matrix_sha256": self.matrix_sha256,
            "assignment_sha256": self.assignment_sha256,
            "worker_count": self.worker_count,
            "parallel_execution_authorized": self.parallel_execution_authorized,
            "worker_execution_contract": (
                "single_process_sequential_v1"
                if self._worker_launcher is None
                else self._worker_launcher.execution_contract
            ),
            "coordination_sequence_index": sequence_index,
            "barrier_scope": scope,
            "frontier_interval_ticks": (OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS),
            "frontier_tick": target_tick,
            "qualification_frontier_tick": qualification_frontier_after,
            "matrix_frontier_tick": matrix_frontier_after,
            "previous_frontier_receipt_sha256": previous_receipt_sha256,
            "qualification": {
                "task_ids": qualification_ids,
                "milestone_tick": OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK,
                "prioritized_before_other_tasks": True,
                "milestone_reached_or_terminal": (
                    qualification_frontier_after
                    >= OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
                ),
                "interim_acceptance_or_tuning_authorized": False,
            },
            "static_worker_assignments": [
                assignment.to_dict() for assignment in self.assignments
            ],
            "process_worker_frontier_batch": (
                None
                if self._last_worker_batch_summary is None
                else dict(self._last_worker_batch_summary)
            ),
            "barrier_intent": dict(barrier_intent),
            "barrier_recovery": (
                None if barrier_recovery is None else dict(barrier_recovery)
            ),
            "tasks": [dict(record) for record in frontier_records],
            "storage": _storage_scan_payload(
                storage_scan,
                validation_mode=storage_validation_mode,
                predecessor_entry_manifest_sha256=(predecessor_storage_manifest_sha256),
            ),
            "resource_health": {
                "before_advance": dict(health_before),
                "frontier_quiescent": dict(health_after),
            },
            "all_started_tasks_quiescent": True,
            "all_tasks_at_common_frontier": all_tasks_at_common_frontier,
            "next_interval_authorized": (
                matrix_frontier_after < OPEN_ECOLOGY_PHASE_D_TARGET_TICKS
            ),
            "pruning_or_deletion_performed": False,
            "pruning_or_deletion_available": False,
        }
        payload_sha256 = _sha256_bytes(_canonical_json_bytes(payload))
        envelope = {
            "schema_version": (OPEN_ECOLOGY_CAMPAIGN_FRONTIER_ENVELOPE_SCHEMA_VERSION),
            "payload": payload,
            "payload_sha256": payload_sha256,
        }
        receipt_bytes = _canonical_json_bytes(envelope)
        if len(receipt_bytes) > OPEN_ECOLOGY_CAMPAIGN_MAX_RECEIPT_BYTES:
            raise OpenEcologyCampaignCoordinatorError(
                "frontier receipt exceeds its 16-MiB ceiling"
            )
        _atomic_create(destination, receipt_bytes)
        loaded = load_campaign_frontier_receipt(
            destination,
            expected_receipt_sha256=_sha256_bytes(receipt_bytes),
        )
        if loaded.payload_sha256 != payload_sha256:
            raise OpenEcologyCampaignCoordinatorError(
                "frontier receipt readback changed"
            )
        return loaded

    def _validate_runner_positions(self) -> None:
        retained_pins = (
            {}
            if self._previous_frontier_receipt is None
            else self._previous_frontier_receipt.resume_pins_by_task()
        )
        for task in self.tasks:
            runner = self._runners.get(task.task_id)
            expected_tick = (
                max(
                    self._qualification_frontier_tick,
                    self._matrix_frontier_tick,
                )
                if task.task_id
                in {
                    item.task_id
                    for item in prioritized_persistent_island_triplet(self.tasks)
                }
                else self._matrix_frontier_tick
            )
            if runner is None:
                if expected_tick != 0 and self._worker_launcher is None:
                    raise OpenEcologyCampaignCoordinatorError(
                        f"started task {task.task_id} is missing its runner"
                    )
                continue
            observed_tick = _nonnegative_int(
                runner.observed_tick,
                field=f"{task.task_id}.observed_tick",
            )
            if expected_tick == 0:
                if (
                    observed_tick != 0
                    or runner.extinct
                    or (
                        runner.latest_aggregate_resume_pins is not None
                        and not _is_exact_genesis_pins(
                            runner.latest_aggregate_resume_pins,
                            task=task,
                            campaign_id=self.campaign_id,
                            frontier=runner.campaign_barrier_frontier(),
                        )
                    )
                ):
                    raise OpenEcologyCampaignCoordinatorError(
                        "fresh campaign runners must all start at uncommitted tick zero"
                    )
                continue
            if task.task_id not in retained_pins:
                raise OpenEcologyCampaignCoordinatorError(
                    "retained frontier receipt is missing a task pin"
                )
            current_pins = runner.latest_aggregate_resume_pins
            if current_pins != retained_pins[task.task_id]:
                raise OpenEcologyCampaignCoordinatorError(
                    f"runner CURRENT does not match retained pins for {task.task_id}"
                )
            retained_tick = retained_pins[task.task_id].identity.tick
            if retained_tick != observed_tick - 1:
                raise OpenEcologyCampaignCoordinatorError(
                    f"runner tick does not match retained pins for {task.task_id}"
                )
            if runner.extinct:
                if observed_tick > expected_tick:
                    raise OpenEcologyCampaignCoordinatorError(
                        f"terminal runner exceeds retained frontier for {task.task_id}"
                    )
            elif observed_tick != expected_tick:
                raise OpenEcologyCampaignCoordinatorError(
                    f"live runner is stale at retained frontier for {task.task_id}"
                )

    def _validate_receipt_identity(self, receipt: CampaignFrontierReceipt) -> None:
        payload = receipt.payload
        expected = {
            "campaign_id": self.campaign_id,
            "campaign_root": str(self.campaign_root),
            "receipt_directory": str(self.receipt_directory),
            "source_git_sha": self.source_git_sha,
            "source_manifest_sha256": self.source_manifest_sha256,
            "launch_authority": (
                None if self._launch_authority is None else dict(self._launch_authority)
            ),
            "matrix_sha256": self.matrix_sha256,
            "assignment_sha256": self.assignment_sha256,
            "worker_count": self.worker_count,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise OpenEcologyCampaignCoordinatorError(
                    f"retained frontier receipt {key} drifted"
                )
        task_records = _sequence(payload.get("tasks"), field="retained tasks")
        if [
            record.get("task_id") for record in map(_mapping_unlabelled, task_records)
        ] != [task.task_id for task in self.tasks]:
            raise OpenEcologyCampaignCoordinatorError(
                "retained frontier task order is not the exact matrix"
            )
        retained_assignments = _sequence(
            payload.get("static_worker_assignments"),
            field="retained static_worker_assignments",
        )
        if _canonical_json_bytes(retained_assignments) != _canonical_json_bytes(
            [assignment.to_dict() for assignment in self.assignments]
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "retained frontier static worker assignments drifted"
            )
        qualification = _mapping(
            payload.get("qualification"),
            field="retained qualification",
        )
        expected_qualification_ids = [
            task.task_id for task in prioritized_persistent_island_triplet(self.tasks)
        ]
        if qualification.get("task_ids") != expected_qualification_ids:
            raise OpenEcologyCampaignCoordinatorError(
                "retained qualification task identity drifted"
            )
        expected_ticks, selected_task_ids = _validated_receipt_frontier_state(
            payload,
            self.tasks,
        )
        for task, raw_record in zip(self.tasks, task_records, strict=True):
            record = _mapping(raw_record, field=f"retained task {task.task_id}")
            _validate_retained_task_record(
                task,
                record,
                source_git_sha=self.source_git_sha,
            )
            if record.get("expected_task_tick") != expected_ticks[task.task_id]:
                raise OpenEcologyCampaignCoordinatorError(
                    f"retained expected frontier drifted for {task.task_id}"
                )
        process_batch = payload.get("process_worker_frontier_batch")
        worker_contract = payload.get("worker_execution_contract")
        if (process_batch is None) is (
            worker_contract == OPEN_ECOLOGY_PROCESS_WORKER_LAUNCHER_CONTRACT
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "retained process worker contract/batch evidence drifted"
            )
        if process_batch is not None:
            batch = _mapping(process_batch, field="retained process worker batch")
            if (
                batch.get("target_tick") != receipt.frontier_tick
                or batch.get("assignment_sha256") != self.assignment_sha256
                or batch.get("selected_task_ids") != selected_task_ids
            ):
                raise OpenEcologyCampaignCoordinatorError(
                    "retained process worker batch identity drifted"
                )
        intent_binding = _mapping(
            payload.get("barrier_intent"),
            field="retained barrier_intent",
        )
        if set(intent_binding) != {
            "attempt_authorities_sha256",
            "intent_sha256",
            "path",
            "payload_sha256",
        }:
            raise OpenEcologyCampaignCoordinatorError(
                "retained barrier intent binding schema drifted"
            )
        expected_intent_path = self.receipt_directory / (
            f"barrier-intent-{payload['coordination_sequence_index']:04d}-"
            f"{payload['barrier_scope']}-{receipt.frontier_tick:08d}.json"
        )
        if intent_binding.get("path") != str(expected_intent_path):
            raise OpenEcologyCampaignCoordinatorError(
                "retained barrier intent path drifted"
            )
        (
            intent_payload,
            intent_sha256,
            intent_payload_sha256,
        ) = _load_barrier_intent(expected_intent_path)
        expected_intent_identity = {
            "campaign_id": self.campaign_id,
            "campaign_root": str(self.campaign_root),
            "receipt_directory": str(self.receipt_directory),
            "source_git_sha": self.source_git_sha,
            "source_manifest_sha256": self.source_manifest_sha256,
            "launch_authority": payload.get("launch_authority"),
            "matrix_sha256": self.matrix_sha256,
            "assignment_sha256": self.assignment_sha256,
            "coordination_sequence_index": payload["coordination_sequence_index"],
            "barrier_scope": payload["barrier_scope"],
            "target_tick": receipt.frontier_tick,
            "selected_task_ids": selected_task_ids,
            "previous_frontier_receipt_sha256": payload[
                "previous_frontier_receipt_sha256"
            ],
        }
        for field, expected_value in expected_intent_identity.items():
            if intent_payload.get(field) != expected_value:
                raise OpenEcologyCampaignCoordinatorError(
                    f"retained barrier intent {field} drifted"
                )
        if (
            intent_binding.get("intent_sha256") != intent_sha256
            or intent_binding.get("payload_sha256") != intent_payload_sha256
            or intent_binding.get("attempt_authorities_sha256")
            != intent_payload.get("attempt_authorities_sha256")
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "retained barrier intent digest binding drifted"
            )
        recovery_payload = payload.get("barrier_recovery")
        if recovery_payload is None:
            if process_batch is not None and "reconciliation_sha256" in batch:
                raise OpenEcologyCampaignCoordinatorError(
                    "retained normal frontier carries reconciliation evidence"
                )
            return
        recovery = dict(
            _mapping(
                recovery_payload,
                field="retained barrier_recovery",
            )
        )
        expected_recovery_keys = {
            "attempt_authorities_sha256",
            "disposition_by_task",
            "failure_journal_preserved",
            "failure_observation",
            "intent_path",
            "intent_payload_sha256",
            "intent_sha256",
            "new_attempt_authorized",
            "original_intent_reused",
            "process_reconciliation_sha256",
            "pruning_or_deletion_authorized",
            "recovery_sha256",
            "schema_version",
        }
        if set(recovery) != expected_recovery_keys:
            raise OpenEcologyCampaignCoordinatorError(
                "retained barrier recovery schema drifted"
            )
        if (
            recovery.get("schema_version")
            != OPEN_ECOLOGY_CAMPAIGN_BARRIER_RECOVERY_SCHEMA_VERSION
            or recovery.get("intent_path") != str(expected_intent_path)
            or recovery.get("intent_sha256") != intent_sha256
            or recovery.get("intent_payload_sha256") != intent_payload_sha256
            or recovery.get("attempt_authorities_sha256")
            != intent_payload.get("attempt_authorities_sha256")
            or recovery.get("original_intent_reused") is not True
            or recovery.get("new_attempt_authorized") is not False
            or recovery.get("failure_journal_preserved") is not True
            or recovery.get("pruning_or_deletion_authorized") is not False
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "retained barrier recovery authority drifted"
            )
        observed_recovery_sha256 = recovery.pop("recovery_sha256")
        if _sha256(
            observed_recovery_sha256,
            field="retained recovery SHA256",
        ) != _sha256_bytes(_canonical_json_bytes(recovery)):
            raise OpenEcologyCampaignCoordinatorError(
                "retained barrier recovery digest drifted"
            )
        failure_binding = _mapping(
            recovery.get("failure_observation"),
            field="retained recovery failure_observation",
        )
        expected_failure_path = expected_intent_path.with_name(
            expected_intent_path.name.replace(
                "barrier-intent-",
                "barrier-failure-",
                1,
            )
        )
        loaded_failure = _load_barrier_failure_observation(
            expected_failure_path,
            expected_intent_path=expected_intent_path,
            expected_intent_sha256=intent_sha256,
            campaign_id=self.campaign_id,
            source_git_sha=self.source_git_sha,
            matrix_sha256=self.matrix_sha256,
        )
        if _canonical_json_bytes(failure_binding) != _canonical_json_bytes(
            loaded_failure
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "retained barrier failure journal binding drifted"
            )
        if process_batch is None:
            raise OpenEcologyCampaignCoordinatorError(
                "retained barrier recovery lacks process reconciliation evidence"
            )
        dispositions = _mapping(
            recovery.get("disposition_by_task"),
            field="retained recovery dispositions",
        )
        if (
            batch.get("intent_sha256") != intent_sha256
            or batch.get("reconciliation_sha256")
            != recovery.get("process_reconciliation_sha256")
            or _canonical_json_bytes(batch.get("disposition_by_task"))
            != _canonical_json_bytes(dispositions)
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "retained process reconciliation binding drifted"
            )

    def _ensure_barrier_failure_observation(
        self,
        *,
        barrier_intent: Mapping[str, object],
        selected_task_ids: set[str],
    ) -> dict[str, object]:
        raw_intent_path = barrier_intent.get("path")
        if not isinstance(raw_intent_path, str):
            raise OpenEcologyCampaignCoordinatorError(
                "barrier recovery intent path is not a string"
            )
        intent_path = Path(raw_intent_path)
        failure_path = intent_path.with_name(
            intent_path.name.replace("barrier-intent-", "barrier-failure-", 1)
        )
        if not failure_path.exists() and not failure_path.is_symlink():
            self._record_barrier_failure(
                barrier_intent=barrier_intent,
                selected_task_ids=selected_task_ids,
                error=OpenEcologyCampaignCoordinatorError(
                    "coordinator restarted with an unresolved barrier intent; "
                    "the original failure observation was unavailable"
                ),
            )
        return _load_barrier_failure_observation(
            failure_path,
            expected_intent_path=intent_path,
            expected_intent_sha256=_sha256(
                barrier_intent.get("intent_sha256"),
                field="barrier recovery intent SHA256",
            ),
            campaign_id=self.campaign_id,
            source_git_sha=self.source_git_sha,
            matrix_sha256=self.matrix_sha256,
        )

    def _record_barrier_failure(
        self,
        *,
        barrier_intent: Mapping[str, object],
        selected_task_ids: set[str],
        error: BaseException,
    ) -> None:
        raw_intent_path = barrier_intent.get("path")
        if not isinstance(raw_intent_path, str):
            raise OpenEcologyCampaignCoordinatorError(
                "failed barrier intent path is not a string"
            ) from error
        intent_path = Path(raw_intent_path)
        destination = intent_path.with_name(
            intent_path.name.replace("barrier-intent-", "barrier-failure-", 1)
        )
        if destination.exists() or destination.is_symlink():
            return
        partial_results: list[dict[str, object]] = []
        raw_partial = getattr(error, "partial_results", None)
        if isinstance(raw_partial, Mapping):
            for task_id in sorted(raw_partial):
                result = raw_partial[task_id]
                to_dict = getattr(result, "to_dict", None)
                if callable(to_dict):
                    partial_results.append(dict(to_dict()))
        visible_runner_states: list[dict[str, object]] = []
        for task in self.tasks:
            if task.task_id not in selected_task_ids:
                continue
            runner = self._runners.get(task.task_id)
            if runner is None:
                continue
            pins = runner.latest_aggregate_resume_pins
            visible_runner_states.append(
                {
                    "task_id": task.task_id,
                    "observed_tick": runner.observed_tick,
                    "terminal_extinct": runner.extinct,
                    "aggregate_resume_pins": (
                        None if pins is None else _resume_pins_to_dict(pins)
                    ),
                }
            )
        unsigned: dict[str, object] = {
            "schema_version": OPEN_ECOLOGY_CAMPAIGN_BARRIER_FAILURE_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "source_git_sha": self.source_git_sha,
            "matrix_sha256": self.matrix_sha256,
            "intent_path": str(intent_path),
            "intent_sha256": _sha256(
                barrier_intent.get("intent_sha256"),
                field="failed barrier intent SHA256",
            ),
            "error_type": type(error).__name__,
            "error_message": str(error)[:4096] or "<no message>",
            "partial_process_results": partial_results,
            "visible_parent_runner_states": visible_runner_states,
            "failure_observation_may_be_incomplete": True,
            "receipt_published": False,
            "new_attempt_authorized": False,
            "pruning_or_deletion_authorized": False,
        }
        payload = {
            **unsigned,
            "failure_observation_sha256": _sha256_bytes(
                _canonical_json_bytes(unsigned)
            ),
        }
        encoded = _canonical_json_bytes(payload)
        if len(encoded) > OPEN_ECOLOGY_CAMPAIGN_MAX_BARRIER_INTENT_BYTES:
            raise OpenEcologyCampaignCoordinatorError(
                "barrier failure observation exceeds its 4-MiB ceiling"
            ) from error
        _atomic_create(destination, encoded)

    def _verified_source_snapshot(self) -> dict[str, object]:
        snapshot = dict(self._health_safe_mapping(self._source_probe()))
        if (
            snapshot.get("source_git_sha") != self.source_git_sha
            or snapshot.get("source_manifest_sha256") != self.source_manifest_sha256
            or snapshot.get("git_clean") is not True
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "live source authority does not match the sealed campaign"
            )
        return snapshot

    def _health_snapshot(self, phase: str, frontier_tick: int) -> dict[str, object]:
        if self._health_probe is None:
            raise OpenEcologyCampaignCoordinatorError(
                "a production resource/health probe is required before advance"
            )
        snapshot = dict(
            self._health_safe_mapping(self._health_probe(phase, frontier_tick))
        )
        if (
            snapshot.get("phase") != phase
            or snapshot.get("frontier_tick") != frontier_tick
            or snapshot.get("healthy") is not True
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "resource/health probe did not affirm this exact phase and frontier"
            )
        return snapshot

    @staticmethod
    def _health_safe_mapping(
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        if not isinstance(payload, Mapping):
            raise OpenEcologyCampaignCoordinatorError(
                "resource/source probe must return a mapping"
            )
        cloned = json.loads(_canonical_json_bytes(dict(payload)))
        if not isinstance(cloned, dict):
            raise OpenEcologyCampaignCoordinatorError(
                "resource/source probe root must remain a mapping"
            )
        return cloned

    @contextmanager
    def _default_barrier_context(
        self,
        runner_frontiers: Sequence[Mapping[str, object]],
    ) -> Iterator[object]:
        campaign_barrier = getattr(
            PersistentIslandRunner,
            "campaign_storage_barrier",
            None,
        )
        if campaign_barrier is not None:
            with campaign_barrier(
                campaign_root=self.campaign_root,
                campaign_id=self.campaign_id,
                source_git_sha=self.source_git_sha,
                frontiers=runner_frontiers,
            ) as held:
                yield held
            return
        with CampaignStorageLock(
            default_storage_lock_path(self.campaign_root),
            campaign_id=self.campaign_id,
            source_git_sha=self.source_git_sha,
        ) as held:
            yield held


def _read_bounded_authority_file(
    path: Path,
    *,
    max_bytes: int,
    field: str,
    forbid_group_or_other_write: bool,
) -> bytes:
    if not path.is_absolute():
        raise OpenEcologyCampaignCoordinatorError(f"{field} path must be absolute")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError as error:
        raise OpenEcologyCampaignCoordinatorError(f"{field} is missing") from error
    except OSError as error:
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} cannot be opened as a regular file without following links"
        ) from error
    try:
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
            or (forbid_group_or_other_write and mode & 0o022)
        ):
            raise OpenEcologyCampaignCoordinatorError(
                f"{field} must be one bounded single-link regular file"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    descriptor_identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    descriptor_identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    try:
        path_entry = os.lstat(path)
    except OSError as error:
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} path entry changed while reading"
        ) from error
    path_identity = (
        path_entry.st_dev,
        path_entry.st_ino,
        path_entry.st_mode,
        path_entry.st_nlink,
        path_entry.st_size,
        path_entry.st_mtime_ns,
        path_entry.st_ctime_ns,
    )
    if (
        descriptor_identity_before != descriptor_identity_after
        or descriptor_identity_after != path_identity
        or len(encoded) != before.st_size
        or len(encoded) > max_bytes
    ):
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} identity or path entry changed while reading"
        )
    return encoded


def load_campaign_frontier_receipt(
    path: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
) -> CampaignFrontierReceipt:
    """Load one strict immutable frontier receipt."""

    receipt_path = Path(path)
    if not receipt_path.is_absolute():
        raise OpenEcologyCampaignCoordinatorError(
            "frontier receipt path must be absolute"
        )
    payload_bytes = _read_bounded_authority_file(
        receipt_path,
        max_bytes=OPEN_ECOLOGY_CAMPAIGN_MAX_RECEIPT_BYTES,
        field="frontier receipt",
        forbid_group_or_other_write=False,
    )
    envelope = _strict_json_object(payload_bytes, field="frontier receipt")
    if set(envelope) != {"schema_version", "payload", "payload_sha256"}:
        raise OpenEcologyCampaignCoordinatorError(
            "frontier receipt envelope schema is not exact"
        )
    if (
        envelope["schema_version"]
        != OPEN_ECOLOGY_CAMPAIGN_FRONTIER_ENVELOPE_SCHEMA_VERSION
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "frontier receipt envelope version drifted"
        )
    payload = _mapping(envelope["payload"], field="frontier receipt payload")
    if (
        payload.get("schema_version")
        != OPEN_ECOLOGY_CAMPAIGN_FRONTIER_RECEIPT_SCHEMA_VERSION
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "frontier receipt payload version drifted"
        )
    payload_sha256 = _sha256(
        envelope["payload_sha256"],
        field="frontier receipt payload_sha256",
    )
    if _sha256_bytes(_canonical_json_bytes(payload)) != payload_sha256:
        raise OpenEcologyCampaignCoordinatorError(
            "frontier receipt payload digest mismatch"
        )
    receipt_sha256 = _sha256_bytes(payload_bytes)
    external_verified = False
    if expected_receipt_sha256 is not None:
        expected = _sha256(
            expected_receipt_sha256,
            field="expected frontier receipt SHA256",
        )
        if receipt_sha256 != expected:
            raise OpenEcologyCampaignCoordinatorError(
                "frontier receipt does not match external SHA256 authority"
            )
        external_verified = True
    return CampaignFrontierReceipt(
        path=receipt_path,
        payload=payload,
        payload_sha256=payload_sha256,
        receipt_sha256=receipt_sha256,
        external_sha256_verified=external_verified,
    )


def _load_barrier_intent(
    path: Path,
) -> tuple[dict[str, object], str, str]:
    if not path.is_absolute():
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent path must be absolute"
        )
    encoded = _read_bounded_authority_file(
        path,
        max_bytes=OPEN_ECOLOGY_CAMPAIGN_MAX_BARRIER_INTENT_BYTES,
        field="barrier intent",
        forbid_group_or_other_write=True,
    )
    envelope = _strict_json_object(encoded, field="barrier intent")
    if set(envelope) != {"schema_version", "payload", "payload_sha256"}:
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent envelope schema is not exact"
        )
    if (
        envelope.get("schema_version")
        != OPEN_ECOLOGY_CAMPAIGN_BARRIER_INTENT_ENVELOPE_SCHEMA_VERSION
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent envelope version drifted"
        )
    payload = dict(_mapping(envelope.get("payload"), field="barrier intent payload"))
    expected_payload_keys = {
        "assignment_sha256",
        "attempt_authorities",
        "attempt_authorities_sha256",
        "barrier_scope",
        "campaign_id",
        "campaign_root",
        "coordination_sequence_index",
        "intent_payload_sha256",
        "launch_authority",
        "matrix_frontier_tick_before",
        "matrix_sha256",
        "no_task_mutation_precedes_this_intent",
        "predecessor_states",
        "previous_frontier_receipt_sha256",
        "pruning_or_deletion_authorized",
        "qualification_frontier_tick_before",
        "receipt_directory",
        "schema_version",
        "selected_task_ids",
        "source_git_sha",
        "source_manifest_sha256",
        "target_tick",
        "unresolved_intent_blocks_new_attempt",
        "worker_count",
        "worker_execution_contract",
    }
    if set(payload) != expected_payload_keys:
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent payload schema is not exact"
        )
    if (
        payload.get("schema_version")
        != OPEN_ECOLOGY_CAMPAIGN_BARRIER_INTENT_SCHEMA_VERSION
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent payload version drifted"
        )
    expected_envelope_payload_sha256 = _sha256_bytes(_canonical_json_bytes(payload))
    if envelope.get("payload_sha256") != expected_envelope_payload_sha256:
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent envelope digest drifted"
        )
    unsigned_payload = dict(payload)
    observed_inner_digest = unsigned_payload.pop("intent_payload_sha256")
    if observed_inner_digest != _sha256_bytes(_canonical_json_bytes(unsigned_payload)):
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent payload digest drifted"
        )
    attempts = _sequence(
        payload.get("attempt_authorities"),
        field="barrier intent attempt_authorities",
    )
    if payload.get("attempt_authorities_sha256") != _sha256_bytes(
        _canonical_json_bytes(attempts)
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent attempt authority digest drifted"
        )
    selected_task_ids = [
        _identifier(task_id, field="barrier intent selected_task_ids[]")
        for task_id in _sequence(
            payload.get("selected_task_ids"),
            field="barrier intent selected_task_ids",
        )
    ]
    if len(set(selected_task_ids)) != len(selected_task_ids):
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent selected task identity is duplicated"
        )
    predecessor_states = _sequence(
        payload.get("predecessor_states"),
        field="barrier intent predecessor_states",
    )
    if len(predecessor_states) != len(selected_task_ids) or len(attempts) != len(
        selected_task_ids
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent predecessor/attempt matrix is incomplete"
        )
    target_tick = _frontier_tick(payload.get("target_tick"), allow_zero=False)
    campaign_id = _identifier(
        payload.get("campaign_id"),
        field="barrier intent campaign_id",
    )
    for task_id, raw_state, raw_attempt in zip(
        selected_task_ids,
        predecessor_states,
        attempts,
        strict=True,
    ):
        state = _mapping(raw_state, field=f"{task_id}.intent predecessor state")
        if (
            set(state)
            != {
                "aggregate_resume_pins",
                "observed_tick",
                "task_id",
                "terminal_extinct",
            }
            or state.get("task_id") != task_id
        ):
            raise OpenEcologyCampaignCoordinatorError(
                f"barrier intent predecessor state drifted for {task_id}"
            )
        observed_tick = _nonnegative_int(
            state.get("observed_tick"),
            field=f"{task_id}.intent predecessor observed_tick",
        )
        terminal = _boolean(
            state.get("terminal_extinct"),
            field=f"{task_id}.intent predecessor terminal_extinct",
        )
        pins_payload = state.get("aggregate_resume_pins")
        if pins_payload is None:
            if observed_tick != 0 or terminal:
                raise OpenEcologyCampaignCoordinatorError(
                    f"barrier intent pending predecessor drifted for {task_id}"
                )
            predecessor_commit = "genesis"
            run_generation_id = _persistent_run_generation_id(
                campaign_id=campaign_id,
                task_id=task_id,
            )
        else:
            pins = _resume_pins_from_dict(
                _mapping(
                    pins_payload,
                    field=f"{task_id}.intent predecessor pins",
                )
            )
            if (
                pins.identity.island_id != task_id
                or pins.identity.tick != observed_tick - 1
                or (
                    not terminal
                    and observed_tick + OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS
                    != target_tick
                )
                or (terminal and observed_tick > target_tick)
            ):
                raise OpenEcologyCampaignCoordinatorError(
                    f"barrier intent predecessor pins drifted for {task_id}"
                )
            predecessor_commit = pins.commit_sha256
            run_generation_id = pins.identity.run_generation_id
        attempt = _mapping(
            raw_attempt,
            field=f"{task_id}.intent attempt authority",
        )
        if set(attempt) != {
            "attempt_authority_sha256",
            "predecessor_aggregate_commit_sha256",
            "requested_target_tick",
            "run_generation_id",
            "task_id",
        }:
            raise OpenEcologyCampaignCoordinatorError(
                f"barrier intent attempt schema drifted for {task_id}"
            )
        if (
            attempt.get("task_id") != task_id
            or attempt.get("run_generation_id") != run_generation_id
            or attempt.get("predecessor_aggregate_commit_sha256") != predecessor_commit
            or attempt.get("requested_target_tick") != target_tick
            or attempt.get("attempt_authority_sha256")
            != persistent_interval_attempt_authority_payload_sha256(
                task_id=task_id,
                run_generation_id=run_generation_id,
                predecessor_aggregate_commit_sha256=predecessor_commit,
                requested_target_tick=target_tick,
            )
        ):
            raise OpenEcologyCampaignCoordinatorError(
                f"barrier intent attempt authority drifted for {task_id}"
            )
    if (
        payload.get("no_task_mutation_precedes_this_intent") is not True
        or payload.get("unresolved_intent_blocks_new_attempt") is not True
        or payload.get("pruning_or_deletion_authorized") is not False
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "barrier intent lifecycle flags drifted"
        )
    return (
        payload,
        _sha256_bytes(encoded),
        expected_envelope_payload_sha256,
    )


def _load_barrier_failure_observation(
    path: Path,
    *,
    expected_intent_path: Path,
    expected_intent_sha256: str,
    campaign_id: str,
    source_git_sha: str,
    matrix_sha256: str,
) -> dict[str, object]:
    if not path.is_absolute() or not expected_intent_path.is_absolute():
        raise OpenEcologyCampaignCoordinatorError(
            "barrier failure paths must be absolute"
        )
    encoded = _read_bounded_authority_file(
        path,
        max_bytes=OPEN_ECOLOGY_CAMPAIGN_MAX_BARRIER_INTENT_BYTES,
        field="barrier failure observation",
        forbid_group_or_other_write=True,
    )
    payload = _strict_json_object(encoded, field="barrier failure observation")
    expected_keys = {
        "campaign_id",
        "error_message",
        "error_type",
        "failure_observation_may_be_incomplete",
        "failure_observation_sha256",
        "intent_path",
        "intent_sha256",
        "matrix_sha256",
        "new_attempt_authorized",
        "partial_process_results",
        "pruning_or_deletion_authorized",
        "receipt_published",
        "schema_version",
        "source_git_sha",
        "visible_parent_runner_states",
    }
    if set(payload) != expected_keys:
        raise OpenEcologyCampaignCoordinatorError(
            "barrier failure observation schema is not exact"
        )
    if (
        payload.get("schema_version")
        != OPEN_ECOLOGY_CAMPAIGN_BARRIER_FAILURE_SCHEMA_VERSION
        or payload.get("campaign_id") != campaign_id
        or payload.get("source_git_sha") != source_git_sha
        or payload.get("matrix_sha256") != matrix_sha256
        or payload.get("intent_path") != str(expected_intent_path)
        or payload.get("intent_sha256") != expected_intent_sha256
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "barrier failure observation identity drifted"
        )
    for field in ("error_type", "error_message"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise OpenEcologyCampaignCoordinatorError(
                f"barrier failure observation {field} is invalid"
            )
    _sequence(
        payload.get("partial_process_results"),
        field="barrier failure partial_process_results",
    )
    _sequence(
        payload.get("visible_parent_runner_states"),
        field="barrier failure visible_parent_runner_states",
    )
    if (
        payload.get("failure_observation_may_be_incomplete") is not True
        or payload.get("receipt_published") is not False
        or payload.get("new_attempt_authorized") is not False
        or payload.get("pruning_or_deletion_authorized") is not False
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "barrier failure observation lifecycle flags drifted"
        )
    unsigned = dict(payload)
    observed_digest = unsigned.pop("failure_observation_sha256")
    failure_observation_sha256 = _sha256(
        observed_digest,
        field="barrier failure observation SHA256",
    )
    if failure_observation_sha256 != _sha256_bytes(_canonical_json_bytes(unsigned)):
        raise OpenEcologyCampaignCoordinatorError(
            "barrier failure observation digest drifted"
        )
    return {
        "path": str(path),
        "failure_observation_sha256": failure_observation_sha256,
        "file_sha256": _sha256_bytes(encoded),
    }


def restore_campaign_runners_from_receipt(
    *,
    tasks: Sequence[PersistentIslandTask],
    receipt: CampaignFrontierReceipt,
    campaign_root: str | Path,
    campaign_id: str,
) -> dict[str, PersistentIslandRunner]:
    """Restore every started runner only from exact externally retained pins."""

    _validate_exact_task_matrix(tasks)
    if not receipt.external_sha256_verified:
        raise OpenEcologyCampaignCoordinatorError(
            "resume requires an externally retained frontier receipt SHA256"
        )
    root = _existing_absolute_directory(campaign_root, field="campaign_root")
    if receipt.payload.get("campaign_root") != str(root):
        raise OpenEcologyCampaignCoordinatorError(
            "resume receipt campaign root drifted"
        )
    if receipt.payload.get("campaign_id") != campaign_id:
        raise OpenEcologyCampaignCoordinatorError("resume receipt campaign id drifted")
    pins_by_task = receipt.resume_pins_by_task()
    runners: dict[str, PersistentIslandRunner] = {}
    for task in tasks:
        if task.task_id in pins_by_task:
            runners[task.task_id] = PersistentIslandRunner.restore_from_current(
                task,
                campaign_root=root,
                campaign_id=campaign_id,
                pins=pins_by_task[task.task_id],
                evidence_directory=root / task.task_id,
                aggregate_directory=root / "aggregates" / task.task_id,
            )
    return runners


def _frontier_record(
    task: PersistentIslandTask,
    runner: CampaignRunner,
    *,
    expected_tick: int,
    barrier_target_tick: int,
    campaign_id: str,
    campaign_root: Path,
    source_git_sha: str,
) -> dict[str, object]:
    observed_tick = _nonnegative_int(
        runner.observed_tick,
        field=f"{task.task_id}.observed_tick",
    )
    if runner.extinct:
        if observed_tick > expected_tick:
            raise OpenEcologyCampaignCoordinatorError(
                f"terminal task {task.task_id} is ahead of the frontier"
            )
    elif observed_tick != expected_tick:
        raise OpenEcologyCampaignCoordinatorError(
            f"live task {task.task_id} is stale at the frontier"
        )
    quiescent = _mapping(
        runner.quiescent_checkpoint,
        field=f"{task.task_id}.quiescent_checkpoint",
    )
    expected_quiescent = {
        "task_id": task.task_id,
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "source_git_sha": source_git_sha,
        "observed_tick": observed_tick,
        "archive_safe_while_runner_idle": True,
        "restartable_world_checkpoint_present": True,
        "aggregate_current_restartable": True,
        "aggregate_resume_authorized": True,
    }
    for key, value in expected_quiescent.items():
        if quiescent.get(key) != value:
            raise OpenEcologyCampaignCoordinatorError(
                f"task {task.task_id} quiescent {key} is not authoritative"
            )
    pins = runner.latest_aggregate_resume_pins
    if pins is None:
        raise OpenEcologyCampaignCoordinatorError(
            f"task {task.task_id} has no aggregate CURRENT pins"
        )
    if (
        pins.identity.source_git_sha != source_git_sha
        or pins.identity.island_id != task.task_id
        or observed_tick <= 0
        or pins.identity.tick != observed_tick - 1
    ):
        raise OpenEcologyCampaignCoordinatorError(
            f"task {task.task_id} aggregate identity drifted"
        )
    record: dict[str, object] = {
        "task_id": task.task_id,
        "learner_index": task.learner_index,
        "learner_seed": task.learner_seed,
        "island_index": task.island_index,
        "environment_seed": task.environment_seed,
        "genome_stream_seed": task.genome_stream_seed,
        "arm": task.arm,
        "world_config_sha256": task.world_config_sha256,
        "task_sha256": _sha256_bytes(_canonical_json_bytes(task.to_dict())),
        "task_state": "terminal_extinct" if runner.extinct else "active",
        "frontier_observation_mode": "live_quiescent",
        "requested_barrier_target_tick": barrier_target_tick,
        "expected_task_tick": expected_tick,
        "observed_tick": observed_tick,
        "terminal_extinct": runner.extinct,
        "aggregate_resume_pins": _resume_pins_to_dict(pins),
        "quiescent_boundary_sha256": _sha256(
            quiescent.get("boundary_sha256"),
            field=f"{task.task_id}.quiescent boundary_sha256",
        ),
    }
    record["frontier_record_sha256"] = _sha256_bytes(_canonical_json_bytes(record))
    return record


def _pending_frontier_record(
    task: PersistentIslandTask,
) -> dict[str, object]:
    record: dict[str, object] = {
        "task_id": task.task_id,
        "learner_index": task.learner_index,
        "learner_seed": task.learner_seed,
        "island_index": task.island_index,
        "environment_seed": task.environment_seed,
        "genome_stream_seed": task.genome_stream_seed,
        "arm": task.arm,
        "world_config_sha256": task.world_config_sha256,
        "task_sha256": _sha256_bytes(_canonical_json_bytes(task.to_dict())),
        "task_state": "pending_unstarted",
        "frontier_observation_mode": "pending_unstarted",
        "requested_barrier_target_tick": None,
        "expected_task_tick": 0,
        "observed_tick": 0,
        "terminal_extinct": False,
        "aggregate_resume_pins": None,
        "quiescent_boundary_sha256": None,
    }
    record["frontier_record_sha256"] = _sha256_bytes(_canonical_json_bytes(record))
    return record


def _validate_retained_task_record(
    task: PersistentIslandTask,
    record: Mapping[str, object],
    *,
    source_git_sha: str,
) -> None:
    exact_keys = {
        "task_id",
        "learner_index",
        "learner_seed",
        "island_index",
        "environment_seed",
        "genome_stream_seed",
        "arm",
        "world_config_sha256",
        "task_sha256",
        "task_state",
        "frontier_observation_mode",
        "requested_barrier_target_tick",
        "expected_task_tick",
        "observed_tick",
        "terminal_extinct",
        "aggregate_resume_pins",
        "quiescent_boundary_sha256",
        "frontier_record_sha256",
    }
    if set(record) != exact_keys:
        raise OpenEcologyCampaignCoordinatorError(
            f"retained task record keys drifted for {task.task_id}"
        )
    expected_binding = {
        "task_id": task.task_id,
        "learner_index": task.learner_index,
        "learner_seed": task.learner_seed,
        "island_index": task.island_index,
        "environment_seed": task.environment_seed,
        "genome_stream_seed": task.genome_stream_seed,
        "arm": task.arm,
        "world_config_sha256": task.world_config_sha256,
        "task_sha256": _sha256_bytes(_canonical_json_bytes(task.to_dict())),
    }
    for key, value in expected_binding.items():
        if record.get(key) != value:
            raise OpenEcologyCampaignCoordinatorError(
                f"retained task binding {key} drifted for {task.task_id}"
            )
    recorded_digest = _sha256(
        record.get("frontier_record_sha256"),
        field=f"{task.task_id}.frontier_record_sha256",
    )
    unsigned = dict(record)
    unsigned.pop("frontier_record_sha256")
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != recorded_digest:
        raise OpenEcologyCampaignCoordinatorError(
            f"retained task record digest drifted for {task.task_id}"
        )
    state = record.get("task_state")
    if state == "pending_unstarted":
        if dict(record) != _pending_frontier_record(task):
            raise OpenEcologyCampaignCoordinatorError(
                f"retained pending task record drifted for {task.task_id}"
            )
        return
    if state not in {"active", "terminal_extinct"}:
        raise OpenEcologyCampaignCoordinatorError(
            f"retained task state is unknown for {task.task_id}"
        )
    if record.get("frontier_observation_mode") not in {
        "live_quiescent",
        "retained_unmaterialized_worker",
    }:
        raise OpenEcologyCampaignCoordinatorError(
            f"retained task observation mode drifted for {task.task_id}"
        )
    expected_tick = _positive_int(
        record.get("expected_task_tick"),
        field=f"{task.task_id}.expected_task_tick",
    )
    observed_tick = _positive_int(
        record.get("observed_tick"),
        field=f"{task.task_id}.observed_tick",
    )
    terminal = _boolean(
        record.get("terminal_extinct"),
        field=f"{task.task_id}.terminal_extinct",
    )
    if terminal is not (state == "terminal_extinct"):
        raise OpenEcologyCampaignCoordinatorError(
            f"retained terminal state drifted for {task.task_id}"
        )
    if (terminal and observed_tick > expected_tick) or (
        not terminal and observed_tick != expected_tick
    ):
        raise OpenEcologyCampaignCoordinatorError(
            f"retained task frontier tick drifted for {task.task_id}"
        )
    pins = _resume_pins_from_dict(
        _mapping(
            record.get("aggregate_resume_pins"),
            field=f"{task.task_id}.aggregate_resume_pins",
        )
    )
    if (
        pins.identity.source_git_sha != source_git_sha
        or pins.identity.island_id != task.task_id
        or pins.identity.tick != observed_tick - 1
    ):
        raise OpenEcologyCampaignCoordinatorError(
            f"retained task aggregate identity drifted for {task.task_id}"
        )
    _sha256(
        record.get("quiescent_boundary_sha256"),
        field=f"{task.task_id}.quiescent_boundary_sha256",
    )


def _runner_barrier_frontier(
    runner: CampaignRunner,
) -> Mapping[str, object]:
    try:
        frontier = runner.campaign_barrier_frontier()
    except (AttributeError, TypeError) as error:
        raise OpenEcologyCampaignCoordinatorError(
            "runner does not expose a callable campaign barrier frontier"
        ) from error
    return _mapping(frontier, field="runner campaign barrier frontier")


def _storage_scan_payload(
    scan: StorageScan,
    *,
    validation_mode: str | None = None,
    predecessor_entry_manifest_sha256: str | None = None,
) -> dict[str, object]:
    entry_records = [
        {
            **entry.manifest_record(),
            "device": entry.device,
            "inode": entry.inode,
            "link_count": entry.link_count,
            "mtime_ns": entry.mtime_ns,
        }
        for entry in scan.entries
    ]
    payload: dict[str, object] = {
        "root": str(scan.root),
        "total_file_bytes": scan.total_file_bytes,
        "free_bytes": scan.free_bytes,
        "filesystem_total_bytes": scan.filesystem_total_bytes,
        "file_count": scan.file_count,
        "directory_count": scan.directory_count,
        "entry_manifest_sha256": _sha256_bytes(_canonical_json_bytes(entry_records)),
        "scan_complete": True,
        "active_root_archived_or_pruned": False,
    }
    if validation_mode is not None:
        payload["validation_mode"] = validation_mode
        payload["predecessor_entry_manifest_sha256"] = predecessor_entry_manifest_sha256
    return payload


def _assert_storage_scan_matches_receipt(
    scan: StorageScan,
    receipt: CampaignFrontierReceipt,
) -> None:
    expected_storage = _mapping(
        receipt.payload.get("storage"),
        field="retained frontier storage",
    )
    current_storage = _storage_scan_payload(scan)
    for key in (
        "root",
        "total_file_bytes",
        "file_count",
        "directory_count",
        "entry_manifest_sha256",
    ):
        if current_storage[key] != expected_storage.get(key):
            raise OpenEcologyCampaignCoordinatorError(
                "retained frontier storage receipt is stale"
            )


def _process_worker_batch_summary(
    batch: ProcessWorkerFrontierBatch,
) -> dict[str, object]:
    return {
        "schema_version": batch.schema_version,
        "target_tick": batch.target_tick,
        "selected_task_ids": list(batch.selected_task_ids),
        "result_sha256_by_task": {
            task_id: batch.results_by_task[task_id].result_sha256
            for task_id in batch.selected_task_ids
        },
        "worker_statuses": [status.to_dict() for status in batch.worker_statuses],
        "assignment_sha256": batch.assignment_sha256,
        "batch_sha256": batch.batch_sha256,
    }


def _process_worker_reconciliation_batch_summary(
    batch: ProcessWorkerReconciliationBatch,
) -> dict[str, object]:
    summary = _process_worker_batch_summary(batch.frontier_batch)
    summary.update(
        {
            "reconciliation_schema_version": batch.schema_version,
            "intent_sha256": batch.intent_sha256,
            "attempt_authority_sha256_by_task": dict(
                batch.attempt_authority_sha256_by_task
            ),
            "disposition_by_task": dict(batch.disposition_by_task),
            "reconciliation_sha256": batch.reconciliation_sha256,
        }
    )
    return summary


def _resume_pins_to_dict(
    pins: OpenEcologyAggregateResumePins,
) -> dict[str, object]:
    return {
        "identity": {
            "source_git_sha": pins.identity.source_git_sha,
            "config_contract_sha256": pins.identity.config_contract_sha256,
            "seed_contract_sha256": pins.identity.seed_contract_sha256,
            "run_generation_id": pins.identity.run_generation_id,
            "island_id": pins.identity.island_id,
            "simulation_generation_index": (pins.identity.simulation_generation_index),
            "tick": pins.identity.tick,
        },
        "aggregate_generation_index": pins.aggregate_generation_index,
        "commit_sha256": pins.commit_sha256,
        "checkpoint_sha256": pins.checkpoint_sha256,
        "checkpoint_generation_identity_sha256": (
            pins.checkpoint_generation_identity_sha256
        ),
        "evidence_manifest_sha256": pins.evidence_manifest_sha256,
        "evidence_manifest_status": pins.evidence_manifest_status,
    }


def _is_exact_genesis_pins(
    pins: OpenEcologyAggregateResumePins,
    *,
    task: PersistentIslandTask,
    campaign_id: str,
    frontier: Mapping[str, object],
) -> bool:
    identity = pins.identity
    return bool(
        identity.source_git_sha == task.artifact.source_commit
        and identity.run_generation_id
        == _persistent_run_generation_id(
            campaign_id=campaign_id,
            task_id=task.task_id,
        )
        and identity.island_id == task.task_id
        and identity.simulation_generation_index == 0
        and identity.tick == 0
        and pins.aggregate_generation_index == 0
        and pins.evidence_manifest_status == "open"
        and frontier.get("task_id") == task.task_id
        and frontier.get("task_sha256")
        == _sha256_bytes(_canonical_json_bytes(task.to_dict()))
        and frontier.get("observed_tick") == 0
        and frontier.get("config_contract_sha256") == identity.config_contract_sha256
        and frontier.get("seed_contract_sha256") == identity.seed_contract_sha256
        and frontier.get("aggregate_generation_index") == 0
        and frontier.get("aggregate_completed_world_tick") == 0
        and frontier.get("aggregate_commit_sha256") == pins.commit_sha256
        and frontier.get("checkpoint_sha256") == pins.checkpoint_sha256
    )


def _persistent_run_generation_id(*, campaign_id: str, task_id: str) -> str:
    campaign_digest = hashlib.sha256(campaign_id.encode("utf-8")).hexdigest()[:16]
    return f"phase-d:{task_id}:{campaign_digest}"


def _resume_pins_from_dict(
    payload: Mapping[str, object],
) -> OpenEcologyAggregateResumePins:
    identity = _mapping(payload.get("identity"), field="resume pins identity")
    return OpenEcologyAggregateResumePins(
        identity=OpenEcologyAggregateIdentityPins(
            source_git_sha=_git_sha(
                identity.get("source_git_sha"),
                field="resume identity source_git_sha",
            ),
            config_contract_sha256=_sha256(
                identity.get("config_contract_sha256"),
                field="resume identity config_contract_sha256",
            ),
            seed_contract_sha256=_sha256(
                identity.get("seed_contract_sha256"),
                field="resume identity seed_contract_sha256",
            ),
            run_generation_id=_identifier(
                identity.get("run_generation_id"),
                field="resume identity run_generation_id",
            ),
            island_id=_identifier(
                identity.get("island_id"),
                field="resume identity island_id",
            ),
            simulation_generation_index=_nonnegative_int(
                identity.get("simulation_generation_index"),
                field="resume identity simulation_generation_index",
            ),
            tick=_nonnegative_int(
                identity.get("tick"),
                field="resume identity tick",
            ),
        ),
        aggregate_generation_index=_nonnegative_int(
            payload.get("aggregate_generation_index"),
            field="resume aggregate_generation_index",
        ),
        commit_sha256=_sha256(
            payload.get("commit_sha256"),
            field="resume commit_sha256",
        ),
        checkpoint_sha256=_sha256(
            payload.get("checkpoint_sha256"),
            field="resume checkpoint_sha256",
        ),
        checkpoint_generation_identity_sha256=_sha256(
            payload.get("checkpoint_generation_identity_sha256"),
            field="resume checkpoint_generation_identity_sha256",
        ),
        evidence_manifest_sha256=_sha256(
            payload.get("evidence_manifest_sha256"),
            field="resume evidence_manifest_sha256",
        ),
        evidence_manifest_status=str(payload.get("evidence_manifest_status")),
    )


def _validate_exact_task_matrix(tasks: Sequence[PersistentIslandTask]) -> None:
    if len(tasks) != OPEN_ECOLOGY_PHASE_D_TASK_COUNT:
        raise OpenEcologyCampaignCoordinatorError(
            "campaign requires the exact 48-task matrix"
        )
    expected: list[tuple[int, int, str]] = []
    for learner_index in range(OPEN_ECOLOGY_PHASE_D_LEARNER_COUNT):
        for island_index in range(OPEN_ECOLOGY_PHASE_D_ISLAND_COUNT):
            for arm in OPEN_ECOLOGY_PHASE_D_ARM_ORDER:
                expected.append((learner_index, island_index, arm))
    observed = [(task.learner_index, task.island_index, task.arm) for task in tasks]
    if observed != expected or len({task.task_id for task in tasks}) != len(tasks):
        raise OpenEcologyCampaignCoordinatorError(
            "campaign task order or identity drifted from learner/island/H-Z-R"
        )
    densities = {task.selected_density for task in tasks}
    targets = {task.target_ticks for task in tasks}
    if len(densities) != 1 or targets != {OPEN_ECOLOGY_PHASE_D_TARGET_TICKS}:
        raise OpenEcologyCampaignCoordinatorError(
            "campaign tasks do not share one sealed density and 50,000-tick target"
        )
    prioritized_persistent_island_triplet(tasks)


def _validated_receipt_frontier_state(
    payload: Mapping[str, object],
    tasks: Sequence[PersistentIslandTask],
) -> tuple[dict[str, int], list[str]]:
    scope = payload.get("barrier_scope")
    frontier_tick = _frontier_tick(payload.get("frontier_tick"), allow_zero=False)
    qualification_tick = _frontier_tick(
        payload.get("qualification_frontier_tick"),
        allow_zero=True,
    )
    matrix_tick = _frontier_tick(
        payload.get("matrix_frontier_tick"),
        allow_zero=True,
    )
    priority_ids = [
        task.task_id for task in prioritized_persistent_island_triplet(tasks)
    ]
    priority_set = set(priority_ids)
    all_ids = [task.task_id for task in tasks]
    if scope == "qualification_triplet":
        valid = (
            frontier_tick in {5_000, 10_000}
            and qualification_tick == frontier_tick
            and matrix_tick == 0
        )
        expected_ticks = {
            task_id: (frontier_tick if task_id in priority_set else 0)
            for task_id in all_ids
        }
        selected_ids = priority_ids
        expected_sequence = frontier_tick // 5_000
    elif scope == "matrix_catchup":
        valid = (
            frontier_tick in {5_000, 10_000}
            and qualification_tick == OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
            and matrix_tick == frontier_tick
        )
        expected_ticks = {
            task_id: (
                OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
                if task_id in priority_set
                else frontier_tick
            )
            for task_id in all_ids
        }
        selected_ids = [task_id for task_id in all_ids if task_id not in priority_set]
        expected_sequence = 2 + frontier_tick // 5_000
    elif scope == "full_matrix":
        valid = (
            frontier_tick > OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
            and qualification_tick == OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
            and matrix_tick == frontier_tick
        )
        expected_ticks = {task_id: frontier_tick for task_id in all_ids}
        selected_ids = all_ids
        expected_sequence = (
            4 + (frontier_tick - OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK) // 5_000
        )
    else:
        raise OpenEcologyCampaignCoordinatorError(
            "retained frontier barrier scope is unknown"
        )
    expected_common = scope == "full_matrix" or (
        scope == "matrix_catchup"
        and frontier_tick == OPEN_ECOLOGY_CAMPAIGN_QUALIFICATION_TICK
    )
    if (
        not valid
        or payload.get("coordination_sequence_index") != expected_sequence
        or payload.get("all_tasks_at_common_frontier") is not expected_common
        or payload.get("next_interval_authorized")
        is not (matrix_tick < OPEN_ECOLOGY_PHASE_D_TARGET_TICKS)
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "retained frontier stage semantics drifted"
        )
    return expected_ticks, selected_ids


def _default_health_probe(phase: str, frontier_tick: int) -> dict[str, object]:
    load_average = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    return {
        "phase": phase,
        "frontier_tick": frontier_tick,
        "process_id": os.getpid(),
        "load_average_1m": load_average[0],
        "load_average_5m": load_average[1],
        "load_average_15m": load_average[2],
        "hook_status": "host_basic_only_external_gpu_health_hook_may_extend",
    }


class _CoordinatorOutputLock(AbstractContextManager["_CoordinatorOutputLock"]):
    def __init__(
        self,
        path: Path,
        *,
        campaign_id: str,
        source_git_sha: str,
        matrix_sha256: str,
    ) -> None:
        self.path = path
        self.identity = {
            "schema_version": OPEN_ECOLOGY_CAMPAIGN_OUTPUT_LOCK_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "source_git_sha": source_git_sha,
            "matrix_sha256": matrix_sha256,
        }
        self._descriptor: int | None = None

    def __enter__(self) -> _CoordinatorOutputLock:
        if not self.path.is_absolute() or self.path.is_symlink():
            raise OpenEcologyCampaignCoordinatorError(
                "coordinator output lock path is unsafe"
            )
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > _OUTPUT_LOCK_MAX_BYTES
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise OpenEcologyCampaignCoordinatorError(
                    "coordinator output lock is not private and bounded"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise OpenEcologyCampaignCoordinatorError(
                    "campaign coordinator output is locked by another process"
                ) from error
            expected = _canonical_json_bytes(self.identity)
            os.lseek(descriptor, 0, os.SEEK_SET)
            existing = os.read(descriptor, _OUTPUT_LOCK_MAX_BYTES + 1)
            if existing and existing != expected:
                raise OpenEcologyCampaignCoordinatorError(
                    "coordinator output lock identity drifted"
                )
            if not existing:
                os.lseek(descriptor, 0, os.SEEK_SET)
                _write_all(descriptor, expected)
                os.ftruncate(descriptor, len(expected))
                os.fsync(descriptor)
            self._descriptor = descriptor
            return self
        except BaseException:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(descriptor)
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._descriptor is None:
            return
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._descriptor = None


def _prepare_receipt_directory(
    value: str | Path,
    *,
    campaign_root: Path,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise OpenEcologyCampaignCoordinatorError("receipt_directory must be absolute")
    try:
        normalized = _normalize_storage_real_tree_path(
            path,
            field="receipt_directory",
        )
    except CampaignStorageError as error:
        raise OpenEcologyCampaignCoordinatorError(
            f"receipt_directory is unsafe: {error}"
        ) from error
    if normalized == campaign_root or campaign_root in normalized.parents:
        raise OpenEcologyCampaignCoordinatorError(
            "frontier receipts must be outside the scanned active campaign root"
        )
    if normalized in campaign_root.parents:
        raise OpenEcologyCampaignCoordinatorError(
            "receipt directory may not contain the active campaign root"
        )
    try:
        concrete = ensure_real_directory_tree(
            normalized,
            field="receipt_directory",
            mode=0o700,
        )
    except CampaignStorageError as error:
        raise OpenEcologyCampaignCoordinatorError(
            f"receipt_directory is unsafe: {error}"
        ) from error
    try:
        _, receipt_descriptor = _open_storage_real_directory_tree(
            concrete,
            field="receipt_directory",
            create=False,
            mode=0o700,
        )
    except CampaignStorageError as error:
        raise OpenEcologyCampaignCoordinatorError(
            f"receipt_directory is unsafe: {error}"
        ) from error
    try:
        receipt_metadata = os.fstat(receipt_descriptor)
    finally:
        os.close(receipt_descriptor)
    if stat.S_IMODE(receipt_metadata.st_mode) & 0o077:
        raise OpenEcologyCampaignCoordinatorError(
            "receipt_directory must not grant group or other permissions"
        )
    try:
        concrete.relative_to(campaign_root)
    except ValueError:
        pass
    else:
        raise OpenEcologyCampaignCoordinatorError(
            "frontier receipts must be outside the scanned active campaign root"
        )
    try:
        campaign_root.relative_to(concrete)
    except ValueError:
        pass
    else:
        raise OpenEcologyCampaignCoordinatorError(
            "receipt directory may not contain the active campaign root"
        )
    return concrete


def _existing_absolute_directory(value: str | Path, *, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise OpenEcologyCampaignCoordinatorError(f"{field} must be absolute")
    try:
        concrete = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise OpenEcologyCampaignCoordinatorError(f"{field} does not exist") from error
    if concrete != path or not concrete.is_dir() or concrete.is_symlink():
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} must be one canonical existing directory"
        )
    return concrete


def _frontier_tick(value: object, *, allow_zero: bool) -> int:
    tick = _nonnegative_int(value, field="frontier tick")
    if (
        (tick == 0 and not allow_zero)
        or tick > OPEN_ECOLOGY_PHASE_D_TARGET_TICKS
        or tick % OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS != 0
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "frontier tick must be a 5,000-tick multiple within the campaign"
        )
    return tick


def _atomic_create(path: Path, payload: bytes) -> None:
    destination = Path(path)
    if not destination.is_absolute():
        raise OpenEcologyCampaignCoordinatorError(
            "immutable frontier evidence path must be absolute"
        )
    try:
        parent, parent_descriptor = _open_storage_real_directory_tree(
            destination.parent,
            field="frontier evidence parent",
            create=False,
            mode=0o700,
        )
    except CampaignStorageError as error:
        raise OpenEcologyCampaignCoordinatorError(
            f"frontier evidence parent is unsafe: {error}"
        ) from error
    destination = parent / destination.name
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    stage_name: str | None = None
    descriptor: int | None = None
    try:
        for _attempt in range(32):
            candidate = (
                f".{destination.name}.pending-{os.getpid()}-{secrets.token_hex(16)}"
            )
            try:
                descriptor = os.open(
                    candidate,
                    flags,
                    0o444,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            except OSError as error:
                raise OpenEcologyCampaignCoordinatorError(
                    "cannot create an immutable frontier publication stage"
                ) from error
            stage_name = candidate
            break
        else:
            raise OpenEcologyCampaignCoordinatorError(
                "cannot allocate an immutable frontier publication stage"
            )

        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        try:
            named_stage = os.stat(
                stage_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise OpenEcologyCampaignCoordinatorError(
                "frontier publication stage disappeared before publication"
            ) from error
        stage_bytes = _read_exact_descriptor(
            descriptor,
            expected_bytes=len(payload),
        )
        finished_stage = os.fstat(descriptor)
        if (
            not stat.S_ISREG(staged.st_mode)
            or staged.st_nlink != 1
            or staged.st_size != len(payload)
            or stat.S_IMODE(staged.st_mode) != 0o444
            or not _same_file_identity(staged, named_stage)
            or not _same_open_file_state(staged, finished_stage)
            or stage_bytes != payload
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "frontier publication stage identity or bytes changed"
            )
        try:
            _rename_name_no_replace(
                parent_descriptor,
                stage_name,
                destination.name,
            )
        except FileExistsError as error:
            raise OpenEcologyCampaignCoordinatorError(
                "refusing to overwrite immutable frontier evidence"
            ) from error
        except OSError as error:
            raise OpenEcologyCampaignCoordinatorError(
                "cannot publish immutable frontier evidence"
            ) from error
        try:
            published = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise OpenEcologyCampaignCoordinatorError(
                "immutable frontier evidence disappeared during publication"
            ) from error
        if not _same_file_identity(staged, published):
            raise OpenEcologyCampaignCoordinatorError(
                "immutable frontier evidence identity changed during publication"
            )
        _verify_published_file_at(
            parent_descriptor,
            destination.name,
            staged,
            payload=payload,
        )
        os.fsync(parent_descriptor)
    finally:
        # Failed stages are evidence of a failed publication attempt.  Do not
        # unlink by pathname: an attacker may have replaced that name, and
        # POSIX provides no portable identity-conditional unlink primitive.
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _rename_path_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a same-directory path without replacing authority."""

    parent_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        source_parent, parent_descriptor = _open_storage_real_directory_tree(
            source.parent,
            field="exclusive frontier publication parent",
            create=False,
            mode=0o700,
        )
        destination_parent, destination_descriptor = _open_storage_real_directory_tree(
            destination.parent,
            field="exclusive frontier publication parent",
            create=False,
            mode=0o700,
        )
    except CampaignStorageError as error:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise OpenEcologyCampaignCoordinatorError(
            f"exclusive frontier publication parent is unsafe: {error}"
        ) from error
    try:
        assert parent_descriptor is not None
        assert destination_descriptor is not None
        destination_metadata = os.fstat(destination_descriptor)
        parent_metadata = os.fstat(parent_descriptor)
        if (
            source_parent != destination_parent
            or parent_metadata.st_dev != destination_metadata.st_dev
            or parent_metadata.st_ino != destination_metadata.st_ino
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "exclusive frontier publication requires one parent"
            )
        _rename_name_no_replace(
            parent_descriptor,
            source.name,
            destination.name,
        )
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _rename_name_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically publish a same-directory name without replacing authority."""

    if not source_name or not destination_name:
        raise OpenEcologyCampaignCoordinatorError(
            "exclusive frontier publication requires non-empty names"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(destination_name),
            0x00000004 | 0x00000010,
        )
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(destination_name),
            1,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "exclusive frontier publication is unsupported",
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                destination_name,
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and stat.S_IMODE(left.st_mode) == stat.S_IMODE(right.st_mode)
        and left.st_nlink == right.st_nlink
    )


def _same_open_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_file_identity(left, right)
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _read_exact_descriptor(descriptor: int, *, expected_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = expected_bytes + 1
    while remaining:
        chunk = os.read(
            descriptor,
            min(OPEN_ECOLOGY_CAMPAIGN_MAX_RECEIPT_BYTES, remaining),
        )
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _verify_published_file_at(
    parent_descriptor: int,
    name: str,
    expected_metadata: os.stat_result,
    *,
    payload: bytes,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise OpenEcologyCampaignCoordinatorError(
            "cannot read back immutable frontier evidence"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(
            expected_metadata, opened
        ):
            raise OpenEcologyCampaignCoordinatorError(
                "immutable frontier evidence identity changed during publication"
            )
        observed = _read_exact_descriptor(
            descriptor,
            expected_bytes=len(payload),
        )
        finished = os.fstat(descriptor)
        if observed != payload or not _same_open_file_state(opened, finished):
            raise OpenEcologyCampaignCoordinatorError(
                "immutable frontier evidence bytes changed during publication"
            )
        try:
            final_path = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise OpenEcologyCampaignCoordinatorError(
                "immutable frontier evidence disappeared after readback"
            ) from error
        if not _same_file_identity(finished, final_path):
            raise OpenEcologyCampaignCoordinatorError(
                "immutable frontier evidence path changed after readback"
            )
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            raise OpenEcologyCampaignCoordinatorError(
                "short write while creating frontier evidence"
            )
        written += count


def _strict_json_object(payload: bytes, *, field: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OpenEcologyCampaignCoordinatorError(
                    f"{field} has duplicate JSON keys"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} has non-finite JSON value {value}"
        )

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} is not strict JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise OpenEcologyCampaignCoordinatorError(f"{field} root must be an object")
    return decoded


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return canonical_json_bytes(payload)
    except (TypeError, ValueError) as error:
        raise OpenEcologyCampaignCoordinatorError(
            "campaign metadata is not canonical finite JSON"
        ) from error


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OpenEcologyCampaignCoordinatorError(f"{field} must be an object")
    return value


def _mapping_unlabelled(value: object) -> Mapping[str, object]:
    return _mapping(value, field="frontier task")


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise OpenEcologyCampaignCoordinatorError(f"{field} must be an array")
    return value


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} must be a conservative identifier"
        )
    return value


def _git_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} must be a full lowercase Git commit"
        )
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OpenEcologyCampaignCoordinatorError(f"{field} must be lowercase SHA256")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _positive_int(value: object, *, field: str) -> int:
    parsed = _nonnegative_int(value, field=field)
    if parsed <= 0:
        raise OpenEcologyCampaignCoordinatorError(f"{field} must be a positive integer")
    return parsed


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} must be a nonnegative integer"
        )
    return value


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise OpenEcologyCampaignCoordinatorError(f"{field} must be boolean")
    return value


__all__ = [
    "CampaignFrontierReceipt",
    "OpenEcologyCampaignCoordinator",
    "OpenEcologyCampaignCoordinatorError",
    "StaticWorkerAssignment",
    "build_static_worker_assignments",
    "load_campaign_frontier_receipt",
    "restore_campaign_runners_from_receipt",
]
