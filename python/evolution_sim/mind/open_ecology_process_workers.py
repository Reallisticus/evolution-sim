"""Bounded OS-process workers for persistent open-ecology island frontiers.

The parent and children exchange only canonical JSON messages.  Live worlds,
policies, Torch modules, writers, and locks never cross the process boundary.
Each child owns a deterministic static task queue, constructs a task on first
use (or restores it from externally retained aggregate pins), and reuses that
runner at later 5,000-tick barriers.

This module has no deletion or pruning operation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
import re
import selectors
import socket
import stat
import subprocess
import time
from types import MappingProxyType
from typing import NoReturn

from evolution_sim.io.open_ecology_aggregate_commit import (
    OpenEcologyAggregateIdentityPins,
    OpenEcologyAggregateResumePins,
)
from evolution_sim.io.open_ecology_bounded_subprocess import (
    OpenEcologyProcessGroupError,
    terminate_process_group_before_reap,
    wait_for_leader_exit_without_reaping,
)
from evolution_sim.io.open_ecology_campaign_storage import canonical_json_bytes
from evolution_sim.io.source_manifest import source_file_hash_manifest
from evolution_sim.mind.open_ecology_persistent_island import (
    OPEN_ECOLOGY_PHASE_D_ARM_ORDER,
    OPEN_ECOLOGY_PHASE_D_ISLAND_COUNT,
    OPEN_ECOLOGY_PHASE_D_LEARNER_COUNT,
    OPEN_ECOLOGY_PHASE_D_TASK_COUNT,
    OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS,
    PersistentArtifactBinding,
    PersistentIslandRunner,
    PersistentIslandTask,
)


OPEN_ECOLOGY_PROCESS_WORKER_LAUNCHER_CONTRACT = (
    "open_ecology_persistent_process_or_host_worker_queue_v1"
)
OPEN_ECOLOGY_PROCESS_WORKER_PROTOCOL_SCHEMA_VERSION = (
    "mind_v3_open_ecology_process_worker_protocol_v1"
)
OPEN_ECOLOGY_PROCESS_WORKER_ASSIGNMENT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_process_worker_assignment_v1"
)
OPEN_ECOLOGY_PROCESS_WORKER_RESULT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_process_worker_task_result_v1"
)
OPEN_ECOLOGY_PROCESS_WORKER_BATCH_SCHEMA_VERSION = (
    "mind_v3_open_ecology_process_worker_frontier_batch_v1"
)
OPEN_ECOLOGY_PROCESS_WORKER_RECONCILIATION_BATCH_SCHEMA_VERSION = (
    "mind_v3_open_ecology_process_worker_reconciliation_batch_v1"
)
OPEN_ECOLOGY_PROCESS_WORKER_RECONCILIATION_DISPOSITIONS = frozenset(
    {
        "advanced_from_exact_predecessor",
        "already_committed_target",
        "recovered_original_precurrent_attempt",
    }
)
OPEN_ECOLOGY_PROCESS_WORKER_MIN_MESSAGE_BYTES = 64 * 1024
OPEN_ECOLOGY_PROCESS_WORKER_DEFAULT_MAX_MESSAGE_BYTES = 8 * 1024 * 1024
OPEN_ECOLOGY_PROCESS_WORKER_MAX_MESSAGE_BYTES = 32 * 1024 * 1024
OPEN_ECOLOGY_PROCESS_WORKER_DEFAULT_TIMEOUT_SECONDS = 3_600.0
OPEN_ECOLOGY_PROCESS_WORKER_DEFAULT_STARTUP_TIMEOUT_SECONDS = 120.0
OPEN_ECOLOGY_PROCESS_WORKER_DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30.0
OPEN_ECOLOGY_PROCESS_WORKER_FATAL_MARKER_SCHEMA_VERSION = (
    "mind_v3_open_ecology_process_worker_fatal_marker_v1"
)
OPEN_ECOLOGY_PROCESS_WORKER_FATAL_MARKER_NAME = ".open-ecology-fatal-worker-pids.json"
OPEN_ECOLOGY_PROCESS_WORKER_MAX_FATAL_MARKER_BYTES = 64 * 1024
OPEN_ECOLOGY_GIT_EXECUTABLE_MAX_BYTES = 64 * 1024 * 1024
OPEN_ECOLOGY_GIT_OUTPUT_MAX_BYTES = 1024 * 1024
OPEN_ECOLOGY_FIXED_COMMAND_PATH = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_FRONTIER_INTERVAL = OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS


class OpenEcologyProcessWorkerError(RuntimeError):
    """The process-worker protocol could not authorize a complete frontier."""

    def __init__(
        self,
        message: str,
        *,
        partial_results: Mapping[str, ProcessWorkerTaskResult] | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_results = dict(partial_results or {})


@dataclass(frozen=True, slots=True)
class ProcessWorkerSlot:
    """One explicit local host/device/thread allocation and immutable queue."""

    worker_index: int
    host_identity: str
    device_kind: str
    device_index: int | None
    torch_threads: int
    task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonnegative_int(self.worker_index, field="slot.worker_index")
        _identifier(self.host_identity, field="slot.host_identity")
        if self.device_kind != "cpu":
            raise OpenEcologyProcessWorkerError(
                "persistent island workers are CPU-only; the runner explicitly "
                "copies its frozen policy to CPU, so a CUDA slot would hide "
                "oversubscription rather than accelerate this workload"
            )
        if self.device_index is not None:
            raise OpenEcologyProcessWorkerError(
                "CPU process worker device_index must be null"
            )
        _positive_int(self.torch_threads, field="slot.torch_threads")
        if not self.task_ids:
            raise OpenEcologyProcessWorkerError(
                "process worker slot must own a non-empty static queue"
            )
        normalized = tuple(
            _identifier(task_id, field="slot.task_ids[]") for task_id in self.task_ids
        )
        if normalized != self.task_ids or len(set(normalized)) != len(normalized):
            raise OpenEcologyProcessWorkerError(
                "process worker slot task queue is duplicated or non-canonical"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_index": self.worker_index,
            "host_identity": self.host_identity,
            "device_kind": self.device_kind,
            "device_index": self.device_index,
            "torch_threads": self.torch_threads,
            "task_ids": list(self.task_ids),
        }


@dataclass(frozen=True, slots=True)
class ProcessWorkerExpectedState:
    """Externally authoritative state from which one task may advance."""

    task_id: str
    observed_tick: int
    terminal_extinct: bool
    aggregate_resume_pins: OpenEcologyAggregateResumePins | None

    def __post_init__(self) -> None:
        _identifier(self.task_id, field="expected.task_id")
        observed_tick = _nonnegative_int(
            self.observed_tick,
            field="expected.observed_tick",
        )
        if not isinstance(self.terminal_extinct, bool):
            raise OpenEcologyProcessWorkerError(
                "expected.terminal_extinct must be boolean"
            )
        if observed_tick == 0:
            if self.terminal_extinct or self.aggregate_resume_pins is not None:
                raise OpenEcologyProcessWorkerError(
                    "pending task state must be tick zero, nonterminal, and unpinned"
                )
            return
        pins = self.aggregate_resume_pins
        if not isinstance(pins, OpenEcologyAggregateResumePins):
            raise OpenEcologyProcessWorkerError(
                "started task state requires exact external aggregate resume pins"
            )
        if (
            pins.identity.island_id != self.task_id
            or pins.identity.tick != observed_tick - 1
        ):
            raise OpenEcologyProcessWorkerError(
                "expected task tick and aggregate identity disagree"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "observed_tick": self.observed_tick,
            "terminal_extinct": self.terminal_extinct,
            "aggregate_resume_pins": (
                None
                if self.aggregate_resume_pins is None
                else _pins_to_dict(self.aggregate_resume_pins)
            ),
        }


@dataclass(frozen=True, slots=True)
class ProcessWorkerTaskResult:
    """Exact bounded operational result for one quiescent task frontier."""

    task_id: str
    worker_index: int
    worker_pid: int
    requested_target_tick: int
    observed_tick: int
    extinct: bool
    extinction_tick: int | None
    model_state_sha256: str
    same_world_instance: bool
    summary_count: int
    summary_sha256: str
    milestone_count: int
    milestone_sha256: str
    quiescent_checkpoint: Mapping[str, object]
    campaign_barrier_frontier: Mapping[str, object]
    aggregate_resume_pins: OpenEcologyAggregateResumePins
    result_sha256: str
    schema_version: str = OPEN_ECOLOGY_PROCESS_WORKER_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPEN_ECOLOGY_PROCESS_WORKER_RESULT_SCHEMA_VERSION:
            raise OpenEcologyProcessWorkerError("worker task result version drifted")
        unsigned = self.to_dict()
        unsigned.pop("result_sha256")
        if _stable_digest(unsigned) != _sha256(
            self.result_sha256,
            field="worker task result result_sha256",
        ):
            raise OpenEcologyProcessWorkerError("worker task result digest mismatch")
        object.__setattr__(
            self,
            "quiescent_checkpoint",
            MappingProxyType(_json_clone_mapping(self.quiescent_checkpoint)),
        )
        object.__setattr__(
            self,
            "campaign_barrier_frontier",
            MappingProxyType(_json_clone_mapping(self.campaign_barrier_frontier)),
        )

    @classmethod
    def from_dict(cls, raw_payload: Mapping[str, object]) -> ProcessWorkerTaskResult:
        payload = _mapping(raw_payload, field="worker task result")
        _exact_keys(
            payload,
            {
                "schema_version",
                "task_id",
                "worker_index",
                "worker_pid",
                "requested_target_tick",
                "observed_tick",
                "extinct",
                "extinction_tick",
                "model_state_sha256",
                "same_world_instance",
                "summary_count",
                "summary_sha256",
                "milestone_count",
                "milestone_sha256",
                "quiescent_checkpoint",
                "campaign_barrier_frontier",
                "aggregate_resume_pins",
                "result_sha256",
            },
            field="worker task result",
        )
        if (
            payload["schema_version"]
            != OPEN_ECOLOGY_PROCESS_WORKER_RESULT_SCHEMA_VERSION
        ):
            raise OpenEcologyProcessWorkerError("worker task result version drifted")
        result_sha256 = _sha256(
            payload["result_sha256"],
            field="worker task result result_sha256",
        )
        unsigned = dict(payload)
        unsigned.pop("result_sha256")
        if _stable_digest(unsigned) != result_sha256:
            raise OpenEcologyProcessWorkerError("worker task result digest mismatch")
        extinct = _boolean(payload["extinct"], field="worker task result extinct")
        same_world = _boolean(
            payload["same_world_instance"],
            field="worker task result same_world_instance",
        )
        extinction_tick = payload["extinction_tick"]
        if extinction_tick is not None:
            extinction_tick = _nonnegative_int(
                extinction_tick,
                field="worker task result extinction_tick",
            )
        pins = _pins_from_dict(
            _mapping(
                payload["aggregate_resume_pins"],
                field="worker task result aggregate_resume_pins",
            )
        )
        return cls(
            task_id=_identifier(
                payload["task_id"],
                field="worker task result task_id",
            ),
            worker_index=_nonnegative_int(
                payload["worker_index"],
                field="worker task result worker_index",
            ),
            worker_pid=_positive_int(
                payload["worker_pid"],
                field="worker task result worker_pid",
            ),
            requested_target_tick=_frontier_tick(
                payload["requested_target_tick"],
                field="worker task result requested_target_tick",
            ),
            observed_tick=_nonnegative_int(
                payload["observed_tick"],
                field="worker task result observed_tick",
            ),
            extinct=extinct,
            extinction_tick=extinction_tick,
            model_state_sha256=_sha256(
                payload["model_state_sha256"],
                field="worker task result model_state_sha256",
            ),
            same_world_instance=same_world,
            summary_count=_nonnegative_int(
                payload["summary_count"],
                field="worker task result summary_count",
            ),
            summary_sha256=_sha256(
                payload["summary_sha256"],
                field="worker task result summary_sha256",
            ),
            milestone_count=_nonnegative_int(
                payload["milestone_count"],
                field="worker task result milestone_count",
            ),
            milestone_sha256=_sha256(
                payload["milestone_sha256"],
                field="worker task result milestone_sha256",
            ),
            quiescent_checkpoint=_json_clone_mapping(
                _mapping(
                    payload["quiescent_checkpoint"],
                    field="worker task result quiescent_checkpoint",
                )
            ),
            campaign_barrier_frontier=_json_clone_mapping(
                _mapping(
                    payload["campaign_barrier_frontier"],
                    field="worker task result campaign_barrier_frontier",
                )
            ),
            aggregate_resume_pins=pins,
            result_sha256=result_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "worker_index": self.worker_index,
            "worker_pid": self.worker_pid,
            "requested_target_tick": self.requested_target_tick,
            "observed_tick": self.observed_tick,
            "extinct": self.extinct,
            "extinction_tick": self.extinction_tick,
            "model_state_sha256": self.model_state_sha256,
            "same_world_instance": self.same_world_instance,
            "summary_count": self.summary_count,
            "summary_sha256": self.summary_sha256,
            "milestone_count": self.milestone_count,
            "milestone_sha256": self.milestone_sha256,
            "quiescent_checkpoint": _json_clone_mapping(self.quiescent_checkpoint),
            "campaign_barrier_frontier": _json_clone_mapping(
                self.campaign_barrier_frontier
            ),
            "aggregate_resume_pins": _pins_to_dict(self.aggregate_resume_pins),
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProcessWorkerStatus:
    worker_index: int
    worker_pid: int
    host_identity: str
    device_kind: str
    device_index: int | None
    torch_threads: int
    source_before: Mapping[str, object]
    source_after: Mapping[str, object]
    completed_task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_before",
            MappingProxyType(_json_clone_mapping(self.source_before)),
        )
        object.__setattr__(
            self,
            "source_after",
            MappingProxyType(_json_clone_mapping(self.source_after)),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProcessWorkerStatus:
        _exact_keys(
            payload,
            {
                "worker_index",
                "worker_pid",
                "host_identity",
                "device_kind",
                "device_index",
                "torch_threads",
                "source_before",
                "source_after",
                "completed_task_ids",
            },
            field="worker status",
        )
        completed = _string_tuple(
            payload["completed_task_ids"],
            field="worker status completed_task_ids",
        )
        if len(set(completed)) != len(completed):
            raise OpenEcologyProcessWorkerError(
                "worker status repeats a completed task"
            )
        device_index = payload["device_index"]
        if device_index is not None:
            device_index = _nonnegative_int(
                device_index,
                field="worker status device_index",
            )
        return cls(
            worker_index=_nonnegative_int(
                payload["worker_index"],
                field="worker status worker_index",
            ),
            worker_pid=_positive_int(
                payload["worker_pid"],
                field="worker status worker_pid",
            ),
            host_identity=_identifier(
                payload["host_identity"],
                field="worker status host_identity",
            ),
            device_kind=_string(
                payload["device_kind"],
                field="worker status device_kind",
            ),
            device_index=device_index,
            torch_threads=_positive_int(
                payload["torch_threads"],
                field="worker status torch_threads",
            ),
            source_before=_json_clone_mapping(
                _mapping(payload["source_before"], field="worker status source_before")
            ),
            source_after=_json_clone_mapping(
                _mapping(payload["source_after"], field="worker status source_after")
            ),
            completed_task_ids=completed,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_index": self.worker_index,
            "worker_pid": self.worker_pid,
            "host_identity": self.host_identity,
            "device_kind": self.device_kind,
            "device_index": self.device_index,
            "torch_threads": self.torch_threads,
            "source_before": _json_clone_mapping(self.source_before),
            "source_after": _json_clone_mapping(self.source_after),
            "completed_task_ids": list(self.completed_task_ids),
        }


@dataclass(frozen=True, slots=True)
class ProcessWorkerFrontierBatch:
    target_tick: int
    selected_task_ids: tuple[str, ...]
    results_by_task: Mapping[str, ProcessWorkerTaskResult]
    worker_statuses: tuple[ProcessWorkerStatus, ...]
    assignment_sha256: str
    batch_sha256: str
    schema_version: str = OPEN_ECOLOGY_PROCESS_WORKER_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPEN_ECOLOGY_PROCESS_WORKER_BATCH_SCHEMA_VERSION:
            raise OpenEcologyProcessWorkerError("worker batch version drifted")
        _frontier_tick(self.target_tick, field="worker batch target_tick")
        if (
            not self.selected_task_ids
            or len(set(self.selected_task_ids)) != len(self.selected_task_ids)
            or set(self.results_by_task) != set(self.selected_task_ids)
        ):
            raise OpenEcologyProcessWorkerError(
                "worker batch result set is incomplete or duplicated"
            )
        for task_id in self.selected_task_ids:
            if self.results_by_task[task_id].task_id != task_id:
                raise OpenEcologyProcessWorkerError(
                    "worker batch result mapping identity drifted"
                )
        assignment_sha256 = _sha256(
            self.assignment_sha256,
            field="worker batch assignment_sha256",
        )
        unsigned = {
            "schema_version": self.schema_version,
            "target_tick": self.target_tick,
            "selected_task_ids": list(self.selected_task_ids),
            "result_sha256_by_task": {
                task_id: self.results_by_task[task_id].result_sha256
                for task_id in self.selected_task_ids
            },
            "worker_statuses": [
                status.to_dict()
                for status in sorted(
                    self.worker_statuses,
                    key=lambda item: item.worker_index,
                )
            ],
            "assignment_sha256": assignment_sha256,
        }
        if _stable_digest(unsigned) != _sha256(
            self.batch_sha256,
            field="worker batch batch_sha256",
        ):
            raise OpenEcologyProcessWorkerError("worker batch digest mismatch")
        object.__setattr__(
            self,
            "results_by_task",
            MappingProxyType(dict(self.results_by_task)),
        )

    def runner_views(
        self,
        tasks_by_id: Mapping[str, PersistentIslandTask],
    ) -> dict[str, ProcessWorkerRunnerView]:
        selected_task_ids = set(self.selected_task_ids)
        supplied_task_ids = set(tasks_by_id)
        if not selected_task_ids.issubset(supplied_task_ids):
            missing = sorted(selected_task_ids - supplied_task_ids)
            raise OpenEcologyProcessWorkerError(
                "runner view task map is incomplete: " + ",".join(missing)
            )
        return {
            task_id: ProcessWorkerRunnerView(
                task=tasks_by_id[task_id],
                result=self.results_by_task[task_id],
            )
            for task_id in self.selected_task_ids
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_tick": self.target_tick,
            "selected_task_ids": list(self.selected_task_ids),
            "results": [
                self.results_by_task[task_id].to_dict()
                for task_id in self.selected_task_ids
            ],
            "worker_statuses": [status.to_dict() for status in self.worker_statuses],
            "assignment_sha256": self.assignment_sha256,
            "batch_sha256": self.batch_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProcessWorkerReconciliationBatch:
    """One intent-bound repair of a partially published process frontier."""

    frontier_batch: ProcessWorkerFrontierBatch
    intent_sha256: str
    attempt_authority_sha256_by_task: Mapping[str, str]
    disposition_by_task: Mapping[str, str]
    reconciliation_sha256: str
    schema_version: str = (
        OPEN_ECOLOGY_PROCESS_WORKER_RECONCILIATION_BATCH_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != OPEN_ECOLOGY_PROCESS_WORKER_RECONCILIATION_BATCH_SCHEMA_VERSION
        ):
            raise OpenEcologyProcessWorkerError(
                "worker reconciliation batch version drifted"
            )
        intent_sha256 = _sha256(
            self.intent_sha256,
            field="worker reconciliation intent_sha256",
        )
        selected = self.frontier_batch.selected_task_ids
        if set(self.attempt_authority_sha256_by_task) != set(selected) or set(
            self.disposition_by_task
        ) != set(selected):
            raise OpenEcologyProcessWorkerError(
                "worker reconciliation authority/disposition matrix is incomplete"
            )
        authorities = {
            task_id: _sha256(
                self.attempt_authority_sha256_by_task[task_id],
                field=f"{task_id}.worker reconciliation attempt authority",
            )
            for task_id in selected
        }
        dispositions = {
            task_id: _identifier(
                self.disposition_by_task[task_id],
                field=f"{task_id}.worker reconciliation disposition",
            )
            for task_id in selected
        }
        if set(dispositions.values()) - (
            OPEN_ECOLOGY_PROCESS_WORKER_RECONCILIATION_DISPOSITIONS
        ):
            raise OpenEcologyProcessWorkerError(
                "worker reconciliation disposition is not authorized"
            )
        unsigned = {
            "schema_version": self.schema_version,
            "frontier_batch_sha256": self.frontier_batch.batch_sha256,
            "intent_sha256": intent_sha256,
            "attempt_authority_sha256_by_task": authorities,
            "disposition_by_task": dispositions,
        }
        if _stable_digest(unsigned) != _sha256(
            self.reconciliation_sha256,
            field="worker reconciliation reconciliation_sha256",
        ):
            raise OpenEcologyProcessWorkerError(
                "worker reconciliation batch digest mismatch"
            )
        object.__setattr__(
            self,
            "attempt_authority_sha256_by_task",
            MappingProxyType(authorities),
        )
        object.__setattr__(
            self,
            "disposition_by_task",
            MappingProxyType(dispositions),
        )

    @property
    def target_tick(self) -> int:
        return self.frontier_batch.target_tick

    @property
    def selected_task_ids(self) -> tuple[str, ...]:
        return self.frontier_batch.selected_task_ids

    @property
    def results_by_task(self) -> Mapping[str, ProcessWorkerTaskResult]:
        return self.frontier_batch.results_by_task

    @property
    def worker_statuses(self) -> tuple[ProcessWorkerStatus, ...]:
        return self.frontier_batch.worker_statuses

    @property
    def assignment_sha256(self) -> str:
        return self.frontier_batch.assignment_sha256

    def runner_views(
        self,
        tasks_by_id: Mapping[str, PersistentIslandTask],
    ) -> dict[str, ProcessWorkerRunnerView]:
        return self.frontier_batch.runner_views(tasks_by_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frontier_batch": self.frontier_batch.to_dict(),
            "intent_sha256": self.intent_sha256,
            "attempt_authority_sha256_by_task": dict(
                self.attempt_authority_sha256_by_task
            ),
            "disposition_by_task": dict(self.disposition_by_task),
            "reconciliation_sha256": self.reconciliation_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProcessWorkerRunnerView:
    """Read-only CampaignRunner-compatible view of one child-owned runner."""

    task: PersistentIslandTask
    result: ProcessWorkerTaskResult

    def __post_init__(self) -> None:
        if self.task.task_id != self.result.task_id:
            raise OpenEcologyProcessWorkerError(
                "process worker runner view task/result mismatch"
            )

    @property
    def observed_tick(self) -> int:
        return self.result.observed_tick

    @property
    def extinct(self) -> bool:
        return self.result.extinct

    @property
    def latest_aggregate_resume_pins(self) -> OpenEcologyAggregateResumePins:
        return self.result.aggregate_resume_pins

    @property
    def quiescent_checkpoint(self) -> Mapping[str, object]:
        return _json_clone_mapping(self.result.quiescent_checkpoint)

    def campaign_barrier_frontier(self) -> Mapping[str, object]:
        return _json_clone_mapping(self.result.campaign_barrier_frontier)

    def advance_to(self, target_tick: int) -> NoReturn:
        del target_tick
        raise OpenEcologyProcessWorkerError(
            "process worker runner views are read-only; use launcher.advance_frontier"
        )


class PersistentProcessWorkerLauncher:
    """Own long-lived local OS workers for the exact static task queues."""

    execution_contract = OPEN_ECOLOGY_PROCESS_WORKER_LAUNCHER_CONTRACT

    def __init__(
        self,
        *,
        tasks: Sequence[PersistentIslandTask],
        campaign_root: str | Path,
        campaign_id: str,
        repository_root: str | Path,
        source_git_sha: str,
        source_manifest_sha256: str,
        git_executable: str | Path | None = None,
        git_executable_sha256: str | None = None,
        slots: Sequence[ProcessWorkerSlot],
        assignment_sha256: str,
        response_timeout_seconds: float = (
            OPEN_ECOLOGY_PROCESS_WORKER_DEFAULT_TIMEOUT_SECONDS
        ),
        startup_timeout_seconds: float = (
            OPEN_ECOLOGY_PROCESS_WORKER_DEFAULT_STARTUP_TIMEOUT_SECONDS
        ),
        shutdown_timeout_seconds: float = (
            OPEN_ECOLOGY_PROCESS_WORKER_DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
        ),
        max_message_bytes: int = (
            OPEN_ECOLOGY_PROCESS_WORKER_DEFAULT_MAX_MESSAGE_BYTES
        ),
        allow_cpu_oversubscription: bool = False,
        _test_behaviors: Mapping[int, str] | None = None,
        _multiprocessing_start_method: str = "spawn",
    ) -> None:
        self.tasks = tuple(tasks)
        if len(self.tasks) != OPEN_ECOLOGY_PHASE_D_TASK_COUNT:
            raise OpenEcologyProcessWorkerError(
                "process worker launcher requires the exact 48-task authority"
            )
        self._task_by_id = {task.task_id: task for task in self.tasks}
        if len(self._task_by_id) != len(self.tasks):
            raise OpenEcologyProcessWorkerError("launcher task authority is duplicated")
        expected_order = tuple(
            (learner_index, island_index, arm)
            for learner_index in range(OPEN_ECOLOGY_PHASE_D_LEARNER_COUNT)
            for island_index in range(OPEN_ECOLOGY_PHASE_D_ISLAND_COUNT)
            for arm in OPEN_ECOLOGY_PHASE_D_ARM_ORDER
        )
        if (
            tuple(
                (task.learner_index, task.island_index, task.arm) for task in self.tasks
            )
            != expected_order
        ):
            raise OpenEcologyProcessWorkerError(
                "process worker task authority drifted from learner/island/H-Z-R "
                "matrix order"
            )
        self.campaign_root = _absolute_directory(
            campaign_root,
            field="campaign_root",
        )
        self.repository_root = _absolute_directory(
            repository_root,
            field="repository_root",
        )
        self.campaign_id = _identifier(campaign_id, field="campaign_id")
        self.source_git_sha = _git_sha(source_git_sha, field="source_git_sha")
        self.source_manifest_sha256 = _sha256(
            source_manifest_sha256,
            field="source_manifest_sha256",
        )
        _refuse_fatal_worker_marker(self.campaign_root)
        if {task.artifact.source_commit for task in self.tasks} != {
            self.source_git_sha
        }:
            raise OpenEcologyProcessWorkerError(
                "launcher source SHA does not match every task artifact"
            )
        self.slots = tuple(slots)
        self._validate_slots(allow_cpu_oversubscription)
        self._slot_by_index = {slot.worker_index: slot for slot in self.slots}
        self._task_to_worker = {
            task_id: slot.worker_index
            for slot in self.slots
            for task_id in slot.task_ids
        }
        self.assignment_records = _assignment_records(self.tasks, self.slots)
        computed_assignment_sha256 = _assignment_sha256(self.assignment_records)
        self.assignment_sha256 = _sha256(
            assignment_sha256,
            field="assignment_sha256",
        )
        if self.assignment_sha256 != computed_assignment_sha256:
            raise OpenEcologyProcessWorkerError(
                "launcher static assignment digest does not match its queues"
            )
        self.response_timeout_seconds = _positive_float(
            response_timeout_seconds,
            field="response_timeout_seconds",
        )
        self.startup_timeout_seconds = _positive_float(
            startup_timeout_seconds,
            field="startup_timeout_seconds",
        )
        self.shutdown_timeout_seconds = _positive_float(
            shutdown_timeout_seconds,
            field="shutdown_timeout_seconds",
        )
        self.max_message_bytes = _bounded_message_bytes(max_message_bytes)
        if _test_behaviors is None:
            self._backend = "persistent_island_v1"
            behaviors: dict[int, str] = {}
        else:
            self._backend = "protocol_test_v1"
            behaviors = dict(_test_behaviors)
            if set(behaviors) - set(self._slot_by_index):
                raise OpenEcologyProcessWorkerError(
                    "test behavior names an unknown worker"
                )
        allowed_test_behaviors = {
            "normal",
            "die_on_advance",
            "sleep_on_advance",
            "partial_frontier",
            "reconcile_forged",
            "reconcile_mixed",
            "source_mismatch_startup",
            "source_mismatch_after",
        }
        if set(behaviors.values()) - allowed_test_behaviors:
            raise OpenEcologyProcessWorkerError("unknown protocol test behavior")
        self._test_behaviors = behaviors
        if self._backend == "persistent_island_v1":
            if git_executable is None or git_executable_sha256 is None:
                raise OpenEcologyProcessWorkerError(
                    "production workers require absolute pinned Git authority"
                )
            self.git_authority = pin_open_ecology_git_executable(
                Path(git_executable),
                expected_sha256=git_executable_sha256,
            )
        else:
            self.git_authority = None
        try:
            self._context = multiprocessing.get_context(_multiprocessing_start_method)
        except ValueError as error:
            raise OpenEcologyProcessWorkerError(
                "unsupported multiprocessing start method"
            ) from error
        self._processes: dict[int, multiprocessing.Process] = {}
        self._connections: dict[int, Connection] = {}
        self._connection_workers: dict[Connection, int] = {}
        self._worker_pids: dict[int, int] = {}
        self._sequence_index = 0
        self._last_results: dict[str, ProcessWorkerTaskResult] = {}
        self._closed = False
        self._poisoned = False
        self._spawn_workers()

    def __enter__(self) -> PersistentProcessWorkerLauncher:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    @property
    def worker_pids(self) -> Mapping[int, int]:
        return dict(self._worker_pids)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def advance_frontier(
        self,
        *,
        selected_task_ids: Sequence[str],
        target_tick: int,
        expected_states: Mapping[str, ProcessWorkerExpectedState],
    ) -> ProcessWorkerFrontierBatch:
        """Advance one exact selected scope and return only after every join."""

        target, selected, expected, by_worker = (
            self._validate_selected_frontier_request(
                selected_task_ids=selected_task_ids,
                target_tick=target_tick,
                expected_states=expected_states,
                require_retained_result_match=True,
            )
        )

        self._sequence_index += 1
        sequence_index = self._sequence_index
        for worker_index in sorted(by_worker):
            payload = {
                "message_type": "advance",
                "sequence_index": sequence_index,
                "target_tick": target,
                "expected_states": [
                    expected[task_id].to_dict() for task_id in by_worker[worker_index]
                ],
            }
            self._send(worker_index, payload)

        partial_results: dict[str, ProcessWorkerTaskResult] = {}
        statuses: list[ProcessWorkerStatus] = []
        pending = set(by_worker)
        deadline = time.monotonic() + self.response_timeout_seconds
        try:
            while pending:
                self._fail_if_worker_died(pending)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OpenEcologyProcessWorkerError(
                        "process worker frontier timed out",
                        partial_results=partial_results,
                    )
                ready = wait(
                    [self._connections[index] for index in pending],
                    timeout=min(remaining, 0.5),
                )
                if not ready:
                    continue
                for connection in ready:
                    worker_index = self._connection_workers[connection]
                    response = self._receive(worker_index)
                    message_type = response.get("message_type")
                    if (
                        response.get("sequence_index") != sequence_index
                        or response.get("worker_index") != worker_index
                    ):
                        raise OpenEcologyProcessWorkerError(
                            "worker response sequence or identity drifted",
                            partial_results=partial_results,
                        )
                    if message_type == "error":
                        for raw_result in _sequence(
                            response.get("completed_results"),
                            field="worker error completed_results",
                        ):
                            result = ProcessWorkerTaskResult.from_dict(
                                _mapping(raw_result, field="worker partial result")
                            )
                            partial_results[result.task_id] = result
                        error_message = _string(
                            response.get("error_message"),
                            field="worker error error_message",
                        )
                        raise OpenEcologyProcessWorkerError(
                            f"worker {worker_index} failed: {error_message}",
                            partial_results=partial_results,
                        )
                    if message_type != "advance_result":
                        raise OpenEcologyProcessWorkerError(
                            "worker returned an unexpected message type",
                            partial_results=partial_results,
                        )
                    worker_results: list[ProcessWorkerTaskResult] = []
                    for raw_result in _sequence(
                        response.get("results"),
                        field="worker results",
                    ):
                        result = ProcessWorkerTaskResult.from_dict(
                            _mapping(raw_result, field="worker result")
                        )
                        if result.task_id in partial_results:
                            raise OpenEcologyProcessWorkerError(
                                "worker result duplicates a task",
                                partial_results=partial_results,
                            )
                        partial_results[result.task_id] = result
                        worker_results.append(result)
                    status = ProcessWorkerStatus.from_dict(
                        _mapping(response.get("status"), field="worker result status")
                    )
                    self._validate_worker_response(
                        worker_index=worker_index,
                        expected_task_ids=tuple(by_worker[worker_index]),
                        target_tick=target,
                        expected=expected,
                        results=tuple(worker_results),
                        status=status,
                    )
                    statuses.append(status)
                    pending.remove(worker_index)
            if set(partial_results) != set(selected):
                raise OpenEcologyProcessWorkerError(
                    "worker batch omitted selected tasks",
                    partial_results=partial_results,
                )
        except BaseException:
            self._poison_workers()
            raise

        self._last_results.update(partial_results)
        unsigned_batch = {
            "schema_version": OPEN_ECOLOGY_PROCESS_WORKER_BATCH_SCHEMA_VERSION,
            "target_tick": target,
            "selected_task_ids": list(selected),
            "result_sha256_by_task": {
                task_id: partial_results[task_id].result_sha256 for task_id in selected
            },
            "worker_statuses": [
                status.to_dict()
                for status in sorted(
                    statuses,
                    key=lambda item: item.worker_index,
                )
            ],
            "assignment_sha256": self.assignment_sha256,
        }
        return ProcessWorkerFrontierBatch(
            target_tick=target,
            selected_task_ids=selected,
            results_by_task=dict(partial_results),
            worker_statuses=tuple(sorted(statuses, key=lambda item: item.worker_index)),
            assignment_sha256=self.assignment_sha256,
            batch_sha256=_stable_digest(unsigned_batch),
        )

    def reconcile_frontier(
        self,
        *,
        selected_task_ids: Sequence[str],
        target_tick: int,
        expected_states: Mapping[str, ProcessWorkerExpectedState],
        intent_sha256: str,
        attempt_authority_sha256_by_task: Mapping[str, str],
    ) -> ProcessWorkerReconciliationBatch:
        """Resolve one original intent across a mixed durable worker frontier."""

        target, selected, expected, by_worker = (
            self._validate_selected_frontier_request(
                selected_task_ids=selected_task_ids,
                target_tick=target_tick,
                expected_states=expected_states,
                require_retained_result_match=False,
            )
        )
        intent = _sha256(intent_sha256, field="reconciliation intent_sha256")
        if set(attempt_authority_sha256_by_task) != set(selected):
            raise OpenEcologyProcessWorkerError(
                "reconciliation attempt authority map must exactly match selected tasks"
            )
        authorities = {
            task_id: _sha256(
                attempt_authority_sha256_by_task[task_id],
                field=f"{task_id}.reconciliation attempt authority",
            )
            for task_id in selected
        }

        self._sequence_index += 1
        sequence_index = self._sequence_index
        for worker_index in sorted(by_worker):
            task_ids = by_worker[worker_index]
            self._send(
                worker_index,
                {
                    "message_type": "reconcile",
                    "sequence_index": sequence_index,
                    "target_tick": target,
                    "expected_states": [
                        expected[task_id].to_dict() for task_id in task_ids
                    ],
                    "intent_sha256": intent,
                    "attempt_authority_sha256_by_task": {
                        task_id: authorities[task_id] for task_id in task_ids
                    },
                },
            )

        partial_results: dict[str, ProcessWorkerTaskResult] = {}
        dispositions: dict[str, str] = {}
        statuses: list[ProcessWorkerStatus] = []
        pending = set(by_worker)
        deadline = time.monotonic() + self.response_timeout_seconds
        try:
            while pending:
                self._fail_if_worker_died(pending)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OpenEcologyProcessWorkerError(
                        "process worker reconciliation timed out",
                        partial_results=partial_results,
                    )
                ready = wait(
                    [self._connections[index] for index in pending],
                    timeout=min(remaining, 0.5),
                )
                if not ready:
                    continue
                for connection in ready:
                    worker_index = self._connection_workers[connection]
                    response = self._receive(worker_index)
                    message_type = response.get("message_type")
                    if (
                        response.get("sequence_index") != sequence_index
                        or response.get("worker_index") != worker_index
                    ):
                        raise OpenEcologyProcessWorkerError(
                            "reconciliation response sequence or identity drifted",
                            partial_results=partial_results,
                        )
                    if message_type == "error":
                        for raw_result in _sequence(
                            response.get("completed_results"),
                            field="worker error completed_results",
                        ):
                            result = ProcessWorkerTaskResult.from_dict(
                                _mapping(raw_result, field="worker partial result")
                            )
                            partial_results[result.task_id] = result
                        error_message = _string(
                            response.get("error_message"),
                            field="worker error error_message",
                        )
                        raise OpenEcologyProcessWorkerError(
                            f"worker {worker_index} failed: {error_message}",
                            partial_results=partial_results,
                        )
                    if message_type != "reconcile_result":
                        raise OpenEcologyProcessWorkerError(
                            "worker returned an unexpected reconciliation message",
                            partial_results=partial_results,
                        )
                    if response.get("intent_sha256") != intent:
                        raise OpenEcologyProcessWorkerError(
                            "worker reconciliation intent binding drifted",
                            partial_results=partial_results,
                        )
                    worker_results: list[ProcessWorkerTaskResult] = []
                    for raw_result in _sequence(
                        response.get("results"),
                        field="worker reconciliation results",
                    ):
                        result = ProcessWorkerTaskResult.from_dict(
                            _mapping(raw_result, field="worker reconciliation result")
                        )
                        if result.task_id in partial_results:
                            raise OpenEcologyProcessWorkerError(
                                "worker reconciliation duplicates a task",
                                partial_results=partial_results,
                            )
                        partial_results[result.task_id] = result
                        worker_results.append(result)
                    raw_dispositions = _mapping(
                        response.get("disposition_by_task"),
                        field="worker reconciliation dispositions",
                    )
                    expected_worker_ids = tuple(by_worker[worker_index])
                    if set(raw_dispositions) != set(expected_worker_ids):
                        raise OpenEcologyProcessWorkerError(
                            "worker reconciliation disposition matrix is incomplete",
                            partial_results=partial_results,
                        )
                    for task_id in expected_worker_ids:
                        dispositions[task_id] = _identifier(
                            raw_dispositions[task_id],
                            field=f"{task_id}.worker reconciliation disposition",
                        )
                    if set(dispositions.values()) - (
                        OPEN_ECOLOGY_PROCESS_WORKER_RECONCILIATION_DISPOSITIONS
                    ):
                        raise OpenEcologyProcessWorkerError(
                            "worker reconciliation disposition is not authorized",
                            partial_results=partial_results,
                        )
                    status = ProcessWorkerStatus.from_dict(
                        _mapping(
                            response.get("status"),
                            field="worker reconciliation status",
                        )
                    )
                    self._validate_worker_response(
                        worker_index=worker_index,
                        expected_task_ids=expected_worker_ids,
                        target_tick=target,
                        expected=expected,
                        results=tuple(worker_results),
                        status=status,
                    )
                    statuses.append(status)
                    pending.remove(worker_index)
            if set(partial_results) != set(selected) or set(dispositions) != set(
                selected
            ):
                raise OpenEcologyProcessWorkerError(
                    "worker reconciliation omitted selected tasks",
                    partial_results=partial_results,
                )
        except BaseException:
            self._poison_workers()
            raise

        self._last_results.update(partial_results)
        sorted_statuses = tuple(sorted(statuses, key=lambda item: item.worker_index))
        unsigned_batch = {
            "schema_version": OPEN_ECOLOGY_PROCESS_WORKER_BATCH_SCHEMA_VERSION,
            "target_tick": target,
            "selected_task_ids": list(selected),
            "result_sha256_by_task": {
                task_id: partial_results[task_id].result_sha256 for task_id in selected
            },
            "worker_statuses": [status.to_dict() for status in sorted_statuses],
            "assignment_sha256": self.assignment_sha256,
        }
        frontier_batch = ProcessWorkerFrontierBatch(
            target_tick=target,
            selected_task_ids=selected,
            results_by_task=dict(partial_results),
            worker_statuses=sorted_statuses,
            assignment_sha256=self.assignment_sha256,
            batch_sha256=_stable_digest(unsigned_batch),
        )
        unsigned_reconciliation = {
            "schema_version": (
                OPEN_ECOLOGY_PROCESS_WORKER_RECONCILIATION_BATCH_SCHEMA_VERSION
            ),
            "frontier_batch_sha256": frontier_batch.batch_sha256,
            "intent_sha256": intent,
            "attempt_authority_sha256_by_task": authorities,
            "disposition_by_task": {
                task_id: dispositions[task_id] for task_id in selected
            },
        }
        return ProcessWorkerReconciliationBatch(
            frontier_batch=frontier_batch,
            intent_sha256=intent,
            attempt_authority_sha256_by_task=authorities,
            disposition_by_task={
                task_id: dispositions[task_id] for task_id in selected
            },
            reconciliation_sha256=_stable_digest(unsigned_reconciliation),
        )

    def _validate_selected_frontier_request(
        self,
        *,
        selected_task_ids: Sequence[str],
        target_tick: int,
        expected_states: Mapping[str, ProcessWorkerExpectedState],
        require_retained_result_match: bool,
    ) -> tuple[
        int,
        tuple[str, ...],
        dict[str, ProcessWorkerExpectedState],
        dict[int, list[str]],
    ]:
        self._require_live()
        target = _frontier_tick(target_tick, field="target_tick")
        selected = _ordered_selected_tasks(self.tasks, selected_task_ids)
        if set(expected_states) != set(selected):
            raise OpenEcologyProcessWorkerError(
                "expected state map must exactly match selected tasks"
            )
        expected: dict[str, ProcessWorkerExpectedState] = {}
        for task_id in selected:
            state = expected_states[task_id]
            if not isinstance(state, ProcessWorkerExpectedState):
                raise OpenEcologyProcessWorkerError(
                    "expected state map contains a non-state value"
                )
            if state.task_id != task_id:
                raise OpenEcologyProcessWorkerError(
                    "expected state map key/task identity drifted"
                )
            if not state.terminal_extinct and target != state.observed_tick + (
                _FRONTIER_INTERVAL
            ):
                raise OpenEcologyProcessWorkerError(
                    "nonterminal task must advance exactly one checkpoint interval"
                )
            if state.observed_tick > target:
                raise OpenEcologyProcessWorkerError(
                    "expected task state is ahead of requested frontier"
                )
            previous = self._last_results.get(task_id)
            if (
                require_retained_result_match
                and previous is not None
                and not _expected_matches_result(state, previous)
            ):
                raise OpenEcologyProcessWorkerError(
                    f"expected state does not match retained worker result for {task_id}"
                )
            expected[task_id] = state

        by_worker: dict[int, list[str]] = {}
        for task_id in selected:
            worker_index = self._task_to_worker[task_id]
            by_worker.setdefault(worker_index, []).append(task_id)
        for worker_index, task_ids in by_worker.items():
            slot_queue = self._slot_by_index[worker_index].task_ids
            if tuple(task_ids) != tuple(
                task_id for task_id in slot_queue if task_id in set(task_ids)
            ):
                raise OpenEcologyProcessWorkerError(
                    "selected task queue is not a static-order subsequence"
                )
        return target, selected, expected, by_worker

    def close(self) -> None:
        """Request an orderly worker exit and retain every evidence file."""

        if self._closed:
            return
        if self._poisoned:
            self._join_or_terminate_all()
            self._closed = True
            return
        self._sequence_index += 1
        sequence_index = self._sequence_index
        pending: set[int] = set()
        for worker_index, process in self._processes.items():
            if process.is_alive():
                self._send(
                    worker_index,
                    {
                        "message_type": "shutdown",
                        "sequence_index": sequence_index,
                    },
                )
                pending.add(worker_index)
        deadline = time.monotonic() + self.shutdown_timeout_seconds
        try:
            while pending:
                self._fail_if_worker_died(pending, allow_clean_exit=True)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OpenEcologyProcessWorkerError(
                        "process worker graceful shutdown timed out"
                    )
                ready = wait(
                    [self._connections[index] for index in pending],
                    timeout=min(remaining, 0.25),
                )
                for connection in ready:
                    worker_index = self._connection_workers[connection]
                    payload = self._receive(worker_index)
                    if payload != {
                        "message_type": "shutdown_ack",
                        "sequence_index": sequence_index,
                        "worker_index": worker_index,
                        "worker_pid": self._worker_pids[worker_index],
                        "evidence_deleted": False,
                    }:
                        raise OpenEcologyProcessWorkerError(
                            "worker shutdown acknowledgement drifted"
                        )
                    pending.remove(worker_index)
            for process in self._processes.values():
                process.join(timeout=self.shutdown_timeout_seconds)
                if process.exitcode != 0:
                    raise OpenEcologyProcessWorkerError(
                        "process worker exited nonzero during graceful shutdown"
                    )
        except BaseException:
            self._poison_workers()
            raise
        finally:
            self._close_connections()
            self._closed = True

    def abort(self) -> None:
        """Stop workers after an unsafe outcome without deleting evidence."""

        if self._closed:
            return
        self._poison_workers()

    def _spawn_workers(self) -> None:
        try:
            for slot in self.slots:
                parent_connection, child_connection = self._context.Pipe(duplex=True)
                init_payload = {
                    "message_type": "initialize",
                    "worker_index": slot.worker_index,
                    "slot": slot.to_dict(),
                    "campaign_root": str(self.campaign_root),
                    "campaign_id": self.campaign_id,
                    "repository_root": str(self.repository_root),
                    "source_git_sha": self.source_git_sha,
                    "source_manifest_sha256": self.source_manifest_sha256,
                    "git_authority": self.git_authority,
                    "tasks": [
                        self._task_by_id[task_id].to_dict() for task_id in slot.task_ids
                    ],
                    "assignment_sha256": self.assignment_sha256,
                    "backend": self._backend,
                    "test_behavior": self._test_behaviors.get(
                        slot.worker_index,
                        "normal",
                    ),
                    "max_message_bytes": self.max_message_bytes,
                }
                init_bytes = _encode_message(
                    init_payload,
                    max_message_bytes=self.max_message_bytes,
                )
                process = self._context.Process(
                    target=_worker_process_main,
                    args=(child_connection, init_bytes, self.max_message_bytes),
                    name=f"open-ecology-worker-{slot.worker_index:02d}",
                    daemon=False,
                )
                process.start()
                child_connection.close()
                self._processes[slot.worker_index] = process
                self._connections[slot.worker_index] = parent_connection
                self._connection_workers[parent_connection] = slot.worker_index

            pending = set(self._processes)
            deadline = time.monotonic() + self.startup_timeout_seconds
            while pending:
                self._fail_if_worker_died(pending)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OpenEcologyProcessWorkerError(
                        "process worker startup timed out"
                    )
                ready = wait(
                    [self._connections[index] for index in pending],
                    timeout=min(remaining, 0.25),
                )
                for connection in ready:
                    worker_index = self._connection_workers[connection]
                    payload = self._receive(worker_index)
                    self._validate_ready(worker_index, payload)
                    pending.remove(worker_index)
        except BaseException:
            self._poison_workers()
            raise

    def _validate_ready(
        self,
        worker_index: int,
        payload: Mapping[str, object],
    ) -> None:
        _exact_keys(
            payload,
            {
                "message_type",
                "worker_index",
                "worker_pid",
                "host_identity",
                "device_kind",
                "device_index",
                "torch_threads",
                "source",
                "assignment_sha256",
            },
            field="worker ready",
        )
        slot = self._slot_by_index[worker_index]
        process = self._processes[worker_index]
        worker_pid = _positive_int(
            payload["worker_pid"],
            field="worker ready worker_pid",
        )
        if (
            payload["message_type"] != "ready"
            or payload["worker_index"] != worker_index
            or process.pid != worker_pid
            or payload["host_identity"] != slot.host_identity
            or payload["device_kind"] != slot.device_kind
            or payload["device_index"] != slot.device_index
            or payload["torch_threads"] != slot.torch_threads
            or payload["assignment_sha256"] != self.assignment_sha256
        ):
            raise OpenEcologyProcessWorkerError("process worker ready identity drifted")
        self._validate_source(_mapping(payload["source"], field="worker ready source"))
        self._worker_pids[worker_index] = worker_pid

    def _validate_worker_response(
        self,
        *,
        worker_index: int,
        expected_task_ids: tuple[str, ...],
        target_tick: int,
        expected: Mapping[str, ProcessWorkerExpectedState],
        results: tuple[ProcessWorkerTaskResult, ...],
        status: ProcessWorkerStatus,
    ) -> None:
        slot = self._slot_by_index[worker_index]
        result_ids = tuple(result.task_id for result in results)
        if result_ids != expected_task_ids or status.completed_task_ids != result_ids:
            raise OpenEcologyProcessWorkerError(
                "worker result queue is missing, duplicated, or reordered"
            )
        if (
            status.worker_index != worker_index
            or status.worker_pid != self._worker_pids[worker_index]
            or status.host_identity != slot.host_identity
            or status.device_kind != slot.device_kind
            or status.device_index != slot.device_index
            or status.torch_threads != slot.torch_threads
        ):
            raise OpenEcologyProcessWorkerError("worker result status identity drifted")
        self._validate_source(status.source_before)
        self._validate_source(status.source_after)
        if status.source_before != status.source_after:
            raise OpenEcologyProcessWorkerError(
                "worker source changed during frontier advance"
            )
        for result in results:
            expected_state = expected[result.task_id]
            if (
                result.worker_index != worker_index
                or result.worker_pid != status.worker_pid
                or result.requested_target_tick != target_tick
                or not result.same_world_instance
            ):
                raise OpenEcologyProcessWorkerError(
                    f"worker operational result drifted for {result.task_id}"
                )
            if result.extinct:
                if (
                    result.observed_tick > target_tick
                    or result.extinction_tick != result.observed_tick
                ):
                    raise OpenEcologyProcessWorkerError(
                        f"terminal worker frontier drifted for {result.task_id}"
                    )
            elif result.observed_tick != target_tick:
                raise OpenEcologyProcessWorkerError(
                    f"worker left {result.task_id} at a partial frontier"
                )
            if result.observed_tick < expected_state.observed_tick:
                raise OpenEcologyProcessWorkerError(
                    f"worker moved {result.task_id} backwards"
                )
            if (
                result.aggregate_resume_pins.identity.source_git_sha
                != self.source_git_sha
                or result.aggregate_resume_pins.identity.island_id != result.task_id
                or result.observed_tick <= 0
                or result.aggregate_resume_pins.identity.tick
                != result.observed_tick - 1
            ):
                raise OpenEcologyProcessWorkerError(
                    f"worker aggregate pins drifted for {result.task_id}"
                )
            quiescent = result.quiescent_checkpoint
            frontier = result.campaign_barrier_frontier
            expected_quiescent = {
                "task_id": result.task_id,
                "campaign_id": self.campaign_id,
                "campaign_root": str(self.campaign_root),
                "source_git_sha": self.source_git_sha,
                "observed_tick": result.observed_tick,
                "archive_safe_while_runner_idle": True,
                "restartable_world_checkpoint_present": True,
                "aggregate_current_restartable": True,
                "aggregate_resume_authorized": True,
            }
            for key, value in expected_quiescent.items():
                if quiescent.get(key) != value:
                    raise OpenEcologyProcessWorkerError(
                        f"worker quiescent boundary drifted for {result.task_id}"
                    )
            if (
                frontier.get("task_id") != result.task_id
                or frontier.get("campaign_id") != self.campaign_id
                or frontier.get("campaign_root") != str(self.campaign_root)
                or frontier.get("source_git_sha") != self.source_git_sha
                or frontier.get("observed_tick") != result.observed_tick
                or frontier.get("extinct") is not result.extinct
                or frontier.get("aggregate_commit_sha256")
                != result.aggregate_resume_pins.commit_sha256
                or frontier.get("checkpoint_sha256")
                != result.aggregate_resume_pins.checkpoint_sha256
            ):
                raise OpenEcologyProcessWorkerError(
                    f"worker campaign frontier drifted for {result.task_id}"
                )

    def _validate_source(self, source: Mapping[str, object]) -> None:
        expected = {
            "repository_root": str(self.repository_root),
            "source_git_sha": self.source_git_sha,
            "source_manifest_sha256": self.source_manifest_sha256,
            "git_clean": True,
        }
        if source != expected:
            raise OpenEcologyProcessWorkerError(
                "process worker source authority mismatch"
            )

    def _validate_slots(self, allow_cpu_oversubscription: bool) -> None:
        if not self.slots:
            raise OpenEcologyProcessWorkerError(
                "process worker launcher requires at least one slot"
            )
        indices = tuple(slot.worker_index for slot in self.slots)
        if indices != tuple(range(len(self.slots))):
            raise OpenEcologyProcessWorkerError(
                "process worker slot indices must be contiguous from zero"
            )
        actual_host = local_process_worker_host_identity()
        if {slot.host_identity for slot in self.slots} != {actual_host}:
            raise OpenEcologyProcessWorkerError(
                "local process worker slots must name the current host explicitly"
            )
        queue_ids = tuple(task_id for slot in self.slots for task_id in slot.task_ids)
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(queue_ids) != len(set(queue_ids)) or set(queue_ids) != set(task_ids):
            raise OpenEcologyProcessWorkerError(
                "process worker queues must partition the complete task authority"
            )
        expected_queues: list[list[str]] = [[] for _ in self.slots]
        for global_index, task_id in enumerate(task_ids):
            expected_queues[global_index % len(self.slots)].append(task_id)
        if tuple(slot.task_ids for slot in self.slots) != tuple(
            tuple(queue) for queue in expected_queues
        ):
            raise OpenEcologyProcessWorkerError(
                "process worker queues must use deterministic matrix round-robin "
                "assignment"
            )
        cpu_count = os.cpu_count() or 1
        declared_threads = sum(slot.torch_threads for slot in self.slots)
        if not isinstance(allow_cpu_oversubscription, bool):
            raise OpenEcologyProcessWorkerError(
                "allow_cpu_oversubscription must be boolean"
            )
        if not allow_cpu_oversubscription and declared_threads > cpu_count:
            raise OpenEcologyProcessWorkerError(
                "declared process worker Torch threads oversubscribe host CPUs; "
                "set explicit oversubscription authority only after benchmarking"
            )

    def _send(self, worker_index: int, payload: Mapping[str, object]) -> None:
        connection = self._connections[worker_index]
        try:
            connection.send_bytes(
                _encode_message(
                    payload,
                    max_message_bytes=self.max_message_bytes,
                )
            )
        except (BrokenPipeError, EOFError, OSError) as error:
            raise OpenEcologyProcessWorkerError(
                f"cannot send command to worker {worker_index}"
            ) from error

    def _receive(self, worker_index: int) -> Mapping[str, object]:
        connection = self._connections[worker_index]
        try:
            payload_bytes = connection.recv_bytes(self.max_message_bytes)
        except (EOFError, OSError) as error:
            raise OpenEcologyProcessWorkerError(
                f"cannot receive bounded result from worker {worker_index}"
            ) from error
        return _decode_message(
            payload_bytes,
            max_message_bytes=self.max_message_bytes,
        )

    def _fail_if_worker_died(
        self,
        pending: set[int],
        *,
        allow_clean_exit: bool = False,
    ) -> None:
        for worker_index in pending:
            process = self._processes[worker_index]
            if process.is_alive():
                continue
            if allow_clean_exit and process.exitcode == 0:
                continue
            raise OpenEcologyProcessWorkerError(
                f"process worker {worker_index} died with exit code {process.exitcode}"
            )

    def _require_live(self) -> None:
        if self._closed:
            raise OpenEcologyProcessWorkerError("process worker launcher is closed")
        if self._poisoned:
            raise OpenEcologyProcessWorkerError("process worker launcher is poisoned")
        self._fail_if_worker_died(set(self._processes))

    def _poison_workers(self) -> None:
        self._poisoned = True
        try:
            self._join_or_terminate_all()
        finally:
            self._close_connections()
            self._closed = True

    def _join_or_terminate_all(self) -> None:
        for process in self._processes.values():
            if process.is_alive():
                process.terminate()
        for process in self._processes.values():
            process.join(timeout=self.shutdown_timeout_seconds)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=self.shutdown_timeout_seconds)
        surviving = {
            worker_index: process.pid
            for worker_index, process in self._processes.items()
            if process.is_alive()
        }
        if surviving:
            _write_fatal_worker_marker(
                self.campaign_root,
                campaign_id=self.campaign_id,
                source_git_sha=self.source_git_sha,
                source_manifest_sha256=self.source_manifest_sha256,
                surviving_worker_pids=surviving,
            )
            raise OpenEcologyProcessWorkerError(
                "process workers survived terminate and kill deadlines; "
                "a durable fatal PID marker now blocks campaign restart: "
                + ",".join(
                    f"worker-{worker_index}=pid-{pid}"
                    for worker_index, pid in sorted(surviving.items())
                )
            )

    def _close_connections(self) -> None:
        for connection in self._connections.values():
            try:
                connection.close()
            except OSError:
                pass


def _fatal_worker_marker_path(campaign_root: Path) -> Path:
    return campaign_root / OPEN_ECOLOGY_PROCESS_WORKER_FATAL_MARKER_NAME


def _refuse_fatal_worker_marker(campaign_root: Path) -> None:
    marker = _fatal_worker_marker_path(campaign_root)
    if marker.exists() or marker.is_symlink():
        raise OpenEcologyProcessWorkerError(
            "durable fatal worker PID marker blocks process-worker restart; "
            "the previous launcher could not prove all children stopped"
        )


def _write_fatal_worker_marker(
    campaign_root: Path,
    *,
    campaign_id: str,
    source_git_sha: str,
    source_manifest_sha256: str,
    surviving_worker_pids: Mapping[int, int | None],
) -> Path:
    parsed_survivors = {
        _nonnegative_int(worker_index, field="fatal worker index"): _positive_int(
            pid,
            field="fatal worker PID",
        )
        for worker_index, pid in surviving_worker_pids.items()
    }
    if not parsed_survivors:
        raise OpenEcologyProcessWorkerError(
            "fatal worker marker requires at least one surviving PID"
        )
    unsigned: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PROCESS_WORKER_FATAL_MARKER_SCHEMA_VERSION,
        "campaign_id": _identifier(campaign_id, field="fatal campaign_id"),
        "source_git_sha": _git_sha(
            source_git_sha,
            field="fatal source_git_sha",
        ),
        "source_manifest_sha256": _sha256(
            source_manifest_sha256,
            field="fatal source_manifest_sha256",
        ),
        "launcher_pid": os.getpid(),
        "surviving_worker_pids": {
            str(worker_index): pid
            for worker_index, pid in sorted(parsed_survivors.items())
        },
        "restart_blocked": True,
        "evidence_deleted": False,
        "reason": "worker_survived_terminate_and_kill_deadlines",
    }
    payload = {
        **unsigned,
        "exact_digest": hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    }
    encoded = canonical_json_bytes(payload)
    if len(encoded) > OPEN_ECOLOGY_PROCESS_WORKER_MAX_FATAL_MARKER_BYTES:
        raise OpenEcologyProcessWorkerError("fatal worker marker exceeds 64 KiB")
    destination = _fatal_worker_marker_path(campaign_root)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(destination, flags, 0o400)
    except FileExistsError:
        _refuse_fatal_worker_marker(campaign_root)
        raise AssertionError("fatal marker refusal unexpectedly returned")
    except OSError as error:
        raise OpenEcologyProcessWorkerError(
            f"cannot durably create fatal worker marker: {error}"
        ) from error
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OpenEcologyProcessWorkerError(
                    "fatal worker marker write made no progress"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(
        campaign_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return destination


def build_process_worker_slots(
    tasks: Sequence[PersistentIslandTask],
    *,
    worker_count: int,
    host_identity: str,
    torch_threads_per_worker: int = 1,
) -> tuple[ProcessWorkerSlot, ...]:
    """Build the deterministic full-matrix round-robin process queues."""

    if len(tasks) != OPEN_ECOLOGY_PHASE_D_TASK_COUNT:
        raise OpenEcologyProcessWorkerError(
            "process worker assignment requires the exact 48-task matrix"
        )
    parsed_workers = _positive_int(worker_count, field="worker_count")
    if parsed_workers > len(tasks):
        raise OpenEcologyProcessWorkerError("worker_count cannot exceed task count")
    host = _identifier(host_identity, field="host_identity")
    threads = _positive_int(
        torch_threads_per_worker,
        field="torch_threads_per_worker",
    )
    queues: list[list[str]] = [[] for _ in range(parsed_workers)]
    for global_index, task in enumerate(tasks):
        queues[global_index % parsed_workers].append(task.task_id)
    return tuple(
        ProcessWorkerSlot(
            worker_index=worker_index,
            host_identity=host,
            device_kind="cpu",
            device_index=None,
            torch_threads=threads,
            task_ids=tuple(queue),
        )
        for worker_index, queue in enumerate(queues)
    )


def process_worker_assignment_sha256(
    tasks: Sequence[PersistentIslandTask],
    slots: Sequence[ProcessWorkerSlot],
) -> str:
    return _assignment_sha256(_assignment_records(tuple(tasks), tuple(slots)))


def local_process_worker_host_identity() -> str:
    return _identifier(socket.gethostname(), field="local host identity")


class _ProtocolTestRuntime:
    def __init__(
        self,
        *,
        slot: ProcessWorkerSlot,
        tasks: Mapping[str, PersistentIslandTask],
        campaign_root: Path,
        campaign_id: str,
        source_git_sha: str,
        behavior: str,
    ) -> None:
        self.slot = slot
        self.tasks = tasks
        self.campaign_root = campaign_root
        self.campaign_id = campaign_id
        self.source_git_sha = source_git_sha
        self.behavior = behavior
        self.ticks: dict[str, int] = {}
        self.pins: dict[str, OpenEcologyAggregateResumePins] = {}
        self.advance_count = 0

    def advance(
        self,
        *,
        expected_states: tuple[ProcessWorkerExpectedState, ...],
        target_tick: int,
    ) -> tuple[dict[str, object], ...]:
        self.advance_count += 1
        marker = self.campaign_root / (
            f"protocol-test-worker-{self.slot.worker_index}-"
            f"pid-{os.getpid()}-advance-{self.advance_count}.preserved"
        )
        marker.touch(exist_ok=False)
        if self.behavior == "die_on_advance":
            os._exit(71)
        if self.behavior == "sleep_on_advance":
            time.sleep(60)
        results: list[dict[str, object]] = []
        for state in expected_states:
            existing_tick = self.ticks.get(state.task_id)
            existing_pins = self.pins.get(state.task_id)
            if existing_tick is None:
                existing_tick = state.observed_tick
                existing_pins = state.aggregate_resume_pins
            if (
                existing_tick != state.observed_tick
                or existing_pins != state.aggregate_resume_pins
            ):
                raise OpenEcologyProcessWorkerError(
                    "protocol test runtime external state mismatch"
                )
            observed_tick = (
                target_tick - 1 if self.behavior == "partial_frontier" else target_tick
            )
            pins = _test_pins(
                task_id=state.task_id,
                source_git_sha=self.source_git_sha,
                observed_tick=observed_tick,
                campaign_id=self.campaign_id,
            )
            self.ticks[state.task_id] = observed_tick
            self.pins[state.task_id] = pins
            task = self.tasks[state.task_id]
            quiescent = {
                "task_id": state.task_id,
                "campaign_id": self.campaign_id,
                "campaign_root": str(self.campaign_root),
                "source_git_sha": self.source_git_sha,
                "observed_tick": observed_tick,
                "archive_safe_while_runner_idle": True,
                "restartable_world_checkpoint_present": True,
                "aggregate_current_restartable": True,
                "aggregate_resume_authorized": True,
                "boundary_sha256": _stable_digest(
                    {"task_id": state.task_id, "tick": observed_tick}
                ),
            }
            frontier = {
                "schema_version": ("mind_v3_open_ecology_campaign_barrier_frontier_v1"),
                "task_id": state.task_id,
                "task_sha256": _stable_digest(task.to_dict()),
                "campaign_id": self.campaign_id,
                "campaign_root": str(self.campaign_root),
                "source_git_sha": self.source_git_sha,
                "observed_tick": observed_tick,
                "extinct": False,
                "extinction_tick": None,
                "config_contract_sha256": pins.identity.config_contract_sha256,
                "seed_contract_sha256": pins.identity.seed_contract_sha256,
                "world_config_sha256": task.world_config_sha256,
                "aggregate_generation_index": pins.aggregate_generation_index,
                "aggregate_completed_world_tick": pins.identity.tick,
                "aggregate_commit_sha256": pins.commit_sha256,
                "checkpoint_sha256": pins.checkpoint_sha256,
                "checkpoint_generation_identity_sha256": (
                    pins.checkpoint_generation_identity_sha256
                ),
                "evidence_manifest_sha256": pins.evidence_manifest_sha256,
                "evidence_manifest_status": pins.evidence_manifest_status,
                "evidence_continuation_sha256": "8" * 64,
                "aggregate_current_restartable": True,
                "aggregate_resume_authorized": True,
            }
            frontier["frontier_sha256"] = _stable_digest(frontier)
            results.append(
                _task_result_payload(
                    task_id=state.task_id,
                    worker_index=self.slot.worker_index,
                    requested_target_tick=target_tick,
                    observed_tick=observed_tick,
                    extinct=False,
                    extinction_tick=None,
                    model_state_sha256="4" * 64,
                    same_world_instance=True,
                    summaries=(),
                    milestones=(),
                    quiescent_checkpoint=quiescent,
                    campaign_barrier_frontier=frontier,
                    aggregate_resume_pins=pins,
                )
            )
        return tuple(results)

    def reconcile(
        self,
        *,
        expected_states: tuple[ProcessWorkerExpectedState, ...],
        target_tick: int,
        intent_sha256: str,
        attempt_authority_sha256_by_task: Mapping[str, str],
    ) -> tuple[tuple[dict[str, object], ...], dict[str, str]]:
        _sha256(intent_sha256, field="protocol reconciliation intent_sha256")
        selected = tuple(state.task_id for state in expected_states)
        if set(attempt_authority_sha256_by_task) != set(selected):
            raise OpenEcologyProcessWorkerError(
                "protocol reconciliation authority matrix is incomplete"
            )
        for task_id in selected:
            _sha256(
                attempt_authority_sha256_by_task[task_id],
                field=f"{task_id}.protocol reconciliation authority",
            )
        completed = self.advance(
            expected_states=expected_states,
            target_tick=target_tick,
        )
        disposition_names = (
            "already_committed_target",
            "advanced_from_exact_predecessor",
            "recovered_original_precurrent_attempt",
        )
        dispositions = {
            task_id: disposition_names[index % len(disposition_names)]
            for index, task_id in enumerate(selected)
        }
        if self.behavior == "reconcile_forged":
            first = dict(completed[0])
            first["requested_target_tick"] = target_tick - _FRONTIER_INTERVAL
            completed = (first, *completed[1:])
        return completed, dispositions


class _PersistentIslandRuntime:
    def __init__(
        self,
        *,
        slot: ProcessWorkerSlot,
        tasks: Mapping[str, PersistentIslandTask],
        campaign_root: Path,
        campaign_id: str,
        checkpoint_interval_ticks: int = (
            OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS
        ),
    ) -> None:
        self.slot = slot
        self.tasks = tasks
        self.campaign_root = campaign_root
        self.campaign_id = campaign_id
        self.checkpoint_interval_ticks = _positive_int(
            checkpoint_interval_ticks,
            field="persistent runtime checkpoint_interval_ticks",
        )
        self.runners: dict[str, PersistentIslandRunner] = {}

    def advance(
        self,
        *,
        expected_states: tuple[ProcessWorkerExpectedState, ...],
        target_tick: int,
    ) -> tuple[dict[str, object], ...]:
        results: list[dict[str, object]] = []
        for expected in expected_states:
            task = self.tasks[expected.task_id]
            runner = self.runners.get(expected.task_id)
            if runner is None:
                if expected.observed_tick == 0:
                    runner = PersistentIslandRunner.open(
                        task,
                        evidence_directory=self.campaign_root / task.task_id,
                        campaign_root=self.campaign_root,
                        campaign_id=self.campaign_id,
                        checkpoint_interval_ticks=self.checkpoint_interval_ticks,
                    )
                else:
                    pins = expected.aggregate_resume_pins
                    if pins is None:
                        raise OpenEcologyProcessWorkerError(
                            "started worker task is missing external resume pins"
                        )
                    runner = PersistentIslandRunner.restore_from_current(
                        task,
                        campaign_root=self.campaign_root,
                        campaign_id=self.campaign_id,
                        pins=pins,
                        evidence_directory=self.campaign_root / task.task_id,
                        aggregate_directory=(
                            self.campaign_root / "aggregates" / task.task_id
                        ),
                    )
                self.runners[expected.task_id] = runner
            _assert_runner_matches_expected(runner, expected)
            advance_result = runner.advance_to(target_tick)
            pins = runner.latest_aggregate_resume_pins
            if pins is None:
                raise OpenEcologyProcessWorkerError(
                    "worker task produced no restartable aggregate pins"
                )
            results.append(
                _task_result_payload(
                    task_id=expected.task_id,
                    worker_index=self.slot.worker_index,
                    requested_target_tick=advance_result.requested_target_tick,
                    observed_tick=advance_result.observed_tick,
                    extinct=advance_result.extinct,
                    extinction_tick=advance_result.extinction_tick,
                    model_state_sha256=advance_result.model_state_sha256,
                    same_world_instance=advance_result.same_world_instance,
                    summaries=advance_result.summaries,
                    milestones=advance_result.milestones,
                    quiescent_checkpoint=runner.quiescent_checkpoint,
                    campaign_barrier_frontier=runner.campaign_barrier_frontier(),
                    aggregate_resume_pins=pins,
                )
            )
        return tuple(results)

    def reconcile(
        self,
        *,
        expected_states: tuple[ProcessWorkerExpectedState, ...],
        target_tick: int,
        intent_sha256: str,
        attempt_authority_sha256_by_task: Mapping[str, str],
    ) -> tuple[tuple[dict[str, object], ...], dict[str, str]]:
        _sha256(intent_sha256, field="runtime reconciliation intent_sha256")
        selected = tuple(expected.task_id for expected in expected_states)
        if set(attempt_authority_sha256_by_task) != set(selected):
            raise OpenEcologyProcessWorkerError(
                "runtime reconciliation authority matrix is incomplete"
            )
        results: list[dict[str, object]] = []
        dispositions: dict[str, str] = {}
        for expected in expected_states:
            task = self.tasks[expected.task_id]
            runner, disposition = PersistentIslandRunner.reconcile_interval_attempt(
                task,
                campaign_root=self.campaign_root,
                campaign_id=self.campaign_id,
                predecessor=expected.aggregate_resume_pins,
                predecessor_observed_tick=expected.observed_tick,
                predecessor_terminal=expected.terminal_extinct,
                target_tick=target_tick,
                attempt_authority_sha256=_sha256(
                    attempt_authority_sha256_by_task[expected.task_id],
                    field=f"{expected.task_id}.runtime reconciliation authority",
                ),
                evidence_directory=self.campaign_root / task.task_id,
                aggregate_directory=(self.campaign_root / "aggregates" / task.task_id),
                checkpoint_interval_ticks=self.checkpoint_interval_ticks,
            )
            if (
                disposition
                not in OPEN_ECOLOGY_PROCESS_WORKER_RECONCILIATION_DISPOSITIONS
            ):
                raise OpenEcologyProcessWorkerError(
                    f"runtime reconciliation disposition drifted for {expected.task_id}"
                )
            _assert_reconciled_runner(
                runner,
                expected=expected,
                target_tick=target_tick,
            )
            pins = runner.latest_aggregate_resume_pins
            if pins is None:
                raise OpenEcologyProcessWorkerError(
                    "reconciled worker task produced no restartable aggregate pins"
                )
            milestone = runner.first_milestone
            results.append(
                _task_result_payload(
                    task_id=expected.task_id,
                    worker_index=self.slot.worker_index,
                    requested_target_tick=target_tick,
                    observed_tick=runner.observed_tick,
                    extinct=runner.extinct,
                    extinction_tick=(runner.observed_tick if runner.extinct else None),
                    model_state_sha256=runner.model_state_sha256,
                    same_world_instance=True,
                    summaries=runner.summaries,
                    milestones=(() if milestone is None else (milestone,)),
                    quiescent_checkpoint=runner.quiescent_checkpoint,
                    campaign_barrier_frontier=runner.campaign_barrier_frontier(),
                    aggregate_resume_pins=pins,
                )
            )
            dispositions[expected.task_id] = disposition
            self.runners[expected.task_id] = runner
        return tuple(results), dispositions


def _worker_process_main(
    connection: Connection,
    init_bytes: bytes,
    max_message_bytes: int,
) -> None:
    try:
        init_payload = _decode_message(
            init_bytes,
            max_message_bytes=max_message_bytes,
        )
        _exact_keys(
            init_payload,
            {
                "message_type",
                "worker_index",
                "slot",
                "campaign_root",
                "campaign_id",
                "repository_root",
                "source_git_sha",
                "source_manifest_sha256",
                "git_authority",
                "tasks",
                "assignment_sha256",
                "backend",
                "test_behavior",
                "max_message_bytes",
            },
            field="worker initialize",
        )
        if init_payload["message_type"] != "initialize":
            raise OpenEcologyProcessWorkerError(
                "worker initialization message type drifted"
            )
        slot = _slot_from_dict(
            _mapping(init_payload["slot"], field="worker initialize slot")
        )
        if slot.worker_index != init_payload["worker_index"]:
            raise OpenEcologyProcessWorkerError(
                "worker initialization slot identity drifted"
            )
        campaign_root = _absolute_directory(
            init_payload["campaign_root"],
            field="worker campaign_root",
        )
        repository_root = _absolute_directory(
            init_payload["repository_root"],
            field="worker repository_root",
        )
        campaign_id = _identifier(
            init_payload["campaign_id"],
            field="worker campaign_id",
        )
        source_git_sha = _git_sha(
            init_payload["source_git_sha"],
            field="worker source_git_sha",
        )
        source_manifest_sha256 = _sha256(
            init_payload["source_manifest_sha256"],
            field="worker source_manifest_sha256",
        )
        raw_git_authority = init_payload["git_authority"]
        assignment_sha256 = _sha256(
            init_payload["assignment_sha256"],
            field="worker assignment_sha256",
        )
        if init_payload[
            "max_message_bytes"
        ] != max_message_bytes or max_message_bytes != _bounded_message_bytes(
            max_message_bytes
        ):
            raise OpenEcologyProcessWorkerError(
                "worker maximum message byte contract drifted"
            )
        task_records = _sequence(
            init_payload["tasks"],
            field="worker tasks",
        )
        tasks = tuple(
            _task_from_dict(_mapping(item, field="worker task"))
            for item in task_records
        )
        if tuple(task.task_id for task in tasks) != slot.task_ids:
            raise OpenEcologyProcessWorkerError(
                "worker static task queue/specification drifted"
            )
        task_by_id = {task.task_id: task for task in tasks}
        backend = _string(init_payload["backend"], field="worker backend")
        behavior = _string(
            init_payload["test_behavior"],
            field="worker test_behavior",
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _configure_torch_threads(slot.torch_threads)

        if backend == "persistent_island_v1":
            if behavior != "normal":
                raise OpenEcologyProcessWorkerError(
                    "production worker cannot carry test behavior"
                )
            git_authority = _validate_git_authority(
                _mapping(
                    raw_git_authority,
                    field="worker Git authority",
                ),
                revalidate=True,
            )

            def source_probe() -> dict[str, object]:
                return _strict_source_snapshot(
                    repository_root=repository_root,
                    source_git_sha=source_git_sha,
                    source_manifest_sha256=source_manifest_sha256,
                    git_authority=git_authority,
                )

            runtime: _PersistentIslandRuntime | _ProtocolTestRuntime = (
                _PersistentIslandRuntime(
                    slot=slot,
                    tasks=task_by_id,
                    campaign_root=campaign_root,
                    campaign_id=campaign_id,
                )
            )
        elif backend == "protocol_test_v1":
            if raw_git_authority is not None:
                raise OpenEcologyProcessWorkerError(
                    "protocol test worker cannot carry production Git authority"
                )

            def source_probe() -> dict[str, object]:
                return _test_source_snapshot(
                    repository_root=repository_root,
                    source_git_sha=source_git_sha,
                    source_manifest_sha256=source_manifest_sha256,
                    mismatch=(behavior == "source_mismatch_startup"),
                )

            runtime = _ProtocolTestRuntime(
                slot=slot,
                tasks=task_by_id,
                campaign_root=campaign_root,
                campaign_id=campaign_id,
                source_git_sha=source_git_sha,
                behavior=behavior,
            )
        else:
            raise OpenEcologyProcessWorkerError("worker backend is not authorized")

        source = source_probe()
        _worker_send(
            connection,
            {
                "message_type": "ready",
                "worker_index": slot.worker_index,
                "worker_pid": os.getpid(),
                "host_identity": local_process_worker_host_identity(),
                "device_kind": slot.device_kind,
                "device_index": slot.device_index,
                "torch_threads": slot.torch_threads,
                "source": source,
                "assignment_sha256": assignment_sha256,
            },
            max_message_bytes=max_message_bytes,
        )
        while True:
            try:
                command = _worker_receive(
                    connection,
                    max_message_bytes=max_message_bytes,
                )
            except EOFError:
                # Parent loss while idle carries no authorization and requires no
                # campaign-root cleanup. The child exits without deleting evidence.
                return
            message_type = command.get("message_type")
            sequence_index = _positive_int(
                command.get("sequence_index"),
                field="worker command sequence_index",
            )
            if message_type == "shutdown":
                _exact_keys(
                    command,
                    {"message_type", "sequence_index"},
                    field="worker shutdown",
                )
                _worker_send(
                    connection,
                    {
                        "message_type": "shutdown_ack",
                        "sequence_index": sequence_index,
                        "worker_index": slot.worker_index,
                        "worker_pid": os.getpid(),
                        "evidence_deleted": False,
                    },
                    max_message_bytes=max_message_bytes,
                )
                return
            if message_type not in {"advance", "reconcile"}:
                raise OpenEcologyProcessWorkerError("worker received unknown command")
            if message_type == "advance":
                _exact_keys(
                    command,
                    {
                        "message_type",
                        "sequence_index",
                        "target_tick",
                        "expected_states",
                    },
                    field="worker advance",
                )
                intent_sha256 = None
                attempt_authorities: dict[str, str] = {}
            else:
                _exact_keys(
                    command,
                    {
                        "message_type",
                        "sequence_index",
                        "target_tick",
                        "expected_states",
                        "intent_sha256",
                        "attempt_authority_sha256_by_task",
                    },
                    field="worker reconcile",
                )
                intent_sha256 = _sha256(
                    command["intent_sha256"],
                    field="worker reconciliation intent_sha256",
                )
                raw_authorities = _mapping(
                    command["attempt_authority_sha256_by_task"],
                    field="worker reconciliation attempt authorities",
                )
                attempt_authorities = {
                    _identifier(task_id, field="worker reconciliation task_id"): (
                        _sha256(
                            digest,
                            field=f"{task_id}.worker reconciliation authority",
                        )
                    )
                    for task_id, digest in raw_authorities.items()
                }
            target_tick = _frontier_tick(
                command["target_tick"],
                field="worker target_tick",
            )
            expected_states = tuple(
                _expected_state_from_dict(_mapping(item, field="worker expected state"))
                for item in _sequence(
                    command["expected_states"],
                    field="worker expected states",
                )
            )
            selected_ids = tuple(state.task_id for state in expected_states)
            if (
                not selected_ids
                or len(set(selected_ids)) != len(selected_ids)
                or tuple(
                    task_id for task_id in slot.task_ids if task_id in set(selected_ids)
                )
                != selected_ids
            ):
                raise OpenEcologyProcessWorkerError(
                    "worker command violates its immutable static queue"
                )
            if message_type == "reconcile" and set(attempt_authorities) != set(
                selected_ids
            ):
                raise OpenEcologyProcessWorkerError(
                    "worker reconciliation authority matrix is incomplete"
                )
            source_before = source_probe()
            completed: tuple[dict[str, object], ...] = ()
            try:
                if message_type == "advance":
                    completed = runtime.advance(
                        expected_states=expected_states,
                        target_tick=target_tick,
                    )
                    dispositions: dict[str, str] = {}
                else:
                    if intent_sha256 is None:
                        raise OpenEcologyProcessWorkerError(
                            "worker reconciliation intent is missing"
                        )
                    completed, dispositions = runtime.reconcile(
                        expected_states=expected_states,
                        target_tick=target_tick,
                        intent_sha256=intent_sha256,
                        attempt_authority_sha256_by_task=attempt_authorities,
                    )
                if backend == "protocol_test_v1" and behavior == (
                    "source_mismatch_after"
                ):
                    source_after = _test_source_snapshot(
                        repository_root=repository_root,
                        source_git_sha=source_git_sha,
                        source_manifest_sha256=source_manifest_sha256,
                        mismatch=True,
                    )
                else:
                    source_after = source_probe()
                _worker_send(
                    connection,
                    {
                        "message_type": (
                            "advance_result"
                            if message_type == "advance"
                            else "reconcile_result"
                        ),
                        "sequence_index": sequence_index,
                        "worker_index": slot.worker_index,
                        "results": list(completed),
                        "status": {
                            "worker_index": slot.worker_index,
                            "worker_pid": os.getpid(),
                            "host_identity": local_process_worker_host_identity(),
                            "device_kind": slot.device_kind,
                            "device_index": slot.device_index,
                            "torch_threads": slot.torch_threads,
                            "source_before": source_before,
                            "source_after": source_after,
                            "completed_task_ids": [
                                result["task_id"] for result in completed
                            ],
                        },
                        **(
                            {}
                            if message_type == "advance"
                            else {
                                "intent_sha256": intent_sha256,
                                "disposition_by_task": dispositions,
                            }
                        ),
                    },
                    max_message_bytes=max_message_bytes,
                )
            except Exception as error:
                _worker_send(
                    connection,
                    {
                        "message_type": "error",
                        "sequence_index": sequence_index,
                        "worker_index": slot.worker_index,
                        "error_message": (
                            f"{type(error).__name__}: {str(error)[:2048]}"
                        ),
                        "completed_results": list(completed),
                    },
                    max_message_bytes=max_message_bytes,
                )
                return
    except BaseException:
        try:
            connection.close()
        finally:
            raise
    finally:
        try:
            connection.close()
        except OSError:
            pass


def _task_result_payload(
    *,
    task_id: str,
    worker_index: int,
    requested_target_tick: int,
    observed_tick: int,
    extinct: bool,
    extinction_tick: int | None,
    model_state_sha256: str,
    same_world_instance: bool,
    summaries: Sequence[Mapping[str, object]],
    milestones: Sequence[Mapping[str, object]],
    quiescent_checkpoint: Mapping[str, object],
    campaign_barrier_frontier: Mapping[str, object],
    aggregate_resume_pins: OpenEcologyAggregateResumePins,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PROCESS_WORKER_RESULT_SCHEMA_VERSION,
        "task_id": task_id,
        "worker_index": worker_index,
        "worker_pid": os.getpid(),
        "requested_target_tick": requested_target_tick,
        "observed_tick": observed_tick,
        "extinct": extinct,
        "extinction_tick": extinction_tick,
        "model_state_sha256": model_state_sha256,
        "same_world_instance": same_world_instance,
        "summary_count": len(summaries),
        "summary_sha256": _stable_digest(list(summaries)),
        "milestone_count": len(milestones),
        "milestone_sha256": _stable_digest(list(milestones)),
        "quiescent_checkpoint": _json_clone_mapping(quiescent_checkpoint),
        "campaign_barrier_frontier": _json_clone_mapping(campaign_barrier_frontier),
        "aggregate_resume_pins": _pins_to_dict(aggregate_resume_pins),
    }
    payload["result_sha256"] = _stable_digest(payload)
    return payload


def _assert_runner_matches_expected(
    runner: PersistentIslandRunner,
    expected: ProcessWorkerExpectedState,
) -> None:
    runner_pins = runner.latest_aggregate_resume_pins
    if (
        expected.observed_tick == 0
        and expected.aggregate_resume_pins is None
        and runner.task.task_id == expected.task_id
        and runner.observed_tick == 0
        and not runner.extinct
        and runner_pins is not None
        and _is_exact_runner_genesis_pins(runner, runner_pins)
    ):
        return
    if (
        runner.task.task_id != expected.task_id
        or runner.observed_tick != expected.observed_tick
        or runner.extinct is not expected.terminal_extinct
        or runner_pins != expected.aggregate_resume_pins
    ):
        raise OpenEcologyProcessWorkerError(
            f"child-owned runner state drifted for {expected.task_id}"
        )


def _assert_reconciled_runner(
    runner: PersistentIslandRunner,
    *,
    expected: ProcessWorkerExpectedState,
    target_tick: int,
) -> None:
    pins = runner.latest_aggregate_resume_pins
    if runner.task.task_id != expected.task_id or pins is None:
        raise OpenEcologyProcessWorkerError(
            f"reconciled runner identity drifted for {expected.task_id}"
        )
    if expected.terminal_extinct:
        if (
            not runner.extinct
            or runner.observed_tick != expected.observed_tick
            or pins != expected.aggregate_resume_pins
        ):
            raise OpenEcologyProcessWorkerError(
                f"reconciled terminal predecessor drifted for {expected.task_id}"
            )
    elif runner.extinct:
        if not expected.observed_tick <= runner.observed_tick <= target_tick:
            raise OpenEcologyProcessWorkerError(
                f"reconciled extinction frontier drifted for {expected.task_id}"
            )
    elif runner.observed_tick != target_tick:
        raise OpenEcologyProcessWorkerError(
            f"reconciled runner missed target for {expected.task_id}"
        )
    expected_completed_tick = (
        0 if runner.observed_tick == 0 else runner.observed_tick - 1
    )
    if (
        pins.identity.source_git_sha != runner.task.artifact.source_commit
        or pins.identity.island_id != expected.task_id
        or pins.identity.tick != expected_completed_tick
    ):
        raise OpenEcologyProcessWorkerError(
            f"reconciled aggregate pins drifted for {expected.task_id}"
        )


def _is_exact_runner_genesis_pins(
    runner: PersistentIslandRunner,
    pins: OpenEcologyAggregateResumePins,
) -> bool:
    campaign_id = getattr(runner, "_campaign_id", None)
    identity = pins.identity
    frontier = runner.campaign_barrier_frontier()
    run_generation_id = (
        None
        if not isinstance(campaign_id, str)
        else (
            f"phase-d:{runner.task.task_id}:"
            f"{hashlib.sha256(campaign_id.encode('utf-8')).hexdigest()[:16]}"
        )
    )
    return bool(
        isinstance(campaign_id, str)
        and identity.source_git_sha == runner.task.artifact.source_commit
        and identity.run_generation_id == run_generation_id
        and identity.island_id == runner.task.task_id
        and identity.simulation_generation_index == 0
        and identity.tick == 0
        and pins.aggregate_generation_index == 0
        and pins.evidence_manifest_status == "open"
        and frontier.get("task_id") == runner.task.task_id
        and frontier.get("task_sha256") == _stable_digest(runner.task.to_dict())
        and frontier.get("observed_tick") == 0
        and frontier.get("config_contract_sha256") == identity.config_contract_sha256
        and frontier.get("seed_contract_sha256") == identity.seed_contract_sha256
        and frontier.get("aggregate_generation_index") == 0
        and frontier.get("aggregate_completed_world_tick") == 0
        and frontier.get("aggregate_commit_sha256") == pins.commit_sha256
        and frontier.get("checkpoint_sha256") == pins.checkpoint_sha256
    )


def _expected_matches_result(
    expected: ProcessWorkerExpectedState,
    result: ProcessWorkerTaskResult,
) -> bool:
    return bool(
        expected.task_id == result.task_id
        and expected.observed_tick == result.observed_tick
        and expected.terminal_extinct is result.extinct
        and expected.aggregate_resume_pins == result.aggregate_resume_pins
    )


def _strict_source_snapshot(
    *,
    repository_root: Path,
    source_git_sha: str,
    source_manifest_sha256: str,
    git_authority: Mapping[str, object],
) -> dict[str, object]:
    imported_root = Path(__file__).resolve().parents[3]
    if imported_root != repository_root:
        raise OpenEcologyProcessWorkerError(
            "worker import root does not match launch repository root"
        )

    authority = _validate_git_authority(git_authority, revalidate=True)

    def git(*arguments: str) -> str:
        return _run_pinned_git(
            authority,
            repository_root=repository_root,
            arguments=arguments,
        )

    observed_manifest = source_file_hash_manifest(repository_root)
    snapshot = {
        "repository_root": str(repository_root),
        "source_git_sha": git("rev-parse", "HEAD"),
        "source_manifest_sha256": observed_manifest.get("aggregate_sha256"),
        "git_clean": not bool(git("status", "--porcelain=v1", "--untracked-files=all")),
    }
    expected = {
        "repository_root": str(repository_root),
        "source_git_sha": source_git_sha,
        "source_manifest_sha256": source_manifest_sha256,
        "git_clean": True,
    }
    if snapshot != expected:
        raise OpenEcologyProcessWorkerError(
            "worker live source does not match launch authority"
        )
    return snapshot


def strict_open_ecology_source_snapshot(
    *,
    repository_root: Path,
    source_git_sha: str,
    source_manifest_sha256: str,
    git_executable: Path,
    git_executable_sha256: str,
) -> dict[str, object]:
    """Validate an exact checkout through one externally pinned Git binary."""

    authority = pin_open_ecology_git_executable(
        git_executable,
        expected_sha256=git_executable_sha256,
    )
    return _strict_source_snapshot(
        repository_root=repository_root,
        source_git_sha=source_git_sha,
        source_manifest_sha256=source_manifest_sha256,
        git_authority=authority,
    )


def pin_open_ecology_git_executable(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    expected = _sha256(expected_sha256, field="Git executable SHA256")
    if not path.is_absolute():
        raise OpenEcologyProcessWorkerError("Git executable path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise OpenEcologyProcessWorkerError(
            "Git executable cannot be resolved"
        ) from error
    try:
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise OpenEcologyProcessWorkerError(
            "Git executable cannot be opened safely"
        ) from error
    try:
        before = os.fstat(descriptor)
        mode = before.st_mode & 0o7777
        if (
            not stat.S_ISREG(before.st_mode)
            or not (mode & 0o111)
            or mode & 0o022
            or before.st_size <= 0
            or before.st_size > OPEN_ECOLOGY_GIT_EXECUTABLE_MAX_BYTES
        ):
            raise OpenEcologyProcessWorkerError(
                "Git executable violates regular executable contract"
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    observed = digest.hexdigest()
    if (
        identity_before != identity_after
        or total != before.st_size
        or observed != expected
    ):
        raise OpenEcologyProcessWorkerError(
            "Git executable identity or external SHA256 authority drifted"
        )
    authority: dict[str, object] = {
        "path": str(resolved),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": mode,
        "link_count": before.st_nlink,
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "sha256": observed,
    }
    authority["authority_sha256"] = _stable_digest(authority)
    return authority


def _validate_git_authority(
    value: Mapping[str, object],
    *,
    revalidate: bool,
) -> dict[str, object]:
    required = {
        "path",
        "device",
        "inode",
        "mode",
        "link_count",
        "size",
        "mtime_ns",
        "ctime_ns",
        "sha256",
        "authority_sha256",
    }
    if set(value) != required:
        raise OpenEcologyProcessWorkerError("Git authority schema drifted")
    unsigned = {key: value[key] for key in required if key != "authority_sha256"}
    if value.get("authority_sha256") != _stable_digest(unsigned):
        raise OpenEcologyProcessWorkerError("Git authority digest drifted")
    normalized = dict(value)
    if revalidate:
        current = pin_open_ecology_git_executable(
            Path(str(value["path"])),
            expected_sha256=_sha256(
                value["sha256"],
                field="Git authority sha256",
            ),
        )
        if current != normalized:
            raise OpenEcologyProcessWorkerError("Git executable identity changed")
    return normalized


def _run_pinned_git(
    git_authority: Mapping[str, object],
    *,
    repository_root: Path,
    arguments: Sequence[str],
) -> str:
    authority = _validate_git_authority(git_authority, revalidate=True)
    command = [
        str(authority["path"]),
        "-C",
        str(repository_root),
        *arguments,
    ]
    stdout, stderr, returncode = _run_bounded_git(
        command,
        environment={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": OPEN_ECOLOGY_FIXED_COMMAND_PATH,
        },
    )
    if (
        returncode != 0
        or stderr
        or len(stdout) > OPEN_ECOLOGY_GIT_OUTPUT_MAX_BYTES
        or len(stderr) > OPEN_ECOLOGY_GIT_OUTPUT_MAX_BYTES
    ):
        raise OpenEcologyProcessWorkerError(
            "worker Git source authority command failed"
        )
    if _validate_git_authority(git_authority, revalidate=True) != authority:
        raise OpenEcologyProcessWorkerError(
            "Git executable changed during source verification"
        )
    try:
        return stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise OpenEcologyProcessWorkerError(
            "worker Git source authority output is not UTF-8"
        ) from error


def _run_bounded_git(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
) -> tuple[bytes, bytes, int]:
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=dict(environment),
            start_new_session=True,
        )
    except OSError as error:
        raise OpenEcologyProcessWorkerError(
            "worker cannot verify Git source authority"
        ) from error
    if process.stdout is None or process.stderr is None:
        _terminate_git_process(process)
        raise OpenEcologyProcessWorkerError(
            "worker Git source authority pipes were not created"
        )
    selector = selectors.DefaultSelector()
    buffers: dict[str, bytearray] = {
        "stdout": bytearray(),
        "stderr": bytearray(),
    }
    streams = {
        process.stdout.fileno(): ("stdout", process.stdout),
        process.stderr.fileno(): ("stderr", process.stderr),
    }
    for descriptor, (name, _) in streams.items():
        os.set_blocking(descriptor, False)
        selector.register(descriptor, selectors.EVENT_READ, data=name)
    deadline = time.monotonic() + 30.0
    total_bytes = 0
    group_cleanup_attempted = False
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OpenEcologyProcessWorkerError(
                    "worker Git source authority command timed out"
                )
            events = selector.select(timeout=remaining)
            if not events:
                raise OpenEcologyProcessWorkerError(
                    "worker Git source authority command timed out"
                )
            for key, _ in events:
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                total_bytes += len(chunk)
                if total_bytes > OPEN_ECOLOGY_GIT_OUTPUT_MAX_BYTES:
                    raise OpenEcologyProcessWorkerError(
                        "worker Git source authority output exceeded its bound"
                    )
                buffers[str(key.data)].extend(chunk)
        try:
            wait_for_leader_exit_without_reaping(process, deadline=deadline)
        except TimeoutError as error:
            raise OpenEcologyProcessWorkerError(
                "worker Git source authority command timed out"
            ) from error
        group_cleanup_attempted = True
        returncode = _terminate_git_process(
            process,
            leader_exit_observed=True,
        )
    except BaseException as primary_error:
        if not group_cleanup_attempted:
            try:
                _terminate_git_process(process)
            except OpenEcologyProcessWorkerError as cleanup_error:
                primary_error.add_note(f"additional cleanup failure: {cleanup_error}")
                raise primary_error from cleanup_error
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return bytes(buffers["stdout"]), bytes(buffers["stderr"]), returncode


def _terminate_git_process(
    process: subprocess.Popen[bytes],
    *,
    leader_exit_observed: bool = False,
) -> int:
    try:
        return terminate_process_group_before_reap(
            process,
            wait_timeout_seconds=2.0,
            leader_exit_observed=leader_exit_observed,
        )
    except OpenEcologyProcessGroupError as error:
        raise OpenEcologyProcessWorkerError(
            "worker Git source authority process group survived termination"
        ) from error


def _test_source_snapshot(
    *,
    repository_root: Path,
    source_git_sha: str,
    source_manifest_sha256: str,
    mismatch: bool,
) -> dict[str, object]:
    return {
        "repository_root": str(repository_root),
        "source_git_sha": ("0" * 40 if mismatch else source_git_sha),
        "source_manifest_sha256": source_manifest_sha256,
        "git_clean": True,
    }


def _configure_torch_threads(torch_threads: int) -> None:
    try:
        import torch

        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # The setting is process-global and may already be fixed by import.
            if torch.get_num_interop_threads() != 1:
                raise
    except ImportError as error:
        raise OpenEcologyProcessWorkerError(
            "process worker requires the Mind ML Torch runtime"
        ) from error


def _assignment_records(
    tasks: Sequence[PersistentIslandTask],
    slots: Sequence[ProcessWorkerSlot],
) -> tuple[dict[str, object], ...]:
    task_order = {task.task_id: index for index, task in enumerate(tasks)}
    records: list[dict[str, object]] = []
    for slot in slots:
        for queue_index, task_id in enumerate(slot.task_ids):
            if task_id not in task_order:
                raise OpenEcologyProcessWorkerError(
                    "process worker slot names a task outside launch authority"
                )
            records.append(
                {
                    "task_id": task_id,
                    "worker_index": slot.worker_index,
                    "worker_queue_index": queue_index,
                    "global_queue_index": task_order[task_id],
                    "qualification_priority": task_order[task_id] < 3,
                }
            )
    records.sort(key=lambda item: int(item["global_queue_index"]))
    if tuple(int(item["global_queue_index"]) for item in records) != tuple(
        range(len(tasks))
    ):
        raise OpenEcologyProcessWorkerError(
            "process worker assignment does not cover task order exactly"
        )
    return tuple(records)


def _assignment_sha256(records: Sequence[Mapping[str, object]]) -> str:
    """Match the coordinator/storage canonical newline-bearing digest."""

    return hashlib.sha256(canonical_json_bytes(list(records))).hexdigest()


def _ordered_selected_tasks(
    tasks: Sequence[PersistentIslandTask],
    selected_task_ids: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(selected_task_ids, (str, bytes, bytearray)):
        raise OpenEcologyProcessWorkerError("selected_task_ids must be a task sequence")
    selected = tuple(
        _identifier(task_id, field="selected_task_ids[]")
        for task_id in selected_task_ids
    )
    if not selected or len(set(selected)) != len(selected):
        raise OpenEcologyProcessWorkerError(
            "selected task scope is empty or duplicated"
        )
    authority = tuple(task.task_id for task in tasks)
    unknown = sorted(set(selected) - set(authority))
    if unknown:
        raise OpenEcologyProcessWorkerError(
            "selected task scope is outside authority: " + ",".join(unknown)
        )
    ordered = tuple(task_id for task_id in authority if task_id in set(selected))
    if selected != ordered:
        raise OpenEcologyProcessWorkerError(
            "selected tasks must use immutable matrix order"
        )
    return selected


def _task_from_dict(payload: Mapping[str, object]) -> PersistentIslandTask:
    task_payload = dict(payload)
    artifact_payload = _mapping(
        task_payload.pop("artifact", None),
        field="worker task artifact",
    )
    try:
        artifact = PersistentArtifactBinding(**artifact_payload)
        return PersistentIslandTask(artifact=artifact, **task_payload)
    except (TypeError, ValueError) as error:
        raise OpenEcologyProcessWorkerError(
            "worker task payload is not the exact persistent task schema"
        ) from error


def _slot_from_dict(payload: Mapping[str, object]) -> ProcessWorkerSlot:
    _exact_keys(
        payload,
        {
            "worker_index",
            "host_identity",
            "device_kind",
            "device_index",
            "torch_threads",
            "task_ids",
        },
        field="worker slot",
    )
    return ProcessWorkerSlot(
        worker_index=_nonnegative_int(
            payload["worker_index"],
            field="worker slot worker_index",
        ),
        host_identity=_string(
            payload["host_identity"],
            field="worker slot host_identity",
        ),
        device_kind=_string(
            payload["device_kind"],
            field="worker slot device_kind",
        ),
        device_index=payload["device_index"],
        torch_threads=_positive_int(
            payload["torch_threads"],
            field="worker slot torch_threads",
        ),
        task_ids=_string_tuple(payload["task_ids"], field="worker slot task_ids"),
    )


def _expected_state_from_dict(
    payload: Mapping[str, object],
) -> ProcessWorkerExpectedState:
    _exact_keys(
        payload,
        {
            "task_id",
            "observed_tick",
            "terminal_extinct",
            "aggregate_resume_pins",
        },
        field="worker expected state",
    )
    raw_pins = payload["aggregate_resume_pins"]
    pins = (
        None
        if raw_pins is None
        else _pins_from_dict(_mapping(raw_pins, field="worker expected aggregate pins"))
    )
    return ProcessWorkerExpectedState(
        task_id=_string(payload["task_id"], field="worker expected task_id"),
        observed_tick=_nonnegative_int(
            payload["observed_tick"],
            field="worker expected observed_tick",
        ),
        terminal_extinct=_boolean(
            payload["terminal_extinct"],
            field="worker expected terminal_extinct",
        ),
        aggregate_resume_pins=pins,
    )


def _pins_to_dict(pins: OpenEcologyAggregateResumePins) -> dict[str, object]:
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


def _pins_from_dict(payload: Mapping[str, object]) -> OpenEcologyAggregateResumePins:
    _exact_keys(
        payload,
        {
            "identity",
            "aggregate_generation_index",
            "commit_sha256",
            "checkpoint_sha256",
            "checkpoint_generation_identity_sha256",
            "evidence_manifest_sha256",
            "evidence_manifest_status",
        },
        field="aggregate resume pins",
    )
    identity_payload = _mapping(
        payload["identity"],
        field="aggregate resume identity",
    )
    _exact_keys(
        identity_payload,
        {
            "source_git_sha",
            "config_contract_sha256",
            "seed_contract_sha256",
            "run_generation_id",
            "island_id",
            "simulation_generation_index",
            "tick",
        },
        field="aggregate resume identity",
    )
    try:
        identity = OpenEcologyAggregateIdentityPins(**identity_payload)
        return OpenEcologyAggregateResumePins(
            identity=identity,
            aggregate_generation_index=payload["aggregate_generation_index"],
            commit_sha256=payload["commit_sha256"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            checkpoint_generation_identity_sha256=payload[
                "checkpoint_generation_identity_sha256"
            ],
            evidence_manifest_sha256=payload["evidence_manifest_sha256"],
            evidence_manifest_status=payload["evidence_manifest_status"],
        )
    except (TypeError, ValueError) as error:
        raise OpenEcologyProcessWorkerError(
            "aggregate resume pin payload is invalid"
        ) from error


def _test_pins(
    *,
    task_id: str,
    source_git_sha: str,
    observed_tick: int,
    campaign_id: str,
) -> OpenEcologyAggregateResumePins:
    identity = OpenEcologyAggregateIdentityPins(
        source_git_sha=source_git_sha,
        config_contract_sha256="c" * 64,
        seed_contract_sha256="d" * 64,
        run_generation_id=(
            f"phase-d:{task_id}:"
            f"{hashlib.sha256(campaign_id.encode('utf-8')).hexdigest()[:16]}"
        ),
        island_id=task_id,
        simulation_generation_index=0,
        tick=observed_tick - 1,
    )
    digest = hashlib.sha256(f"{task_id}:{observed_tick}".encode()).hexdigest()
    return OpenEcologyAggregateResumePins(
        identity=identity,
        aggregate_generation_index=max(0, observed_tick // _FRONTIER_INTERVAL - 1),
        commit_sha256=digest,
        checkpoint_sha256="1" * 64,
        checkpoint_generation_identity_sha256="2" * 64,
        evidence_manifest_sha256="3" * 64,
        evidence_manifest_status="open",
    )


def _worker_send(
    connection: Connection,
    payload: Mapping[str, object],
    *,
    max_message_bytes: int,
) -> None:
    connection.send_bytes(_encode_message(payload, max_message_bytes=max_message_bytes))


def _worker_receive(
    connection: Connection,
    *,
    max_message_bytes: int,
) -> Mapping[str, object]:
    return _decode_message(
        connection.recv_bytes(max_message_bytes),
        max_message_bytes=max_message_bytes,
    )


def _encode_message(
    payload: Mapping[str, object],
    *,
    max_message_bytes: int,
) -> bytes:
    if not isinstance(payload, Mapping):
        raise OpenEcologyProcessWorkerError("worker message payload must be an object")
    cloned = _json_clone_mapping(payload)
    envelope = {
        "schema_version": OPEN_ECOLOGY_PROCESS_WORKER_PROTOCOL_SCHEMA_VERSION,
        "payload": cloned,
        "payload_sha256": _stable_digest(cloned),
    }
    try:
        payload_bytes = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise OpenEcologyProcessWorkerError(
            "worker message is not canonical JSON"
        ) from error
    if len(payload_bytes) > max_message_bytes:
        raise OpenEcologyProcessWorkerError(
            "worker message exceeds bounded IPC contract"
        )
    return payload_bytes


def _decode_message(
    payload_bytes: bytes,
    *,
    max_message_bytes: int,
) -> Mapping[str, object]:
    if not payload_bytes or len(payload_bytes) > max_message_bytes:
        raise OpenEcologyProcessWorkerError("worker IPC payload is empty or oversized")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OpenEcologyProcessWorkerError(
                    "worker IPC payload contains duplicate keys"
                )
            result[key] = value
        return result

    try:
        envelope = json.loads(
            payload_bytes,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: _raise_nonfinite(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenEcologyProcessWorkerError(
            "worker IPC payload is not strict JSON"
        ) from error
    envelope = _mapping(envelope, field="worker IPC envelope")
    _exact_keys(
        envelope,
        {"schema_version", "payload", "payload_sha256"},
        field="worker IPC envelope",
    )
    if (
        envelope["schema_version"]
        != OPEN_ECOLOGY_PROCESS_WORKER_PROTOCOL_SCHEMA_VERSION
    ):
        raise OpenEcologyProcessWorkerError("worker IPC schema version drifted")
    payload = _mapping(envelope["payload"], field="worker IPC payload")
    if _stable_digest(payload) != _sha256(
        envelope["payload_sha256"],
        field="worker IPC payload_sha256",
    ):
        raise OpenEcologyProcessWorkerError("worker IPC payload digest mismatch")
    return _json_clone_mapping(payload)


def _raise_nonfinite(value: str) -> NoReturn:
    raise OpenEcologyProcessWorkerError(
        f"worker IPC payload contains non-finite value {value}"
    )


def _stable_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise OpenEcologyProcessWorkerError(
            "process worker value is not canonical JSON"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _json_clone_mapping(value: Mapping[str, object]) -> dict[str, object]:
    try:
        cloned = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise OpenEcologyProcessWorkerError(
            "process worker mapping is not canonical JSON"
        ) from error
    if not isinstance(cloned, dict):
        raise OpenEcologyProcessWorkerError("process worker mapping clone changed type")
    return cloned


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OpenEcologyProcessWorkerError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise OpenEcologyProcessWorkerError(f"{field} must be an array")
    return value


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    return tuple(
        _string(item, field=f"{field}[]") for item in _sequence(value, field=field)
    )


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise OpenEcologyProcessWorkerError(f"{field} must be a non-empty string")
    return value


def _identifier(value: object, *, field: str) -> str:
    parsed = _string(value, field=field)
    if _IDENTIFIER_RE.fullmatch(parsed) is None:
        raise OpenEcologyProcessWorkerError(f"{field} is not a safe identifier")
    return parsed


def _sha256(value: object, *, field: str) -> str:
    parsed = _string(value, field=field)
    if _SHA256_RE.fullmatch(parsed) is None:
        raise OpenEcologyProcessWorkerError(f"{field} must be lowercase SHA256 hex")
    return parsed


def _git_sha(value: object, *, field: str) -> str:
    parsed = _string(value, field=field)
    if _GIT_SHA_RE.fullmatch(parsed) is None:
        raise OpenEcologyProcessWorkerError(f"{field} must be a full lowercase Git SHA")
    return parsed


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise OpenEcologyProcessWorkerError(f"{field} must be boolean")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologyProcessWorkerError(f"{field} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    parsed = _nonnegative_int(value, field=field)
    if parsed <= 0:
        raise OpenEcologyProcessWorkerError(f"{field} must be positive")
    return parsed


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenEcologyProcessWorkerError(f"{field} must be numeric")
    parsed = float(value)
    if not parsed > 0 or not parsed < float("inf"):
        raise OpenEcologyProcessWorkerError(f"{field} must be finite and positive")
    return parsed


def _frontier_tick(value: object, *, field: str) -> int:
    parsed = _positive_int(value, field=field)
    if parsed % _FRONTIER_INTERVAL != 0:
        raise OpenEcologyProcessWorkerError(
            f"{field} must be a {_FRONTIER_INTERVAL}-tick frontier"
        )
    return parsed


def _bounded_message_bytes(value: object) -> int:
    parsed = _positive_int(value, field="max_message_bytes")
    if not (
        OPEN_ECOLOGY_PROCESS_WORKER_MIN_MESSAGE_BYTES
        <= parsed
        <= OPEN_ECOLOGY_PROCESS_WORKER_MAX_MESSAGE_BYTES
    ):
        raise OpenEcologyProcessWorkerError(
            "max_message_bytes is outside the non-weakenable IPC bounds"
        )
    return parsed


def _absolute_directory(value: object, *, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise OpenEcologyProcessWorkerError(f"{field} must be a path")
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise OpenEcologyProcessWorkerError(
            f"{field} must be one existing absolute non-symlink directory"
        )
    return path.resolve()


def _exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    if set(payload) != expected:
        raise OpenEcologyProcessWorkerError(f"{field} keys are not exact")


__all__ = [
    "OPEN_ECOLOGY_PROCESS_WORKER_ASSIGNMENT_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PROCESS_WORKER_BATCH_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PROCESS_WORKER_DEFAULT_MAX_MESSAGE_BYTES",
    "OPEN_ECOLOGY_PROCESS_WORKER_LAUNCHER_CONTRACT",
    "OPEN_ECOLOGY_PROCESS_WORKER_PROTOCOL_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PROCESS_WORKER_RESULT_SCHEMA_VERSION",
    "OpenEcologyProcessWorkerError",
    "PersistentProcessWorkerLauncher",
    "ProcessWorkerExpectedState",
    "ProcessWorkerFrontierBatch",
    "ProcessWorkerRunnerView",
    "ProcessWorkerSlot",
    "ProcessWorkerStatus",
    "ProcessWorkerTaskResult",
    "build_process_worker_slots",
    "local_process_worker_host_identity",
    "process_worker_assignment_sha256",
]
