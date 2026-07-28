"""Run or inspect the storage-gated persistent open-ecology campaign."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import selectors
import stat
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence
import weakref

from evolution_sim.io.open_ecology_bounded_subprocess import (
    OpenEcologyProcessGroupError,
    terminate_process_group_before_reap,
    wait_for_leader_exit_without_reaping,
)
from evolution_sim.io.open_ecology_campaign_storage import (
    MARKER_NAME,
    CampaignStorageLimits,
    canonical_json_bytes,
    seal_closed_bundle,
)
from evolution_sim.mind.open_ecology_campaign_contract import (
    OpenEcologyCampaignCoordinatorError,
)


OPEN_ECOLOGY_CAMPAIGN_LAUNCH_SPEC_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_launch_spec_v1"
)
_MAX_LAUNCH_SPEC_BYTES = 1024 * 1024
_MAX_HEALTH_PROBE_BYTES = 1024 * 1024
_MAX_HEALTH_EXECUTABLE_BYTES = 16 * 1024 * 1024
_HEALTH_PROBE_TIMEOUT_SECONDS = 30.0
_HEALTH_PROBE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class _HealthExecutableSnapshot:
    """Private, read-only executable copies detached from mutable authorities."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="evosim-health-executable-"
        )
        self.root = Path(self._temporary.name).resolve()
        self._sealed = False
        self.executable_path: Path | None = None

    def add(self, name: str, encoded: bytes) -> Path:
        if self._sealed:
            raise OpenEcologyCampaignCoordinatorError(
                "health executable snapshot is already sealed"
            )
        path = self.root / name
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o500,
        )
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OpenEcologyCampaignCoordinatorError(
                        "health executable snapshot write made no progress"
                    )
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(path, 0o500)
        snapshot_bytes, _ = _read_pinned_file(
            path,
            max_bytes=_MAX_HEALTH_EXECUTABLE_BYTES,
            field="health executable snapshot",
            require_single_link=True,
            require_executable=True,
        )
        if snapshot_bytes != encoded:
            raise OpenEcologyCampaignCoordinatorError(
                "health executable snapshot drifted"
            )
        return path

    def seal(self) -> None:
        os.chmod(self.root, 0o500)
        self._sealed = True

    def cleanup(self) -> None:
        try:
            os.chmod(self.root, 0o700)
        except FileNotFoundError:
            pass
        self._temporary.cleanup()


def _darwin_immutable_execution_path(
    authority: Mapping[str, object],
) -> Path | None:
    """Use a canonical system object when macOS rejects its copied code signature."""

    if sys.platform != "darwin":
        return None
    resolved_value = authority.get("resolved_path", authority.get("path"))
    resolved = Path(str(resolved_value))
    try:
        candidates = (resolved, *resolved.parents)
        effective_uid = os.geteuid()
        effective_groups = {os.getegid(), *os.getgroups()}
        for candidate in candidates:
            metadata = candidate.stat()
            mode = stat.S_IMODE(metadata.st_mode)
            if metadata.st_uid == effective_uid:
                writable = bool(mode & stat.S_IWUSR)
            elif metadata.st_gid in effective_groups:
                writable = bool(mode & stat.S_IWGRP)
            else:
                writable = bool(mode & stat.S_IWOTH)
            if writable:
                return None
    except OSError:
        return None
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Advance the exact 48-world open-ecology matrix through globally "
            "storage-gated 5,000-tick frontiers."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--launch-spec", required=True, type=Path)
    run.add_argument("--expected-launch-spec-sha256", required=True)
    run.add_argument("--through-tick", required=True, type=int)
    run.add_argument("--resume-receipt", type=Path)
    run.add_argument("--resume-receipt-sha256")

    status = subparsers.add_parser("status")
    status.add_argument("--receipt", required=True, type=Path)

    seal_bundle = subparsers.add_parser("seal-bundle")
    seal_bundle.add_argument("--launch-spec", required=True, type=Path)
    seal_bundle.add_argument("--expected-launch-spec-sha256", required=True)
    seal_bundle.add_argument("--bundle-dir", required=True, type=Path)
    seal_bundle.add_argument("--bundle-id", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "status":
        from evolution_sim.mind.open_ecology_campaign_coordinator import (
            load_campaign_frontier_receipt,
        )

        receipt = load_campaign_frontier_receipt(arguments.receipt)
        payload = {
            "campaign_id": receipt.payload.get("campaign_id"),
            "frontier_tick": receipt.frontier_tick,
            "receipt_path": str(receipt.path),
            "receipt_sha256": receipt.receipt_sha256,
            "source_git_sha": receipt.source_git_sha,
            "source_manifest_sha256": receipt.source_manifest_sha256,
            "matrix_task_count": len(receipt.payload.get("tasks", [])),
            "started_task_count": len(receipt.resume_pins_by_task()),
            "barrier_scope": receipt.payload.get("barrier_scope"),
            "qualification_frontier_tick": receipt.payload.get(
                "qualification_frontier_tick"
            ),
            "matrix_frontier_tick": receipt.payload.get("matrix_frontier_tick"),
            "storage": receipt.payload.get("storage"),
            "next_interval_authorized": receipt.payload.get("next_interval_authorized"),
            "read_only": True,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    spec = _load_launch_spec(
        arguments.launch_spec,
        expected_sha256=arguments.expected_launch_spec_sha256,
    )
    if arguments.command == "seal-bundle":
        return _seal_bundle(
            spec=spec,
            launch_spec_path=arguments.launch_spec,
            launch_spec_sha256=arguments.expected_launch_spec_sha256,
            bundle_dir=arguments.bundle_dir,
            bundle_id=arguments.bundle_id,
        )
    from evolution_sim.mind.open_ecology_campaign_coordinator import (
        OpenEcologyCampaignCoordinator,
        build_static_worker_assignments,
        restore_campaign_runners_from_receipt,
    )
    from evolution_sim.mind.open_ecology_persistent_island import (
        PersistentArtifactBinding,
        PersistentIslandRunner,
        build_persistent_island_task_matrix,
    )
    from evolution_sim.mind.open_ecology_process_workers import (
        PersistentProcessWorkerLauncher,
        build_process_worker_slots,
        process_worker_assignment_sha256,
    )

    bindings = tuple(
        PersistentArtifactBinding(**artifact) for artifact in _artifact_bindings(spec)
    )
    tasks = build_persistent_island_task_matrix(
        bindings,
        selected_density=_integer(spec, "selected_density"),
    )
    campaign_root = Path(_string(spec, "campaign_root"))
    campaign_id = _string(spec, "campaign_id")
    repository_root = Path(_string(spec, "repository_root"))
    git_executable = Path(_string(spec, "git_executable"))
    git_executable_sha256 = _sha256(
        spec.get("git_executable_sha256"),
        field="launch spec git_executable_sha256",
    )
    if (arguments.resume_receipt is None) != (arguments.resume_receipt_sha256 is None):
        raise OpenEcologyCampaignCoordinatorError(
            "resume receipt path and external SHA256 must be supplied together"
        )
    retained = (
        None
        if arguments.resume_receipt is None
        else load_campaign_frontier_receipt(
            arguments.resume_receipt,
            expected_receipt_sha256=arguments.resume_receipt_sha256,
        )
    )
    worker_count = _integer(spec, "worker_count")
    process_config = _process_worker_config(spec, worker_count=worker_count)
    assignment_records = build_static_worker_assignments(
        tasks,
        worker_count=worker_count,
    )
    assignment_sha256 = _stable_digest(
        [assignment.to_dict() for assignment in assignment_records]
    )
    launcher_context = nullcontext(None)
    if process_config is not None:
        slots = build_process_worker_slots(
            tasks,
            worker_count=worker_count,
            host_identity=_config_string(process_config, "host_identity"),
            torch_threads_per_worker=_config_integer(
                process_config,
                "torch_threads_per_worker",
            ),
        )
        process_assignment_sha256 = process_worker_assignment_sha256(tasks, slots)
        if process_assignment_sha256 != assignment_sha256:
            raise OpenEcologyCampaignCoordinatorError(
                "process worker assignment does not match coordinator authority"
            )
        launcher_context = PersistentProcessWorkerLauncher(
            tasks=tasks,
            campaign_root=campaign_root,
            campaign_id=campaign_id,
            repository_root=Path(_string(spec, "repository_root")),
            source_git_sha=_string(spec, "source_git_sha"),
            source_manifest_sha256=_string(spec, "source_manifest_sha256"),
            git_executable=git_executable,
            git_executable_sha256=git_executable_sha256,
            slots=slots,
            assignment_sha256=assignment_sha256,
            response_timeout_seconds=_config_positive_number(
                process_config,
                "response_timeout_seconds",
            ),
            startup_timeout_seconds=_config_positive_number(
                process_config,
                "startup_timeout_seconds",
            ),
            shutdown_timeout_seconds=_config_positive_number(
                process_config,
                "shutdown_timeout_seconds",
            ),
            max_message_bytes=_config_integer(
                process_config,
                "max_message_bytes",
            ),
            allow_cpu_oversubscription=_config_boolean(
                process_config,
                "allow_cpu_oversubscription",
            ),
        )
    with launcher_context as worker_launcher:
        if worker_launcher is not None:
            runners = {}
        elif retained is None:
            runners = {}
        else:
            runners = restore_campaign_runners_from_receipt(
                tasks=tasks,
                receipt=retained,
                campaign_root=campaign_root,
                campaign_id=campaign_id,
            )
        authorized_frontier_tick = 0 if retained is None else retained.frontier_tick
        coordinator = OpenEcologyCampaignCoordinator(
            tasks=tasks,
            runners=runners,
            campaign_root=campaign_root,
            receipt_directory=Path(_string(spec, "receipt_directory")),
            campaign_id=campaign_id,
            source_git_sha=_string(spec, "source_git_sha"),
            source_manifest_sha256=_string(spec, "source_manifest_sha256"),
            worker_count=worker_count,
            authorized_frontier_tick=authorized_frontier_tick,
            previous_frontier_receipt=retained,
            source_probe=lambda: _campaign_source_snapshot(
                repository_root=repository_root,
                source_git_sha=_string(spec, "source_git_sha"),
                source_manifest_sha256=_string(
                    spec,
                    "source_manifest_sha256",
                ),
                git_executable=git_executable,
                git_executable_sha256=git_executable_sha256,
            ),
            health_probe=_command_health_probe(
                _string_sequence(spec, "health_probe_command"),
                expected_baseline_sha256=_sha256(
                    spec.get("health_baseline_sha256"),
                    field="launch spec health_baseline_sha256",
                ),
                expected_probe_file_sha256=_sha256(
                    spec.get("health_probe_file_sha256"),
                    field="launch spec health_probe_file_sha256",
                ),
            ),
            worker_launcher=worker_launcher,
            pending_runner_factory=(
                None
                if worker_launcher is not None
                else lambda task: PersistentIslandRunner.open(
                    task,
                    evidence_directory=campaign_root / task.task_id,
                    campaign_root=campaign_root,
                    campaign_id=campaign_id,
                )
            ),
            launch_authority={
                "launch_spec_path": str(arguments.launch_spec),
                "launch_spec_sha256": arguments.expected_launch_spec_sha256,
            },
        )
        receipts = coordinator.advance_through(arguments.through_tick)
        final = coordinator.previous_frontier_receipt
        if final is None:
            raise OpenEcologyCampaignCoordinatorError(
                "run produced no retained frontier authority"
            )
    print(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "frontiers_advanced": [receipt.frontier_tick for receipt in receipts],
                "latest_frontier_tick": final.frontier_tick,
                "latest_receipt_path": str(final.path),
                "latest_receipt_sha256": final.receipt_sha256,
                "matrix_task_count": len(final.payload.get("tasks", [])),
                "started_task_count": len(final.resume_pins_by_task()),
                "pruning_or_deletion_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _seal_bundle(
    *,
    spec: Mapping[str, object],
    launch_spec_path: Path,
    launch_spec_sha256: str,
    bundle_dir: Path,
    bundle_id: str,
) -> int:
    campaign_root = Path(_string(spec, "campaign_root"))
    campaign_id = _string(spec, "campaign_id")
    repository_root = Path(_string(spec, "repository_root"))
    source_git_sha = _string(spec, "source_git_sha")
    source_manifest_sha256 = _sha256(
        spec.get("source_manifest_sha256"),
        field="launch spec source_manifest_sha256",
    )
    git_executable = Path(_string(spec, "git_executable"))
    git_executable_sha256 = _sha256(
        spec.get("git_executable_sha256"),
        field="launch spec git_executable_sha256",
    )
    limits = CampaignStorageLimits()
    limits.validate()
    source_arguments = {
        "repository_root": repository_root,
        "source_git_sha": source_git_sha,
        "source_manifest_sha256": source_manifest_sha256,
        "git_executable": git_executable,
        "git_executable_sha256": git_executable_sha256,
    }
    source_before = _campaign_source_snapshot(**source_arguments)
    snapshot = seal_closed_bundle(
        campaign_root,
        bundle_dir,
        campaign_id=campaign_id,
        bundle_id=bundle_id,
        source_git_sha=source_git_sha,
        source_manifest_sha256=source_manifest_sha256,
        limits=limits,
    )
    source_after = _campaign_source_snapshot(**source_arguments)
    if source_after != source_before:
        raise OpenEcologyCampaignCoordinatorError(
            "live source authority changed across closed-bundle sealing"
        )
    payload = {
        "bundle_dir": str(snapshot.root),
        "bundle_id": snapshot.bundle_id,
        "campaign_id": snapshot.campaign_id,
        "content_directory_count": snapshot.marker["directory_count"],
        "content_file_count": snapshot.marker["file_count"],
        "content_total_file_bytes": snapshot.marker["total_file_bytes"],
        "launch_spec_path": str(launch_spec_path),
        "launch_spec_sha256": launch_spec_sha256,
        "marker_path": str(snapshot.root / MARKER_NAME),
        "marker_sha256": snapshot.marker_sha256,
        "pruning_or_deletion_performed": False,
        "source_git_sha": snapshot.source_git_sha,
        "source_manifest_sha256": snapshot.source_manifest_sha256,
        "source_verified_before_and_after": True,
        "storage_limits": {
            "max_campaign_bytes": limits.max_campaign_bytes,
            "max_entries": limits.max_entries,
            "min_campaign_free_bytes": limits.min_campaign_free_bytes,
            "min_remote_free_bytes": limits.min_remote_free_bytes,
        },
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0


def _load_launch_spec(
    path: Path,
    *,
    expected_sha256: str,
) -> Mapping[str, object]:
    expected_digest = _sha256(
        expected_sha256,
        field="expected launch spec SHA256",
    )
    encoded, _ = _read_pinned_file(
        path,
        max_bytes=_MAX_LAUNCH_SPEC_BYTES,
        field="launch spec",
        require_single_link=True,
        require_executable=False,
    )
    if hashlib.sha256(encoded).hexdigest() != expected_digest:
        raise OpenEcologyCampaignCoordinatorError(
            "launch spec does not match its externally supplied SHA256"
        )

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OpenEcologyCampaignCoordinatorError(
                    "launch spec contains duplicate keys"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: _raise_nonfinite(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenEcologyCampaignCoordinatorError(
            "launch spec is not strict JSON"
        ) from error
    if not isinstance(payload, Mapping):
        raise OpenEcologyCampaignCoordinatorError("launch spec root must be an object")
    required = {
        "schema_version",
        "campaign_id",
        "campaign_root",
        "receipt_directory",
        "repository_root",
        "git_executable",
        "git_executable_sha256",
        "source_git_sha",
        "source_manifest_sha256",
        "selected_density",
        "worker_count",
        "process_workers",
        "health_probe_command",
        "health_baseline_sha256",
        "health_probe_file_sha256",
        "artifacts",
    }
    if set(payload) != required:
        raise OpenEcologyCampaignCoordinatorError("launch spec keys are not exact")
    if (
        payload.get("schema_version")
        != OPEN_ECOLOGY_CAMPAIGN_LAUNCH_SPEC_SCHEMA_VERSION
    ):
        raise OpenEcologyCampaignCoordinatorError("launch spec schema version drifted")
    return payload


def _artifact_bindings(
    spec: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    raw_artifacts = spec.get("artifacts")
    if (
        not isinstance(raw_artifacts, Sequence)
        or isinstance(raw_artifacts, (str, bytes, bytearray))
        or len(raw_artifacts) != 4
    ):
        raise OpenEcologyCampaignCoordinatorError(
            "launch spec requires exactly four artifact bindings"
        )
    required = {
        "learner_index",
        "learner_seed",
        "artifact_path",
        "artifact_sha256",
        "artifact_file_sha256",
        "source_commit",
        "terminal_authority_sha256",
    }
    artifacts: list[dict[str, object]] = []
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, Mapping) or set(raw_artifact) != required:
            raise OpenEcologyCampaignCoordinatorError(
                f"artifact binding {index} keys are not exact"
            )
        artifacts.append(dict(raw_artifact))
    return tuple(artifacts)


def _process_worker_config(
    spec: Mapping[str, object],
    *,
    worker_count: int,
) -> Mapping[str, object] | None:
    if worker_count <= 0:
        raise OpenEcologyCampaignCoordinatorError(
            "launch spec worker_count must be positive"
        )
    value = spec.get("process_workers")
    if worker_count == 1:
        if value is not None:
            raise OpenEcologyCampaignCoordinatorError(
                "single-worker launch must set process_workers to null"
            )
        return None
    if not isinstance(value, Mapping):
        raise OpenEcologyCampaignCoordinatorError(
            "multi-worker launch requires an explicit process_workers object"
        )
    required = {
        "host_identity",
        "device_kind",
        "device_index",
        "torch_threads_per_worker",
        "allow_cpu_oversubscription",
        "response_timeout_seconds",
        "startup_timeout_seconds",
        "shutdown_timeout_seconds",
        "max_message_bytes",
    }
    if set(value) != required:
        raise OpenEcologyCampaignCoordinatorError("process_workers keys are not exact")
    if value.get("device_kind") != "cpu" or value.get("device_index") is not None:
        raise OpenEcologyCampaignCoordinatorError(
            "persistent process workers require explicit CPU/null device authority"
        )
    _config_string(value, "host_identity")
    _config_integer(value, "torch_threads_per_worker")
    _config_boolean(value, "allow_cpu_oversubscription")
    _config_positive_number(value, "response_timeout_seconds")
    _config_positive_number(value, "startup_timeout_seconds")
    _config_positive_number(value, "shutdown_timeout_seconds")
    _config_integer(value, "max_message_bytes")
    return value


def _config_string(config: Mapping[str, object], field: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value:
        raise OpenEcologyCampaignCoordinatorError(
            f"process_workers {field} must be a non-empty string"
        )
    return value


def _config_integer(config: Mapping[str, object], field: str) -> int:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenEcologyCampaignCoordinatorError(
            f"process_workers {field} must be a positive integer"
        )
    return value


def _config_positive_number(
    config: Mapping[str, object],
    field: str,
) -> float:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenEcologyCampaignCoordinatorError(
            f"process_workers {field} must be numeric"
        )
    parsed = float(value)
    if not 0 < parsed < float("inf"):
        raise OpenEcologyCampaignCoordinatorError(
            f"process_workers {field} must be finite and positive"
        )
    return parsed


def _config_boolean(config: Mapping[str, object], field: str) -> bool:
    value = config.get(field)
    if not isinstance(value, bool):
        raise OpenEcologyCampaignCoordinatorError(
            f"process_workers {field} must be boolean"
        )
    return value


def _string(spec: Mapping[str, object], field: str) -> str:
    value = spec.get(field)
    if not isinstance(value, str) or not value:
        raise OpenEcologyCampaignCoordinatorError(
            f"launch spec {field} must be a non-empty string"
        )
    return value


def _integer(spec: Mapping[str, object], field: str) -> int:
    value = spec.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenEcologyCampaignCoordinatorError(
            f"launch spec {field} must be an integer"
        )
    return value


def _string_sequence(
    spec: Mapping[str, object],
    field: str,
) -> tuple[str, ...]:
    value = spec.get(field)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise OpenEcologyCampaignCoordinatorError(
            f"launch spec {field} must be a non-empty string array"
        )
    arguments = tuple(value)
    if len(arguments) != 1:
        raise OpenEcologyCampaignCoordinatorError(
            f"launch spec {field} must contain one standalone executable only"
        )
    executable = Path(arguments[0])
    _prepare_health_executable(executable, create_snapshot=False)
    return arguments


def _command_health_probe(
    command: Sequence[str],
    *,
    expected_baseline_sha256: str,
    expected_probe_file_sha256: str,
):
    arguments = tuple(command)
    if not arguments:
        raise OpenEcologyCampaignCoordinatorError(
            "health probe command must not be empty"
        )
    if len(arguments) != 1:
        raise OpenEcologyCampaignCoordinatorError(
            "health probe command must contain one standalone executable only"
        )
    expected_baseline_digest = _sha256(
        expected_baseline_sha256,
        field="expected health baseline SHA256",
    )
    expected_probe_digest = _sha256(
        expected_probe_file_sha256,
        field="expected health probe file SHA256",
    )
    executable_path = Path(arguments[0])
    execution_arguments, executable_authority, snapshot = _prepare_health_executable(
        executable_path,
        create_snapshot=True,
    )
    if executable_authority.get("sha256") != expected_probe_digest:
        if snapshot is not None:
            snapshot.cleanup()
        raise OpenEcologyCampaignCoordinatorError(
            "health probe executable does not match launch-spec SHA256"
        )
    command_authority = {
        "executable": executable_authority,
        "arguments_sha256": _stable_digest(list(arguments)),
        "health_baseline_sha256": expected_baseline_digest,
        "health_probe_file_sha256": expected_probe_digest,
    }
    command_authority["authority_sha256"] = _stable_digest(command_authority)

    def probe(phase: str, frontier_tick: int) -> Mapping[str, object]:
        _, current_executable_authority, _ = _prepare_health_executable(
            executable_path,
            create_snapshot=False,
        )
        if current_executable_authority != executable_authority:
            raise OpenEcologyCampaignCoordinatorError(
                "health probe executable identity or SHA256 changed"
            )
        environment = {
            "EVOSIM_OPEN_ECOLOGY_HEALTH_PHASE": phase,
            "EVOSIM_OPEN_ECOLOGY_FRONTIER_TICK": str(frontier_tick),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": _HEALTH_PROBE_PATH,
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
        stdout, stderr, returncode = _run_bounded_health_probe(
            execution_arguments,
            environment=environment,
            executable=None if snapshot is None else snapshot.executable_path,
        )
        _, post_execution_authority, _ = _prepare_health_executable(
            executable_path,
            create_snapshot=False,
        )
        if post_execution_authority != executable_authority:
            raise OpenEcologyCampaignCoordinatorError(
                "health probe executable identity or SHA256 changed during execution"
            )
        if returncode != 0:
            raise OpenEcologyCampaignCoordinatorError(
                "external resource/health probe failed"
            )
        if stderr:
            raise OpenEcologyCampaignCoordinatorError(
                "external resource/health probe wrote to stderr"
            )

        def reject_duplicates(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise OpenEcologyCampaignCoordinatorError(
                        "external resource/health probe returned duplicate keys"
                    )
                result[key] = value
            return result

        try:
            payload = json.loads(
                stdout,
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda value: _raise_nonfinite(value),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OpenEcologyCampaignCoordinatorError(
                "external resource/health probe did not return strict JSON"
            ) from error
        if not isinstance(payload, Mapping):
            raise OpenEcologyCampaignCoordinatorError(
                "external resource/health probe must return a JSON object"
            )
        if payload.get("baseline_file_sha256") != expected_baseline_digest:
            raise OpenEcologyCampaignCoordinatorError(
                "external resource/health probe baseline SHA256 drifted"
            )
        return {
            **payload,
            "health_probe_authority": command_authority,
        }

    if snapshot is not None:
        weakref.finalize(probe, snapshot.cleanup)
    return probe


def _run_bounded_health_probe(
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str],
    executable: Path | None = None,
) -> tuple[bytes, bytes, int]:
    selector: selectors.BaseSelector | None = None
    stdout = None
    stderr = None
    buffers: dict[str, bytearray] = {
        "stdout": bytearray(),
        "stderr": bytearray(),
    }
    group_cleanup_attempted = False
    try:
        process = subprocess.Popen(
            list(arguments),
            executable=None if executable is None else str(executable),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=dict(environment),
            start_new_session=True,
        )
    except OSError as error:
        raise OpenEcologyCampaignCoordinatorError(
            "external resource/health probe failed to start"
        ) from error
    try:
        stdout = process.stdout
        stderr = process.stderr
        if stdout is None or stderr is None:
            raise OpenEcologyCampaignCoordinatorError(
                "external resource/health probe pipes were not created"
            )
        selector = selectors.DefaultSelector()
        streams = {
            stdout.fileno(): ("stdout", stdout),
            stderr.fileno(): ("stderr", stderr),
        }
        for descriptor, (name, _) in streams.items():
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ, data=name)
        deadline = time.monotonic() + _HEALTH_PROBE_TIMEOUT_SECONDS
        total_bytes = 0
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OpenEcologyCampaignCoordinatorError(
                    "external resource/health probe timed out"
                )
            events = selector.select(timeout=remaining)
            if not events:
                raise OpenEcologyCampaignCoordinatorError(
                    "external resource/health probe timed out"
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
                if total_bytes > _MAX_HEALTH_PROBE_BYTES:
                    raise OpenEcologyCampaignCoordinatorError(
                        "external resource/health probe output exceeds 1 MiB"
                    )
                buffers[str(key.data)].extend(chunk)
        try:
            wait_for_leader_exit_without_reaping(process, deadline=deadline)
        except TimeoutError as error:
            raise OpenEcologyCampaignCoordinatorError(
                "external resource/health probe timed out"
            ) from error
        group_cleanup_attempted = True
        returncode = _terminate_health_probe(
            process,
            leader_exit_observed=True,
        )
    except BaseException as primary_error:
        if not group_cleanup_attempted:
            try:
                _terminate_health_probe(process)
            except OpenEcologyCampaignCoordinatorError as cleanup_error:
                primary_error.add_note(f"additional cleanup failure: {cleanup_error}")
                raise primary_error from cleanup_error
        raise
    finally:
        if selector is not None:
            selector.close()
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()
    return bytes(buffers["stdout"]), bytes(buffers["stderr"]), returncode


def _terminate_health_probe(
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
        raise OpenEcologyCampaignCoordinatorError(
            "health probe process group survived termination"
        ) from error


def _prepare_health_executable(
    path: Path,
    *,
    create_snapshot: bool,
) -> tuple[tuple[str, ...], dict[str, object], _HealthExecutableSnapshot | None]:
    encoded, executable_authority = _read_pinned_executable(path)
    if not encoded.startswith(b"#!"):
        authority = {
            **executable_authority,
            "execution_mode": "snapshotted_native_standalone_v2",
            "interpreter": None,
        }
        unsigned = dict(authority)
        unsigned.pop("identity_sha256")
        authority["identity_sha256"] = _stable_digest(unsigned)
        if not create_snapshot:
            return (str(path),), authority, None
        immutable_path = _darwin_immutable_execution_path(executable_authority)
        if immutable_path is not None:
            return (str(immutable_path),), authority, None
        snapshot = _HealthExecutableSnapshot()
        try:
            snapshot_path = snapshot.add("executable", encoded)
            snapshot.executable_path = snapshot_path
            snapshot.seal()
        except BaseException:
            snapshot.cleanup()
            raise
        return (str(executable_authority["resolved_path"]),), authority, snapshot
    first_line = encoded.splitlines()[0]
    try:
        shebang = first_line[2:].decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise OpenEcologyCampaignCoordinatorError(
            "health probe shebang must be strict UTF-8"
        ) from error
    tokens = shebang.split()
    if len(tokens) != 1 or not Path(tokens[0]).is_absolute():
        raise OpenEcologyCampaignCoordinatorError(
            "health probe script requires one absolute pinned shebang interpreter"
        )
    interpreter_path = Path(tokens[0])
    interpreter_bytes, interpreter_authority = _read_pinned_executable(interpreter_path)
    if interpreter_bytes.startswith(b"#!"):
        raise OpenEcologyCampaignCoordinatorError(
            "health probe shebang interpreter must be a native executable"
        )
    lines = encoded.splitlines()
    marker = b"# evosim_shebang_sha256="
    if len(lines) <= 1 or not lines[1].startswith(marker):
        raise OpenEcologyCampaignCoordinatorError(
            "health probe script requires an embedded shebang interpreter SHA256"
        )
    try:
        declared_interpreter_sha256 = lines[1][len(marker) :].decode("ascii")
    except UnicodeDecodeError as error:
        raise OpenEcologyCampaignCoordinatorError(
            "health probe shebang SHA256 marker is malformed"
        ) from error
    _sha256(
        declared_interpreter_sha256,
        field="health probe shebang SHA256 marker",
    )
    if interpreter_authority.get("sha256") != declared_interpreter_sha256:
        raise OpenEcologyCampaignCoordinatorError(
            "health probe shebang interpreter does not match embedded SHA256"
        )
    authority = {
        **executable_authority,
        "execution_mode": "snapshotted_script_with_pinned_interpreter_v2",
        "interpreter": interpreter_authority,
        "declared_interpreter_sha256": declared_interpreter_sha256,
    }
    unsigned = dict(authority)
    unsigned.pop("identity_sha256")
    authority["identity_sha256"] = _stable_digest(unsigned)
    if not create_snapshot:
        return (str(interpreter_path), str(path)), authority, None
    snapshot = _HealthExecutableSnapshot()
    try:
        interpreter_snapshot_path = _darwin_immutable_execution_path(
            interpreter_authority
        )
        if interpreter_snapshot_path is None:
            interpreter_snapshot_path = snapshot.add(
                "interpreter",
                interpreter_bytes,
            )
            snapshot.executable_path = interpreter_snapshot_path
        script_snapshot_path = snapshot.add("probe", encoded)
        snapshot.seal()
    except BaseException:
        snapshot.cleanup()
        raise
    return (
        (
            str(interpreter_authority["resolved_path"]),
            str(script_snapshot_path),
        ),
        authority,
        snapshot,
    )


def _read_pinned_executable(
    path: Path,
) -> tuple[bytes, dict[str, object]]:
    if not path.is_absolute():
        raise OpenEcologyCampaignCoordinatorError(
            "health probe executable must be one absolute regular file"
        )
    try:
        resolved_before = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise OpenEcologyCampaignCoordinatorError(
            "health probe executable symlink chain cannot be resolved"
        ) from error
    encoded, final_authority = _read_pinned_file(
        resolved_before,
        max_bytes=_MAX_HEALTH_EXECUTABLE_BYTES,
        field="health probe executable",
        require_single_link=False,
        require_executable=True,
    )
    try:
        resolved_after = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise OpenEcologyCampaignCoordinatorError(
            "health probe executable symlink chain changed while reading"
        ) from error
    if resolved_after != resolved_before:
        raise OpenEcologyCampaignCoordinatorError(
            "health probe executable symlink chain changed while reading"
        )
    authority = {
        **final_authority,
        "path": str(path),
        "resolved_path": str(resolved_before),
    }
    unsigned_authority = dict(authority)
    unsigned_authority.pop("identity_sha256")
    authority["identity_sha256"] = _stable_digest(unsigned_authority)
    return encoded, authority


def _raise_nonfinite(value: str) -> object:
    raise OpenEcologyCampaignCoordinatorError(
        f"strict JSON contains non-finite value {value}"
    )


def _campaign_source_snapshot(
    *,
    repository_root: Path,
    source_git_sha: str,
    source_manifest_sha256: str,
    git_executable: Path,
    git_executable_sha256: str,
) -> Mapping[str, object]:
    from evolution_sim.mind.open_ecology_process_workers import (
        OpenEcologyProcessWorkerError,
        strict_open_ecology_source_snapshot,
    )

    try:
        return strict_open_ecology_source_snapshot(
            repository_root=repository_root,
            source_git_sha=source_git_sha,
            source_manifest_sha256=source_manifest_sha256,
            git_executable=git_executable,
            git_executable_sha256=git_executable_sha256,
        )
    except OpenEcologyProcessWorkerError as error:
        raise OpenEcologyCampaignCoordinatorError(
            "cannot verify pinned live source authority"
        ) from error


def _stable_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_pinned_file(
    path: Path,
    *,
    max_bytes: int,
    field: str,
    require_single_link: bool,
    require_executable: bool,
) -> tuple[bytes, dict[str, object]]:
    if not path.is_absolute():
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} must be one absolute regular file"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} cannot be opened without following links"
        ) from error
    try:
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > max_bytes
            or (require_single_link and before.st_nlink != 1)
            or mode & 0o022
            or (require_executable and mode & 0o111 == 0)
        ):
            raise OpenEcologyCampaignCoordinatorError(
                f"{field} violates its regular-file identity contract"
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
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        before_identity != after_identity
        or len(encoded) != before.st_size
        or len(encoded) > max_bytes
    ):
        raise OpenEcologyCampaignCoordinatorError(f"{field} changed while reading")
    authority: dict[str, object] = {
        "path": str(path),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": mode,
        "link_count": before.st_nlink,
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    authority["identity_sha256"] = _stable_digest(authority)
    return encoded, authority


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OpenEcologyCampaignCoordinatorError(
            f"{field} must be 64 lowercase hexadecimal characters"
        )
    return value


if __name__ == "__main__":
    raise SystemExit(main())
