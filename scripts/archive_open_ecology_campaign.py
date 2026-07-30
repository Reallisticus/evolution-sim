#!/usr/bin/env python3
"""Archive one sealed open-ecology bundle and verify it on Google Drive.

The active campaign root is never an archive input.  A remote helper running
from the exact clean source checkout validates a separately sealed bundle,
uses ``archive_evolution_outputs.py`` to create its deterministic ``tar.zst``
triple, and leaves those source objects on the compute host.  The coordinator
streams only those three objects to the immutable campaign-scoped Drive path,
reads every byte back, runs an independent ``rclone check`` from a tiny local
SHA256 sum file, and writes one small receipt only after every gate passes.

There is deliberately no deletion or pruning path in this tool.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ROOT = _REPOSITORY_ROOT / "python"
sys.path = [
    entry for entry in sys.path if Path(entry or os.curdir).resolve() != _PYTHON_ROOT
]
sys.path.insert(0, str(_PYTHON_ROOT))

from evolution_sim.io.open_ecology_campaign_storage import (  # noqa: E402
    CampaignStorageError,
    CampaignStorageLimits,
    CampaignStorageLock,
    ClosedBundleSnapshot,
    DEFAULT_REMOTE_FREE_BYTES,
    canonical_json_bytes,
    check_campaign_storage,
    default_storage_lock_path,
    validate_closed_bundle,
    write_verified_receipt,
)
from evolution_sim.io.open_ecology_archive_authority import (  # noqa: E402
    ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION,
    REMOTE_HELPER_PATHS,
    REMOTE_TOOL_NAMES,
    SEALED_SSH_OPTIONS,
    ArchiveAuthorityError,
    ArchiveToolAuthority,
    FilePin,
    load_archive_tool_authority,
    minimal_subprocess_env,
    parse_ssh_connection_identity,
    verify_effective_ssh_config,
    verify_local_authority_files,
    verify_pinned_file,
)
from evolution_sim.io.open_ecology_bounded_subprocess import (  # noqa: E402
    OpenEcologyProcessGroupError,
    leader_exit_observed_without_reaping,
    terminate_process_group_before_reap,
)
from evolution_sim.io.source_manifest import (  # noqa: E402
    source_file_hash_manifest,
)


DEFAULT_RCLONE_BASE = "gdrive:evolution-sim-backups/archives/open-ecology"
REMOTE_BUILD_SCHEMA_VERSION = "open_ecology_closed_bundle_archive_build_v3"
REMOTE_RECEIPT_SCHEMA_VERSION = "open_ecology_closed_bundle_drive_receipt_v3"
_SSH_TARGET_PATTERN = re.compile(r"[A-Za-z0-9_.@-]+")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_DRIVE_ID_PATTERN = re.compile(r"[^\x00-\x20/\\]+")
_READ_CHUNK_SIZE = 8 * 1024 * 1024
_PIPE_READ_CHUNK_SIZE = 64 * 1024
_GENERIC_PROCESS_TIMEOUT_SECONDS = 15 * 60.0
_TRANSFER_PROCESS_TIMEOUT_SECONDS = 24 * 60 * 60.0
_MAX_CAPTURE_STDOUT_BYTES = 16 * 1024 * 1024
_MAX_CAPTURE_STDERR_BYTES = 4 * 1024 * 1024
_PROCESS_POLL_SECONDS = 0.05
_PROCESS_TERM_GRACE_SECONDS = 0.25


class OpenEcologyArchiveError(CampaignStorageError):
    """The closed-bundle archive or Drive verification failed closed."""


@dataclass(slots=True)
class _PipeDrain:
    stream_name: str
    pipe: object
    byte_limit: int
    consumer: Callable[[bytes], None] | None = None
    payload: bytearray | None = None
    byte_count: int = 0
    error: BaseException | None = None
    done: threading.Event = field(init=False)

    def __post_init__(self) -> None:
        self.done = threading.Event()
        if self.consumer is None:
            self.payload = bytearray()

    def run(self, changed: threading.Event) -> None:
        try:
            while True:
                chunk = self.pipe.read(_PIPE_READ_CHUNK_SIZE)  # type: ignore[attr-defined]
                if not chunk:
                    break
                next_count = self.byte_count + len(chunk)
                if next_count > self.byte_limit:
                    raise OpenEcologyArchiveError(
                        f"subprocess {self.stream_name} exceeded "
                        f"{self.byte_limit} bytes"
                    )
                self.byte_count = next_count
                if self.consumer is None:
                    assert self.payload is not None
                    self.payload.extend(chunk)
                else:
                    self.consumer(chunk)
        except BaseException as exc:  # propagate reader failures through the guard
            self.error = exc
        finally:
            try:
                self.pipe.close()  # type: ignore[attr-defined]
            finally:
                self.done.set()
                changed.set()

    def bytes(self) -> bytes:
        return bytes(self.payload or b"")


class _BoundedProcess:
    """Drain child pipes concurrently and kill the whole session on failure."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        command_name: str,
        timeout_seconds: float,
        stdout_limit: int = _MAX_CAPTURE_STDOUT_BYTES,
        stderr_limit: int = _MAX_CAPTURE_STDERR_BYTES,
        stdout_consumer: Callable[[bytes], None] | None = None,
        drain_stdout: bool = True,
    ) -> None:
        if timeout_seconds <= 0 or stdout_limit < 0 or stderr_limit < 0:
            _terminate_process_group(process)
            raise ValueError("process bounds must be non-negative and finite")
        self.process = process
        self.command_name = command_name
        self.deadline = time.monotonic() + timeout_seconds
        self.changed = threading.Event()
        self.finished = threading.Event()
        self.failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self.drains: list[_PipeDrain] = []
        if drain_stdout and process.stdout is not None:
            self.drains.append(
                _PipeDrain(
                    "stdout",
                    process.stdout,
                    stdout_limit,
                    consumer=stdout_consumer,
                )
            )
        if process.stderr is not None:
            self.drains.append(_PipeDrain("stderr", process.stderr, stderr_limit))
        self._threads = [
            threading.Thread(
                target=drain.run,
                args=(self.changed,),
                daemon=True,
                name=f"{command_name}-{drain.stream_name}-drain",
            )
            for drain in self.drains
        ]
        for thread in self._threads:
            thread.start()
        self._watchdog = threading.Thread(
            target=self._watch,
            daemon=True,
            name=f"{command_name}-deadline",
        )
        self._watchdog.start()

    def _set_failure(self, failure: BaseException) -> None:
        with self._failure_lock:
            if self.failure is None:
                self.failure = failure
        self.changed.set()

    def _watch(self) -> None:
        while not self.finished.is_set():
            for drain in self.drains:
                if drain.error is not None:
                    self._set_failure(drain.error)
                    self._cleanup_process_group()
                    return
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                self._set_failure(
                    OpenEcologyArchiveError(
                        f"{self.command_name} exceeded its total deadline"
                    )
                )
                self._cleanup_process_group()
                return
            self.changed.wait(min(_PROCESS_POLL_SECONDS, remaining))
            self.changed.clear()

    def complete(self) -> tuple[bytes, bytes]:
        leader_exit_observed = False
        while self.process.returncode is None and self.failure is None:
            try:
                leader_exit_observed = leader_exit_observed_without_reaping(
                    self.process
                )
            except OpenEcologyProcessGroupError:
                with self._cleanup_lock:
                    if self.process.returncode is None:
                        self._set_failure(
                            OpenEcologyArchiveError(
                                f"{self.command_name} leader status was lost"
                            )
                        )
                break
            if leader_exit_observed:
                break
            self.changed.wait(_PROCESS_POLL_SECONDS)
            self.changed.clear()
        if self.process.returncode is None:
            if self.failure is None and not leader_exit_observed:
                self._set_failure(
                    OpenEcologyArchiveError(
                        f"{self.command_name} exceeded its total deadline"
                    )
                )
            self._cleanup_process_group(
                leader_exit_observed=leader_exit_observed,
            )
        for thread in self._threads:
            remaining = max(0.0, self.deadline - time.monotonic())
            thread.join(remaining)
            if thread.is_alive():
                self._set_failure(
                    OpenEcologyArchiveError(
                        f"{self.command_name} pipe drain exceeded its total deadline"
                    )
                )
                self._cleanup_process_group()
        # A short-lived child can exit before the watchdog samples a reader
        # failure.  The joined drain thread is authoritative, so propagate its
        # final error before declaring the process complete.
        for drain in self.drains:
            if drain.error is not None:
                self._set_failure(drain.error)
        self.finished.set()
        self.changed.set()
        self._watchdog.join(_PROCESS_TERM_GRACE_SECONDS)
        if self.failure is not None:
            if isinstance(self.failure, OpenEcologyArchiveError):
                raise self.failure
            raise OpenEcologyArchiveError(
                f"{self.command_name} pipe drain failed: {self.failure}"
            ) from self.failure
        stdout = next(
            (drain.bytes() for drain in self.drains if drain.stream_name == "stdout"),
            b"",
        )
        stderr = next(
            (drain.bytes() for drain in self.drains if drain.stream_name == "stderr"),
            b"",
        )
        return stdout, stderr

    def abort(self) -> None:
        self._set_failure(OpenEcologyArchiveError(f"{self.command_name} aborted"))
        self._cleanup_process_group()
        self.finished.set()
        self.changed.set()
        for thread in self._threads:
            thread.join(_PROCESS_TERM_GRACE_SECONDS)
        self._watchdog.join(_PROCESS_TERM_GRACE_SECONDS)

    def _cleanup_process_group(
        self,
        *,
        leader_exit_observed: bool = False,
    ) -> None:
        with self._cleanup_lock:
            if self.process.returncode is not None:
                return
            try:
                _terminate_process_group(
                    self.process,
                    leader_exit_observed=leader_exit_observed,
                )
            except OpenEcologyArchiveError as exc:
                if self.failure is None:
                    self._set_failure(exc)


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    leader_exit_observed: bool = False,
) -> int:
    try:
        return terminate_process_group_before_reap(
            process,
            wait_timeout_seconds=_PROCESS_TERM_GRACE_SECONDS,
            leader_exit_observed=leader_exit_observed,
        )
    except OpenEcologyProcessGroupError as exc:
        raise OpenEcologyArchiveError(
            "archive subprocess group survived termination"
        ) from exc


@dataclass(frozen=True, slots=True)
class RemoteBuildAuthority:
    authority_sha256: str
    tools: tuple[tuple[str, FilePin], ...]
    helpers: tuple[tuple[str, str], ...]

    @classmethod
    def from_archive_authority(
        cls,
        authority: ArchiveToolAuthority,
    ) -> RemoteBuildAuthority:
        return cls(
            authority_sha256=authority.authority_sha256,
            tools=authority.remote_tools,
            helpers=authority.remote_helpers,
        )

    def tool(self, name: str) -> FilePin:
        for candidate, pin in self.tools:
            if candidate == name:
                return pin
        raise OpenEcologyArchiveError(f"missing remote tool authority: {name}")

    def helper_sha256(self, relative_path: str) -> str:
        for candidate, digest in self.helpers:
            if candidate == relative_path:
                return digest
        raise OpenEcologyArchiveError(
            f"missing remote helper authority: {relative_path}"
        )

    def json_text(self) -> str:
        return json.dumps(
            {
                "authority_sha256": self.authority_sha256,
                "helpers": dict(self.helpers),
                "schema_version": ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION,
                "tools": {name: pin.receipt_record() for name, pin in self.tools},
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def receipt_record(self) -> dict[str, object]:
        return {
            "authority_sha256": self.authority_sha256,
            "helpers": dict(self.helpers),
            "tools": {name: pin.receipt_record() for name, pin in self.tools},
        }


@dataclass(frozen=True, slots=True)
class RemoteArchiveOptions:
    ssh_target: str
    remote_repository_root: str
    remote_active_campaign_root: str
    remote_closed_bundle_dir: str
    remote_staging_dir: str
    campaign_id: str
    bundle_id: str
    source_git_sha: str
    source_manifest_sha256: str
    expected_marker_sha256: str
    expected_entry_count: int
    expected_file_count: int
    expected_directory_count: int
    expected_total_file_bytes: int
    receipt_path: Path
    tool_authority_path: Path
    tool_authority_sha256: str
    limits: CampaignStorageLimits = CampaignStorageLimits()
    rclone_base: str = DEFAULT_RCLONE_BASE

    def validate(self) -> None:
        if (
            not isinstance(self.ssh_target, str)
            or not self.ssh_target
            or self.ssh_target.startswith("-")
            or _SSH_TARGET_PATTERN.fullmatch(self.ssh_target) is None
        ):
            raise OpenEcologyArchiveError("ssh target contains unsupported characters")
        repository = _absolute_posix_path(
            self.remote_repository_root,
            field="remote repository root",
        )
        active = _absolute_posix_path(
            self.remote_active_campaign_root,
            field="remote active campaign root",
        )
        bundle = _absolute_posix_path(
            self.remote_closed_bundle_dir,
            field="remote closed bundle",
        )
        staging = _absolute_posix_path(
            self.remote_staging_dir,
            field="remote staging directory",
        )
        if repository == PurePosixPath("/"):
            raise OpenEcologyArchiveError(
                "remote repository root must identify an exact checkout"
            )
        for left_name, left, right_name, right in (
            ("repository", repository, "active campaign", active),
            ("repository", repository, "closed bundle", bundle),
            ("repository", repository, "staging directory", staging),
            ("active campaign", active, "closed bundle", bundle),
            ("active campaign", active, "staging directory", staging),
            ("closed bundle", bundle, "staging directory", staging),
        ):
            if _trees_overlap(left, right):
                raise OpenEcologyArchiveError(
                    f"{left_name} and {right_name} must be separate trees"
                )
        _identifier(self.campaign_id, field="campaign id")
        _identifier(self.bundle_id, field="bundle id")
        if bundle.name != self.bundle_id:
            raise OpenEcologyArchiveError(
                "remote closed-bundle directory name must equal bundle id"
            )
        _git_sha(self.source_git_sha)
        _sha256(self.source_manifest_sha256, field="source manifest SHA256")
        _sha256(self.expected_marker_sha256, field="expected marker SHA256")
        expected_counts = _closed_bundle_counts(
            entry_count=self.expected_entry_count,
            file_count=self.expected_file_count,
            directory_count=self.expected_directory_count,
            total_file_bytes=self.expected_total_file_bytes,
            field="expected closed bundle",
        )
        if expected_counts["entry_count"] == 0:
            raise OpenEcologyArchiveError("expected closed bundle cannot be empty")
        _sha256(self.tool_authority_sha256, field="tool authority SHA256")
        authority_path = Path(self.tool_authority_path)
        if not authority_path.is_absolute():
            raise OpenEcologyArchiveError("tool authority path must be absolute")
        self.limits.validate()
        if self.rclone_base != DEFAULT_RCLONE_BASE:
            raise OpenEcologyArchiveError(
                "open-ecology Drive base is sealed and may not be changed"
            )
        receipt = Path(self.receipt_path)
        if not receipt.is_absolute():
            raise OpenEcologyArchiveError("receipt path must be absolute")
        _require_canonical_directory(receipt.parent, field="receipt parent")
        if receipt.exists() or receipt.is_symlink():
            raise OpenEcologyArchiveError(
                f"refusing to overwrite immutable local receipt: {receipt}"
            )


def execute_remote_archive(options: RemoteArchiveOptions) -> dict[str, object]:
    """Run every upload and verification gate, then write one receipt."""

    options.validate()
    authority = _load_and_validate_authority(options)
    _require_effective_ssh_endpoint(authority)
    _verify_remote_authority(options, authority)
    drive_free_before_bytes = _require_drive_quota(
        options.limits.min_remote_free_bytes,
        authority=authority,
    )
    build = _remote_build(options, authority=authority)
    objects = _validated_remote_build(
        build,
        options=options,
        authority=authority,
    )
    expected = {str(row["name"]): row for row in objects}
    expected_names = tuple(sorted(expected))
    destination_prefix = (
        f"{DEFAULT_RCLONE_BASE}/{options.campaign_id}/{options.bundle_id}"
    )

    initial = _inspect_remote_inventory(
        destination_prefix,
        expected_names=expected_names,
        require_complete=False,
        authority=authority,
    )
    initial_by_name = {str(row["name"]): row for row in initial}
    for name, inventory_row in initial_by_name.items():
        observed = _rclone_readback(
            f"{destination_prefix}/{name}",
            authority=authority,
            max_bytes=int(expected[name]["size"]),
        )
        _require_object_match(
            observed,
            expected[name],
            field=f"existing Drive object {name}",
        )
        if inventory_row["size"] != expected[name]["size"]:
            raise OpenEcologyArchiveError(
                f"existing Drive inventory size mismatch for {name}"
            )

    for name in expected_names:
        if name in initial_by_name:
            continue
        source = expected[name]
        _stream_remote_object(
            ssh_target=options.ssh_target,
            source_path=str(source["source_path"]),
            source_size=int(source["size"]),
            destination_path=f"{destination_prefix}/{name}",
            authority=authority,
        )

    verification_inventory = _inspect_remote_inventory(
        destination_prefix,
        expected_names=expected_names,
        require_complete=True,
        authority=authority,
    )
    readbacks: dict[str, dict[str, object]] = {}
    for name in expected_names:
        observed = _rclone_readback(
            f"{destination_prefix}/{name}",
            authority=authority,
            max_bytes=int(expected[name]["size"]),
        )
        _require_object_match(
            observed,
            expected[name],
            field=f"Drive readback {name}",
        )
        readbacks[name] = observed

    check_rows = _run_independent_rclone_check(
        destination_prefix,
        expected=expected,
        authority=authority,
    )
    final_inventory = _inspect_remote_inventory(
        destination_prefix,
        expected_names=expected_names,
        require_complete=True,
        authority=authority,
    )
    if final_inventory != verification_inventory:
        raise OpenEcologyArchiveError(
            "Drive object IDs, names, or sizes changed during verification"
        )
    final_by_name = {str(row["name"]): row for row in final_inventory}
    for name in expected_names:
        if final_by_name[name]["size"] != expected[name]["size"]:
            raise OpenEcologyArchiveError(
                f"final Drive inventory size mismatch for {name}"
            )
    drive_free_after_bytes = _require_drive_quota(
        options.limits.min_remote_free_bytes,
        authority=authority,
    )
    _verify_remote_authority(options, authority)
    _require_effective_ssh_endpoint(authority)
    verify_local_authority_files(authority)

    payload: dict[str, object] = {
        "archive": {
            "archive_name": build["archive_name"],
            "objects": [
                {
                    "drive_id": final_by_name[name]["id"],
                    "name": name,
                    "sha256": expected[name]["sha256"],
                    "size": expected[name]["size"],
                }
                for name in expected_names
            ],
        },
        "archive_tool_authority": authority.receipt_record(),
        "bundle_id": options.bundle_id,
        "campaign_id": options.campaign_id,
        "closed_bundle": build["closed_bundle"],
        "destination_prefix": destination_prefix,
        "drive_quota": {
            "free_after_bytes": drive_free_after_bytes,
            "free_before_bytes": drive_free_before_bytes,
            "minimum_free_bytes": options.limits.min_remote_free_bytes,
        },
        "local_payload_staged": False,
        "pruning_available": False,
        "rclone_check": {
            "checkfile": "SHA-256",
            "combined_rows": check_rows,
            "matched_objects": 3,
            "one_way": False,
        },
        "readback": {
            name: {
                "sha256": readbacks[name]["sha256"],
                "size": readbacks[name]["size"],
            }
            for name in expected_names
        },
        "remote_source_bytes_deleted": False,
        "schema_version": REMOTE_RECEIPT_SCHEMA_VERSION,
        "source": {
            "git_sha": options.source_git_sha,
            "manifest_sha256": options.source_manifest_sha256,
            "repository_root": options.remote_repository_root,
        },
        "status": "three_objects_byte_readback_and_rclone_check_verified",
    }
    envelope = write_verified_receipt(options.receipt_path, payload)
    return {
        "destination_prefix": destination_prefix,
        "receipt_path": str(options.receipt_path),
        "receipt_payload_sha256": envelope["payload_sha256"],
        "status": payload["status"],
    }


def build_remote_closed_bundle_archive(
    *,
    repository_root: Path,
    active_campaign_root: Path,
    closed_bundle_dir: Path,
    staging_dir: Path,
    campaign_id: str,
    bundle_id: str,
    source_git_sha: str,
    source_manifest_sha256: str,
    expected_marker_sha256: str,
    expected_entry_count: int,
    expected_file_count: int,
    expected_directory_count: int,
    expected_total_file_bytes: int,
    limits: CampaignStorageLimits,
    execution_authority: RemoteBuildAuthority,
) -> dict[str, object]:
    """Remote helper: validate source/bundle and produce the exact triple."""

    repository = _require_exact_source_binding(
        repository_root,
        source_git_sha=source_git_sha,
        source_manifest_sha256=source_manifest_sha256,
        execution_authority=execution_authority,
    )
    active = _require_canonical_directory(
        active_campaign_root,
        field="active campaign root",
    )
    bundle = _require_canonical_directory(
        closed_bundle_dir,
        field="closed bundle",
    )
    _require_separate_local_trees(repository, active)
    _require_separate_local_trees(repository, bundle)
    _require_separate_local_trees(active, bundle)
    staging = _prepare_staging_directory(staging_dir, active=active, bundle=bundle)
    _require_separate_local_trees(repository, staging)
    limits.validate()

    with CampaignStorageLock(
        default_storage_lock_path(active),
        campaign_id=campaign_id,
        source_git_sha=source_git_sha,
    ):
        active_scan = check_campaign_storage(
            active,
            campaign_id=campaign_id,
            source_git_sha=source_git_sha,
            limits=limits,
        )
        snapshot = validate_closed_bundle(
            active,
            bundle,
            campaign_id=campaign_id,
            bundle_id=bundle_id,
            source_git_sha=source_git_sha,
            source_manifest_sha256=source_manifest_sha256,
            limits=limits,
        )
        expected_closed_bundle = {
            "marker_sha256": _sha256(
                expected_marker_sha256,
                field="expected marker SHA256",
            ),
            **_closed_bundle_counts(
                entry_count=expected_entry_count,
                file_count=expected_file_count,
                directory_count=expected_directory_count,
                total_file_bytes=expected_total_file_bytes,
                field="expected closed bundle",
            ),
        }
        observed_closed_bundle = _closed_bundle_contract(snapshot)
        if observed_closed_bundle != expected_closed_bundle:
            raise OpenEcologyArchiveError(
                "closed bundle differs from the externally pinned closure"
            )
        if (
            active_scan.total_file_bytes + snapshot.scan.total_file_bytes
            > limits.max_campaign_bytes
        ):
            raise OpenEcologyArchiveError(
                "active root plus closed bundle exceeds the global campaign budget"
            )
        archive_name = f"{bundle_id}-{snapshot.marker_sha256[:16]}.tar.zst"
        object_paths = (
            staging / archive_name,
            staging / f"{archive_name}.manifest.json",
            staging / f"{archive_name}.sha256",
        )
        present = _staging_object_names(staging)
        expected_names = {path.name for path in object_paths}
        if present and present != expected_names:
            raise OpenEcologyArchiveError(
                "remote staging prefix is partial or contains surplus evidence"
            )
        if not present:
            _run_archive_producer(
                repository=repository,
                bundle=bundle,
                staging=staging,
                archive_name=archive_name,
                execution_authority=execution_authority,
            )
        records = _validate_producer_objects(
            snapshot,
            object_paths,
            execution_authority=execution_authority,
        )
        repeated = validate_closed_bundle(
            active,
            bundle,
            campaign_id=campaign_id,
            bundle_id=bundle_id,
            source_git_sha=source_git_sha,
            source_manifest_sha256=source_manifest_sha256,
            limits=limits,
        )
        if repeated.producer_manifest != snapshot.producer_manifest:
            raise OpenEcologyArchiveError(
                "closed bundle changed during archive production"
            )
    _require_exact_source_binding(
        repository,
        source_git_sha=source_git_sha,
        source_manifest_sha256=source_manifest_sha256,
        execution_authority=execution_authority,
    )
    return {
        "archive_name": archive_name,
        "bundle_id": bundle_id,
        "campaign_id": campaign_id,
        "closed_bundle": observed_closed_bundle,
        "objects": records,
        "remote_input_pruned": False,
        "remote_staging_pruned": False,
        "schema_version": REMOTE_BUILD_SCHEMA_VERSION,
        "source_git_sha": source_git_sha,
        "source_manifest_sha256": source_manifest_sha256,
        "status": "deterministic_tar_zst_triple_validated",
        "tool_authority": execution_authority.receipt_record(),
    }


def _closed_bundle_contract(
    snapshot: ClosedBundleSnapshot,
) -> dict[str, object]:
    return {
        "marker_sha256": snapshot.marker_sha256,
        "entry_count": len(snapshot.scan.entries),
        "file_count": snapshot.scan.file_count,
        "directory_count": snapshot.scan.directory_count,
        "total_file_bytes": snapshot.scan.total_file_bytes,
    }


def _run_archive_producer(
    *,
    repository: Path,
    bundle: Path,
    staging: Path,
    archive_name: str,
    execution_authority: RemoteBuildAuthority,
) -> None:
    tool_path = repository / "scripts" / "archive_evolution_outputs.py"
    python_pin = execution_authority.tool("python")
    zstd_pin = execution_authority.tool("zstd")
    _verify_remote_execution_authority(execution_authority, repository=repository)
    completed = _run_pinned(
        (
            python_pin.path,
            str(tool_path),
            "--input-dir",
            str(bundle),
            "--output-dir",
            str(staging),
            "--archive-name",
            archive_name,
            "--archive-only",
            "--zstd-path",
            zstd_pin.path,
            "--zstd-sha256",
            zstd_pin.sha256,
        ),
        pin=python_pin,
        timeout_seconds=_TRANSFER_PROCESS_TIMEOUT_SECONDS,
    )
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenEcologyArchiveError(
            "archive_evolution_outputs.py did not emit valid JSON"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("archive_name") != archive_name
        or payload.get("status") != "local_archive_verified"
        or payload.get("prune_requested") is not False
        or payload.get("archive_only") is not True
        or payload.get("tool_authority")
        != {
            "rclone": None,
            "zstd": zstd_pin.receipt_record(),
        }
    ):
        raise OpenEcologyArchiveError(
            "archive_evolution_outputs.py result contract mismatch"
        )


def _validate_producer_objects(
    snapshot: ClosedBundleSnapshot,
    object_paths: Sequence[Path],
    *,
    execution_authority: RemoteBuildAuthority | None = None,
) -> list[dict[str, object]]:
    archive_path, manifest_path, sidecar_path = object_paths
    for path in object_paths:
        if path.is_symlink():
            raise OpenEcologyArchiveError("producer object may not be a symlink")
        metadata = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
        ):
            raise OpenEcologyArchiveError(
                "producer output is not one positive regular file"
            )
    manifest_bytes = _read_regular_file(manifest_path)
    try:
        producer_manifest = json.loads(
            manifest_bytes,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenEcologyArchiveError("producer manifest is not strict JSON") from exc
    if producer_manifest != snapshot.producer_manifest:
        raise OpenEcologyArchiveError(
            "producer manifest differs from descriptor-safe preflight snapshot"
        )
    if manifest_bytes != canonical_json_bytes(snapshot.producer_manifest):
        raise OpenEcologyArchiveError(
            "producer manifest bytes are not the expected canonical form"
        )
    archive_sha256 = _validate_archive_against_snapshot(
        archive_path,
        snapshot,
        execution_authority=execution_authority,
    )
    expected_sidecar = f"{archive_sha256}  {archive_path.name}\n".encode()
    if _read_regular_file(sidecar_path) != expected_sidecar:
        raise OpenEcologyArchiveError("archive SHA sidecar is missing or mismatched")

    records: list[dict[str, object]] = []
    for path in object_paths:
        os.chmod(path, stat.S_IMODE(path.stat().st_mode) & ~0o222)
        records.append(
            {
                "name": path.name,
                "sha256": _sha256_path(path),
                "size": path.stat().st_size,
                "source_path": str(path),
            }
        )
    return records


def _validate_archive_against_snapshot(
    archive_path: Path,
    snapshot: ClosedBundleSnapshot,
    *,
    execution_authority: RemoteBuildAuthority | None,
) -> str:
    initial = os.lstat(archive_path)
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
        or initial.st_size <= 0
    ):
        raise OpenEcologyArchiveError("archive is not one positive regular file")
    descriptor = os.open(
        archive_path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    process: subprocess.Popen[bytes] | None = None
    guard: _BoundedProcess | None = None
    try:
        opened = os.fstat(descriptor)
        if not _same_regular_file(initial, opened):
            raise OpenEcologyArchiveError("archive changed while opening")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _READ_CHUNK_SIZE):
            digest.update(chunk)
        if not _same_regular_file(opened, os.fstat(descriptor)):
            raise OpenEcologyArchiveError("archive changed while hashing")
        os.lseek(descriptor, 0, os.SEEK_SET)

        if execution_authority is None:
            zstd_path = shutil.which("zstd")
            if zstd_path is None:
                raise OpenEcologyArchiveError(
                    "zstd is required to validate staged archive members"
                )
            environment = None
            zstd_pin = None
        else:
            zstd_pin = execution_authority.tool("zstd")
            verify_pinned_file(zstd_pin, executable=True, require_nonempty=True)
            zstd_path = zstd_pin.path
            environment = minimal_subprocess_env()
        try:
            process = subprocess.Popen(
                (zstd_path, "--decompress", "--stdout", "--quiet"),
                stdin=descriptor,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                start_new_session=True,
            )
        except OSError as exc:
            raise OpenEcologyArchiveError(
                f"cannot start archive decompressor: {exc}"
            ) from exc
        if process.stdout is None:
            _terminate_process_group(process)
            raise OpenEcologyArchiveError("archive decompressor did not expose stdout")
        guard = _BoundedProcess(
            process,
            command_name="staged archive decompression",
            timeout_seconds=_TRANSFER_PROCESS_TIMEOUT_SECONDS,
            stdout_limit=0,
            stderr_limit=_MAX_CAPTURE_STDERR_BYTES,
            drain_stdout=False,
        )
        validation_error: BaseException | None = None
        try:
            _validate_tar_stream(process.stdout, snapshot)
        except BaseException as exc:
            validation_error = exc
        finally:
            process.stdout.close()
        if validation_error is not None:
            guard.abort()
            if isinstance(validation_error, OpenEcologyArchiveError):
                raise validation_error
            raise OpenEcologyArchiveError(
                f"staged archive member validation failed: {validation_error}"
            ) from validation_error
        _, stderr = guard.complete()
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise OpenEcologyArchiveError(
                "staged archive decompression failed: "
                + (detail or f"exit {process.returncode}")
            )
        if zstd_pin is not None:
            verify_pinned_file(zstd_pin, executable=True, require_nonempty=True)
        finished = os.fstat(descriptor)
        current = os.lstat(archive_path)
        if not _same_regular_file(opened, finished) or not _same_regular_file(
            opened,
            current,
        ):
            raise OpenEcologyArchiveError("archive changed during member validation")
        return digest.hexdigest()
    except BaseException:
        if guard is not None and not guard.finished.is_set():
            guard.abort()
        elif process is not None and process.returncode is None:
            _terminate_process_group(process)
        raise
    finally:
        os.close(descriptor)


def _validate_tar_stream(
    stream: object,
    snapshot: ClosedBundleSnapshot,
) -> None:
    expected = [
        (
            snapshot.root.name,
            "directory",
            0o755,
            0,
            None,
        ),
        *[
            (
                f"{snapshot.root.name}/{entry.path}",
                entry.kind,
                entry.mode,
                entry.size,
                entry.sha256,
            )
            for entry in snapshot.scan.entries
        ],
    ]
    observed_count = 0
    try:
        with tarfile.open(fileobj=stream, mode="r|") as archive:  # type: ignore[arg-type]
            for member in archive:
                if observed_count >= len(expected):
                    raise OpenEcologyArchiveError(
                        "staged archive contains surplus members"
                    )
                (
                    expected_name,
                    expected_kind,
                    expected_mode,
                    expected_size,
                    expected_sha256,
                ) = expected[observed_count]
                observed_count += 1
                member_path = PurePosixPath(member.name)
                if (
                    member_path.is_absolute()
                    or any(part in {"", ".", ".."} for part in member_path.parts)
                    or member.name != expected_name
                ):
                    raise OpenEcologyArchiveError(
                        "staged archive member path/order mismatch"
                    )
                observed_kind = (
                    "file"
                    if member.isreg()
                    else "directory"
                    if member.isdir()
                    else "unsupported"
                )
                if observed_kind != expected_kind:
                    raise OpenEcologyArchiveError(
                        f"staged archive member type mismatch: {member.name}"
                    )
                if (
                    stat.S_IMODE(member.mode) != expected_mode
                    or member.size != expected_size
                ):
                    raise OpenEcologyArchiveError(
                        f"staged archive member metadata mismatch: {member.name}"
                    )
                if expected_kind != "file":
                    continue
                payload = archive.extractfile(member)
                if payload is None:
                    raise OpenEcologyArchiveError(
                        f"staged archive file payload is missing: {member.name}"
                    )
                digest = hashlib.sha256()
                byte_count = 0
                while chunk := payload.read(_READ_CHUNK_SIZE):
                    byte_count += len(chunk)
                    if byte_count > expected_size:
                        raise OpenEcologyArchiveError(
                            f"staged archive file exceeds manifest size: {member.name}"
                        )
                    digest.update(chunk)
                if byte_count != expected_size or digest.hexdigest() != expected_sha256:
                    raise OpenEcologyArchiveError(
                        f"staged archive file digest mismatch: {member.name}"
                    )
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise OpenEcologyArchiveError(
            f"staged archive is not a valid tar stream: {exc}"
        ) from exc
    if observed_count != len(expected):
        raise OpenEcologyArchiveError("staged archive member set is incomplete")


def _remote_build(
    options: RemoteArchiveOptions,
    *,
    authority: ArchiveToolAuthority,
) -> dict[str, object]:
    remote_authority = RemoteBuildAuthority.from_archive_authority(authority)
    python_pin = remote_authority.tool("python")
    command = (
        python_pin.path,
        str(
            PurePosixPath(options.remote_repository_root)
            / "scripts"
            / "archive_open_ecology_campaign.py"
        ),
        "remote-build",
        "--repository-root",
        options.remote_repository_root,
        "--active-campaign-root",
        options.remote_active_campaign_root,
        "--closed-bundle-dir",
        options.remote_closed_bundle_dir,
        "--staging-dir",
        options.remote_staging_dir,
        "--campaign-id",
        options.campaign_id,
        "--bundle-id",
        options.bundle_id,
        "--source-git-sha",
        options.source_git_sha,
        "--source-manifest-sha256",
        options.source_manifest_sha256,
        "--expected-marker-sha256",
        options.expected_marker_sha256,
        "--expected-entry-count",
        str(options.expected_entry_count),
        "--expected-file-count",
        str(options.expected_file_count),
        "--expected-directory-count",
        str(options.expected_directory_count),
        "--expected-total-file-bytes",
        str(options.expected_total_file_bytes),
        "--remote-authority-json",
        remote_authority.json_text(),
        "--max-campaign-bytes",
        str(options.limits.max_campaign_bytes),
        "--min-campaign-free-bytes",
        str(options.limits.min_campaign_free_bytes),
        "--min-remote-free-bytes",
        str(options.limits.min_remote_free_bytes),
        "--max-entries",
        str(options.limits.max_entries),
    )
    completed = _run_local_tool(
        authority,
        "ssh",
        _ssh_command(authority, command),
        timeout_seconds=_TRANSFER_PROCESS_TIMEOUT_SECONDS,
    )
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenEcologyArchiveError("remote build did not emit valid JSON") from exc
    if not isinstance(payload, dict):
        raise OpenEcologyArchiveError("remote build JSON root must be an object")
    return payload


def _validated_remote_build(
    payload: Mapping[str, object],
    *,
    options: RemoteArchiveOptions,
    authority: ArchiveToolAuthority | None = None,
) -> list[dict[str, object]]:
    if set(payload) != {
        "archive_name",
        "bundle_id",
        "campaign_id",
        "closed_bundle",
        "objects",
        "remote_input_pruned",
        "remote_staging_pruned",
        "schema_version",
        "source_git_sha",
        "source_manifest_sha256",
        "status",
        "tool_authority",
    }:
        raise OpenEcologyArchiveError("remote build schema is not exact")
    if (
        payload.get("schema_version") != REMOTE_BUILD_SCHEMA_VERSION
        or payload.get("status") != "deterministic_tar_zst_triple_validated"
        or payload.get("campaign_id") != options.campaign_id
        or payload.get("bundle_id") != options.bundle_id
        or payload.get("source_git_sha") != options.source_git_sha
        or payload.get("source_manifest_sha256") != options.source_manifest_sha256
        or payload.get("remote_input_pruned") is not False
        or payload.get("remote_staging_pruned") is not False
    ):
        raise OpenEcologyArchiveError("remote build provenance mismatch")
    if authority is not None:
        expected_remote_authority = RemoteBuildAuthority.from_archive_authority(
            authority
        ).receipt_record()
        if payload.get("tool_authority") != expected_remote_authority:
            raise OpenEcologyArchiveError("remote build tool authority mismatch")
    expected_closed_bundle = {
        "marker_sha256": options.expected_marker_sha256,
        "entry_count": options.expected_entry_count,
        "file_count": options.expected_file_count,
        "directory_count": options.expected_directory_count,
        "total_file_bytes": options.expected_total_file_bytes,
    }
    raw_closed_bundle = payload.get("closed_bundle")
    if not isinstance(raw_closed_bundle, dict) or set(raw_closed_bundle) != set(
        expected_closed_bundle
    ):
        raise OpenEcologyArchiveError(
            "remote build closed-bundle contract is malformed"
        )
    marker_sha256 = _sha256(
        raw_closed_bundle.get("marker_sha256"),
        field="remote marker SHA256",
    )
    observed_counts = _closed_bundle_counts(
        entry_count=raw_closed_bundle.get("entry_count"),
        file_count=raw_closed_bundle.get("file_count"),
        directory_count=raw_closed_bundle.get("directory_count"),
        total_file_bytes=raw_closed_bundle.get("total_file_bytes"),
        field="remote closed bundle",
    )
    observed_closed_bundle = {
        "marker_sha256": marker_sha256,
        **observed_counts,
    }
    if observed_closed_bundle != expected_closed_bundle:
        raise OpenEcologyArchiveError(
            "remote build differs from the externally pinned closure"
        )
    archive_name = payload.get("archive_name")
    expected_archive_name = f"{options.bundle_id}-{str(marker_sha256)[:16]}.tar.zst"
    if not isinstance(archive_name, str) or archive_name != expected_archive_name:
        raise OpenEcologyArchiveError("remote archive name is invalid")
    expected_names = {
        archive_name,
        f"{archive_name}.manifest.json",
        f"{archive_name}.sha256",
    }
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or len(raw_objects) != 3:
        raise OpenEcologyArchiveError(
            "remote build must describe exactly three objects"
        )
    objects: list[dict[str, object]] = []
    seen: set[str] = set()
    expected_staging = PurePosixPath(options.remote_staging_dir)
    for raw in raw_objects:
        if not isinstance(raw, dict):
            raise OpenEcologyArchiveError("remote build object must be a mapping")
        name = raw.get("name")
        source_path = raw.get("source_path")
        size = raw.get("size")
        sha256 = raw.get("sha256")
        parsed_source_path = (
            _absolute_posix_path(source_path, field=f"remote object {name} source path")
            if isinstance(source_path, str)
            else None
        )
        if (
            not isinstance(name, str)
            or name not in expected_names
            or name in seen
            or parsed_source_path is None
            or parsed_source_path.name != name
            or parsed_source_path.parent != expected_staging
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise OpenEcologyArchiveError("remote build object evidence is invalid")
        _sha256(sha256, field=f"remote object {name} SHA256")
        seen.add(name)
        objects.append(
            {
                "name": name,
                "sha256": sha256,
                "size": size,
                "source_path": str(parsed_source_path),
            }
        )
    if seen != expected_names:
        raise OpenEcologyArchiveError("remote build object set is incomplete")
    return sorted(objects, key=lambda row: str(row["name"]))


def _inspect_remote_inventory(
    destination_prefix: str,
    *,
    expected_names: Sequence[str],
    require_complete: bool,
    authority: ArchiveToolAuthority | None = None,
) -> list[dict[str, object]]:
    command = (
        _rclone_command(
            authority,
            "lsjson",
            destination_prefix,
            "--max-depth",
            "1",
        )
        if authority is not None
        else (
            "rclone",
            "lsjson",
            destination_prefix,
            "--max-depth",
            "1",
        )
    )
    completed = (
        _run_local_tool(
            authority,
            "rclone",
            command,
            allow_missing_remote=True,
        )
        if authority is not None
        else _run(command, allow_missing_remote=True)
    )
    if completed.returncode != 0:
        payload: object = []
    else:
        try:
            payload = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenEcologyArchiveError("Drive inventory is not valid JSON") from exc
    if not isinstance(payload, list):
        raise OpenEcologyArchiveError("Drive inventory root must be a list")
    allowed = set(expected_names)
    records: list[dict[str, object]] = []
    names: set[str] = set()
    ids: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise OpenEcologyArchiveError("Drive inventory row must be an object")
        name = raw.get("Name")
        inventory_path = raw.get("Path")
        size = raw.get("Size")
        drive_id = raw.get("ID")
        if (
            raw.get("IsDir") is not False
            or raw.get("IsLink") is True
            or not isinstance(name, str)
            or inventory_path != name
            or "/" in name
            or name not in allowed
            or name in names
        ):
            raise OpenEcologyArchiveError(
                "Drive inventory has a duplicate or surplus object name"
            )
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(drive_id, str)
            or _DRIVE_ID_PATTERN.fullmatch(drive_id) is None
            or drive_id in ids
        ):
            raise OpenEcologyArchiveError(
                "Drive inventory has invalid size or duplicate/empty object ID"
            )
        names.add(name)
        ids.add(drive_id)
        records.append({"id": drive_id, "name": name, "size": size})
    if require_complete and names != allowed:
        raise OpenEcologyArchiveError(
            "Drive inventory is not the exact three-object bundle"
        )
    return sorted(records, key=lambda row: str(row["name"]))


def _stream_remote_object(
    *,
    ssh_target: str,
    source_path: str,
    source_size: int,
    destination_path: str,
    authority: ArchiveToolAuthority,
) -> None:
    if ssh_target != authority.ssh_target:
        raise OpenEcologyArchiveError("SSH stream target differs from authority")
    reader = (
        "import os,sys\n"
        "p=sys.argv[1]\n"
        "expected=int(sys.argv[2])\n"
        "fd=os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))\n"
        "try:\n"
        " s=os.fstat(fd)\n"
        " if (not __import__('stat').S_ISREG(s.st_mode) or s.st_nlink!=1"
        " or s.st_size!=expected):\n"
        "  raise SystemExit(3)\n"
        " sent=0\n"
        " while True:\n"
        "  b=os.read(fd,8*1024*1024)\n"
        "  if not b: break\n"
        "  sent+=len(b)\n"
        "  if sent>expected: raise SystemExit(4)\n"
        "  sys.stdout.buffer.write(b)\n"
        " if sent!=expected: raise SystemExit(5)\n"
        "finally: os.close(fd)\n"
    )
    try:
        remote_python = authority.remote_tool("python")
        verify_pinned_file(
            authority.local_tool("ssh"),
            executable=True,
            require_nonempty=True,
        )
        ssh_process = subprocess.Popen(
            _ssh_command(
                authority,
                (remote_python.path, "-c", reader, source_path, str(source_size)),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=minimal_subprocess_env(),
            start_new_session=True,
        )
    except OSError as exc:
        raise OpenEcologyArchiveError(f"cannot start SSH object stream: {exc}") from exc
    if ssh_process.stdout is None:
        _terminate_process_group(ssh_process)
        raise OpenEcologyArchiveError("SSH object stream did not expose stdout")
    try:
        rclone_pin = authority.local_tool("rclone")
        verify_pinned_file(rclone_pin, executable=True, require_nonempty=True)
        verify_pinned_file(
            authority.rclone_config,
            executable=False,
            require_nonempty=True,
        )
        rclone_process = subprocess.Popen(
            _rclone_command(
                authority,
                "rcat",
                destination_path,
                "--size",
                str(source_size),
                "--immutable",
            ),
            stdin=ssh_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=minimal_subprocess_env(),
            start_new_session=True,
        )
    except OSError as exc:
        _terminate_process_group(ssh_process)
        raise OpenEcologyArchiveError(f"cannot start rclone upload: {exc}") from exc
    ssh_process.stdout.close()
    ssh_guard = _BoundedProcess(
        ssh_process,
        command_name="SSH object stream",
        timeout_seconds=_TRANSFER_PROCESS_TIMEOUT_SECONDS,
        stdout_limit=_MAX_CAPTURE_STDOUT_BYTES,
        stderr_limit=_MAX_CAPTURE_STDERR_BYTES,
        drain_stdout=False,
    )
    rclone_guard = _BoundedProcess(
        rclone_process,
        command_name="rclone immutable upload",
        timeout_seconds=_TRANSFER_PROCESS_TIMEOUT_SECONDS,
        stdout_limit=_MAX_CAPTURE_STDOUT_BYTES,
        stderr_limit=_MAX_CAPTURE_STDERR_BYTES,
    )
    try:
        _, rclone_stderr = rclone_guard.complete()
        _, ssh_stderr = ssh_guard.complete()
    except BaseException:
        rclone_guard.abort()
        ssh_guard.abort()
        raise
    verify_pinned_file(
        authority.local_tool("ssh"),
        executable=True,
        require_nonempty=True,
    )
    verify_pinned_file(rclone_pin, executable=True, require_nonempty=True)
    verify_pinned_file(
        authority.rclone_config,
        executable=False,
        require_nonempty=True,
    )
    if ssh_process.returncode != 0 or rclone_process.returncode != 0:
        raise OpenEcologyArchiveError(
            "immutable object upload failed: "
            f"ssh_exit={ssh_process.returncode} "
            f"rclone_exit={rclone_process.returncode} "
            f"ssh_stderr={ssh_stderr.decode(errors='replace').strip()!r} "
            f"rclone_stderr={rclone_stderr.decode(errors='replace').strip()!r}"
        )


def _rclone_readback(
    destination_path: str,
    *,
    authority: ArchiveToolAuthority | None = None,
    max_bytes: int | None = None,
) -> dict[str, object]:
    if isinstance(max_bytes, bool) or (
        max_bytes is not None and (not isinstance(max_bytes, int) or max_bytes <= 0)
    ):
        raise OpenEcologyArchiveError("Drive readback byte cap must be positive")
    rclone_pin = authority.local_tool("rclone") if authority is not None else None
    if rclone_pin is not None:
        verify_pinned_file(rclone_pin, executable=True, require_nonempty=True)
        verify_pinned_file(
            authority.rclone_config,
            executable=False,
            require_nonempty=True,
        )
    try:
        process = subprocess.Popen(
            (
                _rclone_command(authority, "cat", destination_path)
                if authority is not None
                else ("rclone", "cat", destination_path)
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=minimal_subprocess_env(),
            start_new_session=True,
        )
    except OSError as exc:
        raise OpenEcologyArchiveError(f"cannot start Drive readback: {exc}") from exc
    if process.stdout is None:
        _terminate_process_group(process)
        raise OpenEcologyArchiveError("Drive readback did not expose stdout")
    digest = hashlib.sha256()
    guard = _BoundedProcess(
        process,
        command_name="Drive full-byte readback",
        timeout_seconds=_TRANSFER_PROCESS_TIMEOUT_SECONDS,
        stdout_limit=(
            max_bytes
            if max_bytes is not None
            else CampaignStorageLimits().max_campaign_bytes
        ),
        stderr_limit=_MAX_CAPTURE_STDERR_BYTES,
        stdout_consumer=digest.update,
    )
    _, stderr = guard.complete()
    size = next(
        drain.byte_count for drain in guard.drains if drain.stream_name == "stdout"
    )
    return_code = process.returncode
    if rclone_pin is not None:
        verify_pinned_file(rclone_pin, executable=True, require_nonempty=True)
        verify_pinned_file(
            authority.rclone_config,
            executable=False,
            require_nonempty=True,
        )
    if return_code != 0:
        raise OpenEcologyArchiveError(
            "Drive full-byte readback failed: "
            + stderr.decode(errors="replace").strip()
        )
    if max_bytes is not None and size != max_bytes:
        raise OpenEcologyArchiveError(
            "Drive full-byte readback size differs from the expected object size"
        )
    return {"sha256": digest.hexdigest(), "size": size}


def _run_independent_rclone_check(
    destination_prefix: str,
    *,
    expected: Mapping[str, Mapping[str, object]],
    authority: ArchiveToolAuthority | None = None,
) -> list[str]:
    names = sorted(expected)
    if len(names) != 3:
        raise OpenEcologyArchiveError("rclone check requires exactly three objects")
    sum_bytes = "".join(
        f"{expected[name]['sha256']}  {name}\n" for name in names
    ).encode()
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="open-ecology-sha256-",
        suffix=".sum",
    ) as handle:
        handle.write(sum_bytes)
        handle.flush()
        os.fsync(handle.fileno())
        command = (
            _rclone_command(
                authority,
                "check",
                handle.name,
                destination_prefix,
                "--checkfile",
                "SHA-256",
                "--combined",
                "-",
            )
            if authority is not None
            else (
                "rclone",
                "check",
                handle.name,
                destination_prefix,
                "--checkfile",
                "SHA-256",
                "--combined",
                "-",
            )
        )
        if "--one-way" in command:
            raise OpenEcologyArchiveError("independent rclone check may not be one-way")
        completed = (
            _run_local_tool(
                authority,
                "rclone",
                command,
                timeout_seconds=_TRANSFER_PROCESS_TIMEOUT_SECONDS,
            )
            if authority is not None
            else _run(
                command,
                timeout_seconds=_TRANSFER_PROCESS_TIMEOUT_SECONDS,
            )
        )
    try:
        rows = [line for line in completed.stdout.decode("utf-8").splitlines() if line]
    except UnicodeDecodeError as exc:
        raise OpenEcologyArchiveError(
            "rclone combined check output is not UTF-8"
        ) from exc
    expected_rows = [f"= {name}" for name in names]
    if sorted(rows) != sorted(expected_rows) or len(rows) != 3:
        raise OpenEcologyArchiveError(
            "rclone check did not emit exactly three canonical match rows"
        )
    return sorted(rows)


def _require_drive_quota(
    minimum_free_bytes: int,
    *,
    authority: ArchiveToolAuthority,
) -> int:
    completed = _run_local_tool(
        authority,
        "rclone",
        _rclone_command(authority, "about", "gdrive:", "--json"),
    )
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenEcologyArchiveError("rclone about did not emit valid JSON") from exc
    free = payload.get("free") if isinstance(payload, dict) else None
    if isinstance(free, bool) or not isinstance(free, int) or free < minimum_free_bytes:
        raise OpenEcologyArchiveError(
            "Google Drive free space is below the sealed campaign floor"
        )
    return free


def _require_object_match(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    field: str,
) -> None:
    if observed.get("sha256") != expected.get("sha256") or observed.get(
        "size"
    ) != expected.get("size"):
        raise OpenEcologyArchiveError(f"{field} SHA256 or size mismatch")


def _require_exact_source_binding(
    repository_root: Path,
    *,
    source_git_sha: str,
    source_manifest_sha256: str,
    execution_authority: RemoteBuildAuthority | None = None,
) -> Path:
    repository = _require_canonical_directory(
        repository_root,
        field="repository root",
    )
    if repository != _REPOSITORY_ROOT:
        raise OpenEcologyArchiveError(
            "archive helper is not running from the claimed repository root"
        )
    storage_module = sys.modules["evolution_sim.io.open_ecology_campaign_storage"]
    storage_path = Path(str(storage_module.__file__)).resolve()
    if not storage_path.is_relative_to(repository / "python"):
        raise OpenEcologyArchiveError(
            "open-ecology storage import root differs from exact checkout"
        )
    manifest_module = sys.modules["evolution_sim.io.source_manifest"]
    manifest_module_path = Path(str(manifest_module.__file__)).resolve()
    if not manifest_module_path.is_relative_to(repository / "python"):
        raise OpenEcologyArchiveError(
            "source-manifest import root differs from exact checkout"
        )
    producer_path = repository / "scripts" / "archive_evolution_outputs.py"
    if not producer_path.is_file() or producer_path.is_symlink():
        raise OpenEcologyArchiveError(
            "exact archive_evolution_outputs.py producer is missing"
        )
    if execution_authority is not None:
        _verify_remote_execution_authority(
            execution_authority,
            repository=repository,
        )
    observed_head = _git_output(
        repository,
        ("rev-parse", "HEAD"),
        execution_authority=execution_authority,
    ).strip()
    if observed_head != source_git_sha:
        raise OpenEcologyArchiveError("exact checkout HEAD differs from source pin")
    observed_branch = _git_output(
        repository,
        ("rev-parse", "--abbrev-ref", "HEAD"),
        execution_authority=execution_authority,
    ).strip()
    if observed_branch != "HEAD":
        raise OpenEcologyArchiveError(
            "exact checkout must be detached at the source pin"
        )
    status = _git_output(
        repository,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        execution_authority=execution_authority,
    )
    if status:
        raise OpenEcologyArchiveError("exact checkout is dirty")
    for relative_path in (
        PurePosixPath("scripts/archive_open_ecology_campaign.py"),
        PurePosixPath("scripts/archive_evolution_outputs.py"),
    ):
        _require_committed_file_bytes(
            repository,
            source_git_sha=source_git_sha,
            relative_path=relative_path,
            execution_authority=execution_authority,
        )
    if execution_authority is not None:
        for relative_path in REMOTE_HELPER_PATHS:
            verify_pinned_file(
                FilePin(
                    path=str(repository / PurePosixPath(relative_path)),
                    sha256=execution_authority.helper_sha256(relative_path),
                ),
                executable=False,
                require_nonempty=True,
            )
    observed_manifest = source_file_hash_manifest(repository).get("aggregate_sha256")
    if observed_manifest != source_manifest_sha256:
        raise OpenEcologyArchiveError("exact checkout source manifest drifted")
    return repository


def _git_output(
    repository: Path,
    arguments: Sequence[str],
    *,
    execution_authority: RemoteBuildAuthority | None = None,
) -> str:
    if execution_authority is None:
        completed = _run(("git", "-C", str(repository), *arguments))
    else:
        git_pin = execution_authority.tool("git")
        completed = _run_pinned(
            (git_pin.path, "-C", str(repository), *arguments),
            pin=git_pin,
        )
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OpenEcologyArchiveError("git output is not UTF-8") from exc


def _require_committed_file_bytes(
    repository: Path,
    *,
    source_git_sha: str,
    relative_path: PurePosixPath,
    execution_authority: RemoteBuildAuthority | None = None,
) -> None:
    worktree_path = repository / relative_path
    worktree_bytes = _read_regular_file(worktree_path)
    command = (
        (
            "git",
            "-C",
            str(repository),
            "show",
            f"{source_git_sha}:{relative_path.as_posix()}",
        )
        if execution_authority is None
        else (
            execution_authority.tool("git").path,
            "-C",
            str(repository),
            "show",
            f"{source_git_sha}:{relative_path.as_posix()}",
        )
    )
    if execution_authority is None:
        committed = _run(command).stdout
    else:
        committed = _run_pinned(
            command,
            pin=execution_authority.tool("git"),
        ).stdout
    if worktree_bytes != committed:
        raise OpenEcologyArchiveError(
            f"exact checkout file differs from committed bytes: {relative_path}"
        )


def _prepare_staging_directory(
    path: Path,
    *,
    active: Path,
    bundle: Path,
) -> Path:
    raw = Path(path)
    if raw.exists():
        staging = _require_canonical_directory(raw, field="staging directory")
    else:
        parent = _require_canonical_directory(raw.parent, field="staging parent")
        raw.mkdir(mode=0o700)
        staging = _require_canonical_directory(raw, field="staging directory")
        _fsync_directory(parent)
    _require_separate_local_trees(active, staging)
    _require_separate_local_trees(bundle, staging)
    return staging


def _staging_object_names(path: Path) -> set[str]:
    names: set[str] = set()
    for child in os.scandir(path):
        if child.name in names:
            raise OpenEcologyArchiveError("duplicate staging object name")
        metadata = os.lstat(child.path)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OpenEcologyArchiveError(
                "staging directory contains non-regular evidence"
            )
        names.add(child.name)
    return names


def _require_canonical_directory(path: Path, *, field: str) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise OpenEcologyArchiveError(f"{field} does not exist") from exc
    if raw != resolved:
        raise OpenEcologyArchiveError(
            f"{field} raw/resolved identity differs or has a symlink ancestor"
        )
    current = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise OpenEcologyArchiveError(
                f"{field} ancestor is a symlink or non-directory"
            )
    if resolved == Path(resolved.anchor):
        raise OpenEcologyArchiveError(f"{field} may not be a filesystem root")
    return resolved


def _require_separate_local_trees(left: Path, right: Path) -> None:
    if left == right or left in right.parents or right in left.parents:
        raise OpenEcologyArchiveError("archive trees must not overlap")


def _trees_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def _absolute_posix_path(value: str, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise OpenEcologyArchiveError(f"{field} must be one safe absolute path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value.rstrip("/"):
        raise OpenEcologyArchiveError(f"{field} must be normalized and absolute")
    return path


def _load_and_validate_authority(
    options: RemoteArchiveOptions,
) -> ArchiveToolAuthority:
    try:
        authority = load_archive_tool_authority(
            options.tool_authority_path,
            expected_sha256=options.tool_authority_sha256,
        )
    except ArchiveAuthorityError as exc:
        raise OpenEcologyArchiveError(str(exc)) from exc
    bindings = {
        "rclone base": (authority.rclone_base, options.rclone_base),
        "remote repository root": (
            authority.remote_repository_root,
            options.remote_repository_root,
        ),
        "source git SHA": (authority.source_git_sha, options.source_git_sha),
        "source manifest SHA256": (
            authority.source_manifest_sha256,
            options.source_manifest_sha256,
        ),
        "SSH target": (authority.ssh_target, options.ssh_target),
    }
    for binding_name, (observed, expected) in bindings.items():
        if observed != expected:
            raise OpenEcologyArchiveError(f"archive authority {binding_name} mismatch")
    return authority


def _require_effective_ssh_endpoint(authority: ArchiveToolAuthority) -> None:
    ssh_pin = authority.local_tool("ssh")
    try:
        verify_effective_ssh_config(authority, run_pinned=_run_pinned)
    except ArchiveAuthorityError as exc:
        raise OpenEcologyArchiveError(str(exc)) from exc
    verbose = _run_pinned(
        (
            ssh_pin.path,
            "-v",
            *SEALED_SSH_OPTIONS,
            authority.ssh_target,
            " ".join(
                shlex.quote(argument)
                for argument in (authority.remote_tool("env").path, "--version")
            ),
        ),
        pin=ssh_pin,
    )
    try:
        connection = parse_ssh_connection_identity(verbose.stderr)
    except ArchiveAuthorityError as exc:
        raise OpenEcologyArchiveError(str(exc)) from exc
    if connection != authority.ssh_connection:
        raise OpenEcologyArchiveError(
            "authenticated SSH host key/address differs from external authority pin"
        )


def _verify_remote_authority(
    options: RemoteArchiveOptions,
    authority: ArchiveToolAuthority,
) -> None:
    expected: dict[str, str] = {
        pin.path: pin.sha256 for _, pin in authority.remote_tools
    }
    repository = PurePosixPath(options.remote_repository_root)
    for relative_path, digest in authority.remote_helpers:
        expected[str(repository / relative_path)] = digest
    sha256sum_pin = authority.remote_tool("sha256sum")
    env_pin = authority.remote_tool("env")
    command = (
        env_pin.path,
        "-i",
        "LANG=C",
        "LC_ALL=C",
        "PATH=/usr/bin:/bin",
        sha256sum_pin.path,
        "--",
        *sorted(expected),
    )
    completed = _run_local_tool(
        authority,
        "ssh",
        _ssh_command(authority, command),
    )
    try:
        rows = completed.stdout.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise OpenEcologyArchiveError(
            "remote tool authority output is not UTF-8"
        ) from exc
    observed: dict[str, str] = {}
    for row in rows:
        parts = row.split("  ", 1)
        if len(parts) != 2:
            raise OpenEcologyArchiveError(
                "remote tool authority output is not canonical sha256sum"
            )
        digest, path = parts
        _sha256(digest, field=f"remote SHA256 for {path}")
        if path in observed:
            raise OpenEcologyArchiveError("duplicate remote authority path")
        observed[path] = digest
    if observed != expected:
        raise OpenEcologyArchiveError(
            "remote executable or helper differs from external authority"
        )


def _parse_remote_build_authority(value: str) -> RemoteBuildAuthority:
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenEcologyArchiveError("remote authority is not strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "authority_sha256",
        "helpers",
        "schema_version",
        "tools",
    }:
        raise OpenEcologyArchiveError("remote authority root contract is invalid")
    if payload["schema_version"] != ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION:
        raise OpenEcologyArchiveError("remote authority schema mismatch")
    authority_sha256 = _sha256(
        payload["authority_sha256"],
        field="remote authority SHA256",
    )
    raw_tools = payload["tools"]
    if not isinstance(raw_tools, dict) or set(raw_tools) != set(REMOTE_TOOL_NAMES):
        raise OpenEcologyArchiveError("remote tool set is incomplete")
    tools: list[tuple[str, FilePin]] = []
    for name in REMOTE_TOOL_NAMES:
        raw = raw_tools[name]
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
            raise OpenEcologyArchiveError(f"remote tool pin is invalid: {name}")
        path = str(_absolute_posix_path(raw["path"], field=f"remote {name} path"))
        digest = _sha256(raw["sha256"], field=f"remote {name} SHA256")
        tools.append((name, FilePin(path=path, sha256=digest)))
    raw_helpers = payload["helpers"]
    if not isinstance(raw_helpers, dict) or set(raw_helpers) != set(
        REMOTE_HELPER_PATHS
    ):
        raise OpenEcologyArchiveError("remote helper set is incomplete")
    helpers = tuple(
        (
            relative_path,
            _sha256(
                raw_helpers[relative_path],
                field=f"remote helper SHA256 for {relative_path}",
            ),
        )
        for relative_path in REMOTE_HELPER_PATHS
    )
    return RemoteBuildAuthority(
        authority_sha256=authority_sha256,
        tools=tuple(tools),
        helpers=helpers,
    )


def _verify_remote_execution_authority(
    authority: RemoteBuildAuthority,
    *,
    repository: Path,
) -> None:
    python_pin = authority.tool("python")
    if Path(sys.executable).resolve(strict=True) != Path(python_pin.path):
        raise OpenEcologyArchiveError(
            "remote archive helper interpreter differs from authority"
        )
    for _, pin in authority.tools:
        verify_pinned_file(pin, executable=True, require_nonempty=True)
    for relative_path, digest in authority.helpers:
        verify_pinned_file(
            FilePin(
                path=str(repository / PurePosixPath(relative_path)),
                sha256=digest,
            ),
            executable=False,
            require_nonempty=True,
        )


def _rclone_command(
    authority: ArchiveToolAuthority,
    *arguments: str,
) -> tuple[str, ...]:
    rclone_pin = authority.local_tool("rclone")
    return (
        rclone_pin.path,
        "--config",
        authority.rclone_config.path,
        *arguments,
    )


def _ssh_command(
    authority: ArchiveToolAuthority,
    arguments: Sequence[str],
) -> tuple[str, ...]:
    ssh_pin = authority.local_tool("ssh")
    return (
        ssh_pin.path,
        *SEALED_SSH_OPTIONS,
        authority.ssh_target,
        " ".join(shlex.quote(argument) for argument in arguments),
    )


def _run_local_tool(
    authority: ArchiveToolAuthority,
    name: str,
    command: Sequence[str],
    *,
    allow_missing_remote: bool = False,
    timeout_seconds: float = _GENERIC_PROCESS_TIMEOUT_SECONDS,
    stdout_limit: int = _MAX_CAPTURE_STDOUT_BYTES,
    stderr_limit: int = _MAX_CAPTURE_STDERR_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    pin = authority.local_tool(name)
    if not command or command[0] != pin.path:
        raise OpenEcologyArchiveError(f"{name} command does not use pinned path")
    if name == "rclone":
        verify_pinned_file(
            authority.rclone_config,
            executable=False,
            require_nonempty=True,
        )
    completed = _run_pinned(
        command,
        pin=pin,
        allow_missing_remote=allow_missing_remote,
        timeout_seconds=timeout_seconds,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )
    if name == "rclone":
        verify_pinned_file(
            authority.rclone_config,
            executable=False,
            require_nonempty=True,
        )
    return completed


def _run_pinned(
    command: Sequence[str],
    *,
    pin: FilePin,
    allow_missing_remote: bool = False,
    timeout_seconds: float = _GENERIC_PROCESS_TIMEOUT_SECONDS,
    stdout_limit: int = _MAX_CAPTURE_STDOUT_BYTES,
    stderr_limit: int = _MAX_CAPTURE_STDERR_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    verify_pinned_file(pin, executable=True, require_nonempty=True)
    completed = _run(
        command,
        allow_missing_remote=allow_missing_remote,
        timeout_seconds=timeout_seconds,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )
    verify_pinned_file(pin, executable=True, require_nonempty=True)
    return completed


def _run(
    command: Sequence[str],
    *,
    allow_missing_remote: bool = False,
    timeout_seconds: float = _GENERIC_PROCESS_TIMEOUT_SECONDS,
    stdout_limit: int = _MAX_CAPTURE_STDOUT_BYTES,
    stderr_limit: int = _MAX_CAPTURE_STDERR_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    try:
        process = subprocess.Popen(
            tuple(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=minimal_subprocess_env(),
            start_new_session=True,
        )
    except OSError as exc:
        raise OpenEcologyArchiveError(f"cannot execute {command[0]}: {exc}") from exc
    stdout, stderr = _BoundedProcess(
        process,
        command_name=str(command[0]),
        timeout_seconds=timeout_seconds,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    ).complete()
    completed = subprocess.CompletedProcess(
        tuple(command),
        process.returncode,
        stdout,
        stderr,
    )
    if completed.returncode == 0:
        return completed
    if (
        allow_missing_remote
        and completed.returncode == 3
        and completed.stdout.strip() in {b"", b"[", b"[]"}
        and b"not found" in completed.stderr.lower()
    ):
        return completed
    raise OpenEcologyArchiveError(
        f"{command[0]} failed with exit {completed.returncode}: "
        + completed.stderr.decode(errors="replace").strip()
    )


def _read_regular_file(path: Path) -> bytes:
    initial = os.lstat(path)
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
        or initial.st_size <= 0
    ):
        raise OpenEcologyArchiveError("evidence object is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not _same_regular_file(initial, metadata):
            raise OpenEcologyArchiveError("evidence object is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _READ_CHUNK_SIZE):
            chunks.append(chunk)
        payload = b"".join(chunks)
        finished = os.fstat(descriptor)
        if len(payload) != metadata.st_size or not _same_regular_file(
            metadata,
            finished,
        ):
            raise OpenEcologyArchiveError("evidence object changed while reading")
        return payload
    finally:
        os.close(descriptor)


def _sha256_path(path: Path) -> str:
    initial = os.lstat(path)
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
        or initial.st_size <= 0
    ):
        raise OpenEcologyArchiveError("evidence object is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not _same_regular_file(initial, opened):
            raise OpenEcologyArchiveError("evidence object changed while opening")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _READ_CHUNK_SIZE):
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if not _same_regular_file(opened, finished):
            raise OpenEcologyArchiveError("evidence object changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _same_regular_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_nlink == right.st_nlink == 1
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OpenEcologyArchiveError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise OpenEcologyArchiveError(f"non-finite JSON constant: {value}")


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise OpenEcologyArchiveError(f"{field} is not a conservative identifier")
    return value


def _git_sha(value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA_PATTERN.fullmatch(value) is None:
        raise OpenEcologyArchiveError("source git SHA must be a full lowercase commit")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise OpenEcologyArchiveError(f"{field} must be lowercase SHA256")
    return value


def _closed_bundle_counts(
    *,
    entry_count: object,
    file_count: object,
    directory_count: object,
    total_file_bytes: object,
    field: str,
) -> dict[str, int]:
    values = {
        "entry_count": entry_count,
        "file_count": file_count,
        "directory_count": directory_count,
        "total_file_bytes": total_file_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OpenEcologyArchiveError(
                f"{field} {name} must be a non-negative integer"
            )
    if entry_count != file_count + directory_count:
        raise OpenEcologyArchiveError(f"{field} entry counts are inconsistent")
    return {name: int(value) for name, value in values.items()}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _limits_from_args(args: argparse.Namespace) -> CampaignStorageLimits:
    return CampaignStorageLimits(
        max_campaign_bytes=args.max_campaign_bytes,
        min_campaign_free_bytes=args.min_campaign_free_bytes,
        min_remote_free_bytes=args.min_remote_free_bytes,
        max_entries=args.max_entries,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive one sealed open-ecology bundle without pruning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    archive = subparsers.add_parser("archive")
    archive.add_argument("--ssh-target", required=True)
    archive.add_argument("--remote-repository-root", required=True)
    archive.add_argument("--remote-active-campaign-root", required=True)
    archive.add_argument("--remote-closed-bundle-dir", required=True)
    archive.add_argument("--remote-staging-dir", required=True)
    archive.add_argument("--campaign-id", required=True)
    archive.add_argument("--bundle-id", required=True)
    archive.add_argument("--source-git-sha", required=True)
    archive.add_argument("--source-manifest-sha256", required=True)
    archive.add_argument("--expected-marker-sha256", required=True)
    archive.add_argument("--expected-entry-count", type=int, required=True)
    archive.add_argument("--expected-file-count", type=int, required=True)
    archive.add_argument("--expected-directory-count", type=int, required=True)
    archive.add_argument("--expected-total-file-bytes", type=int, required=True)
    archive.add_argument("--receipt-path", type=Path, required=True)
    archive.add_argument("--tool-authority-path", type=Path, required=True)
    archive.add_argument("--tool-authority-sha256", required=True)

    remote = subparsers.add_parser("remote-build")
    remote.add_argument("--repository-root", type=Path, required=True)
    remote.add_argument("--active-campaign-root", type=Path, required=True)
    remote.add_argument("--closed-bundle-dir", type=Path, required=True)
    remote.add_argument("--staging-dir", type=Path, required=True)
    remote.add_argument("--campaign-id", required=True)
    remote.add_argument("--bundle-id", required=True)
    remote.add_argument("--source-git-sha", required=True)
    remote.add_argument("--source-manifest-sha256", required=True)
    remote.add_argument("--expected-marker-sha256", required=True)
    remote.add_argument("--expected-entry-count", type=int, required=True)
    remote.add_argument("--expected-file-count", type=int, required=True)
    remote.add_argument("--expected-directory-count", type=int, required=True)
    remote.add_argument("--expected-total-file-bytes", type=int, required=True)
    remote.add_argument("--remote-authority-json", required=True)

    for command in (archive, remote):
        command.add_argument(
            "--max-campaign-bytes",
            type=int,
            default=CampaignStorageLimits().max_campaign_bytes,
        )
        command.add_argument(
            "--min-campaign-free-bytes",
            type=int,
            default=CampaignStorageLimits().min_campaign_free_bytes,
        )
        command.add_argument(
            "--min-remote-free-bytes",
            type=int,
            default=DEFAULT_REMOTE_FREE_BYTES,
        )
        command.add_argument(
            "--max-entries",
            type=int,
            default=CampaignStorageLimits().max_entries,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        limits = _limits_from_args(args)
        if args.command == "remote-build":
            execution_authority = _parse_remote_build_authority(
                args.remote_authority_json
            )
            result = build_remote_closed_bundle_archive(
                repository_root=args.repository_root,
                active_campaign_root=args.active_campaign_root,
                closed_bundle_dir=args.closed_bundle_dir,
                staging_dir=args.staging_dir,
                campaign_id=args.campaign_id,
                bundle_id=args.bundle_id,
                source_git_sha=args.source_git_sha,
                source_manifest_sha256=args.source_manifest_sha256,
                expected_marker_sha256=args.expected_marker_sha256,
                expected_entry_count=args.expected_entry_count,
                expected_file_count=args.expected_file_count,
                expected_directory_count=args.expected_directory_count,
                expected_total_file_bytes=args.expected_total_file_bytes,
                limits=limits,
                execution_authority=execution_authority,
            )
        else:
            result = execute_remote_archive(
                RemoteArchiveOptions(
                    ssh_target=args.ssh_target,
                    remote_repository_root=args.remote_repository_root,
                    remote_active_campaign_root=args.remote_active_campaign_root,
                    remote_closed_bundle_dir=args.remote_closed_bundle_dir,
                    remote_staging_dir=args.remote_staging_dir,
                    campaign_id=args.campaign_id,
                    bundle_id=args.bundle_id,
                    source_git_sha=args.source_git_sha,
                    source_manifest_sha256=args.source_manifest_sha256,
                    expected_marker_sha256=args.expected_marker_sha256,
                    expected_entry_count=args.expected_entry_count,
                    expected_file_count=args.expected_file_count,
                    expected_directory_count=args.expected_directory_count,
                    expected_total_file_bytes=args.expected_total_file_bytes,
                    receipt_path=args.receipt_path,
                    tool_authority_path=args.tool_authority_path,
                    tool_authority_sha256=args.tool_authority_sha256,
                    limits=limits,
                )
            )
    except (ArchiveAuthorityError, CampaignStorageError, OSError, ValueError) as exc:
        print(f"open-ecology archive failed closed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
