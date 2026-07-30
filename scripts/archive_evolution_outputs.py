#!/usr/bin/env python3
"""Archive one exact directory and verify its remote copy without deletion.

The implementation intentionally does not support globs, implicit repository
roots, multiple inputs, or automatic pruning.  The caller must name one
directory, and any later cleanup is a separate reviewed operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ROOT = _REPOSITORY_ROOT / "python"
sys.path = [
    entry for entry in sys.path if Path(entry or os.curdir).resolve() != _PYTHON_ROOT
]
sys.path.insert(0, str(_PYTHON_ROOT))

from evolution_sim.io.open_ecology_bounded_subprocess import (  # noqa: E402
    OpenEcologyProcessGroupError,
    leader_exit_observed_without_reaping,
    terminate_process_group_before_reap,
)


DEFAULT_REMOTE = "gdrive:evolution-sim-backups"
DEFAULT_REMOTE_SUBDIR = "archives"
MANIFEST_SCHEMA_VERSION = "evolution_sim_artifact_manifest_v1"
ARCHIVE_COMPRESSION_LEVEL = 10
READ_CHUNK_SIZE = 1024 * 1024
PIPE_READ_CHUNK_SIZE = 64 * 1024
GENERIC_PROCESS_TIMEOUT_SECONDS = 15 * 60.0
ARCHIVE_PROCESS_TIMEOUT_SECONDS = 24 * 60 * 60.0
TRANSFER_PROCESS_TIMEOUT_SECONDS = 24 * 60 * 60.0
MAX_CAPTURE_STDOUT_BYTES = 16 * 1024 * 1024
MAX_CAPTURE_STDERR_BYTES = 4 * 1024 * 1024
PROCESS_POLL_SECONDS = 0.05
PROCESS_TERM_GRACE_SECONDS = 0.25


class ArchiveError(RuntimeError):
    """A fail-closed archive or verification error."""


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class _OwnedFile:
    """One immutable local evidence object captured before any transfer."""

    path: Path
    identity: _FileIdentity
    sha256: str
    size: int
    mode: int
    uid: int
    gid: int
    link_count: int
    mtime_ns: int
    ctime_ns: int


@dataclass
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
                chunk = self.pipe.read(PIPE_READ_CHUNK_SIZE)  # type: ignore[attr-defined]
                if not chunk:
                    break
                next_count = self.byte_count + len(chunk)
                if next_count > self.byte_limit:
                    raise ArchiveError(
                        f"subprocess {self.stream_name} exceeded "
                        f"{self.byte_limit} bytes"
                    )
                self.byte_count = next_count
                if self.consumer is None:
                    assert self.payload is not None
                    self.payload.extend(chunk)
                else:
                    self.consumer(chunk)
        except BaseException as exc:
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
    """Concurrently drain child pipes and enforce one total deadline."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        command_name: str,
        timeout_seconds: float,
        stdout_limit: int = MAX_CAPTURE_STDOUT_BYTES,
        stderr_limit: int = MAX_CAPTURE_STDERR_BYTES,
        stdout_consumer: Callable[[bytes], None] | None = None,
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
        if process.stdout is not None:
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
                    ArchiveError(f"{self.command_name} exceeded its total deadline")
                )
                self._cleanup_process_group()
                return
            self.changed.wait(min(PROCESS_POLL_SECONDS, remaining))
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
                            ArchiveError(f"{self.command_name} leader status was lost")
                        )
                break
            if leader_exit_observed:
                break
            self.changed.wait(PROCESS_POLL_SECONDS)
            self.changed.clear()
        if self.process.returncode is None:
            if self.failure is None and not leader_exit_observed:
                self._set_failure(
                    ArchiveError(f"{self.command_name} exceeded its total deadline")
                )
            self._cleanup_process_group(
                leader_exit_observed=leader_exit_observed,
            )
        for thread in self._threads:
            remaining = max(0.0, self.deadline - time.monotonic())
            thread.join(remaining)
            if thread.is_alive():
                self._set_failure(
                    ArchiveError(
                        f"{self.command_name} pipe drain exceeded its total deadline"
                    )
                )
                self._cleanup_process_group()
        for drain in self.drains:
            if drain.error is not None:
                self._set_failure(drain.error)
        self.finished.set()
        self.changed.set()
        self._watchdog.join(PROCESS_TERM_GRACE_SECONDS)
        if self.failure is not None:
            if isinstance(self.failure, ArchiveError):
                raise self.failure
            raise ArchiveError(
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
        self._set_failure(ArchiveError(f"{self.command_name} aborted"))
        self._cleanup_process_group()
        self.finished.set()
        self.changed.set()
        for thread in self._threads:
            thread.join(PROCESS_TERM_GRACE_SECONDS)
        self._watchdog.join(PROCESS_TERM_GRACE_SECONDS)

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
            except ArchiveError as exc:
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
            wait_timeout_seconds=PROCESS_TERM_GRACE_SECONDS,
            leader_exit_observed=leader_exit_observed,
        )
    except OpenEcologyProcessGroupError as exc:
        raise ArchiveError("archive subprocess group survived termination") from exc


@dataclass(frozen=True)
class SourceEntry:
    path: str
    kind: str
    mode: int
    size: int
    sha256: str | None
    link_target: str | None
    source_device: int
    source_inode: int
    source_mtime_ns: int

    def manifest_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "mode": f"{self.mode:04o}",
            "path": self.path,
            "type": self.kind,
        }
        if self.kind == "file":
            record["sha256"] = self.sha256
            record["size"] = self.size
        elif self.kind == "symlink":
            record["link_target"] = self.link_target
            record["link_target_sha256"] = hashlib.sha256(
                (self.link_target or "").encode("utf-8")
            ).hexdigest()
        return record


@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    root_name: str
    entries: tuple[SourceEntry, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ArchiveOptions:
    input_dir: Path
    output_dir: Path | None
    archive_name: str | None
    remote: str
    remote_subdir: str
    dry_run: bool
    archive_only: bool
    prune_after_verify: bool
    zstd_path: str | None = None
    zstd_sha256: str | None = None
    rclone_path: str | None = None
    rclone_sha256: str | None = None


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(READ_CHUNK_SIZE)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _sha256_path(
    path: Path,
    *,
    expected_identity: _FileIdentity | None = None,
) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArchiveError(f"cannot safely open regular file {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if expected_identity is not None and (
            opened.st_dev != expected_identity.device
            or opened.st_ino != expected_identity.inode
        ):
            raise ArchiveError(f"owned evidence identity changed: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest = _sha256_stream(stream)
        finished = os.fstat(descriptor)
        if (
            finished.st_dev != opened.st_dev
            or finished.st_ino != opened.st_ino
            or finished.st_size != opened.st_size
            or finished.st_mtime_ns != opened.st_mtime_ns
        ):
            raise ArchiveError(f"regular file changed while hashing: {path}")
        return digest
    finally:
        os.close(descriptor)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _same_source(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _hash_regular_file(path: Path, initial: os.stat_result) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArchiveError(f"cannot safely open regular file {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_source(initial, opened):
            raise ArchiveError(f"file changed while opening it: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest = _sha256_stream(stream)
        finished = os.fstat(descriptor)
        if not _same_source(opened, finished):
            raise ArchiveError(f"file changed while hashing it: {path}")
        return digest
    finally:
        os.close(descriptor)


def _scan_directory(root: Path) -> tuple[SourceEntry, ...]:
    entries: list[SourceEntry] = []

    def visit(directory: Path, relative_directory: PurePosixPath) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda child: child.name)
        except OSError as exc:
            raise ArchiveError(f"cannot enumerate {directory}: {exc}") from exc

        for child in children:
            path = Path(child.path)
            relative = relative_directory / child.name
            relative_text = relative.as_posix()
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise ArchiveError(f"cannot inspect {path}: {exc}") from exc
            mode = stat.S_IMODE(metadata.st_mode)

            if stat.S_ISREG(metadata.st_mode):
                digest = _hash_regular_file(path, metadata)
                entries.append(
                    SourceEntry(
                        path=relative_text,
                        kind="file",
                        mode=mode,
                        size=metadata.st_size,
                        sha256=digest,
                        link_target=None,
                        source_device=metadata.st_dev,
                        source_inode=metadata.st_ino,
                        source_mtime_ns=metadata.st_mtime_ns,
                    )
                )
                continue

            if stat.S_ISDIR(metadata.st_mode):
                entries.append(
                    SourceEntry(
                        path=relative_text,
                        kind="directory",
                        mode=mode,
                        size=0,
                        sha256=None,
                        link_target=None,
                        source_device=metadata.st_dev,
                        source_inode=metadata.st_ino,
                        source_mtime_ns=metadata.st_mtime_ns,
                    )
                )
                visit(path, relative)
                continue

            if stat.S_ISLNK(metadata.st_mode):
                try:
                    link_target = os.readlink(path)
                    resolved_target = path.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise ArchiveError(
                        f"symlink is dangling or cannot be resolved: {path}"
                    ) from exc
                if Path(link_target).is_absolute():
                    raise ArchiveError(
                        f"absolute symlink is not portable or restore-safe: "
                        f"{path} -> {link_target}"
                    )
                if not _is_within(resolved_target, root):
                    raise ArchiveError(
                        f"symlink escapes the exact input directory: "
                        f"{path} -> {link_target}"
                    )
                entries.append(
                    SourceEntry(
                        path=relative_text,
                        kind="symlink",
                        mode=mode,
                        size=0,
                        sha256=None,
                        link_target=link_target,
                        source_device=metadata.st_dev,
                        source_inode=metadata.st_ino,
                        source_mtime_ns=metadata.st_mtime_ns,
                    )
                )
                continue

            raise ArchiveError(
                f"unsupported filesystem entry in exact input directory: {path}"
            )

    visit(root, PurePosixPath())
    return tuple(sorted(entries, key=lambda entry: entry.path))


def build_snapshot(input_dir: Path) -> SourceSnapshot:
    if input_dir.is_symlink():
        raise ArchiveError(f"input directory itself may not be a symlink: {input_dir}")
    try:
        root = input_dir.resolve(strict=True)
    except OSError as exc:
        raise ArchiveError(f"input directory does not exist: {input_dir}") from exc
    if not root.is_dir():
        raise ArchiveError(f"input path is not a directory: {root}")
    if root == Path(root.anchor):
        raise ArchiveError("refusing to archive a filesystem root")

    entries = _scan_directory(root)
    file_entries = [entry for entry in entries if entry.kind == "file"]
    directory_entries = [entry for entry in entries if entry.kind == "directory"]
    symlink_entries = [entry for entry in entries if entry.kind == "symlink"]
    manifest: dict[str, Any] = {
        "directory_count": len(directory_entries),
        "entries": [entry.manifest_record() for entry in entries],
        "file_count": len(file_entries),
        "root_name": root.name,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "symlink_count": len(symlink_entries),
        "total_file_bytes": sum(entry.size for entry in file_entries),
    }
    return SourceSnapshot(
        root=root,
        root_name=root.name,
        entries=entries,
        manifest=manifest,
    )


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _assert_entry_unchanged(path: Path, entry: SourceEntry) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ArchiveError(f"source entry disappeared before archival: {path}") from exc
    expected_kind = {
        "file": stat.S_ISREG,
        "directory": stat.S_ISDIR,
        "symlink": stat.S_ISLNK,
    }[entry.kind]
    if not expected_kind(metadata.st_mode):
        raise ArchiveError(f"source entry changed type before archival: {path}")
    if (
        metadata.st_dev != entry.source_device
        or metadata.st_ino != entry.source_inode
        or metadata.st_mtime_ns != entry.source_mtime_ns
        or stat.S_IMODE(metadata.st_mode) != entry.mode
        or (entry.kind == "file" and metadata.st_size != entry.size)
    ):
        raise ArchiveError(f"source entry changed before archival: {path}")
    if entry.kind == "symlink" and os.readlink(path) != entry.link_target:
        raise ArchiveError(f"symlink target changed before archival: {path}")
    return metadata


def _normalized_tar_info(
    archive_path: str,
    entry: SourceEntry | None,
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(archive_path)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if entry is None:
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
    elif entry.kind == "directory":
        info.type = tarfile.DIRTYPE
        info.mode = entry.mode
        info.size = 0
    elif entry.kind == "symlink":
        info.type = tarfile.SYMTYPE
        info.mode = entry.mode
        info.size = 0
        info.linkname = entry.link_target or ""
    else:
        info.type = tarfile.REGTYPE
        info.mode = entry.mode
        info.size = entry.size
    return info


def _add_snapshot_to_tar(
    archive: tarfile.TarFile,
    snapshot: SourceSnapshot,
) -> None:
    archive.addfile(_normalized_tar_info(snapshot.root_name, None))
    for entry in snapshot.entries:
        source_path = snapshot.root / PurePosixPath(entry.path)
        _assert_entry_unchanged(source_path, entry)
        member_name = f"{snapshot.root_name}/{entry.path}"
        info = _normalized_tar_info(member_name, entry)
        if entry.kind != "file":
            archive.addfile(info)
            continue

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(source_path, flags)
        except OSError as exc:
            raise ArchiveError(
                f"cannot safely reopen regular file {source_path}: {exc}"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != entry.source_device
                or opened.st_ino != entry.source_inode
                or opened.st_size != entry.size
                or opened.st_mtime_ns != entry.source_mtime_ns
            ):
                raise ArchiveError(
                    f"source file changed before archival: {source_path}"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                archive.addfile(info, stream)
        finally:
            os.close(descriptor)


def _require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise ArchiveError(f"required program is unavailable: {name}")


def _resolve_program_authority(
    name: str,
    *,
    explicit_path: str | None = None,
    expected_sha256: str | None = None,
) -> tuple[str, str | None]:
    if (explicit_path is None) != (expected_sha256 is None):
        raise ArchiveError(
            f"{name} requires both an absolute path and an external SHA256 pin"
        )
    if explicit_path is None:
        _require_program(name)
        discovered = shutil.which(name)
        return (
            str(Path(discovered).resolve(strict=True))
            if discovered is not None
            else name,
            None,
        )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ArchiveError(f"{name} external SHA256 pin is invalid")
    raw = Path(os.path.abspath(os.path.expanduser(explicit_path)))
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ArchiveError(f"{name} executable does not exist: {raw}") from exc
    if raw != resolved or not resolved.is_absolute():
        raise ArchiveError(
            f"{name} executable must be one canonical absolute non-symlink path"
        )
    _verify_program_pin(resolved, expected_sha256 or "")
    return str(resolved), expected_sha256


def _verify_program_pin(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    try:
        initial = os.lstat(path)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ArchiveError(
            f"cannot safely open pinned executable {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or not opened.st_mode & 0o111
            or not _same_source(initial, opened)
        ):
            raise ArchiveError(f"pinned executable changed while opening: {path}")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, READ_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
        finished = os.fstat(descriptor)
        if size != opened.st_size or not _same_source(opened, finished):
            raise ArchiveError(f"pinned executable changed while hashing: {path}")
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise ArchiveError(
                f"pinned executable SHA256 mismatch for {path}: "
                f"expected={expected_sha256} observed={observed}"
            )
    finally:
        os.close(descriptor)


def _minimal_tool_env() -> dict[str, str]:
    return {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}


def _run_bounded_command(
    command: Sequence[str],
    *,
    timeout_seconds: float = GENERIC_PROCESS_TIMEOUT_SECONDS,
    stdout_limit: int = MAX_CAPTURE_STDOUT_BYTES,
    stderr_limit: int = MAX_CAPTURE_STDERR_BYTES,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        process = subprocess.Popen(
            tuple(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        raise ArchiveError(f"cannot execute {command[0]}: {exc}") from exc
    stdout, stderr = _BoundedProcess(
        process,
        command_name=str(command[0]),
        timeout_seconds=timeout_seconds,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    ).complete()
    return subprocess.CompletedProcess(
        tuple(command),
        process.returncode,
        stdout,
        stderr,
    )


def _verify_zstd_archive(
    path: Path,
    *,
    zstd_path: str = "zstd",
    zstd_sha256: str | None = None,
) -> None:
    _verify_program_pin(Path(zstd_path), zstd_sha256)
    result = _run_bounded_command(
        (zstd_path, "-tq", str(path)),
        timeout_seconds=ARCHIVE_PROCESS_TIMEOUT_SECONDS,
        env=_minimal_tool_env() if zstd_sha256 is not None else None,
    )
    _verify_program_pin(Path(zstd_path), zstd_sha256)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ArchiveError(
            f"local zstd integrity verification failed: "
            f"{detail or f'exit {result.returncode}'}"
        )


def _create_archive(
    snapshot: SourceSnapshot,
    archive_path: Path,
    *,
    zstd_path: str = "zstd",
    zstd_sha256: str | None = None,
) -> _FileIdentity:
    process: subprocess.Popen[bytes] | None = None
    guard: _BoundedProcess | None = None
    archive_identity: _FileIdentity | None = None
    try:
        try:
            descriptor = os.open(
                archive_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o666,
            )
        except FileExistsError as exc:
            raise ArchiveError(
                f"refusing to overwrite archive: {archive_path}"
            ) from exc
        with os.fdopen(descriptor, "wb") as output_stream:
            archive_metadata = os.fstat(output_stream.fileno())
            archive_identity = _FileIdentity(
                device=archive_metadata.st_dev,
                inode=archive_metadata.st_ino,
            )
            _verify_program_pin(Path(zstd_path), zstd_sha256)
            process = subprocess.Popen(
                [
                    zstd_path,
                    "-q",
                    "-T1",
                    f"-{ARCHIVE_COMPRESSION_LEVEL}",
                    "-c",
                ],
                stdin=subprocess.PIPE,
                stdout=output_stream,
                stderr=subprocess.PIPE,
                env=_minimal_tool_env() if zstd_sha256 is not None else None,
                start_new_session=True,
            )
            if process.stdin is None:
                raise ArchiveError("failed to open zstd input stream")
            guard = _BoundedProcess(
                process,
                command_name="zstd compression",
                timeout_seconds=ARCHIVE_PROCESS_TIMEOUT_SECONDS,
                stdout_limit=MAX_CAPTURE_STDOUT_BYTES,
                stderr_limit=MAX_CAPTURE_STDERR_BYTES,
            )
            tar_error: BaseException | None = None
            try:
                with tarfile.open(
                    fileobj=process.stdin,
                    mode="w|",
                    format=tarfile.PAX_FORMAT,
                ) as tar_stream:
                    _add_snapshot_to_tar(tar_stream, snapshot)
            except BaseException as exc:
                tar_error = exc
            finally:
                if not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except BrokenPipeError as exc:
                        if tar_error is None:
                            tar_error = exc
            try:
                _, stderr = guard.complete()
            except ArchiveError:
                if tar_error is not None and not isinstance(
                    tar_error,
                    (BrokenPipeError, OSError),
                ):
                    raise tar_error
                raise
            if tar_error is not None:
                raise tar_error
            return_code = process.returncode
            _verify_program_pin(Path(zstd_path), zstd_sha256)
            if return_code != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise ArchiveError(
                    f"zstd compression failed: {detail or f'exit {return_code}'}"
                )
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if archive_identity is None:
            raise ArchiveError("archive identity was not established")
        _require_identity(archive_path, archive_identity, expected_links=1)
        _fsync_directory(archive_path.parent)
        return archive_identity
    except BaseException:
        if guard is not None and not guard.finished.is_set():
            guard.abort()
        elif process is not None and process.returncode is None:
            _terminate_process_group(process)
        # Never unlink by pathname during failure cleanup.  POSIX has no
        # atomic "unlink only if this is still my inode" operation, so a
        # stat-then-unlink cleanup can delete a racing replacement.  The
        # incomplete exclusive archive remains non-authoritative because no
        # manifest/sidecar or success receipt is emitted.
        raise


def _write_exclusive(path: Path, payload: bytes) -> _FileIdentity:
    descriptor: int | None = None
    identity: _FileIdentity | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o666,
        )
    except FileExistsError as exc:
        raise ArchiveError(f"refusing to overwrite local evidence: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        identity = _FileIdentity(device=opened.st_dev, inode=opened.st_ino)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except BaseException:
        # Leave the exclusive, incomplete object in place.  Removing it by
        # pathname after an error could unlink an attacker-controlled
        # replacement.
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if identity is None:
        raise ArchiveError("exclusive evidence identity was not established")
    _require_identity(path, identity, expected_links=1)
    _fsync_directory(path.parent)
    return identity


def _require_identity(
    path: Path,
    identity: _FileIdentity,
    *,
    expected_links: int | None = None,
) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise ArchiveError(f"owned evidence disappeared: {path}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_dev != identity.device
        or metadata.st_ino != identity.inode
        or (expected_links is not None and metadata.st_nlink != expected_links)
    ):
        raise ArchiveError(f"owned evidence identity changed: {path}")
    return metadata


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ArchiveError(f"cannot open evidence directory for fsync: {path}") from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _owned_metadata_matches(metadata: os.stat_result, owned: _OwnedFile) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_dev == owned.identity.device
        and metadata.st_ino == owned.identity.inode
        and metadata.st_size == owned.size
        and stat.S_IMODE(metadata.st_mode) == owned.mode
        and metadata.st_uid == owned.uid
        and metadata.st_gid == owned.gid
        and metadata.st_nlink == owned.link_count
        and metadata.st_mtime_ns == owned.mtime_ns
        and metadata.st_ctime_ns == owned.ctime_ns
    )


def _owned_from_metadata(
    path: Path,
    metadata: os.stat_result,
    digest: str,
) -> _OwnedFile:
    return _OwnedFile(
        path=path,
        identity=_FileIdentity(device=metadata.st_dev, inode=metadata.st_ino),
        sha256=digest,
        size=metadata.st_size,
        mode=stat.S_IMODE(metadata.st_mode),
        uid=metadata.st_uid,
        gid=metadata.st_gid,
        link_count=metadata.st_nlink,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _require_owned_path(owned: _OwnedFile) -> os.stat_result:
    try:
        metadata = os.lstat(owned.path)
    except FileNotFoundError as exc:
        raise ArchiveError(f"owned evidence disappeared: {owned.path}") from exc
    if not _owned_metadata_matches(metadata, owned):
        raise ArchiveError(f"owned evidence identity changed: {owned.path}")
    return metadata


def _require_owned_descriptor(
    owned: _OwnedFile,
    descriptor: int,
) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if not _owned_metadata_matches(metadata, owned):
        raise ArchiveError(f"open owned evidence changed: {owned.path}")
    return metadata


def _hash_owned_descriptor(owned: _OwnedFile, descriptor: int) -> str:
    _require_owned_descriptor(owned, descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := os.read(descriptor, READ_CHUNK_SIZE):
        digest.update(chunk)
        byte_count += len(chunk)
    _require_owned_descriptor(owned, descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    observed = digest.hexdigest()
    if byte_count != owned.size or observed != owned.sha256:
        raise ArchiveError(
            f"owned evidence digest changed: {owned.path}: "
            f"expected={owned.sha256} observed={observed}"
        )
    return observed


def _open_owned_file(owned: _OwnedFile) -> int:
    try:
        descriptor = os.open(
            owned.path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ArchiveError(
            f"cannot safely open owned evidence {owned.path}: {exc}"
        ) from exc
    try:
        _require_owned_descriptor(owned, descriptor)
        _require_owned_path(owned)
        _hash_owned_descriptor(owned, descriptor)
        _require_owned_path(owned)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _capture_owned_file(
    path: Path,
    identity: _FileIdentity,
    *,
    expected_sha256: str | None = None,
) -> _OwnedFile:
    _require_identity(path, identity, expected_links=1)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ArchiveError(
            f"cannot safely capture owned evidence {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != identity.device
            or opened.st_ino != identity.inode
            or opened.st_nlink != 1
        ):
            raise ArchiveError(f"owned evidence identity changed: {path}")
        provisional = _owned_from_metadata(path, opened, "")
        try:
            current = os.lstat(path)
        except FileNotFoundError as exc:
            raise ArchiveError(f"owned evidence disappeared: {path}") from exc
        if not _owned_metadata_matches(current, provisional):
            raise ArchiveError(f"owned evidence identity changed: {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        byte_count = 0
        while chunk := os.read(descriptor, READ_CHUNK_SIZE):
            digest.update(chunk)
            byte_count += len(chunk)
        finished = os.fstat(descriptor)
        observed = digest.hexdigest()
        owned = _owned_from_metadata(path, opened, observed)
        if (
            byte_count != opened.st_size
            or not _owned_metadata_matches(finished, owned)
            or not _owned_metadata_matches(os.lstat(path), owned)
        ):
            raise ArchiveError(f"owned evidence changed while capturing: {path}")
        if expected_sha256 is not None and observed != expected_sha256:
            raise ArchiveError(
                f"owned evidence SHA256 mismatch for {path}: "
                f"expected={expected_sha256} observed={observed}"
            )
        return owned
    finally:
        os.close(descriptor)


def _revalidate_owned_file(owned: _OwnedFile) -> None:
    descriptor = _open_owned_file(owned)
    try:
        _hash_owned_descriptor(owned, descriptor)
        _require_owned_path(owned)
    finally:
        os.close(descriptor)


def _assert_snapshot_unchanged(snapshot: SourceSnapshot) -> SourceSnapshot:
    current = build_snapshot(snapshot.root)
    if canonical_manifest_bytes(current.manifest) != canonical_manifest_bytes(
        snapshot.manifest
    ):
        raise ArchiveError(
            "input directory changed after manifest creation; refusing to continue"
        )
    return current


def _validate_remote_subdir(value: str) -> str:
    if not value:
        return ""
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveError(f"unsafe remote subdirectory: {value!r}")
    return path.as_posix()


def _remote_directory(remote: str, subdir: str) -> str:
    if ":" not in remote or remote.startswith(":"):
        raise ArchiveError(
            "rclone remote must include a configured remote name and colon"
        )
    base = remote.rstrip("/")
    normalized_subdir = _validate_remote_subdir(subdir)
    return f"{base}/{normalized_subdir}" if normalized_subdir else base


def _run_rclone(
    arguments: Sequence[str],
    *,
    capture_output: bool = False,
    rclone_path: str = "rclone",
    rclone_sha256: str | None = None,
) -> subprocess.CompletedProcess[str]:
    _verify_program_pin(Path(rclone_path), rclone_sha256)
    completed = _run_bounded_command(
        (rclone_path, *arguments),
        timeout_seconds=TRANSFER_PROCESS_TIMEOUT_SECONDS,
    )
    _verify_program_pin(Path(rclone_path), rclone_sha256)
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    result = subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        stdout if capture_output else None,
        stderr,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip() or f"exit {result.returncode}"
        raise ArchiveError(f"rclone command failed: {detail}")
    return result


def _remote_sha256(
    remote_path: str,
    *,
    expected_size: int,
    rclone_path: str = "rclone",
    rclone_sha256: str | None = None,
) -> str:
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise ArchiveError("remote verification size must be positive")
    _verify_program_pin(Path(rclone_path), rclone_sha256)
    try:
        process = subprocess.Popen(
            (rclone_path, "cat", remote_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise ArchiveError(
            f"cannot stream remote object: {remote_path}: {exc}"
        ) from exc
    if process.stdout is None:
        _terminate_process_group(process)
        raise ArchiveError(f"cannot stream remote object: {remote_path}")
    digest = hashlib.sha256()
    guard = _BoundedProcess(
        process,
        command_name="rclone remote byte verification",
        timeout_seconds=TRANSFER_PROCESS_TIMEOUT_SECONDS,
        stdout_limit=expected_size,
        stderr_limit=MAX_CAPTURE_STDERR_BYTES,
        stdout_consumer=digest.update,
    )
    _, stderr = guard.complete()
    byte_count = next(
        drain.byte_count for drain in guard.drains if drain.stream_name == "stdout"
    )
    return_code = process.returncode
    _verify_program_pin(Path(rclone_path), rclone_sha256)
    if return_code != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise ArchiveError(
            f"remote byte verification failed for {remote_path}: "
            f"{detail or f'exit {return_code}'}"
        )
    if byte_count != expected_size:
        raise ArchiveError(
            f"remote byte verification size mismatch for {remote_path}: "
            f"expected={expected_size} observed={byte_count}"
        )
    return digest.hexdigest()


def _upload_owned_file(
    owned: _OwnedFile,
    remote_path: str,
    *,
    rclone_path: str = "rclone",
    rclone_sha256: str | None = None,
) -> None:
    """Upload exactly the already-captured inode through an open descriptor."""

    descriptor = _open_owned_file(owned)
    process: subprocess.Popen[bytes] | None = None
    guard: _BoundedProcess | None = None
    try:
        _hash_owned_descriptor(owned, descriptor)
        _require_owned_path(owned)
        _verify_program_pin(Path(rclone_path), rclone_sha256)
        try:
            process = subprocess.Popen(
                (
                    rclone_path,
                    "rcat",
                    remote_path,
                    "--size",
                    str(owned.size),
                    "--immutable",
                ),
                stdin=descriptor,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise ArchiveError(
                f"cannot upload owned evidence {owned.path}: {exc}"
            ) from exc
        guard = _BoundedProcess(
            process,
            command_name="rclone descriptor-bound upload",
            timeout_seconds=TRANSFER_PROCESS_TIMEOUT_SECONDS,
        )
        # The child inherited this exact open file description.  A pathname
        # replacement cannot alter the bytes being transferred; these checks
        # also fail closed as soon as a replacement is observable.
        _require_owned_descriptor(owned, descriptor)
        _require_owned_path(owned)
        _, stderr = guard.complete()
        _verify_program_pin(Path(rclone_path), rclone_sha256)
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise ArchiveError(
                f"rclone descriptor-bound upload failed: "
                f"{detail or f'exit {process.returncode}'}"
            )
        _require_owned_descriptor(owned, descriptor)
        _hash_owned_descriptor(owned, descriptor)
        _require_owned_path(owned)
    except BaseException:
        if guard is not None and not guard.finished.is_set():
            guard.abort()
        elif process is not None and process.returncode is None:
            _terminate_process_group(process)
        raise
    finally:
        os.close(descriptor)


def _upload_and_verify(
    owned_files: Sequence[_OwnedFile],
    *,
    remote: str,
    remote_subdir: str,
    rclone_path: str = "rclone",
    rclone_sha256: str | None = None,
) -> tuple[str, ...]:
    remote_directory = _remote_directory(remote, remote_subdir)
    listing = _run_rclone(
        ["lsf", remote_directory, "--files-only", "--format", "p"],
        capture_output=True,
        rclone_path=rclone_path,
        rclone_sha256=rclone_sha256,
    )
    existing_names = set((listing.stdout or "").splitlines())
    collisions = sorted(
        owned.path.name for owned in owned_files if owned.path.name in existing_names
    )
    if collisions:
        raise ArchiveError(
            "refusing to overwrite immutable remote evidence: " + ", ".join(collisions)
        )

    remote_paths: list[str] = []
    for owned in owned_files:
        _revalidate_owned_file(owned)
        remote_path = f"{remote_directory}/{owned.path.name}"
        _upload_owned_file(
            owned,
            remote_path,
            rclone_path=rclone_path,
            rclone_sha256=rclone_sha256,
        )
        remote_paths.append(remote_path)

    for owned, remote_path in zip(owned_files, remote_paths, strict=True):
        _revalidate_owned_file(owned)
        remote_digest = _remote_sha256(
            remote_path,
            expected_size=owned.size,
            rclone_path=rclone_path,
            rclone_sha256=rclone_sha256,
        )
        if remote_digest != owned.sha256:
            raise ArchiveError(
                f"remote SHA256 mismatch for {remote_path}: "
                f"original={owned.sha256} remote={remote_digest}"
            )
    for owned in owned_files:
        _revalidate_owned_file(owned)
    return tuple(remote_paths)


def _default_archive_name(root_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", root_name).strip("._-")
    if not safe_name:
        safe_name = "evolution-output"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{safe_name}.tar.zst"


def _validate_archive_name(value: str) -> str:
    if Path(value).name != value or value in {".", ".."}:
        raise ArchiveError(f"archive name must be one filename: {value!r}")
    if not value.endswith(".tar.zst"):
        raise ArchiveError("archive name must end in .tar.zst")
    return value


def _resolve_output_directory(
    snapshot: SourceSnapshot,
    value: Path | None,
    *,
    dry_run: bool,
) -> Path:
    if value is None and not dry_run:
        raise ArchiveError(
            "--output-dir is required outside dry-run mode so archive placement "
            "is always explicit"
        )
    output_dir = value or (snapshot.root.parent / ".evolution-sim-archives")
    resolved = output_dir.expanduser().resolve(strict=False)
    if _is_within(resolved, snapshot.root):
        raise ArchiveError("archive output directory may not be inside the input")
    return resolved


def execute(options: ArchiveOptions) -> dict[str, Any]:
    if options.prune_after_verify:
        raise ArchiveError(
            "--prune-after-verify is disabled: POSIX cannot atomically unlink "
            "only a previously verified inode, so automatic pruning could "
            "delete a racing replacement"
        )

    snapshot = build_snapshot(options.input_dir.expanduser())
    output_dir = _resolve_output_directory(
        snapshot,
        options.output_dir,
        dry_run=options.dry_run,
    )
    archive_name = _validate_archive_name(
        options.archive_name or _default_archive_name(snapshot.root_name)
    )
    archive_path = output_dir / archive_name
    manifest_path = output_dir / f"{archive_name}.manifest.json"
    sidecar_path = output_dir / f"{archive_name}.sha256"

    result: dict[str, Any] = {
        "archive_name": archive_name,
        "archive_only": options.archive_only,
        "dry_run": options.dry_run,
        "file_count": snapshot.manifest["file_count"],
        "input_dir": str(snapshot.root),
        "output_dir": str(output_dir),
        "prune_requested": options.prune_after_verify,
        "remote": _remote_directory(options.remote, options.remote_subdir),
        "symlink_count": snapshot.manifest["symlink_count"],
        "total_file_bytes": snapshot.manifest["total_file_bytes"],
    }
    if options.dry_run:
        result["status"] = "dry_run_validated"
        return result

    zstd_path, zstd_sha256 = _resolve_program_authority(
        "zstd",
        explicit_path=options.zstd_path,
        expected_sha256=options.zstd_sha256,
    )
    rclone_path: str | None = None
    rclone_sha256: str | None = None
    if not options.archive_only:
        rclone_path, rclone_sha256 = _resolve_program_authority(
            "rclone",
            explicit_path=options.rclone_path,
            expected_sha256=options.rclone_sha256,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (archive_path, manifest_path, sidecar_path):
        if path.exists():
            raise ArchiveError(f"refusing to overwrite local evidence: {path}")

    archive_identity = _create_archive(
        snapshot,
        archive_path,
        zstd_path=zstd_path,
        zstd_sha256=zstd_sha256,
    )
    _verify_zstd_archive(
        archive_path,
        zstd_path=zstd_path,
        zstd_sha256=zstd_sha256,
    )
    _assert_snapshot_unchanged(snapshot)
    _require_identity(archive_path, archive_identity, expected_links=1)
    archive_owned = _capture_owned_file(archive_path, archive_identity)

    manifest_bytes = canonical_manifest_bytes(snapshot.manifest)
    archive_digest = archive_owned.sha256
    sidecar_bytes = f"{archive_digest}  {archive_name}\n".encode("utf-8")
    manifest_identity = _write_exclusive(manifest_path, manifest_bytes)
    sidecar_identity = _write_exclusive(sidecar_path, sidecar_bytes)
    manifest_owned = _capture_owned_file(
        manifest_path,
        manifest_identity,
        expected_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    sidecar_owned = _capture_owned_file(
        sidecar_path,
        sidecar_identity,
        expected_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
    )
    owned_files = (archive_owned, manifest_owned, sidecar_owned)
    for owned in owned_files:
        _revalidate_owned_file(owned)

    result.update(
        {
            "archive_path": str(archive_path),
            "archive_sha256": archive_digest,
            "archive_size": archive_owned.size,
            "manifest_path": str(manifest_path),
            "sidecar_path": str(sidecar_path),
            "status": "local_archive_verified",
            "tool_authority": {
                "rclone": (
                    {
                        "path": rclone_path,
                        "sha256": rclone_sha256,
                    }
                    if rclone_path is not None
                    else None
                ),
                "zstd": {
                    "path": zstd_path,
                    "sha256": zstd_sha256,
                },
            },
        }
    )
    if options.archive_only:
        return result

    remote_paths = _upload_and_verify(
        owned_files,
        remote=options.remote,
        remote_subdir=options.remote_subdir,
        rclone_path=rclone_path or "rclone",
        rclone_sha256=rclone_sha256,
    )
    result["remote_paths"] = list(remote_paths)
    result["remote_verified"] = True
    result["status"] = "remote_archive_verified"
    result["pruned"] = False
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Archive one exact directory, upload immutable evidence with rclone, "
            "and verify the remote bytes without deleting local inputs."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="the one exact directory to archive; globs and multiple inputs are unsupported",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "explicit local archive directory; required except for dry-run, "
            "and must be outside the input"
        ),
    )
    parser.add_argument(
        "--archive-name",
        help="immutable archive filename ending in .tar.zst",
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help=f"configured rclone destination root (default: {DEFAULT_REMOTE})",
    )
    parser.add_argument(
        "--remote-subdir",
        default=DEFAULT_REMOTE_SUBDIR,
        help=f"subdirectory below the remote root (default: {DEFAULT_REMOTE_SUBDIR})",
    )
    parser.add_argument(
        "--zstd-path",
        help="canonical absolute zstd executable path; requires --zstd-sha256",
    )
    parser.add_argument(
        "--zstd-sha256",
        help="external whole-file SHA256 pin for --zstd-path",
    )
    parser.add_argument(
        "--rclone-path",
        help="canonical absolute rclone executable path; requires --rclone-sha256",
    )
    parser.add_argument(
        "--rclone-sha256",
        help="external whole-file SHA256 pin for --rclone-path",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="hash and validate the input without writing, uploading, or pruning",
    )
    mode.add_argument(
        "--archive-only",
        action="store_true",
        help="create and verify local evidence without uploading or pruning",
    )
    parser.add_argument(
        "--prune-after-verify",
        action="store_true",
        help=(
            "unsupported safety sentinel; requests fail closed because atomic "
            "identity-conditional unlink is unavailable"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    options = ArchiveOptions(
        input_dir=parsed.input_dir,
        output_dir=parsed.output_dir,
        archive_name=parsed.archive_name,
        remote=parsed.remote,
        remote_subdir=parsed.remote_subdir,
        dry_run=parsed.dry_run,
        archive_only=parsed.archive_only,
        prune_after_verify=parsed.prune_after_verify,
        zstd_path=parsed.zstd_path,
        zstd_sha256=parsed.zstd_sha256,
        rclone_path=parsed.rclone_path,
        rclone_sha256=parsed.rclone_sha256,
    )
    try:
        result = execute(options)
    except ArchiveError as exc:
        print(f"archive failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
