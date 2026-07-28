"""Fail-closed storage primitives for the sealed open-ecology campaign.

The active campaign directory is only scanned and quota checked.  It is never
an archive input.  An archive input must instead be a separately located,
mechanically sealed bundle whose immutable marker binds every file byte.

This module intentionally contains no deletion, pruning, or retention
eligibility API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import sys
from typing import Any


GIB = 1024**3
ACTIVE_CAMPAIGN_BUDGET_BYTES = 200 * GIB
DEFAULT_REMOTE_FREE_BYTES = 300 * GIB
DEFAULT_ACTIVE_FILESYSTEM_FREE_BYTES = 100 * GIB
ACTIVE_FILESYSTEM_FREE_FRACTION_DENOMINATOR = 5
DEFAULT_MAX_ENTRIES = 2_000_000
MARKER_NAME = ".open-ecology-closed.json"
MARKER_SCHEMA_VERSION = "open_ecology_closed_bundle_v1"
LOCK_SCHEMA_VERSION = "open_ecology_campaign_storage_lock_v1"
RECEIPT_SCHEMA_VERSION = "open_ecology_archive_receipt_envelope_v1"
ARCHIVE_MANIFEST_SCHEMA_VERSION = "evolution_sim_artifact_manifest_v1"
MAX_MARKER_BYTES = 16 * 1024 * 1024
MAX_LOCK_BYTES = 16 * 1024
READ_CHUNK_SIZE = 8 * 1024 * 1024


class CampaignStorageError(RuntimeError):
    """A storage, immutable-bundle, or receipt invariant failed closed."""


@dataclass(frozen=True, slots=True)
class CampaignStorageLimits:
    """Hard resource limits for one campaign on one compute host."""

    max_campaign_bytes: int = ACTIVE_CAMPAIGN_BUDGET_BYTES
    min_campaign_free_bytes: int = DEFAULT_ACTIVE_FILESYSTEM_FREE_BYTES
    min_remote_free_bytes: int = DEFAULT_REMOTE_FREE_BYTES
    max_entries: int = DEFAULT_MAX_ENTRIES

    def validate(self) -> None:
        for field, value in (
            ("max_campaign_bytes", self.max_campaign_bytes),
            ("min_campaign_free_bytes", self.min_campaign_free_bytes),
            ("min_remote_free_bytes", self.min_remote_free_bytes),
            ("max_entries", self.max_entries),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CampaignStorageError(f"{field} must be a positive integer")
        if self.max_campaign_bytes > ACTIVE_CAMPAIGN_BUDGET_BYTES:
            raise CampaignStorageError(
                "max_campaign_bytes may not weaken the 200-GiB campaign ceiling"
            )
        if self.min_campaign_free_bytes < DEFAULT_ACTIVE_FILESYSTEM_FREE_BYTES:
            raise CampaignStorageError(
                "min_campaign_free_bytes may not weaken the 100-GiB floor"
            )
        if self.min_remote_free_bytes < DEFAULT_REMOTE_FREE_BYTES:
            raise CampaignStorageError(
                "min_remote_free_bytes may not weaken the 300-GiB Drive floor"
            )
        if self.max_entries > DEFAULT_MAX_ENTRIES:
            raise CampaignStorageError(
                "max_entries may not weaken the storage-tree entry ceiling"
            )


@dataclass(frozen=True, slots=True)
class StorageEntry:
    """One descriptor-verified filesystem entry."""

    path: str
    kind: str
    mode: int
    size: int
    sha256: str | None
    device: int
    inode: int
    mtime_ns: int
    link_count: int

    def manifest_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "mode": f"{self.mode:04o}",
            "path": self.path,
            "type": self.kind,
        }
        if self.kind == "file":
            record["sha256"] = self.sha256
            record["size"] = self.size
        return record


@dataclass(frozen=True, slots=True)
class StorageScan:
    root: Path
    entries: tuple[StorageEntry, ...]
    total_file_bytes: int
    free_bytes: int
    filesystem_total_bytes: int

    @property
    def file_count(self) -> int:
        return sum(entry.kind == "file" for entry in self.entries)

    @property
    def directory_count(self) -> int:
        return sum(entry.kind == "directory" for entry in self.entries)


@dataclass(frozen=True, slots=True)
class ClosedBundleSnapshot:
    root: Path
    campaign_id: str
    bundle_id: str
    source_git_sha: str
    source_manifest_sha256: str
    marker: Mapping[str, object]
    marker_sha256: str
    scan: StorageScan
    producer_manifest: Mapping[str, object]


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def default_storage_lock_path(campaign_root: Path) -> Path:
    root = _canonical_existing_directory(campaign_root, field="campaign root")
    return root.parent / f".{root.name}.open-ecology-storage.lock"


def ensure_real_directory_tree(
    path: str | Path,
    *,
    field: str,
    mode: int = 0o700,
) -> Path:
    """Create or validate an absolute directory tree without following symlinks.

    ``Path.mkdir(parents=True)`` follows a pre-existing symlink in any ancestor.
    Campaign recovery uses caller-supplied preservation paths, so every path
    component is instead inspected and opened relative to an already verified
    directory descriptor.  A component that races between inspection and open
    is rejected by its device/inode identity.
    """

    if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0 or mode > 0o777:
        raise CampaignStorageError(f"{field} mode is invalid")
    normalized, descriptor = _open_real_directory_tree(
        path,
        field=field,
        create=True,
        mode=mode,
    )
    os.close(descriptor)
    return normalized


class CampaignStorageLock(AbstractContextManager["CampaignStorageLock"]):
    """Nonblocking lock with persistent campaign/source identity."""

    def __init__(self, path: Path, *, campaign_id: str, source_git_sha: str) -> None:
        self.path = Path(path)
        self.identity = {
            "campaign_id": _identifier(campaign_id, field="campaign id"),
            "schema_version": LOCK_SCHEMA_VERSION,
            "source_git_sha": _git_sha(source_git_sha),
        }
        self._descriptor: int | None = None

    def __enter__(self) -> CampaignStorageLock:
        if not self.path.is_absolute():
            raise CampaignStorageError("storage lock path must be absolute")
        _canonical_existing_directory(self.path.parent, field="storage lock parent")
        if self.path.is_symlink():
            raise CampaignStorageError("storage lock may not be a symlink")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise CampaignStorageError(
                f"cannot safely open storage lock: {exc}"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > MAX_LOCK_BYTES
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise CampaignStorageError(
                    "storage lock is not one private small regular file"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CampaignStorageError(
                    "campaign storage is locked by another process"
                ) from exc
            expected = canonical_json_bytes(self.identity)
            os.lseek(descriptor, 0, os.SEEK_SET)
            existing = os.read(descriptor, MAX_LOCK_BYTES + 1)
            if existing and existing != expected:
                raise CampaignStorageError("storage lock identity drifted")
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


def check_campaign_storage(
    campaign_root: Path,
    *,
    campaign_id: str,
    source_git_sha: str,
    limits: CampaignStorageLimits | None = None,
) -> StorageScan:
    """Strictly scan the active root without archiving or changing it."""

    _identifier(campaign_id, field="campaign id")
    _git_sha(source_git_sha)
    concrete_limits = limits or CampaignStorageLimits()
    concrete_limits.validate()
    scan = _scan_tree(
        campaign_root,
        max_entries=concrete_limits.max_entries,
        require_immutable=False,
    )
    _validate_active_storage_budget(scan, concrete_limits)
    return scan


def _validate_active_storage_budget(
    scan: StorageScan,
    limits: CampaignStorageLimits,
) -> None:
    if scan.total_file_bytes > limits.max_campaign_bytes:
        raise CampaignStorageError(
            "active campaign exceeds its hard byte quota: "
            f"{scan.total_file_bytes} > {limits.max_campaign_bytes}"
        )
    required_free_bytes = max(
        limits.min_campaign_free_bytes,
        _ceiling_divide(
            scan.filesystem_total_bytes,
            ACTIVE_FILESYSTEM_FREE_FRACTION_DENOMINATOR,
        ),
    )
    if scan.free_bytes < required_free_bytes:
        raise CampaignStorageError(
            "active campaign filesystem is below its free-space floor: "
            f"{scan.free_bytes} < {required_free_bytes}"
        )


def seal_closed_bundle(
    campaign_root: Path,
    bundle_dir: Path,
    *,
    campaign_id: str,
    bundle_id: str,
    source_git_sha: str,
    source_manifest_sha256: str,
    limits: CampaignStorageLimits | None = None,
) -> ClosedBundleSnapshot:
    """Mechanically close one bundle while holding the campaign storage lock."""

    concrete_limits = limits or CampaignStorageLimits()
    concrete_limits.validate()
    active_root = _canonical_existing_directory(campaign_root, field="campaign root")
    bundle_root = _canonical_existing_directory(bundle_dir, field="closed bundle")
    _require_distinct_trees(active_root, bundle_root)
    parsed_campaign_id = _identifier(campaign_id, field="campaign id")
    parsed_bundle_id = _identifier(bundle_id, field="bundle id")
    if bundle_root.name != parsed_bundle_id:
        raise CampaignStorageError("closed bundle directory name must equal bundle id")
    parsed_git_sha = _git_sha(source_git_sha)
    parsed_manifest_sha256 = _sha256(
        source_manifest_sha256,
        field="source manifest SHA256",
    )
    marker_path = bundle_root / MARKER_NAME
    if marker_path.exists() or marker_path.is_symlink():
        raise CampaignStorageError("closed bundle marker already exists")

    with CampaignStorageLock(
        default_storage_lock_path(active_root),
        campaign_id=parsed_campaign_id,
        source_git_sha=parsed_git_sha,
    ):
        active_scan = check_campaign_storage(
            active_root,
            campaign_id=parsed_campaign_id,
            source_git_sha=parsed_git_sha,
            limits=concrete_limits,
        )
        initial = _scan_tree(
            bundle_root,
            max_entries=concrete_limits.max_entries,
            require_immutable=False,
        )
        if any(entry.path == MARKER_NAME for entry in initial.entries):
            raise CampaignStorageError("closed bundle marker appeared during sealing")
        if (
            active_scan.total_file_bytes + initial.total_file_bytes
            > concrete_limits.max_campaign_bytes
        ):
            raise CampaignStorageError(
                "active root plus closed bundle exceeds the campaign byte ceiling"
            )
        _remove_write_permissions(bundle_root, initial.entries)
        immutable = _scan_tree(
            bundle_root,
            max_entries=concrete_limits.max_entries,
            require_immutable=True,
        )
        marker = _marker_payload(
            immutable,
            campaign_id=parsed_campaign_id,
            bundle_id=parsed_bundle_id,
            source_git_sha=parsed_git_sha,
            source_manifest_sha256=parsed_manifest_sha256,
        )
        marker_bytes = canonical_json_bytes(marker)
        if len(marker_bytes) > MAX_MARKER_BYTES:
            raise CampaignStorageError(
                "closed bundle marker exceeds its hard byte ceiling"
            )
        if (
            active_scan.total_file_bytes
            + immutable.total_file_bytes
            + len(marker_bytes)
            > concrete_limits.max_campaign_bytes
        ):
            raise CampaignStorageError(
                "active root plus closed bundle marker exceeds the campaign "
                "byte ceiling"
            )
        _atomic_create(marker_path, marker_bytes, mode=0o444)
        os.chmod(bundle_root, stat.S_IMODE(bundle_root.stat().st_mode) & ~0o222)
        _fsync_directory(bundle_root)
        _fsync_directory(bundle_root.parent)

    return validate_closed_bundle(
        active_root,
        bundle_root,
        campaign_id=parsed_campaign_id,
        bundle_id=parsed_bundle_id,
        source_git_sha=parsed_git_sha,
        source_manifest_sha256=parsed_manifest_sha256,
        limits=concrete_limits,
    )


def validate_closed_bundle(
    campaign_root: Path,
    bundle_dir: Path,
    *,
    campaign_id: str,
    bundle_id: str,
    source_git_sha: str,
    source_manifest_sha256: str,
    limits: CampaignStorageLimits | None = None,
) -> ClosedBundleSnapshot:
    """Validate one immutable bundle and its marker without changing it."""

    concrete_limits = limits or CampaignStorageLimits()
    concrete_limits.validate()
    active_root = _canonical_existing_directory(campaign_root, field="campaign root")
    bundle_root = _canonical_existing_directory(bundle_dir, field="closed bundle")
    _require_distinct_trees(active_root, bundle_root)
    parsed_campaign_id = _identifier(campaign_id, field="campaign id")
    parsed_bundle_id = _identifier(bundle_id, field="bundle id")
    parsed_git_sha = _git_sha(source_git_sha)
    parsed_manifest_sha256 = _sha256(
        source_manifest_sha256,
        field="source manifest SHA256",
    )
    if bundle_root.name != parsed_bundle_id:
        raise CampaignStorageError("closed bundle directory name must equal bundle id")
    if stat.S_IMODE(bundle_root.stat().st_mode) & 0o222:
        raise CampaignStorageError("closed bundle root remains writable")

    marker_path = bundle_root / MARKER_NAME
    marker_bytes = _read_small_regular_file(
        marker_path,
        max_bytes=MAX_MARKER_BYTES,
        field="closed bundle marker",
    )
    marker = _strict_json_object(marker_bytes, field="closed bundle marker")
    expected_keys = {
        "bundle_id",
        "campaign_id",
        "directory_count",
        "entries",
        "file_count",
        "schema_version",
        "source_git_sha",
        "source_manifest_sha256",
        "total_file_bytes",
    }
    if set(marker) != expected_keys:
        raise CampaignStorageError("closed bundle marker schema is not exact")
    expected_identity = {
        "bundle_id": parsed_bundle_id,
        "campaign_id": parsed_campaign_id,
        "schema_version": MARKER_SCHEMA_VERSION,
        "source_git_sha": parsed_git_sha,
        "source_manifest_sha256": parsed_manifest_sha256,
    }
    for field, expected in expected_identity.items():
        if marker.get(field) != expected:
            raise CampaignStorageError(f"closed bundle marker {field} mismatch")

    full_scan = _scan_tree(
        bundle_root,
        max_entries=concrete_limits.max_entries,
        require_immutable=True,
    )
    content_entries = tuple(
        entry for entry in full_scan.entries if entry.path != MARKER_NAME
    )
    if len(content_entries) == len(full_scan.entries):
        raise CampaignStorageError("closed bundle marker is missing from scan")
    content_scan = StorageScan(
        root=bundle_root,
        entries=content_entries,
        total_file_bytes=sum(
            entry.size for entry in content_entries if entry.kind == "file"
        ),
        free_bytes=full_scan.free_bytes,
        filesystem_total_bytes=full_scan.filesystem_total_bytes,
    )
    expected_marker = _marker_payload(
        content_scan,
        campaign_id=parsed_campaign_id,
        bundle_id=parsed_bundle_id,
        source_git_sha=parsed_git_sha,
        source_manifest_sha256=parsed_manifest_sha256,
    )
    if marker != expected_marker:
        raise CampaignStorageError("closed bundle marker or source bytes were tampered")
    producer_manifest = _producer_manifest(bundle_root.name, full_scan)
    return ClosedBundleSnapshot(
        root=bundle_root,
        campaign_id=parsed_campaign_id,
        bundle_id=parsed_bundle_id,
        source_git_sha=parsed_git_sha,
        source_manifest_sha256=parsed_manifest_sha256,
        marker=marker,
        marker_sha256=sha256_bytes(marker_bytes),
        scan=full_scan,
        producer_manifest=producer_manifest,
    )


def write_verified_receipt(
    path: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    """Write one small immutable receipt after all caller gates have passed."""

    receipt_path = Path(path)
    if not receipt_path.is_absolute():
        raise CampaignStorageError("receipt path must be absolute")
    _canonical_existing_directory(receipt_path.parent, field="receipt parent")
    payload_bytes = canonical_json_bytes(dict(payload))
    envelope: dict[str, object] = {
        "payload": dict(payload),
        "payload_sha256": sha256_bytes(payload_bytes),
        "schema_version": RECEIPT_SCHEMA_VERSION,
    }
    _atomic_create(receipt_path, canonical_json_bytes(envelope), mode=0o444)
    _fsync_directory(receipt_path.parent)
    return envelope


def load_verified_receipt(path: Path) -> dict[str, object]:
    raw = _read_small_regular_file(
        path,
        max_bytes=MAX_MARKER_BYTES,
        field="archive receipt",
    )
    envelope = _strict_json_object(raw, field="archive receipt")
    if set(envelope) != {"payload", "payload_sha256", "schema_version"}:
        raise CampaignStorageError("archive receipt envelope schema is not exact")
    if envelope.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise CampaignStorageError("archive receipt schema version mismatch")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise CampaignStorageError("archive receipt payload must be an object")
    expected = sha256_bytes(canonical_json_bytes(payload))
    if envelope.get("payload_sha256") != expected:
        raise CampaignStorageError("archive receipt payload digest mismatch")
    return envelope


def _marker_payload(
    scan: StorageScan,
    *,
    campaign_id: str,
    bundle_id: str,
    source_git_sha: str,
    source_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "bundle_id": bundle_id,
        "campaign_id": campaign_id,
        "directory_count": scan.directory_count,
        "entries": [entry.manifest_record() for entry in scan.entries],
        "file_count": scan.file_count,
        "schema_version": MARKER_SCHEMA_VERSION,
        "source_git_sha": source_git_sha,
        "source_manifest_sha256": source_manifest_sha256,
        "total_file_bytes": scan.total_file_bytes,
    }


def _producer_manifest(root_name: str, scan: StorageScan) -> dict[str, object]:
    return {
        "directory_count": scan.directory_count,
        "entries": [entry.manifest_record() for entry in scan.entries],
        "file_count": scan.file_count,
        "root_name": root_name,
        "schema_version": ARCHIVE_MANIFEST_SCHEMA_VERSION,
        "symlink_count": 0,
        "total_file_bytes": scan.total_file_bytes,
    }


def _scan_tree(
    root_path: Path,
    *,
    max_entries: int,
    require_immutable: bool,
) -> StorageScan:
    root = _canonical_existing_directory(root_path, field="storage tree")
    root_metadata = os.lstat(root)
    root_device = root_metadata.st_dev
    entries: list[StorageEntry] = []

    def visit(
        directory_descriptor: int,
        directory_metadata: os.stat_result,
        relative: PurePosixPath,
    ) -> None:
        try:
            children = sorted(os.listdir(directory_descriptor))
        except OSError as exc:
            raise CampaignStorageError(f"cannot enumerate storage tree: {exc}") from exc
        for child_name in children:
            if len(entries) >= max_entries:
                raise CampaignStorageError("storage tree exceeds its entry ceiling")
            child_relative = (relative / child_name).as_posix()
            try:
                metadata = _stat_at_without_following(
                    directory_descriptor,
                    child_name,
                )
            except OSError as exc:
                raise CampaignStorageError(
                    f"cannot inspect storage entry {child_relative}: {exc}"
                ) from exc
            if metadata.st_dev != root_device:
                raise CampaignStorageError(
                    f"storage entry crosses a mount boundary: {child_relative}"
                )
            if stat.S_ISLNK(metadata.st_mode):
                raise CampaignStorageError(
                    f"symlinks are forbidden in storage trees: {child_relative}"
                )
            mode = stat.S_IMODE(metadata.st_mode)
            if require_immutable and mode & 0o222:
                raise CampaignStorageError(
                    f"closed bundle entry remains writable: {child_relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                entries.append(
                    StorageEntry(
                        path=child_relative,
                        kind="directory",
                        mode=mode,
                        size=0,
                        sha256=None,
                        device=metadata.st_dev,
                        inode=metadata.st_ino,
                        mtime_ns=metadata.st_mtime_ns,
                        link_count=metadata.st_nlink,
                    )
                )
                child_descriptor, opened = _open_directory_at(
                    directory_descriptor,
                    child_name,
                    metadata,
                    display_path=child_relative,
                )
                try:
                    visit(
                        child_descriptor,
                        opened,
                        relative / child_name,
                    )
                finally:
                    os.close(child_descriptor)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise CampaignStorageError(
                    f"special files are forbidden in storage trees: {child_relative}"
                )
            if metadata.st_nlink != 1:
                raise CampaignStorageError(
                    f"hard links are forbidden in storage trees: {child_relative}"
                )
            digest, opened = _hash_descriptor_safe_at(
                directory_descriptor,
                child_name,
                metadata,
                display_path=child_relative,
            )
            entries.append(
                StorageEntry(
                    path=child_relative,
                    kind="file",
                    mode=mode,
                    size=opened.st_size,
                    sha256=digest,
                    device=opened.st_dev,
                    inode=opened.st_ino,
                    mtime_ns=opened.st_mtime_ns,
                    link_count=opened.st_nlink,
                )
            )
        finished = os.fstat(directory_descriptor)
        if not _same_source(directory_metadata, finished):
            display = relative.as_posix() or "."
            raise CampaignStorageError(
                f"storage directory changed while scanning: {display}"
            )

    root_descriptor, opened_root = _open_directory_path(root, root_metadata)
    try:
        visit(root_descriptor, opened_root, PurePosixPath())
        filesystem_total_bytes, free_bytes = _descriptor_disk_usage(root_descriptor)
    finally:
        os.close(root_descriptor)
    entries.sort(key=lambda entry: entry.path)
    return StorageScan(
        root=root,
        entries=tuple(entries),
        total_file_bytes=sum(entry.size for entry in entries if entry.kind == "file"),
        free_bytes=free_bytes,
        filesystem_total_bytes=filesystem_total_bytes,
    )


def _open_directory_path(
    path: Path,
    initial: os.stat_result,
) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CampaignStorageError(
            f"cannot safely open storage directory {path}: {exc}"
        ) from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or not _same_source(initial, opened):
        os.close(descriptor)
        raise CampaignStorageError(f"storage directory changed while opening: {path}")
    return descriptor, opened


def _stat_at_without_following(
    parent_descriptor: int,
    name: str,
) -> os.stat_result:
    return os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )


def _open_directory_at(
    parent_descriptor: int,
    name: str,
    initial: os.stat_result,
    *,
    display_path: str,
) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(
            name,
            flags,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise CampaignStorageError(
            f"cannot safely open storage directory {display_path}: {exc}"
        ) from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or not _same_source(initial, opened):
        os.close(descriptor)
        raise CampaignStorageError(
            f"storage directory changed while opening: {display_path}"
        )
    return descriptor, opened


def _hash_descriptor_safe_at(
    parent_descriptor: int,
    name: str,
    initial: os.stat_result,
    *,
    display_path: str,
) -> tuple[str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            name,
            flags,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise CampaignStorageError(
            f"cannot safely open storage file {display_path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not _same_source(initial, opened)
        ):
            raise CampaignStorageError(
                f"storage file changed while opening: {display_path}"
            )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, READ_CHUNK_SIZE):
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if not _same_source(opened, finished):
            raise CampaignStorageError(
                f"storage file changed while hashing: {display_path}"
            )
        return digest.hexdigest(), finished
    finally:
        os.close(descriptor)


def _descriptor_disk_usage(descriptor: int) -> tuple[int, int]:
    usage = os.fstatvfs(descriptor)
    return usage.f_frsize * usage.f_blocks, usage.f_frsize * usage.f_bavail


def _same_source(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
        and left.st_nlink == right.st_nlink
    )


def _ceiling_divide(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _remove_write_permissions(root: Path, entries: Sequence[StorageEntry]) -> None:
    for entry in sorted(entries, key=lambda item: item.path.count("/"), reverse=True):
        path = root / PurePosixPath(entry.path)
        current = stat.S_IMODE(os.lstat(path).st_mode)
        os.chmod(path, current & ~0o222, follow_symlinks=False)


def _canonical_existing_directory(path: Path, *, field: str) -> Path:
    normalized, descriptor = _open_real_directory_tree(
        path,
        field=field,
        create=False,
        mode=0o700,
    )
    os.close(descriptor)
    return normalized


def _normalize_real_tree_path(path: str | Path, *, field: str) -> Path:
    candidate = Path(os.path.expanduser(os.fspath(path)))
    if not candidate.is_absolute():
        raise CampaignStorageError(f"{field} must be absolute")
    lexical = Path(os.path.abspath(candidate))
    if candidate != lexical:
        raise CampaignStorageError(f"{field} must be lexically canonical")
    if lexical == Path(lexical.anchor):
        raise CampaignStorageError(f"{field} may not be a filesystem root")

    # macOS exposes /tmp and /var as root-owned compatibility symlinks.  Only
    # resolve that one privileged component.  Joining the untouched suffix is
    # deliberate: resolving ``lexical`` itself would follow a lower,
    # user-controlled symlink before the descriptor walk can reject it.
    if len(lexical.parts) <= 1:
        return lexical
    top_level = Path(lexical.anchor) / lexical.parts[1]
    try:
        top_level_metadata = os.lstat(top_level)
    except OSError as exc:
        raise CampaignStorageError(f"{field} top-level directory is missing") from exc
    if not stat.S_ISLNK(top_level_metadata.st_mode):
        return lexical
    if top_level_metadata.st_uid != 0:
        raise CampaignStorageError(
            f"{field} has a non-root-owned top-level symlink ancestor: {top_level}"
        )
    try:
        raw_target = Path(os.readlink(top_level))
    except OSError as exc:
        raise CampaignStorageError(
            f"{field} top-level compatibility symlink is broken"
        ) from exc
    if not raw_target.is_absolute():
        raw_target = top_level.parent / raw_target
    compatibility_target = Path(os.path.abspath(raw_target))
    normalized = compatibility_target.joinpath(*lexical.parts[2:])
    if normalized == Path(normalized.anchor):
        raise CampaignStorageError(f"{field} may not resolve to a filesystem root")
    return normalized


def _open_real_directory_tree(
    path: str | Path,
    *,
    field: str,
    create: bool,
    mode: int,
) -> tuple[Path, int]:
    normalized = _normalize_real_tree_path(path, field=field)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(normalized.anchor, flags)
        current = Path(normalized.anchor)
        for component in normalized.parts[1:]:
            current /= component
            try:
                metadata = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not create:
                    raise CampaignStorageError(
                        f"{field} does not exist: {current}"
                    ) from None
                try:
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise CampaignStorageError(
                        f"cannot create {field} component {current}: {exc}"
                    ) from exc
                try:
                    metadata = os.stat(
                        component,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise CampaignStorageError(
                        f"cannot inspect created {field} component {current}: {exc}"
                    ) from exc
            except OSError as exc:
                raise CampaignStorageError(
                    f"cannot inspect {field} component {current}: {exc}"
                ) from exc
            if not stat.S_ISDIR(metadata.st_mode):
                raise CampaignStorageError(
                    f"{field} component is not a real directory: {current}"
                )
            try:
                child_descriptor = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise CampaignStorageError(
                    f"cannot safely open {field} component {current}: {exc}"
                ) from exc
            opened = os.fstat(child_descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
            ):
                os.close(child_descriptor)
                raise CampaignStorageError(
                    f"{field} component changed while opening: {current}"
                )
            os.close(descriptor)
            descriptor = child_descriptor
        result = descriptor
        descriptor = None
        return normalized, result
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and stat.S_IMODE(left.st_mode) == stat.S_IMODE(right.st_mode)
        and left.st_nlink == right.st_nlink
    )


def _read_exact_descriptor(descriptor: int, *, expected_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = expected_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(READ_CHUNK_SIZE, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _open_published_file_at(
    parent_descriptor: int,
    name: str,
    expected_metadata: os.stat_result,
    *,
    payload: bytes,
    display_path: Path,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise CampaignStorageError(
            f"cannot read back immutable evidence {display_path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(
            expected_metadata, opened
        ):
            raise CampaignStorageError(
                f"immutable evidence identity changed during publication: {display_path}"
            )
        observed = _read_exact_descriptor(
            descriptor,
            expected_bytes=len(payload),
        )
        finished = os.fstat(descriptor)
        if observed != payload or not _same_source(opened, finished):
            raise CampaignStorageError(
                f"immutable evidence bytes changed during publication: {display_path}"
            )
        try:
            final_path = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise CampaignStorageError(
                f"immutable evidence disappeared after readback: {display_path}"
            ) from exc
        if not _same_file_identity(finished, final_path):
            raise CampaignStorageError(
                f"immutable evidence path changed after readback: {display_path}"
            )
    finally:
        os.close(descriptor)


def _require_distinct_trees(active_root: Path, bundle_root: Path) -> None:
    if (
        active_root == bundle_root
        or active_root in bundle_root.parents
        or bundle_root in active_root.parents
    ):
        raise CampaignStorageError(
            "closed bundle must be a separate tree from the active campaign root"
        )


def _read_small_regular_file(path: Path, *, max_bytes: int, field: str) -> bytes:
    candidate = Path(path)
    if candidate.is_symlink():
        raise CampaignStorageError(f"{field} may not be a symlink")
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise CampaignStorageError(f"{field} is missing: {candidate}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > max_bytes
    ):
        raise CampaignStorageError(f"{field} is not one bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        if not _same_source(metadata, opened):
            raise CampaignStorageError(f"{field} changed while opening")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_SIZE, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        finished = os.fstat(descriptor)
        if len(payload) != opened.st_size or not _same_source(opened, finished):
            raise CampaignStorageError(f"{field} changed while reading")
        return payload
    finally:
        os.close(descriptor)


def _strict_json_object(payload: bytes, *, field: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CampaignStorageError(f"{field} has duplicate JSON keys")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CampaignStorageError(f"{field} has non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignStorageError(f"{field} is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise CampaignStorageError(f"{field} root must be an object")
    return decoded


def _identifier(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or value[0] in ".-"
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in value
        )
    ):
        raise CampaignStorageError(f"{field} is not a conservative identifier")
    return value


def _git_sha(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignStorageError("source git SHA must be a full lowercase commit")
    return value


def _sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignStorageError(f"{field} must be lowercase SHA256")
    return value


def _atomic_create(path: Path, payload: bytes, *, mode: int) -> None:
    destination = Path(path)
    if not destination.is_absolute():
        raise CampaignStorageError("immutable evidence path must be absolute")
    parent, parent_descriptor = _open_real_directory_tree(
        destination.parent,
        field="immutable evidence parent",
        create=False,
        mode=0o700,
    )
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
                    mode,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                raise CampaignStorageError(
                    f"cannot stage immutable evidence {destination}: {exc}"
                ) from exc
            stage_name = candidate
            break
        else:
            raise CampaignStorageError(
                "cannot allocate an immutable evidence publication stage"
            )

        _write_all(descriptor, payload)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        try:
            named_stage = os.stat(
                stage_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise CampaignStorageError(
                "immutable evidence stage disappeared before publication"
            ) from exc
        stage_bytes = _read_exact_descriptor(
            descriptor,
            expected_bytes=len(payload),
        )
        finished_stage = os.fstat(descriptor)
        if (
            not stat.S_ISREG(staged.st_mode)
            or staged.st_nlink != 1
            or staged.st_size != len(payload)
            or stat.S_IMODE(staged.st_mode) != mode
            or not _same_file_identity(staged, named_stage)
            or not _same_source(staged, finished_stage)
            or stage_bytes != payload
        ):
            raise CampaignStorageError(
                "immutable evidence stage identity or bytes changed"
            )
        try:
            _rename_name_no_replace(
                parent_descriptor,
                stage_name,
                destination.name,
            )
        except FileExistsError as exc:
            raise CampaignStorageError(
                f"refusing to overwrite immutable evidence: {destination}"
            ) from exc
        except OSError as exc:
            raise CampaignStorageError(
                f"cannot publish immutable evidence {destination}: {exc}"
            ) from exc
        try:
            published = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise CampaignStorageError(
                "immutable evidence disappeared during publication"
            ) from exc
        if not _same_file_identity(staged, published):
            raise CampaignStorageError(
                "immutable evidence identity changed during publication"
            )
        _open_published_file_at(
            parent_descriptor,
            destination.name,
            staged,
            payload=payload,
            display_path=destination,
        )
        os.fsync(parent_descriptor)
    finally:
        # A failed stage is intentionally preserved.  POSIX has no
        # identity-conditional unlink operation: deleting its pathname after
        # an adversary swaps that name could delete the adversary's replacement.
        # Successful no-replace rename consumes the stage name atomically.
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _rename_name_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically rename one same-directory entry without replacing a peer."""

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
            "exclusive immutable evidence publication is unsupported",
        )
    if result == 0:
        return
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


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(payload):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise CampaignStorageError("short write while creating evidence")
        written += count


def _fsync_directory(path: Path) -> None:
    _, descriptor = _open_real_directory_tree(
        path,
        field="fsync directory",
        create=False,
        mode=0o700,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ACTIVE_CAMPAIGN_BUDGET_BYTES",
    "ACTIVE_FILESYSTEM_FREE_FRACTION_DENOMINATOR",
    "ARCHIVE_MANIFEST_SCHEMA_VERSION",
    "CampaignStorageError",
    "CampaignStorageLimits",
    "CampaignStorageLock",
    "ClosedBundleSnapshot",
    "DEFAULT_ACTIVE_FILESYSTEM_FREE_BYTES",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_REMOTE_FREE_BYTES",
    "MARKER_NAME",
    "StorageScan",
    "canonical_json_bytes",
    "check_campaign_storage",
    "default_storage_lock_path",
    "ensure_real_directory_tree",
    "load_verified_receipt",
    "seal_closed_bundle",
    "sha256_bytes",
    "validate_closed_bundle",
    "write_verified_receipt",
]
