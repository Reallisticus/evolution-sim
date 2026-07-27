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
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
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
    if scan.total_file_bytes > concrete_limits.max_campaign_bytes:
        raise CampaignStorageError(
            "active campaign exceeds its hard byte quota: "
            f"{scan.total_file_bytes} > {concrete_limits.max_campaign_bytes}"
        )
    required_free_bytes = max(
        concrete_limits.min_campaign_free_bytes,
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
    return scan


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

    def visit(directory: Path, relative: PurePosixPath) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise CampaignStorageError(f"cannot enumerate storage tree: {exc}") from exc
        for child in children:
            if len(entries) >= max_entries:
                raise CampaignStorageError("storage tree exceeds its entry ceiling")
            path = Path(child.path)
            child_relative = (relative / child.name).as_posix()
            try:
                metadata = os.lstat(path)
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
                visit(path, relative / child.name)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise CampaignStorageError(
                    f"special files are forbidden in storage trees: {child_relative}"
                )
            if metadata.st_nlink != 1:
                raise CampaignStorageError(
                    f"hard links are forbidden in storage trees: {child_relative}"
                )
            digest, opened = _hash_descriptor_safe(path, metadata)
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

    visit(root, PurePosixPath())
    entries.sort(key=lambda entry: entry.path)
    filesystem_usage = shutil.disk_usage(root)
    return StorageScan(
        root=root,
        entries=tuple(entries),
        total_file_bytes=sum(entry.size for entry in entries if entry.kind == "file"),
        free_bytes=filesystem_usage.free,
        filesystem_total_bytes=filesystem_usage.total,
    )


def _hash_descriptor_safe(
    path: Path,
    initial: os.stat_result,
) -> tuple[str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CampaignStorageError(
            f"cannot safely open storage file {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not _same_source(initial, opened)
        ):
            raise CampaignStorageError(f"storage file changed while opening: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, READ_CHUNK_SIZE):
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if not _same_source(opened, finished):
            raise CampaignStorageError(f"storage file changed while hashing: {path}")
        return digest.hexdigest(), finished
    finally:
        os.close(descriptor)


def _same_source(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
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
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise CampaignStorageError(f"{field} does not exist: {path}") from exc
    if raw != resolved:
        raise CampaignStorageError(
            f"{field} raw and resolved paths differ; symlink ancestors are forbidden"
        )
    if not stat.S_ISDIR(os.lstat(resolved).st_mode):
        raise CampaignStorageError(f"{field} must be a directory")
    current = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise CampaignStorageError(f"{field} has a symlink ancestor: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise CampaignStorageError(
                f"{field} ancestor is not a directory: {current}"
            )
    if resolved == Path(resolved.anchor):
        raise CampaignStorageError(f"{field} may not be a filesystem root")
    return resolved


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
    if path.exists() or path.is_symlink():
        raise CampaignStorageError(f"refusing to overwrite immutable evidence: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise CampaignStorageError(
            f"cannot create immutable evidence {path}: {exc}"
        ) from exc
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(payload):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise CampaignStorageError("short write while creating evidence")
        written += count


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
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
    "load_verified_receipt",
    "seal_closed_bundle",
    "sha256_bytes",
    "validate_closed_bundle",
    "write_verified_receipt",
]
