#!/usr/bin/env python3
"""Archive one exact directory and verify its remote copy before pruning.

The implementation intentionally does not support globs, implicit repository
roots, or multiple inputs.  The caller must name one directory, and pruning
leaves that directory itself in place.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Sequence
from uuid import uuid4


DEFAULT_REMOTE = "gdrive:evolution-sim-backups"
DEFAULT_REMOTE_SUBDIR = "archives"
MANIFEST_SCHEMA_VERSION = "evolution_sim_artifact_manifest_v1"
ARCHIVE_COMPRESSION_LEVEL = 10
READ_CHUNK_SIZE = 1024 * 1024


class ArchiveError(RuntimeError):
    """A fail-closed archive or verification error."""


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


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(READ_CHUNK_SIZE)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _sha256_path(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArchiveError(f"cannot safely open regular file {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return _sha256_stream(stream)
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
                raise ArchiveError(f"source file changed before archival: {source_path}")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                archive.addfile(info, stream)
        finally:
            os.close(descriptor)


def _require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise ArchiveError(f"required program is unavailable: {name}")


def _verify_zstd_archive(path: Path) -> None:
    result = subprocess.run(
        ["zstd", "-tq", str(path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise ArchiveError(f"local zstd integrity verification failed: {detail}")


def _create_archive(snapshot: SourceSnapshot, archive_path: Path) -> None:
    partial_path = archive_path.with_name(
        f".{archive_path.name}.{uuid4().hex}.partial"
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        with partial_path.open("xb") as output_stream:
            process = subprocess.Popen(
                [
                    "zstd",
                    "-q",
                    "-T1",
                    f"-{ARCHIVE_COMPRESSION_LEVEL}",
                    "-c",
                ],
                stdin=subprocess.PIPE,
                stdout=output_stream,
                stderr=subprocess.PIPE,
            )
            if process.stdin is None:
                raise ArchiveError("failed to open zstd input stream")
            try:
                with tarfile.open(
                    fileobj=process.stdin,
                    mode="w|",
                    format=tarfile.PAX_FORMAT,
                ) as tar_stream:
                    _add_snapshot_to_tar(tar_stream, snapshot)
            finally:
                if not process.stdin.closed:
                    process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            if process.stderr is not None:
                process.stderr.close()
            return_code = process.wait()
            if return_code != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise ArchiveError(
                    f"zstd compression failed: {detail or f'exit {return_code}'}"
                )
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if archive_path.exists():
            raise ArchiveError(f"refusing to overwrite archive: {archive_path}")
        os.replace(partial_path, archive_path)
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if process is not None and process.stderr is not None:
            process.stderr.close()
        partial_path.unlink(missing_ok=True)
        raise


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ArchiveError(f"refusing to overwrite local evidence: {path}") from exc


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
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["rclone", *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip() or f"exit {result.returncode}"
        raise ArchiveError(f"rclone command failed: {detail}")
    return result


def _remote_sha256(remote_path: str) -> str:
    process = subprocess.Popen(
        ["rclone", "cat", remote_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        process.kill()
        process.wait()
        raise ArchiveError(f"cannot stream remote object: {remote_path}")
    digest = _sha256_stream(process.stdout)
    process.stdout.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    if process.stderr is not None:
        process.stderr.close()
    return_code = process.wait()
    if return_code != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise ArchiveError(
            f"remote byte verification failed for {remote_path}: "
            f"{detail or f'exit {return_code}'}"
        )
    return digest


def _upload_and_verify(
    local_paths: Sequence[Path],
    *,
    remote: str,
    remote_subdir: str,
) -> tuple[str, ...]:
    remote_directory = _remote_directory(remote, remote_subdir)
    listing = _run_rclone(
        ["lsf", remote_directory, "--files-only", "--format", "p"],
        capture_output=True,
    )
    existing_names = set((listing.stdout or "").splitlines())
    collisions = sorted(path.name for path in local_paths if path.name in existing_names)
    if collisions:
        raise ArchiveError(
            "refusing to overwrite immutable remote evidence: "
            + ", ".join(collisions)
        )

    remote_paths: list[str] = []
    for local_path in local_paths:
        remote_path = f"{remote_directory}/{local_path.name}"
        _run_rclone(["copyto", str(local_path), remote_path, "--immutable"])
        remote_paths.append(remote_path)

    for local_path, remote_path in zip(local_paths, remote_paths, strict=True):
        local_digest = _sha256_path(local_path)
        remote_digest = _remote_sha256(remote_path)
        if remote_digest != local_digest:
            raise ArchiveError(
                f"remote SHA256 mismatch for {remote_path}: "
                f"local={local_digest} remote={remote_digest}"
            )
    return tuple(remote_paths)


def _prune_snapshot_contents(snapshot: SourceSnapshot) -> None:
    files_and_links = [
        entry for entry in snapshot.entries if entry.kind in {"file", "symlink"}
    ]
    directories = [entry for entry in snapshot.entries if entry.kind == "directory"]

    for entry in files_and_links:
        path = snapshot.root / PurePosixPath(entry.path)
        _assert_entry_unchanged(path, entry)
        path.unlink()

    for entry in sorted(
        directories,
        key=lambda item: (len(PurePosixPath(item.path).parts), item.path),
        reverse=True,
    ):
        path = snapshot.root / PurePosixPath(entry.path)
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ArchiveError(
                f"directory disappeared during exact prune: {path}"
            ) from exc
        # Removing manifested children necessarily changes a directory's mtime.
        # Its type, identity, and permissions must still match the snapshot.
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_dev != entry.source_device
            or metadata.st_ino != entry.source_inode
            or stat.S_IMODE(metadata.st_mode) != entry.mode
        ):
            raise ArchiveError(f"directory changed during exact prune: {path}")
        try:
            path.rmdir()
        except OSError as exc:
            raise ArchiveError(
                f"directory changed during exact prune; stopped at {path}: {exc}"
            ) from exc


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
    if options.prune_after_verify and (options.dry_run or options.archive_only):
        raise ArchiveError(
            "--prune-after-verify requires a real remote upload and verification"
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

    _require_program("zstd")
    if not options.archive_only:
        _require_program("rclone")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (archive_path, manifest_path, sidecar_path):
        if path.exists():
            raise ArchiveError(f"refusing to overwrite local evidence: {path}")

    try:
        _create_archive(snapshot, archive_path)
        _verify_zstd_archive(archive_path)
        _assert_snapshot_unchanged(snapshot)
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

    manifest_bytes = canonical_manifest_bytes(snapshot.manifest)
    archive_digest = _sha256_path(archive_path)
    sidecar_bytes = f"{archive_digest}  {archive_name}\n".encode("utf-8")
    try:
        _write_exclusive(manifest_path, manifest_bytes)
        _write_exclusive(sidecar_path, sidecar_bytes)
    except Exception:
        manifest_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)
        raise

    result.update(
        {
            "archive_path": str(archive_path),
            "archive_sha256": archive_digest,
            "archive_size": archive_path.stat().st_size,
            "manifest_path": str(manifest_path),
            "sidecar_path": str(sidecar_path),
            "status": "local_archive_verified",
        }
    )
    if options.archive_only:
        return result

    remote_paths = _upload_and_verify(
        (archive_path, manifest_path, sidecar_path),
        remote=options.remote,
        remote_subdir=options.remote_subdir,
    )
    result["remote_paths"] = list(remote_paths)
    result["remote_verified"] = True
    result["status"] = "remote_archive_verified"

    if options.prune_after_verify:
        current_snapshot = _assert_snapshot_unchanged(snapshot)
        _prune_snapshot_contents(current_snapshot)
        result["pruned"] = True
        result["status"] = "remote_archive_verified_and_input_pruned"
    else:
        result["pruned"] = False
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Archive one exact directory, upload immutable evidence with rclone, "
            "and optionally prune only after byte-for-byte remote verification."
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
            "remove only the exact manifested input contents after all three "
            "remote objects pass SHA256 verification; leaves the input directory"
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
