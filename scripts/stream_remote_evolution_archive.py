#!/usr/bin/env python3
"""Archive a remote simulator output and stream it directly to rclone.

The archive is created on the remote training host by the repository's
``archive_evolution_outputs.py`` tool.  Its archive, manifest, and SHA sidecar
are then streamed over SSH directly into an immutable rclone destination.  No
artifact payload is staged on the coordinating machine.

Successful completion means all three remote source files and all three rclone
objects have matching SHA256 digests.  This tool deliberately does not delete
the remote source or staging files.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import sys
from typing import Any, BinaryIO


DEFAULT_RCLONE_REMOTE = "gdrive:evolution-sim-backups"
DEFAULT_RCLONE_SUBDIR = "archives"
REPORT_SCHEMA_VERSION = "remote_evolution_archive_stream_v1"
_SSH_TARGET_PATTERN = re.compile(r"[A-Za-z0-9_.@-]+")
_ARCHIVE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.zst")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RemoteArchiveError(RuntimeError):
    """A remote archive, transfer, or verification gate failed."""


@dataclass(frozen=True, slots=True)
class RemoteArchiveOptions:
    ssh_target: str
    remote_repository_root: str
    remote_input_dir: str
    remote_staging_dir: str
    archive_name: str
    rclone_remote: str = DEFAULT_RCLONE_REMOTE
    rclone_subdir: str = DEFAULT_RCLONE_SUBDIR

    def validate(self) -> None:
        if (
            not self.ssh_target
            or self.ssh_target.startswith("-")
            or _SSH_TARGET_PATTERN.fullmatch(self.ssh_target) is None
        ):
            raise RemoteArchiveError("ssh target contains unsupported characters")
        repository_root = _absolute_remote_path(
            self.remote_repository_root,
            field="remote repository root",
        )
        input_dir = _absolute_remote_path(
            self.remote_input_dir,
            field="remote input directory",
        )
        staging_dir = _absolute_remote_path(
            self.remote_staging_dir,
            field="remote staging directory",
        )
        if input_dir == PurePosixPath("/"):
            raise RemoteArchiveError("refusing to archive the remote filesystem root")
        if _is_relative_to(staging_dir, input_dir):
            raise RemoteArchiveError(
                "remote staging directory may not be inside the input directory"
            )
        if repository_root == PurePosixPath("/"):
            raise RemoteArchiveError(
                "remote repository root must identify an exact checkout"
            )
        if _ARCHIVE_NAME_PATTERN.fullmatch(self.archive_name) is None:
            raise RemoteArchiveError(
                "archive name must be one conservative filename ending in .tar.zst"
            )
        _rclone_directory(self.rclone_remote, self.rclone_subdir)


def execute(options: RemoteArchiveOptions) -> dict[str, object]:
    """Create, stream, and independently verify one remote archive bundle."""

    options.validate()
    repository_root = PurePosixPath(options.remote_repository_root)
    staging_dir = PurePosixPath(options.remote_staging_dir)
    archive_tool = repository_root / "scripts" / "archive_evolution_outputs.py"
    archive_command = (
        "python3",
        str(archive_tool),
        "--input-dir",
        options.remote_input_dir,
        "--output-dir",
        options.remote_staging_dir,
        "--archive-name",
        options.archive_name,
        "--archive-only",
    )
    archive_result = _run_ssh_json(options.ssh_target, archive_command)
    expected_paths = (
        staging_dir / options.archive_name,
        staging_dir / f"{options.archive_name}.manifest.json",
        staging_dir / f"{options.archive_name}.sha256",
    )
    _validate_archive_result(archive_result, expected_paths)

    source_objects = _remote_file_evidence(options.ssh_target, expected_paths)
    destination_directory = _rclone_directory(
        options.rclone_remote,
        options.rclone_subdir,
    )
    _assert_destination_absent(
        destination_directory,
        tuple(path.name for path in expected_paths),
    )

    destination_paths: list[str] = []
    for source in source_objects:
        source_path = str(source["path"])
        destination_path = f"{destination_directory}/{PurePosixPath(source_path).name}"
        _stream_remote_path(
            ssh_target=options.ssh_target,
            source_path=source_path,
            source_size=int(source["size"]),
            destination_path=destination_path,
        )
        destination_paths.append(destination_path)

    verified_objects: list[dict[str, object]] = []
    for source, destination_path in zip(
        source_objects,
        destination_paths,
        strict=True,
    ):
        destination_sha256 = _rclone_sha256(destination_path)
        source_sha256 = str(source["sha256"])
        if destination_sha256 != source_sha256:
            raise RemoteArchiveError(
                "destination SHA256 mismatch for "
                f"{destination_path}: source={source_sha256} "
                f"destination={destination_sha256}"
            )
        verified_objects.append(
            {
                "destination_path": destination_path,
                "sha256": destination_sha256,
                "size": source["size"],
                "source_path": source["path"],
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "archive_name": options.archive_name,
        "archive_sha256": archive_result["archive_sha256"],
        "archive_size": archive_result["archive_size"],
        "destination_directory": destination_directory,
        "objects": verified_objects,
        "remote_input_pruned": False,
        "remote_staging_pruned": False,
        "source_archive_status": archive_result["status"],
        "status": "remote_archive_streamed_and_byte_verified",
        "transport": "ssh_stdout_to_rclone_stdin_without_local_payload_staging",
    }


def _absolute_remote_path(value: str, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise RemoteArchiveError(f"{field} must be a non-empty safe path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise RemoteArchiveError(f"{field} must be an absolute normalized path")
    return path


def _is_relative_to(path: PurePosixPath, parent: PurePosixPath) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _rclone_directory(remote: str, subdir: str) -> str:
    if (
        not isinstance(remote, str)
        or ":" not in remote
        or remote.startswith(":")
        or "\x00" in remote
        or "\n" in remote
    ):
        raise RemoteArchiveError(
            "rclone remote must include a configured remote name and colon"
        )
    if not isinstance(subdir, str) or not subdir:
        raise RemoteArchiveError("rclone subdirectory must be non-empty")
    path = PurePosixPath(subdir)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise RemoteArchiveError("rclone subdirectory must be a safe relative path")
    return f"{remote.rstrip('/')}/{path.as_posix().strip('/')}"


def _ssh_command(ssh_target: str, remote_argv: Sequence[str]) -> list[str]:
    if not remote_argv:
        raise RemoteArchiveError("remote command may not be empty")
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        ssh_target,
        shlex.join(tuple(remote_argv)),
    ]


def _run(
    command: Sequence[str],
    *,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            tuple(command),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise RemoteArchiveError(
            f"cannot execute required program {command[0]!r}: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RemoteArchiveError(
            f"{command[0]} command failed with exit {completed.returncode}: "
            f"{detail or 'no stderr'}"
        )
    return completed


def _run_ssh_json(
    ssh_target: str,
    remote_argv: Sequence[str],
) -> dict[str, object]:
    completed = _run(_ssh_command(ssh_target, remote_argv))
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteArchiveError(
            "remote archive command did not emit one valid JSON object"
        ) from error
    if not isinstance(payload, dict):
        raise RemoteArchiveError("remote archive result must be a JSON object")
    return payload


def _validate_archive_result(
    result: dict[str, object],
    expected_paths: tuple[PurePosixPath, PurePosixPath, PurePosixPath],
) -> None:
    archive_path, manifest_path, sidecar_path = expected_paths
    expected = {
        "archive_path": str(archive_path),
        "manifest_path": str(manifest_path),
        "sidecar_path": str(sidecar_path),
        "status": "local_archive_verified",
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise RemoteArchiveError(
                f"remote archive result {field} does not match the requested path"
            )
    digest = result.get("archive_sha256")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise RemoteArchiveError("remote archive result has no valid SHA256")
    size = result.get("archive_size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RemoteArchiveError("remote archive result has no positive archive size")


def _remote_file_evidence(
    ssh_target: str,
    paths: Sequence[PurePosixPath],
) -> list[dict[str, object]]:
    program = (
        "import hashlib,json,pathlib,sys\n"
        "rows=[]\n"
        "for raw in sys.argv[1:]:\n"
        " p=pathlib.Path(raw)\n"
        " if not p.is_file() or p.is_symlink(): raise SystemExit(3)\n"
        " h=hashlib.sha256()\n"
        " with p.open('rb') as stream:\n"
        "  while chunk:=stream.read(8*1024*1024): h.update(chunk)\n"
        " rows.append({'path':str(p),'size':p.stat().st_size,'sha256':h.hexdigest()})\n"
        "print(json.dumps(rows,separators=(',',':'),sort_keys=True))\n"
    )
    completed = _run(
        _ssh_command(
            ssh_target,
            ("python3", "-c", program, *(str(path) for path in paths)),
        )
    )
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteArchiveError("remote file evidence is not valid JSON") from error
    if not isinstance(payload, list) or len(payload) != len(paths):
        raise RemoteArchiveError("remote file evidence is incomplete")
    normalized: list[dict[str, object]] = []
    for expected_path, record in zip(paths, payload, strict=True):
        if not isinstance(record, dict) or record.get("path") != str(expected_path):
            raise RemoteArchiveError("remote file evidence path mismatch")
        size = record.get("size")
        sha256 = record.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RemoteArchiveError("remote file evidence has invalid size")
        if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
            raise RemoteArchiveError("remote file evidence has invalid SHA256")
        normalized.append({"path": str(expected_path), "size": size, "sha256": sha256})
    return normalized


def _assert_destination_absent(
    destination_directory: str,
    object_names: Sequence[str],
) -> None:
    completed = _run(
        (
            "rclone",
            "lsf",
            destination_directory,
            "--files-only",
            "--format",
            "p",
        )
    )
    existing = {
        line.strip()
        for line in completed.stdout.decode("utf-8").splitlines()
        if line.strip()
    }
    collisions = sorted(existing.intersection(object_names))
    if collisions:
        raise RemoteArchiveError(
            "refusing to overwrite immutable destination evidence: "
            + ", ".join(collisions)
        )


def _stream_remote_path(
    *,
    ssh_target: str,
    source_path: str,
    source_size: int,
    destination_path: str,
) -> None:
    read_program = (
        "import pathlib,sys\n"
        "with pathlib.Path(sys.argv[1]).open('rb') as stream:\n"
        " while chunk:=stream.read(8*1024*1024): sys.stdout.buffer.write(chunk)\n"
    )
    ssh_process: subprocess.Popen[bytes] | None = None
    rclone_process: subprocess.Popen[bytes] | None = None
    try:
        ssh_process = subprocess.Popen(
            _ssh_command(ssh_target, ("python3", "-c", read_program, source_path)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if ssh_process.stdout is None:
            raise RemoteArchiveError("SSH stream did not expose stdout")
        rclone_process = subprocess.Popen(
            (
                "rclone",
                "rcat",
                destination_path,
                "--size",
                str(source_size),
                "--immutable",
            ),
            stdin=ssh_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ssh_process.stdout.close()
        _, rclone_stderr = rclone_process.communicate()
        _, ssh_stderr = ssh_process.communicate()
    except OSError as error:
        if rclone_process is not None:
            rclone_process.kill()
        if ssh_process is not None:
            ssh_process.kill()
        raise RemoteArchiveError(f"cannot start archive stream: {error}") from error
    if ssh_process.returncode != 0 or rclone_process.returncode != 0:
        raise RemoteArchiveError(
            "archive stream failed: "
            f"ssh_exit={ssh_process.returncode} "
            f"rclone_exit={rclone_process.returncode} "
            f"ssh_stderr={ssh_stderr.decode('utf-8', errors='replace').strip()!r} "
            f"rclone_stderr={rclone_stderr.decode('utf-8', errors='replace').strip()!r}"
        )


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _rclone_sha256(destination_path: str) -> str:
    try:
        process = subprocess.Popen(
            ("rclone", "cat", destination_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise RemoteArchiveError(
            f"cannot start rclone verification: {error}"
        ) from error
    if process.stdout is None:
        process.kill()
        raise RemoteArchiveError("rclone verification did not expose stdout")
    digest = _sha256_stream(process.stdout)
    process.stdout.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code != 0:
        raise RemoteArchiveError(
            "rclone destination verification failed: "
            + stderr.decode("utf-8", errors="replace").strip()
        )
    return digest


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an archive on a remote training host and stream it directly "
            "to immutable rclone storage without staging payload bytes locally."
        )
    )
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--remote-repository-root", required=True)
    parser.add_argument("--remote-input-dir", required=True)
    parser.add_argument("--remote-staging-dir", required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--rclone-remote", default=DEFAULT_RCLONE_REMOTE)
    parser.add_argument("--rclone-subdir", default=DEFAULT_RCLONE_SUBDIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    options = RemoteArchiveOptions(
        ssh_target=parsed.ssh_target,
        remote_repository_root=parsed.remote_repository_root,
        remote_input_dir=parsed.remote_input_dir,
        remote_staging_dir=parsed.remote_staging_dir,
        archive_name=parsed.archive_name,
        rclone_remote=parsed.rclone_remote,
        rclone_subdir=parsed.rclone_subdir,
    )
    try:
        result = execute(options)
    except RemoteArchiveError as error:
        print(f"remote archive failed closed: {error}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
