#!/usr/bin/env python3
"""Seal the exact local/remote tool authority for one archive campaign source."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import stat
import subprocess
import sys
import time


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ROOT = _REPOSITORY_ROOT / "python"
sys.path = [
    entry for entry in sys.path if Path(entry or os.curdir).resolve() != _PYTHON_ROOT
]
sys.path.insert(0, str(_PYTHON_ROOT))

from evolution_sim.io.open_ecology_archive_authority import (  # noqa: E402
    ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION,
    LOCAL_TOOL_NAMES,
    REMOTE_HELPER_PATHS,
    REMOTE_TOOL_NAMES,
    SEALED_SSH_OPTIONS,
    ArchiveAuthorityError,
    FilePin,
    SshConnectionIdentity,
    measure_canonical_file,
    minimal_subprocess_env,
    parse_ssh_connection_identity,
    verify_pinned_file,
)
from evolution_sim.io.open_ecology_bounded_subprocess import (  # noqa: E402
    OpenEcologyProcessGroupError,
    leader_exit_observed_without_reaping,
    terminate_process_group_before_reap,
    wait_for_leader_exit_without_reaping,
)
from evolution_sim.io.source_manifest import source_file_hash_manifest  # noqa: E402


DEFAULT_RCLONE_BASE = "gdrive:evolution-sim-backups/archives/open-ecology"
_SSH_TARGET_PATTERN = re.compile(r"[A-Za-z0-9_.@-]+")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GENERIC_PROCESS_TIMEOUT_SECONDS = 15 * 60.0
_MAX_CAPTURE_STDOUT_BYTES = 16 * 1024 * 1024
_MAX_CAPTURE_STDERR_BYTES = 4 * 1024 * 1024
_PIPE_READ_CHUNK_SIZE = 64 * 1024
_PROCESS_TERM_GRACE_SECONDS = 0.25


class AuthoritySealError(RuntimeError):
    """Authority measurement or exclusive publication failed closed."""


def seal_authority(args: argparse.Namespace) -> dict[str, object]:
    repository = _canonical_directory(args.repository_root, field="repository root")
    if repository != _REPOSITORY_ROOT:
        raise AuthoritySealError("sealer must run from the claimed repository root")
    remote_repository = _absolute_posix(
        args.remote_repository_root,
        field="remote repository root",
    )
    if (
        not isinstance(args.ssh_target, str)
        or _SSH_TARGET_PATTERN.fullmatch(args.ssh_target) is None
    ):
        raise AuthoritySealError("SSH target contains unsupported characters")

    local_paths = {
        "git": args.local_git_path,
        "rclone": args.local_rclone_path,
        "ssh": args.local_ssh_path,
        "zstd": args.local_zstd_path,
    }
    if set(local_paths) != set(LOCAL_TOOL_NAMES):
        raise AuthoritySealError(
            "local tool path set does not exactly match LOCAL_TOOL_NAMES"
        )
    local_tools = {
        name: measure_canonical_file(
            Path(local_paths[name]),
            executable=True,
            require_nonempty=True,
        )
        for name in LOCAL_TOOL_NAMES
    }
    rclone_config = measure_canonical_file(
        args.rclone_config_path,
        executable=False,
        require_nonempty=True,
        private_credential=True,
    )
    git_sha = _local_git_output(
        local_tools["git"],
        repository,
        ("rev-parse", "HEAD"),
    ).strip()
    if _GIT_SHA_PATTERN.fullmatch(git_sha) is None:
        raise AuthoritySealError("local repository HEAD is not a full Git commit")
    if _local_git_output(
        local_tools["git"],
        repository,
        ("status", "--porcelain=v1", "--untracked-files=all"),
    ):
        raise AuthoritySealError("local repository must be clean before authority seal")
    manifest = source_file_hash_manifest(repository)
    source_manifest_sha256 = manifest.get("aggregate_sha256")
    if (
        not isinstance(source_manifest_sha256, str)
        or _SHA256_PATTERN.fullmatch(source_manifest_sha256) is None
    ):
        raise AuthoritySealError("source manifest did not produce a valid SHA256")

    helper_pins = {
        relative_path: measure_canonical_file(
            repository / PurePosixPath(relative_path),
            executable=False,
            require_nonempty=True,
        ).sha256
        for relative_path in REMOTE_HELPER_PATHS
    }
    remote_paths = {
        "env": _absolute_posix(args.remote_env_path, field="remote env path"),
        "git": _absolute_posix(args.remote_git_path, field="remote Git path"),
        "python": _absolute_posix(
            args.remote_python_path,
            field="remote Python path",
        ),
        "sha256sum": _absolute_posix(
            args.remote_sha256sum_path,
            field="remote sha256sum path",
        ),
        "zstd": _absolute_posix(args.remote_zstd_path, field="remote zstd path"),
    }
    remote_paths = _canonicalize_remote_executable_paths(
        ssh_pin=local_tools["ssh"],
        ssh_target=args.ssh_target,
        remote_paths=remote_paths,
    )
    ssh_effective_config_sha256 = _effective_ssh_sha256(
        local_tools["ssh"],
        args.ssh_target,
    )
    ssh_connection = _measure_ssh_connection(
        local_tools["ssh"],
        args.ssh_target,
        remote_paths["env"],
    )
    remote_tools, remote_helpers = _measure_remote(
        ssh_pin=local_tools["ssh"],
        ssh_target=args.ssh_target,
        remote_repository=remote_repository,
        remote_paths=remote_paths,
        expected_helpers=helper_pins,
    )
    _require_remote_source(
        ssh_pin=local_tools["ssh"],
        ssh_target=args.ssh_target,
        env_pin=remote_tools["env"],
        git_pin=remote_tools["git"],
        remote_repository=remote_repository,
        source_git_sha=git_sha,
    )

    payload = {
        "endpoint": {
            "rclone_base": DEFAULT_RCLONE_BASE,
            "rclone_config": rclone_config.receipt_record(),
            "ssh_effective_config_sha256": ssh_effective_config_sha256,
            "ssh_connection": ssh_connection.receipt_record(),
            "ssh_target": args.ssh_target,
        },
        "local_tools": {
            name: local_tools[name].receipt_record() for name in LOCAL_TOOL_NAMES
        },
        "remote_helpers": remote_helpers,
        "remote_tools": {
            name: remote_tools[name].receipt_record() for name in REMOTE_TOOL_NAMES
        },
        "schema_version": ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION,
        "source": {
            "git_sha": git_sha,
            "manifest_sha256": source_manifest_sha256,
            "remote_repository_root": remote_repository,
        },
    }
    authority_bytes = _canonical_json_bytes(payload)

    for pin in local_tools.values():
        verify_pinned_file(
            pin,
            executable=True,
            require_nonempty=True,
        )
    verify_pinned_file(
        rclone_config,
        executable=False,
        require_nonempty=True,
    )
    if (
        _effective_ssh_sha256(local_tools["ssh"], args.ssh_target)
        != ssh_effective_config_sha256
    ):
        raise AuthoritySealError("effective SSH endpoint changed during seal")
    if (
        _measure_ssh_connection(
            local_tools["ssh"],
            args.ssh_target,
            remote_paths["env"],
        )
        != ssh_connection
    ):
        raise AuthoritySealError("authenticated SSH endpoint changed during seal")
    repeated_tools, repeated_helpers = _measure_remote(
        ssh_pin=local_tools["ssh"],
        ssh_target=args.ssh_target,
        remote_repository=remote_repository,
        remote_paths=remote_paths,
        expected_helpers=helper_pins,
    )
    if repeated_tools != remote_tools or repeated_helpers != remote_helpers:
        raise AuthoritySealError("remote tool/helper identity changed during seal")
    _require_remote_source(
        ssh_pin=local_tools["ssh"],
        ssh_target=args.ssh_target,
        env_pin=remote_tools["env"],
        git_pin=remote_tools["git"],
        remote_repository=remote_repository,
        source_git_sha=git_sha,
    )
    output = _write_exclusive(args.output, authority_bytes)
    return {
        "authority_path": str(output),
        "authority_sha256": hashlib.sha256(authority_bytes).hexdigest(),
        "schema_version": ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION,
        "source_git_sha": git_sha,
        "source_manifest_sha256": source_manifest_sha256,
        "status": "archive_tool_authority_sealed",
    }


def _canonicalize_remote_executable_paths(
    *,
    ssh_pin: FilePin,
    ssh_target: str,
    remote_paths: Mapping[str, str],
) -> dict[str, str]:
    bootstrap_python = remote_paths.get("python")
    if bootstrap_python is None:
        raise AuthoritySealError("remote Python path is required")
    resolver = (
        "import os,stat,sys\n"
        "raw=sys.argv[1]\n"
        "resolved=os.path.realpath(raw)\n"
        "metadata=os.stat(resolved,follow_symlinks=False)\n"
        "if (not os.path.isabs(resolved) or not stat.S_ISREG(metadata.st_mode)"
        " or not metadata.st_mode & 0o111): raise SystemExit(3)\n"
        "sys.stdout.write(resolved+'\\n')\n"
    )
    resolved: dict[str, str] = {}
    for name in ("python", *(name for name in REMOTE_TOOL_NAMES if name != "python")):
        raw = remote_paths.get(name)
        if raw is None:
            raise AuthoritySealError(f"remote {name} path is required")
        completed = _run_ssh(
            ssh_pin,
            ssh_target,
            (
                resolved.get("python", bootstrap_python),
                "-I",
                "-S",
                "-c",
                resolver,
                raw,
            ),
        )
        try:
            lines = completed.stdout.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise AuthoritySealError(
                f"remote {name} canonical path is not UTF-8"
            ) from exc
        if len(lines) != 1 or completed.stderr:
            raise AuthoritySealError(
                f"remote {name} canonical path result is not exact"
            )
        resolved[name] = _absolute_posix(
            lines[0],
            field=f"resolved remote {name} path",
        )
    return resolved


def _measure_remote(
    *,
    ssh_pin: FilePin,
    ssh_target: str,
    remote_repository: str,
    remote_paths: Mapping[str, str],
    expected_helpers: Mapping[str, str],
) -> tuple[dict[str, FilePin], dict[str, str]]:
    paths = set(remote_paths.values())
    paths.update(
        str(PurePosixPath(remote_repository) / relative_path)
        for relative_path in REMOTE_HELPER_PATHS
    )
    command = (
        remote_paths["env"],
        "-i",
        "LANG=C",
        "LC_ALL=C",
        "PATH=/usr/bin:/bin",
        remote_paths["sha256sum"],
        "--",
        *sorted(paths),
    )
    completed = _run_ssh(ssh_pin, ssh_target, command)
    observed = _parse_sha256sum(completed.stdout)
    if set(observed) != paths:
        raise AuthoritySealError("remote authority measurement path set mismatch")
    tools = {
        name: FilePin(path=path, sha256=observed[path])
        for name, path in remote_paths.items()
    }
    helpers = {
        relative_path: observed[str(PurePosixPath(remote_repository) / relative_path)]
        for relative_path in REMOTE_HELPER_PATHS
    }
    if helpers != dict(expected_helpers):
        raise AuthoritySealError(
            "remote helper bytes differ from the clean local source checkout"
        )
    return tools, helpers


def _require_remote_source(
    *,
    ssh_pin: FilePin,
    ssh_target: str,
    env_pin: FilePin,
    git_pin: FilePin,
    remote_repository: str,
    source_git_sha: str,
) -> None:
    base = (
        env_pin.path,
        "-i",
        "LANG=C",
        "LC_ALL=C",
        "PATH=/usr/bin:/bin",
        git_pin.path,
        "-C",
        remote_repository,
    )
    head = (
        _run_ssh(
            ssh_pin,
            ssh_target,
            (*base, "rev-parse", "HEAD"),
        )
        .stdout.decode("utf-8")
        .strip()
    )
    branch = (
        _run_ssh(
            ssh_pin,
            ssh_target,
            (*base, "rev-parse", "--abbrev-ref", "HEAD"),
        )
        .stdout.decode("utf-8")
        .strip()
    )
    status = _run_ssh(
        ssh_pin,
        ssh_target,
        (*base, "status", "--porcelain=v1", "--untracked-files=all"),
    ).stdout
    if head != source_git_sha or branch != "HEAD" or status:
        raise AuthoritySealError(
            "remote repository must be clean and detached at the exact source commit"
        )


def _effective_ssh_sha256(ssh_pin: FilePin, ssh_target: str) -> str:
    completed = _run_pinned(
        ssh_pin,
        (
            ssh_pin.path,
            "-G",
            *SEALED_SSH_OPTIONS,
            ssh_target,
        ),
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _measure_ssh_connection(
    ssh_pin: FilePin,
    ssh_target: str,
    remote_env_path: str,
) -> SshConnectionIdentity:
    completed = _run_pinned(
        ssh_pin,
        (
            ssh_pin.path,
            "-v",
            *SEALED_SSH_OPTIONS,
            ssh_target,
            " ".join(
                shlex.quote(argument) for argument in (remote_env_path, "--version")
            ),
        ),
    )
    return parse_ssh_connection_identity(completed.stderr)


def _local_git_output(
    git_pin: FilePin,
    repository: Path,
    arguments: Sequence[str],
) -> str:
    completed = _run_pinned(
        git_pin,
        (git_pin.path, "-C", str(repository), *arguments),
    )
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuthoritySealError("local Git output is not UTF-8") from exc


def _run_ssh(
    ssh_pin: FilePin,
    ssh_target: str,
    remote_arguments: Sequence[str],
) -> subprocess.CompletedProcess[bytes]:
    return _run_pinned(
        ssh_pin,
        (
            ssh_pin.path,
            *SEALED_SSH_OPTIONS,
            ssh_target,
            " ".join(shlex.quote(argument) for argument in remote_arguments),
        ),
    )


def _run_pinned(
    pin: FilePin,
    command: Sequence[str],
    *,
    timeout_seconds: float = _GENERIC_PROCESS_TIMEOUT_SECONDS,
    stdout_limit: int = _MAX_CAPTURE_STDOUT_BYTES,
    stderr_limit: int = _MAX_CAPTURE_STDERR_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    verify_pinned_file(pin, executable=True, require_nonempty=True)
    try:
        process = subprocess.Popen(
            tuple(command),
            env=minimal_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise AuthoritySealError(
            f"cannot execute pinned tool {pin.path}: {exc}"
        ) from exc
    completed = _bounded_process_result(
        process,
        command=tuple(command),
        timeout_seconds=timeout_seconds,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )
    verify_pinned_file(pin, executable=True, require_nonempty=True)
    if completed.returncode != 0:
        raise AuthoritySealError(
            f"pinned tool failed with exit {completed.returncode}: "
            + completed.stderr.decode(errors="replace").strip()
        )
    return completed


def _bounded_process_result(
    process: subprocess.Popen[bytes],
    *,
    command: tuple[str, ...],
    timeout_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
) -> subprocess.CompletedProcess[bytes]:
    if timeout_seconds <= 0 or stdout_limit < 0 or stderr_limit < 0:
        _terminate_process_group(process)
        raise AuthoritySealError("subprocess bounds must be non-negative and finite")
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": stdout_limit, "stderr": stderr_limit}
    for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
        if pipe is None:
            _terminate_process_group(process)
            raise AuthoritySealError(f"pinned tool did not expose {name}")
        os.set_blocking(pipe.fileno(), False)
        selector.register(pipe, selectors.EVENT_READ, name)
    deadline = time.monotonic() + timeout_seconds
    failure: AuthoritySealError | None = None
    leader_exit_observed = False
    group_cleanup_attempted = False
    try:
        while selector.get_map():
            if not leader_exit_observed:
                try:
                    leader_exit_observed = leader_exit_observed_without_reaping(process)
                except OpenEcologyProcessGroupError as exc:
                    if process.returncode is None:
                        failure = AuthoritySealError(
                            f"pinned tool {command[0]} leader status was lost"
                        )
                        failure.__cause__ = exc
                        break
                if leader_exit_observed:
                    group_cleanup_attempted = True
                    _terminate_process_group(
                        process,
                        leader_exit_observed=True,
                    )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = AuthoritySealError(
                    f"pinned tool {command[0]} exceeded its total deadline"
                )
                break
            for key, _ in selector.select(min(0.05, remaining)):
                name = str(key.data)
                try:
                    chunk = os.read(key.fd, _PIPE_READ_CHUNK_SIZE)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                if len(buffers[name]) + len(chunk) > limits[name]:
                    failure = AuthoritySealError(
                        f"pinned tool {name} exceeded {limits[name]} bytes"
                    )
                    break
                buffers[name].extend(chunk)
            if failure is not None:
                break
        if failure is not None:
            if not group_cleanup_attempted:
                group_cleanup_attempted = True
                try:
                    _terminate_process_group(
                        process,
                        leader_exit_observed=leader_exit_observed,
                    )
                except AuthoritySealError:
                    pass
        elif process.returncode is None:
            try:
                wait_for_leader_exit_without_reaping(
                    process,
                    deadline=deadline,
                )
            except TimeoutError:
                failure = AuthoritySealError(
                    f"pinned tool {command[0]} exceeded its total deadline"
                )
                group_cleanup_attempted = True
                try:
                    _terminate_process_group(process)
                except AuthoritySealError:
                    pass
            else:
                group_cleanup_attempted = True
                _terminate_process_group(
                    process,
                    leader_exit_observed=True,
                )
    finally:
        for key in tuple(selector.get_map().values()):
            try:
                selector.unregister(key.fileobj)
            except (KeyError, ValueError):
                pass
            key.fileobj.close()
        selector.close()
    if failure is not None:
        raise failure
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        bytes(buffers["stdout"]),
        bytes(buffers["stderr"]),
    )


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
        raise AuthoritySealError(
            "authority subprocess group survived termination"
        ) from exc


def _parse_sha256sum(payload: bytes) -> dict[str, str]:
    try:
        rows = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise AuthoritySealError("remote sha256sum output is not UTF-8") from exc
    result: dict[str, str] = {}
    for row in rows:
        parts = row.split("  ", 1)
        if len(parts) != 2 or _SHA256_PATTERN.fullmatch(parts[0]) is None:
            raise AuthoritySealError("remote sha256sum output is not canonical")
        digest, path = parts
        if path in result:
            raise AuthoritySealError("remote sha256sum output has a duplicate path")
        result[path] = digest
    return result


def _canonical_directory(path: Path, *, field: str) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise AuthoritySealError(f"{field} does not exist") from exc
    if raw != resolved or not resolved.is_dir():
        raise AuthoritySealError(
            f"{field} must be one canonical absolute non-symlink directory"
        )
    return resolved


def _absolute_posix(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise AuthoritySealError(f"{field} must be one safe absolute path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value.rstrip("/"):
        raise AuthoritySealError(f"{field} must be normalized and absolute")
    return str(path)


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    parent = _canonical_directory(raw.parent, field="authority output parent")
    output = parent / raw.name
    if output != raw or output.exists() or output.is_symlink():
        raise AuthoritySealError("authority output must be a new canonical path")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(output, flags, 0o400)
    except OSError as exc:
        raise AuthoritySealError(
            f"cannot exclusively create authority output: {output}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != 0
            or opened.st_uid != os.getuid()
            or opened.st_mode & 0o222
        ):
            raise AuthoritySealError(
                "new authority output descriptor identity is invalid"
            )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise AuthoritySealError("authority output write made no progress")
            offset += written
        os.fsync(descriptor)
        written_metadata = os.fstat(descriptor)
        if (
            not _same_file_identity(opened, written_metadata)
            or written_metadata.st_nlink != 1
            or written_metadata.st_size != len(payload)
        ):
            raise AuthoritySealError(
                "authority output descriptor changed during publication"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = bytearray()
        while len(observed) <= len(payload):
            chunk = os.read(
                descriptor,
                min(1024 * 1024, len(payload) + 1 - len(observed)),
            )
            if not chunk:
                break
            observed.extend(chunk)
        readback_metadata = os.fstat(descriptor)
        if bytes(observed) != payload or not _same_open_file(
            written_metadata, readback_metadata
        ):
            raise AuthoritySealError(
                "authority output bytes changed during exact descriptor readback"
            )
        try:
            path_metadata = os.lstat(output)
        except OSError as exc:
            raise AuthoritySealError(
                "authority output path disappeared during publication"
            ) from exc
        if not _same_open_file(readback_metadata, path_metadata):
            raise AuthoritySealError("authority output path changed during publication")
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    try:
        metadata = os.lstat(output)
    except OSError as exc:
        raise AuthoritySealError(
            "published authority file cannot be inspected"
        ) from exc
    if not _same_open_file(readback_metadata, metadata):
        raise AuthoritySealError("published authority file identity is invalid")
    return output


def _same_open_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_file_identity(left, right)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_nlink == right.st_nlink
        and stat.S_IMODE(left.st_mode) == stat.S_IMODE(right.st_mode)
        and left.st_uid == right.st_uid
        and left.st_gid == right.st_gid
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal exact local/remote archive tool and helper identities",
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--remote-repository-root", required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--rclone-config-path", type=Path, required=True)
    parser.add_argument("--local-ssh-path", required=True)
    parser.add_argument("--local-rclone-path", required=True)
    parser.add_argument("--local-git-path", required=True)
    parser.add_argument("--local-zstd-path", required=True)
    parser.add_argument("--remote-python-path", required=True)
    parser.add_argument("--remote-git-path", required=True)
    parser.add_argument("--remote-zstd-path", required=True)
    parser.add_argument("--remote-sha256sum-path", required=True)
    parser.add_argument("--remote-env-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = seal_authority(args)
    except (ArchiveAuthorityError, AuthoritySealError, OSError, ValueError) as exc:
        print(f"archive authority seal failed closed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
