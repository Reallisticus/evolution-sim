"""Bounded, descriptor-stable Git authority for open-ecology source checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import selectors
import stat
import subprocess
import time

from evolution_sim.io.open_ecology_bounded_subprocess import (
    OpenEcologyProcessGroupError,
    terminate_process_group_before_reap,
    wait_for_leader_exit_without_reaping,
)


OPEN_ECOLOGY_GIT_COMMAND_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
OPEN_ECOLOGY_SYSTEM_GIT_PATH = Path("/usr/bin/git")
OPEN_ECOLOGY_GIT_COMMAND_TIMEOUT_SECONDS = 30.0
OPEN_ECOLOGY_GIT_EXECUTABLE_MAX_BYTES = 64 * 1024 * 1024
OPEN_ECOLOGY_GIT_OUTPUT_MAX_BYTES = 1024 * 1024
_READ_ONLY_GIT_QUERIES = frozenset(
    {
        ("rev-parse", "--show-toplevel"),
        ("rev-parse", "HEAD"),
        ("status", "--porcelain", "--untracked-files=all"),
        ("status", "--porcelain=v1", "--untracked-files=all"),
        ("status", "--porcelain=v1", "--untracked-files=normal"),
    }
)


class OpenEcologyGitAuthorityError(RuntimeError):
    """The bounded Git executable or command authority failed closed."""


@dataclass(frozen=True, slots=True)
class PinnedGitExecutable:
    """One exact absolute Git executable identity measured from a descriptor."""

    path: str
    device: int
    inode: int
    mode: int
    link_count: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str

    def receipt(self) -> dict[str, object]:
        return asdict(self)


def discover_pinned_git_executable() -> PinnedGitExecutable:
    """Pin the explicit root-owned system Git; never search PATH."""

    try:
        canonical = OPEN_ECOLOGY_SYSTEM_GIT_PATH.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise OpenEcologyGitAuthorityError(
            "the explicit /usr/bin/git authority cannot be resolved"
        ) from error
    _require_root_owned_nonwritable_path(OPEN_ECOLOGY_SYSTEM_GIT_PATH)
    _require_root_owned_nonwritable_path(canonical)
    return pin_git_executable(canonical)


def pin_git_executable(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> PinnedGitExecutable:
    """Measure one absolute executable through an O_NOFOLLOW descriptor."""

    candidate = Path(path)
    if not candidate.is_absolute():
        raise OpenEcologyGitAuthorityError("Git executable path must be absolute")
    try:
        canonical = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise OpenEcologyGitAuthorityError(
            "Git executable cannot be resolved"
        ) from error
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = os.lstat(canonical)
        descriptor = os.open(canonical, flags)
    except OSError as error:
        raise OpenEcologyGitAuthorityError(
            "Git executable cannot be opened safely"
        ) from error
    try:
        before = os.fstat(descriptor)
        mode = before.st_mode & 0o7777
        if (
            not _same_file(before_path, before)
            or not stat.S_ISREG(before.st_mode)
            or not mode & 0o111
            or mode & 0o022
            or before.st_size <= 0
            or before.st_size > OPEN_ECOLOGY_GIT_EXECUTABLE_MAX_BYTES
        ):
            raise OpenEcologyGitAuthorityError(
                "Git executable violates the regular immutable executable contract"
            )
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed_sha256 = digest.hexdigest()
    if (
        not _same_file(before, after)
        or total != before.st_size
        or (expected_sha256 is not None and observed_sha256 != _sha256(expected_sha256))
    ):
        raise OpenEcologyGitAuthorityError(
            "Git executable identity or SHA256 authority drifted"
        )
    return PinnedGitExecutable(
        path=str(canonical),
        device=before.st_dev,
        inode=before.st_ino,
        mode=mode,
        link_count=before.st_nlink,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
        sha256=observed_sha256,
    )


def run_pinned_git(
    authority: PinnedGitExecutable,
    *,
    repository_root: str | Path,
    arguments: Sequence[str],
) -> str:
    """Run one read-only Git query with bounded output, time, and descendants."""

    root = Path(repository_root)
    if not root.is_absolute():
        raise OpenEcologyGitAuthorityError("Git repository root must be absolute")
    try:
        canonical_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise OpenEcologyGitAuthorityError(
            "Git repository root cannot be resolved"
        ) from error
    if not canonical_root.is_dir():
        raise OpenEcologyGitAuthorityError("Git repository root is not a directory")
    normalized_arguments = tuple(arguments)
    if normalized_arguments not in _READ_ONLY_GIT_QUERIES:
        raise OpenEcologyGitAuthorityError(
            "Git authority permits only the sealed read-only source queries"
        )
    before = pin_git_executable(
        authority.path,
        expected_sha256=authority.sha256,
    )
    if before != authority:
        raise OpenEcologyGitAuthorityError(
            "Git executable identity changed before execution"
        )
    command = (
        authority.path,
        "--no-pager",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-C",
        str(canonical_root),
        *normalized_arguments,
    )
    stdout, stderr, returncode = _run_bounded_command(command)
    after = pin_git_executable(
        authority.path,
        expected_sha256=authority.sha256,
    )
    if after != authority:
        raise OpenEcologyGitAuthorityError(
            "Git executable identity changed during execution"
        )
    if returncode != 0 or stderr:
        raise OpenEcologyGitAuthorityError("bounded Git authority command failed")
    try:
        return stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise OpenEcologyGitAuthorityError(
            "bounded Git authority output is not UTF-8"
        ) from error


def _run_bounded_command(
    command: Sequence[str],
) -> tuple[bytes, bytes, int]:
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": OPEN_ECOLOGY_GIT_COMMAND_PATH,
            },
            start_new_session=True,
            bufsize=0,
        )
    except OSError as error:
        raise OpenEcologyGitAuthorityError(
            "bounded Git authority command failed to start"
        ) from error
    if process.stdout is None or process.stderr is None:
        _terminate_process_group(process)
        raise OpenEcologyGitAuthorityError(
            "bounded Git authority command pipes were not created"
        )
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    streams = {
        process.stdout.fileno(): ("stdout", process.stdout),
        process.stderr.fileno(): ("stderr", process.stderr),
    }
    for descriptor, (name, _stream) in streams.items():
        os.set_blocking(descriptor, False)
        selector.register(descriptor, selectors.EVENT_READ, data=name)
    deadline = time.monotonic() + OPEN_ECOLOGY_GIT_COMMAND_TIMEOUT_SECONDS
    total_bytes = 0
    group_cleanup_attempted = False
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OpenEcologyGitAuthorityError(
                    "bounded Git authority command timed out"
                )
            events = selector.select(timeout=remaining)
            if not events:
                raise OpenEcologyGitAuthorityError(
                    "bounded Git authority command timed out"
                )
            for key, _mask in events:
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                total_bytes += len(chunk)
                if total_bytes > OPEN_ECOLOGY_GIT_OUTPUT_MAX_BYTES:
                    raise OpenEcologyGitAuthorityError(
                        "bounded Git authority output exceeded its byte limit"
                    )
                buffers[str(key.data)].extend(chunk)
        try:
            wait_for_leader_exit_without_reaping(process, deadline=deadline)
        except TimeoutError as error:
            raise OpenEcologyGitAuthorityError(
                "bounded Git authority command timed out"
            ) from error
        group_cleanup_attempted = True
        returncode = _terminate_process_group(
            process,
            leader_exit_observed=True,
        )
    except BaseException:
        if not group_cleanup_attempted:
            _terminate_process_group(process)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return bytes(buffers["stdout"]), bytes(buffers["stderr"]), returncode


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    leader_exit_observed: bool = False,
) -> int:
    """Terminate the original session even after its leader has exited."""

    try:
        return terminate_process_group_before_reap(
            process,
            wait_timeout_seconds=2.0,
            leader_exit_observed=leader_exit_observed,
        )
    except OpenEcologyProcessGroupError as error:
        raise OpenEcologyGitAuthorityError(
            "bounded Git authority process group survived termination"
        ) from error


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _require_root_owned_nonwritable_path(path: Path) -> None:
    """Reject a system executable reached through any mutable path component."""

    current = path
    while True:
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise OpenEcologyGitAuthorityError(
                "system Git authority path cannot be inspected"
            ) from error
        if metadata.st_uid != 0 or metadata.st_mode & 0o022:
            raise OpenEcologyGitAuthorityError(
                "system Git authority path is not root-owned and non-writable"
            )
        if current.parent == current:
            break
        current = current.parent


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OpenEcologyGitAuthorityError(
            "Git executable SHA256 must be lowercase hexadecimal"
        )
    return value


def authority_receipt(authority: PinnedGitExecutable) -> Mapping[str, object]:
    """Return the immutable serializable executable identity."""

    return authority.receipt()
