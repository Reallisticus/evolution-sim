"""Strict executable and endpoint authority for open-ecology Drive archival.

The authority document is external evidence.  Its whole-file SHA-256 is passed
separately on the command line, so changing the document, a tool binary, the
effective SSH endpoint, the rclone configuration, or an exact source helper
fails closed before any upload is attempted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import time
from typing import Final, Protocol

from evolution_sim.io.open_ecology_bounded_subprocess import (
    OpenEcologyProcessGroupError,
    leader_exit_observed_without_reaping,
    terminate_process_group_before_reap,
    wait_for_leader_exit_without_reaping,
)


ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION: Final = "open_ecology_archive_tool_authority_v1"
LOCAL_TOOL_NAMES: Final = ("git", "rclone", "ssh", "zstd")
REMOTE_TOOL_NAMES: Final = ("env", "git", "python", "sha256sum", "zstd")
REMOTE_HELPER_PATHS: Final = (
    "python/evolution_sim/io/open_ecology_archive_authority.py",
    "python/evolution_sim/io/open_ecology_bounded_subprocess.py",
    "python/evolution_sim/io/open_ecology_campaign_storage.py",
    "python/evolution_sim/io/open_ecology_runtime_venv_authority.py",
    "python/evolution_sim/io/source_manifest.py",
    "python/evolution_sim/mind/open_ecology_phase_a_guardian.py",
    "python/evolution_sim/cli/open_ecology_phase_a_guardian.py",
    "scripts/archive_evolution_outputs.py",
    "scripts/archive_open_ecology_campaign.py",
)
SEALED_SSH_OPTIONS: Final = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ClearAllForwardings=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ControlPersist=no",
    "-o",
    "KbdInteractiveAuthentication=no",
    "-o",
    "PasswordAuthentication=no",
    "-o",
    "PermitLocalCommand=no",
    "-o",
    "ProxyCommand=none",
    "-o",
    "ProxyJump=none",
    "-o",
    "RequestTTY=no",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "UpdateHostKeys=no",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SSH_HOST_KEY_LINE = re.compile(
    r"^debug1: Server host key: "
    r"(?P<algorithm>[A-Za-z0-9@._+-]+) "
    r"(?P<fingerprint>SHA256:[A-Za-z0-9+/=]{40,64})$"
)
_SSH_AUTHENTICATED_LINE = re.compile(
    r"^Authenticated to (?P<host>\S+) "
    r"\(\[(?P<address>[^\]\r\n]+)\]:(?P<port>[0-9]{1,5})\) "
    r'using "(?P<authentication>[^"\r\n]+)"\.$'
)
_READ_CHUNK_SIZE = 1024 * 1024
_EFFECTIVE_SSH_CONFIG_TIMEOUT_SECONDS = 30.0
_EFFECTIVE_SSH_CONFIG_STDOUT_LIMIT_BYTES = 1024 * 1024
_EFFECTIVE_SSH_CONFIG_STDERR_LIMIT_BYTES = 256 * 1024
_EFFECTIVE_SSH_CONFIG_READ_CHUNK_BYTES = 64 * 1024
_EFFECTIVE_SSH_CONFIG_KILL_TIMEOUT_SECONDS = 5.0


class ArchiveAuthorityError(RuntimeError):
    """The external archive authority or one of its pinned files is invalid."""


@dataclass(frozen=True, slots=True)
class FilePin:
    path: str
    sha256: str
    private_credential: bool = False

    def receipt_record(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


class _PinnedCommandRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        pin: FilePin,
    ) -> subprocess.CompletedProcess[bytes]: ...


@dataclass(frozen=True, slots=True)
class SshConnectionIdentity:
    authenticated_host: str
    address: str
    port: int
    authentication: str
    host_key: str

    def receipt_record(self) -> dict[str, str | int]:
        return {
            "address": self.address,
            "authenticated_host": self.authenticated_host,
            "authentication": self.authentication,
            "host_key": self.host_key,
            "port": self.port,
        }


@dataclass(frozen=True, slots=True)
class ArchiveToolAuthority:
    authority_path: str
    authority_sha256: str
    source_git_sha: str
    source_manifest_sha256: str
    remote_repository_root: str
    ssh_target: str
    ssh_effective_config_sha256: str
    ssh_connection: SshConnectionIdentity
    rclone_base: str
    rclone_config: FilePin
    local_tools: tuple[tuple[str, FilePin], ...]
    remote_tools: tuple[tuple[str, FilePin], ...]
    remote_helpers: tuple[tuple[str, str], ...]

    def local_tool(self, name: str) -> FilePin:
        return _named_pin(self.local_tools, name, field="local tool")

    def remote_tool(self, name: str) -> FilePin:
        return _named_pin(self.remote_tools, name, field="remote tool")

    def remote_helper_sha256(self, relative_path: str) -> str:
        for name, digest in self.remote_helpers:
            if name == relative_path:
                return digest
        raise ArchiveAuthorityError(f"missing remote helper pin: {relative_path}")

    def receipt_record(self) -> dict[str, object]:
        return {
            "authority_sha256": self.authority_sha256,
            "endpoint": {
                "rclone_base": self.rclone_base,
                "rclone_config_sha256": self.rclone_config.sha256,
                "ssh_effective_config_sha256": (self.ssh_effective_config_sha256),
                "ssh_connection": self.ssh_connection.receipt_record(),
                "ssh_target": self.ssh_target,
            },
            "local_tools": {
                name: pin.receipt_record() for name, pin in self.local_tools
            },
            "remote_helpers": {name: digest for name, digest in self.remote_helpers},
            "remote_tools": {
                name: pin.receipt_record() for name, pin in self.remote_tools
            },
            "schema_version": ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION,
            "source": {
                "git_sha": self.source_git_sha,
                "manifest_sha256": self.source_manifest_sha256,
                "remote_repository_root": self.remote_repository_root,
            },
        }


def load_archive_tool_authority(
    path: Path,
    *,
    expected_sha256: str,
) -> ArchiveToolAuthority:
    """Read one strict authority file once and validate its external digest."""

    _require_sha256(expected_sha256, field="authority SHA256")
    canonical = _canonical_absolute_path(path, field="authority file")
    payload_bytes = read_pinned_file(
        FilePin(path=str(canonical), sha256=expected_sha256),
        executable=False,
        require_nonempty=True,
    )
    try:
        payload = json.loads(
            payload_bytes,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveAuthorityError("archive authority is not strict JSON") from exc
    root = _exact_mapping(
        payload,
        (
            "endpoint",
            "local_tools",
            "remote_helpers",
            "remote_tools",
            "schema_version",
            "source",
        ),
        field="archive authority",
    )
    if root["schema_version"] != ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION:
        raise ArchiveAuthorityError("archive authority schema version is unsupported")

    source = _exact_mapping(
        root["source"],
        ("git_sha", "manifest_sha256", "remote_repository_root"),
        field="authority source",
    )
    source_git_sha = _require_git_sha(source["git_sha"])
    source_manifest_sha256 = _require_sha256(
        source["manifest_sha256"],
        field="source manifest SHA256",
    )
    remote_repository_root = _require_absolute_posix_text(
        source["remote_repository_root"],
        field="remote repository root",
    )

    endpoint = _exact_mapping(
        root["endpoint"],
        (
            "rclone_base",
            "rclone_config",
            "ssh_effective_config_sha256",
            "ssh_connection",
            "ssh_target",
        ),
        field="authority endpoint",
    )
    ssh_target = _require_text(endpoint["ssh_target"], field="SSH target")
    rclone_base = _require_text(endpoint["rclone_base"], field="rclone base")
    ssh_effective_config_sha256 = _require_sha256(
        endpoint["ssh_effective_config_sha256"],
        field="effective SSH config SHA256",
    )
    rclone_config = _parse_file_pin(
        endpoint["rclone_config"],
        field="rclone config",
        private_credential=True,
    )
    ssh_connection = _parse_ssh_connection_identity(endpoint["ssh_connection"])

    local_tools = _parse_named_pins(
        root["local_tools"],
        names=LOCAL_TOOL_NAMES,
        field="local tools",
    )
    remote_tools = _parse_named_pins(
        root["remote_tools"],
        names=REMOTE_TOOL_NAMES,
        field="remote tools",
    )
    remote_helpers = _parse_remote_helpers(root["remote_helpers"])

    authority = ArchiveToolAuthority(
        authority_path=str(canonical),
        authority_sha256=expected_sha256,
        source_git_sha=source_git_sha,
        source_manifest_sha256=source_manifest_sha256,
        remote_repository_root=remote_repository_root,
        ssh_target=ssh_target,
        ssh_effective_config_sha256=ssh_effective_config_sha256,
        ssh_connection=ssh_connection,
        rclone_base=rclone_base,
        rclone_config=rclone_config,
        local_tools=local_tools,
        remote_tools=remote_tools,
        remote_helpers=remote_helpers,
    )
    verify_local_authority_files(authority)
    return authority


def verify_local_authority_files(authority: ArchiveToolAuthority) -> None:
    """Revalidate every local executable and the explicit rclone config."""

    for _, pin in authority.local_tools:
        verify_pinned_file(pin, executable=True, require_nonempty=True)
    verify_pinned_file(
        authority.rclone_config,
        executable=False,
        require_nonempty=True,
    )


def verify_effective_ssh_config(
    authority: ArchiveToolAuthority,
    *,
    run_pinned: _PinnedCommandRunner | None = None,
) -> None:
    """Reexecute pinned ``ssh -G`` and require the sealed effective config."""

    ssh_pin = authority.local_tool("ssh")
    command = (
        ssh_pin.path,
        "-G",
        *SEALED_SSH_OPTIONS,
        authority.ssh_target,
    )
    if run_pinned is None:
        completed = _run_effective_ssh_config_probe(command, pin=ssh_pin)
    else:
        completed = run_pinned(command, pin=ssh_pin)
    if completed.returncode != 0:
        raise ArchiveAuthorityError(
            f"pinned effective SSH config probe failed with exit {completed.returncode}"
        )
    if not isinstance(completed.stdout, bytes):
        raise ArchiveAuthorityError(
            "effective SSH config probe did not return byte output"
        )
    observed = hashlib.sha256(completed.stdout).hexdigest()
    if observed != authority.ssh_effective_config_sha256:
        raise ArchiveAuthorityError(
            "effective SSH endpoint/config differs from external authority pin"
        )


def _run_effective_ssh_config_probe(
    command: Sequence[str],
    *,
    pin: FilePin,
) -> subprocess.CompletedProcess[bytes]:
    if not command or command[0] != pin.path:
        raise ArchiveAuthorityError(
            "effective SSH config probe does not use its pinned executable"
        )
    verify_pinned_file(pin, executable=True, require_nonempty=True)
    try:
        process = subprocess.Popen(
            tuple(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=minimal_subprocess_env(),
            start_new_session=True,
        )
    except OSError as exc:
        raise ArchiveAuthorityError(
            f"cannot execute pinned effective SSH config probe: {exc}"
        ) from exc
    group_cleanup_attempted = False
    try:
        stdout, stderr = _read_bounded_effective_ssh_config_output(process)
        try:
            wait_for_leader_exit_without_reaping(
                process,
                deadline=(
                    time.monotonic() + _EFFECTIVE_SSH_CONFIG_KILL_TIMEOUT_SECONDS
                ),
            )
        except TimeoutError as exc:
            raise ArchiveAuthorityError(
                "effective SSH config probe leader exceeded its exit ceiling"
            ) from exc
        group_cleanup_attempted = True
        returncode = _terminate_effective_ssh_config_probe(
            process,
            leader_exit_observed=True,
        )
    except BaseException:
        if not group_cleanup_attempted:
            _terminate_effective_ssh_config_probe(process)
        verify_pinned_file(pin, executable=True, require_nonempty=True)
        raise
    finally:
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
    verify_pinned_file(pin, executable=True, require_nonempty=True)
    if returncode != 0:
        raise ArchiveAuthorityError(
            f"pinned effective SSH config probe failed with exit {returncode}"
        )
    return subprocess.CompletedProcess(tuple(command), returncode, stdout, stderr)


def _read_bounded_effective_ssh_config_output(
    process: subprocess.Popen[bytes],
) -> tuple[bytes, bytes]:
    if process.stdout is None or process.stderr is None:
        raise ArchiveAuthorityError(
            "effective SSH config probe did not expose bounded output pipes"
        )
    selector = selectors.DefaultSelector()
    outputs = {
        "stdout": bytearray(),
        "stderr": bytearray(),
    }
    limits = {
        "stdout": _EFFECTIVE_SSH_CONFIG_STDOUT_LIMIT_BYTES,
        "stderr": _EFFECTIVE_SSH_CONFIG_STDERR_LIMIT_BYTES,
    }
    streams = {
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    deadline = time.monotonic() + _EFFECTIVE_SSH_CONFIG_TIMEOUT_SECONDS
    try:
        for name, stream in streams.items():
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, data=name)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ArchiveAuthorityError(
                    "effective SSH config probe exceeded its time ceiling"
                )
            events = selector.select(remaining)
            if not events:
                raise ArchiveAuthorityError(
                    "effective SSH config probe exceeded its time ceiling"
                )
            for key, _mask in events:
                stream = key.fileobj
                name = str(key.data)
                try:
                    chunk = os.read(
                        key.fd,
                        _EFFECTIVE_SSH_CONFIG_READ_CHUNK_BYTES,
                    )
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    streams[name].close()
                    continue
                output = outputs[name]
                if len(output) + len(chunk) > limits[name]:
                    raise ArchiveAuthorityError(
                        f"effective SSH config probe {name} exceeded its byte ceiling"
                    )
                output.extend(chunk)
    finally:
        selector.close()
    return bytes(outputs["stdout"]), bytes(outputs["stderr"])


def _terminate_effective_ssh_config_probe(
    process: subprocess.Popen[bytes],
    *,
    leader_exit_observed: bool | None = None,
) -> int:
    try:
        leader_exited = (
            leader_exit_observed_without_reaping(process)
            if leader_exit_observed is None
            else leader_exit_observed
        )
        return terminate_process_group_before_reap(
            process,
            wait_timeout_seconds=_EFFECTIVE_SSH_CONFIG_KILL_TIMEOUT_SECONDS,
            leader_exit_observed=leader_exited,
        )
    except (OSError, OpenEcologyProcessGroupError) as exc:
        raise ArchiveAuthorityError(
            "effective SSH config probe process group could not be proven closed"
        ) from exc


def read_pinned_file(
    pin: FilePin,
    *,
    executable: bool,
    require_nonempty: bool,
) -> bytes:
    """Descriptor-read one canonical regular file and enforce its exact digest."""

    payload = _verify_and_optionally_read(
        pin,
        executable=executable,
        require_nonempty=require_nonempty,
        capture_bytes=True,
    )
    assert payload is not None
    return payload


def verify_pinned_file(
    pin: FilePin,
    *,
    executable: bool,
    require_nonempty: bool,
) -> None:
    """Hash one pinned file without retaining its potentially large contents."""

    _verify_and_optionally_read(
        pin,
        executable=executable,
        require_nonempty=require_nonempty,
        capture_bytes=False,
    )


def measure_canonical_file(
    path: Path,
    *,
    executable: bool,
    require_nonempty: bool,
    private_credential: bool = False,
) -> FilePin:
    """Measure one canonical file for a new, not-yet-authoritative seal."""

    canonical = _canonical_absolute_path(path, field="authority measurement file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        initial = os.lstat(canonical)
        descriptor = os.open(canonical, flags)
    except OSError as exc:
        raise ArchiveAuthorityError(
            f"cannot safely open authority measurement file {canonical}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not _same_file(initial, opened):
            raise ArchiveAuthorityError(
                f"authority measurement file changed while opening: {canonical}"
            )
        _require_private_credential(
            opened,
            canonical,
            required=private_credential,
        )
        if executable and not opened.st_mode & 0o111:
            raise ArchiveAuthorityError(
                f"authority measurement file is not executable: {canonical}"
            )
        if require_nonempty and opened.st_size <= 0:
            raise ArchiveAuthorityError(
                f"authority measurement file is empty: {canonical}"
            )
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, _READ_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
        finished = os.fstat(descriptor)
        if size != opened.st_size or not _same_file(opened, finished):
            raise ArchiveAuthorityError(
                f"authority measurement file changed while hashing: {canonical}"
            )
        _require_private_credential(
            finished,
            canonical,
            required=private_credential,
        )
        return FilePin(
            path=str(canonical),
            sha256=digest.hexdigest(),
            private_credential=private_credential,
        )
    finally:
        os.close(descriptor)


def _verify_and_optionally_read(
    pin: FilePin,
    *,
    executable: bool,
    require_nonempty: bool,
    capture_bytes: bool,
) -> bytes | None:
    _require_sha256(pin.sha256, field=f"file SHA256 for {pin.path}")
    path = _canonical_absolute_path(Path(pin.path), field="pinned file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        initial = os.lstat(path)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArchiveAuthorityError(
            f"cannot safely open pinned file {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not _same_file(initial, opened):
            raise ArchiveAuthorityError(f"pinned file changed while opening: {path}")
        _require_private_credential(
            opened,
            path,
            required=pin.private_credential,
        )
        if executable and not opened.st_mode & 0o111:
            raise ArchiveAuthorityError(f"pinned executable is not executable: {path}")
        if require_nonempty and opened.st_size <= 0:
            raise ArchiveAuthorityError(f"pinned file is empty: {path}")
        chunks: list[bytes] | None = [] if capture_bytes else None
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, _READ_CHUNK_SIZE):
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
            size += len(chunk)
        finished = os.fstat(descriptor)
        if size != opened.st_size or not _same_file(opened, finished):
            raise ArchiveAuthorityError(f"pinned file changed while hashing: {path}")
        _require_private_credential(
            finished,
            path,
            required=pin.private_credential,
        )
        observed = digest.hexdigest()
        if observed != pin.sha256:
            raise ArchiveAuthorityError(
                f"pinned file SHA256 mismatch for {path}: "
                f"expected={pin.sha256} observed={observed}"
            )
        return b"".join(chunks) if chunks is not None else None
    finally:
        os.close(descriptor)


def minimal_subprocess_env() -> dict[str, str]:
    """Environment for archival tools; credentials/config are explicit arguments."""

    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def parse_ssh_connection_identity(payload: bytes) -> SshConnectionIdentity:
    """Extract the authenticated host-key/address tuple from pinned OpenSSH."""

    try:
        rows = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ArchiveAuthorityError("SSH verbose output is not UTF-8") from exc
    host_key_matches = [
        match
        for row in rows
        if (match := _SSH_HOST_KEY_LINE.fullmatch(row)) is not None
    ]
    authenticated_matches = [
        match
        for row in rows
        if (match := _SSH_AUTHENTICATED_LINE.fullmatch(row)) is not None
    ]
    if len(host_key_matches) != 1 or len(authenticated_matches) != 1:
        raise ArchiveAuthorityError(
            "SSH verbose output lacks one unambiguous authenticated endpoint"
        )
    host_key_match = host_key_matches[0]
    authenticated_match = authenticated_matches[0]
    port = int(authenticated_match.group("port"))
    if not 1 <= port <= 65535:
        raise ArchiveAuthorityError("SSH authenticated endpoint port is invalid")
    authentication = authenticated_match.group("authentication")
    if authentication != "publickey":
        raise ArchiveAuthorityError(
            "SSH archive endpoint must authenticate with publickey"
        )
    return SshConnectionIdentity(
        authenticated_host=authenticated_match.group("host"),
        address=authenticated_match.group("address"),
        port=port,
        authentication=authentication,
        host_key=(
            f"{host_key_match.group('algorithm')} {host_key_match.group('fingerprint')}"
        ),
    )


def _parse_ssh_connection_identity(value: object) -> SshConnectionIdentity:
    mapping = _exact_mapping(
        value,
        (
            "address",
            "authenticated_host",
            "authentication",
            "host_key",
            "port",
        ),
        field="SSH connection identity",
    )
    authenticated_host = _require_text(
        mapping["authenticated_host"],
        field="SSH authenticated host",
    )
    address = _require_text(mapping["address"], field="SSH authenticated address")
    authentication = _require_text(
        mapping["authentication"],
        field="SSH authentication method",
    )
    host_key = _require_text(mapping["host_key"], field="SSH host key")
    if (
        any(character.isspace() for character in authenticated_host)
        or any(character.isspace() for character in address)
        or _SSH_HOST_KEY_LINE.fullmatch(f"debug1: Server host key: {host_key}") is None
    ):
        raise ArchiveAuthorityError("SSH connection identity is not canonical")
    port = mapping["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ArchiveAuthorityError("SSH authenticated port must be 1..65535")
    if authentication != "publickey":
        raise ArchiveAuthorityError(
            "SSH archive endpoint must authenticate with publickey"
        )
    return SshConnectionIdentity(
        authenticated_host=authenticated_host,
        address=address,
        port=port,
        authentication=authentication,
        host_key=host_key,
    )


def _parse_named_pins(
    value: object,
    *,
    names: tuple[str, ...],
    field: str,
) -> tuple[tuple[str, FilePin], ...]:
    mapping = _exact_mapping(value, names, field=field)
    return tuple(
        (name, _parse_file_pin(mapping[name], field=f"{field}.{name}"))
        for name in names
    )


def _parse_remote_helpers(value: object) -> tuple[tuple[str, str], ...]:
    mapping = _exact_mapping(
        value,
        REMOTE_HELPER_PATHS,
        field="remote helpers",
    )
    return tuple(
        (
            relative_path,
            _require_sha256(
                mapping[relative_path],
                field=f"remote helper SHA256 for {relative_path}",
            ),
        )
        for relative_path in REMOTE_HELPER_PATHS
    )


def _parse_file_pin(
    value: object,
    *,
    field: str,
    private_credential: bool = False,
) -> FilePin:
    mapping = _exact_mapping(value, ("path", "sha256"), field=field)
    path = _require_absolute_posix_text(mapping["path"], field=f"{field} path")
    digest = _require_sha256(mapping["sha256"], field=f"{field} SHA256")
    return FilePin(
        path=path,
        sha256=digest,
        private_credential=private_credential,
    )


def _named_pin(
    values: tuple[tuple[str, FilePin], ...],
    name: str,
    *,
    field: str,
) -> FilePin:
    for candidate, pin in values:
        if candidate == name:
            return pin
    raise ArchiveAuthorityError(f"missing {field}: {name}")


def _canonical_absolute_path(path: Path, *, field: str) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ArchiveAuthorityError(f"{field} does not exist: {raw}") from exc
    if raw != resolved:
        raise ArchiveAuthorityError(
            f"{field} must be an absolute canonical path without symlink ancestors"
        )
    current = Path(resolved.anchor)
    for part in resolved.parts[1:-1]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise ArchiveAuthorityError(
                f"cannot inspect {field} ancestor: {current}"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ArchiveAuthorityError(f"{field} has a non-directory ancestor")
    return resolved


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
        and left.st_nlink == right.st_nlink
    )


def _require_private_credential(
    metadata: os.stat_result,
    path: Path,
    *,
    required: bool,
) -> None:
    if not required:
        return
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ArchiveAuthorityError(
            "rclone credential config must be a current-UID-owned private "
            f"regular file with one hard link: {path}"
        )


def _exact_mapping(
    value: object,
    expected_keys: tuple[str, ...],
    *,
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(expected_keys):
        raise ArchiveAuthorityError(
            f"{field} must contain exactly: {', '.join(expected_keys)}"
        )
    return value


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ArchiveAuthorityError(f"{field} must be one non-empty safe string")
    return value


def _require_absolute_posix_text(value: object, *, field: str) -> str:
    text = _require_text(value, field=field)
    path = Path(text)
    if not path.is_absolute() or ".." in path.parts or str(path) != text.rstrip("/"):
        raise ArchiveAuthorityError(f"{field} must be normalized and absolute")
    return text


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ArchiveAuthorityError(f"{field} must be lowercase SHA256")
    return value


def _require_git_sha(value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA_PATTERN.fullmatch(value) is None:
        raise ArchiveAuthorityError("source git SHA must be one full lowercase commit")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ArchiveAuthorityError(f"non-finite JSON constant: {value}")
