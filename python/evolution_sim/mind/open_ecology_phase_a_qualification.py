"""Raw host qualification primitives for open-ecology Phase-A authority.

The functions in this module collect observations.  They do not decide whether
the observations authorize a campaign; the independent reconstruction and
threshold logic lives in :mod:`open_ecology_phase_a_readiness`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import contextvars
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import shutil
import stat
import ssl
import subprocess
import sys
import time
from typing import Any

from evolution_sim.io.open_ecology_archive_authority import (
    ArchiveToolAuthority,
    SEALED_SSH_OPTIONS,
    minimal_subprocess_env,
    parse_ssh_connection_identity,
    verify_local_authority_files,
)
from evolution_sim.io.open_ecology_campaign_storage import canonical_json_bytes
from evolution_sim.io.open_ecology_bounded_subprocess import (
    OpenEcologyProcessGroupError,
    terminate_process_group_before_reap,
    wait_for_leader_exit_without_reaping,
)
from evolution_sim.io.source_manifest import source_file_hash_manifest
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX,
    OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX,
    OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
    OPEN_ECOLOGY_SEED_REGISTRY,
)


SOURCE_OBSERVATION_SCHEMA_VERSION = "mind_v3_open_ecology_exact_source_observation_v1"
COMMAND_RECEIPT_SCHEMA_VERSION = "mind_v3_open_ecology_command_receipt_v1"
AUTHENTICATED_HTTPS_RECEIPT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_authenticated_https_receipt_v1"
)
RESOURCE_TELEMETRY_SCHEMA_VERSION = "mind_v3_open_ecology_phase_a_resource_telemetry_v1"
STORAGE_MEASUREMENT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_storage_measurement_v3"
)
REMOTE_STORAGE_PROBE_SCHEMA_VERSION = "mind_v3_open_ecology_remote_storage_probe_v1"
OUTPUT_LOCK_PROBE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_output_lock_contention_probe_v1"
)
GITHUB_REPOSITORY = "Reallisticus/evolution-sim"
GITHUB_TORCH_CHECK_NAME = "mind-recurrent-cpu"
RESOURCE_SAMPLE_INTERVAL_SECONDS = 5.0
COMMAND_OUTPUT_LIMIT_BYTES = 256 * 1024 * 1024
LOCK_PROBE_TIMEOUT_SECONDS = 20.0
BENCHMARK_TIMEOUT_SECONDS = 6 * 60 * 60.0
_SSH_TARGET_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}")
_SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SSH_EFFECTIVE_CONFIG_REDACTION_MARKER = "<redacted:ssh-effective-config>"
SSH_VERBOSE_LOG_REDACTION_MARKER = "<redacted:ssh-verbose-log>"
_GITHUB_API_HOST = "api.github.com"
_GITHUB_API_VERSION = "2022-11-28"
_EPHEMERAL_GITHUB_CREDENTIAL: contextvars.ContextVar[
    _OneShotGithubCredential | None
] = contextvars.ContextVar(
    "open_ecology_ephemeral_github_credential",
    default=None,
)


class _OneShotGithubCredential:
    __slots__ = ("_closed", "_payload")

    def __init__(self, payload: bytearray) -> None:
        if (
            len(payload) < 20
            or len(payload) > 4096
            or any(byte < 0x21 or byte > 0x7E for byte in payload)
        ):
            raise RuntimeError("ephemeral GitHub credential is malformed")
        self._payload = payload
        self._closed = False

    def __reduce__(self) -> object:
        raise TypeError("ephemeral GitHub credential cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("ephemeral GitHub credential cannot be serialized")

    def authorization_value(self) -> str:
        if self._closed:
            raise RuntimeError("ephemeral GitHub credential is closed")
        return "Bearer " + self._payload.decode("ascii")

    def reject_echo(self, *values: bytes | str) -> None:
        """Reject an exact credential reflection before a receipt can exist."""

        if self._closed:
            raise RuntimeError("ephemeral GitHub credential is closed")
        for value in values:
            haystack = value if isinstance(value, bytes) else value.encode("utf-8")
            if haystack.find(self._payload) >= 0:
                raise RuntimeError(
                    "GitHub API response violated the credential-redaction contract"
                )

    def close(self) -> None:
        for index in range(len(self._payload)):
            self._payload[index] = 0
        self._closed = True


@contextmanager
def ephemeral_github_credential(
    payload: bytearray,
) -> Any:
    """Install one process-local credential and zero it on every exit path."""

    if _EPHEMERAL_GITHUB_CREDENTIAL.get() is not None:
        raise RuntimeError("ephemeral GitHub credential is already installed")
    credential = _OneShotGithubCredential(payload)
    token = _EPHEMERAL_GITHUB_CREDENTIAL.set(credential)
    try:
        yield credential
    finally:
        _EPHEMERAL_GITHUB_CREDENTIAL.reset(token)
        credential.close()


_IMPORT_PROBE_CODE = r"""
import hashlib
import json
from pathlib import Path
import platform
import evolution_sim

module_file = Path(evolution_sim.__file__).resolve()
repository_root = Path(__import__("os").environ["EVOLUTION_SIM_REPOSITORY_ROOT"]).resolve()
print(json.dumps({
    "module_relative_path": module_file.relative_to(repository_root).as_posix(),
    "module_sha256": hashlib.sha256(module_file.read_bytes()).hexdigest(),
    "python_version": platform.python_version(),
}, sort_keys=True, separators=(",", ":")))
""".strip()


_LOCK_PROBE_CODE = r"""
import json
import os
from pathlib import Path
import sys
import time
from evolution_sim.io.open_ecology_campaign_storage import (
    CampaignStorageError,
    CampaignStorageLock,
)

lock_path = Path(sys.argv[1])
campaign_id = sys.argv[2]
source_git_sha = sys.argv[3]
hold_seconds = float(sys.argv[4])
started_ns = time.monotonic_ns()
try:
    with CampaignStorageLock(
        lock_path,
        campaign_id=campaign_id,
        source_git_sha=source_git_sha,
    ):
        acquired_ns = time.monotonic_ns()
        print(json.dumps({
            "acquired_monotonic_ns": acquired_ns,
            "outcome": "admitted",
            "pid": os.getpid(),
            "started_monotonic_ns": started_ns,
        }, sort_keys=True, separators=(",", ":")), flush=True)
        time.sleep(hold_seconds)
except CampaignStorageError as error:
    print(json.dumps({
        "error": str(error),
        "outcome": "rejected",
        "pid": os.getpid(),
        "started_monotonic_ns": started_ns,
    }, sort_keys=True, separators=(",", ":")), flush=True)
    raise SystemExit(3)
""".strip()


_REMOTE_STORAGE_PROBE_CODE = r"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

requested = Path(sys.argv[1])
if not requested.is_absolute() or requested.is_symlink():
    raise SystemExit("target directory must be one absolute non-symlink path")
try:
    target = requested.resolve(strict=True)
except OSError as error:
    raise SystemExit(f"target directory resolution failed: {error}") from error
if target != requested or not target.is_dir():
    raise SystemExit("target directory must be canonical and existing")
flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(target, flags)
try:
    before = os.fstat(descriptor)
    filesystem = os.fstatvfs(descriptor)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
if not stat.S_ISDIR(after.st_mode):
    raise SystemExit("target descriptor is not a directory")
if (
    before.st_dev != after.st_dev
    or before.st_ino != after.st_ino
    or before.st_mode != after.st_mode
):
    raise SystemExit("target directory identity changed during measurement")
boot_id_path = Path("/proc/sys/kernel/random/boot_id")
boot_id = boot_id_path.read_text(encoding="ascii").strip()
python_executable = Path(sys.executable).resolve(strict=True)
digest = hashlib.sha256()
with python_executable.open("rb") as handle:
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
print(json.dumps({
    "schema_version": "mind_v3_open_ecology_remote_storage_probe_v1",
    "target_directory": str(target),
    "remote_hostname": os.uname().nodename,
    "remote_boot_id": boot_id,
    "remote_python": {
        "path": str(python_executable),
        "sha256": digest.hexdigest(),
    },
    "target_filesystem": {
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "fragment_size": int(filesystem.f_frsize),
        "blocks": int(filesystem.f_blocks),
        "available_blocks": int(filesystem.f_bavail),
    },
    "filesystem_measurement_contract": (
        "remote_python_open_nofollow_directory_fstatvfs_fstat_identity_v1"
    ),
}, sort_keys=True, separators=(",", ":")))
""".strip()


def collect_exact_source_observation(repository_root: str | Path) -> dict[str, object]:
    """Capture exact HEAD, porcelain state, runtime manifest, and import root."""

    root = _canonical_directory(repository_root, field="repository root")
    head = run_command_receipt(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        timeout_seconds=30.0,
    )
    status = run_command_receipt(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=root,
        timeout_seconds=30.0,
    )
    environment = dict(os.environ)
    python_root = str(root / "python")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        python_root
        if not existing_pythonpath
        else os.pathsep.join((python_root, existing_pythonpath))
    )
    environment["EVOLUTION_SIM_REPOSITORY_ROOT"] = str(root)
    import_probe = run_command_receipt(
        (sys.executable, "-c", _IMPORT_PROBE_CODE),
        cwd=root,
        timeout_seconds=60.0,
        environment=environment,
    )
    return {
        "schema_version": SOURCE_OBSERVATION_SCHEMA_VERSION,
        "head": head,
        "status": status,
        "source_manifest": source_file_hash_manifest(root),
        "import_probe": import_probe,
        "import_probe_code_sha256": hashlib.sha256(
            _IMPORT_PROBE_CODE.encode("utf-8")
        ).hexdigest(),
    }


def collect_github_check_runs(
    repository_root: str | Path,
    *,
    source_commit: str,
) -> dict[str, object]:
    """Query concrete GitHub check-runs for one exact commit."""

    root = _canonical_directory(repository_root, field="repository root")
    source_commit = _source_commit(source_commit)
    credential = _EPHEMERAL_GITHUB_CREDENTIAL.get()
    if credential is not None:
        return _collect_github_check_runs_https(
            source_commit=source_commit,
            credential=credential,
        )
    executable = shutil.which("gh")
    if executable is None:
        raise RuntimeError("GitHub CLI is unavailable")
    endpoint = (
        f"repos/{GITHUB_REPOSITORY}/commits/{source_commit}/check-runs?per_page=100"
    )
    return run_command_receipt(
        (
            executable,
            "api",
            "--method",
            "GET",
            "-H",
            "Accept: application/vnd.github+json",
            endpoint,
        ),
        cwd=root,
        timeout_seconds=120.0,
    )


def _collect_github_check_runs_https(
    *,
    source_commit: str,
    credential: _OneShotGithubCredential,
) -> dict[str, object]:
    endpoint = (
        f"/repos/{GITHUB_REPOSITORY}/commits/{source_commit}/check-runs?per_page=100"
    )
    started_at = _utc_now()
    started_ns = time.monotonic_ns()
    connection = http.client.HTTPSConnection(
        _GITHUB_API_HOST,
        timeout=120.0,
        context=ssl.create_default_context(),
    )
    try:
        connection.request(
            "GET",
            endpoint,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": credential.authorization_value(),
                "User-Agent": "evolution-sim-open-ecology-guardian",
                "X-GitHub-Api-Version": _GITHUB_API_VERSION,
            },
        )
        response = connection.getresponse()
        payload = response.read(COMMAND_OUTPUT_LIMIT_BYTES + 1)
        if len(payload) > COMMAND_OUTPUT_LIMIT_BYTES:
            raise RuntimeError("GitHub API response exceeded its byte ceiling")
        status = response.status
        reason = response.reason
        if not isinstance(reason, str):
            raise RuntimeError("GitHub API response reason was not text")
        credential.reject_echo(payload, reason)
    finally:
        connection.close()
    elapsed_ns = time.monotonic_ns() - started_ns
    try:
        response_text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("GitHub API response was not UTF-8") from error
    return {
        "schema_version": AUTHENTICATED_HTTPS_RECEIPT_SCHEMA_VERSION,
        "request": {
            "authentication": "bearer_one_shot_in_memory_redacted",
            "credential_in_argv": False,
            "credential_in_environment": False,
            "credential_in_receipt": False,
            "credential_on_disk": False,
            "host": _GITHUB_API_HOST,
            "method": "GET",
            "path": endpoint,
            "response_byte_ceiling": COMMAND_OUTPUT_LIMIT_BYTES,
            "tls_context": "python_default_verified_context",
        },
        "response": {
            "body": response_text,
            "body_sha256": hashlib.sha256(payload).hexdigest(),
            "reason": reason,
            "status": status,
        },
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_ns": elapsed_ns,
    }


def run_torch_cuda_validation(
    repository_root: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, object]:
    """Run the complete repository Torch-gated suite plus CUDA smoke."""

    root = _canonical_directory(repository_root, field="repository root")
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    if output.exists() or output.is_symlink():
        raise RuntimeError("Torch validation output must be new")
    return run_command_receipt(
        (
            sys.executable,
            str(root / "scripts" / "validate_mind_torch.py"),
            "--require-cuda",
            "--output",
            str(output),
        ),
        cwd=root,
        timeout_seconds=30 * 60.0,
    )


def run_fresh_phase_a_benchmarks(
    repository_root: str | Path,
    *,
    source_commit: str,
    host_class: str,
    campaign_root: str | Path,
) -> dict[str, object]:
    """Run both full-shaped benchmark arms while sampling live host health."""

    root = _canonical_directory(repository_root, field="repository root")
    source_commit = _source_commit(source_commit)
    sample_root = _canonical_directory(campaign_root, field="campaign root")
    modes = ("heritable", "zero_all")
    benchmark_receipts: dict[str, object] = {}
    benchmark_reports: dict[str, object] = {}
    samples: list[dict[str, object]] = []
    for mode in modes:
        command = _phase_a_benchmark_command(
            root,
            population_mode=mode,
        )
        receipt, report, mode_samples = _run_monitored_benchmark(
            command,
            cwd=root,
            campaign_root=sample_root,
            population_mode=mode,
        )
        benchmark_receipts[mode] = receipt
        benchmark_reports[mode] = report
        samples.extend(mode_samples)
    return {
        "schema_version": RESOURCE_TELEMETRY_SCHEMA_VERSION,
        "source_commit": source_commit,
        "host_class": host_class,
        "sample_interval_seconds": RESOURCE_SAMPLE_INTERVAL_SECONDS,
        "benchmark_receipts": benchmark_receipts,
        "benchmark_reports": benchmark_reports,
        "resource_samples": samples,
    }


def measure_campaign_storage(
    target_directory: str | Path,
    *,
    archive_authority: ArchiveToolAuthority,
    drive_remote: str,
) -> dict[str, object]:
    """Measure remote campaign storage over SSH and local Drive via rclone."""

    ssh_target = _storage_ssh_target(archive_authority.ssh_target)
    target = _canonical_remote_directory(target_directory)
    verify_local_authority_files(archive_authority)
    ssh_pin = archive_authority.local_tool("ssh")
    rclone_pin = archive_authority.local_tool("rclone")
    remote_python_pin = archive_authority.remote_tool("python")
    repository_root = Path(__file__).resolve().parents[3]
    ssh_config_receipt = run_command_receipt(
        (
            ssh_pin.path,
            "-G",
            *SEALED_SSH_OPTIONS,
            ssh_target,
        ),
        cwd=repository_root,
        timeout_seconds=30.0,
        environment=minimal_subprocess_env(),
    )
    if (
        ssh_config_receipt["returncode"] != 0
        or str(ssh_config_receipt["stderr"]).strip()
        or not str(ssh_config_receipt["stdout"]).strip()
    ):
        raise RuntimeError("effective SSH configuration probe failed")
    ssh_effective_config_sha256 = hashlib.sha256(
        str(ssh_config_receipt["stdout"]).encode("utf-8")
    ).hexdigest()
    if ssh_effective_config_sha256 != archive_authority.ssh_effective_config_sha256:
        raise RuntimeError("effective SSH endpoint differs from authority pin")
    redacted_ssh_config_receipt = dict(ssh_config_receipt)
    redacted_ssh_config_receipt["stdout"] = SSH_EFFECTIVE_CONFIG_REDACTION_MARKER
    remote_command = storage_remote_probe_command(
        target,
        remote_python_path=remote_python_pin.path,
    )
    remote_probe_receipt = run_command_receipt(
        (
            ssh_pin.path,
            "-v",
            *SEALED_SSH_OPTIONS,
            ssh_target,
            remote_command,
        ),
        cwd=repository_root,
        timeout_seconds=120.0,
        environment=minimal_subprocess_env(),
    )
    if remote_probe_receipt["returncode"] != 0:
        raise RuntimeError("remote campaign filesystem probe failed")
    connection = parse_ssh_connection_identity(
        str(remote_probe_receipt["stderr"]).encode("utf-8")
    )
    if connection != archive_authority.ssh_connection:
        raise RuntimeError("authenticated SSH endpoint differs from authority pin")
    ssh_verbose_stderr_sha256 = hashlib.sha256(
        str(remote_probe_receipt["stderr"]).encode("utf-8")
    ).hexdigest()
    redacted_remote_probe_receipt = dict(remote_probe_receipt)
    redacted_remote_probe_receipt["stderr"] = SSH_VERBOSE_LOG_REDACTION_MARKER
    remote_probe = _strict_json_text(
        str(remote_probe_receipt["stdout"]),
        field="remote campaign filesystem probe",
    )
    if (
        remote_probe.get("schema_version") != REMOTE_STORAGE_PROBE_SCHEMA_VERSION
        or remote_probe.get("target_directory") != target
        or remote_probe.get("remote_python")
        != {
            "path": remote_python_pin.path,
            "sha256": remote_python_pin.sha256,
        }
    ):
        raise RuntimeError("remote campaign filesystem probe identity drifted")
    drive_receipt = run_command_receipt(
        (
            rclone_pin.path,
            "--config",
            archive_authority.rclone_config.path,
            "about",
            drive_remote,
            "--json",
        ),
        cwd=repository_root,
        timeout_seconds=120.0,
        environment=minimal_subprocess_env(),
    )
    if drive_receipt["returncode"] != 0 or str(drive_receipt["stderr"]).strip():
        raise RuntimeError("Drive capacity probe failed")
    _strict_json_text(
        str(drive_receipt["stdout"]),
        field="Drive capacity probe",
    )
    verify_local_authority_files(archive_authority)
    return {
        "schema_version": STORAGE_MEASUREMENT_SCHEMA_VERSION,
        "checked_at_utc": _utc_now(),
        "measurement_location": "remote_ssh_target_with_local_drive_v1",
        "target_ssh": ssh_target,
        "target_directory": target,
        "archive_authority_sha256": archive_authority.authority_sha256,
        "ssh_effective_config_sha256": ssh_effective_config_sha256,
        "ssh_verbose_stderr_sha256": ssh_verbose_stderr_sha256,
        "ssh_connection": connection.receipt_record(),
        "ssh_config_receipt": redacted_ssh_config_receipt,
        "remote_probe_code_sha256": remote_storage_probe_code_sha256(),
        "remote_probe_receipt": redacted_remote_probe_receipt,
        "drive_remote": drive_remote,
        "rclone_config": archive_authority.rclone_config.receipt_record(),
        "drive_receipt": drive_receipt,
    }


def run_output_lock_probe(
    lock_path: str | Path,
    *,
    campaign_id: str,
    source_git_sha: str,
) -> dict[str, object]:
    """Exercise contention, identity drift, release, and reacquisition."""

    source_git_sha = _source_commit(source_git_sha)
    lock = Path(lock_path)
    parent = _canonical_directory(lock.parent, field="output lock parent")
    if lock.parent.resolve() != parent or lock.exists() or lock.is_symlink():
        raise RuntimeError("output lock probe path must be new in one real parent")
    expected_identity = canonical_json_bytes(
        {
            "campaign_id": campaign_id,
            "schema_version": "open_ecology_campaign_storage_lock_v1",
            "source_git_sha": source_git_sha,
        }
    )
    first: subprocess.Popen[bytes] | None = None
    admitted = False
    try:
        first = _start_lock_worker(
            lock,
            campaign_id=campaign_id,
            source_git_sha=source_git_sha,
            hold_seconds=1.0,
        )
        first_event = _read_first_worker_event(first)
        if first_event.get("outcome") != "admitted":
            raise RuntimeError("first lock contender was not admitted")
        admitted = True

        second_receipt = _finish_lock_worker(
            _start_lock_worker(
                lock,
                campaign_id=campaign_id,
                source_git_sha=source_git_sha,
                hold_seconds=0.0,
            )
        )
        first_receipt = _finish_lock_worker(first, first_event=first_event)
        if lock.read_bytes() != expected_identity:
            raise RuntimeError("output lock identity did not persist after release")

        hostile_identity = canonical_json_bytes(
            {
                "campaign_id": f"{campaign_id}-identity-drift",
                "schema_version": "open_ecology_campaign_storage_lock_v1",
                "source_git_sha": source_git_sha,
            }
        )
        _restore_lock_identity(lock, hostile_identity)
        try:
            identity_receipt = _finish_lock_worker(
                _start_lock_worker(
                    lock,
                    campaign_id=campaign_id,
                    source_git_sha=source_git_sha,
                    hold_seconds=0.0,
                )
            )
        finally:
            _restore_lock_identity(lock, expected_identity)
        reacquire_receipt = _finish_lock_worker(
            _start_lock_worker(
                lock,
                campaign_id=campaign_id,
                source_git_sha=source_git_sha,
                hold_seconds=0.0,
            )
        )
        return {
            "schema_version": OUTPUT_LOCK_PROBE_SCHEMA_VERSION,
            "worker_program_sha256": hashlib.sha256(
                _LOCK_PROBE_CODE.encode("utf-8")
            ).hexdigest(),
            "campaign_id": campaign_id,
            "source_git_sha": source_git_sha,
            "lock_identity_sha256": hashlib.sha256(expected_identity).hexdigest(),
            "concurrent_contenders": [first_receipt, second_receipt],
            "identity_drift_attempt": identity_receipt,
            "post_release_reacquire": reacquire_receipt,
        }
    finally:
        try:
            if first is not None and first.poll() is None:
                _terminate(first)
        finally:
            try:
                if first is not None:
                    _close_process_pipes(first)
            finally:
                if admitted:
                    _restore_lock_identity(lock, expected_identity)


def run_command_receipt(
    command: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run one argv-only command and retain bounded exact output."""

    if not command or any(not isinstance(value, str) or not value for value in command):
        raise ValueError("command must be a non-empty sequence of strings")
    root = _canonical_directory(cwd, field="command cwd")
    executable = Path(command[0])
    if not executable.is_absolute():
        discovered = shutil.which(command[0])
        if discovered is None:
            raise RuntimeError(f"command executable is unavailable: {command[0]}")
        executable = Path(discovered)
    executable = executable.resolve(strict=True)
    if not executable.is_file() or executable.is_symlink():
        raise RuntimeError("command executable must be one real file")
    command = (str(executable), *command[1:])
    executable_before = executable.stat()
    executable_sha256 = _sha256_file(executable)
    started_at = _utc_now()
    started_ns = time.monotonic_ns()
    try:
        returncode, stdout_bytes, stderr_bytes = _run_bounded_subprocess(
            command,
            cwd=root,
            timeout_seconds=timeout_seconds,
            environment=environment,
            field="qualification command",
        )
    except OSError as error:
        raise RuntimeError(f"command failed to execute: {command[0]}") from error
    elapsed_ns = time.monotonic_ns() - started_ns
    try:
        stdout = stdout_bytes.decode("utf-8")
        stderr = stderr_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("command output was not UTF-8") from error
    executable_after = executable.stat()
    if (
        any(
            getattr(executable_before, field) != getattr(executable_after, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        )
        or _sha256_file(executable) != executable_sha256
    ):
        raise RuntimeError("command executable changed during qualification")
    return {
        "schema_version": COMMAND_RECEIPT_SCHEMA_VERSION,
        "argv": list(command),
        "executable": {
            "device": int(executable_after.st_dev),
            "inode": int(executable_after.st_ino),
            "size": int(executable_after.st_size),
            "sha256": executable_sha256,
        },
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_ns": elapsed_ns,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def import_probe_code_sha256() -> str:
    return hashlib.sha256(_IMPORT_PROBE_CODE.encode("utf-8")).hexdigest()


def lock_probe_code_sha256() -> str:
    return hashlib.sha256(_LOCK_PROBE_CODE.encode("utf-8")).hexdigest()


def remote_storage_probe_code_sha256() -> str:
    return hashlib.sha256(_REMOTE_STORAGE_PROBE_CODE.encode("utf-8")).hexdigest()


def storage_remote_probe_command(
    target_directory: str | Path,
    *,
    remote_python_path: str,
) -> str:
    target = _canonical_remote_directory(target_directory)
    python_path = _canonical_remote_file(remote_python_path)
    return shlex.join((python_path, "-c", _REMOTE_STORAGE_PROBE_CODE, target))


def _phase_a_benchmark_command(
    root: Path,
    *,
    population_mode: str,
) -> tuple[str, ...]:
    benchmark_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]
    return (
        sys.executable,
        str(root / "scripts" / "benchmark_recurrent_pipeline.py"),
        "--worker-counts",
        "1,2,4,8,16",
        "--repeats",
        "3",
        "--updates",
        "1",
        "--worlds-per-update",
        "16",
        "--rollout-ticks",
        "128",
        "--scenarios",
        "broad",
        "--device",
        "cuda",
        "--input-contract",
        "tokenized",
        "--genome-conditioning",
        "actor_film_v1",
        "--genome-population-mode",
        population_mode,
        "--genome-stream-seed",
        str(benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX]),
        "--encoder-size",
        "256",
        "--hidden-size",
        "256",
        "--recurrent-layers",
        "1",
        "--learner-seed",
        str(benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX]),
        "--update-epochs",
        "4",
        "--sequence-minibatch-size",
        "16",
        "--tbptt-steps",
        "128",
        "--burn-in-steps",
        "16",
        "--fixed-batch-capacity",
        "320",
        "--preregistered-gate-member",
    )


def _run_monitored_benchmark(
    command: Sequence[str],
    *,
    cwd: Path,
    campaign_root: Path,
    population_mode: str,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    from evolution_sim.cli import open_ecology_health

    started_at = _utc_now()
    started_ns = time.monotonic_ns()
    samples: list[dict[str, object]] = []

    def sample_resources() -> None:
        samples.append(
            {
                "population_mode": population_mode,
                "sampled_at_utc": _utc_now(),
                "host": open_ecology_health._collect_host_observations(campaign_root),
            }
        )

    # The two boundary samples prove the host state immediately before process
    # creation and immediately after process exit.  Interior samples are
    # collected by the bounded pipe-consumption loop.
    sample_resources()
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONPATH"] = str(cwd / "python")
    environment["EVOLUTION_SIM_REPOSITORY_ROOT"] = str(cwd)
    try:
        returncode, stdout, stderr = _run_bounded_subprocess(
            command,
            cwd=cwd,
            timeout_seconds=BENCHMARK_TIMEOUT_SECONDS,
            environment=environment,
            field=f"full-shaped {population_mode} benchmark",
            sample_callback=sample_resources,
            sample_interval_seconds=RESOURCE_SAMPLE_INTERVAL_SECONDS,
        )
    except OSError as error:
        raise RuntimeError("full-shaped benchmark failed to start") from error
    sample_resources()
    elapsed_ns = time.monotonic_ns() - started_ns
    try:
        stdout_text = stdout.decode("utf-8")
        stderr_text = stderr.decode("utf-8")
        report = json.loads(
            stdout_text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("benchmark did not emit strict UTF-8 JSON") from error
    if returncode != 0 or stderr_text.strip() or not isinstance(report, dict):
        raise RuntimeError(
            f"full-shaped {population_mode} benchmark failed closed: "
            f"returncode={returncode}"
        )
    receipt = {
        "schema_version": COMMAND_RECEIPT_SCHEMA_VERSION,
        "argv": list(command),
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_ns": elapsed_ns,
        "returncode": returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_byte_length": len(stdout),
        "stderr": stderr_text,
    }
    return receipt, report, samples


def _start_lock_worker(
    lock_path: Path,
    *,
    campaign_id: str,
    source_git_sha: str,
    hold_seconds: float,
) -> subprocess.Popen[bytes]:
    environment = dict(os.environ)
    repository_root = Path(__file__).resolve().parents[3]
    python_root = str(repository_root / "python")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (python_root, environment.get("PYTHONPATH", "")) if value
    )
    try:
        return subprocess.Popen(
            (
                sys.executable,
                "-c",
                _LOCK_PROBE_CODE,
                str(lock_path),
                campaign_id,
                source_git_sha,
                str(hold_seconds),
            ),
            cwd=repository_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise RuntimeError("output lock probe worker failed to start") from error


def _read_first_worker_event(process: subprocess.Popen[bytes]) -> dict[str, object]:
    if process.stdout is None:
        _terminate(process)
        raise RuntimeError("output lock probe worker has no stdout")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        events = selector.select(timeout=LOCK_PROBE_TIMEOUT_SECONDS)
        if not events:
            _terminate(process)
            raise RuntimeError("output lock probe worker timed out")
        line = process.stdout.readline()
    finally:
        selector.close()
    return _strict_json_line(line, field="output lock first contender")


def _finish_lock_worker(
    process: subprocess.Popen[bytes],
    *,
    first_event: Mapping[str, object] | None = None,
) -> dict[str, object]:
    try:
        try:
            stdout, stderr = process.communicate(timeout=LOCK_PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            _terminate(process)
            raise RuntimeError("output lock probe worker timed out") from error
        events = [] if first_event is None else [dict(first_event)]
        events.extend(
            _strict_json_line(line, field="output lock worker")
            for line in stdout.splitlines()
            if line
        )
        if len(events) != 1:
            raise RuntimeError("output lock worker emitted an invalid event count")
        try:
            stderr_text = stderr.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("output lock worker stderr was not UTF-8") from error
        return {
            "returncode": int(process.returncode),
            "stderr": stderr_text,
            "observed_finished_monotonic_ns": time.monotonic_ns(),
            "event": events[0],
        }
    finally:
        _close_process_pipes(process)


def _strict_json_line(payload: bytes, *, field: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{field} did not emit strict JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} JSON root must be an object")
    return value


def _strict_json_text(payload: str, *, field: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"{field} did not emit strict JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} JSON root must be an object")
    return value


def _terminate(
    process: subprocess.Popen[Any],
    *,
    leader_exit_observed: bool = False,
) -> int:
    """Terminate every process in the child-created session, then force kill."""

    try:
        return terminate_process_group_before_reap(
            process,
            wait_timeout_seconds=5.0,
            leader_exit_observed=leader_exit_observed,
        )
    except OpenEcologyProcessGroupError as error:
        raise RuntimeError(
            "qualification subprocess group survived termination"
        ) from error


def _close_process_pipes(process: subprocess.Popen[Any]) -> None:
    """Close owned Popen streams on every success and failure path."""

    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is None or stream.closed:
            continue
        try:
            stream.close()
        except OSError:
            pass


def _run_bounded_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None,
    field: str,
    sample_callback: Any | None = None,
    sample_interval_seconds: float | None = None,
) -> tuple[int, bytes, bytes]:
    """Consume child pipes incrementally and kill the whole group on breach."""

    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("subprocess timeout must be finite and positive")
    if sample_callback is None:
        if sample_interval_seconds is not None:
            raise ValueError("sample interval requires a callback")
    elif (
        sample_interval_seconds is None
        or sample_interval_seconds <= 0
        or not math.isfinite(sample_interval_seconds)
    ):
        raise ValueError("sample interval must be finite and positive")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=None if environment is None else dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            bufsize=0,
        )
    except OSError:
        raise
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    streams = {
        process.stdout.fileno(): ("stdout", process.stdout),
        process.stderr.fileno(): ("stderr", process.stderr),
    }
    buffers: dict[str, bytearray] = {
        "stdout": bytearray(),
        "stderr": bytearray(),
    }
    for descriptor, (_name, stream) in streams.items():
        os.set_blocking(descriptor, False)
        selector.register(stream, selectors.EVENT_READ, descriptor)
    started = time.monotonic()
    deadline = started + timeout_seconds
    next_sample = started
    group_cleanup_attempted = False
    try:
        while selector.get_map():
            now = time.monotonic()
            if sample_callback is not None and now >= next_sample:
                sample_callback()
                assert sample_interval_seconds is not None
                next_sample = now + sample_interval_seconds
            if now >= deadline:
                raise RuntimeError(f"{field} timed out")
            wake_at = deadline
            if sample_callback is not None:
                wake_at = min(wake_at, next_sample)
            events = selector.select(timeout=max(0.0, wake_at - now))
            for key, _mask in events:
                descriptor = int(key.data)
                name, stream = streams[descriptor]
                retained = len(buffers["stdout"]) + len(buffers["stderr"])
                read_size = min(
                    64 * 1024,
                    max(1, COMMAND_OUTPUT_LIMIT_BYTES - retained + 1),
                )
                try:
                    chunk = os.read(descriptor, read_size)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    continue
                buffers[name].extend(chunk)
                if (
                    len(buffers["stdout"]) + len(buffers["stderr"])
                    > COMMAND_OUTPUT_LIMIT_BYTES
                ):
                    raise RuntimeError(
                        f"{field} output exceeded the qualification byte limit"
                    )
        try:
            wait_for_leader_exit_without_reaping(process, deadline=deadline)
        except TimeoutError as error:
            raise RuntimeError(f"{field} timed out") from error
        group_cleanup_attempted = True
        returncode = _terminate(process, leader_exit_observed=True)
    except BaseException:
        if not group_cleanup_attempted:
            _terminate(process)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])


def _storage_ssh_target(value: object) -> str:
    if not isinstance(value, str) or _SSH_TARGET_PATTERN.fullmatch(value) is None:
        raise RuntimeError("storage SSH target must be one explicit host alias")
    return value


def _canonical_remote_directory(path: str | Path) -> str:
    if isinstance(path, Path):
        value = path.as_posix()
    elif isinstance(path, str):
        value = path
    else:
        raise RuntimeError("remote target directory must be text")
    candidate = PurePosixPath(value)
    if (
        not value
        or not candidate.is_absolute()
        or str(candidate) != value
        or ".." in candidate.parts
    ):
        raise RuntimeError(
            "remote target directory must be one canonical absolute path"
        )
    return value


def _canonical_remote_file(path: str | Path) -> str:
    value = _canonical_remote_directory(path)
    if value == "/":
        raise RuntimeError("remote executable path must name one file")
    return value


def _canonical_directory(path: str | Path, *, field: str) -> Path:
    candidate = Path(path)
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not candidate.is_dir()
        or candidate.resolve() != candidate
    ):
        raise RuntimeError(f"{field} must be one canonical existing directory")
    return candidate


def _source_commit(value: object) -> str:
    if not isinstance(value, str) or _SOURCE_COMMIT_PATTERN.fullmatch(value) is None:
        raise RuntimeError("source commit must be one lowercase 40-character SHA")
    return value


def _restore_lock_identity(path: Path, identity: bytes) -> None:
    """Restore an existing regular lock file without following path aliases."""

    flags = os.O_WRONLY | os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError("output lock identity path is not one regular file")
        view = memoryview(identity)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise RuntimeError("output lock identity restore made no progress")
            written += count
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or after.st_nlink != 1
            or after.st_size != len(identity)
        ):
            raise RuntimeError("output lock identity changed during restoration")
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "COMMAND_RECEIPT_SCHEMA_VERSION",
    "GITHUB_REPOSITORY",
    "GITHUB_TORCH_CHECK_NAME",
    "OUTPUT_LOCK_PROBE_SCHEMA_VERSION",
    "REMOTE_STORAGE_PROBE_SCHEMA_VERSION",
    "RESOURCE_TELEMETRY_SCHEMA_VERSION",
    "SSH_EFFECTIVE_CONFIG_REDACTION_MARKER",
    "SSH_VERBOSE_LOG_REDACTION_MARKER",
    "SOURCE_OBSERVATION_SCHEMA_VERSION",
    "STORAGE_MEASUREMENT_SCHEMA_VERSION",
    "collect_exact_source_observation",
    "collect_github_check_runs",
    "import_probe_code_sha256",
    "lock_probe_code_sha256",
    "measure_campaign_storage",
    "remote_storage_probe_code_sha256",
    "run_fresh_phase_a_benchmarks",
    "run_output_lock_probe",
    "run_torch_cuda_validation",
    "storage_remote_probe_command",
]
