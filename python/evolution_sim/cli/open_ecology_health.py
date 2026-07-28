"""Fail-closed host health probe for the persistent open-ecology campaign."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence


OPEN_ECOLOGY_HEALTH_BASELINE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_host_health_baseline_v3"
)
OPEN_ECOLOGY_HEALTH_SNAPSHOT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_host_health_snapshot_v1"
)
OPEN_ECOLOGY_HEALTH_MAX_BASELINE_BYTES = 64 * 1024
OPEN_ECOLOGY_HEALTH_MAX_COMMAND_BYTES = 16 * 1024 * 1024
OPEN_ECOLOGY_HEALTH_MAX_EXECUTABLE_BYTES = 16 * 1024 * 1024
OPEN_ECOLOGY_HEALTH_COMMAND_TIMEOUT_SECONDS = 15.0
OPEN_ECOLOGY_HEALTH_MIN_FREE_BYTES = 100 * 1024**3
OPEN_ECOLOGY_HEALTH_FREE_FRACTION_DENOMINATOR = 5
OPEN_ECOLOGY_HEALTH_MAX_RAM_SHARE = 0.80
OPEN_ECOLOGY_HEALTH_MAX_GPU_MEMORY_SHARE = 0.80
OPEN_ECOLOGY_FATAL_WORKER_MARKER_NAME = ".open-ecology-fatal-worker-pids.json"
OPEN_ECOLOGY_HEALTH_COMMAND_PATH = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
OPEN_ECOLOGY_HEALTH_COMMAND_NAMES = ("nvidia-smi", "journalctl")
_GENERIC_MAIN_SUFFIX = b"""

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpenEcologyHealthError as error:
        print(f"open-ecology health error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
"""

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_INTEGER_RE = re.compile(r"^-?[0-9]+$")
_TEMPERATURE_RE = re.compile(r"GPU (Current|Slowdown) Temp\s*:\s*([0-9]+|N/A)\s*C?")


class OpenEcologyHealthError(RuntimeError):
    """The campaign host cannot provide trustworthy health evidence."""


class OpenEcologyProcessGroupError(RuntimeError):
    """A health-probe subprocess group cannot be proven closed."""


class _HealthCommandSnapshot:
    """Private, read-only executable copies detached from mutable authorities."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="evosim-health-command-")
        self.root = Path(self._temporary.name).resolve()
        self._sealed = False
        self.executable_path: Path | None = None

    def add(self, name: str, encoded: bytes) -> Path:
        if self._sealed:
            raise OpenEcologyHealthError("health command snapshot is already sealed")
        path = self.root / name
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o500,
        )
        try:
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(path, 0o500)
        snapshot_bytes, _ = _pin_wrapper_input(
            path,
            field="health command executable snapshot",
            require_executable=True,
        )
        if snapshot_bytes != encoded:
            raise OpenEcologyHealthError("health command executable snapshot drifted")
        return path

    def seal(self) -> None:
        os.chmod(self.root, 0o500)
        self._sealed = True

    def cleanup(self) -> None:
        try:
            os.chmod(self.root, 0o700)
        except FileNotFoundError:
            pass
        self._temporary.cleanup()


def _darwin_immutable_execution_path(
    authority: Mapping[str, object],
) -> Path | None:
    """Use a canonical system object when macOS rejects its copied code signature."""

    if sys.platform != "darwin":
        return None
    resolved = Path(str(authority.get("path")))
    try:
        candidates = (resolved, *resolved.parents)
        effective_uid = os.geteuid()
        effective_groups = {os.getegid(), *os.getgroups()}
        for candidate in candidates:
            metadata = candidate.stat()
            mode = stat.S_IMODE(metadata.st_mode)
            if metadata.st_uid == effective_uid:
                writable = bool(mode & stat.S_IWUSR)
            elif metadata.st_gid in effective_groups:
                writable = bool(mode & stat.S_IWGRP)
            else:
                writable = bool(mode & stat.S_IWOTH)
            if writable:
                return None
    except OSError:
        return None
    return resolved


def wait_for_leader_exit_without_reaping(
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
) -> None:
    """Wait for leader exit while retaining its PID as an unreaped zombie."""

    required = ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT", "waitid")
    if any(not hasattr(os, name) for name in required):
        raise OpenEcologyProcessGroupError(
            "waitid WNOWAIT is unavailable for safe process-group cleanup"
        )
    options = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        try:
            status = os.waitid(os.P_PID, process.pid, options)
        except InterruptedError:
            continue
        except ChildProcessError as error:
            raise OpenEcologyProcessGroupError(
                "subprocess leader was reaped before group cleanup"
            ) from error
        if status is not None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("subprocess leader did not exit before deadline")
        time.sleep(min(0.01, remaining))


def terminate_process_group_before_reap(
    process: subprocess.Popen[bytes],
    *,
    wait_timeout_seconds: float,
    leader_exit_observed: bool = False,
) -> int:
    """Signal the original group before reaping its leader, then return status."""

    if process.returncode is not None:
        raise OpenEcologyProcessGroupError(
            "subprocess leader was already reaped before group cleanup"
        )
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        # SIGKILL below is the authoritative closure attempt.
        pass
    kill_error: OSError | None = None
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as error:
        kill_error = error
    try:
        returncode = process.wait(timeout=wait_timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise OpenEcologyProcessGroupError(
            "subprocess group leader survived force-kill"
        ) from error
    if kill_error is not None and not (
        leader_exit_observed and isinstance(kill_error, PermissionError)
    ):
        raise OpenEcologyProcessGroupError(
            "subprocess group could not be force-killed"
        ) from kill_error
    return returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or check an immutable host-health baseline for an "
            "open-ecology campaign."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "check"):
        command = subparsers.add_parser(name)
        command.add_argument("--baseline", required=True, type=Path)
        command.add_argument("--campaign-root", required=True, type=Path)
        command.add_argument("--campaign-id", required=True)
        command.add_argument("--source-git-sha", required=True)
        command.add_argument("--source-manifest-sha256", required=True)
        if name == "check":
            command.add_argument("--expected-baseline-sha256", required=True)
    wrapper = subparsers.add_parser("build-wrapper")
    wrapper.add_argument("--output", required=True, type=Path)
    wrapper.add_argument("--python-executable", required=True, type=Path)
    wrapper.add_argument("--health-script", required=True, type=Path)
    wrapper.add_argument("--expected-health-script-sha256", required=True)
    wrapper.add_argument("--baseline", required=True, type=Path)
    wrapper.add_argument("--expected-baseline-sha256", required=True)
    wrapper.add_argument("--campaign-root", required=True, type=Path)
    wrapper.add_argument("--campaign-id", required=True)
    wrapper.add_argument("--source-git-sha", required=True)
    wrapper.add_argument("--source-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    identity = _identity(
        campaign_root=arguments.campaign_root,
        campaign_id=arguments.campaign_id,
        source_git_sha=arguments.source_git_sha,
        source_manifest_sha256=arguments.source_manifest_sha256,
    )
    if arguments.command == "build-wrapper":
        result = build_health_probe_wrapper(
            arguments.output,
            python_executable=arguments.python_executable,
            health_script=arguments.health_script,
            expected_health_script_sha256=(arguments.expected_health_script_sha256),
            baseline=arguments.baseline,
            expected_baseline_sha256=arguments.expected_baseline_sha256,
            identity=identity,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "init":
        baseline, baseline_file_sha256 = _initialize_health_baseline_with_digest(
            arguments.baseline,
            identity=identity,
        )
        print(
            json.dumps(
                {
                    "baseline": baseline,
                    "health_baseline_sha256": baseline_file_sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    snapshot = check_campaign_health(
        arguments.baseline,
        identity=identity,
        phase=os.environ.get("EVOSIM_OPEN_ECOLOGY_HEALTH_PHASE"),
        frontier_tick=os.environ.get("EVOSIM_OPEN_ECOLOGY_FRONTIER_TICK"),
        expected_baseline_sha256=arguments.expected_baseline_sha256,
    )
    print(json.dumps(snapshot, separators=(",", ":"), sort_keys=True))
    return 0


def initialize_health_baseline(
    path: str | Path,
    *,
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Create one immutable baseline before any campaign process starts."""

    baseline, _ = _initialize_health_baseline_with_digest(path, identity=identity)
    return baseline


def _initialize_health_baseline_with_digest(
    path: str | Path,
    *,
    identity: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    baseline_path = _new_baseline_path(path)
    normalized_identity = _validate_identity(identity)
    command_authorities = _discover_command_authorities()
    host = _collect_host_observations(
        Path(normalized_identity["campaign_root"]),
        command_authorities=command_authorities,
    )
    baseline: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_HEALTH_BASELINE_SCHEMA_VERSION,
        **normalized_identity,
        "boot_id": host["boot_id"],
        "swap_used_bytes": host["memory"]["swap_used_bytes"],
        "xid_count": host["xid_count"],
        "oom_count": host["oom_count"],
        "command_authorities": command_authorities,
        "created_at_utc": _utc_now(),
    }
    baseline["exact_digest"] = _payload_digest(baseline)
    payload = _canonical_json_bytes(baseline)
    if len(payload) > OPEN_ECOLOGY_HEALTH_MAX_BASELINE_BYTES:
        raise OpenEcologyHealthError("health baseline exceeds 64 KiB")
    descriptor = os.open(
        baseline_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(baseline_path.parent)
    file_sha256 = hashlib.sha256(payload).hexdigest()
    loaded = _load_health_baseline(
        baseline_path,
        identity=normalized_identity,
        expected_file_sha256=file_sha256,
    )
    return loaded, file_sha256


def build_health_probe_wrapper(
    path: str | Path,
    *,
    python_executable: str | Path,
    health_script: str | Path,
    expected_health_script_sha256: object,
    baseline: str | Path,
    expected_baseline_sha256: object,
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Write one sealed, argument-free executable wrapper for campaign use."""

    output = _new_wrapper_path(path)
    normalized_identity = _validate_identity(identity)
    baseline_digest = _sha256(
        expected_baseline_sha256,
        field="expected health baseline SHA256",
    )
    _load_health_baseline(
        baseline,
        identity=normalized_identity,
        expected_file_sha256=baseline_digest,
    )
    _, python_authority = _pin_wrapper_input(
        Path(python_executable),
        field="health wrapper Python executable",
        require_executable=True,
    )
    source_bytes, script_authority = _pin_wrapper_input(
        Path(health_script),
        field="health wrapper source script",
        require_executable=False,
    )
    expected_script_digest = _sha256(
        expected_health_script_sha256,
        field="expected health wrapper source SHA256",
    )
    if script_authority["sha256"] != expected_script_digest:
        raise OpenEcologyHealthError(
            "health wrapper source does not match external SHA256 authority"
        )
    baseline_path = Path(baseline)
    if (
        not baseline_path.is_absolute()
        or baseline_path.is_symlink()
        or baseline_path.resolve(strict=True) != baseline_path
    ):
        raise OpenEcologyHealthError("health wrapper baseline path must be canonical")
    bound_arguments = (
        "check",
        "--baseline",
        str(baseline_path),
        "--campaign-root",
        str(normalized_identity["campaign_root"]),
        "--campaign-id",
        str(normalized_identity["campaign_id"]),
        "--source-git-sha",
        str(normalized_identity["source_git_sha"]),
        "--source-manifest-sha256",
        str(normalized_identity["source_manifest_sha256"]),
        "--expected-baseline-sha256",
        baseline_digest,
    )
    if source_bytes.startswith(b"#!") or not source_bytes.endswith(
        _GENERIC_MAIN_SUFFIX
    ):
        raise OpenEcologyHealthError(
            "health wrapper source does not have the exact generic main suffix"
        )
    bound_terminal = (
        '\n\nif __name__ == "__main__":\n'
        "    try:\n"
        f"        raise SystemExit(main({list(bound_arguments)!r}))\n"
        "    except OpenEcologyHealthError as error:\n"
        '        print(f"open-ecology health error: {error}", file=sys.stderr)\n'
        "        raise SystemExit(2) from error\n"
    ).encode("utf-8")
    python_path = str(python_authority["path"])
    if any(character.isspace() for character in python_path):
        raise OpenEcologyHealthError(
            "health wrapper Python path cannot contain whitespace"
        )
    encoded = (
        (
            f"#!{python_path}\n# evosim_shebang_sha256={python_authority['sha256']}\n"
        ).encode("utf-8")
        + source_bytes[: -len(_GENERIC_MAIN_SUFFIX)]
        + bound_terminal
    )
    descriptor = os.open(
        output,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o500,
    )
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(output.parent)
    return {
        "wrapper_path": str(output),
        "wrapper_sha256": hashlib.sha256(encoded).hexdigest(),
        "python_executable": python_authority,
        "health_script": script_authority,
        "health_baseline_sha256": baseline_digest,
    }


def check_campaign_health(
    path: str | Path,
    *,
    identity: Mapping[str, object],
    phase: object,
    frontier_tick: object,
    expected_baseline_sha256: object,
) -> dict[str, object]:
    """Measure the live host and affirm health only when every gate passes."""

    normalized_identity = _validate_identity(identity)
    parsed_phase = _phase(phase)
    parsed_frontier = _nonnegative_integer(frontier_tick, field="frontier_tick")
    baseline = _load_health_baseline(
        path,
        identity=normalized_identity,
        expected_file_sha256=_sha256(
            expected_baseline_sha256,
            field="expected health baseline SHA256",
        ),
    )
    command_authorities = _validate_command_authorities(
        baseline["command_authorities"],
        revalidate=True,
    )
    blockers: list[str] = []
    fatal_worker_marker = (
        Path(normalized_identity["campaign_root"])
        / OPEN_ECOLOGY_FATAL_WORKER_MARKER_NAME
    )
    if fatal_worker_marker.exists() or fatal_worker_marker.is_symlink():
        blockers.append("fatal_worker_pid_marker_present")
    try:
        host = _collect_host_observations(
            Path(normalized_identity["campaign_root"]),
            command_authorities=command_authorities,
        )
    except OpenEcologyHealthError as error:
        host = {"collection_error": str(error)}
        blockers.append(f"collection_error:{error}")
    else:
        memory = _mapping(host["memory"], field="memory")
        filesystem = _mapping(host["filesystem"], field="filesystem")
        if host["boot_id"] != baseline["boot_id"]:
            blockers.append("host_boot_id_changed")
        if int(memory["swap_used_bytes"]) > int(baseline["swap_used_bytes"]):
            blockers.append("swap_usage_increased")
        if float(memory["ram_used_share"]) >= OPEN_ECOLOGY_HEALTH_MAX_RAM_SHARE:
            blockers.append("host_ram_share_at_or_above_0.80")
        if int(filesystem["free_bytes"]) < int(filesystem["required_free_bytes"]):
            blockers.append("campaign_filesystem_below_free_space_floor")
        if int(host["xid_count"]) != int(baseline["xid_count"]):
            blockers.append("nvidia_xid_count_changed")
        if int(host["oom_count"]) != int(baseline["oom_count"]):
            blockers.append("kernel_oom_count_changed")
        for gpu in _sequence(host["gpus"], field="gpus"):
            record = _mapping(gpu, field="gpu")
            if (
                float(record["memory_used_share"])
                >= OPEN_ECOLOGY_HEALTH_MAX_GPU_MEMORY_SHARE
            ):
                blockers.append(f"gpu_{record['index']}_memory_share_at_or_above_0.80")
            if int(record["temperature_c"]) >= int(record["slowdown_temperature_c"]):
                blockers.append(
                    f"gpu_{record['index']}_at_or_above_slowdown_temperature"
                )
    snapshot: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_HEALTH_SNAPSHOT_SCHEMA_VERSION,
        **normalized_identity,
        "phase": parsed_phase,
        "frontier_tick": parsed_frontier,
        "healthy": not blockers,
        "blockers": blockers,
        "observed_at_utc": _utc_now(),
        "baseline_exact_digest": baseline["exact_digest"],
        "baseline_file_sha256": _sha256(
            expected_baseline_sha256,
            field="expected health baseline SHA256",
        ),
        "host": host,
    }
    snapshot["snapshot_sha256"] = _payload_digest(snapshot)
    return snapshot


def collect_host_observations(campaign_root: Path) -> dict[str, object]:
    """Collect one live host snapshot for an injected research consumer."""

    return _collect_host_observations(campaign_root)


def _collect_host_observations(
    campaign_root: Path,
    *,
    command_authorities: Mapping[str, object] | None = None,
) -> dict[str, object]:
    authorities = (
        _discover_command_authorities()
        if command_authorities is None
        else _validate_command_authorities(command_authorities, revalidate=True)
    )
    memory = _read_memory()
    filesystem = _read_filesystem(campaign_root)
    gpus = _read_gpus(authorities)
    return {
        "boot_id": _read_boot_id(),
        "memory": memory,
        "filesystem": filesystem,
        "gpus": gpus,
        "xid_count": _read_xid_count(authorities),
        "oom_count": _read_oom_count(authorities),
        "load_average": list(os.getloadavg()),
        "process_id": os.getpid(),
    }


def _read_memory() -> dict[str, object]:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OpenEcologyHealthError("cannot read /proc/meminfo") from error
    values: dict[str, int] = {}
    for line in lines:
        key, separator, raw = line.partition(":")
        if not separator:
            continue
        parts = raw.strip().split()
        if not parts or not _INTEGER_RE.fullmatch(parts[0]):
            continue
        multiplier = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
        values[key] = int(parts[0]) * multiplier
    required = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    if any(key not in values for key in required):
        raise OpenEcologyHealthError("/proc/meminfo omits required fields")
    total = values["MemTotal"]
    available = values["MemAvailable"]
    swap_total = values["SwapTotal"]
    swap_free = values["SwapFree"]
    if total <= 0 or not 0 <= available <= total or not 0 <= swap_free <= swap_total:
        raise OpenEcologyHealthError("/proc/meminfo values are inconsistent")
    return {
        "ram_total_bytes": total,
        "ram_available_bytes": available,
        "ram_used_share": (total - available) / total,
        "swap_total_bytes": swap_total,
        "swap_used_bytes": swap_total - swap_free,
    }


def _read_filesystem(campaign_root: Path) -> dict[str, object]:
    if (
        not campaign_root.is_absolute()
        or not campaign_root.is_dir()
        or campaign_root.is_symlink()
        or campaign_root.resolve() != campaign_root
    ):
        raise OpenEcologyHealthError(
            "campaign root must be one canonical existing directory"
        )
    stats = os.statvfs(campaign_root)
    capacity = stats.f_blocks * stats.f_frsize
    free = stats.f_bavail * stats.f_frsize
    required = max(
        OPEN_ECOLOGY_HEALTH_MIN_FREE_BYTES,
        (capacity + OPEN_ECOLOGY_HEALTH_FREE_FRACTION_DENOMINATOR - 1)
        // OPEN_ECOLOGY_HEALTH_FREE_FRACTION_DENOMINATOR,
    )
    return {
        "capacity_bytes": capacity,
        "free_bytes": free,
        "required_free_bytes": required,
    }


def _read_boot_id() -> str:
    try:
        boot_id = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        )
    except OSError as error:
        raise OpenEcologyHealthError("cannot read Linux boot identity") from error
    if re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot_id) is None:
        raise OpenEcologyHealthError("Linux boot identity is malformed")
    return boot_id


def _read_gpus(
    command_authorities: Mapping[str, object],
) -> list[dict[str, object]]:
    nvidia_smi = _command_path(
        command_authorities,
        name="nvidia-smi",
    )
    inventory = _run_command(
        (
            str(nvidia_smi),
            "--query-gpu=index,name,memory.total,memory.used,"
            "temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        command_authority=_mapping(
            command_authorities["nvidia-smi"],
            field="nvidia-smi authority",
        ),
    )
    temperature_report = _run_command(
        (str(nvidia_smi), "-q", "-d", "TEMPERATURE"),
        command_authority=_mapping(
            command_authorities["nvidia-smi"],
            field="nvidia-smi authority",
        ),
    )
    current_and_slowdown = _parse_temperature_report(temperature_report)
    rows: list[dict[str, object]] = []
    try:
        parsed_rows = list(csv.reader(inventory.splitlines(), skipinitialspace=True))
    except csv.Error as error:
        raise OpenEcologyHealthError("nvidia-smi inventory CSV is malformed") from error
    if not parsed_rows:
        raise OpenEcologyHealthError("nvidia-smi returned no GPU")
    for raw in parsed_rows:
        if len(raw) != 6:
            raise OpenEcologyHealthError("nvidia-smi inventory field count drifted")
        index = _nonnegative_integer(raw[0].strip(), field="GPU index")
        total_mib = _positive_integer(raw[2].strip(), field="GPU memory total")
        used_mib = _nonnegative_integer(raw[3].strip(), field="GPU memory used")
        current_c = _nonnegative_integer(raw[4].strip(), field="GPU temperature")
        utilization = _nonnegative_integer(raw[5].strip(), field="GPU utilization")
        if used_mib > total_mib or utilization > 100:
            raise OpenEcologyHealthError("nvidia-smi inventory values are inconsistent")
        try:
            detailed_current, slowdown_c = current_and_slowdown[index]
        except KeyError as error:
            raise OpenEcologyHealthError(
                "nvidia-smi detailed temperature count differs from inventory"
            ) from error
        if current_c != detailed_current:
            raise OpenEcologyHealthError(
                "GPU current temperature differs between inventory views"
            )
        rows.append(
            {
                "index": index,
                "name": raw[1].strip(),
                "memory_total_bytes": total_mib * 1024**2,
                "memory_used_bytes": used_mib * 1024**2,
                "memory_used_share": used_mib / total_mib,
                "temperature_c": current_c,
                "slowdown_temperature_c": slowdown_c,
                "utilization_percent": utilization,
            }
        )
    if [row["index"] for row in rows] != list(range(len(rows))):
        raise OpenEcologyHealthError("GPU indices are not canonical and contiguous")
    return rows


def _parse_temperature_report(value: str) -> dict[int, tuple[int, int]]:
    records: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    for line in value.splitlines():
        if line.startswith("GPU ") and re.fullmatch(
            r"GPU [0-9A-Fa-f:.]+", line.strip()
        ):
            current = {}
            records.append(current)
            continue
        if current is None:
            continue
        match = _TEMPERATURE_RE.search(line)
        if match is None or match.group(2) == "N/A":
            continue
        current[match.group(1).lower()] = int(match.group(2))
    if not records or any(set(record) != {"current", "slowdown"} for record in records):
        raise OpenEcologyHealthError(
            "nvidia-smi does not expose current and slowdown temperatures"
        )
    return {
        index: (record["current"], record["slowdown"])
        for index, record in enumerate(records)
    }


def _read_xid_count(command_authorities: Mapping[str, object]) -> int:
    return _read_kernel_event_count(
        r"NVRM: Xid|Xid \(",
        command_authorities=command_authorities,
    )


def _read_oom_count(command_authorities: Mapping[str, object]) -> int:
    return _read_kernel_event_count(
        r"Out of memory:|Killed process .* total-vm:|oom-kill:",
        command_authorities=command_authorities,
    )


def _read_kernel_event_count(
    pattern: str,
    *,
    command_authorities: Mapping[str, object] | None = None,
) -> int:
    authorities = (
        _discover_command_authorities()
        if command_authorities is None
        else _validate_command_authorities(command_authorities, revalidate=True)
    )
    journalctl = _command_path(authorities, name="journalctl")
    output = _run_command(
        (
            str(journalctl),
            "-k",
            "-b",
            "--no-pager",
            "-o",
            "cat",
            "--grep",
            pattern,
        ),
        accepted_returncodes=(0, 1),
        command_authority=_mapping(
            authorities["journalctl"],
            field="journalctl authority",
        ),
    )
    return sum(bool(line.strip()) for line in output.splitlines())


def _run_command(
    command: Sequence[str],
    *,
    accepted_returncodes: Sequence[int] = (0,),
    command_authority: Mapping[str, object],
) -> str:
    if not command or not Path(command[0]).is_absolute():
        raise OpenEcologyHealthError(
            "health command must use one absolute pinned executable"
        )
    expected_authority = _validate_command_authority(
        command_authority,
        expected_name=str(command_authority.get("name")),
        revalidate=True,
    )
    if command[0] != expected_authority["path"]:
        raise OpenEcologyHealthError(
            f"health command path is not the pinned authority: {command[0]}"
        )
    execution_command, snapshot = _prepare_command_execution(
        command,
        expected_authority=expected_authority,
    )
    try:
        stdout_bytes, stderr_bytes, returncode = _run_bounded_command(
            execution_command,
            executable=None if snapshot is None else snapshot.executable_path,
        )
    finally:
        if snapshot is not None:
            snapshot.cleanup()
    if returncode not in accepted_returncodes:
        raise OpenEcologyHealthError(
            f"health command returned {returncode}: {command[0]}"
        )
    try:
        stdout = stdout_bytes.decode("utf-8")
        stderr = stderr_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OpenEcologyHealthError(
            f"health command output is not UTF-8: {command[0]}"
        ) from error
    if stderr.strip():
        raise OpenEcologyHealthError(
            f"health command reported stderr and cannot prove success: {command[0]}"
        )
    try:
        post_execution_authority = _validate_command_authority(
            command_authority,
            expected_name=str(command_authority.get("name")),
            revalidate=True,
        )
    except OpenEcologyHealthError as error:
        raise OpenEcologyHealthError(
            f"health command executable changed during execution: {command[0]}"
        ) from error
    if post_execution_authority != expected_authority:
        raise OpenEcologyHealthError(
            f"health command executable changed during execution: {command[0]}"
        )
    return stdout


def _prepare_command_execution(
    command: Sequence[str],
    *,
    expected_authority: Mapping[str, object],
) -> tuple[tuple[str, ...], _HealthCommandSnapshot | None]:
    encoded, current_authority, interpreter_bytes = _read_pinned_command_executable(
        Path(str(expected_authority["path"])),
        name=str(expected_authority["name"]),
    )
    if current_authority != dict(expected_authority):
        raise OpenEcologyHealthError(
            f"health command executable identity or SHA256 changed: "
            f"{expected_authority['name']}"
        )
    tail = tuple(command[1:])
    mode = str(expected_authority["execution_mode"])
    if mode == "validated_native_snapshot_v1":
        immutable_path = _darwin_immutable_execution_path(current_authority)
        if immutable_path is not None:
            return (str(immutable_path), *tail), None
        snapshot = _HealthCommandSnapshot()
        try:
            executable_path = snapshot.add("executable", encoded)
            snapshot.executable_path = executable_path
            snapshot.seal()
        except BaseException:
            snapshot.cleanup()
            raise
        return (str(expected_authority["path"]), *tail), snapshot
    if mode != "validated_script_and_interpreter_snapshot_v1":
        raise OpenEcologyHealthError("health command execution mode drifted")
    interpreter_authority = _mapping(
        expected_authority["interpreter"],
        field="health command interpreter authority",
    )
    if interpreter_bytes is None:
        raise OpenEcologyHealthError("health command interpreter bytes are absent")
    snapshot = _HealthCommandSnapshot()
    try:
        interpreter_path = _darwin_immutable_execution_path(interpreter_authority)
        if interpreter_path is None:
            interpreter_path = snapshot.add("interpreter", interpreter_bytes)
            snapshot.executable_path = interpreter_path
        script_path = snapshot.add("script", encoded)
        snapshot.seal()
    except BaseException:
        snapshot.cleanup()
        raise
    return (
        str(interpreter_authority["path"]),
        str(script_path),
        *tail,
    ), snapshot


def _run_bounded_command(
    command: Sequence[str],
    *,
    executable: Path | None = None,
) -> tuple[bytes, bytes, int]:
    selector: selectors.BaseSelector | None = None
    stdout = None
    stderr = None
    buffers: dict[str, bytearray] = {
        "stdout": bytearray(),
        "stderr": bytearray(),
    }
    group_cleanup_attempted = False
    try:
        process = subprocess.Popen(
            list(command),
            executable=None if executable is None else str(executable),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": OPEN_ECOLOGY_HEALTH_COMMAND_PATH,
            },
            start_new_session=True,
        )
    except OSError as error:
        raise OpenEcologyHealthError(
            f"health command failed to execute: {command[0]}"
        ) from error
    try:
        stdout = process.stdout
        stderr = process.stderr
        if stdout is None or stderr is None:
            raise OpenEcologyHealthError(
                f"health command pipes were not created: {command[0]}"
            )
        selector = selectors.DefaultSelector()
        streams = {
            stdout.fileno(): ("stdout", stdout),
            stderr.fileno(): ("stderr", stderr),
        }
        for descriptor, (name, _) in streams.items():
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ, data=name)
        deadline = time.monotonic() + OPEN_ECOLOGY_HEALTH_COMMAND_TIMEOUT_SECONDS
        total_bytes = 0
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OpenEcologyHealthError(f"health command timed out: {command[0]}")
            events = selector.select(timeout=remaining)
            if not events:
                raise OpenEcologyHealthError(f"health command timed out: {command[0]}")
            for key, _ in events:
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                total_bytes += len(chunk)
                if total_bytes > OPEN_ECOLOGY_HEALTH_MAX_COMMAND_BYTES:
                    raise OpenEcologyHealthError(
                        f"health command output exceeds 16 MiB: {command[0]}"
                    )
                buffers[str(key.data)].extend(chunk)
        try:
            wait_for_leader_exit_without_reaping(process, deadline=deadline)
        except TimeoutError as error:
            raise OpenEcologyHealthError(
                f"health command timed out: {command[0]}"
            ) from error
        group_cleanup_attempted = True
        returncode = _terminate_command(
            process,
            leader_exit_observed=True,
        )
    except BaseException as primary_error:
        if not group_cleanup_attempted:
            try:
                _terminate_command(process)
            except OpenEcologyHealthError as cleanup_error:
                primary_error.add_note(f"additional cleanup failure: {cleanup_error}")
                raise primary_error from cleanup_error
        raise
    finally:
        if selector is not None:
            selector.close()
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()
    return bytes(buffers["stdout"]), bytes(buffers["stderr"]), returncode


def _terminate_command(
    process: subprocess.Popen[bytes],
    *,
    leader_exit_observed: bool = False,
) -> int:
    try:
        return terminate_process_group_before_reap(
            process,
            wait_timeout_seconds=2.0,
            leader_exit_observed=leader_exit_observed,
        )
    except OpenEcologyProcessGroupError as error:
        raise OpenEcologyHealthError(
            "health command process group survived termination"
        ) from error


def _discover_command_authorities() -> dict[str, object]:
    authorities: dict[str, object] = {}
    for name in OPEN_ECOLOGY_HEALTH_COMMAND_NAMES:
        discovered = shutil.which(name, path=OPEN_ECOLOGY_HEALTH_COMMAND_PATH)
        if discovered is None:
            raise OpenEcologyHealthError(
                f"required health command is absent from constrained path: {name}"
            )
        authorities[name] = _pin_command_executable(Path(discovered), name=name)
    return _validate_command_authorities(authorities, revalidate=True)


def _validate_command_authorities(
    value: object,
    *,
    revalidate: bool,
) -> dict[str, object]:
    authorities = _mapping(value, field="health command authorities")
    if set(authorities) != set(OPEN_ECOLOGY_HEALTH_COMMAND_NAMES):
        raise OpenEcologyHealthError("health command authority names are not exact")
    return {
        name: _validate_command_authority(
            _mapping(authorities[name], field=f"{name} authority"),
            expected_name=name,
            revalidate=revalidate,
        )
        for name in OPEN_ECOLOGY_HEALTH_COMMAND_NAMES
    }


def _validate_command_authority(
    value: Mapping[str, object],
    *,
    expected_name: str,
    revalidate: bool,
) -> dict[str, object]:
    required = {
        "name",
        "path",
        "device",
        "inode",
        "mode",
        "link_count",
        "size",
        "mtime_ns",
        "ctime_ns",
        "sha256",
        "execution_mode",
        "interpreter",
        "authority_sha256",
    }
    if set(value) != required or value.get("name") != expected_name:
        raise OpenEcologyHealthError(
            f"health command authority schema drifted: {expected_name}"
        )
    unsigned = {key: value[key] for key in required if key != "authority_sha256"}
    if value.get("authority_sha256") != _payload_digest(unsigned):
        raise OpenEcologyHealthError(
            f"health command authority digest drifted: {expected_name}"
        )
    execution_mode = value.get("execution_mode")
    if execution_mode == "validated_native_snapshot_v1":
        if value.get("interpreter") is not None:
            raise OpenEcologyHealthError(
                f"native health command cannot have an interpreter: {expected_name}"
            )
    elif execution_mode == "validated_script_and_interpreter_snapshot_v1":
        _validate_command_interpreter_authority(
            _mapping(
                value.get("interpreter"),
                field=f"{expected_name} interpreter authority",
            ),
            expected_name=expected_name,
            revalidate=False,
        )
    else:
        raise OpenEcologyHealthError(
            f"health command execution mode drifted: {expected_name}"
        )
    path = Path(str(value["path"]))
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise OpenEcologyHealthError(
            f"health command authority path is not canonical: {expected_name}"
        )
    normalized = dict(value)
    if revalidate:
        current = _pin_command_executable(path, name=expected_name)
        if current != normalized:
            raise OpenEcologyHealthError(
                f"health command executable identity or SHA256 changed: {expected_name}"
            )
    return normalized


def _pin_command_executable(path: Path, *, name: str) -> dict[str, object]:
    _, authority, _ = _read_pinned_command_executable(path, name=name)
    return authority


def _read_pinned_command_executable(
    path: Path,
    *,
    name: str,
) -> tuple[bytes, dict[str, object], bytes | None]:
    if name not in OPEN_ECOLOGY_HEALTH_COMMAND_NAMES or not path.is_absolute():
        raise OpenEcologyHealthError("health command identity is malformed")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise OpenEcologyHealthError(
            f"health command cannot be resolved: {name}"
        ) from error
    encoded, executable = _pin_wrapper_input(
        resolved,
        field=f"health command executable {name}",
        require_executable=True,
    )
    interpreter_bytes: bytes | None = None
    interpreter: dict[str, object] | None = None
    if encoded.startswith(b"#!"):
        first_line = encoded.splitlines()[0]
        try:
            shebang = first_line[2:].decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise OpenEcologyHealthError(
                f"health command shebang is not strict UTF-8: {name}"
            ) from error
        tokens = shebang.split()
        if len(tokens) != 1 or not Path(tokens[0]).is_absolute():
            raise OpenEcologyHealthError(
                f"health command requires one absolute shebang interpreter: {name}"
            )
        interpreter_bytes, interpreter = _pin_wrapper_input(
            Path(tokens[0]),
            field=f"health command shebang interpreter {name}",
            require_executable=True,
        )
        if interpreter_bytes.startswith(b"#!"):
            raise OpenEcologyHealthError(
                f"health command shebang interpreter must be native: {name}"
            )
        execution_mode = "validated_script_and_interpreter_snapshot_v1"
    else:
        execution_mode = "validated_native_snapshot_v1"
    executable_without_digest = {
        key: value for key, value in executable.items() if key != "authority_sha256"
    }
    authority: dict[str, object] = {
        "name": name,
        **executable_without_digest,
        "execution_mode": execution_mode,
        "interpreter": interpreter,
    }
    authority["authority_sha256"] = _payload_digest(authority)
    return encoded, authority, interpreter_bytes


def _validate_command_interpreter_authority(
    value: Mapping[str, object],
    *,
    expected_name: str,
    revalidate: bool,
) -> dict[str, object]:
    required = {
        "path",
        "device",
        "inode",
        "mode",
        "link_count",
        "size",
        "mtime_ns",
        "ctime_ns",
        "sha256",
        "authority_sha256",
    }
    if set(value) != required:
        raise OpenEcologyHealthError(
            f"health command interpreter authority schema drifted: {expected_name}"
        )
    unsigned = {key: value[key] for key in required if key != "authority_sha256"}
    if value.get("authority_sha256") != _payload_digest(unsigned):
        raise OpenEcologyHealthError(
            f"health command interpreter authority digest drifted: {expected_name}"
        )
    path = Path(str(value["path"]))
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise OpenEcologyHealthError(
            f"health command interpreter path is not canonical: {expected_name}"
        )
    normalized = dict(value)
    if revalidate:
        encoded, current = _pin_wrapper_input(
            path,
            field=f"health command shebang interpreter {expected_name}",
            require_executable=True,
        )
        if encoded.startswith(b"#!") or current != normalized:
            raise OpenEcologyHealthError(
                f"health command interpreter identity or SHA256 changed: "
                f"{expected_name}"
            )
    return normalized


def _command_path(
    command_authorities: Mapping[str, object],
    *,
    name: str,
) -> Path:
    authority = _mapping(
        command_authorities.get(name),
        field=f"{name} authority",
    )
    return Path(str(authority["path"]))


def _load_health_baseline(
    path: str | Path,
    *,
    identity: Mapping[str, object],
    expected_file_sha256: str,
) -> dict[str, object]:
    baseline_path = Path(path)
    expected_digest = _sha256(
        expected_file_sha256,
        field="expected health baseline SHA256",
    )
    if not baseline_path.is_absolute():
        raise OpenEcologyHealthError(
            "health baseline must be one absolute regular file"
        )
    try:
        descriptor = os.open(
            baseline_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise OpenEcologyHealthError(
            "health baseline cannot be opened without following links"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_size <= 0
            or before.st_size > OPEN_ECOLOGY_HEALTH_MAX_BASELINE_BYTES
        ):
            raise OpenEcologyHealthError(
                "health baseline must be private, owned, bounded, and non-hardlinked"
            )
        chunks: list[bytes] = []
        remaining = OPEN_ECOLOGY_HEALTH_MAX_BASELINE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or len(encoded) != before.st_size
        or hashlib.sha256(encoded).hexdigest() != expected_digest
    ):
        raise OpenEcologyHealthError(
            "health baseline changed or does not match external SHA256 authority"
        )
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: _raise_nonfinite(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenEcologyHealthError("health baseline is not strict JSON") from error
    baseline = dict(_mapping(payload, field="health baseline"))
    required = {
        "schema_version",
        "campaign_id",
        "campaign_root",
        "source_git_sha",
        "source_manifest_sha256",
        "boot_id",
        "swap_used_bytes",
        "xid_count",
        "oom_count",
        "command_authorities",
        "created_at_utc",
        "exact_digest",
    }
    if set(baseline) != required:
        raise OpenEcologyHealthError("health baseline keys are not exact")
    if (
        baseline["schema_version"] != OPEN_ECOLOGY_HEALTH_BASELINE_SCHEMA_VERSION
        or {
            key: baseline[key]
            for key in (
                "campaign_id",
                "campaign_root",
                "source_git_sha",
                "source_manifest_sha256",
            )
        }
        != _validate_identity(identity)
        or _payload_digest(
            {key: value for key, value in baseline.items() if key != "exact_digest"}
        )
        != baseline["exact_digest"]
    ):
        raise OpenEcologyHealthError("health baseline identity or digest drifted")
    _nonnegative_integer(baseline["swap_used_bytes"], field="baseline swap")
    _nonnegative_integer(baseline["xid_count"], field="baseline Xid count")
    _nonnegative_integer(baseline["oom_count"], field="baseline OOM count")
    _validate_command_authorities(
        baseline["command_authorities"],
        revalidate=True,
    )
    _parse_utc(baseline["created_at_utc"])
    return baseline


def _identity(
    *,
    campaign_root: Path,
    campaign_id: str,
    source_git_sha: str,
    source_manifest_sha256: str,
) -> dict[str, object]:
    return _validate_identity(
        {
            "campaign_id": campaign_id,
            "campaign_root": str(campaign_root),
            "source_git_sha": source_git_sha,
            "source_manifest_sha256": source_manifest_sha256,
        }
    )


def _validate_identity(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != {
        "campaign_id",
        "campaign_root",
        "source_git_sha",
        "source_manifest_sha256",
    }:
        raise OpenEcologyHealthError("health identity keys are not exact")
    campaign_id = value["campaign_id"]
    root = Path(str(value["campaign_root"]))
    source_git_sha = value["source_git_sha"]
    source_manifest_sha256 = value["source_manifest_sha256"]
    if (
        not isinstance(campaign_id, str)
        or _IDENTIFIER_RE.fullmatch(campaign_id) is None
    ):
        raise OpenEcologyHealthError("campaign id is malformed")
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve() != root
    ):
        raise OpenEcologyHealthError(
            "campaign root must be one canonical existing directory"
        )
    if (
        not isinstance(source_git_sha, str)
        or _GIT_SHA_RE.fullmatch(source_git_sha) is None
    ):
        raise OpenEcologyHealthError("source Git SHA is malformed")
    if (
        not isinstance(source_manifest_sha256, str)
        or _SHA256_RE.fullmatch(source_manifest_sha256) is None
    ):
        raise OpenEcologyHealthError("source manifest SHA256 is malformed")
    return {
        "campaign_id": campaign_id,
        "campaign_root": str(root),
        "source_git_sha": source_git_sha,
        "source_manifest_sha256": source_manifest_sha256,
    }


def _new_baseline_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise OpenEcologyHealthError(
            "new health baseline path must be absolute and absent"
        )
    if (
        not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve() != path.parent
    ):
        raise OpenEcologyHealthError(
            "health baseline parent must be one canonical existing directory"
        )
    return path


def _new_wrapper_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise OpenEcologyHealthError(
            "new health wrapper path must be absolute and absent"
        )
    if (
        not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve() != path.parent
    ):
        raise OpenEcologyHealthError(
            "health wrapper parent must be one canonical existing directory"
        )
    return path


def _pin_wrapper_input(
    path: Path,
    *,
    field: str,
    require_executable: bool,
) -> tuple[bytes, dict[str, object]]:
    if not path.is_absolute():
        raise OpenEcologyHealthError(f"{field} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise OpenEcologyHealthError(f"{field} cannot be resolved") from error
    try:
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise OpenEcologyHealthError(f"{field} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > OPEN_ECOLOGY_HEALTH_MAX_EXECUTABLE_BYTES
            or mode & 0o022
            or (require_executable and mode & 0o111 == 0)
        ):
            raise OpenEcologyHealthError(f"{field} violates its regular-file contract")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > OPEN_ECOLOGY_HEALTH_MAX_EXECUTABLE_BYTES:
                raise OpenEcologyHealthError(f"{field} exceeds its size bound")
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or total != before.st_size:
        raise OpenEcologyHealthError(f"{field} changed while hashing")
    authority: dict[str, object] = {
        "path": str(resolved),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": mode,
        "link_count": before.st_nlink,
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "sha256": digest.hexdigest(),
    }
    authority["authority_sha256"] = _payload_digest(authority)
    return b"".join(chunks), authority


def _phase(value: object) -> str:
    if value not in {"before_advance", "frontier_quiescent"}:
        raise OpenEcologyHealthError("health phase is missing or unsupported")
    return str(value)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OpenEcologyHealthError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise OpenEcologyHealthError(f"{field} must be an array")
    return value


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, str) and _INTEGER_RE.fullmatch(value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologyHealthError(f"{field} must be a non-negative integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    parsed = _nonnegative_integer(value, field=field)
    if parsed <= 0:
        raise OpenEcologyHealthError(f"{field} must be positive")
    return parsed


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OpenEcologyHealthError(
            f"{field} must be 64 lowercase hexadecimal characters"
        )
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise OpenEcologyHealthError("health baseline contains duplicate keys")
        payload[key] = value
    return payload


def _raise_nonfinite(value: str) -> object:
    raise OpenEcologyHealthError(f"health baseline contains non-finite {value}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _payload_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(value))).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise OpenEcologyHealthError("baseline time must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise OpenEcologyHealthError("baseline time is malformed") from error
    if parsed.tzinfo != timezone.utc:
        raise OpenEcologyHealthError("baseline time must be UTC")
    return parsed


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OpenEcologyHealthError("health baseline write made no progress")
        offset += written


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpenEcologyHealthError as error:
        print(f"open-ecology health error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
