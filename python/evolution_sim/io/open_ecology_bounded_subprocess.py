"""POSIX subprocess-group cleanup without leader PID reuse races."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any


class OpenEcologyProcessGroupError(RuntimeError):
    """Raised when a bounded subprocess group cannot be proven closed."""


def leader_exit_observed_without_reaping(
    process: subprocess.Popen[Any],
) -> bool:
    """Return whether the leader exited while retaining its PID as a zombie."""

    required = ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT", "waitid")
    if any(not hasattr(os, name) for name in required):
        raise OpenEcologyProcessGroupError(
            "waitid WNOWAIT is unavailable for safe process-group cleanup"
        )
    options = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        try:
            return os.waitid(os.P_PID, process.pid, options) is not None
        except InterruptedError:
            continue
        except ChildProcessError as error:
            raise OpenEcologyProcessGroupError(
                "subprocess leader was reaped before group cleanup"
            ) from error


def wait_for_leader_exit_without_reaping(
    process: subprocess.Popen[Any],
    *,
    deadline: float,
) -> None:
    """Wait for leader exit while retaining its PID as an unreaped zombie."""

    while True:
        if leader_exit_observed_without_reaping(process):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("subprocess leader did not exit before deadline")
        time.sleep(min(0.01, remaining))


def terminate_process_group_before_reap(
    process: subprocess.Popen[Any],
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


__all__ = [
    "OpenEcologyProcessGroupError",
    "leader_exit_observed_without_reaping",
    "terminate_process_group_before_reap",
    "wait_for_leader_exit_without_reaping",
]
