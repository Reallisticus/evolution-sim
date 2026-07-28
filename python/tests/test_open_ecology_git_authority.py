from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import evolution_sim.io.open_ecology_git_authority as git_authority


class OpenEcologyGitAuthorityTests(unittest.TestCase):
    def test_usr_local_shadow_cannot_replace_explicit_system_git(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hostile = Path(temporary) / "git"
            hostile.write_text("#!/bin/sh\nprintf hostile\n", encoding="utf-8")
            hostile.chmod(0o700)
            with (
                patch.dict(os.environ, {"PATH": temporary}),
                patch.object(
                    git_authority,
                    "OPEN_ECOLOGY_GIT_COMMAND_PATH",
                    temporary,
                ),
            ):
                authority = git_authority.discover_pinned_git_executable()
                head = git_authority.run_pinned_git(
                    authority,
                    repository_root=Path(__file__).resolve().parents[2],
                    arguments=("rev-parse", "HEAD"),
                )
        self.assertTrue(Path(authority.path).is_absolute())
        self.assertNotEqual(Path(authority.path), hostile)
        self.assertEqual(
            Path(authority.path),
            Path(git_authority.OPEN_ECOLOGY_SYSTEM_GIT_PATH).resolve(strict=True),
        )
        self.assertRegex(head, r"^[0-9a-f]{40}$")

    def test_discovery_rejects_non_root_owned_explicit_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hostile = Path(temporary) / "git"
            hostile.write_text("#!/bin/sh\nprintf hostile\n", encoding="utf-8")
            hostile.chmod(0o700)
            with (
                patch.object(
                    git_authority,
                    "OPEN_ECOLOGY_SYSTEM_GIT_PATH",
                    hostile,
                ),
                self.assertRaisesRegex(
                    git_authority.OpenEcologyGitAuthorityError,
                    "root-owned and non-writable",
                ),
            ):
                git_authority.discover_pinned_git_executable()

    def test_pinned_git_rejects_replacement_during_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "git"
            executable.write_text("#!/bin/sh\nprintf original\n", encoding="utf-8")
            executable.chmod(0o700)
            authority = git_authority.pin_git_executable(executable)
            replacement = root / "replacement"
            replacement.write_text("#!/bin/sh\nprintf replacement\n", encoding="utf-8")
            replacement.chmod(0o700)

            def replace_during_command(
                _command: object,
            ) -> tuple[bytes, bytes, int]:
                os.replace(replacement, executable)
                return b"original\n", b"", 0

            with (
                patch.object(
                    git_authority,
                    "_run_bounded_command",
                    side_effect=replace_during_command,
                ),
                self.assertRaisesRegex(
                    git_authority.OpenEcologyGitAuthorityError,
                    "identity|SHA256",
                ),
            ):
                git_authority.run_pinned_git(
                    authority,
                    repository_root=root,
                    arguments=("rev-parse", "HEAD"),
                )

    def test_output_breach_kills_descendant_after_leader_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            child_pid_path = Path(temporary) / "child.pid"
            parent_code = _leader_exit_code(
                child_pid_path,
                child_body=(
                    "import os,signal,time\n"
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                    "while True:\n"
                    " os.write(1,b'x'*4096)\n"
                    " time.sleep(0.001)\n"
                ),
            )
            with (
                patch.object(
                    git_authority,
                    "OPEN_ECOLOGY_GIT_OUTPUT_MAX_BYTES",
                    8192,
                ),
                self.assertRaisesRegex(
                    git_authority.OpenEcologyGitAuthorityError,
                    "output exceeded",
                ),
            ):
                git_authority._run_bounded_command((sys.executable, "-c", parent_code))
            _assert_process_gone(self, child_pid_path)

    def test_timeout_kills_quiet_descendant_after_leader_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            child_pid_path = Path(temporary) / "child.pid"
            parent_code = _leader_exit_code(
                child_pid_path,
                child_body=(
                    "import signal,time\n"
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                    "time.sleep(60)\n"
                ),
            )
            with (
                patch.object(
                    git_authority,
                    "OPEN_ECOLOGY_GIT_COMMAND_TIMEOUT_SECONDS",
                    0.1,
                ),
                self.assertRaisesRegex(
                    git_authority.OpenEcologyGitAuthorityError,
                    "timed out",
                ),
            ):
                git_authority._run_bounded_command((sys.executable, "-c", parent_code))
            _assert_process_gone(self, child_pid_path)

    def test_successful_leader_cannot_leave_stubborn_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            child_pid_path = Path(temporary) / "child.pid"
            child_body = (
                "import signal,time\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                "time.sleep(60)\n"
            )
            parent_code = (
                "import pathlib,subprocess,sys\n"
                f"child=subprocess.Popen([sys.executable,'-c',{child_body!r}],"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
                f"pathlib.Path({str(child_pid_path)!r}).write_text("
                "str(child.pid),encoding='ascii')\n"
                "print('leader-complete')\n"
            )
            stdout, stderr, returncode = git_authority._run_bounded_command(
                (sys.executable, "-c", parent_code)
            )
            self.assertEqual(returncode, 0)
            self.assertEqual(stdout, b"leader-complete\n")
            self.assertEqual(stderr, b"")
            _assert_process_gone(self, child_pid_path)

    def test_external_digest_and_safe_mode_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "git"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o700)
            digest = hashlib.sha256(executable.read_bytes()).hexdigest()
            authority = git_authority.pin_git_executable(
                executable,
                expected_sha256=digest,
            )
            self.assertEqual(authority.sha256, digest)
            with self.assertRaisesRegex(
                git_authority.OpenEcologyGitAuthorityError,
                "SHA256",
            ):
                git_authority.pin_git_executable(
                    executable,
                    expected_sha256="0" * 64,
                )
            executable.chmod(0o722)
            with self.assertRaisesRegex(
                git_authority.OpenEcologyGitAuthorityError,
                "immutable executable",
            ):
                git_authority.pin_git_executable(executable)

    def test_mutating_git_commands_are_not_in_authority(self) -> None:
        authority = git_authority.discover_pinned_git_executable()
        with self.assertRaisesRegex(
            git_authority.OpenEcologyGitAuthorityError,
            "read-only source queries",
        ):
            git_authority.run_pinned_git(
                authority,
                repository_root=Path(__file__).resolve().parents[2],
                arguments=("clean", "-fd"),
            )


def _leader_exit_code(child_pid_path: Path, *, child_body: str) -> str:
    return (
        "import pathlib,subprocess,sys\n"
        f"body={child_body!r}\n"
        "child=subprocess.Popen([sys.executable,'-c',body])\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text("
        "str(child.pid),encoding='ascii')\n"
    )


def _assert_process_gone(
    case: unittest.TestCase,
    child_pid_path: Path,
) -> None:
    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 3.0
    state = ""
    while time.monotonic() < deadline:
        completed = subprocess.run(
            ("ps", "-p", str(child_pid), "-o", "stat="),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        state = completed.stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.05)
    case.assertTrue(
        not state or state.startswith("Z"),
        f"descendant survived process-group cleanup: {state}",
    )


if __name__ == "__main__":
    unittest.main()
