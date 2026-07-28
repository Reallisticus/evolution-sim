from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from evolution_sim.cli.open_ecology_health import (
    OPEN_ECOLOGY_FATAL_WORKER_MARKER_NAME,
    OpenEcologyHealthError,
    build_health_probe_wrapper,
    collect_host_observations,
    main as health_main,
    _parse_temperature_report,
    _pin_command_executable,
    _read_kernel_event_count,
    _read_gpus,
    _run_bounded_command,
    _run_command,
    check_campaign_health,
    initialize_health_baseline,
)


_SOURCE_COMMIT = "a" * 40
_SOURCE_MANIFEST = "b" * 64


class OpenEcologyHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.campaign_root = self.root / "campaign"
        self.campaign_root.mkdir()
        self.baseline_path = self.root / "health-baseline.json"
        self.identity = {
            "campaign_id": "health-test",
            "campaign_root": str(self.campaign_root),
            "source_git_sha": _SOURCE_COMMIT,
            "source_manifest_sha256": _SOURCE_MANIFEST,
        }
        self.nvidia_smi = self.root / "nvidia-smi"
        self.journalctl = self.root / "journalctl"
        for executable in (self.nvidia_smi, self.journalctl):
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o700)
        self.command_authorities = {
            "nvidia-smi": _pin_command_executable(
                self.nvidia_smi,
                name="nvidia-smi",
            ),
            "journalctl": _pin_command_executable(
                self.journalctl,
                name="journalctl",
            ),
        }
        discovery = patch(
            "evolution_sim.cli.open_ecology_health._discover_command_authorities",
            return_value=self.command_authorities,
        )
        discovery.start()
        self.addCleanup(discovery.stop)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_public_host_observer_delegates_to_health_collector(self) -> None:
        observations = self._observations()
        with patch(
            "evolution_sim.cli.open_ecology_health._collect_host_observations",
            return_value=observations,
        ) as collect:
            self.assertIs(
                collect_host_observations(self.campaign_root),
                observations,
            )
        collect.assert_called_once_with(self.campaign_root)

    def test_immutable_baseline_and_healthy_snapshot_bind_exact_frontier(self) -> None:
        observations = self._observations()
        with patch(
            "evolution_sim.cli.open_ecology_health._collect_host_observations",
            return_value=observations,
        ):
            baseline = initialize_health_baseline(
                self.baseline_path,
                identity=self.identity,
            )
            snapshot = check_campaign_health(
                self.baseline_path,
                identity=self.identity,
                phase="before_advance",
                frontier_tick="5000",
                expected_baseline_sha256=self._baseline_sha256(),
            )

        self.assertEqual(self.baseline_path.stat().st_mode & 0o777, 0o400)
        self.assertEqual(baseline["campaign_id"], "health-test")
        self.assertEqual(snapshot["phase"], "before_advance")
        self.assertEqual(snapshot["frontier_tick"], 5_000)
        self.assertTrue(snapshot["healthy"])
        self.assertEqual(snapshot["blockers"], [])
        self.assertEqual(snapshot["host"], observations)
        self.assertEqual(snapshot["baseline_exact_digest"], baseline["exact_digest"])

    def test_every_resource_drift_fails_health_closed(self) -> None:
        baseline_observations = self._observations()
        with patch(
            "evolution_sim.cli.open_ecology_health._collect_host_observations",
            return_value=baseline_observations,
        ):
            initialize_health_baseline(
                self.baseline_path,
                identity=self.identity,
            )
        drifted = self._observations()
        drifted["boot_id"] = "12345678-1234-1234-1234-123456789abc"
        drifted["xid_count"] = 1
        drifted["oom_count"] = 1
        drifted["memory"] = {
            **drifted["memory"],
            "ram_used_share": 0.81,
            "swap_used_bytes": 1,
        }
        drifted["filesystem"] = {
            **drifted["filesystem"],
            "free_bytes": 99 * 1024**3,
        }
        drifted["gpus"] = [
            {
                **drifted["gpus"][0],
                "memory_used_share": 0.80,
                "temperature_c": 96,
            }
        ]
        with patch(
            "evolution_sim.cli.open_ecology_health._collect_host_observations",
            return_value=drifted,
        ):
            snapshot = check_campaign_health(
                self.baseline_path,
                identity=self.identity,
                phase="frontier_quiescent",
                frontier_tick=10_000,
                expected_baseline_sha256=self._baseline_sha256(),
            )

        self.assertFalse(snapshot["healthy"])
        self.assertEqual(
            snapshot["blockers"],
            [
                "host_boot_id_changed",
                "swap_usage_increased",
                "host_ram_share_at_or_above_0.80",
                "campaign_filesystem_below_free_space_floor",
                "nvidia_xid_count_changed",
                "kernel_oom_count_changed",
                "gpu_0_memory_share_at_or_above_0.80",
                "gpu_0_at_or_above_slowdown_temperature",
            ],
        )

    def test_tamper_hardlink_and_identity_drift_are_rejected(self) -> None:
        with patch(
            "evolution_sim.cli.open_ecology_health._collect_host_observations",
            return_value=self._observations(),
        ):
            initialize_health_baseline(
                self.baseline_path,
                identity=self.identity,
            )
        os.chmod(self.baseline_path, 0o600)
        payload = json.loads(self.baseline_path.read_bytes())
        payload["swap_used_bytes"] = 99
        self.baseline_path.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(self.baseline_path, 0o400)
        with self.assertRaisesRegex(OpenEcologyHealthError, "digest drifted"):
            check_campaign_health(
                self.baseline_path,
                identity=self.identity,
                phase="before_advance",
                frontier_tick=5_000,
                expected_baseline_sha256=self._baseline_sha256(),
            )

        self.baseline_path.unlink()
        with patch(
            "evolution_sim.cli.open_ecology_health._collect_host_observations",
            return_value=self._observations(),
        ):
            initialize_health_baseline(
                self.baseline_path,
                identity=self.identity,
            )
        hardlink = self.root / "health-hardlink.json"
        os.link(self.baseline_path, hardlink)
        with self.assertRaisesRegex(OpenEcologyHealthError, "non-hardlinked"):
            check_campaign_health(
                self.baseline_path,
                identity=self.identity,
                phase="before_advance",
                frontier_tick=5_000,
                expected_baseline_sha256=self._baseline_sha256(),
            )
        hardlink.unlink()
        with self.assertRaisesRegex(OpenEcologyHealthError, "identity or digest"):
            check_campaign_health(
                self.baseline_path,
                identity={**self.identity, "campaign_id": "other-campaign"},
                phase="before_advance",
                frontier_tick=5_000,
                expected_baseline_sha256=self._baseline_sha256(),
            )

    def test_parses_nvidia_current_and_slowdown_temperatures(self) -> None:
        report = """
GPU 00000000:01:00.0
    Temperature
        GPU Current Temp                               : 42 C
        GPU T.Limit Temp                               : N/A
        GPU Shutdown Temp                              : 101 C
        GPU Slowdown Temp                              : 96 C
GPU 00000000:02:00.0
    Temperature
        GPU Current Temp                               : 50 C
        GPU Slowdown Temp                              : 92 C
"""
        self.assertEqual(
            _parse_temperature_report(report),
            {0: (42, 96), 1: (50, 92)},
        )
        with self.assertRaisesRegex(
            OpenEcologyHealthError,
            "current and slowdown",
        ):
            _parse_temperature_report("GPU 00000000:01:00.0\nGPU Current Temp : 42 C\n")

    def test_gpu_at_slowdown_temperature_remains_observable(self) -> None:
        inventory = "0, NVIDIA Test GPU, 8192, 4096, 96, 75\n"
        temperature_report = """
GPU 00000000:01:00.0
    Temperature
        GPU Current Temp                               : 96 C
        GPU Slowdown Temp                              : 96 C
"""
        with patch(
            "evolution_sim.cli.open_ecology_health._run_command",
            side_effect=(inventory, temperature_report),
        ):
            rows = _read_gpus(self.command_authorities)

        self.assertEqual(rows[0]["temperature_c"], 96)
        self.assertEqual(rows[0]["slowdown_temperature_c"], 96)

    def test_fatal_worker_marker_forces_unhealthy_snapshot(self) -> None:
        with patch(
            "evolution_sim.cli.open_ecology_health._collect_host_observations",
            return_value=self._observations(),
        ):
            initialize_health_baseline(
                self.baseline_path,
                identity=self.identity,
            )
        marker = self.campaign_root / OPEN_ECOLOGY_FATAL_WORKER_MARKER_NAME
        marker.write_text("fatal\n", encoding="ascii")
        marker.chmod(0o400)

        with patch(
            "evolution_sim.cli.open_ecology_health._collect_host_observations",
            return_value=self._observations(),
        ):
            snapshot = check_campaign_health(
                self.baseline_path,
                identity=self.identity,
                phase="before_advance",
                frontier_tick=5_000,
                expected_baseline_sha256=self._baseline_sha256(),
            )

        self.assertFalse(snapshot["healthy"])
        self.assertIn("fatal_worker_pid_marker_present", snapshot["blockers"])

    def test_journal_permission_stderr_cannot_be_misread_as_zero_events(self) -> None:
        with (
            patch(
                "evolution_sim.cli.open_ecology_health._run_bounded_command",
                return_value=(
                    b"",
                    b"Failed to open journal: Permission denied\n",
                    1,
                ),
            ),
            self.assertRaisesRegex(
                OpenEcologyHealthError,
                "stderr and cannot prove success",
            ),
        ):
            _read_kernel_event_count("Xid")

        with patch(
            "evolution_sim.cli.open_ecology_health._run_bounded_command",
            return_value=(b"", b"", 1),
        ):
            self.assertEqual(_read_kernel_event_count("Xid"), 0)

    def test_nested_health_command_output_is_streaming_bounded(self) -> None:
        noisy = self.root / "noisy-health-command"
        noisy.write_text(
            "#!/bin/sh\nprintf '0123456789abcdef0123456789abcdef'\n",
            encoding="utf-8",
        )
        noisy.chmod(0o700)
        authority = _pin_command_executable(noisy, name="journalctl")

        with (
            patch(
                "evolution_sim.cli.open_ecology_health."
                "OPEN_ECOLOGY_HEALTH_MAX_COMMAND_BYTES",
                16,
            ),
            self.assertRaisesRegex(OpenEcologyHealthError, "output exceeds"),
        ):
            _run_command(
                (str(noisy),),
                accepted_returncodes=(0,),
                command_authority=authority,
            )

    def test_nested_native_and_script_commands_never_execute_restore_attack_bytes(
        self,
    ) -> None:
        real_run = _run_bounded_command
        for label, benign_payload, arguments in (
            (
                "native",
                Path(sys.executable).resolve().read_bytes(),
                ("-c", "print('benign')"),
            ),
            ("script", b"#!/bin/sh\nprintf 'benign\\n'\n", ()),
        ):
            with self.subTest(label=label):
                command_path = self.root / f"restore-attack-{label}"
                command_path.write_bytes(benign_payload)
                command_path.chmod(0o500)
                authority = _pin_command_executable(
                    command_path,
                    name="journalctl",
                )
                original_backup = self.root / f"restore-attack-{label}.original"
                displaced_hostile = self.root / f"restore-attack-{label}.hostile"
                hostile_marker = self.root / f"restore-attack-{label}.executed"
                hostile = self.root / f"restore-attack-{label}.replacement"
                hostile.write_text(
                    f"#!/bin/sh\ntouch {str(hostile_marker)!r}\nprintf 'benign\\n'\n",
                    encoding="utf-8",
                )
                hostile.chmod(0o500)

                def replace_launch_restore(command, **kwargs):
                    os.replace(command_path, original_backup)
                    os.replace(hostile, command_path)
                    try:
                        return real_run(command, **kwargs)
                    finally:
                        os.replace(command_path, displaced_hostile)
                        os.replace(original_backup, command_path)

                with (
                    patch(
                        "evolution_sim.cli.open_ecology_health._run_bounded_command",
                        side_effect=replace_launch_restore,
                    ),
                    self.assertRaisesRegex(
                        OpenEcologyHealthError,
                        "changed during execution",
                    ),
                ):
                    _run_command(
                        (str(command_path), *arguments),
                        command_authority=authority,
                    )
                self.assertFalse(
                    hostile_marker.exists(),
                    f"unchecked {label} replacement executed before post-validation",
                )

    def test_nested_script_interpreter_executes_validated_snapshot(self) -> None:
        interpreter_path = self.root / "nested-interpreter"
        shutil.copyfile(Path(sys.executable).resolve(), interpreter_path)
        interpreter_path.chmod(0o500)
        command_path = self.root / "nested-script"
        command_path.write_text(
            f"#!{interpreter_path}\nprint('benign')\n",
            encoding="utf-8",
        )
        command_path.chmod(0o500)
        authority = _pin_command_executable(command_path, name="journalctl")
        hostile_marker = self.root / "nested-hostile-interpreter.executed"
        hostile_interpreter = self.root / "nested-hostile-interpreter"
        hostile_interpreter.write_text(
            f"#!/bin/sh\ntouch {str(hostile_marker)!r}\nprintf 'benign\\n'\n",
            encoding="utf-8",
        )
        hostile_interpreter.chmod(0o500)
        original_backup = self.root / "nested-interpreter.original"
        displaced_hostile = self.root / "nested-interpreter.hostile"
        real_popen = subprocess.Popen

        def replace_launch_restore(*args, **kwargs):
            os.replace(interpreter_path, original_backup)
            os.replace(hostile_interpreter, interpreter_path)
            try:
                return real_popen(*args, **kwargs)
            finally:
                os.replace(interpreter_path, displaced_hostile)
                os.replace(original_backup, interpreter_path)

        with (
            patch(
                "evolution_sim.cli.open_ecology_health.subprocess.Popen",
                side_effect=replace_launch_restore,
            ),
            self.assertRaisesRegex(
                OpenEcologyHealthError,
                "changed during execution",
            ),
        ):
            _run_command(
                (str(command_path),),
                command_authority=authority,
            )
        self.assertFalse(
            hostile_marker.exists(),
            "unchecked interpreter replacement executed before post-validation",
        )

    def test_health_command_breach_kills_descendant_after_leader_exit(self) -> None:
        child_pid_path = self.root / "health-child.pid"
        child_body = (
            "import os,signal,time\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "while True:\n"
            " os.write(1,b'x'*4096)\n"
            " time.sleep(0.001)\n"
        )
        parent_code = (
            "import pathlib,subprocess,sys\n"
            f"child=subprocess.Popen([sys.executable,'-c',{child_body!r}])\n"
            f"pathlib.Path({str(child_pid_path)!r}).write_text("
            "str(child.pid),encoding='ascii')\n"
        )
        with (
            patch(
                "evolution_sim.cli.open_ecology_health."
                "OPEN_ECOLOGY_HEALTH_MAX_COMMAND_BYTES",
                8192,
            ),
            self.assertRaisesRegex(OpenEcologyHealthError, "output exceeds"),
        ):
            _run_bounded_command((sys.executable, "-c", parent_code))
        _assert_process_gone(self, child_pid_path)

    def test_health_command_setup_failures_close_the_process_group(self) -> None:
        for setup_step in (
            "selector",
            "fileno",
            "set_blocking",
            "register",
        ):
            with self.subTest(setup_step=setup_step):
                self._assert_health_command_setup_failure_closes_group(setup_step)

    def test_external_baseline_and_command_authorities_reject_replacement(
        self,
    ) -> None:
        with patch(
            "evolution_sim.cli.open_ecology_health._collect_host_observations",
            return_value=self._observations(),
        ):
            initialize_health_baseline(
                self.baseline_path,
                identity=self.identity,
            )
        expected = self._baseline_sha256()
        payload = json.loads(self.baseline_path.read_bytes())
        unsigned = {
            key: value for key, value in payload.items() if key != "exact_digest"
        }
        unsigned["swap_used_bytes"] = 7
        from evolution_sim.cli.open_ecology_health import _payload_digest

        replacement_payload = {
            **unsigned,
            "exact_digest": _payload_digest(unsigned),
        }
        replacement = self.root / "replacement-baseline.json"
        replacement.write_text(
            json.dumps(replacement_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        replacement.chmod(0o400)
        os.replace(replacement, self.baseline_path)
        with self.assertRaisesRegex(
            OpenEcologyHealthError,
            "external SHA256 authority",
        ):
            check_campaign_health(
                self.baseline_path,
                identity=self.identity,
                phase="before_advance",
                frontier_tick=5_000,
                expected_baseline_sha256=expected,
            )

        self.baseline_path.unlink()
        with patch(
            "evolution_sim.cli.open_ecology_health._collect_host_observations",
            return_value=self._observations(),
        ):
            initialize_health_baseline(
                self.baseline_path,
                identity=self.identity,
            )
        hostile_journal = self.root / "hostile-journalctl"
        hostile_journal.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        hostile_journal.chmod(0o700)
        os.replace(hostile_journal, self.journalctl)
        with self.assertRaisesRegex(
            OpenEcologyHealthError,
            "identity or SHA256 changed",
        ):
            check_campaign_health(
                self.baseline_path,
                identity=self.identity,
                phase="before_advance",
                frontier_tick=5_000,
                expected_baseline_sha256=self._baseline_sha256(),
            )

    def test_init_digest_has_no_reopen_and_generated_wrapper_is_hermetic(
        self,
    ) -> None:
        cli_baseline = self.root / "cli-baseline.json"
        output = io.StringIO()
        with (
            patch(
                "evolution_sim.cli.open_ecology_health._collect_host_observations",
                return_value=self._observations(),
            ),
            patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("baseline path was reopened"),
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(
                health_main(
                    (
                        "init",
                        "--baseline",
                        str(cli_baseline),
                        "--campaign-root",
                        str(self.campaign_root),
                        "--campaign-id",
                        "health-test",
                        "--source-git-sha",
                        _SOURCE_COMMIT,
                        "--source-manifest-sha256",
                        _SOURCE_MANIFEST,
                    )
                ),
                0,
            )
        init_result = json.loads(output.getvalue())
        expected_baseline_sha256 = init_result["health_baseline_sha256"]
        self.assertEqual(
            expected_baseline_sha256,
            __import__("hashlib").sha256(cli_baseline.read_bytes()).hexdigest(),
        )

        import evolution_sim.cli.open_ecology_health as health_module

        original_source = Path(health_module.__file__).read_bytes()
        terminal = original_source.rfind(b'\n\nif __name__ == "__main__":')
        self.assertGreater(terminal, 0)
        override = (
            "\n\ndef _collect_host_observations("
            "campaign_root, *, command_authorities=None):\n"
            f"    return {self._observations()!r}\n"
        ).encode("utf-8")
        health_source = self.root / "sealed-health-source.py"
        health_source.write_bytes(
            original_source[:terminal] + override + original_source[terminal:]
        )
        health_source.chmod(0o400)
        python_copy = self.root / "sealed-python"
        shutil.copy2(Path(sys.executable).resolve(), python_copy)
        python_copy.chmod(0o500)
        wrapper = self.root / "health-wrapper"
        result = build_health_probe_wrapper(
            wrapper,
            python_executable=python_copy,
            health_script=health_source,
            expected_health_script_sha256=__import__("hashlib")
            .sha256(health_source.read_bytes())
            .hexdigest(),
            baseline=cli_baseline,
            expected_baseline_sha256=expected_baseline_sha256,
            identity=self.identity,
        )
        self.assertEqual(wrapper.stat().st_mode & 0o777, 0o500)
        environment = {
            "EVOSIM_OPEN_ECOLOGY_HEALTH_PHASE": "before_advance",
            "EVOSIM_OPEN_ECOLOGY_FRONTIER_TICK": "5000",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
        completed = subprocess.run(
            (str(wrapper),),
            capture_output=True,
            check=False,
            env=environment,
            timeout=10.0,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout)["baseline_file_sha256"],
            expected_baseline_sha256,
        )

        health_source.chmod(0o600)
        health_source.write_text("raise SystemExit(99)\n", encoding="utf-8")
        completed_after_source_swap = subprocess.run(
            (str(wrapper),),
            capture_output=True,
            check=False,
            env=environment,
            timeout=10.0,
        )
        self.assertEqual(
            completed_after_source_swap.returncode,
            0,
            completed_after_source_swap.stderr,
        )

        hostile_python = self.root / "hostile-python"
        true_executable = shutil.which("true")
        self.assertIsNotNone(true_executable)
        assert true_executable is not None
        shutil.copyfile(Path(true_executable).resolve(strict=True), hostile_python)
        hostile_python.chmod(0o500)
        os.replace(hostile_python, python_copy)
        from evolution_sim.cli.open_ecology_campaign import _command_health_probe
        from evolution_sim.mind.open_ecology_campaign_contract import (
            OpenEcologyCampaignCoordinatorError,
        )

        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "embedded SHA256",
        ):
            _command_health_probe(
                (str(wrapper),),
                expected_baseline_sha256=expected_baseline_sha256,
                expected_probe_file_sha256=result["wrapper_sha256"],
            )
        with self.assertRaisesRegex(OpenEcologyHealthError, "absent"):
            build_health_probe_wrapper(
                wrapper,
                python_executable=python_copy,
                health_script=health_source,
                expected_health_script_sha256=__import__("hashlib")
                .sha256(health_source.read_bytes())
                .hexdigest(),
                baseline=cli_baseline,
                expected_baseline_sha256=expected_baseline_sha256,
                identity=self.identity,
            )

    def _assert_health_command_setup_failure_closes_group(
        self,
        setup_step: str,
    ) -> None:
        import evolution_sim.cli.open_ecology_health as health

        child_pid_path = self.root / f"setup-{setup_step}-child.pid"
        child_body = (
            "import signal,time\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "while True: time.sleep(1)\n"
        )
        parent_code = (
            "import pathlib,subprocess,sys,time\n"
            f"child=subprocess.Popen([sys.executable,'-c',{child_body!r}])\n"
            f"pathlib.Path({str(child_pid_path)!r}).write_text("
            "str(child.pid),encoding='ascii')\n"
            "while True: time.sleep(1)\n"
        )
        failure = RuntimeError(f"simulated {setup_step} setup failure")
        started: list[subprocess.Popen[bytes]] = []
        real_popen = subprocess.Popen
        real_selector = health.selectors.DefaultSelector

        class FailingFileno:
            def __init__(self, stream):
                self._stream = stream

            def fileno(self):
                raise failure

            def close(self):
                self._stream.close()

        class FailingRegisterSelector:
            def __init__(self):
                self._selector = real_selector()

            def register(self, *args, **kwargs):
                raise failure

            def close(self):
                self._selector.close()

        def start_command(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            deadline = time.monotonic() + 2.0
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not child_pid_path.exists():
                raise AssertionError("health-command descendant did not start")
            if setup_step == "fileno":
                assert process.stdout is not None
                process.stdout = FailingFileno(process.stdout)
            return process

        try:
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        health.subprocess,
                        "Popen",
                        side_effect=start_command,
                    )
                )
                if setup_step == "selector":
                    stack.enter_context(
                        patch.object(
                            health.selectors,
                            "DefaultSelector",
                            side_effect=failure,
                        )
                    )
                elif setup_step == "set_blocking":
                    stack.enter_context(
                        patch.object(
                            health.os,
                            "set_blocking",
                            side_effect=failure,
                        )
                    )
                elif setup_step == "register":
                    stack.enter_context(
                        patch.object(
                            health.selectors,
                            "DefaultSelector",
                            side_effect=FailingRegisterSelector,
                        )
                    )
                with self.assertRaises(RuntimeError) as caught:
                    _run_bounded_command(
                        (sys.executable, "-c", parent_code),
                    )
            self.assertIs(caught.exception, failure)
            self.assertEqual(len(started), 1)
            self.assertIsNotNone(started[0].returncode)
            _assert_process_gone(self, child_pid_path)
        finally:
            for process in started:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2.0)
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    def _baseline_sha256(self) -> str:
        import hashlib

        return hashlib.sha256(self.baseline_path.read_bytes()).hexdigest()

    @staticmethod
    def _observations() -> dict[str, object]:
        return {
            "boot_id": "00000000-0000-0000-0000-000000000000",
            "memory": {
                "ram_total_bytes": 64 * 1024**3,
                "ram_available_bytes": 48 * 1024**3,
                "ram_used_share": 0.25,
                "swap_total_bytes": 8 * 1024**3,
                "swap_used_bytes": 0,
            },
            "filesystem": {
                "capacity_bytes": 2 * 1024**4,
                "free_bytes": 1 * 1024**4,
                "required_free_bytes": (2 * 1024**4) // 5,
            },
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 4070 SUPER",
                    "memory_total_bytes": 12 * 1024**3,
                    "memory_used_bytes": 0,
                    "memory_used_share": 0.0,
                    "temperature_c": 42,
                    "slowdown_temperature_c": 96,
                    "utilization_percent": 0,
                }
            ],
            "xid_count": 0,
            "oom_count": 0,
            "load_average": [0.0, 0.0, 0.0],
            "process_id": 123,
        }


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
        f"health-command descendant survived cleanup: {state}",
    )


if __name__ == "__main__":
    unittest.main()
