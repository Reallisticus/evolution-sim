from __future__ import annotations

import hashlib
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
from contextlib import ExitStack, redirect_stdout
from unittest import mock

from evolution_sim.cli.open_ecology_campaign import (
    OPEN_ECOLOGY_CAMPAIGN_LAUNCH_SPEC_SCHEMA_VERSION,
    _command_health_probe,
    _load_launch_spec,
    _process_worker_config,
    _run_bounded_health_probe,
    main,
)
from evolution_sim.io import open_ecology_campaign_storage as storage
from evolution_sim.io.open_ecology_campaign_storage import (
    ACTIVE_CAMPAIGN_BUDGET_BYTES,
    DEFAULT_ACTIVE_FILESYSTEM_FREE_BYTES,
    DEFAULT_MAX_ENTRIES,
    DEFAULT_REMOTE_FREE_BYTES,
    CampaignStorageError,
)
from evolution_sim.mind.open_ecology_campaign_coordinator import (
    OpenEcologyCampaignCoordinatorError,
)
from evolution_sim.mind.open_ecology_process_workers import (
    local_process_worker_host_identity,
)

_BASELINE_SHA256 = "d" * 64


def _sealed_shell_probe(body: str) -> str:
    interpreter = Path("/bin/sh")
    interpreter_sha256 = hashlib.sha256(interpreter.resolve().read_bytes()).hexdigest()
    return f"#!{interpreter}\n# evosim_shebang_sha256={interpreter_sha256}\n{body}"


class OpenEcologyCampaignCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_launch_spec_requires_explicit_process_worker_authority(self) -> None:
        single_payload = self._launch_payload(process_workers=None)
        single = self._write_json("single.json", single_payload)
        loaded_single = _load_launch_spec(
            single,
            expected_sha256=hashlib.sha256(single.read_bytes()).hexdigest(),
        )
        self.assertIsNone(_process_worker_config(loaded_single, worker_count=1))

        multi_payload = self._launch_payload(
            process_workers=self._process_worker_payload()
        )
        multi = self._write_json("multi.json", multi_payload)
        loaded_multi = _load_launch_spec(
            multi,
            expected_sha256=hashlib.sha256(multi.read_bytes()).hexdigest(),
        )
        config = _process_worker_config(loaded_multi, worker_count=4)
        self.assertEqual(config["device_kind"], "cpu")
        self.assertIsNone(config["device_index"])
        self.assertFalse(config["allow_cpu_oversubscription"])

        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "requires an explicit",
        ):
            _process_worker_config(single_payload, worker_count=2)
        hostile = self._process_worker_payload()
        hostile["device_kind"] = "cuda"
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "CPU/null",
        ):
            _process_worker_config(
                self._launch_payload(process_workers=hostile),
                worker_count=2,
            )

    def test_health_command_parses_strict_phase_bound_json(self) -> None:
        executable = self.base / "health-probe"
        executable.write_text(
            _sealed_shell_probe(
                'printf \'{"phase":"%s","frontier_tick":%s,'
                f'"healthy":true,"baseline_file_sha256":"{_BASELINE_SHA256}"}}\' '
                '"$EVOSIM_OPEN_ECOLOGY_HEALTH_PHASE" '
                '"$EVOSIM_OPEN_ECOLOGY_FRONTIER_TICK"\n'
            ),
            encoding="utf-8",
        )
        executable.chmod(0o700)

        payload = self._health_probe(executable)(
            "frontier_quiescent",
            10_000,
        )

        authority = payload.pop("health_probe_authority")
        self.assertEqual(
            payload,
            {
                "phase": "frontier_quiescent",
                "frontier_tick": 10_000,
                "healthy": True,
                "baseline_file_sha256": _BASELINE_SHA256,
            },
        )
        self.assertEqual(authority["executable"]["path"], str(executable))
        self.assertEqual(
            authority["executable"]["sha256"],
            hashlib.sha256(executable.read_bytes()).hexdigest(),
        )

        duplicate = self.base / "duplicate-probe"
        duplicate.write_text(
            _sealed_shell_probe(
                'printf \'{"phase":"x","phase":"y",'
                f'"frontier_tick":1,"healthy":true,'
                f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}\'\n'
            ),
            encoding="utf-8",
        )
        duplicate.chmod(0o700)
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "duplicate keys",
        ):
            self._health_probe(duplicate)("x", 1)

    def test_launch_spec_and_health_executable_identity_fail_closed(self) -> None:
        spec = self._write_json(
            "launch.json",
            self._launch_payload(process_workers=None),
        )
        expected = hashlib.sha256(spec.read_bytes()).hexdigest()
        loaded = _load_launch_spec(spec, expected_sha256=expected)
        self.assertEqual(loaded["campaign_id"], "campaign-test")

        replacement = self._write_json(
            "replacement.json",
            {**self._launch_payload(process_workers=None), "campaign_id": "forged"},
        )
        os.replace(replacement, spec)
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "externally supplied SHA256",
        ):
            _load_launch_spec(spec, expected_sha256=expected)

        real_probe = self.base / "real-probe"
        real_probe.write_text(
            _sealed_shell_probe(
                "printf '"
                '{"phase":"x","frontier_tick":1,"healthy":true,'
                f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
                "'\n"
            ),
            encoding="utf-8",
        )
        real_probe.chmod(0o700)
        probe_link = self.base / "probe-link"
        probe_link.symlink_to(real_probe)
        linked_payload = self._health_probe(probe_link)("x", 1)
        linked_authority = linked_payload["health_probe_authority"]["executable"]
        self.assertEqual(linked_authority["path"], str(probe_link))
        self.assertEqual(linked_authority["resolved_path"], str(real_probe))

        sealed_probe = self.base / "sealed-probe"
        sealed_probe.write_bytes(real_probe.read_bytes())
        sealed_probe.chmod(0o700)
        sealed_sha256 = hashlib.sha256(sealed_probe.read_bytes()).hexdigest()
        hostile_before_launch = self.base / "hostile-before-launch"
        hostile_before_launch.write_text(
            _sealed_shell_probe(
                "# replacement\n"
                "printf '"
                '{"phase":"x","frontier_tick":1,"healthy":true,'
                f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
                "'\n"
            ),
            encoding="utf-8",
        )
        hostile_before_launch.chmod(0o700)
        os.replace(hostile_before_launch, sealed_probe)
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "launch-spec SHA256",
        ):
            self._health_probe(
                sealed_probe,
                expected_probe_file_sha256=sealed_sha256,
            )

        pinned_probe = self.base / "pinned-probe"
        pinned_probe.write_bytes(real_probe.read_bytes())
        pinned_probe.chmod(0o700)
        probe = self._health_probe(pinned_probe)
        hostile_probe = self.base / "hostile-probe"
        hostile_probe.write_text(
            _sealed_shell_probe(
                "printf '"
                '{"phase":"x","frontier_tick":1,"healthy":false,'
                f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
                "'\n"
            ),
            encoding="utf-8",
        )
        hostile_probe.chmod(0o700)
        os.replace(hostile_probe, pinned_probe)
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "identity or SHA256 changed",
        ):
            probe("x", 1)

    def test_seal_bundle_closes_separate_bundle_under_pinned_source(self) -> None:
        active = self.base / "active"
        bundle = self.base / "bundle-0001"
        active.mkdir()
        bundle.mkdir()
        (active / "frontier.json").write_text("active", encoding="utf-8")
        (bundle / "evidence.json").write_text("closed", encoding="utf-8")
        payload = self._launch_payload(process_workers=None)
        payload["campaign_root"] = str(active)
        launch_spec = self._write_json("launch-seal.json", payload)
        launch_spec_sha256 = hashlib.sha256(launch_spec.read_bytes()).hexdigest()
        source_snapshot = {
            "git_clean": True,
            "repository_root": payload["repository_root"],
            "source_git_sha": payload["source_git_sha"],
            "source_manifest_sha256": payload["source_manifest_sha256"],
        }
        output = io.StringIO()

        with (
            mock.patch(
                "evolution_sim.cli.open_ecology_campaign._campaign_source_snapshot",
                return_value=source_snapshot,
            ) as source_probe,
            mock.patch.object(
                storage,
                "_descriptor_disk_usage",
                return_value=(1024**4, 512 * 1024**3),
            ),
            redirect_stdout(output),
        ):
            result = main(
                (
                    "seal-bundle",
                    "--launch-spec",
                    str(launch_spec),
                    "--expected-launch-spec-sha256",
                    launch_spec_sha256,
                    "--bundle-dir",
                    str(bundle),
                    "--bundle-id",
                    bundle.name,
                )
            )

        self.assertEqual(result, 0)
        self.assertEqual(source_probe.call_count, 2)
        report = json.loads(output.getvalue())
        self.assertEqual(output.getvalue().count("\n"), 1)
        self.assertEqual(report["bundle_dir"], str(bundle))
        self.assertEqual(report["bundle_id"], bundle.name)
        self.assertEqual(report["campaign_id"], "campaign-test")
        self.assertEqual(report["content_file_count"], 1)
        self.assertEqual(report["content_directory_count"], 0)
        self.assertTrue(report["source_verified_before_and_after"])
        self.assertFalse(report["pruning_or_deletion_performed"])
        self.assertEqual(
            report["storage_limits"],
            {
                "max_campaign_bytes": ACTIVE_CAMPAIGN_BUDGET_BYTES,
                "max_entries": DEFAULT_MAX_ENTRIES,
                "min_campaign_free_bytes": DEFAULT_ACTIVE_FILESYSTEM_FREE_BYTES,
                "min_remote_free_bytes": DEFAULT_REMOTE_FREE_BYTES,
            },
        )
        self.assertTrue((bundle / storage.MARKER_NAME).is_file())
        self.assertFalse(bundle.stat().st_mode & 0o222)
        self.assertFalse((bundle / "evidence.json").stat().st_mode & 0o222)

    def test_seal_bundle_source_drift_fails_closed_before_or_after_seal(
        self,
    ) -> None:
        active = self.base / "active"
        active.mkdir()
        payload = self._launch_payload(process_workers=None)
        payload["campaign_root"] = str(active)
        launch_spec = self._write_json("launch-drift.json", payload)
        launch_spec_sha256 = hashlib.sha256(launch_spec.read_bytes()).hexdigest()
        arguments = (
            "seal-bundle",
            "--launch-spec",
            str(launch_spec),
            "--expected-launch-spec-sha256",
            launch_spec_sha256,
        )

        before_bundle = self.base / "bundle-before-drift"
        before_bundle.mkdir()
        (before_bundle / "evidence").write_bytes(b"source-bound")
        before_output = io.StringIO()
        with (
            mock.patch(
                "evolution_sim.cli.open_ecology_campaign._campaign_source_snapshot",
                side_effect=OpenEcologyCampaignCoordinatorError(
                    "cannot verify pinned live source authority"
                ),
            ),
            mock.patch(
                "evolution_sim.cli.open_ecology_campaign.seal_closed_bundle",
            ) as sealer,
            redirect_stdout(before_output),
            self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "cannot verify pinned live source authority",
            ),
        ):
            main(
                arguments
                + (
                    "--bundle-dir",
                    str(before_bundle),
                    "--bundle-id",
                    before_bundle.name,
                )
            )
        sealer.assert_not_called()
        self.assertEqual(before_output.getvalue(), "")
        self.assertFalse((before_bundle / storage.MARKER_NAME).exists())

        after_bundle = self.base / "bundle-after-drift"
        after_bundle.mkdir()
        (after_bundle / "evidence").write_bytes(b"source-bound")
        source_snapshot = {
            "git_clean": True,
            "repository_root": payload["repository_root"],
            "source_git_sha": payload["source_git_sha"],
            "source_manifest_sha256": payload["source_manifest_sha256"],
        }
        after_output = io.StringIO()
        with (
            mock.patch(
                "evolution_sim.cli.open_ecology_campaign._campaign_source_snapshot",
                side_effect=(
                    source_snapshot,
                    OpenEcologyCampaignCoordinatorError(
                        "cannot verify pinned live source authority"
                    ),
                ),
            ),
            mock.patch.object(
                storage,
                "_descriptor_disk_usage",
                return_value=(1024**4, 512 * 1024**3),
            ),
            redirect_stdout(after_output),
            self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "cannot verify pinned live source authority",
            ),
        ):
            main(
                arguments
                + (
                    "--bundle-dir",
                    str(after_bundle),
                    "--bundle-id",
                    after_bundle.name,
                )
            )
        self.assertEqual(after_output.getvalue(), "")
        self.assertTrue((after_bundle / storage.MARKER_NAME).exists())

        changed_bundle = self.base / "bundle-changed-source"
        changed_bundle.mkdir()
        (changed_bundle / "evidence").write_bytes(b"source-bound")
        changed_output = io.StringIO()
        with (
            mock.patch(
                "evolution_sim.cli.open_ecology_campaign._campaign_source_snapshot",
                side_effect=(
                    source_snapshot,
                    {**source_snapshot, "git_clean": False},
                ),
            ),
            mock.patch.object(
                storage,
                "_descriptor_disk_usage",
                return_value=(1024**4, 512 * 1024**3),
            ),
            redirect_stdout(changed_output),
            self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "changed across closed-bundle sealing",
            ),
        ):
            main(
                arguments
                + (
                    "--bundle-dir",
                    str(changed_bundle),
                    "--bundle-id",
                    changed_bundle.name,
                )
            )
        self.assertEqual(changed_output.getvalue(), "")
        self.assertTrue((changed_bundle / storage.MARKER_NAME).exists())

    def test_seal_bundle_rejects_overlap_marker_and_malformed_paths(self) -> None:
        active = self.base / "active"
        active.mkdir()
        payload = self._launch_payload(process_workers=None)
        payload["campaign_root"] = str(active)
        launch_spec = self._write_json("launch-hostile.json", payload)
        launch_spec_sha256 = hashlib.sha256(launch_spec.read_bytes()).hexdigest()
        common = (
            "seal-bundle",
            "--launch-spec",
            str(launch_spec),
            "--expected-launch-spec-sha256",
            launch_spec_sha256,
        )
        source_snapshot = {
            "git_clean": True,
            "repository_root": payload["repository_root"],
            "source_git_sha": payload["source_git_sha"],
            "source_manifest_sha256": payload["source_manifest_sha256"],
        }

        overlap = active / "bundle-overlap"
        overlap.mkdir()
        (overlap / "evidence").write_bytes(b"x")
        existing = self.base / "bundle-existing"
        existing.mkdir()
        (existing / storage.MARKER_NAME).write_text("{}", encoding="utf-8")
        canonical = self.base / "bundle-canonical"
        canonical.mkdir()
        (canonical / "evidence").write_bytes(b"x")
        noncanonical = self.base / "subdir" / ".." / canonical.name
        wrong_name = self.base / "bundle-wrong-name"
        wrong_name.mkdir()
        (wrong_name / "evidence").write_bytes(b"x")

        with mock.patch(
            "evolution_sim.cli.open_ecology_campaign._campaign_source_snapshot",
            return_value=source_snapshot,
        ):
            with self.assertRaisesRegex(CampaignStorageError, "separate tree"):
                main(
                    common
                    + (
                        "--bundle-dir",
                        str(overlap),
                        "--bundle-id",
                        overlap.name,
                    )
                )
            with self.assertRaisesRegex(CampaignStorageError, "already exists"):
                main(
                    common
                    + (
                        "--bundle-dir",
                        str(existing),
                        "--bundle-id",
                        existing.name,
                    )
                )
            with self.assertRaisesRegex(CampaignStorageError, "lexically canonical"):
                main(
                    common
                    + (
                        "--bundle-dir",
                        str(noncanonical),
                        "--bundle-id",
                        canonical.name,
                    )
                )
            with self.assertRaisesRegex(
                CampaignStorageError,
                "directory name must equal bundle id",
            ):
                main(
                    common
                    + (
                        "--bundle-dir",
                        str(wrong_name),
                        "--bundle-id",
                        "bundle-other-name",
                    )
                )
            with self.assertRaisesRegex(
                CampaignStorageError,
                "conservative identifier",
            ):
                main(
                    common
                    + (
                        "--bundle-dir",
                        str(canonical),
                        "--bundle-id",
                        "../hostile",
                    )
                )

    def test_health_probe_rejects_mid_launch_replacement_and_bounds_both_pipes(
        self,
    ) -> None:
        pinned_probe = self.base / "pinned-probe"
        pinned_probe.write_text(
            _sealed_shell_probe(
                "printf '"
                '{"phase":"x","frontier_tick":1,"healthy":true,'
                f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
                "'\n"
            ),
            encoding="utf-8",
        )
        pinned_probe.chmod(0o700)
        probe = self._health_probe(pinned_probe)
        hostile_probe = self.base / "hostile-probe"
        hostile_probe.write_text(
            _sealed_shell_probe(
                "printf '"
                '{"phase":"x","frontier_tick":1,"healthy":true,'
                f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
                "'\n"
            ),
            encoding="utf-8",
        )
        hostile_probe.chmod(0o700)
        real_popen = subprocess.Popen

        def replace_then_launch(*args, **kwargs):
            os.replace(hostile_probe, pinned_probe)
            return real_popen(*args, **kwargs)

        with (
            mock.patch(
                "evolution_sim.cli.open_ecology_campaign.subprocess.Popen",
                side_effect=replace_then_launch,
            ),
            self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "changed during execution",
            ),
        ):
            probe("x", 1)

        stderr_probe = self.base / "stderr-probe"
        stderr_probe.write_text(
            _sealed_shell_probe(
                "printf warning >&2\n"
                "printf '"
                '{"phase":"x","frontier_tick":1,"healthy":true,'
                f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
                "'\n"
            ),
            encoding="utf-8",
        )
        stderr_probe.chmod(0o700)
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "wrote to stderr",
        ):
            self._health_probe(stderr_probe)("x", 1)

        oversized_probe = self.base / "oversized-probe"
        oversized_probe.write_text(
            _sealed_shell_probe("printf '" + ("x" * (1024 * 1024 + 1)) + "'\n"),
            encoding="utf-8",
        )
        oversized_probe.chmod(0o700)
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "output exceeds 1 MiB",
        ):
            self._health_probe(oversized_probe)("x", 1)

        script_argument = self.base / "replaceable-probe.py"
        script_argument.write_text(
            'print(\'{"phase":"x","frontier_tick":1,"healthy":true}\')\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "one standalone executable",
        ):
            _command_health_probe(
                ("/usr/bin/python3", str(script_argument)),
                expected_baseline_sha256=_BASELINE_SHA256,
                expected_probe_file_sha256="e" * 64,
            )

        unsealed_interpreter = self.base / "unsealed-interpreter-probe"
        unsealed_interpreter.write_text(
            "#!/bin/sh\nprintf '{}'\n",
            encoding="utf-8",
        )
        unsealed_interpreter.chmod(0o700)
        with self.assertRaisesRegex(
            OpenEcologyCampaignCoordinatorError,
            "embedded shebang interpreter SHA256",
        ):
            self._health_probe(unsealed_interpreter)

    def test_native_health_probe_never_executes_restore_attack_bytes(self) -> None:
        probe_path = self.base / "native-probe"
        shutil.copyfile(Path(sys.executable).resolve(), probe_path)
        probe_path.chmod(0o500)
        probe = self._health_probe(
            probe_path,
            expected_probe_file_sha256=hashlib.sha256(
                probe_path.read_bytes()
            ).hexdigest(),
        )
        original_backup = self.base / "native-probe.original"
        displaced_hostile = self.base / "native-probe.hostile"
        hostile_marker = self.base / "hostile-native-executed"
        hostile_target = self.base / "hostile-native-target"
        hostile_target.write_text(
            "#!/bin/sh\n"
            f"touch {str(hostile_marker)!r}\n"
            "printf '"
            '{"phase":"x","frontier_tick":1,"healthy":true,'
            f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
            "'\n",
            encoding="utf-8",
        )
        hostile_target.chmod(0o500)
        real_popen = subprocess.Popen

        def replace_launch_restore(*args, **kwargs):
            os.replace(probe_path, original_backup)
            os.replace(hostile_target, probe_path)
            try:
                return real_popen(*args, **kwargs)
            finally:
                os.replace(probe_path, displaced_hostile)
                os.replace(original_backup, probe_path)

        with (
            mock.patch(
                "evolution_sim.cli.open_ecology_campaign.subprocess.Popen",
                side_effect=replace_launch_restore,
            ),
            self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "changed during execution",
            ),
        ):
            probe("x", 1)
        self.assertFalse(
            hostile_marker.exists(),
            "unchecked native replacement executed before post-validation",
        )

    def test_script_probe_and_interpreter_execute_validated_snapshots(self) -> None:
        # Canonical interpreter links are allowed: execution uses snapshotted bytes,
        # sealed runtime authority separately binds venv/site packages, and persistent
        # source drift through any hardlink alias is rejected by postvalidation.
        benign_interpreter = Path(sys.executable).resolve()
        interpreter_path = self.base / "probe-interpreter"
        interpreter_path.symlink_to(benign_interpreter)
        interpreter_sha256 = hashlib.sha256(benign_interpreter.read_bytes()).hexdigest()

        benign_script = self.base / "benign-script"
        benign_script.write_text(
            f"#!{interpreter_path}\n"
            f"# evosim_shebang_sha256={interpreter_sha256}\n"
            "print('"
            '{"phase":"x","frontier_tick":1,"healthy":true,'
            f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
            "')\n",
            encoding="utf-8",
        )
        benign_script.chmod(0o500)
        script_path = self.base / "script-probe"
        script_path.symlink_to(benign_script)
        probe = self._health_probe(script_path)

        hostile_marker = self.base / "hostile-script-executed"
        hostile_script = self.base / "hostile-script"
        hostile_script.write_text(
            f"#!{interpreter_path}\n"
            f"# evosim_shebang_sha256={interpreter_sha256}\n"
            "from pathlib import Path\n"
            f"Path({str(hostile_marker)!r}).write_text('hostile')\n"
            "print('"
            '{"phase":"x","frontier_tick":1,"healthy":true,'
            f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
            "')\n",
            encoding="utf-8",
        )
        hostile_script.chmod(0o500)
        real_popen = subprocess.Popen

        def replace_script_launch_restore(*args, **kwargs):
            script_path.unlink()
            script_path.symlink_to(hostile_script)
            try:
                return real_popen(*args, **kwargs)
            finally:
                script_path.unlink()
                script_path.symlink_to(benign_script)

        with mock.patch(
            "evolution_sim.cli.open_ecology_campaign.subprocess.Popen",
            side_effect=replace_script_launch_restore,
        ):
            payload = probe("x", 1)
        self.assertTrue(payload["healthy"])
        self.assertFalse(
            hostile_marker.exists(),
            "unchecked script replacement executed before post-validation",
        )

        hostile_interpreter = self.base / "hostile-interpreter"
        shutil.copyfile(Path("/usr/bin/false").resolve(), hostile_interpreter)
        hostile_interpreter.chmod(0o500)

        def replace_interpreter_launch_restore(*args, **kwargs):
            interpreter_path.unlink()
            interpreter_path.symlink_to(hostile_interpreter)
            try:
                return real_popen(*args, **kwargs)
            finally:
                interpreter_path.unlink()
                interpreter_path.symlink_to(benign_interpreter)

        with mock.patch(
            "evolution_sim.cli.open_ecology_campaign.subprocess.Popen",
            side_effect=replace_interpreter_launch_restore,
        ):
            payload = probe("x", 1)
        self.assertTrue(payload["healthy"])

    def test_mutable_probe_interpreter_never_executes_restore_attack_bytes(
        self,
    ) -> None:
        interpreter_path = self.base / "mutable-interpreter"
        shutil.copyfile(Path(sys.executable).resolve(), interpreter_path)
        interpreter_path.chmod(0o500)
        interpreter_sha256 = hashlib.sha256(interpreter_path.read_bytes()).hexdigest()
        probe_path = self.base / "interpreter-probe"
        probe_path.write_text(
            f"#!{interpreter_path}\n"
            f"# evosim_shebang_sha256={interpreter_sha256}\n"
            "print('"
            '{"phase":"x","frontier_tick":1,"healthy":true,'
            f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
            "')\n",
            encoding="utf-8",
        )
        probe_path.chmod(0o500)
        probe = self._health_probe(probe_path)
        hostile_marker = self.base / "hostile-interpreter-executed"
        hostile_interpreter = self.base / "hostile-interpreter-script"
        hostile_interpreter.write_text(
            "#!/bin/sh\n"
            f"touch {str(hostile_marker)!r}\n"
            "printf '"
            '{"phase":"x","frontier_tick":1,"healthy":true,'
            f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
            "'\n",
            encoding="utf-8",
        )
        hostile_interpreter.chmod(0o500)
        original_backup = self.base / "mutable-interpreter.original"
        displaced_hostile = self.base / "mutable-interpreter.hostile"
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
            mock.patch(
                "evolution_sim.cli.open_ecology_campaign.subprocess.Popen",
                side_effect=replace_launch_restore,
            ),
            self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "changed during execution",
            ),
        ):
            probe("x", 1)
        self.assertFalse(
            hostile_marker.exists(),
            "unchecked interpreter replacement executed before post-validation",
        )

    def test_health_probe_breach_kills_descendant_after_leader_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            child_pid_path = Path(temporary) / "child.pid"
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
                mock.patch(
                    "evolution_sim.cli.open_ecology_campaign._MAX_HEALTH_PROBE_BYTES",
                    8192,
                ),
                self.assertRaisesRegex(
                    OpenEcologyCampaignCoordinatorError,
                    "output exceeds",
                ),
            ):
                _run_bounded_health_probe(
                    (sys.executable, "-c", parent_code),
                    environment={
                        "LANG": "C",
                        "LC_ALL": "C",
                        "PATH": "/usr/bin:/bin",
                    },
                )
            _assert_process_gone(self, child_pid_path)

    def test_health_probe_setup_failures_close_the_process_group(self) -> None:
        for setup_step in (
            "selector",
            "fileno",
            "set_blocking",
            "register",
        ):
            with self.subTest(setup_step=setup_step):
                self._assert_health_probe_setup_failure_closes_group(setup_step)

    def test_health_probe_cleanup_failure_preserves_primary_output_error(self) -> None:
        from evolution_sim.cli import open_ecology_campaign as campaign

        noisy = self.base / "noisy-health-probe"
        noisy.write_text(
            "#!/bin/sh\nprintf '0123456789abcdef0123456789abcdef'\n",
            encoding="utf-8",
        )
        noisy.chmod(0o700)
        real_terminate = campaign._terminate_health_probe

        def terminate_then_report_failure(process, *, leader_exit_observed=False):
            real_terminate(
                process,
                leader_exit_observed=leader_exit_observed,
            )
            raise OpenEcologyCampaignCoordinatorError(
                "simulated cleanup verification failure"
            )

        with (
            mock.patch.object(campaign, "_MAX_HEALTH_PROBE_BYTES", 16),
            mock.patch.object(
                campaign,
                "_terminate_health_probe",
                side_effect=terminate_then_report_failure,
            ),
            self.assertRaisesRegex(
                OpenEcologyCampaignCoordinatorError,
                "output exceeds",
            ),
        ):
            _run_bounded_health_probe(
                (str(noisy),),
                environment={
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PATH": "/usr/bin:/bin",
                },
            )

    def test_health_probe_ignores_hostile_inherited_python_environment(self) -> None:
        probe_path = self.base / "environment-probe"
        probe_path.write_text(
            _sealed_shell_probe(
                'test "${PYTHONPATH+x}" != x\n'
                'test "${PYTHONHOME+x}" != x\n'
                'test "$PYTHONNOUSERSITE" = 1\n'
                'test "$PYTHONSAFEPATH" = 1\n'
                "printf '"
                '{"phase":"x","frontier_tick":1,"healthy":true,'
                f'"baseline_file_sha256":"{_BASELINE_SHA256}"}}'
                "'\n"
            ),
            encoding="utf-8",
        )
        probe_path.chmod(0o700)
        with mock.patch.dict(
            os.environ,
            {
                "PYTHONPATH": "/hostile",
                "PYTHONHOME": "/hostile",
                "PYTHONNOUSERSITE": "0",
                "PYTHONSAFEPATH": "0",
            },
        ):
            payload = self._health_probe(probe_path)("x", 1)
        self.assertTrue(payload["healthy"])

    def _assert_health_probe_setup_failure_closes_group(
        self,
        setup_step: str,
    ) -> None:
        from evolution_sim.cli import open_ecology_campaign as campaign

        child_pid_path = self.base / f"setup-{setup_step}-child.pid"
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
        real_selector = campaign.selectors.DefaultSelector

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

        def start_probe(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            deadline = time.monotonic() + 2.0
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not child_pid_path.exists():
                raise AssertionError("probe descendant did not start")
            if setup_step == "fileno":
                assert process.stdout is not None
                process.stdout = FailingFileno(process.stdout)
            return process

        try:
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        campaign.subprocess,
                        "Popen",
                        side_effect=start_probe,
                    )
                )
                if setup_step == "selector":
                    stack.enter_context(
                        mock.patch.object(
                            campaign.selectors,
                            "DefaultSelector",
                            side_effect=failure,
                        )
                    )
                elif setup_step == "set_blocking":
                    stack.enter_context(
                        mock.patch.object(
                            campaign.os,
                            "set_blocking",
                            side_effect=failure,
                        )
                    )
                elif setup_step == "register":
                    stack.enter_context(
                        mock.patch.object(
                            campaign.selectors,
                            "DefaultSelector",
                            side_effect=FailingRegisterSelector,
                        )
                    )
                with self.assertRaises(RuntimeError) as caught:
                    _run_bounded_health_probe(
                        (sys.executable, "-c", parent_code),
                        environment={
                            "LANG": "C",
                            "LC_ALL": "C",
                            "PATH": "/usr/bin:/bin",
                        },
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

    def _health_probe(
        self,
        path: Path,
        *,
        expected_probe_file_sha256: str | None = None,
    ):
        expected = (
            hashlib.sha256(path.resolve().read_bytes()).hexdigest()
            if expected_probe_file_sha256 is None
            else expected_probe_file_sha256
        )
        return _command_health_probe(
            (str(path),),
            expected_baseline_sha256=_BASELINE_SHA256,
            expected_probe_file_sha256=expected,
        )

    def _write_json(self, name: str, payload: object) -> Path:
        path = self.base / name
        path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def _launch_payload(self, *, process_workers):
        return {
            "schema_version": OPEN_ECOLOGY_CAMPAIGN_LAUNCH_SPEC_SCHEMA_VERSION,
            "campaign_id": "campaign-test",
            "campaign_root": str(self.base),
            "receipt_directory": str(self.base / "receipts"),
            "repository_root": str(Path(__file__).resolve().parents[2]),
            "git_executable": str(
                Path(shutil.which("git") or "/usr/bin/git").resolve()
            ),
            "git_executable_sha256": hashlib.sha256(
                Path(shutil.which("git") or "/usr/bin/git").resolve().read_bytes()
            ).hexdigest(),
            "source_git_sha": "a" * 40,
            "source_manifest_sha256": "b" * 64,
            "selected_density": 64,
            "worker_count": 1 if process_workers is None else 4,
            "process_workers": process_workers,
            "health_probe_command": ["/usr/bin/true"],
            "health_baseline_sha256": _BASELINE_SHA256,
            "health_probe_file_sha256": hashlib.sha256(
                Path("/usr/bin/true").resolve().read_bytes()
            ).hexdigest(),
            "artifacts": [
                {
                    "learner_index": index,
                    "learner_seed": index + 1,
                    "artifact_path": str(self.base / f"artifact-{index}.json"),
                    "artifact_sha256": f"{index + 1:x}" * 64,
                    "artifact_file_sha256": f"{index + 5:x}" * 64,
                    "source_commit": "a" * 40,
                    "terminal_authority_sha256": f"{index + 9:x}" * 64,
                }
                for index in range(4)
            ],
        }

    @staticmethod
    def _process_worker_payload() -> dict[str, object]:
        return {
            "host_identity": local_process_worker_host_identity(),
            "device_kind": "cpu",
            "device_index": None,
            "torch_threads_per_worker": 1,
            "allow_cpu_oversubscription": False,
            "response_timeout_seconds": 3_600.0,
            "startup_timeout_seconds": 120.0,
            "shutdown_timeout_seconds": 30.0,
            "max_message_bytes": 8 * 1024 * 1024,
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
        f"health-probe descendant survived cleanup: {state}",
    )


if __name__ == "__main__":
    unittest.main()
