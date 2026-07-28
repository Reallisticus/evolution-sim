from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evolution_sim.io import open_ecology_campaign_storage as storage
from evolution_sim.io.open_ecology_archive_authority import (
    FilePin,
    verify_pinned_file,
)
from scripts import archive_evolution_outputs as archive_outputs
from scripts import archive_open_ecology_campaign as campaign_archive
from scripts import seal_open_ecology_archive_authority as authority_sealer


class OpenEcologyArchiveLivenessTests(unittest.TestCase):
    def test_campaign_generic_command_caps_noisy_child_without_deadlock(self) -> None:
        started = time.monotonic()
        with self.assertRaisesRegex(
            campaign_archive.OpenEcologyArchiveError,
            "stderr exceeded 4096 bytes",
        ):
            campaign_archive._run(
                (
                    sys.executable,
                    "-c",
                    ("import os\nwhile True:\n os.write(2,b'x'*65536)\n"),
                ),
                timeout_seconds=2.0,
                stdout_limit=4096,
                stderr_limit=4096,
            )
        self.assertLess(time.monotonic() - started, 2.0)

    def test_campaign_fast_exit_propagates_joined_drain_failure(self) -> None:
        def watchdog_waits_for_completion(
            guard: campaign_archive._BoundedProcess,
        ) -> None:
            guard.finished.wait()

        with patch.object(
            campaign_archive._BoundedProcess,
            "_watch",
            watchdog_waits_for_completion,
        ):
            with self.assertRaisesRegex(
                campaign_archive.OpenEcologyArchiveError,
                "stdout exceeded 1 bytes",
            ):
                campaign_archive._run(
                    (
                        sys.executable,
                        "-c",
                        "import os; os.write(1, b'fast-exit')",
                    ),
                    timeout_seconds=2.0,
                    stdout_limit=1,
                    stderr_limit=4096,
                )

    def test_campaign_deadline_kills_the_child_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "descendant-survived"
            child_code = (
                "import pathlib,signal,time\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                "time.sleep(0.7)\n"
                f"pathlib.Path({str(marker)!r}).write_text('bad')\n"
            )
            parent_code = (
                "import signal,subprocess,sys,time\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                f"subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
                "time.sleep(30)\n"
            )
            started = time.monotonic()
            with self.assertRaisesRegex(
                campaign_archive.OpenEcologyArchiveError,
                "total deadline",
            ):
                campaign_archive._run(
                    (sys.executable, "-c", parent_code),
                    timeout_seconds=0.1,
                )
            self.assertLess(time.monotonic() - started, 1.0)
            time.sleep(0.8)
            self.assertFalse(marker.exists())

    def test_successful_campaign_command_kills_lingering_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "campaign-descendant-survived"
            completed = campaign_archive._run(
                self._successful_parent_with_lingering_descendant(marker),
                timeout_seconds=2.0,
            )
            self.assertEqual(completed.returncode, 0)
            time.sleep(0.7)
            self.assertFalse(marker.exists())

    def test_successful_archive_command_kills_lingering_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "archive-descendant-survived"
            completed = archive_outputs._run_bounded_command(
                self._successful_parent_with_lingering_descendant(marker),
                timeout_seconds=2.0,
            )
            self.assertEqual(completed.returncode, 0)
            time.sleep(0.7)
            self.assertFalse(marker.exists())

    def test_drive_readback_caps_stderr_while_stdout_is_streamed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_rclone = Path(directory) / "rclone"
            fake_rclone.write_text(
                "#!/bin/sh\nwhile :; do printf '0123456789abcdef' >&2; done\n",
                encoding="utf-8",
            )
            fake_rclone.chmod(0o755)
            with (
                patch.object(
                    campaign_archive,
                    "minimal_subprocess_env",
                    return_value={
                        "LANG": "C",
                        "LC_ALL": "C",
                        "PATH": f"{directory}:/usr/bin:/bin",
                    },
                ),
                patch.object(
                    campaign_archive,
                    "_MAX_CAPTURE_STDERR_BYTES",
                    4096,
                ),
            ):
                with self.assertRaisesRegex(
                    campaign_archive.OpenEcologyArchiveError,
                    "stderr exceeded 4096 bytes",
                ):
                    campaign_archive._rclone_readback(
                        "gdrive:test/object",
                        max_bytes=1,
                    )

    def test_drive_readback_enforces_exact_in_flight_stdout_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_rclone = Path(directory) / "rclone"
            fake_rclone.write_text(
                "#!/bin/sh\nprintf 'too-large'\n",
                encoding="utf-8",
            )
            fake_rclone.chmod(0o755)
            with patch.object(
                campaign_archive,
                "minimal_subprocess_env",
                return_value={
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PATH": f"{directory}:/usr/bin:/bin",
                },
            ):
                with self.assertRaisesRegex(
                    campaign_archive.OpenEcologyArchiveError,
                    "stdout exceeded 1 bytes",
                ):
                    campaign_archive._rclone_readback(
                        "gdrive:test/object",
                        max_bytes=1,
                    )

    def test_authority_sealer_caps_noisy_pinned_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory).resolve() / "noisy-tool"
            executable.write_text(
                "#!/bin/sh\nwhile :; do printf '0123456789abcdef' >&2; done\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            pin = FilePin(
                path=str(executable),
                sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
            )
            started = time.monotonic()
            with self.assertRaisesRegex(
                authority_sealer.AuthoritySealError,
                "stderr exceeded 4096 bytes",
            ):
                authority_sealer._run_pinned(
                    pin,
                    (str(executable),),
                    timeout_seconds=2.0,
                    stdout_limit=4096,
                    stderr_limit=4096,
                )
            self.assertLess(time.monotonic() - started, 2.0)

    def test_authority_sealer_enforces_total_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory).resolve() / "stalled-tool"
            executable.write_text(
                "#!/bin/sh\nwhile :; do :; done\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            pin = FilePin(
                path=str(executable),
                sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(
                authority_sealer.AuthoritySealError,
                "total deadline",
            ):
                authority_sealer._run_pinned(
                    pin,
                    (str(executable),),
                    timeout_seconds=0.1,
                )

    def test_successful_authority_tool_kills_lingering_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory).resolve() / "spawning-tool"
            marker = Path(directory) / "authority-descendant-survived"
            child_code = (
                "import pathlib,time\n"
                "time.sleep(0.5)\n"
                f"pathlib.Path({str(marker)!r}).write_text('bad')\n"
            )
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import subprocess,sys\n"
                f"subprocess.Popen([sys.executable,'-c',{child_code!r}])\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            pin = FilePin(
                path=str(executable),
                sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
            )
            completed = authority_sealer._run_pinned(
                pin,
                (str(executable),),
                timeout_seconds=2.0,
            )
            self.assertEqual(completed.returncode, 0)
            time.sleep(0.7)
            self.assertFalse(marker.exists())

    def test_remote_python_symlink_is_sealed_as_consumer_valid_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            resolved_python = root / "python-real"
            resolved_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            resolved_python.chmod(0o755)
            python_symlink = root / "python"
            python_symlink.symlink_to(resolved_python.name)
            tools: dict[str, str] = {}
            for name in authority_sealer.REMOTE_TOOL_NAMES:
                if name == "python":
                    tools[name] = str(python_symlink)
                    continue
                executable = root / name
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(0o755)
                tools[name] = str(executable)

            def resolve_remote(
                _ssh_pin: FilePin,
                _ssh_target: str,
                command: tuple[str, ...],
            ) -> subprocess.CompletedProcess[bytes]:
                raw = Path(command[-1])
                resolved = raw.resolve(strict=True)
                return subprocess.CompletedProcess(
                    command,
                    0,
                    f"{resolved}\n".encode(),
                    b"",
                )

            with patch.object(
                authority_sealer,
                "_run_ssh",
                side_effect=resolve_remote,
            ) as run_ssh:
                resolved = authority_sealer._canonicalize_remote_executable_paths(
                    ssh_pin=FilePin(path="/unused/ssh", sha256="0" * 64),
                    ssh_target="trainer",
                    remote_paths=tools,
                )

            self.assertEqual(resolved["python"], str(resolved_python))
            self.assertEqual(
                run_ssh.call_args_list[0].args[2][0],
                str(python_symlink),
            )
            for call in run_ssh.call_args_list[1:]:
                self.assertEqual(call.args[2][0], str(resolved_python))
            verify_pinned_file(
                FilePin(
                    path=resolved["python"],
                    sha256=hashlib.sha256(resolved_python.read_bytes()).hexdigest(),
                ),
                executable=True,
                require_nonempty=True,
            )

    def test_authority_sealer_requires_exact_local_tool_name_set(self) -> None:
        args = SimpleNamespace(
            repository_root=authority_sealer._REPOSITORY_ROOT,
            remote_repository_root="/remote/repository",
            ssh_target="trainer-node",
            local_git_path="/tool/git",
            local_rclone_path="/tool/rclone",
            local_ssh_path="/tool/ssh",
            local_zstd_path="/tool/zstd",
        )
        with (
            patch.object(
                authority_sealer,
                "LOCAL_TOOL_NAMES",
                ("git", "rclone", "ssh"),
            ),
            self.assertRaisesRegex(
                authority_sealer.AuthoritySealError,
                "exactly match LOCAL_TOOL_NAMES",
            ),
        ):
            authority_sealer.seal_authority(args)

    def test_authority_sealer_exclusive_write_has_exact_readback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory).resolve() / "authority.json"
            payload = b'{"sealed":true}\n'

            observed = authority_sealer._write_exclusive(destination, payload)
            metadata = observed.stat()

            self.assertEqual(observed, destination)
            self.assertEqual(observed.read_bytes(), payload)
            self.assertEqual(metadata.st_nlink, 1)
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o400)

    def test_authority_sealer_exact_descriptor_readback_detects_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory).resolve() / "authority.json"
            payload = b'{"sealed":true}\n'
            replacement = b'{"sealed":fals}\n'
            real_fsync = os.fsync
            mutated = False

            def fsync_then_mutate(descriptor: int) -> None:
                nonlocal mutated
                real_fsync(descriptor)
                metadata = os.fstat(descriptor)
                if not mutated and stat.S_ISREG(metadata.st_mode):
                    os.pwrite(descriptor, replacement, 0)
                    real_fsync(descriptor)
                    mutated = True

            with (
                patch.object(
                    authority_sealer.os,
                    "fsync",
                    side_effect=fsync_then_mutate,
                ),
                self.assertRaisesRegex(
                    authority_sealer.AuthoritySealError,
                    "exact descriptor readback",
                ),
            ):
                authority_sealer._write_exclusive(destination, payload)

            self.assertEqual(destination.read_bytes(), replacement)

    def test_authority_sealer_never_accepts_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory).resolve() / "authority.json"
            payload = b'{"sealed":true}\n'
            competitor = b'{"other":true}\n'
            real_write = os.write
            replaced = False

            def write_then_replace(descriptor: int, chunk: bytes) -> int:
                nonlocal replaced
                written = real_write(descriptor, chunk)
                if not replaced:
                    destination.unlink()
                    destination.write_bytes(competitor)
                    destination.chmod(0o444)
                    replaced = True
                return written

            with (
                patch.object(
                    authority_sealer.os,
                    "write",
                    side_effect=write_then_replace,
                ),
                self.assertRaisesRegex(
                    authority_sealer.AuthoritySealError,
                    "descriptor changed during publication",
                ),
            ):
                authority_sealer._write_exclusive(destination, payload)

            self.assertEqual(destination.read_bytes(), competitor)

    def test_compressor_stderr_flood_cannot_deadlock_tar_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            (root / "payload.bin").write_bytes(b"x" * (2 * 1024 * 1024))
            snapshot = archive_outputs.build_snapshot(root)
            fake_zstd = Path(directory) / "zstd"
            fake_zstd.write_text(
                "#!/bin/sh\nwhile :; do printf '0123456789abcdef' >&2; done\n",
                encoding="utf-8",
            )
            fake_zstd.chmod(0o755)
            destination = Path(directory) / "bundle.tar.zst"
            with (
                patch.object(archive_outputs, "MAX_CAPTURE_STDERR_BYTES", 4096),
                patch.object(archive_outputs, "ARCHIVE_PROCESS_TIMEOUT_SECONDS", 2.0),
            ):
                with self.assertRaisesRegex(
                    archive_outputs.ArchiveError,
                    "stderr exceeded 4096 bytes",
                ):
                    archive_outputs._create_archive(
                        snapshot,
                        destination,
                        zstd_path=str(fake_zstd),
                    )
            self.assertTrue(destination.exists())
            self.assertEqual(
                list(Path(directory).glob(".*.partial")),
                [],
            )

    def test_generic_archive_command_has_a_total_deadline(self) -> None:
        with self.assertRaisesRegex(
            archive_outputs.ArchiveError,
            "total deadline",
        ):
            archive_outputs._run_bounded_command(
                (
                    sys.executable,
                    "-c",
                    "import time; time.sleep(30)",
                ),
                timeout_seconds=0.1,
            )

    def test_generic_remote_readback_enforces_the_expected_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "rclone"
            executable.write_text("#!/bin/sh\nprintf 'too-large'\n", encoding="utf-8")
            executable.chmod(0o755)
            with self.assertRaisesRegex(
                archive_outputs.ArchiveError,
                "stdout exceeded 1 bytes",
            ):
                archive_outputs._remote_sha256(
                    "gdrive:test/object",
                    expected_size=1,
                    rclone_path=str(executable),
                )

    @unittest.skipUnless(shutil.which("zstd"), "zstd is required")
    def test_staged_archive_must_match_closed_bundle_member_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot = self._closed_bundle_snapshot(root)
            forged_parent = root / "forged"
            forged_parent.mkdir()
            forged_bundle = forged_parent / snapshot.root.name
            shutil.copytree(snapshot.root, forged_bundle)
            evidence = forged_bundle / "evidence"
            evidence_mode = stat.S_IMODE(evidence.stat().st_mode)
            evidence.chmod(0o644)
            evidence.write_bytes(b"forged")
            evidence.chmod(evidence_mode)
            forged_snapshot = archive_outputs.build_snapshot(forged_bundle)
            archive = root / "bundle.tar.zst"
            archive_outputs._create_archive(
                forged_snapshot,
                archive,
                zstd_path=shutil.which("zstd") or "zstd",
            )
            manifest, sidecar = self._producer_sidecars(
                root,
                archive,
                snapshot,
            )

            with self.assertRaisesRegex(
                campaign_archive.OpenEcologyArchiveError,
                "digest mismatch",
            ):
                campaign_archive._validate_producer_objects(
                    snapshot,
                    (archive, manifest, sidecar),
                )

    @unittest.skipUnless(shutil.which("zstd"), "zstd is required")
    def test_staged_archive_rejects_non_tar_and_unsafe_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot = self._closed_bundle_snapshot(root)
            payloads: dict[str, bytes] = {"non-tar": b"not a tar stream"}
            tar_payload = io.BytesIO()
            with tarfile.open(fileobj=tar_payload, mode="w:") as archive:
                member = tarfile.TarInfo("../escape")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
            payloads["unsafe-path"] = tar_payload.getvalue()

            for name, payload in payloads.items():
                with self.subTest(name=name):
                    case = root / name
                    case.mkdir()
                    archive = case / "bundle.tar.zst"
                    with archive.open("wb") as output:
                        subprocess.run(
                            (shutil.which("zstd") or "zstd", "-q", "-c"),
                            input=payload,
                            stdout=output,
                            check=True,
                        )
                    manifest, sidecar = self._producer_sidecars(
                        case,
                        archive,
                        snapshot,
                    )
                    with self.assertRaises(
                        campaign_archive.OpenEcologyArchiveError,
                    ):
                        campaign_archive._validate_producer_objects(
                            snapshot,
                            (archive, manifest, sidecar),
                        )

    def test_archive_creation_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "payload").write_bytes(b"source")
            snapshot = archive_outputs.build_snapshot(source)
            destination = root / "archive.tar.zst"
            destination.write_bytes(b"competitor")

            with self.assertRaisesRegex(
                archive_outputs.ArchiveError,
                "refusing to overwrite",
            ):
                archive_outputs._create_archive(
                    snapshot,
                    destination,
                    zstd_path="must-not-run",
                )

            self.assertEqual(destination.read_bytes(), b"competitor")

    def test_descriptor_bound_upload_never_reads_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "artifact"
            original = b"original-authoritative-bytes"
            replacement = b"replacement-bytes"
            local.write_bytes(original)
            metadata = local.stat()
            owned = archive_outputs._capture_owned_file(
                local,
                archive_outputs._FileIdentity(
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                ),
            )
            marker = root / "upload-started"
            proceed = root / "proceed"
            remote = root / "remote-object"
            fake_rclone = root / "rclone"
            fake_rclone.write_text(
                "\n".join(
                    (
                        "#!/usr/bin/env python3",
                        "import pathlib",
                        "import sys",
                        "import time",
                        f"marker = pathlib.Path({str(marker)!r})",
                        f"proceed = pathlib.Path({str(proceed)!r})",
                        f"remote = pathlib.Path({str(remote)!r})",
                        "marker.write_text('ready')",
                        "while not proceed.exists():",
                        "    time.sleep(0.005)",
                        "remote.write_bytes(sys.stdin.buffer.read())",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            fake_rclone.chmod(0o755)

            def replace_after_child_inherits_descriptor() -> None:
                deadline = time.monotonic() + 2.0
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                local.unlink()
                local.write_bytes(replacement)
                proceed.write_text("go", encoding="utf-8")

            racer = threading.Thread(
                target=replace_after_child_inherits_descriptor,
                daemon=True,
            )
            racer.start()
            with self.assertRaisesRegex(
                archive_outputs.ArchiveError,
                "(?:open owned evidence changed|owned evidence identity changed)",
            ):
                archive_outputs._upload_owned_file(
                    owned,
                    "gdrive:test/artifact",
                    rclone_path=str(fake_rclone),
                )
            racer.join(timeout=2.0)

            self.assertFalse(racer.is_alive())
            self.assertEqual(local.read_bytes(), replacement)
            self.assertEqual(remote.read_bytes(), original)

    def test_pretransfer_replacement_is_rejected_before_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "artifact"
            local.write_bytes(b"original")
            metadata = local.stat()
            owned = archive_outputs._capture_owned_file(
                local,
                archive_outputs._FileIdentity(
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                ),
            )
            local.unlink()
            local.write_bytes(b"replacement")
            listing = subprocess.CompletedProcess(
                args=("rclone", "lsf"),
                returncode=0,
                stdout="",
                stderr="",
            )

            with (
                patch.object(archive_outputs, "_run_rclone", return_value=listing),
                patch.object(archive_outputs, "_upload_owned_file") as upload,
                self.assertRaisesRegex(
                    archive_outputs.ArchiveError,
                    "(?:open owned evidence changed|owned evidence identity changed)",
                ),
            ):
                archive_outputs._upload_and_verify(
                    (owned,),
                    remote="gdrive:test",
                    remote_subdir="archive",
                )

            upload.assert_not_called()
            self.assertEqual(local.read_bytes(), b"replacement")

    def test_execute_never_unlinks_racing_archive_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            output = base / "output"
            source.mkdir()
            (source / "payload").write_bytes(b"source")
            options = self._archive_options(source, output)
            destination = output / "bundle.tar.zst"

            def lose_publication(*_: object, **__: object) -> object:
                destination.write_bytes(b"competitor")
                raise archive_outputs.ArchiveError("publication race")

            with (
                patch.object(
                    archive_outputs,
                    "_resolve_program_authority",
                    return_value=("unused-zstd", None),
                ),
                patch.object(
                    archive_outputs,
                    "_create_archive",
                    side_effect=lose_publication,
                ),
            ):
                with self.assertRaisesRegex(
                    archive_outputs.ArchiveError,
                    "publication race",
                ):
                    archive_outputs.execute(options)

            self.assertEqual(destination.read_bytes(), b"competitor")

    def test_exception_cleanup_does_not_unlink_replacement_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            output = base / "output"
            source.mkdir()
            (source / "payload").write_bytes(b"source")
            options = self._archive_options(source, output)
            destination = output / "bundle.tar.zst"

            def create_owned_archive(
                _snapshot: object,
                path: Path,
                **_: object,
            ) -> archive_outputs._FileIdentity:
                path.write_bytes(b"owned")
                metadata = path.stat()
                return archive_outputs._FileIdentity(
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                )

            def replace_then_fail(path: Path, **_: object) -> None:
                path.unlink()
                path.write_bytes(b"competitor")
                raise archive_outputs.ArchiveError("verification race")

            with (
                patch.object(
                    archive_outputs,
                    "_resolve_program_authority",
                    return_value=("unused-zstd", None),
                ),
                patch.object(
                    archive_outputs,
                    "_create_archive",
                    side_effect=create_owned_archive,
                ),
                patch.object(
                    archive_outputs,
                    "_verify_zstd_archive",
                    side_effect=replace_then_fail,
                ),
            ):
                with self.assertRaisesRegex(
                    archive_outputs.ArchiveError,
                    "verification race",
                ):
                    archive_outputs.execute(options)

            self.assertEqual(destination.read_bytes(), b"competitor")

    def test_manifest_publication_race_preserves_competitor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            output = base / "output"
            source.mkdir()
            (source / "payload").write_bytes(b"source")
            options = self._archive_options(source, output)
            archive_path = output / "bundle.tar.zst"
            manifest_path = output / "bundle.tar.zst.manifest.json"

            def create_owned_archive(
                _snapshot: object,
                destination: Path,
                **_: object,
            ) -> archive_outputs._FileIdentity:
                destination.write_bytes(b"owned archive")
                metadata = destination.stat()
                return archive_outputs._FileIdentity(
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                )

            def lose_manifest(destination: Path, _payload: bytes) -> object:
                destination.write_bytes(b"competitor manifest")
                raise archive_outputs.ArchiveError("manifest race")

            with (
                patch.object(
                    archive_outputs,
                    "_resolve_program_authority",
                    return_value=("unused-zstd", None),
                ),
                patch.object(
                    archive_outputs,
                    "_create_archive",
                    side_effect=create_owned_archive,
                ),
                patch.object(archive_outputs, "_verify_zstd_archive"),
                patch.object(
                    archive_outputs,
                    "_assert_snapshot_unchanged",
                ),
                patch.object(
                    archive_outputs,
                    "_write_exclusive",
                    side_effect=lose_manifest,
                ),
            ):
                with self.assertRaisesRegex(
                    archive_outputs.ArchiveError,
                    "manifest race",
                ):
                    archive_outputs.execute(options)

            self.assertEqual(archive_path.read_bytes(), b"owned archive")
            self.assertEqual(manifest_path.read_bytes(), b"competitor manifest")

    @staticmethod
    def _successful_parent_with_lingering_descendant(
        marker: Path,
    ) -> tuple[str, ...]:
        child_code = (
            "import pathlib,time\n"
            "time.sleep(0.5)\n"
            f"pathlib.Path({str(marker)!r}).write_text('bad')\n"
        )
        parent_code = (
            "import subprocess,sys\n"
            f"subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
        )
        return (sys.executable, "-c", parent_code)

    @staticmethod
    def _closed_bundle_snapshot(
        root: Path,
    ) -> storage.ClosedBundleSnapshot:
        active = root / "active"
        bundle = root / "bundle-1"
        active.mkdir()
        bundle.mkdir()
        (active / "writer").write_bytes(b"active")
        (bundle / "evidence").write_bytes(b"sealed")
        with patch.object(
            storage,
            "_descriptor_disk_usage",
            return_value=(10**18, 10**18),
        ):
            return storage.seal_closed_bundle(
                active,
                bundle,
                campaign_id="campaign-1",
                bundle_id="bundle-1",
                source_git_sha="a" * 40,
                source_manifest_sha256="b" * 64,
                limits=storage.CampaignStorageLimits(
                    max_campaign_bytes=1024 * 1024,
                    max_entries=100,
                ),
            )

    @staticmethod
    def _producer_sidecars(
        root: Path,
        archive: Path,
        snapshot: storage.ClosedBundleSnapshot,
    ) -> tuple[Path, Path]:
        manifest = root / f"{archive.name}.manifest.json"
        sidecar = root / f"{archive.name}.sha256"
        manifest.write_bytes(storage.canonical_json_bytes(snapshot.producer_manifest))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        return manifest, sidecar

    @staticmethod
    def _archive_options(
        source: Path,
        output: Path,
    ) -> archive_outputs.ArchiveOptions:
        return archive_outputs.ArchiveOptions(
            input_dir=source,
            output_dir=output,
            archive_name="bundle.tar.zst",
            remote=archive_outputs.DEFAULT_REMOTE,
            remote_subdir=archive_outputs.DEFAULT_REMOTE_SUBDIR,
            dry_run=False,
            archive_only=True,
            prune_after_verify=False,
        )


if __name__ == "__main__":
    unittest.main()
