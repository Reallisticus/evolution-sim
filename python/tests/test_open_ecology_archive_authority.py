from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from evolution_sim.io import open_ecology_archive_authority as authority


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "archive_evolution_outputs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "open_ecology_pinned_archive_producer",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
archive_outputs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = archive_outputs
SPEC.loader.exec_module(archive_outputs)


class RemoteHelperAuthorityTests(unittest.TestCase):
    def test_transitive_process_group_helper_is_pinned(self) -> None:
        self.assertIn(
            "python/evolution_sim/io/open_ecology_bounded_subprocess.py",
            authority.REMOTE_HELPER_PATHS,
        )


class SshConnectionIdentityTests(unittest.TestCase):
    def test_parses_authenticated_host_key_address_and_public_key_method(self) -> None:
        observed = authority.parse_ssh_connection_identity(
            b"\n".join(
                (
                    b"debug1: unrelated",
                    (
                        b"debug1: Server host key: ssh-ed25519 "
                        b"SHA256:t7jdHhzxiAjpWPe1Su1syEIcvufekR/gENfpD4qxeAk"
                    ),
                    (
                        b"Authenticated to 192.0.2.153 "
                        b'([192.0.2.153]:22) using "publickey".'
                    ),
                    b"",
                )
            )
        )

        self.assertEqual(
            observed,
            authority.SshConnectionIdentity(
                authenticated_host="192.0.2.153",
                address="192.0.2.153",
                port=22,
                authentication="publickey",
                host_key=(
                    "ssh-ed25519 SHA256:t7jdHhzxiAjpWPe1Su1syEIcvufekR/gENfpD4qxeAk"
                ),
            ),
        )

    def test_rejects_ambiguous_or_non_public_key_endpoint(self) -> None:
        host_key = (
            b"debug1: Server host key: ssh-ed25519 "
            b"SHA256:t7jdHhzxiAjpWPe1Su1syEIcvufekR/gENfpD4qxeAk"
        )
        password = b'Authenticated to host ([192.0.2.1]:22) using "password".'
        with self.assertRaisesRegex(
            authority.ArchiveAuthorityError,
            "publickey",
        ):
            authority.parse_ssh_connection_identity(b"\n".join((host_key, password)))
        with self.assertRaisesRegex(
            authority.ArchiveAuthorityError,
            "unambiguous",
        ):
            authority.parse_ssh_connection_identity(
                b"\n".join((host_key, host_key, password))
            )


class RcloneCredentialAuthorityTests(unittest.TestCase):
    def test_private_single_link_current_uid_config_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory).resolve() / "rclone.conf"
            config.write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
            config.chmod(0o600)
            pin = authority.measure_canonical_file(
                config,
                executable=False,
                require_nonempty=True,
                private_credential=True,
            )

            authority.verify_pinned_file(
                pin,
                executable=False,
                require_nonempty=True,
            )

            self.assertTrue(pin.private_credential)

    def test_group_or_world_accessible_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory).resolve() / "rclone.conf"
            config.write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
            config.chmod(0o640)
            pin = authority.FilePin(
                path=str(config),
                sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                private_credential=True,
            )

            with self.assertRaisesRegex(
                authority.ArchiveAuthorityError,
                "current-UID-owned private regular file",
            ):
                authority.verify_pinned_file(
                    pin,
                    executable=False,
                    require_nonempty=True,
                )

    def test_multiply_linked_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = root / "rclone.conf"
            alias = root / "credential-alias"
            config.write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
            config.chmod(0o600)
            os.link(config, alias)
            pin = authority.FilePin(
                path=str(config),
                sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                private_credential=True,
            )

            with self.assertRaisesRegex(
                authority.ArchiveAuthorityError,
                "one hard link",
            ):
                authority.verify_pinned_file(
                    pin,
                    executable=False,
                    require_nonempty=True,
                )

    def test_non_current_uid_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory).resolve() / "rclone.conf"
            config.write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
            config.chmod(0o600)
            pin = authority.FilePin(
                path=str(config),
                sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                private_credential=True,
            )

            with (
                patch.object(authority.os, "getuid", return_value=os.getuid() + 1),
                self.assertRaisesRegex(
                    authority.ArchiveAuthorityError,
                    "current-UID-owned private regular file",
                ),
            ):
                authority.verify_pinned_file(
                    pin,
                    executable=False,
                    require_nonempty=True,
                )


class ArchiveAuthorityPathTests(unittest.TestCase):
    def test_ancestor_lstat_error_is_converted_to_authority_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ancestor = root / "authority"
            ancestor.mkdir()
            target = ancestor / "seal.json"
            target.write_text("{}\n", encoding="utf-8")
            real_lstat = os.lstat

            def fail_one_ancestor(candidate: os.PathLike[str] | str):
                if Path(candidate) == ancestor:
                    raise PermissionError("hostile ancestor")
                return real_lstat(candidate)

            with (
                patch.object(authority.Path, "resolve", return_value=target),
                patch.object(
                    authority.os,
                    "lstat",
                    side_effect=fail_one_ancestor,
                ),
                self.assertRaisesRegex(
                    authority.ArchiveAuthorityError,
                    "cannot inspect pinned file ancestor",
                ),
            ):
                authority._canonical_absolute_path(target, field="pinned file")


class OpenEcologyArchiveProducerAuthorityTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("zstd"), "zstd is required")
    def test_explicit_zstd_pin_ignores_hostile_path(self) -> None:
        zstd = Path(shutil.which("zstd") or "").resolve(strict=True)
        zstd_sha256 = hashlib.sha256(zstd.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source = base / "source"
            source.mkdir()
            (source / "payload.bin").write_bytes(b"open-ecology\n")
            snapshot = archive_outputs.build_snapshot(source)
            archive = base / "bundle.tar.zst"
            hostile = base / "hostile"
            hostile.mkdir()
            marker = base / "hostile-zstd-ran"
            hostile_zstd = hostile / "zstd"
            hostile_zstd.write_text(
                f"#!/bin/sh\n/usr/bin/touch {marker}\nexit 91\n",
                encoding="utf-8",
            )
            hostile_zstd.chmod(0o755)

            with patch.dict(os.environ, {"PATH": str(hostile)}):
                archive_outputs._create_archive(
                    snapshot,
                    archive,
                    zstd_path=str(zstd),
                    zstd_sha256=zstd_sha256,
                )
                archive_outputs._verify_zstd_archive(
                    archive,
                    zstd_path=str(zstd),
                    zstd_sha256=zstd_sha256,
                )

            self.assertTrue(archive.is_file())
            self.assertFalse(marker.exists())

    def test_zstd_replacement_after_external_pin_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory).resolve() / "zstd"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            digest = hashlib.sha256(executable.read_bytes()).hexdigest()
            resolved, observed_digest = archive_outputs._resolve_program_authority(
                "zstd",
                explicit_path=str(executable),
                expected_sha256=digest,
            )
            self.assertEqual(resolved, str(executable))
            self.assertEqual(observed_digest, digest)

            executable.write_text("#!/bin/sh\nexit 44\n", encoding="utf-8")
            executable.chmod(0o755)
            with self.assertRaisesRegex(
                archive_outputs.ArchiveError,
                "SHA256 mismatch",
            ):
                archive_outputs._verify_program_pin(executable, digest)


if __name__ == "__main__":
    unittest.main()
