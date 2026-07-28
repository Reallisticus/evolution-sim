from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import PurePosixPath
import unittest
from unittest.mock import patch

from scripts import stream_remote_evolution_archive as remote_archive


class RemoteEvolutionArchiveTests(unittest.TestCase):
    def _options(self, **overrides: object) -> remote_archive.RemoteArchiveOptions:
        values: dict[str, object] = {
            "ssh_target": "trainer-node",
            "remote_repository_root": "/srv/evolution-sim/checkout",
            "remote_input_dir": "/srv/evolution-sim/runs/campaign",
            "remote_staging_dir": "/srv/evolution-sim/archives",
            "archive_name": "20260727T120000Z-campaign.tar.zst",
            "rclone_remote": "gdrive:evolution-sim-backups",
            "rclone_subdir": "archives",
        }
        values.update(overrides)
        return remote_archive.RemoteArchiveOptions(**values)  # type: ignore[arg-type]

    def test_options_reject_shell_like_target_and_unsafe_paths(self) -> None:
        for overrides, message in (
            ({"ssh_target": "-oProxyCommand=bad"}, "ssh target"),
            ({"remote_input_dir": "/"}, "filesystem root"),
            ({"remote_input_dir": "relative"}, "absolute normalized"),
            (
                {"remote_staging_dir": ("/srv/evolution-sim/runs/campaign/staging")},
                "may not be inside",
            ),
            ({"archive_name": "../campaign.tar.zst"}, "conservative filename"),
            ({"rclone_subdir": "../archives"}, "safe relative"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(
                    remote_archive.RemoteArchiveError,
                    message,
                ):
                    self._options(**overrides).validate()

    def test_archive_result_requires_exact_requested_paths(self) -> None:
        paths = (
            PurePosixPath("/staging/a.tar.zst"),
            PurePosixPath("/staging/a.tar.zst.manifest.json"),
            PurePosixPath("/staging/a.tar.zst.sha256"),
        )
        valid = {
            "archive_path": str(paths[0]),
            "manifest_path": str(paths[1]),
            "sidecar_path": str(paths[2]),
            "archive_sha256": "a" * 64,
            "archive_size": 123,
            "status": "local_archive_verified",
        }

        remote_archive._validate_archive_result(valid, paths)
        for field, value in (
            ("archive_path", "/other/a.tar.zst"),
            ("archive_sha256", "not-a-digest"),
            ("archive_size", 0),
            ("status", "unverified"),
        ):
            invalid = dict(valid)
            invalid[field] = value
            with self.subTest(field=field):
                with self.assertRaises(remote_archive.RemoteArchiveError):
                    remote_archive._validate_archive_result(invalid, paths)

    @patch.object(remote_archive, "_rclone_sha256")
    @patch.object(remote_archive, "_stream_remote_path")
    @patch.object(remote_archive, "_assert_destination_absent")
    @patch.object(remote_archive, "_remote_file_evidence")
    @patch.object(remote_archive, "_run_ssh_json")
    def test_execute_streams_three_objects_without_pruning(
        self,
        run_ssh_json,
        remote_file_evidence,
        assert_destination_absent,
        stream_remote_path,
        rclone_sha256,
    ) -> None:
        options = self._options()
        staging = PurePosixPath(options.remote_staging_dir)
        paths = (
            staging / options.archive_name,
            staging / f"{options.archive_name}.manifest.json",
            staging / f"{options.archive_name}.sha256",
        )
        run_ssh_json.return_value = {
            "archive_path": str(paths[0]),
            "manifest_path": str(paths[1]),
            "sidecar_path": str(paths[2]),
            "archive_sha256": "a" * 64,
            "archive_size": 100,
            "status": "local_archive_verified",
        }
        evidence = [
            {"path": str(path), "size": index + 1, "sha256": chr(97 + index) * 64}
            for index, path in enumerate(paths)
        ]
        remote_file_evidence.return_value = evidence
        rclone_sha256.side_effect = [record["sha256"] for record in evidence]

        result = remote_archive.execute(options)

        self.assertEqual(
            result["status"],
            "remote_archive_streamed_and_byte_verified",
        )
        self.assertEqual(len(result["objects"]), 3)
        self.assertFalse(result["remote_input_pruned"])
        self.assertFalse(result["remote_staging_pruned"])
        assert_destination_absent.assert_called_once()
        self.assertEqual(stream_remote_path.call_count, 3)

    @patch.object(remote_archive, "_rclone_sha256", return_value="f" * 64)
    @patch.object(remote_archive, "_stream_remote_path")
    @patch.object(remote_archive, "_assert_destination_absent")
    @patch.object(remote_archive, "_remote_file_evidence")
    @patch.object(remote_archive, "_run_ssh_json")
    def test_execute_fails_closed_on_destination_digest_mismatch(
        self,
        run_ssh_json,
        remote_file_evidence,
        _assert_destination_absent,
        _stream_remote_path,
        _rclone_sha256,
    ) -> None:
        options = self._options()
        staging = PurePosixPath(options.remote_staging_dir)
        paths = (
            staging / options.archive_name,
            staging / f"{options.archive_name}.manifest.json",
            staging / f"{options.archive_name}.sha256",
        )
        run_ssh_json.return_value = {
            "archive_path": str(paths[0]),
            "manifest_path": str(paths[1]),
            "sidecar_path": str(paths[2]),
            "archive_sha256": "a" * 64,
            "archive_size": 100,
            "status": "local_archive_verified",
        }
        remote_file_evidence.return_value = [
            {"path": str(path), "size": 1, "sha256": "a" * 64} for path in paths
        ]

        with self.assertRaisesRegex(
            remote_archive.RemoteArchiveError,
            "destination SHA256 mismatch",
        ):
            remote_archive.execute(options)

    def test_invalid_cli_emits_no_success_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = remote_archive.main(
                [
                    "--ssh-target",
                    "bad target",
                    "--remote-repository-root",
                    "/repo",
                    "--remote-input-dir",
                    "/input",
                    "--remote-staging-dir",
                    "/staging",
                    "--archive-name",
                    "a.tar.zst",
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("failed closed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
