from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import unittest

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

    def test_execute_is_deauthorized_before_starting_any_transport(self) -> None:
        options = self._options()
        with self.assertRaisesRegex(
            remote_archive.RemoteArchiveError,
            "deauthorized",
        ):
            remote_archive.execute(options)
        for retired_name in (
            "_deprecated_execute_implementation",
            "_run",
            "_run_ssh_json",
            "_ssh_command",
            "_stream_remote_path",
        ):
            with self.subTest(retired_name=retired_name):
                self.assertFalse(hasattr(remote_archive, retired_name))

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
