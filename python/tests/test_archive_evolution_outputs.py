from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "archive_evolution_outputs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "archive_evolution_outputs",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
archive_outputs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = archive_outputs
SPEC.loader.exec_module(archive_outputs)


class ArchiveEvolutionOutputsTests(unittest.TestCase):
    def test_manifest_is_sorted_and_hashes_every_regular_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "evidence"
            root.mkdir()
            (root / "z.json").write_text('{"z":1}\n', encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "a.txt").write_text("alpha\n", encoding="utf-8")

            first = archive_outputs.build_snapshot(root)
            second = archive_outputs.build_snapshot(root)

        self.assertEqual(
            archive_outputs.canonical_manifest_bytes(first.manifest),
            archive_outputs.canonical_manifest_bytes(second.manifest),
        )
        records = first.manifest["entries"]
        self.assertEqual(
            [record["path"] for record in records],
            ["nested", "nested/a.txt", "z.json"],
        )
        file_records = {
            record["path"]: record for record in records if record["type"] == "file"
        }
        self.assertEqual(
            file_records["nested/a.txt"]["sha256"],
            hashlib.sha256(b"alpha\n").hexdigest(),
        )
        self.assertEqual(
            file_records["z.json"]["sha256"],
            hashlib.sha256(b'{"z":1}\n').hexdigest(),
        )

    def test_symlink_outside_input_fails_closed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "evidence"
            root.mkdir()
            outside = base / "outside.txt"
            outside.write_text("private\n", encoding="utf-8")
            (root / "escape").symlink_to("../outside.txt")

            with self.assertRaisesRegex(
                archive_outputs.ArchiveError,
                "symlink escapes",
            ):
                archive_outputs.build_snapshot(root)

    def test_absolute_symlink_is_rejected_even_when_target_is_inside_input(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "evidence"
            root.mkdir()
            target = root / "target.txt"
            target.write_text("payload\n", encoding="utf-8")
            (root / "absolute-alias").symlink_to(target.resolve())

            with self.assertRaisesRegex(
                archive_outputs.ArchiveError,
                "absolute symlink",
            ):
                archive_outputs.build_snapshot(root)

    @unittest.skipUnless(shutil.which("zstd"), "zstd is required")
    def test_tar_zstd_is_deterministic_and_does_not_follow_internal_symlink(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "evidence"
            root.mkdir()
            (root / "payload.txt").write_text("payload\n", encoding="utf-8")
            (root / "alias").symlink_to("payload.txt")
            snapshot = archive_outputs.build_snapshot(root)
            first = base / "first.tar.zst"
            second = base / "second.tar.zst"

            archive_outputs._create_archive(snapshot, first)
            archive_outputs._create_archive(snapshot, second)
            first_digest = archive_outputs._sha256_path(first)
            second_digest = archive_outputs._sha256_path(second)
            tar_bytes = subprocess.run(
                ["zstd", "-dc", str(first)],
                check=True,
                stdout=subprocess.PIPE,
            ).stdout

        self.assertEqual(first_digest, second_digest)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
            alias = archive.getmember("evidence/alias")
            payload = archive.getmember("evidence/payload.txt")
        self.assertTrue(alias.issym())
        self.assertEqual(alias.linkname, "payload.txt")
        self.assertTrue(payload.isfile())

    def test_dry_run_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "evidence"
            output = base / "archives"
            root.mkdir()
            (root / "report.json").write_text("{}\n", encoding="utf-8")
            options = self._options(
                root=root,
                output=output,
                dry_run=True,
            )

            result = archive_outputs.execute(options)

            self.assertEqual(result["status"], "dry_run_validated")
            self.assertFalse(output.exists())
            self.assertTrue((root / "report.json").exists())

    def test_non_dry_run_requires_explicit_output_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "evidence"
            root.mkdir()
            (root / "report.json").write_text("{}\n", encoding="utf-8")
            options = archive_outputs.ArchiveOptions(
                input_dir=root,
                output_dir=None,
                archive_name="test-evidence.tar.zst",
                remote=archive_outputs.DEFAULT_REMOTE,
                remote_subdir=archive_outputs.DEFAULT_REMOTE_SUBDIR,
                dry_run=False,
                archive_only=True,
                prune_after_verify=False,
            )

            with self.assertRaisesRegex(
                archive_outputs.ArchiveError,
                "--output-dir is required",
            ):
                archive_outputs.execute(options)

    @unittest.skipUnless(shutil.which("zstd"), "zstd is required")
    def test_archive_only_creates_three_verified_local_objects(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "evidence"
            output = base / "archives"
            root.mkdir()
            (root / "report.json").write_text('{"ok":true}\n', encoding="utf-8")
            options = self._options(
                root=root,
                output=output,
                archive_only=True,
            )

            result = archive_outputs.execute(options)

            archive_path = Path(result["archive_path"])
            manifest_path = Path(result["manifest_path"])
            sidecar_path = Path(result["sidecar_path"])
            sidecar_digest = sidecar_path.read_text(encoding="utf-8").split()[0]

            self.assertEqual(result["status"], "local_archive_verified")
            self.assertTrue(archive_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(sidecar_path.exists())
            self.assertEqual(
                sidecar_digest,
                archive_outputs._sha256_path(archive_path),
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["file_count"], 1)

    def test_prune_request_fails_closed_before_archive_or_upload(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "evidence"
            output = base / "archives"
            root.mkdir()
            evidence = root / "report.json"
            evidence.write_text("{}\n", encoding="utf-8")
            options = self._options(
                root=root,
                output=output,
                prune_after_verify=True,
            )

            with patch.object(archive_outputs, "_upload_and_verify") as upload:
                with self.assertRaisesRegex(
                    archive_outputs.ArchiveError,
                    "automatic pruning could delete a racing replacement",
                ):
                    archive_outputs.execute(options)

            upload.assert_not_called()
            self.assertTrue(evidence.exists())
            self.assertEqual(evidence.read_text(encoding="utf-8"), "{}\n")
            self.assertFalse(output.exists())

    def test_prune_request_never_removes_nested_input(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "evidence"
            output = base / "archives"
            root.mkdir()
            (root / "nested").mkdir()
            (root / "nested" / "report.json").write_text("{}\n", encoding="utf-8")
            options = self._options(
                root=root,
                output=output,
                prune_after_verify=True,
            )

            with self.assertRaisesRegex(
                archive_outputs.ArchiveError,
                "--prune-after-verify is disabled",
            ):
                archive_outputs.execute(options)

            self.assertEqual(
                (root / "nested" / "report.json").read_text(encoding="utf-8"),
                "{}\n",
            )

    def test_manifest_drift_blocks_archival_continuation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "evidence"
            root.mkdir()
            original = root / "original.json"
            original.write_text("{}\n", encoding="utf-8")
            snapshot = archive_outputs.build_snapshot(root)
            added = root / "added-after-archive.json"
            added.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                archive_outputs.ArchiveError,
                "input directory changed",
            ):
                archive_outputs._assert_snapshot_unchanged(snapshot)

            self.assertTrue(original.exists())
            self.assertTrue(added.exists())

    @staticmethod
    def _options(
        *,
        root: Path,
        output: Path,
        dry_run: bool = False,
        archive_only: bool = False,
        prune_after_verify: bool = False,
    ):
        return archive_outputs.ArchiveOptions(
            input_dir=root,
            output_dir=output,
            archive_name="test-evidence.tar.zst",
            remote=archive_outputs.DEFAULT_REMOTE,
            remote_subdir=archive_outputs.DEFAULT_REMOTE_SUBDIR,
            dry_run=dry_run,
            archive_only=archive_only,
            prune_after_verify=prune_after_verify,
        )


if __name__ == "__main__":
    unittest.main()
