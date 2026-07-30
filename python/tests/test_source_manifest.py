from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from evolution_sim.io.source_manifest import (
    SourceManifestError,
    source_file_hash_manifest,
)


class SourceManifestTests(unittest.TestCase):
    def test_manifest_is_torch_free_canonical_and_content_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "python" / "evolution_sim"
            package.mkdir(parents=True)
            source = package / "example.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            (root / "package.json").write_text("{}\n", encoding="utf-8")
            (root / "requirements-mind-ml.txt").write_text(
                "torch\n",
                encoding="utf-8",
            )

            first = source_file_hash_manifest(root)
            second = source_file_hash_manifest(root)
            self.assertEqual(first, second)
            self.assertEqual(first["file_count"], 3)
            self.assertEqual(
                list(first["files"]),
                [
                    "package.json",
                    "python/evolution_sim/example.py",
                    "requirements-mind-ml.txt",
                ],
            )

            source.write_text("VALUE = 2\n", encoding="utf-8")
            changed = source_file_hash_manifest(root)
            self.assertNotEqual(
                first["aggregate_sha256"],
                changed["aggregate_sha256"],
            )

    def test_manifest_fails_closed_without_runtime_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(SourceManifestError, "found no files"):
                source_file_hash_manifest(temporary)


if __name__ == "__main__":
    unittest.main()
