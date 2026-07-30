from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prove_open_ecology_capture_noninterference as proof_script


REQUIRES_MIND_ML = True

class CaptureNoninterferenceProofScriptTests(unittest.TestCase):
    def test_execute_rejects_a_dirty_checkout_before_running_worlds(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                proof_script,
                "_loaded_repository_roots",
                return_value=frozenset({Path(temporary).resolve()}),
            ),
            patch.object(
                proof_script,
                "_git",
                return_value=" M python/evolution_sim/example.py",
            ),
            patch.object(
                proof_script.open_ecology_selection,
                "build_open_ecology_capture_noninterference_proof",
            ) as build_proof,
            self.assertRaisesRegex(RuntimeError, "clean exact checkout"),
        ):
            proof_script.execute(
                repository_root=Path(temporary),
                output_path=Path(temporary) / "proof.json",
            )
        build_proof.assert_not_called()

    def test_execute_binds_live_commit_and_manifest_and_writes_atomically(
        self,
    ) -> None:
        payload: dict[str, object] = {
            "schema_version": "proof-v1",
            "exact_digest": "a" * 64,
            "cases": [{"case_index": 0}],
            "passed": True,
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                proof_script,
                "_loaded_repository_roots",
                return_value=frozenset({Path(temporary).resolve()}),
            ),
            patch.object(
                proof_script,
                "_git",
                side_effect=["", "b" * 40],
            ),
            patch.object(
                proof_script.recurrent_scale_campaign,
                "source_file_hash_manifest",
                return_value={"aggregate_sha256": "c" * 64},
            ),
            patch.object(
                proof_script.open_ecology_selection,
                "build_open_ecology_capture_noninterference_proof",
                return_value=payload,
            ) as build_proof,
        ):
            output = Path(temporary) / "evidence" / "proof.json"
            result = proof_script.execute(
                repository_root=Path(temporary),
                output_path=output,
            )
            observed = json.loads(output.read_text(encoding="utf-8"))
            undeclared = [
                path.name
                for path in output.parent.iterdir()
                if path.name != output.name
            ]

        self.assertEqual(result, payload)
        self.assertEqual(observed, payload)
        self.assertEqual(undeclared, [])
        build_proof.assert_called_once_with(
            source_commit="b" * 40,
            source_manifest_sha256="c" * 64,
        )

    def test_execute_rejects_clean_root_that_differs_from_loaded_source(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                proof_script,
                "_loaded_repository_roots",
                return_value=frozenset({Path("/different/source/root")}),
            ),
            patch.object(proof_script, "_git") as git,
            patch.object(
                proof_script.open_ecology_selection,
                "build_open_ecology_capture_noninterference_proof",
            ) as build_proof,
            self.assertRaisesRegex(RuntimeError, "imports do not match"),
        ):
            proof_script.execute(
                repository_root=Path(temporary),
                output_path=Path(temporary) / "proof.json",
            )
        git.assert_not_called()
        build_proof.assert_not_called()


if __name__ == "__main__":
    unittest.main()
