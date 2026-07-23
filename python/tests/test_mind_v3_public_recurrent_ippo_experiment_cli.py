from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.mind.provenance import stable_payload_digest
    from evolution_sim.mind.recurrent_seed_registry import (
        CANONICAL_SEED_REGISTRY_SHA256,
    )
    from evolution_sim.cli.mind_v3_public_recurrent_ippo_experiment import main


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class MindV3PublicRecurrentIPPOExperimentCLITests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    def test_requires_explicit_development_run_flag(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--development-run"):
            main([])

    def test_package_entrypoint_exists(self) -> None:
        root = Path(__file__).resolve().parents[2]
        package = json.loads((root / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:public-recurrent-ippo-development-experiment"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_public_recurrent_ippo_experiment"
            ),
        )

    def test_artifact_requires_explicit_source_pin_before_training(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--source-commit"):
            main(["--development-run", "--artifact", "unused.json"])
        with self.assertRaisesRegex(SystemExit, "checked-out Git HEAD"):
            main(
                [
                    "--development-run",
                    "--artifact",
                    "unused.json",
                    "--source-commit",
                    "0" * 40,
                ]
            )

    def test_minimal_real_experiment_writes_closed_exact_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            result = main(
                [
                    "--development-run",
                    "--updates",
                    "1",
                    "--worlds-per-update",
                    "1",
                    "--rollout-ticks",
                    "3",
                    "--scenarios",
                    "carrion_only",
                    "--learner-seed",
                    "404337389",
                    "--device",
                    "cpu",
                    "--encoder-size",
                    "16",
                    "--hidden-size",
                    "16",
                    "--update-epochs",
                    "1",
                    "--sequence-minibatch-size",
                    "32",
                    "--tbptt-steps",
                    "2",
                    "--burn-in-steps",
                    "1",
                    "--report",
                    str(report_path),
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(
            report["seed_registry_sha256"],
            CANONICAL_SEED_REGISTRY_SHA256,
        )
        self.assertEqual(report["training"]["total_worlds"], 1)
        self.assertGreater(report["training"]["total_transitions"], 0)
        self.assertTrue(report["development_experiment"])
        self.assertFalse(report["campaign_training_slice_consumed"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        observed = report.pop("exact_digest")
        self.assertEqual(observed, stable_payload_digest(report))


if __name__ == "__main__":
    unittest.main()
