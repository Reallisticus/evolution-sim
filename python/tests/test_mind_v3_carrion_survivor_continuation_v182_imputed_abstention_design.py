from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v181_v180_failure_response as v181,
    carrion_survivor_continuation_v182_imputed_abstention_design as v182,
)
from tests.test_mind_v3_carrion_survivor_continuation_v181_v180_failure_response import (
    _write_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV182ImputedAbstentionDesignTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v182-imputed-abstention-design"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v182_imputed_abstention_design"
            ),
        )

    def test_valid_v181_v180_sources_write_design_report_without_training(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v181_inputs(tmpdir)

            report = v182.run_carrion_survivor_continuation_v182_imputed_abstention_design(
                v181_report_path=paths["v181_report"],
                v180_report_path=paths["v180_report"],
                v180_artifact_path=paths["artifact"],
                transition_dataset_path=paths["dataset"],
                output_path=paths["root"] / "v182-report.json",
                expected_v181_report_exact_digest=paths["v181_exact_digest"],
                expected_v180_report_exact_digest=paths["v180_exact_digest"],
                expected_v180_artifact_digest=paths["artifact_digest"],
                expected_dataset_digest=paths["dataset_digest"],
                run_shadow_evaluation=False,
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["slice_2_training_consumed"])
        self.assertFalse(report["shadow_eval_ran"])
        self.assertEqual(
            report["classification"]["primary"],
            v182.V182_SHADOW_SKIPPED_CLASSIFICATION,
        )
        self.assertEqual(
            report["route"]["next_route"],
            "run_v182_shadow_design_eval_before_any_slice_2_training",
        )
        self.assertTrue(
            report["contract"][
                "override_abstains_when_any_valid_action_score_is_imputed"
            ]
        )
        self.assertEqual(report["exact_digest"], v182._digest_without_exact(report))

    def test_v181_digest_mismatch_fails_closed_without_shadow_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v181_inputs(tmpdir)

            report = v182.run_carrion_survivor_continuation_v182_imputed_abstention_design(
                v181_report_path=paths["v181_report"],
                v180_report_path=paths["v180_report"],
                v180_artifact_path=paths["artifact"],
                transition_dataset_path=paths["dataset"],
                output_path=paths["root"] / "v182-report.json",
                expected_v181_report_exact_digest="wrong",
                expected_v180_report_exact_digest=paths["v180_exact_digest"],
                expected_v180_artifact_digest=paths["artifact_digest"],
                expected_dataset_digest=paths["dataset_digest"],
                run_shadow_evaluation=True,
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v181_report_exact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertFalse(report["shadow_eval_ran"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["promotion_authorized"])
        self.assertEqual(
            report["classification"]["primary"],
            v182.V182_SOURCE_INVALID_CLASSIFICATION,
        )

    def test_route_expands_support_when_carrion_observed_coverage_zero(
        self,
    ) -> None:
        route = v182._route(
            source_validation={"passed": True},
            shadow_evaluation={"ran": True},
            design_diagnostics={
                "strict_observed_support_leaves_carrion_coverage_zero": True,
                "broad_regressions_remain": False,
            },
        )

        self.assertEqual(
            route["next_route"],
            "exact_transition_support_expansion_before_any_slice_2_training",
        )
        self.assertFalse(route["slice_2_training_consumed"])
        self.assertFalse(route["training_authorized"])

    def test_cli_writes_parseable_report_and_prints_lifecycle_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v181_inputs(tmpdir)
            output_path = paths["root"] / "v182-report.json"

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v182_imputed_abstention_design",
                    "--v181-report",
                    str(paths["v181_report"]),
                    "--v180-report",
                    str(paths["v180_report"]),
                    "--v180-artifact",
                    str(paths["artifact"]),
                    "--transition-dataset",
                    str(paths["dataset"]),
                    "--output",
                    str(output_path),
                    "--expected-v181-report-exact-digest",
                    paths["v181_exact_digest"],
                    "--expected-v180-report-exact-digest",
                    paths["v180_exact_digest"],
                    "--expected-v180-artifact-digest",
                    paths["artifact_digest"],
                    "--expected-dataset-digest",
                    paths["dataset_digest"],
                    "--skip-shadow-evaluation",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("training_artifact_created=False", result.stdout)
        self.assertIn("runtime_artifact_created=False", result.stdout)
        self.assertIn("runtime_action_selection_changed=False", result.stdout)
        self.assertIn("promotion_authorized=False", result.stdout)
        self.assertIn("slice_2_training_consumed=False", result.stdout)
        self.assertEqual(
            report["classification"]["primary"],
            v182.V182_SHADOW_SKIPPED_CLASSIFICATION,
        )


def _write_v181_inputs(tmpdir: str) -> dict[str, object]:
    paths = _write_inputs(tmpdir)
    v181_report_path = paths["root"] / "v181-report.json"
    v181_report = v181.run_carrion_survivor_continuation_v181_v180_failure_response(
        v180_report_path=paths["v180_report"],
        v180_artifact_path=paths["artifact"],
        transition_dataset_path=paths["dataset"],
        output_path=v181_report_path,
        expected_v180_report_exact_digest=paths["v180_exact_digest"],
        expected_v180_artifact_digest=paths["artifact_digest"],
        expected_dataset_digest=paths["dataset_digest"],
        run_trace_replay=False,
    )
    return {
        **paths,
        "v181_report": v181_report_path,
        "v181_exact_digest": v181_report["exact_digest"],
    }


if __name__ == "__main__":
    unittest.main()
