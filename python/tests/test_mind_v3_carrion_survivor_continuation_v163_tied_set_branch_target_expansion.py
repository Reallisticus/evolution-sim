from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind.carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy import (
    run_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy,
)
from evolution_sim.mind.carrion_survivor_continuation_v163_tied_set_branch_target_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v163_tied_set_branch_target_expansion,
)

try:
    from python.tests.test_mind_v3_carrion_survivor_continuation_v159_scorer_readiness import (
        _write_report_with_exact_digest,
    )
    from python.tests.test_mind_v3_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy import (
        _write_tied_collapsed_v161_inputs,
    )
except ModuleNotFoundError:  # pragma: no cover - unittest discovery fallback.
    from test_mind_v3_carrion_survivor_continuation_v159_scorer_readiness import (
        _write_report_with_exact_digest,
    )
    from test_mind_v3_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy import (
        _write_tied_collapsed_v161_inputs,
    )

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV163TiedSetBranchTargetExpansionTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v163-tied-set-branch-target-expansion"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v163_tied_set_branch_target_expansion"
            ),
        )

    def test_source_invalid_v162_classification_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, evidence_path = _write_ready_v162_inputs(tmpdir)
            v162 = json.loads(paths["v162_report"].read_text(encoding="utf-8"))
            v162["classification"]["primary"] = "unexpected"
            _write_report_with_exact_digest(paths["v162_report"], v162)

            result = run_carrion_survivor_continuation_v163_tied_set_branch_target_expansion(
                v162_report_path=paths["v162_report"],
                trajectory_paths=[evidence_path],
                output_path=paths["v163_report"],
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v162_unexpected_classification",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            "source_invalid_closed_no_live_ab",
        )
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_plan_only_selects_tied_points_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, evidence_path = _write_ready_v162_inputs(tmpdir)

            result = run_carrion_survivor_continuation_v163_tied_set_branch_target_expansion(
                v162_report_path=paths["v162_report"],
                trajectory_paths=[evidence_path],
                output_path=paths["v163_report"],
                strict_broad_seeds=(5, 13),
                attempt_branch_replay=False,
            )

        selection = result["selection_plan"]
        materialization = result["branch_materialization"]
        lifecycle = result["lifecycle_proof"]
        self.assertTrue(result["source_validation"]["passed"])
        self.assertEqual(selection["selected_branch_point_count"], 2)
        self.assertEqual(
            selection["selected_tied_set_counts"],
            {"stay|eat": 2},
        )
        self.assertFalse(materialization["passed"])
        self.assertEqual(
            result["classification"]["primary"],
            "exact_branch_materialization_blocked_no_training",
        )
        self.assertTrue(lifecycle["diagnostics_only"])
        self.assertFalse(lifecycle["runtime_artifact_created"])
        self.assertFalse(lifecycle["live_ab_allowed"])
        self.assertFalse(lifecycle["promotion_authorized"])
        self.assertFalse(lifecycle["runtime_action_selection_changed"])

    def test_cli_writes_source_invalid_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "v162-invalid.json"
            output_path = Path(tmpdir) / "v163-report.json"
            _write_report_with_exact_digest(
                report_path,
                {
                    "schema_version": "wrong",
                    "policy": "wrong",
                    "classification": {"primary": "wrong", "labels": ["wrong"]},
                    "runtime_action_selection_changed": False,
                    "runtime_artifact_created": False,
                    "live_ab_allowed": False,
                    "promotion_authorized": False,
                    "runtime_promotion_allowed": False,
                },
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v163_tied_set_branch_target_expansion",
                    "--v162-report",
                    str(report_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v163_tied_set_branch_target_expansion_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION,
        )
        self.assertEqual(
            written["classification"]["primary"],
            "source_invalid_closed_no_live_ab",
        )
        self.assertIn("exact_digest", written)


def _write_ready_v162_inputs(tmpdir: str) -> tuple[dict[str, Path], Path]:
    paths, evidence_path = _write_tied_collapsed_v161_inputs(tmpdir)
    run_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy(
        v160_report_path=paths["v160_report"],
        v160_artifact_path=paths["v160_artifact"],
        v161_report_path=paths["v161_report"],
        trajectory_paths=[evidence_path],
        output_path=paths["v162_report"],
    )
    paths["v163_report"] = Path(tmpdir) / "v163-report.json"
    return paths, evidence_path


if __name__ == "__main__":
    unittest.main()
