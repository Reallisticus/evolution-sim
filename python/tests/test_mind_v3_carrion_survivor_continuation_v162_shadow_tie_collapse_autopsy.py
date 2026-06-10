from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind.carrion_survivor_continuation_v161_shadow_eval import (
    run_carrion_survivor_continuation_v161_shadow_eval,
)
from evolution_sim.mind.carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy import (
    M3_CARRION_SURVIVOR_CONTINUATION_V162_SHADOW_TIE_COLLAPSE_AUTOPSY_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy,
)
from evolution_sim.mind.provenance import stable_payload_digest

try:
    from python.tests.test_mind_v3_carrion_survivor_continuation_v159_scorer_readiness import (
        _write_report_with_exact_digest,
    )
    from python.tests.test_mind_v3_carrion_survivor_continuation_v161_shadow_eval import (
        _write_ready_v160_inputs,
        _write_shadow_evidence,
    )
except ModuleNotFoundError:  # pragma: no cover - unittest discovery fallback.
    from test_mind_v3_carrion_survivor_continuation_v159_scorer_readiness import (
        _write_report_with_exact_digest,
    )
    from test_mind_v3_carrion_survivor_continuation_v161_shadow_eval import (
        _write_ready_v160_inputs,
        _write_shadow_evidence,
    )

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV162ShadowTieCollapseAutopsyTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v162-shadow-tie-collapse-autopsy"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy"
            ),
        )

    def test_source_invalid_v161_report_closes_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, evidence_path = _write_tied_collapsed_v161_inputs(tmpdir)
            v161 = json.loads(paths["v161_report"].read_text(encoding="utf-8"))
            v161["source_validation"]["passed"] = False
            _write_report_with_exact_digest(paths["v161_report"], v161)

            result = run_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy(
                v160_report_path=paths["v160_report"],
                v160_artifact_path=paths["v160_artifact"],
                v161_report_path=paths["v161_report"],
                trajectory_paths=[evidence_path],
                output_path=paths["v162_report"],
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v161_source_validation_failed",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy_"
                "source_invalid_closed_no_live_ab"
            ),
        )
        self.assertFalse(result["diagnostic_autopsy_ran"])
        self.assertEqual(result["shadow_tie_collapse_autopsy"]["prediction_count"], 0)
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_tied_top_value_stay_collapse_is_attributed_without_runtime_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, evidence_path = _write_tied_collapsed_v161_inputs(tmpdir)

            result = run_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy(
                v160_report_path=paths["v160_report"],
                v160_artifact_path=paths["v160_artifact"],
                v161_report_path=paths["v161_report"],
                trajectory_paths=[evidence_path],
                output_path=paths["v162_report"],
            )

        autopsy = result["shadow_tie_collapse_autopsy"]
        tie = autopsy["deterministic_tie_break_attribution"]
        top_sets = autopsy[
            "top_value_tied_candidate_sets_after_public_action_mask_filtering"
        ]
        membership = autopsy["top_value_candidate_membership_counts"]
        comparison = autopsy["comparison_only_would_change_metrics_from_v161"]

        self.assertTrue(result["source_validation"]["passed"])
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy_"
                "tie_collapse_archive_feature_support_blocked_no_live_ab"
            ),
        )
        self.assertEqual(autopsy["predicted_action_counts"], {"stay": 4})
        self.assertEqual(top_sets["counts"], {"stay|eat": 4})
        self.assertEqual(membership, {"stay": 4, "eat": 4})
        self.assertEqual(tie["stay_prediction_count"], 4)
        self.assertEqual(tie["stay_predictions_from_tied_top_value_sets"], 4)
        self.assertEqual(
            tie["stay_tie_break_attributed_share_of_stay_predictions"],
            1.0,
        )
        self.assertTrue(autopsy["set_valued_candidates_noncollapsed_but_broad"])
        self.assertFalse(
            comparison["runtime_requested_action_used_as_scorer_input"]
        )
        self.assertTrue(
            comparison["recomputed_from_v162_autopsy"][
                "matches_v161_predicted_action_counts"
            ]
        )
        self.assertFalse(result["runtime_artifact_created"])
        self.assertFalse(result["live_ab_allowed"])
        self.assertFalse(result["promotion_authorized"])
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_validated_digest_and_lifecycle_fields_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, evidence_path = _write_tied_collapsed_v161_inputs(tmpdir)

            result = run_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy(
                v160_report_path=paths["v160_report"],
                v160_artifact_path=paths["v160_artifact"],
                v161_report_path=paths["v161_report"],
                trajectory_paths=[evidence_path],
                output_path=paths["v162_report"],
            )

        digests = result["validated_source_digests"]
        lifecycle = result["lifecycle_proof"]
        self.assertTrue(digests["v161_collapsed_classification_validated"])
        self.assertTrue(
            digests["v160_report_exact_digest_validation"]["passed"]
        )
        self.assertTrue(
            digests["v160_artifact_exact_digest_validation"]["passed"]
        )
        self.assertTrue(
            digests["v161_report_exact_digest_validation"]["passed"]
        )
        self.assertFalse(lifecycle["runtime_action_selection_changed"])
        self.assertFalse(lifecycle["runtime_artifact_created"])
        self.assertFalse(lifecycle["live_ab_allowed"])
        self.assertFalse(lifecycle["promotion_authorized"])

    def test_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, evidence_path = _write_tied_collapsed_v161_inputs(tmpdir)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy",
                    "--v160-report",
                    str(paths["v160_report"]),
                    "--v160-artifact",
                    str(paths["v160_artifact"]),
                    "--v161-report",
                    str(paths["v161_report"]),
                    "--trajectory",
                    str(evidence_path),
                    "--output",
                    str(paths["v162_report"]),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v162_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V162_SHADOW_TIE_COLLAPSE_AUTOPSY_SCHEMA_VERSION,
        )
        self.assertEqual(written["source_validation"]["passed"], True)
        self.assertIn("exact_digest", written)


def _write_tied_collapsed_v161_inputs(tmpdir: str) -> tuple[dict[str, Path], Path]:
    paths = _write_ready_v160_inputs(tmpdir)
    paths["v162_report"] = Path(tmpdir) / "v162-report.json"
    _rewrite_v160_artifact_with_stay_eat_ties(paths)
    evidence_path = _write_shadow_evidence(tmpdir, paths)
    run_carrion_survivor_continuation_v161_shadow_eval(
        v160_report_path=paths["v160_report"],
        v160_artifact_path=paths["v160_artifact"],
        trajectory_paths=[evidence_path],
        output_path=paths["v161_report"],
    )
    return paths, evidence_path


def _rewrite_v160_artifact_with_stay_eat_ties(paths: dict[str, Path]) -> None:
    artifact = json.loads(paths["v160_artifact"].read_text(encoding="utf-8"))
    for row in artifact["training_rows"]:
        for target in row["action_value_targets"]:
            action = target["action"]
            if action in {"stay", "eat"}:
                target["public_mask"] = True
                target["target_available"] = True
                target["safe_target"] = True
                target["robust_safe_action"] = True
                target["value_target"] = 2.0
            elif target.get("public_mask") is True:
                target["target_available"] = True
                target["safe_target"] = False
                target["robust_safe_action"] = False
                target["value_target"] = 0.0
    _write_report_with_exact_digest(paths["v160_artifact"], artifact)
    artifact = json.loads(paths["v160_artifact"].read_text(encoding="utf-8"))
    v160 = json.loads(paths["v160_report"].read_text(encoding="utf-8"))
    v160["artifact"]["artifact_digest"] = stable_payload_digest(artifact)
    _write_report_with_exact_digest(paths["v160_report"], v160)


if __name__ == "__main__":
    unittest.main()
