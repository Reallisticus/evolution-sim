from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    run_carrion_survivor_continuation_action_value_target_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    EXPECTED_V158_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V159_SCORER_READINESS_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v159_scorer_readiness,
)
from evolution_sim.mind.provenance import stable_payload_digest

try:
    from python.tests.test_mind_v3_carrion_survivor_continuation_action_value_target_dataset import (
        _load_jsonl,
        _write_v154_to_v157_inputs,
    )
except ModuleNotFoundError:  # pragma: no cover - unittest discovery fallback.
    from test_mind_v3_carrion_survivor_continuation_action_value_target_dataset import (
        _load_jsonl,
        _write_v154_to_v157_inputs,
    )

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV159ScorerReadinessTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v159-scorer-readiness"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v159_scorer_readiness"
            ),
        )

    def test_source_validation_requires_v158_expected_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v154_to_v158_inputs(tmpdir, mode="multi_safe")
            v158 = json.loads(paths["v158_report"].read_text(encoding="utf-8"))
            v158["classification"]["primary"] = "unexpected"
            _write_report_with_exact_digest(paths["v158_report"], v158)

            result = run_carrion_survivor_continuation_v159_scorer_readiness(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                v156_report_path=paths["v156_report"],
                v157_report_path=paths["v157_report"],
                v158_report_path=paths["v158_report"],
                v158_dataset_path=paths["v158_dataset"],
                output_path=paths["v159_report"],
                min_near_exact_mask_comparator_coverage_share=0.0,
                min_near_exact_mask_shadow_safe_hit_share=0.0,
                max_dominant_proposed_action_share=1.0,
                min_exact_shadow_minus_best_trivial_coverage_share=0.0,
            )

        self.assertIn(
            "v158_unexpected_classification",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["source_validation"]["observed_v158_classification"],
            "unexpected",
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v159_scorer_readiness_"
                "source_invalid_closed_no_training"
            ),
        )
        self.assertFalse(
            result["route_recommendation"][
                "future_opt_in_training_diagnostic_recommended"
            ]
        )
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["runtime_promotion_allowed"])

    def test_digest_checks_reject_mutated_v158_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v154_to_v158_inputs(tmpdir, mode="multi_safe")
            rows = _load_jsonl(paths["v158_dataset"])
            rows[0]["safe_action_set"] = ["eat"]
            _write_jsonl(paths["v158_dataset"], rows)

            result = run_carrion_survivor_continuation_v159_scorer_readiness(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                v156_report_path=paths["v156_report"],
                v157_report_path=paths["v157_report"],
                v158_report_path=paths["v158_report"],
                v158_dataset_path=paths["v158_dataset"],
                output_path=paths["v159_report"],
            )

        self.assertIn(
            "v158_dataset_digest_mismatch",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v159_scorer_readiness_"
                "source_invalid_closed_no_training"
            ),
        )

    def test_leakage_rejection_uses_digest_consistent_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v154_to_v158_inputs(tmpdir, mode="multi_safe")
            rows = _load_jsonl(paths["v158_dataset"])
            rows[0]["trainable_public_features"]["seed"] = 13
            _rewrite_v158_dataset_and_report(paths, rows)

            result = run_carrion_survivor_continuation_v159_scorer_readiness(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                v156_report_path=paths["v156_report"],
                v157_report_path=paths["v157_report"],
                v158_report_path=paths["v158_report"],
                v158_dataset_path=paths["v158_dataset"],
                output_path=paths["v159_report"],
                min_near_exact_mask_comparator_coverage_share=0.0,
                min_near_exact_mask_shadow_safe_hit_share=0.0,
                max_dominant_proposed_action_share=1.0,
                min_exact_shadow_minus_best_trivial_coverage_share=0.0,
            )

        self.assertTrue(result["source_validation"]["passed"])
        self.assertFalse(result["leakage_scan"]["passed"])
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v159_scorer_readiness_"
                "leakage_failed_closed_no_training"
            ),
        )

    def test_action_support_collapse_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v154_to_v158_inputs(tmpdir, mode="multi_safe")
            rows = _load_jsonl(paths["v158_dataset"])
            rows = [_force_unique_safe_action(row, "eat") for row in rows]
            _rewrite_v158_dataset_and_report(paths, rows)

            result = run_carrion_survivor_continuation_v159_scorer_readiness(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                v156_report_path=paths["v156_report"],
                v157_report_path=paths["v157_report"],
                v158_report_path=paths["v158_report"],
                v158_dataset_path=paths["v158_dataset"],
                output_path=paths["v159_report"],
                min_near_exact_mask_comparator_coverage_share=0.0,
                min_near_exact_mask_shadow_safe_hit_share=0.0,
                max_dominant_proposed_action_share=1.0,
                min_exact_shadow_minus_best_trivial_coverage_share=0.0,
            )

        self.assertTrue(result["source_validation"]["passed"])
        self.assertFalse(result["action_support"]["action_support_non_collapsed"])
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v159_scorer_readiness_"
                "action_support_collapsed_closed_no_training"
            ),
        )

    def test_cli_writes_parseable_recommendation_report_under_explicit_thresholds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v154_to_v158_inputs(tmpdir, mode="multi_safe")
            base_row = _load_jsonl(paths["v158_dataset"])[0]
            stay_row = _force_unique_safe_action(base_row, "stay")
            eat_row = _force_unique_safe_action(base_row, "eat")
            eat_row["trainable_public_features"] = dict(
                eat_row["trainable_public_features"]
            )
            eat_row["trainable_public_features"]["public_bucket"] = 1
            _rewrite_v158_dataset_and_report(paths, [stay_row, eat_row])
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v159_scorer_readiness",
                    "--v154-report",
                    str(paths["v154_report"]),
                    "--v154-dataset",
                    str(paths["v154_dataset"]),
                    "--v155-report",
                    str(paths["v155_report"]),
                    "--v156-report",
                    str(paths["v156_report"]),
                    "--v157-report",
                    str(paths["v157_report"]),
                    "--v158-report",
                    str(paths["v158_report"]),
                    "--v158-dataset",
                    str(paths["v158_dataset"]),
                    "--output",
                    str(paths["v159_report"]),
                    "--min-near-exact-mask-shadow-safe-hit-share",
                    "0",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v159_report"].read_text(encoding="utf-8"))
            rows = _load_jsonl(paths["v158_dataset"])

        self.assertIn(
            "future_opt_in_training_diagnostic_recommended=True",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V159_SCORER_READINESS_SCHEMA_VERSION,
        )
        self.assertEqual(
            written["source_validation"]["observed_v158_classification"],
            EXPECTED_V158_CLASSIFICATION,
        )
        self.assertTrue(
            written["route_recommendation"][
                "future_opt_in_training_diagnostic_recommended"
            ]
        )
        self.assertEqual(stable_payload_digest(rows), written["dataset_digest"])
        exact_payload = dict(written)
        exact_payload.pop("exact_digest", None)
        self.assertEqual(stable_payload_digest(exact_payload), written["exact_digest"])
        self.assertFalse(written["training_ran"])
        self.assertFalse(written["artifact_created"])
        self.assertFalse(written["runtime_action_selection_changed"])


def _write_v154_to_v158_inputs(tmpdir: str, *, mode: str) -> dict[str, Path]:
    paths = _write_v154_to_v157_inputs(tmpdir, mode=mode)
    paths["v159_report"] = Path(tmpdir) / "v159-report.json"
    run_carrion_survivor_continuation_action_value_target_dataset(
        v154_report_path=paths["v154_report"],
        v154_dataset_path=paths["v154_dataset"],
        v155_report_path=paths["v155_report"],
        v156_report_path=paths["v156_report"],
        v157_report_path=paths["v157_report"],
        output_path=paths["v158_report"],
        target_dataset_output_path=paths["v158_dataset"],
    )
    return paths


def _rewrite_v158_dataset_and_report(
    paths: dict[str, Path],
    rows: list[dict[str, object]],
) -> None:
    _write_jsonl(paths["v158_dataset"], rows)
    v158 = json.loads(paths["v158_report"].read_text(encoding="utf-8"))
    dataset_digest = stable_payload_digest(rows)
    v158["dataset"]["dataset_digest"] = dataset_digest
    v158["dataset"]["action_value_target_row_count"] = len(rows)
    v158["dataset"]["group_count"] = len(rows)
    v158["target_build_validation"]["target_dataset_row_count"] = len(rows)
    _write_report_with_exact_digest(paths["v158_report"], v158)


def _write_report_with_exact_digest(path: Path, report: dict[str, object]) -> None:
    payload = dict(report)
    payload.pop("exact_digest", None)
    payload["exact_digest"] = stable_payload_digest(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _force_unique_safe_action(
    row: dict[str, object],
    action: str,
) -> dict[str, object]:
    payload = deepcopy(row)
    payload["safe_action_set"] = [action]
    payload["target_classification"] = "unique_robust_winner"
    payload["robust_winner_action"] = action
    for target in payload["action_value_targets"]:
        target_action = target["action"]
        safe = target_action == action
        target["safe_target"] = safe
        target["robust_safe_action"] = safe
        if target.get("public_mask") is True:
            target["target_available"] = True
            target["score_target"] = 2.0 if safe else 0.0
            target["value_target"] = 2.0 if safe else 0.0
            target["safe_run_count"] = 2 if safe else 0
            target["safe_share"] = 1.0 if safe else 0.0
            target["continuation_run_count"] = max(
                int(target.get("continuation_run_count", 0)),
                2,
            )
    return payload


if __name__ == "__main__":
    unittest.main()
