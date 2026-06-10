from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    run_carrion_survivor_continuation_v159_scorer_readiness,
)
from evolution_sim.mind.carrion_survivor_continuation_v160_action_value_scorer import (
    M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v160_action_value_scorer,
)
from evolution_sim.mind.provenance import stable_payload_digest

try:
    from python.tests.test_mind_v3_carrion_survivor_continuation_v159_scorer_readiness import (
        _force_unique_safe_action,
        _load_jsonl,
        _rewrite_v158_dataset_and_report,
        _write_report_with_exact_digest,
        _write_v154_to_v158_inputs,
    )
except ModuleNotFoundError:  # pragma: no cover - unittest discovery fallback.
    from test_mind_v3_carrion_survivor_continuation_v159_scorer_readiness import (
        _force_unique_safe_action,
        _load_jsonl,
        _rewrite_v158_dataset_and_report,
        _write_report_with_exact_digest,
        _write_v154_to_v158_inputs,
    )

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV160ActionValueScorerTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v160-action-value-scorer"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v160_action_value_scorer"
            ),
        )

    def test_source_validation_requires_v159_ready_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_ready_v159_inputs(tmpdir, mode="failing_loo")
            v159 = json.loads(paths["v159_report"].read_text(encoding="utf-8"))
            v159["classification"]["primary"] = "unexpected"
            _write_report_with_exact_digest(paths["v159_report"], v159)

            result = run_carrion_survivor_continuation_v160_action_value_scorer(
                v158_dataset_path=paths["v158_dataset"],
                v159_report_path=paths["v159_report"],
                output_path=paths["v160_report"],
                artifact_output_path=paths["v160_artifact"],
            )

        self.assertIn(
            "v159_unexpected_classification",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v160_action_value_scorer_"
                "source_invalid_closed_no_shadow"
            ),
        )
        self.assertFalse(
            result["route_recommendation"]["future_shadow_evaluation_recommended"]
        )
        self.assertFalse(result["runtime_promotion_allowed"])

    def test_leakage_rejection_uses_digest_consistent_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_ready_v159_inputs(tmpdir, mode="failing_loo")
            rows = _load_jsonl(paths["v158_dataset"])
            rows[0]["trainable_public_features"]["seed"] = 13
            _write_jsonl(paths["v158_dataset"], rows)
            v159 = json.loads(paths["v159_report"].read_text(encoding="utf-8"))
            v159["dataset_digest"] = stable_payload_digest(rows)
            _write_report_with_exact_digest(paths["v159_report"], v159)

            result = run_carrion_survivor_continuation_v160_action_value_scorer(
                v158_dataset_path=paths["v158_dataset"],
                v159_report_path=paths["v159_report"],
                output_path=paths["v160_report"],
                artifact_output_path=paths["v160_artifact"],
            )

        self.assertTrue(result["source_validation"]["passed"])
        self.assertFalse(result["leakage_scan"]["passed"])
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v160_action_value_scorer_"
                "leakage_failed_closed_no_shadow"
            ),
        )

    def test_loo_generalization_failure_closes_and_recommends_archive_expansion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_ready_v159_inputs(tmpdir, mode="failing_loo")

            result = run_carrion_survivor_continuation_v160_action_value_scorer(
                v158_dataset_path=paths["v158_dataset"],
                v159_report_path=paths["v159_report"],
                output_path=paths["v160_report"],
                artifact_output_path=paths["v160_artifact"],
            )
            artifact = json.loads(paths["v160_artifact"].read_text(encoding="utf-8"))

        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v160_action_value_scorer_"
                "loo_generalization_failed_closed_archive_expansion"
            ),
        )
        self.assertEqual(result["leave_one_row_out"]["unsupported_prediction_count"], 0)
        self.assertEqual(result["leave_one_row_out"]["dominant_predicted_action_share"], 0.5)
        self.assertFalse(result["leave_one_row_out"]["safe_hit_margin_floor_passed"])
        self.assertEqual(
            result["route_recommendation"]["recommended_next_route"],
            "expand_carrion_survivor_continuation_archive_before_more_scorer_training",
        )
        self.assertFalse(
            result["route_recommendation"]["future_shadow_evaluation_recommended"]
        )
        self.assertEqual(
            artifact["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertFalse(artifact["contract"]["runtime_artifact"])
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_cli_writes_ready_report_for_controlled_public_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_ready_v159_inputs(tmpdir, mode="passing_loo")
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v160_action_value_scorer",
                    "--v158-dataset",
                    str(paths["v158_dataset"]),
                    "--v159-report",
                    str(paths["v159_report"]),
                    "--output",
                    str(paths["v160_report"]),
                    "--artifact-output",
                    str(paths["v160_artifact"]),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v160_report"].read_text(encoding="utf-8"))
            artifact = json.loads(paths["v160_artifact"].read_text(encoding="utf-8"))
            rows = _load_jsonl(paths["v158_dataset"])

        self.assertIn("future_shadow_evaluation_recommended=True", completed.stdout)
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION,
        )
        self.assertEqual(
            written["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v160_action_value_scorer_"
                "diagnostic_scorer_ready_for_future_shadow_eval"
            ),
        )
        self.assertEqual(written["leave_one_row_out"]["safe_hit_rate"], 1.0)
        self.assertEqual(written["leave_one_row_out"]["dominant_predicted_action_share"], 0.5)
        self.assertEqual(stable_payload_digest(rows), written["dataset_digest"])
        self.assertTrue(written["artifact_created"])
        self.assertFalse(written["runtime_artifact_created"])
        self.assertFalse(written["runtime_action_selection_changed"])
        self.assertEqual(
            artifact["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(len(artifact["training_rows"]), len(rows))


def _write_ready_v159_inputs(tmpdir: str, *, mode: str) -> dict[str, Path]:
    paths = _write_v154_to_v158_inputs(tmpdir, mode="multi_safe")
    paths["v160_report"] = Path(tmpdir) / "v160-report.json"
    paths["v160_artifact"] = Path(tmpdir) / "v160-artifact.json"
    base = _load_jsonl(paths["v158_dataset"])[0]
    if mode == "failing_loo":
        rows = [
            _row_variant(base, "stay", bucket=0, variant=0),
            _row_variant(base, "eat", bucket=1, variant=0),
        ]
    elif mode == "passing_loo":
        rows = [
            _row_variant(base, "stay", bucket=0, variant=0),
            _row_variant(base, "stay", bucket=0, variant=0.01),
            _row_variant(base, "eat", bucket=1, variant=0),
            _row_variant(base, "eat", bucket=1, variant=0.01),
        ]
    else:
        raise AssertionError(mode)
    _rewrite_v158_dataset_and_report(paths, rows)
    run_carrion_survivor_continuation_v159_scorer_readiness(
        v154_report_path=paths["v154_report"],
        v154_dataset_path=paths["v154_dataset"],
        v155_report_path=paths["v155_report"],
        v156_report_path=paths["v156_report"],
        v157_report_path=paths["v157_report"],
        v158_report_path=paths["v158_report"],
        v158_dataset_path=paths["v158_dataset"],
        output_path=paths["v159_report"],
        min_near_exact_mask_shadow_safe_hit_share=0.0,
    )
    return paths


def _row_variant(
    base: dict[str, object],
    safe_action: str,
    *,
    bucket: int,
    variant: float,
) -> dict[str, object]:
    row = _force_unique_safe_action(base, safe_action)
    features = deepcopy(row["trainable_public_features"])
    features["public_bucket"] = bucket
    features["public_variant"] = variant
    row["trainable_public_features"] = features
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
