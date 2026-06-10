from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v165_preterminal_target_dataset_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v166_source_split_action_value_scorer import (
    M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION,
    source_split_diagnostics,
)
from evolution_sim.mind.carrion_survivor_continuation_v167_source_split_failure_autopsy import (
    EXPECTED_V166_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION,
    V166_SHADOW_READY_CLASSIFICATION,
    run_carrion_survivor_continuation_v167_source_split_failure_autopsy,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV167SourceSplitFailureAutopsyTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v167-source-split-failure-autopsy"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v167_source_split_failure_autopsy"
            ),
        )

    def test_v166_digest_or_artifact_mismatch_closes_source_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_missing_rows()
            paths, report, artifact = _write_v166_inputs(tmpdir, rows=rows)

            result = run_carrion_survivor_continuation_v167_source_split_failure_autopsy(
                v166_report_path=paths["v166_report"],
                v166_artifact_path=paths["artifact"],
                v165_dataset_path=paths["dataset"],
                output_path=paths["v167_report"],
                expected_v166_exact_digest="wrong",
                expected_v166_artifact_digest=artifact["exact_digest"],
                expected_v165_dataset_digest=stable_payload_digest(rows),
                expected_zero_safe_hit_preterminal_source_seeds=[5, 13],
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v166_unexpected_exact_digest",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v167_source_split_failure_autopsy_"
                "source_invalid_closed_no_shadow"
            ),
        )
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_unexpected_shadow_ready_input_closes_without_autopsy_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _aliasing_rows()
            paths, report, artifact = _write_v166_inputs(
                tmpdir,
                rows=rows,
                classification=V166_SHADOW_READY_CLASSIFICATION,
            )

            result = run_carrion_survivor_continuation_v167_source_split_failure_autopsy(
                v166_report_path=paths["v166_report"],
                v166_artifact_path=paths["artifact"],
                v165_dataset_path=paths["dataset"],
                output_path=paths["v167_report"],
                expected_v166_exact_digest=report["exact_digest"],
                expected_v166_artifact_digest=artifact["exact_digest"],
                expected_v165_dataset_digest=stable_payload_digest(rows),
                expected_zero_safe_hit_preterminal_source_seeds=[5, 13],
            )

        self.assertTrue(result["source_validation"]["passed"])
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v167_source_split_failure_autopsy_"
                "unexpected_shadow_ready_input_closed_no_autopsy"
            ),
        )
        self.assertFalse(result["route_recommendation"]["shadow_eval_recommended"])

    def test_classifies_source_action_support_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _support_missing_rows()
            paths, report, artifact = _write_v166_inputs(tmpdir, rows=rows)

            result = run_carrion_survivor_continuation_v167_source_split_failure_autopsy(
                v166_report_path=paths["v166_report"],
                v166_artifact_path=paths["artifact"],
                v165_dataset_path=paths["dataset"],
                output_path=paths["v167_report"],
                expected_v166_exact_digest=report["exact_digest"],
                expected_v166_artifact_digest=artifact["exact_digest"],
                expected_v165_dataset_digest=stable_payload_digest(rows),
                expected_zero_safe_hit_preterminal_source_seeds=[5, 13],
            )

        self.assertEqual(
            result["failure_mode_summary"]["primary_failure_mode"],
            "source_action_support_missing",
        )
        self.assertEqual(
            result["route_recommendation"]["recommended_next_route"],
            "targeted_source_support_expansion",
        )

    def test_classifies_nearest_neighbor_feature_aliasing_and_reports_row_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _aliasing_rows()
            paths, report, artifact = _write_v166_inputs(tmpdir, rows=rows)

            result = run_carrion_survivor_continuation_v167_source_split_failure_autopsy(
                v166_report_path=paths["v166_report"],
                v166_artifact_path=paths["artifact"],
                v165_dataset_path=paths["dataset"],
                output_path=paths["v167_report"],
                expected_v166_exact_digest=report["exact_digest"],
                expected_v166_artifact_digest=artifact["exact_digest"],
                expected_v165_dataset_digest=stable_payload_digest(rows),
                expected_zero_safe_hit_preterminal_source_seeds=[5, 13],
            )

        self.assertEqual(
            result["failure_mode_summary"]["primary_failure_mode"],
            "nearest_neighbor_feature_aliasing",
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v167_source_split_failure_autopsy_"
                "nearest_neighbor_feature_aliasing_closed_feature_contract_expansion"
            ),
        )
        self.assertEqual(
            result["route_recommendation"]["recommended_next_route"],
            "public_feature_contract_expansion",
        )
        first = result["preterminal_row_autopsy"][0]
        for key in (
            "row_index",
            "source_seed",
            "safe_action_set",
            "predicted_action",
            "safe_hit",
            "nearest_neighbor_row_index",
            "nearest_neighbor_source_seed",
            "nearest_neighbor_source_group",
            "nearest_neighbor_safe_action_set",
            "distance_to_predicted_neighbor",
            "nearest_safe_support_neighbor_row_index",
            "nearest_safe_support_source_group",
            "distance_to_nearest_safe_support_neighbor",
            "rank_of_first_safe_support_neighbor",
            "distance_gap_predicted_to_nearest_safe_support",
        ):
            self.assertIn(key, first)
        self.assertIn("3", first["top_k_safe_support"])
        self.assertTrue(
            result["top_k_diagnostics"]["top_k"]["5"][
                "analysis_only_no_k_tuning_recommended"
            ]
        )
        self.assertFalse(result["route_recommendation"]["k_tuning_recommended"])

    def test_classifies_mixed_support_and_feature_aliasing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _mixed_rows()
            paths, report, artifact = _write_v166_inputs(tmpdir, rows=rows)

            result = run_carrion_survivor_continuation_v167_source_split_failure_autopsy(
                v166_report_path=paths["v166_report"],
                v166_artifact_path=paths["artifact"],
                v165_dataset_path=paths["dataset"],
                output_path=paths["v167_report"],
                expected_v166_exact_digest=report["exact_digest"],
                expected_v166_artifact_digest=artifact["exact_digest"],
                expected_v165_dataset_digest=stable_payload_digest(rows),
                expected_zero_safe_hit_preterminal_source_seeds=[5, 13],
            )

        self.assertEqual(
            result["failure_mode_summary"]["primary_failure_mode"],
            "mixed_source_support_and_feature_aliasing",
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v167_source_split_failure_autopsy_"
                "mixed_source_support_and_feature_aliasing_closed_source_expansion"
            ),
        )

    def test_cli_writes_report_with_exact_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _aliasing_rows()
            paths, report, artifact = _write_v166_inputs(tmpdir, rows=rows)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v167_source_split_failure_autopsy",
                    "--v166-report",
                    str(paths["v166_report"]),
                    "--v166-artifact",
                    str(paths["artifact"]),
                    "--v165-dataset",
                    str(paths["dataset"]),
                    "--output",
                    str(paths["v167_report"]),
                    "--expected-v166-exact-digest",
                    report["exact_digest"],
                    "--expected-v166-artifact-digest",
                    artifact["exact_digest"],
                    "--expected-v165-dataset-digest",
                    stable_payload_digest(rows),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v167_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v167_source_split_failure_autopsy_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION,
        )
        self.assertEqual(written["exact_digest"], _digest_without_exact(written))
        self.assertFalse(written["artifact_created"])
        self.assertFalse(written["runtime_action_selection_changed"])


def _support_missing_rows() -> list[dict[str, object]]:
    return [
        _row(seed=5, safe_action="stay", top_action="stay", bucket=0.0),
        _row(seed=13, safe_action="eat", top_action="eat", bucket=0.1),
    ]


def _aliasing_rows() -> list[dict[str, object]]:
    return [
        _row(seed=5, safe_action="stay", top_action="stay", bucket=0.0),
        _row(seed=13, safe_action="eat", top_action="eat", bucket=0.1),
        _row(schema="base", safe_action="stay", top_action="stay", bucket=10.0),
        _row(schema="base", safe_action="eat", top_action="eat", bucket=10.1),
    ]


def _mixed_rows() -> list[dict[str, object]]:
    return [
        _row(seed=5, safe_action="stay", top_action="stay", bucket=0.0),
        _row(seed=13, safe_action="eat", top_action="eat", bucket=0.1),
        _row(schema="base", safe_action="stay", top_action="stay", bucket=10.0),
    ]


def _row(
    *,
    safe_action: str,
    top_action: str,
    bucket: float,
    schema: str = "v165",
    seed: int | None = None,
) -> dict[str, object]:
    mask = {action: action in {"stay", "eat"} for action in ACTION_NAMES}
    targets = []
    for action in ACTION_NAMES:
        available = action in {"stay", "eat"}
        value = 10.0 if action == top_action else 1.0
        targets.append(
            {
                "action": action,
                "public_mask": bool(mask[action]),
                "target_available": available,
                "safe_target": action == safe_action,
                "robust_safe_action": action == safe_action,
                "value_target": value if available else None,
                "score_target": value if available else None,
            }
        )
    row: dict[str, object] = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
            if schema == "v165"
            else M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION
        ),
        "feature_policy_id": "fixture_public_features",
        "trainable_public_features": {
            "public_observation": {"bucket": bucket},
            "action_mask": mask,
        },
        "public_action_mask": mask,
        "action_value_targets": targets,
        "safe_action_set": [safe_action],
        "target_classification": "unique_robust_winner",
        "robust_winner_action": safe_action,
    }
    if schema == "v165":
        if seed is None:
            raise AssertionError("v165 rows require a seed")
        row["row_origin"] = "v165_preterminal_branch_target"
        row["candidate_action_value_targets"] = [
            target for target in targets if target["target_available"] is True
        ]
        row["target_local_primary_support_actions"] = [safe_action]
        row["terminal_population_guard_support_actions"] = [safe_action]
        row["metadata"] = {
            "seed": seed,
            "fixture": "broad",
            "branch_tick": 10,
            "agent_id": seed,
            "branch_id": f"branch-{seed}",
            "source_path": "source.jsonl",
            "source_record_digest": f"digest-{seed}",
            "source_seed_is_support_provenance_not_future_promotion_holdout": True,
            "runtime_requested_action_used_as_scorer_input": False,
            "future_outcomes_used_as_trainable_input": False,
        }
    return row


def _write_v166_inputs(
    tmpdir: str,
    *,
    rows: list[dict[str, object]],
    classification: str = EXPECTED_V166_CLASSIFICATION,
) -> tuple[dict[str, Path], dict[str, object], dict[str, object]]:
    root = Path(tmpdir)
    paths = {
        "dataset": root / "v165.jsonl",
        "v166_report": root / "v166.json",
        "artifact": root / "v166-artifact.json",
        "v167_report": root / "v167.json",
    }
    paths["dataset"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    artifact = _v166_artifact(rows)
    report = _v166_report(rows, classification=classification, artifact=artifact)
    paths["artifact"].write_text(
        json.dumps(artifact, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["v166_report"].write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths, report, artifact


def _v166_report(
    rows: list[dict[str, object]],
    *,
    classification: str,
    artifact: dict[str, object],
) -> dict[str, object]:
    split = source_split_diagnostics(rows=rows)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_POLICY,
        "contract": {
            "diagnostics_only": True,
            "runtime_artifact_created": False,
            "runtime_policy_integration_allowed": False,
            "shadow_live_ab_allowed": False,
            "live_runtime_override_allowed": False,
            "threshold_tuning_recommended": False,
            "runtime_action_selection_changed": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
        },
        "artifact": {"artifact_digest": artifact["exact_digest"]},
        "source_split_diagnostics": split,
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": {
            "threshold_tuning_recommended": False,
            "live_ab_allowed": False,
            "runtime_policy_integration_allowed": False,
            "runtime_override_path_allowed": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "runtime_action_selection_changed": False,
        },
        "dataset_digest": stable_payload_digest(rows),
        "runtime_artifact_created": False,
        "training_ran": True,
        "threshold_tuning_recommended": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "runtime_action_selection_changed": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = _digest_without_exact(report)
    return report


def _v166_artifact(rows: list[dict[str, object]]) -> dict[str, object]:
    artifact = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_POLICY,
        "contract": {
            "diagnostics_only": True,
            "runtime_artifact": False,
            "runtime_policy_integration_allowed": False,
            "runtime_action_selection_changed": False,
        },
        "source": {"dataset_digest": stable_payload_digest(rows)},
    }
    artifact["exact_digest"] = _digest_without_exact(artifact)
    return artifact


def _digest_without_exact(payload: dict[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(without_digest, sort_keys=True, allow_nan=False))
    )


if __name__ == "__main__":
    unittest.main()
