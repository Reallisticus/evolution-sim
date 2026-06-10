from __future__ import annotations

from copy import deepcopy
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
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v166_source_split_action_value_scorer import (
    EXPECTED_V165_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v166_source_split_action_value_scorer,
    source_split_diagnostics,
    trainable_feature_payloads,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]
STRICT_SEEDS = [5, 13, 19, 29, 37, 41]


class MindV3CarrionSurvivorContinuationV166SourceSplitActionValueScorerTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v166-source-split-action-value-scorer"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v166_source_split_action_value_scorer"
            ),
        )

    def test_v165_digest_or_classification_mismatch_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _passing_rows()
            report = _v165_report(rows, classification="unexpected")
            paths = _write_inputs(tmpdir, rows=rows, report=report)

            result = run_carrion_survivor_continuation_v166_source_split_action_value_scorer(
                v165_report_path=paths["report"],
                v165_dataset_path=paths["dataset"],
                output_path=paths["v166_report"],
                artifact_output_path=paths["artifact"],
                expected_v165_exact_digest="expected-digest",
                expected_v165_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v165_unexpected_classification",
            result["source_validation"]["failures"],
        )
        self.assertIn(
            "v165_unexpected_exact_digest",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v166_source_split_action_value_scorer_"
                "source_invalid_closed_no_shadow"
            ),
        )
        self.assertFalse(
            result["route_recommendation"][
                "future_diagnostics_only_shadow_evaluation_recommended"
            ]
        )

    def test_v165_runtime_authorization_field_true_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _passing_rows()
            report = _v165_report(rows)
            report["contract"]["shadow_live_ab_allowed"] = True
            report["exact_digest"] = _digest_without_exact(report)
            paths = _write_inputs(tmpdir, rows=rows, report=report)

            result = run_carrion_survivor_continuation_v166_source_split_action_value_scorer(
                v165_report_path=paths["report"],
                v165_dataset_path=paths["dataset"],
                output_path=paths["v166_report"],
                artifact_output_path=paths["artifact"],
                expected_v165_exact_digest=report["exact_digest"],
                expected_v165_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v165_lifecycle_runtime_authorization_not_false",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v166_source_split_action_value_scorer_"
                "source_invalid_closed_no_shadow"
            ),
        )

    def test_source_split_excludes_same_v165_provenance_seed_rows(self) -> None:
        rows = [
            _row(seed=5, safe_action="stay", top_action="stay", bucket=0.0),
            _row(seed=5, safe_action="stay", top_action="stay", bucket=0.01),
            _row(seed=13, safe_action="eat", top_action="eat", bucket=0.02),
        ]

        split = source_split_diagnostics(rows=rows)
        seed_five_predictions = [
            prediction
            for prediction in split["predictions"]
            if prediction["source_seed"] == 5
        ]

        self.assertEqual(len(seed_five_predictions), 2)
        for prediction in seed_five_predictions:
            self.assertEqual(prediction["excluded_row_indexes"], [0, 1])
            self.assertNotIn(
                prediction["nearest_neighbor_row_indexes"][0],
                prediction["excluded_row_indexes"],
            )

    def test_trainable_feature_payload_has_no_provenance_private_or_future_leakage(
        self,
    ) -> None:
        row = _row(seed=5, safe_action="stay", top_action="stay", bucket=0.0)
        row["metadata"]["branch_id"] = "branch-seed-5"
        row["metadata"]["source_record_digest"] = "digest-private"
        row["metadata"]["runtime_requested_action"] = "eat"
        row["metadata"]["future_outcome_marker"] = "terminal_extinction"

        payload_text = json.dumps(trainable_feature_payloads([row]), sort_keys=True)

        for forbidden in (
            "seed",
            "fixture",
            "branch",
            "tick",
            "agent",
            "path",
            "digest",
            "provenance",
            "private",
            "future",
            "outcome",
            "runtime_requested_action",
        ):
            self.assertNotIn(forbidden, payload_text)

    def test_floor_classifications_close_as_expected(self) -> None:
        cases = [
            (
                "unsupported",
                [_row(seed=5, safe_action="stay", top_action="stay", bucket=0.0)],
                {},
                (
                    "m3_carrion_survivor_continuation_v166_source_split_action_value_scorer_"
                    "unsupported_predictions_closed_no_shadow"
                ),
            ),
            (
                "dominant",
                [
                    _row(seed=5, safe_action="stay", top_action="stay", bucket=0.0),
                    _row(seed=13, safe_action="stay", top_action="stay", bucket=0.01),
                    _row(schema="base", safe_action="stay", top_action="stay", bucket=1.0),
                    _row(schema="base", safe_action="stay", top_action="stay", bucket=1.01),
                ],
                {},
                (
                    "m3_carrion_survivor_continuation_v166_source_split_action_value_scorer_"
                    "predicted_action_dominance_closed_no_shadow"
                ),
            ),
            (
                "safe_hit_margin",
                [
                    _row(seed=5, safe_action="stay", top_action="stay", bucket=0.0),
                    _row(seed=13, safe_action="eat", top_action="eat", bucket=0.01),
                    _row(schema="base", safe_action="stay", top_action="stay", bucket=1.0),
                    _row(schema="base", safe_action="eat", top_action="eat", bucket=1.01),
                ],
                {"swap_target_pairs": True},
                (
                    "m3_carrion_survivor_continuation_v166_source_split_action_value_scorer_"
                    "source_split_generalization_failed_closed_archive_source_expansion"
                ),
            ),
            (
                "zero_source_seed",
                [
                    _row(seed=5, safe_action="stay", top_action="eat", bucket=0.0),
                    _row(seed=13, safe_action="eat", top_action="eat", bucket=0.01),
                    _row(schema="base", safe_action="stay", top_action="stay", bucket=1.0),
                    _row(schema="base", safe_action="stay", top_action="stay", bucket=1.01),
                ],
                {"min_safe_hit_margin_over_best_trivial": -1.0},
                (
                    "m3_carrion_survivor_continuation_v166_source_split_action_value_scorer_"
                    "preterminal_source_seed_zero_safe_hits_closed_source_expansion"
                ),
            ),
        ]
        for name, rows, options, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                if options.pop("swap_target_pairs", False):
                    rows = _swap_neighbor_targets(rows)
                report = _v165_report(rows)
                paths = _write_inputs(tmpdir, rows=rows, report=report)
                min_margin = options.get(
                    "min_safe_hit_margin_over_best_trivial",
                    0.05,
                )

                result = run_carrion_survivor_continuation_v166_source_split_action_value_scorer(
                    v165_report_path=paths["report"],
                    v165_dataset_path=paths["dataset"],
                    output_path=paths["v166_report"],
                    artifact_output_path=paths["artifact"],
                    expected_v165_exact_digest=report["exact_digest"],
                    expected_v165_dataset_digest=stable_payload_digest(rows),
                    min_safe_hit_margin_over_best_trivial=min_margin,
                )

            self.assertEqual(result["classification"]["primary"], expected)
            self.assertFalse(result["runtime_action_selection_changed"])

    def test_cli_writes_report_and_diagnostics_artifact_with_exact_digests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _passing_rows()
            report = _v165_report(rows)
            paths = _write_inputs(tmpdir, rows=rows, report=report)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v166_source_split_action_value_scorer",
                    "--v165-report",
                    str(paths["report"]),
                    "--v165-dataset",
                    str(paths["dataset"]),
                    "--output",
                    str(paths["v166_report"]),
                    "--artifact-output",
                    str(paths["artifact"]),
                    "--expected-v165-exact-digest",
                    report["exact_digest"],
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
            written = json.loads(paths["v166_report"].read_text(encoding="utf-8"))
            artifact = json.loads(paths["artifact"].read_text(encoding="utf-8"))

        self.assertIn(
            "future_diagnostics_only_shadow_evaluation_recommended=True",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION,
        )
        self.assertEqual(
            artifact["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(written["source_split_diagnostics"]["safe_hit_rate"], 1.0)
        self.assertEqual(
            written["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v166_source_split_action_value_scorer_"
                "diagnostic_scorer_ready_for_future_diagnostics_only_shadow_eval"
            ),
        )
        self.assertEqual(written["exact_digest"], _digest_without_exact(written))
        self.assertEqual(artifact["exact_digest"], _digest_without_exact(artifact))
        self.assertEqual(
            written["artifact"]["artifact_digest"],
            artifact["exact_digest"],
        )
        self.assertFalse(written["runtime_artifact_created"])
        self.assertFalse(written["runtime_action_selection_changed"])


def _passing_rows() -> list[dict[str, object]]:
    return [
        _row(seed=5, safe_action="stay", top_action="stay", bucket=0.0),
        _row(seed=13, safe_action="stay", top_action="stay", bucket=0.01),
        _row(schema="base", safe_action="eat", top_action="eat", bucket=1.0),
        _row(schema="base", safe_action="eat", top_action="eat", bucket=1.01),
    ]


def _row(
    *,
    safe_action: str,
    top_action: str,
    bucket: float,
    schema: str = "v165",
    seed: int | None = None,
) -> dict[str, object]:
    public_actions = {"stay", "eat"}
    mask = {action: action in public_actions for action in ACTION_NAMES}
    values = {"stay": 1.0, "eat": 1.0}
    values[top_action] = 10.0
    targets = []
    for action in ACTION_NAMES:
        available = action in public_actions
        targets.append(
            {
                "action": action,
                "public_mask": bool(mask[action]),
                "target_available": available,
                "safe_target": action == safe_action,
                "robust_safe_action": action == safe_action,
                "value_target": values[action] if available else None,
                "score_target": values[action] if available else None,
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
            raise AssertionError("v165 fixture rows require seed")
        row["row_origin"] = "v165_preterminal_branch_target"
        row["candidate_action_value_targets"] = [
            target for target in targets if target["target_available"] is True
        ]
        row["target_local_primary_support_actions"] = [safe_action]
        row["terminal_population_guard_support_actions"] = [safe_action]
        row["metadata"] = {
            "metadata_schema_version": (
                "m3_carrion_survivor_continuation_v165_preterminal_target_metadata_v1"
            ),
            "source": "v164_preterminal_tied_set_branch_replay",
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


def _swap_neighbor_targets(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    swapped = deepcopy(rows)
    _set_top_action(swapped[0], "stay")
    _set_top_action(swapped[1], "eat")
    _set_top_action(swapped[2], "stay")
    _set_top_action(swapped[3], "eat")
    return swapped


def _set_top_action(row: dict[str, object], action: str) -> None:
    for target in row["action_value_targets"]:
        if target["target_available"] is True:
            target["value_target"] = 10.0 if target["action"] == action else 1.0
            target["score_target"] = target["value_target"]


def _v165_report(
    rows: list[dict[str, object]],
    *,
    classification: str = EXPECTED_V165_CLASSIFICATION,
) -> dict[str, object]:
    dataset_digest = stable_payload_digest(rows)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_POLICY,
        "contract": {
            "diagnostics_only": True,
            "training_authorized": False,
            "training_ran": False,
            "runtime_artifact_created": False,
            "runtime_policy_integration_allowed": False,
            "live_ab_allowed": False,
            "runtime_override_path_created": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "threshold_tuning_recommended": False,
            "runtime_action_selection_changed": False,
        },
        "source_validation": {"passed": True},
        "row_build_validation": {"passed": True},
        "leakage_scan": {"passed": True, "failure_count": 0, "failures": []},
        "dataset": {
            "combined_row_count": len(rows),
            "dataset_digest": dataset_digest,
        },
        "future_evaluation_policy": {
            "leave_source_seed_out_diagnostics_required": True,
            "new_held_out_broad_seeds_required": True,
            "v164_support_provenance_seed_exclusion_required_for_future_promotion": True,
            "v164_strict_broad_seeds_become_support_provenance_seeds_not_heldout": True,
            "support_provenance_seeds": STRICT_SEEDS,
            "disallowed_future_promotion_heldout_seed_reuse": STRICT_SEEDS,
            "promotion_authorized": False,
            "live_ab_allowed": False,
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": {
            "training_authorized": False,
            "live_ab_allowed": False,
            "promotion_authorized": False,
            "runtime_policy_integration_allowed": False,
            "runtime_action_selection_changed": False,
            "threshold_tuning_recommended": False,
        },
        "dataset_created": True,
        "training_ran": False,
        "training_authorized": False,
        "artifact_created": False,
        "runtime_artifact_created": False,
        "live_ab_allowed": False,
        "live_ab_ran": False,
        "runtime_override_path_created": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "threshold_tuning_recommended": False,
        "runtime_action_selection_changed": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = _digest_without_exact(report)
    return report


def _write_inputs(
    tmpdir: str,
    *,
    rows: list[dict[str, object]],
    report: dict[str, object],
) -> dict[str, Path]:
    root = Path(tmpdir)
    paths = {
        "report": root / "v165.json",
        "dataset": root / "v165.jsonl",
        "v166_report": root / "v166.json",
        "artifact": root / "v166-artifact.json",
    }
    paths["report"].write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["dataset"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return paths


def _digest_without_exact(payload: dict[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(without_digest, sort_keys=True, allow_nan=False))
    )


if __name__ == "__main__":
    unittest.main()
