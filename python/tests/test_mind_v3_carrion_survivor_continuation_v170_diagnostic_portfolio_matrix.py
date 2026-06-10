from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.carrion_sequence_context_comparator_support_closeout import (
    M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY,
    M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    EXPECTED_V154_SUPPORT_READY_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v165_preterminal_target_dataset_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v167_source_split_failure_autopsy import (
    M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v168_public_feature_contract_probe import (
    M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v169_public_temporal_context_probe import (
    M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v170_diagnostic_portfolio_matrix import (
    EXPECTED_V155_CLASSIFICATION,
    EXPECTED_V169_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_SCHEMA_VERSION,
    V155_REFERENCE_CONFLICTING_GROUP_COUNT,
    V155_REFERENCE_CONFLICTING_ROW_COUNT,
    V155_REFERENCE_EXACT_FEATURE_GROUP_COUNT,
    run_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix,
    sequence_feature_leakage_scan,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV170DiagnosticPortfolioMatrixTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v170-diagnostic-portfolio-matrix"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix"
            ),
        )

    def test_source_digest_mismatch_closes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)

            result = run_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix(
                v169_report_path=paths["v169_report"],
                v168_report_path=paths["v168_report"],
                v167_report_path=paths["v167_report"],
                v165_dataset_path=paths["v165_dataset"],
                v155_report_path=paths["v155_report"],
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v153_report_path=paths["v153_report"],
                output_path=paths["v170_report"],
                shard_plan_output_path=paths["shard_plan"],
                expected_v169_exact_digest="wrong",
                expected_v168_exact_digest=payloads["v168_report"]["exact_digest"],
                expected_v167_exact_digest=payloads["v167_report"]["exact_digest"],
                expected_v165_dataset_digest=stable_payload_digest(payloads["v165_rows"]),
                expected_v155_exact_digest=payloads["v155_report"]["exact_digest"],
                expected_v154_exact_digest=payloads["v154_report"]["exact_digest"],
                expected_v153_exact_digest=payloads["v153_report"]["exact_digest"],
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn("v169_unexpected_exact_digest", result["source_validation"]["failures"])
        self.assertTrue(result["classification"]["primary"].endswith("closed_invalid"))
        self.assertFalse(result["training_ran"])
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_leakage_scan_rejects_label_and_identity_payloads(self) -> None:
        scan = sequence_feature_leakage_scan(
            [
                {
                    "observation_input": {"bucket": 1.0},
                    "label_action": "eat",
                    "source_seed": 13,
                }
            ]
        )

        self.assertFalse(scan["passed"])
        reasons = {failure["reason"] for failure in scan["failures"]}
        self.assertIn("forbidden_key_token", reasons)

    def test_report_contains_set_matrix_sequence_alias_and_shard_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)

            result = run_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix(
                v169_report_path=paths["v169_report"],
                v168_report_path=paths["v168_report"],
                v167_report_path=paths["v167_report"],
                v165_dataset_path=paths["v165_dataset"],
                v155_report_path=paths["v155_report"],
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v153_report_path=paths["v153_report"],
                output_path=paths["v170_report"],
                shard_plan_output_path=paths["shard_plan"],
                expected_v169_exact_digest=payloads["v169_report"]["exact_digest"],
                expected_v168_exact_digest=payloads["v168_report"]["exact_digest"],
                expected_v167_exact_digest=payloads["v167_report"]["exact_digest"],
                expected_v165_dataset_digest=stable_payload_digest(payloads["v165_rows"]),
                expected_v155_exact_digest=payloads["v155_report"]["exact_digest"],
                expected_v154_exact_digest=payloads["v154_report"]["exact_digest"],
                expected_v153_exact_digest=payloads["v153_report"]["exact_digest"],
            )
            written = json.loads(paths["v170_report"].read_text(encoding="utf-8"))
            shard_lines = [
                json.loads(line)
                for line in paths["shard_plan"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertTrue(result["source_validation"]["passed"])
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_SCHEMA_VERSION,
        )
        self.assertEqual(written["exact_digest"], _digest_without_exact(written))
        set_lane = result["portfolio_lanes"]["set_valued_ranking"]
        self.assertEqual(set_lane["top_n_values"], [1, 3, 5])
        self.assertTrue(set_lane["no_runtime_top_n_selected"])
        self.assertFalse(result["k_tuning_ran"])
        sequence_lane = result["portfolio_lanes"]["sequence_memory_alias"]
        self.assertEqual(
            sequence_lane["v155_reference_alias_counts"][
                "conflicting_exact_feature_row_count"
            ],
            V155_REFERENCE_CONFLICTING_ROW_COUNT,
        )
        self.assertEqual(len(shard_lines), 6)
        self.assertEqual(
            [row["seed"] for row in shard_lines],
            [13, 19, 29, 41, 5, 37],
        )
        self.assertTrue(all(row["plan_only_not_evidence"] for row in shard_lines))
        self.assertTrue(all("--fail-on-partial-shard" in row["expected_command_shape"] for row in shard_lines))

    def test_cli_writes_report_and_shard_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_inputs(tmpdir)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix",
                    "--v169-report",
                    str(paths["v169_report"]),
                    "--v168-report",
                    str(paths["v168_report"]),
                    "--v167-report",
                    str(paths["v167_report"]),
                    "--v165-dataset",
                    str(paths["v165_dataset"]),
                    "--v155-report",
                    str(paths["v155_report"]),
                    "--v154-report",
                    str(paths["v154_report"]),
                    "--v154-dataset",
                    str(paths["v154_dataset"]),
                    "--v153-report",
                    str(paths["v153_report"]),
                    "--output",
                    str(paths["v170_report"]),
                    "--shard-plan-output",
                    str(paths["shard_plan"]),
                    "--expected-v169-exact-digest",
                    str(payloads["v169_report"]["exact_digest"]),
                    "--expected-v168-exact-digest",
                    str(payloads["v168_report"]["exact_digest"]),
                    "--expected-v167-exact-digest",
                    str(payloads["v167_report"]["exact_digest"]),
                    "--expected-v165-dataset-digest",
                    stable_payload_digest(payloads["v165_rows"]),
                    "--expected-v155-exact-digest",
                    str(payloads["v155_report"]["exact_digest"]),
                    "--expected-v154-exact-digest",
                    str(payloads["v154_report"]["exact_digest"]),
                    "--expected-v153-exact-digest",
                    str(payloads["v153_report"]["exact_digest"]),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v170_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v170_diagnostic_portfolio_matrix_report=",
            completed.stdout,
        )
        self.assertIn("shard_plan_rows=6", completed.stdout)
        self.assertEqual(written["exact_digest"], _digest_without_exact(written))
        self.assertFalse(written["training_ran"])
        self.assertFalse(written["runtime_artifact_created"])


def _write_inputs(tmpdir: str) -> tuple[dict[str, Path], dict[str, object]]:
    root = Path(tmpdir)
    paths = {
        "v165_dataset": root / "v165.jsonl",
        "v167_report": root / "v167.json",
        "v168_report": root / "v168.json",
        "v169_report": root / "v169.json",
        "v154_dataset": root / "v154.jsonl",
        "v154_report": root / "v154.json",
        "v155_report": root / "v155.json",
        "v153_report": root / "v153.json",
        "v170_report": root / "v170.json",
        "shard_plan": root / "v170-shards.jsonl",
    }
    v165_rows = [
        _v165_row(seed=13, tick=97, bucket=1.0, safe_action="move_north"),
        _v165_row(seed=19, tick=99, bucket=2.0, safe_action="eat"),
        _v165_row(seed=29, tick=100, bucket=3.0, safe_action="move_north"),
        _v165_row(seed=41, tick=100, bucket=4.0, safe_action="eat"),
        _v165_row(seed=5, tick=99, bucket=1.1, safe_action="move_north"),
        _v165_row(seed=37, tick=99, bucket=2.1, safe_action="stay"),
    ]
    _write_jsonl(paths["v165_dataset"], v165_rows)
    v165_digest = stable_payload_digest(v165_rows)
    v167_report = _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION
            ),
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_POLICY,
            "classification": {"primary": "synthetic_v167_closed"},
            "dataset_digest": v165_digest,
            "preterminal_row_autopsy": [
                {
                    "row_index": 0,
                    "source_seed": 13,
                    "safe_hit": False,
                    "predicted_action": "eat",
                    "failure_mode": "feature_neighbor_aliasing",
                },
                {
                    "row_index": 1,
                    "source_seed": 19,
                    "safe_hit": False,
                    "predicted_action": "move_north",
                    "failure_mode": "feature_neighbor_aliasing",
                },
            ],
            "diagnostics_only": True,
            "training_ran": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
        }
    )
    v168_report = _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION
            ),
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_POLICY,
            "classification": {"primary": "synthetic_v168_partial"},
            "dataset_digest": v165_digest,
            "source_validation": {"passed": True},
            "diagnostics_only": True,
            "training_ran": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "runtime_observation_schema_changed": False,
            "runtime_policy_changed": False,
            "replay_viewer_schema_changed": False,
            "threshold_tuning_ran": False,
            "k_tuning_ran": False,
            "shadow_eval_ran": False,
            "live_ab_allowed": False,
            "promotion_authorized": False,
        }
    )
    v169_report = _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_SCHEMA_VERSION
            ),
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_POLICY,
            "classification": {"primary": EXPECTED_V169_CLASSIFICATION},
            "dataset_digest": v165_digest,
            "source_validation": {"passed": True},
            "v168_baseline_family_recomputed": {
                "zero_safe_hit_preterminal_source_seeds": [13, 19, 29, 41]
            },
            "best_candidate": {"new_zero_hit_seeds_introduced": [5]},
            "diagnostics_only": True,
            "training_ran": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "runtime_observation_schema_changed": False,
            "runtime_policy_changed": False,
            "replay_viewer_schema_changed": False,
            "threshold_tuning_ran": False,
            "k_tuning_ran": False,
            "shadow_eval_ran": False,
            "live_ab_allowed": False,
            "promotion_authorized": False,
        }
    )
    v154_rows = [
        _v154_row(bucket=1.0, label="eat"),
        _v154_row(bucket=1.0, label="stay"),
        _v154_row(bucket=2.0, label="move_north"),
        _v154_row(bucket=2.0, label="move_north"),
    ]
    _write_jsonl(paths["v154_dataset"], v154_rows)
    v154_digest = stable_payload_digest(v154_rows)
    v154_report = _report(
        {
            "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION,
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY,
            "classification": {"primary": EXPECTED_V154_SUPPORT_READY_CLASSIFICATION},
            "dataset": {"row_count": len(v154_rows), "dataset_digest": v154_digest},
            "continuation_branch_results": [],
            "diagnostics_only": True,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
        }
    )
    v155_report = _report(
        {
            "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION,
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY,
            "classification": {"primary": EXPECTED_V155_CLASSIFICATION},
            "source_validation": {"dataset_digest": v154_digest},
            "pretraining_review": {
                "nearest_neighbor_alias_collision_audit": {
                    "exact_feature_group_count": V155_REFERENCE_EXACT_FEATURE_GROUP_COUNT,
                    "conflicting_exact_feature_group_count": (
                        V155_REFERENCE_CONFLICTING_GROUP_COUNT
                    ),
                    "conflicting_exact_feature_row_count": (
                        V155_REFERENCE_CONFLICTING_ROW_COUNT
                    ),
                }
            },
            "diagnostics_only": True,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
        }
    )
    v153_report = _report(
        {
            "schema_version": (
                M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION
            ),
            "policy": M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY,
            "classification": {"primary": "synthetic_v153_closed"},
            "diagnostics_only": True,
        }
    )
    for key, payload in (
        ("v167_report", v167_report),
        ("v168_report", v168_report),
        ("v169_report", v169_report),
        ("v154_report", v154_report),
        ("v155_report", v155_report),
        ("v153_report", v153_report),
    ):
        paths[key].write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return (
        paths,
        {
            "v165_rows": v165_rows,
            "v167_report": v167_report,
            "v168_report": v168_report,
            "v169_report": v169_report,
            "v154_rows": v154_rows,
            "v154_report": v154_report,
            "v155_report": v155_report,
            "v153_report": v153_report,
        },
    )


def _v165_row(
    *,
    seed: int,
    tick: int,
    bucket: float,
    safe_action: str,
) -> dict[str, object]:
    mask = _mask()
    return {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ),
        "feature_policy_id": "synthetic_public_features",
        "trainable_public_features": {
            "public_observation": {"bucket": float(bucket)},
            "action_mask": mask,
        },
        "public_action_mask": mask,
        "action_value_targets": [
            {
                "action": action,
                "public_mask": bool(mask[action]),
                "target_available": bool(mask[action]),
                "safe_target": action == safe_action,
                "robust_safe_action": action == safe_action,
                "value_target": 1.0 if action == safe_action else 0.0 if mask[action] else None,
                "score_target": 1.0 if action == safe_action else 0.0 if mask[action] else None,
            }
            for action in ACTION_NAMES
        ],
        "safe_action_set": [safe_action],
        "target_classification": "unique_robust_winner",
        "robust_winner_action": safe_action,
        "metadata": {
            "seed": int(seed),
            "branch_tick": int(tick),
            "agent_id": int(seed),
            "source_path": "",
            "line_number": 0,
            "branch_id": f"synthetic-seed-{seed}-tick-{tick}",
        },
    }


def _v154_row(*, bucket: float, label: str) -> dict[str, object]:
    return {
        "schema_version": "m3_carrion_survivor_continuation_archive_dataset_row_v1",
        "metadata": {"seed": 13, "branch_id": "metadata-only"},
        "trainable": {
            "feature_policy": "public_observation_action_mask_and_empty_public_prior_context_v1",
            "features": {
                "observation_input": {"bucket": float(bucket)},
                "action_mask": _mask(),
                "prior_public_context": [],
            },
            "label": {
                "action": label,
                "label_policy": "replay_verified_survivor_continuation_outcome_improvement_v1",
            },
        },
    }


def _mask() -> dict[str, bool]:
    return {action: action in {"stay", "eat", "move_north"} for action in ACTION_NAMES}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _report(payload: dict[str, object]) -> dict[str, object]:
    report = dict(payload)
    report["exact_digest"] = _digest_without_exact(report)
    return report


def _digest_without_exact(payload: dict[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(without_digest, sort_keys=True, allow_nan=False))
    )


if __name__ == "__main__":
    unittest.main()
