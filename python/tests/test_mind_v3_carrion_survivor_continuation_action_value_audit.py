from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zlib

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_DTYPE,
    OBSERVATION_INPUT_VALUE_RANGE,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_QUANTIZATION_SCALE,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_STORAGE_DTYPE,
    OBSERVATION_STORAGE_ENCODING,
    SELF_INPUT_FIELDS,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    EXPECTED_V156_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_AUDIT_SCHEMA_VERSION,
    hard_trainable_feature_leakage_scan,
    run_carrion_survivor_continuation_action_value_audit,
)
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_DATASET_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_POLICY,
)
from evolution_sim.mind.carrion_survivor_continuation_feature_sufficiency_audit import (
    M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_SCHEMA_VERSION,
    candidate_feature_policies,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    EXPECTED_V154_SUPPORT_READY_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]
TARGET_SEEDS = (13, 19, 29, 37, 41, 43)


class MindV3CarrionSurvivorContinuationActionValueAuditTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-action-value-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_action_value_audit"
            ),
        )

    def test_source_validation_requires_v156_insufficient_classification(self) -> None:
        rows, branches = _synthetic_inputs(mode="multi_safe")
        v154_report = _v154_report(rows, branches)
        v155_report = _v155_report(rows)
        v156_report = _v156_report(rows)
        v156_report["classification"]["primary"] = "unexpected"

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, v154_report, rows, v155_report, v156_report)
            result = run_carrion_survivor_continuation_action_value_audit(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                v156_report_path=paths["v156_report"],
                output_path=paths["output"],
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v156_unexpected_classification",
            result["source_validation"]["failures"],
        )
        self.assertEqual(result["feature_policy_count"], 0)
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_public_state_action_value_audit_"
                "source_invalid_closed_no_training"
            ),
        )
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["runtime_promotion_allowed"])

    def test_multi_action_safe_set_reduces_single_label_conflicts(self) -> None:
        rows, branches = _synthetic_inputs(mode="multi_safe")
        v154_report = _v154_report(rows, branches)
        v155_report = _v155_report(rows)
        v156_report = _v156_report(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, v154_report, rows, v155_report, v156_report)
            result = run_carrion_survivor_continuation_action_value_audit(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                v156_report_path=paths["v156_report"],
                output_path=paths["output"],
            )

        self.assertEqual(
            result["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_AUDIT_SCHEMA_VERSION,
        )
        self.assertTrue(result["source_validation"]["passed"])
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_public_state_action_value_audit_"
                "set_or_action_value_targets_reduce_conflicts_no_training"
            ),
        )
        best = result["best_feature_policy"]
        self.assertEqual(best["single_label_conflicting_row_count"], len(rows))
        self.assertEqual(
            best["action_value_resolvable_conflicting_row_count"],
            len(rows),
        )
        self.assertEqual(best["classification_counts"]["multi_action_safe_set"], 1)
        self.assertTrue(
            result["conflict_reduction"]["would_reduce_single_label_conflicting_rows"]
        )
        self.assertFalse(result["artifact_created"])
        self.assertFalse(result["training_ran"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["promotion_authorized"])
        self.assertFalse(result["runtime_promotion_allowed"])

    def test_no_robust_action_keeps_public_state_ambiguity_closed(self) -> None:
        rows, branches = _synthetic_inputs(mode="ambiguous")
        v154_report = _v154_report(rows, branches)
        v155_report = _v155_report(rows)
        v156_report = _v156_report(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, v154_report, rows, v155_report, v156_report)
            result = run_carrion_survivor_continuation_action_value_audit(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                v156_report_path=paths["v156_report"],
                output_path=paths["output"],
            )

        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_public_state_action_value_audit_"
                "public_state_ambiguity_remains_no_training"
            ),
        )
        best = result["best_feature_policy"]
        self.assertEqual(best["action_value_resolvable_conflicting_row_count"], 0)
        self.assertEqual(best["classification_counts"]["conflicting_no_public_winner"], 1)
        self.assertFalse(
            result["conflict_reduction"]["would_reduce_single_label_conflicting_rows"]
        )

    def test_hard_leakage_scan_flags_branch_reason_feature(self) -> None:
        scan = hard_trainable_feature_leakage_scan(
            [{"public_branch_reason": "movement_stall"}]
        )

        self.assertFalse(scan["passed"])
        reasons = {failure["reason"] for failure in scan["failures"]}
        self.assertIn("forbidden_trainable_feature_key_token", reasons)
        self.assertIn("forbidden_trainable_feature_string_marker", reasons)

    def test_cli_writes_parseable_closed_report(self) -> None:
        rows, branches = _synthetic_inputs(mode="ambiguous")
        v154_report = _v154_report(rows, branches)
        v155_report = _v155_report(rows)
        v156_report = _v156_report(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, v154_report, rows, v155_report, v156_report)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"
            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_action_value_audit",
                    "--v154-report",
                    str(paths["v154_report"]),
                    "--v154-dataset",
                    str(paths["v154_dataset"]),
                    "--v155-report",
                    str(paths["v155_report"]),
                    "--v156-report",
                    str(paths["v156_report"]),
                    "--output",
                    str(paths["output"]),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            self.assertIn("training_ran=False", completed.stdout)
            self.assertIn("runtime_promotion_allowed=False", completed.stdout)
            written = json.loads(paths["output"].read_text(encoding="utf-8"))

        self.assertFalse(written["artifact_created"])
        self.assertFalse(written["training_ran"])
        self.assertFalse(written["shadow_live_ab_ran"])
        self.assertEqual(
            written["source_validation"]["observed_v156_classification"],
            EXPECTED_V156_CLASSIFICATION,
        )


def _write_inputs(
    tmpdir: str,
    v154_report: dict[str, object],
    rows: list[dict[str, object]],
    v155_report: dict[str, object],
    v156_report: dict[str, object],
) -> dict[str, Path]:
    base = Path(tmpdir)
    paths = {
        "v154_report": base / "v154-report.json",
        "v154_dataset": base / "v154-dataset.jsonl",
        "v155_report": base / "v155-report.json",
        "v156_report": base / "v156-report.json",
        "output": base / "v157-report.json",
    }
    paths["v154_report"].write_text(
        json.dumps(v154_report, sort_keys=True),
        encoding="utf-8",
    )
    paths["v154_dataset"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    paths["v155_report"].write_text(
        json.dumps(v155_report, sort_keys=True),
        encoding="utf-8",
    )
    paths["v156_report"].write_text(
        json.dumps(v156_report, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def _synthetic_inputs(*, mode: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    for seed in TARGET_SEEDS:
        branch_id = f"synthetic-branch-{seed}"
        for sample_index in range(20):
            action = "eat" if sample_index % 2 == 0 else "stay"
            rows.append(
                _row(
                    row_index=len(rows),
                    seed=seed,
                    branch_id=branch_id,
                    sample_index=sample_index,
                    action=action,
                )
            )
    branches = [
        _branch_result(seed=seed, mode=mode)
        for seed in TARGET_SEEDS
    ]
    return rows, branches


def _v154_report(
    rows: list[dict[str, object]],
    branches: list[dict[str, object]],
) -> dict[str, object]:
    action_counts = {}
    for row in rows:
        action = row["trainable"]["label"]["action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    dominant_share = max(action_counts.values()) / max(len(rows), 1)
    dataset_digest = stable_payload_digest(rows)
    return {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION,
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY,
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "classification": {
            "primary": EXPECTED_V154_SUPPORT_READY_CLASSIFICATION,
            "labels": [EXPECTED_V154_SUPPORT_READY_CLASSIFICATION],
        },
        "source_integrity": {
            "passed": True,
            "failures": [],
            "replay_verification_complete": True,
            "leakage_scan_passed": True,
            "heuristic_action_source_count": 0,
            "action_coverage_complete": True,
        },
        "support_floors": {"passed": True, "floors": []},
        "generation_status": {"state": "complete", "partial": False},
        "dataset": {
            "row_count": len(rows),
            "dataset_digest": dataset_digest,
            "dominant_label_action_share": round(dominant_share, 6),
            "feature_policy": M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_POLICY,
            "row_schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_DATASET_ROW_SCHEMA_VERSION
            ),
            "leakage_scan": {"passed": True, "failures": []},
        },
        "continuation_branch_results": branches,
        "continuation_branch_evidence_digest": stable_payload_digest(branches),
    }


def _v155_report(rows: list[dict[str, object]]) -> dict[str, object]:
    dataset_digest = stable_payload_digest(rows)
    return {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION,
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY,
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "classification": {
            "primary": (
                "m3_carrion_survivor_continuation_train_eval_"
                "pretraining_alias_prior_blocked_closed_no_training"
            ),
            "labels": [
                "m3_carrion_survivor_continuation_train_eval_"
                "pretraining_alias_prior_blocked_closed_no_training"
            ],
        },
        "source_validation": {
            "passed": True,
            "failures": [],
            "dataset_digest": dataset_digest,
        },
        "pretraining_review": {
            "nearest_neighbor_alias_collision_audit": {
                "conflicting_exact_feature_row_count": len(rows),
            }
        },
        "artifact": {"created": False},
        "training": {"diagnostic_training_ran": False},
        "evaluation": {"skipped": True},
    }


def _v156_report(rows: list[dict[str, object]]) -> dict[str, object]:
    dataset_digest = stable_payload_digest(rows)
    return {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_POLICY,
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "artifact_created": False,
        "training_ran": False,
        "shadow_live_ab_ran": False,
        "classification": {
            "primary": EXPECTED_V156_CLASSIFICATION,
            "labels": [EXPECTED_V156_CLASSIFICATION],
        },
        "source_validation": {
            "passed": True,
            "failures": [],
            "dataset_digest": dataset_digest,
            "label_count": len(rows),
        },
        "feature_policies": [
            {
                "policy_id": policy["policy_id"],
                "conflicting_row_count": len(rows),
                "leakage_scan": {"passed": True, "failures": []},
            }
            for policy in candidate_feature_policies()
        ],
        "feature_policy_count": len(candidate_feature_policies()),
    }


def _row(
    *,
    row_index: int,
    seed: int,
    branch_id: str,
    sample_index: int,
    action: str,
) -> dict[str, object]:
    return {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_DATASET_ROW_SCHEMA_VERSION,
        "metadata": {
            "row_index": row_index,
            "seed": seed,
            "fixture": "carrion_only",
            "branch_id": branch_id,
            "branch_tick": 7,
            "agent_id": 1,
            "continuation_index": sample_index % 2,
        },
        "trainable": {
            "feature_policy": M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_POLICY,
            "features": {
                "observation_input": _observation_input(),
                "action_mask": _action_mask(),
                "prior_public_context": [],
            },
            "label": {
                "action": action,
                "label_policy": (
                    "replay_verified_survivor_continuation_outcome_improvement_v1"
                ),
            },
        },
    }


def _branch_result(*, seed: int, mode: str) -> dict[str, object]:
    branch_id = f"synthetic-branch-{seed}"
    return {
        "branch_id": branch_id,
        "seed": seed,
        "fixture": "carrion_only",
        "branch_index": 0,
        "branch_tick": 7,
        "agent_id": 1,
        "candidate_actions": ["stay", "eat", "move_north"],
        "continuation_indexes": [0, 1],
        "public_features": {
            "observation_input": _observation_input(),
            "action_mask": _action_mask(),
        },
        "carrion_archive_context": {
            "reason_evidence": {
                "requested_action": "eat",
                "resolved_action": "eat",
                "moved": False,
                "ate": True,
                "food_source": "carcass",
                "hydration_ratio_before": 0.4,
                "hydration_ratio_after": 0.35,
                "energy_ratio_before": 0.4,
                "energy_ratio_after": 0.8,
                "health_ratio_before": 0.8,
                "health_ratio_after": 0.8,
                "died_after_action": False,
            }
        },
        "continuation_runs": _runs(mode=mode),
    }


def _runs(*, mode: str) -> list[dict[str, object]]:
    if mode == "multi_safe":
        return [
            _run("stay", 0, reproduction=True),
            _run("stay", 1, reproduction=True),
            _run("eat", 0, hydration=True),
            _run("eat", 1, hydration=True),
            _run("move_north", 0),
            _run("move_north", 1),
        ]
    if mode == "ambiguous":
        return [
            _run("stay", 0, reproduction=True),
            _run("stay", 1),
            _run("eat", 0, hydration=True),
            _run("eat", 1, hydration=True, resolved_invalid_delta=1),
            _run("move_north", 0),
            _run("move_north", 1),
        ]
    raise AssertionError(mode)


def _run(
    action: str,
    continuation_index: int,
    *,
    survival: bool = False,
    hydration: bool = False,
    reproduction: bool = False,
    blockers: bool = False,
    resolved_invalid_delta: int = 0,
    unsupported_requested_count: int = 0,
) -> dict[str, object]:
    labels = []
    if survival:
        labels.append("survival")
    if hydration:
        labels.append("hydration_recovery")
    if reproduction:
        labels.append("reproduction_readiness")
    if blockers:
        labels.append("fewer_blockers")
    return {
        "forced_action": action,
        "continuation_index": continuation_index,
        "forced_action_used": True,
        "forced_action_supported": True,
        "unsupported_requested_action_count": unsupported_requested_count,
        "unsupported_resolved_action_count": max(0, resolved_invalid_delta),
        "heuristic_action_source_count": 0,
        "deltas_vs_baseline": {
            "alive_agents": 1 if survival else 0,
            "births": 1 if reproduction else 0,
            "deaths": -1 if blockers else 0,
            "target_alive": 1 if survival else 0,
            "target_hydration_ratio": 0.2 if hydration else 0.0,
            "unsupported_requested_action_count": -1 if blockers else 0,
            "unsupported_resolved_action_count": resolved_invalid_delta,
        },
        "first_action_outcome": {
            "outcome": {
                "reproduced": reproduction,
                "reproduction_ready_after": reproduction,
                "resolution_action_valid": resolved_invalid_delta <= 0,
            }
        },
        "outcome_improvement": {
            "improved": bool(labels),
            "labels": labels,
            "survival_improved": survival,
            "hydration_recovery": hydration,
            "reproduction_readiness": reproduction,
            "fewer_blockers": blockers,
        },
        "replay_verification": {"verified": True},
    }


def _observation_input() -> dict[str, object]:
    values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
    values[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.5
    values[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.5
    values[SELF_INPUT_FIELDS.index("health_ratio")] = 0.5
    quantized = [
        int(round(max(-1.0, min(1.0, value)) * OBSERVATION_QUANTIZATION_SCALE))
        for value in values
    ]
    packed = struct.pack(f"<{len(quantized)}h", *quantized)
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "encoder_version": OBSERVATION_ENCODER_VERSION,
        "decoded_dtype": OBSERVATION_INPUT_DTYPE,
        "storage_dtype": OBSERVATION_STORAGE_DTYPE,
        "storage_encoding": OBSERVATION_STORAGE_ENCODING,
        "shape": [OBSERVATION_INPUT_VECTOR_SIZE],
        "value_range": list(OBSERVATION_INPUT_VALUE_RANGE),
        "data": base64.b64encode(zlib.compress(packed, level=6)).decode("ascii"),
    }


def _action_mask() -> dict[str, bool]:
    legal = {"stay", "eat", "move_north"}
    return {action: action in legal for action in ACTION_NAMES}


if __name__ == "__main__":
    unittest.main()
