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
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_DATASET_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_POLICY,
)
from evolution_sim.mind.carrion_survivor_continuation_feature_sufficiency_audit import (
    EXPECTED_V155_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_SCHEMA_VERSION,
    feature_leakage_scan,
    run_carrion_survivor_continuation_feature_sufficiency_audit,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    EXPECTED_V154_SUPPORT_READY_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]
TARGET_SEEDS = (13, 19, 29, 37, 41, 43)
EAT_SEEDS = {13, 29, 41}


class MindV3CarrionSurvivorContinuationFeatureSufficiencyAuditTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-feature-sufficiency-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_feature_sufficiency_audit"
            ),
        )

    def test_source_validation_requires_v155_closed_classification(self) -> None:
        rows = _synthetic_rows(public_recent_context_separates=True)
        v154_report = _v154_report(rows, public_recent_context_separates=True)
        v155_report = _v155_report(rows)
        v155_report["classification"]["primary"] = "unexpected"

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, v154_report, rows, v155_report)
            result = run_carrion_survivor_continuation_feature_sufficiency_audit(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                output_path=paths["output"],
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v155_unexpected_classification",
            result["source_validation"]["failures"],
        )
        self.assertEqual(result["feature_policy_count"], 0)
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_public_feature_sufficiency_"
                "source_invalid_closed_no_training"
            ),
        )
        self.assertFalse(result["artifact_created"])
        self.assertFalse(result["training_ran"])
        self.assertFalse(result["shadow_live_ab_ran"])

    def test_public_recent_features_can_make_synthetic_support_ready(self) -> None:
        rows = _synthetic_rows(public_recent_context_separates=True)
        v154_report = _v154_report(rows, public_recent_context_separates=True)
        v155_report = _v155_report(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, v154_report, rows, v155_report)
            result = run_carrion_survivor_continuation_feature_sufficiency_audit(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                output_path=paths["output"],
            )

        self.assertEqual(
            result["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_SUFFICIENCY_SCHEMA_VERSION,
        )
        self.assertTrue(result["source_validation"]["passed"])
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_public_feature_sufficiency_"
                "support_ready_no_training"
            ),
        )
        by_policy = {
            str(report["policy_id"]): report for report in result["feature_policies"]
        }
        base = by_policy["current_observation_action_mask_only"]
        recent = by_policy[
            "current_observation_action_mask_public_recent_action_result_history"
        ]
        self.assertEqual(base["conflicting_row_count"], len(rows))
        self.assertEqual(recent["baseline_conflicting_row_count"], len(rows))
        self.assertEqual(recent["conflicting_row_count"], 0)
        self.assertTrue(recent["support_ready_no_training"])
        self.assertGreaterEqual(
            recent["nearest_neighbor_minus_best_trivial_accuracy"],
            0.05,
        )
        self.assertTrue(recent["leakage_scan"]["passed"])
        self.assertFalse(result["artifact_created"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["promotion_authorized"])
        self.assertFalse(result["runtime_promotion_allowed"])

    def test_public_feature_surface_insufficient_when_aliases_remain(self) -> None:
        rows = _synthetic_rows(public_recent_context_separates=False)
        v154_report = _v154_report(rows, public_recent_context_separates=False)
        v155_report = _v155_report(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, v154_report, rows, v155_report)
            result = run_carrion_survivor_continuation_feature_sufficiency_audit(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                output_path=paths["output"],
            )

        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_public_feature_sufficiency_"
                "public_feature_surface_insufficient_no_training"
            ),
        )
        self.assertFalse(
            any(
                policy["support_ready_no_training"]
                for policy in result["feature_policies"]
            )
        )
        self.assertTrue(
            all(
                policy["leakage_scan"]["passed"]
                for policy in result["feature_policies"]
            )
        )

    def test_leakage_scan_flags_forbidden_feature_tokens(self) -> None:
        scan = feature_leakage_scan([{"branch_id": "m3-branch-13"}])

        self.assertFalse(scan["passed"])
        self.assertEqual(scan["failure_count"], 2)

    def test_cli_writes_closed_report(self) -> None:
        rows = _synthetic_rows(public_recent_context_separates=False)
        v154_report = _v154_report(rows, public_recent_context_separates=False)
        v155_report = _v155_report(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, v154_report, rows, v155_report)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"
            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_feature_sufficiency_audit",
                    "--v154-report",
                    str(paths["v154_report"]),
                    "--v154-dataset",
                    str(paths["v154_dataset"]),
                    "--v155-report",
                    str(paths["v155_report"]),
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

            self.assertIn("artifact_created=False", completed.stdout)
            self.assertIn("training_ran=False", completed.stdout)
            written = json.loads(paths["output"].read_text(encoding="utf-8"))

        self.assertEqual(
            written["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_public_feature_sufficiency_"
                "public_feature_surface_insufficient_no_training"
            ),
        )
        self.assertFalse(written["artifact_created"])
        self.assertFalse(written["training_ran"])


def _write_inputs(
    tmpdir: str,
    v154_report: dict[str, object],
    rows: list[dict[str, object]],
    v155_report: dict[str, object],
) -> dict[str, Path]:
    base = Path(tmpdir)
    paths = {
        "v154_report": base / "v154-report.json",
        "v154_dataset": base / "v154-dataset.jsonl",
        "v155_report": base / "v155-report.json",
        "output": base / "v156-report.json",
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
    return paths


def _v154_report(
    rows: list[dict[str, object]],
    *,
    public_recent_context_separates: bool,
) -> dict[str, object]:
    action_counts: dict[str, int] = {}
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
        "continuation_branch_results": _branch_results(
            rows,
            public_recent_context_separates=public_recent_context_separates,
        ),
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
            "primary": EXPECTED_V155_CLASSIFICATION,
            "labels": [EXPECTED_V155_CLASSIFICATION],
        },
        "source_validation": {
            "passed": True,
            "failures": [],
            "dataset_digest": dataset_digest,
        },
        "artifact": {"created": False},
        "training": {"diagnostic_training_ran": False},
        "evaluation": {"skipped": True},
    }


def _branch_results(
    rows: list[dict[str, object]],
    *,
    public_recent_context_separates: bool,
) -> list[dict[str, object]]:
    seen = {}
    for row in rows:
        metadata = row["metadata"]
        branch_id = metadata["branch_id"]
        if branch_id in seen:
            continue
        label_action = row["trainable"]["label"]["action"]
        if not public_recent_context_separates:
            label_action = "eat"
        seen[branch_id] = {
            "branch_id": branch_id,
            "baseline_action": "stay",
            "seed": metadata["seed"],
            "fixture": "carrion_only",
            "public_features": row["trainable"]["features"],
            "carrion_archive_context": {
                "reason_evidence": _reason_evidence(label_action),
            },
        }
    return list(seen.values())


def _synthetic_rows(*, public_recent_context_separates: bool) -> list[dict[str, object]]:
    del public_recent_context_separates
    rows = []
    for seed in TARGET_SEEDS:
        action = "eat" if seed in EAT_SEEDS else "stay"
        branch_id = f"synthetic-branch-{seed}"
        for sample_index in range(20):
            rows.append(
                _row(
                    row_index=len(rows),
                    seed=seed,
                    branch_id=branch_id,
                    sample_index=sample_index,
                    action=action,
                )
            )
    return rows


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
            "continuation_index": sample_index % 4,
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


def _reason_evidence(label_action: str) -> dict[str, object]:
    if label_action == "eat":
        return {
            "requested_action": "eat",
            "resolved_action": "eat",
            "moved": False,
            "ate": True,
            "food_source": "carcass",
            "hydration_ratio_before": 0.42,
            "hydration_ratio_after": 0.37,
            "energy_ratio_before": 0.31,
            "energy_ratio_after": 0.74,
            "health_ratio_before": 0.82,
            "health_ratio_after": 0.81,
            "died_after_action": False,
        }
    return {
        "requested_action": "stay",
        "resolved_action": "stay",
        "moved": False,
        "ate": False,
        "food_source": None,
        "hydration_ratio_before": 0.35,
        "hydration_ratio_after": 0.35,
        "energy_ratio_before": 0.72,
        "energy_ratio_after": 0.71,
        "health_ratio_before": 0.84,
        "health_ratio_after": 0.84,
        "died_after_action": False,
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
