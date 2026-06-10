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
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION,
    EXPECTED_V154_SUPPORT_READY_CLASSIFICATION,
    build_pretraining_review,
    run_carrion_survivor_continuation_train_eval,
    validate_v154_train_eval_inputs,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]
TARGET_SEEDS = (13, 19, 29, 37, 41, 43)


class MindV3CarrionSurvivorContinuationTrainEvalTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-train-eval"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_train_eval"
            ),
        )

    def test_source_validation_requires_v154_support_ready(self) -> None:
        rows = _generalizable_rows()
        report = _v154_report(rows)
        report["classification"]["primary"] = "unexpected"

        validation = validate_v154_train_eval_inputs(
            v154_report=report,
            dataset_rows=rows,
            min_label_count=len(rows),
        )

        self.assertFalse(validation["passed"])
        self.assertIn("v154_not_support_ready", validation["failures"])

    def test_source_validation_requires_v154_runtime_action_selection_unchanged(
        self,
    ) -> None:
        rows = _generalizable_rows()
        report = _v154_report(rows)
        report["runtime_action_selection_changed"] = True

        validation = validate_v154_train_eval_inputs(
            v154_report=report,
            dataset_rows=rows,
            min_label_count=len(rows),
        )

        self.assertFalse(validation["passed"])
        self.assertIn(
            "v154_runtime_action_selection_changed_not_false",
            validation["failures"],
        )

    def test_pretraining_review_blocks_exact_public_feature_label_conflicts(self) -> None:
        rows = _conflicting_alias_rows()
        review = build_pretraining_review(
            dataset_rows=rows,
            branch_results=_branch_results(rows),
            target_seeds=TARGET_SEEDS,
        )

        self.assertFalse(review["passed"])
        reasons = {blocker["reason"] for blocker in review["blockers"]}
        self.assertIn("exact_public_feature_alias_conflicts", reasons)
        self.assertGreater(
            review["nearest_neighbor_alias_collision_audit"][
                "conflicting_exact_feature_group_count"
            ],
            0,
        )

    def test_trivial_action_mask_prior_fails_closed_before_training(self) -> None:
        rows = _trivial_mask_prior_rows()
        report = _v154_report(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, report, rows)
            result = run_carrion_survivor_continuation_train_eval(
                v154_report_path=paths["report"],
                v154_dataset_path=paths["dataset"],
                artifact_output_path=paths["artifact"],
                output_path=paths["output"],
                min_label_count=len(rows),
                run_evaluation=False,
            )

            self.assertEqual(
                result["classification"]["primary"],
                (
                    "m3_carrion_survivor_continuation_train_eval_"
                    "pretraining_alias_prior_blocked_closed_no_training"
                ),
            )
            self.assertFalse(result["artifact"]["created"])
            self.assertFalse(result["training"]["diagnostic_training_ran"])
            self.assertFalse(result["runtime_action_selection_changed"])
            self.assertFalse(result["artifact"]["runtime_action_selection_changed"])
            self.assertFalse(paths["artifact"].exists())
            reasons = {
                blocker["reason"]
                for blocker in result["pretraining_review"]["blockers"]
            }
            self.assertIn("mask_only_baseline_too_predictive", reasons)

    def test_pretraining_pass_creates_opt_in_artifact_when_evaluation_skipped(self) -> None:
        rows = _generalizable_rows()
        report = _v154_report(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, report, rows)
            result = run_carrion_survivor_continuation_train_eval(
                v154_report_path=paths["report"],
                v154_dataset_path=paths["dataset"],
                artifact_output_path=paths["artifact"],
                output_path=paths["output"],
                min_label_count=len(rows),
                run_evaluation=False,
            )

            self.assertEqual(
                result["schema_version"],
                M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION,
            )
            self.assertTrue(result["source_validation"]["passed"])
            self.assertTrue(result["pretraining_review"]["passed"])
            self.assertTrue(result["artifact"]["created"])
            self.assertTrue(paths["artifact"].exists())
            self.assertTrue(result["training"]["diagnostic_training_ran"])
            self.assertFalse(result["training_authorized"])
            self.assertFalse(result["promotion_authorized"])
            self.assertFalse(result["runtime_promotion_allowed"])
            self.assertFalse(result["runtime_action_selection_changed"])
            self.assertFalse(result["artifact"]["runtime_action_selection_changed"])
            self.assertTrue(result["diagnostics_only"])
            self.assertTrue(result["evaluation"]["skipped"])

    def test_cli_writes_fail_closed_report(self) -> None:
        rows = _conflicting_alias_rows()
        report = _v154_report(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_inputs(tmpdir, report, rows)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"
            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_train_eval",
                    "--v154-report",
                    str(paths["report"]),
                    "--v154-dataset",
                    str(paths["dataset"]),
                    "--artifact-output",
                    str(paths["artifact"]),
                    "--output",
                    str(paths["output"]),
                    "--min-label-count",
                    str(len(rows)),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            self.assertIn("artifact_created=False", completed.stdout)
            written = json.loads(paths["output"].read_text(encoding="utf-8"))
            self.assertFalse(written["artifact"]["created"])
            self.assertFalse(written["training"]["diagnostic_training_ran"])
            self.assertEqual(
                written["classification"]["primary"],
                (
                    "m3_carrion_survivor_continuation_train_eval_"
                    "pretraining_alias_prior_blocked_closed_no_training"
                ),
            )


def _write_inputs(
    tmpdir: str,
    report: dict[str, object],
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    base = Path(tmpdir)
    report_path = base / "v154-report.json"
    dataset_path = base / "v154-dataset.jsonl"
    output_path = base / "v155-report.json"
    artifact_path = base / "v155-artifact.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    dataset_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return {
        "report": report_path,
        "dataset": dataset_path,
        "output": output_path,
        "artifact": artifact_path,
    }


def _v154_report(rows: list[dict[str, object]]) -> dict[str, object]:
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
        "continuation_branch_results": _branch_results(rows),
    }


def _branch_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = {}
    for row in rows:
        metadata = row["metadata"]
        branch_id = metadata["branch_id"]
        if branch_id in seen:
            continue
        seen[branch_id] = {
            "branch_id": branch_id,
            "baseline_action": "stay",
            "seed": metadata["seed"],
            "fixture": "carrion_only",
            "public_features": row["trainable"]["features"],
        }
    return list(seen.values())


def _generalizable_rows() -> list[dict[str, object]]:
    rows = []
    for index, seed in enumerate(TARGET_SEEDS):
        eat_source = seed in {13, 29, 41}
        rows.append(
            _row(
                row_index=index,
                seed=seed,
                branch_id=f"branch-{seed}",
                action="eat" if eat_source else "stay",
                energy=0.2 if eat_source else 0.8,
            )
        )
    return rows


def _conflicting_alias_rows() -> list[dict[str, object]]:
    rows = []
    for index, seed in enumerate(TARGET_SEEDS):
        rows.append(
            _row(
                row_index=index,
                seed=seed,
                branch_id=f"branch-{seed}",
                action="eat" if index % 2 == 0 else "stay",
                energy=0.4,
            )
        )
    return rows


def _trivial_mask_prior_rows() -> list[dict[str, object]]:
    rows = []
    for index, seed in enumerate(TARGET_SEEDS):
        eat_source = index < 3
        rows.append(
            _row(
                row_index=index,
                seed=seed,
                branch_id=f"branch-{seed}",
                action="eat" if eat_source else "stay",
                energy=0.1 + index * 0.1,
                mask_kind="eat_mask" if eat_source else "stay_mask",
            )
        )
    return rows


def _row(
    *,
    row_index: int,
    seed: int,
    branch_id: str,
    action: str,
    energy: float,
    mask_kind: str = "default",
) -> dict[str, object]:
    return {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_DATASET_ROW_SCHEMA_VERSION,
        "metadata": {
            "row_index": row_index,
            "seed": seed,
            "fixture": "carrion_only",
            "branch_id": branch_id,
            "branch_tick": 1,
            "agent_id": 1,
        },
        "trainable": {
            "feature_policy": M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_POLICY,
            "features": {
                "observation_input": _observation_input(energy=energy),
                "action_mask": _action_mask(mask_kind=mask_kind),
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


def _observation_input(*, energy: float) -> dict[str, object]:
    values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
    values[SELF_INPUT_FIELDS.index("energy_ratio")] = energy
    values[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.6
    values[SELF_INPUT_FIELDS.index("health_ratio")] = 0.9
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


def _action_mask(*, mask_kind: str = "default") -> dict[str, bool]:
    if mask_kind == "eat_mask":
        legal = {"stay", "eat"}
    elif mask_kind == "stay_mask":
        legal = {"stay", "move_north"}
    else:
        legal = {"stay", "eat"}
    return {action: action in legal for action in ACTION_NAMES}


if __name__ == "__main__":
    unittest.main()
