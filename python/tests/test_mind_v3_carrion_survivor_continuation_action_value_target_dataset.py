from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    run_carrion_survivor_continuation_action_value_audit,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    EXPECTED_V157_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_SCHEMA_VERSION,
    run_carrion_survivor_continuation_action_value_target_dataset,
)
from evolution_sim.mind.provenance import stable_payload_digest

try:
    from python.tests.test_mind_v3_carrion_survivor_continuation_action_value_audit import (
        _run,
        _synthetic_inputs,
        _v154_report,
        _v155_report,
        _v156_report,
        _write_inputs,
    )
except ModuleNotFoundError:  # pragma: no cover - unittest discovery fallback.
    from test_mind_v3_carrion_survivor_continuation_action_value_audit import (
        _run,
        _synthetic_inputs,
        _v154_report,
        _v155_report,
        _v156_report,
        _write_inputs,
    )

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationActionValueTargetDatasetTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-action-value-target-dataset"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_action_value_target_dataset"
            ),
        )

    def test_source_validation_requires_v157_expected_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v154_to_v157_inputs(tmpdir, mode="multi_safe")
            v157 = json.loads(paths["v157_report"].read_text(encoding="utf-8"))
            v157["classification"]["primary"] = "unexpected"
            paths["v157_report"].write_text(
                json.dumps(v157, sort_keys=True),
                encoding="utf-8",
            )

            result = run_carrion_survivor_continuation_action_value_target_dataset(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                v156_report_path=paths["v156_report"],
                v157_report_path=paths["v157_report"],
                output_path=paths["v158_report"],
                target_dataset_output_path=paths["v158_dataset"],
            )

            dataset_text = paths["v158_dataset"].read_text(encoding="utf-8")

        self.assertIn(
            "v157_unexpected_classification",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_public_state_action_value_"
                "target_dataset_source_invalid_closed_no_training"
            ),
        )
        self.assertEqual(result["dataset"]["action_value_target_row_count"], 0)
        self.assertEqual(dataset_text, "")
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["runtime_promotion_allowed"])

    def test_multi_action_safe_set_emits_set_targets_without_single_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v154_to_v157_inputs(tmpdir, mode="multi_safe")
            result = run_carrion_survivor_continuation_action_value_target_dataset(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                v156_report_path=paths["v156_report"],
                v157_report_path=paths["v157_report"],
                output_path=paths["v158_report"],
                target_dataset_output_path=paths["v158_dataset"],
            )
            rows = _load_jsonl(paths["v158_dataset"])

        self.assertEqual(
            result["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_SCHEMA_VERSION,
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_public_state_action_value_"
                "target_dataset_support_ready_no_training"
            ),
        )
        self.assertEqual(result["source_validation"]["observed_v157_classification"], EXPECTED_V157_CLASSIFICATION)
        self.assertEqual(result["dataset"]["group_count"], 1)
        self.assertEqual(result["dataset"]["action_value_target_row_count"], 1)
        self.assertEqual(result["dataset"]["unique_winner_group_count"], 0)
        self.assertEqual(result["dataset"]["multi_action_safe_set_group_count"], 1)
        self.assertEqual(result["dataset"]["unresolved_group_count"], 0)
        self.assertTrue(result["dataset"]["action_support_non_collapsed"])
        self.assertEqual(result["dataset"]["dominant_safe_action_share"], 0.5)
        self.assertEqual(stable_payload_digest(rows), result["dataset"]["dataset_digest"])
        self.assertEqual(rows[0]["schema_version"], M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION)
        self.assertEqual(rows[0]["safe_action_set"], ["stay", "eat"])
        self.assertNotIn("robust_winner_action", rows[0])
        self.assertNotIn("label_action", json.dumps(rows[0], sort_keys=True))
        self.assertNotIn("single_label", json.dumps(rows[0], sort_keys=True))
        self.assertEqual(set(rows[0]["public_action_mask"]), set(ACTION_NAMES))
        self.assertEqual(len(rows[0]["action_value_targets"]), len(ACTION_NAMES))
        self.assertFalse(result["artifact_created"])
        self.assertFalse(result["training_ran"])
        self.assertFalse(result["shadow_live_ab_ran"])

    def test_unique_winner_rows_do_not_override_collapsed_support_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v154_to_v157_inputs(tmpdir, mode="unique_eat")
            result = run_carrion_survivor_continuation_action_value_target_dataset(
                v154_report_path=paths["v154_report"],
                v154_dataset_path=paths["v154_dataset"],
                v155_report_path=paths["v155_report"],
                v156_report_path=paths["v156_report"],
                v157_report_path=paths["v157_report"],
                output_path=paths["v158_report"],
                target_dataset_output_path=paths["v158_dataset"],
            )
            rows = _load_jsonl(paths["v158_dataset"])

        self.assertEqual(rows[0]["target_classification"], "unique_robust_winner")
        self.assertEqual(rows[0]["safe_action_set"], ["eat"])
        self.assertEqual(rows[0]["robust_winner_action"], "eat")
        self.assertEqual(result["dataset"]["unresolved_group_count"], 0)
        self.assertFalse(result["dataset"]["action_support_non_collapsed"])
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_public_state_action_value_"
                "target_dataset_action_support_collapsed_closed_no_training"
            ),
        )

    def test_cli_writes_parseable_dataset_and_digest_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_v154_to_v157_inputs(tmpdir, mode="multi_safe")
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"
            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_action_value_target_dataset",
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
                    "--output",
                    str(paths["v158_report"]),
                    "--target-dataset-output",
                    str(paths["v158_dataset"]),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v158_report"].read_text(encoding="utf-8"))
            rows = _load_jsonl(paths["v158_dataset"])

        self.assertIn("training_ran=False", completed.stdout)
        self.assertIn("runtime_promotion_allowed=False", completed.stdout)
        self.assertEqual(stable_payload_digest(rows), written["dataset"]["dataset_digest"])
        exact_payload = dict(written)
        exact_payload.pop("exact_digest", None)
        self.assertEqual(stable_payload_digest(exact_payload), written["exact_digest"])
        for row in rows:
            self.assertIsInstance(row["trainable_public_features"], dict)
            self.assertIsInstance(row["public_action_mask"], dict)
            self.assertIsInstance(row["action_value_targets"], list)


def _write_v154_to_v157_inputs(tmpdir: str, *, mode: str) -> dict[str, Path]:
    rows, branches = _synthetic_inputs(mode="multi_safe")
    if mode == "unique_eat":
        branches = [_unique_eat_branch(branch) for branch in branches]
    elif mode != "multi_safe":
        raise AssertionError(mode)
    v154_report = _v154_report(rows, branches)
    v155_report = _v155_report(rows)
    v156_report = _v156_report(rows)
    paths = _write_inputs(tmpdir, v154_report, rows, v155_report, v156_report)
    paths["v157_report"] = paths["output"]
    paths["v158_report"] = Path(tmpdir) / "v158-report.json"
    paths["v158_dataset"] = Path(tmpdir) / "v158-dataset.jsonl"
    run_carrion_survivor_continuation_action_value_audit(
        v154_report_path=paths["v154_report"],
        v154_dataset_path=paths["v154_dataset"],
        v155_report_path=paths["v155_report"],
        v156_report_path=paths["v156_report"],
        output_path=paths["v157_report"],
    )
    return paths


def _unique_eat_branch(branch: dict[str, object]) -> dict[str, object]:
    payload = dict(branch)
    payload["continuation_runs"] = [
        _run("stay", 0, reproduction=True),
        _run("stay", 1, reproduction=True),
        _run("eat", 0, hydration=True, reproduction=True),
        _run("eat", 1, hydration=True, reproduction=True),
        _run("move_north", 0),
        _run("move_north", 1),
    ]
    return payload


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            assert isinstance(payload, dict)
            rows.append(payload)
    return rows


if __name__ == "__main__":
    unittest.main()
