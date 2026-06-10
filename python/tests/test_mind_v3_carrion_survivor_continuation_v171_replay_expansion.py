from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.cli.mind_v3_carrion_survivor_continuation_v171_replay_expansion import (
    build_parser,
)
from evolution_sim.mind.carrion_survivor_continuation_v170_diagnostic_portfolio_matrix import (
    M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    EXPECTED_V170_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION,
    execute_v171_shard,
    load_v170_shard_plan,
    merge_v171_shards,
    trainable_payload_leakage_scan,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV171ReplayExpansionTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v171-replay-expansion"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v171_replay_expansion"
            ),
        )

    def test_v170_source_digest_mismatch_closes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_v170_inputs(tmpdir)

            report = execute_v171_shard(
                v170_report_path=paths["v170_report"],
                shard_plan_path=paths["shard_plan"],
                shard_id=str(payloads["plan_rows"][0]["shard_id"]),
                seed_include=int(payloads["plan_rows"][0]["seed"]),
                branch_windows=["92:102"],
                output_path=paths["shard_report"],
                dataset_output_path=paths["shard_dataset"],
                expected_v170_exact_digest="wrong",
                expected_v170_shard_plan_digest=stable_payload_digest(
                    payloads["plan_rows"]
                ),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v170_unexpected_exact_digest",
            report["source_validation"]["failures"],
        )
        self.assertTrue(report["classification"]["primary"].endswith("closed_invalid"))
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertEqual(report["exact_digest"], _digest_without_exact(report))

    def test_v170_jsonl_command_shapes_parse_without_explicit_shard_plan(self) -> None:
        plan_path = (
            ROOT
            / "output/mind/mind-v3-v170-carrion-survivor-continuation-replay-expansion-shards.jsonl"
        )
        if not plan_path.exists():
            self.skipTest("real v170 shard-plan artifact is not present")
        rows = load_v170_shard_plan(plan_path)
        parser = build_parser()

        for row in rows:
            command = list(row["expected_command_shape"])
            self.assertNotIn("--shard-plan", command)
            args = command[command.index("--") + 1 :]
            parsed = parser.parse_args(args)
            self.assertEqual(parsed.shard_id, row["shard_id"])
            self.assertEqual(parsed.seed_include, row["seed"])
            self.assertTrue(parsed.fail_on_partial_shard)

    def test_trainable_payload_leakage_scan_rejects_identity_and_safe_action_keys(self) -> None:
        scan = trainable_payload_leakage_scan(
            [
                {
                    "public_observation": {"bucket": 1.0},
                    "action_mask": {"eat": True},
                    "source_seed": 13,
                    "target_safe_action": "eat",
                }
            ]
        )

        self.assertFalse(scan["passed"])
        tokens = {failure["token"] for failure in scan["failures"]}
        self.assertIn("seed", tokens)
        self.assertIn("target_safe", tokens)

    def test_merge_partial_shards_without_allow_closes_as_partial_not_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_v170_inputs(tmpdir)
            shard_reports = []
            shard_datasets = []
            for row in payloads["plan_rows"]:
                shard_report = Path(tmpdir) / f"{row['shard_id']}.json"
                shard_dataset = Path(tmpdir) / f"{row['shard_id']}.jsonl"
                _write_report(shard_report, _synthetic_partial_shard_report(row))
                shard_dataset.write_text("", encoding="utf-8")
                shard_reports.append(shard_report)
                shard_datasets.append(shard_dataset)

            report = merge_v171_shards(
                v170_report_path=paths["v170_report"],
                shard_plan_path=paths["shard_plan"],
                merge_shard_reports=shard_reports,
                merge_shard_datasets=shard_datasets,
                output_path=paths["merged_report"],
                dataset_output_path=paths["merged_dataset"],
                expected_v170_exact_digest=payloads["v170_report"]["exact_digest"],
                expected_v170_shard_plan_digest=stable_payload_digest(
                    payloads["plan_rows"]
                ),
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["merge_validation"]["passed"])
        self.assertTrue(report["partial_shard_status"]["partial"])
        self.assertTrue(
            report["classification"]["primary"].endswith(
                "partial_shard_closed_no_training"
            )
        )
        self.assertEqual(report["exact_digest"], _digest_without_exact(report))
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["runtime_artifact_created"])

    def test_merge_missing_planned_shards_is_partial_not_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_v170_inputs(tmpdir)
            row = payloads["plan_rows"][3]
            shard_report = Path(tmpdir) / f"{row['shard_id']}.json"
            shard_dataset = Path(tmpdir) / f"{row['shard_id']}.jsonl"
            _write_report(shard_report, _synthetic_complete_limited_shard_report(row))
            shard_dataset.write_text("", encoding="utf-8")

            report = merge_v171_shards(
                v170_report_path=paths["v170_report"],
                shard_plan_path=paths["shard_plan"],
                merge_shard_reports=[shard_report],
                merge_shard_datasets=[shard_dataset],
                output_path=paths["merged_report"],
                dataset_output_path=paths["merged_dataset"],
                expected_v170_exact_digest=payloads["v170_report"]["exact_digest"],
                expected_v170_shard_plan_digest=stable_payload_digest(
                    payloads["plan_rows"]
                ),
            )

        self.assertTrue(report["merge_validation"]["passed"])
        self.assertFalse(report["merge_validation"]["all_required_shards_present"])
        self.assertTrue(report["partial_shard_status"]["partial"])
        self.assertTrue(
            report["classification"]["primary"].endswith(
                "partial_shard_closed_no_training"
            )
        )

    def test_cli_merge_writes_report_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, payloads = _write_v170_inputs(tmpdir)
            shard_reports = []
            shard_datasets = []
            for row in payloads["plan_rows"]:
                shard_report = Path(tmpdir) / f"{row['shard_id']}.json"
                shard_dataset = Path(tmpdir) / f"{row['shard_id']}.jsonl"
                _write_report(shard_report, _synthetic_partial_shard_report(row))
                shard_dataset.write_text("", encoding="utf-8")
                shard_reports.extend(["--merge-shard-report", str(shard_report)])
                shard_datasets.extend(["--merge-shard-dataset", str(shard_dataset)])
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"
            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v171_replay_expansion",
                    "--merge-shards",
                    "--v170-report",
                    str(paths["v170_report"]),
                    "--shard-plan",
                    str(paths["shard_plan"]),
                    "--output",
                    str(paths["merged_report"]),
                    "--dataset-output",
                    str(paths["merged_dataset"]),
                    "--expected-v170-exact-digest",
                    str(payloads["v170_report"]["exact_digest"]),
                    "--expected-v170-shard-plan-digest",
                    stable_payload_digest(payloads["plan_rows"]),
                    *shard_reports,
                    *shard_datasets,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["merged_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v171_replay_expansion_report=",
            completed.stdout,
        )
        self.assertIn("classification=", completed.stdout)
        self.assertEqual(written["exact_digest"], _digest_without_exact(written))


def _write_v170_inputs(tmpdir: str) -> tuple[dict[str, Path], dict[str, object]]:
    root = Path(tmpdir)
    paths = {
        "v170_report": root / "v170.json",
        "shard_plan": root / "v170-shards.jsonl",
        "shard_report": root / "shard.json",
        "shard_dataset": root / "shard.jsonl",
        "merged_report": root / "merged.json",
        "merged_dataset": root / "merged.jsonl",
    }
    plan_rows = _plan_rows()
    _write_jsonl(paths["shard_plan"], plan_rows)
    v170_report = _report(
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_SCHEMA_VERSION
            ),
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_POLICY,
            "classification": {"primary": EXPECTED_V170_CLASSIFICATION},
            "inputs": {"v165_dataset": str(root / "unused-v165.jsonl")},
            "shard_plan_output": {
                "shard_plan_digest": stable_payload_digest(plan_rows),
                "jsonl_row_count": len(plan_rows),
            },
            "portfolio_lanes": {
                "set_valued_ranking": {
                    "best_matrix_entry": {"average_set_width": 3.5}
                }
            },
            "diagnostics_only": True,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "runtime_observation_schema_changed": False,
            "runtime_policy_changed": False,
            "replay_viewer_schema_changed": False,
            "threshold_tuning_ran": False,
            "k_tuning_ran": False,
            "training_ran": False,
            "shadow_eval_ran": False,
            "live_ab_allowed": False,
            "promotion_authorized": False,
            "staging_authorized": False,
            "commit_authorized": False,
            "reset_authorized": False,
            "clean_authorized": False,
        }
    )
    _write_report(paths["v170_report"], v170_report)
    return paths, {"plan_rows": plan_rows, "v170_report": v170_report}


def _plan_rows() -> list[dict[str, object]]:
    seeds = [13, 19, 29, 41, 5, 37]
    rows = []
    for seed in seeds:
        shard_id = f"v171-carrion-survivor-continuation-expansion-seed-{seed:03d}"
        rows.append(
            {
                "schema_version": "m3_carrion_survivor_continuation_v170_replay_expansion_shard_plan_row_v1",
                "policy": "diagnostics_only_m3_carrion_survivor_continuation_v170_replay_expansion_shard_planner_v1",
                "shard_id": shard_id,
                "seed": seed,
                "rationale": ["synthetic"],
                "source_row_count": 1,
                "failed_row_count": 1,
                "proposed_branch_windows": [
                    {
                        "source_tick": 97,
                        "start_tick": 92,
                        "end_tick": 102,
                        "source_row_index": 0,
                        "safe_action_set": ["eat"],
                        "window_radius": 5,
                        "failed_row_source": True,
                    }
                ],
                "expected_command_shape": [
                    "npm",
                    "run",
                    "sim:mind:v3:carrion-survivor-continuation-v171-replay-expansion",
                    "--",
                    "--shard-id",
                    shard_id,
                    "--seed-include",
                    str(seed),
                    "--branch-window",
                    "92:102",
                    "--output",
                    f"output/mind/shards/{shard_id}.json",
                    "--dataset-output",
                    f"output/mind/shards/{shard_id}.jsonl",
                    "--fail-on-partial-shard",
                ],
                "plan_only_not_evidence": True,
                "long_branch_replay_ran": False,
                "training_authorized": False,
                "runtime_action_selection_changed": False,
            }
        )
    return rows


def _synthetic_partial_shard_report(plan_row: dict[str, object]) -> dict[str, object]:
    return _report(
        {
            "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION,
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY,
            "mode": "shard",
            "inputs": {
                "shard_id": plan_row["shard_id"],
                "seed_include": plan_row["seed"],
            },
            "source_validation": {"passed": True},
            "shard_validation": {"passed": True},
            "shard_plan_row": plan_row,
            "selection": {"selected_branch_point_count": 0},
            "branch_materialization": {"materialized_branch_point_count": 0},
            "branch_results": [],
            "metrics": {
                "candidate_run_count": 0,
                "replay_verified_run_count": 0,
                "all_replays_verified": False,
            },
            "leakage_scan": {"passed": True},
            "partial_shard_status": {"partial": True},
            "dataset": {"row_count": 0, "dataset_digest": stable_payload_digest([])},
            "classification": {
                "primary": (
                    "m3_carrion_survivor_continuation_v171_replay_expansion_"
                    "partial_shard_closed_no_training"
                )
            },
            "diagnostics_only": True,
            "training_ran": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
        }
    )


def _synthetic_complete_limited_shard_report(plan_row: dict[str, object]) -> dict[str, object]:
    return _report(
        {
            "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION,
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY,
            "mode": "shard",
            "inputs": {
                "shard_id": plan_row["shard_id"],
                "seed_include": plan_row["seed"],
            },
            "source_validation": {"passed": True},
            "shard_validation": {"passed": True},
            "shard_plan_row": plan_row,
            "selection": {"selected_branch_point_count": 1},
            "branch_materialization": {"materialized_branch_point_count": 1},
            "branch_results": [],
            "metrics": {
                "candidate_run_count": 0,
                "replay_verified_run_count": 0,
                "all_replays_verified": False,
            },
            "leakage_scan": {"passed": True},
            "partial_shard_status": {"partial": False},
            "dataset": {"row_count": 0, "dataset_digest": stable_payload_digest([])},
            "classification": {
                "primary": (
                    "m3_carrion_survivor_continuation_v171_replay_expansion_"
                    "replay_expansion_support_limited_no_training"
                )
            },
            "diagnostics_only": True,
            "training_ran": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
        }
    )


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


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
