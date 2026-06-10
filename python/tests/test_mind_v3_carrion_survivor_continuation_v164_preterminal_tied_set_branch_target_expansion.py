from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind.carrion_survivor_continuation_v163_tied_set_branch_target_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v164_preterminal_tied_set_branch_target_expansion import (
    EXPECTED_V163_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v164_preterminal_tied_set_branch_target_expansion,
    select_preterminal_tied_branch_points,
    summarize_available_preterminal_candidates,
)

try:
    from python.tests.test_mind_v3_carrion_survivor_continuation_v159_scorer_readiness import (
        _write_report_with_exact_digest,
    )
except ModuleNotFoundError:  # pragma: no cover - unittest discovery fallback.
    from test_mind_v3_carrion_survivor_continuation_v159_scorer_readiness import (
        _write_report_with_exact_digest,
    )

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV164PreterminalTiedSetBranchTargetExpansionTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v164-preterminal-tied-set-branch-target-expansion"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v164_preterminal_tied_set_branch_target_expansion"
            ),
        )

    def test_source_invalid_v163_digest_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            v163_report = Path(tmpdir) / "v163-report.json"
            output_path = Path(tmpdir) / "v164-report.json"
            _write_report_with_exact_digest(
                v163_report,
                _minimal_v163_report(),
            )

            result = run_carrion_survivor_continuation_v164_preterminal_tied_set_branch_target_expansion(
                v163_report_path=v163_report,
                output_path=output_path,
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v163_unexpected_exact_digest",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            "source_invalid_closed_no_live_ab",
        )
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_selector_excludes_final_tick_and_covers_rows_without_duplicate_tick(self) -> None:
        predictions, evidence = _selection_fixture()

        available = summarize_available_preterminal_candidates(
            predictions=predictions,
            evidence_records=evidence,
            strict_broad_seeds=(5, 13),
            ticks=120,
            max_branch_tick=100,
        )
        selected = select_preterminal_tied_branch_points(
            predictions=predictions,
            evidence_records=evidence,
            strict_broad_seeds=(5, 13),
            ticks=120,
            max_branch_tick=100,
            max_branch_points_per_seed=2,
        )

        self.assertEqual(
            available["all_strict_broad_tied_candidate_count_by_tick_bucket"],
            {"080-099": 3, "100-119": 5},
        )
        self.assertEqual(
            available[
                "eligible_preterminal_preferred_candidate_count_by_tick_bucket"
            ],
            {"080-099": 2, "100-119": 3},
        )
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(point.branch_tick <= 100 for point in selected))
        self.assertNotIn(119, {point.branch_tick for point in selected})
        self.assertEqual(
            {point.nearest_neighbor_row_index for point in selected if point.seed == 5},
            {4, 7},
        )
        self.assertEqual(
            [point.branch_tick for point in selected if point.seed == 5],
            [100, 99],
        )

    def test_plan_only_valid_source_fails_closed_without_branch_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            v163_report = Path(tmpdir) / "v163-report.json"
            output_path = Path(tmpdir) / "v164-report.json"
            payload = _minimal_v163_report()
            _write_report_with_exact_digest(v163_report, payload)
            written = json.loads(v163_report.read_text(encoding="utf-8"))

            result = run_carrion_survivor_continuation_v164_preterminal_tied_set_branch_target_expansion(
                v163_report_path=v163_report,
                output_path=output_path,
                expected_v163_exact_digest=written["exact_digest"],
                attempt_branch_replay=False,
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v162_source_load_failed",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            "source_invalid_closed_no_live_ab",
        )
        self.assertFalse(result["runtime_artifact_created"])
        self.assertFalse(result["live_ab_allowed"])
        self.assertFalse(result["promotion_authorized"])

    def test_cli_writes_source_invalid_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "v163-invalid.json"
            output_path = Path(tmpdir) / "v164-report.json"
            _write_report_with_exact_digest(
                report_path,
                {
                    "schema_version": "wrong",
                    "policy": "wrong",
                    "classification": {"primary": "wrong", "labels": ["wrong"]},
                    "runtime_action_selection_changed": False,
                    "runtime_artifact_created": False,
                    "live_ab_allowed": False,
                    "promotion_authorized": False,
                },
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v164_preterminal_tied_set_branch_target_expansion",
                    "--v163-report",
                    str(report_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v164_preterminal_tied_set_branch_target_expansion_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION,
        )
        self.assertEqual(
            written["classification"]["primary"],
            "source_invalid_closed_no_live_ab",
        )
        self.assertIn("exact_digest", written)


def _minimal_v163_report() -> dict[str, object]:
    return {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY,
        "classification": {
            "primary": EXPECTED_V163_CLASSIFICATION,
            "labels": [EXPECTED_V163_CLASSIFICATION],
        },
        "inputs": {
            "v162_report": "missing-v162.json",
            "v160_artifact": "missing-artifact.json",
            "trajectory_paths": [],
            "trajectory_glob": "missing-*.jsonl.gz",
        },
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "lifecycle_proof": {
            "runtime_action_selection_changed": False,
            "runtime_artifact_created": False,
            "live_ab_allowed": False,
            "promotion_authorized": False,
        },
    }


def _selection_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = [
        (5, 119, 7, ["stay", "eat", "move_north", "move_south"], 1),
        (5, 100, 7, ["stay", "eat", "move_north", "move_south"], 2),
        (5, 100, 4, ["stay", "eat", "move_north"], 3),
        (5, 99, 4, ["stay", "eat", "move_north"], 4),
        (13, 100, 7, ["stay", "eat", "move_north", "move_south"], 5),
        (13, 99, 4, ["stay", "eat", "move_north"], 6),
        (13, 90, 7, ["stay", "eat"], 7),
        (13, 119, 4, ["stay", "eat", "move_north"], 8),
    ]
    predictions: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    for index, (seed, tick, row, top_set, agent_id) in enumerate(rows):
        predictions.append(
            {
                "record_index": index,
                "source_seed": seed,
                "nearest_neighbor_row_index": row,
                "predicted_action": "stay",
                "top_value_candidate_set": top_set,
            }
        )
        evidence.append(
            {
                "source_path": f"seed-{seed}.jsonl.gz",
                "source_seed": seed,
                "line_number": index + 1,
                "record": {
                    "tick": tick,
                    "agent_id": agent_id,
                    "requested_action": "stay",
                    "resolved_action": "stay",
                    "public_action_mask": {
                        "stay": True,
                        "eat": True,
                        "move_north": True,
                        "move_south": True,
                    },
                    "observation_input": {"energy_ratio": 0.5},
                    "observation_schema": "mind_observation_v3",
                    "observation_digest": f"digest-{index}",
                    "before": {"agent_id": agent_id, "tick": tick},
                },
            }
        )
    return predictions, evidence


if __name__ == "__main__":
    unittest.main()
