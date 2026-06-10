from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import OBSERVATION_INPUT_VECTOR_SIZE
from evolution_sim.mind.broad_regression_branch_intervention import (
    MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION,
    BroadRegressionBranchPoint,
    _candidate_actions,
    _canonical_json,
    _configure_manual_summary_run,
    _dataset_leakage_scan,
    _dataset_rows_from_branch_results,
    _execute_action_branch,
    _ordered_regression_seeds,
    _source_precheck,
    write_broad_regression_branch_intervention_dataset,
    write_broad_regression_branch_intervention_report,
)
from evolution_sim.mind.carrion_branch_explore import _branch_state_digest
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

ROOT = Path(__file__).resolve().parents[2]


class MindV3BroadRegressionBranchInterventionTests(unittest.TestCase):
    def test_branch_selection_prioritizes_first_failed_seed_then_regressions(self) -> None:
        live_report = {
            "acceptance": {"first_failing_seed": 5},
            "broad": {
                "per_seed_delta": [
                    {"seed": 13, "alive_delta": -1, "births_delta": 0},
                    {"seed": 5, "alive_delta": -2, "births_delta": 0},
                    {"seed": 41, "alive_delta": 1, "births_delta": 0},
                    {"seed": 37, "alive_delta": 0, "births_delta": -3},
                ]
            },
        }

        self.assertEqual(_ordered_regression_seeds(live_report), (5, 13, 37))

    def test_candidate_actions_keep_baseline_v142_then_public_mask_order(self) -> None:
        point = _synthetic_point(
            baseline_action="eat",
            v142_requested_action="drink",
            action_mask={
                action: action in {"stay", "eat", "drink", "move_east"}
                for action in ACTION_NAMES
            },
        )

        actions = _candidate_actions(point, max_candidate_actions=0)

        self.assertEqual(actions[:3], ["eat", "drink", "stay"])
        self.assertEqual(actions, ["eat", "drink", "stay", "move_east"])

    def test_branch_replay_is_deterministic(self) -> None:
        point = _synthetic_point(
            baseline_action="eat",
            v142_requested_action="stay",
            action_mask={action: action == "stay" for action in ACTION_NAMES},
            ticks=3,
        )
        ref = {
            "alive_agents": 0,
            "births": 0,
            "deaths": 0,
            "requested_action_counts": {},
            "resolved_action_counts": {},
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 0,
            "target_terminal_by_agent": {},
        }

        run = _execute_action_branch(
            point,
            forced_action="stay",
            baseline_ref=ref,
            override_ref=ref,
            verify_replay=True,
        )

        self.assertTrue(run["forced_action_used"])
        self.assertEqual(run["forced_action"], "stay")
        self.assertEqual(run["replay_verification"]["verified"], True)
        self.assertEqual(run["heuristic_action_source_count"], 0)

    def test_dataset_rows_are_leakage_safe_and_keep_metadata_non_trainable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "live.json"
            scorer = Path(tmp) / "scorer.json"
            source.write_text("{}", encoding="utf-8")
            scorer.write_text("{}", encoding="utf-8")
            rows = _dataset_rows_from_branch_results(
                branch_results=[
                    {
                        "branch_id": "b1",
                        "seed": 5,
                        "fixture": "broad",
                        "branch_tick": 0,
                        "record_index": 0,
                        "agent_id": 4,
                        "source_trajectory_path": "private/path.jsonl.gz",
                        "branch_state_digest": "abc",
                        "public_features": {
                            "observation_input": {"values": [0.0, 1.0]},
                            "action_mask": {"stay": True, "eat": True},
                        },
                        "intervention_supported": True,
                        "intervention_label_action": "stay",
                        "best_supported_action_run": {
                            "forced_action": "stay",
                            "replay_digest": "def",
                            "deltas_vs_v142_override": {
                                "alive_agents": 1,
                                "births": 0,
                            },
                            "deltas_vs_baseline": {
                                "alive_agents": 0,
                                "births": 0,
                            },
                        },
                    }
                ],
                source_report_path=source,
                scorer_report_path=scorer,
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(
            row["schema_version"],
            MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION,
        )
        self.assertEqual(row["trainable"]["label"]["action"], "stay")
        self.assertIn("seed", row["metadata"])
        self.assertNotIn("seed", _canonical_json(row["trainable"], indent=None))
        self.assertTrue(_dataset_leakage_scan(rows)["passed"])

    def test_source_integrity_precheck_reports_v142_failures(self) -> None:
        precheck = _source_precheck(
            {
                "schema_version": "wrong",
                "classification": {"primary": "blocked"},
            },
            {
                "schema_version": "wrong",
                "classification": {"primary": "passed"},
                "acceptance": {
                    "first_failed_floor": "carrion_alive",
                    "first_failing_fixture": "carrion_only",
                },
                "matrix": {
                    "broad_seeds": [5],
                    "ticks": 1,
                    "fixture_seeds": [13],
                    "fixture_ticks": 1,
                },
                "broad": {"per_seed_delta": []},
            },
        )

        self.assertFalse(precheck["passed"])
        self.assertIn("v142_scorer_not_ready", precheck["failures"])
        self.assertIn(
            "v142_live_report_not_blocked_on_broad_regression",
            precheck["failures"],
        )

    def test_json_and_jsonl_writes_are_deterministic(self) -> None:
        report = {"schema_version": "test", "z": 2, "a": {"b": 1}}
        rows = [
            {
                "schema_version": "row",
                "trainable": {"features": {"x": 1}, "label": {"action": "stay"}},
                "metadata": {"seed": 5},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.json"
            b = Path(tmp) / "b.json"
            aj = Path(tmp) / "a.jsonl"
            bj = Path(tmp) / "b.jsonl"
            write_broad_regression_branch_intervention_report(report, a)
            write_broad_regression_branch_intervention_report(report, b)
            write_broad_regression_branch_intervention_dataset(rows, aj)
            write_broad_regression_branch_intervention_dataset(rows, bj)

            self.assertEqual(a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8"))
            self.assertEqual(aj.read_text(encoding="utf-8"), bj.read_text(encoding="utf-8"))

    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        script = package["scripts"]["sim:mind:v3:broad-regression-branch-intervention"]

        self.assertIn(
            "evolution_sim.cli.mind_v3_broad_regression_branch_intervention",
            script,
        )


def _synthetic_point(
    *,
    baseline_action: str,
    v142_requested_action: str,
    action_mask: dict[str, bool],
    ticks: int = 1,
) -> BroadRegressionBranchPoint:
    world = SimulationWorld(
        WorldConfig(seed=5, max_ticks=ticks),
        policy=MindV3EvolutionPolicy(seed=5),
    )
    _configure_manual_summary_run(world)
    agent_id = min(int(agent_id) for agent_id in world.agents)
    branch_id = "synthetic-branch"
    return BroadRegressionBranchPoint(
        branch_id=branch_id,
        seed=5,
        fixture="broad",
        ticks=ticks,
        branch_tick=0,
        record_index=0,
        branch_index=0,
        agent_id=agent_id,
        baseline_action=baseline_action,
        v142_requested_action=v142_requested_action,
        v142_resolved_action=v142_requested_action,
        action_mask=action_mask,
        observation_input={"values": [0.0] * OBSERVATION_INPUT_VECTOR_SIZE},
        observation_schema="test",
        observation_digest="digest",
        source_trajectory_path=None,
        branch_state_digest=_branch_state_digest(
            world,
            branch_id=branch_id,
            branch_tick=0,
        ),
        world=world,
    )


if __name__ == "__main__":
    unittest.main()
