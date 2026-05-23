from __future__ import annotations

import json
from random import Random
from types import SimpleNamespace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.mind import carrion_branch_explore
from evolution_sim.cli import (
    mind_v3_branch_action_oracle_audit,
    mind_v3_carrion_branch_explore,
)
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.branch_action_oracle_audit import (
    DEPLETED_RESOURCE_TRAP_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
    FAILURE_FRONTIER_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
    MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
    MODE_BALANCED_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
    build_branch_action_oracle_audit_report,
)
from evolution_sim.mind.carrion_branch_explore import (
    MIND_V3_CARRION_BRANCH_EXPLORE_CURRENT_POLICY_SOURCE,
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
    CarrionBranchExploreError,
    build_carrion_branch_explore_report,
    build_current_policy_carrion_branch_explore_report,
)
from evolution_sim.mind.carrion_counterfactual import CarrionCounterfactualPolicy
from evolution_sim.mind.evolution import founder_mind_v3_metadata
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy


class MindV3CarrionBranchExploreTests(unittest.TestCase):
    def test_branch_explore_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-branch-explore"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_branch_explore"
            ),
        )
        self.assertEqual(
            package["scripts"]["sim:mind:v3:branch-action-oracle-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_branch_action_oracle_audit"
            ),
        )

    def test_report_branches_from_post_contact_state_and_verifies_replay(self) -> None:
        report = build_carrion_branch_explore_report(
            seeds=(29,),
            ticks=8,
            continuation_scripts=("hydration_safe_carrion_cycle",),
            verify_replay=True,
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        )
        self.assertTrue(report["scope"]["state_restore_available"])
        self.assertEqual(report["aggregate"]["branch_point_count"], 1)
        self.assertEqual(report["aggregate"]["branch_run_count"], 1)
        self.assertEqual(report["aggregate"]["positive_seed_count"], 1)
        self.assertTrue(report["aggregate"]["replay_verified"])
        outcome = report["aggregate"]["outcome_metrics"]
        self.assertEqual(outcome["terminal_survivor_run_count"], 1)
        self.assertGreater(outcome["total_terminal_alive_agents"], 0)
        self.assertIn("total_scavenger_animal_resource_events", outcome)
        self.assertTrue(report["acceptance"]["diagnostic_acceptance_passed"])
        branch = report["branch_points"][0]
        self.assertEqual(branch["contact"]["food_source"], "carcass")
        run = report["branch_runs"][0]
        self.assertGreater(run["alive_agents"], 0)
        self.assertIn("outcome_metrics", run)
        self.assertIn("scavenging", run["outcome_metrics"])
        self.assertEqual(run["heuristic_action_source_count"], 0)
        self.assertTrue(run["replay_verification"]["verified"])

    def test_branch_explore_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "branch-explore.json"

            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_branch_explore",
                    "--seeds",
                    "29",
                    "--ticks",
                    "8",
                    "--continuation-script",
                    "hydration_safe_carrion_cycle",
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_branch_explore.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        )
        self.assertEqual(payload["aggregate"]["branch_point_count"], 1)
        self.assertTrue(payload["acceptance"]["diagnostic_acceptance_passed"])

    def test_current_policy_source_extracts_intended_candidate(self) -> None:
        metadata = founder_mind_v3_metadata(agent_id=7, rng=Random(11))
        with TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "search.json"
            source_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mind_v3_evolution_search_v1",
                        "search": {"seeds": [13], "holdout_seeds": [29]},
                        "generations": [
                            {
                                "generation_index": 0,
                                "candidates": [
                                    {
                                        "candidate_id": "wrong",
                                        "controller_metadata": founder_mind_v3_metadata(
                                            agent_id=8,
                                            rng=Random(12),
                                        ),
                                    },
                                    {
                                        "candidate_id": "target",
                                        "controller_metadata": metadata,
                                    },
                                ],
                            }
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            point = _test_branch_point(
                seed=13,
                branch_id="branch-a",
                source_candidate_id="target",
            )

            with patch(
                "evolution_sim.mind.carrion_branch_explore."
                "_discover_current_policy_branch_points",
                return_value={
                    "branch_points": [point],
                    "report": {
                        "seed": 13,
                        "source_action_source_counts": {
                            "mind_v3_autonomous_evolution_policy_v1": 1,
                        },
                        "source_policy_id_counts": {
                            "mind_v3_autonomous_evolution_policy": 1,
                        },
                        "source_unsupported_requested_action_count": 0,
                        "source_unsupported_resolved_action_count": 0,
                    },
                },
            ), patch(
                "evolution_sim.mind.carrion_branch_explore._run_branch_continuation",
                return_value=_test_branch_run(
                    seed=13,
                    branch_id="branch-a",
                    script="hydration_safe_carrion_cycle",
                    alive=1,
                ),
            ):
                report = build_current_policy_carrion_branch_explore_report(
                    source_search_report_path=source_path,
                    source_candidate_id="target",
                    seeds=(13,),
                    ticks=8,
                    continuation_scripts=("hydration_safe_carrion_cycle",),
                    verify_replay=False,
                )

        self.assertEqual(
            report["source_mode"],
            MIND_V3_CARRION_BRANCH_EXPLORE_CURRENT_POLICY_SOURCE,
        )
        self.assertEqual(report["source"]["source_candidate_id"], "target")
        self.assertEqual(
            report["source"]["source_candidate_architecture"],
            metadata["architecture"],
        )
        self.assertEqual(
            report["source"]["source_replay_heuristic_action_source_count"],
            0,
        )
        self.assertEqual(
            report["branch_points"][0]["source_policy_kind"],
            "current_mind_v3_candidate_policy",
        )
        self.assertEqual(report["branch_runs"][0]["continuation_script"], "hydration_safe_carrion_cycle")

    def test_current_policy_branch_points_use_mind_v3_source_policy(self) -> None:
        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(17))
        captured: dict[str, object] = {}

        class FakeWorld:
            def __init__(self) -> None:
                self.tick = 0
                self.births = 0
                self.deaths = 0
                self.trajectory_records: list[dict[str, object]] = []
                self.tick_trajectory_records: list[dict[str, object]] = []

            def _run_tick(self) -> None:
                self.tick_trajectory_records = [
                    {
                        "tick": self.tick,
                        "agent_id": 4,
                        "requested_action": "eat",
                        "resolved_action": "eat",
                        "action_valid": True,
                        "resolution_action_valid": True,
                        "action_source": "mind_v3_autonomous_evolution_policy_v1",
                        "policy_id": "mind_v3_autonomous_evolution_policy",
                        "action_mask": {"eat": True, "drink": False},
                        "before": {
                            "x": 4,
                            "y": 5,
                            "energy_ratio": 0.4,
                            "hydration_ratio": 0.3,
                            "health_ratio": 0.8,
                            "alive": True,
                        },
                        "after": {
                            "x": 4,
                            "y": 5,
                            "energy_ratio": 0.7,
                            "hydration_ratio": 0.25,
                            "health_ratio": 0.8,
                            "alive": True,
                        },
                        "outcome": {
                            "feeding": {
                                "ate": True,
                                "food_source": "carcass",
                                "gained_energy": 0.3,
                                "consumed": 0.3,
                            }
                        },
                    }
                ]
                self.trajectory_records.extend(self.tick_trajectory_records)

            def alive_agents(self) -> list[object]:
                return [object()]

        def fake_fixture_world(**kwargs: object) -> FakeWorld:
            policy = kwargs["policy"]
            captured["policy"] = policy
            self.assertIsInstance(policy, MindV3EvolutionPolicy)
            self.assertNotIsInstance(policy, CarrionCounterfactualPolicy)
            return FakeWorld()

        with patch(
            "evolution_sim.mind.carrion_branch_explore._fixture_world",
            side_effect=fake_fixture_world,
        ), patch(
            "evolution_sim.mind.carrion_branch_explore._branch_state_digest",
            return_value="digest-current",
        ):
            discovered = carrion_branch_explore._discover_current_policy_branch_points(
                seed=13,
                ticks=2,
                fixture_name="carrion_only",
                source_candidate_id="g-test",
                source_candidate_metadata=metadata,
                max_branch_points=1,
                min_branch_tick=0,
            )

        self.assertIsInstance(captured["policy"], MindV3EvolutionPolicy)
        self.assertEqual(len(discovered["branch_points"]), 1)
        point = discovered["branch_points"][0]
        self.assertEqual(point.base_script, "current_policy:g-test")
        self.assertEqual(point.source_candidate_id, "g-test")
        self.assertEqual(point.public_observation_bucket["hydration_bin"], "low")
        self.assertEqual(
            discovered["report"]["source_action_source_counts"],
            {"mind_v3_autonomous_evolution_policy_v1": 1},
        )

    def test_current_policy_split_metadata_and_support_are_deterministic(self) -> None:
        metadata = founder_mind_v3_metadata(agent_id=9, rng=Random(19))
        with TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "search.json"
            source_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mind_v3_evolution_search_v1",
                        "search": {"seeds": [13], "holdout_seeds": [29]},
                        "best_candidate": {
                            "candidate_id": "g-target",
                            "controller_metadata": metadata,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            points = [
                _test_branch_point(
                    seed=13,
                    branch_id="branch-train",
                    digest="digest-train",
                    source_candidate_id="g-target",
                ),
                _test_branch_point(
                    seed=29,
                    branch_id="branch-heldout",
                    digest="digest-heldout",
                    source_candidate_id="g-target",
                ),
            ]

            def fake_discovery(**kwargs: object) -> dict[str, object]:
                seed = int(kwargs["seed"])
                selected = [point for point in points if point.seed == seed]
                return {
                    "branch_points": selected,
                    "report": {
                        "seed": seed,
                        "source_action_source_counts": {
                            "mind_v3_autonomous_evolution_policy_v1": len(selected),
                        },
                        "source_policy_id_counts": {
                            "mind_v3_autonomous_evolution_policy": len(selected),
                        },
                        "source_unsupported_requested_action_count": 0,
                        "source_unsupported_resolved_action_count": 0,
                    },
                }

            def fake_run(
                branch_point: object,
                *,
                continuation_script: str,
                **_: object,
            ) -> dict[str, object]:
                point = branch_point
                assert isinstance(point, carrion_branch_explore._BranchPoint)
                alive = (
                    1
                    if point.seed == 13
                    and continuation_script == "hydration_safe_carrion_cycle"
                    else 0
                )
                return _test_branch_run(
                    seed=point.seed,
                    branch_id=point.branch_id,
                    script=continuation_script,
                    alive=alive,
                )

            with patch(
                "evolution_sim.mind.carrion_branch_explore."
                "_discover_current_policy_branch_points",
                side_effect=fake_discovery,
            ), patch(
                "evolution_sim.mind.carrion_branch_explore._run_branch_continuation",
                side_effect=fake_run,
            ):
                first = build_current_policy_carrion_branch_explore_report(
                    source_search_report_path=source_path,
                    source_candidate_id="g-target",
                    seeds=(13, 29),
                    ticks=8,
                    continuation_scripts=(
                        "hydration_safe_carrion_cycle",
                        "conserve_after_carrion",
                    ),
                    verify_replay=False,
                )
                second = build_current_policy_carrion_branch_explore_report(
                    source_search_report_path=source_path,
                    source_candidate_id="g-target",
                    seeds=(13, 29),
                    ticks=8,
                    continuation_scripts=(
                        "hydration_safe_carrion_cycle",
                        "conserve_after_carrion",
                    ),
                    verify_replay=False,
                )

        self.assertEqual(
            first["train_heldout_split_metadata"],
            second["train_heldout_split_metadata"],
        )
        split = first["train_heldout_split_metadata"]
        self.assertEqual(split["by_seed"], {"13": "source_search_train", "29": "source_search_holdout"})
        self.assertEqual(
            split["by_branch_state_digest"],
            [
                {
                    "branch_id": "branch-train",
                    "branch_state_digest": "digest-train",
                    "seed": 13,
                    "split": "source_search_train",
                },
                {
                    "branch_id": "branch-heldout",
                    "branch_state_digest": "digest-heldout",
                    "seed": 29,
                    "split": "source_search_holdout",
                },
            ],
        )
        support = first["aggregate"]["post_contact_survival_rate_by_seed_and_script"]
        self.assertEqual(
            support["13"]["hydration_safe_carrion_cycle"][
                "terminal_survivor_run_count"
            ],
            1,
        )
        self.assertEqual(
            support["29"]["hydration_safe_carrion_cycle"][
                "terminal_survivor_run_count"
            ],
            0,
        )
        self.assertEqual(
            first["aggregate"]["unrecoverable_state_summary"][
                "unrecoverable_branch_state_count"
            ],
            1,
        )

    def test_current_policy_missing_candidate_id_fails_clearly(self) -> None:
        metadata = founder_mind_v3_metadata(agent_id=2, rng=Random(23))
        with TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "search.json"
            source_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mind_v3_evolution_search_v1",
                        "best_candidate": {
                            "candidate_id": "present",
                            "controller_metadata": metadata,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                CarrionBranchExploreError,
                "source candidate id 'missing' was not found",
            ):
                build_current_policy_carrion_branch_explore_report(
                    source_search_report_path=source_path,
                    source_candidate_id="missing",
                    seeds=(13,),
                    ticks=8,
                    continuation_scripts=("hydration_safe_carrion_cycle",),
                    verify_replay=False,
                )

    def test_current_policy_branch_explore_cli_writes_source_report(self) -> None:
        metadata = founder_mind_v3_metadata(agent_id=5, rng=Random(29))
        with TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "search.json"
            output_path = Path(tmpdir) / "branch-current.json"
            source_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mind_v3_evolution_search_v1",
                        "best_candidate": {
                            "candidate_id": "g-cli",
                            "controller_metadata": metadata,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            point = _test_branch_point(
                seed=13,
                branch_id="branch-cli",
                source_candidate_id="g-cli",
            )
            with patch(
                "evolution_sim.mind.carrion_branch_explore."
                "_discover_current_policy_branch_points",
                return_value={
                    "branch_points": [point],
                    "report": {
                        "seed": 13,
                        "source_action_source_counts": {
                            "mind_v3_autonomous_evolution_policy_v1": 1,
                        },
                        "source_policy_id_counts": {
                            "mind_v3_autonomous_evolution_policy": 1,
                        },
                        "source_unsupported_requested_action_count": 0,
                        "source_unsupported_resolved_action_count": 0,
                    },
                },
            ), patch(
                "evolution_sim.mind.carrion_branch_explore._run_branch_continuation",
                return_value=_test_branch_run(
                    seed=13,
                    branch_id="branch-cli",
                    script="hydration_safe_carrion_cycle",
                    alive=1,
                ),
            ), patch(
                "sys.argv",
                [
                    "mind_v3_carrion_branch_explore",
                    "--source-search-report",
                    str(source_path),
                    "--source-candidate-id",
                    "g-cli",
                    "--seeds",
                    "13",
                    "--ticks",
                    "8",
                    "--continuation-script",
                    "hydration_safe_carrion_cycle",
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_branch_explore.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["source"]["source_candidate_id"], "g-cli")
        self.assertEqual(payload["aggregate"]["branch_point_count"], 1)
        self.assertEqual(
            payload["scope"]["source_action_semantics"],
            "source replay is autonomous Mind v3 with no heuristic fallback",
        )

    def test_branch_action_oracle_audit_finds_changed_action_gain(self) -> None:
        report = build_branch_action_oracle_audit_report(
            seeds=(37,),
            ticks=70,
            candidate_actions=("drink", "eat", "stay"),
            max_branch_points_per_seed=1,
            verify_replay=True,
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
        )
        self.assertGreaterEqual(report["aggregate"]["branch_point_count"], 1)
        self.assertGreaterEqual(report["aggregate"]["oracle_changed_action_count"], 1)
        self.assertGreaterEqual(
            report["aggregate"]["terminal_alive_gain_total_vs_logged"],
            1,
        )
        self.assertEqual(report["aggregate"]["heuristic_action_source_count"], 0)
        self.assertTrue(report["aggregate"]["replay_verified"])
        self.assertTrue(report["acceptance"]["diagnostic_acceptance_passed"])
        first = report["aggregate"]["first_material_oracle_gain"]
        self.assertIsNotNone(first)
        self.assertNotEqual(first["logged_action"], first["oracle_best_action"])
        action_run = report["branch_results"][0]["action_runs"][0]
        self.assertIn("first_action_outcome", action_run)
        self.assertIn("target_horizon_trace", action_run)
        self.assertIn("population_horizon_trace", action_run)
        self.assertGreaterEqual(len(action_run["target_horizon_trace"]), 1)
        self.assertGreaterEqual(len(action_run["population_horizon_trace"]), 1)
        policy_state = report["branch_points"][0]["policy_state"]
        self.assertEqual(
            policy_state["observation_input"]["schema_version"],
            "mind_observation_v3",
        )
        self.assertEqual(
            set(policy_state["action_mask"]),
            set(ACTION_NAMES),
        )
        self.assertIn("public_history_trace", policy_state)
        self.assertIsInstance(policy_state["public_history_trace"], list)

    def test_branch_action_oracle_audit_can_select_mode_balanced_points(self) -> None:
        report = build_branch_action_oracle_audit_report(
            seeds=(37,),
            ticks=30,
            candidate_actions=tuple(ACTION_NAMES),
            target_labels=tuple(ACTION_NAMES),
            max_branch_points_per_seed=2,
            min_oracle_changed_action_count=0,
            min_terminal_alive_gain_total=0,
            verify_replay=False,
            branch_selection_policy=MODE_BALANCED_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
        )

        discovery = report["discovery"][0]

        self.assertEqual(
            report["contract"]["branch_selection_policy"],
            MODE_BALANCED_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
        )
        self.assertTrue(discovery["scanned_all_eligible_rows"])
        self.assertGreaterEqual(discovery["eligible_row_count"], 1)
        self.assertIn("skipped_row_counts", discovery)
        self.assertIn("selected_bucket_counts", discovery)

    def test_branch_action_oracle_audit_can_select_failure_frontier_points(self) -> None:
        report = build_branch_action_oracle_audit_report(
            seeds=(37,),
            ticks=40,
            candidate_actions=tuple(ACTION_NAMES),
            target_labels=tuple(ACTION_NAMES),
            max_branch_points_per_seed=3,
            min_oracle_changed_action_count=0,
            min_terminal_alive_gain_total=0,
            verify_replay=False,
            branch_selection_policy=FAILURE_FRONTIER_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
        )

        discovery = report["discovery"][0]

        self.assertEqual(
            report["contract"]["branch_selection_policy"],
            FAILURE_FRONTIER_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
        )
        self.assertTrue(discovery["scanned_all_eligible_rows"])
        self.assertGreaterEqual(discovery["eligible_row_count"], 1)
        self.assertIn("eligible_multi_move_reposition_row_count", discovery)
        self.assertIn("selected_multi_move_reposition_row_count", discovery)
        self.assertIn("selected_failure_frontier_score_summary", discovery)

    def test_branch_action_oracle_audit_can_select_depleted_resource_traps(self) -> None:
        report = build_branch_action_oracle_audit_report(
            seeds=(41,),
            ticks=120,
            candidate_actions=tuple(ACTION_NAMES),
            target_labels=tuple(ACTION_NAMES),
            max_branch_points_per_seed=1,
            min_oracle_changed_action_count=0,
            min_terminal_alive_gain_total=0,
            verify_replay=False,
            branch_selection_policy=DEPLETED_RESOURCE_TRAP_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
        )

        discovery = report["discovery"][0]

        self.assertEqual(
            report["contract"]["branch_selection_policy"],
            DEPLETED_RESOURCE_TRAP_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
        )
        self.assertTrue(discovery["scanned_all_eligible_rows"])
        self.assertGreaterEqual(
            discovery["eligible_depleted_resource_trap_row_count"],
            1,
        )
        self.assertEqual(
            discovery["selected_depleted_resource_trap_row_count"],
            report["aggregate"]["branch_point_count"],
        )
        self.assertIn("selected_depleted_resource_trap_score_summary", discovery)

    def test_branch_action_oracle_audit_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "branch-action-oracle.json"

            with patch(
                "sys.argv",
                [
                    "mind_v3_branch_action_oracle_audit",
                    "--seeds",
                    "37",
                    "--ticks",
                    "30",
                    "--max-branch-points-per-seed",
                    "1",
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_branch_action_oracle_audit.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(payload["aggregate"]["branch_point_count"], 1)
        self.assertIn("acceptance", payload)


def _test_branch_point(
    *,
    seed: int,
    branch_id: str,
    source_candidate_id: str,
    digest: str | None = None,
) -> carrion_branch_explore._BranchPoint:
    return carrion_branch_explore._BranchPoint(
        branch_id=branch_id,
        seed=seed,
        fixture_name="carrion_only",
        branch_tick=3,
        branch_index=0,
        base_script=f"current_policy:{source_candidate_id}",
        source_policy_kind="current_mind_v3_candidate_policy",
        source_candidate_id=source_candidate_id,
        contact={
            "tick": 3,
            "agent_id": 4,
            "requested_action": "eat",
            "resolved_action": "eat",
            "food_source": "carcass",
            "gained_energy": 0.3,
            "consumed": 0.3,
            "before": {"energy_ratio": 0.4, "hydration_ratio": 0.4, "health_ratio": 0.8},
            "after": {"energy_ratio": 0.7, "hydration_ratio": 0.35, "health_ratio": 0.8},
        },
        public_observation_bucket={
            "policy": "post_contact_public_observation_bucket_v1",
            "branch_tick": 3,
            "branch_tick_bin": "early",
            "energy_bin": "medium",
            "hydration_bin": "low",
            "health_bin": "high",
            "water_distance": 0.4,
            "water_distance_bin": "mid",
            "drink_available": False,
            "eat_available": True,
            "movement_available": True,
        },
        alive_agents_at_branch=1,
        births_at_branch=0,
        deaths_at_branch=0,
        trajectory_record_count_at_branch=1,
        branch_state_digest=digest or f"digest-{branch_id}",
        world=SimpleNamespace(),
    )


def _test_branch_run(
    *,
    seed: int,
    branch_id: str,
    script: str,
    alive: int,
) -> dict[str, object]:
    return {
        "branch_id": branch_id,
        "seed": seed,
        "fixture": "carrion_only",
        "ticks": 8,
        "branch_tick": 3,
        "branch_index": 0,
        "base_script": "current_policy:g-target",
        "continuation_script": script,
        "branch_state_digest": f"digest-{branch_id}",
        "ticks_executed": 8,
        "alive_agents": alive,
        "births": 1 if alive else 0,
        "deaths": 0 if alive else 1,
        "trajectory_record_count": 2,
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": 0,
        "requested_action_counts": {"eat": 1},
        "resolved_action_counts": {"eat": 1},
        "action_source_counts": {f"counterfactual_script:{script}": 1},
        "policy_id_counts": {f"mind_v3_counterfactual_{script}": 1},
        "dominant_requested_action": "eat",
        "dominant_requested_action_count": 1,
        "dominant_requested_action_share": 1.0,
        "outcome_metrics": {},
        "replay_verification": None,
    }


if __name__ == "__main__":
    unittest.main()
