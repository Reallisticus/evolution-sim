from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_carrion_autopsy
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime import trajectory as runtime_trajectory
from evolution_sim.env.runtime.observations import (
    LOCAL_PATCH_RADIUS,
    OBSERVATION_SCHEMA_VERSION,
    encode_observation_input,
)
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind.carrion_autopsy import (
    MIND_V3_CARRION_AUTOPSY_SCHEMA_VERSION,
    build_carrion_autopsy_report,
)
from evolution_sim.mind.dataset import TrajectoryJsonlDataset


class MindV3CarrionAutopsyTests(unittest.TestCase):
    def test_autopsy_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-autopsy"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_autopsy"
            ),
        )

    def test_autopsy_identifies_post_contact_movement_energy_death(self) -> None:
        records = (
            _trajectory_record(
                tick=0,
                action="eat",
                before_energy=0.4,
                after_energy=0.7,
                after_hydration=0.9,
                after_health=1.0,
                food_source="carcass",
                gained_energy=0.31,
            ),
            _trajectory_record(
                tick=1,
                action="move_east",
                before_energy=0.7,
                after_energy=0.08,
                after_hydration=0.6,
                after_health=0.9,
            ),
            _trajectory_record(
                tick=2,
                action="move_east",
                before_energy=0.08,
                after_energy=-0.02,
                after_hydration=0.55,
                after_health=0.88,
                died=True,
                death_cause="energy_depletion",
            ),
        )
        dataset = _dataset(records)

        report = build_carrion_autopsy_report(
            [dataset],
            post_contact_window_ticks=120,
            sequence_record_limit=4,
            max_examples=2,
        )

        aggregate = report["aggregate"]
        self.assertEqual(
            report["schema_version"],
            MIND_V3_CARRION_AUTOPSY_SCHEMA_VERSION,
        )
        self.assertEqual(aggregate["contact_episode_count"], 1)
        self.assertEqual(aggregate["death_after_contact_count"], 1)
        self.assertEqual(
            aggregate["dominant_death_path"]["path"],
            "movement_energy_depletion_after_carrion_contact",
        )
        self.assertEqual(
            aggregate["terminal_bottleneck_counts"],
            {"energy": 1},
        )
        example = report["examples"][0]
        self.assertEqual(example["death_tick"], 2)
        self.assertEqual(example["animal_resource_event_count"], 1)
        self.assertEqual(example["movement_action_count"], 2)
        self.assertEqual(len(example["sequence_excerpt"]), 3)
        self.assertIn(
            "continued spending movement energy",
            example["dominant_failure_hypothesis"],
        )

    def test_fixture_trace_computes_contact_drink_window_and_death(self) -> None:
        records = (
            _trajectory_record(
                tick=0,
                action="eat",
                before_energy=0.3,
                after_energy=0.7,
                before_hydration=0.4,
                after_hydration=0.4,
                after_health=0.9,
                food_source="carcass",
                gained_energy=0.4,
                observation_input=_observation_input(
                    carrion_distance=3.0,
                    carrion_strength=0.8,
                    water_distance=2.0,
                    water_strength=0.6,
                ),
                policy_decision_diagnostics={
                    "rollout_context_post_carrion_context": True,
                    "rollout_context_selected_score_delta": 0.25,
                },
                action_valid=True,
                resolution_action_valid=True,
                action_mask=_action_mask(eat=True, drink=False, move=True),
                resolution_action_mask=_action_mask(eat=True, drink=False, move=True),
            ),
            _trajectory_record(
                tick=1,
                action="drink",
                before_energy=0.7,
                after_energy=0.65,
                before_hydration=0.4,
                after_hydration=0.8,
                after_health=0.9,
                drank=True,
                action_valid=True,
                resolution_action_valid=True,
                action_mask=_action_mask(eat=False, drink=True, move=True),
                resolution_action_mask=_action_mask(eat=False, drink=True, move=True),
            ),
            _trajectory_record(
                tick=2,
                action="stay",
                before_energy=0.65,
                after_energy=0.6,
                before_hydration=0.8,
                after_hydration=0.7,
                after_health=0.9,
                action_valid=True,
                resolution_action_valid=True,
                action_mask=_action_mask(eat=False, drink=False, move=True),
                resolution_action_mask=_action_mask(eat=False, drink=False, move=True),
            ),
            _trajectory_record(
                tick=0,
                agent_id=2,
                action="eat",
                before_energy=0.2,
                after_energy=0.5,
                before_hydration=0.3,
                after_hydration=0.3,
                after_health=0.8,
                food_source="fresh_kill",
                gained_energy=0.3,
                action_valid=True,
                resolution_action_valid=True,
            ),
            _trajectory_record(
                tick=2,
                agent_id=2,
                action="move_west",
                resolved_action="stay",
                before_energy=0.5,
                after_energy=0.3,
                before_hydration=0.3,
                after_hydration=0.0,
                after_health=0.7,
                died=True,
                death_cause="hydration_depletion",
                action_valid=True,
                resolution_action_valid=False,
                invalid_reason="not_in_resolution_action_mask",
            ),
        )

        report = build_carrion_autopsy_report(
            [_dataset(records)],
            post_contact_window_ticks=2,
            sequence_record_limit=4,
            max_examples=2,
        )

        trace = report["fixture_trace"]
        aggregate = trace["aggregate"]
        self.assertEqual(aggregate["contact_agent_count"], 2)
        self.assertEqual(aggregate["drink_after_carrion_rate"], 0.5)
        self.assertEqual(aggregate["survival_after_carrion_rate"], 0.5)
        self.assertEqual(aggregate["mean_hydration_delta_after_carrion"], 0.0)
        self.assertEqual(aggregate["mean_energy_delta_after_carrion"], -0.15)
        self.assertEqual(aggregate["mean_health_delta_after_carrion"], -0.05)
        self.assertEqual(aggregate["death_ticks_after_carrion"], [2])
        self.assertEqual(
            aggregate["death_cause_counts"],
            {"hydration_depletion": 1},
        )
        self.assertEqual(
            aggregate["unsupported_resolved_breakdown"]["by_requested_action"],
            {"move_west": 1},
        )
        self.assertEqual(
            aggregate["unsupported_resolved_breakdown"]["by_seed"]["1"][
                "by_resolved_action"
            ],
            {"stay": 1},
        )

    def test_fixture_trace_tolerates_missing_policy_diagnostics(self) -> None:
        report = build_carrion_autopsy_report(
            [
                _dataset(
                    (
                        _trajectory_record(
                            tick=0,
                            action="eat",
                            before_energy=0.4,
                            after_energy=0.6,
                            after_hydration=0.5,
                            after_health=0.9,
                            food_source="carcass",
                            gained_energy=0.2,
                        ),
                    )
                )
            ],
            post_contact_window_ticks=1,
        )

        aggregate = report["fixture_trace"]["aggregate"]
        self.assertEqual(
            aggregate["rollout_context_diagnostics"]["missing_count"],
            1,
        )
        self.assertEqual(
            aggregate["missing_field_counts"]["policy_decision_diagnostics"],
            1,
        )

    def test_fixture_trace_unsupported_breakdown_accumulates_same_seed_agents(
        self,
    ) -> None:
        report = build_carrion_autopsy_report(
            [
                _dataset(
                    (
                        _trajectory_record(
                            tick=0,
                            agent_id=1,
                            action="move_east",
                            resolved_action="stay",
                            before_energy=0.5,
                            after_energy=0.4,
                            after_hydration=0.5,
                            after_health=0.9,
                            action_valid=True,
                            resolution_action_valid=False,
                            invalid_reason="not_in_resolution_action_mask",
                        ),
                        _trajectory_record(
                            tick=0,
                            agent_id=2,
                            action="move_west",
                            resolved_action="stay",
                            before_energy=0.5,
                            after_energy=0.4,
                            after_hydration=0.5,
                            after_health=0.9,
                            action_valid=True,
                            resolution_action_valid=False,
                            invalid_reason="not_in_resolution_action_mask",
                        ),
                    )
                )
            ],
            post_contact_window_ticks=1,
        )

        breakdown = report["fixture_trace"]["aggregate"][
            "unsupported_resolved_breakdown"
        ]
        self.assertEqual(breakdown["count"], 2)
        self.assertEqual(breakdown["by_seed"]["1"]["count"], 2)
        self.assertEqual(
            breakdown["by_seed"]["1"]["by_requested_action"],
            {"move_east": 1, "move_west": 1},
        )

    def test_fixture_trace_decodes_navigation_targets_from_observation_input(
        self,
    ) -> None:
        report = build_carrion_autopsy_report(
            [
                _dataset(
                    (
                        _trajectory_record(
                            tick=0,
                            action="eat",
                            before_energy=0.4,
                            after_energy=0.6,
                            after_hydration=0.5,
                            after_health=0.9,
                            food_source="carcass",
                            gained_energy=0.2,
                            observation_input=_observation_input(
                                carrion_distance=4.0,
                                carrion_strength=3.0,
                                water_distance=1.0,
                                water_strength=1.0,
                            ),
                        ),
                    )
                )
            ],
            post_contact_window_ticks=1,
        )

        agent = next(iter(report["fixture_trace"]["by_agent"].values()))
        navigation_at_contact = agent["first_animal_resource_contact"][
            "navigation_at_contact"
        ]
        self.assertEqual(navigation_at_contact["carrion"]["distance"], 4.0)
        self.assertAlmostEqual(
            navigation_at_contact["carrion"]["strength"],
            0.75,
            places=3,
        )
        aggregate_navigation = report["fixture_trace"]["aggregate"][
            "navigation_target_observations"
        ]
        self.assertEqual(aggregate_navigation["water"]["strength_positive_count"], 1)
        self.assertEqual(aggregate_navigation["water"]["mean_distance"], 1.0)

    def test_autopsy_cli_writes_empty_contact_report_for_valid_trajectory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            output_path = Path(tmpdir) / "autopsy.json"
            writer = JsonlTrajectoryWriter(
                trajectory_path,
                source_seeds=[7],
                split_id="autopsy-smoke",
            )
            SimulationWorld(WorldConfig(seed=7, max_ticks=1)).run(
                mode=RunMode.SUMMARY_ONLY,
                trajectory_sink=writer,
            )

            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_autopsy",
                    "--trajectory",
                    str(trajectory_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_autopsy.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_AUTOPSY_SCHEMA_VERSION,
        )
        self.assertEqual(payload["aggregate"]["contact_episode_count"], 0)
        self.assertEqual(payload["examples"], [])

    def test_autopsy_cli_accepts_nested_rollout_context_diagnostics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            output_path = Path(tmpdir) / "autopsy.json"
            world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
            result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
            record = dict(world.trajectory_records[0])
            record["policy_decision_diagnostics"] = {
                "rollout_context_selected_score_delta": 0.0,
                "rollout_context_score_delta_by_action": {
                    "eat": 0.0,
                    "drink": 0.1,
                },
            }
            writer = JsonlTrajectoryWriter(
                trajectory_path,
                source_seeds=[7],
                split_id="autopsy-nested-diagnostics",
                include_policy_decision_diagnostics=True,
            )
            writer.begin(
                run_id=world.run_id,
                config=world.config.to_dict(),
                contract=runtime_trajectory.trajectory_contract(
                    world.config.signals
                ),
            )
            writer.write_record(record)
            writer.finish(summary=dict(result.summary))

            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_autopsy",
                    "--trajectory",
                    str(trajectory_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_autopsy.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["source"]["record_count"], 1)
        self.assertEqual(
            payload["fixture_trace"]["aggregate"][
                "rollout_context_diagnostics"
            ]["missing_count"],
            0,
        )


def _dataset(records: tuple[dict[str, object], ...]) -> TrajectoryJsonlDataset:
    provenance = {
        "source_seeds": [1],
        "config_digest": "synthetic-config",
        "contract_digest": "synthetic-contract",
        "split_id": "synthetic-carrion",
        "trajectory_paths": ["synthetic-carrion.jsonl.gz"],
        "record_count": len(records),
    }
    return TrajectoryJsonlDataset(
        path=Path("synthetic-carrion.jsonl.gz"),
        header={"provenance": provenance},
        records=records,
        footer={"provenance": provenance},
    )


def _trajectory_record(
    *,
    tick: int,
    agent_id: int = 1,
    action: str,
    resolved_action: str | None = None,
    before_energy: float,
    after_energy: float,
    before_hydration: float | None = None,
    after_hydration: float,
    after_health: float,
    food_source: str | None = None,
    gained_energy: float = 0.0,
    drank: bool = False,
    died: bool = False,
    death_cause: str | None = None,
    action_valid: bool | None = None,
    resolution_action_valid: bool | None = None,
    invalid_reason: str | None = None,
    action_mask: dict[str, bool] | None = None,
    resolution_action_mask: dict[str, bool] | None = None,
    observation_input: dict[str, object] | None = None,
    policy_decision_diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "tick": tick,
        "agent_id": agent_id,
        "requested_action": action,
        "resolved_action": resolved_action or action,
        "before": {
            "alive": True,
            "energy_ratio": before_energy,
            "hydration_ratio": (
                after_hydration if before_hydration is None else before_hydration
            ),
            "health_ratio": after_health,
            "x": tick,
            "y": 0,
        },
        "after": {
            "alive": not died,
            "energy_ratio": after_energy,
            "hydration_ratio": after_hydration,
            "health_ratio": after_health,
            "x": tick + (1 if action.startswith("move_") else 0),
            "y": 0,
        },
        "outcome": {
            "died": died,
            "reproduced": False,
            "feeding": {
                "ate": food_source is not None,
                "food_source": food_source,
                "gained_energy": gained_energy,
            },
            "drinking": {"drank": drank},
            "movement": {"moved": action.startswith("move_")},
            "passive": {
                "death_cause": death_cause,
                "died_after_action": died,
            },
        },
    }
    if invalid_reason is not None:
        payload["outcome"]["invalid_reason"] = invalid_reason  # type: ignore[index]
    if action_valid is not None:
        payload["action_valid"] = action_valid
    if resolution_action_valid is not None:
        payload["resolution_action_valid"] = resolution_action_valid
    if action_mask is not None:
        payload["action_mask"] = action_mask
    if resolution_action_mask is not None:
        payload["resolution_action_mask"] = resolution_action_mask
    if observation_input is not None:
        payload["observation_input"] = observation_input
    if policy_decision_diagnostics is not None:
        payload["policy_decision_diagnostics"] = policy_decision_diagnostics
    return payload


def _action_mask(*, eat: bool, drink: bool, move: bool) -> dict[str, bool]:
    return {
        "stay": True,
        "eat": eat,
        "drink": drink,
        "move_north": move,
        "move_south": move,
        "move_east": move,
        "move_west": move,
    }


def _observation_input(
    *,
    carrion_distance: float,
    carrion_strength: float,
    water_distance: float,
    water_strength: float,
) -> dict[str, object]:
    patch = []
    for dy in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1):
        for dx in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1):
            patch.append(
                {
                    "dx": dx,
                    "dy": dy,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "same_lineage": False,
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_type": "none",
                    "hazard_level": 0.0,
                    "ecology_state": "stable",
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                    "reproductive_signal": 0.0,
                    "communication_signal": 0.0,
                }
            )
    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "metadata": {"agent_id": 1},
        "self": {
            "energy_ratio": 0.5,
            "hydration_ratio": 0.5,
            "health_ratio": 0.9,
            "injury_load": 0.0,
            "age_norm": 0.1,
            "reproduction_ready": False,
            "matched_diet_ratio": 1.0,
            "trophic_role": "carnivore",
            "meat_mode": "scavenger",
            "season": "wet",
            "water_access_reason": "none",
            "hydrology_support_code": 1,
            "refuge_score": 0.0,
            "hazard_type": "none",
            "hazard_level": 0.0,
            "tile_vegetation": 0.0,
            "tile_recovery_debt": 0.0,
            "reproductive_stage": "stage0_asexual",
            "reproductive_expression": "asexual",
            "sexual_reproduction_unlocked": False,
            "reproductive_signal": 0.0,
            "communication_signal": 0.0,
            "mind_inheritance_available": False,
        },
        "local_patch": patch,
        "navigation": {
            "water": {
                "dx": 0,
                "dy": -int(water_distance),
                "distance": water_distance,
                "strength": water_strength,
            },
            "plant": {"dx": 0, "dy": 0, "distance": 0.0, "strength": 0.0},
            "carrion": {
                "dx": int(carrion_distance),
                "dy": 0,
                "distance": carrion_distance,
                "strength": carrion_strength,
            },
            "prey": {"dx": 0, "dy": 0, "distance": 0.0, "strength": 0.0},
        },
        "action_mask": _action_mask(eat=True, drink=True, move=True),
    }
    return encode_observation_input(observation)
