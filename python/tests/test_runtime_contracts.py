from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.contracts import (
    FULL_ONLY_SUMMARY_FIELDS,
    REPLAY_TOP_LEVEL_KEYS,
    SHARED_SUMMARY_FIELDS,
    VIEWER_AGENT_ENCODING,
    VIEWER_MAP_KEYS,
)
from evolution_sim.env.runtime.biotic import diffuse_biotic_field, diffuse_sparse_biotic_field
from evolution_sim.env.runtime.action_space import build_action_mask
from evolution_sim.env.runtime.observations import (
    OBSERVATION_SCHEMA_VERSION,
    build_observation,
    observation_digest,
)
from evolution_sim.env.runtime.trajectory import (
    REWARD_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
)
from evolution_sim.env.taxonomy import apply_replay_taxonomy
from evolution_sim.io import write_json_replay

GOLDEN_SPECIATION_SEED = 3


class RuntimeContractTests(unittest.TestCase):
    def test_full_replay_contract_orders_are_frozen(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=20)).run()

        self.assertEqual(tuple(result.viewer["map"]), VIEWER_MAP_KEYS)
        self.assertEqual(tuple(result.viewer["agent_encoding"]), VIEWER_AGENT_ENCODING)

        legend_keys = (
            "terrain_legend",
            "hydrology_primary_legend",
            "hydrology_support_bits",
            "refuge_legend",
            "hazard_legend",
            "trophic_role_legend",
            "meat_mode_legend",
            "ecology_legend",
        )
        for legend_key in legend_keys:
            legend = result.viewer["map"][legend_key]
            self.assertTrue(all(isinstance(key, str) for key in legend), msg=legend_key)
            self.assertEqual(
                list(legend),
                [str(code) for code in sorted(int(code) for code in legend)],
                msg=legend_key,
            )

        with TemporaryDirectory() as tmpdir:
            replay_path = write_json_replay(result, Path(tmpdir) / "contract-order.json")
            payload = json.loads(replay_path.read_text(encoding="utf-8"))
        self.assertEqual(tuple(payload), REPLAY_TOP_LEVEL_KEYS)

    def test_summary_only_matches_shared_summary_and_omits_replay_surfaces(self) -> None:
        config = WorldConfig(seed=7, max_ticks=40)

        full = SimulationWorld(config).run()
        summary_only = SimulationWorld(config).run(mode=RunMode.SUMMARY_ONLY)

        self.assertEqual(summary_only.mode, RunMode.SUMMARY_ONLY)
        self.assertIsNone(summary_only.events)
        self.assertIsNone(summary_only.viewer)
        self.assertEqual(tuple(summary_only.summary), SHARED_SUMMARY_FIELDS)
        for field in SHARED_SUMMARY_FIELDS:
            self.assertEqual(summary_only.summary[field], full.summary[field], msg=field)
        for field in FULL_ONLY_SUMMARY_FIELDS:
            self.assertNotIn(field, summary_only.summary)

    def test_summary_only_is_byte_deterministic_under_repeated_runs(self) -> None:
        for seed, ticks in ((7, 40), (GOLDEN_SPECIATION_SEED, 80)):
            first = SimulationWorld(WorldConfig(seed=seed, max_ticks=ticks)).run(
                mode=RunMode.SUMMARY_ONLY
            )
            second = SimulationWorld(WorldConfig(seed=seed, max_ticks=ticks)).run(
                mode=RunMode.SUMMARY_ONLY
            )
            first_bytes = json.dumps(first.summary, separators=(",", ":")).encode("utf-8")
            second_bytes = json.dumps(second.summary, separators=(",", ":")).encode("utf-8")
            self.assertEqual(first_bytes, second_bytes, msg=f"seed={seed} ticks={ticks}")

    def test_summary_only_release_span_is_byte_deterministic_under_repeated_runs(self) -> None:
        config = WorldConfig(
            seed=GOLDEN_SPECIATION_SEED,
            max_ticks=800,
            width=16,
            height=12,
            initial_agents=8,
            max_agents=80,
        )

        first = SimulationWorld(config).run(mode=RunMode.SUMMARY_ONLY)
        second = SimulationWorld(config).run(mode=RunMode.SUMMARY_ONLY)

        first_bytes = json.dumps(first.summary, separators=(",", ":")).encode("utf-8")
        second_bytes = json.dumps(second.summary, separators=(",", ":")).encode("utf-8")
        self.assertEqual(first_bytes, second_bytes)

    def test_summary_only_never_invokes_full_replay_paths(self) -> None:
        with patch(
            "evolution_sim.env.runtime.collectors.apply_replay_taxonomy",
            side_effect=AssertionError("taxonomy should not run in summary-only mode"),
        ), patch.object(
            SimulationWorld,
            "_capture_frame",
            side_effect=AssertionError("summary-only should not capture frames"),
        ), patch.object(
            SimulationWorld,
            "_build_viewer_payload",
            side_effect=AssertionError("summary-only should not build viewer payloads"),
        ):
            result = SimulationWorld(WorldConfig(seed=7, max_ticks=20)).run(
                mode=RunMode.SUMMARY_ONLY
            )
        self.assertIsNone(result.viewer)
        self.assertIsNone(result.events)

    def test_summary_only_does_not_retain_replay_bookkeeping(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=20))

        result = world.run(mode=RunMode.SUMMARY_ONLY)

        self.assertIsNone(result.events)
        self.assertIsNone(result.viewer)
        self.assertEqual(world.events, [])
        self.assertEqual(world.viewer_frames, [])
        self.assertTrue(world.record_events)
        self.assertTrue(world.record_tick_details)

    def test_action_mask_rejects_impossible_eat_as_stay(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        for y, row in enumerate(world.grid):
            for x, tile in enumerate(row):
                if tile.terrain != "water" and world._water_access_reason(x, y) == "none":
                    world.grid[agent.y][agent.x].occupant_id = None
                    agent.x = x
                    agent.y = y
                    tile.occupant_id = agent.agent_id
                    break
            else:
                continue
            break

        tile = world.grid[agent.y][agent.x]
        tile.food = 0.0
        tile.fresh_kill_deposits.clear()
        tile.carcass_deposits.clear()
        agent.energy = agent.genome.max_energy
        agent.hydration = agent.genome.max_hydration
        energy_before = agent.energy

        mask = build_action_mask(world, agent)
        moved = world._resolve_action(agent, "eat")

        self.assertFalse(mask["eat"])
        self.assertFalse(moved)
        self.assertEqual(agent.energy, energy_before)
        self.assertFalse(any(event.type.value == "agent_ate" for event in world.events))

    def test_observation_contract_is_serializable_and_unprivileged(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]

        observation = build_observation(world, agent)
        digest = observation_digest(observation)

        self.assertEqual(observation["schema_version"], OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(set(observation), {"schema_version", "agent_id", "self", "local_patch", "action_mask"})
        self.assertEqual(len(observation["local_patch"]), 25)
        self.assertNotIn("world", observation)
        self.assertNotIn("grid", observation)
        self.assertNotIn("agents", observation)
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)
        json.dumps(observation)

    def test_full_replay_records_mind_trajectory_contract(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()

        trajectory = result.viewer["trajectory"]
        records = trajectory["records"]
        first_record = records[0]

        self.assertEqual(result.summary["mind_contracts"]["schema_version"], TRAJECTORY_SCHEMA_VERSION)
        self.assertEqual(trajectory["schema_version"], TRAJECTORY_SCHEMA_VERSION)
        self.assertEqual(
            trajectory["observation_contract"]["schema_version"],
            OBSERVATION_SCHEMA_VERSION,
        )
        self.assertFalse(trajectory["observation_contract"]["privileged_world_state"])
        self.assertGreater(trajectory["record_count"], 0)
        self.assertEqual(trajectory["record_count"], len(records))
        self.assertEqual(first_record["observation_schema"], OBSERVATION_SCHEMA_VERSION)
        self.assertIn(first_record["requested_action"], first_record["action_mask"])
        self.assertIn("resource_gain", first_record["outcome"])
        self.assertEqual(first_record["reward"]["schema_version"], REWARD_SCHEMA_VERSION)
        self.assertIn("invalid_action_penalty", first_record["reward"]["components"])
        self.assertEqual(result.summary["mind_contracts"]["record_count"], len(records))

    def test_summary_only_does_not_retain_trajectory_bookkeeping(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=4))

        result = world.run(mode=RunMode.SUMMARY_ONLY)

        self.assertIsNone(result.viewer)
        self.assertEqual(world.trajectory_records, [])
        self.assertEqual(world.tick_trajectory_records, [])
        self.assertTrue(world.record_trajectory)

    def test_write_json_replay_rejects_summary_only_results(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=20)).run(mode=RunMode.SUMMARY_ONLY)
        with TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "full replay"):
                write_json_replay(result, Path(tmpdir) / "summary-only.json")

    def test_apply_replay_taxonomy_is_idempotent(self) -> None:
        config = WorldConfig(seed=7, max_ticks=30)
        result = SimulationWorld(config).run()

        updated_summary, updated_viewer = apply_replay_taxonomy(
            config=config,
            summary=copy.deepcopy(result.summary),
            events=copy.deepcopy(result.events),
            viewer=copy.deepcopy(result.viewer),
        )

        self.assertEqual(updated_summary, result.summary)
        self.assertEqual(updated_viewer, result.viewer)

    def test_reset_derived_caches_forces_derived_rebuilds(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=5))

        first_climate = world._climate_state()
        second_climate = world._climate_state()
        first_habitat = world._habitat_state_grid()
        second_habitat = world._habitat_state_grid()
        first_biotic = world._current_biotic_state()
        second_biotic = world._current_biotic_state()

        self.assertIs(first_climate, second_climate)
        self.assertIs(first_habitat[0], second_habitat[0])
        self.assertIs(first_habitat[1], second_habitat[1])
        self.assertIs(first_biotic, second_biotic)

        world.reset_derived_caches()

        self.assertIsNot(world._climate_state(), first_climate)
        rebuilt_habitat = world._habitat_state_grid()
        self.assertIsNot(rebuilt_habitat[0], first_habitat[0])
        self.assertIsNot(rebuilt_habitat[1], first_habitat[1])
        self.assertIsNot(world._current_biotic_state(), first_biotic)

    def test_cached_biotic_diffusion_targets_match_naive_diffusion(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        radius = max(1, world.config.biotic_fields.diffusion_radius)
        sources = [
            [0.0 for _ in range(world.config.width)]
            for _ in range(world.config.height)
        ]
        for x, y, value in ((3, 4, 1.25), (20, 12, 0.5), (40, 24, 2.0)):
            if world.grid[y][x].terrain != "water":
                sources[y][x] = value

        actual = diffuse_biotic_field(world, sources)
        sparse_sources = {
            y * world.config.width + x: source
            for y, row in enumerate(sources)
            for x, source in enumerate(row)
            if source > 1e-9
        }
        sparse_actual = diffuse_sparse_biotic_field(world, sparse_sources)
        expected = [
            [0.0 for _ in range(world.config.width)]
            for _ in range(world.config.height)
        ]
        for sy, row in enumerate(sources):
            for sx, source in enumerate(row):
                if source <= 1e-9 or world.grid[sy][sx].terrain == "water":
                    continue
                for dy in range(-radius, radius + 1):
                    span = radius - abs(dy)
                    for dx in range(-span, span + 1):
                        distance = abs(dx) + abs(dy)
                        if distance > radius:
                            continue
                        x = sx + dx
                        y = sy + dy
                        if (
                            x < 0
                            or y < 0
                            or x >= world.config.width
                            or y >= world.config.height
                            or world.grid[y][x].terrain == "water"
                        ):
                            continue
                        expected[y][x] += source / (distance + 1.0)

        self.assertEqual(actual, expected)
        self.assertEqual(sparse_actual, expected)


if __name__ == "__main__":
    unittest.main()
