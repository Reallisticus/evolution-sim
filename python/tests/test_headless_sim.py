from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.env.world import Agent
from evolution_sim.genome import Genome
from evolution_sim.genome.species import genome_vector
from evolution_sim.io import write_json_replay


class HeadlessSimulationTests(unittest.TestCase):
    def test_same_seed_produces_identical_summary_events_and_viewer_frames(self) -> None:
        config = WorldConfig(seed=21, max_ticks=180)

        first = SimulationWorld(config).run()
        second = SimulationWorld(config).run()

        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.events, second.events)
        self.assertEqual(first.viewer, second.viewer)

    def test_long_run_has_births_deaths_and_turnover_beyond_alive_cap(self) -> None:
        config = WorldConfig(seed=7, max_ticks=500)
        result = SimulationWorld(config).run()

        self.assertGreater(result.summary["births"], 0)
        self.assertGreater(result.summary["deaths"], 0)
        self.assertGreater(result.summary["total_agents_seen"], config.max_agents)
        self.assertGreater(result.summary["peak_alive_agents"], config.initial_agents)
        self.assertIsNotNone(result.summary["last_birth_tick"])
        self.assertEqual(len(result.viewer["frames"]), result.summary["ticks_executed"])
        self.assertGreater(result.summary["species_created"], 1)
        self.assertGreater(result.summary["alive_species_count"], 1)
        self.assertEqual(
            len(result.viewer["frames"][-1]["species_counts"]),
            result.summary["alive_species_count"],
        )
        self.assertEqual(result.viewer["agent_encoding"][-1], "species_id")
        self.assertIn("field_state", result.viewer["frames"][-1])
        self.assertIn("field_stats", result.summary)
        self.assertIn("terrain_counts", result.summary)
        self.assertIn("analytics", result.viewer)
        self.assertEqual(
            len(result.viewer["analytics"]["population"]["alive_agents"]),
            result.summary["ticks_executed"],
        )
        self.assertEqual(
            len(result.viewer["analytics"]["ticks"]),
            result.summary["ticks_executed"],
        )
        self.assertIn("species_metrics", result.viewer["frames"][-1])
        self.assertIn("ecotype_metrics", result.viewer["frames"][-1])
        self.assertIn("trait_means", result.viewer["frames"][-1])
        self.assertIn("disturbance_type", result.viewer["frames"][-1]["field_state"])
        self.assertIn("disturbance_at_end", result.summary)
        self.assertIn("hydrology_primary_counts", result.viewer["frames"][-1])
        self.assertIn("hydrology_primary_codes", result.viewer["frames"][-1])
        self.assertIn("hydrology_primary_stats", result.viewer["frames"][-1])
        self.assertIn("hydrology_support_counts", result.viewer["frames"][-1])
        self.assertIn("hydrology_support_codes", result.viewer["frames"][-1])
        self.assertIn("refuge_counts", result.viewer["frames"][-1])
        self.assertIn("refuge_codes", result.viewer["frames"][-1])
        self.assertIn("refuge_score_codes", result.viewer["frames"][-1])
        self.assertIn("refuge_stats", result.viewer["frames"][-1])
        self.assertIn("hazard_counts", result.viewer["frames"][-1])
        self.assertIn("hazard_type_codes", result.viewer["frames"][-1])
        self.assertIn("hazard_level_codes", result.viewer["frames"][-1])
        self.assertIn("hazard_stats", result.viewer["frames"][-1])
        self.assertIn("carcass_energy_codes", result.viewer["frames"][-1])
        self.assertIn("carcass_freshness_codes", result.viewer["frames"][-1])
        self.assertIn("carcass_stats", result.viewer["frames"][-1])
        self.assertIn("combat_stats", result.viewer["frames"][-1])
        self.assertIn("diet_stats", result.viewer["frames"][-1])
        self.assertIn("trophic_role_codes", result.viewer["frames"][-1])
        self.assertIn("meat_mode_codes", result.viewer["frames"][-1])
        self.assertIn("habitat_state_counts", result.viewer["frames"][-1])
        self.assertIn("habitat_state_codes", result.viewer["frames"][-1])
        self.assertIn("ecology_state_counts", result.viewer["frames"][-1])
        self.assertIn("ecology_state_codes", result.viewer["frames"][-1])
        self.assertIn("ecology_stats", result.viewer["frames"][-1])
        self.assertIn("habitat_state_counts_at_end", result.summary)
        self.assertIn("hydrology_primary_counts_at_end", result.summary)
        self.assertIn("hydrology_support_counts_at_end", result.summary)
        self.assertIn("hydrology_primary_stats_at_end", result.summary)
        self.assertIn("refuge_counts_at_end", result.summary)
        self.assertIn("refuge_stats_at_end", result.summary)
        self.assertIn("hazard_counts_at_end", result.summary)
        self.assertIn("hazard_stats_at_end", result.summary)
        self.assertIn("carcass_stats_at_end", result.summary)
        self.assertIn("combat_end", result.summary)
        self.assertIn("carcass_end", result.summary)
        self.assertIn("diet_end", result.summary)
        self.assertIn("meat_mode_counts_at_end", result.summary)
        self.assertIn("ecology_state_counts_at_end", result.summary)
        self.assertIn("ecology_stats_at_end", result.summary)
        self.assertEqual(
            result.viewer["analytics"]["hydrology_primary"]["hard_access_tiles"][-1],
            result.viewer["frames"][-1]["hydrology_primary_stats"]["hard_access_tiles"],
        )
        self.assertEqual(
            result.viewer["analytics"]["refuge"]["avg_refuge_score_forest_tiles"][-1],
            result.viewer["frames"][-1]["refuge_stats"]["avg_refuge_score_forest_tiles"],
        )
        sample_metrics = next(iter(result.viewer["frames"][-1]["species_metrics"].values()))
        self.assertEqual(
            set(sample_metrics["terrain_occupancy"]),
            {"plain", "forest", "wetland", "rocky", "water_access"},
        )
        self.assertEqual(
            set(sample_metrics["hydrology_exposure_counts"]),
            {
                "primary_none",
                "primary_adjacent_water",
                "primary_wetland",
                "primary_flooded",
                "shoreline_support",
                "wetland_support",
                "flooded_support",
                "refuge_exposed",
            },
        )
        self.assertEqual(
            set(sample_metrics["habitat_occupancy"]),
            {"stable", "bloom", "flooded", "parched"},
        )
        self.assertEqual(
            set(sample_metrics["ecology_occupancy"]),
            {"stable", "lush", "recovering", "depleted"},
        )
        self.assertIn("avg_tile_vegetation", sample_metrics)
        self.assertIn("avg_recovery_debt", sample_metrics)
        self.assertIn("avg_health_ratio", sample_metrics)
        self.assertIn("avg_refuge_score_occupied_tiles", sample_metrics)
        self.assertIn("refuge_exposure_rate", sample_metrics)
        self.assertIn("injury_rate", sample_metrics)
        self.assertIn("hazard_exposure_rate", sample_metrics)
        self.assertIn("attack_attempts", sample_metrics)
        self.assertIn("successful_attacks", sample_metrics)
        self.assertIn("kills", sample_metrics)
        self.assertIn("damage_dealt", sample_metrics)
        self.assertIn("damage_taken", sample_metrics)
        self.assertIn("hazard_damage_taken", sample_metrics)
        self.assertIn("plant_consumption", sample_metrics)
        self.assertIn("plant_energy_consumed", sample_metrics)
        self.assertIn("carcass_consumption", sample_metrics)
        self.assertIn("carcass_energy_consumed", sample_metrics)
        self.assertIn("realized_plant_share", sample_metrics)
        self.assertIn("realized_carcass_share", sample_metrics)
        self.assertIn("hazard_occupancy", sample_metrics)
        self.assertIn("trophic_role_occupancy", sample_metrics)
        self.assertIn("meat_mode_occupancy", sample_metrics)
        self.assertIn("habitat", result.viewer["analytics"])
        self.assertIn("hydrology_primary", result.viewer["analytics"])
        self.assertIn("hydrology_support", result.viewer["analytics"])
        self.assertIn("refuge", result.viewer["analytics"])
        self.assertIn("hazards", result.viewer["analytics"])
        self.assertIn("carcasses", result.viewer["analytics"])
        self.assertIn("diet", result.viewer["analytics"])
        self.assertIn("combat", result.viewer["analytics"])
        self.assertIn("ecology", result.viewer["analytics"])
        self.assertIn("ecotype_count", result.viewer["analytics"]["population"])
        self.assertIn("taxonomy", result.viewer)
        self.assertIn("ecotype_catalog", result.viewer)
        self.assertTrue(
            all(
                payload["genome"]["reproduction_threshold"] <= 0.98
                for payload in result.viewer["agent_catalog"].values()
            )
        )
        self.assertEqual(
            result.viewer["agent_encoding"][-6:],
            [
                "water_access_reason",
                "soft_refuge_reason",
                "hydrology_support_code",
                "refuge_score",
                "ecotype_id",
                "species_id",
            ],
        )
        self.assertIn("health", result.viewer["agent_encoding"])
        self.assertIn("trophic_role", result.viewer["agent_encoding"])
        self.assertIn("meat_mode", result.viewer["agent_encoding"])
        self.assertIn("last_damage_source", result.viewer["agent_encoding"])
        sample_agent = next(iter(result.viewer["agent_catalog"].values()))
        self.assertIn("birth_tick", sample_agent)
        self.assertIn("death_tick", sample_agent)

    def test_environment_fields_are_present_and_spatially_varied(self) -> None:
        result = SimulationWorld(WorldConfig(seed=13, max_ticks=90)).run()

        environment_fields = result.viewer["map"]["environment_fields"]
        fertility = environment_fields["fertility"]
        moisture = environment_fields["moisture"]
        heat = environment_fields["heat"]

        self.assertEqual(len(fertility), 32)
        self.assertEqual(len(fertility[0]), 48)
        self.assertEqual(len(moisture), 32)
        self.assertEqual(len(heat[0]), 48)
        terrain_counts = result.summary["terrain_counts"]
        self.assertEqual(set(terrain_counts), {"plain", "forest", "wetland", "rocky", "water"})
        self.assertGreater(terrain_counts["wetland"], 0)
        self.assertGreater(terrain_counts["rocky"], 0)

        fertility_values = [value for row in fertility for value in row]
        moisture_values = [value for row in moisture for value in row]
        heat_values = [value for row in heat for value in row]

        self.assertTrue(all(0.0 <= value <= 1.0 for value in fertility_values))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in moisture_values))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in heat_values))
        self.assertGreater(max(fertility_values) - min(fertility_values), 0.35)
        self.assertGreater(max(moisture_values) - min(moisture_values), 0.35)
        self.assertGreater(max(heat_values) - min(heat_values), 0.35)
        self.assertIn("base_tile_fields", result.viewer["map"])
        self.assertEqual(
            set(result.summary["field_stats"].keys()),
            {
                "land_fertility",
                "land_moisture",
                "land_heat",
                "land_vegetation",
                "land_recovery_debt",
                "water_moisture",
                "water_heat",
            },
        )
        first_state = result.viewer["frames"][0]["field_state"]
        later_state = result.viewer["frames"][45]["field_state"]
        self.assertNotEqual(first_state["moisture_front_x"], later_state["moisture_front_x"])
        self.assertNotEqual(first_state["heat_front_y"], later_state["heat_front_y"])
        self.assertNotEqual(
            (first_state["disturbance_center_x"], first_state["disturbance_center_y"]),
            (later_state["disturbance_center_x"], later_state["disturbance_center_y"]),
        )
        self.assertGreaterEqual(
            result.summary["field_stats"]["land_vegetation"]["max"],
            result.summary["field_stats"]["land_vegetation"]["min"],
        )
        self.assertGreaterEqual(
            result.summary["field_stats"]["land_recovery_debt"]["max"],
            result.summary["field_stats"]["land_recovery_debt"]["min"],
        )

    def test_ecology_layer_changes_over_time_and_surfaces_recovery_states(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=240)).run()

        first_frame = result.viewer["frames"][0]
        final_frame = result.viewer["frames"][-1]

        self.assertNotEqual(
            first_frame["ecology_stats"]["avg_vegetation"],
            final_frame["ecology_stats"]["avg_vegetation"],
        )
        self.assertNotEqual(
            first_frame["ecology_state_counts"],
            final_frame["ecology_state_counts"],
        )
        self.assertGreater(
            final_frame["ecology_state_counts"]["recovering"]
            + final_frame["ecology_state_counts"]["depleted"],
            0,
        )
        self.assertGreater(
            result.viewer["analytics"]["ecology"]["avg_recovery_debt"][-1],
            0.0,
        )

    def test_hydrology_and_ecology_accounting_match_land_tiles(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=180)).run()

        land_tiles = result.summary["land_tile_count"]
        total_tiles = result.config["width"] * result.config["height"]
        self.assertEqual(
            total_tiles - result.summary["terrain_counts"]["water"],
            land_tiles,
        )

        for frame in (result.viewer["frames"][0], result.viewer["frames"][-1]):
            self.assertEqual(sum(frame["hydrology_primary_counts"].values()), land_tiles)
            self.assertEqual(sum(frame["hazard_counts"].values()), land_tiles)
            self.assertEqual(sum(frame["refuge_counts"].values()), land_tiles)
            self.assertEqual(sum(frame["ecology_state_counts"].values()), land_tiles)
            self.assertEqual(
                frame["hydrology_primary_stats"]["hard_access_tiles"],
                frame["hydrology_primary_counts"]["adjacent_water"]
                + frame["hydrology_primary_counts"]["wetland"]
                + frame["hydrology_primary_counts"]["flooded"],
            )
            self.assertLess(
                sum(frame["hydrology_support_counts"].values()),
                land_tiles * 2,
            )
            self.assertGreaterEqual(
                frame["hydrology_support_counts"]["shoreline_support"],
                frame["hydrology_primary_counts"]["adjacent_water"],
            )
            self.assertEqual(
                frame["hydrology_support_counts"]["wetland_support"],
                frame["hydrology_primary_counts"]["wetland"],
            )
            self.assertLessEqual(
                frame["hydrology_primary_counts"]["flooded"],
                frame["hydrology_support_counts"]["flooded_support"],
            )

            hydrology_non_land = sum(
                1
                for row in frame["hydrology_primary_codes"]
                for code in row
                if code == -1
            )
            hydrology_support_non_land = sum(
                1
                for row in frame["hydrology_support_codes"]
                for code in row
                if code == -1
            )
            refuge_non_land = sum(
                1
                for row in frame["refuge_codes"]
                for code in row
                if code == -1
            )
            refuge_score_non_land = sum(
                1
                for row in frame["refuge_score_codes"]
                for code in row
                if code == -1
            )
            ecology_non_land = sum(
                1
                for row in frame["ecology_state_codes"]
                for code in row
                if code == -1
            )
            hazard_non_land = sum(
                1
                for row in frame["hazard_type_codes"]
                for code in row
                if code == -1
            )
            hazard_level_non_land = sum(
                1
                for row in frame["hazard_level_codes"]
                for code in row
                if code == -1
            )
            carcass_non_land = sum(
                1
                for row in frame["carcass_energy_codes"]
                for code in row
                if code == -1
            )
            carcass_freshness_non_land = sum(
                1
                for row in frame["carcass_freshness_codes"]
                for code in row
                if code == -1
            )
            self.assertEqual(hydrology_non_land, result.summary["terrain_counts"]["water"])
            self.assertEqual(hydrology_support_non_land, result.summary["terrain_counts"]["water"])
            self.assertEqual(refuge_non_land, result.summary["terrain_counts"]["water"])
            self.assertEqual(refuge_score_non_land, result.summary["terrain_counts"]["water"])
            self.assertEqual(ecology_non_land, result.summary["terrain_counts"]["water"])
            self.assertEqual(hazard_non_land, result.summary["terrain_counts"]["water"])
            self.assertEqual(hazard_level_non_land, result.summary["terrain_counts"]["water"])
            self.assertEqual(carcass_non_land, result.summary["terrain_counts"]["water"])
            self.assertEqual(carcass_freshness_non_land, result.summary["terrain_counts"]["water"])

        self.assertEqual(
            result.viewer["frames"][-1]["hydrology_primary_counts"],
            result.summary["hydrology_primary_counts_at_end"],
        )
        self.assertEqual(
            result.viewer["frames"][-1]["hydrology_support_counts"],
            result.summary["hydrology_support_counts_at_end"],
        )
        self.assertEqual(
            result.viewer["frames"][-1]["hazard_counts"],
            result.summary["hazard_counts_at_end"],
        )
        self.assertEqual(
            result.viewer["frames"][-1]["refuge_counts"],
            result.summary["refuge_counts_at_end"],
        )
        self.assertEqual(
            result.viewer["frames"][-1]["ecology_state_counts"],
            result.summary["ecology_state_counts_at_end"],
        )
        self.assertEqual(
            sum(result.summary["hydrology_primary_counts_at_end"].values()),
            land_tiles,
        )
        self.assertEqual(
            sum(result.summary["hazard_counts_at_end"].values()),
            land_tiles,
        )
        self.assertEqual(
            sum(result.summary["refuge_counts_at_end"].values()),
            land_tiles,
        )
        self.assertEqual(
            sum(result.summary["ecology_state_counts_at_end"].values()),
            land_tiles,
        )

    def test_hydrology_and_refuge_semantics_are_denominator_explicit(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=180)).run()

        frame = result.viewer["frames"][-1]
        species_metrics = next(iter(frame["species_metrics"].values()))

        self.assertIn("avg_refuge_score_forest_tiles", frame["refuge_stats"])
        self.assertNotIn("avg_refuge_score", frame["refuge_stats"])
        self.assertIn("avg_refuge_score_occupied_tiles", species_metrics)
        self.assertNotIn("avg_refuge_score", species_metrics)
        self.assertIn("refuge_exposure_rate", species_metrics)
        self.assertIn("hydrology_primary", result.viewer["analytics"])
        self.assertIn("hydrology_support", result.viewer["analytics"])
        self.assertNotIn("hydrology", result.viewer["analytics"])
        self.assertIn("avg_hazard_level", frame["hazard_stats"])
        self.assertIn("carcass_tiles", frame["carcass_stats"])
        self.assertIn("combat_stats", frame)
        self.assertEqual(
            set(frame["hydrology_primary_counts"]),
            {"none", "adjacent_water", "wetland", "flooded"},
        )
        self.assertEqual(
            set(frame["hydrology_support_counts"]),
            {"shoreline_support", "wetland_support", "flooded_support"},
        )
        self.assertEqual(set(frame["hazard_counts"]), {"none", "exposure", "instability"})

    def test_hidden_forest_water_access_does_not_exist(self) -> None:
        seeds = [3, 7, 11]

        for seed in seeds:
            world = SimulationWorld(WorldConfig(seed=seed, max_ticks=220))
            world.run()
            hidden_forest_sources = 0
            forest_tiles = 0
            for y in range(world.config.height):
                for x in range(world.config.width):
                    tile = world.grid[y][x]
                    if tile.terrain != "forest":
                        continue
                    forest_tiles += 1
                    reason = world._water_access_reason(x, y)
                    if reason not in {"none", "flooded"} and not world._adjacent_to_water(x, y):
                        hidden_forest_sources += 1
            self.assertGreater(forest_tiles, 0)
            self.assertEqual(hidden_forest_sources, 0, msg=f"seed {seed} created hidden forest water")

    def test_shoreline_adjacent_water_niche_exists_across_targeted_seeds(self) -> None:
        seeds = [1, 3, 7, 11, 29]

        for seed in seeds:
            world = SimulationWorld(WorldConfig(seed=seed, max_ticks=220))
            initial_adjacent = sum(
                1
                for y in range(world.config.height)
                for x in range(world.config.width)
                if world.grid[y][x].terrain != "water"
                and world._water_access_reason(x, y) == "adjacent_water"
            )
            self.assertGreater(initial_adjacent, 0, msg=f"seed {seed} built no shoreline niche")

            result = world.run()
            final_adjacent = result.summary["hydrology_primary_counts_at_end"]["adjacent_water"]
            self.assertGreater(final_adjacent, 0, msg=f"seed {seed} lost shoreline niche by run end")
            self.assertGreater(
                result.summary["hydrology_support_counts_at_end"]["shoreline_support"],
                final_adjacent,
                msg=f"seed {seed} shoreline telemetry collapsed to only primary adjacent-water tiles",
            )

    def test_soft_refuge_is_selective_and_changes_over_time(self) -> None:
        seeds = [1, 7, 29]

        for seed in seeds:
            result = SimulationWorld(WorldConfig(seed=seed, max_ticks=240)).run()
            forest_tiles = result.summary["terrain_counts"]["forest"]
            refuge_series = result.viewer["analytics"]["refuge"]["canopy_refuge_tiles"]

            self.assertGreater(max(refuge_series), 0, msg=f"seed {seed} never formed any refuge")
            self.assertLess(
                max(refuge_series),
                forest_tiles,
                msg=f"seed {seed} treated the full forest layer as refuge",
            )
            self.assertGreater(
                max(refuge_series) - min(refuge_series),
                8,
                msg=f"seed {seed} refuge did not change meaningfully over time",
            )
            self.assertLess(
                result.summary["refuge_counts_at_end"]["canopy_refuge"],
                forest_tiles,
                msg=f"seed {seed} ended with every forest tile flagged as refuge",
            )

    def test_replay_writer_persists_summary_events_and_viewer_payload(self) -> None:
        result = SimulationWorld(WorldConfig(seed=5, max_ticks=120)).run()

        with TemporaryDirectory() as tmpdir:
            destination = write_json_replay(result, Path(tmpdir) / "run.json")
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(payload["run_id"], result.run_id)
        self.assertEqual(payload["summary"], result.summary)
        self.assertEqual(len(payload["events"]), len(result.events))
        self.assertEqual(payload["viewer"], result.viewer)
        self.assertIn("taxonomy", payload["viewer"])
        self.assertIn("species_catalog", payload["viewer"])
        self.assertIn("ecotype_catalog", payload["viewer"])
        self.assertIn("environment_fields", payload["viewer"]["map"])
        self.assertIn("base_tile_fields", payload["viewer"]["map"])
        self.assertIn("hydrology_primary_legend", payload["viewer"]["map"])
        self.assertNotIn("hydrology_legend", payload["viewer"]["map"])
        self.assertIn("analytics", payload["viewer"])
        self.assertIn("hydrology_primary", payload["viewer"]["analytics"])
        self.assertIn("hydrology_support", payload["viewer"]["analytics"])
        self.assertIn("refuge", payload["viewer"]["analytics"])
        self.assertIn("hazards", payload["viewer"]["analytics"])
        self.assertIn("carcasses", payload["viewer"]["analytics"])
        self.assertIn("combat", payload["viewer"]["analytics"])
        self.assertIn("ecology", payload["viewer"]["analytics"])
        self.assertIn("hydrology_primary_codes", payload["viewer"]["frames"][-1])
        self.assertIn("hydrology_support_codes", payload["viewer"]["frames"][-1])
        self.assertIn("refuge_codes", payload["viewer"]["frames"][-1])
        self.assertIn("refuge_score_codes", payload["viewer"]["frames"][-1])
        self.assertIn("hazard_type_codes", payload["viewer"]["frames"][-1])
        self.assertIn("hazard_level_codes", payload["viewer"]["frames"][-1])
        self.assertIn("carcass_energy_codes", payload["viewer"]["frames"][-1])
        self.assertIn("carcass_freshness_codes", payload["viewer"]["frames"][-1])
        self.assertIn("ecology_state_codes", payload["viewer"]["frames"][-1])
        self.assertIn("ecotype_counts", payload["viewer"]["frames"][-1])
        self.assertIn("ecotype_metrics", payload["viewer"]["frames"][-1])

    def test_species_ids_are_lineage_grounded_and_stable_while_alive(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=320)).run()

        agent_catalog = {int(agent_id): payload for agent_id, payload in result.viewer["agent_catalog"].items()}
        field_map = {
            field: index for index, field in enumerate(result.viewer["agent_encoding"])
        }
        lineage_splits = 0

        for frame in result.viewer["frames"]:
            seen_species: dict[int, int] = {}
            for row in frame["agents"]:
                agent_id = row[field_map["agent_id"]]
                lineage_id = agent_catalog[agent_id]["lineage_id"]
                species_id = row[field_map["species_id"]]
                self.assertEqual(species_id, lineage_id)
                if lineage_id in seen_species and seen_species[lineage_id] != species_id:
                    lineage_splits += 1
                seen_species[lineage_id] = species_id

        self.assertEqual(lineage_splits, 0)
        self.assertEqual(result.summary["alive_species_count"], len(result.summary["alive_lineages"]))
        self.assertEqual(result.summary["taxonomy_mode"], "lineage_species_v1")

    def test_multi_seed_runs_remain_viable_and_reproductive(self) -> None:
        seeds = [3, 7, 11, 17, 29]
        initial_agents = WorldConfig().initial_agents

        for seed in seeds:
            result = SimulationWorld(WorldConfig(seed=seed, max_ticks=800)).run()
            self.assertFalse(result.summary["extinct"], msg=f"seed {seed} went extinct")
            self.assertGreater(
                result.summary["births"],
                initial_agents,
                msg=f"seed {seed} stalled before sustained reproduction",
            )
            self.assertGreater(result.summary["alive_species_count"], 0, msg=f"seed {seed} lost all species")
            self.assertIsNotNone(result.summary["last_birth_tick"], msg=f"seed {seed} stopped reproducing")
            self.assertGreater(
                result.summary["last_birth_tick"],
                320,
                msg=f"seed {seed} stopped reproducing too early",
            )
            self.assertEqual(
                sum(result.summary["ecology_state_counts_at_end"].values()),
                result.summary["land_tile_count"],
                msg=f"seed {seed} miscounted ecology land tiles",
            )
            self.assertGreater(
                result.summary["hydrology_primary_counts_at_end"]["adjacent_water"],
                0,
                msg=f"seed {seed} lost shoreline water access in long run",
            )

    def test_trophic_roles_are_coupled_and_gradual(self) -> None:
        rng = Random(7)
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        base = Genome.sample_initial(rng)
        herbivore = replace(
            base,
            attack_power=0.5,
            meat_efficiency=0.42,
            plant_bias=1.4,
            carrion_bias=0.28,
            live_prey_bias=0.24,
        )
        omnivore = replace(
            base,
            attack_power=0.98,
            meat_efficiency=0.88,
            plant_bias=1.02,
            carrion_bias=0.92,
            live_prey_bias=0.84,
        )
        carnivore = replace(
            base,
            attack_power=1.34,
            meat_efficiency=1.22,
            plant_bias=0.82,
            carrion_bias=1.22,
            live_prey_bias=1.26,
        )
        self.assertEqual(world._trophic_role_for_genome(herbivore), "herbivore")
        self.assertEqual(world._trophic_role_for_genome(omnivore), "omnivore")
        self.assertEqual(world._trophic_role_for_genome(carnivore), "carnivore")

        stable_predator = replace(carnivore, mutation_scale=0.025)
        mutated_roles = {
            world._trophic_role_for_genome(stable_predator.mutate(Random(seed)))
            for seed in range(20)
        }
        self.assertTrue(mutated_roles <= {"omnivore", "carnivore"})

    def test_herbivores_cannot_attack_and_carcasses_are_meat_only(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        herbivore = world.alive_agents()[0]
        herbivore.genome = replace(
            herbivore.genome,
            attack_power=0.42,
            meat_efficiency=0.34,
            plant_bias=1.42,
            carrion_bias=0.24,
            live_prey_bias=0.22,
        )
        self.assertEqual(world._trophic_role(herbivore), "herbivore")
        self.assertFalse(world._can_attack(herbivore))

        tile = world.grid[herbivore.y][herbivore.x]
        tile.food = 0.0
        world._deposit_carcass(
            tile,
            x=herbivore.x,
            y=herbivore.y,
            energy=0.6,
            source_species=None,
            source_agent_id=None,
            cause="test",
            killer_id=None,
        )
        before_energy = herbivore.energy
        self.assertFalse(world._eat(herbivore))
        self.assertEqual(before_energy, herbivore.energy)
        self.assertEqual(tile.carcass_energy, 0.6)

    def test_carnivore_plant_fallback_is_low_yield_relative_to_carcass(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1, initial_agents=0))
        seeded_world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        base = seeded_world.alive_agents()[0].genome
        carnivore = replace(
            base,
            attack_power=1.55,
            meat_efficiency=1.45,
            food_efficiency=0.65,
            plant_bias=0.5,
            carrion_bias=1.45,
            live_prey_bias=1.45,
        )

        x = y = 0
        for row_y, row in enumerate(world.grid):
            for row_x, tile in enumerate(row):
                if tile.terrain != "water":
                    x = row_x
                    y = row_y
                    break
            else:
                continue
            break

        agent = Agent(
            agent_id=1,
            parent_id=None,
            lineage_id=1,
            birth_tick=0,
            death_tick=None,
            x=x,
            y=y,
            energy=0.2,
            hydration=carnivore.max_hydration * 0.8,
            health=carnivore.max_health * 0.9,
            max_health=carnivore.max_health,
            injury_load=0.0,
            age=0,
            alive=True,
            last_reproduction_tick=-10_000,
            last_damage_source="none",
            genome_vector=genome_vector(carnivore),
            genome=carnivore,
        )
        world._place_agent(agent)
        tile = world.grid[y][x]
        tile.food = 0.8

        plant_before = agent.energy
        world._consume_plant(agent)
        plant_gain = agent.energy - plant_before

        agent.energy = 0.2
        world._deposit_carcass(
            tile,
            x=x,
            y=y,
            energy=0.8,
            source_species=None,
            source_agent_id=None,
            cause="test",
            killer_id=None,
        )
        carcass_before = agent.energy
        world._consume_carcass(agent)
        carcass_gain = agent.energy - carcass_before

        self.assertEqual(world._trophic_role(agent), "carnivore")
        self.assertLess(plant_gain, 0.06)
        self.assertGreater(carcass_gain, plant_gain * 4)
        self.assertGreater(world.run_diet_totals["carcass_energy"], world.run_diet_totals["plant_energy"])

    def test_carcass_patches_stack_and_preserve_source_breakdown(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        first, second = world.alive_agents()[:2]
        shared_x, shared_y = first.x, first.y
        world.current_species_map[first.agent_id] = 1
        world.current_species_map[second.agent_id] = 2

        first_yield = world._project_carcass_yield(first)
        world._kill_agent(first, cause="test_first")
        tile = world.grid[shared_y][shared_x]
        self.assertAlmostEqual(tile.carcass_energy, first_yield, places=4)
        self.assertEqual(len(tile.carcass_deposits), 1)

        world.grid[second.y][second.x].occupant_id = None
        second.x = shared_x
        second.y = shared_y
        tile.occupant_id = second.agent_id
        second_yield = world._project_carcass_yield(second)
        world._kill_agent(second, cause="test_second")

        self.assertAlmostEqual(tile.carcass_energy, first_yield + second_yield, places=4)
        self.assertEqual(len(tile.carcass_deposits), 2)
        self.assertIsNone(tile.carcass_source_species)
        self.assertEqual(
            {entry["source_species"] for entry in world.events[-1].data["tile_source_breakdown_after"]},
            {1, 2},
        )
        self.assertTrue(world.events[-1].data["tile_mixed_sources_after"])

    def test_new_carcass_does_not_refresh_existing_patch_and_consumption_prefers_freshest(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        tile = world.grid[agent.y][agent.x]

        world._deposit_carcass(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.4,
            source_species=1,
            source_agent_id=101,
            cause="old",
            killer_id=None,
        )
        tile.carcass_deposits[0].freshness = 0.2
        world._deposit_carcass(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.3,
            source_species=2,
            source_agent_id=202,
            cause="new",
            killer_id=None,
        )

        old_deposit = next(
            deposit for deposit in tile.carcass_deposits if deposit.source_agent_id == 101
        )
        new_deposit = next(
            deposit for deposit in tile.carcass_deposits if deposit.source_agent_id == 202
        )
        self.assertAlmostEqual(old_deposit.freshness, 0.2, places=4)
        self.assertAlmostEqual(new_deposit.freshness, 1.0, places=4)
        self.assertLess(tile.carcass_decay, 1.0)

        consume_info = world._consume_carcass_from_tile(tile, 0.25)
        self.assertAlmostEqual(old_deposit.energy_remaining, 0.4, places=4)
        self.assertEqual(consume_info["deposit_breakdown"][0]["source_agent_id"], 202)

    def test_carcass_replay_events_include_location_and_source_breakdown(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        consumer = world.alive_agents()[0]
        consumer.genome = replace(
            consumer.genome,
            attack_power=0.82,
            meat_efficiency=1.08,
            plant_bias=0.22,
            carrion_bias=1.28,
            live_prey_bias=0.34,
        )
        tile = world.grid[consumer.y][consumer.x]

        world._deposit_carcass(
            tile,
            x=consumer.x,
            y=consumer.y,
            energy=0.18,
            source_species=3,
            source_agent_id=301,
            cause="test_a",
            killer_id=None,
        )
        world._deposit_carcass(
            tile,
            x=consumer.x,
            y=consumer.y,
            energy=0.12,
            source_species=4,
            source_agent_id=302,
            cause="test_b",
            killer_id=None,
        )

        self.assertTrue(world._can_consume_carcass(consumer))
        world._consume_carcass(consumer)

        event = world.events[-1]
        self.assertEqual(event.type.value, "agent_ate")
        self.assertEqual(event.data["food_source"], "carcass")
        self.assertEqual(event.data["x"], consumer.x)
        self.assertEqual(event.data["y"], consumer.y)
        self.assertTrue(event.data["source_breakdown"])
        self.assertTrue(event.data["deposit_breakdown"])
        self.assertIn("tile_carcass_energy_after", event.data)

    def test_attack_and_hazard_damage_are_distinguishable_and_carcasses_are_consumed(self) -> None:
        result = SimulationWorld(WorldConfig(seed=11, max_ticks=160)).run()
        damage_sources = {
            event["data"]["source"]
            for event in result.events
            if event["type"] == "agent_damaged"
        }
        self.assertIn("attack", damage_sources)
        self.assertTrue(any(source.startswith("hazard_") for source in damage_sources))
        self.assertGreater(result.summary["combat_end"]["attack_attempts"], 0)
        self.assertGreater(result.summary["combat_end"]["successful_attacks"], 0)
        self.assertGreater(result.summary["carcass_end"]["consumption_events"], 0)
        self.assertGreater(result.summary["carcass_end"]["deposition_events"], 0)
        self.assertAlmostEqual(result.summary["carcass_end"]["conservation_error"], 0.0, places=4)
        self.assertGreater(result.summary["carcass_stats_at_end"]["carcass_tiles"], 0)


if __name__ == "__main__":
    unittest.main()
