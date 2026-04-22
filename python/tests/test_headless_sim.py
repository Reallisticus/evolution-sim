from __future__ import annotations

import json
import statistics
import unittest
from dataclasses import replace
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from evolution_sim.config import ClimateConfig, ResourceRegrowthConfig, WorldConfig
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
        self.assertIn("biotic_fields", result.viewer["frames"][-1])
        self.assertIn("biotic_field_stats", result.viewer["frames"][-1])
        self.assertIn("fresh_kill_energy_codes", result.viewer["frames"][-1])
        self.assertIn("fresh_kill_stats", result.viewer["frames"][-1])
        self.assertIn("fresh_kill_patches", result.viewer["frames"][-1])
        self.assertIn("fresh_kill_flow", result.viewer["frames"][-1])
        self.assertIn("carcass_energy_codes", result.viewer["frames"][-1])
        self.assertIn("carcass_freshness_codes", result.viewer["frames"][-1])
        self.assertIn("carcass_stats", result.viewer["frames"][-1])
        self.assertIn("carcass_patches", result.viewer["frames"][-1])
        self.assertIn("combat_stats", result.viewer["frames"][-1])
        self.assertIn("diet_stats", result.viewer["frames"][-1])
        self.assertIn("diet_by_trophic_role", result.viewer["frames"][-1])
        self.assertIn("diet_by_meat_mode", result.viewer["frames"][-1])
        self.assertIn("trophic_role_counts", result.viewer["frames"][-1])
        self.assertIn("meat_mode_counts", result.viewer["frames"][-1])
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
        self.assertIn("biotic_field_stats_at_end", result.summary)
        self.assertIn("fresh_kill_stats_at_end", result.summary)
        self.assertIn("carcass_stats_at_end", result.summary)
        self.assertIn("combat_end", result.summary)
        self.assertIn("fresh_kill_end", result.summary)
        self.assertIn("carcass_end", result.summary)
        self.assertIn("conservation_error", result.summary["carcass_end"])
        self.assertIn("diet_end", result.summary)
        self.assertIn("diet_by_trophic_role_end", result.summary)
        self.assertIn("diet_by_meat_mode_end", result.summary)
        self.assertIn("top_species_by_realized_fresh_kill_share", result.summary)
        self.assertIn("top_species_by_realized_animal_share", result.summary)
        self.assertIn("top_carnivore_species", result.summary)
        self.assertIn("top_hunter_species", result.summary)
        self.assertIn("top_scavenger_species", result.summary)
        self.assertIn("top_species_by_realized_carcass_share", result.summary)
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
        self.assertIn("fresh_kill_consumption", sample_metrics)
        self.assertIn("fresh_kill_energy_consumed", sample_metrics)
        self.assertIn("carcass_consumption", sample_metrics)
        self.assertIn("carcass_energy_consumed", sample_metrics)
        self.assertIn("avg_matched_diet_ratio", sample_metrics)
        self.assertIn("realized_plant_share", sample_metrics)
        self.assertIn("realized_animal_share", sample_metrics)
        self.assertIn("realized_fresh_kill_share", sample_metrics)
        self.assertIn("realized_carcass_share", sample_metrics)
        self.assertIn("hazard_occupancy", sample_metrics)
        self.assertIn("trophic_role_occupancy", sample_metrics)
        self.assertIn("meat_mode_occupancy", sample_metrics)
        self.assertIn("habitat", result.viewer["analytics"])
        self.assertIn("hydrology_primary", result.viewer["analytics"])
        self.assertIn("hydrology_support", result.viewer["analytics"])
        self.assertIn("refuge", result.viewer["analytics"])
        self.assertIn("hazards", result.viewer["analytics"])
        self.assertIn("biotic_fields", result.viewer["analytics"])
        self.assertIn("fresh_kill", result.viewer["analytics"])
        self.assertIn("carcasses", result.viewer["analytics"])
        self.assertIn("diet", result.viewer["analytics"])
        self.assertIn("diet_by_trophic_role", result.viewer["analytics"])
        self.assertIn("diet_by_meat_mode", result.viewer["analytics"])
        self.assertIn("combat", result.viewer["analytics"])
        self.assertIn("ecology", result.viewer["analytics"])
        self.assertIn("trophic_roles", result.viewer["analytics"])
        self.assertIn("meat_modes", result.viewer["analytics"])
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
        self.assertIn("carcass_patches", payload["viewer"]["frames"][-1])
        self.assertIn("ecology_state_codes", payload["viewer"]["frames"][-1])
        self.assertIn("ecotype_counts", payload["viewer"]["frames"][-1])
        self.assertIn("ecotype_metrics", payload["viewer"]["frames"][-1])

    def test_species_ids_only_change_on_logged_speciation_events(self) -> None:
        result = None
        for seed in [3, 7, 11, 17, 29]:
            candidate = SimulationWorld(WorldConfig(seed=seed, max_ticks=320)).run()
            if candidate.summary["speciation_events"] > 0:
                result = candidate
                break

        self.assertIsNotNone(result)
        assert result is not None

        field_map = {
            field: index for index, field in enumerate(result.viewer["agent_encoding"])
        }
        species_catalog = result.viewer["species_catalog"]
        event_ticks = {
            int(event["tick"]) for event in result.viewer["taxonomy"]["events"]
        }
        prior_species_by_agent: dict[int, int] = {}
        change_count = 0

        for frame in result.viewer["frames"]:
            tick = int(frame["tick"])
            for row in frame["agents"]:
                agent_id = int(row[field_map["agent_id"]])
                species_id = int(row[field_map["species_id"]])
                species_record = species_catalog[str(species_id)]
                self.assertLessEqual(species_record["first_seen_tick"], tick)
                self.assertGreaterEqual(species_record["last_seen_tick"], tick)
                previous_species_id = prior_species_by_agent.get(agent_id)
                if previous_species_id is not None and previous_species_id != species_id:
                    change_count += 1
                    self.assertIn(tick, event_ticks)
                    self.assertEqual(
                        species_catalog[str(species_id)]["parent_species_id"],
                        previous_species_id,
                    )
                prior_species_by_agent[agent_id] = species_id

        self.assertGreater(change_count, 0)
        self.assertGreater(result.summary["speciation_events"], 0)
        self.assertEqual(result.summary["taxonomy_mode"], "replay_clade_species_v2")
        self.assertEqual(
            result.summary["species_status_counts"],
            result.viewer["taxonomy"]["species_status_counts"],
        )

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

    def _archetype_genomes(self) -> dict[str, Genome]:
        base = Genome.sample_initial(Random(13))
        return {
            "herbivore": replace(
                base,
                attack_power=0.44,
                attack_cost_multiplier=1.24,
                defense_rating=0.76,
                meat_efficiency=0.24,
                food_efficiency=1.58,
                water_efficiency=1.22,
                plant_bias=1.72,
                carrion_bias=0.22,
                live_prey_bias=0.2,
                reproduction_threshold=0.68,
                mutation_scale=0.03,
            ),
            "hunter": replace(
                base,
                attack_power=1.6,
                attack_cost_multiplier=0.76,
                defense_rating=1.18,
                meat_efficiency=1.56,
                food_efficiency=0.5,
                water_efficiency=0.94,
                plant_bias=0.46,
                carrion_bias=0.74,
                live_prey_bias=1.58,
                reproduction_threshold=0.72,
                mutation_scale=0.03,
            ),
            "scavenger": replace(
                base,
                attack_power=0.82,
                attack_cost_multiplier=0.96,
                defense_rating=0.98,
                meat_efficiency=1.5,
                food_efficiency=0.56,
                water_efficiency=1.0,
                plant_bias=0.48,
                carrion_bias=1.6,
                live_prey_bias=0.38,
                reproduction_threshold=0.72,
                mutation_scale=0.03,
            ),
            "omnivore": replace(
                base,
                attack_power=1.0,
                attack_cost_multiplier=0.92,
                defense_rating=0.94,
                meat_efficiency=1.02,
                food_efficiency=1.12,
                water_efficiency=1.06,
                plant_bias=1.06,
                carrion_bias=0.94,
                live_prey_bias=0.9,
                reproduction_threshold=0.69,
                mutation_scale=0.03,
            ),
        }

    def _fixture_world(
        self,
        *,
        seed: int,
        max_ticks: int,
        width: int = 18,
        height: int = 12,
        max_agents: int = 180,
        resources: ResourceRegrowthConfig | None = None,
        climate: ClimateConfig | None = None,
    ) -> SimulationWorld:
        config = WorldConfig(
            seed=seed,
            width=width,
            height=height,
            max_ticks=max_ticks,
            initial_agents=0,
            max_agents=max_agents,
            resources=resources or ResourceRegrowthConfig(),
            climate=climate or ClimateConfig(),
        )
        return SimulationWorld(config)

    def _configure_uniform_arena(
        self,
        world: SimulationWorld,
        *,
        terrain: str = "wetland",
        food: float,
        vegetation: float,
        shelter: float,
        recovery_debt: float,
        fertility: float,
        moisture: float,
        heat: float,
    ) -> None:
        for row in world.grid:
            for tile in row:
                tile.terrain = terrain
                tile.food = food
                tile.water = 1.0 if terrain == "water" else 0.0
                tile.fertility = fertility
                tile.moisture = moisture
                tile.heat = heat
                tile.vegetation = vegetation
                tile.shelter = shelter
                tile.recovery_debt = recovery_debt
                tile.fresh_kill_deposits = []
                tile.carcass_deposits = []
                tile.occupant_id = None
        world.cached_habitat_tick = None
        world.cached_habitat_grid = None
        world.cached_habitat_counts = None
        world.cached_climate_tick = None
        world.cached_climate_state = None
        world._invalidate_biotic_state()

    def _refresh_fixture_world(self, world: SimulationWorld) -> None:
        world.cached_habitat_tick = None
        world.cached_habitat_grid = None
        world.cached_habitat_counts = None
        world.cached_climate_tick = None
        world.cached_climate_state = None
        world._invalidate_biotic_state()

    def _set_water_tile(self, world: SimulationWorld, x: int, y: int, *, heat: float = 0.3) -> None:
        tile = world.grid[y][x]
        tile.terrain = "water"
        tile.food = 0.0
        tile.water = 1.0
        tile.fertility = 0.0
        tile.moisture = 1.0
        tile.heat = heat
        tile.vegetation = 0.0
        tile.shelter = 0.0
        tile.recovery_debt = 0.0
        tile.fresh_kill_deposits = []
        tile.carcass_deposits = []
        tile.occupant_id = None

    def _assert_fixture_profile(
        self,
        world: SimulationWorld,
        genome: Genome,
        *,
        role: str,
        meat_mode: str,
    ) -> None:
        profile = world._trophic_profile_for_genome(genome)
        self.assertEqual(profile.role, role)
        self.assertEqual(profile.meat_mode, meat_mode)

    def _add_agent(
        self,
        world: SimulationWorld,
        *,
        genome: Genome,
        x: int,
        y: int,
        lineage_id: int,
        energy_ratio: float = 0.9,
        hydration_ratio: float = 0.9,
        health_ratio: float = 0.96,
        age: int = 32,
    ) -> int:
        agent_id = world.next_agent_id
        agent = Agent(
            agent_id=agent_id,
            parent_id=None,
            lineage_id=lineage_id,
            birth_tick=0,
            death_tick=None,
            x=x,
            y=y,
            energy=genome.max_energy * energy_ratio,
            hydration=genome.max_hydration * hydration_ratio,
            health=genome.max_health * health_ratio,
            max_health=genome.max_health,
            injury_load=0.0,
            age=age,
            alive=True,
            last_reproduction_tick=-10_000,
            last_damage_source="none",
            recent_plant_energy=0.0,
            recent_fresh_kill_energy=0.0,
            recent_carcass_energy=0.0,
            genome_vector=genome_vector(genome),
            genome=genome,
        )
        world._place_agent(agent)
        world.next_agent_id += 1
        return agent_id

    def _seed_archetypes(
        self,
        world: SimulationWorld,
        *,
        counts: dict[str, int],
        genomes: dict[str, Genome],
    ) -> dict[int, int]:
        initial_counts: dict[int, int] = {}
        placements = {
            "herbivore": (2, 2),
            "hunter": (world.config.width - 3, 2),
            "scavenger": (2, world.config.height - 3),
            "omnivore": (world.config.width - 3, world.config.height - 3),
        }
        lineage_ids = {"herbivore": 101, "hunter": 202, "scavenger": 303, "omnivore": 404}
        for key, count in counts.items():
            start_x, start_y = placements[key]
            lineage_id = lineage_ids[key]
            initial_counts[lineage_id] = count
            for offset in range(count):
                x = start_x + (offset % 3)
                y = start_y + (offset // 3)
                self._add_agent(
                    world,
                    genome=genomes[key],
                    x=min(x, world.config.width - 1),
                    y=min(y, world.config.height - 1),
                    lineage_id=lineage_id,
                )
        world.current_species_map = {
            agent.agent_id: agent.lineage_id for agent in world.alive_agents()
        }
        world.agent_last_species_map = world.current_species_map.copy()
        world._invalidate_biotic_state()
        return initial_counts

    def _lineage_stats(
        self,
        result,
        initial_counts: dict[int, int],
    ) -> tuple[dict[int, dict[str, int]], dict[int, dict[str, float]], dict[int, int]]:
        lineage_by_agent = {
            int(payload["agent_id"]): int(payload["lineage_id"])
            for payload in result.viewer["agent_catalog"].values()
        }
        population: dict[int, dict[str, int]] = {}
        for lineage_id, initial in initial_counts.items():
            agents = [
                payload
                for payload in result.viewer["agent_catalog"].values()
                if int(payload["lineage_id"]) == lineage_id
            ]
            population[lineage_id] = {
                "total": len(agents),
                "alive": sum(payload["death_tick"] is None for payload in agents),
                "births": len(agents) - initial,
            }

        diet: dict[int, dict[str, float]] = {
            lineage_id: {"plant": 0.0, "fresh_kill": 0.0, "carcass": 0.0}
            for lineage_id in initial_counts
        }
        kills = {lineage_id: 0 for lineage_id in initial_counts}
        for event in result.events:
            if event["type"] == "agent_ate":
                lineage_id = lineage_by_agent.get(int(event["agent_id"]))
                if lineage_id in diet:
                    diet[lineage_id][str(event["data"]["food_source"])] += float(
                        event["data"]["gained_energy"]
                    )
            elif event["type"] == "agent_attacked" and event["data"]["kill"]:
                lineage_id = lineage_by_agent.get(int(event["agent_id"]))
                if lineage_id in kills:
                    kills[lineage_id] += 1

        diet_shares: dict[int, dict[str, float]] = {}
        for lineage_id, totals in diet.items():
            total = totals["plant"] + totals["fresh_kill"] + totals["carcass"]
            diet_shares[lineage_id] = {
                key: (value / total if total > 0 else 0.0)
                for key, value in totals.items()
            }
            diet_shares[lineage_id]["animal"] = (
                (totals["fresh_kill"] + totals["carcass"]) / total if total > 0 else 0.0
            )
        return population, diet_shares, kills

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

    def test_initial_population_reaches_specialist_and_secondary_meat_axes(self) -> None:
        trophic_counts: dict[str, int] = {"herbivore": 0, "omnivore": 0, "carnivore": 0}
        carnivore_seed_hits = 0
        scavenger_seed_hits = 0
        hunter_seed_hits = 0

        for seed in range(1, 61):
            world = SimulationWorld(WorldConfig(seed=seed, max_ticks=1))
            roles = [world._trophic_role(agent) for agent in world.alive_agents()]
            meat_modes = [world._meat_mode(agent) for agent in world.alive_agents()]
            for role in roles:
                trophic_counts[role] += 1
            if "carnivore" in roles:
                carnivore_seed_hits += 1
            if "scavenger" in meat_modes:
                scavenger_seed_hits += 1
            if "hunter" in meat_modes:
                hunter_seed_hits += 1

        self.assertGreater(trophic_counts["herbivore"], 0)
        self.assertGreater(trophic_counts["carnivore"], 0)
        self.assertGreater(carnivore_seed_hits, 50)
        self.assertGreater(scavenger_seed_hits, 50)
        self.assertGreater(hunter_seed_hits, 50)

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
            recent_plant_energy=0.0,
            recent_fresh_kill_energy=0.0,
            recent_carcass_energy=0.0,
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
        self.assertLessEqual(plant_gain, max(0.001, carcass_gain * 0.05))
        self.assertGreater(world.run_diet_totals["carcass_energy"], world.run_diet_totals["plant_energy"])

    def test_feeding_telemetry_tracks_realized_gain_after_caps(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        tile = world.grid[agent.y][agent.x]

        agent.genome = replace(
            agent.genome,
            food_efficiency=1.55,
            plant_bias=1.65,
            attack_power=0.45,
            meat_efficiency=0.28,
            carrion_bias=0.2,
            live_prey_bias=0.2,
        )
        agent.energy = agent.genome.max_energy - 0.02
        tile.food = 0.8
        world.tick_feeding_events.clear()
        plant_before = agent.energy
        self.assertFalse(world._consume_plant(agent))
        plant_event = world.tick_feeding_events[-1]
        self.assertAlmostEqual(plant_event["gained_energy"], agent.energy - plant_before, places=4)
        self.assertAlmostEqual(plant_event["energy_after"] - plant_event["energy_before"], plant_event["gained_energy"], places=4)
        self.assertGreater(plant_event["potential_energy"], plant_event["gained_energy"])

        agent.genome = replace(
            agent.genome,
            attack_power=1.32,
            meat_efficiency=1.35,
            food_efficiency=0.55,
            plant_bias=0.45,
            carrion_bias=1.4,
            live_prey_bias=1.32,
        )
        agent.energy = agent.genome.max_energy - 0.02
        world.tick_feeding_events.clear()
        world._deposit_carcass(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.6,
            source_species=None,
            source_agent_id=None,
            cause="test",
            killer_id=None,
        )
        carcass_before = agent.energy
        self.assertFalse(world._consume_carcass(agent))
        carcass_event = world.tick_feeding_events[-1]
        self.assertAlmostEqual(
            carcass_event["gained_energy"],
            agent.energy - carcass_before,
            places=4,
        )
        self.assertAlmostEqual(
            carcass_event["energy_after"] - carcass_event["energy_before"],
            carcass_event["gained_energy"],
            places=4,
        )
        self.assertGreater(carcass_event["potential_energy"], carcass_event["gained_energy"])

        target = world.alive_agents()[1]
        target.health = 0.01
        world.grid[target.y][target.x].occupant_id = None
        target.x = agent.x + 1 if agent.x + 1 < world.config.width else agent.x - 1
        target.y = agent.y
        world.grid[target.y][target.x].occupant_id = target.agent_id
        world.tick_feeding_events.clear()
        world._deposit_fresh_kill(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.55,
            source_species=None,
            source_agent_id=target.agent_id,
            killer_id=agent.agent_id,
        )
        self.assertFalse(world._consume_fresh_kill(agent))
        fresh_kill_event = world.tick_feeding_events[-1]
        self.assertEqual(fresh_kill_event["food_source"], "fresh_kill")
        self.assertAlmostEqual(
            fresh_kill_event["energy_after"] - fresh_kill_event["energy_before"],
            fresh_kill_event["gained_energy"],
            places=4,
        )

    def test_healthy_full_energy_carnivore_does_not_waste_turn_on_carcass(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        agent.genome = replace(
            agent.genome,
            attack_power=1.52,
            meat_efficiency=1.42,
            food_efficiency=0.58,
            plant_bias=0.46,
            carrion_bias=1.44,
            live_prey_bias=1.42,
        )
        agent.energy = agent.genome.max_energy
        agent.health = agent.max_health
        agent.hydration = agent.genome.max_hydration
        tile = world.grid[agent.y][agent.x]
        tile.food = 0.0
        world._deposit_carcass(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.5,
            source_species=None,
            source_agent_id=None,
            cause="test",
            killer_id=None,
        )
        world._deposit_fresh_kill(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.5,
            source_species=None,
            source_agent_id=None,
            killer_id=None,
        )

        self.assertEqual(world._trophic_role(agent), "carnivore")
        self.assertNotEqual(world._choose_action(agent), "eat")

    def test_carnivore_tile_scoring_is_not_led_by_plant_richness(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        agent.genome = replace(
            agent.genome,
            attack_power=1.48,
            meat_efficiency=1.4,
            food_efficiency=0.6,
            plant_bias=0.44,
            carrion_bias=1.42,
            live_prey_bias=1.42,
        )
        profile = world._trophic_profile(agent)
        tile = world.grid[agent.y][agent.x]
        original_food = tile.food
        original_vegetation = tile.vegetation
        season = world._season_state()["name"]
        base_score = world._candidate_tile_score(
            agent,
            agent.x,
            agent.y,
            season,
            water_urgency=0.2,
            food_urgency=0.8,
            profile=profile,
        )
        tile.food = 1.0
        tile.vegetation = 1.0
        rich_score = world._candidate_tile_score(
            agent,
            agent.x,
            agent.y,
            season,
            water_urgency=0.2,
            food_urgency=0.8,
            profile=profile,
        )
        tile.food = original_food
        tile.vegetation = original_vegetation

        self.assertEqual(profile.role, "carnivore")
        self.assertLess(rich_score - base_score, 0.02)

    def test_unknown_carcass_provenance_stays_unknown_when_mixed_with_known_source(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        tile = world.grid[agent.y][agent.x]

        world._deposit_carcass(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.18,
            source_species=5,
            source_agent_id=501,
            cause="known",
            killer_id=None,
        )
        world._deposit_carcass(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.12,
            source_species=None,
            source_agent_id=None,
            cause="unknown",
            killer_id=None,
        )

        patch_state = world._carcass_tile_summary_for_position(agent.x, agent.y)
        self.assertIsNone(tile.carcass_source_species)
        self.assertIsNone(patch_state["dominant_source_species"])
        self.assertTrue(patch_state["mixed_sources"])

    def test_unknown_fresh_kill_provenance_stays_unknown_when_mixed_with_known_source(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        tile = world.grid[agent.y][agent.x]

        world._deposit_fresh_kill(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.18,
            source_species=5,
            source_agent_id=501,
            killer_id=11,
        )
        world._deposit_fresh_kill(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.12,
            source_species=None,
            source_agent_id=None,
            killer_id=None,
        )

        patch_state = world._fresh_kill_tile_summary_for_position(agent.x, agent.y)
        self.assertIsNone(patch_state["dominant_source_species"])
        self.assertTrue(patch_state["mixed_sources"])

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
        self.assertEqual(
            {entry["source_agent_id"] for entry in world.events[-1].data["tile_source_breakdown_after"]},
            {first.agent_id, second.agent_id},
        )
        self.assertTrue(
            all("death_tick" in entry for entry in world.events[-1].data["tile_source_breakdown_after"])
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
        self.assertIn("tile_source_breakdown_after", event.data)
        self.assertTrue(all("source_agent_id" in entry for entry in event.data["source_breakdown"]))
        self.assertTrue(all("death_tick" in entry for entry in event.data["source_breakdown"]))
        self.assertIn("tile_carcass_energy_after", event.data)
        deposit_events = [event for event in world.events if event.type.value == "carcass_deposited"]
        self.assertEqual(len(deposit_events), 2)
        self.assertTrue(all("tile_source_breakdown_after" in event.data for event in deposit_events))

    def test_attack_and_hazard_damage_are_distinguishable_and_carcasses_are_consumed(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=160)).run()
        damage_sources = {
            event["data"]["source"]
            for event in result.events
            if event["type"] == "agent_damaged"
        }
        self.assertIn("attack", damage_sources)
        self.assertTrue(any(source.startswith("hazard_") for source in damage_sources))
        self.assertGreater(result.summary["combat_end"]["attack_attempts"], 0)
        self.assertGreater(result.summary["combat_end"]["successful_attacks"], 0)
        self.assertGreater(result.summary["carcass_end"]["deposition_events"], 0)
        self.assertAlmostEqual(result.summary["carcass_end"]["conservation_error"], 0.0, places=4)
        self.assertTrue(
            any(frame["carcass_stats"]["carcass_tiles"] > 0 for frame in result.viewer["frames"])
        )

    def test_fixture_plant_only_arena_favors_herbivore_births(self) -> None:
        genomes = self._archetype_genomes()
        plant_omnivore = replace(
            genomes["omnivore"],
            food_efficiency=1.28,
            plant_bias=1.18,
            carrion_bias=0.74,
            live_prey_bias=0.72,
            reproduction_threshold=0.71,
        )
        world = self._fixture_world(seed=91, max_ticks=90, max_agents=140)
        self._configure_uniform_arena(
            world,
            terrain="forest",
            food=0.94,
            vegetation=0.92,
            shelter=0.18,
            recovery_debt=0.04,
            fertility=0.82,
            moisture=0.84,
            heat=0.22,
        )
        self._assert_fixture_profile(world, genomes["herbivore"], role="herbivore", meat_mode="none")
        self._assert_fixture_profile(world, plant_omnivore, role="omnivore", meat_mode="mixed")
        self._assert_fixture_profile(world, genomes["hunter"], role="carnivore", meat_mode="hunter")
        self._assert_fixture_profile(world, genomes["scavenger"], role="carnivore", meat_mode="scavenger")

        mid = world.config.width // 2
        for y in range(world.config.height):
            for x in (mid - 1, mid):
                self._set_water_tile(world, x, y, heat=0.22)

        initial_counts = {101: 6, 404: 4, 202: 2, 303: 2}
        for offset in range(6):
            self._add_agent(
                world,
                genome=genomes["herbivore"],
                x=2 + (offset % 3),
                y=2 + (offset // 3),
                lineage_id=101,
            )
        for offset in range(4):
            self._add_agent(
                world,
                genome=plant_omnivore,
                x=4 + (offset % 2),
                y=7 + (offset // 2),
                lineage_id=404,
            )
        for offset in range(2):
            self._add_agent(
                world,
                genome=genomes["hunter"],
                x=world.config.width - 4 + offset,
                y=2,
                lineage_id=202,
            )
        for offset in range(2):
            self._add_agent(
                world,
                genome=genomes["scavenger"],
                x=world.config.width - 4 + offset,
                y=7,
                lineage_id=303,
            )
        world.current_species_map = {
            agent.agent_id: agent.lineage_id for agent in world.alive_agents()
        }
        world.agent_last_species_map = world.current_species_map.copy()
        self._refresh_fixture_world(world)
        result = world.run()
        population, _diet_shares, _kills = self._lineage_stats(result, initial_counts)

        herbivore_births = population[101]["births"]
        omnivore_births = population[404]["births"]
        carnivore_births = population[202]["births"] + population[303]["births"]
        self.assertGreater(herbivore_births, omnivore_births)
        self.assertGreater(omnivore_births, carnivore_births)

    def test_fixture_carrion_only_arena_favors_scavenger_over_hunter(self) -> None:
        genomes = self._archetype_genomes()
        world = self._fixture_world(seed=92, max_ticks=72, width=24, height=16, max_agents=120)
        self._configure_uniform_arena(
            world,
            food=0.0,
            vegetation=0.18,
            shelter=0.12,
            recovery_debt=0.22,
            fertility=0.46,
            moisture=0.78,
            heat=0.26,
        )
        mid_left = world.config.width // 2 - 1
        mid_right = world.config.width // 2
        for y in range(world.config.height):
            for x in (mid_left, mid_right):
                tile = world.grid[y][x]
                tile.terrain = "water"
                tile.food = 0.0
                tile.water = 1.0
                tile.fertility = 0.0
                tile.moisture = 1.0
                tile.heat = 0.26
                tile.vegetation = 0.0
                tile.shelter = 0.0
                tile.recovery_debt = 0.0
                tile.fresh_kill_deposits = []
                tile.carcass_deposits = []
                tile.occupant_id = None
        for y in range(2, world.config.height - 1, 3):
            for x in list(range(2, mid_left - 1, 3)) + list(
                range(mid_right + 2, world.config.width - 1, 3)
            ):
                world._deposit_carcass(
                    world.grid[y][x],
                    x=x,
                    y=y,
                    energy=1.2,
                    source_species=999,
                    source_agent_id=None,
                    cause="fixture",
                    killer_id=None,
                )
        initial_counts = self._seed_archetypes(
            world,
            counts={"scavenger": 4, "hunter": 1},
            genomes=genomes,
        )
        result = world.run()
        population, diet_shares, kills = self._lineage_stats(result, initial_counts)

        self.assertGreater(population[303]["alive"], 0)
        self.assertGreater(population[303]["births"], 0)
        self.assertEqual(population[202]["births"], 0)
        self.assertEqual(population[202]["alive"], 0)
        self.assertGreater(diet_shares[303]["carcass"], 0.9)
        self.assertEqual(kills[202], 0)

    def test_fixture_prey_rich_arena_favors_hunter_over_scavenger(self) -> None:
        genomes = self._archetype_genomes()
        world = self._fixture_world(seed=93, max_ticks=96, width=24, height=16, max_agents=180)
        self._configure_uniform_arena(
            world,
            food=0.92,
            vegetation=0.9,
            shelter=0.2,
            recovery_debt=0.05,
            fertility=0.84,
            moisture=0.82,
            heat=0.24,
        )
        initial_counts = self._seed_archetypes(
            world,
            counts={"herbivore": 16, "hunter": 3, "scavenger": 2},
            genomes=genomes,
        )
        result = world.run()
        population, _diet_shares, kills = self._lineage_stats(result, initial_counts)

        self.assertGreater(population[202]["births"], population[303]["births"])
        self.assertGreater(kills[202], kills[303])

    def test_fixture_mixed_stable_arena_specialists_outperform_omnivore_in_own_channels(self) -> None:
        genomes = self._archetype_genomes()
        mixed_omnivore = replace(
            genomes["omnivore"],
            attack_power=1.16,
            meat_efficiency=1.32,
            food_efficiency=0.9,
            plant_bias=0.9,
            carrion_bias=1.3,
            live_prey_bias=0.86,
            reproduction_threshold=0.72,
        )
        world = self._fixture_world(seed=94, max_ticks=100, max_agents=180, width=24, height=16)
        self._configure_uniform_arena(
            world,
            terrain="forest",
            food=0.35,
            vegetation=0.4,
            shelter=0.16,
            recovery_debt=0.08,
            fertility=0.76,
            moisture=0.8,
            heat=0.28,
        )
        self._assert_fixture_profile(world, genomes["herbivore"], role="herbivore", meat_mode="none")
        self._assert_fixture_profile(world, mixed_omnivore, role="omnivore", meat_mode="mixed")
        self._assert_fixture_profile(world, genomes["hunter"], role="carnivore", meat_mode="hunter")
        self._assert_fixture_profile(world, genomes["scavenger"], role="carnivore", meat_mode="scavenger")

        for y, row in enumerate(world.grid):
            for x, tile in enumerate(row):
                if x == 0 or x == world.config.width - 1:
                    self._set_water_tile(world, x, y, heat=0.28)
                    continue
                if x < 8:
                    tile.food = 0.92
                    tile.vegetation = 0.88
                    tile.shelter = 0.2
                elif x < 16:
                    tile.food = 0.05
                    tile.vegetation = 0.16
                    tile.shelter = 0.1
                else:
                    tile.food = 0.25
                    tile.vegetation = 0.32
                    tile.shelter = 0.14

        mid_y = world.config.height // 2
        for x in range(8, 16, 2):
            world._deposit_carcass(
                world.grid[mid_y][x],
                x=x,
                y=mid_y,
                energy=1.4,
                source_species=777,
                source_agent_id=None,
                cause="fixture",
                killer_id=None,
            )
        initial_counts = {101: 6, 202: 4, 303: 4, 404: 4}
        for offset in range(6):
            self._add_agent(
                world,
                genome=genomes["herbivore"],
                x=2 + (offset % 3),
                y=2 + (offset // 3),
                lineage_id=101,
            )
        for offset in range(4):
            self._add_agent(
                world,
                genome=genomes["hunter"],
                x=18 + (offset % 2),
                y=2 + (offset // 2),
                lineage_id=202,
            )
        for offset in range(4):
            self._add_agent(
                world,
                genome=genomes["scavenger"],
                x=8 + (offset % 2),
                y=mid_y - 1 + (offset // 2),
                lineage_id=303,
            )
        for offset in range(4):
            self._add_agent(
                world,
                genome=mixed_omnivore,
                x=10 + (offset % 2),
                y=mid_y - 1 + (offset // 2),
                lineage_id=404,
            )
        world.current_species_map = {
            agent.agent_id: agent.lineage_id for agent in world.alive_agents()
        }
        world.agent_last_species_map = world.current_species_map.copy()
        self._refresh_fixture_world(world)
        result = world.run()
        _population, diet_shares, _kills = self._lineage_stats(result, initial_counts)

        self.assertGreater(diet_shares[101]["plant"], diet_shares[404]["plant"])
        self.assertGreater(diet_shares[202]["fresh_kill"], diet_shares[404]["fresh_kill"])
        self.assertGreater(diet_shares[303]["carcass"], diet_shares[404]["carcass"])

    def test_fixture_shocky_seasonal_arena_favors_omnivore_survival_not_reproduction(self) -> None:
        genomes = self._archetype_genomes()
        shock_herbivore = replace(
            genomes["herbivore"],
            water_efficiency=0.64,
            food_efficiency=1.46,
            reproduction_threshold=0.66,
        )
        shock_omnivore = replace(
            genomes["omnivore"],
            attack_power=1.0,
            meat_efficiency=1.32,
            food_efficiency=1.08,
            water_efficiency=1.4,
            defense_rating=1.16,
            plant_bias=0.94,
            carrion_bias=1.3,
            live_prey_bias=0.94,
            reproduction_threshold=0.82,
        )
        climate = replace(
            ClimateConfig(),
            season_length=12,
            dry_plain_penalty=1.0,
            dry_forest_penalty=0.34,
            wet_plain_bonus=0.14,
            wet_forest_bonus=0.08,
            seasonal_hydration_shift=0.03,
        )
        world = self._fixture_world(
            seed=95,
            max_ticks=60,
            max_agents=180,
            climate=climate,
            width=24,
            height=16,
        )
        self._configure_uniform_arena(
            world,
            terrain="plain",
            food=0.06,
            vegetation=0.1,
            shelter=0.03,
            recovery_debt=0.28,
            fertility=0.26,
            moisture=0.34,
            heat=0.46,
        )
        self._assert_fixture_profile(world, shock_herbivore, role="herbivore", meat_mode="none")
        self._assert_fixture_profile(world, shock_omnivore, role="omnivore", meat_mode="mixed")
        self._assert_fixture_profile(world, genomes["hunter"], role="carnivore", meat_mode="hunter")
        self._assert_fixture_profile(world, genomes["scavenger"], role="carnivore", meat_mode="scavenger")

        mid_x = world.config.width // 2
        mid_y = world.config.height // 2
        for y in range(world.config.height):
            for x in (mid_x - 2, mid_x - 1, mid_x, mid_x + 1):
                self._set_water_tile(world, x, y, heat=0.28)

        for y, row in enumerate(world.grid):
            for x, tile in enumerate(row):
                if tile.terrain == "water":
                    continue
                if x < 6:
                    tile.terrain = "plain"
                    tile.food = 0.48
                    tile.vegetation = 0.36
                    tile.shelter = 0.01
                    tile.recovery_debt = 0.24
                    tile.fertility = 0.46
                    tile.moisture = 0.32
                    tile.heat = 0.48
                elif mid_x + 2 <= x <= mid_x + 5:
                    tile.terrain = "forest"
                    tile.food = 0.28
                    tile.vegetation = 0.34
                    tile.shelter = 0.4
                    tile.recovery_debt = 0.04
                    tile.fertility = 0.56
                    tile.moisture = 0.8
                    tile.heat = 0.28
                elif x > 17:
                    tile.terrain = "rocky"
                    tile.food = 0.02
                    tile.vegetation = 0.04
                    tile.shelter = 0.01
                    tile.recovery_debt = 0.36
                    tile.fertility = 0.08
                    tile.moisture = 0.2
                    tile.heat = 0.54

        for y in (5, 9):
            for x in (mid_x + 2, mid_x + 3):
                world._deposit_carcass(
                    world.grid[y][x],
                    x=x,
                    y=y,
                    energy=1.2,
                    source_species=555,
                    source_agent_id=None,
                    cause="fixture",
                    killer_id=None,
                )

        initial_counts = {101: 4, 202: 4, 303: 4, 404: 4}
        for offset in range(4):
            self._add_agent(
                world,
                genome=shock_herbivore,
                x=1 + (offset % 2),
                y=2 + (offset // 2),
                lineage_id=101,
            )
        for offset in range(4):
            self._add_agent(
                world,
                genome=genomes["hunter"],
                x=19 + (offset % 2),
                y=2 + (offset // 2),
                lineage_id=202,
            )
        for offset in range(4):
            self._add_agent(
                world,
                genome=genomes["scavenger"],
                x=mid_x + 2 + (offset % 2),
                y=mid_y + 2 + (offset // 2),
                lineage_id=303,
            )
        for offset in range(4):
            self._add_agent(
                world,
                genome=shock_omnivore,
                x=mid_x + 4 + (offset % 2),
                y=mid_y - 1 + (offset // 2),
                lineage_id=404,
                energy_ratio=1.0,
                hydration_ratio=1.0,
                age=4,
            )
        world.current_species_map = {
            agent.agent_id: agent.lineage_id for agent in world.alive_agents()
        }
        world.agent_last_species_map = world.current_species_map.copy()
        self._refresh_fixture_world(world)
        result = world.run()
        population, _diet_shares, _kills = self._lineage_stats(result, initial_counts)

        omnivore_survival_rate = population[404]["alive"] / max(population[404]["total"], 1)
        specialist_survival_rate = max(
            population[101]["alive"] / max(population[101]["total"], 1),
            population[202]["alive"] / max(population[202]["total"], 1),
            population[303]["alive"] / max(population[303]["total"], 1),
        )
        specialist_births = max(
            population[101]["births"],
            population[202]["births"],
            population[303]["births"],
        )
        self.assertGreaterEqual(omnivore_survival_rate, specialist_survival_rate)
        self.assertGreater(specialist_births, population[404]["births"])

    def test_fixture_low_productivity_cascade_hurts_animal_specialists_more_than_herbivores(self) -> None:
        genomes = self._archetype_genomes()
        plant_omnivore = replace(
            genomes["omnivore"],
            food_efficiency=1.2,
            plant_bias=1.12,
            carrion_bias=0.88,
            live_prey_bias=0.7,
            reproduction_threshold=0.72,
        )
        resources = replace(
            ResourceRegrowthConfig(),
            plain_food_rate=0.006,
            forest_food_rate=0.009,
            wetland_food_rate=0.008,
            rocky_food_rate=0.004,
            vegetation_regrowth_rate=0.02,
        )
        world = self._fixture_world(
            seed=96,
            max_ticks=90,
            max_agents=150,
            resources=resources,
            width=24,
            height=16,
        )
        self._configure_uniform_arena(
            world,
            terrain="forest",
            food=0.34,
            vegetation=0.38,
            shelter=0.22,
            recovery_debt=0.3,
            fertility=0.38,
            moisture=0.66,
            heat=0.4,
        )
        self._assert_fixture_profile(world, genomes["herbivore"], role="herbivore", meat_mode="none")
        self._assert_fixture_profile(world, plant_omnivore, role="omnivore", meat_mode="mixed")
        self._assert_fixture_profile(world, genomes["hunter"], role="carnivore", meat_mode="hunter")
        self._assert_fixture_profile(world, genomes["scavenger"], role="carnivore", meat_mode="scavenger")

        mid = world.config.width // 2
        for y in range(world.config.height):
            for x in (mid - 1, mid):
                self._set_water_tile(world, x, y, heat=0.3)
        for y in range(5, 11):
            for x in (2, 3):
                self._set_water_tile(world, x, y, heat=0.3)

        initial_counts = {101: 6, 202: 4, 303: 4, 404: 4}
        for offset in range(6):
            self._add_agent(
                world,
                genome=genomes["herbivore"],
                x=4 + (offset % 3),
                y=2 + (offset // 3),
                lineage_id=101,
            )
        for offset in range(4):
            self._add_agent(
                world,
                genome=plant_omnivore,
                x=5 + (offset % 2),
                y=12 + (offset // 2),
                lineage_id=404,
            )
        for offset in range(4):
            self._add_agent(
                world,
                genome=genomes["hunter"],
                x=world.config.width - 4 + (offset % 2),
                y=2 + (offset // 2),
                lineage_id=202,
            )
        for offset in range(4):
            self._add_agent(
                world,
                genome=genomes["scavenger"],
                x=world.config.width - 4 + (offset % 2),
                y=8 + (offset // 2),
                lineage_id=303,
            )
        world.current_species_map = {
            agent.agent_id: agent.lineage_id for agent in world.alive_agents()
        }
        world.agent_last_species_map = world.current_species_map.copy()
        self._refresh_fixture_world(world)
        result = world.run()
        population, _diet_shares, _kills = self._lineage_stats(result, initial_counts)

        self.assertGreater(population[101]["alive"], population[202]["alive"])
        self.assertGreater(population[101]["alive"], population[303]["alive"])
        self.assertGreater(population[101]["births"], population[202]["births"])
        self.assertGreater(population[101]["births"], population[303]["births"])

    def test_production_readiness_mixed_world_sweep(self) -> None:
        carnivore_survival_seeds = 0
        hunter_survival_seeds = 0
        scavenger_survival_seeds = 0
        carnivore_shares: list[float] = []
        herbivore_shares: list[float] = []

        for seed in range(1, 21):
            result = SimulationWorld(WorldConfig(seed=seed, max_ticks=200)).run()
            final_frame = result.viewer["frames"][-1]
            species_metrics = final_frame["species_metrics"]
            if result.summary["trophic_role_counts_at_end"]["carnivore"] > 0:
                carnivore_survival_seeds += 1
                self.assertTrue(result.summary["top_carnivore_species"])
            if result.summary["top_hunter_species"]:
                hunter_survival_seeds += 1
            if result.summary["top_scavenger_species"]:
                scavenger_survival_seeds += 1

            for metrics in species_metrics.values():
                if metrics["alive_count"] <= 0:
                    continue
                if metrics["trophic_role_occupancy"]["carnivore"] > 0:
                    carnivore_shares.append(float(metrics["realized_animal_share"]))
                if metrics["trophic_role_occupancy"]["herbivore"] > 0:
                    herbivore_shares.append(float(metrics["realized_animal_share"]))

        self.assertGreaterEqual(carnivore_survival_seeds, 6)
        self.assertTrue(carnivore_shares)
        self.assertTrue(herbivore_shares)
        self.assertGreaterEqual(statistics.median(carnivore_shares), 0.75)
        self.assertLessEqual(statistics.median(herbivore_shares), 0.05)
        self.assertGreaterEqual(hunter_survival_seeds, 1)
        self.assertGreaterEqual(scavenger_survival_seeds, 1)


if __name__ == "__main__":
    unittest.main()
