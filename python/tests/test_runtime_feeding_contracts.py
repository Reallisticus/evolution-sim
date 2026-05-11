from __future__ import annotations

from python.tests.runtime_test_helpers import *


class RuntimeFeedingContractTests(RuntimeContractTestHelpers):
    def test_mixed_carcass_drive_has_omnivore_floor(self) -> None:
        world = SimulationWorld(self._ready_reproduction_config())
        genome = Genome(
            max_energy=1.0,
            max_hydration=1.0,
            max_health=1.0,
            move_cost=0.04,
            food_efficiency=1.0,
            water_efficiency=1.0,
            attack_power=0.8,
            attack_cost_multiplier=1.0,
            defense_rating=0.8,
            meat_efficiency=0.9,
            healing_efficiency=1.0,
            plant_bias=1.0,
            carrion_bias=0.75,
            live_prey_bias=0.75,
            forest_affinity=1.0,
            plain_affinity=1.0,
            wetland_affinity=1.0,
            rocky_affinity=1.0,
            heat_tolerance=1.0,
            reproduction_threshold=0.7,
            mutation_scale=0.01,
        )
        profile = world._trophic_profile_for_genome(genome)
        raw_drive = profile.scavenger_drive + profile.hunter_drive * 0.24

        self.assertEqual(profile.meat_mode, "mixed")
        self.assertLess(raw_drive, 0.16)
        self.assertEqual(world._carcass_drive(profile), 0.16)

    def test_scavenger_can_resolve_fresh_kill_intake(self) -> None:
        world = SimulationWorld(self._ready_reproduction_config())
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._scavenger_genome(),
        )
        agent.energy = agent.genome.max_energy * 0.4
        tile = world.grid[agent.y][agent.x]
        tile.food = 0.0
        world._deposit_fresh_kill(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.4,
            source_species=None,
            source_agent_id=None,
            killer_id=None,
        )

        outcome = world._eat_action_outcome(agent)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["food_source"], "fresh_kill")
        self.assertGreater(outcome["gained_energy"], 0.0)

    def test_hunter_fresh_kill_intake_restores_hydration(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                carcasses=CarcassConfig(fresh_kill_hydration_fraction=0.2)
            )
        )
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._hunter_genome(),
        )
        agent.energy = agent.genome.max_energy * 0.4
        agent.hydration = agent.genome.max_hydration * 0.3
        tile = world.grid[agent.y][agent.x]
        world._deposit_fresh_kill(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.4,
            source_species=None,
            source_agent_id=None,
            killer_id=None,
        )

        hydration_before = agent.hydration
        outcome = world._consume_fresh_kill_outcome(agent)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["food_source"], "fresh_kill")
        self.assertGreater(agent.hydration, hydration_before)

    def test_hunter_fresh_kill_intake_uses_hunter_healing_multiplier(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                carcasses=CarcassConfig(
                    healing_fraction=0.1,
                    fresh_kill_hunter_healing_multiplier=2.0,
                )
            )
        )
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._hunter_genome(),
        )
        agent.energy = agent.genome.max_energy
        agent.health = agent.max_health * 0.5
        tile = world.grid[agent.y][agent.x]
        world._deposit_fresh_kill(
            tile,
            x=agent.x,
            y=agent.y,
            energy=0.4,
            source_species=None,
            source_agent_id=None,
            killer_id=None,
        )

        health_before = agent.health
        outcome = world._consume_fresh_kill_outcome(agent)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["food_source"], "fresh_kill")
        self.assertGreater(agent.health, health_before + 0.05)

    def test_mixed_carcass_intake_restores_hydration(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                carcasses=CarcassConfig(mixed_carcass_hydration_fraction=0.2)
            )
        )
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._mixed_genome(),
        )
        agent.energy = agent.genome.max_energy * 0.4
        agent.hydration = agent.genome.max_hydration * 0.3
        tile = world.grid[agent.y][agent.x]
        tile.carcass_deposits.append(
            CarcassDeposit(
                energy_remaining=0.4,
                freshness=1.0,
                source_species=None,
                source_agent_id=None,
                death_tick=0,
                cause="test",
            )
        )

        hydration_before = agent.hydration
        outcome = world._consume_carcass_outcome(agent)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["food_source"], "carcass")
        self.assertGreater(agent.hydration, hydration_before)

    def test_hunter_carcass_intake_restores_hydration_when_severely_thirsty(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                carcasses=CarcassConfig(
                    hunter_carcass_hydration_fraction=0.2,
                    hunter_carcass_hydration_max_ratio=0.5,
                )
            )
        )
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._hunter_genome(),
        )
        agent.energy = agent.genome.max_energy * 0.4
        agent.hydration = agent.genome.max_hydration * 0.3
        tile = world.grid[agent.y][agent.x]
        tile.carcass_deposits.append(
            CarcassDeposit(
                energy_remaining=0.4,
                freshness=1.0,
                source_species=None,
                source_agent_id=None,
                death_tick=0,
                cause="test",
            )
        )

        hydration_before = agent.hydration
        outcome = world._consume_carcass_outcome(agent)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["food_source"], "carcass")
        self.assertGreater(agent.hydration, hydration_before)

    def test_hunter_carcass_intake_does_not_hydrate_above_rescue_threshold(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                carcasses=CarcassConfig(
                    hunter_carcass_hydration_fraction=0.2,
                    hunter_carcass_hydration_max_ratio=0.5,
                )
            )
        )
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._hunter_genome(),
        )
        agent.energy = agent.genome.max_energy * 0.4
        agent.hydration = agent.genome.max_hydration * 0.6
        tile = world.grid[agent.y][agent.x]
        tile.carcass_deposits.append(
            CarcassDeposit(
                energy_remaining=0.4,
                freshness=1.0,
                source_species=None,
                source_agent_id=None,
                death_tick=0,
                cause="test",
            )
        )

        hydration_before = agent.hydration
        outcome = world._consume_carcass_outcome(agent)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["food_source"], "carcass")
        self.assertEqual(agent.hydration, hydration_before)

    def test_scavenger_carcass_intake_restores_hydration(self) -> None:
        world = SimulationWorld(self._ready_reproduction_config())
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._scavenger_genome(),
        )
        agent.energy = agent.genome.max_energy * 0.4
        agent.hydration = agent.genome.max_hydration * 0.3
        tile = world.grid[agent.y][agent.x]
        tile.carcass_deposits.append(
            CarcassDeposit(
                energy_remaining=0.4,
                freshness=1.0,
                source_species=None,
                source_agent_id=None,
                death_tick=0,
                cause="test",
            )
        )

        hydration_before = agent.hydration
        outcome = world._consume_carcass_outcome(agent)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["food_source"], "carcass")
        self.assertGreater(agent.hydration, hydration_before)

    def test_scavenger_carcass_intake_can_be_hydration_useful_only(self) -> None:
        world = SimulationWorld(self._ready_reproduction_config())
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._scavenger_genome(),
        )
        agent.energy = agent.genome.max_energy
        agent.health = agent.max_health
        agent.hydration = agent.genome.max_hydration * 0.45
        tile = world.grid[agent.y][agent.x]
        tile.carcass_deposits.append(
            CarcassDeposit(
                energy_remaining=0.4,
                freshness=1.0,
                source_species=None,
                source_agent_id=None,
                death_tick=0,
                cause="test",
            )
        )

        hydration_before = agent.hydration
        outcome = world._eat_action_outcome(agent)
        hydration_gain = agent.hydration - hydration_before

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["food_source"], "carcass")
        self.assertEqual(outcome["gained_energy"], 0.0)
        self.assertGreater(hydration_gain, 0.0)
        self.assertGreater(
            hydration_gain,
            outcome["consumed"] * world.config.carcasses.scavenger_hydration_fraction,
        )

    def test_scavenger_can_eat_adjacent_blocked_carcass(self) -> None:
        world = SimulationWorld(self._ready_reproduction_config())
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._scavenger_genome(),
        )
        agent.energy = agent.genome.max_energy * 0.34
        source_tile = world.grid[1][2]
        source_tile.occupant_id = 12345
        source_tile.carcass_deposits.append(
            CarcassDeposit(
                energy_remaining=0.4,
                freshness=1.0,
                source_species=None,
                source_agent_id=None,
                death_tick=0,
                cause="test",
            )
        )

        action_mask = build_action_mask(world._action_mask_context(agent))
        outcome = world._eat_action_outcome(agent)

        self.assertTrue(action_mask["eat"])
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["food_source"], "carcass")
        self.assertEqual((outcome["x"], outcome["y"]), (2, 1))
        self.assertGreater(outcome["gained_energy"], 0.0)

    def test_hunter_can_drink_from_adjacent_blocked_wetland_when_near_death(self) -> None:
        world = SimulationWorld(self._ready_reproduction_config())
        x, y = next(
            (
                (x, y)
                for y in range(world.config.height)
                for x in range(world.config.width - 1)
                if world._water_access_reason(x, y) == "none"
                and world._water_access_reason(x + 1, y) == "none"
            )
        )
        agent = self._place_ready_agent(
            world,
            x=x,
            y=y,
            genome=self._hunter_genome(),
        )
        blocker = self._place_ready_agent(world, x=x + 1, y=y, lineage_id=2)
        world.grid[blocker.y][blocker.x].terrain = "wetland"
        agent.hydration = agent.genome.max_hydration * 0.03

        action_mask = build_action_mask(world._action_mask_context(agent))
        outcome = world._drink_action_outcome(agent)

        self.assertTrue(action_mask["drink"])
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["water_access_reason"], "wetland")
        self.assertEqual((outcome["source_x"], outcome["source_y"]), (x + 1, y))
        self.assertGreater(outcome["gained_hydration"], 0.0)

    def test_hunter_cannot_share_blocked_wetland_above_critical_hydration(self) -> None:
        world = SimulationWorld(self._ready_reproduction_config())
        x, y = next(
            (
                (x, y)
                for y in range(world.config.height)
                for x in range(world.config.width - 1)
                if world._water_access_reason(x, y) == "none"
                and world._water_access_reason(x + 1, y) == "none"
            )
        )
        agent = self._place_ready_agent(
            world,
            x=x,
            y=y,
            genome=self._hunter_genome(),
        )
        blocker = self._place_ready_agent(world, x=x + 1, y=y, lineage_id=2)
        world.grid[blocker.y][blocker.x].terrain = "wetland"
        agent.hydration = agent.genome.max_hydration * 0.5

        action_mask = build_action_mask(world._action_mask_context(agent))
        outcome = world._drink_action_outcome(agent)

        self.assertFalse(action_mask["drink"])
        self.assertIsNone(outcome)

    def test_carcass_opportunity_reports_policy_blockers(self) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                initial_agents=0,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
            )
        )
        agent = self._place_ready_agent(
            world,
            x=1,
            y=2,
            genome=self._scavenger_genome(),
        )
        blocker = self._place_ready_agent(world, x=3, y=2, lineage_id=2)
        world.grid[blocker.y][blocker.x].carcass_deposits.append(
            CarcassDeposit(
                energy_remaining=0.4,
                freshness=1.0,
                source_species=None,
                source_agent_id=None,
                death_tick=0,
                cause="test",
            )
        )

        reachability = world._animal_resource_reachability_by_meat_mode([agent])
        scavenger_counts = reachability["scavenger"]

        self.assertEqual(scavenger_counts["carcass_reachable_agents"], 1)
        self.assertEqual(scavenger_counts["carcass_policy_actionable_agents"], 0)
        self.assertEqual(
            scavenger_counts["carcass_reachable_policy_blocked_agents"],
            1,
        )
        self.assertEqual(
            scavenger_counts["carcass_policy_blocked_by_occupant_agents"],
            1,
        )
        self.assertEqual(
            scavenger_counts["animal_resource_policy_blocked_by_occupant_agents"],
            1,
        )

    def test_bfs_reachable_carcass_can_be_blocked_by_direct_water_step(self) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                initial_agents=0,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
            )
        )
        agent = self._place_ready_agent(
            world,
            x=2,
            y=2,
            genome=self._scavenger_genome(),
        )
        world.grid[2][3].terrain = "water"
        world.grid[2][4].carcass_deposits.append(
            CarcassDeposit(
                energy_remaining=0.4,
                freshness=1.0,
                source_species=None,
                source_agent_id=None,
                death_tick=0,
                cause="test",
            )
        )

        reachability = world._animal_resource_reachability_by_meat_mode([agent])
        scavenger_counts = reachability["scavenger"]

        self.assertEqual(scavenger_counts["carcass_reachable_agents"], 1)
        self.assertEqual(scavenger_counts["carcass_policy_actionable_agents"], 0)
        self.assertEqual(
            scavenger_counts["carcass_policy_blocked_by_water_agents"],
            1,
        )

    def test_carrion_navigation_routes_first_step_around_water(self) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                initial_agents=0,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
            )
        )
        agent = self._place_ready_agent(
            world,
            x=2,
            y=2,
            genome=self._scavenger_genome(),
        )
        world.grid[2][3].terrain = "water"
        world.grid[2][4].carcass_deposits.append(
            CarcassDeposit(
                energy_remaining=0.4,
                freshness=1.0,
                source_species=None,
                source_agent_id=None,
                death_tick=0,
                cause="test",
            )
        )

        navigation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )["navigation"]["carrion"]

        self.assertEqual(navigation["dx"], 0)
        self.assertEqual(navigation["dy"], -1)
        self.assertEqual(navigation["distance"], 4)
        self.assertGreater(navigation["strength"], 0.0)

    def test_resource_runtime_boundary_preserves_consumption_provenance(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        tile = world.grid[agent.y][agent.x]
        tile.carcass_deposits = [
            CarcassDeposit(
                energy_remaining=0.4,
                freshness=0.25,
                source_species=1,
                source_agent_id=101,
                death_tick=2,
                cause="old",
            ),
            CarcassDeposit(
                energy_remaining=0.3,
                freshness=0.95,
                source_species=2,
                source_agent_id=202,
                death_tick=3,
                cause="new",
            ),
        ]
        tile.fresh_kill_deposits = [
            FreshKillDeposit(
                energy_remaining=0.2,
                source_species=3,
                source_agent_id=303,
                death_tick=4,
                killer_id=900,
            ),
            FreshKillDeposit(
                energy_remaining=0.25,
                source_species=4,
                source_agent_id=404,
                death_tick=8,
                killer_id=901,
            ),
        ]

        carcass_info = runtime_resources.consume_carcass_from_tile(
            tile,
            0.2,
            max_tile_deposits=world.config.carcasses.max_tile_deposits,
            freshness_merge_bucket=world.config.carcasses.freshness_merge_bucket,
        )
        fresh_kill_info = runtime_resources.consume_fresh_kill_from_tile(
            tile,
            0.15,
            max_tile_deposits=world.config.carcasses.max_tile_deposits,
        )

        self.assertEqual(carcass_info["deposit_breakdown"][0]["source_agent_id"], 202)
        self.assertEqual(carcass_info["source_breakdown"][0]["source_species"], 2)
        self.assertEqual(fresh_kill_info["deposit_breakdown"][0]["source_agent_id"], 404)
        self.assertEqual(fresh_kill_info["source_breakdown"][0]["killer_id"], 901)

    def test_feeding_telemetry_uses_runtime_accounting_for_all_food_sources(self) -> None:
        observed_sources: list[str] = []
        observed_contexts: list[runtime_feeding.FeedingContext] = []
        original_record_feeding_event = runtime_feeding.record_feeding_event

        def record_spy(*args: object, **kwargs: object) -> None:
            observed_sources.append(str(args[2]))
            observed_contexts.append(kwargs["context"])
            original_record_feeding_event(*args, **kwargs)

        with patch(
            "evolution_sim.env.runtime.feeding.record_feeding_event",
            side_effect=record_spy,
        ):
            plant_world = SimulationWorld(self._ready_reproduction_config())
            plant_agent = self._place_ready_agent(
                plant_world,
                x=1,
                y=1,
                genome=self._mixed_genome(),
            )
            plant_agent.energy = plant_agent.genome.max_energy * 0.4
            plant_tile = plant_world.grid[plant_agent.y][plant_agent.x]
            plant_tile.food = 0.5
            plant_outcome = plant_world._consume_plant_outcome(plant_agent)

            fresh_world = SimulationWorld(self._ready_reproduction_config())
            fresh_agent = self._place_ready_agent(
                fresh_world,
                x=1,
                y=1,
                genome=self._hunter_genome(),
            )
            fresh_agent.energy = fresh_agent.genome.max_energy * 0.4
            fresh_tile = fresh_world.grid[fresh_agent.y][fresh_agent.x]
            fresh_world._deposit_fresh_kill(
                fresh_tile,
                x=fresh_agent.x,
                y=fresh_agent.y,
                energy=0.4,
                source_species=None,
                source_agent_id=None,
                killer_id=None,
            )
            fresh_outcome = fresh_world._consume_fresh_kill_outcome(fresh_agent)

            carcass_world = SimulationWorld(self._ready_reproduction_config())
            carcass_agent = self._place_ready_agent(
                carcass_world,
                x=1,
                y=1,
                genome=self._scavenger_genome(),
            )
            carcass_agent.energy = carcass_agent.genome.max_energy * 0.4
            carcass_tile = carcass_world.grid[carcass_agent.y][carcass_agent.x]
            carcass_world._deposit_carcass(
                carcass_tile,
                x=carcass_agent.x,
                y=carcass_agent.y,
                energy=0.4,
                source_species=None,
                source_agent_id=None,
                cause="test",
                killer_id=None,
            )
            carcass_outcome = carcass_world._consume_carcass_outcome(carcass_agent)

        self.assertIsNotNone(plant_outcome)
        self.assertIsNotNone(fresh_outcome)
        self.assertIsNotNone(carcass_outcome)
        self.assertEqual(observed_sources, ["plant", "fresh_kill", "carcass"])
        self.assertTrue(
            all(
                isinstance(context, runtime_feeding.FeedingContext)
                for context in observed_contexts
            )
        )
        for world, food_source in (
            (plant_world, "plant"),
            (fresh_world, "fresh_kill"),
            (carcass_world, "carcass"),
        ):
            self.assertEqual(world.tick_feeding_events[-1]["food_source"], food_source)
            self.assertEqual(world.run_diet_totals[f"{food_source}_events"], 1.0)
            self.assertGreater(world.run_diet_totals[f"{food_source}_energy"], 0.0)
        self.assertGreater(
            plant_world.run_resource_pressure_totals["plant_energy_removed"],
            0.0,
        )

    def test_feeding_context_controls_telemetry_authority(self) -> None:
        world = SimulationWorld(self._ready_reproduction_config())
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._mixed_genome(),
        )
        profile = self._test_trophic_profile("none")
        context = runtime_feeding.FeedingContext(
            config=world.config,
            emit=world._emit,
            species_id_for_agent=lambda agent_id: 77,
            trophic_profile=lambda checked_agent: profile,
            matched_diet_ratio=lambda checked_agent, checked_profile: 0.42,
            agent_reachable_animal_resources=lambda checked_agent, **kwargs: {},
            scavenger_carcass_hydration_fraction=lambda checked_agent: 0.0,
            hydration_ratio=lambda checked_agent: 1.0,
            clamp01=lambda value: max(0.0, min(1.0, value)),
            fresh_kill_tile_summary_for_position=lambda x, y: {},
            carcass_tile_summary_for_position=lambda x, y: {},
        )

        with (
            patch.object(
                world,
                "_species_id_for_agent",
                side_effect=AssertionError("feeding context was bypassed"),
            ),
            patch.object(
                world,
                "_matched_diet_ratio",
                side_effect=AssertionError("feeding context was bypassed"),
            ),
        ):
            runtime_feeding.record_feeding_event(
                world,
                agent,
                "plant",
                consumed=0.2,
                gained_energy=0.15,
                profile=profile,
                energy_before=0.3,
                energy_after=0.45,
                context=context,
            )

        event = world.tick_feeding_events[-1]
        self.assertEqual(event["species_id"], 77)
        self.assertEqual(event["matched_diet_ratio"], 0.42)
        self.assertEqual(world.run_diet_totals["plant_events"], 1)

    def test_animal_resource_opportunity_accounting_uses_explicit_inputs(self) -> None:
        run_counts = {
            "hunter": runtime_feeding.empty_animal_resource_opportunity_counts(),
            "scavenger": runtime_feeding.empty_animal_resource_opportunity_counts(),
        }
        tick_consumption = {
            "hunter": runtime_feeding.empty_animal_resource_consumption_counts(),
            "scavenger": runtime_feeding.empty_animal_resource_consumption_counts(),
        }
        tick_consumption["hunter"]["fresh_kill_consumption_events"] = 1
        tick_consumption["hunter"]["fresh_kill_energy_consumed"] = 0.3
        tick_consumption["hunter"]["fresh_kill_gained_energy"] = 0.2
        tick_consumption["hunter"]["animal_resource_consumption_events"] = 1
        tick_consumption["hunter"]["animal_resource_energy_consumed"] = 0.3
        tick_consumption["hunter"]["animal_resource_gained_energy"] = 0.2
        reachability = {
            "hunter": {
                **runtime_feeding.empty_animal_resource_reachability_tick_counts(),
                "animal_resource_reachable_agents": 2,
                "animal_resource_policy_actionable_agents": 1,
                "animal_resource_reachable_policy_blocked_agents": 1,
                "animal_resource_policy_blocked_by_hazard_agents": 1,
                "fresh_kill_reachable_agents": 2,
                "fresh_kill_policy_actionable_agents": 1,
                "fresh_kill_reachable_policy_blocked_agents": 1,
                "fresh_kill_policy_blocked_by_hazard_agents": 1,
            },
            "scavenger": runtime_feeding.empty_animal_resource_reachability_tick_counts(),
        }

        runtime_feeding_opportunity.record_animal_resource_opportunity_tick_from_inputs(
            run_counts,
            meat_mode_counts={"hunter": 3, "scavenger": 0},
            tick_consumption_by_meat_mode=tick_consumption,
            reachability_by_meat_mode=reachability,
            resource_presence={
                "animal_resource": True,
                "fresh_kill": True,
                "carcass": False,
            },
        )

        hunter = run_counts["hunter"]
        self.assertEqual(hunter["alive_ticks"], 1)
        self.assertEqual(hunter["alive_agent_ticks"], 3)
        self.assertEqual(hunter["fresh_kill_consumption_events"], 1)
        self.assertEqual(hunter["fresh_kill_reachable_agent_ticks"], 2)
        self.assertEqual(hunter["fresh_kill_policy_actionable_agent_ticks"], 1)
        self.assertEqual(
            hunter["fresh_kill_policy_blocked_by_hazard_agent_ticks"],
            1,
        )
        self.assertEqual(run_counts["scavenger"]["alive_ticks"], 0)

    def test_opportunity_recording_uses_decision_time_resource_presence(self) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                initial_agents=0,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
            )
        )
        world.tick_animal_resource_consumption_by_meat_mode = (
            runtime_feeding.empty_grouped_animal_resource_consumption_counts(
                MEAT_MODE_CODES
            )
        )
        world.run_animal_resource_opportunity_by_meat_mode = (
            runtime_feeding.empty_grouped_animal_resource_opportunity_counts(
                MEAT_MODE_CODES
            )
        )
        world.grid[1][1].fresh_kill_deposits.append(
            FreshKillDeposit(
                energy_remaining=0.4,
                source_species=None,
                source_agent_id=None,
                killer_id=None,
                death_tick=0,
            )
        )
        self.assertTrue(
            runtime_feeding.animal_resource_presence_this_tick(world)[
                "animal_resource"
            ]
        )

        runtime_feeding.record_animal_resource_opportunity_tick(
            world,
            {"hunter": 1, "mixed": 0, "none": 0, "scavenger": 0},
            {
                mode: runtime_feeding.empty_animal_resource_reachability_tick_counts()
                for mode in MEAT_MODE_CODES
            },
            resource_presence={
                "animal_resource": False,
                "fresh_kill": False,
                "carcass": False,
            },
        )

        hunter = world.run_animal_resource_opportunity_by_meat_mode["hunter"]
        self.assertEqual(hunter["alive_ticks"], 1)
        self.assertEqual(hunter["animal_resource_present_ticks"], 0)
        self.assertEqual(hunter["animal_resource_absent_ticks"], 1)
        self.assertEqual(hunter["fresh_kill_present_ticks"], 0)

    def test_same_tick_generated_consumption_still_counts_consumed_tick(self) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                initial_agents=0,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
            )
        )
        world.tick_animal_resource_consumption_by_meat_mode = (
            runtime_feeding.empty_grouped_animal_resource_consumption_counts(
                MEAT_MODE_CODES
            )
        )
        world.run_animal_resource_opportunity_by_meat_mode = (
            runtime_feeding.empty_grouped_animal_resource_opportunity_counts(
                MEAT_MODE_CODES
            )
        )
        world.tick_animal_resource_consumption_by_meat_mode["hunter"][
            "fresh_kill_consumption_events"
        ] = 1
        world.tick_animal_resource_consumption_by_meat_mode["hunter"][
            "animal_resource_consumption_events"
        ] = 1

        runtime_feeding.record_animal_resource_opportunity_tick(
            world,
            {"hunter": 1, "mixed": 0, "none": 0, "scavenger": 0},
            {
                mode: runtime_feeding.empty_animal_resource_reachability_tick_counts()
                for mode in MEAT_MODE_CODES
            },
            resource_presence={
                "animal_resource": False,
                "fresh_kill": False,
                "carcass": False,
            },
        )

        hunter = world.run_animal_resource_opportunity_by_meat_mode["hunter"]
        self.assertEqual(hunter["animal_resource_absent_ticks"], 1)
        self.assertEqual(hunter["animal_resource_present_ticks"], 0)
        self.assertEqual(hunter["animal_resource_consumed_ticks"], 1)
        self.assertEqual(hunter["fresh_kill_present_ticks"], 0)
        self.assertEqual(hunter["fresh_kill_consumed_ticks"], 1)

    def test_tick_opportunity_accounting_reuses_decision_time_presence(self) -> None:
        no_presence = {
            "animal_resource": False,
            "fresh_kill": False,
            "carcass": False,
        }
        later_presence = {
            "animal_resource": True,
            "fresh_kill": True,
            "carcass": False,
        }
        with patch(
            "evolution_sim.env.runtime.feeding.animal_resource_presence_this_tick",
            side_effect=[no_presence, later_presence],
        ) as presence:
            world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
            result = world.run(mode=RunMode.SUMMARY_ONLY)

        self.assertEqual(presence.call_count, 1)
        total_present_ticks = sum(
            counts["animal_resource_present_ticks"]
            for counts in result.summary[
                "animal_resource_opportunity_by_meat_mode_end"
            ].values()
        )
        self.assertEqual(total_present_ticks, 0)

    def test_summary_only_never_invokes_full_replay_paths(self) -> None:
        with patch(
            "evolution_sim.env.runtime.collectors.apply_replay_taxonomy",
            side_effect=AssertionError("taxonomy should not run in summary-only mode"),
        ), patch.object(
            SimulationWorld,
            "_capture_frame",
            side_effect=AssertionError("summary-only should not capture frames"),
        ), patch(
            "evolution_sim.env.runtime.frames.capture_frame",
            side_effect=AssertionError("summary-only should not build frame payloads"),
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

    def test_summary_only_opportunity_counters_do_not_build_replay_payloads(self) -> None:
        with patch(
            "evolution_sim.env.runtime.collectors.apply_replay_taxonomy",
            side_effect=AssertionError("taxonomy should not run for opportunity counters"),
        ), patch.object(
            SimulationWorld,
            "_capture_frame",
            side_effect=AssertionError("opportunity counters should not capture frames"),
        ), patch(
            "evolution_sim.env.runtime.frames.capture_frame",
            side_effect=AssertionError("opportunity counters should not build frames"),
        ), patch.object(
            SimulationWorld,
            "_build_viewer_payload",
            side_effect=AssertionError("opportunity counters should not build viewer payloads"),
        ):
            world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
            agent = world.alive_agents()[0]
            world._deposit_carcass(
                world.grid[agent.y][agent.x],
                x=agent.x,
                y=agent.y,
                energy=0.4,
                source_species=None,
                source_agent_id=None,
                cause="test",
                killer_id=None,
            )

            result = world.run(mode=RunMode.SUMMARY_ONLY)

        opportunity = result.summary["animal_resource_opportunity_by_meat_mode_end"]
        self.assertIsNone(result.viewer)
        self.assertIsNone(result.events)
        self.assertEqual(world.viewer_frames, [])
        self.assertGreater(
            sum(
                counts["animal_resource_present_ticks"]
                for counts in opportunity.values()
            ),
            0,
        )

    def test_summary_only_does_not_retain_replay_bookkeeping(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=20))

        result = world.run(mode=RunMode.SUMMARY_ONLY)

        self.assertIsNone(result.events)
        self.assertIsNone(result.viewer)
        self.assertEqual(world.events, [])
        self.assertEqual(world.viewer_frames, [])
        self.assertTrue(world.record_events)
        self.assertTrue(world.record_tick_details)
