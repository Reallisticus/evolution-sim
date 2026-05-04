from __future__ import annotations

from python.tests.runtime_test_helpers import *


class RuntimeSummaryContractTests(RuntimeContractTestHelpers):
    def test_collectors_use_explicit_context(self) -> None:
        @dataclass(frozen=True, slots=True)
        class EventRecord:
            payload: dict[str, object]

            def to_dict(self) -> dict[str, object]:
                return self.payload

        calls: list[tuple[str, object]] = []

        def capture_frame(*, births_this_tick: int, deaths_this_tick: int) -> None:
            calls.append(
                (
                    "capture_frame",
                    {
                        "births_this_tick": births_this_tick,
                        "deaths_this_tick": deaths_this_tick,
                    },
                )
            )

        def build_summary(*, mode: RunMode) -> dict[str, object]:
            calls.append(("build_summary", mode))
            return {
                "run_id": "context-only",
                "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                "mode": mode.value,
                "full_replay_only": True,
            }

        def build_viewer_payload() -> dict[str, object]:
            calls.append(("build_viewer_payload", True))
            return {"frames": []}

        def refresh_population_snapshots(*, include_species: bool = True) -> object:
            calls.append(("refresh_population_snapshots", include_species))
            return object()

        context = runtime_collectors.CollectorContext(
            config=WorldConfig(seed=123),
            events=[EventRecord({"type": "test"})],
            capture_frame=capture_frame,
            build_summary=build_summary,
            build_viewer_payload=build_viewer_payload,
            refresh_population_snapshots=refresh_population_snapshots,
        )

        full_replay = runtime_collectors.FullReplayCollector()
        full_replay.on_tick(context, births_this_tick=2, deaths_this_tick=1)
        with patch(
            "evolution_sim.env.runtime.collectors.apply_replay_taxonomy",
            side_effect=lambda *, config, summary, events, viewer: (
                {**summary, "taxonomy_seed": config.seed},
                viewer,
            ),
        ) as apply_taxonomy:
            summary, events, viewer = full_replay.finalize(context)

        self.assertEqual(
            calls[:3],
            [
                (
                    "capture_frame",
                    {"births_this_tick": 2, "deaths_this_tick": 1},
                ),
                ("build_summary", RunMode.FULL_REPLAY),
                ("build_viewer_payload", True),
            ],
        )
        self.assertEqual(summary["taxonomy_seed"], 123)
        self.assertEqual(events, [{"type": "test"}])
        self.assertEqual(viewer, {"frames": []})
        apply_taxonomy.assert_called_once()

        summary_only = runtime_collectors.SummaryCollector()
        summary_only.on_tick(context, births_this_tick=0, deaths_this_tick=0)
        filtered_summary, event_payloads, summary_viewer = summary_only.finalize(context)

        self.assertEqual(
            calls[-3:],
            [
                ("refresh_population_snapshots", False),
                ("refresh_population_snapshots", False),
                ("build_summary", RunMode.SUMMARY_ONLY),
            ],
        )
        self.assertEqual(
            filtered_summary,
            {
                "run_id": "context-only",
                "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            },
        )
        self.assertIsNone(event_payloads)
        self.assertIsNone(summary_viewer)

    def test_trophic_lifecycle_summary_uses_explicit_context(self) -> None:
        genome = self._hunter_genome()

        def agent(
            *,
            agent_id: int,
            parent_id: int | None,
            birth_tick: int,
            death_tick: int | None,
            alive: bool,
        ) -> Agent:
            return Agent(
                agent_id=agent_id,
                parent_id=parent_id,
                lineage_id=agent_id,
                birth_tick=birth_tick,
                death_tick=death_tick,
                x=agent_id,
                y=0,
                energy=1.0,
                hydration=1.0,
                health=1.0,
                max_health=1.0,
                injury_load=0.0,
                age=0,
                alive=alive,
                last_reproduction_tick=-10_000,
                last_damage_source="none",
                recent_plant_energy=0.0,
                recent_fresh_kill_energy=0.0,
                recent_carcass_energy=0.0,
                genome_vector=genome_vector(genome),
                genome=genome,
            )

        agents = {
            1: agent(
                agent_id=1,
                parent_id=None,
                birth_tick=0,
                death_tick=5,
                alive=False,
            ),
            2: agent(
                agent_id=2,
                parent_id=None,
                birth_tick=0,
                death_tick=None,
                alive=True,
            ),
            3: agent(
                agent_id=3,
                parent_id=2,
                birth_tick=22,
                death_tick=28,
                alive=False,
            ),
        }
        roles = {1: "carnivore", 2: "carnivore", 3: "omnivore"}
        meat_modes = {1: "hunter", 2: "scavenger", 3: "mixed"}
        death_causes = {1: "starvation", 3: "predation"}
        context = runtime_lifecycle_summary.TrophicLifecycleSummaryContext(
            agents=agents,
            run_death_cause_counts={"starvation": 1, "predation": 1},
            run_death_causes_by_trophic_role={
                "carnivore": {"starvation": 1},
                "omnivore": {"predation": 1},
            },
            run_death_causes_by_meat_mode={
                "hunter": {"starvation": 1},
                "mixed": {"predation": 1},
            },
            trophic_role=lambda candidate: roles[candidate.agent_id],
            meat_mode=lambda candidate: meat_modes[candidate.agent_id],
            death_cause=lambda candidate: death_causes[candidate.agent_id],
        )

        summary = runtime_lifecycle_summary.build_trophic_lifecycle_summary(
            context=context,
            ticks_executed=40,
            trophic_role_codes=TROPHIC_ROLE_CODES,
            meat_mode_codes=MEAT_MODE_CODES,
        )

        self.assertEqual(summary["initial_trophic_role_counts"]["carnivore"], 2)
        self.assertEqual(summary["initial_meat_mode_counts"]["hunter"], 1)
        self.assertEqual(summary["initial_meat_mode_counts"]["scavenger"], 1)
        self.assertEqual(summary["births_by_parent_meat_mode"]["scavenger"], 1)
        self.assertEqual(summary["births_by_child_trophic_role"]["omnivore"], 1)
        self.assertEqual(summary["births_by_child_meat_mode"]["mixed"], 1)
        self.assertEqual(summary["deaths_by_meat_mode"]["hunter"], 1)
        self.assertEqual(summary["deaths_by_meat_mode"]["mixed"], 1)
        self.assertEqual(
            summary["death_causes"],
            {"predation": 1, "starvation": 1},
        )
        self.assertEqual(
            summary["death_causes_by_meat_mode"]["mixed"],
            {"predation": 1},
        )

        persistence = summary["meat_mode_persistence"]
        self.assertEqual(
            persistence["last_alive_tick_by_meat_mode"],
            {"none": None, "scavenger": 39, "hunter": 4, "mixed": 27},
        )
        self.assertEqual(
            persistence["births_by_parent_meat_mode_by_tick_band"]["late"][
                "scavenger"
            ],
            1,
        )
        self.assertEqual(
            persistence["deaths_by_meat_mode_by_tick_band"]["early"]["hunter"],
            1,
        )
        self.assertEqual(
            persistence["death_causes_by_meat_mode_by_tick_band"]["late"]["mixed"],
            {"predation": 1},
        )

        late_window = summary["late_window"]
        self.assertEqual(late_window["sample_ticks"], [30, 34, 39])
        self.assertEqual(late_window["presence_ticks_by_meat_mode"]["scavenger"], 3)

    def test_scavenger_reproduction_health_floor_uses_carrion_match(self) -> None:
        world = SimulationWorld(self._ready_reproduction_config())
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._scavenger_genome(),
        )
        agent.health = agent.max_health * (
            world.config.reproduction.scavenger_min_health_fraction + 0.02
        )
        agent.recent_carcass_energy = 1.0
        profile = world._trophic_profile(agent)

        blockers = world._biological_reproduction_block_reasons(agent, profile)

        self.assertNotIn("health", blockers)
        self.assertNotIn("matched_diet", blockers)

    def test_world_config_rejects_invalid_ranges_early(self) -> None:
        invalid_configs = (
            ("width", lambda: WorldConfig(width=0)),
            ("height", lambda: WorldConfig(height=0)),
            ("max_ticks", lambda: WorldConfig(max_ticks=0)),
            (
                "climate.season_length",
                lambda: WorldConfig(climate=ClimateConfig(season_length=0)),
            ),
            ("water_tile_ratio", lambda: WorldConfig(water_tile_ratio=-0.1)),
            ("water_tile_ratio", lambda: WorldConfig(water_tile_ratio=1.2)),
            (
                "hazards.exposure_damage_rate",
                lambda: WorldConfig(hazards=HazardConfig(exposure_damage_rate=-0.1)),
            ),
            (
                "combat.attack_energy_cost",
                lambda: WorldConfig(combat=CombatConfig(attack_energy_cost=-0.1)),
            ),
            (
                "combat.hunter_mode_attack_damage_multiplier",
                lambda: WorldConfig(
                    combat=CombatConfig(hunter_mode_attack_damage_multiplier=0.0)
                ),
            ),
            (
                "combat.hunter_wounded_prey_damage_bonus",
                lambda: WorldConfig(
                    combat=CombatConfig(hunter_wounded_prey_damage_bonus=-0.1)
                ),
            ),
            (
                "carcasses.fresh_kill_conversion_rate",
                lambda: WorldConfig(
                    carcasses=CarcassConfig(fresh_kill_conversion_rate=2.0)
                ),
            ),
            (
                "carcasses.fresh_kill_hydration_fraction",
                lambda: WorldConfig(
                    carcasses=CarcassConfig(fresh_kill_hydration_fraction=-0.1)
                ),
            ),
            (
                "carcasses.mixed_carcass_hydration_fraction",
                lambda: WorldConfig(
                    carcasses=CarcassConfig(mixed_carcass_hydration_fraction=-0.1)
                ),
            ),
            (
                "carcasses.hunter_carcass_hydration_fraction",
                lambda: WorldConfig(
                    carcasses=CarcassConfig(hunter_carcass_hydration_fraction=-0.1)
                ),
            ),
            (
                "carcasses.hunter_carcass_hydration_max_ratio",
                lambda: WorldConfig(
                    carcasses=CarcassConfig(hunter_carcass_hydration_max_ratio=1.1)
                ),
            ),
            (
                "carcasses.fresh_kill_hunter_healing_multiplier",
                lambda: WorldConfig(
                    carcasses=CarcassConfig(fresh_kill_hunter_healing_multiplier=0.0)
                ),
            ),
            (
                "carcasses.scavenger_healing_multiplier",
                lambda: WorldConfig(
                    carcasses=CarcassConfig(scavenger_healing_multiplier=0.0)
                ),
            ),
            (
                "carcasses.scavenger_hydration_fraction",
                lambda: WorldConfig(
                    carcasses=CarcassConfig(scavenger_hydration_fraction=-0.1)
                ),
            ),
            (
                "reproduction.energy_cost",
                lambda: WorldConfig(reproduction=ReproductionConfig(energy_cost=2.0)),
            ),
            (
                "reproduction.animal_mode_energy_requirement_multiplier",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        animal_mode_energy_requirement_multiplier=0.0,
                    )
                ),
            ),
            (
                "reproduction.animal_mode_reproduction_cost_multiplier",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        animal_mode_reproduction_cost_multiplier=0.0,
                    )
                ),
            ),
            (
                "reproduction.animal_mode_offspring_trait_stability",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        animal_mode_offspring_trait_stability=1.2,
                    )
                ),
            ),
            (
                "reproduction.scavenger_min_health_fraction",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        scavenger_min_health_fraction=1.2,
                    )
                ),
            ),
            (
                "reproduction.sexual_reproduction_enabled",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        sexual_reproduction_enabled=1,  # type: ignore[arg-type]
                    )
                ),
            ),
            (
                "reproduction.sexual_partner_radius",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(sexual_partner_radius=0)
                ),
            ),
            (
                "reproduction.sexual_parent_cost_multiplier",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        sexual_parent_cost_multiplier=0.0,
                    )
                ),
            ),
            (
                "reproduction.role_differentiation_threshold",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        role_differentiation_threshold=1.2,
                    )
                ),
            ),
            (
                "reproduction.xyz_expression_threshold",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        role_differentiation_threshold=0.8,
                        xyz_expression_threshold=0.7,
                    )
                ),
            ),
            (
                "reproduction.z_plasticity_threshold",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(z_plasticity_threshold=-0.1)
                ),
            ),
            (
                "reproduction.role_complementarity_bonus",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(role_complementarity_bonus=1.2)
                ),
            ),
            (
                "reproduction.z_z_pairing_penalty",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(z_z_pairing_penalty=1.2)
                ),
            ),
            (
                "reproduction.multi_offspring_enabled",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        multi_offspring_enabled=1,  # type: ignore[arg-type]
                    )
                ),
            ),
            (
                "reproduction.multi_offspring_threshold",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        multi_offspring_threshold=1.2,
                    )
                ),
            ),
            (
                "reproduction.multi_offspring_max_count",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        multi_offspring_max_count=0,
                    )
                ),
            ),
            (
                "reproduction.multi_offspring_max_count",
                lambda: WorldConfig(
                    reproduction=ReproductionConfig(
                        multi_offspring_enabled=True,
                        multi_offspring_max_count=1,
                    )
                ),
            ),
        )

        for expected_message, build_config in invalid_configs:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    build_config()

    def test_world_config_rejects_invalid_cross_field_relationships(self) -> None:
        invalid_configs = (
            ("initial_agents", lambda: WorldConfig(initial_agents=30, max_agents=20)),
            (
                "terrain tile ratios",
                lambda: WorldConfig(water_tile_ratio=0.8, forest_tile_ratio=0.3),
            ),
            (
                "estimated land tiles",
                lambda: WorldConfig(
                    width=4,
                    height=4,
                    initial_agents=1,
                    max_agents=1,
                    water_tile_ratio=1.0,
                    forest_tile_ratio=0.0,
                    wetland_tile_ratio=0.0,
                    rocky_tile_ratio=0.0,
                ),
            ),
            ("max_age", lambda: WorldConfig(max_age=20)),
        )

        for expected_message, build_config in invalid_configs:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    build_config()

    def test_simulation_world_validates_mutated_config_before_runtime_setup(self) -> None:
        config = WorldConfig(seed=7, max_ticks=1)
        config.climate.season_length = 0

        with self.assertRaisesRegex(ValueError, "climate.season_length"):
            SimulationWorld(config)

    def test_simulation_world_run_is_one_shot(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=2))

        result = world.run()

        self.assertEqual(len(result.viewer["frames"]), result.summary["ticks_executed"])
        with self.assertRaisesRegex(RuntimeError, "one-shot"):
            world.run()

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
        self.assertEqual(
            summary_only.summary["summary_schema_version"],
            SUMMARY_SCHEMA_VERSION,
        )
        self.assertEqual(tuple(summary_only.summary), SHARED_SUMMARY_FIELDS)
        for field in SHARED_SUMMARY_FIELDS:
            self.assertEqual(summary_only.summary[field], full.summary[field], msg=field)
        for field in FULL_ONLY_SUMMARY_FIELDS:
            self.assertNotIn(field, summary_only.summary)

    def test_summary_only_resource_and_selection_analytics_do_not_build_replay_surfaces(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=12))

        with (
            patch("evolution_sim.env.runtime.frames.capture_frame") as capture_frame,
            patch(
                "evolution_sim.env.runtime.surfaces.materialize_frame_surfaces"
            ) as materialize_frame_surfaces,
        ):
            result = world.run(mode=RunMode.SUMMARY_ONLY)

        self.assertIn("resource_pressure", result.summary)
        self.assertIn("selection_heredity", result.summary)
        self.assertIsNone(result.viewer)
        self.assertIsNone(result.events)
        capture_frame.assert_not_called()
        materialize_frame_surfaces.assert_not_called()

    def test_summary_gene_averages_distinguish_historical_and_alive_agents(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=40)).run()
        agent_catalog = result.viewer["agent_catalog"]
        all_genomes = [record["genome"] for record in agent_catalog.values()]
        alive_genomes = [
            record["genome"]
            for record in agent_catalog.values()
            if record["death_tick"] is None
        ]

        def avg_gene(genomes: list[dict[str, float]], field: str) -> float:
            return round(
                sum(float(genome[field]) for genome in genomes) / max(len(genomes), 1),
                4,
            )

        self.assertGreater(len(all_genomes), len(alive_genomes))
        self.assertEqual(
            result.summary["avg_max_energy_gene"],
            result.summary["avg_historical_max_energy_gene"],
        )
        self.assertEqual(
            result.summary["avg_historical_max_energy_gene"],
            avg_gene(all_genomes, "max_energy"),
        )
        self.assertEqual(
            result.summary["avg_alive_max_energy_gene"],
            avg_gene(alive_genomes, "max_energy"),
        )
        self.assertEqual(
            result.summary["avg_historical_attack_power_gene"],
            avg_gene(all_genomes, "attack_power"),
        )
        self.assertEqual(
            result.summary["avg_alive_attack_power_gene"],
            avg_gene(alive_genomes, "attack_power"),
        )
        selection = result.summary["selection_heredity"]
        initial_energy = selection["initial_trait_distributions"]["max_energy"]
        terminal_energy = selection["terminal_alive_trait_distributions"]["max_energy"]
        self.assertEqual(initial_energy["count"], result.config["initial_agents"])
        self.assertEqual(terminal_energy["count"], result.summary["alive_agents"])
        self.assertIn("median", initial_energy)
        self.assertIn(
            "max_energy",
            selection["terminal_minus_initial_mean"],
        )
