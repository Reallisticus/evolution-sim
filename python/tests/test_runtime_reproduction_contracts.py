from __future__ import annotations

from python.tests.runtime_test_helpers import *


class RuntimeReproductionContractTests(RuntimeContractTestHelpers):
    def test_reproduction_blocked_by_max_population_is_reported(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(width=4, height=4, max_agents=1)
        )
        self._place_ready_agent(world, x=1, y=1)

        result = world.run(mode=RunMode.FULL_REPLAY)

        blocked_events = [
            event
            for event in result.events
            if event["type"] == "agent_reproduction_blocked"
        ]
        self.assertEqual(result.summary["births"], 0)
        self.assertEqual(result.summary["max_agents"], 1)
        self.assertEqual(result.summary["max_agent_saturation_at_end"], 1.0)
        self.assertEqual(result.summary["carrying_capacity"]["at_cap_ticks"], 1)
        self.assertEqual(
            result.summary["carrying_capacity"]["at_cap_tick_share"],
            1.0,
        )
        self.assertEqual(
            result.summary["reproduction_end"]["blocked_run_counts"]["max_population"],
            1,
        )
        self.assertEqual(blocked_events[0]["data"]["reason"], "max_population")

    def test_carrying_capacity_counts_saturation_births(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(width=5, height=5, max_agents=3)
        )
        self._place_ready_agent(world, x=1, y=1, lineage_id=1)
        self._place_ready_agent(world, x=3, y=3, lineage_id=2)

        result = world.run(mode=RunMode.SUMMARY_ONLY)
        capacity = result.summary["carrying_capacity"]

        self.assertEqual(result.summary["births"], 1)
        self.assertEqual(capacity["at_cap_ticks"], 1)
        self.assertEqual(capacity["saturation_births"], 1)
        self.assertEqual(capacity["saturation_deaths"], 0)

    def test_carrying_capacity_counts_saturation_deaths(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(width=4, height=4, max_agents=1)
        )
        agent = self._place_ready_agent(world, x=1, y=1)
        agent.age = world.config.max_age

        result = world.run(mode=RunMode.SUMMARY_ONLY)
        capacity = result.summary["carrying_capacity"]

        self.assertEqual(result.summary["deaths"], 1)
        self.assertEqual(capacity["at_cap_ticks"], 1)
        self.assertEqual(capacity["saturation_deaths"], 1)

    def test_resource_pressure_budget_tracks_plant_removal_and_energy_spend(self) -> None:
        class FixedPolicy:
            policy_id = "fixed_policy"
            policy_version = "fixed_policy_v1"

            def decide(
                self,
                observation: dict[str, object],
                action_mask: dict[str, bool],
            ) -> ActionDecision:
                return ActionDecision(
                    requested_action="eat",
                    source=self.policy_id,
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                )

        world = SimulationWorld(
            self._ready_reproduction_config(
                width=4,
                height=4,
                max_age=2_000,
                base_energy_drain=0.1,
                reproduction=ReproductionConfig(
                    min_age=1_000,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    energy_cost=0.0,
                ),
            ),
            policy=FixedPolicy(),
        )
        agent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            genome=self._mixed_genome(),
        )
        agent.energy = agent.genome.max_energy * 0.4
        world.grid[agent.y][agent.x].food = 0.5

        result = world.run(mode=RunMode.SUMMARY_ONLY)
        pressure = result.summary["resource_pressure"]
        plant_budget = pressure["plant_budget"]
        energy_spend = pressure["energy_spend"]

        self.assertGreater(plant_budget["energy_removed"], 0.0)
        self.assertGreaterEqual(plant_budget["energy_created"], 0.0)
        self.assertGreater(energy_spend["metabolism"], 0.0)
        self.assertEqual(energy_spend["movement"], 0.0)
        self.assertGreater(energy_spend["total"], 0.0)

    def test_reproduction_phase_counts_births_against_max_population(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(width=5, height=5, max_agents=3)
        )
        first = self._place_ready_agent(world, x=1, y=1, lineage_id=1)
        second = self._place_ready_agent(world, x=3, y=3, lineage_id=2)

        births = runtime_reproduction.run_reproduction_phase(world)

        blocked_events = [
            event
            for event in world.events
            if event.type == EventType.AGENT_REPRODUCTION_BLOCKED
        ]
        self.assertEqual(births, 1)
        self.assertEqual(world.births, 1)
        self.assertEqual(
            len([agent for agent in world.agents.values() if agent.alive]),
            3,
        )
        self.assertEqual(len(blocked_events), 1)
        self.assertEqual(blocked_events[0].agent_id, second.agent_id)
        self.assertEqual(blocked_events[0].data["reason"], "max_population")
        self.assertEqual(blocked_events[0].data["alive_agents"], 3)
        self.assertEqual(first.last_reproduction_tick, world.tick)
        self.assertNotEqual(second.last_reproduction_tick, world.tick)

    def test_reproduction_blocked_by_local_crowding_is_reported(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(width=3, height=3, max_agents=20)
        )
        genome = Genome.sample_initial(world.rng)
        for y in range(3):
            for x in range(3):
                self._place_ready_agent(
                    world,
                    x=x,
                    y=y,
                    lineage_id=y * 3 + x + 1,
                    genome=genome,
                )

        result = world.run(mode=RunMode.FULL_REPLAY)

        blocked_events = [
            event
            for event in result.events
            if event["type"] == "agent_reproduction_blocked"
        ]
        self.assertEqual(result.summary["births"], 0)
        self.assertGreaterEqual(
            result.summary["reproduction_end"]["blocked_run_counts"]["local_crowding"],
            1,
        )
        self.assertEqual(
            result.summary["reproduction_end"]["blocked_by_local_crowding_agents"],
            9,
        )
        self.assertTrue(
            all(event["data"]["reason"] == "local_crowding" for event in blocked_events)
        )

    def test_reproduction_placement_context_controls_max_population_boundary(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(width=4, height=4, max_agents=20)
        )
        parent = self._place_ready_agent(world, x=1, y=1)
        placement = replace(
            world._reproduction_placement_context(),
            max_agents=1,
            current_alive_count=lambda: 1,
            has_empty_neighbor=lambda x, y: (_ for _ in ()).throw(
                AssertionError("local crowding should not be checked after saturation")
            ),
        )

        availability = runtime_reproduction.reproduction_availability(
            world,
            placement_context=placement,
        )
        reason = runtime_reproduction.reproduction_block_reason(
            world,
            parent,
            availability,
            placement_context=placement,
        )

        self.assertTrue(availability.population_saturated)
        self.assertEqual(reason, "max_population")

    def test_reproduction_placement_context_controls_local_crowding_boundary(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(width=4, height=4, max_agents=20)
        )
        parent = self._place_ready_agent(world, x=1, y=1)
        placement = replace(
            world._reproduction_placement_context(),
            max_agents=20,
            current_alive_count=lambda: 1,
            has_empty_neighbor=lambda x, y: False,
        )

        with patch.object(
            world,
            "_has_empty_neighbor",
            side_effect=AssertionError("placement context was bypassed"),
        ):
            reason = runtime_reproduction.reproduction_block_reason(
                world,
                parent,
                placement_context=placement,
            )

        self.assertEqual(reason, "local_crowding")

    def test_reproduction_context_controls_biological_readiness_without_world(
        self,
    ) -> None:
        config = self._ready_reproduction_config(
            reproduction=ReproductionConfig(
                min_age=12,
                cooldown_ticks=5,
                min_hydration_fraction=0.8,
                energy_cost=0.1,
            ),
            combat=CombatConfig(min_reproduction_health_ratio=0.75),
        )
        genome = self._hunter_genome()
        agent = Agent(
            agent_id=1,
            parent_id=None,
            lineage_id=1,
            birth_tick=0,
            death_tick=None,
            x=1,
            y=1,
            energy=0.1,
            hydration=0.4,
            health=0.4,
            max_health=1.0,
            injury_load=0.0,
            age=4,
            alive=True,
            last_reproduction_tick=8,
            last_damage_source="none",
            recent_plant_energy=0.0,
            recent_fresh_kill_energy=0.0,
            recent_carcass_energy=0.0,
            genome_vector=genome_vector(genome),
            genome=genome,
            reproductive_group_id=1,
            reproductive_stage=STAGE0_ASEXUAL,
            reproductive_expression="asexual",
        )
        profile = TrophicProfile(
            plant_share=0.0,
            animal_share=1.0,
            scavenger_share=0.0,
            hunter_share=1.0,
            breadth=0.25,
            plant_drive=0.0,
            animal_drive=1.0,
            scavenger_drive=0.0,
            hunter_drive=1.0,
            role="carnivore",
            meat_mode="hunter",
        )
        placement = runtime_reproduction.ReproductionPlacementContext(
            max_agents=10,
            current_alive_count=lambda: 1,
            has_empty_neighbor=lambda x, y: True,
            find_empty_neighbor=lambda x, y: (x + 1, y),
            sibling_destination_candidates=lambda x, y: [(x + 1, y)],
            can_place_at=lambda x, y: True,
            place_agent=lambda child: None,
            invalidate_spatial_state=lambda: None,
        )
        context = runtime_reproduction.ReproductionContext(
            config=config,
            rng=Random(7),
            tick=10,
            next_agent_id=2,
            placement=placement,
            trophic_profile=lambda checked_agent: profile,
            trophic_profile_for_genome=lambda checked_genome: profile,
            trophic_role=lambda checked_agent: profile.role,
            meat_mode=lambda checked_agent: profile.meat_mode,
            matched_diet_ratio=lambda checked_agent, checked_profile: 0.1,
            matched_diet_threshold=lambda checked_profile: 0.6,
            health_ratio=lambda checked_agent: 0.4,
            emit=lambda *args, **kwargs: None,
        )

        reasons = runtime_reproduction.biological_reproduction_block_reasons(
            None,
            agent,
            context=context,
        )
        reason = runtime_reproduction.reproduction_block_reason(
            None,
            agent,
            context=context,
        )

        self.assertEqual(
            reasons,
            ["age", "cooldown", "energy", "hydration", "health", "matched_diet"],
        )
        self.assertEqual(reason, "biological")

    def test_asexual_birth_plan_is_context_only_and_does_not_mutate_parent(
        self,
    ) -> None:
        config = self._ready_reproduction_config(
            reproduction=ReproductionConfig(
                min_age=1,
                cooldown_ticks=0,
                min_hydration_fraction=0.0,
                energy_cost=0.2,
            )
        )
        profile = self._test_trophic_profile("none")
        context = self._context_only_reproduction_context(
            config=config,
            profile=profile,
            next_agent_id=700,
        )
        genome = self._mixed_genome()
        parent = Agent(
            agent_id=1,
            parent_id=None,
            lineage_id=11,
            birth_tick=0,
            death_tick=None,
            x=1,
            y=1,
            energy=1.2,
            hydration=1.0,
            health=1.0,
            max_health=1.0,
            injury_load=0.0,
            age=10,
            alive=True,
            last_reproduction_tick=-10_000,
            last_damage_source="none",
            recent_plant_energy=0.0,
            recent_fresh_kill_energy=0.0,
            recent_carcass_energy=0.0,
            genome_vector=genome_vector(genome),
            genome=genome,
            reproductive_group_id=11,
            reproductive_stage=STAGE0_ASEXUAL,
            reproductive_expression="asexual",
        )
        energy_before = parent.energy

        plan = runtime_reproduction.build_asexual_birth_plan(
            None,
            parent,
            (2, 1),
            profile,
            context=context,
        )

        self.assertEqual(plan.reproduction_mode, runtime_mating.ASEXUAL_REPRODUCTION_MODE)
        self.assertEqual(plan.parents, (parent,))
        self.assertEqual(plan.parent_energy_costs, (0.2,))
        self.assertEqual(plan.multi_offspring_desired_count, 1)
        self.assertEqual(plan.multi_offspring_limit_reasons, ())
        self.assertEqual(plan.destinations, ((2, 1),))
        self.assertEqual(plan.children[0].agent_id, 700)
        self.assertEqual(plan.children[0].parent_id, parent.agent_id)
        self.assertEqual(plan.children[0].lineage_id, parent.lineage_id)
        self.assertEqual(plan.children[0].birth_tick, 25)
        self.assertEqual(parent.energy, energy_before)

    def test_sexual_birth_plan_is_context_only_and_preserves_sibling_metadata(
        self,
    ) -> None:
        config = self._ready_reproduction_config(
            reproduction=ReproductionConfig(
                min_age=1,
                cooldown_ticks=0,
                min_hydration_fraction=0.0,
                energy_cost=0.1,
                sexual_partner_radius=1,
                multi_offspring_enabled=True,
                multi_offspring_threshold=0.8,
                multi_offspring_max_count=3,
            )
        )
        profile = self._test_trophic_profile("none")
        context = self._context_only_reproduction_context(
            config=config,
            profile=profile,
            next_agent_id=900,
        )
        genome = self._sexualized_genome(self._mixed_genome())
        fecund_genome = replace(
            genome,
            reproductive=replace(
                genome.reproductive,
                fecundity_potential=1.0,
            ),
        )
        parent = Agent(
            agent_id=1,
            parent_id=None,
            lineage_id=1,
            birth_tick=0,
            death_tick=None,
            x=2,
            y=2,
            energy=1.4,
            hydration=1.0,
            health=1.0,
            max_health=1.0,
            injury_load=0.0,
            age=10,
            alive=True,
            last_reproduction_tick=-10_000,
            last_damage_source="none",
            recent_plant_energy=0.0,
            recent_fresh_kill_energy=0.0,
            recent_carcass_energy=0.0,
            genome_vector=genome_vector(fecund_genome),
            genome=fecund_genome,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
        )
        partner = replace(
            parent,
            agent_id=2,
            x=2,
            y=3,
            genome_vector=genome_vector(fecund_genome),
        )
        mate_candidate = runtime_mating.MateCandidate(
            agent=partner,
            distance=1,
            compatibility_score=0.95,
            inbreeding_penalty=0.0,
        )

        plan = runtime_reproduction.build_sexual_birth_plan(
            None,
            parent,
            partner,
            [(3, 2), (1, 2)],
            profile,
            profile,
            mate_candidate,
            parent_cost=0.05,
            partner_cost=0.06,
            multi_offspring_desired_count=3,
            multi_offspring_limit_reasons=("local_destination_capacity",),
            context=context,
        )

        self.assertEqual(plan.reproduction_mode, SEXUAL_REPRODUCTION_MODE)
        self.assertEqual(plan.parents, (parent, partner))
        self.assertEqual(plan.parent_energy_costs, (0.05, 0.06))
        self.assertEqual(plan.multi_offspring_desired_count, 3)
        self.assertEqual(
            plan.multi_offspring_limit_reasons,
            ("local_destination_capacity",),
        )
        self.assertEqual(plan.destinations, ((3, 2), (1, 2)))
        self.assertEqual([child.agent_id for child in plan.children], [900, 901])
        self.assertEqual(plan.sibling_child_ids, [900, 901])
        self.assertTrue(
            all(
                child.secondary_parent_id == partner.agent_id
                for child in plan.children
            )
        )

    def test_shared_reproduction_context_allocates_unique_birth_ids(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=7,
                max_agents=12,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    energy_cost=0.0,
                ),
            )
        )
        first_parent = self._place_ready_agent(world, x=1, y=1, lineage_id=1)
        second_parent = self._place_ready_agent(world, x=5, y=5, lineage_id=2)
        shared_context = runtime_reproduction.build_reproduction_context(world)

        first_profile = shared_context.trophic_profile(first_parent)
        second_profile = shared_context.trophic_profile(second_parent)
        first_birth = runtime_reproduction.reproduce_asexual(
            world,
            first_parent,
            (1, 2),
            first_profile,
            context=shared_context,
        )
        second_birth = runtime_reproduction.reproduce_asexual(
            world,
            second_parent,
            (5, 4),
            second_profile,
            context=shared_context,
        )

        children = [
            agent
            for agent in world.agents.values()
            if agent.parent_id in {first_parent.agent_id, second_parent.agent_id}
        ]

        self.assertTrue(first_birth)
        self.assertTrue(second_birth)
        self.assertEqual(len(children), 2)
        self.assertEqual(
            len({child.agent_id for child in children}),
            len(children),
        )
        self.assertEqual(world.births, 2)

    def test_reproduction_biological_blockers_are_grouped_by_role_and_mode(self) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                initial_agents=0,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
                base_energy_drain=0.0,
                base_hydration_drain=0.0,
                reproduction=ReproductionConfig(min_age=30, cooldown_ticks=24),
                combat=CombatConfig(min_reproduction_health_ratio=0.9),
            )
        )
        agent = self._place_ready_agent(world, x=1, y=1)
        agent.age = 0
        agent.last_reproduction_tick = 0
        agent.energy = agent.reproduction_threshold() * 0.1
        agent.hydration = agent.genome.max_hydration * 0.1
        agent.health = agent.max_health * 0.5
        agent.recent_plant_energy = 0.0
        agent.recent_fresh_kill_energy = 0.0
        agent.recent_carcass_energy = 0.0

        result = world.run(mode=RunMode.SUMMARY_ONLY)

        reproduction = result.summary["reproduction_end"]
        role = next(
            role
            for role, counts in reproduction["by_trophic_role"].items()
            if counts["alive_agents"] == 1
        )
        mode = next(
            mode
            for mode, counts in reproduction["by_meat_mode"].items()
            if counts["alive_agents"] == 1
        )
        self.assertEqual(reproduction["biologically_ready_agents"], 0)
        for reason in ("age", "cooldown", "energy", "hydration", "health", "matched_diet"):
            self.assertEqual(reproduction["biological_blocker_counts"][reason], 1)
            self.assertEqual(
                reproduction["biological_blocker_counts_by_trophic_role"][role][reason],
                1,
            )
            self.assertEqual(
                reproduction["biological_blocker_counts_by_meat_mode"][mode][reason],
                1,
            )
        self.assertEqual(
            reproduction["by_trophic_role"][role]["biologically_ready_agents"],
            0,
        )
        self.assertEqual(
            reproduction["by_meat_mode"][mode]["biologically_ready_agents"],
            0,
        )
        role_energy = reproduction["energy_readiness_by_trophic_role"][role]
        mode_energy = reproduction["energy_readiness_by_meat_mode"][mode]
        for energy_counts in (role_energy, mode_energy):
            self.assertEqual(energy_counts["alive_agents"], 1)
            self.assertEqual(energy_counts["energy_shortfall_agents"], 1)
            self.assertGreater(
                energy_counts["energy_required_total"],
                energy_counts["energy_total"],
            )
            self.assertGreater(energy_counts["energy_gap_total"], 0.0)

    def test_genome_recombination_contract_groups_all_inherited_genes(self) -> None:
        contract = genome_recombination_contract()
        grouped_genes = [gene for group in GENOME_GROUPS for gene in group.genes]
        grouped_reproductive_traits = [
            trait for group in GENOME_GROUPS for trait in group.reproductive_traits
        ]

        self.assertEqual(
            contract["schema_version"],
            GENOME_RECOMBINATION_CONTRACT_VERSION,
        )
        self.assertEqual(sorted(grouped_genes), sorted(GENE_LIMITS))
        self.assertEqual(len(grouped_genes), len(set(grouped_genes)))
        self.assertEqual(
            sorted(grouped_reproductive_traits),
            sorted(REPRODUCTIVE_GENE_LIMITS),
        )
        self.assertEqual(
            len(grouped_reproductive_traits),
            len(set(grouped_reproductive_traits)),
        )
        json.dumps(contract)

    def test_grouped_recombine_is_deterministic_and_group_coherent(self) -> None:
        left = Genome(
            **{gene: limits[0] for gene, limits in GENE_LIMITS.items()},
            reproductive=ReproductiveGenome(
                **{
                    trait: limits[0]
                    for trait, limits in REPRODUCTIVE_GENE_LIMITS.items()
                }
            ),
        )
        right = Genome(
            **{gene: limits[1] for gene, limits in GENE_LIMITS.items()},
            reproductive=ReproductiveGenome(
                **{
                    trait: limits[1]
                    for trait, limits in REPRODUCTIVE_GENE_LIMITS.items()
                }
            ),
        )

        child = recombine_genomes(left, right, Random(11))
        repeated = recombine_genomes(left, right, Random(11))

        self.assertEqual(child.to_dict(), repeated.to_dict())
        for group in GENOME_GROUPS:
            first_gene = group.genes[0]
            source = (
                left
                if getattr(child, first_gene) == getattr(left, first_gene)
                else right
            )
            for gene in group.genes:
                self.assertEqual(getattr(child, gene), getattr(source, gene), msg=gene)
            for trait in group.reproductive_traits:
                self.assertEqual(
                    getattr(child.reproductive, trait),
                    getattr(source.reproductive, trait),
                    msg=trait,
                )

    def test_reproductive_group_registry_tracks_stage0_asexual_groups(self) -> None:
        config = WorldConfig(seed=7, max_ticks=40)
        result = SimulationWorld(config).run()

        summary = result.summary["reproductive_groups_end"]
        catalog = result.viewer["reproductive_group_catalog"]

        self.assertEqual(
            reproductive_group_contract()["schema_version"],
            REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        )
        self.assertEqual(summary["schema_version"], REPRODUCTIVE_GROUP_CONTRACT_VERSION)
        self.assertEqual(catalog["schema_version"], REPRODUCTIVE_GROUP_CONTRACT_VERSION)
        self.assertEqual(summary["group_count"], config.initial_agents)
        self.assertEqual(summary["asexual_births"], result.summary["births"])
        self.assertEqual(summary["sexual_births"], 0)
        self.assertEqual(summary["hybrid_births"], 0)
        self.assertEqual(
            summary["stage_counts"],
            {STAGE0_ASEXUAL: config.initial_agents},
        )
        self.assertEqual(
            summary["alive_stage_counts"],
            {STAGE0_ASEXUAL: result.summary["alive_agents"]},
        )
        self.assertEqual(
            sum(group["member_count"] for group in catalog["groups"].values()),
            result.summary["total_agents_seen"],
        )
        self.assertEqual(
            sum(group["alive_member_count"] for group in catalog["groups"].values()),
            result.summary["alive_agents"],
        )
        self.assertEqual(
            sum(summary["alive_expression_counts"].values()),
            result.summary["alive_agents"],
        )
        self.assertEqual(
            summary["alive_expression_counts_by_stage"],
            {
                STAGE0_ASEXUAL: {
                    "asexual": result.summary["alive_agents"],
                }
            },
        )
        self.assertEqual(
            result.summary["reproduction_end"]["reproductive_capability_counts"],
            {
                "hybridization": 0,
                "multi_offspring": 0,
                "proto_role_differentiation": 0,
                "sexual_reproduction": 0,
                "xyz_expression": 0,
            },
        )
        self.assertEqual(
            result.summary["reproduction_end"]["mate_search_run_counts"],
            runtime_reproduction.empty_reproduction_mate_search_counts(),
        )
        for agent in result.viewer["agent_catalog"].values():
            group_id = str(agent["reproductive_group_id"])
            self.assertIn(group_id, catalog["groups"])
            self.assertEqual(agent["reproductive_stage"], STAGE0_ASEXUAL)
            self.assertEqual(agent["reproductive_expression"], "asexual")
        for group in catalog["groups"].values():
            self.assertEqual(
                sum(group["alive_stage_counts"].values()),
                group["alive_member_count"],
            )
            self.assertEqual(
                sum(group["alive_expression_counts"].values()),
                group["alive_member_count"],
            )

    def test_reproductive_stage_classifies_proto_and_xyz_expression(self) -> None:
        config = ReproductionConfig()
        proto_x = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=-0.8,
        )
        proto_y = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=0.8,
        )
        proto_z = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=0.0,
            plasticity=0.8,
        )
        xyz_x = self._role_genome(
            self._mixed_genome(),
            role_drive=0.78,
            expression_bias=-0.8,
        )
        xyz_y = self._role_genome(
            self._mixed_genome(),
            role_drive=0.78,
            expression_bias=0.8,
        )
        xyz_z = self._role_genome(
            self._mixed_genome(),
            role_drive=0.78,
            expression_bias=0.0,
            plasticity=0.8,
        )

        self.assertEqual(
            runtime_mating.reproductive_stage_for_genome(proto_x, config),
            STAGE2_PROTO_ROLES,
        )
        self.assertEqual(
            runtime_mating.reproductive_expression_for_genome(proto_x, config),
            PROTO_X_EXPRESSION,
        )
        self.assertEqual(
            runtime_mating.reproductive_expression_for_genome(proto_y, config),
            PROTO_Y_EXPRESSION,
        )
        self.assertEqual(
            runtime_mating.reproductive_expression_for_genome(proto_z, config),
            PROTO_Z_EXPRESSION,
        )
        self.assertEqual(
            runtime_mating.reproductive_stage_for_genome(xyz_x, config),
            STAGE3_X_Y_Z,
        )
        self.assertEqual(
            runtime_mating.reproductive_expression_for_genome(xyz_x, config),
            X_EXPRESSION,
        )
        self.assertEqual(
            runtime_mating.reproductive_expression_for_genome(xyz_y, config),
            Y_EXPRESSION,
        )
        self.assertEqual(
            runtime_mating.reproductive_expression_for_genome(xyz_z, config),
            Z_EXPRESSION,
        )
        self.assertEqual(
            runtime_mating.reproductive_capabilities_for_genome(xyz_z, config),
            {
                "sexual_reproduction": True,
                "proto_role_differentiation": True,
                "xyz_expression": True,
                "hybridization": False,
                "multi_offspring": False,
            },
        )

    def test_multi_offspring_capability_is_config_gated_and_observable(self) -> None:
        gated_config = ReproductionConfig(
            min_age=1,
            cooldown_ticks=0,
            min_hydration_fraction=0.0,
            multi_offspring_enabled=True,
            multi_offspring_threshold=0.8,
            multi_offspring_max_count=2,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=5,
                reproduction=gated_config,
            )
        )
        sexualized_genome = self._sexualized_genome(self._mixed_genome())
        genome = replace(
            sexualized_genome,
            reproductive=replace(
                sexualized_genome.reproductive,
                fecundity_potential=0.9,
            ),
        )
        stage = runtime_mating.reproductive_stage_for_genome(genome, gated_config)
        expression = runtime_mating.reproductive_expression_for_genome(
            genome,
            gated_config,
        )
        self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=stage,
            reproductive_expression=expression,
            genome=genome,
        )

        default_config = ReproductionConfig()
        stats = world._reproduction_readiness_counts(world.alive_agents())

        self.assertFalse(
            runtime_mating.multi_offspring_unlocked(genome, default_config)
        )
        self.assertTrue(runtime_mating.multi_offspring_unlocked(genome, gated_config))
        self.assertEqual(
            runtime_mating.reproductive_capabilities_for_genome(
                genome,
                gated_config,
            )["multi_offspring"],
            True,
        )
        self.assertEqual(
            stats["reproductive_capability_counts"]["multi_offspring"],
            1,
        )

    def test_proto_role_mating_requires_complementary_expression(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=6,
                height=6,
                max_agents=20,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    sexual_partner_radius=1,
                ),
            )
        )
        config = world.config.reproduction
        parent_genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=-0.8,
        )
        same_role_genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=-0.8,
        )
        compatible_genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=0.8,
        )

        def place_role_agent(x: int, y: int, genome: Genome) -> Agent:
            return self._place_ready_agent(
                world,
                x=x,
                y=y,
                lineage_id=1,
                reproductive_group_id=1,
                reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                    genome,
                    config,
                ),
                reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                    genome,
                    config,
                ),
                genome=genome,
            )

        parent = place_role_agent(2, 2, parent_genome)
        same_role = place_role_agent(2, 3, same_role_genome)
        compatible = place_role_agent(3, 2, compatible_genome)

        self.assertFalse(
            runtime_mating.expression_compatibility(parent, same_role, config)[0]
        )
        mate = runtime_mating.choose_same_group_mate(
            parent,
            world.agents.values(),
            config=config,
            biologically_ready=lambda agent: True,
        )
        report = runtime_mating.same_group_mate_search_report(
            parent,
            world.agents.values(),
            config=config,
            biologically_ready=lambda agent: True,
        )

        self.assertIsNotNone(mate)
        self.assertEqual(mate.agent.agent_id, compatible.agent_id)
        self.assertIsNotNone(report.selected)
        self.assertEqual(report.selected.agent.agent_id, compatible.agent_id)
        self.assertEqual(report.reason_counts["expression_incompatible"], 1)
        self.assertEqual(report.constraint_counts["expression_incompatible"], 1)
        self.assertEqual(report.expression_compatible_candidates, 1)
        self.assertEqual(report.expression_incompatible_candidates, 1)
        self.assertGreater(mate.compatibility_score, 0.0)

    def test_mate_search_report_preserves_overlapping_constraints(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=7,
                max_agents=20,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    sexual_partner_radius=1,
                ),
            )
        )
        config = world.config.reproduction
        genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=-0.8,
        )
        parent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                genome,
                config,
            ),
            genome=genome,
        )
        partner = self._place_ready_agent(
            world,
            x=4,
            y=1,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                genome,
                config,
            ),
            genome=genome,
        )

        report = runtime_mating.same_group_mate_search_report(
            parent,
            world.agents.values(),
            config=config,
            biologically_ready=lambda agent: agent.agent_id != partner.agent_id,
        )

        self.assertIsNone(report.selected)
        self.assertEqual(report.same_group_candidates, 1)
        self.assertEqual(report.same_group_sexual_candidates, 1)
        self.assertEqual(report.in_radius_candidates, 0)
        self.assertEqual(report.biologically_ready_candidates, 0)
        self.assertEqual(report.reason_counts["partner_out_of_radius"], 1)
        self.assertEqual(report.constraint_counts["partner_out_of_radius"], 1)
        self.assertEqual(report.constraint_counts["partner_not_ready"], 1)
        self.assertEqual(report.constraint_counts["expression_incompatible"], 1)
        self.assertEqual(report.expression_incompatible_candidates, 1)

    def test_mate_search_counts_expression_compatible_candidates_losslessly(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=7,
                max_agents=20,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    sexual_partner_radius=1,
                ),
            )
        )
        config = world.config.reproduction
        parent_genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=-0.8,
        )
        compatible_genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=0.8,
        )
        parent = self._place_ready_agent(
            world,
            x=1,
            y=1,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                parent_genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                parent_genome,
                config,
            ),
            genome=parent_genome,
        )
        partner = self._place_ready_agent(
            world,
            x=4,
            y=1,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                compatible_genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                compatible_genome,
                config,
            ),
            genome=compatible_genome,
        )

        report = runtime_mating.same_group_mate_search_report(
            parent,
            world.agents.values(),
            config=config,
            biologically_ready=lambda agent: agent.agent_id != partner.agent_id,
        )

        self.assertIsNone(report.selected)
        self.assertEqual(report.same_group_sexual_candidates, 1)
        self.assertEqual(report.expression_compatible_candidates, 1)
        self.assertEqual(report.expression_incompatible_candidates, 0)
        self.assertEqual(report.reason_counts["partner_out_of_radius"], 1)
        self.assertEqual(report.constraint_counts["partner_out_of_radius"], 1)
        self.assertEqual(report.constraint_counts["partner_not_ready"], 1)

    def test_reproduction_readiness_reports_role_stage_capabilities(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(width=6, height=6, max_agents=20)
        )
        config = world.config.reproduction
        proto_x = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=-0.8,
        )
        xyz_z = self._role_genome(
            self._mixed_genome(),
            role_drive=0.78,
            expression_bias=0.0,
            plasticity=0.8,
        )
        self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                proto_x,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                proto_x,
                config,
            ),
            genome=proto_x,
        )
        self._place_ready_agent(
            world,
            x=3,
            y=2,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                xyz_z,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                xyz_z,
                config,
            ),
            genome=xyz_z,
        )

        stats = world._reproduction_readiness_counts(world.alive_agents())

        self.assertEqual(
            stats["reproductive_stage_counts"],
            {
                STAGE2_PROTO_ROLES: 1,
                STAGE3_X_Y_Z: 1,
            },
        )
        self.assertEqual(
            stats["reproductive_expression_counts"],
            {
                PROTO_X_EXPRESSION: 1,
                Z_EXPRESSION: 1,
            },
        )
        self.assertEqual(
            stats["reproductive_capability_counts"],
            {
                "hybridization": 0,
                "multi_offspring": 0,
                "proto_role_differentiation": 2,
                "sexual_reproduction": 2,
                "xyz_expression": 1,
            },
        )
        self.assertEqual(stats["biologically_ready_group_count"], 1)
        self.assertEqual(stats["ready_group_count"], 1)

    def test_xyz_plastic_expression_can_pair_with_fixed_expression(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=6,
                height=6,
                max_agents=20,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    sexual_partner_radius=1,
                ),
            )
        )
        config = world.config.reproduction
        x_genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.78,
            expression_bias=-0.8,
        )
        z_genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.78,
            expression_bias=0.0,
            plasticity=0.8,
        )
        parent = self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                x_genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                x_genome,
                config,
            ),
            genome=x_genome,
        )
        plastic_partner = self._place_ready_agent(
            world,
            x=3,
            y=2,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                z_genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                z_genome,
                config,
            ),
            genome=z_genome,
        )

        mate = runtime_mating.choose_same_group_mate(
            parent,
            world.agents.values(),
            config=config,
            biologically_ready=lambda agent: True,
        )

        self.assertIsNotNone(mate)
        self.assertEqual(mate.agent.agent_id, plastic_partner.agent_id)
        self.assertGreater(mate.compatibility_score, 0.0)

    def test_reproductive_group_records_promote_to_highest_member_stage(self) -> None:
        parent = Agent(
            agent_id=1,
            parent_id=None,
            lineage_id=1,
            birth_tick=0,
            death_tick=None,
            x=0,
            y=0,
            energy=1.0,
            hydration=1.0,
            health=1.0,
            max_health=1.0,
            injury_load=0.0,
            age=10,
            alive=True,
            last_reproduction_tick=-10_000,
            last_damage_source="none",
            recent_plant_energy=0.0,
            recent_fresh_kill_energy=0.0,
            recent_carcass_energy=0.0,
            genome_vector=(),
            genome=self._mixed_genome(),
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
        )
        child = replace(
            parent,
            agent_id=2,
            parent_id=1,
            reproductive_stage=STAGE3_X_Y_Z,
            reproductive_expression=Z_EXPRESSION,
        )
        registry = {
            1: runtime_reproduction.ReproductiveGroupRecord(
                group_id=1,
                founder_lineage_id=1,
                founder_agent_id=1,
                created_tick=0,
                stage=STAGE1_FACULTATIVE_SEX,
            )
        }

        runtime_reproduction.record_asexual_birth(registry, parent, child, tick=1)

        self.assertEqual(registry[1].stage, STAGE3_X_Y_Z)

    def test_role_mate_search_fallback_records_expression_incompatibility(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=6,
                height=6,
                max_agents=20,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    sexual_partner_radius=1,
                ),
            )
        )
        config = world.config.reproduction
        genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=-0.8,
        )
        parent = self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                genome,
                config,
            ),
            genome=genome,
        )
        self._place_ready_agent(
            world,
            x=2,
            y=3,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                genome,
                config,
            ),
            genome=genome,
        )

        self.assertTrue(runtime_reproduction.reproduce(world, parent))

        child = next(
            agent
            for agent in world.agents.values()
            if agent.parent_id == parent.agent_id
        )
        run_counts = world.run_reproduction_mate_search_counts
        frame_stats = runtime_reproduction.build_frame_reproduction_stats(
            world,
            world.alive_agents(),
            trophic_role_codes=TROPHIC_ROLE_CODES,
            meat_mode_codes=MEAT_MODE_CODES,
        )

        self.assertIsNone(child.secondary_parent_id)
        self.assertEqual(run_counts["sexual_parent_candidates"], 1)
        self.assertEqual(run_counts["sexual_searches"], 1)
        self.assertEqual(run_counts["sexual_successes"], 0)
        self.assertEqual(run_counts["asexual_fallbacks_after_sexual_candidate"], 1)
        self.assertEqual(run_counts["candidate_expression_incompatible"], 1)
        self.assertEqual(run_counts["fallback_expression_incompatible"], 1)
        self.assertEqual(
            world.tick_reproduction_mate_search_events[0]["fallback_reason"],
            "expression_incompatible",
        )
        self.assertEqual(
            frame_stats["mate_search_this_tick"]["fallback_expression_incompatible"],
            1,
        )
        self.assertEqual(
            frame_stats["mate_search_events"][0]["reason_counts"][
                "expression_incompatible"
            ],
            1,
        )
        self.assertEqual(
            frame_stats["mate_search_events"][0]["constraint_counts"][
                "expression_incompatible"
            ],
            1,
        )

    def test_mate_search_diagnostics_skip_event_details_when_not_recording_ticks(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=6,
                height=6,
                max_agents=20,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    sexual_partner_radius=1,
                ),
            )
        )
        world.record_tick_details = False
        config = world.config.reproduction
        genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=-0.8,
        )
        parent = self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                genome,
                config,
            ),
            genome=genome,
        )
        self._place_ready_agent(
            world,
            x=2,
            y=3,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                genome,
                config,
            ),
            genome=genome,
        )

        self.assertTrue(runtime_reproduction.reproduce(world, parent))

        self.assertEqual(world.tick_reproduction_mate_search_events, [])
        self.assertEqual(
            world.run_reproduction_mate_search_counts[
                "constraint_expression_incompatible"
            ],
            1,
        )

    def test_stage1_same_group_sexual_reproduction_uses_local_partner(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=5,
                max_agents=20,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    energy_cost=0.2,
                    sexual_partner_radius=1,
                ),
            )
        )
        genome = self._sexualized_genome(self._mixed_genome())
        parent = self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=genome,
        )
        partner = self._place_ready_agent(
            world,
            x=2,
            y=3,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=genome,
        )
        parent_energy = parent.energy
        partner_energy = partner.energy

        self.assertTrue(runtime_reproduction.reproduce(world, parent))

        child = next(
            agent
            for agent in world.agents.values()
            if agent.parent_id == parent.agent_id
        )
        parent_cost = world._sexual_reproduction_energy_cost(
            world._trophic_profile(parent)
        )
        partner_cost = world._sexual_reproduction_energy_cost(
            world._trophic_profile(partner)
        )
        event = world.events[-1].to_dict()
        summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)

        self.assertEqual(child.secondary_parent_id, partner.agent_id)
        self.assertEqual(child.reproductive_group_id, 1)
        self.assertEqual(parent.last_reproduction_tick, world.tick)
        self.assertEqual(partner.last_reproduction_tick, world.tick)
        self.assertAlmostEqual(parent.energy, parent_energy - parent_cost)
        self.assertAlmostEqual(partner.energy, partner_energy - partner_cost)
        self.assertEqual(
            event["data"]["reproduction_mode"],
            SEXUAL_REPRODUCTION_MODE,
        )
        self.assertEqual(event["data"]["parent_ids"], [parent.agent_id, partner.agent_id])
        self.assertEqual(event["data"]["schema_version"], REPRODUCTION_EVENT_SCHEMA_VERSION)
        self.assertEqual(event["data"]["parent_lineage_ids"], [1, 1])
        self.assertEqual(event["data"]["parent_reproductive_group_ids"], [1, 1])
        self.assertEqual(event["data"]["child_lineage_id"], child.lineage_id)
        self.assertEqual(event["data"]["child_reproductive_group_id"], 1)
        self.assertEqual(event["data"]["child_reproductive_stage"], child.reproductive_stage)
        self.assertEqual(
            event["data"]["child_reproductive_expression"],
            child.reproductive_expression,
        )
        self.assertEqual(event["data"]["offspring_count"], 1)
        self.assertEqual(event["data"]["mate_distance"], 1)
        self.assertIsNotNone(event["data"]["compatibility_score"])
        self.assertEqual(len(event["data"]["parent_energy_costs"]), 2)
        self.assertEqual(
            event["data"]["parent_energy_costs"][0],
            {"agent_id": parent.agent_id, "energy_cost": round(parent_cost, 4)},
        )
        self.assertEqual(
            event["data"]["parent_energy_costs"][1],
            {"agent_id": partner.agent_id, "energy_cost": round(partner_cost, 4)},
        )
        self.assertEqual(
            event["data"]["mind_inheritance"]["schema_version"],
            MIND_INHERITANCE_PLACEHOLDER_VERSION,
        )
        self.assertEqual(summary["reproductive_groups_end"]["sexual_births"], 1)
        self.assertEqual(summary["reproductive_groups_end"]["asexual_births"], 0)
        self.assertEqual(
            world.run_reproduction_mate_search_counts["sexual_successes"],
            1,
        )
        self.assertEqual(
            world.tick_reproduction_mate_search_events[0]["selected_partner_id"],
            partner.agent_id,
        )
        self.assertIsNone(
            world.tick_reproduction_mate_search_events[0]["fallback_reason"]
        )

    def test_gated_multi_offspring_sexual_reproduction_emits_sibling_births(self) -> None:
        reproduction = ReproductionConfig(
            min_age=1,
            cooldown_ticks=0,
            min_hydration_fraction=0.0,
            energy_cost=0.1,
            sexual_partner_radius=1,
            multi_offspring_enabled=True,
            multi_offspring_threshold=0.8,
            multi_offspring_max_count=3,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=7,
                max_agents=20,
                reproduction=reproduction,
            )
        )
        sexualized = self._sexualized_genome(self._mixed_genome())
        fecund_genome = replace(
            sexualized,
            reproductive=replace(
                sexualized.reproductive,
                fecundity_potential=1.0,
            ),
        )
        parent = self._place_ready_agent(
            world,
            x=3,
            y=3,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=fecund_genome,
        )
        partner = self._place_ready_agent(
            world,
            x=3,
            y=4,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=fecund_genome,
        )
        parent_energy = parent.energy
        partner_energy = partner.energy
        parent_cost = world._sexual_reproduction_energy_cost(
            world._trophic_profile(parent)
        )
        partner_cost = world._sexual_reproduction_energy_cost(
            world._trophic_profile(partner)
        )

        birth_count = runtime_reproduction.reproduce_birth_count(world, parent)

        children = [
            agent
            for agent in world.agents.values()
            if agent.parent_id == parent.agent_id
            and agent.secondary_parent_id == partner.agent_id
        ]
        events = [
            event.to_dict()
            for event in world.events
            if event.type == EventType.AGENT_REPRODUCED
        ]
        child_ids = sorted(child.agent_id for child in children)
        summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)

        self.assertEqual(birth_count, 3)
        self.assertEqual(world.births, 3)
        self.assertEqual(len(children), 3)
        self.assertEqual(len(events), 3)
        self.assertEqual(
            sorted(child_id for _, child_id in world.tick_birth_pairs),
            child_ids,
        )
        self.assertAlmostEqual(parent.energy, parent_energy - parent_cost * 3)
        self.assertAlmostEqual(partner.energy, partner_energy - partner_cost * 3)
        self.assertEqual(summary["reproductive_groups_end"]["sexual_births"], 3)
        self.assertEqual(
            world.run_reproduction_mate_search_counts["sexual_successes"],
            1,
        )
        for index, event in enumerate(events, start=1):
            data = event["data"]
            self.assertEqual(data["reproduction_mode"], SEXUAL_REPRODUCTION_MODE)
            self.assertEqual(data["offspring_count"], 3)
            self.assertEqual(data["multi_offspring_desired_count"], 3)
            self.assertEqual(data["multi_offspring_actual_count"], 3)
            self.assertEqual(data["multi_offspring_limit_reasons"], [])
            self.assertEqual(data["offspring_index"], index)
            self.assertEqual(data["sibling_child_ids"], child_ids)
            self.assertTrue(data["multi_offspring"])
            self.assertEqual(data["parent_ids"], [parent.agent_id, partner.agent_id])
            self.assertEqual(
                data["parent_energy_costs"][0],
                {"agent_id": parent.agent_id, "energy_cost": round(parent_cost, 4)},
            )
            self.assertEqual(
                data["parent_energy_costs"][1],
                {"agent_id": partner.agent_id, "energy_cost": round(partner_cost, 4)},
            )
            self.assertEqual(
                data["parent_energy_costs_total"][0],
                {"agent_id": parent.agent_id, "energy_cost": round(parent_cost * 3, 4)},
            )
            self.assertEqual(
                data["parent_energy_costs_total"][1],
                {"agent_id": partner.agent_id, "energy_cost": round(partner_cost * 3, 4)},
            )

    def test_reproduction_placement_context_controls_sibling_destinations(self) -> None:
        reproduction = ReproductionConfig(
            min_age=1,
            cooldown_ticks=0,
            min_hydration_fraction=0.0,
            energy_cost=0.0,
            sexual_partner_radius=1,
            multi_offspring_enabled=True,
            multi_offspring_threshold=0.8,
            multi_offspring_max_count=3,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=7,
                max_agents=20,
                reproduction=reproduction,
            )
        )
        sexualized = self._sexualized_genome(self._mixed_genome())
        fecund_genome = replace(
            sexualized,
            reproductive=replace(
                sexualized.reproductive,
                fecundity_potential=1.0,
            ),
        )
        parent = self._place_ready_agent(
            world,
            x=3,
            y=3,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=fecund_genome,
        )
        self._place_ready_agent(
            world,
            x=3,
            y=4,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=fecund_genome,
        )
        original_place_agent = world._place_agent
        original_invalidate = world._invalidate_biotic_state
        placements: list[tuple[int, int]] = []
        invalidations = 0

        def place_child(child: Agent) -> None:
            placements.append((child.x, child.y))
            original_place_agent(child)

        def invalidate_spatial_state() -> None:
            nonlocal invalidations
            invalidations += 1
            original_invalidate()

        placement = replace(
            world._reproduction_placement_context(),
            max_agents=20,
            current_alive_count=lambda: 2,
            find_empty_neighbor=lambda x, y: (4, 3),
            sibling_destination_candidates=lambda x, y: [
                (4, 3),
                (2, 3),
                (3, 2),
                (3, 4),
            ],
            can_place_at=lambda x, y: (x, y) in {(4, 3), (2, 3), (3, 2)},
            place_agent=place_child,
            invalidate_spatial_state=invalidate_spatial_state,
        )

        with (
            patch.object(
                world,
                "_find_empty_neighbor",
                side_effect=AssertionError("placement context was bypassed"),
            ),
            patch.object(
                world,
                "_can_move_to",
                side_effect=AssertionError("placement context was bypassed"),
            ),
            patch.object(
                world,
                "_place_agent",
                side_effect=AssertionError("placement context was bypassed"),
            ),
            patch.object(
                world,
                "_invalidate_biotic_state",
                side_effect=AssertionError("placement context was bypassed"),
            ),
        ):
            birth_count = runtime_reproduction.reproduce_birth_count(
                world,
                parent,
                alive_count=2,
                placement_context=placement,
            )

        events = [
            event.to_dict()
            for event in world.events
            if event.type == EventType.AGENT_REPRODUCED
        ]
        self.assertEqual(birth_count, 3)
        self.assertEqual(placements, [(4, 3), (2, 3), (3, 2)])
        self.assertEqual(invalidations, 1)
        self.assertEqual(len(events), 3)
        self.assertTrue(
            all(
                event["data"]["multi_offspring_limit_reasons"] == []
                for event in events
            )
        )

    def test_multi_offspring_respects_population_capacity(self) -> None:
        reproduction = ReproductionConfig(
            min_age=1,
            cooldown_ticks=0,
            min_hydration_fraction=0.0,
            energy_cost=0.0,
            sexual_partner_radius=1,
            multi_offspring_enabled=True,
            multi_offspring_threshold=0.8,
            multi_offspring_max_count=3,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=7,
                max_agents=3,
                reproduction=reproduction,
            )
        )
        sexualized = self._sexualized_genome(self._mixed_genome())
        fecund_genome = replace(
            sexualized,
            reproductive=replace(
                sexualized.reproductive,
                fecundity_potential=1.0,
            ),
        )
        parent = self._place_ready_agent(
            world,
            x=3,
            y=3,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=fecund_genome,
        )
        self._place_ready_agent(
            world,
            x=3,
            y=4,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=fecund_genome,
        )

        birth_count = runtime_reproduction.reproduce_birth_count(
            world,
            parent,
            alive_count=2,
        )

        event = world.events[-1].to_dict()
        self.assertEqual(birth_count, 1)
        self.assertEqual(world.births, 1)
        self.assertEqual(len(world.alive_agents()), 3)
        self.assertEqual(event["data"]["offspring_count"], 1)
        self.assertEqual(event["data"]["multi_offspring_desired_count"], 3)
        self.assertEqual(event["data"]["multi_offspring_actual_count"], 1)
        self.assertEqual(
            event["data"]["multi_offspring_limit_reasons"],
            ["population_capacity"],
        )
        self.assertNotIn("multi_offspring", event["data"])

    def test_multi_offspring_requires_both_parents_to_qualify(self) -> None:
        reproduction = ReproductionConfig(
            min_age=1,
            cooldown_ticks=0,
            min_hydration_fraction=0.0,
            energy_cost=0.0,
            sexual_partner_radius=1,
            multi_offspring_enabled=True,
            multi_offspring_threshold=0.8,
            multi_offspring_max_count=3,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=7,
                max_agents=20,
                reproduction=reproduction,
            )
        )
        sexualized = self._sexualized_genome(self._mixed_genome())
        fecund_genome = replace(
            sexualized,
            reproductive=replace(
                sexualized.reproductive,
                fecundity_potential=1.0,
            ),
        )
        low_fecundity_genome = replace(
            sexualized,
            reproductive=replace(
                sexualized.reproductive,
                fecundity_potential=0.2,
            ),
        )
        parent = self._place_ready_agent(
            world,
            x=3,
            y=3,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=fecund_genome,
        )
        self._place_ready_agent(
            world,
            x=3,
            y=4,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=low_fecundity_genome,
        )

        birth_count = runtime_reproduction.reproduce_birth_count(world, parent)

        event = world.events[-1].to_dict()
        self.assertEqual(birth_count, 1)
        self.assertEqual(world.births, 1)
        self.assertEqual(event["data"]["offspring_count"], 1)
        self.assertNotIn("multi_offspring", event["data"])

    def test_sexual_child_starting_fraction_blends_parent_profiles(self) -> None:
        plant_profile = self._test_trophic_profile("none")
        hunter_profile = self._test_trophic_profile("hunter")

        forward = runtime_reproduction.sexual_child_starting_fraction(
            0.4,
            1.5,
            plant_profile,
            hunter_profile,
        )
        reverse = runtime_reproduction.sexual_child_starting_fraction(
            0.4,
            1.5,
            hunter_profile,
            plant_profile,
        )

        self.assertAlmostEqual(forward, 0.5)
        self.assertAlmostEqual(reverse, forward)

    def test_sexual_child_stabilization_is_symmetric_for_mixed_parent_modes(self) -> None:
        world = SimulationWorld(self._ready_reproduction_config())
        child_genome = self._mixed_genome()
        plant_profile = self._test_trophic_profile("none")
        hunter_profile = self._test_trophic_profile("hunter")

        forward = runtime_reproduction.sexual_child_stabilized_genome(
            world,
            self._hunter_genome(),
            self._mixed_genome(),
            child_genome,
            hunter_profile,
            plant_profile,
        )
        reverse = runtime_reproduction.sexual_child_stabilized_genome(
            world,
            self._mixed_genome(),
            self._hunter_genome(),
            child_genome,
            plant_profile,
            hunter_profile,
        )

        self.assertEqual(forward.to_dict(), reverse.to_dict())
        self.assertNotEqual(forward.to_dict(), child_genome.to_dict())
        self.assertGreaterEqual(forward.meat_efficiency, child_genome.meat_efficiency)
        self.assertLessEqual(forward.plant_bias, child_genome.plant_bias)
        self.assertGreaterEqual(forward.live_prey_bias, child_genome.live_prey_bias)

    def test_sexual_child_reproductive_state_resolves_symmetrically(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=5,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    energy_cost=0.0,
                    sexual_partner_radius=1,
                ),
            )
        )
        config = world.config.reproduction
        primary_genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=-0.8,
        )
        secondary_genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=0.8,
        )
        child_genome = self._role_genome(
            self._mixed_genome(),
            role_drive=0.55,
            expression_bias=0.8,
        )
        primary = self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=1,
            reproductive_group_id=10,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                primary_genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                primary_genome,
                config,
            ),
            genome=primary_genome,
        )
        secondary = self._place_ready_agent(
            world,
            x=2,
            y=3,
            lineage_id=2,
            reproductive_group_id=10,
            reproductive_stage=runtime_mating.reproductive_stage_for_genome(
                secondary_genome,
                config,
            ),
            reproductive_expression=runtime_mating.reproductive_expression_for_genome(
                secondary_genome,
                config,
            ),
            genome=secondary_genome,
        )

        forward = runtime_reproduction.sexual_reproductive_state_for_child(
            world,
            primary,
            secondary,
            child_genome,
        )
        reverse = runtime_reproduction.sexual_reproductive_state_for_child(
            world,
            secondary,
            primary,
            child_genome,
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(forward.group_id, 10)
        self.assertEqual(
            forward.stage,
            runtime_mating.reproductive_stage_for_genome(child_genome, config),
        )
        self.assertEqual(
            forward.expression,
            runtime_mating.reproductive_expression_for_genome(child_genome, config),
        )

    def test_stage1_cross_lineage_same_group_sexual_child_gets_new_lineage(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=5,
                max_agents=20,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    energy_cost=0.0,
                    sexual_partner_radius=1,
                ),
            )
        )
        genome = self._sexualized_genome(self._mixed_genome())
        parent = self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=1,
            reproductive_group_id=10,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=genome,
        )
        partner = self._place_ready_agent(
            world,
            x=2,
            y=3,
            lineage_id=2,
            reproductive_group_id=10,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=genome,
        )
        expected_child_agent_id = world.next_agent_id

        self.assertTrue(runtime_reproduction.reproduce(world, parent))

        child = world.agents[expected_child_agent_id]
        event = world.events[-1].to_dict()

        self.assertEqual(child.secondary_parent_id, partner.agent_id)
        self.assertEqual(child.lineage_id, expected_child_agent_id)
        self.assertEqual(child.reproductive_group_id, 10)
        self.assertEqual(event["data"]["parent_lineage_ids"], [1, 2])
        self.assertEqual(
            event["data"]["child_lineage_id"],
            expected_child_agent_id,
        )

    def test_stage1_sexual_reproduction_falls_back_without_same_group_partner(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(width=5, height=5, max_agents=20)
        )
        genome = self._sexualized_genome(self._mixed_genome())
        parent = self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=1,
            reproductive_group_id=1,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=genome,
        )
        self._place_ready_agent(
            world,
            x=2,
            y=3,
            lineage_id=2,
            reproductive_group_id=2,
            reproductive_stage=STAGE1_FACULTATIVE_SEX,
            reproductive_expression=SEXUAL_EXPRESSION,
            genome=genome,
        )

        self.assertTrue(world._reproduce(parent))

        child = next(
            agent
            for agent in world.agents.values()
            if agent.parent_id == parent.agent_id
        )
        event = world.events[-1].to_dict()
        summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)

        self.assertIsNone(child.secondary_parent_id)
        self.assertEqual(event["data"]["reproduction_mode"], "asexual")
        self.assertEqual(event["data"]["schema_version"], REPRODUCTION_EVENT_SCHEMA_VERSION)
        self.assertEqual(event["data"]["parent_ids"], [parent.agent_id])
        self.assertEqual(event["data"]["parent_lineage_ids"], [1])
        self.assertEqual(event["data"]["parent_reproductive_group_ids"], [1])
        self.assertEqual(event["data"]["child_lineage_id"], child.lineage_id)
        self.assertEqual(event["data"]["offspring_count"], 1)
        self.assertIsNone(event["data"]["compatibility_score"])
        self.assertIsNone(event["data"]["inbreeding_penalty"])
        self.assertEqual(len(event["data"]["parent_energy_costs"]), 1)
        self.assertEqual(summary["reproductive_groups_end"]["sexual_births"], 0)
        self.assertEqual(summary["reproductive_groups_end"]["asexual_births"], 1)
        self.assertEqual(
            world.run_reproduction_mate_search_counts[
                "fallback_no_same_group_partner"
            ],
            1,
        )
        self.assertEqual(
            world.tick_reproduction_mate_search_events[0]["fallback_reason"],
            "no_same_group_partner",
        )
