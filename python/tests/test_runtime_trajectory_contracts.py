from __future__ import annotations

from python.tests.runtime_test_helpers import *


class RuntimeTrajectoryContractTests(RuntimeContractTestHelpers):
    def test_capture_agent_state_uses_explicit_ratio_context(self) -> None:
        genome = self._hunter_genome()
        agent = Agent(
            agent_id=42,
            parent_id=None,
            lineage_id=3,
            birth_tick=0,
            death_tick=None,
            x=2,
            y=1,
            energy=0.625,
            hydration=0.5,
            health=0.875,
            max_health=genome.max_health,
            injury_load=0.0,
            age=12,
            alive=True,
            last_reproduction_tick=-10_000,
            last_damage_source="none",
            recent_plant_energy=0.0,
            recent_fresh_kill_energy=0.0,
            recent_carcass_energy=0.0,
            genome_vector=genome_vector(genome),
            genome=genome,
        )
        seen_agent_ids: list[int] = []

        def ratio(value: float):
            def _ratio(candidate: Agent) -> float:
                seen_agent_ids.append(candidate.agent_id)
                return value

            return _ratio

        state = capture_agent_state(
            agent,
            context=TrajectoryStateContext(
                energy_ratio=ratio(0.625),
                hydration_ratio=ratio(0.5),
                health_ratio=ratio(0.875),
            ),
        )

        self.assertEqual(state["x"], 2)
        self.assertEqual(state["y"], 1)
        self.assertEqual(state["energy_ratio"], 0.625)
        self.assertEqual(state["hydration_ratio"], 0.5)
        self.assertEqual(state["health_ratio"], 0.875)
        self.assertEqual(seen_agent_ids, [42, 42, 42])

    def test_full_replay_records_mind_trajectory_contract(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()

        trajectory = result.viewer["trajectory"]
        records = trajectory["records"]
        first_record = records[0]

        self.assertEqual(result.summary["mind_contracts"]["schema_version"], TRAJECTORY_SCHEMA_VERSION)
        self.assertEqual(trajectory["schema_version"], TRAJECTORY_SCHEMA_VERSION)
        self.assertEqual(
            result.summary["mind_contracts"]["policy_interface_version"],
            POLICY_INTERFACE_VERSION,
        )
        self.assertEqual(trajectory["policy_interface_version"], POLICY_INTERFACE_VERSION)
        self.assertEqual(trajectory["action_contract_version"], ACTION_CONTRACT_VERSION)
        self.assertEqual(
            result.summary["mind_contracts"]["action_contract_version"],
            ACTION_CONTRACT_VERSION,
        )
        self.assertEqual(
            trajectory["reproductive_group_contract_version"],
            REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        )
        self.assertEqual(
            result.summary["mind_contracts"]["reproductive_group_contract_version"],
            REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        )
        self.assertEqual(
            trajectory["genome_recombination_contract_version"],
            GENOME_RECOMBINATION_CONTRACT_VERSION,
        )
        self.assertEqual(
            result.summary["mind_contracts"]["genome_recombination_contract_version"],
            GENOME_RECOMBINATION_CONTRACT_VERSION,
        )
        self.assertIn(MATE_ACTION, trajectory["action_contract"]["reserved_action_keys"])
        self.assertEqual(
            trajectory["reproductive_group_contract"]["schema_version"],
            REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        )
        self.assertEqual(
            trajectory["genome_recombination_contract"]["schema_version"],
            GENOME_RECOMBINATION_CONTRACT_VERSION,
        )
        self.assertEqual(
            trajectory["observation_contract"]["schema_version"],
            OBSERVATION_SCHEMA_VERSION,
        )
        self.assertFalse(trajectory["observation_contract"]["privileged_world_state"])
        self.assertGreater(trajectory["record_count"], 0)
        self.assertEqual(trajectory["record_count"], len(records))
        self.assertEqual(first_record["observation_schema"], OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(
            first_record["observation_metadata"],
            {"agent_id": first_record["agent_id"]},
        )
        self.assertEqual(
            first_record["observation_input"]["encoder_version"],
            OBSERVATION_ENCODER_VERSION,
        )
        self.assertEqual(
            first_record["observation_input"]["shape"],
            [OBSERVATION_INPUT_VECTOR_SIZE],
        )
        self.assertEqual(
            len(decode_observation_input(first_record["observation_input"])),
            OBSERVATION_INPUT_VECTOR_SIZE,
        )
        self.assertEqual(
            trajectory["reward_contract"]["schema_version"],
            REWARD_SCHEMA_VERSION,
        )
        self.assertIn(
            "invalid_action_penalty",
            trajectory["reward_contract"]["component_bounds"],
        )
        self.assertEqual(
            trajectory["action_outcome_schema_version"],
            ACTION_OUTCOME_SCHEMA_VERSION,
        )
        self.assertEqual(
            result.summary["mind_contracts"]["action_outcome_schema_version"],
            ACTION_OUTCOME_SCHEMA_VERSION,
        )
        self.assertIn("invalid_observation_action_count", trajectory)
        self.assertIn("invalid_resolution_action_count", trajectory)
        self.assertEqual(
            trajectory["invalid_action_count"],
            trajectory["invalid_observation_action_count"],
        )
        self.assertEqual(
            result.summary["mind_contracts"]["invalid_observation_action_count"],
            trajectory["invalid_observation_action_count"],
        )
        self.assertEqual(
            result.summary["mind_contracts"]["invalid_resolution_action_count"],
            trajectory["invalid_resolution_action_count"],
        )
        self.assertIn(first_record["requested_action"], first_record["action_mask"])
        self.assertIn(first_record["requested_action"], first_record["resolution_action_mask"])
        self.assertFalse(first_record["action_mask"][MATE_ACTION])
        self.assertEqual(first_record["policy_id"], OBSERVATION_HEURISTIC_POLICY_ID)
        self.assertEqual(
            first_record["policy_version"],
            OBSERVATION_HEURISTIC_POLICY_VERSION,
        )
        self.assertIn("resolution_action_valid", first_record)
        self.assertEqual(first_record["outcome"]["schema_version"], ACTION_OUTCOME_SCHEMA_VERSION)
        self.assertIn("signal", first_record["outcome"])
        self.assertFalse(first_record["outcome"]["signal"]["emitted"])
        self.assertIn("resource_gain", first_record["outcome"])
        self.assertEqual(first_record["reward"]["schema_version"], REWARD_SCHEMA_VERSION)
        self.assertIn("invalid_action_penalty", first_record["reward"]["components"])
        self.assertEqual(result.summary["mind_contracts"]["record_count"], len(records))

    def test_species_centroid_units_are_explicit(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=40)).run()
        species_catalog = result.viewer["species_catalog"]
        ecotype_catalog = result.viewer["ecotype_catalog"]

        self.assertTrue(species_catalog)
        for entry in species_catalog.values():
            self.assertEqual(entry["identity_mode"], REPLAY_TAXONOMY_MODE)
            self.assertEqual(entry["centroid_units"], "raw_gene_values")
            self.assertEqual(
                entry["normalized_centroid_units"],
                "unit_interval_by_gene_limits",
            )
            for gene, value in entry["centroid"].items():
                lower, upper = GENE_LIMITS[gene]
                self.assertGreaterEqual(value, lower, msg=gene)
                self.assertLessEqual(value, upper, msg=gene)
            for gene, value in entry["normalized_centroid"].items():
                self.assertIn(gene, GENE_LIMITS)
                self.assertGreaterEqual(value, 0.0, msg=gene)
                self.assertLessEqual(value, 1.0, msg=gene)

        self.assertTrue(ecotype_catalog)
        for entry in ecotype_catalog.values():
            self.assertEqual(entry["identity_mode"], "frame_local_genome_cluster")
            self.assertEqual(entry["centroid_units"], "raw_gene_values")

    def test_run_trajectory_exposes_metabolism_after_state(self) -> None:
        class FixedPolicy:
            policy_id = "fixed_policy"
            policy_version = "fixed_policy_v1"

            def __init__(self, action: str) -> None:
                self.action = action

            def decide(
                self,
                observation: dict[str, object],
                action_mask: dict[str, bool],
            ) -> ActionDecision:
                return ActionDecision(
                    requested_action=self.action,
                    source=self.policy_id,
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                )

        def run_one(action: str) -> dict[str, object]:
            world = SimulationWorld(
                WorldConfig(
                    seed=7,
                    max_ticks=1,
                    width=3,
                    height=3,
                    initial_agents=0,
                    max_agents=4,
                    water_tile_ratio=0.0,
                    forest_tile_ratio=0.0,
                    wetland_tile_ratio=0.0,
                    rocky_tile_ratio=0.0,
                    base_energy_drain=0.02,
                    base_hydration_drain=0.02,
                    hazards=HazardConfig(
                        exposure_damage_rate=0.0,
                        instability_damage_rate=0.0,
                    ),
                    reproduction=ReproductionConfig(min_age=720),
                ),
                policy=FixedPolicy(action),
            )
            agent = self._place_ready_agent(
                world,
                x=1,
                y=1,
                genome=self._hunter_genome(),
            )
            agent.energy = agent.genome.max_energy * 0.8
            agent.hydration = agent.genome.max_hydration * 0.8
            world.current_species_map = {agent.agent_id: agent.lineage_id}
            world.agent_last_species_map = world.current_species_map.copy()

            result = world.run(mode=RunMode.FULL_REPLAY, record_trajectory=True)
            return result.viewer["trajectory"]["records"][0]

        stationary = run_one("stay")
        moved = run_one("move_east")
        stationary_energy_loss = (
            float(stationary["before"]["energy"]) - float(stationary["after"]["energy"])
        )
        stationary_hydration_loss = (
            float(stationary["before"]["hydration"])
            - float(stationary["after"]["hydration"])
        )
        moved_energy_loss = (
            float(moved["before"]["energy"]) - float(moved["after"]["energy"])
        )
        moved_hydration_loss = (
            float(moved["before"]["hydration"]) - float(moved["after"]["hydration"])
        )

        self.assertFalse(stationary["moved"])
        self.assertTrue(moved["moved"])
        self.assertGreater(stationary_energy_loss, 0.0)
        self.assertGreater(stationary_hydration_loss, 0.0)
        self.assertGreater(moved_energy_loss, stationary_energy_loss)
        self.assertGreater(moved_hydration_loss, stationary_hydration_loss)
        self.assertEqual(stationary["after"]["age"], stationary["before"]["age"] + 1)
        self.assertEqual(moved["after"]["age"], moved["before"]["age"] + 1)

    def test_reward_components_are_versioned_and_bounded(self) -> None:
        before = {
            "energy_ratio": 1.4,
            "hydration_ratio": 0.9,
            "health_ratio": 1.0,
        }
        after = {
            "energy_ratio": -0.2,
            "hydration_ratio": 1.8,
            "health_ratio": -0.4,
        }

        reward = build_reward(
            before=before,
            after=after,
            action_valid=False,
            moved=True,
            resource_gain=4.5,
            reproduced=True,
            died=True,
            reproduction_ready_after=True,
        )

        contract = reward_contract()
        self.assertEqual(reward["schema_version"], REWARD_SCHEMA_VERSION)
        for name, value in reward["components"].items():
            lower, upper = contract["component_bounds"][name]
            self.assertGreaterEqual(value, lower, msg=name)
            self.assertLessEqual(value, upper, msg=name)
        total_lower, total_upper = contract["total_bounds"]
        self.assertGreaterEqual(reward["total"], total_lower)
        self.assertLessEqual(reward["total"], total_upper)
        self.assertEqual(reward["components"]["invalid_action_penalty"], -0.05)
        self.assertEqual(reward["components"]["movement_cost"], -0.005)
        self.assertEqual(reward["components"]["resource_acquisition"], 1.0)
        self.assertEqual(reward["components"]["survival_continuation"], -1.0)
        self.assertEqual(reward["components"]["reproduction_success"], 1.0)

    def test_trajectory_attack_outcome_records_target_damage_and_kill(self) -> None:
        result, attacker_id, _target_id = self._run_scripted_lethal_attack()
        attack_records = [
            record
            for record in result.viewer["trajectory"]["records"]
            if record["agent_id"] == attacker_id and record["outcome"]["attack"]["attempted"]
        ]

        self.assertTrue(attack_records)
        lethal_records = [record for record in attack_records if record["outcome"]["attack"]["kill"]]
        self.assertTrue(lethal_records)
        attack = lethal_records[0]["outcome"]["attack"]
        self.assertIsInstance(attack["target_id"], int)
        self.assertGreater(attack["damage"], 0)
        self.assertTrue(attack["success"])
        if attack["immediate_kill_feed"]:
            self.assertTrue(lethal_records[0]["outcome"]["feeding"]["ate"])
            self.assertEqual(
                lethal_records[0]["outcome"]["feeding"]["food_source"],
                "fresh_kill",
            )

    def test_trajectory_records_passive_killed_before_action(self) -> None:
        result, _attacker_id, target_id = self._run_scripted_lethal_attack()
        passive_records = [
            record
            for record in result.viewer["trajectory"]["records"]
            if (
                record["agent_id"] == target_id
                and record["outcome"]["passive"]["died_before_action"]
            )
        ]

        self.assertTrue(passive_records)
        record = passive_records[0]
        passive = record["outcome"]["passive"]
        self.assertEqual(record["action_source"], "passive")
        self.assertEqual(record["requested_action"], "stay")
        self.assertFalse(passive["acted"])
        self.assertTrue(passive["killed"])
        self.assertEqual(passive["death_cause"], "attack")
        self.assertIsInstance(passive["killer_id"], int)
        self.assertGreater(passive["attack_damage_taken"], 0)

    def test_trajectory_records_passive_death_after_action(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=40)).run()
        records = [
            record
            for record in result.viewer["trajectory"]["records"]
            if record["outcome"]["passive"]["died_after_action"]
        ]

        self.assertTrue(records)
        passive = records[0]["outcome"]["passive"]
        self.assertNotEqual(records[0]["action_source"], "passive")
        self.assertTrue(passive["acted"])
        self.assertTrue(passive["killed"])
        self.assertIsNotNone(passive["death_cause"])

    def test_trajectory_feeding_outcome_records_source_tile_and_gain(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()
        feeding_records = [
            record
            for record in result.viewer["trajectory"]["records"]
            if record["outcome"]["feeding"]["ate"]
        ]

        self.assertTrue(feeding_records)
        feeding = feeding_records[0]["outcome"]["feeding"]
        self.assertIn(feeding["food_source"], {"plant", "fresh_kill", "carcass"})
        self.assertIsInstance(feeding["x"], int)
        self.assertIsInstance(feeding["y"], int)
        self.assertGreater(feeding["consumed"], 0)
        self.assertGreaterEqual(feeding["gained_energy"], 0)
        self.assertEqual(
            feeding_records[0]["outcome"]["resource_gain"],
            feeding["gained_energy"],
        )

    def test_trajectory_credits_both_parents_for_single_sexual_birth(self) -> None:
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
        world.current_species_map = {
            parent.agent_id: parent.lineage_id,
            partner.agent_id: partner.lineage_id,
        }
        world.agent_last_species_map = world.current_species_map.copy()

        def choose_scripted_action(
            agent: Agent,
            observation: dict[str, object] | None = None,
        ) -> str:
            world._policy_action_source = "scripted_stay"
            world._policy_id = "scripted_stay"
            world._policy_version = "scripted_stay_v1"
            return "stay"

        with patch.object(world, "_choose_action", side_effect=choose_scripted_action):
            result = world.run(mode=RunMode.FULL_REPLAY, record_trajectory=True)

        reproduced_events = [
            event
            for event in result.events
            if event["type"] == EventType.AGENT_REPRODUCED.value
        ]
        records = {
            record["agent_id"]: record
            for record in result.viewer["trajectory"]["records"]
            if record["agent_id"] in {parent.agent_id, partner.agent_id}
        }

        self.assertEqual(len(reproduced_events), 1)
        self.assertEqual(
            reproduced_events[0]["data"]["parent_ids"],
            [parent.agent_id, partner.agent_id],
        )
        self.assertEqual(set(records), {parent.agent_id, partner.agent_id})
        for agent_id in (parent.agent_id, partner.agent_id):
            self.assertTrue(records[agent_id]["outcome"]["reproduced"])
            self.assertEqual(
                records[agent_id]["reward"]["components"]["reproduction_success"],
                1.0,
            )

    def test_trajectory_records_real_tick_resolution_conflict(self) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                width=3,
                height=3,
                initial_agents=0,
                max_agents=10,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
                base_energy_drain=0.0,
                base_hydration_drain=0.0,
                reproduction=ReproductionConfig(min_age=720),
            )
        )
        genome = Genome.sample_initial(world.rng)
        first = self._place_ready_agent(world, x=0, y=1, lineage_id=1, genome=genome)
        second = self._place_ready_agent(world, x=1, y=2, lineage_id=2, genome=genome)
        first.age = 1
        second.age = 1
        requested_actions = {
            first.agent_id: "move_east",
            second.agent_id: "move_north",
        }

        def choose_scripted_action(
            agent: Agent,
            observation: dict[str, object] | None = None,
        ) -> str:
            world._policy_action_source = "scripted_conflict"
            world._policy_id = "scripted_conflict"
            world._policy_version = "scripted_conflict_v1"
            return requested_actions[agent.agent_id]

        with patch.object(world, "_choose_action", side_effect=choose_scripted_action):
            result = world.run(mode=RunMode.FULL_REPLAY)

        records = {
            record["agent_id"]: record
            for record in result.viewer["trajectory"]["records"]
            if record["agent_id"] in requested_actions
        }

        first_record = records[first.agent_id]
        second_record = records[second.agent_id]
        self.assertTrue(first_record["resolution_action_valid"])
        self.assertEqual(first_record["resolved_action"], "move_east")
        self.assertTrue(second_record["action_valid"])
        self.assertFalse(second_record["resolution_action_valid"])
        self.assertEqual(second_record["resolved_action"], "stay")
        self.assertEqual(second_record["action_source"], "scripted_conflict")
        self.assertEqual(
            second_record["outcome"]["invalid_reason"],
            "not_in_resolution_action_mask",
        )
        self.assertEqual(result.viewer["trajectory"]["invalid_action_count"], 0)
        self.assertEqual(
            result.viewer["trajectory"]["invalid_observation_action_count"],
            0,
        )
        self.assertEqual(
            result.viewer["trajectory"]["invalid_resolution_action_count"],
            1,
        )
        self.assertEqual(
            result.summary["mind_contracts"]["invalid_resolution_action_count"],
            1,
        )

    def test_summary_only_does_not_retain_trajectory_bookkeeping(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=4))

        result = world.run(mode=RunMode.SUMMARY_ONLY)

        self.assertIsNone(result.viewer)
        self.assertEqual(world.trajectory_records, [])
        self.assertEqual(world.tick_trajectory_records, [])
        self.assertTrue(world.record_trajectory)

    def test_summary_only_can_record_trajectory_without_replay_surfaces(self) -> None:
        with patch(
            "evolution_sim.env.runtime.collectors.apply_replay_taxonomy",
            side_effect=AssertionError("taxonomy should not run in trajectory summary mode"),
        ), patch.object(
            SimulationWorld,
            "_capture_frame",
            side_effect=AssertionError("trajectory summary mode should not capture frames"),
        ), patch(
            "evolution_sim.env.runtime.frames.capture_frame",
            side_effect=AssertionError("trajectory summary mode should not build frame payloads"),
        ), patch.object(
            SimulationWorld,
            "_build_viewer_payload",
            side_effect=AssertionError("trajectory summary mode should not build viewer payload"),
        ):
            world = SimulationWorld(
                WorldConfig(
                    seed=7,
                    max_ticks=4,
                    signals=SignalConfig(
                        communication_token_count=2,
                        communication_profiles_per_token=3,
                    ),
                )
            )
            result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)

        self.assertIsNone(result.viewer)
        self.assertIsNone(result.events)
        self.assertEqual(tuple(result.summary), SHARED_SUMMARY_FIELDS)
        self.assertGreater(len(world.trajectory_records), 0)
        self.assertEqual(world.viewer_frames, [])
        self.assertIn("observation_input", world.trajectory_records[0])
        self.assertEqual(
            world.trajectory_records[0]["observation_input"]["encoder_version"],
            OBSERVATION_ENCODER_VERSION,
        )

    def test_streaming_trajectory_sink_avoids_replay_and_record_retention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trajectory.jsonl.gz"
            writer = JsonlTrajectoryWriter(output_path)
            world = SimulationWorld(
                WorldConfig(
                    seed=7,
                    max_ticks=4,
                    signals=SignalConfig(
                        communication_token_count=2,
                        communication_profiles_per_token=3,
                    ),
                )
            )

            result = world.run(mode=RunMode.SUMMARY_ONLY, trajectory_sink=writer)

            with gzip.open(output_path, "rt", encoding="utf-8") as handle:
                lines = [json.loads(line) for line in handle]

        header = lines[0]
        footer = lines[-1]
        records = [line["record"] for line in lines if line["type"] == "record"]
        self.assertIsNone(result.viewer)
        self.assertIsNone(result.events)
        self.assertEqual(world.viewer_frames, [])
        self.assertEqual(world.trajectory_records, [])
        self.assertGreater(writer.record_count, 0)
        self.assertEqual(writer.record_count, len(records))
        self.assertEqual(
            set(header),
            {
                "type",
                "format",
                "compression",
                "run_id",
                "config",
                "trajectory_contract",
                "provenance",
            },
        )
        self.assertEqual(header["type"], "header")
        self.assertEqual(header["format"], "evolution_sim_trajectory_jsonl_v1")
        self.assertEqual(header["compression"], "gzip")
        self.assertIsInstance(header["config"], dict)
        self.assertNotIn("viewer", header)
        self.assertNotIn("events", header)
        self.assertEqual(
            header["trajectory_contract"]["schema_version"],
            TRAJECTORY_SCHEMA_VERSION,
        )
        self.assertEqual(header["provenance"]["source_seeds"], [])
        self.assertEqual(header["provenance"]["split_id"], "unsplit")
        self.assertIsNone(header["provenance"]["record_count"])
        self.assertIn("config_digest", header["provenance"])
        self.assertIn("contract_digest", header["provenance"])
        self.assertEqual(
            header["trajectory_contract"]["policy_interface_version"],
            POLICY_INTERFACE_VERSION,
        )
        self.assertEqual(
            header["trajectory_contract"]["action_contract_version"],
            ACTION_CONTRACT_VERSION,
        )
        self.assertEqual(
            header["trajectory_contract"]["action_contract"]["schema_version"],
            ACTION_CONTRACT_VERSION,
        )
        self.assertEqual(
            header["trajectory_contract"]["observation_contract"]["action_contract"][
                "schema_version"
            ],
            ACTION_CONTRACT_VERSION,
        )
        self.assertEqual(
            header["trajectory_contract"]["reproductive_group_contract_version"],
            REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        )
        self.assertEqual(
            header["trajectory_contract"]["genome_recombination_contract_version"],
            GENOME_RECOMBINATION_CONTRACT_VERSION,
        )
        self.assertEqual(
            header["trajectory_contract"]["observation_contract"]["signal_contract"][
                "communication_token_count"
            ],
            2,
        )
        self.assertEqual(
            len(
                header["trajectory_contract"]["observation_contract"][
                    "signal_contract"
                ]["reserved_profiles"]
            ),
            6,
        )
        self.assertEqual(
            set(footer),
            {"type", "summary", "trajectory_summary", "provenance"},
        )
        self.assertEqual(footer["type"], "footer")
        self.assertNotIn("viewer", footer)
        self.assertNotIn("events", footer)
        self.assertEqual(
            footer["trajectory_summary"]["schema_version"],
            TRAJECTORY_SCHEMA_VERSION,
        )
        self.assertEqual(
            footer["trajectory_summary"]["policy_interface_version"],
            POLICY_INTERFACE_VERSION,
        )
        self.assertEqual(
            footer["trajectory_summary"]["action_contract_version"],
            ACTION_CONTRACT_VERSION,
        )
        self.assertEqual(
            footer["trajectory_summary"]["reproductive_group_contract_version"],
            REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        )
        self.assertEqual(
            footer["trajectory_summary"]["genome_recombination_contract_version"],
            GENOME_RECOMBINATION_CONTRACT_VERSION,
        )
        self.assertEqual(
            footer["trajectory_summary"]["action_outcome_schema_version"],
            ACTION_OUTCOME_SCHEMA_VERSION,
        )
        self.assertEqual(footer["trajectory_summary"]["record_count"], len(records))
        self.assertEqual(footer["provenance"]["record_count"], len(records))
        self.assertEqual(
            footer["provenance"]["config_digest"],
            header["provenance"]["config_digest"],
        )
        self.assertEqual(
            footer["provenance"]["contract_digest"],
            header["provenance"]["contract_digest"],
        )
        self.assertIn(
            "invalid_observation_action_count",
            footer["trajectory_summary"],
        )
        self.assertIn(
            "invalid_resolution_action_count",
            footer["trajectory_summary"],
        )
        self.assertEqual(footer["summary"]["run_id"], result.summary["run_id"])
        self.assertIn("observation_input", records[0])
        self.assertEqual(
            len(decode_observation_input(records[0]["observation_input"])),
            OBSERVATION_INPUT_VECTOR_SIZE,
        )

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

    def test_derived_tile_memo_uses_explicit_context_callables(self) -> None:
        calls = {
            "effective": 0,
            "water": 0,
            "soft_refuge": 0,
            "refuge": 0,
            "hazard": 0,
            "biotic": 0,
        }
        biotic_state = BioticFieldState(
            prey_biomass=[[0.1]],
            carrion=[[0.2]],
            predator_risk=[[0.3]],
        )

        def effective_fields(x: int, y: int) -> tuple[float, float, float]:
            calls["effective"] += 1
            return (float(x), float(y), 0.5)

        def water_reason(x: int, y: int) -> str:
            calls["water"] += 1
            return "wetland"

        def soft_refuge_reason(x: int, y: int) -> str:
            calls["soft_refuge"] += 1
            return "canopy_refuge"

        def refuge_score(x: int, y: int) -> float:
            calls["refuge"] += 1
            return 0.9

        def hazard(x: int, y: int) -> tuple[str, float]:
            calls["hazard"] += 1
            return ("exposure", 0.4)

        def current_biotic_state() -> BioticFieldState:
            calls["biotic"] += 1
            return biotic_state

        memo = DerivedTileMemo(
            season="wet",
            climate_state={"name": "wet"},
            effective_fields_for=effective_fields,
            water_reason_for=water_reason,
            soft_refuge_reason_for=soft_refuge_reason,
            refuge_score_for=refuge_score,
            hazard_for=hazard,
            current_biotic_state_for=current_biotic_state,
        )

        self.assertEqual(memo.effective_fields(2, 3), (2.0, 3.0, 0.5))
        self.assertEqual(memo.effective_fields(2, 3), (2.0, 3.0, 0.5))
        self.assertEqual(memo.water_reason(2, 3), "wetland")
        self.assertEqual(memo.water_reason(2, 3), "wetland")
        self.assertEqual(memo.soft_refuge_reason(2, 3), "canopy_refuge")
        self.assertEqual(memo.soft_refuge_reason(2, 3), "canopy_refuge")
        self.assertEqual(memo.refuge_score(2, 3), 0.9)
        self.assertEqual(memo.refuge_score(2, 3), 0.9)
        self.assertEqual(memo.hazard(2, 3), ("exposure", 0.4))
        self.assertEqual(memo.hazard(2, 3), ("exposure", 0.4))
        self.assertIs(memo.current_biotic_state(), biotic_state)
        self.assertIs(memo.current_biotic_state(), biotic_state)
        self.assertEqual(
            calls,
            {
                "effective": 1,
                "water": 1,
                "soft_refuge": 1,
                "refuge": 1,
                "hazard": 1,
                "biotic": 1,
            },
        )

    def test_trophic_profile_cache_uses_consistent_raw_genome_key(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]

        self.assertNotEqual(genome_profile_key(agent.genome), agent.genome_vector)
        with patch.object(
            world,
            "_compute_trophic_profile_for_genome",
            wraps=world._compute_trophic_profile_for_genome,
        ) as compute_profile:
            first = world._trophic_profile_for_genome(agent.genome)
            second = world._trophic_profile(agent)
            third = world._trophic_profile(agent)

        self.assertIs(first, second)
        self.assertIs(second, third)
        self.assertEqual(compute_profile.call_count, 1)

    def test_effective_tile_fields_is_current_tick_only(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))

        self.assertEqual(
            tuple(signature(world._effective_tile_fields).parameters),
            ("x", "y"),
        )
        self.assertEqual(world._effective_tile_fields(0, 0), world._effective_tile_fields(0, 0))

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

        diffusion_context = world._biotic_diffusion_context()
        actual = diffuse_biotic_field(sources, context=diffusion_context)
        sparse_sources = {
            y * world.config.width + x: source
            for y, row in enumerate(sources)
            for x, source in enumerate(row)
            if source > 1e-9
        }
        sparse_actual = diffuse_sparse_biotic_field(
            sparse_sources,
            context=diffusion_context,
        )
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
