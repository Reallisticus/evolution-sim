from __future__ import annotations

from python.tests.runtime_test_helpers import *
from evolution_sim.env.contracts import (
    SUMMARY_SCHEMA_VERSION,
    TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION,
)
from evolution_sim.env.runtime.observations import (
    TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION,
    TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
)
from evolution_sim.env.runtime.signals import (
    COMMUNICATION_AGGREGATE_PROJECTION,
    COMMUNICATION_GLOBAL_REPORTING_POLICY,
    COMMUNICATION_RECEIVER_OBSERVATION_POLICY,
    COMMUNICATION_RECEIVER_PROJECTION_SCHEMA_VERSION,
    TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION,
    parse_communication_token_field_name,
)
from evolution_sim.env.runtime.trajectory import (
    TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION,
)
from evolution_sim.mind.policy_inputs import (
    TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ecological_policy_input_values,
)


class _AlwaysOpaqueSignalPolicy:
    policy_id = "test_always_opaque_signal"
    policy_version = "test_always_opaque_signal_v1"

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        action = (
            "signal_0_profile_0"
            if action_mask.get("signal_0_profile_0", False)
            else "stay"
        )
        return ActionDecision(
            requested_action=action,
            source=self.policy_id,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
        )


class RuntimeSignalContractTests(RuntimeContractTestHelpers):
    def test_starving_scavenger_moves_to_nearby_low_signal_carrion_before_plant(
        self,
    ) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 1, "dy": 0, "distance": 1, "strength": 0.0}

        observation = self._policy_observation(
            energy_ratio=0.28,
            hydration_ratio=0.7,
            navigation=navigation,
            center_food=0.9,
            trophic_role="carnivore",
            meat_mode="scavenger",
        )
        observation["local_patch"].append(
            self._policy_cell(
                1,
                0,
                carcass_energy=0.001,
                carrion_signal=0.001,
            )
        )

        decision = ObservationHeuristicPolicy().decide(
            observation,
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_starving_scavenger_ignores_weak_signal_only_carrion(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 1, "dy": 0, "distance": 1, "strength": 0.03}
        navigation["plant"] = {"dx": -1, "dy": 0, "distance": 1, "strength": 0.8}

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.28,
                hydration_ratio=0.7,
                navigation=navigation,
                trophic_role="carnivore",
                meat_mode="scavenger",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_west")

    def test_action_outcome_v2_completes_partial_signal_metadata(self) -> None:
        outcome = complete_action_outcome(
            {
                "schema_version": "stale",
                "requested_action": "stay",
                "resolved_action": "stay",
                "observation_action_valid": True,
                "resolution_action_valid": True,
                "signal": {"emitted": False},
            },
            resource_gain=0.0,
            reproduced=False,
            died=False,
            reproduction_ready_after=False,
        )

        self.assertEqual(outcome["schema_version"], ACTION_OUTCOME_SCHEMA_VERSION)
        self.assertEqual(
            outcome["signal"],
            {
                "emitted": False,
                "token_id": None,
                "profile_index": None,
                "intensity": 0.0,
                "radius": 0,
                "duration_ticks": 0,
                "decay_rate": 0.0,
                "energy_cost": 0.0,
                "invalid_reason": None,
            },
        )

    def test_action_contract_tracks_signal_config_reserved_communication_slots(
        self,
    ) -> None:
        signal_config = SignalConfig(
            communication_token_count=2,
            communication_profiles_per_token=3,
        )
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1, signals=signal_config))
        agent = world.alive_agents()[0]
        mask = build_action_mask(world._action_mask_context(agent))
        contract = action_contract(signal_config)

        expected_communication_actions = [
            "signal_0_profile_0",
            "signal_0_profile_1",
            "signal_0_profile_2",
            "signal_1_profile_0",
            "signal_1_profile_1",
            "signal_1_profile_2",
        ]

        self.assertEqual(contract["communication"]["token_count"], 2)
        self.assertEqual(contract["communication"]["profiles_per_token"], 3)
        self.assertEqual(
            contract["communication"]["action_keys"],
            expected_communication_actions,
        )
        self.assertEqual(
            contract["reserved_action_keys"],
            [MATE_ACTION, *expected_communication_actions],
        )
        self.assertEqual(set(mask), set(action_names(signal_config)))
        for action in expected_communication_actions:
            self.assertIn(action, mask)
            self.assertFalse(mask[action], msg=action)
        self.assertNotIn("signal_2_profile_0", mask)

    def test_signal_actions_stay_inactive_when_signal_substrate_disabled(self) -> None:
        signal_config = SignalConfig(
            enabled=False,
            communication_signal_emission_enabled=True,
            communication_token_count=2,
            communication_profiles_per_token=2,
        )
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1, signals=signal_config))
        agent = world.alive_agents()[0]

        contract = action_contract(signal_config)
        signal_contract = runtime_signals.signal_contract(signal_config)
        mask = build_action_mask(world._action_mask_context(agent))
        signal_specs = [
            action
            for action in contract["actions"]
            if str(action["key"]).startswith("signal_")
        ]

        self.assertFalse(contract["communication"]["emission_enabled"])
        self.assertFalse(signal_contract["signal_substrate_enabled"])
        self.assertFalse(signal_contract["reproductive_signal_emission_enabled"])
        self.assertFalse(signal_contract["communication_signal_emission_enabled"])
        self.assertEqual(
            signal_contract["reproductive_readiness_profile"]["intensity"],
            0.0,
        )
        self.assertEqual(
            signal_contract["reproductive_readiness_profile"]["radius"],
            0,
        )
        self.assertEqual(
            signal_contract["communication_token_count"],
            signal_config.communication_token_count,
        )
        self.assertTrue(signal_specs)
        for action in signal_specs:
            self.assertFalse(action["active"], msg=action["key"])
            self.assertNotIn(action["key"], contract["active_action_keys"])
            self.assertFalse(mask[action["key"]], msg=action["key"])

    def test_signal_actions_stay_inactive_without_runtime_signal_capacity(self) -> None:
        signal_config = SignalConfig(
            communication_signal_emission_enabled=True,
            max_intensity=0.0,
            reproductive_signal_base_intensity=0.0,
            reproductive_signal_trait_intensity_bonus=0.0,
            communication_signal_base_intensity=0.0,
            communication_signal_trait_intensity_bonus=0.0,
        )
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1, signals=signal_config))
        agent = world.alive_agents()[0]
        agent.genome = replace(
            agent.genome,
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )

        contract = action_contract(signal_config)
        signal_contract = runtime_signals.signal_contract(signal_config)
        mask = build_action_mask(world._action_mask_context(agent))

        self.assertFalse(contract["communication"]["emission_enabled"])
        self.assertFalse(signal_contract["reproductive_signal_emission_enabled"])
        self.assertFalse(signal_contract["communication_signal_emission_enabled"])
        self.assertEqual(
            signal_contract["reproductive_readiness_profile"]["intensity"],
            0.0,
        )
        self.assertNotIn("signal_0_profile_0", contract["active_action_keys"])
        self.assertFalse(mask["signal_0_profile_0"])

    def test_signal_actions_stay_inactive_without_configured_signal_intensity(
        self,
    ) -> None:
        signal_config = SignalConfig(
            reproductive_signal_base_intensity=0.0,
            reproductive_signal_trait_intensity_bonus=0.0,
            communication_signal_emission_enabled=True,
            communication_signal_base_intensity=0.0,
            communication_signal_trait_intensity_bonus=0.0,
        )
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1, signals=signal_config))
        agent = world.alive_agents()[0]
        agent.genome = replace(
            agent.genome,
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )

        contract = action_contract(signal_config)
        signal_contract = runtime_signals.signal_contract(signal_config)
        mask = build_action_mask(world._action_mask_context(agent))
        reproductive_totals = runtime_signals.emit_reproductive_readiness_signals(
            [agent],
            context=world._signal_runtime_context(),
        )

        self.assertFalse(contract["communication"]["emission_enabled"])
        self.assertFalse(signal_contract["reproductive_signal_emission_enabled"])
        self.assertFalse(signal_contract["communication_signal_emission_enabled"])
        self.assertEqual(
            signal_contract["reproductive_readiness_profile"]["intensity"],
            0.0,
        )
        self.assertEqual(reproductive_totals["reproductive_emissions"], 0)
        self.assertNotIn("signal_0_profile_0", contract["active_action_keys"])
        self.assertFalse(mask["signal_0_profile_0"])

    def test_pre_mind_reproductive_slots_start_without_emitted_signals(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )
        decoded = decode_observation_input(encode_observation_input(observation))

        self.assertEqual(agent.reproductive_group_id, agent.lineage_id)
        self.assertEqual(agent.reproductive_stage, "stage0_asexual")
        self.assertEqual(agent.reproductive_expression, "asexual")
        self.assertEqual(
            agent.mind_inheritance_metadata["schema_version"],
            MIND_INHERITANCE_PLACEHOLDER_VERSION,
        )
        self.assertFalse(agent.mind_inheritance_metadata["inherited_state"])
        self.assertTrue(
            all(value == 0.0 for value in agent.genome.reproductive.to_dict().values())
        )

        self_state = observation["self"]
        self.assertEqual(self_state["reproductive_stage"], "stage0_asexual")
        self.assertEqual(self_state["reproductive_expression"], "asexual")
        self.assertFalse(self_state["sexual_reproduction_unlocked"])
        self.assertEqual(self_state["reproductive_signal"], 0.0)
        self.assertEqual(self_state["communication_signal"], 0.0)
        self.assertFalse(self_state["mind_inheritance_available"])
        self.assertTrue(
            all(
                cell["reproductive_signal"] == 0.0
                and cell["communication_signal"] == 0.0
                for cell in observation["local_patch"]
            )
        )
        self.assertEqual(len(decoded), OBSERVATION_INPUT_VECTOR_SIZE)

    def test_reproductive_signal_emits_for_ready_agents_and_decays(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=7,
                signals=SignalConfig(
                    reproductive_signal_radius=2,
                    reproductive_signal_duration_ticks=3,
                    reproductive_signal_decay_rate=0.5,
                    reproductive_signal_base_intensity=0.2,
                    reproductive_signal_trait_intensity_bonus=0.3,
                    base_emission_energy_cost=0.05,
                ),
            )
        )
        genome = replace(
            self._mixed_genome(),
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )
        agent = self._place_ready_agent(world, x=2, y=2, genome=genome)
        energy_before = agent.energy

        totals = runtime_signals.emit_reproductive_readiness_signals(
            [agent],
            context=world._signal_runtime_context(),
        )
        signal_state = world._current_signal_state()
        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )
        decoded = decode_observation_input(encode_observation_input(observation))

        self.assertEqual(totals["reproductive_emissions"], 1)
        self.assertEqual(totals["communication_emissions"], 0)
        self.assertEqual(len(world.reproductive_signal_emissions), 1)
        self.assertEqual(len(world.tick_signal_emission_events), 1)
        emission = world.reproductive_signal_emissions[0]
        emission_event = world.tick_signal_emission_events[0]
        self.assertEqual(emission.profile_id, "reproductive_readiness")
        self.assertEqual(emission.source_agent_id, agent.agent_id)
        self.assertIsNone(emission.token_id)
        self.assertEqual(emission.radius, 2)
        self.assertEqual(emission.duration_ticks, 3)
        self.assertEqual(emission.decay_rate, 0.5)
        self.assertFalse(emission_event["policy_visible"])
        self.assertEqual(emission_event["source_agent_id"], agent.agent_id)
        self.assertEqual(emission_event["profile_id"], "reproductive_readiness")
        self.assertEqual(emission_event["token_id"], None)
        self.assertAlmostEqual(float(totals["energy_spent"]), 0.025)
        self.assertAlmostEqual(agent.energy, energy_before - 0.025)
        self.assertAlmostEqual(signal_state.reproductive_signal[2][2], 0.5)
        self.assertGreater(signal_state.reproductive_signal[2][2], 0.0)
        self.assertGreater(
            signal_state.reproductive_signal[2][2],
            signal_state.reproductive_signal[2][3],
        )
        self.assertEqual(signal_state.communication_signal[2][2], 0.0)
        self.assertGreater(observation["self"]["reproductive_signal"], 0.0)
        self.assertEqual(observation["self"]["communication_signal"], 0.0)
        self.assertTrue(
            any(
                cell["reproductive_signal"] > 0.0 for cell in observation["local_patch"]
            )
        )
        self.assertEqual(len(decoded), OBSERVATION_INPUT_VECTOR_SIZE)

        runtime_signals.decay_signal_emissions(
            context=world._signal_runtime_context(),
        )
        decayed_state = world._current_signal_state()
        self.assertEqual(world.reproductive_signal_emissions[0].remaining_ticks, 2)
        self.assertAlmostEqual(world.reproductive_signal_emissions[0].intensity, 0.25)

        self.assertLess(
            decayed_state.reproductive_signal[2][2],
            signal_state.reproductive_signal[2][2],
        )
        self.assertEqual(decayed_state.communication_signal[2][2], 0.0)

    def test_decayed_reproductive_signal_retains_emission_source_snapshot(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=7,
                signals=SignalConfig(
                    reproductive_signal_radius=2,
                    reproductive_signal_duration_ticks=3,
                    reproductive_signal_decay_rate=0.5,
                    reproductive_signal_base_intensity=0.2,
                    reproductive_signal_trait_intensity_bonus=0.3,
                    base_emission_energy_cost=0.0,
                ),
            )
        )
        genome = replace(
            self._mixed_genome(),
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )
        agent = self._place_ready_agent(world, x=2, y=2, genome=genome)

        runtime_signals.emit_reproductive_readiness_signals(
            [agent],
            context=world._signal_runtime_context(),
        )
        emission = world.reproductive_signal_emissions[0]
        source_agent_id = agent.agent_id
        agent.x = 6
        agent.y = 6
        agent.alive = False
        agent.death_tick = world.tick
        world.tick_signal_emission_events = []

        runtime_signals.decay_signal_emissions(
            context=world._signal_runtime_context(),
        )
        snapshot = runtime_signals.signal_emission_debug_snapshot(
            context=world._signal_runtime_context(),
        )
        decayed_state = world._current_signal_state()

        self.assertEqual(snapshot["events"], [])
        self.assertEqual(snapshot["active_counts"]["reproductive_signal"], 1)
        self.assertEqual(emission.source_agent_id, source_agent_id)
        self.assertEqual(emission.x, 2)
        self.assertEqual(emission.y, 2)
        self.assertEqual(emission.emitted_tick, 0)
        self.assertEqual(emission.remaining_ticks, 2)
        self.assertAlmostEqual(decayed_state.reproductive_signal[2][2], 0.25)
        self.assertEqual(decayed_state.reproductive_signal[6][6], 0.0)

    def test_decayed_signal_active_debug_snapshot_is_opt_in(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=7,
                signals=SignalConfig(
                    reproductive_signal_radius=2,
                    reproductive_signal_duration_ticks=3,
                    reproductive_signal_decay_rate=0.5,
                    reproductive_signal_base_intensity=0.2,
                    reproductive_signal_trait_intensity_bonus=0.3,
                    base_emission_energy_cost=0.0,
                ),
            )
        )
        genome = replace(
            self._mixed_genome(),
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )
        agent = self._place_ready_agent(world, x=2, y=2, genome=genome)

        runtime_signals.emit_reproductive_readiness_signals(
            [agent],
            context=world._signal_runtime_context(),
        )
        agent.x = 6
        agent.y = 6
        agent.alive = False
        agent.death_tick = world.tick
        world.tick_signal_emission_events = []
        runtime_signals.decay_signal_emissions(
            context=world._signal_runtime_context(),
        )

        default_snapshot = runtime_signals.signal_emission_debug_snapshot(
            context=world._signal_runtime_context(),
        )
        detailed_snapshot = runtime_signals.signal_emission_debug_snapshot(
            context=world._signal_runtime_context(),
            include_active_emissions=True,
        )
        active_reproductive = detailed_snapshot["active_emissions"][
            "reproductive_signal"
        ]

        self.assertNotIn("active_emissions", default_snapshot)
        self.assertEqual(len(active_reproductive), 1)
        self.assertEqual(active_reproductive[0]["source_agent_id"], agent.agent_id)
        self.assertEqual(active_reproductive[0]["x"], 2)
        self.assertEqual(active_reproductive[0]["y"], 2)
        self.assertEqual(active_reproductive[0]["emitted_tick"], 0)
        self.assertEqual(active_reproductive[0]["remaining_ticks"], 2)
        self.assertAlmostEqual(active_reproductive[0]["intensity"], 0.25)

    def test_reproductive_signal_is_biology_gated(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=5,
                signals=SignalConfig(
                    reproductive_signal_radius=1,
                    reproductive_signal_duration_ticks=2,
                ),
            )
        )
        agent = self._place_ready_agent(world, x=2, y=2)
        agent.energy = 0.0

        totals = runtime_signals.emit_reproductive_readiness_signals(
            [agent],
            context=world._signal_runtime_context(),
        )
        signal_state = world._current_signal_state()

        self.assertEqual(totals["reproductive_emissions"], 0)
        self.assertEqual(signal_state.reproductive_signal[2][2], 0.0)
        self.assertEqual(world.run_signal_totals["reproductive_emissions"], 0.0)

    def test_reproductive_signal_cost_preserves_readiness_truthfulness(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=5,
                max_agents=10,
                signals=SignalConfig(
                    reproductive_signal_base_intensity=1.0,
                    reproductive_signal_trait_intensity_bonus=0.0,
                    base_emission_energy_cost=0.2,
                ),
            )
        )
        agent = self._place_ready_agent(world, x=2, y=2)
        profile = world._trophic_profile(agent)
        energy_required = world._reproduction_energy_requirement(agent, profile)
        agent.energy = energy_required + 0.05

        births = runtime_reproduction.run_reproduction_phase(
            world,
            context=world._reproduction_context(),
        )

        self.assertEqual(births, 1)
        self.assertEqual(world.tick_signal_totals["reproductive_emissions"], 0.0)
        self.assertEqual(world.tick_signal_totals["energy_spent"], 0.0)
        self.assertEqual(world.reproductive_signal_emissions, [])

    def test_communication_signal_actions_are_opt_in_and_trait_gated(self) -> None:
        default_world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        default_agent = default_world.alive_agents()[0]
        default_agent.genome = replace(
            default_agent.genome,
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )
        default_contract = action_contract(default_world.config.signals)
        self.assertFalse(default_contract["communication"]["emission_enabled"])
        self.assertNotIn(
            "signal_0_profile_0",
            default_contract["active_action_keys"],
        )
        self.assertFalse(
            build_action_mask(default_world._action_mask_context(default_agent))[
                "signal_0_profile_0"
            ]
        )

        enabled_world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                signals=SignalConfig(communication_signal_emission_enabled=True),
            )
        )
        gated_agent = enabled_world.alive_agents()[0]
        gated_agent.genome = replace(
            gated_agent.genome,
            reproductive=ReproductiveGenome(signal_emission_bias=0.0),
        )
        enabled_contract = action_contract(enabled_world.config.signals)
        self.assertTrue(enabled_contract["communication"]["emission_enabled"])
        self.assertIn(
            "signal_0_profile_0",
            enabled_contract["active_action_keys"],
        )
        self.assertFalse(
            build_action_mask(enabled_world._action_mask_context(gated_agent))[
                "signal_0_profile_0"
            ]
        )

    def test_communication_signal_action_emits_opaque_numeric_field(self) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                signals=SignalConfig(
                    communication_signal_emission_enabled=True,
                    communication_signal_radius=2,
                    communication_signal_duration_ticks=3,
                    communication_signal_decay_rate=0.5,
                    communication_signal_base_intensity=0.2,
                    communication_signal_trait_intensity_bonus=0.3,
                    base_emission_energy_cost=0.05,
                ),
            )
        )
        agent = world.alive_agents()[0]
        agent.genome = replace(
            agent.genome,
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )
        agent.energy = 1.0
        action = "signal_1_profile_1"
        mask = build_action_mask(world._action_mask_context(agent))
        contract = action_contract(world.config.signals)
        signal_spec = next(
            spec for spec in contract["actions"] if spec["key"] == action
        )

        self.assertTrue(mask[action])
        self.assertTrue(signal_spec["active"])
        self.assertTrue(signal_spec["reserved"])
        moved, outcome = world._resolve_action_with_outcome(
            agent,
            action,
            observation_action_mask=mask,
            resolution_action_mask=mask,
        )
        signal_outcome = outcome["signal"]
        signal_state = world._current_signal_state()
        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )
        event = world.tick_signal_emission_events[0]

        self.assertFalse(moved)
        self.assertEqual(outcome["resolved_action"], action)
        self.assertTrue(signal_outcome["emitted"])
        self.assertEqual(signal_outcome["token_id"], 1)
        self.assertEqual(signal_outcome["profile_index"], 1)
        self.assertNotIn("profile_id", signal_outcome)
        self.assertAlmostEqual(signal_outcome["intensity"], 0.5)
        self.assertEqual(signal_outcome["radius"], 3)
        self.assertEqual(signal_outcome["duration_ticks"], 4)
        self.assertAlmostEqual(signal_outcome["energy_cost"], 0.05)
        self.assertAlmostEqual(agent.energy, 0.95)
        self.assertEqual(len(world.communication_signal_emissions), 1)
        self.assertEqual(world.tick_signal_totals["communication_emissions"], 1.0)
        self.assertEqual(world.run_signal_totals["communication_emissions"], 1.0)
        self.assertEqual(event["field_name"], "communication_signal_token_1")
        self.assertEqual(event["source_agent_id"], agent.agent_id)
        self.assertEqual(event["token_id"], 1)
        self.assertEqual(event["profile_index"], 1)
        self.assertFalse(event["policy_visible"])
        self.assertGreater(signal_state.communication_signal[agent.y][agent.x], 0.0)
        self.assertEqual(len(signal_state.communication_token_signals), 4)
        self.assertEqual(
            signal_state.communication_token_signals[0][agent.y][agent.x],
            0.0,
        )
        self.assertGreater(
            signal_state.communication_token_signals[1][agent.y][agent.x],
            0.0,
        )
        self.assertEqual(signal_state.reproductive_signal[agent.y][agent.x], 0.0)
        self.assertEqual(observation["self"]["communication_signal"], 0.0)
        self.assertEqual(
            observation["self"]["communication_signal_token_0"],
            0.0,
        )
        self.assertEqual(
            observation["self"]["communication_signal_token_1"],
            0.0,
        )
        self.assertTrue(
            all(
                cell["communication_signal"] == 0.0
                and cell["communication_signal_token_1"] == 0.0
                for cell in observation["local_patch"]
            )
        )

    def test_opaque_communication_sender_never_observes_its_only_emission(
        self,
    ) -> None:
        signal_config = SignalConfig(
            communication_signal_emission_enabled=True,
            communication_token_count=1,
            communication_profiles_per_token=1,
            communication_signal_radius=2,
            communication_signal_duration_ticks=3,
            communication_signal_decay_rate=0.5,
            communication_signal_base_intensity=0.8,
            communication_signal_trait_intensity_bonus=0.0,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=5,
                signals=signal_config,
            )
        )
        genome = replace(
            self._mixed_genome(),
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )
        emitter = self._place_ready_agent(world, x=2, y=2, genome=genome)
        mask = build_action_mask(world._action_mask_context(emitter))

        _, outcome = world._resolve_action_with_outcome(
            emitter,
            "signal_0_profile_0",
            observation_action_mask=mask,
            resolution_action_mask=mask,
        )

        self.assertTrue(outcome["signal"]["emitted"])
        self.assertGreater(
            world._current_signal_state().field("communication_signal_token_0")[
                emitter.y
            ][emitter.x],
            0.0,
        )
        for _ in range(2):
            observation = world._observe_agent(emitter)
            self.assertEqual(observation["self"]["communication_signal"], 0.0)
            self.assertEqual(
                observation["self"]["communication_signal_token_0"],
                0.0,
            )
            self.assertTrue(
                all(
                    cell["communication_signal"] == 0.0
                    and cell["communication_signal_token_0"] == 0.0
                    for cell in observation["local_patch"]
                )
            )
            runtime_signals.decay_signal_emissions(
                context=world._signal_runtime_context()
            )

    def test_opaque_communication_listener_receives_sender_signal(self) -> None:
        signal_config = SignalConfig(
            communication_signal_emission_enabled=True,
            communication_token_count=1,
            communication_profiles_per_token=1,
            communication_signal_radius=2,
            communication_signal_duration_ticks=3,
            communication_signal_decay_rate=0.5,
            communication_signal_base_intensity=0.6,
            communication_signal_trait_intensity_bonus=0.0,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=5,
                signals=signal_config,
            )
        )
        genome = replace(
            self._mixed_genome(),
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )
        emitter = self._place_ready_agent(
            world,
            x=1,
            y=2,
            lineage_id=1,
            genome=genome,
        )
        listener = self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=2,
            genome=genome,
        )
        mask = build_action_mask(world._action_mask_context(emitter))
        world._resolve_action_with_outcome(
            emitter,
            "signal_0_profile_0",
            observation_action_mask=mask,
            resolution_action_mask=mask,
        )

        emitter_observation = world._observe_agent(emitter)
        listener_observation = world._observe_agent(listener)

        self.assertEqual(
            emitter_observation["self"]["communication_signal_token_0"],
            0.0,
        )
        self.assertAlmostEqual(
            listener_observation["self"]["communication_signal_token_0"],
            0.3,
        )
        self.assertAlmostEqual(
            listener_observation["self"]["communication_signal"],
            0.3,
        )

    def test_same_token_multi_emitter_projection_excludes_only_receiver(
        self,
    ) -> None:
        signal_config = SignalConfig(
            communication_signal_emission_enabled=True,
            communication_token_count=1,
            communication_profiles_per_token=1,
            communication_signal_radius=2,
            communication_signal_duration_ticks=3,
            communication_signal_decay_rate=0.5,
            communication_signal_base_intensity=0.8,
            communication_signal_trait_intensity_bonus=0.0,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=5,
                signals=signal_config,
            )
        )
        genome = replace(
            self._mixed_genome(),
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )
        first = self._place_ready_agent(
            world,
            x=1,
            y=2,
            lineage_id=1,
            genome=genome,
        )
        second = self._place_ready_agent(
            world,
            x=2,
            y=2,
            lineage_id=2,
            genome=genome,
        )
        for emitter in (first, second):
            mask = build_action_mask(world._action_mask_context(emitter))
            world._resolve_action_with_outcome(
                emitter,
                "signal_0_profile_0",
                observation_action_mask=mask,
                resolution_action_mask=mask,
            )

        state = world._current_signal_state()
        first_observation = world._observe_agent(first)
        second_observation = world._observe_agent(second)

        self.assertAlmostEqual(
            state.field("communication_signal_token_0")[first.y][first.x],
            1.0,
        )
        self.assertAlmostEqual(
            state.field("communication_signal_token_0")[second.y][second.x],
            1.0,
        )
        self.assertAlmostEqual(
            first_observation["self"]["communication_signal_token_0"],
            0.4,
        )
        self.assertAlmostEqual(
            second_observation["self"]["communication_signal_token_0"],
            0.4,
        )
        self.assertAlmostEqual(
            first_observation["self"]["communication_signal"],
            0.4,
        )
        self.assertAlmostEqual(
            second_observation["self"]["communication_signal"],
            0.4,
        )

    def test_receiver_projection_matches_independent_clamp_and_diffusion_model(
        self,
    ) -> None:
        signal_config = SignalConfig(
            communication_signal_emission_enabled=True,
            communication_token_count=2,
            communication_profiles_per_token=1,
            max_intensity=1.0,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=4,
                signals=signal_config,
            )
        )
        emissions = [
            runtime_signals.SignalEmission(
                field_name="communication_signal_token_0",
                profile_id="token-0-a",
                source_kind="test",
                source_agent_id=101,
                token_id=0,
                profile_index=0,
                x=1,
                y=1,
                intensity=0.8,
                radius=1,
                duration_ticks=3,
                remaining_ticks=3,
                decay_rate=0.0,
                energy_cost=0.0,
                emitted_tick=0,
            ),
            runtime_signals.SignalEmission(
                field_name="communication_signal_token_0",
                profile_id="token-0-b",
                source_kind="test",
                source_agent_id=202,
                token_id=0,
                profile_index=0,
                x=1,
                y=1,
                intensity=0.7,
                radius=1,
                duration_ticks=3,
                remaining_ticks=3,
                decay_rate=0.0,
                energy_cost=0.0,
                emitted_tick=0,
            ),
            runtime_signals.SignalEmission(
                field_name="communication_signal_token_0",
                profile_id="token-0-c",
                source_kind="test",
                source_agent_id=101,
                token_id=0,
                profile_index=0,
                x=2,
                y=1,
                intensity=0.9,
                radius=2,
                duration_ticks=3,
                remaining_ticks=3,
                decay_rate=0.0,
                energy_cost=0.0,
                emitted_tick=0,
            ),
            runtime_signals.SignalEmission(
                field_name="communication_signal_token_1",
                profile_id="token-1-a",
                source_kind="test",
                source_agent_id=101,
                token_id=1,
                profile_index=0,
                x=1,
                y=1,
                intensity=0.65,
                radius=1,
                duration_ticks=3,
                remaining_ticks=3,
                decay_rate=0.0,
                energy_cost=0.0,
                emitted_tick=0,
            ),
            runtime_signals.SignalEmission(
                field_name="communication_signal_token_1",
                profile_id="token-1-b",
                source_kind="test",
                source_agent_id=303,
                token_id=1,
                profile_index=0,
                x=3,
                y=2,
                intensity=0.95,
                radius=3,
                duration_ticks=3,
                remaining_ticks=3,
                decay_rate=0.0,
                energy_cost=0.0,
                emitted_tick=0,
            ),
        ]
        world.communication_signal_emissions.extend(emissions)
        state = runtime_signals.build_signal_state(
            context=world._signal_runtime_context()
        )

        def independent_value(
            field_name: str,
            *,
            x: int,
            y: int,
            excluded_agent_id: int | None,
        ) -> float:
            sources: dict[tuple[int, int, int], float] = {}
            for emission in emissions:
                if (
                    excluded_agent_id is not None
                    and emission.source_agent_id == excluded_agent_id
                ):
                    continue
                if field_name != "communication_signal" and (
                    field_name
                    != f"communication_signal_token_{emission.token_id}"
                ):
                    continue
                key = (emission.radius, emission.x, emission.y)
                sources[key] = sources.get(key, 0.0) + emission.intensity
            raw_value = 0.0
            for (radius, source_x, source_y), intensity in sources.items():
                distance = abs(x - source_x) + abs(y - source_y)
                if distance <= radius:
                    raw_value += min(signal_config.max_intensity, intensity) / (
                        distance + 1.0
                    )
            return min(signal_config.max_intensity, raw_value)

        fields = (
            "communication_signal",
            "communication_signal_token_0",
            "communication_signal_token_1",
        )
        receivers = (101, 202, 303, 999)
        height = len(world.grid)
        width = len(world.grid[0])
        for field_name in fields:
            global_field = state.field(field_name)
            for y in range(height):
                for x in range(width):
                    self.assertAlmostEqual(
                        global_field[y][x],
                        independent_value(
                            field_name,
                            x=x,
                            y=y,
                            excluded_agent_id=None,
                        ),
                    )
                    for receiver_agent_id in receivers:
                        self.assertAlmostEqual(
                            state.field_value_for_receiver(
                                field_name,
                                x=x,
                                y=y,
                                receiver_agent_id=receiver_agent_id,
                            ),
                            independent_value(
                                field_name,
                                x=x,
                                y=y,
                                excluded_agent_id=receiver_agent_id,
                            ),
                        )

        projection = state.communication_receiver_projection
        self.assertIsNotNone(projection)
        assert projection is not None
        self.assertGreater(len(projection.receiver_value_cache), 0)
        self.assertGreater(len(projection.raw_global_value_cache), 0)

        copied_state = copy.deepcopy(state)
        copied_projection = copied_state.communication_receiver_projection
        self.assertIsNotNone(copied_projection)
        assert copied_projection is not None
        self.assertIsNot(
            copied_projection.receiver_value_cache,
            projection.receiver_value_cache,
        )
        self.assertIsNot(
            copied_projection.raw_global_value_cache,
            projection.raw_global_value_cache,
        )
        for field_name in fields:
            for receiver_agent_id in receivers:
                for y in range(height):
                    for x in range(width):
                        self.assertAlmostEqual(
                            copied_state.field_value_for_receiver(
                                field_name,
                                x=x,
                                y=y,
                                receiver_agent_id=receiver_agent_id,
                            ),
                            independent_value(
                                field_name,
                                x=x,
                                y=y,
                                excluded_agent_id=receiver_agent_id,
                            ),
                        )

    def test_opaque_communication_radius_timing_and_decay_are_preserved(
        self,
    ) -> None:
        signal_config = SignalConfig(
            communication_signal_emission_enabled=True,
            communication_token_count=1,
            communication_profiles_per_token=1,
            communication_signal_radius=2,
            communication_signal_duration_ticks=3,
            communication_signal_decay_rate=0.5,
            communication_signal_base_intensity=0.6,
            communication_signal_trait_intensity_bonus=0.0,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=7,
                height=5,
                signals=signal_config,
            )
        )
        genome = replace(
            self._mixed_genome(),
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )
        emitter = self._place_ready_agent(
            world,
            x=1,
            y=2,
            lineage_id=1,
            genome=genome,
        )
        inner_listener = self._place_ready_agent(
            world,
            x=3,
            y=2,
            lineage_id=2,
            genome=genome,
        )
        outer_listener = self._place_ready_agent(
            world,
            x=4,
            y=2,
            lineage_id=3,
            genome=genome,
        )
        mask = build_action_mask(world._action_mask_context(emitter))
        world._resolve_action_with_outcome(
            emitter,
            "signal_0_profile_0",
            observation_action_mask=mask,
            resolution_action_mask=mask,
        )

        self.assertAlmostEqual(
            world._observe_agent(inner_listener)["self"][
                "communication_signal_token_0"
            ],
            0.2,
        )
        self.assertEqual(
            world._observe_agent(outer_listener)["self"][
                "communication_signal_token_0"
            ],
            0.0,
        )
        expected_inner_values = (0.1, 0.05, 0.0)
        for expected in expected_inner_values:
            runtime_signals.decay_signal_emissions(
                context=world._signal_runtime_context()
            )
            self.assertAlmostEqual(
                world._observe_agent(inner_listener)["self"][
                    "communication_signal_token_0"
                ],
                expected,
            )
            self.assertEqual(
                world._observe_agent(emitter)["self"]["communication_signal_token_0"],
                0.0,
            )

    def test_active_opaque_communication_full_replay_is_deterministic(
        self,
    ) -> None:
        config = WorldConfig(
            seed=7,
            max_ticks=4,
            initial_agents=4,
            water_tile_ratio=0.0,
            signals=SignalConfig(
                communication_signal_emission_enabled=True,
                communication_token_count=1,
                communication_profiles_per_token=1,
                communication_signal_radius=2,
                communication_signal_duration_ticks=3,
                communication_signal_decay_rate=0.5,
                communication_signal_base_intensity=0.6,
                communication_signal_trait_intensity_bonus=0.0,
            ),
        )

        def run() -> SimulationWorldResult:
            world = SimulationWorld(
                copy.deepcopy(config),
                policy=_AlwaysOpaqueSignalPolicy(),
            )
            for agent in world.alive_agents():
                agent.genome = replace(
                    agent.genome,
                    reproductive=ReproductiveGenome(signal_emission_bias=1.0),
                )
                agent.genome_vector = genome_vector(agent.genome)
            return world.run()

        first = run()
        second = run()

        self.assertGreater(
            first.summary["signal_end"]["communication_emissions"],
            0,
        )
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.events, second.events)
        self.assertEqual(first.viewer, second.viewer)

    def test_opaque_communication_tokens_have_distinct_public_spatial_channels(
        self,
    ) -> None:
        signal_config = SignalConfig(
            communication_signal_emission_enabled=True,
            communication_token_count=2,
            communication_profiles_per_token=1,
            communication_signal_radius=0,
            communication_signal_duration_ticks=3,
            communication_signal_base_intensity=0.2,
            communication_signal_trait_intensity_bonus=0.3,
        )
        world = SimulationWorld(
            self._ready_reproduction_config(
                width=5,
                height=5,
                signals=signal_config,
            )
        )
        genome = replace(
            self._mixed_genome(),
            reproductive=ReproductiveGenome(signal_emission_bias=1.0),
        )
        token_0_source = self._place_ready_agent(
            world,
            x=1,
            y=2,
            lineage_id=1,
            genome=genome,
        )
        token_1_source = self._place_ready_agent(
            world,
            x=3,
            y=2,
            lineage_id=2,
            genome=genome,
        )

        for agent, action in (
            (token_0_source, "signal_0_profile_0"),
            (token_1_source, "signal_1_profile_0"),
        ):
            mask = build_action_mask(world._action_mask_context(agent))
            _, outcome = world._resolve_action_with_outcome(
                agent,
                action,
                observation_action_mask=mask,
                resolution_action_mask=mask,
            )
            self.assertTrue(outcome["signal"]["emitted"])

        state = world._current_signal_state()
        observation = build_observation(
            world,
            token_0_source,
            observation_context=world._observation_context(token_0_source),
        )
        encoded = encode_observation_input(observation)
        decoded = decode_observation_input(encoded)
        contract = observation_contract(signal_config)
        policy_input = contract["policy_input"]
        self_fields = policy_input["self_input_fields"]
        patch_fields = policy_input["patch_input_fields"]
        token_1_patch_index = 2 * 5 + 4
        token_1_value_index = (
            len(self_fields)
            + token_1_patch_index * len(patch_fields)
            + patch_fields.index("communication_signal_token_1")
        )

        self.assertGreater(state.communication_signal[2][1], 0.0)
        self.assertGreater(state.communication_signal[2][3], 0.0)
        self.assertGreater(state.field("communication_signal_token_0")[2][1], 0.0)
        self.assertEqual(state.field("communication_signal_token_0")[2][3], 0.0)
        self.assertEqual(state.field("communication_signal_token_1")[2][1], 0.0)
        self.assertGreater(state.field("communication_signal_token_1")[2][3], 0.0)
        self.assertEqual(
            observation["self"]["communication_signal_token_0"],
            0.0,
        )
        self.assertEqual(
            observation["self"]["communication_signal_token_1"],
            0.0,
        )
        token_1_patch = next(
            cell
            for cell in observation["local_patch"]
            if cell["dx"] == 2 and cell["dy"] == 0
        )
        self.assertEqual(token_1_patch["communication_signal_token_0"], 0.0)
        self.assertGreater(token_1_patch["communication_signal_token_1"], 0.0)
        self.assertEqual(
            observation["schema_version"],
            TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
        )
        self.assertEqual(
            encoded["encoder_version"],
            TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION,
        )
        self.assertEqual(encoded["communication_token_count"], 2)
        self.assertEqual(
            encoded["communication_token_field_order"],
            [
                "communication_signal_token_0",
                "communication_signal_token_1",
            ],
        )
        self.assertEqual(encoded["shape"], [OBSERVATION_INPUT_VECTOR_SIZE + 52])
        self.assertGreater(decoded[token_1_value_index], 0.0)
        policy_values = ecological_policy_input_values(encoded)
        self.assertEqual(len(policy_values), encoded["shape"][0] - 1)
        self.assertEqual(
            policy_values[token_1_value_index - 1],
            decoded[token_1_value_index],
        )

    def test_token_observation_encoder_rejects_noncanonical_field_aliases(
        self,
    ) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                signals=SignalConfig(
                    communication_signal_emission_enabled=True,
                    communication_token_count=2,
                    communication_profiles_per_token=1,
                ),
            )
        )
        agent = world.alive_agents()[0]
        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )
        observation["self"]["communication_signal_token_00"] = observation["self"].pop(
            "communication_signal_token_0"
        )
        for cell in observation["local_patch"]:
            cell["communication_signal_token_00"] = cell.pop(
                "communication_signal_token_0"
            )

        self.assertIsNone(
            parse_communication_token_field_name("communication_signal_token_00")
        )
        with self.assertRaisesRegex(
            ValueError,
            "communication token observation fields must use contiguous token ids",
        ):
            encode_observation_input(observation)

    def test_previous_token_observation_schema_fails_closed(self) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                signals=SignalConfig(
                    communication_signal_emission_enabled=True,
                    communication_token_count=2,
                    communication_profiles_per_token=1,
                ),
            )
        )
        agent = world.alive_agents()[0]
        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )

        self.assertEqual(
            observation["schema_version"],
            TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
        )
        observation["schema_version"] = "mind_observation_v4"
        with self.assertRaisesRegex(ValueError, "missing or stale"):
            encode_observation_input(observation)

    def test_tokenized_signal_contracts_are_versioned_without_default_drift(
        self,
    ) -> None:
        default_result = SimulationWorld(WorldConfig(seed=7, max_ticks=2)).run()
        default_signal_contract = default_result.viewer["trajectory"][
            "observation_contract"
        ]["signal_contract"]
        default_signal_fields = default_result.viewer["frames"][0]["signal_fields"]
        default_record = default_result.viewer["trajectory"]["records"][0]

        self.assertEqual(
            default_result.summary["summary_schema_version"],
            SUMMARY_SCHEMA_VERSION,
        )
        self.assertEqual(
            default_result.viewer["trajectory"]["schema_version"],
            TRAJECTORY_SCHEMA_VERSION,
        )
        self.assertEqual(
            default_result.viewer["trajectory"]["observation_schema_version"],
            OBSERVATION_SCHEMA_VERSION,
        )
        self.assertEqual(
            default_signal_contract["schema_version"],
            SIGNAL_CONTRACT_VERSION,
        )
        self.assertEqual(
            TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION,
            "foundation_signal_contract_v4",
        )
        self.assertEqual(
            TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
            "mind_observation_v5",
        )
        self.assertEqual(
            TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION,
            "mind_observation_encoder_v4",
        )
        self.assertEqual(
            TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION,
            "mind_trajectory_v3",
        )
        self.assertEqual(
            TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
            "mind_ecological_policy_input_v3",
        )
        self.assertNotIn(
            "communication_token_observation",
            default_signal_contract,
        )
        self.assertEqual(
            set(default_signal_fields),
            {"reproductive_signal", "communication_signal"},
        )
        self.assertEqual(
            default_record["observation_input"]["shape"],
            [OBSERVATION_INPUT_VECTOR_SIZE],
        )

        enabled_config = WorldConfig(
            seed=7,
            max_ticks=2,
            signals=SignalConfig(
                communication_signal_emission_enabled=True,
                communication_token_count=2,
                communication_profiles_per_token=1,
            ),
        )
        first = SimulationWorld(enabled_config).run()
        second = SimulationWorld(copy.deepcopy(enabled_config)).run()
        trajectory = first.viewer["trajectory"]
        tokenized_signal_contract = trajectory["observation_contract"][
            "signal_contract"
        ]
        token_fields = {
            "communication_signal_token_0",
            "communication_signal_token_1",
        }

        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.events, second.events)
        self.assertEqual(first.viewer, second.viewer)
        self.assertEqual(
            first.summary["summary_schema_version"],
            TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION,
        )
        self.assertEqual(
            trajectory["schema_version"],
            TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION,
        )
        self.assertEqual(
            trajectory["observation_schema_version"],
            TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
        )
        self.assertEqual(
            tokenized_signal_contract["schema_version"],
            TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION,
        )
        self.assertFalse(
            tokenized_signal_contract[
                "communication_tokens_have_simulator_assigned_meaning"
            ]
        )
        self.assertTrue(
            tokenized_signal_contract["communication_token_observation"][
                "policy_visible"
            ]
        )
        self.assertEqual(
            tokenized_signal_contract["communication_token_observation"]["aggregation"],
            COMMUNICATION_AGGREGATE_PROJECTION,
        )
        self.assertEqual(
            tokenized_signal_contract["communication_token_observation"][
                "receiver_projection_schema_version"
            ],
            COMMUNICATION_RECEIVER_PROJECTION_SCHEMA_VERSION,
        )
        self.assertEqual(
            tokenized_signal_contract["communication_token_observation"][
                "receiver_observation_policy"
            ],
            COMMUNICATION_RECEIVER_OBSERVATION_POLICY,
        )
        self.assertEqual(
            tokenized_signal_contract["communication_token_observation"][
                "global_reporting_policy"
            ],
            COMMUNICATION_GLOBAL_REPORTING_POLICY,
        )
        self.assertEqual(
            trajectory["observation_contract"]["communication_token_channels"][
                "aggregate_projection"
            ],
            COMMUNICATION_AGGREGATE_PROJECTION,
        )
        self.assertEqual(
            set(first.viewer["frames"][0]["signal_fields"]),
            {
                "reproductive_signal",
                "communication_signal",
                *token_fields,
            },
        )
        self.assertTrue(
            token_fields.issubset(set(first.summary["signal_field_stats_at_end"]))
        )
        self.assertTrue(
            token_fields.issubset(set(first.viewer["analytics"]["signal_fields"]))
        )
        self.assertNotIn(
            "communication_signal",
            trajectory["reward_contract"]["component_bounds"],
        )

    def test_signal_contract_declares_debug_only_profile_provenance(self) -> None:
        signal_config = SignalConfig(
            communication_token_count=2,
            communication_profiles_per_token=3,
            communication_signal_decay_rate=0.42,
            max_signal_radius=5,
            max_duration_ticks=12,
        )
        contract = runtime_signals.signal_contract(signal_config)
        required_debug_fields = {
            "profile_id",
            "source_agent_id",
            "token_id",
            "profile_index",
            "radius",
            "duration_ticks",
            "decay_rate",
            "energy_cost",
            "emitted_tick",
        }

        self.assertEqual(contract["schema_version"], SIGNAL_CONTRACT_VERSION)
        self.assertEqual(contract["policy_semantics"], "opaque")
        self.assertTrue(contract["signal_substrate_enabled"])
        self.assertTrue(contract["reproductive_signal_emission_enabled"])
        self.assertFalse(contract["communication_signal_emission_enabled"])
        self.assertFalse(contract["profile_metadata_policy_visible"])
        self.assertTrue(
            required_debug_fields.issubset(
                set(contract["emission_debug_metadata_fields"])
            )
        )
        self.assertEqual(
            contract["reproductive_readiness_profile"]["field_name"],
            "reproductive_signal",
        )
        self.assertFalse(contract["reproductive_readiness_profile"]["policy_visible"])
        self.assertEqual(contract["communication_token_count"], 2)
        self.assertEqual(contract["communication_profiles_per_token"], 3)
        self.assertEqual(contract["communication_signal_decay_rate"], 0.42)
        self.assertEqual(contract["max_signal_radius"], 5)
        self.assertEqual(contract["max_duration_ticks"], 12)
        self.assertEqual(len(contract["reserved_profiles"]), 6)
        self.assertTrue(
            all(
                not profile["policy_visible"]
                for profile in contract["reserved_profiles"]
            )
        )
        self.assertTrue(
            all(
                profile["field_name"] == "communication_signal"
                for profile in contract["reserved_profiles"]
            )
        )
        self.assertEqual(
            [
                (profile["token_id"], profile["profile_index"])
                for profile in contract["reserved_profiles"]
            ],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        )

    def test_signal_config_rejects_invalid_scaffold_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "signals.enabled"):
            SignalConfig(enabled=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(
            ValueError,
            "signals.reproductive_signal_emission_enabled",
        ):
            SignalConfig(reproductive_signal_emission_enabled=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "signals.communication_token_count"):
            SignalConfig(communication_token_count=0)
        with self.assertRaisesRegex(ValueError, "signals.communication_token_count"):
            SignalConfig(
                communication_token_count=(
                    SignalConfig.MAX_COMMUNICATION_TOKEN_COUNT + 1
                )
            )
        with self.assertRaisesRegex(
            ValueError,
            "signals.communication_profiles_per_token",
        ):
            SignalConfig(
                communication_profiles_per_token=(
                    SignalConfig.MAX_COMMUNICATION_PROFILES_PER_TOKEN + 1
                )
            )
        with self.assertRaisesRegex(ValueError, "signals.max_signal_radius"):
            SignalConfig(max_signal_radius=-1)
        with self.assertRaisesRegex(ValueError, "signals.reproductive_signal_radius"):
            SignalConfig(reproductive_signal_radius=9)
        with self.assertRaisesRegex(
            ValueError,
            "signals.reproductive_signal_duration_ticks",
        ):
            SignalConfig(reproductive_signal_duration_ticks=25)
        with self.assertRaisesRegex(
            ValueError,
            "signals.reproductive_signal_decay_rate",
        ):
            SignalConfig(reproductive_signal_decay_rate=1.1)
        with self.assertRaisesRegex(
            ValueError,
            "signals.communication_signal_decay_rate",
        ):
            SignalConfig(communication_signal_decay_rate=1.1)
        with self.assertRaisesRegex(
            ValueError,
            "signals.communication_signal_emission_enabled",
        ):
            SignalConfig(communication_signal_emission_enabled=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(
            ValueError,
            "signals.communication_signal_radius",
        ):
            SignalConfig(communication_signal_radius=9)
        with self.assertRaisesRegex(
            ValueError,
            "signals.communication_signal_duration_ticks",
        ):
            SignalConfig(communication_signal_duration_ticks=25)
        with self.assertRaisesRegex(
            ValueError,
            "signals.communication_signal_base_intensity",
        ):
            SignalConfig(communication_signal_base_intensity=0.9)
        with self.assertRaisesRegex(
            ValueError,
            "signals.reproductive_signal_base_intensity",
        ):
            SignalConfig(reproductive_signal_base_intensity=0.9)

    def test_reproduction_phase_runs_signaling_and_birth_flow(self) -> None:
        world = SimulationWorld(
            self._ready_reproduction_config(width=5, height=5, max_agents=20)
        )
        parent = self._place_ready_agent(world, x=2, y=2)

        births = runtime_reproduction.run_reproduction_phase(
            world,
            context=world._reproduction_context(),
        )

        child = next(
            agent
            for agent in world.agents.values()
            if agent.parent_id == parent.agent_id
        )
        self.assertEqual(births, 1)
        self.assertEqual(world.births, 1)
        self.assertEqual(world.tick_birth_pairs, [(parent.agent_id, child.agent_id)])
        self.assertEqual(world.tick_signal_totals["reproductive_emissions"], 1.0)
        self.assertEqual(world.run_signal_totals["reproductive_emissions"], 1.0)
        self.assertEqual(len(world.tick_signal_emission_events), 1)
        self.assertEqual(
            world.run_reproduction_mate_search_counts,
            runtime_reproduction.empty_reproduction_mate_search_counts(),
        )
        self.assertEqual(
            world.tick_signal_emission_events[0]["source_agent_id"],
            parent.agent_id,
        )

    def test_full_replay_signal_contract_tracks_world_signal_config(self) -> None:
        config = WorldConfig(
            seed=7,
            max_ticks=2,
            signals=SignalConfig(
                communication_token_count=2,
                communication_profiles_per_token=3,
                communication_signal_decay_rate=0.42,
                max_signal_radius=5,
                max_duration_ticks=12,
            ),
        )

        result = SimulationWorld(config).run()
        signal_contract = result.viewer["trajectory"]["observation_contract"][
            "signal_contract"
        ]

        self.assertEqual(signal_contract["schema_version"], SIGNAL_CONTRACT_VERSION)
        self.assertEqual(signal_contract["communication_token_count"], 2)
        self.assertEqual(signal_contract["communication_profiles_per_token"], 3)
        self.assertEqual(signal_contract["communication_signal_decay_rate"], 0.42)
        self.assertEqual(signal_contract["max_signal_radius"], 5)
        self.assertEqual(signal_contract["max_duration_ticks"], 12)
        self.assertEqual(len(signal_contract["reserved_profiles"]), 6)
        action_contract_payload = result.viewer["trajectory"]["action_contract"]
        self.assertEqual(action_contract_payload["communication"]["token_count"], 2)
        self.assertEqual(
            action_contract_payload["communication"]["profiles_per_token"],
            3,
        )
        self.assertIn(
            "signal_1_profile_2",
            action_contract_payload["reserved_action_keys"],
        )
        self.assertNotIn(
            "signal_2_profile_0",
            action_contract_payload["reserved_action_keys"],
        )

    def test_signal_reproduction_config_edges_keep_contracts_deterministic(
        self,
    ) -> None:
        cases = {
            "signals_disabled_communication_requested": WorldConfig(
                seed=7,
                max_ticks=6,
                signals=SignalConfig(
                    enabled=False,
                    communication_signal_emission_enabled=True,
                ),
            ),
            "communication_duration_zero": WorldConfig(
                seed=7,
                max_ticks=6,
                signals=SignalConfig(
                    communication_signal_emission_enabled=True,
                    communication_signal_duration_ticks=0,
                ),
            ),
            "communication_enabled_small_capacity": WorldConfig(
                seed=7,
                max_ticks=6,
                signals=SignalConfig(
                    communication_signal_emission_enabled=True,
                    communication_token_count=2,
                    communication_profiles_per_token=1,
                    communication_signal_base_intensity=0.1,
                    communication_signal_trait_intensity_bonus=0.0,
                ),
            ),
            "sexual_disabled_low_thresholds": WorldConfig(
                seed=7,
                max_ticks=6,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    sexual_reproduction_enabled=False,
                    sexual_drive_threshold=0.0,
                    sexual_recombination_threshold=0.0,
                ),
            ),
        }

        for name, config in cases.items():
            with self.subTest(name=name):
                first = SimulationWorld(config).run()
                second = SimulationWorld(copy.deepcopy(config)).run()
                self.assertEqual(first.summary, second.summary)
                self.assertEqual(first.events, second.events)

                trajectory = first.viewer["trajectory"]
                signal_contract = trajectory["observation_contract"]["signal_contract"]
                action_contract_payload = trajectory["action_contract"]
                communication_actions = set(
                    action_contract_payload["communication"]["action_keys"]
                )
                active_actions = set(action_contract_payload["active_action_keys"])
                self.assertEqual(
                    action_contract_payload["communication"]["emission_enabled"],
                    signal_contract["communication_signal_emission_enabled"],
                )
                if signal_contract["communication_signal_emission_enabled"]:
                    self.assertTrue(communication_actions.issubset(active_actions))
                else:
                    self.assertTrue(communication_actions.isdisjoint(active_actions))

                reproduction_events = [
                    event
                    for event in first.events
                    if event.get("type") == "agent_reproduced"
                ]
                group_summary = first.summary["reproductive_groups_end"]
                self.assertEqual(
                    group_summary["asexual_births"] + group_summary["sexual_births"],
                    len(reproduction_events),
                )
                self.assertLessEqual(
                    group_summary["hybrid_births"],
                    group_summary["sexual_births"],
                )
                if name == "sexual_disabled_low_thresholds":
                    self.assertEqual(
                        first.summary["reproduction_end"]["reproductive_stage_counts"],
                        {STAGE0_ASEXUAL: first.summary["alive_agents"]},
                    )
