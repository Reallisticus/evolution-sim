from __future__ import annotations

from python.tests.runtime_test_helpers import *


class RuntimeActionContractTests(RuntimeContractTestHelpers):
    def test_desperate_meat_policy_prioritizes_urgent_water_over_prey(self) -> None:
        navigation = self._empty_navigation()
        navigation["water"] = {"dx": 1, "dy": 0, "distance": 1, "strength": 1.0}
        navigation["prey"] = {"dx": -3, "dy": 0, "distance": 3, "strength": 1.0}
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.52,
                hydration_ratio=0.42,
                navigation=navigation,
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_desperate_meat_policy_uses_carrion_before_distant_prey(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 1, "dy": 0, "distance": 1, "strength": 0.2}
        navigation["prey"] = {"dx": -4, "dy": 0, "distance": 4, "strength": 1.0}
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.45,
                hydration_ratio=0.75,
                navigation=navigation,
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_scavenger_policy_uses_carrion_before_plant_fallback(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 1, "dy": 0, "distance": 1, "strength": 0.2}
        navigation["plant"] = {"dx": -1, "dy": 0, "distance": 1, "strength": 0.8}
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.82,
                hydration_ratio=0.8,
                navigation=navigation,
                center_food=0.3,
                trophic_role="omnivore",
                meat_mode="scavenger",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_scavenger_policy_uses_local_food_before_out_of_range_carrion(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 9, "dy": 0, "distance": 9, "strength": 0.2}
        navigation["plant"] = {"dx": -1, "dy": 0, "distance": 1, "strength": 0.8}
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.82,
                hydration_ratio=0.8,
                navigation=navigation,
                center_food=0.3,
                trophic_role="omnivore",
                meat_mode="scavenger",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "eat")

    def test_starving_meat_policy_uses_local_food_before_carrion_chase(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 4, "dy": 0, "distance": 4, "strength": 0.6}

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.28,
                hydration_ratio=0.7,
                navigation=navigation,
                center_food=0.9,
                trophic_role="omnivore",
                meat_mode="hunter",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "eat")

    def test_low_hydration_meat_policy_uses_water_before_carrion_chase(self) -> None:
        navigation = self._empty_navigation()
        navigation["water"] = {"dx": -1, "dy": 0, "distance": 1, "strength": 1.0}
        navigation["carrion"] = {"dx": 4, "dy": 0, "distance": 4, "strength": 0.6}

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.45,
                hydration_ratio=0.6,
                navigation=navigation,
                trophic_role="carnivore",
                meat_mode="scavenger",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_west")

    def test_thirsty_scavenger_uses_current_carcass_when_energy_is_high(self) -> None:
        observation = self._policy_observation(
            energy_ratio=0.98,
            hydration_ratio=0.5,
            navigation=self._empty_navigation(),
            trophic_role="carnivore",
            meat_mode="scavenger",
        )
        observation["local_patch"][0]["carcass_energy"] = 0.3

        decision = ObservationHeuristicPolicy().decide(
            observation,
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "eat")

    def test_scavenger_policy_eats_adjacent_blocked_carcass(self) -> None:
        observation = self._policy_observation(
            energy_ratio=0.34,
            hydration_ratio=0.7,
            navigation=self._empty_navigation(),
            trophic_role="carnivore",
            meat_mode="scavenger",
        )
        observation["local_patch"].append(
            self._policy_cell(
                1,
                0,
                occupant="agent",
                carcass_energy=0.4,
            )
        )

        decision = ObservationHeuristicPolicy().decide(
            observation,
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "eat")

    def test_thirsty_starving_scavenger_uses_local_food_before_long_carrion(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 4, "dy": 0, "distance": 4, "strength": 0.6}

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.45,
                hydration_ratio=0.5,
                navigation=navigation,
                center_food=0.9,
                trophic_role="carnivore",
                meat_mode="scavenger",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "eat")

    def test_starving_scavenger_eats_local_food_before_long_carrion_chase(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 4, "dy": 0, "distance": 4, "strength": 0.6}

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.28,
                hydration_ratio=0.7,
                navigation=navigation,
                center_food=0.9,
                trophic_role="carnivore",
                meat_mode="scavenger",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "eat")

    def test_desperate_meat_policy_does_not_conserve_on_hazard(self) -> None:
        navigation = self._empty_navigation()
        navigation["plant"] = {"dx": 1, "dy": 0, "distance": 1, "strength": 1.0}

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.35,
                hydration_ratio=0.7,
                navigation=navigation,
                center_hazard_level=0.48,
                trophic_role="carnivore",
                meat_mode="scavenger",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_desperate_animal_mode_forages_locally_before_long_carrion_chase(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 9, "dy": 0, "distance": 9, "strength": 1.0}
        navigation["plant"] = {"dx": -1, "dy": 0, "distance": 1, "strength": 0.8}

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.45,
                hydration_ratio=0.75,
                navigation=navigation,
                trophic_role="carnivore",
                meat_mode="scavenger",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_west")

    def test_desperate_scavenger_detours_when_direct_carrion_step_is_blocked(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": -8, "dy": 0, "distance": 8, "strength": 0.5}
        action_mask = self._policy_action_mask()
        action_mask["move_west"] = False

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.45,
                hydration_ratio=0.75,
                navigation=navigation,
                trophic_role="carnivore",
                meat_mode="scavenger",
            ),
            action_mask,
        )

        self.assertEqual(decision.requested_action, "move_north")

    def test_mixed_policy_seeks_nearby_carrion_before_plant_fallback(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 2, "dy": 0, "distance": 2, "strength": 0.4}
        navigation["plant"] = {"dx": -1, "dy": 0, "distance": 1, "strength": 1.0}

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.88,
                hydration_ratio=0.8,
                navigation=navigation,
                center_food=0.8,
                trophic_role="omnivore",
                meat_mode="mixed",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_mixed_policy_keeps_release_horizon_carrion_actionable(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 10, "dy": 0, "distance": 10, "strength": 0.9}
        navigation["plant"] = {"dx": -1, "dy": 0, "distance": 1, "strength": 1.0}

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.62,
                hydration_ratio=0.74,
                navigation=navigation,
                center_food=0.9,
                trophic_role="omnivore",
                meat_mode="mixed",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_mixed_policy_uses_local_carrion_resource_before_rich_plant(self) -> None:
        navigation = self._empty_navigation()
        navigation["plant"] = {"dx": -1, "dy": 0, "distance": 1, "strength": 1.0}
        observation = self._policy_observation(
            energy_ratio=0.7,
            hydration_ratio=0.86,
            navigation=navigation,
            center_food=0.9,
            trophic_role="omnivore",
            meat_mode="mixed",
        )
        observation["local_patch"].append(
            self._policy_cell(1, 1, fresh_kill_energy=0.04, food=0.8)
        )

        decision = ObservationHeuristicPolicy().decide(
            observation,
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_hunter_policy_seeks_nearby_carrion_when_no_adjacent_prey(self) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {"dx": 0, "dy": 2, "distance": 2, "strength": 0.4}

        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.8,
                hydration_ratio=0.8,
                navigation=navigation,
                center_food=0.8,
                trophic_role="carnivore",
                meat_mode="hunter",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_south")

    def test_hunter_policy_attacks_adjacent_prey_before_plant_fallback(self) -> None:
        navigation = self._empty_navigation()
        observation = self._policy_observation(
            energy_ratio=0.82,
            hydration_ratio=0.8,
            navigation=navigation,
            center_food=0.8,
            trophic_role="omnivore",
            meat_mode="hunter",
        )
        observation["local_patch"].append(
            self._policy_cell(
                1,
                0,
                occupant="agent",
                prey_biomass=1.0,
                predator_risk=0.0,
            )
        )
        action_mask = self._policy_action_mask()
        action_mask["attack_east"] = True

        decision = ObservationHeuristicPolicy().decide(observation, action_mask)

        self.assertEqual(decision.requested_action, "attack_east")

    def test_desperate_meat_policy_conserves_instead_of_chasing_distant_prey(self) -> None:
        navigation = self._empty_navigation()
        navigation["prey"] = {"dx": -4, "dy": 0, "distance": 4, "strength": 1.0}
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.45,
                hydration_ratio=0.75,
                navigation=navigation,
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "stay")

    def test_hunter_matched_diet_deficit_pursues_nearby_prey_before_plants(self) -> None:
        navigation = self._empty_navigation()
        navigation["prey"] = {"dx": 1, "dy": 0, "distance": 1, "strength": 1.0}
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.86,
                hydration_ratio=0.82,
                navigation=navigation,
                center_food=0.9,
                trophic_role="omnivore",
                meat_mode="hunter",
                matched_diet_ratio=0.0,
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_hunter_pursues_water_before_low_energy_plant_fallback(self) -> None:
        navigation = self._empty_navigation()
        observation = self._policy_observation(
            energy_ratio=0.48,
            hydration_ratio=0.62,
            navigation=navigation,
            center_food=0.9,
            trophic_role="carnivore",
            meat_mode="hunter",
        )
        observation["local_patch"].append(
            self._policy_cell(1, 0, water_access_reason="adjacent_water")
        )

        decision = ObservationHeuristicPolicy().decide(
            observation,
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_hunter_eats_local_food_before_nonadjacent_water_when_critically_weak(
        self,
    ) -> None:
        navigation = self._empty_navigation()
        navigation["water"] = {"dx": 1, "dy": 0, "distance": 3, "strength": 1.0}
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.3,
                hydration_ratio=0.12,
                navigation=navigation,
                center_food=0.3,
                trophic_role="carnivore",
                meat_mode="hunter",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "eat")

    def test_hunter_still_pursues_adjacent_water_when_critically_thirsty(self) -> None:
        navigation = self._empty_navigation()
        navigation["water"] = {"dx": 1, "dy": 0, "distance": 1, "strength": 1.0}
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.3,
                hydration_ratio=0.12,
                navigation=navigation,
                center_food=0.3,
                trophic_role="carnivore",
                meat_mode="hunter",
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_hunter_waits_when_critical_water_route_is_blocked(self) -> None:
        navigation = self._empty_navigation()
        navigation["water"] = {"dx": 0, "dy": 1, "distance": 1, "strength": 1.0}
        navigation["plant"] = {"dx": 0, "dy": -1, "distance": 1, "strength": 0.8}
        action_mask = self._policy_action_mask()
        action_mask["move_south"] = False
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.28,
                hydration_ratio=0.08,
                navigation=navigation,
                center_food=0.02,
                trophic_role="carnivore",
                meat_mode="hunter",
            ),
            action_mask,
        )

        self.assertEqual(decision.requested_action, "stay")

    def test_hunter_detours_when_nonadjacent_water_route_is_blocked(self) -> None:
        navigation = self._empty_navigation()
        navigation["water"] = {"dx": 0, "dy": 2, "distance": 2, "strength": 1.0}
        action_mask = self._policy_action_mask()
        action_mask["move_south"] = False
        action_mask["move_east"] = True
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.68,
                hydration_ratio=0.08,
                navigation=navigation,
                center_food=0.9,
                trophic_role="omnivore",
                meat_mode="mixed",
            ),
            action_mask,
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_hunter_eats_local_food_when_adjacent_water_route_blocked_and_starving(
        self,
    ) -> None:
        navigation = self._empty_navigation()
        navigation["water"] = {"dx": 0, "dy": 1, "distance": 1, "strength": 1.0}
        action_mask = self._policy_action_mask()
        action_mask["move_south"] = False
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.12,
                hydration_ratio=0.12,
                navigation=navigation,
                center_food=0.3,
                trophic_role="carnivore",
                meat_mode="hunter",
            ),
            action_mask,
        )

        self.assertEqual(decision.requested_action, "eat")

    def test_hunter_water_fallback_uses_full_local_patch_radius(self) -> None:
        navigation = self._empty_navigation()
        navigation["water"] = {"dx": 0, "dy": 1, "distance": 1, "strength": 1.0}
        observation = self._policy_observation(
            energy_ratio=0.48,
            hydration_ratio=0.62,
            navigation=navigation,
            center_food=0.9,
            trophic_role="carnivore",
            meat_mode="hunter",
        )
        observation["local_patch"].append(
            self._policy_cell(2, 2, water_access_reason="adjacent_water")
        )
        action_mask = self._policy_action_mask()
        action_mask["move_south"] = False

        decision = ObservationHeuristicPolicy().decide(
            observation,
            action_mask,
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_scavenger_matched_diet_deficit_follows_reachable_carrion_before_plants(
        self,
    ) -> None:
        navigation = self._empty_navigation()
        navigation["carrion"] = {
            "dx": 1,
            "dy": 0,
            "distance": 4,
            "strength": 0.5,
        }
        decision = ObservationHeuristicPolicy().decide(
            self._policy_observation(
                energy_ratio=0.22,
                hydration_ratio=0.72,
                navigation=navigation,
                center_food=0.9,
                trophic_role="carnivore",
                meat_mode="scavenger",
                health_ratio=0.28,
                matched_diet_ratio=0.0,
            ),
            self._policy_action_mask(),
        )

        self.assertEqual(decision.requested_action, "move_east")

    def test_hunter_fresh_kill_drive_has_specialist_floor(self) -> None:
        world = SimulationWorld(WorldConfig(seed=3, max_ticks=1))
        agent = next(
            agent
            for agent in world.alive_agents()
            if world._trophic_profile(agent).meat_mode == "hunter"
            and world._trophic_profile(agent).hunter_drive < 0.09
        )
        profile = world._trophic_profile(agent)
        raw_drive = profile.hunter_drive + profile.scavenger_drive * 0.28

        self.assertLess(raw_drive, 0.45)
        self.assertEqual(world._fresh_kill_drive(profile), 0.45)

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

    def test_action_mask_builder_uses_explicit_context(self) -> None:
        mask = build_action_mask_from_context(
            ActionMaskContext(
                action_names=(
                    "stay",
                    "move_north",
                    "move_south",
                    "move_east",
                    "move_west",
                    "eat",
                    "drink",
                    "attack_north",
                    "attack_south",
                    "attack_east",
                    "attack_west",
                    "mate",
                    "signal_0_profile_0",
                ),
                can_eat=True,
                can_drink=False,
                movement=(
                    MovementActionAvailability(
                        action="move_north",
                        dx=0,
                        dy=-1,
                        can_move=True,
                        can_attack=False,
                    ),
                    MovementActionAvailability(
                        action="move_east",
                        dx=1,
                        dy=0,
                        can_move=False,
                        can_attack=True,
                    ),
                ),
                communication_action_available={"signal_0_profile_0": True},
            )
        )

        self.assertTrue(mask["stay"])
        self.assertTrue(mask["eat"])
        self.assertFalse(mask["drink"])
        self.assertTrue(mask["move_north"])
        self.assertFalse(mask["attack_north"])
        self.assertFalse(mask["move_east"])
        self.assertTrue(mask["attack_east"])
        self.assertTrue(mask["signal_0_profile_0"])
        self.assertFalse(mask["move_south"])
        self.assertFalse(mask["mate"])

    def test_action_contract_reserves_future_slots_without_enabling_them(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        mask = build_action_mask(world, agent)
        contract = action_contract()

        self.assertEqual(contract["schema_version"], ACTION_CONTRACT_VERSION)
        self.assertEqual(contract["mate_action_key"], MATE_ACTION)
        for action in ACTIVE_ACTION_NAMES:
            self.assertIn(action, mask)
        for action in RESERVED_ACTION_NAMES:
            self.assertIn(action, mask)
            self.assertFalse(mask[action], msg=action)
            self.assertIn(action, contract["reserved_action_keys"])

        moved, outcome = world._resolve_action_with_outcome(
            agent,
            MATE_ACTION,
            observation_action_mask=mask,
            resolution_action_mask=mask,
        )

        self.assertFalse(moved)
        self.assertEqual(outcome["resolved_action"], "stay")
        self.assertFalse(outcome["observation_action_valid"])
        self.assertFalse(outcome["resolution_action_valid"])
        self.assertEqual(
            outcome["invalid_reason"],
            "not_in_observation_or_resolution_mask",
        )
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

    def test_policy_boundary_receives_observation_and_action_mask(self) -> None:
        class RecordingPolicy:
            policy_id = "recording_policy"
            policy_version = "recording_policy_v1"

            def __init__(self) -> None:
                self.calls: list[tuple[dict[str, object], dict[str, bool]]] = []

            def decide(
                self,
                observation: dict[str, object],
                action_mask: dict[str, bool],
            ) -> ActionDecision:
                self.calls.append((copy.deepcopy(observation), dict(action_mask)))
                return ActionDecision(
                    requested_action="stay",
                    source=self.policy_id,
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                )

        policy = RecordingPolicy()
        result = SimulationWorld(
            WorldConfig(seed=7, max_ticks=1),
            policy=policy,
        ).run(mode=RunMode.FULL_REPLAY)

        self.assertGreater(len(policy.calls), 0)
        observation, action_mask = policy.calls[0]
        self.assertEqual(
            set(observation),
            {
                "schema_version",
                "metadata",
                "self",
                "local_patch",
                "navigation",
                "action_mask",
            },
        )
        self.assertEqual(action_mask, observation["action_mask"])
        self.assertNotIn("agent_id", observation)
        first_record = result.viewer["trajectory"]["records"][0]
        self.assertEqual(first_record["requested_action"], "stay")
        self.assertEqual(first_record["action_source"], policy.policy_id)
        self.assertEqual(first_record["policy_id"], policy.policy_id)
        self.assertEqual(first_record["policy_version"], policy.policy_version)

    def test_default_policy_does_not_call_legacy_world_reading_heuristic(self) -> None:
        with patch(
            "evolution_sim.env.runtime.actions.choose_action",
            side_effect=AssertionError("default policy must not read live world state"),
        ):
            result = SimulationWorld(WorldConfig(seed=7, max_ticks=2)).run(
                mode=RunMode.SUMMARY_ONLY
            )

        self.assertEqual(tuple(result.summary), SHARED_SUMMARY_FIELDS)

    def test_action_outcome_records_resolution_mask_invalid_reason(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        observation_mask = world._action_mask(agent)
        resolution_mask = dict(observation_mask)
        observation_mask["eat"] = True
        resolution_mask["eat"] = False

        moved, outcome = world._resolve_action_with_outcome(
            agent,
            "eat",
            observation_action_mask=observation_mask,
            resolution_action_mask=resolution_mask,
        )

        self.assertFalse(moved)
        self.assertEqual(outcome["resolved_action"], "stay")
        self.assertTrue(outcome["observation_action_valid"])
        self.assertFalse(outcome["resolution_action_valid"])
        self.assertEqual(outcome["invalid_reason"], "not_in_resolution_action_mask")

    def test_action_outcome_records_resolution_mask_conflicts_by_action_family(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        base_mask = world._action_mask(agent)

        for action in ("move_east", "eat", "attack_north", MATE_ACTION):
            with self.subTest(action=action):
                observation_mask = dict(base_mask)
                resolution_mask = dict(base_mask)
                observation_mask[action] = True
                resolution_mask[action] = False

                moved, outcome = world._resolve_action_with_outcome(
                    agent,
                    action,
                    observation_action_mask=observation_mask,
                    resolution_action_mask=resolution_mask,
                )

                self.assertFalse(moved)
                self.assertEqual(outcome["resolved_action"], "stay")
                self.assertTrue(outcome["observation_action_valid"])
                self.assertFalse(outcome["resolution_action_valid"])
                self.assertEqual(
                    outcome["invalid_reason"],
                    "not_in_resolution_action_mask",
                )
                self.assertEqual(outcome["movement"], {"moved": False})
                self.assertEqual(outcome["feeding"], {"ate": False})
                self.assertEqual(outcome["attack"], {"attempted": False})
