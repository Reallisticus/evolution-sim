from __future__ import annotations

import copy
import gzip
import json
import unittest
from dataclasses import replace
from inspect import signature
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.config import (
    CarcassConfig,
    ClimateConfig,
    CombatConfig,
    DietMatchingConfig,
    HazardConfig,
    ReproductionConfig,
    WorldConfig,
    SignalConfig,
)
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.contracts import (
    FULL_ONLY_SUMMARY_FIELDS,
    REPLAY_TOP_LEVEL_KEYS,
    SHARED_SUMMARY_FIELDS,
    VIEWER_AGENT_ENCODING,
    VIEWER_MAP_KEYS,
)
from evolution_sim.env.runtime.biotic import diffuse_biotic_field, diffuse_sparse_biotic_field
from evolution_sim.env.runtime.action_contract import (
    ACTION_CONTRACT_VERSION,
    ACTIVE_ACTION_NAMES,
    MATE_ACTION,
    RESERVED_ACTION_NAMES,
    action_contract,
)
from evolution_sim.env.runtime.action_space import build_action_mask
from evolution_sim.env.runtime.signals import SIGNAL_CONTRACT_VERSION
import evolution_sim.env.runtime.signals as runtime_signals
from evolution_sim.env.runtime.state import (
    MIND_INHERITANCE_PLACEHOLDER_VERSION,
    Agent,
    CarcassDeposit,
    SimulationWorldResult,
)
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_DTYPE,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    PATCH_CELL_COUNT,
    build_observation,
    decode_observation_input,
    encode_observation_input,
    observation_digest,
    observation_contract,
)
from evolution_sim.env.runtime.policy import (
    ActionDecision,
    OBSERVATION_HEURISTIC_POLICY_ID,
    OBSERVATION_HEURISTIC_POLICY_VERSION,
    ObservationHeuristicPolicy,
    POLICY_INTERFACE_VERSION,
)
import evolution_sim.env.runtime.reproduction as runtime_reproduction
from evolution_sim.env.runtime.reproduction import (
    REPRODUCTIVE_GROUP_CONTRACT_VERSION,
    STAGE0_ASEXUAL,
    reproductive_group_contract,
)
from evolution_sim.env.runtime.trajectory import (
    ACTION_OUTCOME_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    build_reward,
    reward_contract,
)
from evolution_sim.env.runtime.lifecycle import genome_profile_key
from evolution_sim.env.runtime.mating import (
    SEXUAL_EXPRESSION,
    SEXUAL_REPRODUCTION_MODE,
    STAGE1_FACULTATIVE_SEX,
)
from evolution_sim.genome import Genome, ReproductiveGenome
from evolution_sim.genome.recombination import (
    GENOME_GROUPS,
    GENOME_RECOMBINATION_CONTRACT_VERSION,
    genome_recombination_contract,
    recombine_genomes,
)
from evolution_sim.genome.schema import GENE_LIMITS, REPRODUCTIVE_GENE_LIMITS
from evolution_sim.genome.species import genome_vector
from evolution_sim.env.taxonomy import REPLAY_TAXONOMY_MODE, apply_replay_taxonomy
from evolution_sim.io import JsonlTrajectoryWriter, write_json_replay

GOLDEN_SPECIATION_SEED = 3


class RuntimeContractTests(unittest.TestCase):
    def _policy_action_mask(self) -> dict[str, bool]:
        return {
            "stay": True,
            "move_north": True,
            "move_south": True,
            "move_east": True,
            "move_west": True,
            "eat": True,
            "drink": False,
            "reproduce": False,
            "attack_north": False,
            "attack_south": False,
            "attack_east": False,
            "attack_west": False,
        }

    def _policy_cell(self, dx: int, dy: int, **overrides: object) -> dict[str, object]:
        cell: dict[str, object] = {
            "dx": dx,
            "dy": dy,
            "in_bounds": True,
            "terrain": "plain",
            "occupant": "none",
            "food": 0.0,
            "vegetation": 0.0,
            "recovery_debt": 0.0,
            "hazard_level": 0.0,
            "water_access_reason": "none",
            "fresh_kill_energy": 0.0,
            "carcass_energy": 0.0,
            "prey_biomass": 0.0,
            "carrion_signal": 0.0,
            "predator_risk": 0.0,
        }
        cell.update(overrides)
        return cell

    def _policy_observation(
        self,
        *,
        energy_ratio: float,
        hydration_ratio: float,
        navigation: dict[str, dict[str, object]],
        center_food: float = 0.0,
        center_hazard_level: float = 0.0,
        trophic_role: str = "carnivore",
        meat_mode: str = "hunter",
        health_ratio: float = 1.0,
        matched_diet_ratio: float = 1.0,
    ) -> dict[str, object]:
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "metadata": {},
            "self": {
                "energy_ratio": energy_ratio,
                "hydration_ratio": hydration_ratio,
                "health_ratio": health_ratio,
                "age_ratio": 0.1,
                "matched_diet_ratio": matched_diet_ratio,
                "trophic_role": trophic_role,
                "meat_mode": meat_mode,
            },
            "local_patch": [
                self._policy_cell(0, 0, food=center_food, hazard_level=center_hazard_level),
            ],
            "navigation": navigation,
            "action_mask": self._policy_action_mask(),
        }

    def _empty_navigation(self) -> dict[str, dict[str, object]]:
        return {
            "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
        }

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

    def test_starving_scavenger_moves_to_nearby_low_signal_carrion_before_plant(self) -> None:
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

    def _ready_reproduction_config(self, **overrides: object) -> WorldConfig:
        values = {
            "seed": 7,
            "max_ticks": 1,
            "initial_agents": 0,
            "water_tile_ratio": 0.0,
            "forest_tile_ratio": 0.0,
            "wetland_tile_ratio": 0.0,
            "rocky_tile_ratio": 0.0,
            "base_energy_drain": 0.0,
            "base_hydration_drain": 0.0,
            "reproduction": ReproductionConfig(
                min_age=1,
                cooldown_ticks=0,
                min_hydration_fraction=0.0,
                energy_cost=0.0,
            ),
            "diet_matching": DietMatchingConfig(
                specialist_threshold=0.0,
                omnivore_threshold=0.0,
            ),
            "combat": CombatConfig(
                min_attack_health_ratio=1.0,
                min_attack_energy_ratio=1.0,
                min_attack_hydration_ratio=1.0,
            ),
        }
        values.update(overrides)
        return WorldConfig(**values)

    def _place_ready_agent(
        self,
        world: SimulationWorld,
        *,
        x: int,
        y: int,
        lineage_id: int = 1,
        genome: Genome | None = None,
        reproductive_group_id: int | None = None,
        reproductive_stage: str = "stage0_asexual",
        reproductive_expression: str = "asexual",
    ) -> Agent:
        genome = genome or Genome.sample_initial(world.rng)
        agent = Agent(
            agent_id=world.next_agent_id,
            parent_id=None,
            lineage_id=lineage_id,
            birth_tick=0,
            death_tick=None,
            x=x,
            y=y,
            energy=genome.max_energy * 1.25,
            hydration=genome.max_hydration,
            health=genome.max_health,
            max_health=genome.max_health,
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
            reproductive_group_id=reproductive_group_id or lineage_id,
            reproductive_stage=reproductive_stage,
            reproductive_expression=reproductive_expression,
        )
        world._place_agent(agent)
        world.next_agent_id += 1
        return agent

    def _hunter_genome(self) -> Genome:
        return Genome(
            max_energy=1.0,
            max_hydration=1.0,
            max_health=1.0,
            move_cost=0.03,
            food_efficiency=0.4,
            water_efficiency=1.0,
            attack_power=1.2,
            attack_cost_multiplier=1.0,
            defense_rating=0.9,
            meat_efficiency=1.8,
            healing_efficiency=1.0,
            plant_bias=0.35,
            carrion_bias=0.2,
            live_prey_bias=1.8,
            forest_affinity=1.0,
            plain_affinity=1.0,
            wetland_affinity=1.0,
            rocky_affinity=1.0,
            heat_tolerance=1.0,
            reproduction_threshold=0.7,
            mutation_scale=0.01,
        )

    def _mixed_genome(self) -> Genome:
        return Genome(
            max_energy=1.0,
            max_hydration=1.0,
            max_health=1.0,
            move_cost=0.03,
            food_efficiency=1.0,
            water_efficiency=1.0,
            attack_power=0.8,
            attack_cost_multiplier=1.0,
            defense_rating=0.9,
            meat_efficiency=1.1,
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

    def _scavenger_genome(self) -> Genome:
        return Genome(
            max_energy=1.0,
            max_hydration=1.0,
            max_health=1.0,
            move_cost=0.03,
            food_efficiency=0.4,
            water_efficiency=1.0,
            attack_power=0.35,
            attack_cost_multiplier=1.45,
            defense_rating=0.45,
            meat_efficiency=1.8,
            healing_efficiency=1.0,
            plant_bias=0.45,
            carrion_bias=1.8,
            live_prey_bias=0.2,
            forest_affinity=1.0,
            plain_affinity=1.0,
            wetland_affinity=1.0,
            rocky_affinity=1.0,
            heat_tolerance=1.0,
            reproduction_threshold=0.7,
            mutation_scale=0.01,
        )

    def _sexualized_genome(self, genome: Genome) -> Genome:
        return replace(
            genome,
            reproductive=ReproductiveGenome(
                sexual_reproduction_drive=0.9,
                recombination_affinity=0.9,
                role_differentiation_drive=0.0,
                sex_expression_bias=0.0,
                sex_plasticity=0.0,
                hybridization_tolerance=0.0,
                fecundity_potential=0.0,
                signal_emission_bias=0.0,
                signal_sensitivity=0.0,
            ),
        )

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

        action_mask = build_action_mask(world, agent)
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

        action_mask = build_action_mask(world, agent)
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

        action_mask = build_action_mask(world, agent)
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

        navigation = build_observation(world, agent)["navigation"]["carrion"]

        self.assertEqual(navigation["dx"], 0)
        self.assertEqual(navigation["dy"], -1)
        self.assertEqual(navigation["distance"], 4)
        self.assertGreater(navigation["strength"], 0.0)

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

    def _run_scripted_lethal_attack(self) -> tuple[SimulationWorldResult, int, int]:
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
                combat=CombatConfig(
                    min_attack_health_ratio=0.0,
                    min_attack_energy_ratio=0.0,
                    min_attack_hydration_ratio=0.0,
                    base_attack_damage=3.0,
                ),
                reproduction=ReproductionConfig(min_age=720),
            )
        )
        base_genome = Genome.sample_initial(world.rng)
        hunter = replace(
            base_genome,
            attack_power=1.8,
            meat_efficiency=1.8,
            live_prey_bias=1.8,
            carrion_bias=0.2,
            plant_bias=0.45,
        )
        prey_genome = replace(
            base_genome,
            max_health=0.7,
            defense_rating=0.45,
            attack_power=0.35,
            plant_bias=1.8,
            live_prey_bias=0.2,
            carrion_bias=0.2,
        )
        attacker = self._place_ready_agent(
            world,
            x=1,
            y=1,
            lineage_id=1,
            genome=hunter,
        )
        target = self._place_ready_agent(
            world,
            x=1,
            y=0,
            lineage_id=2,
            genome=prey_genome,
        )
        target.health = 0.05
        world.current_species_map = {
            agent.agent_id: agent.lineage_id for agent in world.alive_agents()
        }
        world.agent_last_species_map = world.current_species_map.copy()

        def choose_scripted_action(
            agent: Agent,
            observation: dict[str, object] | None = None,
        ) -> str:
            world._policy_action_source = "scripted_attack"
            world._policy_id = "scripted_attack"
            world._policy_version = "scripted_attack_v1"
            return "attack_north" if agent.agent_id == attacker.agent_id else "stay"

        with patch.object(world, "_choose_action", side_effect=choose_scripted_action):
            result = world.run(mode=RunMode.FULL_REPLAY)
        return result, attacker.agent_id, target.agent_id

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
        self.assertEqual(tuple(summary_only.summary), SHARED_SUMMARY_FIELDS)
        for field in SHARED_SUMMARY_FIELDS:
            self.assertEqual(summary_only.summary[field], full.summary[field], msg=field)
        for field in FULL_ONLY_SUMMARY_FIELDS:
            self.assertNotIn(field, summary_only.summary)

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
        self.assertEqual(
            result.summary["reproduction_end"]["blocked_run_counts"]["max_population"],
            1,
        )
        self.assertEqual(blocked_events[0]["data"]["reason"], "max_population")

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
        encoded = encode_observation_input(observation)
        decoded = decode_observation_input(encoded)
        contract = observation_contract()

        self.assertEqual(observation["schema_version"], OBSERVATION_SCHEMA_VERSION)
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
        self.assertNotIn("agent_id", observation)
        self.assertEqual(observation["metadata"], {"agent_id": agent.agent_id})
        self.assertEqual(len(observation["local_patch"]), PATCH_CELL_COUNT)
        self.assertNotIn("world", observation)
        self.assertNotIn("grid", observation)
        self.assertNotIn("agents", observation)
        self.assertTrue(contract["metadata_policy_excluded"])
        self.assertEqual(
            contract["policy_input"]["encoder_version"],
            OBSERVATION_ENCODER_VERSION,
        )
        self.assertEqual(
            contract["action_contract"]["schema_version"],
            ACTION_CONTRACT_VERSION,
        )
        self.assertEqual(
            contract["signal_contract"]["schema_version"],
            SIGNAL_CONTRACT_VERSION,
        )
        self.assertFalse(contract["mind_inheritance_placeholder"]["policy_visible"])
        self.assertEqual(contract["policy_input"]["shape"], [OBSERVATION_INPUT_VECTOR_SIZE])
        self.assertEqual(encoded["decoded_dtype"], OBSERVATION_INPUT_DTYPE)
        self.assertEqual(encoded["shape"], [OBSERVATION_INPUT_VECTOR_SIZE])
        self.assertEqual(len(decoded), OBSERVATION_INPUT_VECTOR_SIZE)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in decoded))
        self.assertTrue(all(isinstance(value, float) for value in decoded))
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)
        json.dumps(observation)
        json.dumps(encoded)

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

    def test_pre_mind_reproductive_slots_start_without_emitted_signals(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        observation = build_observation(world, agent)
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
                width=5,
                height=5,
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

        totals = runtime_signals.emit_reproductive_readiness_signals(world, [agent])
        signal_state = world._current_signal_state()
        observation = build_observation(world, agent)
        decoded = decode_observation_input(encode_observation_input(observation))

        self.assertEqual(totals["reproductive_emissions"], 1)
        self.assertEqual(totals["communication_emissions"], 0)
        self.assertAlmostEqual(float(totals["energy_spent"]), 0.025)
        self.assertAlmostEqual(agent.energy, energy_before - 0.025)
        self.assertAlmostEqual(signal_state.reproductive_signal[2][2], 0.5)
        self.assertGreater(signal_state.reproductive_signal[2][2], 0.0)
        self.assertGreater(signal_state.reproductive_signal[2][2], signal_state.reproductive_signal[2][3])
        self.assertEqual(signal_state.communication_signal[2][2], 0.0)
        self.assertGreater(observation["self"]["reproductive_signal"], 0.0)
        self.assertEqual(observation["self"]["communication_signal"], 0.0)
        self.assertTrue(
            any(cell["reproductive_signal"] > 0.0 for cell in observation["local_patch"])
        )
        self.assertEqual(len(decoded), OBSERVATION_INPUT_VECTOR_SIZE)

        runtime_signals.decay_signal_emissions(world)
        decayed_state = world._current_signal_state()

        self.assertLess(
            decayed_state.reproductive_signal[2][2],
            signal_state.reproductive_signal[2][2],
        )
        self.assertEqual(decayed_state.communication_signal[2][2], 0.0)

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

        totals = runtime_signals.emit_reproductive_readiness_signals(world, [agent])
        signal_state = world._current_signal_state()

        self.assertEqual(totals["reproductive_emissions"], 0)
        self.assertEqual(signal_state.reproductive_signal[2][2], 0.0)
        self.assertEqual(world.run_signal_totals["reproductive_emissions"], 0.0)

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
            "signals.reproductive_signal_base_intensity",
        ):
            SignalConfig(reproductive_signal_base_intensity=0.9)


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
        for agent in result.viewer["agent_catalog"].values():
            group_id = str(agent["reproductive_group_id"])
            self.assertIn(group_id, catalog["groups"])
            self.assertEqual(agent["reproductive_stage"], STAGE0_ASEXUAL)
            self.assertEqual(agent["reproductive_expression"], "asexual")

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
        self.assertEqual(summary["reproductive_groups_end"]["sexual_births"], 1)
        self.assertEqual(summary["reproductive_groups_end"]["asexual_births"], 0)

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
        self.assertEqual(summary["reproductive_groups_end"]["sexual_births"], 0)
        self.assertEqual(summary["reproductive_groups_end"]["asexual_births"], 1)

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
        ), patch.object(
            SimulationWorld,
            "_build_viewer_payload",
            side_effect=AssertionError("trajectory summary mode should not build viewer payload"),
        ):
            world = SimulationWorld(WorldConfig(seed=7, max_ticks=4))
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
            world = SimulationWorld(WorldConfig(seed=7, max_ticks=4))

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
        self.assertEqual(header["type"], "header")
        self.assertEqual(header["format"], "evolution_sim_trajectory_jsonl_v1")
        self.assertEqual(
            header["trajectory_contract"]["schema_version"],
            TRAJECTORY_SCHEMA_VERSION,
        )
        self.assertEqual(
            header["trajectory_contract"]["policy_interface_version"],
            POLICY_INTERFACE_VERSION,
        )
        self.assertEqual(
            header["trajectory_contract"]["reproductive_group_contract_version"],
            REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        )
        self.assertEqual(
            header["trajectory_contract"]["genome_recombination_contract_version"],
            GENOME_RECOMBINATION_CONTRACT_VERSION,
        )
        self.assertEqual(footer["type"], "footer")
        self.assertEqual(
            footer["trajectory_summary"]["policy_interface_version"],
            POLICY_INTERFACE_VERSION,
        )
        self.assertEqual(
            footer["trajectory_summary"]["reproductive_group_contract_version"],
            REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        )
        self.assertEqual(
            footer["trajectory_summary"]["genome_recombination_contract_version"],
            GENOME_RECOMBINATION_CONTRACT_VERSION,
        )
        self.assertEqual(footer["trajectory_summary"]["record_count"], len(records))
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
