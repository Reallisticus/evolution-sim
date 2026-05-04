from __future__ import annotations

import copy
import gzip
import json
import unittest
from contextlib import ExitStack
from dataclasses import dataclass, replace
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
from evolution_sim.env.events import EventType
from evolution_sim.env.world import MEAT_MODE_CODES, TROPHIC_ROLE_CODES
from evolution_sim.env.contracts import (
    FULL_ONLY_SUMMARY_FIELDS,
    REPLAY_TOP_LEVEL_KEYS,
    SHARED_SUMMARY_FIELDS,
    SUMMARY_SCHEMA_VERSION,
    VIEWER_AGENT_ENCODING,
    VIEWER_MAP_KEYS,
)
from evolution_sim.env.runtime.biotic import diffuse_biotic_field, diffuse_sparse_biotic_field
from evolution_sim.env.runtime.derived import DerivedTileMemo
import evolution_sim.env.runtime.resources as runtime_resources
from evolution_sim.env.runtime.action_contract import (
    ACTION_CONTRACT_VERSION,
    ACTIVE_ACTION_NAMES,
    MATE_ACTION,
    RESERVED_ACTION_NAMES,
    action_contract,
    action_names,
)
from evolution_sim.env.runtime.action_space import (
    ActionMaskContext,
    MovementActionAvailability,
    build_action_mask,
    build_action_mask_from_context,
)
import evolution_sim.env.runtime.feeding as runtime_feeding
import evolution_sim.env.runtime.feeding_opportunity as runtime_feeding_opportunity
import evolution_sim.env.runtime.actions as runtime_actions
import evolution_sim.env.runtime.collectors as runtime_collectors
import evolution_sim.env.runtime.lifecycle_summary as runtime_lifecycle_summary
from evolution_sim.env.runtime.signals import SIGNAL_CONTRACT_VERSION
import evolution_sim.env.runtime.signals as runtime_signals
import evolution_sim.env.runtime.surface_snapshots as runtime_surface_snapshots
import evolution_sim.env.runtime.surfaces as runtime_surfaces
import evolution_sim.env.runtime.mating as runtime_mating
from evolution_sim.env.runtime.state import (
    MIND_INHERITANCE_PLACEHOLDER_VERSION,
    Agent,
    BioticFieldState,
    CarcassDeposit,
    FreshKillDeposit,
    SimulationWorldResult,
    TrophicProfile,
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
import evolution_sim.env.runtime.summary as runtime_summary
from evolution_sim.env.runtime.reproduction import (
    REPRODUCTION_EVENT_SCHEMA_VERSION,
    REPRODUCTIVE_GROUP_CONTRACT_VERSION,
    STAGE0_ASEXUAL,
    reproductive_group_contract,
)
from evolution_sim.env.runtime.trajectory import (
    ACTION_OUTCOME_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryStateContext,
    build_reward,
    capture_agent_state,
    complete_action_outcome,
    reward_contract,
)
from evolution_sim.env.runtime.lifecycle import genome_profile_key
from evolution_sim.env.runtime.mating import (
    PROTO_X_EXPRESSION,
    PROTO_Y_EXPRESSION,
    PROTO_Z_EXPRESSION,
    SEXUAL_EXPRESSION,
    SEXUAL_REPRODUCTION_MODE,
    STAGE1_FACULTATIVE_SEX,
    STAGE2_PROTO_ROLES,
    STAGE3_X_Y_Z,
    X_EXPRESSION,
    Y_EXPRESSION,
    Z_EXPRESSION,
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

class RuntimeContractTestHelpers(unittest.TestCase):
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

    def _role_genome(
        self,
        genome: Genome,
        *,
        role_drive: float,
        expression_bias: float,
        plasticity: float = 0.0,
    ) -> Genome:
        sexualized = self._sexualized_genome(genome)
        return replace(
            sexualized,
            reproductive=replace(
                sexualized.reproductive,
                role_differentiation_drive=role_drive,
                sex_expression_bias=expression_bias,
                sex_plasticity=plasticity,
            ),
        )

    def _test_trophic_profile(self, meat_mode: str) -> TrophicProfile:
        return TrophicProfile(
            plant_share=1.0 if meat_mode == "none" else 0.25,
            animal_share=0.0 if meat_mode == "none" else 0.75,
            scavenger_share=0.5 if meat_mode in {"scavenger", "mixed"} else 0.0,
            hunter_share=0.5 if meat_mode in {"hunter", "mixed"} else 0.0,
            breadth=0.0,
            plant_drive=1.0 if meat_mode == "none" else 0.25,
            animal_drive=0.0 if meat_mode == "none" else 0.75,
            scavenger_drive=0.5 if meat_mode in {"scavenger", "mixed"} else 0.0,
            hunter_drive=0.5 if meat_mode in {"hunter", "mixed"} else 0.0,
            role="herbivore" if meat_mode == "none" else "carnivore",
            meat_mode=meat_mode,
        )

    def _context_only_reproduction_context(
        self,
        *,
        config: WorldConfig | None = None,
        profile: TrophicProfile | None = None,
        next_agent_id: int = 100,
    ) -> runtime_reproduction.ReproductionContext:
        profile = profile or self._test_trophic_profile("none")

        def missing_signal_context() -> runtime_signals.SignalRuntimeContext:
            raise AssertionError("context-only reproduction test did not provide signals")

        placement = runtime_reproduction.ReproductionPlacementContext(
            max_agents=20,
            current_alive_count=lambda: 1,
            has_empty_neighbor=lambda x, y: True,
            find_empty_neighbor=lambda x, y: (x + 1, y),
            sibling_destination_candidates=lambda x, y: [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
            ],
            can_place_at=lambda x, y: True,
            place_agent=lambda child: None,
            invalidate_spatial_state=lambda: None,
        )
        return runtime_reproduction.ReproductionContext(
            config=config or self._ready_reproduction_config(),
            rng=Random(7),
            tick=25,
            next_agent_id=next_agent_id,
            placement=placement,
            trophic_profile=lambda checked_agent: profile,
            trophic_profile_for_genome=lambda checked_genome: profile,
            trophic_role=lambda checked_agent: profile.role,
            meat_mode=lambda checked_agent: profile.meat_mode,
            matched_diet_ratio=lambda checked_agent, checked_profile: 1.0,
            matched_diet_threshold=lambda checked_profile: 0.0,
            health_ratio=lambda checked_agent: 1.0,
            emit=lambda *args, **kwargs: None,
            signal_runtime_context=missing_signal_context,
        )

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
