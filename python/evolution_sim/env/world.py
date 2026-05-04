from __future__ import annotations

from collections import defaultdict, deque
from math import cos, pi, sin
from random import Random

from evolution_sim.config import WorldConfig
from evolution_sim.env.contracts import VIEWER_AGENT_ENCODING
from evolution_sim.env.events import Event, EventType
from evolution_sim.env.fields import EnvironmentFieldMaps, generate_environment_fields
from evolution_sim.env.runtime.actions import DecisionContext, build_decision_context
import evolution_sim.env.runtime.action_space as runtime_action_space
import evolution_sim.env.runtime.actions as runtime_actions
import evolution_sim.env.runtime.feeding as runtime_feeding
import evolution_sim.env.runtime.frames as runtime_frames
import evolution_sim.env.runtime.lifecycle as runtime_lifecycle
import evolution_sim.env.runtime.mating as runtime_mating
import evolution_sim.env.runtime.policy as runtime_policy
import evolution_sim.env.runtime.resources as runtime_resources
import evolution_sim.env.runtime.signals as runtime_signals
import evolution_sim.env.runtime.surface_snapshots as runtime_surface_snapshots
import evolution_sim.env.runtime.surfaces as runtime_surfaces
from evolution_sim.env.runtime.biotic import (
    BioticDiffusionContext,
    BioticStateContext,
    build_biotic_state as build_runtime_biotic_state,
    diffuse_biotic_field as diffuse_runtime_biotic_field,
    invalidate_biotic_state as invalidate_runtime_biotic_state,
)
from evolution_sim.env.runtime.bootstrap import build_static_topology, terrain_neighbor_ratio
from evolution_sim.env.runtime.collectors import CollectorContext, collector_for_mode
from evolution_sim.env.runtime.derived import DerivedTileMemo, reset_derived_caches
from evolution_sim.env.runtime.lifecycle import cached_trophic_profile
import evolution_sim.env.runtime.lifecycle_summary as runtime_lifecycle_summary
import evolution_sim.env.runtime.observations as runtime_observations
import evolution_sim.env.runtime.reproduction as runtime_reproduction
import evolution_sim.env.runtime.summary as runtime_summary
import evolution_sim.env.runtime.ticks as runtime_ticks
from evolution_sim.env.runtime.reporting import (
    build_collapse_events,
    build_replay_analytics,
    build_species_metrics as build_shared_species_metrics,
    empty_carcass_totals,
    empty_combat_totals,
    empty_fresh_kill_totals,
    empty_hydrology_exposure_counts,
    empty_terrain_occupancy,
    SpeciesMetricSample,
)
import evolution_sim.env.runtime.trajectory as runtime_trajectory
from evolution_sim.env.runtime.state import (
    Agent,
    BioticFieldState,
    CarcassDeposit,
    FreshKillDeposit,
    RunMode,
    SimulationWorldResult,
    Tile,
    TrophicProfile,
    empty_mind_inheritance_metadata,
)
from evolution_sim.genome import Genome, SpeciesMember, SpeciesRecord
from evolution_sim.genome.schema import GENE_LIMITS
from evolution_sim.genome.species import (
    GENE_ORDER,
    centroid_from_members,
    euclidean_distance,
    genome_vector,
    vector_from_centroid,
)

LAND_TERRAINS = ("plain", "forest", "wetland", "rocky")
TERRAIN_CODES = {"plain": 0, "forest": 1, "wetland": 2, "rocky": 3, "water": 4}
HABITAT_STATE_CODES = {"stable": 0, "bloom": 1, "flooded": 2, "parched": 3}
ECOLOGY_STATE_CODES = {"stable": 0, "lush": 1, "recovering": 2, "depleted": 3}
HAZARD_TYPE_CODES = {"none": 0, "exposure": 1, "instability": 2}
TROPHIC_ROLE_CODES = {"none": 0, "herbivore": 1, "omnivore": 2, "carnivore": 3}
MEAT_MODE_CODES = {"none": 0, "scavenger": 1, "hunter": 2, "mixed": 3}
ANIMAL_RESOURCE_KINDS = runtime_feeding.ANIMAL_RESOURCE_KINDS
ANIMAL_RESOURCE_POLICY_BLOCKERS = runtime_feeding.ANIMAL_RESOURCE_POLICY_BLOCKERS
NON_LAND_ECOLOGY_CODE = -1
HYDROLOGY_REASON_CODES = {"none": 0, "adjacent_water": 1, "wetland": 2, "flooded": 3}
HYDROLOGY_SUPPORT_FLAGS = {"adjacent_to_water": 1, "wetland": 2, "flooded": 4}
SOFT_REFUGE_CODES = {"none": 0, "canopy_refuge": 1}
TERRAIN_FOOD_BASE = {
    "plain": 0.96,
    "forest": 1.14,
    "wetland": 1.08,
    "rocky": 0.82,
}
TERRAIN_ENERGY_BASE = {
    "plain": 1.0,
    "forest": 0.93,
    "wetland": 1.06,
    "rocky": 1.1,
}
TERRAIN_HYDRATION_BASE = {
    "plain": 1.0,
    "forest": 0.95,
    "wetland": 0.82,
    "rocky": 1.04,
}
TERRAIN_VEGETATION_BASE = {
    "plain": 0.5,
    "forest": 0.72,
    "wetland": 0.68,
    "rocky": 0.3,
}
TERRAIN_RESILIENCE_BASE = {
    "plain": 0.54,
    "forest": 0.7,
    "wetland": 0.78,
    "rocky": 0.6,
}
TERRAIN_SHELTER_BASE = {
    "plain": 0.08,
    "forest": 0.52,
    "wetland": 0.14,
    "rocky": 0.06,
}


class SimulationWorld:
    def __init__(
        self,
        config: WorldConfig,
        *,
        policy: runtime_policy.Policy | None = None,
    ):
        config.validate()
        self.config = config
        self.rng = Random(config.seed)
        self.run_id = f"seed-{config.seed}-ticks-{config.max_ticks}"
        self._has_run = False
        self.environment_fields = self._build_environment_fields()
        self.terrain_map = self._build_terrain_map()
        self.static_topology = build_static_topology(self.terrain_map)
        self.grid = self._build_grid()
        self.agents: dict[int, Agent] = {}
        self.events: list[Event] = []
        self.record_events = True
        self.record_tick_details = True
        self.viewer_frames: list[dict[str, object]] = []
        self.tick = 0
        self.next_agent_id = 1
        self.births = 0
        self.deaths = 0
        self.peak_alive_agents = 0
        self.carrying_capacity_near_cap_ticks = 0
        self.carrying_capacity_at_cap_ticks = 0
        self.carrying_capacity_saturation_births = 0
        self.carrying_capacity_saturation_deaths = 0
        self.last_birth_tick: int | None = None
        self.species_registry: dict[int, dict[str, object]] = {}
        self.reproductive_groups: dict[
            int,
            runtime_reproduction.ReproductiveGroupRecord,
        ] = {}
        self.next_ecotype_id = 1
        self.ecotype_registry: dict[int, dict[str, object]] = {}
        self.current_species_map: dict[int, int] = {}
        self.current_species_records: list[SpeciesRecord] = []
        self.current_ecotype_map: dict[int, int] = {}
        self.current_ecotype_records: list[SpeciesRecord] = []
        self.agent_last_species_map: dict[int, int] = {}
        self.agent_last_ecotype_map: dict[int, int] = {}
        self.tick_birth_pairs: list[tuple[int, int]] = []
        self.tick_death_agent_ids: list[int] = []
        self.tick_death_events: list[dict[str, object]] = []
        self.tick_attack_events: list[dict[str, object]] = []
        self.tick_damage_events: list[dict[str, object]] = []
        self.tick_carcass_deposit_events: list[dict[str, object]] = []
        self.tick_carcass_events: list[dict[str, object]] = []
        self.tick_fresh_kill_events: list[dict[str, object]] = []
        self.tick_fresh_kill_deposit_events: list[dict[str, object]] = []
        self.tick_reproduction_blocked_events: list[dict[str, object]] = []
        self.tick_reproduction_mate_search_events: list[dict[str, object]] = []
        self.tick_fresh_kill_to_carcass_energy = 0.0
        self.tick_carcass_energy_decayed = 0.0
        self.tick_feeding_events: list[dict[str, object]] = []
        self.tick_trajectory_records: list[dict[str, object]] = []
        self.trajectory_records: list[dict[str, object]] = []
        self.record_trajectory = True
        self.retain_trajectory_records = True
        self.trajectory_sink: runtime_trajectory.TrajectorySink | None = None
        self.policy = policy or runtime_policy.ObservationHeuristicPolicy()
        self._policy_action_source = self.policy.policy_id
        self._policy_id = self.policy.policy_id
        self._policy_version = self.policy.policy_version
        self.tick_hazard_exposure_agents: set[int] = set()
        self.run_combat_totals = self._empty_combat_totals()
        self.run_fresh_kill_totals = self._empty_fresh_kill_totals()
        self.run_carcass_totals = self._empty_carcass_totals()
        self.run_diet_totals = self._empty_diet_totals()
        self.run_diet_by_trophic_role = self._empty_grouped_diet_totals(
            [role for role in TROPHIC_ROLE_CODES if role != "none"]
        )
        self.run_diet_by_meat_mode = self._empty_grouped_diet_totals(MEAT_MODE_CODES)
        self.run_animal_resource_opportunity_by_meat_mode = (
            self._empty_grouped_animal_resource_opportunity_counts(MEAT_MODE_CODES)
        )
        self.run_resource_pressure_totals = (
            runtime_resources.empty_resource_pressure_totals()
        )
        self.tick_animal_resource_consumption_by_meat_mode = (
            self._empty_grouped_animal_resource_consumption_counts(MEAT_MODE_CODES)
        )
        self.tick_fresh_kill_deposited_energy = 0.0
        self.tick_carcass_deposited_energy = 0.0
        self.run_death_cause_counts: dict[str, int] = {}
        self.run_death_causes_by_trophic_role = {
            role: {} for role in TROPHIC_ROLE_CODES if role != "none"
        }
        self.run_death_causes_by_meat_mode = {mode: {} for mode in MEAT_MODE_CODES}
        self.run_reproduction_blocked_counts = self._empty_reproduction_blocked_counts()
        self.run_reproduction_blocked_counts_by_trophic_role = {
            role: self._empty_reproduction_blocked_counts()
            for role in TROPHIC_ROLE_CODES
            if role != "none"
        }
        self.run_reproduction_blocked_counts_by_meat_mode = {
            mode: self._empty_reproduction_blocked_counts()
            for mode in MEAT_MODE_CODES
        }
        self.run_reproduction_mate_search_counts = (
            runtime_reproduction.empty_reproduction_mate_search_counts()
        )
        self.biotic_state_revision = 0
        self.cached_biotic_state_revision: int | None = None
        self.cached_biotic_state: BioticFieldState | None = None
        self.reproductive_signal_emissions: list[runtime_signals.SignalEmission] = []
        self.communication_signal_emissions: list[runtime_signals.SignalEmission] = []
        self.tick_signal_emission_events: list[dict[str, object]] = []
        self.tick_signal_totals = runtime_signals.empty_signal_totals()
        self.tick_reproduction_mate_search_counts = (
            runtime_reproduction.empty_reproduction_mate_search_counts()
        )
        self.run_signal_totals = runtime_signals.empty_signal_totals()
        self.signal_state_revision = 0
        self.cached_signal_state_revision: int | None = None
        self.cached_signal_state: runtime_signals.SignalFieldState | None = None
        self.climate_phase = self._build_climate_phase()
        self.cached_climate_tick: int | None = None
        self.cached_climate_state: dict[str, object] | None = None
        self.cached_effective_fields_tick: int | None = None
        self.cached_effective_fields_grid: list[list[tuple[float, float, float]]] | None = None
        self.cached_habitat_tick: int | None = None
        self.cached_habitat_grid: list[list[str]] | None = None
        self.cached_habitat_counts: dict[str, int] | None = None
        self._trophic_profile_cache: dict[tuple[float, ...], TrophicProfile] = {}
        self._biotic_diffusion_target_cache: dict[
            int,
            tuple[tuple[tuple[int, float], ...], ...],
        ] = {}
        self._signal_diffusion_target_cache: dict[
            int,
            tuple[tuple[tuple[int, float], ...], ...],
        ] = {}
        self.runtime_cost_counters = self._empty_runtime_cost_counters()

        self._spawn_initial_agents()
        initial_alive = self.alive_agents()
        self.current_species_map = {
            agent.agent_id: agent.lineage_id for agent in initial_alive
        }
        self.agent_last_species_map = self.current_species_map.copy()
        self.peak_alive_agents = len(self.alive_agents())

    def reset_derived_caches(self, *, include_biotic: bool = True) -> None:
        reset_derived_caches(
            self,
            include_biotic=include_biotic,
            biotic_diffusion_target_cache=self._biotic_diffusion_target_cache,
            signal_diffusion_target_cache=self._signal_diffusion_target_cache,
            invalidate_biotic_state=self._invalidate_biotic_state,
            invalidate_signal_state=self._invalidate_signal_state,
        )

    @staticmethod
    def _empty_runtime_cost_counters() -> dict[str, int]:
        return {
            "observation_builds": 0,
            "action_mask_builds": 0,
            "biotic_state_builds": 0,
            "biotic_state_cache_hits": 0,
            "biotic_state_invalidations": 0,
            "biotic_diffusions": 0,
            "biotic_diffusion_target_cache_hits": 0,
            "biotic_diffusion_target_cache_misses": 0,
            "signal_state_builds": 0,
            "signal_state_cache_hits": 0,
            "signal_state_invalidations": 0,
            "signal_diffusions": 0,
            "signal_diffusion_target_cache_hits": 0,
            "signal_diffusion_target_cache_misses": 0,
            "signal_emissions": 0,
            "resource_pressure_accounting_updates": 0,
        }

    def _record_runtime_cost(self, name: str, amount: int = 1) -> None:
        self.runtime_cost_counters[name] = self.runtime_cost_counters.get(name, 0) + amount

    def run(
        self,
        mode: RunMode = RunMode.FULL_REPLAY,
        *,
        record_trajectory: bool | None = None,
        trajectory_sink: runtime_trajectory.TrajectorySink | None = None,
    ) -> SimulationWorldResult:
        if self._has_run:
            raise RuntimeError(
                "SimulationWorld instances are one-shot; create a new world for each run."
            )
        self._has_run = True
        previous_record_events = self.record_events
        previous_record_tick_details = self.record_tick_details
        previous_record_trajectory = self.record_trajectory
        previous_retain_trajectory_records = self.retain_trajectory_records
        previous_trajectory_sink = self.trajectory_sink
        should_record_trajectory = (
            mode == RunMode.FULL_REPLAY if record_trajectory is None else record_trajectory
        )
        if trajectory_sink is not None:
            should_record_trajectory = True
        self.record_events = mode == RunMode.FULL_REPLAY
        self.record_tick_details = mode == RunMode.FULL_REPLAY or should_record_trajectory
        self.record_trajectory = should_record_trajectory
        self.retain_trajectory_records = (
            mode == RunMode.FULL_REPLAY
            or (should_record_trajectory and trajectory_sink is None)
        )
        self.trajectory_sink = trajectory_sink
        collector = collector_for_mode(mode)
        collector_context = self._collector_context()
        sink_finished = False
        try:
            if trajectory_sink is not None:
                trajectory_sink.begin(
                    run_id=self.run_id,
                    config=self.config.to_dict(),
                    contract=runtime_trajectory.trajectory_contract(
                        self.config.signals
                    ),
                )
            self._emit(EventType.RUN_STARTED, data={"run_id": self.run_id})
            for tick in range(self.config.max_ticks):
                self.tick = tick
                births_this_tick, deaths_this_tick = self._run_tick()
                collector.on_tick(
                    collector_context,
                    births_this_tick=births_this_tick,
                    deaths_this_tick=deaths_this_tick,
                )
                if not self.alive_agents():
                    break
            self._emit(
                EventType.RUN_COMPLETED,
                data={
                    "ticks_executed": self.tick + 1,
                    "alive_agents": len(self.alive_agents()),
                    "births": self.births,
                    "deaths": self.deaths,
                },
            )
            summary, event_payloads, viewer = collector.finalize(collector_context)
            if trajectory_sink is not None:
                trajectory_sink.finish(summary=summary)
                sink_finished = True
            return SimulationWorldResult(
                run_id=self.run_id,
                config=self.config.to_dict(),
                summary=summary,
                events=event_payloads,
                viewer=viewer,
                mode=mode,
            )
        except Exception:
            if trajectory_sink is not None and not sink_finished:
                trajectory_sink.abort()
            raise
        finally:
            self.record_events = previous_record_events
            self.record_tick_details = previous_record_tick_details
            self.record_trajectory = previous_record_trajectory
            self.retain_trajectory_records = previous_retain_trajectory_records
            self.trajectory_sink = previous_trajectory_sink

    def _run_tick(self) -> tuple[int, int]:
        return runtime_ticks.run_tick(
            self,
            meat_mode_codes=MEAT_MODE_CODES,
            tick_context=self._tick_phase_context(),
        )

    def _collector_context(self) -> CollectorContext:
        return CollectorContext(
            config=self.config,
            events=self.events,
            capture_frame=self._capture_frame,
            build_summary=self._build_summary,
            build_viewer_payload=self._build_viewer_payload,
            refresh_population_snapshots=self._refresh_population_snapshots,
        )

    def _tick_phase_context(self) -> runtime_ticks.TickPhaseContext:
        return runtime_ticks.TickPhaseContext(
            invalidate_biotic_state=self._invalidate_biotic_state,
            decay_signal_emissions=self._decay_signal_emissions,
            climate_state=self._climate_state,
            season_state=self._season_state,
            emit=self._emit,
            regrow_resources=self._regrow_resources,
            population_trophic_counts=self._population_trophic_counts,
            observe_agent=self._observe_agent,
            animal_resource_reachability_by_meat_mode=(
                self._animal_resource_reachability_by_meat_mode
            ),
            animal_resource_presence_this_tick=self._animal_resource_presence_this_tick,
            decay_recent_diet=self._decay_recent_diet,
            choose_action=self._choose_action,
            action_mask=self._action_mask,
            action_resolution_context=self._action_resolution_context,
            lifecycle_context=self._lifecycle_context(),
            kill_agent=self._kill_agent,
            finalize_trajectory_decisions=self._finalize_trajectory_decisions,
            record_animal_resource_opportunity_tick=(
                self._record_animal_resource_opportunity_tick
            ),
            begin_trajectory_decision=self._begin_trajectory_decision,
            policy_metadata=lambda: {
                "action_source": self._policy_action_source,
                "policy_id": self._policy_id,
                "policy_version": self._policy_version,
            },
        )

    def _build_grid(self) -> list[list[Tile]]:
        grid: list[list[Tile]] = []
        for y in range(self.config.height):
            row: list[Tile] = []
            for x in range(self.config.width):
                terrain = self.terrain_map[y][x]
                fertility = self._base_tile_fertility(x, y, terrain)
                moisture = self._base_tile_moisture(x, y, terrain)
                heat = self._base_tile_heat(x, y, terrain)
                vegetation = self._base_tile_vegetation(terrain, fertility, moisture, heat)
                shelter = self._base_tile_shelter(x, y, terrain, fertility, moisture, heat, vegetation)
                recovery_debt = self._base_tile_recovery_debt(
                    terrain,
                    fertility,
                    moisture,
                    heat,
                    vegetation,
                )
                food = 0.0
                water = 1.0 if terrain == "water" else 0.0
                if terrain == "plain":
                    food = self.rng.uniform(0.08, 0.32 + fertility * 0.38)
                elif terrain == "forest":
                    food = self.rng.uniform(0.18, 0.42 + fertility * 0.42)
                elif terrain == "wetland":
                    food = self.rng.uniform(0.16, 0.38 + fertility * 0.34)
                elif terrain == "rocky":
                    food = self.rng.uniform(0.05, 0.18 + fertility * 0.22)
                if terrain != "water":
                    food = self._clamp01(
                        food * (0.72 + vegetation * 0.44) * (1.0 - recovery_debt * 0.18)
                    )
                row.append(
                    Tile(
                        terrain=terrain,
                        food=food,
                        water=water,
                        fertility=fertility,
                        moisture=moisture,
                        heat=heat,
                        vegetation=vegetation,
                        shelter=shelter,
                        recovery_debt=recovery_debt,
                    )
                )
            grid.append(row)
        return grid

    def _build_environment_fields(self) -> EnvironmentFieldMaps:
        environment = self.config.environment
        return generate_environment_fields(
            width=self.config.width,
            height=self.config.height,
            seed=self.config.seed,
            coarse_width=environment.control_points_x,
            coarse_height=environment.control_points_y,
        )

    def _build_climate_phase(self) -> dict[str, float]:
        rng = Random(self.config.seed * 4_237 + 19)
        return {
            "moisture_front": rng.random(),
            "heat_front": rng.random(),
            "disturbance_x": rng.random(),
            "disturbance_y": rng.random(),
            "disturbance_strength": rng.random(),
        }

    def _build_terrain_map(self) -> list[list[str]]:
        terrain_map = [
            ["plain" for _ in range(self.config.width)] for _ in range(self.config.height)
        ]
        total_tiles = self.config.width * self.config.height
        target_counts = {
            "water": int(total_tiles * self.config.water_tile_ratio),
            "wetland": int(total_tiles * self.config.wetland_tile_ratio),
            "forest": int(total_tiles * self.config.forest_tile_ratio),
            "rocky": int(total_tiles * self.config.rocky_tile_ratio),
        }

        self._assign_top_scoring_tiles(
            terrain_map,
            "water",
            target_counts["water"],
            self._water_terrain_score,
        )
        water_distance_map = self._terrain_water_distance_map(terrain_map)
        reserved_shoreline = self._reserved_shoreline_tiles(
            terrain_map,
            water_distance_map,
            target_counts["wetland"],
        )
        self._assign_wetland_tiles(
            terrain_map,
            target_counts["wetland"],
            water_distance_map,
            reserved_shoreline,
        )
        self._assign_top_scoring_tiles(
            terrain_map,
            "forest",
            target_counts["forest"],
            self._forest_terrain_score,
        )
        self._assign_top_scoring_tiles(
            terrain_map,
            "rocky",
            target_counts["rocky"],
            self._rocky_terrain_score,
        )
        return terrain_map

    def _assign_top_scoring_tiles(
        self,
        terrain_map: list[list[str]],
        terrain: str,
        target_count: int,
        score_fn,
    ) -> None:
        candidates: list[tuple[float, int, int]] = []
        for y in range(self.config.height):
            for x in range(self.config.width):
                if terrain_map[y][x] != "plain":
                    continue
                candidates.append((score_fn(x, y, terrain_map), x, y))

        candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
        for _, x, y in candidates[:target_count]:
            terrain_map[y][x] = terrain

    def _assign_wetland_tiles(
        self,
        terrain_map: list[list[str]],
        target_count: int,
        water_distance_map: list[list[int | None]],
        reserved_shoreline: set[tuple[int, int]],
    ) -> None:
        candidates: list[tuple[float, int, int]] = []
        for y in range(self.config.height):
            for x in range(self.config.width):
                if terrain_map[y][x] != "plain":
                    continue
                if (x, y) in reserved_shoreline:
                    continue
                candidates.append(
                    (
                        self._wetland_terrain_score(x, y, terrain_map, water_distance_map),
                        x,
                        y,
                    )
                )

        candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
        for _, x, y in candidates[:target_count]:
            terrain_map[y][x] = "wetland"

    def _water_terrain_score(self, x: int, y: int, terrain_map: list[list[str]]) -> float:
        moisture = self.environment_fields.moisture[y][x]
        fertility = self.environment_fields.fertility[y][x]
        heat = self.environment_fields.heat[y][x]
        edge_distance = min(x, y, self.config.width - 1 - x, self.config.height - 1 - y)
        interior_bias = min(edge_distance, 3) / 3
        return moisture * 0.7 + (1.0 - fertility) * 0.16 + (1.0 - heat) * 0.06 + interior_bias * 0.08

    def _wetland_terrain_score(
        self,
        x: int,
        y: int,
        terrain_map: list[list[str]],
        water_distance_map: list[list[int | None]] | None = None,
    ) -> float:
        moisture = self.environment_fields.moisture[y][x]
        fertility = self.environment_fields.fertility[y][x]
        heat = self.environment_fields.heat[y][x]
        water_proximity = self._terrain_water_proximity(x, y, terrain_map)
        water_distance = (
            water_distance_map[y][x] if water_distance_map is not None else None
        )
        shoreline_bias = 0.0
        if water_distance == 1:
            shoreline_bias -= 0.18
        elif water_distance == 2:
            shoreline_bias += 0.08
        return (
            moisture * 0.4
            + fertility * 0.26
            + water_proximity * 0.34
            - heat * 0.12
            + shoreline_bias
        )

    def _forest_terrain_score(self, x: int, y: int, terrain_map: list[list[str]]) -> float:
        moisture = self.environment_fields.moisture[y][x]
        fertility = self.environment_fields.fertility[y][x]
        heat = self.environment_fields.heat[y][x]
        water_proximity = self._terrain_water_proximity(x, y, terrain_map)
        return fertility * 0.5 + moisture * 0.21 + (1.0 - heat) * 0.21 + water_proximity * 0.08

    def _rocky_terrain_score(self, x: int, y: int, terrain_map: list[list[str]]) -> float:
        moisture = self.environment_fields.moisture[y][x]
        fertility = self.environment_fields.fertility[y][x]
        heat = self.environment_fields.heat[y][x]
        water_proximity = self._terrain_water_proximity(x, y, terrain_map)
        return heat * 0.42 + (1.0 - moisture) * 0.26 + (1.0 - fertility) * 0.36 - water_proximity * 0.08

    def _terrain_water_proximity(
        self,
        x: int,
        y: int,
        terrain_map: list[list[str]],
    ) -> float:
        best = 0.0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                distance = abs(dx) + abs(dy)
                if distance == 0 or distance > 2:
                    continue
                nx = x + dx
                ny = y + dy
                if not self._in_bounds(nx, ny):
                    continue
                if terrain_map[ny][nx] == "water":
                    best = max(best, 1.0 if distance == 1 else 0.55)
        return best

    def _terrain_neighbor_ratio(
        self,
        x: int,
        y: int,
        terrain_filter: set[str],
        radius: int = 1,
    ) -> float:
        return terrain_neighbor_ratio(
            terrain_map=self.terrain_map,
            topology=self.static_topology,
            x=x,
            y=y,
            terrain_filter=terrain_filter,
            radius=radius,
        )

    def _is_shoreline_tile(self, x: int, y: int) -> bool:
        return self.grid[y][x].terrain != "water" and self._adjacent_to_water(x, y)

    def _terrain_water_distance_map(
        self,
        terrain_map: list[list[str]],
    ) -> list[list[int | None]]:
        distances: list[list[int | None]] = [
            [None for _ in range(self.config.width)] for _ in range(self.config.height)
        ]
        frontier: deque[tuple[int, int]] = deque()
        for y in range(self.config.height):
            for x in range(self.config.width):
                if terrain_map[y][x] == "water":
                    distances[y][x] = 0
                    frontier.append((x, y))

        while frontier:
            x, y = frontier.popleft()
            current_distance = distances[y][x]
            if current_distance is None:
                continue
            for _, dx, dy in self._movement_actions():
                nx = x + dx
                ny = y + dy
                if not self._in_bounds(nx, ny) or distances[ny][nx] is not None:
                    continue
                distances[ny][nx] = current_distance + 1
                frontier.append((nx, ny))
        return distances

    def _reserved_shoreline_tiles(
        self,
        terrain_map: list[list[str]],
        water_distance_map: list[list[int | None]],
        wetland_target_count: int,
    ) -> set[tuple[int, int]]:
        shoreline_tiles: list[tuple[float, int, int]] = []
        for y in range(self.config.height):
            for x in range(self.config.width):
                if terrain_map[y][x] != "plain" or water_distance_map[y][x] != 1:
                    continue
                shoreline_tiles.append((self._shoreline_reservation_score(x, y), x, y))

        shoreline_tiles.sort(key=lambda item: (-item[0], item[2], item[1]))
        reserve_target = min(
            len(shoreline_tiles),
            max(
                18,
                min(
                    int(len(shoreline_tiles) * self.config.environment.shoreline_reserve_ratio),
                    max(0, len(shoreline_tiles) - max(int(wetland_target_count * 0.45), 12)),
                ),
            ),
        )
        reserved = {
            (x, y)
            for _, x, y in shoreline_tiles[:reserve_target]
        }
        return reserved

    def _shoreline_reservation_score(self, x: int, y: int) -> float:
        moisture = self.environment_fields.moisture[y][x]
        fertility = self.environment_fields.fertility[y][x]
        heat = self.environment_fields.heat[y][x]
        return fertility * 0.42 + moisture * 0.12 + (1.0 - heat) * 0.24

    def _base_tile_fertility(self, x: int, y: int, terrain: str) -> float:
        base = self.environment_fields.fertility[y][x]
        if terrain == "forest":
            base += 0.12
        elif terrain == "wetland":
            base += 0.18
        elif terrain == "rocky":
            base -= 0.1
        elif terrain == "water":
            base -= 0.08
        return self._clamp01(base)

    def _base_tile_moisture(self, x: int, y: int, terrain: str) -> float:
        base = self.environment_fields.moisture[y][x]
        if terrain == "forest":
            base += 0.1
        elif terrain == "wetland":
            base += 0.22
        elif terrain == "rocky":
            base -= 0.06
        elif terrain == "water":
            base += 0.35
        return self._clamp01(base)

    def _base_tile_heat(self, x: int, y: int, terrain: str) -> float:
        base = self.environment_fields.heat[y][x]
        if terrain == "forest":
            base -= 0.08
        elif terrain == "wetland":
            base -= 0.06
        elif terrain == "plain":
            base += 0.04
        elif terrain == "rocky":
            base += 0.08
        elif terrain == "water":
            base -= 0.18
        return self._clamp01(base)

    def _base_tile_vegetation(
        self,
        terrain: str,
        fertility: float,
        moisture: float,
        heat: float,
    ) -> float:
        if terrain == "water":
            return 1.0
        return self._clamp01(
            TERRAIN_VEGETATION_BASE.get(terrain, 0.5)
            + fertility * 0.24
            + moisture * 0.18
            - heat * 0.14
            + self.rng.uniform(-0.08, 0.08)
        )

    def _base_tile_recovery_debt(
        self,
        terrain: str,
        fertility: float,
        moisture: float,
        heat: float,
        vegetation: float,
    ) -> float:
        if terrain == "water":
            return 0.0
        base = (1.0 - TERRAIN_RESILIENCE_BASE.get(terrain, 0.56)) * 0.2
        base += max(0.0, heat - moisture) * 0.14
        base += max(0.0, 0.48 - vegetation) * 0.18
        base -= fertility * 0.06
        base += self.rng.uniform(0.0, 0.04)
        return self._clamp01(base)

    def _base_tile_shelter(
        self,
        x: int,
        y: int,
        terrain: str,
        fertility: float,
        moisture: float,
        heat: float,
        vegetation: float,
    ) -> float:
        if terrain == "water":
            return 0.0
        forest_density = self._terrain_neighbor_ratio(x, y, terrain_filter={"forest"})
        base = TERRAIN_SHELTER_BASE.get(terrain, 0.08)
        base += vegetation * (0.12 if terrain == "forest" else 0.08)
        base += forest_density * (0.16 if terrain == "forest" else 0.04)
        base += fertility * 0.04 + moisture * 0.05 - heat * 0.08
        base += self.rng.uniform(-0.06, 0.06)
        if terrain != "forest":
            base *= 0.28
        return self._clamp01(base)

    def _spawn_initial_agents(self) -> None:
        for _ in range(self.config.initial_agents):
            x, y = self._random_initial_spawn_tile()
            genome = Genome.sample_initial(self.rng)
            reproductive_state = runtime_reproduction.founder_reproductive_state(
                self.next_agent_id
            )
            agent = Agent(
                agent_id=self.next_agent_id,
                parent_id=None,
                lineage_id=self.next_agent_id,
                birth_tick=0,
                death_tick=None,
                x=x,
                y=y,
                energy=genome.max_energy * 0.8,
                hydration=genome.max_hydration * 0.8,
                health=genome.max_health * 0.92,
                max_health=genome.max_health,
                injury_load=0.0,
                age=0,
                alive=True,
                last_reproduction_tick=-10_000,
                last_damage_source="none",
                recent_plant_energy=0.0,
                recent_fresh_kill_energy=0.0,
                recent_carcass_energy=0.0,
                genome_vector=genome_vector(genome),
                genome=genome,
                reproductive_group_id=reproductive_state.group_id,
                reproductive_stage=reproductive_state.stage,
                reproductive_expression=reproductive_state.expression,
                mind_inheritance_metadata=empty_mind_inheritance_metadata(),
            )
            self._place_agent(agent)
            runtime_reproduction.register_founder_group(
                self.reproductive_groups,
                agent,
                tick=0,
            )
            self.next_agent_id += 1

    def _random_initial_spawn_tile(self) -> tuple[int, int]:
        viable_tiles: list[tuple[int, int]] = []
        fallback_tiles: list[tuple[int, int]] = []
        for y in range(self.config.height):
            for x in range(self.config.width):
                tile = self.grid[y][x]
                if tile.terrain == "water" or tile.occupant_id is not None:
                    continue
                fallback_tiles.append((x, y))
                if self._spawn_tile_viability_score(x, y) >= 0.6:
                    viable_tiles.append((x, y))

        pool = viable_tiles if viable_tiles else fallback_tiles
        if not pool:
            raise RuntimeError("No spawnable land tiles were available.")
        return pool[self.rng.randrange(len(pool))]

    def _spawn_tile_viability_score(self, x: int, y: int) -> float:
        tile = self.grid[y][x]
        water_reason = self._water_access_reason(x, y)
        hazard_type, hazard_level = self._hazard_at(x, y)
        water_bonus = (
            0.36
            if water_reason == "adjacent_water"
            else 0.34
            if water_reason == "wetland"
            else 0.18
            if water_reason == "flooded"
            else 0.0
        )
        refuge_bonus = self._refuge_score(x, y) * 0.08
        terrain_bonus = 0.18 if tile.terrain in {"forest", "wetland"} else 0.0
        return (
            tile.food * 0.48
            + tile.vegetation * 0.22
            + tile.moisture * 0.22
            + tile.fertility * 0.16
            + water_bonus
            + refuge_bonus
            + terrain_bonus
            + (1.0 - tile.recovery_debt) * 0.18
            - tile.heat * 0.12
            - hazard_level * (0.14 if hazard_type == "exposure" else 0.1)
        )

    def _random_empty_land_tile(self) -> tuple[int, int]:
        while True:
            x = self.rng.randrange(self.config.width)
            y = self.rng.randrange(self.config.height)
            tile = self.grid[y][x]
            if tile.terrain != "water" and tile.occupant_id is None:
                return x, y

    def _place_agent(self, agent: Agent) -> None:
        self.agents[agent.agent_id] = agent
        self.grid[agent.y][agent.x].occupant_id = agent.agent_id

    @staticmethod
    def _empty_combat_totals() -> dict[str, float]:
        return {
            "attack_attempts": 0,
            "successful_attacks": 0,
            "kills": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "attack_damage_taken": 0.0,
            "hazard_damage_taken": 0.0,
        }

    @staticmethod
    def _empty_carcass_totals() -> dict[str, float]:
        return empty_carcass_totals()

    @staticmethod
    def _empty_fresh_kill_totals() -> dict[str, float]:
        return empty_fresh_kill_totals()

    @staticmethod
    def _empty_diet_totals() -> dict[str, float]:
        return runtime_feeding.empty_diet_totals()

    @staticmethod
    def _empty_grouped_diet_totals(groups: list[str] | dict[str, int]) -> dict[str, dict[str, float]]:
        return runtime_feeding.empty_grouped_diet_totals(groups)

    @staticmethod
    def _empty_animal_resource_consumption_counts() -> dict[str, int | float]:
        return runtime_feeding.empty_animal_resource_consumption_counts()

    @staticmethod
    def _empty_grouped_animal_resource_consumption_counts(
        groups: list[str] | dict[str, int],
    ) -> dict[str, dict[str, int | float]]:
        return runtime_feeding.empty_grouped_animal_resource_consumption_counts(groups)

    @staticmethod
    def _empty_animal_resource_opportunity_counts() -> dict[str, int | float]:
        return runtime_feeding.empty_animal_resource_opportunity_counts()

    @staticmethod
    def _empty_grouped_animal_resource_opportunity_counts(
        groups: list[str] | dict[str, int],
    ) -> dict[str, dict[str, int | float]]:
        return runtime_feeding.empty_grouped_animal_resource_opportunity_counts(groups)

    @staticmethod
    def _empty_reproduction_blocked_counts() -> dict[str, int]:
        return runtime_reproduction.empty_reproduction_blocked_counts()

    @staticmethod
    def _empty_reproduction_mate_search_counts() -> dict[str, int]:
        return runtime_reproduction.empty_reproduction_mate_search_counts()

    @staticmethod
    def _empty_reproduction_readiness_counts() -> dict[str, int]:
        return runtime_reproduction.empty_reproduction_readiness_counts()

    @staticmethod
    def _empty_reproduction_biological_blocker_counts() -> dict[str, int]:
        return runtime_reproduction.empty_reproduction_biological_blocker_counts()

    @staticmethod
    def _empty_reproduction_energy_readiness_counts() -> dict[str, int | float]:
        return runtime_reproduction.empty_reproduction_energy_readiness_counts()

    @staticmethod
    def _finalize_reproduction_energy_readiness_counts(
        counts: dict[str, int | float],
    ) -> dict[str, int | float]:
        return runtime_reproduction.finalize_reproduction_energy_readiness_counts(
            counts
        )

    @staticmethod
    def _accumulate_diet_totals(
        totals: dict[str, float],
        food_source: str,
        gained_energy: float,
    ) -> None:
        runtime_feeding.accumulate_diet_totals(totals, food_source, gained_energy)

    def _emit(
        self,
        event_type: EventType,
        agent_id: int | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        if not self.record_events:
            return
        self.events.append(
            Event(tick=self.tick, type=event_type, agent_id=agent_id, data=data or {})
        )

    def alive_agents(self) -> list[Agent]:
        return [agent for agent in self.agents.values() if agent.alive]

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _species_id_for_agent(self, agent_id: int | None) -> int | None:
        if agent_id is None:
            return None
        return self.current_species_map.get(agent_id, self.agent_last_species_map.get(agent_id))

    @staticmethod
    def _prune_fresh_kill_deposits(tile: Tile) -> None:
        runtime_resources.prune_fresh_kill_deposits(tile)

    @staticmethod
    def _prune_carcass_deposits(tile: Tile) -> None:
        runtime_resources.prune_carcass_deposits(tile)

    def _invalidate_biotic_state(self) -> None:
        self._record_runtime_cost("biotic_state_invalidations")
        invalidate_runtime_biotic_state(self)

    def _invalidate_signal_state(self) -> None:
        self._record_runtime_cost("signal_state_invalidations")
        self.signal_state_revision += 1
        self.cached_signal_state_revision = None
        self.cached_signal_state = None

    def _decay_signal_emissions(self) -> None:
        runtime_signals.decay_signal_emissions(
            context=self._signal_runtime_context(),
        )

    @staticmethod
    def _merge_fresh_kill_deposit_group(deposits: list[FreshKillDeposit]) -> FreshKillDeposit:
        return runtime_resources.merge_fresh_kill_deposit_group(deposits)

    def _carcass_freshness_bucket(self, freshness: float) -> int:
        return runtime_resources.carcass_freshness_bucket(
            freshness,
            freshness_merge_bucket=self.config.carcasses.freshness_merge_bucket,
        )

    @staticmethod
    def _merge_carcass_deposit_group(deposits: list[CarcassDeposit]) -> CarcassDeposit:
        return runtime_resources.merge_carcass_deposit_group(deposits)

    def _compact_carcass_deposits(self, tile: Tile) -> None:
        runtime_resources.compact_carcass_deposits(
            tile,
            max_tile_deposits=self.config.carcasses.max_tile_deposits,
            freshness_merge_bucket=self.config.carcasses.freshness_merge_bucket,
        )

    def _compact_fresh_kill_deposits(self, tile: Tile) -> None:
        runtime_resources.compact_fresh_kill_deposits(
            tile,
            max_tile_deposits=self.config.carcasses.max_tile_deposits,
        )

    def _carcass_source_breakdown(
        self,
        deposits: list[CarcassDeposit],
    ) -> list[dict[str, object]]:
        return runtime_resources.carcass_source_breakdown(deposits)

    def _fresh_kill_source_breakdown(
        self,
        deposits: list[FreshKillDeposit],
    ) -> list[dict[str, object]]:
        return runtime_resources.fresh_kill_source_breakdown(deposits)

    @staticmethod
    def _resolved_source_species(source_breakdown: list[dict[str, object]]) -> int | None:
        return runtime_resources.resolved_source_species(source_breakdown)

    def _carcass_tile_state(self, tile: Tile) -> dict[str, object]:
        return runtime_resources.carcass_tile_state(self, tile)

    def _fresh_kill_tile_state(self, tile: Tile) -> dict[str, object]:
        return runtime_resources.fresh_kill_tile_state(self, tile)

    def _carcass_tile_summary_for_position(self, x: int, y: int) -> dict[str, object]:
        return runtime_resources.carcass_tile_summary_for_position(self, x, y)

    def _fresh_kill_tile_summary_for_position(self, x: int, y: int) -> dict[str, object]:
        return runtime_resources.fresh_kill_tile_summary_for_position(self, x, y)

    def _carcass_patch_summaries(self) -> list[dict[str, object]]:
        return runtime_resources.carcass_patch_summaries(self)

    def _fresh_kill_patch_summaries(self) -> list[dict[str, object]]:
        return runtime_resources.fresh_kill_patch_summaries(self)

    def _resource_context(self) -> runtime_resources.ResourceContext:
        return runtime_resources.ResourceContext(
            effective_tile_fields=self._effective_tile_fields,
            habitat_state_at=self._habitat_state_at,
            habitat_state_grid=self._habitat_state_grid,
            terrain_neighbor_ratio=self._terrain_neighbor_ratio,
            season_state=self._season_state,
            clamp01=self._clamp01,
            emit=self._emit,
        )

    def _deposit_fresh_kill(
        self,
        tile: Tile,
        *,
        x: int,
        y: int,
        energy: float,
        source_species: int | None,
        source_agent_id: int | None,
        killer_id: int | None,
    ) -> dict[str, object]:
        return runtime_resources.deposit_fresh_kill(
            self,
            tile,
            x=x,
            y=y,
            energy=energy,
            source_species=source_species,
            source_agent_id=source_agent_id,
            killer_id=killer_id,
        )

    def _convert_fresh_kill_to_carcass(
        self,
        tile: Tile,
        *,
        x: int,
        y: int,
        conversion_rate: float,
    ) -> float:
        return runtime_resources.convert_fresh_kill_to_carcass(
            self,
            tile,
            x=x,
            y=y,
            conversion_rate=conversion_rate,
        )

    def _deposit_carcass(
        self,
        tile: Tile,
        *,
        x: int,
        y: int,
        energy: float,
        source_species: int | None,
        source_agent_id: int | None,
        cause: str,
        killer_id: int | None,
    ) -> dict[str, object]:
        return runtime_resources.deposit_carcass(
            self,
            tile,
            resource_context=self._resource_context(),
            x=x,
            y=y,
            energy=energy,
            source_species=source_species,
            source_agent_id=source_agent_id,
            cause=cause,
            killer_id=killer_id,
        )

    def _decay_carcass_tile(self, tile: Tile, *, decay: float) -> float:
        return runtime_resources.decay_carcass_tile(
            tile,
            decay=decay,
            max_tile_deposits=self.config.carcasses.max_tile_deposits,
            freshness_merge_bucket=self.config.carcasses.freshness_merge_bucket,
        )

    def _consume_carcass_from_tile(self, tile: Tile, requested_amount: float) -> dict[str, object]:
        return runtime_resources.consume_carcass_from_tile(
            tile,
            requested_amount,
            max_tile_deposits=self.config.carcasses.max_tile_deposits,
            freshness_merge_bucket=self.config.carcasses.freshness_merge_bucket,
        )

    def _consume_fresh_kill_from_tile(
        self,
        tile: Tile,
        requested_amount: float,
    ) -> dict[str, object]:
        return runtime_resources.consume_fresh_kill_from_tile(
            tile,
            requested_amount,
            max_tile_deposits=self.config.carcasses.max_tile_deposits,
        )

    def _season_state(self) -> dict[str, object]:
        season_index = (self.tick // self.config.climate.season_length) % 2
        season_length = max(self.config.climate.season_length, 1)
        season_tick = self.tick % season_length
        return {
            "index": season_index,
            "name": "wet" if season_index == 0 else "dry",
            "progress": season_tick / season_length,
        }

    def _climate_state(self) -> dict[str, object]:
        if self.cached_climate_tick == self.tick and self.cached_climate_state is not None:
            return self.cached_climate_state

        season = self._season_state()
        environment = self.config.environment
        drift_period = max(environment.drift_period_ticks, 1)
        disturbance_period = max(environment.disturbance_period_ticks, 1)

        moisture_front_x = ((self.tick / drift_period) + self.climate_phase["moisture_front"]) % 1.0
        heat_front_y = (
            (self.tick / (drift_period * 1.18)) + self.climate_phase["heat_front"]
        ) % 1.0

        disturbance_progress = self.tick / disturbance_period
        disturbance_center_x = 0.5 + 0.34 * sin(
            2 * pi * (disturbance_progress + self.climate_phase["disturbance_x"])
        )
        disturbance_center_y = 0.5 + 0.28 * cos(
            2 * pi * (disturbance_progress * 0.83 + self.climate_phase["disturbance_y"])
        )
        disturbance_strength = environment.disturbance_strength * (
            0.78
            + 0.22
            * (
                0.5
                + 0.5
                * sin(
                    2
                    * pi
                    * (disturbance_progress * 0.61 + self.climate_phase["disturbance_strength"])
                )
            )
        )

        climate_state = {
            "season": season["name"],
            "season_progress": round(season["progress"], 4),
            "moisture_shift": round(
                environment.moisture_season_swing
                if season["name"] == "wet"
                else -environment.moisture_season_swing,
                4,
            ),
            "heat_shift": round(
                -environment.heat_season_swing
                if season["name"] == "wet"
                else environment.heat_season_swing,
                4,
            ),
            "moisture_front_x": round(moisture_front_x, 4),
            "heat_front_y": round(heat_front_y, 4),
            "disturbance_type": "storm" if season["name"] == "wet" else "drought",
            "disturbance_center_x": round(self._clamp01(disturbance_center_x), 4),
            "disturbance_center_y": round(self._clamp01(disturbance_center_y), 4),
            "disturbance_strength": round(disturbance_strength, 4),
        }
        self.cached_climate_tick = self.tick
        self.cached_climate_state = climate_state
        return climate_state

    def _habitat_state_grid(self) -> tuple[list[list[str]], dict[str, int]]:
        if self.cached_habitat_tick == self.tick and self.cached_habitat_grid is not None and self.cached_habitat_counts is not None:
            return self.cached_habitat_grid, self.cached_habitat_counts

        climate_state = self._climate_state()
        grid: list[list[str]] = []
        counts = {state: 0 for state in HABITAT_STATE_CODES}
        for y, row in enumerate(self.grid):
            state_row: list[str] = []
            for x, tile in enumerate(row):
                if tile.terrain == "water":
                    state = "stable"
                else:
                    fertility, moisture, heat = self._effective_tile_fields(x, y)
                    if (
                        climate_state["disturbance_type"] == "storm"
                        and tile.terrain in {"plain", "wetland"}
                        and moisture >= (0.76 if tile.terrain == "wetland" else 0.84)
                    ):
                        state = "flooded"
                    elif (
                        tile.terrain != "wetland"
                        and moisture <= 0.24
                        and heat >= 0.62
                    ):
                        state = "parched"
                    elif fertility >= 0.72 and 0.42 <= moisture <= 0.86 and heat <= 0.58:
                        state = "bloom"
                    else:
                        state = "stable"
                    counts[state] += 1
                state_row.append(state)
            grid.append(state_row)

        self.cached_habitat_tick = self.tick
        self.cached_habitat_grid = grid
        self.cached_habitat_counts = counts
        return grid, counts

    def _habitat_state_at(self, x: int, y: int) -> str:
        return self._habitat_state_grid()[0][y][x]

    def _ecology_state_for_tile(self, tile: Tile) -> str:
        if tile.terrain == "water":
            raise ValueError("Water tiles do not belong to land ecology state accounting.")
        if tile.vegetation <= 0.26 and tile.recovery_debt >= 0.58:
            return "depleted"
        if tile.vegetation >= 0.72 and tile.recovery_debt <= 0.24:
            return "lush"
        if tile.recovery_debt >= 0.32 or tile.vegetation <= 0.44:
            return "recovering"
        return "stable"

    def _ecology_state_at(self, x: int, y: int) -> str:
        tile = self.grid[y][x]
        return "stable" if tile.terrain == "water" else self._ecology_state_for_tile(tile)

    def _refuge_score(self, x: int, y: int) -> float:
        tile = self.grid[y][x]
        if tile.terrain != "forest":
            return 0.0
        habitat_state = self._habitat_state_at(x, y)
        if habitat_state in {"parched", "flooded"}:
            return 0.0
        ecology_state = self._ecology_state_at(x, y)
        forest_density = self._terrain_neighbor_ratio(x, y, terrain_filter={"forest"}, radius=1)
        _, moisture, heat = self._effective_tile_fields(x, y)
        if (
            tile.shelter < 0.56
            or tile.vegetation < 0.62
            or forest_density < 0.48
            or tile.recovery_debt > 0.28
        ):
            return 0.0
        score = (
            tile.shelter * 0.32
            + tile.vegetation * 0.15
            + forest_density * 0.24
            + max(0.0, moisture - 0.38) * 0.16
            + max(0.0, 0.6 - heat) * 0.12
            - tile.recovery_debt * 0.22
        )
        if ecology_state == "lush":
            score += 0.06
        elif ecology_state == "recovering":
            score -= 0.06
        elif ecology_state == "depleted":
            score -= 0.18
        if habitat_state == "bloom":
            score += 0.05
        return self._clamp01(score)

    def _soft_refuge_reason(self, x: int, y: int) -> str:
        return "canopy_refuge" if self._refuge_score(x, y) >= 0.82 else "none"

    def _is_flooded_tile(self, x: int, y: int) -> bool:
        return self.grid[y][x].terrain != "water" and self._habitat_state_at(x, y) == "flooded"

    def _hydrology_support_code(self, x: int, y: int) -> int:
        tile = self.grid[y][x]
        if tile.terrain == "water":
            return NON_LAND_ECOLOGY_CODE

        code = 0
        if self._adjacent_to_water(x, y):
            code |= HYDROLOGY_SUPPORT_FLAGS["adjacent_to_water"]
        if tile.terrain == "wetland":
            code |= HYDROLOGY_SUPPORT_FLAGS["wetland"]
        if self._is_flooded_tile(x, y):
            code |= HYDROLOGY_SUPPORT_FLAGS["flooded"]
        return code

    def _water_access_reason(self, x: int, y: int) -> str:
        tile = self.grid[y][x]
        if tile.terrain == "water":
            return "none"
        if tile.terrain == "wetland":
            return "wetland"
        if self._adjacent_to_water(x, y):
            return "adjacent_water"
        if self._is_flooded_tile(x, y):
            return "flooded"
        return "none"

    def _hydrology_snapshot(
        self,
    ) -> tuple[list[list[int]], list[list[int]], dict[str, int], dict[str, int], dict[str, int]]:
        primary_codes: list[list[int]] = []
        support_codes: list[list[int]] = []
        primary_counts = {reason: 0 for reason in HYDROLOGY_REASON_CODES}
        support_counts = {
            "shoreline_support": 0,
            "wetland_support": 0,
            "flooded_support": 0,
        }
        primary_stats = {"hard_access_tiles": 0}
        for y, row in enumerate(self.grid):
            primary_row: list[int] = []
            support_row: list[int] = []
            for x, tile in enumerate(row):
                if tile.terrain == "water":
                    primary_row.append(NON_LAND_ECOLOGY_CODE)
                    support_row.append(NON_LAND_ECOLOGY_CODE)
                    continue
                reason = self._water_access_reason(x, y)
                support_code = self._hydrology_support_code(x, y)
                primary_row.append(HYDROLOGY_REASON_CODES[reason])
                support_row.append(support_code)
                primary_counts[reason] += 1
                if reason != "none":
                    primary_stats["hard_access_tiles"] += 1
                if support_code & HYDROLOGY_SUPPORT_FLAGS["adjacent_to_water"]:
                    support_counts["shoreline_support"] += 1
                if support_code & HYDROLOGY_SUPPORT_FLAGS["wetland"]:
                    support_counts["wetland_support"] += 1
                if support_code & HYDROLOGY_SUPPORT_FLAGS["flooded"]:
                    support_counts["flooded_support"] += 1
            primary_codes.append(primary_row)
            support_codes.append(support_row)
        return primary_codes, support_codes, primary_counts, support_counts, primary_stats

    def _refuge_snapshot(
        self,
    ) -> tuple[list[list[int]], list[list[int]], dict[str, int], dict[str, float]]:
        codes: list[list[int]] = []
        score_codes: list[list[int]] = []
        counts = {reason: 0 for reason in SOFT_REFUGE_CODES}
        scores: list[float] = []
        for y, row in enumerate(self.grid):
            code_row: list[int] = []
            score_row: list[int] = []
            for x, tile in enumerate(row):
                if tile.terrain == "water":
                    code_row.append(NON_LAND_ECOLOGY_CODE)
                    score_row.append(NON_LAND_ECOLOGY_CODE)
                    continue
                score = self._refuge_score(x, y)
                reason = self._soft_refuge_reason(x, y)
                code_row.append(SOFT_REFUGE_CODES[reason])
                score_row.append(round(score * 100))
                counts[reason] += 1
                if tile.terrain == "forest":
                    scores.append(score)
            codes.append(code_row)
            score_codes.append(score_row)
        return (
            codes,
            score_codes,
            counts,
            {
                "avg_refuge_score_forest_tiles": round(sum(scores) / max(len(scores), 1), 4),
                "forest_tiles_evaluated": len(scores),
            },
        )

    def _ecology_snapshot(self) -> tuple[list[list[int]], dict[str, int], dict[str, float]]:
        codes: list[list[int]] = []
        counts = {state: 0 for state in ECOLOGY_STATE_CODES}
        vegetation_values: list[float] = []
        recovery_values: list[float] = []
        for row in self.grid:
            code_row: list[int] = []
            for tile in row:
                if tile.terrain == "water":
                    code_row.append(NON_LAND_ECOLOGY_CODE)
                    continue
                state = self._ecology_state_for_tile(tile)
                code_row.append(ECOLOGY_STATE_CODES[state])
                counts[state] += 1
                vegetation_values.append(tile.vegetation)
                recovery_values.append(tile.recovery_debt)
            codes.append(code_row)
        return (
            codes,
            counts,
            {
                "avg_vegetation": round(sum(vegetation_values) / max(len(vegetation_values), 1), 4),
                "avg_recovery_debt": round(sum(recovery_values) / max(len(recovery_values), 1), 4),
            },
        )

    def _hazard_at(self, x: int, y: int) -> tuple[str, float]:
        tile = self.grid[y][x]
        if tile.terrain == "water":
            return "none", 0.0

        _, moisture, heat = self._effective_tile_fields(x, y)
        habitat_state = self._habitat_state_at(x, y)
        support_code = self._hydrology_support_code(x, y)
        refuge_score = self._refuge_score(x, y)
        soft_refuge = self._soft_refuge_reason(x, y)
        dryness = max(0.0, heat - moisture)
        openness = max(0.0, 0.4 - tile.shelter)
        vegetation_stress = max(0.0, 0.42 - tile.vegetation)

        exposure = (
            max(0.0, heat - 0.42) * 0.52
            + dryness * 0.36
            + openness * 0.32
            + vegetation_stress * 0.22
            + tile.recovery_debt * 0.14
        )
        if habitat_state == "parched":
            exposure += 0.2 if tile.terrain != "rocky" else 0.08
        if soft_refuge == "canopy_refuge":
            exposure -= 0.1 + refuge_score * 0.08
        exposure -= tile.shelter * 0.16
        exposure = self._clamp01(exposure)

        instability = tile.recovery_debt * 0.28 + vegetation_stress * 0.18
        if tile.terrain == "rocky":
            instability += 0.22 + max(0.0, heat - 0.46) * 0.18
        if support_code & HYDROLOGY_SUPPORT_FLAGS["flooded"]:
            instability += 0.28
        if habitat_state == "flooded":
            instability += 0.12 if tile.terrain == "wetland" else 0.2
        instability -= tile.shelter * 0.08
        instability = self._clamp01(instability)

        hazard_type = "exposure" if exposure >= instability else "instability"
        hazard_level = max(exposure, instability)
        if hazard_level < self.config.hazards.min_hazard_level:
            return "none", 0.0
        return hazard_type, round(hazard_level, 4)

    def _hazard_snapshot(self) -> tuple[list[list[int]], list[list[int]], dict[str, int], dict[str, float]]:
        type_codes: list[list[int]] = []
        level_codes: list[list[int]] = []
        counts = {hazard_type: 0 for hazard_type in HAZARD_TYPE_CODES}
        hazard_values: list[float] = []
        for y, row in enumerate(self.grid):
            type_row: list[int] = []
            level_row: list[int] = []
            for x, tile in enumerate(row):
                if tile.terrain == "water":
                    type_row.append(NON_LAND_ECOLOGY_CODE)
                    level_row.append(NON_LAND_ECOLOGY_CODE)
                    continue
                hazard_type, hazard_level = self._hazard_at(x, y)
                type_row.append(HAZARD_TYPE_CODES[hazard_type])
                level_row.append(round(hazard_level * 100))
                counts[hazard_type] += 1
                hazard_values.append(hazard_level)
            type_codes.append(type_row)
            level_codes.append(level_row)
        return (
            type_codes,
            level_codes,
            counts,
            {
                "hazardous_tiles": counts["exposure"] + counts["instability"],
                "avg_hazard_level": round(sum(hazard_values) / max(len(hazard_values), 1), 4),
            },
        )

    def _carcass_snapshot(self) -> tuple[list[list[int]], list[list[int]], dict[str, float]]:
        return runtime_resources.carcass_snapshot(self)

    def _fresh_kill_snapshot(self) -> tuple[list[list[int]], dict[str, float]]:
        return runtime_resources.fresh_kill_snapshot(self)

    def _biotic_field_snapshot(
        self,
    ) -> tuple[dict[str, list[list[float]]], dict[str, dict[str, float]]]:
        state = self._current_biotic_state()
        maps = state.to_serializable()
        stats: dict[str, dict[str, float]] = {}
        for name, field in (
            ("prey_biomass", state.prey_biomass),
            ("carrion", state.carrion),
            ("predator_risk", state.predator_risk),
        ):
            values = [
                field[y][x]
                for y in range(self.config.height)
                for x in range(self.config.width)
                if self.grid[y][x].terrain != "water"
            ]
            stats[name] = {
                "avg": round(sum(values) / max(len(values), 1), 4),
                "max": round(max(values, default=0.0), 4),
            }
        return maps, stats

    def _signal_field_snapshot(
        self,
    ) -> tuple[dict[str, list[list[float]]], dict[str, dict[str, float]]]:
        state = self._current_signal_state()
        return state.to_serializable(), runtime_signals.signal_field_stats(
            state,
            self.grid,
        )

    def _trophic_role_grid(self, alive: list[Agent]) -> list[list[int]]:
        grid = [
            [TROPHIC_ROLE_CODES["none"] for _ in range(self.config.width)]
            for _ in range(self.config.height)
        ]
        for agent in alive:
            grid[agent.y][agent.x] = TROPHIC_ROLE_CODES[self._trophic_role(agent)]
        return grid

    def _meat_mode_grid(self, alive: list[Agent]) -> list[list[int]]:
        grid = [
            [MEAT_MODE_CODES["none"] for _ in range(self.config.width)]
            for _ in range(self.config.height)
        ]
        for agent in alive:
            grid[agent.y][agent.x] = MEAT_MODE_CODES[self._meat_mode(agent)]
        return grid

    @staticmethod
    def _energy_ratio(agent: Agent) -> float:
        return agent.energy / max(agent.genome.max_energy, 1e-9)

    @staticmethod
    def _hydration_ratio(agent: Agent) -> float:
        return agent.hydration / max(agent.genome.max_hydration, 1e-9)

    @staticmethod
    def _health_ratio(agent: Agent) -> float:
        return agent.health / max(agent.max_health, 1e-9)

    @staticmethod
    def _normalized_gene(name: str, value: float) -> float:
        lower, upper = GENE_LIMITS[name]
        if upper <= lower:
            return 0.0
        return max(0.0, min(1.0, (value - lower) / (upper - lower)))

    @staticmethod
    def _normalized_focus(primary: float, secondary: float, power: float) -> float:
        primary_term = max(primary, 0.0) ** power
        secondary_term = max(secondary, 0.0) ** power
        total = primary_term + secondary_term
        if total <= 1e-9:
            return 0.5
        return primary_term / total

    @staticmethod
    def _energy_headroom(agent: Agent) -> float:
        return max(0.0, agent.genome.max_energy - agent.energy)

    @staticmethod
    def _health_headroom(agent: Agent) -> float:
        return max(0.0, agent.max_health - agent.health)

    @staticmethod
    def _hydration_headroom(agent: Agent) -> float:
        return max(0.0, agent.genome.max_hydration - agent.hydration)

    def _decay_recent_diet(self, agent: Agent) -> None:
        decay = self.config.diet_matching.reservoir_decay
        agent.recent_plant_energy *= decay
        agent.recent_fresh_kill_energy *= decay
        agent.recent_carcass_energy *= decay

    @staticmethod
    def _recent_diet_total(agent: Agent) -> float:
        return (
            agent.recent_plant_energy
            + agent.recent_fresh_kill_energy
            + agent.recent_carcass_energy
        )

    def _matched_diet_ratio(self, agent: Agent, profile: TrophicProfile | None = None) -> float:
        profile = profile or self._trophic_profile(agent)
        if profile.meat_mode in {"hunter", "scavenger"}:
            # Low-yield plant fallback keeps animal specialists alive, but it should
            # not drown out sparse successful animal-resource meals in reproduction
            # readiness. A specialist still needs actual meat/carrion for a match.
            plant_fallback_weight = min(0.04, max(0.0, profile.plant_drive))
            total_recent = (
                agent.recent_plant_energy * plant_fallback_weight
                + agent.recent_fresh_kill_energy
                + agent.recent_carcass_energy
            )
        else:
            total_recent = self._recent_diet_total(agent)
        if total_recent <= 1e-9:
            return 0.0
        if profile.role == "herbivore":
            matched_recent = agent.recent_plant_energy
        elif profile.meat_mode == "hunter":
            matched_recent = (
                agent.recent_fresh_kill_energy
                + agent.recent_carcass_energy * 0.45
            )
        elif profile.meat_mode == "scavenger":
            matched_recent = (
                agent.recent_carcass_energy
                + agent.recent_fresh_kill_energy * 0.55
            )
        else:
            matched_recent = (
                agent.recent_plant_energy * 0.65
                + agent.recent_fresh_kill_energy * 0.85
                + agent.recent_carcass_energy * 0.75
            )
        return self._clamp01(matched_recent / total_recent)

    def _matched_diet_threshold(self, profile: TrophicProfile) -> float:
        if profile.role == "herbivore" or (
            profile.role == "carnivore" and profile.meat_mode in {"hunter", "scavenger"}
        ):
            return self.config.diet_matching.specialist_threshold
        return self.config.diet_matching.omnivore_threshold

    @staticmethod
    def _record_recent_diet(agent: Agent, food_source: str, gained_energy: float) -> None:
        runtime_feeding.record_recent_diet(agent, food_source, gained_energy)

    def _compute_trophic_profile_for_genome(self, genome: Genome) -> TrophicProfile:
        attack_cost_efficiency = 1.0 - self._normalized_gene(
            "attack_cost_multiplier",
            genome.attack_cost_multiplier,
        )
        plant_trait = (
            self._normalized_gene("food_efficiency", genome.food_efficiency) * 0.35
            + self._normalized_gene("plant_bias", genome.plant_bias) * 0.65
        )
        scavenger_trait = (
            self._normalized_gene("meat_efficiency", genome.meat_efficiency) * 0.35
            + self._normalized_gene("carrion_bias", genome.carrion_bias) * 0.65
        )
        hunter_trait = (
            self._normalized_gene("attack_power", genome.attack_power) * 0.28
            + attack_cost_efficiency * 0.24
            + self._normalized_gene("live_prey_bias", genome.live_prey_bias) * 0.3
            + self._normalized_gene("defense_rating", genome.defense_rating) * 0.18
        )
        animal_trait = max(
            scavenger_trait * 0.56 + hunter_trait * 0.44,
            scavenger_trait * 0.82,
            hunter_trait * 0.82,
        )
        total_trait = plant_trait + animal_trait
        if total_trait <= 1e-9:
            plant_share = 0.5
            animal_share = 0.5
        else:
            plant_share = plant_trait / total_trait
            animal_share = animal_trait / total_trait
        meat_total = scavenger_trait + hunter_trait
        if meat_total <= 1e-9:
            scavenger_share = 0.5
            hunter_share = 0.5
        else:
            scavenger_share = scavenger_trait / meat_total
            hunter_share = hunter_trait / meat_total
        breadth = 1.0 - abs(plant_share - animal_share)
        plant_focus = self._normalized_focus(
            plant_share,
            animal_share,
            self.config.trophic.channel_focus_power,
        )
        animal_focus = self._normalized_focus(
            animal_share,
            plant_share,
            self.config.trophic.channel_focus_power,
        )
        scavenger_focus = self._normalized_focus(
            scavenger_share,
            hunter_share,
            self.config.trophic.mode_focus_power,
        )
        hunter_focus = self._normalized_focus(
            hunter_share,
            scavenger_share,
            self.config.trophic.mode_focus_power,
        )
        breadth_penalty = max(0.18, 1.0 - breadth * self.config.trophic.breadth_penalty)
        focused_plant_drive = plant_trait * plant_focus * breadth_penalty
        focused_animal_drive = animal_trait * animal_focus * breadth_penalty

        specialist_threshold = self.config.trophic.specialist_share_threshold
        if plant_focus >= specialist_threshold and focused_plant_drive >= 0.2:
            role = "herbivore"
        elif animal_focus >= specialist_threshold and focused_animal_drive >= 0.26:
            role = "carnivore"
        else:
            role = "omnivore"

        if role == "omnivore":
            breadth_channel_support = breadth * 1.8
            plant_drive = max(
                focused_plant_drive,
                plant_trait * plant_share * breadth_channel_support,
            )
            animal_drive = max(
                focused_animal_drive,
                animal_trait * animal_share * breadth_channel_support,
            )
        else:
            plant_drive = focused_plant_drive
            animal_drive = focused_animal_drive

        scavenger_drive = animal_drive * (0.22 + scavenger_trait * 0.78) * scavenger_focus
        hunter_drive = animal_drive * (0.18 + hunter_trait * 0.82) * hunter_focus

        if role == "herbivore" or animal_share < self.config.trophic.animal_channel_threshold:
            meat_mode = "none"
        elif scavenger_focus >= 0.66 and scavenger_drive >= hunter_drive * 1.08:
            meat_mode = "scavenger"
        elif hunter_focus >= 0.66 and hunter_drive >= scavenger_drive * 1.08:
            meat_mode = "hunter"
        else:
            meat_mode = "mixed"

        return TrophicProfile(
            plant_share=plant_share,
            animal_share=animal_share,
            scavenger_share=scavenger_share,
            hunter_share=hunter_share,
            breadth=breadth,
            plant_drive=self._clamp01(plant_drive),
            animal_drive=self._clamp01(animal_drive),
            scavenger_drive=self._clamp01(scavenger_drive),
            hunter_drive=self._clamp01(hunter_drive),
            role=role,
            meat_mode=meat_mode,
        )

    def _lifecycle_context(self) -> runtime_lifecycle.LifecycleContext:
        return runtime_lifecycle.LifecycleContext(
            trophic_profile_cache=self._trophic_profile_cache,
            compute_trophic_profile_for_genome=self._compute_trophic_profile_for_genome,
            season_state=self._season_state,
            trophic_profile=self._trophic_profile,
            trophic_role=self._trophic_role,
            meat_mode=self._meat_mode,
            agent_energy_drain_modifier=self._agent_energy_drain_modifier,
            agent_hydration_drain_modifier=self._agent_hydration_drain_modifier,
            energy_ratio=self._energy_ratio,
            hydration_ratio=self._hydration_ratio,
            hazard_at=self._hazard_at,
            refuge_score=self._refuge_score,
            clamp01=self._clamp01,
            emit=self._emit,
            kill_agent=self._kill_agent,
        )

    def _trophic_profile_for_genome(self, genome: Genome) -> TrophicProfile:
        return cached_trophic_profile(
            genome,
            lifecycle_context=self._lifecycle_context(),
        )

    def _trophic_profile(self, agent: Agent) -> TrophicProfile:
        return cached_trophic_profile(
            agent.genome,
            lifecycle_context=self._lifecycle_context(),
        )

    def _trophic_role_for_genome(self, genome: Genome) -> str:
        return self._trophic_profile_for_genome(genome).role

    def _trophic_role(self, agent: Agent) -> str:
        return self._trophic_profile(agent).role

    def _meat_mode(self, agent: Agent) -> str:
        return self._trophic_profile(agent).meat_mode

    def _can_consume_animal(self, agent: Agent) -> bool:
        profile = self._trophic_profile(agent)
        return (
            self._has_animal_resource_channel(profile)
            and max(profile.scavenger_drive, profile.hunter_drive)
            >= self.config.trophic.animal_use_drive_threshold
        )

    def _has_animal_resource_channel(self, profile: TrophicProfile) -> bool:
        return (
            profile.role != "herbivore"
            and profile.animal_share >= self.config.trophic.animal_channel_threshold
        )

    def _can_consume_fresh_kill(self, agent: Agent) -> bool:
        profile = self._trophic_profile(agent)
        return self._has_animal_resource_channel(profile) and (
            profile.meat_mode in {"hunter", "scavenger", "mixed"}
            or max(profile.scavenger_drive, profile.hunter_drive)
            >= self.config.trophic.animal_use_drive_threshold
        )

    def _can_consume_carcass(self, agent: Agent) -> bool:
        profile = self._trophic_profile(agent)
        return self._has_animal_resource_channel(profile) and (
            profile.meat_mode in {"hunter", "scavenger", "mixed"}
            or max(profile.scavenger_drive, profile.hunter_drive)
            >= self.config.trophic.animal_use_drive_threshold
        )

    def _can_attack(self, agent: Agent) -> bool:
        combat = self.config.combat
        profile = self._trophic_profile(agent)
        return (
            profile.hunter_drive >= self.config.trophic.attack_channel_threshold
            and self._health_ratio(agent) >= combat.min_attack_health_ratio
            and self._energy_ratio(agent) >= combat.min_attack_energy_ratio
            and self._hydration_ratio(agent) >= combat.min_attack_hydration_ratio
        )

    def _plant_intake_useful(self, agent: Agent) -> bool:
        return self._energy_headroom(agent) >= 0.015

    def _animal_intake_useful(self, agent: Agent) -> bool:
        return self._energy_headroom(agent) >= 0.015 or self._health_headroom(agent) >= 0.03

    def _carcass_intake_useful(self, agent: Agent) -> bool:
        if self._animal_intake_useful(agent):
            return True
        profile = self._trophic_profile(agent)
        return (
            profile.meat_mode == "scavenger"
            and self.config.carcasses.scavenger_hydration_fraction > 0.0
            and self._hydration_headroom(agent) >= 0.015
        )

    def _fresh_kill_intake_useful(self, agent: Agent) -> bool:
        return self._animal_intake_useful(agent)

    def _plant_food_value(
        self,
        agent: Agent,
        tile: Tile,
        profile: TrophicProfile,
        consumed: float | None = None,
        *,
        x: int | None = None,
        y: int | None = None,
    ) -> float:
        if tile.terrain == "water":
            return 0.0
        amount = min(tile.food, self.config.resources.eat_amount) if consumed is None else consumed
        if amount <= 0:
            return 0.0
        tile_x = agent.x if x is None else x
        tile_y = agent.y if y is None else y
        gained = (
            amount
            * self._terrain_food_multiplier(agent, tile.terrain)
            * max(0.62, 0.82 + tile.vegetation * 0.32 - tile.recovery_debt * 0.2)
        )
        habitat_state = self._habitat_state_at(tile_x, tile_y)
        if habitat_state == "bloom":
            gained *= 1.12
        elif habitat_state == "flooded" and tile.terrain != "wetland":
            gained *= 0.92
        elif habitat_state == "parched":
            gained *= 0.9 if tile.terrain == "rocky" else 0.82
        ecology_state = self._ecology_state_at(tile_x, tile_y)
        if ecology_state == "lush":
            gained *= 1.06
        elif ecology_state == "recovering":
            gained *= 0.94
        elif ecology_state == "depleted":
            gained *= 0.82
        animal_mode_survival_floor = 0.0
        if (
            profile.meat_mode != "none"
            and self._energy_ratio(agent)
            < self.config.trophic.animal_mode_plant_survival_energy_threshold
        ):
            animal_mode_survival_floor = (
                self.config.trophic.animal_mode_plant_survival_floor
            )
        if profile.role == "carnivore":
            fallback_drive = max(
                profile.plant_drive * 0.04,
                0.035 + profile.plant_share * 0.035,
                animal_mode_survival_floor,
            )
            return gained * fallback_drive
        elif profile.meat_mode == "hunter":
            gained *= max(
                0.24 + profile.plant_share * 0.32,
                animal_mode_survival_floor,
            )
            return gained
        if profile.meat_mode != "none":
            return gained * max(profile.plant_drive, animal_mode_survival_floor)
        return gained * profile.plant_drive

    @staticmethod
    def _fresh_kill_drive(profile: TrophicProfile) -> float:
        drive = profile.hunter_drive + profile.scavenger_drive * 0.28
        if profile.meat_mode == "hunter":
            return max(drive, 0.45)
        if profile.meat_mode == "scavenger":
            return max(drive, 0.14)
        return drive

    @staticmethod
    def _carcass_drive(profile: TrophicProfile) -> float:
        drive = profile.scavenger_drive + profile.hunter_drive * 0.24
        if profile.meat_mode == "scavenger":
            return max(drive, 0.34)
        if profile.meat_mode == "mixed":
            return max(drive, 0.16)
        return drive

    def _fresh_kill_nutrition(
        self,
        agent: Agent,
        consumed: float,
        profile: TrophicProfile,
    ) -> float:
        if consumed <= 0:
            return 0.0
        return (
            consumed
            * agent.genome.meat_efficiency
            * self._fresh_kill_drive(profile)
            * 1.58
        )

    def _scavenger_carcass_hydration_fraction(self, agent: Agent) -> float:
        base_fraction = self.config.carcasses.scavenger_hydration_fraction
        # Below this point calories are still the limiting resource, so avoid
        # turning weak carrion intake into a broad scavenger survival boost.
        if (
            self._energy_ratio(agent)
            < self.config.trophic.animal_mode_plant_survival_energy_threshold
        ):
            return base_fraction
        hydration_deficit = max(
            0.0,
            self.config.reproduction.min_hydration_fraction
            - self._hydration_ratio(agent),
        )
        return base_fraction * (1.0 + hydration_deficit * 2.0)

    def _carcass_nutrition(
        self,
        agent: Agent,
        consumed: float,
        freshness: float,
        profile: TrophicProfile,
    ) -> float:
        if consumed <= 0:
            return 0.0
        return (
            consumed
            * agent.genome.meat_efficiency
            * self._carcass_drive(profile)
            * (0.76 + freshness * 0.24)
            * 1.28
        )

    def _fresh_kill_food_value(
        self,
        agent: Agent,
        tile: Tile,
        profile: TrophicProfile,
        consumed: float | None = None,
    ) -> float:
        amount = (
            min(tile.fresh_kill_energy, self.config.resources.eat_amount * 1.1)
            if consumed is None
            else consumed
        )
        if amount <= 0:
            return 0.0
        return self._fresh_kill_nutrition(agent, amount, profile)

    def _carcass_food_value(
        self,
        agent: Agent,
        tile: Tile,
        profile: TrophicProfile,
        consumed: float | None = None,
    ) -> float:
        amount = (
            min(tile.carcass_energy, self.config.resources.eat_amount * 1.25)
            if consumed is None
            else consumed
        )
        if amount <= 0:
            return 0.0
        freshness = 0.7 + tile.carcass_decay * 0.3
        return self._carcass_nutrition(agent, amount, freshness, profile)

    def _project_carcass_yield(self, agent: Agent) -> float:
        return min(
            1.25,
            self.config.carcasses.base_energy
            + self._health_ratio(agent) * self.config.carcasses.health_ratio_yield
            + (agent.max_health / max(GENE_LIMITS["max_health"][1], 1e-9))
            * self.config.carcasses.body_capacity_yield,
        )

    def _record_feeding_event(
        self,
        agent: Agent,
        food_source: str,
        consumed: float,
        gained_energy: float,
        profile: TrophicProfile,
        energy_before: float,
        energy_after: float,
        potential_energy: float | None = None,
    ) -> None:
        runtime_feeding.record_feeding_event(
            self,
            agent,
            food_source,
            consumed,
            gained_energy,
            profile,
            energy_before=energy_before,
            energy_after=energy_after,
            potential_energy=potential_energy,
        )

    def _record_animal_resource_consumption(
        self,
        meat_mode: str,
        food_source: str,
        consumed: float,
        gained_energy: float,
    ) -> None:
        runtime_feeding.record_animal_resource_consumption(
            self,
            meat_mode,
            food_source,
            consumed,
            gained_energy,
        )

    def _animal_resource_presence_this_tick(self) -> dict[str, bool]:
        return runtime_feeding.animal_resource_presence_this_tick(self)

    @staticmethod
    def _empty_animal_resource_reachability_tick_counts() -> dict[str, int]:
        return runtime_feeding.empty_animal_resource_reachability_tick_counts()

    def _agent_reachable_animal_resources(
        self,
        agent: Agent,
        *,
        radius: int,
        action_mask: dict[str, bool] | None = None,
    ) -> dict[str, object]:
        can_consume_fresh_kill = self._can_consume_fresh_kill(agent)
        can_consume_carcass = self._can_consume_carcass(agent)
        empty: dict[str, object] = {
            "fresh_kill": False,
            "carcass": False,
            "animal_resource": False,
            "fresh_kill_policy_actionable": False,
            "carcass_policy_actionable": False,
            "animal_resource_policy_actionable": False,
            "fresh_kill_policy_blockers": set(),
            "carcass_policy_blockers": set(),
            "animal_resource_policy_blockers": set(),
        }
        if not can_consume_fresh_kill and not can_consume_carcass:
            return empty

        targets: dict[str, list[tuple[int, int, int]]] = {
            "fresh_kill": [],
            "carcass": [],
        }
        visited = {(agent.x, agent.y)}
        frontier = deque([(agent.x, agent.y, 0)])
        while frontier:
            x, y, distance = frontier.popleft()
            tile = self.grid[y][x]
            if can_consume_fresh_kill and tile.fresh_kill_energy > 1e-9:
                targets["fresh_kill"].append((distance, x, y))
            if can_consume_carcass and tile.carcass_energy > 1e-9:
                targets["carcass"].append((distance, x, y))
            if distance >= radius:
                continue
            for nx, ny in (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ):
                if (nx, ny) in visited:
                    continue
                if not self._in_bounds(nx, ny):
                    continue
                if self.grid[ny][nx].terrain == "water":
                    continue
                visited.add((nx, ny))
                frontier.append((nx, ny, distance + 1))

        policy_action_mask = (
            action_mask
            if action_mask is not None
            else runtime_action_space.build_action_mask(
                self._action_mask_context(agent)
            )
        )
        fresh_kill_actionable, fresh_kill_blockers = self._policy_actionable_resource(
            agent,
            "fresh_kill",
            targets["fresh_kill"],
            policy_action_mask,
        )
        carcass_actionable, carcass_blockers = self._policy_actionable_resource(
            agent,
            "carcass",
            targets["carcass"],
            policy_action_mask,
        )
        animal_blockers = set(fresh_kill_blockers) | set(carcass_blockers)
        return {
            "fresh_kill": bool(targets["fresh_kill"]),
            "carcass": bool(targets["carcass"]),
            "animal_resource": bool(targets["fresh_kill"] or targets["carcass"]),
            "fresh_kill_policy_actionable": fresh_kill_actionable,
            "carcass_policy_actionable": carcass_actionable,
            "animal_resource_policy_actionable": fresh_kill_actionable
            or carcass_actionable,
            "fresh_kill_policy_blockers": fresh_kill_blockers,
            "carcass_policy_blockers": carcass_blockers,
            "animal_resource_policy_blockers": animal_blockers,
        }

    def _policy_actionable_resource(
        self,
        agent: Agent,
        resource: str,
        targets: list[tuple[int, int, int]],
        action_mask: dict[str, bool],
    ) -> tuple[bool, set[str]]:
        if not targets:
            return False, set()
        blockers: set[str] = set()
        for _, x, y in sorted(targets):
            candidate_blockers = self._policy_resource_blockers(
                agent,
                resource,
                x,
                y,
                action_mask,
            )
            if not candidate_blockers:
                return True, set()
            blockers.update(candidate_blockers)
        return False, blockers

    def _policy_resource_blockers(
        self,
        agent: Agent,
        resource: str,
        target_x: int,
        target_y: int,
        action_mask: dict[str, bool],
    ) -> set[str]:
        if target_x == agent.x and target_y == agent.y:
            return set() if bool(action_mask.get("eat", False)) else {"movement_mask"}

        blockers: set[str] = set()
        target_tile = self.grid[target_y][target_x]
        if (
            resource == "carcass"
            and self._meat_mode(agent) == "scavenger"
            and abs(target_x - agent.x) + abs(target_y - agent.y) == 1
            and target_tile.terrain != "water"
            and target_tile.carcass_energy > 1e-9
            and bool(action_mask.get("eat", False))
        ):
            return set()
        if (
            target_tile.occupant_id is not None
            and target_tile.occupant_id != agent.agent_id
        ):
            blockers.add("occupant")
        target_hazard_type, target_hazard_level = self._hazard_at(target_x, target_y)
        if (
            target_hazard_type != "none"
            and target_hazard_level >= self.config.hazards.min_hazard_level
        ):
            blockers.add("hazard")

        dx = target_x - agent.x
        dy = target_y - agent.y
        step_actions = self._candidate_step_actions_for_delta(dx, dy)
        if not step_actions:
            blockers.add("movement_mask")
            return blockers

        step_blockers: set[str] = set()
        for action in step_actions:
            action_dx, action_dy = self._movement_delta_for_action(action)
            next_x = agent.x + action_dx
            next_y = agent.y + action_dy
            candidate_step_blockers = self._movement_step_blockers(
                action,
                next_x,
                next_y,
                action_mask,
            )
            if not candidate_step_blockers and not blockers:
                return set()
            step_blockers.update(candidate_step_blockers)
        blockers.update(step_blockers or {"movement_mask"})
        return blockers

    @staticmethod
    def _candidate_step_actions_for_delta(dx: int, dy: int) -> list[str]:
        candidates: list[str] = []
        if abs(dx) >= abs(dy):
            candidates.append("move_east" if dx > 0 else "move_west")
            if dy != 0:
                candidates.append("move_south" if dy > 0 else "move_north")
        else:
            candidates.append("move_south" if dy > 0 else "move_north")
            if dx != 0:
                candidates.append("move_east" if dx > 0 else "move_west")
        return candidates

    @staticmethod
    def _movement_delta_for_action(action: str) -> tuple[int, int]:
        if action == "move_north":
            return (0, -1)
        if action == "move_south":
            return (0, 1)
        if action == "move_east":
            return (1, 0)
        if action == "move_west":
            return (-1, 0)
        return (0, 0)

    def _movement_step_blockers(
        self,
        action: str,
        x: int,
        y: int,
        action_mask: dict[str, bool],
    ) -> set[str]:
        blockers: set[str] = set()
        if not self._in_bounds(x, y):
            blockers.add("movement_mask")
            return blockers
        tile = self.grid[y][x]
        if tile.terrain == "water":
            blockers.add("water")
        if tile.occupant_id is not None:
            blockers.add("occupant")
        hazard_type, hazard_level = self._hazard_at(x, y)
        if hazard_type != "none" and hazard_level >= self.config.hazards.min_hazard_level:
            blockers.add("hazard")
        if not bool(action_mask.get(action, False)) and not blockers:
            blockers.add("movement_mask")
        return blockers

    def _animal_resource_reachability_by_meat_mode(
        self,
        agents: list[Agent],
        *,
        action_masks_by_agent: dict[int, dict[str, bool]] | None = None,
        resource_presence: dict[str, bool] | None = None,
    ) -> dict[str, dict[str, int]]:
        return runtime_feeding.animal_resource_reachability_by_meat_mode(
            self,
            agents,
            meat_mode_codes=MEAT_MODE_CODES,
            radius=runtime_observations.NAVIGATION_RADIUS,
            action_masks_by_agent=action_masks_by_agent,
            resource_presence=resource_presence,
        )

    def _record_animal_resource_opportunity_tick(
        self,
        meat_mode_counts: dict[str, int],
        reachability_by_meat_mode: dict[str, dict[str, int]],
    ) -> None:
        runtime_feeding.record_animal_resource_opportunity_tick(
            self,
            meat_mode_counts,
            reachability_by_meat_mode,
        )

    def _agent_biomass(self, agent: Agent) -> float:
        return (
            self._health_ratio(agent) * 0.42
            + self._energy_ratio(agent) * 0.34
            + self._hydration_ratio(agent) * 0.24
        )

    def _prey_vulnerability(self, agent: Agent) -> float:
        return (
            1.0
            + agent.injury_load * 0.9
            + (1.0 - self._energy_ratio(agent)) * 0.55
            + (1.0 - self._hydration_ratio(agent)) * 0.45
            + (1.0 - self._health_ratio(agent)) * 0.7
        )

    def _biotic_diffusion_context(self) -> BioticDiffusionContext:
        return BioticDiffusionContext(
            width=self.config.width,
            height=self.config.height,
            grid=self.grid,
            diffusion_radius=self.config.biotic_fields.diffusion_radius,
            target_cache=self._biotic_diffusion_target_cache,
            record_runtime_cost=self._record_runtime_cost,
        )

    def _biotic_state_context(self) -> BioticStateContext:
        return BioticStateContext(
            width=self.config.width,
            grid=self.grid,
            alive_agents=self.alive_agents(),
            trophic_profile=self._trophic_profile,
            agent_biomass=self._agent_biomass,
            prey_vulnerability=self._prey_vulnerability,
            health_ratio=self._health_ratio,
            diffusion=self._biotic_diffusion_context(),
        )

    def _diffuse_biotic_field(self, sources: list[list[float]]) -> list[list[float]]:
        return diffuse_runtime_biotic_field(
            sources,
            context=self._biotic_diffusion_context(),
        )

    def _build_biotic_state(self) -> BioticFieldState:
        self._record_runtime_cost("biotic_state_builds")
        return build_runtime_biotic_state(self._biotic_state_context())

    def _current_biotic_state(self) -> BioticFieldState:
        if (
            self.cached_biotic_state_revision == self.biotic_state_revision
            and self.cached_biotic_state is not None
        ):
            self._record_runtime_cost("biotic_state_cache_hits")
            return self.cached_biotic_state
        self.cached_biotic_state = self._build_biotic_state()
        self.cached_biotic_state_revision = self.biotic_state_revision
        return self.cached_biotic_state

    def _signal_runtime_context(self) -> runtime_signals.SignalRuntimeContext:
        return runtime_signals.SignalRuntimeContext(
            config=self.config.signals,
            width=self.config.width,
            height=self.config.height,
            grid=self.grid,
            tick=self.tick,
            reproductive_signal_emissions=self.reproductive_signal_emissions,
            communication_signal_emissions=self.communication_signal_emissions,
            tick_signal_emission_events=self.tick_signal_emission_events,
            tick_signal_totals=self.tick_signal_totals,
            run_signal_totals=self.run_signal_totals,
            diffusion_target_cache=self._signal_diffusion_target_cache,
            record_runtime_cost=self._record_runtime_cost,
            invalidate_signal_state=self._invalidate_signal_state,
            is_biologically_reproduction_ready=(
                self._is_biologically_reproduction_ready
            ),
            trophic_profile=self._trophic_profile,
            reproduction_energy_requirement=self._reproduction_energy_requirement,
        )

    def _build_signal_state(self) -> runtime_signals.SignalFieldState:
        self._record_runtime_cost("signal_state_builds")
        return runtime_signals.build_signal_state(
            context=self._signal_runtime_context(),
        )

    def _current_signal_state(self) -> runtime_signals.SignalFieldState:
        if (
            self.cached_signal_state_revision == self.signal_state_revision
            and self.cached_signal_state is not None
        ):
            self._record_runtime_cost("signal_state_cache_hits")
            return self.cached_signal_state
        self.cached_signal_state = self._build_signal_state()
        self.cached_signal_state_revision = self.signal_state_revision
        return self.cached_signal_state

    def _local_prey_vulnerability_score(self, hunter: Agent, x: int, y: int) -> float:
        score = 0.0
        for ny in range(max(0, y - 1), min(self.config.height, y + 2)):
            for nx in range(max(0, x - 1), min(self.config.width, x + 2)):
                distance = abs(nx - x) + abs(ny - y)
                if distance == 0 or distance > 1:
                    continue
                target_id = self.grid[ny][nx].occupant_id
                if target_id is None or target_id == hunter.agent_id:
                    continue
                target = self.agents.get(target_id)
                if target is None or not target.alive:
                    continue
                target_profile = self._trophic_profile(target)
                if target_profile.role == "herbivore":
                    score += self._prey_vulnerability(target) / (distance + 0.4)
                elif target_profile.role == "omnivore":
                    weakened = max(0.0, self._prey_vulnerability(target) - 1.1)
                    if weakened > 0:
                        score += weakened * 0.7 / (distance + 0.4)
        return score

    def _biotic_field_score(
        self,
        agent: Agent,
        x: int,
        y: int,
        profile: TrophicProfile,
        biotic_state: BioticFieldState | None = None,
    ) -> float:
        if profile.role == "herbivore":
            biotic_state = biotic_state or self._current_biotic_state()
            return -biotic_state.predator_risk[y][x] * self.config.biotic_fields.predator_risk_weight

        biotic_state = biotic_state or self._current_biotic_state()
        prey_score = biotic_state.prey_biomass[y][x] * self.config.biotic_fields.prey_weight
        carrion_score = biotic_state.carrion[y][x] * self.config.biotic_fields.carrion_weight
        predator_penalty = (
            biotic_state.predator_risk[y][x] * self.config.biotic_fields.predator_risk_weight
        )
        vulnerability_bonus = (
            self._local_prey_vulnerability_score(agent, x, y)
            * self.config.biotic_fields.hunter_vulnerability_weight
        )
        if profile.meat_mode == "hunter":
            return prey_score + vulnerability_bonus - predator_penalty
        if profile.meat_mode == "scavenger":
            return carrion_score - predator_penalty
        hunter_component = (prey_score + vulnerability_bonus) * profile.animal_share * profile.hunter_share
        scavenger_component = carrion_score * profile.animal_share * profile.scavenger_share
        return hunter_component + scavenger_component - predator_penalty * (
            0.7 + profile.plant_share * 0.3
        )

    @staticmethod
    def _band_influence(position: float, center: float, width: float) -> float:
        if width <= 0:
            return 0.0
        distance = abs(position - center)
        return max(0.0, 1.0 - distance / width)

    @staticmethod
    def _radial_influence(
        x: float,
        y: float,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> float:
        if radius <= 0:
            return 0.0
        distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        return max(0.0, 1.0 - distance / radius)

    def _effective_tile_fields(
        self,
        x: int,
        y: int,
    ) -> tuple[float, float, float]:
        climate_state = self._climate_state()
        if (
            self.cached_effective_fields_tick == self.tick
            and self.cached_effective_fields_grid is not None
        ):
            return self.cached_effective_fields_grid[y][x]

        self.cached_effective_fields_grid = self._build_effective_tile_fields_grid(
            climate_state,
        )
        self.cached_effective_fields_tick = self.tick
        return self.cached_effective_fields_grid[y][x]

    def _build_effective_tile_fields_grid(
        self,
        climate_state: dict[str, object],
    ) -> list[list[tuple[float, float, float]]]:
        return [
            [
                self._compute_effective_tile_fields(x, y, climate_state)
                for x in range(self.config.width)
            ]
            for y in range(self.config.height)
        ]

    def _compute_effective_tile_fields(
        self,
        x: int,
        y: int,
        climate_state: dict[str, object],
    ) -> tuple[float, float, float]:
        tile = self.grid[y][x]
        environment = self.config.environment
        moisture = tile.moisture + float(climate_state["moisture_shift"])
        heat = tile.heat + float(climate_state["heat_shift"])
        fertility = tile.fertility

        x_norm = x / max(self.config.width - 1, 1)
        y_norm = y / max(self.config.height - 1, 1)
        moisture += environment.moisture_front_strength * self._band_influence(
            x_norm,
            float(climate_state["moisture_front_x"]),
            environment.front_width,
        )
        heat += environment.heat_front_strength * self._band_influence(
            y_norm,
            float(climate_state["heat_front_y"]),
            environment.front_width,
        )

        if tile.terrain != "water" and self._adjacent_to_water(x, y):
            moisture += environment.adjacent_water_moisture_bonus

        disturbance_influence = self._radial_influence(
            x_norm,
            y_norm,
            float(climate_state["disturbance_center_x"]),
            float(climate_state["disturbance_center_y"]),
            environment.disturbance_radius,
        )
        if climate_state["disturbance_type"] == "storm":
            moisture += float(climate_state["disturbance_strength"]) * disturbance_influence
            heat -= float(climate_state["disturbance_strength"]) * 0.62 * disturbance_influence
            fertility += float(climate_state["disturbance_strength"]) * 0.16 * disturbance_influence
        else:
            moisture -= float(climate_state["disturbance_strength"]) * disturbance_influence
            heat += float(climate_state["disturbance_strength"]) * 0.75 * disturbance_influence
            fertility -= float(climate_state["disturbance_strength"]) * 0.18 * disturbance_influence

        moisture = self._clamp01(moisture)
        heat = self._clamp01(heat)
        fertility = self._clamp01(
            fertility + (moisture - 0.5) * environment.fertility_moisture_coupling
        )
        return fertility, moisture, heat

    def _vegetation_target(self, x: int, y: int, season: str) -> float:
        return runtime_resources.vegetation_target(
            self,
            x,
            y,
            season,
            resource_context=self._resource_context(),
        )
    def _shelter_target(self, x: int, y: int, season: str) -> float:
        return runtime_resources.shelter_target(
            self,
            x,
            y,
            season,
            resource_context=self._resource_context(),
        )
    def _food_capacity(self, x: int, y: int, season: str) -> float:
        return runtime_resources.food_capacity(
            self,
            x,
            y,
            season,
            resource_context=self._resource_context(),
        )
    def _field_growth_multiplier(self, x: int, y: int, season: str) -> float:
        return runtime_resources.field_growth_multiplier(
            self,
            x,
            y,
            season,
            resource_context=self._resource_context(),
        )
    def _field_preference_score(
        self,
        agent: Agent,
        x: int,
        y: int,
        season: str,
        water_urgency: float,
        food_urgency: float,
        profile: TrophicProfile | None = None,
        tile_memo=None,
    ) -> float:
        profile = profile or self._trophic_profile(agent)
        tile = self.grid[y][x]
        terrain = tile.terrain
        if tile_memo is not None:
            fertility, moisture, heat = tile_memo.effective_fields(x, y)
            water_reason = tile_memo.water_reason(x, y)
            refuge_score = tile_memo.refuge_score(x, y)
            soft_refuge_reason = tile_memo.soft_refuge_reason(x, y)
        else:
            fertility, moisture, heat = self._effective_tile_fields(x, y)
            water_reason = self._water_access_reason(x, y)
            refuge_score = self._refuge_score(x, y)
            soft_refuge_reason = self._soft_refuge_reason(x, y)
        habitat_state = self._habitat_state_at(x, y)
        ecology_state = self._ecology_state_at(x, y)
        heat_capacity = min(1.0, max(0.0, agent.genome.heat_tolerance / 1.8))
        heat_mismatch = max(0.0, heat - heat_capacity)
        plant_interest = profile.plant_drive
        refuge_bonus = 0.0
        if terrain == "wetland":
            refuge_bonus += water_urgency * 0.08
        elif terrain == "rocky":
            refuge_bonus += (0.08 + water_urgency * 0.06) * max(0.0, heat - 0.42)
        if habitat_state == "bloom":
            refuge_bonus += plant_interest * (0.06 + food_urgency * 0.08)
        elif habitat_state == "flooded":
            refuge_bonus += water_urgency * 0.12 - plant_interest * food_urgency * 0.04
        elif habitat_state == "parched":
            refuge_bonus -= 0.08 + water_urgency * 0.1
        vegetation_support = tile.vegetation * (
            plant_interest * (0.08 + food_urgency * 0.14) + water_urgency * 0.05
        )
        recovery_penalty = tile.recovery_debt * (
            plant_interest * (0.08 + food_urgency * 0.12) + water_urgency * 0.08
        )
        if ecology_state == "lush":
            refuge_bonus += plant_interest * (0.06 + food_urgency * 0.08)
        elif ecology_state == "recovering":
            refuge_bonus -= plant_interest * 0.03
        elif ecology_state == "depleted":
            refuge_bonus -= plant_interest * (0.08 + food_urgency * 0.08)
        refuge_bonus += refuge_score * (0.03 + water_urgency * 0.1)
        if soft_refuge_reason == "canopy_refuge":
            refuge_bonus += 0.02 + water_urgency * 0.05
        if water_reason == "adjacent_water":
            refuge_bonus += water_urgency * 0.14
        elif water_reason == "wetland":
            refuge_bonus += water_urgency * 0.1
        elif water_reason == "flooded":
            refuge_bonus += water_urgency * 0.06
        return (
            fertility * (0.04 + plant_interest * (0.08 + food_urgency * 0.28))
            + moisture * water_urgency * 0.38
            - heat_mismatch * (0.18 + water_urgency * 0.22)
            + vegetation_support
            - recovery_penalty
            + refuge_bonus
        )

    def _field_energy_modifier(self, x: int, y: int, season: str) -> float:
        tile = self.grid[y][x]
        fertility, moisture, heat = self._effective_tile_fields(x, y)
        modifier = max(0.88, 0.96 + max(0.0, heat - fertility) * 0.18 - moisture * 0.06)
        habitat_state = self._habitat_state_at(x, y)
        if habitat_state == "bloom":
            modifier *= 0.96
        elif habitat_state == "flooded":
            modifier *= 1.08
        elif habitat_state == "parched":
            modifier *= 1.05
        modifier *= 0.96 + tile.recovery_debt * 0.08
        return modifier

    def _field_hydration_modifier(self, x: int, y: int, season: str) -> float:
        tile = self.grid[y][x]
        terrain = tile.terrain
        _, moisture, heat = self._effective_tile_fields(x, y)
        modifier = max(0.74, 0.88 + heat * 0.42 - moisture * 0.26)
        habitat_state = self._habitat_state_at(x, y)
        water_reason = self._water_access_reason(x, y)
        refuge_score = self._refuge_score(x, y)
        soft_refuge_reason = self._soft_refuge_reason(x, y)
        if habitat_state == "bloom":
            modifier *= 0.95
        elif habitat_state == "flooded":
            modifier *= 0.82
        elif habitat_state == "parched":
            modifier *= 1.18 if terrain != "rocky" else 0.94
        if terrain == "wetland":
            modifier -= 0.08
        elif terrain == "rocky":
            modifier -= 0.12 if season == "dry" else 0.05
        if water_reason == "adjacent_water":
            modifier -= 0.05
        elif water_reason == "flooded":
            modifier -= 0.08
        modifier -= tile.shelter * 0.04
        if soft_refuge_reason == "canopy_refuge":
            modifier -= 0.04 + refuge_score * 0.04
        modifier += tile.recovery_debt * 0.16
        modifier -= tile.vegetation * 0.12
        return max(0.62, modifier)

    def _regrow_resources(self) -> None:
        runtime_resources.regrow_resources(self, resource_context=self._resource_context())
    def _habitat_regrowth_modifier(self, x: int, y: int) -> float:
        return runtime_resources.habitat_regrowth_modifier(
            self,
            x,
            y,
            resource_context=self._resource_context(),
        )
    def _terrain_regrowth_rate(self, terrain: str) -> float:
        return runtime_resources.terrain_regrowth_rate(self, terrain)
    def _terrain_growth_modifier(self, terrain: str, season: str) -> float:
        return runtime_resources.terrain_growth_modifier(self, terrain, season)
    def _choose_action(
        self,
        agent: Agent,
        observation: dict[str, object] | None = None,
    ) -> str:
        if observation is None:
            observation = self._observe_agent(agent)
        action_mask = dict(observation["action_mask"])
        decision = self.policy.decide(observation, action_mask)
        self._policy_action_source = decision.source
        self._policy_id = decision.policy_id
        self._policy_version = decision.policy_version
        return decision.requested_action
    def _action_scoring_context(
        self,
        agent: Agent,
    ) -> runtime_actions.ActionScoringContext:
        season = str(self._season_state()["name"])

        def set_policy_action_source(source: str) -> None:
            self._policy_action_source = source

        return runtime_actions.ActionScoringContext(
            width=self.config.width,
            height=self.config.height,
            default_vision_radius=self.config.default_vision_radius,
            animal_channel_threshold=self.config.trophic.animal_channel_threshold,
            hunter_vulnerability_weight=(
                self.config.biotic_fields.hunter_vulnerability_weight
            ),
            season=season,
            grid=self.grid,
            agents=self.agents,
            movement_actions=tuple(self._movement_actions()),
            tile_memo=DerivedTileMemo(
                season=season,
                climate_state=self._climate_state(),
                effective_fields_for=self._effective_tile_fields,
                water_reason_for=self._water_access_reason,
                soft_refuge_reason_for=self._soft_refuge_reason,
                refuge_score_for=self._refuge_score,
                hazard_for=self._hazard_at,
                current_biotic_state_for=self._current_biotic_state,
            ),
            random_choice=self.rng.choice,
            set_policy_action_source=set_policy_action_source,
            profile_for=self._trophic_profile,
            trophic_role=self._trophic_role,
            energy_ratio=self._energy_ratio,
            hydration_ratio=self._hydration_ratio,
            health_ratio=self._health_ratio,
            plant_intake_useful=self._plant_intake_useful,
            plant_food_value=self._plant_food_value,
            can_consume_fresh_kill=self._can_consume_fresh_kill,
            fresh_kill_intake_useful=self._fresh_kill_intake_useful,
            fresh_kill_food_value=self._fresh_kill_food_value,
            can_consume_carcass=self._can_consume_carcass,
            carcass_intake_useful=self._carcass_intake_useful,
            carcass_food_value=self._carcass_food_value,
            has_water_access=self._has_water_access,
            can_attack=self._can_attack,
            attack_value=self._attack_value,
            prey_vulnerability=self._prey_vulnerability,
            biotic_field_score=self._biotic_field_score,
            can_move_to=self._can_move_to,
            in_bounds=self._in_bounds,
            water_access_reason=self._water_access_reason,
            refuge_score=self._refuge_score,
            hazard_at=self._hazard_at,
            soft_refuge_reason=self._soft_refuge_reason,
            terrain_preference_score=self._terrain_preference_score,
            field_preference_score=self._field_preference_score,
        )

    def _decision_context(
        self,
        agent: Agent,
        *,
        profile: TrophicProfile | None = None,
        season: str | None = None,
        context: DecisionContext | None = None,
    ) -> DecisionContext:
        if context is not None:
            return context
        return runtime_actions.build_decision_context(
            self,
            agent,
            profile=profile,
            season=season,
            scoring_context=self._action_scoring_context(agent),
        )

    def _action_mask_context(self, agent: Agent) -> runtime_action_space.ActionMaskContext:
        profile = self._trophic_profile(agent)
        tile = self.grid[agent.y][agent.x]

        plant_intake_useful = self._plant_intake_useful(agent)
        plant_food_value = (
            self._plant_food_value(agent, tile, profile)
            if plant_intake_useful
            else 0.0
        )
        can_consume_fresh_kill = self._can_consume_fresh_kill(agent)
        fresh_kill_intake_useful = (
            self._fresh_kill_intake_useful(agent)
            if can_consume_fresh_kill
            else False
        )
        fresh_kill_food_value = (
            self._fresh_kill_food_value(agent, tile, profile)
            if can_consume_fresh_kill and fresh_kill_intake_useful
            else 0.0
        )
        can_consume_carcass = self._can_consume_carcass(agent)
        carcass_intake_useful = (
            self._carcass_intake_useful(agent)
            if can_consume_carcass
            else False
        )
        carcass_food_value = (
            self._carcass_food_value(agent, tile, profile)
            if can_consume_carcass and carcass_intake_useful
            else 0.0
        )
        adjacent_carcass_available = (
            self._adjacent_scavenger_carcass_target(agent, profile) is not None
            if can_consume_carcass and carcass_intake_useful
            else False
        )
        can_attack = self._can_attack(agent)
        movement_options: list[runtime_action_space.MovementActionAvailability] = []
        for action, dx, dy in self._movement_actions():
            x = agent.x + dx
            y = agent.y + dy
            target_id = self.grid[y][x].occupant_id if self._in_bounds(x, y) else None
            target = self.agents.get(target_id) if target_id is not None else None
            movement_options.append(
                runtime_action_space.MovementActionAvailability(
                    action=action,
                    dx=dx,
                    dy=dy,
                    can_move=self._can_move_to(x, y),
                    can_attack=(
                        can_attack
                        and target_id is not None
                        and target_id != agent.agent_id
                        and target is not None
                        and target.alive
                    ),
                )
            )
        action_names = runtime_action_space.action_names_for_config(self)
        signal_context = self._signal_runtime_context()
        return runtime_action_space.ActionMaskContext(
            action_names=action_names,
            can_eat=runtime_action_space.can_eat_from_values(
                plant_intake_useful=plant_intake_useful,
                plant_food_value=plant_food_value,
                can_consume_fresh_kill=can_consume_fresh_kill,
                fresh_kill_intake_useful=fresh_kill_intake_useful,
                fresh_kill_food_value=fresh_kill_food_value,
                can_consume_carcass=can_consume_carcass,
                carcass_intake_useful=carcass_intake_useful,
                carcass_food_value=carcass_food_value,
                adjacent_carcass_available=adjacent_carcass_available,
            ),
            can_drink=self._has_water_access(agent),
            movement=tuple(movement_options),
            communication_action_available={
                action: runtime_signals.communication_signal_action_available(
                    agent,
                    action,
                    context=signal_context,
                )
                for action in action_names
                if action.startswith("signal_")
            },
        )
    def _action_resolution_context(
        self,
        agent: Agent,
    ) -> runtime_actions.ActionResolutionContext:
        def eat_action_outcome() -> dict[str, object] | None:
            return self._eat_action_outcome(agent)

        def drink_action_outcome() -> dict[str, object] | None:
            return self._drink_action_outcome(agent)

        def signal_action_outcome(action: str) -> dict[str, object]:
            return runtime_signals.emit_communication_signal_action(
                agent,
                action,
                context=self._signal_runtime_context(),
            )

        def attack_action_outcome(
            action: str,
            dx: int,
            dy: int,
        ) -> tuple[dict[str, object], dict[str, object] | None]:
            return self._attack_action_outcome(agent, agent.x + dx, agent.y + dy)

        def move_action(
            action: str,
            dx: int,
            dy: int,
        ) -> tuple[bool, dict[str, object] | None]:
            nx = agent.x + dx
            ny = agent.y + dy
            if not self._can_move_to(nx, ny):
                return False, None
            from_x = agent.x
            from_y = agent.y
            self.grid[agent.y][agent.x].occupant_id = None
            agent.x = nx
            agent.y = ny
            self.grid[agent.y][agent.x].occupant_id = agent.agent_id
            movement = {
                "moved": True,
                "from_x": from_x,
                "from_y": from_y,
                "to_x": agent.x,
                "to_y": agent.y,
            }
            self._emit(
                EventType.AGENT_MOVED,
                agent_id=agent.agent_id,
                data={"x": agent.x, "y": agent.y, "action": action},
            )
            return True, movement

        return runtime_actions.ActionResolutionContext(
            movement_actions=tuple(self._movement_actions()),
            eat_action_outcome=eat_action_outcome,
            drink_action_outcome=drink_action_outcome,
            signal_action_outcome=signal_action_outcome,
            attack_action_outcome=attack_action_outcome,
            move_action=move_action,
        )
    def _action_mask(self, agent: Agent) -> dict[str, bool]:
        self._record_runtime_cost("action_mask_builds")
        return runtime_action_space.build_action_mask(
            self._action_mask_context(agent)
        )
    def _observation_context(
        self,
        agent: Agent,
    ) -> runtime_observations.ObservationContext:
        width = self.config.width
        height = self.config.height
        water_reasons: dict[tuple[int, int], str] = {}
        hydrology_support_codes: dict[tuple[int, int], int] = {}
        refuge_scores: dict[tuple[int, int], float] = {}
        hazards: dict[tuple[int, int], tuple[str, float]] = {}
        ecology_states: dict[tuple[int, int], str] = {}
        profile_for = self._trophic_profile
        water_access_reason_for = self._water_access_reason
        hydrology_support_code_for = self._hydrology_support_code
        refuge_score_for = self._refuge_score
        hazard_at_for = self._hazard_at
        ecology_state_at_for = self._ecology_state_at
        profile = self._trophic_profile(agent)
        reproduction_ready = self._is_reproduction_ready(agent)
        matched_diet_ratio = self._matched_diet_ratio(agent, profile)
        self._record_runtime_cost("action_mask_builds")

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < width and 0 <= y < height

        def energy_ratio(candidate: Agent) -> float:
            return candidate.energy / max(candidate.genome.max_energy, 1e-9)

        def hydration_ratio(candidate: Agent) -> float:
            return candidate.hydration / max(candidate.genome.max_hydration, 1e-9)

        def health_ratio(candidate: Agent) -> float:
            return candidate.health / max(candidate.max_health, 1e-9)

        def prey_vulnerability(candidate: Agent) -> float:
            return (
                1.0
                + candidate.injury_load * 0.9
                + (1.0 - energy_ratio(candidate)) * 0.55
                + (1.0 - hydration_ratio(candidate)) * 0.45
                + (1.0 - health_ratio(candidate)) * 0.7
            )

        def water_access_reason(x: int, y: int) -> str:
            if not in_bounds(x, y):
                return "none"
            key = (x, y)
            if key not in water_reasons:
                water_reasons[key] = water_access_reason_for(x, y)
            return water_reasons[key]

        def hydrology_support_code(x: int, y: int) -> int:
            if not in_bounds(x, y):
                return -1
            key = (x, y)
            if key not in hydrology_support_codes:
                hydrology_support_codes[key] = hydrology_support_code_for(x, y)
            return hydrology_support_codes[key]

        def refuge_score(x: int, y: int) -> float:
            if not in_bounds(x, y):
                return 0.0
            key = (x, y)
            if key not in refuge_scores:
                refuge_scores[key] = refuge_score_for(x, y)
            return refuge_scores[key]

        def hazard_at(x: int, y: int) -> tuple[str, float]:
            if not in_bounds(x, y):
                return ("none", 0.0)
            key = (x, y)
            if key not in hazards:
                hazards[key] = hazard_at_for(x, y)
            return hazards[key]

        def ecology_state_at(x: int, y: int) -> str:
            if not in_bounds(x, y):
                return "none"
            key = (x, y)
            if key not in ecology_states:
                tile = self.grid[y][x]
                ecology_states[key] = (
                    ecology_state_at_for(x, y) if tile.terrain != "water" else "none"
                )
            return ecology_states[key]

        return runtime_observations.ObservationContext(
            width=width,
            height=height,
            max_age=self.config.max_age,
            grid=self.grid,
            agents=self.agents,
            climate_state=self._climate_state(),
            biotic_state=self._current_biotic_state(),
            signal_state=self._current_signal_state(),
            action_mask=runtime_action_space.build_action_mask(
                self._action_mask_context(agent)
            ),
            movement_actions=tuple(self._movement_actions()),
            profile_for=profile_for,
            energy_ratio=energy_ratio,
            hydration_ratio=hydration_ratio,
            health_ratio=health_ratio,
            is_reproduction_ready=lambda candidate: (
                reproduction_ready if candidate.agent_id == agent.agent_id else False
            ),
            matched_diet_ratio=lambda candidate, candidate_profile: (
                matched_diet_ratio if candidate.agent_id == agent.agent_id else 0.0
            ),
            water_access_reason=water_access_reason,
            hydrology_support_code=hydrology_support_code,
            refuge_score=refuge_score,
            hazard_at=hazard_at,
            ecology_state_at=ecology_state_at,
            in_bounds=in_bounds,
            prey_vulnerability=prey_vulnerability,
        )
    def _observe_agent(self, agent: Agent) -> dict[str, object]:
        self._record_runtime_cost("observation_builds")
        return runtime_observations.build_observation(
            self,
            agent,
            observation_context=self._observation_context(agent),
        )
    def _observation_digest(self, observation: dict[str, object]) -> str:
        return runtime_observations.observation_digest(observation)

    def _trajectory_state_context(self) -> runtime_trajectory.TrajectoryStateContext:
        return runtime_trajectory.TrajectoryStateContext(
            energy_ratio=self._energy_ratio,
            hydration_ratio=self._hydration_ratio,
            health_ratio=self._health_ratio,
        )

    def _begin_trajectory_decision(
        self,
        agent: Agent,
        observation: dict[str, object],
    ) -> dict[str, object]:
        return {
            "agent_id": agent.agent_id,
            "runtime_species_id": self.current_species_map.get(
                agent.agent_id,
                self.agent_last_species_map.get(agent.agent_id),
            ),
            "runtime_ecotype_id": self.current_ecotype_map.get(
                agent.agent_id,
                self.agent_last_ecotype_map.get(agent.agent_id),
            ),
            "observation_metadata": dict(observation["metadata"]),
            "observation_input": runtime_observations.encode_observation_input(observation),
            "observation_digest": self._observation_digest(observation),
            "action_mask": dict(observation["action_mask"]),
            "policy_id": None,
            "policy_version": None,
            "before": runtime_trajectory.capture_agent_state(
                agent,
                context=self._trajectory_state_context(),
            ),
        }
    def _finalize_trajectory_decisions(
        self,
        pending_records: list[dict[str, object]],
    ) -> None:
        records = runtime_trajectory.finalize_trajectory_decision_records(
            self,
            pending_records,
            state_context=self._trajectory_state_context(),
            passive_outcome_for_agent=self._passive_outcome_for_agent,
            is_reproduction_ready=self._is_reproduction_ready,
        )
        for record in records:
            self.tick_trajectory_records.append(record)
            if self.trajectory_sink is not None:
                self.trajectory_sink.write_record(record)
            if self.retain_trajectory_records:
                self.trajectory_records.append(record)

    def _passive_outcome_for_agent(self, agent_id: int, *, acted: bool) -> dict[str, object]:
        damage_events = [
            event for event in self.tick_damage_events if int(event["agent_id"]) == agent_id
        ]
        damage_taken = sum(float(event["amount"]) for event in damage_events)
        attack_damage_taken = sum(
            float(event["amount"]) for event in damage_events if event["source"] == "attack"
        )
        hazard_damage_taken = sum(
            float(event["amount"])
            for event in damage_events
            if str(event["source"]).startswith("hazard_")
        )
        death_event = next(
            (event for event in self.tick_death_events if int(event["agent_id"]) == agent_id),
            None,
        )
        killed = death_event is not None
        return {
            "acted": acted,
            "damage_taken": round(damage_taken, 4),
            "attack_damage_taken": round(attack_damage_taken, 4),
            "hazard_damage_taken": round(hazard_damage_taken, 4),
            "killed": killed,
            "death_cause": death_event["cause"] if death_event is not None else None,
            "killer_id": death_event["killer_id"] if death_event is not None else None,
            "died_before_action": killed and not acted,
            "died_after_action": killed and acted,
        }
    def _best_visible_biotic_action(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
        context: DecisionContext | None = None,
    ) -> str | None:
        decision_context = self._decision_context(
            agent,
            profile=profile,
            context=context,
        )
        return runtime_actions.best_visible_biotic_action(
            self, agent, profile=profile, context=decision_context
        )
    def _best_visible_fresh_kill_action(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
        context: DecisionContext | None = None,
    ) -> str | None:
        decision_context = self._decision_context(
            agent,
            profile=profile,
            context=context,
        )
        return runtime_actions.best_visible_fresh_kill_action(
            self, agent, profile=profile, context=decision_context
        )
    def _best_visible_carrion_action(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
        context: DecisionContext | None = None,
    ) -> str | None:
        decision_context = self._decision_context(
            agent,
            profile=profile,
            context=context,
        )
        return runtime_actions.best_visible_carrion_action(
            self, agent, profile=profile, context=decision_context
        )
    def _attack_damage(
        self,
        attacker: Agent,
        target: Agent,
        profile: TrophicProfile | None = None,
    ) -> float:
        profile = profile or self._trophic_profile(attacker)
        damage = max(
            0.0,
            self.config.combat.base_attack_damage * self._attack_edge(attacker, target, profile),
        )
        if profile.meat_mode in {"hunter", "mixed"}:
            hunter_weight = 1.0 if profile.meat_mode == "hunter" else profile.hunter_share
            damage *= 1.0 + (
                self.config.combat.hunter_mode_attack_damage_multiplier - 1.0
            ) * hunter_weight
            damage *= 1.0 + max(0.0, self._prey_vulnerability(target) - 1.25) * (
                self.config.combat.hunter_wounded_prey_damage_bonus * hunter_weight
            )
        return damage

    def _attack_value(
        self,
        attacker: Agent,
        target: Agent,
        profile: TrophicProfile | None = None,
    ) -> float:
        profile = profile or self._trophic_profile(attacker)
        damage = self._attack_damage(attacker, target, profile)
        if damage <= 0:
            return 0.0
        target_health = max(target.health, 1e-9)
        target_health_ratio = self._health_ratio(target)
        vulnerability = self._prey_vulnerability(target)
        kill_probability = self._clamp01(
            (damage / target_health) ** 0.65 * min(1.35, 0.82 + vulnerability * 0.18)
        )
        projected_meat_gain = self._fresh_kill_nutrition(
            attacker,
            self._project_carcass_yield(target),
            profile,
        ) * min(1.35, 0.82 + vulnerability * 0.22)
        attack_cost_multiplier = attacker.genome.attack_cost_multiplier
        cost_penalty = (
            self.config.combat.attack_energy_cost * attack_cost_multiplier / max(attacker.genome.max_energy, 1e-9)
            + self.config.combat.attack_hydration_cost
            * attack_cost_multiplier
            / max(attacker.genome.max_hydration, 1e-9)
        )
        retaliation_risk = max(
            0.0,
            target.genome.attack_power - attacker.genome.defense_rating * 0.9,
        ) * 0.05
        return (
            projected_meat_gain * kill_probability
            - cost_penalty * 0.42
            - retaliation_risk
        )

    def _best_adjacent_attack_action(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> str | None:
        return runtime_actions.best_adjacent_attack_action(
            self,
            agent,
            profile=profile,
            scoring_context=self._action_scoring_context(agent),
        )
    def _best_visible_prey_action(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
        context: DecisionContext | None = None,
    ) -> str | None:
        decision_context = self._decision_context(
            agent,
            profile=profile,
            context=context,
        )
        return runtime_actions.best_visible_prey_action(
            self, agent, profile=profile, context=decision_context
        )
    def _biotic_opportunity_score(
        self,
        agent: Agent,
        x: int,
        y: int,
        profile: TrophicProfile,
        context: DecisionContext | None = None,
    ) -> float:
        decision_context = self._decision_context(
            agent,
            profile=profile,
            context=context,
        )
        return runtime_actions.biotic_opportunity_score(
            self, agent, x, y, profile, context=decision_context
        )
    def _step_toward_target(
        self,
        agent: Agent,
        target_x: int,
        target_y: int,
        season: str,
        profile: TrophicProfile | None = None,
        include_plant_channel: bool = True,
        include_vegetation_channel: bool = True,
        include_fresh_kill_channel: bool = True,
        include_carcass_channel: bool = True,
        include_animal_signal: bool = True,
        context: DecisionContext | None = None,
    ) -> str | None:
        decision_context = self._decision_context(
            agent,
            profile=profile,
            season=season,
            context=context,
        )
        return runtime_actions.step_toward_target(
            self,
            agent,
            target_x,
            target_y,
            season,
            profile=profile,
            include_plant_channel=include_plant_channel,
            include_vegetation_channel=include_vegetation_channel,
            include_fresh_kill_channel=include_fresh_kill_channel,
            include_carcass_channel=include_carcass_channel,
            include_animal_signal=include_animal_signal,
            context=decision_context,
        )
    def _candidate_tile_score(
        self,
        agent: Agent,
        x: int,
        y: int,
        season: str,
        water_urgency: float,
        food_urgency: float,
        profile: TrophicProfile | None = None,
        include_plant_channel: bool = True,
        include_vegetation_channel: bool = True,
        include_fresh_kill_channel: bool = True,
        include_carcass_channel: bool = True,
        include_animal_signal: bool = True,
        context: DecisionContext | None = None,
    ) -> float:
        return runtime_actions.candidate_tile_score(
            self,
            agent,
            x,
            y,
            season,
            water_urgency,
            food_urgency,
            profile=profile,
            include_plant_channel=include_plant_channel,
            include_vegetation_channel=include_vegetation_channel,
            include_fresh_kill_channel=include_fresh_kill_channel,
            include_carcass_channel=include_carcass_channel,
            include_animal_signal=include_animal_signal,
            context=context,
            scoring_context=self._action_scoring_context(agent),
        )
    def _best_visible_action_toward_need(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
        context: DecisionContext | None = None,
    ) -> str | None:
        decision_context = self._decision_context(
            agent,
            profile=profile,
            context=context,
        )
        return runtime_actions.best_visible_action_toward_need(
            self, agent, profile=profile, context=decision_context
        )
    def _resolve_action(self, agent: Agent, action: str) -> bool:
        """Apply an action request and return whether it moved the agent."""
        return runtime_actions.resolve_action(
            self,
            agent,
            action,
            resolution_action_mask=self._action_mask(agent),
            resolution_context=self._action_resolution_context(agent),
        )

    def _resolve_action_with_outcome(
        self,
        agent: Agent,
        action: str,
        *,
        observation_action_mask: dict[str, bool] | None = None,
        resolution_action_mask: dict[str, bool] | None = None,
    ) -> tuple[bool, dict[str, object]]:
        """Apply an action request and return movement plus the causal outcome."""
        resolved_action_mask = (
            resolution_action_mask
            if resolution_action_mask is not None
            else self._action_mask(agent)
        )
        return runtime_actions.resolve_action_with_outcome(
            self,
            agent,
            action,
            observation_action_mask=observation_action_mask,
            resolution_action_mask=resolved_action_mask,
            resolution_context=self._action_resolution_context(agent),
        )

    def _eat(self, agent: Agent) -> bool:
        self._eat_action_outcome(agent)
        return False

    def _eat_action_outcome(self, agent: Agent) -> dict[str, object] | None:
        tile = self.grid[agent.y][agent.x]
        profile = self._trophic_profile(agent)
        plant_value = self._plant_food_value(agent, tile, profile)
        fresh_kill_value = (
            self._fresh_kill_food_value(agent, tile, profile)
            if self._can_consume_fresh_kill(agent)
            else 0.0
        )
        carcass_value = (
            self._carcass_food_value(agent, tile, profile)
            if self._can_consume_carcass(agent)
            else 0.0
        )
        if (
            profile.meat_mode == "scavenger"
            and carcass_value > 0
            and self._carcass_intake_useful(agent)
        ):
            return self._consume_carcass_outcome(agent, profile=profile)
        if (
            profile.meat_mode == "scavenger"
            and fresh_kill_value > 0
            and self._fresh_kill_intake_useful(agent)
        ):
            return self._consume_fresh_kill_outcome(agent, profile=profile)
        if (
            profile.meat_mode in {"hunter", "mixed"}
            and fresh_kill_value > 0
            and self._fresh_kill_intake_useful(agent)
        ):
            return self._consume_fresh_kill_outcome(agent, profile=profile)
        if (
            profile.meat_mode in {"hunter", "mixed"}
            and carcass_value > 0
            and self._carcass_intake_useful(agent)
        ):
            return self._consume_carcass_outcome(agent, profile=profile)
        if (
            fresh_kill_value > 0
            and self._fresh_kill_intake_useful(agent)
            and fresh_kill_value >= max(plant_value, carcass_value) * 0.92
        ):
            return self._consume_fresh_kill_outcome(agent, profile=profile)
        if (
            carcass_value > 0
            and self._carcass_intake_useful(agent)
            and carcass_value >= plant_value * 0.92
        ):
            return self._consume_carcass_outcome(agent, profile=profile)
        adjacent_carcass_target = self._adjacent_scavenger_carcass_target(agent, profile)
        if adjacent_carcass_target is not None:
            return self._consume_carcass_outcome(
                agent,
                profile=profile,
                source_x=adjacent_carcass_target[0],
                source_y=adjacent_carcass_target[1],
            )
        if plant_value > 0 and self._plant_intake_useful(agent):
            return self._consume_plant_outcome(agent, profile=profile)
        if fresh_kill_value > 0 and self._fresh_kill_intake_useful(agent):
            return self._consume_fresh_kill_outcome(agent, profile=profile)
        if carcass_value > 0 and self._carcass_intake_useful(agent):
            return self._consume_carcass_outcome(agent, profile=profile)
        return None

    def _adjacent_scavenger_carcass_target(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> tuple[int, int] | None:
        profile = profile or self._trophic_profile(agent)
        if profile.meat_mode != "scavenger" or not self._carcass_intake_useful(agent):
            return None
        if (
            self._energy_ratio(agent)
            >= runtime_policy.SCAVENGER_ADJACENT_CARCASS_SURVIVAL_ENERGY_THRESHOLD
            and self._hydration_ratio(agent)
            >= runtime_policy.SCAVENGER_ADJACENT_CARCASS_SURVIVAL_HYDRATION_THRESHOLD
        ):
            return None
        best_target: tuple[int, int] | None = None
        best_value = 0.0
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            x = agent.x + dx
            y = agent.y + dy
            if not self._in_bounds(x, y):
                continue
            tile = self.grid[y][x]
            if tile.terrain == "water" or tile.carcass_energy <= 1e-9:
                continue
            if not self._is_blocked_adjacent_scavenger_carcass(agent, x, y):
                continue
            value = self._carcass_food_value(agent, tile, profile)
            if value > best_value:
                best_value = value
                best_target = (x, y)
        return best_target

    def _is_blocked_adjacent_scavenger_carcass(
        self,
        agent: Agent,
        x: int,
        y: int,
    ) -> bool:
        if abs(x - agent.x) + abs(y - agent.y) != 1:
            return False
        tile = self.grid[y][x]
        if tile.terrain == "water" or tile.carcass_energy <= 1e-9:
            return False
        if tile.occupant_id is not None and tile.occupant_id != agent.agent_id:
            return True
        return not self._can_move_to(x, y)

    def _consume_plant(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> bool:
        self._consume_plant_outcome(agent, profile=profile)
        return False

    def _consume_plant_outcome(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> dict[str, object] | None:
        profile = profile or self._trophic_profile(agent)
        if not self._plant_intake_useful(agent):
            return None
        tile = self.grid[agent.y][agent.x]
        consumed = min(tile.food, self.config.resources.eat_amount)
        if consumed <= 0:
            return None
        potential_gain = self._plant_food_value(
            agent,
            tile,
            profile,
            consumed=consumed,
            x=agent.x,
            y=agent.y,
        )
        energy_before = agent.energy
        tile.food -= consumed
        runtime_resources.record_plant_removed(self, consumed)
        tile.vegetation = self._clamp01(
            tile.vegetation
            - consumed
            * (
                0.16
                if tile.terrain == "forest"
                else 0.2
                if tile.terrain in {"plain", "wetland"}
                else 0.12
            )
        )
        tile.recovery_debt = self._clamp01(
            tile.recovery_debt
            + consumed * (0.08 if tile.terrain == "wetland" else 0.11)
            + max(0.0, 0.28 - tile.vegetation) * 0.03
        )
        tile.shelter = self._clamp01(
            tile.shelter
            - consumed
            * (
                0.12
                if tile.terrain == "forest"
                else 0.08
                if tile.terrain in {"plain", "wetland"}
                else 0.05
            )
            - max(0.0, 0.36 - tile.vegetation) * 0.02
        )
        agent.energy = min(agent.genome.max_energy, agent.energy + potential_gain)
        return runtime_feeding.record_plant_intake(
            self,
            agent,
            profile,
            consumed=consumed,
            energy_before=energy_before,
            energy_after=agent.energy,
            potential_gain=potential_gain,
            tile=tile,
        )

    def _consume_fresh_kill(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
        *,
        immediate_kill_feed: bool = False,
        source_x: int | None = None,
        source_y: int | None = None,
    ) -> bool:
        self._consume_fresh_kill_outcome(
            agent,
            profile=profile,
            immediate_kill_feed=immediate_kill_feed,
            source_x=source_x,
            source_y=source_y,
        )
        return False

    def _consume_fresh_kill_outcome(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
        *,
        immediate_kill_feed: bool = False,
        source_x: int | None = None,
        source_y: int | None = None,
    ) -> dict[str, object] | None:
        profile = profile or self._trophic_profile(agent)
        if not self._fresh_kill_intake_useful(agent):
            return None
        x = agent.x if source_x is None else source_x
        y = agent.y if source_y is None else source_y
        tile = self.grid[y][x]
        consume_info = self._consume_fresh_kill_from_tile(
            tile,
            min(tile.fresh_kill_energy, self.config.resources.eat_amount * 1.1),
        )
        consumed = consume_info["consumed"]
        if consumed <= 0:
            return None
        return self._apply_meat_intake(
            agent,
            food_source="fresh_kill",
            consumed=consumed,
            potential_nutrition=self._fresh_kill_nutrition(agent, consumed, profile),
            profile=profile,
            x=x,
            y=y,
            source_breakdown=consume_info["source_breakdown"],
            deposit_breakdown=consume_info["deposit_breakdown"],
            immediate_kill_feed=immediate_kill_feed,
            freshness=None,
        )

    def _consume_carcass(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
        *,
        source_x: int | None = None,
        source_y: int | None = None,
    ) -> bool:
        self._consume_carcass_outcome(
            agent,
            profile=profile,
            source_x=source_x,
            source_y=source_y,
        )
        return False

    def _consume_carcass_outcome(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
        *,
        source_x: int | None = None,
        source_y: int | None = None,
    ) -> dict[str, object] | None:
        profile = profile or self._trophic_profile(agent)
        if not self._carcass_intake_useful(agent):
            return None
        x = agent.x if source_x is None else source_x
        y = agent.y if source_y is None else source_y
        tile = self.grid[y][x]
        consume_info = self._consume_carcass_from_tile(
            tile,
            min(tile.carcass_energy, self.config.resources.eat_amount * 1.25),
        )
        consumed = consume_info["consumed"]
        if consumed <= 0:
            return None
        freshness = float(consume_info["avg_freshness"])
        return self._apply_meat_intake(
            agent,
            food_source="carcass",
            consumed=consumed,
            potential_nutrition=self._carcass_nutrition(agent, consumed, freshness, profile),
            profile=profile,
            x=x,
            y=y,
            source_breakdown=consume_info["source_breakdown"],
            deposit_breakdown=consume_info["deposit_breakdown"],
            immediate_kill_feed=False,
            freshness=freshness,
        )

    def _apply_meat_intake(
        self,
        agent: Agent,
        food_source: str,
        consumed: float,
        potential_nutrition: float,
        profile: TrophicProfile,
        x: int,
        y: int,
        source_breakdown: list[dict[str, object]],
        deposit_breakdown: list[dict[str, object]],
        immediate_kill_feed: bool,
        freshness: float | None,
    ) -> dict[str, object]:
        return runtime_feeding.apply_meat_intake(
            self,
            agent,
            food_source,
            consumed,
            potential_nutrition,
            profile,
            x,
            y,
            source_breakdown,
            deposit_breakdown,
            immediate_kill_feed,
            freshness,
        )

    def _drink(self, agent: Agent) -> bool:
        self._drink_action_outcome(agent)
        return False

    def _drink_action_outcome(self, agent: Agent) -> dict[str, object] | None:
        water_reason = self._water_access_reason(agent.x, agent.y)
        source_x = agent.x
        source_y = agent.y
        if water_reason == "none":
            adjacent_water = self._adjacent_blocked_water_access_target(agent)
            if adjacent_water is None:
                return None
            source_x, source_y, water_reason = adjacent_water
        hydration_before = agent.hydration
        gained = self.config.resources.drink_amount * agent.genome.water_efficiency
        agent.hydration = min(agent.genome.max_hydration, agent.hydration + gained)
        actual_gain = agent.hydration - hydration_before
        self._emit(
            EventType.AGENT_DRANK,
            agent_id=agent.agent_id,
            data={
                "hydration": round(agent.hydration, 4),
                "water_access_reason": water_reason,
                "source_x": source_x,
                "source_y": source_y,
            },
        )
        return {
            "drank": True,
            "x": agent.x,
            "y": agent.y,
            "source_x": source_x,
            "source_y": source_y,
            "water_access_reason": water_reason,
            "hydration_before": round(hydration_before, 4),
            "hydration_after": round(agent.hydration, 4),
            "gained_hydration": round(actual_gain, 4),
        }

    def _attack(self, attacker: Agent, target_x: int, target_y: int) -> bool:
        self._attack_action_outcome(attacker, target_x, target_y)
        return False

    def _attack_action_outcome(
        self,
        attacker: Agent,
        target_x: int,
        target_y: int,
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        fallback = {
            "attempted": False,
            "target_id": None,
            "target_x": target_x,
            "target_y": target_y,
            "damage": 0.0,
            "success": False,
            "kill": False,
            "immediate_kill_feed": False,
        }
        if not self._can_attack(attacker) or not self._in_bounds(target_x, target_y):
            return fallback, None
        target_id = self.grid[target_y][target_x].occupant_id
        if target_id is None or target_id == attacker.agent_id:
            return fallback, None
        target = self.agents.get(target_id)
        if target is None or not target.alive:
            return fallback, None

        profile = self._trophic_profile(attacker)
        attack_cost_multiplier = attacker.genome.attack_cost_multiplier
        attack_energy_cost = (
            self.config.combat.attack_energy_cost * attack_cost_multiplier
        )
        attacker.energy -= attack_energy_cost
        attacker.hydration -= self.config.combat.attack_hydration_cost * attack_cost_multiplier
        runtime_resources.record_energy_spent(self, "attack", attack_energy_cost)

        damage = self._attack_damage(attacker, target, profile)
        success = damage >= 0.025
        kill = False
        if success:
            runtime_lifecycle.apply_damage(
                self,
                target,
                damage,
                source="attack",
                lifecycle_context=self._lifecycle_context(),
                attacker_id=attacker.agent_id,
            )
            kill = target.health <= 0 and target.alive
            if kill:
                self._kill_agent(target, cause="attack", killer_id=attacker.agent_id)
        if self.record_tick_details:
            self.tick_attack_events.append(
                {
                    "attacker_id": attacker.agent_id,
                    "target_id": target.agent_id,
                    "damage": round(damage, 4),
                    "success": success,
                    "kill": kill,
                }
            )
        self.run_combat_totals["attack_attempts"] += 1
        if success:
            self.run_combat_totals["successful_attacks"] += 1
            self.run_combat_totals["damage_dealt"] += damage
            self.run_combat_totals["attack_damage_taken"] += damage
        if kill:
            self.run_combat_totals["kills"] += 1
            immediate_feed: dict[str, object] | None = None
            if self._can_consume_fresh_kill(attacker) and self._fresh_kill_intake_useful(attacker):
                immediate_feed = self._consume_fresh_kill_outcome(
                    attacker,
                    profile=profile,
                    immediate_kill_feed=True,
                    source_x=target_x,
                    source_y=target_y,
                )
        else:
            immediate_feed = None
        self._emit(
            EventType.AGENT_ATTACKED,
            agent_id=attacker.agent_id,
            data={
                "target_id": target.agent_id,
                "damage": round(damage, 4),
                "success": success,
                "kill": kill,
            },
        )
        return (
            {
                "attempted": True,
                "target_id": target.agent_id,
                "target_x": target_x,
                "target_y": target_y,
                "damage": round(damage, 4),
                "success": success,
                "kill": kill,
                "immediate_kill_feed": immediate_feed is not None,
            },
            immediate_feed,
        )

    def _terrain_defense_multiplier(self, x: int, y: int) -> float:
        tile = self.grid[y][x]
        habitat_state = self._habitat_state_at(x, y)
        multiplier = 1.0 + tile.shelter * 0.24
        if tile.terrain == "forest":
            multiplier += 0.1
        elif tile.terrain == "rocky":
            multiplier += 0.08
        if habitat_state == "flooded":
            multiplier -= 0.1
        elif habitat_state == "parched":
            multiplier -= 0.04
        return max(0.68, multiplier)

    def _attack_edge(
        self,
        attacker: Agent,
        target: Agent,
        profile: TrophicProfile | None = None,
    ) -> float:
        profile = profile or self._trophic_profile(attacker)
        attack_strength = (
            attacker.genome.attack_power
            * (0.72 + self._health_ratio(attacker) * 0.28)
            * (0.72 + self._energy_ratio(attacker) * 0.28)
            * (0.66 + attacker.genome.live_prey_bias * 0.16 + profile.hunter_drive * 0.18)
        )
        defense_strength = (
            target.genome.defense_rating
            * (0.72 + self._health_ratio(target) * 0.28)
            * self._terrain_defense_multiplier(target.x, target.y)
        )
        return max(0.0, attack_strength - defense_strength * 0.48)

    def _apply_metabolism(self, agent: Agent, moved: bool) -> None:
        runtime_lifecycle.apply_metabolism(
            self,
            agent,
            moved,
            lifecycle_context=self._lifecycle_context(),
        )

    def _agent_energy_drain_modifier(self, agent: Agent, season: str | None = None) -> float:
        terrain = self.grid[agent.y][agent.x].terrain
        season_name = season or self._season_state()["name"]
        return self._terrain_energy_modifier(agent, terrain) * self._field_energy_modifier(
            agent.x,
            agent.y,
            season_name,
        )

    def _agent_hydration_drain_modifier(self, agent: Agent, season: str | None = None) -> float:
        terrain = self.grid[agent.y][agent.x].terrain
        season_name = season or self._season_state()["name"]
        return (
            self._terrain_hydration_modifier(agent, terrain)
            * self._seasonal_hydration_modifier(agent, season_name)
            * self._field_hydration_modifier(agent.x, agent.y, season_name)
        )

    def _can_reproduce(self, agent: Agent) -> bool:
        return runtime_reproduction.can_reproduce(self, agent)

    def _reproduction_energy_requirement(
        self,
        agent: Agent,
        profile: TrophicProfile,
    ) -> float:
        return runtime_reproduction.reproduction_energy_requirement(
            self,
            agent,
            profile,
        )

    def _biological_reproduction_block_reasons(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> list[str]:
        return runtime_reproduction.biological_reproduction_block_reasons(
            self,
            agent,
            profile,
        )

    def _is_biologically_reproduction_ready(self, agent: Agent) -> bool:
        return runtime_reproduction.is_biologically_reproduction_ready(self, agent)

    def _reproduction_block_reason(self, agent: Agent) -> str | None:
        return runtime_reproduction.reproduction_block_reason(self, agent)

    def _is_reproduction_ready(self, agent: Agent) -> bool:
        return runtime_reproduction.is_reproduction_ready(self, agent)

    def _record_reproduction_blocked(self, agent: Agent, reason: str) -> None:
        runtime_reproduction.record_reproduction_blocked(self, agent, reason)

    def _reproduction_readiness_counts(self, alive: list[Agent]) -> dict[str, object]:
        return runtime_reproduction.reproduction_readiness_counts(
            self,
            alive,
            trophic_role_codes=TROPHIC_ROLE_CODES,
            meat_mode_codes=MEAT_MODE_CODES,
        )

    def _population_trophic_counts(
        self,
        alive: list[Agent],
    ) -> tuple[dict[str, int], dict[str, int]]:
        trophic_role_counts = {role: 0 for role in TROPHIC_ROLE_CODES if role != "none"}
        meat_mode_counts = {mode: 0 for mode in MEAT_MODE_CODES}
        for agent in alive:
            trophic_role_counts[self._trophic_role(agent)] += 1
            meat_mode_counts[self._meat_mode(agent)] += 1
        return trophic_role_counts, meat_mode_counts

    def _trophic_lifecycle_summary_context(
        self,
    ) -> runtime_lifecycle_summary.TrophicLifecycleSummaryContext:
        return runtime_lifecycle_summary.TrophicLifecycleSummaryContext(
            agents=self.agents,
            run_death_cause_counts=self.run_death_cause_counts,
            run_death_causes_by_trophic_role=self.run_death_causes_by_trophic_role,
            run_death_causes_by_meat_mode=self.run_death_causes_by_meat_mode,
            trophic_role=self._trophic_role,
            meat_mode=self._meat_mode,
            death_cause=lambda agent: runtime_lifecycle.death_cause(self, agent),
        )

    def _trophic_lifecycle_summary(self, *, ticks_executed: int) -> dict[str, object]:
        return runtime_lifecycle_summary.build_trophic_lifecycle_summary(
            context=self._trophic_lifecycle_summary_context(),
            ticks_executed=ticks_executed,
            trophic_role_codes=TROPHIC_ROLE_CODES,
            meat_mode_codes=MEAT_MODE_CODES,
        )

    def _animal_mode_stabilized_child_genome(
        self,
        parent_genome: Genome,
        child_genome: Genome,
        parent_profile: TrophicProfile,
    ) -> Genome:
        return runtime_reproduction.animal_mode_stabilized_child_genome(
            self,
            parent_genome,
            child_genome,
            parent_profile,
        )

    def _child_starting_fraction(
        self,
        base_fraction: float,
        multiplier: float,
        parent_profile: TrophicProfile,
    ) -> float:
        return runtime_reproduction.child_starting_fraction(
            base_fraction,
            multiplier,
            parent_profile,
        )

    def _reproduction_energy_cost(
        self,
        parent_profile: TrophicProfile,
    ) -> float:
        return runtime_reproduction.reproduction_energy_cost(self, parent_profile)

    def _sexual_reproduction_energy_cost(
        self,
        parent_profile: TrophicProfile,
    ) -> float:
        return runtime_reproduction.sexual_reproduction_energy_cost(
            self,
            parent_profile,
        )

    def _sexual_partner_ready(self, agent: Agent) -> bool:
        return runtime_reproduction.sexual_partner_ready(self, agent)

    def _reproduction_placement_context(
        self,
    ) -> runtime_reproduction.ReproductionPlacementContext:
        def sibling_destination_candidates(x: int, y: int) -> list[tuple[int, int]]:
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            self.rng.shuffle(neighbors)
            return neighbors

        return runtime_reproduction.ReproductionPlacementContext(
            max_agents=self.config.max_agents,
            current_alive_count=lambda: len(self.alive_agents()),
            has_empty_neighbor=self._has_empty_neighbor,
            find_empty_neighbor=self._find_empty_neighbor,
            sibling_destination_candidates=sibling_destination_candidates,
            can_place_at=self._can_move_to,
            place_agent=self._place_agent,
            invalidate_spatial_state=self._invalidate_biotic_state,
        )

    def _reproduce(self, parent: Agent) -> bool:
        return runtime_reproduction.reproduce(self, parent)

    def _reproduce_asexual(
        self,
        parent: Agent,
        destination: tuple[int, int],
        parent_profile: TrophicProfile,
    ) -> bool:
        return runtime_reproduction.reproduce_asexual(
            self,
            parent,
            destination,
            parent_profile,
        )

    def _reproduce_sexual(
        self,
        parent: Agent,
        mate_candidate: runtime_mating.MateCandidate,
        destination: tuple[int, int],
        parent_profile: TrophicProfile,
    ) -> bool:
        return runtime_reproduction.reproduce_sexual(
            self,
            parent,
            mate_candidate,
            destination,
            parent_profile,
        )

    def _kill_agent(
        self,
        agent: Agent,
        cause: str | None = None,
        killer_id: int | None = None,
    ) -> None:
        death_cause = cause or runtime_lifecycle.death_cause(self, agent)
        runtime_lifecycle.record_death_cause(
            self,
            agent,
            death_cause,
            lifecycle_context=self._lifecycle_context(),
        )
        agent.alive = False
        agent.death_tick = self.tick
        self.grid[agent.y][agent.x].occupant_id = None
        tile = self.grid[agent.y][agent.x]
        source_species = self._species_id_for_agent(agent.agent_id)
        projected_carcass_energy = (
            self._project_carcass_yield(agent) if tile.terrain != "water" else 0.0
        )
        resource_emission = runtime_resources.emit_death_resources(
            self,
            agent,
            resource_context=self._resource_context(),
            death_cause=death_cause,
            killer_id=killer_id,
            source_species=source_species,
            projected_carcass_energy=projected_carcass_energy,
        )
        self.deaths += 1
        if self.record_tick_details:
            self.tick_death_agent_ids.append(agent.agent_id)
            self.tick_death_events.append(
                {
                    "agent_id": agent.agent_id,
                    "cause": death_cause,
                    "killer_id": killer_id,
                    "x": agent.x,
                    "y": agent.y,
                }
            )
        self._invalidate_biotic_state()
        self._emit(
            EventType.AGENT_DIED,
            agent_id=agent.agent_id,
            data={
                "age": agent.age,
                "energy": round(agent.energy, 4),
                "hydration": round(agent.hydration, 4),
                "health": round(agent.health, 4),
                "cause": death_cause,
                "killer_id": killer_id,
                "x": agent.x,
                "y": agent.y,
                "source_species": source_species,
                **runtime_resources.death_resource_event_fields(resource_emission),
            },
        )

    def _find_empty_neighbor(self, x: int, y: int) -> tuple[int, int] | None:
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        self.rng.shuffle(neighbors)
        for nx, ny in neighbors:
            if self._can_move_to(nx, ny):
                return nx, ny
        return None

    def _has_empty_neighbor(self, x: int, y: int) -> bool:
        return any(
            self._can_move_to(nx, ny)
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        )

    def _can_move_to(self, x: int, y: int) -> bool:
        return (
            self._in_bounds(x, y)
            and self.grid[y][x].terrain != "water"
            and self.grid[y][x].occupant_id is None
        )

    def _has_water_access(self, agent: Agent) -> bool:
        return (
            self._tile_has_water_access(agent.x, agent.y)
            or self._adjacent_blocked_water_access_target(agent) is not None
        )

    def _tile_has_water_access(self, x: int, y: int) -> bool:
        return self._water_access_reason(x, y) != "none"

    def _adjacent_blocked_water_access_target(
        self,
        agent: Agent,
    ) -> tuple[int, int, str] | None:
        profile = self._trophic_profile(agent)
        if profile.meat_mode not in {"hunter", "mixed"}:
            return None
        if (
            self._hydration_ratio(agent)
            >= runtime_policy.ANIMAL_BLOCKED_WATER_SURVIVAL_HYDRATION
        ):
            return None
        best_target: tuple[int, int, str] | None = None
        best_score = float("-inf")
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            x = agent.x + dx
            y = agent.y + dy
            if not self._in_bounds(x, y):
                continue
            tile = self.grid[y][x]
            if tile.terrain == "water":
                continue
            water_reason = self._water_access_reason(x, y)
            if water_reason == "none":
                continue
            if tile.occupant_id is None and self._can_move_to(x, y):
                continue
            _, hazard_level = self._hazard_at(x, y)
            score = 1.0
            score -= hazard_level * 0.25
            if water_reason == "wetland":
                score += 0.05
            if score > best_score:
                best_score = score
                best_target = (x, y, water_reason)
        return best_target

    def _adjacent_to_water(self, x: int, y: int) -> bool:
        return self.static_topology.adjacent_to_water[y][x]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    @staticmethod
    def _movement_actions() -> tuple[tuple[str, int, int], ...]:
        return (
            ("move_north", 0, -1),
            ("move_south", 0, 1),
            ("move_east", 1, 0),
            ("move_west", -1, 0),
        )

    def _terrain_affinity(self, agent: Agent, terrain: str) -> float:
        if terrain == "forest":
            return agent.genome.forest_affinity
        if terrain == "plain":
            return agent.genome.plain_affinity
        if terrain == "wetland":
            return agent.genome.wetland_affinity
        if terrain == "rocky":
            return agent.genome.rocky_affinity
        return 1.0

    def _terrain_food_multiplier(self, agent: Agent, terrain: str) -> float:
        affinity = self._terrain_affinity(agent, terrain)
        return agent.genome.food_efficiency * TERRAIN_FOOD_BASE.get(terrain, 1.0) * (
            0.68 + affinity * 0.32
        )

    def _terrain_preference_score(self, agent: Agent, terrain: str) -> float:
        if terrain == "forest":
            return (agent.genome.forest_affinity - 1.0) * 0.22
        if terrain == "plain":
            return (agent.genome.plain_affinity - 1.0) * 0.22
        if terrain == "wetland":
            return (agent.genome.wetland_affinity - 1.0) * 0.24
        if terrain == "rocky":
            return (agent.genome.rocky_affinity - 1.0) * 0.24
        return 0.0

    def _terrain_energy_modifier(self, agent: Agent, terrain: str) -> float:
        affinity = self._terrain_affinity(agent, terrain)
        terrain_base = TERRAIN_ENERGY_BASE.get(terrain, 1.0)
        return max(0.62, terrain_base * (1.22 - affinity * 0.28))

    def _terrain_hydration_modifier(self, agent: Agent, terrain: str) -> float:
        affinity = self._terrain_affinity(agent, terrain)
        terrain_base = TERRAIN_HYDRATION_BASE.get(terrain, 1.0)
        return max(0.58, terrain_base * (1.18 - affinity * 0.18))

    def _seasonal_hydration_modifier(self, agent: Agent, season: str) -> float:
        climate = self.config.climate
        if season == "dry":
            return max(
                0.85,
                1.0 + climate.seasonal_hydration_shift * (2.0 - agent.genome.heat_tolerance) * 4.0,
            )
        return max(
            0.78,
            1.0 - climate.seasonal_hydration_shift * agent.genome.heat_tolerance * 2.2,
        )

    def _lineage_species_snapshot(
        self,
        alive: list[Agent],
    ) -> tuple[dict[int, int], list[SpeciesRecord]]:
        if not alive:
            for registry in self.species_registry.values():
                registry["current_members"] = 0
            return {}, []

        members_by_species: dict[int, list[SpeciesMember]] = defaultdict(list)
        species_map: dict[int, int] = {}
        for agent in alive:
            species_id = agent.lineage_id
            species_map[agent.agent_id] = species_id
            members_by_species[species_id].append(
                SpeciesMember(
                    agent_id=agent.agent_id,
                    lineage_id=agent.lineage_id,
                    genome=agent.genome,
                )
            )

        species_records: list[SpeciesRecord] = []
        for species_id, members in members_by_species.items():
            centroid = centroid_from_members(members)
            registry = self.species_registry.get(species_id)
            if registry is None:
                first_seen_tick = min(self.agents[member.agent_id].birth_tick for member in members)
                registry = {
                    "label": f"L{species_id:03d}",
                    "first_seen_tick": first_seen_tick,
                    "observed_ticks": 0,
                    "peak_members": 0,
                    "lineages": {species_id},
                    "taxonomy_origin": "lineage",
                    "founder_agent_id": species_id,
                }
                self.species_registry[species_id] = registry

            registry["vector"] = vector_from_centroid(centroid)
            registry["centroid"] = centroid
            registry["last_seen_tick"] = self.tick
            registry["observed_ticks"] += 1
            registry["current_members"] = len(members)
            registry["peak_members"] = max(registry["peak_members"], len(members))
            species_records.append(
                SpeciesRecord(
                    species_id=species_id,
                    label=registry["label"],
                    member_count=len(members),
                    lineages=[species_id],
                    centroid={gene: round(value, 4) for gene, value in centroid.items()},
                )
            )

        for species_id, registry in self.species_registry.items():
            if species_id not in members_by_species:
                registry["current_members"] = 0

        species_records.sort(key=lambda record: (-record.member_count, record.species_id))
        return species_map, species_records

    def _refresh_population_snapshots(
        self,
        *,
        include_species: bool = True,
    ) -> tuple[list[Agent], dict[int, int], list[SpeciesRecord], dict[int, int], list[SpeciesRecord]]:
        alive = sorted(self.alive_agents(), key=lambda agent: agent.agent_id)
        if include_species:
            species_map, species_records = self._lineage_species_snapshot(alive)
        else:
            species_map, species_records = {}, []
        self.current_species_map = species_map
        self.current_species_records = species_records
        ecotype_map, ecotype_records = self._cluster_ecotypes()
        self.current_ecotype_map = ecotype_map
        self.current_ecotype_records = ecotype_records
        return alive, species_map, species_records, ecotype_map, ecotype_records

    def _frame_surface_context(self) -> runtime_surfaces.FrameSurfaceContext:
        return runtime_surfaces.FrameSurfaceContext(
            climate_state=self._climate_state(),
            habitat_snapshot=self._habitat_state_grid(),
            hydrology_snapshot=self._hydrology_snapshot(),
            refuge_snapshot=self._refuge_snapshot(),
            ecology_snapshot=self._ecology_snapshot(),
            hazard_snapshot=self._hazard_snapshot(),
            biotic_field_snapshot=self._biotic_field_snapshot(),
            signal_field_snapshot=self._signal_field_snapshot(),
            fresh_kill_snapshot=self._fresh_kill_snapshot(),
            carcass_snapshot=self._carcass_snapshot(),
            energy_ratio=self._energy_ratio,
            hydration_ratio=self._hydration_ratio,
            health_ratio=self._health_ratio,
            agent_energy_drain_modifier=self._agent_energy_drain_modifier,
            agent_hydration_drain_modifier=self._agent_hydration_drain_modifier,
            is_reproduction_ready=self._is_reproduction_ready,
            trophic_role=self._trophic_role,
            meat_mode=self._meat_mode,
            refuge_score=self._refuge_score,
            matched_diet_ratio=self._matched_diet_ratio,
        )

    def _surface_snapshot_context(
        self,
    ) -> runtime_surface_snapshots.SurfaceSnapshotContext:
        (
            _,
            _,
            hydrology_primary_counts,
            hydrology_support_counts,
            hydrology_primary_stats,
        ) = self._hydrology_snapshot()
        _, _, refuge_counts, refuge_stats = self._refuge_snapshot()
        _, _, hazard_counts, hazard_stats = self._hazard_snapshot()
        _, fresh_kill_stats = self._fresh_kill_snapshot()
        _, _, carcass_stats = self._carcass_snapshot()
        _, biotic_field_stats = self._biotic_field_snapshot()
        _, signal_field_stats = self._signal_field_snapshot()
        _, ecology_counts, ecology_stats = self._ecology_snapshot()
        habitat_counts = self._habitat_state_grid()[1]
        return runtime_surface_snapshots.SurfaceSnapshotContext(
            viewer_frames=self.viewer_frames,
            hydrology_primary_counts=hydrology_primary_counts,
            hydrology_support_counts=hydrology_support_counts,
            hydrology_primary_stats=hydrology_primary_stats,
            refuge_counts=refuge_counts,
            refuge_stats=refuge_stats,
            hazard_counts=hazard_counts,
            hazard_stats=hazard_stats,
            fresh_kill_stats=fresh_kill_stats,
            carcass_stats=carcass_stats,
            biotic_field_stats=biotic_field_stats,
            signal_field_stats=signal_field_stats,
            ecology_counts=ecology_counts,
            ecology_stats=ecology_stats,
            habitat_counts=habitat_counts,
        )

    def _materialize_frame_surfaces(
        self,
        surface_context: runtime_surfaces.FrameSurfaceContext | None = None,
    ) -> dict[str, object]:
        return runtime_surfaces.materialize_frame_surfaces(
            self,
            surface_context=(
                surface_context
                if surface_context is not None
                else self._frame_surface_context()
            ),
        )
    def _build_agent_frame_telemetry(
        self,
        alive: list[Agent],
        *,
        season: str,
        surfaces: dict[str, object],
        surface_context: runtime_surfaces.FrameSurfaceContext | None = None,
    ) -> dict[int, dict[str, object]]:
        return runtime_surfaces.build_agent_frame_telemetry(
            self,
            alive,
            season=season,
            surfaces=surfaces,
            surface_context=surface_context,
        )
    def _capture_frame(self, births_this_tick: int, deaths_this_tick: int) -> None:
        runtime_frames.capture_frame(
            self,
            births_this_tick=births_this_tick,
            deaths_this_tick=deaths_this_tick,
            trophic_role_codes=TROPHIC_ROLE_CODES,
            meat_mode_codes=MEAT_MODE_CODES,
        )

    def _build_species_metrics(
        self,
        alive: list[Agent],
        species_map: dict[int, int],
        previous_species_map: dict[int, int],
        *,
        agent_telemetry: dict[int, dict[str, object]] | None = None,
    ) -> dict[str, dict[str, object]]:
        telemetry = agent_telemetry or {}
        if not telemetry:
            telemetry = self._build_agent_frame_telemetry(
                alive,
                season=self._season_state()["name"],
                surfaces=self._materialize_frame_surfaces(),
            )
        occupancy_samples = (
            SpeciesMetricSample(
                species_id=species_map.get(agent.agent_id, 0),
                terrain=self.grid[agent.y][agent.x].terrain,
                habitat=str(telemetry[agent.agent_id]["habitat_state"]),
                ecology=str(telemetry[agent.agent_id]["ecology_state"]),
                hazard=str(telemetry[agent.agent_id]["hazard_type"]),
                trophic_role=str(telemetry[agent.agent_id]["trophic_role"]),
                meat_mode=str(telemetry[agent.agent_id]["meat_mode"]),
                water_reason=str(telemetry[agent.agent_id]["water_reason"]),
                has_water_access=bool(telemetry[agent.agent_id]["has_water_access"]),
                shoreline_support=bool(telemetry[agent.agent_id]["shoreline_support"]),
                wetland_support=bool(telemetry[agent.agent_id]["wetland_support"]),
                flooded_support=bool(telemetry[agent.agent_id]["flooded_support"]),
                refuge_exposed=str(telemetry[agent.agent_id]["soft_refuge_reason"]) != "none",
                energy_ratio=float(telemetry[agent.agent_id]["energy_ratio"]),
                hydration_ratio=float(telemetry[agent.agent_id]["hydration_ratio"]),
                health_ratio=float(telemetry[agent.agent_id]["health_ratio"]),
                matched_diet_ratio=float(telemetry[agent.agent_id]["matched_diet_ratio"]),
                age=float(agent.age),
                vegetation=float(self.grid[agent.y][agent.x].vegetation),
                recovery_debt=float(self.grid[agent.y][agent.x].recovery_debt),
                refuge_score=float(telemetry[agent.agent_id]["refuge_score"]),
                injury_load=float(agent.injury_load),
                reproduction_ready=bool(telemetry[agent.agent_id]["reproduction_ready"]),
            )
            for agent in alive
        )
        return build_shared_species_metrics(
            occupancy_samples=occupancy_samples,
            species_map=species_map,
            previous_species_map=previous_species_map,
            birth_pairs=self.tick_birth_pairs,
            dead_agent_ids=self.tick_death_agent_ids,
            attack_records=(
                (
                    int(event["attacker_id"]),
                    bool(event["success"]),
                    float(event["damage"]),
                    bool(event["kill"]),
                )
                for event in self.tick_attack_events
            ),
            damage_records=(
                (
                    int(event["agent_id"]),
                    float(event["amount"]),
                    str(event["source"]),
                )
                for event in self.tick_damage_events
            ),
            carcass_deposit_records=(
                (
                    int(event.get("source_species") or 0),
                    float(event["deposited_energy"]),
                )
                for event in self.tick_carcass_deposit_events
            ),
            fresh_kill_deposit_records=(
                (
                    int(event.get("source_species") or 0),
                    float(event["deposited_energy"]),
                )
                for event in self.tick_fresh_kill_deposit_events
            ),
            fresh_kill_consumption_records=(
                (
                    int(event["agent_id"]),
                    float(event["energy"]),
                    float(event["gained_energy"]),
                )
                for event in self.tick_fresh_kill_events
            ),
            carcass_consumption_records=(
                (
                    int(event["agent_id"]),
                    float(event["energy"]),
                    float(event["gained_energy"]),
                )
                for event in self.tick_carcass_events
            ),
            feeding_records=(
                (
                    int(event["agent_id"]),
                    str(event["food_source"]),
                    float(event["gained_energy"]),
                )
                for event in self.tick_feeding_events
            ),
        )

    @staticmethod
    def _empty_terrain_occupancy() -> dict[str, int]:
        return empty_terrain_occupancy()

    @staticmethod
    def _empty_hydrology_exposure_counts() -> dict[str, int]:
        return empty_hydrology_exposure_counts()

    def _trait_means(self, agents: list[Agent]) -> dict[str, float]:
        genomes = [agent.genome for agent in agents]
        if not genomes:
            return {
                "avg_max_energy": 0.0,
                "avg_max_health": 0.0,
                "avg_move_cost": 0.0,
                "avg_heat_tolerance": 0.0,
                "avg_food_efficiency": 0.0,
                "avg_water_efficiency": 0.0,
                "avg_attack_power": 0.0,
                "avg_meat_efficiency": 0.0,
                "avg_carrion_bias": 0.0,
                "avg_live_prey_bias": 0.0,
                "avg_wetland_affinity": 0.0,
                "avg_rocky_affinity": 0.0,
            }
        count = len(genomes)
        return {
            "avg_max_energy": round(sum(genome.max_energy for genome in genomes) / count, 4),
            "avg_max_health": round(sum(genome.max_health for genome in genomes) / count, 4),
            "avg_move_cost": round(sum(genome.move_cost for genome in genomes) / count, 4),
            "avg_heat_tolerance": round(
                sum(genome.heat_tolerance for genome in genomes) / count, 4
            ),
            "avg_food_efficiency": round(
                sum(genome.food_efficiency for genome in genomes) / count,
                4,
            ),
            "avg_water_efficiency": round(
                sum(genome.water_efficiency for genome in genomes) / count,
                4,
            ),
            "avg_attack_power": round(sum(genome.attack_power for genome in genomes) / count, 4),
            "avg_meat_efficiency": round(
                sum(genome.meat_efficiency for genome in genomes) / count,
                4,
            ),
            "avg_carrion_bias": round(sum(genome.carrion_bias for genome in genomes) / count, 4),
            "avg_live_prey_bias": round(
                sum(genome.live_prey_bias for genome in genomes) / count,
                4,
            ),
            "avg_wetland_affinity": round(
                sum(genome.wetland_affinity for genome in genomes) / count,
                4,
            ),
            "avg_rocky_affinity": round(
                sum(genome.rocky_affinity for genome in genomes) / count,
                4,
            ),
        }

    def _cluster_ecotypes(self) -> tuple[dict[int, int], list[SpeciesRecord]]:
        alive = sorted(self.alive_agents(), key=lambda agent: agent.agent_id)
        if not alive:
            return {}, []

        threshold = self.config.species_distance_threshold
        provisional_clusters: list[dict[str, object]] = []
        for agent in alive:
            member = SpeciesMember(
                agent_id=agent.agent_id,
                lineage_id=agent.lineage_id,
                genome=agent.genome,
            )
            vector = agent.genome_vector
            best_cluster_index: int | None = None
            best_distance = float("inf")
            for index, cluster in enumerate(provisional_clusters):
                distance = euclidean_distance(vector, cluster["vector"])
                if distance < best_distance:
                    best_distance = distance
                    best_cluster_index = index

            if best_cluster_index is None or best_distance > threshold:
                provisional_clusters.append(
                    {
                        "members": [member],
                        "vector": vector,
                        "vector_sum": list(vector),
                        "count": 1,
                    }
                )
                continue

            cluster = provisional_clusters[best_cluster_index]
            cluster["members"].append(member)
            cluster["count"] += 1
            cluster["vector_sum"] = [
                total + component
                for total, component in zip(cluster["vector_sum"], vector, strict=True)
            ]
            cluster["vector"] = tuple(
                total / cluster["count"] for total in cluster["vector_sum"]
            )

        species_map: dict[int, int] = {}
        species_records: list[SpeciesRecord] = []
        assigned_existing: set[int] = set()

        for cluster in sorted(
            provisional_clusters,
            key=lambda item: (min(member.agent_id for member in item["members"]),),
        ):
            members = cluster["members"]
            centroid = centroid_from_members(members)
            centroid_vector = vector_from_centroid(centroid)
            species_id = self._match_or_create_ecotype(centroid_vector, assigned_existing)
            assigned_existing.add(species_id)
            self._update_ecotype_registry(species_id, centroid, members)

            for member in members:
                species_map[member.agent_id] = species_id

            species_records.append(
                SpeciesRecord(
                    species_id=species_id,
                    label=self.ecotype_registry[species_id]["label"],
                    member_count=len(members),
                    lineages=sorted({member.lineage_id for member in members}),
                    centroid={gene: round(value, 4) for gene, value in centroid.items()},
                )
            )

        for species_id, registry in self.ecotype_registry.items():
            if species_id not in assigned_existing:
                registry["current_members"] = 0

        species_records.sort(key=lambda record: (-record.member_count, record.species_id))
        return species_map, species_records

    def _match_or_create_ecotype(
        self,
        centroid_vector: tuple[float, ...],
        assigned_existing: set[int],
    ) -> int:
        best_species_id: int | None = None
        best_distance = float("inf")
        for species_id, registry in sorted(self.ecotype_registry.items()):
            if species_id in assigned_existing:
                continue
            distance = euclidean_distance(centroid_vector, registry["vector"])
            if distance < best_distance:
                best_distance = distance
                best_species_id = species_id

        if best_species_id is not None and best_distance <= self.config.species_distance_threshold * 1.2:
            return best_species_id

        species_id = self.next_ecotype_id
        self.next_ecotype_id += 1
        return species_id

    def _update_ecotype_registry(
        self,
        species_id: int,
        centroid: dict[str, float],
        members: list[SpeciesMember],
    ) -> None:
        registry = self.ecotype_registry.get(species_id)
        if registry is None:
            registry = {
                "label": f"E{species_id:03d}",
                "first_seen_tick": self.tick,
                "observed_ticks": 0,
                "peak_members": 0,
                "lineages": set(),
            }
            self.ecotype_registry[species_id] = registry

        registry["vector"] = vector_from_centroid(centroid)
        registry["centroid"] = centroid
        registry["last_seen_tick"] = self.tick
        registry["observed_ticks"] += 1
        registry["current_members"] = len(members)
        registry["peak_members"] = max(registry["peak_members"], len(members))
        registry["lineages"].update(member.lineage_id for member in members)

    def _build_viewer_payload(self) -> dict[str, object]:
        terrain_counts = self._terrain_counts()
        return {
            "map": {
                "width": self.config.width,
                "height": self.config.height,
                "terrain_codes": [
                    [TERRAIN_CODES[tile.terrain] for tile in row] for row in self.grid
                ],
                "terrain_legend": {
                    str(code): terrain for terrain, code in sorted(TERRAIN_CODES.items(), key=lambda item: item[1])
                },
                "hydrology_primary_legend": {
                    str(code): reason
                    for reason, code in sorted(HYDROLOGY_REASON_CODES.items(), key=lambda item: item[1])
                },
                "hydrology_support_bits": {
                    str(flag): name
                    for name, flag in sorted(
                        HYDROLOGY_SUPPORT_FLAGS.items(),
                        key=lambda item: item[1],
                    )
                },
                "refuge_legend": {
                    str(code): reason
                    for reason, code in sorted(SOFT_REFUGE_CODES.items(), key=lambda item: item[1])
                },
                "hazard_legend": {
                    str(code): hazard_type
                    for hazard_type, code in sorted(HAZARD_TYPE_CODES.items(), key=lambda item: item[1])
                },
                "trophic_role_legend": {
                    str(code): role
                    for role, code in sorted(TROPHIC_ROLE_CODES.items(), key=lambda item: item[1])
                },
                "meat_mode_legend": {
                    str(code): mode
                    for mode, code in sorted(MEAT_MODE_CODES.items(), key=lambda item: item[1])
                },
                "ecology_legend": {
                    str(code): state
                    for state, code in sorted(ECOLOGY_STATE_CODES.items(), key=lambda item: item[1])
                },
                "terrain_counts": terrain_counts,
                "environment_fields": self.environment_fields.to_serializable(),
                "base_tile_fields": {
                    "fertility": [[round(tile.fertility, 4) for tile in row] for row in self.grid],
                    "moisture": [[round(tile.moisture, 4) for tile in row] for row in self.grid],
                    "heat": [[round(tile.heat, 4) for tile in row] for row in self.grid],
                    "shelter": [[round(tile.shelter, 4) for tile in row] for row in self.grid],
                },
            },
            "agent_catalog": {
                str(agent.agent_id): {
                    "agent_id": agent.agent_id,
                    "parent_id": agent.parent_id,
                    "secondary_parent_id": agent.secondary_parent_id,
                    "parent_ids": [
                        parent_id
                        for parent_id in (agent.parent_id, agent.secondary_parent_id)
                        if parent_id is not None
                    ],
                    "lineage_id": agent.lineage_id,
                    "reproductive_group_id": agent.reproductive_group_id,
                    "reproductive_stage": agent.reproductive_stage,
                    "reproductive_expression": agent.reproductive_expression,
                    "birth_tick": agent.birth_tick,
                    "death_tick": agent.death_tick,
                    "genome": agent.genome.to_dict(),
                    "mind_inheritance": dict(agent.mind_inheritance_metadata),
                }
                for agent in sorted(self.agents.values(), key=lambda item: item.agent_id)
            },
            "reproductive_group_catalog": runtime_reproduction.build_reproductive_group_catalog(
                self.reproductive_groups,
                self.agents.values(),
            ),
            "taxonomy": {
                "species_identity": "lineage",
                "ecotype_identity": "frame_local_genome_cluster",
            },
            "species_catalog": {
                str(species_id): {
                    "species_id": species_id,
                    "label": registry["label"],
                    "first_seen_tick": registry["first_seen_tick"],
                    "last_seen_tick": registry.get("last_seen_tick", registry["first_seen_tick"]),
                    "observed_ticks": registry["observed_ticks"],
                    "peak_members": registry["peak_members"],
                    "lineages": sorted(registry["lineages"]),
                    "taxonomy_origin": registry.get("taxonomy_origin", "lineage"),
                    "identity_mode": "runtime_lineage",
                    "founder_agent_id": registry.get("founder_agent_id"),
                    "centroid_units": "raw_gene_values",
                    "centroid": {
                        gene: round(value, 4)
                        for gene, value in registry.get("centroid", {}).items()
                    },
                    "normalized_centroid_units": "unit_interval_by_gene_limits",
                    "normalized_centroid": {
                        gene: round(value, 4)
                        for gene, value in zip(
                            GENE_ORDER,
                            registry.get("vector", ()),
                            strict=False,
                        )
                    },
                }
                for species_id, registry in sorted(self.species_registry.items())
            },
            "ecotype_catalog": {
                str(ecotype_id): {
                    "ecotype_id": ecotype_id,
                    "label": registry["label"],
                    "first_seen_tick": registry["first_seen_tick"],
                    "last_seen_tick": registry.get("last_seen_tick", registry["first_seen_tick"]),
                    "observed_ticks": registry["observed_ticks"],
                    "peak_members": registry["peak_members"],
                    "lineages": sorted(registry["lineages"]),
                    "identity_mode": "frame_local_genome_cluster",
                    "centroid_units": "raw_gene_values",
                    "centroid": {
                        gene: round(value, 4)
                        for gene, value in registry.get("centroid", {}).items()
                    },
                    "normalized_centroid_units": "unit_interval_by_gene_limits",
                    "normalized_centroid": {
                        gene: round(value, 4)
                        for gene, value in zip(
                            GENE_ORDER,
                            registry.get("vector", ()),
                            strict=False,
                        )
                    },
                }
                for ecotype_id, registry in sorted(self.ecotype_registry.items())
            },
            "frames": self.viewer_frames,
            "analytics": self._build_analytics(),
            "trajectory": self._build_trajectory_payload(),
            "agent_encoding": list(VIEWER_AGENT_ENCODING),
        }

    def _build_analytics(self) -> dict[str, object]:
        return build_replay_analytics(
            frames=self.viewer_frames,
            species_ids=sorted(self.species_registry),
        )
    def _build_trajectory_payload(self) -> dict[str, object]:
        return runtime_trajectory.build_trajectory_payload(
            self.trajectory_records,
            signal_config=self.config.signals,
        )

    def _build_collapse_events(
        self,
        ticks: list[int],
        species_population: dict[str, list[int]],
    ) -> list[dict[str, object]]:
        return build_collapse_events(ticks, species_population)

    def _build_summary(self, mode: RunMode = RunMode.FULL_REPLAY) -> dict[str, object]:
        return runtime_summary.build_summary(
            self,
            mode=mode,
            summary_context=self._summary_context(mode=mode),
        )

    def _summary_context(self, *, mode: RunMode) -> runtime_summary.SummaryContext:
        alive = self.alive_agents()
        ticks_executed = self.tick + 1
        trophic_role_counts, meat_mode_counts = self._population_trophic_counts(alive)
        return runtime_summary.SummaryContext(
            field_stats=self._field_stats(),
            climate_end=self._climate_state(),
            terrain_counts=self._terrain_counts(),
            end_surfaces=runtime_surface_snapshots.summary_end_surface_state(
                mode,
                context=self._surface_snapshot_context(),
            ),
            trophic_role_counts=trophic_role_counts,
            meat_mode_counts=meat_mode_counts,
            reproduction_end=self._reproduction_readiness_counts(alive),
            trophic_lifecycle=self._trophic_lifecycle_summary(
                ticks_executed=ticks_executed,
            ),
            season_name=str(self._season_state()["name"]),
        )

    def _field_stats(self) -> dict[str, dict[str, float]]:
        land_tiles = [tile for row in self.grid for tile in row if tile.terrain != "water"]
        water_tiles = [tile for row in self.grid for tile in row if tile.terrain == "water"]
        return {
            "land_fertility": self._series_stats([tile.fertility for tile in land_tiles]),
            "land_moisture": self._series_stats([tile.moisture for tile in land_tiles]),
            "land_heat": self._series_stats([tile.heat for tile in land_tiles]),
            "land_vegetation": self._series_stats([tile.vegetation for tile in land_tiles]),
            "land_recovery_debt": self._series_stats(
                [tile.recovery_debt for tile in land_tiles]
            ),
            "water_moisture": self._series_stats([tile.moisture for tile in water_tiles]),
            "water_heat": self._series_stats([tile.heat for tile in water_tiles]),
        }

    def _terrain_counts(self) -> dict[str, int]:
        counts = {terrain: 0 for terrain in TERRAIN_CODES}
        for row in self.grid:
            for tile in row:
                counts[tile.terrain] += 1
        return counts

    @staticmethod
    def _series_stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "mean": round(sum(values) / len(values), 4),
        }
