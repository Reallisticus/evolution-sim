from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import cos, pi, sin
from random import Random

from evolution_sim.config import WorldConfig
from evolution_sim.env.events import Event, EventType
from evolution_sim.env.fields import EnvironmentFieldMaps, generate_environment_fields
from evolution_sim.env.taxonomy import REPLAY_TAXONOMY_MODE, apply_replay_taxonomy
from evolution_sim.genome import Genome, SpeciesMember, SpeciesRecord
from evolution_sim.genome.schema import GENE_LIMITS
from evolution_sim.genome.species import (
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


@dataclass(slots=True)
class Tile:
    terrain: str
    food: float
    water: float
    fertility: float
    moisture: float
    heat: float
    vegetation: float
    shelter: float
    recovery_debt: float
    fresh_kill_deposits: list[FreshKillDeposit] = field(default_factory=list)
    carcass_deposits: list[CarcassDeposit] = field(default_factory=list)
    occupant_id: int | None = None

    @property
    def fresh_kill_energy(self) -> float:
        return sum(deposit.energy_remaining for deposit in self.fresh_kill_deposits)

    @property
    def carcass_energy(self) -> float:
        return sum(deposit.energy_remaining for deposit in self.carcass_deposits)

    @property
    def carcass_decay(self) -> float:
        total_energy = self.carcass_energy
        if total_energy <= 0:
            return 0.0
        return sum(
            deposit.energy_remaining * deposit.freshness for deposit in self.carcass_deposits
        ) / total_energy

    @property
    def carcass_source_species(self) -> int | None:
        active_deposits = [
            deposit for deposit in self.carcass_deposits if deposit.energy_remaining > 0
        ]
        if not active_deposits:
            return None
        source_species = {deposit.source_species for deposit in active_deposits}
        if None in source_species:
            return None
        if len(source_species) == 1:
            return next(iter(source_species))
        return None


@dataclass(slots=True)
class CarcassDeposit:
    energy_remaining: float
    freshness: float
    source_species: int | None
    source_agent_id: int | None
    death_tick: int
    cause: str
    killer_id: int | None = None


@dataclass(slots=True)
class FreshKillDeposit:
    energy_remaining: float
    source_species: int | None
    source_agent_id: int | None
    death_tick: int
    killer_id: int | None = None


@dataclass(slots=True)
class Agent:
    agent_id: int
    parent_id: int | None
    lineage_id: int
    birth_tick: int
    death_tick: int | None
    x: int
    y: int
    energy: float
    hydration: float
    health: float
    max_health: float
    injury_load: float
    age: int
    alive: bool
    last_reproduction_tick: int
    last_damage_source: str
    recent_plant_energy: float
    recent_fresh_kill_energy: float
    recent_carcass_energy: float
    genome_vector: tuple[float, ...]
    genome: Genome

    def reproduction_threshold(self) -> float:
        return self.genome.max_energy * self.genome.reproduction_threshold


@dataclass(slots=True)
class SimulationWorldResult:
    run_id: str
    config: dict[str, object]
    summary: dict[str, object]
    events: list[dict[str, object]]
    viewer: dict[str, object]


@dataclass(frozen=True, slots=True)
class TrophicProfile:
    plant_share: float
    animal_share: float
    scavenger_share: float
    hunter_share: float
    breadth: float
    plant_drive: float
    animal_drive: float
    scavenger_drive: float
    hunter_drive: float
    role: str
    meat_mode: str


@dataclass(frozen=True, slots=True)
class BioticFieldState:
    prey_biomass: list[list[float]]
    carrion: list[list[float]]
    predator_risk: list[list[float]]

    def to_serializable(self) -> dict[str, list[list[float]]]:
        return {
            "prey_biomass": [
                [round(value, 4) for value in row] for row in self.prey_biomass
            ],
            "carrion": [[round(value, 4) for value in row] for row in self.carrion],
            "predator_risk": [
                [round(value, 4) for value in row] for row in self.predator_risk
            ],
        }


class SimulationWorld:
    def __init__(self, config: WorldConfig):
        self.config = config
        self.rng = Random(config.seed)
        self.run_id = f"seed-{config.seed}-ticks-{config.max_ticks}"
        self.environment_fields = self._build_environment_fields()
        self.terrain_map = self._build_terrain_map()
        self.grid = self._build_grid()
        self.agents: dict[int, Agent] = {}
        self.events: list[Event] = []
        self.viewer_frames: list[dict[str, object]] = []
        self.tick = 0
        self.next_agent_id = 1
        self.births = 0
        self.deaths = 0
        self.peak_alive_agents = 0
        self.last_birth_tick: int | None = None
        self.species_registry: dict[int, dict[str, object]] = {}
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
        self.tick_attack_events: list[dict[str, object]] = []
        self.tick_damage_events: list[dict[str, object]] = []
        self.tick_carcass_deposit_events: list[dict[str, object]] = []
        self.tick_carcass_events: list[dict[str, object]] = []
        self.tick_fresh_kill_events: list[dict[str, object]] = []
        self.tick_fresh_kill_deposit_events: list[dict[str, object]] = []
        self.tick_fresh_kill_to_carcass_energy = 0.0
        self.tick_carcass_energy_decayed = 0.0
        self.tick_feeding_events: list[dict[str, object]] = []
        self.tick_hazard_exposure_agents: set[int] = set()
        self.run_combat_totals = self._empty_combat_totals()
        self.run_fresh_kill_totals = self._empty_fresh_kill_totals()
        self.run_carcass_totals = self._empty_carcass_totals()
        self.run_diet_totals = self._empty_diet_totals()
        self.run_diet_by_trophic_role = self._empty_grouped_diet_totals(
            [role for role in TROPHIC_ROLE_CODES if role != "none"]
        )
        self.run_diet_by_meat_mode = self._empty_grouped_diet_totals(MEAT_MODE_CODES)
        self.biotic_state_revision = 0
        self.cached_biotic_state_revision: int | None = None
        self.cached_biotic_state: BioticFieldState | None = None
        self.climate_phase = self._build_climate_phase()
        self.cached_climate_tick: int | None = None
        self.cached_climate_state: dict[str, object] | None = None
        self.cached_habitat_tick: int | None = None
        self.cached_habitat_grid: list[list[str]] | None = None
        self.cached_habitat_counts: dict[str, int] | None = None

        self._spawn_initial_agents()
        initial_alive = self.alive_agents()
        self.current_species_map = {
            agent.agent_id: agent.lineage_id for agent in initial_alive
        }
        self.agent_last_species_map = self.current_species_map.copy()
        self.peak_alive_agents = len(self.alive_agents())

    def run(self) -> SimulationWorldResult:
        self._emit(EventType.RUN_STARTED, data={"run_id": self.run_id})
        for tick in range(self.config.max_ticks):
            self.tick = tick
            births_this_tick, deaths_this_tick = self._run_tick()
            self._capture_frame(births_this_tick=births_this_tick, deaths_this_tick=deaths_this_tick)
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
        summary = self._build_summary()
        viewer = self._build_viewer_payload()
        event_payloads = [event.to_dict() for event in self.events]
        summary, viewer = apply_replay_taxonomy(
            config=self.config,
            summary=summary,
            events=event_payloads,
            viewer=viewer,
        )
        return SimulationWorldResult(
            run_id=self.run_id,
            config=self.config.to_dict(),
            summary=summary,
            events=event_payloads,
            viewer=viewer,
        )

    def _run_tick(self) -> tuple[int, int]:
        births_this_tick = 0
        deaths_this_tick = 0
        self.tick_birth_pairs = []
        self.tick_death_agent_ids = []
        self.tick_attack_events = []
        self.tick_damage_events = []
        self.tick_carcass_deposit_events = []
        self.tick_carcass_events = []
        self.tick_fresh_kill_events = []
        self.tick_fresh_kill_deposit_events = []
        self.tick_fresh_kill_to_carcass_energy = 0.0
        self.tick_carcass_energy_decayed = 0.0
        self.tick_feeding_events = []
        self.tick_hazard_exposure_agents = set()
        self._invalidate_biotic_state()
        climate_state = self._climate_state()
        self._emit(
            EventType.TICK_STARTED,
            data={
                "alive_agents": len(self.alive_agents()),
                "season": self._season_state()["name"],
                "disturbance_type": climate_state["disturbance_type"],
                "disturbance_strength": climate_state["disturbance_strength"],
            },
        )
        self._regrow_resources()

        for agent_id in sorted(self.agents):
            agent = self.agents[agent_id]
            if not agent.alive:
                continue
            self._decay_recent_diet(agent)
            action = self._choose_action(agent)
            moved = self._resolve_action(agent, action)
            self._apply_metabolism(agent, moved=moved)
            self._apply_health_and_hazards(agent, moved=moved)
            agent.age += 1
            self._invalidate_biotic_state()

        for agent_id in sorted(self.agents):
            agent = self.agents[agent_id]
            if agent.alive and self._can_reproduce(agent):
                if self._reproduce(agent):
                    births_this_tick += 1

        for agent_id in sorted(self.agents):
            agent = self.agents[agent_id]
            if agent.alive and self._should_die(agent):
                self._kill_agent(agent, cause=self._death_cause(agent))
                deaths_this_tick += 1

        alive_count = len(self.alive_agents())
        self.peak_alive_agents = max(self.peak_alive_agents, alive_count)
        self._emit(
            EventType.TICK_COMPLETED,
            data={
                "alive_agents": alive_count,
                "births": births_this_tick,
                "deaths": deaths_this_tick,
                "season": self._season_state()["name"],
                "disturbance_type": climate_state["disturbance_type"],
                "disturbance_strength": climate_state["disturbance_strength"],
            },
        )
        return births_this_tick, deaths_this_tick

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
        total = 0
        matches = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if not self._in_bounds(nx, ny):
                    continue
                total += 1
                if self.terrain_map[ny][nx] in terrain_filter:
                    matches += 1
        return matches / total if total else 0.0

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
            )
            self._place_agent(agent)
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
        return {
            "carcass_tiles": 0,
            "total_carcass_energy": 0.0,
            "deposition_events": 0,
            "energy_deposited": 0.0,
            "energy_decayed": 0.0,
            "consumption_events": 0,
            "energy_consumed": 0.0,
            "gained_energy": 0.0,
        }

    @staticmethod
    def _empty_fresh_kill_totals() -> dict[str, float]:
        return {
            "fresh_kill_tiles": 0,
            "total_fresh_kill_energy": 0.0,
            "deposition_events": 0,
            "energy_deposited": 0.0,
            "energy_converted_to_carcass": 0.0,
            "consumption_events": 0,
            "energy_consumed": 0.0,
            "gained_energy": 0.0,
        }

    @staticmethod
    def _empty_diet_totals() -> dict[str, float]:
        return {
            "plant_events": 0,
            "plant_energy": 0.0,
            "fresh_kill_events": 0,
            "fresh_kill_energy": 0.0,
            "carcass_events": 0,
            "carcass_energy": 0.0,
        }

    @staticmethod
    def _empty_grouped_diet_totals(groups: list[str] | dict[str, int]) -> dict[str, dict[str, float]]:
        return {group: SimulationWorld._empty_diet_totals() for group in groups}

    @staticmethod
    def _accumulate_diet_totals(
        totals: dict[str, float],
        food_source: str,
        gained_energy: float,
    ) -> None:
        if food_source not in {"plant", "fresh_kill", "carcass"}:
            raise ValueError(f"Unsupported food source: {food_source}")
        totals[f"{food_source}_events"] += 1
        totals[f"{food_source}_energy"] += gained_energy

    @staticmethod
    def _finalize_diet_totals(totals: dict[str, float]) -> dict[str, float]:
        animal_events = totals["fresh_kill_events"] + totals["carcass_events"]
        animal_energy = totals["fresh_kill_energy"] + totals["carcass_energy"]
        total_energy = totals["plant_energy"] + animal_energy
        return {
            **{
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in totals.items()
            },
            "animal_events": animal_events,
            "animal_energy": round(animal_energy, 4),
            "plant_energy_share": round(
                totals["plant_energy"] / max(total_energy, 1e-9),
                4,
            )
            if total_energy > 0
            else 0.0,
            "animal_energy_share": round(
                animal_energy / max(total_energy, 1e-9),
                4,
            )
            if total_energy > 0
            else 0.0,
            "fresh_kill_energy_share": round(
                totals["fresh_kill_energy"] / max(total_energy, 1e-9),
                4,
            )
            if total_energy > 0
            else 0.0,
            "carcass_energy_share": round(
                totals["carcass_energy"] / max(total_energy, 1e-9),
                4,
            )
            if total_energy > 0
            else 0.0,
        }

    def _finalize_grouped_diet_totals(
        self,
        grouped_totals: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        return {
            group: self._finalize_diet_totals(totals)
            for group, totals in grouped_totals.items()
        }

    def _emit(
        self,
        event_type: EventType,
        agent_id: int | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
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
        tile.fresh_kill_deposits = [
            deposit for deposit in tile.fresh_kill_deposits if deposit.energy_remaining > 1e-9
        ]

    @staticmethod
    def _prune_carcass_deposits(tile: Tile) -> None:
        tile.carcass_deposits = [
            deposit for deposit in tile.carcass_deposits if deposit.energy_remaining > 1e-9
        ]

    def _invalidate_biotic_state(self) -> None:
        self.biotic_state_revision += 1

    @staticmethod
    def _merge_fresh_kill_deposit_group(deposits: list[FreshKillDeposit]) -> FreshKillDeposit:
        if len(deposits) == 1:
            return deposits[0]
        total_energy = sum(deposit.energy_remaining for deposit in deposits)
        if total_energy <= 0:
            return deposits[0]
        source_species = deposits[0].source_species
        if any(deposit.source_species != source_species for deposit in deposits[1:]):
            source_species = None
        source_agent_id = deposits[0].source_agent_id
        if any(deposit.source_agent_id != source_agent_id for deposit in deposits[1:]):
            source_agent_id = None
        killer_id = deposits[0].killer_id
        if any(deposit.killer_id != killer_id for deposit in deposits[1:]):
            killer_id = None
        return FreshKillDeposit(
            energy_remaining=total_energy,
            source_species=source_species,
            source_agent_id=source_agent_id,
            death_tick=min(deposit.death_tick for deposit in deposits),
            killer_id=killer_id,
        )

    def _carcass_freshness_bucket(self, freshness: float) -> int:
        bucket_size = max(self.config.carcasses.freshness_merge_bucket, 1e-6)
        return int(self._clamp01(freshness) / bucket_size)

    @staticmethod
    def _merge_carcass_deposit_group(deposits: list[CarcassDeposit]) -> CarcassDeposit:
        if len(deposits) == 1:
            return deposits[0]
        total_energy = sum(deposit.energy_remaining for deposit in deposits)
        if total_energy <= 0:
            return deposits[0]
        source_species = deposits[0].source_species
        if any(deposit.source_species != source_species for deposit in deposits[1:]):
            source_species = None
        source_agent_id = deposits[0].source_agent_id
        if any(deposit.source_agent_id != source_agent_id for deposit in deposits[1:]):
            source_agent_id = None
        cause = deposits[0].cause
        if any(deposit.cause != cause for deposit in deposits[1:]):
            cause = "mixed"
        killer_id = deposits[0].killer_id
        if any(deposit.killer_id != killer_id for deposit in deposits[1:]):
            killer_id = None
        return CarcassDeposit(
            energy_remaining=total_energy,
            freshness=sum(
                deposit.energy_remaining * deposit.freshness for deposit in deposits
            )
            / total_energy,
            source_species=source_species,
            source_agent_id=source_agent_id,
            death_tick=min(deposit.death_tick for deposit in deposits),
            cause=cause,
            killer_id=killer_id,
        )

    def _compact_carcass_deposits(self, tile: Tile) -> None:
        self._prune_carcass_deposits(tile)
        limit = max(1, self.config.carcasses.max_tile_deposits)
        if len(tile.carcass_deposits) <= limit:
            return

        grouped: dict[tuple[int | None, int, str, int | None], list[CarcassDeposit]] = defaultdict(list)
        for deposit in tile.carcass_deposits:
            key = (
                deposit.source_species,
                self._carcass_freshness_bucket(deposit.freshness),
                deposit.cause,
                deposit.killer_id,
            )
            grouped[key].append(deposit)

        compacted = [
            self._merge_carcass_deposit_group(group)
            for group in grouped.values()
        ]
        compacted.sort(
            key=lambda deposit: (
                deposit.source_species is None,
                int(deposit.source_species or 0),
                -deposit.freshness,
                deposit.death_tick,
            )
        )

        while len(compacted) > limit:
            merge_index: int | None = None
            merge_delta = float("inf")
            for index in range(len(compacted) - 1):
                current = compacted[index]
                following = compacted[index + 1]
                if current.source_species != following.source_species:
                    continue
                delta = abs(current.freshness - following.freshness)
                if delta < merge_delta:
                    merge_delta = delta
                    merge_index = index
            if merge_index is None:
                break
            merged = self._merge_carcass_deposit_group(
                compacted[merge_index : merge_index + 2]
            )
            compacted[merge_index : merge_index + 2] = [merged]

        tile.carcass_deposits = compacted

    def _compact_fresh_kill_deposits(self, tile: Tile) -> None:
        self._prune_fresh_kill_deposits(tile)
        limit = max(1, self.config.carcasses.max_tile_deposits)
        if len(tile.fresh_kill_deposits) <= limit:
            return

        grouped: dict[tuple[int | None, int | None], list[FreshKillDeposit]] = defaultdict(list)
        for deposit in tile.fresh_kill_deposits:
            grouped[(deposit.source_species, deposit.killer_id)].append(deposit)

        compacted = [
            self._merge_fresh_kill_deposit_group(group)
            for group in grouped.values()
        ]
        compacted.sort(
            key=lambda deposit: (
                deposit.source_species is None,
                int(deposit.source_species or 0),
                -deposit.death_tick,
            )
        )
        if len(compacted) > limit:
            overflow = compacted[limit - 1 :]
            compacted = compacted[: limit - 1] + [self._merge_fresh_kill_deposit_group(overflow)]
        tile.fresh_kill_deposits = compacted

    def _carcass_source_breakdown(
        self,
        deposits: list[CarcassDeposit],
    ) -> list[dict[str, object]]:
        source_energy: dict[tuple[int | None, int | None, int | None], float] = defaultdict(float)
        for deposit in deposits:
            if deposit.energy_remaining <= 0:
                continue
            source_energy[
                (
                    deposit.source_agent_id,
                    deposit.death_tick,
                    deposit.source_species,
                )
            ] += deposit.energy_remaining
        breakdown = [
            {
                "source_agent_id": source_agent_id,
                "death_tick": death_tick,
                "source_species": source_species,
                "energy": round(energy, 4),
            }
            for (source_agent_id, death_tick, source_species), energy in source_energy.items()
            if energy > 0
        ]
        breakdown.sort(
            key=lambda item: (
                -float(item["energy"]),
                item["source_species"] is None,
                item["source_agent_id"] is None,
                int(item["source_species"] or 0),
                int(item["source_agent_id"] or 0),
                int(item["death_tick"]) if item["death_tick"] is not None else -1,
            )
        )
        return breakdown

    def _fresh_kill_source_breakdown(
        self,
        deposits: list[FreshKillDeposit],
    ) -> list[dict[str, object]]:
        source_energy: dict[tuple[int | None, int | None, int | None, int | None], float] = (
            defaultdict(float)
        )
        for deposit in deposits:
            if deposit.energy_remaining <= 0:
                continue
            source_energy[
                (
                    deposit.source_agent_id,
                    deposit.death_tick,
                    deposit.source_species,
                    deposit.killer_id,
                )
            ] += deposit.energy_remaining
        breakdown = [
            {
                "source_agent_id": source_agent_id,
                "death_tick": death_tick,
                "source_species": source_species,
                "killer_id": killer_id,
                "energy": round(energy, 4),
            }
            for (
                source_agent_id,
                death_tick,
                source_species,
                killer_id,
            ), energy in source_energy.items()
            if energy > 0
        ]
        breakdown.sort(
            key=lambda item: (
                -float(item["energy"]),
                item["source_species"] is None,
                item["source_agent_id"] is None,
                int(item["source_species"] or 0),
                int(item["source_agent_id"] or 0),
                int(item["death_tick"]) if item["death_tick"] is not None else -1,
            )
        )
        return breakdown

    @staticmethod
    def _resolved_source_species(source_breakdown: list[dict[str, object]]) -> int | None:
        if not source_breakdown:
            return None
        source_species = {entry.get("source_species") for entry in source_breakdown}
        if None in source_species:
            return None
        if len(source_species) == 1:
            return next(iter(source_species))
        return None

    def _carcass_tile_state(self, tile: Tile) -> dict[str, object]:
        total_energy = tile.carcass_energy
        source_breakdown = self._carcass_source_breakdown(tile.carcass_deposits)
        dominant_source_species = self._resolved_source_species(source_breakdown)
        return {
            "deposit_count": len(tile.carcass_deposits),
            "total_energy": round(total_energy, 4),
            "avg_freshness": round(tile.carcass_decay, 4),
            "dominant_source_species": dominant_source_species,
            "mixed_sources": len(source_breakdown) > 1,
            "source_breakdown": source_breakdown,
        }

    def _fresh_kill_tile_state(self, tile: Tile) -> dict[str, object]:
        total_energy = tile.fresh_kill_energy
        source_breakdown = self._fresh_kill_source_breakdown(tile.fresh_kill_deposits)
        dominant_source_species = self._resolved_source_species(source_breakdown)
        return {
            "deposit_count": len(tile.fresh_kill_deposits),
            "total_energy": round(total_energy, 4),
            "dominant_source_species": dominant_source_species,
            "mixed_sources": len(source_breakdown) > 1,
            "source_breakdown": source_breakdown,
        }

    def _carcass_tile_summary_for_position(self, x: int, y: int) -> dict[str, object]:
        summary = self._carcass_tile_state(self.grid[y][x])
        return {
            "x": x,
            "y": y,
            **summary,
        }

    def _fresh_kill_tile_summary_for_position(self, x: int, y: int) -> dict[str, object]:
        summary = self._fresh_kill_tile_state(self.grid[y][x])
        return {
            "x": x,
            "y": y,
            **summary,
        }

    def _carcass_patch_summaries(self) -> list[dict[str, object]]:
        patches: list[dict[str, object]] = []
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                if tile.terrain == "water" or tile.carcass_energy <= 0:
                    continue
                patches.append(self._carcass_tile_summary_for_position(x, y))
        return patches

    def _fresh_kill_patch_summaries(self) -> list[dict[str, object]]:
        patches: list[dict[str, object]] = []
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                if tile.terrain == "water" or tile.fresh_kill_energy <= 0:
                    continue
                patches.append(self._fresh_kill_tile_summary_for_position(x, y))
        return patches

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
        if energy <= 0:
            return self._fresh_kill_tile_summary_for_position(x, y)
        tile.fresh_kill_deposits.append(
            FreshKillDeposit(
                energy_remaining=energy,
                source_species=source_species,
                source_agent_id=source_agent_id,
                death_tick=self.tick,
                killer_id=killer_id,
            )
        )
        self._compact_fresh_kill_deposits(tile)
        self.run_fresh_kill_totals["deposition_events"] += 1
        self.run_fresh_kill_totals["energy_deposited"] += energy
        patch_state = self._fresh_kill_tile_summary_for_position(x, y)
        source_breakdown = self._fresh_kill_source_breakdown(
            [
                FreshKillDeposit(
                    energy_remaining=energy,
                    source_species=source_species,
                    source_agent_id=source_agent_id,
                    death_tick=self.tick,
                    killer_id=killer_id,
                )
            ]
        )
        self.tick_fresh_kill_deposit_events.append(
            {
                "source_agent_id": source_agent_id,
                "source_species": self._resolved_source_species(source_breakdown),
                "deposited_energy": round(energy, 4),
                "killer_id": killer_id,
                "x": x,
                "y": y,
                "deposit_count": patch_state["deposit_count"],
                "tile_fresh_kill_energy": patch_state["total_energy"],
                "dominant_source_species": patch_state["dominant_source_species"],
                "mixed_sources": patch_state["mixed_sources"],
                "source_breakdown": source_breakdown,
            }
        )
        return patch_state

    def _convert_fresh_kill_to_carcass(
        self,
        tile: Tile,
        *,
        x: int,
        y: int,
        conversion_rate: float,
    ) -> float:
        if conversion_rate <= 0 or not tile.fresh_kill_deposits:
            return 0.0
        converted_deposits: list[CarcassDeposit] = []
        converted_energy = 0.0
        for deposit in tile.fresh_kill_deposits:
            amount = min(deposit.energy_remaining, deposit.energy_remaining * conversion_rate)
            if amount <= 0:
                continue
            deposit.energy_remaining -= amount
            converted_energy += amount
            converted_deposits.append(
                CarcassDeposit(
                    energy_remaining=amount,
                    freshness=1.0,
                    source_species=deposit.source_species,
                    source_agent_id=deposit.source_agent_id,
                    death_tick=deposit.death_tick,
                    cause="fresh_kill_decay",
                    killer_id=deposit.killer_id,
                )
            )
        self._compact_fresh_kill_deposits(tile)
        if not converted_deposits:
            return 0.0
        tile.carcass_deposits.extend(converted_deposits)
        self._compact_carcass_deposits(tile)
        self.run_fresh_kill_totals["energy_converted_to_carcass"] += converted_energy
        self.tick_fresh_kill_to_carcass_energy += converted_energy
        self.run_carcass_totals["deposition_events"] += len(converted_deposits)
        self.run_carcass_totals["energy_deposited"] += converted_energy
        patch_state = self._carcass_tile_summary_for_position(x, y)
        source_breakdown = self._carcass_source_breakdown(converted_deposits)
        dominant_source_species = self._resolved_source_species(source_breakdown)
        self.tick_carcass_deposit_events.append(
            {
                "source_agent_id": None,
                "source_species": dominant_source_species,
                "deposited_energy": round(converted_energy, 4),
                "x": x,
                "y": y,
                "deposit_count": patch_state["deposit_count"],
                "tile_carcass_energy": patch_state["total_energy"],
                "tile_avg_freshness": patch_state["avg_freshness"],
                "dominant_source_species": patch_state["dominant_source_species"],
                "mixed_sources": patch_state["mixed_sources"],
                "converted_from_fresh_kill": True,
                "source_breakdown": source_breakdown,
            }
        )
        return converted_energy

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
        if energy <= 0:
            return self._carcass_tile_summary_for_position(x, y)
        tile.carcass_deposits.append(
            CarcassDeposit(
                energy_remaining=energy,
                freshness=1.0,
                source_species=source_species,
                source_agent_id=source_agent_id,
                death_tick=self.tick,
                cause=cause,
                killer_id=killer_id,
            )
        )
        self._compact_carcass_deposits(tile)
        self.run_carcass_totals["deposition_events"] += 1
        self.run_carcass_totals["energy_deposited"] += energy
        patch_state = self._carcass_tile_summary_for_position(x, y)
        source_breakdown = self._carcass_source_breakdown(
            [
                CarcassDeposit(
                    energy_remaining=energy,
                    freshness=1.0,
                    source_species=source_species,
                    source_agent_id=source_agent_id,
                    death_tick=self.tick,
                    cause=cause,
                    killer_id=killer_id,
                )
            ]
        )
        self.tick_carcass_deposit_events.append(
            {
                "source_agent_id": source_agent_id,
                "source_species": self._resolved_source_species(source_breakdown),
                "deposited_energy": round(energy, 4),
                "x": x,
                "y": y,
                "deposit_count": patch_state["deposit_count"],
                "tile_carcass_energy": patch_state["total_energy"],
                "tile_avg_freshness": patch_state["avg_freshness"],
                "dominant_source_species": patch_state["dominant_source_species"],
                "mixed_sources": patch_state["mixed_sources"],
                "source_breakdown": source_breakdown,
            }
        )
        self._emit(
            EventType.CARCASS_DEPOSITED,
            agent_id=source_agent_id,
            data={
                "source_agent_id": source_agent_id,
                "source_species": self._resolved_source_species(source_breakdown),
                "deposited_energy": round(energy, 4),
                "cause": cause,
                "killer_id": killer_id,
                "x": x,
                "y": y,
                "tile_carcass_energy_after": patch_state["total_energy"],
                "tile_avg_freshness_after": patch_state["avg_freshness"],
                "tile_deposit_count_after": patch_state["deposit_count"],
                "tile_mixed_sources_after": patch_state["mixed_sources"],
                "tile_dominant_source_species_after": patch_state["dominant_source_species"],
                "tile_source_breakdown_after": patch_state["source_breakdown"],
            },
        )
        return patch_state

    def _decay_carcass_tile(self, tile: Tile, *, decay: float) -> float:
        if decay <= 0 or not tile.carcass_deposits:
            return 0.0
        energy_decayed = 0.0
        for deposit in tile.carcass_deposits:
            before_energy = deposit.energy_remaining
            deposit.freshness = max(0.0, deposit.freshness - decay)
            deposit.energy_remaining = max(
                0.0,
                deposit.energy_remaining - decay * (0.42 + deposit.energy_remaining * 0.56),
            )
            energy_decayed += before_energy - deposit.energy_remaining
        self._compact_carcass_deposits(tile)
        return energy_decayed

    def _consume_carcass_from_tile(self, tile: Tile, requested_amount: float) -> dict[str, object]:
        remaining = max(0.0, requested_amount)
        if remaining <= 0 or not tile.carcass_deposits:
            return {
                "consumed": 0.0,
                "avg_freshness": 0.0,
                "deposit_breakdown": [],
                "source_breakdown": [],
            }
        consumed = 0.0
        freshness_weighted = 0.0
        deposit_breakdown: list[dict[str, object]] = []
        deposits = sorted(
            tile.carcass_deposits,
            key=lambda deposit: (-deposit.freshness, -deposit.death_tick, -(deposit.source_agent_id or 0)),
        )
        for deposit in deposits:
            if remaining <= 0:
                break
            amount = min(deposit.energy_remaining, remaining)
            if amount <= 0:
                continue
            freshness = 0.7 + deposit.freshness * 0.3
            deposit.energy_remaining -= amount
            deposit.freshness = max(0.0, deposit.freshness - amount * 0.18)
            remaining -= amount
            consumed += amount
            freshness_weighted += amount * freshness
            deposit_breakdown.append(
                {
                    "source_agent_id": deposit.source_agent_id,
                    "source_species": deposit.source_species,
                    "consumed": round(amount, 4),
                    "freshness": round(freshness, 4),
                    "death_tick": deposit.death_tick,
                    "cause": deposit.cause,
                }
            )
        self._compact_carcass_deposits(tile)
        return {
            "consumed": consumed,
            "avg_freshness": freshness_weighted / consumed if consumed > 0 else 0.0,
            "deposit_breakdown": deposit_breakdown,
            "source_breakdown": self._carcass_source_breakdown(
                [
                    CarcassDeposit(
                        energy_remaining=float(entry["consumed"]),
                        freshness=0.0,
                        source_species=entry["source_species"],
                        source_agent_id=entry["source_agent_id"],
                        death_tick=int(entry["death_tick"]),
                        cause=str(entry["cause"]),
                    )
                    for entry in deposit_breakdown
                ]
            ),
        }

    def _consume_fresh_kill_from_tile(
        self,
        tile: Tile,
        requested_amount: float,
    ) -> dict[str, object]:
        remaining = max(0.0, requested_amount)
        if remaining <= 0 or not tile.fresh_kill_deposits:
            return {
                "consumed": 0.0,
                "deposit_breakdown": [],
                "source_breakdown": [],
            }
        consumed = 0.0
        deposit_breakdown: list[dict[str, object]] = []
        deposits = sorted(
            tile.fresh_kill_deposits,
            key=lambda deposit: (-deposit.death_tick, -(deposit.source_agent_id or 0)),
        )
        for deposit in deposits:
            if remaining <= 0:
                break
            amount = min(deposit.energy_remaining, remaining)
            if amount <= 0:
                continue
            deposit.energy_remaining -= amount
            remaining -= amount
            consumed += amount
            deposit_breakdown.append(
                {
                    "source_agent_id": deposit.source_agent_id,
                    "source_species": deposit.source_species,
                    "consumed": round(amount, 4),
                    "death_tick": deposit.death_tick,
                    "killer_id": deposit.killer_id,
                }
            )
        self._compact_fresh_kill_deposits(tile)
        return {
            "consumed": consumed,
            "deposit_breakdown": deposit_breakdown,
            "source_breakdown": self._fresh_kill_source_breakdown(
                [
                    FreshKillDeposit(
                        energy_remaining=float(entry["consumed"]),
                        source_species=entry["source_species"],
                        source_agent_id=entry["source_agent_id"],
                        death_tick=int(entry["death_tick"]),
                        killer_id=entry["killer_id"],
                    )
                    for entry in deposit_breakdown
                ]
            ),
        }

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

        season = self._season_state()["name"]
        climate_state = self._climate_state()
        grid: list[list[str]] = []
        counts = {state: 0 for state in HABITAT_STATE_CODES}
        for y, row in enumerate(self.grid):
            state_row: list[str] = []
            for x, tile in enumerate(row):
                if tile.terrain == "water":
                    state = "stable"
                else:
                    fertility, moisture, heat = self._effective_tile_fields(x, y, season)
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

        season = self._season_state()["name"]
        _, moisture, heat = self._effective_tile_fields(x, y, season)
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
        energy_codes: list[list[int]] = []
        freshness_codes: list[list[int]] = []
        count = 0
        total_energy = 0.0
        total_freshness_energy = 0.0
        total_deposits = 0
        mixed_source_tiles = 0
        for row in self.grid:
            energy_row: list[int] = []
            freshness_row: list[int] = []
            for tile in row:
                if tile.terrain == "water":
                    energy_row.append(NON_LAND_ECOLOGY_CODE)
                    freshness_row.append(NON_LAND_ECOLOGY_CODE)
                    continue
                if tile.carcass_energy <= 0:
                    energy_row.append(0)
                    freshness_row.append(0)
                    continue
                count += 1
                total_energy += tile.carcass_energy
                total_freshness_energy += tile.carcass_energy * tile.carcass_decay
                total_deposits += len(tile.carcass_deposits)
                if len(self._carcass_source_breakdown(tile.carcass_deposits)) > 1:
                    mixed_source_tiles += 1
                energy_row.append(round(self._clamp01(tile.carcass_energy) * 100))
                freshness_row.append(round(self._clamp01(tile.carcass_decay) * 100))
            energy_codes.append(energy_row)
            freshness_codes.append(freshness_row)
        return (
            energy_codes,
            freshness_codes,
            {
                "carcass_tiles": count,
                "total_carcass_energy": round(total_energy, 4),
                "avg_carcass_freshness": round(
                    total_freshness_energy / total_energy if total_energy > 0 else 0.0,
                    4,
                ),
                "deposit_count": total_deposits,
                "mixed_source_tiles": mixed_source_tiles,
            },
        )

    def _fresh_kill_snapshot(self) -> tuple[list[list[int]], dict[str, float]]:
        energy_codes: list[list[int]] = []
        count = 0
        total_energy = 0.0
        total_deposits = 0
        mixed_source_tiles = 0
        for row in self.grid:
            energy_row: list[int] = []
            for tile in row:
                if tile.terrain == "water":
                    energy_row.append(NON_LAND_ECOLOGY_CODE)
                    continue
                if tile.fresh_kill_energy <= 0:
                    energy_row.append(0)
                    continue
                count += 1
                total_energy += tile.fresh_kill_energy
                total_deposits += len(tile.fresh_kill_deposits)
                if len(self._fresh_kill_source_breakdown(tile.fresh_kill_deposits)) > 1:
                    mixed_source_tiles += 1
                energy_row.append(round(self._clamp01(tile.fresh_kill_energy) * 100))
            energy_codes.append(energy_row)
        return (
            energy_codes,
            {
                "fresh_kill_tiles": count,
                "total_fresh_kill_energy": round(total_energy, 4),
                "deposit_count": total_deposits,
                "mixed_source_tiles": mixed_source_tiles,
            },
        )

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
        if gained_energy <= 0:
            return
        if food_source == "plant":
            agent.recent_plant_energy += gained_energy
        elif food_source == "fresh_kill":
            agent.recent_fresh_kill_energy += gained_energy
        elif food_source == "carcass":
            agent.recent_carcass_energy += gained_energy

    def _trophic_profile_for_genome(self, genome: Genome) -> TrophicProfile:
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
        plant_drive = plant_trait * plant_focus * breadth_penalty
        animal_drive = animal_trait * animal_focus * breadth_penalty
        scavenger_drive = animal_drive * (0.22 + scavenger_trait * 0.78) * scavenger_focus
        hunter_drive = animal_drive * (0.18 + hunter_trait * 0.82) * hunter_focus

        specialist_threshold = self.config.trophic.specialist_share_threshold
        if plant_focus >= specialist_threshold and plant_drive >= 0.2:
            role = "herbivore"
        elif animal_focus >= specialist_threshold and animal_drive >= 0.26:
            role = "carnivore"
        else:
            role = "omnivore"

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

    def _trophic_profile(self, agent: Agent) -> TrophicProfile:
        return self._trophic_profile_for_genome(agent.genome)

    def _trophic_role_for_genome(self, genome: Genome) -> str:
        return self._trophic_profile_for_genome(genome).role

    def _trophic_role(self, agent: Agent) -> str:
        return self._trophic_profile(agent).role

    def _meat_mode(self, agent: Agent) -> str:
        return self._trophic_profile(agent).meat_mode

    def _can_consume_animal(self, agent: Agent) -> bool:
        profile = self._trophic_profile(agent)
        return (
            profile.role != "herbivore"
            and profile.animal_share >= self.config.trophic.animal_channel_threshold
            and max(profile.scavenger_drive, profile.hunter_drive)
            >= self.config.trophic.animal_use_drive_threshold
        )

    def _can_consume_fresh_kill(self, agent: Agent) -> bool:
        return self._can_consume_animal(agent)

    def _can_consume_carcass(self, agent: Agent) -> bool:
        return self._can_consume_animal(agent)

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
        return self._animal_intake_useful(agent)

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
        if profile.role == "carnivore":
            gained *= 0.04
        elif profile.meat_mode == "hunter":
            gained *= 0.24 + profile.plant_share * 0.32
        return gained * profile.plant_drive

    @staticmethod
    def _fresh_kill_drive(profile: TrophicProfile) -> float:
        return profile.hunter_drive + profile.scavenger_drive * 0.28

    @staticmethod
    def _carcass_drive(profile: TrophicProfile) -> float:
        return profile.scavenger_drive + profile.hunter_drive * 0.24

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
            * 1.34
        )

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
            * 1.12
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
        self._record_recent_diet(agent, food_source, gained_energy)
        self.tick_feeding_events.append(
            {
                "agent_id": agent.agent_id,
                "species_id": self.current_species_map.get(
                    agent.agent_id,
                    self.agent_last_species_map.get(agent.agent_id),
                ),
                "food_source": food_source,
                "consumed": round(consumed, 4),
                "gained_energy": round(gained_energy, 4),
                "energy_before": round(energy_before, 4),
                "energy_after": round(energy_after, 4),
                "potential_energy": round(
                    potential_energy if potential_energy is not None else gained_energy,
                    4,
                ),
                "trophic_role": profile.role,
                "meat_mode": profile.meat_mode,
                "matched_diet_ratio": round(self._matched_diet_ratio(agent, profile), 4),
            }
        )
        self._accumulate_diet_totals(self.run_diet_totals, food_source, gained_energy)
        self._accumulate_diet_totals(
            self.run_diet_by_trophic_role[profile.role],
            food_source,
            gained_energy,
        )
        self._accumulate_diet_totals(
            self.run_diet_by_meat_mode[profile.meat_mode],
            food_source,
            gained_energy,
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

    def _diffuse_biotic_field(self, sources: list[list[float]]) -> list[list[float]]:
        radius = max(1, self.config.biotic_fields.diffusion_radius)
        field = [
            [0.0 for _ in range(self.config.width)] for _ in range(self.config.height)
        ]
        for sy, row in enumerate(sources):
            for sx, source in enumerate(row):
                if source <= 1e-9 or self.grid[sy][sx].terrain == "water":
                    continue
                for dy in range(-radius, radius + 1):
                    span = radius - abs(dy)
                    for dx in range(-span, span + 1):
                        distance = abs(dx) + abs(dy)
                        if distance > radius:
                            continue
                        x = sx + dx
                        y = sy + dy
                        if not self._in_bounds(x, y) or self.grid[y][x].terrain == "water":
                            continue
                        field[y][x] += source / (distance + 1.0)
        return field

    def _build_biotic_state(self) -> BioticFieldState:
        prey_sources = [
            [0.0 for _ in range(self.config.width)] for _ in range(self.config.height)
        ]
        carrion_sources = [
            [0.0 for _ in range(self.config.width)] for _ in range(self.config.height)
        ]
        predator_sources = [
            [0.0 for _ in range(self.config.width)] for _ in range(self.config.height)
        ]

        for agent in self.alive_agents():
            tile = self.grid[agent.y][agent.x]
            if tile.terrain == "water":
                continue
            profile = self._trophic_profile(agent)
            biomass = self._agent_biomass(agent)
            vulnerability = self._prey_vulnerability(agent)
            if profile.role == "herbivore":
                prey_sources[agent.y][agent.x] += biomass * vulnerability
            elif profile.role == "omnivore":
                weakened = max(0.0, vulnerability - 1.12)
                if weakened > 0:
                    prey_sources[agent.y][agent.x] += biomass * weakened * 0.56
            if profile.role == "carnivore" or profile.hunter_drive >= 0.12:
                predator_sources[agent.y][agent.x] += (
                    profile.hunter_drive
                    * agent.genome.attack_power
                    * self._health_ratio(agent)
                )

        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                if tile.terrain == "water":
                    continue
                carrion_sources[y][x] = tile.carcass_energy * tile.carcass_decay + tile.fresh_kill_energy

        return BioticFieldState(
            prey_biomass=self._diffuse_biotic_field(prey_sources),
            carrion=self._diffuse_biotic_field(carrion_sources),
            predator_risk=self._diffuse_biotic_field(predator_sources),
        )

    def _current_biotic_state(self) -> BioticFieldState:
        if (
            self.cached_biotic_state_revision == self.biotic_state_revision
            and self.cached_biotic_state is not None
        ):
            return self.cached_biotic_state
        self.cached_biotic_state = self._build_biotic_state()
        self.cached_biotic_state_revision = self.biotic_state_revision
        return self.cached_biotic_state

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
        season: str | None = None,
    ) -> tuple[float, float, float]:
        tile = self.grid[y][x]
        environment = self.config.environment
        climate_state = self._climate_state()
        season_name = season or climate_state["season"]
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
        tile = self.grid[y][x]
        if tile.terrain == "water":
            return 1.0
        fertility, moisture, heat = self._effective_tile_fields(x, y, season)
        habitat_state = self._habitat_state_at(x, y)
        target = (
            TERRAIN_VEGETATION_BASE.get(tile.terrain, 0.5)
            + fertility * 0.18
            + moisture * 0.24
            - heat * 0.16
        )
        if habitat_state == "bloom":
            target += 0.08
        elif habitat_state == "flooded":
            target += 0.06 if tile.terrain == "wetland" else -0.04
        elif habitat_state == "parched":
            target -= 0.08 if tile.terrain == "rocky" else 0.14
        return self._clamp01(target)

    def _shelter_target(self, x: int, y: int, season: str) -> float:
        tile = self.grid[y][x]
        if tile.terrain == "water":
            return 0.0
        fertility, moisture, heat = self._effective_tile_fields(x, y, season)
        forest_density = self._terrain_neighbor_ratio(x, y, terrain_filter={"forest"}, radius=1)
        habitat_state = self._habitat_state_at(x, y)
        target = (
            TERRAIN_SHELTER_BASE.get(tile.terrain, 0.08)
            + tile.vegetation * 0.28
            + forest_density * (0.34 if tile.terrain == "forest" else 0.08)
            + fertility * 0.06
            + moisture * 0.08
            - heat * 0.1
            - tile.recovery_debt * 0.18
        )
        if habitat_state == "bloom":
            target += 0.04
        elif habitat_state == "flooded":
            target -= 0.12
        elif habitat_state == "parched":
            target -= 0.22
        if tile.terrain != "forest":
            target *= 0.22
        return self._clamp01(target)

    def _food_capacity(self, x: int, y: int, season: str) -> float:
        tile = self.grid[y][x]
        if tile.terrain == "water":
            return 0.0
        fertility, moisture, heat = self._effective_tile_fields(x, y, season)
        capacity = (
            0.06
            + tile.vegetation * 0.7
            + fertility * 0.18
            + moisture * 0.08
            - heat * 0.06
            - tile.recovery_debt * 0.18
        )
        if tile.terrain == "forest":
            capacity += 0.08
        elif tile.terrain == "wetland":
            capacity += 0.04
        elif tile.terrain == "rocky":
            capacity -= 0.04
        return self._clamp01(capacity)

    def _field_growth_multiplier(self, x: int, y: int, season: str) -> float:
        tile = self.grid[y][x]
        fertility, moisture, heat = self._effective_tile_fields(x, y, season)
        growth = 0.42 + fertility * 0.72 + moisture * 0.44
        heat_penalty = max(0.0, heat - moisture) * 0.34
        vegetation_bonus = tile.vegetation * 0.34
        recovery_penalty = tile.recovery_debt * 0.42
        return max(0.22, growth + vegetation_bonus - heat_penalty - recovery_penalty)

    def _field_preference_score(
        self,
        agent: Agent,
        x: int,
        y: int,
        season: str,
        water_urgency: float,
        food_urgency: float,
        profile: TrophicProfile | None = None,
    ) -> float:
        profile = profile or self._trophic_profile(agent)
        tile = self.grid[y][x]
        terrain = tile.terrain
        fertility, moisture, heat = self._effective_tile_fields(x, y, season)
        habitat_state = self._habitat_state_at(x, y)
        ecology_state = self._ecology_state_at(x, y)
        water_reason = self._water_access_reason(x, y)
        refuge_score = self._refuge_score(x, y)
        soft_refuge_reason = self._soft_refuge_reason(x, y)
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
        fertility, moisture, heat = self._effective_tile_fields(x, y, season)
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
        _, moisture, heat = self._effective_tile_fields(x, y, season)
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
        season = self._season_state()["name"]
        self._habitat_state_grid()
        resources = self.config.resources
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                if tile.terrain == "water":
                    tile.water = self.config.resources.water_refresh_amount
                    continue

                if tile.fresh_kill_energy > 0:
                    self._convert_fresh_kill_to_carcass(
                        tile,
                        x=x,
                        y=y,
                        conversion_rate=self.config.carcasses.fresh_kill_conversion_rate,
                    )

                if tile.carcass_energy > 0:
                    _, moisture, heat = self._effective_tile_fields(x, y, season)
                    decay = (
                        self.config.carcasses.decay_base_rate
                        + heat * self.config.carcasses.decay_heat_factor
                        + moisture * self.config.carcasses.decay_moisture_factor
                    )
                    energy_decayed = self._decay_carcass_tile(tile, decay=decay)
                    self.tick_carcass_energy_decayed += energy_decayed
                    self.run_carcass_totals["energy_decayed"] += energy_decayed

                fertility, moisture, heat = self._effective_tile_fields(x, y, season)
                habitat_state = self._habitat_state_at(x, y)
                field_growth = self._field_growth_multiplier(x, y, season)
                vegetation_target = self._vegetation_target(x, y, season)
                shelter_target = self._shelter_target(x, y, season)
                forest_density = self._terrain_neighbor_ratio(x, y, terrain_filter={"forest"}, radius=1)
                recovery_support = max(0.28, 1.0 - tile.recovery_debt * 0.72)
                vegetation_growth = (
                    resources.vegetation_regrowth_rate
                    * self._terrain_growth_modifier(tile.terrain, season)
                    * field_growth
                    * recovery_support
                )
                vegetation_stress = resources.terrain_degradation_rate * (
                    max(0.0, heat - moisture) * 0.64
                    + max(0.0, 0.45 - tile.food) * 0.18
                )
                if habitat_state == "bloom":
                    vegetation_growth *= 1.24
                elif habitat_state == "flooded":
                    if tile.terrain == "wetland":
                        vegetation_growth *= 1.08
                    else:
                        vegetation_stress += resources.terrain_degradation_rate * 0.16
                elif habitat_state == "parched":
                    vegetation_stress += (
                        resources.terrain_degradation_rate
                        * (0.62 if tile.terrain == "rocky" else 0.92)
                    )

                if tile.vegetation <= vegetation_target:
                    tile.vegetation = self._clamp01(
                        tile.vegetation
                        + (vegetation_target - tile.vegetation) * vegetation_growth
                        - vegetation_stress * 0.18
                    )
                else:
                    tile.vegetation = self._clamp01(
                        tile.vegetation
                        - (tile.vegetation - vegetation_target)
                        * (0.18 + vegetation_stress)
                    )

                degradation = resources.terrain_degradation_rate * (
                    max(0.0, 0.42 - tile.vegetation) * 0.94
                    + max(0.0, heat - moisture) * 0.58
                )
                recovery = resources.terrain_recovery_rate * (
                    0.44
                    + tile.vegetation * 0.84
                    + fertility * 0.3
                    + moisture * 0.24
                    + TERRAIN_RESILIENCE_BASE.get(tile.terrain, 0.56) * 0.32
                )
                if habitat_state == "bloom":
                    recovery *= 1.18
                elif habitat_state == "flooded" and tile.terrain != "wetland":
                    degradation *= 1.16
                elif habitat_state == "parched":
                    degradation *= 1.34 if tile.terrain != "rocky" else 1.16
                    recovery *= 0.72
                tile.recovery_debt = self._clamp01(tile.recovery_debt + degradation - recovery)

                shelter_growth = (
                    resources.shelter_regrowth_rate
                    * (0.44 + tile.vegetation * 0.42)
                    * (0.36 + forest_density * 0.64)
                    * max(0.28, 1.0 - tile.recovery_debt * 0.6)
                )
                shelter_stress = resources.shelter_degradation_rate * (
                    max(0.0, heat - moisture) * 0.56
                    + tile.recovery_debt * 0.34
                    + max(0.0, 0.42 - tile.vegetation) * 0.38
                )
                if habitat_state == "bloom":
                    shelter_growth *= 1.1
                elif habitat_state == "flooded":
                    shelter_stress += resources.shelter_degradation_rate * 0.3
                elif habitat_state == "parched":
                    shelter_stress += resources.shelter_degradation_rate * 0.42
                    shelter_growth *= 0.72

                if tile.shelter <= shelter_target:
                    tile.shelter = self._clamp01(
                        tile.shelter
                        + (shelter_target - tile.shelter) * shelter_growth
                        - shelter_stress * 0.1
                    )
                else:
                    tile.shelter = self._clamp01(
                        tile.shelter
                        - (tile.shelter - shelter_target) * (0.14 + shelter_stress)
                    )

                if habitat_state == "parched":
                    tile.food = max(
                        0.0,
                        tile.food - (0.008 if tile.terrain == "plain" else 0.0045),
                    )
                elif habitat_state == "flooded" and tile.terrain == "plain":
                    tile.food = max(0.0, tile.food - 0.003)

                food_regrowth = (
                    self._terrain_regrowth_rate(tile.terrain)
                    * self._terrain_growth_modifier(tile.terrain, season)
                    * field_growth
                    * (0.4 + tile.vegetation * 0.84)
                    * max(0.24, 1.0 - tile.recovery_debt * 0.72)
                    * self._habitat_regrowth_modifier(x, y)
                )
                tile.food = min(1.0, tile.food + food_regrowth)
                tile.food = min(
                    tile.food,
                    max(0.04, self._food_capacity(x, y, season)),
                )

    def _habitat_regrowth_modifier(self, x: int, y: int) -> float:
        habitat_state = self._habitat_state_at(x, y)
        terrain = self.grid[y][x].terrain
        if habitat_state == "bloom":
            return 1.028
        if habitat_state == "flooded":
            return 1.012 if terrain == "wetland" else 0.986
        if habitat_state == "parched":
            return 0.972 if terrain == "rocky" else 0.94
        return 1.0

    def _terrain_regrowth_rate(self, terrain: str) -> float:
        resources = self.config.resources
        if terrain == "forest":
            return resources.forest_food_rate
        if terrain == "wetland":
            return resources.wetland_food_rate
        if terrain == "rocky":
            return resources.rocky_food_rate
        return resources.plain_food_rate

    def _terrain_growth_modifier(self, terrain: str, season: str) -> float:
        climate = self.config.climate
        if terrain == "forest":
            return (
                1.0 + climate.wet_forest_bonus
                if season == "wet"
                else 1.0 - climate.dry_forest_penalty
            )
        if terrain == "wetland":
            return (
                1.0 + climate.wet_forest_bonus * 0.85
                if season == "wet"
                else 1.0 - climate.dry_forest_penalty * 0.4
            )
        if terrain == "rocky":
            return (
                1.0 + climate.wet_plain_bonus * 0.35
                if season == "wet"
                else 1.0 - climate.dry_plain_penalty * 0.52
            )
        return (
            1.0 + climate.wet_plain_bonus
            if season == "wet"
            else 1.0 - climate.dry_plain_penalty
        )

    def _choose_action(self, agent: Agent) -> str:
        tile = self.grid[agent.y][agent.x]
        energy_ratio = self._energy_ratio(agent)
        hydration_ratio = self._hydration_ratio(agent)
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

        if fresh_kill_value > 0 and self._fresh_kill_intake_useful(agent) and (
            profile.role == "carnivore"
            or profile.meat_mode == "hunter"
            or fresh_kill_value >= max(plant_value, carcass_value) * (0.92 if energy_ratio < 0.72 else 1.04)
        ):
            return "eat"
        if carcass_value > 0 and self._carcass_intake_useful(agent) and (
            profile.role == "carnivore"
            or profile.meat_mode == "scavenger"
            or carcass_value >= plant_value * (0.92 if energy_ratio < 0.72 else 1.06)
        ):
            return "eat"
        if (
            self._has_water_access(agent)
            and hydration_ratio < 0.82
            and hydration_ratio <= energy_ratio + 0.08
        ):
            return "drink"
        prefer_biotic = (
            profile.role == "carnivore"
            or profile.meat_mode == "hunter"
            or profile.animal_drive > profile.plant_drive * 1.12
        )
        if prefer_biotic:
            biotic_target = self._best_visible_biotic_action(agent, profile=profile)
            if biotic_target is not None:
                return biotic_target
        if plant_value > 0 and self._plant_intake_useful(agent) and energy_ratio < 0.84 and (
            carcass_value <= 0
            or profile.role == "herbivore"
            or plant_value >= carcass_value * (0.9 if profile.role == "herbivore" else 1.04)
        ):
            return "eat"

        if not prefer_biotic:
            biotic_target = self._best_visible_biotic_action(agent, profile=profile)
            if biotic_target is not None:
                return biotic_target

        target = self._best_visible_action_toward_need(agent, profile=profile)
        if target is not None:
            return target

        return self.rng.choice(
            ["stay", "move_north", "move_south", "move_east", "move_west"]
        )

    def _best_visible_biotic_action(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> str | None:
        profile = profile or self._trophic_profile(agent)
        if profile.animal_share < self.config.trophic.animal_channel_threshold:
            return None

        energy_ratio = self._energy_ratio(agent)
        tile = self.grid[agent.y][agent.x]
        plant_value = self._plant_food_value(agent, tile, profile)
        fresh_kill_action = self._best_visible_fresh_kill_action(agent, profile=profile)
        carrion_action = self._best_visible_carrion_action(agent, profile=profile)
        attack_action = self._best_adjacent_attack_action(agent, profile=profile)
        prey_move_action = self._best_visible_prey_action(agent, profile=profile)

        if profile.role == "carnivore":
            if profile.meat_mode == "hunter":
                return attack_action or prey_move_action or fresh_kill_action or carrion_action
            if profile.meat_mode == "scavenger":
                return carrion_action or fresh_kill_action or attack_action or prey_move_action
            if profile.hunter_drive >= profile.scavenger_drive * 0.92:
                return attack_action or prey_move_action or fresh_kill_action or carrion_action
            return carrion_action or fresh_kill_action or attack_action or prey_move_action

        if profile.meat_mode == "hunter":
            if attack_action is not None and (
                profile.hunter_drive >= max(0.22, profile.plant_drive * 0.72)
                or plant_value < 0.12
            ):
                return attack_action
            if prey_move_action is not None and (
                profile.hunter_drive >= profile.plant_drive * 0.7
                or energy_ratio < 0.72
            ):
                return prey_move_action
            if fresh_kill_action is not None:
                return fresh_kill_action
            if carrion_action is not None and energy_ratio < 0.56:
                return carrion_action

        if fresh_kill_action is not None and (
            profile.hunter_drive >= profile.plant_drive * 0.9
            or energy_ratio < 0.62
        ):
            return fresh_kill_action
        if carrion_action is not None and (
            energy_ratio < 0.7
            or plant_value < 0.08
            or profile.scavenger_drive >= profile.plant_drive
        ):
            return carrion_action
        if attack_action is not None and profile.hunter_drive >= profile.plant_drive * 0.88:
            return attack_action
        if (
            attack_action is not None
            and profile.hunter_drive >= 0.24
            and energy_ratio < 0.6
            and plant_value < 0.1
        ):
            return attack_action
        if (
            prey_move_action is not None
            and profile.hunter_drive >= 0.26
            and energy_ratio < 0.54
            and plant_value < 0.08
        ):
            return prey_move_action
        return None

    def _best_visible_fresh_kill_action(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> str | None:
        if not self._can_consume_fresh_kill(agent):
            return None
        if not self._fresh_kill_intake_useful(agent):
            return None
        profile = profile or self._trophic_profile(agent)
        radius = max(
            1,
            self.config.default_vision_radius
            + (2 if profile.role == "carnivore" or profile.hunter_drive >= 0.18 else 0),
        )
        season = self._season_state()["name"]
        water_urgency = max(0.0, 1.0 - self._hydration_ratio(agent))
        food_urgency = max(0.0, 1.0 - self._energy_ratio(agent))
        best_target: tuple[int, int] | None = None
        best_score = float("-inf")
        for y in range(max(0, agent.y - radius), min(self.config.height, agent.y + radius + 1)):
            for x in range(max(0, agent.x - radius), min(self.config.width, agent.x + radius + 1)):
                distance = abs(x - agent.x) + abs(y - agent.y)
                if distance > radius:
                    continue
                tile = self.grid[y][x]
                if tile.terrain == "water" or tile.fresh_kill_energy <= 0:
                    continue
                if distance == 0:
                    return "eat"
                if tile.occupant_id is not None:
                    continue
                score = self._fresh_kill_food_value(agent, tile, profile)
                score += self._candidate_tile_score(
                    agent,
                    x,
                    y,
                    season,
                    water_urgency,
                    food_urgency,
                    profile=profile,
                    include_plant_channel=False,
                    include_vegetation_channel=False,
                    include_fresh_kill_channel=False,
                    include_carcass_channel=False,
                )
                score -= distance * 0.05
                if score > best_score:
                    best_score = score
                    best_target = (x, y)
        if best_target is None:
            return None
        return self._step_toward_target(
            agent,
            best_target[0],
            best_target[1],
            season,
            profile=profile,
            include_plant_channel=False,
            include_vegetation_channel=False,
            include_fresh_kill_channel=False,
            include_carcass_channel=False,
        )

    def _best_visible_carrion_action(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> str | None:
        if not self._can_consume_carcass(agent):
            return None
        if not self._carcass_intake_useful(agent):
            return None
        profile = profile or self._trophic_profile(agent)
        radius = max(
            1,
            self.config.default_vision_radius
            + (
                2
                if profile.role == "carnivore" or profile.scavenger_drive >= 0.18
                else 0
            ),
        )
        season = self._season_state()["name"]
        water_urgency = max(0.0, 1.0 - self._hydration_ratio(agent))
        food_urgency = max(0.0, 1.0 - self._energy_ratio(agent))
        best_target: tuple[int, int] | None = None
        best_score = float("-inf")
        for y in range(max(0, agent.y - radius), min(self.config.height, agent.y + radius + 1)):
            for x in range(max(0, agent.x - radius), min(self.config.width, agent.x + radius + 1)):
                distance = abs(x - agent.x) + abs(y - agent.y)
                if distance > radius:
                    continue
                tile = self.grid[y][x]
                if tile.terrain == "water" or tile.carcass_energy <= 0:
                    continue
                if distance == 0:
                    return "eat" if self._carcass_intake_useful(agent) else None
                if tile.occupant_id is not None:
                    continue
                score = self._carcass_food_value(agent, tile, profile)
                score += self._candidate_tile_score(
                    agent,
                    x,
                    y,
                    season,
                    water_urgency,
                    food_urgency,
                    profile=profile,
                    include_plant_channel=False,
                    include_vegetation_channel=False,
                    include_fresh_kill_channel=False,
                    include_carcass_channel=False,
                )
                score -= distance * 0.05
                if score > best_score:
                    best_score = score
                    best_target = (x, y)
        if best_target is None:
            return None
        return self._step_toward_target(
            agent,
            best_target[0],
            best_target[1],
            season,
            profile=profile,
            include_plant_channel=False,
            include_vegetation_channel=False,
            include_fresh_kill_channel=False,
            include_carcass_channel=False,
        )

    def _attack_damage(
        self,
        attacker: Agent,
        target: Agent,
        profile: TrophicProfile | None = None,
    ) -> float:
        return max(
            0.0,
            self.config.combat.base_attack_damage * self._attack_edge(attacker, target, profile),
        )

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
        if not self._can_attack(agent):
            return None
        profile = profile or self._trophic_profile(agent)
        best_action: str | None = None
        best_score = float("-inf")
        for action, dx, dy in self._movement_actions():
            target = (
                self.agents.get(self.grid[agent.y + dy][agent.x + dx].occupant_id)
                if self._in_bounds(agent.x + dx, agent.y + dy)
                else None
            )
            if target is None or not target.alive or target.agent_id == agent.agent_id:
                continue
            score = self._attack_value(agent, target, profile)
            if score <= 0.005:
                continue
            target_role = self._trophic_role(target)
            if target_role == "herbivore":
                score += 0.06
            if self._health_ratio(target) > self._health_ratio(agent) + 0.16:
                score -= 0.12
            if score > best_score:
                best_score = score
                best_action = action.replace("move_", "attack_")
        return best_action

    def _best_visible_prey_action(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> str | None:
        if not self._can_attack(agent):
            return None
        profile = profile or self._trophic_profile(agent)
        radius = max(
            1,
            self.config.default_vision_radius
            + (2 if profile.role == "carnivore" or profile.hunter_drive >= 0.18 else 0),
        )
        season = self._season_state()["name"]
        water_urgency = max(0.0, 1.0 - self._hydration_ratio(agent))
        food_urgency = max(0.0, 1.0 - self._energy_ratio(agent))
        best_target: tuple[int, int] | None = None
        best_score = float("-inf")
        for y in range(max(0, agent.y - radius), min(self.config.height, agent.y + radius + 1)):
            for x in range(max(0, agent.x - radius), min(self.config.width, agent.x + radius + 1)):
                distance = abs(x - agent.x) + abs(y - agent.y)
                if distance <= 1 or distance > radius:
                    continue
                tile = self.grid[y][x]
                target_id = tile.occupant_id
                if tile.terrain == "water" or target_id is None or target_id == agent.agent_id:
                    continue
                target = self.agents.get(target_id)
                if target is None or not target.alive:
                    continue
                target_role = self._trophic_role(target)
                score = self._attack_value(agent, target, profile)
                score += (
                    self._prey_vulnerability(target)
                    * self.config.biotic_fields.hunter_vulnerability_weight
                )
                score += self._candidate_tile_score(
                    agent,
                    x,
                    y,
                    season,
                    water_urgency,
                    food_urgency,
                    profile=profile,
                    include_plant_channel=False,
                    include_vegetation_channel=False,
                    include_fresh_kill_channel=False,
                    include_carcass_channel=False,
                )
                if target_role == "herbivore":
                    score += 0.08
                score -= distance * 0.025
                if score > best_score:
                    best_score = score
                    best_target = (x, y)
        if best_target is None:
            return None
        return self._step_toward_target(
            agent,
            best_target[0],
            best_target[1],
            season,
            profile=profile,
            include_plant_channel=False,
            include_vegetation_channel=False,
            include_fresh_kill_channel=False,
            include_carcass_channel=False,
        )

    def _biotic_opportunity_score(
        self,
        agent: Agent,
        x: int,
        y: int,
        profile: TrophicProfile,
    ) -> float:
        return self._biotic_field_score(agent, x, y, profile)

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
    ) -> str | None:
        profile = profile or self._trophic_profile(agent)
        water_urgency = max(0.0, 1.0 - self._hydration_ratio(agent))
        food_urgency = max(0.0, 1.0 - self._energy_ratio(agent))
        current_distance = abs(target_x - agent.x) + abs(target_y - agent.y)
        best_action: str | None = None
        best_score = float("-inf")
        for action, dx, dy in self._movement_actions():
            x = agent.x + dx
            y = agent.y + dy
            if not self._can_move_to(x, y):
                continue
            next_distance = abs(target_x - x) + abs(target_y - y)
            if next_distance >= current_distance:
                continue
            score = self._candidate_tile_score(
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
            )
            score -= next_distance * 0.04
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

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
    ) -> float:
        profile = profile or self._trophic_profile(agent)
        tile = self.grid[y][x]
        score = 0.0
        water_reason = self._water_access_reason(x, y)
        refuge_score = self._refuge_score(x, y)
        hazard_type, hazard_level = self._hazard_at(x, y)
        if water_reason != "none":
            hard_water_bonus = {
                "adjacent_water": 1.52,
                "wetland": 1.44,
                "flooded": 1.14,
            }.get(water_reason, 1.4)
            score += hard_water_bonus * water_urgency * agent.genome.water_efficiency
        else:
            score += refuge_score * 0.16 * water_urgency
            if self._soft_refuge_reason(x, y) == "canopy_refuge":
                score += 0.06 * water_urgency
        if include_plant_channel and profile.plant_drive > 0:
            score += self._plant_food_value(agent, tile, profile, x=x, y=y) * (
                0.28 + food_urgency * 0.84
            )
        score += self._terrain_preference_score(agent, tile.terrain)
        score += self._field_preference_score(
            agent,
            x,
            y,
            season,
            water_urgency,
            food_urgency,
            profile=profile,
        )
        if include_vegetation_channel and profile.plant_drive > 0:
            score += tile.vegetation * profile.plant_drive * (0.05 + food_urgency * 0.14)
            score += (1.0 - tile.recovery_debt) * profile.plant_drive * 0.08
        score -= hazard_level * (0.14 + (1.0 - self._health_ratio(agent)) * 0.34)
        if hazard_type == "exposure" and self._soft_refuge_reason(x, y) == "canopy_refuge":
            score += 0.05
        if (
            include_fresh_kill_channel
            and tile.fresh_kill_energy > 0
            and self._can_consume_fresh_kill(agent)
        ):
            score += self._fresh_kill_food_value(agent, tile, profile) * (0.52 + food_urgency * 1.12)
        if include_carcass_channel and tile.carcass_energy > 0 and self._can_consume_carcass(agent):
            score += self._carcass_food_value(agent, tile, profile) * (0.4 + food_urgency * 1.1)
        if include_animal_signal and (profile.role == "herbivore" or profile.animal_drive > 0):
            score += self._biotic_opportunity_score(agent, x, y, profile) * (
                0.22 + food_urgency * 0.78
            )
        return score

    def _best_visible_action_toward_need(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> str | None:
        profile = profile or self._trophic_profile(agent)
        energy_ratio = agent.energy / agent.genome.max_energy
        hydration_ratio = agent.hydration / agent.genome.max_hydration
        water_urgency = max(0.0, 1.0 - hydration_ratio)
        food_urgency = max(0.0, 1.0 - energy_ratio)
        season = self._season_state()["name"]
        radius = max(1, self.config.default_vision_radius)
        best_target: tuple[int, int] | None = None
        best_target_score = float("-inf")

        for y in range(max(0, agent.y - radius), min(self.config.height, agent.y + radius + 1)):
            for x in range(max(0, agent.x - radius), min(self.config.width, agent.x + radius + 1)):
                distance = abs(x - agent.x) + abs(y - agent.y)
                if distance == 0 or distance > radius:
                    continue
                tile = self.grid[y][x]
                if tile.terrain == "water":
                    continue
                if tile.occupant_id is not None and tile.occupant_id != agent.agent_id:
                    continue
                score = self._candidate_tile_score(
                    agent,
                    x,
                    y,
                    season,
                    water_urgency,
                    food_urgency,
                    profile=profile,
                )
                score -= distance * (0.05 + agent.genome.move_cost * 1.45)
                if score > best_target_score:
                    best_target_score = score
                    best_target = (x, y)

        if best_target is None:
            return None

        target_x, target_y = best_target
        current_distance = abs(target_x - agent.x) + abs(target_y - agent.y)
        best_action: str | None = None
        best_score = float("-inf")
        for action, dx, dy in self._movement_actions():
            x = agent.x + dx
            y = agent.y + dy
            if not self._can_move_to(x, y):
                continue
            next_distance = abs(target_x - x) + abs(target_y - y)
            if next_distance >= current_distance:
                continue
            score = self._candidate_tile_score(
                agent,
                x,
                y,
                season,
                water_urgency,
                food_urgency,
                profile=profile,
            )
            score -= next_distance * 0.04
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _resolve_action(self, agent: Agent, action: str) -> bool:
        if action == "eat":
            return self._eat(agent)
        if action == "drink":
            return self._drink(agent)
        if action == "stay":
            return False
        if action.startswith("attack_"):
            for candidate, dx, dy in self._movement_actions():
                if action != candidate.replace("move_", "attack_"):
                    continue
                return self._attack(agent, agent.x + dx, agent.y + dy)
            return False

        for candidate, dx, dy in self._movement_actions():
            if candidate != action:
                continue
            nx = agent.x + dx
            ny = agent.y + dy
            if not self._can_move_to(nx, ny):
                return False
            self.grid[agent.y][agent.x].occupant_id = None
            agent.x = nx
            agent.y = ny
            self.grid[agent.y][agent.x].occupant_id = agent.agent_id
            self._emit(
                EventType.AGENT_MOVED,
                agent_id=agent.agent_id,
                data={"x": agent.x, "y": agent.y, "action": action},
            )
            return True

        return False

    def _eat(self, agent: Agent) -> bool:
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
            fresh_kill_value > 0
            and self._fresh_kill_intake_useful(agent)
            and fresh_kill_value >= max(plant_value, carcass_value) * 0.92
        ):
            return self._consume_fresh_kill(agent, profile=profile)
        if (
            carcass_value > 0
            and self._carcass_intake_useful(agent)
            and carcass_value >= plant_value * 0.92
        ):
            return self._consume_carcass(agent, profile=profile)
        if plant_value > 0 and self._plant_intake_useful(agent):
            return self._consume_plant(agent, profile=profile)
        if fresh_kill_value > 0 and self._fresh_kill_intake_useful(agent):
            return self._consume_fresh_kill(agent, profile=profile)
        if carcass_value > 0 and self._carcass_intake_useful(agent):
            return self._consume_carcass(agent, profile=profile)
        return False

    def _consume_plant(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> bool:
        profile = profile or self._trophic_profile(agent)
        if not self._plant_intake_useful(agent):
            return False
        tile = self.grid[agent.y][agent.x]
        consumed = min(tile.food, self.config.resources.eat_amount)
        if consumed <= 0:
            return False
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
        gained = agent.energy - energy_before
        self._record_feeding_event(
            agent,
            "plant",
            consumed,
            gained,
            profile,
            energy_before=energy_before,
            energy_after=agent.energy,
            potential_energy=potential_gain,
        )
        self._emit(
            EventType.AGENT_ATE,
            agent_id=agent.agent_id,
            data={
                "food_source": "plant",
                "consumed": round(consumed, 4),
                "energy_before": round(energy_before, 4),
                "energy": round(agent.energy, 4),
                "gained_energy": round(gained, 4),
                "potential_gained_energy": round(potential_gain, 4),
                "trophic_role": profile.role,
                "meat_mode": profile.meat_mode,
                "matched_diet_ratio": round(self._matched_diet_ratio(agent, profile), 4),
                "vegetation": round(tile.vegetation, 4),
                "recovery_debt": round(tile.recovery_debt, 4),
                "shelter": round(tile.shelter, 4),
            },
        )
        return False

    def _consume_fresh_kill(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
        *,
        immediate_kill_feed: bool = False,
        source_x: int | None = None,
        source_y: int | None = None,
    ) -> bool:
        profile = profile or self._trophic_profile(agent)
        if not self._fresh_kill_intake_useful(agent):
            return False
        x = agent.x if source_x is None else source_x
        y = agent.y if source_y is None else source_y
        tile = self.grid[y][x]
        consume_info = self._consume_fresh_kill_from_tile(
            tile,
            min(tile.fresh_kill_energy, self.config.resources.eat_amount * 1.1),
        )
        consumed = consume_info["consumed"]
        if consumed <= 0:
            return False
        self._apply_meat_intake(
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
        return False

    def _consume_carcass(
        self,
        agent: Agent,
        profile: TrophicProfile | None = None,
    ) -> bool:
        profile = profile or self._trophic_profile(agent)
        if not self._carcass_intake_useful(agent):
            return False
        tile = self.grid[agent.y][agent.x]
        consume_info = self._consume_carcass_from_tile(
            tile,
            min(tile.carcass_energy, self.config.resources.eat_amount * 1.25),
        )
        consumed = consume_info["consumed"]
        if consumed <= 0:
            return False
        freshness = float(consume_info["avg_freshness"])
        self._apply_meat_intake(
            agent,
            food_source="carcass",
            consumed=consumed,
            potential_nutrition=self._carcass_nutrition(agent, consumed, freshness, profile),
            profile=profile,
            x=agent.x,
            y=agent.y,
            source_breakdown=consume_info["source_breakdown"],
            deposit_breakdown=consume_info["deposit_breakdown"],
            immediate_kill_feed=False,
            freshness=freshness,
        )
        return False

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
    ) -> None:
        energy_before = agent.energy
        agent.energy = min(agent.genome.max_energy, agent.energy + potential_nutrition)
        nutrition = agent.energy - energy_before
        healed = consumed * self.config.carcasses.healing_fraction * agent.genome.healing_efficiency
        if healed > 0:
            previous_health = agent.health
            agent.health = min(agent.max_health, agent.health + healed)
            if agent.health > previous_health:
                agent.injury_load = self._clamp01(
                    max(
                        0.0,
                        agent.injury_load - (agent.health - previous_health) / max(agent.max_health, 1e-9),
                    )
                )
        event_bucket = (
            self.tick_fresh_kill_events if food_source == "fresh_kill" else self.tick_carcass_events
        )
        event_bucket.append(
            {
                "agent_id": agent.agent_id,
                "species_id": self._species_id_for_agent(agent.agent_id),
                "consumed": round(consumed, 4),
                "energy": round(consumed, 4),
                "gained_energy": round(nutrition, 4),
                "potential_gained_energy": round(potential_nutrition, 4),
                "meat_mode": profile.meat_mode,
                "x": x,
                "y": y,
                "source_breakdown": source_breakdown,
                "immediate_kill_feed": immediate_kill_feed,
            }
        )
        if food_source == "fresh_kill":
            self.run_fresh_kill_totals["consumption_events"] += 1
            self.run_fresh_kill_totals["energy_consumed"] += consumed
            self.run_fresh_kill_totals["gained_energy"] += nutrition
        else:
            self.run_carcass_totals["consumption_events"] += 1
            self.run_carcass_totals["energy_consumed"] += consumed
            self.run_carcass_totals["gained_energy"] += nutrition
        self._record_feeding_event(
            agent,
            food_source,
            consumed,
            nutrition,
            profile,
            energy_before=energy_before,
            energy_after=agent.energy,
            potential_energy=potential_nutrition,
        )
        tile_state = (
            self._fresh_kill_tile_summary_for_position(x, y)
            if food_source == "fresh_kill"
            else self._carcass_tile_summary_for_position(x, y)
        )
        self._emit(
            EventType.AGENT_ATE,
            agent_id=agent.agent_id,
            data={
                "food_source": food_source,
                "consumed": round(consumed, 4),
                "energy_before": round(energy_before, 4),
                "energy": round(agent.energy, 4),
                "gained_energy": round(nutrition, 4),
                "potential_gained_energy": round(potential_nutrition, 4),
                "trophic_role": profile.role,
                "meat_mode": profile.meat_mode,
                "immediate_kill_feed": immediate_kill_feed,
                "health": round(agent.health, 4),
                "matched_diet_ratio": round(self._matched_diet_ratio(agent, profile), 4),
                "x": x,
                "y": y,
                "source_breakdown": source_breakdown,
                "deposit_breakdown": deposit_breakdown,
                "tile_mixed_sources_after": tile_state["mixed_sources"],
                "tile_dominant_source_species_after": tile_state["dominant_source_species"],
                "tile_source_breakdown_after": tile_state["source_breakdown"],
            },
        )
        if freshness is not None:
            self.events[-1].data["freshness"] = round(freshness, 4)
            self.events[-1].data["tile_carcass_energy_after"] = tile_state["total_energy"]
            self.events[-1].data["tile_avg_freshness_after"] = tile_state["avg_freshness"]
        else:
            self.events[-1].data["tile_fresh_kill_energy_after"] = tile_state["total_energy"]

    def _drink(self, agent: Agent) -> bool:
        water_reason = self._water_access_reason(agent.x, agent.y)
        if water_reason == "none":
            return False
        gained = self.config.resources.drink_amount * agent.genome.water_efficiency
        agent.hydration = min(agent.genome.max_hydration, agent.hydration + gained)
        self._emit(
            EventType.AGENT_DRANK,
            agent_id=agent.agent_id,
            data={
                "hydration": round(agent.hydration, 4),
                "water_access_reason": water_reason,
            },
        )
        return False

    def _attack(self, attacker: Agent, target_x: int, target_y: int) -> bool:
        if not self._can_attack(attacker) or not self._in_bounds(target_x, target_y):
            return False
        target_id = self.grid[target_y][target_x].occupant_id
        if target_id is None or target_id == attacker.agent_id:
            return False
        target = self.agents.get(target_id)
        if target is None or not target.alive:
            return False

        profile = self._trophic_profile(attacker)
        attack_cost_multiplier = attacker.genome.attack_cost_multiplier
        attacker.energy -= self.config.combat.attack_energy_cost * attack_cost_multiplier
        attacker.hydration -= self.config.combat.attack_hydration_cost * attack_cost_multiplier

        damage = self._attack_damage(attacker, target, profile)
        success = damage >= 0.025
        kill = False
        if success:
            self._apply_damage(
                target,
                damage,
                source="attack",
                attacker_id=attacker.agent_id,
            )
            kill = target.health <= 0 and target.alive
            if kill:
                self._kill_agent(target, cause="attack", killer_id=attacker.agent_id)
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
            if self._can_consume_fresh_kill(attacker) and self._fresh_kill_intake_useful(attacker):
                self._consume_fresh_kill(
                    attacker,
                    profile=profile,
                    immediate_kill_feed=True,
                    source_x=target_x,
                    source_y=target_y,
                )
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
        return False

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
        season = self._season_state()["name"]
        move_cost = (
            agent.genome.move_cost * (1.0 + agent.injury_load * 0.42) if moved else 0.0
        )
        profile = self._trophic_profile(agent)
        energy_modifier = self._agent_energy_drain_modifier(agent, season)
        hydration_modifier = self._agent_hydration_drain_modifier(agent, season)
        energy_modifier *= 1.0 + agent.injury_load * 0.14
        hydration_modifier *= 1.0 + agent.injury_load * 0.08
        energy_modifier *= 1.0 + profile.breadth * self.config.trophic.breadth_metabolism_penalty
        hydration_modifier *= 1.0 + profile.breadth * self.config.trophic.breadth_hydration_penalty

        base_energy_cost = self.config.base_energy_drain + move_cost
        agent.energy -= base_energy_cost * energy_modifier
        agent.hydration -= (
            self.config.base_hydration_drain
            * hydration_modifier
            + (0.004 if moved else 0.0)
        )

    def _apply_health_and_hazards(self, agent: Agent, moved: bool) -> None:
        if not agent.alive:
            return

        hazard_type, hazard_level = self._hazard_at(agent.x, agent.y)
        if hazard_type != "none" and hazard_level > 0:
            tile = self.grid[agent.y][agent.x]
            if hazard_type == "exposure":
                resistance = (
                    agent.genome.heat_tolerance * 0.16
                    + tile.shelter * 0.18
                    + self._refuge_score(agent.x, agent.y) * 0.12
                )
                damage = self.config.hazards.exposure_damage_rate * hazard_level * max(
                    0.42, 1.0 - resistance
                )
            else:
                resistance = agent.genome.defense_rating * 0.14 + tile.shelter * 0.06
                if tile.terrain == "rocky":
                    resistance += agent.genome.rocky_affinity * 0.08
                damage = self.config.hazards.instability_damage_rate * hazard_level * max(
                    0.46, 1.0 - resistance
                )
                if moved:
                    damage *= 1.08
            if damage > 0:
                self.tick_hazard_exposure_agents.add(agent.agent_id)
                self._apply_damage(
                    agent,
                    damage,
                    source=f"hazard_{hazard_type}",
                    hazard_type=hazard_type,
                )
                if agent.health <= 0 and agent.alive:
                    self._kill_agent(agent, cause=f"hazard_{hazard_type}")
                    return

        if agent.health >= agent.max_health:
            agent.injury_load = self._clamp01(max(0.0, agent.injury_load - 0.004))
            return

        hazards = self.config.hazards
        if (
            self._energy_ratio(agent) >= hazards.min_energy_ratio_for_healing
            and self._hydration_ratio(agent) >= hazards.min_hydration_ratio_for_healing
            and hazard_level < hazards.healing_hazard_threshold
        ):
            heal_amount = (
                hazards.healing_base_rate
                * agent.genome.healing_efficiency
                * (0.44 + self._energy_ratio(agent) * 0.28 + self._hydration_ratio(agent) * 0.28)
                * (1.0 - hazard_level * 0.6)
            )
            previous = agent.health
            agent.health = min(agent.max_health, agent.health + heal_amount)
            if agent.health > previous:
                healed = agent.health - previous
                agent.injury_load = self._clamp01(
                    max(0.0, agent.injury_load - healed / max(agent.max_health, 1e-9) * 0.84)
                )
                self._emit(
                    EventType.AGENT_HEALED,
                    agent_id=agent.agent_id,
                    data={
                        "amount": round(healed, 4),
                        "health": round(agent.health, 4),
                    },
                )

    def _apply_damage(
        self,
        agent: Agent,
        amount: float,
        source: str,
        hazard_type: str | None = None,
        attacker_id: int | None = None,
    ) -> None:
        if amount <= 0 or not agent.alive:
            return
        agent.health -= amount
        agent.injury_load = self._clamp01(agent.injury_load + amount / max(agent.max_health, 1e-9))
        agent.last_damage_source = source
        self.tick_damage_events.append(
            {
                "agent_id": agent.agent_id,
                "amount": round(amount, 4),
                "source": source,
                "hazard_type": hazard_type,
                "attacker_id": attacker_id,
            }
        )
        self.run_combat_totals["damage_taken"] += amount
        if source.startswith("hazard_"):
            self.run_combat_totals["hazard_damage_taken"] += amount
        self._emit(
            EventType.AGENT_DAMAGED,
            agent_id=agent.agent_id,
            data={
                "amount": round(amount, 4),
                "health": round(agent.health, 4),
                "source": source,
                "hazard_type": hazard_type,
                "attacker_id": attacker_id,
            },
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
        return len(self.alive_agents()) < self.config.max_agents and self._is_reproduction_ready(agent)

    def _is_reproduction_ready(self, agent: Agent) -> bool:
        profile = self._trophic_profile(agent)
        matched_diet_ratio = self._matched_diet_ratio(agent, profile)
        return (
            agent.age >= self.config.reproduction.min_age
            and self.tick - agent.last_reproduction_tick >= self.config.reproduction.cooldown_ticks
            and agent.energy
            >= agent.reproduction_threshold()
            * (1.0 + profile.breadth * self.config.trophic.breadth_reproduction_penalty)
            and agent.hydration
            >= agent.genome.max_hydration * self.config.reproduction.min_hydration_fraction
            and self._health_ratio(agent) >= self.config.combat.min_reproduction_health_ratio
            and matched_diet_ratio >= self._matched_diet_threshold(profile)
            and self._has_empty_neighbor(agent.x, agent.y)
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

    def _reproduce(self, parent: Agent) -> bool:
        destination = self._find_empty_neighbor(parent.x, parent.y)
        if destination is None:
            return False

        child_genome = parent.genome.mutate(self.rng)
        child = Agent(
            agent_id=self.next_agent_id,
            parent_id=parent.agent_id,
            lineage_id=parent.lineage_id,
            birth_tick=self.tick,
            death_tick=None,
            x=destination[0],
            y=destination[1],
            energy=child_genome.max_energy * self.config.reproduction.child_energy_fraction,
            hydration=child_genome.max_hydration * self.config.reproduction.child_hydration_fraction,
            health=child_genome.max_health * self.config.reproduction.child_health_fraction,
            max_health=child_genome.max_health,
            injury_load=0.0,
            age=0,
            alive=True,
            last_reproduction_tick=-10_000,
            last_damage_source="none",
            recent_plant_energy=0.0,
            recent_fresh_kill_energy=0.0,
            recent_carcass_energy=0.0,
            genome_vector=genome_vector(child_genome),
            genome=child_genome,
        )
        parent.energy -= self.config.reproduction.energy_cost
        parent.last_reproduction_tick = self.tick
        self._place_agent(child)
        self.next_agent_id += 1
        self.births += 1
        self.last_birth_tick = self.tick
        self.tick_birth_pairs.append((parent.agent_id, child.agent_id))
        self._invalidate_biotic_state()
        self._emit(
            EventType.AGENT_REPRODUCED,
            agent_id=parent.agent_id,
            data={
                "child_id": child.agent_id,
                "child_x": child.x,
                "child_y": child.y,
                "lineage_id": child.lineage_id,
                "birth_tick": child.birth_tick,
            },
        )
        return True

    def _should_die(self, agent: Agent) -> bool:
        return (
            agent.energy <= 0
            or agent.hydration <= 0
            or agent.health <= 0
            or agent.age >= self.config.max_age
        )

    def _death_cause(self, agent: Agent) -> str:
        if agent.health <= 0:
            return agent.last_damage_source or "health_depletion"
        if agent.energy <= 0:
            return "energy_depletion"
        if agent.hydration <= 0:
            return "hydration_depletion"
        if agent.age >= self.config.max_age:
            return "old_age"
        return "unknown"

    def _kill_agent(
        self,
        agent: Agent,
        cause: str | None = None,
        killer_id: int | None = None,
    ) -> None:
        death_cause = cause or self._death_cause(agent)
        agent.alive = False
        agent.death_tick = self.tick
        self.grid[agent.y][agent.x].occupant_id = None
        carcass_energy = 0.0
        tile = self.grid[agent.y][agent.x]
        source_species = self._species_id_for_agent(agent.agent_id)
        tile_state = self._carcass_tile_summary_for_position(agent.x, agent.y)
        fresh_kill_state = self._fresh_kill_tile_summary_for_position(agent.x, agent.y)
        if tile.terrain != "water":
            carcass_energy = self._project_carcass_yield(agent)
            if death_cause == "attack" and killer_id is not None:
                fresh_kill_state = self._deposit_fresh_kill(
                    tile,
                    x=agent.x,
                    y=agent.y,
                    energy=carcass_energy,
                    source_species=source_species,
                    source_agent_id=agent.agent_id,
                    killer_id=killer_id,
                )
            else:
                tile_state = self._deposit_carcass(
                    tile,
                    x=agent.x,
                    y=agent.y,
                    energy=carcass_energy,
                    source_species=source_species,
                    source_agent_id=agent.agent_id,
                    cause=death_cause,
                    killer_id=killer_id,
                )
        self.deaths += 1
        self.tick_death_agent_ids.append(agent.agent_id)
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
                "carcass_energy": round(carcass_energy, 4),
                "fresh_kill_energy": round(fresh_kill_state["total_energy"], 4),
                "tile_carcass_energy_after": tile_state["total_energy"],
                "tile_avg_freshness_after": tile_state["avg_freshness"],
                "tile_deposit_count_after": tile_state["deposit_count"],
                "tile_mixed_sources_after": tile_state["mixed_sources"],
                "tile_dominant_source_species_after": tile_state["dominant_source_species"],
                "tile_source_breakdown_after": tile_state["source_breakdown"],
                "tile_fresh_kill_energy_after": fresh_kill_state["total_energy"],
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
        return self._tile_has_water_access(agent.x, agent.y)

    def _tile_has_water_access(self, x: int, y: int) -> bool:
        return self._water_access_reason(x, y) != "none"

    def _adjacent_to_water(self, x: int, y: int) -> bool:
        for _, dx, dy in self._movement_actions():
            nx = x + dx
            ny = y + dy
            if self._in_bounds(nx, ny) and self.grid[ny][nx].terrain == "water":
                return True
        return False

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

    def _capture_frame(self, births_this_tick: int, deaths_this_tick: int) -> None:
        alive = sorted(self.alive_agents(), key=lambda agent: agent.agent_id)
        species_map, species_records = self._lineage_species_snapshot(alive)
        self.current_species_map = species_map
        self.current_species_records = species_records
        ecotype_map, ecotype_records = self._cluster_ecotypes()
        self.current_ecotype_map = ecotype_map
        self.current_ecotype_records = ecotype_records
        species_metrics = self._build_species_metrics(
            alive,
            species_map,
            self.agent_last_species_map,
        )
        ecotype_metrics = self._build_species_metrics(
            alive,
            ecotype_map,
            self.agent_last_ecotype_map,
        )
        trait_means = self._trait_means(alive)
        ecology_codes, ecology_counts, ecology_stats = self._ecology_snapshot()
        hazard_type_codes, hazard_level_codes, hazard_counts, hazard_stats = self._hazard_snapshot()
        biotic_fields, biotic_field_stats = self._biotic_field_snapshot()
        fresh_kill_energy_codes, fresh_kill_stats = self._fresh_kill_snapshot()
        carcass_energy_codes, carcass_freshness_codes, carcass_stats = self._carcass_snapshot()
        fresh_kill_patches = self._fresh_kill_patch_summaries()
        carcass_patches = self._carcass_patch_summaries()
        (
            hydrology_primary_codes,
            hydrology_support_codes,
            hydrology_primary_counts,
            hydrology_support_counts,
            hydrology_primary_stats,
        ) = self._hydrology_snapshot()
        refuge_codes, refuge_score_codes, refuge_counts, refuge_stats = self._refuge_snapshot()
        season = self._season_state()["name"]
        combat_stats = {
            "attack_attempts": len(self.tick_attack_events),
            "successful_attacks": sum(1 for event in self.tick_attack_events if event["success"]),
            "kills": sum(1 for event in self.tick_attack_events if event["kill"]),
            "damage_dealt": round(
                sum(float(event["damage"]) for event in self.tick_attack_events if event["success"]),
                4,
            ),
            "attack_damage_taken": round(
                sum(
                    float(event["amount"])
                    for event in self.tick_damage_events
                    if event["source"] == "attack"
                ),
                4,
            ),
            "hazard_damage_taken": round(
                sum(
                    float(event["amount"])
                    for event in self.tick_damage_events
                    if str(event["source"]).startswith("hazard_")
                ),
                4,
            ),
            "fresh_kill_consumption_events": len(self.tick_fresh_kill_events),
            "fresh_kill_energy_consumed": round(
                sum(float(event["energy"]) for event in self.tick_fresh_kill_events),
                4,
            ),
            "fresh_kill_gained_energy": round(
                sum(float(event["gained_energy"]) for event in self.tick_fresh_kill_events),
                4,
            ),
            "carcass_consumption_events": len(self.tick_carcass_events),
            "carcass_energy_consumed": round(
                sum(float(event["energy"]) for event in self.tick_carcass_events),
                4,
            ),
            "carcass_gained_energy": round(
                sum(float(event["gained_energy"]) for event in self.tick_carcass_events),
                4,
            ),
        }
        fresh_kill_flow = {
            "deposition_events": len(self.tick_fresh_kill_deposit_events),
            "fresh_kill_energy_deposited": round(
                sum(float(event["deposited_energy"]) for event in self.tick_fresh_kill_deposit_events),
                4,
            ),
            "fresh_kill_energy_converted_to_carcass": round(
                self.tick_fresh_kill_to_carcass_energy,
                4,
            ),
            "consumption_events": len(self.tick_fresh_kill_events),
            "fresh_kill_energy_consumed": round(
                sum(float(event["energy"]) for event in self.tick_fresh_kill_events),
                4,
            ),
            "fresh_kill_gained_energy": round(
                sum(float(event["gained_energy"]) for event in self.tick_fresh_kill_events),
                4,
            ),
        }
        carcass_flow = {
            "deposition_events": len(self.tick_carcass_deposit_events),
            "carcass_energy_deposited": round(
                sum(float(event["deposited_energy"]) for event in self.tick_carcass_deposit_events),
                4,
            ),
            "carcass_energy_decayed": round(self.tick_carcass_energy_decayed, 4),
            "consumption_events": len(self.tick_carcass_events),
            "carcass_energy_consumed": round(
                sum(float(event["energy"]) for event in self.tick_carcass_events),
                4,
            ),
            "carcass_gained_energy": round(
                sum(float(event["gained_energy"]) for event in self.tick_carcass_events),
                4,
            ),
        }
        diet_totals = self._empty_diet_totals()
        diet_by_trophic_role = self._empty_grouped_diet_totals(
            [role for role in TROPHIC_ROLE_CODES if role != "none"]
        )
        diet_by_meat_mode = self._empty_grouped_diet_totals(MEAT_MODE_CODES)
        for event in self.tick_feeding_events:
            food_source = str(event["food_source"])
            gained_energy = float(event["gained_energy"])
            trophic_role = str(event["trophic_role"])
            meat_mode = str(event["meat_mode"])
            self._accumulate_diet_totals(diet_totals, food_source, gained_energy)
            self._accumulate_diet_totals(
                diet_by_trophic_role[trophic_role],
                food_source,
                gained_energy,
            )
            self._accumulate_diet_totals(
                diet_by_meat_mode[meat_mode],
                food_source,
                gained_energy,
            )
        diet_stats = self._finalize_diet_totals(diet_totals)
        frame_diet_by_trophic_role = self._finalize_grouped_diet_totals(diet_by_trophic_role)
        frame_diet_by_meat_mode = self._finalize_grouped_diet_totals(diet_by_meat_mode)
        trophic_role_counts, meat_mode_counts = self._population_trophic_counts(alive)
        self.run_fresh_kill_totals["fresh_kill_tiles"] = fresh_kill_stats["fresh_kill_tiles"]
        self.run_fresh_kill_totals["total_fresh_kill_energy"] = fresh_kill_stats["total_fresh_kill_energy"]
        self.run_carcass_totals["carcass_tiles"] = carcass_stats["carcass_tiles"]
        self.run_carcass_totals["total_carcass_energy"] = carcass_stats["total_carcass_energy"]
        self.viewer_frames.append(
            {
                "tick": self.tick,
                "season": season,
                "field_state": self._climate_state(),
                "biotic_fields": biotic_fields,
                "biotic_field_stats": biotic_field_stats,
                "habitat_state_counts": self._habitat_state_grid()[1],
                "habitat_state_codes": [
                    [HABITAT_STATE_CODES[state] for state in row]
                    for row in self._habitat_state_grid()[0]
                ],
                "hydrology_primary_counts": hydrology_primary_counts,
                "hydrology_primary_codes": hydrology_primary_codes,
                "hydrology_primary_stats": hydrology_primary_stats,
                "hydrology_support_counts": hydrology_support_counts,
                "hydrology_support_codes": hydrology_support_codes,
                "refuge_counts": refuge_counts,
                "refuge_codes": refuge_codes,
                "refuge_score_codes": refuge_score_codes,
                "refuge_stats": refuge_stats,
                "hazard_counts": hazard_counts,
                "hazard_type_codes": hazard_type_codes,
                "hazard_level_codes": hazard_level_codes,
                "hazard_stats": hazard_stats,
                "fresh_kill_energy_codes": fresh_kill_energy_codes,
                "fresh_kill_stats": fresh_kill_stats,
                "fresh_kill_patches": fresh_kill_patches,
                "fresh_kill_flow": fresh_kill_flow,
                "carcass_energy_codes": carcass_energy_codes,
                "carcass_freshness_codes": carcass_freshness_codes,
                "carcass_stats": carcass_stats,
                "carcass_patches": carcass_patches,
                "carcass_flow": carcass_flow,
                "combat_stats": combat_stats,
                "diet_stats": diet_stats,
                "diet_by_trophic_role": frame_diet_by_trophic_role,
                "diet_by_meat_mode": frame_diet_by_meat_mode,
                "trophic_role_counts": trophic_role_counts,
                "meat_mode_counts": meat_mode_counts,
                "trophic_role_codes": self._trophic_role_grid(alive),
                "meat_mode_codes": self._meat_mode_grid(alive),
                "ecology_state_counts": ecology_counts,
                "ecology_state_codes": ecology_codes,
                "ecology_stats": ecology_stats,
                "alive_agents": len(alive),
                "births": births_this_tick,
                "deaths": deaths_this_tick,
                "trait_means": trait_means,
                "species_metrics": species_metrics,
                "ecotype_metrics": ecotype_metrics,
                "species_counts": [
                    [record.species_id, record.member_count] for record in species_records
                ],
                "ecotype_counts": [
                    [record.species_id, record.member_count] for record in ecotype_records
                ],
                "agents": [
                    [
                        agent.agent_id,
                        agent.x,
                        agent.y,
                        round(agent.energy, 4),
                        round(self._energy_ratio(agent), 4),
                        round(agent.hydration, 4),
                        round(self._hydration_ratio(agent), 4),
                        round(agent.health, 4),
                        round(self._health_ratio(agent), 4),
                        round(agent.injury_load, 4),
                        agent.age,
                        round(self._agent_energy_drain_modifier(agent, season), 4),
                        round(self._agent_hydration_drain_modifier(agent, season), 4),
                        round(self.grid[agent.y][agent.x].vegetation, 4),
                        round(self.grid[agent.y][agent.x].recovery_debt, 4),
                        int(self._is_reproduction_ready(agent)),
                        self._trophic_role(agent),
                        self._meat_mode(agent),
                        agent.last_damage_source,
                        self._water_access_reason(agent.x, agent.y),
                        self._soft_refuge_reason(agent.x, agent.y),
                        self._hydrology_support_code(agent.x, agent.y),
                        round(self._refuge_score(agent.x, agent.y), 4),
                        round(self._matched_diet_ratio(agent), 4),
                        ecotype_map.get(agent.agent_id, 0),
                        species_map.get(agent.agent_id, 0),
                    ]
                    for agent in alive
                ],
            }
        )
        self.agent_last_species_map = species_map.copy()
        self.agent_last_ecotype_map = ecotype_map.copy()

    def _build_species_metrics(
        self,
        alive: list[Agent],
        species_map: dict[int, int],
        previous_species_map: dict[int, int],
    ) -> dict[str, dict[str, object]]:
        metrics: dict[int, dict[str, object]] = {}
        births_by_child_species: dict[int, int] = defaultdict(int)
        births_by_parent_species: dict[int, int] = defaultdict(int)
        deaths_by_species: dict[int, int] = defaultdict(int)
        attack_stats_by_species: dict[int, dict[str, float]] = defaultdict(self._empty_combat_totals)
        fresh_kill_stats_by_species: dict[int, dict[str, float]] = defaultdict(
            self._empty_fresh_kill_totals
        )
        carcass_stats_by_species: dict[int, dict[str, float]] = defaultdict(self._empty_carcass_totals)
        diet_stats_by_species: dict[int, dict[str, float]] = defaultdict(self._empty_diet_totals)

        def resolve_species_id(agent_id: int | None) -> int:
            if agent_id is None:
                return 0
            return species_map.get(agent_id, previous_species_map.get(agent_id, 0))

        for parent_id, child_id in self.tick_birth_pairs:
            child_species = species_map.get(child_id)
            if child_species is not None:
                births_by_child_species[child_species] += 1

            parent_species = species_map.get(parent_id, previous_species_map.get(parent_id))
            if parent_species is not None:
                births_by_parent_species[parent_species] += 1

        for dead_id in self.tick_death_agent_ids:
            dead_species = previous_species_map.get(dead_id)
            if dead_species is not None:
                deaths_by_species[dead_species] += 1

        for event in self.tick_attack_events:
            species_id = resolve_species_id(int(event["attacker_id"]))
            attack_stats_by_species[species_id]["attack_attempts"] += 1
            if event["success"]:
                attack_stats_by_species[species_id]["successful_attacks"] += 1
                attack_stats_by_species[species_id]["damage_dealt"] += float(event["damage"])
            if event["kill"]:
                attack_stats_by_species[species_id]["kills"] += 1

        for event in self.tick_damage_events:
            species_id = resolve_species_id(int(event["agent_id"]))
            attack_stats_by_species[species_id]["damage_taken"] += float(event["amount"])
            if str(event["source"]).startswith("hazard_"):
                attack_stats_by_species[species_id]["hazard_damage_taken"] += float(event["amount"])

        for event in self.tick_carcass_deposit_events:
            species_id = int(event.get("source_species") or 0)
            carcass_stats_by_species[species_id]["deposition_events"] += 1
            carcass_stats_by_species[species_id]["energy_deposited"] += float(
                event["deposited_energy"]
            )

        for event in self.tick_fresh_kill_deposit_events:
            species_id = int(event.get("source_species") or 0)
            fresh_kill_stats_by_species[species_id]["deposition_events"] += 1
            fresh_kill_stats_by_species[species_id]["energy_deposited"] += float(
                event["deposited_energy"]
            )

        for event in self.tick_fresh_kill_events:
            species_id = resolve_species_id(int(event["agent_id"]))
            fresh_kill_stats_by_species[species_id]["consumption_events"] += 1
            fresh_kill_stats_by_species[species_id]["energy_consumed"] += float(event["energy"])
            fresh_kill_stats_by_species[species_id]["gained_energy"] += float(
                event["gained_energy"]
            )

        for event in self.tick_carcass_events:
            species_id = resolve_species_id(int(event["agent_id"]))
            carcass_stats_by_species[species_id]["consumption_events"] += 1
            carcass_stats_by_species[species_id]["energy_consumed"] += float(event["energy"])
            carcass_stats_by_species[species_id]["gained_energy"] += float(
                event["gained_energy"]
            )

        for event in self.tick_feeding_events:
            species_id = resolve_species_id(int(event["agent_id"]))
            prefix = str(event["food_source"])
            diet_stats_by_species[species_id][f"{prefix}_events"] += 1
            diet_stats_by_species[species_id][f"{prefix}_energy"] += float(event["gained_energy"])

        for agent in alive:
            species_id = species_map.get(agent.agent_id, 0)
            record = metrics.setdefault(
                species_id,
                {
                    "alive_count": 0,
                    "terrain_occupancy": self._empty_terrain_occupancy(),
                    "hydrology_exposure_counts": self._empty_hydrology_exposure_counts(),
                    "habitat_occupancy": {state: 0 for state in HABITAT_STATE_CODES},
                    "ecology_occupancy": {state: 0 for state in ECOLOGY_STATE_CODES},
                    "hazard_occupancy": {hazard_type: 0 for hazard_type in HAZARD_TYPE_CODES},
                    "trophic_role_occupancy": {role: 0 for role in TROPHIC_ROLE_CODES if role != "none"},
                    "meat_mode_occupancy": {mode: 0 for mode in MEAT_MODE_CODES if mode != "none"},
                    "energy_ratio_total": 0.0,
                    "hydration_ratio_total": 0.0,
                    "health_ratio_total": 0.0,
                    "age_total": 0.0,
                    "vegetation_total": 0.0,
                    "recovery_total": 0.0,
                    "refuge_score_total": 0.0,
                    "refuge_exposed_count": 0,
                    "injury_count": 0,
                    "hazard_exposed_count": 0,
                    "energy_stressed_count": 0,
                    "hydration_stressed_count": 0,
                    "reproduction_ready_count": 0,
                    "matched_diet_ratio_total": 0.0,
                    "births": births_by_child_species.get(species_id, 0),
                    "deaths": deaths_by_species.get(species_id, 0),
                    "reproduction_success": births_by_parent_species.get(species_id, 0),
                },
            )

            tile = self.grid[agent.y][agent.x]
            energy_ratio = self._energy_ratio(agent)
            hydration_ratio = self._hydration_ratio(agent)
            health_ratio = self._health_ratio(agent)
            hazard_type, _ = self._hazard_at(agent.x, agent.y)
            trophic_role = self._trophic_role(agent)
            meat_mode = self._meat_mode(agent)
            record["alive_count"] += 1
            record["energy_ratio_total"] += energy_ratio
            record["hydration_ratio_total"] += hydration_ratio
            record["health_ratio_total"] += health_ratio
            record["matched_diet_ratio_total"] += self._matched_diet_ratio(agent)
            record["age_total"] += agent.age
            record["vegetation_total"] += tile.vegetation
            record["recovery_total"] += tile.recovery_debt
            record["refuge_score_total"] += self._refuge_score(agent.x, agent.y)
            record["terrain_occupancy"][tile.terrain] += 1
            record["habitat_occupancy"][self._habitat_state_at(agent.x, agent.y)] += 1
            record["ecology_occupancy"][self._ecology_state_at(agent.x, agent.y)] += 1
            record["hazard_occupancy"][hazard_type] += 1
            record["trophic_role_occupancy"][trophic_role] += 1
            if meat_mode != "none":
                record["meat_mode_occupancy"][meat_mode] += 1
            water_reason = self._water_access_reason(agent.x, agent.y)
            support_code = self._hydrology_support_code(agent.x, agent.y)
            record["hydrology_exposure_counts"][f"primary_{water_reason}"] += 1
            if self._tile_has_water_access(agent.x, agent.y):
                record["terrain_occupancy"]["water_access"] += 1
            if support_code & HYDROLOGY_SUPPORT_FLAGS["adjacent_to_water"]:
                record["hydrology_exposure_counts"]["shoreline_support"] += 1
            if support_code & HYDROLOGY_SUPPORT_FLAGS["wetland"]:
                record["hydrology_exposure_counts"]["wetland_support"] += 1
            if support_code & HYDROLOGY_SUPPORT_FLAGS["flooded"]:
                record["hydrology_exposure_counts"]["flooded_support"] += 1
            if self._soft_refuge_reason(agent.x, agent.y) != "none":
                record["hydrology_exposure_counts"]["refuge_exposed"] += 1
                record["refuge_exposed_count"] += 1
            if hazard_type != "none":
                record["hazard_exposed_count"] += 1
            if agent.injury_load >= 0.08:
                record["injury_count"] += 1
            if energy_ratio < 0.35:
                record["energy_stressed_count"] += 1
            if hydration_ratio < 0.35:
                record["hydration_stressed_count"] += 1
            if self._is_reproduction_ready(agent):
                record["reproduction_ready_count"] += 1

        referenced_species = (
            set(births_by_child_species)
            | set(births_by_parent_species)
            | set(deaths_by_species)
            | set(attack_stats_by_species)
            | set(fresh_kill_stats_by_species)
            | set(carcass_stats_by_species)
            | set(diet_stats_by_species)
        )
        for species_id in referenced_species:
            metrics.setdefault(
                species_id,
                {
                    "alive_count": 0,
                    "terrain_occupancy": self._empty_terrain_occupancy(),
                    "hydrology_exposure_counts": self._empty_hydrology_exposure_counts(),
                    "habitat_occupancy": {state: 0 for state in HABITAT_STATE_CODES},
                    "ecology_occupancy": {state: 0 for state in ECOLOGY_STATE_CODES},
                    "hazard_occupancy": {hazard_type: 0 for hazard_type in HAZARD_TYPE_CODES},
                    "trophic_role_occupancy": {role: 0 for role in TROPHIC_ROLE_CODES if role != "none"},
                    "meat_mode_occupancy": {mode: 0 for mode in MEAT_MODE_CODES if mode != "none"},
                    "energy_ratio_total": 0.0,
                    "hydration_ratio_total": 0.0,
                    "health_ratio_total": 0.0,
                    "age_total": 0.0,
                    "vegetation_total": 0.0,
                    "recovery_total": 0.0,
                    "refuge_score_total": 0.0,
                    "refuge_exposed_count": 0,
                    "injury_count": 0,
                    "hazard_exposed_count": 0,
                    "energy_stressed_count": 0,
                    "hydration_stressed_count": 0,
                    "reproduction_ready_count": 0,
                    "matched_diet_ratio_total": 0.0,
                    "births": births_by_child_species.get(species_id, 0),
                    "deaths": deaths_by_species.get(species_id, 0),
                    "reproduction_success": births_by_parent_species.get(species_id, 0),
                },
            )

        finalized: dict[str, dict[str, object]] = {}
        for species_id, record in metrics.items():
            alive_count = max(record["alive_count"], 1)
            plant_energy = diet_stats_by_species[species_id]["plant_energy"]
            fresh_kill_energy = diet_stats_by_species[species_id]["fresh_kill_energy"]
            carcass_energy = diet_stats_by_species[species_id]["carcass_energy"]
            animal_energy = fresh_kill_energy + carcass_energy
            total_diet_energy = plant_energy + animal_energy
            finalized[str(species_id)] = {
                "alive_count": record["alive_count"],
                "births": record["births"],
                "deaths": record["deaths"],
                "reproduction_success": record["reproduction_success"],
                "avg_energy_ratio": round(record["energy_ratio_total"] / alive_count, 4),
                "avg_hydration_ratio": round(record["hydration_ratio_total"] / alive_count, 4),
                "avg_health_ratio": round(record["health_ratio_total"] / alive_count, 4),
                "avg_age": round(record["age_total"] / alive_count, 2),
                "avg_tile_vegetation": round(record["vegetation_total"] / alive_count, 4),
                "avg_recovery_debt": round(record["recovery_total"] / alive_count, 4),
                "avg_refuge_score_occupied_tiles": round(
                    record["refuge_score_total"] / alive_count,
                    4,
                ),
                "refuge_exposure_rate": round(record["refuge_exposed_count"] / alive_count, 4),
                "injury_rate": round(record["injury_count"] / alive_count, 4),
                "hazard_exposure_rate": round(record["hazard_exposed_count"] / alive_count, 4),
                "energy_stress_rate": round(record["energy_stressed_count"] / alive_count, 4),
                "hydration_stress_rate": round(record["hydration_stressed_count"] / alive_count, 4),
                "reproduction_ready_rate": round(
                    record["reproduction_ready_count"] / alive_count,
                    4,
                ),
                "avg_matched_diet_ratio": round(
                    record["matched_diet_ratio_total"] / alive_count,
                    4,
                ),
                "attack_attempts": int(attack_stats_by_species[species_id]["attack_attempts"]),
                "successful_attacks": int(
                    attack_stats_by_species[species_id]["successful_attacks"]
                ),
                "kills": int(attack_stats_by_species[species_id]["kills"]),
                "damage_dealt": round(attack_stats_by_species[species_id]["damage_dealt"], 4),
                "damage_taken": round(attack_stats_by_species[species_id]["damage_taken"], 4),
                "hazard_damage_taken": round(
                    attack_stats_by_species[species_id]["hazard_damage_taken"],
                    4,
                ),
                "plant_consumption": int(diet_stats_by_species[species_id]["plant_events"]),
                "plant_energy_consumed": round(plant_energy, 4),
                "fresh_kill_deposition": int(
                    fresh_kill_stats_by_species[species_id]["deposition_events"]
                ),
                "fresh_kill_energy_deposited": round(
                    fresh_kill_stats_by_species[species_id]["energy_deposited"],
                    4,
                ),
                "fresh_kill_consumption": int(
                    fresh_kill_stats_by_species[species_id]["consumption_events"]
                ),
                "fresh_kill_energy_consumed": round(
                    fresh_kill_stats_by_species[species_id]["energy_consumed"],
                    4,
                ),
                "fresh_kill_gained_energy": round(
                    fresh_kill_stats_by_species[species_id]["gained_energy"],
                    4,
                ),
                "carcass_deposition": int(carcass_stats_by_species[species_id]["deposition_events"]),
                "carcass_energy_deposited": round(
                    carcass_stats_by_species[species_id]["energy_deposited"],
                    4,
                ),
                "carcass_consumption": int(carcass_stats_by_species[species_id]["consumption_events"]),
                "carcass_energy_consumed": round(
                    carcass_stats_by_species[species_id]["energy_consumed"],
                    4,
                ),
                "carcass_gained_energy": round(
                    carcass_stats_by_species[species_id]["gained_energy"],
                    4,
                ),
                "realized_plant_share": round(
                    plant_energy / max(total_diet_energy, 1e-9),
                    4,
                )
                if total_diet_energy > 0
                else 0.0,
                "realized_animal_share": round(
                    animal_energy / max(total_diet_energy, 1e-9),
                    4,
                )
                if total_diet_energy > 0
                else 0.0,
                "realized_fresh_kill_share": round(
                    fresh_kill_energy / max(total_diet_energy, 1e-9),
                    4,
                )
                if total_diet_energy > 0
                else 0.0,
                "realized_carcass_share": round(
                    carcass_energy / max(total_diet_energy, 1e-9),
                    4,
                )
                if total_diet_energy > 0
                else 0.0,
                "terrain_occupancy": record["terrain_occupancy"],
                "hydrology_exposure_counts": record["hydrology_exposure_counts"],
                "habitat_occupancy": record["habitat_occupancy"],
                "ecology_occupancy": record["ecology_occupancy"],
                "hazard_occupancy": record["hazard_occupancy"],
                "trophic_role_occupancy": record["trophic_role_occupancy"],
                "meat_mode_occupancy": record["meat_mode_occupancy"],
            }
        return finalized

    @staticmethod
    def _empty_terrain_occupancy() -> dict[str, int]:
        return {**{terrain: 0 for terrain in LAND_TERRAINS}, "water_access": 0}

    @staticmethod
    def _empty_hydrology_exposure_counts() -> dict[str, int]:
        return {
            **{f"primary_{reason}": 0 for reason in HYDROLOGY_REASON_CODES},
            "shoreline_support": 0,
            "wetland_support": 0,
            "flooded_support": 0,
            "refuge_exposed": 0,
        }

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
                    "lineage_id": agent.lineage_id,
                    "birth_tick": agent.birth_tick,
                    "death_tick": agent.death_tick,
                    "genome": agent.genome.to_dict(),
                }
                for agent in sorted(self.agents.values(), key=lambda item: item.agent_id)
            },
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
                    "founder_agent_id": registry.get("founder_agent_id"),
                    "centroid": {
                        gene: round(value, 4)
                        for gene, value in registry.get("centroid", {}).items()
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
                    "centroid": {
                        gene: round(value, 4)
                        for gene, value in registry.get("centroid", {}).items()
                    },
                }
                for ecotype_id, registry in sorted(self.ecotype_registry.items())
            },
            "frames": self.viewer_frames,
            "analytics": self._build_analytics(),
            "agent_encoding": [
                "agent_id",
                "x",
                "y",
                "energy",
                "energy_ratio",
                "hydration",
                "hydration_ratio",
                "health",
                "health_ratio",
                "injury_load",
                "age",
                "energy_modifier",
                "hydration_modifier",
                "tile_vegetation",
                "tile_recovery_debt",
                "reproduction_ready",
                "trophic_role",
                "meat_mode",
                "last_damage_source",
                "water_access_reason",
                "soft_refuge_reason",
                "hydrology_support_code",
                "refuge_score",
                "matched_diet_ratio",
                "ecotype_id",
                "species_id",
            ],
        }

    def _build_analytics(self) -> dict[str, object]:
        ticks = [frame["tick"] for frame in self.viewer_frames]
        population = {
            "alive_agents": [frame["alive_agents"] for frame in self.viewer_frames],
            "species_count": [len(frame["species_counts"]) for frame in self.viewer_frames],
            "ecotype_count": [len(frame["ecotype_counts"]) for frame in self.viewer_frames],
            "births": [frame["births"] for frame in self.viewer_frames],
            "deaths": [frame["deaths"] for frame in self.viewer_frames],
        }
        traits = {
            "avg_max_energy": [
                frame["trait_means"]["avg_max_energy"] for frame in self.viewer_frames
            ],
            "avg_max_health": [
                frame["trait_means"]["avg_max_health"] for frame in self.viewer_frames
            ],
            "avg_move_cost": [
                frame["trait_means"]["avg_move_cost"] for frame in self.viewer_frames
            ],
            "avg_heat_tolerance": [
                frame["trait_means"]["avg_heat_tolerance"] for frame in self.viewer_frames
            ],
            "avg_food_efficiency": [
                frame["trait_means"]["avg_food_efficiency"] for frame in self.viewer_frames
            ],
            "avg_water_efficiency": [
                frame["trait_means"]["avg_water_efficiency"] for frame in self.viewer_frames
            ],
            "avg_attack_power": [
                frame["trait_means"]["avg_attack_power"] for frame in self.viewer_frames
            ],
            "avg_meat_efficiency": [
                frame["trait_means"]["avg_meat_efficiency"] for frame in self.viewer_frames
            ],
            "avg_carrion_bias": [
                frame["trait_means"]["avg_carrion_bias"] for frame in self.viewer_frames
            ],
            "avg_live_prey_bias": [
                frame["trait_means"]["avg_live_prey_bias"] for frame in self.viewer_frames
            ],
            "avg_wetland_affinity": [
                frame["trait_means"]["avg_wetland_affinity"] for frame in self.viewer_frames
            ],
            "avg_rocky_affinity": [
                frame["trait_means"]["avg_rocky_affinity"] for frame in self.viewer_frames
            ],
        }
        trophic_roles = {
            role: [frame["trophic_role_counts"].get(role, 0) for frame in self.viewer_frames]
            for role in TROPHIC_ROLE_CODES
            if role != "none"
        }
        meat_modes = {
            mode: [frame["meat_mode_counts"].get(mode, 0) for frame in self.viewer_frames]
            for mode in MEAT_MODE_CODES
        }
        diet_by_trophic_role = {
            role: {
                metric: [
                    frame["diet_by_trophic_role"][role][metric] for frame in self.viewer_frames
                ]
                for metric in (
                    "plant_events",
                    "plant_energy",
                    "fresh_kill_events",
                    "fresh_kill_energy",
                    "carcass_events",
                    "carcass_energy",
                    "plant_energy_share",
                    "animal_energy_share",
                    "fresh_kill_energy_share",
                    "carcass_energy_share",
                )
            }
            for role in TROPHIC_ROLE_CODES
            if role != "none"
        }
        diet_by_meat_mode = {
            mode: {
                metric: [
                    frame["diet_by_meat_mode"][mode][metric] for frame in self.viewer_frames
                ]
                for metric in (
                    "plant_events",
                    "plant_energy",
                    "fresh_kill_events",
                    "fresh_kill_energy",
                    "carcass_events",
                    "carcass_energy",
                    "plant_energy_share",
                    "animal_energy_share",
                    "fresh_kill_energy_share",
                    "carcass_energy_share",
                )
            }
            for mode in MEAT_MODE_CODES
        }

        species_population: dict[str, list[int]] = {
            str(species_id): [0 for _ in ticks]
            for species_id in sorted(self.species_registry)
        }
        for frame_index, frame in enumerate(self.viewer_frames):
            frame_counts = {str(species_id): count for species_id, count in frame["species_counts"]}
            for species_id in species_population:
                species_population[species_id][frame_index] = frame_counts.get(species_id, 0)

        return {
            "ticks": ticks,
            "population": population,
            "traits": traits,
            "trophic_roles": trophic_roles,
            "meat_modes": meat_modes,
            "species_population": species_population,
            "collapse_events": self._build_collapse_events(ticks, species_population),
            "habitat": {
                state: [frame["habitat_state_counts"].get(state, 0) for frame in self.viewer_frames]
                for state in HABITAT_STATE_CODES
            },
            "hydrology_primary": {
                **{
                    reason: [
                        frame["hydrology_primary_counts"].get(reason, 0)
                        for frame in self.viewer_frames
                    ]
                    for reason in HYDROLOGY_REASON_CODES
                },
                "hard_access_tiles": [
                    frame["hydrology_primary_stats"]["hard_access_tiles"]
                    for frame in self.viewer_frames
                ],
            },
            "hydrology_support": {
                "shoreline_support": [
                    frame["hydrology_support_counts"].get("shoreline_support", 0)
                    for frame in self.viewer_frames
                ],
                "wetland_support": [
                    frame["hydrology_support_counts"].get("wetland_support", 0)
                    for frame in self.viewer_frames
                ],
                "flooded_support": [
                    frame["hydrology_support_counts"].get("flooded_support", 0)
                    for frame in self.viewer_frames
                ],
            },
            "refuge": {
                **{
                    reason: [
                        frame["refuge_counts"].get(reason, 0)
                        for frame in self.viewer_frames
                    ]
                    for reason in SOFT_REFUGE_CODES
                },
                "canopy_refuge_tiles": [
                    frame["refuge_counts"].get("canopy_refuge", 0)
                    for frame in self.viewer_frames
                ],
                "avg_refuge_score_forest_tiles": [
                    frame["refuge_stats"]["avg_refuge_score_forest_tiles"]
                    for frame in self.viewer_frames
                ],
            },
            "ecology": {
                **{
                    state: [
                        frame["ecology_state_counts"].get(state, 0)
                        for frame in self.viewer_frames
                    ]
                    for state in ECOLOGY_STATE_CODES
                },
                "avg_vegetation": [
                    frame["ecology_stats"]["avg_vegetation"] for frame in self.viewer_frames
                ],
                "avg_recovery_debt": [
                    frame["ecology_stats"]["avg_recovery_debt"]
                    for frame in self.viewer_frames
                ],
            },
            "hazards": {
                **{
                    hazard_type: [
                        frame["hazard_counts"].get(hazard_type, 0) for frame in self.viewer_frames
                    ]
                    for hazard_type in HAZARD_TYPE_CODES
                },
                "hazardous_tiles": [
                    frame["hazard_stats"]["hazardous_tiles"] for frame in self.viewer_frames
                ],
                "avg_hazard_level": [
                    frame["hazard_stats"]["avg_hazard_level"] for frame in self.viewer_frames
                ],
            },
            "biotic_fields": {
                field_name: {
                    metric: [
                        frame["biotic_field_stats"][field_name][metric]
                        for frame in self.viewer_frames
                    ]
                    for metric in ("avg", "max")
                }
                for field_name in ("prey_biomass", "carrion", "predator_risk")
            },
            "fresh_kill": {
                "fresh_kill_tiles": [
                    frame["fresh_kill_stats"]["fresh_kill_tiles"] for frame in self.viewer_frames
                ],
                "total_fresh_kill_energy": [
                    frame["fresh_kill_stats"]["total_fresh_kill_energy"]
                    for frame in self.viewer_frames
                ],
                "deposit_count": [
                    frame["fresh_kill_stats"]["deposit_count"] for frame in self.viewer_frames
                ],
                "mixed_source_tiles": [
                    frame["fresh_kill_stats"]["mixed_source_tiles"]
                    for frame in self.viewer_frames
                ],
                "deposition_events": [
                    frame["fresh_kill_flow"]["deposition_events"] for frame in self.viewer_frames
                ],
                "fresh_kill_energy_deposited": [
                    frame["fresh_kill_flow"]["fresh_kill_energy_deposited"]
                    for frame in self.viewer_frames
                ],
                "fresh_kill_energy_converted_to_carcass": [
                    frame["fresh_kill_flow"]["fresh_kill_energy_converted_to_carcass"]
                    for frame in self.viewer_frames
                ],
                "consumption_events": [
                    frame["fresh_kill_flow"]["consumption_events"]
                    for frame in self.viewer_frames
                ],
                "fresh_kill_energy_consumed": [
                    frame["fresh_kill_flow"]["fresh_kill_energy_consumed"]
                    for frame in self.viewer_frames
                ],
                "fresh_kill_gained_energy": [
                    frame["fresh_kill_flow"]["fresh_kill_gained_energy"]
                    for frame in self.viewer_frames
                ],
            },
            "carcasses": {
                "carcass_tiles": [
                    frame["carcass_stats"]["carcass_tiles"] for frame in self.viewer_frames
                ],
                "total_carcass_energy": [
                    frame["carcass_stats"]["total_carcass_energy"] for frame in self.viewer_frames
                ],
                "avg_carcass_freshness": [
                    frame["carcass_stats"]["avg_carcass_freshness"] for frame in self.viewer_frames
                ],
                "deposit_count": [
                    frame["carcass_stats"]["deposit_count"] for frame in self.viewer_frames
                ],
                "mixed_source_tiles": [
                    frame["carcass_stats"]["mixed_source_tiles"] for frame in self.viewer_frames
                ],
                "deposition_events": [
                    frame["carcass_flow"]["deposition_events"] for frame in self.viewer_frames
                ],
                "carcass_energy_deposited": [
                    frame["carcass_flow"]["carcass_energy_deposited"] for frame in self.viewer_frames
                ],
                "carcass_energy_decayed": [
                    frame["carcass_flow"]["carcass_energy_decayed"] for frame in self.viewer_frames
                ],
                "consumption_events": [
                    frame["carcass_flow"]["consumption_events"]
                    for frame in self.viewer_frames
                ],
                "carcass_energy_consumed": [
                    frame["carcass_flow"]["carcass_energy_consumed"]
                    for frame in self.viewer_frames
                ],
                "carcass_gained_energy": [
                    frame["carcass_flow"]["carcass_gained_energy"]
                    for frame in self.viewer_frames
                ],
            },
            "diet": {
                "plant_events": [
                    frame["diet_stats"]["plant_events"] for frame in self.viewer_frames
                ],
                "plant_energy": [
                    frame["diet_stats"]["plant_energy"] for frame in self.viewer_frames
                ],
                "fresh_kill_events": [
                    frame["diet_stats"]["fresh_kill_events"] for frame in self.viewer_frames
                ],
                "fresh_kill_energy": [
                    frame["diet_stats"]["fresh_kill_energy"] for frame in self.viewer_frames
                ],
                "carcass_events": [
                    frame["diet_stats"]["carcass_events"] for frame in self.viewer_frames
                ],
                "carcass_energy": [
                    frame["diet_stats"]["carcass_energy"] for frame in self.viewer_frames
                ],
                "animal_events": [
                    frame["diet_stats"]["animal_events"] for frame in self.viewer_frames
                ],
                "animal_energy": [
                    frame["diet_stats"]["animal_energy"] for frame in self.viewer_frames
                ],
                "plant_energy_share": [
                    frame["diet_stats"]["plant_energy_share"] for frame in self.viewer_frames
                ],
                "animal_energy_share": [
                    frame["diet_stats"]["animal_energy_share"] for frame in self.viewer_frames
                ],
                "fresh_kill_energy_share": [
                    frame["diet_stats"]["fresh_kill_energy_share"] for frame in self.viewer_frames
                ],
                "carcass_energy_share": [
                    frame["diet_stats"]["carcass_energy_share"] for frame in self.viewer_frames
                ],
            },
            "diet_by_trophic_role": diet_by_trophic_role,
            "diet_by_meat_mode": diet_by_meat_mode,
            "combat": {
                "attack_attempts": [
                    frame["combat_stats"]["attack_attempts"] for frame in self.viewer_frames
                ],
                "successful_attacks": [
                    frame["combat_stats"]["successful_attacks"] for frame in self.viewer_frames
                ],
                "kills": [frame["combat_stats"]["kills"] for frame in self.viewer_frames],
                "damage_dealt": [
                    frame["combat_stats"]["damage_dealt"] for frame in self.viewer_frames
                ],
                "attack_damage_taken": [
                    frame["combat_stats"]["attack_damage_taken"] for frame in self.viewer_frames
                ],
                "hazard_damage_taken": [
                    frame["combat_stats"]["hazard_damage_taken"] for frame in self.viewer_frames
                ],
            },
        }

    def _build_collapse_events(
        self,
        ticks: list[int],
        species_population: dict[str, list[int]],
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for species_id, counts in species_population.items():
            peak = 0
            collapse_recorded = False
            previous = 0
            for frame_index, count in enumerate(counts):
                peak = max(peak, count)
                tick = ticks[frame_index]
                if (
                    not collapse_recorded
                    and peak >= 12
                    and count > 0
                    and count <= int(peak * 0.4)
                ):
                    events.append(
                        {
                            "tick": tick,
                            "species_id": int(species_id),
                            "type": "collapse",
                            "peak": peak,
                            "current": count,
                        }
                    )
                    collapse_recorded = True

                if previous > 0 and count == 0:
                    events.append(
                        {
                            "tick": tick,
                            "species_id": int(species_id),
                            "type": "extinction",
                            "peak": peak,
                            "current": count,
                        }
                    )
                previous = count
        events.sort(key=lambda event: (event["tick"], event["species_id"], event["type"]))
        return events

    def _build_summary(self) -> dict[str, object]:
        alive = self.alive_agents()
        genomes = [agent.genome for agent in self.agents.values()]
        lineage_sizes: dict[int, int] = {}
        alive_lineage_sizes: dict[int, int] = {}
        for agent in self.agents.values():
            lineage_sizes[agent.lineage_id] = lineage_sizes.get(agent.lineage_id, 0) + 1
            if agent.alive:
                alive_lineage_sizes[agent.lineage_id] = alive_lineage_sizes.get(agent.lineage_id, 0) + 1

        top_species = sorted(
            (
                {
                    "species_id": species_id,
                    "label": registry["label"],
                    "peak_members": registry["peak_members"],
                    "alive_members": registry.get("current_members", 0),
                    "lineages": sorted(registry["lineages"]),
                }
                for species_id, registry in self.species_registry.items()
            ),
            key=lambda item: (-item["alive_members"], -item["peak_members"], item["species_id"]),
        )[:10]

        field_stats = self._field_stats()
        climate_end = self._climate_state()
        terrain_counts = self._terrain_counts()
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
        _, ecology_counts, ecology_stats = self._ecology_snapshot()
        land_tile_count = self.config.width * self.config.height - terrain_counts["water"]
        trophic_role_counts, meat_mode_counts = self._population_trophic_counts(alive)
        latest_species_metrics = self.viewer_frames[-1]["species_metrics"] if self.viewer_frames else {}
        fresh_kill_conservation_error = (
            self.run_fresh_kill_totals["energy_deposited"]
            - self.run_fresh_kill_totals["energy_converted_to_carcass"]
            - self.run_fresh_kill_totals["energy_consumed"]
            - fresh_kill_stats["total_fresh_kill_energy"]
        )
        carcass_conservation_error = (
            self.run_carcass_totals["energy_deposited"]
            - self.run_carcass_totals["energy_decayed"]
            - self.run_carcass_totals["energy_consumed"]
            - carcass_stats["total_carcass_energy"]
        )

        return {
            "run_id": self.run_id,
            "seed": self.config.seed,
            "ticks_executed": self.tick + 1,
            "births": self.births,
            "deaths": self.deaths,
            "alive_agents": len(alive),
            "peak_alive_agents": self.peak_alive_agents,
            "extinct": len(alive) == 0,
            "total_agents_seen": len(self.agents),
            "season_at_end": self._season_state()["name"],
            "disturbance_at_end": climate_end["disturbance_type"],
            "disturbance_strength_at_end": climate_end["disturbance_strength"],
            "lineages": sorted({agent.lineage_id for agent in self.agents.values()}),
            "alive_lineages": sorted({agent.lineage_id for agent in alive}),
            "taxonomy_mode": REPLAY_TAXONOMY_MODE,
            "top_lineages": sorted(
                (
                    {
                        "lineage_id": lineage_id,
                        "total_agents": size,
                        "alive_agents": alive_lineage_sizes.get(lineage_id, 0),
                    }
                    for lineage_id, size in lineage_sizes.items()
                ),
                key=lambda item: (-item["alive_agents"], -item["total_agents"], item["lineage_id"]),
            )[:10],
            "species_created": len(self.species_registry),
            "alive_species_count": len(self.current_species_records),
            "alive_species": [record.species_id for record in self.current_species_records],
            "ecotypes_created": len(self.ecotype_registry),
            "alive_ecotype_count": len(self.current_ecotype_records),
            "top_species": top_species,
            "last_birth_tick": self.last_birth_tick,
            "field_stats": field_stats,
            "terrain_counts": terrain_counts,
            "land_tile_count": land_tile_count,
            "habitat_state_counts_at_end": self._habitat_state_grid()[1],
            "hydrology_primary_counts_at_end": hydrology_primary_counts,
            "hydrology_support_counts_at_end": hydrology_support_counts,
            "hydrology_primary_stats_at_end": hydrology_primary_stats,
            "refuge_counts_at_end": refuge_counts,
            "refuge_stats_at_end": refuge_stats,
            "hazard_counts_at_end": hazard_counts,
            "hazard_stats_at_end": hazard_stats,
            "biotic_field_stats_at_end": biotic_field_stats,
            "fresh_kill_stats_at_end": fresh_kill_stats,
            "carcass_stats_at_end": carcass_stats,
            "trophic_role_counts_at_end": trophic_role_counts,
            "meat_mode_counts_at_end": meat_mode_counts,
            "combat_end": {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in self.run_combat_totals.items()
            },
            "fresh_kill_end": {
                **{
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in self.run_fresh_kill_totals.items()
                },
                "conservation_error": round(fresh_kill_conservation_error, 6),
            },
            "carcass_end": {
                **{
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in self.run_carcass_totals.items()
                },
                "conservation_error": round(carcass_conservation_error, 6),
            },
            "diet_end": self._finalize_diet_totals(self.run_diet_totals),
            "diet_by_trophic_role_end": self._finalize_grouped_diet_totals(
                self.run_diet_by_trophic_role
            ),
            "diet_by_meat_mode_end": self._finalize_grouped_diet_totals(self.run_diet_by_meat_mode),
            "top_species_by_realized_animal_share": sorted(
                (
                    {
                        "species_id": int(species_id),
                        "alive_count": int(metrics["alive_count"]),
                        "realized_animal_share": float(metrics["realized_animal_share"]),
                        "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                        "realized_carcass_share": float(metrics["realized_carcass_share"]),
                        "reproduction_success": int(metrics["reproduction_success"]),
                    }
                    for species_id, metrics in latest_species_metrics.items()
                    if metrics["alive_count"] > 0 or metrics["reproduction_success"] > 0
                ),
                key=lambda item: (
                    -item["realized_animal_share"],
                    -item["reproduction_success"],
                    item["species_id"],
                ),
            )[:10],
            "top_species_by_realized_fresh_kill_share": sorted(
                (
                    {
                        "species_id": int(species_id),
                        "alive_count": int(metrics["alive_count"]),
                        "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                        "fresh_kill_gained_energy": float(metrics["fresh_kill_gained_energy"]),
                        "kills": int(metrics["kills"]),
                    }
                    for species_id, metrics in latest_species_metrics.items()
                    if metrics["alive_count"] > 0
                    or metrics["fresh_kill_gained_energy"] > 0
                    or metrics["kills"] > 0
                ),
                key=lambda item: (
                    -item["realized_fresh_kill_share"],
                    -item["fresh_kill_gained_energy"],
                    -item["kills"],
                    item["species_id"],
                ),
            )[:10],
            "top_species_by_realized_carcass_share": sorted(
                (
                    {
                        "species_id": int(species_id),
                        "alive_count": int(metrics["alive_count"]),
                        "realized_carcass_share": float(metrics["realized_carcass_share"]),
                        "carcass_gained_energy": float(metrics["carcass_gained_energy"]),
                        "carcass_energy_consumed": float(metrics["carcass_energy_consumed"]),
                        "attack_attempts": int(metrics["attack_attempts"]),
                        "kills": int(metrics["kills"]),
                    }
                    for species_id, metrics in latest_species_metrics.items()
                    if metrics["alive_count"] > 0
                    or metrics["carcass_gained_energy"] > 0
                    or metrics["attack_attempts"] > 0
                ),
                key=lambda item: (
                    -item["realized_carcass_share"],
                    -item["carcass_gained_energy"],
                    -item["attack_attempts"],
                    item["species_id"],
                ),
            )[:10],
            "top_carnivore_species": sorted(
                (
                    {
                        "species_id": int(species_id),
                        "alive_count": int(metrics["alive_count"]),
                        "realized_animal_share": float(metrics["realized_animal_share"]),
                        "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                        "realized_carcass_share": float(metrics["realized_carcass_share"]),
                    }
                    for species_id, metrics in latest_species_metrics.items()
                    if int(metrics["trophic_role_occupancy"]["carnivore"]) > 0
                ),
                key=lambda item: (
                    -item["alive_count"],
                    -item["realized_animal_share"],
                    item["species_id"],
                ),
            )[:10],
            "top_hunter_species": sorted(
                (
                    {
                        "species_id": int(species_id),
                        "alive_count": int(metrics["alive_count"]),
                        "kills": int(metrics["kills"]),
                        "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                    }
                    for species_id, metrics in latest_species_metrics.items()
                    if int(metrics["meat_mode_occupancy"]["hunter"]) > 0
                ),
                key=lambda item: (
                    -item["alive_count"],
                    -item["kills"],
                    -item["realized_fresh_kill_share"],
                    item["species_id"],
                ),
            )[:10],
            "top_scavenger_species": sorted(
                (
                    {
                        "species_id": int(species_id),
                        "alive_count": int(metrics["alive_count"]),
                        "realized_carcass_share": float(metrics["realized_carcass_share"]),
                        "carcass_gained_energy": float(metrics["carcass_gained_energy"]),
                    }
                    for species_id, metrics in latest_species_metrics.items()
                    if int(metrics["meat_mode_occupancy"]["scavenger"]) > 0
                ),
                key=lambda item: (
                    -item["alive_count"],
                    -item["realized_carcass_share"],
                    -item["carcass_gained_energy"],
                    item["species_id"],
                ),
            )[:10],
            "ecology_state_counts_at_end": ecology_counts,
            "ecology_stats_at_end": ecology_stats,
            "avg_max_energy_gene": round(
                sum(genome.max_energy for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_max_health_gene": round(
                sum(genome.max_health for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_move_cost_gene": round(
                sum(genome.move_cost for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_food_efficiency_gene": round(
                sum(genome.food_efficiency for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_water_efficiency_gene": round(
                sum(genome.water_efficiency for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_attack_power_gene": round(
                sum(genome.attack_power for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_meat_efficiency_gene": round(
                sum(genome.meat_efficiency for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_carrion_bias_gene": round(
                sum(genome.carrion_bias for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_live_prey_bias_gene": round(
                sum(genome.live_prey_bias for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_forest_affinity_gene": round(
                sum(genome.forest_affinity for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_plain_affinity_gene": round(
                sum(genome.plain_affinity for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_wetland_affinity_gene": round(
                sum(genome.wetland_affinity for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_rocky_affinity_gene": round(
                sum(genome.rocky_affinity for genome in genomes) / max(len(genomes), 1), 4
            ),
            "avg_heat_tolerance_gene": round(
                sum(genome.heat_tolerance for genome in genomes) / max(len(genomes), 1), 4
            ),
        }

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
