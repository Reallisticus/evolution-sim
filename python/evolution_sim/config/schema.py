from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite


def _check_integer(name: str, value: int, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _check_number(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive_minimum: bool = True,
    inclusive_maximum: bool = True,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        too_low = value < minimum if inclusive_minimum else value <= minimum
        if too_low:
            operator = ">=" if inclusive_minimum else ">"
            raise ValueError(f"{name} must be {operator} {minimum:g}")
    if maximum is not None:
        too_high = value > maximum if inclusive_maximum else value >= maximum
        if too_high:
            operator = "<=" if inclusive_maximum else "<"
            raise ValueError(f"{name} must be {operator} {maximum:g}")


def _check_fraction(name: str, value: float) -> None:
    _check_number(name, value, minimum=0.0, maximum=1.0)


def _check_nonnegative(name: str, value: float) -> None:
    _check_number(name, value, minimum=0.0)


def _check_positive(name: str, value: float) -> None:
    _check_number(name, value, minimum=0.0, inclusive_minimum=False)


@dataclass(slots=True)
class ClimateConfig:
    season_length: int = 90
    dry_plain_penalty: float = 0.55
    dry_forest_penalty: float = 0.2
    wet_plain_bonus: float = 0.25
    wet_forest_bonus: float = 0.1
    seasonal_hydration_shift: float = 0.01

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_integer("climate.season_length", self.season_length, minimum=1)
        _check_fraction("climate.dry_plain_penalty", self.dry_plain_penalty)
        _check_fraction("climate.dry_forest_penalty", self.dry_forest_penalty)
        _check_nonnegative("climate.wet_plain_bonus", self.wet_plain_bonus)
        _check_nonnegative("climate.wet_forest_bonus", self.wet_forest_bonus)
        _check_nonnegative(
            "climate.seasonal_hydration_shift",
            self.seasonal_hydration_shift,
        )


@dataclass(slots=True)
class EnvironmentFieldConfig:
    control_points_x: int = 7
    control_points_y: int = 5
    shoreline_reserve_ratio: float = 0.38
    moisture_season_swing: float = 0.18
    heat_season_swing: float = 0.16
    fertility_moisture_coupling: float = 0.14
    adjacent_water_moisture_bonus: float = 0.08
    front_width: float = 0.22
    moisture_front_strength: float = 0.14
    heat_front_strength: float = 0.14
    drift_period_ticks: int = 160
    disturbance_strength: float = 0.2
    disturbance_radius: float = 0.18
    disturbance_period_ticks: int = 210

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_integer("environment.control_points_x", self.control_points_x, minimum=2)
        _check_integer("environment.control_points_y", self.control_points_y, minimum=2)
        _check_fraction(
            "environment.shoreline_reserve_ratio",
            self.shoreline_reserve_ratio,
        )
        _check_nonnegative("environment.moisture_season_swing", self.moisture_season_swing)
        _check_nonnegative("environment.heat_season_swing", self.heat_season_swing)
        _check_nonnegative(
            "environment.fertility_moisture_coupling",
            self.fertility_moisture_coupling,
        )
        _check_nonnegative(
            "environment.adjacent_water_moisture_bonus",
            self.adjacent_water_moisture_bonus,
        )
        _check_positive("environment.front_width", self.front_width)
        _check_nonnegative("environment.moisture_front_strength", self.moisture_front_strength)
        _check_nonnegative("environment.heat_front_strength", self.heat_front_strength)
        _check_integer("environment.drift_period_ticks", self.drift_period_ticks, minimum=1)
        _check_nonnegative("environment.disturbance_strength", self.disturbance_strength)
        _check_fraction("environment.disturbance_radius", self.disturbance_radius)
        _check_integer(
            "environment.disturbance_period_ticks",
            self.disturbance_period_ticks,
            minimum=1,
        )


@dataclass(slots=True)
class ResourceRegrowthConfig:
    plain_food_rate: float = 0.013
    forest_food_rate: float = 0.026
    wetland_food_rate: float = 0.022
    rocky_food_rate: float = 0.01
    vegetation_regrowth_rate: float = 0.044
    shelter_regrowth_rate: float = 0.028
    shelter_degradation_rate: float = 0.017
    terrain_recovery_rate: float = 0.012
    terrain_degradation_rate: float = 0.018
    water_refresh_amount: float = 1.0
    eat_amount: float = 0.3
    drink_amount: float = 0.34

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_nonnegative("resources.plain_food_rate", self.plain_food_rate)
        _check_nonnegative("resources.forest_food_rate", self.forest_food_rate)
        _check_nonnegative("resources.wetland_food_rate", self.wetland_food_rate)
        _check_nonnegative("resources.rocky_food_rate", self.rocky_food_rate)
        _check_nonnegative(
            "resources.vegetation_regrowth_rate",
            self.vegetation_regrowth_rate,
        )
        _check_nonnegative("resources.shelter_regrowth_rate", self.shelter_regrowth_rate)
        _check_nonnegative(
            "resources.shelter_degradation_rate",
            self.shelter_degradation_rate,
        )
        _check_nonnegative("resources.terrain_recovery_rate", self.terrain_recovery_rate)
        _check_nonnegative("resources.terrain_degradation_rate", self.terrain_degradation_rate)
        _check_nonnegative("resources.water_refresh_amount", self.water_refresh_amount)
        _check_positive("resources.eat_amount", self.eat_amount)
        _check_positive("resources.drink_amount", self.drink_amount)


@dataclass(slots=True)
class HazardConfig:
    exposure_damage_rate: float = 0.042
    instability_damage_rate: float = 0.038
    min_hazard_level: float = 0.12
    healing_base_rate: float = 0.032
    healing_hazard_threshold: float = 0.35
    min_energy_ratio_for_healing: float = 0.65
    min_hydration_ratio_for_healing: float = 0.65

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_nonnegative("hazards.exposure_damage_rate", self.exposure_damage_rate)
        _check_nonnegative("hazards.instability_damage_rate", self.instability_damage_rate)
        _check_fraction("hazards.min_hazard_level", self.min_hazard_level)
        _check_nonnegative("hazards.healing_base_rate", self.healing_base_rate)
        _check_fraction("hazards.healing_hazard_threshold", self.healing_hazard_threshold)
        _check_fraction(
            "hazards.min_energy_ratio_for_healing",
            self.min_energy_ratio_for_healing,
        )
        _check_fraction(
            "hazards.min_hydration_ratio_for_healing",
            self.min_hydration_ratio_for_healing,
        )


@dataclass(slots=True)
class CombatConfig:
    min_attack_health_ratio: float = 0.55
    min_attack_energy_ratio: float = 0.38
    min_attack_hydration_ratio: float = 0.32
    min_reproduction_health_ratio: float = 0.72
    base_attack_damage: float = 0.5
    hunter_mode_attack_damage_multiplier: float = 1.18
    hunter_wounded_prey_damage_bonus: float = 0.35
    attack_energy_cost: float = 0.045
    attack_hydration_cost: float = 0.03

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_fraction("combat.min_attack_health_ratio", self.min_attack_health_ratio)
        _check_fraction("combat.min_attack_energy_ratio", self.min_attack_energy_ratio)
        _check_fraction("combat.min_attack_hydration_ratio", self.min_attack_hydration_ratio)
        _check_fraction(
            "combat.min_reproduction_health_ratio",
            self.min_reproduction_health_ratio,
        )
        _check_nonnegative("combat.base_attack_damage", self.base_attack_damage)
        _check_positive(
            "combat.hunter_mode_attack_damage_multiplier",
            self.hunter_mode_attack_damage_multiplier,
        )
        _check_nonnegative(
            "combat.hunter_wounded_prey_damage_bonus",
            self.hunter_wounded_prey_damage_bonus,
        )
        _check_nonnegative("combat.attack_energy_cost", self.attack_energy_cost)
        _check_nonnegative("combat.attack_hydration_cost", self.attack_hydration_cost)


@dataclass(slots=True)
class CarcassConfig:
    base_energy: float = 0.3
    health_ratio_yield: float = 0.5
    body_capacity_yield: float = 0.24
    fresh_kill_conversion_rate: float = 0.5
    decay_base_rate: float = 0.012
    decay_heat_factor: float = 0.014
    decay_moisture_factor: float = 0.008
    healing_fraction: float = 0.1
    scavenger_healing_multiplier: float = 3.0
    scavenger_hydration_fraction: float = 0.18
    max_tile_deposits: int = 8
    freshness_merge_bucket: float = 0.12

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_nonnegative("carcasses.base_energy", self.base_energy)
        _check_nonnegative("carcasses.health_ratio_yield", self.health_ratio_yield)
        _check_nonnegative("carcasses.body_capacity_yield", self.body_capacity_yield)
        _check_fraction(
            "carcasses.fresh_kill_conversion_rate",
            self.fresh_kill_conversion_rate,
        )
        _check_nonnegative("carcasses.decay_base_rate", self.decay_base_rate)
        _check_nonnegative("carcasses.decay_heat_factor", self.decay_heat_factor)
        _check_nonnegative("carcasses.decay_moisture_factor", self.decay_moisture_factor)
        _check_fraction("carcasses.healing_fraction", self.healing_fraction)
        _check_positive(
            "carcasses.scavenger_healing_multiplier",
            self.scavenger_healing_multiplier,
        )
        _check_nonnegative(
            "carcasses.scavenger_hydration_fraction",
            self.scavenger_hydration_fraction,
        )
        _check_integer("carcasses.max_tile_deposits", self.max_tile_deposits, minimum=1)
        _check_number(
            "carcasses.freshness_merge_bucket",
            self.freshness_merge_bucket,
            minimum=0.0,
            maximum=1.0,
            inclusive_minimum=False,
        )


@dataclass(slots=True)
class TrophicConfig:
    specialist_share_threshold: float = 0.64
    animal_channel_threshold: float = 0.14
    animal_use_drive_threshold: float = 0.08
    attack_channel_threshold: float = 0.08
    breadth_penalty: float = 0.7
    breadth_metabolism_penalty: float = 0.28
    breadth_hydration_penalty: float = 0.14
    breadth_reproduction_penalty: float = 0.28
    animal_mode_plant_survival_floor: float = 0.34
    animal_mode_plant_survival_energy_threshold: float = 0.65
    channel_focus_power: float = 3.0
    mode_focus_power: float = 2.1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_fraction(
            "trophic.specialist_share_threshold",
            self.specialist_share_threshold,
        )
        _check_fraction("trophic.animal_channel_threshold", self.animal_channel_threshold)
        _check_fraction("trophic.animal_use_drive_threshold", self.animal_use_drive_threshold)
        _check_fraction("trophic.attack_channel_threshold", self.attack_channel_threshold)
        _check_nonnegative("trophic.breadth_penalty", self.breadth_penalty)
        _check_nonnegative(
            "trophic.breadth_metabolism_penalty",
            self.breadth_metabolism_penalty,
        )
        _check_nonnegative(
            "trophic.breadth_hydration_penalty",
            self.breadth_hydration_penalty,
        )
        _check_nonnegative(
            "trophic.breadth_reproduction_penalty",
            self.breadth_reproduction_penalty,
        )
        _check_fraction(
            "trophic.animal_mode_plant_survival_floor",
            self.animal_mode_plant_survival_floor,
        )
        _check_fraction(
            "trophic.animal_mode_plant_survival_energy_threshold",
            self.animal_mode_plant_survival_energy_threshold,
        )
        _check_positive("trophic.channel_focus_power", self.channel_focus_power)
        _check_positive("trophic.mode_focus_power", self.mode_focus_power)


@dataclass(slots=True)
class BioticFieldConfig:
    diffusion_radius: int = 18
    prey_weight: float = 1.9
    carrion_weight: float = 1.14
    predator_risk_weight: float = 0.24
    hunter_vulnerability_weight: float = 1.08

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_integer("biotic_fields.diffusion_radius", self.diffusion_radius, minimum=1)
        _check_nonnegative("biotic_fields.prey_weight", self.prey_weight)
        _check_nonnegative("biotic_fields.carrion_weight", self.carrion_weight)
        _check_nonnegative("biotic_fields.predator_risk_weight", self.predator_risk_weight)
        _check_nonnegative(
            "biotic_fields.hunter_vulnerability_weight",
            self.hunter_vulnerability_weight,
        )


@dataclass(slots=True)
class TaxonomyConfig:
    min_branch_member_count: int = 10
    min_branch_peak_members: int = 5
    min_branch_persistence_ticks: int = 70
    min_overlap_ticks: int = 28
    min_overlap_members: int = 3
    min_split_gap_ticks: int = 24
    min_genetic_distance: float = 0.17
    min_ecotype_divergence: float = 0.28
    min_ecological_divergence: float = 0.2
    min_split_score: float = 4.6

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_integer(
            "taxonomy.min_branch_member_count",
            self.min_branch_member_count,
            minimum=1,
        )
        _check_integer(
            "taxonomy.min_branch_peak_members",
            self.min_branch_peak_members,
            minimum=1,
        )
        _check_integer(
            "taxonomy.min_branch_persistence_ticks",
            self.min_branch_persistence_ticks,
            minimum=1,
        )
        _check_integer("taxonomy.min_overlap_ticks", self.min_overlap_ticks, minimum=1)
        _check_integer("taxonomy.min_overlap_members", self.min_overlap_members, minimum=1)
        _check_integer("taxonomy.min_split_gap_ticks", self.min_split_gap_ticks, minimum=0)
        _check_positive("taxonomy.min_genetic_distance", self.min_genetic_distance)
        _check_nonnegative("taxonomy.min_ecotype_divergence", self.min_ecotype_divergence)
        _check_nonnegative("taxonomy.min_ecological_divergence", self.min_ecological_divergence)
        _check_nonnegative("taxonomy.min_split_score", self.min_split_score)


@dataclass(slots=True)
class ReproductionConfig:
    min_age: int = 30
    cooldown_ticks: int = 24
    min_hydration_fraction: float = 0.72
    energy_cost: float = 0.52
    animal_mode_energy_requirement_multiplier: float = 0.48
    animal_mode_reproduction_cost_multiplier: float = 0.82
    child_energy_fraction: float = 0.3
    child_hydration_fraction: float = 0.58
    child_health_fraction: float = 0.86
    animal_mode_child_energy_fraction_multiplier: float = 1.6
    animal_mode_child_hydration_fraction_multiplier: float = 1.18
    animal_mode_offspring_trait_stability: float = 0.82
    scavenger_min_health_fraction: float = 0.48

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_integer("reproduction.min_age", self.min_age, minimum=1)
        _check_integer("reproduction.cooldown_ticks", self.cooldown_ticks, minimum=0)
        _check_fraction("reproduction.min_hydration_fraction", self.min_hydration_fraction)
        _check_number(
            "reproduction.energy_cost",
            self.energy_cost,
            minimum=0.0,
            maximum=1.0,
        )
        _check_number(
            "reproduction.animal_mode_energy_requirement_multiplier",
            self.animal_mode_energy_requirement_multiplier,
            minimum=0.0,
            maximum=1.0,
            inclusive_minimum=False,
        )
        _check_number(
            "reproduction.animal_mode_reproduction_cost_multiplier",
            self.animal_mode_reproduction_cost_multiplier,
            minimum=0.0,
            maximum=1.0,
            inclusive_minimum=False,
        )
        _check_fraction("reproduction.child_energy_fraction", self.child_energy_fraction)
        _check_fraction("reproduction.child_hydration_fraction", self.child_hydration_fraction)
        _check_fraction("reproduction.child_health_fraction", self.child_health_fraction)
        _check_positive(
            "reproduction.animal_mode_child_energy_fraction_multiplier",
            self.animal_mode_child_energy_fraction_multiplier,
        )
        _check_positive(
            "reproduction.animal_mode_child_hydration_fraction_multiplier",
            self.animal_mode_child_hydration_fraction_multiplier,
        )
        _check_fraction(
            "reproduction.animal_mode_offspring_trait_stability",
            self.animal_mode_offspring_trait_stability,
        )
        _check_fraction(
            "reproduction.scavenger_min_health_fraction",
            self.scavenger_min_health_fraction,
        )


@dataclass(slots=True)
class DietMatchingConfig:
    reservoir_decay: float = 0.96
    specialist_threshold: float = 0.7
    omnivore_threshold: float = 0.45

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_fraction("diet_matching.reservoir_decay", self.reservoir_decay)
        _check_fraction("diet_matching.specialist_threshold", self.specialist_threshold)
        _check_fraction("diet_matching.omnivore_threshold", self.omnivore_threshold)


@dataclass(slots=True)
class WorldConfig:
    seed: int = 7
    width: int = 48
    height: int = 32
    max_ticks: int = 2_000
    initial_agents: int = 20
    max_agents: int = 320
    water_tile_ratio: float = 0.1
    forest_tile_ratio: float = 0.18
    wetland_tile_ratio: float = 0.08
    rocky_tile_ratio: float = 0.12
    base_energy_drain: float = 0.02
    base_hydration_drain: float = 0.024
    max_age: int = 720
    default_vision_radius: int = 4
    species_distance_threshold: float = 0.42
    environment: EnvironmentFieldConfig = field(default_factory=EnvironmentFieldConfig)
    resources: ResourceRegrowthConfig = field(default_factory=ResourceRegrowthConfig)
    hazards: HazardConfig = field(default_factory=HazardConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    carcasses: CarcassConfig = field(default_factory=CarcassConfig)
    trophic: TrophicConfig = field(default_factory=TrophicConfig)
    biotic_fields: BioticFieldConfig = field(default_factory=BioticFieldConfig)
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    diet_matching: DietMatchingConfig = field(default_factory=DietMatchingConfig)
    climate: ClimateConfig = field(default_factory=ClimateConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_integer("seed", self.seed)
        _check_integer("width", self.width, minimum=1)
        _check_integer("height", self.height, minimum=1)
        _check_integer("max_ticks", self.max_ticks, minimum=1)
        _check_integer("initial_agents", self.initial_agents, minimum=0)
        _check_integer("max_agents", self.max_agents, minimum=1)
        if self.initial_agents > self.max_agents:
            raise ValueError("initial_agents must be <= max_agents")

        terrain_ratios = {
            "water_tile_ratio": self.water_tile_ratio,
            "forest_tile_ratio": self.forest_tile_ratio,
            "wetland_tile_ratio": self.wetland_tile_ratio,
            "rocky_tile_ratio": self.rocky_tile_ratio,
        }
        for name, value in terrain_ratios.items():
            _check_fraction(name, value)
        terrain_ratio_sum = sum(terrain_ratios.values())
        if terrain_ratio_sum > 1.0:
            raise ValueError("terrain tile ratios must sum to <= 1")
        total_tiles = self.width * self.height
        estimated_water_tiles = int(total_tiles * self.water_tile_ratio)
        estimated_land_tiles = total_tiles - estimated_water_tiles
        if self.initial_agents > estimated_land_tiles:
            raise ValueError("initial_agents must fit on estimated land tiles")

        _check_nonnegative("base_energy_drain", self.base_energy_drain)
        _check_nonnegative("base_hydration_drain", self.base_hydration_drain)
        _check_integer("max_age", self.max_age, minimum=1)
        _check_integer("default_vision_radius", self.default_vision_radius, minimum=0)
        _check_positive("species_distance_threshold", self.species_distance_threshold)

        nested_configs = (
            self.environment,
            self.resources,
            self.hazards,
            self.combat,
            self.carcasses,
            self.trophic,
            self.biotic_fields,
            self.taxonomy,
            self.reproduction,
            self.diet_matching,
            self.climate,
        )
        for nested_config in nested_configs:
            nested_config.validate()
        if self.max_age < self.reproduction.min_age:
            raise ValueError("max_age must be >= reproduction.min_age")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
