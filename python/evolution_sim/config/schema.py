from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ClimateConfig:
    season_length: int = 90
    dry_plain_penalty: float = 0.55
    dry_forest_penalty: float = 0.2
    wet_plain_bonus: float = 0.25
    wet_forest_bonus: float = 0.1
    seasonal_hydration_shift: float = 0.01


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


@dataclass(slots=True)
class HazardConfig:
    exposure_damage_rate: float = 0.042
    instability_damage_rate: float = 0.038
    min_hazard_level: float = 0.12
    healing_base_rate: float = 0.032
    healing_hazard_threshold: float = 0.35
    min_energy_ratio_for_healing: float = 0.65
    min_hydration_ratio_for_healing: float = 0.65


@dataclass(slots=True)
class CombatConfig:
    min_attack_health_ratio: float = 0.55
    min_attack_energy_ratio: float = 0.45
    min_attack_hydration_ratio: float = 0.4
    min_reproduction_health_ratio: float = 0.72
    base_attack_damage: float = 0.18
    attack_energy_cost: float = 0.055
    attack_hydration_cost: float = 0.038


@dataclass(slots=True)
class CarcassConfig:
    base_energy: float = 0.24
    health_ratio_yield: float = 0.42
    body_capacity_yield: float = 0.18
    decay_base_rate: float = 0.016
    decay_heat_factor: float = 0.018
    decay_moisture_factor: float = 0.011
    healing_fraction: float = 0.08


@dataclass(slots=True)
class TrophicConfig:
    specialist_share_threshold: float = 0.62
    animal_channel_threshold: float = 0.18
    attack_channel_threshold: float = 0.17
    breadth_penalty: float = 0.24
    breadth_metabolism_penalty: float = 0.08
    breadth_hydration_penalty: float = 0.04


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


@dataclass(slots=True)
class ReproductionConfig:
    min_age: int = 30
    cooldown_ticks: int = 24
    min_hydration_fraction: float = 0.72
    energy_cost: float = 0.52
    child_energy_fraction: float = 0.3
    child_hydration_fraction: float = 0.58
    child_health_fraction: float = 0.86


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
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    climate: ClimateConfig = field(default_factory=ClimateConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
