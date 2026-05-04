from __future__ import annotations

LAND_TERRAINS = ("plain", "forest", "wetland", "rocky")
HYDROLOGY_REASONS = ("none", "adjacent_water", "wetland", "flooded")
HABITAT_STATES = ("stable", "bloom", "flooded", "parched")
ECOLOGY_STATES = ("stable", "lush", "recovering", "depleted")
HAZARD_TYPES = ("none", "exposure", "instability")
TROPHIC_ROLES = ("herbivore", "omnivore", "carnivore")
MEAT_MODES = ("scavenger", "hunter", "mixed")
MEAT_MODE_SERIES = ("none", "scavenger", "hunter", "mixed")
REFUGE_REASONS = ("none", "canopy_refuge")
DIET_SERIES_METRICS = (
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
BIOTIC_FIELD_NAMES = ("prey_biomass", "carrion", "predator_risk")
SIGNAL_FIELD_NAMES = ("reproductive_signal", "communication_signal")
