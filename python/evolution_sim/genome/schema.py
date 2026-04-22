from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random


GENE_LIMITS: dict[str, tuple[float, float]] = {
    "max_energy": (0.7, 1.8),
    "max_hydration": (0.7, 1.8),
    "max_health": (0.7, 1.9),
    "move_cost": (0.02, 0.12),
    "food_efficiency": (0.4, 1.8),
    "water_efficiency": (0.4, 1.8),
    "attack_power": (0.35, 1.8),
    "attack_cost_multiplier": (0.65, 1.45),
    "defense_rating": (0.45, 1.8),
    "meat_efficiency": (0.2, 1.8),
    "healing_efficiency": (0.45, 1.8),
    "plant_bias": (0.45, 1.8),
    "carrion_bias": (0.2, 1.8),
    "live_prey_bias": (0.2, 1.8),
    "forest_affinity": (0.3, 1.8),
    "plain_affinity": (0.3, 1.8),
    "wetland_affinity": (0.3, 1.8),
    "rocky_affinity": (0.3, 1.8),
    "heat_tolerance": (0.4, 1.8),
    "reproduction_threshold": (0.55, 0.98),
    "mutation_scale": (0.01, 0.25),
}


def _clamp(name: str, value: float) -> float:
    lower, upper = GENE_LIMITS[name]
    return max(lower, min(upper, value))


def _mix(lower: float, upper: float, share: float) -> float:
    return lower + (upper - lower) * share


@dataclass(slots=True)
class Genome:
    max_energy: float
    max_hydration: float
    max_health: float
    move_cost: float
    food_efficiency: float
    water_efficiency: float
    attack_power: float
    attack_cost_multiplier: float
    defense_rating: float
    meat_efficiency: float
    healing_efficiency: float
    plant_bias: float
    carrion_bias: float
    live_prey_bias: float
    forest_affinity: float
    plain_affinity: float
    wetland_affinity: float
    rocky_affinity: float
    heat_tolerance: float
    reproduction_threshold: float
    mutation_scale: float

    @classmethod
    def sample_initial(cls, rng: Random) -> "Genome":
        plant_channel = rng.gammavariate(0.7, 1.0)
        scavenger_channel = rng.gammavariate(0.24, 1.0)
        hunter_channel = rng.gammavariate(0.24, 1.0)
        if rng.random() < 0.32:
            boost_roll = rng.random()
            boost_factor = rng.uniform(1.5, 2.5)
            if boost_roll < 0.55:
                plant_channel *= boost_factor
            elif boost_roll < 0.775:
                scavenger_channel *= boost_factor
            else:
                hunter_channel *= boost_factor
        total_channel = max(plant_channel + scavenger_channel + hunter_channel, 1e-9)
        plant_share = plant_channel / total_channel
        scavenger_share = scavenger_channel / total_channel
        hunter_share = hunter_channel / total_channel
        animal_share = scavenger_share + hunter_share
        hunter_mode_share = hunter_share / max(animal_share, 1e-9)
        scavenger_mode_share = scavenger_share / max(animal_share, 1e-9)
        specialization_gap = abs(plant_share - animal_share)
        breadth = 1.0 - specialization_gap
        specialist_push = 0.42 + specialization_gap * 0.58

        def jitter(scale: float) -> float:
            return rng.gauss(0.0, scale)

        return cls(
            max_energy=_clamp(
                "max_energy",
                0.96
                + animal_share * 0.16
                + hunter_mode_share * 0.05
                - breadth * 0.04
                + jitter(0.05),
            ),
            max_hydration=_clamp(
                "max_hydration",
                0.96 + plant_share * 0.12 + scavenger_mode_share * 0.04 + jitter(0.05),
            ),
            max_health=_clamp(
                "max_health",
                0.94
                + animal_share * 0.24
                + hunter_mode_share * 0.16
                - breadth * 0.03
                + jitter(0.06),
            ),
            move_cost=_clamp(
                "move_cost",
                0.037 + breadth * 0.012 + hunter_mode_share * 0.008 + jitter(0.004),
            ),
            food_efficiency=_clamp(
                "food_efficiency",
                _mix(0.42, 1.72, plant_share**1.28 * specialist_push + breadth * 0.12)
                + jitter(0.08),
            ),
            water_efficiency=_clamp(
                "water_efficiency",
                0.82 + plant_share * 0.18 + scavenger_mode_share * 0.04 + jitter(0.05),
            ),
            attack_power=_clamp(
                "attack_power",
                0.38
                + animal_share**1.18
                * (0.18 + hunter_mode_share * 1.24 + specialization_gap * 0.18)
                + jitter(0.08),
            ),
            attack_cost_multiplier=_clamp(
                "attack_cost_multiplier",
                1.2
                - animal_share * (0.08 + hunter_mode_share * 0.42 + specialization_gap * 0.14)
                + jitter(0.03),
            ),
            defense_rating=_clamp(
                "defense_rating",
                0.66
                + animal_share * (0.18 + hunter_mode_share * 0.42 + specialization_gap * 0.12)
                + jitter(0.07),
            ),
            meat_efficiency=_clamp(
                "meat_efficiency",
                _mix(
                    0.22,
                    1.74,
                    animal_share**1.22 * specialist_push + breadth * 0.08,
                )
                + jitter(0.08),
            ),
            healing_efficiency=_clamp(
                "healing_efficiency",
                0.82 + animal_share * 0.1 + scavenger_mode_share * 0.08 + jitter(0.05),
            ),
            plant_bias=_clamp(
                "plant_bias",
                _mix(0.45, 1.8, plant_share**1.36 * (0.82 + specialization_gap * 0.18))
                + jitter(0.09),
            ),
            carrion_bias=_clamp(
                "carrion_bias",
                0.2
                + animal_share
                * (0.16 + scavenger_mode_share * 1.34 + specialization_gap * 0.16)
                + jitter(0.1),
            ),
            live_prey_bias=_clamp(
                "live_prey_bias",
                0.2
                + animal_share
                * (0.16 + hunter_mode_share * 1.34 + specialization_gap * 0.16)
                + jitter(0.1),
            ),
            forest_affinity=rng.uniform(0.75, 1.25),
            plain_affinity=rng.uniform(0.75, 1.25),
            wetland_affinity=rng.uniform(0.75, 1.25),
            rocky_affinity=rng.uniform(0.75, 1.25),
            heat_tolerance=rng.uniform(0.75, 1.25),
            reproduction_threshold=rng.uniform(0.72, 0.94),
            mutation_scale=rng.uniform(0.02, 0.09),
        )

    def mutate(self, rng: Random) -> "Genome":
        sigma = self.mutation_scale
        plant_shift = rng.gauss(0.0, sigma * 0.85)
        scavenger_shift = rng.gauss(0.0, sigma * 0.75)
        hunter_shift = rng.gauss(0.0, sigma * 0.75)
        return Genome(
            max_energy=_clamp("max_energy", self.max_energy + rng.gauss(0.0, sigma)),
            max_hydration=_clamp(
                "max_hydration", self.max_hydration + rng.gauss(0.0, sigma)
            ),
            max_health=_clamp("max_health", self.max_health + rng.gauss(0.0, sigma)),
            move_cost=_clamp(
                "move_cost", self.move_cost + rng.gauss(0.0, sigma * 0.2)
            ),
            food_efficiency=_clamp(
                "food_efficiency",
                self.food_efficiency + plant_shift + rng.gauss(0.0, sigma * 0.35),
            ),
            water_efficiency=_clamp(
                "water_efficiency", self.water_efficiency + rng.gauss(0.0, sigma)
            ),
            attack_power=_clamp(
                "attack_power",
                self.attack_power + hunter_shift * 0.95 + rng.gauss(0.0, sigma * 0.35),
            ),
            attack_cost_multiplier=_clamp(
                "attack_cost_multiplier",
                self.attack_cost_multiplier
                - hunter_shift * 0.18
                + rng.gauss(0.0, sigma * 0.18),
            ),
            defense_rating=_clamp(
                "defense_rating",
                self.defense_rating + hunter_shift * 0.52 + rng.gauss(0.0, sigma * 0.3),
            ),
            meat_efficiency=_clamp(
                "meat_efficiency",
                self.meat_efficiency
                + (scavenger_shift + hunter_shift) * 0.62
                + rng.gauss(0.0, sigma * 0.3),
            ),
            healing_efficiency=_clamp(
                "healing_efficiency", self.healing_efficiency + rng.gauss(0.0, sigma)
            ),
            plant_bias=_clamp(
                "plant_bias",
                self.plant_bias + plant_shift * 1.05 + rng.gauss(0.0, sigma * 0.3),
            ),
            carrion_bias=_clamp(
                "carrion_bias",
                self.carrion_bias + scavenger_shift * 1.08 + rng.gauss(0.0, sigma * 0.28),
            ),
            live_prey_bias=_clamp(
                "live_prey_bias",
                self.live_prey_bias + hunter_shift * 1.08 + rng.gauss(0.0, sigma * 0.28),
            ),
            forest_affinity=_clamp(
                "forest_affinity", self.forest_affinity + rng.gauss(0.0, sigma)
            ),
            plain_affinity=_clamp(
                "plain_affinity", self.plain_affinity + rng.gauss(0.0, sigma)
            ),
            wetland_affinity=_clamp(
                "wetland_affinity", self.wetland_affinity + rng.gauss(0.0, sigma)
            ),
            rocky_affinity=_clamp(
                "rocky_affinity", self.rocky_affinity + rng.gauss(0.0, sigma)
            ),
            heat_tolerance=_clamp(
                "heat_tolerance", self.heat_tolerance + rng.gauss(0.0, sigma)
            ),
            reproduction_threshold=_clamp(
                "reproduction_threshold",
                self.reproduction_threshold + rng.gauss(0.0, sigma),
            ),
            mutation_scale=_clamp(
                "mutation_scale",
                self.mutation_scale + rng.gauss(0.0, sigma * 0.15),
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
