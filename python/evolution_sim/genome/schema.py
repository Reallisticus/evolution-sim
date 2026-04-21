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
        return cls(
            max_energy=rng.uniform(0.9, 1.25),
            max_hydration=rng.uniform(0.9, 1.25),
            max_health=rng.uniform(0.92, 1.28),
            move_cost=rng.uniform(0.03, 0.07),
            food_efficiency=rng.uniform(0.8, 1.2),
            water_efficiency=rng.uniform(0.8, 1.2),
            attack_power=rng.uniform(0.62, 1.24),
            attack_cost_multiplier=rng.uniform(0.88, 1.12),
            defense_rating=rng.uniform(0.82, 1.18),
            meat_efficiency=rng.uniform(0.5, 1.08),
            healing_efficiency=rng.uniform(0.8, 1.14),
            plant_bias=rng.uniform(0.88, 1.28),
            carrion_bias=rng.uniform(0.46, 1.04),
            live_prey_bias=rng.uniform(0.38, 0.92),
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
                "food_efficiency", self.food_efficiency + rng.gauss(0.0, sigma)
            ),
            water_efficiency=_clamp(
                "water_efficiency", self.water_efficiency + rng.gauss(0.0, sigma)
            ),
            attack_power=_clamp(
                "attack_power", self.attack_power + rng.gauss(0.0, sigma)
            ),
            attack_cost_multiplier=_clamp(
                "attack_cost_multiplier",
                self.attack_cost_multiplier + rng.gauss(0.0, sigma * 0.4),
            ),
            defense_rating=_clamp(
                "defense_rating", self.defense_rating + rng.gauss(0.0, sigma)
            ),
            meat_efficiency=_clamp(
                "meat_efficiency", self.meat_efficiency + rng.gauss(0.0, sigma)
            ),
            healing_efficiency=_clamp(
                "healing_efficiency", self.healing_efficiency + rng.gauss(0.0, sigma)
            ),
            plant_bias=_clamp("plant_bias", self.plant_bias + rng.gauss(0.0, sigma)),
            carrion_bias=_clamp(
                "carrion_bias", self.carrion_bias + rng.gauss(0.0, sigma)
            ),
            live_prey_bias=_clamp(
                "live_prey_bias", self.live_prey_bias + rng.gauss(0.0, sigma)
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
