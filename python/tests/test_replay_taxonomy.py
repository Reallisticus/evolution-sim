from __future__ import annotations

import unittest
from dataclasses import replace
from random import Random

from evolution_sim.config import TaxonomyConfig, WorldConfig
from evolution_sim.env.taxonomy import apply_replay_taxonomy
from evolution_sim.genome import Genome

AGENT_ENCODING = [
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
]


def encode_row(
    *,
    agent_id: int,
    x: int,
    y: int,
    age: int,
    trophic_role: str,
    meat_mode: str,
    ecotype_id: int,
    refuge_score: float,
    tile_vegetation: float,
    tile_recovery_debt: float,
    matched_diet_ratio: float = 0.0,
    soft_refuge_reason: str = "none",
) -> list[object]:
    return [
        agent_id,
        x,
        y,
        1.0,
        0.8,
        1.0,
        0.8,
        1.0,
        1.0,
        0.0,
        age,
        1.0,
        1.0,
        tile_vegetation,
        tile_recovery_debt,
        0,
        trophic_role,
        meat_mode,
        "none",
        "none",
        soft_refuge_reason,
        0,
        refuge_score,
        matched_diet_ratio,
        ecotype_id,
        1,
    ]


def make_viewer(
    *,
    frames: list[dict[str, object]],
    agent_catalog: dict[str, object],
) -> dict[str, object]:
    return {
        "map": {
            "width": 2,
            "height": 2,
            "terrain_codes": [[0, 1], [2, 3]],
            "terrain_legend": {"0": "plain", "1": "forest", "2": "wetland", "3": "rocky"},
            "hazard_legend": {"0": "none", "1": "exposure", "2": "instability"},
            "ecology_legend": {"0": "stable", "1": "lush", "2": "recovering", "3": "depleted"},
        },
        "agent_catalog": agent_catalog,
        "species_catalog": {},
        "ecotype_catalog": {},
        "frames": frames,
        "analytics": {
            "ticks": [frame["tick"] for frame in frames],
            "population": {
                "alive_agents": [frame["alive_agents"] for frame in frames],
                "species_count": [len(frame["species_counts"]) for frame in frames],
                "births": [frame["births"] for frame in frames],
                "deaths": [frame["deaths"] for frame in frames],
                "ecotype_count": [len(frame["ecotype_counts"]) for frame in frames],
            },
            "species_population": {},
            "collapse_events": [],
            "speciation_events": [],
        },
        "agent_encoding": AGENT_ENCODING,
    }


def make_frame(
    tick: int,
    agents: list[list[object]],
    births: int = 0,
    *,
    deaths: int = 0,
    carcass_patches: list[dict[str, object]] | None = None,
    fresh_kill_patches: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "tick": tick,
        "season": "wet",
        "field_state": {"disturbance_type": "none", "disturbance_strength": 0.0},
        "habitat_state_counts": {"stable": 4, "bloom": 0, "flooded": 0, "parched": 0},
        "habitat_state_codes": [[0, 0], [0, 0]],
        "hydrology_primary_counts": {"none": 4, "adjacent_water": 0, "wetland": 0, "flooded": 0},
        "hydrology_primary_codes": [[0, 0], [0, 0]],
        "hydrology_primary_stats": {"hard_access_tiles": 0},
        "hydrology_support_counts": {
            "shoreline_support": 0,
            "wetland_support": 0,
            "flooded_support": 0,
        },
        "hydrology_support_codes": [[0, 0], [0, 0]],
        "refuge_counts": {"none": 4, "canopy_refuge": 0},
        "refuge_codes": [[0, 0], [0, 0]],
        "refuge_score_codes": [[20, 60], [70, 10]],
        "refuge_stats": {"avg_refuge_score_forest_tiles": 0.4},
        "hazard_counts": {"none": 4, "exposure": 0, "instability": 0},
        "hazard_type_codes": [[0, 0], [0, 0]],
        "hazard_level_codes": [[0, 0], [0, 0]],
        "hazard_stats": {"hazardous_tiles": 0, "avg_hazard_level": 0.0},
        "biotic_fields": {
            "prey_biomass": [[0.0, 0.0], [0.0, 0.0]],
            "carrion": [[0.0, 0.0], [0.0, 0.0]],
            "predator_risk": [[0.0, 0.0], [0.0, 0.0]],
        },
        "biotic_field_stats": {
            "prey_biomass_peak": 0.0,
            "carrion_peak": 0.0,
            "predator_risk_peak": 0.0,
        },
        "fresh_kill_energy_codes": [[0, 0], [0, 0]],
        "fresh_kill_stats": {"fresh_kill_tiles": 0, "total_fresh_kill_energy": 0.0},
        "fresh_kill_flow": {
            "deposition_events": 0,
            "fresh_kill_energy_deposited": 0.0,
            "consumption_events": 0,
            "fresh_kill_energy_consumed": 0.0,
            "fresh_kill_gained_energy": 0.0,
        },
        "fresh_kill_patches": fresh_kill_patches or [],
        "carcass_energy_codes": [[0, 0], [0, 0]],
        "carcass_freshness_codes": [[0, 0], [0, 0]],
        "carcass_stats": {"carcass_tiles": 0, "total_carcass_energy": 0.0},
        "carcass_patches": carcass_patches or [],
        "carcass_flow": {
            "deposition_events": 0,
            "carcass_energy_deposited": 0.0,
            "carcass_energy_decayed": 0.0,
            "consumption_events": 0,
            "carcass_energy_consumed": 0.0,
            "carcass_gained_energy": 0.0,
        },
        "combat_stats": {
            "attack_attempts": 0,
            "successful_attacks": 0,
            "kills": 0,
            "damage_dealt": 0.0,
            "attack_damage_taken": 0.0,
            "hazard_damage_taken": 0.0,
            "carcass_consumption_events": 0,
            "carcass_energy_consumed": 0.0,
            "carcass_gained_energy": 0.0,
        },
        "diet_stats": {
            "plant_events": 0,
            "plant_energy": 0.0,
            "fresh_kill_events": 0,
            "fresh_kill_energy": 0.0,
            "carcass_events": 0,
            "carcass_energy": 0.0,
            "animal_events": 0,
            "animal_energy": 0.0,
            "plant_energy_share": 0.0,
            "fresh_kill_energy_share": 0.0,
            "carcass_energy_share": 0.0,
            "animal_energy_share": 0.0,
        },
        "diet_by_trophic_role": {},
        "diet_by_meat_mode": {},
        "trophic_role_counts": {"herbivore": 0, "omnivore": 0, "carnivore": 0},
        "meat_mode_counts": {"scavenger": 0, "hunter": 0, "mixed": 0},
        "trophic_role_codes": [[1, 2], [2, 1]],
        "meat_mode_codes": [[0, 1], [3, 2]],
        "ecology_state_counts": {"stable": 4, "lush": 0, "recovering": 0, "depleted": 0},
        "ecology_state_codes": [[0, 0], [0, 0]],
        "ecology_stats": {"avg_vegetation": 0.5, "avg_recovery_debt": 0.25},
        "alive_agents": len(agents),
        "births": births,
        "deaths": deaths,
        "trait_means": {},
        "species_metrics": {},
        "ecotype_metrics": {},
        "species_counts": [[1, len(agents)]],
        "ecotype_counts": [],
        "agents": agents,
    }


class ReplayTaxonomyTests(unittest.TestCase):
    def _config(self, **taxonomy_overrides: object) -> WorldConfig:
        return replace(
            WorldConfig(seed=1, max_ticks=8),
            taxonomy=replace(TaxonomyConfig(), **taxonomy_overrides),
        )

    def _base_genomes(self) -> tuple[Genome, Genome, Genome]:
        rng = Random(7)
        founder = Genome.sample_initial(rng)
        daughter = replace(
            founder,
            forest_affinity=1.7,
            plant_bias=0.7,
            carrion_bias=1.35,
            live_prey_bias=1.15,
            heat_tolerance=1.35,
        )
        nested = replace(
            daughter,
            wetland_affinity=1.75,
            forest_affinity=0.72,
            water_efficiency=1.55,
            attack_power=1.45,
            carrion_bias=0.9,
        )
        return founder, daughter, nested

    def _provenance_rewrite_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], int, int]:
        founder, daughter, _nested = self._base_genomes()
        carcass_after_death = [
            {
                "source_agent_id": 2,
                "death_tick": 4,
                "source_species": 1,
                "energy": 0.18,
            }
        ]
        carcass_after_feed = [
            {
                "source_agent_id": 2,
                "death_tick": 4,
                "source_species": 1,
                "energy": 0.12,
            }
        ]
        fresh_kill_after_feed = [
            {
                "source_agent_id": 2,
                "death_tick": 4,
                "source_species": 1,
                "killer_id": 1,
                "energy": 0.08,
            }
        ]
        frames = [
            make_frame(0, [encode_row(agent_id=1, x=0, y=0, age=0, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4)]),
            make_frame(1, [
                encode_row(agent_id=1, x=0, y=0, age=1, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=0, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, matched_diet_ratio=0.9, soft_refuge_reason="canopy_refuge"),
            ], births=1),
            make_frame(2, [
                encode_row(agent_id=1, x=0, y=0, age=2, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=1, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, matched_diet_ratio=0.9, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=3, x=0, y=0, age=0, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
            ], births=1),
            make_frame(3, [
                encode_row(agent_id=1, x=0, y=0, age=3, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=2, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, matched_diet_ratio=0.92, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=3, x=0, y=0, age=1, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=4, x=1, y=0, age=0, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, matched_diet_ratio=0.94, soft_refuge_reason="canopy_refuge"),
            ], births=1),
            make_frame(
                4,
                [
                    encode_row(agent_id=1, x=0, y=0, age=4, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                    encode_row(agent_id=3, x=0, y=0, age=2, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                    encode_row(agent_id=4, x=1, y=0, age=1, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, matched_diet_ratio=0.96, soft_refuge_reason="canopy_refuge"),
                ],
                deaths=1,
                carcass_patches=[
                    {
                        "x": 1,
                        "y": 0,
                        "deposit_count": 1,
                        "total_energy": 0.12,
                        "avg_freshness": 0.82,
                        "dominant_source_species": 1,
                        "mixed_sources": False,
                        "source_breakdown": carcass_after_feed,
                    }
                ],
                fresh_kill_patches=[
                    {
                        "x": 1,
                        "y": 0,
                        "deposit_count": 1,
                        "total_energy": 0.08,
                        "dominant_source_species": 1,
                        "mixed_sources": False,
                        "source_breakdown": fresh_kill_after_feed,
                    }
                ],
            ),
        ]
        agent_catalog = {
            "1": {"agent_id": 1, "parent_id": None, "lineage_id": 1, "birth_tick": 0, "death_tick": None, "genome": founder.to_dict()},
            "2": {"agent_id": 2, "parent_id": 1, "lineage_id": 1, "birth_tick": 1, "death_tick": 4, "genome": daughter.to_dict()},
            "3": {"agent_id": 3, "parent_id": 1, "lineage_id": 1, "birth_tick": 2, "death_tick": None, "genome": founder.to_dict()},
            "4": {"agent_id": 4, "parent_id": 2, "lineage_id": 1, "birth_tick": 3, "death_tick": None, "genome": daughter.to_dict()},
        }
        events = [
            {"tick": 1, "type": "agent_reproduced", "agent_id": 1, "data": {"child_id": 2}},
            {"tick": 2, "type": "agent_reproduced", "agent_id": 1, "data": {"child_id": 3}},
            {"tick": 3, "type": "agent_reproduced", "agent_id": 2, "data": {"child_id": 4}},
            {
                "tick": 4,
                "type": "agent_died",
                "agent_id": 2,
                "data": {
                    "age": 2,
                    "energy": 0.44,
                    "hydration": 0.72,
                    "health": 0.0,
                    "cause": "attack",
                    "killer_id": 1,
                    "x": 1,
                    "y": 0,
                    "source_species": 1,
                    "carcass_energy": 0.18,
                    "fresh_kill_energy": 0.12,
                    "tile_carcass_energy_after": 0.18,
                    "tile_avg_freshness_after": 0.84,
                    "tile_deposit_count_after": 1,
                    "tile_mixed_sources_after": False,
                    "tile_dominant_source_species_after": 1,
                    "tile_source_breakdown_after": carcass_after_death,
                    "tile_fresh_kill_energy_after": 0.12,
                },
            },
            {
                "tick": 4,
                "type": "carcass_deposited",
                "agent_id": 2,
                "data": {
                    "source_agent_id": 2,
                    "source_species": 1,
                    "deposited_energy": 0.18,
                    "cause": "attack",
                    "killer_id": 1,
                    "x": 1,
                    "y": 0,
                    "tile_carcass_energy_after": 0.18,
                    "tile_avg_freshness_after": 0.84,
                    "tile_deposit_count_after": 1,
                    "tile_mixed_sources_after": False,
                    "tile_dominant_source_species_after": 1,
                    "tile_source_breakdown_after": carcass_after_death,
                    "source_breakdown": carcass_after_death,
                },
            },
            {
                "tick": 4,
                "type": "agent_ate",
                "agent_id": 1,
                "data": {
                    "food_source": "carcass",
                    "consumed": 0.06,
                    "energy_before": 0.72,
                    "energy": 0.78,
                    "gained_energy": 0.06,
                    "potential_gained_energy": 0.06,
                    "trophic_role": "herbivore",
                    "meat_mode": "scavenger",
                    "immediate_kill_feed": False,
                    "health": 1.0,
                    "matched_diet_ratio": 0.4,
                    "x": 1,
                    "y": 0,
                    "source_breakdown": [
                        {
                            "source_agent_id": 2,
                            "death_tick": 4,
                            "source_species": 1,
                            "energy": 0.06,
                        }
                    ],
                    "deposit_breakdown": [
                        {
                            "source_agent_id": 2,
                            "death_tick": 4,
                            "source_species": 1,
                            "consumed": 0.06,
                            "freshness": 0.84,
                            "cause": "attack",
                        }
                    ],
                    "tile_mixed_sources_after": False,
                    "tile_dominant_source_species_after": 1,
                    "tile_source_breakdown_after": carcass_after_feed,
                    "tile_carcass_energy_after": 0.12,
                    "tile_avg_freshness_after": 0.82,
                },
            },
            {
                "tick": 4,
                "type": "agent_ate",
                "agent_id": 4,
                "data": {
                    "food_source": "fresh_kill",
                    "consumed": 0.04,
                    "energy_before": 0.78,
                    "energy": 0.82,
                    "gained_energy": 0.04,
                    "potential_gained_energy": 0.04,
                    "trophic_role": "carnivore",
                    "meat_mode": "hunter",
                    "immediate_kill_feed": True,
                    "health": 1.0,
                    "matched_diet_ratio": 0.96,
                    "x": 1,
                    "y": 0,
                    "source_breakdown": [
                        {
                            "source_agent_id": 2,
                            "death_tick": 4,
                            "source_species": 1,
                            "killer_id": 1,
                            "energy": 0.04,
                        }
                    ],
                    "deposit_breakdown": [
                        {
                            "source_agent_id": 2,
                            "death_tick": 4,
                            "source_species": 1,
                            "killer_id": 1,
                            "consumed": 0.04,
                        }
                    ],
                    "tile_mixed_sources_after": False,
                    "tile_dominant_source_species_after": 1,
                    "tile_source_breakdown_after": fresh_kill_after_feed,
                    "tile_fresh_kill_energy_after": 0.08,
                },
            },
        ]
        summary = {
            "run_id": "synthetic-provenance",
            "seed": 7,
            "alive_lineages": [1],
            "top_species_by_realized_carcass_share": [{"species_id": 999}],
        }
        viewer = make_viewer(frames=frames, agent_catalog=agent_catalog)

        updated_summary, updated_viewer = apply_replay_taxonomy(
            config=self._config(
                min_branch_member_count=2,
                min_branch_peak_members=2,
                min_branch_persistence_ticks=3,
                min_overlap_ticks=3,
                min_overlap_members=1,
                min_split_gap_ticks=0,
                min_genetic_distance=0.08,
                min_ecotype_divergence=0.1,
                min_ecological_divergence=0.1,
                min_split_score=3.2,
            ),
            summary=summary,
            events=events,
            viewer=viewer,
        )
        speciation_event = updated_viewer["taxonomy"]["events"][0]
        return (
            updated_summary,
            updated_viewer,
            events,
            speciation_event["continuation_species_id"],
            speciation_event["daughter_species_id"],
        )

    def test_branch_split_creates_durable_continuation_and_daughter_species(self) -> None:
        founder, daughter, _nested = self._base_genomes()
        frames = [
            make_frame(0, [encode_row(agent_id=1, x=0, y=0, age=0, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4)]),
            make_frame(1, [
                encode_row(agent_id=1, x=0, y=0, age=1, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=0, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
            ], births=1),
            make_frame(2, [
                encode_row(agent_id=1, x=0, y=0, age=2, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=1, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=3, x=0, y=0, age=0, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
            ], births=1),
            make_frame(3, [
                encode_row(agent_id=1, x=0, y=0, age=3, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=2, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=3, x=0, y=0, age=1, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=4, x=1, y=0, age=0, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
            ], births=1),
            make_frame(4, [
                encode_row(agent_id=1, x=0, y=0, age=4, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=3, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=3, x=0, y=0, age=2, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=4, x=1, y=0, age=1, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
            ]),
        ]
        agent_catalog = {
            "1": {"agent_id": 1, "parent_id": None, "lineage_id": 1, "birth_tick": 0, "death_tick": None, "genome": founder.to_dict()},
            "2": {"agent_id": 2, "parent_id": 1, "lineage_id": 1, "birth_tick": 1, "death_tick": None, "genome": daughter.to_dict()},
            "3": {"agent_id": 3, "parent_id": 1, "lineage_id": 1, "birth_tick": 2, "death_tick": None, "genome": founder.to_dict()},
            "4": {"agent_id": 4, "parent_id": 2, "lineage_id": 1, "birth_tick": 3, "death_tick": None, "genome": daughter.to_dict()},
        }
        events = [
            {"tick": 1, "type": "agent_reproduced", "agent_id": 1, "data": {"child_id": 2}},
            {"tick": 2, "type": "agent_reproduced", "agent_id": 1, "data": {"child_id": 3}},
            {"tick": 3, "type": "agent_reproduced", "agent_id": 2, "data": {"child_id": 4}},
        ]
        summary = {"run_id": "synthetic-1", "seed": 1, "alive_lineages": [1]}
        viewer = make_viewer(frames=frames, agent_catalog=agent_catalog)

        updated_summary, updated_viewer = apply_replay_taxonomy(
            config=self._config(
                min_branch_member_count=2,
                min_branch_peak_members=2,
                min_branch_persistence_ticks=3,
                min_overlap_ticks=3,
                min_overlap_members=1,
                min_split_gap_ticks=0,
                min_genetic_distance=0.08,
                min_ecotype_divergence=0.1,
                min_ecological_divergence=0.1,
                min_split_score=3.2,
            ),
            summary=summary,
            events=events,
            viewer=viewer,
        )

        self.assertEqual(updated_summary["taxonomy_mode"], "replay_clade_species_v2")
        self.assertEqual(updated_viewer["taxonomy"]["version"], "replay_clade_species_v2")
        self.assertEqual(updated_summary["species_created"], 3)
        self.assertEqual(updated_summary["alive_species_count"], 2)
        self.assertEqual(updated_summary["speciation_events"], 1)
        speciation_event = updated_viewer["taxonomy"]["events"][0]
        continuation_id = speciation_event["continuation_species_id"]
        daughter_id = speciation_event["daughter_species_id"]

        frame_species = []
        species_index = updated_viewer["agent_encoding"].index("species_id")
        agent_index = updated_viewer["agent_encoding"].index("agent_id")
        for frame in updated_viewer["frames"]:
            frame_species.append({row[agent_index]: row[species_index] for row in frame["agents"]})

        self.assertEqual(frame_species[0][1], 1)
        self.assertEqual(frame_species[1][1], continuation_id)
        self.assertEqual(frame_species[1][2], daughter_id)
        self.assertEqual(frame_species[2][3], continuation_id)
        self.assertEqual(frame_species[3][4], daughter_id)
        self.assertEqual(updated_viewer["species_catalog"]["1"]["last_seen_tick"], 0)
        self.assertEqual(updated_viewer["species_catalog"][str(daughter_id)]["parent_species_id"], 1)

    def test_nested_split_reassigns_branch_only_at_logged_event_ticks(self) -> None:
        founder, daughter, nested = self._base_genomes()
        frames = [
            make_frame(0, [encode_row(agent_id=1, x=0, y=0, age=0, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4)]),
            make_frame(1, [
                encode_row(agent_id=1, x=0, y=0, age=1, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=0, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
            ], births=1),
            make_frame(2, [
                encode_row(agent_id=1, x=0, y=0, age=2, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=1, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=3, x=0, y=0, age=0, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=4, x=1, y=0, age=0, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
            ], births=2),
            make_frame(3, [
                encode_row(agent_id=1, x=0, y=0, age=3, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=2, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=3, x=0, y=0, age=1, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=4, x=1, y=0, age=1, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=5, x=0, y=1, age=0, trophic_role="omnivore", meat_mode="mixed", ecotype_id=3, refuge_score=0.7, tile_vegetation=0.8, tile_recovery_debt=0.05),
            ], births=1),
            make_frame(4, [
                encode_row(agent_id=1, x=0, y=0, age=4, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=3, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=3, x=0, y=0, age=2, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=4, x=1, y=0, age=2, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=5, x=0, y=1, age=1, trophic_role="omnivore", meat_mode="mixed", ecotype_id=3, refuge_score=0.7, tile_vegetation=0.8, tile_recovery_debt=0.05),
                encode_row(agent_id=6, x=0, y=1, age=0, trophic_role="omnivore", meat_mode="mixed", ecotype_id=3, refuge_score=0.7, tile_vegetation=0.8, tile_recovery_debt=0.05),
            ], births=1),
            make_frame(5, [
                encode_row(agent_id=1, x=0, y=0, age=5, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=4, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=3, x=0, y=0, age=3, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=4, x=1, y=0, age=3, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
                encode_row(agent_id=5, x=0, y=1, age=2, trophic_role="omnivore", meat_mode="mixed", ecotype_id=3, refuge_score=0.7, tile_vegetation=0.8, tile_recovery_debt=0.05),
                encode_row(agent_id=6, x=0, y=1, age=1, trophic_role="omnivore", meat_mode="mixed", ecotype_id=3, refuge_score=0.7, tile_vegetation=0.8, tile_recovery_debt=0.05),
            ]),
        ]
        agent_catalog = {
            "1": {"agent_id": 1, "parent_id": None, "lineage_id": 1, "birth_tick": 0, "death_tick": None, "genome": founder.to_dict()},
            "2": {"agent_id": 2, "parent_id": 1, "lineage_id": 1, "birth_tick": 1, "death_tick": None, "genome": daughter.to_dict()},
            "3": {"agent_id": 3, "parent_id": 1, "lineage_id": 1, "birth_tick": 2, "death_tick": None, "genome": founder.to_dict()},
            "4": {"agent_id": 4, "parent_id": 2, "lineage_id": 1, "birth_tick": 2, "death_tick": None, "genome": daughter.to_dict()},
            "5": {"agent_id": 5, "parent_id": 2, "lineage_id": 1, "birth_tick": 3, "death_tick": None, "genome": nested.to_dict()},
            "6": {"agent_id": 6, "parent_id": 5, "lineage_id": 1, "birth_tick": 4, "death_tick": None, "genome": nested.to_dict()},
        }
        events = [
            {"tick": 1, "type": "agent_reproduced", "agent_id": 1, "data": {"child_id": 2}},
            {"tick": 2, "type": "agent_reproduced", "agent_id": 1, "data": {"child_id": 3}},
            {"tick": 2, "type": "agent_reproduced", "agent_id": 2, "data": {"child_id": 4}},
            {"tick": 3, "type": "agent_reproduced", "agent_id": 2, "data": {"child_id": 5}},
            {"tick": 4, "type": "agent_reproduced", "agent_id": 5, "data": {"child_id": 6}},
        ]
        summary = {"run_id": "synthetic-2", "seed": 2, "alive_lineages": [1]}
        viewer = make_viewer(frames=frames, agent_catalog=agent_catalog)

        updated_summary, updated_viewer = apply_replay_taxonomy(
            config=self._config(
                min_branch_member_count=2,
                min_branch_peak_members=2,
                min_branch_persistence_ticks=2,
                min_overlap_ticks=2,
                min_overlap_members=1,
                min_split_gap_ticks=0,
                min_genetic_distance=0.08,
                min_ecotype_divergence=0.08,
                min_ecological_divergence=0.08,
                min_split_score=3.0,
            ),
            summary=summary,
            events=events,
            viewer=viewer,
        )

        self.assertEqual(updated_summary["speciation_events"], 2)
        species_index = updated_viewer["agent_encoding"].index("species_id")
        agent_index = updated_viewer["agent_encoding"].index("agent_id")
        frame_species = [
            {row[agent_index]: row[species_index] for row in frame["agents"]}
            for frame in updated_viewer["frames"]
        ]
        event_ticks = {event["tick"] for event in updated_viewer["taxonomy"]["events"]}
        agent_two_history = [
            (frame["tick"], mapping[2])
            for frame, mapping in zip(updated_viewer["frames"], frame_species, strict=False)
            if 2 in mapping
        ]
        changes: list[tuple[int, int, int]] = []
        previous_species_id = None
        for tick, species_id in agent_two_history:
            if previous_species_id is not None and species_id != previous_species_id:
                changes.append((tick, previous_species_id, species_id))
            previous_species_id = species_id
        self.assertEqual([tick for tick, _old, _new in changes], [3])
        self.assertTrue(all(tick in event_ticks for tick, _old, _new in changes))
        species_catalog = updated_viewer["species_catalog"]
        for _tick, old_species_id, new_species_id in changes:
            self.assertEqual(
                species_catalog[str(new_species_id)]["parent_species_id"],
                old_species_id,
            )
        self.assertGreater(updated_summary["species_created"], 3)

    def test_transient_branch_does_not_split(self) -> None:
        founder, daughter, _nested = self._base_genomes()
        frames = [
            make_frame(0, [encode_row(agent_id=1, x=0, y=0, age=0, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4)]),
            make_frame(1, [
                encode_row(agent_id=1, x=0, y=0, age=1, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4),
                encode_row(agent_id=2, x=1, y=0, age=0, trophic_role="carnivore", meat_mode="hunter", ecotype_id=2, refuge_score=0.6, tile_vegetation=0.7, tile_recovery_debt=0.1, soft_refuge_reason="canopy_refuge"),
            ], births=1),
            make_frame(2, [encode_row(agent_id=1, x=0, y=0, age=2, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4)]),
            make_frame(3, [encode_row(agent_id=1, x=0, y=0, age=3, trophic_role="herbivore", meat_mode="scavenger", ecotype_id=1, refuge_score=0.2, tile_vegetation=0.3, tile_recovery_debt=0.4)]),
        ]
        agent_catalog = {
            "1": {"agent_id": 1, "parent_id": None, "lineage_id": 1, "birth_tick": 0, "death_tick": None, "genome": founder.to_dict()},
            "2": {"agent_id": 2, "parent_id": 1, "lineage_id": 1, "birth_tick": 1, "death_tick": 2, "genome": daughter.to_dict()},
        }
        events = [
            {"tick": 1, "type": "agent_reproduced", "agent_id": 1, "data": {"child_id": 2}},
            {"tick": 1, "type": "agent_died", "agent_id": 2, "data": {"carcass_energy": 0.0}},
        ]
        summary = {"run_id": "synthetic-3", "seed": 3, "alive_lineages": [1]}
        viewer = make_viewer(frames=frames, agent_catalog=agent_catalog)

        updated_summary, updated_viewer = apply_replay_taxonomy(
            config=self._config(
                min_branch_member_count=2,
                min_branch_peak_members=1,
                min_branch_persistence_ticks=2,
                min_overlap_ticks=2,
                min_overlap_members=1,
                min_split_gap_ticks=0,
                min_genetic_distance=0.05,
                min_ecotype_divergence=0.0,
                min_ecological_divergence=0.0,
                min_split_score=2.5,
            ),
            summary=summary,
            events=events,
            viewer=viewer,
        )

        self.assertEqual(updated_summary["species_created"], 1)
        self.assertEqual(updated_summary["speciation_events"], 0)
        self.assertEqual(updated_viewer["taxonomy"]["events"], [])

    def test_replay_rewrites_carcass_provenance_to_replay_species(self) -> None:
        _summary, updated_viewer, events, _continuation_id, daughter_id = (
            self._provenance_rewrite_fixture()
        )

        death_event = next(event for event in events if event["type"] == "agent_died")
        deposit_event = next(event for event in events if event["type"] == "carcass_deposited")
        carcass_feed_event = next(
            event
            for event in events
            if event["type"] == "agent_ate" and event["data"]["food_source"] == "carcass"
        )
        final_carcass_patch = updated_viewer["frames"][-1]["carcass_patches"][0]

        self.assertEqual(death_event["data"]["source_species"], daughter_id)
        self.assertEqual(
            death_event["data"]["tile_dominant_source_species_after"],
            daughter_id,
        )
        self.assertEqual(deposit_event["data"]["source_species"], daughter_id)
        self.assertEqual(
            deposit_event["data"]["source_breakdown"][0]["source_species"],
            daughter_id,
        )
        self.assertEqual(
            carcass_feed_event["data"]["source_breakdown"][0]["source_species"],
            daughter_id,
        )
        self.assertEqual(
            carcass_feed_event["data"]["deposit_breakdown"][0]["source_species"],
            daughter_id,
        )
        self.assertEqual(
            carcass_feed_event["data"]["tile_dominant_source_species_after"],
            daughter_id,
        )
        self.assertEqual(
            final_carcass_patch["dominant_source_species"],
            daughter_id,
        )
        self.assertEqual(
            final_carcass_patch["source_breakdown"][0]["source_species"],
            daughter_id,
        )

    def test_replay_rewrites_fresh_kill_provenance_to_replay_species(self) -> None:
        _summary, updated_viewer, events, _continuation_id, daughter_id = (
            self._provenance_rewrite_fixture()
        )

        fresh_kill_feed_event = next(
            event
            for event in events
            if event["type"] == "agent_ate" and event["data"]["food_source"] == "fresh_kill"
        )
        final_fresh_kill_patch = updated_viewer["frames"][-1]["fresh_kill_patches"][0]

        self.assertEqual(
            fresh_kill_feed_event["data"]["source_breakdown"][0]["source_species"],
            daughter_id,
        )
        self.assertEqual(
            fresh_kill_feed_event["data"]["deposit_breakdown"][0]["source_species"],
            daughter_id,
        )
        self.assertEqual(
            fresh_kill_feed_event["data"]["tile_dominant_source_species_after"],
            daughter_id,
        )
        self.assertEqual(
            final_fresh_kill_patch["dominant_source_species"],
            daughter_id,
        )
        self.assertEqual(
            final_fresh_kill_patch["source_breakdown"][0]["source_species"],
            daughter_id,
        )

    def test_split_source_species_does_not_emit_false_extinction(self) -> None:
        _summary, updated_viewer, _events, _continuation_id, _daughter_id = (
            self._provenance_rewrite_fixture()
        )

        speciation_event = updated_viewer["taxonomy"]["events"][0]
        false_extinctions = [
            event
            for event in updated_viewer["analytics"]["collapse_events"]
            if event["type"] == "extinction"
            and event["species_id"] == speciation_event["source_species_id"]
            and event["tick"] == speciation_event["tick"]
        ]
        self.assertEqual(false_extinctions, [])

    def test_summary_carcass_leaderboard_rebuilds_from_post_taxonomy_metrics(self) -> None:
        updated_summary, updated_viewer, _events, _continuation_id, _daughter_id = (
            self._provenance_rewrite_fixture()
        )

        latest_species_metrics = updated_viewer["frames"][-1]["species_metrics"]
        expected_species_ids = [
            int(item["species_id"])
            for item in sorted(
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
            )[:10]
        ]

        self.assertNotEqual(
            updated_summary["top_species_by_realized_carcass_share"],
            [{"species_id": 999}],
        )
        self.assertEqual(
            [
                int(item["species_id"])
                for item in updated_summary["top_species_by_realized_carcass_share"]
            ],
            expected_species_ids,
        )


if __name__ == "__main__":
    unittest.main()
