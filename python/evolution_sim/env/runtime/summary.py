from __future__ import annotations

from typing import Any

from evolution_sim.env.contracts import SUMMARY_SCHEMA_VERSION
import evolution_sim.env.runtime.capacity as runtime_capacity
import evolution_sim.env.runtime.reproduction as runtime_reproduction
import evolution_sim.env.runtime.resources as runtime_resources
import evolution_sim.env.runtime.signals as runtime_signals
import evolution_sim.env.runtime.trajectory as runtime_trajectory
from evolution_sim.env.runtime.reporting import (
    build_run_top_species,
    build_species_metric_leaderboards,
    finalize_carcass_run_totals,
    finalize_diet_totals,
    finalize_fresh_kill_run_totals,
    finalize_grouped_animal_resource_opportunity_counts,
    finalize_grouped_diet_totals,
    SurfaceSnapshotContext,
    summary_end_surface_state,
)
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.taxonomy import REPLAY_TAXONOMY_MODE
from evolution_sim.genome.schema import Genome

GENE_SUMMARY_FIELDS = (
    "max_energy",
    "max_health",
    "move_cost",
    "food_efficiency",
    "water_efficiency",
    "attack_power",
    "meat_efficiency",
    "carrion_bias",
    "live_prey_bias",
    "forest_affinity",
    "plain_affinity",
    "wetland_affinity",
    "rocky_affinity",
    "heat_tolerance",
)


def _average_gene(source_genomes: list[Genome], field: str) -> float:
    return round(
        sum(float(getattr(genome, field)) for genome in source_genomes)
        / max(len(source_genomes), 1),
        4,
    )


def _quantile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(sorted_values[0], 4)
    position = (len(sorted_values) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    value = (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )
    return round(value, 4)


def _trait_distribution(source_genomes: list[Genome], field: str) -> dict[str, object]:
    values = sorted(float(getattr(genome, field)) for genome in source_genomes)
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": round(values[0], 4),
        "p10": _quantile(values, 0.1),
        "median": _quantile(values, 0.5),
        "mean": round(sum(values) / len(values), 4),
        "p90": _quantile(values, 0.9),
        "max": round(values[-1], 4),
    }


def _trait_distributions(source_genomes: list[Genome]) -> dict[str, dict[str, object]]:
    return {
        field: _trait_distribution(source_genomes, field)
        for field in GENE_SUMMARY_FIELDS
    }


def _mean_delta(
    terminal_distribution: dict[str, dict[str, object]],
    initial_distribution: dict[str, dict[str, object]],
) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for field in GENE_SUMMARY_FIELDS:
        terminal_mean = terminal_distribution[field]["mean"]
        initial_mean = initial_distribution[field]["mean"]
        if isinstance(terminal_mean, (int, float)) and isinstance(
            initial_mean,
            (int, float),
        ):
            deltas[field] = round(float(terminal_mean) - float(initial_mean), 4)
        else:
            deltas[field] = None
    return deltas


def _gene_averages(world: Any, alive: list[Any]) -> dict[str, float]:
    genomes = [agent.genome for agent in world.agents.values()]
    alive_genomes = [agent.genome for agent in alive]
    historical_gene_averages = {
        f"avg_{field}_gene": _average_gene(genomes, field)
        for field in GENE_SUMMARY_FIELDS
    }
    explicit_historical_gene_averages = {
        f"avg_historical_{field}_gene": value
        for field, value in (
            (field, historical_gene_averages[f"avg_{field}_gene"])
            for field in GENE_SUMMARY_FIELDS
        )
    }
    alive_gene_averages = {
        f"avg_alive_{field}_gene": _average_gene(alive_genomes, field)
        for field in GENE_SUMMARY_FIELDS
    }
    return {
        **historical_gene_averages,
        **explicit_historical_gene_averages,
        **alive_gene_averages,
    }


def _selection_heredity_summary(world: Any, alive: list[Any]) -> dict[str, object]:
    initial_genomes = [
        agent.genome
        for agent in world.agents.values()
        if agent.parent_id is None and agent.birth_tick == 0
    ]
    terminal_genomes = [agent.genome for agent in alive]
    initial_distribution = _trait_distributions(initial_genomes)
    terminal_distribution = _trait_distributions(terminal_genomes)
    return {
        "initial_trait_distributions": initial_distribution,
        "terminal_alive_trait_distributions": terminal_distribution,
        "terminal_minus_initial_mean": _mean_delta(
            terminal_distribution,
            initial_distribution,
        ),
    }


def _lineage_counts(world: Any) -> tuple[dict[int, int], dict[int, int]]:
    lineage_sizes: dict[int, int] = {}
    alive_lineage_sizes: dict[int, int] = {}
    for agent in world.agents.values():
        lineage_sizes[agent.lineage_id] = lineage_sizes.get(agent.lineage_id, 0) + 1
        if agent.alive:
            alive_lineage_sizes[agent.lineage_id] = (
                alive_lineage_sizes.get(agent.lineage_id, 0) + 1
            )
    return lineage_sizes, alive_lineage_sizes


def build_summary(
    world: Any,
    mode: RunMode = RunMode.FULL_REPLAY,
    *,
    surface_snapshot_context: SurfaceSnapshotContext,
) -> dict[str, object]:
    alive = world.alive_agents()
    lineage_sizes, alive_lineage_sizes = _lineage_counts(world)

    field_stats = world._field_stats()
    climate_end = world._climate_state()
    terrain_counts = world._terrain_counts()
    end_surfaces = summary_end_surface_state(
        mode,
        context=surface_snapshot_context,
    )
    hydrology_primary_counts = end_surfaces["hydrology_primary_counts"]
    hydrology_support_counts = end_surfaces["hydrology_support_counts"]
    hydrology_primary_stats = end_surfaces["hydrology_primary_stats"]
    refuge_counts = end_surfaces["refuge_counts"]
    refuge_stats = end_surfaces["refuge_stats"]
    hazard_counts = end_surfaces["hazard_counts"]
    hazard_stats = end_surfaces["hazard_stats"]
    fresh_kill_stats = end_surfaces["fresh_kill_stats"]
    carcass_stats = end_surfaces["carcass_stats"]
    biotic_field_stats = end_surfaces["biotic_field_stats"]
    signal_field_stats = end_surfaces["signal_field_stats"]
    ecology_counts = end_surfaces["ecology_counts"]
    ecology_stats = end_surfaces["ecology_stats"]
    habitat_counts = end_surfaces["habitat_counts"]
    latest_species_metrics = end_surfaces["latest_species_metrics"]
    land_tile_count = world.config.width * world.config.height - terrain_counts["water"]
    trophic_role_counts, meat_mode_counts = world._population_trophic_counts(alive)
    reproduction_end = world._reproduction_readiness_counts(alive)
    reproductive_groups_end = runtime_reproduction.build_reproductive_group_summary(
        world.reproductive_groups,
        world.agents.values(),
    )
    ticks_executed = world.tick + 1
    trophic_lifecycle = world._trophic_lifecycle_summary(
        ticks_executed=ticks_executed
    )
    fresh_kill_totals = finalize_fresh_kill_run_totals(
        world.run_fresh_kill_totals,
        fresh_kill_stats,
    )
    carcass_totals = finalize_carcass_run_totals(
        world.run_carcass_totals,
        carcass_stats,
    )
    fresh_kill_conservation_error = (
        fresh_kill_totals["energy_deposited"]
        - fresh_kill_totals["energy_converted_to_carcass"]
        - fresh_kill_totals["energy_consumed"]
        - fresh_kill_stats["total_fresh_kill_energy"]
    )
    carcass_conservation_error = (
        carcass_totals["energy_deposited"]
        - carcass_totals["energy_decayed"]
        - carcass_totals["energy_consumed"]
        - carcass_stats["total_carcass_energy"]
    )

    summary = {
        "run_id": world.run_id,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "seed": world.config.seed,
        "ticks_executed": ticks_executed,
        "births": world.births,
        "deaths": world.deaths,
        "alive_agents": len(alive),
        "max_agents": world.config.max_agents,
        "max_agent_saturation_at_end": reproduction_end["saturation_at_end"],
        "peak_max_agent_saturation": reproduction_end["peak_saturation"],
        "peak_alive_agents": world.peak_alive_agents,
        "carrying_capacity": runtime_capacity.build_carrying_capacity_summary(
            world,
            ticks_executed=ticks_executed,
        ),
        "extinct": len(alive) == 0,
        "total_agents_seen": len(world.agents),
        "season_at_end": world._season_state()["name"],
        "disturbance_at_end": climate_end["disturbance_type"],
        "disturbance_strength_at_end": climate_end["disturbance_strength"],
        "lineages": sorted({agent.lineage_id for agent in world.agents.values()}),
        "alive_lineages": sorted({agent.lineage_id for agent in alive}),
        "top_lineages": sorted(
            (
                {
                    "lineage_id": lineage_id,
                    "total_agents": size,
                    "alive_agents": alive_lineage_sizes.get(lineage_id, 0),
                }
                for lineage_id, size in lineage_sizes.items()
            ),
            key=lambda item: (
                -item["alive_agents"],
                -item["total_agents"],
                item["lineage_id"],
            ),
        )[:10],
        "reproductive_groups_end": reproductive_groups_end,
        "ecotypes_created": len(world.ecotype_registry),
        "alive_ecotype_count": len(world.current_ecotype_records),
        "last_birth_tick": world.last_birth_tick,
        "field_stats": field_stats,
        "terrain_counts": terrain_counts,
        "land_tile_count": land_tile_count,
        "habitat_state_counts_at_end": habitat_counts,
        "hydrology_primary_counts_at_end": hydrology_primary_counts,
        "hydrology_support_counts_at_end": hydrology_support_counts,
        "hydrology_primary_stats_at_end": hydrology_primary_stats,
        "refuge_counts_at_end": refuge_counts,
        "refuge_stats_at_end": refuge_stats,
        "hazard_counts_at_end": hazard_counts,
        "hazard_stats_at_end": hazard_stats,
        "biotic_field_stats_at_end": biotic_field_stats,
        "signal_field_stats_at_end": signal_field_stats,
        "fresh_kill_stats_at_end": fresh_kill_stats,
        "carcass_stats_at_end": carcass_stats,
        "trophic_role_counts_at_end": trophic_role_counts,
        "meat_mode_counts_at_end": meat_mode_counts,
        "trophic_lifecycle": trophic_lifecycle,
        "reproduction_end": reproduction_end,
        "signal_end": runtime_signals.finalize_signal_totals(
            world.run_signal_totals
        ),
        "combat_end": {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in world.run_combat_totals.items()
        },
        "fresh_kill_end": {
            **{
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in fresh_kill_totals.items()
            },
            "conservation_error": round(fresh_kill_conservation_error, 6),
        },
        "carcass_end": {
            **{
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in carcass_totals.items()
            },
            "conservation_error": round(carcass_conservation_error, 6),
        },
        "diet_end": finalize_diet_totals(world.run_diet_totals),
        "diet_by_trophic_role_end": finalize_grouped_diet_totals(
            world.run_diet_by_trophic_role
        ),
        "diet_by_meat_mode_end": finalize_grouped_diet_totals(
            world.run_diet_by_meat_mode
        ),
        "animal_resource_opportunity_by_meat_mode_end": (
            finalize_grouped_animal_resource_opportunity_counts(
                world.run_animal_resource_opportunity_by_meat_mode
            )
        ),
        "resource_pressure": runtime_resources.finalize_resource_pressure(world),
        "selection_heredity": _selection_heredity_summary(world, alive),
        "ecology_state_counts_at_end": ecology_counts,
        "ecology_stats_at_end": ecology_stats,
        **_gene_averages(world, alive),
    }
    if mode == RunMode.FULL_REPLAY:
        summary.update(
            {
                "taxonomy_mode": REPLAY_TAXONOMY_MODE,
                "mind_contracts": runtime_trajectory.build_trajectory_summary(
                    world.trajectory_records,
                    signal_config=world.config.signals,
                ),
                "species_created": len(world.species_registry),
                "alive_species_count": len(world.current_species_records),
                "alive_species": [
                    record.species_id for record in world.current_species_records
                ],
                "top_species": build_run_top_species(world.species_registry),
            }
        )
        summary.update(build_species_metric_leaderboards(latest_species_metrics))
    return summary
