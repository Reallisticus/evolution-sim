from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evolution_sim.env.events import EventType
from evolution_sim.env.runtime.state import CarcassDeposit, FreshKillDeposit


NON_LAND_ECOLOGY_CODE = -1
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


@dataclass(frozen=True)
class DeathResourceEmission:
    carcass_energy: float
    carcass_tile_state: dict[str, object]
    fresh_kill_tile_state: dict[str, object]
    deposited_resource: str | None


@dataclass(frozen=True, slots=True)
class ResourceContext:
    effective_tile_fields: Callable[[int, int], tuple[float, float, float]]
    habitat_state_at: Callable[[int, int], str]
    habitat_state_grid: Callable[[], tuple[list[list[str]], dict[str, int]]]
    terrain_neighbor_ratio: Callable[..., float]
    season_state: Callable[[], dict[str, object]]
    clamp01: Callable[[float], float]
    emit: Callable[[EventType, int | None, dict[str, object] | None], None]


def empty_resource_pressure_totals() -> dict[str, float]:
    return {
        "plant_energy_created": 0.0,
        "plant_energy_removed": 0.0,
        "plant_energy_lost": 0.0,
        "metabolism_energy_spent": 0.0,
        "movement_energy_spent": 0.0,
        "attack_energy_spent": 0.0,
        "reproduction_energy_spent": 0.0,
    }


def _record_resource_pressure_accounting_update(world: Any) -> None:
    counters = getattr(world, "runtime_cost_counters", None)
    if isinstance(counters, dict):
        counters["resource_pressure_accounting_updates"] = (
            int(counters.get("resource_pressure_accounting_updates", 0)) + 1
        )


def record_plant_created(world: Any, amount: float) -> None:
    if amount > 0:
        world.run_resource_pressure_totals["plant_energy_created"] += amount
        _record_resource_pressure_accounting_update(world)


def record_plant_removed(world: Any, amount: float) -> None:
    if amount > 0:
        world.run_resource_pressure_totals["plant_energy_removed"] += amount
        _record_resource_pressure_accounting_update(world)


def record_plant_lost(world: Any, amount: float) -> None:
    if amount > 0:
        world.run_resource_pressure_totals["plant_energy_lost"] += amount
        _record_resource_pressure_accounting_update(world)


def record_energy_spent(world: Any, channel: str, amount: float) -> None:
    if amount <= 0:
        return
    key = f"{channel}_energy_spent"
    if key not in world.run_resource_pressure_totals:
        raise ValueError(f"Unsupported energy-spend channel: {channel}")
    world.run_resource_pressure_totals[key] += amount
    _record_resource_pressure_accounting_update(world)


def finalize_resource_pressure(world: Any) -> dict[str, object]:
    totals = world.run_resource_pressure_totals
    signal_energy_spent = float(world.run_signal_totals.get("energy_spent", 0.0))
    plant_created = float(totals.get("plant_energy_created", 0.0))
    plant_removed = float(totals.get("plant_energy_removed", 0.0))
    plant_lost = float(totals.get("plant_energy_lost", 0.0))
    plant_available = sum(
        tile.food
        for row in world.grid
        for tile in row
        if tile.terrain != "water"
    )
    energy_spend = {
        "metabolism": float(totals.get("metabolism_energy_spent", 0.0)),
        "movement": float(totals.get("movement_energy_spent", 0.0)),
        "attack": float(totals.get("attack_energy_spent", 0.0)),
        "reproduction": float(totals.get("reproduction_energy_spent", 0.0)),
        "signal": signal_energy_spent,
    }
    total_energy_spent = sum(energy_spend.values())
    return {
        "plant_budget": {
            "energy_created": round(plant_created, 4),
            "energy_removed": round(plant_removed, 4),
            "energy_lost": round(plant_lost, 4),
            "net_created_minus_removed_lost": round(
                plant_created - plant_removed - plant_lost,
                4,
            ),
            "energy_available_at_end": round(plant_available, 4),
        },
        "energy_spend": {
            **{
                channel: round(value, 4)
                for channel, value in energy_spend.items()
            },
            "total": round(total_energy_spent, 4),
        },
    }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def prune_fresh_kill_deposits(tile: Any) -> None:
    tile.fresh_kill_deposits = [
        deposit
        for deposit in tile.fresh_kill_deposits
        if deposit.energy_remaining > 1e-9
    ]


def prune_carcass_deposits(tile: Any) -> None:
    tile.carcass_deposits = [
        deposit
        for deposit in tile.carcass_deposits
        if deposit.energy_remaining > 1e-9
    ]


def merge_fresh_kill_deposit_group(
    deposits: list[FreshKillDeposit],
) -> FreshKillDeposit:
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


def carcass_freshness_bucket(
    freshness: float,
    *,
    freshness_merge_bucket: float,
) -> int:
    bucket_size = max(freshness_merge_bucket, 1e-6)
    return int(clamp01(freshness) / bucket_size)


def merge_carcass_deposit_group(
    deposits: list[CarcassDeposit],
) -> CarcassDeposit:
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


def compact_carcass_deposits(
    tile: Any,
    *,
    max_tile_deposits: int,
    freshness_merge_bucket: float,
) -> None:
    prune_carcass_deposits(tile)
    limit = max(1, max_tile_deposits)
    if len(tile.carcass_deposits) <= limit:
        return

    grouped: dict[
        tuple[int | None, int, str, int | None],
        list[CarcassDeposit],
    ] = defaultdict(list)
    for deposit in tile.carcass_deposits:
        key = (
            deposit.source_species,
            carcass_freshness_bucket(
                deposit.freshness,
                freshness_merge_bucket=freshness_merge_bucket,
            ),
            deposit.cause,
            deposit.killer_id,
        )
        grouped[key].append(deposit)

    compacted = [
        merge_carcass_deposit_group(group)
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
        merged = merge_carcass_deposit_group(
            compacted[merge_index : merge_index + 2]
        )
        compacted[merge_index : merge_index + 2] = [merged]

    tile.carcass_deposits = compacted


def compact_fresh_kill_deposits(
    tile: Any,
    *,
    max_tile_deposits: int,
) -> None:
    prune_fresh_kill_deposits(tile)
    limit = max(1, max_tile_deposits)
    if len(tile.fresh_kill_deposits) <= limit:
        return

    grouped: dict[tuple[int | None, int | None], list[FreshKillDeposit]] = (
        defaultdict(list)
    )
    for deposit in tile.fresh_kill_deposits:
        grouped[(deposit.source_species, deposit.killer_id)].append(deposit)

    compacted = [
        merge_fresh_kill_deposit_group(group)
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
        compacted = compacted[: limit - 1] + [
            merge_fresh_kill_deposit_group(overflow)
        ]
    tile.fresh_kill_deposits = compacted


def carcass_source_breakdown(
    deposits: list[CarcassDeposit],
) -> list[dict[str, object]]:
    source_energy: dict[tuple[int | None, int | None, int | None], float] = (
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


def fresh_kill_source_breakdown(
    deposits: list[FreshKillDeposit],
) -> list[dict[str, object]]:
    source_energy: dict[
        tuple[int | None, int | None, int | None, int | None],
        float,
    ] = defaultdict(float)
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


def resolved_source_species(
    source_breakdown: list[dict[str, object]],
) -> int | None:
    if not source_breakdown:
        return None
    source_species = {entry.get("source_species") for entry in source_breakdown}
    if None in source_species:
        return None
    if len(source_species) == 1:
        source = next(iter(source_species))
        return source if isinstance(source, int) else None
    return None


def carcass_tile_state(world: Any, tile: Any) -> dict[str, object]:
    total_energy = tile.carcass_energy
    source_breakdown = carcass_source_breakdown(tile.carcass_deposits)
    dominant_source_species = resolved_source_species(source_breakdown)
    return {
        "deposit_count": len(tile.carcass_deposits),
        "total_energy": round(total_energy, 4),
        "avg_freshness": round(tile.carcass_decay, 4),
        "dominant_source_species": dominant_source_species,
        "mixed_sources": len(source_breakdown) > 1,
        "source_breakdown": source_breakdown,
    }


def fresh_kill_tile_state(world: Any, tile: Any) -> dict[str, object]:
    total_energy = tile.fresh_kill_energy
    source_breakdown = fresh_kill_source_breakdown(tile.fresh_kill_deposits)
    dominant_source_species = resolved_source_species(source_breakdown)
    return {
        "deposit_count": len(tile.fresh_kill_deposits),
        "total_energy": round(total_energy, 4),
        "dominant_source_species": dominant_source_species,
        "mixed_sources": len(source_breakdown) > 1,
        "source_breakdown": source_breakdown,
    }


def carcass_tile_summary_for_position(
    world: Any,
    x: int,
    y: int,
) -> dict[str, object]:
    summary = carcass_tile_state(world, world.grid[y][x])
    return {
        "x": x,
        "y": y,
        **summary,
    }


def fresh_kill_tile_summary_for_position(
    world: Any,
    x: int,
    y: int,
) -> dict[str, object]:
    summary = fresh_kill_tile_state(world, world.grid[y][x])
    return {
        "x": x,
        "y": y,
        **summary,
    }


def carcass_patch_summaries(world: Any) -> list[dict[str, object]]:
    patches: list[dict[str, object]] = []
    for y, row in enumerate(world.grid):
        for x, tile in enumerate(row):
            if tile.terrain == "water" or tile.carcass_energy <= 0:
                continue
            patches.append(carcass_tile_summary_for_position(world, x, y))
    return patches


def fresh_kill_patch_summaries(world: Any) -> list[dict[str, object]]:
    patches: list[dict[str, object]] = []
    for y, row in enumerate(world.grid):
        for x, tile in enumerate(row):
            if tile.terrain == "water" or tile.fresh_kill_energy <= 0:
                continue
            patches.append(fresh_kill_tile_summary_for_position(world, x, y))
    return patches


def carcass_snapshot(
    world: Any,
) -> tuple[list[list[int]], list[list[int]], dict[str, float]]:
    energy_codes: list[list[int]] = []
    freshness_codes: list[list[int]] = []
    count = 0
    total_energy = 0.0
    total_freshness_energy = 0.0
    total_deposits = 0
    mixed_source_tiles = 0
    for row in world.grid:
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
            if len(carcass_source_breakdown(tile.carcass_deposits)) > 1:
                mixed_source_tiles += 1
            energy_row.append(round(clamp01(tile.carcass_energy) * 100))
            freshness_row.append(round(clamp01(tile.carcass_decay) * 100))
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


def fresh_kill_snapshot(world: Any) -> tuple[list[list[int]], dict[str, float]]:
    energy_codes: list[list[int]] = []
    count = 0
    total_energy = 0.0
    total_deposits = 0
    mixed_source_tiles = 0
    for row in world.grid:
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
            if len(fresh_kill_source_breakdown(tile.fresh_kill_deposits)) > 1:
                mixed_source_tiles += 1
            energy_row.append(round(clamp01(tile.fresh_kill_energy) * 100))
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


def emit_death_resources(
    world: Any,
    agent: Any,
    *,
    resource_context: ResourceContext,
    death_cause: str,
    killer_id: int | None,
    source_species: int | None,
    projected_carcass_energy: float,
) -> DeathResourceEmission:
    tile = world.grid[agent.y][agent.x]
    carcass_tile_state = carcass_tile_summary_for_position(world, agent.x, agent.y)
    fresh_kill_tile_state = fresh_kill_tile_summary_for_position(world, agent.x, agent.y)
    carcass_energy = 0.0
    deposited_resource: str | None = None

    if tile.terrain != "water":
        carcass_energy = projected_carcass_energy
        if death_cause == "attack" and killer_id is not None:
            fresh_kill_tile_state = deposit_fresh_kill(
                world,
                tile,
                x=agent.x,
                y=agent.y,
                energy=carcass_energy,
                source_species=source_species,
                source_agent_id=agent.agent_id,
                killer_id=killer_id,
            )
            deposited_resource = "fresh_kill"
        else:
            carcass_tile_state = deposit_carcass(
                world,
                tile,
                resource_context=resource_context,
                x=agent.x,
                y=agent.y,
                energy=carcass_energy,
                source_species=source_species,
                source_agent_id=agent.agent_id,
                cause=death_cause,
                killer_id=killer_id,
            )
            deposited_resource = "carcass"

    return DeathResourceEmission(
        carcass_energy=carcass_energy,
        carcass_tile_state=carcass_tile_state,
        fresh_kill_tile_state=fresh_kill_tile_state,
        deposited_resource=deposited_resource,
    )


def death_resource_event_fields(
    emission: DeathResourceEmission,
) -> dict[str, object]:
    carcass_state = emission.carcass_tile_state
    fresh_kill_state = emission.fresh_kill_tile_state
    return {
        "carcass_energy": round(emission.carcass_energy, 4),
        "fresh_kill_energy": round(float(fresh_kill_state["total_energy"]), 4),
        "tile_carcass_energy_after": carcass_state["total_energy"],
        "tile_avg_freshness_after": carcass_state["avg_freshness"],
        "tile_deposit_count_after": carcass_state["deposit_count"],
        "tile_mixed_sources_after": carcass_state["mixed_sources"],
        "tile_dominant_source_species_after": carcass_state[
            "dominant_source_species"
        ],
        "tile_source_breakdown_after": carcass_state["source_breakdown"],
        "tile_fresh_kill_energy_after": fresh_kill_state["total_energy"],
        "tile_fresh_kill_deposit_count_after": fresh_kill_state["deposit_count"],
        "tile_fresh_kill_mixed_sources_after": fresh_kill_state["mixed_sources"],
        "tile_fresh_kill_dominant_source_species_after": fresh_kill_state[
            "dominant_source_species"
        ],
        "tile_fresh_kill_source_breakdown_after": fresh_kill_state[
            "source_breakdown"
        ],
    }


def deposit_fresh_kill(
    world: Any,
    tile: Any,
    *,
    x: int,
    y: int,
    energy: float,
    source_species: int | None,
    source_agent_id: int | None,
    killer_id: int | None,
) -> dict[str, object]:
    if energy <= 0:
        return fresh_kill_tile_summary_for_position(world, x, y)
    tile.fresh_kill_deposits.append(
        FreshKillDeposit(
            energy_remaining=energy,
            source_species=source_species,
            source_agent_id=source_agent_id,
            death_tick=world.tick,
            killer_id=killer_id,
        )
    )
    compact_fresh_kill_deposits(
        tile,
        max_tile_deposits=world.config.carcasses.max_tile_deposits,
    )
    world.run_fresh_kill_totals["deposition_events"] += 1
    world.run_fresh_kill_totals["energy_deposited"] += energy
    world.tick_fresh_kill_deposited_energy += energy
    patch_state = fresh_kill_tile_summary_for_position(world, x, y)
    if world.record_tick_details:
        source_breakdown = fresh_kill_source_breakdown(
            [
                FreshKillDeposit(
                    energy_remaining=energy,
                    source_species=source_species,
                    source_agent_id=source_agent_id,
                    death_tick=world.tick,
                    killer_id=killer_id,
                )
            ]
        )
        world.tick_fresh_kill_deposit_events.append(
            {
                "source_agent_id": source_agent_id,
                "source_species": resolved_source_species(source_breakdown),
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


def convert_fresh_kill_to_carcass(
    world: Any,
    tile: Any,
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
        amount = min(
            deposit.energy_remaining,
            deposit.energy_remaining * conversion_rate,
        )
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
    compact_fresh_kill_deposits(
        tile,
        max_tile_deposits=world.config.carcasses.max_tile_deposits,
    )
    if not converted_deposits:
        return 0.0
    tile.carcass_deposits.extend(converted_deposits)
    compact_carcass_deposits(
        tile,
        max_tile_deposits=world.config.carcasses.max_tile_deposits,
        freshness_merge_bucket=world.config.carcasses.freshness_merge_bucket,
    )
    world.run_fresh_kill_totals["energy_converted_to_carcass"] += converted_energy
    world.tick_fresh_kill_to_carcass_energy += converted_energy
    world.run_carcass_totals["deposition_events"] += len(converted_deposits)
    world.run_carcass_totals["energy_deposited"] += converted_energy
    world.tick_carcass_deposited_energy += converted_energy
    if world.record_tick_details:
        patch_state = carcass_tile_summary_for_position(world, x, y)
        source_breakdown = carcass_source_breakdown(converted_deposits)
        dominant_source_species = resolved_source_species(source_breakdown)
        world.tick_carcass_deposit_events.append(
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


def deposit_carcass(
    world: Any,
    tile: Any,
    *,
    resource_context: ResourceContext,
    x: int,
    y: int,
    energy: float,
    source_species: int | None,
    source_agent_id: int | None,
    cause: str,
    killer_id: int | None,
) -> dict[str, object]:
    if energy <= 0:
        return carcass_tile_summary_for_position(world, x, y)
    tile.carcass_deposits.append(
        CarcassDeposit(
            energy_remaining=energy,
            freshness=1.0,
            source_species=source_species,
            source_agent_id=source_agent_id,
            death_tick=world.tick,
            cause=cause,
            killer_id=killer_id,
        )
    )
    compact_carcass_deposits(
        tile,
        max_tile_deposits=world.config.carcasses.max_tile_deposits,
        freshness_merge_bucket=world.config.carcasses.freshness_merge_bucket,
    )
    world.run_carcass_totals["deposition_events"] += 1
    world.run_carcass_totals["energy_deposited"] += energy
    world.tick_carcass_deposited_energy += energy
    patch_state = carcass_tile_summary_for_position(world, x, y)
    if world.record_tick_details or world.record_events:
        source_breakdown = carcass_source_breakdown(
            [
                CarcassDeposit(
                    energy_remaining=energy,
                    freshness=1.0,
                    source_species=source_species,
                    source_agent_id=source_agent_id,
                    death_tick=world.tick,
                    cause=cause,
                    killer_id=killer_id,
                )
            ]
        )
        resolved_species = resolved_source_species(source_breakdown)
    else:
        source_breakdown = []
        resolved_species = None
    if world.record_tick_details:
        world.tick_carcass_deposit_events.append(
            {
                "source_agent_id": source_agent_id,
                "source_species": resolved_species,
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
    if world.record_events:
        resource_context.emit(
            EventType.CARCASS_DEPOSITED,
            agent_id=source_agent_id,
            data={
                "source_agent_id": source_agent_id,
                "source_species": resolved_species,
                "deposited_energy": round(energy, 4),
                "cause": cause,
                "killer_id": killer_id,
                "x": x,
                "y": y,
                "tile_carcass_energy_after": patch_state["total_energy"],
                "tile_avg_freshness_after": patch_state["avg_freshness"],
                "tile_deposit_count_after": patch_state["deposit_count"],
                "tile_mixed_sources_after": patch_state["mixed_sources"],
                "tile_dominant_source_species_after": (
                    patch_state["dominant_source_species"]
                ),
                "tile_source_breakdown_after": patch_state["source_breakdown"],
            },
        )
    return patch_state


def decay_carcass_tile(
    tile: Any,
    *,
    decay: float,
    max_tile_deposits: int,
    freshness_merge_bucket: float,
) -> float:
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
    compact_carcass_deposits(
        tile,
        max_tile_deposits=max_tile_deposits,
        freshness_merge_bucket=freshness_merge_bucket,
    )
    return energy_decayed


def consume_carcass_from_tile(
    tile: Any,
    requested_amount: float,
    *,
    max_tile_deposits: int,
    freshness_merge_bucket: float,
) -> dict[str, object]:
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
        key=lambda deposit: (
            -deposit.freshness,
            -deposit.death_tick,
            -(deposit.source_agent_id or 0),
        ),
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
    compact_carcass_deposits(
        tile,
        max_tile_deposits=max_tile_deposits,
        freshness_merge_bucket=freshness_merge_bucket,
    )
    return {
        "consumed": consumed,
        "avg_freshness": freshness_weighted / consumed if consumed > 0 else 0.0,
        "deposit_breakdown": deposit_breakdown,
        "source_breakdown": carcass_source_breakdown(
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


def consume_fresh_kill_from_tile(
    tile: Any,
    requested_amount: float,
    *,
    max_tile_deposits: int,
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
    compact_fresh_kill_deposits(
        tile,
        max_tile_deposits=max_tile_deposits,
    )
    return {
        "consumed": consumed,
        "deposit_breakdown": deposit_breakdown,
        "source_breakdown": fresh_kill_source_breakdown(
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


def vegetation_target(
    world: Any,
    x: int,
    y: int,
    season: str,
    *,
    resource_context: ResourceContext,
) -> float:
    tile = world.grid[y][x]
    if tile.terrain == "water":
        return 1.0
    fertility, moisture, heat = resource_context.effective_tile_fields(x, y)
    habitat_state = resource_context.habitat_state_at(x, y)
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
    return resource_context.clamp01(target)


def shelter_target(
    world: Any,
    x: int,
    y: int,
    season: str,
    *,
    resource_context: ResourceContext,
) -> float:
    tile = world.grid[y][x]
    if tile.terrain == "water":
        return 0.0
    fertility, moisture, heat = resource_context.effective_tile_fields(x, y)
    forest_density = resource_context.terrain_neighbor_ratio(
        x,
        y,
        terrain_filter={"forest"},
        radius=1,
    )
    habitat_state = resource_context.habitat_state_at(x, y)
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
    return resource_context.clamp01(target)


def food_capacity(
    world: Any,
    x: int,
    y: int,
    season: str,
    *,
    resource_context: ResourceContext,
) -> float:
    tile = world.grid[y][x]
    if tile.terrain == "water":
        return 0.0
    fertility, moisture, heat = resource_context.effective_tile_fields(x, y)
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
    return resource_context.clamp01(capacity)


def field_growth_multiplier(
    world: Any,
    x: int,
    y: int,
    season: str,
    *,
    resource_context: ResourceContext,
) -> float:
    tile = world.grid[y][x]
    fertility, moisture, heat = resource_context.effective_tile_fields(x, y)
    growth = 0.42 + fertility * 0.72 + moisture * 0.44
    heat_penalty = max(0.0, heat - moisture) * 0.34
    vegetation_bonus = tile.vegetation * 0.34
    recovery_penalty = tile.recovery_debt * 0.42
    return max(0.22, growth + vegetation_bonus - heat_penalty - recovery_penalty)


def regrow_resources(
    world: Any,
    *,
    resource_context: ResourceContext,
) -> None:
    season = resource_context.season_state()["name"]
    resource_context.habitat_state_grid()
    resources = world.config.resources
    for y, row in enumerate(world.grid):
        for x, tile in enumerate(row):
            if tile.terrain == "water":
                tile.water = world.config.resources.water_refresh_amount
                continue

            if tile.fresh_kill_energy > 0:
                convert_fresh_kill_to_carcass(
                    world,
                    tile,
                    x=x,
                    y=y,
                    conversion_rate=world.config.carcasses.fresh_kill_conversion_rate,
                )

            if tile.carcass_energy > 0:
                _, moisture, heat = resource_context.effective_tile_fields(x, y)
                decay = (
                    world.config.carcasses.decay_base_rate
                    + heat * world.config.carcasses.decay_heat_factor
                    + moisture * world.config.carcasses.decay_moisture_factor
                )
                energy_decayed = decay_carcass_tile(
                    tile,
                    decay=decay,
                    max_tile_deposits=world.config.carcasses.max_tile_deposits,
                    freshness_merge_bucket=world.config.carcasses.freshness_merge_bucket,
                )
                world.tick_carcass_energy_decayed += energy_decayed
                world.run_carcass_totals["energy_decayed"] += energy_decayed

            fertility, moisture, heat = resource_context.effective_tile_fields(x, y)
            habitat_state = resource_context.habitat_state_at(x, y)
            field_growth = field_growth_multiplier(
                world,
                x,
                y,
                season,
                resource_context=resource_context,
            )
            vegetation_goal = vegetation_target(
                world,
                x,
                y,
                season,
                resource_context=resource_context,
            )
            shelter_goal = shelter_target(
                world,
                x,
                y,
                season,
                resource_context=resource_context,
            )
            forest_density = resource_context.terrain_neighbor_ratio(
                x,
                y,
                terrain_filter={"forest"},
                radius=1,
            )
            recovery_support = max(0.28, 1.0 - tile.recovery_debt * 0.72)
            vegetation_growth = (
                resources.vegetation_regrowth_rate
                * terrain_growth_modifier(world, tile.terrain, season)
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

            if tile.vegetation <= vegetation_goal:
                tile.vegetation = resource_context.clamp01(
                    tile.vegetation
                    + (vegetation_goal - tile.vegetation) * vegetation_growth
                    - vegetation_stress * 0.18
                )
            else:
                tile.vegetation = resource_context.clamp01(
                    tile.vegetation - (tile.vegetation - vegetation_goal) * (0.18 + vegetation_stress)
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
            tile.recovery_debt = resource_context.clamp01(
                tile.recovery_debt + degradation - recovery
            )

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

            if tile.shelter <= shelter_goal:
                tile.shelter = resource_context.clamp01(
                    tile.shelter
                    + (shelter_goal - tile.shelter) * shelter_growth
                    - shelter_stress * 0.1
                )
            else:
                tile.shelter = resource_context.clamp01(
                    tile.shelter - (tile.shelter - shelter_goal) * (0.14 + shelter_stress)
                )

            if habitat_state == "parched":
                food_before_loss = tile.food
                tile.food = max(
                    0.0,
                    tile.food - (0.008 if tile.terrain == "plain" else 0.0045),
                )
                record_plant_lost(world, food_before_loss - tile.food)
            elif habitat_state == "flooded" and tile.terrain == "plain":
                food_before_loss = tile.food
                tile.food = max(0.0, tile.food - 0.003)
                record_plant_lost(world, food_before_loss - tile.food)

            food_regrowth = (
                terrain_regrowth_rate(world, tile.terrain)
                * terrain_growth_modifier(world, tile.terrain, season)
                * field_growth
                * (0.4 + tile.vegetation * 0.84)
                * max(0.24, 1.0 - tile.recovery_debt * 0.72)
                * habitat_regrowth_modifier(
                    world,
                    x,
                    y,
                    resource_context=resource_context,
                )
            )
            food_before_regrowth = tile.food
            tile.food = min(1.0, tile.food + food_regrowth)
            tile.food = min(
                tile.food,
                max(
                    0.04,
                    food_capacity(
                        world,
                        x,
                        y,
                        season,
                        resource_context=resource_context,
                    ),
                ),
            )
            food_delta = tile.food - food_before_regrowth
            if food_delta > 0:
                record_plant_created(world, food_delta)
            elif food_delta < 0:
                record_plant_lost(world, -food_delta)


def habitat_regrowth_modifier(
    world: Any,
    x: int,
    y: int,
    *,
    resource_context: ResourceContext,
) -> float:
    habitat_state = resource_context.habitat_state_at(x, y)
    terrain = world.grid[y][x].terrain
    if habitat_state == "bloom":
        return 1.028
    if habitat_state == "flooded":
        return 1.012 if terrain == "wetland" else 0.986
    if habitat_state == "parched":
        return 0.972 if terrain == "rocky" else 0.94
    return 1.0


def terrain_regrowth_rate(world: Any, terrain: str) -> float:
    resources = world.config.resources
    if terrain == "forest":
        return resources.forest_food_rate
    if terrain == "wetland":
        return resources.wetland_food_rate
    if terrain == "rocky":
        return resources.rocky_food_rate
    return resources.plain_food_rate


def terrain_growth_modifier(world: Any, terrain: str, season: str) -> float:
    climate = world.config.climate
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
