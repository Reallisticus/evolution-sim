from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evolution_sim.env.runtime.state import RunMode


@dataclass(frozen=True, slots=True)
class SurfaceSnapshotContext:
    """Explicit authority for summary end-surface snapshots."""

    viewer_frames: list[dict[str, Any]]
    hydrology_primary_counts: dict[str, int]
    hydrology_support_counts: dict[str, int]
    hydrology_primary_stats: dict[str, float]
    refuge_counts: dict[str, int]
    refuge_stats: dict[str, float]
    hazard_counts: dict[str, int]
    hazard_stats: dict[str, float]
    fresh_kill_stats: dict[str, float]
    carcass_stats: dict[str, float]
    biotic_field_stats: dict[str, float]
    signal_field_stats: dict[str, float]
    ecology_counts: dict[str, int]
    ecology_stats: dict[str, float]
    habitat_counts: dict[str, int]


def build_surface_snapshot_context(world: Any) -> SurfaceSnapshotContext:
    (
        _,
        _,
        hydrology_primary_counts,
        hydrology_support_counts,
        hydrology_primary_stats,
    ) = world._hydrology_snapshot()
    _, _, refuge_counts, refuge_stats = world._refuge_snapshot()
    _, _, hazard_counts, hazard_stats = world._hazard_snapshot()
    _, fresh_kill_stats = world._fresh_kill_snapshot()
    _, _, carcass_stats = world._carcass_snapshot()
    _, biotic_field_stats = world._biotic_field_snapshot()
    _, signal_field_stats = world._signal_field_snapshot()
    _, ecology_counts, ecology_stats = world._ecology_snapshot()
    habitat_counts = world._habitat_state_grid()[1]
    return SurfaceSnapshotContext(
        viewer_frames=world.viewer_frames,
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


def _resolve_surface_snapshot_context(
    world: Any | None,
    context: SurfaceSnapshotContext | None = None,
) -> SurfaceSnapshotContext:
    if context is not None:
        return context
    if world is None:
        raise ValueError("world is required when surface snapshot context is omitted")
    return build_surface_snapshot_context(world)


def summary_end_surface_state(
    world: Any | None,
    mode: RunMode,
    *,
    context: SurfaceSnapshotContext | None = None,
) -> dict[str, object]:
    snapshot_context = _resolve_surface_snapshot_context(world, context)
    latest_frame = (
        snapshot_context.viewer_frames[-1]
        if snapshot_context.viewer_frames and mode == RunMode.FULL_REPLAY
        else None
    )
    if latest_frame is not None:
        return {
            "hydrology_primary_counts": latest_frame["hydrology_primary_counts"],
            "hydrology_support_counts": latest_frame["hydrology_support_counts"],
            "hydrology_primary_stats": latest_frame["hydrology_primary_stats"],
            "refuge_counts": latest_frame["refuge_counts"],
            "refuge_stats": latest_frame["refuge_stats"],
            "hazard_counts": latest_frame["hazard_counts"],
            "hazard_stats": latest_frame["hazard_stats"],
            "fresh_kill_stats": latest_frame["fresh_kill_stats"],
            "carcass_stats": latest_frame["carcass_stats"],
            "biotic_field_stats": latest_frame["biotic_field_stats"],
            "signal_field_stats": latest_frame["signal_field_stats"],
            "ecology_counts": latest_frame["ecology_state_counts"],
            "ecology_stats": latest_frame["ecology_stats"],
            "habitat_counts": latest_frame["habitat_state_counts"],
            "latest_species_metrics": latest_frame["species_metrics"],
        }
    return {
        "hydrology_primary_counts": snapshot_context.hydrology_primary_counts,
        "hydrology_support_counts": snapshot_context.hydrology_support_counts,
        "hydrology_primary_stats": snapshot_context.hydrology_primary_stats,
        "refuge_counts": snapshot_context.refuge_counts,
        "refuge_stats": snapshot_context.refuge_stats,
        "hazard_counts": snapshot_context.hazard_counts,
        "hazard_stats": snapshot_context.hazard_stats,
        "fresh_kill_stats": snapshot_context.fresh_kill_stats,
        "carcass_stats": snapshot_context.carcass_stats,
        "biotic_field_stats": snapshot_context.biotic_field_stats,
        "signal_field_stats": snapshot_context.signal_field_stats,
        "ecology_counts": snapshot_context.ecology_counts,
        "ecology_stats": snapshot_context.ecology_stats,
        "habitat_counts": snapshot_context.habitat_counts,
        "latest_species_metrics": {},
    }
