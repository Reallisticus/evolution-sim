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


def summary_end_surface_state(
    mode: RunMode,
    *,
    context: SurfaceSnapshotContext,
) -> dict[str, object]:
    snapshot_context = context
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
