from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evolution_sim.env.contracts import SHARED_SUMMARY_FIELDS, filter_summary_fields
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.taxonomy import apply_replay_taxonomy


@dataclass(slots=True)
class FullReplayCollector:
    mode: RunMode = RunMode.FULL_REPLAY

    def on_tick(self, world: Any, *, births_this_tick: int, deaths_this_tick: int) -> None:
        world._capture_frame(births_this_tick=births_this_tick, deaths_this_tick=deaths_this_tick)

    def finalize(
        self,
        world: Any,
    ) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        summary = world._build_summary(mode=self.mode)
        viewer = world._build_viewer_payload()
        events = [event.to_dict() for event in world.events]
        summary, viewer = apply_replay_taxonomy(
            config=world.config,
            summary=summary,
            events=events,
            viewer=viewer,
        )
        return summary, events, viewer


@dataclass(slots=True)
class SummaryCollector:
    mode: RunMode = RunMode.SUMMARY_ONLY

    def on_tick(self, world: Any, *, births_this_tick: int, deaths_this_tick: int) -> None:
        return None

    def finalize(
        self,
        world: Any,
    ) -> tuple[dict[str, object], None, None]:
        world._refresh_population_snapshots(include_species=False)
        summary = world._build_summary(mode=self.mode)
        return filter_summary_fields(summary, SHARED_SUMMARY_FIELDS), None, None


def collector_for_mode(mode: RunMode) -> FullReplayCollector | SummaryCollector:
    if mode == RunMode.SUMMARY_ONLY:
        return SummaryCollector()
    return FullReplayCollector()
