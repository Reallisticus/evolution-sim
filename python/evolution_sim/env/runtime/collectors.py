from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from evolution_sim.env.contracts import SHARED_SUMMARY_FIELDS, filter_summary_fields
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.taxonomy import apply_replay_taxonomy


@dataclass(frozen=True, slots=True)
class CollectorContext:
    config: Any
    events: Sequence[Any]
    capture_frame: Callable[..., None]
    build_summary: Callable[..., dict[str, object]]
    build_viewer_payload: Callable[[], dict[str, object]]
    refresh_population_snapshots: Callable[..., object]


@dataclass(slots=True)
class FullReplayCollector:
    mode: RunMode = RunMode.FULL_REPLAY

    def on_tick(
        self,
        context: CollectorContext,
        *,
        births_this_tick: int,
        deaths_this_tick: int,
    ) -> None:
        context.capture_frame(
            births_this_tick=births_this_tick,
            deaths_this_tick=deaths_this_tick,
        )

    def finalize(
        self,
        context: CollectorContext,
    ) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        summary = context.build_summary(mode=self.mode)
        viewer = context.build_viewer_payload()
        events = [event.to_dict() for event in context.events]
        summary, viewer = apply_replay_taxonomy(
            config=context.config,
            summary=summary,
            events=events,
            viewer=viewer,
        )
        return summary, events, viewer


@dataclass(slots=True)
class SummaryCollector:
    mode: RunMode = RunMode.SUMMARY_ONLY

    def on_tick(
        self,
        context: CollectorContext,
        *,
        births_this_tick: int,
        deaths_this_tick: int,
    ) -> None:
        context.refresh_population_snapshots(include_species=False)

    def finalize(
        self,
        context: CollectorContext,
    ) -> tuple[dict[str, object], None, None]:
        context.refresh_population_snapshots(include_species=False)
        summary = context.build_summary(mode=self.mode)
        return filter_summary_fields(summary, SHARED_SUMMARY_FIELDS), None, None


def collector_for_mode(mode: RunMode) -> FullReplayCollector | SummaryCollector:
    if mode == RunMode.SUMMARY_ONLY:
        return SummaryCollector()
    return FullReplayCollector()
