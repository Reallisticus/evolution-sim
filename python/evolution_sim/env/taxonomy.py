"""Replay-time taxonomy pass for durable species identity.

This pass treats replay data and ancestry as the source of truth. It never
recomputes durable species from the current alive set alone. Instead it:

- starts from founder lineages as historical species roots
- searches replay ancestry for descendant branches that satisfy sustained split evidence
- records one-way speciation events with explicit evidence channels
- rewrites per-frame ``species_id`` using only the resulting event tree
- leaves transient frame-local clustering in the separate ``ecotype`` layer

The result is event-sourced species identity: agents change species only on
logged replay-time split events, and never because a frame-local centroid drifted.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from math import log2
from typing import Any

from evolution_sim.config import WorldConfig
from evolution_sim.env.runtime.reporting import (
    build_species_metrics as build_shared_species_metrics,
    rebuild_taxonomy_summary_and_analytics,
    SpeciesMetricSample,
)
from evolution_sim.genome.species import GENE_ORDER, euclidean_distance, genome_vector_from_values

LAND_TERRAINS = ("plain", "forest", "wetland", "rocky")
HYDROLOGY_REASONS = ("none", "adjacent_water", "wetland", "flooded")
HABITAT_STATES = ("stable", "bloom", "flooded", "parched")
ECOLOGY_STATES = ("stable", "lush", "recovering", "depleted")
HAZARD_TYPES = ("none", "exposure", "instability")
TROPHIC_ROLES = ("herbivore", "omnivore", "carnivore")
MEAT_MODES = ("scavenger", "hunter", "mixed")
REPLAY_TAXONOMY_MODE = "replay_clade_species_v2"
ECOTYPE_IDENTITY_MODE = "frame_local_genome_cluster"
HYDROLOGY_SUPPORT_FLAGS = {"adjacent_to_water": 1, "wetland": 2, "flooded": 4}
HABITAT_CODE_LEGEND = {0: "stable", 1: "bloom", 2: "flooded", 3: "parched"}
REQUIRED_AGENT_FIELDS = {
    "agent_id",
    "x",
    "y",
    "energy_ratio",
    "hydration_ratio",
    "health_ratio",
    "injury_load",
    "age",
    "tile_vegetation",
    "tile_recovery_debt",
    "reproduction_ready",
    "trophic_role",
    "meat_mode",
    "water_access_reason",
    "soft_refuge_reason",
    "hydrology_support_code",
    "refuge_score",
    "matched_diet_ratio",
    "ecotype_id",
    "species_id",
}


@dataclass(slots=True)
class ReplayAgentRecord:
    agent_id: int
    parent_id: int | None
    lineage_id: int
    birth_tick: int
    death_tick: int | None
    genome_values: dict[str, float]
    genome_vector: tuple[float, ...]

    def last_alive_tick(self, final_tick: int) -> int:
        if self.death_tick is None:
            return final_tick
        return self.death_tick - 1


@dataclass(slots=True)
class CandidateSplit:
    parent_agent_id: int
    child_agent_id: int
    split_tick: int
    branch_member_count: int
    branch_peak_members: int
    branch_persistence_ticks: int
    overlap_ticks: int
    genetic_distance: float
    ecotype_divergence: float
    ecological_divergence: float
    balance_ratio: float
    score: float
    daughter_members: frozenset[int]
    continuation_members: frozenset[int]


@dataclass(slots=True)
class SpeciesSegment:
    species_id: int
    label: str
    lineage_id: int
    parent_species_id: int | None
    source_species_id: int | None
    start_tick: int
    founder_agent_id: int
    origin_kind: str
    members: frozenset[int]
    evidence: dict[str, float]
    split_parent_agent_id: int | None = None
    split_child_agent_id: int | None = None
    end_tick: int | None = None
    child_species_ids: list[int] = field(default_factory=list)


class ReplayTaxonomyPass:
    def __init__(
        self,
        *,
        config: WorldConfig,
        summary: dict[str, object],
        events: list[dict[str, object]],
        viewer: dict[str, object],
    ) -> None:
        self.config = config
        self.thresholds = config.taxonomy
        self.summary = summary
        self.events = events
        self.viewer = viewer
        self.frames: list[dict[str, object]] = list(viewer.get("frames", []))
        self.field_map = {
            field: index for index, field in enumerate(viewer.get("agent_encoding", []))
        }
        missing_fields = sorted(REQUIRED_AGENT_FIELDS - set(self.field_map))
        if missing_fields:
            raise ValueError(
                "Replay taxonomy requires agent encoding fields: "
                + ", ".join(missing_fields)
            )
        self.final_tick = self.frames[-1]["tick"] if self.frames else -1
        self.terrain_codes = viewer["map"]["terrain_codes"]
        self.terrain_legend = {int(code): name for code, name in viewer["map"]["terrain_legend"].items()}
        self.habitat_legend = HABITAT_CODE_LEGEND.copy()
        self.ecology_legend = {
            int(code): state for code, state in viewer["map"]["ecology_legend"].items()
        }
        self.hazard_legend = {
            int(code): state for code, state in viewer["map"]["hazard_legend"].items()
        }
        self.agents = self._load_agents(viewer["agent_catalog"])
        self.children_by_parent: dict[int, list[int]] = defaultdict(list)
        self.lineage_roots: list[int] = []
        for record in self.agents.values():
            if record.parent_id is None:
                self.lineage_roots.append(record.agent_id)
            else:
                self.children_by_parent[record.parent_id].append(record.agent_id)
        self.lineage_roots.sort()
        self.tin: dict[int, int] = {}
        self.tout: dict[int, int] = {}
        self._dfs_index = 0
        for root_agent_id in self.lineage_roots:
            self._assign_dfs(root_agent_id)
        self.frame_ticks = [int(frame["tick"]) for frame in self.frames]
        self.species_status_counts: dict[str, int] = {}
        self.frame_rows: list[dict[int, list[object]]] = []
        self.frame_alive_ids: list[list[int]] = []
        for frame in self.frames:
            rows_by_agent: dict[int, list[object]] = {}
            alive_ids: list[int] = []
            for row in frame["agents"]:
                agent_id = int(row[self.field_map["agent_id"]])
                rows_by_agent[agent_id] = row
                alive_ids.append(agent_id)
            self.frame_rows.append(rows_by_agent)
            self.frame_alive_ids.append(alive_ids)
        self.birth_events_by_tick: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.death_events_by_tick: dict[int, list[dict[str, object]]] = defaultdict(list)
        self.attack_events_by_tick: dict[int, list[dict[str, object]]] = defaultdict(list)
        self.damage_events_by_tick: dict[int, list[dict[str, object]]] = defaultdict(list)
        self.feed_events_by_tick: dict[int, list[dict[str, object]]] = defaultdict(list)
        self._index_events()
        self.segments: dict[int, SpeciesSegment] = {}
        self.speciation_events: list[dict[str, object]] = []
        self.next_species_id = max((record.lineage_id for record in self.agents.values()), default=0) + 1

    def apply(self) -> tuple[dict[str, object], dict[str, object]]:
        if not self.frames or not self.agents:
            self.viewer["species_catalog"] = {}
            self.viewer["taxonomy"] = self._build_taxonomy_payload(events=[])
            self.summary["taxonomy_mode"] = REPLAY_TAXONOMY_MODE
            self.summary["species_created"] = 0
            self.summary["alive_species_count"] = 0
            self.summary["alive_species"] = []
            self.summary["top_species"] = []
            self.summary["speciation_events"] = 0
            self.summary["species_status_counts"] = {}
            analytics = self.viewer.setdefault("analytics", {})
            analytics["species_population"] = {}
            analytics["collapse_events"] = []
            analytics["speciation_events"] = []
            return self.summary, self.viewer

        for root_agent_id in self.lineage_roots:
            lineage_id = self.agents[root_agent_id].lineage_id
            lineage_members = frozenset(
                agent_id
                for agent_id, agent in self.agents.items()
                if agent.lineage_id == lineage_id
            )
            self.segments[lineage_id] = SpeciesSegment(
                species_id=lineage_id,
                label=f"S{lineage_id:03d}",
                lineage_id=lineage_id,
                parent_species_id=None,
                source_species_id=None,
                start_tick=self.agents[root_agent_id].birth_tick,
                founder_agent_id=root_agent_id,
                origin_kind="founder",
                members=lineage_members,
                evidence={},
            )

        for lineage_id in sorted(self.segments):
            self._split_segment(self.segments[lineage_id])

        self.speciation_events.sort(
            key=lambda event: (
                event["tick"],
                event["lineage_id"],
                event["source_species_id"],
                event["daughter_species_id"],
            )
        )
        self._rewrite_species_assignments()
        self._rebuild_summary_and_analytics()
        return self.summary, self.viewer

    def _load_agents(self, agent_catalog: dict[str, object]) -> dict[int, ReplayAgentRecord]:
        records: dict[int, ReplayAgentRecord] = {}
        for raw_agent_id, payload in agent_catalog.items():
            genome_values = payload["genome"]
            records[int(raw_agent_id)] = ReplayAgentRecord(
                agent_id=int(payload["agent_id"]),
                parent_id=payload["parent_id"],
                lineage_id=int(payload["lineage_id"]),
                birth_tick=int(payload.get("birth_tick", 0)),
                death_tick=payload.get("death_tick"),
                genome_values={gene: float(genome_values[gene]) for gene in GENE_ORDER},
                genome_vector=genome_vector_from_values(genome_values),
            )
        return records

    def _assign_dfs(self, agent_id: int) -> None:
        self.tin[agent_id] = self._dfs_index
        self._dfs_index += 1
        for child_id in sorted(self.children_by_parent.get(agent_id, [])):
            self._assign_dfs(child_id)
        self.tout[agent_id] = self._dfs_index

    def _index_events(self) -> None:
        for event in self.events:
            tick = int(event["tick"])
            event_type = str(event["type"])
            if event_type == "agent_reproduced":
                self.birth_events_by_tick[tick].append(
                    (int(event["agent_id"]), int(event["data"]["child_id"]))
                )
            elif event_type == "agent_died":
                self.death_events_by_tick[tick].append(event)
            elif event_type == "agent_attacked":
                self.attack_events_by_tick[tick].append(event)
            elif event_type == "agent_damaged":
                self.damage_events_by_tick[tick].append(event)
            elif event_type == "agent_ate":
                self.feed_events_by_tick[tick].append(event)

    def _is_descendant(self, ancestor_id: int, agent_id: int) -> bool:
        return self.tin[ancestor_id] <= self.tin[agent_id] < self.tout[ancestor_id]

    def _split_segment(self, segment: SpeciesSegment) -> None:
        candidate = self._best_split_for_segment(segment)
        if candidate is None:
            return

        split_tick = candidate.split_tick
        segment.end_tick = split_tick - 1
        continuation_species_id = self.next_species_id
        self.next_species_id += 1
        daughter_species_id = self.next_species_id
        self.next_species_id += 1

        continuation_segment = SpeciesSegment(
            species_id=continuation_species_id,
            label=f"S{continuation_species_id:03d}",
            lineage_id=segment.lineage_id,
            parent_species_id=segment.species_id,
            source_species_id=segment.species_id,
            start_tick=split_tick,
            founder_agent_id=candidate.parent_agent_id,
            origin_kind="continuation",
            members=candidate.continuation_members,
            evidence={
                "score": round(candidate.score, 4),
                "branch_peak_members": float(candidate.branch_peak_members),
                "branch_persistence_ticks": float(candidate.branch_persistence_ticks),
                "overlap_ticks": float(candidate.overlap_ticks),
                "genetic_distance": round(candidate.genetic_distance, 4),
                "ecotype_divergence": round(candidate.ecotype_divergence, 4),
                "ecological_divergence": round(candidate.ecological_divergence, 4),
                "balance_ratio": round(candidate.balance_ratio, 4),
            },
            split_parent_agent_id=candidate.parent_agent_id,
            split_child_agent_id=candidate.child_agent_id,
        )
        daughter_segment = SpeciesSegment(
            species_id=daughter_species_id,
            label=f"S{daughter_species_id:03d}",
            lineage_id=segment.lineage_id,
            parent_species_id=segment.species_id,
            source_species_id=segment.species_id,
            start_tick=split_tick,
            founder_agent_id=candidate.child_agent_id,
            origin_kind="daughter",
            members=candidate.daughter_members,
            evidence=continuation_segment.evidence.copy(),
            split_parent_agent_id=candidate.parent_agent_id,
            split_child_agent_id=candidate.child_agent_id,
        )
        segment.child_species_ids = [continuation_species_id, daughter_species_id]
        self.segments[continuation_species_id] = continuation_segment
        self.segments[daughter_species_id] = daughter_segment
        evidence = {
            "genealogical_separation": 1.0,
            "branch_member_count": float(candidate.branch_member_count),
            "branch_peak_members": float(candidate.branch_peak_members),
            "branch_persistence_ticks": float(candidate.branch_persistence_ticks),
            "overlap_ticks": float(candidate.overlap_ticks),
            "genetic_distance": round(candidate.genetic_distance, 4),
            "ecotype_divergence": round(candidate.ecotype_divergence, 4),
            "ecological_divergence": round(candidate.ecological_divergence, 4),
            "balance_ratio": round(candidate.balance_ratio, 4),
            "split_score": round(candidate.score, 4),
        }
        self.speciation_events.append(
            {
                "type": "speciation",
                "tick": split_tick,
                "origin_tick": split_tick,
                "lineage_id": segment.lineage_id,
                "source_species_id": segment.species_id,
                "continuation_species_id": continuation_species_id,
                "daughter_species_id": daughter_species_id,
                "parent_agent_id": candidate.parent_agent_id,
                "child_agent_id": candidate.child_agent_id,
                "branch_root_agent_id": candidate.child_agent_id,
                "score": round(candidate.score, 4),
                "branch_member_count": candidate.branch_member_count,
                "branch_peak_members": candidate.branch_peak_members,
                "branch_persistence_ticks": candidate.branch_persistence_ticks,
                "overlap_ticks": candidate.overlap_ticks,
                "genetic_distance": round(candidate.genetic_distance, 4),
                "ecotype_divergence": round(candidate.ecotype_divergence, 4),
                "ecological_divergence": round(candidate.ecological_divergence, 4),
                "balance_ratio": round(candidate.balance_ratio, 4),
                "evidence": evidence,
            }
        )
        self._split_segment(continuation_segment)
        self._split_segment(daughter_segment)

    def _best_split_for_segment(self, segment: SpeciesSegment) -> CandidateSplit | None:
        best: CandidateSplit | None = None
        member_list = tuple(segment.members)
        if len(member_list) < self.thresholds.min_branch_member_count * 2:
            return None

        for agent_id in sorted(member_list):
            record = self.agents[agent_id]
            if record.parent_id is None:
                continue
            if record.birth_tick < segment.start_tick + self.thresholds.min_split_gap_ticks:
                continue
            if record.parent_id not in segment.members:
                continue
            daughter_members = frozenset(
                candidate_id
                for candidate_id in member_list
                if self._is_descendant(agent_id, candidate_id)
            )
            if len(daughter_members) < self.thresholds.min_branch_member_count:
                continue
            continuation_members = frozenset(member_list) - daughter_members
            if len(continuation_members) < self.thresholds.min_overlap_members:
                continue
            candidate = self._evaluate_candidate(
                segment=segment,
                parent_agent_id=record.parent_id,
                child_agent_id=agent_id,
                daughter_members=daughter_members,
                continuation_members=continuation_members,
            )
            if candidate is None:
                continue
            if best is None or self._candidate_sort_key(candidate) > self._candidate_sort_key(best):
                best = candidate
        return best

    @staticmethod
    def _candidate_sort_key(candidate: CandidateSplit) -> tuple[float, float, float, int]:
        return (
            round(candidate.score, 6),
            round(candidate.genetic_distance + candidate.ecological_divergence, 6),
            float(candidate.overlap_ticks),
            -candidate.split_tick,
        )

    def _evaluate_candidate(
        self,
        *,
        segment: SpeciesSegment,
        parent_agent_id: int,
        child_agent_id: int,
        daughter_members: frozenset[int],
        continuation_members: frozenset[int],
    ) -> CandidateSplit | None:
        split_tick = self.agents[child_agent_id].birth_tick
        branch_peak_members = 0
        overlap_ticks = 0
        branch_last_alive_tick = split_tick - 1
        branch_overlap_observations = 0
        continuation_overlap_observations = 0
        branch_genome_totals = [0.0 for _ in self.agents[child_agent_id].genome_vector]
        continuation_genome_totals = [0.0 for _ in self.agents[child_agent_id].genome_vector]
        branch_ecotype_counts: Counter[int] = Counter()
        continuation_ecotype_counts: Counter[int] = Counter()
        branch_profiles = self._empty_profile_counters()
        continuation_profiles = self._empty_profile_counters()

        start_index = bisect_left(self.frame_ticks, split_tick)
        for frame_index in range(start_index, len(self.frames)):
            tick = self.frame_ticks[frame_index]
            if tick < segment.start_tick:
                continue
            rows_by_agent = self.frame_rows[frame_index]
            branch_rows: list[list[object]] = []
            continuation_rows: list[list[object]] = []
            for agent_id in self.frame_alive_ids[frame_index]:
                if agent_id in daughter_members:
                    branch_rows.append(rows_by_agent[agent_id])
                elif agent_id in continuation_members:
                    continuation_rows.append(rows_by_agent[agent_id])

            branch_count = len(branch_rows)
            continuation_count = len(continuation_rows)
            branch_peak_members = max(branch_peak_members, branch_count)
            if branch_count > 0:
                branch_last_alive_tick = tick

            if (
                branch_count < self.thresholds.min_overlap_members
                or continuation_count < self.thresholds.min_overlap_members
            ):
                continue

            overlap_ticks += 1
            for row in branch_rows:
                agent_id = int(row[self.field_map["agent_id"]])
                branch_overlap_observations += 1
                self._add_vector(branch_genome_totals, self.agents[agent_id].genome_vector)
                branch_ecotype_counts[int(row[self.field_map["ecotype_id"]])] += 1
                self._accumulate_profile(branch_profiles, frame_index, row)

            for row in continuation_rows:
                agent_id = int(row[self.field_map["agent_id"]])
                continuation_overlap_observations += 1
                self._add_vector(
                    continuation_genome_totals,
                    self.agents[agent_id].genome_vector,
                )
                continuation_ecotype_counts[int(row[self.field_map["ecotype_id"]])] += 1
                self._accumulate_profile(continuation_profiles, frame_index, row)

        branch_persistence_ticks = (
            branch_last_alive_tick - split_tick + 1 if branch_last_alive_tick >= split_tick else 0
        )
        if branch_peak_members < self.thresholds.min_branch_peak_members:
            return None
        if branch_persistence_ticks < self.thresholds.min_branch_persistence_ticks:
            return None
        if overlap_ticks < self.thresholds.min_overlap_ticks:
            return None
        if branch_overlap_observations <= 0 or continuation_overlap_observations <= 0:
            return None

        branch_centroid = tuple(
            total / max(branch_overlap_observations, 1) for total in branch_genome_totals
        )
        continuation_centroid = tuple(
            total / max(continuation_overlap_observations, 1)
            for total in continuation_genome_totals
        )
        genetic_distance = euclidean_distance(branch_centroid, continuation_centroid)
        ecotype_divergence = _js_divergence(
            branch_ecotype_counts,
            continuation_ecotype_counts,
        )
        ecological_divergence = self._ecological_divergence(
            branch_profiles,
            continuation_profiles,
            branch_overlap_observations,
            continuation_overlap_observations,
        )
        if genetic_distance < self.thresholds.min_genetic_distance:
            return None
        if (
            ecotype_divergence < self.thresholds.min_ecotype_divergence
            and ecological_divergence < self.thresholds.min_ecological_divergence
        ):
            return None

        balance_ratio = min(branch_overlap_observations, continuation_overlap_observations) / max(
            branch_overlap_observations,
            continuation_overlap_observations,
        )
        score = (
            0.8
            * _saturating_ratio(
                len(daughter_members),
                self.thresholds.min_branch_member_count,
            )
            + 0.8
            * _saturating_ratio(
                branch_peak_members,
                self.thresholds.min_branch_peak_members,
            )
            + 0.9
            * _saturating_ratio(
                branch_persistence_ticks,
                self.thresholds.min_branch_persistence_ticks,
            )
            + 1.0 * _saturating_ratio(overlap_ticks, self.thresholds.min_overlap_ticks)
            + 1.2 * _saturating_ratio(genetic_distance, self.thresholds.min_genetic_distance)
            + 0.8
            * _saturating_ratio(
                ecotype_divergence,
                self.thresholds.min_ecotype_divergence,
            )
            + 0.9
            * _saturating_ratio(
                ecological_divergence,
                self.thresholds.min_ecological_divergence,
            )
            + 0.4 * balance_ratio
        )
        if score < self.thresholds.min_split_score:
            return None

        return CandidateSplit(
            parent_agent_id=parent_agent_id,
            child_agent_id=child_agent_id,
            split_tick=split_tick,
            branch_member_count=len(daughter_members),
            branch_peak_members=branch_peak_members,
            branch_persistence_ticks=branch_persistence_ticks,
            overlap_ticks=overlap_ticks,
            genetic_distance=genetic_distance,
            ecotype_divergence=ecotype_divergence,
            ecological_divergence=ecological_divergence,
            balance_ratio=balance_ratio,
            score=score,
            daughter_members=daughter_members,
            continuation_members=continuation_members,
        )

    @staticmethod
    def _empty_profile_counters() -> dict[str, Counter[str]]:
        return {
            "trophic": Counter(),
            "meat_mode": Counter(),
            "terrain": Counter(),
            "water": Counter(),
            "habitat": Counter(),
            "ecology": Counter(),
            "hazard": Counter(),
            "refuge": Counter(),
        }

    def _accumulate_profile(
        self,
        profiles: dict[str, Counter[str]],
        frame_index: int,
        row: list[object],
    ) -> None:
        x = int(row[self.field_map["x"]])
        y = int(row[self.field_map["y"]])
        profiles["trophic"][str(row[self.field_map["trophic_role"]])] += 1
        profiles["meat_mode"][str(row[self.field_map["meat_mode"]])] += 1
        profiles["terrain"][self.terrain_legend[int(self.terrain_codes[y][x])]] += 1
        profiles["water"][str(row[self.field_map["water_access_reason"]])] += 1
        profiles["habitat"][
            self.habitat_legend[int(self.frames[frame_index]["habitat_state_codes"][y][x])]
        ] += 1
        profiles["ecology"][
            self.ecology_legend[int(self.frames[frame_index]["ecology_state_codes"][y][x])]
        ] += 1
        profiles["hazard"][
            self.hazard_legend[int(self.frames[frame_index]["hazard_type_codes"][y][x])]
        ] += 1
        profiles["refuge"][
            "covered"
            if str(row[self.field_map["soft_refuge_reason"]]) != "none"
            else "open"
        ] += 1

    def _ecological_divergence(
        self,
        branch_profiles: dict[str, Counter[str]],
        continuation_profiles: dict[str, Counter[str]],
        branch_total: int,
        continuation_total: int,
    ) -> float:
        components = [
            _total_variation_distance(branch_profiles["trophic"], continuation_profiles["trophic"]),
            _total_variation_distance(
                branch_profiles["meat_mode"],
                continuation_profiles["meat_mode"],
            ),
            _total_variation_distance(branch_profiles["terrain"], continuation_profiles["terrain"]),
            _total_variation_distance(branch_profiles["water"], continuation_profiles["water"]),
            _total_variation_distance(branch_profiles["habitat"], continuation_profiles["habitat"]),
            _total_variation_distance(branch_profiles["ecology"], continuation_profiles["ecology"]),
            _total_variation_distance(branch_profiles["hazard"], continuation_profiles["hazard"]),
            _total_variation_distance(branch_profiles["refuge"], continuation_profiles["refuge"]),
        ]
        branch_refuge_rate = branch_profiles["refuge"]["covered"] / max(branch_total, 1)
        continuation_refuge_rate = continuation_profiles["refuge"]["covered"] / max(
            continuation_total,
            1,
        )
        branch_hazard_rate = (
            branch_total - branch_profiles["hazard"]["none"]
        ) / max(branch_total, 1)
        continuation_hazard_rate = (
            continuation_total - continuation_profiles["hazard"]["none"]
        ) / max(continuation_total, 1)
        components.append(abs(branch_refuge_rate - continuation_refuge_rate))
        components.append(abs(branch_hazard_rate - continuation_hazard_rate))
        return sum(components) / max(len(components), 1)

    @staticmethod
    def _add_vector(target: list[float], vector: tuple[float, ...]) -> None:
        for index, value in enumerate(vector):
            target[index] += value

    def _rewrite_species_assignments(self) -> None:
        species_column = self.field_map["species_id"]
        previous_species_map: dict[int, int] = {}
        species_catalog = self._initial_species_catalog()
        for frame_index, frame in enumerate(self.frames):
            tick = self.frame_ticks[frame_index]
            current_species_map: dict[int, int] = {}
            counts: Counter[int] = Counter()
            for row in frame["agents"]:
                agent_id = int(row[self.field_map["agent_id"]])
                species_id = self._resolve_species_for_agent(agent_id, tick)
                row[species_column] = species_id
                current_species_map[agent_id] = species_id
                counts[species_id] += 1

            frame["species_counts"] = [
                [species_id, count]
                for species_id, count in sorted(
                    counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            frame["species_metrics"] = self._build_frame_species_metrics(
                frame_index=frame_index,
                current_species_map=current_species_map,
                previous_species_map=previous_species_map,
            )
            for species_id, count in counts.items():
                record = species_catalog[str(species_id)]
                record["last_seen_tick"] = tick
                record["observed_ticks"] += 1
                record["peak_members"] = max(record["peak_members"], count)
                record["_current_members"] = count
            for species_id, record in species_catalog.items():
                if int(species_id) not in counts:
                    record["_current_members"] = 0
            previous_species_map = current_species_map

        final_counts = {
            int(species_id): int(record["_current_members"])
            for species_id, record in species_catalog.items()
        }
        for species_id, record in species_catalog.items():
            current_members = final_counts[int(species_id)]
            record["current_members"] = current_members
            record["status"] = self._segment_status(
                self.segments[int(species_id)],
                current_members=current_members,
            )
            record["is_alive_terminal"] = record["status"] == "extant"
            record.pop("_current_members", None)
        self.species_status_counts = dict(
            sorted(
                Counter(record["status"] for record in species_catalog.values()).items(),
                key=lambda item: item[0],
            )
        )
        self.viewer["species_catalog"] = {
            species_id: {
                key: value
                for key, value in record.items()
                if not key.startswith("_")
            }
            for species_id, record in sorted(species_catalog.items(), key=lambda item: int(item[0]))
        }
        self._rewrite_provenance_surfaces()
        self.viewer["taxonomy"] = self._build_taxonomy_payload(events=self.speciation_events)

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _resolve_provenance_species(
        self,
        *,
        source_agent_id: object,
        death_tick: object,
        fallback_source_species: object,
    ) -> int | None:
        fallback = self._optional_int(fallback_source_species)
        agent_id = self._optional_int(source_agent_id)
        tick = self._optional_int(death_tick)
        if agent_id is None or tick is None or agent_id not in self.agents:
            return fallback
        return self._resolve_species_for_agent(agent_id, tick)

    @staticmethod
    def _resolved_source_species(source_breakdown: object) -> int | None:
        if not isinstance(source_breakdown, list) or not source_breakdown:
            return None
        source_species = {
            entry.get("source_species")
            for entry in source_breakdown
            if isinstance(entry, dict)
        }
        if not source_species or None in source_species:
            return None
        if len(source_species) == 1:
            return next(iter(source_species))
        return None

    def _rewrite_source_breakdown(self, source_breakdown: object) -> list[dict[str, object]]:
        if not isinstance(source_breakdown, list):
            return []
        for entry in source_breakdown:
            if not isinstance(entry, dict):
                continue
            entry["source_species"] = self._resolve_provenance_species(
                source_agent_id=entry.get("source_agent_id"),
                death_tick=entry.get("death_tick"),
                fallback_source_species=entry.get("source_species"),
            )
        return [entry for entry in source_breakdown if isinstance(entry, dict)]

    def _rewrite_provenance_payload(
        self,
        payload: dict[str, object],
        *,
        breakdown_key: str,
        dominant_key: str | None = None,
        mixed_key: str | None = None,
        aggregate_key: str | None = None,
    ) -> list[dict[str, object]]:
        rewritten = self._rewrite_source_breakdown(payload.get(breakdown_key))
        if not rewritten:
            return []
        resolved_species = self._resolved_source_species(rewritten)
        if dominant_key is not None:
            payload[dominant_key] = resolved_species
        if mixed_key is not None:
            payload[mixed_key] = len(rewritten) > 1
        if aggregate_key is not None:
            payload[aggregate_key] = resolved_species
        return rewritten

    def _rewrite_event_provenance(self, event: dict[str, object]) -> None:
        data = event.get("data")
        if not isinstance(data, dict):
            return
        event_type = str(event.get("type"))
        if event_type == "agent_died":
            data["source_species"] = self._resolve_provenance_species(
                source_agent_id=event.get("agent_id"),
                death_tick=event.get("tick"),
                fallback_source_species=data.get("source_species"),
            )
            self._rewrite_provenance_payload(
                data,
                breakdown_key="tile_source_breakdown_after",
                dominant_key="tile_dominant_source_species_after",
                mixed_key="tile_mixed_sources_after",
            )
            return
        if event_type == "carcass_deposited":
            rewritten = self._rewrite_provenance_payload(
                data,
                breakdown_key="source_breakdown",
                aggregate_key="source_species",
            )
            if not rewritten:
                data["source_species"] = self._resolve_provenance_species(
                    source_agent_id=data.get("source_agent_id", event.get("agent_id")),
                    death_tick=event.get("tick"),
                    fallback_source_species=data.get("source_species"),
                )
            self._rewrite_provenance_payload(
                data,
                breakdown_key="tile_source_breakdown_after",
                dominant_key="tile_dominant_source_species_after",
                mixed_key="tile_mixed_sources_after",
            )
            return
        if event_type != "agent_ate":
            return
        food_source = str(data.get("food_source"))
        if food_source not in {"carcass", "fresh_kill"}:
            return
        self._rewrite_provenance_payload(data, breakdown_key="source_breakdown")
        self._rewrite_provenance_payload(data, breakdown_key="deposit_breakdown")
        rewritten_tile = self._rewrite_provenance_payload(
            data,
            breakdown_key="tile_source_breakdown_after",
            dominant_key="tile_dominant_source_species_after",
            mixed_key="tile_mixed_sources_after",
        )
        if not rewritten_tile and "tile_dominant_source_species_after" in data:
            data["tile_dominant_source_species_after"] = self._optional_int(
                data.get("tile_dominant_source_species_after")
            )

    def _rewrite_patch_provenance(self, patch: object) -> None:
        if not isinstance(patch, dict):
            return
        self._rewrite_provenance_payload(
            patch,
            breakdown_key="source_breakdown",
            dominant_key="dominant_source_species",
            mixed_key="mixed_sources",
        )

    def _rewrite_provenance_surfaces(self) -> None:
        for event in self.events:
            if isinstance(event, dict):
                self._rewrite_event_provenance(event)
        for frame in self.frames:
            for patch in frame.get("carcass_patches", []):
                self._rewrite_patch_provenance(patch)
            for patch in frame.get("fresh_kill_patches", []):
                self._rewrite_patch_provenance(patch)

    def _initial_species_catalog(self) -> dict[str, dict[str, object]]:
        catalog: dict[str, dict[str, object]] = {}
        for species_id, segment in sorted(self.segments.items()):
            catalog[str(species_id)] = {
                "species_id": species_id,
                "label": segment.label,
                "start_tick": segment.start_tick,
                "end_tick": segment.end_tick,
                "first_seen_tick": segment.start_tick,
                "last_seen_tick": segment.end_tick if segment.end_tick is not None else segment.start_tick,
                "observed_ticks": 0,
                "peak_members": 0,
                "lineages": [segment.lineage_id],
                "taxonomy_origin": segment.origin_kind,
                "identity_mode": REPLAY_TAXONOMY_MODE,
                "status": "pending",
                "child_species_ids": sorted(segment.child_species_ids),
                "is_leaf_species": not segment.child_species_ids,
                "founder_agent_id": segment.founder_agent_id,
                "parent_species_id": segment.parent_species_id,
                "source_species_id": segment.source_species_id,
                "split_parent_agent_id": segment.split_parent_agent_id,
                "split_child_agent_id": segment.split_child_agent_id,
                "split_tick": segment.start_tick if segment.parent_species_id is not None else None,
                "evidence": segment.evidence,
                "centroid_units": "raw_gene_values",
                "centroid": self._segment_centroid(segment),
                "normalized_centroid_units": "unit_interval_by_gene_limits",
                "normalized_centroid": self._segment_normalized_centroid(segment),
            }
        return catalog

    def _segment_centroid(self, segment: SpeciesSegment) -> dict[str, float]:
        return self._segment_centroid_from_values(segment, normalized=False)

    def _segment_normalized_centroid(self, segment: SpeciesSegment) -> dict[str, float]:
        return self._segment_centroid_from_values(segment, normalized=True)

    def _segment_centroid_from_values(
        self,
        segment: SpeciesSegment,
        *,
        normalized: bool,
    ) -> dict[str, float]:
        if not segment.members:
            return {}
        end_tick = segment.end_tick if segment.end_tick is not None else self.final_tick
        totals = {gene: 0.0 for gene in GENE_ORDER}
        count = 0
        for agent_id in segment.members:
            record = self.agents[agent_id]
            if record.birth_tick > end_tick:
                continue
            if record.last_alive_tick(self.final_tick) < segment.start_tick:
                continue
            for index, gene in enumerate(GENE_ORDER):
                if normalized:
                    totals[gene] += record.genome_vector[index]
                else:
                    totals[gene] += record.genome_values[gene]
            count += 1
        if count <= 0:
            return {}
        return {
            gene: round(totals[gene] / count, 4) for gene in GENE_ORDER
        }

    @staticmethod
    def _segment_status(segment: SpeciesSegment, *, current_members: int) -> str:
        if segment.child_species_ids:
            if current_members > 0:
                return "ancestral_remainder"
            return "split_ancestor"
        if current_members > 0:
            return "extant"
        return "extinct"

    def _build_taxonomy_payload(self, *, events: list[dict[str, object]]) -> dict[str, object]:
        return {
            "version": REPLAY_TAXONOMY_MODE,
            "species_identity": REPLAY_TAXONOMY_MODE,
            "ecotype_identity": ECOTYPE_IDENTITY_MODE,
            "thresholds": asdict(self.thresholds),
            "species_status_counts": self.species_status_counts,
            "events": events,
            "invariants": [
                "species assignments are replay-derived, not frame-cluster-derived",
                "species changes are one-way and only occur on logged taxonomy event ticks",
                "species centroids are raw gene values and normalized_centroid is unit interval",
                "ecotypes remain transient frame-local genome clusters",
            ],
        }

    def _resolve_species_for_agent(self, agent_id: int, tick: int) -> int:
        species_id = self.agents[agent_id].lineage_id
        while True:
            next_species_id: int | None = None
            for child_species_id in self.segments[species_id].child_species_ids:
                child_segment = self.segments[child_species_id]
                if tick >= child_segment.start_tick and agent_id in child_segment.members:
                    next_species_id = child_species_id
                    break
            if next_species_id is None:
                return species_id
            species_id = next_species_id

    def _build_frame_species_metrics(
        self,
        *,
        frame_index: int,
        current_species_map: dict[int, int],
        previous_species_map: dict[int, int],
    ) -> dict[str, dict[str, object]]:
        frame = self.frames[frame_index]
        tick = self.frame_ticks[frame_index]
        return build_shared_species_metrics(
            occupancy_samples=(
                SpeciesMetricSample(
                    species_id=current_species_map.get(int(row[self.field_map["agent_id"]]), 0),
                    terrain=self.terrain_legend[int(self.terrain_codes[int(row[self.field_map["y"]])][int(row[self.field_map["x"]])])],
                    habitat=self.habitat_legend[int(frame["habitat_state_codes"][int(row[self.field_map["y"]])][int(row[self.field_map["x"]])])],
                    ecology=self.ecology_legend[int(frame["ecology_state_codes"][int(row[self.field_map["y"]])][int(row[self.field_map["x"]])])],
                    hazard=self.hazard_legend[int(frame["hazard_type_codes"][int(row[self.field_map["y"]])][int(row[self.field_map["x"]])])],
                    trophic_role=str(row[self.field_map["trophic_role"]]),
                    meat_mode=str(row[self.field_map["meat_mode"]]),
                    water_reason=str(row[self.field_map["water_access_reason"]]),
                    has_water_access=str(row[self.field_map["water_access_reason"]]) != "none",
                    shoreline_support=bool(
                        int(row[self.field_map["hydrology_support_code"]])
                        & HYDROLOGY_SUPPORT_FLAGS["adjacent_to_water"]
                    ),
                    wetland_support=bool(
                        int(row[self.field_map["hydrology_support_code"]])
                        & HYDROLOGY_SUPPORT_FLAGS["wetland"]
                    ),
                    flooded_support=bool(
                        int(row[self.field_map["hydrology_support_code"]])
                        & HYDROLOGY_SUPPORT_FLAGS["flooded"]
                    ),
                    refuge_exposed=str(row[self.field_map["soft_refuge_reason"]]) != "none",
                    energy_ratio=float(row[self.field_map["energy_ratio"]]),
                    hydration_ratio=float(row[self.field_map["hydration_ratio"]]),
                    health_ratio=float(row[self.field_map["health_ratio"]]),
                    matched_diet_ratio=float(row[self.field_map["matched_diet_ratio"]]),
                    age=float(row[self.field_map["age"]]),
                    vegetation=float(row[self.field_map["tile_vegetation"]]),
                    recovery_debt=float(row[self.field_map["tile_recovery_debt"]]),
                    refuge_score=float(row[self.field_map["refuge_score"]]),
                    injury_load=float(row[self.field_map["injury_load"]]),
                    reproduction_ready=int(row[self.field_map["reproduction_ready"]]) > 0,
                )
                for row in frame["agents"]
            ),
            species_map=current_species_map,
            previous_species_map=previous_species_map,
            birth_pairs=self.birth_events_by_tick.get(tick, []),
            dead_agent_ids=(int(event["agent_id"]) for event in self.death_events_by_tick.get(tick, [])),
            attack_records=(
                (
                    int(event["agent_id"]),
                    bool(event["data"].get("success")),
                    float(event["data"].get("damage", 0.0)),
                    bool(event["data"].get("kill")),
                )
                for event in self.attack_events_by_tick.get(tick, [])
            ),
            damage_records=(
                (
                    int(event["agent_id"]),
                    float(event["data"].get("amount", 0.0)),
                    str(event["data"].get("source", "")),
                )
                for event in self.damage_events_by_tick.get(tick, [])
            ),
            carcass_deposit_records=(
                (
                    previous_species_map.get(int(event["agent_id"])),
                    float(event["data"].get("carcass_energy", 0.0)),
                )
                for event in self.death_events_by_tick.get(tick, [])
                if float(event["data"].get("carcass_energy", 0.0)) > 0
            ),
            fresh_kill_deposit_records=(
                (
                    previous_species_map.get(int(event["agent_id"])),
                    float(event["data"].get("fresh_kill_energy", 0.0)),
                )
                for event in self.death_events_by_tick.get(tick, [])
                if float(event["data"].get("fresh_kill_energy", 0.0)) > 0
            ),
            fresh_kill_consumption_records=(
                (
                    int(event["agent_id"]),
                    float(event["data"].get("consumed", 0.0)),
                    float(event["data"].get("gained_energy", 0.0)),
                )
                for event in self.feed_events_by_tick.get(tick, [])
                if str(event["data"].get("food_source")) == "fresh_kill"
            ),
            carcass_consumption_records=(
                (
                    int(event["agent_id"]),
                    float(event["data"].get("consumed", 0.0)),
                    float(event["data"].get("gained_energy", 0.0)),
                )
                for event in self.feed_events_by_tick.get(tick, [])
                if str(event["data"].get("food_source")) == "carcass"
            ),
            feeding_records=(
                (
                    int(event["agent_id"]),
                    str(event["data"].get("food_source", "plant")),
                    float(event["data"].get("gained_energy", 0.0)),
                )
                for event in self.feed_events_by_tick.get(tick, [])
            ),
        )

    def _rebuild_summary_and_analytics(self) -> None:
        rebuild_taxonomy_summary_and_analytics(
            summary=self.summary,
            viewer=self.viewer,
            frames=self.frames,
            frame_ticks=self.frame_ticks,
            speciation_events=self.speciation_events,
            species_status_counts=self.species_status_counts,
            taxonomy_mode=REPLAY_TAXONOMY_MODE,
        )


def apply_replay_taxonomy(
    *,
    config: WorldConfig,
    summary: dict[str, object],
    events: list[dict[str, object]],
    viewer: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    taxonomy_payload = viewer.get("taxonomy")
    if (
        isinstance(taxonomy_payload, dict)
        and taxonomy_payload.get("version") == REPLAY_TAXONOMY_MODE
        and summary.get("taxonomy_mode") == REPLAY_TAXONOMY_MODE
    ):
        return summary, viewer
    return ReplayTaxonomyPass(
        config=config,
        summary=summary,
        events=events,
        viewer=viewer,
    ).apply()


def _saturating_ratio(value: float, threshold: float, cap: float = 2.0) -> float:
    if threshold <= 0:
        return 0.0
    return min(value / threshold, cap)


def _total_variation_distance(left: Counter[Any], right: Counter[Any]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        return 0.0
    keys = sorted(set(left) | set(right))
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total) for key in keys
    )


def _js_divergence(left: Counter[Any], right: Counter[Any]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        return 0.0
    keys = sorted(set(left) | set(right))
    divergence = 0.0
    for key in keys:
        left_prob = left.get(key, 0) / left_total
        right_prob = right.get(key, 0) / right_total
        mean_prob = (left_prob + right_prob) / 2
        if left_prob > 0:
            divergence += 0.5 * left_prob * log2(left_prob / mean_prob)
        if right_prob > 0:
            divergence += 0.5 * right_prob * log2(right_prob / mean_prob)
    return divergence
