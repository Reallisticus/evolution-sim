from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TextIO

from evolution_sim.mind.dataset import (
    TRAJECTORY_EPISODE_ID_FIELD,
    TrajectoryJsonlDataset,
    combined_dataset_provenance,
    records_with_trajectory_context,
)
from evolution_sim.mind.horizon_labels import ANIMAL_RESOURCE_FOOD_SOURCES
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_CARRION_AUTOPSY_SCHEMA_VERSION = "mind_v3_carrion_failure_autopsy_v1"
MIND_V3_CARRION_AUTOPSY_POLICY = (
    "post_animal_resource_contact_death_path_attribution_v1"
)
DEFAULT_POST_CONTACT_WINDOW_TICKS = 120
DEFAULT_SEQUENCE_RECORD_LIMIT = 12
DEFAULT_MAX_EXAMPLES = 8
DEFAULT_CRITICAL_RATIO_FLOOR = 0.5


class CarrionAutopsyError(ValueError):
    pass


def build_carrion_autopsy_report(
    datasets: Sequence[TrajectoryJsonlDataset],
    *,
    post_contact_window_ticks: int = DEFAULT_POST_CONTACT_WINDOW_TICKS,
    sequence_record_limit: int = DEFAULT_SEQUENCE_RECORD_LIMIT,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    critical_ratio_floor: float = DEFAULT_CRITICAL_RATIO_FLOOR,
) -> dict[str, object]:
    if not datasets:
        raise CarrionAutopsyError("at least one trajectory dataset is required")
    window_ticks = _positive_int(
        post_contact_window_ticks,
        field="post_contact_window_ticks",
    )
    sequence_limit = _positive_int(
        sequence_record_limit,
        field="sequence_record_limit",
    )
    example_limit = _nonnegative_int(max_examples, field="max_examples")
    ratio_floor = _ratio_floor(critical_ratio_floor)
    contextual_records = [
        {**record, "source_record_index": index}
        for index, record in enumerate(records_with_trajectory_context(datasets))
    ]
    episodes = _post_contact_episodes(
        contextual_records,
        window_ticks=window_ticks,
        sequence_limit=sequence_limit,
        ratio_floor=ratio_floor,
    )
    aggregate = _aggregate_episodes(episodes)
    contract = _autopsy_contract(
        window_ticks=window_ticks,
        sequence_limit=sequence_limit,
        example_limit=example_limit,
        ratio_floor=ratio_floor,
    )
    examples = _select_examples(
        episodes,
        dominant_death_path=aggregate["dominant_death_path"]["path"],
        limit=example_limit,
    )
    provenance = combined_dataset_provenance(datasets)
    return {
        "schema_version": MIND_V3_CARRION_AUTOPSY_SCHEMA_VERSION,
        "autopsy_policy": MIND_V3_CARRION_AUTOPSY_POLICY,
        "autopsy_contract": contract,
        "provenance": {
            **provenance,
            "autopsy_contract_digest": stable_payload_digest(contract),
        },
        "source": {
            "trajectory_count": len(datasets),
            "trajectory_paths": [str(dataset.path) for dataset in datasets],
            "record_count": len(contextual_records),
        },
        "aggregate": aggregate,
        "examples": examples,
    }


def write_carrion_autopsy_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(
            report,
            handle,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def _autopsy_contract(
    *,
    window_ticks: int,
    sequence_limit: int,
    example_limit: int,
    ratio_floor: float,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_AUTOPSY_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_AUTOPSY_POLICY,
        "animal_resource_food_sources": sorted(ANIMAL_RESOURCE_FOOD_SOURCES),
        "contact_definition": (
            "first_posted_trajectory_record_with_feeding_food_source_in_"
            "carcass_or_fresh_kill_v1"
        ),
        "window_policy": "first_contact_tick_through_contact_plus_window_or_death_v1",
        "post_contact_window_ticks": window_ticks,
        "sequence_record_limit": sequence_limit,
        "max_examples": example_limit,
        "critical_ratio_floor": ratio_floor,
        "death_path_policy": "death_cause_action_and_terminal_constraint_rules_v1",
    }


def _post_contact_episodes(
    records: Sequence[Mapping[str, object]],
    *,
    window_ticks: int,
    sequence_limit: int,
    ratio_floor: float,
) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    for (episode_id, agent_id), timeline in _group_agent_timelines(records).items():
        contact_index = _first_animal_resource_contact_index(timeline)
        if contact_index is None:
            continue
        episodes.append(
            _summarize_contact_episode(
                episode_id=episode_id,
                agent_id=agent_id,
                timeline=timeline,
                contact_index=contact_index,
                window_ticks=window_ticks,
                sequence_limit=sequence_limit,
                ratio_floor=ratio_floor,
            )
        )
    episodes.sort(
        key=lambda episode: (
            str(episode["episode_id"]),
            int(episode["agent_id"]),
            int(episode["first_contact_tick"]),
        )
    )
    return episodes


def _group_agent_timelines(
    records: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], list[Mapping[str, object]]]:
    timelines: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for record in records:
        episode_id = record.get(TRAJECTORY_EPISODE_ID_FIELD)
        if episode_id is None:
            episode_id = "inferred-episode-0"
        if not isinstance(episode_id, str) or not episode_id:
            raise CarrionAutopsyError("trajectory episode id must be a non-empty string")
        agent_id = _record_agent_id(record)
        timelines.setdefault((episode_id, agent_id), []).append(record)
    for timeline in timelines.values():
        timeline.sort(key=lambda record: (_record_tick(record), _record_index(record)))
    return timelines


def _summarize_contact_episode(
    *,
    episode_id: str,
    agent_id: int,
    timeline: Sequence[Mapping[str, object]],
    contact_index: int,
    window_ticks: int,
    sequence_limit: int,
    ratio_floor: float,
) -> dict[str, object]:
    first_contact = timeline[contact_index]
    first_contact_tick = _record_tick(first_contact)
    max_tick = first_contact_tick + window_ticks
    window_records = [
        record
        for record in timeline[contact_index:]
        if _record_tick(record) <= max_tick
    ]
    if not window_records:
        window_records = [first_contact]
    death_record = next(
        (record for record in window_records if _record_dead_after(record)),
        None,
    )
    terminal_record = death_record or window_records[-1]
    terminal_state = _state_payload(terminal_record, "after")
    minimum_state = _minimum_state(window_records)
    terminal_bottleneck = _terminal_bottleneck(
        terminal_state,
        ratio_floor=ratio_floor,
    )
    action_counts = Counter(str(record.get("requested_action", "")) for record in window_records)
    resolved_action_counts = Counter(
        str(record.get("resolved_action", "")) for record in window_records
    )
    death_cause = _death_cause(death_record) if death_record is not None else None
    summary = {
        "episode_id": episode_id,
        "agent_id": agent_id,
        "source_record_index": _record_index(first_contact),
        "first_contact_tick": first_contact_tick,
        "analysis_end_tick": _record_tick(terminal_record),
        "post_contact_window_ticks": window_ticks,
        "contact_to_terminal_ticks": _record_tick(terminal_record) - first_contact_tick,
        "died_after_contact": death_record is not None,
        "death_tick": _record_tick(death_record) if death_record is not None else None,
        "death_cause": death_cause,
        "terminal_state": terminal_state,
        "minimum_state": minimum_state,
        "terminal_bottleneck": terminal_bottleneck,
        "action_counts_after_contact": dict(sorted(action_counts.items())),
        "resolved_action_counts_after_contact": dict(
            sorted(resolved_action_counts.items())
        ),
        "movement_action_count": _movement_action_count(window_records),
        "drink_count": _drink_count(window_records),
        "animal_resource_event_count": _animal_resource_event_count(window_records),
        "animal_resource_gain": _round(_animal_resource_gain(window_records)),
        "low_gain_eat_count": _low_gain_eat_count(window_records),
        "reproduction_after_contact_count": _reproduction_count(window_records),
        "first_contact": _event_excerpt(first_contact),
        "sequence_excerpt": _sequence_excerpt(
            window_records,
            limit=sequence_limit,
        ),
        "pre_terminal_excerpt": _sequence_excerpt(
            window_records[-sequence_limit:],
            limit=sequence_limit,
        ),
    }
    summary["death_path"] = _death_path(summary)
    summary["dominant_failure_hypothesis"] = _failure_hypothesis(summary)
    return summary


def _aggregate_episodes(episodes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    contact_count = len(episodes)
    died = [episode for episode in episodes if episode.get("died_after_contact") is True]
    survived = contact_count - len(died)
    death_path_counts = Counter(
        str(episode.get("death_path", "unknown")) for episode in died
    )
    contact_outcome_path_counts = Counter(
        str(episode.get("death_path", "unknown")) for episode in episodes
    )
    death_cause_counts = Counter(
        str(episode.get("death_cause") or "none") for episode in episodes
    )
    bottleneck_counts = Counter(
        str(episode.get("terminal_bottleneck") or "none") for episode in episodes
    )
    action_counts: Counter[str] = Counter()
    resolved_action_counts: Counter[str] = Counter()
    for episode in episodes:
        action_counts.update(_counter_mapping(episode.get("action_counts_after_contact")))
        resolved_action_counts.update(
            _counter_mapping(episode.get("resolved_action_counts_after_contact"))
        )
    return {
        "contact_episode_count": contact_count,
        "death_after_contact_count": len(died),
        "survived_contact_window_count": survived,
        "post_contact_survival_rate": _safe_rate(survived, contact_count),
        "dominant_death_path": _dominant_counter_payload(
            death_path_counts,
            denominator=len(died),
        ),
        "dominant_contact_outcome_path": _dominant_counter_payload(
            contact_outcome_path_counts,
            denominator=contact_count,
        ),
        "death_path_counts": dict(sorted(death_path_counts.items())),
        "contact_outcome_path_counts": dict(
            sorted(contact_outcome_path_counts.items())
        ),
        "death_cause_counts": dict(sorted(death_cause_counts.items())),
        "terminal_bottleneck_counts": dict(sorted(bottleneck_counts.items())),
        "action_counts_after_contact": dict(sorted(action_counts.items())),
        "resolved_action_counts_after_contact": dict(
            sorted(resolved_action_counts.items())
        ),
        "animal_resource_event_count": sum(
            int(episode.get("animal_resource_event_count", 0)) for episode in episodes
        ),
        "animal_resource_gain": _round(
            sum(float(episode.get("animal_resource_gain", 0.0)) for episode in episodes)
        ),
        "drink_count": sum(int(episode.get("drink_count", 0)) for episode in episodes),
        "movement_action_count": sum(
            int(episode.get("movement_action_count", 0)) for episode in episodes
        ),
        "low_gain_eat_count": sum(
            int(episode.get("low_gain_eat_count", 0)) for episode in episodes
        ),
        "reproduction_after_contact_count": sum(
            int(episode.get("reproduction_after_contact_count", 0))
            for episode in episodes
        ),
        "mean_contact_to_terminal_ticks": _mean(
            int(episode.get("contact_to_terminal_ticks", 0)) for episode in episodes
        ),
        "mean_terminal_energy_ratio": _mean_optional(
            _mapping_float(episode.get("terminal_state"), "energy_ratio")
            for episode in episodes
        ),
        "mean_terminal_hydration_ratio": _mean_optional(
            _mapping_float(episode.get("terminal_state"), "hydration_ratio")
            for episode in episodes
        ),
        "mean_terminal_health_ratio": _mean_optional(
            _mapping_float(episode.get("terminal_state"), "health_ratio")
            for episode in episodes
        ),
    }


def _select_examples(
    episodes: Sequence[Mapping[str, object]],
    *,
    dominant_death_path: object,
    limit: int,
) -> list[dict[str, object]]:
    if limit <= 0:
        return []
    dominant_path = dominant_death_path if isinstance(dominant_death_path, str) else None
    selected = sorted(
        episodes,
        key=lambda episode: (
            episode.get("died_after_contact") is not True,
            str(episode.get("death_path")) != str(dominant_path),
            int(episode.get("contact_to_terminal_ticks", 0)),
            str(episode.get("episode_id", "")),
            int(episode.get("agent_id", 0)),
        ),
    )
    return [dict(episode) for episode in selected[:limit]]


def _first_animal_resource_contact_index(
    timeline: Sequence[Mapping[str, object]],
) -> int | None:
    for index, record in enumerate(timeline):
        feeding = _feeding(record)
        if feeding.get("food_source") not in ANIMAL_RESOURCE_FOOD_SOURCES:
            continue
        if feeding.get("ate") is False:
            continue
        return index
    return None


def _death_path(summary: Mapping[str, object]) -> str:
    if summary.get("died_after_contact") is not True:
        bottleneck = summary.get("terminal_bottleneck")
        if isinstance(bottleneck, str) and bottleneck != "none":
            return f"survived_window_but_{bottleneck}_bottleneck"
        return "survived_contact_window"
    death_cause = str(summary.get("death_cause") or "unknown")
    terminal_state = summary.get("terminal_state")
    terminal_action = _terminal_requested_action(summary)
    action_counts = _counter_mapping(summary.get("action_counts_after_contact"))
    low_gain_eat_count = int(summary.get("low_gain_eat_count", 0))
    record_count = max(1, sum(action_counts.values()))
    eat_share = action_counts.get("eat", 0) / record_count
    energy_ratio = _mapping_float(terminal_state, "energy_ratio")
    hydration_ratio = _mapping_float(terminal_state, "hydration_ratio")
    health_ratio = _mapping_float(terminal_state, "health_ratio")
    if death_cause == "energy_depletion" or (
        energy_ratio is not None and energy_ratio <= 0.0
    ):
        if terminal_action.startswith("move_"):
            return "movement_energy_depletion_after_carrion_contact"
        if eat_share >= 0.5 and low_gain_eat_count >= max(1, record_count // 3):
            return "low_gain_eat_energy_depletion_after_carrion_contact"
        return "energy_depletion_after_carrion_contact"
    if death_cause == "hydration_depletion" or (
        hydration_ratio is not None and hydration_ratio <= 0.0
    ):
        if int(summary.get("drink_count", 0)) == 0:
            return "no_drink_hydration_depletion_after_carrion_contact"
        return "hydration_depletion_after_carrion_contact"
    if death_cause in {"health_depletion", "hazard", "attack"} or (
        health_ratio is not None and health_ratio <= 0.0
    ):
        return "health_loss_after_carrion_contact"
    return f"{_safe_label(death_cause)}_after_carrion_contact"


def _failure_hypothesis(summary: Mapping[str, object]) -> str:
    path = str(summary.get("death_path", "unknown"))
    if path == "movement_energy_depletion_after_carrion_contact":
        return (
            "agent consumed animal resource, then continued spending movement "
            "energy until energy-depletion death"
        )
    if path == "low_gain_eat_energy_depletion_after_carrion_contact":
        return (
            "agent kept selecting eat after contact but logged too little "
            "animal-resource gain to sustain energy"
        )
    if path == "energy_depletion_after_carrion_contact":
        return "post-contact energy balance stayed negative through terminal state"
    if path == "no_drink_hydration_depletion_after_carrion_contact":
        return "agent did not drink after animal-resource contact before hydration death"
    if path == "hydration_depletion_after_carrion_contact":
        return "post-contact hydration balance stayed negative through terminal state"
    if path == "health_loss_after_carrion_contact":
        return "post-contact health damage, hazard, or combat dominated terminal failure"
    if path.startswith("survived_window_but_"):
        return "agent survived the contact window but remained below a terminal viability floor"
    if path == "survived_contact_window":
        return "agent survived the configured post-contact window"
    return "post-contact terminal path needs manual review"


def _terminal_requested_action(summary: Mapping[str, object]) -> str:
    excerpt = summary.get("pre_terminal_excerpt")
    if not isinstance(excerpt, list) or not excerpt:
        return ""
    last = excerpt[-1]
    if not isinstance(last, Mapping):
        return ""
    action = last.get("requested_action")
    return str(action) if isinstance(action, str) else ""


def _event_excerpt(record: Mapping[str, object]) -> dict[str, object]:
    outcome = _outcome(record)
    feeding = _feeding(record)
    drinking = outcome.get("drinking")
    movement = outcome.get("movement")
    passive = outcome.get("passive")
    return {
        "tick": _record_tick(record),
        "source_record_index": _record_index(record),
        "requested_action": str(record.get("requested_action", "")),
        "resolved_action": str(record.get("resolved_action", "")),
        "before": _state_payload(record, "before"),
        "after": _state_payload(record, "after"),
        "feeding": {
            "ate": bool(feeding.get("ate", False)),
            "food_source": (
                str(feeding["food_source"])
                if isinstance(feeding.get("food_source"), str)
                else None
            ),
            "gained_energy": _round(_finite_float(feeding.get("gained_energy"))),
        },
        "drank": bool(drinking.get("drank", False)) if isinstance(drinking, Mapping) else False,
        "moved": bool(movement.get("moved", False)) if isinstance(movement, Mapping) else False,
        "reproduced": bool(outcome.get("reproduced", False)),
        "died": _record_dead_after(record),
        "death_cause": (
            passive.get("death_cause")
            if isinstance(passive, Mapping)
            and isinstance(passive.get("death_cause"), str)
            else None
        ),
    }


def _sequence_excerpt(
    records: Sequence[Mapping[str, object]],
    *,
    limit: int,
) -> list[dict[str, object]]:
    return [_event_excerpt(record) for record in records[:limit]]


def _state_payload(record: Mapping[str, object], key: str) -> dict[str, object]:
    state = record.get(key)
    state_mapping = state if isinstance(state, Mapping) else {}
    return {
        "energy_ratio": _round_optional(_mapping_float(state_mapping, "energy_ratio")),
        "hydration_ratio": _round_optional(
            _mapping_float(state_mapping, "hydration_ratio")
        ),
        "health_ratio": _round_optional(_mapping_float(state_mapping, "health_ratio")),
        "alive": bool(state_mapping.get("alive", True)),
        "x": _optional_int(state_mapping.get("x")),
        "y": _optional_int(state_mapping.get("y")),
    }


def _minimum_state(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "energy_ratio": _round_optional(
            _min_optional(_mapping_float(_state(record, "after"), "energy_ratio") for record in records)
        ),
        "hydration_ratio": _round_optional(
            _min_optional(
                _mapping_float(_state(record, "after"), "hydration_ratio")
                for record in records
            )
        ),
        "health_ratio": _round_optional(
            _min_optional(_mapping_float(_state(record, "after"), "health_ratio") for record in records)
        ),
    }


def _terminal_bottleneck(
    terminal_state: Mapping[str, object],
    *,
    ratio_floor: float,
) -> str:
    values = {
        "energy": _mapping_float(terminal_state, "energy_ratio"),
        "hydration": _mapping_float(terminal_state, "hydration_ratio"),
        "health": _mapping_float(terminal_state, "health_ratio"),
    }
    below = {
        name: value
        for name, value in values.items()
        if value is not None and value < ratio_floor
    }
    if not below:
        return "none"
    return min(below.items(), key=lambda item: item[1])[0]


def _movement_action_count(records: Sequence[Mapping[str, object]]) -> int:
    return sum(
        1 for record in records if str(record.get("requested_action", "")).startswith("move_")
    )


def _drink_count(records: Sequence[Mapping[str, object]]) -> int:
    count = 0
    for record in records:
        drinking = _outcome(record).get("drinking")
        if isinstance(drinking, Mapping) and drinking.get("drank") is True:
            count += 1
    return count


def _animal_resource_event_count(records: Sequence[Mapping[str, object]]) -> int:
    return sum(
        1
        for record in records
        if _feeding(record).get("food_source") in ANIMAL_RESOURCE_FOOD_SOURCES
        and _feeding(record).get("ate") is not False
    )


def _animal_resource_gain(records: Sequence[Mapping[str, object]]) -> float:
    total = 0.0
    for record in records:
        feeding = _feeding(record)
        if feeding.get("food_source") not in ANIMAL_RESOURCE_FOOD_SOURCES:
            continue
        total += _finite_float(feeding.get("gained_energy"))
    return total


def _low_gain_eat_count(records: Sequence[Mapping[str, object]]) -> int:
    count = 0
    for record in records:
        if str(record.get("requested_action", "")) != "eat":
            continue
        feeding = _feeding(record)
        if _finite_float(feeding.get("gained_energy")) <= 0.01:
            count += 1
    return count


def _reproduction_count(records: Sequence[Mapping[str, object]]) -> int:
    return sum(1 for record in records if _outcome(record).get("reproduced") is True)


def _death_cause(record: Mapping[str, object] | None) -> str | None:
    if record is None:
        return None
    passive = _outcome(record).get("passive")
    if not isinstance(passive, Mapping):
        return None
    cause = passive.get("death_cause")
    return str(cause) if isinstance(cause, str) and cause else None


def _record_dead_after(record: Mapping[str, object]) -> bool:
    if _outcome(record).get("died") is True:
        return True
    return _state(record, "after").get("alive") is False


def _feeding(record: Mapping[str, object]) -> Mapping[str, object]:
    feeding = _outcome(record).get("feeding")
    return feeding if isinstance(feeding, Mapping) else {}


def _outcome(record: Mapping[str, object]) -> Mapping[str, object]:
    outcome = record.get("outcome")
    return outcome if isinstance(outcome, Mapping) else {}


def _state(record: Mapping[str, object], key: str) -> Mapping[str, object]:
    state = record.get(key)
    return state if isinstance(state, Mapping) else {}


def _record_tick(record: Mapping[str, object]) -> int:
    tick = record.get("tick")
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise CarrionAutopsyError("trajectory record tick must be an integer")
    return tick


def _record_agent_id(record: Mapping[str, object]) -> int:
    agent_id = record.get("agent_id")
    if isinstance(agent_id, bool) or not isinstance(agent_id, int):
        raise CarrionAutopsyError("trajectory record agent_id must be an integer")
    return agent_id


def _record_index(record: Mapping[str, object]) -> int:
    index = record.get("source_record_index")
    if isinstance(index, bool) or not isinstance(index, int):
        return 0
    return index


def _counter_mapping(value: object) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int):
            continue
        counter[str(key)] += int(count)
    return counter


def _dominant_counter_payload(
    counter: Counter[str],
    *,
    denominator: int,
) -> dict[str, object]:
    if not counter or denominator <= 0:
        return {
            "path": None,
            "count": 0,
            "share": 0.0,
        }
    path, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
    return {
        "path": path,
        "count": int(count),
        "share": _safe_rate(count, denominator),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _round(numerator / denominator)


def _mean(values: Iterable[int | float]) -> float:
    parsed = [float(value) for value in values]
    if not parsed:
        return 0.0
    return _round(sum(parsed) / len(parsed))


def _mean_optional(values: Iterable[float | None]) -> float | None:
    parsed = [value for value in values if value is not None]
    if not parsed:
        return None
    return _round(sum(parsed) / len(parsed))


def _min_optional(values: Iterable[float | None]) -> float | None:
    parsed = [value for value in values if value is not None]
    if not parsed:
        return None
    return min(parsed)


def _mapping_float(value: object, key: str) -> float | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    parsed = float(raw)
    if not math.isfinite(parsed):
        return None
    return parsed


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return _round(value)


def _round(value: float) -> float:
    return round(float(value), 4)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CarrionAutopsyError(f"{field} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CarrionAutopsyError(f"{field} must be a non-negative integer")
    return int(value)


def _ratio_floor(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CarrionAutopsyError("critical_ratio_floor must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise CarrionAutopsyError("critical_ratio_floor must be in [0.0, 1.0]")
    return parsed


def _safe_label(value: str) -> str:
    safe = [
        character.lower()
        if character.isalnum() or character == "_"
        else "_"
        for character in value
    ]
    label = "".join(safe).strip("_")
    return label or "unknown"


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
