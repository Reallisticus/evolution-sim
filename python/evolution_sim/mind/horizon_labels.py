from __future__ import annotations

import gzip
import json
import math
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TextIO

from evolution_sim.env.runtime.observations import (
    SELF_INPUT_FIELDS,
    decode_observation_input,
)
from evolution_sim.mind.dataset import (
    TRAJECTORY_EPISODE_ID_FIELD,
    TrajectoryJsonlDataset,
    combined_dataset_provenance,
    records_with_trajectory_context,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_HORIZON_LABEL_SCHEMA_VERSION = "mind_horizon_labels_v1"
MIND_HORIZON_LABEL_POLICY = "agent_future_trajectory_horizon_labels_v1"
DEFAULT_HORIZON_TICKS: tuple[int, ...] = (20, 40, 80, 120)
VIABILITY_RATIO_FLOOR = 0.5
ANIMAL_RESOURCE_FOOD_SOURCES = frozenset({"fresh_kill", "carcass"})
_MATCHED_DIET_INPUT_INDEX = SELF_INPUT_FIELDS.index("matched_diet_ratio")


class HorizonLabelError(ValueError):
    pass


def normalize_horizon_ticks(horizons: Iterable[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for horizon in horizons:
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise HorizonLabelError("horizon ticks must be integers")
        if horizon <= 0:
            raise HorizonLabelError("horizon ticks must be positive")
        if horizon in seen:
            continue
        seen.add(horizon)
        normalized.append(horizon)
    if not normalized:
        raise HorizonLabelError("at least one horizon tick is required")
    return tuple(sorted(normalized))


def parse_horizon_ticks(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise HorizonLabelError("horizons must be a comma-separated integer list")
    try:
        return normalize_horizon_ticks(int(part) for part in parts)
    except ValueError as exc:
        raise HorizonLabelError("horizons must be a comma-separated integer list") from exc


def build_horizon_label_report(
    datasets: Sequence[TrajectoryJsonlDataset],
    *,
    horizons: Iterable[int] = DEFAULT_HORIZON_TICKS,
) -> dict[str, object]:
    if not datasets:
        raise HorizonLabelError("at least one trajectory dataset is required")
    horizon_ticks = normalize_horizon_ticks(horizons)
    contextual_records = tuple(records_with_trajectory_context(datasets))
    labels = build_horizon_label_records(
        contextual_records,
        horizons=horizon_ticks,
    )
    aggregate = _aggregate_labels(labels, horizon_ticks=horizon_ticks)
    provenance = combined_dataset_provenance(datasets)
    contract = _label_contract(horizon_ticks)
    return {
        "schema_version": MIND_HORIZON_LABEL_SCHEMA_VERSION,
        "label_policy": MIND_HORIZON_LABEL_POLICY,
        "label_contract": contract,
        "provenance": {
            **provenance,
            "label_contract_digest": stable_payload_digest(contract),
        },
        "source": {
            "trajectory_count": len(datasets),
            "trajectory_paths": [str(dataset.path) for dataset in datasets],
            "record_count": len(contextual_records),
        },
        "aggregate": aggregate,
        "labels": labels,
    }


def build_horizon_label_records(
    records: Sequence[Mapping[str, object]],
    *,
    horizons: Iterable[int] = DEFAULT_HORIZON_TICKS,
) -> list[dict[str, object]]:
    horizon_ticks = normalize_horizon_ticks(horizons)
    labels: list[dict[str, object]] = []
    for episode_id, episode_records in _episode_records(records):
        labels.extend(
            _episode_horizon_labels(
                episode_id=episode_id,
                records=episode_records,
                horizons=horizon_ticks,
            )
        )
    return labels


def write_horizon_label_report(
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
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")


def _label_contract(horizons: tuple[int, ...]) -> dict[str, object]:
    return {
        "schema_version": MIND_HORIZON_LABEL_SCHEMA_VERSION,
        "policy": MIND_HORIZON_LABEL_POLICY,
        "horizon_ticks": list(horizons),
        "censoring_policy": (
            "horizon_observed_when_target_tick_seen_or_agent_terminal_before_target_v1"
        ),
        "survival_target": "alive_after_horizon_tick",
        "reproduction_target": "any_reproduction_event_from_source_through_horizon",
        "end_state_policy": "target_tick_before_state_or_terminal_after_state_v1",
        "matched_diet_source": "decoded_observation_input_when_available",
        "viability_ratio_floor": VIABILITY_RATIO_FLOOR,
        "animal_resource_food_sources": sorted(ANIMAL_RESOURCE_FOOD_SOURCES),
    }


def _episode_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, tuple[Mapping[str, object], ...]], ...]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    order: list[str] = []
    for index, record in enumerate(records):
        episode_id = record.get(TRAJECTORY_EPISODE_ID_FIELD)
        if episode_id is None:
            episode_id = "inferred-episode-0"
        if not isinstance(episode_id, str) or not episode_id:
            raise HorizonLabelError("trajectory episode id must be a non-empty string")
        contextual = dict(record)
        contextual["source_record_index"] = index
        if episode_id not in grouped:
            grouped[episode_id] = []
            order.append(episode_id)
        grouped[episode_id].append(contextual)
    return tuple((episode_id, tuple(grouped[episode_id])) for episode_id in order)


def _episode_horizon_labels(
    *,
    episode_id: str,
    records: Sequence[Mapping[str, object]],
    horizons: tuple[int, ...],
) -> list[dict[str, object]]:
    episode_max_tick = max((_record_tick(record) for record in records), default=0)
    timelines: dict[int, list[Mapping[str, object]]] = {}
    for record in records:
        timelines.setdefault(_record_agent_id(record), []).append(record)
    for timeline in timelines.values():
        timeline.sort(key=lambda record: (_record_tick(record), _record_index(record)))

    labels: list[dict[str, object]] = []
    for agent_id in sorted(timelines):
        timeline = timelines[agent_id]
        ticks = [_record_tick(record) for record in timeline]
        death_positions = [
            index
            for index, record in enumerate(timeline)
            if _record_dead_after(record)
        ]
        reproduction_prefix = _reproduction_prefix_counts(timeline)
        for position, record in enumerate(timeline):
            labels.append(
                _label_for_record(
                    episode_id=episode_id,
                    episode_max_tick=episode_max_tick,
                    timeline=timeline,
                    ticks=ticks,
                    death_positions=death_positions,
                    reproduction_prefix=reproduction_prefix,
                    position=position,
                    horizons=horizons,
                )
            )
    labels.sort(key=lambda label: int(label["source_record_index"]))
    return labels


def _label_for_record(
    *,
    episode_id: str,
    episode_max_tick: int,
    timeline: Sequence[Mapping[str, object]],
    ticks: Sequence[int],
    death_positions: Sequence[int],
    reproduction_prefix: Sequence[int],
    position: int,
    horizons: tuple[int, ...],
) -> dict[str, object]:
    record = timeline[position]
    tick = _record_tick(record)
    horizon_payload = {
        str(horizon): _horizon_payload(
            timeline=timeline,
            ticks=ticks,
            death_positions=death_positions,
            reproduction_prefix=reproduction_prefix,
            position=position,
            source_tick=tick,
            episode_max_tick=episode_max_tick,
            horizon=horizon,
        )
        for horizon in horizons
    }
    return {
        "schema_version": MIND_HORIZON_LABEL_SCHEMA_VERSION,
        "episode_id": episode_id,
        "source_record_index": _record_index(record),
        "tick": tick,
        "agent_id": _record_agent_id(record),
        "lineage_id": _optional_int(record.get("lineage_id")),
        "runtime_species_id": _optional_int(record.get("runtime_species_id")),
        "runtime_ecotype_id": _optional_int(record.get("runtime_ecotype_id")),
        "requested_action": str(record.get("requested_action", "")),
        "resolved_action": str(record.get("resolved_action", "")),
        "action_source": str(record.get("action_source", "")),
        "policy_id": _optional_string(record.get("policy_id")),
        "policy_version": _optional_string(record.get("policy_version")),
        "source_state": _source_state(record),
        "immediate": _immediate_payload(record),
        "horizons": horizon_payload,
    }


def _horizon_payload(
    *,
    timeline: Sequence[Mapping[str, object]],
    ticks: Sequence[int],
    death_positions: Sequence[int],
    reproduction_prefix: Sequence[int],
    position: int,
    source_tick: int,
    episode_max_tick: int,
    horizon: int,
) -> dict[str, object]:
    target_tick = source_tick + horizon
    target_position = bisect_left(ticks, target_tick, lo=position)
    target_seen = target_position < len(timeline)
    death_lookup_position = bisect_left(death_positions, position)
    death_position = (
        death_positions[death_lookup_position]
        if death_lookup_position < len(death_positions)
        else None
    )
    death_tick = (
        _record_tick(timeline[death_position])
        if death_position is not None
        else None
    )
    terminal_before_target = death_tick is not None and death_tick <= target_tick
    observed = target_seen or terminal_before_target
    if not observed:
        return {
            "target_tick": target_tick,
            "observed": False,
            "censored": True,
            "last_observed_tick": episode_max_tick,
            "survived": None,
            "reproduced": None,
            "reproduction_event_count": None,
            "alive_decision_count": None,
            "end_tick": None,
            "end_state": None,
            "viability": None,
            "animal_resource": None,
        }

    end_position = (
        death_position
        if terminal_before_target and death_position is not None
        else target_position
    )
    end_record = timeline[end_position]
    end_tick = _record_tick(end_record)
    survived = not terminal_before_target and bool(
        _state_mapping(end_record, "after").get("alive", True)
    )
    reproduction_count = (
        reproduction_prefix[end_position + 1] - reproduction_prefix[position]
    )
    animal_resource_payload = _animal_resource_window_payload(
        timeline=timeline,
        position=position,
        end_position=end_position,
        survived=survived,
    )
    end_state = _end_state(end_record, terminal=terminal_before_target)
    return {
        "target_tick": target_tick,
        "observed": True,
        "censored": False,
        "last_observed_tick": episode_max_tick,
        "survived": survived,
        "reproduced": reproduction_count > 0,
        "reproduction_event_count": reproduction_count,
        "alive_decision_count": _alive_decision_count(
            timeline,
            start=position,
            end=end_position,
        ),
        "end_tick": end_tick,
        "end_state": end_state,
        "viability": _viability_payload(end_state),
        "animal_resource": animal_resource_payload,
    }


def _source_state(record: Mapping[str, object]) -> dict[str, object]:
    before = _state_mapping(record, "before")
    return {
        "energy_ratio": _optional_ratio(before.get("energy_ratio")),
        "hydration_ratio": _optional_ratio(before.get("hydration_ratio")),
        "health_ratio": _optional_ratio(before.get("health_ratio")),
        "matched_diet_ratio": _matched_diet_ratio(record),
    }


def _end_state(record: Mapping[str, object], *, terminal: bool) -> dict[str, object]:
    state = _state_mapping(record, "after" if terminal else "before")
    return {
        "state_source": "terminal_after" if terminal else "target_before",
        "energy_ratio": _optional_ratio(state.get("energy_ratio")),
        "hydration_ratio": _optional_ratio(state.get("hydration_ratio")),
        "health_ratio": _optional_ratio(state.get("health_ratio")),
        "matched_diet_ratio": _matched_diet_ratio(record),
        "alive": bool(state.get("alive", True)),
    }


def _viability_payload(end_state: Mapping[str, object]) -> dict[str, object]:
    energy = _optional_ratio(end_state.get("energy_ratio"))
    hydration = _optional_ratio(end_state.get("hydration_ratio"))
    health = _optional_ratio(end_state.get("health_ratio"))
    matched_diet = _optional_ratio(end_state.get("matched_diet_ratio"))
    core_values = [
        value for value in (energy, hydration, health) if value is not None
    ]
    balanced_core_min = min(core_values) if core_values else None
    return {
        "ratio_floor": VIABILITY_RATIO_FLOOR,
        "energy": _floor_label(energy),
        "hydration": _floor_label(hydration),
        "health": _floor_label(health),
        "matched_diet": _floor_label(matched_diet),
        "balanced_core_min": _round_optional(balanced_core_min),
    }


def _immediate_payload(record: Mapping[str, object]) -> dict[str, object]:
    outcome = _outcome_mapping(record)
    feeding = outcome.get("feeding")
    feeding_payload = feeding if isinstance(feeding, Mapping) else {}
    food_source = feeding_payload.get("food_source")
    animal_resource = food_source in ANIMAL_RESOURCE_FOOD_SOURCES
    passive = outcome.get("passive")
    passive_payload = passive if isinstance(passive, Mapping) else {}
    return {
        "reproduced": bool(outcome.get("reproduced", False)),
        "died": _record_dead_after(record),
        "resource_gain": _finite_float(outcome.get("resource_gain"), default=0.0),
        "ate": bool(feeding_payload.get("ate", False)),
        "food_source": str(food_source) if isinstance(food_source, str) else None,
        "animal_resource_consumed": animal_resource,
        "hazard_damage_taken": _finite_float(
            passive_payload.get("hazard_damage_taken"),
            default=0.0,
        ),
        "attack_damage_taken": _finite_float(
            passive_payload.get("attack_damage_taken"),
            default=0.0,
        ),
    }


def _animal_resource_window_payload(
    *,
    timeline: Sequence[Mapping[str, object]],
    position: int,
    end_position: int,
    survived: bool,
) -> dict[str, object]:
    food_counts: Counter[str] = Counter()
    first_contact_tick: int | None = None
    gained_energy = 0.0
    for record in timeline[position : end_position + 1]:
        outcome = _outcome_mapping(record)
        feeding = outcome.get("feeding")
        if not isinstance(feeding, Mapping):
            continue
        food_source = feeding.get("food_source")
        if food_source not in ANIMAL_RESOURCE_FOOD_SOURCES:
            continue
        food_source_text = str(food_source)
        food_counts[food_source_text] += 1
        gained_energy += _finite_float(feeding.get("gained_energy"), default=0.0)
        if first_contact_tick is None:
            first_contact_tick = _record_tick(record)
    consumed = sum(food_counts.values()) > 0
    return {
        "animal_resource_consumed": consumed,
        "first_contact_tick": first_contact_tick,
        "fresh_kill_events": int(food_counts["fresh_kill"]),
        "carcass_events": int(food_counts["carcass"]),
        "gained_energy": _round(gained_energy),
        "survived_after_first_contact": survived if consumed else None,
    }


def _aggregate_labels(
    labels: Sequence[Mapping[str, object]],
    *,
    horizon_ticks: tuple[int, ...],
) -> dict[str, object]:
    by_horizon: dict[str, object] = {}
    for horizon in horizon_ticks:
        key = str(horizon)
        payloads = [
            label["horizons"][key]  # type: ignore[index]
            for label in labels
            if isinstance(label.get("horizons"), Mapping)
        ]
        observed = [
            payload
            for payload in payloads
            if isinstance(payload, Mapping) and bool(payload.get("observed", False))
        ]
        survived = [payload for payload in observed if payload.get("survived") is True]
        reproduced = [
            payload for payload in observed if payload.get("reproduced") is True
        ]
        animal_payloads = [
            payload.get("animal_resource")
            for payload in observed
            if isinstance(payload.get("animal_resource"), Mapping)
        ]
        animal_contacts = [
            payload
            for payload in animal_payloads
            if isinstance(payload, Mapping)
            and bool(payload.get("animal_resource_consumed", False))
        ]
        post_contact_survived = [
            payload
            for payload in animal_contacts
            if payload.get("survived_after_first_contact") is True
        ]
        viability_payloads = [
            payload.get("viability")
            for payload in observed
            if isinstance(payload.get("viability"), Mapping)
        ]
        by_horizon[key] = {
            "record_count": len(payloads),
            "observed_count": len(observed),
            "censored_count": len(payloads) - len(observed),
            "survival_rate": _safe_rate(len(survived), len(observed)),
            "reproduction_rate": _safe_rate(len(reproduced), len(observed)),
            "animal_resource_contact_count": len(animal_contacts),
            "post_contact_survival_rate": _safe_rate(
                len(post_contact_survived),
                len(animal_contacts),
            ),
            "mean_balanced_core_min": _mean_optional(
                _mapping_float(viability, "balanced_core_min")
                for viability in viability_payloads
                if isinstance(viability, Mapping)
            ),
        }
    return {
        "label_count": len(labels),
        "horizons": by_horizon,
    }


def _reproduction_prefix_counts(
    timeline: Sequence[Mapping[str, object]],
) -> list[int]:
    prefix = [0]
    total = 0
    for record in timeline:
        if bool(_outcome_mapping(record).get("reproduced", False)):
            total += 1
        prefix.append(total)
    return prefix


def _alive_decision_count(
    timeline: Sequence[Mapping[str, object]],
    *,
    start: int,
    end: int,
) -> int:
    count = 0
    for record in timeline[start : end + 1]:
        before = _state_mapping(record, "before")
        if bool(before.get("alive", True)):
            count += 1
    return count


def _matched_diet_ratio(record: Mapping[str, object]) -> float | None:
    observation_input = record.get("observation_input")
    if not isinstance(observation_input, dict):
        return None
    try:
        values = decode_observation_input(observation_input)
    except ValueError:
        return None
    if _MATCHED_DIET_INPUT_INDEX >= len(values):
        return None
    return _optional_ratio(values[_MATCHED_DIET_INPUT_INDEX])


def _floor_label(value: float | None) -> bool | None:
    if value is None:
        return None
    return value >= VIABILITY_RATIO_FLOOR


def _record_tick(record: Mapping[str, object]) -> int:
    tick = record.get("tick")
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise HorizonLabelError("trajectory record tick must be an integer")
    return tick


def _record_agent_id(record: Mapping[str, object]) -> int:
    agent_id = record.get("agent_id")
    if isinstance(agent_id, bool) or not isinstance(agent_id, int):
        raise HorizonLabelError("trajectory record agent_id must be an integer")
    return agent_id


def _record_index(record: Mapping[str, object]) -> int:
    index = record.get("source_record_index")
    if isinstance(index, bool) or not isinstance(index, int):
        return 0
    return index


def _record_dead_after(record: Mapping[str, object]) -> bool:
    outcome = _outcome_mapping(record)
    if bool(outcome.get("died", False)):
        return True
    after = _state_mapping(record, "after")
    return after.get("alive") is False


def _outcome_mapping(record: Mapping[str, object]) -> Mapping[str, object]:
    outcome = record.get("outcome")
    return outcome if isinstance(outcome, Mapping) else {}


def _state_mapping(record: Mapping[str, object], key: str) -> Mapping[str, object]:
    state = record.get(key)
    return state if isinstance(state, Mapping) else {}


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value


def _optional_ratio(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return _round(min(1.0, max(0.0, parsed)))


def _finite_float(value: object, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    parsed = float(value)
    if not math.isfinite(parsed):
        return default
    return parsed


def _mapping_float(payload: object, key: str) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _round(numerator / denominator)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    parsed = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not parsed:
        return None
    return _round(sum(parsed) / len(parsed))


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return _round(value)


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
