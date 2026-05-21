from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TextIO

from evolution_sim.env.runtime.observations import (
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_RADIUS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
)
from evolution_sim.env.runtime.trajectory import TRAJECTORY_RECORD_FIELDS
from evolution_sim.mind.dataset import (
    TRAJECTORY_JSONL_FORMAT,
    TRAJECTORY_DATASET_INDEX_FIELD,
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
MIND_V3_CARRION_FIXTURE_TRACE_SCHEMA_VERSION = (
    "mind_v3_carrion_fixture_trace_v1"
)
MIND_V3_CARRION_FIXTURE_TRACE_POLICY = (
    "offline_trajectory_only_carrion_fixture_trace_v1"
)
DEFAULT_POST_CONTACT_WINDOW_TICKS = 120
DEFAULT_SEQUENCE_RECORD_LIMIT = 12
DEFAULT_MAX_EXAMPLES = 8
DEFAULT_CRITICAL_RATIO_FLOOR = 0.5
MOVEMENT_ACTIONS: tuple[str, ...] = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
)


class CarrionAutopsyError(ValueError):
    pass


def load_carrion_autopsy_trajectory_jsonl(
    path: str | Path,
) -> TrajectoryJsonlDataset:
    """Load trajectory JSONL for offline autopsy without scalar-only diagnostics.

    The general Mind dataset loader keeps training-facing policy diagnostics
    scalar-only. Carrion autopsy is report-only and must tolerate nested v5
    rollout-context diagnostics already present in generated trajectory files.
    """
    resolved_path = Path(path)
    payloads = list(_read_autopsy_jsonl_payloads(resolved_path))
    if len(payloads) < 2:
        raise CarrionAutopsyError("trajectory JSONL must include header and footer")
    header = payloads[0]
    footer = payloads[-1]
    records = payloads[1:-1]
    if header.get("type") != "header":
        raise CarrionAutopsyError("first trajectory JSONL row must be a header")
    if header.get("format") != TRAJECTORY_JSONL_FORMAT:
        raise CarrionAutopsyError("trajectory JSONL header format is unsupported")
    if footer.get("type") != "footer":
        raise CarrionAutopsyError("last trajectory JSONL row must be a footer")
    parsed_records = [
        _autopsy_record_payload(payload, index)
        for index, payload in enumerate(records)
    ]
    return TrajectoryJsonlDataset(
        path=resolved_path,
        header=dict(header),
        records=tuple(parsed_records),
        footer=dict(footer),
    )


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
    contextual_records = _contextual_records_with_source_seed(datasets)
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
        "fixture_trace": _build_fixture_trace(
            contextual_records,
            window_ticks=window_ticks,
        ),
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
        "fixture_trace_schema_version": MIND_V3_CARRION_FIXTURE_TRACE_SCHEMA_VERSION,
        "fixture_trace_policy": MIND_V3_CARRION_FIXTURE_TRACE_POLICY,
    }


def _contextual_records_with_source_seed(
    datasets: Sequence[TrajectoryJsonlDataset],
) -> list[dict[str, object]]:
    seed_by_dataset_index = {
        index: _dataset_source_seed(dataset)
        for index, dataset in enumerate(datasets)
    }
    contextual_records: list[dict[str, object]] = []
    for index, record in enumerate(records_with_trajectory_context(datasets)):
        contextual = {**record, "source_record_index": index}
        dataset_index = _optional_int(contextual.get(TRAJECTORY_DATASET_INDEX_FIELD))
        source_seed = seed_by_dataset_index.get(dataset_index)
        if source_seed is not None:
            contextual["source_seed"] = source_seed
        contextual_records.append(contextual)
    return contextual_records


def _dataset_source_seed(dataset: TrajectoryJsonlDataset) -> int | None:
    provenance = dataset.header.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    seeds = provenance.get("source_seeds")
    if not isinstance(seeds, list) or len(seeds) != 1:
        return None
    seed = seeds[0]
    if isinstance(seed, bool) or not isinstance(seed, int):
        return None
    return int(seed)


def _read_autopsy_jsonl_payloads(path: Path) -> Iterable[dict[str, object]]:
    if not path.exists():
        raise CarrionAutopsyError(f"trajectory path does not exist: {path}")
    with _open_input(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(
                    stripped,
                    parse_constant=_reject_json_constant,
                )
            except json.JSONDecodeError as exc:
                raise CarrionAutopsyError(
                    f"line {line_number} is not valid JSON: {exc.msg}"
                ) from exc
            except ValueError as exc:
                raise CarrionAutopsyError(
                    f"line {line_number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise CarrionAutopsyError(
                    f"line {line_number} must be a JSON object"
                )
            yield payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not supported")


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _autopsy_record_payload(
    payload: dict[str, object],
    index: int,
) -> dict[str, object]:
    if payload.get("type") != "record":
        raise CarrionAutopsyError(f"trajectory row {index + 2} must be a record")
    record = payload.get("record")
    if not isinstance(record, dict):
        raise CarrionAutopsyError(f"trajectory record {index} must be an object")
    missing = [field for field in TRAJECTORY_RECORD_FIELDS if field not in record]
    if missing:
        raise CarrionAutopsyError(
            f"trajectory record {index} is missing fields: {missing}"
        )
    return dict(record)


def _build_fixture_trace(
    records: Sequence[Mapping[str, object]],
    *,
    window_ticks: int,
) -> dict[str, object]:
    agent_traces = [
        _summarize_fixture_agent_trace(
            episode_id=episode_id,
            agent_id=agent_id,
            timeline=timeline,
            window_ticks=window_ticks,
        )
        for (episode_id, agent_id), timeline in _group_agent_timelines(records).items()
    ]
    agent_traces.sort(
        key=lambda trace: (
            str(trace["seed"]),
            str(trace["episode_id"]),
            int(trace["agent_id"]),
        )
    )
    return {
        "schema_version": MIND_V3_CARRION_FIXTURE_TRACE_SCHEMA_VERSION,
        "trace_policy": MIND_V3_CARRION_FIXTURE_TRACE_POLICY,
        "offline_only": True,
        "record_count": len(records),
        "post_contact_window_ticks": int(window_ticks),
        "aggregate": _aggregate_fixture_agent_traces(agent_traces),
        "by_seed": _fixture_trace_by_seed(agent_traces),
        "by_agent": {
            _fixture_agent_key(trace): trace
            for trace in agent_traces
        },
    }


def _summarize_fixture_agent_trace(
    *,
    episode_id: str,
    agent_id: int,
    timeline: Sequence[Mapping[str, object]],
    window_ticks: int,
) -> dict[str, object]:
    requested_counts: Counter[str] = Counter()
    resolved_counts: Counter[str] = Counter()
    food_source_counts: Counter[str] = Counter()
    death_cause_counts: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    unsupported_requested = _empty_unsupported_breakdown()
    unsupported_resolved = _empty_unsupported_breakdown()
    action_mask_availability = _empty_mask_availability()
    resolution_mask_availability = _empty_mask_availability()
    navigation_accumulator = _empty_navigation_accumulator()
    rollout_context = _empty_rollout_context_accumulator()

    eat_attempt_count = 0
    successful_eat_count = 0
    animal_resource_successful_eat_count = 0
    resource_gain_total = 0.0
    drink_request_count = 0
    successful_drink_count = 0
    no_gain_eat_count = 0
    no_gain_eat_streak = 0
    max_no_gain_eat_streak = 0
    death_tick: int | None = None
    death_cause: str | None = None

    for record in timeline:
        requested = str(record.get("requested_action", "unknown"))
        resolved = str(record.get("resolved_action", "unknown"))
        requested_counts.update([requested])
        resolved_counts.update([resolved])
        if requested == "eat":
            eat_attempt_count += 1
        if requested == "drink":
            drink_request_count += 1

        feeding = _feeding(record)
        gained_energy = _finite_float(feeding.get("gained_energy"))
        if feeding.get("ate") is True:
            successful_eat_count += 1
            food_source = str(feeding.get("food_source") or "unknown")
            food_source_counts.update([food_source])
            resource_gain_total += gained_energy
            if food_source in ANIMAL_RESOURCE_FOOD_SOURCES:
                animal_resource_successful_eat_count += 1
        if requested == "eat" and gained_energy <= 0.01:
            no_gain_eat_count += 1
            no_gain_eat_streak += 1
            max_no_gain_eat_streak = max(max_no_gain_eat_streak, no_gain_eat_streak)
        else:
            no_gain_eat_streak = 0

        drinking = _outcome(record).get("drinking")
        if isinstance(drinking, Mapping) and drinking.get("drank") is True:
            successful_drink_count += 1

        _accumulate_mask_availability(
            action_mask_availability,
            record.get("action_mask"),
            missing_fields=missing_fields,
            missing_field_name="action_mask",
        )
        _accumulate_mask_availability(
            resolution_mask_availability,
            record.get("resolution_action_mask"),
            missing_fields=missing_fields,
            missing_field_name="resolution_action_mask",
        )
        _accumulate_validity_breakdown(
            unsupported_requested,
            record=record,
            validity_key="action_valid",
            missing_fields=missing_fields,
        )
        _accumulate_validity_breakdown(
            unsupported_resolved,
            record=record,
            validity_key="resolution_action_valid",
            missing_fields=missing_fields,
        )
        _accumulate_rollout_context_diagnostics(
            rollout_context,
            record=record,
            missing_fields=missing_fields,
        )
        _accumulate_navigation_summary(
            navigation_accumulator,
            record=record,
            missing_fields=missing_fields,
        )
        if "before" not in record or not isinstance(record.get("before"), Mapping):
            missing_fields.update(["before"])
        if "after" not in record or not isinstance(record.get("after"), Mapping):
            missing_fields.update(["after"])
        if "outcome" not in record or not isinstance(record.get("outcome"), Mapping):
            missing_fields.update(["outcome"])

        if death_tick is None and _record_dead_after(record):
            death_tick = _record_tick(record)
            death_cause = _death_cause(record) or "unknown"
            death_cause_counts.update([death_cause])

    contact_index = _first_animal_resource_contact_index(timeline)
    contact_summary = _contact_window_summary(
        timeline=timeline,
        contact_index=contact_index,
        window_ticks=window_ticks,
    )
    seed = _record_seed(timeline[0]) if timeline else "unknown"
    return {
        "seed": seed,
        "episode_id": episode_id,
        "agent_id": int(agent_id),
        "record_count": len(timeline),
        "first_tick": _record_tick(timeline[0]) if timeline else None,
        "last_tick": _record_tick(timeline[-1]) if timeline else None,
        "requested_action_counts": dict(sorted(requested_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_counts.items())),
        "unsupported_requested_action_count": int(unsupported_requested["count"]),
        "unsupported_requested_breakdown": _finalize_unsupported_breakdown(
            unsupported_requested,
            seed=seed,
        ),
        "unsupported_resolved_action_count": int(unsupported_resolved["count"]),
        "unsupported_resolved_breakdown": _finalize_unsupported_breakdown(
            unsupported_resolved,
            seed=seed,
        ),
        "action_mask_availability": _finalize_mask_availability(
            action_mask_availability
        ),
        "resolution_mask_availability": _finalize_mask_availability(
            resolution_mask_availability
        ),
        "eat_attempt_count": int(eat_attempt_count),
        "successful_eat_count": int(successful_eat_count),
        "animal_resource_successful_eat_count": int(
            animal_resource_successful_eat_count
        ),
        "food_source_counts": dict(sorted(food_source_counts.items())),
        "resource_gain_total": _round(resource_gain_total),
        "drink_request_count": int(drink_request_count),
        "successful_drink_count": int(successful_drink_count),
        "death_tick": death_tick,
        "death_cause": death_cause,
        "death_cause_counts": dict(sorted(death_cause_counts.items())),
        "no_gain_eat_count": int(no_gain_eat_count),
        "max_no_gain_eat_streak": int(max_no_gain_eat_streak),
        "rollout_context_diagnostics": _finalize_rollout_context_summary(
            rollout_context
        ),
        "navigation_target_observations": _finalize_navigation_summary(
            navigation_accumulator
        ),
        "missing_field_counts": dict(sorted(missing_fields.items())),
        "first_animal_resource_contact": contact_summary,
    }


def _contact_window_summary(
    *,
    timeline: Sequence[Mapping[str, object]],
    contact_index: int | None,
    window_ticks: int,
) -> dict[str, object]:
    if contact_index is None:
        return {
            "present": False,
            "first_contact_tick": None,
            "window_observed_tick": None,
            "window_observed_tick_delta": None,
            "state_at_contact": {},
            "state_after_window": {},
            "energy_delta_after_carrion": None,
            "hydration_delta_after_carrion": None,
            "health_delta_after_carrion": None,
            "drink_after_carrion_count": 0,
            "drink_after_carrion": False,
            "survived_after_carrion": None,
            "death_tick": None,
            "death_ticks_after_carrion": None,
            "death_cause": None,
            "navigation_at_contact": {},
            "post_carrion_rollout_context": None,
            "rollout_context_selected_score_delta": None,
        }
    first_contact = timeline[contact_index]
    contact_tick = _record_tick(first_contact)
    max_tick = contact_tick + int(window_ticks)
    window_records = [
        record
        for record in timeline[contact_index:]
        if _record_tick(record) <= max_tick
    ] or [first_contact]
    after_contact_records = [
        record for record in window_records if _record_tick(record) > contact_tick
    ]
    terminal = window_records[-1]
    death_record = next(
        (record for record in window_records if _record_dead_after(record)),
        None,
    )
    contact_state = _state_payload(first_contact, "after")
    window_state = _state_payload(terminal, "after")
    diagnostics = first_contact.get("policy_decision_diagnostics")
    diagnostics_mapping = diagnostics if isinstance(diagnostics, Mapping) else {}
    return {
        "present": True,
        "first_contact_tick": contact_tick,
        "window_observed_tick": _record_tick(terminal),
        "window_observed_tick_delta": _record_tick(terminal) - contact_tick,
        "state_at_contact": contact_state,
        "state_after_window": window_state,
        "energy_delta_after_carrion": _state_delta(
            contact_state,
            window_state,
            "energy_ratio",
        ),
        "hydration_delta_after_carrion": _state_delta(
            contact_state,
            window_state,
            "hydration_ratio",
        ),
        "health_delta_after_carrion": _state_delta(
            contact_state,
            window_state,
            "health_ratio",
        ),
        "drink_after_carrion_count": _drink_count(after_contact_records),
        "drink_after_carrion": _drink_count(after_contact_records) > 0,
        "survived_after_carrion": death_record is None,
        "death_tick": _record_tick(death_record) if death_record is not None else None,
        "death_ticks_after_carrion": (
            _record_tick(death_record) - contact_tick
            if death_record is not None
            else None
        ),
        "death_cause": _death_cause(death_record) if death_record is not None else None,
        "navigation_at_contact": _navigation_payload_from_record(first_contact),
        "post_carrion_rollout_context": (
            bool(diagnostics_mapping.get("rollout_context_post_carrion_context"))
            if diagnostics_mapping
            else None
        ),
        "rollout_context_selected_score_delta": _round_optional(
            _mapping_float(
                diagnostics_mapping,
                "rollout_context_selected_score_delta",
            )
        ),
    }


def _aggregate_fixture_agent_traces(
    traces: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return _aggregate_fixture_trace_subset(traces)


def _fixture_trace_by_seed(
    traces: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    for trace in traces:
        by_seed.setdefault(str(trace.get("seed", "unknown")), []).append(trace)
    return {
        seed: _aggregate_fixture_trace_subset(seed_traces)
        for seed, seed_traces in sorted(by_seed.items())
    }


def _aggregate_fixture_trace_subset(
    traces: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    requested_counts: Counter[str] = Counter()
    resolved_counts: Counter[str] = Counter()
    food_source_counts: Counter[str] = Counter()
    death_cause_counts: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    unsupported_requested = _empty_unsupported_breakdown()
    unsupported_resolved = _empty_unsupported_breakdown()
    action_mask = _empty_mask_availability()
    resolution_mask = _empty_mask_availability()
    navigation = _empty_navigation_accumulator()
    rollout = _empty_rollout_context_accumulator()
    contact_summaries: list[Mapping[str, object]] = []

    for trace in traces:
        requested_counts.update(_counter_mapping(trace.get("requested_action_counts")))
        resolved_counts.update(_counter_mapping(trace.get("resolved_action_counts")))
        food_source_counts.update(_counter_mapping(trace.get("food_source_counts")))
        death_cause_counts.update(_counter_mapping(trace.get("death_cause_counts")))
        missing_fields.update(_counter_mapping(trace.get("missing_field_counts")))
        _merge_unsupported_breakdown(
            unsupported_requested,
            trace.get("unsupported_requested_breakdown"),
        )
        _merge_unsupported_breakdown(
            unsupported_resolved,
            trace.get("unsupported_resolved_breakdown"),
        )
        _merge_mask_availability(action_mask, trace.get("action_mask_availability"))
        _merge_mask_availability(
            resolution_mask,
            trace.get("resolution_mask_availability"),
        )
        _merge_rollout_context_summary(
            rollout,
            trace.get("rollout_context_diagnostics"),
        )
        _merge_navigation_summary(
            navigation,
            trace.get("navigation_target_observations"),
        )
        contact = trace.get("first_animal_resource_contact")
        if isinstance(contact, Mapping) and contact.get("present") is True:
            contact_summaries.append(contact)

    contact_count = len(contact_summaries)
    survived_count = sum(
        1 for contact in contact_summaries if contact.get("survived_after_carrion") is True
    )
    drink_after_count = sum(
        1 for contact in contact_summaries if contact.get("drink_after_carrion") is True
    )
    return {
        "agent_count": len(traces),
        "record_count": sum(int(trace.get("record_count", 0)) for trace in traces),
        "contact_agent_count": contact_count,
        "requested_action_counts": dict(sorted(requested_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_counts.items())),
        "unsupported_requested_action_count": int(unsupported_requested["count"]),
        "unsupported_requested_breakdown": _finalize_unsupported_breakdown(
            unsupported_requested,
            seed=None,
        ),
        "unsupported_resolved_action_count": int(unsupported_resolved["count"]),
        "unsupported_resolved_breakdown": _finalize_unsupported_breakdown(
            unsupported_resolved,
            seed=None,
        ),
        "action_mask_availability": _finalize_mask_availability(action_mask),
        "resolution_mask_availability": _finalize_mask_availability(resolution_mask),
        "eat_attempt_count": sum(int(trace.get("eat_attempt_count", 0)) for trace in traces),
        "successful_eat_count": sum(
            int(trace.get("successful_eat_count", 0)) for trace in traces
        ),
        "animal_resource_successful_eat_count": sum(
            int(trace.get("animal_resource_successful_eat_count", 0))
            for trace in traces
        ),
        "food_source_counts": dict(sorted(food_source_counts.items())),
        "resource_gain_total": _round(
            sum(float(trace.get("resource_gain_total", 0.0)) for trace in traces)
        ),
        "drink_request_count": sum(
            int(trace.get("drink_request_count", 0)) for trace in traces
        ),
        "successful_drink_count": sum(
            int(trace.get("successful_drink_count", 0)) for trace in traces
        ),
        "death_cause_counts": dict(sorted(death_cause_counts.items())),
        "no_gain_eat_count": sum(
            int(trace.get("no_gain_eat_count", 0)) for trace in traces
        ),
        "max_no_gain_eat_streak": max(
            [int(trace.get("max_no_gain_eat_streak", 0)) for trace in traces]
            or [0]
        ),
        "drink_after_carrion_rate": _safe_rate(drink_after_count, contact_count),
        "survival_after_carrion_rate": _safe_rate(survived_count, contact_count),
        "mean_energy_delta_after_carrion": _mean_optional(
            _mapping_float(contact, "energy_delta_after_carrion")
            for contact in contact_summaries
        ),
        "mean_hydration_delta_after_carrion": _mean_optional(
            _mapping_float(contact, "hydration_delta_after_carrion")
            for contact in contact_summaries
        ),
        "mean_health_delta_after_carrion": _mean_optional(
            _mapping_float(contact, "health_delta_after_carrion")
            for contact in contact_summaries
        ),
        "death_ticks_after_carrion": [
            int(contact["death_ticks_after_carrion"])
            for contact in contact_summaries
            if isinstance(contact.get("death_ticks_after_carrion"), int)
        ],
        "rollout_context_diagnostics": _finalize_rollout_context_summary(rollout),
        "navigation_target_observations": _finalize_navigation_summary(navigation),
        "missing_field_counts": dict(sorted(missing_fields.items())),
    }


def _fixture_agent_key(trace: Mapping[str, object]) -> str:
    return (
        f"seed-{trace.get('seed', 'unknown')}:"
        f"{trace.get('episode_id', 'unknown')}:"
        f"agent-{trace.get('agent_id', 'unknown')}"
    )


def _empty_mask_availability() -> dict[str, Counter[str]]:
    return {
        "eat": Counter(),
        "drink": Counter(),
        "movement": Counter(),
    }


def _accumulate_mask_availability(
    accumulator: dict[str, Counter[str]],
    raw_mask: object,
    *,
    missing_fields: Counter[str],
    missing_field_name: str,
) -> None:
    if not isinstance(raw_mask, Mapping):
        missing_fields.update([missing_field_name])
        for counter in accumulator.values():
            counter.update(["missing"])
        return
    _update_binary_availability(accumulator["eat"], raw_mask.get("eat"))
    _update_binary_availability(accumulator["drink"], raw_mask.get("drink"))
    movement_values = [
        raw_mask.get(action)
        for action in MOVEMENT_ACTIONS
        if action in raw_mask
    ]
    if not movement_values:
        accumulator["movement"].update(["missing"])
    elif any(value is True for value in movement_values):
        accumulator["movement"].update(["available"])
    else:
        accumulator["movement"].update(["unavailable"])


def _update_binary_availability(counter: Counter[str], value: object) -> None:
    if value is True:
        counter.update(["available"])
    elif value is False:
        counter.update(["unavailable"])
    else:
        counter.update(["missing"])


def _finalize_mask_availability(
    accumulator: Mapping[str, Counter[str]],
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for key in ("eat", "drink", "movement"):
        counter = accumulator.get(key, Counter())
        total = sum(counter.values())
        payload[key] = {
            "available": int(counter.get("available", 0)),
            "unavailable": int(counter.get("unavailable", 0)),
            "missing": int(counter.get("missing", 0)),
            "available_share": _safe_rate(int(counter.get("available", 0)), total),
        }
    return payload


def _merge_mask_availability(
    accumulator: dict[str, Counter[str]],
    payload: object,
) -> None:
    if not isinstance(payload, Mapping):
        return
    for key in ("eat", "drink", "movement"):
        raw = payload.get(key)
        if not isinstance(raw, Mapping):
            continue
        for bucket in ("available", "unavailable", "missing"):
            value = raw.get(bucket)
            if isinstance(value, int) and not isinstance(value, bool):
                accumulator[key][bucket] += int(value)


def _empty_unsupported_breakdown() -> dict[str, object]:
    return {
        "count": 0,
        "by_requested_action": Counter(),
        "by_resolved_action": Counter(),
        "by_invalid_reason": Counter(),
        "by_requested_resolved_invalid_reason": Counter(),
        "by_seed": {},
    }


def _accumulate_validity_breakdown(
    accumulator: dict[str, object],
    *,
    record: Mapping[str, object],
    validity_key: str,
    missing_fields: Counter[str],
) -> None:
    if validity_key not in record:
        missing_fields.update([validity_key])
        return
    if record.get(validity_key) is not False:
        return
    requested = str(record.get("requested_action", "unknown"))
    resolved = str(record.get("resolved_action", "unknown"))
    reason = _record_invalid_reason(record)
    accumulator["count"] = int(accumulator["count"]) + 1
    accumulator["by_requested_action"].update([requested])  # type: ignore[union-attr]
    accumulator["by_resolved_action"].update([resolved])  # type: ignore[union-attr]
    accumulator["by_invalid_reason"].update([reason])  # type: ignore[union-attr]
    accumulator["by_requested_resolved_invalid_reason"].update(  # type: ignore[union-attr]
        [(requested, resolved, reason)]
    )


def _finalize_unsupported_breakdown(
    accumulator: Mapping[str, object],
    *,
    seed: int | str | None,
) -> dict[str, object]:
    by_tuple = accumulator.get("by_requested_resolved_invalid_reason")
    tuple_counter = by_tuple if isinstance(by_tuple, Counter) else Counter()
    payload: dict[str, object] = {
        "by_requested_action": dict(
            sorted(_counter_object(accumulator.get("by_requested_action")).items())
        ),
        "by_resolved_action": dict(
            sorted(_counter_object(accumulator.get("by_resolved_action")).items())
        ),
        "by_invalid_reason": dict(
            sorted(_counter_object(accumulator.get("by_invalid_reason")).items())
        ),
        "by_requested_resolved_invalid_reason": [
            {
                "requested_action": requested,
                "resolved_action": resolved,
                "invalid_reason": reason,
                "count": int(count),
            }
            for (requested, resolved, reason), count in sorted(tuple_counter.items())
        ],
        "count": int(accumulator.get("count", 0)),
    }
    if seed is not None:
        payload["seed"] = seed
    by_seed = accumulator.get("by_seed")
    if isinstance(by_seed, dict) and by_seed:
        seed_payloads: dict[str, object] = {}
        for seed_key, seed_value in sorted(by_seed.items()):
            if not isinstance(seed_value, Mapping):
                continue
            seed_payloads[str(seed_key)] = _finalize_unsupported_breakdown(
                seed_value,
                seed=seed_value.get("seed", seed_key),
            )
        if seed_payloads:
            payload["by_seed"] = seed_payloads
    return payload


def _merge_unsupported_breakdown(
    accumulator: dict[str, object],
    payload: object,
) -> None:
    if not isinstance(payload, Mapping):
        return
    count = payload.get("count")
    if isinstance(count, int) and not isinstance(count, bool):
        accumulator["count"] = int(accumulator["count"]) + int(count)
    accumulator["by_requested_action"].update(  # type: ignore[union-attr]
        _counts_from_plain_mapping(payload.get("by_requested_action"))
    )
    accumulator["by_resolved_action"].update(  # type: ignore[union-attr]
        _counts_from_plain_mapping(payload.get("by_resolved_action"))
    )
    accumulator["by_invalid_reason"].update(  # type: ignore[union-attr]
        _counts_from_plain_mapping(payload.get("by_invalid_reason"))
    )
    raw_tuples = payload.get("by_requested_resolved_invalid_reason")
    if isinstance(raw_tuples, list):
        for item in raw_tuples:
            if not isinstance(item, Mapping):
                continue
            accumulator["by_requested_resolved_invalid_reason"].update(  # type: ignore[union-attr]
                [
                    (
                        str(item.get("requested_action", "unknown")),
                        str(item.get("resolved_action", "unknown")),
                        str(item.get("invalid_reason", "unknown")),
                    )
                ]
                * int(item.get("count", 0))
            )
    seed = payload.get("seed")
    if seed is not None:
        seed_key = str(seed)
        by_seed = accumulator["by_seed"]  # type: ignore[index]
        if not isinstance(by_seed, dict):
            return
        seed_accumulator = by_seed.get(seed_key)
        if not isinstance(seed_accumulator, dict):
            seed_accumulator = _empty_unsupported_breakdown()
            seed_accumulator["seed"] = seed
            by_seed[seed_key] = seed_accumulator
        seed_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"seed", "by_seed"}
        }
        _merge_unsupported_breakdown(seed_accumulator, seed_payload)


def _record_invalid_reason(record: Mapping[str, object]) -> str:
    outcome = record.get("outcome")
    if isinstance(outcome, Mapping):
        reason = outcome.get("invalid_reason")
        if isinstance(reason, str) and reason:
            return reason
    reason = record.get("invalid_reason")
    if isinstance(reason, str) and reason:
        return reason
    return "unknown"


def _counter_object(value: object) -> Counter[str]:
    return value if isinstance(value, Counter) else Counter()


def _counts_from_plain_mapping(value: object) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, raw_count in value.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            continue
        counter[str(key)] += int(raw_count)
    return counter


def _empty_rollout_context_accumulator() -> dict[str, object]:
    return {
        "diagnostic_count": 0,
        "missing_count": 0,
        "post_carrion_context_count": 0,
        "selected_score_delta_count": 0,
        "selected_score_delta_sum": 0.0,
        "selected_score_delta_abs_sum": 0.0,
        "selected_score_delta_abs_max": 0.0,
    }


def _accumulate_rollout_context_diagnostics(
    accumulator: dict[str, object],
    *,
    record: Mapping[str, object],
    missing_fields: Counter[str],
) -> None:
    diagnostics = record.get("policy_decision_diagnostics")
    if not isinstance(diagnostics, Mapping):
        missing_fields.update(["policy_decision_diagnostics"])
        accumulator["missing_count"] = int(accumulator["missing_count"]) + 1
        return
    accumulator["diagnostic_count"] = int(accumulator["diagnostic_count"]) + 1
    if bool(diagnostics.get("rollout_context_post_carrion_context", False)):
        accumulator["post_carrion_context_count"] = (
            int(accumulator["post_carrion_context_count"]) + 1
        )
    delta = _mapping_float(diagnostics, "rollout_context_selected_score_delta")
    if delta is None:
        return
    accumulator["selected_score_delta_count"] = (
        int(accumulator["selected_score_delta_count"]) + 1
    )
    accumulator["selected_score_delta_sum"] = (
        float(accumulator["selected_score_delta_sum"]) + delta
    )
    accumulator["selected_score_delta_abs_sum"] = (
        float(accumulator["selected_score_delta_abs_sum"]) + abs(delta)
    )
    accumulator["selected_score_delta_abs_max"] = max(
        float(accumulator["selected_score_delta_abs_max"]),
        abs(delta),
    )


def _finalize_rollout_context_summary(
    accumulator: Mapping[str, object],
) -> dict[str, object]:
    diagnostic_count = int(accumulator.get("diagnostic_count", 0))
    delta_count = int(accumulator.get("selected_score_delta_count", 0))
    return {
        "diagnostic_count": diagnostic_count,
        "missing_count": int(accumulator.get("missing_count", 0)),
        "post_carrion_context_count": int(
            accumulator.get("post_carrion_context_count", 0)
        ),
        "post_carrion_context_share": _safe_rate(
            int(accumulator.get("post_carrion_context_count", 0)),
            diagnostic_count,
        ),
        "selected_score_delta_count": delta_count,
        "selected_score_delta_mean": (
            _round(float(accumulator.get("selected_score_delta_sum", 0.0)) / delta_count)
            if delta_count
            else None
        ),
        "selected_score_delta_abs_mean": (
            _round(
                float(accumulator.get("selected_score_delta_abs_sum", 0.0))
                / delta_count
            )
            if delta_count
            else None
        ),
        "selected_score_delta_abs_max": (
            _round(float(accumulator.get("selected_score_delta_abs_max", 0.0)))
            if delta_count
            else None
        ),
    }


def _merge_rollout_context_summary(
    accumulator: dict[str, object],
    payload: object,
) -> None:
    if not isinstance(payload, Mapping):
        return
    diagnostic_count = int(payload.get("diagnostic_count", 0))
    delta_count = int(payload.get("selected_score_delta_count", 0))
    accumulator["diagnostic_count"] = int(accumulator["diagnostic_count"]) + diagnostic_count
    accumulator["missing_count"] = int(accumulator["missing_count"]) + int(
        payload.get("missing_count", 0)
    )
    accumulator["post_carrion_context_count"] = int(
        accumulator["post_carrion_context_count"]
    ) + int(payload.get("post_carrion_context_count", 0))
    if delta_count:
        mean = _finite_float(payload.get("selected_score_delta_mean"))
        abs_mean = _finite_float(payload.get("selected_score_delta_abs_mean"))
        accumulator["selected_score_delta_count"] = (
            int(accumulator["selected_score_delta_count"]) + delta_count
        )
        accumulator["selected_score_delta_sum"] = (
            float(accumulator["selected_score_delta_sum"]) + mean * delta_count
        )
        accumulator["selected_score_delta_abs_sum"] = (
            float(accumulator["selected_score_delta_abs_sum"]) + abs_mean * delta_count
        )
        accumulator["selected_score_delta_abs_max"] = max(
            float(accumulator["selected_score_delta_abs_max"]),
            _finite_float(payload.get("selected_score_delta_abs_max")),
        )


def _empty_navigation_accumulator() -> dict[str, dict[str, object]]:
    return {
        target: {
            "count": 0,
            "missing_count": 0,
            "distance_sum": 0.0,
            "strength_sum": 0.0,
            "strength_positive_count": 0,
            "min_distance": None,
            "max_strength": None,
        }
        for target in ("carrion", "water")
    }


def _accumulate_navigation_summary(
    accumulator: dict[str, dict[str, object]],
    *,
    record: Mapping[str, object],
    missing_fields: Counter[str],
) -> None:
    payload = _navigation_payload_from_record(record)
    if not payload:
        missing_fields.update(["observation_input"])
        for target in accumulator.values():
            target["missing_count"] = int(target["missing_count"]) + 1
        return
    for target_name, target_payload in payload.items():
        if target_name not in accumulator or not isinstance(target_payload, Mapping):
            continue
        distance = _mapping_float(target_payload, "distance")
        strength = _mapping_float(target_payload, "strength")
        if distance is None or strength is None:
            accumulator[target_name]["missing_count"] = (
                int(accumulator[target_name]["missing_count"]) + 1
            )
            continue
        _accumulate_navigation_target(accumulator[target_name], distance, strength)


def _navigation_payload_from_record(
    record: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    observation_input = record.get("observation_input")
    if not isinstance(observation_input, dict):
        return {}
    try:
        values = decode_observation_input(observation_input)
    except ValueError:
        return {}
    navigation_base = len(SELF_INPUT_FIELDS) + (
        PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
    )
    target_payloads: dict[str, dict[str, object]] = {}
    for target in ("carrion", "water"):
        if target not in NAVIGATION_TARGETS:
            continue
        target_index = NAVIGATION_TARGETS.index(target)
        offset = navigation_base + target_index * len(NAVIGATION_INPUT_FIELDS)
        if offset + len(NAVIGATION_INPUT_FIELDS) > len(values):
            continue
        raw = values[offset : offset + len(NAVIGATION_INPUT_FIELDS)]
        target_payloads[target] = {
            "dx": _round(raw[0] * NAVIGATION_RADIUS),
            "dy": _round(raw[1] * NAVIGATION_RADIUS),
            "distance": _round(raw[2] * NAVIGATION_RADIUS),
            "distance_norm": _round(raw[2]),
            "strength": _round(raw[3]),
        }
    return target_payloads


def _accumulate_navigation_target(
    accumulator: dict[str, object],
    distance: float,
    strength: float,
) -> None:
    accumulator["count"] = int(accumulator["count"]) + 1
    accumulator["distance_sum"] = float(accumulator["distance_sum"]) + distance
    accumulator["strength_sum"] = float(accumulator["strength_sum"]) + strength
    if strength > 0:
        accumulator["strength_positive_count"] = (
            int(accumulator["strength_positive_count"]) + 1
        )
    min_distance = accumulator.get("min_distance")
    accumulator["min_distance"] = (
        distance
        if min_distance is None
        else min(float(min_distance), distance)
    )
    max_strength = accumulator.get("max_strength")
    accumulator["max_strength"] = (
        strength
        if max_strength is None
        else max(float(max_strength), strength)
    )


def _finalize_navigation_summary(
    accumulator: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for target in ("carrion", "water"):
        raw = accumulator.get(target, {})
        count = int(raw.get("count", 0))
        summary[target] = {
            "count": count,
            "missing_count": int(raw.get("missing_count", 0)),
            "mean_distance": (
                _round(float(raw.get("distance_sum", 0.0)) / count)
                if count
                else None
            ),
            "mean_strength": (
                _round(float(raw.get("strength_sum", 0.0)) / count)
                if count
                else None
            ),
            "strength_positive_count": int(raw.get("strength_positive_count", 0)),
            "strength_positive_share": _safe_rate(
                int(raw.get("strength_positive_count", 0)),
                count,
            ),
            "min_distance": _round_optional(
                float(raw["min_distance"]) if raw.get("min_distance") is not None else None
            ),
            "max_strength": _round_optional(
                float(raw["max_strength"]) if raw.get("max_strength") is not None else None
            ),
        }
    return summary


def _merge_navigation_summary(
    accumulator: dict[str, dict[str, object]],
    payload: object,
) -> None:
    if not isinstance(payload, Mapping):
        return
    for target in ("carrion", "water"):
        raw = payload.get(target)
        if not isinstance(raw, Mapping):
            continue
        count = int(raw.get("count", 0))
        accumulator[target]["count"] = int(accumulator[target]["count"]) + count
        accumulator[target]["missing_count"] = int(
            accumulator[target]["missing_count"]
        ) + int(raw.get("missing_count", 0))
        mean_distance = _mapping_float(raw, "mean_distance")
        mean_strength = _mapping_float(raw, "mean_strength")
        if mean_distance is not None:
            accumulator[target]["distance_sum"] = (
                float(accumulator[target]["distance_sum"]) + mean_distance * count
            )
        if mean_strength is not None:
            accumulator[target]["strength_sum"] = (
                float(accumulator[target]["strength_sum"]) + mean_strength * count
            )
        accumulator[target]["strength_positive_count"] = int(
            accumulator[target]["strength_positive_count"]
        ) + int(raw.get("strength_positive_count", 0))
        min_distance = _mapping_float(raw, "min_distance")
        if min_distance is not None:
            current_min = accumulator[target].get("min_distance")
            accumulator[target]["min_distance"] = (
                min_distance
                if current_min is None
                else min(float(current_min), min_distance)
            )
        max_strength = _mapping_float(raw, "max_strength")
        if max_strength is not None:
            current_max = accumulator[target].get("max_strength")
            accumulator[target]["max_strength"] = (
                max_strength
                if current_max is None
                else max(float(current_max), max_strength)
            )


def _state_delta(
    before: Mapping[str, object],
    after: Mapping[str, object],
    key: str,
) -> float | None:
    before_value = _mapping_float(before, key)
    after_value = _mapping_float(after, key)
    if before_value is None or after_value is None:
        return None
    return _round(after_value - before_value)


def _record_seed(record: Mapping[str, object]) -> int | str:
    seed = record.get("source_seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        return int(seed)
    return "unknown"


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
