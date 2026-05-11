from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import SELF_INPUT_FIELDS, decode_observation_input
from evolution_sim.mind.carrion_counterfactual import DEFAULT_COUNTERFACTUAL_SCRIPTS
from evolution_sim.mind.dataset import (
    TRAJECTORY_EPISODE_ID_FIELD,
    TrajectoryJsonlDataset,
    combined_dataset_provenance,
)
from evolution_sim.mind.horizon_labels import (
    ANIMAL_RESOURCE_FOOD_SOURCES,
    DEFAULT_HORIZON_TICKS,
    VIABILITY_RATIO_FLOOR,
    build_horizon_label_records,
    normalize_horizon_ticks,
)
from evolution_sim.mind.provenance import (
    stable_payload_digest,
    validate_dataset_provenance,
)

MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION = (
    "mind_v3_carrion_counterfactual_labels_v1"
)
MIND_V3_CARRION_COUNTERFACTUAL_LABEL_POLICY = (
    "policy_visible_counterfactual_iql_action_value_labels_v1"
)
MIND_V3_CARRION_COUNTERFACTUAL_VALUE_POLICY = (
    "survival_homeostasis_animal_resource_value_proxy_v1"
)
DEFAULT_PRIMARY_COUNTERFACTUAL_LABEL_HORIZON = 120
VALUE_COMPONENT_WEIGHTS: dict[str, float] = {
    "terminal_alive": 0.45,
    "balanced_core_min": 0.2,
    "matched_diet": 0.15,
    "animal_resource_gain": 0.1,
    "reproduced": 0.1,
}
_MATCHED_DIET_INPUT_INDEX = SELF_INPUT_FIELDS.index("matched_diet_ratio")


class CarrionCounterfactualLabelError(ValueError):
    pass


def parse_counterfactual_label_horizons(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise CarrionCounterfactualLabelError(
            "horizons must be a comma-separated integer list"
        )
    try:
        return normalize_horizon_ticks(int(part) for part in parts)
    except ValueError as exc:
        raise CarrionCounterfactualLabelError(
            "horizons must be a comma-separated integer list"
        ) from exc


def build_carrion_counterfactual_label_report(
    datasets: Sequence[TrajectoryJsonlDataset],
    *,
    horizons: Iterable[int] = DEFAULT_HORIZON_TICKS,
    primary_horizon: int = DEFAULT_PRIMARY_COUNTERFACTUAL_LABEL_HORIZON,
    scripts: Sequence[str] | None = None,
    source_counterfactual_report: Mapping[str, object] | None = None,
    source_counterfactual_report_path: str | Path | None = None,
) -> dict[str, object]:
    if not datasets:
        raise CarrionCounterfactualLabelError(
            "at least one trajectory dataset is required"
        )
    horizon_ticks = normalize_horizon_ticks(horizons)
    primary = _primary_horizon(primary_horizon, horizon_ticks=horizon_ticks)
    script_filters = _script_filters(
        scripts,
        source_counterfactual_report=source_counterfactual_report,
    )
    contextual_records = _counterfactual_records_with_source_context(datasets)
    filtered_records = [
        record
        for record in contextual_records
        if _record_matches_script_filter(record, script_filters)
    ]
    if not filtered_records:
        raise CarrionCounterfactualLabelError(
            "no trajectory records matched the requested counterfactual scripts"
        )
    rollout_targets = _rollout_terminal_targets(filtered_records)
    horizon_labels = build_horizon_label_records(
        filtered_records,
        horizons=horizon_ticks,
    )
    if len(horizon_labels) != len(filtered_records):
        raise CarrionCounterfactualLabelError(
            "counterfactual horizon labels must align with filtered records"
        )
    labels = [
        _counterfactual_label(
            record=record,
            horizon_label=horizon_label,
            horizon_ticks=horizon_ticks,
            primary_horizon=primary,
            filtered_record_index=index,
            rollout_terminal_target=rollout_targets[index],
        )
        for index, (record, horizon_label) in enumerate(
            zip(filtered_records, horizon_labels, strict=True)
        )
    ]
    contract = _label_contract(
        horizon_ticks=horizon_ticks,
        primary_horizon=primary,
        script_filters=script_filters,
        source_counterfactual_report_path=source_counterfactual_report_path,
    )
    provenance = combined_dataset_provenance(datasets)
    source_report_digest = (
        stable_payload_digest(source_counterfactual_report)
        if source_counterfactual_report is not None
        else None
    )
    return {
        "schema_version": MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
        "label_policy": MIND_V3_CARRION_COUNTERFACTUAL_LABEL_POLICY,
        "label_contract": contract,
        "provenance": {
            **provenance,
            "label_contract_digest": stable_payload_digest(contract),
            "source_counterfactual_report_digest": source_report_digest,
        },
        "source": {
            "trajectory_count": len(datasets),
            "trajectory_paths": [str(dataset.path) for dataset in datasets],
            "record_count": len(contextual_records),
            "filtered_record_count": len(filtered_records),
            "script_filters": list(script_filters) if script_filters else None,
            "source_counterfactual_report_path": (
                str(source_counterfactual_report_path)
                if source_counterfactual_report_path is not None
                else None
            ),
        },
        "aggregate": _aggregate_counterfactual_labels(
            labels,
            primary_horizon=primary,
        ),
        "labels": labels,
    }


def write_carrion_counterfactual_label_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _counterfactual_records_with_source_context(
    datasets: Sequence[TrajectoryJsonlDataset],
) -> list[dict[str, object]]:
    contextual_records: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(datasets):
        provenance = validate_dataset_provenance(dataset.footer.get("provenance"))
        split_id = str(provenance["split_id"])
        source_seeds = [int(seed) for seed in list(provenance["source_seeds"])]
        source_seed_text = ",".join(str(seed) for seed in source_seeds)
        episode_id = (
            f"{dataset_index}:{split_id}:seed={source_seed_text}:path={dataset.path}"
        )
        fixture_seed = source_seeds[0] if len(source_seeds) == 1 else None
        for dataset_record_index, record in enumerate(dataset.records):
            contextual = dict(record)
            contextual[TRAJECTORY_EPISODE_ID_FIELD] = episode_id
            contextual["counterfactual_dataset_index"] = dataset_index
            contextual["counterfactual_dataset_record_index"] = dataset_record_index
            contextual["counterfactual_trajectory_path"] = str(dataset.path)
            contextual["counterfactual_source_seeds"] = list(source_seeds)
            contextual["counterfactual_fixture_seed"] = fixture_seed
            contextual_records.append(contextual)
    return contextual_records


def _counterfactual_label(
    *,
    record: Mapping[str, object],
    horizon_label: Mapping[str, object],
    horizon_ticks: tuple[int, ...],
    primary_horizon: int,
    filtered_record_index: int,
    rollout_terminal_target: Mapping[str, object],
) -> dict[str, object]:
    source_script = _script_name(record)
    action_support = _action_support(record)
    horizons = {
        str(horizon): _horizon_action_value_label(
            horizon_payload=_horizon_payload(horizon_label, horizon),
        )
        for horizon in horizon_ticks
    }
    primary_payload = horizons[str(primary_horizon)]
    return {
        "schema_version": MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
        "episode_id": str(horizon_label["episode_id"]),
        "filtered_record_index": filtered_record_index,
        "dataset_index": _optional_int(record.get("counterfactual_dataset_index")),
        "dataset_record_index": _optional_int(
            record.get("counterfactual_dataset_record_index")
        ),
        "trajectory_path": _optional_string(
            record.get("counterfactual_trajectory_path")
        ),
        "source_seeds": [
            int(seed)
            for seed in list(record.get("counterfactual_source_seeds", []))
            if isinstance(seed, int) and not isinstance(seed, bool)
        ],
        "fixture_seed": _optional_int(record.get("counterfactual_fixture_seed")),
        "tick": _required_int(horizon_label.get("tick"), field="tick"),
        "agent_id": _required_int(horizon_label.get("agent_id"), field="agent_id"),
        "source_script": source_script,
        "requested_action": str(horizon_label.get("requested_action", "")),
        "resolved_action": str(horizon_label.get("resolved_action", "")),
        "logged_action": action_support["logged_action"],
        "action_source": str(horizon_label.get("action_source", "")),
        "policy_id": horizon_label.get("policy_id"),
        "policy_version": horizon_label.get("policy_version"),
        "source_state": dict(horizon_label.get("source_state", {})),
        "immediate": dict(horizon_label.get("immediate", {})),
        "action_support": action_support,
        "primary_horizon": str(primary_horizon),
        "primary_target": primary_payload,
        "rollout_terminal_target": dict(rollout_terminal_target),
        "horizons": horizons,
    }


def _horizon_action_value_label(
    *,
    horizon_payload: Mapping[str, object],
) -> dict[str, object]:
    observed = bool(horizon_payload.get("observed", False))
    if not observed:
        return {
            "observed": False,
            "censored": True,
            "target_tick": _optional_int(horizon_payload.get("target_tick")),
            "terminal_alive": None,
            "terminal_state": None,
            "viability": None,
            "animal_resource": None,
            "action_value": None,
        }
    terminal_state = _mapping(horizon_payload.get("end_state"))
    viability = _mapping(horizon_payload.get("viability"))
    animal_resource = _mapping(horizon_payload.get("animal_resource"))
    terminal_alive = horizon_payload.get("survived")
    reproduced = horizon_payload.get("reproduced")
    animal_gain = _finite_float(animal_resource.get("gained_energy"), default=0.0)
    matched_diet = _finite_float(
        terminal_state.get("matched_diet_ratio"),
        default=0.0,
    )
    balanced_core_min = _finite_float(
        viability.get("balanced_core_min"),
        default=0.0,
    )
    components = {
        "terminal_alive": 1.0 if terminal_alive is True else 0.0,
        "balanced_core_min": _clamp01(balanced_core_min),
        "matched_diet": _clamp01(matched_diet),
        "animal_resource_gain": _clamp01(animal_gain),
        "reproduced": 1.0 if reproduced is True else 0.0,
    }
    return {
        "observed": True,
        "censored": False,
        "target_tick": _optional_int(horizon_payload.get("target_tick")),
        "end_tick": _optional_int(horizon_payload.get("end_tick")),
        "terminal_alive": terminal_alive if isinstance(terminal_alive, bool) else None,
        "terminal_state": {
            "energy_ratio": _optional_ratio(terminal_state.get("energy_ratio")),
            "hydration_ratio": _optional_ratio(terminal_state.get("hydration_ratio")),
            "health_ratio": _optional_ratio(terminal_state.get("health_ratio")),
            "matched_diet_ratio": _optional_ratio(
                terminal_state.get("matched_diet_ratio")
            ),
        },
        "viability": {
            "ratio_floor": VIABILITY_RATIO_FLOOR,
            "energy": _optional_bool(viability.get("energy")),
            "hydration": _optional_bool(viability.get("hydration")),
            "health": _optional_bool(viability.get("health")),
            "matched_diet": _optional_bool(viability.get("matched_diet")),
            "balanced_core_min": _optional_ratio(
                viability.get("balanced_core_min")
            ),
        },
        "animal_resource": {
            "consumed": bool(animal_resource.get("animal_resource_consumed", False)),
            "first_contact_tick": _optional_int(
                animal_resource.get("first_contact_tick")
            ),
            "fresh_kill_events": _nonnegative_int(
                animal_resource.get("fresh_kill_events"),
            ),
            "carcass_events": _nonnegative_int(
                animal_resource.get("carcass_events"),
            ),
            "gained_energy": _round(animal_gain),
            "survived_after_first_contact": _optional_bool(
                animal_resource.get("survived_after_first_contact")
            ),
        },
        "reproduced": reproduced if isinstance(reproduced, bool) else None,
        "action_value": {
            "policy": MIND_V3_CARRION_COUNTERFACTUAL_VALUE_POLICY,
            "component_weights": dict(VALUE_COMPONENT_WEIGHTS),
            "components": components,
            "score": _round(
                sum(
                    VALUE_COMPONENT_WEIGHTS[name] * value
                    for name, value in components.items()
                )
            ),
        },
    }


def _aggregate_counterfactual_labels(
    labels: Sequence[Mapping[str, object]],
    *,
    primary_horizon: int,
) -> dict[str, object]:
    script_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    legal_logged_count = 0
    primary_observed = 0
    primary_alive = 0
    primary_reproduced = 0
    primary_animal_gain = 0.0
    primary_action_values: list[float] = []
    terminal_viability_counts: Counter[str] = Counter()
    rollout_alive_label_count = 0
    rollout_terminal_values: list[float] = []
    terminal_agents: dict[tuple[str, int], bool] = {}
    for label in labels:
        script_counts[str(label.get("source_script"))] += 1
        action_support = _mapping(label.get("action_support"))
        action_counts[str(action_support.get("logged_action"))] += 1
        if action_support.get("logged_action_legal") is True:
            legal_logged_count += 1
        primary = _mapping(label.get("primary_target"))
        if primary.get("observed") is not True:
            rollout_target = _mapping(label.get("rollout_terminal_target"))
            if rollout_target.get("terminal_alive") is True:
                rollout_alive_label_count += 1
            rollout_value = _mapping(rollout_target.get("action_value"))
            rollout_score = _optional_float(rollout_value.get("score"))
            if rollout_score is not None:
                rollout_terminal_values.append(rollout_score)
            terminal_agents[
                (str(label.get("episode_id")), int(label.get("agent_id", 0)))
            ] = rollout_target.get("terminal_alive") is True
            continue
        primary_observed += 1
        if primary.get("terminal_alive") is True:
            primary_alive += 1
        if primary.get("reproduced") is True:
            primary_reproduced += 1
        animal = _mapping(primary.get("animal_resource"))
        primary_animal_gain += _finite_float(animal.get("gained_energy"), default=0.0)
        action_value = _mapping(primary.get("action_value"))
        score = _optional_float(action_value.get("score"))
        if score is not None:
            primary_action_values.append(score)
        viability = _mapping(primary.get("viability"))
        for component in ("energy", "hydration", "health", "matched_diet"):
            if viability.get(component) is True:
                terminal_viability_counts[component] += 1
        rollout_target = _mapping(label.get("rollout_terminal_target"))
        if rollout_target.get("terminal_alive") is True:
            rollout_alive_label_count += 1
        rollout_value = _mapping(rollout_target.get("action_value"))
        rollout_score = _optional_float(rollout_value.get("score"))
        if rollout_score is not None:
            rollout_terminal_values.append(rollout_score)
        terminal_agents[
            (str(label.get("episode_id")), int(label.get("agent_id", 0)))
        ] = rollout_target.get("terminal_alive") is True
    label_count = len(labels)
    terminal_agent_count = len(terminal_agents)
    terminal_alive_agent_count = sum(
        1 for alive in terminal_agents.values() if alive
    )
    return {
        "label_count": label_count,
        "primary_horizon": primary_horizon,
        "source_script_counts": dict(sorted(script_counts.items())),
        "logged_action_counts": dict(sorted(action_counts.items())),
        "logged_action_legal_count": legal_logged_count,
        "logged_action_legal_rate": _safe_rate(legal_logged_count, label_count),
        "primary_observed_count": primary_observed,
        "primary_terminal_alive_count": primary_alive,
        "primary_terminal_alive_rate": _safe_rate(primary_alive, primary_observed),
        "primary_reproduced_count": primary_reproduced,
        "primary_reproduced_rate": _safe_rate(
            primary_reproduced,
            primary_observed,
        ),
        "primary_animal_resource_gain_total": _round(primary_animal_gain),
        "primary_animal_resource_gain_mean": _safe_mean(
            [primary_animal_gain / primary_observed] if primary_observed else []
        ),
        "primary_action_value_mean": _safe_mean(primary_action_values),
        "primary_action_value_min": (
            _round(min(primary_action_values)) if primary_action_values else None
        ),
        "primary_action_value_max": (
            _round(max(primary_action_values)) if primary_action_values else None
        ),
        "rollout_terminal_alive_label_count": rollout_alive_label_count,
        "rollout_terminal_alive_label_rate": _safe_rate(
            rollout_alive_label_count,
            label_count,
        ),
        "rollout_terminal_agent_count": terminal_agent_count,
        "rollout_terminal_alive_agent_count": terminal_alive_agent_count,
        "rollout_terminal_alive_agent_rate": _safe_rate(
            terminal_alive_agent_count,
            terminal_agent_count,
        ),
        "rollout_terminal_action_value_mean": _safe_mean(
            rollout_terminal_values
        ),
        "rollout_terminal_action_value_min": (
            _round(min(rollout_terminal_values))
            if rollout_terminal_values
            else None
        ),
        "rollout_terminal_action_value_max": (
            _round(max(rollout_terminal_values))
            if rollout_terminal_values
            else None
        ),
        "terminal_viability_true_counts": dict(sorted(terminal_viability_counts.items())),
    }


def _rollout_terminal_targets(
    records: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, object]]:
    positions_by_agent: dict[tuple[str, int], list[int]] = {}
    for index, record in enumerate(records):
        episode_id = str(record.get(TRAJECTORY_EPISODE_ID_FIELD, ""))
        agent_id = _required_int(record.get("agent_id"), field="agent_id")
        positions_by_agent.setdefault((episode_id, agent_id), []).append(index)
    targets: dict[int, dict[str, object]] = {}
    for positions in positions_by_agent.values():
        ordered = sorted(
            positions,
            key=lambda index: (
                _required_int(records[index].get("tick"), field="tick"),
                int(records[index].get("counterfactual_dataset_record_index", index)),
            ),
        )
        terminal_record = records[ordered[-1]]
        terminal_after = _mapping(terminal_record.get("after"))
        terminal_state = _terminal_state(terminal_record)
        terminal_alive = terminal_after.get("alive") is not False and not bool(
            _mapping(terminal_record.get("outcome")).get("died", False)
        )
        suffix_gain = 0.0
        suffix_events = 0
        suffix_reproductions = 0
        suffix_alive_decisions = 0
        for position in reversed(ordered):
            record = records[position]
            animal = _immediate_animal_resource(record)
            suffix_gain += animal["gained_energy"]
            suffix_events += animal["event_count"]
            if bool(_mapping(record.get("outcome")).get("reproduced", False)):
                suffix_reproductions += 1
            if bool(_mapping(record.get("before")).get("alive", True)):
                suffix_alive_decisions += 1
            targets[position] = _rollout_terminal_target(
                terminal_record=terminal_record,
                terminal_state=terminal_state,
                terminal_alive=terminal_alive,
                animal_resource_gain=suffix_gain,
                animal_resource_event_count=suffix_events,
                reproduction_event_count=suffix_reproductions,
                alive_decision_count=suffix_alive_decisions,
            )
    return targets


def _rollout_terminal_target(
    *,
    terminal_record: Mapping[str, object],
    terminal_state: Mapping[str, object],
    terminal_alive: bool,
    animal_resource_gain: float,
    animal_resource_event_count: int,
    reproduction_event_count: int,
    alive_decision_count: int,
) -> dict[str, object]:
    balanced_core_min = _balanced_core_min(terminal_state)
    components = {
        "terminal_alive": 1.0 if terminal_alive else 0.0,
        "balanced_core_min": _clamp01(balanced_core_min or 0.0),
        "matched_diet": _clamp01(
            _finite_float(terminal_state.get("matched_diet_ratio"), default=0.0)
        ),
        "animal_resource_gain": _clamp01(animal_resource_gain),
        "reproduced": 1.0 if reproduction_event_count > 0 else 0.0,
    }
    return {
        "policy": "rollout_episode_terminal_suffix_target_v1",
        "terminal_tick": _required_int(terminal_record.get("tick"), field="tick"),
        "terminal_alive": terminal_alive,
        "terminal_state": dict(terminal_state),
        "animal_resource_gain_to_terminal": _round(animal_resource_gain),
        "animal_resource_event_count_to_terminal": animal_resource_event_count,
        "reproduction_event_count_to_terminal": reproduction_event_count,
        "alive_decision_count_to_terminal": alive_decision_count,
        "action_value": {
            "policy": MIND_V3_CARRION_COUNTERFACTUAL_VALUE_POLICY,
            "component_weights": dict(VALUE_COMPONENT_WEIGHTS),
            "components": components,
            "score": _round(
                sum(
                    VALUE_COMPONENT_WEIGHTS[name] * value
                    for name, value in components.items()
                )
            ),
        },
    }


def _terminal_state(record: Mapping[str, object]) -> dict[str, object]:
    after = _mapping(record.get("after"))
    return {
        "energy_ratio": _optional_ratio(after.get("energy_ratio")),
        "hydration_ratio": _optional_ratio(after.get("hydration_ratio")),
        "health_ratio": _optional_ratio(after.get("health_ratio")),
        "matched_diet_ratio": _optional_ratio(_matched_diet_ratio(record)),
    }


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


def _balanced_core_min(state: Mapping[str, object]) -> float | None:
    values = [
        _optional_ratio(state.get(field))
        for field in ("energy_ratio", "hydration_ratio", "health_ratio")
    ]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return min(finite_values)


def _immediate_animal_resource(record: Mapping[str, object]) -> dict[str, object]:
    outcome = _mapping(record.get("outcome"))
    feeding = _mapping(outcome.get("feeding"))
    food_source = feeding.get("food_source")
    if food_source not in ANIMAL_RESOURCE_FOOD_SOURCES:
        return {"event_count": 0, "gained_energy": 0.0}
    return {
        "event_count": 1,
        "gained_energy": _finite_float(feeding.get("gained_energy"), default=0.0),
    }


def _label_contract(
    *,
    horizon_ticks: tuple[int, ...],
    primary_horizon: int,
    script_filters: tuple[str, ...] | None,
    source_counterfactual_report_path: str | Path | None,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
        "label_policy": MIND_V3_CARRION_COUNTERFACTUAL_LABEL_POLICY,
        "value_policy": MIND_V3_CARRION_COUNTERFACTUAL_VALUE_POLICY,
        "horizon_ticks": list(horizon_ticks),
        "primary_horizon": primary_horizon,
        "script_filters": list(script_filters) if script_filters else None,
        "source_counterfactual_report_path": (
            str(source_counterfactual_report_path)
            if source_counterfactual_report_path is not None
            else None
        ),
        "action_support_policy": (
            "logged_action_and_full_policy_visible_action_mask_v1"
        ),
        "terminal_state_fields": [
            "terminal_alive",
            "energy_ratio",
            "hydration_ratio",
            "health_ratio",
            "matched_diet_ratio",
        ],
        "animal_resource_food_sources": sorted(ANIMAL_RESOURCE_FOOD_SOURCES),
        "value_component_weights": dict(VALUE_COMPONENT_WEIGHTS),
        "trainer_intent": (
            "diagnostic labels for later constrained torch/IQL integration; "
            "this report does not alter the trainer or runtime policy"
        ),
    }


def _script_filters(
    scripts: Sequence[str] | None,
    *,
    source_counterfactual_report: Mapping[str, object] | None,
) -> tuple[str, ...] | None:
    selected = tuple(dict.fromkeys(str(script) for script in (scripts or ()) if script))
    if not selected and source_counterfactual_report is not None:
        aggregate = _mapping(source_counterfactual_report.get("aggregate"))
        successful = aggregate.get("successful_scripts")
        if isinstance(successful, list):
            selected = tuple(
                dict.fromkeys(str(script) for script in successful if str(script))
            )
    if not selected:
        return None
    unsupported = sorted(
        script for script in selected if script not in DEFAULT_COUNTERFACTUAL_SCRIPTS
    )
    if unsupported:
        raise CarrionCounterfactualLabelError(
            "unsupported counterfactual script(s): " + ", ".join(unsupported)
        )
    return selected


def _record_matches_script_filter(
    record: Mapping[str, object],
    script_filters: tuple[str, ...] | None,
) -> bool:
    if script_filters is None:
        return True
    return _script_name(record) in set(script_filters)


def _script_name(record: Mapping[str, object]) -> str | None:
    source = record.get("action_source")
    if not isinstance(source, str):
        return None
    prefix = "counterfactual_script:"
    if source.startswith(prefix):
        return source[len(prefix) :]
    diagnostics = record.get("policy_decision_diagnostics")
    if isinstance(diagnostics, Mapping):
        script = diagnostics.get("script")
        if isinstance(script, str) and script:
            return script
    return None


def _action_support(record: Mapping[str, object]) -> dict[str, object]:
    action_mask = record.get("action_mask")
    mask = action_mask if isinstance(action_mask, Mapping) else {}
    requested_action = str(record.get("requested_action", ""))
    resolved_action = str(record.get("resolved_action", ""))
    logged_action = (
        requested_action
        if bool(record.get("resolution_action_valid", False))
        else resolved_action
    )
    legal_actions = [action for action in ACTION_NAMES if bool(mask.get(action, False))]
    return {
        "logged_action": logged_action,
        "requested_action_legal": bool(mask.get(requested_action, False)),
        "resolved_action_legal": bool(mask.get(resolved_action, False)),
        "logged_action_legal": bool(mask.get(logged_action, False)),
        "legal_action_count": len(legal_actions),
        "legal_actions": legal_actions,
        "action_mask": {action: bool(mask.get(action, False)) for action in ACTION_NAMES},
    }


def _primary_horizon(
    primary_horizon: int,
    *,
    horizon_ticks: tuple[int, ...],
) -> int:
    if isinstance(primary_horizon, bool) or not isinstance(primary_horizon, int):
        raise CarrionCounterfactualLabelError("primary horizon must be an integer")
    if primary_horizon not in horizon_ticks:
        raise CarrionCounterfactualLabelError(
            "primary horizon must be included in horizon ticks"
        )
    return primary_horizon


def _horizon_payload(
    horizon_label: Mapping[str, object],
    horizon: int,
) -> Mapping[str, object]:
    horizons = horizon_label.get("horizons")
    if not isinstance(horizons, Mapping):
        return {}
    payload = horizons.get(str(horizon))
    return payload if isinstance(payload, Mapping) else {}


def _mapping(payload: object) -> Mapping[str, object]:
    return payload if isinstance(payload, Mapping) else {}


def _required_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CarrionCounterfactualLabelError(f"{field} must be an integer")
    return value


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_ratio(value: object) -> float | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return _round(_clamp01(parsed))


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _finite_float(value: object, *, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _round(numerator / float(denominator))


def _safe_mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return _round(sum(finite) / len(finite))


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
