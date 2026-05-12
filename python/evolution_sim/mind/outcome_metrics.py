from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


def build_run_outcome_metrics(
    *,
    summary: Mapping[str, object],
    trajectory_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    lifecycle = _mapping(summary.get("trophic_lifecycle"))
    diet = _mapping(summary.get("diet_end"))
    meat_modes = _mapping(summary.get("meat_mode_counts_at_end"))
    trophic_roles = _mapping(summary.get("trophic_role_counts_at_end"))
    fresh_kill = _mapping(summary.get("fresh_kill_end"))
    carcass = _mapping(summary.get("carcass_end"))
    opportunity = _mapping(summary.get("animal_resource_opportunity_by_meat_mode_end"))
    scavenger_opportunity = _mapping(opportunity.get("scavenger"))
    trajectory_counts = _trajectory_outcome_counts(trajectory_records)
    births = _int(summary.get("births"))
    alive = _int(summary.get("alive_agents"))
    deaths = _int(summary.get("deaths"))
    terminal_scavengers = _int(meat_modes.get("scavenger"))
    parent_births_by_mode = _int_mapping(lifecycle.get("births_by_parent_meat_mode"))
    child_births_by_mode = _int_mapping(lifecycle.get("births_by_child_meat_mode"))
    parent_births_by_role = _int_mapping(
        lifecycle.get("births_by_parent_trophic_role")
    )
    child_births_by_role = _int_mapping(lifecycle.get("births_by_child_trophic_role"))
    animal_events = (
        _int(fresh_kill.get("consumption_events"))
        + _int(carcass.get("consumption_events"))
    )
    animal_gained_energy = _float(diet.get("animal_energy"))
    scavenger_events = _int(
        scavenger_opportunity.get("animal_resource_consumption_events")
    )
    scavenger_gained = _float(
        scavenger_opportunity.get("animal_resource_gained_energy")
    )
    return {
        "terminal": {
            "ticks_executed": _int(summary.get("ticks_executed")),
            "alive_agents": alive,
            "deaths": deaths,
            "extinct": alive <= 0,
            "total_agents_seen": _int(summary.get("total_agents_seen")),
            "terminal_alive_by_meat_mode": _int_mapping(meat_modes),
            "terminal_alive_by_trophic_role": _int_mapping(trophic_roles),
            "terminal_scavenger_agents": terminal_scavengers,
        },
        "reproduction": {
            "births": births,
            "had_births": births > 0,
            "last_birth_tick": _optional_int(summary.get("last_birth_tick")),
            "births_by_parent_meat_mode": parent_births_by_mode,
            "births_by_child_meat_mode": child_births_by_mode,
            "births_by_parent_trophic_role": parent_births_by_role,
            "births_by_child_trophic_role": child_births_by_role,
            "scavenger_parent_births": parent_births_by_mode.get("scavenger", 0),
            "scavenger_child_births": child_births_by_mode.get("scavenger", 0),
            "unique_reproducing_agents": trajectory_counts[
                "unique_reproducing_agents"
            ],
        },
        "feeding": {
            "plant_events": _int(diet.get("plant_events")),
            "plant_energy": _round(_float(diet.get("plant_energy"))),
            "animal_resource_events": animal_events,
            "animal_resource_gained_energy": _round(animal_gained_energy),
            "fresh_kill_events": _int(fresh_kill.get("consumption_events")),
            "fresh_kill_gained_energy": _round(
                _float(fresh_kill.get("gained_energy"))
            ),
            "carcass_events": _int(carcass.get("consumption_events")),
            "carcass_gained_energy": _round(_float(carcass.get("gained_energy"))),
            "animal_energy_share": _round(_float(diet.get("animal_energy_share"))),
            "carcass_energy_share": _round(_float(diet.get("carcass_energy_share"))),
            "fresh_kill_energy_share": _round(
                _float(diet.get("fresh_kill_energy_share"))
            ),
            "unique_agents_that_ate": trajectory_counts["unique_agents_that_ate"],
            "unique_agents_that_ate_animal_resource": trajectory_counts[
                "unique_agents_that_ate_animal_resource"
            ],
            "food_source_event_counts": trajectory_counts["food_source_event_counts"],
        },
        "scavenging": {
            "terminal_scavenger_agents": terminal_scavengers,
            "scavenger_animal_resource_events": scavenger_events,
            "scavenger_animal_resource_gained_energy": _round(scavenger_gained),
            "scavenger_carcass_events": _int(
                scavenger_opportunity.get("carcass_consumption_events")
            ),
            "scavenger_carcass_gained_energy": _round(
                _float(scavenger_opportunity.get("carcass_gained_energy"))
            ),
            "scavenger_fresh_kill_events": _int(
                scavenger_opportunity.get("fresh_kill_consumption_events")
            ),
            "scavenger_fresh_kill_gained_energy": _round(
                _float(scavenger_opportunity.get("fresh_kill_gained_energy"))
            ),
            "scavenger_animal_resource_reachable_ticks": _int(
                scavenger_opportunity.get("animal_resource_reachable_ticks")
            ),
            "scavenger_animal_resource_policy_actionable_ticks": _int(
                scavenger_opportunity.get("animal_resource_policy_actionable_ticks")
            ),
            "unique_agents_that_ate_carcass": trajectory_counts[
                "unique_agents_that_ate_carcass"
            ],
            "unique_agents_that_ate_fresh_kill": trajectory_counts[
                "unique_agents_that_ate_fresh_kill"
            ],
        },
    }


def aggregate_run_outcome_metrics(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    metrics = [
        _mapping(run.get("outcome_metrics"))
        for run in runs
        if isinstance(run.get("outcome_metrics"), Mapping)
    ]
    if not metrics:
        metrics = [_fallback_run_outcome_metrics(run) for run in runs]
    terminal = [_mapping(item.get("terminal")) for item in metrics]
    reproduction = [_mapping(item.get("reproduction")) for item in metrics]
    feeding = [_mapping(item.get("feeding")) for item in metrics]
    scavenging = [_mapping(item.get("scavenging")) for item in metrics]
    food_source_counts: Counter[str] = Counter()
    terminal_mode_counts: Counter[str] = Counter()
    child_birth_mode_counts: Counter[str] = Counter()
    parent_birth_mode_counts: Counter[str] = Counter()
    for item in terminal:
        terminal_mode_counts.update(
            _int_mapping(item.get("terminal_alive_by_meat_mode"))
        )
    for item in reproduction:
        child_birth_mode_counts.update(
            _int_mapping(item.get("births_by_child_meat_mode"))
        )
        parent_birth_mode_counts.update(
            _int_mapping(item.get("births_by_parent_meat_mode"))
        )
    for item in feeding:
        food_source_counts.update(_int_mapping(item.get("food_source_event_counts")))
    alive_values = [_int(item.get("alive_agents")) for item in terminal]
    birth_values = [_int(item.get("births")) for item in reproduction]
    scavenger_event_values = [
        _int(item.get("scavenger_animal_resource_events")) for item in scavenging
    ]
    return {
        "run_count": len(runs),
        "terminal_survivor_run_count": sum(1 for value in alive_values if value > 0),
        "extinct_run_count": sum(1 for value in alive_values if value <= 0),
        "total_terminal_alive_agents": sum(alive_values),
        "terminal_alive_agents_mean": _mean(alive_values),
        "max_terminal_alive_agents": max(alive_values, default=0),
        "total_births": sum(birth_values),
        "births_mean": _mean(birth_values),
        "max_births": max(birth_values, default=0),
        "runs_with_births": sum(1 for value in birth_values if value > 0),
        "total_deaths": sum(_int(item.get("deaths")) for item in terminal),
        "terminal_alive_by_meat_mode": dict(sorted(terminal_mode_counts.items())),
        "births_by_parent_meat_mode": dict(sorted(parent_birth_mode_counts.items())),
        "births_by_child_meat_mode": dict(sorted(child_birth_mode_counts.items())),
        "total_scavenger_terminal_agents": sum(
            _int(item.get("terminal_scavenger_agents")) for item in scavenging
        ),
        "scavenger_terminal_agents_mean": _mean(
            [_int(item.get("terminal_scavenger_agents")) for item in scavenging]
        ),
        "total_scavenger_parent_births": sum(
            _int(item.get("scavenger_parent_births")) for item in reproduction
        ),
        "total_scavenger_child_births": sum(
            _int(item.get("scavenger_child_births")) for item in reproduction
        ),
        "total_animal_resource_consumption_events": sum(
            _int(item.get("animal_resource_events")) for item in feeding
        ),
        "total_animal_resource_gained_energy": _round(
            sum(_float(item.get("animal_resource_gained_energy")) for item in feeding)
        ),
        "total_carcass_consumption_events": sum(
            _int(item.get("carcass_events")) for item in feeding
        ),
        "total_fresh_kill_consumption_events": sum(
            _int(item.get("fresh_kill_events")) for item in feeding
        ),
        "runs_with_animal_resource_consumption": sum(
            1 for item in feeding if _int(item.get("animal_resource_events")) > 0
        ),
        "food_source_event_counts": dict(sorted(food_source_counts.items())),
        "total_scavenger_animal_resource_events": sum(scavenger_event_values),
        "total_scavenger_animal_resource_gained_energy": _round(
            sum(
                _float(item.get("scavenger_animal_resource_gained_energy"))
                for item in scavenging
            )
        ),
        "total_scavenger_carcass_events": sum(
            _int(item.get("scavenger_carcass_events")) for item in scavenging
        ),
        "total_scavenger_fresh_kill_events": sum(
            _int(item.get("scavenger_fresh_kill_events")) for item in scavenging
        ),
        "runs_with_scavenger_animal_resource_consumption": sum(
            1 for value in scavenger_event_values if value > 0
        ),
    }


def _trajectory_outcome_counts(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    food_source_counts: Counter[str] = Counter()
    ate_agents: set[int] = set()
    animal_agents: set[int] = set()
    carcass_agents: set[int] = set()
    fresh_kill_agents: set[int] = set()
    reproducing_agents: set[int] = set()
    for record in records:
        agent_id = _optional_int(record.get("agent_id"))
        outcome = _mapping(record.get("outcome"))
        if bool(outcome.get("reproduced")) and agent_id is not None:
            reproducing_agents.add(agent_id)
        feeding = _mapping(outcome.get("feeding"))
        if not bool(feeding.get("ate")):
            continue
        food_source = str(feeding.get("food_source", "unknown"))
        food_source_counts[food_source] += 1
        if agent_id is not None:
            ate_agents.add(agent_id)
            if food_source in {"carcass", "fresh_kill"}:
                animal_agents.add(agent_id)
            if food_source == "carcass":
                carcass_agents.add(agent_id)
            if food_source == "fresh_kill":
                fresh_kill_agents.add(agent_id)
    return {
        "food_source_event_counts": dict(sorted(food_source_counts.items())),
        "unique_agents_that_ate": len(ate_agents),
        "unique_agents_that_ate_animal_resource": len(animal_agents),
        "unique_agents_that_ate_carcass": len(carcass_agents),
        "unique_agents_that_ate_fresh_kill": len(fresh_kill_agents),
        "unique_reproducing_agents": len(reproducing_agents),
    }


def _fallback_run_outcome_metrics(run: Mapping[str, object]) -> dict[str, object]:
    alive = _int(run.get("alive_agents"))
    births = _int(run.get("births"))
    deaths = _int(run.get("deaths"))
    meat_modes = _int_mapping(run.get("meat_mode_counts_at_end"))
    terminal_scavengers = meat_modes.get("scavenger", 0)
    return {
        "terminal": {
            "alive_agents": alive,
            "deaths": deaths,
            "extinct": alive <= 0,
            "terminal_alive_by_meat_mode": meat_modes,
            "terminal_scavenger_agents": terminal_scavengers,
        },
        "reproduction": {
            "births": births,
            "had_births": births > 0,
            "births_by_parent_meat_mode": {},
            "births_by_child_meat_mode": {},
            "scavenger_parent_births": 0,
            "scavenger_child_births": 0,
        },
        "feeding": {
            "animal_resource_events": 0,
            "animal_resource_gained_energy": 0.0,
            "carcass_events": 0,
            "fresh_kill_events": 0,
            "food_source_event_counts": {},
        },
        "scavenging": {
            "terminal_scavenger_agents": terminal_scavengers,
            "scavenger_animal_resource_events": 0,
            "scavenger_animal_resource_gained_energy": 0.0,
            "scavenger_carcass_events": 0,
            "scavenger_fresh_kill_events": 0,
        },
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _int_mapping(value: object) -> dict[str, int]:
    payload = value if isinstance(value, Mapping) else {}
    return {str(key): _int(raw) for key, raw in payload.items()}


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return 0.0
    return parsed


def _mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return _round(sum(values) / len(values))


def _round(value: float) -> float:
    return round(float(value), 6)
