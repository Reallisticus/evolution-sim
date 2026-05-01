from __future__ import annotations

from typing import Any

TICK_BANDS = ("early", "mid", "late", "terminal")


def _empty_trophic_role_counts(trophic_role_codes: dict[str, int]) -> dict[str, int]:
    return {role: 0 for role in trophic_role_codes if role != "none"}


def _empty_meat_mode_counts(meat_mode_codes: dict[str, int]) -> dict[str, int]:
    return {mode: 0 for mode in meat_mode_codes}


def _agent_counts_by_trophic_surface(
    world: Any,
    agents: list[Any],
    *,
    trophic_role_codes: dict[str, int],
    meat_mode_codes: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    role_counts = _empty_trophic_role_counts(trophic_role_codes)
    mode_counts = _empty_meat_mode_counts(meat_mode_codes)
    for agent in agents:
        role_counts[world._trophic_role(agent)] += 1
        mode_counts[world._meat_mode(agent)] += 1
    return role_counts, mode_counts


def _agents_alive_at_tick(world: Any, tick: int) -> list[Any]:
    return [
        agent
        for agent in world.agents.values()
        if agent.birth_tick <= tick
        and (agent.death_tick is None or agent.death_tick > tick)
    ]


def _survival_record(initial_count: int, alive_count: int) -> dict[str, object]:
    return {
        "initial_agents": initial_count,
        "alive_initial_agents": alive_count,
        "survival_rate": round(alive_count / initial_count, 4)
        if initial_count
        else None,
    }


def _sorted_count_map(counts: dict[str, int]) -> dict[str, int]:
    return {key: int(counts[key]) for key in sorted(counts)}


def _tick_band_name(tick: int, ticks_executed: int) -> str:
    span = max(1, ticks_executed)
    if tick < span * 0.25:
        return "early"
    if tick < span * 0.5:
        return "mid"
    if tick < span * 0.75:
        return "late"
    return "terminal"


def _empty_tick_band_meat_mode_counts(
    meat_mode_codes: dict[str, int],
) -> dict[str, dict[str, int]]:
    return {band: _empty_meat_mode_counts(meat_mode_codes) for band in TICK_BANDS}


def _empty_tick_band_meat_mode_cause_counts(
    meat_mode_codes: dict[str, int],
) -> dict[str, dict[str, dict[str, int]]]:
    return {band: {mode: {} for mode in meat_mode_codes} for band in TICK_BANDS}


def build_trophic_lifecycle_summary(
    world: Any,
    *,
    ticks_executed: int,
    trophic_role_codes: dict[str, int],
    meat_mode_codes: dict[str, int],
) -> dict[str, object]:
    initial_agents = [
        agent for agent in world.agents.values() if agent.parent_id is None
    ]
    born_agents = [
        agent for agent in world.agents.values() if agent.parent_id is not None
    ]
    dead_agents = [agent for agent in world.agents.values() if not agent.alive]

    initial_role_counts, initial_mode_counts = _agent_counts_by_trophic_surface(
        world,
        initial_agents,
        trophic_role_codes=trophic_role_codes,
        meat_mode_codes=meat_mode_codes,
    )
    child_role_counts, child_mode_counts = _agent_counts_by_trophic_surface(
        world,
        born_agents,
        trophic_role_codes=trophic_role_codes,
        meat_mode_codes=meat_mode_codes,
    )
    death_role_counts, death_mode_counts = _agent_counts_by_trophic_surface(
        world,
        dead_agents,
        trophic_role_codes=trophic_role_codes,
        meat_mode_codes=meat_mode_codes,
    )

    parent_role_counts = _empty_trophic_role_counts(trophic_role_codes)
    parent_mode_counts = _empty_meat_mode_counts(meat_mode_codes)
    parent_births_by_band = _empty_tick_band_meat_mode_counts(meat_mode_codes)
    child_births_by_band = _empty_tick_band_meat_mode_counts(meat_mode_codes)
    for child in born_agents:
        if child.parent_id is None:
            continue
        parent = world.agents.get(child.parent_id)
        if parent is None:
            continue
        band = _tick_band_name(child.birth_tick, ticks_executed)
        parent_role_counts[world._trophic_role(parent)] += 1
        parent_mode_counts[world._meat_mode(parent)] += 1
        parent_births_by_band[band][world._meat_mode(parent)] += 1
        child_births_by_band[band][world._meat_mode(child)] += 1

    deaths_by_band = _empty_tick_band_meat_mode_counts(meat_mode_codes)
    death_causes_by_band = _empty_tick_band_meat_mode_cause_counts(meat_mode_codes)
    for agent in dead_agents:
        if agent.death_tick is None:
            continue
        band = _tick_band_name(agent.death_tick, ticks_executed)
        mode = world._meat_mode(agent)
        cause = world._death_cause(agent)
        deaths_by_band[band][mode] += 1
        mode_causes = death_causes_by_band[band][mode]
        mode_causes[cause] = mode_causes.get(cause, 0) + 1

    last_alive_tick_by_mode = {mode: None for mode in meat_mode_codes}
    final_tick = max(0, ticks_executed - 1)
    for agent in world.agents.values():
        mode = world._meat_mode(agent)
        if agent.death_tick is None:
            last_alive_tick = final_tick
        else:
            last_alive_tick = max(
                agent.birth_tick,
                min(final_tick, agent.death_tick - 1),
            )
        current_last = last_alive_tick_by_mode[mode]
        if current_last is None or last_alive_tick > current_last:
            last_alive_tick_by_mode[mode] = last_alive_tick

    alive_initial_agents = [agent for agent in initial_agents if agent.alive]
    alive_initial_role_counts, alive_initial_mode_counts = (
        _agent_counts_by_trophic_surface(
            world,
            alive_initial_agents,
            trophic_role_codes=trophic_role_codes,
            meat_mode_codes=meat_mode_codes,
        )
    )
    initial_role_survival = {
        role: _survival_record(
            initial_role_counts[role],
            alive_initial_role_counts[role],
        )
        for role in initial_role_counts
    }
    initial_mode_survival = {
        mode: _survival_record(
            initial_mode_counts[mode],
            alive_initial_mode_counts[mode],
        )
        for mode in initial_mode_counts
    }

    late_window_size = max(1, ticks_executed // 4)
    late_window_start = max(0, ticks_executed - late_window_size)
    sample_ticks = sorted(
        {
            late_window_start,
            (late_window_start + final_tick) // 2,
            final_tick,
        }
    )
    samples: list[dict[str, object]] = []
    role_presence_ticks = _empty_trophic_role_counts(trophic_role_codes)
    mode_presence_ticks = _empty_meat_mode_counts(meat_mode_codes)
    for sample_tick in sample_ticks:
        sample_agents = _agents_alive_at_tick(world, sample_tick)
        role_counts, mode_counts = _agent_counts_by_trophic_surface(
            world,
            sample_agents,
            trophic_role_codes=trophic_role_codes,
            meat_mode_codes=meat_mode_codes,
        )
        for role, count in role_counts.items():
            if count > 0:
                role_presence_ticks[role] += 1
        for mode, count in mode_counts.items():
            if count > 0:
                mode_presence_ticks[mode] += 1
        samples.append(
            {
                "tick": sample_tick,
                "alive_agents": len(sample_agents),
                "trophic_role_counts": role_counts,
                "meat_mode_counts": mode_counts,
            }
        )

    return {
        "initial_trophic_role_counts": initial_role_counts,
        "initial_meat_mode_counts": initial_mode_counts,
        "births_by_parent_trophic_role": parent_role_counts,
        "births_by_parent_meat_mode": parent_mode_counts,
        "births_by_child_trophic_role": child_role_counts,
        "births_by_child_meat_mode": child_mode_counts,
        "deaths_by_trophic_role": death_role_counts,
        "deaths_by_meat_mode": death_mode_counts,
        "death_causes": _sorted_count_map(world.run_death_cause_counts),
        "death_causes_by_trophic_role": {
            role: _sorted_count_map(counts)
            for role, counts in world.run_death_causes_by_trophic_role.items()
        },
        "death_causes_by_meat_mode": {
            mode: _sorted_count_map(counts)
            for mode, counts in world.run_death_causes_by_meat_mode.items()
        },
        "meat_mode_persistence": {
            "last_alive_tick_by_meat_mode": last_alive_tick_by_mode,
            "births_by_parent_meat_mode_by_tick_band": parent_births_by_band,
            "births_by_child_meat_mode_by_tick_band": child_births_by_band,
            "deaths_by_meat_mode_by_tick_band": deaths_by_band,
            "death_causes_by_meat_mode_by_tick_band": {
                band: {
                    mode: _sorted_count_map(causes)
                    for mode, causes in mode_counts.items()
                }
                for band, mode_counts in death_causes_by_band.items()
            },
        },
        "initial_cohort_survival_by_trophic_role": initial_role_survival,
        "initial_cohort_survival_by_meat_mode": initial_mode_survival,
        "late_window": {
            "start_tick": late_window_start,
            "end_tick": final_tick,
            "sample_ticks": sample_ticks,
            "samples": samples,
            "presence_ticks_by_trophic_role": role_presence_ticks,
            "presence_ticks_by_meat_mode": mode_presence_ticks,
        },
    }
