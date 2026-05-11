from __future__ import annotations

ANIMAL_RESOURCE_KINDS = ("fresh_kill", "carcass")
ANIMAL_RESOURCE_POLICY_BLOCKERS = ("occupant", "hazard", "water", "movement_mask")


def record_animal_resource_opportunity_tick_from_inputs(
    run_opportunity_by_meat_mode: dict[str, dict[str, int | float]],
    *,
    meat_mode_counts: dict[str, int],
    tick_consumption_by_meat_mode: dict[str, dict[str, int | float]],
    reachability_by_meat_mode: dict[str, dict[str, int]],
    resource_presence: dict[str, bool],
) -> None:
    for mode, agent_count in meat_mode_counts.items():
        if agent_count <= 0:
            continue
        counts = run_opportunity_by_meat_mode[mode]
        tick_consumption = tick_consumption_by_meat_mode[mode]
        reachable_counts = reachability_by_meat_mode.get(mode, _empty_reachability())
        counts["alive_ticks"] += 1
        counts["alive_agent_ticks"] += agent_count
        for key, value in tick_consumption.items():
            counts[key] += value

        animal_consumed = tick_consumption["animal_resource_consumption_events"] > 0
        fresh_kill_consumed = tick_consumption["fresh_kill_consumption_events"] > 0
        carcass_consumed = tick_consumption["carcass_consumption_events"] > 0
        if animal_consumed:
            counts["animal_resource_consumed_ticks"] += 1
        if fresh_kill_consumed:
            counts["fresh_kill_consumed_ticks"] += 1
        if carcass_consumed:
            counts["carcass_consumed_ticks"] += 1
        if resource_presence["animal_resource"]:
            _record_resource_opportunity(
                counts,
                resource="animal_resource",
                agent_count=agent_count,
                consumed=animal_consumed,
                reachable_agents=int(
                    reachable_counts.get("animal_resource_reachable_agents", 0)
                ),
                actionable_agents=int(
                    reachable_counts.get(
                        "animal_resource_policy_actionable_agents",
                        0,
                    )
                ),
                blocked_agents=int(
                    reachable_counts.get(
                        "animal_resource_reachable_policy_blocked_agents",
                        0,
                    )
                ),
                reachable_counts=reachable_counts,
            )
        else:
            counts["animal_resource_absent_ticks"] += 1
            counts["animal_resource_absent_agent_ticks"] += agent_count

        if resource_presence["fresh_kill"]:
            _record_resource_opportunity(
                counts,
                resource="fresh_kill",
                agent_count=agent_count,
                consumed=fresh_kill_consumed,
                reachable_agents=int(
                    reachable_counts.get("fresh_kill_reachable_agents", 0)
                ),
                actionable_agents=int(
                    reachable_counts.get("fresh_kill_policy_actionable_agents", 0)
                ),
                blocked_agents=int(
                    reachable_counts.get(
                        "fresh_kill_reachable_policy_blocked_agents",
                        0,
                    )
                ),
                reachable_counts=reachable_counts,
            )

        if resource_presence["carcass"]:
            _record_resource_opportunity(
                counts,
                resource="carcass",
                agent_count=agent_count,
                consumed=carcass_consumed,
                reachable_agents=int(reachable_counts.get("carcass_reachable_agents", 0)),
                actionable_agents=int(
                    reachable_counts.get("carcass_policy_actionable_agents", 0)
                ),
                blocked_agents=int(
                    reachable_counts.get("carcass_reachable_policy_blocked_agents", 0)
                ),
                reachable_counts=reachable_counts,
            )


def _empty_reachability() -> dict[str, int]:
    counts = {
        "animal_resource_reachable_agents": 0,
        "fresh_kill_reachable_agents": 0,
        "carcass_reachable_agents": 0,
    }
    for resource in ("animal_resource", *ANIMAL_RESOURCE_KINDS):
        counts[f"{resource}_policy_actionable_agents"] = 0
        counts[f"{resource}_reachable_policy_blocked_agents"] = 0
        for blocker in ANIMAL_RESOURCE_POLICY_BLOCKERS:
            counts[f"{resource}_policy_blocked_by_{blocker}_agents"] = 0
    return counts


def _record_policy_actionability(
    counts: dict[str, int | float],
    reachable_counts: dict[str, int],
    *,
    resource: str,
    actionable_agents: int,
    blocked_agents: int,
) -> None:
    if actionable_agents > 0:
        counts[f"{resource}_policy_actionable_ticks"] += 1
        counts[f"{resource}_policy_actionable_agent_ticks"] += actionable_agents
    if blocked_agents > 0:
        counts[f"{resource}_reachable_policy_blocked_ticks"] += 1
        counts[f"{resource}_reachable_policy_blocked_agent_ticks"] += blocked_agents
    for blocker in ANIMAL_RESOURCE_POLICY_BLOCKERS:
        blocker_agents = int(
            reachable_counts.get(
                f"{resource}_policy_blocked_by_{blocker}_agents",
                0,
            )
        )
        if blocker_agents > 0:
            counts[f"{resource}_policy_blocked_by_{blocker}_ticks"] += 1
            counts[f"{resource}_policy_blocked_by_{blocker}_agent_ticks"] += (
                blocker_agents
            )


def _record_resource_opportunity(
    counts: dict[str, int | float],
    *,
    resource: str,
    agent_count: int,
    consumed: bool,
    reachable_agents: int,
    actionable_agents: int,
    blocked_agents: int,
    reachable_counts: dict[str, int],
) -> None:
    counts[f"{resource}_present_ticks"] += 1
    counts[f"{resource}_present_agent_ticks"] += agent_count
    if not consumed:
        counts[f"{resource}_present_unconsumed_ticks"] += 1
        counts[f"{resource}_present_unconsumed_agent_ticks"] += agent_count
    if reachable_agents > 0:
        counts[f"{resource}_reachable_ticks"] += 1
        counts[f"{resource}_reachable_agent_ticks"] += reachable_agents
        if not consumed:
            counts[f"{resource}_reachable_unconsumed_ticks"] += 1
            counts[f"{resource}_reachable_unconsumed_agent_ticks"] += reachable_agents
        _record_policy_actionability(
            counts,
            reachable_counts,
            resource=resource,
            actionable_agents=actionable_agents,
            blocked_agents=blocked_agents,
        )
    else:
        counts[f"{resource}_present_unreachable_ticks"] += 1
        counts[f"{resource}_present_unreachable_agent_ticks"] += agent_count
