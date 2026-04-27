from __future__ import annotations

from collections import Counter
from typing import Any

from evolution_sim.env.runtime.observations import (
    OBSERVATION_SCHEMA_VERSION,
    observation_contract,
)
from evolution_sim.env.runtime.state import Agent

TRAJECTORY_SCHEMA_VERSION = "mind_trajectory_v1"
REWARD_SCHEMA_VERSION = "mind_reward_v1"
TRAJECTORY_RECORD_FIELDS: tuple[str, ...] = (
    "tick",
    "agent_id",
    "lineage_id",
    "runtime_species_id",
    "runtime_ecotype_id",
    "observation_schema",
    "observation_digest",
    "action_mask",
    "requested_action",
    "action_source",
    "action_valid",
    "resolved_action",
    "moved",
    "before",
    "after",
    "outcome",
    "reward",
)


def trajectory_contract() -> dict[str, object]:
    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "record_fields": list(TRAJECTORY_RECORD_FIELDS),
        "observation_contract": observation_contract(),
    }


def capture_agent_state(world: Any, agent: Agent) -> dict[str, object]:
    return {
        "x": agent.x,
        "y": agent.y,
        "energy": _round(agent.energy),
        "hydration": _round(agent.hydration),
        "health": _round(agent.health),
        "energy_ratio": _round(world._energy_ratio(agent)),
        "hydration_ratio": _round(world._hydration_ratio(agent)),
        "health_ratio": _round(world._health_ratio(agent)),
        "age": agent.age,
        "alive": agent.alive,
    }


def build_trajectory_record(
    *,
    tick: int,
    agent: Agent,
    before: dict[str, object],
    after: dict[str, object],
    observation_digest: str,
    action_mask: dict[str, bool],
    requested_action: str,
    action_source: str,
    resolved_action: str,
    moved: bool,
    resource_gain: float,
    reproduced: bool,
    died: bool,
    reproduction_ready_after: bool,
    runtime_species_id: int | None,
    runtime_ecotype_id: int | None,
) -> dict[str, object]:
    action_valid = bool(action_mask.get(requested_action, False))
    reward = build_reward(
        before=before,
        after=after,
        action_valid=action_valid,
        moved=moved,
        resource_gain=resource_gain,
        reproduced=reproduced,
        died=died,
        reproduction_ready_after=reproduction_ready_after,
    )
    return {
        "tick": tick,
        "agent_id": agent.agent_id,
        "lineage_id": agent.lineage_id,
        "runtime_species_id": runtime_species_id,
        "runtime_ecotype_id": runtime_ecotype_id,
        "observation_schema": OBSERVATION_SCHEMA_VERSION,
        "observation_digest": observation_digest,
        "action_mask": {action: bool(action_mask[action]) for action in action_mask},
        "requested_action": requested_action,
        "action_source": action_source,
        "action_valid": action_valid,
        "resolved_action": resolved_action,
        "moved": moved,
        "before": before,
        "after": after,
        "outcome": {
            "resource_gain": _round(resource_gain),
            "reproduced": reproduced,
            "died": died,
            "reproduction_ready_after": reproduction_ready_after,
        },
        "reward": reward,
    }


def build_reward(
    *,
    before: dict[str, object],
    after: dict[str, object],
    action_valid: bool,
    moved: bool,
    resource_gain: float,
    reproduced: bool,
    died: bool,
    reproduction_ready_after: bool,
) -> dict[str, object]:
    components = {
        "survival_continuation": -1.0 if died else 0.02,
        "energy_stability": _round(
            float(after["energy_ratio"]) - float(before["energy_ratio"])
        ),
        "hydration_stability": _round(
            float(after["hydration_ratio"]) - float(before["hydration_ratio"])
        ),
        "health_preservation": _round(
            float(after["health_ratio"]) - float(before["health_ratio"])
        ),
        "resource_acquisition": _round(resource_gain),
        "reproduction_readiness": 0.05 if reproduction_ready_after else 0.0,
        "reproduction_success": 1.0 if reproduced else 0.0,
        "invalid_action_penalty": -0.05 if not action_valid else 0.0,
        "movement_cost": -0.005 if moved else 0.0,
    }
    return {
        "schema_version": REWARD_SCHEMA_VERSION,
        "components": components,
        "total": _round(sum(float(value) for value in components.values())),
    }


def build_trajectory_payload(records: list[dict[str, object]]) -> dict[str, object]:
    action_counts = Counter(str(record["requested_action"]) for record in records)
    invalid_count = sum(1 for record in records if not bool(record["action_valid"]))
    total_reward = sum(float(record["reward"]["total"]) for record in records)
    return {
        **trajectory_contract(),
        "record_count": len(records),
        "invalid_action_count": invalid_count,
        "action_counts": {action: action_counts[action] for action in sorted(action_counts)},
        "mean_reward": _round(total_reward / len(records)) if records else 0.0,
        "records": records,
    }


def build_trajectory_summary(records: list[dict[str, object]]) -> dict[str, object]:
    payload = build_trajectory_payload(records)
    return {
        "schema_version": payload["schema_version"],
        "observation_schema_version": payload["observation_schema_version"],
        "reward_schema_version": payload["reward_schema_version"],
        "record_count": payload["record_count"],
        "invalid_action_count": payload["invalid_action_count"],
        "action_counts": payload["action_counts"],
        "mean_reward": payload["mean_reward"],
    }


def _round(value: float) -> float:
    return round(float(value), 4)
