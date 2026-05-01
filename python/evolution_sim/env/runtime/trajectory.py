from __future__ import annotations

from collections import Counter
from typing import Any, Protocol

from evolution_sim.env.runtime.action_contract import (
    ACTION_CONTRACT_VERSION,
    action_contract,
)
from evolution_sim.env.runtime.observations import (
    OBSERVATION_SCHEMA_VERSION,
    observation_contract,
)
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.reproduction import (
    REPRODUCTIVE_GROUP_CONTRACT_VERSION,
    reproductive_group_contract,
)
from evolution_sim.env.runtime.state import Agent
from evolution_sim.genome.recombination import (
    GENOME_RECOMBINATION_CONTRACT_VERSION,
    genome_recombination_contract,
)

TRAJECTORY_SCHEMA_VERSION = "mind_trajectory_v1"
REWARD_SCHEMA_VERSION = "mind_reward_v1"
ACTION_OUTCOME_SCHEMA_VERSION = "mind_action_outcome_v2"
REWARD_COMPONENT_BOUNDS: dict[str, tuple[float, float]] = {
    "survival_continuation": (-1.0, 0.02),
    "energy_stability": (-1.0, 1.0),
    "hydration_stability": (-1.0, 1.0),
    "health_preservation": (-1.0, 1.0),
    "resource_acquisition": (0.0, 1.0),
    "reproduction_readiness": (0.0, 0.05),
    "reproduction_success": (0.0, 1.0),
    "invalid_action_penalty": (-0.05, 0.0),
    "movement_cost": (-0.005, 0.0),
}
REWARD_TOTAL_BOUNDS: tuple[float, float] = (
    round(sum(bounds[0] for bounds in REWARD_COMPONENT_BOUNDS.values()), 4),
    round(sum(bounds[1] for bounds in REWARD_COMPONENT_BOUNDS.values()), 4),
)
TRAJECTORY_RECORD_FIELDS: tuple[str, ...] = (
    "tick",
    "agent_id",
    "lineage_id",
    "runtime_species_id",
    "runtime_ecotype_id",
    "observation_schema",
    "observation_metadata",
    "observation_input",
    "observation_digest",
    "action_mask",
    "resolution_action_mask",
    "requested_action",
    "action_source",
    "policy_id",
    "policy_version",
    "action_valid",
    "resolution_action_valid",
    "resolved_action",
    "moved",
    "before",
    "after",
    "outcome",
    "reward",
)


class TrajectorySink(Protocol):
    def begin(
        self,
        *,
        run_id: str,
        config: dict[str, object],
        contract: dict[str, object],
    ) -> None:
        ...

    def write_record(self, record: dict[str, object]) -> None:
        ...

    def finish(self, *, summary: dict[str, object]) -> None:
        ...

    def abort(self) -> None:
        ...


def trajectory_contract(signal_config: Any | None = None) -> dict[str, object]:
    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "policy_interface_version": POLICY_INTERFACE_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "reproductive_group_contract_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        "genome_recombination_contract_version": GENOME_RECOMBINATION_CONTRACT_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "action_outcome_schema_version": ACTION_OUTCOME_SCHEMA_VERSION,
        "record_fields": list(TRAJECTORY_RECORD_FIELDS),
        "action_contract": action_contract(signal_config),
        "reproductive_group_contract": reproductive_group_contract(),
        "genome_recombination_contract": genome_recombination_contract(),
        "observation_contract": observation_contract(signal_config),
        "reward_contract": reward_contract(),
    }


def reward_contract() -> dict[str, object]:
    return {
        "schema_version": REWARD_SCHEMA_VERSION,
        "component_bounds": {
            name: [bounds[0], bounds[1]]
            for name, bounds in REWARD_COMPONENT_BOUNDS.items()
        },
        "total_bounds": [REWARD_TOTAL_BOUNDS[0], REWARD_TOTAL_BOUNDS[1]],
        "resource_acquisition_units": "clamped_action_energy_gain",
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
    observation_metadata: dict[str, object],
    observation_input: dict[str, object],
    observation_digest: str,
    action_mask: dict[str, bool],
    resolution_action_mask: dict[str, bool],
    requested_action: str,
    action_source: str,
    policy_id: str | None,
    policy_version: str | None,
    resolved_action: str,
    moved: bool,
    action_outcome: dict[str, object],
    resource_gain: float,
    reproduced: bool,
    died: bool,
    reproduction_ready_after: bool,
    runtime_species_id: int | None,
    runtime_ecotype_id: int | None,
) -> dict[str, object]:
    action_valid = bool(action_mask.get(requested_action, False))
    resolution_action_valid = bool(resolution_action_mask.get(requested_action, False))
    outcome = complete_action_outcome(
        action_outcome,
        resource_gain=resource_gain,
        reproduced=reproduced,
        died=died,
        reproduction_ready_after=reproduction_ready_after,
    )
    reward = build_reward(
        before=before,
        after=after,
        action_valid=resolution_action_valid,
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
        "observation_metadata": observation_metadata,
        "observation_input": observation_input,
        "observation_digest": observation_digest,
        "action_mask": {action: bool(action_mask[action]) for action in action_mask},
        "resolution_action_mask": {
            action: bool(resolution_action_mask[action]) for action in resolution_action_mask
        },
        "requested_action": requested_action,
        "action_source": action_source,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "action_valid": action_valid,
        "resolution_action_valid": resolution_action_valid,
        "resolved_action": resolved_action,
        "moved": moved,
        "before": before,
        "after": after,
        "outcome": outcome,
        "reward": reward,
    }


def empty_action_outcome(
    *,
    requested_action: str,
    resolved_action: str,
    observation_action_valid: bool,
    resolution_action_valid: bool,
    invalid_reason: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": ACTION_OUTCOME_SCHEMA_VERSION,
        "requested_action": requested_action,
        "resolved_action": resolved_action,
        "observation_action_valid": observation_action_valid,
        "resolution_action_valid": resolution_action_valid,
        "invalid_reason": invalid_reason,
        "movement": {"moved": False},
        "attack": {"attempted": False},
        "feeding": {"ate": False},
        "drinking": {"drank": False},
        "signal": _empty_signal_outcome(),
        "passive": {
            "acted": True,
            "damage_taken": 0.0,
            "attack_damage_taken": 0.0,
            "hazard_damage_taken": 0.0,
            "killed": False,
            "death_cause": None,
            "killer_id": None,
            "died_before_action": False,
            "died_after_action": False,
        },
        "resource_gain": 0.0,
        "reproduced": False,
        "died": False,
        "reproduction_ready_after": False,
    }


def complete_action_outcome(
    action_outcome: dict[str, object],
    *,
    resource_gain: float,
    reproduced: bool,
    died: bool,
    reproduction_ready_after: bool,
) -> dict[str, object]:
    outcome = dict(action_outcome)
    outcome["schema_version"] = ACTION_OUTCOME_SCHEMA_VERSION
    outcome.setdefault("movement", {"moved": False})
    outcome.setdefault("attack", {"attempted": False})
    outcome.setdefault("feeding", {"ate": False})
    outcome.setdefault("drinking", {"drank": False})
    outcome["signal"] = _complete_signal_outcome(outcome.get("signal"))
    outcome.setdefault(
        "passive",
        {
            "acted": True,
            "damage_taken": 0.0,
            "attack_damage_taken": 0.0,
            "hazard_damage_taken": 0.0,
            "killed": False,
            "death_cause": None,
            "killer_id": None,
            "died_before_action": False,
            "died_after_action": False,
        },
    )
    outcome["resource_gain"] = _round(resource_gain)
    outcome["reproduced"] = reproduced
    outcome["died"] = died
    outcome["reproduction_ready_after"] = reproduction_ready_after
    return outcome


def _empty_signal_outcome() -> dict[str, object]:
    return {
        "emitted": False,
        "token_id": None,
        "profile_index": None,
        "intensity": 0.0,
        "radius": 0,
        "duration_ticks": 0,
        "decay_rate": 0.0,
        "energy_cost": 0.0,
        "invalid_reason": None,
    }


def _complete_signal_outcome(signal: object) -> dict[str, object]:
    complete = _empty_signal_outcome()
    if isinstance(signal, dict):
        complete.update(signal)
    return complete


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
    bounded_components = {
        name: _clamp_to_bounds(name, float(value))
        for name, value in components.items()
    }
    return {
        "schema_version": REWARD_SCHEMA_VERSION,
        "components": bounded_components,
        "total": _clamp_total(
            _round(sum(float(value) for value in bounded_components.values()))
        ),
    }


def build_trajectory_payload(
    records: list[dict[str, object]],
    *,
    signal_config: Any | None = None,
) -> dict[str, object]:
    stats = empty_trajectory_stats()
    for record in records:
        update_trajectory_stats(stats, record)
    return {
        **trajectory_contract(signal_config),
        **build_trajectory_summary_from_stats(stats),
        "records": records,
    }


def build_trajectory_summary(
    records: list[dict[str, object]],
    *,
    signal_config: Any | None = None,
) -> dict[str, object]:
    stats = empty_trajectory_stats()
    for record in records:
        update_trajectory_stats(stats, record)
    payload = {
        **trajectory_contract(signal_config),
        **build_trajectory_summary_from_stats(stats),
    }
    return {
        "schema_version": payload["schema_version"],
        "observation_schema_version": payload["observation_schema_version"],
        "policy_interface_version": payload["policy_interface_version"],
        "action_contract_version": payload["action_contract_version"],
        "reproductive_group_contract_version": payload[
            "reproductive_group_contract_version"
        ],
        "genome_recombination_contract_version": payload[
            "genome_recombination_contract_version"
        ],
        "reward_schema_version": payload["reward_schema_version"],
        "action_outcome_schema_version": payload["action_outcome_schema_version"],
        "record_count": payload["record_count"],
        "invalid_action_count": payload["invalid_action_count"],
        "action_counts": payload["action_counts"],
        "mean_reward": payload["mean_reward"],
    }


def empty_trajectory_stats() -> dict[str, object]:
    return {
        "record_count": 0,
        "invalid_action_count": 0,
        "action_counts": Counter(),
        "total_reward": 0.0,
    }


def update_trajectory_stats(
    stats: dict[str, object],
    record: dict[str, object],
) -> None:
    stats["record_count"] = int(stats["record_count"]) + 1
    if not bool(record["action_valid"]):
        stats["invalid_action_count"] = int(stats["invalid_action_count"]) + 1
    action_counts = stats["action_counts"]
    if not isinstance(action_counts, Counter):
        raise TypeError("trajectory stats action_counts must be a Counter")
    action_counts[str(record["requested_action"])] += 1
    reward = record.get("reward")
    if not isinstance(reward, dict):
        raise ValueError("trajectory record is missing reward payload")
    stats["total_reward"] = float(stats["total_reward"]) + float(reward["total"])


def build_trajectory_summary_from_stats(stats: dict[str, object]) -> dict[str, object]:
    record_count = int(stats["record_count"])
    action_counts = stats["action_counts"]
    if not isinstance(action_counts, Counter):
        raise TypeError("trajectory stats action_counts must be a Counter")
    total_reward = float(stats["total_reward"])
    return {
        "record_count": record_count,
        "invalid_action_count": int(stats["invalid_action_count"]),
        "action_counts": {
            action: action_counts[action] for action in sorted(action_counts)
        },
        "mean_reward": _round(total_reward / record_count) if record_count else 0.0,
    }


def _round(value: float) -> float:
    return round(float(value), 4)


def _clamp_to_bounds(name: str, value: float) -> float:
    lower, upper = REWARD_COMPONENT_BOUNDS[name]
    return _round(min(upper, max(lower, value)))


def _clamp_total(value: float) -> float:
    lower, upper = REWARD_TOTAL_BOUNDS
    return _round(min(upper, max(lower, value)))
