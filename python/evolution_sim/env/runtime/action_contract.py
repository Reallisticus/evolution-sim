from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evolution_sim.config.schema import SignalConfig

ACTION_CONTRACT_VERSION = "mind_action_contract_v1"

MOVEMENT_ACTIONS: tuple[str, ...] = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
)
ATTACK_ACTIONS: tuple[str, ...] = tuple(
    action.replace("move_", "attack_") for action in MOVEMENT_ACTIONS
)
CORE_ACTIONS: tuple[str, ...] = ("stay", "eat", "drink")
MATE_ACTION: str = "mate"
_DEFAULT_SIGNAL_CONFIG = SignalConfig()
COMMUNICATION_TOKEN_COUNT = int(_DEFAULT_SIGNAL_CONFIG.communication_token_count)
COMMUNICATION_PROFILES_PER_TOKEN = int(
    _DEFAULT_SIGNAL_CONFIG.communication_profiles_per_token
)
ACTIVE_ACTION_NAMES: tuple[str, ...] = (
    *CORE_ACTIONS,
    *MOVEMENT_ACTIONS,
    *ATTACK_ACTIONS,
)


def communication_action_names(signal_config: Any | None = None) -> tuple[str, ...]:
    token_count, profiles_per_token = _communication_counts(signal_config)
    return tuple(
        f"signal_{token_index}_profile_{profile_index}"
        for token_index in range(token_count)
        for profile_index in range(profiles_per_token)
    )


def reserved_action_names(signal_config: Any | None = None) -> tuple[str, ...]:
    return (MATE_ACTION, *communication_action_names(signal_config))


def action_names(signal_config: Any | None = None) -> tuple[str, ...]:
    return (*ACTIVE_ACTION_NAMES, *reserved_action_names(signal_config))


def _communication_counts(signal_config: Any | None) -> tuple[int, int]:
    if signal_config is None:
        return COMMUNICATION_TOKEN_COUNT, COMMUNICATION_PROFILES_PER_TOKEN
    return (
        int(getattr(signal_config, "communication_token_count")),
        int(getattr(signal_config, "communication_profiles_per_token")),
    )


COMMUNICATION_ACTIONS: tuple[str, ...] = communication_action_names()
RESERVED_ACTION_NAMES: tuple[str, ...] = reserved_action_names()
ACTION_NAMES: tuple[str, ...] = action_names()


@dataclass(frozen=True, slots=True)
class ActionSpec:
    action_id: int
    key: str
    resolver: str
    category: str
    active: bool
    reserved: bool
    requires_biology_gate: bool
    policy_visible: bool
    debug_label: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def action_specs(signal_config: Any | None = None) -> tuple[ActionSpec, ...]:
    reserved_names = set(reserved_action_names(signal_config))
    specs: list[ActionSpec] = []
    for action_id, key in enumerate(action_names(signal_config)):
        category = _category_for_key(key)
        reserved = key in reserved_names
        active = _active_for_key(key, reserved=reserved, signal_config=signal_config)
        specs.append(
            ActionSpec(
                action_id=action_id,
                key=key,
                resolver=_resolver_for_key(key),
                category=category,
                active=active,
                reserved=reserved,
                requires_biology_gate=reserved,
                policy_visible=True,
                debug_label=key,
            )
        )
    return tuple(specs)


def action_contract(signal_config: Any | None = None) -> dict[str, object]:
    specs = action_specs(signal_config)
    communication_actions = communication_action_names(signal_config)
    token_count, profiles_per_token = _communication_counts(signal_config)
    return {
        "schema_version": ACTION_CONTRACT_VERSION,
        "policy_id_encoding": "zero_based_action_id",
        "debug_key_encoding": "stable_string_key",
        "active_action_keys": list(ACTIVE_ACTION_NAMES),
        "reserved_action_keys": list(reserved_action_names(signal_config)),
        "mate_action_key": MATE_ACTION,
        "communication": {
            "token_count": token_count,
            "profiles_per_token": profiles_per_token,
            "action_keys": list(communication_actions),
            "meaning": "simulator_opaque",
        },
        "actions": [spec.to_dict() for spec in specs],
    }


def _category_for_key(key: str) -> str:
    if key.startswith("move_"):
        return "movement"
    if key.startswith("attack_"):
        return "attack"
    if key == MATE_ACTION:
        return "reproduction"
    if key.startswith("signal_"):
        return "communication"
    return "core"


def _resolver_for_key(key: str) -> str:
    if key in ACTIVE_ACTION_NAMES:
        return "current_world_resolver"
    if key == MATE_ACTION:
        return "reserved_mate_attempt"
    if key.startswith("signal_"):
        return "reserved_signal_emission"
    return "unknown"


def _active_for_key(
    key: str,
    *,
    reserved: bool,
    signal_config: Any | None,
) -> bool:
    if not reserved:
        return True
    if not key.startswith("signal_") or signal_config is None:
        return False
    return bool(getattr(signal_config, "communication_signal_emission_enabled", False))
