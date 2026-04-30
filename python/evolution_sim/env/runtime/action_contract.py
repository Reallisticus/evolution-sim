from __future__ import annotations

from dataclasses import asdict, dataclass

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
COMMUNICATION_TOKEN_COUNT = 4
COMMUNICATION_PROFILES_PER_TOKEN = 2
COMMUNICATION_ACTIONS: tuple[str, ...] = tuple(
    f"signal_{token_index}_profile_{profile_index}"
    for token_index in range(COMMUNICATION_TOKEN_COUNT)
    for profile_index in range(COMMUNICATION_PROFILES_PER_TOKEN)
)
ACTIVE_ACTION_NAMES: tuple[str, ...] = (
    *CORE_ACTIONS,
    *MOVEMENT_ACTIONS,
    *ATTACK_ACTIONS,
)
RESERVED_ACTION_NAMES: tuple[str, ...] = (MATE_ACTION, *COMMUNICATION_ACTIONS)
ACTION_NAMES: tuple[str, ...] = (*ACTIVE_ACTION_NAMES, *RESERVED_ACTION_NAMES)


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


def action_specs() -> tuple[ActionSpec, ...]:
    specs: list[ActionSpec] = []
    for action_id, key in enumerate(ACTION_NAMES):
        category = _category_for_key(key)
        reserved = key in RESERVED_ACTION_NAMES
        specs.append(
            ActionSpec(
                action_id=action_id,
                key=key,
                resolver=_resolver_for_key(key),
                category=category,
                active=not reserved,
                reserved=reserved,
                requires_biology_gate=reserved,
                policy_visible=True,
                debug_label=key,
            )
        )
    return tuple(specs)


def action_contract() -> dict[str, object]:
    specs = action_specs()
    return {
        "schema_version": ACTION_CONTRACT_VERSION,
        "policy_id_encoding": "zero_based_action_id",
        "debug_key_encoding": "stable_string_key",
        "active_action_keys": list(ACTIVE_ACTION_NAMES),
        "reserved_action_keys": list(RESERVED_ACTION_NAMES),
        "mate_action_key": MATE_ACTION,
        "communication": {
            "token_count": COMMUNICATION_TOKEN_COUNT,
            "profiles_per_token": COMMUNICATION_PROFILES_PER_TOKEN,
            "action_keys": list(COMMUNICATION_ACTIONS),
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
