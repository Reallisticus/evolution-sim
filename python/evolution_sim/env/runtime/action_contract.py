from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evolution_sim.config.schema import SignalConfig
import evolution_sim.env.runtime.signals as runtime_signals

ACTION_CONTRACT_VERSION = "mind_action_contract_v1"
ACTION_MASK_CONTRACT_VERSION = "mind_action_mask_semantics_v1"
ACTION_MASK_SEMANTICS_POLICY = (
    "resolution_affordance_mask_not_pure_physical_legality_v1"
)

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
        "active_action_keys": [spec.key for spec in specs if spec.active],
        "reserved_action_keys": list(reserved_action_names(signal_config)),
        "mate_action_key": MATE_ACTION,
        "communication": {
            "token_count": token_count,
            "profiles_per_token": profiles_per_token,
            "action_keys": list(communication_actions),
            "emission_enabled": _communication_emission_enabled(signal_config),
            "meaning": "simulator_opaque",
        },
        "actions": [spec.to_dict() for spec in specs],
    }


def action_mask_contract() -> dict[str, object]:
    return {
        "schema_version": ACTION_MASK_CONTRACT_VERSION,
        "policy": ACTION_MASK_SEMANTICS_POLICY,
        "mask_role": "resolution_affordance_mask",
        "pure_physical_legality": False,
        "description": (
            "The action mask is the Foundation-provided set of actions the "
            "resolver is prepared to accept for the current decision. It mixes "
            "physical reachability with resource-usefulness and biological "
            "condition gates."
        ),
        "action_family_semantics": {
            "eat": {
                "mask_basis": "utility_shaped_intake_affordance",
                "pure_physical_legality": False,
                "uses_intake_usefulness": True,
                "uses_resource_value": True,
                "notes": (
                    "eat is enabled only when at least one plant, fresh-kill, "
                    "or carcass intake path is currently useful and has "
                    "positive/currently actionable resource value."
                ),
            },
            "drink": {
                "mask_basis": "water_access_affordance",
                "pure_physical_legality": False,
                "notes": (
                    "drink is enabled from the water-access affordance exposed "
                    "by Foundation hydrology and tile state."
                ),
            },
            "movement": {
                "mask_basis": "physical_resolution_legality",
                "closer_to_physical_legality": True,
                "notes": (
                    "move_* entries are closest to physical legality: bounds, "
                    "terrain, occupancy, and movement-resolution constraints."
                ),
            },
            "attack": {
                "mask_basis": "physical_adjacency_plus_biological_condition_gates",
                "pure_physical_legality": False,
                "uses_biological_condition_gate": True,
                "notes": (
                    "attack_* requires the directional physical target "
                    "affordance and attack biology/condition gates."
                ),
            },
            "reserved": {
                "mask_basis": "future_or_opt_in_action_slots",
                "notes": (
                    "reserved actions stay in the stable action id space but "
                    "are false unless their feature gate is enabled."
                ),
            },
        },
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
    return _communication_emission_enabled(signal_config)


def _communication_emission_enabled(signal_config: Any | None) -> bool:
    if signal_config is None:
        return False
    return runtime_signals.communication_signal_emission_enabled(signal_config)
