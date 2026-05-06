from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evolution_sim.env.runtime.action_space import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    MEAT_MODE_VOCAB,
    NAVIGATION_FIELDS,
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_FIELDS,
    PATCH_INPUT_FIELDS,
    SELF_FIELDS,
    SELF_INPUT_FIELDS,
    TROPHIC_ROLE_VOCAB,
    decode_observation_input,
)

FEATURE_POLICY_VERSION = "mind_feature_policy_v2"
MOVE_ACTIONS: tuple[str, ...] = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
)
ATTACK_ACTIONS: tuple[str, ...] = (
    "attack_north",
    "attack_south",
    "attack_east",
    "attack_west",
)
CENTER_PATCH_INDEX = PATCH_CELL_COUNT // 2
SELF_FIELD_INDEX = {field: index for index, field in enumerate(SELF_INPUT_FIELDS)}
PATCH_FIELD_INDEX = {field: index for index, field in enumerate(PATCH_INPUT_FIELDS)}
NAVIGATION_FIELD_INDEX = {
    field: index for index, field in enumerate(NAVIGATION_INPUT_FIELDS)
}
PATCH_INPUT_START = len(SELF_INPUT_FIELDS)
NAVIGATION_INPUT_START = PATCH_INPUT_START + PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)


def feature_keys_from_record(record: Mapping[str, object]) -> tuple[str, ...]:
    action_mask = _mapping(record.get("action_mask"))
    observation_input = record.get("observation_input")
    if not isinstance(observation_input, dict):
        raise ValueError("trajectory record observation_input must be an object")
    values = decode_observation_input(observation_input)
    before = _mapping(record.get("before"))
    return _feature_keys(
        energy_ratio=_number(before.get("energy_ratio"), values[SELF_FIELD_INDEX["energy_ratio"]]),
        hydration_ratio=_number(
            before.get("hydration_ratio"),
            values[SELF_FIELD_INDEX["hydration_ratio"]],
        ),
        health_ratio=_number(before.get("health_ratio"), values[SELF_FIELD_INDEX["health_ratio"]]),
        trophic_role=_enum_index_from_code(
            values[SELF_FIELD_INDEX["trophic_role_code"]],
            len(TROPHIC_ROLE_VOCAB),
        ),
        meat_mode=_enum_index_from_code(
            values[SELF_FIELD_INDEX["meat_mode_code"]],
            len(MEAT_MODE_VOCAB),
        ),
        center_food=_center_patch_value(values, "food"),
        center_fresh_kill=_center_patch_value(values, "fresh_kill_energy"),
        center_carcass=_center_patch_value(values, "carcass_energy"),
        local_patch=_local_patch_tokens_from_values(values),
        navigation=_navigation_tokens_from_values(values),
        action_mask=action_mask,
    )


def feature_keys_from_observation(
    observation: Mapping[str, object],
    action_mask: Mapping[str, bool],
) -> tuple[str, ...]:
    self_state = _mapping(observation.get("self"))
    center = _center_patch_cell(observation.get("local_patch"))
    return _feature_keys(
        energy_ratio=_number(self_state.get("energy_ratio"), 0.0),
        hydration_ratio=_number(self_state.get("hydration_ratio"), 0.0),
        health_ratio=_number(self_state.get("health_ratio"), 1.0),
        trophic_role=_enum_index_from_label(
            self_state.get("trophic_role"),
            TROPHIC_ROLE_VOCAB,
        ),
        meat_mode=_enum_index_from_label(self_state.get("meat_mode"), MEAT_MODE_VOCAB),
        center_food=_number(center.get("food"), 0.0),
        center_fresh_kill=_number(center.get("fresh_kill_energy"), 0.0),
        center_carcass=_number(center.get("carcass_energy"), 0.0),
        local_patch=_local_patch_tokens_from_observation(observation.get("local_patch")),
        navigation=_navigation_tokens_from_observation(observation.get("navigation")),
        action_mask=action_mask,
    )


def _feature_keys(
    *,
    energy_ratio: float,
    hydration_ratio: float,
    health_ratio: float,
    trophic_role: int,
    meat_mode: int,
    center_food: float,
    center_fresh_kill: float,
    center_carcass: float,
    local_patch: dict[str, str],
    navigation: dict[str, str],
    action_mask: Mapping[str, bool],
) -> tuple[str, ...]:
    vitals = (
        f"e{_unit_bucket(energy_ratio)}"
        f":h{_unit_bucket(hydration_ratio)}"
        f":hp{_unit_bucket(health_ratio)}"
    )
    role = f"r{trophic_role}:m{meat_mode}"
    resources = (
        f"f{_signal_bucket(center_food)}"
        f":fk{_signal_bucket(center_fresh_kill)}"
        f":ca{_signal_bucket(center_carcass)}"
    )
    local = ":".join(
        f"{name}{local_patch.get(name, 'c0')}"
        for name in ("food", "water", "carrion", "prey", "risk")
    )
    nav = ":".join(f"{target}{navigation.get(target, '0_0_0')}" for target in NAVIGATION_TARGETS)
    mask = _action_mask_token(action_mask)
    return (
        f"{FEATURE_POLICY_VERSION}|{vitals}|{role}|{resources}|{local}|{nav}|{mask}",
        f"{FEATURE_POLICY_VERSION}|{vitals}|{role}|{resources}|{nav}|{mask}",
        f"{FEATURE_POLICY_VERSION}|{vitals}|{role}|{resources}|{local}|{mask}",
        f"{FEATURE_POLICY_VERSION}|{vitals}|{role}|{resources}|{mask}",
        f"{FEATURE_POLICY_VERSION}|{vitals}|{role}|{mask}",
        f"{FEATURE_POLICY_VERSION}|{vitals}|{mask}",
        f"{FEATURE_POLICY_VERSION}|{mask}",
    )


def _action_mask_token(action_mask: Mapping[str, bool]) -> str:
    move_bits = "".join("1" if action_mask.get(action, False) else "0" for action in MOVE_ACTIONS)
    attack_bits = "".join(
        "1" if action_mask.get(action, False) else "0" for action in ATTACK_ACTIONS
    )
    return (
        f"a:e{int(bool(action_mask.get('eat', False)))}"
        f"d{int(bool(action_mask.get('drink', False)))}"
        f"s{int(bool(action_mask.get('stay', False)))}"
        f":m{move_bits}:x{attack_bits}"
    )


def _navigation_tokens_from_values(values: list[float]) -> dict[str, str]:
    tokens: dict[str, str] = {}
    stride = len(NAVIGATION_INPUT_FIELDS)
    for target_index, target in enumerate(NAVIGATION_TARGETS):
        base = NAVIGATION_INPUT_START + target_index * stride
        tokens[target] = _navigation_token(
            dx=values[base + NAVIGATION_FIELD_INDEX["dx"]],
            dy=values[base + NAVIGATION_FIELD_INDEX["dy"]],
            distance=values[base + NAVIGATION_FIELD_INDEX["distance"]],
            strength=values[base + NAVIGATION_FIELD_INDEX["strength"]],
        )
    return tokens


def _navigation_tokens_from_observation(payload: object) -> dict[str, str]:
    navigation = _mapping(payload)
    tokens: dict[str, str] = {}
    for target in NAVIGATION_TARGETS:
        target_payload = _mapping(navigation.get(target))
        tokens[target] = _navigation_token(
            dx=_number(target_payload.get(NAVIGATION_FIELDS[0]), 0.0),
            dy=_number(target_payload.get(NAVIGATION_FIELDS[1]), 0.0),
            distance=_number(target_payload.get(NAVIGATION_FIELDS[2]), 0.0),
            strength=_number(target_payload.get(NAVIGATION_FIELDS[3]), 0.0),
        )
    return tokens


def _navigation_token(*, dx: float, dy: float, distance: float, strength: float) -> str:
    direction = f"{_direction_bucket(dx)}{_direction_bucket(dy)}"
    return f"{direction}_{_unit_bucket(distance)}_{_signal_bucket(strength)}"


def _center_patch_value(values: list[float], field: str) -> float:
    center_base = PATCH_INPUT_START + CENTER_PATCH_INDEX * len(PATCH_INPUT_FIELDS)
    return values[center_base + PATCH_FIELD_INDEX[field]]


def _local_patch_tokens_from_values(values: list[float]) -> dict[str, str]:
    tokens = {
        "food": ("c", 0.0),
        "water": ("c", 0.0),
        "carrion": ("c", 0.0),
        "prey": ("c", 0.0),
        "risk": ("c", 0.0),
    }
    stride = len(PATCH_INPUT_FIELDS)
    for cell_index in range(PATCH_CELL_COUNT):
        base = PATCH_INPUT_START + cell_index * stride
        dx = values[base + PATCH_FIELD_INDEX["dx"]]
        dy = values[base + PATCH_FIELD_INDEX["dy"]]
        if abs(dx) + abs(dy) > 1.01:
            continue
        direction = _local_direction_token(dx=dx, dy=dy)
        _update_patch_token(
            tokens,
            "food",
            direction,
            values[base + PATCH_FIELD_INDEX["food"]],
        )
        water_strength = max(
            values[base + PATCH_FIELD_INDEX["water_access_reason_code"]],
            1.0 if values[base + PATCH_FIELD_INDEX["terrain_code"]] >= 0.99 else 0.0,
        )
        _update_patch_token(tokens, "water", direction, water_strength)
        _update_patch_token(
            tokens,
            "carrion",
            direction,
            max(
                values[base + PATCH_FIELD_INDEX["fresh_kill_energy"]],
                values[base + PATCH_FIELD_INDEX["carcass_energy"]],
                values[base + PATCH_FIELD_INDEX["carrion_signal"]],
            ),
        )
        _update_patch_token(
            tokens,
            "prey",
            direction,
            values[base + PATCH_FIELD_INDEX["prey_biomass"]],
        )
        _update_patch_token(
            tokens,
            "risk",
            direction,
            max(
                values[base + PATCH_FIELD_INDEX["hazard_level"]],
                values[base + PATCH_FIELD_INDEX["predator_risk"]],
            ),
        )
    return {
        name: f"{direction}{_signal_bucket(value)}"
        for name, (direction, value) in tokens.items()
    }


def _local_patch_tokens_from_observation(payload: object) -> dict[str, str]:
    tokens = {
        "food": ("c", 0.0),
        "water": ("c", 0.0),
        "carrion": ("c", 0.0),
        "prey": ("c", 0.0),
        "risk": ("c", 0.0),
    }
    if not isinstance(payload, list):
        return {
            name: f"{direction}{_signal_bucket(value)}"
            for name, (direction, value) in tokens.items()
        }
    for cell in payload:
        if not isinstance(cell, Mapping):
            continue
        dx = _number(cell.get("dx"), 0.0)
        dy = _number(cell.get("dy"), 0.0)
        if abs(dx) + abs(dy) > 1.01:
            continue
        direction = _local_direction_token(dx=dx, dy=dy)
        _update_patch_token(tokens, "food", direction, _number(cell.get("food"), 0.0))
        water_reason = str(cell.get("water_access_reason", "none"))
        water_strength = 1.0 if water_reason != "none" else 0.0
        if str(cell.get("terrain", "plain")) == "water":
            water_strength = 1.0
        _update_patch_token(tokens, "water", direction, water_strength)
        _update_patch_token(
            tokens,
            "carrion",
            direction,
            max(
                _number(cell.get("fresh_kill_energy"), 0.0),
                _number(cell.get("carcass_energy"), 0.0),
                _number(cell.get("carrion_signal"), 0.0),
            ),
        )
        _update_patch_token(
            tokens,
            "prey",
            direction,
            _number(cell.get("prey_biomass"), 0.0),
        )
        _update_patch_token(
            tokens,
            "risk",
            direction,
            max(
                _number(cell.get("hazard_level"), 0.0),
                _number(cell.get("predator_risk"), 0.0),
            ),
        )
    return {
        name: f"{direction}{_signal_bucket(value)}"
        for name, (direction, value) in tokens.items()
    }


def _update_patch_token(
    tokens: dict[str, tuple[str, float]],
    name: str,
    direction: str,
    value: float,
) -> None:
    current_direction, current_value = tokens[name]
    if value <= 0.0 and current_value <= 0.0:
        return
    if (value, _direction_rank(direction)) > (
        current_value,
        _direction_rank(current_direction),
    ):
        tokens[name] = (direction, value)


def _local_direction_token(*, dx: float, dy: float) -> str:
    if abs(dx) <= 0.05 and abs(dy) <= 0.05:
        return "c"
    if abs(dx) >= abs(dy):
        return "e" if dx > 0.0 else "w"
    return "s" if dy > 0.0 else "n"


def _direction_rank(direction: str) -> int:
    return {"c": 0, "n": 1, "s": 2, "e": 3, "w": 4}.get(direction, 0)


def _center_patch_cell(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, list):
        return {}
    if len(payload) > CENTER_PATCH_INDEX and isinstance(payload[CENTER_PATCH_INDEX], dict):
        return payload[CENTER_PATCH_INDEX]
    for cell in payload:
        if not isinstance(cell, dict):
            continue
        if cell.get(PATCH_FIELDS[0]) == 0 and cell.get(PATCH_FIELDS[1]) == 0:
            return cell
    return {}


def _mapping(payload: object) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    return {}


def _number(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(default)
    return float(value)


def _unit_bucket(value: float) -> int:
    if value < 0.28:
        return 0
    if value < 0.56:
        return 1
    if value < 0.82:
        return 2
    return 3


def _signal_bucket(value: float) -> int:
    if value <= 0.0:
        return 0
    if value < 0.08:
        return 1
    if value < 0.24:
        return 2
    return 3


def _direction_bucket(value: float) -> str:
    if value < -0.05:
        return "n"
    if value > 0.05:
        return "p"
    return "z"


def _enum_index_from_label(value: object, vocab: tuple[str, ...]) -> int:
    try:
        return vocab.index(str(value))
    except ValueError:
        return 0


def _enum_index_from_code(value: float, vocab_size: int) -> int:
    if vocab_size <= 1:
        return 0
    return max(0, min(vocab_size - 1, int(round(value * (vocab_size - 1)))))
