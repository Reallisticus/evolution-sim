from __future__ import annotations

import math
from typing import Sequence

from evolution_sim.env.runtime.observations import (
    NAVIGATION_FIELDS,
    NAVIGATION_TARGETS,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
)

ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION = "mind_ecological_policy_input_v1"
ECOLOGICAL_POLICY_INPUT_POLICY = "exclude_controller_diagnostics_v1"
CONTROLLER_DIAGNOSTIC_SELF_FIELDS: tuple[str, ...] = (
    "mind_inheritance_available",
)
CONTROLLER_DIAGNOSTIC_INPUT_FIELDS: tuple[str, ...] = tuple(
    f"self.{field}" for field in CONTROLLER_DIAGNOSTIC_SELF_FIELDS
)
_DIAGNOSTIC_SELF_INDICES = frozenset(
    SELF_INPUT_FIELDS.index(field)
    for field in CONTROLLER_DIAGNOSTIC_SELF_FIELDS
)
ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE = (
    OBSERVATION_INPUT_VECTOR_SIZE - len(_DIAGNOSTIC_SELF_INDICES)
)


class PolicyInputError(ValueError):
    pass


def ecological_policy_input_contract() -> dict[str, object]:
    return {
        "schema_version": ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        "policy": ECOLOGICAL_POLICY_INPUT_POLICY,
        "source_observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "source_vector_size": OBSERVATION_INPUT_VECTOR_SIZE,
        "ecological_vector_size": ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        "excluded_controller_diagnostic_fields": list(
            CONTROLLER_DIAGNOSTIC_INPUT_FIELDS
        ),
        "retained_sections": {
            "self_fields": [
                field
                for index, field in enumerate(SELF_INPUT_FIELDS)
                if index not in _DIAGNOSTIC_SELF_INDICES
            ],
            "local_patch_cell_count": PATCH_CELL_COUNT,
            "local_patch_fields": list(PATCH_INPUT_FIELDS),
            "navigation_targets": list(NAVIGATION_TARGETS),
            "navigation_fields": list(NAVIGATION_FIELDS),
        },
    }


def ecological_policy_input_values(
    observation_input: dict[str, object],
) -> tuple[float, ...]:
    return ecological_policy_values_from_decoded(
        decode_observation_input(observation_input)
    )


def ecological_policy_values_from_decoded(
    values: Sequence[float],
) -> tuple[float, ...]:
    if len(values) != OBSERVATION_INPUT_VECTOR_SIZE:
        raise PolicyInputError(
            "decoded observation input has unexpected vector size: "
            f"{len(values)}"
        )
    parsed = [_finite_unit_interval_value(value) for value in values]
    return tuple(
        value
        for index, value in enumerate(parsed)
        if index not in _DIAGNOSTIC_SELF_INDICES
    )


def ecological_policy_input_payload(
    observation_input: dict[str, object],
) -> dict[str, object]:
    values = ecological_policy_input_values(observation_input)
    return {
        "schema_version": ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        "policy": ECOLOGICAL_POLICY_INPUT_POLICY,
        "values": list(values),
        "shape": [len(values)],
    }


def _finite_unit_interval_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyInputError("decoded observation input values must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PolicyInputError("decoded observation input values must be finite")
    if parsed < -1.0 or parsed > 1.0:
        raise PolicyInputError("decoded observation input values must be in [-1, 1]")
    return round(parsed, 6)
