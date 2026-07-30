from __future__ import annotations

import math
from typing import Sequence

from evolution_sim.env.runtime.observations import (
    NAVIGATION_FIELDS,
    NAVIGATION_TARGETS,
    OBSERVATION_INPUT_VECTOR_SIZE,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
    decode_observation_input,
    observation_input_vector_size,
    observation_schema_version,
    quantized_observation_input_values,
)
from evolution_sim.env.runtime.signals import communication_token_field_names

ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION = "mind_ecological_policy_input_v1"
TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION = "mind_ecological_policy_input_v3"
ECOLOGICAL_POLICY_INPUT_POLICY = "exclude_controller_diagnostics_v1"
ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION = (
    "validated_fused_quantized_diagnostic_filter_v1"
)
CONTROLLER_DIAGNOSTIC_SELF_FIELDS: tuple[str, ...] = ("mind_inheritance_available",)
CONTROLLER_DIAGNOSTIC_INPUT_FIELDS: tuple[str, ...] = tuple(
    f"self.{field}" for field in CONTROLLER_DIAGNOSTIC_SELF_FIELDS
)
_DIAGNOSTIC_SELF_INDICES = frozenset(
    SELF_INPUT_FIELDS.index(field) for field in CONTROLLER_DIAGNOSTIC_SELF_FIELDS
)
ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE = OBSERVATION_INPUT_VECTOR_SIZE - len(
    _DIAGNOSTIC_SELF_INDICES
)


class PolicyInputError(ValueError):
    pass


def ecological_policy_input_schema_version(signal_config: object | None = None) -> str:
    if (
        observation_schema_version(signal_config)
        == TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION
    ):
        return TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
    return ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION


def ecological_policy_input_vector_size(signal_config: object | None = None) -> int:
    return observation_input_vector_size(signal_config) - len(_DIAGNOSTIC_SELF_INDICES)


def ecological_policy_input_contract(
    signal_config: object | None = None,
) -> dict[str, object]:
    token_fields = communication_token_field_names(signal_config)
    return {
        "schema_version": ecological_policy_input_schema_version(signal_config),
        "policy": ECOLOGICAL_POLICY_INPUT_POLICY,
        "source_observation_schema_version": observation_schema_version(signal_config),
        "source_vector_size": observation_input_vector_size(signal_config),
        "ecological_vector_size": ecological_policy_input_vector_size(signal_config),
        "excluded_controller_diagnostic_fields": list(
            CONTROLLER_DIAGNOSTIC_INPUT_FIELDS
        ),
        "retained_sections": {
            "self_fields": [
                field
                for index, field in enumerate(SELF_INPUT_FIELDS)
                if index not in _DIAGNOSTIC_SELF_INDICES
            ]
            + list(token_fields),
            "local_patch_cell_count": PATCH_CELL_COUNT,
            "local_patch_fields": [*PATCH_INPUT_FIELDS, *token_fields],
            "navigation_targets": list(NAVIGATION_TARGETS),
            "navigation_fields": list(NAVIGATION_FIELDS),
        },
    }


def ecological_policy_input_values(
    observation_input: dict[str, object],
) -> tuple[float, ...]:
    source_vector_size = _source_vector_size(observation_input)
    return ecological_policy_values_from_decoded(
        decode_observation_input(observation_input),
        source_vector_size=source_vector_size,
    )


def ecological_policy_values_from_observation(
    observation: dict[str, object],
) -> tuple[float, ...]:
    return tuple(
        quantized_observation_input_values(
            observation,
            excluded_indices=_DIAGNOSTIC_SELF_INDICES,
        )
    )


def ecological_policy_values_from_decoded(
    values: Sequence[float],
    *,
    source_vector_size: int = OBSERVATION_INPUT_VECTOR_SIZE,
) -> tuple[float, ...]:
    if (
        isinstance(source_vector_size, bool)
        or not isinstance(source_vector_size, int)
        or source_vector_size < OBSERVATION_INPUT_VECTOR_SIZE
    ):
        raise PolicyInputError("source observation vector size is invalid")
    if len(values) != source_vector_size:
        raise PolicyInputError(
            "decoded observation input has unexpected vector size: "
            f"{len(values)}; expected {source_vector_size}"
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
    source_schema_version = observation_input.get("schema_version")
    schema_version = (
        TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
        if source_schema_version == TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION
        else ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
    )
    return {
        "schema_version": schema_version,
        "policy": ECOLOGICAL_POLICY_INPUT_POLICY,
        "values": list(values),
        "shape": [len(values)],
    }


def _source_vector_size(observation_input: dict[str, object]) -> int:
    shape = observation_input.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
    ):
        raise PolicyInputError("observation input shape must contain one integer")
    return shape[0]


def _finite_unit_interval_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyInputError("decoded observation input values must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PolicyInputError("decoded observation input values must be finite")
    if parsed < -1.0 or parsed > 1.0:
        raise PolicyInputError("decoded observation input values must be in [-1, 1]")
    return round(parsed, 6)
