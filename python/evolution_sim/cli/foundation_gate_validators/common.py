from __future__ import annotations

from collections.abc import Mapping


def _flag(severity: str, scope: str, field: str, message: str) -> dict[str, object]:
    return {
        "severity": severity,
        "scope": scope,
        "field": field,
        "message": message,
    }


def _as_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_int_count_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        parsed_count = _as_optional_int(raw_count)
        if parsed_count is not None:
            counts[str(key)] = parsed_count
    return counts
