from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def round_float(value: float) -> float:
    return round(float(value), 4)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_ready(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def safe_path_part(value: object) -> str:
    text = str(value).strip().replace("_", "-")
    safe = [
        character.lower()
        if character.isalnum() or character in {"-", "."}
        else "-"
        for character in text
    ]
    return "".join(safe).strip("-") or "run"


def heuristic_action_source_count(counts: Mapping[str, int]) -> int:
    return sum(count for source, count in counts.items() if "heuristic" in source)


def dominant_action_summary(counts: Mapping[str, int]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"action": None, "count": 0, "share": 0.0}
    action, count = max(
        sorted(counts.items()),
        key=lambda item: (int(item[1]), str(item[0])),
    )
    return {
        "action": str(action),
        "count": int(count),
        "share": round_float(int(count) / float(total)),
    }


def dominant_count_key(counts: Mapping[str, int]) -> str | None:
    positive = {key: count for key, count in counts.items() if int(count) > 0}
    if not positive:
        return None
    key, _ = max(
        sorted(positive.items()),
        key=lambda item: (int(item[1]), str(item[0])),
    )
    return str(key)


def int_counter(value: object) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counts
    for key, count in value.items():
        counts[str(key)] += int(count)
    return counts


def share(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round_float(numerator / float(denominator))


def mean(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return round_float(sum(values) / float(len(values)))


def comparison_delta(
    *,
    heuristic: Mapping[str, object],
    mind_v3: Mapping[str, object],
) -> dict[str, object]:
    heuristic_attr = dict(heuristic["reproduction_failure_attribution"])
    mind_v3_attr = dict(mind_v3["reproduction_failure_attribution"])
    return {
        "alive_agents_mean": round_float(
            float(mind_v3["alive_agents_mean"])
            - float(heuristic["alive_agents_mean"])
        ),
        "births_mean": round_float(
            float(mind_v3["births_mean"]) - float(heuristic["births_mean"])
        ),
        "biologically_ready_agents_mean": round_float(
            float(mind_v3_attr["biologically_ready_agents_mean"])
            - float(heuristic_attr["biologically_ready_agents_mean"])
        ),
        "ready_agents_mean": round_float(
            float(mind_v3_attr["ready_agents_mean"])
            - float(heuristic_attr["ready_agents_mean"])
        ),
    }
