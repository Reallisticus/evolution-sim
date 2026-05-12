from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from evolution_sim.mind.horizon_labels import MIND_HORIZON_LABEL_SCHEMA_VERSION

MIND_V3_TEMPORAL_CREDIT_AUDIT_SCHEMA_VERSION = (
    "mind_v3_temporal_credit_audit_v1"
)
MIND_V3_TEMPORAL_CREDIT_AUDIT_POLICY = (
    "long_horizon_label_positive_support_gate_v1"
)


class TemporalCreditAuditError(ValueError):
    pass


def load_temporal_credit_json_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    with _open_input(resolved) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TemporalCreditAuditError(f"report must be a JSON object: {resolved}")
    return payload


def build_temporal_credit_audit_report(
    horizon_label_report: Mapping[str, object],
    *,
    primary_horizon: int = 120,
    comparison_horizon: int | None = 80,
    min_primary_survivor_count: int = 1,
    min_primary_post_contact_survivor_count: int = 1,
    min_primary_survival_rate: float = 0.0,
    max_primary_censored_share: float = 0.25,
    autopsy_report: Mapping[str, object] | None = None,
    fixture_label_report: Mapping[str, object] | None = None,
    horizon_label_report_path: str | Path | None = None,
    autopsy_report_path: str | Path | None = None,
    fixture_label_report_path: str | Path | None = None,
) -> dict[str, object]:
    _validate_horizon_label_report(horizon_label_report)
    primary = _positive_int(primary_horizon, field="primary_horizon")
    comparison = (
        _positive_int(comparison_horizon, field="comparison_horizon")
        if comparison_horizon is not None
        else None
    )
    min_survivors = _nonnegative_int(
        min_primary_survivor_count,
        field="min_primary_survivor_count",
    )
    min_post_contact_survivors = _nonnegative_int(
        min_primary_post_contact_survivor_count,
        field="min_primary_post_contact_survivor_count",
    )
    min_survival_rate = _nonnegative_float(
        min_primary_survival_rate,
        field="min_primary_survival_rate",
    )
    max_censored_share = _unit_float(
        max_primary_censored_share,
        field="max_primary_censored_share",
    )
    labels = _horizon_labels(horizon_label_report)
    available_horizons = _available_horizons(horizon_label_report, labels)
    if str(primary) not in available_horizons:
        raise TemporalCreditAuditError(
            f"primary horizon {primary} is not present in horizon labels"
        )
    if comparison is not None and str(comparison) not in available_horizons:
        raise TemporalCreditAuditError(
            f"comparison horizon {comparison} is not present in horizon labels"
        )

    summaries = {
        horizon: _horizon_summary(labels, horizon)
        for horizon in sorted(available_horizons, key=lambda value: int(value))
    }
    primary_summary = summaries[str(primary)]
    comparison_summary = (
        summaries[str(comparison)] if comparison is not None else None
    )
    contract = {
        "schema_version": MIND_V3_TEMPORAL_CREDIT_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_TEMPORAL_CREDIT_AUDIT_POLICY,
        "primary_horizon": primary,
        "comparison_horizon": comparison,
        "min_primary_survivor_count": min_survivors,
        "min_primary_post_contact_survivor_count": min_post_contact_survivors,
        "min_primary_survival_rate": min_survival_rate,
        "max_primary_censored_share": max_censored_share,
    }
    blockers = _readiness_blockers(
        primary_summary,
        min_survivors=min_survivors,
        min_post_contact_survivors=min_post_contact_survivors,
        min_survival_rate=min_survival_rate,
        max_censored_share=max_censored_share,
    )
    return {
        "schema_version": MIND_V3_TEMPORAL_CREDIT_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_TEMPORAL_CREDIT_AUDIT_POLICY,
        "audit_contract": contract,
        "source": {
            "horizon_label_report_path": (
                str(horizon_label_report_path)
                if horizon_label_report_path is not None
                else None
            ),
            "horizon_label_schema_version": horizon_label_report.get(
                "schema_version"
            ),
            "horizon_label_count": len(labels),
            "autopsy_report_path": (
                str(autopsy_report_path) if autopsy_report_path is not None else None
            ),
            "fixture_label_report_path": (
                str(fixture_label_report_path)
                if fixture_label_report_path is not None
                else None
            ),
        },
        "readiness": {
            "ready": not blockers,
            "blockers": blockers,
        },
        "primary_horizon": str(primary),
        "comparison_horizon": str(comparison) if comparison is not None else None,
        "horizons": summaries,
        "transition": _transition_summary(
            primary_summary=primary_summary,
            comparison_summary=comparison_summary,
        ),
        "primary_action_support": _action_support(labels, str(primary)),
        "autopsy_summary": _autopsy_summary(autopsy_report),
        "fixture_pressure_summary": _fixture_pressure_summary(fixture_label_report),
    }


def write_temporal_credit_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _validate_horizon_label_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != MIND_HORIZON_LABEL_SCHEMA_VERSION:
        raise TemporalCreditAuditError("horizon labels have stale schema_version")
    labels = report.get("labels")
    if not isinstance(labels, list) or not labels:
        raise TemporalCreditAuditError("horizon label report must include labels")


def _horizon_labels(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        label
        for label in list(report.get("labels", []))
        if isinstance(label, Mapping)
    ]


def _available_horizons(
    report: Mapping[str, object],
    labels: Sequence[Mapping[str, object]],
) -> set[str]:
    contract = report.get("label_contract")
    ticks = contract.get("horizon_ticks") if isinstance(contract, Mapping) else None
    horizons = {
        str(tick)
        for tick in list(ticks) if isinstance(tick, int) and not isinstance(tick, bool)
    }
    for label in labels:
        payload = label.get("horizons")
        if isinstance(payload, Mapping):
            horizons.update(str(key) for key in payload.keys())
    return horizons


def _horizon_summary(
    labels: Sequence[Mapping[str, object]],
    horizon: str,
) -> dict[str, object]:
    record_count = 0
    observed_count = 0
    survivor_count = 0
    death_count = 0
    reproduced_count = 0
    animal_contact_count = 0
    post_contact_survivor_count = 0
    post_contact_death_count = 0
    balanced_values: list[float] = []
    for label in labels:
        payload = _horizon_payload(label, horizon)
        if payload is None:
            continue
        record_count += 1
        if payload.get("observed") is not True:
            continue
        observed_count += 1
        if payload.get("survived") is True:
            survivor_count += 1
        elif payload.get("survived") is False:
            death_count += 1
        if payload.get("reproduced") is True:
            reproduced_count += 1
        viability = payload.get("viability")
        if isinstance(viability, Mapping):
            balanced = _optional_float(viability.get("balanced_core_min"))
            if balanced is not None:
                balanced_values.append(balanced)
        animal = payload.get("animal_resource")
        if not isinstance(animal, Mapping):
            continue
        if animal.get("animal_resource_consumed") is not True:
            continue
        animal_contact_count += 1
        if animal.get("survived_after_first_contact") is True:
            post_contact_survivor_count += 1
        elif animal.get("survived_after_first_contact") is False:
            post_contact_death_count += 1
    censored_count = record_count - observed_count
    return {
        "record_count": record_count,
        "observed_count": observed_count,
        "censored_count": censored_count,
        "censored_share": _round(_safe_rate(censored_count, record_count)),
        "survivor_count": survivor_count,
        "death_count": death_count,
        "survival_rate": _round(_safe_rate(survivor_count, observed_count)),
        "reproduced_count": reproduced_count,
        "reproduction_rate": _round(_safe_rate(reproduced_count, observed_count)),
        "animal_resource_contact_count": animal_contact_count,
        "post_contact_survivor_count": post_contact_survivor_count,
        "post_contact_death_count": post_contact_death_count,
        "post_contact_survival_rate": _round(
            _safe_rate(post_contact_survivor_count, animal_contact_count)
        ),
        "mean_balanced_core_min": _round(
            sum(balanced_values) / float(len(balanced_values))
            if balanced_values
            else 0.0
        ),
    }


def _readiness_blockers(
    primary_summary: Mapping[str, object],
    *,
    min_survivors: int,
    min_post_contact_survivors: int,
    min_survival_rate: float,
    max_censored_share: float,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    _append_floor_blocker(
        blockers,
        reason="primary_survivor_count_floor",
        metric="survivor_count",
        value=float(primary_summary["survivor_count"]),
        floor=float(min_survivors),
    )
    _append_floor_blocker(
        blockers,
        reason="primary_post_contact_survivor_count_floor",
        metric="post_contact_survivor_count",
        value=float(primary_summary["post_contact_survivor_count"]),
        floor=float(min_post_contact_survivors),
    )
    _append_floor_blocker(
        blockers,
        reason="primary_survival_rate_floor",
        metric="survival_rate",
        value=float(primary_summary["survival_rate"]),
        floor=float(min_survival_rate),
    )
    censored_share = float(primary_summary["censored_share"])
    if censored_share > max_censored_share:
        blockers.append(
            {
                "reason": "primary_censored_share_ceiling",
                "metric": "censored_share",
                "value": _round(censored_share),
                "ceiling": _round(max_censored_share),
            }
        )
    return blockers


def _append_floor_blocker(
    blockers: list[dict[str, object]],
    *,
    reason: str,
    metric: str,
    value: float,
    floor: float,
) -> None:
    if value >= floor:
        return
    blockers.append(
        {
            "reason": reason,
            "metric": metric,
            "value": _round(value),
            "floor": _round(floor),
        }
    )


def _transition_summary(
    *,
    primary_summary: Mapping[str, object],
    comparison_summary: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if comparison_summary is None:
        return None
    return {
        "survivor_count_delta": int(primary_summary["survivor_count"])
        - int(comparison_summary["survivor_count"]),
        "survival_rate_delta": _round(
            float(primary_summary["survival_rate"])
            - float(comparison_summary["survival_rate"])
        ),
        "post_contact_survivor_count_delta": int(
            primary_summary["post_contact_survivor_count"]
        )
        - int(comparison_summary["post_contact_survivor_count"]),
        "post_contact_survival_rate_delta": _round(
            float(primary_summary["post_contact_survival_rate"])
            - float(comparison_summary["post_contact_survival_rate"])
        ),
        "mean_balanced_core_min_delta": _round(
            float(primary_summary["mean_balanced_core_min"])
            - float(comparison_summary["mean_balanced_core_min"])
        ),
    }


def _action_support(
    labels: Sequence[Mapping[str, object]],
    horizon: str,
) -> dict[str, object]:
    support: dict[str, Counter[str]] = {}
    balanced_by_action: dict[str, list[float]] = {}
    for label in labels:
        payload = _horizon_payload(label, horizon)
        if payload is None or payload.get("observed") is not True:
            continue
        action = _label_action(label)
        if action not in support:
            support[action] = Counter()
            balanced_by_action[action] = []
        support[action]["observed_count"] += 1
        if payload.get("survived") is True:
            support[action]["survivor_count"] += 1
        if payload.get("reproduced") is True:
            support[action]["reproduced_count"] += 1
        viability = payload.get("viability")
        if isinstance(viability, Mapping):
            balanced = _optional_float(viability.get("balanced_core_min"))
            if balanced is not None:
                balanced_by_action[action].append(balanced)
        animal = payload.get("animal_resource")
        if not isinstance(animal, Mapping):
            continue
        if animal.get("animal_resource_consumed") is True:
            support[action]["animal_resource_contact_count"] += 1
            if animal.get("survived_after_first_contact") is True:
                support[action]["post_contact_survivor_count"] += 1
    result: dict[str, object] = {}
    for action in sorted(support):
        counts = support[action]
        observed = int(counts["observed_count"])
        animal_contacts = int(counts["animal_resource_contact_count"])
        balanced_values = balanced_by_action[action]
        result[action] = {
            "observed_count": observed,
            "survivor_count": int(counts["survivor_count"]),
            "survival_rate": _round(_safe_rate(int(counts["survivor_count"]), observed)),
            "reproduced_count": int(counts["reproduced_count"]),
            "animal_resource_contact_count": animal_contacts,
            "post_contact_survivor_count": int(counts["post_contact_survivor_count"]),
            "post_contact_survival_rate": _round(
                _safe_rate(int(counts["post_contact_survivor_count"]), animal_contacts)
            ),
            "mean_balanced_core_min": _round(
                sum(balanced_values) / float(len(balanced_values))
                if balanced_values
                else 0.0
            ),
        }
    return result


def _autopsy_summary(report: Mapping[str, object] | None) -> dict[str, object] | None:
    if report is None:
        return None
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return None
    dominant = aggregate.get("dominant_death_path")
    return {
        "schema_version": report.get("schema_version"),
        "contact_episode_count": int(aggregate.get("contact_episode_count", 0)),
        "death_after_contact_count": int(
            aggregate.get("death_after_contact_count", 0)
        ),
        "post_contact_survival_rate": _optional_float(
            aggregate.get("post_contact_survival_rate")
        ),
        "dominant_death_path": (
            str(dominant.get("path"))
            if isinstance(dominant, Mapping) and dominant.get("path") is not None
            else None
        ),
        "death_path_counts": (
            dict(aggregate.get("death_path_counts"))
            if isinstance(aggregate.get("death_path_counts"), Mapping)
            else {}
        ),
    }


def _fixture_pressure_summary(
    report: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if report is None:
        return None
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return None
    pressure_by_fixture = aggregate.get("pressure_by_fixture")
    return {
        "schema_version": report.get("schema_version"),
        "failed_label_count": int(aggregate.get("failed_label_count", 0)),
        "pressure_total": _round(
            _optional_float(aggregate.get("pressure_total")) or 0.0
        ),
        "pressure_by_fixture": (
            dict(pressure_by_fixture)
            if isinstance(pressure_by_fixture, Mapping)
            else {}
        ),
    }


def _horizon_payload(
    label: Mapping[str, object],
    horizon: str,
) -> Mapping[str, object] | None:
    horizons = label.get("horizons")
    if not isinstance(horizons, Mapping):
        return None
    payload = horizons.get(horizon)
    return payload if isinstance(payload, Mapping) else None


def _label_action(label: Mapping[str, object]) -> str:
    requested = label.get("requested_action")
    if isinstance(requested, str) and requested:
        return requested
    resolved = label.get("resolved_action")
    if isinstance(resolved, str) and resolved:
        return resolved
    return "unknown"


def _positive_int(value: int | None, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TemporalCreditAuditError(f"{field} must be an integer")
    if value <= 0:
        raise TemporalCreditAuditError(f"{field} must be positive")
    return value


def _nonnegative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TemporalCreditAuditError(f"{field} must be an integer")
    if value < 0:
        raise TemporalCreditAuditError(f"{field} must be >= 0")
    return value


def _nonnegative_float(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemporalCreditAuditError(f"{field} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise TemporalCreditAuditError(f"{field} must be >= 0")
    return parsed


def _unit_float(value: float, *, field: str) -> float:
    parsed = _nonnegative_float(value, field=field)
    if parsed > 1.0:
        raise TemporalCreditAuditError(f"{field} must be <= 1")
    return parsed


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
