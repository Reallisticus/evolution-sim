from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.mind.provenance import stable_payload_digest

MIND_FIXTURE_LABEL_SCHEMA_VERSION = "mind_fixture_blocker_labels_v1"
MIND_FIXTURE_LABEL_POLICY = "controlled_fixture_floor_gap_labels_v1"

FIXTURE_FLOOR_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("fixture_alive_floor", "alive_agents_mean", "min_alive"),
    ("fixture_birth_floor", "births_mean", "min_births"),
    (
        "fixture_energy_viability_floor",
        "energy_viability_share_mean",
        "min_energy_viability",
    ),
    (
        "fixture_hydration_viability_floor",
        "hydration_viability_share_mean",
        "min_hydration_viability",
    ),
    (
        "fixture_health_viability_floor",
        "health_viability_share_mean",
        "min_health_viability",
    ),
    (
        "fixture_matched_diet_viability_floor",
        "matched_diet_viability_share_mean",
        "min_matched_diet_viability",
    ),
    (
        "fixture_biological_readiness_floor",
        "biologically_ready_agents_mean",
        "min_biologically_ready",
    ),
)


class FixtureLabelError(ValueError):
    pass


def load_fixture_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    with _open_input(resolved) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise FixtureLabelError(f"fixture report must be a JSON object: {resolved}")
    return payload


def build_fixture_label_report(
    reports: Sequence[Mapping[str, object]],
    *,
    report_paths: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    if not reports:
        raise FixtureLabelError("at least one fixture report is required")
    paths = [str(path) for path in report_paths or ()]
    labels: list[dict[str, object]] = []
    metric_snapshots: list[dict[str, object]] = []
    for report_index, report in enumerate(reports):
        source_report = paths[report_index] if report_index < len(paths) else None
        extracted = _labels_from_report(
            report,
            source_report=source_report,
            report_index=report_index,
        )
        labels.extend(extracted["labels"])
        metric_snapshots.extend(extracted["metric_snapshots"])
    contract = _label_contract()
    return {
        "schema_version": MIND_FIXTURE_LABEL_SCHEMA_VERSION,
        "label_policy": MIND_FIXTURE_LABEL_POLICY,
        "label_contract": contract,
        "source": {
            "report_count": len(reports),
            "report_paths": paths,
        },
        "provenance": {
            "source_report_digest": stable_payload_digest(
                {
                    "report_paths": paths,
                    "report_count": len(reports),
                    "report_schema_versions": [
                        report.get("schema_version") for report in reports
                    ],
                }
            ),
            "label_contract_digest": stable_payload_digest(contract),
        },
        "aggregate": _aggregate_fixture_labels(labels),
        "labels": labels,
        "metric_snapshots": metric_snapshots,
    }


def write_fixture_label_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(
            report,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")


def _label_contract() -> dict[str, object]:
    return {
        "schema_version": MIND_FIXTURE_LABEL_SCHEMA_VERSION,
        "policy": MIND_FIXTURE_LABEL_POLICY,
        "floor_checks": [
            {
                "reason": reason,
                "metric": metric,
                "floor_field": floor_field,
            }
            for reason, metric, floor_field in FIXTURE_FLOOR_CHECKS
        ],
        "special_floor_checks": [
            {
                "fixture": "mixed_stable",
                "reason": "fixture_mixed_stable_birth_floor",
                "metric": "births_mean",
                "floor_field": "min_mixed_stable_births",
            }
        ],
        "gap_policy": "max_floor_minus_value_zero_clamped_v1",
        "pressure_policy": "carrion_and_alive_weighted_gap_v1",
    }


def _labels_from_report(
    report: Mapping[str, object],
    *,
    source_report: str | None,
    report_index: int,
) -> dict[str, list[dict[str, object]]]:
    labels: list[dict[str, object]] = []
    metric_snapshots: list[dict[str, object]] = []
    fixture_gate = report.get("fixture_gate")
    if isinstance(fixture_gate, Mapping):
        labels.extend(
            _labels_from_fixture_gate(
                fixture_gate,
                source_report=source_report,
                report_index=report_index,
            )
        )
    fixture_suite = report.get("fixture_suite")
    if isinstance(fixture_suite, Mapping):
        metric_snapshots.extend(
            _metric_snapshots_from_fixture_suite(
                fixture_suite,
                source_report=source_report,
                report_index=report_index,
            )
        )
    return {
        "labels": labels,
        "metric_snapshots": metric_snapshots,
    }


def _labels_from_fixture_gate(
    fixture_gate: Mapping[str, object],
    *,
    source_report: str | None,
    report_index: int,
) -> list[dict[str, object]]:
    per_horizon = fixture_gate.get("per_horizon")
    if isinstance(per_horizon, Mapping) and per_horizon:
        labels: list[dict[str, object]] = []
        for horizon_key in sorted(per_horizon, key=_horizon_sort_key):
            horizon_gate = per_horizon[horizon_key]
            if not isinstance(horizon_gate, Mapping):
                continue
            labels.extend(
                _labels_from_single_gate(
                    horizon_gate,
                    parent_gate=fixture_gate,
                    source_report=source_report,
                    report_index=report_index,
                    ticks=_optional_int_from_key(horizon_key),
                )
            )
        return labels
    return _labels_from_single_gate(
        fixture_gate,
        parent_gate=fixture_gate,
        source_report=source_report,
        report_index=report_index,
        ticks=_optional_int(fixture_gate.get("ticks")),
    )


def _labels_from_single_gate(
    gate: Mapping[str, object],
    *,
    parent_gate: Mapping[str, object],
    source_report: str | None,
    report_index: int,
    ticks: int | None,
) -> list[dict[str, object]]:
    per_fixture = gate.get("per_fixture")
    labels: list[dict[str, object]] = []
    if isinstance(per_fixture, Mapping) and per_fixture:
        for fixture_name in sorted(str(name) for name in per_fixture):
            fixture_payload = per_fixture.get(fixture_name)
            if not isinstance(fixture_payload, Mapping):
                continue
            metrics = fixture_payload.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            labels.extend(
                _floor_labels_for_metrics(
                    metrics,
                    fixture=fixture_name,
                    ticks=ticks,
                    gate=parent_gate,
                    source_report=source_report,
                    report_index=report_index,
                )
            )
    if labels:
        return labels
    blockers = gate.get("blockers")
    if isinstance(blockers, list):
        return [
            _label_from_blocker(
                blocker,
                source_report=source_report,
                report_index=report_index,
                default_ticks=ticks,
            )
            for blocker in blockers
            if isinstance(blocker, Mapping)
        ]
    return labels


def _floor_labels_for_metrics(
    metrics: Mapping[str, object],
    *,
    fixture: str,
    ticks: int | None,
    gate: Mapping[str, object],
    source_report: str | None,
    report_index: int,
) -> list[dict[str, object]]:
    labels = [
        _metric_floor_label(
            fixture=fixture,
            ticks=ticks,
            reason=reason,
            metric=metric,
            value=_float_metric(metrics.get(metric)),
            floor=_float_metric(gate.get(floor_field)),
            source_report=source_report,
            report_index=report_index,
        )
        for reason, metric, floor_field in FIXTURE_FLOOR_CHECKS
    ]
    if fixture == "mixed_stable":
        labels.append(
            _metric_floor_label(
                fixture=fixture,
                ticks=ticks,
                reason="fixture_mixed_stable_birth_floor",
                metric="births_mean",
                value=_float_metric(metrics.get("births_mean")),
                floor=_float_metric(gate.get("min_mixed_stable_births")),
                source_report=source_report,
                report_index=report_index,
            )
        )
    return labels


def _metric_floor_label(
    *,
    fixture: str,
    ticks: int | None,
    reason: str,
    metric: str,
    value: float,
    floor: float,
    source_report: str | None,
    report_index: int,
) -> dict[str, object]:
    gap = max(0.0, floor - value)
    passed = gap <= 0.0
    return {
        "schema_version": MIND_FIXTURE_LABEL_SCHEMA_VERSION,
        "source_report": source_report,
        "source_report_index": report_index,
        "fixture": fixture,
        "ticks": ticks,
        "reason": reason,
        "metric": metric,
        "value": _round(value),
        "floor": _round(floor),
        "passed": passed,
        "gap": _round(gap),
        "normalized_gap": _normalized_gap(gap, floor),
        "pressure": _round(gap * _pressure_weight(fixture=fixture, reason=reason)),
    }


def _label_from_blocker(
    blocker: Mapping[str, object],
    *,
    source_report: str | None,
    report_index: int,
    default_ticks: int | None,
) -> dict[str, object]:
    fixture = str(blocker.get("fixture", "unknown"))
    reason = str(blocker.get("reason", "unknown"))
    metric = str(blocker.get("metric", "unknown"))
    value = _float_metric(blocker.get("value"))
    floor = _float_metric(blocker.get("floor"))
    ticks = _optional_int(blocker.get("ticks"))
    if ticks is None:
        ticks = default_ticks
    return _metric_floor_label(
        fixture=fixture,
        ticks=ticks,
        reason=reason,
        metric=metric,
        value=value,
        floor=floor,
        source_report=source_report,
        report_index=report_index,
    )


def _metric_snapshots_from_fixture_suite(
    fixture_suite: Mapping[str, object],
    *,
    source_report: str | None,
    report_index: int,
) -> list[dict[str, object]]:
    fixtures = fixture_suite.get("fixtures")
    if not isinstance(fixtures, list):
        return []
    snapshots: list[dict[str, object]] = []
    for raw_fixture in fixtures:
        if not isinstance(raw_fixture, Mapping):
            continue
        fixture_name = str(raw_fixture.get("fixture", "unknown"))
        comparison = raw_fixture.get("comparison")
        mind_v3 = comparison.get("mind_v3") if isinstance(comparison, Mapping) else None
        aggregate = mind_v3.get("aggregate") if isinstance(mind_v3, Mapping) else None
        if not isinstance(aggregate, Mapping):
            continue
        snapshots.append(
            {
                "source_report": source_report,
                "source_report_index": report_index,
                "fixture": fixture_name,
                "ticks": _optional_int(fixture_suite.get("ticks")),
                "alive_agents_mean": _float_metric(aggregate.get("alive_agents_mean")),
                "births_mean": _float_metric(aggregate.get("births_mean")),
                "dominant_requested_action": aggregate.get("dominant_requested_action"),
                "dominant_requested_action_share": _float_metric(
                    aggregate.get("dominant_requested_action_share")
                ),
            }
        )
    return snapshots


def _aggregate_fixture_labels(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failed = [label for label in labels if label.get("passed") is False]
    by_fixture = Counter(str(label.get("fixture", "unknown")) for label in failed)
    by_reason = Counter(str(label.get("reason", "unknown")) for label in failed)
    pressure_by_fixture: dict[str, float] = {}
    for label in failed:
        fixture = str(label.get("fixture", "unknown"))
        pressure_by_fixture[fixture] = pressure_by_fixture.get(fixture, 0.0) + (
            _float_metric(label.get("pressure"))
        )
    return {
        "label_count": len(labels),
        "failed_label_count": len(failed),
        "passed_label_count": len(labels) - len(failed),
        "blocker_count_by_fixture": dict(sorted(by_fixture.items())),
        "blocker_count_by_reason": dict(sorted(by_reason.items())),
        "pressure_total": _round(sum(_float_metric(label.get("pressure")) for label in failed)),
        "pressure_by_fixture": {
            fixture: _round(pressure)
            for fixture, pressure in sorted(pressure_by_fixture.items())
        },
        "carrion_only_failed_label_count": int(by_fixture.get("carrion_only", 0)),
    }


def _pressure_weight(*, fixture: str, reason: str) -> float:
    weight = 1.0
    if fixture == "carrion_only":
        weight += 1.0
    if reason == "fixture_alive_floor":
        weight += 1.0
    return weight


def _normalized_gap(gap: float, floor: float) -> float | None:
    if floor <= 0.0:
        return None
    return _round(gap / floor)


def _horizon_sort_key(value: object) -> tuple[int, str]:
    parsed = _optional_int_from_key(value)
    return (parsed if parsed is not None else 10**9, str(value))


def _optional_int_from_key(value: object) -> int | None:
    try:
        return int(str(value))
    except ValueError:
        return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _float_metric(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


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
