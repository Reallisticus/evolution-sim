from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

MIND_V3_CARRION_OBJECTIVE_AUDIT_SCHEMA_VERSION = (
    "mind_v3_carrion_objective_pressure_audit_v1"
)
MIND_V3_CARRION_OBJECTIVE_AUDIT_POLICY = (
    "diagnostics_only_carrion_objective_pressure_v1"
)
MIND_V3_FOUNDATION_CARRION_BASELINE_SCHEMA_VERSION = (
    "mind_v3_foundation_carrion_only_baseline_v1"
)
MIND_V3_FOUNDATION_CARRION_BASELINE_POLICY = (
    "foundation_default_policy_carrion_only_trajectory_baseline_v1"
)
MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY = (
    "gate_aligned_carrion_recovery_probe_v1"
)

DEFAULT_CARRION_OBJECTIVE_SEARCH_REPORTS: tuple[tuple[str, str], ...] = (
    ("v4", "output/mind/mind-v3-v4-baseline-search-80-120.json"),
    (
        "v5",
        "output/mind/mind-v3-v5-rollout-context-search-80-120-diagnostic.json",
    ),
    (
        "v6",
        "output/mind/mind-v3-v6-recovery-context-search-80-120-diagnostic.json",
    ),
)
DEFAULT_CARRION_OBJECTIVE_TRACE_REPORTS: tuple[tuple[str, str], ...] = (
    ("v4", "output/mind/mind-v3-v4-carrion-only-120-trace.json"),
    ("v5", "output/mind/mind-v3-v5-carrion-only-120-trace.json"),
    ("v6", "output/mind/mind-v3-v6-carrion-only-120-trace.json"),
)
DEFAULT_FOUNDATION_CARRION_BASELINE_SEEDS = (13, 19, 29, 37, 41, 43)
DEFAULT_FOUNDATION_CARRION_BASELINE_TICKS = 120

_SELECTED_CORRELATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("search_score", "animal_resource_gain_total_mean"),
    ("search_score", "births_mean"),
    ("search_score", "carrion_contact_agent_count"),
    ("search_score", "drink_after_carrion_rate"),
    ("search_score", "mean_hydration_delta_after_carrion"),
    ("search_score", "post_contact_survival_rate"),
    ("search_score", "unsupported_resolved_action_count"),
    ("search_score", "dominant_requested_action_share"),
    ("animal_resource_gain_total_mean", "drink_after_carrion_rate"),
    ("animal_resource_gain_total_mean", "mean_hydration_delta_after_carrion"),
)
_CANDIDATE_CORRELATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("search_score", "alive_agents_mean"),
    ("search_score", "alive_agent_ticks_mean"),
    ("search_score", "births_mean"),
    ("search_score", "animal_resource_gain_total_mean"),
    ("search_score", "unsupported_resolved_action_count"),
    ("search_score", "dominant_requested_action_share"),
)
_REQUIRED_SELECTED_METRICS = (
    "search_score",
    "fixture_blocker_count",
    "alive_agents_mean",
    "alive_agent_ticks_mean",
    "births_mean",
    "animal_resource_gain_total_mean",
    "carrion_contact_agent_count",
    "drink_after_carrion_rate",
    "mean_hydration_delta_after_carrion",
    "mean_water_distance",
    "post_contact_survival_rate",
    "unsupported_resolved_action_count",
    "dominant_requested_action_share",
)
_RECOVERY_METRICS = (
    "drink_after_carrion_rate",
    "mean_hydration_delta_after_carrion",
    "post_contact_survival_rate",
)
_CANDIDATE_RECOVERY_PROBE_FIELDS = (
    "post_contact_survival_rate",
    "drink_after_carrion_rate",
    "mean_hydration_delta_after_carrion",
    "mean_energy_delta_after_carrion",
    "mean_health_delta_after_carrion",
    "mean_water_distance",
)
_GATE_ALIGNED_SELECTOR_FIELDS = (
    "fixture_gate_passed",
    "fixture_blocker_count",
    "carrion_only_blocker_count",
    "passed_horizon_count",
    "first_horizon_passed",
    "carrion_only_alive_agents_mean",
    "carrion_only_births_mean",
    "carrion_only_terminal_hydration_viability_share_mean",
    "carrion_only_terminal_energy_viability_share_mean",
    "carrion_only_terminal_health_viability_share_mean",
    "carrion_only_terminal_matched_diet_viability_share_mean",
    "unsupported_requested_action_count",
    "unsupported_resolved_action_count",
    "carrion_only_dominant_requested_action_share",
    "search_score",
)


class CarrionObjectiveAuditError(ValueError):
    pass


def load_json_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CarrionObjectiveAuditError(f"failed to read JSON report: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise CarrionObjectiveAuditError(
            f"invalid JSON report {resolved}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise CarrionObjectiveAuditError(f"JSON report must be an object: {resolved}")
    return payload


def write_json_report(report: Mapping[str, object], path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        raw_label, raw_path = value.split("=", 1)
        label = raw_label.strip()
        path_text = raw_path.strip()
        if not label or not path_text:
            raise CarrionObjectiveAuditError(
                f"labeled path must use LABEL=PATH: {value!r}"
            )
        return label, Path(path_text)
    path = Path(value)
    return _label_from_path(path), path


def build_carrion_objective_audit_report(
    *,
    search_reports: Sequence[tuple[str, Mapping[str, object], str | None]],
    trace_reports: Sequence[tuple[str, Mapping[str, object], str | None]],
    missing_inputs: Sequence[Mapping[str, object]] = (),
    selector_probe: str | None = None,
) -> dict[str, object]:
    if selector_probe is not None and selector_probe != (
        MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY
    ):
        raise CarrionObjectiveAuditError(
            f"unsupported selector probe policy: {selector_probe}"
        )
    trace_by_label = {
        label: _trace_metrics(label=label, report=report, source_path=path)
        for label, report, path in trace_reports
    }
    selected_rows: list[dict[str, object]] = []
    generation_rows: list[dict[str, object]] = []
    rerank_rows: list[dict[str, object]] = []
    selector_comparisons: list[dict[str, object]] = []
    missing_records: list[dict[str, object]] = [
        dict(item) for item in missing_inputs
    ]

    for label, report, path in search_reports:
        trace_metrics = trace_by_label.get(label)
        selected = _selected_candidate_row(
            label=label,
            report=report,
            source_path=path,
            trace_metrics=trace_metrics,
        )
        selected_rows.append(selected)
        _record_missing_metrics(
            selected,
            required_metrics=_REQUIRED_SELECTED_METRICS,
            missing_records=missing_records,
        )
        generation_rows.extend(
            _generation_candidate_rows(
                label=label,
                report=report,
                source_path=path,
                selected_candidate_id=str(selected.get("candidate_id") or ""),
            )
        )
        report_rerank_rows = _fixture_rerank_candidate_rows(
            label=label,
            report=report,
            source_path=path,
            selected_candidate_id=str(selected.get("candidate_id") or ""),
        )
        rerank_rows.extend(report_rerank_rows)
        if selector_probe is not None:
            selector_comparisons.append(
                _gate_aligned_selector_comparison(
                    label=label,
                    report=report,
                    source_path=path,
                    current_selected=selected,
                    rerank_rows=report_rerank_rows,
                )
            )

    report = {
        "schema_version": MIND_V3_CARRION_OBJECTIVE_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_CARRION_OBJECTIVE_AUDIT_POLICY,
        "contract": _audit_contract(),
        "inputs": {
            "search_report_count": len(search_reports),
            "trace_report_count": len(trace_reports),
            "missing_input_count": len(missing_inputs),
            "search_reports": [
                {"label": label, "path": path}
                for label, _report, path in search_reports
            ],
            "trace_reports": [
                {"label": label, "path": path}
                for label, _report, path in trace_reports
            ],
            "missing_inputs": [dict(item) for item in missing_inputs],
        },
        "selected_candidates": selected_rows,
        "candidate_sets": {
            "generation_candidate_count": len(generation_rows),
            "fixture_rerank_candidate_count": len(rerank_rows),
            "generation_top_by_search_score": _rank_rows(
                generation_rows,
                "search_score",
                reverse=True,
                limit=10,
            ),
            "fixture_rerank_top_by_search_score": _rank_rows(
                rerank_rows,
                "search_score",
                reverse=True,
                limit=10,
            ),
            "fixture_rerank_top_by_carrion_gain": _rank_rows(
                rerank_rows,
                "carrion_only_animal_resource_gain_mean",
                reverse=True,
                limit=10,
            ),
            "fixture_rerank_worst_by_carrion_blockers": _rank_rows(
                rerank_rows,
                "carrion_only_blocker_count",
                reverse=True,
                limit=10,
            ),
        },
        "correlations": {
            "selected_candidates": _correlation_report(
                selected_rows,
                _SELECTED_CORRELATION_PAIRS,
            ),
            "generation_candidates": _correlation_report(
                generation_rows,
                _CANDIDATE_CORRELATION_PAIRS,
            ),
            "fixture_rerank_candidates": _correlation_report(
                rerank_rows,
                _CANDIDATE_CORRELATION_PAIRS
                + (
                    ("search_score", "carrion_only_animal_resource_gain_mean"),
                    ("search_score", "carrion_only_blocker_count"),
                    (
                        "search_score",
                        "carrion_only_terminal_hydration_viability_share_mean",
                    ),
                ),
            ),
        },
        "diagnosis": _diagnosis(
            selected_rows=selected_rows,
            generation_rows=generation_rows,
            rerank_rows=rerank_rows,
            missing_records=missing_records,
        ),
        "missing_data": _missing_data_summary(missing_records),
        "foundation_heuristic_baseline": foundation_heuristic_baseline_contract(),
    }
    if selector_probe is not None:
        report["selector_probe"] = _selector_probe_summary(selector_comparisons)
    return report


def foundation_heuristic_baseline_contract(
    *,
    seeds: Sequence[int] = DEFAULT_FOUNDATION_CARRION_BASELINE_SEEDS,
    ticks: int = DEFAULT_FOUNDATION_CARRION_BASELINE_TICKS,
    output_path: str | Path = (
        "output/mind/mind-v3-carrion-only-120-heuristic-baseline.json"
    ),
    trajectory_output_dir: str | Path = (
        "output/mind/mind-v3-carrion-only-120-heuristic-trajectories"
    ),
    trace_output_path: str | Path = (
        "output/mind/mind-v3-carrion-only-120-heuristic-trace.json"
    ),
) -> dict[str, object]:
    seed_list = [int(seed) for seed in seeds]
    output = str(output_path)
    trajectory_dir = str(trajectory_output_dir)
    trace_output = str(trace_output_path)
    seed_arg = ",".join(str(seed) for seed in seed_list)
    return {
        "available": True,
        "schema_version": MIND_V3_FOUNDATION_CARRION_BASELINE_SCHEMA_VERSION,
        "policy": MIND_V3_FOUNDATION_CARRION_BASELINE_POLICY,
        "policy_source": "foundation_default_policy_no_mind",
        "fixture": "carrion_only",
        "seeds": seed_list,
        "ticks": int(ticks),
        "uses_founder_template": False,
        "uses_mind_policy": False,
        "uses_controller_architecture": False,
        "output_path": output,
        "trajectory_output_dir": trajectory_dir,
        "trace_output_path": trace_output,
        "command": (
            "npm run sim:mind:v3:carrion-objective-audit -- "
            "--generate-heuristic-baseline "
            f"--baseline-seeds {seed_arg} "
            f"--baseline-ticks {int(ticks)} "
            f"--baseline-output {output} "
            f"--baseline-trajectory-output-dir {trajectory_dir} "
            f"--baseline-trace-output {trace_output}"
        ),
    }


def foundation_heuristic_baseline_report(
    *,
    runs: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    seeds: Sequence[int],
    ticks: int,
    output_path: str | Path,
    trajectory_output_dir: str | Path,
    trace_output_path: str | Path,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FOUNDATION_CARRION_BASELINE_SCHEMA_VERSION,
        "baseline_policy": MIND_V3_FOUNDATION_CARRION_BASELINE_POLICY,
        "contract": foundation_heuristic_baseline_contract(
            seeds=seeds,
            ticks=ticks,
            output_path=output_path,
            trajectory_output_dir=trajectory_output_dir,
            trace_output_path=trace_output_path,
        ),
        "policy": {
            "policy_source": "foundation_default_policy_no_mind",
            "uses_founder_template": False,
            "uses_mind_policy": False,
            "uses_controller_architecture": False,
        },
        "fixture": "carrion_only",
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "runs": [dict(run) for run in runs],
        "aggregate": dict(aggregate),
        "output_path": str(output_path),
        "trajectory_output_dir": str(trajectory_output_dir),
        "trace_output_path": str(trace_output_path),
    }


def _selected_candidate_row(
    *,
    label: str,
    report: Mapping[str, object],
    source_path: str | None,
    trace_metrics: Mapping[str, object] | None,
) -> dict[str, object]:
    best = _mapping(report.get("best_candidate"))
    holdout_aggregate = _mapping(
        _mapping(report.get("holdout_evaluation")).get("aggregate")
    )
    row = _candidate_metrics(
        label=label,
        source="selected_candidate",
        candidate=best,
        source_path=source_path,
        selected=True,
    )
    _fill_missing_number(
        row,
        "unsupported_resolved_action_count",
        holdout_aggregate.get("unsupported_resolved_action_count"),
    )
    _fill_missing_number(
        row,
        "dominant_requested_action_share",
        holdout_aggregate.get("dominant_requested_action_share"),
    )
    if row.get("dominant_requested_action") is None:
        row["dominant_requested_action"] = _optional_string(
            holdout_aggregate.get("dominant_requested_action")
        )
    row["fixture_gate_passed"] = _optional_bool(
        _mapping(report.get("fixture_gate")).get("passed")
    )
    row["fixture_gate_blocker_count"] = len(
        _list_of_mappings(_mapping(report.get("fixture_gate")).get("blockers"))
    )
    if trace_metrics is not None:
        row.update(trace_metrics)
    else:
        row["trace_report_present"] = False
    return row


def _generation_candidate_rows(
    *,
    label: str,
    report: Mapping[str, object],
    source_path: str | None,
    selected_candidate_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for generation in _list_of_mappings(report.get("generations")):
        generation_index = _optional_int(generation.get("generation_index"))
        for candidate in _list_of_mappings(generation.get("candidates")):
            row = _candidate_metrics(
                label=label,
                source="generation_candidate",
                candidate=candidate,
                source_path=source_path,
                selected=str(candidate.get("candidate_id")) == selected_candidate_id,
            )
            row["generation_index"] = generation_index
            rows.append(row)
    return rows


def _fixture_rerank_candidate_rows(
    *,
    label: str,
    report: Mapping[str, object],
    source_path: str | None,
    selected_candidate_id: str,
) -> list[dict[str, object]]:
    rerank = _mapping(report.get("fixture_rerank"))
    rows: list[dict[str, object]] = []
    for candidate in _list_of_mappings(rerank.get("candidates")):
        row = _candidate_metrics(
            label=label,
            source="fixture_rerank_candidate",
            candidate=candidate,
            source_path=source_path,
            selected=str(candidate.get("candidate_id")) == selected_candidate_id,
        )
        fixture_summary = _mapping(candidate.get("fixture_summary"))
        horizon_summary = _mapping(candidate.get("fixture_horizon_summary"))
        pressure = _mapping(candidate.get("fixture_blocker_pressure"))
        fixture_gate = _mapping(candidate.get("fixture_gate"))
        blockers = _list_of_mappings(fixture_gate.get("blockers"))
        holdout_aggregate = _mapping(candidate.get("holdout_aggregate"))
        carrion_fixture = _fixture_summary_named_fixture(
            fixture_summary,
            "carrion_only",
        )
        row["prefilter_rank"] = _optional_int(candidate.get("prefilter_rank"))
        row["fixture_gate_passed"] = _optional_bool(fixture_gate.get("passed"))
        row["fixture_gate_blocker_count"] = len(blockers)
        row["fixture_blocker_count"] = _first_number(
            pressure.get("blocker_count"),
            len(blockers),
        )
        row["carrion_only_blocker_count"] = _optional_number(
            pressure.get("carrion_only_blocker_count")
        )
        if row["carrion_only_blocker_count"] is None:
            row["carrion_only_blocker_count"] = float(
                sum(
                    1
                    for blocker in blockers
                    if blocker.get("fixture") == "carrion_only"
                )
            )
        row["passed_horizon_count"] = _optional_number(
            horizon_summary.get("passed_horizon_count")
        )
        row["horizon_count"] = _optional_number(horizon_summary.get("horizon_count"))
        row["first_horizon_passed"] = _first_horizon_passed(candidate)
        row["all_horizons_passed"] = _optional_bool(
            horizon_summary.get("all_horizons_passed")
        )
        row["carrion_only_animal_resource_event_mean"] = _optional_number(
            fixture_summary.get("carrion_only_animal_resource_consumption_events_mean")
        )
        row["carrion_only_animal_resource_gain_mean"] = _optional_number(
            fixture_summary.get("carrion_only_animal_resource_gained_energy_mean")
        )
        row["carrion_only_births_mean"] = _optional_number(
            fixture_summary.get("carrion_only_births_mean")
            if fixture_summary.get("carrion_only_births_mean") is not None
            else carrion_fixture.get("births_mean")
        )
        row["carrion_only_alive_agents_mean"] = _optional_number(
            fixture_summary.get("carrion_only_alive_agents_mean")
            if fixture_summary.get("carrion_only_alive_agents_mean") is not None
            else carrion_fixture.get("alive_agents_mean")
        )
        row["carrion_only_terminal_hydration_viability_share_mean"] = _first_number(
            horizon_summary.get("carrion_only_terminal_hydration_viability_share_min"),
            fixture_summary.get("carrion_only_terminal_hydration_viability_share_mean"),
            carrion_fixture.get("terminal_hydration_viability_share_mean"),
        )
        row["carrion_only_terminal_energy_viability_share_mean"] = _first_number(
            horizon_summary.get("carrion_only_terminal_energy_viability_share_min"),
            fixture_summary.get("carrion_only_terminal_energy_viability_share_mean"),
            carrion_fixture.get("terminal_energy_viability_share_mean"),
        )
        row["carrion_only_terminal_health_viability_share_mean"] = _first_number(
            fixture_summary.get("carrion_only_terminal_health_viability_share_mean"),
            carrion_fixture.get("terminal_health_viability_share_mean"),
        )
        row["carrion_only_terminal_matched_diet_viability_share_mean"] = (
            _first_number(
                horizon_summary.get(
                    "carrion_only_terminal_matched_diet_viability_share_min"
                ),
                fixture_summary.get(
                    "carrion_only_terminal_matched_diet_viability_share_mean"
                ),
                carrion_fixture.get("terminal_matched_diet_viability_share_mean"),
            )
        )
        row["carrion_only_alive_agents_min"] = _optional_number(
            horizon_summary.get("carrion_only_alive_agents_min")
        )
        row["carrion_only_dominant_requested_action_share"] = _first_number(
            horizon_summary.get("carrion_only_dominant_requested_action_share_max"),
            carrion_fixture.get("dominant_requested_action_share"),
        )
        row["unsupported_requested_action_count"] = _optional_number(
            holdout_aggregate.get("unsupported_requested_action_count")
        )
        row["unsupported_resolved_action_count"] = _optional_number(
            holdout_aggregate.get("unsupported_resolved_action_count")
        )
        _apply_candidate_recovery_probe(row, candidate)
        rows.append(row)
    return rows


def _apply_candidate_recovery_probe(
    row: dict[str, object],
    candidate: Mapping[str, object],
) -> None:
    probe = _candidate_recovery_probe(candidate)
    row["candidate_recovery_probe_present"] = bool(probe)
    if not probe:
        return
    row["candidate_recovery_probe_policy"] = _optional_string(probe.get("policy"))
    for source_key, row_key in (
        ("post_contact_survival_rate", "post_contact_survival_rate"),
        ("drink_after_carrion_rate", "drink_after_carrion_rate"),
        (
            "mean_hydration_delta_after_carrion",
            "mean_hydration_delta_after_carrion",
        ),
        ("mean_energy_delta_after_carrion", "mean_energy_delta_after_carrion"),
        ("mean_health_delta_after_carrion", "mean_health_delta_after_carrion"),
        ("mean_water_distance", "mean_water_distance"),
        (
            "unsupported_requested_action_count",
            "unsupported_requested_action_count",
        ),
        (
            "unsupported_resolved_action_count",
            "unsupported_resolved_action_count",
        ),
        (
            "dominant_requested_action_share",
            "carrion_only_dominant_requested_action_share",
        ),
    ):
        value = _optional_number(probe.get(source_key))
        if value is not None:
            row[row_key] = value
    for source_key, row_key in (
        ("fixture_blocker_count", "fixture_blocker_count"),
        ("carrion_only_blocker_count", "carrion_only_blocker_count"),
    ):
        value = _optional_number(probe.get(source_key))
        if value is not None:
            row[row_key] = value
    if _optional_bool(probe.get("fixture_gate_passed")) is not None:
        row["fixture_gate_passed"] = _optional_bool(probe.get("fixture_gate_passed"))


def _candidate_recovery_probe(candidate: Mapping[str, object]) -> Mapping[str, object]:
    for key in (
        "gate_aligned_carrion_recovery_probe_v1",
        "carrion_recovery_probe",
        "selector_probe",
    ):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _fixture_summary_named_fixture(
    fixture_summary: Mapping[str, object],
    fixture_name: str,
) -> Mapping[str, object]:
    per_fixture = fixture_summary.get("per_fixture")
    if not isinstance(per_fixture, Mapping):
        return {}
    payload = per_fixture.get(fixture_name)
    return payload if isinstance(payload, Mapping) else {}


def _first_horizon_passed(candidate: Mapping[str, object]) -> bool | None:
    horizons = candidate.get("fixture_horizons")
    if isinstance(horizons, list) and horizons:
        first = horizons[0]
        if isinstance(first, Mapping):
            gate = first.get("fixture_gate")
            if isinstance(gate, Mapping):
                return _optional_bool(gate.get("passed"))
        return False
    gate = candidate.get("fixture_gate")
    if isinstance(gate, Mapping):
        return _optional_bool(gate.get("passed"))
    return None


def _candidate_metrics(
    *,
    label: str,
    source: str,
    candidate: Mapping[str, object],
    source_path: str | None,
    selected: bool,
) -> dict[str, object]:
    fixture_selection = _mapping(candidate.get("fixture_selection"))
    fixture_pressure = _mapping(candidate.get("fixture_blocker_pressure"))
    score_components = _mapping(candidate.get("score_components"))
    row: dict[str, object] = {
        "label": label,
        "source": source,
        "source_path": source_path,
        "candidate_id": _optional_string(candidate.get("candidate_id")),
        "selected": bool(selected),
        "controller_architecture": _optional_string(
            candidate.get("controller_architecture")
        ),
        "search_score": _first_number(
            candidate.get("score"),
            candidate.get("search_score"),
            fixture_selection.get("pre_fixture_score"),
        ),
        "rerank_score": _first_number(
            fixture_selection.get("post_fixture_score"),
            candidate.get("post_fixture_score"),
        ),
        "fixture_blocker_count": _first_number(
            fixture_selection.get("blocker_count"),
            fixture_pressure.get("blocker_count"),
        ),
        "carrion_only_blocker_count": _first_number(
            fixture_selection.get("carrion_only_blocker_count"),
            fixture_pressure.get("carrion_only_blocker_count"),
        ),
        "alive_agents_mean": _first_number(
            candidate.get("alive_agents_mean"),
            candidate.get("search_alive_agents_mean"),
        ),
        "alive_agent_ticks_mean": _first_number(
            candidate.get("alive_agent_ticks_mean"),
            candidate.get("search_alive_agent_ticks_mean"),
        ),
        "alive_agent_ticks_per_tick_mean": _first_number(
            candidate.get("alive_agent_ticks_per_tick_mean"),
            candidate.get("search_alive_agent_ticks_per_tick_mean"),
        ),
        "births_mean": _first_number(
            candidate.get("births_mean"),
            candidate.get("search_births_mean"),
        ),
        "deaths_mean": _optional_number(candidate.get("deaths_mean")),
        "animal_resource_event_rate": _optional_number(
            candidate.get("animal_resource_event_rate")
        ),
        "animal_resource_gain_total_mean": _optional_number(
            candidate.get("animal_resource_gain_total_mean")
        ),
        "animal_resource_score_component": _optional_number(
            score_components.get("animal_resource_events")
        ),
        "carrion_resource_score_component": _optional_number(
            score_components.get("carrion_resource_events")
        ),
        "unsupported_resolved_action_count": _optional_number(
            candidate.get("unsupported_resolved_action_count")
        ),
        "dominant_requested_action": _optional_string(
            candidate.get("dominant_requested_action")
            or candidate.get("search_dominant_requested_action")
        ),
        "dominant_requested_action_share": _first_number(
            candidate.get("dominant_requested_action_share"),
            candidate.get("search_dominant_requested_action_share"),
        ),
    }
    return row


def _trace_metrics(
    *,
    label: str,
    report: Mapping[str, object],
    source_path: str | None,
) -> dict[str, object]:
    fixture_trace = _mapping(report.get("fixture_trace"))
    aggregate = _mapping(fixture_trace.get("aggregate"))
    navigation = _mapping(aggregate.get("navigation_target_observations"))
    water_navigation = _mapping(navigation.get("water"))
    requested_counts = _counter_mapping(aggregate.get("requested_action_counts"))
    total_requested = sum(requested_counts.values())
    dominant_action, dominant_count = _dominant_item(requested_counts)
    return {
        "trace_report_present": True,
        "trace_source_path": source_path,
        "carrion_contact_agent_count": _optional_number(
            aggregate.get("contact_agent_count")
        ),
        "animal_resource_successful_eat_count": _optional_number(
            aggregate.get("animal_resource_successful_eat_count")
        ),
        "drink_after_carrion_rate": _optional_number(
            aggregate.get("drink_after_carrion_rate")
        ),
        "mean_hydration_delta_after_carrion": _optional_number(
            aggregate.get("mean_hydration_delta_after_carrion")
        ),
        "mean_energy_delta_after_carrion": _optional_number(
            aggregate.get("mean_energy_delta_after_carrion")
        ),
        "mean_health_delta_after_carrion": _optional_number(
            aggregate.get("mean_health_delta_after_carrion")
        ),
        "post_contact_survival_rate": _optional_number(
            aggregate.get("survival_after_carrion_rate")
        ),
        "mean_water_distance": _optional_number(water_navigation.get("mean_distance")),
        "trace_unsupported_resolved_action_count": _optional_number(
            aggregate.get("unsupported_resolved_action_count")
        ),
        "trace_dominant_requested_action": dominant_action,
        "trace_dominant_requested_action_share": _safe_rate(
            dominant_count,
            total_requested,
        ),
        "trace_label": label,
    }


def _correlation_report(
    rows: Sequence[Mapping[str, object]],
    pairs: Sequence[tuple[str, str]],
) -> dict[str, object]:
    return {
        f"{left}_vs_{right}": _metric_correlation(rows, left, right)
        for left, right in pairs
    }


def _selector_probe_summary(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selection_changes = [
        comparison
        for comparison in comparisons
        if bool(comparison.get("selection_changes", False))
    ]
    plausible_alternates: list[dict[str, object]] = []
    for comparison in comparisons:
        label = str(comparison.get("label", "unknown"))
        for alternate in _list_of_mappings(comparison.get("plausible_alternates")):
            plausible_alternates.append({"label": label, **dict(alternate)})
    return {
        "policy": MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY,
        "report_only": True,
        "active_selector_implemented": False,
        "changes_runtime_policy": False,
        "changes_candidate_selection": False,
        "candidate_count_inspected": sum(
            int(comparison.get("candidate_count_inspected", 0))
            for comparison in comparisons
        ),
        "selection_change_count": len(selection_changes),
        "labels_with_selection_changes": [
            comparison.get("label") for comparison in selection_changes
        ],
        "plausible_alternate_count": len(plausible_alternates),
        "plausible_alternates": plausible_alternates,
        "comparisons": [dict(comparison) for comparison in comparisons],
    }


def _gate_aligned_selector_comparison(
    *,
    label: str,
    report: Mapping[str, object],
    source_path: str | None,
    current_selected: Mapping[str, object],
    rerank_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    current_id = _selected_candidate_id(report, current_selected)
    candidate_rows = [dict(row) for row in rerank_rows]
    current_row = _row_by_candidate_id(candidate_rows, current_id)
    if current_row is None:
        current_row = dict(current_selected)
    selected_row = (
        max(candidate_rows, key=_gate_aligned_selector_key)
        if candidate_rows
        else None
    )
    selected_id = (
        str(selected_row.get("candidate_id"))
        if selected_row is not None and selected_row.get("candidate_id") is not None
        else None
    )
    missing = _selector_missing_fields(candidate_rows)
    plausible = (
        _plausible_alternates(candidate_rows, current_row)
        if candidate_rows and current_row
        else []
    )
    return {
        "label": label,
        "source_path": source_path,
        "policy": MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY,
        "report_only": True,
        "selector_input_source": "fixture_rerank_candidates"
        if candidate_rows
        else "selected_candidate_only",
        "fixture_rerank_top_k": _optional_int(
            _mapping(report.get("fixture_rerank")).get("top_k")
        ),
        "candidate_count_inspected": len(candidate_rows),
        "current_selected_candidate": _selector_candidate_summary(current_row),
        "report_only_gate_aligned_selected_candidate": (
            _selector_candidate_summary(selected_row) if selected_row else None
        ),
        "selection_changes": bool(selected_id and selected_id != current_id),
        "missing_fields_preventing_full_recovery_selection": missing,
        "full_recovery_aware_selection_available": (
            len(candidate_rows) > 0 and not missing["counts_by_field"]
        ),
        "plausible_alternates": plausible,
    }


def _selected_candidate_id(
    report: Mapping[str, object],
    current_selected: Mapping[str, object],
) -> str:
    rerank = _mapping(report.get("fixture_rerank"))
    candidate_id = _optional_string(rerank.get("selected_candidate_id"))
    if candidate_id is not None:
        return candidate_id
    return str(current_selected.get("candidate_id") or "")


def _row_by_candidate_id(
    rows: Sequence[Mapping[str, object]],
    candidate_id: str,
) -> dict[str, object] | None:
    for row in rows:
        if str(row.get("candidate_id") or "") == candidate_id:
            return dict(row)
    return None


def _gate_aligned_selector_key(row: Mapping[str, object]) -> tuple:
    return (
        bool(row.get("fixture_gate_passed", False)),
        -_number_or_large(row.get("fixture_blocker_count")),
        -_number_or_large(row.get("carrion_only_blocker_count")),
        _number_or_small(row.get("passed_horizon_count")),
        bool(row.get("first_horizon_passed", False)),
        _number_or_small(row.get("carrion_only_alive_agents_min")),
        _number_or_small(row.get("carrion_only_alive_agents_mean")),
        _number_or_small(row.get("carrion_only_births_mean")),
        _number_or_small(
            row.get("carrion_only_terminal_hydration_viability_share_mean")
        ),
        _number_or_small(
            row.get("carrion_only_terminal_energy_viability_share_mean")
        ),
        _number_or_small(
            row.get("carrion_only_terminal_health_viability_share_mean")
        ),
        _number_or_small(
            row.get("carrion_only_terminal_matched_diet_viability_share_mean")
        ),
        -_number_or_large(row.get("unsupported_requested_action_count")),
        -_number_or_large(row.get("unsupported_resolved_action_count")),
        -_number_or_large(row.get("carrion_only_dominant_requested_action_share")),
        _number_or_small(row.get("search_score")),
        -_number_or_large(row.get("prefilter_rank")),
        str(row.get("candidate_id") or ""),
    )


def _selector_missing_fields(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    missing_records: list[dict[str, object]] = []
    if not rows:
        return {
            "counts_by_field": {"fixture_rerank_candidates": 1},
            "candidate_count_with_missing_fields": 0,
            "records": [
                {
                    "field": "fixture_rerank_candidates",
                    "reason": "candidate_pool_missing",
                }
            ],
        }
    required_fields = (
        *_GATE_ALIGNED_SELECTOR_FIELDS,
        *_CANDIDATE_RECOVERY_PROBE_FIELDS,
    )
    for row in rows:
        candidate_id = row.get("candidate_id")
        for field in required_fields:
            if field == "fixture_gate_passed" or field == "first_horizon_passed":
                missing = _optional_bool(row.get(field)) is None
            else:
                missing = _optional_number(row.get(field)) is None
            if missing:
                missing_records.append(
                    {
                        "candidate_id": candidate_id,
                        "field": field,
                        "reason": "metric_missing_or_non_numeric",
                    }
                )
    by_field = Counter(str(record["field"]) for record in missing_records)
    missing_candidate_ids = {
        str(record.get("candidate_id", "unknown"))
        for record in missing_records
    }
    return {
        "counts_by_field": dict(sorted(by_field.items())),
        "candidate_count_with_missing_fields": len(missing_candidate_ids),
        "records": missing_records[:50],
    }


def _selector_candidate_summary(
    row: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        "candidate_id": row.get("candidate_id"),
        "selected": bool(row.get("selected", False)),
        "fixture_gate_passed": _optional_bool(row.get("fixture_gate_passed")),
        "fixture_blocker_count": _optional_number(row.get("fixture_blocker_count")),
        "carrion_only_blocker_count": _optional_number(
            row.get("carrion_only_blocker_count")
        ),
        "passed_horizon_count": _optional_number(row.get("passed_horizon_count")),
        "first_horizon_passed": _optional_bool(row.get("first_horizon_passed")),
        "carrion_only_alive_agents_mean": _optional_number(
            row.get("carrion_only_alive_agents_mean")
        ),
        "carrion_only_births_mean": _optional_number(
            row.get("carrion_only_births_mean")
        ),
        "carrion_only_terminal_hydration_viability_share_mean": _optional_number(
            row.get("carrion_only_terminal_hydration_viability_share_mean")
        ),
        "carrion_only_terminal_energy_viability_share_mean": _optional_number(
            row.get("carrion_only_terminal_energy_viability_share_mean")
        ),
        "carrion_only_terminal_health_viability_share_mean": _optional_number(
            row.get("carrion_only_terminal_health_viability_share_mean")
        ),
        "carrion_only_terminal_matched_diet_viability_share_mean": _optional_number(
            row.get("carrion_only_terminal_matched_diet_viability_share_mean")
        ),
        "unsupported_requested_action_count": _optional_number(
            row.get("unsupported_requested_action_count")
        ),
        "unsupported_resolved_action_count": _optional_number(
            row.get("unsupported_resolved_action_count")
        ),
        "carrion_only_dominant_requested_action_share": _optional_number(
            row.get("carrion_only_dominant_requested_action_share")
        ),
        "post_contact_survival_rate": _optional_number(
            row.get("post_contact_survival_rate")
        ),
        "drink_after_carrion_rate": _optional_number(
            row.get("drink_after_carrion_rate")
        ),
        "mean_hydration_delta_after_carrion": _optional_number(
            row.get("mean_hydration_delta_after_carrion")
        ),
        "mean_energy_delta_after_carrion": _optional_number(
            row.get("mean_energy_delta_after_carrion")
        ),
        "mean_health_delta_after_carrion": _optional_number(
            row.get("mean_health_delta_after_carrion")
        ),
        "mean_water_distance": _optional_number(row.get("mean_water_distance")),
        "search_score": _optional_number(row.get("search_score")),
        "prefilter_rank": _optional_int(row.get("prefilter_rank")),
    }


def _plausible_alternates(
    rows: Sequence[Mapping[str, object]],
    current: Mapping[str, object],
) -> list[dict[str, object]]:
    current_id = str(current.get("candidate_id") or "")
    alternates: list[dict[str, object]] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id == current_id:
            continue
        reasons = _alternate_reasons(row, current)
        if not reasons:
            continue
        alternates.append(
            {
                "candidate_id": candidate_id,
                "reasons": reasons,
                "candidate": _selector_candidate_summary(row),
            }
        )
    alternates.sort(
        key=lambda item: _gate_aligned_selector_key(
            _mapping(item.get("candidate"))
        ),
        reverse=True,
    )
    return alternates[:10]


def _alternate_reasons(
    row: Mapping[str, object],
    current: Mapping[str, object],
) -> list[str]:
    reasons: list[str] = []
    for field in ("fixture_blocker_count", "carrion_only_blocker_count"):
        value = _optional_number(row.get(field))
        current_value = _optional_number(current.get(field))
        if value is not None and current_value is not None and value < current_value:
            reasons.append(f"lower_{field}")
    for field in (
        "post_contact_survival_rate",
        "drink_after_carrion_rate",
        "mean_hydration_delta_after_carrion",
        "mean_energy_delta_after_carrion",
        "mean_health_delta_after_carrion",
    ):
        value = _optional_number(row.get(field))
        current_value = _optional_number(current.get(field))
        if value is not None and current_value is not None and value > current_value:
            reasons.append(f"higher_{field}")
    for field in (
        "mean_water_distance",
        "unsupported_requested_action_count",
        "unsupported_resolved_action_count",
        "carrion_only_dominant_requested_action_share",
    ):
        value = _optional_number(row.get(field))
        current_value = _optional_number(current.get(field))
        if value is not None and current_value is not None and value < current_value:
            reasons.append(f"lower_{field}")
    return reasons


def _metric_correlation(
    rows: Sequence[Mapping[str, object]],
    left: str,
    right: str,
) -> dict[str, object]:
    pairs: list[tuple[float, float]] = []
    labels: list[str] = []
    for row in rows:
        left_value = _optional_number(row.get(left))
        right_value = _optional_number(row.get(right))
        if left_value is None or right_value is None:
            continue
        pairs.append((left_value, right_value))
        labels.append(str(row.get("label") or row.get("candidate_id") or "unknown"))
    pearson = _pearson(pairs)
    spearman = _pearson(_rank_pairs(pairs)) if len(pairs) >= 2 else None
    return {
        "left": left,
        "right": right,
        "n": len(pairs),
        "pearson": _round_optional(pearson),
        "spearman": _round_optional(spearman),
        "direction": _direction(pearson),
        "labels": labels,
    }


def _diagnosis(
    *,
    selected_rows: Sequence[Mapping[str, object]],
    generation_rows: Sequence[Mapping[str, object]],
    rerank_rows: Sequence[Mapping[str, object]],
    missing_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected_correlations = _correlation_report(
        selected_rows,
        _SELECTED_CORRELATION_PAIRS,
    )
    score_gain = _correlation_value(
        selected_correlations,
        "search_score_vs_animal_resource_gain_total_mean",
    )
    score_births = _correlation_value(
        selected_correlations,
        "search_score_vs_births_mean",
    )
    recovery_correlations = {
        metric: _correlation_value(
            selected_correlations,
            f"search_score_vs_{metric}",
        )
        for metric in _RECOVERY_METRICS
    }
    contact_gain_signal = max(
        [
            value
            for value in (score_gain, score_births)
            if value is not None
        ]
        or [0.0]
    )
    recovery_worst_signal = min(
        [
            value
            for value in recovery_correlations.values()
            if value is not None
        ]
        or [0.0]
    )
    findings: list[dict[str, object]] = []
    if contact_gain_signal >= 0.5 and recovery_worst_signal <= -0.5:
        findings.append(
            {
                "finding": "objective_mismatch_contact_gain_over_recovery",
                "severity": 3,
                "evidence": {
                    "search_score_vs_animal_resource_gain_total_mean": (
                        _round_optional(score_gain)
                    ),
                    "search_score_vs_births_mean": _round_optional(score_births),
                    "worst_search_score_vs_recovery_metric": (
                        _round(recovery_worst_signal)
                    ),
                    "recovery_correlations": {
                        key: _round_optional(value)
                        for key, value in recovery_correlations.items()
                    },
                },
                "interpretation": (
                    "Selected reports rank score with carrion gain or births while "
                    "higher score is anti-correlated with post-contact recovery."
                ),
            }
        )
    fixture_blockers = sum(
        int(_optional_number(row.get("fixture_gate_blocker_count")) or 0)
        for row in selected_rows
    )
    carrion_blockers = sum(
        int(_optional_number(row.get("carrion_only_blocker_count")) or 0)
        for row in selected_rows
    )
    if fixture_blockers or carrion_blockers:
        findings.append(
            {
                "finding": "selected_candidates_retain_fixture_blockers",
                "severity": 2,
                "evidence": {
                    "selected_fixture_blocker_count_total": fixture_blockers,
                    "selected_carrion_only_blocker_count_total": carrion_blockers,
                },
                "interpretation": (
                    "Selected candidates still carry fixture blockers, so broad "
                    "score and fixture-rerank pressure are not sufficient evidence "
                    "of carrion recovery."
                ),
            }
        )
    if any(record.get("field") in _RECOVERY_METRICS for record in missing_records):
        findings.append(
            {
                "finding": "recovery_trace_data_missing_for_some_inputs",
                "severity": 1,
                "evidence": _missing_data_summary(missing_records)["counts_by_field"],
                "interpretation": (
                    "Recovery-pressure conclusions are weaker where trace reports "
                    "or post-contact fields are absent."
                ),
            }
        )
    findings.sort(key=lambda item: (-int(item["severity"]), str(item["finding"])))
    ranked_findings = [
        {"rank": index + 1, **finding}
        for index, finding in enumerate(findings)
    ]
    if ranked_findings and ranked_findings[0]["finding"] == (
        "objective_mismatch_contact_gain_over_recovery"
    ):
        assessment = "contact_gain_birth_pressure_over_recovery_likely"
    elif fixture_blockers or carrion_blockers:
        assessment = "fixture_recovery_pressure_insufficient_or_inconclusive"
    else:
        assessment = "objective_mismatch_not_detected_from_available_data"
    return {
        "objective_pressure_assessment": assessment,
        "selected_candidate_count": len(selected_rows),
        "generation_candidate_count": len(generation_rows),
        "fixture_rerank_candidate_count": len(rerank_rows),
        "ranked_findings": ranked_findings,
    }


def _record_missing_metrics(
    row: Mapping[str, object],
    *,
    required_metrics: Sequence[str],
    missing_records: list[dict[str, object]],
) -> None:
    for metric in required_metrics:
        if _optional_number(row.get(metric)) is not None:
            continue
        if row.get(metric) is not None and metric not in {
            "dominant_requested_action_share",
        }:
            continue
        missing_records.append(
            {
                "label": row.get("label"),
                "source": row.get("source"),
                "candidate_id": row.get("candidate_id"),
                "field": metric,
                "reason": "metric_missing_or_non_numeric",
            }
        )


def _missing_data_summary(
    missing_records: Sequence[Mapping[str, object]],
    *,
    example_limit: int = 25,
) -> dict[str, object]:
    by_field = Counter(str(record.get("field", "unknown")) for record in missing_records)
    by_label = Counter(str(record.get("label", "unknown")) for record in missing_records)
    by_reason = Counter(str(record.get("reason", "unknown")) for record in missing_records)
    return {
        "total_count": len(missing_records),
        "counts_by_field": dict(sorted(by_field.items())),
        "counts_by_label": dict(sorted(by_label.items())),
        "counts_by_reason": dict(sorted(by_reason.items())),
        "examples": [dict(record) for record in missing_records[:example_limit]],
    }


def _rank_rows(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    *,
    reverse: bool,
    limit: int,
) -> list[dict[str, object]]:
    ranked = [
        row
        for row in rows
        if _optional_number(row.get(metric)) is not None
    ]
    ranked.sort(
        key=lambda row: (
            _optional_number(row.get(metric)) or 0.0,
            str(row.get("label", "")),
            str(row.get("candidate_id", "")),
        ),
        reverse=reverse,
    )
    return [
        {
            "rank": index + 1,
            "label": row.get("label"),
            "source": row.get("source"),
            "candidate_id": row.get("candidate_id"),
            "selected": bool(row.get("selected")),
            metric: _optional_number(row.get(metric)),
            "search_score": _optional_number(row.get("search_score")),
            "births_mean": _optional_number(row.get("births_mean")),
            "animal_resource_gain_total_mean": _optional_number(
                row.get("animal_resource_gain_total_mean")
            ),
            "carrion_only_animal_resource_gain_mean": _optional_number(
                row.get("carrion_only_animal_resource_gain_mean")
            ),
            "carrion_only_blocker_count": _optional_number(
                row.get("carrion_only_blocker_count")
            ),
        }
        for index, row in enumerate(ranked[:limit])
    ]


def _audit_contract() -> dict[str, object]:
    return {
        "offline_only": True,
        "report_only": True,
        "changes_controller_code": False,
        "changes_score_weights": False,
        "changes_action_masks": False,
        "changes_gates": False,
        "changes_candidate_selection": False,
        "uses_runtime_fixture_identity": False,
        "selector_probe_policy": MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY,
        "selector_probe_role": (
            "retrospective report-only comparison; active selector not implemented"
        ),
        "input_sources": [
            "mind_v3_search_report_json",
            "mind_v3_carrion_autopsy_trace_json",
        ],
        "question": (
            "whether search/rerank objective pressure favors carrion contact, "
            "gain, or births over post-contact hydration recovery and survival"
        ),
    }


def _label_from_path(path: Path) -> str:
    stem = path.stem
    for marker in ("mind-v3-", "-search", "-carrion"):
        stem = stem.replace(marker, "-")
    parts = [part for part in stem.split("-") if part]
    for part in parts:
        if part.startswith("v") and part[1:].isdigit():
            return part
    return path.stem


def _correlation_value(report: Mapping[str, object], key: str) -> float | None:
    item = report.get(key)
    if not isinstance(item, Mapping):
        return None
    return _optional_number(item.get("pearson"))


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    value = numerator / math.sqrt(x_var * y_var)
    if not math.isfinite(value):
        return None
    return max(-1.0, min(1.0, value))


def _rank_pairs(pairs: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    xs = _ranks([pair[0] for pair in pairs])
    ys = _ranks([pair[1] for pair in pairs])
    return list(zip(xs, ys, strict=True))


def _ranks(values: Sequence[float]) -> list[float]:
    sorted_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_values):
        end = index + 1
        while end < len(sorted_values) and sorted_values[end][1] == sorted_values[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _value in sorted_values[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _direction(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value > 0.05:
        return "positive"
    if value < -0.05:
        return "negative"
    return "flat"


def _dominant_item(counter: Mapping[str, int]) -> tuple[str | None, int]:
    if not counter:
        return None, 0
    action, count = sorted(
        counter.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[0]
    return str(action), int(count)


def _counter_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            continue
        counts[str(key)] = int(raw_count)
    return counts


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _first_number(*values: object) -> float | None:
    for value in values:
        number = _optional_number(value)
        if number is not None:
            return number
    return None


def _fill_missing_number(
    row: dict[str, object],
    key: str,
    fallback: object,
) -> None:
    if _optional_number(row.get(key)) is not None:
        return
    number = _optional_number(fallback)
    if number is not None:
        row[key] = number


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
        if math.isfinite(number):
            return _round(number)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    return None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    return None


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _number_or_small(value: object) -> float:
    number = _optional_number(value)
    return number if number is not None else -1.0e12


def _number_or_large(value: object) -> float:
    number = _optional_number(value)
    return number if number is not None else 1.0e12


def _round(value: float) -> float:
    return round(float(value), 6)


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return _round(value)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _round(float(numerator) / float(denominator))
