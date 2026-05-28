from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.dataset import TrajectoryJsonlDataset
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.rollout_context import (
    MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
    RolloutContextConfig,
    rollout_context_feature_contract,
    rollout_context_event_from_record,
)

MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION = (
    "mind_v3_post_carrion_rollout_context_coverage_audit_v1"
)
MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_POLICY = (
    "diagnostics_only_post_carrion_rollout_context_coverage_and_fixture_failure_v1"
)
DEFAULT_MIN_BROAD_POST_CARRION_CONTEXT_SHARE = 0.02
DEFAULT_MIN_BRANCH_ORACLE_EXACT_MATCH_SHARE = 0.5
DEFAULT_MIN_TRAJECTORY_POST_CARRION_CONTEXT_SHARE = 0.02
DEFAULT_MIN_MOVEMENT_UNSUPPORTED_RESOLUTION_SHARE = 0.8
DEFAULT_MIN_CONTEXT_UNSUPPORTED_RESOLUTION_LIFT = 0.1
_SEED_PATTERN = re.compile(r"(?:seed[-_=]?|mind-v3-|heuristic-)(\d+)")


class PostCarrionRolloutContextAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _LoadedReport:
    name: str
    path: str | None
    payload: Mapping[str, object] | None
    evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LoadedTrajectories:
    datasets: tuple[TrajectoryJsonlDataset, ...]
    evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LenientTrajectoryLoad:
    dataset: TrajectoryJsonlDataset
    malformed_record_count: int
    malformed_records: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _CoverageResult:
    report: dict[str, object]
    exact_action_index: Mapping[tuple[int, int, int, str], tuple[dict[str, object], ...]]
    state_index: Mapping[tuple[int, int, int], tuple[dict[str, object], ...]]


def load_post_carrion_rollout_context_json(path: str | Path) -> dict[str, object]:
    with _open_input(Path(path)) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PostCarrionRolloutContextAuditError(
            f"report must be a JSON object: {path}"
        )
    return payload


def build_post_carrion_rollout_context_audit_report(
    *,
    rollout_context_report: Mapping[str, object] | None = None,
    rollout_context_report_path: str | Path | None = None,
    baseline_report: Mapping[str, object] | None = None,
    baseline_report_path: str | Path | None = None,
    v64_rollout_context_audit: Mapping[str, object] | None = None,
    v64_rollout_context_audit_path: str | Path | None = None,
    v69_recovery_action_target_audit: Mapping[str, object] | None = None,
    v69_recovery_action_target_audit_path: str | Path | None = None,
    branch_oracle_audit: Mapping[str, object] | None = None,
    branch_oracle_audit_path: str | Path | None = None,
    trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None = None,
    trajectory_paths: Sequence[str | Path] = (),
    trajectory_glob_patterns: Sequence[str] = (),
    min_broad_post_carrion_context_share: float = (
        DEFAULT_MIN_BROAD_POST_CARRION_CONTEXT_SHARE
    ),
    min_branch_oracle_exact_match_share: float = (
        DEFAULT_MIN_BRANCH_ORACLE_EXACT_MATCH_SHARE
    ),
) -> dict[str, object]:
    if min_broad_post_carrion_context_share < 0.0:
        raise PostCarrionRolloutContextAuditError(
            "min_broad_post_carrion_context_share must be non-negative"
        )
    if not 0.0 <= min_branch_oracle_exact_match_share <= 1.0:
        raise PostCarrionRolloutContextAuditError(
            "min_branch_oracle_exact_match_share must be in [0.0, 1.0]"
        )

    loaded_reports = {
        "rollout_context_report": _resolve_report(
            "rollout_context_report",
            rollout_context_report,
            rollout_context_report_path,
        ),
        "baseline_report": _resolve_report(
            "baseline_report",
            baseline_report,
            baseline_report_path,
        ),
        "v64_rollout_context_audit": _resolve_report(
            "v64_rollout_context_audit",
            v64_rollout_context_audit,
            v64_rollout_context_audit_path,
        ),
        "v69_recovery_action_target_audit": _resolve_report(
            "v69_recovery_action_target_audit",
            v69_recovery_action_target_audit,
            v69_recovery_action_target_audit_path,
        ),
        "branch_oracle_audit": _resolve_report(
            "branch_oracle_audit",
            branch_oracle_audit,
            branch_oracle_audit_path,
        ),
    }
    trajectories = _resolve_trajectories(
        trajectory_datasets=trajectory_datasets,
        trajectory_paths=trajectory_paths,
        trajectory_glob_patterns=trajectory_glob_patterns,
    )
    coverage = _trajectory_coverage(trajectories.datasets)
    source_summary = _source_report_summary(
        rollout_context_report=loaded_reports["rollout_context_report"].payload,
        baseline_report=loaded_reports["baseline_report"].payload,
        v64_rollout_context_audit=loaded_reports[
            "v64_rollout_context_audit"
        ].payload,
        v69_recovery_action_target_audit=loaded_reports[
            "v69_recovery_action_target_audit"
        ].payload,
        branch_oracle_audit=loaded_reports["branch_oracle_audit"].payload,
    )
    fixture_audit = _fixture_failure_audit(
        rollout_context_report=loaded_reports["rollout_context_report"].payload,
        baseline_report=loaded_reports["baseline_report"].payload,
        branch_oracle_audit=loaded_reports["branch_oracle_audit"].payload,
        coverage=coverage,
    )
    evidence = {
        "reports": {
            name: loaded.evidence for name, loaded in sorted(loaded_reports.items())
        },
        "trajectories": trajectories.evidence,
    }
    config = _config_section(
        min_broad_post_carrion_context_share=min_broad_post_carrion_context_share,
        min_branch_oracle_exact_match_share=min_branch_oracle_exact_match_share,
    )
    coverage_section = _coverage_section(
        source_report_summary=source_summary,
        fixture_audit=fixture_audit,
        coverage=coverage.report,
        min_broad_post_carrion_context_share=min_broad_post_carrion_context_share,
    )
    temporal_alignment = _temporal_alignment_section(source_summary)
    fixture_action_support = _fixture_action_support_section(
        source_report_summary=source_summary,
        fixture_audit=fixture_audit,
        min_branch_oracle_exact_match_share=min_branch_oracle_exact_match_share,
    )
    policy_scoring_pressure = _policy_scoring_pressure_section(
        fixture_audit=fixture_audit,
        coverage=coverage.report,
    )
    unsupported_resolution = _unsupported_resolution_section(
        source_report_summary=source_summary,
        coverage=coverage.report,
    )
    seed29_birth_regression = _seed29_birth_regression_section(source_summary)
    planned_sections = {
        "coverage": coverage_section,
        "temporal_alignment": temporal_alignment,
        "fixture_action_support": fixture_action_support,
        "policy_scoring_pressure": policy_scoring_pressure,
        "unsupported_resolution": unsupported_resolution,
        "seed29_birth_regression": seed29_birth_regression,
    }
    classification = _classification(
        evidence=evidence,
        planned_sections=planned_sections,
    )
    research_recommendation = _research_recommendation_section(
        planned_sections=planned_sections,
        classification=classification,
    )
    contract = {
        "schema_version": MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_POLICY,
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "artifact_promotion_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "gate_effect": "none",
        "fixture_floor_effect": "none",
        "action_mask_effect": "none",
        "cache_internal_effect": "none",
        "missing_evidence_policy": "classify_inconclusive_without_runtime_workaround_v1",
        "rollout_context_report_option": "--rollout-context-report",
        "rollout_context_feature_policy": MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
        "rollout_context_feature_contract": rollout_context_feature_contract(
            RolloutContextConfig()
        ),
    }
    return {
        "schema_version": MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_POLICY,
        "contract": contract,
        "provenance": {"contract_digest": stable_payload_digest(contract)},
        "evidence": evidence,
        "config": config,
        "coverage": coverage_section,
        "temporal_alignment": temporal_alignment,
        "fixture_action_support": fixture_action_support,
        "policy_scoring_pressure": policy_scoring_pressure,
        "unsupported_resolution": unsupported_resolution,
        "seed29_birth_regression": seed29_birth_regression,
        "research_recommendation": research_recommendation,
        "source_report_summary": source_summary,
        "trajectory_rollout_context_coverage": coverage.report,
        "fixture_failure_audit": fixture_audit,
        "classification": classification,
        "non_promoted": True,
    }


def write_post_carrion_rollout_context_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _resolve_report(
    name: str,
    payload: Mapping[str, object] | None,
    path: str | Path | None,
) -> _LoadedReport:
    path_string = str(path) if path is not None else None
    if payload is not None:
        return _LoadedReport(
            name=name,
            path=path_string,
            payload=payload,
            evidence=_report_evidence(
                path=path_string,
                payload=payload,
                loaded=True,
                load_error=None,
                source="inline" if path is None else "inline_with_path",
            ),
        )
    if path is None:
        return _LoadedReport(
            name=name,
            path=None,
            payload=None,
            evidence=_report_evidence(
                path=None,
                payload=None,
                loaded=False,
                load_error="not_provided",
                source="missing",
            ),
        )
    try:
        loaded = load_post_carrion_rollout_context_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _LoadedReport(
            name=name,
            path=path_string,
            payload=None,
            evidence=_report_evidence(
                path=path_string,
                payload=None,
                loaded=False,
                load_error=f"{type(exc).__name__}: {exc}",
                source="path",
            ),
        )
    return _LoadedReport(
        name=name,
        path=path_string,
        payload=loaded,
        evidence=_report_evidence(
            path=path_string,
            payload=loaded,
            loaded=True,
            load_error=None,
            source="path",
        ),
    )


def _report_evidence(
    *,
    path: str | None,
    payload: Mapping[str, object] | None,
    loaded: bool,
    load_error: str | None,
    source: str,
) -> dict[str, object]:
    return {
        "path": path,
        "source": source,
        "loaded": loaded,
        "load_error": load_error,
        "schema_version": payload.get("schema_version") if payload is not None else None,
        "digest": stable_payload_digest(payload) if payload is not None else None,
    }


def _resolve_trajectories(
    *,
    trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None,
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
) -> _LoadedTrajectories:
    if trajectory_datasets is not None:
        datasets = tuple(trajectory_datasets)
        return _LoadedTrajectories(
            datasets=datasets,
            evidence={
                "source": "inline",
                "requested_paths": [str(path) for path in trajectory_paths],
                "requested_globs": list(trajectory_glob_patterns),
                "loaded_path_count": len(datasets),
                "load_failure_count": 0,
                "load_failures": [],
                "malformed_record_count": 0,
                "malformed_records": [],
                "loaded_paths": [str(dataset.path) for dataset in datasets],
            },
        )
    datasets: list[TrajectoryJsonlDataset] = []
    failures: list[dict[str, object]] = []
    malformed_record_count = 0
    malformed_records: list[dict[str, object]] = []
    for path in trajectory_paths:
        try:
            loaded = _load_trajectory_jsonl_lenient(path)
            datasets.append(loaded.dataset)
            malformed_record_count += loaded.malformed_record_count
            malformed_records.extend(loaded.malformed_records)
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return _LoadedTrajectories(
        datasets=tuple(datasets),
        evidence={
            "source": "path",
            "requested_paths": [str(path) for path in trajectory_paths],
            "requested_globs": list(trajectory_glob_patterns),
            "loaded_path_count": len(datasets),
            "load_failure_count": len(failures),
            "load_failures": failures,
            "malformed_record_count": malformed_record_count,
            "malformed_records": malformed_records[:12],
            "loaded_paths": [str(dataset.path) for dataset in datasets],
        },
    )


def _load_trajectory_jsonl_lenient(path: str | Path) -> _LenientTrajectoryLoad:
    resolved_path = Path(path)
    payloads: list[dict[str, object]] = []
    with _open_input(resolved_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise PostCarrionRolloutContextAuditError(
                    f"trajectory payload at line {line_number} must be a JSON object"
                )
            payloads.append(payload)
    if len(payloads) < 2:
        raise PostCarrionRolloutContextAuditError(
            "trajectory JSONL must include header and footer"
        )
    records: list[dict[str, object]] = []
    malformed_records: list[dict[str, object]] = []
    for index, payload in enumerate(payloads[1:-1]):
        try:
            records.append(_trajectory_record_payload(payload, index))
        except PostCarrionRolloutContextAuditError as exc:
            malformed_records.append(
                {
                    "path": str(resolved_path),
                    "record_index": index,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
    dataset = TrajectoryJsonlDataset(
        path=resolved_path,
        header=dict(payloads[0]),
        records=tuple(records),
        footer=dict(payloads[-1]),
    )
    return _LenientTrajectoryLoad(
        dataset=dataset,
        malformed_record_count=len(malformed_records),
        malformed_records=tuple(malformed_records),
    )


def _trajectory_record_payload(
    payload: Mapping[str, object],
    index: int,
) -> dict[str, object]:
    record = payload.get("record")
    if isinstance(record, Mapping):
        return dict(record)
    if payload.get("type") == "record":
        return dict(payload)
    raise PostCarrionRolloutContextAuditError(
        f"trajectory record {index} must be wrapped as a record payload"
    )


def _source_report_summary(
    *,
    rollout_context_report: Mapping[str, object] | None,
    baseline_report: Mapping[str, object] | None,
    v64_rollout_context_audit: Mapping[str, object] | None,
    v69_recovery_action_target_audit: Mapping[str, object] | None,
    branch_oracle_audit: Mapping[str, object] | None,
) -> dict[str, object]:
    rollout_summary = _search_report_summary(rollout_context_report)
    baseline_summary = _search_report_summary(baseline_report)
    return {
        "rollout_context_report": rollout_summary,
        "baseline_report": baseline_summary,
        "broad_holdout_deltas_vs_baseline": _broad_holdout_deltas(
            rollout_context_report,
            baseline_report,
        ),
        "v64_rollout_context_audit": _v64_summary(v64_rollout_context_audit),
        "v69_recovery_action_target_audit": _recovery_action_target_summary(
            v69_recovery_action_target_audit
        ),
        "branch_oracle_audit": _branch_oracle_summary(branch_oracle_audit),
    }


def _search_report_summary(
    report: Mapping[str, object] | None,
) -> dict[str, object]:
    if report is None:
        return {"loaded": False}
    best = _mapping(report.get("best_candidate"))
    metadata = _mapping(best.get("controller_metadata"))
    aggregate = _mapping(_mapping(report.get("holdout_evaluation")).get("aggregate"))
    fixture_gate = _mapping(report.get("fixture_gate"))
    return {
        "loaded": True,
        "schema_version": report.get("schema_version"),
        "best_candidate": {
            "candidate_id": best.get("candidate_id"),
            "generation_index": best.get("generation_index"),
            "score": _number_or_none(best.get("score")),
            "controller_architecture": metadata.get("architecture"),
            "controller_schema_version": metadata.get("schema_version"),
        },
        "holdout_aggregate": _selected_holdout_metrics(aggregate),
        "fixture_gate": {
            "passed": fixture_gate.get("passed"),
            "blocker_count": len(_list(fixture_gate.get("blockers"))),
            "fixture_names": _list(fixture_gate.get("fixture_names")),
            "horizon_ticks": _list(fixture_gate.get("horizon_ticks")),
            "carrion_only_120": _fixture_horizon_payload(
                report,
                fixture_name="carrion_only",
                horizon="120",
            ),
        },
    }


def _selected_holdout_metrics(
    aggregate: Mapping[str, object],
) -> dict[str, object]:
    keys = (
        "alive_agents_mean",
        "births_mean",
        "deaths_mean",
        "movement_event_rate",
        "dominant_requested_action",
        "dominant_requested_action_count",
        "dominant_requested_action_share",
        "heuristic_action_source_count",
        "unsupported_requested_action_count",
        "unsupported_resolved_action_count",
        "unsupported_requested_action_breakdown",
        "unsupported_resolved_action_breakdown",
        "rollout_context_decision_count",
        "rollout_context_non_empty_count",
        "rollout_context_non_empty_share",
        "rollout_context_post_carrion_context_count",
        "rollout_context_post_carrion_context_share",
        "rollout_context_selected_score_delta_abs_mean",
        "rollout_context_selected_score_delta_abs_max",
    )
    return {key: aggregate.get(key) for key in keys if key in aggregate}


def _broad_holdout_deltas(
    candidate_report: Mapping[str, object] | None,
    baseline_report: Mapping[str, object] | None,
) -> dict[str, object]:
    candidate = _mapping(
        _mapping(candidate_report or {}).get("holdout_evaluation")
    )
    baseline = _mapping(_mapping(baseline_report or {}).get("holdout_evaluation"))
    candidate_aggregate = _mapping(candidate.get("aggregate"))
    baseline_aggregate = _mapping(baseline.get("aggregate"))
    metric_keys = (
        "alive_agents_mean",
        "births_mean",
        "deaths_mean",
        "movement_event_rate",
        "unsupported_resolved_action_count",
        "dominant_requested_action_share",
        "rollout_context_post_carrion_context_share",
    )
    aggregate_deltas = {
        key: _delta(candidate_aggregate.get(key), baseline_aggregate.get(key))
        for key in metric_keys
    }
    baseline_runs = {
        _int(run.get("seed")): run
        for run in _list(baseline.get("runs"))
        if isinstance(run, Mapping)
    }
    per_seed: list[dict[str, object]] = []
    for run in _list(candidate.get("runs")):
        if not isinstance(run, Mapping):
            continue
        seed = _int(run.get("seed"))
        baseline_run = _mapping(baseline_runs.get(seed))
        per_seed.append(
            {
                "seed": seed,
                "alive_agents_delta": _delta(
                    run.get("alive_agents"),
                    baseline_run.get("alive_agents"),
                ),
                "births_delta": _delta(run.get("births"), baseline_run.get("births")),
                "deaths_delta": _delta(run.get("deaths"), baseline_run.get("deaths")),
                "movement_event_rate_delta": _delta(
                    run.get("movement_event_rate"),
                    baseline_run.get("movement_event_rate"),
                ),
                "unsupported_resolved_action_delta": _delta(
                    run.get("unsupported_resolved_action_count"),
                    baseline_run.get("unsupported_resolved_action_count"),
                ),
                "rollout_context_post_carrion_context_count": run.get(
                    "rollout_context_post_carrion_context_count"
                ),
            }
        )
    return {
        "aggregate": aggregate_deltas,
        "per_seed": sorted(per_seed, key=lambda item: int(item.get("seed", -1))),
    }


def _v64_summary(report: Mapping[str, object] | None) -> dict[str, object]:
    if report is None:
        return {"loaded": False}
    assessment = _mapping(report.get("failure_mode_assessment"))
    split = _mapping(report.get("train_heldout_split"))
    return {
        "loaded": True,
        "schema_version": report.get("schema_version"),
        "materially_improves_v62_failure_mode": assessment.get(
            "materially_improves_v62_failure_mode"
        ),
        "movement_drink_stay_absolute_rate_reduction": assessment.get(
            "movement_drink_stay_absolute_rate_reduction"
        ),
        "post_carrion_absolute_rate_reduction": assessment.get(
            "post_carrion_absolute_rate_reduction"
        ),
        "blockers": _list(assessment.get("blockers")),
        "train_record_count": split.get("train_record_count"),
        "heldout_record_count": split.get("heldout_record_count"),
    }


def _recovery_action_target_summary(
    report: Mapping[str, object] | None,
) -> dict[str, object]:
    if report is None:
        return {"loaded": False}
    classification = _mapping(report.get("classification"))
    scoring = _mapping(report.get("heldout_scoring"))
    oracle = _mapping(report.get("oracle_comparison"))
    return {
        "loaded": True,
        "schema_version": report.get("schema_version"),
        "classification_primary": classification.get("primary"),
        "classification_labels": _list(classification.get("labels")),
        "decision_row_count": scoring.get("decision_row_count"),
        "oracle_loaded": oracle.get("loaded"),
        "oracle_matched_row_count": oracle.get("matched_row_count"),
        "oracle_unmatched_row_count": oracle.get("unmatched_row_count"),
    }


def _branch_oracle_summary(report: Mapping[str, object] | None) -> dict[str, object]:
    if report is None:
        return {"loaded": False}
    aggregate = _mapping(report.get("aggregate"))
    acceptance = _mapping(report.get("acceptance"))
    return {
        "loaded": True,
        "schema_version": report.get("schema_version"),
        "branch_point_count": aggregate.get("branch_point_count"),
        "oracle_changed_action_count": aggregate.get("oracle_changed_action_count"),
        "material_oracle_gain_count": aggregate.get("material_oracle_gain_count"),
        "terminal_alive_gain_total_vs_logged": aggregate.get(
            "terminal_alive_gain_total_vs_logged"
        ),
        "birth_gain_total_vs_logged": aggregate.get("birth_gain_total_vs_logged"),
        "zero_heuristic_runtime_actions": aggregate.get(
            "zero_heuristic_runtime_actions"
        ),
        "oracle_best_action_counts": _mapping(
            aggregate.get("oracle_best_action_counts")
        ),
        "diagnostic_acceptance_passed": acceptance.get(
            "diagnostic_acceptance_passed"
        ),
        "materially_supports_branch_action_oracle": acceptance.get(
            "materially_supports_branch_action_oracle"
        ),
        "blocker_count": len(_list(acceptance.get("blockers"))),
    }


def _trajectory_coverage(
    datasets: Sequence[TrajectoryJsonlDataset],
) -> _CoverageResult:
    aggregate = _empty_coverage_acc()
    by_seed: dict[int, dict[str, object]] = {}
    by_source: list[dict[str, object]] = []
    exact_index: dict[tuple[int, int, int, str], list[dict[str, object]]] = {}
    state_index: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for dataset in datasets:
        seed = _dataset_seed(dataset)
        source_acc = _empty_coverage_acc()
        seed_acc = by_seed.setdefault(seed, _empty_coverage_acc())
        for record_index, record in enumerate(dataset.records):
            row = _coverage_row(dataset, record, record_index, seed=seed)
            _update_coverage_acc(aggregate, row)
            _update_coverage_acc(source_acc, row)
            _update_coverage_acc(seed_acc, row)
            if row["decision_row"]:
                exact_key = (
                    seed,
                    _int(row.get("tick")),
                    _int(row.get("agent_id")),
                    str(row.get("requested_action")),
                )
                state_key = (
                    seed,
                    _int(row.get("tick")),
                    _int(row.get("agent_id")),
                )
                exact_index.setdefault(exact_key, []).append(_branch_join_row(row))
                state_index.setdefault(state_key, []).append(_branch_join_row(row))
        source_summary = _finalize_coverage_acc(source_acc)
        source_summary.update(_dataset_footer_summary(dataset))
        source_summary["path"] = str(dataset.path)
        source_summary["seed"] = seed
        source_summary["source_kind"] = _source_kind(dataset.path)
        by_source.append(source_summary)
    aggregate_summary = _finalize_coverage_acc(aggregate)
    aggregate_summary["trajectory_count"] = len(datasets)
    aggregate_summary["terminal_summary"] = _terminal_summary(datasets)
    by_seed_summary = {
        str(seed): _finalize_coverage_acc(acc)
        for seed, acc in sorted(by_seed.items())
    }
    return _CoverageResult(
        report={
            "aggregate": aggregate_summary,
            "by_seed": by_seed_summary,
            "by_source": sorted(by_source, key=lambda item: str(item.get("path"))),
        },
        exact_action_index={
            key: tuple(value) for key, value in sorted(exact_index.items())
        },
        state_index={key: tuple(value) for key, value in sorted(state_index.items())},
    )


def _empty_coverage_acc() -> dict[str, object]:
    return {
        "record_count": 0,
        "decision_row_count": 0,
        "rollout_context_diagnostic_count": 0,
        "rollout_context_non_empty_count": 0,
        "rollout_context_post_carrion_context_count": 0,
        "update_trace_count": 0,
        "previous_context_post_carrion_count": 0,
        "animal_resource_gain_record_count": 0,
        "heuristic_action_source_count": 0,
        "unsupported_resolved_action_count": 0,
        "unsupported_resolved_post_carrion_context_count": 0,
        "action_source_counts": Counter(),
        "requested_action_counts": Counter(),
        "resolved_action_counts": Counter(),
        "unsupported_requested_action_counts": Counter(),
        "unsupported_resolved_action_counts": Counter(),
        "unsupported_invalid_reason_counts": Counter(),
        "post_carrion_unsupported_requested_action_counts": Counter(),
        "post_carrion_requested_action_counts": Counter(),
        "post_carrion_resolved_action_counts": Counter(),
        "all_selected_score_deltas": [],
        "post_carrion_selected_score_deltas": [],
        "examples": [],
    }


def _coverage_row(
    dataset: TrajectoryJsonlDataset,
    record: Mapping[str, object],
    record_index: int,
    *,
    seed: int,
) -> dict[str, object]:
    diagnostics = _mapping(record.get("policy_decision_diagnostics"))
    update_trace = _mapping(record.get("policy_update_trace"))
    rollout_update = _mapping(update_trace.get("rollout_context_update_trace"))
    previous_context = _mapping(rollout_update.get("previous_context"))
    post_from_diag = _optional_bool(
        diagnostics.get("rollout_context_post_carrion_context")
    )
    previous_post = _optional_bool(previous_context.get("post_carrion_contact"))
    post_carrion = post_from_diag if post_from_diag is not None else previous_post
    non_empty = _optional_bool(diagnostics.get("rollout_context_non_empty"))
    if non_empty is None and previous_context:
        non_empty = _context_non_empty(previous_context)
    score_delta = _number_or_none(
        diagnostics.get("rollout_context_selected_score_delta")
    )
    event = rollout_context_event_from_record(record)
    requested = str(record.get("requested_action", "unknown"))
    resolved = str(record.get("resolved_action", "unknown"))
    action_source = str(record.get("action_source", "unknown"))
    outcome = _mapping(record.get("outcome"))
    resolution_action_valid = _optional_bool(record.get("resolution_action_valid"))
    if resolution_action_valid is None:
        resolution_action_valid = _optional_bool(
            outcome.get("resolution_action_valid")
        )
    invalid_reason = record.get("invalid_reason")
    if invalid_reason is None:
        invalid_reason = outcome.get("invalid_reason")
    return {
        "path": str(dataset.path),
        "source_kind": _source_kind(dataset.path),
        "seed": seed,
        "record_index": record_index,
        "tick": record.get("tick"),
        "agent_id": record.get("agent_id"),
        "requested_action": requested,
        "resolved_action": resolved,
        "action_source": action_source,
        "decision_row": action_source != "passive",
        "resolution_action_valid": resolution_action_valid,
        "unsupported_resolved_action": resolution_action_valid is False,
        "invalid_reason": invalid_reason,
        "rollout_context_diagnostic_present": bool(diagnostics),
        "rollout_context_non_empty": non_empty,
        "rollout_context_post_carrion_context": post_carrion,
        "update_trace_present": bool(rollout_update),
        "previous_context_post_carrion": previous_post,
        "animal_resource_gain": bool(event.get("animal_resource_gain")),
        "score_delta": score_delta,
        "previous_context": {
            "ticks_since_animal_resource_gain": previous_context.get(
                "ticks_since_animal_resource_gain"
            ),
            "ticks_since_drink": previous_context.get("ticks_since_drink"),
            "recovery_phase_remaining": previous_context.get(
                "recovery_phase_remaining"
            ),
        },
    }


def _update_coverage_acc(
    acc: dict[str, object],
    row: Mapping[str, object],
) -> None:
    acc["record_count"] = _int(acc.get("record_count")) + 1
    decision = bool(row.get("decision_row"))
    if decision:
        acc["decision_row_count"] = _int(acc.get("decision_row_count")) + 1
        _as_counter(acc["action_source_counts"]).update([str(row.get("action_source"))])
        _as_counter(acc["requested_action_counts"]).update(
            [str(row.get("requested_action"))]
        )
        _as_counter(acc["resolved_action_counts"]).update([str(row.get("resolved_action"))])
        if _is_heuristic_action_source(str(row.get("action_source"))):
            acc["heuristic_action_source_count"] = (
                _int(acc.get("heuristic_action_source_count")) + 1
            )
        if bool(row.get("unsupported_resolved_action")):
            acc["unsupported_resolved_action_count"] = (
                _int(acc.get("unsupported_resolved_action_count")) + 1
            )
            _as_counter(acc["unsupported_requested_action_counts"]).update(
                [str(row.get("requested_action"))]
            )
            _as_counter(acc["unsupported_resolved_action_counts"]).update(
                [str(row.get("resolved_action"))]
            )
            invalid_reason = row.get("invalid_reason")
            _as_counter(acc["unsupported_invalid_reason_counts"]).update(
                [str(invalid_reason if invalid_reason is not None else "unknown")]
            )
            if bool(row.get("rollout_context_post_carrion_context")):
                acc["unsupported_resolved_post_carrion_context_count"] = (
                    _int(acc.get("unsupported_resolved_post_carrion_context_count"))
                    + 1
                )
                _as_counter(
                    acc["post_carrion_unsupported_requested_action_counts"]
                ).update([str(row.get("requested_action"))])
    if bool(row.get("rollout_context_diagnostic_present")):
        acc["rollout_context_diagnostic_count"] = (
            _int(acc.get("rollout_context_diagnostic_count")) + 1
        )
    if bool(row.get("rollout_context_non_empty")):
        acc["rollout_context_non_empty_count"] = (
            _int(acc.get("rollout_context_non_empty_count")) + 1
        )
    if bool(row.get("rollout_context_post_carrion_context")):
        acc["rollout_context_post_carrion_context_count"] = (
            _int(acc.get("rollout_context_post_carrion_context_count")) + 1
        )
        _as_counter(acc["post_carrion_requested_action_counts"]).update(
            [str(row.get("requested_action"))]
        )
        _as_counter(acc["post_carrion_resolved_action_counts"]).update(
            [str(row.get("resolved_action"))]
        )
        examples = acc["examples"]
        if isinstance(examples, list) and len(examples) < 12:
            examples.append(_post_carrion_example(row))
    if bool(row.get("update_trace_present")):
        acc["update_trace_count"] = _int(acc.get("update_trace_count")) + 1
    if bool(row.get("previous_context_post_carrion")):
        acc["previous_context_post_carrion_count"] = (
            _int(acc.get("previous_context_post_carrion_count")) + 1
        )
    if bool(row.get("animal_resource_gain")):
        acc["animal_resource_gain_record_count"] = (
            _int(acc.get("animal_resource_gain_record_count")) + 1
        )
    score_delta = row.get("score_delta")
    if isinstance(score_delta, (int, float)) and not isinstance(score_delta, bool):
        all_values = acc["all_selected_score_deltas"]
        if isinstance(all_values, list):
            all_values.append(float(score_delta))
        if bool(row.get("rollout_context_post_carrion_context")):
            post_values = acc["post_carrion_selected_score_deltas"]
            if isinstance(post_values, list):
                post_values.append(float(score_delta))


def _finalize_coverage_acc(acc: Mapping[str, object]) -> dict[str, object]:
    decision_count = _int(acc.get("decision_row_count"))
    post_count = _int(acc.get("rollout_context_post_carrion_context_count"))
    unsupported_count = _int(acc.get("unsupported_resolved_action_count"))
    unsupported_post_count = _int(
        acc.get("unsupported_resolved_post_carrion_context_count")
    )
    return {
        "record_count": _int(acc.get("record_count")),
        "decision_row_count": decision_count,
        "rollout_context_diagnostic_count": _int(
            acc.get("rollout_context_diagnostic_count")
        ),
        "rollout_context_non_empty_count": _int(
            acc.get("rollout_context_non_empty_count")
        ),
        "rollout_context_non_empty_share": _share(
            _int(acc.get("rollout_context_non_empty_count")),
            decision_count,
        ),
        "rollout_context_post_carrion_context_count": post_count,
        "rollout_context_post_carrion_context_share": _share(
            post_count,
            decision_count,
        ),
        "update_trace_count": _int(acc.get("update_trace_count")),
        "previous_context_post_carrion_count": _int(
            acc.get("previous_context_post_carrion_count")
        ),
        "animal_resource_gain_record_count": _int(
            acc.get("animal_resource_gain_record_count")
        ),
        "heuristic_action_source_count": _int(acc.get("heuristic_action_source_count")),
        "unsupported_resolved_action_count": unsupported_count,
        "unsupported_resolved_action_share": _share(
            unsupported_count,
            decision_count,
        ),
        "unsupported_resolved_post_carrion_context_count": unsupported_post_count,
        "unsupported_resolved_post_carrion_context_share": _share(
            unsupported_post_count,
            unsupported_count,
        ),
        "action_source_counts": _counter_payload(acc.get("action_source_counts")),
        "requested_action_counts": _ordered_action_counts(
            _as_counter(acc.get("requested_action_counts"))
        ),
        "resolved_action_counts": _ordered_action_counts(
            _as_counter(acc.get("resolved_action_counts"))
        ),
        "unsupported_requested_action_counts": _ordered_action_counts(
            _as_counter(acc.get("unsupported_requested_action_counts"))
        ),
        "unsupported_resolved_action_counts": _ordered_action_counts(
            _as_counter(acc.get("unsupported_resolved_action_counts"))
        ),
        "unsupported_invalid_reason_counts": _counter_payload(
            acc.get("unsupported_invalid_reason_counts")
        ),
        "post_carrion_unsupported_requested_action_counts": _ordered_action_counts(
            _as_counter(acc.get("post_carrion_unsupported_requested_action_counts"))
        ),
        "post_carrion_requested_action_counts": _ordered_action_counts(
            _as_counter(acc.get("post_carrion_requested_action_counts"))
        ),
        "post_carrion_resolved_action_counts": _ordered_action_counts(
            _as_counter(acc.get("post_carrion_resolved_action_counts"))
        ),
        "selected_score_delta": _stats(acc.get("all_selected_score_deltas")),
        "post_carrion_selected_score_delta": _stats(
            acc.get("post_carrion_selected_score_deltas")
        ),
        "examples": list(_list(acc.get("examples"))),
    }


def _post_carrion_example(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": row.get("path"),
        "seed": row.get("seed"),
        "tick": row.get("tick"),
        "agent_id": row.get("agent_id"),
        "record_index": row.get("record_index"),
        "requested_action": row.get("requested_action"),
        "resolved_action": row.get("resolved_action"),
        "rollout_context_non_empty": row.get("rollout_context_non_empty"),
        "rollout_context_selected_score_delta": row.get("score_delta"),
        "previous_context": row.get("previous_context"),
    }


def _branch_join_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": row.get("path"),
        "source_kind": row.get("source_kind"),
        "seed": row.get("seed"),
        "tick": row.get("tick"),
        "agent_id": row.get("agent_id"),
        "record_index": row.get("record_index"),
        "requested_action": row.get("requested_action"),
        "resolved_action": row.get("resolved_action"),
        "rollout_context_non_empty": row.get("rollout_context_non_empty"),
        "rollout_context_post_carrion_context": row.get(
            "rollout_context_post_carrion_context"
        ),
        "rollout_context_selected_score_delta": row.get("score_delta"),
    }


def _fixture_failure_audit(
    *,
    rollout_context_report: Mapping[str, object] | None,
    baseline_report: Mapping[str, object] | None,
    branch_oracle_audit: Mapping[str, object] | None,
    coverage: _CoverageResult,
) -> dict[str, object]:
    comparison = _fixture_gate_comparison(rollout_context_report, baseline_report)
    branch_overlap = _branch_oracle_overlap(branch_oracle_audit, coverage)
    return {
        "fixture_gate_comparison": comparison,
        "branch_oracle_overlap": branch_overlap,
    }


def _fixture_gate_comparison(
    candidate_report: Mapping[str, object] | None,
    baseline_report: Mapping[str, object] | None,
) -> dict[str, object]:
    candidate_gate = _mapping(_mapping(candidate_report or {}).get("fixture_gate"))
    baseline_gate = _mapping(_mapping(baseline_report or {}).get("fixture_gate"))
    candidate_blockers = _blocker_index(_list(candidate_gate.get("blockers")))
    baseline_blockers = _blocker_index(_list(baseline_gate.get("blockers")))
    added_blockers = [
        dict(candidate_blockers[key])
        for key in sorted(candidate_blockers.keys() - baseline_blockers.keys())
    ]
    shared_blockers = [
        {
            "key": list(key),
            "candidate": dict(candidate_blockers[key]),
            "baseline": dict(baseline_blockers[key]),
            "value_delta": _delta(
                candidate_blockers[key].get("value"),
                baseline_blockers[key].get("value"),
            ),
        }
        for key in sorted(candidate_blockers.keys() & baseline_blockers.keys())
    ]
    return {
        "candidate_passed": candidate_gate.get("passed"),
        "baseline_passed": baseline_gate.get("passed"),
        "candidate_blocker_count": len(candidate_blockers),
        "baseline_blocker_count": len(baseline_blockers),
        "added_candidate_blockers": added_blockers,
        "shared_blockers": shared_blockers,
        "carrion_only_120": _fixture_horizon_comparison(
            candidate_report,
            baseline_report,
            fixture_name="carrion_only",
            horizon="120",
        ),
    }


def _fixture_horizon_comparison(
    candidate_report: Mapping[str, object] | None,
    baseline_report: Mapping[str, object] | None,
    *,
    fixture_name: str,
    horizon: str,
) -> dict[str, object]:
    candidate = _fixture_horizon_payload(
        candidate_report,
        fixture_name=fixture_name,
        horizon=horizon,
    )
    baseline = _fixture_horizon_payload(
        baseline_report,
        fixture_name=fixture_name,
        horizon=horizon,
    )
    candidate_metrics = _mapping(candidate.get("metrics"))
    baseline_metrics = _mapping(baseline.get("metrics"))
    metric_deltas = {
        key: _delta(candidate_metrics.get(key), baseline_metrics.get(key))
        for key in sorted(set(candidate_metrics.keys()) | set(baseline_metrics.keys()))
    }
    return {
        "fixture": fixture_name,
        "horizon": horizon,
        "candidate": candidate,
        "baseline": baseline,
        "metric_deltas": metric_deltas,
    }


def _fixture_horizon_payload(
    report: Mapping[str, object] | None,
    *,
    fixture_name: str,
    horizon: str,
) -> dict[str, object]:
    gate = _mapping(_mapping(report or {}).get("fixture_gate"))
    per_horizon = _mapping(gate.get("per_horizon"))
    horizon_payload = _mapping(per_horizon.get(str(horizon)))
    per_fixture = _mapping(horizon_payload.get("per_fixture"))
    payload = _mapping(per_fixture.get(fixture_name))
    if not payload:
        payload = _mapping(_mapping(gate.get("per_fixture")).get(fixture_name))
    return {
        "passed": payload.get("passed"),
        "metrics": dict(_mapping(payload.get("metrics"))),
        "blockers": _list(payload.get("blockers")),
    }


def _branch_oracle_overlap(
    branch_oracle_audit: Mapping[str, object] | None,
    coverage: _CoverageResult,
) -> dict[str, object]:
    if branch_oracle_audit is None:
        return {
            "loaded": False,
            "branch_result_count": 0,
            "exact_action_matched_branch_result_count": 0,
            "state_matched_branch_result_count": 0,
            "diagnostics_only": True,
        }
    branch_results = [
        result
        for result in _list(branch_oracle_audit.get("branch_results"))
        if isinstance(result, Mapping)
    ]
    exact_matches = 0
    state_matches = 0
    exact_matched_rows = 0
    exact_post_context = 0
    exact_non_empty = 0
    exact_oracle_changed = 0
    exact_oracle_changed_rows = 0
    exact_oracle_changed_post_context = 0
    changed_score_deltas: list[float] = []
    matched_logged_actions: Counter[str] = Counter()
    matched_oracle_best_actions: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    for result in branch_results:
        seed = _int(result.get("seed"))
        branch_tick = _int(result.get("branch_tick"))
        agent_id = _int(result.get("agent_id"))
        logged_action = str(result.get("logged_action", ""))
        oracle_best_action = str(result.get("oracle_best_action", ""))
        exact_key = (seed, branch_tick, agent_id, logged_action)
        state_key = (seed, branch_tick, agent_id)
        exact_rows = tuple(coverage.exact_action_index.get(exact_key, ()))
        state_rows = tuple(coverage.state_index.get(state_key, ()))
        if exact_rows:
            exact_matches += 1
            exact_matched_rows += len(exact_rows)
            matched_logged_actions.update([logged_action])
            if oracle_best_action:
                matched_oracle_best_actions.update([oracle_best_action])
            exact_post_context += sum(
                1
                for row in exact_rows
                if bool(row.get("rollout_context_post_carrion_context"))
            )
            exact_non_empty += sum(
                1 for row in exact_rows if bool(row.get("rollout_context_non_empty"))
            )
            if oracle_best_action and oracle_best_action != logged_action:
                exact_oracle_changed += 1
                exact_oracle_changed_rows += len(exact_rows)
                exact_oracle_changed_post_context += sum(
                    1
                    for row in exact_rows
                    if bool(row.get("rollout_context_post_carrion_context"))
                )
                for row in exact_rows:
                    score_delta = row.get("rollout_context_selected_score_delta")
                    if isinstance(score_delta, (int, float)) and not isinstance(
                        score_delta,
                        bool,
                    ):
                        changed_score_deltas.append(float(score_delta))
        if state_rows:
            state_matches += 1
        if len(examples) < 12 and (exact_rows or state_rows):
            examples.append(
                {
                    "branch_id": result.get("branch_id"),
                    "seed": seed,
                    "branch_tick": branch_tick,
                    "agent_id": agent_id,
                    "logged_action": logged_action,
                    "oracle_best_action": result.get("oracle_best_action"),
                    "exact_action_match_count": len(exact_rows),
                    "state_match_count": len(state_rows),
                    "exact_action_matches": list(exact_rows[:4]),
                    "state_matches": list(state_rows[:4]),
                }
            )
    return {
        "loaded": True,
        "schema_version": branch_oracle_audit.get("schema_version"),
        "branch_result_count": len(branch_results),
        "exact_action_matched_branch_result_count": exact_matches,
        "state_matched_branch_result_count": state_matches,
        "exact_action_matched_share": _share(exact_matches, len(branch_results)),
        "state_matched_share": _share(state_matches, len(branch_results)),
        "exact_action_matched_row_count": exact_matched_rows,
        "exact_action_matched_post_carrion_context_count": exact_post_context,
        "exact_action_matched_post_carrion_context_share": _share(
            exact_post_context,
            exact_matched_rows,
        ),
        "exact_action_matched_non_empty_count": exact_non_empty,
        "exact_action_matched_non_empty_share": _share(
            exact_non_empty,
            exact_matched_rows,
        ),
        "exact_action_matched_oracle_changed_count": exact_oracle_changed,
        "exact_action_matched_oracle_changed_row_count": exact_oracle_changed_rows,
        "exact_action_matched_oracle_changed_post_carrion_context_count": (
            exact_oracle_changed_post_context
        ),
        "exact_action_matched_oracle_changed_post_carrion_context_share": _share(
            exact_oracle_changed_post_context,
            exact_oracle_changed_rows,
        ),
        "exact_action_matched_logged_action_counts": _counter_payload(
            matched_logged_actions
        ),
        "exact_action_matched_oracle_best_action_counts": _counter_payload(
            matched_oracle_best_actions
        ),
        "exact_action_matched_oracle_changed_selected_score_delta": _stats(
            changed_score_deltas
        ),
        "examples": examples,
        "diagnostics_only": True,
    }


def _config_section(
    *,
    min_broad_post_carrion_context_share: float,
    min_branch_oracle_exact_match_share: float,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "missing_evidence_policy": "classify_inconclusive_without_runtime_workaround_v1",
        "rollout_context_report_option": "--rollout-context-report",
        "v105_report_compatibility_alias": "--v105-report",
        "trajectory_reader": "local_lenient_jsonl_preserve_diagnostics_v1",
        "min_broad_post_carrion_context_share": (
            min_broad_post_carrion_context_share
        ),
        "min_trajectory_post_carrion_context_share": (
            DEFAULT_MIN_TRAJECTORY_POST_CARRION_CONTEXT_SHARE
        ),
        "min_branch_oracle_exact_match_share": (
            min_branch_oracle_exact_match_share
        ),
        "min_movement_unsupported_resolution_share": (
            DEFAULT_MIN_MOVEMENT_UNSUPPORTED_RESOLUTION_SHARE
        ),
        "min_context_unsupported_resolution_lift": (
            DEFAULT_MIN_CONTEXT_UNSUPPORTED_RESOLUTION_LIFT
        ),
    }


def _coverage_section(
    *,
    source_report_summary: Mapping[str, object],
    fixture_audit: Mapping[str, object],
    coverage: Mapping[str, object],
    min_broad_post_carrion_context_share: float,
) -> dict[str, object]:
    rollout_summary = _mapping(source_report_summary.get("rollout_context_report"))
    broad = _mapping(rollout_summary.get("holdout_aggregate"))
    broad_count = _int(broad.get("rollout_context_post_carrion_context_count"))
    broad_share = _number_or_none(
        broad.get("rollout_context_post_carrion_context_share")
    )
    broad_decision_count = _int(broad.get("rollout_context_decision_count"))
    aggregate = _mapping(coverage.get("aggregate"))
    trajectory_count = _int(
        aggregate.get("rollout_context_post_carrion_context_count")
    )
    trajectory_share = _number_or_none(
        aggregate.get("rollout_context_post_carrion_context_share")
    )
    trajectory_decision_count = _int(aggregate.get("decision_row_count"))
    fixture_failed = _fixture_failed(fixture_audit)
    broad_answer = _post_carrion_context_answer(
        count=broad_count,
        share=broad_share,
        decision_count=broad_decision_count,
        floor=min_broad_post_carrion_context_share,
        fixture_failed=False,
    )
    trajectory_answer = _post_carrion_context_answer(
        count=trajectory_count,
        share=trajectory_share,
        decision_count=trajectory_decision_count,
        floor=DEFAULT_MIN_TRAJECTORY_POST_CARRION_CONTEXT_SHARE,
        fixture_failed=fixture_failed,
    )
    answer = broad_answer
    if trajectory_answer == "post_carrion_context_adequate_but_unhelpful":
        answer = trajectory_answer
    elif broad_answer == "post_carrion_context_absent":
        answer = broad_answer
    elif broad_answer is None:
        answer = trajectory_answer
    elif broad_answer == "post_carrion_context_too_rare":
        answer = broad_answer
    return {
        "question": (
            "whether post-carrion context is absent, too rare, or "
            "adequate-but-unhelpful"
        ),
        "answer": answer,
        "allowed_answers": [
            "post_carrion_context_absent",
            "post_carrion_context_too_rare",
            "post_carrion_context_adequate_but_unhelpful",
        ],
        "broad_holdout": {
            "answer": broad_answer,
            "rollout_context_decision_count": broad_decision_count,
            "post_carrion_context_count": broad_count,
            "post_carrion_context_share": broad_share,
            "share_floor": min_broad_post_carrion_context_share,
        },
        "fixture_trajectory": {
            "answer": trajectory_answer,
            "decision_row_count": trajectory_decision_count,
            "post_carrion_context_count": trajectory_count,
            "post_carrion_context_share": trajectory_share,
            "share_floor": DEFAULT_MIN_TRAJECTORY_POST_CARRION_CONTEXT_SHARE,
            "fixture_failed": fixture_failed,
        },
    }


def _temporal_alignment_section(
    source_report_summary: Mapping[str, object],
) -> dict[str, object]:
    summary = _mapping(source_report_summary.get("v69_recovery_action_target_audit"))
    labels = [str(label) for label in _list(summary.get("classification_labels"))]
    primary = str(summary.get("classification_primary", ""))
    matched_count = _int(summary.get("oracle_matched_row_count"))
    if not bool(summary.get("loaded")):
        answer = "transition_alignment_inconclusive"
    elif primary == "first_record_sampling_gap" or "first_record_sampling_gap" in labels:
        answer = "first_recovery_decision_sampling_gap"
    elif matched_count > 0:
        answer = "transition_aligned_recovery_context_present"
    else:
        answer = "first_recovery_decision_sampling_gap"
    return {
        "question": (
            "whether first recovery decision alignment is present or has a "
            "sampling gap"
        ),
        "answer": answer,
        "allowed_answers": [
            "transition_aligned_recovery_context_present",
            "first_recovery_decision_sampling_gap",
            "transition_alignment_inconclusive",
        ],
        "source_classification_primary": primary or None,
        "source_classification_labels": labels,
        "decision_row_count": summary.get("decision_row_count"),
        "oracle_loaded": summary.get("oracle_loaded"),
        "oracle_matched_row_count": summary.get("oracle_matched_row_count"),
        "oracle_unmatched_row_count": summary.get("oracle_unmatched_row_count"),
    }


def _fixture_action_support_section(
    *,
    source_report_summary: Mapping[str, object],
    fixture_audit: Mapping[str, object],
    min_branch_oracle_exact_match_share: float,
) -> dict[str, object]:
    branch_summary = _mapping(source_report_summary.get("branch_oracle_audit"))
    overlap = _mapping(fixture_audit.get("branch_oracle_overlap"))
    exact_share = _number_or_none(overlap.get("exact_action_matched_share"))
    if not bool(overlap.get("loaded")):
        answer = None
    elif exact_share is None or exact_share < min_branch_oracle_exact_match_share:
        answer = "heldout_fixture_recovery_support_gap"
    else:
        answer = "heldout_fixture_recovery_support_adequate"
    return {
        "question": (
            "whether held-out fixture recovery action support is sparse or adequate"
        ),
        "answer": answer,
        "allowed_answers": [
            "heldout_fixture_recovery_support_gap",
            "heldout_fixture_recovery_support_adequate",
        ],
        "branch_oracle_material_support": {
            "loaded": branch_summary.get("loaded"),
            "diagnostic_acceptance_passed": branch_summary.get(
                "diagnostic_acceptance_passed"
            ),
            "materially_supports_branch_action_oracle": branch_summary.get(
                "materially_supports_branch_action_oracle"
            ),
            "material_oracle_gain_count": branch_summary.get(
                "material_oracle_gain_count"
            ),
            "oracle_changed_action_count": branch_summary.get(
                "oracle_changed_action_count"
            ),
        },
        "heldout_overlap": {
            "branch_result_count": overlap.get("branch_result_count"),
            "exact_action_matched_branch_result_count": overlap.get(
                "exact_action_matched_branch_result_count"
            ),
            "exact_action_matched_share": exact_share,
            "exact_action_share_floor": min_branch_oracle_exact_match_share,
            "state_matched_branch_result_count": overlap.get(
                "state_matched_branch_result_count"
            ),
            "state_matched_share": overlap.get("state_matched_share"),
        },
    }


def _policy_scoring_pressure_section(
    *,
    fixture_audit: Mapping[str, object],
    coverage: Mapping[str, object],
) -> dict[str, object]:
    overlap = _mapping(fixture_audit.get("branch_oracle_overlap"))
    aggregate = _mapping(coverage.get("aggregate"))
    changed_count = _int(overlap.get("exact_action_matched_oracle_changed_count"))
    exact_count = _int(overlap.get("exact_action_matched_branch_result_count"))
    if not bool(overlap.get("loaded")):
        answer = "policy_scoring_inconclusive_missing_score_surface"
        suppresses = None
    elif changed_count > 0:
        answer = "policy_scoring_suppresses_supported_recovery_action"
        suppresses = True
    elif exact_count > 0:
        answer = "policy_scoring_inconclusive_missing_score_surface"
        suppresses = False
    else:
        answer = "policy_scoring_inconclusive_missing_score_surface"
        suppresses = None
    return {
        "question": "whether policy scoring suppresses supported recovery actions",
        "answer": answer,
        "suppresses_supported_recovery_actions": suppresses,
        "allowed_answers": [
            "policy_scoring_suppresses_supported_recovery_action",
            "policy_scoring_inconclusive_missing_score_surface",
        ],
        "selected_score_delta": aggregate.get("selected_score_delta"),
        "post_carrion_selected_score_delta": aggregate.get(
            "post_carrion_selected_score_delta"
        ),
        "branch_oracle_overlap": {
            "exact_action_matched_branch_result_count": exact_count,
            "exact_action_matched_oracle_changed_count": changed_count,
            "exact_action_matched_oracle_changed_post_carrion_context_count": (
                overlap.get(
                    "exact_action_matched_oracle_changed_post_carrion_context_count"
                )
            ),
            "exact_action_matched_logged_action_counts": overlap.get(
                "exact_action_matched_logged_action_counts"
            ),
            "exact_action_matched_oracle_best_action_counts": overlap.get(
                "exact_action_matched_oracle_best_action_counts"
            ),
            "exact_action_matched_oracle_changed_selected_score_delta": overlap.get(
                "exact_action_matched_oracle_changed_selected_score_delta"
            ),
        },
    }


def _unsupported_resolution_section(
    *,
    source_report_summary: Mapping[str, object],
    coverage: Mapping[str, object],
) -> dict[str, object]:
    rollout_summary = _mapping(source_report_summary.get("rollout_context_report"))
    holdout = _mapping(rollout_summary.get("holdout_aggregate"))
    total = _int(holdout.get("unsupported_resolved_action_count"))
    breakdown = _mapping(holdout.get("unsupported_resolved_action_breakdown"))
    by_requested = _mapping(breakdown.get("by_requested_action"))
    by_invalid_reason = _mapping(breakdown.get("by_invalid_reason"))
    movement_count = sum(
        _int(count)
        for action, count in by_requested.items()
        if str(action).startswith("move_")
    )
    movement_share = _share(movement_count, total)
    mask_reason_count = _int(by_invalid_reason.get("not_in_resolution_action_mask"))
    aggregate = _mapping(coverage.get("aggregate"))
    trajectory_unsupported_count = _int(
        aggregate.get("unsupported_resolved_action_count")
    )
    trajectory_unsupported_post_share = _number_or_none(
        aggregate.get("unsupported_resolved_post_carrion_context_share")
    )
    trajectory_post_share = _number_or_none(
        aggregate.get("rollout_context_post_carrion_context_share")
    )
    context_lift = (
        _round(trajectory_unsupported_post_share - trajectory_post_share)
        if trajectory_unsupported_post_share is not None
        and trajectory_post_share is not None
        else None
    )
    context_correlated = (
        context_lift is not None
        and context_lift >= DEFAULT_MIN_CONTEXT_UNSUPPORTED_RESOLUTION_LIFT
    )
    movement_mask_drift = (
        movement_share is not None
        and movement_share >= DEFAULT_MIN_MOVEMENT_UNSUPPORTED_RESOLUTION_SHARE
        and (mask_reason_count == total or total == 0)
    )
    if total <= 0 and trajectory_unsupported_count <= 0:
        answer = "unsupported_resolution_not_context_correlated"
    elif context_correlated:
        answer = "unsupported_resolution_context_correlated"
    elif movement_mask_drift:
        answer = "unsupported_resolution_movement_mask_drift"
    else:
        answer = "unsupported_resolution_not_context_correlated"
    return {
        "question": (
            "whether unsupported resolution is movement-mask drift or "
            "context-correlated"
        ),
        "answer": answer,
        "movement_mask_drift": movement_mask_drift,
        "context_correlated": context_correlated,
        "allowed_answers": [
            "unsupported_resolution_movement_mask_drift",
            "unsupported_resolution_context_correlated",
            "unsupported_resolution_not_context_correlated",
        ],
        "broad_holdout": {
            "unsupported_resolved_action_count": total,
            "movement_requested_count": movement_count,
            "movement_requested_share": movement_share,
            "not_in_resolution_action_mask_count": mask_reason_count,
            "by_requested_action": dict(by_requested),
            "by_invalid_reason": dict(by_invalid_reason),
        },
        "trajectory_context": {
            "unsupported_resolved_action_count": trajectory_unsupported_count,
            "unsupported_resolved_post_carrion_context_count": aggregate.get(
                "unsupported_resolved_post_carrion_context_count"
            ),
            "unsupported_resolved_post_carrion_context_share": (
                trajectory_unsupported_post_share
            ),
            "post_carrion_context_share": trajectory_post_share,
            "post_carrion_unsupported_lift": context_lift,
            "post_carrion_unsupported_lift_floor": (
                DEFAULT_MIN_CONTEXT_UNSUPPORTED_RESOLUTION_LIFT
            ),
            "unsupported_requested_action_counts": aggregate.get(
                "unsupported_requested_action_counts"
            ),
            "unsupported_invalid_reason_counts": aggregate.get(
                "unsupported_invalid_reason_counts"
            ),
        },
    }


def _seed29_birth_regression_section(
    source_report_summary: Mapping[str, object],
) -> dict[str, object]:
    deltas = _mapping(source_report_summary.get("broad_holdout_deltas_vs_baseline"))
    seed29 = _seed_delta(deltas, 29)
    births_delta = _number_or_none(seed29.get("births_delta"))
    unsupported_delta = _number_or_none(
        seed29.get("unsupported_resolved_action_delta")
    )
    movement_delta = _number_or_none(seed29.get("movement_event_rate_delta"))
    post_carrion_count = _int(
        seed29.get("rollout_context_post_carrion_context_count")
    )
    if not seed29:
        answer = "seed29_birth_regression_inconclusive_missing_baseline_trajectory"
    elif births_delta is None or births_delta >= 0.0:
        answer = "seed29_birth_regression_inconclusive_missing_baseline_trajectory"
    elif post_carrion_count > 0 and (unsupported_delta is None or unsupported_delta <= 0):
        answer = "seed29_birth_regression_recovery_action_choice"
    elif (unsupported_delta is not None and unsupported_delta > 0.0) or (
        movement_delta is not None and movement_delta < 0.0
    ):
        answer = "seed29_birth_regression_movement_failure"
    elif _number_or_none(seed29.get("alive_agents_delta")) is not None:
        answer = "seed29_birth_regression_reproduction_timing_side_effect"
    else:
        answer = "seed29_birth_regression_inconclusive_missing_baseline_trajectory"
    return {
        "question": (
            "whether seed 29 birth regression ties to recovery action choice, "
            "movement failure, reproduction timing, or is inconclusive"
        ),
        "answer": answer,
        "allowed_answers": [
            "seed29_birth_regression_recovery_action_choice",
            "seed29_birth_regression_movement_failure",
            "seed29_birth_regression_reproduction_timing_side_effect",
            "seed29_birth_regression_inconclusive_missing_baseline_trajectory",
        ],
        "seed": 29,
        "births_delta": births_delta,
        "alive_agents_delta": seed29.get("alive_agents_delta"),
        "deaths_delta": seed29.get("deaths_delta"),
        "movement_event_rate_delta": movement_delta,
        "unsupported_resolved_action_delta": unsupported_delta,
        "rollout_context_post_carrion_context_count": post_carrion_count,
    }


def _research_recommendation_section(
    *,
    planned_sections: Mapping[str, Mapping[str, object]],
    classification: Mapping[str, object],
) -> dict[str, object]:
    answers = {
        name: section.get("answer")
        for name, section in sorted(planned_sections.items())
    }
    recommendation = (
        "do_not_promote_v105; keep rollout-context evidence diagnostics-only "
        "and target first-recovery sampling plus branch-oracle action support "
        "before any runtime policy change"
    )
    return {
        "classification_primary": classification.get("primary"),
        "diagnostic_answers": answers,
        "recommendation": recommendation,
        "promote_v105": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_relaxation_recommended": False,
        "v107_planning_run": False,
    }


def _classification(
    *,
    evidence: Mapping[str, object],
    planned_sections: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    scores = {
        "post_carrion_context_absent": 0,
        "post_carrion_context_too_rare": 0,
        "post_carrion_context_adequate_but_unhelpful": 0,
        "transition_aligned_recovery_context_present": 0,
        "first_recovery_decision_sampling_gap": 0,
        "transition_alignment_inconclusive": 0,
        "heldout_fixture_recovery_support_gap": 0,
        "heldout_fixture_recovery_support_adequate": 0,
        "policy_scoring_suppresses_supported_recovery_action": 0,
        "policy_scoring_inconclusive_missing_score_surface": 0,
        "unsupported_resolution_movement_mask_drift": 0,
        "unsupported_resolution_context_correlated": 0,
        "unsupported_resolution_not_context_correlated": 0,
        "seed29_birth_regression_recovery_action_choice": 0,
        "seed29_birth_regression_movement_failure": 0,
        "seed29_birth_regression_reproduction_timing_side_effect": 0,
        "seed29_birth_regression_inconclusive_missing_baseline_trajectory": 0,
    }
    weights = {
        "coverage": 30,
        "temporal_alignment": 20,
        "fixture_action_support": 18,
        "policy_scoring_pressure": 16,
        "unsupported_resolution": 8,
        "seed29_birth_regression": 6,
    }
    missing = _missing_evidence(evidence)
    answers = {}
    for name, section in sorted(planned_sections.items()):
        answer = section.get("answer")
        if isinstance(answer, str):
            answers[name] = answer
    facts = list(answers.values())
    if missing:
        return {
            "primary": None,
            "labels": [],
            "category_scores": scores,
            "evidence": facts,
            "diagnostic_answers": answers,
            "missing_evidence": missing,
            "diagnostics_only": True,
        }
    for section_name, answer in answers.items():
        if answer in scores:
            scores[answer] += weights.get(section_name, 1)
    positive = [label for label, score in scores.items() if score > 0]
    if not positive:
        positive = ["no_material_blocker_detected"]
        primary = "no_material_blocker_detected"
    else:
        primary = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return {
        "primary": primary,
        "labels": positive,
        "category_scores": scores,
        "evidence": facts,
        "diagnostic_answers": answers,
        "missing_evidence": missing,
        "diagnostics_only": True,
    }


def _fixture_failed(fixture_audit: Mapping[str, object]) -> bool:
    gate_comparison = _mapping(fixture_audit.get("fixture_gate_comparison"))
    carrion_120 = _mapping(gate_comparison.get("carrion_only_120"))
    candidate = _mapping(carrion_120.get("candidate"))
    return gate_comparison.get("candidate_passed") is False or (
        candidate.get("passed") is False
    )


def _post_carrion_context_answer(
    *,
    count: int,
    share: float | None,
    decision_count: int,
    floor: float,
    fixture_failed: bool,
) -> str | None:
    if decision_count <= 0 and count <= 0:
        return None
    if count <= 0:
        return "post_carrion_context_absent"
    if share is not None and share < floor:
        return "post_carrion_context_too_rare"
    if fixture_failed:
        return "post_carrion_context_adequate_but_unhelpful"
    return "post_carrion_context_adequate_but_unhelpful"


def _seed_delta(
    deltas: Mapping[str, object],
    seed: int,
) -> Mapping[str, object]:
    for item in _list(deltas.get("per_seed")):
        if isinstance(item, Mapping) and _int(item.get("seed"), default=-1) == seed:
            return item
    return {}


def _missing_evidence(evidence: Mapping[str, object]) -> list[str]:
    missing: list[str] = []
    reports = _mapping(evidence.get("reports"))
    for name, payload in sorted(reports.items()):
        if not bool(_mapping(payload).get("loaded")):
            missing.append(str(name))
    trajectories = _mapping(evidence.get("trajectories"))
    if _int(trajectories.get("loaded_path_count")) <= 0:
        missing.append("trajectories")
    if _int(trajectories.get("load_failure_count")) > 0:
        missing.append("trajectory_load_failures")
    return missing


def _blocker_index(
    blockers: Sequence[object],
) -> dict[tuple[str, str, str, str], Mapping[str, object]]:
    indexed: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
    for blocker in blockers:
        if not isinstance(blocker, Mapping):
            continue
        key = (
            str(blocker.get("fixture")),
            str(blocker.get("ticks", "")),
            str(blocker.get("metric")),
            str(blocker.get("reason")),
        )
        indexed[key] = blocker
    return indexed


def _dataset_footer_summary(dataset: TrajectoryJsonlDataset) -> dict[str, object]:
    summary = _mapping(dataset.footer.get("summary"))
    trajectory_summary = _mapping(dataset.footer.get("trajectory_summary"))
    return {
        "terminal": {
            "alive_agents": summary.get("alive_agents"),
            "births": summary.get("births"),
            "deaths": summary.get("deaths"),
            "ticks_executed": summary.get("ticks_executed"),
            "record_count": trajectory_summary.get("record_count"),
            "invalid_resolution_action_count": trajectory_summary.get(
                "invalid_resolution_action_count"
            ),
        }
    }


def _terminal_summary(
    datasets: Sequence[TrajectoryJsonlDataset],
) -> dict[str, object]:
    alive_values: list[float] = []
    birth_values: list[float] = []
    death_values: list[float] = []
    extinct_count = 0
    for dataset in datasets:
        summary = _mapping(dataset.footer.get("summary"))
        alive = _number_or_none(summary.get("alive_agents"))
        births = _number_or_none(summary.get("births"))
        deaths = _number_or_none(summary.get("deaths"))
        if alive is not None:
            alive_values.append(alive)
            if alive <= 0.0:
                extinct_count += 1
        if births is not None:
            birth_values.append(births)
        if deaths is not None:
            death_values.append(deaths)
    return {
        "trajectory_count": len(datasets),
        "extinct_trajectory_count": extinct_count,
        "alive_agents": _stats(alive_values),
        "births": _stats(birth_values),
        "deaths": _stats(death_values),
    }


def _dataset_seed(dataset: TrajectoryJsonlDataset) -> int:
    header_config = _mapping(dataset.header.get("config"))
    footer_summary = _mapping(dataset.footer.get("summary"))
    for value in (header_config.get("seed"), footer_summary.get("seed")):
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    match = _SEED_PATTERN.search(str(dataset.path))
    if match is not None:
        return int(match.group(1))
    return -1


def _source_kind(path: str | Path) -> str:
    name = Path(path).name
    if name.startswith("fixture-carrion-only-mind-v3"):
        return "fixture_carrion_only_mind_v3"
    if name.startswith("open-mind-v3"):
        return "open_mind_v3"
    if "mind-v3" in name:
        return "mind_v3"
    return "unknown"


def _context_non_empty(context: Mapping[str, object]) -> bool:
    for key in (
        "recent_requested_actions",
        "recent_resolved_actions",
        "recent_moved_flags",
        "recent_drank_flags",
        "recent_ate_flags",
        "recent_resource_gain",
    ):
        value = context.get(key)
        if isinstance(value, Sequence) and len(value) > 0:
            return True
    return False


def _is_heuristic_action_source(source: str) -> bool:
    return "heuristic" in source.lower()


def _ordered_action_counts(counter: Counter[str]) -> dict[str, int]:
    payload = {action: int(counter.get(action, 0)) for action in ACTION_NAMES}
    for key, value in sorted(counter.items()):
        if key not in payload:
            payload[key] = int(value)
    return payload


def _counter_payload(value: object) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(_as_counter(value).items())}


def _as_counter(value: object) -> Counter[str]:
    if isinstance(value, Counter):
        return value
    return Counter()


def _stats(values: object) -> dict[str, object]:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ] if isinstance(values, Sequence) else []
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "abs_mean": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numbers),
        "mean": _round(sum(numbers) / float(len(numbers))),
        "abs_mean": _round(sum(abs(value) for value in numbers) / float(len(numbers))),
        "min": _round(min(numbers)),
        "max": _round(max(numbers)),
    }


def _delta(candidate: object, baseline: object) -> float | None:
    candidate_number = _number_or_none(candidate)
    baseline_number = _number_or_none(baseline)
    if candidate_number is None or baseline_number is None:
        return None
    return _round(candidate_number - baseline_number)


def _share(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _round(numerator / float(denominator))


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    return None


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return int(value)


def _round(value: float) -> float:
    return round(float(value), 6)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
