from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from evolution_sim.cli.mind_v3_evaluate import (
    _dominant_action_summary,
    _fixture_world,
    _heuristic_action_source_count,
    _json_ready,
)
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES, MOVEMENT_ACTIONS
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.mind.branch_action_oracle_audit import (
    DEFAULT_TARGET_HORIZON_TRACE_TICKS,
)
from evolution_sim.mind.carrion_branch_explore import (
    _branch_state_digest,
    _configure_manual_summary_run,
)
from evolution_sim.mind.evolution import load_mind_v3_founder_template
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.transition_aligned_recovery_audit import (
    MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_SCHEMA_VERSION,
    reconstruct_transition_aligned_first_recovery_rows,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_branch_oracle_audit_v1"
)
MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_POLICY = (
    "diagnostics_only_first_recovery_branch_oracle_support_v1"
)

ALLOWED_CLASSIFICATION_LABELS: tuple[str, ...] = (
    "first_recovery_branch_replay_verified",
    "first_recovery_branch_replay_partial",
    "first_recovery_branch_replay_failed",
    "first_recovery_oracle_improves_logged_action",
    "first_recovery_oracle_matches_logged_action",
    "first_recovery_oracle_no_material_gain",
    "first_recovery_oracle_action_distribution_clean",
    "first_recovery_oracle_action_distribution_collapsed",
    "oracle_supported_actions_observation_legal",
    "oracle_supported_actions_not_observation_legal",
    "oracle_supported_actions_resolution_legal",
    "oracle_supported_actions_resolution_invalid",
    "resolution_invalid_occupancy_race",
    "resolution_invalid_blocked_route",
    "resolution_invalid_depleted_resource",
    "resolution_invalid_other_public_path",
    "resolution_invalid_inconclusive",
    "seed29_public_attribution_oracle_movement_failure",
    "seed29_public_attribution_unsupported_resolution",
    "seed29_public_attribution_missed_drink",
    "seed29_public_attribution_delayed_reproduction_readiness",
    "seed29_public_attribution_inconclusive",
    "heldout_branch_support_sufficient_for_v109",
    "heldout_branch_support_insufficient_for_v109",
    "missing_evidence_inconclusive",
)
CLASSIFICATION_LABEL_SET = frozenset(ALLOWED_CLASSIFICATION_LABELS)

FIXTURE_SOURCE_KIND = "fixture_carrion_only"
OPEN_SOURCE_KIND = "open_mind_v3"
UNKNOWN_SOURCE_KIND = "unknown"
DEFAULT_MAX_TARGETS_PER_SEED_SOURCE = 1
DEFAULT_DOMINANT_ORACLE_ACTION_SHARE_MAX = 0.50
DEFAULT_MIN_MATERIAL_GAIN_SEED_COUNT = 3
_SEED_PATTERN = re.compile(r"(?:seed[-_=]?|mind-v3-)(\d+)")


class FirstRecoveryBranchOracleAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _LoadedReport:
    name: str
    path: str | None
    payload: Mapping[str, object] | None
    evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class _BranchTarget:
    branch_id: str
    source_path: str
    source_kind: str
    seed: int
    ticks: int
    branch_tick: int
    record_index: int
    agent_id: int
    logged_action: str
    gain_tick: int | None
    gain_record_index: int
    observation_digest: str | None
    observation_schema: str | None
    action_mask: dict[str, bool]
    resolution_action_mask: dict[str, bool]
    before: dict[str, object]
    branch_state_digest: str
    world: SimulationWorld


class _ForcedFirstActionThenDelegatePolicy:
    policy_id = "mind_v3_v108_first_recovery_branch_oracle_forced_first_action"
    policy_version = "mind_v3_v108_first_recovery_branch_oracle_forced_first_action_v1"

    def __init__(
        self,
        *,
        target_agent_id: int,
        forced_action: str,
        delegate: object,
    ) -> None:
        self.target_agent_id = int(target_agent_id)
        self.forced_action = str(forced_action)
        self.delegate = delegate
        self.used = False

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        metadata = observation.get("metadata")
        agent_id = metadata.get("agent_id") if isinstance(metadata, Mapping) else None
        if (
            not self.used
            and agent_id == self.target_agent_id
            and self.forced_action in ACTION_NAMES
            and bool(action_mask.get(self.forced_action, False))
        ):
            self.used = True
            return ActionDecision(
                requested_action=self.forced_action,
                source=f"branch_oracle_force:{self.forced_action}",
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                diagnostics={
                    "forced_first_action": self.forced_action,
                    "target_agent_id": self.target_agent_id,
                    "diagnostic_only": True,
                    "heuristic_free": True,
                },
            )
        decide = getattr(self.delegate, "decide", None)
        if not callable(decide):
            raise FirstRecoveryBranchOracleAuditError(
                "branch delegate policy does not implement decide"
            )
        return decide(observation, action_mask)

    def observe_transition(self, record: dict[str, object]) -> dict[str, object] | None:
        observe = getattr(self.delegate, "observe_transition", None)
        if not callable(observe):
            return None
        feedback = dict(record)
        if (
            _int(feedback.get("agent_id"), default=-1) == self.target_agent_id
            and feedback.get("action_source")
            == f"branch_oracle_force:{self.forced_action}"
        ):
            feedback["policy_id"] = getattr(self.delegate, "policy_id", None)
            feedback["policy_version"] = getattr(self.delegate, "policy_version", None)
        return observe(feedback)


def load_first_recovery_branch_oracle_json(path: str | Path) -> dict[str, object]:
    with _open_input(Path(path)) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise FirstRecoveryBranchOracleAuditError(
            f"report must be a JSON object: {path}"
        )
    return payload


def write_first_recovery_branch_oracle_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def build_first_recovery_branch_oracle_audit_report(
    *,
    v107_report: Mapping[str, object] | None = None,
    v107_report_path: str | Path | None = None,
    rollout_context_report: Mapping[str, object] | None = None,
    rollout_context_report_path: str | Path | None = None,
    baseline_report: Mapping[str, object] | None = None,
    baseline_report_path: str | Path | None = None,
    previous_branch_oracle_audit: Mapping[str, object] | None = None,
    previous_branch_oracle_audit_path: str | Path | None = None,
    trajectory_paths: Sequence[str | Path] = (),
    trajectory_glob_patterns: Sequence[str] = (),
    trajectory_datasets: Sequence[object] | None = None,
    max_targets_per_seed_source: int = DEFAULT_MAX_TARGETS_PER_SEED_SOURCE,
    include_open: bool = False,
    exhaustive: bool = False,
    verify_replay: bool = True,
) -> dict[str, object]:
    target_limit = _nonnegative_int(
        max_targets_per_seed_source,
        field="max_targets_per_seed_source",
    )
    contract = _contract(
        max_targets_per_seed_source=target_limit,
        include_open=include_open,
        exhaustive=exhaustive,
        verify_replay=verify_replay,
    )
    loaded_reports = {
        "v107_report": _resolve_json_report(
            "v107_report",
            v107_report,
            v107_report_path,
            expected_schema=MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_SCHEMA_VERSION,
        ),
        "rollout_context_report": _resolve_json_report(
            "rollout_context_report",
            rollout_context_report,
            rollout_context_report_path,
            expected_schema="mind_v3_evolution_search_v1",
        ),
        "baseline_report": _resolve_json_report(
            "baseline_report",
            baseline_report,
            baseline_report_path,
            expected_schema="mind_v3_evolution_search_v1",
        ),
        "previous_branch_oracle_audit": _resolve_json_report(
            "previous_branch_oracle_audit",
            previous_branch_oracle_audit,
            previous_branch_oracle_audit_path,
            expected_schema=None,
            optional=True,
        ),
    }
    reconstructed = reconstruct_transition_aligned_first_recovery_rows(
        trajectory_datasets=trajectory_datasets,
        trajectory_paths=trajectory_paths,
        trajectory_glob_patterns=trajectory_glob_patterns,
    )
    rows = [dict(row) for row in _list(reconstructed.get("rows"))]
    path_metadata = _mapping(reconstructed.get("path_metadata"))
    evidence = {
        "reports": {
            name: loaded.evidence for name, loaded in sorted(loaded_reports.items())
        },
        "trajectories": reconstructed.get("evidence", {}),
    }
    row_alignment = _v107_row_alignment(
        v107_report=loaded_reports["v107_report"].payload,
        rows=rows,
        path_metadata=path_metadata,
    )
    selection = _select_branch_targets(
        rows,
        path_metadata=path_metadata,
        max_targets_per_seed_source=target_limit,
        include_open=include_open,
        exhaustive=exhaustive,
    )
    selected_rows = [dict(row) for row in _list(selection.get("selected_rows"))]
    branch_points: list[_BranchTarget] = []
    materialization_failures: list[dict[str, object]] = []
    branch_results: list[dict[str, object]] = []
    template = None
    setup_error: dict[str, object] | None = None
    if selected_rows and not _alignment_mismatched(row_alignment):
        try:
            if rollout_context_report_path is None:
                raise FirstRecoveryBranchOracleAuditError(
                    "rollout_context_report_path is required to recreate v5 policy"
                )
            template = load_mind_v3_founder_template(rollout_context_report_path)
        except Exception as exc:
            setup_error = {
                "reason": type(exc).__name__,
                "message": str(exc),
            }
        if template is not None:
            branch_points, materialization_failures = _materialize_branch_targets(
                selected_rows,
                path_metadata=path_metadata,
                founder_template=template,
            )
            branch_results = [
                _evaluate_branch_target(
                    point,
                    verify_replay=verify_replay,
                )
                for point in branch_points
            ]
    branch_target_selection = dict(selection)
    branch_target_selection.pop("selected_rows", None)
    if setup_error is not None:
        branch_target_selection["setup_error"] = setup_error
        branch_target_selection["skipped_selected_target_count"] = len(selected_rows)
        branch_target_selection["skip_reason_counts"] = _merge_counts(
            _mapping(branch_target_selection.get("skip_reason_counts")),
            {"policy_recreation_failed": len(selected_rows)},
        )
    if _alignment_mismatched(row_alignment):
        branch_target_selection["skipped_selected_target_count"] = len(selected_rows)
        branch_target_selection["skip_reason_counts"] = _merge_counts(
            _mapping(branch_target_selection.get("skip_reason_counts")),
            {"v107_row_alignment_mismatch": len(selected_rows)},
        )
    branch_point_section = _branch_point_section(
        branch_points,
        materialization_failures=materialization_failures,
        selected_row_count=len(selected_rows),
    )
    outcome_oracle = _outcome_oracle(branch_results)
    action_support = _action_support_legality(branch_results)
    resolution_drift = _resolution_drift_root_cause(branch_results)
    seed29 = _seed29_attribution(
        branch_results,
        rollout_context_report=loaded_reports["rollout_context_report"].payload,
        baseline_report=loaded_reports["baseline_report"].payload,
    )
    heldout = _heldout_support(
        row_alignment=row_alignment,
        branch_target_selection=branch_target_selection,
        branch_points=branch_point_section,
        outcome_oracle=outcome_oracle,
        action_support_legality=action_support,
        resolution_drift_root_cause=resolution_drift,
    )
    sections = {
        "v107_row_alignment": row_alignment,
        "branch_target_selection": branch_target_selection,
        "branch_points": branch_point_section,
        "outcome_oracle": outcome_oracle,
        "action_support_legality": action_support,
        "resolution_drift_root_cause": resolution_drift,
        "seed29_attribution": seed29,
        "heldout_support": heldout,
    }
    classification = _classification(evidence=evidence, sections=sections)
    recommendation = _research_recommendation(
        classification=classification,
        sections=sections,
    )
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_POLICY,
        "contract": contract,
        "provenance": _provenance(
            contract=contract,
            loaded_reports=loaded_reports,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=trajectory_glob_patterns,
        ),
        "evidence": evidence,
        "v107_row_alignment": row_alignment,
        "branch_target_selection": branch_target_selection,
        "branch_points": branch_point_section,
        "branch_results": branch_results,
        "outcome_oracle": outcome_oracle,
        "action_support_legality": action_support,
        "resolution_drift_root_cause": resolution_drift,
        "seed29_attribution": seed29,
        "heldout_support": heldout,
        "classification": classification,
        "research_recommendation": recommendation,
        "non_promoted": True,
    }


def _resolve_json_report(
    name: str,
    payload: Mapping[str, object] | None,
    path: str | Path | None,
    *,
    expected_schema: str | None,
    optional: bool = False,
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
                source="inline",
                load_error=None,
                expected_schema=expected_schema,
                optional=optional,
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
                source="missing",
                load_error=None,
                expected_schema=expected_schema,
                optional=optional,
            ),
        )
    try:
        with _open_input(Path(path)) as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise FirstRecoveryBranchOracleAuditError(
                f"{name} must be a JSON object"
            )
        return _LoadedReport(
            name=name,
            path=path_string,
            payload=loaded,
            evidence=_report_evidence(
                path=path_string,
                payload=loaded,
                source="path",
                load_error=None,
                expected_schema=expected_schema,
                optional=optional,
            ),
        )
    except Exception as exc:
        return _LoadedReport(
            name=name,
            path=path_string,
            payload=None,
            evidence=_report_evidence(
                path=path_string,
                payload=None,
                source="path",
                load_error={
                    "reason": type(exc).__name__,
                    "message": str(exc),
                },
                expected_schema=expected_schema,
                optional=optional,
            ),
        )


def _report_evidence(
    *,
    path: str | None,
    payload: Mapping[str, object] | None,
    source: str,
    load_error: Mapping[str, object] | None,
    expected_schema: str | None,
    optional: bool,
) -> dict[str, object]:
    schema = payload.get("schema_version") if payload is not None else None
    schema_matches = (
        None if expected_schema is None or payload is None else schema == expected_schema
    )
    return {
        "path": path,
        "source": source,
        "loaded": payload is not None,
        "optional": optional,
        "schema_version": schema,
        "expected_schema_version": expected_schema,
        "schema_matches": schema_matches,
        "digest": stable_payload_digest(payload) if payload is not None else None,
        "file_sha256": _file_sha256(path) if path is not None else None,
        "load_error": dict(load_error) if load_error is not None else None,
    }


def _v107_row_alignment(
    *,
    v107_report: Mapping[str, object] | None,
    rows: Sequence[Mapping[str, object]],
    path_metadata: Mapping[str, object],
) -> dict[str, object]:
    expected = _int_or_none(
        _mapping(_mapping(v107_report or {}).get("transition_aligned_first_recovery")).get(
            "constructible_first_recovery_row_count"
        )
    )
    by_path: Counter[str] = Counter()
    by_seed: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_seed_source: Counter[str] = Counter()
    for row in rows:
        path = str(row.get("path", ""))
        seed = _int(row.get("seed"), default=_seed_from_path(path))
        source = _source_kind(path)
        by_path[path] += 1
        by_seed[str(seed)] += 1
        by_source[source] += 1
        by_seed_source[f"{source}:{seed}"] += 1
    mismatches: list[dict[str, object]] = []
    if expected is None:
        mismatches.append(
            {
                "field": (
                    "transition_aligned_first_recovery."
                    "constructible_first_recovery_row_count"
                ),
                "reason": "missing_expected_v107_count",
            }
        )
    elif expected != len(rows):
        mismatches.append(
            {
                "field": (
                    "transition_aligned_first_recovery."
                    "constructible_first_recovery_row_count"
                ),
                "expected": expected,
                "observed": len(rows),
                "reason": "reconstructed_row_count_mismatch",
            }
        )
    answer = (
        "missing_evidence_inconclusive"
        if mismatches
        else "first_recovery_branch_replay_partial"
    )
    return {
        "answer": answer,
        "expected_constructible_first_recovery_row_count": expected,
        "reconstructed_first_recovery_row_count": len(rows),
        "row_count_matches_v107": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "by_path": _counter_to_ordered_dict(by_path),
        "by_seed": _counter_to_ordered_dict(by_seed),
        "by_source": _counter_to_ordered_dict(by_source),
        "by_seed_source": _counter_to_ordered_dict(by_seed_source),
        "path_metadata": {
            str(path): dict(_mapping(meta))
            for path, meta in sorted(path_metadata.items())
        },
    }


def _select_branch_targets(
    rows: Sequence[Mapping[str, object]],
    *,
    path_metadata: Mapping[str, object],
    max_targets_per_seed_source: int,
    include_open: bool,
    exhaustive: bool,
) -> dict[str, object]:
    sorted_rows = sorted((dict(row) for row in rows), key=_row_sort_key)
    selected: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    selected_counts: Counter[tuple[str, int]] = Counter()
    for row in sorted_rows:
        path = str(row.get("path", ""))
        source = _source_kind(path)
        seed = _int(row.get("seed"), default=_seed_from_path(path))
        row["source_kind"] = source
        row["ticks"] = _row_ticks(row, path_metadata)
        if source == OPEN_SOURCE_KIND and not (include_open or exhaustive):
            skipped.append(_skip_row(row, "open_source_skipped_by_default"))
            continue
        if source == UNKNOWN_SOURCE_KIND:
            skipped.append(_skip_row(row, "unknown_source_kind"))
            continue
        key = (source, seed)
        if not exhaustive and selected_counts[key] >= max_targets_per_seed_source:
            skipped.append(_skip_row(row, "max_targets_per_seed_source_reached"))
            continue
        selected.append(row)
        selected_counts[key] += 1
    by_seed_source: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_action: Counter[str] = Counter()
    for row in selected:
        by_seed_source[f"{row.get('source_kind')}:{row.get('seed')}"] += 1
        by_source[str(row.get("source_kind"))] += 1
        by_action[str(row.get("requested_action"))] += 1
    skip_counts = Counter(str(item["reason"]) for item in skipped)
    return {
        "answer": "first_recovery_branch_replay_partial",
        "selection_policy": (
            "all_rows_exhaustive_v1"
            if exhaustive
            else "fixture_first_one_target_per_seed_source_v1"
        ),
        "exhaustive": bool(exhaustive),
        "include_open": bool(include_open),
        "max_targets_per_seed_source": max_targets_per_seed_source,
        "candidate_row_count": len(sorted_rows),
        "selected_target_count": len(selected),
        "skipped_target_count": len(skipped),
        "selected_rows": selected,
        "selected_targets": [_target_row_excerpt(row) for row in selected],
        "skipped_examples": skipped[:24],
        "skip_reason_counts": _counter_to_ordered_dict(skip_counts),
        "selected_by_seed_source": _counter_to_ordered_dict(by_seed_source),
        "selected_by_source": _counter_to_ordered_dict(by_source),
        "selected_by_logged_action": _counter_to_ordered_dict(by_action),
        "fixture_seed_coverage": sorted(
            {
                _int(row.get("seed"))
                for row in selected
                if row.get("source_kind") == FIXTURE_SOURCE_KIND
            }
        ),
        "seed29_selected": any(_int(row.get("seed")) == 29 for row in selected),
    }


def _materialize_branch_targets(
    selected_rows: Sequence[Mapping[str, object]],
    *,
    path_metadata: Mapping[str, object],
    founder_template: object,
) -> tuple[list[_BranchTarget], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in selected_rows:
        path = str(row.get("path", ""))
        source = _source_kind(path)
        seed = _int(row.get("seed"), default=_seed_from_path(path))
        ticks = _row_ticks(row, path_metadata)
        grouped[(path, source, seed, ticks)].append(dict(row))
    points: list[_BranchTarget] = []
    failures: list[dict[str, object]] = []
    for (path, source, seed, ticks), rows in sorted(grouped.items()):
        try:
            group_points, group_failures = _materialize_target_group(
                rows,
                source_path=path,
                source_kind=source,
                seed=seed,
                ticks=ticks,
                founder_template=founder_template,
            )
            points.extend(group_points)
            failures.extend(group_failures)
        except Exception as exc:
            failures.extend(
                {
                    **_target_row_excerpt(row),
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
                for row in rows
            )
    points.sort(key=lambda point: (point.source_path, point.record_index))
    return points, failures


def _materialize_target_group(
    rows: Sequence[Mapping[str, object]],
    *,
    source_path: str,
    source_kind: str,
    seed: int,
    ticks: int,
    founder_template: object,
) -> tuple[list[_BranchTarget], list[dict[str, object]]]:
    world = _world_for_source(
        source_kind=source_kind,
        seed=seed,
        ticks=ticks,
        founder_template=founder_template,
    )
    _configure_manual_summary_run(world)
    targets_by_index = {
        _int(row.get("recovery_record_index"), default=-1): dict(row) for row in rows
    }
    remaining = set(targets_by_index)
    points: list[_BranchTarget] = []
    failures: list[dict[str, object]] = []
    record_index = 0
    for tick in range(ticks):
        world.tick = tick
        snapshot = deepcopy(world)
        world._run_tick()
        for record in world.tick_trajectory_records:
            row = targets_by_index.get(record_index)
            if row is not None:
                matched, mismatch = _target_matches_record(
                    row,
                    record,
                    record_index=record_index,
                    source_path=source_path,
                    source_kind=source_kind,
                )
                if matched:
                    branch_index = len(points)
                    branch_id = _branch_id(
                        source_kind=source_kind,
                        seed=seed,
                        branch_index=branch_index,
                        tick=tick,
                        agent_id=_int(row.get("agent_id")),
                        action=str(row.get("requested_action")),
                    )
                    points.append(
                        _BranchTarget(
                            branch_id=branch_id,
                            source_path=source_path,
                            source_kind=source_kind,
                            seed=seed,
                            ticks=ticks,
                            branch_tick=_int(row.get("recovery_tick")),
                            record_index=record_index,
                            agent_id=_int(row.get("agent_id")),
                            logged_action=str(row.get("requested_action")),
                            gain_tick=_int_or_none(row.get("gain_tick")),
                            gain_record_index=_int(
                                row.get("gain_record_index"),
                                default=-1,
                            ),
                            observation_digest=_optional_string(
                                row.get("observation_digest")
                            ),
                            observation_schema=_optional_string(
                                record.get("observation_schema")
                            ),
                            action_mask=_bool_mapping(row.get("action_mask")),
                            resolution_action_mask=_bool_mapping(
                                row.get("resolution_action_mask")
                            ),
                            before=dict(_mapping(row.get("before"))),
                            branch_state_digest=_branch_state_digest(
                                snapshot,
                                branch_id=branch_id,
                                branch_tick=tick,
                            ),
                            world=snapshot,
                        )
                    )
                else:
                    failures.append(
                        {
                            **_target_row_excerpt(row),
                            "reason": "target_materialization_mismatch",
                            "mismatches": mismatch,
                        }
                    )
                remaining.discard(record_index)
            record_index += 1
        if not remaining:
            break
        if not world.alive_agents():
            break
    for missing_index in sorted(remaining):
        failures.append(
            {
                **_target_row_excerpt(targets_by_index[missing_index]),
                "reason": "target_record_not_materialized",
            }
        )
    return points, failures


def _world_for_source(
    *,
    source_kind: str,
    seed: int,
    ticks: int,
    founder_template: object,
) -> SimulationWorld:
    policy = MindV3EvolutionPolicy(
        seed=seed,
        founder_template_metadata=founder_template,  # type: ignore[arg-type]
    )
    if source_kind == FIXTURE_SOURCE_KIND:
        return _fixture_world(
            fixture_name="carrion_only",
            seed=seed,
            ticks=ticks,
            policy=policy,
        )
    if source_kind == OPEN_SOURCE_KIND:
        return SimulationWorld(WorldConfig(seed=seed, max_ticks=ticks), policy=policy)
    raise FirstRecoveryBranchOracleAuditError(f"unsupported source kind: {source_kind}")


def _evaluate_branch_target(
    point: _BranchTarget,
    *,
    verify_replay: bool,
) -> dict[str, object]:
    action_runs = [
        _execute_action_branch(point, forced_action=action, verify_replay=verify_replay)
        for action in _legal_candidate_actions(point.action_mask)
    ]
    logged = _run_for_action(action_runs, point.logged_action)
    best = _best_oracle_run(action_runs, logged)
    deltas = _oracle_deltas(best, logged)
    material_gain = any(
        _number(deltas.get(field)) > 0.0
        for field in (
            "target_alive_delta",
            "terminal_alive_delta",
            "birth_delta",
            "target_recovery_score_delta",
            "death_reduction_delta",
        )
    )
    return {
        "branch_id": point.branch_id,
        "source_path": point.source_path,
        "source_kind": point.source_kind,
        "seed": point.seed,
        "ticks": point.ticks,
        "branch_tick": point.branch_tick,
        "record_index": point.record_index,
        "agent_id": point.agent_id,
        "logged_action": point.logged_action,
        "gain_tick": point.gain_tick,
        "gain_record_index": point.gain_record_index,
        "observation_digest": point.observation_digest,
        "branch_state_digest": point.branch_state_digest,
        "candidate_action_count": len(action_runs),
        "action_runs": action_runs,
        "logged_action_run": _run_excerpt(logged),
        "oracle_best_action_run": _run_excerpt(best),
        "oracle_best_action": best.get("forced_action") if best is not None else None,
        "oracle_changed_action": bool(
            best is not None and best.get("forced_action") != point.logged_action
        ),
        "oracle_deltas_vs_logged": deltas,
        "material_oracle_gain": bool(material_gain),
        "diagnostics_only": True,
    }


def _execute_action_branch(
    point: _BranchTarget,
    *,
    forced_action: str,
    verify_replay: bool,
) -> dict[str, object]:
    run, digest = _execute_once(point, forced_action=forced_action)
    verification = None
    if verify_replay:
        replay, replay_digest = _execute_once(point, forced_action=forced_action)
        verification = {
            "verified": replay_digest == digest,
            "expected_digest": digest,
            "actual_digest": replay_digest,
            "replay_alive_agents": replay.get("alive_agents"),
            "replay_births": replay.get("births"),
            "replay_deaths": replay.get("deaths"),
        }
    run["replay_verification"] = verification
    return run


def _execute_once(
    point: _BranchTarget,
    *,
    forced_action: str,
) -> tuple[dict[str, object], str]:
    world = deepcopy(point.world)
    delegate = world.policy
    world.policy = _ForcedFirstActionThenDelegatePolicy(
        target_agent_id=point.agent_id,
        forced_action=forced_action,
        delegate=delegate,
    )
    _configure_manual_summary_run(world)
    population_horizon_trace: list[dict[str, object]] = []
    for tick in range(point.branch_tick, point.ticks):
        world.tick = tick
        world._run_tick()
        horizon_delta = tick - point.branch_tick
        if horizon_delta in DEFAULT_TARGET_HORIZON_TRACE_TICKS:
            population_horizon_trace.append(
                _population_horizon_snapshot(
                    world,
                    point=point,
                    forced_action=forced_action,
                    horizon_tick_delta=horizon_delta,
                )
            )
        if not world.alive_agents():
            break
    policy = world.policy
    forced_used = bool(getattr(policy, "used", False))
    run = _summarize_branch_world(
        world,
        point=point,
        forced_action=forced_action,
        forced_action_used=forced_used,
        population_horizon_trace=population_horizon_trace,
    )
    return run, stable_payload_digest(_digest_payload(run))


def _summarize_branch_world(
    world: SimulationWorld,
    *,
    point: _BranchTarget,
    forced_action: str,
    forced_action_used: bool,
    population_horizon_trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)
    action_source_counts = Counter(
        str(record.get("action_source", "unknown"))
        for record in world.trajectory_records
    )
    policy_id_counts = Counter(
        str(record.get("policy_id", "unknown")) for record in world.trajectory_records
    )
    requested_action_counts = Counter(
        str(record["requested_action"])
        for record in world.trajectory_records
        if isinstance(record.get("requested_action"), str)
    )
    target = world.agents.get(point.agent_id)
    target_alive = bool(target is not None and target.alive)
    dominant = _dominant_action_summary(requested_action_counts)
    first_outcome = _first_target_action_outcome(
        world.trajectory_records,
        point=point,
        forced_action=forced_action,
    )
    return {
        "branch_id": point.branch_id,
        "source_kind": point.source_kind,
        "seed": point.seed,
        "ticks": point.ticks,
        "branch_tick": point.branch_tick,
        "record_index": point.record_index,
        "agent_id": point.agent_id,
        "logged_action": point.logged_action,
        "forced_action": forced_action,
        "forced_action_used": bool(forced_action_used),
        "forced_action_supported": bool(point.action_mask.get(forced_action, False)),
        "branch_state_digest": point.branch_state_digest,
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "terminal_alive_agents": int(summary["alive_agents"]),
        "target_alive_at_end": target_alive,
        "target_energy_ratio_at_end": (
            _round(world._energy_ratio(target)) if target_alive else None
        ),
        "target_hydration_ratio_at_end": (
            _round(world._hydration_ratio(target)) if target_alive else None
        ),
        "target_health_ratio_at_end": (
            _round(world._health_ratio(target)) if target_alive else None
        ),
        "target_recovery_score_at_end": _target_recovery_score_from_values(
            (
                _round(world._energy_ratio(target)) if target_alive else None,
                _round(world._hydration_ratio(target)) if target_alive else None,
                _round(world._health_ratio(target)) if target_alive else None,
            )
        ),
        "first_action_outcome": first_outcome,
        "target_horizon_trace": _target_horizon_trace(
            world.trajectory_records,
            point=point,
            forced_action=forced_action,
            horizons=DEFAULT_TARGET_HORIZON_TRACE_TICKS,
        ),
        "population_horizon_trace": [dict(item) for item in population_horizon_trace],
        "trajectory_record_count": len(world.trajectory_records),
        "heuristic_action_source_count": _heuristic_action_source_count(
            action_source_counts
        ),
        "diagnostic_forced_action_source_count": sum(
            count
            for source, count in action_source_counts.items()
            if source.startswith("branch_oracle_force:")
        ),
        "zero_heuristic_runtime_actions": (
            _heuristic_action_source_count(action_source_counts) == 0
        ),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "dominant_requested_action": dominant["action"],
        "dominant_requested_action_count": dominant["count"],
        "dominant_requested_action_share": dominant["share"],
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
        "summary_excerpt": {
            "trophic_role_counts_at_end": _json_ready(
                summary.get("trophic_role_counts_at_end", {})
            ),
            "meat_mode_counts_at_end": _json_ready(
                summary.get("meat_mode_counts_at_end", {})
            ),
        },
    }


def _branch_point_section(
    branch_points: Sequence[_BranchTarget],
    *,
    materialization_failures: Sequence[Mapping[str, object]],
    selected_row_count: int,
) -> dict[str, object]:
    by_seed_source: Counter[str] = Counter()
    for point in branch_points:
        by_seed_source[f"{point.source_kind}:{point.seed}"] += 1
    if materialization_failures:
        answer = "first_recovery_branch_replay_failed"
    elif branch_points and len(branch_points) == selected_row_count:
        answer = "first_recovery_branch_replay_partial"
    else:
        answer = "missing_evidence_inconclusive"
    return {
        "answer": answer,
        "selected_row_count": selected_row_count,
        "materialized_branch_point_count": len(branch_points),
        "materialization_failure_count": len(materialization_failures),
        "materialization_failures": [dict(item) for item in materialization_failures[:24]],
        "by_seed_source": _counter_to_ordered_dict(by_seed_source),
        "points": [_branch_point_payload(point) for point in branch_points],
        "private_world_state_serialized": False,
    }


def _outcome_oracle(branch_results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    action_runs = [
        run
        for result in branch_results
        for run in _list_of_mappings(result.get("action_runs"))
    ]
    replay_items = [
        item
        for run in action_runs
        for item in [_mapping(run.get("replay_verification"))]
        if item
    ]
    replay_verified = bool(replay_items) and all(
        bool(item.get("verified")) for item in replay_items
    )
    heuristic_count = sum(_int(run.get("heuristic_action_source_count")) for run in action_runs)
    force_count = sum(
        _int(run.get("diagnostic_forced_action_source_count")) for run in action_runs
    )
    oracle_actions = [
        str(result.get("oracle_best_action"))
        for result in branch_results
        if result.get("oracle_best_action") is not None
    ]
    action_counts = Counter(oracle_actions)
    dominant = _dominant_count_share(action_counts)
    material_results = [
        result for result in branch_results if result.get("material_oracle_gain") is True
    ]
    changed = [
        result for result in branch_results if result.get("oracle_changed_action") is True
    ]
    if not branch_results:
        answer = "missing_evidence_inconclusive"
    elif material_results:
        answer = "first_recovery_oracle_improves_logged_action"
    elif all(result.get("oracle_changed_action") is False for result in branch_results):
        answer = "first_recovery_oracle_matches_logged_action"
    else:
        answer = "first_recovery_oracle_no_material_gain"
    distribution_answer = (
        "first_recovery_oracle_action_distribution_collapsed"
        if _number(dominant.get("share")) > DEFAULT_DOMINANT_ORACLE_ACTION_SHARE_MAX
        else "first_recovery_oracle_action_distribution_clean"
    )
    replay_answer = (
        "first_recovery_branch_replay_verified"
        if replay_verified
        else (
            "first_recovery_branch_replay_failed"
            if replay_items
            else "missing_evidence_inconclusive"
        )
    )
    return {
        "answer": answer,
        "replay_answer": replay_answer,
        "distribution_answer": distribution_answer,
        "branch_result_count": len(branch_results),
        "action_run_count": len(action_runs),
        "replay_verification_count": len(replay_items),
        "replay_verified": replay_verified,
        "heuristic_action_source_count": heuristic_count,
        "diagnostic_forced_action_source_count": force_count,
        "zero_heuristic_runtime_actions_except_diagnostic_force": heuristic_count == 0,
        "oracle_changed_action_count": len(changed),
        "material_oracle_gain_count": len(material_results),
        "material_oracle_gain_seed_count": len(
            {int(result.get("seed", -1)) for result in material_results}
        ),
        "oracle_action_counts": dict(sorted(action_counts.items())),
        "dominant_oracle_action": dominant["key"],
        "dominant_oracle_action_count": dominant["count"],
        "dominant_oracle_action_share": dominant["share"],
        "examples": [_oracle_result_excerpt(result) for result in branch_results[:24]],
    }


def _action_support_legality(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    observation_counts = Counter()
    resolution_counts = Counter()
    examples: list[dict[str, object]] = []
    for result in branch_results:
        run = _mapping(result.get("oracle_best_action_run"))
        action = str(result.get("oracle_best_action", ""))
        first = _mapping(run.get("first_action_outcome"))
        observation_legal = bool(first.get("observation_legal", False))
        resolution_legal = bool(first.get("resolution_legal", False))
        observation_counts[
            "oracle_supported_actions_observation_legal"
            if observation_legal
            else "oracle_supported_actions_not_observation_legal"
        ] += 1
        resolution_counts[
            "oracle_supported_actions_resolution_legal"
            if resolution_legal
            else "oracle_supported_actions_resolution_invalid"
        ] += 1
        if len(examples) < 16:
            examples.append(
                {
                    "branch_id": result.get("branch_id"),
                    "seed": result.get("seed"),
                    "oracle_best_action": action,
                    "observation_legal": observation_legal,
                    "resolution_legal": resolution_legal,
                    "invalid_reason": first.get("invalid_reason"),
                }
            )
    answer = (
        "oracle_supported_actions_not_observation_legal"
        if observation_counts["oracle_supported_actions_not_observation_legal"] > 0
        else (
            "oracle_supported_actions_observation_legal"
            if observation_counts["oracle_supported_actions_observation_legal"] > 0
            else "missing_evidence_inconclusive"
        )
    )
    resolution_answer = (
        "oracle_supported_actions_resolution_invalid"
        if resolution_counts["oracle_supported_actions_resolution_invalid"] > 0
        else (
            "oracle_supported_actions_resolution_legal"
            if resolution_counts["oracle_supported_actions_resolution_legal"] > 0
            else "missing_evidence_inconclusive"
        )
    )
    labels = [answer]
    if resolution_answer != answer:
        labels.append(resolution_answer)
    return {
        "answer": answer,
        "resolution_answer": resolution_answer,
        "labels": labels,
        "oracle_action_count": len(branch_results),
        "observation_category_counts": _counter_to_ordered_dict(observation_counts),
        "resolution_category_counts": _counter_to_ordered_dict(resolution_counts),
        "observation_legal_share": _share(
            observation_counts["oracle_supported_actions_observation_legal"],
            len(branch_results),
        ),
        "resolution_invalid_count": resolution_counts[
            "oracle_supported_actions_resolution_invalid"
        ],
        "examples": examples,
    }


def _resolution_drift_root_cause(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts = Counter()
    examples: list[dict[str, object]] = []
    for result in branch_results:
        run = _mapping(result.get("oracle_best_action_run"))
        first = _mapping(run.get("first_action_outcome"))
        if bool(first.get("resolution_legal", False)):
            continue
        cause = _resolution_invalid_cause(
            action=str(result.get("oracle_best_action", "")),
            observation_legal=bool(first.get("observation_legal", False)),
        )
        counts[cause] += 1
        if len(examples) < 16:
            examples.append(
                {
                    "branch_id": result.get("branch_id"),
                    "seed": result.get("seed"),
                    "oracle_best_action": result.get("oracle_best_action"),
                    "cause": cause,
                    "first_action_outcome": dict(first),
                }
            )
    if counts["resolution_invalid_occupancy_race"] > 0:
        answer = "resolution_invalid_occupancy_race"
    elif counts["resolution_invalid_blocked_route"] > 0:
        answer = "resolution_invalid_blocked_route"
    elif counts["resolution_invalid_depleted_resource"] > 0:
        answer = "resolution_invalid_depleted_resource"
    elif counts["resolution_invalid_other_public_path"] > 0:
        answer = "resolution_invalid_other_public_path"
    else:
        answer = "resolution_invalid_inconclusive"
    return {
        "answer": answer,
        "invalid_oracle_action_count": sum(counts.values()),
        "category_counts": _counter_to_ordered_dict(counts),
        "examples": examples,
    }


def _seed29_attribution(
    branch_results: Sequence[Mapping[str, object]],
    *,
    rollout_context_report: Mapping[str, object] | None,
    baseline_report: Mapping[str, object] | None,
) -> dict[str, object]:
    results = [result for result in branch_results if _int(result.get("seed")) == 29]
    candidate = _seed_run(rollout_context_report, seed=29)
    baseline = _seed_run(baseline_report, seed=29)
    births_delta = _delta(candidate.get("births"), baseline.get("births"))
    answer = "seed29_public_attribution_inconclusive"
    if results:
        oracle_runs = [_mapping(result.get("oracle_best_action_run")) for result in results]
        first_outcomes = [_mapping(run.get("first_action_outcome")) for run in oracle_runs]
        if any(first.get("resolution_legal") is False for first in first_outcomes):
            answer = "seed29_public_attribution_unsupported_resolution"
        elif any(
            str(result.get("oracle_best_action")) in MOVEMENT_ACTIONS
            and _mapping(_mapping(result.get("oracle_best_action_run")).get("first_action_outcome")).get(
                "moved"
            )
            is False
            for result in results
        ):
            answer = "seed29_public_attribution_oracle_movement_failure"
        elif any(
            result.get("logged_action") != "drink"
            and result.get("oracle_best_action") == "drink"
            and (
                _mapping(_mapping(result.get("oracle_best_action_run")).get("first_action_outcome")).get(
                    "drank"
                )
                is True
                or _number(
                    _mapping(
                        _mapping(result.get("oracle_best_action_run")).get(
                            "first_action_outcome"
                        )
                    ).get("hydration_ratio_delta")
                )
                > 0.0
            )
            for result in results
        ):
            answer = "seed29_public_attribution_missed_drink"
        elif any(
            _number(_mapping(result.get("oracle_deltas_vs_logged")).get("birth_delta"))
            > 0.0
            or _number(
                _mapping(result.get("oracle_deltas_vs_logged")).get(
                    "target_recovery_score_delta"
                )
            )
            > 0.0
            for result in results
        ):
            answer = "seed29_public_attribution_delayed_reproduction_readiness"
    return {
        "answer": answer,
        "seed": 29,
        "branch_result_count": len(results),
        "births_delta_candidate_vs_baseline": births_delta,
        "examples": [_oracle_result_excerpt(result) for result in results[:12]],
    }


def _heldout_support(
    *,
    row_alignment: Mapping[str, object],
    branch_target_selection: Mapping[str, object],
    branch_points: Mapping[str, object],
    outcome_oracle: Mapping[str, object],
    action_support_legality: Mapping[str, object],
    resolution_drift_root_cause: Mapping[str, object],
) -> dict[str, object]:
    fixture_seeds = set(_list(branch_target_selection.get("fixture_seed_coverage")))
    all_fixture_seeds = {13, 19, 29, 37, 41, 43}
    reconstructed = row_alignment.get("row_count_matches_v107") is True
    branch_replay = outcome_oracle.get("replay_verified") is True
    heuristic_clean = (
        outcome_oracle.get("zero_heuristic_runtime_actions_except_diagnostic_force")
        is True
    )
    observation_legal_share = _number(
        action_support_legality.get("observation_legal_share")
    )
    resolution_invalid_count = _int(
        action_support_legality.get("resolution_invalid_count")
    )
    resolution_answer = str(resolution_drift_root_cause.get("answer"))
    material_seed_count = _int(outcome_oracle.get("material_oracle_gain_seed_count"))
    dominant_share = _number(outcome_oracle.get("dominant_oracle_action_share"))
    materialized = _int(branch_points.get("materialized_branch_point_count"))
    materialization_failures = _int(branch_points.get("materialization_failure_count"))
    sufficient = (
        reconstructed
        and branch_replay
        and all_fixture_seeds.issubset(fixture_seeds)
        and 29 in fixture_seeds
        and heuristic_clean
        and observation_legal_share >= 0.9
        and (
            resolution_invalid_count == 0
            or resolution_answer != "resolution_invalid_inconclusive"
        )
        and material_seed_count >= DEFAULT_MIN_MATERIAL_GAIN_SEED_COUNT
        and dominant_share <= DEFAULT_DOMINANT_ORACLE_ACTION_SHARE_MAX
        and materialized > 0
        and materialization_failures == 0
    )
    answer = (
        "heldout_branch_support_sufficient_for_v109"
        if sufficient
        else "heldout_branch_support_insufficient_for_v109"
    )
    blockers: list[dict[str, object]] = []
    if not reconstructed:
        blockers.append({"reason": "v107_row_reconstruction_mismatch"})
    if not branch_replay:
        blockers.append({"reason": "branch_replay_not_verified"})
    if not all_fixture_seeds.issubset(fixture_seeds):
        blockers.append(
            {
                "reason": "not_all_fixture_seeds_verified",
                "observed": sorted(fixture_seeds),
                "required": sorted(all_fixture_seeds),
            }
        )
    if not heuristic_clean:
        blockers.append({"reason": "heuristic_action_source_count_nonzero"})
    if observation_legal_share < 0.9:
        blockers.append(
            {
                "reason": "oracle_actions_not_mostly_observation_legal",
                "observed": observation_legal_share,
            }
        )
    if resolution_invalid_count > 0 and resolution_answer == "resolution_invalid_inconclusive":
        blockers.append({"reason": "resolution_invalidity_not_publicly_explained"})
    if material_seed_count < DEFAULT_MIN_MATERIAL_GAIN_SEED_COUNT:
        blockers.append(
            {
                "reason": "material_oracle_gains_sparse",
                "observed": material_seed_count,
                "required": DEFAULT_MIN_MATERIAL_GAIN_SEED_COUNT,
            }
        )
    if dominant_share > DEFAULT_DOMINANT_ORACLE_ACTION_SHARE_MAX:
        blockers.append(
            {
                "reason": "oracle_action_distribution_collapsed",
                "observed": dominant_share,
                "required_max": DEFAULT_DOMINANT_ORACLE_ACTION_SHARE_MAX,
            }
        )
    if materialization_failures > 0:
        blockers.append(
            {
                "reason": "target_materialization_failures",
                "observed": materialization_failures,
            }
        )
    return {
        "answer": answer,
        "reconstructed_v107_rows": reconstructed,
        "branch_replay_verified": branch_replay,
        "fixture_seed_coverage": sorted(fixture_seeds),
        "seed29_covered": 29 in fixture_seeds,
        "heuristic_action_source_count_clean": heuristic_clean,
        "oracle_observation_legal_share": observation_legal_share,
        "resolution_invalid_oracle_action_count": resolution_invalid_count,
        "resolution_invalid_public_cause": resolution_answer,
        "material_oracle_gain_seed_count": material_seed_count,
        "dominant_oracle_action_share": dominant_share,
        "blocker_count": len(blockers),
        "blockers": blockers,
    }


def _classification(
    *,
    evidence: Mapping[str, object],
    sections: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    missing = _missing_evidence(evidence, sections)
    scores = {label: 0 for label in ALLOWED_CLASSIFICATION_LABELS}
    if missing:
        scores["missing_evidence_inconclusive"] = 100
        return {
            "primary": "missing_evidence_inconclusive",
            "labels": ["missing_evidence_inconclusive"],
            "category_scores": scores,
            "diagnostic_answers": _diagnostic_answers(sections),
            "missing_evidence": missing,
        }
    labels: list[str] = []
    for section in sections.values():
        for label in _section_labels(section):
            if label in CLASSIFICATION_LABEL_SET and label not in labels:
                labels.append(label)
                scores[label] += 1
    if (
        sections["branch_target_selection"].get("exhaustive") is not True
        and "first_recovery_branch_replay_verified" in labels
    ):
        labels.remove("first_recovery_branch_replay_verified")
        if "first_recovery_branch_replay_partial" not in labels:
            labels.insert(0, "first_recovery_branch_replay_partial")
        scores["first_recovery_branch_replay_partial"] += 10
    if not labels:
        labels = ["missing_evidence_inconclusive"]
        scores["missing_evidence_inconclusive"] = 1
    primary = _primary_label(labels)
    return {
        "primary": primary,
        "labels": labels,
        "category_scores": scores,
        "diagnostic_answers": _diagnostic_answers(sections),
        "missing_evidence": [],
    }


def _research_recommendation(
    *,
    classification: Mapping[str, object],
    sections: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    heldout = _mapping(sections.get("heldout_support"))
    sufficient = heldout.get("answer") == "heldout_branch_support_sufficient_for_v109"
    return {
        "classification_primary": classification.get("primary"),
        "recommend_v109": bool(sufficient),
        "recommendation": (
            "planner_may_start_v109_from_first_recovery_branch_oracle_support"
            if sufficient
            else "do_not_promote; collect_or_align_more_first_recovery_branch_oracle_support_before_v109"
        ),
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_relaxation_recommended": False,
        "next_research_direction": (
            "Planner-led branch-oracle evidence exactly over v107 first-recovery rows."
        ),
        "go_stop": {
            "go_for_v109": bool(sufficient),
            "stop_if_exact_target_rows_cannot_be_replayed": True,
            "stop_if_seed29_remains_inconclusive": True,
            "stop_if_material_oracle_gains_are_sparse": True,
            "stop_if_oracle_actions_collapse": True,
            "stop_if_resolution_invalidity_lacks_public_cause": True,
        },
    }


def _contract(
    *,
    max_targets_per_seed_source: int,
    include_open: bool,
    exhaustive: bool,
    verify_replay: bool,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_POLICY,
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "private_world_state_serialized": False,
        "fixture_identity_runtime_policy_input": False,
        "hidden_heuristic_action_selection": False,
        "cache_internal_mutation": False,
        "candidate_action_policy": "observation_legal_actions_in_ACTION_NAMES_order",
        "first_action_policy": "diagnostic_forced_first_action_then_delegate_v5_policy",
        "continuation_policy": "copied_mind_v3_rollout_context_policy_state",
        "selection": {
            "max_targets_per_seed_source": int(max_targets_per_seed_source),
            "include_open": bool(include_open),
            "exhaustive": bool(exhaustive),
        },
        "verify_replay": bool(verify_replay),
    }


def _provenance(
    *,
    contract: Mapping[str, object],
    loaded_reports: Mapping[str, _LoadedReport],
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
) -> dict[str, object]:
    return {
        "contract_digest": stable_payload_digest(contract),
        "input_reports": {
            name: {
                "path": loaded.path,
                "digest": loaded.evidence.get("digest"),
                "file_sha256": loaded.evidence.get("file_sha256"),
                "loaded": loaded.payload is not None,
            }
            for name, loaded in sorted(loaded_reports.items())
        },
        "trajectory_paths": [str(path) for path in trajectory_paths],
        "trajectory_globs": list(trajectory_glob_patterns),
        "trajectory_file_sha256": {
            str(path): _file_sha256(str(path)) for path in trajectory_paths
        },
    }


def _target_matches_record(
    row: Mapping[str, object],
    record: Mapping[str, object],
    *,
    record_index: int,
    source_path: str,
    source_kind: str,
) -> tuple[bool, list[dict[str, object]]]:
    checks = {
        "source_path": (source_path, str(row.get("path", ""))),
        "source_kind": (source_kind, _source_kind(str(row.get("path", "")))),
        "seed": (_int(row.get("seed")), _int(row.get("seed"))),
        "tick": (_int(record.get("tick"), default=-1), _int(row.get("recovery_tick"), default=-2)),
        "agent_id": (_int(record.get("agent_id"), default=-1), _int(row.get("agent_id"), default=-2)),
        "requested_action": (
            str(record.get("requested_action", "")),
            str(row.get("requested_action", "")),
        ),
        "record_index": (record_index, _int(row.get("recovery_record_index"), default=-1)),
    }
    digest = _optional_string(row.get("observation_digest"))
    if digest is not None:
        checks["observation_digest"] = (
            _optional_string(record.get("observation_digest")),
            digest,
        )
    mismatches = [
        {"field": field, "observed": observed, "expected": expected}
        for field, (observed, expected) in checks.items()
        if observed != expected
    ]
    return not mismatches, mismatches


def _legal_candidate_actions(action_mask: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(action for action in ACTION_NAMES if bool(action_mask.get(action, False)))


def _best_oracle_run(
    runs: Sequence[Mapping[str, object]],
    logged: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if not runs:
        return None
    return max(
        runs,
        key=lambda run: (
            int(bool(run.get("target_alive_at_end", False))),
            _int(run.get("terminal_alive_agents")),
            _int(run.get("births")),
            _number(run.get("target_recovery_score_at_end")),
            -_int(run.get("deaths")),
            str(run.get("forced_action", "")),
        ),
    )


def _oracle_deltas(
    best: Mapping[str, object] | None,
    logged: Mapping[str, object] | None,
) -> dict[str, object]:
    if best is None or logged is None:
        return {
            "target_alive_delta": None,
            "terminal_alive_delta": None,
            "birth_delta": None,
            "target_recovery_score_delta": None,
            "death_reduction_delta": None,
        }
    return {
        "target_alive_delta": int(bool(best.get("target_alive_at_end"))) - int(
            bool(logged.get("target_alive_at_end"))
        ),
        "terminal_alive_delta": _int(best.get("terminal_alive_agents")) - _int(
            logged.get("terminal_alive_agents")
        ),
        "birth_delta": _int(best.get("births")) - _int(logged.get("births")),
        "target_recovery_score_delta": _round(
            _number(best.get("target_recovery_score_at_end"))
            - _number(logged.get("target_recovery_score_at_end"))
        ),
        "death_reduction_delta": _int(logged.get("deaths")) - _int(best.get("deaths")),
    }


def _first_target_action_outcome(
    trajectory_records: Sequence[Mapping[str, object]],
    *,
    point: _BranchTarget,
    forced_action: str,
) -> dict[str, object]:
    for record in trajectory_records:
        if (
            _int(record.get("tick"), default=-1) == point.branch_tick
            and _int(record.get("agent_id"), default=-1) == point.agent_id
        ):
            return _target_record_outcome(
                record,
                point=point,
                forced_action=forced_action,
                horizon_tick_delta=0,
            )
    return {
        "record_found": False,
        "branch_tick": point.branch_tick,
        "agent_id": point.agent_id,
        "forced_action": forced_action,
    }


def _target_horizon_trace(
    trajectory_records: Sequence[Mapping[str, object]],
    *,
    point: _BranchTarget,
    forced_action: str,
    horizons: Sequence[int],
) -> list[dict[str, object]]:
    records_by_tick = {
        _int(record.get("tick"), default=-1): record
        for record in trajectory_records
        if _int(record.get("agent_id"), default=-1) == point.agent_id
    }
    trace: list[dict[str, object]] = []
    for horizon in horizons:
        tick_delta = max(0, int(horizon))
        record = records_by_tick.get(point.branch_tick + tick_delta)
        if record is None:
            trace.append(
                {
                    "record_found": False,
                    "horizon_tick_delta": tick_delta,
                    "tick": point.branch_tick + tick_delta,
                    "agent_id": point.agent_id,
                    "forced_action": forced_action,
                }
            )
            continue
        trace.append(
            _target_record_outcome(
                record,
                point=point,
                forced_action=forced_action,
                horizon_tick_delta=tick_delta,
            )
        )
    return trace


def _target_record_outcome(
    record: Mapping[str, object],
    *,
    point: _BranchTarget,
    forced_action: str,
    horizon_tick_delta: int,
) -> dict[str, object]:
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    outcome = _mapping(record.get("outcome"))
    drinking = _mapping(outcome.get("drinking"))
    feeding = _mapping(outcome.get("feeding"))
    passive = _mapping(outcome.get("passive"))
    action_mask = _bool_mapping(record.get("action_mask"))
    resolution_mask = _bool_mapping(record.get("resolution_action_mask"))
    invalid_reason = record.get("invalid_reason")
    if invalid_reason is None:
        invalid_reason = outcome.get("invalid_reason")
    return {
        "record_found": True,
        "horizon_tick_delta": int(horizon_tick_delta),
        "tick": _int(record.get("tick"), default=-1),
        "branch_tick": point.branch_tick,
        "agent_id": point.agent_id,
        "forced_action": forced_action,
        "requested_action": _optional_string(record.get("requested_action")),
        "resolved_action": _optional_string(record.get("resolved_action")),
        "action_valid": bool(record.get("action_valid", False)),
        "resolution_action_valid": bool(record.get("resolution_action_valid", False)),
        "observation_legal": bool(action_mask.get(forced_action, False)),
        "resolution_legal": bool(resolution_mask.get(forced_action, False)),
        "invalid_reason": invalid_reason,
        "moved": bool(record.get("moved", False)),
        "alive_before": bool(before.get("alive", False)),
        "alive_after": bool(after.get("alive", False)),
        "x_delta": _int(after.get("x")) - _int(before.get("x")),
        "y_delta": _int(after.get("y")) - _int(before.get("y")),
        "energy_ratio_before": _number_or_none(before.get("energy_ratio")),
        "energy_ratio_after": _number_or_none(after.get("energy_ratio")),
        "energy_ratio_delta": _delta(after.get("energy_ratio"), before.get("energy_ratio")),
        "hydration_ratio_before": _number_or_none(before.get("hydration_ratio")),
        "hydration_ratio_after": _number_or_none(after.get("hydration_ratio")),
        "hydration_ratio_delta": _delta(
            after.get("hydration_ratio"),
            before.get("hydration_ratio"),
        ),
        "health_ratio_before": _number_or_none(before.get("health_ratio")),
        "health_ratio_after": _number_or_none(after.get("health_ratio")),
        "health_ratio_delta": _delta(after.get("health_ratio"), before.get("health_ratio")),
        "resource_gain": _number_or_none(outcome.get("resource_gain")),
        "drank": bool(drinking.get("drank", False)),
        "ate": bool(feeding.get("ate", False)),
        "died": bool(outcome.get("died", False)),
        "death_cause": _optional_string(passive.get("death_cause")),
        "died_after_action": bool(passive.get("died_after_action", False)),
        "action_mask": action_mask,
        "resolution_action_mask": resolution_mask,
    }


def _population_horizon_snapshot(
    world: SimulationWorld,
    *,
    point: _BranchTarget,
    forced_action: str,
    horizon_tick_delta: int,
) -> dict[str, object]:
    requested_action_counts = Counter(
        str(record.get("requested_action"))
        for record in world.tick_trajectory_records
        if isinstance(record.get("requested_action"), str)
    )
    dominant = _dominant_action_summary(requested_action_counts)
    target = world.agents.get(point.agent_id)
    target_alive = bool(target is not None and target.alive)
    tick_resource_gain = 0.0
    for record in world.tick_trajectory_records:
        outcome = _mapping(record.get("outcome"))
        value = outcome.get("resource_gain")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            tick_resource_gain += float(value)
    return {
        "horizon_tick_delta": int(horizon_tick_delta),
        "tick": int(world.tick),
        "branch_tick": point.branch_tick,
        "agent_id": point.agent_id,
        "forced_action": forced_action,
        "alive_agents": len(world.alive_agents()),
        "births": int(world.births),
        "deaths": int(world.deaths),
        "target_alive": target_alive,
        "target_energy_ratio": (
            _round(world._energy_ratio(target)) if target_alive else None
        ),
        "target_hydration_ratio": (
            _round(world._hydration_ratio(target)) if target_alive else None
        ),
        "target_health_ratio": (
            _round(world._health_ratio(target)) if target_alive else None
        ),
        "tick_resource_gain": _round(tick_resource_gain),
        "tick_trajectory_record_count": len(world.tick_trajectory_records),
        "tick_requested_action_counts": dict(sorted(requested_action_counts.items())),
        "tick_dominant_requested_action": dominant["action"],
        "tick_dominant_requested_action_share": dominant["share"],
    }


def _resolution_invalid_cause(*, action: str, observation_legal: bool) -> str:
    if action in MOVEMENT_ACTIONS or action.startswith("move_"):
        return (
            "resolution_invalid_occupancy_race"
            if observation_legal
            else "resolution_invalid_blocked_route"
        )
    if action in ("eat", "drink") and observation_legal:
        return "resolution_invalid_depleted_resource"
    return "resolution_invalid_other_public_path"


def _source_kind(path: str) -> str:
    name = Path(path).name
    if name.startswith("fixture-carrion-only-mind-v3"):
        return FIXTURE_SOURCE_KIND
    if name.startswith("open-mind-v3"):
        return OPEN_SOURCE_KIND
    return UNKNOWN_SOURCE_KIND


def _row_sort_key(row: Mapping[str, object]) -> tuple[int, int, int, int, str, str]:
    source = _source_kind(str(row.get("path", "")))
    priority = 0 if source == FIXTURE_SOURCE_KIND else 1 if source == OPEN_SOURCE_KIND else 2
    return (
        priority,
        _int(row.get("seed"), default=_seed_from_path(str(row.get("path", "")))),
        _int(row.get("recovery_tick")),
        _int(row.get("recovery_record_index")),
        str(row.get("requested_action", "")),
        str(row.get("path", "")),
    )


def _row_ticks(row: Mapping[str, object], path_metadata: Mapping[str, object]) -> int:
    path = str(row.get("path", ""))
    meta = _mapping(path_metadata.get(path))
    ticks = _int(meta.get("max_ticks"))
    if ticks > 0:
        return ticks
    return max(1, _int(row.get("recovery_tick")) + 1)


def _skip_row(row: Mapping[str, object], reason: str) -> dict[str, object]:
    return {**_target_row_excerpt(row), "reason": reason}


def _target_row_excerpt(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "source_path": row.get("path"),
        "source_kind": row.get("source_kind") or _source_kind(str(row.get("path", ""))),
        "seed": row.get("seed"),
        "ticks": row.get("ticks"),
        "gain_tick": row.get("gain_tick"),
        "gain_record_index": row.get("gain_record_index"),
        "recovery_tick": row.get("recovery_tick"),
        "recovery_record_index": row.get("recovery_record_index"),
        "agent_id": row.get("agent_id"),
        "requested_action": row.get("requested_action"),
        "observation_digest": row.get("observation_digest"),
    }


def _branch_point_payload(point: _BranchTarget) -> dict[str, object]:
    return {
        "branch_id": point.branch_id,
        "source_path": point.source_path,
        "source_kind": point.source_kind,
        "seed": point.seed,
        "ticks": point.ticks,
        "branch_tick": point.branch_tick,
        "record_index": point.record_index,
        "agent_id": point.agent_id,
        "logged_action": point.logged_action,
        "gain_tick": point.gain_tick,
        "gain_record_index": point.gain_record_index,
        "observation_schema": point.observation_schema,
        "observation_digest": point.observation_digest,
        "branch_state_digest": point.branch_state_digest,
        "action_mask": dict(point.action_mask),
        "resolution_action_mask": dict(point.resolution_action_mask),
        "before": dict(point.before),
    }


def _run_excerpt(run: Mapping[str, object] | None) -> dict[str, object] | None:
    if run is None:
        return None
    return {
        "forced_action": run.get("forced_action"),
        "forced_action_used": run.get("forced_action_used"),
        "forced_action_supported": run.get("forced_action_supported"),
        "alive_agents": run.get("alive_agents"),
        "births": run.get("births"),
        "deaths": run.get("deaths"),
        "target_alive_at_end": run.get("target_alive_at_end"),
        "target_recovery_score_at_end": run.get("target_recovery_score_at_end"),
        "first_action_outcome": run.get("first_action_outcome"),
        "heuristic_action_source_count": run.get("heuristic_action_source_count"),
        "diagnostic_forced_action_source_count": run.get(
            "diagnostic_forced_action_source_count"
        ),
        "replay_verification": run.get("replay_verification"),
    }


def _oracle_result_excerpt(result: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": result.get("branch_id"),
        "source_kind": result.get("source_kind"),
        "seed": result.get("seed"),
        "branch_tick": result.get("branch_tick"),
        "record_index": result.get("record_index"),
        "agent_id": result.get("agent_id"),
        "logged_action": result.get("logged_action"),
        "oracle_best_action": result.get("oracle_best_action"),
        "oracle_changed_action": result.get("oracle_changed_action"),
        "oracle_deltas_vs_logged": result.get("oracle_deltas_vs_logged"),
        "material_oracle_gain": result.get("material_oracle_gain"),
    }


def _digest_payload(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": run.get("branch_id"),
        "seed": run.get("seed"),
        "source_kind": run.get("source_kind"),
        "ticks": run.get("ticks"),
        "branch_tick": run.get("branch_tick"),
        "record_index": run.get("record_index"),
        "forced_action": run.get("forced_action"),
        "forced_action_used": run.get("forced_action_used"),
        "branch_state_digest": run.get("branch_state_digest"),
        "ticks_executed": run.get("ticks_executed"),
        "alive_agents": run.get("alive_agents"),
        "births": run.get("births"),
        "deaths": run.get("deaths"),
        "target_alive_at_end": run.get("target_alive_at_end"),
        "target_recovery_score_at_end": run.get("target_recovery_score_at_end"),
        "first_action_outcome": run.get("first_action_outcome"),
        "requested_action_counts": run.get("requested_action_counts"),
        "action_source_counts": run.get("action_source_counts"),
        "policy_id_counts": run.get("policy_id_counts"),
    }


def _run_for_action(
    runs: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next((run for run in runs if run.get("forced_action") == action), None)


def _target_recovery_score_from_values(values: Sequence[float | None]) -> float:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return 0.0
    return _round(sum(usable) / len(usable))


def _dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    if not counts:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return {"key": key, "count": int(count), "share": _share(count, sum(counts.values()))}


def _field_summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "mean": _round(sum(values) / len(values)),
    }


def _section_labels(section: Mapping[str, object]) -> list[str]:
    labels: list[str] = []
    for key in ("answer", "replay_answer", "distribution_answer", "resolution_answer"):
        value = section.get(key)
        if isinstance(value, str):
            labels.append(value)
    for value in _list(section.get("labels")):
        if isinstance(value, str):
            labels.append(value)
    return labels


def _diagnostic_answers(
    sections: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    answers: dict[str, object] = {}
    for name, section in sorted(sections.items()):
        for key in ("answer", "replay_answer", "distribution_answer", "resolution_answer"):
            if isinstance(section.get(key), str):
                answers[f"{name}.{key}"] = section.get(key)
    return answers


def _primary_label(labels: Sequence[str]) -> str:
    priority = [
        "missing_evidence_inconclusive",
        "first_recovery_branch_replay_failed",
        "first_recovery_branch_replay_partial",
        "heldout_branch_support_insufficient_for_v109",
        "first_recovery_oracle_action_distribution_collapsed",
        "oracle_supported_actions_not_observation_legal",
        "oracle_supported_actions_resolution_invalid",
        "first_recovery_oracle_improves_logged_action",
        "first_recovery_branch_replay_verified",
        "heldout_branch_support_sufficient_for_v109",
    ]
    for label in priority:
        if label in labels:
            return label
    return sorted(labels)[0]


def _missing_evidence(
    evidence: Mapping[str, object],
    sections: Mapping[str, Mapping[str, object]],
) -> list[str]:
    missing: list[str] = []
    reports = _mapping(evidence.get("reports"))
    for name, payload in sorted(reports.items()):
        report = _mapping(payload)
        if report.get("optional") is True:
            continue
        if report.get("loaded") is not True:
            missing.append(str(name))
        elif report.get("schema_matches") is False:
            missing.append(f"{name}_schema_mismatch")
    trajectories = _mapping(evidence.get("trajectories"))
    if _int(trajectories.get("loaded_path_count")) <= 0:
        missing.append("trajectories")
    if _int(trajectories.get("load_failure_count")) > 0:
        missing.append("trajectory_load_failures")
    alignment = _mapping(sections.get("v107_row_alignment"))
    if alignment.get("row_count_matches_v107") is not True:
        missing.append("v107_row_alignment_mismatch")
    selection = _mapping(sections.get("branch_target_selection"))
    if selection.get("setup_error") is not None:
        missing.append("policy_recreation_failed")
    return missing


def _alignment_mismatched(row_alignment: Mapping[str, object]) -> bool:
    return row_alignment.get("row_count_matches_v107") is not True


def _seed_run(report: Mapping[str, object] | None, *, seed: int) -> Mapping[str, object]:
    holdout = _mapping(_mapping(report or {}).get("holdout_evaluation"))
    for run in _list(holdout.get("runs")):
        if isinstance(run, Mapping) and _int(run.get("seed"), default=-1) == seed:
            return run
    return {}


def _branch_id(
    *,
    source_kind: str,
    seed: int,
    branch_index: int,
    tick: int,
    agent_id: int,
    action: str,
) -> str:
    return (
        f"first-recovery-{_safe_path_part(source_kind)}-seed-{int(seed)}-"
        f"branch-{int(branch_index)}-tick-{int(tick)}-agent-{int(agent_id)}-"
        f"logged-{_safe_path_part(action)}"
    )


def _seed_from_path(path: str) -> int:
    match = _SEED_PATTERN.search(path)
    return int(match.group(1)) if match is not None else -1


def _file_sha256(path: str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_counts(
    left: Mapping[str, object],
    right: Mapping[str, int],
) -> dict[str, int]:
    counts = Counter({str(key): _int(value) for key, value in left.items()})
    counts.update({str(key): int(value) for key, value in right.items()})
    return _counter_to_ordered_dict(counts)


def _counter_to_ordered_dict(counter: Mapping[str, int] | Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _bool_mapping(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): bool(item) for key, item in value.items()}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    return [item for item in _list(value) if isinstance(item, Mapping)]


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return int(value)


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FirstRecoveryBranchOracleAuditError(f"{field} must be a nonnegative integer")
    return int(value)


def _number(value: object) -> float:
    parsed = _number_or_none(value)
    return parsed if parsed is not None else 0.0


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _delta(candidate: object, baseline: object) -> float | None:
    candidate_number = _number_or_none(candidate)
    baseline_number = _number_or_none(baseline)
    if candidate_number is None or baseline_number is None:
        return None
    return _round(candidate_number - baseline_number)


def _share(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return _round(float(numerator) / float(denominator))


def _round(value: float) -> float:
    return round(float(value), 6)


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _safe_path_part(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
