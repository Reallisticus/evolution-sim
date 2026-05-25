from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES, MOVEMENT_ACTIONS
from evolution_sim.mind.first_recovery_branch_oracle_audit import (
    DEFAULT_DOMINANT_ORACLE_ACTION_SHARE_MAX,
    FIXTURE_SOURCE_KIND,
    MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_SCHEMA_VERSION,
    OPEN_SOURCE_KIND,
    UNKNOWN_SOURCE_KIND,
    FirstRecoveryBranchOracleAuditError,
    _BranchTarget,
    _alignment_mismatched,
    _best_oracle_run,
    _bool_mapping,
    _counter_to_ordered_dict,
    _evaluate_branch_target,
    _file_sha256,
    _int,
    _int_or_none,
    _legal_candidate_actions,
    _list,
    _list_of_mappings,
    _mapping,
    _materialize_branch_targets,
    _number,
    _number_or_none,
    _oracle_deltas,
    _optional_string,
    _report_evidence,
    _resolve_json_report,
    _round,
    _row_sort_key,
    _row_ticks,
    _run_for_action,
    _share,
    _source_kind,
    _target_row_excerpt,
    _v107_row_alignment,
)
from evolution_sim.mind.evolution import load_mind_v3_founder_template
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.transition_aligned_recovery_audit import (
    MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_SCHEMA_VERSION,
    reconstruct_transition_aligned_first_recovery_rows,
)

MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION = (
    "mind_v3_first_recovery_branch_archive_v1"
)
MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_POLICY = (
    "diagnostics_and_training_data_only_first_recovery_branch_archive_v1"
)
TRAINABLE_PUBLIC_INPUT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_branch_archive_trainable_public_input_v1"
)

DEFAULT_MAX_FIXTURE_TARGETS_PER_SEED = 3
DEFAULT_MIN_MATERIAL_GAIN_SEED_COUNT = 3

ALLOWED_CLASSIFICATION_LABELS: tuple[str, ...] = (
    "branch_archive_replay_verified",
    "branch_archive_replay_partial",
    "branch_archive_replay_failed",
    "archive_support_sufficient_for_shadow_scorer",
    "archive_support_insufficient_for_shadow_scorer",
    "open_rows_replay_supported",
    "open_rows_replay_unsupported",
    "open_rows_material_gain_present",
    "open_rows_material_gain_absent",
    "oracle_action_distribution_clean",
    "oracle_action_distribution_collapsed",
    "oracle_actions_observation_legal",
    "oracle_actions_not_observation_legal",
    "oracle_actions_resolution_legal",
    "oracle_actions_resolution_invalid",
    "seed29_archive_support_present",
    "seed29_archive_support_inconclusive",
    "material_gain_broad",
    "material_gain_sparse",
    "missing_evidence_inconclusive",
)
CLASSIFICATION_LABEL_SET = frozenset(ALLOWED_CLASSIFICATION_LABELS)

FORBIDDEN_TRAINABLE_KEYS = frozenset(
    {
        "seed",
        "fixture",
        "fixture_name",
        "source",
        "source_path",
        "source_kind",
        "branch",
        "branch_id",
        "branch_state_digest",
        "private_world_state",
        "world",
        "simulation_world",
        "logged_action",
        "logged_action_fallback",
        "logged_action_run",
        "agent_id",
        "record_index",
    }
)


class FirstRecoveryBranchArchiveError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FirstRecoveryBranchArchiveBuild:
    report: dict[str, object]
    archive_rows: tuple[dict[str, object], ...]


def load_first_recovery_branch_archive_json(path: str | Path) -> dict[str, object]:
    with _open_input(Path(path)) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise FirstRecoveryBranchArchiveError(f"report must be a JSON object: {path}")
    return payload


def write_first_recovery_branch_archive_outputs(
    build: FirstRecoveryBranchArchiveBuild,
    *,
    output_path: str | Path,
    archive_rows_path: str | Path | None = None,
) -> None:
    output = Path(output_path)
    rows_path = Path(archive_rows_path) if archive_rows_path is not None else (
        default_archive_rows_path(output)
    )
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl_gzip(build.archive_rows, rows_path)
    report = dict(build.report)
    report["archive_rows_path"] = str(rows_path)
    report["branch_archive_summary"] = {
        **_mapping(report.get("branch_archive_summary")),
        "archive_rows_sha256": _file_sha256(str(rows_path)),
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def default_archive_rows_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    name = output.name
    if name.endswith(".json"):
        return output.with_name(name[: -len(".json")] + ".jsonl.gz")
    return output.with_suffix(output.suffix + ".jsonl.gz")


def build_first_recovery_branch_archive(
    *,
    v108_report: Mapping[str, object] | None = None,
    v108_report_path: str | Path | None = None,
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
    max_fixture_targets_per_seed: int = DEFAULT_MAX_FIXTURE_TARGETS_PER_SEED,
    include_open: bool = True,
    exhaustive: bool = False,
    verify_replay: bool = True,
    archive_rows_path: str | Path | None = None,
    precomputed_branch_results: Sequence[Mapping[str, object]] | None = None,
) -> FirstRecoveryBranchArchiveBuild:
    target_limit = _nonnegative_int(
        max_fixture_targets_per_seed,
        field="max_fixture_targets_per_seed",
    )
    contract = _contract(
        max_fixture_targets_per_seed=target_limit,
        include_open=include_open,
        exhaustive=exhaustive,
        verify_replay=verify_replay,
    )
    loaded_reports = {
        "v108_report": _resolve_json_report(
            "v108_report",
            v108_report,
            v108_report_path,
            expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_SCHEMA_VERSION,
        ),
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
    source_reports = {
        name: loaded.evidence for name, loaded in sorted(loaded_reports.items())
    }
    evidence = {
        "source_reports": source_reports,
        "trajectories": reconstructed.get("evidence", {}),
    }
    row_reconstruction = _row_reconstruction(
        v107_report=loaded_reports["v107_report"].payload,
        v108_report=loaded_reports["v108_report"].payload,
        rows=rows,
        path_metadata=path_metadata,
    )
    target_selection = select_first_recovery_archive_targets(
        rows,
        path_metadata=path_metadata,
        max_fixture_targets_per_seed=target_limit,
        include_open=include_open,
        exhaustive=exhaustive,
    )
    selected_rows = [dict(row) for row in _list(target_selection.get("selected_rows"))]
    materialization_failures: list[dict[str, object]] = []
    branch_points: list[_BranchTarget] = []
    branch_results: list[dict[str, object]] = []
    setup_error: dict[str, object] | None = None
    can_branch = (
        precomputed_branch_results is None
        and selected_rows
        and row_reconstruction.get("row_count_matches_v107") is True
        and row_reconstruction.get("row_count_matches_v108") is True
    )
    if precomputed_branch_results is not None:
        branch_results = [dict(result) for result in precomputed_branch_results]
    elif can_branch:
        try:
            if rollout_context_report_path is None:
                raise FirstRecoveryBranchArchiveError(
                    "rollout_context_report_path is required to recreate v5 policy"
                )
            template = load_mind_v3_founder_template(rollout_context_report_path)
            branch_points, materialization_failures = _materialize_branch_targets(
                selected_rows,
                path_metadata=path_metadata,
                founder_template=template,
            )
            branch_results = [
                _evaluate_branch_target(point, verify_replay=verify_replay)
                for point in branch_points
            ]
        except Exception as exc:
            setup_error = {"reason": type(exc).__name__, "message": str(exc)}
    target_selection_public = dict(target_selection)
    target_selection_public.pop("selected_rows", None)
    if setup_error is not None:
        target_selection_public["setup_error"] = setup_error
    if row_reconstruction.get("row_count_matches_v107") is not True:
        target_selection_public["skipped_selected_target_count"] = len(selected_rows)
        target_selection_public["skip_reason_counts"] = _merge_counts(
            _mapping(target_selection_public.get("skip_reason_counts")),
            {"v107_row_alignment_mismatch": len(selected_rows)},
        )
    if row_reconstruction.get("row_count_matches_v108") is not True:
        target_selection_public["skipped_selected_target_count"] = len(selected_rows)
        target_selection_public["skip_reason_counts"] = _merge_counts(
            _mapping(target_selection_public.get("skip_reason_counts")),
            {"v108_row_alignment_mismatch": len(selected_rows)},
        )

    points_by_id = {point.branch_id: point for point in branch_points}
    archive_rows = tuple(
        build_first_recovery_archive_rows(
            branch_results,
            branch_points_by_id=points_by_id,
        )
    )
    branch_archive_summary = _branch_archive_summary(
        selected_row_count=len(selected_rows),
        branch_points=branch_points,
        materialization_failures=materialization_failures,
        branch_results=branch_results,
        archive_rows=archive_rows,
        exhaustive=exhaustive,
        reconstructed_row_count=len(rows),
        precomputed=precomputed_branch_results is not None,
    )
    split_metadata = _split_metadata(archive_rows)
    oracle_label_summary = _oracle_label_summary(branch_results, archive_rows)
    legality_summary = _legality_summary(archive_rows)
    resolution_drift_summary = _resolution_drift_summary(archive_rows)
    seed29_summary = _seed29_summary(archive_rows, branch_results)
    open_row_summary = _source_row_summary(
        rows=rows,
        selected_rows=selected_rows,
        branch_results=branch_results,
        source_kind=OPEN_SOURCE_KIND,
        archive_rows=archive_rows,
    )
    fixture_row_summary = _source_row_summary(
        rows=rows,
        selected_rows=selected_rows,
        branch_results=branch_results,
        source_kind=FIXTURE_SOURCE_KIND,
        archive_rows=archive_rows,
    )
    action_distribution = _action_distribution(archive_rows, branch_results)
    leakage = trainable_public_input_leakage(archive_rows)
    learnability_readiness = _learnability_readiness(
        row_reconstruction=row_reconstruction,
        target_selection=target_selection_public,
        branch_archive_summary=branch_archive_summary,
        split_metadata=split_metadata,
        oracle_label_summary=oracle_label_summary,
        legality_summary=legality_summary,
        open_row_summary=open_row_summary,
        seed29_summary=seed29_summary,
        leakage=leakage,
    )
    stop_rules = _stop_rules(
        row_reconstruction=row_reconstruction,
        branch_archive_summary=branch_archive_summary,
        oracle_label_summary=oracle_label_summary,
        legality_summary=legality_summary,
        open_row_summary=open_row_summary,
        seed29_summary=seed29_summary,
        leakage=leakage,
    )
    sections = {
        "row_reconstruction": row_reconstruction,
        "target_selection": target_selection_public,
        "branch_archive_summary": branch_archive_summary,
        "oracle_label_summary": oracle_label_summary,
        "legality_summary": legality_summary,
        "resolution_drift_summary": resolution_drift_summary,
        "seed29_summary": seed29_summary,
        "open_row_summary": open_row_summary,
        "fixture_row_summary": fixture_row_summary,
        "learnability_readiness": learnability_readiness,
    }
    classification = _classification(evidence=evidence, sections=sections)
    recommendation = _research_recommendation(
        classification=classification,
        learnability_readiness=learnability_readiness,
        stop_rules=stop_rules,
    )
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
        "archive_policy": MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_POLICY,
        "contract": contract,
        "provenance": _provenance(
            contract=contract,
            loaded_reports=loaded_reports,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=trajectory_glob_patterns,
            archive_rows_path=archive_rows_path,
        ),
        "source_reports": source_reports,
        "row_reconstruction": row_reconstruction,
        "target_selection": target_selection_public,
        "branch_archive_summary": branch_archive_summary,
        "archive_rows_path": str(archive_rows_path) if archive_rows_path else None,
        "split_metadata": split_metadata,
        "oracle_label_summary": oracle_label_summary,
        "legality_summary": legality_summary,
        "resolution_drift_summary": resolution_drift_summary,
        "seed29_summary": seed29_summary,
        "open_row_summary": open_row_summary,
        "fixture_row_summary": fixture_row_summary,
        "action_distribution": action_distribution,
        "learnability_readiness": learnability_readiness,
        "stop_rules": stop_rules,
        "research_recommendation": recommendation,
        "classification": classification,
        "non_promoted": True,
    }
    return FirstRecoveryBranchArchiveBuild(report=report, archive_rows=archive_rows)


def select_first_recovery_archive_targets(
    rows: Sequence[Mapping[str, object]],
    *,
    path_metadata: Mapping[str, object],
    max_fixture_targets_per_seed: int = DEFAULT_MAX_FIXTURE_TARGETS_PER_SEED,
    include_open: bool = True,
    exhaustive: bool = False,
) -> dict[str, object]:
    sorted_rows = [dict(row) for row in sorted(rows, key=_row_sort_key)]
    for row in sorted_rows:
        path = str(row.get("path", ""))
        row["source_kind"] = _source_kind(path)
        row["ticks"] = _row_ticks(row, path_metadata)
    selected: list[dict[str, object]] = []
    selected_keys: set[tuple[str, int]] = set()
    selected_reasons: dict[tuple[str, int], str] = {}
    skipped: list[dict[str, object]] = []
    skipped_keys: set[tuple[str, int]] = set()
    fixture_counts: Counter[int] = Counter()

    def key_for(row: Mapping[str, object]) -> tuple[str, int]:
        return (
            str(row.get("path", "")),
            _int(row.get("recovery_record_index"), default=-1),
        )

    def add(
        row: Mapping[str, object],
        reason: str,
        *,
        allow_fixture_over_limit: bool = False,
    ) -> bool:
        key = key_for(row)
        if key in selected_keys:
            return False
        source = str(row.get("source_kind"))
        seed = _int(row.get("seed"))
        if (
            source == FIXTURE_SOURCE_KIND
            and not exhaustive
            and not allow_fixture_over_limit
            and fixture_counts[seed] >= max_fixture_targets_per_seed
        ):
            return False
        selected.append(dict(row))
        selected_keys.add(key)
        selected_reasons[key] = reason
        if source == FIXTURE_SOURCE_KIND:
            fixture_counts[seed] += 1
        return True

    for row in sorted_rows:
        source = str(row.get("source_kind"))
        if source == UNKNOWN_SOURCE_KIND:
            skipped.append(_skip_row(row, "unknown_source_kind"))
            skipped_keys.add(key_for(row))
            continue
        if exhaustive:
            add(row, "exhaustive_all_rows")
            continue
        if source == OPEN_SOURCE_KIND:
            if include_open:
                add(row, "all_open_rows")
            else:
                skipped.append(_skip_row(row, "open_rows_skipped_by_option"))
                skipped_keys.add(key_for(row))
    if not exhaustive:
        fixture_rows = [row for row in sorted_rows if row.get("source_kind") == FIXTURE_SOURCE_KIND]
        fixture_seeds = sorted({_int(row.get("seed")) for row in fixture_rows})
        for seed in fixture_seeds:
            group = [row for row in fixture_rows if _int(row.get("seed")) == seed]
            if not group:
                continue
            add(group[0], "fixture_seed_coverage")
            first = group[0]
            for row in group[1:]:
                if (
                    row.get("requested_action") != first.get("requested_action")
                    or row.get("recovery_tick") != first.get("recovery_tick")
                    or row.get("agent_id") != first.get("agent_id")
                ):
                    if add(row, "fixture_seed_second_row"):
                        break
        selected_actions = {str(row.get("requested_action")) for row in selected}
        all_actions = sorted({str(row.get("requested_action")) for row in sorted_rows})
        for action in all_actions:
            if action in selected_actions:
                continue
            candidates = [
                row
                for row in fixture_rows
                if str(row.get("requested_action")) == action
                and key_for(row) not in selected_keys
            ]
            candidates.sort(
                key=lambda row: (
                    fixture_counts[_int(row.get("seed"))],
                    _int(row.get("seed")),
                    _int(row.get("recovery_tick")),
                    _int(row.get("recovery_record_index")),
                )
            )
            if candidates:
                if not add(candidates[0], "logged_action_coverage"):
                    add(
                        candidates[0],
                        "logged_action_coverage_over_limit",
                        allow_fixture_over_limit=True,
                    )
                selected_actions.add(action)
        for seed in fixture_seeds:
            group = [row for row in fixture_rows if _int(row.get("seed")) == seed]
            while fixture_counts[seed] < max_fixture_targets_per_seed:
                chosen = _next_diverse_fixture_row(group, selected, selected_keys, key_for)
                if chosen is None:
                    break
                add(chosen, "fixture_seed_diversity_fill")
    for row in sorted_rows:
        row_key = key_for(row)
        if row_key in selected_keys or row_key in skipped_keys:
            continue
        source = str(row.get("source_kind"))
        if source == UNKNOWN_SOURCE_KIND:
            continue
        if source == OPEN_SOURCE_KIND and include_open:
            skipped.append(_skip_row(row, "not_selected_unexpected_open_gap"))
        elif source == FIXTURE_SOURCE_KIND and not exhaustive:
            skipped.append(_skip_row(row, "stratified_fixture_limit"))
        elif source == OPEN_SOURCE_KIND:
            skipped.append(_skip_row(row, "open_rows_skipped_by_option"))
    selected.sort(key=_row_sort_key)
    skipped.sort(key=lambda row: (
        str(row.get("reason")),
        str(row.get("source_path")),
        _int(row.get("recovery_record_index"), default=-1),
    ))
    by_source = Counter(str(row.get("source_kind")) for row in selected)
    by_seed = Counter(str(row.get("seed")) for row in selected)
    by_seed_source = Counter(
        f"{row.get('source_kind')}:{row.get('seed')}" for row in selected
    )
    by_action = Counter(str(row.get("requested_action")) for row in selected)
    selected_ticks = {
        f"{row.get('source_kind')}:{row.get('seed')}": sorted(
            {
                _int(item.get("recovery_tick"))
                for item in selected
                if item.get("source_kind") == row.get("source_kind")
                and item.get("seed") == row.get("seed")
            }
        )
        for row in selected
    }
    selected_agents = {
        f"{row.get('source_kind')}:{row.get('seed')}": sorted(
            {
                _int(item.get("agent_id"))
                for item in selected
                if item.get("source_kind") == row.get("source_kind")
                and item.get("seed") == row.get("seed")
            }
        )
        for row in selected
    }
    skip_counts = Counter(str(row.get("reason")) for row in skipped)
    reason_counts = Counter(selected_reasons.values())
    open_candidate_count = sum(1 for row in sorted_rows if row.get("source_kind") == OPEN_SOURCE_KIND)
    open_selected_count = sum(1 for row in selected if row.get("source_kind") == OPEN_SOURCE_KIND)
    return {
        "answer": "branch_archive_replay_partial",
        "selection_policy": (
            "all_first_recovery_rows_exhaustive_v1"
            if exhaustive
            else "all_open_plus_stratified_fixture_recovery_v1"
        ),
        "exhaustive": bool(exhaustive),
        "include_open": bool(include_open),
        "max_fixture_targets_per_seed": int(max_fixture_targets_per_seed),
        "candidate_row_count": len(sorted_rows),
        "selected_target_count": len(selected),
        "skipped_target_count": len(skipped),
        "selected_rows": selected,
        "selected_targets": [
            {**_target_row_excerpt(row), "selection_reason": selected_reasons[key_for(row)]}
            for row in selected
        ],
        "selected_by_source": _counter_to_ordered_dict(by_source),
        "selected_by_seed": _counter_to_ordered_dict(by_seed),
        "selected_by_seed_source": _counter_to_ordered_dict(by_seed_source),
        "selected_by_logged_action": _counter_to_ordered_dict(by_action),
        "selection_reason_counts": _counter_to_ordered_dict(reason_counts),
        "skipped_examples": skipped[:32],
        "skip_reason_counts": _counter_to_ordered_dict(skip_counts),
        "open_candidate_row_count": open_candidate_count,
        "open_selected_row_count": open_selected_count,
        "all_open_rows_selected": (
            bool(include_open) and open_candidate_count == open_selected_count
        ),
        "fixture_seed_coverage": sorted(
            {
                _int(row.get("seed"))
                for row in selected
                if row.get("source_kind") == FIXTURE_SOURCE_KIND
            }
        ),
        "seed29_selected": any(_int(row.get("seed")) == 29 for row in selected),
        "selected_ticks_by_seed_source": selected_ticks,
        "selected_agents_by_seed_source": selected_agents,
    }


def build_first_recovery_archive_rows(
    branch_results: Sequence[Mapping[str, object]],
    *,
    branch_points_by_id: Mapping[str, _BranchTarget] | None = None,
) -> list[dict[str, object]]:
    points = branch_points_by_id or {}
    rows: list[dict[str, object]] = []
    for result in sorted(branch_results, key=_branch_result_sort_key):
        point = points.get(str(result.get("branch_id", "")))
        action_runs = _list_of_mappings(result.get("action_runs"))
        logged = _run_for_action(action_runs, str(result.get("logged_action", "")))
        ranks = _oracle_ranks(action_runs)
        for run in sorted(action_runs, key=lambda item: str(item.get("forced_action", ""))):
            rows.append(
                _archive_row_from_run(
                    result,
                    run,
                    logged=logged,
                    oracle_rank=ranks.get(str(run.get("forced_action")), 0),
                    point=point,
                )
            )
    return rows


def trainable_public_input_leakage(
    archive_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    leaks: list[dict[str, object]] = []
    for index, row in enumerate(archive_rows):
        trainable = _mapping(row.get("trainable_public_input"))
        for path in _forbidden_trainable_paths(trainable):
            leaks.append(
                {
                    "archive_row_index": index,
                    "branch_id": _mapping(row.get("provenance")).get("branch_id"),
                    "path": path,
                }
            )
            if len(leaks) >= 24:
                break
        if len(leaks) >= 24:
            break
    return {
        "leakage_detected": bool(leaks),
        "leak_count": len(leaks),
        "forbidden_keys": sorted(FORBIDDEN_TRAINABLE_KEYS),
        "examples": leaks,
    }


def _row_reconstruction(
    *,
    v107_report: Mapping[str, object] | None,
    v108_report: Mapping[str, object] | None,
    rows: Sequence[Mapping[str, object]],
    path_metadata: Mapping[str, object],
) -> dict[str, object]:
    v107 = _v107_row_alignment(
        v107_report=v107_report,
        rows=rows,
        path_metadata=path_metadata,
    )
    v108_alignment = _mapping(_mapping(v108_report or {}).get("v107_row_alignment"))
    expected_v108 = _int_or_none(
        v108_alignment.get("reconstructed_first_recovery_row_count")
    )
    v108_matches = (
        expected_v108 is not None
        and expected_v108 == len(rows)
        and v108_alignment.get("row_count_matches_v107") is True
    )
    mismatches = list(_list(v107.get("mismatches")))
    if expected_v108 is None:
        mismatches.append(
            {
                "field": "v108_report.v107_row_alignment.reconstructed_first_recovery_row_count",
                "reason": "missing_expected_v108_count",
            }
        )
    elif expected_v108 != len(rows):
        mismatches.append(
            {
                "field": "v108_report.v107_row_alignment.reconstructed_first_recovery_row_count",
                "expected": expected_v108,
                "observed": len(rows),
                "reason": "reconstructed_row_count_mismatch_v108",
            }
        )
    answer = "branch_archive_replay_partial" if not mismatches else "missing_evidence_inconclusive"
    by_logged_action = Counter(str(row.get("requested_action")) for row in rows)
    payload = {
        **v107,
        "answer": answer,
        "expected_v108_reconstructed_first_recovery_row_count": expected_v108,
        "row_count_matches_v108": bool(v108_matches),
        "mismatch_count": len(mismatches),
        "mismatches": [dict(item) for item in mismatches],
        "by_logged_action": _counter_to_ordered_dict(by_logged_action),
    }
    return payload


def _next_diverse_fixture_row(
    group: Sequence[Mapping[str, object]],
    selected: Sequence[Mapping[str, object]],
    selected_keys: set[tuple[str, int]],
    key_for: object,
) -> Mapping[str, object] | None:
    key_func = key_for
    selected_for_seed = [
        row
        for row in selected
        if row.get("source_kind") == FIXTURE_SOURCE_KIND
        and row.get("seed") == (group[0].get("seed") if group else None)
    ]
    actions = {str(row.get("requested_action")) for row in selected_for_seed}
    ticks = {_int(row.get("recovery_tick")) for row in selected_for_seed}
    agents = {_int(row.get("agent_id")) for row in selected_for_seed}
    candidates = [row for row in group if key_func(row) not in selected_keys]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            -int(str(row.get("requested_action")) not in actions),
            -int(_int(row.get("recovery_tick")) not in ticks),
            -int(_int(row.get("agent_id")) not in agents),
            _int(row.get("recovery_tick")),
            _int(row.get("recovery_record_index")),
        )
    )
    return candidates[0]


def _archive_row_from_run(
    result: Mapping[str, object],
    run: Mapping[str, object],
    *,
    logged: Mapping[str, object] | None,
    oracle_rank: int,
    point: _BranchTarget | None,
) -> dict[str, object]:
    first = _mapping(run.get("first_action_outcome"))
    forced_action = str(run.get("forced_action", ""))
    candidate_deltas = _oracle_deltas(run, logged)
    material_gain = _material_gain_from_deltas(candidate_deltas)
    observation_legal = bool(first.get("observation_legal", False))
    resolution_legal = bool(first.get("resolution_legal", False))
    action_mask = dict(point.action_mask) if point is not None else _bool_mapping(
        result.get("action_mask")
    )
    resolution_mask = (
        dict(point.resolution_action_mask)
        if point is not None
        else _bool_mapping(result.get("resolution_action_mask"))
    )
    before = dict(point.before) if point is not None else dict(_mapping(result.get("before")))
    branch_tick = _int(result.get("branch_tick"))
    gain_tick = _int_or_none(result.get("gain_tick"))
    replay = _mapping(run.get("replay_verification"))
    biology = _biology_labels(
        first,
        run=run,
        candidate_deltas=candidate_deltas,
    )
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
        "archive_row_id": _archive_row_id(result, forced_action),
        "source_path": result.get("source_path"),
        "source_kind": result.get("source_kind"),
        "seed": result.get("seed"),
        "tick": branch_tick,
        "agent_id": result.get("agent_id"),
        "record_index": result.get("record_index"),
        "observation_digest": result.get("observation_digest"),
        "action_mask": action_mask,
        "resolution_action_mask": resolution_mask,
        "candidate_action": forced_action,
        "forced_first_action_diagnostic_source": f"branch_oracle_force:{forced_action}",
        "replay_verification_digest": replay.get("expected_digest"),
        "replay_verification_result": replay.get("verified"),
        "first_action_outcome": dict(first),
        "recovery_vitals_deltas": {
            "energy_ratio_delta": first.get("energy_ratio_delta"),
            "hydration_ratio_delta": first.get("hydration_ratio_delta"),
            "health_ratio_delta": first.get("health_ratio_delta"),
            "target_recovery_score_delta": candidate_deltas.get(
                "target_recovery_score_delta"
            ),
        },
        "terminal_alive_delta": candidate_deltas.get("terminal_alive_delta"),
        "birth_delta": candidate_deltas.get("birth_delta"),
        "death_delta": (
            -_number(candidate_deltas.get("death_reduction_delta"))
            if candidate_deltas.get("death_reduction_delta") is not None
            else None
        ),
        "death_reduction_delta": candidate_deltas.get("death_reduction_delta"),
        "material_gain_label": bool(material_gain),
        "oracle_rank": int(oracle_rank),
        "oracle_best_action": result.get("oracle_best_action"),
        "observation_legal": observation_legal,
        "resolution_legal": resolution_legal,
        "resolution_invalid_public_cause": (
            _archive_resolution_invalid_cause(forced_action, observation_legal)
            if not resolution_legal
            else None
        ),
        "biology_homeostasis_labels": biology,
        "behavior_descriptors": _behavior_descriptors(
            forced_action=forced_action,
            first=first,
            run=run,
            candidate_deltas=candidate_deltas,
        ),
        "trainable_public_input": _trainable_public_input(
            before=before,
            action_mask=action_mask,
            candidate_action=forced_action,
            branch_tick=branch_tick,
            gain_tick=gain_tick,
            gain_record_index=_int_or_none(result.get("gain_record_index")),
            record_index=_int_or_none(result.get("record_index")),
        ),
        "provenance": {
            "branch_id": result.get("branch_id"),
            "source_path": result.get("source_path"),
            "source_kind": result.get("source_kind"),
            "seed": result.get("seed"),
            "agent_id": result.get("agent_id"),
            "record_index": result.get("record_index"),
            "gain_tick": result.get("gain_tick"),
            "gain_record_index": result.get("gain_record_index"),
            "logged_action": result.get("logged_action"),
            "branch_state_digest": result.get("branch_state_digest"),
            "diagnostics_only": True,
        },
    }


def _trainable_public_input(
    *,
    before: Mapping[str, object],
    action_mask: Mapping[str, object],
    candidate_action: str,
    branch_tick: int,
    gain_tick: int | None,
    gain_record_index: int | None,
    record_index: int | None,
) -> dict[str, object]:
    return {
        "schema_version": TRAINABLE_PUBLIC_INPUT_SCHEMA_VERSION,
        "post_carrion_first_recovery": True,
        "candidate_action": candidate_action,
        "candidate_action_index": ACTION_NAMES.index(candidate_action)
        if candidate_action in ACTION_NAMES
        else None,
        "action_mask": {str(action): bool(action_mask.get(action, False)) for action in ACTION_NAMES},
        "target_public_state_before": {
            "alive": bool(before.get("alive", False)),
            "energy_ratio": _number_or_none(before.get("energy_ratio")),
            "hydration_ratio": _number_or_none(before.get("hydration_ratio")),
            "health_ratio": _number_or_none(before.get("health_ratio")),
            "age": _int_or_none(before.get("age")),
        },
        "public_transition_context": {
            "ticks_after_animal_resource_gain": (
                branch_tick - gain_tick if gain_tick is not None else None
            ),
            "records_after_animal_resource_gain": (
                record_index - gain_record_index
                if record_index is not None and gain_record_index is not None
                else None
            ),
        },
    }


def _biology_labels(
    first: Mapping[str, object],
    *,
    run: Mapping[str, object],
    candidate_deltas: Mapping[str, object],
) -> dict[str, object]:
    energy_delta = _number_or_none(first.get("energy_ratio_delta"))
    hydration_delta = _number_or_none(first.get("hydration_ratio_delta"))
    health_delta = _number_or_none(first.get("health_ratio_delta"))
    return {
        "energy_debt_delta": _round(-energy_delta) if energy_delta is not None else None,
        "hydration_debt_delta": (
            _round(-hydration_delta) if hydration_delta is not None else None
        ),
        "health_injury_delta": health_delta,
        "reproduction_readiness_delta": candidate_deltas.get("birth_delta"),
        "target_alive": bool(run.get("target_alive_at_end", False)),
        "target_recovery_score": _number_or_none(
            run.get("target_recovery_score_at_end")
        ),
        "population_alive_delta": candidate_deltas.get("terminal_alive_delta"),
        "birth_delta": candidate_deltas.get("birth_delta"),
        "death_delta": (
            -_number(candidate_deltas.get("death_reduction_delta"))
            if candidate_deltas.get("death_reduction_delta") is not None
            else None
        ),
    }


def _behavior_descriptors(
    *,
    forced_action: str,
    first: Mapping[str, object],
    run: Mapping[str, object],
    candidate_deltas: Mapping[str, object],
) -> list[str]:
    descriptors = ["post_carrion_recovery"]
    if forced_action == "eat":
        descriptors.append("eat_loop")
    if forced_action in MOVEMENT_ACTIONS or forced_action.startswith("move_"):
        if first.get("moved") is True:
            descriptors.append("patch_leave")
        else:
            descriptors.append("failed_movement")
    if forced_action == "drink" and (
        first.get("drank") is True or _number(first.get("hydration_ratio_delta")) > 0.0
    ):
        descriptors.append("drink_recovery")
    if bool(first.get("observation_legal")) and not bool(first.get("resolution_legal")):
        descriptors.append("resolution_drift")
    cause = _archive_resolution_invalid_cause(
        forced_action,
        bool(first.get("observation_legal")),
    )
    if not bool(first.get("resolution_legal")) and cause == "resolution_invalid_depleted_resource":
        descriptors.append("resource_depletion")
    if (
        _number(candidate_deltas.get("birth_delta")) > 0.0
        or _number(candidate_deltas.get("target_recovery_score_delta")) > 0.0
    ):
        descriptors.append("delayed_reproduction_readiness")
    counts = _mapping(run.get("requested_action_counts"))
    total = sum(_int(value) for value in counts.values())
    if forced_action == "eat" and total and _share(_int(counts.get("eat")), total) > 0.5:
        if "eat_loop" not in descriptors:
            descriptors.append("eat_loop")
    return sorted(set(descriptors))


def _branch_archive_summary(
    *,
    selected_row_count: int,
    branch_points: Sequence[_BranchTarget],
    materialization_failures: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    archive_rows: Sequence[Mapping[str, object]],
    exhaustive: bool,
    reconstructed_row_count: int,
    precomputed: bool,
) -> dict[str, object]:
    action_runs = [
        run
        for result in branch_results
        for run in _list_of_mappings(result.get("action_runs"))
    ]
    replay_items = [
        _mapping(run.get("replay_verification"))
        for run in action_runs
        if _mapping(run.get("replay_verification"))
    ]
    replay_verified = bool(replay_items) and all(
        item.get("verified") is True for item in replay_items
    )
    if materialization_failures or any(item.get("verified") is False for item in replay_items):
        answer = "branch_archive_replay_failed"
    elif replay_verified and exhaustive and selected_row_count == reconstructed_row_count:
        answer = "branch_archive_replay_verified"
    elif branch_results:
        answer = "branch_archive_replay_partial"
    else:
        answer = "missing_evidence_inconclusive"
    heuristic_count = sum(_int(run.get("heuristic_action_source_count")) for run in action_runs)
    forced_count = sum(
        _int(run.get("diagnostic_forced_action_source_count")) for run in action_runs
    )
    return {
        "answer": answer,
        "selected_row_count": int(selected_row_count),
        "materialized_branch_point_count": len(branch_points)
        if not precomputed
        else len(branch_results),
        "materialization_failure_count": len(materialization_failures),
        "materialization_failures": [dict(item) for item in materialization_failures[:24]],
        "branch_result_count": len(branch_results),
        "action_run_count": len(action_runs),
        "archive_row_count": len(archive_rows),
        "replay_verification_count": len(replay_items),
        "replay_verified": replay_verified,
        "heuristic_action_source_count": heuristic_count,
        "diagnostic_forced_action_source_count": forced_count,
        "zero_heuristic_runtime_actions_except_diagnostic_force": heuristic_count == 0,
        "full_reconstruction_row_count": int(reconstructed_row_count),
        "exhaustive": bool(exhaustive),
        "precomputed_branch_results": bool(precomputed),
        "private_world_state_serialized": False,
    }


def _split_metadata(archive_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    seeds = sorted(
        {
            _int(_mapping(row.get("provenance")).get("seed"), default=-1)
            for row in archive_rows
            if _int(_mapping(row.get("provenance")).get("seed"), default=-1) >= 0
        }
    )
    sources = sorted(
        {
            str(_mapping(row.get("provenance")).get("source_kind"))
            for row in archive_rows
            if _mapping(row.get("provenance")).get("source_kind") is not None
        }
    )
    by_seed = Counter(
        str(_mapping(row.get("provenance")).get("seed")) for row in archive_rows
    )
    by_source = Counter(
        str(_mapping(row.get("provenance")).get("source_kind")) for row in archive_rows
    )
    return {
        "split_policy": "leave_one_seed_and_leave_one_source_public_archive_v1",
        "leave_one_seed": [
            {
                "holdout_seed": seed,
                "holdout_archive_row_count": by_seed[str(seed)],
                "train_archive_row_count": len(archive_rows) - by_seed[str(seed)],
            }
            for seed in seeds
        ],
        "leave_one_source": [
            {
                "holdout_source_kind": source,
                "holdout_archive_row_count": by_source[source],
                "train_archive_row_count": len(archive_rows) - by_source[source],
            }
            for source in sources
        ],
        "leave_one_seed_viable": len(seeds) >= 2
        and all(by_seed[str(seed)] > 0 for seed in seeds),
        "leave_one_source_viable": len(sources) >= 2
        and all(by_source[source] > 0 for source in sources),
        "seed_count": len(seeds),
        "source_count": len(sources),
        "archive_rows_by_seed": _counter_to_ordered_dict(by_seed),
        "archive_rows_by_source": _counter_to_ordered_dict(by_source),
    }


def _oracle_label_summary(
    branch_results: Sequence[Mapping[str, object]],
    archive_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    oracle_actions = [
        str(result.get("oracle_best_action"))
        for result in branch_results
        if result.get("oracle_best_action") is not None
    ]
    action_counts = Counter(oracle_actions)
    dominant = _dominant_count_share(action_counts)
    material_rows = [row for row in archive_rows if row.get("material_gain_label") is True]
    material_branches = [
        result for result in branch_results if result.get("material_oracle_gain") is True
    ]
    material_seed_count = len(
        {_int(result.get("seed"), default=-1) for result in material_branches}
    )
    material_answer = (
        "material_gain_broad"
        if material_seed_count >= DEFAULT_MIN_MATERIAL_GAIN_SEED_COUNT
        else "material_gain_sparse"
    )
    distribution_answer = (
        "oracle_action_distribution_collapsed"
        if _number(dominant.get("share")) > DEFAULT_DOMINANT_ORACLE_ACTION_SHARE_MAX
        else "oracle_action_distribution_clean"
    )
    return {
        "answer": material_answer,
        "distribution_answer": distribution_answer,
        "oracle_action_counts": _counter_to_ordered_dict(action_counts),
        "dominant_oracle_action": dominant["key"],
        "dominant_oracle_action_count": dominant["count"],
        "dominant_oracle_action_share": dominant["share"],
        "oracle_action_entropy": _entropy(action_counts),
        "branch_result_count": len(branch_results),
        "archive_row_count": len(archive_rows),
        "material_gain_archive_row_count": len(material_rows),
        "material_gain_branch_count": len(material_branches),
        "material_gain_seed_count": material_seed_count,
    }


def _legality_summary(archive_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    observation_counts = Counter()
    resolution_counts = Counter()
    for row in archive_rows:
        observation_counts[
            "oracle_actions_observation_legal"
            if row.get("observation_legal") is True
            else "oracle_actions_not_observation_legal"
        ] += 1
        resolution_counts[
            "oracle_actions_resolution_legal"
            if row.get("resolution_legal") is True
            else "oracle_actions_resolution_invalid"
        ] += 1
    answer = (
        "oracle_actions_not_observation_legal"
        if observation_counts["oracle_actions_not_observation_legal"] > 0
        else (
            "oracle_actions_observation_legal"
            if observation_counts["oracle_actions_observation_legal"] > 0
            else "missing_evidence_inconclusive"
        )
    )
    resolution_answer = (
        "oracle_actions_resolution_invalid"
        if resolution_counts["oracle_actions_resolution_invalid"] > 0
        else (
            "oracle_actions_resolution_legal"
            if resolution_counts["oracle_actions_resolution_legal"] > 0
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
        "archive_row_count": len(archive_rows),
        "observation_category_counts": _counter_to_ordered_dict(observation_counts),
        "resolution_category_counts": _counter_to_ordered_dict(resolution_counts),
        "observation_legal_share": _share(
            observation_counts["oracle_actions_observation_legal"],
            len(archive_rows),
        ),
        "resolution_invalid_count": resolution_counts[
            "oracle_actions_resolution_invalid"
        ],
    }


def _resolution_drift_summary(
    archive_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts = Counter(
        str(row.get("resolution_invalid_public_cause"))
        for row in archive_rows
        if row.get("resolution_invalid_public_cause") is not None
    )
    return {
        "invalid_archive_row_count": sum(counts.values()),
        "category_counts": _counter_to_ordered_dict(counts),
        "examples": [
            {
                "archive_row_id": row.get("archive_row_id"),
                "candidate_action": row.get("candidate_action"),
                "cause": row.get("resolution_invalid_public_cause"),
                "first_action_outcome": row.get("first_action_outcome"),
            }
            for row in archive_rows
            if row.get("resolution_invalid_public_cause") is not None
        ][:16],
    }


def _seed29_summary(
    archive_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = [
        row
        for row in archive_rows
        if _int(_mapping(row.get("provenance")).get("seed"), default=-1) == 29
    ]
    branch_count = sum(1 for result in branch_results if _int(result.get("seed")) == 29)
    answer = (
        "seed29_archive_support_present"
        if rows and branch_count > 0
        else "seed29_archive_support_inconclusive"
    )
    return {
        "answer": answer,
        "seed": 29,
        "archive_row_count": len(rows),
        "branch_result_count": branch_count,
        "material_gain_archive_row_count": sum(
            1 for row in rows if row.get("material_gain_label") is True
        ),
        "candidate_action_counts": _counter_to_ordered_dict(
            Counter(str(row.get("candidate_action")) for row in rows)
        ),
        "examples": [_archive_row_excerpt(row) for row in rows[:16]],
    }


def _source_row_summary(
    *,
    rows: Sequence[Mapping[str, object]],
    selected_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    source_kind: str,
    archive_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    source_rows = [row for row in rows if _source_kind(str(row.get("path", ""))) == source_kind]
    source_selected = [
        row for row in selected_rows if _source_kind(str(row.get("path", ""))) == source_kind
    ]
    source_branch_results = [
        result for result in branch_results if result.get("source_kind") == source_kind
    ]
    source_archive_rows = [
        row for row in archive_rows if row.get("source_kind") == source_kind
    ]
    if source_kind == OPEN_SOURCE_KIND:
        replay_answer = (
            "open_rows_replay_supported"
            if source_rows and len(source_branch_results) == len(source_selected) and source_selected
            else "open_rows_replay_unsupported"
        )
        material_answer = (
            "open_rows_material_gain_present"
            if any(row.get("material_gain_label") is True for row in source_archive_rows)
            else "open_rows_material_gain_absent"
        )
    else:
        replay_answer = "branch_archive_replay_partial"
        material_answer = None
    payload = {
        "source_kind": source_kind,
        "candidate_row_count": len(source_rows),
        "selected_row_count": len(source_selected),
        "branch_result_count": len(source_branch_results),
        "archive_row_count": len(source_archive_rows),
        "by_seed": _counter_to_ordered_dict(
            Counter(str(row.get("seed")) for row in source_rows)
        ),
        "selected_by_seed": _counter_to_ordered_dict(
            Counter(str(row.get("seed")) for row in source_selected)
        ),
        "candidate_action_counts": _counter_to_ordered_dict(
            Counter(str(row.get("candidate_action")) for row in source_archive_rows)
        ),
    }
    if replay_answer is not None:
        payload["answer"] = replay_answer
    if material_answer is not None:
        payload["material_answer"] = material_answer
    return payload


def _action_distribution(
    archive_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidate_counts = Counter(str(row.get("candidate_action")) for row in archive_rows)
    logged_counts = Counter(str(result.get("logged_action")) for result in branch_results)
    oracle_counts = Counter(
        str(result.get("oracle_best_action"))
        for result in branch_results
        if result.get("oracle_best_action") is not None
    )
    return {
        "candidate_action_counts": _counter_to_ordered_dict(candidate_counts),
        "logged_action_counts": _counter_to_ordered_dict(logged_counts),
        "oracle_action_counts": _counter_to_ordered_dict(oracle_counts),
        "candidate_action_entropy": _entropy(candidate_counts),
        "oracle_action_entropy": _entropy(oracle_counts),
    }


def _learnability_readiness(
    *,
    row_reconstruction: Mapping[str, object],
    target_selection: Mapping[str, object],
    branch_archive_summary: Mapping[str, object],
    split_metadata: Mapping[str, object],
    oracle_label_summary: Mapping[str, object],
    legality_summary: Mapping[str, object],
    open_row_summary: Mapping[str, object],
    seed29_summary: Mapping[str, object],
    leakage: Mapping[str, object],
) -> dict[str, object]:
    checks = {
        "row_alignment": row_reconstruction.get("row_count_matches_v107") is True
        and row_reconstruction.get("row_count_matches_v108") is True,
        "coverage_by_seed": len(_list(target_selection.get("fixture_seed_coverage"))) >= 6,
        "coverage_by_source": _int(open_row_summary.get("selected_row_count")) > 0
        and _int(_mapping(target_selection.get("selected_by_source")).get(FIXTURE_SOURCE_KIND))
        > 0,
        "coverage_by_action": len(_mapping(target_selection.get("selected_by_logged_action")))
        >= 4,
        "oracle_entropy_present": _number(oracle_label_summary.get("oracle_action_entropy"))
        > 0.0,
        "material_gain_support": oracle_label_summary.get("answer")
        == "material_gain_broad",
        "open_row_support": open_row_summary.get("answer") == "open_rows_replay_supported",
        "leave_one_seed_split_viable": split_metadata.get("leave_one_seed_viable") is True,
        "leave_one_source_split_viable": split_metadata.get("leave_one_source_viable") is True,
        "no_trainable_leakage": leakage.get("leakage_detected") is False,
        "dominant_oracle_action_share_ok": _number(
            oracle_label_summary.get("dominant_oracle_action_share")
        )
        <= DEFAULT_DOMINANT_ORACLE_ACTION_SHARE_MAX,
        "oracle_actions_observation_legal": legality_summary.get("answer")
        == "oracle_actions_observation_legal",
        "seed29_represented": seed29_summary.get("answer")
        == "seed29_archive_support_present",
        "heuristic_clean": branch_archive_summary.get(
            "zero_heuristic_runtime_actions_except_diagnostic_force"
        )
        is True,
        "branch_replay_has_no_material_failures": branch_archive_summary.get("answer")
        in ("branch_archive_replay_verified", "branch_archive_replay_partial"),
    }
    blockers = [
        {"reason": name}
        for name, passed in sorted(checks.items())
        if not passed
    ]
    answer = (
        "archive_support_sufficient_for_shadow_scorer"
        if not blockers
        else "archive_support_insufficient_for_shadow_scorer"
    )
    return {
        "answer": answer,
        "checks": checks,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "trainable_public_input_leakage": dict(leakage),
    }


def _stop_rules(
    *,
    row_reconstruction: Mapping[str, object],
    branch_archive_summary: Mapping[str, object],
    oracle_label_summary: Mapping[str, object],
    legality_summary: Mapping[str, object],
    open_row_summary: Mapping[str, object],
    seed29_summary: Mapping[str, object],
    leakage: Mapping[str, object],
) -> dict[str, object]:
    resolution_invalid = _int(legality_summary.get("resolution_invalid_count"))
    return {
        "exact_rows_cannot_be_replayed": branch_archive_summary.get("answer")
        == "branch_archive_replay_failed",
        "material_gains_vanish_outside_v108_rows": oracle_label_summary.get("answer")
        == "material_gain_sparse",
        "open_rows_show_no_support": open_row_summary.get("material_answer")
        == "open_rows_material_gain_absent",
        "dominant_oracle_action_share_exceeds_0_5": _number(
            oracle_label_summary.get("dominant_oracle_action_share")
        )
        > DEFAULT_DOMINANT_ORACLE_ACTION_SHARE_MAX,
        "oracle_actions_not_observation_legal": legality_summary.get("answer")
        == "oracle_actions_not_observation_legal",
        "resolution_invalid_unexplained": resolution_invalid > 0,
        "seed29_not_represented": seed29_summary.get("answer")
        != "seed29_archive_support_present",
        "row_alignment_failed": row_reconstruction.get("row_count_matches_v107")
        is not True
        or row_reconstruction.get("row_count_matches_v108") is not True,
        "trainable_feature_leakage_detected": leakage.get("leakage_detected") is True,
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
    learnability_readiness: Mapping[str, object],
    stop_rules: Mapping[str, object],
) -> dict[str, object]:
    ready = (
        learnability_readiness.get("answer")
        == "archive_support_sufficient_for_shadow_scorer"
    )
    return {
        "classification_primary": classification.get("primary"),
        "recommend_v110_shadow_scorer": bool(ready),
        "recommendation": (
            "planner_may_start_v110_shadow_scorer_from_first_recovery_branch_archive"
            if ready
            else "do_not_start_v110_until_archive_support_blockers_are_resolved"
        ),
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_relaxation_recommended": False,
        "next_research_direction": (
            "Train or shadow-evaluate only from leakage-checked public archive rows."
        ),
        "go_stop": {
            "go_for_v110_shadow_scorer": bool(ready),
            **{str(key): bool(value) for key, value in sorted(stop_rules.items())},
        },
    }


def _contract(
    *,
    max_fixture_targets_per_seed: int,
    include_open: bool,
    exhaustive: bool,
    verify_replay: bool,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
        "archive_policy": MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_POLICY,
        "diagnostics_and_training_data_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "private_world_state_serialized": False,
        "fixture_identity_policy_input": False,
        "seed_identity_policy_input": False,
        "branch_identity_policy_input": False,
        "source_identity_policy_input": False,
        "logged_action_fallback_policy_input": False,
        "heuristic_action_selection_effect": "none",
        "first_action_policy": "diagnostic_forced_first_action_then_delegate_v5_policy",
        "continuation_policy": "copied_mind_v3_rollout_context_policy_state",
        "candidate_action_policy": "observation_legal_actions_in_ACTION_NAMES_order",
        "selection": {
            "max_fixture_targets_per_seed": int(max_fixture_targets_per_seed),
            "include_open": bool(include_open),
            "exhaustive": bool(exhaustive),
        },
        "verify_replay": bool(verify_replay),
    }


def _provenance(
    *,
    contract: Mapping[str, object],
    loaded_reports: Mapping[str, object],
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
    archive_rows_path: str | Path | None,
) -> dict[str, object]:
    return {
        "contract_digest": stable_payload_digest(contract),
        "input_reports": {
            name: {
                "path": getattr(loaded, "path", None),
                "digest": _mapping(getattr(loaded, "evidence", {})).get("digest"),
                "file_sha256": _mapping(getattr(loaded, "evidence", {})).get(
                    "file_sha256"
                ),
                "loaded": getattr(loaded, "payload", None) is not None,
            }
            for name, loaded in sorted(loaded_reports.items())
        },
        "trajectory_paths": [str(path) for path in trajectory_paths],
        "trajectory_globs": list(trajectory_glob_patterns),
        "trajectory_file_sha256": {
            str(path): _file_sha256(str(path)) for path in trajectory_paths
        },
        "archive_rows_path": str(archive_rows_path) if archive_rows_path else None,
    }


def _missing_evidence(
    evidence: Mapping[str, object],
    sections: Mapping[str, Mapping[str, object]],
) -> list[str]:
    missing: list[str] = []
    for name, payload in sorted(_mapping(evidence.get("source_reports")).items()):
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
    reconstruction = _mapping(sections.get("row_reconstruction"))
    if reconstruction.get("row_count_matches_v107") is not True:
        missing.append("v107_row_alignment_mismatch")
    if reconstruction.get("row_count_matches_v108") is not True:
        missing.append("v108_row_alignment_mismatch")
    selection = _mapping(sections.get("target_selection"))
    if selection.get("setup_error") is not None:
        missing.append("policy_recreation_failed")
    return missing


def _diagnostic_answers(
    sections: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    answers: dict[str, object] = {}
    for name, section in sorted(sections.items()):
        for key in ("answer", "replay_answer", "distribution_answer", "resolution_answer", "material_answer"):
            if isinstance(section.get(key), str):
                answers[f"{name}.{key}"] = section.get(key)
    return answers


def _section_labels(section: Mapping[str, object]) -> list[str]:
    labels: list[str] = []
    for key in ("answer", "replay_answer", "distribution_answer", "resolution_answer", "material_answer"):
        value = section.get(key)
        if isinstance(value, str):
            labels.append(value)
    for value in _list(section.get("labels")):
        if isinstance(value, str):
            labels.append(value)
    return labels


def _primary_label(labels: Sequence[str]) -> str:
    priority = [
        "missing_evidence_inconclusive",
        "branch_archive_replay_failed",
        "branch_archive_replay_partial",
        "archive_support_insufficient_for_shadow_scorer",
        "oracle_action_distribution_collapsed",
        "oracle_actions_not_observation_legal",
        "oracle_actions_resolution_invalid",
        "open_rows_replay_unsupported",
        "seed29_archive_support_inconclusive",
        "material_gain_sparse",
        "branch_archive_replay_verified",
        "archive_support_sufficient_for_shadow_scorer",
    ]
    for label in priority:
        if label in labels:
            return label
    return sorted(labels)[0]


def _oracle_ranks(runs: Sequence[Mapping[str, object]]) -> dict[str, int]:
    ranked = sorted(runs, key=_oracle_rank_key, reverse=True)
    return {str(run.get("forced_action")): index + 1 for index, run in enumerate(ranked)}


def _oracle_rank_key(run: Mapping[str, object]) -> tuple[int, int, int, float, int, str]:
    return (
        int(bool(run.get("target_alive_at_end", False))),
        _int(run.get("terminal_alive_agents")),
        _int(run.get("births")),
        _number(run.get("target_recovery_score_at_end")),
        -_int(run.get("deaths")),
        str(run.get("forced_action", "")),
    )


def _material_gain_from_deltas(deltas: Mapping[str, object]) -> bool:
    return any(
        _number(deltas.get(field)) > 0.0
        for field in (
            "target_alive_delta",
            "terminal_alive_delta",
            "birth_delta",
            "target_recovery_score_delta",
            "death_reduction_delta",
        )
    )


def _archive_resolution_invalid_cause(action: str, observation_legal: bool) -> str:
    if action in MOVEMENT_ACTIONS or action.startswith("move_"):
        return (
            "resolution_invalid_occupancy_race"
            if observation_legal
            else "resolution_invalid_blocked_route"
        )
    if action in ("eat", "drink") and observation_legal:
        return "resolution_invalid_depleted_resource"
    return "resolution_invalid_other_public_path"


def _archive_row_id(result: Mapping[str, object], forced_action: str) -> str:
    return f"{result.get('branch_id')}::action::{forced_action}"


def _archive_row_excerpt(row: Mapping[str, object]) -> dict[str, object]:
    provenance = _mapping(row.get("provenance"))
    return {
        "archive_row_id": row.get("archive_row_id"),
        "source_kind": row.get("source_kind"),
        "seed": row.get("seed"),
        "tick": row.get("tick"),
        "agent_id": row.get("agent_id"),
        "candidate_action": row.get("candidate_action"),
        "logged_action": provenance.get("logged_action"),
        "material_gain_label": row.get("material_gain_label"),
        "oracle_rank": row.get("oracle_rank"),
    }


def _branch_result_sort_key(result: Mapping[str, object]) -> tuple[str, int, int, int, str]:
    return (
        str(result.get("source_path", "")),
        _int(result.get("record_index"), default=-1),
        _int(result.get("branch_tick"), default=-1),
        _int(result.get("agent_id"), default=-1),
        str(result.get("branch_id", "")),
    )


def _skip_row(row: Mapping[str, object], reason: str) -> dict[str, object]:
    return {**_target_row_excerpt(row), "reason": reason}


def _forbidden_trainable_paths(
    value: object,
    *,
    prefix: str = "trainable_public_input",
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_string = str(key)
            path = f"{prefix}.{key_string}"
            if key_string in FORBIDDEN_TRAINABLE_KEYS:
                paths.append(path)
            paths.extend(_forbidden_trainable_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_forbidden_trainable_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def _dominant_count_share(counts: Mapping[str, int] | Counter[str]) -> dict[str, object]:
    if not counts:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return {"key": key, "count": int(count), "share": _share(count, sum(counts.values()))}


def _entropy(counts: Mapping[str, int] | Counter[str]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        count = int(value)
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return _round(entropy)


def _merge_counts(left: Mapping[str, object], right: Mapping[str, int]) -> dict[str, int]:
    counts = Counter({str(key): _int(value) for key, value in left.items()})
    counts.update({str(key): int(value) for key, value in right.items()})
    return _counter_to_ordered_dict(counts)


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FirstRecoveryBranchArchiveError(f"{field} must be a nonnegative integer")
    return int(value)


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _write_jsonl_gzip(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gzip_file:
            with io.TextIOWrapper(gzip_file, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
                    handle.write("\n")


def archive_rows_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
