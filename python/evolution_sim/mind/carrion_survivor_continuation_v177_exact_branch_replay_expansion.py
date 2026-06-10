from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import gzip
import json
from pathlib import Path

from evolution_sim.env import RunMode
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.broad_regression_branch_intervention import (
    _configure_manual_summary_run,
    _normal_mind_v3_delegate,
    _target_terminal,
)
from evolution_sim.mind.candidate_campaign import (
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v163_tied_set_branch_target_expansion import (
    V163MaterializedBranchPoint,
    V163SelectedBranchPoint,
    _ForcedTiedCandidateThenMindV3Policy,
    _bool_action_mask,
    _record_materialization_payload,
    materialize_selected_branch_points,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    _json_round_trip_digest,
    load_jsonl_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v176_transition_diagnostic_planner import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V176_REPORT_PATH,
    DEFAULT_V177_SHARD_PLAN_OUTPUT_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v177_exact_branch_replay_expansion_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v177_exact_branch_replay_expansion_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v177_compact_transition_diagnostic_row_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY = (
    "current_forced_next_public_transition_context_v1"
)
EXPECTED_V176_CLASSIFICATION = "v176_exact_branch_replay_plan_ready_no_training"
EXPECTED_V176_EXACT_DIGEST = (
    "f24086c6f40bd508e75eef1b18c5f160408a039d759b774c3008ab0ded15bc69"
)
EXPECTED_V177_SHARD_PLAN_DIGEST = (
    "a2583369d4a05ad93504dfdbccbbe2ee65b72710f7d3581a6f3b2c4b653b6f33"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v177-carrion-survivor-continuation-exact-branch-replay-expansion.json"
)
DEFAULT_TRANSITION_DATASET_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v177-carrion-survivor-continuation-compact-transition-rows.jsonl"
)
DEFAULT_TICKS = 120
FORBIDDEN_TRAINABLE_KEY_TOKENS = (
    "seed",
    "fixture",
    "branch",
    "tick",
    "agent_id",
    "line_number",
    "path",
    "digest",
    "provenance",
    "private",
    "outcome",
    "target",
    "runtime",
    "requested",
    "resolved",
    "safe_action",
    "failed",
)
TRAINABLE_PUBLIC_FEATURE_KEYS = {
    "current_public_observation",
    "current_public_action_mask",
    "forced_action",
    "previous_same_agent_public_context",
    "next_public_observation",
    "next_public_action_mask",
}


class CarrionSurvivorContinuationV177ExactBranchReplayExpansionError(ValueError):
    pass


def run_carrion_survivor_continuation_v177_exact_branch_replay_expansion(
    *,
    v176_report_path: str | Path = DEFAULT_V176_REPORT_PATH,
    v177_shard_plan_path: str | Path = DEFAULT_V177_SHARD_PLAN_OUTPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    transition_dataset_output_path: str | Path = DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    expected_v176_exact_digest: str | None = EXPECTED_V176_EXACT_DIGEST,
    expected_v176_classification: str = EXPECTED_V176_CLASSIFICATION,
    expected_v177_shard_plan_digest: str | None = EXPECTED_V177_SHARD_PLAN_DIGEST,
    seed_include: int | None = None,
    priority_include: Sequence[int] | None = None,
    max_plan_rows: int | None = None,
    ticks: int = DEFAULT_TICKS,
    verify_replay: bool = True,
) -> dict[str, object]:
    v176_report = load_json_report(v176_report_path)
    plan_rows = load_jsonl_dataset(v177_shard_plan_path)
    selected_plan_rows = select_v177_plan_rows(
        plan_rows,
        seed_include=seed_include,
        priority_include=priority_include,
        max_plan_rows=max_plan_rows,
    )
    source_validation = validate_v177_sources(
        v176_report=v176_report,
        plan_rows=plan_rows,
        selected_plan_rows=selected_plan_rows,
        expected_v176_exact_digest=expected_v176_exact_digest,
        expected_v176_classification=expected_v176_classification,
        expected_v177_shard_plan_digest=expected_v177_shard_plan_digest,
    )
    selected: list[V163SelectedBranchPoint] = []
    context_by_branch_id: dict[str, dict[str, object]] = {}
    selection = _empty_selection_report("source_validation_failed")
    materialization = _empty_materialization_report("source_validation_failed")
    transition_rows: list[dict[str, object]] = []
    if source_validation.get("passed") is True:
        selected, context_by_branch_id, selection = build_selected_branch_points(
            selected_plan_rows,
            ticks=int(ticks),
        )
        if selection.get("passed") is True and selected:
            materialized, materialization = materialize_selected_branch_points(
                selected,
                ticks=int(ticks),
            )
            if materialization.get("passed") is True:
                transition_rows = build_compact_transition_rows(
                    materialized,
                    context_by_branch_id=context_by_branch_id,
                    verify_replay=bool(verify_replay),
                )
        else:
            materialization = _empty_materialization_report(
                "selection_failed_or_empty"
            )
    leakage_scan = trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in transition_rows]
    )
    row_schema_validation = validate_v177_transition_rows(transition_rows)
    metrics = transition_dataset_metrics(
        selected=selected,
        materialization=materialization,
        transition_rows=transition_rows,
        verify_replay=bool(verify_replay),
    )
    classification = _classification(
        source_validation=source_validation,
        selection=selection,
        materialization=materialization,
        leakage_scan=leakage_scan,
        row_schema_validation=row_schema_validation,
        metrics=metrics,
    )
    dataset_digest = stable_payload_digest(transition_rows)
    _write_jsonl(transition_dataset_output_path, transition_rows)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v176_report": str(v176_report_path),
            "v177_shard_plan": str(v177_shard_plan_path),
            "expected_v176_exact_digest": expected_v176_exact_digest,
            "expected_v176_classification": expected_v176_classification,
            "expected_v177_shard_plan_digest": expected_v177_shard_plan_digest,
            "seed_include": seed_include,
            "priority_include": [
                int(priority) for priority in priority_include or []
            ],
            "max_plan_rows": max_plan_rows,
            "ticks": int(ticks),
            "verify_replay": bool(verify_replay),
            "transition_dataset_output": str(transition_dataset_output_path),
        },
        "source_validation": source_validation,
        "selection": selection,
        "branch_materialization": materialization,
        "metrics": metrics,
        "leakage_scan": leakage_scan,
        "row_schema_validation": row_schema_validation,
        "dataset": {
            "path": str(transition_dataset_output_path),
            "row_count": len(transition_rows),
            "dataset_digest": dataset_digest,
            "row_schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION
            ),
            "feature_policy_id": (
                M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY
            ),
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def select_v177_plan_rows(
    plan_rows: Sequence[Mapping[str, object]],
    *,
    seed_include: int | None,
    priority_include: Sequence[int] | None,
    max_plan_rows: int | None,
) -> list[dict[str, object]]:
    priorities = (
        {int(priority) for priority in priority_include}
        if priority_include
        else None
    )
    selected = []
    for row in sorted(
        (dict(row) for row in plan_rows),
        key=lambda item: (_int(item.get("priority"), default=10**9)),
    ):
        if seed_include is not None and _int(row.get("seed")) != int(seed_include):
            continue
        if priorities is not None and _int(row.get("priority")) not in priorities:
            continue
        selected.append(row)
        if max_plan_rows is not None and len(selected) >= int(max_plan_rows):
            break
    return selected


def validate_v177_sources(
    *,
    v176_report: Mapping[str, object],
    plan_rows: Sequence[Mapping[str, object]],
    selected_plan_rows: Sequence[Mapping[str, object]],
    expected_v176_exact_digest: str | None,
    expected_v176_classification: str,
    expected_v177_shard_plan_digest: str | None,
) -> dict[str, object]:
    failures: list[str] = []
    observed_classification = str(
        _mapping(v176_report.get("classification")).get("primary") or ""
    )
    if (
        v176_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_SCHEMA_VERSION
    ):
        failures.append("v176_schema_version_mismatch")
    if (
        v176_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_POLICY
    ):
        failures.append("v176_policy_mismatch")
    if observed_classification != expected_v176_classification:
        failures.append("v176_unexpected_classification")
    exact = exact_digest_validation_report(v176_report)
    observed_exact = str(v176_report.get("exact_digest") or "")
    if exact.get("passed") is not True:
        failures.append("v176_exact_digest_mismatch")
    if expected_v176_exact_digest and observed_exact != expected_v176_exact_digest:
        failures.append("v176_unexpected_exact_digest")
    plan_digest = stable_payload_digest([dict(row) for row in plan_rows])
    if expected_v177_shard_plan_digest and plan_digest != expected_v177_shard_plan_digest:
        failures.append("v177_shard_plan_digest_mismatch")
    shard_output = _mapping(v176_report.get("v177_shard_plan_output"))
    reported_digest = shard_output.get("plan_digest")
    if reported_digest not in (None, plan_digest):
        failures.append("v176_reported_v177_shard_plan_digest_mismatch")
    plan_validation = validate_v177_plan_rows(plan_rows)
    if plan_validation.get("passed") is not True:
        failures.append("v177_shard_plan_rows_invalid")
    if not selected_plan_rows:
        failures.append("v177_selected_plan_rows_empty")
    lifecycle = _v176_lifecycle_validation(v176_report)
    if lifecycle.get("passed") is not True:
        failures.append("v176_lifecycle_not_diagnostics_only")
    return {
        "policy": "m3_carrion_survivor_continuation_v177_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v176_classification": expected_v176_classification,
        "observed_v176_classification": observed_classification,
        "expected_v176_exact_digest": expected_v176_exact_digest,
        "observed_v176_exact_digest": observed_exact,
        "v176_exact_digest_validation": exact,
        "expected_v177_shard_plan_digest": expected_v177_shard_plan_digest,
        "observed_v177_shard_plan_digest": plan_digest,
        "v176_reported_v177_shard_plan_digest": reported_digest,
        "plan_row_validation": plan_validation,
        "selected_plan_row_count": len(selected_plan_rows),
        "v176_lifecycle_validation": lifecycle,
    }


def validate_v177_plan_rows(
    plan_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    priorities: set[int] = set()
    branch_ids: set[str] = set()
    for row_index, row in enumerate(plan_rows):
        if (
            row.get("schema_version")
            != "m3_carrion_survivor_continuation_v177_exact_branch_replay_shard_row_v1"
        ):
            _append_row_failure(failures, row_index, "schema_version_mismatch")
        if row.get("route") != "exact_branch_replay_expansion":
            _append_row_failure(failures, row_index, "route_mismatch")
        priority = _int(row.get("priority"), default=-1)
        if priority <= 0:
            _append_row_failure(failures, row_index, "invalid_priority")
        if priority in priorities:
            _append_row_failure(failures, row_index, "duplicate_priority")
        priorities.add(priority)
        branch_id = str(row.get("branch_id") or "")
        if not branch_id:
            _append_row_failure(failures, row_index, "missing_branch_id")
        if branch_id in branch_ids:
            _append_row_failure(failures, row_index, "duplicate_branch_id")
        branch_ids.add(branch_id)
        for field in ("seed", "branch_tick", "agent_id", "line_number"):
            if _int(row.get(field), default=-1) < 0:
                _append_row_failure(failures, row_index, f"invalid_{field}")
        if not str(row.get("source_path") or ""):
            _append_row_failure(failures, row_index, "missing_source_path")
        actions = _ordered_actions(row.get("candidate_forced_actions"))
        if not actions:
            _append_row_failure(failures, row_index, "missing_candidate_forced_actions")
        if row.get("training_authorized") is not False:
            _append_row_failure(failures, row_index, "training_authorized")
        if row.get("runtime_artifact_authorized") is not False:
            _append_row_failure(failures, row_index, "runtime_artifact_authorized")
    return {
        "policy": "m3_carrion_survivor_continuation_v177_plan_row_validation_v1",
        "passed": bool(plan_rows) and not failures,
        "failure_count": len(failures),
        "failures": failures[:96],
        "row_count": len(plan_rows),
        "priority_count": len(priorities),
        "branch_id_count": len(branch_ids),
    }


def build_selected_branch_points(
    plan_rows: Sequence[Mapping[str, object]],
    *,
    ticks: int,
) -> tuple[list[V163SelectedBranchPoint], dict[str, dict[str, object]], dict[str, object]]:
    records_by_path: dict[Path, list[tuple[int, dict[str, object]]]] = {}
    selected: list[V163SelectedBranchPoint] = []
    context_by_branch_id: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    by_seed: Counter[int] = Counter()
    by_action_width: Counter[int] = Counter()
    for row_index, row in enumerate(plan_rows):
        source_path = Path(str(row.get("source_path") or ""))
        if source_path not in records_by_path:
            try:
                records_by_path[source_path] = _source_records(source_path)
            except OSError as exc:
                failures.append(
                    {
                        "row_index": row_index,
                        "reason": "source_record_load_failed",
                        "source_path": str(source_path),
                        "error": str(exc),
                    }
                )
                continue
        records = records_by_path[source_path]
        line_number = _int(row.get("line_number"), default=-1)
        record = _record_at_line(records, line_number=line_number)
        if record is None:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "source_record_line_not_found",
                    "source_path": str(source_path),
                    "line_number": line_number,
                }
            )
            continue
        mismatches = _source_record_plan_mismatches(row=row, record=record)
        if mismatches:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "source_record_plan_mismatch",
                    "mismatches": mismatches,
                }
            )
            continue
        action_mask = _bool_action_mask(
            record.get("public_action_mask") or record.get("action_mask")
        )
        actions = _ordered_actions(row.get("candidate_forced_actions"))
        unsupported = [action for action in actions if action_mask.get(action) is not True]
        if unsupported:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "candidate_forced_action_not_public_mask_supported",
                    "actions": unsupported,
                }
            )
            continue
        branch_id = str(row.get("branch_id"))
        point = V163SelectedBranchPoint(
            branch_id=branch_id,
            seed=_int(row.get("seed")),
            ticks=int(ticks),
            branch_tick=_int(row.get("branch_tick")),
            record_index=_int(row.get("line_number")),
            branch_index=len(selected),
            agent_id=_int(row.get("agent_id")),
            source_path=str(source_path),
            line_number=line_number,
            runtime_requested_action=str(record.get("requested_action") or ""),
            runtime_resolved_action=str(record.get("resolved_action") or ""),
            predicted_action="",
            nearest_neighbor_row_index=_int(row.get("row_index"), default=-1),
            top_value_candidate_set=tuple(actions),
            action_mask=action_mask,
            observation_input=deepcopy(dict(_mapping(record.get("observation_input")))),
            observation_schema=_optional_string(record.get("observation_schema")),
            observation_digest=_optional_string(record.get("observation_digest")),
            source_record_digest=stable_payload_digest(
                _record_materialization_payload(record)
            ),
            selection_rationale={
                "v177_exact_branch_replay_expansion": True,
                "v177_selected_public_features": {
                    "observation_input": deepcopy(
                        dict(_mapping(record.get("observation_input")))
                    ),
                    "action_mask": action_mask,
                },
                "candidate_forced_actions_from_v176_plan": actions,
                "source_identity_used_for_exact_materialization_only": True,
                "source_identity_used_as_trainable_input": False,
                "runtime_requested_or_resolved_action_used_as_trainable_input": False,
                "future_outcome_used_as_trainable_input": False,
            },
        )
        selected.append(point)
        context_by_branch_id[branch_id] = {
            "plan_row": dict(row),
            "source_record": deepcopy(record),
            "previous_same_agent_public_context": _previous_same_agent_public_context(
                records,
                line_number=line_number,
                agent_id=point.agent_id,
            ),
        }
        by_seed.update([point.seed])
        by_action_width.update([len(actions)])
    return (
        selected,
        context_by_branch_id,
        {
            "policy": "m3_carrion_survivor_continuation_v177_selection_v1",
            "passed": not failures and len(selected) == len(plan_rows) and bool(selected),
            "failure_count": len(failures),
            "failures": failures[:96],
            "plan_row_count": len(plan_rows),
            "selected_branch_point_count": len(selected),
            "selected_by_seed": {
                str(seed): int(count) for seed, count in sorted(by_seed.items())
            },
            "candidate_forced_action_width_counts": {
                str(width): int(count) for width, count in sorted(by_action_width.items())
            },
            "private_world_state_serialized": False,
        },
    )


def build_compact_transition_rows(
    materialized: Sequence[V163MaterializedBranchPoint],
    *,
    context_by_branch_id: Mapping[str, Mapping[str, object]],
    verify_replay: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in materialized:
        context = _mapping(context_by_branch_id.get(point.selected.branch_id))
        for action in point.selected.top_value_candidate_set:
            if point.selected.action_mask.get(action) is not True:
                continue
            run, digest = _execute_candidate_transition_once(
                point,
                forced_action=action,
                context=context,
            )
            verification = {
                "verified": None,
                "expected_digest": digest,
                "actual_digest": None,
            }
            if verify_replay:
                replay, replay_digest = _execute_candidate_transition_once(
                    point,
                    forced_action=action,
                    context=context,
                )
                verification = {
                    "verified": replay_digest == digest,
                    "expected_digest": digest,
                    "actual_digest": replay_digest,
                }
                run["deterministic_replay_match_sample"] = {
                    "forced_action_used": replay.get("forced_action_used"),
                    "transition_done": replay.get("transition_done"),
                    "next_public_observation_available": replay.get(
                        "next_public_observation_available"
                    ),
                }
            rows.append(_compact_transition_row(run, verification=verification))
    return rows


def _execute_candidate_transition_once(
    point: V163MaterializedBranchPoint,
    *,
    forced_action: str,
    context: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    world = deepcopy(point.world)
    delegate = _normal_mind_v3_delegate(world.policy)
    world.policy = _ForcedTiedCandidateThenMindV3Policy(
        target_agent_id=point.selected.agent_id,
        forced_action=forced_action,
        delegate=delegate,
    )
    _configure_manual_summary_run(world)
    current_index: int | None = None
    current_record: Mapping[str, object] | None = None
    next_record: Mapping[str, object] | None = None
    for tick in range(point.selected.branch_tick, point.selected.ticks):
        world.tick = tick
        world._run_tick()
        if current_record is None:
            current_index, current_record = _current_transition_record(
                world.trajectory_records,
                selected=point.selected,
            )
        if current_index is not None and current_record is not None:
            next_record = _next_same_agent_record(
                world.trajectory_records,
                start_index=current_index,
                agent_id=point.selected.agent_id,
            )
            if next_record is not None or _record_after_dead(current_record):
                break
        if not world.alive_agents():
            break
    forced_used = bool(getattr(world.policy, "used", False))
    summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)
    run = {
        "branch_id": point.selected.branch_id,
        "seed": point.selected.seed,
        "fixture": "broad",
        "ticks": point.selected.ticks,
        "branch_tick": point.selected.branch_tick,
        "agent_id": point.selected.agent_id,
        "source_path": point.selected.source_path,
        "line_number": point.selected.line_number,
        "source_row_index": _mapping(context.get("plan_row")).get("row_index"),
        "failed_safe_action": _mapping(context.get("plan_row")).get(
            "failed_safe_action"
        ),
        "failure_types": _mapping(context.get("plan_row")).get("failure_types"),
        "forced_action": forced_action,
        "forced_action_used": forced_used,
        "branch_state_digest": point.branch_state_digest,
        "source_record_digest": point.selected.source_record_digest,
        "materialized_record_digest": point.materialized_record_digest,
        "current_record": deepcopy(current_record) if current_record else None,
        "next_record": deepcopy(next_record) if next_record else None,
        "previous_same_agent_public_context": deepcopy(
            _mapping(context.get("previous_same_agent_public_context"))
        ),
        "transition_done": _transition_done(current_record, next_record),
        "next_public_observation_available": (
            isinstance(next_record, Mapping)
            and isinstance(next_record.get("observation_input"), Mapping)
        ),
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "target_terminal": _target_terminal(world, point.selected.agent_id),
        "trajectory_record_count": len(world.trajectory_records),
        "diagnostics_only": True,
    }
    return run, stable_payload_digest(_transition_run_digest_payload(run))


def _compact_transition_row(
    run: Mapping[str, object],
    *,
    verification: Mapping[str, object],
) -> dict[str, object]:
    current_record = _mapping(run.get("current_record"))
    next_record = _mapping(run.get("next_record"))
    current_observation = deepcopy(dict(_mapping(current_record.get("observation_input"))))
    current_mask = _complete_action_mask(
        _mapping(current_record.get("public_action_mask") or current_record.get("action_mask"))
    )
    next_observation = (
        deepcopy(dict(_mapping(next_record.get("observation_input"))))
        if next_record
        else None
    )
    next_mask = (
        _complete_action_mask(
            _mapping(next_record.get("public_action_mask") or next_record.get("action_mask"))
        )
        if next_record
        else None
    )
    previous_context = _public_previous_context_for_trainable_payload(
        _mapping(run.get("previous_same_agent_public_context"))
    )
    forced_action = str(run.get("forced_action") or "")
    trainable_features = {
        "current_public_observation": current_observation,
        "current_public_action_mask": current_mask,
        "forced_action": forced_action,
        "previous_same_agent_public_context": previous_context,
        "next_public_observation": next_observation,
        "next_public_action_mask": next_mask,
    }
    metadata = {
        "metadata_schema_version": (
            "m3_carrion_survivor_continuation_v177_compact_transition_metadata_v1"
        ),
        "branch_id": run.get("branch_id"),
        "seed": run.get("seed"),
        "fixture": run.get("fixture"),
        "branch_tick": run.get("branch_tick"),
        "agent_id": run.get("agent_id"),
        "source_path": run.get("source_path"),
        "line_number": run.get("line_number"),
        "source_row_index": run.get("source_row_index"),
        "failed_safe_action": run.get("failed_safe_action"),
        "failure_types": run.get("failure_types"),
        "source_record_digest": run.get("source_record_digest"),
        "materialized_record_digest": run.get("materialized_record_digest"),
        "branch_state_digest": run.get("branch_state_digest"),
        "replay_verification_digest": verification.get("expected_digest"),
        "source_identity_used_for_exact_materialization_only": True,
        "source_identity_used_as_trainable_input": False,
        "runtime_requested_or_resolved_action_used_as_trainable_input": False,
        "current_or_future_outcome_used_as_trainable_input": False,
        "diagnostic_target_used_as_trainable_input": False,
    }
    return {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION
        ),
        "row_origin": "v177_exact_branch_replay_from_v176_shard_plan",
        "feature_policy_id": (
            M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY
        ),
        "trainable_public_features": trainable_features,
        "current_public_observation": current_observation,
        "current_public_action_mask": current_mask,
        "forced_action": forced_action,
        "previous_same_agent_public_context": previous_context,
        "next_public_observation": next_observation,
        "next_public_action_mask": next_mask,
        "next_public_observation_available": next_observation is not None,
        "next_public_action_mask_available": next_mask is not None,
        "transition_done": run.get("transition_done") is True,
        "short_horizon_public_outcome_summary": {
            "forced_action_used": run.get("forced_action_used") is True,
            "current_requested_action": current_record.get("requested_action"),
            "current_resolved_action": current_record.get("resolved_action"),
            "current_action_valid": current_record.get("action_valid"),
            "current_resolution_action_valid": current_record.get(
                "resolution_action_valid"
            ),
            "current_moved": current_record.get("moved"),
            "current_reward_total": _reward_total(current_record.get("reward")),
            "current_resource_gain": _resource_gain(current_record.get("outcome")),
            "target_terminal": run.get("target_terminal"),
            "alive_agents": run.get("alive_agents"),
            "births": run.get("births"),
            "deaths": run.get("deaths"),
        },
        "replay_verification": dict(verification),
        "metadata": metadata,
    }


def trainable_payload_leakage_scan(
    feature_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for row_index, payload in enumerate(feature_payloads):
        _scan_trainable_payload(
            value=payload,
            row_index=row_index,
            path=("trainable_public_features",),
            failures=failures,
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v177_trainable_payload_leakage_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:96],
        "forbidden_key_tokens": list(FORBIDDEN_TRAINABLE_KEY_TOKENS),
        "payload_count": len(feature_payloads),
    }


def validate_v177_transition_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    schema_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    by_seed: Counter[int] = Counter()
    for row_index, row in enumerate(rows):
        schema = str(row.get("schema_version") or "")
        schema_counts.update([schema])
        if schema != M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION:
            _append_row_failure(failures, row_index, "schema_version_mismatch")
        if (
            str(row.get("feature_policy_id") or "")
            != M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY
        ):
            _append_row_failure(failures, row_index, "feature_policy_mismatch")
        features = _mapping(row.get("trainable_public_features"))
        if set(features) != TRAINABLE_PUBLIC_FEATURE_KEYS:
            _append_row_failure(failures, row_index, "trainable_feature_key_set_mismatch")
        current_observation = _mapping(row.get("current_public_observation"))
        if not current_observation:
            _append_row_failure(failures, row_index, "current_public_observation_missing")
        if current_observation != _mapping(features.get("current_public_observation")):
            _append_row_failure(failures, row_index, "current_observation_feature_mismatch")
        current_mask = _complete_action_mask(_mapping(row.get("current_public_action_mask")))
        if current_mask != _complete_action_mask(
            _mapping(features.get("current_public_action_mask"))
        ):
            _append_row_failure(failures, row_index, "current_action_mask_feature_mismatch")
        action = str(row.get("forced_action") or "")
        action_counts.update([action])
        if action not in ACTION_NAMES:
            _append_row_failure(failures, row_index, "invalid_forced_action")
        if features.get("forced_action") != action:
            _append_row_failure(failures, row_index, "forced_action_feature_mismatch")
        if current_mask.get(action) is not True:
            _append_row_failure(failures, row_index, "forced_action_not_current_mask_supported")
        previous = _mapping(row.get("previous_same_agent_public_context"))
        if previous != _mapping(features.get("previous_same_agent_public_context")):
            _append_row_failure(failures, row_index, "previous_context_feature_mismatch")
        if "available" not in previous:
            _append_row_failure(failures, row_index, "previous_context_missing_available")
        done = row.get("transition_done") is True
        next_obs = row.get("next_public_observation")
        next_mask = row.get("next_public_action_mask")
        if done:
            if next_obs is not None or next_mask is not None:
                _append_row_failure(failures, row_index, "done_transition_has_next_fields")
        else:
            if not isinstance(next_obs, Mapping) or not next_obs:
                _append_row_failure(failures, row_index, "next_public_observation_missing")
            if not isinstance(next_mask, Mapping):
                _append_row_failure(failures, row_index, "next_public_action_mask_missing")
        if next_obs != features.get("next_public_observation"):
            _append_row_failure(failures, row_index, "next_observation_feature_mismatch")
        if (
            _complete_action_mask(_mapping(next_mask))
            if isinstance(next_mask, Mapping)
            else None
        ) != (
            _complete_action_mask(_mapping(features.get("next_public_action_mask")))
            if isinstance(features.get("next_public_action_mask"), Mapping)
            else None
        ):
            _append_row_failure(failures, row_index, "next_action_mask_feature_mismatch")
        replay = _mapping(row.get("replay_verification"))
        if replay and replay.get("verified") is not True:
            _append_row_failure(failures, row_index, "replay_verification_failed")
        metadata = _mapping(row.get("metadata"))
        seed = _int(metadata.get("seed"), default=-1)
        by_seed.update([seed])
        if seed < 0:
            _append_row_failure(failures, row_index, "metadata_seed_missing")
        for flag in (
            "source_identity_used_as_trainable_input",
            "runtime_requested_or_resolved_action_used_as_trainable_input",
            "current_or_future_outcome_used_as_trainable_input",
            "diagnostic_target_used_as_trainable_input",
        ):
            if metadata.get(flag) is not False:
                _append_row_failure(failures, row_index, f"{flag}_not_false")
    return {
        "policy": "m3_carrion_survivor_continuation_v177_row_schema_validation_v1",
        "passed": bool(rows) and not failures,
        "failure_count": len(failures),
        "failures": failures[:128],
        "row_count": len(rows),
        "schema_counts": dict(sorted(schema_counts.items())),
        "forced_action_counts": dict(sorted(action_counts.items(), key=lambda item: _action_order(item[0]))),
        "row_counts_by_seed": {
            str(seed): int(count) for seed, count in sorted(by_seed.items())
        },
    }


def transition_dataset_metrics(
    *,
    selected: Sequence[V163SelectedBranchPoint],
    materialization: Mapping[str, object],
    transition_rows: Sequence[Mapping[str, object]],
    verify_replay: bool,
) -> dict[str, object]:
    by_seed: Counter[int] = Counter()
    by_action: Counter[str] = Counter()
    done_count = 0
    next_count = 0
    previous_count = 0
    replay_verified = 0
    forced_used = 0
    for row in transition_rows:
        metadata = _mapping(row.get("metadata"))
        by_seed.update([_int(metadata.get("seed"), default=-1)])
        action = str(row.get("forced_action") or "")
        by_action.update([action])
        if row.get("transition_done") is True:
            done_count += 1
        if row.get("next_public_observation_available") is True:
            next_count += 1
        if _mapping(row.get("previous_same_agent_public_context")).get("available") is True:
            previous_count += 1
        if _mapping(row.get("replay_verification")).get("verified") is True:
            replay_verified += 1
        if (
            _mapping(row.get("short_horizon_public_outcome_summary")).get(
                "forced_action_used"
            )
            is True
        ):
            forced_used += 1
    row_count = len(transition_rows)
    all_verified = (not verify_replay) or (row_count > 0 and replay_verified == row_count)
    return {
        "policy": "m3_carrion_survivor_continuation_v177_transition_dataset_metrics_v1",
        "selected_branch_point_count": len(selected),
        "materialized_branch_point_count": _int(
            materialization.get("materialized_branch_point_count")
        ),
        "transition_row_count": row_count,
        "forced_action_used_count": forced_used,
        "all_forced_actions_used": row_count > 0 and forced_used == row_count,
        "replay_verification_enabled": bool(verify_replay),
        "replay_verified_row_count": replay_verified,
        "all_replays_verified": all_verified,
        "transition_done_count": done_count,
        "rows_with_next_public_observation": next_count,
        "rows_with_previous_same_agent_public_context": previous_count,
        "row_counts_by_seed": {
            str(seed): int(count) for seed, count in sorted(by_seed.items()) if seed >= 0
        },
        "forced_action_counts": dict(
            sorted(by_action.items(), key=lambda item: _action_order(item[0]))
        ),
        "compact_transition_support_ready": (
            row_count > 0
            and forced_used == row_count
            and all_verified
            and next_count + done_count == row_count
        ),
    }


def _source_records(path: Path) -> list[tuple[int, dict[str, object]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[tuple[int, dict[str, object]]] = []
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = payload.get("record") if isinstance(payload, Mapping) else None
            if isinstance(record, Mapping):
                records.append((line_number, dict(record)))
    return records


def _record_at_line(
    records: Sequence[tuple[int, dict[str, object]]],
    *,
    line_number: int,
) -> dict[str, object] | None:
    for observed_line, record in records:
        if int(observed_line) == int(line_number):
            return record
    return None


def _source_record_plan_mismatches(
    *,
    row: Mapping[str, object],
    record: Mapping[str, object],
) -> list[dict[str, object]]:
    comparisons = {
        "seed": (row.get("seed"), _seed_from_record(record, fallback=row.get("seed"))),
        "branch_tick": (row.get("branch_tick"), record.get("tick")),
        "agent_id": (row.get("agent_id"), record.get("agent_id")),
    }
    mismatches = []
    for field, (expected, observed) in comparisons.items():
        if _int(expected, default=-1) != _int(observed, default=-1):
            mismatches.append(
                {"field": field, "expected": expected, "observed": observed}
            )
    if not isinstance(record.get("observation_input"), Mapping):
        mismatches.append({"field": "observation_input", "reason": "missing"})
    if not isinstance(record.get("action_mask"), Mapping):
        mismatches.append({"field": "action_mask", "reason": "missing"})
    return mismatches


def _previous_same_agent_public_context(
    records: Sequence[tuple[int, dict[str, object]]],
    *,
    line_number: int,
    agent_id: int,
) -> dict[str, object]:
    previous_record: dict[str, object] | None = None
    for observed_line, record in records:
        if int(observed_line) >= int(line_number):
            break
        if _int(record.get("agent_id"), default=-1) == int(agent_id):
            previous_record = record
    if previous_record is None:
        return {
            "available": False,
            "public_observation": None,
            "public_action_mask": None,
            "public_action": None,
            "moved": None,
        }
    return {
        "available": True,
        "public_observation": deepcopy(
            dict(_mapping(previous_record.get("observation_input")))
        ),
        "public_action_mask": _complete_action_mask(
            _mapping(
                previous_record.get("public_action_mask")
                or previous_record.get("action_mask")
            )
        ),
        "public_action": str(previous_record.get("resolved_action") or ""),
        "moved": previous_record.get("moved") is True,
    }


def _public_previous_context_for_trainable_payload(
    context: Mapping[str, object],
) -> dict[str, object]:
    available = context.get("available") is True
    return {
        "available": available,
        "public_observation": (
            deepcopy(dict(_mapping(context.get("public_observation"))))
            if available
            else None
        ),
        "public_action_mask": (
            _complete_action_mask(_mapping(context.get("public_action_mask")))
            if available
            else None
        ),
        "public_action": str(context.get("public_action") or "") if available else None,
        "moved": context.get("moved") is True if available else None,
    }


def _current_transition_record(
    records: Sequence[Mapping[str, object]],
    *,
    selected: V163SelectedBranchPoint,
) -> tuple[int | None, Mapping[str, object] | None]:
    for index, record in enumerate(records):
        if (
            _int(record.get("tick"), default=-1) == int(selected.branch_tick)
            and _int(record.get("agent_id"), default=-1) == int(selected.agent_id)
        ):
            return index, record
    return None, None


def _next_same_agent_record(
    records: Sequence[Mapping[str, object]],
    *,
    start_index: int,
    agent_id: int,
) -> Mapping[str, object] | None:
    for record in records[int(start_index) + 1 :]:
        if _int(record.get("agent_id"), default=-1) == int(agent_id):
            return record
    return None


def _transition_done(
    current_record: Mapping[str, object] | None,
    next_record: Mapping[str, object] | None,
) -> bool:
    if current_record is None:
        return True
    if _record_after_dead(current_record):
        return True
    return next_record is None


def _record_after_dead(current_record: Mapping[str, object]) -> bool:
    after = current_record.get("after")
    return isinstance(after, Mapping) and after.get("alive") is False


def _transition_run_digest_payload(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": run.get("branch_id"),
        "seed": run.get("seed"),
        "fixture": run.get("fixture"),
        "ticks": run.get("ticks"),
        "branch_tick": run.get("branch_tick"),
        "agent_id": run.get("agent_id"),
        "forced_action": run.get("forced_action"),
        "forced_action_used": run.get("forced_action_used"),
        "branch_state_digest": run.get("branch_state_digest"),
        "current_public_observation": _mapping(_mapping(run.get("current_record")).get("observation_input")),
        "current_public_action_mask": _complete_action_mask(
            _mapping(_mapping(run.get("current_record")).get("action_mask"))
        ),
        "next_public_observation": _mapping(_mapping(run.get("next_record")).get("observation_input")),
        "next_public_action_mask": _complete_action_mask(
            _mapping(_mapping(run.get("next_record")).get("action_mask"))
        )
        if run.get("next_record") is not None
        else None,
        "previous_same_agent_public_context": run.get(
            "previous_same_agent_public_context"
        ),
        "transition_done": run.get("transition_done"),
        "ticks_executed": run.get("ticks_executed"),
        "alive_agents": run.get("alive_agents"),
        "births": run.get("births"),
        "deaths": run.get("deaths"),
        "target_terminal": run.get("target_terminal"),
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    selection: Mapping[str, object],
    materialization: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    metrics: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v177_exact_branch_replay_expansion_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if selection.get("passed") is not True:
        return prefix + "selection_invalid_closed_no_training"
    if materialization.get("passed") is not True:
        return prefix + "materialization_blocked_no_training"
    if metrics.get("all_replays_verified") is not True:
        return prefix + "replay_not_deterministic_closed_no_training"
    if (
        leakage_scan.get("passed") is not True
        or row_schema_validation.get("passed") is not True
    ):
        return prefix + "compact_transition_rows_invalid_closed_no_training"
    if metrics.get("compact_transition_support_ready") is True:
        return prefix + "compact_transition_rows_ready_no_training"
    return prefix + "compact_transition_rows_limited_no_training"


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification.endswith("compact_transition_rows_ready_no_training")
    return {
        "policy": "m3_carrion_survivor_continuation_v177_route_recommendation_v1",
        "recommended_next_route": (
            "v178_transition_row_dataset_audit_no_training"
            if ready
            else "repair_v177_exact_branch_replay_sources_before_capacity_work"
        ),
        "transition_row_dataset_audit_recommended": ready,
        "world_model_or_neural_capacity_work_unblocked_by_source_data": ready,
        "training_authorized": False,
        "runtime_integration_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_allowed": False,
        "fit_allowed": False,
        "runtime_artifact_allowed": False,
        "runtime_action_change_allowed": False,
        "shadow_or_live_eval_allowed": False,
        "promotion_allowed": False,
        "gate_relaxation_allowed": False,
        "replay_viewer_schema_change_allowed": False,
        "source_identity_metadata_only": True,
        "current_and_next_public_fields_are_dataset_inputs": True,
        "short_horizon_outcomes_are_diagnostic_targets_only": True,
    }


def _lifecycle_flags() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_ran": False,
        "training_authorized": False,
        "fit_ran": False,
        "scorer_retraining_ran": False,
        "scorer_retraining_authorized": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": False,
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_ran": False,
        "replay_viewer_schema_changed": False,
        "non_promoted": True,
    }


def _v176_lifecycle_validation(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    for field in (
        "training_ran",
        "fit_ran",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "runtime_observation_schema_changed",
        "runtime_policy_changed",
        "shadow_eval_ran",
        "live_ab_ran",
        "promotion_authorized",
        "replay_viewer_schema_changed",
    ):
        if field in report and report.get(field) is not False:
            failures.append({"field": field, "observed": report.get(field)})
    return {
        "policy": "m3_carrion_survivor_continuation_v177_v176_lifecycle_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def _empty_selection_report(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v177_selection_v1",
        "passed": False,
        "reason": reason,
        "plan_row_count": 0,
        "selected_branch_point_count": 0,
        "failure_count": 0,
        "failures": [],
    }


def _empty_materialization_report(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v163_exact_branch_materialization_v1",
        "selected_branch_point_count": 0,
        "materialized_branch_point_count": 0,
        "materialization_failure_count": 0,
        "materialization_failures": [],
        "exact_materialization_proven": False,
        "passed": False,
        "reason": reason,
    }


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _scan_trainable_payload(
    *,
    value: object,
    row_index: int,
    path: tuple[str, ...],
    failures: list[dict[str, object]],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current_path = (*path, key_text)
            token = _matching_forbidden_token(key_text)
            if token:
                failures.append(
                    {
                        "row_index": row_index,
                        "path": ".".join(current_path),
                        "reason": "forbidden_key_token",
                        "token": token,
                    }
                )
            _scan_trainable_payload(
                value=item,
                row_index=row_index,
                path=current_path,
                failures=failures,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_trainable_payload(
                value=item,
                row_index=row_index,
                path=(*path, str(index)),
                failures=failures,
            )


def _matching_forbidden_token(text: str) -> str | None:
    lowered = text.lower()
    parts = {
        part
        for chunk in lowered.split(".")
        for part in chunk.replace("-", "_").split("_")
        if part
    }
    for token in FORBIDDEN_TRAINABLE_KEY_TOKENS:
        token_parts = tuple(part for part in token.split("_") if part)
        if len(token_parts) == 1:
            if token_parts[0] in parts:
                return token
        elif token in lowered:
            return token
    return None


def _append_row_failure(
    failures: list[dict[str, object]],
    row_index: int,
    reason: str,
    **extra: object,
) -> None:
    payload = {"row_index": row_index, "reason": reason}
    payload.update(extra)
    failures.append(payload)


def _ordered_actions(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return sorted(
        {str(action) for action in value if str(action) in ACTION_NAMES},
        key=_action_order,
    )


def _complete_action_mask(value: Mapping[str, object]) -> dict[str, bool]:
    return {action: value.get(action) is True for action in ACTION_NAMES}


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _seed_from_record(record: Mapping[str, object], *, fallback: object) -> int:
    metadata = _mapping(record.get("observation_metadata"))
    seed = metadata.get("seed")
    return _int(seed, default=_int(fallback, default=-1))


def _reward_total(value: object) -> float:
    reward = _mapping(value)
    total = reward.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return 0.0
    return _round(float(total))


def _resource_gain(value: object) -> float:
    outcome = _mapping(value)
    gain = outcome.get("resource_gain")
    if isinstance(gain, bool) or not isinstance(gain, (int, float)):
        return 0.0
    return _round(float(gain))
