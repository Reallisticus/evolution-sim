from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import _int, _mapping, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    load_jsonl_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V172_REPORT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V172_DATASET_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION,
    SUPPORT_PROVENANCE_SEEDS,
    trainable_payload_leakage_scan as v172_trainable_payload_leakage_scan,
    validate_v172_target_rows,
)
from evolution_sim.mind.carrion_survivor_continuation_v173_source_split_scorer import (
    EXPECTED_V172_CLASSIFICATION,
    EXPECTED_V172_DATASET_DIGEST,
    EXPECTED_V172_EXACT_DIGEST,
    EXPECTED_V172_ROW_COUNT,
)
from evolution_sim.mind.carrion_survivor_continuation_v177_exact_branch_replay_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V177_REPORT_PATH,
    DEFAULT_TRANSITION_DATASET_OUTPUT_PATH as DEFAULT_V177_TRANSITION_DATASET_PATH,
    EXPECTED_V177_SHARD_PLAN_DIGEST,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION,
    build_compact_transition_rows,
    build_selected_branch_points,
    materialize_selected_branch_points,
    trainable_payload_leakage_scan as v177_trainable_payload_leakage_scan,
    transition_dataset_metrics,
    validate_v177_transition_rows,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V179_EXACT_BRANCH_TRANSITION_ROW_EXPANSION_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V179_EXACT_BRANCH_TRANSITION_ROW_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion_v1"
)
V179_SUPPORT_READY_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion_"
    "compact_transition_rows_support_ready_for_v178_default_audit_no_training"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v179-carrion-survivor-continuation-exact-branch-transition-row-expansion.json"
)
DEFAULT_TRANSITION_DATASET_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v179-carrion-survivor-continuation-compact-transition-rows.jsonl"
)
DEFAULT_BRANCHES_PER_SEED = 5
DEFAULT_MAX_FORCED_ACTIONS_PER_BRANCH = 6
DEFAULT_TICKS = 120
V178_DEFAULT_MIN_ROW_COUNT = 128
V178_DEFAULT_MIN_SEED_COUNT = 3
V178_DEFAULT_MIN_BRANCH_COUNT = 24
V178_DEFAULT_MIN_FORCED_ACTION_COUNT = 6
V177_EXPECTED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v177_exact_branch_replay_expansion_"
    "compact_transition_rows_ready_no_training"
)


class CarrionSurvivorContinuationV179ExactBranchTransitionRowExpansionError(
    ValueError
):
    pass


def run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
    *,
    v172_report_path: str | Path = DEFAULT_V172_REPORT_PATH,
    v172_dataset_path: str | Path = DEFAULT_V172_DATASET_PATH,
    v177_report_path: str | Path = DEFAULT_V177_REPORT_PATH,
    v177_transition_dataset_path: str | Path = DEFAULT_V177_TRANSITION_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    transition_dataset_output_path: str | Path = DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    expected_v172_exact_digest: str | None = EXPECTED_V172_EXACT_DIGEST,
    expected_v172_dataset_digest: str | None = EXPECTED_V172_DATASET_DIGEST,
    expected_v172_classification: str = EXPECTED_V172_CLASSIFICATION,
    expected_v172_row_count: int = EXPECTED_V172_ROW_COUNT,
    expected_v177_report_exact_digest: str | None = None,
    expected_v177_dataset_digest: str | None = None,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
    branches_per_seed: int = DEFAULT_BRANCHES_PER_SEED,
    max_forced_actions_per_branch: int | None = DEFAULT_MAX_FORCED_ACTIONS_PER_BRANCH,
    ticks: int = DEFAULT_TICKS,
    verify_replay: bool = True,
) -> dict[str, object]:
    v172_report = load_json_report(v172_report_path)
    v172_rows = load_jsonl_dataset(v172_dataset_path)
    v177_report = load_json_report(v177_report_path)
    v177_rows = load_jsonl_dataset(v177_transition_dataset_path)
    source_validation = validate_v179_sources(
        v172_report=v172_report,
        v172_rows=v172_rows,
        v177_report=v177_report,
        v177_rows=v177_rows,
        expected_v172_exact_digest=expected_v172_exact_digest,
        expected_v172_dataset_digest=expected_v172_dataset_digest,
        expected_v172_classification=expected_v172_classification,
        expected_v172_row_count=expected_v172_row_count,
        expected_v177_report_exact_digest=expected_v177_report_exact_digest,
        expected_v177_dataset_digest=expected_v177_dataset_digest,
        support_provenance_seeds=support_provenance_seeds,
    )
    plan_rows: list[dict[str, object]] = []
    plan_generation = _empty_plan_generation("source_validation_failed")
    selection = _empty_selection_report("source_validation_failed")
    materialization = _empty_materialization_report("source_validation_failed")
    transition_rows: list[dict[str, object]] = []
    selected = []
    if source_validation.get("passed") is True:
        plan_rows, plan_generation = build_v179_plan_rows(
            v172_rows,
            support_provenance_seeds=support_provenance_seeds,
            branches_per_seed=int(branches_per_seed),
            max_forced_actions_per_branch=max_forced_actions_per_branch,
        )
        if plan_generation.get("passed") is True:
            selected, context_by_branch_id, selection = build_selected_branch_points(
                plan_rows,
                ticks=int(ticks),
            )
            if selection.get("passed") is True and selected:
                materialized, materialization = materialize_selected_branch_points(
                    selected,
                    ticks=int(ticks),
                )
                if materialization.get("passed") is True:
                    transition_rows = _tag_v179_rows(
                        build_compact_transition_rows(
                            materialized,
                            context_by_branch_id=context_by_branch_id,
                            verify_replay=bool(verify_replay),
                        )
                    )
            else:
                materialization = _empty_materialization_report(
                    "selection_failed_or_empty"
                )
    leakage_scan = v177_trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in transition_rows]
    )
    row_schema_validation = validate_v177_transition_rows(transition_rows)
    metrics = _v179_transition_dataset_metrics(
        selected=selected,
        materialization=materialization,
        transition_rows=transition_rows,
        verify_replay=bool(verify_replay),
    )
    support_summary = transition_row_support_summary(transition_rows)
    classification = _classification(
        source_validation=source_validation,
        plan_generation=plan_generation,
        selection=selection,
        materialization=materialization,
        leakage_scan=leakage_scan,
        row_schema_validation=row_schema_validation,
        metrics=metrics,
        support_summary=support_summary,
    )
    dataset_digest = stable_payload_digest(transition_rows)
    _write_jsonl(transition_dataset_output_path, transition_rows)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V179_EXACT_BRANCH_TRANSITION_ROW_EXPANSION_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V179_EXACT_BRANCH_TRANSITION_ROW_EXPANSION_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v172_report": str(v172_report_path),
            "v172_dataset": str(v172_dataset_path),
            "v177_report": str(v177_report_path),
            "v177_transition_dataset": str(v177_transition_dataset_path),
            "expected_v172_exact_digest": expected_v172_exact_digest,
            "expected_v172_dataset_digest": expected_v172_dataset_digest,
            "expected_v172_classification": expected_v172_classification,
            "expected_v172_row_count": int(expected_v172_row_count),
            "expected_v177_report_exact_digest": expected_v177_report_exact_digest,
            "expected_v177_dataset_digest": expected_v177_dataset_digest,
            "support_provenance_seeds": [
                int(seed) for seed in support_provenance_seeds
            ],
            "branches_per_seed": int(branches_per_seed),
            "max_forced_actions_per_branch": (
                None
                if max_forced_actions_per_branch is None
                else int(max_forced_actions_per_branch)
            ),
            "ticks": int(ticks),
            "verify_replay": bool(verify_replay),
            "transition_dataset_output": str(transition_dataset_output_path),
        },
        "source_validation": source_validation,
        "plan_generation": plan_generation,
        "selection": selection,
        "branch_materialization": materialization,
        "metrics": metrics,
        "support_summary": support_summary,
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
        "route_recommendation": _route_recommendation(
            classification=classification,
            source_validation=source_validation,
        ),
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v179_sources(
    *,
    v172_report: Mapping[str, object],
    v172_rows: Sequence[Mapping[str, object]],
    v177_report: Mapping[str, object],
    v177_rows: Sequence[Mapping[str, object]],
    expected_v172_exact_digest: str | None,
    expected_v172_dataset_digest: str | None,
    expected_v172_classification: str,
    expected_v172_row_count: int,
    expected_v177_report_exact_digest: str | None,
    expected_v177_dataset_digest: str | None,
    support_provenance_seeds: Sequence[int],
) -> dict[str, object]:
    failures: list[str] = []
    observed_v172_classification = str(
        _mapping(v172_report.get("classification")).get("primary") or ""
    )
    if (
        v172_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION
    ):
        failures.append("v172_schema_version_mismatch")
    if (
        v172_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_POLICY
    ):
        failures.append("v172_policy_mismatch")
    if observed_v172_classification != expected_v172_classification:
        failures.append("v172_unexpected_classification")
    v172_exact = exact_digest_validation_report(v172_report)
    observed_v172_exact = str(v172_report.get("exact_digest") or "")
    if v172_exact.get("passed") is not True:
        failures.append("v172_exact_digest_mismatch")
    if expected_v172_exact_digest and observed_v172_exact != expected_v172_exact_digest:
        failures.append("v172_unexpected_exact_digest")
    v172_dataset_digest = stable_payload_digest([dict(row) for row in v172_rows])
    v172_reported_dataset = _mapping(v172_report.get("dataset"))
    if expected_v172_dataset_digest and v172_dataset_digest != expected_v172_dataset_digest:
        failures.append("v172_unexpected_dataset_digest")
    if (
        str(v172_reported_dataset.get("dataset_digest") or "")
        and str(v172_reported_dataset.get("dataset_digest") or "") != v172_dataset_digest
    ):
        failures.append("v172_reported_dataset_digest_mismatch")
    if len(v172_rows) != int(expected_v172_row_count):
        failures.append("v172_row_count_mismatch")
    if (
        _int(v172_reported_dataset.get("row_count"), default=-1) >= 0
        and _int(v172_reported_dataset.get("row_count"), default=-1) != len(v172_rows)
    ):
        failures.append("v172_reported_row_count_mismatch")
    v172_lifecycle = _lifecycle_validation(v172_report, policy="v179_v172_lifecycle_validation")
    if v172_lifecycle.get("passed") is not True:
        failures.append("v172_lifecycle_not_diagnostics_only")
    v172_row_schema = validate_v172_target_rows(
        v172_rows,
        support_provenance_seeds=support_provenance_seeds,
    )
    if v172_row_schema.get("passed") is not True:
        failures.append("v172_row_schema_invalid")
    v172_leakage = v172_trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in v172_rows]
    )
    if v172_leakage.get("passed") is not True:
        failures.append("v172_trainable_leakage_failed")
    source_row_integrity = v179_source_row_integrity_scan(v172_rows)
    if source_row_integrity.get("passed") is not True:
        failures.append("v172_source_row_integrity_failed")

    observed_v177_classification = str(
        _mapping(v177_report.get("classification")).get("primary") or ""
    )
    if (
        v177_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION
    ):
        failures.append("v177_schema_version_mismatch")
    if (
        v177_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY
    ):
        failures.append("v177_policy_mismatch")
    if observed_v177_classification != V177_EXPECTED_CLASSIFICATION:
        failures.append("v177_unexpected_classification")
    v177_exact = exact_digest_validation_report(v177_report)
    if v177_exact.get("passed") is not True:
        failures.append("v177_exact_digest_mismatch")
    observed_v177_exact = str(v177_report.get("exact_digest") or "")
    if (
        expected_v177_report_exact_digest
        and observed_v177_exact != expected_v177_report_exact_digest
    ):
        failures.append("v177_unexpected_exact_digest")
    v177_dataset_digest = stable_payload_digest([dict(row) for row in v177_rows])
    if expected_v177_dataset_digest and v177_dataset_digest != expected_v177_dataset_digest:
        failures.append("v177_unexpected_dataset_digest")
    v177_reported_dataset = _mapping(v177_report.get("dataset"))
    if (
        str(v177_reported_dataset.get("dataset_digest") or "")
        and str(v177_reported_dataset.get("dataset_digest") or "") != v177_dataset_digest
    ):
        failures.append("v177_reported_dataset_digest_mismatch")
    if (
        _int(v177_reported_dataset.get("row_count"), default=-1) >= 0
        and _int(v177_reported_dataset.get("row_count"), default=-1) != len(v177_rows)
    ):
        failures.append("v177_reported_row_count_mismatch")
    v177_lifecycle = _lifecycle_validation(
        v177_report,
        policy="v179_v177_lifecycle_validation",
        required_contract_true_fields=(
            "diagnostics_only",
            "source_identity_metadata_only",
            "current_and_next_public_fields_are_dataset_inputs",
            "short_horizon_outcomes_are_diagnostic_targets_only",
        ),
        required_contract_false_fields=(
            "training_allowed",
            "fit_allowed",
            "runtime_artifact_allowed",
            "runtime_action_change_allowed",
            "shadow_or_live_eval_allowed",
            "promotion_allowed",
            "gate_relaxation_allowed",
            "replay_viewer_schema_change_allowed",
        ),
    )
    if v177_lifecycle.get("passed") is not True:
        failures.append("v177_lifecycle_not_diagnostics_only")
    v177_row_schema = validate_v177_transition_rows(v177_rows)
    if v177_row_schema.get("passed") is not True:
        failures.append("v177_transition_rows_invalid")
    v177_leakage = v177_trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in v177_rows]
    )
    if v177_leakage.get("passed") is not True:
        failures.append("v177_trainable_leakage_failed")
    return {
        "policy": "m3_carrion_survivor_continuation_v179_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v172_classification": expected_v172_classification,
        "observed_v172_classification": observed_v172_classification,
        "expected_v172_exact_digest": expected_v172_exact_digest,
        "observed_v172_exact_digest": observed_v172_exact,
        "v172_exact_digest_validation": v172_exact,
        "expected_v172_dataset_digest": expected_v172_dataset_digest,
        "observed_v172_dataset_digest": v172_dataset_digest,
        "expected_v172_row_count": int(expected_v172_row_count),
        "observed_v172_row_count": len(v172_rows),
        "v172_lifecycle_validation": v172_lifecycle,
        "v172_row_schema_validation": v172_row_schema,
        "v172_leakage_scan": v172_leakage,
        "v172_source_row_integrity": source_row_integrity,
        "observed_v177_classification": observed_v177_classification,
        "v177_exact_digest_validation": v177_exact,
        "expected_v177_report_exact_digest": expected_v177_report_exact_digest,
        "observed_v177_report_exact_digest": observed_v177_exact,
        "expected_v177_report_exact_digest_provided": bool(
            expected_v177_report_exact_digest
        ),
        "expected_v177_dataset_digest": expected_v177_dataset_digest,
        "observed_v177_dataset_digest": v177_dataset_digest,
        "expected_v177_dataset_digest_provided": bool(expected_v177_dataset_digest),
        "v177_source_digests_pinned": bool(
            expected_v177_report_exact_digest and expected_v177_dataset_digest
        ),
        "v177_lifecycle_validation": v177_lifecycle,
        "v177_row_schema_validation": v177_row_schema,
        "v177_leakage_scan": v177_leakage,
        "v176_v177_shard_plan_digest_used_as_historical_provenance": (
            EXPECTED_V177_SHARD_PLAN_DIGEST
        ),
    }


def v179_source_row_integrity_scan(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    branch_ids: Counter[str] = Counter()
    source_identity_branch_ids: dict[tuple[object, ...], set[str]] = defaultdict(set)
    source_record_digest_branch_ids: dict[str, set[str]] = defaultdict(set)
    for row_index, row in enumerate(rows):
        metadata = _mapping(row.get("metadata"))
        branch_id = str(metadata.get("branch_id") or "")
        if branch_id:
            branch_ids.update([branch_id])
        else:
            _append_row_failure(failures, row_index, "missing_branch_id")
        for field in ("seed", "branch_tick", "agent_id", "line_number"):
            if _int(metadata.get(field), default=-1) < 0:
                _append_row_failure(failures, row_index, f"invalid_{field}")
        if not str(metadata.get("source_path") or ""):
            _append_row_failure(failures, row_index, "missing_source_path")
        source_record_digest = str(metadata.get("source_record_digest") or "")
        branch_state_digest = str(metadata.get("branch_state_digest") or "")
        if not source_record_digest:
            _append_row_failure(failures, row_index, "missing_source_record_digest")
        if not branch_state_digest:
            _append_row_failure(failures, row_index, "missing_branch_state_digest")
        source_identity = (
            _int(metadata.get("seed"), default=-1),
            str(metadata.get("source_path") or ""),
            _int(metadata.get("line_number"), default=-1),
            _int(metadata.get("branch_tick"), default=-1),
            _int(metadata.get("agent_id"), default=-1),
        )
        if branch_id:
            if (
                source_identity[0] >= 0
                and source_identity[1]
                and source_identity[2] >= 0
                and source_identity[3] >= 0
                and source_identity[4] >= 0
            ):
                source_identity_branch_ids[source_identity].add(branch_id)
            if source_record_digest:
                source_record_digest_branch_ids[source_record_digest].add(branch_id)
        if not _public_actions(row):
            _append_row_failure(failures, row_index, "empty_public_action_mask")
        if metadata.get("source_identity_used_as_trainable_input") is not False:
            _append_row_failure(
                failures,
                row_index,
                "source_identity_used_as_trainable_input",
            )
        if (
            metadata.get("runtime_requested_or_resolved_action_used_as_trainable_input")
            is not False
        ):
            _append_row_failure(
                failures,
                row_index,
                "runtime_requested_or_resolved_action_used_as_trainable_input",
            )
    for branch_id, count in sorted(branch_ids.items()):
        if count > 1:
            failures.append(
                {
                    "reason": "duplicate_branch_id",
                    "branch_id": branch_id,
                    "count": int(count),
                }
            )
    failures.extend(
        _multi_branch_identity_failures(
            reason="source_materialization_identity_reused_across_branches",
            branch_ids_by_identity=source_identity_branch_ids,
        )
    )
    failures.extend(
        _multi_branch_identity_failures(
            reason="source_record_digest_reused_across_branches",
            branch_ids_by_identity=source_record_digest_branch_ids,
        )
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v179_source_row_integrity_scan_v1",
        "passed": bool(rows) and not failures,
        "failure_count": len(failures),
        "failures": failures[:128],
        "row_count": len(rows),
        "branch_id_count": len(branch_ids),
    }


def _multi_branch_identity_failures(
    *,
    reason: str,
    branch_ids_by_identity: Mapping[object, set[str]],
) -> list[dict[str, object]]:
    failures = []
    for identity, branch_ids in branch_ids_by_identity.items():
        if len(branch_ids) > 1:
            failures.append(
                {
                    "reason": reason,
                    "identity": repr(identity),
                    "branch_count": len(branch_ids),
                    "branch_ids": sorted(branch_ids)[:16],
                }
            )
    return failures[:32]


def build_v179_plan_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    support_provenance_seeds: Sequence[int],
    branches_per_seed: int,
    max_forced_actions_per_branch: int | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidates_by_seed: dict[int, list[tuple[int, Mapping[str, object]]]] = defaultdict(list)
    failures: list[dict[str, object]] = []
    support_seed_set = {int(seed) for seed in support_provenance_seeds}
    max_actions = (
        None
        if max_forced_actions_per_branch is None
        else max(1, int(max_forced_actions_per_branch))
    )
    for row_index, row in enumerate(rows):
        metadata = _mapping(row.get("metadata"))
        seed = _int(metadata.get("seed"), default=-1)
        if seed not in support_seed_set:
            continue
        actions = _public_actions(row, max_actions=max_actions)
        if not actions:
            _append_row_failure(failures, row_index, "no_public_actions")
            continue
        if max_actions is not None and len(actions) < max_actions:
            continue
        candidates_by_seed[seed].append((row_index, row))
    plan_rows: list[dict[str, object]] = []
    per_seed_selected: Counter[int] = Counter()
    for seed in sorted(support_seed_set):
        candidates = sorted(
            candidates_by_seed.get(seed, []),
            key=lambda item: _candidate_sort_key(item[0], item[1]),
        )
        for row_index, row in candidates[: max(0, int(branches_per_seed))]:
            plan_rows.append(
                _plan_row_from_v172_row(
                    row,
                    row_index=row_index,
                    max_forced_actions_per_branch=max_actions,
                )
            )
            per_seed_selected.update([seed])
    for priority, row in enumerate(plan_rows, start=1):
        row["priority"] = priority
    branch_ids = [str(row.get("branch_id") or "") for row in plan_rows]
    duplicate_branch_ids = sorted(
        branch_id
        for branch_id, count in Counter(branch_ids).items()
        if branch_id and count > 1
    )
    for branch_id in duplicate_branch_ids:
        failures.append({"reason": "duplicate_selected_branch_id", "branch_id": branch_id})
    forced_actions = sorted(
        {action for row in plan_rows for action in _ordered_actions(row.get("candidate_forced_actions"))},
        key=_action_order,
    )
    passed = (
        not failures
        and len(plan_rows) >= V178_DEFAULT_MIN_BRANCH_COUNT
        and len(per_seed_selected) >= V178_DEFAULT_MIN_SEED_COUNT
        and len(forced_actions) >= V178_DEFAULT_MIN_FORCED_ACTION_COUNT
    )
    return (
        plan_rows,
        {
            "policy": "m3_carrion_survivor_continuation_v179_plan_generation_v1",
            "passed": passed,
            "failure_count": len(failures),
            "failures": failures[:96],
            "source_row_count": len(rows),
            "branches_per_seed": int(branches_per_seed),
            "max_forced_actions_per_branch": max_actions,
            "selected_plan_row_count": len(plan_rows),
            "selected_seed_count": len(per_seed_selected),
            "selected_by_seed": {
                str(seed): int(count)
                for seed, count in sorted(per_seed_selected.items())
            },
            "candidate_forced_action_count": len(forced_actions),
            "candidate_forced_actions": forced_actions,
            "v178_default_minimums": _v178_default_minimums(),
            "plan_digest": stable_payload_digest(plan_rows),
            "training_authorized": False,
            "runtime_artifact_authorized": False,
        },
    )


def transition_row_support_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    seeds: Counter[int] = Counter()
    branches: set[str] = set()
    actions: Counter[str] = Counter()
    for row in rows:
        metadata = _mapping(row.get("metadata"))
        seed = _int(metadata.get("seed"), default=-1)
        if seed >= 0:
            seeds.update([seed])
        branch_id = str(metadata.get("branch_id") or "")
        if branch_id:
            branches.add(branch_id)
        action = str(row.get("forced_action") or "")
        if action:
            actions.update([action])
    observed = {
        "row_count": len(rows),
        "seed_count": len(seeds),
        "branch_count": len(branches),
        "forced_action_count": len(actions),
    }
    minimums = _v178_default_minimums()
    failures = [
        {
            "reason": f"{field}_below_v178_default_minimum",
            "observed": int(observed[field]),
            "minimum": int(minimum),
        }
        for field, minimum in minimums.items()
        if int(observed[field]) < int(minimum)
    ]
    return {
        "policy": "m3_carrion_survivor_continuation_v179_support_summary_v1",
        "passed": bool(rows) and not failures,
        "failure_count": len(failures),
        "failures": failures,
        "minimums": minimums,
        "observed": observed,
        "row_counts_by_seed": {
            str(seed): int(count) for seed, count in sorted(seeds.items())
        },
        "forced_action_counts": dict(
            sorted(actions.items(), key=lambda item: _action_order(item[0]))
        ),
        "v178_default_support_thresholds_met": bool(rows) and not failures,
    }


def _v179_transition_dataset_metrics(
    *,
    selected: Sequence[object],
    materialization: Mapping[str, object],
    transition_rows: Sequence[Mapping[str, object]],
    verify_replay: bool,
) -> dict[str, object]:
    metrics = dict(
        transition_dataset_metrics(
            selected=selected,
            materialization=materialization,
            transition_rows=transition_rows,
            verify_replay=bool(verify_replay),
        )
    )
    metrics["v179_replay_verification_required_for_support_ready"] = True
    if not verify_replay:
        metrics["all_replays_verified"] = False
        metrics["compact_transition_support_ready"] = False
        metrics["v179_replay_verification_failure_reason"] = (
            "replay_verification_disabled"
        )
    return metrics


def _plan_row_from_v172_row(
    row: Mapping[str, object],
    *,
    row_index: int,
    max_forced_actions_per_branch: int | None,
) -> dict[str, object]:
    metadata = _mapping(row.get("metadata"))
    return {
        "schema_version": "m3_carrion_survivor_continuation_v177_exact_branch_replay_shard_row_v1",
        "route": "exact_branch_replay_expansion",
        "priority": 0,
        "seed": _int(metadata.get("seed"), default=-1),
        "branch_tick": _int(metadata.get("branch_tick"), default=-1),
        "agent_id": _int(metadata.get("agent_id"), default=-1),
        "branch_id": str(metadata.get("branch_id") or ""),
        "source_path": str(metadata.get("source_path") or ""),
        "line_number": _int(metadata.get("line_number"), default=-1),
        "row_index": int(row_index),
        "failed_safe_action": _single_safe_action(row),
        "failure_types": ["v179_support_expansion"],
        "candidate_forced_actions": _public_actions(
            row,
            max_actions=max_forced_actions_per_branch,
        ),
        "source_v172_dataset_row_index": int(row_index),
        "source_v172_branch_state_digest": metadata.get("branch_state_digest"),
        "source_record_digest": metadata.get("source_record_digest"),
        "replay_expansion_goal": (
            "expand exact current/forced/next public transition rows above "
            "v178 default support thresholds"
        ),
        "training_authorized": False,
        "runtime_artifact_authorized": False,
    }


def _tag_v179_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    tagged = []
    for row in rows:
        payload = deepcopy(dict(row))
        payload["row_origin"] = (
            "v179_exact_branch_replay_from_v172_support_expansion"
        )
        metadata = dict(_mapping(payload.get("metadata")))
        metadata["source_v179_expansion"] = True
        metadata["source_identity_used_for_exact_materialization_only"] = True
        metadata["source_identity_used_as_trainable_input"] = False
        metadata["runtime_requested_or_resolved_action_used_as_trainable_input"] = False
        metadata["current_or_future_outcome_used_as_trainable_input"] = False
        metadata["diagnostic_target_used_as_trainable_input"] = False
        payload["metadata"] = metadata
        tagged.append(payload)
    return tagged


def _classification(
    *,
    source_validation: Mapping[str, object],
    plan_generation: Mapping[str, object],
    selection: Mapping[str, object],
    materialization: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    metrics: Mapping[str, object],
    support_summary: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if plan_generation.get("passed") is not True:
        return prefix + "plan_invalid_closed_no_training"
    if selection.get("passed") is not True:
        return prefix + "selection_invalid_closed_no_training"
    if materialization.get("passed") is not True:
        return prefix + "materialization_blocked_no_training"
    if metrics.get("all_replays_verified") is not True:
        return prefix + "replay_not_deterministic_closed_no_training"
    if metrics.get("all_forced_actions_used") is not True:
        return prefix + "forced_action_not_used_closed_no_training"
    if (
        leakage_scan.get("passed") is not True
        or row_schema_validation.get("passed") is not True
    ):
        return prefix + "compact_transition_rows_invalid_closed_no_training"
    if support_summary.get("passed") is True:
        return V179_SUPPORT_READY_CLASSIFICATION
    return prefix + "compact_transition_rows_support_limited_no_training"


def _route_recommendation(
    *,
    classification: str,
    source_validation: Mapping[str, object],
) -> dict[str, object]:
    ready = classification == V179_SUPPORT_READY_CLASSIFICATION
    v177_source_pinned = source_validation.get("v177_source_digests_pinned") is True
    return {
        "policy": "m3_carrion_survivor_continuation_v179_route_recommendation_v1",
        "recommended_next_route": (
            "v178_transition_row_dataset_audit_default_thresholds_no_training"
            if ready and v177_source_pinned
            else "rerun_v179_with_expected_v177_source_digests_before_v178_audit"
            if ready
            else "repair_v179_transition_row_expansion_before_capacity_work"
        ),
        "v178_default_threshold_audit_recommended": ready and v177_source_pinned,
        "v177_source_digests_pinned": v177_source_pinned,
        "transition_row_training_authorized": False,
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
        "v172_selector_evidence_is_not_trainable_transition_rows": True,
        "current_and_next_public_fields_are_materialized_by_exact_replay": True,
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


def _lifecycle_validation(
    report: Mapping[str, object],
    *,
    policy: str,
    required_contract_true_fields: Sequence[str] = (),
    required_contract_false_fields: Sequence[str] = (),
) -> dict[str, object]:
    failures = []
    for field in (
        "training_ran",
        "training_authorized",
        "scorer_retraining_ran",
        "scorer_retraining_authorized",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "runtime_observation_schema_changed",
        "runtime_policy_changed",
        "shadow_eval_ran",
        "live_ab_ran",
        "live_ab_allowed",
        "promotion_authorized",
        "replay_viewer_schema_changed",
    ):
        if report.get(field) is not False:
            failures.append(
                {"field": field, "expected": False, "observed": report.get(field)}
            )
    for field in ("diagnostics_only", "non_promoted"):
        if report.get(field) is not True:
            failures.append(
                {"field": field, "expected": True, "observed": report.get(field)}
            )
    if required_contract_true_fields or required_contract_false_fields:
        contract = _mapping(report.get("contract"))
        if not contract:
            failures.append(
                {
                    "field": "contract",
                    "expected": "mapping",
                    "observed": report.get("contract"),
                }
            )
        for field in required_contract_true_fields:
            if contract.get(field) is not True:
                failures.append(
                    {
                        "field": f"contract.{field}",
                        "expected": True,
                        "observed": contract.get(field),
                    }
                )
        for field in required_contract_false_fields:
            if contract.get(field) is not False:
                failures.append(
                    {
                        "field": f"contract.{field}",
                        "expected": False,
                        "observed": contract.get(field),
                    }
                )
    return {
        "policy": policy,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def _empty_plan_generation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v179_plan_generation_v1",
        "passed": False,
        "reason": reason,
        "selected_plan_row_count": 0,
        "selected_seed_count": 0,
        "candidate_forced_action_count": 0,
        "training_authorized": False,
        "runtime_artifact_authorized": False,
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


def _candidate_sort_key(row_index: int, row: Mapping[str, object]) -> tuple[object, ...]:
    metadata = _mapping(row.get("metadata"))
    return (
        _int(metadata.get("branch_tick"), default=10**9),
        -len(_public_actions(row)),
        str(metadata.get("branch_id") or ""),
        int(row_index),
    )


def _public_actions(
    row: Mapping[str, object],
    *,
    max_actions: int | None = None,
) -> list[str]:
    mask = _mapping(row.get("public_action_mask"))
    actions = [action for action in ACTION_NAMES if mask.get(action) is True]
    if max_actions is None:
        return actions
    return actions[: max(0, int(max_actions))]


def _single_safe_action(row: Mapping[str, object]) -> str | None:
    safe = _ordered_actions(row.get("safe_action_set"))
    return safe[0] if len(safe) == 1 else None


def _ordered_actions(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return sorted(
        {str(action) for action in value if str(action) in ACTION_NAMES},
        key=_action_order,
    )


def _v178_default_minimums() -> dict[str, int]:
    return {
        "row_count": V178_DEFAULT_MIN_ROW_COUNT,
        "seed_count": V178_DEFAULT_MIN_SEED_COUNT,
        "branch_count": V178_DEFAULT_MIN_BRANCH_COUNT,
        "forced_action_count": V178_DEFAULT_MIN_FORCED_ACTION_COUNT,
    }


def _append_row_failure(
    failures: list[dict[str, object]],
    row_index: int,
    reason: str,
    **extra: object,
) -> None:
    payload = {"row_index": row_index, "reason": reason}
    payload.update(extra)
    failures.append(payload)


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _json_round_trip_digest(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    )
