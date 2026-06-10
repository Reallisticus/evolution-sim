from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import gzip
import json
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    _dominant_count_share,
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
    load_v154_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v163_tied_set_branch_target_expansion import (
    DEFAULT_TICKS,
    V163SelectedBranchPoint,
    _bool_action_mask,
    _candidate_set_key,
    _optional_string,
    _outcome_key,
    _record_materialization_payload,
    evaluate_materialized_branch_points,
    materialize_selected_branch_points,
)
from evolution_sim.mind.carrion_survivor_continuation_v166_source_split_action_value_scorer import (
    DEFAULT_V165_DATASET_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v170_diagnostic_portfolio_matrix import (
    BASELINE_ZERO_HIT_SEEDS,
    DEFAULT_OUTPUT_PATH as DEFAULT_V170_REPORT_PATH,
    DEFAULT_SHARD_PLAN_OUTPUT_PATH as DEFAULT_V170_SHARD_PLAN_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_SCHEMA_VERSION,
    REPLAY_EXPANSION_TARGET_SEEDS,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v171_replay_expansion_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v171_replay_expansion_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_ROW_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v171_replay_expansion_dataset_row_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_DATASET_FEATURE_POLICY = (
    "public_observation_action_mask_v171_replay_expansion_v1"
)
EXPECTED_V170_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix_"
    "set_valued_ranking_support_partial_no_runtime"
)
EXPECTED_V170_EXACT_DIGEST = (
    "ad44a0e89fe9248297a5ec6a3d158c70db6947d76ceb58e9a0ce32eca28da366"
)
EXPECTED_V170_SHARD_PLAN_DIGEST = (
    "599a5dbab5635124e9a024a0140140fd529b90b20f3d8ceed06e567fe09cd840"
)
DEFAULT_SHARD_DIR = Path("output/mind/shards")
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v171-carrion-survivor-continuation-replay-expansion.json"
)
DEFAULT_DATASET_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v171-carrion-survivor-continuation-replay-expansion.jsonl"
)
FORBIDDEN_TRAINABLE_PAYLOAD_TOKENS = (
    "seed",
    "fixture",
    "branch",
    "tick",
    "agent_id",
    "path",
    "digest",
    "provenance",
    "private",
    "future",
    "outcome",
    "label",
    "runtime",
    "requested",
    "resolved",
    "target_safe",
    "safe_action",
)


class CarrionSurvivorContinuationV171ReplayExpansionError(ValueError):
    pass


def run_carrion_survivor_continuation_v171_replay_expansion(
    *,
    v170_report_path: str | Path = DEFAULT_V170_REPORT_PATH,
    shard_plan_path: str | Path = DEFAULT_V170_SHARD_PLAN_PATH,
    v165_dataset_path: str | Path | None = None,
    shard_id: str | None = None,
    seed_include: int | None = None,
    branch_windows: Sequence[str | tuple[int, int]] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    dataset_output_path: str | Path = DEFAULT_DATASET_OUTPUT_PATH,
    fail_on_partial_shard: bool = False,
    merge_shards: bool = False,
    merge_shard_reports: Sequence[str | Path] | None = None,
    merge_shard_datasets: Sequence[str | Path] | None = None,
    allow_partial_shard_evidence: bool = False,
    expected_v170_exact_digest: str | None = EXPECTED_V170_EXACT_DIGEST,
    expected_v170_classification: str = EXPECTED_V170_CLASSIFICATION,
    expected_v170_shard_plan_digest: str | None = EXPECTED_V170_SHARD_PLAN_DIGEST,
) -> dict[str, object]:
    if merge_shards:
        return merge_v171_shards(
            v170_report_path=v170_report_path,
            shard_plan_path=shard_plan_path,
            merge_shard_reports=merge_shard_reports or (),
            merge_shard_datasets=merge_shard_datasets or (),
            output_path=output_path,
            dataset_output_path=dataset_output_path,
            allow_partial_shard_evidence=allow_partial_shard_evidence,
            expected_v170_exact_digest=expected_v170_exact_digest,
            expected_v170_classification=expected_v170_classification,
            expected_v170_shard_plan_digest=expected_v170_shard_plan_digest,
        )
    if shard_id:
        return execute_v171_shard(
            v170_report_path=v170_report_path,
            shard_plan_path=shard_plan_path,
            v165_dataset_path=v165_dataset_path,
            shard_id=shard_id,
            seed_include=seed_include,
            branch_windows=branch_windows,
            output_path=output_path,
            dataset_output_path=dataset_output_path,
            fail_on_partial_shard=fail_on_partial_shard,
            expected_v170_exact_digest=expected_v170_exact_digest,
            expected_v170_classification=expected_v170_classification,
            expected_v170_shard_plan_digest=expected_v170_shard_plan_digest,
        )
    return execute_all_v171_shards_and_merge(
        v170_report_path=v170_report_path,
        shard_plan_path=shard_plan_path,
        v165_dataset_path=v165_dataset_path,
        output_path=output_path,
        dataset_output_path=dataset_output_path,
        expected_v170_exact_digest=expected_v170_exact_digest,
        expected_v170_classification=expected_v170_classification,
        expected_v170_shard_plan_digest=expected_v170_shard_plan_digest,
    )


def execute_v171_shard(
    *,
    v170_report_path: str | Path = DEFAULT_V170_REPORT_PATH,
    shard_plan_path: str | Path = DEFAULT_V170_SHARD_PLAN_PATH,
    v165_dataset_path: str | Path | None = None,
    shard_id: str,
    seed_include: int | None,
    branch_windows: Sequence[str | tuple[int, int]] | None,
    output_path: str | Path,
    dataset_output_path: str | Path,
    fail_on_partial_shard: bool = False,
    expected_v170_exact_digest: str | None = EXPECTED_V170_EXACT_DIGEST,
    expected_v170_classification: str = EXPECTED_V170_CLASSIFICATION,
    expected_v170_shard_plan_digest: str | None = EXPECTED_V170_SHARD_PLAN_DIGEST,
    attempt_branch_replay: bool = True,
    verify_replay: bool = True,
) -> dict[str, object]:
    v170_report = load_json_report(v170_report_path)
    shard_rows = load_v170_shard_plan(shard_plan_path)
    source_validation = validate_v171_sources(
        v170_report=v170_report,
        shard_plan_rows=shard_rows,
        expected_v170_exact_digest=expected_v170_exact_digest,
        expected_v170_classification=expected_v170_classification,
        expected_v170_shard_plan_digest=expected_v170_shard_plan_digest,
    )
    plan_row = _find_shard_plan_row(shard_rows, shard_id=shard_id)
    shard_validation = validate_shard_request(
        plan_row=plan_row,
        shard_id=shard_id,
        seed_include=seed_include,
        branch_windows=branch_windows,
    )
    selected: list[V163SelectedBranchPoint] = []
    selection: dict[str, object] = _empty_selection_report(
        shard_id=shard_id,
        seed=seed_include,
        reason="source_or_shard_validation_failed",
    )
    materialization: dict[str, object] = _empty_materialization_report(
        reason="source_or_shard_validation_failed"
    )
    branch_results: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    if source_validation.get("passed") is True and shard_validation.get("passed") is True:
        v165_rows = load_v154_dataset(_resolve_v165_dataset_path(v170_report, v165_dataset_path))
        selected = select_v171_shard_branch_points(
            plan_row=plan_row,
            v165_rows=v165_rows,
            branch_windows=branch_windows,
        )
        selection = selection_report(selected=selected, plan_row=plan_row)
        if selected and attempt_branch_replay:
            materialized, materialization = materialize_selected_branch_points(
                selected,
                ticks=DEFAULT_TICKS,
            )
            if materialization.get("materialized_branch_point_count", 0):
                _trim_materialized_branch_histories(materialized)
                branch_results = evaluate_materialized_branch_points(
                    materialized,
                    verify_replay=bool(verify_replay),
                )
                dataset_rows = build_v171_dataset_rows(branch_results)
                branch_results = _strip_dataset_feature_payloads(branch_results)
        elif selected:
            materialization = _empty_materialization_report(
                reason="branch_replay_disabled",
                selected_branch_point_count=len(selected),
            )
    leakage_scan = trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in dataset_rows]
    )
    metrics = replay_expansion_metrics(
        selected=selected,
        materialization=materialization,
        branch_results=branch_results,
        dataset_rows=dataset_rows,
        v170_report=v170_report,
    )
    partial_status = partial_shard_status(
        selected=selected,
        materialization=materialization,
        branch_results=branch_results,
        fail_on_partial_shard=fail_on_partial_shard,
    )
    classification = _classification(
        source_validation=source_validation,
        shard_validation=shard_validation,
        leakage_scan=leakage_scan,
        metrics=metrics,
        partial_status=partial_status,
    )
    _write_jsonl(dataset_output_path, dataset_rows)
    report = {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION,
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY,
        "mode": "shard",
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v170_report": str(v170_report_path),
            "shard_plan": str(shard_plan_path),
            "v165_dataset": str(_resolve_v165_dataset_path(v170_report, v165_dataset_path)),
            "shard_id": shard_id,
            "seed_include": seed_include,
            "branch_windows": _window_strings(_effective_windows(plan_row, branch_windows)),
            "output": str(output_path),
            "dataset_output": str(dataset_output_path),
            "fail_on_partial_shard": bool(fail_on_partial_shard),
            "attempt_branch_replay": bool(attempt_branch_replay),
            "verify_replay": bool(verify_replay),
        },
        "source_validation": source_validation,
        "shard_validation": shard_validation,
        "source_row_policy": _source_row_policy(),
        "shard_plan_row": dict(plan_row) if plan_row else None,
        "selection": selection,
        "branch_materialization": materialization,
        "branch_results": branch_results,
        "metrics": metrics,
        "leakage_scan": leakage_scan,
        "partial_shard_status": partial_status,
        "dataset": {
            "path": str(dataset_output_path),
            "row_count": len(dataset_rows),
            "dataset_digest": stable_payload_digest(dataset_rows),
            "trainable_payload_policy": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_DATASET_FEATURE_POLICY,
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "shard_digest": stable_payload_digest(
            {
                "shard_id": shard_id,
                "dataset_digest": stable_payload_digest(dataset_rows),
                "metrics": metrics,
                "classification": classification,
            }
        ),
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def execute_all_v171_shards_and_merge(
    *,
    v170_report_path: str | Path = DEFAULT_V170_REPORT_PATH,
    shard_plan_path: str | Path = DEFAULT_V170_SHARD_PLAN_PATH,
    v165_dataset_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    dataset_output_path: str | Path = DEFAULT_DATASET_OUTPUT_PATH,
    expected_v170_exact_digest: str | None = EXPECTED_V170_EXACT_DIGEST,
    expected_v170_classification: str = EXPECTED_V170_CLASSIFICATION,
    expected_v170_shard_plan_digest: str | None = EXPECTED_V170_SHARD_PLAN_DIGEST,
) -> dict[str, object]:
    shard_rows = load_v170_shard_plan(shard_plan_path)
    shard_reports: list[Path] = []
    shard_datasets: list[Path] = []
    for row in sorted(shard_rows, key=lambda item: (_int(item.get("seed")), str(item.get("shard_id")))):
        shard_id = str(row.get("shard_id"))
        report_path = DEFAULT_SHARD_DIR / f"{shard_id}.json"
        dataset_path = DEFAULT_SHARD_DIR / f"{shard_id}.jsonl"
        execute_v171_shard(
            v170_report_path=v170_report_path,
            shard_plan_path=shard_plan_path,
            v165_dataset_path=v165_dataset_path,
            shard_id=shard_id,
            seed_include=_int(row.get("seed")),
            branch_windows=_window_strings(_plan_windows(row)),
            output_path=report_path,
            dataset_output_path=dataset_path,
            fail_on_partial_shard=False,
            expected_v170_exact_digest=expected_v170_exact_digest,
            expected_v170_classification=expected_v170_classification,
            expected_v170_shard_plan_digest=expected_v170_shard_plan_digest,
        )
        shard_reports.append(report_path)
        shard_datasets.append(dataset_path)
    return merge_v171_shards(
        v170_report_path=v170_report_path,
        shard_plan_path=shard_plan_path,
        merge_shard_reports=shard_reports,
        merge_shard_datasets=shard_datasets,
        output_path=output_path,
        dataset_output_path=dataset_output_path,
        allow_partial_shard_evidence=False,
        expected_v170_exact_digest=expected_v170_exact_digest,
        expected_v170_classification=expected_v170_classification,
        expected_v170_shard_plan_digest=expected_v170_shard_plan_digest,
    )


def merge_v171_shards(
    *,
    v170_report_path: str | Path = DEFAULT_V170_REPORT_PATH,
    shard_plan_path: str | Path = DEFAULT_V170_SHARD_PLAN_PATH,
    merge_shard_reports: Sequence[str | Path],
    merge_shard_datasets: Sequence[str | Path],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    dataset_output_path: str | Path = DEFAULT_DATASET_OUTPUT_PATH,
    allow_partial_shard_evidence: bool = False,
    expected_v170_exact_digest: str | None = EXPECTED_V170_EXACT_DIGEST,
    expected_v170_classification: str = EXPECTED_V170_CLASSIFICATION,
    expected_v170_shard_plan_digest: str | None = EXPECTED_V170_SHARD_PLAN_DIGEST,
) -> dict[str, object]:
    v170_report = load_json_report(v170_report_path)
    shard_plan_rows = load_v170_shard_plan(shard_plan_path)
    source_validation = validate_v171_sources(
        v170_report=v170_report,
        shard_plan_rows=shard_plan_rows,
        expected_v170_exact_digest=expected_v170_exact_digest,
        expected_v170_classification=expected_v170_classification,
        expected_v170_shard_plan_digest=expected_v170_shard_plan_digest,
    )
    shard_reports = [load_json_report(path) for path in merge_shard_reports]
    shard_dataset_rows_by_path = [
        load_jsonl_dataset(path) for path in merge_shard_datasets
    ]
    dataset_rows = [
        row
        for dataset_rows_for_path in shard_dataset_rows_by_path
        for row in dataset_rows_for_path
    ]
    _write_jsonl(dataset_output_path, dataset_rows)
    merge_validation = validate_merge_inputs(
        shard_plan_rows=shard_plan_rows,
        shard_reports=shard_reports,
        merge_shard_reports=merge_shard_reports,
        merge_shard_datasets=merge_shard_datasets,
        allow_partial_shard_evidence=allow_partial_shard_evidence,
    )
    leakage_scan = trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in dataset_rows]
    )
    metrics = merged_replay_expansion_metrics(
        shard_reports=shard_reports,
        dataset_rows=dataset_rows,
        v170_report=v170_report,
    )
    partial_status = merged_partial_status(
        merge_validation=merge_validation,
        shard_reports=shard_reports,
        allow_partial_shard_evidence=allow_partial_shard_evidence,
    )
    classification = _classification(
        source_validation=source_validation,
        shard_validation=merge_validation,
        leakage_scan=leakage_scan,
        metrics=metrics,
        partial_status=partial_status,
    )
    report = {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION,
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY,
        "mode": "merge",
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v170_report": str(v170_report_path),
            "shard_plan": str(shard_plan_path),
            "merge_shard_reports": [str(path) for path in merge_shard_reports],
            "merge_shard_datasets": [str(path) for path in merge_shard_datasets],
            "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
            "output": str(output_path),
            "dataset_output": str(dataset_output_path),
        },
        "source_validation": source_validation,
        "merge_validation": merge_validation,
        "source_row_policy": _source_row_policy(),
        "shard_reports": [_shard_report_summary(report) for report in shard_reports],
        "metrics": metrics,
        "leakage_scan": leakage_scan,
        "partial_shard_status": partial_status,
        "dataset": {
            "path": str(dataset_output_path),
            "row_count": len(dataset_rows),
            "dataset_digest": stable_payload_digest(dataset_rows),
            "trainable_payload_policy": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_DATASET_FEATURE_POLICY,
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "merge_digest": stable_payload_digest(
            {
                "dataset_digest": stable_payload_digest(dataset_rows),
                "metrics": metrics,
                "classification": classification,
                "shard_report_digests": [
                    report.get("exact_digest") for report in shard_reports
                ],
            }
        ),
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v171_sources(
    *,
    v170_report: Mapping[str, object],
    shard_plan_rows: Sequence[Mapping[str, object]],
    expected_v170_exact_digest: str | None = EXPECTED_V170_EXACT_DIGEST,
    expected_v170_classification: str = EXPECTED_V170_CLASSIFICATION,
    expected_v170_shard_plan_digest: str | None = EXPECTED_V170_SHARD_PLAN_DIGEST,
) -> dict[str, object]:
    failures: list[str] = []
    observed_classification = str(
        _mapping(v170_report.get("classification")).get("primary") or ""
    )
    if (
        v170_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_SCHEMA_VERSION
    ):
        failures.append("v170_schema_version_mismatch")
    if v170_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_POLICY:
        failures.append("v170_policy_mismatch")
    if observed_classification != expected_v170_classification:
        failures.append("v170_unexpected_classification")
    exact = exact_digest_validation_report(v170_report)
    observed_exact = str(v170_report.get("exact_digest") or "")
    if exact.get("passed") is not True:
        failures.append("v170_exact_digest_mismatch")
    if expected_v170_exact_digest and observed_exact != expected_v170_exact_digest:
        failures.append("v170_unexpected_exact_digest")
    plan_digest = stable_payload_digest([dict(row) for row in shard_plan_rows])
    if expected_v170_shard_plan_digest and plan_digest != expected_v170_shard_plan_digest:
        failures.append("v170_shard_plan_digest_mismatch")
    reported_plan = _mapping(v170_report.get("shard_plan_output"))
    if reported_plan.get("shard_plan_digest") not in (None, plan_digest):
        failures.append("v170_reported_shard_plan_digest_mismatch")
    plan_rows = validate_v170_shard_plan_rows(shard_plan_rows)
    if plan_rows.get("passed") is not True:
        failures.append("v170_shard_plan_rows_invalid")
    lifecycle = _source_lifecycle_scan(v170_report)
    if lifecycle.get("passed") is not True:
        failures.append("v170_lifecycle_not_diagnostics_only")
    return {
        "policy": "m3_carrion_survivor_continuation_v171_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v170_classification": expected_v170_classification,
        "observed_v170_classification": observed_classification,
        "expected_v170_exact_digest": expected_v170_exact_digest,
        "observed_v170_exact_digest": observed_exact,
        "v170_exact_digest_validation": exact,
        "expected_v170_shard_plan_digest": expected_v170_shard_plan_digest,
        "observed_v170_shard_plan_digest": plan_digest,
        "v170_reported_shard_plan_digest": reported_plan.get("shard_plan_digest"),
        "shard_plan_row_validation": plan_rows,
        "lifecycle_validation": lifecycle,
    }


def validate_v170_shard_plan_rows(
    shard_plan_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    observed_seeds = []
    observed_ids = set()
    for row_index, row in enumerate(shard_plan_rows):
        seed = _int(row.get("seed"), default=-1)
        shard_id = str(row.get("shard_id") or "")
        observed_seeds.append(seed)
        if not shard_id:
            failures.append({"row_index": row_index, "reason": "missing_shard_id"})
        if shard_id in observed_ids:
            failures.append({"row_index": row_index, "reason": "duplicate_shard_id"})
        observed_ids.add(shard_id)
        if row.get("plan_only_not_evidence") is not True:
            failures.append({"row_index": row_index, "reason": "not_plan_only"})
        if row.get("long_branch_replay_ran") is not False:
            failures.append({"row_index": row_index, "reason": "v170_replay_already_ran"})
        if row.get("training_authorized") is not False:
            failures.append({"row_index": row_index, "reason": "training_authorized"})
        if row.get("runtime_action_selection_changed") is not False:
            failures.append(
                {"row_index": row_index, "reason": "runtime_action_selection_changed"}
            )
        if not _plan_windows(row):
            failures.append({"row_index": row_index, "reason": "missing_windows"})
        command = row.get("expected_command_shape")
        if not isinstance(command, list) or not command:
            failures.append({"row_index": row_index, "reason": "missing_command_shape"})
        elif "--shard-id" not in command or "--seed-include" not in command:
            failures.append({"row_index": row_index, "reason": "invalid_command_shape"})
    expected_seeds = [int(seed) for seed in REPLAY_EXPANSION_TARGET_SEEDS]
    if observed_seeds != expected_seeds:
        failures.append(
            {
                "reason": "target_seed_order_mismatch",
                "expected": expected_seeds,
                "observed": observed_seeds,
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v171_shard_plan_row_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_target_seeds": expected_seeds,
        "observed_target_seeds": observed_seeds,
        "row_count": len(shard_plan_rows),
    }


def validate_shard_request(
    *,
    plan_row: Mapping[str, object] | None,
    shard_id: str,
    seed_include: int | None,
    branch_windows: Sequence[str | tuple[int, int]] | None,
) -> dict[str, object]:
    failures: list[str] = []
    if not plan_row:
        failures.append("unknown_shard_id")
        plan_seed = None
    else:
        plan_seed = _int(plan_row.get("seed"))
        if str(plan_row.get("shard_id")) != shard_id:
            failures.append("shard_id_mismatch")
    if seed_include is not None and plan_seed is not None and int(seed_include) != int(plan_seed):
        failures.append("seed_include_mismatch")
    expected_windows = _window_strings(_plan_windows(plan_row or {}))
    requested_windows = _window_strings(_effective_windows(plan_row, branch_windows))
    if requested_windows and expected_windows and requested_windows != expected_windows:
        failures.append("branch_window_mismatch")
    return {
        "policy": "m3_carrion_survivor_continuation_v171_shard_request_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "shard_id": shard_id,
        "plan_seed": plan_seed,
        "seed_include": seed_include,
        "expected_branch_windows": expected_windows,
        "requested_branch_windows": requested_windows,
    }


def select_v171_shard_branch_points(
    *,
    plan_row: Mapping[str, object],
    v165_rows: Sequence[Mapping[str, object]],
    branch_windows: Sequence[str | tuple[int, int]] | None = None,
) -> list[V163SelectedBranchPoint]:
    seed = _int(plan_row.get("seed"))
    windows = _effective_windows(plan_row, branch_windows)
    source_paths = _source_paths_for_seed(v165_rows=v165_rows, seed=seed)
    selected: list[V163SelectedBranchPoint] = []
    seen: set[tuple[str, int]] = set()
    for source_path in source_paths:
        for line_number, record in iter_source_records(source_path):
            tick = _int(record.get("tick"), default=-1)
            if not any(start <= tick <= end for start, end in windows):
                continue
            action_mask = _bool_action_mask(
                record.get("public_action_mask") or record.get("action_mask")
            )
            legal_actions = tuple(action for action in ACTION_NAMES if action_mask.get(action) is True)
            if not legal_actions:
                continue
            key = (str(source_path), int(line_number))
            if key in seen:
                continue
            seen.add(key)
            branch_index = len(selected)
            agent_id = _int(record.get("agent_id"))
            branch_id = (
                f"v171-seed-{seed:03d}-source-line-{int(line_number):06d}-"
                f"tick-{tick}-agent-{agent_id}"
            )
            selected.append(
                V163SelectedBranchPoint(
                    branch_id=branch_id,
                    seed=seed,
                    ticks=DEFAULT_TICKS,
                    branch_tick=tick,
                    record_index=int(line_number),
                    branch_index=branch_index,
                    agent_id=agent_id,
                    source_path=str(source_path),
                    line_number=int(line_number),
                    runtime_requested_action=_optional_string(
                        record.get("requested_action")
                    )
                    or "",
                    runtime_resolved_action=_optional_string(
                        record.get("resolved_action")
                    )
                    or "",
                    predicted_action="",
                    nearest_neighbor_row_index=-1,
                    top_value_candidate_set=legal_actions,
                    action_mask=action_mask,
                    observation_input=deepcopy(dict(_mapping(record.get("observation_input")))),
                    observation_schema=_optional_string(record.get("observation_schema")),
                    observation_digest=_optional_string(record.get("observation_digest")),
                    source_record_digest=stable_payload_digest(
                        _record_materialization_payload(record)
                    ),
                    selection_rationale={
                        "v171_replay_expansion": True,
                        "source_plan_shard_id": str(plan_row.get("shard_id")),
                        "v171_selected_public_features": {
                            "observation_input": deepcopy(
                                dict(_mapping(record.get("observation_input")))
                            ),
                            "action_mask": action_mask,
                        },
                        "public_mask_legal_first_actions_only": True,
                        "legal_first_action_count": len(legal_actions),
                        "legal_first_action_set_key": _candidate_set_key(legal_actions),
                        "seed_tick_agent_path_digest_for_materialization_only": True,
                        "target_safe_action_used_as_trainable_input": False,
                        "future_outcome_used_as_trainable_input": False,
                        "runtime_requested_action_used_as_scorer_input": False,
                        "runtime_resolved_action_used_as_scorer_input": False,
                    },
                )
            )
    return selected


def iter_source_records(path: str | Path) -> list[tuple[int, Mapping[str, object]]]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    records: list[tuple[int, Mapping[str, object]]] = []
    with opener(source, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = payload.get("record") if isinstance(payload, Mapping) else None
            if not isinstance(record, Mapping):
                continue
            if not isinstance(record.get("tick"), int):
                continue
            if not isinstance(record.get("agent_id"), int):
                continue
            if not isinstance(record.get("observation_input"), Mapping):
                continue
            records.append((line_number, record))
    return records


def selection_report(
    *,
    selected: Sequence[V163SelectedBranchPoint],
    plan_row: Mapping[str, object],
) -> dict[str, object]:
    by_tick = Counter(point.branch_tick for point in selected)
    by_width = Counter(len(point.top_value_candidate_set) for point in selected)
    illegal_count = sum(
        len(_illegal_actions(point.action_mask))
        for point in selected
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v171_shard_branch_point_selection_v1",
        "shard_id": plan_row.get("shard_id"),
        "seed": plan_row.get("seed"),
        "branch_windows": _window_strings(_plan_windows(plan_row)),
        "selected_branch_point_count": len(selected),
        "branch_points_by_tick": {str(key): value for key, value in sorted(by_tick.items())},
        "legal_first_action_width_counts": {str(key): value for key, value in sorted(by_width.items())},
        "illegal_skipped_action_count": illegal_count,
        "selected_branch_points_sample": [_selected_payload(point) for point in selected[:24]],
        "all_source_identity_metadata_non_trainable": True,
    }


def build_v171_dataset_rows(
    branch_results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for result in branch_results:
        selected = _selected_public_features_from_result(result)
        action_mask = _bool_action_mask(selected.get("action_mask"))
        best_actions = _ordered_actions(result.get("best_outcome_actions"))
        candidate_runs = _list_of_mappings(result.get("candidate_runs"))
        target_classification = (
            "unique_replay_verified_winner"
            if len(best_actions) == 1
            else "multi_action_replay_verified_safe_set"
            if len(best_actions) > 1
            else "unresolved_replay_expansion_support"
        )
        row = {
            "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_ROW_SCHEMA_VERSION,
            "feature_policy_id": M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_DATASET_FEATURE_POLICY,
            "trainable_public_features": {
                "public_observation": deepcopy(dict(_mapping(selected.get("observation_input")))),
                "action_mask": action_mask,
            },
            "public_action_mask": action_mask,
            "action_value_targets": _action_value_targets(
                action_mask=action_mask,
                candidate_runs=candidate_runs,
                safe_action_set=best_actions,
            ),
            "safe_action_set": best_actions,
            "target_classification": target_classification,
            "robust_winner_action": best_actions[0] if len(best_actions) == 1 else None,
            "metadata": {
                "metadata_schema_version": "m3_carrion_survivor_continuation_v171_replay_expansion_metadata_v1",
                "branch_id": result.get("branch_id"),
                "seed": result.get("seed"),
                "fixture": result.get("fixture"),
                "branch_tick": result.get("branch_tick"),
                "agent_id": result.get("agent_id"),
                "source_path": result.get("source_path"),
                "line_number": result.get("line_number"),
                "source_record_digest": result.get("source_record_digest"),
                "materialized_record_digest": result.get("materialized_record_digest"),
                "branch_state_digest": result.get("branch_state_digest"),
                "replay_digests_by_action": {
                    str(run.get("forced_action")): run.get("replay_digest")
                    for run in candidate_runs
                    if str(run.get("forced_action")) in ACTION_NAMES
                },
                "source_identity_used_for_exact_materialization_only": True,
                "source_identity_used_as_trainable_input": False,
                "runtime_requested_or_resolved_action_used_as_trainable_input": False,
                "future_outcome_used_as_trainable_input": False,
                "target_safe_action_used_as_trainable_input": False,
            },
        }
        rows.append(row)
    return rows


def replay_expansion_metrics(
    *,
    selected: Sequence[V163SelectedBranchPoint],
    materialization: Mapping[str, object],
    branch_results: Sequence[Mapping[str, object]],
    dataset_rows: Sequence[Mapping[str, object]],
    v170_report: Mapping[str, object],
) -> dict[str, object]:
    per_seed = _per_seed_support_counts(branch_results)
    replay_verified_run_count = _replay_verified_run_count(branch_results)
    candidate_run_count = sum(
        len(_list_of_mappings(result.get("candidate_runs")))
        for result in branch_results
    )
    action_distribution = _action_distribution(branch_results)
    illegal_skipped = sum(
        len(_illegal_actions(point.action_mask))
        for point in selected
    )
    per_source = _per_source_row_action_support(branch_results)
    zero_hit_support = {
        str(seed): bool(per_seed.get(str(seed), {}).get("best_action_support_count", 0))
        for seed in BASELINE_ZERO_HIT_SEEDS
    }
    support_widths = [
        len(_ordered_actions(result.get("best_outcome_actions")))
        for result in branch_results
        if _ordered_actions(result.get("best_outcome_actions"))
    ]
    avg_support_width = _round(sum(support_widths) / len(support_widths)) if support_widths else 0.0
    v170_width = _round(_v170_best_average_set_width(v170_report))
    narrows = bool(support_widths) and avg_support_width < v170_width
    all_replays_verified = candidate_run_count > 0 and replay_verified_run_count == candidate_run_count
    all_planned_zero_hit_have_support = all(zero_hit_support.values())
    return {
        "policy": "m3_carrion_survivor_continuation_v171_replay_expansion_metrics_v1",
        "materialized_branch_point_count": materialization.get("materialized_branch_point_count", 0),
        "selected_branch_point_count": len(selected),
        "branch_result_count": len(branch_results),
        "replay_verified_run_count": replay_verified_run_count,
        "candidate_run_count": candidate_run_count,
        "all_replays_verified": all_replays_verified,
        "per_seed_support_counts": per_seed,
        "per_source_row_action_support": per_source,
        "action_distribution": action_distribution,
        "unsupported_requested_action_count": sum(
            _int(run.get("unsupported_requested_action_count"))
            for result in branch_results
            for run in _list_of_mappings(result.get("candidate_runs"))
        ),
        "illegal_skipped_action_count": illegal_skipped,
        "dataset_row_count": len(dataset_rows),
        "zero_hit_seeds_from_v170_now_have_replay_support": zero_hit_support,
        "all_v170_zero_hit_seeds_have_replay_support": all_planned_zero_hit_have_support,
        "average_replay_safe_set_width": avg_support_width,
        "v170_best_set_average_width": v170_width,
        "support_narrows_broad_set_valued_candidates": narrows,
        "ready_support_criteria_passed": (
            all_replays_verified
            and all_planned_zero_hit_have_support
            and narrows
            and bool(dataset_rows)
        ),
    }


def merged_replay_expansion_metrics(
    *,
    shard_reports: Sequence[Mapping[str, object]],
    dataset_rows: Sequence[Mapping[str, object]],
    v170_report: Mapping[str, object],
) -> dict[str, object]:
    branch_results = [
        result
        for report in shard_reports
        for result in _list_of_mappings(report.get("branch_results"))
    ]
    selected_count = sum(
        _int(_mapping(report.get("selection")).get("selected_branch_point_count"))
        for report in shard_reports
    )
    materialized_count = sum(
        _int(_mapping(report.get("branch_materialization")).get("materialized_branch_point_count"))
        for report in shard_reports
    )
    materialization = {"materialized_branch_point_count": materialized_count}
    metrics = replay_expansion_metrics(
        selected=[],
        materialization=materialization,
        branch_results=branch_results,
        dataset_rows=dataset_rows,
        v170_report=v170_report,
    )
    metrics["selected_branch_point_count"] = selected_count
    metrics["materialized_branch_point_count"] = materialized_count
    metrics["shard_count"] = len(shard_reports)
    metrics["planned_seed_count"] = len(REPLAY_EXPANSION_TARGET_SEEDS)
    metrics["all_planned_seed_shards_present"] = len(shard_reports) == len(REPLAY_EXPANSION_TARGET_SEEDS)
    metrics["illegal_skipped_action_count"] = sum(
        _int(_mapping(report.get("metrics")).get("illegal_skipped_action_count"))
        for report in shard_reports
    )
    metrics["unsupported_requested_action_count"] = sum(
        _int(_mapping(report.get("metrics")).get("unsupported_requested_action_count"))
        for report in shard_reports
    )
    return metrics


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
        "policy": "m3_carrion_survivor_continuation_v171_trainable_payload_leakage_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
        "forbidden_tokens": list(FORBIDDEN_TRAINABLE_PAYLOAD_TOKENS),
        "payload_count": len(feature_payloads),
    }


def partial_shard_status(
    *,
    selected: Sequence[V163SelectedBranchPoint],
    materialization: Mapping[str, object],
    branch_results: Sequence[Mapping[str, object]],
    fail_on_partial_shard: bool,
) -> dict[str, object]:
    selected_count = len(selected)
    materialized_count = _int(materialization.get("materialized_branch_point_count"))
    branch_count = len(branch_results)
    partial = (
        selected_count == 0
        or materialized_count != selected_count
        or branch_count != materialized_count
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v171_partial_shard_status_v1",
        "partial": partial,
        "fail_on_partial_shard": bool(fail_on_partial_shard),
        "explicit_partial_allow": not fail_on_partial_shard,
        "selected_branch_point_count": selected_count,
        "materialized_branch_point_count": materialized_count,
        "branch_result_count": branch_count,
        "reason": (
            "partial_or_empty_shard"
            if partial
            else "all_selected_branch_points_materialized_and_evaluated"
        ),
    }


def merged_partial_status(
    *,
    merge_validation: Mapping[str, object],
    shard_reports: Sequence[Mapping[str, object]],
    allow_partial_shard_evidence: bool,
) -> dict[str, object]:
    partial_shards = [
        str(_mapping(report.get("inputs")).get("shard_id") or _mapping(report.get("shard_plan_row")).get("shard_id") or "")
        for report in shard_reports
        if _mapping(report.get("partial_shard_status")).get("partial") is True
        or str(_mapping(report.get("classification")).get("primary", "")).endswith(
            "partial_shard_closed_no_training"
        )
    ]
    partial = merge_validation.get("all_required_shards_present") is not True or bool(partial_shards)
    return {
        "policy": "m3_carrion_survivor_continuation_v171_merged_partial_status_v1",
        "partial": partial,
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "partial_shard_ids": sorted(set(filter(None, partial_shards))),
        "missing_shard_ids": list(merge_validation.get("missing_shard_ids") or []),
        "merge_has_all_required_shards": merge_validation.get("all_required_shards_present"),
        "reason": "partial_shard_or_missing_required_shard" if partial else "complete_merge",
    }


def validate_merge_inputs(
    *,
    shard_plan_rows: Sequence[Mapping[str, object]],
    shard_reports: Sequence[Mapping[str, object]],
    merge_shard_reports: Sequence[str | Path],
    merge_shard_datasets: Sequence[str | Path],
    allow_partial_shard_evidence: bool,
) -> dict[str, object]:
    failures: list[str] = []
    expected_ids = {str(row.get("shard_id")) for row in shard_plan_rows}
    observed_ids = {
        str(_mapping(report.get("inputs")).get("shard_id") or _mapping(report.get("shard_plan_row")).get("shard_id") or "")
        for report in shard_reports
    }
    if len(merge_shard_reports) != len(merge_shard_datasets):
        failures.append("merge_report_dataset_count_mismatch")
    missing = sorted(expected_ids - observed_ids)
    extra = sorted(observed_ids - expected_ids)
    if extra:
        failures.append("unexpected_shard_reports")
    for index, report in enumerate(shard_reports):
        if (
            report.get("schema_version")
            != M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION
        ):
            failures.append(f"shard_{index}_schema_mismatch")
        if report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY:
            failures.append(f"shard_{index}_policy_mismatch")
        if exact_digest_validation_report(report).get("passed") is not True:
            failures.append(f"shard_{index}_exact_digest_mismatch")
        if _mapping(report.get("source_validation")).get("passed") is not True:
            failures.append(f"shard_{index}_source_validation_failed")
        if _mapping(report.get("leakage_scan")).get("passed") is not True:
            failures.append(f"shard_{index}_leakage_failed")
    partial_shard_ids = [
        str(_mapping(report.get("inputs")).get("shard_id") or _mapping(report.get("shard_plan_row")).get("shard_id") or "")
        for report in shard_reports
        if _mapping(report.get("partial_shard_status")).get("partial") is True
    ]
    return {
        "policy": "m3_carrion_survivor_continuation_v171_merge_input_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_shard_ids": sorted(expected_ids),
        "observed_shard_ids": sorted(filter(None, observed_ids)),
        "missing_shard_ids": missing,
        "unexpected_shard_ids": extra,
        "all_required_shards_present": not missing and not extra and bool(expected_ids),
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "partial_shard_ids": sorted(set(filter(None, partial_shard_ids))),
        "partial_without_allow": bool(partial_shard_ids) and not allow_partial_shard_evidence,
    }


def load_v170_shard_plan(path: str | Path) -> list[dict[str, object]]:
    return [dict(row) for row in load_jsonl_dataset(path)]


def load_jsonl_dataset(path: str | Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise CarrionSurvivorContinuationV171ReplayExpansionError(
                        f"expected JSON object row in {path}"
                    )
                rows.append(payload)
    return rows


def _classification(
    *,
    source_validation: Mapping[str, object],
    shard_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    metrics: Mapping[str, object],
    partial_status: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v171_replay_expansion_"
    if (
        source_validation.get("passed") is not True
        or shard_validation.get("passed") is not True
        or leakage_scan.get("passed") is not True
        or metrics.get("all_replays_verified") is False
        and _int(metrics.get("candidate_run_count")) > 0
    ):
        return prefix + "closed_invalid"
    if partial_status.get("partial") is True and (
        partial_status.get("allow_partial_shard_evidence") is not True
        and partial_status.get("explicit_partial_allow") is not True
    ):
        return prefix + "partial_shard_closed_no_training"
    if metrics.get("ready_support_criteria_passed") is True:
        return prefix + "replay_expansion_support_ready_for_v172_target_dataset_expansion_no_training"
    return prefix + "replay_expansion_support_limited_no_training"


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification.endswith(
        "replay_expansion_support_ready_for_v172_target_dataset_expansion_no_training"
    )
    invalid = classification.endswith("closed_invalid")
    partial = classification.endswith("partial_shard_closed_no_training")
    return {
        "policy": "m3_carrion_survivor_continuation_v171_route_recommendation_v1",
        "recommended_next_route": (
            "repair_source_leakage_or_replay_failure"
            if invalid
            else "rerun_complete_v171_shards_before_dataset_use"
            if partial
            else "v172_target_dataset_expansion_design_no_training"
            if ready
            else "continue_archive_source_expansion_no_training"
        ),
        "v172_target_dataset_expansion_recommended": ready,
        "training_authorized": False,
        "runtime_integration_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
    }


def _find_shard_plan_row(
    rows: Sequence[Mapping[str, object]],
    *,
    shard_id: str,
) -> Mapping[str, object] | None:
    for row in rows:
        if str(row.get("shard_id")) == shard_id:
            return row
    return None


def _resolve_v165_dataset_path(
    v170_report: Mapping[str, object],
    override: str | Path | None,
) -> Path:
    if override is not None:
        return Path(override)
    path = _mapping(v170_report.get("inputs")).get("v165_dataset")
    return Path(str(path)) if path else DEFAULT_V165_DATASET_PATH


def _source_paths_for_seed(
    *,
    v165_rows: Sequence[Mapping[str, object]],
    seed: int,
) -> list[Path]:
    paths = []
    for row in v165_rows:
        metadata = _mapping(row.get("metadata"))
        if _int(metadata.get("seed")) != int(seed):
            continue
        source_path = str(metadata.get("source_path") or "")
        if source_path:
            paths.append(Path(source_path))
    return sorted(set(paths), key=str)


def _selected_payload(point: V163SelectedBranchPoint) -> dict[str, object]:
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": "broad",
        "branch_tick": point.branch_tick,
        "record_index": point.record_index,
        "branch_index": point.branch_index,
        "agent_id": point.agent_id,
        "source_path": point.source_path,
        "line_number": point.line_number,
        "legal_first_action_count": len(point.top_value_candidate_set),
        "legal_first_action_set": list(point.top_value_candidate_set),
        "illegal_skipped_actions": _illegal_actions(point.action_mask),
        "observation_schema": point.observation_schema,
        "observation_digest": point.observation_digest,
        "source_record_digest": point.source_record_digest,
        "selection_rationale": dict(point.selection_rationale),
    }


def _selected_public_features_from_result(result: Mapping[str, object]) -> dict[str, object]:
    rationale = _mapping(result.get("selection_rationale"))
    branch_id = str(result.get("branch_id") or "")
    payload = rationale.get("v171_selected_public_features")
    if isinstance(payload, Mapping):
        return dict(payload)
    raise CarrionSurvivorContinuationV171ReplayExpansionError(
        f"missing v171 selected public features for {branch_id}"
    )


def _action_value_targets(
    *,
    action_mask: Mapping[str, bool],
    candidate_runs: Sequence[Mapping[str, object]],
    safe_action_set: Sequence[str],
) -> list[dict[str, object]]:
    by_action = {
        str(run.get("forced_action")): run
        for run in candidate_runs
        if str(run.get("forced_action")) in ACTION_NAMES
    }
    safe = set(_ordered_actions(safe_action_set))
    targets = []
    for action in ACTION_NAMES:
        run = by_action.get(action)
        replay = _mapping(run.get("replay_verification")) if run else {}
        outcome_key = list(_outcome_key(run)) if run else None
        targets.append(
            {
                "action": action,
                "public_mask": action_mask.get(action) is True,
                "target_available": run is not None,
                "replay_verified": replay.get("verified") is True if run else False,
                "safe_target": action in safe,
                "replay_outcome_key": outcome_key,
                "score_target": 1.0 if action in safe else 0.0 if run else None,
                "value_target": 1.0 if action in safe else 0.0 if run else None,
            }
        )
    return targets


def _per_seed_support_counts(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    payloads: dict[int, dict[str, object]] = defaultdict(
        lambda: {
            "branch_result_count": 0,
            "candidate_run_count": 0,
            "replay_verified_run_count": 0,
            "best_action_support_count": 0,
            "unique_best_support_row_count": 0,
            "multi_best_support_row_count": 0,
            "best_action_counts": Counter(),
        }
    )
    for result in branch_results:
        seed = _int(result.get("seed"))
        payload = payloads[seed]
        payload["branch_result_count"] = _int(payload["branch_result_count"]) + 1
        candidate_runs = _list_of_mappings(result.get("candidate_runs"))
        payload["candidate_run_count"] = _int(payload["candidate_run_count"]) + len(candidate_runs)
        payload["replay_verified_run_count"] = _int(payload["replay_verified_run_count"]) + sum(
            _mapping(run.get("replay_verification")).get("verified") is True
            for run in candidate_runs
        )
        best_actions = _ordered_actions(result.get("best_outcome_actions"))
        payload["best_action_support_count"] = _int(payload["best_action_support_count"]) + len(best_actions)
        payload["unique_best_support_row_count"] = _int(payload["unique_best_support_row_count"]) + int(len(best_actions) == 1)
        payload["multi_best_support_row_count"] = _int(payload["multi_best_support_row_count"]) + int(len(best_actions) > 1)
        counter = payload["best_action_counts"]
        if isinstance(counter, Counter):
            counter.update(best_actions)
    return {
        str(seed): {
            **{key: value for key, value in payload.items() if key != "best_action_counts"},
            "best_action_counts": dict(sorted(payload["best_action_counts"].items())) if isinstance(payload.get("best_action_counts"), Counter) else {},
        }
        for seed, payload in sorted(payloads.items())
    }


def _per_source_row_action_support(
    branch_results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for result in branch_results:
        candidate_runs = _list_of_mappings(result.get("candidate_runs"))
        best_actions = _ordered_actions(result.get("best_outcome_actions"))
        rows.append(
            {
                "branch_id": result.get("branch_id"),
                "seed": result.get("seed"),
                "source_path": result.get("source_path"),
                "line_number": result.get("line_number"),
                "branch_tick": result.get("branch_tick"),
                "agent_id": result.get("agent_id"),
                "legal_candidate_actions": _ordered_actions(result.get("top_value_candidate_set")),
                "candidate_run_count": len(candidate_runs),
                "replay_verified_run_count": sum(
                    _mapping(run.get("replay_verification")).get("verified") is True
                    for run in candidate_runs
                ),
                "best_outcome_actions": best_actions,
                "best_outcome_action_count": len(best_actions),
                "candidate_action_support": [
                    {
                        "action": run.get("forced_action"),
                        "replay_verified": _mapping(run.get("replay_verification")).get("verified") is True,
                        "outcome_key": list(_outcome_key(run)),
                        "alive_agents": run.get("alive_agents"),
                        "births": run.get("births"),
                        "deaths": run.get("deaths"),
                    }
                    for run in candidate_runs
                ],
            }
        )
    return rows


def _action_distribution(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    forced = Counter()
    best = Counter()
    for result in branch_results:
        forced.update(
            str(run.get("forced_action"))
            for run in _list_of_mappings(result.get("candidate_runs"))
            if str(run.get("forced_action")) in ACTION_NAMES
        )
        best.update(_ordered_actions(result.get("best_outcome_actions")))
    return {
        "policy": "m3_carrion_survivor_continuation_v171_action_distribution_v1",
        "candidate_forced_action_counts": dict(sorted(forced.items(), key=lambda item: _action_order(item[0]))),
        "best_outcome_action_counts": dict(sorted(best.items(), key=lambda item: _action_order(item[0]))),
        "dominant_best_outcome_action": _dominant_count_share(best),
    }


def _replay_verified_run_count(branch_results: Sequence[Mapping[str, object]]) -> int:
    return sum(
        _mapping(run.get("replay_verification")).get("verified") is True
        for result in branch_results
        for run in _list_of_mappings(result.get("candidate_runs"))
    )


def _illegal_actions(action_mask: Mapping[str, bool]) -> list[str]:
    return [action for action in ACTION_NAMES if action_mask.get(action) is not True]


def _ordered_actions(value: object) -> list[str]:
    return sorted(
        [str(action) for action in value or [] if str(action) in ACTION_NAMES],
        key=_action_order,
    )


def _v170_best_average_set_width(v170_report: Mapping[str, object]) -> float:
    lanes = _mapping(v170_report.get("portfolio_lanes"))
    set_lane = _mapping(lanes.get("set_valued_ranking"))
    best = _mapping(set_lane.get("best_matrix_entry"))
    value = best.get("average_set_width")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _source_lifecycle_scan(v170_report: Mapping[str, object]) -> dict[str, object]:
    false_fields = (
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "runtime_observation_schema_changed",
        "runtime_policy_changed",
        "replay_viewer_schema_changed",
        "threshold_tuning_ran",
        "k_tuning_ran",
        "training_ran",
        "shadow_eval_ran",
        "live_ab_allowed",
        "promotion_authorized",
        "staging_authorized",
        "commit_authorized",
        "reset_authorized",
        "clean_authorized",
    )
    failures = []
    if v170_report.get("diagnostics_only") is not True:
        failures.append({"field": "diagnostics_only", "observed": v170_report.get("diagnostics_only")})
    for field in false_fields:
        if field in v170_report and v170_report.get(field) is not False:
            failures.append({"field": field, "observed": v170_report.get(field)})
    return {
        "policy": "m3_carrion_survivor_continuation_v171_v170_lifecycle_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _source_row_policy() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v171_source_row_policy_v1",
        "source_path_tick_agent_used_for_exact_materialization_lookup_only": True,
        "source_identity_allowed_in_trainable_payload": False,
        "target_safe_action_allowed_in_trainable_payload": False,
        "future_outcome_allowed_in_trainable_payload": False,
        "runtime_requested_or_resolved_action_allowed_in_trainable_payload": False,
        "public_mask_legal_first_actions_only": True,
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "training_allowed": False,
        "scorer_retraining_allowed": False,
        "runtime_artifact_allowed": False,
        "runtime_action_change_allowed": False,
        "shadow_or_live_eval_allowed": False,
        "k_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "replay_viewer_schema_change_allowed": False,
        "promotion_allowed": False,
        "staging_allowed": False,
        "commit_allowed": False,
        "reset_allowed": False,
        "clean_allowed": False,
    }


def _lifecycle_flags() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_ran": False,
        "scorer_retraining_ran": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": False,
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "k_tuning_ran": False,
        "threshold_tuning_ran": False,
        "replay_viewer_schema_changed": False,
        "promotion_authorized": False,
        "staging_authorized": False,
        "commit_authorized": False,
        "reset_authorized": False,
        "clean_authorized": False,
        "non_promoted": True,
    }


def _plan_windows(plan_row: Mapping[str, object]) -> list[tuple[int, int]]:
    windows = []
    for window in _list_of_mappings(plan_row.get("proposed_branch_windows")):
        windows.append((_int(window.get("start_tick")), _int(window.get("end_tick"))))
    return windows


def _effective_windows(
    plan_row: Mapping[str, object] | None,
    branch_windows: Sequence[str | tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    if branch_windows:
        return [_parse_window(window) for window in branch_windows]
    return _plan_windows(plan_row or {})


def _parse_window(value: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        start, end = value
    else:
        parts = str(value).split(":", maxsplit=1)
        if len(parts) != 2:
            raise CarrionSurvivorContinuationV171ReplayExpansionError(
                f"invalid branch window {value!r}; expected START:END"
            )
        start, end = int(parts[0]), int(parts[1])
    if int(start) < 0 or int(end) < int(start):
        raise CarrionSurvivorContinuationV171ReplayExpansionError(
            f"invalid branch window {value!r}; end must be >= start"
        )
    return int(start), int(end)


def _window_strings(windows: Sequence[tuple[int, int]]) -> list[str]:
    return [f"{int(start)}:{int(end)}" for start, end in windows]


def _empty_selection_report(
    *,
    shard_id: str,
    seed: int | None,
    reason: str,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v171_shard_branch_point_selection_v1",
        "shard_id": shard_id,
        "seed": seed,
        "selected_branch_point_count": 0,
        "branch_points_by_tick": {},
        "legal_first_action_width_counts": {},
        "illegal_skipped_action_count": 0,
        "selected_branch_points_sample": [],
        "all_source_identity_metadata_non_trainable": True,
        "reason": reason,
    }


def _empty_materialization_report(
    *,
    reason: str,
    selected_branch_point_count: int = 0,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v171_exact_branch_materialization_v1",
        "selected_branch_point_count": int(selected_branch_point_count),
        "materialized_branch_point_count": 0,
        "materialization_failure_count": int(selected_branch_point_count > 0),
        "materialization_failures": (
            [{"reason": reason}] if selected_branch_point_count > 0 else []
        ),
        "reference_runs": [],
        "exact_materialization_proven": False,
        "passed": False,
    }


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
    for token in FORBIDDEN_TRAINABLE_PAYLOAD_TOKENS:
        if token in lowered:
            return token
    return None


def _shard_report_summary(report: Mapping[str, object]) -> dict[str, object]:
    metrics = _mapping(report.get("metrics"))
    inputs = _mapping(report.get("inputs"))
    return {
        "shard_id": inputs.get("shard_id"),
        "classification": _mapping(report.get("classification")).get("primary"),
        "exact_digest": report.get("exact_digest"),
        "dataset_digest": _mapping(report.get("dataset")).get("dataset_digest"),
        "materialized_branch_point_count": metrics.get("materialized_branch_point_count"),
        "replay_verified_run_count": metrics.get("replay_verified_run_count"),
        "partial": _mapping(report.get("partial_shard_status")).get("partial"),
    }


def _strip_dataset_feature_payloads(
    branch_results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    stripped = []
    for result in branch_results:
        payload = deepcopy(dict(result))
        rationale = dict(_mapping(payload.get("selection_rationale")))
        rationale.pop("v171_selected_public_features", None)
        payload["selection_rationale"] = rationale
        stripped.append(payload)
    return stripped


def _trim_materialized_branch_histories(materialized: Sequence[object]) -> None:
    for point in materialized:
        world = getattr(point, "world", None)
        if world is None:
            continue
        if hasattr(world, "trajectory_records"):
            world.trajectory_records = []
        if hasattr(world, "tick_trajectory_records"):
            world.tick_trajectory_records = []


def _json_round_trip_digest(payload: Mapping[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(without_digest, sort_keys=True, allow_nan=False))
    )


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(dict(row), handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
