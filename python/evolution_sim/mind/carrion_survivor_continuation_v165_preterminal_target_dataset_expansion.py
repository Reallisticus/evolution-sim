from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Mapping, Sequence
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
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V158_DATASET_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION,
    exact_digest_validation_report,
    target_dataset_leakage_scan,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v163_tied_set_branch_target_expansion import (
    STRICT_BROAD_SEEDS,
    _bool_action_mask,
    _record_materialization_payload,
)
from evolution_sim.mind.carrion_survivor_continuation_v164_preterminal_tied_set_branch_target_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V164_REPORT_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_preterminal_target_dataset_expansion_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_preterminal_target_dataset_expansion_row_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_preterminal_target_dataset_expansion_v1"
)
EXPECTED_V164_CLASSIFICATION = "preterminal_tied_set_target_support_ready_no_training"
EXPECTED_V164_EXACT_DIGEST = (
    "a375cbb7f287d4829bb94cab02c5fdb79133fd567a7f8269ab5da5be5332936f"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v165-carrion-survivor-continuation-preterminal-target-dataset-expansion.json"
)
DEFAULT_TARGET_DATASET_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v165-carrion-survivor-continuation-preterminal-target-dataset-expansion.jsonl"
)
DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE = 0.75


class CarrionSurvivorContinuationV165PreterminalTargetDatasetExpansionError(
    ValueError
):
    pass


def run_carrion_survivor_continuation_v165_preterminal_target_dataset_expansion(
    *,
    v164_report_path: str | Path = DEFAULT_V164_REPORT_PATH,
    base_dataset_path: str | Path = DEFAULT_V158_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    target_dataset_output_path: str | Path = DEFAULT_TARGET_DATASET_OUTPUT_PATH,
    expected_v164_exact_digest: str | None = EXPECTED_V164_EXACT_DIGEST,
    strict_broad_heldout_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    max_dominant_safe_action_share: float = DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
) -> dict[str, object]:
    v164_report = load_json_report(v164_report_path)
    base_rows, base_load = _load_base_dataset(base_dataset_path)
    source_validation = validate_v165_sources(
        v164_report,
        base_rows=base_rows,
        expected_v164_exact_digest=expected_v164_exact_digest,
    )
    preterminal_rows: list[dict[str, object]] = []
    row_build_validation = _skipped_row_build_validation("source_validation_failed")
    if source_validation.get("passed") is True:
        preterminal_rows, row_build_validation = build_v165_preterminal_rows(
            v164_report,
        )
    combined_rows = [dict(row) for row in base_rows] + preterminal_rows
    leakage_scan = target_dataset_leakage_scan(combined_rows)
    preterminal_summary = summarize_v165_preterminal_rows(
        preterminal_rows,
        max_dominant_safe_action_share=max_dominant_safe_action_share,
    )
    combined_summary = summarize_combined_rows(
        combined_rows,
        base_row_count=len(base_rows),
        preterminal_row_count=len(preterminal_rows),
        max_dominant_safe_action_share=max_dominant_safe_action_share,
    )
    seed_overlap = source_seed_overlap_report(
        v164_report,
        strict_broad_heldout_seeds=strict_broad_heldout_seeds,
    )
    classification = _classification(
        source_validation=source_validation,
        row_build_validation=row_build_validation,
        leakage_scan=leakage_scan,
        preterminal_summary=preterminal_summary,
    )
    dataset_digest = stable_payload_digest(combined_rows)
    _write_jsonl(target_dataset_output_path, combined_rows)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v164_report": str(v164_report_path),
            "expected_v164_exact_digest": expected_v164_exact_digest,
            "base_v158_dataset": str(base_dataset_path),
            "target_dataset_output": str(target_dataset_output_path),
            "strict_broad_heldout_seeds": [
                int(seed) for seed in strict_broad_heldout_seeds
            ],
            "max_dominant_safe_action_share": _round(
                max_dominant_safe_action_share
            ),
        },
        "source_validation": source_validation,
        "source_v164_digest_validation": _source_v164_digest_validation(
            source_validation
        ),
        "base_dataset_load": base_load,
        "row_build_validation": row_build_validation,
        "leakage_scan": leakage_scan,
        "dataset": {
            "path": str(target_dataset_output_path),
            "combined_row_count": len(combined_rows),
            "base_row_count": len(base_rows),
            "preterminal_row_count": len(preterminal_rows),
            "base_rows_preserved": True,
            "base_row_schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION
            ),
            "preterminal_row_schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
            ),
            "dataset_digest": dataset_digest,
        },
        "base_rows": {
            "path": str(base_dataset_path),
            "row_count": len(base_rows),
            "dataset_digest": stable_payload_digest(base_rows),
        },
        "preterminal_rows": preterminal_summary,
        "combined_dataset_summary": combined_summary,
        "source_seed_overlap": seed_overlap,
        "future_evaluation_policy": _future_evaluation_policy(seed_overlap),
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "dataset_created": True,
        "training_ran": False,
        "training_authorized": False,
        "artifact_created": False,
        "runtime_artifact_created": False,
        "live_ab_allowed": False,
        "live_ab_ran": False,
        "runtime_override_path_created": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "threshold_tuning_recommended": False,
        "runtime_action_selection_changed": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v165_sources(
    v164_report: Mapping[str, object],
    *,
    base_rows: Sequence[Mapping[str, object]],
    expected_v164_exact_digest: str | None = EXPECTED_V164_EXACT_DIGEST,
) -> dict[str, object]:
    failures: list[str] = []
    if (
        v164_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION
    ):
        failures.append("v164_schema_version_mismatch")
    if (
        v164_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY
    ):
        failures.append("v164_policy_mismatch")
    observed_classification = _mapping(v164_report.get("classification")).get(
        "primary"
    )
    if observed_classification != EXPECTED_V164_CLASSIFICATION:
        failures.append("v164_unexpected_classification")
    digest_validation = exact_digest_validation_report(v164_report)
    if digest_validation.get("passed") is not True:
        failures.append("v164_exact_digest_mismatch")
    observed_digest = str(v164_report.get("exact_digest") or "")
    if expected_v164_exact_digest and observed_digest != expected_v164_exact_digest:
        failures.append("v164_unexpected_exact_digest")
    materialization = _mapping(v164_report.get("branch_materialization"))
    if materialization.get("passed") is not True:
        failures.append("v164_branch_materialization_not_passed")
    if materialization.get("exact_materialization_proven") is not True:
        failures.append("v164_exact_materialization_not_proven")
    outcome = _mapping(v164_report.get("outcome_support"))
    if outcome.get("all_replays_verified") is not True:
        failures.append("v164_replay_verification_not_passed")
    if outcome.get("preterminal_target_support_noncollapsed") is not True:
        failures.append("v164_preterminal_support_not_noncollapsed")
    lifecycle = _v164_lifecycle_validation(v164_report)
    if lifecycle.get("passed") is not True:
        failures.append("v164_lifecycle_not_diagnostics_only")
    if not base_rows:
        failures.append("base_v158_dataset_empty_or_missing")
    branch_results = _list_of_mappings(v164_report.get("branch_results"))
    branch_replay_failures = []
    for branch in branch_results:
        if branch.get("replay_verification_passed") is not True:
            branch_replay_failures.append(str(branch.get("branch_id")))
        for run in _list_of_mappings(branch.get("candidate_runs")):
            replay = _mapping(run.get("replay_verification"))
            if replay.get("verified") is not True:
                branch_replay_failures.append(
                    f"{branch.get('branch_id')}:{run.get('forced_action')}"
                )
    if branch_replay_failures:
        failures.append("v164_candidate_replay_digest_mismatch")
    return {
        "policy": "m3_carrion_survivor_continuation_v165_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v164_classification": EXPECTED_V164_CLASSIFICATION,
        "observed_v164_classification": observed_classification,
        "expected_v164_exact_digest": expected_v164_exact_digest,
        "observed_v164_exact_digest": observed_digest,
        "v164_exact_digest_validation": digest_validation,
        "v164_lifecycle_validation": lifecycle,
        "v164_branch_materialization_passed": materialization.get("passed"),
        "v164_exact_materialization_proven": materialization.get(
            "exact_materialization_proven"
        ),
        "v164_all_replays_verified": outcome.get("all_replays_verified"),
        "v164_branch_result_count": len(branch_results),
        "v164_candidate_run_count": outcome.get("candidate_run_count"),
        "base_row_count": len(base_rows),
        "base_dataset_digest": stable_payload_digest(base_rows),
        "branch_replay_failure_examples": branch_replay_failures[:16],
    }


def build_v165_preterminal_rows(
    v164_report: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for branch in _list_of_mappings(v164_report.get("branch_results")):
        record_result = _load_branch_source_record(branch)
        if record_result.get("passed") is not True:
            failures.append(
                {
                    "branch_id": branch.get("branch_id"),
                    "reason": record_result.get("reason"),
                    "error": record_result.get("error"),
                }
            )
            continue
        record = _mapping(record_result.get("record"))
        source_digest = stable_payload_digest(_record_materialization_payload(record))
        if source_digest != branch.get("source_record_digest"):
            failures.append(
                {
                    "branch_id": branch.get("branch_id"),
                    "reason": "source_record_digest_mismatch",
                    "expected": branch.get("source_record_digest"),
                    "observed": source_digest,
                }
            )
            continue
        rows.append(
            _build_preterminal_row(
                branch=branch,
                record=record,
                source_record_digest=source_digest,
                v164_digest=str(v164_report.get("exact_digest") or ""),
            )
        )
    return (
        rows,
        {
            "policy": "m3_carrion_survivor_continuation_v165_row_build_validation_v1",
            "passed": not failures and len(rows) == len(_list_of_mappings(v164_report.get("branch_results"))),
            "failures": failures[:24],
            "failure_count": len(failures),
            "v164_branch_result_count": len(
                _list_of_mappings(v164_report.get("branch_results"))
            ),
            "preterminal_row_count": len(rows),
        },
    )


def summarize_v165_preterminal_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    max_dominant_safe_action_share: float = DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
) -> dict[str, object]:
    classification_counts = Counter(
        str(row.get("target_classification", "")) for row in rows
    )
    support_counts: Counter[str] = Counter()
    value_target_counts: Counter[str] = Counter()
    target_local_winner_counts: Counter[str] = Counter()
    terminal_winner_counts: Counter[str] = Counter()
    candidate_action_counts: Counter[str] = Counter()
    for row in rows:
        for action in _action_list(row.get("safe_action_set")):
            support_counts.update([action])
        for action in _action_list(row.get("target_local_primary_support_actions")):
            target_local_winner_counts.update([action])
        for action in _action_list(row.get("terminal_population_guard_support_actions")):
            terminal_winner_counts.update([action])
        for target in _list_of_mappings(row.get("action_value_targets")):
            action = str(target.get("action", ""))
            if target.get("target_available") is True and action in ACTION_NAMES:
                value_target_counts.update([action])
                candidate_action_counts.update([action])
    support_counts = _ordered_counter(support_counts)
    dominant = _dominant_count_share(support_counts)
    unresolved = sum(
        count
        for label, count in classification_counts.items()
        if label not in {"unique_robust_winner", "multi_action_safe_set"}
    )
    support_ready = (
        bool(rows)
        and unresolved == 0
        and len(support_counts) >= 2
        and float(dominant.get("share") or 0.0)
        <= float(max_dominant_safe_action_share)
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v165_preterminal_row_summary_v1",
        "row_count": len(rows),
        "classification_counts": dict(sorted(classification_counts.items())),
        "unique_winner_count": int(
            classification_counts.get("unique_robust_winner", 0)
        ),
        "multi_action_safe_set_count": int(
            classification_counts.get("multi_action_safe_set", 0)
        ),
        "unresolved_count": int(unresolved),
        "per_action_support_counts": dict(support_counts),
        "per_action_value_target_counts": dict(_ordered_counter(value_target_counts)),
        "target_local_primary_support_counts": dict(
            _ordered_counter(target_local_winner_counts)
        ),
        "terminal_population_guard_support_counts": dict(
            _ordered_counter(terminal_winner_counts)
        ),
        "candidate_action_counts": dict(_ordered_counter(candidate_action_counts)),
        "dominant_safe_action": dominant.get("key"),
        "dominant_safe_action_count": dominant.get("count"),
        "dominant_safe_action_share": dominant.get("share"),
        "max_dominant_safe_action_share": _round(max_dominant_safe_action_share),
        "support_ready": support_ready,
    }


def summarize_combined_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    base_row_count: int,
    preterminal_row_count: int,
    max_dominant_safe_action_share: float,
) -> dict[str, object]:
    support_counts: Counter[str] = Counter()
    for row in rows:
        for action in _action_list(row.get("safe_action_set")):
            support_counts.update([action])
    dominant = _dominant_count_share(support_counts)
    return {
        "policy": "m3_carrion_survivor_continuation_v165_combined_dataset_summary_v1",
        "combined_row_count": len(rows),
        "base_row_count": int(base_row_count),
        "preterminal_row_count": int(preterminal_row_count),
        "per_action_support_counts": dict(_ordered_counter(support_counts)),
        "dominant_safe_action": dominant.get("key"),
        "dominant_safe_action_count": dominant.get("count"),
        "dominant_safe_action_share": dominant.get("share"),
        "max_dominant_safe_action_share": _round(max_dominant_safe_action_share),
    }


def source_seed_overlap_report(
    v164_report: Mapping[str, object],
    *,
    strict_broad_heldout_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
) -> dict[str, object]:
    source_seeds = sorted(
        {
            _int(branch.get("seed"))
            for branch in _list_of_mappings(v164_report.get("branch_results"))
        }
    )
    heldout = sorted({int(seed) for seed in strict_broad_heldout_seeds})
    overlap = sorted(set(source_seeds) & set(heldout))
    return {
        "policy": "m3_carrion_survivor_continuation_v165_source_seed_overlap_v1",
        "v164_support_provenance_seeds": source_seeds,
        "strict_broad_heldout_matrix_seeds": heldout,
        "overlap_seeds": overlap,
        "overlap_count": len(overlap),
        "v164_strict_broad_seeds_become_support_provenance_seeds": True,
        "v164_strict_broad_seeds_remain_future_promotion_heldout_seeds": False,
    }


def _build_preterminal_row(
    *,
    branch: Mapping[str, object],
    record: Mapping[str, object],
    source_record_digest: str,
    v164_digest: str,
) -> dict[str, object]:
    action_mask = _bool_action_mask(record.get("public_action_mask") or record.get("action_mask"))
    targets, safe_set, local_support, terminal_support = _action_value_targets_for_branch(
        branch,
        action_mask=action_mask,
    )
    classification = (
        "unique_robust_winner"
        if len(safe_set) == 1
        else "multi_action_safe_set"
        if len(safe_set) > 1
        else "unresolved_no_guarded_target_local_support"
    )
    metadata = {
        "metadata_schema_version": (
            "m3_carrion_survivor_continuation_v165_preterminal_target_metadata_v1"
        ),
        "source": "v164_preterminal_tied_set_branch_replay",
        "v164_report_exact_digest": v164_digest,
        "seed": branch.get("seed"),
        "fixture": branch.get("fixture"),
        "branch_tick": branch.get("branch_tick"),
        "remaining_horizon": branch.get("remaining_horizon"),
        "agent_id": branch.get("agent_id"),
        "branch_id": branch.get("branch_id"),
        "source_path": branch.get("source_path"),
        "line_number": branch.get("line_number"),
        "record_index": branch.get("record_index"),
        "runtime_requested_action": branch.get("runtime_requested_action"),
        "runtime_resolved_action": branch.get("runtime_resolved_action"),
        "predicted_action": branch.get("predicted_action"),
        "nearest_neighbor_row_index": branch.get("nearest_neighbor_row_index"),
        "top_value_candidate_set": branch.get("top_value_candidate_set"),
        "source_record_digest": source_record_digest,
        "reported_source_record_digest": branch.get("source_record_digest"),
        "materialized_record_digest": branch.get("materialized_record_digest"),
        "branch_state_digest": branch.get("branch_state_digest"),
        "replay_digests_by_action": {
            str(run.get("forced_action")): run.get("replay_digest")
            for run in _list_of_mappings(branch.get("candidate_runs"))
        },
        "replay_verification_digests_by_action": {
            str(run.get("forced_action")): _mapping(
                run.get("replay_verification")
            ).get("actual_digest")
            for run in _list_of_mappings(branch.get("candidate_runs"))
        },
        "seed_tick_agent_path_digest_for_materialization_only": True,
        "source_seed_is_support_provenance_not_future_promotion_holdout": True,
        "runtime_requested_action_used_as_scorer_input": False,
        "future_outcomes_used_as_trainable_input": False,
    }
    row: dict[str, object] = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ),
        "row_origin": "v165_preterminal_branch_target",
        "feature_policy_id": "public_mind_v3_observation_and_action_mask",
        "trainable_public_features": {
            "public_observation": dict(_mapping(record.get("observation_input"))),
            "action_mask": action_mask,
        },
        "public_action_mask": action_mask,
        "candidate_action_value_targets": [
            target for target in targets if target.get("target_available") is True
        ],
        "action_value_targets": targets,
        "target_local_primary_support_actions": local_support,
        "terminal_population_guard_support_actions": terminal_support,
        "safe_action_set": safe_set,
        "target_classification": classification,
        "metadata": metadata,
    }
    if len(safe_set) == 1:
        row["robust_winner_action"] = safe_set[0]
    return row


def _action_value_targets_for_branch(
    branch: Mapping[str, object],
    *,
    action_mask: Mapping[str, bool],
) -> tuple[list[dict[str, object]], list[str], list[str], list[str]]:
    candidate_runs = _list_of_mappings(branch.get("candidate_runs"))
    run_by_action = {
        str(run.get("forced_action")): run
        for run in candidate_runs
        if str(run.get("forced_action")) in ACTION_NAMES
    }
    local_scores = {
        action: _target_local_score(run)
        for action, run in run_by_action.items()
    }
    terminal_scores = {
        action: _terminal_population_score(run)
        for action, run in run_by_action.items()
    }
    local_ranks = _score_ranks(local_scores)
    terminal_ranks = _score_ranks(terminal_scores)
    local_support = _top_score_actions(local_scores)
    terminal_support = _top_score_actions(terminal_scores)
    guarded_actions = [
        action
        for action, run in run_by_action.items()
        if _terminal_population_guard_passed(run)
    ]
    guarded_local = {
        action: local_scores[action] for action in guarded_actions
    }
    safe_set = _top_score_actions(guarded_local)
    targets: list[dict[str, object]] = []
    for action in ACTION_NAMES:
        run = _mapping(run_by_action.get(action))
        available = bool(run)
        first = _mapping(run.get("first_action_outcome_summary"))
        local_delta = _mapping(run.get("target_local_delta_vs_reference"))
        terminal_delta = _mapping(run.get("terminal_population_delta_vs_reference"))
        targets.append(
            {
                "action": action,
                "public_mask": bool(action_mask.get(action, False)),
                "target_available": available,
                "replayed_tied_set_candidate": available,
                "value_target": _round(local_scores.get(action)) if available else None,
                "score_target": _round(local_scores.get(action)) if available else None,
                "target_local_continuation_score": (
                    _round(local_scores.get(action)) if available else None
                ),
                "target_local_continuation_rank": local_ranks.get(action),
                "terminal_population_score": (
                    _round(terminal_scores.get(action)) if available else None
                ),
                "terminal_population_rank": terminal_ranks.get(action),
                "terminal_population_guard_passed": (
                    _terminal_population_guard_passed(run) if available else False
                ),
                "safe_target": action in safe_set,
                "robust_safe_action": action in safe_set,
                "target_local_primary_support": action in local_support,
                "terminal_population_guard_support": action in terminal_support,
                "target_alive_at_end": run.get("target_alive_at_end")
                if available
                else None,
                "target_energy_ratio_at_end": run.get("target_energy_ratio_at_end")
                if available
                else None,
                "target_hydration_ratio_at_end": run.get(
                    "target_hydration_ratio_at_end"
                )
                if available
                else None,
                "target_health_ratio_at_end": run.get("target_health_ratio_at_end")
                if available
                else None,
                "target_local_delta_vs_reference": dict(local_delta)
                if available
                else {},
                "terminal_population_delta_vs_reference": dict(terminal_delta)
                if available
                else {},
                "first_action_outcome_summary": dict(first) if available else {},
                "continuation_run_count": 1 if available else 0,
                "safe_run_count": 1 if action in safe_set else 0,
                "safe_share": 1.0 if action in safe_set else 0.0,
                "unsupported_requested_action_total": _int(
                    run.get("unsupported_requested_action_count")
                )
                if available
                else 0,
            }
        )
    return targets, safe_set, local_support, terminal_support


def _target_local_score(run: Mapping[str, object]) -> float:
    target = _mapping(run.get("target_terminal"))
    first = _mapping(run.get("first_action_outcome_summary"))
    return _round(
        _bool_score(target.get("alive")) * 1000.0
        + _number(target.get("energy_ratio")) * 100.0
        + _number(target.get("hydration_ratio")) * 50.0
        + _number(target.get("health_ratio")) * 25.0
        + _number(first.get("resource_gain")) * 10.0
        - float(_int(run.get("unsupported_requested_action_count"))) * 100.0
    )


def _terminal_population_score(run: Mapping[str, object]) -> float:
    return _round(
        float(_int(run.get("alive_agents"))) * 100.0
        + float(_int(run.get("births"))) * 25.0
        - float(_int(run.get("deaths"))) * 25.0
        - float(_int(run.get("unsupported_requested_action_count"))) * 100.0
    )


def _terminal_population_guard_passed(run: Mapping[str, object]) -> bool:
    delta = _mapping(run.get("terminal_population_delta_vs_reference"))
    return (
        _int(delta.get("alive_agents")) >= 0
        and _int(delta.get("births")) >= 0
        and _int(delta.get("deaths")) <= 0
        and _int(delta.get("unsupported_requested_action_count")) <= 0
    )


def _score_ranks(scores: Mapping[str, float]) -> dict[str, int]:
    ordered_values = sorted(set(scores.values()), reverse=True)
    return {
        action: ordered_values.index(score) + 1
        for action, score in scores.items()
    }


def _top_score_actions(scores: Mapping[str, float]) -> list[str]:
    if not scores:
        return []
    best = max(scores.values())
    return sorted(
        [action for action, score in scores.items() if score == best],
        key=_action_order,
    )


def _load_branch_source_record(branch: Mapping[str, object]) -> dict[str, object]:
    path = Path(str(branch.get("source_path") or ""))
    line_number = _int(branch.get("line_number"))
    if not path or not str(path):
        return {"passed": False, "reason": "missing_source_path"}
    if line_number <= 0:
        return {"passed": False, "reason": "missing_line_number"}
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
            for index, line in enumerate(handle, start=1):
                if index != line_number:
                    continue
                payload = json.loads(line)
                record = _mapping(payload.get("record") if isinstance(payload, Mapping) else {})
                if not record:
                    return {"passed": False, "reason": "source_line_has_no_record"}
                return {"passed": True, "record": dict(record)}
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "reason": "source_record_load_failed", "error": str(exc)}
    return {"passed": False, "reason": "source_line_number_not_found"}


def _load_base_dataset(path: str | Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise CarrionSurvivorContinuationV165PreterminalTargetDatasetExpansionError(
                        f"base dataset row {line_number} must be an object"
                    )
                rows.append(payload)
    except OSError as exc:
        return [], {
            "policy": "m3_carrion_survivor_continuation_v165_base_dataset_load_v1",
            "passed": False,
            "path": str(path),
            "error": str(exc),
            "row_count": 0,
        }
    return rows, {
        "policy": "m3_carrion_survivor_continuation_v165_base_dataset_load_v1",
        "passed": True,
        "path": str(path),
        "row_count": len(rows),
        "dataset_digest": stable_payload_digest(rows),
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    row_build_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    preterminal_summary: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True or row_build_validation.get("passed") is not True:
        return "source_invalid_closed_no_training"
    if leakage_scan.get("passed") is not True:
        return "leakage_failed_closed_no_training"
    if preterminal_summary.get("support_ready") is True:
        return "expanded_target_dataset_support_ready_no_training"
    return "expanded_target_dataset_support_limited_no_training"


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification == "expanded_target_dataset_support_ready_no_training"
    return {
        "policy": "m3_carrion_survivor_continuation_v165_route_recommendation_v1",
        "target_dataset_support_ready": ready,
        "recommended_next_route": (
            "v166_diagnostics_only_scorer_retraining_with_source_split_evaluation"
            if ready
            else "keep_preterminal_target_dataset_diagnostics_closed_until_support_or_leakage_issue_is_resolved"
        ),
        "v166_diagnostics_only_scorer_retraining_recommended": ready,
        "source_split_evaluation_required": True,
        "threshold_tuning_recommended": False,
        "training_authorized": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "runtime_policy_integration_allowed": False,
        "runtime_action_selection_changed": False,
    }


def _future_evaluation_policy(seed_overlap: Mapping[str, object]) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v165_future_evaluation_policy_v1",
        "leave_source_seed_out_diagnostics_required": True,
        "new_held_out_broad_seeds_required": True,
        "v164_support_provenance_seed_exclusion_required_for_future_promotion": True,
        "v164_strict_broad_seeds_become_support_provenance_seeds_not_heldout": True,
        "support_provenance_seeds": seed_overlap.get("v164_support_provenance_seeds"),
        "disallowed_future_promotion_heldout_seed_reuse": seed_overlap.get(
            "overlap_seeds"
        ),
        "promotion_authorized": False,
        "live_ab_allowed": False,
    }


def _v164_lifecycle_validation(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    for field in (
        "training_ran",
        "artifact_created",
        "runtime_artifact_created",
        "live_ab_allowed",
        "promotion_authorized",
        "runtime_action_selection_changed",
        "runtime_override_path_created",
        "threshold_tuning_recommended",
    ):
        if field in report and report.get(field) is not False:
            failures.append({"field": field, "observed": report.get(field)})
    lifecycle = _mapping(report.get("lifecycle_proof"))
    for field in (
        "runtime_action_selection_changed",
        "runtime_artifact_created",
        "live_ab_allowed",
        "promotion_authorized",
        "runtime_override_path_created",
        "threshold_tuning_recommended",
    ):
        if field in lifecycle and lifecycle.get(field) is not False:
            failures.append(
                {"field": f"lifecycle_proof.{field}", "observed": lifecycle.get(field)}
            )
    return {
        "policy": "m3_carrion_survivor_continuation_v165_v164_lifecycle_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _source_v164_digest_validation(source_validation: Mapping[str, object]) -> dict[str, object]:
    return {
        "expected_v164_classification": source_validation.get(
            "expected_v164_classification"
        ),
        "observed_v164_classification": source_validation.get(
            "observed_v164_classification"
        ),
        "expected_v164_exact_digest": source_validation.get(
            "expected_v164_exact_digest"
        ),
        "observed_v164_exact_digest": source_validation.get(
            "observed_v164_exact_digest"
        ),
        "v164_exact_digest_validation": source_validation.get(
            "v164_exact_digest_validation"
        ),
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "target_dataset_expansion_only": True,
        "validates_v164_exact_digest_before_build": True,
        "requires_v164_support_ready_classification": True,
        "future_outcomes_are_targets_not_inputs": True,
        "uses_seed_tick_agent_path_digest_provenance_as_trainable_input": False,
        "uses_runtime_requested_actions_as_scorer_input": False,
        "uses_runtime_requested_actions_for_comparison_only": True,
        "v164_strict_broad_seeds_become_support_provenance_seeds": True,
        "v164_strict_broad_seeds_are_future_promotion_heldout_seeds": False,
        "training_authorized": False,
        "training_ran": False,
        "runtime_artifact_created": False,
        "runtime_policy_integration_allowed": False,
        "live_ab_allowed": False,
        "runtime_override_path_created": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "threshold_tuning_recommended": False,
        "runtime_action_selection_changed": False,
    }


def _skipped_row_build_validation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v165_row_build_validation_v1",
        "passed": False,
        "failures": [{"reason": reason}],
        "failure_count": 1,
        "preterminal_row_count": 0,
    }


def _ordered_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        action: int(counter.get(action, 0))
        for action in ACTION_NAMES
        if int(counter.get(action, 0)) > 0
    }


def _action_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(action) for action in value if str(action) in ACTION_NAMES]


def _bool_score(value: object) -> float:
    return 1.0 if value is True else 0.0


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    output.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _json_round_trip_digest(payload: Mapping[str, object]) -> str:
    json_payload = json.loads(json.dumps(dict(payload), sort_keys=True, allow_nan=False))
    return stable_payload_digest(json_payload)
