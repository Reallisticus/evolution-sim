from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    _float,
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
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    load_jsonl_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V172_REPORT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V172_DATASET_PATH,
    SUPPORT_PROVENANCE_SEEDS,
    trainable_payload_leakage_scan,
    validate_v172_target_rows,
)
from evolution_sim.mind.carrion_survivor_continuation_v173_source_split_scorer import (
    EXPECTED_V172_DATASET_DIGEST,
    _complete_action_mask,
    _json_round_trip_digest,
    _public_observation_values,
    _rate,
    _safe_action_set,
    _source_seed,
)
from evolution_sim.mind.carrion_survivor_continuation_v174_mechanism_failure_battery import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V174_REPORT_PATH,
    EXPECTED_V173_EXACT_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v175_v174_route_correction_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V175_REPORT_PATH,
    DEFAULT_V176_PLAN_OUTPUT_PATH,
    EXPECTED_V174_EXACT_DIGEST,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v176_transition_diagnostic_planner_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v176_transition_diagnostic_planner_v1"
)
EXPECTED_V176_PLAN_DIGEST = (
    "5f28206ae4ccd4359678518f4f6a7e2c9f51f540b60f6fe980d327f33e3a7d2f"
)
EXPECTED_V175_EXACT_DIGEST = (
    "ab398ae0c5958f3551598437602d3ca59efcae627a9125ee18963f15d64ab869"
)
EXPECTED_V175_CLASSIFICATION = "v175_v174_route_overstated_static_ranking_not_ready"
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v176-carrion-survivor-continuation-transition-diagnostic-planner.json"
)
DEFAULT_GROUP_RELATIVE_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v176-carrion-survivor-continuation-group-relative-transition-experience.jsonl"
)
DEFAULT_V177_SHARD_PLAN_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v177-carrion-survivor-continuation-exact-branch-replay-shards.jsonl"
)
FAILED_SUPPORT_SEEDS = (19, 41)
FAILED_SEED_PRIORITY = (41, 19)
MAX_PRIORITY_ROWS_PER_FAILED_SEED = 16
MAX_SHARD_ROWS_PER_FAILED_SEED = 8
MIN_CLEAR_GROUPS_PER_FAILED_SEED = 8
ALLOWED_REASON_CODES = (
    "improves_survival_proxy",
    "preserves_energy_proxy",
    "improves_position_proxy",
    "unsafe_or_unavailable",
    "tied_outcome",
    "insufficient_transition_evidence",
)


class CarrionSurvivorContinuationV176TransitionDiagnosticPlannerError(ValueError):
    pass


def run_carrion_survivor_continuation_v176_transition_diagnostic_planner(
    *,
    v175_report_path: str | Path = DEFAULT_V175_REPORT_PATH,
    v176_plan_path: str | Path = DEFAULT_V176_PLAN_OUTPUT_PATH,
    v174_report_path: str | Path = DEFAULT_V174_REPORT_PATH,
    v172_report_path: str | Path = DEFAULT_V172_REPORT_PATH,
    v172_dataset_path: str | Path = DEFAULT_V172_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    group_relative_output_path: str | Path | None = DEFAULT_GROUP_RELATIVE_OUTPUT_PATH,
    v177_shard_plan_output_path: str | Path | None = DEFAULT_V177_SHARD_PLAN_OUTPUT_PATH,
    expected_v175_exact_digest: str | None = EXPECTED_V175_EXACT_DIGEST,
    expected_v175_classification: str = EXPECTED_V175_CLASSIFICATION,
    expected_v176_plan_digest: str | None = EXPECTED_V176_PLAN_DIGEST,
    expected_v174_exact_digest: str | None = EXPECTED_V174_EXACT_DIGEST,
    expected_v172_dataset_digest: str | None = EXPECTED_V172_DATASET_DIGEST,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> dict[str, object]:
    v175_report = load_json_report(v175_report_path)
    v176_plan_rows = _read_jsonl(v176_plan_path)
    v174_report = load_json_report(v174_report_path)
    v172_report = load_json_report(v172_report_path)
    rows = load_jsonl_dataset(v172_dataset_path)
    source_validation = validate_v176_sources(
        v175_report=v175_report,
        v176_plan_rows=v176_plan_rows,
        v174_report=v174_report,
        v172_report=v172_report,
        rows=rows,
        expected_v175_exact_digest=expected_v175_exact_digest,
        expected_v175_classification=expected_v175_classification,
        expected_v176_plan_digest=expected_v176_plan_digest,
        expected_v174_exact_digest=expected_v174_exact_digest,
        expected_v172_dataset_digest=expected_v172_dataset_digest,
        support_provenance_seeds=support_provenance_seeds,
    )
    leakage_scan = trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in rows]
    )
    row_schema_validation = validate_v172_target_rows(
        rows,
        support_provenance_seeds=support_provenance_seeds,
    )
    if (
        source_validation.get("passed") is True
        and leakage_scan.get("passed") is True
        and row_schema_validation.get("passed") is True
    ):
        ranking_predictions = _recompute_all_train_predictions(rows)
        lane_a = lane_a_failed_seed_replay_target_map(
            rows,
            ranking_predictions=ranking_predictions,
        )
        lane_b = lane_b_existing_transition_evidence_audit(rows)
        lane_c = lane_c_compact_world_model_target_contract()
        lane_e, experience_rows = lane_e_training_free_group_relative_probe(rows)
        lane_d = lane_d_exact_branch_replay_shard_planner(
            lane_a=lane_a,
            lane_b=lane_b,
            lane_e=lane_e,
        )
    else:
        lane_a = _skipped_lane("lane_a_failed_seed_replay_target_map")
        lane_b = _skipped_lane("lane_b_existing_transition_evidence_audit")
        lane_c = _skipped_lane("lane_c_compact_world_model_target_contract")
        lane_d = _skipped_lane("lane_d_exact_branch_replay_shard_planner")
        lane_e = _skipped_lane("lane_e_training_free_group_relative_probe")
        experience_rows = []
    lane_f = lane_f_route_decision(
        source_validation=source_validation,
        leakage_scan=leakage_scan,
        row_schema_validation=row_schema_validation,
        lane_b=lane_b,
        lane_d=lane_d,
        lane_e=lane_e,
    )
    classification = str(lane_f.get("classification"))
    experience_output = _maybe_write_jsonl(
        rows=experience_rows,
        output_path=group_relative_output_path,
        reason_if_skipped="no_group_relative_transition_experience_rows_emitted",
    )
    shard_rows = (
        _list_of_mappings(lane_d.get("v177_shard_plan_rows"))
        if lane_d.get("write_v177_shard_plan") is True
        else []
    )
    shard_output = _maybe_write_jsonl(
        rows=shard_rows,
        output_path=v177_shard_plan_output_path,
        reason_if_skipped="exact_branch_replay_expansion_not_needed",
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V176_TRANSITION_DIAGNOSTIC_PLANNER_POLICY,
        "contract": _diagnostics_only_contract(),
        "research_note": {
            "inspiration": (
                "arXiv:2510.08191 Training-Free Group Relative Policy "
                "Optimization"
            ),
            "used_only_as_group_relative_no_parameter_update_inspiration": True,
            "llm_calls_used": False,
            "natural_language_prompting_used": False,
            "semantic_summarization_used": False,
            "nondeterministic_priors_used": False,
        },
        "inputs": {
            "v175_report": str(v175_report_path),
            "v176_plan": str(v176_plan_path),
            "v174_report": str(v174_report_path),
            "v172_report": str(v172_report_path),
            "v172_dataset": str(v172_dataset_path),
            "expected_v175_exact_digest": expected_v175_exact_digest,
            "expected_v175_classification": expected_v175_classification,
            "expected_v176_plan_digest": expected_v176_plan_digest,
            "expected_v174_exact_digest": expected_v174_exact_digest,
            "expected_v172_dataset_digest": expected_v172_dataset_digest,
            "failed_support_seeds": list(FAILED_SUPPORT_SEEDS),
            "failed_seed_priority": list(FAILED_SEED_PRIORITY),
            "support_provenance_seeds": [
                int(seed) for seed in support_provenance_seeds
            ],
        },
        "source_validation": source_validation,
        "leakage_scan": leakage_scan,
        "row_schema_validation": row_schema_validation,
        "lanes": {
            "lane_a_failed_seed_replay_target_map": lane_a,
            "lane_b_existing_transition_evidence_audit": lane_b,
            "lane_c_compact_world_model_target_contract": lane_c,
            "lane_d_exact_branch_replay_shard_planner": lane_d,
            "lane_e_training_free_group_relative_transition_experience_probe": lane_e,
            "lane_f_route_decision": lane_f,
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(lane_f),
        "group_relative_transition_experience_output": experience_output,
        "v177_shard_plan_output": shard_output,
        "dataset_digest": stable_payload_digest(rows),
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v176_sources(
    *,
    v175_report: Mapping[str, object],
    v176_plan_rows: Sequence[Mapping[str, object]],
    v174_report: Mapping[str, object],
    v172_report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v175_exact_digest: str | None,
    expected_v175_classification: str,
    expected_v176_plan_digest: str | None,
    expected_v174_exact_digest: str | None,
    expected_v172_dataset_digest: str | None,
    support_provenance_seeds: Sequence[int],
) -> dict[str, object]:
    failures: list[str] = []
    v175_exact = exact_digest_validation_report(v175_report)
    observed_v175_exact = str(v175_report.get("exact_digest") or "")
    observed_v175_classification = str(
        _mapping(v175_report.get("classification")).get("primary") or ""
    )
    if v175_exact.get("passed") is not True:
        failures.append("v175_exact_digest_mismatch")
    if expected_v175_exact_digest and observed_v175_exact != expected_v175_exact_digest:
        failures.append("v175_unexpected_exact_digest")
    if observed_v175_classification != expected_v175_classification:
        failures.append("v175_unexpected_classification")
    v176_plan_digest = stable_payload_digest(list(v176_plan_rows))
    if expected_v176_plan_digest and v176_plan_digest != expected_v176_plan_digest:
        failures.append("v176_plan_digest_mismatch")
    v175_plan_output = _mapping(v175_report.get("v176_plan_output"))
    if v175_plan_output.get("plan_digest") != v176_plan_digest:
        failures.append("v175_reported_v176_plan_digest_mismatch")
    v174_exact = exact_digest_validation_report(v174_report)
    observed_v174_exact = str(v174_report.get("exact_digest") or "")
    if v174_exact.get("passed") is not True:
        failures.append("v174_exact_digest_mismatch")
    if expected_v174_exact_digest and observed_v174_exact != expected_v174_exact_digest:
        failures.append("v174_unexpected_exact_digest")
    dataset_digest = stable_payload_digest(rows)
    if expected_v172_dataset_digest and dataset_digest != expected_v172_dataset_digest:
        failures.append("v172_unexpected_dataset_digest")
    if str(v174_report.get("dataset_digest") or "") != dataset_digest:
        failures.append("v174_dataset_digest_mismatch")
    if str(_mapping(v172_report.get("dataset")).get("dataset_digest") or "") != dataset_digest:
        failures.append("v172_reported_dataset_digest_mismatch")
    v175_lifecycle = _lifecycle_validation(v175_report, policy="v176_v175_lifecycle_validation")
    v174_lifecycle = _lifecycle_validation(v174_report, policy="v176_v174_lifecycle_validation")
    if v175_lifecycle.get("passed") is not True:
        failures.append("v175_lifecycle_not_diagnostics_only")
    if v174_lifecycle.get("passed") is not True:
        failures.append("v174_lifecycle_not_diagnostics_only")
    plan_validation = _v176_plan_validation(v176_plan_rows)
    if plan_validation.get("passed") is not True:
        failures.append("v176_plan_rows_invalid")
    source_seed_validation = _source_seed_validation(
        rows,
        support_provenance_seeds=support_provenance_seeds,
    )
    if source_seed_validation.get("passed") is not True:
        failures.append("support_seed_rows_invalid")
    return {
        "policy": "m3_carrion_survivor_continuation_v176_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v175_exact_digest": expected_v175_exact_digest,
        "observed_v175_exact_digest": observed_v175_exact,
        "expected_v175_classification": expected_v175_classification,
        "observed_v175_classification": observed_v175_classification,
        "v175_exact_digest_validation": v175_exact,
        "expected_v176_plan_digest": expected_v176_plan_digest,
        "observed_v176_plan_digest": v176_plan_digest,
        "expected_v174_exact_digest": expected_v174_exact_digest,
        "observed_v174_exact_digest": observed_v174_exact,
        "v174_exact_digest_validation": v174_exact,
        "expected_v172_dataset_digest": expected_v172_dataset_digest,
        "observed_v172_dataset_digest": dataset_digest,
        "v175_lifecycle_validation": v175_lifecycle,
        "v174_lifecycle_validation": v174_lifecycle,
        "v176_plan_validation": plan_validation,
        "support_provenance_seed_validation": source_seed_validation,
    }


def lane_a_failed_seed_replay_target_map(
    rows: Sequence[Mapping[str, object]],
    *,
    ranking_predictions: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    groups: dict[tuple[object, ...], dict[str, object]] = {}
    priority_rows = []
    failed_counts_by_seed: Counter[int] = Counter()
    for row_index, row in enumerate(rows):
        seed = _source_seed(row)
        if seed not in FAILED_SUPPORT_SEEDS:
            continue
        safe_set = _safe_action_set(row)
        if len(safe_set) != 1:
            continue
        prediction = _mapping(ranking_predictions.get(row_index))
        top_1 = _list_of_strings(prediction.get("top_1_actions"))
        top_3 = _list_of_strings(prediction.get("top_3_actions"))
        safe_action = safe_set[0]
        failure_types = []
        if safe_action not in top_1:
            failure_types.append("unique_top_1_miss")
        if safe_action not in top_3:
            failure_types.append("unique_top_3_miss")
        if not failure_types:
            continue
        failed_counts_by_seed.update([seed])
        for failure_type in failure_types:
            metadata = _mapping(row.get("metadata"))
            public_mask = _complete_action_mask(_mapping(row.get("public_action_mask")))
            key = (
                seed,
                _int(metadata.get("branch_tick")),
                _mask_width(public_mask),
                len(safe_set),
                safe_action,
                failure_type,
            )
            if key not in groups:
                groups[key] = {
                    "seed": seed,
                    "branch_tick": _int(metadata.get("branch_tick")),
                    "public_action_mask_width": _mask_width(public_mask),
                    "safe_set_width": len(safe_set),
                    "failed_safe_action": safe_action,
                    "action_failure_type": failure_type,
                    "row_count": 0,
                    "row_indexes": [],
                }
            groups[key]["row_count"] = _int(groups[key].get("row_count")) + 1
            _list = groups[key].setdefault("row_indexes", [])
            if isinstance(_list, list) and len(_list) < 12:
                _list.append(row_index)
        priority_rows.append(_priority_row(rows, row_index, prediction, failure_types))
    priority_rows = sorted(priority_rows, key=_priority_sort_key)
    capped = []
    per_seed_taken: Counter[int] = Counter()
    for row in priority_rows:
        seed = _int(row.get("seed"))
        if per_seed_taken[seed] >= MAX_PRIORITY_ROWS_PER_FAILED_SEED:
            continue
        capped.append(row)
        per_seed_taken.update([seed])
    return {
        "policy": "m3_carrion_survivor_continuation_v176_lane_a_failed_seed_replay_target_map_v1",
        "failed_support_seeds_from_v175": list(FAILED_SUPPORT_SEEDS),
        "failed_seed_priority": list(FAILED_SEED_PRIORITY),
        "unique_row_failure_counts_by_seed": {
            str(seed): int(failed_counts_by_seed.get(seed, 0))
            for seed in FAILED_SUPPORT_SEEDS
        },
        "group_count": len(groups),
        "groups": sorted(groups.values(), key=_group_sort_key),
        "priority_rows": capped,
        "priority_rows_are_metadata_for_replay_expansion_only": True,
        "floor_passed": False,
    }


def lane_b_existing_transition_evidence_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    outcome_widths: Counter[int] = Counter()
    available_counts: Counter[str] = Counter()
    verified_counts: Counter[str] = Counter()
    rows_with_outcomes = 0
    for row in rows:
        row_has_outcome = False
        for target in _list_of_mappings(row.get("action_value_targets")):
            action = str(target.get("action") or "")
            if target.get("target_available") is True:
                available_counts.update([action])
            if target.get("replay_verified") is True:
                verified_counts.update([action])
            summary = target.get("replay_outcome_summary")
            if isinstance(summary, list):
                row_has_outcome = True
                outcome_widths.update([len(summary)])
        if row_has_outcome:
            rows_with_outcomes += 1
    field_scan = _transition_field_scan(rows)
    missing = [
        field
        for field, present in field_scan.items()
        if field
        in {
            "next_public_observation",
            "next_public_action_mask",
            "previous_same_agent_public_context",
        }
        and present is not True
    ]
    only_summary = (
        rows_with_outcomes == len(rows)
        and not field_scan.get("next_public_observation")
        and not field_scan.get("next_public_action_mask")
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v176_lane_b_existing_transition_evidence_audit_v1",
        "row_count": len(rows),
        "rows_with_replay_outcome_summary": rows_with_outcomes,
        "action_value_targets_available_by_action": dict(_ordered_counter(available_counts)),
        "action_value_targets_replay_verified_by_action": dict(_ordered_counter(verified_counts)),
        "replay_outcome_summary_width_counts": {
            str(width): int(count) for width, count in sorted(outcome_widths.items())
        },
        "field_presence": field_scan,
        "missing_transition_fields": missing,
        "current_rows_contain_only_terminal_or_summary_outcome_keys": only_summary,
        "existing_transition_evidence_enough_for_compact_diagnostic_dataset": (
            field_scan.get("current_public_observation") is True
            and field_scan.get("current_public_action_mask") is True
            and field_scan.get("next_public_observation") is True
            and field_scan.get("next_public_action_mask") is True
            and rows_with_outcomes == len(rows)
        ),
        "diagnostic_targets_not_trainable_inputs": True,
        "floor_passed": False,
    }


def lane_c_compact_world_model_target_contract() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v176_lane_c_compact_world_model_target_contract_v1",
        "minimum_future_dataset_row_contract": {
            "schema_version": "m3_carrion_survivor_continuation_v177_compact_transition_diagnostic_row_v1",
            "trainable_public_inputs": [
                "current_public_observation",
                "current_public_action_mask",
                "forced_action",
                "next_public_observation_if_available",
                "next_public_action_mask_if_available",
            ],
            "diagnostic_targets_metadata_only": [
                "short_horizon_public_outcome_summary",
                "group_relative_action_rank",
                "winner_loser_action_comparisons",
            ],
            "metadata_only": [
                "seed",
                "fixture",
                "branch_id",
                "branch_tick",
                "agent_id",
                "source_path",
                "line_number",
                "record_digest",
                "replay_verification_digest",
                "provenance",
            ],
        },
        "leakage_rules": {
            "seed_fixture_branch_tick_agent_path_digest_provenance_private_outcome_target_fields_are_trainable_inputs": False,
            "metadata_only_fields_may_be_used_for_exact_replay_materialization": True,
            "private_state_allowed": False,
            "runtime_requested_or_resolved_action_allowed_as_input": False,
            "forced_action_is_public_experimental_condition": True,
        },
        "floor_passed": True,
    }


def lane_d_exact_branch_replay_shard_planner(
    *,
    lane_a: Mapping[str, object],
    lane_b: Mapping[str, object],
    lane_e: Mapping[str, object],
) -> dict[str, object]:
    expansion_needed = (
        lane_b.get("existing_transition_evidence_enough_for_compact_diagnostic_dataset")
        is not True
    )
    priority_rows = _list_of_mappings(lane_a.get("priority_rows"))
    shard_rows = []
    per_seed_taken: Counter[int] = Counter()
    for row in priority_rows:
        seed = _int(row.get("seed"))
        if seed not in FAILED_SUPPORT_SEEDS:
            continue
        if per_seed_taken[seed] >= MAX_SHARD_ROWS_PER_FAILED_SEED:
            continue
        shard_rows.append(
            {
                "schema_version": "m3_carrion_survivor_continuation_v177_exact_branch_replay_shard_row_v1",
                "route": "exact_branch_replay_expansion",
                "priority": len(shard_rows) + 1,
                "seed": seed,
                "branch_tick": row.get("branch_tick"),
                "agent_id": row.get("agent_id"),
                "branch_id": row.get("branch_id"),
                "source_path": row.get("source_path"),
                "line_number": row.get("line_number"),
                "row_index": row.get("row_index"),
                "failed_safe_action": row.get("safe_action"),
                "failure_types": row.get("failure_types"),
                "candidate_forced_actions": row.get("public_actions"),
                "replay_expansion_goal": (
                    "materialize next public observation, next public action "
                    "mask, and short-horizon public outcome for candidate actions"
                ),
                "training_authorized": False,
                "runtime_artifact_authorized": False,
            }
        )
        per_seed_taken.update([seed])
    return {
        "policy": "m3_carrion_survivor_continuation_v176_lane_d_exact_branch_replay_shard_planner_v1",
        "exact_branch_replay_expansion_needed": expansion_needed,
        "reason": (
            "missing_next_public_observation_or_next_action_mask"
            if expansion_needed
            else "existing_transition_rows_already_have_compact_fields"
        ),
        "failed_seed_priority": list(FAILED_SEED_PRIORITY),
        "max_shard_rows_per_failed_seed": MAX_SHARD_ROWS_PER_FAILED_SEED,
        "v177_shard_plan_rows": shard_rows if expansion_needed else [],
        "write_v177_shard_plan": expansion_needed and bool(shard_rows),
        "group_relative_rows_available": lane_e.get("candidate_transition_experience_row_count"),
        "floor_passed": bool(shard_rows) if expansion_needed else True,
    }


def lane_e_training_free_group_relative_probe(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    experience_rows = []
    group_counts: Counter[str] = Counter()
    clear_by_seed: Counter[int] = Counter()
    tied_by_seed: Counter[int] = Counter()
    insufficient_by_seed: Counter[int] = Counter()
    for row_index, row in enumerate(rows):
        group = _group_relative_payload(row, row_index=row_index)
        seed = _source_seed(row)
        if group["group_status"] == "clear_winner_loser":
            clear_by_seed.update([seed])
        elif group["group_status"] == "all_tied_outcomes":
            tied_by_seed.update([seed])
        else:
            insufficient_by_seed.update([seed])
        group_counts.update([group["group_status"]])
        experience_rows.extend(group["experience_rows"])
    failed_seed_support = {
        str(seed): {
            "clear_winner_loser_group_count": int(clear_by_seed.get(seed, 0)),
            "all_tied_group_count": int(tied_by_seed.get(seed, 0)),
            "insufficient_group_count": int(insufficient_by_seed.get(seed, 0)),
            "enough_winner_loser_contrast": int(clear_by_seed.get(seed, 0))
            >= MIN_CLEAR_GROUPS_PER_FAILED_SEED,
        }
        for seed in FAILED_SUPPORT_SEEDS
    }
    enough_failed = all(
        payload["enough_winner_loser_contrast"]
        for payload in failed_seed_support.values()
    )
    digest = stable_payload_digest(experience_rows)
    return (
        {
            "policy": "m3_carrion_survivor_continuation_v176_lane_e_training_free_group_relative_probe_v1",
            "training_free_group_relative_policy": (
                "one branch replay point is a group; available forced actions "
                "are group members; replay outcome summaries define deterministic "
                "relative advantages; no parameter updates"
            ),
            "llm_calls_used": False,
            "natural_language_prompting_used": False,
            "semantic_summarization_used": False,
            "candidate_transition_experience_row_count": len(experience_rows),
            "candidate_transition_experience_digest": digest,
            "group_status_counts": dict(sorted(group_counts.items())),
            "branch_groups_with_clear_winner_loser_actions": int(
                group_counts.get("clear_winner_loser", 0)
            ),
            "branch_groups_with_all_tied_outcomes": int(
                group_counts.get("all_tied_outcomes", 0)
            ),
            "per_seed_group_relative_support": {
                str(seed): {
                    "clear_winner_loser_group_count": int(clear_by_seed.get(seed, 0)),
                    "all_tied_group_count": int(tied_by_seed.get(seed, 0)),
                    "insufficient_group_count": int(insufficient_by_seed.get(seed, 0)),
                }
                for seed in sorted({_source_seed(row) for row in rows})
            },
            "failed_seed_support": failed_seed_support,
            "failed_seeds_have_enough_winner_loser_contrast": enough_failed,
            "supports_v177_compact_transition_world_model_diagnostics": False,
            "support_blocker": (
                "candidate rows lack next public observation/action-mask "
                "transition fields"
            ),
            "allowed_reason_codes": list(ALLOWED_REASON_CODES),
            "floor_passed": enough_failed,
        },
        experience_rows,
    )


def lane_f_route_decision(
    *,
    source_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    lane_b: Mapping[str, object],
    lane_d: Mapping[str, object],
    lane_e: Mapping[str, object],
) -> dict[str, object]:
    if (
        source_validation.get("passed") is not True
        or leakage_scan.get("passed") is not True
        or row_schema_validation.get("passed") is not True
    ):
        classification = "v176_invalid_closed_no_training"
        route = "close_invalid_without_training"
    elif (
        lane_b.get("existing_transition_evidence_enough_for_compact_diagnostic_dataset")
        is True
    ):
        classification = (
            "v176_existing_transition_evidence_ready_for_v177_dataset_no_training"
        )
        route = "v177_compact_transition_dataset_no_training"
    elif (
        lane_e.get("supports_v177_compact_transition_world_model_diagnostics") is True
    ):
        classification = (
            "v176_group_relative_transition_experience_ready_for_v177_dataset_no_training"
        )
        route = "v177_group_relative_transition_experience_dataset_no_training"
    elif lane_d.get("exact_branch_replay_expansion_needed") is True:
        classification = "v176_exact_branch_replay_plan_ready_no_training"
        route = "v177_exact_branch_replay_expansion_no_training"
    else:
        classification = "v176_invalid_closed_no_training"
        route = "close_without_training"
    return {
        "policy": "m3_carrion_survivor_continuation_v176_lane_f_route_decision_v1",
        "classification": classification,
        "recommended_next_route": route,
        "training_authorized": False,
        "fit_authorized": False,
        "runtime_artifact_authorized": False,
        "runtime_action_selection_change_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
    }


def _recompute_all_train_predictions(
    rows: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, object]]:
    row_seeds = [_source_seed(row) for row in rows]
    vectors = _normalized_public_vectors(rows)
    predictions = {}
    for row_index, row in enumerate(rows):
        seed = row_seeds[row_index]
        distances = sorted(
            (
                sum((a - b) * (a - b) for a, b in zip(vectors[row_index], vectors[i])),
                i,
            )
            for i, other_seed in enumerate(row_seeds)
            if other_seed != seed
        )
        scores = _ranking_scores(rows, row_index=row_index, distances=distances)
        ranked = sorted(
            scores,
            key=lambda action: (-float(scores[action]), _action_order(action)),
        )
        predictions[row_index] = {
            "scores": scores,
            "top_1_actions": ranked[:1],
            "top_3_actions": ranked[:3],
            "ranked_actions": ranked,
        }
    return predictions


def _ranking_scores(
    rows: Sequence[Mapping[str, object]],
    *,
    row_index: int,
    distances: Sequence[tuple[float, int]],
) -> dict[str, float]:
    public_mask = _complete_action_mask(_mapping(rows[row_index].get("public_action_mask")))
    numerators = {action: 0.0 for action in ACTION_NAMES if public_mask.get(action)}
    denominators = {action: 0.0 for action in numerators}
    for distance, train_index in distances:
        train_mask = _complete_action_mask(
            _mapping(rows[train_index].get("public_action_mask"))
        )
        safe = set(_safe_action_set(rows[train_index]))
        weight = 1.0 / (1.0 + float(distance))
        for action in numerators:
            if train_mask.get(action):
                denominators[action] += weight
                numerators[action] += weight * (1.0 if action in safe else 0.0)
    return {
        action: _round(
            numerators[action] / denominators[action]
            if denominators[action] > 0.0
            else 0.0
        )
        for action in numerators
    }


def _normalized_public_vectors(rows: Sequence[Mapping[str, object]]) -> list[list[float]]:
    raw = []
    for row in rows:
        features = _mapping(row.get("trainable_public_features"))
        observation = _mapping(features.get("public_observation"))
        values = _public_observation_values(observation)
        mask = _complete_action_mask(_mapping(row.get("public_action_mask")))
        values.extend(1.0 if mask.get(action) else 0.0 for action in ACTION_NAMES)
        raw.append(values)
    width = max((len(vector) for vector in raw), default=0)
    padded = [vector + [0.0] * (width - len(vector)) for vector in raw]
    if not padded:
        return []
    mins = [min(vector[i] for vector in padded) for i in range(width)]
    maxs = [max(vector[i] for vector in padded) for i in range(width)]
    normalized = []
    for vector in padded:
        normalized.append(
            [
                0.0 if maxs[i] == mins[i] else (float(value) - mins[i]) / (maxs[i] - mins[i])
                for i, value in enumerate(vector)
            ]
        )
    return normalized


def _priority_row(
    rows: Sequence[Mapping[str, object]],
    row_index: int,
    prediction: Mapping[str, object],
    failure_types: Sequence[str],
) -> dict[str, object]:
    row = rows[row_index]
    metadata = _mapping(row.get("metadata"))
    public_mask = _complete_action_mask(_mapping(row.get("public_action_mask")))
    safe_set = _safe_action_set(row)
    return {
        "row_index": row_index,
        "seed": _source_seed(row),
        "branch_tick": _int(metadata.get("branch_tick")),
        "agent_id": metadata.get("agent_id"),
        "branch_id": metadata.get("branch_id"),
        "source_path": metadata.get("source_path"),
        "line_number": metadata.get("line_number"),
        "source_record_digest": metadata.get("source_record_digest"),
        "materialized_record_digest": metadata.get("materialized_record_digest"),
        "public_action_mask_width": _mask_width(public_mask),
        "safe_set_width": len(safe_set),
        "safe_action": safe_set[0] if safe_set else None,
        "failure_types": list(failure_types),
        "top_1_actions": prediction.get("top_1_actions"),
        "top_3_actions": prediction.get("top_3_actions"),
        "public_actions": [action for action in ACTION_NAMES if public_mask.get(action)],
    }


def _group_relative_payload(
    row: Mapping[str, object],
    *,
    row_index: int,
) -> dict[str, object]:
    available = [
        target
        for target in _list_of_mappings(row.get("action_value_targets"))
        if target.get("target_available") is True
        and isinstance(target.get("replay_outcome_summary"), list)
    ]
    seed = _source_seed(row)
    metadata = _mapping(row.get("metadata"))
    if len(available) < 2:
        return {
            "group_status": "insufficient_transition_evidence",
            "experience_rows": [
                _experience_row(
                    row,
                    row_index=row_index,
                    target={},
                    relative_rank=None,
                    advantage=0.0,
                    winner_actions=[],
                    loser_actions=[],
                    reason_code="insufficient_transition_evidence",
                )
            ],
        }
    scored = [
        {
            "action": str(target.get("action")),
            "target": target,
            "score": _outcome_score(target.get("replay_outcome_summary")),
        }
        for target in available
    ]
    scores = [tuple(item["score"]) for item in scored]
    best = max(scores)
    worst = min(scores)
    mean_score = sum(_scalar_score(score) for score in scores) / len(scores)
    winner_actions = sorted(
        [str(item["action"]) for item in scored if tuple(item["score"]) == best],
        key=_action_order,
    )
    loser_actions = sorted(
        [str(item["action"]) for item in scored if tuple(item["score"]) == worst],
        key=_action_order,
    )
    tied = best == worst
    ranked_scores = sorted(set(scores), reverse=True)
    rows = []
    for item in sorted(scored, key=lambda payload: _action_order(str(payload["action"]))):
        rank = ranked_scores.index(tuple(item["score"])) + 1
        reason = (
            "tied_outcome"
            if tied
            else _reason_code(item["score"], worst)
            if str(item["action"]) in winner_actions
            else "unsafe_or_unavailable"
            if item["target"].get("public_mask") is not True
            else "tied_outcome"
            if len(winner_actions) > 1
            else "insufficient_transition_evidence"
            if not item["score"]
            else "unsafe_or_unavailable"
        )
        rows.append(
            _experience_row(
                row,
                row_index=row_index,
                target=_mapping(item["target"]),
                relative_rank=rank,
                advantage=_scalar_score(item["score"]) - mean_score,
                winner_actions=winner_actions,
                loser_actions=loser_actions,
                reason_code=reason,
            )
        )
    return {
        "group_status": "all_tied_outcomes" if tied else "clear_winner_loser",
        "experience_rows": rows,
        "seed": seed,
        "branch_tick": metadata.get("branch_tick"),
    }


def _experience_row(
    row: Mapping[str, object],
    *,
    row_index: int,
    target: Mapping[str, object],
    relative_rank: int | None,
    advantage: float,
    winner_actions: Sequence[str],
    loser_actions: Sequence[str],
    reason_code: str,
) -> dict[str, object]:
    metadata = _mapping(row.get("metadata"))
    forced_action = str(target.get("action") or "")
    return {
        "schema_version": "m3_carrion_survivor_continuation_v176_group_relative_transition_experience_row_v1",
        "row_origin": "v176_from_v172_replay_outcome_summaries",
        "trainable_public_inputs": {
            "current_public_feature_summary": _current_public_feature_summary(row),
            "action_mask_geometry": _action_mask_geometry(row),
            "forced_action": forced_action,
        },
        "diagnostic_targets_metadata_only": {
            "relative_outcome_rank": relative_rank,
            "group_relative_advantage": _round(advantage),
            "winner_actions": list(winner_actions),
            "loser_actions": list(loser_actions),
            "winner_loser_action_comparisons": [
                {"winner_action": winner, "loser_action": loser}
                for winner in winner_actions
                for loser in loser_actions
                if winner != loser
            ][:16],
            "short_public_outcome_summary": target.get("replay_outcome_summary"),
            "reason_code": reason_code
            if reason_code in ALLOWED_REASON_CODES
            else "insufficient_transition_evidence",
        },
        "metadata": {
            "metadata_only": True,
            "source_row_index": row_index,
            "seed": metadata.get("seed"),
            "branch_tick": metadata.get("branch_tick"),
            "agent_id": metadata.get("agent_id"),
            "branch_id": metadata.get("branch_id"),
            "source_path": metadata.get("source_path"),
            "line_number": metadata.get("line_number"),
            "source_record_digest": metadata.get("source_record_digest"),
            "materialized_record_digest": metadata.get("materialized_record_digest"),
        },
        "leakage_policy": {
            "metadata_provenance_private_outcome_target_fields_used_as_trainable_inputs": False,
            "runtime_action_selection_changed": False,
        },
    }


def _current_public_feature_summary(row: Mapping[str, object]) -> dict[str, object]:
    observation = _mapping(
        _mapping(row.get("trainable_public_features")).get("public_observation")
    )
    values = _public_observation_values(observation)
    if not values:
        return {"value_count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "value_count": len(values),
        "mean": _round(sum(values) / len(values)),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "first_values": [_round(value) for value in values[:8]],
    }


def _action_mask_geometry(row: Mapping[str, object]) -> dict[str, object]:
    mask = _complete_action_mask(_mapping(row.get("public_action_mask")))
    movement = ["move_north", "move_south", "move_east", "move_west"]
    resource = ["eat", "drink", "mate", "stay"]
    attack = ["attack_north", "attack_south", "attack_east", "attack_west"]
    return {
        "public_action_mask_width": _mask_width(mask),
        "movement_action_count": sum(1 for action in movement if mask.get(action)),
        "resource_action_count": sum(1 for action in resource if mask.get(action)),
        "attack_action_count": sum(1 for action in attack if mask.get(action)),
        "signal_action_count": sum(
            1 for action in ACTION_NAMES if action.startswith("signal_") and mask.get(action)
        ),
        "public_actions": [action for action in ACTION_NAMES if mask.get(action)],
    }


def _outcome_score(summary: object) -> tuple[float, float, float, float]:
    values = [float(value) for value in summary] if isinstance(summary, list) else []
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    survival = values[0] if len(values) > 0 else 0.0
    births = values[1] if len(values) > 1 else 0.0
    energy = values[4] if len(values) > 4 else 0.0
    position = values[5] if len(values) > 5 else 0.0
    return (_round(survival), _round(births), _round(energy), _round(position))


def _scalar_score(score: Sequence[float]) -> float:
    values = list(score)
    weights = (1000.0, 100.0, 10.0, 1.0)
    return sum(float(value) * weights[index] for index, value in enumerate(values[:4]))


def _reason_code(score: Sequence[float], worst: Sequence[float]) -> str:
    if not score:
        return "insufficient_transition_evidence"
    if score[0] > worst[0]:
        return "improves_survival_proxy"
    if len(score) > 2 and score[2] > worst[2]:
        return "preserves_energy_proxy"
    if len(score) > 3 and score[3] > worst[3]:
        return "improves_position_proxy"
    return "tied_outcome"


def _transition_field_scan(rows: Sequence[Mapping[str, object]]) -> dict[str, bool]:
    keys = set()
    for row in rows[:32]:
        keys.update(_flatten_keys(row))
    return {
        "current_public_observation": all(
            bool(_mapping(_mapping(row.get("trainable_public_features")).get("public_observation")))
            for row in rows
        ),
        "current_public_action_mask": all(bool(_mapping(row.get("public_action_mask"))) for row in rows),
        "action_value_targets": all(bool(_list_of_mappings(row.get("action_value_targets"))) for row in rows),
        "replay_outcome_summary": any("replay_outcome_summary" in key for key in keys),
        "next_public_observation": any(
            key.endswith("next_public_observation")
            or key.endswith("after_public_observation")
            or key.endswith("next_observation")
            for key in keys
        ),
        "next_public_action_mask": any(
            key.endswith("next_public_action_mask")
            or key.endswith("after_public_action_mask")
            or key.endswith("next_action_mask")
            for key in keys
        ),
        "previous_same_agent_public_context": any(
            "previous_same_agent" in key or "public_context" in key
            for key in keys
        ),
    }


def _flatten_keys(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        keys = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_flatten_keys(item, path))
        return keys
    if isinstance(value, list):
        keys = []
        for index, item in enumerate(value[:4]):
            keys.extend(_flatten_keys(item, f"{prefix}[{index}]"))
        return keys
    return []


def _v176_plan_validation(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    failures = []
    routes = {str(row.get("route") or "") for row in rows}
    if "exact_branch_replay_expansion" not in routes:
        failures.append("missing_exact_branch_replay_expansion_route")
    if "world_model_transition_diagnostic" not in routes:
        failures.append("missing_world_model_transition_diagnostic_route")
    for index, row in enumerate(rows):
        if row.get("training_authorized") is not False:
            failures.append(f"row_{index}_training_authorized_not_false")
        if row.get("runtime_artifact_authorized") is not False:
            failures.append(f"row_{index}_runtime_artifact_authorized_not_false")
    return {
        "policy": "m3_carrion_survivor_continuation_v176_v176_plan_validation_v1",
        "passed": not failures and bool(rows),
        "failures": failures,
        "row_count": len(rows),
        "routes": sorted(routes),
    }


def _source_seed_validation(
    rows: Sequence[Mapping[str, object]],
    *,
    support_provenance_seeds: Sequence[int],
) -> dict[str, object]:
    expected = sorted({int(seed) for seed in support_provenance_seeds})
    counts = Counter(_source_seed(row) for row in rows)
    observed = sorted(seed for seed in counts if seed > 0)
    failures = []
    if observed != expected:
        failures.append("source_seed_set_mismatch")
    return {
        "policy": "m3_carrion_survivor_continuation_v176_support_seed_validation_v1",
        "passed": not failures,
        "failures": failures,
        "expected_support_provenance_seeds": expected,
        "observed_source_seeds": observed,
        "row_counts_by_seed": {
            str(seed): int(counts.get(seed, 0)) for seed in expected
        },
    }


def _lifecycle_validation(report: Mapping[str, object], *, policy: str) -> dict[str, object]:
    failures = []
    if report.get("diagnostics_only") is not True:
        failures.append({"field": "diagnostics_only", "observed": report.get("diagnostics_only")})
    for field in (
        "training_ran",
        "training_authorized",
        "fit_ran",
        "scorer_retraining_ran",
        "scorer_retraining_authorized",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "runtime_observation_schema_changed",
        "runtime_policy_changed",
        "shadow_eval_ran",
        "live_ab_ran",
        "live_ab_allowed",
        "k_tuning_ran",
        "threshold_tuning_ran",
        "replay_viewer_schema_changed",
        "promotion_authorized",
        "staging_authorized",
        "commit_authorized",
        "reset_authorized",
        "clean_authorized",
    ):
        if field in report and report.get(field) is not False:
            failures.append({"field": field, "observed": report.get(field)})
    return {
        "policy": policy,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "transition_diagnostic_planner_only": True,
        "training_allowed": False,
        "fit_allowed": False,
        "runtime_artifact_allowed": False,
        "runtime_action_change_allowed": False,
        "shadow_or_live_eval_allowed": False,
        "promotion_allowed": False,
        "gate_relaxation_allowed": False,
        "support_provenance_seed_reuse_as_promotion_heldout_allowed": False,
        "staging_allowed": False,
        "commit_allowed": False,
        "reset_allowed": False,
        "clean_allowed": False,
    }


def _lifecycle_flags() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "artifact_created": False,
        "diagnostic_artifact_created": False,
        "runtime_artifact_created": False,
        "training_ran": False,
        "training_authorized": False,
        "fit_ran": False,
        "scorer_retraining_ran": False,
        "scorer_retraining_authorized": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": False,
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "k_tuning_ran": False,
        "threshold_tuning_ran": False,
        "replay_expansion_ran": False,
        "world_model_training_ran": False,
        "replay_viewer_schema_changed": False,
        "promotion_authorized": False,
        "staging_authorized": False,
        "commit_authorized": False,
        "reset_authorized": False,
        "clean_authorized": False,
        "non_promoted": True,
    }


def _route_recommendation(lane_f: Mapping[str, object]) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v176_route_recommendation_v1",
        "recommended_next_route": lane_f.get("recommended_next_route"),
        "classification": lane_f.get("classification"),
        "training_authorized": False,
        "fit_authorized": False,
        "runtime_integration_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
    }


def _maybe_write_jsonl(
    *,
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path | None,
    reason_if_skipped: str,
) -> dict[str, object]:
    if not rows or output_path is None:
        return {
            "written": False,
            "reason": reason_if_skipped if not rows else "output_path_disabled",
            "jsonl_row_count": 0,
        }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return {
        "written": True,
        "path": str(path),
        "jsonl_row_count": len(rows),
        "digest": stable_payload_digest(list(rows)),
    }


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _skipped_lane(policy: str) -> dict[str, object]:
    return {
        "policy": f"m3_carrion_survivor_continuation_v176_{policy}_v1",
        "skipped": True,
        "reason": "source_validation_or_row_contract_failed",
        "floor_passed": False,
    }


def _priority_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    seed = _int(row.get("seed"))
    seed_priority = FAILED_SEED_PRIORITY.index(seed) if seed in FAILED_SEED_PRIORITY else 99
    severity = 0 if "unique_top_3_miss" in _list_of_strings(row.get("failure_types")) else 1
    return (seed_priority, severity, _int(row.get("branch_tick")), _int(row.get("row_index")))


def _group_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    seed = _int(row.get("seed"))
    seed_priority = FAILED_SEED_PRIORITY.index(seed) if seed in FAILED_SEED_PRIORITY else 99
    return (
        seed_priority,
        _int(row.get("branch_tick")),
        _int(row.get("public_action_mask_width")),
        _int(row.get("safe_set_width")),
        str(row.get("failed_safe_action")),
        str(row.get("action_failure_type")),
    )


def _mask_width(mask: Mapping[str, bool]) -> int:
    return sum(1 for action in ACTION_NAMES if mask.get(action))


def _ordered_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        action: int(counter.get(action, 0))
        for action in ACTION_NAMES
        if int(counter.get(action, 0)) > 0
    }


def _list_of_strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
