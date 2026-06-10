from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from math import isfinite
from pathlib import Path
from statistics import mean

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import decode_observation_input
from evolution_sim.mind.candidate_campaign import _int, _mapping, _round, write_json
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
from evolution_sim.mind.carrion_survivor_continuation_v177_exact_branch_replay_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V177_REPORT_PATH,
    DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION,
    trainable_payload_leakage_scan,
    validate_v177_transition_rows,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V178_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V178_TRANSITION_ROW_DATASET_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_v1"
)
EXPECTED_V177_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v177_exact_branch_replay_expansion_"
    "compact_transition_rows_ready_no_training"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v178-carrion-survivor-continuation-transition-row-dataset-audit.json"
)
DEFAULT_MIN_ROW_COUNT = 128
DEFAULT_MIN_SEED_COUNT = 3
DEFAULT_MIN_BRANCH_COUNT = 24
DEFAULT_MIN_FORCED_ACTION_COUNT = 6


def run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
    *,
    transition_dataset_path: str | Path = DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    v177_report_path: str | Path = DEFAULT_V177_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v177_report_exact_digest: str | None = None,
    expected_dataset_digest: str | None = None,
    min_row_count: int = DEFAULT_MIN_ROW_COUNT,
    min_seed_count: int = DEFAULT_MIN_SEED_COUNT,
    min_branch_count: int = DEFAULT_MIN_BRANCH_COUNT,
    min_forced_action_count: int = DEFAULT_MIN_FORCED_ACTION_COUNT,
) -> dict[str, object]:
    v177_report = load_json_report(v177_report_path)
    rows = [dict(row) for row in load_jsonl_dataset(transition_dataset_path)]
    source_validation = validate_v178_sources(
        v177_report=v177_report,
        rows=rows,
        expected_v177_report_exact_digest=expected_v177_report_exact_digest,
        expected_dataset_digest=expected_dataset_digest,
    )
    row_schema_validation = validate_v177_transition_rows(rows)
    leakage_scan = trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in rows]
    )
    identity_audit = transition_row_identity_audit(rows)
    action_mask_audit = transition_row_action_mask_audit(rows)
    observation_audit = transition_row_observation_audit(rows)
    coverage_audit = transition_row_coverage_audit(rows)
    target_audit = transition_row_target_audit(rows)
    support_readiness = transition_row_support_readiness(
        rows=rows,
        coverage_audit=coverage_audit,
        identity_audit=identity_audit,
        min_row_count=int(min_row_count),
        min_seed_count=int(min_seed_count),
        min_branch_count=int(min_branch_count),
        min_forced_action_count=int(min_forced_action_count),
    )
    classification = _classification(
        source_validation=source_validation,
        row_schema_validation=row_schema_validation,
        leakage_scan=leakage_scan,
        identity_audit=identity_audit,
        action_mask_audit=action_mask_audit,
        observation_audit=observation_audit,
        support_readiness=support_readiness,
    )
    dataset_digest = stable_payload_digest(rows)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V178_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V178_TRANSITION_ROW_DATASET_AUDIT_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v177_report": str(v177_report_path),
            "transition_dataset": str(transition_dataset_path),
            "expected_v177_report_exact_digest": expected_v177_report_exact_digest,
            "expected_dataset_digest": expected_dataset_digest,
            "min_row_count": int(min_row_count),
            "min_seed_count": int(min_seed_count),
            "min_branch_count": int(min_branch_count),
            "min_forced_action_count": int(min_forced_action_count),
        },
        "source_validation": source_validation,
        "row_schema_validation": row_schema_validation,
        "leakage_scan": leakage_scan,
        "identity_audit": identity_audit,
        "action_mask_audit": action_mask_audit,
        "observation_audit": observation_audit,
        "coverage_audit": coverage_audit,
        "target_audit": target_audit,
        "support_readiness": support_readiness,
        "dataset": {
            "path": str(transition_dataset_path),
            "row_count": len(rows),
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
            support_readiness=support_readiness,
        ),
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v178_sources(
    *,
    v177_report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v177_report_exact_digest: str | None,
    expected_dataset_digest: str | None,
) -> dict[str, object]:
    failures: list[str] = []
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
    observed_classification = str(
        _mapping(v177_report.get("classification")).get("primary") or ""
    )
    if observed_classification != EXPECTED_V177_CLASSIFICATION:
        failures.append("v177_unexpected_classification")
    exact = exact_digest_validation_report(v177_report)
    observed_exact = str(v177_report.get("exact_digest") or "")
    if exact.get("passed") is not True:
        failures.append("v177_exact_digest_mismatch")
    if (
        expected_v177_report_exact_digest
        and observed_exact != expected_v177_report_exact_digest
    ):
        failures.append("v177_unexpected_exact_digest")
    observed_dataset_digest = stable_payload_digest([dict(row) for row in rows])
    if expected_dataset_digest and observed_dataset_digest != expected_dataset_digest:
        failures.append("v177_dataset_digest_mismatch")
    dataset = _mapping(v177_report.get("dataset"))
    reported_digest = dataset.get("dataset_digest")
    reported_row_count = dataset.get("row_count")
    if reported_digest not in (None, observed_dataset_digest):
        failures.append("v177_reported_dataset_digest_mismatch")
    if (
        reported_row_count is not None
        and _int(reported_row_count, default=-1) != len(rows)
    ):
        failures.append("v177_reported_dataset_row_count_mismatch")
    lifecycle = _v177_lifecycle_validation(v177_report)
    if lifecycle.get("passed") is not True:
        failures.append("v177_lifecycle_not_diagnostics_only")
    return {
        "policy": "m3_carrion_survivor_continuation_v178_source_validation_v1",
        "passed": bool(rows) and not failures,
        "failures": sorted(set(failures)),
        "expected_v177_classification": EXPECTED_V177_CLASSIFICATION,
        "observed_v177_classification": observed_classification,
        "expected_v177_report_exact_digest": expected_v177_report_exact_digest,
        "observed_v177_report_exact_digest": observed_exact,
        "v177_exact_digest_validation": exact,
        "expected_dataset_digest": expected_dataset_digest,
        "observed_dataset_digest": observed_dataset_digest,
        "v177_reported_dataset_digest": reported_digest,
        "v177_reported_dataset_row_count": reported_row_count,
        "observed_dataset_row_count": len(rows),
        "v177_lifecycle_validation": lifecycle,
    }


def transition_row_identity_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    branch_action_counts: Counter[tuple[str, str]] = Counter()
    feature_digest_counts: Counter[str] = Counter()
    current_state_digest_counts: Counter[str] = Counter()
    branch_digest_counts: Counter[str] = Counter()
    failures: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        metadata = _mapping(row.get("metadata"))
        branch_id = str(metadata.get("branch_id") or "")
        action = str(row.get("forced_action") or "")
        branch_action_counts.update([(branch_id, action)])
        feature_digest_counts.update(
            [stable_payload_digest(_mapping(row.get("trainable_public_features")))]
        )
        current_state_digest_counts.update(
            [
                stable_payload_digest(
                    {
                        "current_public_observation": row.get(
                            "current_public_observation"
                        ),
                        "current_public_action_mask": row.get(
                            "current_public_action_mask"
                        ),
                        "previous_same_agent_public_context": row.get(
                            "previous_same_agent_public_context"
                        ),
                    }
                )
            ]
        )
        branch_state_digest = str(metadata.get("branch_state_digest") or "")
        if branch_state_digest:
            branch_digest_counts.update([branch_state_digest])
        if not branch_id:
            failures.append({"row_index": row_index, "reason": "missing_branch_id"})
        if action not in ACTION_NAMES:
            failures.append(
                {"row_index": row_index, "reason": "invalid_forced_action"}
            )
    duplicate_branch_actions = [
        {"branch_id": branch_id, "forced_action": action, "count": int(count)}
        for (branch_id, action), count in sorted(branch_action_counts.items())
        if count > 1
    ]
    for duplicate in duplicate_branch_actions[:32]:
        failures.append(
            {
                "reason": "duplicate_branch_forced_action",
                "branch_id": duplicate["branch_id"],
                "forced_action": duplicate["forced_action"],
                "count": duplicate["count"],
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v178_identity_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:96],
        "row_count": len(rows),
        "unique_branch_action_count": len(branch_action_counts),
        "duplicate_branch_action_count": sum(
            int(item["count"]) - 1 for item in duplicate_branch_actions
        ),
        "duplicate_branch_action_examples": duplicate_branch_actions[:16],
        "duplicate_trainable_feature_payload_count": _duplicate_member_count(
            feature_digest_counts
        ),
        "duplicate_current_state_payload_count": _duplicate_member_count(
            current_state_digest_counts
        ),
        "branch_state_digest_count": len(branch_digest_counts),
        "branch_state_digest_duplicate_count": _duplicate_member_count(
            branch_digest_counts
        ),
    }


def transition_row_action_mask_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    current_widths: Counter[int] = Counter()
    next_widths: Counter[int] = Counter()
    forced_still_supported = 0
    gained_counts: Counter[str] = Counter()
    lost_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        action = str(row.get("forced_action") or "")
        current_mask = _complete_action_mask(_mapping(row.get("current_public_action_mask")))
        next_mask_payload = row.get("next_public_action_mask")
        next_mask = (
            _complete_action_mask(_mapping(next_mask_payload))
            if isinstance(next_mask_payload, Mapping)
            else None
        )
        current_widths.update([_mask_width(current_mask)])
        if action not in ACTION_NAMES:
            failures.append(
                {"row_index": row_index, "reason": "invalid_forced_action"}
            )
        elif current_mask.get(action) is not True:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "forced_action_not_current_mask_supported",
                    "forced_action": action,
                }
            )
        if row.get("transition_done") is True:
            if next_mask is not None:
                failures.append(
                    {"row_index": row_index, "reason": "done_transition_has_next_mask"}
                )
            continue
        if next_mask is None:
            failures.append(
                {"row_index": row_index, "reason": "next_action_mask_missing"}
            )
            continue
        next_widths.update([_mask_width(next_mask)])
        if action in ACTION_NAMES and next_mask.get(action) is True:
            forced_still_supported += 1
        for candidate in ACTION_NAMES:
            if current_mask.get(candidate) is not True and next_mask.get(candidate) is True:
                gained_counts.update([candidate])
            if current_mask.get(candidate) is True and next_mask.get(candidate) is not True:
                lost_counts.update([candidate])
    return {
        "policy": "m3_carrion_survivor_continuation_v178_action_mask_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:96],
        "row_count": len(rows),
        "current_mask_width_counts": _counter_dict(current_widths),
        "next_mask_width_counts": _counter_dict(next_widths),
        "forced_action_supported_in_current_count": len(rows) - len(
            [
                failure
                for failure in failures
                if failure.get("reason") == "forced_action_not_current_mask_supported"
            ]
        ),
        "forced_action_supported_in_next_count": forced_still_supported,
        "next_mask_action_gained_counts": _action_counter_dict(gained_counts),
        "next_mask_action_lost_counts": _action_counter_dict(lost_counts),
    }


def transition_row_observation_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    current_next_deltas: list[float] = []
    current_next_changed_counts: list[int] = []
    previous_current_deltas: list[float] = []
    current_decoded = 0
    next_decoded = 0
    previous_decoded = 0
    current_next_exact = 0
    previous_current_exact = 0
    shape_counts: Counter[str] = Counter()
    encoder_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        current_payload = _mapping(row.get("current_public_observation"))
        current_values, current_error = _decode_values(current_payload)
        if current_error:
            failures.append(
                {
                    "row_index": row_index,
                    "field": "current_public_observation",
                    "reason": current_error,
                }
            )
        else:
            current_decoded += 1
            shape_counts.update([_shape_key(current_payload.get("shape"))])
            encoder_counts.update([str(current_payload.get("encoder_version") or "")])
        next_payload = row.get("next_public_observation")
        next_values: list[float] | None = None
        if next_payload is not None:
            next_values, next_error = _decode_values(_mapping(next_payload))
            if next_error:
                failures.append(
                    {
                        "row_index": row_index,
                        "field": "next_public_observation",
                        "reason": next_error,
                    }
                )
            else:
                next_decoded += 1
        previous_context = _mapping(row.get("previous_same_agent_public_context"))
        previous_payload = previous_context.get("public_observation")
        previous_values: list[float] | None = None
        if previous_context.get("available") is True:
            previous_values, previous_error = _decode_values(_mapping(previous_payload))
            if previous_error:
                failures.append(
                    {
                        "row_index": row_index,
                        "field": "previous_same_agent_public_context.public_observation",
                        "reason": previous_error,
                    }
                )
            else:
                previous_decoded += 1
        if current_values is not None and next_values is not None:
            delta = _vector_delta(current_values, next_values)
            current_next_deltas.append(delta["mean_absolute_delta"])
            current_next_changed_counts.append(delta["changed_element_count"])
            if delta["changed_element_count"] == 0:
                current_next_exact += 1
        if current_values is not None and previous_values is not None:
            delta = _vector_delta(previous_values, current_values)
            previous_current_deltas.append(delta["mean_absolute_delta"])
            if delta["changed_element_count"] == 0:
                previous_current_exact += 1
    return {
        "policy": "m3_carrion_survivor_continuation_v178_observation_audit_v1",
        "passed": not failures and current_decoded == len(rows),
        "failure_count": len(failures),
        "failures": failures[:96],
        "row_count": len(rows),
        "current_observation_decoded_count": current_decoded,
        "next_observation_decoded_count": next_decoded,
        "previous_observation_decoded_count": previous_decoded,
        "current_observation_shape_counts": _counter_dict(shape_counts),
        "current_observation_encoder_counts": _counter_dict(encoder_counts),
        "current_next_exact_match_count": current_next_exact,
        "current_next_mean_absolute_delta": _number_stats(current_next_deltas),
        "current_next_changed_element_count": _integer_stats(
            current_next_changed_counts
        ),
        "previous_current_exact_match_count": previous_current_exact,
        "previous_current_mean_absolute_delta": _number_stats(
            previous_current_deltas
        ),
    }


def transition_row_coverage_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    seeds: Counter[int] = Counter()
    branches_by_seed: dict[int, set[str]] = defaultdict(set)
    actions: Counter[str] = Counter()
    actions_by_seed: dict[int, Counter[str]] = defaultdict(Counter)
    branch_actions: dict[str, set[str]] = defaultdict(set)
    failure_types: Counter[str] = Counter()
    failed_safe_actions: Counter[str] = Counter()
    source_paths: Counter[str] = Counter()
    branch_ticks: Counter[int] = Counter()
    previous_count = 0
    next_count = 0
    done_count = 0
    for row in rows:
        metadata = _mapping(row.get("metadata"))
        seed = _int(metadata.get("seed"), default=-1)
        branch_id = str(metadata.get("branch_id") or "")
        action = str(row.get("forced_action") or "")
        if seed >= 0:
            seeds.update([seed])
        if seed >= 0 and branch_id:
            branches_by_seed[seed].add(branch_id)
        if action:
            actions.update([action])
            if seed >= 0:
                actions_by_seed[seed].update([action])
            if branch_id:
                branch_actions[branch_id].add(action)
        for failure_type in _strings(metadata.get("failure_types")):
            failure_types.update([failure_type])
        failed_safe = str(metadata.get("failed_safe_action") or "")
        if failed_safe:
            failed_safe_actions.update([failed_safe])
        source_path = str(metadata.get("source_path") or "")
        if source_path:
            source_paths.update([source_path])
        branch_tick = _int(metadata.get("branch_tick"), default=-1)
        if branch_tick >= 0:
            branch_ticks.update([branch_tick])
        if _mapping(row.get("previous_same_agent_public_context")).get("available") is True:
            previous_count += 1
        if row.get("next_public_observation_available") is True:
            next_count += 1
        if row.get("transition_done") is True:
            done_count += 1
    action_widths = Counter(len(action_set) for action_set in branch_actions.values())
    return {
        "policy": "m3_carrion_survivor_continuation_v178_coverage_audit_v1",
        "row_count": len(rows),
        "seed_count": len(seeds),
        "branch_count": len(branch_actions),
        "forced_action_count": len(actions),
        "row_counts_by_seed": {
            str(seed): int(count) for seed, count in sorted(seeds.items())
        },
        "branch_counts_by_seed": {
            str(seed): len(branches)
            for seed, branches in sorted(branches_by_seed.items())
        },
        "forced_action_counts": _action_counter_dict(actions),
        "forced_actions_by_seed": {
            str(seed): _action_counter_dict(counter)
            for seed, counter in sorted(actions_by_seed.items())
        },
        "branch_forced_action_width_counts": _counter_dict(action_widths),
        "failure_type_counts": _counter_dict(failure_types),
        "failed_safe_action_counts": _action_counter_dict(failed_safe_actions),
        "source_path_count": len(source_paths),
        "source_path_row_counts": _counter_dict(source_paths),
        "branch_tick_min": min(branch_ticks) if branch_ticks else None,
        "branch_tick_max": max(branch_ticks) if branch_ticks else None,
        "rows_with_previous_same_agent_public_context": previous_count,
        "rows_with_next_public_observation": next_count,
        "transition_done_count": done_count,
        "consumed_support_seed_note": (
            "These seeds are source/provenance support for this lane, not clean "
            "promotion-heldout evidence."
        ),
    }


def transition_row_target_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    requested_actions: Counter[str] = Counter()
    resolved_actions: Counter[str] = Counter()
    action_valid_counts: Counter[str] = Counter()
    resolution_valid_counts: Counter[str] = Counter()
    moved_counts: Counter[str] = Counter()
    target_terminal_count = 0
    forced_used_count = 0
    reward_by_action: dict[str, list[float]] = defaultdict(list)
    resource_by_action: dict[str, list[float]] = defaultdict(list)
    alive_values: list[int] = []
    birth_values: list[int] = []
    death_values: list[int] = []
    for row in rows:
        action = str(row.get("forced_action") or "")
        summary = _mapping(row.get("short_horizon_public_outcome_summary"))
        requested_actions.update([str(summary.get("current_requested_action") or "")])
        resolved_actions.update([str(summary.get("current_resolved_action") or "")])
        action_valid_counts.update([str(summary.get("current_action_valid"))])
        resolution_valid_counts.update(
            [str(summary.get("current_resolution_action_valid"))]
        )
        moved_counts.update([str(summary.get("current_moved"))])
        if summary.get("target_terminal") is True:
            target_terminal_count += 1
        if summary.get("forced_action_used") is True:
            forced_used_count += 1
        reward = _float(summary.get("current_reward_total"))
        if reward is not None:
            reward_by_action[action].append(reward)
        resource_gain = _float(summary.get("current_resource_gain"))
        if resource_gain is not None:
            resource_by_action[action].append(resource_gain)
        alive_values.append(_int(summary.get("alive_agents"), default=0))
        birth_values.append(_int(summary.get("births"), default=0))
        death_values.append(_int(summary.get("deaths"), default=0))
    return {
        "policy": "m3_carrion_survivor_continuation_v178_target_audit_v1",
        "row_count": len(rows),
        "forced_action_used_count": forced_used_count,
        "all_forced_actions_used": len(rows) > 0 and forced_used_count == len(rows),
        "current_requested_action_counts": _action_counter_dict(requested_actions),
        "current_resolved_action_counts": _action_counter_dict(resolved_actions),
        "current_action_valid_counts": _counter_dict(action_valid_counts),
        "current_resolution_action_valid_counts": _counter_dict(
            resolution_valid_counts
        ),
        "current_moved_counts": _counter_dict(moved_counts),
        "target_terminal_count": target_terminal_count,
        "reward_stats_by_forced_action": {
            action: _number_stats(values)
            for action, values in sorted(
                reward_by_action.items(), key=lambda item: _action_order(item[0])
            )
        },
        "resource_gain_stats_by_forced_action": {
            action: _number_stats(values)
            for action, values in sorted(
                resource_by_action.items(), key=lambda item: _action_order(item[0])
            )
        },
        "alive_agents_stats": _integer_stats(alive_values),
        "births_stats": _integer_stats(birth_values),
        "deaths_stats": _integer_stats(death_values),
    }


def transition_row_support_readiness(
    *,
    rows: Sequence[Mapping[str, object]],
    coverage_audit: Mapping[str, object],
    identity_audit: Mapping[str, object],
    min_row_count: int,
    min_seed_count: int,
    min_branch_count: int,
    min_forced_action_count: int,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    checks = {
        "row_count": (len(rows), int(min_row_count)),
        "seed_count": (_int(coverage_audit.get("seed_count")), int(min_seed_count)),
        "branch_count": (
            _int(coverage_audit.get("branch_count")),
            int(min_branch_count),
        ),
        "forced_action_count": (
            _int(coverage_audit.get("forced_action_count")),
            int(min_forced_action_count),
        ),
    }
    for name, (observed, minimum) in checks.items():
        if observed < minimum:
            failures.append(
                {
                    "reason": f"{name}_below_minimum",
                    "observed": observed,
                    "minimum": minimum,
                }
            )
    if identity_audit.get("duplicate_branch_action_count") not in (None, 0):
        failures.append(
            {
                "reason": "duplicate_branch_action_rows",
                "observed": identity_audit.get("duplicate_branch_action_count"),
                "minimum": 0,
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v178_support_readiness_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "minimums": {
            "row_count": int(min_row_count),
            "seed_count": int(min_seed_count),
            "branch_count": int(min_branch_count),
            "forced_action_count": int(min_forced_action_count),
        },
        "observed": {
            "row_count": len(rows),
            "seed_count": _int(coverage_audit.get("seed_count")),
            "branch_count": _int(coverage_audit.get("branch_count")),
            "forced_action_count": _int(coverage_audit.get("forced_action_count")),
        },
        "training_scale_capacity_work_unblocked": False,
        "training_authorized": False,
        "promotion_authorized": False,
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    identity_audit: Mapping[str, object],
    action_mask_audit: Mapping[str, object],
    observation_audit: Mapping[str, object],
    support_readiness: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if (
        row_schema_validation.get("passed") is not True
        or leakage_scan.get("passed") is not True
        or identity_audit.get("passed") is not True
        or action_mask_audit.get("passed") is not True
        or observation_audit.get("passed") is not True
    ):
        return prefix + "dataset_contract_invalid_closed_no_training"
    if support_readiness.get("passed") is not True:
        return prefix + "valid_support_limited_expand_before_training"
    return prefix + "valid_route_decision_ready_no_training"


def _route_recommendation(
    *,
    classification: str,
    support_readiness: Mapping[str, object],
) -> dict[str, object]:
    contract_valid = classification.endswith(
        "valid_route_decision_ready_no_training"
    ) or classification.endswith("valid_support_limited_expand_before_training")
    support_ready = support_readiness.get("passed") is True
    if not contract_valid:
        route = "repair_v177_transition_rows_before_capacity_work"
    elif not support_ready:
        route = "v179_expand_exact_branch_transition_rows_no_training"
    else:
        route = "v179_transition_row_model_design_audit_no_training"
    return {
        "policy": "m3_carrion_survivor_continuation_v178_route_recommendation_v1",
        "recommended_next_route": route,
        "dataset_contract_valid": contract_valid,
        "support_minimums_met": support_ready,
        "transition_row_training_authorized": False,
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
        "input_rows_are_v177_public_transition_rows": True,
        "short_horizon_outcomes_remain_diagnostic_targets_only": True,
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


def _v177_lifecycle_validation(report: Mapping[str, object]) -> dict[str, object]:
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
        "policy": "m3_carrion_survivor_continuation_v178_v177_lifecycle_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def _decode_values(payload: Mapping[str, object]) -> tuple[list[float] | None, str | None]:
    try:
        return decode_observation_input(dict(payload)), None
    except (TypeError, ValueError) as exc:
        return None, str(exc)


def _vector_delta(left: Sequence[float], right: Sequence[float]) -> dict[str, object]:
    if len(left) != len(right):
        return {
            "mean_absolute_delta": 0.0,
            "changed_element_count": 0,
            "length_mismatch": True,
        }
    differences = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    return {
        "mean_absolute_delta": _round(mean(differences)) if differences else 0.0,
        "changed_element_count": sum(1 for value in differences if value > 1e-9),
        "length_mismatch": False,
    }


def _complete_action_mask(value: Mapping[str, object]) -> dict[str, bool]:
    return {action: value.get(action) is True for action in ACTION_NAMES}


def _mask_width(mask: Mapping[str, bool]) -> int:
    return sum(1 for action in ACTION_NAMES if mask.get(action) is True)


def _duplicate_member_count(counter: Counter[object]) -> int:
    return sum(int(count) - 1 for count in counter.values() if count > 1)


def _strings(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item)]


def _float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _shape_key(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "x".join(str(item) for item in value)
    return ""


def _counter_dict(counter: Counter[object]) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(counter.items())}


def _action_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        str(action): int(count)
        for action, count in sorted(counter.items(), key=lambda item: _action_order(item[0]))
        if str(action)
    }


def _number_stats(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "mean": _round(mean(values)),
    }


def _integer_stats(values: Sequence[int]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": _round(mean(values)),
    }


def _json_round_trip_digest(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    )
