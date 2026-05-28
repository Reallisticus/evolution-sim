from __future__ import annotations

import gzip
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    LOCAL_PATCH_RADIUS,
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
)
from evolution_sim.mind.branch_action_oracle_audit import (
    DEFAULT_BRANCH_ACTION_ORACLE_HISTORY_STEPS,
    MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION = (
    "mind_v3_branch_action_oracle_labels_v1"
)
MIND_V3_BRANCH_CONTINUATION_ARCHIVE_SCORER_SCHEMA_VERSION = (
    "mind_v3_branch_continuation_archive_scorer_v1"
)
MIND_V3_BRANCH_ACTION_ORACLE_LABEL_POLICY = (
    "replay_verified_policy_visible_first_action_oracle_labels_v1"
)
MIND_V3_BRANCH_ACTION_ORACLE_OBJECTIVE = (
    "lexicographic_terminal_alive_birth_target_alive_deaths_diversity_v1"
)
DEFAULT_BRANCH_ACTION_ORACLE_LABEL_MAX_DOMINANT_ACTION_SHARE = 0.5
COMPACT_OUTCOME_WORLD_MODEL_ACCURACY_FLOOR = 0.5
COMPACT_OPTION_MODE_ACCURACY_FLOOR = 0.6
COMPACT_OPTION_MODE_MAX_DOMINANT_PREDICTION_SHARE = 0.75
COMPACT_REPOSITION_DIRECTION_ACCURACY_FLOOR = 0.7
COMPACT_REPOSITION_MULTI_MOVE_ACCURACY_FLOOR = 0.55
SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR = 0.5
BRANCH_CONTINUATION_ARCHIVE_SCORER_ACCURACY_FLOOR = 0.55
SHORT_HORIZON_TRACE_TICKS: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    5,
    8,
    13,
    21,
    34,
    55,
    89,
)
BRANCH_CONTINUATION_ARCHIVE_TRACE_TICKS: tuple[int, ...] = (21, 55, 89)
_SELF_FIELD_INDEX = {field: index for index, field in enumerate(SELF_INPUT_FIELDS)}
_PATCH_FIELD_INDEX = {field: index for index, field in enumerate(PATCH_INPUT_FIELDS)}
_NAVIGATION_FIELD_INDEX = {
    field: index for index, field in enumerate(NAVIGATION_INPUT_FIELDS)
}
_PATCH_INPUT_START = len(SELF_INPUT_FIELDS)
_PATCH_STRIDE = len(PATCH_INPUT_FIELDS)
_NAVIGATION_INPUT_START = _PATCH_INPUT_START + PATCH_CELL_COUNT * _PATCH_STRIDE
_NAVIGATION_STRIDE = len(NAVIGATION_INPUT_FIELDS)
_CENTER_PATCH_INDEX = PATCH_CELL_COUNT // 2
_MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_north": (0, -1),
    "move_south": (0, 1),
    "move_east": (1, 0),
    "move_west": (-1, 0),
}
_COMPACT_WORLD_MODEL_FEATURE_CONTRACT: tuple[str, ...] = (
    "self_vitals_and_deficits",
    "self_role_mode_season_hazard_reproduction",
    "center_cell_food_water_carrion_prey_risk",
    "radius1_and_radius2_local_resource_risk_summaries",
    "navigation_water_plant_carrion_prey_direction_distance_strength",
    "action_family_and_direction",
    "action_mask_counts",
    "action_specific_immediate_drink_eat_stay_affordance",
    "action_specific_move_target_cell_and_navigation_alignment",
)
_BRANCH_CONTINUATION_ARCHIVE_FEATURE_CONTRACT: tuple[str, ...] = (
    "compact_policy_visible_branch_state",
    "candidate_action_family_direction_and_mask",
    "same_agent_public_history_trace",
    "coarse_archive_cell_quantization",
)
_BRANCH_CONTINUATION_ARCHIVE_TARGET_CONTRACT: tuple[str, ...] = (
    "terminal_alive_agents",
    "terminal_births",
    "multi_horizon_target_survival_area",
    "multi_horizon_population_alive_delta_area",
    "multi_horizon_birth_delta_area",
    "multi_horizon_target_vital_area",
    "multi_horizon_resource_gain_area",
    "multi_horizon_death_delta_penalty",
    "multi_horizon_action_collapse_penalty",
)


class BranchActionOracleLabelError(ValueError):
    pass


def build_branch_action_oracle_label_report(
    branch_action_oracle_audit: Mapping[str, object],
    *,
    require_accepted_audit: bool = True,
    min_material_label_count: int = 1,
    max_dominant_oracle_action_share: float = (
        DEFAULT_BRANCH_ACTION_ORACLE_LABEL_MAX_DOMINANT_ACTION_SHARE
    ),
) -> dict[str, object]:
    if (
        branch_action_oracle_audit.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION
    ):
        raise BranchActionOracleLabelError(
            "branch action oracle audit has unsupported schema_version"
        )
    if min_material_label_count < 0:
        raise BranchActionOracleLabelError(
            "min_material_label_count must be non-negative"
        )
    if (
        max_dominant_oracle_action_share < 0.0
        or max_dominant_oracle_action_share > 1.0
    ):
        raise BranchActionOracleLabelError(
            "max_dominant_oracle_action_share must be in [0.0, 1.0]"
        )

    audit_acceptance = _mapping(branch_action_oracle_audit.get("acceptance"))
    if (
        require_accepted_audit
        and audit_acceptance.get("diagnostic_acceptance_passed") is not True
    ):
        raise BranchActionOracleLabelError(
            "branch action oracle audit must pass diagnostic acceptance"
        )

    branch_points = _list_of_mappings(
        branch_action_oracle_audit.get("branch_points"),
        field="branch_points",
    )
    branch_results = _list_of_mappings(
        branch_action_oracle_audit.get("branch_results"),
        field="branch_results",
    )
    branch_point_by_id = {
        str(point.get("branch_id")): point
        for point in branch_points
        if point.get("branch_id") is not None
    }
    labels = [
        _label_from_branch_result(result, branch_point_by_id=branch_point_by_id)
        for result in branch_results
    ]
    aggregate = _aggregate_labels(labels)
    classifier_probe = _oracle_action_classifier_support_probe(labels)
    action_value_probe = _action_value_ranker_support_probe(labels)
    compact_world_model_probe = _compact_outcome_world_model_support_probe(labels)
    compact_first_step_probe = _compact_first_step_world_model_support_probe(labels)
    first_step_augmented_probe = (
        _first_step_augmented_terminal_world_model_support_probe(labels)
    )
    compact_option_mode_probe = _compact_option_mode_world_model_support_probe(
        labels
    )
    compact_reposition_direction_probe = (
        _compact_reposition_direction_world_model_support_probe(labels)
    )
    short_horizon_trace_probe = (
        _short_horizon_trace_terminal_world_model_support_probe(labels)
    )
    actual_horizon_alignment_probe = _actual_horizon_trace_alignment_probe(labels)
    actual_population_horizon_probe = _actual_population_horizon_alignment_probe(labels)
    material_actual_population_horizon_probe = (
        _actual_population_horizon_alignment_probe(labels, material_only=True)
    )
    compact_population_horizon_probe = (
        _compact_population_horizon_world_model_support_probe(labels)
    )
    material_compact_population_horizon_probe = (
        _compact_population_horizon_world_model_support_probe(
            labels,
            material_only=True,
        )
    )
    policy_observation_population_horizon_probe = (
        _policy_observation_population_horizon_world_model_support_probe(labels)
    )
    material_policy_observation_population_horizon_probe = (
        _policy_observation_population_horizon_world_model_support_probe(
            labels,
            material_only=True,
        )
    )
    policy_observation_history_population_horizon_probe = (
        _policy_observation_history_population_horizon_world_model_support_probe(
            labels
        )
    )
    material_policy_observation_history_population_horizon_probe = (
        _policy_observation_history_population_horizon_world_model_support_probe(
            labels,
            material_only=True,
        )
    )
    branch_continuation_archive_probe = (
        _branch_continuation_archive_scorer_support_probe(labels)
    )
    material_branch_continuation_archive_probe = (
        _branch_continuation_archive_scorer_support_probe(
            labels,
            material_only=True,
        )
    )
    acceptance = _acceptance(
        aggregate,
        min_material_label_count=min_material_label_count,
        max_dominant_oracle_action_share=max_dominant_oracle_action_share,
    )
    contract = {
        "schema_version": MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
        "label_policy": MIND_V3_BRANCH_ACTION_ORACLE_LABEL_POLICY,
        "oracle_objective": MIND_V3_BRANCH_ACTION_ORACLE_OBJECTIVE,
        "source_audit_schema_version": branch_action_oracle_audit.get(
            "schema_version"
        ),
        "source_contract_digest": _mapping(
            branch_action_oracle_audit.get("provenance")
        ).get("contract_digest"),
        "require_accepted_audit": bool(require_accepted_audit),
        "min_material_label_count": int(min_material_label_count),
        "max_dominant_oracle_action_share": _round(
            max_dominant_oracle_action_share
        ),
        "runtime_input_policy": (
            "labels serialize observation_input, observation_digest, action_mask, "
            "and optional same-agent public_history_trace as runtime policy "
            "support; fixture, seed, tick, and agent_id are provenance only"
        ),
    }
    return {
        "schema_version": MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
        "label_policy": MIND_V3_BRANCH_ACTION_ORACLE_LABEL_POLICY,
        "label_contract": contract,
        "provenance": {
            "source_audit_schema_version": branch_action_oracle_audit.get(
                "schema_version"
            ),
            "source_audit_digest": stable_payload_digest(
                {
                    "schema_version": branch_action_oracle_audit.get(
                        "schema_version"
                    ),
                    "contract": branch_action_oracle_audit.get("contract"),
                    "aggregate": branch_action_oracle_audit.get("aggregate"),
                    "acceptance": branch_action_oracle_audit.get("acceptance"),
                    "branch_point_count": len(branch_points),
                    "branch_result_count": len(branch_results),
                }
            ),
            "label_contract_digest": stable_payload_digest(contract),
        },
        "aggregate": aggregate,
        "support_probe": classifier_probe,
        "support_probes": {
            "oracle_action_classifier": classifier_probe,
            "action_conditioned_value_ranker": action_value_probe,
            "compact_outcome_world_model": compact_world_model_probe,
            "compact_first_step_world_model": compact_first_step_probe,
            "first_step_augmented_terminal_world_model": first_step_augmented_probe,
            "compact_option_mode_world_model": compact_option_mode_probe,
            "compact_reposition_direction_world_model": (
                compact_reposition_direction_probe
            ),
            "short_horizon_trace_terminal_world_model": short_horizon_trace_probe,
            "actual_horizon_trace_alignment": actual_horizon_alignment_probe,
            "actual_population_horizon_alignment": actual_population_horizon_probe,
            "material_only_actual_population_horizon_alignment": (
                material_actual_population_horizon_probe
            ),
            "compact_population_horizon_world_model": (
                compact_population_horizon_probe
            ),
            "material_only_compact_population_horizon_world_model": (
                material_compact_population_horizon_probe
            ),
            "policy_observation_population_horizon_world_model": (
                policy_observation_population_horizon_probe
            ),
            "material_only_policy_observation_population_horizon_world_model": (
                material_policy_observation_population_horizon_probe
            ),
            "policy_observation_history_population_horizon_world_model": (
                policy_observation_history_population_horizon_probe
            ),
            "material_only_policy_observation_history_population_horizon_world_model": (
                material_policy_observation_history_population_horizon_probe
            ),
            "branch_continuation_archive_scorer": (
                branch_continuation_archive_probe
            ),
            "material_only_branch_continuation_archive_scorer": (
                material_branch_continuation_archive_probe
            ),
        },
        "acceptance": acceptance,
        "labels": labels,
    }


def load_branch_action_oracle_audit_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    try:
        with _open_input(resolved) as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise BranchActionOracleLabelError(
            f"failed to read branch action oracle audit: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BranchActionOracleLabelError(
            f"branch action oracle audit is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BranchActionOracleLabelError(
            "branch action oracle audit must be a JSON object"
        )
    return payload


def load_branch_action_oracle_label_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    try:
        with _open_input(resolved) as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise BranchActionOracleLabelError(
            f"failed to read branch action oracle labels: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BranchActionOracleLabelError(
            f"branch action oracle labels are not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BranchActionOracleLabelError(
            "branch action oracle labels must be a JSON object"
        )
    return payload


def write_branch_action_oracle_label_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def build_branch_continuation_archive_scorer_report(
    branch_action_oracle_labels: Mapping[str, object],
) -> dict[str, object]:
    if (
        branch_action_oracle_labels.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION
    ):
        raise BranchActionOracleLabelError(
            "branch action oracle labels have unsupported schema_version"
        )
    labels = _list_of_mappings(
        branch_action_oracle_labels.get("labels"),
        field="labels",
    )
    all_label_probe = _branch_continuation_archive_scorer_support_probe(labels)
    material_only_probe = _branch_continuation_archive_scorer_support_probe(
        labels,
        material_only=True,
    )
    passed = (
        all_label_probe.get(
            "materially_supports_branch_continuation_archive_scorer"
        )
        is True
    )
    blockers = []
    if not passed:
        blockers.append(
            {
                "reason": "branch_continuation_archive_support_below_floor",
                "observed": all_label_probe.get("best_accuracy"),
                "required_min": all_label_probe.get(
                    "material_support_accuracy_floor"
                ),
            }
        )
    contract = {
        "schema_version": MIND_V3_BRANCH_CONTINUATION_ARCHIVE_SCORER_SCHEMA_VERSION,
        "source_label_schema_version": branch_action_oracle_labels.get(
            "schema_version"
        ),
        "gate": (
            "leave-one-source-seed-out exact-action ranking must reach "
            "0.55 accuracy on all labels before any runtime policy training"
        ),
        "runtime_policy_trained": False,
    }
    return {
        "schema_version": MIND_V3_BRANCH_CONTINUATION_ARCHIVE_SCORER_SCHEMA_VERSION,
        "contract": contract,
        "provenance": {
            "source_label_digest": stable_payload_digest(
                {
                    "schema_version": branch_action_oracle_labels.get(
                        "schema_version"
                    ),
                    "aggregate": branch_action_oracle_labels.get("aggregate"),
                    "acceptance": branch_action_oracle_labels.get("acceptance"),
                    "label_count": len(labels),
                }
            ),
            "contract_digest": stable_payload_digest(contract),
        },
        "source_label_aggregate": dict(
            _mapping(branch_action_oracle_labels.get("aggregate"))
        ),
        "support_probes": {
            "branch_continuation_archive_scorer": all_label_probe,
            "material_only_branch_continuation_archive_scorer": (
                material_only_probe
            ),
        },
        "acceptance": {
            "branch_continuation_archive_scorer_gate_passed": passed,
            "runtime_training_allowed": passed,
            "blockers": blockers,
        },
    }


def write_branch_continuation_archive_scorer_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _label_from_branch_result(
    result: Mapping[str, object],
    *,
    branch_point_by_id: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    branch_id = str(result.get("branch_id", ""))
    if not branch_id:
        raise BranchActionOracleLabelError("branch result missing branch_id")
    branch_point = branch_point_by_id.get(branch_id)
    if branch_point is None:
        raise BranchActionOracleLabelError(
            f"branch result has no matching branch point: {branch_id}"
        )
    oracle_action = str(result.get("oracle_best_action", ""))
    if oracle_action not in ACTION_NAMES:
        raise BranchActionOracleLabelError(
            f"branch result has unsupported oracle action: {oracle_action}"
        )
    logged_action = str(result.get("logged_action", ""))
    if logged_action not in ACTION_NAMES:
        raise BranchActionOracleLabelError(
            f"branch result has unsupported logged action: {logged_action}"
        )
    policy_state = _policy_state(branch_point)
    action_mask = _action_mask(policy_state)
    action_values = _action_values(result)
    oracle_supported = bool(action_mask.get(oracle_action, False))
    logged_supported = bool(action_mask.get(logged_action, False))
    return {
        "schema_version": MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
        "branch_id": branch_id,
        "source": {
            "fixture": _optional_string(result.get("fixture")),
            "seed": _optional_int(result.get("seed")),
            "branch_tick": _optional_int(result.get("branch_tick")),
            "branch_index": _optional_int(result.get("branch_index")),
            "record_index": _optional_int(result.get("record_index")),
            "agent_id": _optional_int(result.get("agent_id")),
            "base_script": _optional_string(result.get("base_script")),
            "continuation_script": _optional_string(
                result.get("continuation_script")
            ),
            "branch_state_digest": _optional_string(
                result.get("branch_state_digest")
            ),
        },
        "policy_state": policy_state,
        "context": dict(_mapping(result.get("context"))),
        "before": dict(_mapping(result.get("before"))),
        "oracle_label": {
            "action": oracle_action,
            "logged_action": logged_action,
            "oracle_changed_action": bool(result.get("oracle_changed_action")),
            "material_oracle_gain": bool(result.get("material_oracle_gain")),
            "oracle_alive_delta_vs_logged": _int(
                result.get("oracle_alive_delta_vs_logged")
            ),
            "oracle_birth_delta_vs_logged": _int(
                result.get("oracle_birth_delta_vs_logged")
            ),
            "oracle_target_alive_delta_vs_logged": _int(
                result.get("oracle_target_alive_delta_vs_logged")
            ),
            "oracle_action_supported_by_mask": oracle_supported,
            "logged_action_supported_by_mask": logged_supported,
        },
        "action_value_targets": {
            "objective": MIND_V3_BRANCH_ACTION_ORACLE_OBJECTIVE,
            "actions": action_values,
        },
        "quality": {
            "policy_state_complete": _policy_state_complete(policy_state),
            "replay_verified": _all_action_runs_replay_verified(result),
            "zero_heuristic_runtime_actions": _all_action_runs_zero_heuristic(result),
            "forced_action_used_for_all_candidates": (
                _all_action_runs_forced_action_used(result)
            ),
        },
    }


def _policy_state(branch_point: Mapping[str, object]) -> dict[str, object]:
    state = _mapping(branch_point.get("policy_state"))
    observation_input = _mapping(state.get("observation_input"))
    return {
        "observation_input": dict(observation_input),
        "observation_digest": _optional_string(state.get("observation_digest")),
        "observation_schema": _optional_string(state.get("observation_schema")),
        "action_mask": _complete_action_mask(_mapping(state.get("action_mask"))),
        "public_history_trace": _public_history_trace(state),
        "valid_actions": [
            action
            for action in ACTION_NAMES
            if bool(_mapping(state.get("action_mask")).get(action, False))
        ],
    }


def _public_history_trace(state: Mapping[str, object]) -> list[dict[str, object]]:
    trace = state.get("public_history_trace")
    if not isinstance(trace, list):
        return []
    return [
        _public_history_item(item)
        for item in trace
        if isinstance(item, Mapping)
    ]


def _public_history_item(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "tick": _int(item.get("tick")),
        "tick_delta": _int(item.get("tick_delta")),
        "record_index": _int(item.get("record_index")),
        "record_index_delta": _int(item.get("record_index_delta")),
        "requested_action": _optional_string(item.get("requested_action")),
        "resolved_action": _optional_string(item.get("resolved_action")),
        "action_valid": bool(item.get("action_valid", False)),
        "resolution_action_valid": bool(
            item.get("resolution_action_valid", False)
        ),
        "moved": bool(item.get("moved", False)),
        "x_delta": _int(item.get("x_delta")),
        "y_delta": _int(item.get("y_delta")),
        "energy_ratio_before": _optional_float(item.get("energy_ratio_before")),
        "energy_ratio_after": _optional_float(item.get("energy_ratio_after")),
        "energy_ratio_delta": _optional_float(item.get("energy_ratio_delta")),
        "hydration_ratio_before": _optional_float(
            item.get("hydration_ratio_before")
        ),
        "hydration_ratio_after": _optional_float(item.get("hydration_ratio_after")),
        "hydration_ratio_delta": _optional_float(item.get("hydration_ratio_delta")),
        "health_ratio_before": _optional_float(item.get("health_ratio_before")),
        "health_ratio_after": _optional_float(item.get("health_ratio_after")),
        "health_ratio_delta": _optional_float(item.get("health_ratio_delta")),
        "resource_gain": _optional_float(item.get("resource_gain")),
        "drank": bool(item.get("drank", False)),
        "ate": bool(item.get("ate", False)),
        "died": bool(item.get("died", False)),
        "death_cause": _optional_string(item.get("death_cause")),
        "died_after_action": bool(item.get("died_after_action", False)),
        "post_carrion_contact": bool(item.get("post_carrion_contact", False)),
        "ticks_since_animal_resource_gain": _optional_float(
            item.get("ticks_since_animal_resource_gain")
        ),
        "ticks_since_drink": _optional_float(item.get("ticks_since_drink")),
    }


def _action_mask(policy_state: Mapping[str, object]) -> dict[str, bool]:
    return {
        action: bool(_mapping(policy_state.get("action_mask")).get(action, False))
        for action in ACTION_NAMES
    }


def _complete_action_mask(raw: Mapping[str, object]) -> dict[str, bool]:
    return {action: bool(raw.get(action, False)) for action in ACTION_NAMES}


def _action_values(result: Mapping[str, object]) -> list[dict[str, object]]:
    runs = _list_of_mappings(result.get("action_runs"), field="action_runs")
    ranked = sorted(runs, key=_run_objective, reverse=True)
    rank_by_action = {
        str(run.get("forced_action")): index + 1 for index, run in enumerate(ranked)
    }
    return [
        {
            "action": str(run.get("forced_action")),
            "rank": int(rank_by_action.get(str(run.get("forced_action")), 0)),
            "terminal_alive_agents": _int(run.get("alive_agents")),
            "births": _int(run.get("births")),
            "deaths": _int(run.get("deaths")),
            "target_alive_at_end": bool(run.get("target_alive_at_end")),
            "dominant_requested_action_share": _optional_float(
                run.get("dominant_requested_action_share")
            ),
            "forced_action_used": bool(run.get("forced_action_used")),
            "replay_verified": _run_replay_verified(run),
            "heuristic_action_source_count": _int(
                run.get("heuristic_action_source_count")
            ),
            "first_action_outcome": _first_action_outcome(run),
            "target_horizon_trace": _target_horizon_trace(run),
            "population_horizon_trace": _population_horizon_trace(run),
            "objective_tuple": list(_run_objective(run)),
        }
        for run in sorted(runs, key=lambda item: str(item.get("forced_action", "")))
    ]


def _aggregate_labels(labels: Sequence[Mapping[str, object]]) -> dict[str, object]:
    oracle_counts: Counter[str] = Counter()
    logged_counts: Counter[str] = Counter()
    material_count = 0
    changed_count = 0
    alive_gain_total = 0
    birth_gain_total = 0
    target_alive_gain_total = 0
    replay_verified_count = 0
    zero_heuristic_count = 0
    complete_policy_state_count = 0
    unsupported_oracle_count = 0
    unsupported_logged_count = 0
    observation_digest_actions: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        oracle = _mapping(label.get("oracle_label"))
        action = str(oracle.get("action"))
        logged = str(oracle.get("logged_action"))
        oracle_counts[action] += 1
        logged_counts[logged] += 1
        if oracle.get("material_oracle_gain") is True:
            material_count += 1
        if oracle.get("oracle_changed_action") is True:
            changed_count += 1
        alive_gain_total += max(0, _int(oracle.get("oracle_alive_delta_vs_logged")))
        birth_gain_total += max(0, _int(oracle.get("oracle_birth_delta_vs_logged")))
        target_alive_gain_total += max(
            0,
            _int(oracle.get("oracle_target_alive_delta_vs_logged")),
        )
        if oracle.get("oracle_action_supported_by_mask") is not True:
            unsupported_oracle_count += 1
        if oracle.get("logged_action_supported_by_mask") is not True:
            unsupported_logged_count += 1
        quality = _mapping(label.get("quality"))
        if quality.get("replay_verified") is True:
            replay_verified_count += 1
        if quality.get("zero_heuristic_runtime_actions") is True:
            zero_heuristic_count += 1
        if quality.get("policy_state_complete") is True:
            complete_policy_state_count += 1
        policy_state = _mapping(label.get("policy_state"))
        digest = _optional_string(policy_state.get("observation_digest"))
        if digest is not None:
            observation_digest_actions[digest].add(action)
    label_count = len(labels)
    dominant_action, dominant_count = _dominant_count(oracle_counts)
    conflicting_digest_count = sum(
        1 for actions in observation_digest_actions.values() if len(actions) > 1
    )
    return {
        "label_count": label_count,
        "oracle_action_counts": dict(sorted(oracle_counts.items())),
        "logged_action_counts": dict(sorted(logged_counts.items())),
        "dominant_oracle_action": dominant_action,
        "dominant_oracle_action_count": dominant_count,
        "dominant_oracle_action_share": _safe_rate(dominant_count, label_count),
        "oracle_changed_label_count": changed_count,
        "material_oracle_gain_label_count": material_count,
        "terminal_alive_gain_total_vs_logged": int(alive_gain_total),
        "birth_gain_total_vs_logged": int(birth_gain_total),
        "target_alive_gain_total_vs_logged": int(target_alive_gain_total),
        "replay_verified_label_count": replay_verified_count,
        "replay_verified_all_labels": replay_verified_count == label_count,
        "zero_heuristic_label_count": zero_heuristic_count,
        "zero_heuristic_all_labels": zero_heuristic_count == label_count,
        "policy_state_complete_label_count": complete_policy_state_count,
        "policy_state_complete_all_labels": complete_policy_state_count == label_count,
        "unsupported_oracle_action_count": unsupported_oracle_count,
        "unsupported_logged_action_count": unsupported_logged_count,
        "observation_digest_count": len(observation_digest_actions),
        "conflicting_observation_digest_count": conflicting_digest_count,
    }


def _oracle_action_classifier_support_probe(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = []
    for label in labels:
        policy_state = _mapping(label.get("policy_state"))
        source = _mapping(label.get("source"))
        oracle = _mapping(label.get("oracle_label"))
        observation_input = _mapping(policy_state.get("observation_input"))
        try:
            values = decode_observation_input(dict(observation_input))
        except (ValueError, TypeError, zlib.error):
            values = []
        rows.append(
            {
                "branch_id": str(label.get("branch_id", "")),
                "seed": _optional_int(source.get("seed")),
                "action": str(oracle.get("action", "")),
                "logged_action": str(oracle.get("logged_action", "")),
                "action_mask": _action_mask(policy_state),
                "values": values,
            }
        )

    predictions: list[dict[str, object]] = []
    for row in rows:
        seed = row["seed"]
        values = row["values"]
        if not isinstance(values, list) or seed is None:
            continue
        by_action: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for other in rows:
            if other is row or other["seed"] == seed:
                continue
            other_action = str(other["action"])
            if bool(_mapping(row["action_mask"]).get(other_action, False)):
                by_action[other_action].append(other)
        if not by_action:
            continue
        distances = {
            action: min(
                _squared_distance(
                    values,
                    list(other.get("values", [])),
                )
                for other in action_rows
            )
            for action, action_rows in by_action.items()
        }
        predicted = min(distances, key=lambda action: (distances[action], action))
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": row["action"],
                "logged_action": row["logged_action"],
                "predicted_action": predicted,
                "correct": predicted == row["action"],
                "distance": _round(distances[predicted]),
            }
        )

    correct_count = sum(1 for item in predictions if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    eligible_count = len(predictions)
    accuracy = _safe_rate(correct_count, eligible_count)
    return {
        "policy": "leave_one_source_seed_out_nearest_policy_vector_v1",
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "eligible_label_count": eligible_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            eligible_count,
        ),
        "logged_matches_oracle_count": sum(
            1 for row in rows if row["logged_action"] == row["action"]
        ),
        "materially_supports_runtime_classifier": accuracy >= 0.5,
        "interpretation": (
            "negative_support_probe_do_not_train_runtime_classifier"
            if accuracy < 0.5
            else "positive_support_probe_candidate_classifier_worth_testing"
        ),
        "examples": predictions[:12],
    }


def _action_value_ranker_support_probe(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = _probe_rows(labels)
    results = [
        _action_value_ranker_result(rows, k=k)
        for k in (1, 3, 5)
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    return {
        "policy": "leave_one_source_seed_out_action_conditioned_nearest_value_v1",
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "objective": MIND_V3_BRANCH_ACTION_ORACLE_OBJECTIVE,
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "materially_supports_action_value_model": float(best["accuracy"]) >= 0.5,
        "interpretation": (
            "negative_support_probe_do_not_train_action_value_model"
            if float(best["accuracy"]) < 0.5
            else "positive_support_probe_action_value_model_worth_testing"
        ),
        "results": results,
    }


def _compact_outcome_world_model_support_probe(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = _compact_world_model_rows(labels)
    results = [_compact_world_model_result(rows, k=k) for k in (1, 3, 5, 9)]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    return {
        "policy": "leave_one_source_seed_out_compact_policy_visible_outcome_model_v1",
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "objective": MIND_V3_BRANCH_ACTION_ORACLE_OBJECTIVE,
        "feature_contract": {
            "schema_version": "compact_policy_visible_outcome_features_v1",
            "source_observation_schema": "mind_observation_v3",
            "privileged_world_state": False,
            "uses_fixture_identity": False,
            "features": list(_COMPACT_WORLD_MODEL_FEATURE_CONTRACT),
            "feature_digest": stable_payload_digest(
                {
                    "schema_version": "compact_policy_visible_outcome_features_v1",
                    "features": list(_COMPACT_WORLD_MODEL_FEATURE_CONTRACT),
                }
            ),
        },
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "material_support_accuracy_floor": _round(
            COMPACT_OUTCOME_WORLD_MODEL_ACCURACY_FLOOR
        ),
        "materially_supports_compact_world_model": (
            float(best["accuracy"]) >= COMPACT_OUTCOME_WORLD_MODEL_ACCURACY_FLOOR
        ),
        "interpretation": (
            "negative_support_probe_do_not_train_compact_world_model_policy"
            if float(best["accuracy"]) < COMPACT_OUTCOME_WORLD_MODEL_ACCURACY_FLOOR
            else "positive_support_probe_compact_world_model_policy_worth_testing"
        ),
        "results": results,
    }


def _short_horizon_trace_terminal_world_model_support_probe(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = _compact_world_model_rows(labels)
    results = [
        _short_horizon_trace_terminal_result(rows, k=k, max_horizon=max_horizon)
        for max_horizon in SHORT_HORIZON_TRACE_TICKS
        for k in (1, 3, 5, 9)
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            int(result["max_horizon_tick_delta"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    material = (
        float(best["accuracy"]) >= SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
    )
    return {
        "policy": (
            "leave_one_source_seed_out_terminal_value_with_actual_short_horizon_"
            "target_trace_upper_bound_v1"
        ),
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "objective": MIND_V3_BRANCH_ACTION_ORACLE_OBJECTIVE,
        "runtime_policy_status": (
            "diagnostic_upper_bound_only_actual_future_target_trace_not_available_"
            "at_decision_time"
        ),
        "trace_horizons": list(SHORT_HORIZON_TRACE_TICKS),
        "best_max_horizon_tick_delta": best["max_horizon_tick_delta"],
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "material_support_accuracy_floor": _round(
            SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
        ),
        "materially_supports_short_horizon_trace_model": material,
        "interpretation": (
            "negative_upper_bound_need_deeper_branch_value_or_more_coverage"
            if not material
            else "positive_upper_bound_short_horizon_trace_explains_terminal_oracle"
        ),
        "results": results,
    }


def _actual_horizon_trace_alignment_probe(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = _compact_world_model_rows(labels)
    results = [
        _actual_horizon_trace_alignment_result(rows, max_horizon=max_horizon)
        for max_horizon in SHORT_HORIZON_TRACE_TICKS
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            int(result["max_horizon_tick_delta"]),
        ),
    )
    material = float(best["accuracy"]) >= SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
    return {
        "policy": "actual_short_horizon_target_trace_need_score_alignment_v1",
        "objective": "actual_target_alive_vitals_resource_movement_trace_score_v1",
        "runtime_policy_status": (
            "diagnostic_only_actual_future_target_trace_not_available_at_decision_time"
        ),
        "trace_horizons": list(SHORT_HORIZON_TRACE_TICKS),
        "best_max_horizon_tick_delta": best["max_horizon_tick_delta"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "material_support_accuracy_floor": _round(
            SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
        ),
        "materially_supports_short_horizon_trace_signal": material,
        "interpretation": (
            "negative_actual_trace_signal_population_level_or_longer_credit_needed"
            if not material
            else "positive_actual_trace_signal_worth_modeling"
        ),
        "results": results,
    }


def _actual_population_horizon_alignment_probe(
    labels: Sequence[Mapping[str, object]],
    *,
    material_only: bool = False,
) -> dict[str, object]:
    rows = _horizon_probe_rows(labels, material_only=material_only)
    results = [
        _actual_population_horizon_alignment_result(rows, max_horizon=max_horizon)
        for max_horizon in SHORT_HORIZON_TRACE_TICKS
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            int(result["max_horizon_tick_delta"]),
        ),
    )
    material = float(best["accuracy"]) >= SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
    return {
        "policy": "actual_branch_population_horizon_trace_alignment_v1",
        "scope": _horizon_probe_scope(material_only),
        "objective": "alive_birth_target_death_diversity_population_trace_score_v1",
        "runtime_policy_status": (
            "diagnostic_only_actual_future_population_trace_not_available_"
            "at_decision_time"
        ),
        "trace_horizons": list(SHORT_HORIZON_TRACE_TICKS),
        "best_max_horizon_tick_delta": best["max_horizon_tick_delta"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "material_support_accuracy_floor": _round(
            SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
        ),
        "materially_supports_population_horizon_trace_signal": material,
        "interpretation": (
            "negative_population_trace_signal_needs_longer_archive_or_value_model"
            if not material
            else "positive_population_trace_signal_explains_terminal_oracle"
        ),
        "results": results,
    }


def _compact_population_horizon_world_model_support_probe(
    labels: Sequence[Mapping[str, object]],
    *,
    material_only: bool = False,
) -> dict[str, object]:
    rows = _horizon_probe_rows(labels, material_only=material_only)
    results = [
        _compact_population_horizon_result(rows, k=k, max_horizon=max_horizon)
        for max_horizon in SHORT_HORIZON_TRACE_TICKS
        for k in (1, 3, 5, 9)
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            int(result["max_horizon_tick_delta"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    material = float(best["accuracy"]) >= SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
    return {
        "policy": "leave_one_source_seed_out_compact_population_horizon_model_v1",
        "scope": _horizon_probe_scope(material_only),
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "objective": "predict_branch_population_horizon_score_then_rank_actions_v1",
        "best_max_horizon_tick_delta": best["max_horizon_tick_delta"],
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "material_support_accuracy_floor": _round(
            SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
        ),
        "materially_supports_population_horizon_model": material,
        "interpretation": (
            "negative_support_probe_do_not_train_population_horizon_model"
            if not material
            else "positive_support_probe_population_horizon_model_worth_testing"
        ),
        "results": results,
    }


def _policy_observation_population_horizon_world_model_support_probe(
    labels: Sequence[Mapping[str, object]],
    *,
    material_only: bool = False,
) -> dict[str, object]:
    rows = _horizon_probe_rows(labels, material_only=material_only)
    results = [
        _policy_observation_population_horizon_result(
            rows,
            k=k,
            max_horizon=max_horizon,
        )
        for max_horizon in SHORT_HORIZON_TRACE_TICKS
        for k in (1, 3, 5, 9)
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            int(result["max_horizon_tick_delta"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    material = float(best["accuracy"]) >= SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
    return {
        "policy": "leave_one_source_seed_out_policy_observation_population_horizon_model_v1",
        "scope": _horizon_probe_scope(material_only),
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "feature_contract": (
            "full_decoded_policy_observation_values_plus_action_mask_and_"
            "candidate_action_one_hot_v1"
        ),
        "objective": "predict_branch_population_horizon_score_then_rank_actions_v1",
        "best_max_horizon_tick_delta": best["max_horizon_tick_delta"],
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "material_support_accuracy_floor": _round(
            SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
        ),
        "materially_supports_population_horizon_model": material,
        "interpretation": (
            "negative_support_probe_do_not_train_full_observation_population_horizon_model"
            if not material
            else "positive_support_probe_full_observation_population_horizon_model_worth_testing"
        ),
        "results": results,
    }


def _policy_observation_history_population_horizon_world_model_support_probe(
    labels: Sequence[Mapping[str, object]],
    *,
    material_only: bool = False,
) -> dict[str, object]:
    rows = _horizon_probe_rows(labels, material_only=material_only)
    results = [
        _policy_observation_population_horizon_result(
            rows,
            k=k,
            max_horizon=max_horizon,
            include_history=True,
        )
        for max_horizon in SHORT_HORIZON_TRACE_TICKS
        for k in (1, 3, 5, 9)
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            int(result["max_horizon_tick_delta"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    material = float(best["accuracy"]) >= SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
    return {
        "policy": "leave_one_source_seed_out_policy_observation_history_population_horizon_model_v1",
        "scope": _horizon_probe_scope(material_only),
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "feature_contract": (
            "full_decoded_policy_observation_values_plus_action_mask_"
            "candidate_action_one_hot_and_same_agent_public_history_v1"
        ),
        "history_contract": {
            "max_steps": DEFAULT_BRANCH_ACTION_ORACLE_HISTORY_STEPS,
            "source": "same_agent_previous_public_trajectory_rows",
        },
        "objective": "predict_branch_population_horizon_score_then_rank_actions_v1",
        "best_max_horizon_tick_delta": best["max_horizon_tick_delta"],
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "material_support_accuracy_floor": _round(
            SHORT_HORIZON_TRACE_TERMINAL_ACCURACY_FLOOR
        ),
        "materially_supports_population_horizon_model": material,
        "interpretation": (
            "negative_support_probe_do_not_train_history_population_horizon_model"
            if not material
            else "positive_support_probe_history_population_horizon_model_worth_testing"
        ),
        "results": results,
    }


def _branch_continuation_archive_scorer_support_probe(
    labels: Sequence[Mapping[str, object]],
    *,
    material_only: bool = False,
) -> dict[str, object]:
    rows = _horizon_probe_rows(labels, material_only=material_only)
    results = [
        _branch_continuation_archive_result(
            rows,
            k=k,
            max_horizon=max_horizon,
            neighbor_policy=neighbor_policy,
        )
        for max_horizon in BRANCH_CONTINUATION_ARCHIVE_TRACE_TICKS
        for k in (1, 3, 5)
        for neighbor_policy in (
            "same_action",
            "same_option_mode",
            "all_archive_actions",
        )
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            int(result["max_horizon_tick_delta"]),
            _neighbor_policy_rank(str(result["neighbor_policy"])),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    material = (
        float(best["accuracy"])
        >= BRANCH_CONTINUATION_ARCHIVE_SCORER_ACCURACY_FLOOR
    )
    return {
        "policy": "leave_one_source_seed_out_branch_continuation_archive_scorer_v1",
        "scope": _horizon_probe_scope(material_only),
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "feature_contract": {
            "schema_version": "branch_continuation_archive_features_v1",
            "source_observation_schema": "mind_observation_v3",
            "privileged_world_state": False,
            "uses_fixture_identity": False,
            "uses_source_seed_tick_or_agent_id": False,
            "features": list(_BRANCH_CONTINUATION_ARCHIVE_FEATURE_CONTRACT),
            "history_contract": {
                "max_steps": DEFAULT_BRANCH_ACTION_ORACLE_HISTORY_STEPS,
                "source": "same_agent_previous_public_trajectory_rows",
            },
            "feature_digest": stable_payload_digest(
                {
                    "schema_version": "branch_continuation_archive_features_v1",
                    "features": list(
                        _BRANCH_CONTINUATION_ARCHIVE_FEATURE_CONTRACT
                    ),
                    "history_steps": DEFAULT_BRANCH_ACTION_ORACLE_HISTORY_STEPS,
                }
            ),
        },
        "target_contract": {
            "schema_version": "branch_continuation_archive_targets_v1",
            "target_source": "replay_verified_branch_population_horizon_trace",
            "first_action_imitation_target": False,
            "targets": list(_BRANCH_CONTINUATION_ARCHIVE_TARGET_CONTRACT),
            "target_digest": stable_payload_digest(
                {
                    "schema_version": "branch_continuation_archive_targets_v1",
                    "targets": list(
                        _BRANCH_CONTINUATION_ARCHIVE_TARGET_CONTRACT
                    ),
                }
            ),
        },
        "reference_pointwise_accuracy_range": {
            "lower": 0.35,
            "upper": 0.42,
            "source": "v84_v87_pointwise_population_horizon_support_range",
        },
        "trace_horizons": list(BRANCH_CONTINUATION_ARCHIVE_TRACE_TICKS),
        "best_max_horizon_tick_delta": best["max_horizon_tick_delta"],
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_neighbor_policy": best["neighbor_policy"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "material_support_accuracy_floor": _round(
            BRANCH_CONTINUATION_ARCHIVE_SCORER_ACCURACY_FLOOR
        ),
        "materially_supports_branch_continuation_archive_scorer": material,
        "interpretation": (
            "negative_support_probe_do_not_train_branch_continuation_policy"
            if not material
            else "positive_support_probe_branch_continuation_training_worth_testing"
        ),
        "results": results,
    }


def _horizon_probe_rows(
    labels: Sequence[Mapping[str, object]],
    *,
    material_only: bool,
) -> list[dict[str, object]]:
    rows = _compact_world_model_rows(labels)
    if not material_only:
        return rows
    return [
        row
        for row in rows
        if row.get("material_oracle_gain") is True
    ]


def _horizon_probe_scope(material_only: bool) -> str:
    if material_only:
        return "material_oracle_gain_labels_only"
    return "all_labels"


def _branch_continuation_archive_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
    max_horizon: int,
    neighbor_policy: str,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    all_examples = _branch_continuation_archive_examples(
        rows,
        held_out_seed=None,
        max_horizon=max_horizon,
    )
    for row in rows:
        seed = row["seed"]
        if seed is None:
            continue
        training_examples = [
            example for example in all_examples if example.get("seed") != seed
        ]
        if not training_examples:
            continue
        predicted_scores: dict[str, tuple[float, ...]] = {}
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _branch_continuation_archive_feature_vector(
                row=row,
                action=action,
            )
            if not features:
                continue
            neighbors = [
                (
                    _squared_distance(features, example["features"]),  # type: ignore[arg-type]
                    example["continuation_score"],  # type: ignore[index]
                )
                for example in training_examples
                if _archive_neighbor_matches(
                    action=action,
                    example_action=str(example.get("action", "")),
                    neighbor_policy=neighbor_policy,
                )
            ]
            if not neighbors:
                continue
            neighbors.sort(key=lambda item: item[0])
            selected = neighbors[: max(1, int(k))]
            predicted_scores[action] = _mean_continuation_score_tuple(
                [
                    score
                    for _, score in selected
                    if isinstance(score, tuple)
                ]
            )
        if not predicted_scores:
            continue
        predicted = max(
            predicted_scores,
            key=lambda action: (predicted_scores[action], action),
        )
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": row["action"],
                "predicted_action": predicted,
                "correct": predicted == row["action"],
                "neighbor_policy": neighbor_policy,
                "predicted_continuation_score": [
                    _round(value) for value in predicted_scores[predicted]
                ],
            }
        )
    correct_count = sum(1 for item in predictions if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    eligible_count = len(predictions)
    return {
        "max_horizon_tick_delta": int(max_horizon),
        "nearest_neighbor_k": int(k),
        "neighbor_policy": str(neighbor_policy),
        "eligible_label_count": eligible_count,
        "correct_count": correct_count,
        "accuracy": _safe_rate(correct_count, eligible_count),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            eligible_count,
        ),
        "examples": predictions[:12],
    }


def _branch_continuation_archive_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    held_out_seed: object | None,
    max_horizon: int,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row in rows:
        if row.get("seed") == held_out_seed:
            continue
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            continuation_score = _branch_continuation_outcome_score(
                candidate,
                max_horizon=max_horizon,
            )
            if continuation_score is None:
                continue
            features = _branch_continuation_archive_feature_vector(
                row=row,
                action=action,
            )
            if not features:
                continue
            examples.append(
                {
                    "seed": row.get("seed"),
                    "branch_id": row.get("branch_id"),
                    "action": action,
                    "features": features,
                    "continuation_score": continuation_score,
                }
            )
    return examples


def _branch_continuation_archive_feature_vector(
    *,
    row: Mapping[str, object],
    action: str,
) -> tuple[float, ...]:
    state = _mapping(row.get("compact_state"))
    if not state:
        return ()
    self_state = _mapping(state.get("self"))
    center = _mapping(state.get("center"))
    local = _mapping(state.get("local"))
    navigation = _mapping(state.get("navigation"))
    adjacent = _mapping(state.get("adjacent"))
    if not self_state or not center:
        return ()
    action_mask = _mapping(row.get("action_mask"))
    action_dx, action_dy = _MOVE_DELTAS.get(action, (0, 0))
    target_cell = _target_cell_for_action(action, adjacent)
    current_carrion = _cell_carrion(center)
    target_carrion = _cell_carrion(target_cell)
    history = _list_of_mappings(
        row.get("public_history_trace"),
        field="public_history_trace",
    )
    supported_count = sum(1 for value in action_mask.values() if bool(value))
    features = tuple(
        _round(value)
        for value in (
            _feature_float(self_state.get("energy_ratio")),
            _feature_float(self_state.get("hydration_ratio")),
            _feature_float(self_state.get("health_ratio")),
            1.0 - _feature_float(self_state.get("energy_ratio")),
            1.0 - _feature_float(self_state.get("hydration_ratio")),
            1.0 - _feature_float(self_state.get("health_ratio")),
            _feature_float(self_state.get("injury_load")),
            _feature_float(self_state.get("trophic_role_code")),
            _feature_float(self_state.get("meat_mode_code")),
            _feature_float(center.get("water")),
            _feature_float(center.get("food")),
            current_carrion,
            _cell_risk(center),
            _feature_float(local.get("radius1_water")),
            _feature_float(local.get("radius1_food")),
            _feature_float(local.get("radius1_carrion")),
            _feature_float(local.get("radius1_risk")),
            _feature_float(local.get("radius2_water")),
            _feature_float(local.get("radius2_food")),
            _feature_float(local.get("radius2_carrion")),
            _feature_float(local.get("radius2_risk")),
            *_navigation_features(navigation),
            *_action_one_hot(action),
            _round(action_dx),
            _round(action_dy),
            1.0 if bool(action_mask.get(action, False)) else 0.0,
            min(float(supported_count) / max(float(len(ACTION_NAMES)), 1.0), 1.0),
            1.0 if action in _MOVE_DELTAS else 0.0,
            1.0 if action == "drink" else 0.0,
            1.0 if action == "eat" else 0.0,
            1.0 if action == "stay" else 0.0,
            action_dx * _feature_float(target_cell.get("water")),
            action_dy * _feature_float(target_cell.get("water")),
            _feature_float(target_cell.get("food")),
            target_carrion,
            _cell_risk(target_cell),
            *_movement_navigation_alignment(action_dx, action_dy, navigation),
            *_public_history_summary_feature_vector(history),
        )
    )
    return tuple(_archive_quantize_feature(value) for value in features)


def _public_history_summary_feature_vector(
    history: Sequence[Mapping[str, object]],
) -> tuple[float, ...]:
    if not history:
        return (0.0,) * 16
    count = float(len(history))
    latest = _mapping(history[-1])
    return tuple(
        _round(value)
        for value in (
            min(count / float(DEFAULT_BRANCH_ACTION_ORACLE_HISTORY_STEPS), 1.0),
            sum(1.0 for item in history if bool(item.get("moved", False))) / count,
            sum(1.0 for item in history if bool(item.get("drank", False))) / count,
            sum(1.0 for item in history if bool(item.get("ate", False))) / count,
            sum(
                1.0
                for item in history
                if bool(item.get("post_carrion_contact", False))
            )
            / count,
            sum(_feature_float(item.get("resource_gain")) for item in history) / count,
            _feature_float(latest.get("energy_ratio_after")),
            _clamped_delta(latest.get("energy_ratio_delta")),
            _feature_float(latest.get("hydration_ratio_after")),
            _clamped_delta(latest.get("hydration_ratio_delta")),
            _feature_float(latest.get("health_ratio_after")),
            _clamped_delta(latest.get("health_ratio_delta")),
            min(_feature_float(latest.get("tick_delta")) / 120.0, 1.0),
            min(
                _feature_float(latest.get("ticks_since_animal_resource_gain")) / 32.0,
                1.0,
            ),
            min(_feature_float(latest.get("ticks_since_drink")) / 32.0, 1.0),
            _clamped_delta(latest.get("x_delta"))
            + _clamped_delta(latest.get("y_delta")),
        )
    )


def _branch_continuation_outcome_score(
    candidate: Mapping[str, object],
    *,
    max_horizon: int,
) -> tuple[float, ...] | None:
    trace_payload = candidate.get("population_horizon_trace")
    if not isinstance(trace_payload, list):
        return None
    trace = [
        item
        for item in _list_of_mappings(trace_payload, field="population_horizon_trace")
        if _int(item.get("horizon_tick_delta")) <= max_horizon
    ]
    if not trace:
        return None
    first = min(trace, key=lambda item: _int(item.get("horizon_tick_delta")))
    latest = max(trace, key=lambda item: _int(item.get("horizon_tick_delta")))
    count = float(len(trace))
    first_alive = float(_int(first.get("alive_agents")))
    first_births = float(_int(first.get("births")))
    first_deaths = float(_int(first.get("deaths")))
    alive_area_delta = sum(
        float(_int(item.get("alive_agents"))) - first_alive for item in trace
    ) / count
    birth_area_delta = sum(
        float(_int(item.get("births"))) - first_births for item in trace
    ) / count
    death_area_delta = sum(
        float(_int(item.get("deaths"))) - first_deaths for item in trace
    ) / count
    target_alive_area = sum(
        1.0 if bool(item.get("target_alive", False)) else 0.0 for item in trace
    ) / count
    target_vital_area = sum(_target_vital_floor(item) for item in trace) / count
    resource_gain_area = sum(
        _feature_float(item.get("tick_resource_gain")) for item in trace
    ) / count
    collapse_area = sum(
        _feature_float(item.get("tick_dominant_requested_action_share"))
        for item in trace
    ) / count
    return tuple(
        _round(value)
        for value in (
            float(_int(latest.get("alive_agents"))),
            float(_int(latest.get("births"))),
            target_alive_area,
            alive_area_delta,
            birth_area_delta,
            target_vital_area,
            resource_gain_area,
            -death_area_delta,
            -collapse_area,
        )
    )


def _mean_continuation_score_tuple(
    values: Sequence[tuple[float, ...]],
) -> tuple[float, ...]:
    if not values:
        return (0.0,) * len(_BRANCH_CONTINUATION_ARCHIVE_TARGET_CONTRACT)
    width = len(values[0])
    aligned = [value for value in values if len(value) == width]
    if not aligned:
        return (0.0,) * len(_BRANCH_CONTINUATION_ARCHIVE_TARGET_CONTRACT)
    count = float(len(aligned))
    return tuple(
        sum(value[index] for value in aligned) / count
        for index in range(width)
    )


def _archive_neighbor_matches(
    *,
    action: str,
    example_action: str,
    neighbor_policy: str,
) -> bool:
    if neighbor_policy == "same_action":
        return action == example_action
    if neighbor_policy == "same_option_mode":
        return _action_option_mode(action) == _action_option_mode(example_action)
    return neighbor_policy == "all_archive_actions"


def _neighbor_policy_rank(neighbor_policy: str) -> int:
    if neighbor_policy == "same_action":
        return 3
    if neighbor_policy == "same_option_mode":
        return 2
    return 1


def _archive_quantize_feature(value: float) -> float:
    return _round(round(float(value) * 8.0) / 8.0)


def _target_vital_floor(item: Mapping[str, object]) -> float:
    if item.get("target_alive") is not True:
        return 0.0
    vitals = (
        _feature_float(item.get("target_energy_ratio")),
        _feature_float(item.get("target_hydration_ratio")),
        _feature_float(item.get("target_health_ratio")),
    )
    return min(vitals)


def _compact_population_horizon_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
    max_horizon: int,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        seed = row["seed"]
        state = _mapping(row.get("compact_state"))
        if seed is None or not state:
            continue
        training_examples = _population_horizon_training_examples(
            rows,
            held_out_seed=seed,
            max_horizon=max_horizon,
        )
        if not training_examples:
            continue
        predicted_scores: dict[str, tuple[float, float, float, float, float, float]] = {}
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _compact_action_feature_vector(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
            )
            if not features:
                continue
            neighbors = [
                (
                    _squared_distance(features, example["features"]),  # type: ignore[arg-type]
                    example["population_score"],  # type: ignore[index]
                )
                for example in training_examples
                if example.get("action") == action
            ]
            if not neighbors:
                continue
            neighbors.sort(key=lambda item: item[0])
            selected = neighbors[: max(1, int(k))]
            predicted_scores[action] = _mean_population_score_tuple(
                [
                    score
                    for _, score in selected
                    if isinstance(score, tuple)
                ]
            )
        if not predicted_scores:
            continue
        predicted = max(
            predicted_scores,
            key=lambda action: (predicted_scores[action], action),
        )
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": row["action"],
                "predicted_action": predicted,
                "correct": predicted == row["action"],
                "predicted_population_score": [
                    _round(value) for value in predicted_scores[predicted]
                ],
            }
        )
    correct_count = sum(1 for item in predictions if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    eligible_count = len(predictions)
    return {
        "max_horizon_tick_delta": int(max_horizon),
        "nearest_neighbor_k": int(k),
        "eligible_label_count": eligible_count,
        "correct_count": correct_count,
        "accuracy": _safe_rate(correct_count, eligible_count),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            eligible_count,
        ),
        "examples": predictions[:12],
    }


def _policy_observation_population_horizon_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
    max_horizon: int,
    include_history: bool = False,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        seed = row["seed"]
        if seed is None:
            continue
        training_examples = _policy_observation_population_horizon_training_examples(
            rows,
            held_out_seed=seed,
            max_horizon=max_horizon,
            include_history=include_history,
        )
        if not training_examples:
            continue
        predicted_scores: dict[str, tuple[float, float, float, float, float, float]] = {}
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _policy_observation_population_feature_vector(
                row=row,
                action=action,
                include_history=include_history,
            )
            if not features:
                continue
            neighbors = [
                (
                    _squared_distance(features, example["features"]),  # type: ignore[arg-type]
                    example["population_score"],  # type: ignore[index]
                )
                for example in training_examples
                if example.get("action") == action
            ]
            if not neighbors:
                continue
            neighbors.sort(key=lambda item: item[0])
            selected = neighbors[: max(1, int(k))]
            predicted_scores[action] = _mean_population_score_tuple(
                [
                    score
                    for _, score in selected
                    if isinstance(score, tuple)
                ]
            )
        if not predicted_scores:
            continue
        predicted = max(
            predicted_scores,
            key=lambda action: (predicted_scores[action], action),
        )
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": row["action"],
                "predicted_action": predicted,
                "correct": predicted == row["action"],
                "predicted_population_score": [
                    _round(value) for value in predicted_scores[predicted]
                ],
            }
        )
    correct_count = sum(1 for item in predictions if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    eligible_count = len(predictions)
    return {
        "max_horizon_tick_delta": int(max_horizon),
        "nearest_neighbor_k": int(k),
        "eligible_label_count": eligible_count,
        "correct_count": correct_count,
        "accuracy": _safe_rate(correct_count, eligible_count),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            eligible_count,
        ),
        "examples": predictions[:12],
    }


def _population_horizon_training_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    held_out_seed: object,
    max_horizon: int,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row in rows:
        if row.get("seed") == held_out_seed:
            continue
        state = _mapping(row.get("compact_state"))
        if not state:
            continue
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            population_score = _actual_population_horizon_score(
                candidate,
                max_horizon=max_horizon,
            )
            if population_score is None:
                continue
            features = _compact_action_feature_vector(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
            )
            if not features:
                continue
            examples.append(
                {
                    "action": action,
                    "features": features,
                    "population_score": population_score,
                }
            )
    return examples


def _policy_observation_population_horizon_training_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    held_out_seed: object,
    max_horizon: int,
    include_history: bool = False,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row in rows:
        if row.get("seed") == held_out_seed:
            continue
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            population_score = _actual_population_horizon_score(
                candidate,
                max_horizon=max_horizon,
            )
            if population_score is None:
                continue
            features = _policy_observation_population_feature_vector(
                row=row,
                action=action,
                include_history=include_history,
            )
            if not features:
                continue
            examples.append(
                {
                    "action": action,
                    "features": features,
                    "population_score": population_score,
                }
            )
    return examples


def _policy_observation_population_feature_vector(
    *,
    row: Mapping[str, object],
    action: str,
    include_history: bool,
) -> tuple[float, ...]:
    observation_values = _observation_values(row)
    if not observation_values:
        return ()
    features = _policy_observation_action_feature_vector(
        observation_values=observation_values,
        action=action,
        action_mask=_mapping(row.get("action_mask")),
    )
    if not features:
        return ()
    if not include_history:
        return features
    history = _list_of_mappings(
        row.get("public_history_trace"),
        field="public_history_trace",
    )
    return features + _public_history_feature_vector(history)


def _mean_population_score_tuple(
    values: Sequence[tuple[float, float, float, float, float, float]],
) -> tuple[float, float, float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    count = float(len(values))
    return tuple(sum(value[index] for value in values) / count for index in range(6))


def _actual_population_horizon_alignment_result(
    rows: Sequence[Mapping[str, object]],
    *,
    max_horizon: int,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        scored: list[tuple[tuple[float, float, float, float, float, float], str]] = []
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            score = _actual_population_horizon_score(
                candidate,
                max_horizon=max_horizon,
            )
            if score is None:
                continue
            scored.append((score, action))
        if not scored:
            continue
        predicted = max(scored, key=lambda item: (item[0], item[1]))[1]
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": row["seed"],
                "oracle_action": row["action"],
                "predicted_action": predicted,
                "correct": predicted == row["action"],
            }
        )
    correct_count = sum(1 for item in predictions if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    eligible_count = len(predictions)
    return {
        "max_horizon_tick_delta": int(max_horizon),
        "eligible_label_count": eligible_count,
        "correct_count": correct_count,
        "accuracy": _safe_rate(correct_count, eligible_count),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            eligible_count,
        ),
        "examples": predictions[:12],
    }


def _actual_population_horizon_score(
    candidate: Mapping[str, object],
    *,
    max_horizon: int,
) -> tuple[float, float, float, float, float, float] | None:
    trace_payload = candidate.get("population_horizon_trace")
    if not isinstance(trace_payload, list):
        return None
    trace = [
        item
        for item in _list_of_mappings(trace_payload, field="population_horizon_trace")
        if _int(item.get("horizon_tick_delta")) <= max_horizon
    ]
    if not trace:
        return None
    latest = max(trace, key=lambda item: _int(item.get("horizon_tick_delta")))
    return (
        float(_int(latest.get("alive_agents"))),
        float(_int(latest.get("births"))),
        1.0 if bool(latest.get("target_alive", False)) else 0.0,
        -float(_int(latest.get("deaths"))),
        _feature_float(latest.get("tick_resource_gain")),
        -_feature_float(latest.get("tick_dominant_requested_action_share")),
    )


def _actual_horizon_trace_alignment_result(
    rows: Sequence[Mapping[str, object]],
    *,
    max_horizon: int,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        state = _mapping(row.get("compact_state"))
        if not state:
            continue
        scored: list[tuple[tuple[float, float, float, float, float, float], str]] = []
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            score = _actual_horizon_trace_score(
                state=state,
                action=action,
                candidate=candidate,
                max_horizon=max_horizon,
            )
            if score is None:
                continue
            scored.append((score, action))
        if not scored:
            continue
        predicted = max(scored, key=lambda item: (item[0], item[1]))[1]
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": row["seed"],
                "oracle_action": row["action"],
                "predicted_action": predicted,
                "correct": predicted == row["action"],
            }
        )
    correct_count = sum(1 for item in predictions if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    eligible_count = len(predictions)
    return {
        "max_horizon_tick_delta": int(max_horizon),
        "eligible_label_count": eligible_count,
        "correct_count": correct_count,
        "accuracy": _safe_rate(correct_count, eligible_count),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            eligible_count,
        ),
        "examples": predictions[:12],
    }


def _actual_horizon_trace_score(
    *,
    state: Mapping[str, object],
    action: str,
    candidate: Mapping[str, object],
    max_horizon: int,
) -> tuple[float, float, float, float, float, float] | None:
    trace_payload = candidate.get("target_horizon_trace")
    if not isinstance(trace_payload, list):
        return None
    trace = [
        item
        for item in _list_of_mappings(trace_payload, field="target_horizon_trace")
        if _int(item.get("horizon_tick_delta")) <= max_horizon
        and item.get("record_found") is True
    ]
    if not trace:
        return None
    latest = max(trace, key=lambda item: _int(item.get("horizon_tick_delta")))
    self_state = _mapping(state.get("self"))
    start_energy = _feature_float(self_state.get("energy_ratio"))
    start_hydration = _feature_float(self_state.get("hydration_ratio"))
    start_health = _feature_float(self_state.get("health_ratio"))
    energy = _feature_float(latest.get("energy_ratio_after"))
    hydration = _feature_float(latest.get("hydration_ratio_after"))
    health = _feature_float(latest.get("health_ratio_after"))
    resource_gain = sum(_feature_float(item.get("resource_gain")) for item in trace)
    moved_count = sum(1 for item in trace if bool(item.get("moved", False)))
    ate_count = sum(1 for item in trace if bool(item.get("ate", False)))
    drank_count = sum(1 for item in trace if bool(item.get("drank", False)))
    died = any(bool(item.get("died", False)) for item in trace)
    alive = latest.get("alive_after") is True and not died
    need_weighted_recovery = (
        (1.0 - start_energy) * (energy - start_energy)
        + (1.0 - start_hydration) * (hydration - start_hydration)
        + (1.0 - start_health) * (health - start_health)
        + resource_gain
    )
    action_mode_bonus = 0.0
    if action == "drink":
        action_mode_bonus = drank_count * max(0.0, 1.0 - start_hydration)
    elif action == "eat":
        action_mode_bonus = ate_count * max(0.0, 1.0 - start_energy)
    elif action in _MOVE_DELTAS:
        action_mode_bonus = min(1.0, moved_count / max(1.0, float(max_horizon + 1)))
    elif action == "stay":
        action_mode_bonus = 0.2 if moved_count == 0 else 0.0
    return (
        1.0 if alive else 0.0,
        -1.0 if died else 0.0,
        _round(min(energy, hydration, health)),
        _round(need_weighted_recovery + action_mode_bonus),
        _round(resource_gain),
        -_round(float(max_horizon)),
    )


def _short_horizon_trace_terminal_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
    max_horizon: int,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        seed = row["seed"]
        state = _mapping(row.get("compact_state"))
        if seed is None or not state:
            continue
        training_examples = _short_horizon_trace_terminal_examples(
            rows,
            held_out_seed=seed,
            max_horizon=max_horizon,
        )
        if not training_examples:
            continue
        predicted_scores: dict[str, tuple[float, float, float, float, float]] = {}
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _short_horizon_trace_terminal_features(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
                candidate=candidate,
                max_horizon=max_horizon,
            )
            if not features:
                continue
            neighbors = [
                (
                    _squared_distance(features, example["features"]),  # type: ignore[arg-type]
                    example["objective"],  # type: ignore[index]
                )
                for example in training_examples
                if example.get("action") == action
            ]
            if not neighbors:
                continue
            neighbors.sort(key=lambda item: item[0])
            selected = neighbors[: max(1, int(k))]
            predicted_scores[action] = _mean_objective_tuple(
                [
                    objective
                    for _, objective in selected
                    if isinstance(objective, tuple)
                ]
            )
        if not predicted_scores:
            continue
        predicted = max(
            predicted_scores,
            key=lambda action: (predicted_scores[action], action),
        )
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": row["action"],
                "predicted_action": predicted,
                "correct": predicted == row["action"],
                "predicted_objective_tuple": [
                    _round(value) for value in predicted_scores[predicted]
                ],
            }
        )
    correct_count = sum(1 for item in predictions if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    eligible_count = len(predictions)
    return {
        "max_horizon_tick_delta": int(max_horizon),
        "nearest_neighbor_k": int(k),
        "eligible_label_count": eligible_count,
        "correct_count": correct_count,
        "accuracy": _safe_rate(correct_count, eligible_count),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            eligible_count,
        ),
        "examples": predictions[:12],
    }


def _short_horizon_trace_terminal_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    held_out_seed: object,
    max_horizon: int,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row in rows:
        if row.get("seed") == held_out_seed:
            continue
        state = _mapping(row.get("compact_state"))
        if not state:
            continue
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _short_horizon_trace_terminal_features(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
                candidate=candidate,
                max_horizon=max_horizon,
            )
            if not features:
                continue
            examples.append(
                {
                    "action": action,
                    "features": features,
                    "objective": _objective_tuple(candidate.get("objective_tuple")),
                }
            )
    return examples


def _short_horizon_trace_terminal_features(
    *,
    state: Mapping[str, object],
    action: str,
    action_mask: Mapping[str, object],
    candidate: Mapping[str, object],
    max_horizon: int,
) -> tuple[float, ...]:
    base = _compact_action_feature_vector(
        state=state,
        action=action,
        action_mask=action_mask,
    )
    trace_features = _horizon_trace_features(
        candidate,
        max_horizon=max_horizon,
    )
    if not base or not trace_features:
        return ()
    return tuple(base) + trace_features


def _horizon_trace_features(
    candidate: Mapping[str, object],
    *,
    max_horizon: int,
) -> tuple[float, ...]:
    trace_payload = candidate.get("target_horizon_trace")
    if not isinstance(trace_payload, list):
        return ()
    trace_items = _list_of_mappings(trace_payload, field="target_horizon_trace")
    by_horizon = {
        _int(item.get("horizon_tick_delta")): item
        for item in trace_items
    }
    values: list[float] = []
    for horizon in SHORT_HORIZON_TRACE_TICKS:
        if horizon > max_horizon:
            continue
        item = _mapping(by_horizon.get(horizon))
        found = item.get("record_found") is True
        values.extend(
            [
                1.0 if found else 0.0,
                1.0 if bool(item.get("alive_after", False)) else 0.0,
                1.0 if bool(item.get("died", False)) else 0.0,
                _feature_float(item.get("energy_ratio_after")),
                _feature_float(item.get("hydration_ratio_after")),
                _feature_float(item.get("health_ratio_after")),
                _feature_float(item.get("energy_ratio_delta")),
                _feature_float(item.get("hydration_ratio_delta")),
                _feature_float(item.get("health_ratio_delta")),
                _feature_float(item.get("resource_gain")),
                1.0 if bool(item.get("moved", False)) else 0.0,
                1.0 if bool(item.get("drank", False)) else 0.0,
                1.0 if bool(item.get("ate", False)) else 0.0,
                _clamped_delta(item.get("x_delta")),
                _clamped_delta(item.get("y_delta")),
            ]
        )
    return tuple(_round(value) for value in values)


def _compact_option_mode_world_model_support_probe(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = _compact_world_model_rows(labels)
    results = [_compact_option_mode_world_model_result(rows, k=k) for k in (1, 3, 5, 9)]
    best = max(
        results,
        key=lambda result: (
            float(result["mode_accuracy"]),
            int(result["mode_correct_count"]),
            float(result["exact_action_accuracy"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    material = (
        float(best["mode_accuracy"]) >= COMPACT_OPTION_MODE_ACCURACY_FLOOR
        and float(best["dominant_prediction_mode_share"])
        <= COMPACT_OPTION_MODE_MAX_DOMINANT_PREDICTION_SHARE
    )
    return {
        "policy": "leave_one_source_seed_out_compact_option_mode_world_model_v1",
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "objective": MIND_V3_BRANCH_ACTION_ORACLE_OBJECTIVE,
        "option_modes": {
            "recover_hydration": ["drink"],
            "exploit_resource": ["eat"],
            "reposition": [
                "move_north",
                "move_south",
                "move_east",
                "move_west",
            ],
            "conserve": ["stay"],
            "other": [
                action
                for action in ACTION_NAMES
                if _action_option_mode(action) == "other"
            ],
        },
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_mode_accuracy": best["mode_accuracy"],
        "best_mode_correct_count": best["mode_correct_count"],
        "best_exact_action_accuracy": best["exact_action_accuracy"],
        "material_support_mode_accuracy_floor": _round(
            COMPACT_OPTION_MODE_ACCURACY_FLOOR
        ),
        "max_dominant_prediction_mode_share": _round(
            COMPACT_OPTION_MODE_MAX_DOMINANT_PREDICTION_SHARE
        ),
        "materially_supports_option_mode_model": material,
        "interpretation": (
            "negative_support_probe_do_not_train_option_mode_policy"
            if not material
            else "positive_support_probe_option_mode_policy_worth_testing"
        ),
        "results": results,
    }


def _compact_reposition_direction_world_model_support_probe(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = [
        row
        for row in _compact_world_model_rows(labels)
        if _action_option_mode(str(row.get("action", ""))) == "reposition"
    ]
    results = [
        _compact_reposition_direction_world_model_result(rows, k=k)
        for k in (1, 3, 5, 9)
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            float(result["multi_move_accuracy"]),
            int(result["correct_count"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    material = (
        float(best["accuracy"]) >= COMPACT_REPOSITION_DIRECTION_ACCURACY_FLOOR
        and float(best["multi_move_accuracy"])
        >= COMPACT_REPOSITION_MULTI_MOVE_ACCURACY_FLOOR
    )
    return {
        "policy": "leave_one_source_seed_out_compact_reposition_direction_model_v1",
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "objective": MIND_V3_BRANCH_ACTION_ORACLE_OBJECTIVE,
        "scope": (
            "only rows whose oracle option mode is reposition; single legal move "
            "rows are counted separately from multi-move rows"
        ),
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "best_multi_move_accuracy": best["multi_move_accuracy"],
        "best_multi_move_correct_count": best["multi_move_correct_count"],
        "material_support_accuracy_floor": _round(
            COMPACT_REPOSITION_DIRECTION_ACCURACY_FLOOR
        ),
        "material_support_multi_move_accuracy_floor": _round(
            COMPACT_REPOSITION_MULTI_MOVE_ACCURACY_FLOOR
        ),
        "materially_supports_reposition_direction_model": material,
        "interpretation": (
            "negative_support_probe_do_not_train_reposition_direction_head"
            if not material
            else "positive_support_probe_reposition_direction_head_worth_testing"
        ),
        "results": results,
    }


def _compact_reposition_direction_world_model_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        seed = row["seed"]
        state = _mapping(row.get("compact_state"))
        if seed is None or not state:
            continue
        move_candidates = [
            candidate
            for candidate in _list_of_mappings(
                row.get("action_values"),
                field="action_values",
            )
            if str(candidate.get("action", "")) in _MOVE_DELTAS
        ]
        if not move_candidates:
            continue
        if len(move_candidates) == 1:
            predicted = str(move_candidates[0].get("action", ""))
        else:
            training_examples = _compact_training_examples(rows, held_out_seed=seed)
            predicted_scores: dict[str, tuple[float, float, float, float, float]] = {}
            for candidate in move_candidates:
                action = str(candidate.get("action", ""))
                features = _compact_action_feature_vector(
                    state=state,
                    action=action,
                    action_mask=_mapping(row.get("action_mask")),
                )
                if not features:
                    continue
                neighbors = [
                    (
                        _squared_distance(features, example["features"]),  # type: ignore[arg-type]
                        example["objective"],  # type: ignore[index]
                    )
                    for example in training_examples
                    if example.get("action") == action
                ]
                if not neighbors:
                    continue
                neighbors.sort(key=lambda item: item[0])
                selected = neighbors[: max(1, int(k))]
                predicted_scores[action] = _mean_objective_tuple(
                    [
                        objective
                        for _, objective in selected
                        if isinstance(objective, tuple)
                    ]
                )
            if not predicted_scores:
                continue
            predicted = max(
                predicted_scores,
                key=lambda action: (predicted_scores[action], action),
            )
        oracle = str(row.get("action", ""))
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": oracle,
                "predicted_action": predicted,
                "correct": predicted == oracle,
                "move_candidate_count": len(move_candidates),
            }
        )
    correct = sum(1 for item in predictions if item["correct"] is True)
    multi = [
        item for item in predictions if int(item["move_candidate_count"]) > 1
    ]
    multi_correct = sum(1 for item in multi if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    count = len(predictions)
    return {
        "nearest_neighbor_k": int(k),
        "eligible_label_count": count,
        "correct_count": correct,
        "accuracy": _safe_rate(correct, count),
        "multi_move_label_count": len(multi),
        "multi_move_correct_count": multi_correct,
        "multi_move_accuracy": _safe_rate(multi_correct, len(multi)),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(dominant_count, count),
        "examples": predictions[:12],
    }


def _compact_option_mode_world_model_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        seed = row["seed"]
        state = _mapping(row.get("compact_state"))
        if seed is None or not state:
            continue
        training_examples = _compact_training_examples(rows, held_out_seed=seed)
        if not training_examples:
            continue
        predicted_scores: dict[str, tuple[float, float, float, float, float]] = {}
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _compact_action_feature_vector(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
            )
            if not features:
                continue
            neighbors = [
                (
                    _squared_distance(features, example["features"]),  # type: ignore[arg-type]
                    example["objective"],  # type: ignore[index]
                )
                for example in training_examples
                if example.get("action") == action
            ]
            if not neighbors:
                continue
            neighbors.sort(key=lambda item: item[0])
            selected = neighbors[: max(1, int(k))]
            predicted_scores[action] = _mean_objective_tuple(
                [
                    objective
                    for _, objective in selected
                    if isinstance(objective, tuple)
                ]
            )
        if not predicted_scores:
            continue
        predicted = max(
            predicted_scores,
            key=lambda action: (predicted_scores[action], action),
        )
        oracle_action = str(row["action"])
        predicted_mode = _action_option_mode(predicted)
        oracle_mode = _action_option_mode(oracle_action)
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": oracle_action,
                "oracle_mode": oracle_mode,
                "predicted_action": predicted,
                "predicted_mode": predicted_mode,
                "exact_action_correct": predicted == oracle_action,
                "mode_correct": predicted_mode == oracle_mode,
            }
        )
    exact_correct = sum(
        1 for item in predictions if item["exact_action_correct"] is True
    )
    mode_correct = sum(1 for item in predictions if item["mode_correct"] is True)
    action_counts = Counter(str(item["predicted_action"]) for item in predictions)
    mode_counts = Counter(str(item["predicted_mode"]) for item in predictions)
    dominant_action, dominant_action_count = _dominant_count(action_counts)
    dominant_mode, dominant_mode_count = _dominant_count(mode_counts)
    count = len(predictions)
    return {
        "nearest_neighbor_k": int(k),
        "eligible_label_count": count,
        "exact_action_correct_count": exact_correct,
        "exact_action_accuracy": _safe_rate(exact_correct, count),
        "mode_correct_count": mode_correct,
        "mode_accuracy": _safe_rate(mode_correct, count),
        "prediction_action_counts": dict(sorted(action_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_action_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_action_count,
            count,
        ),
        "prediction_mode_counts": dict(sorted(mode_counts.items())),
        "dominant_prediction_mode": dominant_mode,
        "dominant_prediction_mode_count": dominant_mode_count,
        "dominant_prediction_mode_share": _safe_rate(dominant_mode_count, count),
        "examples": predictions[:12],
    }


def _compact_first_step_world_model_support_probe(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = _compact_world_model_rows(labels)
    alignment = _actual_first_step_terminal_alignment(rows)
    results = [_compact_first_step_world_model_result(rows, k=k) for k in (1, 3, 5, 9)]
    best = max(
        results,
        key=lambda result: (
            float(result["terminal_oracle_accuracy"]),
            float(result["actual_first_step_accuracy"]),
            int(result["terminal_oracle_correct_count"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    material = (
        float(alignment["terminal_oracle_accuracy"])
        >= COMPACT_OUTCOME_WORLD_MODEL_ACCURACY_FLOOR
        and float(best["terminal_oracle_accuracy"])
        >= COMPACT_OUTCOME_WORLD_MODEL_ACCURACY_FLOOR
        and float(best["actual_first_step_accuracy"])
        >= COMPACT_OUTCOME_WORLD_MODEL_ACCURACY_FLOOR
    )
    return {
        "policy": "leave_one_source_seed_out_compact_first_step_world_model_v1",
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "objective": (
            "first_step_alive_need_weighted_vital_delta_resource_movement_v1"
        ),
        "feature_contract": {
            "schema_version": "compact_policy_visible_first_step_features_v1",
            "source_observation_schema": "mind_observation_v3",
            "source_action_outcome_schema": "mind_action_outcome_v2",
            "privileged_world_state": False,
            "uses_fixture_identity": False,
            "features": list(_COMPACT_WORLD_MODEL_FEATURE_CONTRACT),
            "targets": [
                "alive_after",
                "died",
                "energy_ratio_delta",
                "hydration_ratio_delta",
                "health_ratio_delta",
                "resource_gain",
                "moved",
                "drank",
                "ate",
            ],
            "feature_digest": stable_payload_digest(
                {
                    "schema_version": "compact_policy_visible_first_step_features_v1",
                    "features": list(_COMPACT_WORLD_MODEL_FEATURE_CONTRACT),
                    "targets": [
                        "alive_after",
                        "died",
                        "energy_ratio_delta",
                        "hydration_ratio_delta",
                        "health_ratio_delta",
                        "resource_gain",
                        "moved",
                        "drank",
                        "ate",
                    ],
                }
            ),
        },
        "actual_first_step_terminal_alignment": alignment,
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_actual_first_step_accuracy": best["actual_first_step_accuracy"],
        "best_terminal_oracle_accuracy": best["terminal_oracle_accuracy"],
        "best_terminal_oracle_correct_count": best[
            "terminal_oracle_correct_count"
        ],
        "material_support_accuracy_floor": _round(
            COMPACT_OUTCOME_WORLD_MODEL_ACCURACY_FLOOR
        ),
        "materially_supports_first_step_world_model_policy": material,
        "interpretation": (
            "negative_support_probe_first_step_model_not_sufficient_for_training"
            if not material
            else "positive_support_probe_first_step_model_policy_worth_testing"
        ),
        "results": results,
    }


def _first_step_augmented_terminal_world_model_support_probe(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = _compact_world_model_rows(labels)
    results = [
        _first_step_augmented_terminal_result(rows, k=k) for k in (1, 3, 5, 9)
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    material = float(best["accuracy"]) >= COMPACT_OUTCOME_WORLD_MODEL_ACCURACY_FLOOR
    return {
        "policy": (
            "leave_one_source_seed_out_terminal_value_with_actual_first_step_"
            "outcome_upper_bound_v1"
        ),
        "split_policy": "hold_out_all_labels_from_same_source_seed_v1",
        "objective": MIND_V3_BRANCH_ACTION_ORACLE_OBJECTIVE,
        "runtime_policy_status": (
            "diagnostic_upper_bound_only_actual_first_step_outcome_not_available_"
            "at_decision_time"
        ),
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "material_support_accuracy_floor": _round(
            COMPACT_OUTCOME_WORLD_MODEL_ACCURACY_FLOOR
        ),
        "materially_supports_first_step_augmented_terminal_model": material,
        "interpretation": (
            "negative_upper_bound_deeper_temporal_credit_required"
            if not material
            else "positive_upper_bound_trainable_if_first_step_model_is_accurate"
        ),
        "results": results,
    }


def _first_step_augmented_terminal_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        seed = row["seed"]
        state = _mapping(row.get("compact_state"))
        if seed is None or not state:
            continue
        training_examples = _first_step_augmented_terminal_examples(
            rows,
            held_out_seed=seed,
        )
        if not training_examples:
            continue
        predicted_scores: dict[str, tuple[float, float, float, float, float]] = {}
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _first_step_augmented_terminal_features(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
                candidate=candidate,
            )
            if not features:
                continue
            neighbors = [
                (
                    _squared_distance(features, example["features"]),  # type: ignore[arg-type]
                    example["objective"],  # type: ignore[index]
                )
                for example in training_examples
                if example.get("action") == action
            ]
            if not neighbors:
                continue
            neighbors.sort(key=lambda item: item[0])
            selected = neighbors[: max(1, int(k))]
            predicted_scores[action] = _mean_objective_tuple(
                [
                    objective
                    for _, objective in selected
                    if isinstance(objective, tuple)
                ]
            )
        if not predicted_scores:
            continue
        predicted = max(
            predicted_scores,
            key=lambda action: (predicted_scores[action], action),
        )
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": row["action"],
                "predicted_action": predicted,
                "correct": predicted == row["action"],
                "predicted_objective_tuple": [
                    _round(value) for value in predicted_scores[predicted]
                ],
            }
        )
    correct_count = sum(1 for item in predictions if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    eligible_count = len(predictions)
    return {
        "nearest_neighbor_k": int(k),
        "eligible_label_count": eligible_count,
        "correct_count": correct_count,
        "accuracy": _safe_rate(correct_count, eligible_count),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            eligible_count,
        ),
        "examples": predictions[:12],
    }


def _first_step_augmented_terminal_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    held_out_seed: object,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row in rows:
        if row.get("seed") == held_out_seed:
            continue
        state = _mapping(row.get("compact_state"))
        if not state:
            continue
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _first_step_augmented_terminal_features(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
                candidate=candidate,
            )
            if not features:
                continue
            examples.append(
                {
                    "action": action,
                    "features": features,
                    "objective": _objective_tuple(candidate.get("objective_tuple")),
                }
            )
    return examples


def _first_step_augmented_terminal_features(
    *,
    state: Mapping[str, object],
    action: str,
    action_mask: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[float, ...]:
    first_step = _first_step_outcome_tuple(candidate)
    if first_step is None:
        return ()
    base = _compact_action_feature_vector(
        state=state,
        action=action,
        action_mask=action_mask,
    )
    if not base:
        return ()
    return tuple(base) + tuple(_round(value) for value in first_step)


def _actual_first_step_terminal_alignment(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    predictions = []
    for row in rows:
        state = _mapping(row.get("compact_state"))
        if not state:
            continue
        actual = _best_first_step_action(row, state=state, predicted_outcomes=None)
        if actual is None:
            continue
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": row["seed"],
                "oracle_action": row["action"],
                "actual_first_step_action": actual,
                "matches_terminal_oracle": actual == row["action"],
            }
        )
    correct = sum(
        1 for item in predictions if item["matches_terminal_oracle"] is True
    )
    count = len(predictions)
    counts = Counter(str(item["actual_first_step_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(counts)
    return {
        "eligible_label_count": count,
        "terminal_oracle_correct_count": correct,
        "terminal_oracle_accuracy": _safe_rate(correct, count),
        "actual_first_step_action_counts": dict(sorted(counts.items())),
        "dominant_actual_first_step_action": dominant_action,
        "dominant_actual_first_step_action_count": dominant_count,
        "dominant_actual_first_step_action_share": _safe_rate(
            dominant_count,
            count,
        ),
        "examples": predictions[:12],
    }


def _compact_first_step_world_model_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        seed = row["seed"]
        state = _mapping(row.get("compact_state"))
        if seed is None or not state:
            continue
        training_examples = _first_step_training_examples(rows, held_out_seed=seed)
        if not training_examples:
            continue
        predicted_outcomes: dict[
            str,
            tuple[float, float, float, float, float, float, float, float, float],
        ] = {}
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _compact_action_feature_vector(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
            )
            if not features:
                continue
            neighbors = [
                (
                    _squared_distance(features, example["features"]),  # type: ignore[arg-type]
                    example["first_step_outcome"],  # type: ignore[index]
                )
                for example in training_examples
                if example.get("action") == action
            ]
            if not neighbors:
                continue
            neighbors.sort(key=lambda item: item[0])
            selected = neighbors[: max(1, int(k))]
            predicted_outcomes[action] = _mean_first_step_outcome_tuple(
                [
                    outcome
                    for _, outcome in selected
                    if isinstance(outcome, tuple)
                ]
            )
        if not predicted_outcomes:
            continue
        predicted = _best_first_step_action(
            row,
            state=state,
            predicted_outcomes=predicted_outcomes,
        )
        actual = _best_first_step_action(row, state=state, predicted_outcomes=None)
        if predicted is None or actual is None:
            continue
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": row["action"],
                "actual_first_step_action": actual,
                "predicted_action": predicted,
                "matches_actual_first_step": predicted == actual,
                "matches_terminal_oracle": predicted == row["action"],
                "predicted_first_step_tuple": [
                    _round(value) for value in predicted_outcomes[predicted]
                ],
            }
        )
    actual_correct = sum(
        1 for item in predictions if item["matches_actual_first_step"] is True
    )
    terminal_correct = sum(
        1 for item in predictions if item["matches_terminal_oracle"] is True
    )
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    count = len(predictions)
    return {
        "nearest_neighbor_k": int(k),
        "eligible_label_count": count,
        "actual_first_step_correct_count": actual_correct,
        "actual_first_step_accuracy": _safe_rate(actual_correct, count),
        "terminal_oracle_correct_count": terminal_correct,
        "terminal_oracle_accuracy": _safe_rate(terminal_correct, count),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            count,
        ),
        "examples": predictions[:12],
    }


def _first_step_training_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    held_out_seed: object,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row in rows:
        if row.get("seed") == held_out_seed:
            continue
        state = _mapping(row.get("compact_state"))
        if not state:
            continue
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            outcome = _first_step_outcome_tuple(candidate)
            if outcome is None:
                continue
            features = _compact_action_feature_vector(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
            )
            if not features:
                continue
            examples.append(
                {
                    "action": action,
                    "features": features,
                    "first_step_outcome": outcome,
                }
            )
    return examples


def _best_first_step_action(
    row: Mapping[str, object],
    *,
    state: Mapping[str, object],
    predicted_outcomes: Mapping[
        str,
        tuple[float, float, float, float, float, float, float, float, float],
    ]
    | None,
) -> str | None:
    scored: list[tuple[tuple[float, float, float, float, float, float], str]] = []
    for candidate in _list_of_mappings(
        row.get("action_values"),
        field="action_values",
    ):
        action = str(candidate.get("action", ""))
        outcome = (
            predicted_outcomes.get(action)
            if predicted_outcomes is not None
            else _first_step_outcome_tuple(candidate)
        )
        if outcome is None:
            continue
        scored.append((_first_step_score_tuple(state, action, outcome), action))
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], item[1]))[1]


def _first_step_score_tuple(
    state: Mapping[str, object],
    action: str,
    outcome: tuple[float, float, float, float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    self_state = _mapping(state.get("self"))
    adjacent = _mapping(state.get("adjacent"))
    navigation = _mapping(state.get("navigation"))
    energy = _feature_float(self_state.get("energy_ratio"))
    hydration = _feature_float(self_state.get("hydration_ratio"))
    health = _feature_float(self_state.get("health_ratio"))
    energy_delta = outcome[2]
    hydration_delta = outcome[3]
    health_delta = outcome[4]
    resource_gain = outcome[5]
    moved = outcome[6]
    action_dx, action_dy = _MOVE_DELTAS.get(action, (0, 0))
    target = _target_cell_for_action(action, adjacent)
    movement_alignment = max(
        _movement_navigation_alignment(action_dx, action_dy, navigation),
        default=0.0,
    )
    movement_resource = 0.0
    if action in _MOVE_DELTAS:
        movement_resource = max(
            _feature_float(target.get("water")) * (1.0 - hydration),
            _feature_float(target.get("food")) * (1.0 - energy),
            _cell_carrion(target) * (1.0 - energy),
            _feature_float(target.get("prey_biomass")) * (1.0 - energy),
            movement_alignment,
        ) - _cell_risk(target) * 0.1
    need_weighted_delta = (
        (1.0 - energy) * energy_delta
        + (1.0 - hydration) * hydration_delta
        + (1.0 - health) * health_delta
        + resource_gain
        + movement_resource
    )
    return (
        outcome[0],
        -outcome[1],
        _round(need_weighted_delta),
        _round(resource_gain),
        _round(moved),
        -_round(_cell_risk(target) if action in _MOVE_DELTAS else 0.0),
    )


def _first_step_outcome_tuple(
    candidate: Mapping[str, object],
) -> tuple[float, float, float, float, float, float, float, float, float] | None:
    outcome = _mapping(candidate.get("first_action_outcome"))
    if outcome.get("record_found") is not True:
        return None
    return (
        1.0 if bool(outcome.get("alive_after", False)) else 0.0,
        1.0 if bool(outcome.get("died", False)) else 0.0,
        _feature_float(outcome.get("energy_ratio_delta")),
        _feature_float(outcome.get("hydration_ratio_delta")),
        _feature_float(outcome.get("health_ratio_delta")),
        _feature_float(outcome.get("resource_gain")),
        1.0 if bool(outcome.get("moved", False)) else 0.0,
        1.0 if bool(outcome.get("drank", False)) else 0.0,
        1.0 if bool(outcome.get("ate", False)) else 0.0,
    )


def _mean_first_step_outcome_tuple(
    values: Sequence[
        tuple[float, float, float, float, float, float, float, float, float]
    ],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    count = float(len(values))
    return tuple(sum(value[index] for value in values) / count for index in range(9))


def _compact_world_model_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        seed = row["seed"]
        state = _mapping(row.get("compact_state"))
        if seed is None or not state:
            continue
        training_examples = _compact_training_examples(rows, held_out_seed=seed)
        if not training_examples:
            continue
        predicted_scores: dict[str, tuple[float, float, float, float, float]] = {}
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _compact_action_feature_vector(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
            )
            if not features:
                continue
            neighbors = [
                (
                    _squared_distance(features, example["features"]),  # type: ignore[arg-type]
                    example["objective"],  # type: ignore[index]
                )
                for example in training_examples
                if example.get("action") == action
            ]
            if not neighbors:
                continue
            neighbors.sort(key=lambda item: item[0])
            selected = neighbors[: max(1, int(k))]
            predicted_scores[action] = _mean_objective_tuple(
                [
                    objective
                    for _, objective in selected
                    if isinstance(objective, tuple)
                ]
            )
        if not predicted_scores:
            continue
        predicted = max(
            predicted_scores,
            key=lambda action: (predicted_scores[action], action),
        )
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": row["action"],
                "predicted_action": predicted,
                "correct": predicted == row["action"],
                "predicted_objective_tuple": [
                    _round(value) for value in predicted_scores[predicted]
                ],
            }
        )
    correct_count = sum(1 for item in predictions if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    eligible_count = len(predictions)
    return {
        "nearest_neighbor_k": int(k),
        "eligible_label_count": eligible_count,
        "correct_count": correct_count,
        "accuracy": _safe_rate(correct_count, eligible_count),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            eligible_count,
        ),
        "examples": predictions[:12],
    }


def _compact_training_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    held_out_seed: object,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row in rows:
        if row.get("seed") == held_out_seed:
            continue
        state = _mapping(row.get("compact_state"))
        if not state:
            continue
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            features = _compact_action_feature_vector(
                state=state,
                action=action,
                action_mask=_mapping(row.get("action_mask")),
            )
            if not features:
                continue
            examples.append(
                {
                    "action": action,
                    "features": features,
                    "objective": _objective_tuple(candidate.get("objective_tuple")),
                }
            )
    return examples


def _action_value_ranker_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        seed = row["seed"]
        values = list(row.get("values", []))
        if seed is None or not values:
            continue
        predicted_scores: dict[str, tuple[float, float, float, float, float]] = {}
        for candidate in _list_of_mappings(
            row.get("action_values"),
            field="action_values",
        ):
            action = str(candidate.get("action", ""))
            neighbors: list[tuple[float, tuple[float, float, float, float, float]]] = []
            for other in rows:
                if other is row or other.get("seed") == seed:
                    continue
                other_values = list(other.get("values", []))
                if not other_values:
                    continue
                for other_candidate in _list_of_mappings(
                    other.get("action_values"),
                    field="action_values",
                ):
                    if str(other_candidate.get("action", "")) != action:
                        continue
                    neighbors.append(
                        (
                            _squared_distance(values, other_values),
                            _objective_tuple(other_candidate.get("objective_tuple")),
                        )
                    )
            if neighbors:
                neighbors.sort(key=lambda item: item[0])
                selected = neighbors[: max(1, int(k))]
                predicted_scores[action] = _mean_objective_tuple(
                    [item[1] for item in selected]
                )
        if not predicted_scores:
            continue
        predicted = max(
            predicted_scores,
            key=lambda action: (predicted_scores[action], action),
        )
        predictions.append(
            {
                "branch_id": row["branch_id"],
                "seed": seed,
                "oracle_action": row["action"],
                "predicted_action": predicted,
                "correct": predicted == row["action"],
                "predicted_objective_tuple": [
                    _round(value) for value in predicted_scores[predicted]
                ],
            }
        )
    correct_count = sum(1 for item in predictions if item["correct"] is True)
    prediction_counts = Counter(str(item["predicted_action"]) for item in predictions)
    dominant_action, dominant_count = _dominant_count(prediction_counts)
    eligible_count = len(predictions)
    return {
        "nearest_neighbor_k": int(k),
        "eligible_label_count": eligible_count,
        "correct_count": correct_count,
        "accuracy": _safe_rate(correct_count, eligible_count),
        "prediction_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(
            dominant_count,
            eligible_count,
        ),
        "examples": predictions[:12],
    }


def _probe_rows(labels: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for label in labels:
        policy_state = _mapping(label.get("policy_state"))
        source = _mapping(label.get("source"))
        oracle = _mapping(label.get("oracle_label"))
        observation_input = _mapping(policy_state.get("observation_input"))
        try:
            values = decode_observation_input(dict(observation_input))
        except (ValueError, TypeError, zlib.error):
            values = []
        action_values = _mapping(label.get("action_value_targets")).get("actions", [])
        rows.append(
            {
                "branch_id": str(label.get("branch_id", "")),
                "seed": _optional_int(source.get("seed")),
                "action": str(oracle.get("action", "")),
                "values": values,
                "action_values": list(action_values)
                if isinstance(action_values, list)
                else [],
            }
        )
    return rows


def _compact_world_model_rows(
    labels: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for label in labels:
        policy_state = _mapping(label.get("policy_state"))
        source = _mapping(label.get("source"))
        oracle = _mapping(label.get("oracle_label"))
        observation_input = _mapping(policy_state.get("observation_input"))
        compact_state: dict[str, object] = {}
        values: Sequence[float] = ()
        try:
            values = decode_observation_input(dict(observation_input))
            compact_state = _compact_state_from_values(values)
        except (ValueError, TypeError, zlib.error):
            compact_state = {}
        action_values = _mapping(label.get("action_value_targets")).get("actions", [])
        rows.append(
            {
                "branch_id": str(label.get("branch_id", "")),
                "seed": _optional_int(source.get("seed")),
                "action": str(oracle.get("action", "")),
                "material_oracle_gain": bool(oracle.get("material_oracle_gain")),
                "action_mask": _action_mask(policy_state),
                "compact_state": compact_state,
                "observation_values": tuple(_round(float(value)) for value in values)
                if compact_state
                else (),
                "public_history_trace": list(
                    _list_of_mappings(
                        policy_state.get("public_history_trace"),
                        field="public_history_trace",
                    )
                ),
                "action_values": list(action_values)
                if isinstance(action_values, list)
                else [],
            }
        )
    return rows


def _compact_state_from_values(values: Sequence[float]) -> dict[str, object]:
    expected_size = _NAVIGATION_INPUT_START + len(NAVIGATION_TARGETS) * _NAVIGATION_STRIDE
    if len(values) != expected_size:
        return {}
    cells = [_patch_cell_from_values(values, index) for index in range(PATCH_CELL_COUNT)]
    center = next(
        (
            cell
            for cell in cells
            if _int(cell.get("dx")) == 0 and _int(cell.get("dy")) == 0
        ),
        cells[_CENTER_PATCH_INDEX],
    )
    adjacent = {
        _cell_direction(cell): cell
        for cell in cells
        if abs(_int(cell.get("dx"))) + abs(_int(cell.get("dy"))) == 1
    }
    local = _local_summaries(cells)
    navigation = {
        target: _navigation_state(values, target_index)
        for target_index, target in enumerate(NAVIGATION_TARGETS)
    }
    self_state = {
        field: _round(float(values[index]))
        for field, index in _SELF_FIELD_INDEX.items()
    }
    return {
        "self": self_state,
        "center": center,
        "adjacent": adjacent,
        "local": local,
        "navigation": navigation,
    }


def _patch_cell_from_values(
    values: Sequence[float],
    cell_index: int,
) -> dict[str, object]:
    base = _PATCH_INPUT_START + cell_index * _PATCH_STRIDE
    return {
        "dx": _round_int(float(values[base + _PATCH_FIELD_INDEX["dx"]]) * LOCAL_PATCH_RADIUS),
        "dy": _round_int(float(values[base + _PATCH_FIELD_INDEX["dy"]]) * LOCAL_PATCH_RADIUS),
        "in_bounds": float(values[base + _PATCH_FIELD_INDEX["in_bounds"]]),
        "terrain_code": float(values[base + _PATCH_FIELD_INDEX["terrain_code"]]),
        "occupant_code": float(values[base + _PATCH_FIELD_INDEX["occupant_code"]]),
        "same_lineage": float(values[base + _PATCH_FIELD_INDEX["same_lineage"]]),
        "water": max(
            float(values[base + _PATCH_FIELD_INDEX["water_access_reason_code"]]),
            1.0
            if float(values[base + _PATCH_FIELD_INDEX["terrain_code"]]) >= 0.99
            else 0.0,
        ),
        "food": float(values[base + _PATCH_FIELD_INDEX["food"]]),
        "vegetation": float(values[base + _PATCH_FIELD_INDEX["vegetation"]]),
        "recovery_debt": float(values[base + _PATCH_FIELD_INDEX["recovery_debt"]]),
        "fresh_kill": float(values[base + _PATCH_FIELD_INDEX["fresh_kill_energy"]]),
        "carcass": float(values[base + _PATCH_FIELD_INDEX["carcass_energy"]]),
        "hazard_type_code": float(values[base + _PATCH_FIELD_INDEX["hazard_type_code"]]),
        "hazard_level": float(values[base + _PATCH_FIELD_INDEX["hazard_level"]]),
        "prey_biomass": float(values[base + _PATCH_FIELD_INDEX["prey_biomass"]]),
        "carrion_signal": float(values[base + _PATCH_FIELD_INDEX["carrion_signal"]]),
        "predator_risk": float(values[base + _PATCH_FIELD_INDEX["predator_risk"]]),
    }


def _navigation_state(
    values: Sequence[float],
    target_index: int,
) -> dict[str, object]:
    base = _NAVIGATION_INPUT_START + target_index * _NAVIGATION_STRIDE
    return {
        "dx": float(values[base + _NAVIGATION_FIELD_INDEX["dx"]]),
        "dy": float(values[base + _NAVIGATION_FIELD_INDEX["dy"]]),
        "distance": float(values[base + _NAVIGATION_FIELD_INDEX["distance"]]),
        "strength": float(values[base + _NAVIGATION_FIELD_INDEX["strength"]]),
    }


def _local_summaries(
    cells: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    return {
        "radius1_food": _max_cell_signal(cells, "food", radius=1),
        "radius1_water": _max_cell_signal(cells, "water", radius=1),
        "radius1_carrion": _max_carrion_signal(cells, radius=1),
        "radius1_prey": _max_cell_signal(cells, "prey_biomass", radius=1),
        "radius1_risk": _max_risk_signal(cells, radius=1),
        "radius2_food": _max_cell_signal(cells, "food", radius=2),
        "radius2_water": _max_cell_signal(cells, "water", radius=2),
        "radius2_carrion": _max_carrion_signal(cells, radius=2),
        "radius2_prey": _max_cell_signal(cells, "prey_biomass", radius=2),
        "radius2_risk": _max_risk_signal(cells, radius=2),
    }


def _compact_action_feature_vector(
    *,
    state: Mapping[str, object],
    action: str,
    action_mask: Mapping[str, object],
) -> tuple[float, ...]:
    self_state = _mapping(state.get("self"))
    center = _mapping(state.get("center"))
    local = _mapping(state.get("local"))
    navigation = _mapping(state.get("navigation"))
    adjacent = _mapping(state.get("adjacent"))
    if not self_state or not center:
        return ()
    energy = _feature_float(self_state.get("energy_ratio"))
    hydration = _feature_float(self_state.get("hydration_ratio"))
    health = _feature_float(self_state.get("health_ratio"))
    target_cell = _target_cell_for_action(action, adjacent)
    action_dx, action_dy = _MOVE_DELTAS.get(action, (0, 0))
    action_is_move = 1.0 if action in _MOVE_DELTAS else 0.0
    action_is_drink = 1.0 if action == "drink" else 0.0
    action_is_eat = 1.0 if action == "eat" else 0.0
    action_is_stay = 1.0 if action == "stay" else 0.0
    current_carrion = _cell_carrion(center)
    target_carrion = _cell_carrion(target_cell)
    current_risk = _cell_risk(center)
    target_risk = _cell_risk(target_cell)
    movement_alignment = _movement_navigation_alignment(action_dx, action_dy, navigation)
    supported_count = sum(1 for value in action_mask.values() if bool(value))
    return tuple(
        _round(value)
        for value in (
            energy,
            hydration,
            health,
            _feature_float(self_state.get("injury_load")),
            _feature_float(self_state.get("age_norm")),
            _feature_float(self_state.get("matched_diet_ratio")),
            _feature_float(self_state.get("trophic_role_code")),
            _feature_float(self_state.get("meat_mode_code")),
            _feature_float(self_state.get("season_code")),
            _feature_float(self_state.get("water_access_reason_code")),
            _feature_float(self_state.get("hazard_level")),
            _feature_float(self_state.get("tile_vegetation")),
            _feature_float(self_state.get("tile_recovery_debt")),
            1.0 - energy,
            1.0 - hydration,
            1.0 - health,
            min(energy, hydration, health),
            _feature_float(center.get("water")),
            _feature_float(center.get("food")),
            current_carrion,
            _feature_float(center.get("prey_biomass")),
            current_risk,
            _feature_float(local.get("radius1_food")),
            _feature_float(local.get("radius1_water")),
            _feature_float(local.get("radius1_carrion")),
            _feature_float(local.get("radius1_prey")),
            _feature_float(local.get("radius1_risk")),
            _feature_float(local.get("radius2_food")),
            _feature_float(local.get("radius2_water")),
            _feature_float(local.get("radius2_carrion")),
            _feature_float(local.get("radius2_prey")),
            _feature_float(local.get("radius2_risk")),
            *_navigation_features(navigation),
            action_is_drink,
            action_is_eat,
            action_is_stay,
            action_is_move,
            _round(action_dx / 1.0),
            _round(action_dy / 1.0),
            1.0 if bool(action_mask.get(action, False)) else 0.0,
            min(float(supported_count) / max(float(len(ACTION_NAMES)), 1.0), 1.0),
            action_is_drink * (1.0 - hydration),
            action_is_drink
            * max(
                _feature_float(center.get("water")),
                _feature_float(local.get("radius1_water")),
            ),
            action_is_eat * (1.0 - energy),
            action_is_eat * max(_feature_float(center.get("food")), current_carrion),
            action_is_stay * max(_feature_float(center.get("food")), current_carrion),
            action_is_stay * _feature_float(center.get("water")),
            action_is_stay * current_risk,
            action_is_move * _feature_float(target_cell.get("in_bounds")),
            action_is_move * _feature_float(target_cell.get("water")),
            action_is_move * _feature_float(target_cell.get("food")),
            action_is_move * target_carrion,
            action_is_move * _feature_float(target_cell.get("prey_biomass")),
            action_is_move * target_risk,
            action_is_move * (_feature_float(target_cell.get("food")) - _feature_float(center.get("food"))),
            action_is_move * (target_carrion - current_carrion),
            action_is_move * (target_risk - current_risk),
            *movement_alignment,
        )
    )


def _policy_observation_action_feature_vector(
    *,
    observation_values: Sequence[float],
    action: str,
    action_mask: Mapping[str, object],
) -> tuple[float, ...]:
    if action not in ACTION_NAMES:
        return ()
    action_index = ACTION_NAMES.index(action)
    action_one_hot = [
        1.0 if index == action_index else 0.0
        for index, _ in enumerate(ACTION_NAMES)
    ]
    mask_values = [
        1.0 if bool(action_mask.get(name, False)) else 0.0
        for name in ACTION_NAMES
    ]
    return tuple(
        _round(value)
        for value in (
            *observation_values,
            *mask_values,
            *action_one_hot,
            1.0 if bool(action_mask.get(action, False)) else 0.0,
        )
    )


def _public_history_feature_vector(
    history: Sequence[Mapping[str, object]],
) -> tuple[float, ...]:
    selected = list(history)[-DEFAULT_BRANCH_ACTION_ORACLE_HISTORY_STEPS:]
    padded: list[Mapping[str, object] | None] = [None] * (
        DEFAULT_BRANCH_ACTION_ORACLE_HISTORY_STEPS - len(selected)
    )
    padded.extend(selected)
    values: list[float] = []
    empty_slot = (0.0,) * len(_public_history_item_features({}))
    for item in padded:
        if item is None:
            values.extend(empty_slot)
            continue
        values.extend(_public_history_item_features(item))
    return tuple(_round(value) for value in values)


def _public_history_item_features(item: Mapping[str, object]) -> tuple[float, ...]:
    return tuple(
        _round(value)
        for value in (
            1.0,
            min(_feature_float(item.get("tick_delta")) / 120.0, 1.0),
            min(_feature_float(item.get("record_index_delta")) / 256.0, 1.0),
            *_action_one_hot(_optional_string(item.get("requested_action"))),
            *_action_one_hot(_optional_string(item.get("resolved_action"))),
            1.0 if bool(item.get("action_valid", False)) else 0.0,
            1.0 if bool(item.get("resolution_action_valid", False)) else 0.0,
            1.0 if bool(item.get("moved", False)) else 0.0,
            1.0 if bool(item.get("drank", False)) else 0.0,
            1.0 if bool(item.get("ate", False)) else 0.0,
            1.0 if bool(item.get("died", False)) else 0.0,
            1.0 if bool(item.get("died_after_action", False)) else 0.0,
            1.0 if bool(item.get("post_carrion_contact", False)) else 0.0,
            _clamped_delta(item.get("x_delta")),
            _clamped_delta(item.get("y_delta")),
            _feature_float(item.get("resource_gain")),
            _feature_float(item.get("energy_ratio_after")),
            _clamped_delta(item.get("energy_ratio_delta")),
            _feature_float(item.get("hydration_ratio_after")),
            _clamped_delta(item.get("hydration_ratio_delta")),
            _feature_float(item.get("health_ratio_after")),
            _clamped_delta(item.get("health_ratio_delta")),
            min(
                _feature_float(item.get("ticks_since_animal_resource_gain")) / 32.0,
                1.0,
            ),
            min(_feature_float(item.get("ticks_since_drink")) / 32.0, 1.0),
        )
    )


def _action_one_hot(action: str | None) -> tuple[float, ...]:
    return tuple(1.0 if action == name else 0.0 for name in ACTION_NAMES)


def _observation_values(row: Mapping[str, object]) -> tuple[float, ...]:
    values = row.get("observation_values")
    if not isinstance(values, tuple):
        return ()
    return values


def _navigation_features(navigation: Mapping[str, object]) -> tuple[float, ...]:
    features: list[float] = []
    for target in NAVIGATION_TARGETS:
        item = _mapping(navigation.get(target))
        features.extend(
            [
                _feature_float(item.get("dx")),
                _feature_float(item.get("dy")),
                _feature_float(item.get("distance")),
                _feature_float(item.get("strength")),
            ]
        )
    return tuple(features)


def _movement_navigation_alignment(
    action_dx: int,
    action_dy: int,
    navigation: Mapping[str, object],
) -> tuple[float, ...]:
    if action_dx == 0 and action_dy == 0:
        return tuple(0.0 for _ in NAVIGATION_TARGETS)
    values = []
    for target in NAVIGATION_TARGETS:
        item = _mapping(navigation.get(target))
        dx = _feature_float(item.get("dx"))
        dy = _feature_float(item.get("dy"))
        strength = _feature_float(item.get("strength"))
        aligned = max(0.0, action_dx * dx + action_dy * dy)
        values.append(_round(aligned * strength))
    return tuple(values)


def _target_cell_for_action(
    action: str,
    adjacent: Mapping[str, object],
) -> Mapping[str, object]:
    direction = {
        "move_north": "north",
        "move_south": "south",
        "move_east": "east",
        "move_west": "west",
    }.get(action)
    if direction is None:
        return {}
    return _mapping(adjacent.get(direction))


def _cell_direction(cell: Mapping[str, object]) -> str:
    dx = _int(cell.get("dx"))
    dy = _int(cell.get("dy"))
    if dx == 0 and dy < 0:
        return "north"
    if dx == 0 and dy > 0:
        return "south"
    if dx > 0 and dy == 0:
        return "east"
    if dx < 0 and dy == 0:
        return "west"
    return "center"


def _max_cell_signal(
    cells: Sequence[Mapping[str, object]],
    field: str,
    *,
    radius: int,
) -> float:
    values = [
        _feature_float(cell.get(field))
        for cell in cells
        if abs(_int(cell.get("dx"))) + abs(_int(cell.get("dy"))) <= radius
    ]
    return max(values) if values else 0.0


def _max_carrion_signal(
    cells: Sequence[Mapping[str, object]],
    *,
    radius: int,
) -> float:
    values = [
        _cell_carrion(cell)
        for cell in cells
        if abs(_int(cell.get("dx"))) + abs(_int(cell.get("dy"))) <= radius
    ]
    return max(values) if values else 0.0


def _max_risk_signal(
    cells: Sequence[Mapping[str, object]],
    *,
    radius: int,
) -> float:
    values = [
        _cell_risk(cell)
        for cell in cells
        if abs(_int(cell.get("dx"))) + abs(_int(cell.get("dy"))) <= radius
    ]
    return max(values) if values else 0.0


def _cell_carrion(cell: Mapping[str, object]) -> float:
    return max(
        _feature_float(cell.get("fresh_kill")),
        _feature_float(cell.get("carcass")),
        _feature_float(cell.get("carrion_signal")),
    )


def _cell_risk(cell: Mapping[str, object]) -> float:
    return max(
        _feature_float(cell.get("hazard_level")),
        _feature_float(cell.get("predator_risk")),
    )


def _feature_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _clamped_delta(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(-1.0, min(1.0, float(value)))


def _round_int(value: float) -> int:
    return int(round(float(value)))



def _acceptance(
    aggregate: Mapping[str, object],
    *,
    min_material_label_count: int,
    max_dominant_oracle_action_share: float,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    if _int(aggregate.get("label_count")) <= 0:
        blockers.append({"reason": "no_branch_action_oracle_labels"})
    if _int(aggregate.get("material_oracle_gain_label_count")) < min_material_label_count:
        blockers.append(
            {
                "reason": "material_oracle_gain_label_count_below_floor",
                "required": int(min_material_label_count),
                "observed": _int(
                    aggregate.get("material_oracle_gain_label_count")
                ),
            }
        )
    dominant_share = _float(aggregate.get("dominant_oracle_action_share"))
    if dominant_share > max_dominant_oracle_action_share:
        blockers.append(
            {
                "reason": "dominant_oracle_action_share_above_cap",
                "required_max": _round(max_dominant_oracle_action_share),
                "observed": _round(dominant_share),
            }
        )
    if aggregate.get("replay_verified_all_labels") is not True:
        blockers.append({"reason": "not_all_labels_replay_verified"})
    if aggregate.get("zero_heuristic_all_labels") is not True:
        blockers.append({"reason": "labels_include_heuristic_runtime_actions"})
    if aggregate.get("policy_state_complete_all_labels") is not True:
        blockers.append({"reason": "labels_missing_policy_visible_state"})
    if _int(aggregate.get("unsupported_oracle_action_count")) != 0:
        blockers.append(
            {
                "reason": "oracle_action_not_supported_by_action_mask",
                "observed": _int(aggregate.get("unsupported_oracle_action_count")),
            }
        )
    if _int(aggregate.get("conflicting_observation_digest_count")) != 0:
        blockers.append(
            {
                "reason": "conflicting_oracle_actions_for_observation_digest",
                "observed": _int(
                    aggregate.get("conflicting_observation_digest_count")
                ),
            }
        )
    return {
        "label_archive_acceptance_passed": not blockers,
        "materially_supports_oracle_distillation": not blockers,
        "blockers": blockers,
    }


def _policy_state_complete(policy_state: Mapping[str, object]) -> bool:
    observation_input = _mapping(policy_state.get("observation_input"))
    action_mask = _mapping(policy_state.get("action_mask"))
    return (
        observation_input.get("schema_version") == "mind_observation_v3"
        and _optional_string(policy_state.get("observation_digest")) is not None
        and set(action_mask) == set(ACTION_NAMES)
    )


def _all_action_runs_replay_verified(result: Mapping[str, object]) -> bool:
    runs = _list_of_mappings(result.get("action_runs"), field="action_runs")
    return bool(runs) and all(_run_replay_verified(run) for run in runs)


def _all_action_runs_zero_heuristic(result: Mapping[str, object]) -> bool:
    runs = _list_of_mappings(result.get("action_runs"), field="action_runs")
    return bool(runs) and all(
        _int(run.get("heuristic_action_source_count")) == 0 for run in runs
    )


def _all_action_runs_forced_action_used(result: Mapping[str, object]) -> bool:
    runs = _list_of_mappings(result.get("action_runs"), field="action_runs")
    return bool(runs) and all(bool(run.get("forced_action_used")) for run in runs)


def _run_replay_verified(run: Mapping[str, object]) -> bool:
    verification = _mapping(run.get("replay_verification"))
    return verification.get("verified") is True


def _run_objective(
    run: Mapping[str, object],
) -> tuple[float, float, float, float, float, str]:
    return (
        float(_int(run.get("alive_agents"))),
        float(_int(run.get("births"))),
        float(int(bool(run.get("target_alive_at_end")))),
        -float(_int(run.get("deaths"))),
        -_float(run.get("dominant_requested_action_share")),
        str(run.get("forced_action", "")),
    )


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _objective_tuple(value: object) -> tuple[float, float, float, float, float]:
    if not isinstance(value, list):
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    items = [
        float(item)
        for item in value[:5]
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    ]
    while len(items) < 5:
        items.append(0.0)
    return tuple(items[:5])  # type: ignore[return-value]


def _first_action_outcome(run: Mapping[str, object]) -> dict[str, object]:
    outcome = _mapping(run.get("first_action_outcome"))
    if not outcome:
        return {"record_found": False}
    return {
        "record_found": bool(outcome.get("record_found", False)),
        "requested_action": _optional_string(outcome.get("requested_action")),
        "resolved_action": _optional_string(outcome.get("resolved_action")),
        "action_valid": bool(outcome.get("action_valid", False)),
        "resolution_action_valid": bool(
            outcome.get("resolution_action_valid", False)
        ),
        "moved": bool(outcome.get("moved", False)),
        "alive_before": bool(outcome.get("alive_before", False)),
        "alive_after": bool(outcome.get("alive_after", False)),
        "x_delta": _int(outcome.get("x_delta")),
        "y_delta": _int(outcome.get("y_delta")),
        "energy_ratio_before": _optional_float(
            outcome.get("energy_ratio_before")
        ),
        "energy_ratio_after": _optional_float(outcome.get("energy_ratio_after")),
        "energy_ratio_delta": _optional_float(
            outcome.get("energy_ratio_delta")
        ),
        "hydration_ratio_before": _optional_float(
            outcome.get("hydration_ratio_before")
        ),
        "hydration_ratio_after": _optional_float(
            outcome.get("hydration_ratio_after")
        ),
        "hydration_ratio_delta": _optional_float(
            outcome.get("hydration_ratio_delta")
        ),
        "health_ratio_before": _optional_float(
            outcome.get("health_ratio_before")
        ),
        "health_ratio_after": _optional_float(outcome.get("health_ratio_after")),
        "health_ratio_delta": _optional_float(
            outcome.get("health_ratio_delta")
        ),
        "resource_gain": _optional_float(outcome.get("resource_gain")),
        "drank": bool(outcome.get("drank", False)),
        "ate": bool(outcome.get("ate", False)),
        "died": bool(outcome.get("died", False)),
        "death_cause": _optional_string(outcome.get("death_cause")),
        "died_after_action": bool(outcome.get("died_after_action", False)),
    }


def _target_horizon_trace(run: Mapping[str, object]) -> list[dict[str, object]]:
    trace = run.get("target_horizon_trace")
    if not isinstance(trace, list):
        return []
    return [_horizon_outcome(item) for item in trace if isinstance(item, Mapping)]


def _population_horizon_trace(run: Mapping[str, object]) -> list[dict[str, object]]:
    trace = run.get("population_horizon_trace")
    if not isinstance(trace, list):
        return []
    return [
        _population_horizon_item(item)
        for item in trace
        if isinstance(item, Mapping)
    ]


def _population_horizon_item(item: Mapping[str, object]) -> dict[str, object]:
    action_counts = _mapping(item.get("tick_requested_action_counts"))
    return {
        "horizon_tick_delta": _int(item.get("horizon_tick_delta")),
        "tick": _int(item.get("tick")),
        "alive_agents": _int(item.get("alive_agents")),
        "births": _int(item.get("births")),
        "deaths": _int(item.get("deaths")),
        "target_alive": bool(item.get("target_alive", False)),
        "target_energy_ratio": _optional_float(item.get("target_energy_ratio")),
        "target_hydration_ratio": _optional_float(
            item.get("target_hydration_ratio")
        ),
        "target_health_ratio": _optional_float(item.get("target_health_ratio")),
        "tick_resource_gain": _optional_float(item.get("tick_resource_gain")),
        "tick_trajectory_record_count": _int(
            item.get("tick_trajectory_record_count")
        ),
        "tick_requested_action_counts": {
            str(action): _int(count)
            for action, count in sorted(action_counts.items())
        },
        "tick_dominant_requested_action": _optional_string(
            item.get("tick_dominant_requested_action")
        ),
        "tick_dominant_requested_action_share": _optional_float(
            item.get("tick_dominant_requested_action_share")
        ),
    }


def _horizon_outcome(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "record_found": bool(item.get("record_found", False)),
        "horizon_tick_delta": _int(item.get("horizon_tick_delta")),
        "tick": _int(item.get("tick")),
        "requested_action": _optional_string(item.get("requested_action")),
        "resolved_action": _optional_string(item.get("resolved_action")),
        "action_valid": bool(item.get("action_valid", False)),
        "resolution_action_valid": bool(
            item.get("resolution_action_valid", False)
        ),
        "moved": bool(item.get("moved", False)),
        "alive_before": bool(item.get("alive_before", False)),
        "alive_after": bool(item.get("alive_after", False)),
        "x_delta": _int(item.get("x_delta")),
        "y_delta": _int(item.get("y_delta")),
        "energy_ratio_before": _optional_float(item.get("energy_ratio_before")),
        "energy_ratio_after": _optional_float(item.get("energy_ratio_after")),
        "energy_ratio_delta": _optional_float(item.get("energy_ratio_delta")),
        "hydration_ratio_before": _optional_float(
            item.get("hydration_ratio_before")
        ),
        "hydration_ratio_after": _optional_float(item.get("hydration_ratio_after")),
        "hydration_ratio_delta": _optional_float(
            item.get("hydration_ratio_delta")
        ),
        "health_ratio_before": _optional_float(item.get("health_ratio_before")),
        "health_ratio_after": _optional_float(item.get("health_ratio_after")),
        "health_ratio_delta": _optional_float(item.get("health_ratio_delta")),
        "resource_gain": _optional_float(item.get("resource_gain")),
        "drank": bool(item.get("drank", False)),
        "ate": bool(item.get("ate", False)),
        "died": bool(item.get("died", False)),
        "death_cause": _optional_string(item.get("death_cause")),
        "died_after_action": bool(item.get("died_after_action", False)),
    }


def _mean_objective_tuple(
    values: Sequence[tuple[float, float, float, float, float]],
) -> tuple[float, float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    count = float(len(values))
    return tuple(sum(value[index] for value in values) / count for index in range(5))


def _dominant_count(counts: Counter[str]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    action, count = max(
        sorted(counts.items()),
        key=lambda item: (int(item[1]), str(item[0])),
    )
    return str(action), int(count)


def _action_option_mode(action: str) -> str:
    if action == "drink":
        return "recover_hydration"
    if action == "eat":
        return "exploit_resource"
    if action == "stay":
        return "conserve"
    if action in _MOVE_DELTAS:
        return "reposition"
    return "other"


def _list_of_mappings(value: object, *, field: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise BranchActionOracleLabelError(f"{field} must be a list")
    items: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise BranchActionOracleLabelError(f"{field} entries must be objects")
        items.append(item)
    return items


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return _round(float(value))


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _safe_rate(count: int, total: int) -> float:
    return _round(count / float(total)) if total else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
