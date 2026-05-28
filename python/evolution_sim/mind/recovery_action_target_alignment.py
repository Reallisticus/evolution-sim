from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES, MOVEMENT_ACTIONS
from evolution_sim.env.runtime.observations import (
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
)
from evolution_sim.mind.carrion_recovery_archive import (
    MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_recovery_distill import (
    MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_recovery_residual_audit import (
    MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.dataset import TRAJECTORY_DATASET_RECORD_INDEX_FIELD
from evolution_sim.mind.evolution import score_mind_v3_metadata
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.rollout_context import RolloutContextConfig, RolloutContextState
from evolution_sim.mind.v3_neural import (
    MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
    score_mind_v3_neural_artifact,
    validate_mind_v3_neural_artifact,
)
from evolution_sim.mind.v3_policy import (
    MindV3EvolutionPolicy,
    _best_action,
    _blend_neural_with_linear_anchor,
    _neural_residual_context_gate_status,
    _neural_residual_safety_guard_reason,
)

MIND_V3_RECOVERY_ACTION_TARGET_ALIGNMENT_SCHEMA_VERSION = (
    "mind_v3_recovery_action_target_alignment_audit_v1"
)
MIND_V3_RECOVERY_ACTION_TARGET_ALIGNMENT_POLICY = (
    "diagnostics_only_recovery_residual_action_target_alignment_v1"
)

WINDOW_NAMES: tuple[str, ...] = (
    "first_record",
    "all_records",
    "post_carrion",
    "recovery_phase",
    "drink_available",
    "low_hydration_after_carrion",
    "water_directed_move",
    "conserve_or_stay_candidate",
    "eat_available_non_eat_logged",
)
ACTION_SURFACES: tuple[tuple[str, str], ...] = (
    ("logged", "logged_action"),
    ("neural_top", "neural_top_action"),
    ("linear_top", "linear_top_action"),
    ("configured", "configured_blended_action"),
    ("margin_ignored", "margin_ignored_action"),
    ("forced_gate", "forced_gate_action"),
)
SHADOW_SURFACES: tuple[tuple[str, str], ...] = (
    ("configured", "configured_blended_action"),
    ("margin_ignored", "margin_ignored_action"),
    ("forced_gate", "forced_gate_action"),
)
ALIGNMENT_KEYS: tuple[str, ...] = (
    "drink",
    "water_directed_movement",
    "appropriate_stay_conserve",
    "avoid_extra_eat",
    "useful_recovery_action_count",
)
LOW_HYDRATION_AFTER_CARRION_THRESHOLD = 0.7
MIN_INFORMATIVE_ORACLE_MATCHES = 5


class RecoveryActionTargetAlignmentError(ValueError):
    pass


def load_recovery_action_target_alignment_json(
    path: str | Path,
) -> dict[str, object]:
    with _open_input(Path(path)) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RecoveryActionTargetAlignmentError(
            f"report must be a JSON object: {path}"
        )
    return payload


def build_recovery_action_target_alignment_report(
    *,
    artifact: Mapping[str, object] | None = None,
    artifact_path: str | Path | None = None,
    distill_report: Mapping[str, object] | None = None,
    distill_report_path: str | Path | None = None,
    evaluation_report: Mapping[str, object] | None = None,
    evaluation_report_path: str | Path | None = None,
    split_report: Mapping[str, object] | None = None,
    split_report_path: str | Path | None = None,
    residual_audit: Mapping[str, object] | None = None,
    residual_audit_path: str | Path | None = None,
    oracle_audit: Mapping[str, object] | None = None,
    oracle_audit_path: str | Path | None = None,
) -> dict[str, object]:
    if artifact is None:
        if artifact_path is None:
            raise RecoveryActionTargetAlignmentError("artifact_path is required")
        artifact = load_recovery_action_target_alignment_json(artifact_path)
    validate_mind_v3_neural_artifact(artifact)
    if artifact.get("schema_version") != MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION:
        raise RecoveryActionTargetAlignmentError("artifact has stale schema_version")

    if distill_report is None:
        if distill_report_path is None:
            raise RecoveryActionTargetAlignmentError("distill_report_path is required")
        distill_report = load_recovery_action_target_alignment_json(distill_report_path)
    if (
        distill_report.get("schema_version")
        != MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION
    ):
        raise RecoveryActionTargetAlignmentError(
            "distill report has stale schema_version"
        )

    if evaluation_report is None:
        if evaluation_report_path is None:
            raise RecoveryActionTargetAlignmentError(
                "evaluation_report_path is required"
            )
        evaluation_report = load_recovery_action_target_alignment_json(
            evaluation_report_path
        )

    if split_report is None:
        if split_report_path is None:
            raise RecoveryActionTargetAlignmentError("split_report_path is required")
        split_report = load_recovery_action_target_alignment_json(split_report_path)
    if (
        split_report.get("schema_version")
        != MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION
    ):
        raise RecoveryActionTargetAlignmentError("split report has stale schema_version")

    if residual_audit is None:
        if residual_audit_path is None:
            raise RecoveryActionTargetAlignmentError("residual_audit_path is required")
        residual_audit = load_recovery_action_target_alignment_json(
            residual_audit_path
        )
    if (
        residual_audit.get("schema_version")
        != MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_SCHEMA_VERSION
    ):
        raise RecoveryActionTargetAlignmentError(
            "residual audit has stale schema_version"
        )

    oracle_load_error = None
    if oracle_audit is None and oracle_audit_path is not None:
        try:
            oracle_audit = load_recovery_action_target_alignment_json(
                oracle_audit_path
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            oracle_load_error = str(exc)

    heldout_rows = _split_rows(split_report, split="heldout")
    scoring = _score_heldout_split_rows(
        artifact=artifact,
        heldout_rows=heldout_rows,
    )
    scored_rows = scoring["decision_rows"]
    assert isinstance(scored_rows, list)
    window_summaries = _window_summaries(scored_rows)
    oracle_comparison = _oracle_comparison(
        scored_rows,
        oracle_audit=oracle_audit,
        oracle_audit_path=oracle_audit_path,
        oracle_load_error=oracle_load_error,
    )
    classification = _classification(
        window_summaries=window_summaries,
        oracle_comparison=oracle_comparison,
        residual_audit=residual_audit,
    )

    contract = {
        "schema_version": MIND_V3_RECOVERY_ACTION_TARGET_ALIGNMENT_SCHEMA_VERSION,
        "policy": MIND_V3_RECOVERY_ACTION_TARGET_ALIGNMENT_POLICY,
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "artifact_promotion_effect": "none",
        "action_mask_effect": "none",
        "fixture_floor_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "heldout_sampling_policy": "all_records_per_heldout_split_trajectory_v1",
        "policy_context_order": "decide_current_row_then_observe_transition_v1",
        "water_directed_movement_policy": (
            "policy_visible_observation_input_navigation_water_dx_dy_plus_"
            "action_mask_and_action_v1"
        ),
        "low_hydration_after_carrion_threshold": (
            LOW_HYDRATION_AFTER_CARRION_THRESHOLD
        ),
    }
    return {
        "schema_version": MIND_V3_RECOVERY_ACTION_TARGET_ALIGNMENT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_RECOVERY_ACTION_TARGET_ALIGNMENT_POLICY,
        "contract": contract,
        "provenance": {
            "contract_digest": stable_payload_digest(contract),
            "artifact_path": str(artifact_path) if artifact_path is not None else None,
            "artifact_digest": stable_payload_digest(artifact),
            "distill_report_path": (
                str(distill_report_path) if distill_report_path is not None else None
            ),
            "distill_report_digest": stable_payload_digest(distill_report),
            "evaluation_report_path": (
                str(evaluation_report_path)
                if evaluation_report_path is not None
                else None
            ),
            "evaluation_report_digest": stable_payload_digest(evaluation_report),
            "split_report_path": (
                str(split_report_path) if split_report_path is not None else None
            ),
            "split_report_digest": stable_payload_digest(split_report),
            "residual_audit_path": (
                str(residual_audit_path) if residual_audit_path is not None else None
            ),
            "residual_audit_digest": stable_payload_digest(residual_audit),
            "oracle_audit_path": (
                str(oracle_audit_path) if oracle_audit_path is not None else None
            ),
            "oracle_audit_digest": (
                stable_payload_digest(oracle_audit)
                if isinstance(oracle_audit, Mapping)
                else None
            ),
        },
        "artifact_config": _artifact_config(artifact),
        "source_report_summary": {
            "distill_acceptance": _mapping(distill_report.get("acceptance")),
            "evaluation_keys": sorted(str(key) for key in evaluation_report.keys()),
            "split_aggregate": _mapping(split_report.get("aggregate")),
            "residual_failure_classification": _mapping(
                residual_audit.get("failure_classification")
            ),
            "residual_calibration_classification": _mapping(
                residual_audit.get("calibration_classification")
            ),
        },
        "heldout_scoring": {
            "heldout_split_row_count": len(heldout_rows),
            "decision_row_count": len(scored_rows),
            "load_failure_count": len(scoring["load_failures"]),
            "malformed_record_count": int(scoring["malformed_record_count"]),
            "split_rows": scoring["split_rows"],
            "load_failures": scoring["load_failures"],
        },
        "windows": window_summaries,
        "oracle_comparison": oracle_comparison,
        "classification": classification,
        "decision_rows": scored_rows,
        "non_promoted": True,
    }


def write_recovery_action_target_alignment_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _artifact_config(artifact: Mapping[str, object]) -> dict[str, object]:
    contract = _mapping(artifact.get("input_contract"))
    return {
        "schema_version": artifact.get("schema_version"),
        "artifact_mode": artifact.get("artifact_mode"),
        "model_type": artifact.get("model_type"),
        "input_policy": artifact.get("input_policy"),
        "input_contract_digest": _mapping(artifact.get("provenance")).get(
            "input_contract_digest"
        ),
        "ecological_vector_size": contract.get("ecological_vector_size"),
        "neural_residual_scale": artifact.get("neural_residual_scale"),
        "neural_residual_max_linear_override_margin": artifact.get(
            "neural_residual_max_linear_override_margin"
        ),
        "neural_residual_context_gate": artifact.get("neural_residual_context_gate"),
        "neural_residual_recovery_phase_ticks": artifact.get(
            "neural_residual_recovery_phase_ticks"
        ),
        "trained_record_count": artifact.get("trained_record_count"),
        "hidden_units": artifact.get("hidden_units"),
    }


def _score_heldout_split_rows(
    *,
    artifact: Mapping[str, object],
    heldout_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    scored_rows: list[dict[str, object]] = []
    split_summaries: list[dict[str, object]] = []
    load_failures: list[dict[str, object]] = []
    malformed_count = 0
    context_config = RolloutContextConfig(
        recovery_phase_ticks=max(
            1,
            _int(artifact.get("neural_residual_recovery_phase_ticks")),
        )
    )
    for split_index, split_row in enumerate(heldout_rows):
        path = split_row.get("trajectory_path")
        if not isinstance(path, str) or not path:
            load_failures.append(
                {
                    "split_record_index": split_index,
                    "record_id": split_row.get("record_id"),
                    "reason": "missing_trajectory_path",
                }
            )
            continue
        try:
            _header, records = _load_trajectory_payload(path)
        except Exception as exc:
            load_failures.append(
                {
                    "split_record_index": split_index,
                    "record_id": split_row.get("record_id"),
                    "trajectory_path": path,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        policy = MindV3EvolutionPolicy(
            seed=_int(split_row.get("seed")),
            neural_artifact=dict(artifact),
        )
        context_states: dict[int, RolloutContextState] = {}
        start_count = len(scored_rows)
        window_counts: Counter[str] = Counter()
        action_source_counts: Counter[str] = Counter()
        logged_action_counts: Counter[str] = Counter()
        for record_index, record in enumerate(records):
            agent_id = _int(record.get("agent_id"))
            context_state = context_states.setdefault(
                agent_id,
                RolloutContextState(context_config),
            )
            context_snapshot = context_state.snapshot()
            scored = _score_decision_record(
                record,
                artifact=artifact,
                policy=policy,
                split_row=split_row,
                split_record_index=split_index,
                trajectory_record_position=record_index,
                context_snapshot=context_snapshot,
            )
            if scored is None:
                malformed_count += 1
            else:
                scored_rows.append(scored)
                for window in scored["windows"]:
                    window_counts[str(window)] += 1
                action_source_counts[str(scored["action_source"])] += 1
                logged_action_counts[str(scored["logged_action"])] += 1
            policy.observe_transition(dict(record))
            context_state.update_from_record(record)
        split_summaries.append(
            {
                "split": "heldout",
                "split_record_index": split_index,
                "record_id": split_row.get("record_id"),
                "seed": split_row.get("seed"),
                "branch_id": split_row.get("branch_id"),
                "branch_state_digest": split_row.get("branch_state_digest"),
                "branch_tick": split_row.get("branch_tick"),
                "continuation_script": split_row.get("continuation_script"),
                "terminal_survivor": bool(split_row.get("terminal_survivor")),
                "outcome_class": _outcome_class(split_row),
                "trajectory_path": path,
                "trajectory_record_count": len(records),
                "scored_decision_count": len(scored_rows) - start_count,
                "action_source_counts": dict(sorted(action_source_counts.items())),
                "logged_action_counts": dict(sorted(logged_action_counts.items())),
                "window_row_counts": dict(sorted(window_counts.items())),
            }
        )
    return {
        "decision_rows": scored_rows,
        "split_rows": split_summaries,
        "load_failures": load_failures,
        "malformed_record_count": malformed_count,
    }


def _score_decision_record(
    record: Mapping[str, object],
    *,
    artifact: Mapping[str, object],
    policy: MindV3EvolutionPolicy,
    split_row: Mapping[str, object],
    split_record_index: int,
    trajectory_record_position: int,
    context_snapshot: Mapping[str, object],
) -> dict[str, object] | None:
    observation_input = record.get("observation_input")
    action_mask = record.get("action_mask")
    if not isinstance(observation_input, Mapping) or not isinstance(action_mask, Mapping):
        return None
    mask = {action: bool(action_mask.get(action, False)) for action in ACTION_NAMES}
    observation = {
        "metadata": record.get("observation_metadata", {}),
        "observation_input": dict(observation_input),
    }
    try:
        decision = policy.decide(observation, mask)
        diagnostics = decision.diagnostics
        agent_id = _int(record.get("agent_id"))
        metadata = policy.agent_mind_metadata(agent_id=agent_id)
        observation_values = decode_observation_input(dict(observation_input))
        recovery_remaining = _int(
            diagnostics.get("neural_residual_recovery_phase_remaining")
        )
        neural_scores = score_mind_v3_neural_artifact(
            artifact=artifact,
            observation_input=dict(observation_input),
            action_mask=mask,
            recovery_phase_remaining=recovery_remaining,
        )
        linear_scores = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=observation_values,
            action_mask=mask,
        )
    except Exception:
        return None

    scale = _float(artifact.get("neural_residual_scale"))
    margin = _float(artifact.get("neural_residual_max_linear_override_margin"))
    gate = str(artifact.get("neural_residual_context_gate", "none"))
    gate_status = _neural_residual_context_gate_status(
        observation_values,
        gate=gate,
        recovery_phase_remaining=recovery_remaining,
    )
    safety_guard = _neural_residual_safety_guard_reason(
        observation_values,
        neural_scores=neural_scores,
        linear_scores=linear_scores,
        action_mask=mask,
    )
    configured_scale = (
        scale if bool(gate_status.get("passed")) and safety_guard is None else 0.0
    )
    forced_gate_scale = scale if safety_guard is None else 0.0
    configured_scores = _blend_neural_with_linear_anchor(
        neural_scores=neural_scores,
        linear_scores=linear_scores,
        action_mask=mask,
        residual_scale=configured_scale,
        max_linear_override_margin=margin,
    )
    margin_ignored_scores = _blend_neural_with_linear_anchor(
        neural_scores=neural_scores,
        linear_scores=linear_scores,
        action_mask=mask,
        residual_scale=configured_scale,
        max_linear_override_margin=float("inf"),
    )
    forced_gate_scores = _blend_neural_with_linear_anchor(
        neural_scores=neural_scores,
        linear_scores=linear_scores,
        action_mask=mask,
        residual_scale=forced_gate_scale,
        max_linear_override_margin=margin,
    )
    linear_action, _linear_score = _best_action(linear_scores, mask)
    neural_action, _neural_score = _best_action(neural_scores, mask)
    configured_action, _configured_score = _best_action(configured_scores, mask)
    margin_ignored_action, _ignored_score = _best_action(margin_ignored_scores, mask)
    forced_gate_action, _forced_score = _best_action(forced_gate_scores, mask)
    logged_action = _logged_action(record)
    water_dx = _navigation_feature(observation_values, "water", "dx")
    water_dy = _navigation_feature(observation_values, "water", "dy")
    water_actions = _water_directed_actions(observation_values, mask)
    hydration = _self_feature(observation_values, "hydration_ratio")
    windows = _window_memberships(
        trajectory_record_position=trajectory_record_position,
        context_snapshot=context_snapshot,
        recovery_phase_remaining=recovery_remaining,
        action_mask=mask,
        logged_action=logged_action,
        observation_values=observation_values,
    )
    rank = _score_rank(neural_scores, logged_action)
    seed = _int(split_row.get("seed"))
    terminal_survivor = bool(split_row.get("terminal_survivor"))
    outcome = _outcome_class(split_row)
    return {
        "split": "heldout",
        "split_record_index": split_record_index,
        "split_record_id": split_row.get("record_id"),
        "seed": seed,
        "branch_id": split_row.get("branch_id"),
        "branch_state_digest": split_row.get("branch_state_digest"),
        "branch_tick": split_row.get("branch_tick"),
        "continuation_script": split_row.get("continuation_script"),
        "terminal_survivor": terminal_survivor,
        "outcome_class": outcome,
        "trajectory_path": split_row.get("trajectory_path"),
        "trajectory_record_position": trajectory_record_position,
        "trajectory_record_index": _int(
            record.get(TRAJECTORY_DATASET_RECORD_INDEX_FIELD),
            default=trajectory_record_position,
        ),
        "tick": record.get("tick"),
        "agent_id": record.get("agent_id"),
        "action_source": record.get("action_source"),
        "logged_action": logged_action,
        "resolved_action": record.get("resolved_action"),
        "logged_action_rank": rank,
        "neural_top_action": neural_action,
        "linear_top_action": linear_action,
        "configured_blended_action": configured_action,
        "margin_ignored_action": margin_ignored_action,
        "forced_gate_action": forced_gate_action,
        "configured_would_change": configured_action != linear_action,
        "margin_ignored_would_change": margin_ignored_action != linear_action,
        "forced_gate_would_change": forced_gate_action != linear_action,
        "recovery_phase_remaining": recovery_remaining,
        "residual_context_gate_passed": bool(gate_status.get("passed")),
        "residual_context_gate_reason": str(gate_status.get("reason", "unknown")),
        "residual_safety_guard_reason": safety_guard or "none",
        "post_carrion_contact": bool(context_snapshot.get("post_carrion_contact")),
        "drink_available": bool(mask.get("drink", False)),
        "eat_available_non_eat_logged": (
            bool(mask.get("eat", False)) and logged_action != "eat"
        ),
        "hydration_ratio": _round(hydration),
        "water_navigation_dx": _round(water_dx),
        "water_navigation_dy": _round(water_dy),
        "water_directed_actions": list(water_actions),
        "windows": list(windows),
    }


def _window_memberships(
    *,
    trajectory_record_position: int,
    context_snapshot: Mapping[str, object],
    recovery_phase_remaining: int,
    action_mask: Mapping[str, bool],
    logged_action: str,
    observation_values: Sequence[float],
) -> tuple[str, ...]:
    windows = ["all_records"]
    if trajectory_record_position == 0:
        windows.append("first_record")
    post_carrion = bool(context_snapshot.get("post_carrion_contact", False))
    recovery_phase = int(recovery_phase_remaining) > 0
    drink_available = bool(action_mask.get("drink", False))
    water_actions = _water_directed_actions(observation_values, action_mask)
    hydration = _self_feature(observation_values, "hydration_ratio")
    if post_carrion:
        windows.append("post_carrion")
    if recovery_phase:
        windows.append("recovery_phase")
    if drink_available:
        windows.append("drink_available")
    if post_carrion and hydration < LOW_HYDRATION_AFTER_CARRION_THRESHOLD:
        windows.append("low_hydration_after_carrion")
    if logged_action in water_actions:
        windows.append("water_directed_move")
    if (
        bool(action_mask.get("stay", False))
        and post_carrion
        and not drink_available
        and not water_actions
    ):
        windows.append("conserve_or_stay_candidate")
    if bool(action_mask.get("eat", False)) and logged_action != "eat":
        windows.append("eat_available_non_eat_logged")
    return tuple(window for window in WINDOW_NAMES if window in set(windows))


def _window_summaries(
    scored_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    windows = {
        window: {"window": window, "by_outcome_class": _empty_outcome_stats()}
        for window in WINDOW_NAMES
    }
    for row in scored_rows:
        row_windows = row.get("windows")
        for window in row_windows if isinstance(row_windows, list) else []:
            if window not in windows:
                continue
            by_outcome = windows[str(window)]["by_outcome_class"]
            assert isinstance(by_outcome, dict)
            outcome = str(row.get("outcome_class", "unknown"))
            _update_window_stats(by_outcome["all"], row)
            if outcome not in by_outcome:
                by_outcome[outcome] = _empty_stats()
            _update_window_stats(by_outcome[outcome], row)
    return {
        window: {
            "window": window,
            "by_outcome_class": {
                outcome: _finalize_stats(stats)
                for outcome, stats in sorted(
                    _mapping(payload["by_outcome_class"]).items()
                )
            },
        }
        for window, payload in windows.items()
    }


def _empty_outcome_stats() -> dict[str, dict[str, object]]:
    return {
        "all": _empty_stats(),
        "failure": _empty_stats(),
        "survivor": _empty_stats(),
    }


def _empty_stats() -> dict[str, object]:
    return {
        "row_count": 0,
        "survivor_row_count": 0,
        "failure_row_count": 0,
        "logged_action_counts": Counter(),
        "neural_top_action_counts": Counter(),
        "linear_top_action_counts": Counter(),
        "configured_blended_action_counts": Counter(),
        "margin_ignored_action_counts": Counter(),
        "forced_gate_action_counts": Counter(),
        "logged_survivor_action_rank_counts": Counter(),
        "logged_failure_action_rank_counts": Counter(),
        "logged_survivor_action_rank_total": 0,
        "logged_failure_action_rank_total": 0,
        "logged_survivor_action_rank_count": 0,
        "logged_failure_action_rank_count": 0,
        "configured_would_change_count": 0,
        "margin_ignored_would_change_count": 0,
        "forced_gate_would_change_count": 0,
        "would_change_counts_by_final_action": {
            "configured": Counter(),
            "margin_ignored": Counter(),
            "forced_gate": Counter(),
        },
        "would_change_counts_by_seed": {
            "configured": Counter(),
            "margin_ignored": Counter(),
            "forced_gate": Counter(),
        },
        "would_change_counts_by_final_action_and_seed": {
            "configured": {},
            "margin_ignored": {},
            "forced_gate": {},
        },
        "alignment_counts": {
            surface: Counter() for surface, _field in ACTION_SURFACES
        },
        "extra_eat_pressure_counts": Counter(),
        "extra_eat_pressure_counts_by_seed": {
            "configured": Counter(),
            "margin_ignored": Counter(),
        },
    }


def _update_window_stats(
    stats: dict[str, object],
    row: Mapping[str, object],
) -> None:
    stats["row_count"] = _int(stats.get("row_count")) + 1
    if bool(row.get("terminal_survivor", False)):
        stats["survivor_row_count"] = _int(stats.get("survivor_row_count")) + 1
        rank_key = "logged_survivor_action_rank_counts"
        total_key = "logged_survivor_action_rank_total"
        count_key = "logged_survivor_action_rank_count"
    else:
        stats["failure_row_count"] = _int(stats.get("failure_row_count")) + 1
        rank_key = "logged_failure_action_rank_counts"
        total_key = "logged_failure_action_rank_total"
        count_key = "logged_failure_action_rank_count"
    _counter(stats, "logged_action_counts").update([str(row.get("logged_action"))])
    _counter(stats, "neural_top_action_counts").update(
        [str(row.get("neural_top_action"))]
    )
    _counter(stats, "linear_top_action_counts").update(
        [str(row.get("linear_top_action"))]
    )
    _counter(stats, "configured_blended_action_counts").update(
        [str(row.get("configured_blended_action"))]
    )
    _counter(stats, "margin_ignored_action_counts").update(
        [str(row.get("margin_ignored_action"))]
    )
    _counter(stats, "forced_gate_action_counts").update(
        [str(row.get("forced_gate_action"))]
    )
    rank = _int(row.get("logged_action_rank"))
    _counter(stats, rank_key).update([str(rank)])
    stats[total_key] = _int(stats.get(total_key)) + rank
    stats[count_key] = _int(stats.get(count_key)) + 1

    for surface, field in SHADOW_SURFACES:
        changed_key = f"{surface}_would_change"
        if not bool(row.get(changed_key, False)):
            continue
        count_key_name = f"{surface}_would_change_count"
        stats[count_key_name] = _int(stats.get(count_key_name)) + 1
        final_action = str(row.get(field, "unknown"))
        seed = str(_int(row.get("seed")))
        change_by_action = _mapping(stats["would_change_counts_by_final_action"])
        _as_counter(change_by_action[surface]).update([final_action])
        change_by_seed = _mapping(stats["would_change_counts_by_seed"])
        _as_counter(change_by_seed[surface]).update([seed])
        nested = _mapping(stats["would_change_counts_by_final_action_and_seed"])
        surface_payload = nested.setdefault(surface, {})
        if isinstance(surface_payload, dict):
            action_counter = surface_payload.setdefault(final_action, Counter())
            _as_counter(action_counter).update([seed])
        if surface in {"configured", "margin_ignored"} and final_action == "eat":
            _counter(stats, "extra_eat_pressure_counts").update([surface])
            pressure_by_seed = _mapping(stats["extra_eat_pressure_counts_by_seed"])
            _as_counter(pressure_by_seed[surface]).update([seed])

    alignment_counts = _mapping(stats["alignment_counts"])
    for surface, field in ACTION_SURFACES:
        action = str(row.get(field, "unknown"))
        alignment = _action_alignment(row, action)
        counter = _as_counter(alignment_counts[surface])
        for key, value in alignment.items():
            if value:
                counter.update([key])


def _finalize_stats(stats: Mapping[str, object]) -> dict[str, object]:
    survivor_count = _int(stats.get("logged_survivor_action_rank_count"))
    failure_count = _int(stats.get("logged_failure_action_rank_count"))
    survivor_mean = (
        _round(_int(stats.get("logged_survivor_action_rank_total")) / survivor_count)
        if survivor_count
        else None
    )
    failure_mean = (
        _round(_int(stats.get("logged_failure_action_rank_total")) / failure_count)
        if failure_count
        else None
    )
    return {
        "row_count": _int(stats.get("row_count")),
        "survivor_row_count": _int(stats.get("survivor_row_count")),
        "failure_row_count": _int(stats.get("failure_row_count")),
        "logged_action_counts": _counter_payload(stats, "logged_action_counts"),
        "neural_top_action_counts": _counter_payload(stats, "neural_top_action_counts"),
        "linear_top_action_counts": _counter_payload(stats, "linear_top_action_counts"),
        "configured_blended_action_counts": _counter_payload(
            stats,
            "configured_blended_action_counts",
        ),
        "margin_ignored_action_counts": _counter_payload(
            stats,
            "margin_ignored_action_counts",
        ),
        "forced_gate_action_counts": _counter_payload(
            stats,
            "forced_gate_action_counts",
        ),
        "logged_survivor_action_rank_counts": _counter_payload(
            stats,
            "logged_survivor_action_rank_counts",
        ),
        "logged_failure_action_rank_counts": _counter_payload(
            stats,
            "logged_failure_action_rank_counts",
        ),
        "logged_survivor_action_rank_mean": survivor_mean,
        "logged_failure_action_rank_mean": failure_mean,
        "survivor_minus_failure_rank_delta": (
            _round(float(survivor_mean) - float(failure_mean))
            if survivor_mean is not None and failure_mean is not None
            else None
        ),
        "configured_would_change_count": _int(
            stats.get("configured_would_change_count")
        ),
        "margin_ignored_would_change_count": _int(
            stats.get("margin_ignored_would_change_count")
        ),
        "forced_gate_would_change_count": _int(
            stats.get("forced_gate_would_change_count")
        ),
        "would_change_counts_by_final_action": _nested_counter_payload(
            _mapping(stats.get("would_change_counts_by_final_action"))
        ),
        "would_change_counts_by_seed": _nested_counter_payload(
            _mapping(stats.get("would_change_counts_by_seed"))
        ),
        "would_change_counts_by_final_action_and_seed": (
            _two_level_nested_counter_payload(
                _mapping(stats.get("would_change_counts_by_final_action_and_seed"))
            )
        ),
        "alignment_counts": {
            surface: _alignment_counter_payload(_as_counter(counter))
            for surface, counter in sorted(
                _mapping(stats.get("alignment_counts")).items()
            )
        },
        "extra_eat_pressure_counts": {
            "configured": _int(
                _as_counter(stats.get("extra_eat_pressure_counts")).get(
                    "configured"
                )
            ),
            "margin_ignored": _int(
                _as_counter(stats.get("extra_eat_pressure_counts")).get(
                    "margin_ignored"
                )
            ),
        },
        "extra_eat_pressure_counts_by_seed": _nested_counter_payload(
            _mapping(stats.get("extra_eat_pressure_counts_by_seed"))
        ),
    }


def _action_alignment(
    row: Mapping[str, object],
    action: str,
) -> dict[str, bool]:
    drink = action == "drink" and bool(row.get("drink_available", False))
    water_move = action in set(
        str(item) for item in row.get("water_directed_actions", [])
    )
    stay = (
        action == "stay"
        and "conserve_or_stay_candidate" in set(row.get("windows", []))
    )
    avoid_eat = bool(row.get("eat_available_non_eat_logged", False)) and action != "eat"
    useful = drink or water_move or stay or avoid_eat
    return {
        "drink": drink,
        "water_directed_movement": water_move,
        "appropriate_stay_conserve": stay,
        "avoid_extra_eat": avoid_eat,
        "useful_recovery_action_count": useful,
    }


def _oracle_comparison(
    scored_rows: Sequence[Mapping[str, object]],
    *,
    oracle_audit: Mapping[str, object] | None,
    oracle_audit_path: str | Path | None,
    oracle_load_error: str | None,
) -> dict[str, object]:
    provided = oracle_audit_path is not None or oracle_audit is not None
    if oracle_audit is None:
        return {
            "provided": provided,
            "loaded": False,
            "load_error": oracle_load_error,
            "matched_row_count": 0,
            "unmatched_row_count": len(scored_rows),
            "match_policy_counts": {},
            "oracle_best_action_counts": {},
            "action_match_counts": _empty_oracle_action_match_counts(),
            "diagnostics_only": True,
        }
    index = _oracle_index(oracle_audit)
    matched = 0
    unmatched = 0
    match_policy_counts: Counter[str] = Counter()
    oracle_actions: Counter[str] = Counter()
    action_matches = {
        "neural_top": Counter(),
        "configured": Counter(),
        "margin_ignored": Counter(),
        "forced_gate": Counter(),
    }
    examples: list[dict[str, object]] = []
    for row in scored_rows:
        lookup = _oracle_lookup(row, index)
        if lookup is None:
            unmatched += 1
            continue
        matched += 1
        match_policy_counts[str(lookup["match_policy"])] += 1
        oracle_best = str(lookup["oracle_best_action"])
        oracle_actions[oracle_best] += 1
        comparisons = {
            "neural_top": str(row.get("neural_top_action")),
            "configured": str(row.get("configured_blended_action")),
            "margin_ignored": str(row.get("margin_ignored_action")),
            "forced_gate": str(row.get("forced_gate_action")),
        }
        for surface, action in comparisons.items():
            action_matches[surface].update(
                ["match" if action == oracle_best else "mismatch"]
            )
        if len(examples) < 12:
            examples.append(
                {
                    "seed": row.get("seed"),
                    "branch_id": row.get("branch_id"),
                    "branch_state_digest": row.get("branch_state_digest"),
                    "branch_tick": row.get("branch_tick"),
                    "tick": row.get("tick"),
                    "agent_id": row.get("agent_id"),
                    "logged_action": row.get("logged_action"),
                    "oracle_best_action": oracle_best,
                    "neural_top_action": row.get("neural_top_action"),
                    "configured_blended_action": row.get(
                        "configured_blended_action"
                    ),
                    "margin_ignored_action": row.get("margin_ignored_action"),
                    "match_policy": lookup["match_policy"],
                }
            )
    return {
        "provided": provided,
        "loaded": True,
        "schema_version": oracle_audit.get("schema_version"),
        "matched_row_count": matched,
        "unmatched_row_count": unmatched,
        "matched_share": _share(matched, len(scored_rows)),
        "match_policy_counts": dict(sorted(match_policy_counts.items())),
        "oracle_best_action_counts": _counter_to_dict(oracle_actions),
        "action_match_counts": {
            surface: {
                "match": _int(counter.get("match")),
                "mismatch": _int(counter.get("mismatch")),
                "match_share": _share(_int(counter.get("match")), matched),
            }
            for surface, counter in sorted(action_matches.items())
        },
        "examples": examples,
        "diagnostics_only": True,
    }


def _oracle_index(
    oracle_audit: Mapping[str, object],
) -> dict[str, dict[tuple[object, ...], Mapping[str, object]]]:
    indexes: dict[str, dict[tuple[object, ...], Mapping[str, object]]] = {
        "exact_branch_id": {},
        "digest_only": {},
    }
    results = oracle_audit.get("branch_results")
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, Mapping):
            continue
        oracle_best = result.get("oracle_best_action")
        if not isinstance(oracle_best, str) or not oracle_best:
            continue
        exact = (
            result.get("branch_state_digest"),
            result.get("branch_id"),
            _int(result.get("seed")),
            _int(result.get("branch_tick")),
            _int(result.get("agent_id")),
            result.get("logged_action"),
        )
        digest = (
            result.get("branch_state_digest"),
            _int(result.get("seed")),
            _int(result.get("branch_tick")),
            _int(result.get("agent_id")),
            result.get("logged_action"),
        )
        indexes["exact_branch_id"][exact] = result
        indexes["digest_only"][digest] = result
    return indexes


def _oracle_lookup(
    row: Mapping[str, object],
    index: Mapping[str, Mapping[tuple[object, ...], Mapping[str, object]]],
) -> dict[str, object] | None:
    branch_ticks = (
        _int(row.get("branch_tick")),
        _int(row.get("tick")),
    )
    for branch_tick in branch_ticks:
        exact = (
            row.get("branch_state_digest"),
            row.get("branch_id"),
            _int(row.get("seed")),
            branch_tick,
            _int(row.get("agent_id")),
            row.get("logged_action"),
        )
        match = _mapping(index.get("exact_branch_id")).get(exact)
        if isinstance(match, Mapping):
            return {
                "match_policy": "exact_branch_id",
                "oracle_best_action": match.get("oracle_best_action"),
            }
        digest = (
            row.get("branch_state_digest"),
            _int(row.get("seed")),
            branch_tick,
            _int(row.get("agent_id")),
            row.get("logged_action"),
        )
        match = _mapping(index.get("digest_only")).get(digest)
        if isinstance(match, Mapping):
            return {
                "match_policy": "digest_only",
                "oracle_best_action": match.get("oracle_best_action"),
            }
    return None


def _classification(
    *,
    window_summaries: Mapping[str, object],
    oracle_comparison: Mapping[str, object],
    residual_audit: Mapping[str, object],
) -> dict[str, object]:
    scores = {
        "target_alignment_good_but_margin_blocked": 0,
        "target_alignment_bad_extra_eat": 0,
        "first_record_sampling_gap": 0,
        "trajectory_label_objective_too_coarse": 0,
        "training_signal_weak": 0,
        "oracle_alignment_inconclusive": 0,
    }
    evidence: list[str] = []
    first = _window_all_stats(window_summaries, "first_record")
    all_records = _window_all_stats(window_summaries, "all_records")
    recovery = _window_all_stats(window_summaries, "recovery_phase")
    post = _window_all_stats(window_summaries, "post_carrion")
    focus = recovery if _int(recovery.get("row_count")) > 0 else post
    if _int(focus.get("row_count")) <= 0:
        focus = all_records

    margin_useful = _alignment_useful(focus, "margin_ignored")
    configured_useful = _alignment_useful(focus, "configured")
    first_margin_useful = _alignment_useful(first, "margin_ignored")
    all_margin_useful = _alignment_useful(all_records, "margin_ignored")
    margin_changes = _int(focus.get("margin_ignored_would_change_count"))
    configured_changes = _int(focus.get("configured_would_change_count"))
    margin_extra = _extra_eat(focus, "margin_ignored")
    configured_extra = _extra_eat(focus, "configured")
    extra_total = margin_extra + configured_extra
    useful_total = margin_useful + configured_useful
    rank_delta = focus.get("survivor_minus_failure_rank_delta")

    if extra_total > 0 and extra_total >= max(1, useful_total):
        scores["target_alignment_bad_extra_eat"] += 10
        evidence.append("configured_or_margin_ignored_changes_mostly_end_in_eat")

    if (
        _int(all_records.get("row_count")) > _int(first.get("row_count"))
        and all_margin_useful > first_margin_useful
    ):
        scores["first_record_sampling_gap"] += 9
        evidence.append("all_record_margin_ignored_alignment_exceeds_first_record")

    if (
        margin_useful > configured_useful
        and margin_useful > extra_total
        and margin_changes > configured_changes
        and isinstance(rank_delta, (int, float))
        and float(rank_delta) < 0.0
    ):
        scores["target_alignment_good_but_margin_blocked"] += 7
        evidence.append("margin_ignored_has_useful_recovery_alignment_blocked_by_margin")

    if isinstance(rank_delta, (int, float)) and float(rank_delta) >= 0.0:
        scores["trajectory_label_objective_too_coarse"] += 7
        evidence.append("survivor_logged_actions_do_not_rank_above_failure_actions")
    elif rank_delta is None and _int(focus.get("row_count")) > 0:
        scores["training_signal_weak"] += 3
        evidence.append("survivor_failure_rank_delta_unavailable")

    if margin_changes <= 0 and configured_changes <= 0 and margin_useful <= 0:
        scores["training_signal_weak"] += 5
        evidence.append("residual_shadow_actions_do_not_change_linear_anchor")

    if not bool(oracle_comparison.get("loaded")):
        scores["oracle_alignment_inconclusive"] += 2
        evidence.append("oracle_audit_absent_or_unloaded")
    elif _int(oracle_comparison.get("matched_row_count")) < MIN_INFORMATIVE_ORACLE_MATCHES:
        scores["oracle_alignment_inconclusive"] += 2
        evidence.append("oracle_match_count_below_informative_floor")

    residual_calibration = _mapping(residual_audit.get("calibration_classification"))
    if residual_calibration.get("primary") == "margin domination":
        evidence.append("v68_residual_audit_classified_margin_domination")

    positive = [label for label, score in scores.items() if score > 0]
    if not positive:
        scores["training_signal_weak"] = 1
        positive = ["training_signal_weak"]
        evidence.append("no_positive_alignment_classification_evidence")
    primary = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return {
        "primary": primary,
        "labels": positive,
        "category_scores": scores,
        "evidence": evidence,
        "diagnostics_only": True,
    }


def _window_all_stats(
    window_summaries: Mapping[str, object],
    window: str,
) -> Mapping[str, object]:
    return _mapping(
        _mapping(_mapping(window_summaries.get(window)).get("by_outcome_class")).get(
            "all"
        )
    )


def _alignment_useful(stats: Mapping[str, object], surface: str) -> int:
    alignment = _mapping(_mapping(stats.get("alignment_counts")).get(surface))
    return _int(alignment.get("useful_recovery_action_count"))


def _extra_eat(stats: Mapping[str, object], surface: str) -> int:
    return _int(_mapping(stats.get("extra_eat_pressure_counts")).get(surface))


def _split_rows(
    split_report: Mapping[str, object],
    *,
    split: str,
) -> list[Mapping[str, object]]:
    rows = _mapping(split_report.get("records")).get(split)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _load_trajectory_payload(
    path: str | Path,
) -> tuple[Mapping[str, object], tuple[dict[str, object], ...]]:
    records: list[dict[str, object]] = []
    header: Mapping[str, object] = {}
    with _open_input(Path(path)) as handle:
        for line_index, line in enumerate(handle):
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                continue
            if line_index == 0:
                header = payload
            record = payload.get("record")
            if isinstance(record, Mapping):
                records.append(dict(record))
            elif _looks_like_trajectory_record(payload):
                records.append(dict(payload))
    if not records:
        raise RecoveryActionTargetAlignmentError(
            f"trajectory has no record payloads: {path}"
        )
    return header, tuple(records)


def _looks_like_trajectory_record(payload: Mapping[str, object]) -> bool:
    return (
        "observation_input" in payload
        and "action_mask" in payload
        and "requested_action" in payload
    )


def _outcome_class(row: Mapping[str, object]) -> str:
    outcome = row.get("outcome_class")
    if isinstance(outcome, str) and outcome:
        return outcome
    return "survivor" if bool(row.get("terminal_survivor", False)) else "failure"


def _logged_action(record: Mapping[str, object]) -> str:
    requested = record.get("requested_action")
    if isinstance(requested, str) and requested:
        return requested
    resolved = record.get("resolved_action")
    if isinstance(resolved, str) and resolved:
        return resolved
    return "unknown"


def _score_rank(scores: Mapping[str, float], action: str) -> int:
    ranked = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
    for index, (candidate, _score) in enumerate(ranked, start=1):
        if candidate == action:
            return index
    return len(ranked) + 1


def _self_feature(values: Sequence[float], field: str) -> float:
    try:
        return _float(values[SELF_INPUT_FIELDS.index(field)])
    except (ValueError, IndexError):
        return 0.0


def _navigation_feature(values: Sequence[float], target: str, field: str) -> float:
    try:
        target_index = NAVIGATION_TARGETS.index(target)
        field_index = NAVIGATION_INPUT_FIELDS.index(field)
    except ValueError:
        return 0.0
    start = len(SELF_INPUT_FIELDS) + PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
    index = start + target_index * len(NAVIGATION_INPUT_FIELDS) + field_index
    if index < 0 or index >= len(values):
        return 0.0
    return _float(values[index])


def _water_directed_actions(
    observation_values: Sequence[float],
    action_mask: Mapping[str, bool],
) -> tuple[str, ...]:
    dx = _navigation_feature(observation_values, "water", "dx")
    dy = _navigation_feature(observation_values, "water", "dy")
    actions: list[str] = []
    tolerance = 1e-6
    if dx > tolerance and bool(action_mask.get("move_east", False)):
        actions.append("move_east")
    if dx < -tolerance and bool(action_mask.get("move_west", False)):
        actions.append("move_west")
    if dy > tolerance and bool(action_mask.get("move_south", False)):
        actions.append("move_south")
    if dy < -tolerance and bool(action_mask.get("move_north", False)):
        actions.append("move_north")
    return tuple(action for action in MOVEMENT_ACTIONS if action in set(actions))


def _empty_oracle_action_match_counts() -> dict[str, dict[str, object]]:
    return {
        surface: {"match": 0, "mismatch": 0, "match_share": 0.0}
        for surface in ("configured", "forced_gate", "margin_ignored", "neural_top")
    }


def _alignment_counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter.get(key, 0)) for key in ALIGNMENT_KEYS}


def _nested_counter_payload(value: Mapping[str, object]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for key, payload in sorted(value.items()):
        result[str(key)] = _counter_to_dict(_as_counter(payload))
    return result


def _two_level_nested_counter_payload(
    value: Mapping[str, object],
) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for surface, payload in sorted(value.items()):
        surface_payload: dict[str, dict[str, int]] = {}
        for action, counter in sorted(_mapping(payload).items()):
            surface_payload[str(action)] = _counter_to_dict(_as_counter(counter))
        result[str(surface)] = surface_payload
    return result


def _counter_payload(stats: Mapping[str, object], key: str) -> dict[str, int]:
    return _counter_to_dict(_as_counter(stats.get(key)))


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items())
        if int(value) > 0
    }


def _counter(stats: dict[str, object], key: str) -> Counter[str]:
    counter = stats.get(key)
    if not isinstance(counter, Counter):
        counter = Counter()
        stats[key] = counter
    return counter


def _as_counter(value: object) -> Counter[str]:
    if isinstance(value, Counter):
        return value
    if isinstance(value, Mapping):
        return Counter({str(key): _int(count) for key, count in value.items()})
    return Counter()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return default


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _round(value: object) -> float:
    return round(_float(value), 6)


def _share(count: int, total: int) -> float:
    return _round(count / float(total)) if total > 0 else 0.0


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
