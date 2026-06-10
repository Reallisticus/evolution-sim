from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO

from evolution_sim.mind.evaluation_harness import run_mind_v3_fixture_suite
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import decode_observation_input
from evolution_sim.mind.carrion_recovery_archive import (
    MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
    MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION,
    load_carrion_recovery_json_report,
)
from evolution_sim.mind.evolution import score_mind_v3_metadata
from evolution_sim.mind.policy_inputs import (
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    ecological_policy_input_contract,
    ecological_policy_input_values,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_neural import (
    MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
    score_mind_v3_neural_artifact,
    validate_mind_v3_neural_artifact,
)
from evolution_sim.mind.v3_policy import (
    MindV3EvolutionPolicy,
    _best_action,
    _blend_neural_with_linear_anchor,
    _neural_residual_shadow_reason,
    _neural_residual_context_gate_status,
    _neural_residual_safety_guard_reason,
    _normalized_legal_scores,
)

MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_SCHEMA_VERSION = (
    "mind_v3_carrion_recovery_residual_activation_audit_v1"
)
MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_POLICY = (
    "diagnostics_only_recovery_residual_activation_audit_v1"
)


class CarrionRecoveryResidualAuditError(ValueError):
    pass


def load_carrion_recovery_residual_audit_json(path: str | Path) -> dict[str, object]:
    with _open_input(Path(path)) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise CarrionRecoveryResidualAuditError(f"report must be a JSON object: {path}")
    return payload


def build_carrion_recovery_residual_audit_report(
    *,
    artifact: Mapping[str, object] | None = None,
    artifact_path: str | Path | None = None,
    distill_report: Mapping[str, object] | None = None,
    distill_report_path: str | Path | None = None,
    evaluation_report: Mapping[str, object] | None = None,
    evaluation_report_path: str | Path | None = None,
    activation_audit: Mapping[str, object] | None = None,
    activation_audit_path: str | Path | None = None,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = None,
    split_report: Mapping[str, object] | None = None,
    split_report_path: str | Path | None = None,
    fixture_seeds: Sequence[int] = (),
    fixture_ticks: int = 120,
    margin_sweep: Sequence[float] = (),
    scale_sweep: Sequence[float] = (),
) -> dict[str, object]:
    if artifact is None:
        if artifact_path is None:
            raise CarrionRecoveryResidualAuditError("artifact_path is required")
        artifact = load_carrion_recovery_residual_audit_json(artifact_path)
    validate_mind_v3_neural_artifact(artifact)
    if artifact.get("schema_version") != MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION:
        raise CarrionRecoveryResidualAuditError("artifact has stale schema_version")
    if distill_report is None:
        if distill_report_path is None:
            raise CarrionRecoveryResidualAuditError("distill_report_path is required")
        distill_report = load_carrion_recovery_residual_audit_json(distill_report_path)
    if evaluation_report is None and evaluation_report_path is not None:
        evaluation_report = load_carrion_recovery_residual_audit_json(
            evaluation_report_path
        )
    if activation_audit is None and activation_audit_path is not None:
        activation_audit = load_carrion_recovery_residual_audit_json(
            activation_audit_path
        )
    if archive_report is None:
        if archive_report_path is None:
            raise CarrionRecoveryResidualAuditError("archive_report_path is required")
        archive_report = load_carrion_recovery_json_report(archive_report_path)
    if archive_report.get("schema_version") != MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION:
        raise CarrionRecoveryResidualAuditError(
            "archive report has stale schema_version"
        )
    if split_report is None:
        if split_report_path is None:
            raise CarrionRecoveryResidualAuditError("split_report_path is required")
        split_report = load_carrion_recovery_json_report(split_report_path)
    if split_report.get("schema_version") != MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION:
        raise CarrionRecoveryResidualAuditError("split report has stale schema_version")
    seed_values = tuple(int(seed) for seed in fixture_seeds)
    if not seed_values:
        raise CarrionRecoveryResidualAuditError("fixture_seeds must not be empty")
    ticks = _positive_int(fixture_ticks, "fixture_ticks")

    train_rows = _split_rows(split_report, split="train")
    heldout_rows = _split_rows(split_report, split="heldout")
    activation_feature_checks = _mapping(
        activation_audit.get("feature_contract_checks")
    ) if activation_audit is not None else {}
    if activation_feature_checks:
        feature_checks = dict(activation_feature_checks)
    else:
        feature_checks = _feature_contract_checks(
            artifact=artifact,
            train_rows=train_rows,
            heldout_rows=heldout_rows,
        )
    activation_offline = _mapping(
        activation_audit.get("offline_score_summaries")
    ) if activation_audit is not None else {}
    if _mapping(activation_offline.get("train")) and _mapping(
        activation_offline.get("heldout")
    ):
        offline = {
            "train": dict(_mapping(activation_offline.get("train"))),
            "heldout": dict(_mapping(activation_offline.get("heldout"))),
        }
    else:
        offline = {
            "train": _offline_score_summary(
                rows=train_rows,
                artifact=artifact,
                split_name="train",
            ),
            "heldout": _offline_score_summary(
                rows=heldout_rows,
                artifact=artifact,
                split_name="heldout",
            ),
        }
    margin_values = tuple(_float(value) for value in margin_sweep)
    scale_values = tuple(_float(value) for value in scale_sweep)
    needs_fixture_records = bool(margin_values or scale_values)
    activation_fixture = _mapping(
        activation_audit.get("fixture_replay_diagnostics")
    ) if activation_audit is not None else {}
    if activation_fixture and not needs_fixture_records:
        fixture_replay = dict(activation_fixture)
        fixture_records: tuple[dict[str, object], ...] = ()
    else:
        fixture_replay, fixture_records = _fixture_replay_diagnostics(
            artifact=dict(artifact),
            seeds=seed_values,
            ticks=ticks,
        )
    classification = _failure_classification(
        feature_checks=feature_checks,
        offline=offline,
        fixture_replay=fixture_replay,
        distill_report=distill_report,
    )
    contract = {
        "schema_version": MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_POLICY,
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "artifact_promotion_effect": "none",
        "shadow_forced_gate_is_offline_only": True,
        "shadow_margin_ignored_is_offline_only": True,
        "shadow_margin_sweep_is_offline_only": True,
        "shadow_scale_sweep_is_offline_only": True,
    }
    report = {
        "schema_version": MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_POLICY,
        "contract": contract,
        "provenance": {
            "contract_digest": stable_payload_digest(contract),
            "artifact_path": str(artifact_path) if artifact_path is not None else None,
            "artifact_digest": stable_payload_digest(artifact),
            "distill_report_path": (
                str(distill_report_path)
                if distill_report_path is not None
                else None
            ),
            "distill_report_digest": stable_payload_digest(distill_report),
            "evaluation_report_path": (
                str(evaluation_report_path)
                if evaluation_report_path is not None
                else None
            ),
            "evaluation_report_digest": (
                stable_payload_digest(evaluation_report)
                if evaluation_report is not None
                else None
            ),
            "activation_audit_path": (
                str(activation_audit_path)
                if activation_audit_path is not None
                else None
            ),
            "activation_audit_digest": (
                stable_payload_digest(activation_audit)
                if activation_audit is not None
                else None
            ),
            "archive_report_path": (
                str(archive_report_path)
                if archive_report_path is not None
                else None
            ),
            "archive_report_digest": stable_payload_digest(archive_report),
            "split_report_path": (
                str(split_report_path) if split_report_path is not None else None
            ),
            "split_report_digest": stable_payload_digest(split_report),
        },
        "artifact_config": _artifact_config(artifact),
        "feature_contract_checks": feature_checks,
        "offline_score_summaries": offline,
        "heldout_branch_state_decision_comparison": (
            _heldout_branch_state_decision_comparison(offline["heldout"])
        ),
        "fixture_replay_diagnostics": fixture_replay,
        "failure_classification": classification,
        "non_promoted": True,
    }
    if evaluation_report is not None or activation_audit is not None:
        report["residual_counter_reconciliation"] = _counter_reconciliation(
            evaluation_report=evaluation_report,
            distill_report=distill_report,
            activation_audit=activation_audit,
            heldout_artifact_replay=offline["heldout"],
            current_fixture_replay=fixture_replay,
        )
    if margin_values or scale_values:
        report["shadow_calibration"] = _shadow_calibration(
            artifact=artifact,
            heldout_rows=heldout_rows,
            fixture_records=fixture_records,
            margin_sweep=margin_values,
            scale_sweep=scale_values,
        )
        report["calibration_classification"] = _calibration_classification(
            feature_checks=feature_checks,
            failure_classification=classification,
            shadow_calibration=_mapping(report.get("shadow_calibration")),
        )
    return report


def write_carrion_recovery_residual_audit_report(
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
        "training_contract_digest": _mapping(artifact.get("provenance")).get(
            "contract_digest"
        ),
    }


def _feature_contract_checks(
    *,
    artifact: Mapping[str, object],
    train_rows: Sequence[Mapping[str, object]],
    heldout_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    artifact_contract = _mapping(artifact.get("input_contract"))
    runtime_contract = ecological_policy_input_contract()
    artifact_fields = _field_names_from_contract(artifact_contract)
    runtime_fields = _field_names_from_contract(runtime_contract)
    counts = Counter()
    for row in list(train_rows) + list(heldout_rows):
        path = row.get("trajectory_path")
        if not isinstance(path, str) or not path:
            counts["missing_trajectory_path"] += 1
            continue
        try:
            records = _load_trajectory_records(path)
        except Exception:
            counts["trajectory_load_failure"] += 1
            continue
        for record in records:
            observation = record.get("observation_input")
            if not isinstance(observation, dict):
                counts["missing_observation_input"] += 1
                continue
            try:
                values = ecological_policy_input_values(observation)
            except Exception:
                counts["malformed_observation_input"] += 1
                continue
            if len(values) != ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE:
                counts["vector_length_mismatch"] += 1
            if any(not math.isfinite(float(value)) for value in values):
                counts["non_finite_value"] += 1
            counts["checked_record"] += 1
    missing_fields = sorted(set(runtime_fields) - set(artifact_fields))
    extra_fields = sorted(set(artifact_fields) - set(runtime_fields))
    blockers = []
    if artifact_contract.get("ecological_vector_size") != ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE:
        blockers.append("artifact_vector_size_mismatch")
    if missing_fields:
        blockers.append("artifact_missing_runtime_fields")
    if extra_fields:
        blockers.append("artifact_has_extra_fields")
    if counts["malformed_observation_input"] or counts["vector_length_mismatch"]:
        blockers.append("runtime_observation_feature_mismatch")
    if counts["non_finite_value"]:
        blockers.append("runtime_observation_non_finite_values")
    return {
        "artifact_input_contract_digest": _mapping(artifact.get("provenance")).get(
            "input_contract_digest"
        ),
        "runtime_input_contract_digest": stable_payload_digest(runtime_contract),
        "artifact_vector_size": artifact_contract.get("ecological_vector_size"),
        "runtime_vector_size": ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        "field_names": artifact_fields,
        "field_name_count": len(artifact_fields),
        "runtime_field_name_count": len(runtime_fields),
        "missing_field_count": len(missing_fields),
        "missing_fields": missing_fields,
        "extra_field_count": len(extra_fields),
        "extra_fields": extra_fields,
        "non_finite_count": int(counts["non_finite_value"]),
        "malformed_record_count": int(counts["malformed_observation_input"]),
        "vector_length_mismatch_count": int(counts["vector_length_mismatch"]),
        "checked_record_count": int(counts["checked_record"]),
        "blockers": blockers,
        "passed": not blockers,
    }


def _field_names_from_contract(contract: Mapping[str, object]) -> list[str]:
    retained = _mapping(contract.get("retained_sections"))
    self_fields = [str(field) for field in retained.get("self_fields", [])]
    patch_fields = [str(field) for field in retained.get("local_patch_fields", [])]
    patch_count = _int(retained.get("local_patch_cell_count"))
    nav_targets = [str(target) for target in retained.get("navigation_targets", [])]
    nav_fields = [str(field) for field in retained.get("navigation_fields", [])]
    fields = [f"self.{field}" for field in self_fields]
    fields.extend(
        f"local_patch[{index}].{field}"
        for index in range(max(0, patch_count))
        for field in patch_fields
    )
    fields.extend(
        f"navigation.{target}.{field}"
        for target in nav_targets
        for field in nav_fields
    )
    return fields


def _offline_score_summary(
    *,
    rows: Sequence[Mapping[str, object]],
    artifact: Mapping[str, object],
    split_name: str,
) -> dict[str, object]:
    stats = _empty_score_stats()
    for row in rows:
        path = row.get("trajectory_path")
        if not isinstance(path, str) or not path:
            stats["load_failures"] += 1
            continue
        try:
            records = _load_trajectory_records(path)
        except Exception:
            stats["load_failures"] += 1
            continue
        survivor_trajectory = bool(row.get("terminal_survivor"))
        policy = MindV3EvolutionPolicy(
            seed=_int(row.get("seed")),
            neural_artifact=dict(artifact),
        )
        for record in records:
            _score_record(
                record,
                artifact=artifact,
                policy=policy,
                survivor_trajectory=survivor_trajectory,
                stats=stats,
            )
            policy.observe_transition(dict(record))
    return _finalize_score_stats(stats, split_name=split_name)


def _score_record(
    record: Mapping[str, object],
    *,
    artifact: Mapping[str, object],
    policy: MindV3EvolutionPolicy,
    survivor_trajectory: bool,
    stats: dict[str, object],
) -> None:
    observation_input = record.get("observation_input")
    action_mask = record.get("action_mask")
    if not isinstance(observation_input, Mapping) or not isinstance(action_mask, Mapping):
        stats["malformed_records"] = int(stats["malformed_records"]) + 1
        return
    observation = {
        "metadata": record.get("observation_metadata", {}),
        "observation_input": dict(observation_input),
    }
    decision = policy.decide(observation, {str(k): bool(v) for k, v in action_mask.items()})
    diagnostics = decision.diagnostics
    agent_id = _int(record.get("agent_id"))
    metadata = policy.agent_mind_metadata(agent_id=agent_id)
    observation_values = decode_observation_input(dict(observation_input))
    mask = {str(k): bool(v) for k, v in action_mask.items()}
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
    scale = _float(artifact.get("neural_residual_scale"))
    margin = _float(artifact.get("neural_residual_max_linear_override_margin"))
    gate_status = _neural_residual_context_gate_status(
        observation_values,
        gate=str(artifact.get("neural_residual_context_gate", "none")),
        recovery_phase_remaining=recovery_remaining,
    )
    safety_guard = _neural_residual_safety_guard_reason(
        observation_values,
        neural_scores=neural_scores,
        linear_scores=linear_scores,
        action_mask=mask,
    )
    guard = safety_guard if bool(gate_status.get("passed")) else None
    configured_scale = scale if bool(gate_status.get("passed")) and guard is None else 0.0
    forced_gate_scale = scale if safety_guard is None else 0.0
    margin_ignored_scale = configured_scale
    configured_scores = _blend_neural_with_linear_anchor(
        neural_scores=neural_scores,
        linear_scores=linear_scores,
        action_mask=mask,
        residual_scale=configured_scale,
        max_linear_override_margin=margin,
    )
    forced_gate_scores = _blend_neural_with_linear_anchor(
        neural_scores=neural_scores,
        linear_scores=linear_scores,
        action_mask=mask,
        residual_scale=forced_gate_scale,
        max_linear_override_margin=margin,
    )
    margin_ignored_scores = _blend_neural_with_linear_anchor(
        neural_scores=neural_scores,
        linear_scores=linear_scores,
        action_mask=mask,
        residual_scale=margin_ignored_scale,
        max_linear_override_margin=float("inf"),
    )
    linear_action, _linear_score = _best_action(linear_scores, mask)
    configured_action, _configured_score = _best_action(configured_scores, mask)
    forced_gate_action, _forced_score = _best_action(forced_gate_scores, mask)
    margin_ignored_action, _ignored_score = _best_action(margin_ignored_scores, mask)
    neural_action, _neural_score = _best_action(neural_scores, mask)
    logged_action = str(record.get("requested_action", "unknown"))
    residual_delta = max(
        (
            abs(_float(configured_scores.get(action)) - _float(linear_scores.get(action)))
            for action in configured_scores
        ),
        default=0.0,
    )
    _update_score_stats(
        stats,
        neural_scores=neural_scores,
        linear_action=linear_action,
        neural_action=neural_action,
        configured_action=configured_action,
        forced_gate_action=forced_gate_action,
        margin_ignored_action=margin_ignored_action,
        logged_action=logged_action,
        survivor_trajectory=survivor_trajectory,
        residual_delta=residual_delta,
        diagnostics=diagnostics,
    )


def _empty_score_stats() -> dict[str, object]:
    return {
        "decision_count": 0,
        "malformed_records": 0,
        "load_failures": 0,
        "neural_scores_by_action": {action: [] for action in ACTION_NAMES},
        "neural_top_action_counts": Counter(),
        "linear_top_action_counts": Counter(),
        "configured_blended_top_action_counts": Counter(),
        "shadow_forced_gate_top_action_counts": Counter(),
        "shadow_margin_ignored_top_action_counts": Counter(),
        "logged_survivor_action_rank_counts": Counter(),
        "logged_survivor_action_rank_total": 0,
        "logged_survivor_action_rank_count": 0,
        "residual_delta_total": 0.0,
        "residual_delta_abs_max": 0.0,
        "artifact_linear_agreement_count": 0,
        "configured_would_change_count": 0,
        "shadow_forced_gate_would_change_count": 0,
        "shadow_margin_ignored_would_change_count": 0,
        "neural_linear_disagreement_count": 0,
        "residual_applied_count": 0,
        "actual_changed_linear_count": 0,
        "shadow_reason_counts": Counter(),
    }


def _update_score_stats(
    stats: dict[str, object],
    *,
    neural_scores: Mapping[str, float],
    linear_action: str,
    neural_action: str,
    configured_action: str,
    forced_gate_action: str,
    margin_ignored_action: str,
    logged_action: str,
    survivor_trajectory: bool,
    residual_delta: float,
    diagnostics: Mapping[str, object],
) -> None:
    stats["decision_count"] = int(stats["decision_count"]) + 1
    score_lists = stats["neural_scores_by_action"]
    assert isinstance(score_lists, dict)
    for action, score in neural_scores.items():
        values = score_lists.setdefault(action, [])
        if isinstance(values, list):
            values.append(_float(score))
    _counter(stats, "neural_top_action_counts").update([neural_action])
    _counter(stats, "linear_top_action_counts").update([linear_action])
    _counter(stats, "configured_blended_top_action_counts").update([configured_action])
    _counter(stats, "shadow_forced_gate_top_action_counts").update([forced_gate_action])
    _counter(stats, "shadow_margin_ignored_top_action_counts").update(
        [margin_ignored_action]
    )
    if configured_action == linear_action:
        stats["artifact_linear_agreement_count"] = (
            int(stats["artifact_linear_agreement_count"]) + 1
        )
    else:
        stats["configured_would_change_count"] = (
            int(stats["configured_would_change_count"]) + 1
        )
    if forced_gate_action != linear_action:
        stats["shadow_forced_gate_would_change_count"] = (
            int(stats["shadow_forced_gate_would_change_count"]) + 1
        )
    if margin_ignored_action != linear_action:
        stats["shadow_margin_ignored_would_change_count"] = (
            int(stats["shadow_margin_ignored_would_change_count"]) + 1
        )
    if neural_action != linear_action:
        stats["neural_linear_disagreement_count"] = (
            int(stats["neural_linear_disagreement_count"]) + 1
        )
    if bool(diagnostics.get("neural_residual_applied")):
        stats["residual_applied_count"] = int(stats["residual_applied_count"]) + 1
    if bool(diagnostics.get("neural_residual_changed_linear_action")):
        stats["actual_changed_linear_count"] = (
            int(stats["actual_changed_linear_count"]) + 1
        )
    shadow_reason = diagnostics.get("neural_residual_shadow_reason")
    if isinstance(shadow_reason, str) and shadow_reason:
        _counter(stats, "shadow_reason_counts").update([shadow_reason])
    stats["residual_delta_total"] = float(stats["residual_delta_total"]) + abs(
        residual_delta
    )
    stats["residual_delta_abs_max"] = max(
        float(stats["residual_delta_abs_max"]),
        abs(residual_delta),
    )
    if survivor_trajectory and logged_action in neural_scores:
        rank = _score_rank(neural_scores, logged_action)
        _counter(stats, "logged_survivor_action_rank_counts").update([str(rank)])
        stats["logged_survivor_action_rank_total"] = (
            int(stats["logged_survivor_action_rank_total"]) + rank
        )
        stats["logged_survivor_action_rank_count"] = (
            int(stats["logged_survivor_action_rank_count"]) + 1
        )


def _finalize_score_stats(
    stats: Mapping[str, object],
    *,
    split_name: str,
) -> dict[str, object]:
    decision_count = int(stats.get("decision_count", 0))
    survivor_rank_count = int(stats.get("logged_survivor_action_rank_count", 0))
    scores_by_action = _mapping(stats.get("neural_scores_by_action"))
    variances = {
        action: _round(_variance(values if isinstance(values, list) else []))
        for action, values in sorted(scores_by_action.items())
        if isinstance(values, list) and values
    }
    return {
        "split": split_name,
        "decision_count": decision_count,
        "load_failure_count": int(stats.get("load_failures", 0)),
        "malformed_record_count": int(stats.get("malformed_records", 0)),
        "neural_score_variance_by_action": variances,
        "neural_score_variance_abs_max": max(variances.values(), default=0.0),
        "neural_top_action_counts": _counter_dict(stats, "neural_top_action_counts"),
        "linear_top_action_counts": _counter_dict(stats, "linear_top_action_counts"),
        "configured_blended_top_action_counts": _counter_dict(
            stats,
            "configured_blended_top_action_counts",
        ),
        "shadow_forced_gate_top_action_counts": _counter_dict(
            stats,
            "shadow_forced_gate_top_action_counts",
        ),
        "shadow_margin_ignored_top_action_counts": _counter_dict(
            stats,
            "shadow_margin_ignored_top_action_counts",
        ),
        "logged_survivor_action_rank_mean": _round(
            int(stats.get("logged_survivor_action_rank_total", 0))
            / survivor_rank_count
        )
        if survivor_rank_count
        else None,
        "logged_survivor_action_rank_counts": _counter_dict(
            stats,
            "logged_survivor_action_rank_counts",
        ),
        "residual_delta_abs_mean": _round(
            float(stats.get("residual_delta_total", 0.0)) / decision_count
        )
        if decision_count
        else 0.0,
        "residual_delta_abs_max": _round(stats.get("residual_delta_abs_max", 0.0)),
        "artifact_vs_linear_action_agreement_count": int(
            stats.get("artifact_linear_agreement_count", 0)
        ),
        "artifact_vs_linear_action_agreement_share": _share(
            int(stats.get("artifact_linear_agreement_count", 0)),
            decision_count,
        ),
        "configured_would_change_count": int(
            stats.get("configured_would_change_count", 0)
        ),
        "shadow_forced_gate_would_change_count": int(
            stats.get("shadow_forced_gate_would_change_count", 0)
        ),
        "shadow_margin_ignored_would_change_count": int(
            stats.get("shadow_margin_ignored_would_change_count", 0)
        ),
        "neural_top_vs_linear_top_disagreement_count": int(
            stats.get("neural_linear_disagreement_count", 0)
        ),
        "residual_applied_count": int(stats.get("residual_applied_count", 0)),
        "actual_changed_linear_count": int(
            stats.get("actual_changed_linear_count", 0)
        ),
        "shadow_reason_counts": _counter_dict(stats, "shadow_reason_counts"),
    }


def _fixture_replay_diagnostics(
    *,
    artifact: dict[str, object],
    seeds: Sequence[int],
    ticks: int,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    with TemporaryDirectory() as tmpdir:
        suite = run_mind_v3_fixture_suite(
            suite="basic",
            fixture_names=["carrion_only"],
            seeds=list(seeds),
            ticks=ticks,
            founder_template=None,
            neural_artifact=artifact,
            trajectory_output_dir=Path(tmpdir),
            trajectory_prefix="residual_activation_audit",
        )
        trajectories = _fixture_trajectory_paths(suite)
        diagnostics = []
        records_by_path = []
        for path in trajectories:
            header, records = _load_trajectory_payload(path)
            seed = _int(_mapping(header.get("config")).get("seed"))
            records = tuple(_record_with_audit_seed(record, seed) for record in records)
            records_by_path.extend(records)
            diagnostics.extend(
                record.get("policy_decision_diagnostics")
                for record in records
                if isinstance(record.get("policy_decision_diagnostics"), Mapping)
            )
    return (
        _fixture_diagnostic_summary(
            diagnostics,
            suite=suite,
            seeds=seeds,
            ticks=ticks,
        ),
        tuple(records_by_path),
    )


def _fixture_trajectory_paths(suite: Mapping[str, object]) -> list[Path]:
    paths = []
    fixtures = suite.get("fixtures")
    for fixture in fixtures if isinstance(fixtures, list) else []:
        if not isinstance(fixture, Mapping):
            continue
        comparison = _mapping(fixture.get("comparison"))
        learned = _mapping(comparison.get("mind_v3"))
        runs = learned.get("runs")
        for run in runs if isinstance(runs, list) else []:
            if not isinstance(run, Mapping):
                continue
            path = run.get("trajectory_path")
            if isinstance(path, str) and path:
                paths.append(Path(path))
    return paths


def _fixture_diagnostic_summary(
    diagnostics: Sequence[object],
    *,
    suite: Mapping[str, object],
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    stats = Counter()
    gate_reasons = Counter()
    shadow_reasons = Counter()
    effective_scale_buckets = Counter()
    linear_margin_buckets = Counter()
    recovery_buckets = Counter()
    for item in diagnostics:
        if not isinstance(item, Mapping):
            continue
        if not isinstance(item.get("neural_linear_anchor_policy"), str):
            continue
        stats["decision_count"] += 1
        if item.get("neural_residual_context_gate_passed") is True:
            stats["context_gate_pass_count"] += 1
        else:
            stats["context_gate_fail_count"] += 1
        gate_reasons.update([str(item.get("neural_residual_context_gate_reason", "unknown"))])
        shadow_reasons.update([str(item.get("neural_residual_shadow_reason", "unknown"))])
        effective_scale_buckets.update(
            [_effective_scale_bucket(_float(item.get("neural_residual_effective_scale")))]
        )
        linear_margin_buckets.update(
            [_linear_margin_bucket(_float(item.get("linear_anchor_score_margin")))]
        )
        recovery_buckets.update(
            [_recovery_phase_bucket(_int(item.get("neural_residual_recovery_phase_remaining")))]
        )
        if item.get("neural_top_action") != item.get("linear_anchor_action"):
            stats["neural_top_linear_top_disagreement_count"] += 1
        if item.get("anchored_action") != item.get("linear_anchor_action"):
            stats["configured_residual_would_change_count"] += 1
        if item.get("neural_residual_changed_linear_action") is True:
            stats["actual_changed_linear_count"] += 1
        if item.get("neural_residual_applied") is True:
            stats["residual_applied_count"] += 1
    fixture = _mapping(suite.get("fixtures")[0]) if suite.get("fixtures") else {}
    learned = _mapping(_mapping(_mapping(fixture.get("comparison")).get("mind_v3")).get("aggregate"))
    return {
        "fixture": "carrion_only",
        "seeds": list(seeds),
        "ticks": ticks,
        "run_count": len(seeds),
        "terminal_alive_agents_mean": learned.get("alive_agents_mean"),
        "births_mean": learned.get("births_mean"),
        "heuristic_action_source_count": learned.get("heuristic_action_source_count"),
        "decision_count": int(stats["decision_count"]),
        "context_gate_pass_count": int(stats["context_gate_pass_count"]),
        "context_gate_fail_count": int(stats["context_gate_fail_count"]),
        "context_gate_reason_counts": dict(sorted(gate_reasons.items())),
        "recovery_phase_remaining_buckets": dict(sorted(recovery_buckets.items())),
        "safety_guard_shadow_reason_counts": dict(sorted(shadow_reasons.items())),
        "effective_scale_buckets": dict(sorted(effective_scale_buckets.items())),
        "linear_margin_buckets": dict(sorted(linear_margin_buckets.items())),
        "neural_top_vs_linear_top_disagreement_count": int(
            stats["neural_top_linear_top_disagreement_count"]
        ),
        "configured_residual_would_change_count": int(
            stats["configured_residual_would_change_count"]
        ),
        "actual_changed_linear_count": int(stats["actual_changed_linear_count"]),
        "residual_applied_count": int(stats["residual_applied_count"]),
    }


def _load_trajectory_records(path: str | Path) -> tuple[dict[str, object], ...]:
    _header, records = _load_trajectory_payload(path)
    return records


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
        raise CarrionRecoveryResidualAuditError(
            f"trajectory has no record payloads: {path}"
        )
    return header, tuple(records)


def _record_with_audit_seed(
    record: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    copied = dict(record)
    if seed:
        copied["__residual_audit_seed"] = seed
    return copied


def _looks_like_trajectory_record(payload: Mapping[str, object]) -> bool:
    return (
        "observation_input" in payload
        and "action_mask" in payload
        and "requested_action" in payload
    )


def _heldout_branch_state_decision_comparison(
    heldout_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "decision_count": heldout_summary.get("decision_count", 0),
        "artifact_vs_linear_action_agreement_count": heldout_summary.get(
            "artifact_vs_linear_action_agreement_count",
            0,
        ),
        "artifact_vs_linear_action_agreement_share": heldout_summary.get(
            "artifact_vs_linear_action_agreement_share",
            0.0,
        ),
        "survivor_logged_action_rank_mean": heldout_summary.get(
            "logged_survivor_action_rank_mean"
        ),
        "survivor_logged_action_rank_counts": heldout_summary.get(
            "logged_survivor_action_rank_counts",
            {},
        ),
        "configured_would_change_count": heldout_summary.get(
            "configured_would_change_count",
            0,
        ),
        "shadow_forced_gate_would_change_count": heldout_summary.get(
            "shadow_forced_gate_would_change_count",
            0,
        ),
        "shadow_margin_ignored_would_change_count": heldout_summary.get(
            "shadow_margin_ignored_would_change_count",
            0,
        ),
    }


def _counter_reconciliation(
    *,
    evaluation_report: Mapping[str, object] | None,
    distill_report: Mapping[str, object],
    activation_audit: Mapping[str, object] | None,
    heldout_artifact_replay: Mapping[str, object],
    current_fixture_replay: Mapping[str, object],
) -> dict[str, object]:
    surfaces: list[dict[str, object]] = []
    if evaluation_report is not None:
        surfaces.extend(_evaluation_report_surfaces(evaluation_report))
    heldout_logged = _mapping(distill_report.get("heldout_branch_state_evaluation"))
    heldout_balance = _mapping(heldout_logged.get("aggregate_action_balance"))
    if heldout_balance:
        surfaces.append(
            _surface_from_action_balance(
                name="heldout_branch_logged_trajectories",
                action_balance=heldout_balance,
                evidence_mode="logged_trajectory_evidence",
                live_artifact_replay=False,
            )
        )
    surfaces.append(
        _surface_from_offline_summary(
            name="heldout_branch_artifact_decision_replay",
            summary=heldout_artifact_replay,
            evidence_mode="offline_artifact_decision_replay",
        )
    )
    fixture_source = (
        _mapping(activation_audit.get("fixture_replay_diagnostics"))
        if activation_audit is not None
        else current_fixture_replay
    )
    surfaces.append(
        _surface_from_fixture_summary(
            name="v66_fixture_replay",
            summary=fixture_source,
            evidence_mode="live_artifact_replay",
        )
    )
    classification = _mismatch_classification(surfaces)
    return {
        "policy": "diagnostics_only_residual_counter_reconciliation_v1",
        "surface_count": len(surfaces),
        "surfaces": surfaces,
        "classification": classification,
        "diagnostics_only": True,
    }


def _evaluation_report_surfaces(
    evaluation_report: Mapping[str, object],
) -> list[dict[str, object]]:
    surfaces: list[dict[str, object]] = []
    action_balance = _mapping(evaluation_report.get("action_balance_diagnostics"))
    open_balance = _mapping(_mapping(action_balance.get("open")).get("mind_v3_recovery_distilled"))
    open_aggregate = _mapping(
        _mapping(
            _mapping(_mapping(evaluation_report.get("open")).get("comparison")).get(
                "mind_v3_recovery_distilled"
            )
        ).get("aggregate")
    )
    if open_balance or open_aggregate:
        surfaces.append(
            _surface_from_evaluation_aggregate(
                name="broad_open_evaluation",
                aggregate=open_aggregate,
                action_balance=open_balance,
                evidence_mode="live_evaluation_aggregate",
            )
        )
    fixture_balance = _mapping(
        _mapping(
            _mapping(action_balance.get("fixture")).get("mind_v3_recovery_distilled")
        ).get("carrion_only")
    )
    fixture_aggregate = _candidate_fixture_aggregate(evaluation_report)
    if fixture_balance or fixture_aggregate:
        surfaces.append(
            _surface_from_evaluation_aggregate(
                name="carrion_fixture_evaluation",
                aggregate=fixture_aggregate,
                action_balance=fixture_balance,
                evidence_mode="live_fixture_evaluation_aggregate",
            )
        )
    return surfaces


def _candidate_fixture_aggregate(
    evaluation_report: Mapping[str, object],
) -> Mapping[str, object]:
    fixtures = _mapping(
        _mapping(evaluation_report.get("fixture")).get("candidate_suite")
    ).get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        return {}
    fixture = _mapping(fixtures[0])
    return _mapping(
        _mapping(_mapping(fixture.get("comparison")).get("mind_v3")).get("aggregate")
    )


def _surface_from_evaluation_aggregate(
    *,
    name: str,
    aggregate: Mapping[str, object],
    action_balance: Mapping[str, object],
    evidence_mode: str,
) -> dict[str, object]:
    neural = _mapping(aggregate.get("neural_anchor_diagnostics"))
    decision_count = _int(neural.get("decision_count"))
    action_balance_changed = _int(action_balance.get("changed_linear_decision_count"))
    neural_changed = _int(neural.get("changed_linear_action_count"))
    requested_count = _counter_total(
        _mapping(aggregate.get("requested_action_counts"))
        or _mapping(action_balance.get("requested_action_counts"))
    )
    context_gate_fail = _context_gate_fail_count(
        _mapping(neural.get("shadow_reason_counts"))
    )
    surface = {
        "surface": name,
        "evidence_mode": evidence_mode,
        "logged_trajectory_evidence": False,
        "live_artifact_replay": True,
        "record_count": _int(aggregate.get("trajectory_record_count"))
        or _int(action_balance.get("record_count")),
        "decision_diagnostics_present_count": decision_count,
        "neural_diagnostics_present_count": decision_count,
        "context_gate_pass_count": max(0, decision_count - context_gate_fail),
        "context_gate_fail_count": context_gate_fail,
        "effective_scale_positive_count": _int(neural.get("residual_applied_count")),
        "effective_scale_zero_count": max(
            0,
            decision_count - _int(neural.get("residual_applied_count")),
        ),
        "shadow_reason_counts": _mapping(neural.get("shadow_reason_counts")),
        "linear_anchor_action_present_count": _counter_total(
            _mapping(neural.get("linear_anchor_action_counts"))
        ),
        "requested_action_present_count": requested_count,
        "actual_changed_linear_count": action_balance_changed,
        "configured_would_change_count": neural_changed,
        "reported_action_balance_changed_linear_count": action_balance_changed,
        "neural_anchor_changed_linear_action_count": neural_changed,
        "residual_application_count": _int(action_balance.get("residual_application_count")),
        "neural_anchor_residual_applied_count": _int(neural.get("residual_applied_count")),
        "counter_mismatch": action_balance_changed != neural_changed,
    }
    return surface


def _surface_from_action_balance(
    *,
    name: str,
    action_balance: Mapping[str, object],
    evidence_mode: str,
    live_artifact_replay: bool,
) -> dict[str, object]:
    record_count = _int(action_balance.get("record_count"))
    missing = _int(action_balance.get("missing_decision_diagnostics_count"))
    return {
        "surface": name,
        "evidence_mode": evidence_mode,
        "logged_trajectory_evidence": not live_artifact_replay,
        "live_artifact_replay": live_artifact_replay,
        "record_count": record_count,
        "decision_diagnostics_present_count": max(0, record_count - missing),
        "neural_diagnostics_present_count": 0,
        "context_gate_pass_count": 0,
        "context_gate_fail_count": 0,
        "effective_scale_positive_count": _int(
            action_balance.get("residual_application_count")
        ),
        "effective_scale_zero_count": max(
            0,
            record_count - _int(action_balance.get("residual_application_count")),
        ),
        "shadow_reason_counts": {},
        "linear_anchor_action_present_count": 0,
        "requested_action_present_count": _counter_total(
            _mapping(action_balance.get("requested_action_counts"))
        ),
        "actual_changed_linear_count": _int(
            action_balance.get("changed_linear_decision_count")
        ),
        "configured_would_change_count": _int(
            action_balance.get("changed_linear_decision_count")
        ),
        "reported_action_balance_changed_linear_count": _int(
            action_balance.get("changed_linear_decision_count")
        ),
        "neural_anchor_changed_linear_action_count": 0,
        "residual_application_count": _int(
            action_balance.get("residual_application_count")
        ),
        "neural_anchor_residual_applied_count": 0,
        "counter_mismatch": False,
    }


def _surface_from_offline_summary(
    *,
    name: str,
    summary: Mapping[str, object],
    evidence_mode: str,
) -> dict[str, object]:
    decision_count = _int(summary.get("decision_count"))
    shadow_counts = _mapping(summary.get("shadow_reason_counts"))
    gate_fail = _context_gate_fail_count(shadow_counts)
    changed = _int(summary.get("actual_changed_linear_count"))
    configured = _int(summary.get("configured_would_change_count"))
    return {
        "surface": name,
        "evidence_mode": evidence_mode,
        "logged_trajectory_evidence": False,
        "live_artifact_replay": evidence_mode == "live_artifact_replay",
        "record_count": decision_count,
        "decision_diagnostics_present_count": decision_count,
        "neural_diagnostics_present_count": decision_count,
        "context_gate_pass_count": max(0, decision_count - gate_fail),
        "context_gate_fail_count": gate_fail,
        "effective_scale_positive_count": _int(summary.get("residual_applied_count")),
        "effective_scale_zero_count": max(
            0,
            decision_count - _int(summary.get("residual_applied_count")),
        ),
        "shadow_reason_counts": shadow_counts,
        "linear_anchor_action_present_count": decision_count,
        "requested_action_present_count": decision_count,
        "actual_changed_linear_count": changed,
        "configured_would_change_count": configured,
        "reported_action_balance_changed_linear_count": None,
        "neural_anchor_changed_linear_action_count": configured,
        "residual_application_count": _int(summary.get("residual_applied_count")),
        "neural_anchor_residual_applied_count": _int(summary.get("residual_applied_count")),
        "counter_mismatch": changed != configured,
    }


def _surface_from_fixture_summary(
    *,
    name: str,
    summary: Mapping[str, object],
    evidence_mode: str,
) -> dict[str, object]:
    decision_count = _int(summary.get("decision_count"))
    changed = _int(summary.get("actual_changed_linear_count"))
    configured = _int(summary.get("configured_residual_would_change_count"))
    return {
        "surface": name,
        "evidence_mode": evidence_mode,
        "logged_trajectory_evidence": False,
        "live_artifact_replay": True,
        "record_count": decision_count,
        "decision_diagnostics_present_count": decision_count,
        "neural_diagnostics_present_count": decision_count,
        "context_gate_pass_count": _int(summary.get("context_gate_pass_count")),
        "context_gate_fail_count": _int(summary.get("context_gate_fail_count")),
        "effective_scale_positive_count": _int(summary.get("residual_applied_count")),
        "effective_scale_zero_count": max(
            0,
            decision_count - _int(summary.get("residual_applied_count")),
        ),
        "shadow_reason_counts": _mapping(summary.get("safety_guard_shadow_reason_counts")),
        "linear_anchor_action_present_count": decision_count,
        "requested_action_present_count": decision_count,
        "actual_changed_linear_count": changed,
        "configured_would_change_count": configured,
        "reported_action_balance_changed_linear_count": None,
        "neural_anchor_changed_linear_action_count": configured,
        "residual_application_count": _int(summary.get("residual_applied_count")),
        "neural_anchor_residual_applied_count": _int(summary.get("residual_applied_count")),
        "counter_mismatch": changed != configured,
    }


def _mismatch_classification(
    surfaces: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    evidence: list[str] = []
    category_scores = {
        "expected surface mismatch": 0,
        "reporting bug": 0,
        "missing diagnostics": 0,
        "evaluation path mismatch": 0,
    }
    for surface in surfaces:
        name = str(surface.get("surface", "unknown"))
        record_count = _int(surface.get("record_count"))
        neural_count = _int(surface.get("neural_diagnostics_present_count"))
        configured = _int(surface.get("configured_would_change_count"))
        actual = _int(surface.get("actual_changed_linear_count"))
        if record_count > 0 and bool(surface.get("live_artifact_replay")) and neural_count <= 0:
            category_scores["missing diagnostics"] += 5
            evidence.append(f"{name}:missing_neural_diagnostics")
        if (
            surface.get("reported_action_balance_changed_linear_count") is not None
            and configured > 0
            and actual == 0
            and neural_count > 0
        ):
            category_scores["reporting bug"] += 6
            evidence.append(f"{name}:action_balance_zero_but_neural_changed")
        if bool(surface.get("logged_trajectory_evidence")) and configured == 0:
            category_scores["expected surface mismatch"] += 1
            evidence.append(f"{name}:logged_or_scripted_surface_not_artifact_replay")
        if configured > 0 and actual > 0 and configured != actual:
            category_scores["evaluation path mismatch"] += 2
            evidence.append(f"{name}:configured_actual_changed_disagree")
    primary = sorted(
        category_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]
    return {
        "primary": primary,
        "category_scores": category_scores,
        "evidence": evidence,
        "diagnostic_only": True,
    }


def _shadow_calibration(
    *,
    artifact: Mapping[str, object],
    heldout_rows: Sequence[Mapping[str, object]],
    fixture_records: Sequence[Mapping[str, object]],
    margin_sweep: Sequence[float],
    scale_sweep: Sequence[float],
) -> dict[str, object]:
    heldout_snapshots = _calibration_snapshots_for_rows(
        rows=heldout_rows,
        artifact=artifact,
    )
    fixture_snapshots = _calibration_snapshots_for_records(
        records=fixture_records,
        artifact=artifact,
    )
    configured_margin = _float(
        artifact.get("neural_residual_max_linear_override_margin")
    )
    configured_scale = _float(artifact.get("neural_residual_scale"))
    sections = {
        "heldout_branch_states": _shadow_calibration_for_snapshots(
            heldout_snapshots,
            configured_margin=configured_margin,
            configured_scale=configured_scale,
            margin_sweep=margin_sweep,
            scale_sweep=scale_sweep,
        ),
        "carrion_fixture_rows": _shadow_calibration_for_snapshots(
            fixture_snapshots,
            configured_margin=configured_margin,
            configured_scale=configured_scale,
            margin_sweep=margin_sweep,
            scale_sweep=scale_sweep,
        ),
    }
    strongest = _strongest_shadow_setting(sections)
    return {
        "policy": "diagnostics_only_residual_shadow_margin_scale_calibration_v1",
        "configured_margin": _round(configured_margin),
        "configured_scale": _round(configured_scale),
        "margin_sweep": [_round(value) for value in margin_sweep],
        "scale_sweep": [_round(value) for value in scale_sweep],
        "heldout_branch_state_sampling_policy": (
            "first_record_per_split_trajectory_v1"
        ),
        "carrion_fixture_row_sampling_policy": "all_fixture_decision_rows_v1",
        "heldout_branch_states": sections["heldout_branch_states"],
        "carrion_fixture_rows": sections["carrion_fixture_rows"],
        "strongest_shadow_setting": strongest,
        "diagnostics_only": True,
    }


def _calibration_snapshots_for_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    artifact: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    snapshots: list[dict[str, object]] = []
    for row in rows:
        path = row.get("trajectory_path")
        if not isinstance(path, str) or not path:
            continue
        try:
            records = _load_trajectory_records(path)
        except Exception:
            continue
        if records:
            records = records[:1]
        policy = MindV3EvolutionPolicy(
            seed=_int(row.get("seed")),
            neural_artifact=dict(artifact),
        )
        for record in records:
            snapshot = _calibration_snapshot(
                record,
                artifact=artifact,
                policy=policy,
                seed=_int(row.get("seed")),
            )
            if snapshot is not None:
                snapshots.append(snapshot)
            policy.observe_transition(dict(record))
    return tuple(snapshots)


def _calibration_snapshots_for_records(
    *,
    records: Sequence[Mapping[str, object]],
    artifact: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    snapshots: list[dict[str, object]] = []
    policies: dict[int, MindV3EvolutionPolicy] = {}
    for record in records:
        seed = _int(record.get("__residual_audit_seed"))
        policy = policies.get(seed)
        if policy is None:
            policy = MindV3EvolutionPolicy(seed=seed, neural_artifact=dict(artifact))
            policies[seed] = policy
        snapshot = _calibration_snapshot(
            record,
            artifact=artifact,
            policy=policy,
            seed=seed,
        )
        if snapshot is not None:
            snapshots.append(snapshot)
        policy.observe_transition(dict(record))
    return tuple(snapshots)


def _calibration_snapshot(
    record: Mapping[str, object],
    *,
    artifact: Mapping[str, object],
    policy: MindV3EvolutionPolicy,
    seed: int,
) -> dict[str, object] | None:
    observation_input = record.get("observation_input")
    action_mask = record.get("action_mask")
    if not isinstance(observation_input, Mapping) or not isinstance(action_mask, Mapping):
        return None
    observation = {
        "metadata": record.get("observation_metadata", {}),
        "observation_input": dict(observation_input),
    }
    decision = policy.decide(observation, {str(k): bool(v) for k, v in action_mask.items()})
    diagnostics = decision.diagnostics
    agent_id = _int(record.get("agent_id"))
    metadata = policy.agent_mind_metadata(agent_id=agent_id)
    observation_values = decode_observation_input(dict(observation_input))
    mask = {str(k): bool(v) for k, v in action_mask.items()}
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
    linear_action, _linear_score = _best_action(linear_scores, mask)
    neural_action, _neural_score = _best_action(neural_scores, mask)
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
    return {
        "seed": seed,
        "requested_action": str(record.get("requested_action", "unknown")),
        "linear_action": linear_action,
        "neural_action": neural_action,
        "linear_margin": _score_margin(linear_scores, mask),
        "gate": gate,
        "gate_passed": bool(gate_status.get("passed")),
        "gate_reason": str(gate_status.get("reason", "unknown")),
        "safety_guard_reason": safety_guard or "none",
        "recovery_phase_remaining": recovery_remaining,
        "action_mask": mask,
        "linear_scores": {str(k): _float(v) for k, v in linear_scores.items()},
        "neural_scores": {str(k): _float(v) for k, v in neural_scores.items()},
    }


def _shadow_calibration_for_snapshots(
    snapshots: Sequence[Mapping[str, object]],
    *,
    configured_margin: float,
    configured_scale: float,
    margin_sweep: Sequence[float],
    scale_sweep: Sequence[float],
) -> dict[str, object]:
    configured = _calibration_result(
        snapshots,
        label="configured",
        margin=configured_margin,
        scale=configured_scale,
    )
    forced_gate = _calibration_result(
        snapshots,
        label="forced_gate",
        margin=configured_margin,
        scale=configured_scale,
        force_gate=True,
    )
    margin_ignored = _calibration_result(
        snapshots,
        label="margin_ignored",
        margin=float("inf"),
        scale=configured_scale,
    )
    margin_results = [
        _calibration_result(
            snapshots,
            label=f"margin_{_round(margin):g}",
            margin=margin,
            scale=configured_scale,
        )
        for margin in margin_sweep
    ]
    scale_results = [
        _calibration_result(
            snapshots,
            label=f"scale_{_round(scale):g}",
            margin=configured_margin,
            scale=scale,
        )
        for scale in scale_sweep
    ]
    return {
        "snapshot_count": len(snapshots),
        "configured": configured,
        "forced_gate": forced_gate,
        "margin_ignored": margin_ignored,
        "margin_threshold_sweep": margin_results,
        "scale_sweep": scale_results,
    }


def _calibration_result(
    snapshots: Sequence[Mapping[str, object]],
    *,
    label: str,
    margin: float,
    scale: float,
    force_gate: bool = False,
) -> dict[str, object]:
    stats = {
        "decision_count": 0,
        "would_change_count": 0,
        "linear_action_counts": Counter(),
        "final_action_counts": Counter(),
        "would_change_by_action": Counter(),
        "would_change_by_seed": Counter(),
        "linear_margin_bucket_outcomes": {},
        "safety_guard_reason_counts": Counter(),
        "shadow_reason_counts": Counter(),
        "context_gate_pass_count": 0,
        "context_gate_fail_count": 0,
    }
    for snapshot in snapshots:
        result = _calibrated_action(
            snapshot,
            margin=margin,
            scale=scale,
            force_gate=force_gate,
        )
        linear_action = str(snapshot.get("linear_action", "none"))
        final_action = str(result["final_action"])
        seed = str(_int(snapshot.get("seed")))
        changed = final_action != linear_action
        stats["decision_count"] = int(stats["decision_count"]) + 1
        _counter(stats, "linear_action_counts").update([linear_action])
        _counter(stats, "final_action_counts").update([final_action])
        _counter(stats, "safety_guard_reason_counts").update(
            [str(snapshot.get("safety_guard_reason", "unknown"))]
        )
        _counter(stats, "shadow_reason_counts").update([str(result["shadow_reason"])])
        if bool(result["gate_passed"]):
            stats["context_gate_pass_count"] = int(stats["context_gate_pass_count"]) + 1
        else:
            stats["context_gate_fail_count"] = int(stats["context_gate_fail_count"]) + 1
        bucket = _linear_margin_bucket(_float(snapshot.get("linear_margin")))
        bucket_entry = _mapping(stats["linear_margin_bucket_outcomes"].get(bucket))
        bucket_counts = {
            "decision_count": _int(bucket_entry.get("decision_count")) + 1,
            "would_change_count": _int(bucket_entry.get("would_change_count"))
            + (1 if changed else 0),
        }
        stats["linear_margin_bucket_outcomes"][bucket] = bucket_counts
        if changed:
            stats["would_change_count"] = int(stats["would_change_count"]) + 1
            _counter(stats, "would_change_by_action").update([final_action])
            _counter(stats, "would_change_by_seed").update([seed])
    decision_count = int(stats["decision_count"])
    final_counts = _counter_dict(stats, "final_action_counts")
    linear_counts = _counter_dict(stats, "linear_action_counts")
    dominant_action, dominant_count = _dominant_action(final_counts)
    return {
        "label": label,
        "margin": None if not math.isfinite(margin) else _round(margin),
        "scale": _round(scale),
        "force_gate": force_gate,
        "decision_count": decision_count,
        "would_change_count": int(stats["would_change_count"]),
        "would_change_share": _share(int(stats["would_change_count"]), decision_count),
        "would_change_by_action": _counter_dict(stats, "would_change_by_action"),
        "would_change_by_seed": _counter_dict(stats, "would_change_by_seed"),
        "linear_action_counts": linear_counts,
        "final_action_counts": final_counts,
        "action_balance_delta": _action_balance_delta(final_counts, linear_counts),
        "dominant_final_action": dominant_action,
        "dominant_final_action_share": _share(dominant_count, decision_count),
        "action_collapse_risk": _share(dominant_count, decision_count) > 0.5,
        "context_gate_pass_count": int(stats["context_gate_pass_count"]),
        "context_gate_fail_count": int(stats["context_gate_fail_count"]),
        "linear_margin_bucket_outcomes": {
            str(key): value
            for key, value in sorted(stats["linear_margin_bucket_outcomes"].items())
        },
        "safety_guard_reason_counts": _counter_dict(
            stats,
            "safety_guard_reason_counts",
        ),
        "shadow_reason_counts": _counter_dict(stats, "shadow_reason_counts"),
    }


def _calibrated_action(
    snapshot: Mapping[str, object],
    *,
    margin: float,
    scale: float,
    force_gate: bool,
) -> dict[str, object]:
    mask = _mapping(snapshot.get("action_mask"))
    linear_scores = {
        str(k): _float(v)
        for k, v in _mapping(snapshot.get("linear_scores")).items()
    }
    neural_scores = {
        str(k): _float(v)
        for k, v in _mapping(snapshot.get("neural_scores")).items()
    }
    legal_actions = tuple(action for action in sorted(mask) if bool(mask[action]))
    gate_passed = force_gate or bool(snapshot.get("gate_passed"))
    safety_guard = str(snapshot.get("safety_guard_reason", "none"))
    residual_scale = max(0.0, scale) if gate_passed and safety_guard == "none" else 0.0
    scores = _blend_neural_with_linear_anchor(
        neural_scores=neural_scores,
        linear_scores=linear_scores,
        action_mask={str(k): bool(v) for k, v in mask.items()},
        residual_scale=residual_scale,
        max_linear_override_margin=margin,
    )
    final_action, _final_score = _best_action(scores, {str(k): bool(v) for k, v in mask.items()})
    neural_normalized = _normalized_legal_scores(neural_scores, legal_actions)
    blend_shadow_reason = _neural_residual_shadow_reason(
        neural_normalized,
        linear_scores=linear_scores,
        max_linear_override_margin=margin,
    )
    if not gate_passed:
        shadow_reason = f"context_gate:{snapshot.get('gate', 'unknown')}"
    elif safety_guard != "none":
        shadow_reason = safety_guard
    elif residual_scale <= 0.0:
        shadow_reason = "zero_scale"
    else:
        shadow_reason = blend_shadow_reason
    return {
        "final_action": final_action,
        "gate_passed": gate_passed,
        "shadow_reason": shadow_reason,
    }


def _calibration_classification(
    *,
    feature_checks: Mapping[str, object],
    failure_classification: Mapping[str, object],
    shadow_calibration: Mapping[str, object],
) -> dict[str, object]:
    scores = {
        "gate issue": 0,
        "margin domination": 0,
        "training weakness": 0,
        "schema issue": 0,
    }
    evidence: list[str] = []
    if not bool(feature_checks.get("passed", False)):
        scores["schema issue"] += 10
        evidence.append("feature_contract_check_failed")
    if failure_classification.get("primary") == "training failure":
        scores["training weakness"] += 5
        evidence.append("neural_training_signal_weak")
    for section_name in ("heldout_branch_states", "carrion_fixture_rows"):
        section = _mapping(shadow_calibration.get(section_name))
        configured = _mapping(section.get("configured"))
        forced_gate = _mapping(section.get("forced_gate"))
        margin_ignored = _mapping(section.get("margin_ignored"))
        if _int(forced_gate.get("would_change_count")) > _int(
            configured.get("would_change_count")
        ):
            scores["gate issue"] += 2
            evidence.append(f"{section_name}:forced_gate_increases_changes")
        if _int(margin_ignored.get("would_change_count")) > _int(
            configured.get("would_change_count")
        ):
            scores["margin domination"] += 3
            evidence.append(f"{section_name}:margin_ignored_increases_changes")
        shadow_reasons = _mapping(configured.get("shadow_reason_counts"))
        if _int(shadow_reasons.get("linear_margin_guard")) > 0:
            scores["margin domination"] += 3
            evidence.append(f"{section_name}:linear_margin_guard_present")
    primary = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return {
        "primary": primary,
        "category_scores": scores,
        "evidence": evidence,
        "diagnostic_only": True,
    }


def _failure_classification(
    *,
    feature_checks: Mapping[str, object],
    offline: Mapping[str, object],
    fixture_replay: Mapping[str, object],
    distill_report: Mapping[str, object],
) -> dict[str, object]:
    heldout = _mapping(offline.get("heldout"))
    train = _mapping(offline.get("train"))
    distill_acceptance = _mapping(distill_report.get("acceptance"))
    blockers = []
    category_scores = {
        "feature/schema mismatch": 0,
        "training failure": 0,
        "gating failure": 0,
        "margin domination": 0,
        "evaluation gap": 0,
    }
    if not bool(feature_checks.get("passed", False)):
        category_scores["feature/schema mismatch"] += 10
        blockers.append("feature_contract_check_failed")
    variance = max(
        _float(train.get("neural_score_variance_abs_max")),
        _float(heldout.get("neural_score_variance_abs_max")),
    )
    if variance <= 1e-10:
        category_scores["training failure"] += 5
        blockers.append("neural_scores_have_near_zero_variance")
    if _int(fixture_replay.get("context_gate_pass_count")) <= 0:
        category_scores["gating failure"] += 6
        blockers.append("fixture_context_gate_never_passed")
    if (
        _int(heldout.get("shadow_forced_gate_would_change_count"))
        > _int(heldout.get("configured_would_change_count"))
    ):
        category_scores["gating failure"] += 3
        blockers.append("forced_gate_shadow_changes_more_than_configured")
    shadow_reasons = _mapping(fixture_replay.get("safety_guard_shadow_reason_counts"))
    if _int(shadow_reasons.get("linear_margin_guard")) > 0:
        category_scores["margin domination"] += 5
        blockers.append("linear_margin_guard_shadowed_fixture_decisions")
    if (
        _int(heldout.get("shadow_margin_ignored_would_change_count"))
        > _int(heldout.get("configured_would_change_count"))
    ):
        category_scores["margin domination"] += 3
        blockers.append("margin_ignored_shadow_changes_more_than_configured")
    if _int(fixture_replay.get("decision_count")) <= 0:
        category_scores["evaluation gap"] += 5
        blockers.append("fixture_replay_produced_no_neural_diagnostics")
    if distill_acceptance.get("promotion_candidate_passed") is False:
        category_scores["evaluation gap"] += 1
    primary = sorted(
        category_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]
    return {
        "primary": primary,
        "category_scores": category_scores,
        "evidence": blockers,
        "distill_promotion_candidate_passed": distill_acceptance.get(
            "promotion_candidate_passed"
        ),
        "diagnostic_only": True,
    }


def _split_rows(
    split_report: Mapping[str, object],
    *,
    split: str,
) -> list[Mapping[str, object]]:
    rows = _mapping(split_report.get("records")).get(split)
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _score_rank(scores: Mapping[str, float], action: str) -> int:
    ranked = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
    for index, (candidate, _score) in enumerate(ranked, start=1):
        if candidate == action:
            return index
    return len(ranked) + 1


def _variance(values: Sequence[object]) -> float:
    parsed = [_float(value) for value in values]
    if not parsed:
        return 0.0
    mean = sum(parsed) / len(parsed)
    return sum((value - mean) ** 2 for value in parsed) / len(parsed)


def _counter(stats: dict[str, object], key: str) -> Counter[str]:
    counter = stats.get(key)
    if not isinstance(counter, Counter):
        counter = Counter()
        stats[key] = counter
    return counter


def _counter_dict(stats: Mapping[str, object], key: str) -> dict[str, int]:
    value = stats.get(key)
    if isinstance(value, Counter):
        return dict(sorted((str(k), int(v)) for k, v in value.items()))
    return {}


def _effective_scale_bucket(value: float) -> str:
    if value <= 0.0:
        return "zero"
    if value <= 0.01:
        return "positive_le_0.01"
    if value <= 0.03:
        return "positive_le_0.03"
    return "positive_gt_0.03"


def _linear_margin_bucket(value: float) -> str:
    if value <= 0.008:
        return "le_configured_margin"
    if value <= 0.05:
        return "le_0.05"
    if value <= 0.15:
        return "le_0.15"
    return "gt_0.15"


def _recovery_phase_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 2:
        return "1_2"
    if value <= 8:
        return "3_8"
    return "gt_8"


def _counter_total(values: Mapping[str, object]) -> int:
    return sum(_int(value) for value in values.values())


def _context_gate_fail_count(shadow_reason_counts: Mapping[str, object]) -> int:
    return sum(
        _int(count)
        for reason, count in shadow_reason_counts.items()
        if str(reason).startswith("context_gate:")
    )


def _score_margin(
    scores: Mapping[str, float],
    action_mask: Mapping[str, bool],
) -> float:
    legal_scores = sorted(
        (
            _float(scores.get(action))
            for action in action_mask
            if bool(action_mask[action])
        ),
        reverse=True,
    )
    if len(legal_scores) < 2:
        return 0.0
    return max(0.0, legal_scores[0] - legal_scores[1])


def _dominant_action(action_counts: Mapping[str, object]) -> tuple[str | None, int]:
    if not action_counts:
        return None, 0
    action, count = sorted(
        ((str(action), _int(count)) for action, count in action_counts.items()),
        key=lambda item: (-item[1], item[0]),
    )[0]
    return action, count


def _action_balance_delta(
    final_counts: Mapping[str, object],
    linear_counts: Mapping[str, object],
) -> dict[str, int]:
    return {
        "drink": _int(final_counts.get("drink")) - _int(linear_counts.get("drink")),
        "eat": _int(final_counts.get("eat")) - _int(linear_counts.get("eat")),
        "stay": _int(final_counts.get("stay")) - _int(linear_counts.get("stay")),
        "move": _action_family_count(final_counts, "move_")
        - _action_family_count(linear_counts, "move_"),
        "attack": _action_family_count(final_counts, "attack_")
        - _action_family_count(linear_counts, "attack_"),
    }


def _action_family_count(
    action_counts: Mapping[str, object],
    prefix: str,
) -> int:
    return sum(
        _int(count)
        for action, count in action_counts.items()
        if str(action).startswith(prefix)
    )


def _strongest_shadow_setting(
    sections: Mapping[str, object],
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for section_name, section_value in sections.items():
        section = _mapping(section_value)
        for key in ("configured", "forced_gate", "margin_ignored"):
            result = _mapping(section.get(key))
            if result:
                candidates.append(_strongest_candidate(section_name, key, result))
        for key in ("margin_threshold_sweep", "scale_sweep"):
            values = section.get(key)
            for result in values if isinstance(values, list) else []:
                if isinstance(result, Mapping):
                    candidates.append(
                        _strongest_candidate(section_name, key, result)
                    )
    if not candidates:
        return {
            "surface": None,
            "setting": None,
            "would_change_count": 0,
            "action_collapse_risk": False,
        }
    return sorted(
        candidates,
        key=lambda item: (
            -_int(item.get("would_change_count")),
            bool(item.get("action_collapse_risk")),
            str(item.get("surface")),
            str(item.get("label")),
        ),
    )[0]


def _strongest_candidate(
    section_name: str,
    setting: str,
    result: Mapping[str, object],
) -> dict[str, object]:
    return {
        "surface": section_name,
        "setting": setting,
        "label": result.get("label"),
        "margin": result.get("margin"),
        "scale": result.get("scale"),
        "force_gate": result.get("force_gate"),
        "would_change_count": result.get("would_change_count", 0),
        "would_change_share": result.get("would_change_share", 0.0),
        "dominant_final_action": result.get("dominant_final_action"),
        "dominant_final_action_share": result.get("dominant_final_action_share", 0.0),
        "action_collapse_risk": bool(result.get("action_collapse_risk")),
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: object, field: str) -> int:
    parsed = _int(value)
    if parsed <= 0:
        raise CarrionRecoveryResidualAuditError(f"{field} must be positive")
    return parsed


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _round(value: object) -> float:
    return round(_float(value), 6)


def _share(count: int, total: int) -> float:
    return _round(count / total) if total > 0 else 0.0


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
