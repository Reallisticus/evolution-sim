from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO

from evolution_sim.cli.mind_v3_evaluate import run_mind_v3_fixture_suite
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
    _neural_residual_context_gate_status,
    _neural_residual_safety_guard_reason,
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
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = None,
    split_report: Mapping[str, object] | None = None,
    split_report_path: str | Path | None = None,
    fixture_seeds: Sequence[int] = (),
    fixture_ticks: int = 120,
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
    feature_checks = _feature_contract_checks(
        artifact=artifact,
        train_rows=train_rows,
        heldout_rows=heldout_rows,
    )
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
    fixture_replay = _fixture_replay_diagnostics(
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
    }
    return {
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
) -> dict[str, object]:
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
        for path in trajectories:
            records = _load_trajectory_records(path)
            diagnostics.extend(
                record.get("policy_decision_diagnostics")
                for record in records
                if isinstance(record.get("policy_decision_diagnostics"), Mapping)
            )
    return _fixture_diagnostic_summary(
        diagnostics,
        suite=suite,
        seeds=seeds,
        ticks=ticks,
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
    records: list[dict[str, object]] = []
    with _open_input(Path(path)) as handle:
        for line in handle:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                continue
            record = payload.get("record")
            if isinstance(record, Mapping):
                records.append(dict(record))
            elif _looks_like_trajectory_record(payload):
                records.append(dict(payload))
    if not records:
        raise CarrionRecoveryResidualAuditError(
            f"trajectory has no record payloads: {path}"
        )
    return tuple(records)


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
