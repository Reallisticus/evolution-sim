from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.cli.mind_v3_evaluate import (
    _aggregate_runs,
    _comparison_delta,
    _run_once,
    mind_v3_fixture_gate_config,
    mind_v3_fixture_gate_status,
    run_mind_v3_fixture_suite,
)
from evolution_sim.mind.carrion_recovery_archive import (
    MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
    MIND_V3_CARRION_RECOVERY_SPLIT_POLICY_BRANCH_DIGEST_SEED_STRATIFIED,
    MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION,
    load_carrion_recovery_json_report,
)
from evolution_sim.mind.dataset import load_trajectory_jsonl
from evolution_sim.mind.horizon_labels import (
    DEFAULT_HORIZON_TICKS,
    build_horizon_label_report,
    normalize_horizon_ticks,
    write_horizon_label_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_neural import (
    MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED,
    MIND_V3_NEURAL_ARTIFACT_MODE_HORIZON_FIXTURE,
    MIND_V3_NEURAL_DEFAULT_HIDDEN_UNITS,
    MIND_V3_NEURAL_DEFAULT_SEED,
    MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_DEFAULT_MAX_ABS,
    MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_DEFAULT_SCALE,
    MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_POLICY,
    train_mind_v3_neural_artifact,
    write_mind_v3_neural_artifact,
)
from evolution_sim.mind.v3_policy import (
    MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_NONE,
    MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_OR_RECOVERY_PHASE,
    MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_SCAVENGER,
    MindV3EvolutionPolicy,
)

MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION = (
    "mind_v3_carrion_recovery_distill_v1"
)
MIND_V3_CARRION_RECOVERY_DISTILL_POLICY = (
    "outcome_weighted_recovery_archive_horizon_distillation_v1"
)
MIND_V3_CARRION_RECOVERY_DISTILL_WEIGHT_POLICY = (
    "survivor_quality_failure_boundary_trajectory_weight_v1"
)
MIND_V3_CARRION_RECOVERY_DISTILL_UNIFORM_WEIGHT_POLICY = "uniform_trajectory_weight_v1"
DEFAULT_CARRION_RECOVERY_DISTILL_ARTIFACT_MODE = (
    MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED
)
DEFAULT_CARRION_RECOVERY_DISTILL_HIDDEN_UNITS = 16
DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_SCALE = 0.03
DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_MAX_LINEAR_OVERRIDE_MARGIN = 0.008
DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_CONTEXT_GATE = (
    MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_OR_RECOVERY_PHASE
)
DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_RECOVERY_PHASE_TICKS = 8
MIND_V3_CARRION_RECOVERY_PHASE_ACTION_BIAS_POLICY = (
    MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_POLICY
)
DEFAULT_CARRION_RECOVERY_PHASE_ACTION_BIAS_SCALE = (
    MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_DEFAULT_SCALE
)
DEFAULT_CARRION_RECOVERY_PHASE_ACTION_BIAS_MAX_ABS = (
    MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_DEFAULT_MAX_ABS
)
DEFAULT_CARRION_RECOVERY_DISTILL_EVAL_SEEDS: tuple[int, ...] = (29, 37)
DEFAULT_CARRION_RECOVERY_DISTILL_EVAL_TICKS = 120
DEFAULT_CARRION_RECOVERY_DISTILL_FIXTURES: tuple[str, ...] = ("carrion_only",)


class CarrionRecoveryDistillError(ValueError):
    pass


def build_carrion_recovery_distillation_report(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = None,
    archive_split_report: Mapping[str, object] | None = None,
    archive_split_report_path: str | Path | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZON_TICKS,
    artifact_mode: str = DEFAULT_CARRION_RECOVERY_DISTILL_ARTIFACT_MODE,
    hidden_units: int = DEFAULT_CARRION_RECOVERY_DISTILL_HIDDEN_UNITS,
    seed: int = MIND_V3_NEURAL_DEFAULT_SEED,
    neural_residual_scale: float | None = (
        DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_SCALE
    ),
    neural_residual_max_linear_override_margin: float | None = (
        DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_MAX_LINEAR_OVERRIDE_MARGIN
    ),
    neural_residual_context_gate: str | None = (
        DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_CONTEXT_GATE
    ),
    neural_residual_recovery_phase_ticks: int | None = (
        DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_RECOVERY_PHASE_TICKS
    ),
    weight_policy: str = MIND_V3_CARRION_RECOVERY_DISTILL_WEIGHT_POLICY,
    eval_seeds: Sequence[int] = DEFAULT_CARRION_RECOVERY_DISTILL_EVAL_SEEDS,
    eval_ticks: int = DEFAULT_CARRION_RECOVERY_DISTILL_EVAL_TICKS,
    fixture_names: Sequence[str] = DEFAULT_CARRION_RECOVERY_DISTILL_FIXTURES,
    fixture_seeds: Sequence[int] | None = None,
    fixture_ticks: int | None = None,
    horizon_output_path: str | Path | None = None,
    artifact_output_path: str | Path | None = None,
    evaluation_output_path: str | Path | None = None,
) -> dict[str, object]:
    if archive_report is None:
        if archive_report_path is None:
            raise CarrionRecoveryDistillError("archive_report_path is required")
        archive_report = load_carrion_recovery_json_report(archive_report_path)
    _validate_archive_report(archive_report)
    if archive_split_report is None and archive_split_report_path is not None:
        archive_split_report = load_carrion_recovery_json_report(
            archive_split_report_path
        )
    if archive_split_report is not None:
        _validate_archive_split_report(archive_split_report)
    horizon_ticks = normalize_horizon_ticks(horizons)
    eval_seed_values = _validated_seeds(eval_seeds, field="eval_seeds")
    fixture_seed_values = (
        _validated_seeds(fixture_seeds, field="fixture_seeds")
        if fixture_seeds is not None
        else eval_seed_values
    )
    eval_tick_count = _positive_int(eval_ticks, field="eval_ticks")
    fixture_tick_count = (
        _positive_int(fixture_ticks, field="fixture_ticks")
        if fixture_ticks is not None
        else eval_tick_count
    )
    fixture_name_values = _validated_fixture_names(fixture_names)
    mode = _validated_artifact_mode(artifact_mode)
    train_record_ids = (
        _split_record_ids(archive_split_report, split="train")
        if archive_split_report is not None
        else None
    )
    selected = _selected_trajectory_examples(
        archive_report,
        weight_policy=weight_policy,
        include_record_ids=train_record_ids,
    )
    datasets = [load_trajectory_jsonl(item["trajectory_path"]) for item in selected]
    trajectory_weights = [float(item["trajectory_weight"]) for item in selected]
    horizon_report = build_horizon_label_report(datasets, horizons=horizon_ticks)
    if horizon_output_path is not None:
        write_horizon_label_report(horizon_report, horizon_output_path)
    recovery_phase_action_bias = _recovery_phase_action_bias(
        datasets=datasets,
        selected=selected,
        trajectory_weights=trajectory_weights,
        recovery_phase_ticks=int(neural_residual_recovery_phase_ticks or 0),
    )
    artifact = train_mind_v3_neural_artifact(
        datasets,
        horizon_label_report=horizon_report,
        hidden_units=_positive_int(hidden_units, field="hidden_units"),
        seed=int(seed),
        trajectory_weight_multipliers=trajectory_weights,
        artifact_mode=mode,
        neural_residual_scale=neural_residual_scale,
        neural_residual_max_linear_override_margin=(
            neural_residual_max_linear_override_margin
        ),
        neural_residual_context_gate=neural_residual_context_gate,
        neural_residual_recovery_phase_ticks=(
            neural_residual_recovery_phase_ticks
        ),
        recovery_phase_action_bias=recovery_phase_action_bias,
    )
    if artifact_output_path is not None:
        write_mind_v3_neural_artifact(artifact, artifact_output_path)
    evaluation = _evaluate_artifact(
        artifact=artifact,
        eval_seeds=eval_seed_values,
        eval_ticks=eval_tick_count,
        fixture_names=fixture_name_values,
        fixture_seeds=fixture_seed_values,
        fixture_ticks=fixture_tick_count,
    )
    evaluation["action_balance_diagnostics"] = _evaluation_action_balance(evaluation)
    heldout_branch_evaluation = _heldout_branch_state_evaluation(
        archive_split_report,
    )
    if evaluation_output_path is not None:
        _write_json(evaluation, evaluation_output_path)
    contract = {
        "schema_version": MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_RECOVERY_DISTILL_POLICY,
        "source_archive_schema_version": archive_report.get("schema_version"),
        "source_archive_split_schema_version": (
            archive_split_report.get("schema_version")
            if archive_split_report is not None
            else None
        ),
        "horizon_ticks": list(horizon_ticks),
        "artifact_mode": mode,
        "hidden_units": int(hidden_units),
        "seed": int(seed),
        "neural_residual_scale": artifact.get("neural_residual_scale"),
        "neural_residual_max_linear_override_margin": artifact.get(
            "neural_residual_max_linear_override_margin"
        ),
        "neural_residual_context_gate": artifact.get(
            "neural_residual_context_gate",
            MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_NONE,
        ),
        "neural_residual_recovery_phase_ticks": artifact.get(
            "neural_residual_recovery_phase_ticks",
            0,
        ),
        "recovery_phase_action_bias_policy": (
            _mapping(artifact.get("recovery_phase_action_bias")).get("policy")
        ),
        "weight_policy": weight_policy,
        "eval_seeds": list(eval_seed_values),
        "eval_ticks": int(eval_tick_count),
        "fixture_names": list(fixture_name_values),
        "fixture_seeds": list(fixture_seed_values),
        "fixture_ticks": int(fixture_tick_count),
        "archive_split_policy": (
            archive_split_report.get("split_policy")
            if archive_split_report is not None
            else None
        ),
        "non_promoted_negative_control": True,
    }
    report = {
        "schema_version": MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
        "distillation_policy": MIND_V3_CARRION_RECOVERY_DISTILL_POLICY,
        "contract": contract,
        "provenance": {
            "contract_digest": stable_payload_digest(contract),
            "source_archive_report_digest": stable_payload_digest(
                {
                    "schema_version": archive_report.get("schema_version"),
                    "archive_contract": archive_report.get("archive_contract"),
                    "aggregate": archive_report.get("aggregate"),
                    "dataset": archive_report.get("dataset"),
                }
            ),
            "source_archive_split_report_digest": (
                stable_payload_digest(archive_split_report)
                if archive_split_report is not None
                else None
            ),
        },
        "source": {
            "archive_report_path": (
                str(archive_report_path) if archive_report_path is not None else None
            ),
            "archive_acceptance": archive_report.get("acceptance"),
            "archive_aggregate": archive_report.get("aggregate"),
            "archive_split_report_path": (
                str(archive_split_report_path)
                if archive_split_report_path is not None
                else None
            ),
            "archive_split_acceptance": (
                archive_split_report.get("acceptance")
                if archive_split_report is not None
                else None
            ),
            "archive_split_aggregate": (
                archive_split_report.get("aggregate")
                if archive_split_report is not None
                else None
            ),
        },
        "training": {
            "selected_trajectory_count": len(selected),
            "selected_trajectories": selected,
            "archive_split_consumed": archive_split_report is not None,
            "train_record_ids_from_split": (
                sorted(train_record_ids) if train_record_ids is not None else None
            ),
            "horizon_label_output_path": (
                str(horizon_output_path) if horizon_output_path is not None else None
            ),
            "horizon_label_source": horizon_report.get("source"),
            "horizon_label_aggregate": horizon_report.get("aggregate"),
            "artifact_output_path": (
                str(artifact_output_path) if artifact_output_path is not None else None
            ),
            "artifact_mode": artifact.get("artifact_mode"),
            "neural_residual_scale": artifact.get("neural_residual_scale"),
            "neural_residual_max_linear_override_margin": artifact.get(
                "neural_residual_max_linear_override_margin"
            ),
            "neural_residual_context_gate": artifact.get(
                "neural_residual_context_gate",
                MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_NONE,
            ),
            "neural_residual_recovery_phase_ticks": artifact.get(
                "neural_residual_recovery_phase_ticks",
                0,
            ),
            "recovery_phase_action_bias": artifact.get(
                "recovery_phase_action_bias"
            ),
            "model_type": artifact.get("model_type"),
            "trained_record_count": artifact.get("trained_record_count"),
            "artifact_training_summary": artifact.get("training_summary"),
            "artifact_action_value_summary": artifact.get("action_value_summary"),
        },
        "heldout_branch_state_evaluation": heldout_branch_evaluation,
        "promotion": {
            "promoted": False,
            "reason": (
                "negative_control_recovery_archive_distill_is_diagnostic_only"
            ),
            "requires_separate_promotion_review": True,
        },
        "evaluation": evaluation,
    }
    report["acceptance"] = _acceptance(report)
    return report


def write_carrion_recovery_distillation_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    _write_json(report, output_path)


def _validate_archive_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION:
        raise CarrionRecoveryDistillError("archive report has stale schema_version")
    dataset = report.get("dataset")
    records = dataset.get("records") if isinstance(dataset, Mapping) else None
    if not isinstance(records, list) or not records:
        raise CarrionRecoveryDistillError("archive report must include dataset records")


def _validate_archive_split_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION:
        raise CarrionRecoveryDistillError(
            "archive split report has stale schema_version"
        )
    if (
        report.get("split_policy")
        != MIND_V3_CARRION_RECOVERY_SPLIT_POLICY_BRANCH_DIGEST_SEED_STRATIFIED
    ):
        raise CarrionRecoveryDistillError("archive split report has unsupported policy")
    acceptance = _mapping(report.get("acceptance"))
    if bool(acceptance.get("training_blocked", True)):
        raise CarrionRecoveryDistillError(
            "archive split report blocks training: "
            + ",".join(str(item) for item in acceptance.get("blockers", []))
        )
    if not _split_record_ids(report, split="train"):
        raise CarrionRecoveryDistillError("archive split report has no train records")


def _split_record_ids(
    split_report: Mapping[str, object] | None,
    *,
    split: str,
) -> set[str]:
    if split_report is None:
        return set()
    records = _mapping(split_report.get("records"))
    rows = records.get(split)
    record_ids = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        record_id = row.get("record_id")
        if isinstance(record_id, str) and record_id:
            record_ids.add(record_id)
    return record_ids


def _selected_trajectory_examples(
    archive_report: Mapping[str, object],
    *,
    weight_policy: str,
    include_record_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    dataset = archive_report.get("dataset")
    records = dataset.get("records") if isinstance(dataset, Mapping) else []
    if not isinstance(records, list):
        records = []
    selected_by_path: dict[str, dict[str, object]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        record_id = record.get("record_id")
        if include_record_ids is not None and record_id not in include_record_ids:
            continue
        source = record.get("source")
        source_payload = source if isinstance(source, Mapping) else {}
        raw_path = source_payload.get("trajectory_path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            raise CarrionRecoveryDistillError(
                f"selected recovery trajectory does not exist: {path}"
            )
        weight = _trajectory_weight(record, weight_policy=weight_policy)
        payload = {
            "dataset_record_index": index,
            "record_id": record_id,
            "trajectory_path": path,
            "trajectory_weight": weight,
            "outcome_class": _label_value(record, "outcome_class"),
            "terminal_survivor": bool(_label_value(record, "terminal_survivor")),
            "alive_agents": _int(_label_value(record, "alive_agents")),
            "births": _int(_label_value(record, "births")),
            "seed": source_payload.get("seed"),
            "continuation_script": source_payload.get("continuation_script"),
        }
        existing = selected_by_path.get(str(path))
        if existing is None or float(payload["trajectory_weight"]) > float(
            existing["trajectory_weight"]
        ):
            selected_by_path[str(path)] = payload
    if not selected_by_path:
        raise CarrionRecoveryDistillError(
            "archive dataset records did not include usable trajectory paths"
        )
    return [
        {
            **payload,
            "trajectory_path": str(payload["trajectory_path"]),
        }
        for _path, payload in sorted(
            selected_by_path.items(),
            key=lambda item: int(item[1]["dataset_record_index"]),
        )
    ]


def _trajectory_weight(
    record: Mapping[str, object],
    *,
    weight_policy: str,
) -> float:
    if weight_policy == MIND_V3_CARRION_RECOVERY_DISTILL_UNIFORM_WEIGHT_POLICY:
        return 1.0
    if weight_policy != MIND_V3_CARRION_RECOVERY_DISTILL_WEIGHT_POLICY:
        raise CarrionRecoveryDistillError(
            f"unsupported recovery distillation weight policy: {weight_policy}"
        )
    terminal_survivor = bool(_label_value(record, "terminal_survivor"))
    if not terminal_survivor:
        return 0.25
    alive = _int(_label_value(record, "alive_agents"))
    births = _int(_label_value(record, "births"))
    outcome = record.get("outcome_metrics")
    outcome_payload = outcome if isinstance(outcome, Mapping) else {}
    scavenging = outcome_payload.get("scavenging")
    scavenging_payload = scavenging if isinstance(scavenging, Mapping) else {}
    scavenger_events = _int(
        scavenging_payload.get("scavenger_animal_resource_events")
    )
    return _round(min(4.0, 1.0 + alive * 0.5 + births * 0.15 + scavenger_events * 0.01))


def _recovery_phase_action_bias(
    *,
    datasets: Sequence[object],
    selected: Sequence[Mapping[str, object]],
    trajectory_weights: Sequence[float],
    recovery_phase_ticks: int,
) -> dict[str, object]:
    survivor_counts = {action: 0.0 for action in ACTION_NAMES}
    failure_counts = {action: 0.0 for action in ACTION_NAMES}
    record_count = 0
    if recovery_phase_ticks > 0:
        for dataset, meta, weight in zip(
            datasets,
            selected,
            trajectory_weights,
            strict=True,
        ):
            parsed_weight = _positive_finite_weight(weight)
            if parsed_weight <= 0.0:
                continue
            outcome_class = str(meta.get("outcome_class", ""))
            if outcome_class == "survivor":
                target_counts = survivor_counts
            elif outcome_class == "failure":
                target_counts = failure_counts
            else:
                continue
            active_remaining_by_agent: dict[int, int] = {}
            records = getattr(dataset, "records", ())
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                agent_id = _record_agent_id(record)
                if agent_id is None:
                    continue
                previous = max(0, int(active_remaining_by_agent.get(agent_id, 0)))
                if previous > 0:
                    action = _record_action(record)
                    if action in target_counts:
                        target_counts[action] += parsed_weight
                        record_count += 1
                alive_after = _record_alive_after(record)
                activated = _record_consumed_animal_resource(record) and alive_after
                if not alive_after:
                    active_remaining_by_agent.pop(agent_id, None)
                elif activated:
                    active_remaining_by_agent[agent_id] = int(recovery_phase_ticks)
                elif previous > 1:
                    active_remaining_by_agent[agent_id] = previous - 1
                else:
                    active_remaining_by_agent.pop(agent_id, None)

    action_bias = _branch_action_log_odds_bias(
        survivor_counts=survivor_counts,
        failure_counts=failure_counts,
        scale=DEFAULT_CARRION_RECOVERY_PHASE_ACTION_BIAS_SCALE,
        max_abs_bias=DEFAULT_CARRION_RECOVERY_PHASE_ACTION_BIAS_MAX_ABS,
    )
    survivor_total = sum(survivor_counts.values())
    failure_total = sum(failure_counts.values())
    return {
        "policy": MIND_V3_CARRION_RECOVERY_PHASE_ACTION_BIAS_POLICY,
        "scale": DEFAULT_CARRION_RECOVERY_PHASE_ACTION_BIAS_SCALE,
        "max_abs_bias": DEFAULT_CARRION_RECOVERY_PHASE_ACTION_BIAS_MAX_ABS,
        "action_bias": action_bias,
        "survivor_action_weight": {
            action: _round(value) for action, value in survivor_counts.items()
        },
        "failure_action_weight": {
            action: _round(value) for action, value in failure_counts.items()
        },
        "survivor_total_weight": _round(survivor_total),
        "failure_total_weight": _round(failure_total),
        "record_count": record_count,
    }


def _branch_action_log_odds_bias(
    *,
    survivor_counts: Mapping[str, float],
    failure_counts: Mapping[str, float],
    scale: float,
    max_abs_bias: float,
) -> dict[str, float]:
    action_domain = [
        action
        for action in ACTION_NAMES
        if (
            float(survivor_counts.get(action, 0.0))
            + float(failure_counts.get(action, 0.0))
        )
        > 0.0
    ]
    if not action_domain:
        return {action: 0.0 for action in ACTION_NAMES}
    survivor_total = sum(
        float(survivor_counts.get(action, 0.0)) for action in action_domain
    )
    failure_total = sum(
        float(failure_counts.get(action, 0.0)) for action in action_domain
    )
    if survivor_total <= 0.0 or failure_total <= 0.0:
        return {action: 0.0 for action in ACTION_NAMES}
    prior = 1.0
    action_count = float(len(action_domain))
    raw = {}
    for action in action_domain:
        survivor_rate = (
            float(survivor_counts.get(action, 0.0)) + prior
        ) / (survivor_total + action_count * prior)
        failure_rate = (
            float(failure_counts.get(action, 0.0)) + prior
        ) / (failure_total + action_count * prior)
        raw[action] = math.log(survivor_rate) - math.log(failure_rate)
    mean = sum(raw.values()) / float(len(raw))
    result = {action: 0.0 for action in ACTION_NAMES}
    result.update({
        action: _round(_clamp((raw[action] - mean) * scale, -max_abs_bias, max_abs_bias))
        for action in action_domain
    })
    return result


def _record_agent_id(record: Mapping[str, object]) -> int | None:
    value = record.get("agent_id")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _record_action(record: Mapping[str, object]) -> str:
    resolved = record.get("resolved_action")
    if isinstance(resolved, str) and resolved:
        return resolved
    requested = record.get("requested_action")
    if isinstance(requested, str):
        return requested
    return ""


def _record_alive_after(record: Mapping[str, object]) -> bool:
    after = record.get("after")
    if not isinstance(after, Mapping):
        return True
    return after.get("alive") is not False


def _record_consumed_animal_resource(record: Mapping[str, object]) -> bool:
    outcome = record.get("outcome")
    outcome_payload = outcome if isinstance(outcome, Mapping) else {}
    feeding = outcome_payload.get("feeding")
    feeding_payload = feeding if isinstance(feeding, Mapping) else {}
    food_source = feeding_payload.get("food_source")
    if food_source not in {"carcass", "fresh_kill"}:
        return False
    feeding_gain = _metric(feeding_payload, "gained_energy")
    resource_gain = _metric(outcome_payload, "resource_gain")
    return bool(feeding_payload.get("ate", False)) or max(
        feeding_gain,
        resource_gain,
    ) > 0.0


def _heldout_branch_state_evaluation(
    split_report: Mapping[str, object] | None,
) -> dict[str, object]:
    if split_report is None:
        return {
            "enabled": False,
            "reason": "archive_split_report_not_provided",
        }
    rows = _split_rows(split_report, split="heldout")
    diagnostics = []
    load_failures = []
    for row in rows:
        path = row.get("trajectory_path")
        if not isinstance(path, str) or not path:
            load_failures.append(
                {
                    "record_id": row.get("record_id"),
                    "reason": "trajectory_path_missing",
                }
            )
            continue
        try:
            dataset = load_trajectory_jsonl(path)
        except (OSError, ValueError) as exc:
            load_failures.append(
                {
                    "record_id": row.get("record_id"),
                    "trajectory_path": path,
                    "reason": str(exc),
                }
            )
            continue
        diagnostics.append(
            {
                "record_id": row.get("record_id"),
                "seed": row.get("seed"),
                "branch_id": row.get("branch_id"),
                "branch_state_key": row.get("branch_state_key"),
                "branch_tick": row.get("branch_tick"),
                "continuation_script": row.get("continuation_script"),
                "terminal_survivor": bool(row.get("terminal_survivor")),
                "trajectory_path": path,
                "action_balance": _trajectory_action_balance(dataset.records),
            }
        )
    aggregate = _heldout_branch_aggregate(diagnostics)
    return {
        "enabled": True,
        "split_policy": split_report.get("split_policy"),
        "heldout_record_count": len(rows),
        "loaded_record_count": len(diagnostics),
        "load_failure_count": len(load_failures),
        "load_failures": load_failures,
        "survivor_count": sum(
            1 for row in rows if bool(row.get("terminal_survivor"))
        ),
        "failure_count": sum(
            1 for row in rows if not bool(row.get("terminal_survivor"))
        ),
        "by_seed": _heldout_count_table(rows, "seed"),
        "by_continuation_script": _heldout_count_table(
            rows,
            "continuation_script",
        ),
        "aggregate_action_balance": aggregate,
        "records": diagnostics,
    }


def _split_rows(
    split_report: Mapping[str, object],
    *,
    split: str,
) -> list[Mapping[str, object]]:
    records = _mapping(split_report.get("records"))
    rows = records.get(split)
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _trajectory_action_balance(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    requested = Counter()
    resolved = Counter()
    action_source = Counter()
    unsupported_requested = 0
    unsupported_resolved = 0
    changed_linear = 0
    recovery_phase_residual = 0
    residual_application = 0
    missing_decision_diagnostics = 0
    for record in records:
        requested_action = str(record.get("requested_action", "unknown"))
        resolved_action = str(record.get("resolved_action", requested_action))
        requested[requested_action] += 1
        resolved[resolved_action] += 1
        action_source[str(record.get("action_source", "unknown"))] += 1
        if record.get("action_valid") is False:
            unsupported_requested += 1
        if record.get("resolution_action_valid") is False:
            unsupported_resolved += 1
        diagnostics = record.get("policy_decision_diagnostics")
        if not isinstance(diagnostics, Mapping):
            missing_decision_diagnostics += 1
            continue
        if bool(diagnostics.get("neural_residual_changed_linear_decision")):
            changed_linear += 1
        elif (
            diagnostics.get("linear_anchor_action") is not None
            and diagnostics.get("requested_action") is not None
            and diagnostics.get("linear_anchor_action")
            != diagnostics.get("requested_action")
        ):
            changed_linear += 1
        effective_scale = _metric(diagnostics, "neural_residual_effective_scale")
        if effective_scale > 0.0:
            residual_application += 1
        if (
            effective_scale > 0.0
            and diagnostics.get("neural_residual_context_gate_reason")
            == "recovery_phase"
        ):
            recovery_phase_residual += 1
    total = sum(requested.values())
    dominant_action, dominant_count = _dominant_count(requested)
    return {
        "record_count": total,
        "requested_action_counts": dict(sorted(requested.items())),
        "resolved_action_counts": dict(sorted(resolved.items())),
        "action_source_counts": dict(sorted(action_source.items())),
        "dominant_requested_action": dominant_action,
        "dominant_requested_action_share": (
            _round(dominant_count / total) if total else 0.0
        ),
        "drink_requested_count": requested.get("drink", 0),
        "eat_requested_count": requested.get("eat", 0),
        "stay_requested_count": requested.get("stay", 0),
        "move_requested_count": sum(
            count for action, count in requested.items() if action.startswith("move_")
        ),
        "unsupported_requested_action_count": unsupported_requested,
        "unsupported_resolved_action_count": unsupported_resolved,
        "changed_linear_decision_count": changed_linear,
        "recovery_phase_residual_application_count": recovery_phase_residual,
        "residual_application_count": residual_application,
        "missing_decision_diagnostics_count": missing_decision_diagnostics,
    }


def _heldout_branch_aggregate(
    diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    requested = Counter()
    resolved = Counter()
    action_source = Counter()
    totals = Counter()
    for item in diagnostics:
        balance = _mapping(item.get("action_balance"))
        requested.update(_counter_from_mapping(balance.get("requested_action_counts")))
        resolved.update(_counter_from_mapping(balance.get("resolved_action_counts")))
        action_source.update(_counter_from_mapping(balance.get("action_source_counts")))
        for key in (
            "record_count",
            "drink_requested_count",
            "eat_requested_count",
            "stay_requested_count",
            "move_requested_count",
            "unsupported_requested_action_count",
            "unsupported_resolved_action_count",
            "changed_linear_decision_count",
            "recovery_phase_residual_application_count",
            "residual_application_count",
            "missing_decision_diagnostics_count",
        ):
            totals[key] += _int(balance.get(key))
    total = int(totals["record_count"])
    dominant_action, dominant_count = _dominant_count(requested)
    return {
        "record_count": total,
        "requested_action_counts": dict(sorted(requested.items())),
        "resolved_action_counts": dict(sorted(resolved.items())),
        "action_source_counts": dict(sorted(action_source.items())),
        "dominant_requested_action": dominant_action,
        "dominant_requested_action_share": (
            _round(dominant_count / total) if total else 0.0
        ),
        **{key: int(value) for key, value in sorted(totals.items())},
    }


def _heldout_count_table(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = {}
    for row in rows:
        key = str(row.get(field, "unknown"))
        outcome = "survivor" if bool(row.get("terminal_survivor")) else "failure"
        table.setdefault(key, Counter())[outcome] += 1
        table[key]["total"] += 1
    return {key: dict(sorted(counts.items())) for key, counts in sorted(table.items())}


def _evaluation_action_balance(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    open_eval = _mapping(evaluation.get("open"))
    comparison = _mapping(open_eval.get("comparison"))
    result = {
        "open": {
            "mind_v3_linear": _run_summary_action_balance(
                _mapping(comparison.get("mind_v3_linear")).get("runs")
            ),
            "mind_v3_recovery_distilled": _run_summary_action_balance(
                _mapping(comparison.get("mind_v3_recovery_distilled")).get("runs")
            ),
        },
        "fixture": {},
    }
    candidate = result["open"]["mind_v3_recovery_distilled"]
    linear = result["open"]["mind_v3_linear"]
    result["open"]["candidate_vs_linear_delta"] = _action_balance_delta(
        candidate,
        linear,
    )
    fixture = _mapping(evaluation.get("fixture"))
    candidate_suite = _mapping(fixture.get("candidate_suite"))
    linear_suite = _mapping(fixture.get("linear_baseline_suite"))
    result["fixture"] = {
        "mind_v3_linear": _fixture_suite_action_balance(linear_suite),
        "mind_v3_recovery_distilled": _fixture_suite_action_balance(candidate_suite),
    }
    return result


def _run_summary_action_balance(value: object) -> dict[str, object]:
    runs = [run for run in value if isinstance(run, Mapping)] if isinstance(value, list) else []
    requested = Counter()
    resolved = Counter()
    totals = Counter()
    for run in runs:
        requested.update(_counter_from_mapping(run.get("requested_action_counts")))
        resolved.update(_counter_from_mapping(run.get("resolved_action_counts")))
        for key in (
            "unsupported_requested_action_count",
            "unsupported_resolved_action_count",
            "heuristic_action_source_count",
            "changed_linear_decision_count",
            "recovery_phase_residual_application_count",
            "residual_application_count",
        ):
            totals[key] += _int(run.get(key))
    total = sum(requested.values())
    dominant_action, dominant_count = _dominant_count(requested)
    return {
        "run_count": len(runs),
        "requested_action_counts": dict(sorted(requested.items())),
        "resolved_action_counts": dict(sorted(resolved.items())),
        "dominant_requested_action": dominant_action,
        "dominant_requested_action_share": (
            _round(dominant_count / total) if total else 0.0
        ),
        "drink_requested_count": requested.get("drink", 0),
        "eat_requested_count": requested.get("eat", 0),
        "stay_requested_count": requested.get("stay", 0),
        "move_requested_count": sum(
            count for action, count in requested.items() if action.startswith("move_")
        ),
        **{key: int(value) for key, value in sorted(totals.items())},
    }


def _fixture_suite_action_balance(
    suite: Mapping[str, object],
) -> dict[str, object]:
    policy_key = str(suite.get("evaluated_policy_key", "mind_v3"))
    fixtures = suite.get("fixtures")
    result = {}
    for fixture in fixtures if isinstance(fixtures, list) else []:
        if not isinstance(fixture, Mapping):
            continue
        comparison = _mapping(fixture.get("comparison"))
        policy = _mapping(comparison.get(policy_key))
        aggregate = _mapping(policy.get("aggregate"))
        result[str(fixture.get("fixture", "unknown"))] = {
            "dominant_requested_action": aggregate.get("dominant_requested_action"),
            "dominant_requested_action_share": aggregate.get(
                "dominant_requested_action_share"
            ),
            "unsupported_requested_action_count": _int(
                aggregate.get("unsupported_requested_action_count")
            ),
            "unsupported_resolved_action_count": _int(
                aggregate.get("unsupported_resolved_action_count")
            ),
            "heuristic_action_source_count": _int(
                aggregate.get("heuristic_action_source_count")
            ),
            "requested_action_counts": aggregate.get("requested_action_counts", {}),
        }
    return result


def _action_balance_delta(
    candidate: Mapping[str, object],
    linear: Mapping[str, object],
) -> dict[str, object]:
    result = {}
    for key in (
        "drink_requested_count",
        "eat_requested_count",
        "stay_requested_count",
        "move_requested_count",
        "unsupported_requested_action_count",
        "unsupported_resolved_action_count",
        "changed_linear_decision_count",
        "recovery_phase_residual_application_count",
        "residual_application_count",
    ):
        result[f"{key}_delta"] = _int(candidate.get(key)) - _int(linear.get(key))
    result["dominant_requested_action_changed"] = (
        candidate.get("dominant_requested_action")
        != linear.get("dominant_requested_action")
    )
    result["dominant_requested_action_share_delta"] = _round(
        _metric(candidate, "dominant_requested_action_share")
        - _metric(linear, "dominant_requested_action_share")
    )
    return result


def _counter_from_mapping(value: object) -> Counter[str]:
    payload = _mapping(value)
    return Counter({str(key): _int(count) for key, count in payload.items()})


def _dominant_count(counter: Counter[str]) -> tuple[str | None, int]:
    if not counter:
        return None, 0
    action, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
    return action, int(count)


def _positive_finite_weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        return 0.0
    return parsed


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _label_value(record: Mapping[str, object], key: str) -> object:
    label = record.get("label")
    payload = label if isinstance(label, Mapping) else {}
    return payload.get(key)


def _evaluate_artifact(
    *,
    artifact: Mapping[str, object],
    eval_seeds: Sequence[int],
    eval_ticks: int,
    fixture_names: Sequence[str],
    fixture_seeds: Sequence[int],
    fixture_ticks: int,
) -> dict[str, object]:
    heuristic_runs = [
        _run_once(seed=seed, ticks=eval_ticks, policy=None)
        for seed in eval_seeds
    ]
    linear_runs = [
        _run_once(
            seed=seed,
            ticks=eval_ticks,
            policy=MindV3EvolutionPolicy(seed=seed),
        )
        for seed in eval_seeds
    ]
    candidate_runs = [
        _run_once(
            seed=seed,
            ticks=eval_ticks,
            policy=MindV3EvolutionPolicy(seed=seed, neural_artifact=artifact),
        )
        for seed in eval_seeds
    ]
    heuristic_aggregate = _aggregate_runs(heuristic_runs)
    linear_aggregate = _aggregate_runs(linear_runs)
    candidate_aggregate = _aggregate_runs(candidate_runs)
    fixture_config = mind_v3_fixture_gate_config(
        suite="basic",
        seeds=list(fixture_seeds),
        ticks=fixture_ticks,
        min_alive=0.0,
        min_births=0.0,
        min_mixed_stable_births=0.0,
        min_energy_viability=0.0,
        min_hydration_viability=0.0,
        min_health_viability=0.0,
        min_matched_diet_viability=0.0,
        min_biologically_ready=0.0,
    )
    fixture_suite = run_mind_v3_fixture_suite(
        suite="basic",
        fixture_names=list(fixture_names),
        seeds=list(fixture_seeds),
        ticks=fixture_ticks,
        founder_template=None,
        neural_artifact=dict(artifact),
    )
    linear_fixture_suite = run_mind_v3_fixture_suite(
        suite="basic",
        fixture_names=list(fixture_names),
        seeds=list(fixture_seeds),
        ticks=fixture_ticks,
        founder_template=None,
        neural_artifact=None,
        trajectory_prefix="fixture_linear_baseline",
    )
    fixture_gate = mind_v3_fixture_gate_status(
        fixture_suite=fixture_suite,
        fixture_config=fixture_config,
    )
    linear_fixture_gate = mind_v3_fixture_gate_status(
        fixture_suite=linear_fixture_suite,
        fixture_config=fixture_config,
    )
    return {
        "open": {
            "seeds": list(eval_seeds),
            "ticks": int(eval_ticks),
            "comparison": {
                "heuristic": {
                    "runs": heuristic_runs,
                    "aggregate": heuristic_aggregate,
                },
                "mind_v3_linear": {
                    "runs": linear_runs,
                    "aggregate": linear_aggregate,
                },
                "mind_v3_recovery_distilled": {
                    "runs": candidate_runs,
                    "aggregate": candidate_aggregate,
                },
                "candidate_vs_linear_delta": _comparison_delta(
                    heuristic=linear_aggregate,
                    mind_v3=candidate_aggregate,
                ),
                "candidate_vs_linear_per_seed": _open_seed_delta_rows(
                    candidate_runs=candidate_runs,
                    linear_runs=linear_runs,
                ),
                "candidate_vs_heuristic_delta": _comparison_delta(
                    heuristic=heuristic_aggregate,
                    mind_v3=candidate_aggregate,
                ),
            },
        },
        "fixture": {
            "fixture_config": fixture_config,
            "candidate_suite": fixture_suite,
            "candidate_gate": fixture_gate,
            "linear_baseline_suite": linear_fixture_suite,
            "linear_baseline_gate": linear_fixture_gate,
            "candidate_vs_linear_summary": _fixture_delta_summary(
                candidate_suite=fixture_suite,
                linear_suite=linear_fixture_suite,
            ),
        },
    }


def _fixture_delta_summary(
    *,
    candidate_suite: Mapping[str, object],
    linear_suite: Mapping[str, object],
) -> dict[str, object]:
    candidate_by_name = _fixture_aggregate_by_name(candidate_suite)
    linear_by_name = _fixture_aggregate_by_name(linear_suite)
    summary = {}
    for fixture_name in sorted(set(candidate_by_name) | set(linear_by_name)):
        candidate = candidate_by_name.get(fixture_name, {})
        linear = linear_by_name.get(fixture_name, {})
        candidate_outcome = _mapping(candidate.get("outcome_metrics"))
        linear_outcome = _mapping(linear.get("outcome_metrics"))
        summary[fixture_name] = {
            "alive_agents_mean_delta": _metric(candidate, "alive_agents_mean")
            - _metric(linear, "alive_agents_mean"),
            "births_mean_delta": _metric(candidate, "births_mean")
            - _metric(linear, "births_mean"),
            "terminal_survivor_run_count_delta": (
                _metric(candidate_outcome, "terminal_survivor_run_count")
                - _metric(linear_outcome, "terminal_survivor_run_count")
            ),
            "scavenger_animal_resource_events_delta": (
                _metric(candidate_outcome, "total_scavenger_animal_resource_events")
                - _metric(linear_outcome, "total_scavenger_animal_resource_events")
            ),
            "animal_resource_events_delta": (
                _metric(candidate_outcome, "total_animal_resource_consumption_events")
                - _metric(linear_outcome, "total_animal_resource_consumption_events")
            ),
        }
    return summary


def _fixture_aggregate_by_name(
    fixture_suite: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    fixtures = fixture_suite.get("fixtures")
    result: dict[str, Mapping[str, object]] = {}
    for fixture in fixtures if isinstance(fixtures, list) else []:
        if not isinstance(fixture, Mapping):
            continue
        comparison = fixture.get("comparison")
        if not isinstance(comparison, Mapping):
            continue
        policy_key = str(fixture_suite.get("evaluated_policy_key", "mind_v3"))
        policy = comparison.get(policy_key)
        if not isinstance(policy, Mapping):
            continue
        aggregate = policy.get("aggregate")
        if isinstance(aggregate, Mapping):
            result[str(fixture.get("fixture", "unknown"))] = aggregate
    return result


def _open_seed_delta_rows(
    *,
    candidate_runs: Sequence[Mapping[str, object]],
    linear_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    linear_by_seed = {
        int(run["seed"]): run
        for run in linear_runs
        if isinstance(run, Mapping) and "seed" in run
    }
    rows = []
    for candidate in candidate_runs:
        if not isinstance(candidate, Mapping) or "seed" not in candidate:
            continue
        seed = int(candidate["seed"])
        linear = linear_by_seed.get(seed)
        if linear is None:
            continue
        candidate_alive = _int(candidate.get("alive_agents"))
        linear_alive = _int(linear.get("alive_agents"))
        candidate_births = _int(candidate.get("births"))
        linear_births = _int(linear.get("births"))
        candidate_deaths = _int(candidate.get("deaths"))
        linear_deaths = _int(linear.get("deaths"))
        rows.append(
            {
                "seed": seed,
                "candidate_alive_agents": candidate_alive,
                "linear_alive_agents": linear_alive,
                "alive_agents_delta": candidate_alive - linear_alive,
                "candidate_births": candidate_births,
                "linear_births": linear_births,
                "births_delta": candidate_births - linear_births,
                "candidate_deaths": candidate_deaths,
                "linear_deaths": linear_deaths,
                "deaths_delta": candidate_deaths - linear_deaths,
            }
        )
    return sorted(rows, key=lambda row: int(row["seed"]))


def _acceptance(report: Mapping[str, object]) -> dict[str, object]:
    training = _mapping(report.get("training"))
    evaluation = _mapping(report.get("evaluation"))
    open_eval = _mapping(evaluation.get("open"))
    comparison = _mapping(open_eval.get("comparison"))
    candidate = _mapping(comparison.get("mind_v3_recovery_distilled"))
    candidate_aggregate = _mapping(candidate.get("aggregate"))
    linear = _mapping(comparison.get("mind_v3_linear"))
    linear_aggregate = _mapping(linear.get("aggregate"))
    fixture = _mapping(evaluation.get("fixture"))
    fixture_summary = _mapping(fixture.get("candidate_vs_linear_summary"))
    blockers = []
    if _int(training.get("selected_trajectory_count")) <= 0:
        blockers.append("no_selected_recovery_trajectories")
    if _int(training.get("trained_record_count")) <= 0:
        blockers.append("artifact_trained_record_count_zero")
    if _int(candidate_aggregate.get("heuristic_action_source_count")) != 0:
        blockers.append("candidate_used_heuristic_runtime_actions")
    promotion_blockers = []
    if _metric(candidate_aggregate, "alive_agents_mean") < _metric(
        linear_aggregate,
        "alive_agents_mean",
    ):
        promotion_blockers.append("open_alive_regression_vs_linear")
    if _metric(candidate_aggregate, "births_mean") < _metric(
        linear_aggregate,
        "births_mean",
    ):
        promotion_blockers.append("open_birth_regression_vs_linear")
    open_seed_summary = _open_per_seed_regression_summary(
        comparison.get("candidate_vs_linear_per_seed"),
        expected_seeds=_seed_values(open_eval.get("seeds")),
    )
    if not bool(open_seed_summary["coverage_passed"]):
        promotion_blockers.append("open_per_seed_delta_coverage_mismatch")
    for row in open_seed_summary["regressions"]:
        seed = int(row["seed"])
        if _metric(row, "alive_agents_delta") < 0:
            promotion_blockers.append(
                f"open_seed_{seed}_alive_regression_vs_linear"
            )
        if _metric(row, "births_delta") < 0:
            promotion_blockers.append(
                f"open_seed_{seed}_birth_regression_vs_linear"
            )
    for fixture_name, delta in fixture_summary.items():
        delta_payload = delta if isinstance(delta, Mapping) else {}
        if _metric(delta_payload, "terminal_survivor_run_count_delta") < 0:
            promotion_blockers.append(
                f"{fixture_name}_terminal_survivor_regression_vs_linear"
            )
        if _metric(delta_payload, "births_mean_delta") < 0:
            promotion_blockers.append(f"{fixture_name}_birth_regression_vs_linear")
        if _metric(delta_payload, "scavenger_animal_resource_events_delta") < 0:
            promotion_blockers.append(
                f"{fixture_name}_scavenger_event_regression_vs_linear"
            )
    return {
        "data_path_acceptance_passed": not blockers,
        "data_path_blockers": blockers,
        "promotion_candidate_passed": not promotion_blockers,
        "promotion_blockers": promotion_blockers,
        "open_per_seed_regression_summary": open_seed_summary,
        "requires_zero_heuristic_runtime_actions": True,
        "promotion_requires_no_open_alive_or_birth_regression_vs_linear": True,
        "promotion_requires_no_open_per_seed_alive_or_birth_regression_vs_linear": (
            True
        ),
        "promotion_requires_no_fixture_birth_or_scavenger_regression_vs_linear": True,
    }


def _open_per_seed_regression_summary(
    value: object,
    *,
    expected_seeds: Sequence[int] = (),
) -> dict[str, object]:
    rows = (
        [row for row in value if isinstance(row, Mapping)]
        if isinstance(value, list)
        else []
    )
    row_seeds = [_int(row.get("seed")) for row in rows]
    row_seed_counts = Counter(row_seeds)
    expected_seed_values = tuple(int(seed) for seed in expected_seeds)
    expected_seed_set = set(expected_seed_values)
    row_seed_set = set(row_seeds)
    missing_seeds = sorted(expected_seed_set - row_seed_set)
    unexpected_seeds = sorted(row_seed_set - expected_seed_set)
    duplicate_seeds = sorted(seed for seed, count in row_seed_counts.items() if count > 1)
    coverage_passed = (
        not expected_seed_values
        or (
            not missing_seeds
            and not unexpected_seeds
            and not duplicate_seeds
            and len(row_seeds) == len(expected_seed_values)
        )
    )
    regressions = [
        {
            "seed": _int(row.get("seed")),
            "alive_agents_delta": _metric(row, "alive_agents_delta"),
            "births_delta": _metric(row, "births_delta"),
        }
        for row in rows
        if _metric(row, "alive_agents_delta") < 0
        or _metric(row, "births_delta") < 0
    ]
    alive_deltas = [_metric(row, "alive_agents_delta") for row in rows]
    birth_deltas = [_metric(row, "births_delta") for row in rows]
    return {
        "expected_seed_count": len(expected_seed_values),
        "seed_count": len(rows),
        "coverage_passed": coverage_passed,
        "missing_seeds": missing_seeds,
        "unexpected_seeds": unexpected_seeds,
        "duplicate_seeds": duplicate_seeds,
        "min_alive_agents_delta": min(alive_deltas) if alive_deltas else 0.0,
        "min_births_delta": min(birth_deltas) if birth_deltas else 0.0,
        "regression_count": len(regressions),
        "regressions": regressions,
    }


def _validated_artifact_mode(value: str) -> str:
    mode = str(value).strip()
    if mode not in {
        MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED,
        MIND_V3_NEURAL_ARTIFACT_MODE_HORIZON_FIXTURE,
    }:
        raise CarrionRecoveryDistillError(f"unsupported artifact mode: {mode}")
    return mode


def _validated_seeds(
    values: Sequence[int] | None,
    *,
    field: str,
) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values or ())
    if not parsed:
        raise CarrionRecoveryDistillError(f"{field} must include at least one seed")
    return parsed


def _validated_fixture_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(str(value) for value in values if str(value)))
    if not names:
        raise CarrionRecoveryDistillError("fixture_names must not be empty")
    supported = {"plant_only", "carrion_only", "prey_rich", "mixed_stable"}
    unsupported = sorted(name for name in names if name not in supported)
    if unsupported:
        raise CarrionRecoveryDistillError(
            "unsupported fixture(s): " + ", ".join(unsupported)
        )
    return names


def _positive_int(value: int | None, *, field: str) -> int:
    parsed = int(value if value is not None else 0)
    if parsed <= 0:
        raise CarrionRecoveryDistillError(f"{field} must be positive")
    return parsed


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _seed_values(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    parsed = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(parsed)


def _metric(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _round(value: float) -> float:
    return round(float(value), 6)


def _write_json(payload: Mapping[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
