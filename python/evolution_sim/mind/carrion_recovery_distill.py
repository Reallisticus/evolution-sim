from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

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
    train_mind_v3_neural_artifact,
    write_mind_v3_neural_artifact,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

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
DEFAULT_CARRION_RECOVERY_DISTILL_EVAL_SEEDS: tuple[int, ...] = (29, 37)
DEFAULT_CARRION_RECOVERY_DISTILL_EVAL_TICKS = 120
DEFAULT_CARRION_RECOVERY_DISTILL_FIXTURES: tuple[str, ...] = ("carrion_only",)


class CarrionRecoveryDistillError(ValueError):
    pass


def build_carrion_recovery_distillation_report(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZON_TICKS,
    artifact_mode: str = DEFAULT_CARRION_RECOVERY_DISTILL_ARTIFACT_MODE,
    hidden_units: int = DEFAULT_CARRION_RECOVERY_DISTILL_HIDDEN_UNITS,
    seed: int = MIND_V3_NEURAL_DEFAULT_SEED,
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
    selected = _selected_trajectory_examples(
        archive_report,
        weight_policy=weight_policy,
    )
    datasets = [load_trajectory_jsonl(item["trajectory_path"]) for item in selected]
    trajectory_weights = [float(item["trajectory_weight"]) for item in selected]
    horizon_report = build_horizon_label_report(datasets, horizons=horizon_ticks)
    if horizon_output_path is not None:
        write_horizon_label_report(horizon_report, horizon_output_path)
    artifact = train_mind_v3_neural_artifact(
        datasets,
        horizon_label_report=horizon_report,
        hidden_units=_positive_int(hidden_units, field="hidden_units"),
        seed=int(seed),
        trajectory_weight_multipliers=trajectory_weights,
        artifact_mode=mode,
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
    if evaluation_output_path is not None:
        _write_json(evaluation, evaluation_output_path)
    contract = {
        "schema_version": MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_RECOVERY_DISTILL_POLICY,
        "source_archive_schema_version": archive_report.get("schema_version"),
        "horizon_ticks": list(horizon_ticks),
        "artifact_mode": mode,
        "hidden_units": int(hidden_units),
        "seed": int(seed),
        "weight_policy": weight_policy,
        "eval_seeds": list(eval_seed_values),
        "eval_ticks": int(eval_tick_count),
        "fixture_names": list(fixture_name_values),
        "fixture_seeds": list(fixture_seed_values),
        "fixture_ticks": int(fixture_tick_count),
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
        },
        "source": {
            "archive_report_path": (
                str(archive_report_path) if archive_report_path is not None else None
            ),
            "archive_acceptance": archive_report.get("acceptance"),
            "archive_aggregate": archive_report.get("aggregate"),
        },
        "training": {
            "selected_trajectory_count": len(selected),
            "selected_trajectories": selected,
            "horizon_label_output_path": (
                str(horizon_output_path) if horizon_output_path is not None else None
            ),
            "horizon_label_source": horizon_report.get("source"),
            "horizon_label_aggregate": horizon_report.get("aggregate"),
            "artifact_output_path": (
                str(artifact_output_path) if artifact_output_path is not None else None
            ),
            "artifact_mode": artifact.get("artifact_mode"),
            "model_type": artifact.get("model_type"),
            "trained_record_count": artifact.get("trained_record_count"),
            "artifact_training_summary": artifact.get("training_summary"),
            "artifact_action_value_summary": artifact.get("action_value_summary"),
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


def _selected_trajectory_examples(
    archive_report: Mapping[str, object],
    *,
    weight_policy: str,
) -> list[dict[str, object]]:
    dataset = archive_report.get("dataset")
    records = dataset.get("records") if isinstance(dataset, Mapping) else []
    if not isinstance(records, list):
        records = []
    selected_by_path: dict[str, dict[str, object]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
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
            "record_id": record.get("record_id"),
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
    for fixture_name, delta in fixture_summary.items():
        delta_payload = delta if isinstance(delta, Mapping) else {}
        if _metric(delta_payload, "terminal_survivor_run_count_delta") < 0:
            promotion_blockers.append(
                f"{fixture_name}_terminal_survivor_regression_vs_linear"
            )
    return {
        "data_path_acceptance_passed": not blockers,
        "data_path_blockers": blockers,
        "promotion_candidate_passed": not promotion_blockers,
        "promotion_blockers": promotion_blockers,
        "requires_zero_heuristic_runtime_actions": True,
        "promotion_requires_no_open_alive_or_birth_regression_vs_linear": True,
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
