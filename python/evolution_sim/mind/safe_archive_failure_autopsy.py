from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.cli import mind_v3_evaluate as evaluate_cli
from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.mind.branch_intervention_residual import write_json
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.support_gated_residual import (
    load_support_gated_residual_artifact,
)
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

M3_SAFE_ARCHIVE_FAILURE_AUTOPSY_SCHEMA_VERSION = (
    "m3_safe_archive_failure_autopsy_report_v1"
)
M3_SAFE_ARCHIVE_FAILURE_AUTOPSY_POLICY = (
    "diagnostics_only_m3_safe_archive_failure_autopsy_v1"
)
M3_SAFE_ARCHIVE_TRAIN_EVAL_EXPECTED_CLASSIFICATION = (
    "m3_safe_archive_diagnostic_failed_non_promotional"
)
DEFAULT_BP3_DIAGNOSTIC_ARTIFACT_PATH = Path(
    "output/mind/shards/bp3-safe-archive-diagnostic-artifact.json"
)
DEFAULT_BP3_TRAIN_EVAL_REPORT_PATH = Path(
    "output/mind/shards/bp3-safe-archive-train-eval-report.json"
)
DEFAULT_BP3_DATASET_PATH = Path("output/mind/shards/bp3-merged-dataset.jsonl")
DEFAULT_BP3_BRANCH_EVIDENCE_PATH = Path(
    "output/mind/shards/bp3-merged-branch-evidence.json"
)
DEFAULT_BP3_FAILURE_AUTOPSY_OUTPUT_PATH = Path(
    "output/mind/shards/bp3-safe-archive-failure-autopsy-report.json"
)
DEFAULT_FAILURE_CASES = (
    ("broad", 19),
    ("broad", 29),
    ("broad", 37),
    ("carrion_only", 13),
    ("carrion_only", 37),
    ("carrion_only", 43),
)


def load_json_report(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must be an object: {path}")
    return payload


def load_jsonl_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def write_safe_archive_failure_autopsy_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    write_json(output_path, dict(report))


def build_safe_archive_failure_autopsy_report(
    *,
    artifact: Mapping[str, object],
    train_eval_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    branch_evidence: Mapping[str, object],
    ticks: int = 120,
    failure_cases: Sequence[tuple[str, int]] = DEFAULT_FAILURE_CASES,
    rerun_cases: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    support_artifact = load_support_gated_residual_artifact(artifact)
    validation = _validate_autopsy_inputs(
        support_artifact=support_artifact,
        train_eval_report=train_eval_report,
        dataset_rows=dataset_rows,
        branch_evidence=branch_evidence,
    )
    dataset_by_index = {
        _int(_mapping(row.get("metadata")).get("row_index"), default=index): dict(row)
        for index, row in enumerate(dataset_rows)
    }
    seed_delta = _seed_delta_by_case(train_eval_report)
    expected_cases = tuple((str(fixture), int(seed)) for fixture, seed in failure_cases)
    if rerun_cases is None:
        case_reports = [
            _rerun_failure_case(
                fixture=fixture,
                seed=seed,
                ticks=int(ticks),
                artifact=support_artifact,
                dataset_by_index=dataset_by_index,
                seed_delta=seed_delta,
            )
            for fixture, seed in expected_cases
        ]
    else:
        case_reports = [dict(item) for item in rerun_cases]
    traces = [
        dict(trace)
        for case in case_reports
        for trace in _list_of_mappings(case.get("override_traces"))
    ]
    complete = {
        (str(case.get("fixture")), _int(case.get("seed"), default=-1))
        for case in case_reports
    } == set(expected_cases)
    causes = _failure_cause_assessment(
        traces=traces,
        train_eval_report=train_eval_report,
        case_reports=case_reports,
    )
    recommended = _recommended_next_route(causes)
    classification = (
        "m3_safe_archive_failure_autopsy_complete_no_training"
        if complete
        else "m3_safe_archive_failure_autopsy_incomplete_no_training"
    )
    return {
        "schema_version": M3_SAFE_ARCHIVE_FAILURE_AUTOPSY_SCHEMA_VERSION,
        "policy": M3_SAFE_ARCHIVE_FAILURE_AUTOPSY_POLICY,
        "classification": {"primary": classification, "labels": [classification]},
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "contract": {
            "diagnostics_only": True,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "runtime_policy_changed": False,
            "gate_relaxation": False,
            "live_ab_enabled": False,
            "trainable_inputs_changed": False,
            "agent_id_seed_fixture_branch_private_state_metadata_only": True,
        },
        "inputs": {
            "artifact_digest": stable_payload_digest(support_artifact),
            "train_eval_report_digest": stable_payload_digest(train_eval_report),
            "dataset_digest": stable_payload_digest(list(dataset_rows)),
            "branch_evidence_digest": branch_evidence.get("branch_evidence_digest"),
            "input_validation": validation,
            "ticks": int(ticks),
            "failure_cases": [
                {"fixture": fixture, "seed": seed} for fixture, seed in expected_cases
            ],
        },
        "artifact": {
            "schema_version": support_artifact.get("schema_version"),
            "policy": support_artifact.get("policy"),
            "training_authorized": support_artifact.get("training_authorized") is True,
            "promotion_authorized": support_artifact.get("promotion_authorized") is True,
            "runtime_promotion_allowed": (
                support_artifact.get("runtime_promotion_allowed") is True
            ),
            "training_row_count": support_artifact.get("training_row_count"),
            "support_action_counts": support_artifact.get("support_action_counts"),
        },
        "train_eval_failure_summary": _train_eval_failure_summary(train_eval_report),
        "case_reports": case_reports,
        "override_traces": traces,
        "aggregates": _autopsy_aggregates(
            traces=traces,
            train_eval_report=train_eval_report,
            case_reports=case_reports,
        ),
        "failure_cause_assessment": causes,
        "recommended_next_route": recommended,
    }


def _rerun_failure_case(
    *,
    fixture: str,
    seed: int,
    ticks: int,
    artifact: Mapping[str, object],
    dataset_by_index: Mapping[int, Mapping[str, object]],
    seed_delta: Mapping[tuple[str, int], Mapping[str, object]],
) -> dict[str, object]:
    policy = MindV3EvolutionPolicy(
        seed=int(seed),
        support_residual_artifact=artifact,
        support_residual_runtime_mode="live",
    )
    if fixture == "carrion_only":
        world = evaluate_cli._fixture_world(
            fixture_name="carrion_only",
            seed=int(seed),
            ticks=int(ticks),
            policy=policy,
        )
    else:
        world = SimulationWorld(
            WorldConfig(seed=int(seed), max_ticks=int(ticks)),
            policy=policy,
        )
    summary = evaluate_cli._run_world(
        world=world,
        seed=int(seed),
        ticks=int(ticks),
        trajectory_output_path=None,
        trajectory_split_id="m3_safe_archive_failure_autopsy",
    )
    delta = dict(seed_delta.get((fixture, int(seed)), {}))
    traces = []
    trajectory_records = [
        _mapping(record) for record in world.trajectory_records
    ]
    diagnostic_records = list(world.policy_decision_diagnostics_records)
    missing_trajectory_records = 0
    for record_index, diagnostic in enumerate(diagnostic_records):
        diagnostic_payload = _mapping(diagnostic)
        if diagnostic_payload.get("support_residual_override_applied") is not True:
            continue
        if record_index < len(trajectory_records):
            trajectory = trajectory_records[record_index]
        else:
            trajectory = {}
            missing_trajectory_records += 1
        traces.append(
            _override_trace(
                fixture=fixture,
                seed=int(seed),
                record_index=record_index,
                diagnostic=diagnostic_payload,
                trajectory=trajectory,
                artifact=artifact,
                dataset_by_index=dataset_by_index,
                seed_delta=delta,
            )
        )
    return {
        "fixture": fixture,
        "seed": int(seed),
        "ticks": int(ticks),
        "summary": {
            "alive_agents": summary.get("alive_agents"),
            "births": summary.get("births"),
            "deaths": summary.get("deaths"),
            "unsupported_resolved_action_count": summary.get(
                "unsupported_resolved_action_count"
            ),
            "unsupported_requested_action_count": summary.get(
                "unsupported_requested_action_count"
            ),
        },
        "trace_alignment": {
            "diagnostic_record_count": len(diagnostic_records),
            "trajectory_record_count": len(trajectory_records),
            "missing_trajectory_record_count": missing_trajectory_records,
            "diagnostic_records_truncated": False,
        },
        "seed_delta": delta,
        "applied_override_count": len(traces),
        "applied_override_action_counts": _counter_dict(
            Counter(str(trace.get("selected_support_action")) for trace in traces)
        ),
        "override_traces": traces,
    }


def _override_trace(
    *,
    fixture: str,
    seed: int,
    record_index: int,
    diagnostic: Mapping[str, object],
    trajectory: Mapping[str, object],
    artifact: Mapping[str, object],
    dataset_by_index: Mapping[int, Mapping[str, object]],
    seed_delta: Mapping[str, object],
) -> dict[str, object]:
    selected_action = str(diagnostic.get("support_residual_proposed_action", ""))
    nearest = _nearest_support_example(
        artifact=artifact,
        observation_input=_mapping(trajectory.get("observation_input")),
        action_mask=_mapping(trajectory.get("action_mask")),
        action=selected_action,
    )
    support_index = _int(nearest.get("support_example_index"), default=-1)
    support_row = dataset_by_index.get(support_index, {})
    support_meta = _support_row_metadata(support_row)
    action_mask = _mapping(trajectory.get("action_mask"))
    resolved_valid = trajectory.get("resolution_action_valid") is True
    return {
        "fixture": fixture,
        "seed": int(seed),
        "tick": trajectory.get("tick"),
        "agent_id": trajectory.get("agent_id"),
        "record_index": int(record_index),
        "linear_action": diagnostic.get("support_residual_linear_action"),
        "selected_support_action": selected_action,
        "final_requested_action": trajectory.get("requested_action"),
        "final_resolved_action": trajectory.get("resolved_action"),
        "override_applied": diagnostic.get("support_residual_override_applied") is True,
        "nearest_support_distance": diagnostic.get(
            "support_residual_nearest_support_distance"
        ),
        "computed_nearest_support_distance": nearest.get("distance"),
        "score_margin": diagnostic.get("support_residual_score_margin"),
        "support_example_index": (
            support_index if support_index >= 0 else None
        ),
        "joined_support_row_metadata": support_meta,
        "action_mask_legality": {
            "selected_support_action_legal": bool(action_mask.get(selected_action)),
            "final_requested_action_valid": trajectory.get("action_valid") is True,
            "final_resolved_action_valid": resolved_valid,
            "invalid_reason": _mapping(trajectory.get("outcome")).get("invalid_reason"),
        },
        "resolved_invalid_contribution": 0 if resolved_valid else 1,
        "seed_resolved_invalid_delta": seed_delta.get(
            "resolved_invalid_action_count_delta"
        ),
        "failed_seed_fixture": True,
        "support_public_history_steps": diagnostic.get(
            "support_residual_public_history_steps"
        ),
        "support_index_reconstruction_policy": (
            "current_public_observation_action_mask_exact_when_history_steps_zero"
        ),
    }


def _validate_autopsy_inputs(
    *,
    support_artifact: Mapping[str, object],
    train_eval_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    branch_evidence: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    artifact_digest = stable_payload_digest(support_artifact)
    dataset_digest = stable_payload_digest(list(dataset_rows))
    branch_evidence_digest = branch_evidence.get("branch_evidence_digest")
    classification = _mapping(train_eval_report.get("classification")).get("primary")
    acceptance = _mapping(train_eval_report.get("acceptance"))
    validation = _mapping(train_eval_report.get("validation"))
    inputs = _mapping(train_eval_report.get("inputs"))
    report_artifact = _mapping(train_eval_report.get("artifact"))

    for key in (
        "diagnostics_only",
        "training_authorized",
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
    ):
        expected = True if key == "diagnostics_only" else False
        if train_eval_report.get(key) is not expected:
            failures.append(f"train_eval_report_{key}_not_{str(expected).lower()}")

    for key in (
        "diagnostics_only",
        "training_authorized",
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
    ):
        expected = True if key == "diagnostics_only" else False
        if support_artifact.get(key) is not expected:
            failures.append(f"artifact_{key}_not_{str(expected).lower()}")

    if classification != M3_SAFE_ARCHIVE_TRAIN_EVAL_EXPECTED_CLASSIFICATION:
        failures.append("train_eval_report_not_failed_non_promotional")
    if acceptance.get("passed") is not False:
        failures.append("train_eval_acceptance_not_failed")
    if not _list_of_mappings(acceptance.get("blockers")):
        failures.append("train_eval_acceptance_blockers_missing")
    if validation.get("passed") is not True:
        failures.append("train_eval_input_validation_not_passed")
    if list(validation.get("failures") or []) != []:
        failures.append("train_eval_input_validation_failures_present")
    if report_artifact.get("digest") != artifact_digest:
        failures.append("artifact_digest_mismatch")
    if validation.get("dataset_digest") != dataset_digest:
        failures.append("dataset_digest_mismatch")
    if inputs.get("expected_dataset_digest") != dataset_digest:
        failures.append("expected_dataset_digest_mismatch")
    if validation.get("branch_evidence_digest") != branch_evidence_digest:
        failures.append("branch_evidence_digest_mismatch")
    if inputs.get("expected_branch_evidence_digest") != branch_evidence_digest:
        failures.append("expected_branch_evidence_digest_mismatch")
    if not dataset_rows:
        failures.append("dataset_rows_missing")

    if failures:
        raise ValueError(
            "safe archive failure autopsy input validation failed: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "policy": "m3_safe_archive_failure_autopsy_input_validation_v1",
        "passed": True,
        "failures": [],
        "artifact_digest": artifact_digest,
        "dataset_digest": dataset_digest,
        "branch_evidence_digest": branch_evidence_digest,
        "train_eval_classification": classification,
    }


def _nearest_support_example(
    *,
    artifact: Mapping[str, object],
    observation_input: Mapping[str, object],
    action_mask: Mapping[str, object],
    action: str,
) -> dict[str, object]:
    row = planner_distilled_runtime_row(
        observation_input=observation_input,
        action_mask=action_mask,
        public_history_trace=(),
    )
    features = candidate_feature_vector(row, action)
    best_index: int | None = None
    best_distance = float("inf")
    for index, example in enumerate(_list_of_mappings(artifact.get("support_examples"))):
        if str(example.get("action")) != action:
            continue
        vector = _float_list(example.get("feature_vector"))
        if len(vector) != len(features):
            continue
        distance = sum((float(a) - float(b)) ** 2 for a, b in zip(features, vector))
        if distance < best_distance:
            best_distance = distance
            best_index = _int(example.get("example_index"), default=index)
    return {
        "support_example_index": best_index,
        "distance": _round(best_distance) if best_index is not None else None,
    }


def _support_row_metadata(row: Mapping[str, object]) -> dict[str, object]:
    metadata = _mapping(row.get("metadata"))
    trainable = _mapping(row.get("trainable"))
    label = _mapping(trainable.get("label"))
    return {
        "source_fixture": metadata.get("fixture"),
        "source_seed": metadata.get("seed"),
        "branch_id": metadata.get("branch_id"),
        "label_action": label.get("action"),
    }


def _seed_delta_by_case(
    train_eval_report: Mapping[str, object],
) -> dict[tuple[str, int], dict[str, object]]:
    metrics = _mapping(_mapping(train_eval_report.get("acceptance")).get("metrics"))
    result: dict[tuple[str, int], dict[str, object]] = {}
    for fixture, key in (
        ("broad", "broad_per_seed_delta"),
        ("carrion_only", "carrion_per_seed_delta"),
    ):
        for row in _list_of_mappings(metrics.get(key)):
            result[(fixture, _int(row.get("seed")))] = dict(row)
    return result


def _autopsy_aggregates(
    *,
    traces: Sequence[Mapping[str, object]],
    train_eval_report: Mapping[str, object],
    case_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_fixture_seed_action = Counter(
        (
            str(trace.get("fixture")),
            _int(trace.get("seed")),
            str(trace.get("selected_support_action")),
        )
        for trace in traces
    )
    failed_distribution = Counter(str(trace.get("selected_support_action")) for trace in traces)
    source_rows = Counter(_source_support_key(trace) for trace in traces)
    same_seed = Counter(
        "same_seed"
        if _int(_mapping(trace.get("joined_support_row_metadata")).get("source_seed"), default=-999)
        == _int(trace.get("seed"), default=-1)
        else "cross_seed"
        for trace in traces
    )
    resolved_invalid_traces = [
        trace for trace in traces if _int(trace.get("seed_resolved_invalid_delta")) > 0
    ]
    resolved_invalid_by_action = Counter(
        str(trace.get("selected_support_action")) for trace in resolved_invalid_traces
    )
    resolved_invalid_by_source = Counter(
        _source_support_key(trace) for trace in resolved_invalid_traces
    )
    zero_support = _zero_support_behavior(
        train_eval_report=train_eval_report,
        case_reports=case_reports,
    )
    return {
        "override_count": len(traces),
        "overrides_by_fixture_seed_action": [
            {
                "fixture": fixture,
                "seed": seed,
                "action": action,
                "count": count,
            }
            for (fixture, seed, action), count in sorted(by_fixture_seed_action.items())
        ],
        "failed_seed_override_action_distribution": _counter_dict(failed_distribution),
        "source_support_rows_used_by_failing_overrides": [
            {"source": key, "count": count}
            for key, count in sorted(source_rows.items())
        ],
        "same_seed_vs_cross_seed_support_usage": _counter_dict(same_seed),
        "zero_support_strict_seed_behavior": zero_support,
        "resolved_invalid_increases_by_action": _counter_dict(resolved_invalid_by_action),
        "resolved_invalid_increases_by_source_support_row": [
            {"source": key, "count": count}
            for key, count in sorted(resolved_invalid_by_source.items())
        ],
        "broad_seed_19_override_trace": [
            dict(trace)
            for trace in traces
            if trace.get("fixture") == "broad" and _int(trace.get("seed")) == 19
        ],
        "carrion_seed_13_37_43_override_trace": [
            dict(trace)
            for trace in traces
            if trace.get("fixture") == "carrion_only"
            and _int(trace.get("seed")) in {13, 37, 43}
        ],
    }


def _zero_support_behavior(
    *,
    train_eval_report: Mapping[str, object],
    case_reports: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    preflight = _mapping(train_eval_report.get("preflight"))
    applied_by_case = {
        (str(case.get("fixture")), _int(case.get("seed"))): _int(
            case.get("applied_override_count")
        )
        for case in case_reports
    }
    return [
        {
            "fixture": item.get("fixture"),
            "seed": item.get("seed"),
            "rerun_in_autopsy": (
                (str(item.get("fixture")), _int(item.get("seed"))) in applied_by_case
            ),
            "applied_override_count": applied_by_case.get(
                (str(item.get("fixture")), _int(item.get("seed"))),
                0,
            ),
            "interpretation": "runtime_can_transfer_cross_seed_support_without_local_labels",
        }
        for item in _list_of_mappings(preflight.get("zero_support_strict_seeds"))
    ]


def _failure_cause_assessment(
    *,
    traces: Sequence[Mapping[str, object]],
    train_eval_report: Mapping[str, object],
    case_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    zero_support_targets = [
        item
        for item in _zero_support_behavior(
            train_eval_report=train_eval_report,
            case_reports=case_reports,
        )
        if item.get("rerun_in_autopsy") is True
    ]
    cross_seed_count = sum(
        1
        for trace in traces
        if _int(_mapping(trace.get("joined_support_row_metadata")).get("source_seed"), default=-999)
        != _int(trace.get("seed"), default=-1)
    )
    zero_distance_failures = sum(
        1
        for trace in traces
        if _float(trace.get("nearest_support_distance"), default=1.0) == 0.0
    )
    broad19 = [
        case
        for case in case_reports
        if case.get("fixture") == "broad" and _int(case.get("seed")) == 19
    ]
    broad19_runtime = bool(
        broad19
        and _int(_mapping(broad19[0].get("seed_delta")).get("alive_delta")) < 0
        and _int(
            _mapping(broad19[0].get("seed_delta")).get(
                "resolved_invalid_action_count_delta"
            )
        )
        <= 0
    )
    return {
        "support_scarcity": {
            "present": bool(zero_support_targets),
            "evidence": zero_support_targets,
        },
        "bad_label_transfer": {
            "present": cross_seed_count > 0,
            "cross_seed_override_count": cross_seed_count,
        },
        "one_step_state_aliasing": {
            "present": zero_distance_failures > 0,
            "zero_distance_failing_override_count": zero_distance_failures,
            "rationale": (
                "nearest support can be exact in one-step public features while "
                "strict live outcomes still regress"
            ),
        },
        "runtime_interaction_effects": {
            "present": broad19_runtime,
            "rationale": (
                "broad seed 19 regressed alive/births without a resolved-invalid "
                "increase, indicating multi-step interaction effects"
            ),
        },
    }


def _recommended_next_route(causes: Mapping[str, object]) -> str:
    if _mapping(causes.get("runtime_interaction_effects")).get("present") is True:
        return "build_sequence_or_rollout_context_archive"
    if _mapping(causes.get("one_step_state_aliasing")).get("present") is True:
        return "build_sequence_or_rollout_context_archive"
    if _mapping(causes.get("support_scarcity")).get("present") is True:
        return "expand_targeted_carrion_alive_archive"
    if _mapping(causes.get("bad_label_transfer")).get("present") is True:
        return "repair_support_labels_before_training"
    return "stop_support_gated_residual_family"


def _train_eval_failure_summary(report: Mapping[str, object]) -> dict[str, object]:
    acceptance = _mapping(report.get("acceptance"))
    return {
        "classification": _mapping(report.get("classification")).get("primary"),
        "passed": acceptance.get("passed") is True,
        "blockers": _list_of_mappings(acceptance.get("blockers")),
        "training_authorized": report.get("training_authorized") is True,
        "promotion_authorized": report.get("promotion_authorized") is True,
        "runtime_promotion_allowed": report.get("runtime_promotion_allowed") is True,
    }


def _source_support_key(trace: Mapping[str, object]) -> str:
    meta = _mapping(trace.get("joined_support_row_metadata"))
    return (
        f"{meta.get('source_fixture')}:{meta.get('source_seed')}:"
        f"{meta.get('label_action')}:{meta.get('branch_id')}"
    )


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return []
    return result


def _int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: float) -> float:
    return round(float(value), 6)
