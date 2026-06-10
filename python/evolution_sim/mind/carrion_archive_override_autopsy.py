from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind import evaluation_harness as evaluate_cli
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.support_gated_residual import (
    load_support_gated_residual_artifact,
)
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION = (
    "m3_carrion_archive_override_autopsy_report_v1"
)
M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_POLICY = (
    "diagnostics_only_m3_carrion_archive_override_autopsy_v1"
)
V148_CARRION_ARCHIVE_REPORT_SCHEMA_VERSION = (
    "m3_carrion_specific_archive_expansion_report_v1"
)
V148_CARRION_ARCHIVE_REPORT_POLICY = (
    "diagnostics_only_m3_carrion_specific_archive_expansion_v1"
)
V149_CARRION_TRAIN_EVAL_SCHEMA_VERSION = (
    "m3_carrion_specific_archive_train_eval_report_v1"
)
V149_CARRION_TRAIN_EVAL_POLICY = (
    "diagnostics_only_m3_carrion_specific_archive_support_gated_train_eval_v1"
)
V149_FAILED_NON_PROMOTIONAL_CLASSIFICATION = (
    "m3_safe_archive_diagnostic_failed_non_promotional"
)

DEFAULT_V148_CARRION_ARCHIVE_REPORT_PATH = Path(
    "output/mind/shards/v148-carrion/merged-report.json"
)
DEFAULT_V148_CARRION_ARCHIVE_DATASET_PATH = Path(
    "output/mind/shards/v148-carrion/merged-dataset.jsonl"
)
DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH = Path(
    "output/mind/mind-v3-v149-carrion-specific-archive-train-eval.json"
)
DEFAULT_V149_CARRION_ARTIFACT_PATH = Path(
    "output/mind/mind-v3-v149-carrion-specific-archive-support-gated-artifact.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v150-carrion-archive-override-autopsy.json"
)
DEFAULT_CARRION_SEEDS = (13, 19, 29, 37, 41, 43)
DEFAULT_TICKS = 120
FORBIDDEN_TRAINABLE_TOKENS = (
    "seed",
    "fixture",
    "branch",
    "agent",
    "tick",
    "path",
    "digest",
    "private",
    "world",
    "future",
)


class CarrionArchiveOverrideAutopsyError(ValueError):
    pass


def load_json_report(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CarrionArchiveOverrideAutopsyError(
            f"JSON report must be an object: {path}"
        )
    return payload


def load_jsonl_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise CarrionArchiveOverrideAutopsyError(
                    f"JSONL row {line_number} must be an object: {path}"
                )
            rows.append(payload)
    return rows


def write_carrion_archive_override_autopsy_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def build_carrion_archive_override_autopsy_report(
    *,
    archive_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    train_eval_report: Mapping[str, object],
    artifact: Mapping[str, object],
    ticks: int = DEFAULT_TICKS,
    target_seeds: Sequence[int] = DEFAULT_CARRION_SEEDS,
    rerun_cases: Sequence[Mapping[str, object]] | None = None,
    input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    support_artifact = load_support_gated_residual_artifact(artifact)
    validation = _validate_inputs(
        archive_report=archive_report,
        dataset_rows=dataset_rows,
        train_eval_report=train_eval_report,
        artifact=support_artifact,
    )
    dataset_by_index = _dataset_by_index(dataset_rows)
    branch_by_id = _branch_result_by_id(archive_report)
    seed_delta = _carrion_seed_delta(train_eval_report)
    seeds = tuple(int(seed) for seed in target_seeds)
    if rerun_cases is None:
        raw_cases = [
            _rerun_carrion_case(
                seed=int(seed),
                ticks=int(ticks),
                artifact=support_artifact,
                seed_delta=seed_delta.get(int(seed), {}),
            )
            for seed in seeds
        ]
    else:
        raw_cases = [dict(case) for case in rerun_cases]
    case_reports = [
        _join_case_report(
            case=case,
            dataset_by_index=dataset_by_index,
            branch_by_id=branch_by_id,
            seed_delta=seed_delta,
        )
        for case in raw_cases
    ]
    traces = [
        dict(trace)
        for case in case_reports
        for trace in _list_of_mappings(case.get("override_traces"))
    ]
    override_count = len(traces)
    expected_override_count = _int(
        _mapping(_mapping(train_eval_report.get("acceptance")).get("metrics")).get(
            "carrion_applied_override_count"
        )
    )
    complete = (
        override_count == expected_override_count
        and {int(case.get("seed", -1)) for case in case_reports} == set(seeds)
    )
    failure_modes = _failure_mode_classification(
        traces=traces,
        seed_delta=seed_delta,
    )
    source_map = _support_label_action_failure_map(traces)
    classification = (
        "m3_carrion_archive_override_autopsy_complete_no_training"
        if complete
        else "m3_carrion_archive_override_autopsy_incomplete_no_training"
    )
    report = {
        "schema_version": M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION,
        "policy": M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_POLICY,
        "classification": {"primary": classification, "labels": [classification]},
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "contract": {
            "diagnostics_only": True,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
            "gate_relaxation": False,
            "trainable_inputs_changed": False,
            "trainable_input_surface": "none_added_diagnostics_only",
            "fixture_identity_private_world_state_trainable_input": False,
            "live_rerun_scope": "opt_in_carrion_fixture_replay_only",
        },
        "inputs": {
            **_input_paths(input_paths),
            "artifact_digest": stable_payload_digest(support_artifact),
            "train_eval_report_digest": stable_payload_digest(train_eval_report),
            "dataset_digest": stable_payload_digest(list(dataset_rows)),
            "branch_evidence_digest": archive_report.get("branch_evidence_digest"),
            "ticks": int(ticks),
            "target_fixture": "carrion_only",
            "target_seeds": [int(seed) for seed in seeds],
            "input_validation": validation,
        },
        "artifact": {
            "schema_version": support_artifact.get("schema_version"),
            "policy": support_artifact.get("policy"),
            "training_row_count": support_artifact.get("training_row_count"),
            "support_action_counts": support_artifact.get("support_action_counts"),
            "support_gate": support_artifact.get("support_gate"),
            "inference_contract": support_artifact.get("inference_contract"),
            "training_authorized": support_artifact.get("training_authorized") is True,
            "promotion_authorized": support_artifact.get("promotion_authorized") is True,
            "runtime_promotion_allowed": (
                support_artifact.get("runtime_promotion_allowed") is True
            ),
        },
        "train_eval_failure_summary": _train_eval_failure_summary(
            train_eval_report
        ),
        "case_reports": case_reports,
        "override_traces": traces,
        "aggregates": _aggregates(
            traces=traces,
            case_reports=case_reports,
            expected_override_count=expected_override_count,
        ),
        "support_label_action_failure_map": source_map,
        "seed_blocker_join": _seed_blocker_join(
            traces=traces,
            seed_delta=seed_delta,
            train_eval_report=train_eval_report,
        ),
        "failure_mode_classification": failure_modes,
        "recommended_next_route": _recommended_next_route(failure_modes),
    }
    exact_payload = {
        "schema_version": report["schema_version"],
        "policy": report["policy"],
        "classification": report["classification"],
        "inputs": {
            "artifact_digest": report["inputs"]["artifact_digest"],
            "dataset_digest": report["inputs"]["dataset_digest"],
            "branch_evidence_digest": report["inputs"]["branch_evidence_digest"],
        },
        "aggregates": report["aggregates"],
        "support_label_action_failure_map": source_map,
        "failure_mode_classification": failure_modes,
    }
    report["exact_digest"] = stable_payload_digest(exact_payload)
    report["provenance"] = {
        "exact_digest_payload_policy": (
            "stable_payload_digest_of_v150_carrion_override_autopsy_core_v1"
        ),
        "exact_digest": report["exact_digest"],
    }
    return report


def trainable_input_leakage_scan(
    dataset_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for row_index, row in enumerate(dataset_rows):
        trainable = _mapping(row.get("trainable"))
        for path, value in _flatten(trainable):
            lower_path = path.lower()
            if any(token in lower_path for token in FORBIDDEN_TRAINABLE_TOKENS):
                failures.append(
                    {
                        "row_index": int(row_index),
                        "path": path,
                        "reason": "forbidden_trainable_path_token",
                    }
                )
            if isinstance(value, str) and _looks_like_digest(value):
                failures.append(
                    {
                        "row_index": int(row_index),
                        "path": path,
                        "reason": "forbidden_trainable_digest_value",
                    }
                )
    return {
        "policy": "m3_carrion_archive_override_autopsy_trainable_leakage_scan_v1",
        "passed": not failures,
        "forbidden_tokens": list(FORBIDDEN_TRAINABLE_TOKENS),
        "failure_count": len(failures),
        "failures": failures[:32],
    }


def _validate_inputs(
    *,
    archive_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    train_eval_report: Mapping[str, object],
    artifact: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    artifact_digest = stable_payload_digest(artifact)
    dataset_digest = stable_payload_digest(list(dataset_rows))
    branch_evidence_digest = archive_report.get("branch_evidence_digest")
    archive_dataset = _mapping(archive_report.get("dataset"))
    archive_integrity = _mapping(archive_report.get("source_integrity"))
    archive_status = _mapping(archive_report.get("generation_status"))
    archive_branch_results = _list_of_mappings(archive_report.get("branch_results"))
    train_validation = _mapping(train_eval_report.get("validation"))
    train_inputs = _mapping(train_eval_report.get("inputs"))
    train_artifact = _mapping(train_eval_report.get("artifact"))
    train_acceptance = _mapping(train_eval_report.get("acceptance"))
    train_classification = _mapping(train_eval_report.get("classification")).get(
        "primary"
    )
    leakage_scan = trainable_input_leakage_scan(dataset_rows)

    if archive_report.get("schema_version") != V148_CARRION_ARCHIVE_REPORT_SCHEMA_VERSION:
        failures.append("archive_report_schema_mismatch")
    if archive_report.get("policy") != V148_CARRION_ARCHIVE_REPORT_POLICY:
        failures.append("archive_report_policy_mismatch")
    if archive_report.get("diagnostics_only") is not True:
        failures.append("archive_report_diagnostics_only_not_true")
    if archive_integrity.get("passed") is not True:
        failures.append("archive_report_source_integrity_not_passed")
    if _list(archive_integrity.get("failures")):
        failures.append("archive_report_source_integrity_failures_present")
    if archive_status.get("state") != "complete" or archive_status.get("partial") is True:
        failures.append("archive_report_generation_not_complete")
    if archive_dataset.get("dataset_digest") != dataset_digest:
        failures.append("archive_report_dataset_digest_mismatch")
    if _int(archive_dataset.get("safe_label_count")) != len(dataset_rows):
        failures.append("archive_report_safe_label_count_mismatch")
    if not archive_branch_results:
        failures.append("archive_report_branch_results_missing")
    if _int(archive_report.get("branch_result_count")) != len(archive_branch_results):
        failures.append("archive_report_branch_result_count_mismatch")
    if branch_evidence_digest != stable_payload_digest(archive_branch_results):
        failures.append("archive_report_branch_evidence_digest_mismatch")
    for key in (
        "training_authorized",
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
    ):
        if archive_report.get(key) is not False:
            failures.append(f"archive_report_{key}_not_false")

    if train_eval_report.get("schema_version") != V149_CARRION_TRAIN_EVAL_SCHEMA_VERSION:
        failures.append("train_eval_schema_mismatch")
    if train_eval_report.get("policy") != V149_CARRION_TRAIN_EVAL_POLICY:
        failures.append("train_eval_policy_mismatch")
    if train_classification != V149_FAILED_NON_PROMOTIONAL_CLASSIFICATION:
        failures.append("train_eval_not_failed_non_promotional")
    if train_acceptance.get("passed") is not False:
        failures.append("train_eval_acceptance_not_failed")
    if not _list_of_mappings(train_acceptance.get("blockers")):
        failures.append("train_eval_blockers_missing")
    if train_validation.get("passed") is not True:
        failures.append("train_eval_validation_not_passed")
    if _list(train_validation.get("failures")):
        failures.append("train_eval_validation_failures_present")

    for key in (
        "diagnostics_only",
        "training_authorized",
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
    ):
        expected = True if key == "diagnostics_only" else False
        if train_eval_report.get(key) is not expected:
            failures.append(f"train_eval_{key}_not_{str(expected).lower()}")
        if artifact.get(key) is not expected:
            failures.append(f"artifact_{key}_not_{str(expected).lower()}")

    if train_artifact.get("digest") != artifact_digest:
        failures.append("artifact_digest_mismatch")
    if train_validation.get("dataset_digest") != dataset_digest:
        failures.append("dataset_digest_mismatch")
    if train_inputs.get("expected_dataset_digest") != dataset_digest:
        failures.append("expected_dataset_digest_mismatch")
    if train_validation.get("branch_evidence_digest") != branch_evidence_digest:
        failures.append("branch_evidence_digest_mismatch")
    if train_inputs.get("expected_branch_evidence_digest") != branch_evidence_digest:
        failures.append("expected_branch_evidence_digest_mismatch")
    if not dataset_rows:
        failures.append("dataset_rows_missing")
    if leakage_scan.get("passed") is not True:
        failures.append("trainable_input_leakage_detected")

    inference = _mapping(artifact.get("inference_contract"))
    if inference.get("uses_fixture_id_as_runtime_feature") is True:
        failures.append("artifact_uses_fixture_id_as_runtime_feature")
    if inference.get("uses_private_simulator_state") is True:
        failures.append("artifact_uses_private_simulator_state")
    if inference.get("uses_seed_id_as_runtime_feature") is True:
        failures.append("artifact_uses_seed_id_as_runtime_feature")

    if failures:
        raise CarrionArchiveOverrideAutopsyError(
            "carrion archive override autopsy input validation failed: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "policy": "m3_carrion_archive_override_autopsy_input_validation_v1",
        "passed": True,
        "failures": [],
        "artifact_digest": artifact_digest,
        "dataset_digest": dataset_digest,
        "branch_evidence_digest": branch_evidence_digest,
        "train_eval_classification": train_classification,
        "trainable_leakage_scan": leakage_scan,
    }


def _rerun_carrion_case(
    *,
    seed: int,
    ticks: int,
    artifact: Mapping[str, object],
    seed_delta: Mapping[str, object],
) -> dict[str, object]:
    policy = MindV3EvolutionPolicy(
        seed=int(seed),
        support_residual_artifact=artifact,
        support_residual_runtime_mode="live",
    )
    world = evaluate_cli._fixture_world(
        fixture_name="carrion_only",
        seed=int(seed),
        ticks=int(ticks),
        policy=policy,
    )
    summary = evaluate_cli._run_world(
        world=world,
        seed=int(seed),
        ticks=int(ticks),
        trajectory_output_path=None,
        trajectory_split_id="m3_carrion_archive_override_autopsy",
    )
    trajectory_records = [_mapping(record) for record in world.trajectory_records]
    diagnostics = [_mapping(record) for record in world.policy_decision_diagnostics_records]
    traces = []
    missing_trajectory_records = 0
    for record_index, diagnostic in enumerate(diagnostics):
        if diagnostic.get("support_residual_override_applied") is not True:
            continue
        if record_index < len(trajectory_records):
            trajectory = trajectory_records[record_index]
        else:
            trajectory = {}
            missing_trajectory_records += 1
        traces.append(
            _raw_override_trace(
                fixture="carrion_only",
                seed=int(seed),
                record_index=record_index,
                diagnostic=diagnostic,
                trajectory=trajectory,
                artifact=artifact,
                seed_delta=seed_delta,
            )
        )
    return {
        "fixture": "carrion_only",
        "seed": int(seed),
        "ticks": int(ticks),
        "summary": {
            "alive_agents": summary.get("alive_agents"),
            "births": summary.get("births"),
            "deaths": summary.get("deaths"),
            "resolved_invalid_action_count": summary.get(
                "resolved_invalid_action_count"
            ),
            "unsupported_requested_action_count": summary.get(
                "unsupported_requested_action_count"
            ),
            "unsupported_resolved_action_count": summary.get(
                "unsupported_resolved_action_count"
            ),
        },
        "trace_alignment": {
            "diagnostic_record_count": len(diagnostics),
            "trajectory_record_count": len(trajectory_records),
            "missing_trajectory_record_count": missing_trajectory_records,
            "diagnostic_records_truncated": False,
        },
        "seed_delta": dict(seed_delta),
        "applied_override_count": len(traces),
        "applied_override_action_counts": _counter_dict(
            Counter(str(trace.get("selected_support_action")) for trace in traces)
        ),
        "override_traces": traces,
    }


def _raw_override_trace(
    *,
    fixture: str,
    seed: int,
    record_index: int,
    diagnostic: Mapping[str, object],
    trajectory: Mapping[str, object],
    artifact: Mapping[str, object],
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
    action_mask = _mapping(trajectory.get("action_mask"))
    resolution_mask = _mapping(trajectory.get("resolution_action_mask"))
    outcome = _mapping(trajectory.get("outcome"))
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
        "override_applied": True,
        "nearest_support_distance": diagnostic.get(
            "support_residual_nearest_support_distance"
        ),
        "computed_nearest_support_distance": nearest.get("distance"),
        "score_margin": diagnostic.get("support_residual_score_margin"),
        "support_example_index": support_index if support_index >= 0 else None,
        "action_mask_legality": {
            "selected_support_action_legal": bool(action_mask.get(selected_action)),
            "selected_support_action_resolution_legal": bool(
                resolution_mask.get(selected_action)
            ),
            "final_requested_action_valid": trajectory.get("action_valid") is True,
            "final_resolved_action_valid": (
                trajectory.get("resolution_action_valid") is True
            ),
            "invalid_reason": outcome.get("invalid_reason"),
        },
        "live_public_outcome_summary": _public_outcome_summary(trajectory),
        "resolved_invalid_contribution": (
            0 if trajectory.get("resolution_action_valid") is True else 1
        ),
        "seed_alive_delta": seed_delta.get("alive_delta"),
        "seed_births_delta": seed_delta.get("births_delta"),
        "seed_resolved_invalid_delta": seed_delta.get(
            "resolved_invalid_action_count_delta"
        ),
        "support_public_history_steps": diagnostic.get(
            "support_residual_public_history_steps"
        ),
        "support_candidate_scores_top": diagnostic.get(
            "support_residual_candidate_scores_top"
        ),
    }


def _join_case_report(
    *,
    case: Mapping[str, object],
    dataset_by_index: Mapping[int, Mapping[str, object]],
    branch_by_id: Mapping[str, Mapping[str, object]],
    seed_delta: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    seed = _int(case.get("seed"))
    delta = dict(case.get("seed_delta") or seed_delta.get(seed, {}))
    traces = [
        _join_override_trace(
            trace=trace,
            dataset_by_index=dataset_by_index,
            branch_by_id=branch_by_id,
            seed_delta=delta,
        )
        for trace in _list_of_mappings(case.get("override_traces"))
    ]
    result = dict(case)
    result["fixture"] = "carrion_only"
    result["seed"] = seed
    result["seed_delta"] = delta
    result["applied_override_count"] = len(traces)
    result["applied_override_action_counts"] = _counter_dict(
        Counter(str(trace.get("selected_support_action")) for trace in traces)
    )
    result["override_traces"] = traces
    return result


def _join_override_trace(
    *,
    trace: Mapping[str, object],
    dataset_by_index: Mapping[int, Mapping[str, object]],
    branch_by_id: Mapping[str, Mapping[str, object]],
    seed_delta: Mapping[str, object],
) -> dict[str, object]:
    support_index = _int(trace.get("support_example_index"), default=-1)
    support_row = dataset_by_index.get(support_index, {})
    metadata = _mapping(support_row.get("metadata"))
    trainable = _mapping(support_row.get("trainable"))
    label = _mapping(trainable.get("label"))
    branch_id = str(metadata.get("branch_id", ""))
    branch_result = branch_by_id.get(branch_id, {})
    action = str(trace.get("selected_support_action") or label.get("action") or "")
    action_run = _branch_action_run(branch_result, action)
    branch_context = _mapping(branch_result.get("carrion_archive_context"))
    result = dict(trace)
    result.update(
        {
            "support_join": {
                "joined": bool(support_row),
                "support_example_index": (
                    support_index if support_index >= 0 else None
                ),
                "support_row_schema_version": support_row.get("schema_version"),
                "source_fixture": metadata.get("fixture"),
                "source_seed": metadata.get("seed"),
                "source_branch_id": branch_id or None,
                "source_branch_tick": metadata.get("branch_tick"),
                "source_agent_id": metadata.get("agent_id"),
                "label_action": label.get("action"),
            },
            "branch_evidence_join": {
                "joined": bool(branch_result),
                "branch_reason": branch_context.get("branch_reason"),
                "reason_rank": branch_context.get("reason_rank"),
                "reason_evidence": branch_context.get("reason_evidence"),
                "source_candidate_actions": branch_result.get("candidate_actions"),
                "source_public_mask_action_legal": _source_mask_legal(
                    branch_result,
                    action,
                ),
                "source_action_run_found": bool(action_run),
                "source_action_run_digest": stable_payload_digest(action_run)
                if action_run
                else None,
                "source_action_replay_verified": _mapping(
                    action_run.get("replay_verification")
                ).get("verified")
                is True,
                "source_action_deltas": _source_action_deltas(action_run),
                "source_first_action_resolution": _first_action_resolution(
                    action_run
                ),
            },
            "regression_association": {
                "seed_alive_delta": seed_delta.get("alive_delta"),
                "seed_births_delta": seed_delta.get("births_delta"),
                "seed_resolved_invalid_delta": seed_delta.get(
                    "resolved_invalid_action_count_delta"
                ),
                "birth_regression_seed": _int(seed_delta.get("births_delta")) < 0,
                "resolved_invalid_increase_seed": _int(
                    seed_delta.get("resolved_invalid_action_count_delta")
                )
                > 0,
                "direct_invalid_override": _int(
                    trace.get("resolved_invalid_contribution")
                )
                > 0,
            },
            "autopsy_tags": _trace_tags(
                trace=trace,
                branch_context=branch_context,
                action_run=action_run,
                seed_delta=seed_delta,
            ),
        }
    )
    return result


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


def _public_outcome_summary(trajectory: Mapping[str, object]) -> dict[str, object]:
    outcome = _mapping(trajectory.get("outcome"))
    movement = _mapping(outcome.get("movement"))
    feeding = _mapping(outcome.get("feeding"))
    drinking = _mapping(outcome.get("drinking"))
    return {
        "requested_action": outcome.get("requested_action"),
        "resolved_action": outcome.get("resolved_action"),
        "observation_action_valid": outcome.get("observation_action_valid") is True,
        "resolution_action_valid": outcome.get("resolution_action_valid") is True,
        "invalid_reason": outcome.get("invalid_reason"),
        "moved": movement.get("moved", trajectory.get("moved")),
        "ate": feeding.get("ate"),
        "food_source": feeding.get("food_source"),
        "drank": drinking.get("drank"),
        "reproduced": outcome.get("reproduced") is True,
        "reproduction_ready_after": outcome.get("reproduction_ready_after") is True,
        "resource_gain": outcome.get("resource_gain"),
    }


def _source_action_deltas(action_run: Mapping[str, object]) -> dict[str, object]:
    return {
        "deltas_vs_baseline": action_run.get("deltas_vs_baseline"),
        "deltas_vs_v142_override": action_run.get("deltas_vs_v142_override"),
        "target_terminal": action_run.get("target_terminal"),
    }


def _first_action_resolution(action_run: Mapping[str, object]) -> dict[str, object]:
    first = _mapping(action_run.get("first_action_outcome"))
    outcome = _mapping(first.get("outcome"))
    movement = _mapping(outcome.get("movement"))
    feeding = _mapping(outcome.get("feeding"))
    return {
        "forced_action": action_run.get("forced_action"),
        "requested_action": first.get("requested_action") or outcome.get("requested_action"),
        "resolved_action": first.get("resolved_action") or outcome.get("resolved_action"),
        "action_valid": first.get("action_valid") is True
        or outcome.get("observation_action_valid") is True,
        "resolution_action_valid": first.get("resolution_action_valid") is True
        or outcome.get("resolution_action_valid") is True,
        "invalid_reason": outcome.get("invalid_reason"),
        "moved": movement.get("moved"),
        "ate": feeding.get("ate"),
        "food_source": feeding.get("food_source"),
        "reproduced": outcome.get("reproduced") is True,
        "reproduction_ready_after": outcome.get("reproduction_ready_after") is True,
        "resource_gain": outcome.get("resource_gain"),
    }


def _trace_tags(
    *,
    trace: Mapping[str, object],
    branch_context: Mapping[str, object],
    action_run: Mapping[str, object],
    seed_delta: Mapping[str, object],
) -> list[str]:
    tags: set[str] = set()
    action = str(trace.get("selected_support_action", ""))
    if _float(trace.get("nearest_support_distance"), default=1.0) == 0.0:
        tags.add("stale_one_step_aliasing")
    if action.startswith("move_") and str(branch_context.get("branch_reason")) in {
        "carrion_contact",
        "movement_stall",
        "post_carrion_hydration_risk",
    }:
        tags.add("movement_near_carrion_side_effect")
    if _int(seed_delta.get("births_delta")) < 0:
        tags.add("birth_regression_associated")
        first = _first_action_resolution(action_run)
        if first.get("reproduced") is not True:
            tags.add("missing_hydration_reproduction_context")
    if _int(seed_delta.get("resolved_invalid_action_count_delta")) > 0:
        tags.add("resolution_conflict_associated")
    if _int(trace.get("resolved_invalid_contribution")) > 0:
        tags.add("direct_invalid_override")
    return sorted(tags)


def _failure_mode_classification(
    *,
    traces: Sequence[Mapping[str, object]],
    seed_delta: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    zero_distance = [
        trace
        for trace in traces
        if _float(trace.get("nearest_support_distance"), default=1.0) == 0.0
    ]
    movement = [
        trace
        for trace in traces
        if str(trace.get("selected_support_action", "")).startswith("move_")
    ]
    birth_regression = [
        trace
        for trace in traces
        if _mapping(trace.get("regression_association")).get(
            "birth_regression_seed"
        )
        is True
    ]
    resolved_invalid = [
        trace
        for trace in traces
        if _mapping(trace.get("regression_association")).get(
            "resolved_invalid_increase_seed"
        )
        is True
    ]
    direct_invalid = [
        trace
        for trace in traces
        if _mapping(trace.get("regression_association")).get(
            "direct_invalid_override"
        )
        is True
    ]
    present_labels = []
    modes = {
        "stale_one_step_aliasing": {
            "present": bool(zero_distance),
            "override_count": len(zero_distance),
            "rationale": (
                "live overrides map to zero-distance one-step public support "
                "rows while strict seed outcomes regress"
            ),
        },
        "movement_near_carrion_side_effects": {
            "present": bool(movement),
            "override_count": len(movement),
            "actions": _counter_dict(
                Counter(str(trace.get("selected_support_action")) for trace in movement)
            ),
            "rationale": (
                "accepted support labels are movement actions selected in "
                "early carrion-contact or movement-stall contexts"
            ),
        },
        "missing_hydration_reproduction_context": {
            "present": bool(birth_regression),
            "override_count": len(birth_regression),
            "affected_seeds": sorted(
                {
                    _int(trace.get("seed"))
                    for trace in birth_regression
                    if _int(trace.get("seed")) >= 0
                }
            ),
            "rationale": (
                "one-step branch labels passed birth floors, but live seed "
                "birth deltas regress after movement overrides"
            ),
        },
        "resolution_conflict": {
            "present": bool(resolved_invalid or direct_invalid),
            "seed_delta_increase_override_count": len(resolved_invalid),
            "direct_invalid_override_count": len(direct_invalid),
            "affected_seeds": sorted(
                seed
                for seed, delta in seed_delta.items()
                if _int(delta.get("resolved_invalid_action_count_delta")) > 0
            ),
            "rationale": (
                "resolved-invalid increases are seed-level downstream effects "
                "unless direct_invalid_override_count is nonzero"
            ),
        },
    }
    for key, mode in modes.items():
        if mode["present"] is True:
            present_labels.append(key)
    if "resolution_conflict" in present_labels and modes["resolution_conflict"][
        "direct_invalid_override_count"
    ]:
        primary = "resolution_conflict"
    elif "missing_hydration_reproduction_context" in present_labels:
        primary = "missing_hydration_reproduction_context"
    elif "movement_near_carrion_side_effects" in present_labels:
        primary = "movement_near_carrion_side_effects"
    elif "stale_one_step_aliasing" in present_labels:
        primary = "stale_one_step_aliasing"
    else:
        primary = "unclassified"
    return {
        "policy": "m3_carrion_archive_override_autopsy_failure_modes_v1",
        "primary": primary,
        "labels": present_labels or [primary],
        "modes": modes,
    }


def _support_label_action_failure_map(
    traces: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for trace in traces:
        key = _source_support_key(trace)
        grouped.setdefault(key, []).append(trace)
    rows = []
    for key, group in grouped.items():
        first = group[0]
        support = _mapping(first.get("support_join"))
        branch = _mapping(first.get("branch_evidence_join"))
        birth_group = [
            trace
            for trace in group
            if _mapping(trace.get("regression_association")).get(
                "birth_regression_seed"
            )
            is True
        ]
        resolved_group = [
            trace
            for trace in group
            if _mapping(trace.get("regression_association")).get(
                "resolved_invalid_increase_seed"
            )
            is True
        ]
        rows.append(
            {
                "source": key,
                "support_example_index": support.get("support_example_index"),
                "source_seed": support.get("source_seed"),
                "source_branch_id": support.get("source_branch_id"),
                "source_branch_tick": support.get("source_branch_tick"),
                "source_branch_reason": branch.get("branch_reason"),
                "label_action": support.get("label_action"),
                "used_override_count": len(group),
                "used_on_seeds": sorted({_int(trace.get("seed")) for trace in group}),
                "birth_regression_override_count": len(birth_group),
                "birth_regression_seeds": sorted(
                    {_int(trace.get("seed")) for trace in birth_group}
                ),
                "resolved_invalid_increase_override_count": len(resolved_group),
                "resolved_invalid_increase_seeds": sorted(
                    {_int(trace.get("seed")) for trace in resolved_group}
                ),
                "direct_invalid_override_count": sum(
                    1
                    for trace in group
                    if _mapping(trace.get("regression_association")).get(
                        "direct_invalid_override"
                    )
                    is True
                ),
                "source_public_mask_action_legal": branch.get(
                    "source_public_mask_action_legal"
                ),
                "source_first_action_resolution": branch.get(
                    "source_first_action_resolution"
                ),
                "source_action_deltas": branch.get("source_action_deltas"),
                "autopsy_tags": sorted(
                    {
                        tag
                        for trace in group
                        for tag in _list(trace.get("autopsy_tags"))
                    }
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -_int(row.get("birth_regression_override_count")),
            -_int(row.get("resolved_invalid_increase_override_count")),
            str(row.get("label_action")),
            str(row.get("source")),
        ),
    )


def _seed_blocker_join(
    *,
    traces: Sequence[Mapping[str, object]],
    seed_delta: Mapping[int, Mapping[str, object]],
    train_eval_report: Mapping[str, object],
) -> list[dict[str, object]]:
    blockers_by_seed: dict[int, list[dict[str, object]]] = {}
    for blocker in _list_of_mappings(_mapping(train_eval_report.get("acceptance")).get("blockers")):
        if blocker.get("fixture") != "carrion_only":
            continue
        seed = _int(blocker.get("seed"), default=-1)
        if seed >= 0:
            blockers_by_seed.setdefault(seed, []).append(dict(blocker))
    traces_by_seed: dict[int, list[Mapping[str, object]]] = {}
    for trace in traces:
        traces_by_seed.setdefault(_int(trace.get("seed")), []).append(trace)
    rows = []
    for seed, delta in sorted(seed_delta.items()):
        seed_traces = traces_by_seed.get(seed, [])
        rows.append(
            {
                "fixture": "carrion_only",
                "seed": int(seed),
                "seed_delta": dict(delta),
                "blockers": blockers_by_seed.get(seed, []),
                "override_count": len(seed_traces),
                "override_action_counts": _counter_dict(
                    Counter(
                        str(trace.get("selected_support_action"))
                        for trace in seed_traces
                    )
                ),
                "support_sources": [
                    {
                        "source": _source_support_key(trace),
                        "label_action": _mapping(trace.get("support_join")).get(
                            "label_action"
                        ),
                        "branch_reason": _mapping(
                            trace.get("branch_evidence_join")
                        ).get("branch_reason"),
                        "autopsy_tags": trace.get("autopsy_tags"),
                    }
                    for trace in seed_traces
                ],
            }
        )
    return rows


def _aggregates(
    *,
    traces: Sequence[Mapping[str, object]],
    case_reports: Sequence[Mapping[str, object]],
    expected_override_count: int,
) -> dict[str, object]:
    birth_traces = [
        trace
        for trace in traces
        if _mapping(trace.get("regression_association")).get(
            "birth_regression_seed"
        )
        is True
    ]
    resolved_traces = [
        trace
        for trace in traces
        if _mapping(trace.get("regression_association")).get(
            "resolved_invalid_increase_seed"
        )
        is True
    ]
    direct_invalid = [
        trace
        for trace in traces
        if _mapping(trace.get("regression_association")).get(
            "direct_invalid_override"
        )
        is True
    ]
    return {
        "expected_live_carrion_override_count": int(expected_override_count),
        "observed_live_carrion_override_count": len(traces),
        "override_count_matches_train_eval": len(traces) == int(expected_override_count),
        "case_count": len(case_reports),
        "live_override_action_counts": _counter_dict(
            Counter(str(trace.get("selected_support_action")) for trace in traces)
        ),
        "live_override_support_branch_reason_counts": _counter_dict(
            Counter(
                str(_mapping(trace.get("branch_evidence_join")).get("branch_reason"))
                for trace in traces
            )
        ),
        "zero_distance_override_count": sum(
            1
            for trace in traces
            if _float(trace.get("nearest_support_distance"), default=1.0) == 0.0
        ),
        "movement_override_count": sum(
            1
            for trace in traces
            if str(trace.get("selected_support_action", "")).startswith("move_")
        ),
        "birth_regression_override_action_counts": _counter_dict(
            Counter(str(trace.get("selected_support_action")) for trace in birth_traces)
        ),
        "resolved_invalid_increase_override_action_counts": _counter_dict(
            Counter(str(trace.get("selected_support_action")) for trace in resolved_traces)
        ),
        "direct_invalid_override_count": len(direct_invalid),
        "all_live_overrides_directly_legal": not direct_invalid
        and all(
            _mapping(trace.get("action_mask_legality")).get(
                "selected_support_action_legal"
            )
            is True
            for trace in traces
        ),
    }


def _recommended_next_route(classification: Mapping[str, object]) -> str:
    labels = set(_list(classification.get("labels")))
    if "missing_hydration_reproduction_context" in labels:
        return "build_carrion_hydration_reproduction_sequence_context_archive"
    if "resolution_conflict" in labels:
        return "add_resolution_context_before_any_runtime_override"
    if "movement_near_carrion_side_effects" in labels:
        return "separate_movement_near_carrion_archive_labels"
    if "stale_one_step_aliasing" in labels:
        return "add_public_sequence_context_to_carrion_support"
    return "stop_carrion_archive_support_gated_override_family"


def _train_eval_failure_summary(report: Mapping[str, object]) -> dict[str, object]:
    acceptance = _mapping(report.get("acceptance"))
    metrics = _mapping(acceptance.get("metrics"))
    return {
        "classification": _mapping(report.get("classification")).get("primary"),
        "acceptance_passed": acceptance.get("passed") is True,
        "blockers": _list_of_mappings(acceptance.get("blockers")),
        "carrion_applied_override_count": metrics.get("carrion_applied_override_count"),
        "carrion_per_seed_delta": metrics.get("carrion_per_seed_delta"),
        "dominant_requested_action_share": metrics.get(
            "dominant_requested_action_share"
        ),
        "heuristic_action_source_count": metrics.get("heuristic_action_source_count"),
    }


def _dataset_by_index(
    dataset_rows: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, object]]:
    return {
        _int(_mapping(row.get("metadata")).get("row_index"), default=index): dict(row)
        for index, row in enumerate(dataset_rows)
    }


def _branch_result_by_id(
    archive_report: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    result = {}
    for branch in _list_of_mappings(archive_report.get("branch_results")):
        branch_id = str(branch.get("branch_id", ""))
        if branch_id:
            result[branch_id] = branch
    return result


def _carrion_seed_delta(
    train_eval_report: Mapping[str, object],
) -> dict[int, dict[str, object]]:
    metrics = _mapping(_mapping(train_eval_report.get("acceptance")).get("metrics"))
    return {
        _int(row.get("seed")): dict(row)
        for row in _list_of_mappings(metrics.get("carrion_per_seed_delta"))
    }


def _branch_action_run(
    branch_result: Mapping[str, object],
    action: str,
) -> Mapping[str, object]:
    for run in _list_of_mappings(branch_result.get("action_runs")):
        if str(run.get("forced_action")) == action:
            return run
    return {}


def _source_mask_legal(branch_result: Mapping[str, object], action: str) -> bool:
    public_features = _mapping(branch_result.get("public_features"))
    action_mask = _mapping(public_features.get("action_mask"))
    if action_mask:
        return action_mask.get(action) is True
    return action in set(str(item) for item in _list(branch_result.get("candidate_actions")))


def _source_support_key(trace: Mapping[str, object]) -> str:
    support = _mapping(trace.get("support_join"))
    return (
        f"{support.get('source_fixture')}:{support.get('source_seed')}:"
        f"{support.get('label_action')}:{support.get('source_branch_id')}"
    )


def _input_paths(
    input_paths: Mapping[str, str | Path] | None,
) -> dict[str, str | None]:
    if input_paths is None:
        return {
            "archive_report": None,
            "archive_dataset": None,
            "train_eval_report": None,
            "artifact": None,
        }
    return {
        "archive_report": _optional_path(input_paths.get("archive_report")),
        "archive_dataset": _optional_path(input_paths.get("archive_dataset")),
        "train_eval_report": _optional_path(input_paths.get("train_eval_report")),
        "artifact": _optional_path(input_paths.get("artifact")),
    }


def _optional_path(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _list(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return []
    return result


def _flatten(value: object, *, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, object]] = []
        for key, nested in value.items():
            path = str(key) if not prefix else f"{prefix}.{key}"
            rows.extend(_flatten(nested, prefix=path))
        return rows
    if isinstance(value, list):
        rows = []
        for index, nested in enumerate(value):
            path = f"{prefix}[{index}]"
            rows.extend(_flatten(nested, prefix=path))
        return rows
    return [(prefix, value)]


def _looks_like_digest(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


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
