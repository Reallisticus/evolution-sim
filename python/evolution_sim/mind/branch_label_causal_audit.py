from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.cli import mind_v3_evaluate as evaluate_cli
from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.mind.branch_intervention_residual import (
    DEFAULT_REPORT_PATH as DEFAULT_V144_REPORT_PATH,
    DEFAULT_V143_DATASET_PATH,
    DEFAULT_V143_REPORT_PATH,
    MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_REPORT_SCHEMA_VERSION,
    STRICT_BROAD_SEEDS,
    STRICT_TICKS,
    load_json_report,
    load_v143_branch_intervention_dataset,
    write_json,
)
from evolution_sim.mind.broad_regression_branch_intervention import (
    MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION,
    MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.support_gated_residual import (
    MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
    MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
    load_support_gated_residual_artifact,
    support_gated_residual_runtime_diagnostics,
    validate_support_gated_residual_artifact,
)
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

MIND_V3_V145_BRANCH_LABEL_CAUSAL_AUDIT_SCHEMA_VERSION = (
    "mind_v3_v145_branch_label_causal_audit_v1"
)
MIND_V3_V145_BRANCH_LABEL_CAUSAL_AUDIT_POLICY = (
    "diagnostics_only_v145_branch_label_causal_audit_v1"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v145-branch-label-causal-audit.json"
)
LABEL_SUPPORTED_AND_POPULATION_SAFE = "label_supported_and_population_safe"
LABEL_HORIZON_FRAGILE = "label_horizon_fragile"
LABEL_AGENT_LOCAL_POSITIVE_POPULATION_NEGATIVE = (
    "label_agent_local_positive_population_negative"
)
LABEL_RESOLUTION_INVALID_RISK = "label_resolution_invalid_risk"
LABEL_INSUFFICIENT_PUBLIC_STATE = "label_insufficient_public_state"
ROUTE_LABEL_BLACKLIST = "recommend_label_blacklist_for_later_archive_rebuild"
ROUTE_CLOSE_EXACT_MATCH = "close_exact_match_residual_family"
ROUTE_V146_ARCHIVE_EXPANSION = "recommend_v146_archive_expansion_with_more_branch_points_per_seed"
ROUTE_SOURCE_INTEGRITY_FAILED = "source_integrity_failed"
MAX_STORED_INVALID_EXAMPLES = 16


class BranchLabelCausalAuditError(ValueError):
    pass


def load_v144_artifact_for_audit(
    *,
    v144_report: Mapping[str, object],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, object], Path]:
    path = _artifact_path_from_report(v144_report, artifact_path=artifact_path)
    artifact = load_support_gated_residual_artifact(path)
    expected_digest = v144_report.get("artifact_digest")
    observed_digest = stable_payload_digest(artifact)
    if isinstance(expected_digest, str) and expected_digest != observed_digest:
        raise BranchLabelCausalAuditError(
            "v144 artifact digest mismatch: "
            f"expected {expected_digest}, observed {observed_digest}"
        )
    return artifact, path


def build_branch_label_causal_audit_report(
    *,
    v143_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    v144_report: Mapping[str, object],
    artifact: Mapping[str, object],
    v143_report_path: str | Path | None = DEFAULT_V143_REPORT_PATH,
    v143_dataset_path: str | Path | None = DEFAULT_V143_DATASET_PATH,
    v144_report_path: str | Path | None = DEFAULT_V144_REPORT_PATH,
    artifact_path: str | Path | None = None,
    trace_runs: Mapping[int, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    validate_support_gated_residual_artifact(artifact)
    source_integrity = _source_integrity(
        v143_report=v143_report,
        dataset_rows=dataset_rows,
        v144_report=v144_report,
        artifact=artifact,
        v143_report_path=v143_report_path,
        v143_dataset_path=v143_dataset_path,
        v144_report_path=v144_report_path,
        artifact_path=artifact_path,
    )
    applied_seed_rows = _v144_applied_seed_rows(v144_report)
    seed_trace_runs = _trace_runs_for_applied_seeds(
        applied_seed_rows=applied_seed_rows,
        artifact=artifact,
        trace_runs=trace_runs,
    )
    applied_labels = _applied_label_audits(
        v143_report=v143_report,
        dataset_rows=dataset_rows,
        v144_report=v144_report,
        artifact=artifact,
        trace_runs=seed_trace_runs,
    )
    broad_explanation = _broad_resolved_invalid_explanation(
        v144_report=v144_report,
        applied_labels=applied_labels,
    )
    seed_5_explanation = _seed_5_birth_regression_explanation(applied_labels)
    route = _route_decision(applied_labels, source_integrity=source_integrity)
    primary = _classification_for_route(route["route"])
    return {
        "schema_version": MIND_V3_V145_BRANCH_LABEL_CAUSAL_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_V145_BRANCH_LABEL_CAUSAL_AUDIT_POLICY,
        "contract": {
            "audit_only": True,
            "runtime_policy_added": False,
            "runtime_action_selection_changed": False,
            "training_executed": False,
            "threshold_tuning": False,
            "runtime_promotion_authorized": False,
            "training_promotion_authorized": False,
            "default_runtime_behavior_changed": False,
            "replay_golden_effect": "none",
            "viewer_effect": "none",
        },
        "inputs": {
            "v143_report": str(v143_report_path) if v143_report_path else None,
            "v143_dataset": str(v143_dataset_path) if v143_dataset_path else None,
            "v144_report": str(v144_report_path) if v144_report_path else None,
            "v144_artifact": str(artifact_path) if artifact_path else None,
        },
        "source_integrity": source_integrity,
        "v144_applied_seed_rows": applied_seed_rows,
        "applied_label_count": len(applied_labels),
        "applied_labels": applied_labels,
        "seed_5_birth_regression_explanation": seed_5_explanation,
        "broad_resolved_invalid_increase_explanation": broad_explanation,
        "route_decision": route,
        "classification": {"primary": primary, "labels": [primary, route["route"]]},
        "non_default_runtime": False,
        "non_promoted": True,
    }


def write_branch_label_causal_audit_report(
    report: Mapping[str, object],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> None:
    write_json(output_path, report)


def _source_integrity(
    *,
    v143_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    v144_report: Mapping[str, object],
    artifact: Mapping[str, object],
    v143_report_path: str | Path | None,
    v143_dataset_path: str | Path | None,
    v144_report_path: str | Path | None,
    artifact_path: str | Path | None,
) -> dict[str, object]:
    failures: list[str] = []
    if v143_report.get("schema_version") != MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION:
        failures.append("v143_report_schema_mismatch")
    if v144_report.get("schema_version") != (
        MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_REPORT_SCHEMA_VERSION
    ):
        failures.append("v144_report_schema_mismatch")
    if artifact.get("schema_version") != (
        MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION
    ):
        failures.append("v144_artifact_schema_mismatch")
    if artifact.get("policy") != MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY:
        failures.append("v144_artifact_policy_mismatch")
    if _mapping(v143_report.get("classification")).get("primary") != (
        "broad_regression_branch_intervention_supported_for_v144_training"
    ):
        failures.append("v143_classification_not_supported")
    if _mapping(v144_report.get("classification")).get("primary") != (
        "branch_intervention_residual_blocked_non_promotable"
    ):
        failures.append("v144_not_blocked_non_promotable")
    acceptance = _mapping(v144_report.get("acceptance"))
    if acceptance.get("runtime_promotion_allowed") is not False:
        failures.append("v144_runtime_promotion_not_false")
    if acceptance.get("training_promotion_allowed") is not False:
        failures.append("v144_training_promotion_not_false")
    dataset_digest = stable_payload_digest(list(dataset_rows))
    report_dataset = _mapping(v143_report.get("dataset"))
    if report_dataset.get("dataset_digest") != dataset_digest:
        failures.append("v143_dataset_digest_mismatch")
    if _int(report_dataset.get("row_count")) != len(dataset_rows):
        failures.append("v143_dataset_row_count_mismatch")
    malformed_rows = [
        index
        for index, row in enumerate(dataset_rows)
        if row.get("schema_version")
        != MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION
    ]
    if malformed_rows:
        failures.append("v143_dataset_row_schema_mismatch")
    expected_artifact_digest = v144_report.get("artifact_digest")
    observed_artifact_digest = stable_payload_digest(artifact)
    if (
        isinstance(expected_artifact_digest, str)
        and expected_artifact_digest != observed_artifact_digest
    ):
        failures.append("v144_artifact_digest_mismatch")
    return {
        "policy": "v145_source_integrity_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v143_report_path": str(v143_report_path) if v143_report_path else None,
        "v143_dataset_path": str(v143_dataset_path) if v143_dataset_path else None,
        "v144_report_path": str(v144_report_path) if v144_report_path else None,
        "artifact_path": str(artifact_path) if artifact_path else None,
        "dataset_row_count": len(dataset_rows),
        "v143_dataset_digest": dataset_digest,
        "v143_report_dataset_digest": report_dataset.get("dataset_digest"),
        "v144_artifact_digest": observed_artifact_digest,
        "v144_report_artifact_digest": expected_artifact_digest,
        "malformed_row_indices": malformed_rows[:16],
    }


def _trace_runs_for_applied_seeds(
    *,
    applied_seed_rows: Sequence[Mapping[str, object]],
    artifact: Mapping[str, object],
    trace_runs: Mapping[int, Mapping[str, object]] | None,
) -> dict[int, Mapping[str, object]]:
    if trace_runs is not None:
        return {int(seed): payload for seed, payload in trace_runs.items()}
    traced: dict[int, Mapping[str, object]] = {}
    for row in applied_seed_rows:
        seed = _int(row.get("seed"))
        if _int(row.get("applied_override_count")) <= 0:
            continue
        traced[seed] = _trace_broad_seed(seed=seed, ticks=STRICT_TICKS, artifact=artifact)
    return traced


def _trace_broad_seed(
    *,
    seed: int,
    ticks: int,
    artifact: Mapping[str, object],
) -> dict[str, object]:
    linear_world = SimulationWorld(
        WorldConfig(seed=int(seed), max_ticks=int(ticks)),
        policy=MindV3EvolutionPolicy(seed=int(seed)),
    )
    linear_run = evaluate_cli._run_world(
        world=linear_world,
        seed=int(seed),
        ticks=int(ticks),
    )
    residual_world = SimulationWorld(
        WorldConfig(seed=int(seed), max_ticks=int(ticks)),
        policy=MindV3EvolutionPolicy(
            seed=int(seed),
            support_residual_artifact=artifact,
            support_residual_runtime_mode="live",
        ),
    )
    residual_run = evaluate_cli._run_world(
        world=residual_world,
        seed=int(seed),
        ticks=int(ticks),
    )
    residual_run["support_residual_diagnostics"] = (
        support_gated_residual_runtime_diagnostics(
            residual_world.policy_decision_diagnostics_records
        )
    )
    return {
        "seed": int(seed),
        "linear": {
            "run": linear_run,
            "records": [dict(record) for record in linear_world.trajectory_records],
            "diagnostics": [
                dict(item) if isinstance(item, Mapping) else None
                for item in linear_world.policy_decision_diagnostics_records
            ],
        },
        "residual": {
            "run": residual_run,
            "records": [dict(record) for record in residual_world.trajectory_records],
            "diagnostics": [
                dict(item) if isinstance(item, Mapping) else None
                for item in residual_world.policy_decision_diagnostics_records
            ],
        },
    }


def _applied_label_audits(
    *,
    v143_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    v144_report: Mapping[str, object],
    artifact: Mapping[str, object],
    trace_runs: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    labels: list[dict[str, object]] = []
    for seed in sorted(trace_runs):
        trace = _mapping(trace_runs[seed])
        residual = _mapping(trace.get("residual"))
        linear = _mapping(trace.get("linear"))
        residual_records = _list_of_mappings(residual.get("records"))
        residual_diagnostics = _list(residual.get("diagnostics"))
        linear_records = _list_of_mappings(linear.get("records"))
        for index, record in enumerate(residual_records):
            diagnostic = (
                residual_diagnostics[index]
                if index < len(residual_diagnostics)
                else None
            )
            if not isinstance(diagnostic, Mapping):
                continue
            if diagnostic.get("support_residual_override_applied") is not True:
                continue
            event = _override_event(
                seed=seed,
                event_index=index,
                record=record,
                diagnostic=diagnostic,
            )
            support = _match_support_example(
                artifact=artifact,
                dataset_rows=dataset_rows,
                record=record,
                residual_action=str(event["residual_action"]),
            )
            downstream = _downstream_invalid_attribution(
                override_event=event,
                residual_records=residual_records,
                linear_records=linear_records,
            )
            v143_evidence = _v143_label_evidence(
                source_match=support,
                dataset_rows=dataset_rows,
                v143_report=v143_report,
            )
            v144_outcome = _v144_seed_outcome(
                v144_report=v144_report,
                seed=seed,
                residual_action=str(event["residual_action"]),
            )
            label_classification = _classify_applied_label(
                source_match=support,
                v143_evidence=v143_evidence,
                v144_outcome=v144_outcome,
                downstream=downstream,
            )
            labels.append(
                {
                    **event,
                    "support_example": support,
                    "label_source_row": _label_source_row_excerpt(
                        dataset_rows=dataset_rows,
                        source_match=support,
                    ),
                    "v143_branch_local_evidence": v143_evidence,
                    "v144_full_run_outcome": v144_outcome,
                    "downstream_invalid_resolution_attribution": downstream,
                    "label_classification": label_classification,
                }
            )
    return labels


def _override_event(
    *,
    seed: int,
    event_index: int,
    record: Mapping[str, object],
    diagnostic: Mapping[str, object],
) -> dict[str, object]:
    return {
        "seed": int(seed),
        "tick": _int(record.get("tick")),
        "agent_id": _int(record.get("agent_id")),
        "event_index": int(event_index),
        "linear_action": str(diagnostic.get("support_residual_linear_action", "")),
        "residual_action": str(diagnostic.get("support_residual_proposed_action", "")),
        "final_action": str(diagnostic.get("support_residual_final_action", "")),
        "resolved_action": str(record.get("resolved_action", "")),
        "nearest_support_distance": diagnostic.get(
            "support_residual_nearest_support_distance"
        ),
        "score_margin": diagnostic.get("support_residual_score_margin"),
        "support_gate_passed": bool(
            diagnostic.get("support_residual_support_gate_passed")
        ),
        "record_action_valid": bool(record.get("action_valid")),
        "record_resolution_action_valid": bool(record.get("resolution_action_valid")),
        "before": _position_and_vitals(record.get("before")),
        "after": _position_and_vitals(record.get("after")),
    }


def _match_support_example(
    *,
    artifact: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    record: Mapping[str, object],
    residual_action: str,
) -> dict[str, object]:
    runtime_row = planner_distilled_runtime_row(
        observation_input=_mapping(record.get("observation_input")),
        action_mask=_mapping(record.get("action_mask")),
        public_history_trace=[],
    )
    features = candidate_feature_vector(runtime_row, residual_action)
    support = _list_of_mappings(artifact.get("support_examples"))
    matches = []
    for support_index, example in enumerate(support):
        if str(example.get("action", "")) != residual_action:
            continue
        vector = _float_list(example.get("feature_vector"))
        distance = _squared_distance(features, vector)
        matches.append(
            {
                "support_index": support_index,
                "example_index": _int(example.get("example_index")),
                "action": str(example.get("action", "")),
                "mode": example.get("mode"),
                "distance": _round(distance),
                "support_weight": _float(example.get("weight")),
            }
        )
    ordered = sorted(matches, key=lambda item: (float(item["distance"]), item["support_index"]))
    best = ordered[0] if ordered else None
    exact_matches = [
        item for item in ordered if abs(float(item["distance"])) <= 1e-12
    ]
    row_index = _int(best.get("example_index")) if best is not None else -1
    row = dataset_rows[row_index] if 0 <= row_index < len(dataset_rows) else {}
    metadata = _mapping(_mapping(row).get("metadata"))
    return {
        "policy": "v145_exact_support_example_match_v1",
        "matched": bool(best is not None and abs(float(best["distance"])) <= 1e-12),
        "ambiguous_exact_match_count": len(exact_matches),
        "nearest": best,
        "candidate_match_count": len(matches),
        "label_source_row_index": row_index if 0 <= row_index < len(dataset_rows) else None,
        "label_source_seed": metadata.get("seed"),
        "label_source_branch_id": metadata.get("branch_id"),
        "label_source_agent_id": metadata.get("agent_id"),
        "label_source_branch_tick": metadata.get("branch_tick"),
    }


def _downstream_invalid_attribution(
    *,
    override_event: Mapping[str, object],
    residual_records: Sequence[Mapping[str, object]],
    linear_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    override_index = _int(override_event.get("event_index"))
    linear_index = _matching_record_index(
        linear_records,
        tick=_int(override_event.get("tick")),
        agent_id=_int(override_event.get("agent_id")),
    )
    residual_invalid = [
        (index, record)
        for index, record in enumerate(residual_records)
        if index > override_index and record.get("resolution_action_valid") is False
    ]
    linear_invalid = [
        (index, record)
        for index, record in enumerate(linear_records)
        if index > linear_index and record.get("resolution_action_valid") is False
    ]
    category_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    examples = []
    for index, record in residual_invalid:
        attribution = _invalid_record_attribution(
            record=record,
            override_event=override_event,
        )
        for key, value in attribution["categories"].items():
            if value is True:
                category_counts.update([key])
        primary_counts.update([str(attribution["primary_attribution"])])
        if len(examples) < MAX_STORED_INVALID_EXAMPLES:
            examples.append(
                {
                    "record_index": index,
                    "tick": _int(record.get("tick")),
                    "agent_id": _int(record.get("agent_id")),
                    "requested_action": record.get("requested_action"),
                    "resolved_action": record.get("resolved_action"),
                    "invalid_reason": _mapping(record.get("outcome")).get(
                        "invalid_reason"
                    ),
                    "before": _position_and_vitals(record.get("before")),
                    "attribution": attribution,
                }
            )
    return {
        "policy": "v145_downstream_invalid_resolution_attribution_v1",
        "override_event_index": override_index,
        "linear_matching_event_index": linear_index,
        "residual_invalid_after_override_count": len(residual_invalid),
        "linear_invalid_after_matching_decision_count": len(linear_invalid),
        "invalid_after_override_delta_vs_linear": len(residual_invalid)
        - len(linear_invalid),
        "category_counts": dict(sorted(category_counts.items())),
        "primary_attribution_counts": dict(sorted(primary_counts.items())),
        "stored_invalid_example_count": len(examples),
        "invalid_examples": examples,
    }


def _invalid_record_attribution(
    *,
    record: Mapping[str, object],
    override_event: Mapping[str, object],
) -> dict[str, object]:
    action = str(record.get("requested_action", ""))
    action_mask = _mapping(record.get("action_mask"))
    resolution_mask = _mapping(record.get("resolution_action_mask"))
    same_agent = _int(record.get("agent_id")) == _int(override_event.get("agent_id"))
    tick_delta = _int(record.get("tick")) - _int(override_event.get("tick"))
    distance = _position_distance(
        _mapping(record.get("before")),
        _mapping(override_event.get("after")) or _mapping(override_event.get("before")),
    )
    local_contention = tick_delta == 0 and distance is not None and distance <= 1
    occupancy_conflict = (
        action.startswith("move_") or action.startswith("attack_")
    ) and bool(action_mask.get(action)) and not bool(resolution_mask.get(action))
    depleted_resources = action in {"eat", "drink"} and bool(
        action_mask.get(action)
    ) and not bool(resolution_mask.get(action))
    unrelated = not (
        same_agent or local_contention or occupancy_conflict or depleted_resources
    )
    categories = {
        "same_agent": same_agent,
        "local_contention": local_contention,
        "depleted_resources": depleted_resources,
        "occupancy_conflicts": occupancy_conflict,
        "unrelated_population_dynamics": unrelated,
    }
    if same_agent:
        primary = "same_agent"
    elif local_contention and occupancy_conflict:
        primary = "local_contention_occupancy_conflict"
    elif occupancy_conflict:
        primary = "occupancy_conflict"
    elif depleted_resources:
        primary = "depleted_resources"
    elif local_contention:
        primary = "local_contention"
    else:
        primary = "unrelated_population_dynamics"
    return {
        "primary_attribution": primary,
        "categories": categories,
        "tick_delta_from_override": tick_delta,
        "manhattan_distance_from_override_after": distance,
    }


def _v143_label_evidence(
    *,
    source_match: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    v143_report: Mapping[str, object],
) -> dict[str, object]:
    row_index = source_match.get("label_source_row_index")
    row = dataset_rows[_int(row_index)] if isinstance(row_index, int) else {}
    metadata = _mapping(_mapping(row).get("metadata"))
    outcome = _mapping(metadata.get("outcome_evidence"))
    baseline_delta = _mapping(outcome.get("deltas_vs_baseline"))
    override_delta = _mapping(outcome.get("deltas_vs_v142_override"))
    label = _mapping(_mapping(_mapping(row).get("trainable")).get("label"))
    return {
        "policy": "v145_v143_branch_local_label_evidence_v1",
        "row_index": row_index,
        "branch_id": metadata.get("branch_id"),
        "seed": metadata.get("seed"),
        "branch_tick": metadata.get("branch_tick"),
        "agent_id": metadata.get("agent_id"),
        "label_action": label.get("action"),
        "deltas_vs_baseline": baseline_delta,
        "deltas_vs_v142_override": override_delta,
        "supported_by_v143_report": bool(source_match.get("matched")),
        "v143_classification": _mapping(v143_report.get("classification")).get(
            "primary"
        ),
    }


def _v144_seed_outcome(
    *,
    v144_report: Mapping[str, object],
    seed: int,
    residual_action: str,
) -> dict[str, object]:
    per_seed = _v144_per_seed_delta(v144_report, seed=seed)
    return {
        "policy": "v145_v144_full_run_seed_action_outcome_v1",
        "seed": int(seed),
        "residual_action": residual_action,
        "alive_delta": _int(per_seed.get("alive_delta")),
        "births_delta": _int(per_seed.get("births_delta")),
        "resolved_invalid_action_count_delta": _int(
            per_seed.get("resolved_invalid_action_count_delta")
        ),
        "unsupported_requested_action_count_delta": _int(
            per_seed.get("unsupported_requested_action_count_delta")
        ),
        "unsupported_proposed_action_count": _int(
            per_seed.get("unsupported_proposed_action_count")
        ),
        "applied_override_count": _int(per_seed.get("applied_override_count")),
        "applied_override_action_counts": dict(
            sorted(_mapping(per_seed.get("applied_override_action_counts")).items())
        ),
    }


def _classify_applied_label(
    *,
    source_match: Mapping[str, object],
    v143_evidence: Mapping[str, object],
    v144_outcome: Mapping[str, object],
    downstream: Mapping[str, object],
) -> str:
    if source_match.get("matched") is not True:
        return LABEL_INSUFFICIENT_PUBLIC_STATE
    if _int(source_match.get("ambiguous_exact_match_count")) > 1:
        return LABEL_INSUFFICIENT_PUBLIC_STATE
    resolved_delta = _int(v144_outcome.get("resolved_invalid_action_count_delta"))
    invalid_after_delta = _int(downstream.get("invalid_after_override_delta_vs_linear"))
    if resolved_delta > 0 or invalid_after_delta > 0:
        return LABEL_RESOLUTION_INVALID_RISK
    branch_baseline = _mapping(v143_evidence.get("deltas_vs_baseline"))
    branch_v142 = _mapping(v143_evidence.get("deltas_vs_v142_override"))
    branch_local_positive = (
        _int(branch_baseline.get("alive_agents")) > 0
        or _int(branch_baseline.get("births")) > 0
        or _int(branch_v142.get("target_alive")) > 0
    )
    population_negative = (
        _int(v144_outcome.get("alive_delta")) < 0
        or _int(v144_outcome.get("births_delta")) < 0
    )
    if branch_local_positive and population_negative:
        return LABEL_AGENT_LOCAL_POSITIVE_POPULATION_NEGATIVE
    if branch_local_positive and not population_negative:
        return LABEL_SUPPORTED_AND_POPULATION_SAFE
    return LABEL_HORIZON_FRAGILE


def _route_decision(
    applied_labels: Sequence[Mapping[str, object]],
    *,
    source_integrity: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True:
        return {
            "policy": "v145_route_decision_v1",
            "route": ROUTE_SOURCE_INTEGRITY_FAILED,
            "recommendation": "Fix source integrity before interpreting label causality.",
            "label_blacklist": [],
        }
    bad = [
        label
        for label in applied_labels
        if label.get("label_classification")
        in {
            LABEL_RESOLUTION_INVALID_RISK,
            LABEL_AGENT_LOCAL_POSITIVE_POPULATION_NEGATIVE,
        }
    ]
    if bad:
        blacklist = [_blacklist_entry(label) for label in bad]
        return {
            "policy": "v145_route_decision_v1",
            "route": ROUTE_LABEL_BLACKLIST,
            "recommendation": (
                "At least one applied v143 label is causally unsafe in the "
                "v144 full run; rebuild a later archive with these label rows "
                "excluded rather than tuning the residual thresholds."
            ),
            "label_blacklist": blacklist,
        }
    fragile = [
        label
        for label in applied_labels
        if label.get("label_classification")
        in {LABEL_HORIZON_FRAGILE, LABEL_INSUFFICIENT_PUBLIC_STATE}
    ]
    if applied_labels and len(fragile) == len(applied_labels):
        return {
            "policy": "v145_route_decision_v1",
            "route": ROUTE_CLOSE_EXACT_MATCH,
            "recommendation": (
                "All applied labels are fragile because the public state or "
                "branch horizon is insufficient; close this exact-match "
                "residual family."
            ),
            "label_blacklist": [],
        }
    return {
        "policy": "v145_route_decision_v1",
        "route": ROUTE_V146_ARCHIVE_EXPANSION,
        "recommendation": (
            "Applied labels are population-safe but too sparse; expand the v146 "
            "archive with more branch points per seed before another runtime test."
        ),
        "label_blacklist": [],
    }


def _blacklist_entry(label: Mapping[str, object]) -> dict[str, object]:
    source = _mapping(label.get("label_source_row"))
    return {
        "seed": label.get("seed"),
        "tick": label.get("tick"),
        "agent_id": label.get("agent_id"),
        "linear_action": label.get("linear_action"),
        "residual_action": label.get("residual_action"),
        "label_source_row_index": source.get("row_index"),
        "branch_id": source.get("branch_id"),
        "classification": label.get("label_classification"),
        "reason": "exclude_causally_bad_branch_label_from_later_archive_rebuild",
    }


def _broad_resolved_invalid_explanation(
    *,
    v144_report: Mapping[str, object],
    applied_labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    broad = _mapping(v144_report.get("live_strict_broad"))
    linear = _mapping(_mapping(broad.get("linear")).get("aggregate"))
    residual = _mapping(_mapping(broad.get("residual")).get("aggregate"))
    seed_rows = []
    for row in _list_of_mappings(broad.get("per_seed_delta")):
        delta = _int(row.get("resolved_invalid_action_count_delta"))
        applied = _int(row.get("applied_override_count"))
        if delta or applied:
            seed_rows.append(
                {
                    "seed": row.get("seed"),
                    "resolved_invalid_action_count_delta": delta,
                    "applied_override_count": applied,
                    "applied_override_action_counts": row.get(
                        "applied_override_action_counts"
                    ),
                    "alive_delta": row.get("alive_delta"),
                    "births_delta": row.get("births_delta"),
                }
            )
    attributions = Counter()
    for label in applied_labels:
        downstream = _mapping(label.get("downstream_invalid_resolution_attribution"))
        attributions.update(
            _mapping(downstream.get("primary_attribution_counts"))
        )
    delta = _int(residual.get("resolved_invalid_action_count")) - _int(
        linear.get("resolved_invalid_action_count")
    )
    return {
        "policy": "v145_broad_resolved_invalid_increase_explanation_v1",
        "linear_resolved_invalid_action_count": linear.get(
            "resolved_invalid_action_count"
        ),
        "residual_resolved_invalid_action_count": residual.get(
            "resolved_invalid_action_count"
        ),
        "delta": delta,
        "seed_deltas_with_applied_or_invalid_change": seed_rows,
        "downstream_primary_attribution_counts": dict(sorted(attributions.items())),
        "explanation": (
            "The broad matrix blocked because the live residual raised resolved-invalid "
            f"actions by {delta}. The increase is localized to seeds with applied "
            "overrides in the v144 report, so v145 treats those labels as causal "
            "audit targets rather than tuning thresholds."
        ),
    }


def _seed_5_birth_regression_explanation(
    applied_labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    seed_5 = next((label for label in applied_labels if _int(label.get("seed")) == 5), None)
    if seed_5 is None:
        return {
            "policy": "v145_seed_5_birth_regression_explanation_v1",
            "found": False,
            "explanation": "No applied v144 override was found for seed 5.",
        }
    evidence = _mapping(seed_5.get("v143_branch_local_evidence"))
    baseline = _mapping(evidence.get("deltas_vs_baseline"))
    outcome = _mapping(seed_5.get("v144_full_run_outcome"))
    downstream = _mapping(seed_5.get("downstream_invalid_resolution_attribution"))
    return {
        "policy": "v145_seed_5_birth_regression_explanation_v1",
        "found": True,
        "seed": 5,
        "tick": seed_5.get("tick"),
        "agent_id": seed_5.get("agent_id"),
        "linear_action": seed_5.get("linear_action"),
        "residual_action": seed_5.get("residual_action"),
        "v143_branch_births_delta_vs_baseline": baseline.get("births"),
        "v143_branch_alive_delta_vs_baseline": baseline.get("alive_agents"),
        "v144_full_run_births_delta": outcome.get("births_delta"),
        "v144_full_run_alive_delta": outcome.get("alive_delta"),
        "v144_full_run_resolved_invalid_delta": outcome.get(
            "resolved_invalid_action_count_delta"
        ),
        "invalid_after_override_delta_vs_linear": downstream.get(
            "invalid_after_override_delta_vs_linear"
        ),
        "label_classification": seed_5.get("label_classification"),
        "explanation": (
            "Seed 5 applied the v143 branch label in the full v144 run, but the "
            "branch-local birth gain did not transfer to the full population "
            "trajectory. The full run gained terminal alive agents while losing "
            "one birth and adding resolved-invalid pressure, so the label is "
            "not population-safe despite its branch-local evidence."
        ),
    }


def _label_source_row_excerpt(
    *,
    dataset_rows: Sequence[Mapping[str, object]],
    source_match: Mapping[str, object],
) -> dict[str, object]:
    row_index = source_match.get("label_source_row_index")
    if not isinstance(row_index, int) or not (0 <= row_index < len(dataset_rows)):
        return {"row_index": None, "found": False}
    row = dataset_rows[row_index]
    metadata = _mapping(row.get("metadata"))
    trainable = _mapping(row.get("trainable"))
    label = _mapping(trainable.get("label"))
    return {
        "found": True,
        "row_index": row_index,
        "seed": metadata.get("seed"),
        "fixture": metadata.get("fixture"),
        "branch_id": metadata.get("branch_id"),
        "branch_tick": metadata.get("branch_tick"),
        "agent_id": metadata.get("agent_id"),
        "label_action": label.get("action"),
        "label_policy": label.get("label_policy"),
        "branch_state_digest": metadata.get("branch_state_digest"),
        "replay_digest": metadata.get("replay_digest"),
    }


def _v144_applied_seed_rows(
    v144_report: Mapping[str, object],
) -> list[dict[str, object]]:
    broad = _mapping(v144_report.get("live_strict_broad"))
    rows = []
    for row in _list_of_mappings(broad.get("per_seed_delta")):
        rows.append(
            {
                "seed": _int(row.get("seed")),
                "applied_override_count": _int(row.get("applied_override_count")),
                "applied_override_action_counts": dict(
                    sorted(
                        _mapping(row.get("applied_override_action_counts")).items()
                    )
                ),
                "alive_delta": _int(row.get("alive_delta")),
                "births_delta": _int(row.get("births_delta")),
                "resolved_invalid_action_count_delta": _int(
                    row.get("resolved_invalid_action_count_delta")
                ),
            }
        )
    return rows


def _v144_per_seed_delta(
    v144_report: Mapping[str, object],
    *,
    seed: int,
) -> Mapping[str, object]:
    broad = _mapping(v144_report.get("live_strict_broad"))
    for row in _list_of_mappings(broad.get("per_seed_delta")):
        if _int(row.get("seed")) == int(seed):
            return row
    return {}


def _matching_record_index(
    records: Sequence[Mapping[str, object]],
    *,
    tick: int,
    agent_id: int,
) -> int:
    for index, record in enumerate(records):
        if _int(record.get("tick")) == int(tick) and _int(record.get("agent_id")) == int(agent_id):
            return index
    return -1


def _artifact_path_from_report(
    v144_report: Mapping[str, object],
    *,
    artifact_path: str | Path | None,
) -> Path:
    if artifact_path is not None:
        return Path(artifact_path)
    raw = v144_report.get("artifact_output")
    if not isinstance(raw, str) or not raw:
        raise BranchLabelCausalAuditError(
            "v144 report does not include artifact_output; pass --artifact"
        )
    return Path(raw)


def _classification_for_route(route: str) -> str:
    if route == ROUTE_LABEL_BLACKLIST:
        return "branch_label_causal_audit_recommends_label_blacklist"
    if route == ROUTE_CLOSE_EXACT_MATCH:
        return "branch_label_causal_audit_closes_exact_match_residual_family"
    if route == ROUTE_V146_ARCHIVE_EXPANSION:
        return "branch_label_causal_audit_recommends_v146_archive_expansion"
    return "branch_label_causal_audit_source_integrity_failed"


def _position_and_vitals(value: object) -> dict[str, object]:
    payload = _mapping(value)
    return {
        "x": payload.get("x"),
        "y": payload.get("y"),
        "energy_ratio": payload.get("energy_ratio"),
        "hydration_ratio": payload.get("hydration_ratio"),
        "health_ratio": payload.get("health_ratio"),
        "alive": payload.get("alive"),
    }


def _position_distance(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> int | None:
    lx = _optional_int(left.get("x"))
    ly = _optional_int(left.get("y"))
    rx = _optional_int(right.get("x"))
    ry = _optional_int(right.get("y"))
    if lx is None or ly is None or rx is None or ry is None:
        return None
    return abs(lx - rx) + abs(ly - ry)


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    output: list[float] = []
    for item in value:
        try:
            output.append(float(item))
        except (TypeError, ValueError):
            return []
    return output


def _int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def load_inputs_for_audit(
    *,
    v143_report_path: str | Path = DEFAULT_V143_REPORT_PATH,
    v143_dataset_path: str | Path = DEFAULT_V143_DATASET_PATH,
    v144_report_path: str | Path = DEFAULT_V144_REPORT_PATH,
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object], dict[str, object], Path]:
    v143_report = load_json_report(v143_report_path)
    dataset_rows = load_v143_branch_intervention_dataset(v143_dataset_path)
    v144_report = load_json_report(v144_report_path)
    artifact, resolved_artifact_path = load_v144_artifact_for_audit(
        v144_report=v144_report,
        artifact_path=artifact_path,
    )
    return v143_report, dataset_rows, v144_report, artifact, resolved_artifact_path
