from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind.branch_intervention_residual import write_json
from evolution_sim.mind.candidate_campaign import FORBIDDEN_TRAINABLE_TOKENS
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.safe_archive_failure_autopsy import (
    DEFAULT_BP3_BRANCH_EVIDENCE_PATH,
    DEFAULT_BP3_DATASET_PATH,
    DEFAULT_BP3_FAILURE_AUTOPSY_OUTPUT_PATH,
    DEFAULT_BP3_TRAIN_EVAL_REPORT_PATH,
    M3_SAFE_ARCHIVE_TRAIN_EVAL_EXPECTED_CLASSIFICATION,
)

M3_SAFE_ARCHIVE_SEQUENCE_CONTEXT_AUDIT_SCHEMA_VERSION = (
    "m3_safe_archive_sequence_context_audit_report_v1"
)
M3_SAFE_ARCHIVE_SEQUENCE_CONTEXT_AUDIT_POLICY = (
    "diagnostics_only_m3_safe_archive_sequence_context_audit_v1"
)
M3_SAFE_ARCHIVE_SEQUENCE_CONTEXT_AUDIT_COMPLETE_CLASSIFICATION = (
    "m3_safe_archive_sequence_context_audit_complete_no_training"
)
M3_SAFE_ARCHIVE_SEQUENCE_CONTEXT_AUDIT_INCOMPLETE_CLASSIFICATION = (
    "m3_safe_archive_sequence_context_audit_incomplete_no_training"
)
M3_SAFE_ARCHIVE_FAILURE_AUTOPSY_COMPLETE_CLASSIFICATION = (
    "m3_safe_archive_failure_autopsy_complete_no_training"
)
DEFAULT_BP3_SEQUENCE_CONTEXT_AUDIT_OUTPUT_PATH = Path(
    "output/mind/shards/bp3-safe-archive-sequence-context-audit-report.json"
)
DEFAULT_SEQUENCE_CONTEXT_FAILURE_CASES = (
    ("broad", 19),
    ("broad", 29),
    ("broad", 37),
    ("carrion_only", 13),
    ("carrion_only", 37),
    ("carrion_only", 43),
)
SEQUENCE_CONTEXT_FORBIDDEN_TRAINABLE_TOKENS = (
    *FORBIDDEN_TRAINABLE_TOKENS,
    "future",
)


def write_safe_archive_sequence_context_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    write_json(output_path, dict(report))


def build_safe_archive_sequence_context_audit_report(
    *,
    autopsy_report: Mapping[str, object],
    train_eval_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    branch_evidence: Mapping[str, object],
    failure_cases: Sequence[tuple[str, int]] = DEFAULT_SEQUENCE_CONTEXT_FAILURE_CASES,
    input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    validation = validate_safe_archive_sequence_context_audit_inputs(
        autopsy_report=autopsy_report,
        train_eval_report=train_eval_report,
        dataset_rows=dataset_rows,
        branch_evidence=branch_evidence,
    )
    dataset_by_index = _dataset_by_row_index(dataset_rows)
    seed_delta = _seed_delta_by_case(train_eval_report)
    expected_cases = tuple((str(fixture), int(seed)) for fixture, seed in failure_cases)
    traces = [
        dict(trace)
        for trace in _list_of_mappings(autopsy_report.get("override_traces"))
        if (str(trace.get("fixture")), _int(trace.get("seed"), default=-1))
        in set(expected_cases)
    ]
    audit_rows = [
        _trace_sequence_context_audit_row(
            trace=trace,
            support_row=dataset_by_index.get(
                _int(trace.get("support_example_index"), default=-1), {}
            ),
            seed_delta=seed_delta.get(
                (str(trace.get("fixture")), _int(trace.get("seed"), default=-1)),
                {},
            ),
        )
        for trace in traces
    ]
    trainable_context_rows = [
        row["candidate_trainable_context"]
        for row in audit_rows
        if isinstance(row.get("candidate_trainable_context"), Mapping)
    ]
    leakage_scan = sequence_context_trainable_leakage_scan(trainable_context_rows)
    if leakage_scan["passed"] is not True:
        raise ValueError(
            "safe archive sequence context audit trainable leakage failed: "
            + ", ".join(
                str(failure.get("path"))
                for failure in _list_of_mappings(leakage_scan.get("failures"))
            )
        )
    complete = _autopsy_cases_complete(
        autopsy_report=autopsy_report,
        expected_cases=expected_cases,
    )
    classification = (
        M3_SAFE_ARCHIVE_SEQUENCE_CONTEXT_AUDIT_COMPLETE_CLASSIFICATION
        if complete
        else M3_SAFE_ARCHIVE_SEQUENCE_CONTEXT_AUDIT_INCOMPLETE_CLASSIFICATION
    )
    comparison = _action_only_vs_sequence_support_comparison(audit_rows)
    explanations = _per_fixture_seed_action_failure_explanation(
        audit_rows=audit_rows,
        train_eval_report=train_eval_report,
    )
    route = _route_decision(comparison=comparison)
    input_path_payload = _input_path_payload(input_paths)
    return {
        "schema_version": M3_SAFE_ARCHIVE_SEQUENCE_CONTEXT_AUDIT_SCHEMA_VERSION,
        "policy": M3_SAFE_ARCHIVE_SEQUENCE_CONTEXT_AUDIT_POLICY,
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
            "trainable_inputs_are_public_only": True,
            "private_identity_metadata_only": True,
            "future_outcome_in_trainable_fields": False,
        },
        "inputs": {
            "autopsy_report": input_path_payload["autopsy_report"],
            "train_eval_report": input_path_payload["train_eval_report"],
            "dataset": input_path_payload["dataset"],
            "branch_evidence": input_path_payload["branch_evidence"],
            "failure_cases": [
                {"fixture": fixture, "seed": seed} for fixture, seed in expected_cases
            ],
            "digests": {
                "autopsy_report_digest": validation["autopsy_report_digest"],
                "train_eval_report_digest": validation["train_eval_report_digest"],
                "dataset_digest": validation["dataset_digest"],
                "branch_evidence_digest": validation["branch_evidence_digest"],
                "artifact_digest": validation["artifact_digest"],
            },
        },
        "input_validation": validation,
        "leakage_scan": leakage_scan,
        "action_only_vs_sequence_support_comparison": comparison,
        "per_fixture_seed_action_failure_explanation": explanations,
        "audit_trace_count": len(audit_rows),
        "audit_trace_summaries": [
            _audit_row_report_summary(row) for row in audit_rows
        ],
        "route_decision": route,
        "recommended_next_route": route["primary"],
    }


def validate_safe_archive_sequence_context_audit_inputs(
    *,
    autopsy_report: Mapping[str, object],
    train_eval_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    branch_evidence: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    autopsy_inputs = _mapping(autopsy_report.get("inputs"))
    train_validation = _mapping(train_eval_report.get("validation"))
    train_inputs = _mapping(train_eval_report.get("inputs"))
    train_artifact = _mapping(train_eval_report.get("artifact"))
    train_acceptance = _mapping(train_eval_report.get("acceptance"))
    branch_contract = _mapping(branch_evidence.get("contract"))
    branch_generation = _mapping(branch_evidence.get("generation_status"))
    branch_source_integrity = _mapping(branch_evidence.get("source_integrity"))

    autopsy_digest = stable_payload_digest(autopsy_report)
    train_eval_digest = stable_payload_digest(train_eval_report)
    dataset_digest = stable_payload_digest(list(dataset_rows))
    branch_evidence_digest = branch_evidence.get("branch_evidence_digest")
    artifact_digest = train_artifact.get("digest")
    if not branch_evidence_digest:
        failures.append("branch_evidence_digest_missing")
    if not artifact_digest:
        failures.append("train_eval_artifact_digest_missing")

    if (
        _mapping(autopsy_report.get("classification")).get("primary")
        != M3_SAFE_ARCHIVE_FAILURE_AUTOPSY_COMPLETE_CLASSIFICATION
    ):
        failures.append("autopsy_report_not_complete_no_training")
    if (
        _mapping(train_eval_report.get("classification")).get("primary")
        != M3_SAFE_ARCHIVE_TRAIN_EVAL_EXPECTED_CLASSIFICATION
    ):
        failures.append("train_eval_report_not_failed_non_promotional")
    if train_acceptance.get("passed") is not False:
        failures.append("train_eval_acceptance_not_failed")
    if train_eval_report.get("non_promoted") is not True:
        failures.append("train_eval_report_non_promoted_not_true")

    _validate_report_flags(
        name="autopsy_report",
        report=autopsy_report,
        failures=failures,
    )
    _validate_report_flags(
        name="train_eval_report",
        report=train_eval_report,
        failures=failures,
    )
    for key in (
        "training_authorized",
        "promotion_authorized",
        "runtime_promotion_allowed",
    ):
        if train_artifact.get(key) is not False:
            failures.append(f"train_eval_artifact_{key}_not_false")

    if autopsy_inputs.get("train_eval_report_digest") != train_eval_digest:
        failures.append("autopsy_train_eval_report_digest_mismatch")
    if autopsy_inputs.get("dataset_digest") != dataset_digest:
        failures.append("autopsy_dataset_digest_mismatch")
    if autopsy_inputs.get("branch_evidence_digest") != branch_evidence_digest:
        failures.append("autopsy_branch_evidence_digest_mismatch")
    if autopsy_inputs.get("artifact_digest") != artifact_digest:
        failures.append("autopsy_artifact_digest_mismatch")

    if train_validation.get("dataset_digest") != dataset_digest:
        failures.append("train_eval_dataset_digest_mismatch")
    if train_inputs.get("expected_dataset_digest") != dataset_digest:
        failures.append("train_eval_expected_dataset_digest_mismatch")
    if train_validation.get("branch_evidence_digest") != branch_evidence_digest:
        failures.append("train_eval_branch_evidence_digest_mismatch")
    if train_inputs.get("expected_branch_evidence_digest") != branch_evidence_digest:
        failures.append("train_eval_expected_branch_evidence_digest_mismatch")
    if train_validation.get("passed") is not True:
        failures.append("train_eval_input_validation_not_passed")
    if list(train_validation.get("failures") or []) != []:
        failures.append("train_eval_input_validation_failures_present")
    if not dataset_rows:
        failures.append("dataset_rows_missing")

    if branch_evidence.get("training_authorized") is not False:
        failures.append("branch_evidence_training_authorized_not_false")
    if branch_evidence.get("promotion_authorized") is not False:
        failures.append("branch_evidence_promotion_authorized_not_false")
    if branch_evidence.get("non_promoted") is not True:
        failures.append("branch_evidence_non_promoted_not_true")
    if branch_contract.get("diagnostics_only") is not True:
        failures.append("branch_evidence_contract_diagnostics_only_not_true")
    if branch_contract.get("training_authorized") is not False:
        failures.append("branch_evidence_contract_training_authorized_not_false")
    if branch_contract.get("promotion_authorized") is not False:
        failures.append("branch_evidence_contract_promotion_authorized_not_false")
    if branch_contract.get("default_runtime_behavior_changed") is not False:
        failures.append(
            "branch_evidence_contract_default_runtime_behavior_changed_not_false"
        )
    if branch_generation.get("partial") is not False:
        failures.append("branch_evidence_partial_not_false")
    if branch_generation.get("state") != "complete":
        failures.append("branch_evidence_not_complete")
    if branch_source_integrity.get("passed") is not True:
        failures.append("branch_evidence_source_integrity_not_passed")
    if list(branch_source_integrity.get("failures") or []) != []:
        failures.append("branch_evidence_source_integrity_failures_present")

    if failures:
        raise ValueError(
            "safe archive sequence context audit input validation failed: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "policy": "m3_safe_archive_sequence_context_audit_input_validation_v1",
        "passed": True,
        "failures": [],
        "autopsy_report_digest": autopsy_digest,
        "train_eval_report_digest": train_eval_digest,
        "dataset_digest": dataset_digest,
        "branch_evidence_digest": branch_evidence_digest,
        "artifact_digest": artifact_digest,
        "autopsy_classification": _mapping(
            autopsy_report.get("classification")
        ).get("primary"),
        "train_eval_classification": _mapping(
            train_eval_report.get("classification")
        ).get("primary"),
    }


def sequence_context_trainable_leakage_scan(
    trainable_context_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    forbidden_failures: list[dict[str, object]] = []
    for row_index, row in enumerate(trainable_context_rows):
        for path, value in _flatten(row):
            lower_path = path.lower()
            if any(
                token in lower_path
                for token in SEQUENCE_CONTEXT_FORBIDDEN_TRAINABLE_TOKENS
            ):
                forbidden_failures.append(
                    {
                        "row_index": row_index,
                        "path": path,
                        "reason": "forbidden_trainable_path_token",
                    }
                )
            if isinstance(value, str) and _looks_like_private_path_or_digest(value):
                forbidden_failures.append(
                    {
                        "row_index": row_index,
                        "path": path,
                        "reason": "forbidden_trainable_private_value",
                    }
                )
    return {
        "policy": "m3_safe_archive_sequence_context_trainable_leakage_scan_v1",
        "passed": not forbidden_failures,
        "row_count": len(trainable_context_rows),
        "forbidden_tokens": list(SEQUENCE_CONTEXT_FORBIDDEN_TRAINABLE_TOKENS),
        "forbidden_failure_count": len(forbidden_failures),
        "failures": forbidden_failures[:24],
    }


def _trace_sequence_context_audit_row(
    *,
    trace: Mapping[str, object],
    support_row: Mapping[str, object],
    seed_delta: Mapping[str, object],
) -> dict[str, object]:
    selected_action = str(trace.get("selected_support_action", ""))
    support_context = _sequence_context_from_dataset_row(
        support_row,
        label_action=selected_action,
    )
    trace_context = _sequence_context_from_trace(
        trace=trace,
        support_row=support_row,
        selected_action=selected_action,
    )
    action_only_alias = _float(trace.get("nearest_support_distance"), default=1.0) == 0.0
    trace_previous = _previous_public_summaries(trace_context)
    support_previous = _previous_public_summaries(support_context)
    current_sequence_available = trace_previous is not None
    support_sequence_available = support_previous is not None
    separates = (
        action_only_alias
        and current_sequence_available
        and support_sequence_available
        and trace_previous != support_previous
    )
    support_meta = _mapping(support_row.get("metadata"))
    support_outcome = _mapping(support_meta.get("outcome_evidence"))
    return {
        "fixture": trace.get("fixture"),
        "seed": trace.get("seed"),
        "selected_support_action": selected_action,
        "linear_action": trace.get("linear_action"),
        "final_requested_action": trace.get("final_requested_action"),
        "final_resolved_action": trace.get("final_resolved_action"),
        "nearest_support_distance": trace.get("nearest_support_distance"),
        "score_margin": trace.get("score_margin"),
        "support_example_index": trace.get("support_example_index"),
        "action_only_alias": action_only_alias,
        "same_seed_support": _int(support_meta.get("seed"), default=-999)
        == _int(trace.get("seed"), default=-1),
        "source_support": {
            "fixture": support_meta.get("fixture"),
            "seed": support_meta.get("seed"),
            "branch_id": support_meta.get("branch_id"),
            "label_action": _mapping(_mapping(support_row.get("trainable")).get("label")).get(
                "action"
            ),
        },
        "sequence_context": {
            "current_override_previous_context_available": current_sequence_available,
            "support_row_previous_context_available": support_sequence_available,
            "current_previous_context_step_count": (
                len(trace_previous) if trace_previous is not None else 0
            ),
            "support_previous_context_step_count": (
                len(support_previous) if support_previous is not None else 0
            ),
            "sequence_context_separates_action_only_alias": separates,
            "current_context_digest": stable_payload_digest(trace_context),
            "support_context_digest": stable_payload_digest(support_context),
            "reason": _sequence_context_reason(
                action_only_alias=action_only_alias,
                current_sequence_available=current_sequence_available,
                support_sequence_available=support_sequence_available,
                separates=separates,
            ),
        },
        "rollout_context": {
            "support_outcome_context_available": bool(support_outcome),
            "support_deltas_vs_baseline": _mapping(
                support_outcome.get("deltas_vs_baseline")
            ),
            "diagnostics_only_not_trainable": True,
        },
        "failure_signals": _failure_signals(
            trace=trace,
            seed_delta=seed_delta,
        ),
        "candidate_trainable_context": trace_context,
    }


def _sequence_context_from_dataset_row(
    row: Mapping[str, object],
    *,
    label_action: str,
) -> dict[str, object]:
    trainable = _mapping(row.get("trainable"))
    features = _mapping(trainable.get("features"))
    previous = _extract_previous_public_transition_summaries(features)
    return {
        "feature_policy": "public_sequence_context_v1",
        "features": {
            "current_public_observation_input": features.get("observation_input"),
            "current_public_action_mask": features.get("action_mask"),
            "previous_public_transition_summaries": previous or [],
        },
        "label": {
            "action": label_action,
            "label_policy": "diagnostic_support_action_label_v1",
        },
    }


def _sequence_context_from_trace(
    *,
    trace: Mapping[str, object],
    support_row: Mapping[str, object],
    selected_action: str,
) -> dict[str, object]:
    support_context = _sequence_context_from_dataset_row(
        support_row,
        label_action=selected_action,
    )
    support_features = _mapping(support_context.get("features"))
    trace_features = _mapping(trace.get("current_public_sequence_context"))
    previous = _extract_previous_public_transition_summaries(trace_features)
    if previous is None:
        previous = _extract_previous_public_transition_summaries(trace)
    return {
        "feature_policy": "public_sequence_context_v1",
        "features": {
            "current_public_observation_input": trace_features.get(
                "observation_input",
                trace_features.get(
                    "current_public_observation_input",
                    support_features.get("current_public_observation_input"),
                ),
            ),
            "current_public_action_mask": trace_features.get(
                "action_mask",
                trace_features.get(
                    "current_public_action_mask",
                    support_features.get("current_public_action_mask"),
                ),
            ),
            "previous_public_transition_summaries": previous or [],
        },
        "label": {
            "action": selected_action,
            "label_policy": "diagnostic_support_action_label_v1",
        },
    }


def _extract_previous_public_transition_summaries(
    value: Mapping[str, object],
) -> list[object] | None:
    for key in (
        "previous_public_transition_summaries",
        "public_previous_transition_summaries",
        "previous_public_history",
    ):
        candidate = value.get(key)
        if isinstance(candidate, list):
            return list(candidate)
    return None


def _previous_public_summaries(context: Mapping[str, object]) -> list[object] | None:
    features = _mapping(context.get("features"))
    if "previous_public_transition_summaries" not in features:
        return None
    value = features.get("previous_public_transition_summaries")
    if not isinstance(value, list):
        return None
    if not value:
        return None
    return list(value)


def _action_only_vs_sequence_support_comparison(
    audit_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    alias_rows = [
        row for row in audit_rows if row.get("action_only_alias") is True
    ]
    separated_rows = [
        row
        for row in alias_rows
        if _mapping(row.get("sequence_context")).get(
            "sequence_context_separates_action_only_alias"
        )
        is True
    ]
    current_available = [
        row
        for row in alias_rows
        if _mapping(row.get("sequence_context")).get(
            "current_override_previous_context_available"
        )
        is True
    ]
    support_available = [
        row
        for row in alias_rows
        if _mapping(row.get("sequence_context")).get(
            "support_row_previous_context_available"
        )
        is True
    ]
    action_counts = Counter(str(row.get("selected_support_action")) for row in alias_rows)
    cross_seed_count = sum(
        1 for row in alias_rows if row.get("same_seed_support") is not True
    )
    all_aliases_separated = bool(alias_rows) and len(separated_rows) == len(alias_rows)
    return {
        "policy": "m3_safe_archive_sequence_context_support_comparison_v1",
        "failing_override_count": len(audit_rows),
        "action_only": {
            "zero_distance_alias_count": len(alias_rows),
            "cross_seed_alias_count": cross_seed_count,
            "alias_action_counts": _counter_dict(action_counts),
            "interpretation": (
                "one_step_public_observation_action_mask_support_can_alias_"
                "failing_live_overrides"
            ),
        },
        "sequence_context": {
            "current_override_previous_context_available_count": len(
                current_available
            ),
            "support_row_previous_context_available_count": len(support_available),
            "separated_alias_count": len(separated_rows),
            "all_action_only_aliases_separated": all_aliases_separated,
            "separation_result": (
                "separated"
                if all_aliases_separated
                else "not_separated_by_available_bp3_sequence_context"
            ),
            "primary_limitation": _sequence_primary_limitation(
                alias_rows=alias_rows,
                current_available=current_available,
                support_available=support_available,
                separated_rows=separated_rows,
            ),
        },
        "rollout_context": {
            "support_outcome_context_available_count": sum(
                1
                for row in audit_rows
                if _mapping(row.get("rollout_context")).get(
                    "support_outcome_context_available"
                )
                is True
            ),
            "support_deltas_only_diagnostics_not_trainable": True,
        },
    }


def _per_fixture_seed_action_failure_explanation(
    *,
    audit_rows: Sequence[Mapping[str, object]],
    train_eval_report: Mapping[str, object],
) -> list[dict[str, object]]:
    seed_delta = _seed_delta_by_case(train_eval_report)
    grouped: dict[tuple[str, int, str], list[Mapping[str, object]]] = {}
    for row in audit_rows:
        key = (
            str(row.get("fixture")),
            _int(row.get("seed"), default=-1),
            str(row.get("selected_support_action")),
        )
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for (fixture, seed, action), rows in sorted(grouped.items()):
        delta = seed_delta.get((fixture, seed), {})
        support_sources = Counter(
            _source_support_key(_mapping(row.get("source_support"))) for row in rows
        )
        separated = sum(
            1
            for row in rows
            if _mapping(row.get("sequence_context")).get(
                "sequence_context_separates_action_only_alias"
            )
            is True
        )
        result.append(
            {
                "fixture": fixture,
                "seed": seed,
                "selected_support_action": action,
                "override_count": len(rows),
                "same_seed_support_count": sum(
                    1 for row in rows if row.get("same_seed_support") is True
                ),
                "cross_seed_support_count": sum(
                    1 for row in rows if row.get("same_seed_support") is not True
                ),
                "sequence_context_separated_count": separated,
                "sequence_context_explanation": _group_sequence_explanation(rows),
                "failure_signals": _case_failure_signals(
                    fixture=fixture,
                    seed=seed,
                    delta=delta,
                ),
                "source_support_rows": [
                    {"source": source, "count": count}
                    for source, count in sorted(support_sources.items())
                ],
            }
        )
    return result


def _failure_signals(
    *,
    trace: Mapping[str, object],
    seed_delta: Mapping[str, object],
) -> dict[str, object]:
    fixture = str(trace.get("fixture"))
    seed = _int(trace.get("seed"), default=-1)
    return {
        "broad_seed_19_live_regression": (
            fixture == "broad"
            and seed == 19
            and _int(seed_delta.get("alive_delta")) < 0
        ),
        "carrion_seed_13_zero_local_support_transfer": (
            fixture == "carrion_only"
            and seed == 13
            and _int(_mapping(trace.get("joined_support_row_metadata")).get("source_seed"), default=-999)
            != seed
        ),
        "resolved_invalid_increase": _int(
            trace.get(
                "seed_resolved_invalid_delta",
                seed_delta.get("resolved_invalid_action_count_delta"),
            )
        )
        > 0,
        "carrion_seed_43_birth_regression": (
            fixture == "carrion_only"
            and seed == 43
            and _int(seed_delta.get("births_delta")) < 0
        ),
    }


def _case_failure_signals(
    *,
    fixture: str,
    seed: int,
    delta: Mapping[str, object],
) -> dict[str, object]:
    return {
        "broad_seed_19_live_regression": (
            fixture == "broad" and seed == 19 and _int(delta.get("alive_delta")) < 0
        ),
        "resolved_invalid_increase": _int(
            delta.get("resolved_invalid_action_count_delta")
        )
        > 0,
        "carrion_seed_43_birth_regression": (
            fixture == "carrion_only"
            and seed == 43
            and _int(delta.get("births_delta")) < 0
        ),
    }


def _route_decision(*, comparison: Mapping[str, object]) -> dict[str, object]:
    sequence = _mapping(comparison.get("sequence_context"))
    if sequence.get("all_action_only_aliases_separated") is True:
        return {
            "primary": "build_sequence_or_rollout_context_archive",
            "secondary": None,
            "scope": "sequence_context_separated_current_bp3_action_only_aliases",
            "rationale": (
                "available public sequence context separates all observed "
                "one-step support aliases"
            ),
        }
    primary_limitation = str(sequence.get("primary_limitation"))
    if primary_limitation == "no_finalized_prior_public_sequence_context_available":
        return {
            "primary": "stop_bp3_action_only_support_gated_residual_family",
            "secondary": "collect_public_sequence_context_branch_evidence",
            "scope": "current_bp3_one_step_support_archive",
            "rationale": (
                "the current bp3 archive only proves one-step support aliasing; "
                "it contains no finalized prior public sequence context to test "
                "a sequence-aware residual"
            ),
        }
    return {
        "primary": "stop_support_gated_residual_family",
        "secondary": None,
        "scope": "available_sequence_context_support_archive",
        "rationale": (
            "available public sequence context did not separate the bad "
            "support transfers"
        ),
    }


def _input_path_payload(
    input_paths: Mapping[str, str | Path] | None,
) -> dict[str, str]:
    defaults = {
        "autopsy_report": DEFAULT_BP3_FAILURE_AUTOPSY_OUTPUT_PATH,
        "train_eval_report": DEFAULT_BP3_TRAIN_EVAL_REPORT_PATH,
        "dataset": DEFAULT_BP3_DATASET_PATH,
        "branch_evidence": DEFAULT_BP3_BRANCH_EVIDENCE_PATH,
    }
    values = dict(defaults)
    if input_paths is not None:
        for key in defaults:
            if key in input_paths:
                values[key] = Path(input_paths[key])
    return {key: str(value) for key, value in values.items()}


def _sequence_context_reason(
    *,
    action_only_alias: bool,
    current_sequence_available: bool,
    support_sequence_available: bool,
    separates: bool,
) -> str:
    if not action_only_alias:
        return "not_an_action_only_zero_distance_alias"
    if separates:
        return "previous_public_sequence_context_differs_from_support_row"
    if not current_sequence_available and not support_sequence_available:
        return "no_finalized_prior_public_sequence_context_available"
    if not current_sequence_available:
        return "current_override_previous_public_context_missing"
    if not support_sequence_available:
        return "support_row_previous_public_context_missing"
    return "previous_public_sequence_context_does_not_separate_alias"


def _sequence_primary_limitation(
    *,
    alias_rows: Sequence[Mapping[str, object]],
    current_available: Sequence[Mapping[str, object]],
    support_available: Sequence[Mapping[str, object]],
    separated_rows: Sequence[Mapping[str, object]],
) -> str:
    if not alias_rows:
        return "no_action_only_aliases_in_scope"
    if not current_available and not support_available:
        return "no_finalized_prior_public_sequence_context_available"
    if not current_available:
        return "current_override_previous_public_context_missing"
    if not support_available:
        return "support_row_previous_public_context_missing"
    if len(separated_rows) < len(alias_rows):
        return "sequence_context_does_not_separate_all_bad_transfers"
    return "none"


def _group_sequence_explanation(rows: Sequence[Mapping[str, object]]) -> str:
    reasons = Counter(
        str(_mapping(row.get("sequence_context")).get("reason")) for row in rows
    )
    if not reasons:
        return "no_overrides"
    return reasons.most_common(1)[0][0]


def _audit_row_report_summary(row: Mapping[str, object]) -> dict[str, object]:
    context = _mapping(row.get("sequence_context"))
    rollout = _mapping(row.get("rollout_context"))
    source = _mapping(row.get("source_support"))
    return {
        "fixture": row.get("fixture"),
        "seed": row.get("seed"),
        "selected_support_action": row.get("selected_support_action"),
        "linear_action": row.get("linear_action"),
        "final_requested_action": row.get("final_requested_action"),
        "final_resolved_action": row.get("final_resolved_action"),
        "nearest_support_distance": row.get("nearest_support_distance"),
        "score_margin": row.get("score_margin"),
        "support_example_index": row.get("support_example_index"),
        "same_seed_support": row.get("same_seed_support"),
        "source_support": source,
        "sequence_context": {
            "current_override_previous_context_available": context.get(
                "current_override_previous_context_available"
            ),
            "support_row_previous_context_available": context.get(
                "support_row_previous_context_available"
            ),
            "sequence_context_separates_action_only_alias": context.get(
                "sequence_context_separates_action_only_alias"
            ),
            "reason": context.get("reason"),
            "current_context_digest": context.get("current_context_digest"),
            "support_context_digest": context.get("support_context_digest"),
        },
        "rollout_context": {
            "support_outcome_context_available": rollout.get(
                "support_outcome_context_available"
            ),
            "diagnostics_only_not_trainable": True,
        },
        "failure_signals": row.get("failure_signals"),
    }


def _autopsy_cases_complete(
    *,
    autopsy_report: Mapping[str, object],
    expected_cases: Sequence[tuple[str, int]],
) -> bool:
    observed = {
        (str(case.get("fixture")), _int(case.get("seed"), default=-1))
        for case in _list_of_mappings(autopsy_report.get("case_reports"))
    }
    return set(expected_cases).issubset(observed)


def _validate_report_flags(
    *,
    name: str,
    report: Mapping[str, object],
    failures: list[str],
) -> None:
    for key in (
        "diagnostics_only",
        "training_authorized",
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
    ):
        expected = True if key == "diagnostics_only" else False
        if report.get(key) is not expected:
            failures.append(f"{name}_{key}_not_{str(expected).lower()}")


def _dataset_by_row_index(
    dataset_rows: Sequence[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    return {
        _int(_mapping(row.get("metadata")).get("row_index"), default=index): row
        for index, row in enumerate(dataset_rows)
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


def _source_support_key(source: Mapping[str, object]) -> str:
    return (
        f"{source.get('fixture')}:{source.get('seed')}:"
        f"{source.get('label_action')}:{source.get('branch_id')}"
    )


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _flatten(value: object, *, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, object]] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(child, prefix=path))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            rows.extend(_flatten(child, prefix=path))
        return rows
    return [(prefix, value)]


def _looks_like_private_path_or_digest(value: str) -> bool:
    lower = value.lower()
    if len(value) >= 32 and all(char in "0123456789abcdef" for char in lower):
        return True
    return False


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
