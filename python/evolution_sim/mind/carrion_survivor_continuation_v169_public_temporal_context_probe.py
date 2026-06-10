from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gzip
import json
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    _float,
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
    load_v154_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    _complete_bool_mask,
)
from evolution_sim.mind.carrion_survivor_continuation_v160_action_value_scorer import (
    _flatten_public_payload,
)
from evolution_sim.mind.carrion_survivor_continuation_v165_preterminal_target_dataset_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v166_source_split_action_value_scorer import (
    DEFAULT_V165_DATASET_PATH,
    EXPECTED_V165_DATASET_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v167_source_split_failure_autopsy import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V167_REPORT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v168_public_feature_contract_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V168_REPORT_PATH,
    EXPECTED_V167_EXACT_DIGEST,
    FORBIDDEN_TRAINABLE_TOKENS,
    M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION,
    _baseline_payload,
    _decoded_numeric_projection,
    _decoded_public_payload,
    _feature_vectors_from_payloads,
    _first_public_action,
    _json_round_trip_digest,
    _list_like,
    _rank_delta,
    _safe_projection_key,
    candidate_feature_leakage_scan,
    evaluate_feature_family,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_public_temporal_context_probe_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_public_temporal_context_probe_v1"
)
EXPECTED_V168_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v168_public_feature_contract_probe_"
    "partial_feature_contract_support_recommend_another_feature_probe"
)
EXPECTED_V168_EXACT_DIGEST = (
    "1aed29173ba099366aaa4e138b4a2abaaec8a11f6e256be3e999afa8b7d20cfc"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v169-carrion-survivor-continuation-public-temporal-context-probe.json"
)
TEMPORAL_WINDOWS = (1, 3, 5)
FEATURE_FAMILIES = (
    "current_encoded_public_observation_mask",
    "decoded_public_numeric_projection",
    "decoded_public_numeric_plus_previous_same_agent_public_summary",
    "decoded_public_numeric_plus_prior_observation_delta_window_1",
    "decoded_public_numeric_plus_prior_observation_delta_window_3",
    "decoded_public_numeric_plus_prior_observation_delta_window_5",
    "decoded_public_numeric_plus_action_mask_transition_window_1",
    "decoded_public_numeric_plus_action_mask_transition_window_3",
    "decoded_public_numeric_plus_action_mask_transition_window_5",
)


class CarrionSurvivorContinuationV169PublicTemporalContextProbeError(ValueError):
    pass


def run_carrion_survivor_continuation_v169_public_temporal_context_probe(
    *,
    v168_report_path: str | Path = DEFAULT_V168_REPORT_PATH,
    v167_report_path: str | Path = DEFAULT_V167_REPORT_PATH,
    v165_dataset_path: str | Path = DEFAULT_V165_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v168_exact_digest: str | None = EXPECTED_V168_EXACT_DIGEST,
    expected_v167_exact_digest: str | None = EXPECTED_V167_EXACT_DIGEST,
    expected_v165_dataset_digest: str | None = EXPECTED_V165_DATASET_DIGEST,
) -> dict[str, object]:
    v168_report = load_json_report(v168_report_path)
    v167_report = load_json_report(v167_report_path)
    rows = load_v154_dataset(v165_dataset_path)
    source_validation = validate_v169_sources(
        v168_report=v168_report,
        v167_report=v167_report,
        rows=rows,
        expected_v168_exact_digest=expected_v168_exact_digest,
        expected_v167_exact_digest=expected_v167_exact_digest,
        expected_v165_dataset_digest=expected_v165_dataset_digest,
    )
    temporal_contexts = load_public_temporal_contexts(rows)
    families = build_candidate_feature_families(
        rows=rows,
        temporal_contexts_by_row=temporal_contexts["contexts_by_row"],
    )
    baseline = evaluate_feature_family(
        rows=rows,
        family=families["current_encoded_public_observation_mask"],
        v167_report=v167_report,
    )
    candidate_reports = []
    leakage_passed = True
    for family_name in FEATURE_FAMILIES:
        family = families[family_name]
        report = evaluate_feature_family(
            rows=rows,
            family=family,
            v167_report=v167_report,
        )
        leakage = candidate_feature_leakage_scan(family["payloads"])
        comparison = _family_baseline_comparison(
            baseline=baseline,
            family_report=report,
        )
        report["leakage_scan"] = leakage
        report["baseline_comparison"] = comparison
        leakage_passed = leakage_passed and leakage.get("passed") is True
        candidate_reports.append(report)
    classification = _classification(
        source_validation=source_validation,
        leakage_passed=leakage_passed,
        baseline=baseline,
        candidate_reports=candidate_reports,
    )
    best_candidate = _best_candidate(
        baseline=baseline,
        reports=candidate_reports,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_POLICY,
        "contract": {
            "diagnostics_only": True,
            "public_temporal_context_probe_only": True,
            "prior_context_public_records_only": True,
            "prior_context_strictly_before_decision_row": True,
            "source_path_tick_agent_used_only_for_prior_record_lookup": True,
            "runtime_observation_schema_changed": False,
            "runtime_policy_changed": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "replay_viewer_schema_changed": False,
            "threshold_tuning_ran": False,
            "k_tuning_ran": False,
            "training_ran": False,
            "shadow_eval_ran": False,
            "live_ab_allowed": False,
            "promotion_authorized": False,
            "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_future_outcome_as_trainable_input": False,
            "uses_runtime_or_requested_or_resolved_action_as_trainable_input": False,
        },
        "inputs": {
            "v168_report": str(v168_report_path),
            "v167_report": str(v167_report_path),
            "v165_dataset": str(v165_dataset_path),
            "expected_v168_exact_digest": expected_v168_exact_digest,
            "expected_v167_exact_digest": expected_v167_exact_digest,
            "expected_v165_dataset_digest": expected_v165_dataset_digest,
            "feature_families": list(FEATURE_FAMILIES),
            "temporal_windows": list(TEMPORAL_WINDOWS),
        },
        "source_validation": source_validation,
        "temporal_context_load": temporal_contexts["summary"],
        "v168_baseline_family_recomputed": baseline,
        "candidate_feature_family_reports": candidate_reports,
        "per_window_ablation_table": _per_window_ablation_table(
            candidate_reports=candidate_reports,
        ),
        "best_candidate": best_candidate,
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "dataset_digest": stable_payload_digest(rows),
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "replay_viewer_schema_changed": False,
        "threshold_tuning_ran": False,
        "k_tuning_ran": False,
        "training_ran": False,
        "shadow_eval_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v169_sources(
    *,
    v168_report: Mapping[str, object],
    v167_report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v168_exact_digest: str | None = EXPECTED_V168_EXACT_DIGEST,
    expected_v167_exact_digest: str | None = EXPECTED_V167_EXACT_DIGEST,
    expected_v165_dataset_digest: str | None = EXPECTED_V165_DATASET_DIGEST,
) -> dict[str, object]:
    failures: list[str] = []
    if (
        v168_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION
    ):
        failures.append("v168_schema_version_mismatch")
    if v168_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_POLICY:
        failures.append("v168_policy_mismatch")
    observed_v168_classification = str(
        _mapping(v168_report.get("classification")).get("primary") or ""
    )
    if observed_v168_classification != EXPECTED_V168_CLASSIFICATION:
        failures.append("v168_unexpected_classification")
    v168_exact_validation = exact_digest_validation_report(v168_report)
    if v168_exact_validation.get("passed") is not True:
        failures.append("v168_exact_digest_mismatch")
    observed_v168_digest = str(v168_report.get("exact_digest") or "")
    if expected_v168_exact_digest and observed_v168_digest != expected_v168_exact_digest:
        failures.append("v168_unexpected_exact_digest")
    v167_exact_validation = exact_digest_validation_report(v167_report)
    if v167_exact_validation.get("passed") is not True:
        failures.append("v167_exact_digest_mismatch")
    observed_v167_digest = str(v167_report.get("exact_digest") or "")
    if expected_v167_exact_digest and observed_v167_digest != expected_v167_exact_digest:
        failures.append("v167_unexpected_exact_digest")
    v168_source = _mapping(v168_report.get("source_validation"))
    if v168_source.get("passed") is not True:
        failures.append("v168_source_validation_not_passed")
    if str(v168_source.get("observed_v167_exact_digest") or "") != observed_v167_digest:
        failures.append("v168_v167_digest_link_mismatch")
    if expected_v167_exact_digest and str(v168_source.get("observed_v167_exact_digest") or "") != expected_v167_exact_digest:
        failures.append("v168_unexpected_source_v167_digest")
    dataset_digest = stable_payload_digest(rows)
    if expected_v165_dataset_digest and dataset_digest != expected_v165_dataset_digest:
        failures.append("v165_dataset_digest_mismatch")
    if v168_report.get("dataset_digest") != dataset_digest:
        failures.append("v168_report_dataset_digest_mismatch")
    if v167_report.get("dataset_digest") != dataset_digest:
        failures.append("v167_report_dataset_digest_mismatch")
    lifecycle = _lifecycle_scan(v168_report)
    if lifecycle.get("passed") is not True:
        failures.append("v168_lifecycle_not_closed")
    return {
        "policy": "m3_carrion_survivor_continuation_v169_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v168_classification": EXPECTED_V168_CLASSIFICATION,
        "observed_v168_classification": observed_v168_classification,
        "expected_v168_exact_digest": expected_v168_exact_digest,
        "observed_v168_exact_digest": observed_v168_digest,
        "v168_exact_digest_validation": v168_exact_validation,
        "expected_v167_exact_digest": expected_v167_exact_digest,
        "observed_v167_exact_digest": observed_v167_digest,
        "v167_exact_digest_validation": v167_exact_validation,
        "v168_reported_source_v167_digest": v168_source.get("observed_v167_exact_digest"),
        "expected_v165_dataset_digest": expected_v165_dataset_digest,
        "observed_v165_dataset_digest": dataset_digest,
        "v168_reported_dataset_digest": v168_report.get("dataset_digest"),
        "v167_reported_dataset_digest": v167_report.get("dataset_digest"),
        "v168_lifecycle_scan": lifecycle,
        "row_count": len(rows),
    }


def load_public_temporal_contexts(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    file_cache: dict[str, list[dict[str, object]]] = {}
    contexts_by_row: dict[int, dict[str, object]] = {}
    failures = []
    preterminal_rows = 0
    prior_counts: list[int] = []
    for row_index, row in enumerate(rows):
        if (
            row.get("schema_version")
            != M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ):
            continue
        preterminal_rows += 1
        metadata = _mapping(row.get("metadata"))
        source_path = str(metadata.get("source_path") or "")
        source_line = _int(metadata.get("line_number"))
        if not source_path or source_line <= 0:
            failures.append({"row_index": row_index, "failure": "missing_source_lookup"})
            continue
        try:
            if source_path not in file_cache:
                file_cache[source_path] = _load_source_records(Path(source_path))
            records = file_cache[source_path]
            context = _public_temporal_context_for_row(
                row=row,
                source_records=records,
                source_line=source_line,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures.append(
                {
                    "row_index": row_index,
                    "failure": "source_context_load_failed",
                    "error": str(exc),
                }
            )
            continue
        contexts_by_row[row_index] = context
        prior_counts.append(_int(context.get("strict_prior_public_record_count")))
    return {
        "contexts_by_row": contexts_by_row,
        "summary": {
            "policy": "m3_carrion_survivor_continuation_v169_public_temporal_context_load_v1",
            "preterminal_row_count": preterminal_rows,
            "context_loaded_row_count": len(contexts_by_row),
            "context_missing_row_count": preterminal_rows - len(contexts_by_row),
            "rows_with_any_prior_public_context": sum(1 for count in prior_counts if count > 0),
            "max_strict_prior_public_record_count": max(prior_counts, default=0),
            "min_strict_prior_public_record_count": min(prior_counts, default=0),
            "lookup_metadata_policy": {
                "source_path_tick_agent_used_only_for_lookup": True,
                "trainable_payload_contains_lookup_metadata": False,
                "prior_records_must_precede_decision_row": True,
                "prior_records_public_observation_input_and_action_mask_only": True,
            },
            "failures": failures[:24],
        },
    }


def build_candidate_feature_families(
    *,
    rows: Sequence[Mapping[str, object]],
    temporal_contexts_by_row: Mapping[int, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    families = {
        "current_encoded_public_observation_mask": _build_family(
            rows,
            family_name="current_encoded_public_observation_mask",
            builder=lambda index, row: _baseline_payload(row),
        ),
        "decoded_public_numeric_projection": _build_family(
            rows,
            family_name="decoded_public_numeric_projection",
            builder=lambda index, row: _decoded_public_payload(row),
        ),
        "decoded_public_numeric_plus_previous_same_agent_public_summary": _build_family(
            rows,
            family_name="decoded_public_numeric_plus_previous_same_agent_public_summary",
            builder=lambda index, row: _temporal_payload(
                row=row,
                context=temporal_contexts_by_row.get(index),
                mode="previous",
                window=1,
            ),
        ),
    }
    for window in TEMPORAL_WINDOWS:
        families[f"decoded_public_numeric_plus_prior_observation_delta_window_{window}"] = _build_family(
            rows,
            family_name=f"decoded_public_numeric_plus_prior_observation_delta_window_{window}",
            builder=lambda index, row, local_window=window: _temporal_payload(
                row=row,
                context=temporal_contexts_by_row.get(index),
                mode="delta",
                window=local_window,
            ),
        )
        families[f"decoded_public_numeric_plus_action_mask_transition_window_{window}"] = _build_family(
            rows,
            family_name=f"decoded_public_numeric_plus_action_mask_transition_window_{window}",
            builder=lambda index, row, local_window=window: _temporal_payload(
                row=row,
                context=temporal_contexts_by_row.get(index),
                mode="mask",
                window=local_window,
            ),
        )
    return families


def _classification(
    *,
    source_validation: Mapping[str, object],
    leakage_passed: bool,
    baseline: Mapping[str, object],
    candidate_reports: Sequence[Mapping[str, object]],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v169_public_temporal_context_probe_"
    if source_validation.get("passed") is not True or not leakage_passed:
        return prefix + "closed_invalid"
    baseline_zero = set(_list_like(baseline.get("zero_safe_hit_preterminal_source_seeds")))
    viable = [
        report
        for report in _non_baseline_reports(candidate_reports)
        if _comparison_is_net_viable(
            baseline_zero=baseline_zero,
            report=report,
        )
    ]
    if not viable:
        return prefix + "public_temporal_context_probe_no_net_viable_projection_closed"
    ready = [
        report
        for report in viable
        if not _list_like(report.get("zero_safe_hit_preterminal_source_seeds"))
        and _int(report.get("unsupported_prediction_count")) == 0
        and _float(report.get("dominant_predicted_action_share")) <= 0.50
        and _float(report.get("safe_hit_margin_over_best_trivial")) >= 0.05
        and _float(report.get("safe_hit_rate")) > _float(baseline.get("safe_hit_rate"))
    ]
    if ready:
        return prefix + "public_temporal_context_candidate_ready_for_dataset_expansion_design"
    return prefix + "public_temporal_context_partial_support_closed_more_context_or_source"


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification.endswith(
        "public_temporal_context_candidate_ready_for_dataset_expansion_design"
    )
    partial = classification.endswith(
        "public_temporal_context_partial_support_closed_more_context_or_source"
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v169_route_recommendation_v1",
        "recommended_next_route": (
            "dataset_expansion_design_from_public_temporal_context_candidate"
            if ready
            else "more_public_temporal_context_or_source_expansion"
            if partial
            else "keep_public_temporal_context_probe_closed"
        ),
        "dataset_expansion_design_recommended": ready,
        "more_context_or_source_recommended": partial,
        "training_recommended": False,
        "shadow_eval_recommended": False,
        "k_tuning_recommended": False,
        "threshold_tuning_recommended": False,
        "live_ab_allowed": False,
        "runtime_policy_integration_allowed": False,
        "promotion_authorized": False,
        "runtime_action_selection_changed": False,
    }


def _temporal_payload(
    *,
    row: Mapping[str, object],
    context: Mapping[str, object] | None,
    mode: str,
    window: int,
) -> dict[str, object]:
    base = _decoded_public_payload(row)
    features = dict(_mapping(base.get("trainable_public_features")))
    prior_entries = _list_of_mappings(_mapping(context).get("prior_public_records"))
    current_decoded = _mapping(_mapping(context).get("current_decoded_projection"))
    current_mask = _complete_bool_mask(row.get("public_action_mask"))
    available = bool(prior_entries and current_decoded)
    features["public_temporal_context_availability"] = {
        "available": float(available),
        "strict_prior_public_record_count_capped": min(len(prior_entries), window),
    }
    if available and mode == "previous":
        features["previous_same_actor_public_summary"] = dict(
            _mapping(prior_entries[0].get("decoded_projection"))
        )
    elif available and mode == "delta":
        features[f"prior_observation_delta_window_{window}"] = _observation_delta_summary(
            current_decoded=current_decoded,
            prior_entries=prior_entries[:window],
        )
    elif available and mode == "mask":
        features[f"action_mask_transition_window_{window}"] = _action_mask_transition_summary(
            current_mask=current_mask,
            prior_entries=prior_entries[:window],
        )
    return {
        "trainable_public_features": features,
        "public_action_mask": current_mask,
    }


def _public_temporal_context_for_row(
    *,
    row: Mapping[str, object],
    source_records: Sequence[Mapping[str, object]],
    source_line: int,
) -> dict[str, object]:
    current_line_payload = next(
        (
            payload
            for payload in source_records
            if _int(payload.get("line_number")) == source_line
        ),
        {},
    )
    current_record = _mapping(current_line_payload.get("record"))
    current_actor_id = _int(current_record.get("agent_id"))
    current_tick = _int(current_record.get("tick"))
    current_public = _public_record_payload(current_record)
    if not current_public:
        current_public = _public_payload_from_row(row)
    prior_public = []
    for payload in reversed(source_records):
        line_number = _int(payload.get("line_number"))
        if line_number >= source_line:
            continue
        record = _mapping(payload.get("record"))
        if _int(record.get("agent_id")) != current_actor_id:
            continue
        if _int(record.get("tick")) >= current_tick:
            continue
        public_payload = _public_record_payload(record)
        if public_payload:
            prior_public.append(public_payload)
        if len(prior_public) >= max(TEMPORAL_WINDOWS):
            break
    return {
        "policy": "m3_carrion_survivor_continuation_v169_row_public_temporal_context_v1",
        "current_decoded_projection": current_public.get("decoded_projection", {}),
        "current_public_action_mask": current_public.get("public_action_mask", {}),
        "prior_public_records": prior_public,
        "strict_prior_public_record_count": len(prior_public),
        "uses_public_observation_input_only": True,
        "uses_public_action_mask_only": True,
        "strictly_before_decision_row": True,
    }


def _public_record_payload(record: Mapping[str, object]) -> dict[str, object]:
    observation_input = _mapping(record.get("observation_input"))
    decoded = _decoded_numeric_projection(observation_input)
    if not decoded:
        return {}
    return {
        "decoded_projection": decoded,
        "public_action_mask": _complete_bool_mask(
            record.get("public_action_mask") or record.get("action_mask")
        ),
    }


def _public_payload_from_row(row: Mapping[str, object]) -> dict[str, object]:
    observation_input = _mapping(
        _mapping(row.get("trainable_public_features")).get("public_observation")
    )
    decoded = _decoded_numeric_projection(observation_input)
    if not decoded:
        return {}
    return {
        "decoded_projection": decoded,
        "public_action_mask": _complete_bool_mask(row.get("public_action_mask")),
    }


def _load_source_records(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            record = _mapping(payload.get("record") if isinstance(payload, Mapping) else {})
            if record:
                records.append({"line_number": line_number, "record": dict(record)})
    return records


def _observation_delta_summary(
    *,
    current_decoded: Mapping[str, object],
    prior_entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    current_flat = _flatten_numeric_projection(current_decoded)
    prior_flats = [
        _flatten_numeric_projection(_mapping(entry.get("decoded_projection")))
        for entry in prior_entries
    ]
    keys = sorted({*current_flat.keys(), *(key for flat in prior_flats for key in flat)})
    if not keys or not prior_flats:
        return {"available": 0.0}
    summary = {"available": 1.0, "prior_count": len(prior_flats)}
    for key in keys:
        prior_mean = sum(float(flat.get(key, 0.0)) for flat in prior_flats) / len(prior_flats)
        safe_key = _safe_projection_key(key).replace(".", "__")
        summary[f"delta_{safe_key}"] = _round(float(current_flat.get(key, 0.0)) - prior_mean)
    return summary


def _action_mask_transition_summary(
    *,
    current_mask: Mapping[str, bool],
    prior_entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    prior_masks = [
        _complete_bool_mask(entry.get("public_action_mask"))
        for entry in prior_entries
    ]
    if not prior_masks:
        return {"available": 0.0}
    latest = prior_masks[0]
    current_allowed = {action for action in ACTION_NAMES if current_mask.get(action) is True}
    latest_allowed = {action for action in ACTION_NAMES if latest.get(action) is True}
    union = current_allowed | latest_allowed
    intersection = current_allowed & latest_allowed
    summary: dict[str, object] = {
        "available": 1.0,
        "prior_count": len(prior_masks),
        "current_available_count": len(current_allowed),
        "prior_available_mean": _round(
            sum(
                sum(1.0 for action in ACTION_NAMES if mask.get(action) is True)
                for mask in prior_masks
            )
            / len(prior_masks)
        ),
        "became_available_count": len(current_allowed - latest_allowed),
        "became_unavailable_count": len(latest_allowed - current_allowed),
        "stable_available_count": len(intersection),
        "stable_unavailable_count": len(
            [action for action in ACTION_NAMES if action not in union]
        ),
        "latest_allowed_jaccard": _round(
            len(intersection) / max(len(union), 1)
        ),
    }
    for action in ACTION_NAMES:
        prior_mean = sum(1.0 if mask.get(action) is True else 0.0 for mask in prior_masks) / len(prior_masks)
        summary[f"available_delta_{_safe_projection_key(action)}"] = _round(
            (1.0 if current_mask.get(action) is True else 0.0) - prior_mean
        )
    return summary


def _flatten_numeric_projection(payload: Mapping[str, object]) -> dict[str, float]:
    return _flatten_public_payload({"decoded_projection": payload})


def _build_family(
    rows: Sequence[Mapping[str, object]],
    *,
    family_name: str,
    builder: Callable[[int, Mapping[str, object]], Mapping[str, object]],
) -> dict[str, object]:
    payloads = []
    fallback_rows = []
    temporal_rows = []
    for index, row in enumerate(rows):
        payload = dict(builder(index, row))
        payloads.append(payload)
        text = json.dumps(payload, sort_keys=True)
        if "v158_public_feature_fallback" in text:
            fallback_rows.append(index)
        if "public_temporal_context_availability" in text:
            temporal_rows.append(index)
    return {
        "feature_family": family_name,
        "payloads": payloads,
        "fallback_handling": {
            "v158_base_or_missing_source_record_fallback_row_count": len(fallback_rows),
            "fallback_row_indexes": fallback_rows,
            "temporal_context_payload_row_count": len(temporal_rows),
            "base_rows_reported_separately": True,
        },
    }


def _family_baseline_comparison(
    *,
    baseline: Mapping[str, object],
    family_report: Mapping[str, object],
) -> dict[str, object]:
    baseline_zero = set(_list_like(baseline.get("zero_safe_hit_preterminal_source_seeds")))
    family_zero = set(_list_like(family_report.get("zero_safe_hit_preterminal_source_seeds")))
    new_zero = sorted(family_zero - baseline_zero)
    removed_zero = sorted(baseline_zero - family_zero)
    count_delta = len(family_zero) - len(baseline_zero)
    return {
        "zero_safe_hit_seed_count_delta_vs_baseline": count_delta,
        "net_zero_hit_seed_count_change": count_delta,
        "zero_safe_hit_seeds_removed_vs_baseline": removed_zero,
        "new_zero_hit_seeds_introduced": new_zero,
        "candidate_only_swaps_failing_seeds": bool(removed_zero and new_zero and len(family_zero) >= len(baseline_zero)),
        "net_viable_zero_hit_reduction_without_new_zero_seeds": bool(
            len(family_zero) < len(baseline_zero) and not new_zero
        ),
        "safe_hit_rate_delta_vs_baseline": _round(
            _float(family_report.get("safe_hit_rate")) - _float(baseline.get("safe_hit_rate"))
        ),
        "safe_hit_margin_delta_vs_baseline": _round(
            _float(family_report.get("safe_hit_margin_over_best_trivial"))
            - _float(baseline.get("safe_hit_margin_over_best_trivial"))
        ),
    }


def _per_window_ablation_table(
    *,
    candidate_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_family = {str(report.get("feature_family")): report for report in candidate_reports}
    rows = []
    for window in TEMPORAL_WINDOWS:
        for mode, family_name in (
            (
                "prior_observation_delta",
                f"decoded_public_numeric_plus_prior_observation_delta_window_{window}",
            ),
            (
                "action_mask_transition",
                f"decoded_public_numeric_plus_action_mask_transition_window_{window}",
            ),
        ):
            report = by_family.get(family_name, {})
            comparison = _mapping(report.get("baseline_comparison"))
            rows.append(
                {
                    "window": window,
                    "mode": mode,
                    "feature_family": family_name,
                    "safe_hit_rate": report.get("safe_hit_rate"),
                    "safe_hit_margin_over_best_trivial": report.get(
                        "safe_hit_margin_over_best_trivial"
                    ),
                    "dominant_predicted_action_share": report.get(
                        "dominant_predicted_action_share"
                    ),
                    "unsupported_prediction_count": report.get(
                        "unsupported_prediction_count"
                    ),
                    "zero_safe_hit_preterminal_source_seeds": report.get(
                        "zero_safe_hit_preterminal_source_seeds"
                    ),
                    "net_zero_hit_seed_count_change": comparison.get(
                        "net_zero_hit_seed_count_change"
                    ),
                    "new_zero_hit_seeds_introduced": comparison.get(
                        "new_zero_hit_seeds_introduced"
                    ),
                    "moved_nearest_safe_support_rank_to_1_count": _mapping(
                        report.get("failed_row_rank_movement")
                    ).get("moved_nearest_safe_support_rank_to_1_count"),
                }
            )
    return {
        "policy": "m3_carrion_survivor_continuation_v169_per_window_ablation_table_v1",
        "rows": rows,
    }


def _best_candidate(
    *,
    baseline: Mapping[str, object],
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline_zero = set(_list_like(baseline.get("zero_safe_hit_preterminal_source_seeds")))
    candidates = _non_baseline_reports(reports)
    if not candidates:
        return {}
    best = sorted(
        candidates,
        key=lambda report: (
            not _comparison_is_net_viable(baseline_zero=baseline_zero, report=report),
            len(_list_like(report.get("zero_safe_hit_preterminal_source_seeds"))),
            len(
                _list_like(
                    _mapping(report.get("baseline_comparison")).get(
                        "new_zero_hit_seeds_introduced"
                    )
                )
            ),
            -_float(report.get("safe_hit_rate")),
            -_float(report.get("safe_hit_margin_over_best_trivial")),
            str(report.get("feature_family")),
        ),
    )[0]
    comparison = _mapping(best.get("baseline_comparison"))
    return {
        "feature_family": best.get("feature_family"),
        "safe_hit_rate": best.get("safe_hit_rate"),
        "safe_hit_margin_over_best_trivial": best.get(
            "safe_hit_margin_over_best_trivial"
        ),
        "dominant_predicted_action_share": best.get(
            "dominant_predicted_action_share"
        ),
        "unsupported_prediction_count": best.get("unsupported_prediction_count"),
        "zero_safe_hit_preterminal_source_seeds": best.get(
            "zero_safe_hit_preterminal_source_seeds"
        ),
        "net_zero_hit_seed_count_change": comparison.get(
            "net_zero_hit_seed_count_change"
        ),
        "new_zero_hit_seeds_introduced": comparison.get(
            "new_zero_hit_seeds_introduced"
        ),
        "net_viable_zero_hit_reduction_without_new_zero_seeds": comparison.get(
            "net_viable_zero_hit_reduction_without_new_zero_seeds"
        ),
    }


def _comparison_is_net_viable(
    *,
    baseline_zero: set[object],
    report: Mapping[str, object],
) -> bool:
    family_zero = set(_list_like(report.get("zero_safe_hit_preterminal_source_seeds")))
    return len(family_zero) < len(baseline_zero) and not (family_zero - baseline_zero)


def _non_baseline_reports(
    reports: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return [
        report
        for report in reports
        if str(report.get("feature_family")) != "current_encoded_public_observation_mask"
    ]


def _lifecycle_scan(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    for field in (
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "runtime_observation_schema_changed",
        "runtime_policy_changed",
        "replay_viewer_schema_changed",
        "threshold_tuning_ran",
        "k_tuning_ran",
        "training_ran",
        "shadow_eval_ran",
        "live_ab_allowed",
        "promotion_authorized",
    ):
        if report.get(field) is not False:
            failures.append({"field": field, "observed": report.get(field)})
    return {
        "policy": "m3_carrion_survivor_continuation_v169_v168_lifecycle_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }
