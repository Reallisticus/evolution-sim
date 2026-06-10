from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import gzip
import glob
import json
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    _dominant_count_share,
    _float,
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_feature_sufficiency_audit import (
    _features_local_resource_contact_indicators,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    _complete_bool_mask,
)
from evolution_sim.mind.carrion_survivor_continuation_v160_action_value_scorer import (
    DEFAULT_ARTIFACT_OUTPUT_PATH as DEFAULT_V160_ARTIFACT_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V160_REPORT_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION,
    _finite_number,
    _flatten_public_payload,
    _safe_rate,
    _vector_distance,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V161_SHADOW_EVAL_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_action_value_scorer_shadow_eval_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V161_SHADOW_EVAL_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_action_value_scorer_shadow_eval_v1"
)
EXPECTED_V160_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v160_action_value_scorer_"
    "diagnostic_scorer_ready_for_future_shadow_eval"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v161-carrion-survivor-continuation-action-value-shadow-eval.json"
)
DEFAULT_TRAJECTORY_GLOB = (
    "output/mind/v138-strict-heldout-trajectories/open-mind-v3-*-120.jsonl.gz"
)
DEFAULT_MAX_DOMINANT_SHADOW_ACTION_SHARE = 0.50
PREDICTION_EXAMPLE_LIMIT = 32


class CarrionSurvivorContinuationV161ShadowEvalError(ValueError):
    pass


def run_carrion_survivor_continuation_v161_shadow_eval(
    *,
    v160_report_path: str | Path = DEFAULT_V160_REPORT_PATH,
    v160_artifact_path: str | Path = DEFAULT_V160_ARTIFACT_PATH,
    trajectory_glob: str = DEFAULT_TRAJECTORY_GLOB,
    trajectory_paths: Sequence[str | Path] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    max_dominant_shadow_action_share: float = (
        DEFAULT_MAX_DOMINANT_SHADOW_ACTION_SHARE
    ),
) -> dict[str, object]:
    v160_report = load_json_report(v160_report_path)
    v160_artifact = load_json_report(v160_artifact_path)
    source_validation = validate_v160_shadow_sources(
        v160_report=v160_report,
        v160_artifact=v160_artifact,
    )
    evidence = load_shadow_evidence(
        trajectory_glob=trajectory_glob,
        trajectory_paths=trajectory_paths,
    )
    predictions: list[dict[str, object]] = []
    if source_validation.get("passed") is True:
        predictions = shadow_score_records(
            artifact=v160_artifact,
            records=_list_of_mappings(evidence.get("records")),
        )
    shadow = summarize_shadow_predictions(
        predictions,
        decision_record_count=_int(evidence.get("decision_record_count")),
        max_dominant_shadow_action_share=max_dominant_shadow_action_share,
    )
    classification = _top_level_classification(
        source_validation=source_validation,
        evidence=evidence,
        shadow=shadow,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V161_SHADOW_EVAL_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V161_SHADOW_EVAL_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v160_report": str(v160_report_path),
            "v160_artifact": str(v160_artifact_path),
            "trajectory_glob": str(trajectory_glob),
            "trajectory_paths": [str(path) for path in _input_paths(
                trajectory_glob=trajectory_glob,
                trajectory_paths=trajectory_paths,
            )],
            "max_dominant_shadow_action_share": _round(
                max_dominant_shadow_action_share
            ),
        },
        "source_validation": source_validation,
        "evidence": _evidence_report(evidence),
        "shadow_evaluation": shadow,
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "prediction_examples": predictions[:PREDICTION_EXAMPLE_LIMIT],
        "shadow_eval_ran": source_validation.get("passed") is True,
        "diagnostic_shadow_evaluation_ran": source_validation.get("passed") is True,
        "training_ran": False,
        "artifact_created": False,
        "runtime_artifact_created": False,
        "live_ab_ran": False,
        "live_override_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "new_heuristic_action_source_count": 0,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = stable_payload_digest(report)
    write_json(output_path, report)
    return report


def validate_v160_shadow_sources(
    *,
    v160_report: Mapping[str, object],
    v160_artifact: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if (
        v160_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION
    ):
        failures.append("v160_report_schema_version_mismatch")
    if v160_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_POLICY:
        failures.append("v160_report_policy_mismatch")
    observed_classification = _mapping(v160_report.get("classification")).get("primary")
    if observed_classification != EXPECTED_V160_CLASSIFICATION:
        failures.append("v160_unexpected_classification")
    if _mapping(v160_report.get("source_validation")).get("passed") is not True:
        failures.append("v160_source_validation_failed")
    if _mapping(v160_report.get("row_contract_validation")).get("passed") is not True:
        failures.append("v160_row_contract_validation_failed")
    if _mapping(v160_report.get("leakage_scan")).get("passed") is not True:
        failures.append("v160_leakage_scan_failed")
    loo = _mapping(v160_report.get("leave_one_row_out"))
    if loo.get("unsupported_prediction_count") != 0:
        failures.append("v160_loo_unsupported_predictions")
    route = _mapping(v160_report.get("route_recommendation"))
    if route.get("future_shadow_evaluation_recommended") is not True:
        failures.append("v160_shadow_evaluation_not_recommended")
    report_exact_digest_validation = exact_digest_validation_report(v160_report)
    if report_exact_digest_validation.get("passed") is not True:
        failures.append("v160_report_exact_digest_mismatch")
    report_lifecycle = _report_lifecycle_validation(v160_report)
    if report_lifecycle.get("passed") is not True:
        failures.append("v160_report_lifecycle_not_diagnostics_only")

    if (
        v160_artifact.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION
    ):
        failures.append("v160_artifact_schema_version_mismatch")
    if v160_artifact.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V160_ACTION_VALUE_SCORER_POLICY:
        failures.append("v160_artifact_policy_mismatch")
    if v160_artifact.get("scorer_type") != "public_feature_mask_1nn_action_value_scorer_v1":
        failures.append("v160_artifact_scorer_type_mismatch")
    artifact_exact_digest_validation = exact_digest_validation_report(v160_artifact)
    if artifact_exact_digest_validation.get("passed") is not True:
        failures.append("v160_artifact_exact_digest_mismatch")
    artifact_digest = stable_payload_digest(v160_artifact)
    reported_artifact_digest = _mapping(v160_report.get("artifact")).get(
        "artifact_digest"
    )
    if reported_artifact_digest != artifact_digest:
        failures.append("v160_report_artifact_digest_mismatch")
    artifact_contract = _artifact_contract_validation(v160_artifact)
    if artifact_contract.get("passed") is not True:
        failures.append("v160_artifact_contract_invalid")
    artifact_source = _artifact_source_validation(
        report=v160_report,
        artifact=v160_artifact,
    )
    if artifact_source.get("passed") is not True:
        failures.append("v160_artifact_source_invalid")
    return {
        "policy": "m3_carrion_survivor_continuation_v161_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v160_classification": EXPECTED_V160_CLASSIFICATION,
        "observed_v160_classification": observed_classification,
        "v160_report_exact_digest": v160_report.get("exact_digest"),
        "v160_report_payload_digest": stable_payload_digest(v160_report),
        "v160_report_exact_digest_validation": report_exact_digest_validation,
        "v160_artifact_exact_digest": v160_artifact.get("exact_digest"),
        "v160_artifact_digest": artifact_digest,
        "v160_reported_artifact_digest": reported_artifact_digest,
        "v160_artifact_exact_digest_validation": artifact_exact_digest_validation,
        "v160_report_lifecycle_validation": report_lifecycle,
        "v160_artifact_contract_validation": artifact_contract,
        "v160_artifact_source_validation": artifact_source,
    }


def load_shadow_evidence(
    *,
    trajectory_glob: str = DEFAULT_TRAJECTORY_GLOB,
    trajectory_paths: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    paths = _input_paths(trajectory_glob=trajectory_glob, trajectory_paths=trajectory_paths)
    records: list[dict[str, object]] = []
    malformed_records: list[dict[str, object]] = []
    source_record_counts: Counter[str] = Counter()
    skipped_non_policy_record_count = 0
    source_seeds: dict[str, list[int]] = {}
    for path in paths:
        loaded = _load_shadow_evidence_path(path)
        source_record_counts.update({str(path): len(loaded["records"])})
        skipped_non_policy_record_count += _int(
            loaded.get("skipped_non_policy_record_count")
        )
        source_seeds[str(path)] = [int(seed) for seed in loaded["source_seeds"]]
        records.extend(loaded["records"])
        malformed_records.extend(loaded["malformed_records"])
    seed_counts: Counter[str] = Counter()
    action_source_counts: Counter[str] = Counter()
    requested_counts: Counter[str] = Counter()
    for record in records:
        seed_counts.update([str(record.get("source_seed", "unknown"))])
        action_source = str(_mapping(record.get("record")).get("action_source", ""))
        if action_source:
            action_source_counts.update([action_source])
        requested = str(_mapping(record.get("record")).get("requested_action", ""))
        if requested:
            requested_counts.update([requested])
    heuristic_count = sum(
        count
        for source, count in action_source_counts.items()
        if "heuristic" in source.lower()
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v161_shadow_evidence_loader_v1",
        "trajectory_glob": trajectory_glob,
        "trajectory_paths": [str(path) for path in paths],
        "strict_heldout_broad_evidence": _looks_like_strict_broad_paths(paths),
        "closest_existing_deterministic_evaluation_records": True,
        "carrion_fixture_evidence": False,
        "record_count": len(records),
        "decision_record_count": len(records),
        "skipped_non_policy_record_count": skipped_non_policy_record_count,
        "malformed_record_count": len(malformed_records),
        "malformed_records": malformed_records[:16],
        "source_record_counts": dict(sorted(source_record_counts.items())),
        "source_seeds": source_seeds,
        "per_seed_decision_counts": dict(sorted(seed_counts.items())),
        "runtime_requested_action_counts": dict(sorted(requested_counts.items())),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "observed_heuristic_action_source_count": int(heuristic_count),
        "records": records,
    }


def shadow_score_records(
    *,
    artifact: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    requested_sequence_before = _runtime_requested_action_sequence(records)
    requested_sequence_after = _runtime_requested_action_sequence(records)
    for index, evidence in enumerate(records):
        record = _mapping(evidence.get("record"))
        features, public_mask, feature_source = _record_public_feature_payload(record)
        predicted, score_source, neighbors, distance = score_v160_artifact_shadow(
            artifact=artifact,
            trainable_public_features=features,
            public_action_mask=public_mask,
        )
        requested = _action_or_empty(record.get("requested_action"))
        supported = bool(predicted) and public_mask.get(predicted) is True
        unsupported = bool(predicted) and not supported
        comparable = bool(requested)
        predictions.append(
            {
                "record_index": index,
                "source_path": evidence.get("source_path"),
                "line_number": evidence.get("line_number"),
                "source_seed": evidence.get("source_seed"),
                "tick": record.get("tick"),
                "agent_id": record.get("agent_id"),
                "predicted_action": predicted,
                "score_source": score_source,
                "nearest_neighbor_row_indexes": neighbors,
                "nearest_neighbor_distance": _round(distance),
                "feature_source": feature_source,
                "supported_prediction": supported,
                "unsupported_prediction": unsupported,
                "public_mask_allows_predicted_action": supported,
                "actual_runtime_requested_action_available": comparable,
                "actual_runtime_requested_action": requested,
                "would_change_action": bool(predicted and requested and predicted != requested),
                "runtime_action_selection_changed": False,
            }
        )
    before_digest = stable_payload_digest(requested_sequence_before)
    after_digest = stable_payload_digest(requested_sequence_after)
    for prediction in predictions:
        prediction["runtime_requested_action_sequence_digest_before_shadow"] = before_digest
        prediction["runtime_requested_action_sequence_digest_after_shadow"] = after_digest
        prediction["runtime_requested_action_sequence_digest_preserved"] = (
            before_digest == after_digest
        )
    return predictions


def score_v160_artifact_shadow(
    *,
    artifact: Mapping[str, object],
    trainable_public_features: Mapping[str, object],
    public_action_mask: Mapping[str, bool],
) -> tuple[str, str, list[int], float]:
    training_rows = _list_of_mappings(artifact.get("training_rows"))
    if not training_rows:
        return "", "artifact_has_no_training_rows", [], 0.0
    feature_keys = [str(key) for key in artifact.get("feature_keys", [])]
    vector = _artifact_feature_vector(
        artifact=artifact,
        trainable_public_features=trainable_public_features,
        public_action_mask=public_action_mask,
    )
    nearest = sorted(
        training_rows,
        key=lambda row: (
            _vector_distance(vector, _mapping(row.get("feature_vector")), feature_keys),
            _int(row.get("row_index")),
        ),
    )
    neighbor = nearest[0]
    distance = _vector_distance(
        vector,
        _mapping(neighbor.get("feature_vector")),
        feature_keys,
    )
    candidates: list[dict[str, object]] = []
    complete_mask = _complete_bool_mask(public_action_mask)
    for target in _list_of_mappings(neighbor.get("action_value_targets")):
        action = str(target.get("action", ""))
        if (
            action in ACTION_NAMES
            and complete_mask.get(action) is True
            and target.get("target_available") is True
            and _finite_number(target.get("value_target"))
        ):
            candidates.append(dict(target))
    if not candidates:
        return (
            "",
            "nearest_neighbor_no_public_value_candidate",
            [_int(neighbor.get("row_index"))],
            distance,
        )
    ranked = sorted(
        candidates,
        key=lambda target: (
            -_float(target.get("value_target")),
            target.get("safe_target") is not True,
            target.get("robust_safe_action") is not True,
            _action_order(str(target.get("action", ""))),
        ),
    )
    return (
        str(ranked[0].get("action", "")),
        "nearest_neighbor_value_target_shadow",
        [_int(neighbor.get("row_index"))],
        distance,
    )


def summarize_shadow_predictions(
    predictions: Sequence[Mapping[str, object]],
    *,
    decision_record_count: int | None = None,
    max_dominant_shadow_action_share: float = (
        DEFAULT_MAX_DOMINANT_SHADOW_ACTION_SHARE
    ),
) -> dict[str, object]:
    predicted_counts: Counter[str] = Counter()
    per_seed_counts: Counter[str] = Counter()
    per_seed_would_change: Counter[str] = Counter()
    per_seed_no_prediction: Counter[str] = Counter()
    requested_counts: Counter[str] = Counter()
    score_source_counts: Counter[str] = Counter()
    feature_source_counts: Counter[str] = Counter()
    unsupported_count = 0
    supported_count = 0
    no_prediction_count = 0
    would_change_count = 0
    comparable_count = 0
    for prediction in predictions:
        seed = str(prediction.get("source_seed", "unknown"))
        score_source = str(prediction.get("score_source", ""))
        if score_source:
            score_source_counts.update([score_source])
        feature_source = str(prediction.get("feature_source", ""))
        if feature_source:
            feature_source_counts.update([feature_source])
        predicted = _action_or_empty(prediction.get("predicted_action"))
        if predicted:
            predicted_counts.update([predicted])
            per_seed_counts.update([seed])
        else:
            no_prediction_count += 1
            per_seed_no_prediction.update([seed])
        if prediction.get("supported_prediction") is True:
            supported_count += 1
        if prediction.get("unsupported_prediction") is True:
            unsupported_count += 1
        requested = _action_or_empty(prediction.get("actual_runtime_requested_action"))
        if requested:
            requested_counts.update([requested])
        if prediction.get("actual_runtime_requested_action_available") is True:
            comparable_count += 1
        if prediction.get("would_change_action") is True:
            would_change_count += 1
            per_seed_would_change.update([seed])
    prediction_count = sum(predicted_counts.values())
    total_decisions = int(
        decision_record_count if decision_record_count is not None else len(predictions)
    )
    dominant = _dominant_count_share(predicted_counts)
    runtime_digest_before = ""
    runtime_digest_after = ""
    if predictions:
        first = predictions[0]
        runtime_digest_before = str(
            first.get("runtime_requested_action_sequence_digest_before_shadow") or ""
        )
        runtime_digest_after = str(
            first.get("runtime_requested_action_sequence_digest_after_shadow") or ""
        )
    noncollapsed = (
        prediction_count > 0
        and len(predicted_counts) >= 2
        and _float(dominant.get("share")) <= float(max_dominant_shadow_action_share)
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v161_shadow_evaluation_summary_v1",
        "scorer_type": "public_feature_mask_1nn_action_value_scorer_v1",
        "decision_record_count": total_decisions,
        "scored_record_count": len(predictions),
        "prediction_count": prediction_count,
        "scorer_coverage_share": _safe_rate(prediction_count, len(predictions)),
        "shadow_decision_coverage_share": _safe_rate(prediction_count, total_decisions),
        "supported_prediction_count": supported_count,
        "unsupported_shadow_prediction_count": unsupported_count,
        "unsupported_prediction_count": unsupported_count,
        "unsupported_shadow_prediction_share": _safe_rate(
            unsupported_count,
            prediction_count,
        ),
        "no_prediction_count": no_prediction_count,
        "predicted_action_counts": dict(sorted(predicted_counts.items())),
        "dominant_predicted_action": dominant.get("key"),
        "dominant_predicted_action_count": dominant.get("count"),
        "dominant_predicted_action_share": dominant.get("share"),
        "max_dominant_shadow_action_share": _round(
            max_dominant_shadow_action_share
        ),
        "predicted_action_distribution_noncollapsed": noncollapsed,
        "per_seed_shadow_prediction_counts": dict(sorted(per_seed_counts.items())),
        "per_seed_no_prediction_counts": dict(sorted(per_seed_no_prediction.items())),
        "runtime_requested_action_counts": dict(sorted(requested_counts.items())),
        "actual_runtime_requested_action_available_count": comparable_count,
        "would_change_count": would_change_count,
        "would_change_share": _safe_rate(would_change_count, comparable_count),
        "per_seed_would_change_counts": dict(sorted(per_seed_would_change.items())),
        "score_source_counts": dict(sorted(score_source_counts.items())),
        "feature_source_counts": dict(sorted(feature_source_counts.items())),
        "runtime_requested_action_sequence_digest_before_shadow": runtime_digest_before,
        "runtime_requested_action_sequence_digest_after_shadow": runtime_digest_after,
        "runtime_requested_action_sequence_digest_preserved": (
            runtime_digest_before == runtime_digest_after
        ),
        "runtime_requested_actions_source": "existing_evaluation_records_read_only",
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "live_override_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "new_heuristic_action_source_count": 0,
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "shadow_eval_only": True,
        "uses_v160_diagnostics_only_artifact": True,
        "validates_v160_report_and_artifact_digest_before_scoring": True,
        "uses_only_public_trainable_features": True,
        "uses_public_action_masks": True,
        "uses_runtime_requested_actions_for_comparison_only": True,
        "uses_runtime_requested_actions_as_scorer_input": False,
        "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
        "uses_private_world_state_as_trainable_input": False,
        "uses_future_outcome_as_trainable_input": False,
        "uses_branch_reason_as_trainable_input": False,
        "training_authorized": False,
        "training_ran": False,
        "serialized_scorer_artifact_created": False,
        "runtime_artifact_created": False,
        "runtime_policy_integration_allowed": False,
        "live_runtime_override_allowed": False,
        "live_override_allowed": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "new_heuristic_action_selection_sources": False,
    }


def _report_lifecycle_validation(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    for field in (
        "runtime_action_selection_changed",
        "runtime_artifact_created",
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
        "shadow_live_ab_ran",
    ):
        if field in report and report.get(field) is not False:
            failures.append(
                {"field": field, "observed": report.get(field), "expected": False}
            )
    contract = _mapping(report.get("contract"))
    for field in (
        "runtime_action_selection_changed",
        "runtime_artifact_created",
        "runtime_policy_integration_allowed",
        "shadow_live_ab_allowed",
        "live_runtime_override_allowed",
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
    ):
        if field in contract and contract.get(field) is not False:
            failures.append(
                {
                    "field": f"contract.{field}",
                    "observed": contract.get(field),
                    "expected": False,
                }
            )
    return {
        "policy": "m3_carrion_survivor_continuation_v161_report_lifecycle_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _artifact_contract_validation(artifact: Mapping[str, object]) -> dict[str, object]:
    failures = []
    contract = _mapping(artifact.get("contract"))
    expected_true = (
        "diagnostics_only",
        "uses_only_public_trainable_features",
        "uses_public_action_masks",
        "uses_safe_action_set_targets",
        "uses_action_value_targets",
    )
    for field in expected_true:
        if contract.get(field) is not True:
            failures.append(
                {
                    "field": f"contract.{field}",
                    "observed": contract.get(field),
                    "expected": True,
                }
            )
    expected_false = (
        "runtime_artifact",
        "runtime_policy_integration_allowed",
        "runtime_action_selection_changed",
        "uses_private_world_state_as_trainable_input",
        "uses_future_outcome_as_trainable_input",
        "uses_branch_reason_as_trainable_input",
    )
    for field in expected_false:
        if contract.get(field) is not False:
            failures.append(
                {
                    "field": f"contract.{field}",
                    "observed": contract.get(field),
                    "expected": False,
                }
            )
    if contract.get("k_neighbors") != 1:
        failures.append(
            {
                "field": "contract.k_neighbors",
                "observed": contract.get("k_neighbors"),
                "expected": 1,
            }
        )
    if not _list_of_mappings(artifact.get("training_rows")):
        failures.append(
            {
                "field": "training_rows",
                "observed": 0,
                "expected": "non_empty",
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v161_artifact_contract_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _artifact_source_validation(
    *,
    report: Mapping[str, object],
    artifact: Mapping[str, object],
) -> dict[str, object]:
    failures = []
    artifact_source = _mapping(artifact.get("source"))
    report_dataset_digest = report.get("dataset_digest")
    if artifact_source.get("dataset_digest") != report_dataset_digest:
        failures.append("artifact_dataset_digest_mismatch")
    if artifact_source.get("source_validation_passed") is not True:
        failures.append("artifact_source_validation_not_passed")
    if artifact_source.get("row_contract_validation_passed") is not True:
        failures.append("artifact_row_contract_validation_not_passed")
    if artifact_source.get("leakage_scan_passed") is not True:
        failures.append("artifact_leakage_scan_not_passed")
    if _int(artifact_source.get("row_count")) != len(_list_of_mappings(artifact.get("training_rows"))):
        failures.append("artifact_row_count_mismatch")
    return {
        "policy": "m3_carrion_survivor_continuation_v161_artifact_source_validation_v1",
        "passed": not failures,
        "failures": failures,
        "report_dataset_digest": report_dataset_digest,
        "artifact_dataset_digest": artifact_source.get("dataset_digest"),
        "artifact_row_count": artifact_source.get("row_count"),
        "training_row_count": len(_list_of_mappings(artifact.get("training_rows"))),
    }


def _input_paths(
    *,
    trajectory_glob: str,
    trajectory_paths: Sequence[str | Path] | None,
) -> list[Path]:
    if trajectory_paths:
        return [Path(path) for path in trajectory_paths]
    return [Path(path) for path in sorted(glob.glob(trajectory_glob))]


def _load_shadow_evidence_path(path: Path) -> dict[str, object]:
    if not path.exists():
        raise CarrionSurvivorContinuationV161ShadowEvalError(
            f"shadow evidence path does not exist: {path}"
        )
    records: list[dict[str, object]] = []
    malformed_records: list[dict[str, object]] = []
    skipped_non_policy_record_count = 0
    source_seeds: list[int] = []
    header: Mapping[str, object] = {}
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                malformed_records.append(
                    {
                        "source_path": str(path),
                        "line_number": line_number,
                        "reason": "json_decode_error",
                        "message": str(exc),
                    }
                )
                continue
            if not isinstance(payload, Mapping):
                malformed_records.append(
                    {
                        "source_path": str(path),
                        "line_number": line_number,
                        "reason": "payload_not_object",
                    }
                )
                continue
            if line_number == 1:
                header = payload
                source_seeds = _source_seeds_from_header_or_path(header, path)
            record = _payload_record(payload)
            if not record:
                continue
            if str(record.get("action_source", "")).lower() == "passive":
                skipped_non_policy_record_count += 1
                continue
            if not _record_has_shadow_inputs(record):
                malformed_records.append(
                    {
                        "source_path": str(path),
                        "line_number": line_number,
                        "reason": "record_missing_public_shadow_inputs",
                    }
                )
                continue
            records.append(
                {
                    "source_path": str(path),
                    "line_number": line_number,
                    "source_seed": _record_seed(record, source_seeds, path),
                    "record": record,
                }
            )
    return {
        "records": records,
        "malformed_records": malformed_records,
        "skipped_non_policy_record_count": skipped_non_policy_record_count,
        "source_seeds": source_seeds or [_seed_from_path(path)],
    }


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _payload_record(payload: Mapping[str, object]) -> dict[str, object]:
    wrapped = payload.get("record")
    if isinstance(wrapped, Mapping):
        return dict(wrapped)
    if (
        isinstance(payload.get("trainable_public_features"), Mapping)
        or isinstance(payload.get("observation_input"), Mapping)
        or isinstance(payload.get("trainable"), Mapping)
    ):
        return dict(payload)
    return {}


def _record_has_shadow_inputs(record: Mapping[str, object]) -> bool:
    if isinstance(record.get("trainable_public_features"), Mapping):
        return isinstance(record.get("public_action_mask"), Mapping) or isinstance(
            record.get("action_mask"),
            Mapping,
        )
    if isinstance(record.get("observation_input"), Mapping) and isinstance(
        record.get("action_mask"),
        Mapping,
    ):
        return True
    features = _mapping(_mapping(record.get("trainable")).get("features"))
    return isinstance(features.get("observation_input"), Mapping) and isinstance(
        features.get("action_mask"),
        Mapping,
    )


def _source_seeds_from_header_or_path(header: Mapping[str, object], path: Path) -> list[int]:
    seeds = _mapping(header.get("provenance")).get("source_seeds")
    if isinstance(seeds, Sequence) and not isinstance(seeds, (str, bytes)):
        parsed = [_int(seed) for seed in seeds if _int(seed) != 0 or str(seed) == "0"]
        if parsed:
            return parsed
    return [_seed_from_path(path)]


def _seed_from_path(path: Path) -> int:
    for piece in path.name.replace(".jsonl", "").replace(".gz", "").split("-"):
        try:
            return int(piece)
        except ValueError:
            continue
    return 0


def _record_seed(
    record: Mapping[str, object],
    source_seeds: Sequence[int],
    path: Path,
) -> int:
    for key in ("seed", "source_seed"):
        if key in record:
            return _int(record.get(key))
    metadata = _mapping(record.get("metadata"))
    if "seed" in metadata:
        return _int(metadata.get("seed"))
    if len(source_seeds) == 1:
        return int(source_seeds[0])
    return _seed_from_path(path)


def _record_public_feature_payload(
    record: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, bool], str]:
    direct_features = _mapping(record.get("trainable_public_features"))
    if direct_features:
        mask = _complete_bool_mask(
            record.get("public_action_mask") or record.get("action_mask")
        )
        return dict(direct_features), mask, "trainable_public_features"
    trainable_features = _mapping(_mapping(record.get("trainable")).get("features"))
    observation_input = _mapping(
        trainable_features.get("observation_input") or record.get("observation_input")
    )
    action_mask = _complete_bool_mask(
        trainable_features.get("action_mask") or record.get("action_mask")
    )
    feature_row = {
        "trainable": {
            "features": {
                "observation_input": observation_input,
                "action_mask": action_mask,
            }
        }
    }
    return (
        _features_local_resource_contact_indicators(feature_row),
        action_mask,
        "public_local_resource_carrion_contact_indicators",
    )


def _artifact_feature_vector(
    *,
    artifact: Mapping[str, object],
    trainable_public_features: Mapping[str, object],
    public_action_mask: Mapping[str, bool],
) -> dict[str, float]:
    raw = _flatten_public_payload(
        {
            "trainable_public_features": _mapping(trainable_public_features),
            "public_action_mask": _complete_bool_mask(public_action_mask),
        }
    )
    feature_keys = [str(key) for key in artifact.get("feature_keys", [])]
    normalization = _mapping(artifact.get("normalization"))
    vector: dict[str, float] = {}
    for key in feature_keys:
        params = _mapping(normalization.get(key))
        minimum = _float(params.get("min"))
        value_range = _float(params.get("range")) or 1.0
        vector[key] = _round((float(raw.get(key, 0.0)) - minimum) / value_range)
    return vector


def _runtime_requested_action_sequence(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    sequence = []
    for record in records:
        payload = _mapping(record.get("record"))
        sequence.append(
            {
                "source_path": record.get("source_path"),
                "line_number": record.get("line_number"),
                "source_seed": record.get("source_seed"),
                "tick": payload.get("tick"),
                "agent_id": payload.get("agent_id"),
                "requested_action": payload.get("requested_action"),
            }
        )
    return sequence


def _evidence_report(evidence: Mapping[str, object]) -> dict[str, object]:
    payload = dict(evidence)
    payload.pop("records", None)
    return payload


def _top_level_classification(
    *,
    source_validation: Mapping[str, object],
    evidence: Mapping[str, object],
    shadow: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v161_shadow_eval_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_live_ab"
    if _int(evidence.get("decision_record_count")) <= 0:
        return prefix + "evidence_missing_closed_archive_or_evaluation_records"
    if _int(evidence.get("malformed_record_count")) > 0:
        return prefix + "evidence_malformed_closed_feature_contract_work"
    if shadow.get("unsupported_shadow_prediction_count") != 0:
        return prefix + "unsupported_predictions_blocked_no_live_ab"
    if _float(shadow.get("shadow_decision_coverage_share")) <= 0.0:
        return prefix + "no_shadow_coverage_closed_archive_expansion"
    if shadow.get("predicted_action_distribution_noncollapsed") is not True:
        return prefix + "action_distribution_collapsed_blocked_no_live_ab"
    return prefix + "diagnostic_shadow_support_ready_for_future_live_ab_no_runtime_change"


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification.endswith(
        "diagnostic_shadow_support_ready_for_future_live_ab_no_runtime_change"
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v161_route_recommendation_v1",
        "future_separate_opt_in_live_ab_diagnostic_recommended": ready,
        "future_shadow_evaluation_recommended": False,
        "future_shadow_evaluation_completed": True,
        "recommended_next_route": (
            "separate_opt_in_live_ab_diagnostic_without_runtime_override_or_gate_relaxation"
            if ready
            else "archive_expansion_or_public_feature_contract_work_before_more_threshold_tuning"
        ),
        "archive_expansion_recommended": not ready,
        "feature_contract_work_recommended": not ready,
        "threshold_tuning_recommended": False,
        "runtime_policy_integration_allowed": False,
        "live_override_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "runtime_action_selection_changed": False,
    }


def _looks_like_strict_broad_paths(paths: Sequence[Path]) -> bool:
    if not paths:
        return False
    names = [path.name for path in paths]
    return all(name.startswith("open-mind-v3-") for name in names)


def _action_or_empty(value: object) -> str:
    action = str(value or "")
    return action if action in ACTION_NAMES else ""
