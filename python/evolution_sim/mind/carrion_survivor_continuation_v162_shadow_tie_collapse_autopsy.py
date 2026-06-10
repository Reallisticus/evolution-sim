from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from math import ceil
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
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    _complete_bool_mask,
)
from evolution_sim.mind.carrion_survivor_continuation_v160_action_value_scorer import (
    DEFAULT_ARTIFACT_OUTPUT_PATH as DEFAULT_V160_ARTIFACT_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V160_REPORT_PATH,
    _finite_number,
    _safe_rate,
    _vector_distance,
)
from evolution_sim.mind.carrion_survivor_continuation_v161_shadow_eval import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V161_REPORT_PATH,
    DEFAULT_TRAJECTORY_GLOB,
    M3_CARRION_SURVIVOR_CONTINUATION_V161_SHADOW_EVAL_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V161_SHADOW_EVAL_SCHEMA_VERSION,
    _action_or_empty,
    _artifact_feature_vector,
    _record_public_feature_payload,
    _runtime_requested_action_sequence,
    load_shadow_evidence,
    validate_v160_shadow_sources,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V162_SHADOW_TIE_COLLAPSE_AUTOPSY_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_shadow_tie_collapse_autopsy_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V162_SHADOW_TIE_COLLAPSE_AUTOPSY_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_shadow_tie_collapse_autopsy_v1"
)
EXPECTED_V161_COLLAPSED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v161_shadow_eval_"
    "action_distribution_collapsed_blocked_no_live_ab"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v162-carrion-survivor-continuation-shadow-tie-collapse-autopsy.json"
)
DEFAULT_MAX_DOMINANT_NEAREST_ROW_SHARE = 0.50
DEFAULT_MAX_TOP2_NEAREST_ROW_SHARE = 0.70
DEFAULT_MIN_STAY_TIE_BREAK_ATTRIBUTED_SHARE = 0.50
DEFAULT_MAX_DOMINANT_TOP_VALUE_MEMBERSHIP_SHARE = 0.50
DEFAULT_MIN_BROAD_TOP_VALUE_SET_SHARE = 0.50


class CarrionSurvivorContinuationV162ShadowTieCollapseAutopsyError(ValueError):
    pass


def run_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy(
    *,
    v160_report_path: str | Path = DEFAULT_V160_REPORT_PATH,
    v160_artifact_path: str | Path = DEFAULT_V160_ARTIFACT_PATH,
    v161_report_path: str | Path = DEFAULT_V161_REPORT_PATH,
    trajectory_glob: str = DEFAULT_TRAJECTORY_GLOB,
    trajectory_paths: Sequence[str | Path] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    max_dominant_nearest_row_share: float = DEFAULT_MAX_DOMINANT_NEAREST_ROW_SHARE,
    max_top2_nearest_row_share: float = DEFAULT_MAX_TOP2_NEAREST_ROW_SHARE,
    min_stay_tie_break_attributed_share: float = (
        DEFAULT_MIN_STAY_TIE_BREAK_ATTRIBUTED_SHARE
    ),
    max_dominant_top_value_membership_share: float = (
        DEFAULT_MAX_DOMINANT_TOP_VALUE_MEMBERSHIP_SHARE
    ),
    min_broad_top_value_set_share: float = DEFAULT_MIN_BROAD_TOP_VALUE_SET_SHARE,
) -> dict[str, object]:
    v160_report = load_json_report(v160_report_path)
    v160_artifact = load_json_report(v160_artifact_path)
    v161_report = load_json_report(v161_report_path)
    source_validation = validate_v162_sources(
        v160_report=v160_report,
        v160_artifact=v160_artifact,
        v161_report=v161_report,
    )
    selected_trajectory_paths = _selected_trajectory_paths(
        v161_report=v161_report,
        trajectory_paths=trajectory_paths,
    )
    selected_trajectory_glob = _selected_trajectory_glob(
        v161_report=v161_report,
        trajectory_glob=trajectory_glob,
    )
    evidence_report: dict[str, object] = _empty_evidence_report(
        trajectory_glob=selected_trajectory_glob,
        trajectory_paths=selected_trajectory_paths,
    )
    predictions: list[dict[str, object]] = []
    autopsy = _empty_autopsy_summary(
        max_dominant_nearest_row_share=max_dominant_nearest_row_share,
        max_top2_nearest_row_share=max_top2_nearest_row_share,
        min_stay_tie_break_attributed_share=min_stay_tie_break_attributed_share,
        max_dominant_top_value_membership_share=(
            max_dominant_top_value_membership_share
        ),
        min_broad_top_value_set_share=min_broad_top_value_set_share,
    )
    if source_validation.get("passed") is True:
        evidence = load_shadow_evidence(
            trajectory_glob=selected_trajectory_glob,
            trajectory_paths=selected_trajectory_paths,
        )
        evidence_report = _evidence_report(evidence)
        predictions = shadow_tie_collapse_autopsy_records(
            artifact=v160_artifact,
            records=_list_of_mappings(evidence.get("records")),
        )
        autopsy = summarize_shadow_tie_collapse_autopsy(
            predictions,
            v161_report=v161_report,
            decision_record_count=_int(evidence.get("decision_record_count")),
            max_dominant_nearest_row_share=max_dominant_nearest_row_share,
            max_top2_nearest_row_share=max_top2_nearest_row_share,
            min_stay_tie_break_attributed_share=(
                min_stay_tie_break_attributed_share
            ),
            max_dominant_top_value_membership_share=(
                max_dominant_top_value_membership_share
            ),
            min_broad_top_value_set_share=min_broad_top_value_set_share,
        )
    classification = _top_level_classification(
        source_validation=source_validation,
        autopsy=autopsy,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V162_SHADOW_TIE_COLLAPSE_AUTOPSY_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V162_SHADOW_TIE_COLLAPSE_AUTOPSY_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v160_report": str(v160_report_path),
            "v160_artifact": str(v160_artifact_path),
            "v161_report": str(v161_report_path),
            "trajectory_glob": selected_trajectory_glob,
            "trajectory_paths": [str(path) for path in selected_trajectory_paths or []],
            "trajectory_paths_source": (
                "explicit_cli" if trajectory_paths else "v161_report_or_glob"
            ),
            "max_dominant_nearest_row_share": _round(
                max_dominant_nearest_row_share
            ),
            "max_top2_nearest_row_share": _round(max_top2_nearest_row_share),
            "min_stay_tie_break_attributed_share": _round(
                min_stay_tie_break_attributed_share
            ),
            "max_dominant_top_value_membership_share": _round(
                max_dominant_top_value_membership_share
            ),
            "min_broad_top_value_set_share": _round(min_broad_top_value_set_share),
        },
        "source_validation": source_validation,
        "validated_source_digests": _validated_source_digests(source_validation),
        "evidence": evidence_report,
        "shadow_tie_collapse_autopsy": autopsy,
        "prediction_examples": predictions[:32],
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(
            classification=classification,
            autopsy=autopsy,
        ),
        "lifecycle_proof": _lifecycle_proof(),
        "diagnostic_autopsy_ran": source_validation.get("passed") is True,
        "training_ran": False,
        "artifact_created": False,
        "runtime_artifact_created": False,
        "live_ab_allowed": False,
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


def validate_v162_sources(
    *,
    v160_report: Mapping[str, object],
    v160_artifact: Mapping[str, object],
    v161_report: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    v160_validation = validate_v160_shadow_sources(
        v160_report=v160_report,
        v160_artifact=v160_artifact,
    )
    if v160_validation.get("passed") is not True:
        failures.append("v160_report_or_artifact_validation_failed")
    v161_validation = _validate_v161_report(
        v160_report=v160_report,
        v160_artifact=v160_artifact,
        v161_report=v161_report,
    )
    if v161_validation.get("passed") is not True:
        failures.append("v161_report_validation_failed")
    if _mapping(v161_report.get("source_validation")).get("passed") is not True:
        failures.append("v161_source_validation_failed")
    return {
        "policy": "m3_carrion_survivor_continuation_v162_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v160_report_artifact_validation": v160_validation,
        "v161_report_validation": v161_validation,
        "analysis_allowed": not failures,
    }


def shadow_tie_collapse_autopsy_records(
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
        scored = score_v160_artifact_shadow_tie_autopsy(
            artifact=artifact,
            trainable_public_features=features,
            public_action_mask=public_mask,
        )
        predicted = _action_or_empty(scored.get("predicted_action"))
        requested = _action_or_empty(record.get("requested_action"))
        top_set = [str(action) for action in scored.get("top_value_candidate_set", [])]
        predictions.append(
            {
                "record_index": index,
                "source_path": evidence.get("source_path"),
                "line_number": evidence.get("line_number"),
                "source_seed": evidence.get("source_seed"),
                "tick": record.get("tick"),
                "agent_id": record.get("agent_id"),
                "feature_source": feature_source,
                "public_feature_vector_digest": scored.get(
                    "public_feature_vector_digest"
                ),
                "nearest_neighbor_row_index": scored.get("nearest_neighbor_row_index"),
                "nearest_neighbor_distance": scored.get("nearest_neighbor_distance"),
                "predicted_action": predicted,
                "score_source": scored.get("score_source"),
                "top_value": scored.get("top_value"),
                "top_value_candidate_set": top_set,
                "top_value_candidate_set_size": len(top_set),
                "top_value_candidate_set_key": _candidate_set_key(top_set),
                "predicted_action_from_tied_top_value_set": (
                    bool(predicted) and len(top_set) > 1
                ),
                "stay_prediction_from_tied_top_value_set": (
                    predicted == "stay" and len(top_set) > 1
                ),
                "candidate_count_after_public_mask_filtering": scored.get(
                    "candidate_count_after_public_mask_filtering"
                ),
                "action_mask_filtering": scored.get("action_mask_filtering"),
                "actual_runtime_requested_action_available": bool(requested),
                "actual_runtime_requested_action": requested,
                "would_change_action": bool(
                    predicted and requested and predicted != requested
                ),
                "runtime_requested_action_used_as_scorer_input": False,
                "runtime_action_selection_changed": False,
            }
        )
    before_digest = stable_payload_digest(requested_sequence_before)
    after_digest = stable_payload_digest(requested_sequence_after)
    for prediction in predictions:
        prediction["runtime_requested_action_sequence_digest_before_autopsy"] = (
            before_digest
        )
        prediction["runtime_requested_action_sequence_digest_after_autopsy"] = (
            after_digest
        )
        prediction["runtime_requested_action_sequence_digest_preserved"] = (
            before_digest == after_digest
        )
    return predictions


def score_v160_artifact_shadow_tie_autopsy(
    *,
    artifact: Mapping[str, object],
    trainable_public_features: Mapping[str, object],
    public_action_mask: Mapping[str, bool],
) -> dict[str, object]:
    training_rows = _list_of_mappings(artifact.get("training_rows"))
    feature_keys = [str(key) for key in artifact.get("feature_keys", [])]
    vector = _artifact_feature_vector(
        artifact=artifact,
        trainable_public_features=trainable_public_features,
        public_action_mask=public_action_mask,
    )
    if not training_rows:
        return {
            "score_source": "artifact_has_no_training_rows",
            "predicted_action": "",
            "nearest_neighbor_row_index": None,
            "nearest_neighbor_distance": 0.0,
            "public_feature_vector_digest": stable_payload_digest(vector),
            "top_value": None,
            "top_value_candidate_set": [],
            "candidate_count_after_public_mask_filtering": 0,
            "action_mask_filtering": _empty_action_mask_filtering(),
        }
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
    filter_summary = _candidate_filter_summary(
        targets=_list_of_mappings(neighbor.get("action_value_targets")),
        public_action_mask=public_action_mask,
    )
    for target in _list_of_mappings(neighbor.get("action_value_targets")):
        action = str(target.get("action", ""))
        if (
            action in ACTION_NAMES
            and _complete_bool_mask(public_action_mask).get(action) is True
            and target.get("target_available") is True
            and _finite_number(target.get("value_target"))
        ):
            candidates.append(dict(target))
    if not candidates:
        return {
            "score_source": "nearest_neighbor_no_public_value_candidate",
            "predicted_action": "",
            "nearest_neighbor_row_index": _int(neighbor.get("row_index")),
            "nearest_neighbor_distance": _round(distance),
            "public_feature_vector_digest": stable_payload_digest(vector),
            "top_value": None,
            "top_value_candidate_set": [],
            "candidate_count_after_public_mask_filtering": 0,
            "action_mask_filtering": filter_summary,
        }
    top_value = max(_float(candidate.get("value_target")) for candidate in candidates)
    top_set = sorted(
        {
            str(candidate.get("action", ""))
            for candidate in candidates
            if _float(candidate.get("value_target")) == top_value
        },
        key=_action_order,
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
    return {
        "score_source": "nearest_neighbor_value_target_shadow_tie_autopsy",
        "predicted_action": str(ranked[0].get("action", "")),
        "nearest_neighbor_row_index": _int(neighbor.get("row_index")),
        "nearest_neighbor_distance": _round(distance),
        "public_feature_vector_digest": stable_payload_digest(vector),
        "top_value": _round(top_value),
        "top_value_candidate_set": top_set,
        "candidate_count_after_public_mask_filtering": len(candidates),
        "action_mask_filtering": filter_summary,
    }


def summarize_shadow_tie_collapse_autopsy(
    predictions: Sequence[Mapping[str, object]],
    *,
    v161_report: Mapping[str, object],
    decision_record_count: int | None = None,
    max_dominant_nearest_row_share: float = DEFAULT_MAX_DOMINANT_NEAREST_ROW_SHARE,
    max_top2_nearest_row_share: float = DEFAULT_MAX_TOP2_NEAREST_ROW_SHARE,
    min_stay_tie_break_attributed_share: float = (
        DEFAULT_MIN_STAY_TIE_BREAK_ATTRIBUTED_SHARE
    ),
    max_dominant_top_value_membership_share: float = (
        DEFAULT_MAX_DOMINANT_TOP_VALUE_MEMBERSHIP_SHARE
    ),
    min_broad_top_value_set_share: float = DEFAULT_MIN_BROAD_TOP_VALUE_SET_SHARE,
) -> dict[str, object]:
    predicted_counts: Counter[str] = Counter()
    nearest_row_counts: Counter[str] = Counter()
    nearest_row_predicted_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    nearest_row_distances: defaultdict[str, list[float]] = defaultdict(list)
    distances: list[float] = []
    distances_by_action: defaultdict[str, list[float]] = defaultdict(list)
    top_set_counts: Counter[str] = Counter()
    top_set_size_counts: Counter[str] = Counter()
    membership_counts: Counter[str] = Counter()
    candidate_count_distribution: Counter[str] = Counter()
    feature_vector_counts: Counter[str] = Counter()
    feature_vector_predicted_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    feature_vector_row_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    feature_source_counts: Counter[str] = Counter()
    filter_counter: Counter[str] = Counter()
    mask_allowed_counts: Counter[str] = Counter()
    candidate_action_counts: Counter[str] = Counter()
    requested_counts: Counter[str] = Counter()
    would_change_count = 0
    comparable_count = 0
    stay_tie_break_attributed_count = 0
    tied_prediction_count = 0
    for prediction in predictions:
        predicted = _action_or_empty(prediction.get("predicted_action"))
        if predicted:
            predicted_counts.update([predicted])
        row_key = str(prediction.get("nearest_neighbor_row_index"))
        if row_key and row_key != "None":
            nearest_row_counts.update([row_key])
            if predicted:
                nearest_row_predicted_counts[row_key].update([predicted])
        distance = _float(prediction.get("nearest_neighbor_distance"))
        distances.append(distance)
        if row_key and row_key != "None":
            nearest_row_distances[row_key].append(distance)
        if predicted:
            distances_by_action[predicted].append(distance)
        top_set = [str(action) for action in prediction.get("top_value_candidate_set", [])]
        set_key = _candidate_set_key(top_set)
        top_set_counts.update([set_key])
        top_set_size_counts.update([str(len(top_set))])
        for action in top_set:
            membership_counts.update([action])
        if prediction.get("predicted_action_from_tied_top_value_set") is True:
            tied_prediction_count += 1
        if prediction.get("stay_prediction_from_tied_top_value_set") is True:
            stay_tie_break_attributed_count += 1
        candidate_count_distribution.update(
            [str(_int(prediction.get("candidate_count_after_public_mask_filtering")))]
        )
        digest = str(prediction.get("public_feature_vector_digest") or "")
        if digest:
            feature_vector_counts.update([digest])
            if predicted:
                feature_vector_predicted_counts[digest].update([predicted])
            if row_key and row_key != "None":
                feature_vector_row_counts[digest].update([row_key])
        feature_source = str(prediction.get("feature_source") or "")
        if feature_source:
            feature_source_counts.update([feature_source])
        _merge_filtering(
            filter_counter=filter_counter,
            mask_allowed_counts=mask_allowed_counts,
            candidate_action_counts=candidate_action_counts,
            filtering=_mapping(prediction.get("action_mask_filtering")),
        )
        requested = _action_or_empty(prediction.get("actual_runtime_requested_action"))
        if requested:
            requested_counts.update([requested])
        if prediction.get("actual_runtime_requested_action_available") is True:
            comparable_count += 1
        if prediction.get("would_change_action") is True:
            would_change_count += 1
    prediction_count = sum(predicted_counts.values())
    total_decisions = int(
        decision_record_count if decision_record_count is not None else len(predictions)
    )
    dominant_nearest = _dominant_count_share(nearest_row_counts)
    top2_nearest_share = _top_n_share(nearest_row_counts, 2)
    stay_prediction_count = int(predicted_counts.get("stay", 0))
    stay_tie_break_share = _safe_rate(
        stay_tie_break_attributed_count,
        stay_prediction_count,
    )
    membership_dominant = _dominant_count_share(membership_counts)
    broad_top_set_count = sum(
        count for size, count in top_set_size_counts.items() if int(size) > 1
    )
    broad_top_set_share = _safe_rate(broad_top_set_count, len(predictions))
    set_valued_noncollapsed = (
        sum(membership_counts.values()) > 0
        and len(membership_counts) >= 2
        and _float(membership_dominant.get("share"))
        <= float(max_dominant_top_value_membership_share)
    )
    set_valued_broad = (
        set_valued_noncollapsed
        and broad_top_set_share >= float(min_broad_top_value_set_share)
    )
    nearest_row_concentration_high = (
        _float(dominant_nearest.get("share"))
        > float(max_dominant_nearest_row_share)
        or top2_nearest_share > float(max_top2_nearest_row_share)
    )
    tie_break_attributed_collapse_high = (
        stay_prediction_count > 0
        and stay_tie_break_share >= float(min_stay_tie_break_attributed_share)
    )
    v161_shadow = _mapping(v161_report.get("shadow_evaluation"))
    comparison = _comparison_only_would_change_metrics(
        v161_shadow=v161_shadow,
        predicted_counts=predicted_counts,
        would_change_count=would_change_count,
        comparable_count=comparable_count,
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy_summary_v1",
        "scorer_type": "public_feature_mask_1nn_action_value_scorer_v1",
        "decision_record_count": total_decisions,
        "scored_record_count": len(predictions),
        "prediction_count": prediction_count,
        "predicted_action_counts": dict(sorted(predicted_counts.items())),
        "nearest_neighbor_row_attribution": {
            "counts": _sorted_counter_dict(nearest_row_counts),
            "top_rows": _nearest_row_top_rows(
                row_counts=nearest_row_counts,
                row_predicted_counts=nearest_row_predicted_counts,
                row_distances=nearest_row_distances,
                total=len(predictions),
            ),
            "dominant_nearest_row": dominant_nearest.get("key"),
            "dominant_nearest_row_count": dominant_nearest.get("count"),
            "dominant_nearest_row_share": dominant_nearest.get("share"),
            "top2_nearest_row_share": _round(top2_nearest_share),
            "max_dominant_nearest_row_share": _round(
                max_dominant_nearest_row_share
            ),
            "max_top2_nearest_row_share": _round(max_top2_nearest_row_share),
            "nearest_row_concentration_high": nearest_row_concentration_high,
        },
        "predicted_action_by_nearest_row": {
            row: dict(sorted(counts.items()))
            for row, counts in sorted(
                nearest_row_predicted_counts.items(),
                key=lambda item: int(item[0]),
            )
        },
        "nearest_distance": {
            "overall": _distance_stats(distances),
            "by_predicted_action": {
                action: _distance_stats(values)
                for action, values in sorted(distances_by_action.items())
            },
        },
        "top_value_tied_candidate_sets_after_public_action_mask_filtering": {
            "counts": _sorted_counter_dict(top_set_counts),
            "tied_top_value_set_count": broad_top_set_count,
            "tied_top_value_set_share": broad_top_set_share,
        },
        "top_value_set_size_distribution": _sorted_counter_dict(top_set_size_counts),
        "top_value_candidate_membership_counts": _sorted_counter_dict(
            membership_counts
        ),
        "top_value_candidate_membership_dominance": {
            "dominant_action": membership_dominant.get("key"),
            "dominant_count": membership_dominant.get("count"),
            "dominant_share": membership_dominant.get("share"),
            "max_dominant_top_value_membership_share": _round(
                max_dominant_top_value_membership_share
            ),
            "set_valued_candidates_noncollapsed": set_valued_noncollapsed,
            "set_valued_candidates_noncollapsed_but_broad": set_valued_broad,
            "min_broad_top_value_set_share": _round(min_broad_top_value_set_share),
        },
        "deterministic_tie_break_attribution": {
            "stay_prediction_count": stay_prediction_count,
            "stay_predictions_from_tied_top_value_sets": (
                stay_tie_break_attributed_count
            ),
            "stay_tie_break_attributed_share_of_stay_predictions": (
                stay_tie_break_share
            ),
            "tied_prediction_count": tied_prediction_count,
            "tied_prediction_share": _safe_rate(tied_prediction_count, len(predictions)),
            "min_stay_tie_break_attributed_share": _round(
                min_stay_tie_break_attributed_share
            ),
            "tie_break_attributed_collapse_high": (
                tie_break_attributed_collapse_high
            ),
        },
        "public_feature_aliasing_concentration": {
            "feature_source_counts": dict(sorted(feature_source_counts.items())),
            "unique_public_feature_vector_count": len(feature_vector_counts),
            "aliased_public_feature_vector_record_count": sum(
                count for count in feature_vector_counts.values() if count > 1
            ),
            "aliased_public_feature_vector_share": _safe_rate(
                sum(count for count in feature_vector_counts.values() if count > 1),
                len(predictions),
            ),
            "dominant_public_feature_vector_digest": _dominant_count_share(
                feature_vector_counts
            ).get("key"),
            "dominant_public_feature_vector_count": _dominant_count_share(
                feature_vector_counts
            ).get("count"),
            "dominant_public_feature_vector_share": _dominant_count_share(
                feature_vector_counts
            ).get("share"),
            "top_public_feature_vector_attributions": (
                _top_feature_vector_attributions(
                    feature_vector_counts=feature_vector_counts,
                    feature_vector_predicted_counts=feature_vector_predicted_counts,
                    feature_vector_row_counts=feature_vector_row_counts,
                    total=len(predictions),
                )
            ),
        },
        "action_mask_filtering_summary": {
            "total_target_count": int(filter_counter.get("total_target_count", 0)),
            "valid_action_target_count": int(
                filter_counter.get("valid_action_target_count", 0)
            ),
            "public_mask_allowed_target_count": int(
                filter_counter.get("public_mask_allowed_target_count", 0)
            ),
            "public_mask_filtered_target_count": int(
                filter_counter.get("public_mask_filtered_target_count", 0)
            ),
            "target_unavailable_count": int(
                filter_counter.get("target_unavailable_count", 0)
            ),
            "nonfinite_value_target_count": int(
                filter_counter.get("nonfinite_value_target_count", 0)
            ),
            "candidate_after_filtering_count": int(
                filter_counter.get("candidate_after_filtering_count", 0)
            ),
            "candidate_count_distribution": _sorted_counter_dict(
                candidate_count_distribution
            ),
            "public_action_mask_allowed_counts": dict(
                sorted(mask_allowed_counts.items(), key=lambda item: _action_order(item[0]))
            ),
            "candidate_after_filtering_action_counts": dict(
                sorted(candidate_action_counts.items(), key=lambda item: _action_order(item[0]))
            ),
        },
        "comparison_only_would_change_metrics_from_v161": comparison,
        "nearest_row_concentration_high": nearest_row_concentration_high,
        "tie_break_attributed_collapse_high": tie_break_attributed_collapse_high,
        "set_valued_candidates_noncollapsed_but_broad": set_valued_broad,
        "runtime_requested_action_used_as_scorer_input": False,
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
    }


def _validate_v161_report(
    *,
    v160_report: Mapping[str, object],
    v160_artifact: Mapping[str, object],
    v161_report: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if (
        v161_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V161_SHADOW_EVAL_SCHEMA_VERSION
    ):
        failures.append("v161_schema_version_mismatch")
    if v161_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V161_SHADOW_EVAL_POLICY:
        failures.append("v161_policy_mismatch")
    observed_classification = _mapping(v161_report.get("classification")).get(
        "primary"
    )
    if observed_classification != EXPECTED_V161_COLLAPSED_CLASSIFICATION:
        failures.append("v161_unexpected_or_noncollapsed_classification")
    report_exact_digest_validation = exact_digest_validation_report(v161_report)
    if report_exact_digest_validation.get("passed") is not True:
        failures.append("v161_report_exact_digest_mismatch")
    v161_source = _mapping(v161_report.get("source_validation"))
    if v161_source.get("passed") is not True:
        failures.append("v161_source_validation_failed")
    v160_report_exact_digest = v160_report.get("exact_digest")
    if v161_source.get("v160_report_exact_digest") != v160_report_exact_digest:
        failures.append("v161_v160_report_exact_digest_mismatch")
    v160_artifact_exact_digest = v160_artifact.get("exact_digest")
    if v161_source.get("v160_artifact_exact_digest") != v160_artifact_exact_digest:
        failures.append("v161_v160_artifact_exact_digest_mismatch")
    v160_artifact_digest = stable_payload_digest(v160_artifact)
    if v161_source.get("v160_artifact_digest") != v160_artifact_digest:
        failures.append("v161_v160_artifact_digest_mismatch")
    shadow = _mapping(v161_report.get("shadow_evaluation"))
    if shadow.get("predicted_action_distribution_noncollapsed") is not False:
        failures.append("v161_shadow_not_collapsed")
    if shadow.get("runtime_action_selection_changed") is not False:
        failures.append("v161_shadow_runtime_action_selection_changed")
    lifecycle = _v161_lifecycle_validation(v161_report)
    if lifecycle.get("passed") is not True:
        failures.append("v161_lifecycle_not_diagnostics_only")
    return {
        "policy": "m3_carrion_survivor_continuation_v162_v161_report_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v161_classification": EXPECTED_V161_COLLAPSED_CLASSIFICATION,
        "observed_v161_classification": observed_classification,
        "v161_report_exact_digest": v161_report.get("exact_digest"),
        "v161_report_exact_digest_validation": report_exact_digest_validation,
        "v161_source_validation_passed": v161_source.get("passed") is True,
        "v161_reported_v160_report_exact_digest": v161_source.get(
            "v160_report_exact_digest"
        ),
        "current_v160_report_exact_digest": v160_report_exact_digest,
        "v161_reported_v160_artifact_exact_digest": v161_source.get(
            "v160_artifact_exact_digest"
        ),
        "current_v160_artifact_exact_digest": v160_artifact_exact_digest,
        "v161_reported_v160_artifact_digest": v161_source.get(
            "v160_artifact_digest"
        ),
        "current_v160_artifact_digest": v160_artifact_digest,
        "v161_lifecycle_validation": lifecycle,
    }


def _candidate_filter_summary(
    *,
    targets: Sequence[Mapping[str, object]],
    public_action_mask: Mapping[str, bool],
) -> dict[str, object]:
    complete_mask = _complete_bool_mask(public_action_mask)
    target_count = 0
    valid_count = 0
    mask_allowed_count = 0
    mask_filtered_count = 0
    unavailable_count = 0
    nonfinite_count = 0
    candidate_count = 0
    mask_allowed_actions: Counter[str] = Counter()
    candidate_actions: Counter[str] = Counter()
    for action in ACTION_NAMES:
        if complete_mask.get(action) is True:
            mask_allowed_actions.update([action])
    for target in targets:
        target_count += 1
        action = str(target.get("action", ""))
        if action not in ACTION_NAMES:
            continue
        valid_count += 1
        mask_allowed = complete_mask.get(action) is True
        target_available = target.get("target_available") is True
        finite_value = _finite_number(target.get("value_target"))
        if mask_allowed:
            mask_allowed_count += 1
        else:
            mask_filtered_count += 1
        if not target_available:
            unavailable_count += 1
        if not finite_value:
            nonfinite_count += 1
        if mask_allowed and target_available and finite_value:
            candidate_count += 1
            candidate_actions.update([action])
    return {
        "total_target_count": target_count,
        "valid_action_target_count": valid_count,
        "public_mask_allowed_target_count": mask_allowed_count,
        "public_mask_filtered_target_count": mask_filtered_count,
        "target_unavailable_count": unavailable_count,
        "nonfinite_value_target_count": nonfinite_count,
        "candidate_after_filtering_count": candidate_count,
        "public_action_mask_allowed_actions": dict(
            sorted(mask_allowed_actions.items(), key=lambda item: _action_order(item[0]))
        ),
        "candidate_after_filtering_actions": dict(
            sorted(candidate_actions.items(), key=lambda item: _action_order(item[0]))
        ),
    }


def _merge_filtering(
    *,
    filter_counter: Counter[str],
    mask_allowed_counts: Counter[str],
    candidate_action_counts: Counter[str],
    filtering: Mapping[str, object],
) -> None:
    for key in (
        "total_target_count",
        "valid_action_target_count",
        "public_mask_allowed_target_count",
        "public_mask_filtered_target_count",
        "target_unavailable_count",
        "nonfinite_value_target_count",
        "candidate_after_filtering_count",
    ):
        filter_counter.update({key: _int(filtering.get(key))})
    mask_allowed_counts.update(
        {
            action: _int(count)
            for action, count in _mapping(
                filtering.get("public_action_mask_allowed_actions")
            ).items()
        }
    )
    candidate_action_counts.update(
        {
            action: _int(count)
            for action, count in _mapping(
                filtering.get("candidate_after_filtering_actions")
            ).items()
        }
    )


def _comparison_only_would_change_metrics(
    *,
    v161_shadow: Mapping[str, object],
    predicted_counts: Counter[str],
    would_change_count: int,
    comparable_count: int,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v162_v161_comparison_only_would_change_metrics_v1",
        "from_v161_report": True,
        "actual_runtime_requested_action_available_count": v161_shadow.get(
            "actual_runtime_requested_action_available_count"
        ),
        "runtime_requested_action_counts": _mapping(
            v161_shadow.get("runtime_requested_action_counts")
        ),
        "would_change_count": v161_shadow.get("would_change_count"),
        "would_change_share": v161_shadow.get("would_change_share"),
        "per_seed_would_change_counts": _mapping(
            v161_shadow.get("per_seed_would_change_counts")
        ),
        "runtime_requested_action_sequence_digest_before_shadow": v161_shadow.get(
            "runtime_requested_action_sequence_digest_before_shadow"
        ),
        "runtime_requested_action_sequence_digest_after_shadow": v161_shadow.get(
            "runtime_requested_action_sequence_digest_after_shadow"
        ),
        "runtime_requested_action_sequence_digest_preserved": v161_shadow.get(
            "runtime_requested_action_sequence_digest_preserved"
        ),
        "recomputed_from_v162_autopsy": {
            "predicted_action_counts": dict(sorted(predicted_counts.items())),
            "would_change_count": would_change_count,
            "would_change_share": _safe_rate(would_change_count, comparable_count),
            "matches_v161_predicted_action_counts": (
                dict(sorted(predicted_counts.items()))
                == _mapping(v161_shadow.get("predicted_action_counts"))
            ),
            "matches_v161_would_change_count": (
                would_change_count == _int(v161_shadow.get("would_change_count"))
            ),
        },
        "runtime_requested_actions_source": "existing_evaluation_records_read_only",
        "runtime_requested_action_used_as_scorer_input": False,
        "runtime_action_selection_changed": False,
    }


def _top_level_classification(
    *,
    source_validation: Mapping[str, object],
    autopsy: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_live_ab"
    if (
        autopsy.get("nearest_row_concentration_high") is True
        or autopsy.get("tie_break_attributed_collapse_high") is True
    ):
        return prefix + "tie_collapse_archive_feature_support_blocked_no_live_ab"
    if autopsy.get("set_valued_candidates_noncollapsed_but_broad") is True:
        return prefix + "set_valued_candidates_broad_resolver_blocked_no_live_ab"
    return prefix + "collapse_unexplained_blocked_no_live_ab"


def _route_recommendation(
    *,
    classification: str,
    autopsy: Mapping[str, object],
) -> dict[str, object]:
    source_valid = "source_invalid" not in classification
    set_valued_broad = autopsy.get("set_valued_candidates_noncollapsed_but_broad") is True
    return {
        "policy": "m3_carrion_survivor_continuation_v162_route_recommendation_v1",
        "recommended_next_route": (
            "repair_v160_v161_source_digest_lineage_before_autopsy"
            if not source_valid
            else (
                "public_feature_contract_expansion_or_separate_set_valued_resolver_diagnostic_no_live_ab"
                if set_valued_broad
                else "archive_feature_support_expansion_before_any_live_ab"
            )
        ),
        "archive_expansion_recommended": source_valid,
        "feature_contract_work_recommended": source_valid,
        "separate_set_valued_resolver_diagnostic_recommended": set_valued_broad,
        "threshold_tuning_recommended": False,
        "future_shadow_evaluation_recommended": False,
        "future_separate_opt_in_live_ab_diagnostic_recommended": False,
        "live_ab_allowed": False,
        "live_override_allowed": False,
        "runtime_policy_integration_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "runtime_action_selection_changed": False,
    }


def _v161_lifecycle_validation(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    for field in (
        "runtime_action_selection_changed",
        "runtime_artifact_created",
        "live_ab_ran",
        "live_override_allowed",
        "promotion_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
    ):
        if field in report and report.get(field) is not False:
            failures.append(
                {"field": field, "observed": report.get(field), "expected": False}
            )
    contract = _mapping(report.get("contract"))
    for field in (
        "runtime_action_selection_changed",
        "runtime_artifact_created",
        "live_ab_allowed",
        "live_override_allowed",
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
        "policy": "m3_carrion_survivor_continuation_v162_v161_lifecycle_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _validated_source_digests(
    source_validation: Mapping[str, object],
) -> dict[str, object]:
    v160 = _mapping(source_validation.get("v160_report_artifact_validation"))
    v161 = _mapping(source_validation.get("v161_report_validation"))
    return {
        "v160_report_exact_digest": v160.get("v160_report_exact_digest"),
        "v160_report_exact_digest_validation": v160.get(
            "v160_report_exact_digest_validation"
        ),
        "v160_artifact_exact_digest": v160.get("v160_artifact_exact_digest"),
        "v160_artifact_exact_digest_validation": v160.get(
            "v160_artifact_exact_digest_validation"
        ),
        "v160_artifact_digest": v160.get("v160_artifact_digest"),
        "v160_reported_artifact_digest": v160.get("v160_reported_artifact_digest"),
        "v161_report_exact_digest": v161.get("v161_report_exact_digest"),
        "v161_report_exact_digest_validation": v161.get(
            "v161_report_exact_digest_validation"
        ),
        "v161_collapsed_classification_validated": (
            v161.get("observed_v161_classification")
            == EXPECTED_V161_COLLAPSED_CLASSIFICATION
        ),
        "expected_v161_collapsed_classification": (
            EXPECTED_V161_COLLAPSED_CLASSIFICATION
        ),
        "observed_v161_classification": v161.get("observed_v161_classification"),
    }


def _selected_trajectory_paths(
    *,
    v161_report: Mapping[str, object],
    trajectory_paths: Sequence[str | Path] | None,
) -> list[Path] | None:
    if trajectory_paths:
        return [Path(path) for path in trajectory_paths]
    report_paths = _mapping(v161_report.get("inputs")).get("trajectory_paths")
    if isinstance(report_paths, Sequence) and not isinstance(report_paths, (str, bytes)):
        parsed = [Path(str(path)) for path in report_paths]
        if parsed:
            return parsed
    return None


def _selected_trajectory_glob(
    *,
    v161_report: Mapping[str, object],
    trajectory_glob: str,
) -> str:
    report_glob = _mapping(v161_report.get("inputs")).get("trajectory_glob")
    return str(report_glob or trajectory_glob)


def _evidence_report(evidence: Mapping[str, object]) -> dict[str, object]:
    payload = dict(evidence)
    payload.pop("records", None)
    return payload


def _empty_evidence_report(
    *,
    trajectory_glob: str,
    trajectory_paths: Sequence[Path] | None,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v162_shadow_evidence_not_loaded_v1",
        "trajectory_glob": trajectory_glob,
        "trajectory_paths": [str(path) for path in trajectory_paths or []],
        "record_count": 0,
        "decision_record_count": 0,
        "not_loaded_reason": "source_validation_not_passed",
    }


def _empty_autopsy_summary(
    *,
    max_dominant_nearest_row_share: float,
    max_top2_nearest_row_share: float,
    min_stay_tie_break_attributed_share: float,
    max_dominant_top_value_membership_share: float,
    min_broad_top_value_set_share: float,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy_summary_v1",
        "decision_record_count": 0,
        "scored_record_count": 0,
        "prediction_count": 0,
        "predicted_action_counts": {},
        "nearest_neighbor_row_attribution": {
            "counts": {},
            "top_rows": [],
            "dominant_nearest_row": None,
            "dominant_nearest_row_count": 0,
            "dominant_nearest_row_share": 0.0,
            "top2_nearest_row_share": 0.0,
            "max_dominant_nearest_row_share": _round(
                max_dominant_nearest_row_share
            ),
            "max_top2_nearest_row_share": _round(max_top2_nearest_row_share),
            "nearest_row_concentration_high": False,
        },
        "predicted_action_by_nearest_row": {},
        "nearest_distance": {"overall": _distance_stats([]), "by_predicted_action": {}},
        "top_value_tied_candidate_sets_after_public_action_mask_filtering": {
            "counts": {},
            "tied_top_value_set_count": 0,
            "tied_top_value_set_share": 0.0,
        },
        "top_value_set_size_distribution": {},
        "top_value_candidate_membership_counts": {},
        "top_value_candidate_membership_dominance": {
            "dominant_action": None,
            "dominant_count": 0,
            "dominant_share": 0.0,
            "max_dominant_top_value_membership_share": _round(
                max_dominant_top_value_membership_share
            ),
            "set_valued_candidates_noncollapsed": False,
            "set_valued_candidates_noncollapsed_but_broad": False,
            "min_broad_top_value_set_share": _round(min_broad_top_value_set_share),
        },
        "deterministic_tie_break_attribution": {
            "stay_prediction_count": 0,
            "stay_predictions_from_tied_top_value_sets": 0,
            "stay_tie_break_attributed_share_of_stay_predictions": 0.0,
            "tied_prediction_count": 0,
            "tied_prediction_share": 0.0,
            "min_stay_tie_break_attributed_share": _round(
                min_stay_tie_break_attributed_share
            ),
            "tie_break_attributed_collapse_high": False,
        },
        "public_feature_aliasing_concentration": {
            "feature_source_counts": {},
            "unique_public_feature_vector_count": 0,
            "aliased_public_feature_vector_record_count": 0,
            "aliased_public_feature_vector_share": 0.0,
            "dominant_public_feature_vector_digest": None,
            "dominant_public_feature_vector_count": 0,
            "dominant_public_feature_vector_share": 0.0,
            "top_public_feature_vector_attributions": [],
        },
        "action_mask_filtering_summary": {},
        "comparison_only_would_change_metrics_from_v161": {},
        "nearest_row_concentration_high": False,
        "tie_break_attributed_collapse_high": False,
        "set_valued_candidates_noncollapsed_but_broad": False,
        "runtime_requested_action_used_as_scorer_input": False,
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
    }


def _empty_action_mask_filtering() -> dict[str, object]:
    return {
        "total_target_count": 0,
        "valid_action_target_count": 0,
        "public_mask_allowed_target_count": 0,
        "public_mask_filtered_target_count": 0,
        "target_unavailable_count": 0,
        "nonfinite_value_target_count": 0,
        "candidate_after_filtering_count": 0,
        "public_action_mask_allowed_actions": {},
        "candidate_after_filtering_actions": {},
    }


def _distance_stats(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2.0
    p95_index = max(0, min(count - 1, ceil(0.95 * count) - 1))
    return {
        "count": count,
        "min": _round(ordered[0]),
        "median": _round(median),
        "p95": _round(ordered[p95_index]),
        "max": _round(ordered[-1]),
    }


def _nearest_row_top_rows(
    *,
    row_counts: Counter[str],
    row_predicted_counts: Mapping[str, Counter[str]],
    row_distances: Mapping[str, Sequence[float]],
    total: int,
) -> list[dict[str, object]]:
    rows = []
    for row, count in sorted(
        row_counts.items(),
        key=lambda item: (-int(item[1]), int(item[0])),
    )[:16]:
        rows.append(
            {
                "row_index": int(row),
                "count": int(count),
                "share": _safe_rate(count, total),
                "predicted_action_counts": dict(
                    sorted(row_predicted_counts.get(row, Counter()).items())
                ),
                "distance": _distance_stats(row_distances.get(row, [])),
            }
        )
    return rows


def _top_feature_vector_attributions(
    *,
    feature_vector_counts: Counter[str],
    feature_vector_predicted_counts: Mapping[str, Counter[str]],
    feature_vector_row_counts: Mapping[str, Counter[str]],
    total: int,
) -> list[dict[str, object]]:
    rows = []
    for digest, count in sorted(
        feature_vector_counts.items(),
        key=lambda item: (-int(item[1]), item[0]),
    )[:16]:
        rows.append(
            {
                "public_feature_vector_digest": digest,
                "count": int(count),
                "share": _safe_rate(count, total),
                "predicted_action_counts": dict(
                    sorted(feature_vector_predicted_counts.get(digest, Counter()).items())
                ),
                "nearest_row_counts": _sorted_counter_dict(
                    feature_vector_row_counts.get(digest, Counter())
                ),
            }
        )
    return rows


def _sorted_counter_dict(counter: Mapping[str, int]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(
            counter.items(),
            key=_counter_sort_key,
        )
    }


def _counter_sort_key(item: tuple[str, int]) -> tuple[int, int, object]:
    key, value = item
    text = str(key)
    if text.isdigit():
        return (-int(value), 0, int(text))
    return (-int(value), 1, text)


def _top_n_share(counter: Counter[str], n: int) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    top_count = sum(count for _, count in counter.most_common(n))
    return _safe_rate(top_count, total)


def _candidate_set_key(actions: Sequence[str]) -> str:
    valid = [action for action in actions if action in ACTION_NAMES]
    if not valid:
        return "<none>"
    return "|".join(sorted(valid, key=_action_order))


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "shadow_autopsy_only": True,
        "uses_v160_diagnostics_only_artifact": True,
        "validates_v160_report_and_artifact_digest_before_analysis": True,
        "validates_v161_report_digest_before_analysis": True,
        "requires_v161_collapsed_classification_before_analysis": True,
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
        "live_ab_allowed": False,
        "live_runtime_override_allowed": False,
        "live_override_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "new_heuristic_action_selection_sources": False,
    }


def _lifecycle_proof() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v162_lifecycle_proof_v1",
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "live_override_allowed": False,
        "threshold_tuning_recommended": False,
        "new_heuristic_action_source_count": 0,
    }
