from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.branch_action_oracle_audit import (
    MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.branch_action_oracle_labels import (
    MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
    _MOVE_DELTAS,
    _branch_continuation_archive_feature_vector,
    _squared_distance,
)
from evolution_sim.mind.branch_mode_objective_audit import (
    build_branch_mode_objective_audit_report,
    _action_option_mode,
    _list_of_mappings,
    _mapping,
    _objective_rows,
    _predicted_action_scores,
    _predicted_option_mode,
    _safe_rate,
    _target_local_score,
    _target_terminal_projection,
    _training_examples,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_REPOSITION_FRONTIER_AUDIT_SCHEMA_VERSION = (
    "mind_v3_reposition_frontier_audit_v1"
)
V91_MIN_LABEL_COUNT = 48
V91_MIN_REPOSITION_MULTI_MOVE_LABEL_COUNT = 36
V91_OPTION_MODE_ACCURACY_FLOOR = 0.70
V91_MATERIAL_ONLY_OPTION_MODE_ACCURACY_FLOOR = 0.60
V91_MAX_DOMINANT_OPTION_MODE_SHARE = 0.75
V91_REPOSITION_DIRECTION_ACCURACY_FLOOR = 0.60
V91_MIN_TARGET_LOCAL_SCORE_DELTA = 0.0
V91_MIN_MEAN_TERMINAL_ALIVE_DELTA = 0.0
V91_MIN_MEAN_BIRTH_DELTA = 0.0


class BranchRepositionFrontierAuditError(ValueError):
    pass


def build_reposition_frontier_audit_report(
    branch_action_oracle_labels: Mapping[str, object],
    *,
    source_branch_action_oracle_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if (
        branch_action_oracle_labels.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION
    ):
        raise BranchRepositionFrontierAuditError(
            "branch action oracle labels have unsupported schema_version"
        )
    if (
        source_branch_action_oracle_audit is not None
        and source_branch_action_oracle_audit.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION
    ):
        raise BranchRepositionFrontierAuditError(
            "source branch action oracle audit has unsupported schema_version"
        )

    labels = _list_of_mappings(branch_action_oracle_labels.get("labels"), "labels")
    rows = _objective_rows(labels)
    mode_objective = build_branch_mode_objective_audit_report(
        branch_action_oracle_labels,
        source_branch_action_oracle_audit=source_branch_action_oracle_audit,
    )
    coverage = _coverage_report(
        rows,
        branch_action_oracle_labels=branch_action_oracle_labels,
        source_branch_action_oracle_audit=source_branch_action_oracle_audit,
    )
    decomposition = _reposition_decomposition_report(rows)
    utility = _branch_utility_report(
        rows,
        option_mode_k=int(
            _mapping(_mapping(mode_objective.get("support")).get("option_mode")).get(
                "best_nearest_neighbor_k",
                1,
            )
        ),
    )
    acceptance = _acceptance(
        coverage=coverage,
        mode_objective=mode_objective,
        decomposition=decomposition,
        utility=utility,
    )
    contract = {
        "schema_version": MIND_V3_REPOSITION_FRONTIER_AUDIT_SCHEMA_VERSION,
        "source_label_schema_version": branch_action_oracle_labels.get(
            "schema_version"
        ),
        "source_audit_schema_version": (
            source_branch_action_oracle_audit.get("schema_version")
            if source_branch_action_oracle_audit is not None
            else None
        ),
        "runtime_policy_trained": False,
        "selection_policy": "failure_frontier_mode_balanced_v1",
        "support_floors": {
            "replay_verified_all_labels": True,
            "heuristic_action_source_count": 0,
            "unsupported_oracle_action_count": 0,
            "min_label_count_unless_insufficient_eligible_rows": (
                V91_MIN_LABEL_COUNT
            ),
            "min_reposition_multi_move_labels_unless_insufficient_eligible_rows": (
                V91_MIN_REPOSITION_MULTI_MOVE_LABEL_COUNT
            ),
            "option_mode_accuracy": V91_OPTION_MODE_ACCURACY_FLOOR,
            "material_only_option_mode_accuracy": (
                V91_MATERIAL_ONLY_OPTION_MODE_ACCURACY_FLOOR
            ),
            "max_dominant_predicted_option_mode": (
                V91_MAX_DOMINANT_OPTION_MODE_SHARE
            ),
            "reposition_exact_direction_accuracy": (
                V91_REPOSITION_DIRECTION_ACCURACY_FLOOR
            ),
            "predicted_branch_utility_mean_target_local_score_delta": (
                V91_MIN_TARGET_LOCAL_SCORE_DELTA
            ),
            "mean_terminal_alive_delta": V91_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            "mean_birth_delta": V91_MIN_MEAN_BIRTH_DELTA,
        },
    }
    return {
        "schema_version": MIND_V3_REPOSITION_FRONTIER_AUDIT_SCHEMA_VERSION,
        "contract": contract,
        "provenance": {
            "source_label_digest": stable_payload_digest(
                {
                    "schema_version": branch_action_oracle_labels.get(
                        "schema_version"
                    ),
                    "contract": branch_action_oracle_labels.get("contract"),
                    "aggregate": branch_action_oracle_labels.get("aggregate"),
                    "labels": labels,
                }
            ),
            "source_audit_digest": (
                stable_payload_digest(
                    {
                        "schema_version": source_branch_action_oracle_audit.get(
                            "schema_version"
                        ),
                        "contract": source_branch_action_oracle_audit.get(
                            "contract"
                        ),
                        "aggregate": source_branch_action_oracle_audit.get(
                            "aggregate"
                        ),
                        "discovery": source_branch_action_oracle_audit.get(
                            "discovery"
                        ),
                    }
                )
                if source_branch_action_oracle_audit is not None
                else None
            ),
            "contract_digest": stable_payload_digest(contract),
        },
        "coverage": coverage,
        "mode_objective_support": _mode_support_excerpt(mode_objective),
        "reposition_decomposition": decomposition,
        "branch_utility": utility,
        "reposition_frontier_support_probe": {
            "policy": "v91_failure_frontier_reposition_diagnostic_v1",
            "best_accuracy": decomposition["best_decoder"]["accuracy"],
            "material_support_accuracy_floor": V91_REPOSITION_DIRECTION_ACCURACY_FLOOR,
            "materially_supports_reposition_frontier": (
                acceptance["v91_reposition_frontier_diagnostic_accepted"]
            ),
            "runtime_policy_status": "diagnostic_only_no_runtime_policy_trained",
        },
        "acceptance": acceptance,
    }


def load_json_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    try:
        with _open_input(resolved) as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise BranchRepositionFrontierAuditError(
            f"failed to read report: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BranchRepositionFrontierAuditError(
            f"report is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BranchRepositionFrontierAuditError("report must be a JSON object")
    return payload


def write_reposition_frontier_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _coverage_report(
    rows: Sequence[Mapping[str, object]],
    *,
    branch_action_oracle_labels: Mapping[str, object],
    source_branch_action_oracle_audit: Mapping[str, object] | None,
) -> dict[str, object]:
    aggregate = _mapping(branch_action_oracle_labels.get("aggregate"))
    frontier = _frontier_source_counts(source_branch_action_oracle_audit)
    reposition_rows = _reposition_multi_move_rows(rows)
    return {
        "label_count": len(rows),
        "labels_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in rows).items())
        ),
        "labels_by_option_mode": dict(
            sorted(Counter(str(row.get("option_mode")) for row in rows).items())
        ),
        "material_labels_by_seed_mode": _material_by_seed_mode(rows),
        "reposition_multi_move_label_count": len(reposition_rows),
        "replay_verified_all_labels": bool(
            aggregate.get("replay_verified_all_labels", False)
        ),
        "heuristic_action_source_count": int(
            aggregate.get("heuristic_action_source_count", 0)
            or (
                0
                if aggregate.get("zero_heuristic_all_labels") is True
                else aggregate.get("label_count", 0)
            )
        ),
        "zero_heuristic_all_labels": bool(
            aggregate.get("zero_heuristic_all_labels", False)
        ),
        "unsupported_oracle_action_count": int(
            aggregate.get("unsupported_oracle_action_count", 0)
        ),
        "source_frontier_selection": frontier,
        "insufficient_label_rows_proven": (
            len(rows) < V91_MIN_LABEL_COUNT
            and int(frontier.get("eligible_row_count", 0)) < V91_MIN_LABEL_COUNT
        ),
        "insufficient_reposition_rows_proven": (
            len(reposition_rows) < V91_MIN_REPOSITION_MULTI_MOVE_LABEL_COUNT
            and int(frontier.get("eligible_multi_move_reposition_row_count", 0))
            < V91_MIN_REPOSITION_MULTI_MOVE_LABEL_COUNT
        ),
    }


def _mode_support_excerpt(mode_objective: Mapping[str, object]) -> dict[str, object]:
    support = _mapping(mode_objective.get("support"))
    return {
        "option_mode": _support_fields(support.get("option_mode")),
        "material_only_option_mode": _support_fields(
            support.get("material_only_option_mode")
        ),
        "population_first": _support_fields(support.get("population_first")),
        "target_local": _support_fields(support.get("target_local")),
    }


def _support_fields(value: object) -> dict[str, object]:
    item = _mapping(value)
    return {
        "best_nearest_neighbor_k": item.get("best_nearest_neighbor_k"),
        "best_mode_accuracy": item.get("best_mode_accuracy"),
        "best_exact_action_accuracy": item.get("best_exact_action_accuracy"),
        "dominant_prediction_mode": item.get("dominant_prediction_mode"),
        "dominant_prediction_mode_share": item.get(
            "dominant_prediction_mode_share"
        ),
    }


def _reposition_decomposition_report(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    reposition_rows = _reposition_multi_move_rows(rows)
    decoder_reports = []
    for k in (1, 3, 5):
        decoder_reports.append(
            _summarize_predictions(
                f"current_value_k{k}",
                _current_value_predictions(reposition_rows, k=k),
            )
        )
        decoder_reports.append(
            _summarize_predictions(
                f"direct_direction_classifier_k{k}",
                _state_classifier_predictions(reposition_rows, k=k, variant="direct"),
            )
        )
        decoder_reports.append(
            _summarize_predictions(
                f"axis_delta_sign_classifier_k{k}",
                _state_classifier_predictions(
                    reposition_rows,
                    k=k,
                    variant="axis_delta_sign",
                ),
            )
        )
    for target in ("water", "carrion", "resource_need"):
        decoder_reports.append(
            _summarize_predictions(
                f"navigation_baseline_toward_{target}",
                _navigation_baseline_predictions(reposition_rows, target=target),
            )
        )
    best = max(
        decoder_reports,
        key=lambda item: (
            float(item["accuracy"]),
            int(item["correct_count"]),
            -str(item["decoder"]).count("navigation_baseline"),
            str(item["decoder"]),
        ),
    )
    return {
        "scope": "option_mode_reposition_rows_with_multiple_legal_moves",
        "eligible_label_count": len(reposition_rows),
        "best_decoder": {
            "decoder": best["decoder"],
            "accuracy": best["accuracy"],
            "correct_count": best["correct_count"],
            "eligible_label_count": best["eligible_label_count"],
        },
        "decoders": decoder_reports,
    }


def _current_value_predictions(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> list[dict[str, object]]:
    predictions = []
    examples = _training_examples(rows, objective_name="option_mode")
    for row in rows:
        seed = row.get("seed")
        training = [example for example in examples if example.get("seed") != seed]
        predicted_scores = _predicted_action_scores(
            row,
            training,
            k=k,
            allowed_actions=set(_move_candidates(row)),
        )
        if not predicted_scores:
            continue
        predicted = max(
            predicted_scores,
            key=lambda action: (predicted_scores[action], action),
        )
        predictions.append(_prediction_payload(row, predicted, f"current_value_k{k}"))
    return predictions


def _state_classifier_predictions(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
    variant: str,
) -> list[dict[str, object]]:
    predictions = []
    training_rows = [
        row
        for row in rows
        if _state_feature_vector(row, variant=variant)
        and str(row.get("option_mode_representative_action", "")) in _MOVE_DELTAS
    ]
    for row in rows:
        seed = row.get("seed")
        features = _state_feature_vector(row, variant=variant)
        candidates = set(_move_candidates(row))
        if not features or not candidates:
            continue
        neighbors = []
        for other in training_rows:
            if other.get("seed") == seed:
                continue
            action = str(other.get("option_mode_representative_action", ""))
            if action not in candidates:
                continue
            other_features = _state_feature_vector(other, variant=variant)
            if not other_features:
                continue
            neighbors.append((_squared_distance(features, other_features), action))
        if not neighbors:
            continue
        neighbors.sort(key=lambda item: (item[0], item[1]))
        selected = neighbors[: max(1, int(k))]
        predicted = _majority_action(selected)
        predictions.append(_prediction_payload(row, predicted, f"{variant}_k{k}"))
    return predictions


def _navigation_baseline_predictions(
    rows: Sequence[Mapping[str, object]],
    *,
    target: str,
) -> list[dict[str, object]]:
    predictions = []
    for row in rows:
        candidates = _move_candidates(row)
        if not candidates:
            continue
        resolved_target = _navigation_target_for_row(row, target)
        predicted = max(
            candidates,
            key=lambda action: (_navigation_alignment(row, action, resolved_target), action),
        )
        predictions.append(
            _prediction_payload(
                row,
                predicted,
                f"navigation_baseline_toward_{target}",
            )
        )
    return predictions


def _summarize_predictions(
    decoder: str,
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    count = len(predictions)
    correct = sum(1 for item in predictions if item.get("correct") is True)
    predicted_counts = Counter(str(item.get("predicted_action")) for item in predictions)
    dominant_action, dominant_count = _dominant_count(predicted_counts)
    return {
        "decoder": decoder,
        "eligible_label_count": count,
        "correct_count": correct,
        "accuracy": _safe_rate(correct, count),
        "prediction_action_counts": dict(sorted(predicted_counts.items())),
        "dominant_prediction_action": dominant_action,
        "dominant_prediction_action_count": dominant_count,
        "dominant_prediction_action_share": _safe_rate(dominant_count, count),
        "confusion_matrix": _confusion_matrix(predictions),
        "accuracy_by_seed": _accuracy_by(predictions, "seed"),
        "accuracy_by_tick_bin": _accuracy_by(predictions, "tick_bin"),
        "accuracy_by_energy_bin": _accuracy_by(predictions, "energy_bin"),
        "accuracy_by_hydration_bin": _accuracy_by(predictions, "hydration_bin"),
        "accuracy_by_water_distance": _accuracy_by(predictions, "water_distance_bin"),
        "accuracy_by_carrion_distance": _accuracy_by(
            predictions,
            "carrion_distance_bin",
        ),
        "accuracy_by_legal_move_count": _accuracy_by(
            predictions,
            "legal_move_count",
        ),
        "accuracy_by_ambiguity_class": _accuracy_by(
            predictions,
            "ambiguity_class",
        ),
        "error_taxonomy": dict(
            sorted(Counter(str(item.get("error_type")) for item in predictions).items())
        ),
        "examples": [dict(item) for item in predictions[:16]],
    }


def _branch_utility_report(
    rows: Sequence[Mapping[str, object]],
    *,
    option_mode_k: int,
) -> dict[str, object]:
    predictions = _option_mode_predictions(rows, k=option_mode_k)
    comparisons = []
    unsupported = 0
    missing = 0
    predicted_actions = Counter()
    predicted_modes = Counter()
    for prediction in predictions:
        row = _mapping(prediction.get("row"))
        predicted_action = str(prediction.get("predicted_action", ""))
        predicted_actions[predicted_action] += 1
        predicted_modes[_action_option_mode(predicted_action)] += 1
        if predicted_action not in ACTION_NAMES:
            unsupported += 1
            continue
        action_values = _list_of_mappings(row.get("action_values"), "action_values")
        predicted_run = _action_value(action_values, predicted_action)
        logged_run = _action_value(action_values, str(row.get("logged_action", "")))
        if predicted_run is None or logged_run is None:
            missing += 1
            continue
        comparisons.append(
            _utility_comparison(
                row,
                predicted_action=predicted_action,
                predicted_run=predicted_run,
                logged_run=logged_run,
            )
        )
    dominant_mode, dominant_mode_count = _dominant_count(predicted_modes)
    dominant_action, dominant_action_count = _dominant_count(predicted_actions)
    return {
        "policy": "leave_one_source_seed_out_option_mode_predictions_replayed_vs_logged_v1",
        "option_mode_nearest_neighbor_k": int(option_mode_k),
        "prediction_count": len(predictions),
        "comparison_count": len(comparisons),
        "missing_branch_outcome_count": int(missing),
        "unsupported_predicted_action_count": int(unsupported),
        "predicted_action_counts": dict(sorted(predicted_actions.items())),
        "predicted_mode_counts": dict(sorted(predicted_modes.items())),
        "dominant_predicted_action": dominant_action,
        "dominant_predicted_action_count": dominant_action_count,
        "dominant_predicted_action_share": _safe_rate(
            dominant_action_count,
            len(predictions),
        ),
        "dominant_predicted_mode": dominant_mode,
        "dominant_predicted_mode_count": dominant_mode_count,
        "dominant_predicted_mode_share": _safe_rate(
            dominant_mode_count,
            len(predictions),
        ),
        "target_local_score_delta_summary": _field_summary(
            comparisons,
            "target_local_score_delta",
        ),
        "terminal_alive_delta_summary": _field_summary(
            comparisons,
            "terminal_alive_delta",
        ),
        "birth_delta_summary": _field_summary(comparisons, "birth_delta"),
        "target_alive_delta_summary": _field_summary(
            comparisons,
            "target_alive_delta",
        ),
        "target_energy_delta_summary": _field_summary(
            comparisons,
            "target_energy_delta",
        ),
        "target_hydration_delta_summary": _field_summary(
            comparisons,
            "target_hydration_delta",
        ),
        "target_health_delta_summary": _field_summary(
            comparisons,
            "target_health_delta",
        ),
        "target_resource_gain_delta_summary": _field_summary(
            comparisons,
            "target_resource_gain_delta",
        ),
        "action_family_confusion_count": sum(
            1 for item in comparisons if item["predicted_mode"] != item["target_mode"]
        ),
        "examples": comparisons[:16],
    }


def _option_mode_predictions(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> list[dict[str, object]]:
    predictions = []
    examples = _training_examples(rows, objective_name="option_mode")
    for row in rows:
        seed = row.get("seed")
        training = [example for example in examples if example.get("seed") != seed]
        predicted_scores = _predicted_action_scores(row, training, k=k)
        if not predicted_scores:
            continue
        predicted_action, predicted_mode = _predicted_option_mode(predicted_scores)
        predictions.append(
            {
                "branch_id": row.get("branch_id"),
                "seed": seed,
                "target_action": row.get("option_mode_representative_action"),
                "target_mode": row.get("option_mode"),
                "predicted_action": predicted_action,
                "predicted_mode": predicted_mode,
                "row": row,
            }
        )
    return predictions


def _utility_comparison(
    row: Mapping[str, object],
    *,
    predicted_action: str,
    predicted_run: Mapping[str, object],
    logged_run: Mapping[str, object],
) -> dict[str, object]:
    predicted_target = _target_terminal_projection(predicted_run)
    logged_target = _target_terminal_projection(logged_run)
    predicted_first = _mapping(predicted_run.get("first_action_outcome"))
    logged_first = _mapping(logged_run.get("first_action_outcome"))
    return {
        "branch_id": row.get("branch_id"),
        "seed": row.get("seed"),
        "logged_action": row.get("logged_action"),
        "predicted_action": predicted_action,
        "target_action": row.get("option_mode_representative_action"),
        "target_mode": row.get("option_mode"),
        "predicted_mode": _action_option_mode(predicted_action),
        "target_local_score_delta": _round(
            _target_local_scalar(predicted_run, row)
            - _target_local_scalar(logged_run, row)
        ),
        "terminal_alive_delta": float(
            _int(predicted_run.get("terminal_alive_agents"))
            - _int(logged_run.get("terminal_alive_agents"))
        ),
        "birth_delta": float(
            _int(predicted_run.get("births")) - _int(logged_run.get("births"))
        ),
        "target_alive_delta": float(
            int(predicted_target.get("alive") is True)
            - int(logged_target.get("alive") is True)
        ),
        "target_energy_delta": _round(
            _float(predicted_target.get("energy_ratio"))
            - _float(logged_target.get("energy_ratio"))
        ),
        "target_hydration_delta": _round(
            _float(predicted_target.get("hydration_ratio"))
            - _float(logged_target.get("hydration_ratio"))
        ),
        "target_health_delta": _round(
            _float(predicted_target.get("health_ratio"))
            - _float(logged_target.get("health_ratio"))
        ),
        "target_resource_gain_delta": _round(
            _float(predicted_first.get("resource_gain"))
            - _float(logged_first.get("resource_gain"))
        ),
    }


def _acceptance(
    *,
    coverage: Mapping[str, object],
    mode_objective: Mapping[str, object],
    decomposition: Mapping[str, object],
    utility: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    mode_support = _mapping(mode_objective.get("support"))
    option = _mapping(mode_support.get("option_mode"))
    material = _mapping(mode_support.get("material_only_option_mode"))
    best_reposition = _mapping(decomposition.get("best_decoder"))
    utility_score = _mapping(utility.get("target_local_score_delta_summary"))
    terminal_alive = _mapping(utility.get("terminal_alive_delta_summary"))
    births = _mapping(utility.get("birth_delta_summary"))
    if coverage.get("replay_verified_all_labels") is not True:
        blockers.append({"reason": "replay_verified_all_labels_false"})
    if int(coverage.get("heuristic_action_source_count", 0)) != 0:
        blockers.append(
            {
                "reason": "heuristic_action_source_count_nonzero",
                "observed": int(coverage.get("heuristic_action_source_count", 0)),
            }
        )
    if int(coverage.get("unsupported_oracle_action_count", 0)) != 0:
        blockers.append(
            {
                "reason": "unsupported_oracle_action_count_nonzero",
                "observed": int(coverage.get("unsupported_oracle_action_count", 0)),
            }
        )
    if (
        int(coverage.get("label_count", 0)) < V91_MIN_LABEL_COUNT
        and coverage.get("insufficient_label_rows_proven") is not True
    ):
        blockers.append(
            {
                "reason": "label_count_below_floor",
                "observed": int(coverage.get("label_count", 0)),
                "required_min": V91_MIN_LABEL_COUNT,
            }
        )
    if (
        int(coverage.get("reposition_multi_move_label_count", 0))
        < V91_MIN_REPOSITION_MULTI_MOVE_LABEL_COUNT
        and coverage.get("insufficient_reposition_rows_proven") is not True
    ):
        blockers.append(
            {
                "reason": "reposition_multi_move_label_count_below_floor",
                "observed": int(coverage.get("reposition_multi_move_label_count", 0)),
                "required_min": V91_MIN_REPOSITION_MULTI_MOVE_LABEL_COUNT,
            }
        )
    if _float(option.get("best_mode_accuracy")) < V91_OPTION_MODE_ACCURACY_FLOOR:
        blockers.append(
            {
                "reason": "option_mode_accuracy_below_floor",
                "observed": option.get("best_mode_accuracy"),
                "required_min": V91_OPTION_MODE_ACCURACY_FLOOR,
            }
        )
    if (
        _float(material.get("best_mode_accuracy"))
        < V91_MATERIAL_ONLY_OPTION_MODE_ACCURACY_FLOOR
    ):
        blockers.append(
            {
                "reason": "material_only_option_mode_accuracy_below_floor",
                "observed": material.get("best_mode_accuracy"),
                "required_min": V91_MATERIAL_ONLY_OPTION_MODE_ACCURACY_FLOOR,
            }
        )
    if (
        _float(option.get("dominant_prediction_mode_share"))
        > V91_MAX_DOMINANT_OPTION_MODE_SHARE
    ):
        blockers.append(
            {
                "reason": "dominant_predicted_option_mode_share_above_cap",
                "observed": option.get("dominant_prediction_mode_share"),
                "required_max": V91_MAX_DOMINANT_OPTION_MODE_SHARE,
            }
        )
    if _float(best_reposition.get("accuracy")) < V91_REPOSITION_DIRECTION_ACCURACY_FLOOR:
        blockers.append(
            {
                "reason": "reposition_exact_direction_accuracy_below_floor",
                "observed": best_reposition.get("accuracy"),
                "required_min": V91_REPOSITION_DIRECTION_ACCURACY_FLOOR,
                "best_decoder": best_reposition.get("decoder"),
            }
        )
    if _float(utility_score.get("mean")) <= V91_MIN_TARGET_LOCAL_SCORE_DELTA:
        blockers.append(
            {
                "reason": "predicted_branch_target_local_score_delta_not_positive",
                "observed": utility_score.get("mean"),
                "required_gt": V91_MIN_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    if _float(terminal_alive.get("mean")) < V91_MIN_MEAN_TERMINAL_ALIVE_DELTA:
        blockers.append(
            {
                "reason": "predicted_branch_terminal_alive_delta_negative",
                "observed": terminal_alive.get("mean"),
                "required_min": V91_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            }
        )
    if _float(births.get("mean")) < V91_MIN_MEAN_BIRTH_DELTA:
        blockers.append(
            {
                "reason": "predicted_branch_birth_delta_negative",
                "observed": births.get("mean"),
                "required_min": V91_MIN_MEAN_BIRTH_DELTA,
            }
        )
    accepted = not blockers
    return {
        "v91_reposition_frontier_diagnostic_accepted": accepted,
        "v92_runtime_work_allowed": accepted,
        "runtime_policy_trained": False,
        "blockers": blockers,
    }


def _prediction_payload(
    row: Mapping[str, object],
    predicted_action: str,
    decoder: str,
) -> dict[str, object]:
    target = str(row.get("option_mode_representative_action", ""))
    bins = _row_bins(row)
    return {
        "branch_id": row.get("branch_id"),
        "seed": row.get("seed"),
        "decoder": decoder,
        "target_action": target,
        "predicted_action": predicted_action,
        "correct": predicted_action == target,
        "error_type": _move_error_type(target, predicted_action),
        **bins,
    }


def _row_bins(row: Mapping[str, object]) -> dict[str, object]:
    source = _mapping(row.get("source"))
    before = _mapping(row.get("before"))
    state = _mapping(row.get("compact_state"))
    navigation = _mapping(state.get("navigation"))
    return {
        "tick_bin": _tick_bin(_int(source.get("branch_tick"))),
        "energy_bin": _ratio_bin(before.get("energy_ratio")),
        "hydration_bin": _ratio_bin(before.get("hydration_ratio")),
        "water_distance_bin": _distance_bin(
            _navigation_distance(navigation, "water")
        ),
        "carrion_distance_bin": _distance_bin(
            _navigation_distance(navigation, "carrion")
        ),
        "legal_move_count": str(len(_move_candidates(row))),
        "ambiguity_class": _ambiguity_class(row),
    }


def _state_feature_vector(
    row: Mapping[str, object],
    *,
    variant: str,
) -> tuple[float, ...]:
    if variant == "direct":
        values = row.get("observation_values")
        if not isinstance(values, tuple):
            return ()
        mask = _mapping(row.get("action_mask"))
        return tuple(float(value) for value in values) + tuple(
            1.0 if bool(mask.get(action, False)) else 0.0 for action in ACTION_NAMES
        )
    state = _mapping(row.get("compact_state"))
    self_state = _mapping(state.get("self"))
    local = _mapping(state.get("local"))
    center = _mapping(state.get("center"))
    navigation = _mapping(state.get("navigation"))
    if not self_state:
        return ()
    features = [
        _float(self_state.get("energy_ratio")),
        _float(self_state.get("hydration_ratio")),
        _float(self_state.get("health_ratio")),
        1.0 - _float(self_state.get("energy_ratio")),
        1.0 - _float(self_state.get("hydration_ratio")),
        1.0 - _float(self_state.get("health_ratio")),
        _float(self_state.get("reproduction_ready")),
        _float(self_state.get("sexual_reproduction_unlocked")),
        _float(center.get("water")),
        _float(center.get("food")),
        _float(center.get("carcass")),
        _float(local.get("radius1_water")),
        _float(local.get("radius1_food")),
        _float(local.get("radius1_carrion")),
        _float(local.get("radius2_water")),
        _float(local.get("radius2_food")),
        _float(local.get("radius2_carrion")),
    ]
    for target in ("water", "plant", "carrion", "prey"):
        item = _mapping(navigation.get(target))
        features.extend(
            [
                _sign(_float(item.get("dx"))),
                _sign(_float(item.get("dy"))),
                _distance_bin_numeric(_float(item.get("distance"))),
                _float(item.get("strength")),
            ]
        )
    mask = _mapping(row.get("action_mask"))
    features.extend(1.0 if bool(mask.get(action, False)) else 0.0 for action in _MOVE_DELTAS)
    return tuple(_round(value) for value in features)


def _navigation_target_for_row(row: Mapping[str, object], target: str) -> str:
    if target != "resource_need":
        return target
    before = _mapping(row.get("before"))
    energy_debt = 1.0 - _float(before.get("energy_ratio"))
    hydration_debt = 1.0 - _float(before.get("hydration_ratio"))
    if hydration_debt > energy_debt:
        return "water"
    navigation = _mapping(_mapping(row.get("compact_state")).get("navigation"))
    carrion = _mapping(navigation.get("carrion"))
    return "carrion" if _float(carrion.get("strength")) > 0.0 else "plant"


def _navigation_alignment(
    row: Mapping[str, object],
    action: str,
    target: str,
) -> float:
    dx, dy = _MOVE_DELTAS.get(action, (0, 0))
    navigation = _mapping(_mapping(row.get("compact_state")).get("navigation"))
    item = _mapping(navigation.get(target))
    return _round(
        (float(dx) * _float(item.get("dx")) + float(dy) * _float(item.get("dy")))
        * max(0.1, _float(item.get("strength")))
    )


def _target_local_scalar(
    candidate: Mapping[str, object],
    row: Mapping[str, object],
) -> float:
    score = _target_local_score(candidate, row)
    weights = (1000.0, 100.0, 25.0, 10.0, 5.0, 0.25, 0.25, 0.25, 1.0)
    return _round(
        sum(float(value) * weights[index] for index, value in enumerate(score))
    )


def _material_by_seed_mode(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("material_oracle_gain") is True:
            counts[str(row.get("seed"))][str(row.get("option_mode"))] += 1
    return {seed: dict(sorted(value.items())) for seed, value in sorted(counts.items())}


def _frontier_source_counts(
    audit: Mapping[str, object] | None,
) -> dict[str, object]:
    if audit is None:
        return {"available": False}
    discovery = _list_of_mappings(audit.get("discovery"), "discovery")
    return {
        "available": True,
        "branch_selection_policy": _mapping(audit.get("contract")).get(
            "branch_selection_policy"
        ),
        "eligible_row_count": sum(_int(item.get("eligible_row_count")) for item in discovery),
        "eligible_multi_move_reposition_row_count": sum(
            _int(item.get("eligible_multi_move_reposition_row_count"))
            for item in discovery
        ),
        "selected_multi_move_reposition_row_count": sum(
            _int(item.get("selected_multi_move_reposition_row_count"))
            for item in discovery
        ),
        "by_seed": [
            {
                "seed": item.get("seed"),
                "eligible_row_count": item.get("eligible_row_count"),
                "eligible_multi_move_reposition_row_count": item.get(
                    "eligible_multi_move_reposition_row_count"
                ),
                "selected_multi_move_reposition_row_count": item.get(
                    "selected_multi_move_reposition_row_count"
                ),
                "selected_failure_frontier_score_summary": item.get(
                    "selected_failure_frontier_score_summary"
                ),
            }
            for item in discovery
        ],
    }


def _reposition_multi_move_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if str(row.get("option_mode")) == "reposition" and len(_move_candidates(row)) > 1
    ]


def _move_candidates(row: Mapping[str, object]) -> list[str]:
    return [
        str(candidate.get("action", ""))
        for candidate in _list_of_mappings(row.get("action_values"), "action_values")
        if str(candidate.get("action", "")) in _MOVE_DELTAS
    ]


def _action_value(
    action_values: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next(
        (item for item in action_values if str(item.get("action", "")) == action),
        None,
    )


def _majority_action(neighbors: Sequence[tuple[float, str]]) -> str:
    scores: dict[str, tuple[float, int, float]] = {}
    for distance, action in neighbors:
        weight = 1.0 / (1.0 + max(0.0, float(distance)))
        total, count, distance_sum = scores.get(action, (0.0, 0, 0.0))
        scores[action] = (total + weight, count + 1, distance_sum + float(distance))
    return max(
        scores,
        key=lambda action: (
            scores[action][0],
            scores[action][1],
            -scores[action][2],
            action,
        ),
    )


def _confusion_matrix(
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for item in predictions:
        matrix[str(item.get("target_action"))][str(item.get("predicted_action"))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(matrix.items())}


def _accuracy_by(
    predictions: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in predictions:
        groups[str(item.get(field))].append(item)
    return {
        key: {
            "count": len(items),
            "correct_count": sum(1 for item in items if item.get("correct") is True),
            "accuracy": _safe_rate(
                sum(1 for item in items if item.get("correct") is True),
                len(items),
            ),
        }
        for key, items in sorted(groups.items())
    }


def _move_error_type(target: str, predicted: str) -> str:
    if target == predicted:
        return "correct"
    if target not in _MOVE_DELTAS or predicted not in _MOVE_DELTAS:
        return "action_family_confusion"
    tx, ty = _MOVE_DELTAS[target]
    px, py = _MOVE_DELTAS[predicted]
    if (tx != 0 and px != 0) or (ty != 0 and py != 0):
        return "opposite_axis"
    return "wrong_axis"


def _ambiguity_class(row: Mapping[str, object]) -> str:
    action_values = _list_of_mappings(row.get("action_values"), "action_values")
    modes = {_action_option_mode(str(item.get("action", ""))) for item in action_values}
    moves = len(_move_candidates(row))
    if moves >= 3 and len(modes) >= 3:
        return "three_plus_moves_cross_mode"
    if moves >= 3:
        return "three_plus_moves"
    if len(modes) >= 3:
        return "two_moves_cross_mode"
    return "two_moves"


def _navigation_distance(
    navigation: Mapping[str, object],
    target: str,
) -> float | None:
    item = _mapping(navigation.get(target))
    value = item.get("distance")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _field_summary(
    values: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, object]:
    series = [_float(item.get(field)) for item in values]
    if not series:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(series),
        "mean": _round(sum(series) / float(len(series))),
        "min": _round(min(series)),
        "max": _round(max(series)),
        "positive_count": sum(1 for value in series if value > 0),
        "negative_count": sum(1 for value in series if value < 0),
        "zero_count": sum(1 for value in series if value == 0),
    }


def _dominant_count(counts: Counter[str]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    item = max(sorted(counts.items()), key=lambda value: (value[1], value[0]))
    return item[0], int(item[1])


def _ratio_bin(value: object) -> str:
    ratio = _float(value)
    if ratio < 0.25:
        return "low"
    if ratio < 0.55:
        return "mid"
    return "high"


def _distance_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 0.25:
        return "near"
    if value <= 0.6:
        return "mid"
    return "far"


def _distance_bin_numeric(value: float) -> float:
    if value <= 0.25:
        return 0.0
    if value <= 0.6:
        return 0.5
    return 1.0


def _tick_bin(tick: int) -> str:
    return f"{(int(tick) // 10) * 10:03d}-{(int(tick) // 10) * 10 + 9:03d}"


def _sign(value: float) -> float:
    if value > 0.05:
        return 1.0
    if value < -0.05:
        return -1.0
    return 0.0


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
    return float(value)


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
