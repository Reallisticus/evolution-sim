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
    _branch_continuation_archive_feature_vector,
    _compact_world_model_rows,
    _objective_tuple,
    _squared_distance,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_BRANCH_MODE_OBJECTIVE_AUDIT_SCHEMA_VERSION = (
    "mind_v3_branch_mode_objective_audit_v1"
)
OPTION_MODE_SUPPORT_ACCURACY_FLOOR = 0.60
OPTION_MODE_MAX_DOMINANT_PREDICTION_SHARE = 0.75
MATERIAL_ONLY_OPTION_MODE_SUPPORT_ACCURACY_FLOOR = 0.55
REPOSITION_MULTI_MOVE_EXACT_DIRECTION_ACCURACY_FLOOR = 0.55


class BranchModeObjectiveAuditError(ValueError):
    pass


def build_branch_mode_objective_audit_report(
    branch_action_oracle_labels: Mapping[str, object],
    *,
    source_branch_action_oracle_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if (
        branch_action_oracle_labels.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION
    ):
        raise BranchModeObjectiveAuditError(
            "branch action oracle labels have unsupported schema_version"
        )
    if (
        source_branch_action_oracle_audit is not None
        and source_branch_action_oracle_audit.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION
    ):
        raise BranchModeObjectiveAuditError(
            "source branch action oracle audit has unsupported schema_version"
        )
    labels = _list_of_mappings(branch_action_oracle_labels.get("labels"), "labels")
    objective_rows = _objective_rows(labels)
    coverage = _coverage_report(
        objective_rows,
        source_branch_action_oracle_audit=source_branch_action_oracle_audit,
    )
    comparison = _objective_comparison(objective_rows)
    support = {
        "population_first": _objective_support_probe(
            objective_rows,
            objective_name="population_first",
            material_only=False,
        ),
        "target_local": _objective_support_probe(
            objective_rows,
            objective_name="target_local",
            material_only=False,
        ),
        "option_mode": _objective_support_probe(
            objective_rows,
            objective_name="option_mode",
            material_only=False,
        ),
        "material_only_option_mode": _objective_support_probe(
            objective_rows,
            objective_name="option_mode",
            material_only=True,
        ),
        "reposition_exact_direction": _reposition_direction_support_probe(
            objective_rows
        ),
    }
    acceptance = _acceptance(coverage, support)
    contract = {
        "schema_version": MIND_V3_BRANCH_MODE_OBJECTIVE_AUDIT_SCHEMA_VERSION,
        "source_label_schema_version": branch_action_oracle_labels.get(
            "schema_version"
        ),
        "source_audit_schema_version": (
            source_branch_action_oracle_audit.get("schema_version")
            if source_branch_action_oracle_audit is not None
            else None
        ),
        "runtime_policy_trained": False,
        "support_floors": {
            "option_mode_accuracy": OPTION_MODE_SUPPORT_ACCURACY_FLOOR,
            "max_dominant_predicted_option_mode": (
                OPTION_MODE_MAX_DOMINANT_PREDICTION_SHARE
            ),
            "material_only_option_mode_accuracy": (
                MATERIAL_ONLY_OPTION_MODE_SUPPORT_ACCURACY_FLOOR
            ),
            "reposition_multi_move_exact_direction_accuracy": (
                REPOSITION_MULTI_MOVE_EXACT_DIRECTION_ACCURACY_FLOOR
            ),
            "unsupported_oracle_actions": 0,
            "heuristic_runtime_action_sources": 0,
        },
    }
    return {
        "schema_version": MIND_V3_BRANCH_MODE_OBJECTIVE_AUDIT_SCHEMA_VERSION,
        "contract": contract,
        "provenance": {
            "source_label_digest": stable_payload_digest(
                {
                    "schema_version": branch_action_oracle_labels.get(
                        "schema_version"
                    ),
                    "contract": branch_action_oracle_labels.get("contract"),
                    "aggregate": branch_action_oracle_labels.get("aggregate"),
                    "acceptance": branch_action_oracle_labels.get("acceptance"),
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
                        "acceptance": source_branch_action_oracle_audit.get(
                            "acceptance"
                        ),
                    }
                )
                if source_branch_action_oracle_audit is not None
                else None
            ),
            "contract_digest": stable_payload_digest(contract),
        },
        "coverage": coverage,
        "objective_comparison": comparison,
        "support": support,
        "acceptance": acceptance,
    }


def load_json_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    try:
        with _open_input(resolved) as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise BranchModeObjectiveAuditError(f"failed to read report: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise BranchModeObjectiveAuditError(
            f"report is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BranchModeObjectiveAuditError("report must be a JSON object")
    return payload


def write_branch_mode_objective_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _objective_rows(labels: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    compact_rows = _compact_world_model_rows(labels)
    rows = []
    label_by_id = {str(label.get("branch_id", "")): label for label in labels}
    for compact in compact_rows:
        branch_id = str(compact.get("branch_id", ""))
        label = _mapping(label_by_id.get(branch_id))
        oracle = _mapping(label.get("oracle_label"))
        action_values = _list_of_mappings(compact.get("action_values"), "action_values")
        population_action = str(oracle.get("action", ""))
        target_action = _best_action_by_score(
            action_values,
            score_fn=lambda candidate: _target_local_score(candidate, label),
        )
        option_mode, option_action = _best_option_mode(action_values, label)
        logged_action = str(oracle.get("logged_action", ""))
        rows.append(
            {
                **dict(compact),
                "logged_action": logged_action,
                "population_first_action": population_action,
                "population_first_mode": _action_option_mode(population_action),
                "target_local_action": target_action,
                "target_local_mode": _action_option_mode(target_action),
                "option_mode": option_mode,
                "option_mode_representative_action": option_action,
                "material_oracle_gain": bool(oracle.get("material_oracle_gain")),
                "source": dict(_mapping(label.get("source"))),
                "before": dict(_mapping(label.get("before"))),
                "oracle_label": dict(oracle),
                "quality": dict(_mapping(label.get("quality"))),
            }
        )
    return rows


def _coverage_report(
    rows: Sequence[Mapping[str, object]],
    *,
    source_branch_action_oracle_audit: Mapping[str, object] | None,
) -> dict[str, object]:
    by_seed = Counter(str(row.get("seed")) for row in rows)
    by_action = Counter(str(row.get("population_first_action")) for row in rows)
    by_mode = Counter(str(row.get("population_first_mode")) for row in rows)
    material_by_seed_mode: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    branch_ticks = []
    unsupported = 0
    zero_heuristic_count = 0
    population_deltas = []
    target_deltas = []
    for row in rows:
        source = _mapping(row.get("source"))
        seed = str(row.get("seed"))
        mode = str(row.get("population_first_mode"))
        if row.get("material_oracle_gain") is True:
            material_by_seed_mode[seed][mode] += 1
        logged = str(row.get("logged_action"))
        oracle = str(row.get("population_first_action"))
        matrix[logged][oracle] += 1
        branch_ticks.append(_int(source.get("branch_tick")))
        if oracle not in ACTION_NAMES:
            unsupported += 1
        if _mapping(row.get("quality")).get("zero_heuristic_runtime_actions") is True:
            zero_heuristic_count += 1
        action_values = _list_of_mappings(row.get("action_values"), "action_values")
        logged_run = _action_value(action_values, logged)
        oracle_run = _action_value(action_values, oracle)
        if logged_run is not None and oracle_run is not None:
            population_deltas.append(_population_delta(oracle_run, logged_run))
            target_deltas.append(_target_delta(oracle_run, logged_run))
    return {
        "label_count": len(rows),
        "labels_by_seed": dict(sorted(by_seed.items())),
        "labels_by_oracle_action": dict(sorted(by_action.items())),
        "labels_by_oracle_option_mode": dict(sorted(by_mode.items())),
        "material_labels_by_seed_mode": {
            seed: dict(sorted(counts.items()))
            for seed, counts in sorted(material_by_seed_mode.items())
        },
        "branch_tick_distribution": _tick_distribution(branch_ticks),
        "logged_action_to_oracle_action_matrix": {
            logged: dict(sorted(counts.items()))
            for logged, counts in sorted(matrix.items())
        },
        "target_delta_summary_vs_logged": _delta_summary(target_deltas),
        "population_delta_summary_vs_logged": _population_delta_summary(
            population_deltas
        ),
        "unsupported_oracle_action_count": unsupported,
        "zero_heuristic_label_count": zero_heuristic_count,
        "zero_heuristic_all_labels": zero_heuristic_count == len(rows),
        "branch_point_selection": _branch_selection_report(
            source_branch_action_oracle_audit
        ),
    }


def _objective_comparison(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    exact_population_target_agree = 0
    mode_population_target_agree = 0
    mode_population_option_agree = 0
    mode_target_option_agree = 0
    target_action_counts = Counter()
    target_mode_counts = Counter()
    option_mode_counts = Counter()
    examples = []
    for row in rows:
        population_action = str(row.get("population_first_action", ""))
        target_action = str(row.get("target_local_action", ""))
        population_mode = str(row.get("population_first_mode", ""))
        target_mode = str(row.get("target_local_mode", ""))
        option_mode = str(row.get("option_mode", ""))
        exact_population_target_agree += int(population_action == target_action)
        mode_population_target_agree += int(population_mode == target_mode)
        mode_population_option_agree += int(population_mode == option_mode)
        mode_target_option_agree += int(target_mode == option_mode)
        target_action_counts[target_action] += 1
        target_mode_counts[target_mode] += 1
        option_mode_counts[option_mode] += 1
        if population_action != target_action or population_mode != option_mode:
            examples.append(
                {
                    "branch_id": row.get("branch_id"),
                    "seed": row.get("seed"),
                    "logged_action": row.get("logged_action"),
                    "population_first_action": population_action,
                    "target_local_action": target_action,
                    "population_first_mode": population_mode,
                    "target_local_mode": target_mode,
                    "option_mode": option_mode,
                }
            )
    count = len(rows)
    return {
        "population_first_vs_target_local_exact_agreement": _safe_rate(
            exact_population_target_agree,
            count,
        ),
        "population_first_vs_target_local_mode_agreement": _safe_rate(
            mode_population_target_agree,
            count,
        ),
        "population_first_vs_option_mode_agreement": _safe_rate(
            mode_population_option_agree,
            count,
        ),
        "target_local_vs_option_mode_agreement": _safe_rate(
            mode_target_option_agree,
            count,
        ),
        "target_local_action_counts": dict(sorted(target_action_counts.items())),
        "target_local_mode_counts": dict(sorted(target_mode_counts.items())),
        "option_mode_counts": dict(sorted(option_mode_counts.items())),
        "examples": examples[:12],
    }


def _objective_support_probe(
    rows: Sequence[Mapping[str, object]],
    *,
    objective_name: str,
    material_only: bool,
) -> dict[str, object]:
    selected_rows = [
        row for row in rows if not material_only or row.get("material_oracle_gain") is True
    ]
    results = [
        _objective_support_result(selected_rows, objective_name=objective_name, k=k)
        for k in (1, 3, 5)
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["mode_accuracy"]),
            int(result["mode_correct_count"]),
            float(result["exact_action_accuracy"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    return {
        "policy": "leave_one_source_seed_out_mode_balanced_objective_scorer_v1",
        "objective": objective_name,
        "scope": "material_oracle_gain_labels_only" if material_only else "all_labels",
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_mode_accuracy": best["mode_accuracy"],
        "best_mode_correct_count": best["mode_correct_count"],
        "best_exact_action_accuracy": best["exact_action_accuracy"],
        "dominant_prediction_mode": best["dominant_prediction_mode"],
        "dominant_prediction_mode_share": best["dominant_prediction_mode_share"],
        "results": results,
    }


def _objective_support_result(
    rows: Sequence[Mapping[str, object]],
    *,
    objective_name: str,
    k: int,
) -> dict[str, object]:
    predictions = []
    examples = _training_examples(rows, objective_name=objective_name)
    for row in rows:
        seed = row.get("seed")
        training = [example for example in examples if example.get("seed") != seed]
        if not training:
            continue
        predicted_scores = _predicted_action_scores(row, training, k=k)
        if not predicted_scores:
            continue
        if objective_name == "option_mode":
            predicted_action, predicted_mode = _predicted_option_mode(
                predicted_scores
            )
        else:
            predicted_action = max(
                predicted_scores,
                key=lambda action: (predicted_scores[action], action),
            )
            predicted_mode = _action_option_mode(predicted_action)
        target_action = _target_action(row, objective_name)
        target_mode = _target_mode(row, objective_name)
        predictions.append(
            {
                "branch_id": row.get("branch_id"),
                "seed": seed,
                "target_action": target_action,
                "target_mode": target_mode,
                "predicted_action": predicted_action,
                "predicted_mode": predicted_mode,
                "exact_action_correct": predicted_action == target_action,
                "mode_correct": predicted_mode == target_mode,
            }
        )
    exact_correct = sum(
        1 for item in predictions if item["exact_action_correct"] is True
    )
    mode_correct = sum(1 for item in predictions if item["mode_correct"] is True)
    mode_counts = Counter(str(item["predicted_mode"]) for item in predictions)
    dominant_mode, dominant_count = _dominant_count(mode_counts)
    count = len(predictions)
    return {
        "nearest_neighbor_k": int(k),
        "eligible_label_count": count,
        "exact_action_correct_count": exact_correct,
        "exact_action_accuracy": _safe_rate(exact_correct, count),
        "mode_correct_count": mode_correct,
        "mode_accuracy": _safe_rate(mode_correct, count),
        "prediction_mode_counts": dict(sorted(mode_counts.items())),
        "dominant_prediction_mode": dominant_mode,
        "dominant_prediction_mode_count": dominant_count,
        "dominant_prediction_mode_share": _safe_rate(dominant_count, count),
        "examples": predictions[:12],
    }


def _reposition_direction_support_probe(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    reposition_rows = [
        row
        for row in rows
        if str(row.get("option_mode")) == "reposition"
        and len(_move_candidates(row)) > 1
    ]
    results = [
        _reposition_direction_result(reposition_rows, k=k)
        for k in (1, 3, 5)
    ]
    best = max(
        results,
        key=lambda result: (
            float(result["accuracy"]),
            int(result["correct_count"]),
            -int(result["nearest_neighbor_k"]),
        ),
    )
    return {
        "policy": "leave_one_source_seed_out_reposition_exact_direction_scorer_v1",
        "scope": "option_mode_reposition_rows_with_multiple_legal_moves",
        "best_nearest_neighbor_k": best["nearest_neighbor_k"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "eligible_label_count": best["eligible_label_count"],
        "results": results,
    }


def _reposition_direction_result(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, object]:
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
        target = str(row.get("option_mode_representative_action", ""))
        predictions.append(
            {
                "branch_id": row.get("branch_id"),
                "seed": seed,
                "target_action": target,
                "predicted_action": predicted,
                "correct": predicted == target,
                "move_candidate_count": len(_move_candidates(row)),
            }
        )
    correct = sum(1 for item in predictions if item["correct"] is True)
    return {
        "nearest_neighbor_k": int(k),
        "eligible_label_count": len(predictions),
        "correct_count": correct,
        "accuracy": _safe_rate(correct, len(predictions)),
        "examples": predictions[:12],
    }


def _acceptance(
    coverage: Mapping[str, object],
    support: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    option = _mapping(support.get("option_mode"))
    material = _mapping(support.get("material_only_option_mode"))
    reposition = _mapping(support.get("reposition_exact_direction"))
    if _float(option.get("best_mode_accuracy")) < OPTION_MODE_SUPPORT_ACCURACY_FLOOR:
        blockers.append(
            {
                "reason": "option_mode_accuracy_below_floor",
                "observed": option.get("best_mode_accuracy"),
                "required_min": OPTION_MODE_SUPPORT_ACCURACY_FLOOR,
            }
        )
    if (
        _float(option.get("dominant_prediction_mode_share"))
        > OPTION_MODE_MAX_DOMINANT_PREDICTION_SHARE
    ):
        blockers.append(
            {
                "reason": "dominant_predicted_option_mode_share_above_cap",
                "observed": option.get("dominant_prediction_mode_share"),
                "required_max": OPTION_MODE_MAX_DOMINANT_PREDICTION_SHARE,
            }
        )
    if (
        _float(material.get("best_mode_accuracy"))
        < MATERIAL_ONLY_OPTION_MODE_SUPPORT_ACCURACY_FLOOR
    ):
        blockers.append(
            {
                "reason": "material_only_option_mode_accuracy_below_floor",
                "observed": material.get("best_mode_accuracy"),
                "required_min": MATERIAL_ONLY_OPTION_MODE_SUPPORT_ACCURACY_FLOOR,
            }
        )
    if (
        _float(reposition.get("best_accuracy"))
        < REPOSITION_MULTI_MOVE_EXACT_DIRECTION_ACCURACY_FLOOR
    ):
        blockers.append(
            {
                "reason": "reposition_exact_direction_accuracy_below_floor",
                "observed": reposition.get("best_accuracy"),
                "required_min": REPOSITION_MULTI_MOVE_EXACT_DIRECTION_ACCURACY_FLOOR,
            }
        )
    if _int(coverage.get("unsupported_oracle_action_count")) != 0:
        blockers.append(
            {
                "reason": "unsupported_oracle_actions_present",
                "observed": coverage.get("unsupported_oracle_action_count"),
            }
        )
    if coverage.get("zero_heuristic_all_labels") is not True:
        blockers.append({"reason": "heuristic_runtime_action_source_present"})
    return {
        "mode_balanced_objective_diagnostic_passed": not blockers,
        "runtime_training_allowed": not blockers,
        "blockers": blockers,
    }


def _training_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    objective_name: str,
) -> list[dict[str, object]]:
    examples = []
    for row in rows:
        for candidate in _list_of_mappings(row.get("action_values"), "action_values"):
            action = str(candidate.get("action", ""))
            features = _branch_continuation_archive_feature_vector(
                row=row,
                action=action,
            )
            if not features:
                continue
            examples.append(
                {
                    "seed": row.get("seed"),
                    "action": action,
                    "features": features,
                    "score": _score(candidate, row, objective_name),
                }
            )
    return examples


def _predicted_action_scores(
    row: Mapping[str, object],
    training: Sequence[Mapping[str, object]],
    *,
    k: int,
    allowed_actions: set[str] | None = None,
) -> dict[str, tuple[float, ...]]:
    predicted: dict[str, tuple[float, ...]] = {}
    for candidate in _list_of_mappings(row.get("action_values"), "action_values"):
        action = str(candidate.get("action", ""))
        if allowed_actions is not None and action not in allowed_actions:
            continue
        features = _branch_continuation_archive_feature_vector(row=row, action=action)
        if not features:
            continue
        neighbors = [
            (
                _squared_distance(features, example["features"]),  # type: ignore[arg-type]
                example["score"],  # type: ignore[index]
            )
            for example in training
            if example.get("action") == action
        ]
        if not neighbors:
            continue
        neighbors.sort(key=lambda item: item[0])
        selected = neighbors[: max(1, int(k))]
        predicted[action] = _mean_score(
            [score for _, score in selected if isinstance(score, tuple)]
        )
    return predicted


def _score(
    candidate: Mapping[str, object],
    row: Mapping[str, object],
    objective_name: str,
) -> tuple[float, ...]:
    if objective_name == "population_first":
        return _objective_tuple(candidate.get("objective_tuple"))
    return _target_local_score(candidate, row)


def _target_action(row: Mapping[str, object], objective_name: str) -> str:
    if objective_name == "population_first":
        return str(row.get("population_first_action", ""))
    if objective_name == "target_local":
        return str(row.get("target_local_action", ""))
    return str(row.get("option_mode_representative_action", ""))


def _target_mode(row: Mapping[str, object], objective_name: str) -> str:
    if objective_name == "population_first":
        return str(row.get("population_first_mode", ""))
    if objective_name == "target_local":
        return str(row.get("target_local_mode", ""))
    return str(row.get("option_mode", ""))


def _target_local_score(
    candidate: Mapping[str, object],
    row: Mapping[str, object],
) -> tuple[float, ...]:
    before = _mapping(row.get("before"))
    terminal = _target_terminal_projection(candidate)
    trace = _list_of_mappings(candidate.get("target_horizon_trace"), "target_horizon_trace")
    found = [item for item in trace if item.get("record_found") is True]
    target_alive_area = _safe_rate(
        sum(1 for item in found if item.get("alive_after") is True),
        len(found),
    )
    resource_gain = sum(_float(item.get("resource_gain")) for item in found)
    start_energy = _float(before.get("energy_ratio"))
    start_hydration = _float(before.get("hydration_ratio"))
    start_health = _float(before.get("health_ratio"))
    energy = _float(terminal.get("energy_ratio"))
    hydration = _float(terminal.get("hydration_ratio"))
    health = _float(terminal.get("health_ratio"))
    need_recovery = (
        (1.0 - start_energy) * (energy - start_energy)
        + (1.0 - start_hydration) * (hydration - start_hydration)
        + (1.0 - start_health) * (health - start_health)
        + resource_gain
    )
    return (
        1.0 if terminal.get("alive") is True else 0.0,
        _round(target_alive_area),
        _round(min(energy, hydration, health)),
        _round(need_recovery),
        _round(resource_gain),
        float(_int(candidate.get("terminal_alive_agents"))),
        float(_int(candidate.get("births"))),
        -float(_int(candidate.get("deaths"))),
        -_float(candidate.get("dominant_requested_action_share")),
    )


def _target_terminal_projection(candidate: Mapping[str, object]) -> dict[str, object]:
    population_trace = _list_of_mappings(
        candidate.get("population_horizon_trace"),
        "population_horizon_trace",
    )
    if population_trace:
        latest = max(population_trace, key=lambda item: _int(item.get("horizon_tick_delta")))
        return {
            "alive": bool(latest.get("target_alive", False)),
            "energy_ratio": latest.get("target_energy_ratio"),
            "hydration_ratio": latest.get("target_hydration_ratio"),
            "health_ratio": latest.get("target_health_ratio"),
        }
    target_trace = [
        item
        for item in _list_of_mappings(
            candidate.get("target_horizon_trace"),
            "target_horizon_trace",
        )
        if item.get("record_found") is True
    ]
    if target_trace:
        latest = max(target_trace, key=lambda item: _int(item.get("horizon_tick_delta")))
        return {
            "alive": bool(latest.get("alive_after", False)),
            "energy_ratio": latest.get("energy_ratio_after"),
            "hydration_ratio": latest.get("hydration_ratio_after"),
            "health_ratio": latest.get("health_ratio_after"),
        }
    return {
        "alive": bool(candidate.get("target_alive_at_end", False)),
        "energy_ratio": None,
        "hydration_ratio": None,
        "health_ratio": None,
    }


def _best_action_by_score(
    action_values: Sequence[Mapping[str, object]],
    *,
    score_fn: object,
) -> str:
    if not action_values:
        return ""
    scored = [
        (score_fn(candidate), str(candidate.get("action", "")))  # type: ignore[misc]
        for candidate in action_values
    ]
    return max(scored, key=lambda item: (item[0], item[1]))[1]


def _best_option_mode(
    action_values: Sequence[Mapping[str, object]],
    row: Mapping[str, object],
) -> tuple[str, str]:
    best_by_mode: dict[str, tuple[tuple[float, ...], str]] = {}
    for candidate in action_values:
        action = str(candidate.get("action", ""))
        mode = _action_option_mode(action)
        score = _target_local_score(candidate, row)
        if mode not in best_by_mode or (score, action) > best_by_mode[mode]:
            best_by_mode[mode] = (score, action)
    if not best_by_mode:
        return "", ""
    mode = max(best_by_mode, key=lambda item: (best_by_mode[item][0], item))
    return mode, best_by_mode[mode][1]


def _predicted_option_mode(
    predicted_scores: Mapping[str, tuple[float, ...]],
) -> tuple[str, str]:
    best_by_mode: dict[str, tuple[tuple[float, ...], str]] = {}
    for action, score in predicted_scores.items():
        mode = _action_option_mode(action)
        if mode not in best_by_mode or (score, action) > best_by_mode[mode]:
            best_by_mode[mode] = (score, action)
    mode = max(best_by_mode, key=lambda item: (best_by_mode[item][0], item))
    return best_by_mode[mode][1], mode


def _move_candidates(row: Mapping[str, object]) -> list[str]:
    return [
        str(candidate.get("action", ""))
        for candidate in _list_of_mappings(row.get("action_values"), "action_values")
        if str(candidate.get("action", "")).startswith("move_")
    ]


def _population_delta(
    oracle: Mapping[str, object],
    logged: Mapping[str, object],
) -> dict[str, float]:
    return {
        "alive_delta": float(_int(oracle.get("terminal_alive_agents")) - _int(logged.get("terminal_alive_agents"))),
        "birth_delta": float(_int(oracle.get("births")) - _int(logged.get("births"))),
    }


def _target_delta(
    oracle: Mapping[str, object],
    logged: Mapping[str, object],
) -> dict[str, float]:
    left = _target_terminal_projection(oracle)
    right = _target_terminal_projection(logged)
    return {
        "target_alive_delta": float(int(left.get("alive") is True) - int(right.get("alive") is True)),
        "target_energy_delta": _float(left.get("energy_ratio")) - _float(right.get("energy_ratio")),
        "target_hydration_delta": _float(left.get("hydration_ratio")) - _float(right.get("hydration_ratio")),
        "target_health_delta": _float(left.get("health_ratio")) - _float(right.get("health_ratio")),
    }


def _delta_summary(values: Sequence[Mapping[str, float]]) -> dict[str, object]:
    fields = (
        "target_alive_delta",
        "target_energy_delta",
        "target_hydration_delta",
        "target_health_delta",
    )
    return {field: _field_summary(values, field) for field in fields}


def _population_delta_summary(values: Sequence[Mapping[str, float]]) -> dict[str, object]:
    return {
        "alive_delta": _field_summary(values, "alive_delta"),
        "birth_delta": _field_summary(values, "birth_delta"),
    }


def _field_summary(values: Sequence[Mapping[str, float]], field: str) -> dict[str, object]:
    series = [float(item.get(field, 0.0)) for item in values]
    if not series:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(series),
        "mean": _round(sum(series) / float(len(series))),
        "min": _round(min(series)),
        "max": _round(max(series)),
        "positive_count": sum(1 for item in series if item > 0),
        "negative_count": sum(1 for item in series if item < 0),
        "zero_count": sum(1 for item in series if item == 0),
    }


def _tick_distribution(ticks: Sequence[int]) -> dict[str, object]:
    if not ticks:
        return {"count": 0, "min": None, "max": None, "bins": {}}
    bins = Counter(f"{(tick // 10) * 10:03d}-{(tick // 10) * 10 + 9:03d}" for tick in ticks)
    return {
        "count": len(ticks),
        "min": min(ticks),
        "max": max(ticks),
        "bins": dict(sorted(bins.items())),
    }


def _branch_selection_report(
    audit: Mapping[str, object] | None,
) -> dict[str, object]:
    if audit is None:
        return {"available": False}
    discovery = _list_of_mappings(audit.get("discovery"), "discovery")
    skipped_total: Counter[str] = Counter()
    selected_buckets: Counter[str] = Counter()
    for item in discovery:
        skipped_total.update(_mapping(item.get("skipped_row_counts")))
        selected_buckets.update(_mapping(item.get("selected_bucket_counts")))
    return {
        "available": True,
        "contract": dict(_mapping(audit.get("contract"))),
        "discovery": [dict(item) for item in discovery],
        "skipped_row_counts_total": dict(sorted(skipped_total.items())),
        "selected_bucket_counts_total": dict(sorted(selected_buckets.items())),
    }


def _action_value(
    action_values: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next(
        (item for item in action_values if str(item.get("action", "")) == action),
        None,
    )


def _mean_score(values: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    if not values:
        return (0.0,)
    width = len(values[0])
    aligned = [value for value in values if len(value) == width]
    if not aligned:
        return (0.0,) * width
    count = float(len(aligned))
    return tuple(sum(value[index] for value in aligned) / count for index in range(width))


def _dominant_count(counts: Counter[str]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    item = max(sorted(counts.items()), key=lambda value: (value[1], value[0]))
    return item[0], int(item[1])


def _action_option_mode(action: str) -> str:
    if action == "drink":
        return "recover_hydration"
    if action == "eat":
        return "exploit_resource"
    if action == "stay":
        return "conserve"
    if action.startswith("move_"):
        return "reposition"
    return "other"


def _list_of_mappings(value: object, field: str) -> list[Mapping[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise BranchModeObjectiveAuditError(f"{field} must be a list")
    items = []
    for item in value:
        if not isinstance(item, Mapping):
            raise BranchModeObjectiveAuditError(f"{field} entries must be objects")
        items.append(item)
    return items


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


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


def _safe_rate(count: int, total: int) -> float:
    return _round(count / float(total)) if total else 0.0


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
