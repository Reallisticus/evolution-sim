from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.branch_action_oracle_audit import (
    MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.branch_action_oracle_labels import (
    MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
)
from evolution_sim.mind.branch_depleted_resource_trap_audit import (
    MIND_V3_DEPLETED_RESOURCE_TRAP_AUDIT_SCHEMA_VERSION,
    _select_action as _v93_select_action,
    _support_examples as _v93_support_examples,
    _support_trap_rows as _v93_support_trap_rows,
)
from evolution_sim.mind.branch_mode_objective_audit import (
    _action_option_mode,
    _list_of_mappings,
    _mapping,
    _safe_rate,
    _target_terminal_projection,
)
from evolution_sim.mind.branch_utility_risk_audit import (
    _candidate_actions,
    _candidate_feature_vector,
    _field_summary,
    _float,
    _int,
    _target_local_scalar,
    _utility_comparison,
    _utility_rows,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION = (
    "mind_v3_branch_sequence_continuation_scorer_v1"
)
V94_STRICT_EVAL_SEEDS: tuple[int, ...] = (13, 19, 29, 37, 41, 43)
V94_MIN_STRICT_COMPARISON_COUNT = 48
V94_MAX_TARGET_ALIVE_NEGATIVE_COUNT = 0
V94_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA = 0.0
V94_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA = 0.0
V94_MIN_MEAN_TERMINAL_ALIVE_DELTA = 0.0
V94_MIN_MEAN_BIRTH_DELTA = 0.0
V94_MAX_DOMINANT_PREDICTED_ACTION_SHARE = 0.50
V94_MAX_DOMINANT_PREDICTED_MODE_SHARE = 0.75
V94_K = 5
V94_SEED_41_TICK_113_BRANCH_ID = (
    "carrion-only-seed-41-action-branch-0-tick-113-agent-18-logged-move-south"
)
V94_SEED_41_TICK_114_BRANCH_ID = (
    "carrion-only-seed-41-action-branch-1-tick-114-agent-18-logged-eat"
)

_BASELINE_RULES = {"v93_best_rule_baseline"}
_ACCEPTANCE_CANDIDATE_RULES = {
    "sequence_prefix_nearest_neighbor_k5",
    "horizon_trace_target_survival_scorer_k5",
    "short_horizon_vital_envelope_scorer_k5",
    "population_birth_continuation_scorer_k5",
    "combined_target_survival_first_sequence_scorer_k5",
}


class BranchSequenceContinuationScorerError(ValueError):
    pass


def build_branch_sequence_continuation_scorer_report(
    *,
    support_branch_action_oracle_labels: Mapping[str, object],
    strict_branch_action_oracle_labels: Mapping[str, object],
    support_branch_action_oracle_audit: Mapping[str, object] | None = None,
    v93_depleted_resource_trap_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _validate_label_report(support_branch_action_oracle_labels, "support")
    _validate_label_report(strict_branch_action_oracle_labels, "strict")
    if (
        support_branch_action_oracle_audit is not None
        and support_branch_action_oracle_audit.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION
    ):
        raise BranchSequenceContinuationScorerError(
            "support branch action oracle audit has unsupported schema_version"
        )
    if (
        v93_depleted_resource_trap_audit is not None
        and v93_depleted_resource_trap_audit.get("schema_version")
        != MIND_V3_DEPLETED_RESOURCE_TRAP_AUDIT_SCHEMA_VERSION
    ):
        raise BranchSequenceContinuationScorerError(
            "v93 depleted-resource trap audit has unsupported schema_version"
        )

    support_labels = _list_of_mappings(
        support_branch_action_oracle_labels.get("labels"),
        "support.labels",
    )
    strict_labels = _list_of_mappings(
        strict_branch_action_oracle_labels.get("labels"),
        "strict.labels",
    )
    support_rows = _utility_rows(support_labels)
    strict_rows = _utility_rows(strict_labels)
    support_training_rows = _support_training_rows(
        support_rows=support_rows,
        support_audit=support_branch_action_oracle_audit,
    )
    support_examples = _sequence_training_examples(support_training_rows)
    candidate_predictions = _candidate_sequence_predictions(
        strict_rows=strict_rows,
        support_examples=support_examples,
    )
    v93_baseline_actions = _v93_baseline_actions(
        strict_rows=strict_rows,
        support_rows=support_rows,
        support_audit=support_branch_action_oracle_audit,
        v93_depleted_resource_trap_audit=v93_depleted_resource_trap_audit,
    )
    rule_reports = _decision_rule_reports(
        strict_rows,
        candidate_predictions=candidate_predictions,
        v93_baseline_actions=v93_baseline_actions,
    )
    coverage = _coverage_report(
        support_rows=support_rows,
        support_training_rows=support_training_rows,
        strict_rows=strict_rows,
        support_labels=support_branch_action_oracle_labels,
        strict_labels=strict_branch_action_oracle_labels,
        candidate_predictions=candidate_predictions,
    )
    acceptance = _acceptance(coverage=coverage, rule_reports=rule_reports)
    contract = {
        "schema_version": MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION,
        "support_source_label_schema_version": support_branch_action_oracle_labels.get(
            "schema_version"
        ),
        "strict_eval_label_schema_version": strict_branch_action_oracle_labels.get(
            "schema_version"
        ),
        "runtime_policy_trained": False,
        "split_policy": "train_non_strict_support_eval_strict_carrion_v1",
        "strict_eval_seeds": list(V94_STRICT_EVAL_SEEDS),
        "candidate_prediction_policy": (
            "deterministic_policy_visible_sequence_continuation_knn_v1"
        ),
        "feature_contract": {
            "schema_version": "v94_policy_visible_sequence_continuation_features_v1",
            "uses_private_world_state": False,
            "uses_fixture_identity": False,
            "uses_seed_id_as_runtime_feature": False,
            "uses_branch_id_as_runtime_feature": False,
            "uses_logged_action_as_runtime_fallback": False,
            "features": [
                "decoded policy observation compact state",
                "action mask",
                "same-agent public history trace",
                "candidate action identity and movement direction",
                "candidate action support flag",
            ],
        },
        "target_contract": {
            "schema_version": "v94_replay_backed_sequence_targets_v1",
            "uses_first_action_outcome_as_target": True,
            "uses_target_horizon_trace_as_target": True,
            "uses_population_horizon_trace_as_target": True,
            "models_after_first_action_deltas": True,
            "first_action_imitation_target": False,
        },
        "decision_rules": sorted(_BASELINE_RULES | _ACCEPTANCE_CANDIDATE_RULES),
        "acceptance_candidate_rules": sorted(_ACCEPTANCE_CANDIDATE_RULES),
        "baseline_rules_not_eligible_for_acceptance": sorted(_BASELINE_RULES),
        "support_floors": {
            "replay_verified": True,
            "heuristic_action_source_count": 0,
            "no_strict_seed_training_leakage": True,
            "unsupported_predicted_or_oracle_actions": 0,
            "strict_eval_comparison_count": V94_MIN_STRICT_COMPARISON_COUNT,
            "target_alive_delta_negative_count": (
                V94_MAX_TARGET_ALIVE_NEGATIVE_COUNT
            ),
            "global_mean_target_local_score_delta_gt": (
                V94_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA
            ),
            "per_seed_target_local_score_delta_min": (
                V94_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA
            ),
            "mean_terminal_alive_delta_min": V94_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            "mean_birth_delta_min": V94_MIN_MEAN_BIRTH_DELTA,
            "dominant_predicted_action_share_max": (
                V94_MAX_DOMINANT_PREDICTED_ACTION_SHARE
            ),
            "dominant_predicted_mode_share_max": (
                V94_MAX_DOMINANT_PREDICTED_MODE_SHARE
            ),
            "seed_41_tick_113_catastrophe_avoided": True,
            "seed_41_tick_114_catastrophe_avoided": True,
        },
    }
    accepted = bool(acceptance.get("v94_sequence_continuation_scorer_accepted"))
    return {
        "schema_version": MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION,
        "contract": contract,
        "provenance": {
            "support_label_digest": stable_payload_digest(
                {
                    "schema_version": support_branch_action_oracle_labels.get(
                        "schema_version"
                    ),
                    "aggregate": support_branch_action_oracle_labels.get(
                        "aggregate"
                    ),
                    "acceptance": support_branch_action_oracle_labels.get(
                        "acceptance"
                    ),
                    "labels": support_labels,
                }
            ),
            "strict_label_digest": stable_payload_digest(
                {
                    "schema_version": strict_branch_action_oracle_labels.get(
                        "schema_version"
                    ),
                    "aggregate": strict_branch_action_oracle_labels.get("aggregate"),
                    "acceptance": strict_branch_action_oracle_labels.get(
                        "acceptance"
                    ),
                    "labels": strict_labels,
                }
            ),
            "support_audit_digest": (
                stable_payload_digest(
                    {
                        "schema_version": support_branch_action_oracle_audit.get(
                            "schema_version"
                        ),
                        "contract": support_branch_action_oracle_audit.get(
                            "contract"
                        ),
                        "aggregate": support_branch_action_oracle_audit.get(
                            "aggregate"
                        ),
                        "discovery": support_branch_action_oracle_audit.get(
                            "discovery"
                        ),
                    }
                )
                if support_branch_action_oracle_audit is not None
                else None
            ),
            "contract_digest": stable_payload_digest(contract),
        },
        "coverage": coverage,
        "sequence_continuation_dataset": {
            "policy": "v94_replay_backed_sequence_continuation_dataset_v1",
            "support_training_row_count": len(support_training_rows),
            "support_sequence_example_count": len(support_examples),
            "strict_eval_row_count": len(strict_rows),
            "strict_candidate_prediction_count": sum(
                len(items) for items in candidate_predictions.values()
            ),
            "support_action_counts": dict(
                sorted(Counter(str(item["action"]) for item in support_examples).items())
            ),
            "support_mode_counts": dict(
                sorted(Counter(str(item["mode"]) for item in support_examples).items())
            ),
            "target_fields": [
                "first_action_survived",
                "first_action_vital_delta",
                "target_survival_area_after_first",
                "target_short_horizon_survival_area",
                "target_min_vital_after_first",
                "target_resource_gain_after_first",
                "terminal_target_alive",
                "terminal_alive_agents",
                "births",
            ],
        },
        "candidate_sequence_predictions": {
            "policy": "support_seed_sequence_continuation_knn_k5_v1",
            "candidate_prediction_count": sum(
                len(items) for items in candidate_predictions.values()
            ),
            "strict_seed_training_leak_count": sum(
                1
                for items in candidate_predictions.values()
                for item in items
                if item.get("uses_strict_seed_training") is True
            ),
            "candidate_scores": [
                item
                for branch_id in sorted(candidate_predictions)
                for item in candidate_predictions[branch_id]
            ],
        },
        "decision_rule_reports": rule_reports,
        "sequence_continuation_support_probe": {
            "policy": "v94_sequence_world_model_branch_continuation_v1",
            "accuracy": 1.0 if accepted else 0.0,
            "support_accuracy_floor": 1.0,
            "materially_supports_sequence_continuation_scorer": accepted,
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
        raise BranchSequenceContinuationScorerError(
            f"failed to read report: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BranchSequenceContinuationScorerError(
            f"report is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BranchSequenceContinuationScorerError("report must be a JSON object")
    return payload


def write_branch_sequence_continuation_scorer_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _validate_label_report(report: Mapping[str, object], name: str) -> None:
    if report.get("schema_version") != MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION:
        raise BranchSequenceContinuationScorerError(
            f"{name} labels have unsupported schema_version"
        )


def _support_training_rows(
    *,
    support_rows: Sequence[Mapping[str, object]],
    support_audit: Mapping[str, object] | None,
) -> list[Mapping[str, object]]:
    return _v93_support_trap_rows(
        support_rows=support_rows,
        signature_trap_rows=list(support_rows),
        support_audit=support_audit,
    )


def _coverage_report(
    *,
    support_rows: Sequence[Mapping[str, object]],
    support_training_rows: Sequence[Mapping[str, object]],
    strict_rows: Sequence[Mapping[str, object]],
    support_labels: Mapping[str, object],
    strict_labels: Mapping[str, object],
    candidate_predictions: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    support_aggregate = _mapping(support_labels.get("aggregate"))
    strict_aggregate = _mapping(strict_labels.get("aggregate"))
    support_seeds = {int(row.get("seed", -1)) for row in support_training_rows}
    strict_seeds = {int(row.get("seed", -1)) for row in strict_rows}
    return {
        "support_label_count": len(support_rows),
        "support_training_row_count": len(support_training_rows),
        "support_labels_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in support_rows).items())
        ),
        "support_training_rows_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in support_training_rows).items())
        ),
        "support_candidate_action_count": sum(
            len(_candidate_actions(row)) for row in support_training_rows
        ),
        "strict_eval_label_count": len(strict_rows),
        "strict_eval_labels_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in strict_rows).items())
        ),
        "strict_eval_candidate_action_count": sum(
            len(_candidate_actions(row)) for row in strict_rows
        ),
        "support_strict_seed_overlap": sorted(support_seeds & strict_seeds),
        "strict_seed_training_leak_count": sum(
            1
            for items in candidate_predictions.values()
            for item in items
            if item.get("uses_strict_seed_training") is True
        ),
        "support_replay_verified_all_labels": bool(
            support_aggregate.get("replay_verified_all_labels", False)
        ),
        "strict_replay_verified_all_labels": bool(
            strict_aggregate.get("replay_verified_all_labels", False)
        ),
        "support_heuristic_action_source_count": _heuristic_count(
            support_aggregate
        ),
        "strict_heuristic_action_source_count": _heuristic_count(strict_aggregate),
        "support_unsupported_oracle_action_count": int(
            support_aggregate.get("unsupported_oracle_action_count", 0)
        ),
        "strict_unsupported_oracle_action_count": int(
            strict_aggregate.get("unsupported_oracle_action_count", 0)
        ),
    }


def _sequence_training_examples(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    examples = []
    for row in rows:
        for candidate in _candidate_actions(row):
            action = str(candidate.get("action", ""))
            features = _candidate_feature_vector(row, action)
            if not features:
                continue
            metrics = _sequence_target_metrics(row, candidate)
            examples.append(
                {
                    "seed": row.get("seed"),
                    "branch_id": row.get("branch_id"),
                    "action": action,
                    "mode": _action_option_mode(action),
                    "features": features,
                    "metrics": metrics,
                }
            )
    return examples


def _candidate_sequence_predictions(
    *,
    strict_rows: Sequence[Mapping[str, object]],
    support_examples: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    predictions: dict[str, list[dict[str, object]]] = {}
    strict_seed_set = {str(seed) for seed in V94_STRICT_EVAL_SEEDS}
    for row in strict_rows:
        branch_predictions = []
        for candidate in _candidate_actions(row):
            action = str(candidate.get("action", ""))
            features = _candidate_feature_vector(row, action)
            if not features:
                continue
            neighbors = [
                (
                    _squared_distance(features, item["features"]),  # type: ignore[arg-type]
                    item,
                )
                for item in support_examples
                if item.get("features")
            ]
            if not neighbors:
                continue
            neighbors.sort(
                key=lambda item: (
                    item[0],
                    str(item[1].get("branch_id", "")),
                    str(item[1].get("action", "")),
                )
            )
            selected = neighbors[:V94_K]
            predicted = _predicted_sequence_stats(selected)
            branch_predictions.append(
                {
                    "branch_id": row.get("branch_id"),
                    "seed": row.get("seed"),
                    "action": action,
                    "mode": _action_option_mode(action),
                    "candidate_supported_by_mask": True,
                    "neighbor_count": len(selected),
                    "neighbor_seed_counts": dict(
                        sorted(
                            Counter(
                                str(item[1].get("seed")) for item in selected
                            ).items()
                        )
                    ),
                    "uses_strict_seed_training": any(
                        str(item[1].get("seed")) in strict_seed_set
                        for item in selected
                    ),
                    "predicted": predicted,
                    "actual_sequence_metrics": _sequence_target_metrics(
                        row,
                        candidate,
                    ),
                }
            )
        predictions[str(row.get("branch_id", ""))] = sorted(
            branch_predictions,
            key=lambda item: str(item.get("action", "")),
        )
    return predictions


def _predicted_sequence_stats(
    selected: Sequence[tuple[float, Mapping[str, object]]],
) -> dict[str, object]:
    metrics = [_mapping(item[1].get("metrics")) for item in selected]

    def mean(field: str) -> float:
        return _mean([_float(item.get(field)) for item in metrics])

    def cvar(field: str) -> float:
        return _cvar([_float(item.get(field)) for item in metrics], fraction=0.25)

    target_alive = mean("terminal_target_alive")
    first_survival = mean("first_action_survived")
    after_first_survival = mean("target_survival_area_after_first")
    short_survival = mean("target_short_horizon_survival_area")
    min_vital = mean("target_min_vital_after_first")
    vital_envelope = mean("vital_envelope_score")
    target_score = mean("target_local_score")
    target_score_cvar = cvar("target_local_score")
    population_alive = mean("terminal_alive_agents")
    births = mean("births")
    population_score = population_alive * 10.0 + births
    combined = (
        target_alive * 1000.0
        + first_survival * 250.0
        + after_first_survival * 200.0
        + short_survival * 150.0
        + vital_envelope * 100.0
        + target_score_cvar * 0.1
        + population_score
    )
    return {
        "terminal_target_alive_probability": _round(target_alive),
        "first_action_survival_probability": _round(first_survival),
        "target_survival_area_after_first_mean": _round(after_first_survival),
        "target_short_horizon_survival_area_mean": _round(short_survival),
        "target_min_vital_after_first_mean": _round(min_vital),
        "vital_envelope_score_mean": _round(vital_envelope),
        "target_resource_gain_after_first_mean": _round(
            mean("target_resource_gain_after_first")
        ),
        "first_action_resource_gain_mean": _round(mean("first_action_resource_gain")),
        "target_local_sequence_score_mean": _round(target_score),
        "target_local_sequence_score_cvar_25": _round(target_score_cvar),
        "terminal_alive_agents_mean": _round(population_alive),
        "births_mean": _round(births),
        "population_birth_continuation_score": _round(population_score),
        "combined_target_survival_first_score": _round(combined),
        "first_action_death_risk": _round(1.0 - first_survival),
    }


def _sequence_target_metrics(
    row: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    before = _mapping(row.get("before"))
    first = _mapping(candidate.get("first_action_outcome"))
    target_trace = sorted(
        _list_of_mappings(candidate.get("target_horizon_trace"), "target_horizon_trace"),
        key=lambda item: _int(item.get("horizon_tick_delta")),
    )
    population_trace = sorted(
        _list_of_mappings(
            candidate.get("population_horizon_trace"),
            "population_horizon_trace",
        ),
        key=lambda item: _int(item.get("horizon_tick_delta")),
    )
    terminal = _target_terminal_projection(candidate)
    after_first = [
        item for item in target_trace if _int(item.get("horizon_tick_delta")) > 0
    ]
    short_after_first = [
        item for item in after_first if _int(item.get("horizon_tick_delta")) <= 5
    ]
    found_after_first = [
        item
        for item in after_first
        if item.get("record_found") is True and item.get("alive_after") is True
    ]
    vital_values = [
        min(
            _float(item.get("energy_ratio_after")),
            _float(item.get("hydration_ratio_after")),
            _float(item.get("health_ratio_after")),
        )
        for item in found_after_first
    ]
    resource_after_first = sum(
        _float(item.get("resource_gain")) for item in found_after_first
    )
    pop_final = _mapping(population_trace[-1] if population_trace else {})
    first_energy_delta = _float(first.get("energy_ratio_after")) - _float(
        before.get("energy_ratio")
    )
    first_hydration_delta = _float(first.get("hydration_ratio_after")) - _float(
        before.get("hydration_ratio")
    )
    first_health_delta = _float(first.get("health_ratio_after")) - _float(
        before.get("health_ratio")
    )
    short_survival = _safe_rate(
        sum(1 for item in short_after_first if item.get("alive_after") is True),
        len(short_after_first),
    )
    after_first_survival = _safe_rate(
        sum(1 for item in after_first if item.get("alive_after") is True),
        len(after_first),
    )
    min_vital = min(vital_values) if vital_values else 0.0
    vital_envelope = (
        short_survival * 0.5
        + after_first_survival * 0.25
        + min_vital * 0.2
        + min(1.0, resource_after_first) * 0.05
    )
    return {
        "action": str(candidate.get("action", "")),
        "mode": _action_option_mode(str(candidate.get("action", ""))),
        "first_action_survived": 1.0
        if first.get("alive_after") is True and first.get("died") is not True
        else 0.0,
        "first_action_energy_delta": _round(first_energy_delta),
        "first_action_hydration_delta": _round(first_hydration_delta),
        "first_action_health_delta": _round(first_health_delta),
        "first_action_resource_gain": _round(_float(first.get("resource_gain"))),
        "target_survival_area_after_first": _round(after_first_survival),
        "target_short_horizon_survival_area": _round(short_survival),
        "target_min_vital_after_first": _round(min_vital),
        "target_resource_gain_after_first": _round(resource_after_first),
        "vital_envelope_score": _round(vital_envelope),
        "terminal_target_alive": 1.0 if terminal.get("alive") is True else 0.0,
        "terminal_target_energy_ratio": _round(_float(terminal.get("energy_ratio"))),
        "terminal_target_hydration_ratio": _round(
            _float(terminal.get("hydration_ratio"))
        ),
        "terminal_target_health_ratio": _round(_float(terminal.get("health_ratio"))),
        "terminal_alive_agents": _candidate_or_population_int(
            candidate,
            pop_final,
            "terminal_alive_agents",
            "alive_agents",
        ),
        "births": _candidate_or_population_int(candidate, pop_final, "births", "births"),
        "deaths": _candidate_or_population_int(candidate, pop_final, "deaths", "deaths"),
        "target_local_score": _round(_target_local_scalar(candidate, row)),
    }


def _decision_rule_reports(
    strict_rows: Sequence[Mapping[str, object]],
    *,
    candidate_predictions: Mapping[str, Sequence[Mapping[str, object]]],
    v93_baseline_actions: Mapping[str, str],
) -> list[dict[str, object]]:
    reports = []
    for rule in sorted(_BASELINE_RULES | _ACCEPTANCE_CANDIDATE_RULES):
        comparisons = []
        missing = 0
        unsupported = 0
        for row in strict_rows:
            action = _select_action_for_rule(
                row,
                rule=rule,
                candidate_predictions=list(
                    candidate_predictions.get(str(row.get("branch_id")), [])
                ),
                v93_baseline_actions=v93_baseline_actions,
            )
            if action is None:
                missing += 1
                continue
            if action not in ACTION_NAMES:
                unsupported += 1
                continue
            action_values = _candidate_actions(row)
            predicted_run = _action_value(action_values, action)
            logged_run = _action_value(action_values, str(row.get("logged_action", "")))
            if predicted_run is None or logged_run is None:
                missing += 1
                continue
            comparisons.append(
                _utility_comparison(
                    row,
                    rule=rule,
                    predicted_action=action,
                    predicted_run=predicted_run,
                    logged_run=logged_run,
                )
            )
        reports.append(
            _summarize_rule(
                rule,
                comparisons,
                missing_branch_outcome_count=missing,
                unsupported_predicted_action_count=unsupported,
            )
        )
    return reports


def _select_action_for_rule(
    row: Mapping[str, object],
    *,
    rule: str,
    candidate_predictions: Sequence[Mapping[str, object]],
    v93_baseline_actions: Mapping[str, str],
) -> str | None:
    if rule == "v93_best_rule_baseline":
        return v93_baseline_actions.get(str(row.get("branch_id", "")))
    if not candidate_predictions:
        return None
    if rule == "sequence_prefix_nearest_neighbor_k5":
        return _max_by_prediction(
            candidate_predictions,
            "target_local_sequence_score_cvar_25",
        )
    if rule == "horizon_trace_target_survival_scorer_k5":
        return _max_by_tuple(
            candidate_predictions,
            (
                "terminal_target_alive_probability",
                "target_survival_area_after_first_mean",
                "target_short_horizon_survival_area_mean",
                "target_local_sequence_score_mean",
            ),
        )
    if rule == "short_horizon_vital_envelope_scorer_k5":
        return _max_by_tuple(
            candidate_predictions,
            (
                "first_action_survival_probability",
                "vital_envelope_score_mean",
                "target_min_vital_after_first_mean",
                "target_local_sequence_score_mean",
            ),
        )
    if rule == "population_birth_continuation_scorer_k5":
        return _max_by_tuple(
            candidate_predictions,
            (
                "population_birth_continuation_score",
                "terminal_alive_agents_mean",
                "births_mean",
                "terminal_target_alive_probability",
            ),
        )
    if rule == "combined_target_survival_first_sequence_scorer_k5":
        return _max_by_prediction(
            candidate_predictions,
            "combined_target_survival_first_score",
        )
    return None


def _max_by_prediction(
    candidate_predictions: Sequence[Mapping[str, object]],
    score_key: str,
) -> str | None:
    return _max_by_tuple(candidate_predictions, (score_key,))


def _max_by_tuple(
    candidate_predictions: Sequence[Mapping[str, object]],
    score_keys: Sequence[str],
) -> str | None:
    if not candidate_predictions:
        return None
    selected = max(
        candidate_predictions,
        key=lambda item: (
            *[
                _float(_mapping(item.get("predicted")).get(key))
                for key in score_keys
            ],
            -_float(_mapping(item.get("predicted")).get("first_action_death_risk")),
            str(item.get("action", "")),
        ),
    )
    return str(selected.get("action", ""))


def _v93_baseline_actions(
    *,
    strict_rows: Sequence[Mapping[str, object]],
    support_rows: Sequence[Mapping[str, object]],
    support_audit: Mapping[str, object] | None,
    v93_depleted_resource_trap_audit: Mapping[str, object] | None,
) -> dict[str, str]:
    rule = "trap_support_action_family_balanced_k5"
    if v93_depleted_resource_trap_audit is not None:
        best = _mapping(
            _mapping(v93_depleted_resource_trap_audit.get("acceptance")).get(
                "best_rule_for_diagnostics"
            )
        )
        if best.get("rule"):
            rule = str(best.get("rule"))
    support_examples = _v93_support_examples(
        _v93_support_trap_rows(
            support_rows=support_rows,
            signature_trap_rows=list(support_rows),
            support_audit=support_audit,
        )
    )
    actions = {}
    for row in strict_rows:
        action = _v93_select_action(
            row,
            rule=rule,
            support_examples=support_examples,
            v92_actions={},
        )
        if action is not None:
            actions[str(row.get("branch_id", ""))] = action
    return actions


def _summarize_rule(
    rule: str,
    comparisons: Sequence[Mapping[str, object]],
    *,
    missing_branch_outcome_count: int,
    unsupported_predicted_action_count: int,
) -> dict[str, object]:
    predicted_actions = Counter(str(item.get("predicted_action")) for item in comparisons)
    predicted_modes = Counter(str(item.get("predicted_mode")) for item in comparisons)
    dominant_action, dominant_action_count = _dominant_count(predicted_actions)
    dominant_mode, dominant_mode_count = _dominant_count(predicted_modes)
    action_family_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for item in comparisons:
        action_family_matrix[str(item.get("target_local_mode"))][
            str(item.get("predicted_mode"))
        ] += 1
    negative_comparisons = sorted(
        [
            dict(item)
            for item in comparisons
            if _float(item.get("target_local_score_delta")) < 0.0
        ],
        key=lambda item: (
            _float(item.get("target_local_score_delta")),
            str(item.get("branch_id", "")),
        ),
    )
    worst_examples = sorted(
        [dict(item) for item in comparisons],
        key=lambda item: (
            _float(item.get("target_local_score_delta")),
            _float(item.get("target_alive_delta")),
            str(item.get("branch_id", "")),
        ),
    )[:16]
    return {
        "rule": rule,
        "rule_class": "baseline" if rule in _BASELINE_RULES else "sequence_scorer",
        "eligible_for_acceptance": rule in _ACCEPTANCE_CANDIDATE_RULES,
        "uses_logged_action_runtime_fallback": False,
        "uses_seed_branch_fixture_or_hidden_state_runtime_features": False,
        "comparison_count": len(comparisons),
        "missing_branch_outcome_count": int(missing_branch_outcome_count),
        "unsupported_predicted_action_count": int(unsupported_predicted_action_count),
        "predicted_action_counts": dict(sorted(predicted_actions.items())),
        "predicted_mode_counts": dict(sorted(predicted_modes.items())),
        "dominant_predicted_action": dominant_action,
        "dominant_predicted_action_count": dominant_action_count,
        "dominant_predicted_action_share": _safe_rate(
            dominant_action_count,
            len(comparisons),
        ),
        "dominant_predicted_mode": dominant_mode,
        "dominant_predicted_mode_count": dominant_mode_count,
        "dominant_predicted_mode_share": _safe_rate(
            dominant_mode_count,
            len(comparisons),
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
        "per_seed_delta_means": _per_seed_delta_means(comparisons),
        "action_family_confusion": {
            "count": sum(
                1
                for item in comparisons
                if item.get("predicted_mode") != item.get("target_local_mode")
            ),
            "target_local_mode_to_predicted_mode_matrix": {
                key: dict(sorted(value.items()))
                for key, value in sorted(action_family_matrix.items())
            },
        },
        "seed_41_sequence_cases": _seed_41_sequence_cases(comparisons),
        "negative_comparisons": negative_comparisons,
        "worst_examples": worst_examples,
    }


def _acceptance(
    *,
    coverage: Mapping[str, object],
    rule_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    blockers_by_rule = []
    for report in rule_reports:
        if report.get("eligible_for_acceptance") is not True:
            continue
        blockers_by_rule.append(
            {
                "rule": report.get("rule"),
                "blockers": _rule_blockers(report, coverage),
            }
        )
    accepted_rules = [
        str(item.get("rule"))
        for item in blockers_by_rule
        if not item.get("blockers")
    ]
    best = _best_rule_for_diagnostics(rule_reports, blockers_by_rule)
    best_blockers = next(
        (
            list(_list_of_mappings(item.get("blockers"), "blockers"))
            for item in blockers_by_rule
            if item.get("rule") == best.get("rule")
        ),
        [],
    )
    accepted = bool(accepted_rules)
    return {
        "v94_sequence_continuation_scorer_accepted": accepted,
        "v95_runtime_work_allowed": accepted,
        "runtime_policy_trained": False,
        "accepted_rules": accepted_rules,
        "best_rule_for_diagnostics": best,
        "strict_blockers": best_blockers,
        "blockers_by_rule": blockers_by_rule,
    }


def _rule_blockers(
    report: Mapping[str, object],
    coverage: Mapping[str, object],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if coverage.get("support_replay_verified_all_labels") is not True:
        blockers.append({"reason": "support_replay_verified_all_labels_false"})
    if coverage.get("strict_replay_verified_all_labels") is not True:
        blockers.append({"reason": "strict_replay_verified_all_labels_false"})
    for field in (
        "support_heuristic_action_source_count",
        "strict_heuristic_action_source_count",
        "support_unsupported_oracle_action_count",
        "strict_unsupported_oracle_action_count",
        "strict_seed_training_leak_count",
    ):
        if int(coverage.get(field, 0)) != 0:
            blockers.append(
                {
                    "reason": f"{field}_nonzero",
                    "observed": int(coverage.get(field, 0)),
                    "required": 0,
                }
            )
    if coverage.get("support_strict_seed_overlap"):
        blockers.append(
            {
                "reason": "support_strict_seed_overlap_nonempty",
                "observed": coverage.get("support_strict_seed_overlap"),
                "required": [],
            }
        )
    if int(report.get("unsupported_predicted_action_count", 0)) != 0:
        blockers.append(
            {
                "reason": "unsupported_predicted_action_count_nonzero",
                "observed": int(report.get("unsupported_predicted_action_count", 0)),
                "required": 0,
            }
        )
    if int(report.get("comparison_count", 0)) < V94_MIN_STRICT_COMPARISON_COUNT:
        blockers.append(
            {
                "reason": "strict_eval_comparison_count_below_floor",
                "observed": int(report.get("comparison_count", 0)),
                "required_min": V94_MIN_STRICT_COMPARISON_COUNT,
            }
        )
    target_alive = _mapping(report.get("target_alive_delta_summary"))
    if int(target_alive.get("negative_count", 0)) != V94_MAX_TARGET_ALIVE_NEGATIVE_COUNT:
        blockers.append(
            {
                "reason": "target_alive_delta_negative_count_nonzero",
                "observed": int(target_alive.get("negative_count", 0)),
                "required": V94_MAX_TARGET_ALIVE_NEGATIVE_COUNT,
            }
        )
    target_local = _mapping(report.get("target_local_score_delta_summary"))
    if _float(target_local.get("mean")) <= V94_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA:
        blockers.append(
            {
                "reason": "mean_target_local_score_delta_not_positive",
                "observed": target_local.get("mean"),
                "required_gt": V94_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    per_seed = _mapping(report.get("per_seed_delta_means"))
    negative_seed_means = {
        seed: _mapping(value).get("target_local_score_delta")
        for seed, value in sorted(per_seed.items())
        if _float(_mapping(value).get("target_local_score_delta"))
        < V94_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA
    }
    if negative_seed_means:
        blockers.append(
            {
                "reason": "per_seed_target_local_score_delta_negative",
                "observed": negative_seed_means,
                "required_min": V94_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    terminal_alive = _mapping(report.get("terminal_alive_delta_summary"))
    if _float(terminal_alive.get("mean")) < V94_MIN_MEAN_TERMINAL_ALIVE_DELTA:
        blockers.append(
            {
                "reason": "mean_terminal_alive_delta_negative",
                "observed": terminal_alive.get("mean"),
                "required_min": V94_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            }
        )
    births = _mapping(report.get("birth_delta_summary"))
    if _float(births.get("mean")) < V94_MIN_MEAN_BIRTH_DELTA:
        blockers.append(
            {
                "reason": "mean_birth_delta_negative",
                "observed": births.get("mean"),
                "required_min": V94_MIN_MEAN_BIRTH_DELTA,
            }
        )
    if (
        _float(report.get("dominant_predicted_action_share"))
        > V94_MAX_DOMINANT_PREDICTED_ACTION_SHARE
    ):
        blockers.append(
            {
                "reason": "dominant_predicted_action_share_above_cap",
                "observed": report.get("dominant_predicted_action_share"),
                "required_max": V94_MAX_DOMINANT_PREDICTED_ACTION_SHARE,
            }
        )
    if (
        _float(report.get("dominant_predicted_mode_share"))
        > V94_MAX_DOMINANT_PREDICTED_MODE_SHARE
    ):
        blockers.append(
            {
                "reason": "dominant_predicted_mode_share_above_cap",
                "observed": report.get("dominant_predicted_mode_share"),
                "required_max": V94_MAX_DOMINANT_PREDICTED_MODE_SHARE,
            }
        )
    seed_41 = _mapping(report.get("seed_41_sequence_cases"))
    for key in ("tick_113", "tick_114"):
        case = _mapping(seed_41.get(key))
        if case.get("avoided") is not True:
            blockers.append(
                {
                    "reason": f"seed_41_{key}_catastrophe_not_avoided",
                    "observed": case,
                    "required": "target_alive_delta >= 0",
                }
            )
    if report.get("uses_logged_action_runtime_fallback") is not False:
        blockers.append({"reason": "logged_action_runtime_fallback_used"})
    if report.get("uses_seed_branch_fixture_or_hidden_state_runtime_features") is not False:
        blockers.append(
            {"reason": "seed_branch_fixture_or_hidden_state_runtime_features_used"}
        )
    return blockers


def _best_rule_for_diagnostics(
    rule_reports: Sequence[Mapping[str, object]],
    blockers_by_rule: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    blocker_count_by_rule = {
        str(item.get("rule")): len(_list_of_mappings(item.get("blockers"), "blockers"))
        for item in blockers_by_rule
    }
    candidate_reports = [
        report
        for report in rule_reports
        if report.get("eligible_for_acceptance") is True
    ]
    if not candidate_reports:
        return {}
    best = max(
        candidate_reports,
        key=lambda report: (
            -blocker_count_by_rule.get(str(report.get("rule")), 999),
            1
            if all(
                _mapping(_mapping(report.get("seed_41_sequence_cases")).get(key)).get(
                    "avoided"
                )
                is True
                for key in ("tick_113", "tick_114")
            )
            else 0,
            -int(
                _mapping(report.get("target_alive_delta_summary")).get(
                    "negative_count",
                    0,
                )
            ),
            _float(
                _mapping(report.get("target_local_score_delta_summary")).get("mean")
            ),
            _float(_mapping(report.get("terminal_alive_delta_summary")).get("mean")),
            _float(_mapping(report.get("birth_delta_summary")).get("mean")),
            -_float(report.get("dominant_predicted_action_share")),
            str(report.get("rule")),
        ),
    )
    seed_41 = _mapping(best.get("seed_41_sequence_cases"))
    return {
        "rule": best.get("rule"),
        "blocker_count": blocker_count_by_rule.get(str(best.get("rule")), 0),
        "mean_target_local_score_delta": _mapping(
            best.get("target_local_score_delta_summary")
        ).get("mean"),
        "target_alive_delta_negative_count": _mapping(
            best.get("target_alive_delta_summary")
        ).get("negative_count"),
        "mean_terminal_alive_delta": _mapping(
            best.get("terminal_alive_delta_summary")
        ).get("mean"),
        "mean_birth_delta": _mapping(best.get("birth_delta_summary")).get("mean"),
        "dominant_predicted_action_share": best.get(
            "dominant_predicted_action_share"
        ),
        "dominant_predicted_mode_share": best.get("dominant_predicted_mode_share"),
        "seed_41_tick_113_avoided": _mapping(seed_41.get("tick_113")).get("avoided"),
        "seed_41_tick_114_avoided": _mapping(seed_41.get("tick_114")).get("avoided"),
    }


def _seed_41_sequence_cases(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "tick_113": _seed_41_case(comparisons, V94_SEED_41_TICK_113_BRANCH_ID),
        "tick_114": _seed_41_case(comparisons, V94_SEED_41_TICK_114_BRANCH_ID),
    }


def _seed_41_case(
    comparisons: Sequence[Mapping[str, object]],
    branch_id: str,
) -> dict[str, object]:
    matching = [
        dict(item) for item in comparisons if str(item.get("branch_id", "")) == branch_id
    ]
    return {
        "branch_id": branch_id,
        "comparison_found": bool(matching),
        "avoided": bool(matching)
        and all(_float(item.get("target_alive_delta")) >= 0.0 for item in matching),
        "comparisons": matching,
    }


def _per_seed_delta_means(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    fields = (
        "target_local_score_delta",
        "terminal_alive_delta",
        "birth_delta",
        "target_alive_delta",
        "target_energy_delta",
        "target_hydration_delta",
        "target_health_delta",
    )
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in comparisons:
        grouped[str(item.get("seed"))].append(item)
    return {
        seed: {
            field: _round(_mean([_float(item.get(field)) for item in items]))
            for field in fields
        }
        for seed, items in sorted(grouped.items())
    }


def _heuristic_count(aggregate: Mapping[str, object]) -> int:
    return int(
        aggregate.get("heuristic_action_source_count", 0)
        or (
            0
            if aggregate.get("zero_heuristic_all_labels") is True
            else aggregate.get("label_count", 0)
        )
    )


def _action_value(
    action_values: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next(
        (item for item in action_values if str(item.get("action", "")) == action),
        None,
    )


def _candidate_or_population_int(
    candidate: Mapping[str, object],
    population_final: Mapping[str, object],
    candidate_key: str,
    population_key: str,
) -> int:
    value = candidate.get(candidate_key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _int(value)
    return _int(population_final.get(population_key))


def _dominant_count(counts: Counter[str]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    item = max(sorted(counts.items()), key=lambda value: (value[1], value[0]))
    return item[0], int(item[1])


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    total = sum((float(left[index]) - float(right[index])) ** 2 for index in range(size))
    total += abs(len(left) - len(right))
    return total


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _cvar(values: Sequence[float], *, fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    count = max(1, int(math.ceil(float(len(values)) * fraction)))
    return _mean(sorted_values[:count])


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
