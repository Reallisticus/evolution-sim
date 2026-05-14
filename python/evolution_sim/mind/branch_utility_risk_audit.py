from __future__ import annotations

import gzip
import json
import math
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import decode_observation_input
from evolution_sim.mind.branch_action_oracle_labels import (
    MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
    _MOVE_DELTAS,
    _branch_continuation_archive_feature_vector,
    _compact_world_model_rows,
    _public_history_feature_vector,
    _squared_distance,
)
from evolution_sim.mind.branch_mode_objective_audit import (
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

MIND_V3_BRANCH_UTILITY_RISK_AUDIT_SCHEMA_VERSION = (
    "mind_v3_branch_utility_risk_audit_v1"
)
V92_MIN_COMPARISON_COUNT = 48
V92_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA = 0.0
V92_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA = 0.0
V92_MAX_TARGET_ALIVE_NEGATIVE_COUNT = 0
V92_MIN_MEAN_TERMINAL_ALIVE_DELTA = 0.0
V92_MIN_MEAN_BIRTH_DELTA = 0.0
V92_MAX_DOMINANT_PREDICTED_ACTION_SHARE = 0.50
V92_MAX_DOMINANT_PREDICTED_MODE_SHARE = 0.75
V92_TARGET_DEATH_RISK_VETO = 0.20
V92_VITAL_REGRESSION_RISK_VETO = 0.45
V92_CATASTROPHIC_SCORE_DELTA_THRESHOLD = -1000.0
V92_K = 5

_BASELINE_RULES = {
    "v91_option_mode_prediction_baseline_k1",
    "navigation_baseline_toward_water",
    "navigation_baseline_toward_carrion",
    "navigation_baseline_toward_resource_need",
}
_ACCEPTANCE_CANDIDATE_RULES = {
    "mean_predicted_target_local_utility_k5",
    "lower_confidence_bound_target_local_utility_k5",
    "cvar_target_local_utility_k5",
    "target_death_risk_veto_mean_utility_k5",
    "vital_regression_risk_veto_mean_utility_k5",
    "catastrophe_veto_lcb_utility_k5",
    "action_family_balanced_risk_scorer_k5",
}


class BranchUtilityRiskAuditError(ValueError):
    pass


def build_branch_utility_risk_audit_report(
    branch_action_oracle_labels: Mapping[str, object],
) -> dict[str, object]:
    if (
        branch_action_oracle_labels.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION
    ):
        raise BranchUtilityRiskAuditError(
            "branch action oracle labels have unsupported schema_version"
        )

    labels = _list_of_mappings(branch_action_oracle_labels.get("labels"), "labels")
    rows = _utility_rows(labels)
    coverage = _coverage_report(
        rows,
        branch_action_oracle_labels=branch_action_oracle_labels,
    )
    candidate_scores = _leave_one_seed_candidate_predictions(rows, k=V92_K)
    rule_reports = _decision_rule_reports(rows, candidate_scores)
    acceptance = _acceptance(coverage=coverage, rule_reports=rule_reports)
    contract = {
        "schema_version": MIND_V3_BRANCH_UTILITY_RISK_AUDIT_SCHEMA_VERSION,
        "source_label_schema_version": branch_action_oracle_labels.get(
            "schema_version"
        ),
        "runtime_policy_trained": False,
        "split_policy": "leave_one_source_seed_out_v1",
        "candidate_prediction_policy": (
            "deterministic_policy_visible_candidate_utility_risk_knn_v1"
        ),
        "feature_contract": {
            "schema_version": "v92_policy_visible_candidate_utility_features_v1",
            "uses_private_world_state": False,
            "uses_fixture_identity": False,
            "uses_seed_id_as_runtime_feature": False,
            "uses_branch_id_as_runtime_feature": False,
            "uses_logged_action_as_runtime_fallback": False,
            "features": [
                "decoded policy observation values",
                "action mask",
                "compact self/local/navigation fields when decodable",
                "same-agent public history trace",
                "candidate action identity and move direction",
            ],
        },
        "decision_rules": sorted(_BASELINE_RULES | _ACCEPTANCE_CANDIDATE_RULES),
        "acceptance_candidate_rules": sorted(_ACCEPTANCE_CANDIDATE_RULES),
        "baseline_rules_not_eligible_for_acceptance": sorted(_BASELINE_RULES),
        "support_floors": {
            "replay_verified_all_labels": True,
            "heuristic_action_source_count": 0,
            "unsupported_predicted_or_oracle_actions": 0,
            "comparison_count": V92_MIN_COMPARISON_COUNT,
            "global_mean_target_local_score_delta_gt": (
                V92_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA
            ),
            "per_seed_mean_target_local_score_delta_min": (
                V92_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA
            ),
            "target_alive_delta_negative_count": (
                V92_MAX_TARGET_ALIVE_NEGATIVE_COUNT
            ),
            "mean_terminal_alive_delta_min": V92_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            "mean_birth_delta_min": V92_MIN_MEAN_BIRTH_DELTA,
            "dominant_predicted_action_share_max": (
                V92_MAX_DOMINANT_PREDICTED_ACTION_SHARE
            ),
            "dominant_predicted_mode_share_max": (
                V92_MAX_DOMINANT_PREDICTED_MODE_SHARE
            ),
            "no_logged_action_runtime_fallback": True,
            "no_seed_branch_fixture_or_hidden_state_runtime_features": True,
        },
    }
    accepted = bool(
        acceptance.get("v92_catastrophe_sensitive_branch_utility_accepted")
    )
    return {
        "schema_version": MIND_V3_BRANCH_UTILITY_RISK_AUDIT_SCHEMA_VERSION,
        "contract": contract,
        "provenance": {
            "source_label_digest": stable_payload_digest(
                {
                    "schema_version": branch_action_oracle_labels.get(
                        "schema_version"
                    ),
                    "label_contract": branch_action_oracle_labels.get(
                        "label_contract"
                    ),
                    "aggregate": branch_action_oracle_labels.get("aggregate"),
                    "acceptance": branch_action_oracle_labels.get("acceptance"),
                    "labels": labels,
                }
            ),
            "contract_digest": stable_payload_digest(contract),
        },
        "coverage": coverage,
        "candidate_utility_predictions": {
            "policy": "leave_one_source_seed_out_candidate_utility_risk_k5_v1",
            "feature_contract": contract["feature_contract"],
            "candidate_prediction_count": sum(
                len(items) for items in candidate_scores.values()
            ),
            "held_out_seed_leak_count": sum(
                1
                for items in candidate_scores.values()
                for item in items
                if item.get("uses_held_out_seed") is True
            ),
            "candidate_scores": [
                item
                for branch_id in sorted(candidate_scores)
                for item in candidate_scores[branch_id]
            ],
        },
        "decision_rule_reports": rule_reports,
        "catastrophe_sensitive_utility_support_probe": {
            "policy": "v92_catastrophe_sensitive_branch_utility_v1",
            "accuracy": 1.0 if accepted else 0.0,
            "support_accuracy_floor": 1.0,
            "materially_supports_catastrophe_sensitive_branch_utility": accepted,
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
        raise BranchUtilityRiskAuditError(f"failed to read report: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise BranchUtilityRiskAuditError(
            f"report is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BranchUtilityRiskAuditError("report must be a JSON object")
    return payload


def write_branch_utility_risk_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _utility_rows(labels: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    objective_rows = _objective_rows(labels)
    compact_rows = {
        str(row.get("branch_id", "")): row for row in _compact_world_model_rows(labels)
    }
    label_by_id = {str(label.get("branch_id", "")): label for label in labels}
    rows: list[dict[str, object]] = []
    for row in objective_rows:
        branch_id = str(row.get("branch_id", ""))
        label = _mapping(label_by_id.get(branch_id))
        policy_state = _mapping(label.get("policy_state"))
        compact = _mapping(compact_rows.get(branch_id))
        enriched = dict(row)
        enriched["policy_observation_values"] = _decode_policy_observation_values(
            policy_state
        )
        enriched["policy_state"] = dict(policy_state)
        if not enriched.get("public_history_trace"):
            enriched["public_history_trace"] = list(
                _list_of_mappings(
                    policy_state.get("public_history_trace"),
                    "public_history_trace",
                )
            )
        if not enriched.get("observation_values") and compact.get(
            "observation_values"
        ):
            enriched["observation_values"] = compact.get("observation_values")
        rows.append(enriched)
    return rows


def _coverage_report(
    rows: Sequence[Mapping[str, object]],
    *,
    branch_action_oracle_labels: Mapping[str, object],
) -> dict[str, object]:
    aggregate = _mapping(branch_action_oracle_labels.get("aggregate"))
    heuristic_count = int(
        aggregate.get("heuristic_action_source_count", 0)
        or (
            0
            if aggregate.get("zero_heuristic_all_labels") is True
            else aggregate.get("label_count", 0)
        )
    )
    return {
        "label_count": len(rows),
        "labels_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in rows).items())
        ),
        "labels_by_target_local_mode": dict(
            sorted(Counter(str(row.get("target_local_mode")) for row in rows).items())
        ),
        "labels_by_option_mode": dict(
            sorted(Counter(str(row.get("option_mode")) for row in rows).items())
        ),
        "candidate_action_count": sum(
            len(_candidate_actions(row)) for row in rows
        ),
        "replay_verified_all_labels": bool(
            aggregate.get("replay_verified_all_labels", False)
        ),
        "heuristic_action_source_count": heuristic_count,
        "zero_heuristic_all_labels": bool(
            aggregate.get("zero_heuristic_all_labels", False)
        ),
        "unsupported_oracle_action_count": int(
            aggregate.get("unsupported_oracle_action_count", 0)
        ),
        "unsupported_logged_action_count": int(
            aggregate.get("unsupported_logged_action_count", 0)
        ),
    }


def _leave_one_seed_candidate_predictions(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, list[dict[str, object]]]:
    examples = _candidate_training_examples(rows)
    predictions: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        branch_id = str(row.get("branch_id", ""))
        held_out_seed = row.get("seed")
        training = [item for item in examples if item.get("seed") != held_out_seed]
        mode_priors = _mode_priors(training)
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
                for item in training
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
            selected = neighbors[: max(1, int(k))]
            stats = _predicted_stats(
                selected,
                action=action,
                mode_priors=mode_priors,
            )
            actual = _candidate_actual_metrics(row, candidate)
            branch_predictions.append(
                {
                    "branch_id": branch_id,
                    "seed": row.get("seed"),
                    "held_out_seed": held_out_seed,
                    "action": action,
                    "mode": _action_option_mode(action),
                    "candidate_supported_by_mask": bool(
                        _mapping(row.get("action_mask")).get(action, False)
                    ),
                    "neighbor_count": len(selected),
                    "neighbor_seed_counts": dict(
                        sorted(
                            Counter(
                                str(item[1].get("seed")) for item in selected
                            ).items()
                        )
                    ),
                    "uses_held_out_seed": any(
                        item[1].get("seed") == held_out_seed for item in selected
                    ),
                    "predicted": stats,
                    "actual": actual,
                }
            )
        predictions[branch_id] = sorted(
            branch_predictions,
            key=lambda item: str(item.get("action", "")),
        )
    return predictions


def _candidate_training_examples(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    examples = []
    for row in rows:
        for candidate in _candidate_actions(row):
            action = str(candidate.get("action", ""))
            features = _candidate_feature_vector(row, action)
            if not features:
                continue
            metrics = _candidate_actual_metrics(row, candidate)
            examples.append(
                {
                    "seed": row.get("seed"),
                    "branch_id": row.get("branch_id"),
                    "action": action,
                    "mode": _action_option_mode(action),
                    "features": features,
                    "target_local_score": metrics["target_local_score"],
                    "target_alive": metrics["target_alive"],
                    "vital_regression_risk": metrics["vital_regression_risk"],
                }
            )
    return examples


def _predicted_stats(
    selected: Sequence[tuple[float, Mapping[str, object]]],
    *,
    action: str,
    mode_priors: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    scores = [_float(item[1].get("target_local_score")) for item in selected]
    death_risks = [
        1.0 - _float(item[1].get("target_alive"))
        for item in selected
    ]
    vital_risks = [
        _float(item[1].get("vital_regression_risk"))
        for item in selected
    ]
    mode = _action_option_mode(action)
    prior = _mapping(mode_priors.get(mode))
    score_mean = _mean(scores)
    score_std = _stddev(scores)
    cvar = _cvar(scores, fraction=0.25)
    death_risk = _mean(death_risks)
    vital_risk = _mean(vital_risks)
    mode_share = _float(prior.get("training_share"))
    mode_death_risk = _float(prior.get("death_risk"))
    return {
        "mean_target_local_utility": _round(score_mean),
        "target_local_utility_stddev": _round(score_std),
        "lower_confidence_bound_utility": _round(score_mean - score_std),
        "cvar_25_target_local_utility": _round(cvar),
        "target_death_risk": _round(death_risk),
        "vital_regression_risk": _round(vital_risk),
        "mode_training_share": _round(mode_share),
        "mode_death_risk": _round(mode_death_risk),
        "action_family_balanced_risk_score": _round(
            cvar
            - 250.0 * death_risk
            - 100.0 * vital_risk
            - 25.0 * mode_death_risk
            - 10.0 * mode_share
        ),
    }


def _mode_priors(
    examples: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    by_mode: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in examples:
        by_mode[str(item.get("mode"))].append(item)
    total = float(len(examples))
    priors: dict[str, dict[str, float]] = {}
    for mode, items in by_mode.items():
        count = float(len(items))
        priors[mode] = {
            "training_share": count / total if total else 0.0,
            "death_risk": _mean(
                [1.0 - _float(item.get("target_alive")) for item in items]
            ),
        }
    return priors


def _decision_rule_reports(
    rows: Sequence[Mapping[str, object]],
    candidate_scores: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    reports = []
    v91_predictions = _v91_option_mode_predictions(rows, k=1)
    for rule in sorted(_BASELINE_RULES | _ACCEPTANCE_CANDIDATE_RULES):
        comparisons: list[dict[str, object]] = []
        missing_count = 0
        unsupported_count = 0
        for row in rows:
            selected = _select_action_for_rule(
                row,
                rule=rule,
                candidate_scores=list(candidate_scores.get(str(row.get("branch_id")), [])),
                v91_prediction=v91_predictions.get(str(row.get("branch_id"))),
            )
            if selected is None:
                missing_count += 1
                continue
            action = str(selected)
            if action not in ACTION_NAMES:
                unsupported_count += 1
                continue
            action_values = _candidate_actions(row)
            predicted_run = _action_value(action_values, action)
            logged_run = _action_value(action_values, str(row.get("logged_action", "")))
            if predicted_run is None or logged_run is None:
                missing_count += 1
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
                missing_branch_outcome_count=missing_count,
                unsupported_predicted_action_count=unsupported_count,
            )
        )
    return reports


def _v91_option_mode_predictions(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, str]:
    predictions: dict[str, str] = {}
    examples = _training_examples(rows, objective_name="option_mode")
    for row in rows:
        seed = row.get("seed")
        training = [example for example in examples if example.get("seed") != seed]
        predicted_scores = _predicted_action_scores(row, training, k=k)
        if not predicted_scores:
            continue
        predicted_action, _ = _predicted_option_mode(predicted_scores)
        predictions[str(row.get("branch_id", ""))] = predicted_action
    return predictions


def _select_action_for_rule(
    row: Mapping[str, object],
    *,
    rule: str,
    candidate_scores: Sequence[Mapping[str, object]],
    v91_prediction: str | None,
) -> str | None:
    if rule == "v91_option_mode_prediction_baseline_k1":
        return v91_prediction
    if rule == "navigation_baseline_toward_water":
        return _navigation_baseline_action(row, target="water")
    if rule == "navigation_baseline_toward_carrion":
        return _navigation_baseline_action(row, target="carrion")
    if rule == "navigation_baseline_toward_resource_need":
        return _navigation_baseline_action(row, target=_resource_need_target(row))
    if not candidate_scores:
        return None
    if rule == "mean_predicted_target_local_utility_k5":
        return _max_by_prediction(candidate_scores, "mean_target_local_utility")
    if rule == "lower_confidence_bound_target_local_utility_k5":
        return _max_by_prediction(candidate_scores, "lower_confidence_bound_utility")
    if rule == "cvar_target_local_utility_k5":
        return _max_by_prediction(candidate_scores, "cvar_25_target_local_utility")
    if rule == "target_death_risk_veto_mean_utility_k5":
        return _veto_then_max(
            candidate_scores,
            score_key="mean_target_local_utility",
            risk_keys=("target_death_risk",),
            risk_caps=(V92_TARGET_DEATH_RISK_VETO,),
        )
    if rule == "vital_regression_risk_veto_mean_utility_k5":
        return _veto_then_max(
            candidate_scores,
            score_key="mean_target_local_utility",
            risk_keys=("vital_regression_risk",),
            risk_caps=(V92_VITAL_REGRESSION_RISK_VETO,),
        )
    if rule == "catastrophe_veto_lcb_utility_k5":
        return _veto_then_max(
            candidate_scores,
            score_key="lower_confidence_bound_utility",
            risk_keys=("target_death_risk", "vital_regression_risk"),
            risk_caps=(V92_TARGET_DEATH_RISK_VETO, V92_VITAL_REGRESSION_RISK_VETO),
        )
    if rule == "action_family_balanced_risk_scorer_k5":
        return _max_by_prediction(
            candidate_scores,
            "action_family_balanced_risk_score",
        )
    return None


def _max_by_prediction(
    candidate_scores: Sequence[Mapping[str, object]],
    score_key: str,
) -> str | None:
    if not candidate_scores:
        return None
    selected = max(
        candidate_scores,
        key=lambda item: (
            _float(_mapping(item.get("predicted")).get(score_key)),
            -_float(_mapping(item.get("predicted")).get("target_death_risk")),
            -_float(_mapping(item.get("predicted")).get("vital_regression_risk")),
            str(item.get("action", "")),
        ),
    )
    return str(selected.get("action", ""))


def _veto_then_max(
    candidate_scores: Sequence[Mapping[str, object]],
    *,
    score_key: str,
    risk_keys: Sequence[str],
    risk_caps: Sequence[float],
) -> str | None:
    eligible = [
        item
        for item in candidate_scores
        if all(
            _float(_mapping(item.get("predicted")).get(key)) <= cap
            for key, cap in zip(risk_keys, risk_caps)
        )
    ]
    if not eligible:
        eligible = sorted(
            candidate_scores,
            key=lambda item: tuple(
                _float(_mapping(item.get("predicted")).get(key))
                for key in risk_keys
            ),
        )
    return _max_by_prediction(eligible, score_key)


def _navigation_baseline_action(
    row: Mapping[str, object],
    *,
    target: str,
) -> str | None:
    candidates = _candidate_actions(row)
    move_candidates = [
        str(item.get("action", ""))
        for item in candidates
        if str(item.get("action", "")) in _MOVE_DELTAS
    ]
    if move_candidates:
        return max(
            move_candidates,
            key=lambda action: (_navigation_alignment(row, action, target), action),
        )
    return _first_supported_candidate(candidates)


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
    )[:12]
    action_family_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for item in comparisons:
        action_family_matrix[str(item.get("target_local_mode"))][
            str(item.get("predicted_mode"))
        ] += 1
    return {
        "rule": rule,
        "rule_class": "baseline" if rule in _BASELINE_RULES else "utility_risk",
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
        "catastrophic_class": _catastrophic_class_report(comparisons),
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
        "v92_catastrophe_sensitive_branch_utility_accepted": accepted,
        "v93_runtime_work_allowed": accepted,
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
    if coverage.get("replay_verified_all_labels") is not True:
        blockers.append({"reason": "replay_verified_all_labels_false"})
    if int(coverage.get("heuristic_action_source_count", 0)) != 0:
        blockers.append(
            {
                "reason": "heuristic_action_source_count_nonzero",
                "observed": int(coverage.get("heuristic_action_source_count", 0)),
                "required": 0,
            }
        )
    if int(coverage.get("unsupported_oracle_action_count", 0)) != 0:
        blockers.append(
            {
                "reason": "unsupported_oracle_action_count_nonzero",
                "observed": int(coverage.get("unsupported_oracle_action_count", 0)),
                "required": 0,
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
    if int(report.get("comparison_count", 0)) < V92_MIN_COMPARISON_COUNT:
        blockers.append(
            {
                "reason": "comparison_count_below_floor",
                "observed": int(report.get("comparison_count", 0)),
                "required_min": V92_MIN_COMPARISON_COUNT,
            }
        )
    target_local = _mapping(report.get("target_local_score_delta_summary"))
    if _float(target_local.get("mean")) <= V92_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA:
        blockers.append(
            {
                "reason": "mean_target_local_score_delta_not_positive",
                "observed": target_local.get("mean"),
                "required_gt": V92_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    per_seed = _mapping(report.get("per_seed_delta_means"))
    negative_seed_means = {
        seed: _mapping(value).get("target_local_score_delta")
        for seed, value in sorted(per_seed.items())
        if _float(_mapping(value).get("target_local_score_delta"))
        < V92_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA
    }
    if negative_seed_means:
        blockers.append(
            {
                "reason": "per_seed_mean_target_local_score_delta_negative",
                "observed": negative_seed_means,
                "required_min": V92_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    target_alive = _mapping(report.get("target_alive_delta_summary"))
    if int(target_alive.get("negative_count", 0)) != V92_MAX_TARGET_ALIVE_NEGATIVE_COUNT:
        blockers.append(
            {
                "reason": "target_alive_delta_negative_count_nonzero",
                "observed": int(target_alive.get("negative_count", 0)),
                "required": V92_MAX_TARGET_ALIVE_NEGATIVE_COUNT,
            }
        )
    terminal_alive = _mapping(report.get("terminal_alive_delta_summary"))
    if _float(terminal_alive.get("mean")) < V92_MIN_MEAN_TERMINAL_ALIVE_DELTA:
        blockers.append(
            {
                "reason": "mean_terminal_alive_delta_negative",
                "observed": terminal_alive.get("mean"),
                "required_min": V92_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            }
        )
    births = _mapping(report.get("birth_delta_summary"))
    if _float(births.get("mean")) < V92_MIN_MEAN_BIRTH_DELTA:
        blockers.append(
            {
                "reason": "mean_birth_delta_negative",
                "observed": births.get("mean"),
                "required_min": V92_MIN_MEAN_BIRTH_DELTA,
            }
        )
    if (
        _float(report.get("dominant_predicted_action_share"))
        > V92_MAX_DOMINANT_PREDICTED_ACTION_SHARE
    ):
        blockers.append(
            {
                "reason": "dominant_predicted_action_share_above_cap",
                "observed": report.get("dominant_predicted_action_share"),
                "required_max": V92_MAX_DOMINANT_PREDICTED_ACTION_SHARE,
            }
        )
    if (
        _float(report.get("dominant_predicted_mode_share"))
        > V92_MAX_DOMINANT_PREDICTED_MODE_SHARE
    ):
        blockers.append(
            {
                "reason": "dominant_predicted_mode_share_above_cap",
                "observed": report.get("dominant_predicted_mode_share"),
                "required_max": V92_MAX_DOMINANT_PREDICTED_MODE_SHARE,
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
        "dominant_predicted_mode_share": best.get(
            "dominant_predicted_mode_share"
        ),
    }


def _utility_comparison(
    row: Mapping[str, object],
    *,
    rule: str,
    predicted_action: str,
    predicted_run: Mapping[str, object],
    logged_run: Mapping[str, object],
) -> dict[str, object]:
    predicted_target = _target_terminal_projection(predicted_run)
    logged_target = _target_terminal_projection(logged_run)
    predicted_first = _mapping(predicted_run.get("first_action_outcome"))
    logged_first = _mapping(logged_run.get("first_action_outcome"))
    predicted_score = _target_local_scalar(predicted_run, row)
    logged_score = _target_local_scalar(logged_run, row)
    target_local_action = str(row.get("target_local_action", ""))
    return {
        "rule": rule,
        "branch_id": row.get("branch_id"),
        "seed": row.get("seed"),
        "logged_action": row.get("logged_action"),
        "predicted_action": predicted_action,
        "target_local_action": target_local_action,
        "target_local_mode": _action_option_mode(target_local_action),
        "predicted_mode": _action_option_mode(predicted_action),
        "predicted_target_local_score": _round(predicted_score),
        "logged_target_local_score": _round(logged_score),
        "target_local_score_delta": _round(predicted_score - logged_score),
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


def _candidate_actual_metrics(
    row: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    before = _mapping(row.get("before"))
    terminal = _target_terminal_projection(candidate)
    energy_delta = _float(terminal.get("energy_ratio")) - _float(
        before.get("energy_ratio")
    )
    hydration_delta = _float(terminal.get("hydration_ratio")) - _float(
        before.get("hydration_ratio")
    )
    health_delta = _float(terminal.get("health_ratio")) - _float(
        before.get("health_ratio")
    )
    return {
        "target_local_score": _round(_target_local_scalar(candidate, row)),
        "target_alive": 1.0 if terminal.get("alive") is True else 0.0,
        "terminal_alive_agents": _int(candidate.get("terminal_alive_agents")),
        "births": _int(candidate.get("births")),
        "target_energy_delta_from_start": _round(energy_delta),
        "target_hydration_delta_from_start": _round(hydration_delta),
        "target_health_delta_from_start": _round(health_delta),
        "vital_regression_risk": 1.0
        if terminal.get("alive") is not True
        or min(energy_delta, hydration_delta, health_delta) < -0.02
        else 0.0,
    }


def _target_local_scalar(
    candidate: Mapping[str, object],
    row: Mapping[str, object],
) -> float:
    score = _target_local_score(candidate, row)
    weights = (1000.0, 100.0, 25.0, 10.0, 5.0, 0.25, 0.25, 0.25, 1.0)
    return _round(
        sum(float(value) * weights[index] for index, value in enumerate(score))
    )


def _candidate_feature_vector(
    row: Mapping[str, object],
    action: str,
) -> tuple[float, ...]:
    archive_features = _branch_continuation_archive_feature_vector(
        row=row,
        action=action,
    )
    if archive_features:
        return archive_features
    values = row.get("policy_observation_values") or row.get("observation_values")
    if not isinstance(values, (tuple, list)):
        return ()
    action_mask = _mapping(row.get("action_mask"))
    supported_count = sum(1 for value in action_mask.values() if bool(value))
    dx, dy = _MOVE_DELTAS.get(action, (0, 0))
    history = _list_of_mappings(row.get("public_history_trace"), "public_history_trace")
    features = [
        *[float(value) for value in values],
        *[1.0 if action == name else 0.0 for name in ACTION_NAMES],
        float(dx),
        float(dy),
        1.0 if bool(action_mask.get(action, False)) else 0.0,
        min(float(supported_count) / max(float(len(ACTION_NAMES)), 1.0), 1.0),
        *_public_history_feature_vector(history),
    ]
    return tuple(_round(value) for value in features)


def _decode_policy_observation_values(
    policy_state: Mapping[str, object],
) -> tuple[float, ...]:
    observation_input = _mapping(policy_state.get("observation_input"))
    try:
        return tuple(
            _round(float(value))
            for value in decode_observation_input(dict(observation_input))
        )
    except (ValueError, TypeError, zlib.error):
        return ()


def _candidate_actions(row: Mapping[str, object]) -> list[Mapping[str, object]]:
    action_mask = _mapping(row.get("action_mask"))
    actions = []
    for candidate in _list_of_mappings(row.get("action_values"), "action_values"):
        action = str(candidate.get("action", ""))
        if action in ACTION_NAMES and bool(action_mask.get(action, False)):
            actions.append(candidate)
    return sorted(actions, key=lambda item: str(item.get("action", "")))


def _action_value(
    action_values: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next(
        (item for item in action_values if str(item.get("action", "")) == action),
        None,
    )


def _first_supported_candidate(
    candidates: Sequence[Mapping[str, object]],
) -> str | None:
    if not candidates:
        return None
    return str(sorted(candidates, key=lambda item: str(item.get("action", "")))[0].get("action", ""))


def _resource_need_target(row: Mapping[str, object]) -> str:
    before = _mapping(row.get("before"))
    if 1.0 - _float(before.get("hydration_ratio")) > 1.0 - _float(
        before.get("energy_ratio")
    ):
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


def _catastrophic_class_report(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    target_death = [
        dict(item)
        for item in comparisons
        if _float(item.get("target_alive_delta")) < 0.0
    ]
    catastrophic_score = [
        dict(item)
        for item in comparisons
        if _float(item.get("target_local_score_delta"))
        <= V92_CATASTROPHIC_SCORE_DELTA_THRESHOLD
    ]
    seed_41 = [
        item
        for item in target_death
        if str(item.get("seed")) == "41"
        or item.get("seed") == 41
    ]
    catastrophic_examples = _dedupe_comparisons([*target_death, *catastrophic_score])
    return {
        "definition": (
            "target_alive_delta < 0 or target_local_score_delta <= "
            f"{V92_CATASTROPHIC_SCORE_DELTA_THRESHOLD}"
        ),
        "target_death_negative_count": len(target_death),
        "catastrophic_score_count": len(catastrophic_score),
        "seed_41_target_death_negative_count": len(seed_41),
        "seed_41_catastrophic_class_avoided": len(seed_41) == 0,
        "worst_seed_41_examples": sorted(
            seed_41,
            key=lambda item: (
                _float(item.get("target_local_score_delta")),
                str(item.get("branch_id", "")),
            ),
        )[:8],
        "worst_examples": sorted(
            catastrophic_examples,
            key=lambda item: (
                _float(item.get("target_local_score_delta")),
                str(item.get("branch_id", "")),
            ),
        )[:12],
    }


def _dedupe_comparisons(
    comparisons: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str]] = set()
    deduped = []
    for item in comparisons:
        key = (
            str(item.get("rule", "")),
            str(item.get("branch_id", "")),
            str(item.get("predicted_action", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
    return deduped


def _field_summary(
    values: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, object]:
    series = [_float(item.get(field)) for item in values]
    if not series:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
        }
    return {
        "count": len(series),
        "mean": _round(_mean(series)),
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


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _stddev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(
        sum((value - mean) ** 2 for value in values) / float(len(values))
    )


def _cvar(values: Sequence[float], *, fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    count = max(1, int(math.ceil(float(len(values)) * fraction)))
    return _mean(sorted_values[:count])


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
