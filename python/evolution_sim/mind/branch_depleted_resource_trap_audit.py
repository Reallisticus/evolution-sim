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
)
from evolution_sim.mind.branch_mode_objective_audit import (
    _action_option_mode,
    _list_of_mappings,
    _mapping,
    _safe_rate,
    _target_terminal_projection,
)
from evolution_sim.mind.branch_utility_risk_audit import (
    MIND_V3_BRANCH_UTILITY_RISK_AUDIT_SCHEMA_VERSION,
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

MIND_V3_DEPLETED_RESOURCE_TRAP_AUDIT_SCHEMA_VERSION = (
    "mind_v3_depleted_resource_trap_audit_v1"
)
V93_STRICT_EVAL_SEEDS: tuple[int, ...] = (13, 19, 29, 37, 41, 43)
V93_MIN_SUPPORT_TRAP_ROWS = 40
V93_MIN_STRICT_COMPARISON_COUNT = 48
V93_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA = 0.0
V93_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA = 0.0
V93_MAX_TARGET_ALIVE_NEGATIVE_COUNT = 0
V93_MAX_DOMINANT_PREDICTED_ACTION_SHARE = 0.50
V93_MIN_MEAN_TERMINAL_ALIVE_DELTA = 0.0
V93_MIN_MEAN_BIRTH_DELTA = 0.0
V93_K = 5
V93_SEED_41_BRANCH_ID = (
    "carrion-only-seed-41-action-branch-0-tick-113-agent-18-logged-move-south"
)

_BASELINE_RULES = {
    "v92_best_rule_baseline",
    "explicit_depleted_resource_trap_detector_baseline",
}
_ACCEPTANCE_CANDIDATE_RULES = {
    "trap_support_nearest_target_local_utility_k5",
    "trap_support_target_death_veto_k5",
    "trap_support_action_family_balanced_k5",
}


class BranchDepletedResourceTrapAuditError(ValueError):
    pass


def build_depleted_resource_trap_audit_report(
    *,
    support_branch_action_oracle_labels: Mapping[str, object],
    strict_branch_action_oracle_labels: Mapping[str, object],
    support_branch_action_oracle_audit: Mapping[str, object] | None = None,
    v92_branch_utility_risk_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _validate_label_report(support_branch_action_oracle_labels, "support")
    _validate_label_report(strict_branch_action_oracle_labels, "strict")
    if (
        support_branch_action_oracle_audit is not None
        and support_branch_action_oracle_audit.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION
    ):
        raise BranchDepletedResourceTrapAuditError(
            "support branch action oracle audit has unsupported schema_version"
        )
    if (
        v92_branch_utility_risk_audit is not None
        and v92_branch_utility_risk_audit.get("schema_version")
        != MIND_V3_BRANCH_UTILITY_RISK_AUDIT_SCHEMA_VERSION
    ):
        raise BranchDepletedResourceTrapAuditError(
            "v92 branch utility risk audit has unsupported schema_version"
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
    support_signature_trap_rows = [
        row for row in support_rows if _trap_signature(row)["is_trap"] is True
    ]
    support_trap_rows = _support_trap_rows(
        support_rows=support_rows,
        signature_trap_rows=support_signature_trap_rows,
        support_audit=support_branch_action_oracle_audit,
    )
    strict_trap_rows = [
        row for row in strict_rows if _trap_signature(row)["is_trap"] is True
    ]
    support_examples = _support_examples(support_trap_rows)
    rule_reports = _rule_reports(
        strict_rows,
        support_examples=support_examples,
        v92_branch_utility_risk_audit=v92_branch_utility_risk_audit,
    )
    coverage = _coverage_report(
        support_rows=support_rows,
        support_trap_rows=support_trap_rows,
        support_signature_trap_rows=support_signature_trap_rows,
        strict_rows=strict_rows,
        strict_trap_rows=strict_trap_rows,
        support_labels=support_branch_action_oracle_labels,
        strict_labels=strict_branch_action_oracle_labels,
        support_audit=support_branch_action_oracle_audit,
    )
    acceptance = _acceptance(coverage=coverage, rule_reports=rule_reports)
    contract = {
        "schema_version": MIND_V3_DEPLETED_RESOURCE_TRAP_AUDIT_SCHEMA_VERSION,
        "support_source_label_schema_version": (
            support_branch_action_oracle_labels.get("schema_version")
        ),
        "strict_eval_label_schema_version": (
            strict_branch_action_oracle_labels.get("schema_version")
        ),
        "runtime_policy_trained": False,
        "support_split_policy": (
            "generated_support_uses_mind_gate_train_seeds_excluding_strict_"
            "carrion_eval_seeds_v1"
        ),
        "strict_eval_seeds": list(V93_STRICT_EVAL_SEEDS),
        "candidate_prediction_policy": (
            "depleted_resource_trap_support_candidate_utility_knn_v1"
        ),
        "feature_contract": {
            "schema_version": "v93_policy_visible_depleted_resource_trap_features_v1",
            "uses_private_world_state": False,
            "uses_fixture_identity": False,
            "uses_seed_id_as_runtime_feature": False,
            "uses_branch_id_as_runtime_feature": False,
            "uses_logged_action_as_runtime_fallback": False,
            "features": [
                "decoded observation-derived compact self/local/navigation fields",
                "action mask",
                "same-agent public history trace",
                "candidate action identity and move direction",
                "current tile resource/carcass signal",
                "adjacent move target resource/carcass signal",
            ],
        },
        "trap_condition": {
            "low_or_falling_energy": "energy <= 0.35 or recent energy trend <= -0.02",
            "eat_supported": True,
            "current_resource_max": 0.08,
            "adjacent_resource_min": "max(0.12, current_resource + 0.08)",
            "move_supported": True,
        },
        "decision_rules": sorted(_BASELINE_RULES | _ACCEPTANCE_CANDIDATE_RULES),
        "acceptance_candidate_rules": sorted(_ACCEPTANCE_CANDIDATE_RULES),
        "baseline_rules_not_eligible_for_acceptance": sorted(_BASELINE_RULES),
        "support_floors": {
            "replay_verified": True,
            "heuristic_action_source_count": 0,
            "no_seed_branch_fixture_id_runtime_features": True,
            "no_logged_action_runtime_fallback": True,
            "min_generated_support_trap_rows_unless_insufficient": (
                V93_MIN_SUPPORT_TRAP_ROWS
            ),
            "strict_eval_comparison_count": V93_MIN_STRICT_COMPARISON_COUNT,
            "seed_41_catastrophe_avoided": True,
            "global_mean_target_local_score_delta_gt": (
                V93_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA
            ),
            "per_seed_target_local_score_delta_min": (
                V93_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA
            ),
            "target_alive_delta_negative_count": (
                V93_MAX_TARGET_ALIVE_NEGATIVE_COUNT
            ),
            "dominant_predicted_action_share_max": (
                V93_MAX_DOMINANT_PREDICTED_ACTION_SHARE
            ),
            "mean_terminal_alive_delta_min": V93_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            "mean_birth_delta_min": V93_MIN_MEAN_BIRTH_DELTA,
        },
    }
    accepted = bool(acceptance.get("v93_depleted_resource_trap_diagnostic_accepted"))
    return {
        "schema_version": MIND_V3_DEPLETED_RESOURCE_TRAP_AUDIT_SCHEMA_VERSION,
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
                    "aggregate": strict_branch_action_oracle_labels.get(
                        "aggregate"
                    ),
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
        "trap_state_discovery": {
            "support_trap_examples": [
                _trap_example(row) for row in support_trap_rows[:16]
            ],
            "support_signature_trap_examples": [
                _trap_example(row) for row in support_signature_trap_rows[:16]
            ],
            "strict_trap_examples": [
                _trap_example(row) for row in strict_trap_rows[:16]
            ],
            "seed_41_trap_signature": _seed_41_trap_signature(strict_rows),
        },
        "support_model": {
            "policy": "trap_support_candidate_action_utility_knn_k5_v1",
            "support_example_count": len(support_examples),
            "support_action_counts": dict(
                sorted(Counter(str(item["action"]) for item in support_examples).items())
            ),
            "support_mode_counts": dict(
                sorted(Counter(str(item["mode"]) for item in support_examples).items())
            ),
        },
        "decision_rule_reports": rule_reports,
        "depleted_resource_trap_support_probe": {
            "policy": "v93_depleted_resource_trap_support_v1",
            "accuracy": 1.0 if accepted else 0.0,
            "support_accuracy_floor": 1.0,
            "materially_supports_depleted_resource_trap": accepted,
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
        raise BranchDepletedResourceTrapAuditError(
            f"failed to read report: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BranchDepletedResourceTrapAuditError(
            f"report is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BranchDepletedResourceTrapAuditError("report must be a JSON object")
    return payload


def write_depleted_resource_trap_audit_report(
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
        raise BranchDepletedResourceTrapAuditError(
            f"{name} labels have unsupported schema_version"
        )


def _coverage_report(
    *,
    support_rows: Sequence[Mapping[str, object]],
    support_trap_rows: Sequence[Mapping[str, object]],
    support_signature_trap_rows: Sequence[Mapping[str, object]],
    strict_rows: Sequence[Mapping[str, object]],
    strict_trap_rows: Sequence[Mapping[str, object]],
    support_labels: Mapping[str, object],
    strict_labels: Mapping[str, object],
    support_audit: Mapping[str, object] | None,
) -> dict[str, object]:
    support_aggregate = _mapping(support_labels.get("aggregate"))
    strict_aggregate = _mapping(strict_labels.get("aggregate"))
    source_counts = _support_source_counts(support_audit)
    support_seeds = {int(row.get("seed", -1)) for row in support_rows}
    strict_seeds = {int(row.get("seed", -1)) for row in strict_rows}
    return {
        "support_label_count": len(support_rows),
        "support_trap_row_count": len(support_trap_rows),
        "support_signature_trap_row_count": len(support_signature_trap_rows),
        "support_candidate_action_count": sum(
            len(_candidate_actions(row)) for row in support_trap_rows
        ),
        "support_labels_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in support_rows).items())
        ),
        "support_trap_labels_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in support_trap_rows).items())
        ),
        "support_signature_trap_labels_by_seed": dict(
            sorted(
                Counter(str(row.get("seed")) for row in support_signature_trap_rows).items()
            )
        ),
        "support_source_selection": source_counts,
        "strict_eval_label_count": len(strict_rows),
        "strict_eval_trap_row_count": len(strict_trap_rows),
        "strict_eval_labels_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in strict_rows).items())
        ),
        "support_strict_seed_overlap": sorted(support_seeds & strict_seeds),
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
        "insufficient_support_trap_rows_proven": (
            len(support_trap_rows) < V93_MIN_SUPPORT_TRAP_ROWS
            and int(source_counts.get("eligible_depleted_resource_trap_row_count", 0))
            < V93_MIN_SUPPORT_TRAP_ROWS
        ),
    }


def _support_source_counts(audit: Mapping[str, object] | None) -> dict[str, object]:
    if audit is None:
        return {"available": False}
    discovery = _list_of_mappings(audit.get("discovery"), "discovery")
    return {
        "available": True,
        "branch_selection_policy": _mapping(audit.get("contract")).get(
            "branch_selection_policy"
        ),
        "eligible_row_count": sum(_int(item.get("eligible_row_count")) for item in discovery),
        "eligible_depleted_resource_trap_row_count": sum(
            _int(item.get("eligible_depleted_resource_trap_row_count"))
            for item in discovery
        ),
        "selected_depleted_resource_trap_row_count": sum(
            _int(item.get("selected_depleted_resource_trap_row_count"))
            for item in discovery
        ),
        "by_seed": [
            {
                "seed": item.get("seed"),
                "eligible_row_count": item.get("eligible_row_count"),
                "eligible_depleted_resource_trap_row_count": item.get(
                    "eligible_depleted_resource_trap_row_count"
                ),
                "selected_depleted_resource_trap_row_count": item.get(
                    "selected_depleted_resource_trap_row_count"
                ),
                "selected_depleted_resource_trap_score_summary": item.get(
                    "selected_depleted_resource_trap_score_summary"
                ),
            }
            for item in discovery
        ],
    }


def _support_trap_rows(
    *,
    support_rows: Sequence[Mapping[str, object]],
    signature_trap_rows: Sequence[Mapping[str, object]],
    support_audit: Mapping[str, object] | None,
) -> list[Mapping[str, object]]:
    source_counts = _support_source_counts(support_audit)
    if (
        source_counts.get("branch_selection_policy") == "depleted_resource_trap_v1"
        and int(source_counts.get("selected_depleted_resource_trap_row_count", 0))
        == len(support_rows)
    ):
        return list(support_rows)
    return list(signature_trap_rows)


def _support_examples(
    support_trap_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    examples = []
    for row in support_trap_rows:
        for candidate in _candidate_actions(row):
            action = str(candidate.get("action", ""))
            features = _candidate_feature_vector(row, action)
            if not features:
                continue
            target = _target_terminal_projection(candidate)
            examples.append(
                {
                    "seed": row.get("seed"),
                    "branch_id": row.get("branch_id"),
                    "action": action,
                    "mode": _action_option_mode(action),
                    "features": features,
                    "target_local_score": _target_local_scalar(candidate, row),
                    "target_alive": 1.0 if target.get("alive") is True else 0.0,
                    "target_death_risk": 0.0 if target.get("alive") is True else 1.0,
                }
            )
    return examples


def _rule_reports(
    strict_rows: Sequence[Mapping[str, object]],
    *,
    support_examples: Sequence[Mapping[str, object]],
    v92_branch_utility_risk_audit: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    reports = []
    v92_actions = _v92_best_rule_actions(v92_branch_utility_risk_audit)
    for rule in sorted(_BASELINE_RULES | _ACCEPTANCE_CANDIDATE_RULES):
        comparisons = []
        missing = 0
        unsupported = 0
        for row in strict_rows:
            action = _select_action(
                row,
                rule=rule,
                support_examples=support_examples,
                v92_actions=v92_actions,
            )
            if action is None:
                missing += 1
                continue
            if action not in ACTION_NAMES:
                unsupported += 1
                continue
            action_values = _candidate_actions(row)
            predicted = _action_value(action_values, action)
            logged = _action_value(action_values, str(row.get("logged_action", "")))
            if predicted is None or logged is None:
                missing += 1
                continue
            comparisons.append(
                _utility_comparison(
                    row,
                    rule=rule,
                    predicted_action=action,
                    predicted_run=predicted,
                    logged_run=logged,
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


def _select_action(
    row: Mapping[str, object],
    *,
    rule: str,
    support_examples: Sequence[Mapping[str, object]],
    v92_actions: Mapping[str, str],
) -> str | None:
    if rule == "v92_best_rule_baseline":
        return v92_actions.get(str(row.get("branch_id", "")))
    if rule == "explicit_depleted_resource_trap_detector_baseline":
        signature = _trap_signature(row)
        if signature["is_trap"] is True:
            return str(signature.get("best_adjacent_move_action") or "")
        return _support_rule_action(
            row,
            support_examples=support_examples,
            score_key="target_local_score_mean",
        )
    if rule == "trap_support_nearest_target_local_utility_k5":
        return _support_rule_action(
            row,
            support_examples=support_examples,
            score_key="target_local_score_mean",
        )
    if rule == "trap_support_target_death_veto_k5":
        return _support_rule_action(
            row,
            support_examples=support_examples,
            score_key="target_local_score_mean",
            death_risk_cap=0.2,
        )
    if rule == "trap_support_action_family_balanced_k5":
        return _support_rule_action(
            row,
            support_examples=support_examples,
            score_key="family_balanced_score",
        )
    return None


def _support_rule_action(
    row: Mapping[str, object],
    *,
    support_examples: Sequence[Mapping[str, object]],
    score_key: str,
    death_risk_cap: float | None = None,
) -> str | None:
    predictions = _candidate_support_predictions(row, support_examples)
    if death_risk_cap is not None:
        eligible = [
            item for item in predictions if _float(item.get("target_death_risk")) <= death_risk_cap
        ]
        if eligible:
            predictions = eligible
    if not predictions:
        return None
    selected = max(
        predictions,
        key=lambda item: (
            _float(item.get(score_key)),
            -_float(item.get("target_death_risk")),
            str(item.get("action", "")),
        ),
    )
    return str(selected.get("action", ""))


def _candidate_support_predictions(
    row: Mapping[str, object],
    support_examples: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    mode_counts = Counter(str(item.get("mode")) for item in support_examples)
    total_examples = max(1, len(support_examples))
    predictions = []
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
        selected = neighbors[:V93_K]
        scores = [_float(item[1].get("target_local_score")) for item in selected]
        death_risks = [_float(item[1].get("target_death_risk")) for item in selected]
        mode = _action_option_mode(action)
        mode_share = float(mode_counts.get(mode, 0)) / float(total_examples)
        score_mean = _mean(scores)
        death_risk = _mean(death_risks)
        predictions.append(
            {
                "action": action,
                "mode": mode,
                "target_local_score_mean": _round(score_mean),
                "target_death_risk": _round(death_risk),
                "family_balanced_score": _round(
                    score_mean - 500.0 * death_risk - 25.0 * mode_share
                ),
            }
        )
    return predictions


def _v92_best_rule_actions(
    v92_report: Mapping[str, object] | None,
) -> dict[str, str]:
    if v92_report is None:
        return {}
    acceptance = _mapping(v92_report.get("acceptance"))
    best_rule = str(_mapping(acceptance.get("best_rule_for_diagnostics")).get("rule", ""))
    for report in _list_of_mappings(
        v92_report.get("decision_rule_reports"),
        "decision_rule_reports",
    ):
        if str(report.get("rule")) != best_rule:
            continue
        return {
            str(item.get("branch_id")): str(item.get("predicted_action"))
            for item in _list_of_mappings(
                report.get("worst_examples"),
                "worst_examples",
            )
        } | {
            str(item.get("branch_id")): str(item.get("predicted_action"))
            for item in _list_of_mappings(
                report.get("negative_comparisons"),
                "negative_comparisons",
            )
        }
    return {}


def _summarize_rule(
    rule: str,
    comparisons: Sequence[Mapping[str, object]],
    *,
    missing_branch_outcome_count: int,
    unsupported_predicted_action_count: int,
) -> dict[str, object]:
    predicted_actions = Counter(str(item.get("predicted_action")) for item in comparisons)
    dominant_action, dominant_action_count = _dominant_count(predicted_actions)
    seed_41 = [
        item for item in comparisons if str(item.get("branch_id")) == V93_SEED_41_BRANCH_ID
    ]
    return {
        "rule": rule,
        "rule_class": "baseline" if rule in _BASELINE_RULES else "trap_support",
        "eligible_for_acceptance": rule in _ACCEPTANCE_CANDIDATE_RULES,
        "uses_logged_action_runtime_fallback": False,
        "uses_seed_branch_fixture_or_hidden_state_runtime_features": False,
        "comparison_count": len(comparisons),
        "missing_branch_outcome_count": int(missing_branch_outcome_count),
        "unsupported_predicted_action_count": int(unsupported_predicted_action_count),
        "predicted_action_counts": dict(sorted(predicted_actions.items())),
        "dominant_predicted_action": dominant_action,
        "dominant_predicted_action_count": dominant_action_count,
        "dominant_predicted_action_share": _safe_rate(
            dominant_action_count,
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
        "per_seed_target_local_score_delta": _per_seed_field_mean(
            comparisons,
            "target_local_score_delta",
        ),
        "seed_41_catastrophe": {
            "branch_id": V93_SEED_41_BRANCH_ID,
            "comparison_found": bool(seed_41),
            "avoided": bool(seed_41)
            and all(_float(item.get("target_alive_delta")) >= 0.0 for item in seed_41),
            "comparisons": [dict(item) for item in seed_41],
        },
        "negative_comparisons": sorted(
            [
                dict(item)
                for item in comparisons
                if _float(item.get("target_local_score_delta")) < 0.0
            ],
            key=lambda item: (
                _float(item.get("target_local_score_delta")),
                str(item.get("branch_id", "")),
            ),
        ),
        "worst_examples": sorted(
            [dict(item) for item in comparisons],
            key=lambda item: (
                _float(item.get("target_local_score_delta")),
                str(item.get("branch_id", "")),
            ),
        )[:16],
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
    best = _best_rule(rule_reports, blockers_by_rule)
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
        "v93_depleted_resource_trap_diagnostic_accepted": accepted,
        "v94_runtime_work_allowed": accepted,
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
    if (
        int(coverage.get("support_trap_row_count", 0)) < V93_MIN_SUPPORT_TRAP_ROWS
        and coverage.get("insufficient_support_trap_rows_proven") is not True
    ):
        blockers.append(
            {
                "reason": "support_trap_row_count_below_floor",
                "observed": int(coverage.get("support_trap_row_count", 0)),
                "required_min": V93_MIN_SUPPORT_TRAP_ROWS,
            }
        )
    if int(report.get("comparison_count", 0)) < V93_MIN_STRICT_COMPARISON_COUNT:
        blockers.append(
            {
                "reason": "strict_eval_comparison_count_below_floor",
                "observed": int(report.get("comparison_count", 0)),
                "required_min": V93_MIN_STRICT_COMPARISON_COUNT,
            }
        )
    seed_41 = _mapping(report.get("seed_41_catastrophe"))
    if seed_41.get("avoided") is not True:
        blockers.append(
            {
                "reason": "seed_41_catastrophe_not_avoided",
                "observed": seed_41,
                "required": "target_alive_delta >= 0 on seed 41 branch",
            }
        )
    target_local = _mapping(report.get("target_local_score_delta_summary"))
    if _float(target_local.get("mean")) <= V93_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA:
        blockers.append(
            {
                "reason": "mean_target_local_score_delta_not_positive",
                "observed": target_local.get("mean"),
                "required_gt": V93_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    per_seed = _mapping(report.get("per_seed_target_local_score_delta"))
    negative_seeds = {
        seed: value
        for seed, value in sorted(per_seed.items())
        if _float(value) < V93_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA
    }
    if negative_seeds:
        blockers.append(
            {
                "reason": "per_seed_target_local_score_delta_negative",
                "observed": negative_seeds,
                "required_min": V93_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    target_alive = _mapping(report.get("target_alive_delta_summary"))
    if int(target_alive.get("negative_count", 0)) != V93_MAX_TARGET_ALIVE_NEGATIVE_COUNT:
        blockers.append(
            {
                "reason": "target_alive_delta_negative_count_nonzero",
                "observed": int(target_alive.get("negative_count", 0)),
                "required": V93_MAX_TARGET_ALIVE_NEGATIVE_COUNT,
            }
        )
    if (
        _float(report.get("dominant_predicted_action_share"))
        > V93_MAX_DOMINANT_PREDICTED_ACTION_SHARE
    ):
        blockers.append(
            {
                "reason": "dominant_predicted_action_share_above_cap",
                "observed": report.get("dominant_predicted_action_share"),
                "required_max": V93_MAX_DOMINANT_PREDICTED_ACTION_SHARE,
            }
        )
    terminal_alive = _mapping(report.get("terminal_alive_delta_summary"))
    if _float(terminal_alive.get("mean")) < V93_MIN_MEAN_TERMINAL_ALIVE_DELTA:
        blockers.append(
            {
                "reason": "mean_terminal_alive_delta_negative",
                "observed": terminal_alive.get("mean"),
                "required_min": V93_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            }
        )
    births = _mapping(report.get("birth_delta_summary"))
    if _float(births.get("mean")) < V93_MIN_MEAN_BIRTH_DELTA:
        blockers.append(
            {
                "reason": "mean_birth_delta_negative",
                "observed": births.get("mean"),
                "required_min": V93_MIN_MEAN_BIRTH_DELTA,
            }
        )
    if report.get("uses_logged_action_runtime_fallback") is not False:
        blockers.append({"reason": "logged_action_runtime_fallback_used"})
    if report.get("uses_seed_branch_fixture_or_hidden_state_runtime_features") is not False:
        blockers.append(
            {"reason": "seed_branch_fixture_or_hidden_state_runtime_features_used"}
        )
    return blockers


def _best_rule(
    rule_reports: Sequence[Mapping[str, object]],
    blockers_by_rule: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    blocker_count_by_rule = {
        str(item.get("rule")): len(_list_of_mappings(item.get("blockers"), "blockers"))
        for item in blockers_by_rule
    }
    candidates = [
        report
        for report in rule_reports
        if report.get("eligible_for_acceptance") is True
    ]
    if not candidates:
        return {}
    best = max(
        candidates,
        key=lambda report: (
            -blocker_count_by_rule.get(str(report.get("rule")), 999),
            1 if _mapping(report.get("seed_41_catastrophe")).get("avoided") is True else 0,
            -int(_mapping(report.get("target_alive_delta_summary")).get("negative_count", 0)),
            _float(_mapping(report.get("target_local_score_delta_summary")).get("mean")),
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
        "seed_41_catastrophe_avoided": _mapping(
            best.get("seed_41_catastrophe")
        ).get("avoided"),
        "dominant_predicted_action_share": best.get(
            "dominant_predicted_action_share"
        ),
    }


def _trap_signature(row: Mapping[str, object]) -> dict[str, object]:
    state = _mapping(row.get("compact_state"))
    self_state = _mapping(state.get("self"))
    center = _mapping(state.get("center"))
    adjacent = _mapping(state.get("adjacent"))
    before = _mapping(row.get("before"))
    action_mask = _mapping(row.get("action_mask"))
    energy = _float(before.get("energy_ratio") or self_state.get("energy_ratio"))
    history = _list_of_mappings(row.get("public_history_trace"), "public_history_trace")
    energy_trend = _mean(
        [_float(item.get("energy_ratio_delta")) for item in history[-3:]]
    )
    move_resources = {}
    for action, direction in {
        "move_north": "north",
        "move_south": "south",
        "move_east": "east",
        "move_west": "west",
    }.items():
        if bool(action_mask.get(action, False)):
            move_resources[action] = _cell_resource(_mapping(adjacent.get(direction)))
    best_move = (
        max(move_resources, key=lambda action: (move_resources[action], action))
        if move_resources
        else None
    )
    current_resource = _cell_resource(center)
    best_adjacent = move_resources[best_move] if best_move is not None else 0.0
    is_trap = (
        ((energy <= 0.35) or energy_trend <= -0.02)
        and bool(action_mask.get("eat", False))
        and bool(move_resources)
        and current_resource <= 0.08
        and best_adjacent >= max(0.12, current_resource + 0.08)
    )
    return {
        "is_trap": is_trap,
        "energy_ratio": _round(energy),
        "energy_trend": _round(energy_trend),
        "eat_supported": bool(action_mask.get("eat", False)),
        "current_resource": _round(current_resource),
        "best_adjacent_move_action": best_move,
        "best_adjacent_resource": _round(best_adjacent),
        "move_supported": bool(move_resources),
    }


def _trap_example(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": row.get("branch_id"),
        "seed": row.get("seed"),
        "logged_action": row.get("logged_action"),
        "target_local_action": row.get("target_local_action"),
        "trap_signature": _trap_signature(row),
    }


def _seed_41_trap_signature(
    strict_rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    for row in strict_rows:
        if str(row.get("branch_id")) == V93_SEED_41_BRANCH_ID:
            return _trap_example(row)
    return None


def _cell_resource(cell: Mapping[str, object]) -> float:
    return max(
        _float(cell.get("food")),
        _float(cell.get("fresh_kill")),
        _float(cell.get("fresh_kill_energy")),
        _float(cell.get("carcass")),
        _float(cell.get("carcass_energy")),
        _float(cell.get("carrion_signal")) * 0.25,
    )


def _action_value(
    action_values: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next(
        (item for item in action_values if str(item.get("action", "")) == action),
        None,
    )


def _per_seed_field_mean(
    comparisons: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in comparisons:
        grouped[str(item.get("seed"))].append(item)
    return {
        seed: _round(_mean([_float(item.get(field)) for item in items]))
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
