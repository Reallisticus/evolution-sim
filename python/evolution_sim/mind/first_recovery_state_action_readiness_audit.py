from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES, MOVEMENT_ACTIONS
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    DEFAULT_HISTORY_WINDOW,
    DEFAULT_SHADOW_RANKER_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_SCHEMA_VERSION,
    SIGNAL_FAMILY_ORDER,
    TargetEvidence,
    _all_state_values,
    _candidate_all_features,
    _counter_to_dict,
    _int,
    _join_evidence_section,
    _list,
    _load_join_records,
    _mapping,
    _number,
    _optional_string,
    _resolve_archive_rows,
    _resolve_json_report,
    _round,
    _share,
    _signature,
    _signature_digest,
    _target_evidence,
    signal_field_leakage,
)
from evolution_sim.mind.first_recovery_shadow_ranker import (
    LOCAL_LENIENT_TRAJECTORY_READER_POLICY,
    MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_SCHEMA_VERSION,
    group_archive_rows_by_branch_target,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_state_action_readiness_audit_v1"
)
MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_POLICY = (
    "diagnostics_only_first_recovery_state_action_interaction_readiness_v1"
)

DEFAULT_PUBLIC_SIGNAL_AUDIT_PATH = Path(
    "output/mind/mind-v3-v112-first-recovery-public-signal-audit.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v113-first-recovery-state-action-readiness-audit.json"
)

MAX_EXAMPLES = 12
ALIAS_DIGIT_POLICIES: tuple[tuple[str, int], ...] = (
    ("exact", 6),
    ("1_digit", 1),
    ("2_digit", 2),
    ("3_digit", 3),
)

CANDIDATE_FIELD_DEFINITIONS: tuple[dict[str, object], ...] = (
    {
        "field": "water_route_blocker_count",
        "family": "water_route_quality",
        "expected_public_field": "public.water_route_blocker_count",
        "public_contractable": True,
        "contract_source": "public_navigation_and_action_mask_resolution_summary",
    },
    {
        "field": "failed_intake_reason",
        "family": "recent_failed_intake",
        "expected_public_field": "public.failed_intake_reason",
        "public_contractable": True,
        "contract_source": "same_agent_public_action_outcome_history",
    },
    {
        "field": "same_patch_residence_duration",
        "family": "patch_residence_diminishing_returns",
        "expected_public_field": "public.same_patch_residence_duration",
        "public_contractable": True,
        "contract_source": "same_agent_public_position_history",
    },
    {
        "field": "competitor_intake_pressure",
        "family": "contestedness_competitor_pressure",
        "expected_public_field": "public.competitor_intake_pressure",
        "public_contractable": True,
        "contract_source": "public_local_patch_occupancy_and_resource_history",
    },
    {
        "field": "recent_carcass_depletion_by_others",
        "family": "carrion_freshness_depletion",
        "expected_public_field": "public.recent_carcass_depletion_by_others",
        "public_contractable": True,
        "contract_source": "public_resource_delta_history_without_actor_identity",
    },
    {
        "field": "carcass_age",
        "family": "carrion_freshness_depletion",
        "expected_public_field": "public.carcass_age",
        "public_contractable": True,
        "contract_source": "public_resource_lifecycle_observation",
    },
    {
        "field": "nearby_compatible_mate_count",
        "family": "reproduction_readiness_debt",
        "expected_public_field": "public.nearby_compatible_mate_count",
        "public_contractable": True,
        "contract_source": "public_reproductive_compatibility_count_not_identity",
    },
)

FORBIDDEN_SIGNAL_FIELD_TOKENS = frozenset(
    {
        "seed",
        "source",
        "source_path",
        "source_kind",
        "fixture",
        "fixture_name",
        "branch",
        "branch_id",
        "record_index",
        "agent_id",
        "tick",
        "logged",
        "logged_action",
        "private",
        "private_world_state",
        "world",
        "simulation_world",
        "provenance",
    }
)


class FirstRecoveryStateActionReadinessAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FirstRecoveryStateActionReadinessAuditBuild:
    report: dict[str, object]


def build_first_recovery_state_action_readiness_audit(
    *,
    public_signal_audit: Mapping[str, object] | None = None,
    public_signal_audit_path: str | Path | None = DEFAULT_PUBLIC_SIGNAL_AUDIT_PATH,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    archive_rows: Sequence[Mapping[str, object]] | None = None,
    archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    shadow_ranker_report: Mapping[str, object] | None = None,
    shadow_ranker_report_path: str | Path | None = DEFAULT_SHADOW_RANKER_REPORT_PATH,
    trajectory_paths: Sequence[str | Path] = (),
    trajectory_glob_patterns: Sequence[str] = (),
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> FirstRecoveryStateActionReadinessAuditBuild:
    history_limit = _positive_int(history_window, field="history_window")
    contract = _contract(history_window=history_limit)
    public_signal_payload, public_signal_evidence = _resolve_json_report(
        public_signal_audit,
        public_signal_audit_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_SCHEMA_VERSION,
    )
    archive_payload, archive_report_evidence = _resolve_json_report(
        archive_report,
        archive_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    shadow_payload, shadow_report_evidence = _resolve_json_report(
        shadow_ranker_report,
        shadow_ranker_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_SCHEMA_VERSION,
    )
    rows, rows_evidence = _resolve_archive_rows(archive_rows, archive_rows_path)
    groups = group_archive_rows_by_branch_target(rows)
    joined = _load_join_records(trajectory_paths, trajectory_glob_patterns)
    targets, target_join = _target_evidence(
        groups,
        joined,
        history_window=history_limit,
    )
    source = _source_section(
        public_signal_audit=public_signal_payload,
        archive_report=archive_payload,
        shadow_ranker_report=shadow_payload,
        rows=rows,
        groups=groups,
        public_signal_evidence=public_signal_evidence,
        archive_report_evidence=archive_report_evidence,
        archive_rows_evidence=rows_evidence,
        shadow_report_evidence=shadow_report_evidence,
    )
    join_evidence = _join_evidence_section(joined, target_join)
    missing_evidence = _missing_evidence(source, join_evidence)
    baseline_evaluation = _baseline_evaluation(targets)
    alias_quantization = _alias_quantization_stability(targets)
    candidate_fields = _candidate_field_readiness(
        public_signal_audit=public_signal_payload,
        targets=targets,
        alias_quantization=alias_quantization,
    )
    support = _support_summary(targets)
    leakage = _leakage_guard(
        baseline_evaluation=baseline_evaluation,
        candidate_field_readiness=candidate_fields,
    )
    classification = _classification(
        missing_evidence=missing_evidence,
        baseline_evaluation=baseline_evaluation,
        alias_quantization=alias_quantization,
        candidate_field_readiness=candidate_fields,
        leakage=leakage,
    )
    recommendation = _recommendation(
        classification=classification,
        baseline_evaluation=baseline_evaluation,
        alias_quantization=alias_quantization,
        candidate_field_readiness=candidate_fields,
    )
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_POLICY,
        "contract": contract,
        "source": source,
        "join_evidence": join_evidence,
        "baseline_evaluation": baseline_evaluation,
        "alias_quantization_stability": alias_quantization,
        "candidate_field_readiness": candidate_fields,
        "support_summary": support,
        "leakage_guard": leakage,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryStateActionReadinessAuditBuild(report=report)


def write_first_recovery_state_action_readiness_audit_report(
    build: FirstRecoveryStateActionReadinessAuditBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def state_action_signal_field_leakage(field_names: Sequence[str]) -> tuple[str, ...]:
    leaks = set(signal_field_leakage(field_names))
    for name in field_names:
        tokens = _field_tokens(name)
        if any(token in FORBIDDEN_SIGNAL_FIELD_TOKENS for token in tokens):
            leaks.add(name)
    return tuple(sorted(leaks))


def _contract(*, history_window: int) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_SCHEMA_VERSION,
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "observation_field_change": False,
        "private_world_state_serialized": False,
        "private_world_state_input": False,
        "fixture_identity_signal": False,
        "seed_identity_signal": False,
        "source_identity_signal": False,
        "branch_identity_signal": False,
        "logged_action_signal": False,
        "branch_outcomes_policy": "offline_oracle_labels_only_not_runtime_inputs",
        "trajectory_reader_policy": LOCAL_LENIENT_TRAJECTORY_READER_POLICY,
        "history_window": int(history_window),
        "history_policy": "same_agent_records_strictly_before_branch_target_only",
        "contract_digest": stable_payload_digest(
            {
                "schema_version": MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_SCHEMA_VERSION,
                "history_window": int(history_window),
                "trajectory_reader_policy": LOCAL_LENIENT_TRAJECTORY_READER_POLICY,
            }
        ),
    }


def _source_section(
    *,
    public_signal_audit: Mapping[str, object] | None,
    archive_report: Mapping[str, object] | None,
    shadow_ranker_report: Mapping[str, object] | None,
    rows: Sequence[Mapping[str, object]],
    groups: Sequence[object],
    public_signal_evidence: Mapping[str, object],
    archive_report_evidence: Mapping[str, object],
    archive_rows_evidence: Mapping[str, object],
    shadow_report_evidence: Mapping[str, object],
) -> dict[str, object]:
    by_seed = Counter()
    by_source = Counter()
    for group in groups:
        seed = getattr(group, "seed", None)
        source_kind = getattr(group, "source_kind", None)
        if seed is not None:
            by_seed[str(seed)] += 1
        if source_kind:
            by_source[str(source_kind)] += 1
    public_classification = _mapping(_mapping(public_signal_audit or {}).get("classification"))
    shadow_classification = _mapping(_mapping(shadow_ranker_report or {}).get("classification"))
    return {
        "public_signal_audit": public_signal_evidence,
        "archive_report": archive_report_evidence,
        "archive_rows": archive_rows_evidence,
        "shadow_ranker_report": shadow_report_evidence,
        "public_signal_schema_version": _mapping(public_signal_audit or {}).get(
            "schema_version"
        ),
        "archive_schema_version": _mapping(archive_report or {}).get("schema_version"),
        "shadow_ranker_schema_version": _mapping(shadow_ranker_report or {}).get(
            "schema_version"
        ),
        "archive_row_count": len(rows),
        "target_group_count": len(groups),
        "groups_by_seed": _counter_to_dict(by_seed),
        "groups_by_source_kind": _counter_to_dict(by_source),
        "v112_primary": public_classification.get("primary"),
        "v112_missing_evidence": _list(public_classification.get("missing_evidence")),
        "v111_primary": shadow_classification.get("primary"),
        "v111_missing_evidence": _list(shadow_classification.get("missing_evidence")),
    }


def _missing_evidence(
    source: Mapping[str, object],
    join_evidence: Mapping[str, object],
) -> list[str]:
    missing: list[str] = []
    for report_key in (
        "public_signal_audit",
        "archive_report",
        "archive_rows",
        "shadow_ranker_report",
    ):
        if _mapping(source.get(report_key)).get("loaded") is not True:
            missing.append(report_key)
    missing.extend(str(item) for item in _list(join_evidence.get("missing_evidence")))
    if _int(join_evidence.get("missing_archive_row_count")):
        missing.append("archive_trajectory_join")
    return sorted(set(missing))


def _baseline_evaluation(
    targets: Sequence[TargetEvidence],
) -> dict[str, object]:
    modes = ("action_only", "state_only", "state_action_interaction", "shuffled_action_control")
    all_data = {
        mode: _evaluate_lookup_ranker(
            targets,
            targets,
            mode=mode,
            split_name="all_data_descriptive",
        )
        for mode in modes
    }
    leave_one_seed = {
        mode: _leave_one_split(targets, mode=mode, split_field="seed")
        for mode in modes
    }
    leave_one_source = {
        mode: _leave_one_split(targets, mode=mode, split_field="source_kind")
        for mode in modes
    }
    fixture_open = {
        mode: _fixture_open_split(targets, mode=mode)
        for mode in modes
    }
    seed_source = _heldout_summary(
        action_only=leave_one_seed["action_only"],
        interaction=leave_one_seed["state_action_interaction"],
    )
    fixture_summary = _heldout_summary(
        action_only=fixture_open["action_only"],
        interaction=fixture_open["state_action_interaction"],
    )
    shuffled_seed = _shuffled_control_summary(
        action_only=leave_one_seed["action_only"],
        shuffled=leave_one_seed["shuffled_action_control"],
    )
    shuffled_fixture = _shuffled_control_summary(
        action_only=fixture_open["action_only"],
        shuffled=fixture_open["shuffled_action_control"],
    )
    action_collapse = (
        _number(
            _mapping(all_data["state_action_interaction"].get("selected_action_distribution")).get(
                "dominant_selected_action_share"
            )
        )
        > _number(
            _mapping(all_data["action_only"].get("selected_action_distribution")).get(
                "dominant_selected_action_share"
            )
        )
    )
    shuffled_artifact = (
        shuffled_seed["shuffled_control_beats_action_only"]
        or shuffled_fixture["shuffled_control_beats_action_only"]
    )
    return {
        "feature_policy": "diagnostic_lookup_baselines_no_training_artifact_v1",
        "diagnostic_scope": (
            "lookup_proxy_diagnostics_only_not_empirical_testing_of_proposed_missing_fields"
        ),
        "action_only": {
            "feature_policy": "candidate_action_semantics_only",
            "all_data_descriptive": all_data["action_only"],
            "leave_one_seed": leave_one_seed["action_only"],
            "leave_one_source": leave_one_source["action_only"],
            "fixture_open": fixture_open["action_only"],
        },
        "state_only": {
            "feature_policy": "decoded_public_state_and_same_agent_history_no_candidate_action",
            "all_data_descriptive": all_data["state_only"],
            "leave_one_seed": leave_one_seed["state_only"],
            "leave_one_source": leave_one_source["state_only"],
            "fixture_open": fixture_open["state_only"],
        },
        "state_action_interaction": {
            "feature_policy": "decoded_public_state_candidate_action_cross_features",
            "all_data_descriptive": all_data["state_action_interaction"],
            "leave_one_seed": leave_one_seed["state_action_interaction"],
            "leave_one_source": leave_one_source["state_action_interaction"],
            "fixture_open": fixture_open["state_action_interaction"],
            "beats_action_only_on_leave_one_seed": seed_source["interaction_beats_action_only"],
            "beats_action_only_on_fixture_open": fixture_summary[
                "interaction_beats_action_only"
            ],
            "action_collapse_worse_than_action_only": action_collapse,
        },
        "shuffled_action_control": {
            "feature_policy": "deterministic_within_target_oracle_label_rotation_control",
            "all_data_descriptive": all_data["shuffled_action_control"],
            "leave_one_seed": leave_one_seed["shuffled_action_control"],
            "leave_one_source": leave_one_source["shuffled_action_control"],
            "fixture_open": fixture_open["shuffled_action_control"],
        },
        "heldout_summary": {
            "leave_one_seed": seed_source,
            "fixture_open": fixture_summary,
            "shuffled_control": {
                "leave_one_seed": shuffled_seed,
                "fixture_open": shuffled_fixture,
                "shuffled_control_artifact_detected": shuffled_artifact,
            },
            "go_criterion_state_action_beats_action_only": (
                seed_source["interaction_beats_action_only"]
                and fixture_summary["interaction_beats_action_only"]
            ),
            "go_criterion_action_collapse_not_worsened": not action_collapse,
            "go_criterion_shuffled_control_not_better_than_action_only": (
                not shuffled_artifact
            ),
        },
    }


def _evaluate_lookup_ranker(
    train_targets: Sequence[TargetEvidence],
    eval_targets: Sequence[TargetEvidence],
    *,
    mode: str,
    split_name: str,
) -> dict[str, object]:
    if not train_targets or not eval_targets:
        return _empty_evaluation(split_name, mode, reason="missing_train_or_eval_targets")
    positive_by_signature: Counter[tuple[tuple[str, float], ...]] = Counter()
    total_by_signature: Counter[tuple[tuple[str, float], ...]] = Counter()
    positive_by_action: Counter[str] = Counter()
    total_by_action: Counter[str] = Counter()
    feature_names: set[str] = set()
    for target in train_targets:
        oracle_actions = set(target.oracle_actions)
        if mode == "shuffled_action_control":
            oracle_actions = set(_rotated_oracle_actions(target))
        for row in target.rows:
            action = _candidate_action(row)
            if action is None:
                continue
            signature = _feature_signature(target, row, mode=mode)
            feature_names.update(name for name, _value in signature)
            total_by_signature[signature] += 1
            total_by_action[action] += 1
            if action in oracle_actions:
                positive_by_signature[signature] += 1
                positive_by_action[action] += 1
    selected_actions: Counter[str] = Counter()
    selected_score_sources: Counter[str] = Counter()
    fallback_source_counts: Counter[str] = Counter()
    group_results: list[dict[str, object]] = []
    top1_matches = 0
    top2_hits = 0
    rank_sum = 0.0
    reciprocal_sum = 0.0
    material_gain_hits = 0
    material_gain_possible = 0
    candidate_row_count = 0
    for target in eval_targets:
        scored: list[tuple[float, int, str, dict[str, object]]] = []
        for row in target.rows:
            action = _candidate_action(row)
            if action is None:
                continue
            signature = _feature_signature(target, row, mode=mode)
            if total_by_signature[signature]:
                score = positive_by_signature[signature] / total_by_signature[signature]
                score_source = "signature_hit"
            elif action is not None and total_by_action[action]:
                score = positive_by_action[action] / total_by_action[action]
                score_source = "action_prior_fallback"
            else:
                score = 0.0
                score_source = "zero_score_fallback"
            candidate_row_count += 1
            fallback_source_counts[score_source] += 1
            scored.append((float(score), _action_sort_key(action), score_source, dict(row)))
        if not scored:
            continue
        scored.sort(key=lambda item: (-item[0], item[1], str(item[3].get("archive_row_id"))))
        selected = scored[0][3]
        selected_score_source = scored[0][2]
        selected_action = _candidate_action(selected) or "unknown"
        selected_actions[selected_action] += 1
        selected_score_sources[selected_score_source] += 1
        oracle_actions = set(target.oracle_actions)
        selected_rank = _oracle_rank(selected)
        selected_is_oracle = selected_action in oracle_actions
        top1_matches += 1 if selected_is_oracle else 0
        top2_actions = {
            _candidate_action(item[3])
            for item in scored[:2]
            if _candidate_action(item[3]) is not None
        }
        top2_hits += 1 if oracle_actions & top2_actions else 0
        oracle_position = _first_oracle_position(scored, oracle_actions)
        if oracle_position is not None:
            reciprocal_sum += 1.0 / float(oracle_position)
        rank_sum += selected_rank
        if any(row.get("material_gain_label") is True for row in target.rows):
            material_gain_possible += 1
            if selected.get("material_gain_label") is True:
                material_gain_hits += 1
        group_results.append(
            {
                "group_key": target.group_key,
                "selected_action": selected_action,
                "selected_oracle_rank": selected_rank,
                "oracle_actions": list(target.oracle_actions),
                "score": _round(scored[0][0]),
                "selected_score_source": selected_score_source,
            }
        )
    group_count = len(group_results)
    leaking_feature_names = state_action_signal_field_leakage(tuple(feature_names))
    return {
        "split_name": split_name,
        "mode": mode,
        "train_group_count": len(train_targets),
        "eval_group_count": len(eval_targets),
        "evaluated_group_count": group_count,
        "feature_count": len(feature_names),
        "top1_oracle_match_rate": _share(top1_matches, group_count),
        "top2_oracle_inclusion_rate": _share(top2_hits, group_count),
        "mrr": _round(reciprocal_sum / float(group_count)) if group_count else 0.0,
        "mean_selected_oracle_rank": _round(rank_sum / float(group_count))
        if group_count
        else 0.0,
        "material_gain_recall": _share(material_gain_hits, material_gain_possible),
        "selected_action_distribution": _selected_action_distribution(selected_actions),
        "candidate_row_score_source_counts": _counter_to_dict(fallback_source_counts),
        "candidate_row_score_source_rates": _source_rates(
            fallback_source_counts,
            candidate_row_count,
        ),
        "selected_score_source_distribution": _selected_source_distribution(
            selected_score_sources
        ),
        "signature_hit_count": int(fallback_source_counts["signature_hit"]),
        "action_prior_fallback_count": int(
            fallback_source_counts["action_prior_fallback"]
        ),
        "zero_score_fallback_count": int(fallback_source_counts["zero_score_fallback"]),
        "candidate_row_count": candidate_row_count,
        "feature_name_leakage_count": len(leaking_feature_names),
        "leaking_feature_names": list(leaking_feature_names),
        "examples": group_results[:MAX_EXAMPLES],
    }


def _leave_one_split(
    targets: Sequence[TargetEvidence],
    *,
    mode: str,
    split_field: str,
) -> dict[str, object]:
    buckets = _targets_by_provenance(targets, split_field)
    split_results: list[dict[str, object]] = []
    aggregate = _metric_accumulator()
    for value, eval_targets in sorted(buckets.items()):
        train_targets = tuple(target for target in targets if target not in eval_targets)
        result = _evaluate_lookup_ranker(
            train_targets,
            eval_targets,
            mode=mode,
            split_name=f"leave_one_{split_field}:{value}",
        )
        split_results.append({"heldout": value, **result})
        _accumulate_metrics(aggregate, result)
    summary = _finalize_metric_accumulator(aggregate)
    return {
        "split_field": split_field,
        "split_count": len(split_results),
        "summary": summary,
        "splits": split_results,
    }


def _fixture_open_split(
    targets: Sequence[TargetEvidence],
    *,
    mode: str,
) -> dict[str, object]:
    by_source = _targets_by_provenance(targets, "source_kind")
    splits: list[dict[str, object]] = []
    aggregate = _metric_accumulator()
    for heldout, eval_targets in sorted(by_source.items()):
        train_targets = tuple(target for target in targets if target not in eval_targets)
        result = _evaluate_lookup_ranker(
            train_targets,
            eval_targets,
            mode=mode,
            split_name=f"fixture_open:{heldout}",
        )
        splits.append({"heldout_source_kind": heldout, **result})
        _accumulate_metrics(aggregate, result)
    return {
        "split_policy": "hold_out_each_source_kind_fixture_open",
        "summary": _finalize_metric_accumulator(aggregate),
        "splits": splits,
    }


def _heldout_summary(
    *,
    action_only: Mapping[str, object],
    interaction: Mapping[str, object],
) -> dict[str, object]:
    action_summary = _mapping(action_only.get("summary"))
    interaction_summary = _mapping(interaction.get("summary"))
    action_top1 = _number(action_summary.get("top1_oracle_match_rate"))
    interaction_top1 = _number(interaction_summary.get("top1_oracle_match_rate"))
    action_mrr = _number(action_summary.get("mrr"))
    interaction_mrr = _number(interaction_summary.get("mrr"))
    return {
        "action_only_top1_oracle_match_rate": action_top1,
        "state_action_top1_oracle_match_rate": interaction_top1,
        "top1_delta": _round(interaction_top1 - action_top1),
        "action_only_mrr": action_mrr,
        "state_action_mrr": interaction_mrr,
        "mrr_delta": _round(interaction_mrr - action_mrr),
        "interaction_beats_action_only": (
            interaction_top1 > action_top1 or interaction_mrr > action_mrr
        ),
    }


def _shuffled_control_summary(
    *,
    action_only: Mapping[str, object],
    shuffled: Mapping[str, object],
) -> dict[str, object]:
    action_summary = _mapping(action_only.get("summary"))
    shuffled_summary = _mapping(shuffled.get("summary"))
    action_top1 = _number(action_summary.get("top1_oracle_match_rate"))
    shuffled_top1 = _number(shuffled_summary.get("top1_oracle_match_rate"))
    action_mrr = _number(action_summary.get("mrr"))
    shuffled_mrr = _number(shuffled_summary.get("mrr"))
    return {
        "action_only_top1_oracle_match_rate": action_top1,
        "shuffled_control_top1_oracle_match_rate": shuffled_top1,
        "top1_delta": _round(shuffled_top1 - action_top1),
        "action_only_mrr": action_mrr,
        "shuffled_control_mrr": shuffled_mrr,
        "mrr_delta": _round(shuffled_mrr - action_mrr),
        "shuffled_control_beats_action_only": (
            shuffled_top1 > action_top1 or shuffled_mrr > action_mrr
        ),
    }


def _alias_quantization_stability(
    targets: Sequence[TargetEvidence],
) -> dict[str, object]:
    policies: list[dict[str, object]] = []
    for label, digits in ALIAS_DIGIT_POLICIES:
        aliases = _alias_groups_for_digits(targets, digits=digits)
        alias_count = len(aliases)
        resolved_count = sum(
            1 for alias in aliases if alias.get("resolved_by_state_action_interaction")
        )
        policies.append(
            {
                "quantization": label,
                "digits": digits,
                "alias_group_count": alias_count,
                "alias_target_group_count": sum(
                    _int(alias.get("target_group_count")) for alias in aliases
                ),
                "resolved_by_state_action_interaction_count": resolved_count,
                "resolution_status": _quantization_resolution_status(
                    alias_count,
                    resolved_count,
                ),
                "alias_groups": aliases[:MAX_EXAMPLES],
            }
        )
    policies_with_aliases = [
        policy for policy in policies if _int(policy.get("alias_group_count")) > 0
    ]
    reference = next(
        (policy for policy in policies if _int(policy.get("digits")) == 2),
        {},
    )
    reference_alias_count = _int(_mapping(reference).get("alias_group_count"))
    stable_resolution = reference_alias_count > 0 and all(
        policy.get("resolution_status") == "resolved"
        for policy in policies_with_aliases
    )
    return {
        "state_signature_policy": "decoded_public_observation_plus_same_agent_public_history_no_provenance_v1",
        "v112_reference_quantization_digits": 2,
        "quantization_policies": policies,
        "aliasing_present_at_v112_reference_quantization": reference_alias_count > 0,
        "alias_precision_without_aliases_is_not_instability": True,
        "state_action_resolution_stable_exact_1_2_3_digit": stable_resolution,
    }


def _quantization_resolution_status(alias_count: int, resolved_count: int) -> str:
    if alias_count <= 0:
        return "no_aliases_at_precision"
    if resolved_count >= alias_count:
        return "resolved"
    return "unresolved"


def _alias_groups_for_digits(
    targets: Sequence[TargetEvidence],
    *,
    digits: int,
) -> list[dict[str, object]]:
    buckets: dict[tuple[tuple[str, float], ...], list[TargetEvidence]] = defaultdict(list)
    for target in targets:
        buckets[_signature(_all_state_values(target), digits=digits)].append(target)
    aliases: list[dict[str, object]] = []
    for signature, members in sorted(buckets.items(), key=lambda item: str(item[0])):
        oracle_actions = sorted(
            {action for member in members for action in member.oracle_actions},
            key=_action_sort_key,
        )
        if len(members) < 2 or len(oracle_actions) < 2:
            continue
        resolved = _alias_resolved_by_interaction(members)
        aliases.append(
            {
                "signature_digest": _signature_digest(signature),
                "target_group_count": len(members),
                "oracle_actions": oracle_actions,
                "resolved_by_state_action_interaction": resolved,
                "examples": [_target_example(member) for member in members[:MAX_EXAMPLES]],
            }
        )
    return aliases


def _alias_resolved_by_interaction(targets: Sequence[TargetEvidence]) -> bool:
    for target in targets:
        oracle_actions = set(target.oracle_actions)
        oracle_signatures = {
            _signature(_candidate_all_features(target, row), digits=3)
            for row in target.rows
            if _candidate_action(row) in oracle_actions
        }
        non_oracle_signatures = {
            _signature(_candidate_all_features(target, row), digits=3)
            for row in target.rows
            if _candidate_action(row) not in oracle_actions
        }
        if not oracle_signatures or not oracle_signatures.isdisjoint(non_oracle_signatures):
            return False
    return True


def _candidate_field_readiness(
    *,
    public_signal_audit: Mapping[str, object] | None,
    targets: Sequence[TargetEvidence],
    alias_quantization: Mapping[str, object],
) -> dict[str, object]:
    signal_families = _mapping(_mapping(public_signal_audit or {}).get("signal_families"))
    fields: list[dict[str, object]] = []
    for definition in CANDIDATE_FIELD_DEFINITIONS:
        field = str(definition["field"])
        family = str(definition["family"])
        expected = str(definition["expected_public_field"])
        family_section = _mapping(signal_families.get(family))
        missing_fields = {str(item) for item in _list(family_section.get("missing_public_fields"))}
        presence = _mapping(family_section.get("public_field_presence"))
        present_fields = {str(item) for item in _list(presence.get("present_fields"))}
        alias_group_count = _int(family_section.get("alias_group_count"))
        support_seeds = _mapping(family_section.get("per_seed_support"))
        support_sources = _mapping(family_section.get("fixture_open_support"))
        leakage = state_action_signal_field_leakage((field, expected))
        resolves_alias = (
            alias_group_count > 0
            or _mapping(_mapping(public_signal_audit or {}).get("classification")).get(
                "primary"
            )
            == "public_recovery_signal_aliasing_detected"
        )
        currently_present = expected in present_fields
        fields.append(
            {
                "field": field,
                "family": family,
                "expected_public_field": expected,
                "currently_present": currently_present,
                "reported_missing_by_v112": expected in missing_fields,
                "public_contractable": bool(definition["public_contractable"]),
                "contract_source": definition["contract_source"],
                "non_leaking": not leakage,
                "leakage_count": len(leakage),
                "leaking_field_names": list(leakage),
                "resolves_concrete_v112_alias_examples": bool(resolves_alias),
                "v112_family_alias_group_count": alias_group_count,
                "per_seed_support": {
                    str(key): int(value)
                    for key, value in sorted(support_seeds.items(), key=lambda item: str(item[0]))
                    if isinstance(value, int)
                },
                "fixture_open_support": {
                    str(key): int(value)
                    for key, value in sorted(support_sources.items(), key=lambda item: str(item[0]))
                    if isinstance(value, int)
                },
            }
        )
    contractable_missing_candidates = [
        field
        for field in fields
        if field["public_contractable"]
        and field["non_leaking"]
        and field["resolves_concrete_v112_alias_examples"]
    ]
    return {
        "candidate_fields": fields,
        "candidate_field_count": len(fields),
        "wording_policy": (
            "candidate fields are contractable missing-field candidates; "
            "v113 does not empirically evaluate proposed missing fields"
        ),
        "contractable_missing_field_candidate_count": len(
            contractable_missing_candidates
        ),
        "contractable_missing_field_candidates": [
            str(field["field"]) for field in contractable_missing_candidates
        ],
        "go_criterion_has_public_contractable_non_leaking_alias_resolver": bool(
            contractable_missing_candidates
        ),
        "alias_resolution_stability_reference": {
            "state_action_resolution_stable_exact_1_2_3_digit": alias_quantization.get(
                "state_action_resolution_stable_exact_1_2_3_digit"
            )
        },
        "target_group_count": len(targets),
    }


def _support_summary(targets: Sequence[TargetEvidence]) -> dict[str, object]:
    by_seed = Counter()
    by_source = Counter()
    action_support = Counter()
    for target in targets:
        seed = target.provenance.get("seed")
        source_kind = target.provenance.get("source_kind")
        if seed is not None:
            by_seed[str(seed)] += 1
        if source_kind is not None:
            by_source[str(source_kind)] += 1
        for action in target.oracle_actions:
            action_support[action] += 1
    return {
        "target_group_count": len(targets),
        "per_seed_support": _counter_to_dict(by_seed),
        "fixture_open_support": _counter_to_dict(by_source),
        "oracle_action_support": _counter_to_dict(action_support),
        "seed_count": len(by_seed),
        "source_kind_count": len(by_source),
    }


def _leakage_guard(
    *,
    baseline_evaluation: Mapping[str, object],
    candidate_field_readiness: Mapping[str, object],
) -> dict[str, object]:
    direct_names: list[str] = []
    direct_names.extend(_action_feature_names())
    direct_names.extend(
        str(field.get("field"))
        for field in _list(candidate_field_readiness.get("candidate_fields"))
        if isinstance(field, Mapping)
    )
    direct_leaks = state_action_signal_field_leakage(direct_names)
    baseline_leakage = _baseline_feature_leakage(baseline_evaluation)
    baseline_leaks = tuple(
        str(item) for item in _list(baseline_leakage.get("baseline_leaking_feature_names"))
    )
    leaks = tuple(sorted(set(direct_leaks) | set(baseline_leaks)))
    return {
        "answer": "leakage_detected" if leaks else "leakage_free",
        "direct_field_leakage_count": len(direct_leaks),
        "direct_leaking_field_names": list(direct_leaks),
        "baseline_feature_name_leakage_count": _int(
            baseline_leakage.get("baseline_feature_name_leakage_count")
        ),
        "baseline_leaking_feature_names": list(baseline_leaks),
        "leakage_count": len(leaks),
        "leaking_signal_fields": list(leaks),
        "forbidden_signal_field_tokens": sorted(FORBIDDEN_SIGNAL_FIELD_TOKENS),
        "provenance_used_for_join_and_examples_only": True,
    }


def _baseline_feature_leakage(value: object) -> dict[str, object]:
    count = 0
    names: set[str] = set()

    def visit(node: object) -> None:
        nonlocal count
        if isinstance(node, Mapping):
            if "feature_name_leakage_count" in node:
                count += _int(node.get("feature_name_leakage_count"))
            if "leaking_feature_names" in node:
                names.update(str(item) for item in _list(node.get("leaking_feature_names")))
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return {
        "baseline_feature_name_leakage_count": count,
        "baseline_leaking_feature_names": sorted(names),
    }


def _classification(
    *,
    missing_evidence: Sequence[str],
    baseline_evaluation: Mapping[str, object],
    alias_quantization: Mapping[str, object],
    candidate_field_readiness: Mapping[str, object],
    leakage: Mapping[str, object],
) -> dict[str, object]:
    labels: list[str] = ["diagnostics_only_no_runtime_promotion"]
    if leakage.get("leakage_count"):
        labels.insert(0, "state_action_signal_leakage_detected")
    if missing_evidence:
        labels.insert(0, "missing_evidence_inconclusive")
    heldout = _mapping(baseline_evaluation.get("heldout_summary"))
    beats = bool(heldout.get("go_criterion_state_action_beats_action_only"))
    collapse_ok = bool(heldout.get("go_criterion_action_collapse_not_worsened"))
    shuffled_ok = bool(
        heldout.get("go_criterion_shuffled_control_not_better_than_action_only")
    )
    stable_alias = bool(
        alias_quantization.get("state_action_resolution_stable_exact_1_2_3_digit")
    )
    candidate_ready = bool(
        candidate_field_readiness.get(
            "go_criterion_has_public_contractable_non_leaking_alias_resolver"
        )
    )
    labels.append(
        "state_action_interaction_beats_action_only"
        if beats
        else "state_action_interaction_not_better_than_action_only"
    )
    labels.append("alias_resolution_stable" if stable_alias else "alias_resolution_unstable")
    labels.append(
        "contractable_missing_field_candidates_present"
        if candidate_ready
        else "contractable_missing_field_candidates_absent"
    )
    labels.append(
        "state_action_action_collapse_not_worsened"
        if collapse_ok
        else "state_action_action_collapse_worsened"
    )
    if not shuffled_ok:
        labels.append("shuffled_control_artifact_detected")
    if (
        not missing_evidence
        and not leakage.get("leakage_count")
        and beats
        and stable_alias
        and candidate_ready
        and collapse_ok
        and shuffled_ok
    ):
        labels.append("state_action_observation_field_readiness_for_v114")
        primary = "state_action_observation_field_readiness_for_v114"
    elif missing_evidence:
        primary = "missing_evidence_inconclusive"
    else:
        labels.append("state_action_observation_field_readiness_blocked")
        primary = "state_action_observation_field_readiness_blocked"
    return {
        "primary": primary,
        "labels": _dedupe_preserve_order(labels),
        "missing_evidence": sorted(set(str(item) for item in missing_evidence)),
    }


def _recommendation(
    *,
    classification: Mapping[str, object],
    baseline_evaluation: Mapping[str, object],
    alias_quantization: Mapping[str, object],
    candidate_field_readiness: Mapping[str, object],
) -> dict[str, object]:
    labels = set(str(item) for item in _list(classification.get("labels")))
    if "missing_evidence_inconclusive" in labels:
        next_step = "reader_or_join_repair"
    elif "state_action_observation_field_readiness_for_v114" in labels:
        next_step = "planner_may_prepare_v114_observation_field_proposal"
    else:
        next_step = "readiness_blocked_repair_audit_or_run_broader_data_diagnostics"
    blockers: list[str] = []
    heldout = _mapping(baseline_evaluation.get("heldout_summary"))
    if "state_action_signal_leakage_detected" in labels:
        blockers.append("state_action_signal_leakage_detected")
    if "shuffled_control_artifact_detected" in labels:
        blockers.append("shuffled_control_artifact_detected")
    if not heldout.get("go_criterion_state_action_beats_action_only"):
        blockers.append("state_action_interaction_does_not_beat_action_only_heldout")
    if not alias_quantization.get("state_action_resolution_stable_exact_1_2_3_digit"):
        blockers.append("alias_resolution_not_stable_under_quantization")
    if not candidate_field_readiness.get(
        "go_criterion_has_public_contractable_non_leaking_alias_resolver"
    ):
        blockers.append("no_candidate_field_resolves_alias_examples")
    if not heldout.get("go_criterion_action_collapse_not_worsened"):
        blockers.append("state_action_interaction_worsens_action_collapse")
    return {
        "next_step": next_step,
        "blockers": blockers,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_made": False,
        "summary": (
            "v113 is diagnostics-only; its lookup/proxy baselines are not empirical "
            "tests of proposed missing fields and it does not add observation fields "
            "or runtime policy behavior."
        ),
    }


def _feature_signature(
    target: TargetEvidence,
    row: Mapping[str, object],
    *,
    mode: str,
) -> tuple[tuple[str, float], ...]:
    if mode == "action_only":
        return _signature(_action_features(row), digits=3)
    if mode == "state_only":
        return _signature(_all_state_values(target), digits=3)
    if mode in {"state_action_interaction", "shuffled_action_control"}:
        return _signature(_candidate_all_features(target, row), digits=3)
    raise FirstRecoveryStateActionReadinessAuditError(f"unknown baseline mode: {mode}")


def _action_features(row: Mapping[str, object]) -> dict[str, float]:
    action = _candidate_action(row) or ""
    values = {f"candidate_action.{name}": 1.0 if action == name else 0.0 for name in ACTION_NAMES}
    values["candidate_action.is_move"] = 1.0 if action in MOVEMENT_ACTIONS else 0.0
    values["candidate_action.is_intake"] = 1.0 if action in {"eat", "drink"} else 0.0
    values["candidate_action.is_reproduction"] = 1.0 if action == "mate" else 0.0
    return values


def _action_feature_names() -> tuple[str, ...]:
    names = [f"candidate_action.{name}" for name in ACTION_NAMES]
    names.extend(
        (
            "candidate_action.is_move",
            "candidate_action.is_intake",
            "candidate_action.is_reproduction",
        )
    )
    return tuple(names)


def _state_feature_names_from_evaluation(
    baseline_evaluation: Mapping[str, object],
) -> tuple[str, ...]:
    examples = _list(
        _mapping(
            _mapping(
                _mapping(baseline_evaluation.get("state_action_interaction")).get(
                    "all_data_descriptive"
                )
            ).get("examples")
        )
    )
    if not examples:
        return ()
    return tuple()


def _rotated_oracle_actions(target: TargetEvidence) -> tuple[str, ...]:
    actions = [_candidate_action(row) for row in target.rows]
    parsed = [action for action in actions if action is not None]
    if not parsed:
        return ()
    oracle = set(target.oracle_actions)
    rotated: list[str] = []
    for index, action in enumerate(parsed):
        if action in oracle:
            rotated.append(parsed[(index + 1) % len(parsed)])
    return tuple(sorted(set(rotated), key=_action_sort_key))


def _targets_by_provenance(
    targets: Sequence[TargetEvidence],
    field: str,
) -> dict[str, tuple[TargetEvidence, ...]]:
    buckets: dict[str, list[TargetEvidence]] = defaultdict(list)
    for target in targets:
        value = target.provenance.get(field)
        if value is not None:
            buckets[str(value)].append(target)
    return {
        key: tuple(value)
        for key, value in sorted(buckets.items(), key=lambda item: str(item[0]))
    }


def _metric_accumulator() -> dict[str, object]:
    return {
        "group_count": 0,
        "candidate_row_count": 0,
        "top1_sum": 0.0,
        "top2_sum": 0.0,
        "mrr_sum": 0.0,
        "rank_sum": 0.0,
        "material_gain_sum": 0.0,
        "selected_actions": Counter(),
        "candidate_row_score_sources": Counter(),
        "selected_score_sources": Counter(),
    }


def _accumulate_metrics(
    aggregate: dict[str, object],
    result: Mapping[str, object],
) -> None:
    count = _int(result.get("evaluated_group_count"))
    if count <= 0:
        return
    aggregate["group_count"] = _int(aggregate.get("group_count")) + count
    aggregate["candidate_row_count"] = _int(
        aggregate.get("candidate_row_count")
    ) + _int(result.get("candidate_row_count"))
    aggregate["top1_sum"] = _number(aggregate.get("top1_sum")) + (
        _number(result.get("top1_oracle_match_rate")) * count
    )
    aggregate["top2_sum"] = _number(aggregate.get("top2_sum")) + (
        _number(result.get("top2_oracle_inclusion_rate")) * count
    )
    aggregate["mrr_sum"] = _number(aggregate.get("mrr_sum")) + (
        _number(result.get("mrr")) * count
    )
    aggregate["rank_sum"] = _number(aggregate.get("rank_sum")) + (
        _number(result.get("mean_selected_oracle_rank")) * count
    )
    aggregate["material_gain_sum"] = _number(aggregate.get("material_gain_sum")) + (
        _number(result.get("material_gain_recall")) * count
    )
    selected = aggregate["selected_actions"]
    if isinstance(selected, Counter):
        selected.update(
            _mapping(_mapping(result.get("selected_action_distribution")).get("counts"))
        )
    candidate_sources = aggregate["candidate_row_score_sources"]
    if isinstance(candidate_sources, Counter):
        candidate_sources.update(_mapping(result.get("candidate_row_score_source_counts")))
    selected_sources = aggregate["selected_score_sources"]
    if isinstance(selected_sources, Counter):
        selected_sources.update(
            _mapping(_mapping(result.get("selected_score_source_distribution")).get("counts"))
        )


def _finalize_metric_accumulator(aggregate: Mapping[str, object]) -> dict[str, object]:
    count = _int(aggregate.get("group_count"))
    candidate_count = _int(aggregate.get("candidate_row_count"))
    selected = aggregate.get("selected_actions")
    selected_counter = selected if isinstance(selected, Counter) else Counter()
    candidate_sources = aggregate.get("candidate_row_score_sources")
    candidate_source_counter = (
        candidate_sources if isinstance(candidate_sources, Counter) else Counter()
    )
    selected_sources = aggregate.get("selected_score_sources")
    selected_source_counter = (
        selected_sources if isinstance(selected_sources, Counter) else Counter()
    )
    return {
        "evaluated_group_count": count,
        "candidate_row_count": candidate_count,
        "top1_oracle_match_rate": _round(_number(aggregate.get("top1_sum")) / count)
        if count
        else 0.0,
        "top2_oracle_inclusion_rate": _round(_number(aggregate.get("top2_sum")) / count)
        if count
        else 0.0,
        "mrr": _round(_number(aggregate.get("mrr_sum")) / count) if count else 0.0,
        "mean_selected_oracle_rank": _round(_number(aggregate.get("rank_sum")) / count)
        if count
        else 0.0,
        "material_gain_recall": _round(
            _number(aggregate.get("material_gain_sum")) / count
        )
        if count
        else 0.0,
        "selected_action_distribution": _selected_action_distribution(selected_counter),
        "candidate_row_score_source_counts": _counter_to_dict(candidate_source_counter),
        "candidate_row_score_source_rates": _source_rates(
            candidate_source_counter,
            candidate_count,
        ),
        "selected_score_source_distribution": _selected_source_distribution(
            selected_source_counter
        ),
        "signature_hit_count": int(candidate_source_counter["signature_hit"]),
        "action_prior_fallback_count": int(
            candidate_source_counter["action_prior_fallback"]
        ),
        "zero_score_fallback_count": int(candidate_source_counter["zero_score_fallback"]),
    }


def _empty_evaluation(split_name: str, mode: str, *, reason: str) -> dict[str, object]:
    return {
        "split_name": split_name,
        "mode": mode,
        "train_group_count": 0,
        "eval_group_count": 0,
        "evaluated_group_count": 0,
        "reason": reason,
        "top1_oracle_match_rate": 0.0,
        "top2_oracle_inclusion_rate": 0.0,
        "mrr": 0.0,
        "mean_selected_oracle_rank": 0.0,
        "material_gain_recall": 0.0,
        "selected_action_distribution": _selected_action_distribution(Counter()),
        "candidate_row_score_source_counts": {},
        "candidate_row_score_source_rates": {},
        "selected_score_source_distribution": _selected_source_distribution(Counter()),
        "signature_hit_count": 0,
        "action_prior_fallback_count": 0,
        "zero_score_fallback_count": 0,
        "candidate_row_count": 0,
        "feature_name_leakage_count": 0,
        "leaking_feature_names": [],
        "examples": [],
    }


def _selected_action_distribution(counter: Mapping[str, int]) -> dict[str, object]:
    total = sum(int(value) for value in counter.values())
    dominant_action = None
    dominant_count = 0
    for action, count in sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0]))):
        if int(count) > dominant_count:
            dominant_action = str(action)
            dominant_count = int(count)
    return {
        "counts": {str(key): int(counter[key]) for key in sorted(counter)},
        "total": total,
        "dominant_selected_action": dominant_action,
        "dominant_selected_action_share": _share(dominant_count, total),
    }


def _selected_source_distribution(counter: Mapping[str, int]) -> dict[str, object]:
    total = sum(int(value) for value in counter.values())
    return {
        "counts": {str(key): int(counter[key]) for key in sorted(counter)},
        "total": total,
    }


def _source_rates(counter: Mapping[str, int], total: int) -> dict[str, float]:
    return {
        "signature_hit": _share(int(counter.get("signature_hit", 0)), total),
        "action_prior_fallback": _share(
            int(counter.get("action_prior_fallback", 0)),
            total,
        ),
        "zero_score_fallback": _share(
            int(counter.get("zero_score_fallback", 0)),
            total,
        ),
    }


def _first_oracle_position(
    scored: Sequence[tuple[float, int, str, Mapping[str, object]]],
    oracle_actions: set[str],
) -> int | None:
    for index, item in enumerate(scored, start=1):
        action = _candidate_action(item[3])
        if action in oracle_actions:
            return index
    return None


def _candidate_action(row: Mapping[str, object]) -> str | None:
    return _optional_string(row.get("candidate_action"))


def _oracle_rank(row: Mapping[str, object]) -> int:
    return _int(row.get("oracle_rank"), default=999)


def _target_example(target: TargetEvidence) -> dict[str, object]:
    return {
        "group_key": target.group_key,
        "oracle_actions": list(target.oracle_actions),
        "seed": target.provenance.get("seed"),
        "source_kind": target.provenance.get("source_kind"),
        "record_index": target.provenance.get("record_index"),
        "agent_id": target.provenance.get("agent_id"),
    }


def _field_tokens(name: str) -> tuple[str, ...]:
    normalized = (
        str(name)
        .replace("=", ".")
        .replace("[", ".")
        .replace("]", "")
        .replace("/", ".")
    )
    return tuple(token for token in normalized.split(".") if token)


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _positive_int(value: object, *, field: str) -> int:
    parsed = _int(value, default=-1)
    if parsed <= 0:
        raise FirstRecoveryStateActionReadinessAuditError(f"{field} must be positive")
    return parsed


def _action_sort_key(action: str | None) -> int:
    if action is None:
        return len(ACTION_NAMES) + 1
    try:
        return ACTION_NAMES.index(action)
    except ValueError:
        return len(ACTION_NAMES) + 1
