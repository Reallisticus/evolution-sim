from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind import (
    carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training as v195,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response as v196,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v197_coverage_abstention_repair_design as v197,
)
from evolution_sim.mind.candidate_campaign import _float, _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.evaluation_harness import (
    _fixture_world,
    _mind_v3_policy,
    _run_world,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.transition_value_scorer import (
    TRANSITION_VALUE_HIGH_SPECIFICITY_SOURCE_KEY_CATEGORIES,
    TRANSITION_VALUE_LOW_SPECIFICITY_SOURCE_KEY_CATEGORIES,
    load_transition_value_scorer_artifact,
    transition_value_source_key_category,
)

M3_CARRION_SURVIVOR_CONTINUATION_V198_HIGH_SPECIFICITY_COVERAGE_OR_MODEL_CAPACITY_REPAIR_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V198_HIGH_SPECIFICITY_COVERAGE_OR_MODEL_CAPACITY_REPAIR_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair_v1"
)

DEFAULT_V197_REPORT_PATH = v197.DEFAULT_OUTPUT_PATH
DEFAULT_V195_ARTIFACT_PATH = v197.DEFAULT_V195_ARTIFACT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v198-carrion-survivor-continuation-high-specificity-coverage-or-model-capacity-repair.json"
)

EXPECTED_V197_REPORT_EXACT_DIGEST = (
    "8523d9cfcfbee2185ff638e463808818f29459d6ec4a51590072b15a3e341c0a"
)
EXPECTED_V197_ROUTE = v197.COVERAGE_MODEL_CAPACITY_ROUTE
EXPECTED_V196_REPORT_EXACT_DIGEST = v197.EXPECTED_V196_REPORT_EXACT_DIGEST
EXPECTED_V195_REPORT_EXACT_DIGEST = v197.EXPECTED_V195_REPORT_EXACT_DIGEST
EXPECTED_V195_ARTIFACT_DIGEST = v197.EXPECTED_V195_ARTIFACT_DIGEST
EXPECTED_V196_MECHANISM = v196.PRIMARY_COVERAGE_COLLAPSE_MECHANISM

HIGH_SPECIFICITY_CATEGORIES = frozenset(
    TRANSITION_VALUE_HIGH_SPECIFICITY_SOURCE_KEY_CATEGORIES
)
LOW_SPECIFICITY_CATEGORIES = frozenset(
    TRANSITION_VALUE_LOW_SPECIFICITY_SOURCE_KEY_CATEGORIES
)
CANDIDATE_CATEGORY_ORDER = (
    "exact_context_feature_hit",
    "self_nav_coarse_feature_hit",
    "self_nav_feature_hit",
    "nav_feature_hit",
    "self_coarse_feature_hit",
    "self_feature_hit",
    "coarse_feature_hit",
    "mask_only_hit",
    "global_hit",
)
MIN_HIGH_SPECIFICITY_SELECTED_SHARE_FOR_FUTURE_SLICE_4 = (
    v197.MIN_EXACT_OR_HIGH_SPECIFICITY_HIT_SHARE_FOR_SLICE_4
)

STOP_ROUTE = "stop_v198_source_pins_invalid_no_training"
INSTRUMENTATION_ROUTE = (
    "v199_high_specificity_probe_instrumentation_repair_no_training"
)
PUBLIC_MODEL_CAPACITY_ROUTE = "v199_public_masked_model_capacity_harness_no_training"
ACTION_COMPLETE_ROUTE = "v199_high_specificity_action_complete_contract_audit_no_training"
FUTURE_EXPLICIT_SLICE_4_ROUTE = (
    "v199_future_explicit_slice_4_training_after_high_specificity_probe_opt_in"
)


def run_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair(
    *,
    v197_report_path: str | Path = DEFAULT_V197_REPORT_PATH,
    v195_artifact_path: str | Path = DEFAULT_V195_ARTIFACT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v197_report_exact_digest: str = EXPECTED_V197_REPORT_EXACT_DIGEST,
    expected_v197_route: str = EXPECTED_V197_ROUTE,
    expected_v196_report_exact_digest: str = EXPECTED_V196_REPORT_EXACT_DIGEST,
    expected_v195_report_exact_digest: str = EXPECTED_V195_REPORT_EXACT_DIGEST,
    expected_v195_artifact_digest: str = EXPECTED_V195_ARTIFACT_DIGEST,
    run_diagnostic_replay: bool = True,
    probe_diagnostics_override: Mapping[str, object] | None = None,
    broad_seeds: Sequence[int] = v195.DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = v195.DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = v195.DEFAULT_TICKS,
) -> dict[str, object]:
    v197_report = load_json_report(v197_report_path)
    v195_artifact = load_json_report(v195_artifact_path)
    source_validation = validate_v198_source_pins(
        v197_report=v197_report,
        v195_artifact=v195_artifact,
        expected_v197_report_exact_digest=expected_v197_report_exact_digest,
        expected_v197_route=expected_v197_route,
        expected_v196_report_exact_digest=expected_v196_report_exact_digest,
        expected_v195_report_exact_digest=expected_v195_report_exact_digest,
        expected_v195_artifact_digest=expected_v195_artifact_digest,
    )
    artifact_table = artifact_table_category_counts(v195_artifact)
    if probe_diagnostics_override is not None:
        high_specificity_probe = dict(probe_diagnostics_override)
        high_specificity_probe["diagnostics_override_used"] = True
        high_specificity_probe["diagnostics_provenance"] = "override"
    elif source_validation.get("passed") is True and run_diagnostic_replay:
        high_specificity_probe = run_high_specificity_coverage_probe(
            artifact=v195_artifact,
            broad_seeds=broad_seeds,
            carrion_fixture_seeds=carrion_fixture_seeds,
            ticks=ticks,
        )
    else:
        high_specificity_probe = skipped_high_specificity_probe(
            "source_validation_failed"
            if source_validation.get("passed") is not True
            else "diagnostic_replay_skipped"
        )
    coverage_assessment = assess_high_specificity_coverage(
        artifact_table=artifact_table,
        probe=high_specificity_probe,
    )
    route_decision = route_decision_for_v198(
        source_validation=source_validation,
        probe=high_specificity_probe,
        coverage_assessment=coverage_assessment,
    )
    classification = classification_for_v198(
        source_validation=source_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V198_HIGH_SPECIFICITY_COVERAGE_OR_MODEL_CAPACITY_REPAIR_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V198_HIGH_SPECIFICITY_COVERAGE_OR_MODEL_CAPACITY_REPAIR_POLICY
        ),
        "contract": contract(
            expected_v197_report_exact_digest=expected_v197_report_exact_digest,
            expected_v197_route=expected_v197_route,
        ),
        "historical_evidence_checkpoint": historical_evidence_checkpoint(),
        "inputs": {
            "v197_report": str(v197_report_path),
            "v195_artifact": str(v195_artifact_path),
            "expected_v197_report_exact_digest": expected_v197_report_exact_digest,
            "expected_v197_route": expected_v197_route,
            "expected_v196_report_exact_digest": expected_v196_report_exact_digest,
            "expected_v195_report_exact_digest": expected_v195_report_exact_digest,
            "expected_v195_artifact_digest": expected_v195_artifact_digest,
            "run_diagnostic_replay": bool(run_diagnostic_replay),
            "broad_seeds": [int(seed) for seed in broad_seeds],
            "carrion_fixture_seeds": [int(seed) for seed in carrion_fixture_seeds],
            "ticks": int(ticks),
        },
        "source_pin_validation": source_validation,
        "artifact_table_category_counts": artifact_table,
        "high_specificity_probe": high_specificity_probe,
        "coverage_assessment": coverage_assessment,
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "high_specificity_coverage_probe",
                "model_capacity_contract_repair",
                "no_training",
                "no_slice_4_consumption",
                "no_runtime_integration",
                "no_gate_relaxation",
                "no_support_expansion",
                "no_promotion",
            ],
        },
        **lifecycle_flags(
            diagnostic_replay_ran=high_specificity_probe.get("ran") is True,
        ),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v198_source_pins(
    *,
    v197_report: Mapping[str, object],
    v195_artifact: Mapping[str, object],
    expected_v197_report_exact_digest: str,
    expected_v197_route: str,
    expected_v196_report_exact_digest: str,
    expected_v195_report_exact_digest: str,
    expected_v195_artifact_digest: str,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v197_report)
    source = _mapping(v197_report.get("source_pin_validation"))
    comparison = _mapping(v197_report.get("specificity_gate_comparison"))
    route = _mapping(v197_report.get("route_decision"))
    after = _mapping(v197_report.get("after_specificity_gate"))
    combined = _mapping(after.get("combined"))
    artifact_digest = stable_payload_digest(v195_artifact)
    checks = {
        "v197_report_schema_matches": (
            v197_report.get("schema_version")
            == v197.M3_CARRION_SURVIVOR_CONTINUATION_V197_COVERAGE_ABSTENTION_REPAIR_SCHEMA_VERSION
        ),
        "v197_report_policy_matches": (
            v197_report.get("policy")
            == v197.M3_CARRION_SURVIVOR_CONTINUATION_V197_COVERAGE_ABSTENTION_REPAIR_POLICY
        ),
        "v197_report_exact_digest_valid": exact_validation.get("passed") is True,
        "v197_report_exact_digest_matches_expected": (
            v197_report.get("exact_digest") == expected_v197_report_exact_digest
        ),
        "v197_selected_route_matches_expected": (
            route.get("selected_route") == expected_v197_route
            and route.get("recommended_next_route") == expected_v197_route
        ),
        "v197_source_pin_validation_passed": source.get("passed") is True,
        "v197_inherited_v196_digest_matches": (
            source.get("observed_v196_report_exact_digest")
            == expected_v196_report_exact_digest
        ),
        "v197_inherited_v195_report_digest_matches": (
            source.get("observed_v195_report_exact_digest")
            == expected_v195_report_exact_digest
        ),
        "v197_inherited_v195_artifact_digest_matches": (
            source.get("observed_v195_artifact_digest")
            == expected_v195_artifact_digest
        ),
        "v195_artifact_digest_matches_expected": (
            artifact_digest == expected_v195_artifact_digest
        ),
        "v197_low_specificity_collapse_blocked": (
            comparison.get("low_specificity_collapse_blocked") is True
        ),
        "v197_exact_high_specificity_hit_share_zero": (
            _float(comparison.get("exact_or_high_specificity_hit_share")) == 0.0
            and _int(comparison.get("exact_or_high_specificity_hit_count")) == 0
        ),
        "v197_real_replay_provenance": v197_probe_has_real_replay_provenance(after),
        "v197_mock_override_not_used": after.get("diagnostics_override_used") is False,
        "v197_gate_enabled": after.get("source_key_specificity_gate_enabled") is True,
        "v197_no_broad_regressions": (
            _int(comparison.get("after_broad_alive_birth_regression_count")) == 0
        ),
        "v197_heuristic_action_sources_zero": (
            _int(comparison.get("after_heuristic_action_source_count")) == 0
        ),
        "v197_dominant_requested_action_under_cap": (
            _float(comparison.get("after_dominant_requested_action_share")) <= 0.50
        ),
        "v197_override_applied_share_zero": (
            _float(combined.get("override_applied_share")) == 0.0
        ),
        "v197_no_training": v197_report.get("training_ran") is False,
        "v197_no_fit": v197_report.get("fit_ran") is False,
        "v197_no_training_artifact": (
            v197_report.get("training_artifact_created") is False
        ),
        "v197_no_slice_4_consumption": (
            v197_report.get("slice_4_training_consumed") is False
        ),
        "v197_no_runtime_artifact": (
            v197_report.get("runtime_artifact_created") is False
        ),
        "v197_no_runtime_integration": (
            v197_report.get("runtime_integration_ran") is False
        ),
        "v197_runtime_action_selection_unchanged": (
            v197_report.get("runtime_action_selection_changed") is False
        ),
        "v197_runtime_policy_unchanged": (
            v197_report.get("runtime_policy_changed") is False
        ),
        "v197_no_gate_relaxation": (
            v197_report.get("gate_relaxation_allowed") is False
            and v197_report.get("gate_relaxation_ran") is False
        ),
        "v197_no_support_expansion": (
            v197_report.get("support_generation_ran") is False
            and v197_report.get("support_expansion_ran") is False
        ),
        "v197_no_promotion": (
            v197_report.get("promotion_authorized") is False
            and v197_report.get("non_promoted") is True
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v198_source_pin_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v197_report_exact_digest": expected_v197_report_exact_digest,
        "observed_v197_report_exact_digest": v197_report.get("exact_digest"),
        "v197_report_exact_digest_validation": exact_validation,
        "expected_v197_route": expected_v197_route,
        "observed_v197_route": route.get("selected_route"),
        "expected_v196_report_exact_digest": expected_v196_report_exact_digest,
        "observed_v196_report_exact_digest": source.get(
            "observed_v196_report_exact_digest"
        ),
        "expected_v195_report_exact_digest": expected_v195_report_exact_digest,
        "observed_v195_report_exact_digest": source.get(
            "observed_v195_report_exact_digest"
        ),
        "expected_v195_artifact_digest": expected_v195_artifact_digest,
        "observed_v195_artifact_digest": artifact_digest,
        "reported_v197_v195_artifact_digest": source.get(
            "observed_v195_artifact_digest"
        ),
        "checks": checks,
    }


def artifact_table_category_counts(artifact: Mapping[str, object]) -> dict[str, object]:
    tables = _mapping(artifact.get("utility_tables"))
    feature_table = _mapping(tables.get("feature_action_utility"))
    category_counts: Counter[str] = Counter()
    action_count_counts: Counter[str] = Counter()
    for key, value in feature_table.items():
        category = transition_value_source_key_category(
            "feature_action_utility",
            key,
        )
        category_counts.update([category])
        action_stats = _mapping(value)
        observed_actions = sum(
            1
            for action in ACTION_NAMES
            if _int(_mapping(action_stats.get(action)).get("count")) > 0
        )
        action_count_counts.update([str(observed_actions)])
    high_count = sum(
        int(category_counts.get(category, 0)) for category in HIGH_SPECIFICITY_CATEGORIES
    )
    low_count = sum(
        int(category_counts.get(category, 0)) for category in LOW_SPECIFICITY_CATEGORIES
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v198_artifact_table_category_counts_v1",
        "feature_action_utility_key_count": len(feature_table),
        "source_key_category_counts": dict(sorted(category_counts.items())),
        "high_specificity_key_count": high_count,
        "low_specificity_key_count": low_count,
        "other_specificity_key_count": len(feature_table) - high_count - low_count,
        "high_specificity_categories_present": sorted(
            category
            for category in HIGH_SPECIFICITY_CATEGORIES
            if int(category_counts.get(category, 0)) > 0
        ),
        "low_specificity_categories_present": sorted(
            category
            for category in LOW_SPECIFICITY_CATEGORIES
            if int(category_counts.get(category, 0)) > 0
        ),
        "observed_action_count_per_key_counts": dict(sorted(action_count_counts.items())),
    }


def run_high_specificity_coverage_probe(
    *,
    artifact: Mapping[str, object],
    broad_seeds: Sequence[int],
    carrion_fixture_seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    scorer = load_transition_value_scorer_artifact(artifact)
    broad = _run_probe_scope(
        scope="broad",
        scorer=scorer,
        seeds=broad_seeds,
        ticks=ticks,
    )
    carrion = _run_probe_scope(
        scope="carrion_only",
        scorer=scorer,
        seeds=carrion_fixture_seeds,
        ticks=ticks,
    )
    combined = combine_probe_scopes(broad, carrion)
    return {
        "policy": "m3_carrion_survivor_continuation_v198_high_specificity_coverage_probe_v1",
        "ran": True,
        "diagnostics_only": True,
        "diagnostics_provenance": "real_replay",
        "diagnostics_override_used": False,
        "real_replay_provenance": True,
        "transition_value_action_override_enabled": False,
        "source_key_specificity_gate_enabled": True,
        "candidate_key_coverage_diagnostics_enabled": True,
        "training_rerun": False,
        "slice_4_training_consumed": False,
        "broad": broad,
        "carrion_only": carrion,
        "combined": combined,
    }


def _run_probe_scope(
    *,
    scope: str,
    scorer: object,
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    stats = _empty_probe_stats()
    per_seed: list[dict[str, object]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        baseline_policy = _mind_v3_policy(seed=seed, founder_template=None)
        probe_policy = _mind_v3_policy(
            seed=seed,
            founder_template=None,
            transition_value_scorer=scorer,
            transition_value_action_override=False,
            transition_value_source_key_specificity_gate_enabled=True,
            transition_value_candidate_key_coverage_diagnostics_enabled=True,
        )
        if scope == "broad":
            baseline_world = SimulationWorld(
                WorldConfig(seed=seed, max_ticks=int(ticks)),
                policy=baseline_policy,
            )
            probe_world = SimulationWorld(
                WorldConfig(seed=seed, max_ticks=int(ticks)),
                policy=probe_policy,
            )
        elif scope == "carrion_only":
            baseline_world = _fixture_world(
                fixture_name="carrion_only",
                seed=seed,
                ticks=int(ticks),
                policy=baseline_policy,
            )
            probe_world = _fixture_world(
                fixture_name="carrion_only",
                seed=seed,
                ticks=int(ticks),
                policy=probe_policy,
            )
        else:
            raise ValueError(f"unsupported v198 probe scope: {scope}")
        baseline = _run_world(
            world=baseline_world,
            seed=seed,
            ticks=int(ticks),
            trajectory_output_path=None,
            trajectory_split_id=f"v198_{scope}_baseline",
        )
        probe = _run_world(
            world=probe_world,
            seed=seed,
            ticks=int(ticks),
            trajectory_output_path=None,
            trajectory_split_id=f"v198_{scope}_probe",
        )
        seed_stats = _probe_stats_from_decisions(
            probe_world.policy_decision_diagnostics_records
        )
        _merge_run_stats(seed_stats, probe)
        _merge_probe_stats(stats, seed_stats)
        alive_delta = _int(probe.get("alive_agents")) - _int(
            baseline.get("alive_agents")
        )
        births_delta = _int(probe.get("births")) - _int(baseline.get("births"))
        regression = alive_delta < 0 or births_delta < 0
        if regression:
            stats["alive_birth_regression_count"] = (
                _int(stats.get("alive_birth_regression_count")) + 1
            )
        per_seed.append(
            {
                "seed": seed,
                "baseline_alive_agents": baseline.get("alive_agents"),
                "probe_alive_agents": probe.get("alive_agents"),
                "alive_agents_delta": alive_delta,
                "baseline_births": baseline.get("births"),
                "probe_births": probe.get("births"),
                "births_delta": births_delta,
                "alive_or_birth_regression": regression,
                "probe_heuristic_action_source_count": probe.get(
                    "heuristic_action_source_count"
                ),
                "probe_requested_action_counts": probe.get("requested_action_counts"),
                "probe_dominant_requested_action": probe.get(
                    "dominant_requested_action"
                ),
                "probe_dominant_requested_action_share": probe.get(
                    "dominant_requested_action_share"
                ),
                "coverage": _finalize_probe_stats(seed_stats),
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v198_high_specificity_scope_probe_v1",
        "scope": scope,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "per_seed": per_seed,
        "per_seed_alive_birth_regressions": [
            item for item in per_seed if item["alive_or_birth_regression"]
        ],
        **_finalize_probe_stats(stats),
    }


def _probe_stats_from_decisions(
    decision_diagnostics: Sequence[Mapping[str, object] | None],
) -> dict[str, object]:
    stats = _empty_probe_stats()
    for diagnostic in decision_diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        transition = _mapping(diagnostic.get("transition_value_scorer"))
        if transition:
            _update_probe_stats(stats, transition)
    return stats


def _empty_probe_stats() -> dict[str, object]:
    return {
        "decision_count": 0,
        "candidate_key_coverage_decision_count": 0,
        "missing_candidate_key_coverage_count": 0,
        "supported_score_count": 0,
        "observed_support_floor_satisfied_count": 0,
        "clear_best_count": 0,
        "runtime_action_selection_changed_count": 0,
        "would_change_action_count": 0,
        "high_specificity_any_key_present_count": 0,
        "high_specificity_any_complete_count": 0,
        "high_specificity_any_observed_support_floor_satisfied_count": 0,
        "high_specificity_selected_source_count": 0,
        "low_specificity_selected_source_count": 0,
        "other_specificity_selected_source_count": 0,
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": 0,
        "alive_birth_regression_count": 0,
        "score_source_counts": Counter(),
        "selected_source_key_category_counts": Counter(),
        "v197_specificity_gate_rejected_reason_counts": Counter(),
        "predicted_action_counts": Counter(),
        "final_requested_action_counts": Counter(),
        "requested_action_counts": Counter(),
        "original_mind_v3_requested_action_counts": Counter(),
        "candidate_key_category_stats": defaultdict(_empty_category_stats),
    }


def _empty_category_stats() -> dict[str, object]:
    return {
        "candidate_key_evaluated_count": 0,
        "key_present_count": 0,
        "key_absent_count": 0,
        "complete_for_current_valid_actions_count": 0,
        "present_but_action_incomplete_count": 0,
        "observed_support_floor_satisfied_count": 0,
        "present_complete_but_observed_support_floor_failed_count": 0,
        "imputed_valid_action_score_count": 0,
        "present_complete_floor_satisfied_but_unclear_best_count": 0,
        "selected_source_category_count": 0,
        "selected_predicted_action_counts": Counter(),
        "selected_final_requested_action_counts": Counter(),
        "selected_rejected_reason_counts": Counter(),
    }


def _update_probe_stats(
    stats: dict[str, object],
    transition: Mapping[str, object],
) -> None:
    stats["decision_count"] = _int(stats.get("decision_count")) + 1
    score_source = str(transition.get("score_source") or "")
    source_category = str(transition.get("source_key_category") or "")
    if not source_category:
        source_category = transition_value_source_key_category(
            score_source,
            transition.get("source_key"),
        )
    _counter(stats, "score_source_counts").update([score_source or "unknown"])
    _counter(stats, "selected_source_key_category_counts").update([source_category])
    final_requested = str(
        transition.get("final_requested_action")
        or transition.get("requested_action")
        or ""
    )
    original_requested = str(transition.get("original_mind_v3_requested_action") or "")
    predicted_action = str(transition.get("predicted_action") or "")
    if final_requested:
        _counter(stats, "final_requested_action_counts").update([final_requested])
    if original_requested:
        _counter(stats, "original_mind_v3_requested_action_counts").update(
            [original_requested]
        )
        _counter(stats, "requested_action_counts").update([original_requested])
    if predicted_action:
        _counter(stats, "predicted_action_counts").update([predicted_action])
    if transition.get("supported_scores_for_all_valid_actions") is True:
        stats["supported_score_count"] = _int(stats.get("supported_score_count")) + 1
    if (
        transition.get("observed_support_floor_satisfied_for_all_valid_actions")
        is True
    ):
        stats["observed_support_floor_satisfied_count"] = (
            _int(stats.get("observed_support_floor_satisfied_count")) + 1
        )
    if transition.get("clear_best_valid_action") is True:
        stats["clear_best_count"] = _int(stats.get("clear_best_count")) + 1
    if transition.get("runtime_action_selection_changed") is True:
        stats["runtime_action_selection_changed_count"] = (
            _int(stats.get("runtime_action_selection_changed_count")) + 1
        )
    if transition.get("would_change_action") is True:
        stats["would_change_action_count"] = (
            _int(stats.get("would_change_action_count")) + 1
        )
    if source_category in HIGH_SPECIFICITY_CATEGORIES:
        stats["high_specificity_selected_source_count"] = (
            _int(stats.get("high_specificity_selected_source_count")) + 1
        )
    elif source_category in LOW_SPECIFICITY_CATEGORIES:
        stats["low_specificity_selected_source_count"] = (
            _int(stats.get("low_specificity_selected_source_count")) + 1
        )
    else:
        stats["other_specificity_selected_source_count"] = (
            _int(stats.get("other_specificity_selected_source_count")) + 1
        )
    gate_reason = v197_specificity_gate_rejected_reason(transition)
    _counter(stats, "v197_specificity_gate_rejected_reason_counts").update(
        [gate_reason or "would_pass"]
    )
    candidate_key_coverage = transition.get("candidate_key_coverage")
    if not isinstance(candidate_key_coverage, list):
        stats["missing_candidate_key_coverage_count"] = (
            _int(stats.get("missing_candidate_key_coverage_count")) + 1
        )
        return
    stats["candidate_key_coverage_decision_count"] = (
        _int(stats.get("candidate_key_coverage_decision_count")) + 1
    )
    high_candidates = []
    selected_category_recorded = False
    for item in candidate_key_coverage:
        if not isinstance(item, Mapping):
            continue
        category = str(item.get("source_key_category") or "")
        if not category:
            continue
        category_stats = _category_stats(stats, category)
        category_stats["candidate_key_evaluated_count"] = (
            _int(category_stats.get("candidate_key_evaluated_count")) + 1
        )
        key_present = item.get("key_present") is True
        complete = item.get("complete_for_current_valid_actions") is True
        floor_satisfied = (
            item.get("observed_support_floor_satisfied_for_all_valid_actions")
            is True
        )
        clear_best = item.get("clear_best_valid_action") is True
        if key_present:
            category_stats["key_present_count"] = (
                _int(category_stats.get("key_present_count")) + 1
            )
        else:
            category_stats["key_absent_count"] = (
                _int(category_stats.get("key_absent_count")) + 1
            )
        if complete:
            category_stats["complete_for_current_valid_actions_count"] = (
                _int(category_stats.get("complete_for_current_valid_actions_count"))
                + 1
            )
        elif key_present:
            category_stats["present_but_action_incomplete_count"] = (
                _int(category_stats.get("present_but_action_incomplete_count")) + 1
            )
        if floor_satisfied:
            category_stats["observed_support_floor_satisfied_count"] = (
                _int(category_stats.get("observed_support_floor_satisfied_count"))
                + 1
            )
        elif key_present and complete:
            category_stats[
                "present_complete_but_observed_support_floor_failed_count"
            ] = (
                _int(
                    category_stats.get(
                        "present_complete_but_observed_support_floor_failed_count"
                    )
                )
                + 1
            )
        if item.get("has_imputed_valid_action_score") is True:
            category_stats["imputed_valid_action_score_count"] = (
                _int(category_stats.get("imputed_valid_action_score_count")) + 1
            )
        if key_present and complete and floor_satisfied and not clear_best:
            category_stats[
                "present_complete_floor_satisfied_but_unclear_best_count"
            ] = (
                _int(
                    category_stats.get(
                        "present_complete_floor_satisfied_but_unclear_best_count"
                    )
                )
                + 1
            )
        if category in HIGH_SPECIFICITY_CATEGORIES:
            high_candidates.append(item)
        if category == source_category and not selected_category_recorded:
            selected_category_recorded = True
            category_stats["selected_source_category_count"] = (
                _int(category_stats.get("selected_source_category_count")) + 1
            )
            _counter(category_stats, "selected_rejected_reason_counts").update(
                [gate_reason or "would_pass"]
            )
            if predicted_action:
                _counter(category_stats, "selected_predicted_action_counts").update(
                    [predicted_action]
                )
            if final_requested:
                _counter(
                    category_stats,
                    "selected_final_requested_action_counts",
                ).update([final_requested])
    if any(item.get("key_present") is True for item in high_candidates):
        stats["high_specificity_any_key_present_count"] = (
            _int(stats.get("high_specificity_any_key_present_count")) + 1
        )
    if any(
        item.get("complete_for_current_valid_actions") is True
        for item in high_candidates
    ):
        stats["high_specificity_any_complete_count"] = (
            _int(stats.get("high_specificity_any_complete_count")) + 1
        )
    if any(
        item.get("observed_support_floor_satisfied_for_all_valid_actions") is True
        for item in high_candidates
    ):
        stats["high_specificity_any_observed_support_floor_satisfied_count"] = (
            _int(
                stats.get("high_specificity_any_observed_support_floor_satisfied_count")
            )
            + 1
        )


def v197_specificity_gate_rejected_reason(
    transition: Mapping[str, object],
) -> str | None:
    predicted_action = transition.get("predicted_action")
    score = _mapping(transition.get("score"))
    valid_actions = tuple(
        str(action) for action in score.get("valid_actions", [])
    )
    if transition.get("supported_scores_for_all_valid_actions") is not True:
        return "missing_supported_scores_for_valid_actions"
    if transition.get("has_imputed_valid_action_score") is True:
        return "imputed_valid_action_score"
    if (
        transition.get("observed_support_floor_satisfied_for_all_valid_actions")
        is not True
    ):
        return "low_observed_support_for_valid_actions"
    if transition.get("clear_best_valid_action") is not True:
        return "no_clear_best_valid_action"
    if not isinstance(predicted_action, str) or not predicted_action:
        return "no_prediction"
    if predicted_action not in set(valid_actions):
        return "invalid_prediction"
    category = str(transition.get("source_key_category") or "")
    allowed = set(
        str(item)
        for item in transition.get("source_key_specificity_allowed_categories", [])
    )
    if (
        transition.get("source_key_specificity_gate_enabled") is True
        and category not in allowed
    ):
        return "low_specificity_feature_key"
    return None


def _merge_run_stats(stats: dict[str, object], run: Mapping[str, object]) -> None:
    stats["heuristic_action_source_count"] = _int(
        stats.get("heuristic_action_source_count")
    ) + _int(run.get("heuristic_action_source_count"))
    stats["unsupported_requested_action_count"] = _int(
        stats.get("unsupported_requested_action_count")
    ) + _int(run.get("unsupported_requested_action_count"))
    stats["unsupported_resolved_action_count"] = _int(
        stats.get("unsupported_resolved_action_count")
    ) + _int(run.get("unsupported_resolved_action_count"))
    _counter(stats, "requested_action_counts").update(
        _int_counter(run.get("requested_action_counts"))
    )


def _merge_probe_stats(target: dict[str, object], source: Mapping[str, object]) -> None:
    for key in (
        "decision_count",
        "candidate_key_coverage_decision_count",
        "missing_candidate_key_coverage_count",
        "supported_score_count",
        "observed_support_floor_satisfied_count",
        "clear_best_count",
        "runtime_action_selection_changed_count",
        "would_change_action_count",
        "high_specificity_any_key_present_count",
        "high_specificity_any_complete_count",
        "high_specificity_any_observed_support_floor_satisfied_count",
        "high_specificity_selected_source_count",
        "low_specificity_selected_source_count",
        "other_specificity_selected_source_count",
        "heuristic_action_source_count",
        "unsupported_requested_action_count",
        "unsupported_resolved_action_count",
        "alive_birth_regression_count",
    ):
        target[key] = _int(target.get(key)) + _int(source.get(key))
    for key in (
        "score_source_counts",
        "selected_source_key_category_counts",
        "v197_specificity_gate_rejected_reason_counts",
        "predicted_action_counts",
        "final_requested_action_counts",
        "requested_action_counts",
        "original_mind_v3_requested_action_counts",
    ):
        _counter(target, key).update(_int_counter(source.get(key)))
    source_categories = source.get("candidate_key_category_stats")
    if isinstance(source_categories, Mapping):
        for category, source_stats in source_categories.items():
            target_stats = _category_stats(target, str(category))
            _merge_category_stats(target_stats, _mapping(source_stats))


def combine_probe_scopes(*scopes: Mapping[str, object]) -> dict[str, object]:
    stats = _empty_probe_stats()
    for scope in scopes:
        _merge_probe_payload(stats, scope)
    return {
        "policy": "m3_carrion_survivor_continuation_v198_combined_high_specificity_probe_v1",
        **_finalize_probe_stats(stats),
    }


def _merge_probe_payload(target: dict[str, object], source: Mapping[str, object]) -> None:
    payload = {
        "decision_count": source.get("decision_count"),
        "candidate_key_coverage_decision_count": source.get(
            "candidate_key_coverage_decision_count"
        ),
        "missing_candidate_key_coverage_count": source.get(
            "missing_candidate_key_coverage_count"
        ),
        "supported_score_count": source.get("supported_score_count"),
        "observed_support_floor_satisfied_count": source.get(
            "observed_support_floor_satisfied_count"
        ),
        "clear_best_count": source.get("clear_best_count"),
        "runtime_action_selection_changed_count": source.get(
            "runtime_action_selection_changed_count"
        ),
        "would_change_action_count": source.get("would_change_action_count"),
        "high_specificity_any_key_present_count": source.get(
            "high_specificity_any_key_present_count"
        ),
        "high_specificity_any_complete_count": source.get(
            "high_specificity_any_complete_count"
        ),
        "high_specificity_any_observed_support_floor_satisfied_count": source.get(
            "high_specificity_any_observed_support_floor_satisfied_count"
        ),
        "high_specificity_selected_source_count": source.get(
            "high_specificity_selected_source_count"
        ),
        "low_specificity_selected_source_count": source.get(
            "low_specificity_selected_source_count"
        ),
        "other_specificity_selected_source_count": source.get(
            "other_specificity_selected_source_count"
        ),
        "heuristic_action_source_count": source.get("heuristic_action_source_count"),
        "unsupported_requested_action_count": source.get(
            "unsupported_requested_action_count"
        ),
        "unsupported_resolved_action_count": source.get(
            "unsupported_resolved_action_count"
        ),
        "alive_birth_regression_count": source.get("alive_birth_regression_count"),
        "score_source_counts": source.get("score_source_counts"),
        "selected_source_key_category_counts": source.get(
            "selected_source_key_category_counts"
        ),
        "v197_specificity_gate_rejected_reason_counts": source.get(
            "v197_specificity_gate_rejected_reason_counts"
        ),
        "predicted_action_counts": source.get("predicted_action_counts"),
        "final_requested_action_counts": source.get("final_requested_action_counts"),
        "requested_action_counts": source.get("requested_action_counts"),
        "original_mind_v3_requested_action_counts": source.get(
            "original_mind_v3_requested_action_counts"
        ),
        "candidate_key_category_stats": (
            source.get("candidate_key_category_stats")
            if isinstance(source.get("candidate_key_category_stats"), Mapping)
            else source.get("candidate_key_category_breakdown")
        ),
    }
    _merge_probe_stats(target, payload)


def _finalize_probe_stats(stats: Mapping[str, object]) -> dict[str, object]:
    decision_count = _int(stats.get("decision_count"))
    selected_categories = _int_counter(stats.get("selected_source_key_category_counts"))
    requested_counts = _int_counter(stats.get("requested_action_counts"))
    final_counts = _int_counter(stats.get("final_requested_action_counts"))
    predicted_counts = _int_counter(stats.get("predicted_action_counts"))
    high_present_count = _int(stats.get("high_specificity_any_key_present_count"))
    high_complete_count = _int(stats.get("high_specificity_any_complete_count"))
    high_floor_count = _int(
        stats.get("high_specificity_any_observed_support_floor_satisfied_count")
    )
    high_selected_count = _int(stats.get("high_specificity_selected_source_count"))
    return {
        "decision_count": decision_count,
        "candidate_key_coverage_decision_count": _int(
            stats.get("candidate_key_coverage_decision_count")
        ),
        "missing_candidate_key_coverage_count": _int(
            stats.get("missing_candidate_key_coverage_count")
        ),
        "supported_score_count": _int(stats.get("supported_score_count")),
        "supported_score_share": _share(
            _int(stats.get("supported_score_count")),
            decision_count,
        ),
        "observed_support_floor_satisfied_count": _int(
            stats.get("observed_support_floor_satisfied_count")
        ),
        "observed_support_floor_satisfied_share": _share(
            _int(stats.get("observed_support_floor_satisfied_count")),
            decision_count,
        ),
        "clear_best_count": _int(stats.get("clear_best_count")),
        "runtime_action_selection_changed_count": _int(
            stats.get("runtime_action_selection_changed_count")
        ),
        "would_change_action_count": _int(stats.get("would_change_action_count")),
        "high_specificity_any_key_present_count": high_present_count,
        "high_specificity_any_key_present_share": _share(
            high_present_count,
            decision_count,
        ),
        "high_specificity_any_complete_count": high_complete_count,
        "high_specificity_any_complete_share": _share(
            high_complete_count,
            decision_count,
        ),
        "high_specificity_any_observed_support_floor_satisfied_count": high_floor_count,
        "high_specificity_any_observed_support_floor_satisfied_share": _share(
            high_floor_count,
            decision_count,
        ),
        "high_specificity_selected_source_count": high_selected_count,
        "high_specificity_selected_source_share": _share(
            high_selected_count,
            decision_count,
        ),
        "low_specificity_selected_source_count": _int(
            stats.get("low_specificity_selected_source_count")
        ),
        "low_specificity_selected_source_share": _share(
            _int(stats.get("low_specificity_selected_source_count")),
            decision_count,
        ),
        "other_specificity_selected_source_count": _int(
            stats.get("other_specificity_selected_source_count")
        ),
        "other_specificity_selected_source_share": _share(
            _int(stats.get("other_specificity_selected_source_count")),
            decision_count,
        ),
        "heuristic_action_source_count": _int(
            stats.get("heuristic_action_source_count")
        ),
        "unsupported_requested_action_count": _int(
            stats.get("unsupported_requested_action_count")
        ),
        "unsupported_resolved_action_count": _int(
            stats.get("unsupported_resolved_action_count")
        ),
        "alive_birth_regression_count": _int(
            stats.get("alive_birth_regression_count")
        ),
        "score_source_counts": dict(
            sorted(_int_counter(stats.get("score_source_counts")).items())
        ),
        "selected_source_key_category_counts": dict(sorted(selected_categories.items())),
        "v197_specificity_gate_rejected_reason_counts": dict(
            sorted(
                _int_counter(
                    stats.get("v197_specificity_gate_rejected_reason_counts")
                ).items()
            )
        ),
        "predicted_action_counts": dict(sorted(predicted_counts.items())),
        "dominant_predicted_action": _dominant(predicted_counts)["action"],
        "dominant_predicted_action_share": _dominant(predicted_counts)["share"],
        "requested_action_counts": dict(sorted(requested_counts.items())),
        "dominant_requested_action": _dominant(requested_counts)["action"],
        "dominant_requested_action_share": _dominant(requested_counts)["share"],
        "final_requested_action_counts": dict(sorted(final_counts.items())),
        "dominant_final_requested_action": _dominant(final_counts)["action"],
        "dominant_final_requested_action_share": _dominant(final_counts)["share"],
        "original_mind_v3_requested_action_counts": dict(
            sorted(
                _int_counter(
                    stats.get("original_mind_v3_requested_action_counts")
                ).items()
            )
        ),
        "candidate_key_category_breakdown": _finalize_category_breakdown(
            _mapping(stats.get("candidate_key_category_stats")),
            decision_count=decision_count,
        ),
    }


def _finalize_category_breakdown(
    category_stats: Mapping[str, object],
    *,
    decision_count: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for category in CANDIDATE_CATEGORY_ORDER:
        stats = _mapping(category_stats.get(category))
        result[category] = _finalize_category_stats(stats, decision_count=decision_count)
    for category, stats in sorted(category_stats.items()):
        if str(category) in result:
            continue
        result[str(category)] = _finalize_category_stats(
            _mapping(stats),
            decision_count=decision_count,
        )
    return result


def _finalize_category_stats(
    stats: Mapping[str, object],
    *,
    decision_count: int,
) -> dict[str, object]:
    present_count = _int(stats.get("key_present_count"))
    absent_count = _int(stats.get("key_absent_count"))
    complete_count = _int(stats.get("complete_for_current_valid_actions_count"))
    incomplete_count = _int(stats.get("present_but_action_incomplete_count"))
    floor_count = _int(stats.get("observed_support_floor_satisfied_count"))
    floor_failed_count = _int(
        stats.get("present_complete_but_observed_support_floor_failed_count")
    )
    imputed_count = _int(stats.get("imputed_valid_action_score_count"))
    unclear_best_count = _int(
        stats.get("present_complete_floor_satisfied_but_unclear_best_count")
    )
    selected_count = _int(stats.get("selected_source_category_count"))
    return {
        "candidate_key_evaluated_count": _int(
            stats.get("candidate_key_evaluated_count")
        ),
        "key_present_count": present_count,
        "key_present_share": _share(present_count, decision_count),
        "key_absent_count": absent_count,
        "key_absent_share": _share(absent_count, decision_count),
        "complete_for_current_valid_actions_count": complete_count,
        "complete_for_current_valid_actions_share": _share(
            complete_count,
            decision_count,
        ),
        "present_but_action_incomplete_count": incomplete_count,
        "present_but_action_incomplete_share": _share(
            incomplete_count,
            decision_count,
        ),
        "observed_support_floor_satisfied_count": floor_count,
        "observed_support_floor_satisfied_share": _share(
            floor_count,
            decision_count,
        ),
        "present_complete_but_observed_support_floor_failed_count": (
            floor_failed_count
        ),
        "present_complete_but_observed_support_floor_failed_share": _share(
            floor_failed_count,
            decision_count,
        ),
        "imputed_valid_action_score_count": imputed_count,
        "imputed_valid_action_score_share": _share(imputed_count, decision_count),
        "present_complete_floor_satisfied_but_unclear_best_count": (
            unclear_best_count
        ),
        "present_complete_floor_satisfied_but_unclear_best_share": _share(
            unclear_best_count,
            decision_count,
        ),
        "selected_source_category_count": selected_count,
        "selected_source_category_share": _share(selected_count, decision_count),
        "selected_predicted_action_counts": dict(
            sorted(_int_counter(stats.get("selected_predicted_action_counts")).items())
        ),
        "selected_final_requested_action_counts": dict(
            sorted(
                _int_counter(
                    stats.get("selected_final_requested_action_counts")
                ).items()
            )
        ),
        "selected_rejected_reason_counts": dict(
            sorted(_int_counter(stats.get("selected_rejected_reason_counts")).items())
        ),
    }


def _merge_category_stats(
    target: dict[str, object],
    source: Mapping[str, object],
) -> None:
    for key in (
        "candidate_key_evaluated_count",
        "key_present_count",
        "key_absent_count",
        "complete_for_current_valid_actions_count",
        "present_but_action_incomplete_count",
        "observed_support_floor_satisfied_count",
        "present_complete_but_observed_support_floor_failed_count",
        "imputed_valid_action_score_count",
        "present_complete_floor_satisfied_but_unclear_best_count",
        "selected_source_category_count",
    ):
        target[key] = _int(target.get(key)) + _int(source.get(key))
    for key in (
        "selected_predicted_action_counts",
        "selected_final_requested_action_counts",
        "selected_rejected_reason_counts",
    ):
        _counter(target, key).update(_int_counter(source.get(key)))


def assess_high_specificity_coverage(
    *,
    artifact_table: Mapping[str, object],
    probe: Mapping[str, object],
) -> dict[str, object]:
    combined = _mapping(probe.get("combined"))
    high_breakdown = _combined_high_specificity_breakdown(combined)
    high_present_share = _float(combined.get("high_specificity_any_key_present_share"))
    high_complete_share = _float(combined.get("high_specificity_any_complete_share"))
    high_floor_share = _float(
        combined.get("high_specificity_any_observed_support_floor_satisfied_share")
    )
    high_selected_share = _float(
        combined.get("high_specificity_selected_source_share")
    )
    blocker = "probe_not_real_replay"
    if probe_has_real_replay_provenance(probe):
        if _int(combined.get("missing_candidate_key_coverage_count")) > 0:
            blocker = "scorer_or_instrumentation_candidate_coverage_missing"
        elif _int(combined.get("runtime_action_selection_changed_count")) > 0:
            blocker = "probe_changed_runtime_action_selection"
        elif high_present_share == 0.0:
            blocker = "absent_high_specificity_public_states"
        elif _int(high_breakdown.get("present_but_action_incomplete_count")) > 0:
            blocker = "high_specificity_action_incomplete"
        elif _int(
            high_breakdown.get(
                "present_complete_but_observed_support_floor_failed_count"
            )
        ) > 0:
            blocker = "high_specificity_observed_support_floor_failure"
        elif _int(high_breakdown.get("imputed_valid_action_score_count")) > 0:
            blocker = "high_specificity_imputed_valid_action_score"
        elif _int(
            high_breakdown.get(
                "present_complete_floor_satisfied_but_unclear_best_count"
            )
        ) > 0:
            blocker = "high_specificity_unclear_best_action"
        elif high_selected_share < MIN_HIGH_SPECIFICITY_SELECTED_SHARE_FOR_FUTURE_SLICE_4:
            blocker = "high_specificity_live_overlap_below_slice_4_floor"
        else:
            blocker = "high_specificity_coverage_unexpectedly_sufficient"
    return {
        "policy": "m3_carrion_survivor_continuation_v198_high_specificity_coverage_assessment_v1",
        "probe_real_replay_provenance": probe_has_real_replay_provenance(probe),
        "artifact_has_high_specificity_keys": (
            _int(artifact_table.get("high_specificity_key_count")) > 0
        ),
        "artifact_high_specificity_key_count": artifact_table.get(
            "high_specificity_key_count"
        ),
        "artifact_high_specificity_categories_present": artifact_table.get(
            "high_specificity_categories_present"
        ),
        "live_high_specificity_any_key_present_share": high_present_share,
        "live_high_specificity_any_complete_share": high_complete_share,
        "live_high_specificity_any_observed_support_floor_satisfied_share": (
            high_floor_share
        ),
        "live_high_specificity_selected_source_share": high_selected_share,
        "live_high_specificity_candidate_breakdown": high_breakdown,
        "minimum_high_specificity_selected_source_share_for_future_slice_4": (
            MIN_HIGH_SPECIFICITY_SELECTED_SHARE_FOR_FUTURE_SLICE_4
        ),
        "primary_blocker": blocker,
        "coverage_sufficient_for_future_explicit_slice_4_route": (
            blocker == "high_specificity_coverage_unexpectedly_sufficient"
        ),
    }


def route_decision_for_v198(
    *,
    source_validation: Mapping[str, object],
    probe: Mapping[str, object],
    coverage_assessment: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif not probe_has_real_replay_provenance(probe):
        route = INSTRUMENTATION_ROUTE
    else:
        blocker = str(coverage_assessment.get("primary_blocker") or "")
        if blocker == "absent_high_specificity_public_states":
            route = PUBLIC_MODEL_CAPACITY_ROUTE
        elif blocker == "high_specificity_live_overlap_below_slice_4_floor":
            route = PUBLIC_MODEL_CAPACITY_ROUTE
        elif blocker in {
            "high_specificity_action_incomplete",
            "high_specificity_observed_support_floor_failure",
            "high_specificity_imputed_valid_action_score",
            "high_specificity_unclear_best_action",
        }:
            route = ACTION_COMPLETE_ROUTE
        elif blocker == "high_specificity_coverage_unexpectedly_sufficient":
            route = FUTURE_EXPLICIT_SLICE_4_ROUTE
        else:
            route = INSTRUMENTATION_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v198_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "source_pins_valid": source_validation.get("passed") is True,
        "real_replay_provenance": probe_has_real_replay_provenance(probe),
        "primary_blocker": coverage_assessment.get("primary_blocker"),
        "future_explicit_slice_4_training_route_authorized": (
            route == FUTURE_EXPLICIT_SLICE_4_ROUTE
        ),
        "direct_slice_4_training_allowed": False,
        "slice_4_training_consumed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "gate_relaxation_allowed": False,
        "support_expansion_allowed": False,
        "promotion_authorized": False,
        "rationale": route_rationale(route),
    }


def classification_for_v198(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v198_"
    if source_validation.get("passed") is not True:
        return prefix + "high_specificity_coverage_source_invalid_closed_no_training"
    route = str(route_decision.get("selected_route") or "")
    if route == PUBLIC_MODEL_CAPACITY_ROUTE:
        return (
            prefix
            + "high_specificity_coverage_gap_routes_to_public_model_capacity_harness_no_training"
        )
    if route == ACTION_COMPLETE_ROUTE:
        return (
            prefix
            + "high_specificity_action_complete_or_support_floor_gap_routes_to_contract_audit_no_training"
        )
    if route == FUTURE_EXPLICIT_SLICE_4_ROUTE:
        return (
            prefix
            + "high_specificity_coverage_sufficient_routes_to_future_explicit_slice_4_opt_in_no_training"
        )
    if route == INSTRUMENTATION_ROUTE:
        return prefix + "high_specificity_probe_instrumentation_repair_no_training"
    return prefix + "high_specificity_coverage_closed_no_training"


def skipped_high_specificity_probe(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v198_high_specificity_coverage_probe_v1",
        "ran": False,
        "reason": reason,
        "diagnostics_provenance": "skipped",
        "diagnostics_override_used": False,
        "real_replay_provenance": False,
        "transition_value_action_override_enabled": False,
        "source_key_specificity_gate_enabled": True,
        "candidate_key_coverage_diagnostics_enabled": False,
        "training_rerun": False,
        "slice_4_training_consumed": False,
    }


def probe_has_real_replay_provenance(probe: Mapping[str, object]) -> bool:
    return (
        probe.get("ran") is True
        and probe.get("policy")
        == "m3_carrion_survivor_continuation_v198_high_specificity_coverage_probe_v1"
        and probe.get("diagnostics_provenance") == "real_replay"
        and probe.get("diagnostics_override_used") is False
        and probe.get("diagnostics_only") is True
        and probe.get("transition_value_action_override_enabled") is False
        and probe.get("source_key_specificity_gate_enabled") is True
        and probe.get("candidate_key_coverage_diagnostics_enabled") is True
        and probe.get("training_rerun") is False
        and probe.get("slice_4_training_consumed") is False
    )


def v197_probe_has_real_replay_provenance(after: Mapping[str, object]) -> bool:
    return (
        after.get("ran") is True
        and after.get("policy")
        == "m3_carrion_survivor_continuation_v197_specificity_gate_diagnostic_replay_v1"
        and after.get("diagnostics_provenance") == "real_replay"
        and after.get("diagnostics_override_used") is False
        and after.get("source_key_specificity_gate_enabled") is True
        and after.get("diagnostics_only") is True
        and after.get("training_rerun") is False
        and after.get("slice_4_training_consumed") is False
    )


def contract(
    *,
    expected_v197_report_exact_digest: str,
    expected_v197_route: str,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_allowed": False,
        "slice_4_training_consumption_allowed": False,
        "runtime_artifact_creation_allowed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "runtime_policy_change_allowed": False,
        "gate_relaxation_allowed": False,
        "support_generation_allowed": False,
        "support_expansion_allowed": False,
        "promotion_authorized": False,
        "requires_v197_report_digest": expected_v197_report_exact_digest,
        "requires_v197_route": expected_v197_route,
        "diagnostic_seed_policy": (
            "broad and carrion_only seeds are campaign-target diagnostic/red-team "
            "surfaces, not clean promotion-heldout evidence"
        ),
        "probe_policy": {
            "frozen_artifact": "v195 transition-value scorer artifact",
            "source_key_specificity_gate_enabled": True,
            "transition_value_action_override_enabled": False,
            "observes_ordered_candidate_key_coverage": True,
            "default_runtime_behavior_changed": False,
            "public_policy_visible_inputs_only": True,
        },
        "route_logic": {
            "source_pin_failure": STOP_ROUTE,
            "probe_not_real_replay": INSTRUMENTATION_ROUTE,
            "high_specificity_public_state_absent": PUBLIC_MODEL_CAPACITY_ROUTE,
            "high_specificity_action_incomplete_or_support_floor_failure": (
                ACTION_COMPLETE_ROUTE
            ),
            "future_training_requires_separate_explicit_opt_in": (
                FUTURE_EXPLICIT_SLICE_4_ROUTE
            ),
        },
    }


def historical_evidence_checkpoint() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v198_historical_evidence_checkpoint_v1",
        "v53_v63": (
            "scalar IQL coefficient/prior/action-distribution/global-bias/"
            "extraction tuning is closed"
        ),
        "v154_v176": (
            "tiny support-archive / nearest-neighbor scorer loop is strategically "
            "saturated"
        ),
        "v180_v186_v195": (
            "training slices failed and are not promotion evidence"
        ),
        "v191_v193": (
            "same-tick occupancy drift was repaired as a support-evidence contract "
            "issue, not illegal script requests"
        ),
        "v196": (
            "low-specificity artifact coverage collapsed to eat with miss abstention"
        ),
        "v197": (
            "low-specificity overrides were blocked, exposing exact/high-specificity "
            "hit share 0.0"
        ),
    }


def lifecycle_flags(*, diagnostic_replay_ran: bool) -> dict[str, object]:
    return {
        "training_ran": False,
        "fit_ran": False,
        "training_artifact_created": False,
        "slice_4_training_started": False,
        "slice_4_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_integration_ran": False,
        "runtime_action_selection_changed": False,
        "runtime_policy_changed": False,
        "gate_relaxation_ran": False,
        "gate_relaxation_allowed": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "diagnostic_replay_ran": bool(diagnostic_replay_ran),
        "promotion_authorized": False,
        "non_promoted": True,
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def route_rationale(route: str) -> str:
    if route == PUBLIC_MODEL_CAPACITY_ROUTE:
        return (
            "The v197 gate prevented the known low-specificity collapse, but the "
            "real replay probe found absent or too-low high-specificity live "
            "public-state overlap. The next step is a public masked "
            "model-capacity harness, not blind slice-4 training."
        )
    if route == ACTION_COMPLETE_ROUTE:
        return (
            "High-specificity keys overlap live replay, but current valid actions "
            "are incomplete or fail the observed-support floor. Audit that "
            "contract before any training."
        )
    if route == FUTURE_EXPLICIT_SLICE_4_ROUTE:
        return (
            "The probe found unexpectedly sufficient high-specificity coverage, "
            "but v198 still consumed no slice. Any slice-4 training requires a "
            "separate explicit opt-in task."
        )
    if route == INSTRUMENTATION_ROUTE:
        return (
            "The probe did not prove real replay provenance, so v198 fails closed "
            "to instrumentation repair and cannot authorize future training."
        )
    if route == STOP_ROUTE:
        return "Required v197 source pins or lifecycle facts failed validation."
    return "Closed no-training route."


def _combined_high_specificity_breakdown(
    combined: Mapping[str, object],
) -> dict[str, object]:
    categories = _mapping(combined.get("candidate_key_category_breakdown"))
    totals = {
        "key_present_count": 0,
        "key_absent_count": 0,
        "complete_for_current_valid_actions_count": 0,
        "present_but_action_incomplete_count": 0,
        "observed_support_floor_satisfied_count": 0,
        "present_complete_but_observed_support_floor_failed_count": 0,
        "imputed_valid_action_score_count": 0,
        "present_complete_floor_satisfied_but_unclear_best_count": 0,
        "selected_source_category_count": 0,
    }
    decision_count = _int(combined.get("decision_count"))
    for category in HIGH_SPECIFICITY_CATEGORIES:
        stats = _mapping(categories.get(category))
        for key in totals:
            totals[key] = _int(totals.get(key)) + _int(stats.get(key))
    return {
        **totals,
        "key_present_share": _share(_int(totals["key_present_count"]), decision_count),
        "complete_for_current_valid_actions_share": _share(
            _int(totals["complete_for_current_valid_actions_count"]),
            decision_count,
        ),
        "observed_support_floor_satisfied_share": _share(
            _int(totals["observed_support_floor_satisfied_count"]),
            decision_count,
        ),
        "selected_source_category_share": _share(
            _int(totals["selected_source_category_count"]),
            decision_count,
        ),
    }


def _category_stats(container: dict[str, object], category: str) -> dict[str, object]:
    categories = container.get("candidate_key_category_stats")
    if not isinstance(categories, defaultdict):
        if isinstance(categories, Mapping):
            rebuilt: defaultdict[str, dict[str, object]] = defaultdict(
                _empty_category_stats
            )
            for key, value in categories.items():
                rebuilt[str(key)] = dict(_mapping(value))
            container["candidate_key_category_stats"] = rebuilt
            categories = rebuilt
        else:
            categories = defaultdict(_empty_category_stats)
            container["candidate_key_category_stats"] = categories
    return categories[category]


def _counter(container: dict[str, object], key: str) -> Counter[str]:
    value = container.get(key)
    if isinstance(value, Counter):
        return value
    counter = Counter()
    if isinstance(value, Mapping):
        counter.update(_int_counter(value))
    container[key] = counter
    return counter


def _int_counter(value: object) -> Counter[str]:
    counter: Counter[str] = Counter()
    if isinstance(value, Mapping):
        for key, count in value.items():
            counter[str(key)] += _int(count)
    return counter


def _share(count: int, total: int) -> float:
    return _round(float(count) / float(total)) if int(total) > 0 else 0.0


def _dominant(counter: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counter.values())
    if total <= 0:
        return {"action": None, "count": 0, "share": 0.0}
    action, count = max(
        sorted(counter.items()),
        key=lambda item: (int(item[1]), str(item[0])),
    )
    return {
        "action": action,
        "count": int(count),
        "share": _share(int(count), total),
    }
