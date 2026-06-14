from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.mind import (
    carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training as v195,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response as v196,
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

M3_CARRION_SURVIVOR_CONTINUATION_V197_COVERAGE_ABSTENTION_REPAIR_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v197_coverage_abstention_repair_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V197_COVERAGE_ABSTENTION_REPAIR_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v197_coverage_abstention_repair_v1"
)

DEFAULT_V196_REPORT_PATH = v196.DEFAULT_OUTPUT_PATH
DEFAULT_V195_REPORT_PATH = v196.DEFAULT_V195_REPORT_PATH
DEFAULT_V195_ARTIFACT_PATH = v196.DEFAULT_V195_ARTIFACT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v197-carrion-survivor-continuation-coverage-abstention-repair-design.json"
)

EXPECTED_V196_REPORT_EXACT_DIGEST = (
    "7b3707b60575a14aee806b3989615582369f021f1ceced80e664e4fdd1794d90"
)
EXPECTED_V196_ROUTE = v196.COVERAGE_OR_ABSTENTION_REPAIR_ROUTE
EXPECTED_V196_MECHANISM = v196.PRIMARY_COVERAGE_COLLAPSE_MECHANISM
EXPECTED_V195_REPORT_EXACT_DIGEST = v196.EXPECTED_V195_REPORT_EXACT_DIGEST
EXPECTED_V195_ARTIFACT_DIGEST = v196.EXPECTED_V195_ARTIFACT_DIGEST

LOW_SPECIFICITY_REJECTION_REASON = "low_specificity_feature_key"
MIN_EXACT_OR_HIGH_SPECIFICITY_HIT_SHARE_FOR_SLICE_4 = 0.20
MAX_COLLAPSE_BLOCKED_OVERRIDE_APPLIED_SHARE = 0.05
MAX_COLLAPSE_BLOCKED_APPLIED_EAT_SHARE = 0.50

STOP_ROUTE = "stop_v197_source_pins_invalid_no_training"
COVERAGE_MODEL_CAPACITY_ROUTE = (
    "v198_high_specificity_coverage_or_model_capacity_repair_before_slice_4_training_no_training"
)
FUTURE_SLICE_4_ROUTE = (
    "v198_specificity_gated_terminal_survival_support_training_slice_4_opt_in"
)
INSTRUMENTATION_ROUTE = "v198_specificity_gate_instrumentation_repair_no_training"


def run_carrion_survivor_continuation_v197_coverage_abstention_repair_design(
    *,
    v196_report_path: str | Path = DEFAULT_V196_REPORT_PATH,
    v195_report_path: str | Path = DEFAULT_V195_REPORT_PATH,
    v195_artifact_path: str | Path = DEFAULT_V195_ARTIFACT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v196_report_exact_digest: str = EXPECTED_V196_REPORT_EXACT_DIGEST,
    expected_v196_route: str = EXPECTED_V196_ROUTE,
    expected_v196_mechanism: str = EXPECTED_V196_MECHANISM,
    expected_v195_report_exact_digest: str = EXPECTED_V195_REPORT_EXACT_DIGEST,
    expected_v195_artifact_digest: str = EXPECTED_V195_ARTIFACT_DIGEST,
    run_diagnostic_replay: bool = True,
    specificity_gate_diagnostics_override: Mapping[str, object] | None = None,
    broad_seeds: Sequence[int] = v195.DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = v195.DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = v195.DEFAULT_TICKS,
) -> dict[str, object]:
    v196_report = load_json_report(v196_report_path)
    v195_report = load_json_report(v195_report_path)
    v195_artifact = load_json_report(v195_artifact_path)
    source_validation = validate_v197_source_pins(
        v196_report=v196_report,
        v195_report=v195_report,
        v195_artifact=v195_artifact,
        expected_v196_report_exact_digest=expected_v196_report_exact_digest,
        expected_v196_route=expected_v196_route,
        expected_v196_mechanism=expected_v196_mechanism,
        expected_v195_report_exact_digest=expected_v195_report_exact_digest,
        expected_v195_artifact_digest=expected_v195_artifact_digest,
    )
    v195_failure_facts = v196.extract_v195_failure_facts(v195_report)
    failure_fact_validation = v196.validate_v195_failure_facts(v195_failure_facts)
    before_specificity_gate = before_specificity_gate_diagnostics(v196_report)
    if specificity_gate_diagnostics_override is not None:
        after_specificity_gate = dict(specificity_gate_diagnostics_override)
        after_specificity_gate["diagnostics_override_used"] = True
        after_specificity_gate["diagnostics_provenance"] = "override"
    elif (
        source_validation.get("passed") is True
        and failure_fact_validation.get("passed") is True
        and run_diagnostic_replay
    ):
        after_specificity_gate = run_specificity_gate_diagnostic_replay(
            artifact=v195_artifact,
            broad_seeds=broad_seeds,
            carrion_fixture_seeds=carrion_fixture_seeds,
            ticks=ticks,
        )
    else:
        after_specificity_gate = skipped_specificity_gate_diagnostics(
            "source_or_failure_validation_failed"
            if source_validation.get("passed") is not True
            or failure_fact_validation.get("passed") is not True
            else "diagnostic_replay_skipped"
        )
    comparison = compare_specificity_gate_diagnostics(
        before=before_specificity_gate,
        after=after_specificity_gate,
    )
    route_decision = route_decision_for_v197(
        source_validation=source_validation,
        failure_fact_validation=failure_fact_validation,
        comparison=comparison,
        after_specificity_gate=after_specificity_gate,
    )
    classification = classification_for_v197(
        source_validation=source_validation,
        failure_fact_validation=failure_fact_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V197_COVERAGE_ABSTENTION_REPAIR_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V197_COVERAGE_ABSTENTION_REPAIR_POLICY,
        "contract": contract(
            expected_v196_report_exact_digest=expected_v196_report_exact_digest,
            expected_v196_route=expected_v196_route,
            expected_v196_mechanism=expected_v196_mechanism,
        ),
        "inputs": {
            "v196_report": str(v196_report_path),
            "v195_report": str(v195_report_path),
            "v195_artifact": str(v195_artifact_path),
            "expected_v196_report_exact_digest": expected_v196_report_exact_digest,
            "expected_v196_route": expected_v196_route,
            "expected_v196_mechanism": expected_v196_mechanism,
            "expected_v195_report_exact_digest": expected_v195_report_exact_digest,
            "expected_v195_artifact_digest": expected_v195_artifact_digest,
            "run_diagnostic_replay": bool(run_diagnostic_replay),
            "broad_seeds": [int(seed) for seed in broad_seeds],
            "carrion_fixture_seeds": [int(seed) for seed in carrion_fixture_seeds],
            "ticks": int(ticks),
        },
        "source_pin_validation": source_validation,
        "v195_failure_facts": v195_failure_facts,
        "failure_fact_validation": failure_fact_validation,
        "before_specificity_gate": before_specificity_gate,
        "after_specificity_gate": after_specificity_gate,
        "specificity_gate_comparison": comparison,
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "coverage_abstention_repair",
                "no_training",
                "no_slice_4_consumption",
                "no_runtime_integration",
                "no_gate_relaxation",
                "no_promotion",
            ],
        },
        **lifecycle_flags(
            diagnostic_replay_ran=after_specificity_gate.get("ran") is True,
        ),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v197_source_pins(
    *,
    v196_report: Mapping[str, object],
    v195_report: Mapping[str, object],
    v195_artifact: Mapping[str, object],
    expected_v196_report_exact_digest: str,
    expected_v196_route: str,
    expected_v196_mechanism: str,
    expected_v195_report_exact_digest: str,
    expected_v195_artifact_digest: str,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v196_report)
    route = _mapping(v196_report.get("route_decision"))
    mechanism = _mapping(v196_report.get("mechanism_analysis"))
    lookup = _mapping(v196_report.get("lookup_coverage_diagnostics"))
    combined = _mapping(lookup.get("combined"))
    v196_source = _mapping(v196_report.get("source_pin_validation"))
    artifact_digest = stable_payload_digest(v195_artifact)
    v195_exact_validation = exact_digest_validation_report(v195_report)
    checks = {
        "v196_report_schema_matches": (
            v196_report.get("schema_version")
            == v196.M3_CARRION_SURVIVOR_CONTINUATION_V196_REPAIRED_CONTRACT_SLICE_3_FAILURE_RESPONSE_SCHEMA_VERSION
        ),
        "v196_report_policy_matches": (
            v196_report.get("policy")
            == v196.M3_CARRION_SURVIVOR_CONTINUATION_V196_REPAIRED_CONTRACT_SLICE_3_FAILURE_RESPONSE_POLICY
        ),
        "v196_report_exact_digest_valid": exact_validation.get("passed") is True,
        "v196_report_exact_digest_matches_expected": (
            v196_report.get("exact_digest") == expected_v196_report_exact_digest
        ),
        "v196_route_matches_expected": (
            route.get("recommended_next_route") == expected_v196_route
        ),
        "v196_mechanism_matches_expected": (
            mechanism.get("primary_mechanism") == expected_v196_mechanism
        ),
        "v196_lookup_replay_ran": lookup.get("ran") is True,
        "v196_low_specificity_collapse_confirmed": (
            _float(combined.get("override_applied_share")) > 0.50
            and combined.get("dominant_applied_override_action") == "eat"
            and _float(combined.get("dominant_applied_override_action_share")) > 0.85
            and _float(combined.get("mask_only_feature_hit_share")) > 0.70
            and _float(combined.get("exact_feature_hit_share")) == 0.0
            and combined.get("missing_states_defaulted_to_stay") is False
        ),
        "v196_source_pin_validation_passed": v196_source.get("passed") is True,
        "v196_lifecycle_training_closed": v196_report.get("training_ran") is False,
        "v196_lifecycle_slice_4_closed": (
            v196_report.get("slice_4_training_consumed") is False
        ),
        "v196_lifecycle_runtime_closed": (
            v196_report.get("runtime_action_selection_changed") is False
        ),
        "v196_lifecycle_promotion_closed": (
            v196_report.get("promotion_authorized") is False
        ),
        "v196_lifecycle_gate_relaxation_closed": (
            v196_report.get("gate_relaxation_allowed") is False
        ),
        "v195_report_exact_digest_valid": v195_exact_validation.get("passed") is True,
        "v195_report_exact_digest_matches_expected": (
            v195_report.get("exact_digest") == expected_v195_report_exact_digest
        ),
        "v195_artifact_digest_matches_expected": (
            artifact_digest == expected_v195_artifact_digest
        ),
        "v196_observed_v195_report_digest_matches_expected": (
            v196_source.get("observed_v195_report_exact_digest")
            == expected_v195_report_exact_digest
        ),
        "v196_observed_v195_artifact_digest_matches_expected": (
            v196_source.get("observed_v195_artifact_digest")
            == expected_v195_artifact_digest
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v197_source_pin_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v196_report_exact_digest": expected_v196_report_exact_digest,
        "observed_v196_report_exact_digest": v196_report.get("exact_digest"),
        "v196_report_exact_digest_validation": exact_validation,
        "expected_v196_route": expected_v196_route,
        "observed_v196_route": route.get("recommended_next_route"),
        "expected_v196_mechanism": expected_v196_mechanism,
        "observed_v196_mechanism": mechanism.get("primary_mechanism"),
        "expected_v195_report_exact_digest": expected_v195_report_exact_digest,
        "observed_v195_report_exact_digest": v195_report.get("exact_digest"),
        "expected_v195_artifact_digest": expected_v195_artifact_digest,
        "observed_v195_artifact_digest": artifact_digest,
        "checks": checks,
    }


def before_specificity_gate_diagnostics(
    v196_report: Mapping[str, object],
) -> dict[str, object]:
    lookup = _mapping(v196_report.get("lookup_coverage_diagnostics"))
    return {
        "policy": "m3_carrion_survivor_continuation_v197_before_specificity_gate_from_v196_v1",
        "source": "pinned_v196_lookup_coverage_diagnostics",
        "ran": lookup.get("ran") is True,
        "broad": _mapping(lookup.get("broad")),
        "carrion_only": _mapping(lookup.get("carrion_only")),
        "combined": _mapping(lookup.get("combined")),
    }


def run_specificity_gate_diagnostic_replay(
    *,
    artifact: Mapping[str, object],
    broad_seeds: Sequence[int],
    carrion_fixture_seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    scorer = load_transition_value_scorer_artifact(artifact)
    broad = _run_lookup_scope(
        scope="broad",
        scorer=scorer,
        seeds=broad_seeds,
        ticks=ticks,
    )
    carrion = _run_lookup_scope(
        scope="carrion_only",
        scorer=scorer,
        seeds=carrion_fixture_seeds,
        ticks=ticks,
    )
    combined = combine_lookup_scopes(broad, carrion)
    return {
        "policy": "m3_carrion_survivor_continuation_v197_specificity_gate_diagnostic_replay_v1",
        "ran": True,
        "diagnostics_only": True,
        "diagnostics_provenance": "real_replay",
        "diagnostics_override_used": False,
        "source_key_specificity_gate_enabled": True,
        "training_rerun": False,
        "slice_4_training_consumed": False,
        "broad": broad,
        "carrion_only": carrion,
        "combined": combined,
    }


def _run_lookup_scope(
    *,
    scope: str,
    scorer: object,
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    stats = _empty_lookup_stats()
    per_seed: list[dict[str, object]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        baseline_policy = _mind_v3_policy(seed=seed, founder_template=None)
        candidate_policy = _mind_v3_policy(
            seed=seed,
            founder_template=None,
            transition_value_scorer=scorer,
            transition_value_action_override=True,
            transition_value_action_override_source_integrity_passed=True,
            transition_value_source_key_specificity_gate_enabled=True,
        )
        if scope == "broad":
            baseline_world = SimulationWorld(
                WorldConfig(seed=seed, max_ticks=int(ticks)),
                policy=baseline_policy,
            )
            candidate_world = SimulationWorld(
                WorldConfig(seed=seed, max_ticks=int(ticks)),
                policy=candidate_policy,
            )
        elif scope == "carrion_only":
            baseline_world = _fixture_world(
                fixture_name="carrion_only",
                seed=seed,
                ticks=int(ticks),
                policy=baseline_policy,
            )
            candidate_world = _fixture_world(
                fixture_name="carrion_only",
                seed=seed,
                ticks=int(ticks),
                policy=candidate_policy,
            )
        else:
            raise ValueError(f"unsupported v197 lookup diagnostic scope: {scope}")
        baseline = _run_world(
            world=baseline_world,
            seed=seed,
            ticks=int(ticks),
            trajectory_output_path=None,
            trajectory_split_id=f"v197_{scope}_baseline",
        )
        candidate = _run_world(
            world=candidate_world,
            seed=seed,
            ticks=int(ticks),
            trajectory_output_path=None,
            trajectory_split_id=f"v197_{scope}_specificity_gate",
        )
        seed_stats = _lookup_stats_from_decisions(
            candidate_world.policy_decision_diagnostics_records
        )
        _merge_run_stats(seed_stats, candidate)
        _merge_lookup_stats(stats, seed_stats)
        alive_delta = _int(candidate.get("alive_agents")) - _int(
            baseline.get("alive_agents")
        )
        births_delta = _int(candidate.get("births")) - _int(baseline.get("births"))
        regression = alive_delta < 0 or births_delta < 0
        if regression:
            stats["alive_birth_regression_count"] = (
                _int(stats.get("alive_birth_regression_count")) + 1
            )
        per_seed.append(
            {
                "seed": seed,
                "baseline_alive_agents": baseline.get("alive_agents"),
                "candidate_alive_agents": candidate.get("alive_agents"),
                "alive_agents_delta": alive_delta,
                "baseline_births": baseline.get("births"),
                "candidate_births": candidate.get("births"),
                "births_delta": births_delta,
                "alive_or_birth_regression": regression,
                "candidate_heuristic_action_source_count": candidate.get(
                    "heuristic_action_source_count"
                ),
                "candidate_requested_action_counts": candidate.get(
                    "requested_action_counts"
                ),
                "candidate_dominant_requested_action": candidate.get(
                    "dominant_requested_action"
                ),
                "candidate_dominant_requested_action_share": candidate.get(
                    "dominant_requested_action_share"
                ),
                "lookup": _finalize_lookup_stats(seed_stats),
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v197_specificity_gate_scope_diagnostics_v1",
        "scope": scope,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "per_seed": per_seed,
        "per_seed_alive_birth_regressions": [
            item for item in per_seed if item["alive_or_birth_regression"]
        ],
        **_finalize_lookup_stats(stats),
    }


def _lookup_stats_from_decisions(
    decision_diagnostics: Sequence[Mapping[str, object] | None],
) -> dict[str, object]:
    stats = _empty_lookup_stats()
    for diagnostic in decision_diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        transition = _mapping(diagnostic.get("transition_value_scorer"))
        if transition:
            _update_lookup_stats(stats, transition)
    return stats


def _empty_lookup_stats() -> dict[str, object]:
    return {
        "decision_count": 0,
        "supported_score_count": 0,
        "observed_support_floor_satisfied_count": 0,
        "clear_best_count": 0,
        "override_applied_count": 0,
        "runtime_action_selection_changed_count": 0,
        "missing_supported_score_count": 0,
        "low_observed_support_count": 0,
        "no_prediction_count": 0,
        "imputed_valid_action_decision_count": 0,
        "low_specificity_rejected_decision_count": 0,
        "source_key_specificity_gate_enabled_count": 0,
        "source_key_specificity_gate_passed_count": 0,
        "source_key_specificity_gate_failed_count": 0,
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": 0,
        "alive_birth_regression_count": 0,
        "score_source_counts": Counter(),
        "source_key_category_counts": Counter(),
        "predicted_action_counts": Counter(),
        "applied_override_action_counts": Counter(),
        "final_requested_action_counts": Counter(),
        "requested_action_counts": Counter(),
        "original_mind_v3_requested_action_counts": Counter(),
        "override_rejected_reason_counts": Counter(),
        "final_requested_action_counts_by_rejected_reason": defaultdict(Counter),
    }


def _update_lookup_stats(
    stats: dict[str, object],
    transition: Mapping[str, object],
) -> None:
    stats["decision_count"] = _int(stats.get("decision_count")) + 1
    score_source = str(transition.get("score_source") or "")
    source_key_category = str(transition.get("source_key_category") or "")
    if not source_key_category:
        source_key_category = transition_value_source_key_category(
            score_source,
            transition.get("source_key"),
        )
    _counter(stats, "score_source_counts").update([score_source or "unknown"])
    _counter(stats, "source_key_category_counts").update([source_key_category])
    final_requested = str(transition.get("final_requested_action") or "")
    original_requested = str(transition.get("original_mind_v3_requested_action") or "")
    predicted_action = str(transition.get("predicted_action") or "")
    if final_requested:
        _counter(stats, "final_requested_action_counts").update([final_requested])
    if original_requested:
        _counter(stats, "original_mind_v3_requested_action_counts").update(
            [original_requested]
        )
    if predicted_action:
        _counter(stats, "predicted_action_counts").update([predicted_action])
    else:
        stats["no_prediction_count"] = _int(stats.get("no_prediction_count")) + 1
    if transition.get("supported_scores_for_all_valid_actions") is True:
        stats["supported_score_count"] = _int(stats.get("supported_score_count")) + 1
    else:
        stats["missing_supported_score_count"] = (
            _int(stats.get("missing_supported_score_count")) + 1
        )
    if (
        transition.get("observed_support_floor_satisfied_for_all_valid_actions")
        is True
    ):
        stats["observed_support_floor_satisfied_count"] = (
            _int(stats.get("observed_support_floor_satisfied_count")) + 1
        )
    if transition.get("clear_best_valid_action") is True:
        stats["clear_best_count"] = _int(stats.get("clear_best_count")) + 1
    if transition.get("has_imputed_valid_action_score") is True:
        stats["imputed_valid_action_decision_count"] = (
            _int(stats.get("imputed_valid_action_decision_count")) + 1
        )
    low_support = _int(transition.get("low_observed_support_valid_action_score_count"))
    if low_support > 0:
        stats["low_observed_support_count"] = (
            _int(stats.get("low_observed_support_count")) + 1
        )
    if transition.get("source_key_specificity_gate_enabled") is True:
        stats["source_key_specificity_gate_enabled_count"] = (
            _int(stats.get("source_key_specificity_gate_enabled_count")) + 1
        )
        if transition.get("source_key_specificity_gate_passed") is True:
            stats["source_key_specificity_gate_passed_count"] = (
                _int(stats.get("source_key_specificity_gate_passed_count")) + 1
            )
        else:
            stats["source_key_specificity_gate_failed_count"] = (
                _int(stats.get("source_key_specificity_gate_failed_count")) + 1
            )
    if transition.get("override_applied") is True:
        stats["override_applied_count"] = _int(stats.get("override_applied_count")) + 1
        if predicted_action:
            _counter(stats, "applied_override_action_counts").update([predicted_action])
    else:
        reason = str(transition.get("override_rejected_reason") or "none")
        _counter(stats, "override_rejected_reason_counts").update([reason])
        if reason == LOW_SPECIFICITY_REJECTION_REASON:
            stats["low_specificity_rejected_decision_count"] = (
                _int(stats.get("low_specificity_rejected_decision_count")) + 1
            )
        if final_requested:
            _nested_counter(
                stats,
                "final_requested_action_counts_by_rejected_reason",
                reason,
            ).update([final_requested])
    if transition.get("runtime_action_selection_changed") is True:
        stats["runtime_action_selection_changed_count"] = (
            _int(stats.get("runtime_action_selection_changed_count")) + 1
        )


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


def _merge_lookup_stats(target: dict[str, object], source: Mapping[str, object]) -> None:
    for key in (
        "decision_count",
        "supported_score_count",
        "observed_support_floor_satisfied_count",
        "clear_best_count",
        "override_applied_count",
        "runtime_action_selection_changed_count",
        "missing_supported_score_count",
        "low_observed_support_count",
        "no_prediction_count",
        "imputed_valid_action_decision_count",
        "low_specificity_rejected_decision_count",
        "source_key_specificity_gate_enabled_count",
        "source_key_specificity_gate_passed_count",
        "source_key_specificity_gate_failed_count",
        "heuristic_action_source_count",
        "unsupported_requested_action_count",
        "unsupported_resolved_action_count",
        "alive_birth_regression_count",
    ):
        target[key] = _int(target.get(key)) + _int(source.get(key))
    for key in (
        "score_source_counts",
        "source_key_category_counts",
        "predicted_action_counts",
        "applied_override_action_counts",
        "final_requested_action_counts",
        "requested_action_counts",
        "original_mind_v3_requested_action_counts",
        "override_rejected_reason_counts",
    ):
        _counter(target, key).update(_int_counter(source.get(key)))
    nested = source.get("final_requested_action_counts_by_rejected_reason")
    if isinstance(nested, Mapping):
        for nested_key, counts in nested.items():
            _nested_counter(
                target,
                "final_requested_action_counts_by_rejected_reason",
                str(nested_key),
            ).update(_int_counter(counts))


def _merge_lookup_payload(target: dict[str, object], source: Mapping[str, object]) -> None:
    payload = {
        "decision_count": source.get("decision_count"),
        "supported_score_count": source.get("supported_score_count"),
        "observed_support_floor_satisfied_count": source.get(
            "observed_support_floor_satisfied_count"
        ),
        "clear_best_count": source.get("clear_best_count"),
        "override_applied_count": source.get("override_applied_count"),
        "runtime_action_selection_changed_count": source.get(
            "runtime_action_selection_changed_count"
        ),
        "missing_supported_score_count": source.get("missing_supported_score_count"),
        "low_observed_support_count": source.get("low_observed_support_count"),
        "no_prediction_count": source.get("no_prediction_count"),
        "imputed_valid_action_decision_count": source.get(
            "imputed_valid_action_decision_count"
        ),
        "low_specificity_rejected_decision_count": source.get(
            "low_specificity_rejected_decision_count"
        ),
        "source_key_specificity_gate_enabled_count": source.get(
            "source_key_specificity_gate_enabled_count"
        ),
        "source_key_specificity_gate_passed_count": source.get(
            "source_key_specificity_gate_passed_count"
        ),
        "source_key_specificity_gate_failed_count": source.get(
            "source_key_specificity_gate_failed_count"
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
        "source_key_category_counts": source.get("source_key_category_counts"),
        "predicted_action_counts": source.get("predicted_action_counts"),
        "applied_override_action_counts": source.get("applied_override_action_counts"),
        "final_requested_action_counts": source.get("final_requested_action_counts"),
        "requested_action_counts": source.get("requested_action_counts"),
        "original_mind_v3_requested_action_counts": source.get(
            "original_mind_v3_requested_action_counts"
        ),
        "override_rejected_reason_counts": source.get(
            "override_rejected_reason_counts"
        ),
        "final_requested_action_counts_by_rejected_reason": source.get(
            "final_requested_action_counts_by_rejected_reason"
        ),
    }
    _merge_lookup_stats(target, payload)


def combine_lookup_scopes(*scopes: Mapping[str, object]) -> dict[str, object]:
    stats = _empty_lookup_stats()
    for scope in scopes:
        _merge_lookup_payload(stats, scope)
    return {
        "policy": "m3_carrion_survivor_continuation_v197_combined_specificity_gate_diagnostics_v1",
        **_finalize_lookup_stats(stats),
    }


def _finalize_lookup_stats(stats: Mapping[str, object]) -> dict[str, object]:
    decision_count = _int(stats.get("decision_count"))
    supported_count = _int(stats.get("supported_score_count"))
    applied_count = _int(stats.get("override_applied_count"))
    changed_count = _int(stats.get("runtime_action_selection_changed_count"))
    missing_count = _int(stats.get("missing_supported_score_count"))
    source_categories = _int_counter(stats.get("source_key_category_counts"))
    applied_counts = _int_counter(stats.get("applied_override_action_counts"))
    final_counts = _int_counter(stats.get("final_requested_action_counts"))
    requested_counts = _int_counter(stats.get("requested_action_counts"))
    predicted_counts = _int_counter(stats.get("predicted_action_counts"))
    low_specificity_count = sum(
        int(source_categories.get(category, 0))
        for category in TRANSITION_VALUE_LOW_SPECIFICITY_SOURCE_KEY_CATEGORIES
    )
    exact_high_count = sum(
        int(source_categories.get(category, 0))
        for category in TRANSITION_VALUE_HIGH_SPECIFICITY_SOURCE_KEY_CATEGORIES
    )
    exact_count = int(source_categories.get("exact_context_feature_hit", 0))
    applied_dominant = _dominant(applied_counts)
    final_dominant = _dominant(final_counts)
    requested_dominant = _dominant(requested_counts)
    predicted_dominant = _dominant(predicted_counts)
    return {
        "decision_count": decision_count,
        "supported_score_count": supported_count,
        "supported_score_share": _share(supported_count, decision_count),
        "observed_support_floor_satisfied_count": _int(
            stats.get("observed_support_floor_satisfied_count")
        ),
        "clear_best_count": _int(stats.get("clear_best_count")),
        "override_applied_count": applied_count,
        "override_applied_share": _share(applied_count, decision_count),
        "runtime_action_selection_changed_count": changed_count,
        "runtime_action_selection_changed_share": _share(changed_count, decision_count),
        "missing_supported_score_count": missing_count,
        "missing_supported_score_share": _share(missing_count, decision_count),
        "low_observed_support_count": _int(stats.get("low_observed_support_count")),
        "no_prediction_count": _int(stats.get("no_prediction_count")),
        "imputed_valid_action_decision_count": _int(
            stats.get("imputed_valid_action_decision_count")
        ),
        "source_key_specificity_gate_enabled_count": _int(
            stats.get("source_key_specificity_gate_enabled_count")
        ),
        "source_key_specificity_gate_passed_count": _int(
            stats.get("source_key_specificity_gate_passed_count")
        ),
        "source_key_specificity_gate_failed_count": _int(
            stats.get("source_key_specificity_gate_failed_count")
        ),
        "low_specificity_feature_hit_count": low_specificity_count,
        "low_specificity_feature_hit_share": _share(
            low_specificity_count,
            decision_count,
        ),
        "low_specificity_rejected_decision_count": _int(
            stats.get("low_specificity_rejected_decision_count")
        ),
        "low_specificity_rejected_decision_share": _share(
            _int(stats.get("low_specificity_rejected_decision_count")),
            decision_count,
        ),
        "exact_feature_hit_count": exact_count,
        "exact_feature_hit_share": _share(exact_count, decision_count),
        "exact_or_high_specificity_hit_count": exact_high_count,
        "exact_or_high_specificity_hit_share": _share(
            exact_high_count,
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
        "score_source_counts": dict(sorted(_int_counter(stats.get("score_source_counts")).items())),
        "source_key_category_counts": dict(sorted(source_categories.items())),
        "predicted_action_counts": dict(sorted(predicted_counts.items())),
        "dominant_predicted_action": predicted_dominant["action"],
        "dominant_predicted_action_share": predicted_dominant["share"],
        "applied_override_action_counts": dict(sorted(applied_counts.items())),
        "dominant_applied_override_action": applied_dominant["action"],
        "dominant_applied_override_action_share": applied_dominant["share"],
        "final_requested_action_counts": dict(sorted(final_counts.items())),
        "dominant_final_requested_action": final_dominant["action"],
        "dominant_final_requested_action_share": final_dominant["share"],
        "requested_action_counts": dict(sorted(requested_counts.items())),
        "dominant_requested_action": requested_dominant["action"],
        "dominant_requested_action_share": requested_dominant["share"],
        "original_mind_v3_requested_action_counts": dict(
            sorted(_int_counter(stats.get("original_mind_v3_requested_action_counts")).items())
        ),
        "override_rejected_reason_counts": dict(
            sorted(_int_counter(stats.get("override_rejected_reason_counts")).items())
        ),
        "final_requested_action_counts_by_rejected_reason": {
            key: dict(sorted(_int_counter(value).items()))
            for key, value in sorted(
                _mapping(stats.get("final_requested_action_counts_by_rejected_reason")).items()
            )
        },
    }


def skipped_specificity_gate_diagnostics(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v197_specificity_gate_diagnostic_replay_v1",
        "ran": False,
        "reason": reason,
        "diagnostics_provenance": "skipped",
        "diagnostics_override_used": False,
        "training_rerun": False,
        "slice_4_training_consumed": False,
    }


def compare_specificity_gate_diagnostics(
    *,
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    before_combined = _mapping(before.get("combined"))
    after_combined = _mapping(after.get("combined"))
    before_override_share = _float(before_combined.get("override_applied_share"))
    after_override_share = _float(after_combined.get("override_applied_share"))
    before_eat_share = _float(
        before_combined.get("dominant_applied_override_action_share")
    )
    after_eat_share = (
        _float(after_combined.get("dominant_applied_override_action_share"))
        if after_combined.get("dominant_applied_override_action") == "eat"
        else 0.0
    )
    low_rejected = _int(after_combined.get("low_specificity_rejected_decision_count"))
    exact_high_share = _float(
        after_combined.get("exact_or_high_specificity_hit_share")
    )
    after_broad = _mapping(after.get("broad"))
    broad_regressions = [
        dict(item)
        for item in after_broad.get("per_seed_alive_birth_regressions", [])
        if isinstance(item, Mapping)
    ]
    collapse_blocked = (
        after.get("ran") is True
        and low_rejected > 0
        and after_override_share <= MAX_COLLAPSE_BLOCKED_OVERRIDE_APPLIED_SHARE
        and after_eat_share <= MAX_COLLAPSE_BLOCKED_APPLIED_EAT_SHARE
    )
    coverage_sufficient = (
        exact_high_share >= MIN_EXACT_OR_HIGH_SPECIFICITY_HIT_SHARE_FOR_SLICE_4
    )
    real_replay_provenance = _specificity_gate_real_replay_provenance(after)
    slice_4_training_justified = (
        collapse_blocked
        and coverage_sufficient
        and real_replay_provenance
        and len(broad_regressions) == 0
        and _int(after_combined.get("heuristic_action_source_count")) == 0
        and _float(after_combined.get("dominant_requested_action_share")) <= 0.50
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v197_specificity_gate_comparison_v1",
        "before_override_applied_share": _round(before_override_share),
        "after_override_applied_share": _round(after_override_share),
        "override_applied_share_delta": _round(
            after_override_share - before_override_share
        ),
        "before_dominant_applied_override_action": before_combined.get(
            "dominant_applied_override_action"
        ),
        "before_dominant_applied_override_action_share": _round(before_eat_share),
        "after_dominant_applied_override_action": after_combined.get(
            "dominant_applied_override_action"
        ),
        "after_dominant_applied_override_action_share": _round(after_eat_share),
        "applied_eat_share_delta": _round(after_eat_share - before_eat_share),
        "low_specificity_rejected_decision_count": low_rejected,
        "low_specificity_rejected_decision_share": after_combined.get(
            "low_specificity_rejected_decision_share"
        ),
        "exact_or_high_specificity_hit_share": after_combined.get(
            "exact_or_high_specificity_hit_share"
        ),
        "exact_or_high_specificity_hit_count": after_combined.get(
            "exact_or_high_specificity_hit_count"
        ),
        "min_exact_or_high_specificity_hit_share_for_slice_4": (
            MIN_EXACT_OR_HIGH_SPECIFICITY_HIT_SHARE_FOR_SLICE_4
        ),
        "low_specificity_collapse_blocked": collapse_blocked,
        "exact_or_high_specificity_coverage_sufficient_for_slice_4": (
            coverage_sufficient
        ),
        "real_replay_provenance_for_slice_4": real_replay_provenance,
        "after_heuristic_action_source_count": after_combined.get(
            "heuristic_action_source_count"
        ),
        "after_dominant_requested_action": after_combined.get(
            "dominant_requested_action"
        ),
        "after_dominant_requested_action_share": after_combined.get(
            "dominant_requested_action_share"
        ),
        "after_broad_alive_birth_regression_count": len(broad_regressions),
        "after_broad_alive_birth_regressions": broad_regressions,
        "slice_4_training_justified": slice_4_training_justified,
        "slice_4_training_justification_blocked_reason": (
            None
            if slice_4_training_justified
            else _slice_4_training_blocked_reason(
                collapse_blocked=collapse_blocked,
                coverage_sufficient=coverage_sufficient,
                real_replay_provenance=real_replay_provenance,
                broad_regression_count=len(broad_regressions),
                heuristic_action_source_count=_int(
                    after_combined.get("heuristic_action_source_count")
                ),
                dominant_requested_action_share=_float(
                    after_combined.get("dominant_requested_action_share")
                ),
            )
        ),
    }


def _specificity_gate_real_replay_provenance(after: Mapping[str, object]) -> bool:
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


def _slice_4_training_blocked_reason(
    *,
    collapse_blocked: bool,
    coverage_sufficient: bool,
    real_replay_provenance: bool,
    broad_regression_count: int,
    heuristic_action_source_count: int,
    dominant_requested_action_share: float,
) -> str:
    if not collapse_blocked:
        return "low_specificity_collapse_not_blocked"
    if not coverage_sufficient:
        return "exact_or_high_specificity_coverage_too_low"
    if not real_replay_provenance:
        return "specificity_gate_diagnostics_not_real_replay"
    if broad_regression_count > 0:
        return "broad_alive_birth_regressions_present"
    if heuristic_action_source_count != 0:
        return "heuristic_action_sources_present"
    if dominant_requested_action_share > 0.50:
        return "dominant_requested_action_share_above_cap"
    return "unknown_closed_no_training"


def route_decision_for_v197(
    *,
    source_validation: Mapping[str, object],
    failure_fact_validation: Mapping[str, object],
    comparison: Mapping[str, object],
    after_specificity_gate: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True or failure_fact_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif after_specificity_gate.get("ran") is not True:
        route = INSTRUMENTATION_ROUTE
    elif comparison.get("slice_4_training_justified") is True:
        route = FUTURE_SLICE_4_ROUTE
    elif comparison.get("low_specificity_collapse_blocked") is True:
        route = COVERAGE_MODEL_CAPACITY_ROUTE
    else:
        route = INSTRUMENTATION_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v197_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "source_pins_valid": source_validation.get("passed") is True,
        "failure_facts_valid": failure_fact_validation.get("passed") is True,
        "low_specificity_collapse_blocked": comparison.get(
            "low_specificity_collapse_blocked"
        )
        is True,
        "exact_or_high_specificity_coverage_sufficient_for_slice_4": (
            comparison.get("exact_or_high_specificity_coverage_sufficient_for_slice_4")
            is True
        ),
        "future_explicit_slice_4_training_route_authorized": (
            route == FUTURE_SLICE_4_ROUTE
        ),
        "direct_slice_4_training_allowed": False,
        "slice_4_training_consumed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "gate_relaxation_allowed": False,
        "promotion_authorized": False,
        "rationale": route_rationale(route),
    }


def classification_for_v197(
    *,
    source_validation: Mapping[str, object],
    failure_fact_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v197_coverage_abstention_repair_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if failure_fact_validation.get("passed") is not True:
        return prefix + "unexpected_v195_failure_facts_closed_no_training"
    route = str(route_decision.get("recommended_next_route") or "")
    if route == COVERAGE_MODEL_CAPACITY_ROUTE:
        return (
            prefix
            + "low_specificity_collapse_blocked_routes_to_coverage_or_model_capacity_no_training"
        )
    if route == FUTURE_SLICE_4_ROUTE:
        return prefix + "specificity_gate_sufficient_routes_to_future_slice_4_opt_in"
    if route == INSTRUMENTATION_ROUTE:
        return prefix + "instrumentation_required_closed_no_training"
    return prefix + "closed_no_training"


def contract(
    *,
    expected_v196_report_exact_digest: str,
    expected_v196_route: str,
    expected_v196_mechanism: str,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_allowed": False,
        "slice_4_training_consumption_allowed": False,
        "runtime_artifact_creation_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "gate_relaxation_allowed": False,
        "promotion_authorized": False,
        "requires_v196_report_digest": expected_v196_report_exact_digest,
        "requires_v196_route": expected_v196_route,
        "requires_v196_mechanism": expected_v196_mechanism,
        "specificity_gate_policy": {
            "enabled_only_for_v197_diagnostic_replay": True,
            "default_runtime_behavior_changed": False,
            "rejection_reason": LOW_SPECIFICITY_REJECTION_REASON,
            "rejected_categories": sorted(
                TRANSITION_VALUE_LOW_SPECIFICITY_SOURCE_KEY_CATEGORIES
            ),
            "allowed_categories": sorted(
                TRANSITION_VALUE_HIGH_SPECIFICITY_SOURCE_KEY_CATEGORIES
            ),
        },
        "route_logic": {
            "source_pin_failure": STOP_ROUTE,
            "collapse_blocked_but_high_specificity_coverage_low": (
                COVERAGE_MODEL_CAPACITY_ROUTE
            ),
            "future_training_requires_real_replay_provenance": True,
            "collapse_blocked_and_slice_4_justified": FUTURE_SLICE_4_ROUTE,
            "otherwise": INSTRUMENTATION_ROUTE,
        },
    }


def lifecycle_flags(*, diagnostic_replay_ran: bool) -> dict[str, object]:
    return {
        "training_ran": False,
        "training_artifact_created": False,
        "fit_ran": False,
        "v195_training_rerun": False,
        "slice_3_training_consumed": False,
        "slice_4_training_started": False,
        "slice_4_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_policy_changed": False,
        "runtime_integration_ran": False,
        "shadow_eval_ran": False,
        "diagnostic_replay_ran": bool(diagnostic_replay_ran),
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "gate_relaxation_ran": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "non_promoted": True,
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def route_rationale(route: str) -> str:
    if route == COVERAGE_MODEL_CAPACITY_ROUTE:
        return (
            "The specificity gate blocked the known low-specificity override "
            "collapse, but exact/high-specificity coverage is still too low to "
            "justify blind slice-4 training. Repair coverage or model capacity "
            "before any future training slice."
        )
    if route == FUTURE_SLICE_4_ROUTE:
        return (
            "The specificity gate blocked the known collapse and high-specificity "
            "coverage met the explicit threshold, so the next action is a separate "
            "explicit opt-in slice-4 training task."
        )
    if route == INSTRUMENTATION_ROUTE:
        return (
            "The current replay did not prove the specificity gate blocks the "
            "known collapse. Add instrumentation before any training."
        )
    return "Source pins or required failure facts did not validate; stop."


def _dominant(counts: Mapping[str, int] | Counter[str]) -> dict[str, object]:
    counter = Counter({str(key): int(value) for key, value in dict(counts).items()})
    total = sum(counter.values())
    if total <= 0:
        return {"action": None, "count": 0, "share": 0.0}
    action, count = max(counter.items(), key=lambda item: (item[1], item[0]))
    return {"action": action, "count": int(count), "share": _share(count, total)}


def _share(count: int | float, total: int | float) -> float:
    total_value = float(total)
    if total_value <= 0.0:
        return 0.0
    return _round(float(count) / total_value)


def _int_counter(value: object) -> Counter[str]:
    return Counter({str(key): int(raw) for key, raw in _mapping(value).items()})


def _counter(stats: dict[str, object], key: str) -> Counter[str]:
    value = stats.get(key)
    if not isinstance(value, Counter):
        value = Counter()
        stats[key] = value
    return value


def _nested_counter(
    stats: dict[str, object],
    key: str,
    nested_key: str,
) -> Counter[str]:
    value = stats.get(key)
    if not isinstance(value, defaultdict):
        value = defaultdict(Counter)
        stats[key] = value
    nested = value[nested_key]
    if not isinstance(nested, Counter):
        nested = Counter()
        value[nested_key] = nested
    return nested
