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
    carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair as v198,
)
from evolution_sim.mind.candidate_campaign import _int, _mapping, _round, write_json
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
    load_transition_value_scorer_artifact,
)

M3_CARRION_SURVIVOR_CONTINUATION_V199_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V199_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit_v1"
)

DEFAULT_V198_REPORT_PATH = v198.DEFAULT_OUTPUT_PATH
DEFAULT_V195_ARTIFACT_PATH = v198.DEFAULT_V195_ARTIFACT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v199-carrion-survivor-continuation-high-specificity-action-complete-contract-audit.json"
)

EXPECTED_V198_REPORT_EXACT_DIGEST = (
    "91d4132c07ea18b0603626d4c080bdfed01f39677594737cda3ca0233fd95af8"
)
EXPECTED_V198_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v198_high_specificity_action_complete_or_support_floor_gap_routes_to_contract_audit_no_training"
)
EXPECTED_V198_ROUTE = v198.ACTION_COMPLETE_ROUTE
EXPECTED_V198_PRIMARY_BLOCKER = "high_specificity_action_incomplete"
EXPECTED_V198_ARCHIVE_PATH = (
    "gdrive:evolution-sim-backups/archives/"
    "20260614T191500Z-v198-high-specificity-coverage-or-model-capacity-repair.tar.zst"
)
EXPECTED_V198_ARCHIVE_SHA256 = (
    "6d9627c305e56d9083861123e4114635c30c1ae2d939d499026fa73d8f63dbcf"
)
EXPECTED_V195_ARTIFACT_DIGEST = v198.EXPECTED_V195_ARTIFACT_DIGEST

HIGH_SPECIFICITY_CATEGORIES = tuple(sorted(v198.HIGH_SPECIFICITY_CATEGORIES))
HIGH_SPECIFICITY_CATEGORY_ORDER = (
    "exact_context_feature_hit",
    "self_nav_coarse_feature_hit",
    "self_nav_feature_hit",
    "nav_feature_hit",
)

STOP_ROUTE = "stop_v199_source_pins_invalid_no_training"
INSTRUMENTATION_ROUTE = (
    "v200_high_specificity_candidate_key_instrumentation_repair_no_training"
)
PUBLIC_MODEL_CAPACITY_ROUTE = "v200_public_masked_model_capacity_harness_no_training"
ACTION_COMPLETE_REPAIR_ROUTE = (
    "v200_high_specificity_action_complete_contract_repair_design_no_training"
)


def run_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit(
    *,
    v198_report_path: str | Path = DEFAULT_V198_REPORT_PATH,
    v195_artifact_path: str | Path = DEFAULT_V195_ARTIFACT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v198_report_exact_digest: str = EXPECTED_V198_REPORT_EXACT_DIGEST,
    expected_v198_classification: str = EXPECTED_V198_CLASSIFICATION,
    expected_v198_route: str = EXPECTED_V198_ROUTE,
    expected_v198_primary_blocker: str = EXPECTED_V198_PRIMARY_BLOCKER,
    expected_v198_archive_path: str = EXPECTED_V198_ARCHIVE_PATH,
    expected_v198_archive_sha256: str = EXPECTED_V198_ARCHIVE_SHA256,
    expected_v195_artifact_digest: str = EXPECTED_V195_ARTIFACT_DIGEST,
    run_diagnostic_replay: bool = True,
    action_audit_override: Mapping[str, object] | None = None,
    broad_seeds: Sequence[int] = v195.DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = v195.DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = v195.DEFAULT_TICKS,
) -> dict[str, object]:
    v198_report = load_json_report(v198_report_path)
    v195_artifact = load_json_report(v195_artifact_path)
    source_validation = validate_v199_source_pins(
        v198_report=v198_report,
        v195_artifact=v195_artifact,
        expected_v198_report_exact_digest=expected_v198_report_exact_digest,
        expected_v198_classification=expected_v198_classification,
        expected_v198_route=expected_v198_route,
        expected_v198_primary_blocker=expected_v198_primary_blocker,
        expected_v198_archive_path=expected_v198_archive_path,
        expected_v198_archive_sha256=expected_v198_archive_sha256,
        expected_v195_artifact_digest=expected_v195_artifact_digest,
    )
    if action_audit_override is not None:
        action_audit = dict(action_audit_override)
        action_audit["diagnostics_override_used"] = True
        action_audit["diagnostics_provenance"] = "override"
    elif source_validation.get("passed") is True and run_diagnostic_replay:
        action_audit = run_high_specificity_action_complete_audit(
            artifact=v195_artifact,
            broad_seeds=broad_seeds,
            carrion_fixture_seeds=carrion_fixture_seeds,
            ticks=ticks,
        )
    else:
        action_audit = skipped_action_complete_audit(
            "source_validation_failed"
            if source_validation.get("passed") is not True
            else "diagnostic_replay_skipped"
        )
    contract_assessment = assess_action_complete_contract(action_audit)
    route_decision = route_decision_for_v199(
        source_validation=source_validation,
        action_audit=action_audit,
        contract_assessment=contract_assessment,
    )
    classification = classification_for_v199(
        source_validation=source_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V199_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_AUDIT_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V199_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_AUDIT_POLICY
        ),
        "contract": contract(
            expected_v198_report_exact_digest=expected_v198_report_exact_digest,
            expected_v198_route=expected_v198_route,
            expected_v198_archive_path=expected_v198_archive_path,
            expected_v198_archive_sha256=expected_v198_archive_sha256,
        ),
        "historical_evidence_checkpoint": historical_evidence_checkpoint(),
        "inputs": {
            "v198_report": str(v198_report_path),
            "v195_artifact": str(v195_artifact_path),
            "expected_v198_report_exact_digest": expected_v198_report_exact_digest,
            "expected_v198_classification": expected_v198_classification,
            "expected_v198_route": expected_v198_route,
            "expected_v198_primary_blocker": expected_v198_primary_blocker,
            "expected_v198_archive_path": expected_v198_archive_path,
            "expected_v198_archive_sha256": expected_v198_archive_sha256,
            "expected_v195_artifact_digest": expected_v195_artifact_digest,
            "run_diagnostic_replay": bool(run_diagnostic_replay),
            "broad_seeds": [int(seed) for seed in broad_seeds],
            "carrion_fixture_seeds": [int(seed) for seed in carrion_fixture_seeds],
            "ticks": int(ticks),
        },
        "source_pin_validation": source_validation,
        "action_complete_audit": action_audit,
        "action_complete_contract": required_action_complete_contract(),
        "contract_assessment": contract_assessment,
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "high_specificity_action_complete_contract_audit",
                "no_training",
                "no_slice_4_consumption",
                "no_runtime_integration",
                "no_gate_relaxation",
                "no_support_generation",
                "no_support_expansion",
                "no_promotion",
            ],
        },
        **lifecycle_flags(
            diagnostic_replay_ran=action_audit.get("ran") is True,
        ),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v199_source_pins(
    *,
    v198_report: Mapping[str, object],
    v195_artifact: Mapping[str, object],
    expected_v198_report_exact_digest: str,
    expected_v198_classification: str,
    expected_v198_route: str,
    expected_v198_primary_blocker: str,
    expected_v198_archive_path: str,
    expected_v198_archive_sha256: str,
    expected_v195_artifact_digest: str,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v198_report)
    classification = _mapping(v198_report.get("classification"))
    route = _mapping(v198_report.get("route_decision"))
    coverage = _mapping(v198_report.get("coverage_assessment"))
    source = _mapping(v198_report.get("source_pin_validation"))
    probe = _mapping(v198_report.get("high_specificity_probe"))
    combined = _mapping(probe.get("combined"))
    artifact_digest = stable_payload_digest(v195_artifact)
    lifecycle_keys = (
        "training_ran",
        "fit_ran",
        "training_artifact_created",
        "slice_4_training_started",
        "slice_4_training_consumed",
        "runtime_artifact_created",
        "runtime_integration_ran",
        "runtime_action_selection_changed",
        "runtime_policy_changed",
        "gate_relaxation_ran",
        "gate_relaxation_allowed",
        "support_generation_ran",
        "support_expansion_ran",
        "promotion_authorized",
    )
    checks = {
        "v198_report_schema_matches": (
            v198_report.get("schema_version")
            == v198.M3_CARRION_SURVIVOR_CONTINUATION_V198_HIGH_SPECIFICITY_COVERAGE_OR_MODEL_CAPACITY_REPAIR_SCHEMA_VERSION
        ),
        "v198_report_policy_matches": (
            v198_report.get("policy")
            == v198.M3_CARRION_SURVIVOR_CONTINUATION_V198_HIGH_SPECIFICITY_COVERAGE_OR_MODEL_CAPACITY_REPAIR_POLICY
        ),
        "v198_report_exact_digest_valid": exact_validation.get("passed") is True,
        "v198_report_exact_digest_matches_expected": (
            v198_report.get("exact_digest") == expected_v198_report_exact_digest
        ),
        "v198_classification_matches_expected": (
            classification.get("primary") == expected_v198_classification
        ),
        "v198_route_matches_expected": (
            route.get("selected_route") == expected_v198_route
            and route.get("recommended_next_route") == expected_v198_route
        ),
        "v198_primary_blocker_matches_expected": (
            route.get("primary_blocker") == expected_v198_primary_blocker
            and coverage.get("primary_blocker") == expected_v198_primary_blocker
        ),
        "v198_source_pins_valid": source.get("passed") is True,
        "v198_real_replay_provenance": v198.probe_has_real_replay_provenance(probe),
        "v198_action_override_disabled": (
            probe.get("transition_value_action_override_enabled") is False
        ),
        "v198_specificity_gate_context_enabled": (
            probe.get("source_key_specificity_gate_enabled") is True
        ),
        "v198_candidate_key_diagnostics_enabled": (
            probe.get("candidate_key_coverage_diagnostics_enabled") is True
        ),
        "v198_high_specificity_action_incomplete_confirmed": (
            _int(
                _mapping(coverage.get("live_high_specificity_candidate_breakdown")).get(
                    "present_but_action_incomplete_count"
                )
            )
            > 0
        ),
        "v198_high_specificity_any_complete_zero": (
            _int(combined.get("high_specificity_any_complete_count")) == 0
        ),
        "v198_archive_path_pin_matches_expected": (
            expected_v198_archive_path == EXPECTED_V198_ARCHIVE_PATH
        ),
        "v198_archive_sha256_pin_matches_expected": (
            expected_v198_archive_sha256 == EXPECTED_V198_ARCHIVE_SHA256
        ),
        "v195_artifact_digest_matches_expected": (
            artifact_digest == expected_v195_artifact_digest
        ),
        "v198_lifecycle_closed": all(
            v198_report.get(key) is False for key in lifecycle_keys
        )
        and v198_report.get("non_promoted") is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v199_source_pin_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v198_report_exact_digest": expected_v198_report_exact_digest,
        "observed_v198_report_exact_digest": v198_report.get("exact_digest"),
        "v198_report_exact_digest_validation": exact_validation,
        "expected_v198_classification": expected_v198_classification,
        "observed_v198_classification": classification.get("primary"),
        "expected_v198_route": expected_v198_route,
        "observed_v198_route": route.get("selected_route"),
        "expected_v198_primary_blocker": expected_v198_primary_blocker,
        "observed_v198_primary_blocker": route.get("primary_blocker"),
        "expected_v198_archive_path": expected_v198_archive_path,
        "expected_v198_archive_sha256": expected_v198_archive_sha256,
        "expected_v195_artifact_digest": expected_v195_artifact_digest,
        "observed_v195_artifact_digest": artifact_digest,
        "checks": checks,
    }


def run_high_specificity_action_complete_audit(
    *,
    artifact: Mapping[str, object],
    broad_seeds: Sequence[int],
    carrion_fixture_seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    scorer = load_transition_value_scorer_artifact(artifact)
    broad = _run_action_complete_scope(
        scope="broad",
        scorer=scorer,
        seeds=broad_seeds,
        ticks=ticks,
    )
    carrion = _run_action_complete_scope(
        scope="carrion_only",
        scorer=scorer,
        seeds=carrion_fixture_seeds,
        ticks=ticks,
    )
    combined = combine_action_complete_scopes(broad, carrion)
    return {
        "policy": "m3_carrion_survivor_continuation_v199_high_specificity_action_complete_audit_v1",
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
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "broad": broad,
        "carrion_only": carrion,
        "combined": combined,
    }


def _run_action_complete_scope(
    *,
    scope: str,
    scorer: object,
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    stats = _empty_audit_stats()
    per_seed: list[dict[str, object]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        probe_policy = _mind_v3_policy(
            seed=seed,
            founder_template=None,
            transition_value_scorer=scorer,
            transition_value_action_override=False,
            transition_value_source_key_specificity_gate_enabled=True,
            transition_value_candidate_key_coverage_diagnostics_enabled=True,
        )
        if scope == "broad":
            probe_world = SimulationWorld(
                WorldConfig(seed=seed, max_ticks=int(ticks)),
                policy=probe_policy,
            )
        elif scope == "carrion_only":
            probe_world = _fixture_world(
                fixture_name="carrion_only",
                seed=seed,
                ticks=int(ticks),
                policy=probe_policy,
            )
        else:
            raise ValueError(f"unsupported v199 audit scope: {scope}")
        probe = _run_world(
            world=probe_world,
            seed=seed,
            ticks=int(ticks),
            trajectory_output_path=None,
            trajectory_split_id=f"v199_{scope}_action_complete_probe",
        )
        seed_stats = _audit_stats_from_decisions(
            probe_world.policy_decision_diagnostics_records,
            scope=scope,
            seed=seed,
        )
        _merge_run_stats(seed_stats, probe)
        _merge_audit_payload(stats, seed_stats)
        per_seed.append(
            {
                "scope": scope,
                "seed": seed,
                "probe_alive_agents": probe.get("alive_agents"),
                "probe_births": probe.get("births"),
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
                "coverage": _finalize_audit_stats(seed_stats, include_keys=False),
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v199_high_specificity_scope_action_complete_audit_v1",
        "scope": scope,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "per_seed": per_seed,
        **_finalize_audit_stats(stats, include_keys=True),
    }


def _audit_stats_from_decisions(
    decision_diagnostics: Sequence[Mapping[str, object] | None],
    *,
    scope: str,
    seed: int,
) -> dict[str, object]:
    stats = _empty_audit_stats()
    for diagnostic in decision_diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        transition = _mapping(diagnostic.get("transition_value_scorer"))
        if not transition:
            continue
        stats["decision_count"] = _int(stats.get("decision_count")) + 1
        if transition.get("runtime_action_selection_changed") is True:
            stats["runtime_action_selection_changed_count"] = (
                _int(stats.get("runtime_action_selection_changed_count")) + 1
            )
        final_requested = str(
            transition.get("final_requested_action")
            or transition.get("requested_action")
            or ""
        )
        if final_requested:
            _counter(stats, "final_requested_action_counts").update(
                [final_requested]
            )
            _counter(stats, "requested_action_counts").update([final_requested])
        candidate_key_coverage = transition.get("candidate_key_coverage")
        if not isinstance(candidate_key_coverage, list):
            stats["missing_candidate_key_coverage_count"] = (
                _int(stats.get("missing_candidate_key_coverage_count")) + 1
            )
            continue
        stats["candidate_key_coverage_decision_count"] = (
            _int(stats.get("candidate_key_coverage_decision_count")) + 1
        )
        for item in candidate_key_coverage:
            if not isinstance(item, Mapping):
                continue
            category = str(item.get("source_key_category") or "")
            if category not in HIGH_SPECIFICITY_CATEGORIES:
                continue
            _record_high_specificity_candidate(
                stats,
                item=item,
                scope=scope,
                seed=seed,
            )
    return stats


def _record_high_specificity_candidate(
    stats: dict[str, object],
    *,
    item: Mapping[str, object],
    scope: str,
    seed: int,
) -> None:
    key = str(item.get("source_key") or "")
    category = str(item.get("source_key_category") or "")
    valid_actions = tuple(str(action) for action in item.get("valid_actions", []))
    support_counts = _mapping(item.get("valid_action_support_counts"))
    missing_actions = tuple(
        action for action in valid_actions if _int(support_counts.get(action)) <= 0
    )
    below_floor_actions = tuple(
        str(action)
        for action in item.get("valid_actions_below_observed_support_floor", [])
    )
    key_present = item.get("key_present") is True
    complete = item.get("complete_for_current_valid_actions") is True
    floor_satisfied = (
        item.get("observed_support_floor_satisfied_for_all_valid_actions") is True
    )
    clear_best = item.get("clear_best_valid_action") is True
    imputed = item.get("has_imputed_valid_action_score") is True
    for target in (
        stats,
        _group_stats(stats, "by_scope", scope),
        _group_stats(stats, "by_seed", f"{scope}:{int(seed)}"),
        _group_stats(stats, "by_high_specificity_category", category),
    ):
        _update_candidate_group(
            target,
            key_present=key_present,
            complete=complete,
            floor_satisfied=floor_satisfied,
            clear_best=clear_best,
            imputed=imputed,
            valid_actions=valid_actions,
            missing_actions=missing_actions,
            below_floor_actions=below_floor_actions,
        )
    if key_present:
        key_stats = _candidate_key_stats(stats, key)
        key_stats["candidate_key_category"] = category
        _counter(key_stats, "scope_counts").update([scope])
        _counter(key_stats, "seed_counts").update([f"{scope}:{int(seed)}"])
        _update_candidate_group(
            key_stats,
            key_present=key_present,
            complete=complete,
            floor_satisfied=floor_satisfied,
            clear_best=clear_best,
            imputed=imputed,
            valid_actions=valid_actions,
            missing_actions=missing_actions,
            below_floor_actions=below_floor_actions,
        )


def _update_candidate_group(
    stats: dict[str, object],
    *,
    key_present: bool,
    complete: bool,
    floor_satisfied: bool,
    clear_best: bool,
    imputed: bool,
    valid_actions: Sequence[str],
    missing_actions: Sequence[str],
    below_floor_actions: Sequence[str],
) -> None:
    stats["high_specificity_candidate_evaluated_count"] = (
        _int(stats.get("high_specificity_candidate_evaluated_count")) + 1
    )
    _counter(stats, "valid_action_demand_counts").update(valid_actions)
    if key_present:
        stats["key_present_count"] = _int(stats.get("key_present_count")) + 1
    else:
        stats["key_absent_count"] = _int(stats.get("key_absent_count")) + 1
        _counter(stats, "absent_valid_action_demand_counts").update(valid_actions)
        return
    if complete:
        stats["complete_for_current_valid_actions_count"] = (
            _int(stats.get("complete_for_current_valid_actions_count")) + 1
        )
    else:
        stats["present_but_action_incomplete_count"] = (
            _int(stats.get("present_but_action_incomplete_count")) + 1
        )
        _counter(stats, "missing_current_valid_action_counts").update(
            missing_actions
        )
    if floor_satisfied:
        stats["observed_support_floor_satisfied_count"] = (
            _int(stats.get("observed_support_floor_satisfied_count")) + 1
        )
    elif complete:
        stats["present_complete_but_observed_support_floor_failed_count"] = (
            _int(
                stats.get("present_complete_but_observed_support_floor_failed_count")
            )
            + 1
        )
        _counter(stats, "below_observed_support_floor_action_counts").update(
            below_floor_actions
        )
    if imputed:
        stats["imputed_valid_action_score_count"] = (
            _int(stats.get("imputed_valid_action_score_count")) + 1
        )
    if key_present and complete and floor_satisfied and not clear_best:
        stats["present_complete_floor_satisfied_but_unclear_best_count"] = (
            _int(stats.get("present_complete_floor_satisfied_but_unclear_best_count"))
            + 1
        )


def combine_action_complete_scopes(*scopes: Mapping[str, object]) -> dict[str, object]:
    stats = _empty_audit_stats()
    for scope in scopes:
        _merge_audit_payload(stats, scope)
    return {
        "policy": "m3_carrion_survivor_continuation_v199_combined_high_specificity_action_complete_audit_v1",
        **_finalize_audit_stats(stats, include_keys=True),
    }


def assess_action_complete_contract(audit: Mapping[str, object]) -> dict[str, object]:
    combined = _mapping(audit.get("combined"))
    blocker = "instrumentation_or_provenance_failure"
    if audit_has_real_replay_provenance(audit):
        if _int(combined.get("missing_candidate_key_coverage_count")) > 0:
            blocker = "instrumentation_or_provenance_failure"
        elif _int(combined.get("runtime_action_selection_changed_count")) > 0:
            blocker = "instrumentation_or_provenance_failure"
        elif _int(combined.get("key_present_count")) == 0:
            blocker = "high_specificity_absent_dominates"
        elif (
            _int(combined.get("present_but_action_incomplete_count")) > 0
            or _int(
                combined.get(
                    "present_complete_but_observed_support_floor_failed_count"
                )
            )
            > 0
            or _int(combined.get("imputed_valid_action_score_count")) > 0
            or _int(
                combined.get(
                    "present_complete_floor_satisfied_but_unclear_best_count"
                )
            )
            > 0
        ):
            blocker = "high_specificity_action_complete_or_support_floor_gap_confirmed"
        elif _int(combined.get("key_absent_count")) > _int(
            combined.get("key_present_count")
        ):
            blocker = "high_specificity_absent_dominates"
        else:
            blocker = "high_specificity_action_complete_contract_unexpectedly_satisfied"
    return {
        "policy": "m3_carrion_survivor_continuation_v199_action_complete_contract_assessment_v1",
        "audit_real_replay_provenance": audit_has_real_replay_provenance(audit),
        "primary_blocker": blocker,
        "high_specificity_candidate_evaluated_count": combined.get(
            "high_specificity_candidate_evaluated_count"
        ),
        "key_absent_count": combined.get("key_absent_count"),
        "key_present_count": combined.get("key_present_count"),
        "present_but_action_incomplete_count": combined.get(
            "present_but_action_incomplete_count"
        ),
        "present_complete_but_observed_support_floor_failed_count": combined.get(
            "present_complete_but_observed_support_floor_failed_count"
        ),
        "missing_current_valid_action_counts": combined.get(
            "missing_current_valid_action_counts"
        ),
        "below_observed_support_floor_action_counts": combined.get(
            "below_observed_support_floor_action_counts"
        ),
        "action_complete_contract_ready_for_future_slice_4_consideration": False,
    }


def route_decision_for_v199(
    *,
    source_validation: Mapping[str, object],
    action_audit: Mapping[str, object],
    contract_assessment: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif not audit_has_real_replay_provenance(action_audit):
        route = INSTRUMENTATION_ROUTE
    else:
        blocker = str(contract_assessment.get("primary_blocker") or "")
        if blocker == "high_specificity_absent_dominates":
            route = PUBLIC_MODEL_CAPACITY_ROUTE
        elif blocker == "high_specificity_action_complete_or_support_floor_gap_confirmed":
            route = ACTION_COMPLETE_REPAIR_ROUTE
        else:
            route = INSTRUMENTATION_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v199_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "source_pins_valid": source_validation.get("passed") is True,
        "real_replay_provenance": audit_has_real_replay_provenance(action_audit),
        "primary_blocker": contract_assessment.get("primary_blocker"),
        "direct_slice_4_training_allowed": False,
        "future_explicit_slice_4_training_route_authorized": False,
        "slice_4_training_consumed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "gate_relaxation_allowed": False,
        "support_generation_allowed": False,
        "support_expansion_allowed": False,
        "promotion_authorized": False,
        "rationale": route_rationale(route),
    }


def classification_for_v199(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v199_"
    if source_validation.get("passed") is not True:
        return prefix + "source_pins_invalid_closed_no_training"
    route = str(route_decision.get("selected_route") or "")
    if route == ACTION_COMPLETE_REPAIR_ROUTE:
        return (
            prefix
            + "high_specificity_action_complete_gap_confirmed_routes_to_repair_design_no_training"
        )
    if route == PUBLIC_MODEL_CAPACITY_ROUTE:
        return (
            prefix
            + "high_specificity_absent_dominates_routes_to_public_model_capacity_no_training"
        )
    if route == INSTRUMENTATION_ROUTE:
        return (
            prefix
            + "high_specificity_candidate_key_instrumentation_repair_no_training"
        )
    return prefix + "closed_no_training"


def skipped_action_complete_audit(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v199_high_specificity_action_complete_audit_v1",
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
        "support_generation_ran": False,
        "support_expansion_ran": False,
    }


def audit_has_real_replay_provenance(audit: Mapping[str, object]) -> bool:
    return (
        audit.get("ran") is True
        and audit.get("policy")
        == "m3_carrion_survivor_continuation_v199_high_specificity_action_complete_audit_v1"
        and audit.get("diagnostics_provenance") == "real_replay"
        and audit.get("diagnostics_override_used") is False
        and audit.get("diagnostics_only") is True
        and audit.get("transition_value_action_override_enabled") is False
        and audit.get("source_key_specificity_gate_enabled") is True
        and audit.get("candidate_key_coverage_diagnostics_enabled") is True
        and audit.get("training_rerun") is False
        and audit.get("slice_4_training_consumed") is False
        and audit.get("support_generation_ran") is False
        and audit.get("support_expansion_ran") is False
    )


def contract(
    *,
    expected_v198_report_exact_digest: str,
    expected_v198_route: str,
    expected_v198_archive_path: str,
    expected_v198_archive_sha256: str,
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
        "requires_v198_report_digest": expected_v198_report_exact_digest,
        "requires_v198_route": expected_v198_route,
        "requires_v198_archive_path": expected_v198_archive_path,
        "requires_v198_archive_sha256": expected_v198_archive_sha256,
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
            "diagnostics_insufficient": INSTRUMENTATION_ROUTE,
            "high_specificity_absent_dominates": PUBLIC_MODEL_CAPACITY_ROUTE,
            "high_specificity_present_but_action_incomplete_or_support_floor_gap": (
                ACTION_COMPLETE_REPAIR_ROUTE
            ),
            "no_direct_training_route": True,
        },
    }


def required_action_complete_contract() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v199_required_action_complete_contract_v1",
        "before_future_specificity_gated_slice_4_training_can_be_considered": {
            "source_pins_valid": True,
            "real_replay_provenance_required": True,
            "transition_value_action_override_enabled_during_probe": False,
            "runtime_action_selection_changed": False,
            "for_each_live_high_specificity_candidate_key": (
                "every current valid action must have an observed non-imputed score"
            ),
            "observed_support_floor": (
                "every current valid action score must meet the configured observed "
                "support floor"
            ),
            "clear_best_action_required": True,
            "low_specificity_keys_rejected": True,
            "public_policy_visible_inputs_only": True,
        },
        "v199_authorizes_slice_4_training": False,
        "v199_authorizes_runtime_integration": False,
        "v199_authorizes_gate_relaxation": False,
        "v199_authorizes_support_generation_or_expansion": False,
        "v199_authorizes_promotion": False,
    }


def historical_evidence_checkpoint() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v199_historical_evidence_checkpoint_v1",
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
        "v198": (
            "live high-specificity keys exist but are incomplete for current valid "
            "actions; v199 audits that contract without training"
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
    if route == ACTION_COMPLETE_REPAIR_ROUTE:
        return (
            "Real replay confirms high-specific candidate keys overlap live "
            "decisions, but those keys lack at least one current valid action or "
            "otherwise fail the action-complete support contract. The next route "
            "is a no-training repair design, not slice-4 training."
        )
    if route == PUBLIC_MODEL_CAPACITY_ROUTE:
        return (
            "High-specific keys are absent on the live public replay surface, so "
            "the next route is a public masked model-capacity harness, not support "
            "expansion or training."
        )
    if route == INSTRUMENTATION_ROUTE:
        return (
            "The audit could not prove real replay candidate-key provenance, so it "
            "fails closed to instrumentation repair."
        )
    if route == STOP_ROUTE:
        return "Required v198 pins or lifecycle facts failed validation."
    return "Closed no-training route."


def _empty_audit_stats() -> dict[str, object]:
    return {
        "decision_count": 0,
        "candidate_key_coverage_decision_count": 0,
        "missing_candidate_key_coverage_count": 0,
        "runtime_action_selection_changed_count": 0,
        "high_specificity_candidate_evaluated_count": 0,
        "key_absent_count": 0,
        "key_present_count": 0,
        "complete_for_current_valid_actions_count": 0,
        "present_but_action_incomplete_count": 0,
        "observed_support_floor_satisfied_count": 0,
        "present_complete_but_observed_support_floor_failed_count": 0,
        "imputed_valid_action_score_count": 0,
        "present_complete_floor_satisfied_but_unclear_best_count": 0,
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": 0,
        "valid_action_demand_counts": Counter(),
        "absent_valid_action_demand_counts": Counter(),
        "missing_current_valid_action_counts": Counter(),
        "below_observed_support_floor_action_counts": Counter(),
        "requested_action_counts": Counter(),
        "final_requested_action_counts": Counter(),
        "by_scope": defaultdict(_empty_audit_stats),
        "by_seed": defaultdict(_empty_audit_stats),
        "by_high_specificity_category": defaultdict(_empty_audit_stats),
        "candidate_key_action_coverage": defaultdict(_empty_candidate_key_stats),
    }


def _empty_candidate_key_stats() -> dict[str, object]:
    stats = _empty_audit_stats()
    stats.pop("by_scope")
    stats.pop("by_seed")
    stats.pop("by_high_specificity_category")
    stats.pop("candidate_key_action_coverage")
    stats.update(
        {
            "candidate_key": None,
            "candidate_key_digest": None,
            "candidate_key_category": None,
            "scope_counts": Counter(),
            "seed_counts": Counter(),
        }
    )
    return stats


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


def _merge_audit_payload(target: dict[str, object], source: Mapping[str, object]) -> None:
    _merge_audit_stats(target, source)
    for key in ("by_scope", "by_seed", "by_high_specificity_category"):
        groups = source.get(key)
        if isinstance(groups, Mapping):
            for group_name, group_stats in groups.items():
                _merge_audit_stats(
                    _group_stats(target, key, str(group_name)),
                    _mapping(group_stats),
                )
    candidates = source.get("candidate_key_action_coverage")
    if isinstance(candidates, Mapping):
        for candidate_key, candidate_stats in candidates.items():
            _merge_audit_stats(
                _candidate_key_stats(target, str(candidate_key)),
                _mapping(candidate_stats),
            )
            _candidate_key_stats(target, str(candidate_key))[
                "candidate_key_category"
            ] = _mapping(candidate_stats).get("candidate_key_category")
            _counter(
                _candidate_key_stats(target, str(candidate_key)), "scope_counts"
            ).update(_int_counter(_mapping(candidate_stats).get("scope_counts")))
            _counter(
                _candidate_key_stats(target, str(candidate_key)), "seed_counts"
            ).update(_int_counter(_mapping(candidate_stats).get("seed_counts")))


def _merge_audit_stats(target: dict[str, object], source: Mapping[str, object]) -> None:
    for key in (
        "decision_count",
        "candidate_key_coverage_decision_count",
        "missing_candidate_key_coverage_count",
        "runtime_action_selection_changed_count",
        "high_specificity_candidate_evaluated_count",
        "key_absent_count",
        "key_present_count",
        "complete_for_current_valid_actions_count",
        "present_but_action_incomplete_count",
        "observed_support_floor_satisfied_count",
        "present_complete_but_observed_support_floor_failed_count",
        "imputed_valid_action_score_count",
        "present_complete_floor_satisfied_but_unclear_best_count",
        "heuristic_action_source_count",
        "unsupported_requested_action_count",
        "unsupported_resolved_action_count",
    ):
        target[key] = _int(target.get(key)) + _int(source.get(key))
    for key in (
        "valid_action_demand_counts",
        "absent_valid_action_demand_counts",
        "missing_current_valid_action_counts",
        "below_observed_support_floor_action_counts",
        "requested_action_counts",
        "final_requested_action_counts",
    ):
        _counter(target, key).update(_int_counter(source.get(key)))


def _finalize_audit_stats(
    stats: Mapping[str, object],
    *,
    include_keys: bool,
) -> dict[str, object]:
    candidate_count = _int(stats.get("high_specificity_candidate_evaluated_count"))
    present_count = _int(stats.get("key_present_count"))
    absent_count = _int(stats.get("key_absent_count"))
    complete_count = _int(stats.get("complete_for_current_valid_actions_count"))
    incomplete_count = _int(stats.get("present_but_action_incomplete_count"))
    floor_count = _int(stats.get("observed_support_floor_satisfied_count"))
    floor_failed_count = _int(
        stats.get("present_complete_but_observed_support_floor_failed_count")
    )
    result: dict[str, object] = {
        "decision_count": _int(stats.get("decision_count")),
        "candidate_key_coverage_decision_count": _int(
            stats.get("candidate_key_coverage_decision_count")
        ),
        "missing_candidate_key_coverage_count": _int(
            stats.get("missing_candidate_key_coverage_count")
        ),
        "runtime_action_selection_changed_count": _int(
            stats.get("runtime_action_selection_changed_count")
        ),
        "high_specificity_candidate_evaluated_count": candidate_count,
        "key_absent_count": absent_count,
        "key_absent_share": _share(absent_count, candidate_count),
        "key_present_count": present_count,
        "key_present_share": _share(present_count, candidate_count),
        "complete_for_current_valid_actions_count": complete_count,
        "complete_for_current_valid_actions_share": _share(
            complete_count,
            candidate_count,
        ),
        "present_but_action_incomplete_count": incomplete_count,
        "present_but_action_incomplete_share": _share(
            incomplete_count,
            candidate_count,
        ),
        "observed_support_floor_satisfied_count": floor_count,
        "observed_support_floor_satisfied_share": _share(
            floor_count,
            candidate_count,
        ),
        "present_complete_but_observed_support_floor_failed_count": (
            floor_failed_count
        ),
        "present_complete_but_observed_support_floor_failed_share": _share(
            floor_failed_count,
            candidate_count,
        ),
        "imputed_valid_action_score_count": _int(
            stats.get("imputed_valid_action_score_count")
        ),
        "present_complete_floor_satisfied_but_unclear_best_count": _int(
            stats.get("present_complete_floor_satisfied_but_unclear_best_count")
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
        "valid_action_demand_counts": dict(
            sorted(_int_counter(stats.get("valid_action_demand_counts")).items())
        ),
        "absent_valid_action_demand_counts": dict(
            sorted(
                _int_counter(stats.get("absent_valid_action_demand_counts")).items()
            )
        ),
        "missing_current_valid_action_counts": dict(
            sorted(
                _int_counter(stats.get("missing_current_valid_action_counts")).items()
            )
        ),
        "below_observed_support_floor_action_counts": dict(
            sorted(
                _int_counter(
                    stats.get("below_observed_support_floor_action_counts")
                ).items()
            )
        ),
        "requested_action_counts": dict(
            sorted(_int_counter(stats.get("requested_action_counts")).items())
        ),
        "final_requested_action_counts": dict(
            sorted(_int_counter(stats.get("final_requested_action_counts")).items())
        ),
    }
    groups = stats.get("by_scope")
    if isinstance(groups, Mapping):
        result["by_scope"] = {
            str(name): _finalize_audit_stats(_mapping(group), include_keys=False)
            for name, group in sorted(groups.items())
        }
    groups = stats.get("by_seed")
    if isinstance(groups, Mapping):
        result["by_seed"] = {
            str(name): _finalize_audit_stats(_mapping(group), include_keys=False)
            for name, group in sorted(groups.items())
        }
    groups = stats.get("by_high_specificity_category")
    if isinstance(groups, Mapping):
        result["by_high_specificity_category"] = {
            category: _finalize_audit_stats(
                _mapping(groups.get(category)), include_keys=False
            )
            for category in HIGH_SPECIFICITY_CATEGORY_ORDER
        }
    if include_keys:
        candidates = stats.get("candidate_key_action_coverage")
        if isinstance(candidates, Mapping):
            result["candidate_key_action_coverage"] = {
                str(key): _finalize_candidate_key_stats(_mapping(value))
                for key, value in sorted(candidates.items())
            }
            result["candidate_key_action_coverage_count"] = len(candidates)
    return result


def _finalize_candidate_key_stats(stats: Mapping[str, object]) -> dict[str, object]:
    result = _finalize_audit_stats(stats, include_keys=False)
    candidate_key = str(stats.get("candidate_key") or "")
    result.update(
        {
            "candidate_key": candidate_key,
            "candidate_key_digest": stats.get("candidate_key_digest")
            or stable_payload_digest(candidate_key),
            "candidate_key_category": stats.get("candidate_key_category"),
            "scope_counts": dict(
                sorted(_int_counter(stats.get("scope_counts")).items())
            ),
            "seed_counts": dict(sorted(_int_counter(stats.get("seed_counts")).items())),
        }
    )
    return result


def _group_stats(
    stats: dict[str, object],
    group_name: str,
    key: str,
) -> dict[str, object]:
    groups = stats.get(group_name)
    if not isinstance(groups, defaultdict):
        groups = defaultdict(_empty_audit_stats)
        stats[group_name] = groups
    return groups[key]


def _candidate_key_stats(stats: dict[str, object], key: str) -> dict[str, object]:
    candidates = stats.get("candidate_key_action_coverage")
    if not isinstance(candidates, defaultdict):
        candidates = defaultdict(_empty_candidate_key_stats)
        stats["candidate_key_action_coverage"] = candidates
    result = candidates[key]
    result["candidate_key"] = key
    result["candidate_key_digest"] = stable_payload_digest(key)
    return result


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
