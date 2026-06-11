from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind.candidate_campaign import _float, _int, _mapping, write_json
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    load_jsonl_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v180_transition_row_policy_training import (
    DEFAULT_ARTIFACT_OUTPUT_PATH as DEFAULT_V180_ARTIFACT_PATH,
    DEFAULT_BROAD_SEEDS,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_OUTPUT_PATH as DEFAULT_V180_REPORT_PATH,
    DEFAULT_TICKS,
    DEFAULT_TRANSITION_DATASET_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v181_v180_failure_response import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V181_REPORT_PATH,
    EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    EXPECTED_V180_ARTIFACT_DIGEST,
    EXPECTED_V180_CLASSIFICATION,
    EXPECTED_V180_REPORT_EXACT_DIGEST,
    V181_AUTOPSY_CLASSIFICATION,
    _artifact_support_summary,
)
from evolution_sim.mind.evaluation_harness import (
    _aggregate_runs,
    _mind_v3_policy,
    _run_once,
    run_controlled_fixture_policy_suite,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.transition_value_scorer import load_transition_value_scorer_artifact

M3_CARRION_SURVIVOR_CONTINUATION_V182_IMPUTED_ABSTENTION_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v182_imputed_abstention_design_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V182_IMPUTED_ABSTENTION_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v182_imputed_abstention_design_v1"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v182-carrion-survivor-continuation-imputed-abstention-design.json"
)
EXPECTED_V181_REPORT_EXACT_DIGEST = (
    "a89745e71daf8c6cc1651ad960a3098776509bca6e7cda84ca259fdbc57f5751"
)
EXPECTED_V181_CLASSIFICATION = V181_AUTOPSY_CLASSIFICATION
EXPECTED_DATASET_DIGEST = EXPECTED_V179_TRANSITION_DATASET_DIGEST
DEFAULT_OBSERVED_SUPPORT_FLOOR = 2
V182_SOURCE_INVALID_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v182_imputed_abstention_design_"
    "source_invalid_closed_no_training"
)
V182_SHADOW_SKIPPED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v182_imputed_abstention_design_"
    "shadow_eval_skipped_no_training"
)
V182_EXACT_SUPPORT_EXPANSION_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v182_imputed_abstention_design_"
    "strict_support_routes_to_exact_transition_support_expansion_no_training"
)
V182_REVIEW_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v182_imputed_abstention_design_"
    "strict_support_abstention_ready_for_review_no_training"
)


def run_carrion_survivor_continuation_v182_imputed_abstention_design(
    *,
    v181_report_path: str | Path = DEFAULT_V181_REPORT_PATH,
    v180_report_path: str | Path = DEFAULT_V180_REPORT_PATH,
    v180_artifact_path: str | Path = DEFAULT_V180_ARTIFACT_PATH,
    transition_dataset_path: str | Path = DEFAULT_TRANSITION_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v181_report_exact_digest: str = EXPECTED_V181_REPORT_EXACT_DIGEST,
    expected_v180_report_exact_digest: str = EXPECTED_V180_REPORT_EXACT_DIGEST,
    expected_v180_artifact_digest: str = EXPECTED_V180_ARTIFACT_DIGEST,
    expected_dataset_digest: str = EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    expected_v181_classification: str = EXPECTED_V181_CLASSIFICATION,
    expected_v180_classification: str = EXPECTED_V180_CLASSIFICATION,
    observed_support_floor: int = DEFAULT_OBSERVED_SUPPORT_FLOOR,
    run_shadow_evaluation: bool = True,
    broad_seeds: Sequence[int] = DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = DEFAULT_TICKS,
) -> dict[str, object]:
    v181_report = load_json_report(v181_report_path)
    v180_report = load_json_report(v180_report_path)
    artifact = load_json_report(v180_artifact_path)
    rows = [dict(row) for row in load_jsonl_dataset(transition_dataset_path)]
    floor = max(1, int(observed_support_floor))
    source_validation = _source_validation(
        v181_report=v181_report,
        v180_report=v180_report,
        artifact=artifact,
        rows=rows,
        expected_v181_report_exact_digest=expected_v181_report_exact_digest,
        expected_v180_report_exact_digest=expected_v180_report_exact_digest,
        expected_v180_artifact_digest=expected_v180_artifact_digest,
        expected_dataset_digest=expected_dataset_digest,
        expected_v181_classification=expected_v181_classification,
        expected_v180_classification=expected_v180_classification,
    )
    artifact_support = _artifact_support_summary(artifact)
    shadow_evaluation = (
        _shadow_evaluation(
            artifact=artifact,
            observed_support_floor=floor,
            broad_seeds=broad_seeds,
            carrion_fixture_seeds=carrion_fixture_seeds,
            ticks=ticks,
        )
        if source_validation.get("passed") is True and run_shadow_evaluation
        else _skipped_shadow_evaluation(
            "source_validation_failed"
            if source_validation.get("passed") is not True
            else "run_shadow_evaluation_false"
        )
    )
    design_diagnostics = _design_diagnostics(
        artifact_support=artifact_support,
        shadow_evaluation=shadow_evaluation,
        observed_support_floor=floor,
    )
    route = _route(
        source_validation=source_validation,
        shadow_evaluation=shadow_evaluation,
        design_diagnostics=design_diagnostics,
    )
    classification = _classification(
        source_validation=source_validation,
        shadow_evaluation=shadow_evaluation,
        route=route,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V182_IMPUTED_ABSTENTION_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V182_IMPUTED_ABSTENTION_POLICY,
        "contract": _contract(observed_support_floor=floor),
        "inputs": {
            "v181_report": str(v181_report_path),
            "v180_report": str(v180_report_path),
            "v180_artifact": str(v180_artifact_path),
            "transition_dataset": str(transition_dataset_path),
            "expected_v181_report_exact_digest": expected_v181_report_exact_digest,
            "expected_v180_report_exact_digest": expected_v180_report_exact_digest,
            "expected_v180_artifact_digest": expected_v180_artifact_digest,
            "expected_dataset_digest": expected_dataset_digest,
            "expected_v181_classification": expected_v181_classification,
            "expected_v180_classification": expected_v180_classification,
            "observed_support_floor": floor,
            "run_shadow_evaluation": bool(run_shadow_evaluation),
            "broad_seeds": [int(seed) for seed in broad_seeds],
            "carrion_fixture_seeds": [int(seed) for seed in carrion_fixture_seeds],
            "ticks": int(ticks),
        },
        "source_validation": source_validation,
        "artifact_support": artifact_support,
        "abstention_design": _abstention_design(floor),
        "shadow_evaluation": shadow_evaluation,
        "design_diagnostics": design_diagnostics,
        "route": route,
        "classification": {"primary": classification, "labels": [classification]},
        **_lifecycle_flags(shadow_eval_ran=shadow_evaluation.get("ran") is True),
    }
    report["exact_digest"] = _digest_without_exact(report)
    write_json(output_path, report)
    return report


def _source_validation(
    *,
    v181_report: Mapping[str, object],
    v180_report: Mapping[str, object],
    artifact: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v181_report_exact_digest: str,
    expected_v180_report_exact_digest: str,
    expected_v180_artifact_digest: str,
    expected_dataset_digest: str,
    expected_v181_classification: str,
    expected_v180_classification: str,
) -> dict[str, object]:
    observed_v181_exact = str(v181_report.get("exact_digest") or "")
    computed_v181_exact = _digest_without_exact(v181_report)
    observed_v180_exact = str(v180_report.get("exact_digest") or "")
    computed_v180_exact = _digest_without_exact(v180_report)
    observed_artifact_digest = stable_payload_digest(artifact)
    observed_dataset_digest = stable_payload_digest([dict(row) for row in rows])
    v181_classification = _mapping(v181_report.get("classification"))
    v180_classification = _mapping(v180_report.get("classification"))
    v181_source = _mapping(v181_report.get("source_validation"))
    v181_mechanism = _mapping(v181_report.get("failure_mechanism"))
    v180_artifact = _mapping(v180_report.get("artifact"))
    v180_dataset = _mapping(v180_report.get("dataset"))
    checks = {
        "v181_report_exact_digest_valid": observed_v181_exact == computed_v181_exact,
        "v181_report_exact_digest_matches_expected": (
            observed_v181_exact == expected_v181_report_exact_digest
        ),
        "v181_classification_matches_expected": (
            v181_classification.get("primary") == expected_v181_classification
        ),
        "v181_source_validation_passed": v181_source.get("passed") is True,
        "v181_training_not_run": v181_report.get("training_ran") is False,
        "v181_training_artifact_not_created": (
            v181_report.get("training_artifact_created") is False
        ),
        "v181_runtime_artifact_not_created": (
            v181_report.get("runtime_artifact_created") is False
        ),
        "v181_runtime_action_selection_unchanged": (
            v181_report.get("runtime_action_selection_changed") is False
        ),
        "v181_promotion_not_authorized": (
            v181_report.get("promotion_authorized") is False
        ),
        "v181_slice_2_training_not_consumed": (
            v181_mechanism.get("slice_2_training_consumed") is False
        ),
        "v180_report_exact_digest_valid": observed_v180_exact == computed_v180_exact,
        "v180_report_exact_digest_matches_expected": (
            observed_v180_exact == expected_v180_report_exact_digest
        ),
        "v180_classification_matches_expected": (
            v180_classification.get("primary") == expected_v180_classification
        ),
        "v180_acceptance_failed": _mapping(v180_report.get("acceptance")).get("passed")
        is False,
        "v180_training_ran": v180_report.get("training_ran") is True,
        "v180_training_artifact_created": (
            v180_report.get("training_artifact_created") is True
        ),
        "v180_runtime_artifact_not_created": (
            v180_report.get("runtime_artifact_created") is False
        ),
        "v180_runtime_action_selection_unchanged": (
            v180_report.get("runtime_action_selection_changed") is False
        ),
        "v180_promotion_not_authorized": (
            v180_report.get("promotion_authorized") is False
        ),
        "v180_artifact_digest_matches_expected": (
            observed_artifact_digest == expected_v180_artifact_digest
        ),
        "v180_artifact_digest_matches_v180_report": (
            observed_artifact_digest == str(v180_artifact.get("digest") or "")
        ),
        "dataset_digest_matches_expected": (
            observed_dataset_digest == expected_dataset_digest
        ),
        "dataset_digest_matches_v180_report": (
            observed_dataset_digest == str(v180_dataset.get("dataset_digest") or "")
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v182_source_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "observed_v181_report_exact_digest": observed_v181_exact,
        "computed_v181_report_exact_digest": computed_v181_exact,
        "expected_v181_report_exact_digest": expected_v181_report_exact_digest,
        "observed_v180_report_exact_digest": observed_v180_exact,
        "computed_v180_report_exact_digest": computed_v180_exact,
        "expected_v180_report_exact_digest": expected_v180_report_exact_digest,
        "observed_v180_artifact_digest": observed_artifact_digest,
        "expected_v180_artifact_digest": expected_v180_artifact_digest,
        "observed_dataset_digest": observed_dataset_digest,
        "expected_dataset_digest": expected_dataset_digest,
        "dataset_row_count": len(rows),
    }


def _shadow_evaluation(
    *,
    artifact: Mapping[str, object],
    observed_support_floor: int,
    broad_seeds: Sequence[int],
    carrion_fixture_seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    scorer = load_transition_value_scorer_artifact(artifact)
    broad_baseline_runs = [
        _run_once(
            seed=int(seed),
            ticks=int(ticks),
            policy=_mind_v3_policy(seed=int(seed), founder_template=None),
        )
        for seed in broad_seeds
    ]
    broad_candidate_runs = [
        _run_once(
            seed=int(seed),
            ticks=int(ticks),
            policy=_mind_v3_policy(
                seed=int(seed),
                founder_template=None,
                transition_value_scorer=scorer,
                transition_value_action_override=True,
                transition_value_action_override_source_integrity_passed=True,
                transition_value_min_observed_support_count=observed_support_floor,
            ),
        )
        for seed in broad_seeds
    ]
    fixture_baseline = run_controlled_fixture_policy_suite(
        suite="basic",
        fixture_names=["carrion_only"],
        seeds=[int(seed) for seed in carrion_fixture_seeds],
        ticks=int(ticks),
        learned_policy_factory=lambda _fixture, seed: _mind_v3_policy(
            seed=int(seed),
            founder_template=None,
        ),
        learned_policy_key="linear_mind_v3",
        learned_policy_name="linear_mind_v3",
    )
    fixture_candidate = run_controlled_fixture_policy_suite(
        suite="basic",
        fixture_names=["carrion_only"],
        seeds=[int(seed) for seed in carrion_fixture_seeds],
        ticks=int(ticks),
        learned_policy_factory=lambda _fixture, seed: _mind_v3_policy(
            seed=int(seed),
            founder_template=None,
            transition_value_scorer=scorer,
            transition_value_action_override=True,
            transition_value_action_override_source_integrity_passed=True,
            transition_value_min_observed_support_count=observed_support_floor,
        ),
        learned_policy_key="v182_imputed_abstention_design",
        learned_policy_name="v182_imputed_abstention_design",
    )
    fixture_baseline_payload = _fixture_policy_payload(
        fixture_baseline,
        "linear_mind_v3",
    )
    fixture_candidate_payload = _fixture_policy_payload(
        fixture_candidate,
        "v182_imputed_abstention_design",
    )
    per_seed_deltas = _per_seed_deltas(
        label="broad",
        baseline_runs=broad_baseline_runs,
        candidate_runs=broad_candidate_runs,
    ) + _per_seed_deltas(
        label="carrion_only",
        baseline_runs=fixture_baseline_payload["runs"],
        candidate_runs=fixture_candidate_payload["runs"],
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v182_shadow_design_eval_v1",
        "ran": True,
        "mode": "offline_shadow_design_eval_no_training_no_artifact",
        "observed_support_floor": int(observed_support_floor),
        "ticks": int(ticks),
        "broad": {
            "seeds": [int(seed) for seed in broad_seeds],
            "baseline": {
                "runs": broad_baseline_runs,
                "aggregate": _aggregate_runs(broad_baseline_runs),
            },
            "candidate": {
                "runs": broad_candidate_runs,
                "aggregate": _aggregate_runs(broad_candidate_runs),
            },
        },
        "controlled_fixture": {
            "suite": "basic",
            "fixture": "carrion_only",
            "seeds": [int(seed) for seed in carrion_fixture_seeds],
            "baseline": fixture_baseline_payload,
            "candidate": fixture_candidate_payload,
        },
        "per_seed_alive_birth_deltas": per_seed_deltas,
    }


def _design_diagnostics(
    *,
    artifact_support: Mapping[str, object],
    shadow_evaluation: Mapping[str, object],
    observed_support_floor: int,
) -> dict[str, object]:
    broad = _suite_design_diagnostics(
        suite="broad",
        suite_payload=_mapping(shadow_evaluation.get("broad")),
        deltas=[
            item
            for item in shadow_evaluation.get("per_seed_alive_birth_deltas", [])
            if isinstance(item, Mapping) and item.get("suite") == "broad"
        ],
    )
    carrion = _suite_design_diagnostics(
        suite="carrion_only",
        suite_payload=_mapping(shadow_evaluation.get("controlled_fixture")),
        deltas=[
            item
            for item in shadow_evaluation.get("per_seed_alive_birth_deltas", [])
            if isinstance(item, Mapping) and item.get("suite") == "carrion_only"
        ],
    )
    carrion_observed_support_count = _int(
        _mapping(carrion.get("transition_value_scorer_diagnostics")).get(
            "observed_support_floor_satisfied_count"
        )
    )
    broad_regressions = _int(broad.get("per_seed_alive_or_birth_regression_count"))
    return {
        "policy": "m3_carrion_survivor_continuation_v182_design_diagnostics_v1",
        "observed_support_floor": int(observed_support_floor),
        "artifact_imputed_action_stat_count": artifact_support.get(
            "imputed_action_stat_count"
        ),
        "artifact_imputed_action_stat_share": artifact_support.get(
            "imputed_action_stat_share"
        ),
        "shadow_evaluation_ran": shadow_evaluation.get("ran") is True,
        "broad": broad,
        "controlled_fixture": carrion,
        "strict_observed_support_leaves_carrion_coverage_zero": (
            shadow_evaluation.get("ran") is True
            and carrion_observed_support_count == 0
        ),
        "broad_regressions_remain": (
            shadow_evaluation.get("ran") is True and broad_regressions > 0
        ),
    }


def _suite_design_diagnostics(
    *,
    suite: str,
    suite_payload: Mapping[str, object],
    deltas: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidate = _mapping(suite_payload.get("candidate"))
    candidate_aggregate = _mapping(candidate.get("aggregate"))
    baseline = _mapping(suite_payload.get("baseline"))
    baseline_aggregate = _mapping(baseline.get("aggregate"))
    diagnostics = _mapping(candidate_aggregate.get("transition_value_scorer_diagnostics"))
    regressions = [
        dict(delta)
        for delta in deltas
        if _int(delta.get("alive_agents_delta")) < 0
        or _int(delta.get("births_delta")) < 0
    ]
    return {
        "suite": suite,
        "baseline_outcomes": _outcome_brief(baseline_aggregate),
        "candidate_outcomes": _outcome_brief(candidate_aggregate),
        "candidate_minus_baseline": {
            "alive_agents_mean_delta": (
                _float(candidate_aggregate.get("alive_agents_mean"))
                - _float(baseline_aggregate.get("alive_agents_mean"))
            ),
            "births_mean_delta": (
                _float(candidate_aggregate.get("births_mean"))
                - _float(baseline_aggregate.get("births_mean"))
            ),
        },
        "transition_value_scorer_diagnostics": diagnostics,
        "per_seed_alive_birth_deltas": [dict(delta) for delta in deltas],
        "per_seed_alive_or_birth_regression_count": len(regressions),
        "per_seed_alive_or_birth_regressions": regressions,
    }


def _outcome_brief(aggregate: Mapping[str, object]) -> dict[str, object]:
    outcome = _mapping(aggregate.get("outcome_metrics"))
    return {
        "alive_agents_mean": aggregate.get("alive_agents_mean"),
        "births_mean": aggregate.get("births_mean"),
        "dominant_requested_action": aggregate.get("dominant_requested_action"),
        "dominant_requested_action_share": aggregate.get(
            "dominant_requested_action_share"
        ),
        "heuristic_action_source_count": aggregate.get("heuristic_action_source_count"),
        "total_terminal_alive_agents": outcome.get("total_terminal_alive_agents"),
        "terminal_survivor_run_count": outcome.get("terminal_survivor_run_count"),
    }


def _route(
    *,
    source_validation: Mapping[str, object],
    shadow_evaluation: Mapping[str, object],
    design_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True:
        return {
            "policy": "m3_carrion_survivor_continuation_v182_route_v1",
            "next_route": "fix_v181_v180_source_pins_before_any_slice_2_training",
            "reason": "source_validation_failed",
            "slice_2_training_consumed": False,
            "training_authorized": False,
        }
    if shadow_evaluation.get("ran") is not True:
        return {
            "policy": "m3_carrion_survivor_continuation_v182_route_v1",
            "next_route": "run_v182_shadow_design_eval_before_any_slice_2_training",
            "reason": "shadow_evaluation_not_run",
            "slice_2_training_consumed": False,
            "training_authorized": False,
        }
    if (
        design_diagnostics.get("strict_observed_support_leaves_carrion_coverage_zero")
        is True
        or design_diagnostics.get("broad_regressions_remain") is True
    ):
        return {
            "policy": "m3_carrion_survivor_continuation_v182_route_v1",
            "next_route": (
                "exact_transition_support_expansion_before_any_slice_2_training"
            ),
            "reason": (
                "carrion_observed_support_zero_or_broad_regressions_remain"
            ),
            "slice_2_training_consumed": False,
            "training_authorized": False,
        }
    return {
        "policy": "m3_carrion_survivor_continuation_v182_route_v1",
        "next_route": "review_v182_design_evidence_before_any_slice_2_training",
        "reason": "strict_support_abstention_shadow_eval_has_no_forced_expansion_blocker",
        "slice_2_training_consumed": False,
        "training_authorized": False,
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    shadow_evaluation: Mapping[str, object],
    route: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True:
        return V182_SOURCE_INVALID_CLASSIFICATION
    if shadow_evaluation.get("ran") is not True:
        return V182_SHADOW_SKIPPED_CLASSIFICATION
    if route.get("next_route") == (
        "exact_transition_support_expansion_before_any_slice_2_training"
    ):
        return V182_EXACT_SUPPORT_EXPANSION_CLASSIFICATION
    return V182_REVIEW_CLASSIFICATION


def _contract(*, observed_support_floor: int) -> dict[str, object]:
    return {
        "failure_response_to": "v181_v180_failure_response_autopsy",
        "diagnostics_only": True,
        "training_ran": False,
        "training_artifact_created": False,
        "slice_2_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_changed": False,
        "default_runtime_behavior_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "observed_support_floor": int(observed_support_floor),
        "override_abstains_when_any_valid_action_score_is_imputed": True,
        "override_abstains_when_any_valid_action_score_is_below_observed_floor": True,
        "uses_fixture_identity_for_training_or_policy": False,
        "uses_private_world_state_for_training_or_policy": False,
        "heuristic_action_selection_added": False,
    }


def _abstention_design(observed_support_floor: int) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v182_imputed_abstention_v1",
        "observed_support_floor": int(observed_support_floor),
        "required_for_override": [
            "source_integrity_passed",
            "all_currently_valid_actions_have_scores",
            "no_currently_valid_action_score_is_imputed",
            "all_currently_valid_action_observed_counts_meet_floor",
            "clear_best_valid_action",
            "predicted_action_is_currently_valid",
        ],
        "imputed_marker": "component_means.imputed_unobserved_action",
        "training_effect": "none",
        "runtime_default_effect": "none",
    }


def _lifecycle_flags(*, shadow_eval_ran: bool) -> dict[str, object]:
    return {
        "training_ran": False,
        "training_artifact_created": False,
        "fit_ran": False,
        "diagnostic_shadow_eval_ran": bool(shadow_eval_ran),
        "shadow_eval_ran": bool(shadow_eval_ran),
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_ran": False,
        "replay_viewer_schema_changed": False,
        "slice_2_training_consumed": False,
        "non_promoted": True,
    }


def _skipped_shadow_evaluation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v182_shadow_design_eval_v1",
        "ran": False,
        "reason": reason,
    }


def _fixture_policy_payload(
    suite_report: Mapping[str, object],
    policy_key: str,
) -> dict[str, object]:
    fixtures = suite_report.get("fixtures")
    fixture = _mapping(fixtures[0] if isinstance(fixtures, list) and fixtures else {})
    comparison = _mapping(fixture.get("comparison"))
    return dict(_mapping(comparison.get(policy_key)))


def _per_seed_deltas(
    *,
    label: str,
    baseline_runs: Sequence[Mapping[str, object]],
    candidate_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline_by_seed = {int(run.get("seed", -1)): run for run in baseline_runs}
    deltas: list[dict[str, object]] = []
    for candidate in candidate_runs:
        seed = int(candidate.get("seed", -1))
        baseline = _mapping(baseline_by_seed.get(seed))
        deltas.append(
            {
                "suite": label,
                "seed": seed,
                "baseline_alive_agents": _int(baseline.get("alive_agents")),
                "candidate_alive_agents": _int(candidate.get("alive_agents")),
                "alive_agents_delta": _int(candidate.get("alive_agents"))
                - _int(baseline.get("alive_agents")),
                "baseline_births": _int(baseline.get("births")),
                "candidate_births": _int(candidate.get("births")),
                "births_delta": _int(candidate.get("births"))
                - _int(baseline.get("births")),
            }
        )
    return deltas


def _digest_without_exact(report: Mapping[str, object]) -> str:
    payload = json.loads(json.dumps(report, sort_keys=True))
    if isinstance(payload, dict):
        payload.pop("exact_digest", None)
    return stable_payload_digest(payload)
