from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import _float, _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    load_jsonl_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v178_transition_row_dataset_audit import (
    V179_SOURCE_PRODUCER,
    validate_v178_transition_row_training_authorization_report,
)
from evolution_sim.mind.evaluation_harness import (
    _aggregate_runs,
    _mind_v3_policy,
    _run_once,
    run_controlled_fixture_policy_suite,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.rollout_context import RolloutContextState
from evolution_sim.mind.transition_value_scorer import (
    FORBIDDEN_SCORE_FEATURE_TOKENS,
    MIND_V3_TRANSITION_VALUE_MODEL_ID,
    MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
    load_transition_value_scorer_artifact,
    transition_value_feature_keys,
)

M3_CARRION_SURVIVOR_CONTINUATION_V180_TRANSITION_ROW_POLICY_TRAINING_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v180_transition_row_policy_training_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V180_TRANSITION_ROW_POLICY_TRAINING_POLICY = (
    "opt_in_m3_carrion_survivor_continuation_v180_transition_row_policy_training_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V180_ARTIFACT_POLICY = (
    "opt_in_m3_carrion_survivor_continuation_v180_public_transition_row_policy_v1"
)
V180_CANDIDATE_KEY = "v180_transition_row_policy"
DEFAULT_AUTHORIZATION_REPORT_PATH = Path(
    "output/mind/"
    "mind-v3-v178-carrion-survivor-continuation-transition-row-dataset-audit-"
    "v179-expanded.json"
)
DEFAULT_TRANSITION_DATASET_PATH = Path(
    "output/mind/"
    "mind-v3-v179-carrion-survivor-continuation-compact-transition-rows.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v180-carrion-survivor-continuation-transition-row-policy-training.json"
)
DEFAULT_ARTIFACT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v180-carrion-survivor-continuation-transition-row-policy-artifact.json"
)
EXPECTED_V178_AUTHORIZATION_REPORT_EXACT_DIGEST = (
    "8d10ee77315de87a15ed296d87482d2335008009e4bcce2d72f60399ede92923"
)
EXPECTED_V179_TRANSITION_DATASET_DIGEST = (
    "df9043666639dc9d606e118c5c3733efc0ae9ac1c76a7126cb8d009b1591e9bf"
)
EXPECTED_V178_AUTHORIZATION_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
    "valid_support_ready_transition_row_training_authorized"
)
DEFAULT_BROAD_SEEDS = (5, 13, 19, 29, 37, 41)
DEFAULT_CARRION_FIXTURE_SEEDS = (13, 19, 29, 37, 41, 43)
DEFAULT_TICKS = 120
DEFAULT_FEATURE_KEY_LIMIT = 6
DEFAULT_UNOBSERVED_ACTION_UTILITY = -5.0
MAX_DOMINANT_REQUESTED_ACTION_SHARE = 0.50


def run_carrion_survivor_continuation_v180_transition_row_policy_training(
    *,
    authorization_report_path: str | Path = DEFAULT_AUTHORIZATION_REPORT_PATH,
    transition_dataset_path: str | Path = DEFAULT_TRANSITION_DATASET_PATH,
    artifact_output_path: str | Path = DEFAULT_ARTIFACT_OUTPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_authorization_report_exact_digest: str = (
        EXPECTED_V178_AUTHORIZATION_REPORT_EXACT_DIGEST
    ),
    expected_dataset_digest: str = EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    expected_authorization_classification: str = (
        EXPECTED_V178_AUTHORIZATION_CLASSIFICATION
    ),
    expected_source_producer: str = V179_SOURCE_PRODUCER,
    feature_key_limit: int = DEFAULT_FEATURE_KEY_LIMIT,
    unobserved_action_utility: float = DEFAULT_UNOBSERVED_ACTION_UTILITY,
    run_evaluation: bool = True,
    broad_seeds: Sequence[int] = DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = DEFAULT_TICKS,
) -> dict[str, object]:
    authorization_report = load_json_report(authorization_report_path)
    rows = [dict(row) for row in load_jsonl_dataset(transition_dataset_path)]
    authorization_validation = (
        validate_v178_transition_row_training_authorization_report(
            authorization_report,
            expected_exact_digest=expected_authorization_report_exact_digest,
            expected_dataset_digest=expected_dataset_digest,
            expected_classification=expected_authorization_classification,
            expected_source_producer=expected_source_producer,
        )
    )
    source_validation = _source_validation(
        rows=rows,
        authorization_validation=authorization_validation,
        expected_dataset_digest=expected_dataset_digest,
    )
    artifact: dict[str, object]
    training: dict[str, object]
    evaluation: dict[str, object]
    acceptance: dict[str, object]
    if source_validation.get("passed") is not True:
        artifact = _artifact_status(
            artifact_output_path,
            created=False,
            reason="source_validation_failed",
        )
        training = _skipped_training("source_validation_failed")
        evaluation = _skipped_evaluation("source_validation_failed")
        acceptance = _closed_acceptance("source_validation_failed")
    else:
        artifact_payload, training = build_v180_transition_row_policy_artifact(
            rows,
            dataset_digest=expected_dataset_digest,
            authorization_report_exact_digest=expected_authorization_report_exact_digest,
            feature_key_limit=feature_key_limit,
            unobserved_action_utility=unobserved_action_utility,
        )
        write_json(artifact_output_path, artifact_payload)
        artifact_digest = stable_payload_digest(artifact_payload)
        artifact = _artifact_status(
            artifact_output_path,
            created=True,
            reason=None,
            payload=artifact_payload,
            digest=artifact_digest,
        )
        if run_evaluation:
            evaluation = run_v180_shadow_evaluation(
                artifact=artifact_payload,
                broad_seeds=broad_seeds,
                carrion_fixture_seeds=carrion_fixture_seeds,
                ticks=ticks,
            )
            acceptance = _acceptance(evaluation)
        else:
            evaluation = _skipped_evaluation("run_evaluation_false")
            acceptance = _closed_acceptance("evaluation_not_run")
    classification = _classification(
        source_validation=source_validation,
        training=training,
        evaluation=evaluation,
        acceptance=acceptance,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V180_TRANSITION_ROW_POLICY_TRAINING_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V180_TRANSITION_ROW_POLICY_TRAINING_POLICY,
        "contract": _contract(),
        "inputs": {
            "authorization_report": str(authorization_report_path),
            "transition_dataset": str(transition_dataset_path),
            "artifact_output": str(artifact_output_path),
            "expected_authorization_report_exact_digest": (
                expected_authorization_report_exact_digest
            ),
            "expected_dataset_digest": expected_dataset_digest,
            "expected_authorization_classification": (
                expected_authorization_classification
            ),
            "expected_source_producer": expected_source_producer,
            "feature_key_limit": int(feature_key_limit),
            "unobserved_action_utility": _round(unobserved_action_utility),
            "run_evaluation": bool(run_evaluation),
            "broad_seeds": [int(seed) for seed in broad_seeds],
            "carrion_fixture_seeds": [int(seed) for seed in carrion_fixture_seeds],
            "ticks": int(ticks),
        },
        "authorization_report_validation": authorization_validation,
        "source_validation": source_validation,
        "dataset": {
            "path": str(transition_dataset_path),
            "row_count": len(rows),
            "dataset_digest": stable_payload_digest(rows),
        },
        "artifact": artifact,
        "training": training,
        "evaluation": evaluation,
        "acceptance": acceptance,
        "classification": {"primary": classification, "labels": [classification]},
        **_lifecycle_flags(
            training_ran=training.get("ran") is True,
            training_artifact_created=artifact.get("created") is True,
            shadow_eval_ran=evaluation.get("ran") is True,
        ),
    }
    report["exact_digest"] = _digest_without_exact(report)
    write_json(output_path, report)
    return report


def build_v180_transition_row_policy_artifact(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset_digest: str,
    authorization_report_exact_digest: str,
    feature_key_limit: int = DEFAULT_FEATURE_KEY_LIMIT,
    unobserved_action_utility: float = DEFAULT_UNOBSERVED_ACTION_UTILITY,
) -> tuple[dict[str, object], dict[str, object]]:
    key_limit = max(1, min(int(feature_key_limit), 8))
    action_totals: defaultdict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "utility_sum": 0.0, "components": defaultdict(float)}
    )
    buckets: defaultdict[str, defaultdict[str, dict[str, object]]] = defaultdict(
        lambda: defaultdict(
            lambda: {"count": 0, "utility_sum": 0.0, "components": defaultdict(float)}
        )
    )
    row_failures: list[dict[str, object]] = []
    row_action_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        try:
            features = _mapping(row.get("trainable_public_features"))
            observation = _mapping(features.get("current_public_observation"))
            action_mask = _mapping(features.get("current_public_action_mask"))
            action = str(features.get("forced_action") or "")
            if action not in ACTION_NAMES:
                raise ValueError("forced_action_not_in_action_names")
            utility = _transition_row_utility(row)
            feature_keys = transition_value_feature_keys(
                observation_input=observation,
                valid_action_mask=action_mask,
                state=RolloutContextState(),
            )[:key_limit]
        except (TypeError, ValueError) as exc:
            row_failures.append(
                {
                    "row_index": row_index,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        row_action_counts.update([action])
        _add_utility(action_totals[action], utility)
        for feature_key in feature_keys:
            _add_utility(buckets[str(feature_key)][action], utility)
    action_priors = _action_priors(
        action_totals,
        unobserved_action_utility=unobserved_action_utility,
    )
    feature_action_utility = {
        feature_key: {
            action: _finalize_action_stats(
                action_bucket.get(action),
                imputed_utility=action_priors[action],
                imputed=action not in action_bucket,
            )
            for action in ACTION_NAMES
        }
        for feature_key, action_bucket in sorted(buckets.items())
    }
    artifact = {
        "schema_version": MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
        "artifact_policy": M3_CARRION_SURVIVOR_CONTINUATION_V180_ARTIFACT_POLICY,
        "model_id": MIND_V3_TRANSITION_VALUE_MODEL_ID,
        "diagnostics_only": False,
        "explicit_opt_in_required": True,
        "runtime_action_selection_authorized": False,
        "runtime_policy_change_requires_explicit_flag": True,
        "promotion_authorized": False,
        "built_from": {
            "source": "v179_exact_branch_transition_rows",
            "training_row_count": len(rows),
            "dataset_digest": dataset_digest,
            "authorization_report_exact_digest": authorization_report_exact_digest,
            "feature_key_limit": key_limit,
            "global_feature_key_excluded": True,
        },
        "utility_policy": {
            "target": "v179_short_horizon_public_transition_outcome_utility",
            "weights": _utility_weights(),
            "unobserved_action_utility": _round(unobserved_action_utility),
            "estimates_action_utility_not_action_frequency": True,
        },
        "feature_policy": {
            "current_decision_inputs": [
                "public observation_input",
                "public action_mask",
            ],
            "prior_history_inputs": [
                "same-agent finalized public trajectory rows before current decision"
            ],
            "excluded_score_features": [
                "seed identity",
                "fixture identity",
                "private world state",
                "future rows at decision time",
                "heuristic recommendations",
                "source path",
                "provenance",
            ],
        },
        "utility_tables": {
            "policy": (
                "v180_mean_v179_transition_utility_by_public_feature_key_and_action_v1"
            ),
            "feature_action_utility": feature_action_utility,
        },
        "action_support_counts": _counter_dict(row_action_counts),
        "unsupported_action_rejection": {
            "policy": "runtime_scores_only_currently_valid_actions_v1",
            "invalid_actions_are_never_selected": True,
        },
    }
    loaded = load_transition_value_scorer_artifact(artifact)
    roundtrip = {
        "policy": "v180_transition_value_artifact_load_check_v1",
        "loaded": loaded.artifact == artifact,
        "schema_version": loaded.artifact.get("schema_version"),
    }
    artifact_leakage_scan = _artifact_feature_leakage_scan(artifact)
    dominant = _dominant_share(row_action_counts)
    training = {
        "policy": "m3_carrion_survivor_continuation_v180_training_summary_v1",
        "ran": True,
        "passed": not row_failures and artifact_leakage_scan.get("passed") is True,
        "training_row_count": len(rows),
        "accepted_training_row_count": sum(row_action_counts.values()),
        "row_failure_count": len(row_failures),
        "row_failures": row_failures[:16],
        "feature_key_count": len(feature_action_utility),
        "feature_key_limit": key_limit,
        "global_feature_key_excluded": True,
        "action_support_counts": _counter_dict(row_action_counts),
        "dominant_training_action": dominant["action"],
        "dominant_training_action_share": dominant["share"],
        "artifact_feature_leakage_scan": artifact_leakage_scan,
        "artifact_load_check": roundtrip,
    }
    return artifact, training


def run_v180_shadow_evaluation(
    *,
    artifact: Mapping[str, object],
    broad_seeds: Sequence[int] = DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = DEFAULT_TICKS,
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
        ),
        learned_policy_key=V180_CANDIDATE_KEY,
        learned_policy_name=V180_CANDIDATE_KEY,
    )
    broad_baseline = _aggregate_runs(broad_baseline_runs)
    broad_candidate = _aggregate_runs(broad_candidate_runs)
    fixture_baseline_payload = _fixture_policy_payload(
        fixture_baseline,
        "linear_mind_v3",
    )
    fixture_candidate_payload = _fixture_policy_payload(
        fixture_candidate,
        V180_CANDIDATE_KEY,
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
        "policy": "m3_carrion_survivor_continuation_v180_shadow_evaluation_v1",
        "ran": True,
        "mode": "offline_shadow_controlled_eval",
        "ticks": int(ticks),
        "broad": {
            "seeds": [int(seed) for seed in broad_seeds],
            "baseline": {"runs": broad_baseline_runs, "aggregate": broad_baseline},
            "candidate": {
                "runs": broad_candidate_runs,
                "aggregate": broad_candidate,
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


def _source_validation(
    *,
    rows: Sequence[Mapping[str, object]],
    authorization_validation: Mapping[str, object],
    expected_dataset_digest: str,
) -> dict[str, object]:
    dataset_digest = stable_payload_digest([dict(row) for row in rows])
    checks = {
        "authorization_report_validation_passed": (
            authorization_validation.get("passed") is True
        ),
        "dataset_digest_matches_expected": dataset_digest == expected_dataset_digest,
        "dataset_digest_matches_authorization_report": (
            dataset_digest
            == str(authorization_validation.get("observed_dataset_digest") or "")
        ),
        "dataset_non_empty": bool(rows),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v180_source_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "dataset_digest": dataset_digest,
        "expected_dataset_digest": expected_dataset_digest,
        "row_count": len(rows),
    }


def _transition_row_utility(row: Mapping[str, object]) -> dict[str, object]:
    summary = _mapping(row.get("short_horizon_public_outcome_summary"))
    target = _mapping(summary.get("target_terminal"))
    components = {
        "target_terminal_alive": 1.0 if target.get("alive") is True else 0.0,
        "alive_agents": _float(summary.get("alive_agents")),
        "births": _float(summary.get("births")),
        "current_reward_total": _float(summary.get("current_reward_total")),
        "current_resource_gain": _float(summary.get("current_resource_gain")),
        "terminal_energy_ratio": _float(target.get("energy_ratio")),
        "terminal_hydration_ratio": _float(target.get("hydration_ratio")),
        "terminal_health_ratio": _float(target.get("health_ratio")),
    }
    total = sum(
        float(components[name]) * float(weight)
        for name, weight in _utility_weights().items()
    )
    return {"utility": _round(total), "components": components}


def _utility_weights() -> dict[str, float]:
    return {
        "target_terminal_alive": 5.0,
        "alive_agents": 0.05,
        "births": 0.08,
        "current_reward_total": 0.25,
        "current_resource_gain": 0.15,
        "terminal_energy_ratio": 0.25,
        "terminal_hydration_ratio": 0.25,
        "terminal_health_ratio": 0.25,
    }


def _add_utility(bucket: dict[str, object], utility: Mapping[str, object]) -> None:
    bucket["count"] = int(bucket.get("count", 0)) + 1
    bucket["utility_sum"] = float(bucket.get("utility_sum", 0.0)) + _float(
        utility.get("utility")
    )
    components = bucket.get("components")
    if not isinstance(components, defaultdict):
        components = defaultdict(float)
        bucket["components"] = components
    for name, value in _mapping(utility.get("components")).items():
        components[str(name)] += _float(value)


def _action_priors(
    action_totals: Mapping[str, Mapping[str, object]],
    *,
    unobserved_action_utility: float,
) -> dict[str, float]:
    priors: dict[str, float] = {}
    for action in ACTION_NAMES:
        bucket = _mapping(action_totals.get(action))
        count = _int(bucket.get("count"))
        priors[action] = (
            _round(_float(bucket.get("utility_sum")) / float(count))
            if count > 0
            else _round(unobserved_action_utility)
        )
    return priors


def _finalize_action_stats(
    bucket: Mapping[str, object] | None,
    *,
    imputed_utility: float,
    imputed: bool,
) -> dict[str, object]:
    if bucket is None or imputed:
        return {
            "count": 1,
            "utility_mean": _round(imputed_utility),
            "component_means": {"imputed_unobserved_action": 1.0},
        }
    count = _int(bucket.get("count"))
    components = _mapping(bucket.get("components"))
    return {
        "count": count,
        "utility_mean": _round(_float(bucket.get("utility_sum")) / float(count)),
        "component_means": {
            name: _round(_float(value) / float(count))
            for name, value in sorted(components.items())
        },
    }


def _artifact_feature_leakage_scan(artifact: Mapping[str, object]) -> dict[str, object]:
    leaks: list[dict[str, object]] = []
    table = _mapping(_mapping(artifact.get("utility_tables")).get("feature_action_utility"))
    for key in sorted(table):
        lowered = str(key).lower()
        for token in FORBIDDEN_SCORE_FEATURE_TOKENS:
            if token in lowered:
                leaks.append({"key": str(key), "token": token})
    return {
        "policy": "v180_transition_row_policy_artifact_feature_leakage_scan_v1",
        "passed": not leaks,
        "forbidden_score_feature_tokens": list(FORBIDDEN_SCORE_FEATURE_TOKENS),
        "leakage_count": len(leaks),
        "leaks": leaks[:24],
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


def _acceptance(evaluation: Mapping[str, object]) -> dict[str, object]:
    broad_candidate = _mapping(_mapping(evaluation.get("broad")).get("candidate"))
    broad_aggregate = _mapping(broad_candidate.get("aggregate"))
    fixture_candidate = _mapping(
        _mapping(evaluation.get("controlled_fixture")).get("candidate")
    )
    fixture_aggregate = _mapping(fixture_candidate.get("aggregate"))
    fixture_outcomes = _mapping(fixture_aggregate.get("outcome_metrics"))
    terminal_survivors = _int(fixture_outcomes.get("total_terminal_alive_agents"))
    dominant_share = max(
        _float(broad_aggregate.get("dominant_requested_action_share")),
        _float(fixture_aggregate.get("dominant_requested_action_share")),
    )
    heuristic_count = _int(broad_aggregate.get("heuristic_action_source_count")) + _int(
        fixture_aggregate.get("heuristic_action_source_count")
    )
    regressions = [
        delta
        for delta in evaluation.get("per_seed_alive_birth_deltas", [])
        if isinstance(delta, Mapping)
        and (
            _int(delta.get("alive_agents_delta")) < 0
            or _int(delta.get("births_delta")) < 0
        )
    ]
    blockers: list[dict[str, object]] = []
    if terminal_survivors <= 0:
        blockers.append(
            _blocker(
                "carrion_only_terminal_survivors_zero",
                observed=terminal_survivors,
                required=">0",
            )
        )
    if dominant_share > MAX_DOMINANT_REQUESTED_ACTION_SHARE:
        blockers.append(
            _blocker(
                "dominant_requested_action_share_above_limit",
                observed=dominant_share,
                required=f"<= {MAX_DOMINANT_REQUESTED_ACTION_SHARE:.2f}",
            )
        )
    if heuristic_count != 0:
        blockers.append(
            _blocker(
                "heuristic_action_source_count_nonzero",
                observed=heuristic_count,
                required=0,
            )
        )
    if regressions:
        blockers.append(
            _blocker(
                "per_seed_alive_or_birth_regression",
                observed=regressions,
                required="all alive/birth deltas >= 0",
            )
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v180_acceptance_v1",
        "passed": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "controlled_fixture": {
            "fixture": "carrion_only",
            "total_terminal_alive_agents": terminal_survivors,
            "terminal_survivor_run_count": _int(
                fixture_outcomes.get("terminal_survivor_run_count")
            ),
            "alive_agents_mean": fixture_aggregate.get("alive_agents_mean"),
            "births_mean": fixture_aggregate.get("births_mean"),
        },
        "dominant_requested_action_share": _round(dominant_share),
        "heuristic_action_source_count": heuristic_count,
        "per_seed_alive_birth_regressions": regressions,
        "promotion_evidence": False,
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    training: Mapping[str, object],
    evaluation: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v180_transition_row_policy_training_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if training.get("passed") is not True:
        return prefix + "training_artifact_invalid_closed_no_promotion"
    if evaluation.get("ran") is not True:
        return prefix + "trained_shadow_eval_not_run_no_promotion"
    if acceptance.get("passed") is True:
        return prefix + "first_slice_shadow_acceptance_passed_no_promotion"
    return prefix + "first_slice_shadow_acceptance_failed_no_promotion"


def _contract() -> dict[str, object]:
    return {
        "explicit_opt_in_training_slice": True,
        "training_requires_v178_default_threshold_authorization_report": True,
        "lowered_min_support_thresholds_authorize_training": False,
        "input_dataset_digest_pinned": EXPECTED_V179_TRANSITION_DATASET_DIGEST,
        "authorization_report_exact_digest_pinned": (
            EXPECTED_V178_AUTHORIZATION_REPORT_EXACT_DIGEST
        ),
        "runtime_integration_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "evaluation_scope": "offline_shadow_controlled_eval_only",
        "promotion_evidence": False,
    }


def _lifecycle_flags(
    *,
    training_ran: bool,
    training_artifact_created: bool,
    shadow_eval_ran: bool,
) -> dict[str, object]:
    return {
        "training_ran": bool(training_ran),
        "training_artifact_created": bool(training_artifact_created),
        "fit_ran": bool(training_ran),
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": bool(shadow_eval_ran),
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_ran": False,
        "replay_viewer_schema_changed": False,
        "non_promoted": True,
    }


def _artifact_status(
    output_path: str | Path,
    *,
    created: bool,
    reason: str | None,
    payload: Mapping[str, object] | None = None,
    digest: str | None = None,
) -> dict[str, object]:
    status = {
        "created": bool(created),
        "path": str(output_path),
        "digest": digest,
        "schema_version": None,
        "artifact_policy": None,
        "reason": reason,
        "explicit_opt_in_required": True,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
    }
    if payload is not None:
        status.update(
            {
                "schema_version": payload.get("schema_version"),
                "artifact_policy": payload.get("artifact_policy"),
                "model_id": payload.get("model_id"),
                "training_row_count": _mapping(payload.get("built_from")).get(
                    "training_row_count"
                ),
                "feature_key_count": len(
                    _mapping(
                        _mapping(payload.get("utility_tables")).get(
                            "feature_action_utility"
                        )
                    )
                ),
            }
        )
    return status


def _skipped_training(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v180_training_summary_v1",
        "ran": False,
        "passed": False,
        "reason": reason,
    }


def _skipped_evaluation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v180_shadow_evaluation_v1",
        "ran": False,
        "reason": reason,
    }


def _closed_acceptance(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v180_acceptance_v1",
        "passed": False,
        "blocker_count": 1,
        "blockers": [_blocker(reason, observed=False, required=True)],
        "promotion_evidence": False,
    }


def _blocker(reason: str, *, observed: object, required: object) -> dict[str, object]:
    return {"reason": reason, "observed": observed, "required": required}


def _dominant_share(counter: Counter[str]) -> dict[str, object]:
    total = sum(counter.values())
    if total <= 0:
        return {"action": None, "count": 0, "share": 0.0}
    action, count = max(
        counter.items(),
        key=lambda item: (int(item[1]), -ACTION_NAMES.index(item[0])),
    )
    return {"action": action, "count": int(count), "share": _round(count / total)}


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {action: int(counter.get(action, 0)) for action in ACTION_NAMES if counter.get(action, 0)}


def _digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)
