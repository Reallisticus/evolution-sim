from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import _float, _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_train_eval import load_json_report
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
from evolution_sim.mind.evaluation_harness import _fixture_world, _mind_v3_policy
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.transition_value_scorer import load_transition_value_scorer_artifact

M3_CARRION_SURVIVOR_CONTINUATION_V181_V180_FAILURE_RESPONSE_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v181_v180_failure_response_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V181_V180_FAILURE_RESPONSE_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v181_v180_failure_response_v1"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v181-carrion-survivor-continuation-v180-failure-response-autopsy.json"
)
EXPECTED_V180_REPORT_EXACT_DIGEST = (
    "ab894238d9f6ed5041b4587fe69ceeddeb36fb6af1aeaffde6339d73a6f8146f"
)
EXPECTED_V180_ARTIFACT_DIGEST = (
    "66f4fd956111bbda031c122643de12ce556b7cdeb35b70f2d0031cc89fac82b0"
)
EXPECTED_V179_TRANSITION_DATASET_DIGEST = (
    "df9043666639dc9d606e118c5c3733efc0ae9ac1c76a7126cb8d009b1591e9bf"
)
EXPECTED_V180_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v180_transition_row_policy_training_"
    "first_slice_shadow_acceptance_failed_no_promotion"
)
V181_AUTOPSY_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v181_v180_failure_response_autopsy_"
    "imputed_sparse_eat_override_and_carrion_no_coverage_no_training"
)
V181_SOURCE_INVALID_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v181_v180_failure_response_"
    "source_invalid_closed_no_training"
)
V181_MECHANISM_UNCLEAR_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v181_v180_failure_response_"
    "mechanism_unclear_no_training"
)


def run_carrion_survivor_continuation_v181_v180_failure_response(
    *,
    v180_report_path: str | Path = DEFAULT_V180_REPORT_PATH,
    v180_artifact_path: str | Path = DEFAULT_V180_ARTIFACT_PATH,
    transition_dataset_path: str | Path = DEFAULT_TRANSITION_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v180_report_exact_digest: str = EXPECTED_V180_REPORT_EXACT_DIGEST,
    expected_v180_artifact_digest: str = EXPECTED_V180_ARTIFACT_DIGEST,
    expected_dataset_digest: str = EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    expected_v180_classification: str = EXPECTED_V180_CLASSIFICATION,
    run_trace_replay: bool = True,
    broad_seeds: Sequence[int] = DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = DEFAULT_TICKS,
) -> dict[str, object]:
    v180_report = load_json_report(v180_report_path)
    artifact = load_json_report(v180_artifact_path)
    rows = [dict(row) for row in load_jsonl_dataset(transition_dataset_path)]
    source_validation = _source_validation(
        v180_report=v180_report,
        artifact=artifact,
        rows=rows,
        expected_v180_report_exact_digest=expected_v180_report_exact_digest,
        expected_v180_artifact_digest=expected_v180_artifact_digest,
        expected_dataset_digest=expected_dataset_digest,
        expected_v180_classification=expected_v180_classification,
    )
    artifact_support = _artifact_support_summary(artifact)
    evaluation_autopsy = _evaluation_autopsy(v180_report)
    trace_replay = (
        _trace_replay_autopsy(
            artifact=artifact,
            broad_seeds=broad_seeds,
            carrion_fixture_seeds=carrion_fixture_seeds,
            ticks=ticks,
        )
        if source_validation.get("passed") is True and run_trace_replay
        else _skipped_trace_replay(
            "source_validation_failed"
            if source_validation.get("passed") is not True
            else "run_trace_replay_false"
        )
    )
    mechanism = _failure_mechanism(
        source_validation=source_validation,
        artifact_support=artifact_support,
        evaluation_autopsy=evaluation_autopsy,
        trace_replay=trace_replay,
    )
    classification = _classification(
        source_validation=source_validation,
        mechanism=mechanism,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V181_V180_FAILURE_RESPONSE_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V181_V180_FAILURE_RESPONSE_POLICY,
        "contract": _contract(),
        "inputs": {
            "v180_report": str(v180_report_path),
            "v180_artifact": str(v180_artifact_path),
            "transition_dataset": str(transition_dataset_path),
            "expected_v180_report_exact_digest": expected_v180_report_exact_digest,
            "expected_v180_artifact_digest": expected_v180_artifact_digest,
            "expected_dataset_digest": expected_dataset_digest,
            "expected_v180_classification": expected_v180_classification,
            "run_trace_replay": bool(run_trace_replay),
            "broad_seeds": [int(seed) for seed in broad_seeds],
            "carrion_fixture_seeds": [int(seed) for seed in carrion_fixture_seeds],
            "ticks": int(ticks),
        },
        "source_validation": source_validation,
        "artifact_support": artifact_support,
        "evaluation_autopsy": evaluation_autopsy,
        "trace_replay": trace_replay,
        "failure_mechanism": mechanism,
        "classification": {"primary": classification, "labels": [classification]},
        **_lifecycle_flags(trace_replay_ran=trace_replay.get("ran") is True),
    }
    report["exact_digest"] = _digest_without_exact(report)
    write_json(output_path, report)
    return report


def _source_validation(
    *,
    v180_report: Mapping[str, object],
    artifact: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v180_report_exact_digest: str,
    expected_v180_artifact_digest: str,
    expected_dataset_digest: str,
    expected_v180_classification: str,
) -> dict[str, object]:
    observed_report_exact = str(v180_report.get("exact_digest") or "")
    computed_report_exact = _digest_without_exact(v180_report)
    observed_artifact_digest = stable_payload_digest(artifact)
    dataset_digest = stable_payload_digest([dict(row) for row in rows])
    report_artifact = _mapping(v180_report.get("artifact"))
    report_dataset = _mapping(v180_report.get("dataset"))
    classification = _mapping(v180_report.get("classification"))
    acceptance = _mapping(v180_report.get("acceptance"))
    checks = {
        "v180_report_exact_digest_valid": observed_report_exact == computed_report_exact,
        "v180_report_exact_digest_matches_expected": (
            observed_report_exact == expected_v180_report_exact_digest
        ),
        "v180_classification_matches_expected": (
            classification.get("primary") == expected_v180_classification
        ),
        "v180_acceptance_failed": acceptance.get("passed") is False,
        "v180_training_ran": v180_report.get("training_ran") is True,
        "v180_training_artifact_created": (
            v180_report.get("training_artifact_created") is True
        ),
        "v180_source_validation_passed": (
            _mapping(v180_report.get("source_validation")).get("passed") is True
        ),
        "v180_authorization_report_validation_passed": (
            _mapping(v180_report.get("authorization_report_validation")).get("passed")
            is True
        ),
        "v180_artifact_created_field_true": report_artifact.get("created") is True,
        "v180_runtime_artifact_not_created": (
            v180_report.get("runtime_artifact_created") is False
        ),
        "v180_runtime_action_selection_unchanged": (
            v180_report.get("runtime_action_selection_changed") is False
        ),
        "v180_promotion_not_authorized": (
            v180_report.get("promotion_authorized") is False
        ),
        "artifact_digest_matches_expected": (
            observed_artifact_digest == expected_v180_artifact_digest
        ),
        "artifact_digest_matches_v180_report": (
            observed_artifact_digest == str(report_artifact.get("digest") or "")
        ),
        "dataset_digest_matches_expected": dataset_digest == expected_dataset_digest,
        "dataset_digest_matches_v180_report": (
            dataset_digest == str(report_dataset.get("dataset_digest") or "")
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v181_source_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "observed_v180_report_exact_digest": observed_report_exact,
        "computed_v180_report_exact_digest": computed_report_exact,
        "expected_v180_report_exact_digest": expected_v180_report_exact_digest,
        "observed_v180_artifact_digest": observed_artifact_digest,
        "expected_v180_artifact_digest": expected_v180_artifact_digest,
        "observed_dataset_digest": dataset_digest,
        "expected_dataset_digest": expected_dataset_digest,
        "dataset_row_count": len(rows),
    }


def _artifact_support_summary(artifact: Mapping[str, object]) -> dict[str, object]:
    table = _mapping(_mapping(artifact.get("utility_tables")).get("feature_action_utility"))
    imputed_by_action: Counter[str] = Counter()
    observed_by_action: Counter[str] = Counter()
    low_support_by_action: Counter[str] = Counter()
    imputed_stats = 0
    observed_stats = 0
    low_support_stats = 0
    feature_keys_with_any_imputed = 0
    feature_keys_with_all_actions_observed = 0
    observed_counts: list[int] = []
    for _feature_key, raw_action_stats in table.items():
        action_stats = _mapping(raw_action_stats)
        key_imputed = False
        all_observed = True
        for action in ACTION_NAMES:
            stats = _mapping(action_stats.get(action))
            count = _int(stats.get("count"))
            component_means = _mapping(stats.get("component_means"))
            imputed = "imputed_unobserved_action" in component_means
            if imputed:
                imputed_stats += 1
                imputed_by_action.update([action])
                key_imputed = True
                all_observed = False
            elif stats:
                observed_stats += 1
                observed_by_action.update([action])
                observed_counts.append(count)
                if count <= 1:
                    low_support_stats += 1
                    low_support_by_action.update([action])
            else:
                all_observed = False
        if key_imputed:
            feature_keys_with_any_imputed += 1
        if all_observed:
            feature_keys_with_all_actions_observed += 1
    total_stats = imputed_stats + observed_stats
    return {
        "policy": "m3_carrion_survivor_continuation_v181_artifact_support_scan_v1",
        "feature_key_count": len(table),
        "action_stat_count": total_stats,
        "observed_action_stat_count": observed_stats,
        "imputed_action_stat_count": imputed_stats,
        "imputed_action_stat_share": _round(_share(imputed_stats, total_stats)),
        "low_support_observed_action_stat_count": low_support_stats,
        "feature_keys_with_any_imputed_action": feature_keys_with_any_imputed,
        "feature_keys_with_all_actions_observed": feature_keys_with_all_actions_observed,
        "observed_action_count_min": min(observed_counts) if observed_counts else 0,
        "observed_action_count_max": max(observed_counts) if observed_counts else 0,
        "observed_by_action": dict(sorted(observed_by_action.items())),
        "imputed_by_action": dict(sorted(imputed_by_action.items())),
        "low_support_by_action": dict(sorted(low_support_by_action.items())),
    }


def _evaluation_autopsy(v180_report: Mapping[str, object]) -> dict[str, object]:
    evaluation = _mapping(v180_report.get("evaluation"))
    per_seed = [
        dict(item)
        for item in evaluation.get("per_seed_alive_birth_deltas", [])
        if isinstance(item, Mapping)
    ]
    return {
        "policy": "m3_carrion_survivor_continuation_v181_v180_eval_autopsy_v1",
        "v180_evaluation_ran": evaluation.get("ran") is True,
        "broad": _suite_autopsy(
            suite="broad",
            suite_payload=_mapping(evaluation.get("broad")),
            deltas=[item for item in per_seed if item.get("suite") == "broad"],
        ),
        "controlled_fixture": _suite_autopsy(
            suite="carrion_only",
            suite_payload=_mapping(evaluation.get("controlled_fixture")),
            deltas=[
                item for item in per_seed if item.get("suite") == "carrion_only"
            ],
        ),
        "per_seed_alive_birth_deltas": per_seed,
    }


def _suite_autopsy(
    *,
    suite: str,
    suite_payload: Mapping[str, object],
    deltas: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline = _mapping(suite_payload.get("baseline"))
    candidate = _mapping(suite_payload.get("candidate"))
    baseline_aggregate = _mapping(baseline.get("aggregate"))
    candidate_aggregate = _mapping(candidate.get("aggregate"))
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
            "alive_agents_mean_delta": _round(
                _float(candidate_aggregate.get("alive_agents_mean"))
                - _float(baseline_aggregate.get("alive_agents_mean"))
            ),
            "births_mean_delta": _round(
                _float(candidate_aggregate.get("births_mean"))
                - _float(baseline_aggregate.get("births_mean"))
            ),
        },
        "candidate_transition_value_scorer_diagnostics": _mapping(
            candidate_aggregate.get("transition_value_scorer_diagnostics")
        ),
        "run_count": len(candidate.get("runs", [])) if isinstance(candidate.get("runs"), list) else 0,
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


def _trace_replay_autopsy(
    *,
    artifact: Mapping[str, object],
    broad_seeds: Sequence[int],
    carrion_fixture_seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    scorer = load_transition_value_scorer_artifact(artifact)
    broad_runs = [
        _trace_run(
            suite="broad",
            seed=int(seed),
            ticks=int(ticks),
            artifact=artifact,
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
    carrion_runs = [
        _trace_run(
            suite="carrion_only",
            seed=int(seed),
            ticks=int(ticks),
            artifact=artifact,
            policy=_mind_v3_policy(
                seed=int(seed),
                founder_template=None,
                transition_value_scorer=scorer,
                transition_value_action_override=True,
                transition_value_action_override_source_integrity_passed=True,
            ),
            fixture_name="carrion_only",
        )
        for seed in carrion_fixture_seeds
    ]
    return {
        "policy": "m3_carrion_survivor_continuation_v181_diagnostic_trace_replay_v1",
        "ran": True,
        "mode": "diagnostics_only_trace_replay_no_training_no_artifact",
        "ticks": int(ticks),
        "broad": {"runs": broad_runs, "aggregate": _aggregate_trace_runs(broad_runs)},
        "controlled_fixture": {
            "fixture": "carrion_only",
            "runs": carrion_runs,
            "aggregate": _aggregate_trace_runs(carrion_runs),
        },
    }


def _trace_run(
    *,
    suite: str,
    seed: int,
    ticks: int,
    artifact: Mapping[str, object],
    policy: object,
    fixture_name: str | None = None,
) -> dict[str, object]:
    if fixture_name:
        world = _fixture_world(
            fixture_name=fixture_name,
            seed=seed,
            ticks=ticks,
            policy=policy,
        )
    else:
        world = SimulationWorld(
            WorldConfig(seed=seed, max_ticks=ticks),
            policy=policy,
        )
    result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
    summary = result.summary
    trace = _trace_summary(
        suite=suite,
        seed=seed,
        diagnostics=world.policy_decision_diagnostics_records,
        records=world.trajectory_records,
        artifact=artifact,
    )
    return {
        "suite": suite,
        "fixture": fixture_name,
        "seed": seed,
        "ticks": ticks,
        "alive_agents": _int(summary.get("alive_agents")),
        "births": _int(summary.get("births")),
        "deaths": _int(summary.get("deaths")),
        "trace": trace,
    }


def _trace_summary(
    *,
    suite: str,
    seed: int,
    diagnostics: Sequence[Mapping[str, object] | None],
    records: Sequence[Mapping[str, object]],
    artifact: Mapping[str, object],
) -> dict[str, object]:
    score_source_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    original_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    runtime_changed_pairs: Counter[str] = Counter()
    override_applied_pairs: Counter[str] = Counter()
    margin_values: list[float] = []
    utility_values: list[float] = []
    examples: list[dict[str, object]] = []
    decision_count = 0
    supported_count = 0
    clear_best_count = 0
    override_applied_count = 0
    runtime_changed_count = 0
    missing_supported_count = 0
    no_prediction_count = 0
    decisions_with_any_imputed_valid_action = 0
    selected_imputed_action_count = 0
    imputed_valid_action_score_count = 0
    observed_valid_action_score_count = 0
    table = _mapping(_mapping(artifact.get("utility_tables")).get("feature_action_utility"))
    for index, raw_diagnostic in enumerate(diagnostics):
        diagnostic = _mapping(raw_diagnostic)
        transition = _mapping(diagnostic.get("transition_value_scorer"))
        if not transition:
            continue
        record = records[index] if index < len(records) else {}
        decision_count += 1
        original = _string(transition.get("original_mind_v3_requested_action"))
        final = _string(transition.get("final_requested_action"))
        predicted = _string(transition.get("predicted_action"))
        if original:
            original_counts.update([original])
        if final:
            final_counts.update([final])
        if predicted:
            predicted_counts.update([predicted])
        source = _string(transition.get("score_source"))
        if source:
            score_source_counts.update([source])
        rejected = _string(transition.get("override_rejected_reason"))
        if rejected:
            rejected_counts.update([rejected])
        if transition.get("supported_scores_for_all_valid_actions") is True:
            supported_count += 1
        else:
            missing_supported_count += 1
        if transition.get("clear_best_valid_action") is True:
            clear_best_count += 1
        if transition.get("override_applied") is True:
            override_applied_count += 1
            if original and final:
                override_applied_pairs.update([f"{original}->{final}"])
        if transition.get("runtime_action_selection_changed") is True:
            runtime_changed_count += 1
            if original and final:
                runtime_changed_pairs.update([f"{original}->{final}"])
        if not predicted:
            no_prediction_count += 1
        margin = _finite_float(transition.get("utility_margin"))
        if margin is not None:
            margin_values.append(margin)
        selected_utility = _finite_float(transition.get("selected_utility"))
        if selected_utility is not None:
            utility_values.append(selected_utility)
        imputed_scan = _decision_imputation_scan(transition, table)
        if imputed_scan["decisions_with_any_imputed_valid_action"]:
            decisions_with_any_imputed_valid_action += 1
        if imputed_scan["selected_imputed_action"]:
            selected_imputed_action_count += 1
        imputed_valid_action_score_count += int(
            imputed_scan["imputed_valid_action_score_count"]
        )
        observed_valid_action_score_count += int(
            imputed_scan["observed_valid_action_score_count"]
        )
        if transition.get("runtime_action_selection_changed") is True and len(examples) < 16:
            examples.append(
                {
                    "suite": suite,
                    "seed": seed,
                    "record_index": index,
                    "tick": record.get("tick"),
                    "agent_id": record.get("agent_id"),
                    "original_mind_v3_requested_action": original,
                    "v180_final_requested_action": final,
                    "predicted_action": predicted,
                    "score_source": source,
                    "utility_margin": transition.get("utility_margin"),
                    "selected_utility": transition.get("selected_utility"),
                    "source_key": transition.get("source_key"),
                    "selected_imputed_action": imputed_scan[
                        "selected_imputed_action"
                    ],
                    "imputed_valid_action_score_count": imputed_scan[
                        "imputed_valid_action_score_count"
                    ],
                    "observed_valid_action_score_count": imputed_scan[
                        "observed_valid_action_score_count"
                    ],
                }
            )
    return {
        "total_decision_record_count": len(diagnostics),
        "transition_value_decision_count": decision_count,
        "supported_score_count": supported_count,
        "supported_score_share": _round(_share(supported_count, decision_count)),
        "clear_best_count": clear_best_count,
        "clear_best_share": _round(_share(clear_best_count, decision_count)),
        "override_applied_count": override_applied_count,
        "override_applied_share": _round(_share(override_applied_count, decision_count)),
        "runtime_action_selection_changed_count": runtime_changed_count,
        "runtime_action_selection_changed_share": _round(
            _share(runtime_changed_count, decision_count)
        ),
        "missing_supported_score_count": missing_supported_count,
        "missing_supported_score_share": _round(
            _share(missing_supported_count, decision_count)
        ),
        "no_prediction_count": no_prediction_count,
        "score_source_counts": dict(sorted(score_source_counts.items())),
        "override_rejected_reason_counts": dict(sorted(rejected_counts.items())),
        "original_mind_v3_requested_action_counts": dict(sorted(original_counts.items())),
        "v180_final_requested_action_counts": dict(sorted(final_counts.items())),
        "predicted_action_counts": dict(sorted(predicted_counts.items())),
        "override_applied_action_pair_counts": dict(
            sorted(override_applied_pairs.items())
        ),
        "runtime_changed_action_pair_counts": dict(
            sorted(runtime_changed_pairs.items())
        ),
        "utility_margin": _numeric_summary(margin_values),
        "selected_utility": _numeric_summary(utility_values),
        "decisions_with_any_imputed_valid_action_score": (
            decisions_with_any_imputed_valid_action
        ),
        "selected_imputed_action_count": selected_imputed_action_count,
        "imputed_valid_action_score_count": imputed_valid_action_score_count,
        "observed_valid_action_score_count": observed_valid_action_score_count,
        "runtime_changed_examples": examples,
    }


def _decision_imputation_scan(
    transition: Mapping[str, object],
    table: Mapping[str, object],
) -> dict[str, object]:
    score = _mapping(transition.get("score"))
    source_key = _string(score.get("source_key") or transition.get("source_key"))
    action_stats = _mapping(table.get(source_key))
    valid_actions = [
        str(action)
        for action in score.get("valid_actions", [])
        if isinstance(action, str)
    ]
    predicted = _string(transition.get("predicted_action"))
    imputed_count = 0
    observed_count = 0
    selected_imputed = False
    for action in valid_actions:
        stats = _mapping(action_stats.get(action))
        if not stats:
            continue
        component_means = _mapping(stats.get("component_means"))
        if "imputed_unobserved_action" in component_means:
            imputed_count += 1
            if action == predicted:
                selected_imputed = True
        else:
            observed_count += 1
    return {
        "decisions_with_any_imputed_valid_action": imputed_count > 0,
        "selected_imputed_action": selected_imputed,
        "imputed_valid_action_score_count": imputed_count,
        "observed_valid_action_score_count": observed_count,
    }


def _aggregate_trace_runs(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    totals = Counter()
    score_source_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    runtime_changed_pairs: Counter[str] = Counter()
    margin_summaries: list[Mapping[str, object]] = []
    utility_summaries: list[Mapping[str, object]] = []
    examples: list[dict[str, object]] = []
    for run in runs:
        trace = _mapping(run.get("trace"))
        for key in (
            "total_decision_record_count",
            "transition_value_decision_count",
            "supported_score_count",
            "clear_best_count",
            "override_applied_count",
            "runtime_action_selection_changed_count",
            "missing_supported_score_count",
            "no_prediction_count",
            "decisions_with_any_imputed_valid_action_score",
            "selected_imputed_action_count",
            "imputed_valid_action_score_count",
            "observed_valid_action_score_count",
        ):
            totals[key] += _int(trace.get(key))
        score_source_counts.update(_int_counter(trace.get("score_source_counts")))
        rejected_counts.update(
            _int_counter(trace.get("override_rejected_reason_counts"))
        )
        predicted_counts.update(_int_counter(trace.get("predicted_action_counts")))
        final_counts.update(_int_counter(trace.get("v180_final_requested_action_counts")))
        runtime_changed_pairs.update(
            _int_counter(trace.get("runtime_changed_action_pair_counts"))
        )
        margin_summaries.append(_mapping(trace.get("utility_margin")))
        utility_summaries.append(_mapping(trace.get("selected_utility")))
        for example in trace.get("runtime_changed_examples", []):
            if isinstance(example, Mapping) and len(examples) < 16:
                examples.append(dict(example))
    decision_count = totals["transition_value_decision_count"]
    return {
        "run_count": len(runs),
        "total_decision_record_count": totals["total_decision_record_count"],
        "transition_value_decision_count": decision_count,
        "supported_score_count": totals["supported_score_count"],
        "supported_score_share": _round(_share(totals["supported_score_count"], decision_count)),
        "clear_best_count": totals["clear_best_count"],
        "clear_best_share": _round(_share(totals["clear_best_count"], decision_count)),
        "override_applied_count": totals["override_applied_count"],
        "override_applied_share": _round(
            _share(totals["override_applied_count"], decision_count)
        ),
        "runtime_action_selection_changed_count": totals[
            "runtime_action_selection_changed_count"
        ],
        "runtime_action_selection_changed_share": _round(
            _share(totals["runtime_action_selection_changed_count"], decision_count)
        ),
        "missing_supported_score_count": totals["missing_supported_score_count"],
        "missing_supported_score_share": _round(
            _share(totals["missing_supported_score_count"], decision_count)
        ),
        "no_prediction_count": totals["no_prediction_count"],
        "score_source_counts": dict(sorted(score_source_counts.items())),
        "override_rejected_reason_counts": dict(sorted(rejected_counts.items())),
        "predicted_action_counts": dict(sorted(predicted_counts.items())),
        "v180_final_requested_action_counts": dict(sorted(final_counts.items())),
        "runtime_changed_action_pair_counts": dict(
            sorted(runtime_changed_pairs.items())
        ),
        "utility_margin": _aggregate_numeric_summaries(margin_summaries),
        "selected_utility": _aggregate_numeric_summaries(utility_summaries),
        "decisions_with_any_imputed_valid_action_score": totals[
            "decisions_with_any_imputed_valid_action_score"
        ],
        "selected_imputed_action_count": totals["selected_imputed_action_count"],
        "imputed_valid_action_score_count": totals[
            "imputed_valid_action_score_count"
        ],
        "observed_valid_action_score_count": totals[
            "observed_valid_action_score_count"
        ],
        "runtime_changed_examples": examples,
    }


def _failure_mechanism(
    *,
    source_validation: Mapping[str, object],
    artifact_support: Mapping[str, object],
    evaluation_autopsy: Mapping[str, object],
    trace_replay: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True:
        return {
            "policy": "m3_carrion_survivor_continuation_v181_failure_mechanism_v1",
            "mechanism_clear": False,
            "diagnostics_only": True,
            "labels": ["source_validation_failed"],
            "primary": "source_validation_failed",
            "slice_2_training_consumed": False,
            "next_route": "fix_v180_source_pins_before_any_v181_response",
        }
    labels: list[str] = []
    broad_eval_diag = _mapping(
        _mapping(evaluation_autopsy.get("broad")).get(
            "candidate_transition_value_scorer_diagnostics"
        )
    )
    carrion_eval_diag = _mapping(
        _mapping(evaluation_autopsy.get("controlled_fixture")).get(
            "candidate_transition_value_scorer_diagnostics"
        )
    )
    broad_trace = _mapping(_mapping(trace_replay.get("broad")).get("aggregate"))
    carrion_trace = _mapping(
        _mapping(trace_replay.get("controlled_fixture")).get("aggregate")
    )
    if _int(artifact_support.get("imputed_action_stat_count")) > _int(
        artifact_support.get("observed_action_stat_count")
    ):
        labels.append("artifact_utility_table_majority_imputed_actions")
    if _int(broad_eval_diag.get("override_applied_count")) > 0:
        labels.append("broad_regression_seeds_had_transition_value_overrides")
    predicted = _int_counter(
        broad_trace.get("predicted_action_counts")
        or broad_eval_diag.get("predicted_action_counts")
    )
    if predicted.get("eat", 0) > sum(predicted.values()) * 0.80:
        labels.append("broad_overrides_concentrated_on_eat")
    if _float(broad_eval_diag.get("missing_supported_score_share")) >= 0.80:
        labels.append("broad_runtime_support_coverage_low")
    if _int(carrion_eval_diag.get("override_applied_count")) == 0:
        labels.append("carrion_fixture_no_transition_value_overrides")
    if _float(carrion_eval_diag.get("missing_supported_score_share")) >= 0.99:
        labels.append("carrion_fixture_no_complete_valid_action_support")
    if _int(broad_trace.get("decisions_with_any_imputed_valid_action_score")) > 0:
        labels.append("complete_valid_action_support_depended_on_imputed_utilities")
    if _int(broad_trace.get("runtime_action_selection_changed_count")) > 0:
        labels.append("v180_changed_original_mind_v3_actions_on_regression_seeds")
    regressions = _int(
        _mapping(evaluation_autopsy.get("broad")).get(
            "per_seed_alive_or_birth_regression_count"
        )
    )
    if regressions > 0:
        labels.append("broad_per_seed_alive_or_birth_regressions_reproduced")
    primary = (
        "imputed_sparse_eat_override_and_carrion_no_coverage"
        if labels
        else "mechanism_unclear"
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v181_failure_mechanism_v1",
        "mechanism_clear": bool(labels),
        "diagnostics_only": True,
        "labels": labels,
        "primary": primary,
        "identified_contributors": {
            "unsupported_or_low_support_feature_keys": (
                "broad_runtime_support_coverage_low" in labels
                or "carrion_fixture_no_complete_valid_action_support" in labels
            ),
            "imputed_action_utilities": (
                "artifact_utility_table_majority_imputed_actions" in labels
                or "complete_valid_action_support_depended_on_imputed_utilities"
                in labels
            ),
            "unsafe_eat_preference": "broad_overrides_concentrated_on_eat" in labels,
            "missing_imputed_utility_abstention": (
                "complete_valid_action_support_depended_on_imputed_utilities"
                in labels
            ),
        },
        "slice_2_training_consumed": False,
        "next_route": (
            "v182_failure_response_design_requires_no_imputed_valid_action_"
            "override_or_more_exact_transition_support_before_training"
        ),
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    mechanism: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True:
        return V181_SOURCE_INVALID_CLASSIFICATION
    if mechanism.get("mechanism_clear") is True:
        return V181_AUTOPSY_CLASSIFICATION
    return V181_MECHANISM_UNCLEAR_CLASSIFICATION


def _contract() -> dict[str, object]:
    return {
        "failure_response_to": "v180_transition_row_policy_training_first_slice",
        "diagnostics_only": True,
        "training_ran": False,
        "training_artifact_created": False,
        "training_slice_consumed": False,
        "carrion_campaign_slice_count_before_v181": 1,
        "carrion_campaign_slice_count_after_v181": 1,
        "runtime_artifact_created": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "uses_fixture_identity_for_training_or_policy": False,
        "uses_private_world_state_for_training_or_policy": False,
        "heuristic_action_selection_added": False,
    }


def _lifecycle_flags(*, trace_replay_ran: bool) -> dict[str, object]:
    return {
        "training_ran": False,
        "training_artifact_created": False,
        "fit_ran": False,
        "diagnostic_trace_replay_ran": bool(trace_replay_ran),
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": False,
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_ran": False,
        "replay_viewer_schema_changed": False,
        "non_promoted": True,
    }


def _skipped_trace_replay(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v181_diagnostic_trace_replay_v1",
        "ran": False,
        "reason": reason,
    }


def _digest_without_exact(report: Mapping[str, object]) -> str:
    payload = json.loads(json.dumps(report, sort_keys=True))
    if isinstance(payload, dict):
        payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def _numeric_summary(values: Sequence[float]) -> dict[str, object]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(finite),
        "min": _round(min(finite)),
        "max": _round(max(finite)),
        "mean": _round(sum(finite) / float(len(finite))),
    }


def _aggregate_numeric_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    count = 0
    weighted_sum = 0.0
    minimum: float | None = None
    maximum: float | None = None
    for summary in summaries:
        item_count = _int(summary.get("count"))
        mean = _finite_float(summary.get("mean"))
        item_min = _finite_float(summary.get("min"))
        item_max = _finite_float(summary.get("max"))
        if item_count <= 0 or mean is None:
            continue
        count += item_count
        weighted_sum += mean * float(item_count)
        if item_min is not None:
            minimum = item_min if minimum is None else min(minimum, item_min)
        if item_max is not None:
            maximum = item_max if maximum is None else max(maximum, item_max)
    if count <= 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": count,
        "min": _round(minimum) if minimum is not None else None,
        "max": _round(maximum) if maximum is not None else None,
        "mean": _round(weighted_sum / float(count)),
    }


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _share(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_counter(value: object) -> Counter[str]:
    if not isinstance(value, Mapping):
        return Counter()
    return Counter({str(key): _int(count) for key, count in value.items()})
