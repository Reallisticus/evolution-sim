from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind import evaluation_harness as eval_harness
from evolution_sim.mind.broad_regression_branch_intervention import (
    _configure_manual_summary_run,
    _normal_mind_v3_delegate,
    _target_terminal,
)
from evolution_sim.mind.candidate_campaign import _int, _mapping, write_json
from evolution_sim.mind.carrion_branch_explore import _branch_state_digest
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v163_tied_set_branch_target_expansion import (
    V163MaterializedBranchPoint,
    V163SelectedBranchPoint,
    _ForcedTiedCandidateThenMindV3Policy,
    _materialization_failure,
    _matching_tick_record,
    _record_materialization_payload,
    _record_mismatches,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    _json_round_trip_digest,
    load_jsonl_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v177_exact_branch_replay_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION,
    _compact_transition_row,
    _complete_action_mask,
    _current_transition_record,
    _next_same_agent_record,
    _optional_string,
    _previous_same_agent_public_context,
    _record_after_dead,
    _record_at_line,
    _source_record_plan_mismatches,
    _source_records,
    _transition_done,
    _transition_run_digest_payload,
    trainable_payload_leakage_scan,
    transition_dataset_metrics,
    validate_v177_transition_rows,
)
from evolution_sim.mind.carrion_survivor_continuation_v179_exact_branch_transition_row_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V179_REPORT_PATH,
    DEFAULT_TRANSITION_DATASET_OUTPUT_PATH as DEFAULT_V179_TRANSITION_DATASET_PATH,
    V179_SUPPORT_READY_CLASSIFICATION,
    transition_row_support_summary,
)
from evolution_sim.mind.carrion_survivor_continuation_v180_transition_row_policy_training import (
    DEFAULT_ARTIFACT_OUTPUT_PATH as DEFAULT_V180_ARTIFACT_PATH,
    DEFAULT_BROAD_SEEDS,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_OUTPUT_PATH as DEFAULT_V180_REPORT_PATH,
    DEFAULT_TICKS,
)
from evolution_sim.mind.carrion_survivor_continuation_v181_v180_failure_response import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V181_REPORT_PATH,
    EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    EXPECTED_V180_ARTIFACT_DIGEST,
    EXPECTED_V180_CLASSIFICATION,
    EXPECTED_V180_REPORT_EXACT_DIGEST,
    V181_AUTOPSY_CLASSIFICATION,
)
from evolution_sim.mind.carrion_survivor_continuation_v182_imputed_abstention_design import (
    DEFAULT_OBSERVED_SUPPORT_FLOOR,
    DEFAULT_OUTPUT_PATH as DEFAULT_V182_REPORT_PATH,
    EXPECTED_V181_REPORT_EXACT_DIGEST,
    M3_CARRION_SURVIVOR_CONTINUATION_V182_IMPUTED_ABSTENTION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V182_IMPUTED_ABSTENTION_SCHEMA_VERSION,
    V182_EXACT_SUPPORT_EXPANSION_CLASSIFICATION,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.transition_value_scorer import load_transition_value_scorer_artifact

M3_CARRION_SURVIVOR_CONTINUATION_V183_EXACT_TRANSITION_SUPPORT_EXPANSION_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v183_exact_transition_support_expansion_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V183_EXACT_TRANSITION_SUPPORT_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v183_exact_transition_support_expansion_v1"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v183-carrion-survivor-continuation-exact-transition-support-expansion.json"
)
DEFAULT_EXPANDED_TRANSITION_DATASET_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v183-carrion-survivor-continuation-expanded-compact-transition-rows.jsonl"
)
DEFAULT_SOURCE_TRAJECTORY_DIR = Path(
    "output/mind/v183-exact-transition-support-expansion-trajectories"
)
EXPECTED_V179_REPORT_EXACT_DIGEST = (
    "1e18703d7f0f3b5666968051f8e3865a05ce7d781046aff7e5d5a07732907171"
)
EXPECTED_V182_REPORT_EXACT_DIGEST = (
    "3ceb89524fbcddf2e9553fa06d932c1812d1c6a1f7d7f9f19c9237c34d22f51b"
)
DEFAULT_BROAD_REGRESSION_SEEDS = (19,)
DEFAULT_CARRION_BRANCHES_PER_SEED = 3
DEFAULT_BROAD_BRANCHES_PER_SEED = 6
DEFAULT_MAX_FORCED_ACTIONS_PER_BRANCH = 6

V183_SOURCE_INVALID_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v183_exact_transition_support_expansion_"
    "source_invalid_closed_no_training"
)
V183_EXPANSION_READY_FOR_V178_AUDIT_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v183_exact_transition_support_expansion_"
    "targeted_exact_support_ready_for_fresh_v178_audit_no_training"
)
V183_EXPANSION_INSUFFICIENT_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v183_exact_transition_support_expansion_"
    "targeted_exact_support_insufficient_no_training"
)


def run_carrion_survivor_continuation_v183_exact_transition_support_expansion(
    *,
    v182_report_path: str | Path = DEFAULT_V182_REPORT_PATH,
    v181_report_path: str | Path = DEFAULT_V181_REPORT_PATH,
    v180_report_path: str | Path = DEFAULT_V180_REPORT_PATH,
    v180_artifact_path: str | Path = DEFAULT_V180_ARTIFACT_PATH,
    v179_report_path: str | Path = DEFAULT_V179_REPORT_PATH,
    v179_transition_dataset_path: str | Path = DEFAULT_V179_TRANSITION_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expanded_transition_dataset_output_path: str
    | Path = DEFAULT_EXPANDED_TRANSITION_DATASET_OUTPUT_PATH,
    source_trajectory_dir: str | Path = DEFAULT_SOURCE_TRAJECTORY_DIR,
    expected_v182_report_exact_digest: str = EXPECTED_V182_REPORT_EXACT_DIGEST,
    expected_v181_report_exact_digest: str = EXPECTED_V181_REPORT_EXACT_DIGEST,
    expected_v180_report_exact_digest: str = EXPECTED_V180_REPORT_EXACT_DIGEST,
    expected_v180_artifact_digest: str = EXPECTED_V180_ARTIFACT_DIGEST,
    expected_v179_report_exact_digest: str = EXPECTED_V179_REPORT_EXACT_DIGEST,
    expected_v179_dataset_digest: str = EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    observed_support_floor: int = DEFAULT_OBSERVED_SUPPORT_FLOOR,
    carrion_fixture_seeds: Sequence[int] = DEFAULT_CARRION_FIXTURE_SEEDS,
    broad_regression_seeds: Sequence[int] = DEFAULT_BROAD_REGRESSION_SEEDS,
    ticks: int = DEFAULT_TICKS,
    carrion_branches_per_seed: int = DEFAULT_CARRION_BRANCHES_PER_SEED,
    broad_branches_per_seed: int = DEFAULT_BROAD_BRANCHES_PER_SEED,
    max_forced_actions_per_branch: int = DEFAULT_MAX_FORCED_ACTIONS_PER_BRANCH,
    generate_source_trajectories: bool = True,
    source_trajectory_specs: Sequence[Mapping[str, object]] = (),
    verify_replay: bool = True,
) -> dict[str, object]:
    v182_report = load_json_report(v182_report_path)
    v181_report = load_json_report(v181_report_path)
    v180_report = load_json_report(v180_report_path)
    v179_report = load_json_report(v179_report_path)
    artifact = load_json_report(v180_artifact_path)
    v179_rows = [dict(row) for row in load_jsonl_dataset(v179_transition_dataset_path)]
    source_validation = validate_v183_sources(
        v182_report=v182_report,
        v181_report=v181_report,
        v180_report=v180_report,
        v179_report=v179_report,
        artifact=artifact,
        v179_rows=v179_rows,
        expected_v182_report_exact_digest=expected_v182_report_exact_digest,
        expected_v181_report_exact_digest=expected_v181_report_exact_digest,
        expected_v180_report_exact_digest=expected_v180_report_exact_digest,
        expected_v180_artifact_digest=expected_v180_artifact_digest,
        expected_v179_report_exact_digest=expected_v179_report_exact_digest,
        expected_v179_dataset_digest=expected_v179_dataset_digest,
    )
    source_generation = _empty_source_generation(
        "source_validation_failed"
        if source_validation.get("passed") is not True
        else "source_generation_disabled"
    )
    specs: list[dict[str, object]] = []
    if source_validation.get("passed") is True:
        if generate_source_trajectories:
            specs, source_generation = generate_v183_source_trajectories(
                artifact=artifact,
                output_dir=source_trajectory_dir,
                observed_support_floor=observed_support_floor,
                carrion_fixture_seeds=carrion_fixture_seeds,
                broad_regression_seeds=broad_regression_seeds,
                ticks=ticks,
            )
        else:
            specs = [dict(spec) for spec in source_trajectory_specs]
            source_generation = _source_generation_from_specs(specs)
    plan_rows, plan_generation = build_v183_plan_rows(
        specs,
        carrion_branches_per_seed=carrion_branches_per_seed,
        broad_branches_per_seed=broad_branches_per_seed,
        max_forced_actions_per_branch=max_forced_actions_per_branch,
        ticks=ticks,
    )
    selected: list[V163SelectedBranchPoint] = []
    context_by_branch_id: dict[str, dict[str, object]] = {}
    selection = _empty_selection("source_validation_or_plan_failed")
    materialized: list[V163MaterializedBranchPoint] = []
    materialization = _empty_materialization("source_validation_or_plan_failed")
    transition_rows: list[dict[str, object]] = []
    if (
        source_validation.get("passed") is True
        and plan_generation.get("selected_plan_row_count", 0)
    ):
        selected, context_by_branch_id, selection = build_selected_branch_points_v183(
            plan_rows,
            ticks=ticks,
        )
        if selection.get("passed") is True:
            scorer = load_transition_value_scorer_artifact(artifact)
            materialized, materialization = materialize_selected_branch_points_v183(
                selected,
                transition_value_scorer=scorer,
                observed_support_floor=observed_support_floor,
                ticks=ticks,
            )
            if materialization.get("passed") is True:
                transition_rows = _tag_v183_rows(
                    build_compact_transition_rows_v183(
                        materialized,
                        context_by_branch_id=context_by_branch_id,
                        verify_replay=verify_replay,
                    )
                )
    leakage_scan = trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in transition_rows]
    )
    row_schema_validation = validate_v177_transition_rows(transition_rows)
    metrics = transition_dataset_metrics(
        selected=selected,
        materialization=materialization,
        transition_rows=transition_rows,
        verify_replay=verify_replay,
    )
    support_summary = transition_row_support_summary(transition_rows)
    target_support = _target_support_delta(
        v182_report=v182_report,
        specs=specs,
        plan_rows=plan_rows,
        transition_rows=transition_rows,
    )
    classification = _classification(
        source_validation=source_validation,
        selection=selection,
        materialization=materialization,
        leakage_scan=leakage_scan,
        row_schema_validation=row_schema_validation,
        metrics=metrics,
        support_summary=support_summary,
        target_support=target_support,
    )
    dataset_digest = stable_payload_digest(transition_rows)
    _write_jsonl(expanded_transition_dataset_output_path, transition_rows)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V183_EXACT_TRANSITION_SUPPORT_EXPANSION_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V183_EXACT_TRANSITION_SUPPORT_EXPANSION_POLICY,
        "contract": _contract(observed_support_floor=observed_support_floor),
        "inputs": {
            "v182_report": str(v182_report_path),
            "v181_report": str(v181_report_path),
            "v180_report": str(v180_report_path),
            "v180_artifact": str(v180_artifact_path),
            "v179_report": str(v179_report_path),
            "v179_transition_dataset": str(v179_transition_dataset_path),
            "expected_v182_report_exact_digest": expected_v182_report_exact_digest,
            "expected_v181_report_exact_digest": expected_v181_report_exact_digest,
            "expected_v180_report_exact_digest": expected_v180_report_exact_digest,
            "expected_v180_artifact_digest": expected_v180_artifact_digest,
            "expected_v179_report_exact_digest": expected_v179_report_exact_digest,
            "expected_v179_dataset_digest": expected_v179_dataset_digest,
            "observed_support_floor": int(observed_support_floor),
            "carrion_fixture_seeds": [int(seed) for seed in carrion_fixture_seeds],
            "broad_regression_seeds": [int(seed) for seed in broad_regression_seeds],
            "ticks": int(ticks),
            "carrion_branches_per_seed": int(carrion_branches_per_seed),
            "broad_branches_per_seed": int(broad_branches_per_seed),
            "max_forced_actions_per_branch": int(max_forced_actions_per_branch),
            "source_trajectory_dir": str(source_trajectory_dir),
            "expanded_transition_dataset_output": str(
                expanded_transition_dataset_output_path
            ),
            "generate_source_trajectories": bool(generate_source_trajectories),
            "verify_replay": bool(verify_replay),
        },
        "source_validation": source_validation,
        "source_generation": source_generation,
        "plan_generation": plan_generation,
        "selection": selection,
        "branch_materialization": materialization,
        "metrics": metrics,
        "support_summary": support_summary,
        "target_support_delta": target_support,
        "leakage_scan": leakage_scan,
        "row_schema_validation": row_schema_validation,
        "dataset": {
            "path": str(expanded_transition_dataset_output_path),
            "row_count": len(transition_rows),
            "dataset_digest": dataset_digest,
            "row_schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION
            ),
            "feature_policy_id": (
                M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY
            ),
            "source_dataset_digest": stable_payload_digest(v179_rows),
            "source_dataset_path": str(v179_transition_dataset_path),
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        **_lifecycle_flags(diagnostic_dataset_created=bool(transition_rows)),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v183_sources(
    *,
    v182_report: Mapping[str, object],
    v181_report: Mapping[str, object],
    v180_report: Mapping[str, object],
    v179_report: Mapping[str, object],
    artifact: Mapping[str, object],
    v179_rows: Sequence[Mapping[str, object]],
    expected_v182_report_exact_digest: str,
    expected_v181_report_exact_digest: str,
    expected_v180_report_exact_digest: str,
    expected_v180_artifact_digest: str,
    expected_v179_report_exact_digest: str,
    expected_v179_dataset_digest: str,
) -> dict[str, object]:
    observed_v182_exact = str(v182_report.get("exact_digest") or "")
    observed_v181_exact = str(v181_report.get("exact_digest") or "")
    observed_v180_exact = str(v180_report.get("exact_digest") or "")
    observed_v179_exact = str(v179_report.get("exact_digest") or "")
    observed_artifact_digest = stable_payload_digest(artifact)
    observed_v179_dataset_digest = stable_payload_digest(
        [dict(row) for row in v179_rows]
    )
    v182_source = _mapping(v182_report.get("source_validation"))
    v182_route = _mapping(v182_report.get("route"))
    v179_dataset = _mapping(v179_report.get("dataset"))
    v180_artifact = _mapping(v180_report.get("artifact"))
    v180_dataset = _mapping(v180_report.get("dataset"))
    checks = {
        "v182_schema_version_matches": (
            v182_report.get("schema_version")
            == M3_CARRION_SURVIVOR_CONTINUATION_V182_IMPUTED_ABSTENTION_SCHEMA_VERSION
        ),
        "v182_policy_matches": (
            v182_report.get("policy")
            == M3_CARRION_SURVIVOR_CONTINUATION_V182_IMPUTED_ABSTENTION_POLICY
        ),
        "v182_exact_digest_valid": (
            exact_digest_validation_report(v182_report).get("passed") is True
        ),
        "v182_exact_digest_matches_expected": (
            observed_v182_exact == expected_v182_report_exact_digest
        ),
        "v182_classification_matches_expected": (
            _mapping(v182_report.get("classification")).get("primary")
            == V182_EXACT_SUPPORT_EXPANSION_CLASSIFICATION
        ),
        "v182_source_validation_passed": v182_source.get("passed") is True,
        "v182_routes_to_exact_support_expansion": (
            v182_route.get("next_route")
            == "exact_transition_support_expansion_before_any_slice_2_training"
        ),
        "v182_training_not_run": v182_report.get("training_ran") is False,
        "v182_slice_2_training_not_consumed": (
            v182_report.get("slice_2_training_consumed") is False
        ),
        "v182_runtime_action_selection_unchanged": (
            v182_report.get("runtime_action_selection_changed") is False
        ),
        "v181_exact_digest_valid": (
            exact_digest_validation_report(v181_report).get("passed") is True
        ),
        "v181_exact_digest_matches_expected": (
            observed_v181_exact == expected_v181_report_exact_digest
        ),
        "v181_classification_matches_expected": (
            _mapping(v181_report.get("classification")).get("primary")
            == V181_AUTOPSY_CLASSIFICATION
        ),
        "v181_training_not_run": v181_report.get("training_ran") is False,
        "v180_exact_digest_valid": (
            exact_digest_validation_report(v180_report).get("passed") is True
        ),
        "v180_exact_digest_matches_expected": (
            observed_v180_exact == expected_v180_report_exact_digest
        ),
        "v180_classification_matches_expected": (
            _mapping(v180_report.get("classification")).get("primary")
            == EXPECTED_V180_CLASSIFICATION
        ),
        "v180_training_slice_1_ran": v180_report.get("training_ran") is True,
        "v180_runtime_action_selection_unchanged": (
            v180_report.get("runtime_action_selection_changed") is False
        ),
        "v180_promotion_not_authorized": (
            v180_report.get("promotion_authorized") is False
        ),
        "v180_artifact_digest_matches_expected": (
            observed_artifact_digest == expected_v180_artifact_digest
        ),
        "v180_artifact_digest_matches_report": (
            observed_artifact_digest == str(v180_artifact.get("digest") or "")
        ),
        "v179_exact_digest_valid": (
            exact_digest_validation_report(v179_report).get("passed") is True
        ),
        "v179_exact_digest_matches_expected": (
            observed_v179_exact == expected_v179_report_exact_digest
        ),
        "v179_classification_matches_expected": (
            _mapping(v179_report.get("classification")).get("primary")
            == V179_SUPPORT_READY_CLASSIFICATION
        ),
        "v179_source_validation_passed": (
            _mapping(v179_report.get("source_validation")).get("passed") is True
        ),
        "v179_support_summary_passed": (
            _mapping(v179_report.get("support_summary")).get("passed") is True
        ),
        "v179_dataset_digest_matches_expected": (
            observed_v179_dataset_digest == expected_v179_dataset_digest
        ),
        "v179_dataset_digest_matches_report": (
            observed_v179_dataset_digest
            == str(v179_dataset.get("dataset_digest") or "")
        ),
        "v179_dataset_digest_matches_v180_report": (
            observed_v179_dataset_digest
            == str(v180_dataset.get("dataset_digest") or "")
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v183_source_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "observed_v182_report_exact_digest": observed_v182_exact,
        "expected_v182_report_exact_digest": expected_v182_report_exact_digest,
        "observed_v181_report_exact_digest": observed_v181_exact,
        "expected_v181_report_exact_digest": expected_v181_report_exact_digest,
        "observed_v180_report_exact_digest": observed_v180_exact,
        "expected_v180_report_exact_digest": expected_v180_report_exact_digest,
        "observed_v180_artifact_digest": observed_artifact_digest,
        "expected_v180_artifact_digest": expected_v180_artifact_digest,
        "observed_v179_report_exact_digest": observed_v179_exact,
        "expected_v179_report_exact_digest": expected_v179_report_exact_digest,
        "observed_v179_dataset_digest": observed_v179_dataset_digest,
        "expected_v179_dataset_digest": expected_v179_dataset_digest,
        "v179_dataset_row_count": len(v179_rows),
    }


def generate_v183_source_trajectories(
    *,
    artifact: Mapping[str, object],
    output_dir: str | Path,
    observed_support_floor: int,
    carrion_fixture_seeds: Sequence[int],
    broad_regression_seeds: Sequence[int],
    ticks: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scorer = load_transition_value_scorer_artifact(artifact)
    specs: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for seed in sorted({int(seed) for seed in broad_regression_seeds}):
        path = output / f"v183-broad-seed-{seed}-v182-candidate-{int(ticks)}.jsonl.gz"
        run = eval_harness._run_once(
            seed=seed,
            ticks=int(ticks),
            policy=eval_harness._mind_v3_policy(
                seed=seed,
                founder_template=None,
                transition_value_scorer=scorer,
                transition_value_action_override=True,
                transition_value_action_override_source_integrity_passed=True,
                transition_value_min_observed_support_count=observed_support_floor,
            ),
            trajectory_output_path=path,
            trajectory_split_id="mind_v3_v183_broad_regression_support_holes",
        )
        runs.append(run)
        specs.append(
            {
                "path": str(path),
                "suite": "broad_seed_19_regression",
                "fixture": "broad",
                "seed": seed,
                "ticks": int(ticks),
            }
        )
    for seed in sorted({int(seed) for seed in carrion_fixture_seeds}):
        path = (
            output
            / f"v183-carrion-only-seed-{seed}-v182-candidate-{int(ticks)}.jsonl.gz"
        )
        run = eval_harness._run_fixture_once(
            fixture_name="carrion_only",
            seed=seed,
            ticks=int(ticks),
            policy=eval_harness._mind_v3_policy(
                seed=seed,
                founder_template=None,
                transition_value_scorer=scorer,
                transition_value_action_override=True,
                transition_value_action_override_source_integrity_passed=True,
                transition_value_min_observed_support_count=observed_support_floor,
            ),
            trajectory_output_path=path,
            trajectory_split_id="mind_v3_v183_carrion_observed_support_zero",
        )
        runs.append(run)
        specs.append(
            {
                "path": str(path),
                "suite": "carrion_observed_support_zero",
                "fixture": "carrion_only",
                "seed": seed,
                "ticks": int(ticks),
            }
        )
    return specs, _source_generation_from_specs(specs, runs=runs)


def build_v183_plan_rows(
    source_trajectory_specs: Sequence[Mapping[str, object]],
    *,
    carrion_branches_per_seed: int,
    broad_branches_per_seed: int,
    max_forced_actions_per_branch: int,
    ticks: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidates: list[dict[str, object]] = []
    scan_counts = Counter()
    failures: list[dict[str, object]] = []
    for spec in source_trajectory_specs:
        path = Path(str(spec.get("path") or ""))
        fixture = str(spec.get("fixture") or "broad")
        suite = str(spec.get("suite") or "")
        seed = _int(spec.get("seed"), default=-1)
        records = _source_records(path)
        for line_number, record in records:
            scan_counts.update([suite])
            diagnostics = _mapping(record.get("policy_decision_diagnostics"))
            if not _is_target_support_hole(
                suite=suite,
                diagnostics=diagnostics,
            ):
                continue
            actions = _valid_forced_actions(
                record,
                max_actions=max_forced_actions_per_branch,
            )
            if len(actions) < int(max_forced_actions_per_branch):
                continue
            branch_tick = _int(record.get("tick"), default=-1)
            agent_id = _int(record.get("agent_id"), default=-1)
            if seed < 0 or branch_tick < 0 or agent_id < 0:
                failures.append(
                    {
                        "reason": "invalid_source_record_identity",
                        "path": str(path),
                        "line_number": int(line_number),
                    }
                )
                continue
            candidates.append(
                {
                    "schema_version": (
                        "m3_carrion_survivor_continuation_v183_exact_transition_support_plan_row_v1"
                    ),
                    "route": "exact_transition_support_expansion",
                    "priority": 0,
                    "suite": suite,
                    "fixture": fixture,
                    "seed": seed,
                    "branch_tick": branch_tick,
                    "agent_id": agent_id,
                    "source_path": str(path),
                    "line_number": int(line_number),
                    "row_index": len(candidates),
                    "failed_safe_action": str(record.get("requested_action") or ""),
                    "failure_types": _target_failure_types(suite, diagnostics),
                    "candidate_forced_actions": actions,
                    "source_record_digest": stable_payload_digest(
                        _record_materialization_payload(record)
                    ),
                    "support_hole_diagnostics": _support_hole_diagnostics(diagnostics),
                    "replay_expansion_goal": (
                        "expand exact current/forced/next public transition rows for "
                        "v182 observed-support-zero and broad seed 19 support holes"
                    ),
                    "training_authorized": False,
                    "runtime_artifact_authorized": False,
                }
            )
    selected = _select_v183_candidates(
        candidates,
        carrion_branches_per_seed=carrion_branches_per_seed,
        broad_branches_per_seed=broad_branches_per_seed,
    )
    for priority, row in enumerate(selected, start=1):
        row["priority"] = priority
    selected_counts = Counter(str(row.get("suite") or "") for row in selected)
    selected_by_seed = Counter(_int(row.get("seed"), default=-1) for row in selected)
    forced_actions = sorted(
        {action for row in selected for action in _ordered_actions(row.get("candidate_forced_actions"))},
        key=_action_order,
    )
    return (
        selected,
        {
            "policy": "m3_carrion_survivor_continuation_v183_plan_generation_v1",
            "passed": bool(selected) and not failures,
            "failure_count": len(failures),
            "failures": failures[:64],
            "source_trajectory_count": len(source_trajectory_specs),
            "candidate_count": len(candidates),
            "selected_plan_row_count": len(selected),
            "selected_by_suite": dict(sorted(selected_counts.items())),
            "selected_by_seed": {
                str(seed): int(count)
                for seed, count in sorted(selected_by_seed.items())
                if seed >= 0
            },
            "candidate_scan_counts_by_suite": dict(sorted(scan_counts.items())),
            "candidate_forced_action_count": len(forced_actions),
            "candidate_forced_actions": forced_actions,
            "max_forced_actions_per_branch": int(max_forced_actions_per_branch),
            "ticks": int(ticks),
            "plan_digest": stable_payload_digest(selected),
            "training_authorized": False,
            "runtime_artifact_authorized": False,
        },
    )


def build_selected_branch_points_v183(
    plan_rows: Sequence[Mapping[str, object]],
    *,
    ticks: int,
) -> tuple[list[V163SelectedBranchPoint], dict[str, dict[str, object]], dict[str, object]]:
    selected: list[V163SelectedBranchPoint] = []
    context_by_branch_id: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    by_suite: Counter[str] = Counter()
    for row_index, row in enumerate(plan_rows):
        path = Path(str(row.get("source_path") or ""))
        records = _source_records(path)
        line_number = _int(row.get("line_number"), default=-1)
        record = _record_at_line(records, line_number=line_number)
        if record is None:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "source_record_not_found",
                    "path": str(path),
                    "line_number": line_number,
                }
            )
            continue
        mismatches = _source_record_plan_mismatches(row=row, record=record)
        if mismatches:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "source_record_plan_mismatch",
                    "mismatches": mismatches,
                }
            )
            continue
        actions = _ordered_actions(row.get("candidate_forced_actions"))
        action_mask = _complete_action_mask(
            _mapping(record.get("public_action_mask") or record.get("action_mask"))
        )
        unsupported = [action for action in actions if action_mask.get(action) is not True]
        if unsupported:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "candidate_forced_action_not_public_mask_supported",
                    "actions": unsupported,
                }
            )
            continue
        suite = str(row.get("suite") or "")
        fixture = str(row.get("fixture") or "broad")
        branch_id = (
            f"v183-{suite}-seed-{_int(row.get('seed'), default=-1)}-"
            f"branch-{len(selected)}-tick-{_int(row.get('branch_tick'), default=-1)}-"
            f"agent-{_int(row.get('agent_id'), default=-1)}"
        )
        point = V163SelectedBranchPoint(
            branch_id=branch_id,
            seed=_int(row.get("seed"), default=-1),
            ticks=int(ticks),
            branch_tick=_int(row.get("branch_tick"), default=-1),
            record_index=int(row_index),
            branch_index=len(selected),
            agent_id=_int(row.get("agent_id"), default=-1),
            source_path=str(path),
            line_number=line_number,
            runtime_requested_action=str(record.get("requested_action") or ""),
            runtime_resolved_action=str(record.get("resolved_action") or ""),
            predicted_action=str(
                _mapping(row.get("support_hole_diagnostics")).get(
                    "transition_value_predicted_action"
                )
                or ""
            ),
            nearest_neighbor_row_index=-1,
            top_value_candidate_set=tuple(actions),
            action_mask=action_mask,
            observation_input=deepcopy(dict(_mapping(record.get("observation_input")))),
            observation_schema=_optional_string(record.get("observation_schema")),
            observation_digest=_optional_string(record.get("observation_digest")),
            source_record_digest=stable_payload_digest(
                _record_materialization_payload(record)
            ),
            selection_rationale={
                "v183_exact_transition_support_expansion": True,
                "suite": suite,
                "fixture": fixture,
                "failure_types": list(row.get("failure_types", [])),
                "support_hole_diagnostics": dict(
                    _mapping(row.get("support_hole_diagnostics"))
                ),
                "candidate_forced_actions_from_current_public_mask": actions,
                "source_identity_used_for_exact_materialization_only": True,
                "source_identity_used_as_trainable_input": False,
                "runtime_requested_or_resolved_action_used_as_trainable_input": False,
                "future_outcome_used_as_trainable_input": False,
            },
        )
        selected.append(point)
        context_by_branch_id[branch_id] = {
            "plan_row": dict(row),
            "source_record": deepcopy(record),
            "fixture": fixture,
            "previous_same_agent_public_context": _previous_same_agent_public_context(
                records,
                line_number=line_number,
                agent_id=point.agent_id,
            ),
        }
        by_suite.update([suite])
    return (
        selected,
        context_by_branch_id,
        {
            "policy": "m3_carrion_survivor_continuation_v183_selection_v1",
            "passed": not failures and len(selected) == len(plan_rows) and bool(selected),
            "failure_count": len(failures),
            "failures": failures[:96],
            "plan_row_count": len(plan_rows),
            "selected_branch_point_count": len(selected),
            "selected_by_suite": dict(sorted(by_suite.items())),
            "private_world_state_serialized": False,
        },
    )


def materialize_selected_branch_points_v183(
    selected: Sequence[V163SelectedBranchPoint],
    *,
    transition_value_scorer: object,
    observed_support_floor: int,
    ticks: int,
) -> tuple[list[V163MaterializedBranchPoint], dict[str, object]]:
    by_fixture_seed: defaultdict[tuple[str, int], list[V163SelectedBranchPoint]]
    by_fixture_seed = defaultdict(list)
    for point in selected:
        fixture = str(point.selection_rationale.get("fixture") or "broad")
        by_fixture_seed[(fixture, int(point.seed))].append(point)
    materialized: list[V163MaterializedBranchPoint] = []
    failures: list[dict[str, object]] = []
    reference_runs: list[dict[str, object]] = []
    for (fixture, seed), points in sorted(by_fixture_seed.items()):
        world = _v183_source_world(
            fixture=fixture,
            seed=seed,
            ticks=ticks,
            transition_value_scorer=transition_value_scorer,
            observed_support_floor=observed_support_floor,
        )
        _configure_manual_summary_run(world)
        points_by_tick: defaultdict[int, list[V163SelectedBranchPoint]]
        points_by_tick = defaultdict(list)
        for point in points:
            points_by_tick[int(point.branch_tick)].append(point)
        pending_ids = {point.branch_id for point in points}
        for tick in range(int(ticks)):
            world.tick = tick
            snapshot = deepcopy(world) if tick in points_by_tick else None
            world._run_tick()
            if tick in points_by_tick and snapshot is not None:
                for point in points_by_tick[tick]:
                    match = _matching_tick_record(world.tick_trajectory_records, point=point)
                    if match is None:
                        failures.append(
                            _materialization_failure(
                                point,
                                reason="selected_record_not_materialized",
                            )
                        )
                        continue
                    mismatches = _record_mismatches(point, match)
                    if mismatches:
                        failures.append(
                            _materialization_failure(
                                point,
                                reason="selected_record_mismatch",
                                mismatches=mismatches,
                                materialized_record_digest=stable_payload_digest(
                                    _record_materialization_payload(match)
                                ),
                            )
                        )
                        continue
                    branch_state = deepcopy(snapshot)
                    materialized.append(
                        V163MaterializedBranchPoint(
                            selected=point,
                            branch_state_digest=_branch_state_digest(
                                branch_state,
                                branch_id=point.branch_id,
                                branch_tick=point.branch_tick,
                            ),
                            world=branch_state,
                            materialized_record_digest=stable_payload_digest(
                                _record_materialization_payload(match)
                            ),
                        )
                    )
                    pending_ids.discard(point.branch_id)
            if not world.alive_agents():
                break
        for branch_id in sorted(pending_ids):
            point = next(item for item in points if item.branch_id == branch_id)
            failures.append(
                _materialization_failure(
                    point,
                    reason="selected_tick_not_reached_before_terminal_run",
                )
            )
        reference_runs.append(
            _reference_run_from_world_v183(
                world,
                fixture=fixture,
                seed=seed,
                ticks=ticks,
            )
        )
    return (
        materialized,
        {
            "policy": "m3_carrion_survivor_continuation_v183_exact_branch_materialization_v1",
            "selected_branch_point_count": len(selected),
            "materialized_branch_point_count": len(materialized),
            "materialization_failure_count": len(failures),
            "materialization_failures": failures[:24],
            "reference_runs": reference_runs,
            "exact_materialization_proven": (
                not failures and len(materialized) == len(selected) and bool(selected)
            ),
            "passed": not failures and len(materialized) == len(selected) and bool(selected),
        },
    )


def build_compact_transition_rows_v183(
    materialized: Sequence[V163MaterializedBranchPoint],
    *,
    context_by_branch_id: Mapping[str, Mapping[str, object]],
    verify_replay: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in materialized:
        context = _mapping(context_by_branch_id.get(point.selected.branch_id))
        for action in point.selected.top_value_candidate_set:
            if point.selected.action_mask.get(action) is not True:
                continue
            run, digest = _execute_candidate_transition_once_v183(
                point,
                forced_action=action,
                context=context,
            )
            verification = {
                "verified": None,
                "expected_digest": digest,
                "actual_digest": None,
            }
            if verify_replay:
                replay, replay_digest = _execute_candidate_transition_once_v183(
                    point,
                    forced_action=action,
                    context=context,
                )
                verification = {
                    "verified": replay_digest == digest,
                    "expected_digest": digest,
                    "actual_digest": replay_digest,
                }
                run["deterministic_replay_match_sample"] = {
                    "forced_action_used": replay.get("forced_action_used"),
                    "transition_done": replay.get("transition_done"),
                    "next_public_observation_available": replay.get(
                        "next_public_observation_available"
                    ),
                }
            rows.append(_compact_transition_row(run, verification=verification))
    return rows


def _execute_candidate_transition_once_v183(
    point: V163MaterializedBranchPoint,
    *,
    forced_action: str,
    context: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    world = deepcopy(point.world)
    delegate = _normal_mind_v3_delegate(world.policy)
    world.policy = _ForcedTiedCandidateThenMindV3Policy(
        target_agent_id=point.selected.agent_id,
        forced_action=forced_action,
        delegate=delegate,
    )
    _configure_manual_summary_run(world)
    current_index: int | None = None
    current_record: Mapping[str, object] | None = None
    next_record: Mapping[str, object] | None = None
    for tick in range(point.selected.branch_tick, point.selected.ticks):
        world.tick = tick
        world._run_tick()
        if current_record is None:
            current_index, current_record = _current_transition_record(
                world.trajectory_records,
                selected=point.selected,
            )
        if current_index is not None and current_record is not None:
            next_record = _next_same_agent_record(
                world.trajectory_records,
                start_index=current_index,
                agent_id=point.selected.agent_id,
            )
            if next_record is not None or _record_after_dead(current_record):
                break
        if not world.alive_agents():
            break
    summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)
    run = {
        "branch_id": point.selected.branch_id,
        "seed": point.selected.seed,
        "fixture": str(point.selected.selection_rationale.get("fixture") or "broad"),
        "ticks": point.selected.ticks,
        "branch_tick": point.selected.branch_tick,
        "agent_id": point.selected.agent_id,
        "source_path": point.selected.source_path,
        "line_number": point.selected.line_number,
        "source_row_index": _mapping(context.get("plan_row")).get("row_index"),
        "failed_safe_action": _mapping(context.get("plan_row")).get(
            "failed_safe_action"
        ),
        "failure_types": _mapping(context.get("plan_row")).get("failure_types"),
        "forced_action": forced_action,
        "forced_action_used": bool(getattr(world.policy, "used", False)),
        "branch_state_digest": point.branch_state_digest,
        "source_record_digest": point.selected.source_record_digest,
        "materialized_record_digest": point.materialized_record_digest,
        "current_record": deepcopy(current_record) if current_record else None,
        "next_record": deepcopy(next_record) if next_record else None,
        "previous_same_agent_public_context": deepcopy(
            _mapping(context.get("previous_same_agent_public_context"))
        ),
        "transition_done": _transition_done(current_record, next_record),
        "next_public_observation_available": (
            isinstance(next_record, Mapping)
            and isinstance(next_record.get("observation_input"), Mapping)
        ),
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "target_terminal": _target_terminal(world, point.selected.agent_id),
        "trajectory_record_count": len(world.trajectory_records),
        "diagnostics_only": True,
    }
    return run, stable_payload_digest(_transition_run_digest_payload(run))


def _v183_source_world(
    *,
    fixture: str,
    seed: int,
    ticks: int,
    transition_value_scorer: object,
    observed_support_floor: int,
) -> SimulationWorld:
    policy = eval_harness._mind_v3_policy(
        seed=int(seed),
        founder_template=None,
        transition_value_scorer=transition_value_scorer,
        transition_value_action_override=True,
        transition_value_action_override_source_integrity_passed=True,
        transition_value_min_observed_support_count=observed_support_floor,
    )
    if fixture == "carrion_only":
        return eval_harness._fixture_world(
            fixture_name="carrion_only",
            seed=int(seed),
            ticks=int(ticks),
            policy=policy,
        )
    return SimulationWorld(
        WorldConfig(seed=int(seed), max_ticks=int(ticks)),
        policy=policy,
    )


def _target_support_delta(
    *,
    v182_report: Mapping[str, object],
    specs: Sequence[Mapping[str, object]],
    plan_rows: Sequence[Mapping[str, object]],
    transition_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    design = _mapping(v182_report.get("design_diagnostics"))
    broad = _mapping(design.get("broad"))
    carrion = _mapping(design.get("controlled_fixture"))
    broad_seed_19 = _broad_seed_delta(v182_report, seed=19)
    plan_counts = Counter(str(row.get("suite") or "") for row in plan_rows)
    row_branch_ids_by_suite: defaultdict[str, set[str]] = defaultdict(set)
    rows_by_suite = Counter()
    actions_by_suite: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in transition_rows:
        metadata = _mapping(row.get("metadata"))
        failure_types = metadata.get("failure_types")
        suite = _suite_from_failure_types(failure_types)
        row_branch_ids_by_suite[suite].add(str(metadata.get("branch_id") or ""))
        rows_by_suite.update([suite])
        actions_by_suite[suite].update([str(row.get("forced_action") or "")])
    return {
        "policy": "m3_carrion_survivor_continuation_v183_target_support_delta_v1",
        "v182_baseline": {
            "carrion_observed_support_floor_satisfied_count": _int(
                _mapping(carrion.get("transition_value_scorer_diagnostics")).get(
                    "observed_support_floor_satisfied_count"
                )
            ),
            "carrion_decision_count": _int(
                _mapping(carrion.get("transition_value_scorer_diagnostics")).get(
                    "decision_count"
                )
            ),
            "broad_seed_19_observed_support_floor_satisfied_count": _int(
                _mapping(
                    broad_seed_19.get("transition_value_scorer_diagnostics")
                ).get("observed_support_floor_satisfied_count")
            ),
            "broad_seed_19_decision_count": _int(
                _mapping(
                    broad_seed_19.get("transition_value_scorer_diagnostics")
                ).get("decision_count")
            ),
            "broad_seed_19_alive_delta": broad_seed_19.get("alive_agents_delta"),
            "broad_seed_19_births_delta": broad_seed_19.get("births_delta"),
            "broad_override_applied_count": _int(
                _mapping(broad.get("transition_value_scorer_diagnostics")).get(
                    "override_applied_count"
                )
            ),
        },
        "source_trajectory_count": len(specs),
        "selected_target_state_count_by_suite": dict(sorted(plan_counts.items())),
        "materialized_target_state_count_by_suite": {
            suite: len(branch_ids)
            for suite, branch_ids in sorted(row_branch_ids_by_suite.items())
        },
        "expanded_transition_row_count_by_suite": dict(sorted(rows_by_suite.items())),
        "expanded_forced_action_counts_by_suite": {
            suite: dict(sorted(counter.items(), key=lambda item: _action_order(item[0])))
            for suite, counter in sorted(actions_by_suite.items())
        },
        "carrion_observed_support_coverage_delta": {
            "before": 0,
            "after_materialized_target_states": len(
                row_branch_ids_by_suite.get("carrion_observed_support_zero", set())
            ),
        },
        "broad_seed_19_support_hole_coverage_delta": {
            "before_observed_support_floor_satisfied_count": _int(
                _mapping(
                    broad_seed_19.get("transition_value_scorer_diagnostics")
                ).get("observed_support_floor_satisfied_count")
            ),
            "after_materialized_target_states": len(
                row_branch_ids_by_suite.get("broad_seed_19_regression", set())
            ),
        },
        "uses_v182_observed_imputed_support_fields": True,
        "legacy_supported_prediction_treated_as_strict_observed_support": False,
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    selection: Mapping[str, object],
    materialization: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    metrics: Mapping[str, object],
    support_summary: Mapping[str, object],
    target_support: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True:
        return V183_SOURCE_INVALID_CLASSIFICATION
    target_counts = _mapping(target_support.get("materialized_target_state_count_by_suite"))
    ready = (
        selection.get("passed") is True
        and materialization.get("passed") is True
        and metrics.get("all_replays_verified") is True
        and metrics.get("all_forced_actions_used") is True
        and leakage_scan.get("passed") is True
        and row_schema_validation.get("passed") is True
        and support_summary.get("passed") is True
        and _int(target_counts.get("carrion_observed_support_zero")) > 0
        and _int(target_counts.get("broad_seed_19_regression")) > 0
    )
    if ready:
        return V183_EXPANSION_READY_FOR_V178_AUDIT_CLASSIFICATION
    return V183_EXPANSION_INSUFFICIENT_CLASSIFICATION


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification == V183_EXPANSION_READY_FOR_V178_AUDIT_CLASSIFICATION
    return {
        "policy": "m3_carrion_survivor_continuation_v183_route_recommendation_v1",
        "recommended_next_route": (
            "fresh_v178_style_transition_row_dataset_audit_before_any_slice_2_training"
            if ready
            else "repair_v183_exact_transition_support_expansion_before_any_slice_2_training"
        ),
        "v178_style_audit_recommended": ready,
        "slice_2_training_authorized": False,
        "transition_row_training_authorized": False,
        "runtime_integration_authorized": False,
        "promotion_authorized": False,
    }


def _select_v183_candidates(
    candidates: Sequence[Mapping[str, object]],
    *,
    carrion_branches_per_seed: int,
    broad_branches_per_seed: int,
) -> list[dict[str, object]]:
    carrion_by_seed: defaultdict[int, list[Mapping[str, object]]] = defaultdict(list)
    broad_by_seed: defaultdict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in candidates:
        suite = str(row.get("suite") or "")
        seed = _int(row.get("seed"), default=-1)
        if suite == "carrion_observed_support_zero":
            carrion_by_seed[seed].append(row)
        elif suite == "broad_seed_19_regression":
            broad_by_seed[seed].append(row)
    selected: list[dict[str, object]] = []
    for seed in sorted(carrion_by_seed):
        rows = sorted(carrion_by_seed[seed], key=_candidate_sort_key)
        selected.extend(dict(row) for row in rows[: max(0, int(carrion_branches_per_seed))])
    for seed in sorted(broad_by_seed):
        rows = sorted(broad_by_seed[seed], key=_broad_candidate_sort_key)
        selected.extend(dict(row) for row in rows[: max(0, int(broad_branches_per_seed))])
    return selected


def _candidate_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _int(row.get("branch_tick"), default=10**9),
        -len(_ordered_actions(row.get("candidate_forced_actions"))),
        _int(row.get("agent_id"), default=10**9),
        _int(row.get("line_number"), default=10**9),
    )


def _broad_candidate_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    diag = _mapping(row.get("support_hole_diagnostics"))
    reason = str(diag.get("transition_value_override_rejected_reason") or "")
    reason_rank = {
        "imputed_valid_action_score": 0,
        "low_observed_support_for_valid_actions": 1,
        "missing_supported_scores_for_valid_actions": 2,
    }.get(reason, 3)
    return (reason_rank, *_candidate_sort_key(row))


def _is_target_support_hole(
    *,
    suite: str,
    diagnostics: Mapping[str, object],
) -> bool:
    observed_satisfied = (
        diagnostics.get(
            "transition_value_observed_support_floor_satisfied_for_all_valid_actions"
        )
        is True
    )
    if observed_satisfied:
        return False
    source = str(diagnostics.get("transition_value_score_source") or "")
    reason = str(diagnostics.get("transition_value_override_rejected_reason") or "")
    if suite == "carrion_observed_support_zero":
        return source == "missing_supported_scores_for_valid_actions"
    if suite == "broad_seed_19_regression":
        return reason in {
            "missing_supported_scores_for_valid_actions",
            "low_observed_support_for_valid_actions",
            "imputed_valid_action_score",
        }
    return False


def _valid_forced_actions(
    record: Mapping[str, object],
    *,
    max_actions: int,
) -> list[str]:
    mask = _mapping(record.get("public_action_mask") or record.get("action_mask"))
    return [action for action in ACTION_NAMES if mask.get(action) is True][
        : max(1, int(max_actions))
    ]


def _ordered_actions(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    seen = set()
    actions = []
    for item in value:
        action = str(item)
        if action in ACTION_NAMES and action not in seen:
            seen.add(action)
            actions.append(action)
    return sorted(actions, key=_action_order)


def _tag_v183_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    tagged = []
    for row in rows:
        payload = dict(row)
        payload["row_origin"] = (
            "v183_exact_transition_support_expansion_from_v182_support_holes"
        )
        metadata = dict(_mapping(payload.get("metadata")))
        metadata["source_v183_expansion"] = True
        metadata["source_identity_used_for_exact_materialization_only"] = True
        metadata["source_identity_used_as_trainable_input"] = False
        metadata["runtime_requested_or_resolved_action_used_as_trainable_input"] = False
        metadata["current_or_future_outcome_used_as_trainable_input"] = False
        metadata["diagnostic_target_used_as_trainable_input"] = False
        payload["metadata"] = metadata
        tagged.append(payload)
    return tagged


def _support_hole_diagnostics(diagnostics: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "transition_value_score_source",
        "transition_value_override_rejected_reason",
        "transition_value_predicted_action",
        "transition_value_observed_scores_for_all_valid_actions",
        "transition_value_has_imputed_valid_action_score",
        "transition_value_imputed_valid_action_score_count",
        "transition_value_observed_valid_action_score_count",
        "transition_value_low_observed_support_valid_action_score_count",
        "transition_value_observed_support_floor_satisfied_for_all_valid_actions",
        "transition_value_valid_action_observed_support_floor",
        "transition_value_supported_scores_for_all_valid_actions",
    )
    return {key: diagnostics.get(key) for key in keys if key in diagnostics}


def _target_failure_types(
    suite: str,
    diagnostics: Mapping[str, object],
) -> list[str]:
    if suite == "carrion_observed_support_zero":
        return ["v183_carrion_observed_support_zero"]
    reason = str(diagnostics.get("transition_value_override_rejected_reason") or "")
    return ["v183_broad_seed_19_regression_state", f"v183_{reason}"]


def _suite_from_failure_types(value: object) -> str:
    failure_types = {str(item) for item in value} if isinstance(value, list) else set()
    if "v183_carrion_observed_support_zero" in failure_types:
        return "carrion_observed_support_zero"
    if "v183_broad_seed_19_regression_state" in failure_types:
        return "broad_seed_19_regression"
    return "unknown"


def _broad_seed_delta(report: Mapping[str, object], *, seed: int) -> dict[str, object]:
    shadow = _mapping(report.get("shadow_evaluation"))
    broad = _mapping(shadow.get("broad"))
    candidate = _mapping(broad.get("candidate"))
    for run in candidate.get("runs", []):
        if isinstance(run, Mapping) and _int(run.get("seed"), default=-1) == int(seed):
            result = dict(run)
            break
    else:
        result = {}
    for delta in shadow.get("per_seed_alive_birth_deltas", []):
        if (
            isinstance(delta, Mapping)
            and delta.get("suite") == "broad"
            and _int(delta.get("seed"), default=-1) == int(seed)
        ):
            result.update(dict(delta))
            break
    return result


def _reference_run_from_world_v183(
    world: SimulationWorld,
    *,
    fixture: str,
    seed: int,
    ticks: int,
) -> dict[str, object]:
    summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)
    return {
        "seed": int(seed),
        "fixture": fixture,
        "ticks": int(ticks),
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
    }


def _source_generation_from_specs(
    specs: Sequence[Mapping[str, object]],
    *,
    runs: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v183_source_generation_v1",
        "ran": bool(runs),
        "source_trajectory_count": len(specs),
        "source_trajectories": [dict(spec) for spec in specs],
        "runs": [dict(run) for run in runs],
        "training_ran": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
    }


def _empty_source_generation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v183_source_generation_v1",
        "ran": False,
        "reason": reason,
        "source_trajectory_count": 0,
        "source_trajectories": [],
        "training_ran": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
    }


def _empty_selection(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v183_selection_v1",
        "passed": False,
        "reason": reason,
        "plan_row_count": 0,
        "selected_branch_point_count": 0,
        "failure_count": 0,
        "failures": [],
    }


def _empty_materialization(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v183_exact_branch_materialization_v1",
        "selected_branch_point_count": 0,
        "materialized_branch_point_count": 0,
        "materialization_failure_count": 0,
        "materialization_failures": [],
        "exact_materialization_proven": False,
        "passed": False,
        "reason": reason,
    }


def _contract(*, observed_support_floor: int) -> dict[str, object]:
    return {
        "failure_response_to": "v182_imputed_abstention_design",
        "diagnostics_only": True,
        "training_allowed": False,
        "training_ran": False,
        "training_artifact_created": False,
        "slice_2_training_consumed": False,
        "runtime_artifact_allowed": False,
        "runtime_artifact_created": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_changed": False,
        "default_runtime_behavior_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "observed_support_floor": int(observed_support_floor),
        "targets_v182_carrion_observed_support_zero": True,
        "targets_v182_broad_seed_19_regression_states": True,
        "uses_v182_observed_imputed_support_fields": True,
        "legacy_supported_prediction_is_not_strict_observed_support": True,
        "fresh_v178_style_audit_required_before_slice_2_training": True,
    }


def _lifecycle_flags(*, diagnostic_dataset_created: bool) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_ran": False,
        "training_artifact_created": False,
        "fit_ran": False,
        "scorer_retraining_ran": False,
        "scorer_retraining_authorized": False,
        "diagnostic_dataset_created": bool(diagnostic_dataset_created),
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
        "slice_2_training_consumed": False,
        "non_promoted": True,
    }


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")

