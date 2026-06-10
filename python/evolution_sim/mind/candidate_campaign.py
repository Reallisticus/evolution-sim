from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from evolution_sim.cli import mind_v3_evaluate as evaluate_cli
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.broad_regression_branch_intervention import (
    DEFAULT_LIVE_REPORT_PATH as DEFAULT_V142_LIVE_REPORT_PATH,
    DEFAULT_SCORER_PATH as DEFAULT_V142_SCORER_REPORT_PATH,
    DEFAULT_V142_TRAJECTORY_DIR,
    BroadRegressionBranchPoint,
    _candidate_actions as _v143_candidate_actions,
    _configure_manual_summary_run as _configure_branch_manual_summary_run,
    _evaluate_branch_point as _v143_evaluate_branch_point,
    _optional_string,
    _ordered_regression_seeds,
    _target_terminal_by_agent_from_records,
)
from evolution_sim.mind.branch_intervention_residual import (
    DEFAULT_ARTIFACT_PATH as DEFAULT_V144_ARTIFACT_PATH,
    DEFAULT_REPORT_PATH as DEFAULT_V144_REPORT_PATH,
    DEFAULT_V143_DATASET_PATH,
    DEFAULT_V143_REPORT_PATH,
    STRICT_BROAD_SEEDS,
    STRICT_CARRION_FIXTURE_SEEDS,
    STRICT_TICKS,
    load_json_report,
    load_v143_branch_intervention_dataset,
    write_json,
)
from evolution_sim.mind.carrion_branch_explore import _branch_state_digest
from evolution_sim.mind.branch_label_causal_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V145_REPORT_PATH,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.support_gated_residual import (
    MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
    MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
    V103_ACTION_PRIOR_BALANCE_PENALTY,
    aggregate_support_gated_residual_runtime_diagnostics,
    load_support_gated_residual_artifact,
    score_support_gated_residual_artifact,
    support_gated_residual_runtime_diagnostics,
    validate_support_gated_residual_artifact,
)
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

MIND_V3_V146_CANDIDATE_CAMPAIGN_REPORT_SCHEMA_VERSION = (
    "mind_v3_v146_candidate_campaign_report_v1"
)
MIND_V3_V146_CANDIDATE_CAMPAIGN_LEDGER_SCHEMA_VERSION = (
    "mind_v3_v146_candidate_campaign_ledger_v1"
)
MIND_V3_V146_CANDIDATE_CAMPAIGN_POLICY = (
    "research_infrastructure_v146_candidate_campaign_v1"
)
M3_SAFE_ARCHIVE_EXPANSION_REPORT_SCHEMA_VERSION = (
    "m3_carrion_broad_safe_archive_expansion_report_v1"
)
M3_SAFE_ARCHIVE_EXPANSION_DATASET_ROW_SCHEMA_VERSION = (
    "m3_carrion_broad_safe_archive_expansion_dataset_row_v1"
)
M3_SAFE_ARCHIVE_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_broad_safe_archive_expansion_001"
)
M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION = (
    "m3_carrion_broad_safe_archive_expansion_branch_evidence_v1"
)
M3_SAFE_ARCHIVE_TRAIN_EVAL_REPORT_SCHEMA_VERSION = (
    "m3_safe_archive_bp3_train_eval_diagnostic_report_v1"
)
M3_SAFE_ARCHIVE_TRAIN_EVAL_POLICY = (
    "diagnostics_only_m3_safe_archive_bp3_support_gated_train_eval_v1"
)
BP3_SAFE_ARCHIVE_DATASET_DIGEST = (
    "c7dffec4a32f3c8cd5b0cccb4be91a6aae6bb20d946328f00eedefc3e7fac20b"
)
BP3_SAFE_ARCHIVE_BRANCH_EVIDENCE_DIGEST = (
    "770d78f4529a0a18c5e23661f4d91758389e9e5eb4aa88d8f8e07f904af20797"
)
DEFAULT_REPORT_PATH = Path("output/mind/mind-v3-v146-candidate-campaign-report.json")
DEFAULT_LEDGER_PATH = Path("output/mind/mind-v3-v146-candidate-campaign-ledger.jsonl")
DEFAULT_OUTPUT_DIR = Path("output/mind/v146-candidate-campaign")
DEFAULT_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_PATH = Path(
    "output/mind/m3-carrion-broad-safe-archive-expansion-001-branch-evidence.json"
)
DEFAULT_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_CHUNK_DIR = Path(
    "output/mind/m3-carrion-broad-safe-archive-expansion-001-branch-evidence-chunks"
)
DEFAULT_SAFE_ARCHIVE_EXPANSION_REPORT_PATH = Path(
    "output/mind/m3-carrion-broad-safe-archive-expansion-001-report.json"
)
DEFAULT_SAFE_ARCHIVE_EXPANSION_DATASET_PATH = Path(
    "output/mind/m3-carrion-broad-safe-archive-expansion-001-dataset.jsonl"
)
DEFAULT_MIN_SAFE_LABEL_COUNT = 20
DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_BRANCH_POINTS_PER_SEED = 1
DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_CANDIDATE_ACTIONS = 0
DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_BROAD_REGRESSION_SEEDS = 1
DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_CARRION_FIXTURE_SEEDS = 1
SAFE_ARCHIVE_TRAIN_EVAL_BROAD_SEEDS = (5, 13, 19, 29, 37, 41)
SAFE_ARCHIVE_TRAIN_EVAL_CARRION_SEEDS = (13, 19, 29, 37, 41, 43)
SAFE_ARCHIVE_TRAIN_EVAL_TICKS = 120
MAX_DOMINANT_SAFE_LABEL_ACTION_SHARE = 0.50
FORBIDDEN_TRAINABLE_TOKENS = (
    "seed",
    "fixture",
    "branch",
    "path",
    "digest",
    "provenance",
    "record_index",
    "tick",
    "agent_id",
    "source",
    "trajectory",
)
CAMPAIGN_ARMS = (
    "linear_control",
    "v144_branch_intervention_residual",
    "v146_blacklist_safe_branch_residual",
    "safe_exact_support",
    "safe_action_conditioned_support",
    "safe_public_history_support",
    "neural_offline",
)
SUPPORT_GATED_ARMS = {
    "v146_blacklist_safe_branch_residual",
    "safe_exact_support",
    "safe_action_conditioned_support",
    "safe_public_history_support",
}


class CandidateCampaignError(ValueError):
    pass


def run_candidate_campaign(
    *,
    mode: str = "smoke",
    workers: int = 1,
    candidate_limit: int | None = None,
    config_path: str | Path | None = None,
    keep_trajectories: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    v143_report_path: str | Path = DEFAULT_V143_REPORT_PATH,
    v143_dataset_path: str | Path = DEFAULT_V143_DATASET_PATH,
    v144_report_path: str | Path = DEFAULT_V144_REPORT_PATH,
    v144_artifact_path: str | Path = DEFAULT_V144_ARTIFACT_PATH,
    v145_report_path: str | Path = DEFAULT_V145_REPORT_PATH,
) -> dict[str, object]:
    campaign_config = _load_campaign_config(config_path)
    min_safe_label_count = _int(
        campaign_config.get("min_safe_label_count"),
        default=DEFAULT_MIN_SAFE_LABEL_COUNT,
    )
    if min_safe_label_count <= 0:
        min_safe_label_count = DEFAULT_MIN_SAFE_LABEL_COUNT
    mode = _validate_mode(mode)
    worker_count = max(1, int(workers))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v143_report = load_json_report(v143_report_path)
    dataset_rows = load_v143_branch_intervention_dataset(v143_dataset_path)
    v144_report = _load_optional_json(v144_report_path)
    v145_report = _load_optional_json(v145_report_path)
    archive = build_safe_branch_label_archive(
        v143_report=v143_report,
        dataset_rows=dataset_rows,
        v145_report=v145_report,
        min_safe_label_count=min_safe_label_count,
    )
    baseline = build_linear_baseline_cache(
        broad_seeds=STRICT_BROAD_SEEDS,
        ticks=STRICT_TICKS,
        fixture_seeds=STRICT_CARRION_FIXTURE_SEEDS,
        fixture_ticks=STRICT_TICKS,
        keep_trajectories=keep_trajectories,
        output_dir=out_dir / "linear_control",
    )
    write_json(out_dir / "linear-baseline.json", baseline)
    specs = build_candidate_specs(
        archive=archive,
        dataset_rows=dataset_rows,
        mode=mode,
        candidate_limit=candidate_limit,
        v144_report=v144_report,
        v144_artifact_path=v144_artifact_path,
        output_dir=out_dir,
        keep_trajectories=keep_trajectories,
    )
    results = evaluate_candidate_specs(
        specs=specs,
        baseline=baseline,
        workers=worker_count,
    )
    controls = [result for result in results if _is_control_result(result)]
    candidates = [result for result in results if not _is_control_result(result)]
    leaderboard = rank_candidate_results(results)
    stop_rules = _stop_rules(results=results, archive=archive)
    report = {
        "schema_version": MIND_V3_V146_CANDIDATE_CAMPAIGN_REPORT_SCHEMA_VERSION,
        "policy": MIND_V3_V146_CANDIDATE_CAMPAIGN_POLICY,
        "contract": {
            "research_acceleration_infrastructure": True,
            "promotion_authorized": False,
            "default_runtime_behavior_changed": False,
            "runtime_policy_added": False,
            "heuristic_fallback_added": False,
            "seed_fixture_private_world_runtime_features_allowed": False,
            "gate_relaxation": False,
            "replay_viewer_golden_contract_changed": False,
            "strict_heldout_training_allowed": False,
        },
        "mode": mode,
        "workers": worker_count,
        "candidate_limit": candidate_limit,
        "config": campaign_config,
        "matrix": {
            "broad_seeds": list(STRICT_BROAD_SEEDS),
            "ticks": STRICT_TICKS,
            "fixture": "carrion_only",
            "fixture_seeds": list(STRICT_CARRION_FIXTURE_SEEDS),
            "fixture_ticks": STRICT_TICKS,
        },
        "inputs": {
            "v143_report": str(v143_report_path),
            "v143_dataset": str(v143_dataset_path),
            "v144_report": str(v144_report_path),
            "v144_artifact": str(v144_artifact_path),
            "v145_report": str(v145_report_path),
        },
        "archive": archive,
        "baseline": _baseline_summary(baseline),
        "control_count": len(controls),
        "controls": controls,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "leaderboard": leaderboard,
        "stop_rules": stop_rules,
        "classification": _campaign_classification(stop_rules),
        "output_dir": str(out_dir),
        "ledger_output": str(ledger_path),
        "non_promoted": True,
    }
    for result in results:
        write_json(out_dir / f"{result['candidate_id']}.json", result)
    write_json(report_path, report)
    write_campaign_ledger(ledger_path, report)
    return report


def build_linear_baseline_cache(
    *,
    broad_seeds: Sequence[int],
    ticks: int,
    fixture_seeds: Sequence[int],
    fixture_ticks: int,
    keep_trajectories: bool,
    output_dir: Path,
) -> dict[str, object]:
    broad_runs = [
        _run_policy(
            seed=seed,
            ticks=ticks,
            artifact=None,
            runtime_mode="live",
            fixture=None,
            trajectory_path=(
                output_dir / "trajectories" / f"broad-linear-seed-{seed}.jsonl.gz"
                if keep_trajectories
                else None
            ),
        )
        for seed in broad_seeds
    ]
    carrion_runs = [
        _run_policy(
            seed=seed,
            ticks=fixture_ticks,
            artifact=None,
            runtime_mode="live",
            fixture="carrion_only",
            trajectory_path=(
                output_dir
                / "trajectories"
                / f"carrion-linear-seed-{seed}.jsonl.gz"
                if keep_trajectories
                else None
            ),
        )
        for seed in fixture_seeds
    ]
    return {
        "policy": "v146_cached_linear_mind_v3_baseline_v1",
        "broad": {
            "seeds": [int(seed) for seed in broad_seeds],
            "ticks": int(ticks),
            "runs": broad_runs,
            "aggregate": _aggregate_runs(broad_runs),
        },
        "carrion_only": {
            "seeds": [int(seed) for seed in fixture_seeds],
            "ticks": int(fixture_ticks),
            "runs": carrion_runs,
            "aggregate": _aggregate_runs(carrion_runs),
        },
    }


def build_safe_branch_label_archive(
    *,
    v143_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    v145_report: Mapping[str, object] | None,
    min_safe_label_count: int = DEFAULT_MIN_SAFE_LABEL_COUNT,
) -> dict[str, object]:
    blacklist = _v145_blacklist(v145_report or {})
    safe_rows = []
    excluded_rows = []
    for index, row in enumerate(dataset_rows):
        identity = _row_identity(index=index, row=row)
        if _blacklisted(identity, blacklist):
            excluded_rows.append({**identity, "excluded_reason": "v145_causal_blacklist"})
            continue
        vet = _safety_vet_label(row)
        if vet["passed"] is True:
            safe_rows.append({**identity, "safety_vet": vet})
        else:
            excluded_rows.append({**identity, "excluded_reason": "safety_vet_failed", "safety_vet": vet})
    safe_action_counts = Counter(str(row.get("label_action")) for row in safe_rows)
    return {
        "policy": "v146_blacklist_safe_branch_label_archive_v1",
        "source_v143_classification": _mapping(v143_report.get("classification")).get(
            "primary"
        ),
        "source_dataset_row_count": len(dataset_rows),
        "v145_blacklist_count": len(blacklist),
        "safe_label_count": len(safe_rows),
        "min_safe_label_count": int(min_safe_label_count),
        "archive_support_sufficient": len(safe_rows) >= int(min_safe_label_count),
        "safe_action_counts": dict(sorted(safe_action_counts.items())),
        "safe_rows": safe_rows,
        "excluded_row_count": len(excluded_rows),
        "excluded_rows": excluded_rows[:64],
        "expansion_plan": {
            "requested": True,
            "multiple_branch_points": True,
            "all_valid_candidate_actions": True,
            "broad_and_carrion_contexts": True,
            "executed": len(safe_rows) >= int(min_safe_label_count),
            "blocked_reason": (
                None
                if len(safe_rows) >= int(min_safe_label_count)
                else "safe_label_count_below_minimum_before_training"
            ),
            "strict_heldout_training_used": False,
        },
    }


def build_safe_archive_expansion_report(
    *,
    branch_results: Sequence[Mapping[str, object]],
    v145_report: Mapping[str, object] | None,
    min_safe_label_count: int = DEFAULT_MIN_SAFE_LABEL_COUNT,
    strict_heldout_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    branch_evidence_status: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    blacklist = _v145_blacklist(v145_report or {})
    strict_seeds = tuple(sorted({int(seed) for seed in strict_heldout_seeds}))
    evidence_status = _mapping(branch_evidence_status)
    partial_branch_evidence = _safe_archive_expansion_status_partial(evidence_status)
    rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    source_integrity_failures: list[str] = []
    branch_point_counts: Counter[str] = Counter()
    branch_points_by_fixture_seed: Counter[str] = Counter()
    all_action_run_count = 0
    replay_verification_count = 0
    replay_verification_failures = 0
    all_valid_action_miss_count = 0
    first_missing_valid_action: dict[str, object] | None = None
    heuristic_action_source_count = 0

    for result_index, result in enumerate(branch_results):
        fixture = str(result.get("fixture", "unknown"))
        seed = _int(result.get("seed"))
        branch_point_counts[fixture] += 1
        branch_points_by_fixture_seed[f"{fixture}:{seed}"] += 1
        action_runs = _list_of_mappings(result.get("action_runs"))
        all_action_run_count += len(action_runs)
        heuristic_action_source_count += sum(
            _int(run.get("heuristic_action_source_count")) for run in action_runs
        )
        replay_items = [
            _mapping(run.get("replay_verification"))
            for run in action_runs
            if run.get("replay_verification") is not None
        ]
        replay_verification_count += len(replay_items)
        replay_verification_failures += sum(
            1 for item in replay_items if item.get("verified") is not True
        )
        if _all_valid_actions_evaluated(result) is not True:
            all_valid_action_miss_count += 1
            if first_missing_valid_action is None:
                first_missing_valid_action = _missing_valid_action_detail(
                    result_index,
                    result,
                )

        selected = _best_safe_expansion_action_run(result)
        if selected is None:
            excluded.append(
                {
                    "source_branch_result_index": result_index,
                    "seed": seed,
                    "fixture": fixture,
                    "branch_id": result.get("branch_id"),
                    "excluded_reason": "no_safety_vetted_label_action",
                }
            )
            continue
        identity = _expanded_row_identity(result_index, result, selected)
        if _blacklisted(identity, blacklist, allow_label_source_row_index=False):
            excluded.append(
                {
                    **identity,
                    "excluded_reason": "v145_causal_blacklist",
                }
            )
            continue
        vet = _safe_archive_expansion_label_vet(selected)
        if vet["passed"] is not True:
            excluded.append(
                {
                    **identity,
                    "excluded_reason": "safety_vet_failed",
                    "safety_vet": vet,
                }
            )
            continue
        rows.append(
            _safe_archive_expansion_dataset_row(
                row_index=len(rows),
                source_branch_result_index=result_index,
                result=result,
                selected_run=selected,
                safety_vet=vet,
            )
        )

    dataset_scan = safe_archive_expansion_leakage_scan(
        rows,
        strict_heldout_seeds=strict_seeds,
    )
    label_counts = Counter(
        str(_mapping(_mapping(row.get("trainable")).get("label")).get("action"))
        for row in rows
    )
    label_counts.pop("None", None)
    dominant = _dominant_count_share(label_counts)
    carrion_row_count = sum(
        1 for row in rows if _mapping(row.get("metadata")).get("fixture") == "carrion_only"
    )
    broad_row_count = sum(
        1 for row in rows if _mapping(row.get("metadata")).get("fixture") == "broad"
    )
    if replay_verification_count <= 0:
        source_integrity_failures.append("no_branch_replay_verification_records")
    if replay_verification_failures > 0:
        source_integrity_failures.append("branch_replay_verification_failed")
    if all_valid_action_miss_count > 0:
        source_integrity_failures.append("not_all_valid_candidate_actions_evaluated")
    if dataset_scan.get("passed") is not True:
        source_integrity_failures.append("safe_archive_expansion_leakage_scan_failed")
    if partial_branch_evidence:
        source_integrity_failures.append("partial_branch_evidence")
    source_integrity = {
        "policy": "m3_safe_archive_expansion_source_integrity_v1",
        "passed": not source_integrity_failures,
        "failures": sorted(set(source_integrity_failures)),
        "branch_result_count": len(branch_results),
        "branch_point_counts": dict(sorted(branch_point_counts.items())),
        "branch_points_by_fixture_seed": dict(sorted(branch_points_by_fixture_seed.items())),
        "action_run_count": all_action_run_count,
        "replay_verification_count": replay_verification_count,
        "replay_verification_failure_count": replay_verification_failures,
        "all_valid_action_miss_count": all_valid_action_miss_count,
        "first_missing_valid_action": first_missing_valid_action,
        "smallest_next_implementation_needed": (
            None
            if first_missing_valid_action is None
            else (
                "regenerate diagnostics branch evidence from a branch snapshot "
                "with one action_run for every valid public action-mask action"
            )
        ),
        "dataset_leakage_scan_passed": dataset_scan.get("passed") is True,
        "branch_evidence_status": (
            dict(evidence_status) if evidence_status else {"state": "unknown"}
        ),
    }
    diagnostics = _safe_archive_expansion_support_diagnostics(
        branch_results=branch_results,
        excluded_rows=excluded,
        source_integrity=source_integrity,
        safe_label_count=len(rows),
        min_safe_label_count=int(min_safe_label_count),
    )
    support_floors = _safe_archive_expansion_support_floors(
        source_integrity=source_integrity,
        dataset_scan=dataset_scan,
        safe_label_count=len(rows),
        min_safe_label_count=int(min_safe_label_count),
        dominant_label_share=_float(dominant.get("share")),
        heuristic_action_source_count=heuristic_action_source_count,
    )
    classification = _safe_archive_expansion_classification(
        source_integrity=source_integrity,
        support_floors=support_floors,
    )
    report = {
        "schema_version": M3_SAFE_ARCHIVE_EXPANSION_REPORT_SCHEMA_VERSION,
        "policy": M3_SAFE_ARCHIVE_EXPANSION_POLICY,
        "contract": {
            "diagnostics_only": True,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
            "heuristic_fallback_added": False,
            "gate_relaxation": False,
            "replay_viewer_golden_contract_changed": False,
            "training_run_by_default": False,
            "training_authorized": False,
            "training_eligible_only_if_support_floors_pass": True,
            "promotion_authorized": False,
            "partial_branch_evidence_cannot_authorize_training": True,
            "branch_point_policy": (
                "multiple_branch_points_per_broad_regression_and_carrion_fixture_seed"
            ),
            "candidate_action_policy": "all_currently_valid_public_action_mask_actions",
            "trainable_fields": [
                "public observation_input",
                "public action_mask",
                "label action",
            ],
            "non_trainable_metadata_only": [
                "seed",
                "fixture",
                "branch id",
                "provenance",
                "paths",
                "replay digests",
                "outcome evidence",
            ],
        },
        "blacklist": {
            "policy": "v145_label_blacklist_exclusion_v1",
            "blacklist_count": len(blacklist),
        },
        "source_integrity": source_integrity,
        "dataset": {
            "row_schema_version": M3_SAFE_ARCHIVE_EXPANSION_DATASET_ROW_SCHEMA_VERSION,
            "safe_label_count": len(rows),
            "min_safe_label_count": int(min_safe_label_count),
            "safe_label_action_counts": dict(sorted(label_counts.items())),
            "dominant_safe_label_action": dominant["key"],
            "dominant_safe_label_action_count": dominant["count"],
            "dominant_safe_label_action_share": dominant["share"],
            "broad_safe_label_count": broad_row_count,
            "carrion_safe_label_count": carrion_row_count,
            "leakage_scan": dataset_scan,
            "dataset_digest": stable_payload_digest(rows),
        },
        "coverage": {
            "broad_safe_label_count": broad_row_count,
            "carrion_safe_label_count": carrion_row_count,
            "branch_point_counts": dict(sorted(branch_point_counts.items())),
            "branch_points_by_fixture_seed": dict(sorted(branch_points_by_fixture_seed.items())),
        },
        "diagnostics": diagnostics,
        "excluded_row_count": len(excluded),
        "excluded_rows": excluded[:96],
        "support_floors": support_floors,
        "classification": {"primary": classification, "labels": [classification]},
        "archive_support_passed": support_floors.get("passed") is True,
        "training_authorized": False,
        "promotion_authorized": False,
        "non_default_runtime": True,
        "non_promoted": True,
    }
    return report, rows


def write_safe_archive_expansion_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    write_json(output_path, dict(report))


def write_safe_archive_expansion_dataset(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
            handle.write("\n")


def build_safe_archive_expansion_branch_evidence(
    *,
    v142_scorer_report_path: str | Path = DEFAULT_V142_SCORER_REPORT_PATH,
    v142_live_report_path: str | Path = DEFAULT_V142_LIVE_REPORT_PATH,
    v142_trajectory_output_dir: str | Path = DEFAULT_V142_TRAJECTORY_DIR,
    max_branch_points_per_seed: int = (
        DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_BRANCH_POINTS_PER_SEED
    ),
    max_candidate_actions: int = DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_CANDIDATE_ACTIONS,
    max_broad_regression_seeds: int = (
        DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_BROAD_REGRESSION_SEEDS
    ),
    max_carrion_fixture_seeds: int = (
        DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_CARRION_FIXTURE_SEEDS
    ),
    regenerate_v142_trajectories: bool = True,
    verify_replay: bool = True,
    carrion_seeds: Sequence[int] = STRICT_CARRION_FIXTURE_SEEDS,
    carrion_ticks: int = STRICT_TICKS,
    branch_evidence_chunk_dir: str | Path | None = None,
    resume_branch_evidence: bool = False,
    max_wall_seconds: float | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    fixtures: Sequence[str] = ("broad", "carrion_only"),
    seed_include: Sequence[int] | None = None,
    branch_index_start: int | None = None,
    branch_index_count: int | None = None,
    branch_index_include: Sequence[int] | None = None,
    shard_id: str | None = None,
) -> dict[str, object]:
    started_at = time.monotonic()
    branch_limit = max(1, int(max_branch_points_per_seed))
    candidate_limit = max(0, int(max_candidate_actions))
    chunk_dir = Path(branch_evidence_chunk_dir) if branch_evidence_chunk_dir else None
    selected_fixtures = _safe_archive_expansion_selected_fixtures(fixtures)
    selected_branch_indexes = _safe_archive_expansion_selected_branch_indexes(
        max_branch_points_per_seed=branch_limit,
        branch_index_start=branch_index_start,
        branch_index_count=branch_index_count,
        branch_index_include=branch_index_include,
    )
    seed_filter = None if seed_include is None else tuple(int(seed) for seed in seed_include)
    resumed_chunks = (
        load_safe_archive_expansion_branch_result_chunks(chunk_dir)
        if chunk_dir is not None and bool(resume_branch_evidence)
        else {}
    )
    if "broad" in selected_fixtures:
        broad = _safe_archive_expansion_broad_branch_evidence(
            v142_live_report_path=v142_live_report_path,
            ticks=STRICT_TICKS,
            max_branch_points_per_seed=branch_limit,
            max_candidate_actions=candidate_limit,
            max_regression_seeds=int(max_broad_regression_seeds),
            verify_replay=bool(verify_replay),
            branch_evidence_chunk_dir=chunk_dir,
            resumed_branch_results=resumed_chunks,
            started_at=started_at,
            max_wall_seconds=max_wall_seconds,
            progress_callback=progress_callback,
            seed_include=seed_filter,
            branch_index_include=selected_branch_indexes,
        )
    else:
        broad = _safe_archive_expansion_unselected_fixture_evidence(
            fixture="broad",
            seeds=(),
            ticks=STRICT_TICKS,
            max_branch_points_per_seed=branch_limit,
            max_candidate_actions=candidate_limit,
        )
    selected_carrion_seeds = _limit_int_sequence(
        carrion_seeds,
        limit=int(max_carrion_fixture_seeds),
    )
    if seed_filter is not None:
        selected_carrion_seeds = _filter_int_sequence(
            selected_carrion_seeds,
            include=seed_filter,
        )
    if _safe_archive_expansion_status_partial(_mapping(broad.get("generation_status"))):
        carrion = _safe_archive_expansion_empty_fixture_evidence(
            fixture="carrion_only",
            stop_reason="skipped_after_broad_partial_branch_evidence",
            seeds=selected_carrion_seeds,
            ticks=int(carrion_ticks),
            max_branch_points_per_seed=branch_limit,
            max_candidate_actions=candidate_limit,
        )
    elif "carrion_only" in selected_fixtures:
        carrion = _safe_archive_expansion_carrion_branch_evidence(
            seeds=selected_carrion_seeds,
            ticks=int(carrion_ticks),
            max_branch_points_per_seed=branch_limit,
            max_candidate_actions=candidate_limit,
            verify_replay=bool(verify_replay),
            branch_evidence_chunk_dir=chunk_dir,
            resumed_branch_results=resumed_chunks,
            started_at=started_at,
            max_wall_seconds=max_wall_seconds,
            progress_callback=progress_callback,
            branch_index_include=selected_branch_indexes,
        )
    else:
        carrion = _safe_archive_expansion_unselected_fixture_evidence(
            fixture="carrion_only",
            seeds=selected_carrion_seeds,
            ticks=int(carrion_ticks),
            max_branch_points_per_seed=branch_limit,
            max_candidate_actions=candidate_limit,
        )
    branch_results = [
        *_list_of_mappings(broad.get("branch_results")),
        *_list_of_mappings(carrion.get("branch_results")),
    ]
    first_missing = _first_missing_valid_action(branch_results)
    broad_source_integrity = _mapping(broad.get("source_integrity"))
    carrion_source_integrity = _mapping(carrion.get("source_integrity"))
    source_failures = []
    if broad_source_integrity.get("passed") is not True:
        source_failures.append("broad_branch_evidence_source_integrity_failed")
    if carrion_source_integrity.get("passed") is not True:
        source_failures.append("carrion_branch_evidence_source_integrity_failed")
    if first_missing is not None:
        source_failures.append("not_all_valid_candidate_actions_evaluated")
    generation_status = _safe_archive_expansion_generation_status(
        broad=_mapping(broad.get("generation_status")),
        carrion=_mapping(carrion.get("generation_status")),
        started_at=started_at,
        max_wall_seconds=max_wall_seconds,
        chunk_dir=chunk_dir,
        resumed_chunk_count=len(resumed_chunks),
    )
    if _safe_archive_expansion_status_partial(generation_status):
        source_failures.append("partial_branch_evidence")
    return {
        "schema_version": M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION,
        "policy": f"{M3_SAFE_ARCHIVE_EXPANSION_POLICY}_branch_evidence_v1",
        "contract": {
            "diagnostics_only": True,
            "training_authorized": False,
            "promotion_authorized": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
            "heuristic_fallback_added": False,
            "gate_relaxation": False,
            "branch_replay_policy": (
                "deterministic_public_mask_valid_action_forced_first_action_v1"
            ),
            "candidate_action_policy": (
                "all_currently_valid_public_action_mask_actions_when_"
                "max_candidate_actions_is_zero"
            ),
            "max_candidate_actions": (
                None if candidate_limit == 0 else int(candidate_limit)
            ),
        },
        "inputs": {
            "v142_scorer_report": str(v142_scorer_report_path),
            "v142_live_report": str(v142_live_report_path),
            "v142_trajectory_output_dir": str(v142_trajectory_output_dir),
            "v142_trajectory_regeneration_requested": bool(
                regenerate_v142_trajectories
            ),
            "v142_trajectory_regeneration_used": False,
            "max_broad_regression_seeds": int(max_broad_regression_seeds),
            "max_carrion_fixture_seeds": int(max_carrion_fixture_seeds),
            "carrion_seed_pool": [int(seed) for seed in carrion_seeds],
            "carrion_seeds": [int(seed) for seed in selected_carrion_seeds],
            "carrion_ticks": int(carrion_ticks),
            "max_branch_points_per_seed": branch_limit,
            "verify_replay": bool(verify_replay),
            "resume_branch_evidence": bool(resume_branch_evidence),
            "branch_evidence_chunk_dir": None if chunk_dir is None else str(chunk_dir),
            "max_wall_seconds": (
                None if max_wall_seconds is None else float(max_wall_seconds)
            ),
            "shard_id": shard_id,
            "fixture_selection": list(selected_fixtures),
            "seed_include": None if seed_filter is None else [int(seed) for seed in seed_filter],
            "branch_index_selection": {
                "start": None if branch_index_start is None else int(branch_index_start),
                "count": None if branch_index_count is None else int(branch_index_count),
                "include": [int(index) for index in selected_branch_indexes],
            },
        },
        "generation_status": generation_status,
        "broad": broad,
        "carrion": carrion,
        "source_integrity": {
            "policy": "m3_safe_archive_expansion_branch_evidence_source_integrity_v1",
            "passed": not source_failures,
            "failures": sorted(set(source_failures)),
            "first_missing_valid_action": first_missing,
            "smallest_next_implementation_needed": (
                None
                if first_missing is None
                else (
                    "regenerate branch evidence with max_candidate_actions=0 "
                    "from the public action mask for the missing branch/action"
                )
            ),
        },
        "coverage": _safe_archive_expansion_evidence_coverage(branch_results),
        "branch_results": branch_results,
        "branch_result_count": len(branch_results),
        "branch_evidence_digest": stable_payload_digest(branch_results),
        "training_authorized": False,
        "promotion_authorized": False,
        "non_default_runtime": True,
        "non_promoted": True,
    }


def write_safe_archive_expansion_branch_evidence(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    write_json(output_path, dict(report))


def write_safe_archive_expansion_branch_result_chunk(
    branch_result: Mapping[str, object],
    chunk_dir: str | Path,
) -> Path:
    path = _safe_archive_expansion_branch_result_chunk_path(
        branch_result,
        chunk_dir=chunk_dir,
    )
    write_json(
        path,
        {
            "schema_version": M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION,
            "policy": "m3_safe_archive_expansion_branch_result_chunk_v1",
            "branch_result_digest": stable_payload_digest(branch_result),
            "branch_result": dict(branch_result),
        },
    )
    return path


def load_safe_archive_expansion_branch_result_chunks(
    chunk_dir: str | Path,
) -> dict[str, dict[str, object]]:
    path = Path(chunk_dir)
    if not path.exists():
        return {}
    if not path.is_dir():
        raise CandidateCampaignError(f"branch evidence chunk path is not a directory: {path}")
    chunks: dict[str, dict[str, object]] = {}
    digests_by_branch: dict[str, str] = {}
    for chunk_path in sorted(path.glob("*.json")):
        payload = load_json_report(chunk_path)
        if payload.get("schema_version") != M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION:
            raise CandidateCampaignError(
                f"branch result chunk schema mismatch: {chunk_path}"
            )
        if payload.get("policy") != "m3_safe_archive_expansion_branch_result_chunk_v1":
            raise CandidateCampaignError(
                f"branch result chunk policy mismatch: {chunk_path}"
            )
        result = _mapping(payload.get("branch_result"))
        branch_id = str(result.get("branch_id", ""))
        if not branch_id:
            raise CandidateCampaignError(
                f"branch result chunk missing branch_id: {chunk_path}"
            )
        expected_digest = str(payload.get("branch_result_digest", ""))
        actual_digest = stable_payload_digest(result)
        if expected_digest and expected_digest != actual_digest:
            raise CandidateCampaignError(
                f"branch result chunk digest mismatch: {chunk_path}"
            )
        prior_digest = digests_by_branch.get(branch_id)
        if prior_digest is not None and prior_digest != actual_digest:
            raise CandidateCampaignError(
                f"conflicting branch result chunks for branch_id={branch_id}"
            )
        chunks[branch_id] = dict(result)
        digests_by_branch[branch_id] = actual_digest
    return dict(sorted(chunks.items()))


def merge_safe_archive_expansion_branch_evidence(
    *,
    shard_evidence_reports: Sequence[Mapping[str, object]] = (),
    shard_chunk_dirs: Sequence[str | Path] = (),
    allow_partial_shard_evidence: bool = False,
) -> dict[str, object]:
    if shard_chunk_dirs and not shard_evidence_reports:
        raise CandidateCampaignError(
            "chunk-dir shard merge requires shard evidence reports for generation identity"
        )
    merged_by_branch_id: dict[str, dict[str, object]] = {}
    digests_by_branch_id: dict[str, str] = {}
    merge_identity: dict[str, object] | None = None
    merge_contract: dict[str, object] | None = None
    source_summaries: list[dict[str, object]] = []
    partial_sources: list[dict[str, object]] = []

    for shard_index, evidence in enumerate(shard_evidence_reports):
        if evidence.get("schema_version") != M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION:
            raise CandidateCampaignError(f"shard {shard_index} schema mismatch")
        if evidence.get("policy") != f"{M3_SAFE_ARCHIVE_EXPANSION_POLICY}_branch_evidence_v1":
            raise CandidateCampaignError(f"shard {shard_index} policy mismatch")
        identity = _safe_archive_expansion_merge_identity(evidence)
        if merge_identity is None:
            merge_identity = identity
            merge_contract = dict(_mapping(evidence.get("contract")))
        elif identity != merge_identity:
            raise CandidateCampaignError(f"shard {shard_index} generation inputs mismatch")
        status = _mapping(evidence.get("generation_status"))
        source_integrity = _mapping(evidence.get("source_integrity"))
        integrity_failures = [
            str(failure) for failure in _list(source_integrity.get("failures"))
        ]
        non_partial_integrity_failures = sorted(
            {
                failure
                for failure in integrity_failures
                if failure != "partial_branch_evidence"
            }
        )
        status_is_partial = _safe_archive_expansion_status_partial(status)
        integrity_is_partial = "partial_branch_evidence" in set(integrity_failures)
        if source_integrity.get("passed") is not True:
            if non_partial_integrity_failures:
                raise CandidateCampaignError(
                    "shard "
                    f"{shard_index} source integrity failed: "
                    f"{non_partial_integrity_failures}"
                )
            if not (status_is_partial or integrity_is_partial):
                raise CandidateCampaignError(
                    f"shard {shard_index} source integrity failed without partial status"
                )
        if status_is_partial or integrity_is_partial:
            summary = {
                "source": f"evidence:{shard_index}",
                "state": status.get("state"),
                "stop_reason": status.get("stop_reason"),
                "source_integrity_failures": integrity_failures,
            }
            if not allow_partial_shard_evidence:
                raise CandidateCampaignError(
                    f"partial shard evidence requires explicit partial merge: {summary}"
                )
            partial_sources.append(summary)
        branch_results = _list_of_mappings(evidence.get("branch_results"))
        for result in branch_results:
            _safe_archive_expansion_add_merged_branch_result(
                merged_by_branch_id=merged_by_branch_id,
                digests_by_branch_id=digests_by_branch_id,
                result=result,
                source=f"evidence:{shard_index}",
            )
        source_summaries.append(
            {
                "source": f"evidence:{shard_index}",
                "branch_result_count": len(branch_results),
                "branch_evidence_digest": evidence.get("branch_evidence_digest"),
                "partial": _safe_archive_expansion_status_partial(status),
                "shard_id": _mapping(evidence.get("inputs")).get("shard_id"),
            }
        )

    for chunk_dir in shard_chunk_dirs:
        chunks = load_safe_archive_expansion_branch_result_chunks(chunk_dir)
        for result in chunks.values():
            branch_id = str(result.get("branch_id", ""))
            if branch_id not in merged_by_branch_id:
                raise CandidateCampaignError(
                    "chunk-dir branch result lacks matching shard evidence report: "
                    f"{chunk_dir}:{branch_id}"
                )
            _safe_archive_expansion_add_merged_branch_result(
                merged_by_branch_id=merged_by_branch_id,
                digests_by_branch_id=digests_by_branch_id,
                result=result,
                source=f"chunk_dir:{chunk_dir}",
            )
        source_summaries.append(
            {
                "source": f"chunk_dir:{chunk_dir}",
                "branch_result_count": len(chunks),
                "partial": False,
            }
        )

    if not merged_by_branch_id:
        raise CandidateCampaignError("safe archive shard merge has no branch results")
    branch_results = sorted(
        merged_by_branch_id.values(),
        key=_safe_archive_expansion_branch_result_sort_key,
    )
    first_missing = _first_missing_valid_action(branch_results)
    source_failures: list[str] = []
    if first_missing is not None:
        source_failures.append("not_all_valid_candidate_actions_evaluated")
    if partial_sources:
        source_failures.append("partial_branch_evidence")
    generation_status = {
        "policy": "m3_safe_archive_expansion_merged_branch_evidence_status_v1",
        "state": "partial" if partial_sources else "complete",
        "partial": bool(partial_sources),
        "stop_reason": "partial_shard_evidence" if partial_sources else None,
        "source_count": len(source_summaries),
        "partial_sources": partial_sources,
        "branch_result_count": len(branch_results),
    }
    return {
        "schema_version": M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION,
        "policy": f"{M3_SAFE_ARCHIVE_EXPANSION_POLICY}_branch_evidence_v1",
        "contract": merge_contract or _safe_archive_expansion_default_branch_evidence_contract(),
        "inputs": {
            **(merge_identity or {}),
            "merge_mode": True,
            "merged_source_count": len(source_summaries),
            "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        },
        "generation_status": generation_status,
        "source_integrity": {
            "policy": "m3_safe_archive_expansion_merged_branch_evidence_source_integrity_v1",
            "passed": not source_failures,
            "failures": sorted(set(source_failures)),
            "first_missing_valid_action": first_missing,
            "smallest_next_implementation_needed": (
                None
                if first_missing is None
                else (
                    "regenerate the shard containing the missing branch/action "
                    "with max_candidate_actions=0"
                )
            ),
        },
        "shard_merge": {
            "policy": "m3_safe_archive_expansion_shard_merge_v1",
            "sources": source_summaries,
            "partial_sources": partial_sources,
            "duplicate_branch_id_policy": "same_digest_allowed_conflict_rejected",
        },
        "coverage": _safe_archive_expansion_evidence_coverage(branch_results),
        "branch_results": branch_results,
        "branch_result_count": len(branch_results),
        "branch_evidence_digest": stable_payload_digest(branch_results),
        "training_authorized": False,
        "promotion_authorized": False,
        "non_default_runtime": True,
        "non_promoted": True,
    }


def _safe_archive_expansion_merge_identity(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    inputs = _mapping(evidence.get("inputs"))
    contract = _mapping(evidence.get("contract"))
    return {
        "schema_version": evidence.get("schema_version"),
        "policy": evidence.get("policy"),
        "branch_replay_policy": contract.get("branch_replay_policy"),
        "candidate_action_policy": contract.get("candidate_action_policy"),
        "max_candidate_actions": contract.get("max_candidate_actions"),
        "v142_scorer_report": inputs.get("v142_scorer_report"),
        "v142_live_report": inputs.get("v142_live_report"),
        "v142_trajectory_output_dir": inputs.get("v142_trajectory_output_dir"),
        "carrion_ticks": inputs.get("carrion_ticks"),
        "max_branch_points_per_seed": inputs.get("max_branch_points_per_seed"),
        "verify_replay": inputs.get("verify_replay"),
    }


def _safe_archive_expansion_default_branch_evidence_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "heuristic_fallback_added": False,
        "gate_relaxation": False,
        "branch_replay_policy": (
            "deterministic_public_mask_valid_action_forced_first_action_v1"
        ),
        "candidate_action_policy": (
            "all_currently_valid_public_action_mask_actions_when_"
            "max_candidate_actions_is_zero"
        ),
        "max_candidate_actions": None,
    }


def _safe_archive_expansion_add_merged_branch_result(
    *,
    merged_by_branch_id: dict[str, dict[str, object]],
    digests_by_branch_id: dict[str, str],
    result: Mapping[str, object],
    source: str,
) -> None:
    branch_id = str(result.get("branch_id", ""))
    if not branch_id:
        raise CandidateCampaignError(f"merged branch result missing branch_id: {source}")
    if not _safe_archive_expansion_branch_result_has_replay_verification(result):
        raise CandidateCampaignError(
            f"merged branch result missing replay verification: {source}:{branch_id}"
        )
    digest = stable_payload_digest(result)
    prior_digest = digests_by_branch_id.get(branch_id)
    if prior_digest is not None and prior_digest != digest:
        raise CandidateCampaignError(
            f"duplicate branch_id with different digest: {branch_id}"
        )
    merged_by_branch_id[branch_id] = dict(result)
    digests_by_branch_id[branch_id] = digest


def _safe_archive_expansion_branch_result_has_replay_verification(
    result: Mapping[str, object],
) -> bool:
    action_runs = _list_of_mappings(result.get("action_runs"))
    if not action_runs:
        return False
    for run in action_runs:
        replay = _mapping(run.get("replay_verification"))
        if replay.get("verified") is not True:
            return False
    return True


def _safe_archive_expansion_branch_result_sort_key(
    result: Mapping[str, object],
) -> tuple[int, int, int, str]:
    fixture = str(result.get("fixture", ""))
    fixture_order = 0 if fixture == "broad" else 1 if fixture == "carrion_only" else 2
    return (
        fixture_order,
        _int(result.get("seed")),
        _int(result.get("branch_index")),
        str(result.get("branch_id", "")),
    )


def _safe_archive_expansion_branch_result_chunk_path(
    branch_result: Mapping[str, object],
    *,
    chunk_dir: str | Path,
) -> Path:
    branch_id = str(branch_result.get("branch_id", ""))
    if not branch_id:
        raise CandidateCampaignError("branch result chunk requires branch_id")
    seed = _int(branch_result.get("seed"))
    fixture = str(branch_result.get("fixture", "unknown"))
    identity_digest = stable_payload_digest(
        {"branch_id": branch_id, "fixture": fixture, "seed": seed}
    )[:20]
    slug = _safe_archive_expansion_chunk_slug(f"{fixture}-{seed}-{branch_id}")
    return Path(chunk_dir) / f"{identity_digest}-{slug}.json"


def _safe_archive_expansion_chunk_slug(value: str) -> str:
    chars = [
        ch if ch.isalnum() or ch in {"-", "_"} else "-"
        for ch in str(value).strip()
    ]
    slug = "".join(chars).strip("-")
    return (slug or "branch-result")[:120]


def _safe_archive_expansion_evaluate_branch_point_checkpointed(
    point: BroadRegressionBranchPoint,
    *,
    reference_runs: Mapping[int, Mapping[str, Mapping[str, object]]],
    max_candidate_actions: int,
    verify_replay: bool,
    chunk_dir: Path | None,
    resumed_branch_results: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], str]:
    cached = resumed_branch_results.get(point.branch_id)
    if cached is not None and _safe_archive_expansion_branch_result_matches_point(
        cached,
        point,
        max_candidate_actions=int(max_candidate_actions),
        verify_replay=bool(verify_replay),
    ):
        return dict(cached), "resumed"
    result = _v143_evaluate_branch_point(
        point,
        reference_runs=reference_runs,
        max_candidate_actions=int(max_candidate_actions),
        verify_replay=bool(verify_replay),
    )
    if chunk_dir is not None:
        write_safe_archive_expansion_branch_result_chunk(result, chunk_dir)
    return dict(result), "generated"


def _safe_archive_expansion_selected_fixtures(
    fixtures: Sequence[str],
) -> tuple[str, ...]:
    if not fixtures:
        return ("broad", "carrion_only")
    selected: list[str] = []
    for fixture in fixtures:
        value = str(fixture)
        if value == "both":
            for item in ("broad", "carrion_only"):
                if item not in selected:
                    selected.append(item)
            continue
        if value not in {"broad", "carrion_only"}:
            raise CandidateCampaignError(f"unsupported safe archive fixture: {value}")
        if value not in selected:
            selected.append(value)
    return tuple(selected)


def _safe_archive_expansion_selected_branch_indexes(
    *,
    max_branch_points_per_seed: int,
    branch_index_start: int | None = None,
    branch_index_count: int | None = None,
    branch_index_include: Sequence[int] | None = None,
) -> tuple[int, ...]:
    limit = max(1, int(max_branch_points_per_seed))
    allowed = set(range(limit))
    if branch_index_include is not None:
        selected = {int(index) for index in branch_index_include}
    elif branch_index_start is not None:
        start = max(0, int(branch_index_start))
        if branch_index_count is None:
            end = limit
        else:
            end = start + max(0, int(branch_index_count))
        selected = set(range(start, end))
    else:
        selected = set(range(limit))
    selected &= allowed
    if not selected:
        raise CandidateCampaignError("branch index shard selection is empty")
    return tuple(sorted(selected))


def _safe_archive_expansion_select_branch_points(
    points: Sequence[BroadRegressionBranchPoint],
    *,
    branch_index_include: Sequence[int],
) -> list[BroadRegressionBranchPoint]:
    selected = {int(index) for index in branch_index_include}
    return [point for point in points if int(point.branch_index) in selected]


def _filter_int_sequence(
    values: Sequence[int],
    *,
    include: Sequence[int],
) -> tuple[int, ...]:
    selected = {int(value) for value in include}
    return tuple(int(value) for value in values if int(value) in selected)


def _safe_archive_expansion_branch_result_matches_point(
    result: Mapping[str, object],
    point: BroadRegressionBranchPoint,
    *,
    max_candidate_actions: int,
    verify_replay: bool,
) -> bool:
    public_features = _mapping(result.get("public_features"))
    expected_actions = _v143_candidate_actions(
        point,
        max_candidate_actions=int(max_candidate_actions),
    )
    action_runs = _list_of_mappings(result.get("action_runs"))
    observed_run_actions = [
        str(run.get("forced_action"))
        for run in action_runs
        if str(run.get("forced_action")) in ACTION_NAMES
    ]
    if observed_run_actions != expected_actions:
        return False
    if _list(result.get("candidate_actions")) != expected_actions:
        return False
    if bool(verify_replay):
        for run in action_runs:
            replay = _mapping(run.get("replay_verification"))
            if replay.get("verified") is not True:
                return False
    expected_identity = {
        "branch_id": point.branch_id,
        "seed": int(point.seed),
        "fixture": point.fixture,
        "ticks": int(point.ticks),
        "branch_tick": int(point.branch_tick),
        "record_index": int(point.record_index),
        "branch_index": int(point.branch_index),
        "agent_id": int(point.agent_id),
        "baseline_action": point.baseline_action,
        "v142_requested_action": point.v142_requested_action,
        "v142_resolved_action": point.v142_resolved_action,
        "branch_state_digest": point.branch_state_digest,
    }
    observed_identity = {
        "branch_id": str(result.get("branch_id", "")),
        "seed": _int(result.get("seed"), default=-1),
        "fixture": str(result.get("fixture", "")),
        "ticks": _int(result.get("ticks"), default=-1),
        "branch_tick": _int(result.get("branch_tick"), default=-1),
        "record_index": _int(result.get("record_index"), default=-1),
        "branch_index": _int(result.get("branch_index"), default=-1),
        "agent_id": _int(result.get("agent_id"), default=-1),
        "baseline_action": str(result.get("baseline_action", "")),
        "v142_requested_action": str(result.get("v142_requested_action", "")),
        "v142_resolved_action": str(result.get("v142_resolved_action", "")),
        "branch_state_digest": str(result.get("branch_state_digest", "")),
    }
    if observed_identity != expected_identity:
        return False
    if _mapping(public_features.get("observation_input")) != point.observation_input:
        return False
    if _bool_action_mask(public_features.get("action_mask")) != _bool_action_mask(
        point.action_mask
    ):
        return False
    return True


def _safe_archive_expansion_empty_fixture_evidence(
    *,
    fixture: str,
    stop_reason: str,
    seeds: Sequence[int],
    ticks: int,
    max_branch_points_per_seed: int,
    max_candidate_actions: int,
) -> dict[str, object]:
    generation_status = {
        "policy": "m3_safe_archive_expansion_fixture_generation_status_v1",
        "fixture": str(fixture),
        "state": "partial",
        "partial": True,
        "stop_reason": str(stop_reason),
        "elapsed_seconds": 0.0,
        "max_wall_seconds": None,
        "branch_point_count": 0,
        "branch_result_count": 0,
        "generated_branch_result_count": 0,
        "resumed_branch_result_count": 0,
        "chunk_dir": None,
    }
    return {
        "policy": f"m3_safe_archive_expansion_{fixture}_branch_evidence_v1",
        "fixture": str(fixture),
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "max_candidate_actions": (
            None if int(max_candidate_actions) == 0 else int(max_candidate_actions)
        ),
        "materialization": {
            "policy": "m3_safe_archive_expansion_skipped_materialization_v1",
            "seed_reports": [],
            "branch_point_count": 0,
            "branch_points_by_seed": {},
            "failure_count": 0,
            "failures": [],
            "passed": False,
        },
        "source_integrity": {
            "policy": "m3_safe_archive_expansion_skipped_source_integrity_v1",
            "passed": False,
            "failures": ["partial_branch_evidence"],
            "first_missing_valid_action": None,
            "replay_verification_count": 0,
            "replay_verification_failure_count": 0,
        },
        "generation_status": generation_status,
        "branch_points": [],
        "branch_results": [],
        "branch_result_count": 0,
    }


def _safe_archive_expansion_unselected_fixture_evidence(
    *,
    fixture: str,
    seeds: Sequence[int],
    ticks: int,
    max_branch_points_per_seed: int,
    max_candidate_actions: int,
) -> dict[str, object]:
    return {
        "policy": f"m3_safe_archive_expansion_{fixture}_branch_evidence_v1",
        "fixture": str(fixture),
        "selected": False,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "max_candidate_actions": (
            None if int(max_candidate_actions) == 0 else int(max_candidate_actions)
        ),
        "materialization": {
            "policy": "m3_safe_archive_expansion_unselected_materialization_v1",
            "seed_reports": [],
            "branch_point_count": 0,
            "branch_points_by_seed": {},
            "failure_count": 0,
            "failures": [],
            "passed": True,
        },
        "source_integrity": {
            "policy": "m3_safe_archive_expansion_unselected_source_integrity_v1",
            "passed": True,
            "failures": [],
            "first_missing_valid_action": None,
            "replay_verification_count": 0,
            "replay_verification_failure_count": 0,
        },
        "generation_status": {
            "policy": "m3_safe_archive_expansion_fixture_generation_status_v1",
            "fixture": str(fixture),
            "state": "complete",
            "partial": False,
            "selected": False,
            "stop_reason": None,
            "elapsed_seconds": 0.0,
            "max_wall_seconds": None,
            "branch_point_count": 0,
            "branch_result_count": 0,
            "generated_branch_result_count": 0,
            "resumed_branch_result_count": 0,
            "chunk_dir": None,
        },
        "branch_points": [],
        "branch_results": [],
        "branch_result_count": 0,
    }


def _safe_archive_expansion_fixture_generation_status(
    *,
    fixture: str,
    stop_reason: str | None,
    started_at: float,
    max_wall_seconds: float | None,
    branch_point_count: int,
    branch_result_count: int,
    generated_branch_result_count: int,
    resumed_branch_result_count: int,
    chunk_dir: Path | None,
) -> dict[str, object]:
    partial = stop_reason is not None or int(branch_result_count) < int(branch_point_count)
    return {
        "policy": "m3_safe_archive_expansion_fixture_generation_status_v1",
        "fixture": str(fixture),
        "state": "partial" if partial else "complete",
        "partial": bool(partial),
        "stop_reason": stop_reason,
        "elapsed_seconds": _safe_archive_expansion_elapsed_seconds(started_at),
        "max_wall_seconds": (
            None if max_wall_seconds is None else float(max_wall_seconds)
        ),
        "branch_point_count": int(branch_point_count),
        "branch_result_count": int(branch_result_count),
        "generated_branch_result_count": int(generated_branch_result_count),
        "resumed_branch_result_count": int(resumed_branch_result_count),
        "chunk_dir": None if chunk_dir is None else str(chunk_dir),
    }


def _safe_archive_expansion_generation_status(
    *,
    broad: Mapping[str, object],
    carrion: Mapping[str, object],
    started_at: float,
    max_wall_seconds: float | None,
    chunk_dir: Path | None,
    resumed_chunk_count: int,
) -> dict[str, object]:
    fixture_statuses = [dict(broad), dict(carrion)]
    partial = any(_safe_archive_expansion_status_partial(status) for status in fixture_statuses)
    stop_reasons = [
        str(status.get("stop_reason"))
        for status in fixture_statuses
        if status.get("stop_reason") is not None
    ]
    return {
        "policy": "m3_safe_archive_expansion_branch_evidence_generation_status_v1",
        "state": "partial" if partial else "complete",
        "partial": bool(partial),
        "stop_reason": stop_reasons[0] if stop_reasons else None,
        "elapsed_seconds": _safe_archive_expansion_elapsed_seconds(started_at),
        "max_wall_seconds": (
            None if max_wall_seconds is None else float(max_wall_seconds)
        ),
        "chunk_dir": None if chunk_dir is None else str(chunk_dir),
        "resumed_chunk_count": int(resumed_chunk_count),
        "fixture_statuses": fixture_statuses,
    }


def _safe_archive_expansion_status_partial(status: Mapping[str, object]) -> bool:
    return status.get("partial") is True or status.get("state") == "partial"


def _safe_archive_expansion_wall_budget_exhausted(
    started_at: float,
    max_wall_seconds: float | None,
) -> bool:
    if max_wall_seconds is None:
        return False
    return (time.monotonic() - float(started_at)) >= max(0.0, float(max_wall_seconds))


def _safe_archive_expansion_elapsed_seconds(started_at: float) -> float:
    return round(max(0.0, time.monotonic() - float(started_at)), 3)


def _safe_archive_expansion_emit_progress(
    progress_callback: Callable[[Mapping[str, object]], None] | None,
    payload: Mapping[str, object],
) -> None:
    if progress_callback is not None:
        progress_callback(dict(payload))


def _safe_archive_expansion_broad_branch_evidence(
    *,
    v142_live_report_path: str | Path,
    ticks: int,
    max_branch_points_per_seed: int,
    max_candidate_actions: int,
    max_regression_seeds: int,
    verify_replay: bool,
    branch_evidence_chunk_dir: str | Path | None = None,
    resumed_branch_results: Mapping[str, Mapping[str, object]] | None = None,
    started_at: float | None = None,
    max_wall_seconds: float | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    seed_include: Sequence[int] | None = None,
    branch_index_include: Sequence[int] | None = None,
) -> dict[str, object]:
    start = time.monotonic() if started_at is None else float(started_at)
    chunk_dir = Path(branch_evidence_chunk_dir) if branch_evidence_chunk_dir else None
    resumed = resumed_branch_results or {}
    live_report = load_json_report(v142_live_report_path)
    regression_seed_pool = _ordered_regression_seeds(live_report)
    regression_seeds = tuple(
        _limit_int_sequence(
            regression_seed_pool,
            limit=int(max_regression_seeds),
        )
    )
    if seed_include is not None:
        regression_seeds = _filter_int_sequence(regression_seeds, include=seed_include)
    selected_branch_indexes = (
        tuple(range(int(max_branch_points_per_seed)))
        if branch_index_include is None
        else tuple(int(index) for index in branch_index_include)
    )
    branch_results: list[dict[str, object]] = []
    branch_points: list[BroadRegressionBranchPoint] = []
    seed_reports: list[dict[str, object]] = []
    reference_runs: dict[int, dict[str, Mapping[str, object]]] = {}
    failures: list[dict[str, object]] = []
    generated_count = 0
    resumed_count = 0
    stop_reason: str | None = None
    for seed in regression_seeds:
        if _safe_archive_expansion_wall_budget_exhausted(start, max_wall_seconds):
            stop_reason = "max_wall_seconds_elapsed_before_seed"
            break
        seed_points, reference, seed_report = _materialize_safe_archive_broad_points(
            seed=int(seed),
            ticks=int(ticks),
            max_branch_points=int(max_branch_points_per_seed),
        )
        seed_reports.append(seed_report)
        branch_points.extend(seed_points)
        failures.extend(_list_of_mappings(seed_report.get("failures")))
        reference_runs[int(seed)] = {
            "baseline": reference,
            "v142_override": reference,
        }
        selected_seed_points = _safe_archive_expansion_select_branch_points(
            seed_points,
            branch_index_include=selected_branch_indexes,
        )
        for point in selected_seed_points:
            if _safe_archive_expansion_wall_budget_exhausted(start, max_wall_seconds):
                stop_reason = "max_wall_seconds_elapsed_before_branch_point"
                break
            result, source = _safe_archive_expansion_evaluate_branch_point_checkpointed(
                point,
                reference_runs=reference_runs,
                max_candidate_actions=int(max_candidate_actions),
                verify_replay=bool(verify_replay),
                chunk_dir=chunk_dir,
                resumed_branch_results=resumed,
            )
            branch_results.append(result)
            if source == "resumed":
                resumed_count += 1
            else:
                generated_count += 1
            _safe_archive_expansion_emit_progress(
                progress_callback,
                {
                    "event": "branch_result",
                    "source": source,
                    "fixture": "broad",
                    "seed": int(point.seed),
                    "branch_id": point.branch_id,
                    "branch_point_index": int(point.branch_index),
                    "action_count": len(_list_of_mappings(result.get("action_runs"))),
                    "elapsed_seconds": _safe_archive_expansion_elapsed_seconds(start),
                },
            )
        if stop_reason is not None:
            break
    selected_branch_point_count = sum(
        1 for point in branch_points if int(point.branch_index) in set(selected_branch_indexes)
    )
    generation_status = _safe_archive_expansion_fixture_generation_status(
        fixture="broad",
        stop_reason=stop_reason,
        started_at=start,
        max_wall_seconds=max_wall_seconds,
        branch_point_count=selected_branch_point_count,
        branch_result_count=len(branch_results),
        generated_branch_result_count=generated_count,
        resumed_branch_result_count=resumed_count,
        chunk_dir=chunk_dir,
    )
    if _safe_archive_expansion_status_partial(generation_status):
        _safe_archive_expansion_emit_progress(
            progress_callback,
            {
                "event": "partial_stop",
                "fixture": "broad",
                "seed": None,
                "branch_id": None,
                "branch_point_index": len(branch_results),
                "action_count": 0,
                "elapsed_seconds": _safe_archive_expansion_elapsed_seconds(start),
                "stop_reason": stop_reason,
            },
        )
    first_missing = _first_missing_valid_action(branch_results)
    replay_items = [
        _mapping(run.get("replay_verification"))
        for result in branch_results
        for run in _list_of_mappings(result.get("action_runs"))
        if run.get("replay_verification") is not None
    ]
    replay_failure_count = sum(
        1 for item in replay_items if item.get("verified") is not True
    )
    source_failures: list[str] = []
    if not regression_seeds:
        source_failures.append("no_broad_regression_seeds")
    if failures:
        source_failures.append("broad_branch_point_materialization_failed")
    if selected_branch_point_count <= 0:
        source_failures.append("no_broad_branch_points")
    if first_missing is not None:
        source_failures.append("not_all_valid_candidate_actions_evaluated")
    if bool(verify_replay) and replay_failure_count > 0:
        source_failures.append("broad_branch_replay_verification_failed")
    if _safe_archive_expansion_status_partial(generation_status):
        source_failures.append("partial_branch_evidence")
    return {
        "policy": "m3_safe_archive_expansion_broad_regression_branch_evidence_v1",
        "fixture": "broad",
        "v142_live_report": str(v142_live_report_path),
        "regression_seed_policy": "v142_live_ab_first_failed_then_regression_seeds_v1",
        "regression_seed_pool": [int(seed) for seed in regression_seed_pool],
        "max_regression_seeds": int(max_regression_seeds),
        "regression_seeds": [int(seed) for seed in regression_seeds],
        "ticks": int(ticks),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "max_candidate_actions": (
            None if int(max_candidate_actions) == 0 else int(max_candidate_actions)
        ),
        "branch_index_selection": [int(index) for index in selected_branch_indexes],
        "materialization": {
            "policy": "m3_safe_archive_expansion_broad_materialization_v1",
            "seed_reports": seed_reports,
            "materialized_branch_point_count": len(branch_points),
            "branch_point_count": sum(
                1
                for point in branch_points
                if int(point.branch_index) in set(selected_branch_indexes)
            ),
            "branch_points_by_seed": dict(
                sorted(
                    Counter(
                        point.seed
                        for point in branch_points
                        if int(point.branch_index) in set(selected_branch_indexes)
                    ).items()
                )
            ),
            "failure_count": len(failures),
            "failures": failures[:24],
            "passed": not failures and bool(branch_points),
        },
        "source_integrity": {
            "policy": "m3_safe_archive_expansion_broad_source_integrity_v1",
            "passed": not source_failures,
            "failures": sorted(set(source_failures)),
            "first_missing_valid_action": first_missing,
            "replay_verification_count": len(replay_items),
            "replay_verification_failure_count": int(replay_failure_count),
        },
        "generation_status": generation_status,
        "branch_points": [
            _safe_archive_expansion_branch_point_payload(point)
            for point in branch_points
            if int(point.branch_index) in set(selected_branch_indexes)
        ],
        "branch_results": branch_results,
        "branch_result_count": len(branch_results),
    }


def _safe_archive_expansion_carrion_branch_evidence(
    *,
    seeds: Sequence[int],
    ticks: int,
    max_branch_points_per_seed: int,
    max_candidate_actions: int,
    verify_replay: bool,
    branch_evidence_chunk_dir: str | Path | None = None,
    resumed_branch_results: Mapping[str, Mapping[str, object]] | None = None,
    started_at: float | None = None,
    max_wall_seconds: float | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    branch_index_include: Sequence[int] | None = None,
) -> dict[str, object]:
    start = time.monotonic() if started_at is None else float(started_at)
    chunk_dir = Path(branch_evidence_chunk_dir) if branch_evidence_chunk_dir else None
    resumed = resumed_branch_results or {}
    branch_results: list[dict[str, object]] = []
    branch_points: list[BroadRegressionBranchPoint] = []
    seed_reports: list[dict[str, object]] = []
    reference_runs: dict[int, dict[str, Mapping[str, object]]] = {}
    failures: list[dict[str, object]] = []
    generated_count = 0
    resumed_count = 0
    stop_reason: str | None = None
    selected_branch_indexes = (
        tuple(range(int(max_branch_points_per_seed)))
        if branch_index_include is None
        else tuple(int(index) for index in branch_index_include)
    )
    for seed in seeds:
        if _safe_archive_expansion_wall_budget_exhausted(start, max_wall_seconds):
            stop_reason = "max_wall_seconds_elapsed_before_seed"
            break
        seed_points, reference, seed_report = _materialize_safe_archive_carrion_points(
            seed=int(seed),
            ticks=int(ticks),
            max_branch_points=int(max_branch_points_per_seed),
        )
        seed_reports.append(seed_report)
        branch_points.extend(seed_points)
        failures.extend(_list_of_mappings(seed_report.get("failures")))
        reference_runs[int(seed)] = {
            "baseline": reference,
            "v142_override": reference,
        }
        selected_seed_points = _safe_archive_expansion_select_branch_points(
            seed_points,
            branch_index_include=selected_branch_indexes,
        )
        for point in selected_seed_points:
            if _safe_archive_expansion_wall_budget_exhausted(start, max_wall_seconds):
                stop_reason = "max_wall_seconds_elapsed_before_branch_point"
                break
            result, source = _safe_archive_expansion_evaluate_branch_point_checkpointed(
                point,
                reference_runs=reference_runs,
                max_candidate_actions=int(max_candidate_actions),
                verify_replay=bool(verify_replay),
                chunk_dir=chunk_dir,
                resumed_branch_results=resumed,
            )
            branch_results.append(result)
            if source == "resumed":
                resumed_count += 1
            else:
                generated_count += 1
            _safe_archive_expansion_emit_progress(
                progress_callback,
                {
                    "event": "branch_result",
                    "source": source,
                    "fixture": "carrion_only",
                    "seed": int(point.seed),
                    "branch_id": point.branch_id,
                    "branch_point_index": int(point.branch_index),
                    "action_count": len(_list_of_mappings(result.get("action_runs"))),
                    "elapsed_seconds": _safe_archive_expansion_elapsed_seconds(start),
                },
            )
        if stop_reason is not None:
            break
    selected_branch_point_count = sum(
        1 for point in branch_points if int(point.branch_index) in set(selected_branch_indexes)
    )
    generation_status = _safe_archive_expansion_fixture_generation_status(
        fixture="carrion_only",
        stop_reason=stop_reason,
        started_at=start,
        max_wall_seconds=max_wall_seconds,
        branch_point_count=selected_branch_point_count,
        branch_result_count=len(branch_results),
        generated_branch_result_count=generated_count,
        resumed_branch_result_count=resumed_count,
        chunk_dir=chunk_dir,
    )
    if _safe_archive_expansion_status_partial(generation_status):
        _safe_archive_expansion_emit_progress(
            progress_callback,
            {
                "event": "partial_stop",
                "fixture": "carrion_only",
                "seed": None,
                "branch_id": None,
                "branch_point_index": len(branch_results),
                "action_count": 0,
                "elapsed_seconds": _safe_archive_expansion_elapsed_seconds(start),
                "stop_reason": stop_reason,
            },
        )
    first_missing = _first_missing_valid_action(branch_results)
    replay_items = [
        _mapping(run.get("replay_verification"))
        for result in branch_results
        for run in _list_of_mappings(result.get("action_runs"))
        if run.get("replay_verification") is not None
    ]
    replay_failure_count = sum(
        1 for item in replay_items if item.get("verified") is not True
    )
    source_failures: list[str] = []
    if failures:
        source_failures.append("carrion_branch_point_materialization_failed")
    if selected_branch_point_count <= 0:
        source_failures.append("no_carrion_branch_points")
    if first_missing is not None:
        source_failures.append("not_all_valid_candidate_actions_evaluated")
    if bool(verify_replay) and replay_failure_count > 0:
        source_failures.append("carrion_branch_replay_verification_failed")
    if _safe_archive_expansion_status_partial(generation_status):
        source_failures.append("partial_branch_evidence")
    return {
        "policy": "m3_safe_archive_expansion_carrion_branch_evidence_v1",
        "fixture": "carrion_only",
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "max_candidate_actions": (
            None if int(max_candidate_actions) == 0 else int(max_candidate_actions)
        ),
        "branch_index_selection": [int(index) for index in selected_branch_indexes],
        "materialization": {
            "policy": "m3_safe_archive_expansion_carrion_materialization_v1",
            "seed_reports": seed_reports,
            "materialized_branch_point_count": len(branch_points),
            "branch_point_count": selected_branch_point_count,
            "branch_points_by_seed": dict(
                sorted(
                    Counter(
                        point.seed
                        for point in branch_points
                        if int(point.branch_index) in set(selected_branch_indexes)
                    ).items()
                )
            ),
            "failure_count": len(failures),
            "failures": failures[:24],
            "passed": not failures and bool(branch_points),
        },
        "source_integrity": {
            "policy": "m3_safe_archive_expansion_carrion_source_integrity_v1",
            "passed": not source_failures,
            "failures": sorted(set(source_failures)),
            "first_missing_valid_action": first_missing,
            "replay_verification_count": len(replay_items),
            "replay_verification_failure_count": int(replay_failure_count),
        },
        "generation_status": generation_status,
        "branch_points": [
            _safe_archive_expansion_branch_point_payload(point)
            for point in branch_points
            if int(point.branch_index) in set(selected_branch_indexes)
        ],
        "branch_results": branch_results,
        "branch_result_count": len(branch_results),
    }


def _materialize_safe_archive_broad_points(
    *,
    seed: int,
    ticks: int,
    max_branch_points: int,
) -> tuple[list[BroadRegressionBranchPoint], dict[str, object], dict[str, object]]:
    world = SimulationWorld(
        WorldConfig(seed=int(seed), max_ticks=int(ticks)),
        policy=MindV3EvolutionPolicy(seed=int(seed)),
    )
    _configure_branch_manual_summary_run(world)
    points: list[BroadRegressionBranchPoint] = []
    failures: list[dict[str, object]] = []
    record_index = 0
    ticks_executed = 0
    for tick in range(int(ticks)):
        world.tick = tick
        snapshot = deepcopy(world)
        world._run_tick()
        ticks_executed = tick + 1
        for record in list(world.tick_trajectory_records):
            if len(points) >= int(max_branch_points):
                break
            action_mask = _bool_action_mask(record.get("action_mask"))
            if not _safe_archive_expansion_broad_record_candidate(
                record,
                action_mask=action_mask,
            ):
                record_index += 1
                continue
            requested = str(record.get("requested_action", ""))
            if requested not in ACTION_NAMES:
                failures.append(
                    {
                        "seed": int(seed),
                        "tick": int(tick),
                        "record_index": int(record_index),
                        "reason": "missing_current_policy_requested_action",
                    }
                )
                record_index += 1
                continue
            observation_input = _mapping(record.get("observation_input"))
            if not observation_input:
                failures.append(
                    {
                        "seed": int(seed),
                        "tick": int(tick),
                        "record_index": int(record_index),
                        "reason": "missing_public_observation_input",
                    }
                )
                record_index += 1
                continue
            agent_id = _int(record.get("agent_id"))
            branch_index = len(points)
            branch_id = (
                f"m3-safe-archive-broad-regression-seed-{seed}-branch-"
                f"{branch_index}-tick-{tick}-agent-{agent_id}"
            )
            branch_snapshot = deepcopy(snapshot)
            points.append(
                BroadRegressionBranchPoint(
                    branch_id=branch_id,
                    seed=int(seed),
                    fixture="broad",
                    ticks=int(ticks),
                    branch_tick=int(tick),
                    record_index=int(record_index),
                    branch_index=int(branch_index),
                    agent_id=int(agent_id),
                    baseline_action=requested,
                    v142_requested_action=requested,
                    v142_resolved_action=str(record.get("resolved_action", "")),
                    action_mask=action_mask,
                    observation_input=dict(observation_input),
                    observation_schema=_optional_string(record.get("observation_schema")),
                    observation_digest=_optional_string(record.get("observation_digest")),
                    source_trajectory_path=None,
                    branch_state_digest=_branch_state_digest(
                        snapshot,
                        branch_id=branch_id,
                        branch_tick=int(tick),
                    ),
                    world=branch_snapshot,
                )
            )
            record_index += 1
        if len(points) >= int(max_branch_points):
            break
        if not world.alive_agents():
            break
    while world.alive_agents() and ticks_executed < int(ticks):
        world.tick = ticks_executed
        world._run_tick()
        ticks_executed += 1
    reference = _safe_archive_expansion_reference_from_world(
        world,
        seed=int(seed),
        ticks=int(ticks),
        runtime="linear_mind_v3_broad_regression_seed",
    )
    return (
        points,
        reference,
        {
            "seed": int(seed),
            "fixture": "broad",
            "ticks_requested": int(ticks),
            "ticks_executed": int(ticks_executed),
            "branch_point_count": len(points),
            "branch_ids": [point.branch_id for point in points],
            "failure_count": len(failures),
            "failures": failures[:12],
            "passed": not failures and bool(points),
        },
    )


def _safe_archive_expansion_broad_record_candidate(
    record: Mapping[str, object],
    *,
    action_mask: Mapping[str, bool],
) -> bool:
    if not action_mask:
        return False
    if _int(record.get("agent_id"), default=-1) < 0:
        return False
    return any(bool(action_mask.get(action)) for action in ACTION_NAMES)


def _materialize_safe_archive_carrion_points(
    *,
    seed: int,
    ticks: int,
    max_branch_points: int,
) -> tuple[list[BroadRegressionBranchPoint], dict[str, object], dict[str, object]]:
    world = evaluate_cli._fixture_world(
        fixture_name="carrion_only",
        seed=int(seed),
        ticks=int(ticks),
        policy=MindV3EvolutionPolicy(seed=int(seed)),
    )
    _configure_branch_manual_summary_run(world)
    points: list[BroadRegressionBranchPoint] = []
    failures: list[dict[str, object]] = []
    record_index = 0
    ticks_executed = 0
    for tick in range(int(ticks)):
        world.tick = tick
        snapshot = deepcopy(world)
        world._run_tick()
        ticks_executed = tick + 1
        for record in list(world.tick_trajectory_records):
            if len(points) >= int(max_branch_points):
                break
            action_mask = _bool_action_mask(record.get("action_mask"))
            if not _safe_archive_expansion_carrion_record_candidate(
                record,
                action_mask=action_mask,
            ):
                record_index += 1
                continue
            requested = str(record.get("requested_action", ""))
            if requested not in ACTION_NAMES:
                failures.append(
                    {
                        "seed": int(seed),
                        "tick": int(tick),
                        "record_index": int(record_index),
                        "reason": "missing_current_policy_requested_action",
                    }
                )
                record_index += 1
                continue
            observation_input = _mapping(record.get("observation_input"))
            if not observation_input:
                failures.append(
                    {
                        "seed": int(seed),
                        "tick": int(tick),
                        "record_index": int(record_index),
                        "reason": "missing_public_observation_input",
                    }
                )
                record_index += 1
                continue
            agent_id = _int(record.get("agent_id"))
            branch_index = len(points)
            branch_id = (
                f"m3-safe-archive-carrion-seed-{seed}-branch-{branch_index}-"
                f"tick-{tick}-agent-{agent_id}"
            )
            branch_snapshot = deepcopy(snapshot)
            points.append(
                BroadRegressionBranchPoint(
                    branch_id=branch_id,
                    seed=int(seed),
                    fixture="carrion_only",
                    ticks=int(ticks),
                    branch_tick=int(tick),
                    record_index=int(record_index),
                    branch_index=int(branch_index),
                    agent_id=int(agent_id),
                    baseline_action=requested,
                    v142_requested_action=requested,
                    v142_resolved_action=str(record.get("resolved_action", "")),
                    action_mask=action_mask,
                    observation_input=dict(observation_input),
                    observation_schema=_optional_string(record.get("observation_schema")),
                    observation_digest=_optional_string(record.get("observation_digest")),
                    source_trajectory_path=None,
                    branch_state_digest=_branch_state_digest(
                        snapshot,
                        branch_id=branch_id,
                        branch_tick=int(tick),
                    ),
                    world=branch_snapshot,
                )
            )
            record_index += 1
        if len(points) >= int(max_branch_points):
            break
        if not world.alive_agents():
            break
    while world.alive_agents() and ticks_executed < int(ticks):
        world.tick = ticks_executed
        world._run_tick()
        ticks_executed += 1
    reference = _safe_archive_expansion_reference_from_world(
        world,
        seed=int(seed),
        ticks=int(ticks),
        runtime="linear_mind_v3_carrion_fixture",
    )
    return (
        points,
        reference,
        {
            "seed": int(seed),
            "fixture": "carrion_only",
            "ticks_requested": int(ticks),
            "ticks_executed": int(ticks_executed),
            "branch_point_count": len(points),
            "branch_ids": [point.branch_id for point in points],
            "failure_count": len(failures),
            "failures": failures[:12],
            "passed": not failures and bool(points),
        },
    )


def _safe_archive_expansion_carrion_record_candidate(
    record: Mapping[str, object],
    *,
    action_mask: Mapping[str, bool],
) -> bool:
    if not action_mask:
        return False
    if bool(action_mask.get("eat")):
        return True
    outcome = _mapping(record.get("outcome"))
    feeding = _mapping(outcome.get("feeding"))
    return str(feeding.get("food_source")) in {"carcass", "fresh_kill"}


def _safe_archive_expansion_reference_from_world(
    world: SimulationWorld,
    *,
    seed: int,
    ticks: int,
    runtime: str,
) -> dict[str, object]:
    summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)
    action_source_counts = Counter(
        str(record.get("action_source", "unknown"))
        for record in world.trajectory_records
    )
    requested_action_counts = Counter(
        str(record.get("requested_action"))
        for record in world.trajectory_records
        if isinstance(record.get("requested_action"), str)
    )
    resolved_action_counts = Counter(
        str(record.get("resolved_action"))
        for record in world.trajectory_records
        if isinstance(record.get("resolved_action"), str)
    )
    unsupported_requested_action_count = sum(
        1
        for record in world.trajectory_records
        if isinstance(record.get("requested_action"), str)
        and record.get("action_valid") is False
    )
    unsupported_resolved_action_count = sum(
        1
        for record in world.trajectory_records
        if isinstance(record.get("resolved_action"), str)
        and record.get("resolution_action_valid") is False
    )
    return {
        "seed": int(seed),
        "ticks": int(ticks),
        "runtime": runtime,
        "source_trajectory_path": None,
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "trajectory_record_count": len(world.trajectory_records),
        "heuristic_action_source_count": evaluate_cli._heuristic_action_source_count(
            action_source_counts
        ),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "unsupported_requested_action_count": int(unsupported_requested_action_count),
        "unsupported_resolved_action_count": int(unsupported_resolved_action_count),
        "target_terminal_by_agent": _target_terminal_by_agent_from_records(
            world.trajectory_records
        ),
    }


def _safe_archive_expansion_branch_point_payload(
    point: BroadRegressionBranchPoint,
) -> dict[str, object]:
    valid_actions = _v143_candidate_actions(point, max_candidate_actions=0)
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture,
        "ticks": point.ticks,
        "branch_tick": point.branch_tick,
        "record_index": point.record_index,
        "branch_index": point.branch_index,
        "agent_id": point.agent_id,
        "baseline_action": point.baseline_action,
        "valid_action_count": len(valid_actions),
        "valid_actions": valid_actions,
        "branch_state_digest": point.branch_state_digest,
        "public_features": {
            "observation_input": point.observation_input,
            "action_mask": dict(sorted(point.action_mask.items())),
        },
    }


def _safe_archive_expansion_evidence_coverage(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts = Counter(str(result.get("fixture", "unknown")) for result in branch_results)
    by_seed_fixture = Counter(
        f"{result.get('fixture', 'unknown')}:{_int(result.get('seed'))}"
        for result in branch_results
    )
    action_run_count = sum(
        len(_list_of_mappings(result.get("action_runs"))) for result in branch_results
    )
    return {
        "branch_result_count": len(branch_results),
        "branch_point_counts": dict(sorted(counts.items())),
        "branch_points_by_fixture_seed": dict(sorted(by_seed_fixture.items())),
        "action_run_count": int(action_run_count),
        "first_missing_valid_action": _first_missing_valid_action(branch_results),
    }


def build_candidate_specs(
    *,
    archive: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]] = (),
    mode: str,
    candidate_limit: int | None,
    v144_report: Mapping[str, object] | None,
    v144_artifact_path: str | Path,
    output_dir: Path,
    keep_trajectories: bool,
) -> list[dict[str, object]]:
    specs = []
    arms = list(CAMPAIGN_ARMS)
    if mode == "smoke":
        arms = [
            "linear_control",
            "v144_branch_intervention_residual",
            "v146_blacklist_safe_branch_residual",
            "safe_exact_support",
            "safe_action_conditioned_support",
            "safe_public_history_support",
            "neural_offline",
        ]
    if candidate_limit is not None:
        arms = arms[: max(0, int(candidate_limit))]
    for order, arm in enumerate(arms):
        spec = {
            "candidate_id": arm,
            "order": order,
            "mode": mode,
            "output_dir": str(output_dir / arm),
            "keep_trajectories": bool(keep_trajectories),
            "requires_shadow": arm != "linear_control",
            "artifact": None,
            "skip_reason": None,
        }
        if arm == "linear_control":
            specs.append(spec)
            continue
        if arm == "v144_branch_intervention_residual":
            artifact = _load_candidate_artifact(v144_artifact_path)
            if artifact is None:
                spec["skip_reason"] = "missing_v144_artifact"
            else:
                spec["artifact"] = artifact
                spec["source_report_classification"] = (
                    _mapping(_mapping(v144_report or {}).get("classification")).get(
                        "primary"
                    )
                )
            specs.append(spec)
            continue
        if arm in SUPPORT_GATED_ARMS:
            if archive.get("archive_support_sufficient") is not True:
                spec["skip_reason"] = "archive_support_insufficient_safe_label_count_lt_20"
            else:
                spec["artifact"] = _build_support_artifact_from_safe_rows(
                    archive=archive,
                    dataset_rows=dataset_rows,
                    candidate_id=arm,
                )
            specs.append(spec)
            continue
        if arm == "neural_offline":
            if _int(archive.get("safe_label_count")) < _int(
                archive.get("min_safe_label_count")
            ):
                spec["skip_reason"] = "safe_label_count_too_small_for_neural_offline_arm"
            else:
                spec["skip_reason"] = "neural_offline_arm_not_configured_for_local_campaign"
            specs.append(spec)
    return specs


def evaluate_candidate_specs(
    *,
    specs: Sequence[Mapping[str, object]],
    baseline: Mapping[str, object],
    workers: int,
) -> list[dict[str, object]]:
    if int(workers) <= 1:
        results = [_evaluate_candidate_worker(dict(spec), dict(baseline)) for spec in specs]
    else:
        results_by_order: dict[int, dict[str, object]] = {}
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {
                executor.submit(_evaluate_candidate_worker, dict(spec), dict(baseline)): _int(
                    spec.get("order")
                )
                for spec in specs
            }
            for future in as_completed(futures):
                result = future.result()
                results_by_order[_int(result.get("order"))] = result
        results = [results_by_order[index] for index in sorted(results_by_order)]
    return results


def _evaluate_candidate_worker(
    spec: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    candidate_id = str(spec.get("candidate_id"))
    output_dir = Path(str(spec.get("output_dir", "")))
    keep_trajectories = bool(spec.get("keep_trajectories"))
    skip_reason = spec.get("skip_reason")
    if isinstance(skip_reason, str) and skip_reason:
        return _skipped_result(spec=spec, baseline=baseline, reason=skip_reason)
    if candidate_id == "linear_control":
        return _linear_control_result(spec=spec, baseline=baseline)
    artifact = _mapping(spec.get("artifact"))
    if not artifact:
        return _skipped_result(
            spec=spec,
            baseline=baseline,
            reason="missing_candidate_artifact",
        )
    validate_support_gated_residual_artifact(artifact)
    shadow = _run_shadow_support_check(
        artifact=artifact,
        keep_trajectories=keep_trajectories,
        output_dir=output_dir / "shadow",
    )
    if shadow["shadow_gate"]["passed"] is not True:
        return _candidate_result(
            spec=spec,
            baseline=baseline,
            artifact=artifact,
            shadow=shadow,
            broad=None,
            carrion=None,
            skipped=False,
        )
    broad_runs = [
        _run_policy(
            seed=seed,
            ticks=STRICT_TICKS,
            artifact=artifact,
            runtime_mode="live",
            fixture=None,
            trajectory_path=(
                output_dir / "trajectories" / f"broad-live-seed-{seed}.jsonl.gz"
                if keep_trajectories
                else None
            ),
        )
        for seed in STRICT_BROAD_SEEDS
    ]
    carrion_runs = [
        _run_policy(
            seed=seed,
            ticks=STRICT_TICKS,
            artifact=artifact,
            runtime_mode="live",
            fixture="carrion_only",
            trajectory_path=(
                output_dir / "trajectories" / f"carrion-live-seed-{seed}.jsonl.gz"
                if keep_trajectories
                else None
            ),
        )
        for seed in STRICT_CARRION_FIXTURE_SEEDS
    ]
    broad = _comparison_against_baseline(
        fixture="broad",
        baseline_runs=_list_of_mappings(_mapping(baseline.get("broad")).get("runs")),
        candidate_runs=broad_runs,
    )
    carrion = _comparison_against_baseline(
        fixture="carrion_only",
        baseline_runs=_list_of_mappings(
            _mapping(baseline.get("carrion_only")).get("runs")
        ),
        candidate_runs=carrion_runs,
    )
    return _candidate_result(
        spec=spec,
        baseline=baseline,
        artifact=artifact,
        shadow=shadow,
        broad=broad,
        carrion=carrion,
        skipped=False,
    )


def _run_shadow_support_check(
    *,
    artifact: Mapping[str, object],
    keep_trajectories: bool,
    output_dir: Path,
) -> dict[str, object]:
    runs = [
        _run_policy(
            seed=seed,
            ticks=STRICT_TICKS,
            artifact=artifact,
            runtime_mode="shadow",
            fixture=None,
            trajectory_path=(
                output_dir / "trajectories" / f"broad-shadow-seed-{seed}.jsonl.gz"
                if keep_trajectories
                else None
            ),
        )
        for seed in STRICT_BROAD_SEEDS
    ]
    aggregate = _aggregate_runs(runs)
    diagnostics = _mapping(aggregate.get("support_residual_diagnostics"))
    blockers: list[dict[str, object]] = []
    _add_floor_blocker(
        blockers,
        name="zero_unsupported_requested_action_count",
        passed=_int(aggregate.get("unsupported_requested_action_count")) == 0,
        observed=aggregate.get("unsupported_requested_action_count"),
        required=0,
        fixture="broad",
    )
    _add_floor_blocker(
        blockers,
        name="zero_unsupported_proposed_action_count",
        passed=_int(diagnostics.get("unsupported_proposed_action_count")) == 0,
        observed=diagnostics.get("unsupported_proposed_action_count"),
        required=0,
        fixture="broad",
    )
    _add_floor_blocker(
        blockers,
        name="shadow_applied_support_nonzero",
        passed=_int(diagnostics.get("gate_accepted_override_count")) > 0,
        observed=diagnostics.get("gate_accepted_override_count"),
        required="> 0",
        fixture="broad",
    )
    _add_floor_blocker(
        blockers,
        name="shadow_dominant_gate_accepted_share_lte_0_50",
        passed=_float(diagnostics.get("dominant_gate_accepted_override_action_share"))
        <= 0.50,
        observed=diagnostics.get("dominant_gate_accepted_override_action_share"),
        required=0.50,
        fixture="broad",
    )
    return {
        "policy": "v146_shadow_support_check_v1",
        "runs": runs,
        "aggregate": aggregate,
        "shadow_gate": {
            "passed": not blockers,
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
    }


def _candidate_result(
    *,
    spec: Mapping[str, object],
    baseline: Mapping[str, object],
    artifact: Mapping[str, object],
    shadow: Mapping[str, object],
    broad: Mapping[str, object] | None,
    carrion: Mapping[str, object] | None,
    skipped: bool,
) -> dict[str, object]:
    floors = _candidate_floors(shadow=shadow, broad=broad, carrion=carrion)
    failed = [floor for floor in floors if floor.get("passed") is not True]
    first_failed = failed[0] if failed else None
    first_failed_floor = _mapping(first_failed).get("name")
    first_failing_seed = _mapping(first_failed).get("seed")
    if first_failing_seed is None and isinstance(first_failed_floor, str):
        first_failing_seed = _infer_first_failing_seed(
            first_failed_floor=first_failed_floor,
            broad=broad,
            carrion=carrion,
        )
    score = _ranking_metrics(floors=floors, broad=broad, carrion=carrion)
    return {
        "schema_version": "mind_v3_v146_candidate_result_v1",
        "candidate_id": spec.get("candidate_id"),
        "order": _int(spec.get("order")),
        "candidate_family": _candidate_family(str(spec.get("candidate_id"))),
        "is_control": False,
        "candidate_evaluation_role": "candidate",
        "control_passed": False,
        "skipped": skipped,
        "accepted": not failed and not skipped,
        "promotion_authorized": False,
        "artifact_digest": stable_payload_digest(artifact),
        "artifact_schema_version": artifact.get("schema_version"),
        "shadow": _shadow_summary(shadow),
        "broad": broad,
        "carrion_only": carrion,
        "floors": floors,
        "failed_floor_count": len(failed),
        "first_failed_floor": first_failed_floor,
        "first_failing_seed": first_failing_seed,
        "first_failing_fixture": _mapping(first_failed).get("fixture"),
        "first_failed_action_distribution": _first_failed_action_distribution(
            broad=broad,
            carrion=carrion,
        ),
        "ranking_metrics": score,
        "baseline_digest": stable_payload_digest(baseline),
    }


def _linear_control_result(
    *,
    spec: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    broad = _comparison_against_baseline(
        fixture="broad",
        baseline_runs=_list_of_mappings(_mapping(baseline.get("broad")).get("runs")),
        candidate_runs=_list_of_mappings(_mapping(baseline.get("broad")).get("runs")),
    )
    carrion = _comparison_against_baseline(
        fixture="carrion_only",
        baseline_runs=_list_of_mappings(
            _mapping(baseline.get("carrion_only")).get("runs")
        ),
        candidate_runs=_list_of_mappings(
            _mapping(baseline.get("carrion_only")).get("runs")
        ),
    )
    floors = _candidate_floors(shadow=None, broad=broad, carrion=carrion)
    floors = [
        _floor(
            "linear_control_baseline_cached",
            True,
            observed="baseline_vs_baseline",
            required="control",
        )
    ]
    return {
        "schema_version": "mind_v3_v146_candidate_result_v1",
        "candidate_id": spec.get("candidate_id"),
        "order": _int(spec.get("order")),
        "candidate_family": "linear_control",
        "is_control": True,
        "candidate_evaluation_role": "control",
        "control_passed": True,
        "skipped": False,
        "accepted": False,
        "promotion_authorized": False,
        "artifact_digest": None,
        "artifact_schema_version": None,
        "shadow": None,
        "broad": broad,
        "carrion_only": carrion,
        "floors": floors,
        "failed_floor_count": 0,
        "first_failed_floor": None,
        "first_failing_seed": None,
        "first_failing_fixture": None,
        "first_failed_action_distribution": _first_failed_action_distribution(
            broad=broad,
            carrion=carrion,
        ),
        "ranking_metrics": {
            "accepted_rank_value": 0,
            "failed_floor_count": 0,
            "broad_per_seed_regression_count": 0,
            "carrion_improved": False,
            "resolved_invalid_delta": 0,
            "dominant_requested_action_share": _round(
                max(
                    _float(
                        _mapping(
                            _mapping(broad.get("candidate")).get("aggregate")
                        ).get("dominant_requested_action_share")
                    ),
                    _float(
                        _mapping(
                            _mapping(carrion.get("candidate")).get("aggregate")
                        ).get("dominant_requested_action_share")
                    ),
                )
            ),
        },
        "baseline_digest": stable_payload_digest(baseline),
    }


def _skipped_result(
    *,
    spec: Mapping[str, object],
    baseline: Mapping[str, object],
    reason: str,
) -> dict[str, object]:
    floor = {
        "name": "candidate_evaluated",
        "passed": False,
        "observed": "skipped",
        "required": "evaluated",
        "reason": reason,
        "fixture": None,
        "seed": None,
    }
    return {
        "schema_version": "mind_v3_v146_candidate_result_v1",
        "candidate_id": spec.get("candidate_id"),
        "order": _int(spec.get("order")),
        "candidate_family": _candidate_family(str(spec.get("candidate_id"))),
        "is_control": False,
        "candidate_evaluation_role": "candidate",
        "control_passed": False,
        "skipped": True,
        "skip_reason": reason,
        "accepted": False,
        "promotion_authorized": False,
        "artifact_digest": None,
        "artifact_schema_version": None,
        "shadow": None,
        "broad": None,
        "carrion_only": None,
        "floors": [floor],
        "failed_floor_count": 1,
        "first_failed_floor": floor["name"],
        "first_failing_seed": None,
        "first_failing_fixture": None,
        "first_failed_action_distribution": {},
        "ranking_metrics": {
            "accepted_rank_value": 0,
            "failed_floor_count": 1,
            "broad_per_seed_regression_count": 999,
            "carrion_improved": False,
            "resolved_invalid_delta": 999,
            "dominant_requested_action_share": 1.0,
        },
        "baseline_digest": stable_payload_digest(baseline),
    }


def _candidate_floors(
    *,
    shadow: Mapping[str, object] | None,
    broad: Mapping[str, object] | None,
    carrion: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    floors: list[dict[str, object]] = []
    if shadow is not None:
        gate = _mapping(shadow.get("shadow_gate"))
        floors.append(
            _floor(
                "shadow_support_action_floors_passed",
                gate.get("passed") is True,
                observed=gate.get("blockers"),
                required=[],
                fixture="broad",
            )
        )
        if gate.get("passed") is not True:
            return floors
    if broad is None or carrion is None:
        return floors
    broad_candidate = _mapping(_mapping(broad.get("candidate")).get("aggregate"))
    carrion_candidate = _mapping(_mapping(carrion.get("candidate")).get("aggregate"))
    broad_baseline = _mapping(_mapping(broad.get("baseline")).get("aggregate"))
    carrion_baseline = _mapping(_mapping(carrion.get("baseline")).get("aggregate"))
    broad_diag = _mapping(broad_candidate.get("support_residual_diagnostics"))
    carrion_diag = _mapping(carrion_candidate.get("support_residual_diagnostics"))
    total_heuristic = _int(broad_candidate.get("heuristic_action_source_count")) + _int(
        carrion_candidate.get("heuristic_action_source_count")
    )
    total_unsupported_requested = _int(
        broad_candidate.get("unsupported_requested_action_count")
    ) + _int(carrion_candidate.get("unsupported_requested_action_count"))
    total_unsupported_proposed = _int(
        broad_diag.get("unsupported_proposed_action_count")
    ) + _int(carrion_diag.get("unsupported_proposed_action_count"))
    max_dominant_share = max(
        _float(broad_candidate.get("dominant_requested_action_share")),
        _float(carrion_candidate.get("dominant_requested_action_share")),
    )
    floors.extend(
        [
            _floor(
                "zero_heuristic_action_source_count",
                total_heuristic == 0,
                observed=total_heuristic,
                required=0,
            ),
            _floor(
                "zero_unsupported_requested_action_count",
                total_unsupported_requested == 0,
                observed=total_unsupported_requested,
                required=0,
            ),
            _floor(
                "zero_unsupported_proposed_action_count",
                total_unsupported_proposed == 0,
                observed=total_unsupported_proposed,
                required=0,
            ),
            _floor(
                "broad_resolved_invalid_not_increased",
                _int(broad_candidate.get("resolved_invalid_action_count"))
                <= _int(broad_baseline.get("resolved_invalid_action_count")),
                observed={
                    "baseline": broad_baseline.get("resolved_invalid_action_count"),
                    "candidate": broad_candidate.get("resolved_invalid_action_count"),
                },
                required="candidate <= baseline",
                fixture="broad",
            ),
            _floor(
                "carrion_resolved_invalid_not_increased",
                _int(carrion_candidate.get("resolved_invalid_action_count"))
                <= _int(carrion_baseline.get("resolved_invalid_action_count")),
                observed={
                    "baseline": carrion_baseline.get("resolved_invalid_action_count"),
                    "candidate": carrion_candidate.get("resolved_invalid_action_count"),
                },
                required="candidate <= baseline",
                fixture="carrion_only",
            ),
            _floor(
                "dominant_requested_action_share_lte_0_50",
                max_dominant_share <= 0.50,
                observed=max_dominant_share,
                required=0.50,
            ),
        ]
    )
    for row in _list_of_mappings(broad.get("per_seed_delta")):
        seed = _int(row.get("seed"))
        floors.append(
            _floor(
                f"broad_seed_{seed}_alive_not_regressed",
                _int(row.get("alive_delta")) >= 0,
                observed=row.get("alive_delta"),
                required=">= 0",
                fixture="broad",
                seed=seed,
            )
        )
        floors.append(
            _floor(
                f"broad_seed_{seed}_births_not_regressed",
                _int(row.get("births_delta")) >= 0,
                observed=row.get("births_delta"),
                required=">= 0",
                fixture="broad",
                seed=seed,
            )
        )
    carrion_delta = _mapping(carrion.get("aggregate_delta"))
    baseline_blockers = _fixture_blocker_count(carrion.get("baseline_fixture_gate"))
    candidate_blockers = _fixture_blocker_count(carrion.get("candidate_fixture_gate"))
    carrion_improved = _float(carrion_delta.get("alive_agents_mean")) > 0.0 or (
        candidate_blockers < baseline_blockers
    )
    floors.append(
        _floor(
            "carrion_alive_or_blocker_improved",
            carrion_improved,
            observed={
                "alive_delta": carrion_delta.get("alive_agents_mean"),
                "baseline_blockers": baseline_blockers,
                "candidate_blockers": candidate_blockers,
            },
            required="alive_delta > 0 or candidate_blockers < baseline_blockers",
            fixture="carrion_only",
        )
    )
    return floors


def _comparison_against_baseline(
    *,
    fixture: str,
    baseline_runs: Sequence[Mapping[str, object]],
    candidate_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline_aggregate = _aggregate_runs(baseline_runs)
    candidate_aggregate = _aggregate_runs(candidate_runs)
    payload = {
        "fixture": fixture,
        "baseline": {
            "runs": [dict(run) for run in baseline_runs],
            "aggregate": baseline_aggregate,
        },
        "candidate": {
            "runs": [dict(run) for run in candidate_runs],
            "aggregate": candidate_aggregate,
        },
        "aggregate_delta": _aggregate_delta(
            baseline=baseline_aggregate,
            candidate=candidate_aggregate,
        ),
        "per_seed_delta": _per_seed_delta(
            baseline_runs=baseline_runs,
            candidate_runs=candidate_runs,
        ),
    }
    if fixture == "carrion_only":
        fixture_config = evaluate_cli.mind_v3_fixture_gate_config(
            suite="basic",
            seeds=list(STRICT_CARRION_FIXTURE_SEEDS),
            ticks=STRICT_TICKS,
            min_alive=1.0,
            min_births=0.0,
            min_mixed_stable_births=0.0,
            min_energy_viability=0.0,
            min_hydration_viability=0.0,
            min_health_viability=0.0,
            min_matched_diet_viability=0.0,
            min_biologically_ready=0.0,
        )
        payload["fixture_config"] = fixture_config
        payload["baseline_fixture_gate"] = _fixture_gate_from_runs(
            runs=baseline_runs,
            fixture_config=fixture_config,
            policy_name="linear_mind_v3",
        )
        payload["candidate_fixture_gate"] = _fixture_gate_from_runs(
            runs=candidate_runs,
            fixture_config=fixture_config,
            policy_name="candidate_mind_v3",
        )
    return payload


def _run_policy(
    *,
    seed: int,
    ticks: int,
    artifact: Mapping[str, object] | None,
    runtime_mode: str,
    fixture: str | None,
    trajectory_path: Path | None,
) -> dict[str, object]:
    policy = (
        MindV3EvolutionPolicy(seed=int(seed))
        if artifact is None
        else MindV3EvolutionPolicy(
            seed=int(seed),
            support_residual_artifact=artifact,
            support_residual_runtime_mode=runtime_mode,
        )
    )
    if fixture == "carrion_only":
        world = evaluate_cli._fixture_world(
            fixture_name="carrion_only",
            seed=int(seed),
            ticks=int(ticks),
            policy=policy,
        )
    else:
        world = SimulationWorld(
            WorldConfig(seed=int(seed), max_ticks=int(ticks)),
            policy=policy,
        )
    run = evaluate_cli._run_world(
        world=world,
        seed=int(seed),
        ticks=int(ticks),
        trajectory_output_path=trajectory_path,
        trajectory_split_id="mind_v3_v146_candidate_campaign",
    )
    diagnostics = support_gated_residual_runtime_diagnostics(
        world.policy_decision_diagnostics_records
    )
    run["support_residual_diagnostics"] = diagnostics
    run["unsupported_proposed_action_count"] = int(
        diagnostics.get("unsupported_proposed_action_count", 0)
    )
    run["resolved_invalid_action_count"] = int(
        run.get("unsupported_resolved_action_count", 0)
    )
    if fixture:
        run["fixture"] = fixture
    return run


def _aggregate_runs(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    aggregate = evaluate_cli._aggregate_runs([dict(run) for run in runs])
    diagnostics = aggregate_support_gated_residual_runtime_diagnostics(runs)
    aggregate["support_residual_diagnostics"] = diagnostics
    aggregate["unsupported_proposed_action_count"] = int(
        diagnostics.get("unsupported_proposed_action_count", 0)
    )
    aggregate["resolved_invalid_action_count"] = int(
        aggregate.get("unsupported_resolved_action_count", 0)
    )
    return aggregate


def _aggregate_delta(
    *,
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    candidate_diag = _mapping(candidate.get("support_residual_diagnostics"))
    return {
        "alive_agents_mean": _round(
            _float(candidate.get("alive_agents_mean"))
            - _float(baseline.get("alive_agents_mean"))
        ),
        "births_mean": _round(
            _float(candidate.get("births_mean"))
            - _float(baseline.get("births_mean"))
        ),
        "deaths_mean": _round(
            _float(candidate.get("deaths_mean")) - _float(baseline.get("deaths_mean"))
        ),
        "resolved_invalid_action_count_delta": _int(
            candidate.get("resolved_invalid_action_count")
        )
        - _int(baseline.get("resolved_invalid_action_count")),
        "unsupported_requested_action_count_delta": _int(
            candidate.get("unsupported_requested_action_count")
        )
        - _int(baseline.get("unsupported_requested_action_count")),
        "heuristic_action_source_count_delta": _int(
            candidate.get("heuristic_action_source_count")
        )
        - _int(baseline.get("heuristic_action_source_count")),
        "dominant_requested_action_share_delta": _round(
            _float(candidate.get("dominant_requested_action_share"))
            - _float(baseline.get("dominant_requested_action_share"))
        ),
        "applied_override_count": _int(candidate_diag.get("applied_override_count")),
        "applied_override_action_counts": dict(
            sorted(_mapping(candidate_diag.get("applied_override_action_counts")).items())
        ),
    }


def _per_seed_delta(
    *,
    baseline_runs: Sequence[Mapping[str, object]],
    candidate_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline_by_seed = {_int(run.get("seed")): run for run in baseline_runs}
    rows = []
    for candidate in sorted(candidate_runs, key=lambda item: _int(item.get("seed"))):
        seed = _int(candidate.get("seed"))
        baseline = _mapping(baseline_by_seed.get(seed))
        diagnostics = _mapping(candidate.get("support_residual_diagnostics"))
        rows.append(
            {
                "seed": seed,
                "baseline_alive_agents": _int(baseline.get("alive_agents")),
                "candidate_alive_agents": _int(candidate.get("alive_agents")),
                "alive_delta": _int(candidate.get("alive_agents"))
                - _int(baseline.get("alive_agents")),
                "baseline_births": _int(baseline.get("births")),
                "candidate_births": _int(candidate.get("births")),
                "births_delta": _int(candidate.get("births"))
                - _int(baseline.get("births")),
                "resolved_invalid_action_count_delta": _int(
                    candidate.get("resolved_invalid_action_count")
                )
                - _int(baseline.get("resolved_invalid_action_count")),
                "unsupported_requested_action_count_delta": _int(
                    candidate.get("unsupported_requested_action_count")
                )
                - _int(baseline.get("unsupported_requested_action_count")),
                "unsupported_proposed_action_count": _int(
                    diagnostics.get("unsupported_proposed_action_count")
                ),
                "applied_override_count": _int(
                    diagnostics.get("applied_override_count")
                ),
                "applied_override_action_counts": dict(
                    sorted(
                        _mapping(
                            diagnostics.get("applied_override_action_counts")
                        ).items()
                    )
                ),
            }
        )
    return rows


def rank_candidate_results(
    results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    candidates = [result for result in results if not _is_control_result(result)]
    ranked = sorted(
        candidates,
        key=lambda result: (
            -_int(_mapping(result.get("ranking_metrics")).get("accepted_rank_value")),
            _int(result.get("failed_floor_count")),
            _int(
                _mapping(result.get("ranking_metrics")).get(
                    "broad_per_seed_regression_count"
                )
            ),
            -_bool_int(_mapping(result.get("ranking_metrics")).get("carrion_improved")),
            _int(_mapping(result.get("ranking_metrics")).get("resolved_invalid_delta")),
            _float(
                _mapping(result.get("ranking_metrics")).get(
                    "dominant_requested_action_share"
                )
            ),
            _int(result.get("order")),
        ),
    )
    return [
        {
            "rank": index + 1,
            "candidate_id": result.get("candidate_id"),
            "candidate_evaluation_role": result.get(
                "candidate_evaluation_role", "candidate"
            ),
            "accepted": result.get("accepted"),
            "skipped": result.get("skipped"),
            "failed_floor_count": result.get("failed_floor_count"),
            "first_failed_floor": result.get("first_failed_floor"),
            "first_failing_seed": result.get("first_failing_seed"),
            "first_failing_fixture": result.get("first_failing_fixture"),
            "ranking_metrics": result.get("ranking_metrics"),
        }
        for index, result in enumerate(ranked)
    ]


def _is_control_result(result: Mapping[str, object]) -> bool:
    return (
        result.get("is_control") is True
        or result.get("candidate_evaluation_role") == "control"
        or result.get("candidate_id") == "linear_control"
    )


def write_campaign_ledger(
    ledger_path: str | Path,
    report: Mapping[str, object],
) -> None:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for candidate in (
            *_list_of_mappings(report.get("controls")),
            *_list_of_mappings(report.get("candidates")),
        ):
            row = {
                "schema_version": MIND_V3_V146_CANDIDATE_CAMPAIGN_LEDGER_SCHEMA_VERSION,
                "campaign_schema_version": report.get("schema_version"),
                "candidate_id": candidate.get("candidate_id"),
                "is_control": candidate.get("is_control") is True,
                "candidate_evaluation_role": candidate.get(
                    "candidate_evaluation_role"
                ),
                "control_passed": candidate.get("control_passed") is True,
                "accepted": candidate.get("accepted"),
                "skipped": candidate.get("skipped"),
                "skip_reason": candidate.get("skip_reason"),
                "failed_floor_count": candidate.get("failed_floor_count"),
                "first_failed_floor": candidate.get("first_failed_floor"),
                "first_failing_seed": candidate.get("first_failing_seed"),
                "first_failing_fixture": candidate.get("first_failing_fixture"),
                "resolved_invalid_delta": _mapping(
                    candidate.get("ranking_metrics")
                ).get("resolved_invalid_delta"),
                "dominant_requested_action_share": _mapping(
                    candidate.get("ranking_metrics")
                ).get("dominant_requested_action_share"),
                "promotion_authorized": False,
            }
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
            handle.write("\n")


def _ranking_metrics(
    *,
    floors: Sequence[Mapping[str, object]],
    broad: Mapping[str, object] | None,
    carrion: Mapping[str, object] | None,
) -> dict[str, object]:
    failed_count = sum(1 for floor in floors if floor.get("passed") is not True)
    broad_regressions = 0
    resolved_delta = 999
    dominant_share = 1.0
    if broad is not None:
        broad_regressions = sum(
            1
            for row in _list_of_mappings(broad.get("per_seed_delta"))
            if _int(row.get("alive_delta")) < 0 or _int(row.get("births_delta")) < 0
        )
        resolved_delta = _int(
            _mapping(broad.get("aggregate_delta")).get(
                "resolved_invalid_action_count_delta"
            )
        )
        dominant_share = _float(
            _mapping(_mapping(broad.get("candidate")).get("aggregate")).get(
                "dominant_requested_action_share"
            )
        )
    carrion_improved = False
    if carrion is not None:
        delta = _mapping(carrion.get("aggregate_delta"))
        carrion_improved = _float(delta.get("alive_agents_mean")) > 0.0 or (
            _fixture_blocker_count(carrion.get("candidate_fixture_gate"))
            < _fixture_blocker_count(carrion.get("baseline_fixture_gate"))
        )
        dominant_share = max(
            dominant_share,
            _float(
                _mapping(_mapping(carrion.get("candidate")).get("aggregate")).get(
                    "dominant_requested_action_share"
                )
            ),
        )
    return {
        "accepted_rank_value": 1 if failed_count == 0 else 0,
        "failed_floor_count": failed_count,
        "broad_per_seed_regression_count": broad_regressions,
        "carrion_improved": carrion_improved,
        "resolved_invalid_delta": resolved_delta,
        "dominant_requested_action_share": _round(dominant_share),
    }


def _stop_rules(
    *,
    results: Sequence[Mapping[str, object]],
    archive: Mapping[str, object],
) -> dict[str, object]:
    support_gated = [
        result
        for result in results
        if result.get("candidate_family") == "support_gated_residual"
        and result.get("skipped") is not True
    ]
    all_support_failed = bool(support_gated) and all(
        result.get("accepted") is not True for result in support_gated
    )
    public_history = next(
        (result for result in results if result.get("candidate_id") == "safe_public_history_support"),
        None,
    )
    exact = next(
        (result for result in results if result.get("candidate_id") == "safe_exact_support"),
        None,
    )
    public_history_beats_exact = (
        public_history is not None
        and exact is not None
        and _int(public_history.get("failed_floor_count"))
        < _int(exact.get("failed_floor_count"))
    )
    carrion_override_total = 0
    for result in results:
        carrion = _mapping(result.get("carrion_only"))
        candidate = _mapping(_mapping(carrion.get("candidate")).get("aggregate"))
        diagnostics = _mapping(candidate.get("support_residual_diagnostics"))
        carrion_override_total += _int(diagnostics.get("applied_override_count"))
    return {
        "policy": "v146_candidate_campaign_stop_rules_v1",
        "safe_label_count": archive.get("safe_label_count"),
        "min_safe_label_count": archive.get("min_safe_label_count"),
        "archive_support_insufficient": archive.get("archive_support_sufficient")
        is not True,
        "all_support_gated_arms_failed": all_support_failed,
        "close_exact_support_residual_family": all_support_failed,
        "public_history_support_beats_exact_support": public_history_beats_exact,
        "next_route_option_memory_public_history_controller": public_history_beats_exact,
        "carrion_applied_override_count": carrion_override_total,
        "next_route_carrion_specific_archive_expansion": carrion_override_total == 0,
    }


def _campaign_classification(stop_rules: Mapping[str, object]) -> dict[str, object]:
    if stop_rules.get("archive_support_insufficient") is True:
        primary = "candidate_campaign_archive_support_insufficient"
    elif stop_rules.get("close_exact_support_residual_family") is True:
        primary = "candidate_campaign_closes_exact_support_residual_family"
    elif stop_rules.get("next_route_option_memory_public_history_controller") is True:
        primary = "candidate_campaign_routes_to_public_history_controller"
    elif stop_rules.get("next_route_carrion_specific_archive_expansion") is True:
        primary = "candidate_campaign_routes_to_carrion_archive_expansion"
    else:
        primary = "candidate_campaign_candidate_family_ready_for_more_work"
    return {"primary": primary, "labels": [primary]}


def _build_support_artifact_from_safe_rows(
    *,
    archive: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    candidate_id: str,
) -> dict[str, object] | None:
    rows = _list_of_mappings(archive.get("safe_rows"))
    if not rows:
        return None
    support_examples = []
    for index, row in enumerate(rows):
        row_index = _int(row.get("row_index"), default=-1)
        source_row = (
            dataset_rows[row_index]
            if 0 <= row_index < len(dataset_rows)
            else {}
        )
        trainable = _mapping(_mapping(source_row).get("trainable"))
        features = _mapping(trainable.get("features"))
        label = _mapping(trainable.get("label"))
        action = str(label.get("action", row.get("label_action") or ""))
        if not action:
            continue
        runtime_row = planner_distilled_runtime_row(
            observation_input=_mapping(features.get("observation_input")),
            action_mask=_mapping(features.get("action_mask")),
            public_history_trace=[],
        )
        vector = candidate_feature_vector(runtime_row, action)
        if vector:
            support_examples.append(
                {
                    "example_index": index,
                    "action": action,
                    "mode": "v146_safe_support",
                    "feature_vector": [_round(value) for value in vector],
                    "weight": 1.0,
                }
            )
    if not support_examples:
        return None
    counts = Counter(str(example["action"]) for example in support_examples)
    artifact = {
        "schema_version": MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
        "policy": MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
        "default_action_policy": "linear_mind_v3",
        "runtime_ready": True,
        "promotion_ready": False,
        "runtime_promotion_allowed": False,
        "training_policy": f"v146_{candidate_id}_non_promotable_campaign_training_v1",
        "scorer_rule": "action_prior_balanced_nearest_support_v1",
        "action_prior_penalty_scale": V103_ACTION_PRIOR_BALANCE_PENALTY,
        "support_gate": {
            "nearest_support_distance_threshold": 0.0,
            "residual_score_margin_threshold": 0.0,
        },
        "inference_contract": _support_artifact_contract(),
        "training_row_count": len(support_examples),
        "support_action_counts": dict(sorted(counts.items())),
        "teacher_action_counts": dict(sorted(counts.items())),
        "support_examples": support_examples,
    }
    validate_support_gated_residual_artifact(artifact)
    return artifact


def load_safe_archive_expansion_dataset(path: str | Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise CandidateCampaignError(
                f"safe archive dataset row {line_number} must be a JSON object"
            )
        rows.append(payload)
    return rows


def validate_safe_archive_train_eval_inputs(
    *,
    safe_archive_report: Mapping[str, object],
    safe_archive_dataset_rows: Sequence[Mapping[str, object]],
    branch_evidence_report: Mapping[str, object],
    expected_report_classification: str = (
        "m3_safe_archive_expansion_support_ready_no_training_run"
    ),
    leakage_strict_heldout_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    expected_safe_label_count: int = 25,
    expected_min_safe_label_count: int = DEFAULT_MIN_SAFE_LABEL_COUNT,
    expected_dataset_digest: str = BP3_SAFE_ARCHIVE_DATASET_DIGEST,
    expected_branch_evidence_digest: str = BP3_SAFE_ARCHIVE_BRANCH_EVIDENCE_DIGEST,
) -> dict[str, object]:
    classification = _mapping(safe_archive_report.get("classification")).get("primary")
    source_integrity = _mapping(safe_archive_report.get("source_integrity"))
    dataset = _mapping(safe_archive_report.get("dataset"))
    evidence_integrity = _mapping(branch_evidence_report.get("source_integrity"))
    evidence_status = _mapping(branch_evidence_report.get("generation_status"))
    dataset_digest = stable_payload_digest(safe_archive_dataset_rows)
    failures: list[str] = []
    if classification != expected_report_classification:
        failures.append("safe_archive_report_not_support_ready")
    if source_integrity.get("passed") is not True:
        failures.append("safe_archive_report_source_integrity_failed")
    if list(source_integrity.get("failures") or []) != []:
        failures.append("safe_archive_report_source_integrity_failures_present")
    if _int(dataset.get("safe_label_count")) != int(expected_safe_label_count):
        failures.append("safe_archive_dataset_safe_label_count_mismatch")
    if _int(dataset.get("min_safe_label_count")) != int(expected_min_safe_label_count):
        failures.append("safe_archive_dataset_min_safe_label_count_mismatch")
    if len(safe_archive_dataset_rows) != int(expected_safe_label_count):
        failures.append("safe_archive_dataset_row_count_mismatch")
    if dataset.get("dataset_digest") != expected_dataset_digest:
        failures.append("safe_archive_report_dataset_digest_mismatch")
    if dataset_digest != expected_dataset_digest:
        failures.append("safe_archive_dataset_digest_mismatch")
    if branch_evidence_report.get("branch_evidence_digest") != expected_branch_evidence_digest:
        failures.append("branch_evidence_digest_mismatch")
    if evidence_status.get("state") != "complete" or evidence_status.get("partial") is True:
        failures.append("branch_evidence_not_complete")
    if evidence_integrity.get("passed") is not True:
        failures.append("branch_evidence_source_integrity_failed")
    if list(evidence_integrity.get("failures") or []) != []:
        failures.append("branch_evidence_source_integrity_failures_present")
    if safe_archive_report.get("training_authorized") is not False:
        failures.append("safe_archive_report_training_authorized_not_false")
    if safe_archive_report.get("promotion_authorized") is not False:
        failures.append("safe_archive_report_promotion_authorized_not_false")
    if branch_evidence_report.get("training_authorized") is not False:
        failures.append("branch_evidence_training_authorized_not_false")
    if branch_evidence_report.get("promotion_authorized") is not False:
        failures.append("branch_evidence_promotion_authorized_not_false")
    leakage_scan = safe_archive_expansion_leakage_scan(
        safe_archive_dataset_rows,
        strict_heldout_seeds=leakage_strict_heldout_seeds,
    )
    if leakage_scan.get("passed") is not True:
        failures.append("safe_archive_trainable_leakage_scan_failed")
    if failures:
        raise CandidateCampaignError(
            "safe archive train/eval input validation failed: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "policy": "m3_safe_archive_train_eval_input_validation_v1",
        "passed": True,
        "failures": [],
        "expected_report_classification": expected_report_classification,
        "expected_safe_label_count": int(expected_safe_label_count),
        "expected_min_safe_label_count": int(expected_min_safe_label_count),
        "dataset_digest": dataset_digest,
        "branch_evidence_digest": branch_evidence_report.get(
            "branch_evidence_digest"
        ),
        "trainable_leakage_scan": leakage_scan,
    }


def build_safe_archive_train_eval_preflight(
    *,
    safe_archive_dataset_rows: Sequence[Mapping[str, object]],
    branch_evidence_report: Mapping[str, object],
    strict_broad_seeds: Sequence[int] = SAFE_ARCHIVE_TRAIN_EVAL_BROAD_SEEDS,
    strict_carrion_seeds: Sequence[int] = SAFE_ARCHIVE_TRAIN_EVAL_CARRION_SEEDS,
    leakage_strict_heldout_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
) -> dict[str, object]:
    label_counts: Counter[tuple[str, str, int]] = Counter()
    action_counts: Counter[str] = Counter()
    fixture_seed_counts: Counter[tuple[str, int]] = Counter()
    branch_counts: Counter[str] = Counter()
    trainable_counts: Counter[str] = Counter()
    for row in safe_archive_dataset_rows:
        metadata = _mapping(row.get("metadata"))
        trainable = _mapping(row.get("trainable"))
        action = str(_mapping(trainable.get("label")).get("action", ""))
        fixture = str(metadata.get("fixture", "unknown"))
        seed = _int(metadata.get("seed"))
        if action:
            label_counts.update([(fixture, action, seed)])
            action_counts.update([action])
        fixture_seed_counts.update([(fixture, seed)])
        branch_id = str(metadata.get("branch_id", ""))
        if branch_id:
            branch_counts.update([branch_id])
        trainable_counts.update([stable_payload_digest(trainable)])
    zero_support = []
    for seed in strict_broad_seeds:
        if fixture_seed_counts.get(("broad", int(seed)), 0) <= 0:
            zero_support.append({"fixture": "broad", "seed": int(seed)})
    for seed in strict_carrion_seeds:
        if fixture_seed_counts.get(("carrion_only", int(seed)), 0) <= 0:
            zero_support.append({"fixture": "carrion_only", "seed": int(seed)})
    branch_results = _list_of_mappings(branch_evidence_report.get("branch_results"))
    all_action_run_count = 0
    replay_verified_count = 0
    replay_missing_count = 0
    replay_failed_count = 0
    incomplete_action_branch_count = 0
    first_missing = None
    for index, result in enumerate(branch_results):
        if not _all_valid_actions_evaluated(result):
            incomplete_action_branch_count += 1
            if first_missing is None:
                first_missing = _missing_valid_action_detail(index, result)
        for run in _list_of_mappings(result.get("action_runs")):
            all_action_run_count += 1
            replay = run.get("replay_verification")
            if replay is None:
                replay_missing_count += 1
                continue
            replay_payload = _mapping(replay)
            if replay_payload.get("verified") is True:
                replay_verified_count += 1
            else:
                replay_failed_count += 1
    dominant = _dominant_count_share(action_counts)
    leakage_scan = safe_archive_expansion_leakage_scan(
        safe_archive_dataset_rows,
        strict_heldout_seeds=leakage_strict_heldout_seeds,
    )
    return {
        "policy": "m3_safe_archive_train_eval_adversarial_preflight_v1",
        "safe_label_count": len(safe_archive_dataset_rows),
        "safe_labels_by_fixture_action_seed": [
            {
                "fixture": fixture,
                "action": action,
                "seed": seed,
                "count": count,
            }
            for (fixture, action, seed), count in sorted(
                label_counts.items(),
                key=lambda item: (item[0][0], _action_order(item[0][1]), item[0][2]),
            )
        ],
        "safe_labels_by_fixture_seed": [
            {"fixture": fixture, "seed": seed, "count": count}
            for (fixture, seed), count in sorted(fixture_seed_counts.items())
        ],
        "safe_label_action_counts": dict(sorted(action_counts.items())),
        "zero_support_strict_seeds": zero_support,
        "broad_seed_41_absent_from_archive_support": (
            fixture_seed_counts.get(("broad", 41), 0) <= 0
        ),
        "carrion_seed_13_safe_label_absent": (
            fixture_seed_counts.get(("carrion_only", 13), 0) <= 0
        ),
        "dominant_label_action": dominant["key"],
        "dominant_label_action_share": dominant["share"],
        "trainable_leakage_scan": leakage_scan,
        "duplicate_trainable_row_count": sum(
            count - 1 for count in trainable_counts.values() if count > 1
        ),
        "duplicate_branch_id_count": sum(
            count - 1 for count in branch_counts.values() if count > 1
        ),
        "candidate_action_coverage": {
            "complete": incomplete_action_branch_count == 0,
            "branch_result_count": len(branch_results),
            "incomplete_branch_result_count": incomplete_action_branch_count,
            "first_missing_valid_action": first_missing,
        },
        "replay_verification": {
            "complete": (
                all_action_run_count > 0
                and replay_verified_count == all_action_run_count
                and replay_missing_count == 0
                and replay_failed_count == 0
            ),
            "action_run_count": all_action_run_count,
            "verified_count": replay_verified_count,
            "missing_count": replay_missing_count,
            "failed_count": replay_failed_count,
        },
    }


def build_safe_archive_diagnostic_artifact(
    *,
    safe_archive_dataset_rows: Sequence[Mapping[str, object]],
    candidate_id: str = "bp3_safe_archive_support_gated_residual",
    support_mode: str = "bp3_safe_archive_support",
    training_source_label: str = "bp3_safe_archive",
) -> dict[str, object]:
    support_examples = []
    for index, row in enumerate(safe_archive_dataset_rows):
        trainable = _mapping(row.get("trainable"))
        features = _mapping(trainable.get("features"))
        label = _mapping(trainable.get("label"))
        action = str(label.get("action", ""))
        if action not in ACTION_NAMES:
            continue
        runtime_row = planner_distilled_runtime_row(
            observation_input=_mapping(features.get("observation_input")),
            action_mask=_mapping(features.get("action_mask")),
            public_history_trace=[],
        )
        vector = candidate_feature_vector(runtime_row, action)
        if not vector:
            continue
        support_examples.append(
                {
                    "example_index": index,
                    "action": action,
                    "mode": support_mode,
                    "feature_vector": [_round(value) for value in vector],
                    "weight": 1.0,
                }
        )
    if not support_examples:
        raise CandidateCampaignError(
            "safe archive diagnostic artifact has no support examples"
        )
    counts = Counter(str(example["action"]) for example in support_examples)
    artifact = {
        "schema_version": MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
        "policy": MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
        "default_action_policy": "linear_mind_v3",
        "runtime_ready": True,
        "promotion_ready": False,
        "runtime_promotion_allowed": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "diagnostics_only": True,
        "default_runtime_behavior_changed": False,
        "training_policy": (
            f"{candidate_id}_diagnostic_training_from_{training_source_label}_v1"
        ),
        "scorer_rule": "action_prior_balanced_nearest_support_v1",
        "action_prior_penalty_scale": V103_ACTION_PRIOR_BALANCE_PENALTY,
        "support_gate": {
            "nearest_support_distance_threshold": 0.0,
            "residual_score_margin_threshold": 0.0,
        },
        "inference_contract": _support_artifact_contract(),
        "training_row_count": len(support_examples),
        "support_action_counts": dict(sorted(counts.items())),
        "teacher_action_counts": dict(sorted(counts.items())),
        "support_examples": support_examples,
    }
    validate_support_gated_residual_artifact(artifact)
    return artifact


def safe_archive_diagnostic_training_summary(
    *,
    artifact: Mapping[str, object],
    safe_archive_dataset_rows: Sequence[Mapping[str, object]],
    branch_evidence_report: Mapping[str, object],
) -> dict[str, object]:
    branch_by_id = {
        str(result.get("branch_id", "")): result
        for result in _list_of_mappings(branch_evidence_report.get("branch_results"))
    }
    correct = 0
    gate_passed_count = 0
    override_allowed_count = 0
    rows = []
    for index, row in enumerate(safe_archive_dataset_rows):
        trainable = _mapping(row.get("trainable"))
        features = _mapping(trainable.get("features"))
        label_action = str(_mapping(trainable.get("label")).get("action", ""))
        metadata = _mapping(row.get("metadata"))
        branch = _mapping(branch_by_id.get(str(metadata.get("branch_id", ""))))
        linear_action = str(branch.get("baseline_action", "stay"))
        scored = score_support_gated_residual_artifact(
            artifact=artifact,
            observation_input=_mapping(features.get("observation_input")),
            action_mask=_mapping(features.get("action_mask")),
            linear_action=linear_action,
        )
        selected = scored.get("selected_action")
        selected_correct = selected == label_action
        if selected_correct:
            correct += 1
        if scored.get("support_gate_passed") is True:
            gate_passed_count += 1
        if scored.get("override_allowed") is True:
            override_allowed_count += 1
        rows.append(
            {
                "row_index": index,
                "label_action": label_action,
                "selected_action": selected,
                "selected_action_correct": selected_correct,
                "support_gate_passed": scored.get("support_gate_passed"),
                "override_allowed": scored.get("override_allowed"),
                "nearest_support_distance": scored.get("nearest_support_distance"),
                "score_margin": scored.get("score_margin"),
            }
        )
    row_count = len(safe_archive_dataset_rows)
    return {
        "policy": "m3_safe_archive_support_gated_training_summary_v1",
        "diagnostic_training_ran": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "row_count": row_count,
        "selected_action_accuracy_count": correct,
        "selected_action_accuracy": _safe_rate(correct, row_count),
        "support_gate_passed_count": gate_passed_count,
        "override_allowed_count": override_allowed_count,
        "accuracy_is_success_criterion": False,
        "rows": rows[:64],
    }


def run_safe_archive_train_eval(
    *,
    safe_archive_report_path: str | Path,
    safe_archive_dataset_path: str | Path,
    branch_evidence_report_path: str | Path,
    artifact_output_path: str | Path,
    output_path: str | Path,
    run_evaluation: bool = True,
    expected_report_classification: str = (
        "m3_safe_archive_expansion_support_ready_no_training_run"
    ),
    leakage_strict_heldout_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    expected_safe_label_count: int = 25,
    expected_min_safe_label_count: int = DEFAULT_MIN_SAFE_LABEL_COUNT,
    expected_dataset_digest: str = BP3_SAFE_ARCHIVE_DATASET_DIGEST,
    expected_branch_evidence_digest: str = BP3_SAFE_ARCHIVE_BRANCH_EVIDENCE_DIGEST,
    candidate_id: str = "bp3_safe_archive_support_gated_residual",
    support_mode: str = "bp3_safe_archive_support",
    training_source_label: str = "bp3_safe_archive",
    report_schema_version: str = M3_SAFE_ARCHIVE_TRAIN_EVAL_REPORT_SCHEMA_VERSION,
    report_policy: str = M3_SAFE_ARCHIVE_TRAIN_EVAL_POLICY,
    require_live_carrion_override: bool = False,
) -> dict[str, object]:
    safe_archive_report = load_json_report(safe_archive_report_path)
    branch_evidence_report = load_json_report(branch_evidence_report_path)
    dataset_rows = load_safe_archive_expansion_dataset(safe_archive_dataset_path)
    validation = validate_safe_archive_train_eval_inputs(
        safe_archive_report=safe_archive_report,
        safe_archive_dataset_rows=dataset_rows,
        branch_evidence_report=branch_evidence_report,
        expected_report_classification=expected_report_classification,
        leakage_strict_heldout_seeds=leakage_strict_heldout_seeds,
        expected_safe_label_count=expected_safe_label_count,
        expected_min_safe_label_count=expected_min_safe_label_count,
        expected_dataset_digest=expected_dataset_digest,
        expected_branch_evidence_digest=expected_branch_evidence_digest,
    )
    preflight = build_safe_archive_train_eval_preflight(
        safe_archive_dataset_rows=dataset_rows,
        branch_evidence_report=branch_evidence_report,
        leakage_strict_heldout_seeds=leakage_strict_heldout_seeds,
    )
    artifact = build_safe_archive_diagnostic_artifact(
        safe_archive_dataset_rows=dataset_rows,
        candidate_id=candidate_id,
        support_mode=support_mode,
        training_source_label=training_source_label,
    )
    artifact_path = Path(artifact_output_path)
    write_json(artifact_path, artifact)
    runtime_artifact = load_support_gated_residual_artifact(artifact_path)
    if stable_payload_digest(runtime_artifact) != stable_payload_digest(artifact):
        raise CandidateCampaignError(
            "safe archive diagnostic artifact changed during serialization round trip"
        )
    training = safe_archive_diagnostic_training_summary(
        artifact=runtime_artifact,
        safe_archive_dataset_rows=dataset_rows,
        branch_evidence_report=branch_evidence_report,
    )
    evaluation: dict[str, object]
    acceptance: dict[str, object]
    if run_evaluation:
        run_dir = Path(output_path).with_suffix("")
        baseline = build_linear_baseline_cache(
            broad_seeds=SAFE_ARCHIVE_TRAIN_EVAL_BROAD_SEEDS,
            ticks=SAFE_ARCHIVE_TRAIN_EVAL_TICKS,
            fixture_seeds=SAFE_ARCHIVE_TRAIN_EVAL_CARRION_SEEDS,
            fixture_ticks=SAFE_ARCHIVE_TRAIN_EVAL_TICKS,
            keep_trajectories=False,
            output_dir=run_dir / "linear_control",
        )
        broad_runs = [
            _run_policy(
                seed=seed,
                ticks=SAFE_ARCHIVE_TRAIN_EVAL_TICKS,
                artifact=runtime_artifact,
                runtime_mode="live",
                fixture=None,
                trajectory_path=None,
            )
            for seed in SAFE_ARCHIVE_TRAIN_EVAL_BROAD_SEEDS
        ]
        carrion_runs = [
            _run_policy(
                seed=seed,
                ticks=SAFE_ARCHIVE_TRAIN_EVAL_TICKS,
                artifact=runtime_artifact,
                runtime_mode="live",
                fixture="carrion_only",
                trajectory_path=None,
            )
            for seed in SAFE_ARCHIVE_TRAIN_EVAL_CARRION_SEEDS
        ]
        broad = _comparison_against_baseline(
            fixture="broad",
            baseline_runs=_list_of_mappings(_mapping(baseline.get("broad")).get("runs")),
            candidate_runs=broad_runs,
        )
        carrion = _comparison_against_baseline(
            fixture="carrion_only",
            baseline_runs=_list_of_mappings(
                _mapping(baseline.get("carrion_only")).get("runs")
            ),
            candidate_runs=carrion_runs,
        )
        evaluation = {
            "baseline": _baseline_summary(baseline),
            "broad": broad,
            "carrion_only": carrion,
        }
        acceptance = build_safe_archive_train_eval_acceptance(
            broad=broad,
            carrion=carrion,
            preflight=preflight,
            require_live_carrion_override=require_live_carrion_override,
        )
    else:
        evaluation = {"skipped": True, "reason": "run_evaluation_false"}
        acceptance = {
            "policy": "m3_safe_archive_train_eval_acceptance_v1",
            "passed": False,
            "blockers": [
                {
                    "reason": "evaluation_not_run",
                    "fixture": None,
                    "seed": None,
                    "action": None,
                    "observed": "skipped",
                    "required": "live broad and carrion evaluation",
                }
            ],
            "first_failing_seed": None,
            "first_failing_fixture": None,
            "first_failing_action": None,
        }
    classification = (
        "m3_safe_archive_diagnostic_passed_non_promotional"
        if acceptance.get("passed") is True
        else "m3_safe_archive_diagnostic_failed_non_promotional"
    )
    report = {
        "schema_version": report_schema_version,
        "policy": report_policy,
        "contract": {
            "diagnostics_only": True,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "gate_relaxation": False,
            "live_ab_enabled_by_default": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_seed_fixture_branch_path_digest_provenance_as_trainable_input": False,
        },
        "inputs": {
            "safe_archive_report": str(safe_archive_report_path),
            "safe_archive_dataset": str(safe_archive_dataset_path),
            "branch_evidence_report": str(branch_evidence_report_path),
            "artifact_output": str(artifact_output_path),
            "expected_report_classification": expected_report_classification,
            "leakage_strict_heldout_seeds": [
                int(seed) for seed in leakage_strict_heldout_seeds
            ],
            "expected_dataset_digest": expected_dataset_digest,
            "expected_branch_evidence_digest": expected_branch_evidence_digest,
            "candidate_id": candidate_id,
            "support_mode": support_mode,
            "training_source_label": training_source_label,
            "require_live_carrion_override": require_live_carrion_override,
        },
        "validation": validation,
        "preflight": preflight,
        "artifact": {
            "path": str(artifact_output_path),
            "digest": stable_payload_digest(artifact),
            "roundtrip_load_passed": True,
            "schema_version": artifact.get("schema_version"),
            "policy": artifact.get("policy"),
            "training_row_count": artifact.get("training_row_count"),
            "support_action_counts": artifact.get("support_action_counts"),
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
        },
        "training": training,
        "evaluation": evaluation,
        "acceptance": acceptance,
        "classification": {"primary": classification, "labels": [classification]},
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    write_json(output_path, report)
    return report


def build_safe_archive_train_eval_acceptance(
    *,
    broad: Mapping[str, object],
    carrion: Mapping[str, object],
    preflight: Mapping[str, object],
    require_live_carrion_override: bool = False,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    for row in _list_of_mappings(broad.get("per_seed_delta")):
        seed = _int(row.get("seed"))
        alive_delta = _int(row.get("alive_delta"))
        births_delta = _int(row.get("births_delta"))
        if alive_delta < 0:
            blockers.append(
                _diagnostic_blocker(
                    "broad_seed_alive_regression",
                    fixture="broad",
                    seed=seed,
                    action=None,
                    observed=alive_delta,
                    required=">= 0",
                )
            )
        if births_delta < 0:
            blockers.append(
                _diagnostic_blocker(
                    "broad_seed_birth_regression",
                    fixture="broad",
                    seed=seed,
                    action=None,
                    observed=births_delta,
                    required=">= 0",
                )
            )
        resolved_invalid_delta = _int(row.get("resolved_invalid_action_count_delta"))
        if resolved_invalid_delta > 0:
            blockers.append(
                _diagnostic_blocker(
                    "broad_seed_resolved_invalid_increase",
                    fixture="broad",
                    seed=seed,
                    action=None,
                    observed=resolved_invalid_delta,
                    required="<= 0",
                )
            )
        unsupported_requested_delta = _int(
            row.get("unsupported_requested_action_count_delta")
        )
        if unsupported_requested_delta > 0:
            blockers.append(
                _diagnostic_blocker(
                    "broad_seed_unsupported_requested_increase",
                    fixture="broad",
                    seed=seed,
                    action=None,
                    observed=unsupported_requested_delta,
                    required="<= 0",
                )
            )
    for row in _list_of_mappings(carrion.get("per_seed_delta")):
        seed = _int(row.get("seed"))
        alive_delta = _int(row.get("alive_delta"))
        births_delta = _int(row.get("births_delta"))
        if alive_delta < 0:
            blockers.append(
                _diagnostic_blocker(
                    "carrion_seed_alive_regression",
                    fixture="carrion_only",
                    seed=seed,
                    action=None,
                    observed=alive_delta,
                    required=">= 0",
                )
            )
        if births_delta < 0:
            blockers.append(
                _diagnostic_blocker(
                    "carrion_seed_birth_regression",
                    fixture="carrion_only",
                    seed=seed,
                    action=None,
                    observed=births_delta,
                    required=">= 0",
                )
            )
        resolved_invalid_delta = _int(row.get("resolved_invalid_action_count_delta"))
        if resolved_invalid_delta > 0:
            blockers.append(
                _diagnostic_blocker(
                    "carrion_seed_resolved_invalid_increase",
                    fixture="carrion_only",
                    seed=seed,
                    action=None,
                    observed=resolved_invalid_delta,
                    required="<= 0",
                )
            )
        unsupported_requested_delta = _int(
            row.get("unsupported_requested_action_count_delta")
        )
        if unsupported_requested_delta > 0:
            blockers.append(
                _diagnostic_blocker(
                    "carrion_seed_unsupported_requested_increase",
                    fixture="carrion_only",
                    seed=seed,
                    action=None,
                    observed=unsupported_requested_delta,
                    required="<= 0",
                )
            )
    broad_candidate = _mapping(_mapping(broad.get("candidate")).get("aggregate"))
    carrion_candidate = _mapping(_mapping(carrion.get("candidate")).get("aggregate"))
    broad_diag = _mapping(broad_candidate.get("support_residual_diagnostics"))
    carrion_diag = _mapping(carrion_candidate.get("support_residual_diagnostics"))
    broad_delta = _mapping(broad.get("aggregate_delta"))
    carrion_delta = _mapping(carrion.get("aggregate_delta"))
    requested_counts = _sum_counter(
        broad_candidate.get("requested_action_counts"),
        carrion_candidate.get("requested_action_counts"),
    )
    action_source_counts = _sum_counter(
        broad_candidate.get("action_source_counts"),
        carrion_candidate.get("action_source_counts"),
    )
    combined_dominant = _dominant_count_share(requested_counts)
    max_dominant_share = max(
        _float(broad_candidate.get("dominant_requested_action_share")),
        _float(carrion_candidate.get("dominant_requested_action_share")),
        _float(combined_dominant.get("share")),
    )
    heuristic_count = _int(broad_candidate.get("heuristic_action_source_count")) + _int(
        carrion_candidate.get("heuristic_action_source_count")
    )
    broad_applied_override_count = _int(broad_diag.get("applied_override_count"))
    carrion_applied_override_count = _int(carrion_diag.get("applied_override_count"))
    if max_dominant_share > MAX_DOMINANT_SAFE_LABEL_ACTION_SHARE:
        blockers.append(
            _diagnostic_blocker(
                "dominant_action_share_gt_0_50",
                fixture=None,
                seed=None,
                action=combined_dominant.get("key"),
                observed=max_dominant_share,
                required=MAX_DOMINANT_SAFE_LABEL_ACTION_SHARE,
            )
        )
    if heuristic_count > 0:
        blockers.append(
            _diagnostic_blocker(
                "heuristic_action_source_count_nonzero",
                fixture=None,
                seed=None,
                action=None,
                observed=heuristic_count,
                required=0,
            )
        )
    if require_live_carrion_override and carrion_applied_override_count <= 0:
        blockers.append(
            _diagnostic_blocker(
                "carrion_live_applied_override_count_zero",
                fixture="carrion_only",
                seed=None,
                action=None,
                observed=carrion_applied_override_count,
                required="> 0",
            )
        )
    carrion_gate = _mapping(carrion.get("candidate_fixture_gate"))
    if _float(carrion_delta.get("alive_agents_mean")) < 0.0:
        blockers.append(
            _diagnostic_blocker(
                "carrion_fixture_alive_regression",
                fixture="carrion_only",
                seed=None,
                action=None,
                observed=carrion_delta.get("alive_agents_mean"),
                required=">= 0",
            )
        )
    if _float(carrion_delta.get("births_mean")) < 0.0:
        blockers.append(
            _diagnostic_blocker(
                "carrion_fixture_birth_regression",
                fixture="carrion_only",
                seed=None,
                action=None,
                observed=carrion_delta.get("births_mean"),
                required=">= 0",
            )
        )
    if carrion_gate.get("passed") is not True:
        blockers.append(
            _diagnostic_blocker(
                "carrion_fixture_remains_blocked",
                fixture="carrion_only",
                seed=None,
                action=None,
                observed=carrion_gate.get("blockers"),
                required="candidate carrion fixture gate passed",
            )
        )
    first = blockers[0] if blockers else {}
    broad_seed_41 = _seed_delta_status(broad, seed=41)
    carrion_seed_13 = _seed_delta_status(carrion, seed=13)
    return {
        "policy": "m3_safe_archive_train_eval_acceptance_v1",
        "passed": not blockers,
        "blockers": blockers,
        "first_failing_seed": first.get("seed"),
        "first_failing_fixture": first.get("fixture"),
        "first_failing_action": first.get("action"),
        "metrics": {
            "broad_mean_alive_delta": broad_delta.get("alive_agents_mean"),
            "broad_mean_births_delta": broad_delta.get("births_mean"),
            "carrion_fixture_alive_delta": carrion_delta.get("alive_agents_mean"),
            "carrion_fixture_births_delta": carrion_delta.get("births_mean"),
            "dominant_requested_action": combined_dominant.get("key"),
            "dominant_requested_action_share": _round(max_dominant_share),
            "combined_dominant_requested_action_share": combined_dominant.get("share"),
            "heuristic_action_source_count": heuristic_count,
            "require_live_carrion_override": require_live_carrion_override,
            "broad_applied_override_count": broad_applied_override_count,
            "carrion_applied_override_count": carrion_applied_override_count,
            "combined_applied_override_count": (
                broad_applied_override_count + carrion_applied_override_count
            ),
            "action_source_counts": dict(sorted(action_source_counts.items())),
            "candidate_action_distribution": dict(sorted(requested_counts.items())),
            "broad_seed_41_absent_from_archive_support": preflight.get(
                "broad_seed_41_absent_from_archive_support"
            ),
            "broad_seed_41_behaves_well": broad_seed_41.get("behaves_well"),
            "broad_seed_41_delta": broad_seed_41,
            "carrion_seed_13_safe_label_absent": preflight.get(
                "carrion_seed_13_safe_label_absent"
            ),
            "carrion_seed_13_behaves_well": carrion_seed_13.get("behaves_well"),
            "carrion_seed_13_delta": carrion_seed_13,
            "broad_per_seed_delta": broad.get("per_seed_delta"),
            "carrion_per_seed_delta": carrion.get("per_seed_delta"),
            "carrion_fixture_gate_passed": carrion_gate.get("passed"),
            "carrion_fixture_gate_blockers": carrion_gate.get("blockers"),
        },
    }


def _diagnostic_blocker(
    reason: str,
    *,
    fixture: str | None,
    seed: int | None,
    action: object,
    observed: object,
    required: object,
) -> dict[str, object]:
    return {
        "reason": reason,
        "fixture": fixture,
        "seed": seed,
        "action": action,
        "observed": observed,
        "required": required,
    }


def _seed_delta_status(
    section: Mapping[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    row = next(
        (
            item
            for item in _list_of_mappings(section.get("per_seed_delta"))
            if _int(item.get("seed")) == int(seed)
        ),
        {},
    )
    alive_delta = _int(row.get("alive_delta"))
    births_delta = _int(row.get("births_delta"))
    return {
        "seed": int(seed),
        "alive_delta": alive_delta,
        "births_delta": births_delta,
        "behaves_well": bool(row) and alive_delta >= 0 and births_delta >= 0,
    }


def _sum_counter(*values: object) -> Counter[str]:
    counter: Counter[str] = Counter()
    for value in values:
        for key, count in _mapping(value).items():
            counter.update({str(key): _int(count)})
    return counter


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return _round(float(numerator) / float(denominator))


def _all_valid_actions_evaluated(result: Mapping[str, object]) -> bool:
    valid_actions = _valid_public_mask_actions(result)
    action_runs = _action_run_actions(result)
    return bool(valid_actions) and valid_actions.issubset(action_runs)


def _valid_public_mask_actions(result: Mapping[str, object]) -> set[str]:
    features = _mapping(result.get("public_features"))
    action_mask = _mapping(features.get("action_mask"))
    valid_actions = {
        action for action in ACTION_NAMES if bool(action_mask.get(action, False))
    }
    if not valid_actions:
        valid_actions = {
            str(action)
            for action in _list(result.get("candidate_actions"))
            if str(action) in ACTION_NAMES
        }
    return valid_actions


def _action_run_actions(result: Mapping[str, object]) -> set[str]:
    return {
        str(run.get("forced_action"))
        for run in _list_of_mappings(result.get("action_runs"))
        if str(run.get("forced_action")) in ACTION_NAMES
    }


def _missing_valid_action_detail(
    result_index: int,
    result: Mapping[str, object],
) -> dict[str, object] | None:
    valid_actions = _valid_public_mask_actions(result)
    missing = sorted(valid_actions - _action_run_actions(result), key=_action_order)
    if not missing:
        return None
    return {
        "source_branch_result_index": int(result_index),
        "branch_id": result.get("branch_id"),
        "seed": _int(result.get("seed")),
        "fixture": result.get("fixture"),
        "missing_action": missing[0],
        "missing_actions": missing,
    }


def _first_missing_valid_action(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    for index, result in enumerate(branch_results):
        missing = _missing_valid_action_detail(index, result)
        if missing is not None:
            return missing
    return None


def _bool_action_mask(value: object) -> dict[str, bool]:
    payload = _mapping(value)
    return {action: bool(payload.get(action, False)) for action in ACTION_NAMES}


def _action_order(action: str) -> int:
    return ACTION_NAMES.index(action) if action in ACTION_NAMES else len(ACTION_NAMES)


def _best_safe_expansion_action_run(
    result: Mapping[str, object],
) -> Mapping[str, object] | None:
    safe_runs = [
        run
        for run in _list_of_mappings(result.get("action_runs"))
        if _safe_archive_expansion_label_vet(run).get("passed") is True
    ]
    if not safe_runs:
        return None

    def key(run: Mapping[str, object]) -> tuple[object, ...]:
        baseline = _mapping(run.get("deltas_vs_baseline"))
        target = _mapping(run.get("target_terminal"))
        action = str(run.get("forced_action", ""))
        action_order = ACTION_NAMES.index(action) if action in ACTION_NAMES else 999
        return (
            _int(baseline.get("alive_agents")),
            _int(baseline.get("births")),
            -_int(baseline.get("deaths")),
            _int(baseline.get("target_alive")),
            _float(target.get("energy_ratio"), default=-1.0),
            _float(target.get("hydration_ratio"), default=-1.0),
            _float(target.get("health_ratio"), default=-1.0),
            -_int(run.get("unsupported_requested_action_count")),
            -_int(baseline.get("unsupported_resolved_action_count")),
            -action_order,
        )

    return max(safe_runs, key=key)


def _expanded_row_identity(
    result_index: int,
    result: Mapping[str, object],
    selected_run: Mapping[str, object],
) -> dict[str, object]:
    return {
        "row_index": int(result_index),
        "seed": _int(result.get("seed")),
        "fixture": result.get("fixture"),
        "branch_id": result.get("branch_id"),
        "branch_tick": _int(result.get("branch_tick")),
        "agent_id": _int(result.get("agent_id")),
        "label_action": selected_run.get("forced_action"),
    }


def _safe_archive_expansion_label_vet(
    run: Mapping[str, object],
) -> dict[str, object]:
    baseline = _mapping(run.get("deltas_vs_baseline"))
    replay = _mapping(run.get("replay_verification"))
    resolved_invalid_delta = baseline.get("unsupported_resolved_action_count")
    alive_delta = baseline.get("alive_agents")
    births_delta = baseline.get("births")
    floors = [
        _floor(
            "forced_action_was_used",
            run.get("forced_action_used") is True,
            observed=run.get("forced_action_used"),
            required=True,
        ),
        _floor(
            "forced_action_supported_by_public_mask",
            run.get("forced_action_supported") is True,
            observed=run.get("forced_action_supported"),
            required=True,
        ),
        _floor(
            "branch_replay_deterministic",
            bool(replay) and replay.get("verified") is True,
            observed=replay.get("verified") if replay else None,
            required=True,
        ),
        _floor(
            "no_unsupported_requested_actions",
            _int(run.get("unsupported_requested_action_count")) == 0,
            observed=run.get("unsupported_requested_action_count"),
            required=0,
        ),
        _floor(
            "no_resolved_invalid_increase",
            _is_finite_number(resolved_invalid_delta)
            and _float(resolved_invalid_delta) <= 0.0,
            observed=resolved_invalid_delta,
            required="finite numeric <= 0",
        ),
        _floor(
            "no_alive_regression",
            _is_finite_number(alive_delta) and _float(alive_delta) >= 0.0,
            observed=alive_delta,
            required="finite numeric >= 0",
        ),
        _floor(
            "no_birth_regression",
            _is_finite_number(births_delta) and _float(births_delta) >= 0.0,
            observed=births_delta,
            required="finite numeric >= 0",
        ),
        _floor(
            "zero_heuristic_action_source_count",
            _int(run.get("heuristic_action_source_count")) == 0,
            observed=run.get("heuristic_action_source_count"),
            required=0,
        ),
    ]
    failed = [floor for floor in floors if floor["passed"] is not True]
    return {
        "policy": "m3_safe_archive_expansion_label_safety_vet_v1",
        "passed": not failed,
        "failed_floor_count": len(failed),
        "floors": floors,
    }


def _safe_archive_expansion_dataset_row(
    *,
    row_index: int,
    source_branch_result_index: int,
    result: Mapping[str, object],
    selected_run: Mapping[str, object],
    safety_vet: Mapping[str, object],
) -> dict[str, object]:
    features = _mapping(result.get("public_features"))
    action = str(selected_run.get("forced_action", ""))
    return {
        "schema_version": M3_SAFE_ARCHIVE_EXPANSION_DATASET_ROW_SCHEMA_VERSION,
        "trainable": {
            "feature_policy": "public_observation_input_and_public_action_mask_v1",
            "features": {
                "observation_input": features.get("observation_input"),
                "action_mask": features.get("action_mask"),
            },
            "label": {
                "action": action,
                "label_policy": (
                    "safety_vetted_branch_replay_no_regression_label_v1"
                ),
            },
        },
        "metadata": {
            "row_index": int(row_index),
            "source_branch_result_index": int(source_branch_result_index),
            "seed": result.get("seed"),
            "fixture": result.get("fixture"),
            "branch_id": result.get("branch_id"),
            "branch_tick": result.get("branch_tick"),
            "record_index": result.get("record_index"),
            "agent_id": result.get("agent_id"),
            "source_trajectory_path": result.get("source_trajectory_path"),
            "branch_state_digest": result.get("branch_state_digest"),
            "replay_digest": selected_run.get("replay_digest"),
            "candidate_actions": result.get("candidate_actions"),
            "all_valid_actions_evaluated": _all_valid_actions_evaluated(result),
            "safety_vet": safety_vet,
            "outcome_evidence": {
                "deltas_vs_baseline": selected_run.get("deltas_vs_baseline"),
                "deltas_vs_v142_override": selected_run.get(
                    "deltas_vs_v142_override"
                ),
                "target_terminal": selected_run.get("target_terminal"),
                "replay_verification": selected_run.get("replay_verification"),
            },
            "provenance": {
                "policy": M3_SAFE_ARCHIVE_EXPANSION_POLICY,
                "public_feature_digest": stable_payload_digest(features),
            },
        },
    }


def safe_archive_expansion_leakage_scan(
    rows: Sequence[Mapping[str, object]],
    *,
    strict_heldout_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
) -> dict[str, object]:
    forbidden_failures: list[dict[str, object]] = []
    finite_failures: list[dict[str, object]] = []
    strict_seed_path_failures: list[dict[str, object]] = []
    strict_tokens = {str(int(seed)) for seed in strict_heldout_seeds}
    for row_index, row in enumerate(rows):
        for path, value in _flatten_trainable(_mapping(row.get("trainable"))):
            lower_path = path.lower()
            if any(token in lower_path for token in FORBIDDEN_TRAINABLE_TOKENS):
                failure = {
                    "row_index": row_index,
                    "path": path,
                    "reason": "forbidden_trainable_path_token",
                }
                forbidden_failures.append(failure)
                if "seed" in lower_path:
                    strict_seed_path_failures.append(failure)
            if isinstance(value, str) and value in strict_tokens and "seed" in lower_path:
                strict_seed_path_failures.append(
                    {
                        "row_index": row_index,
                        "path": path,
                        "reason": "strict_seed_identity_in_trainable_field",
                    }
                )
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    finite_failures.append(
                        {
                            "row_index": row_index,
                            "path": path,
                            "reason": "non_finite_numeric_value",
                        }
                    )
    failures = forbidden_failures + finite_failures + strict_seed_path_failures
    return {
        "policy": "m3_safe_archive_expansion_trainable_leakage_scan_v1",
        "passed": not failures,
        "row_count": len(rows),
        "forbidden_tokens": list(FORBIDDEN_TRAINABLE_TOKENS),
        "strict_heldout_seeds": [int(seed) for seed in strict_heldout_seeds],
        "forbidden_path_failure_count": len(forbidden_failures),
        "finite_value_failure_count": len(finite_failures),
        "strict_heldout_seed_trainable_leakage_count": len(
            strict_seed_path_failures
        ),
        "failures": failures[:24],
    }


def _safe_archive_expansion_support_diagnostics(
    *,
    branch_results: Sequence[Mapping[str, object]],
    excluded_rows: Sequence[Mapping[str, object]],
    source_integrity: Mapping[str, object],
    safe_label_count: int,
    min_safe_label_count: int,
) -> dict[str, object]:
    action_diagnostics = _safe_archive_expansion_action_run_diagnostics(
        branch_results
    )
    excluded_diagnostics = _safe_archive_expansion_excluded_diagnostics(
        excluded_rows
    )
    carrion = _safe_archive_expansion_carrion_failure_diagnostics(branch_results)
    return {
        "policy": "m3_safe_archive_expansion_support_diagnostics_v1",
        "excluded": excluded_diagnostics,
        "safety_vet_failures": action_diagnostics,
        "carrion": carrion,
        "support_limitation_assessment": (
            _safe_archive_expansion_support_limitation_assessment(
                source_integrity=source_integrity,
                safe_label_count=int(safe_label_count),
                min_safe_label_count=int(min_safe_label_count),
                branch_result_count=len(branch_results),
                action_diagnostics=action_diagnostics,
                carrion_diagnostics=carrion,
            )
        ),
    }


def _safe_archive_expansion_action_run_diagnostics(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failed_by_fixture: dict[str, Counter[str]] = {}
    failed_by_action: dict[str, Counter[str]] = {}
    failed_by_fixture_action: dict[str, Counter[str]] = {}
    action_run_counts_by_fixture: Counter[str] = Counter()
    action_run_counts_by_action: Counter[str] = Counter()
    safe_counts_by_fixture: Counter[str] = Counter()
    safe_counts_by_action: Counter[str] = Counter()
    total = 0
    safe = 0
    for result in branch_results:
        fixture = str(result.get("fixture", "unknown"))
        for run in _list_of_mappings(result.get("action_runs")):
            action = str(run.get("forced_action", "unknown"))
            total += 1
            action_run_counts_by_fixture[fixture] += 1
            action_run_counts_by_action[action] += 1
            vet = _safe_archive_expansion_label_vet(run)
            failed = _failed_floor_names(vet)
            if not failed:
                safe += 1
                safe_counts_by_fixture[fixture] += 1
                safe_counts_by_action[action] += 1
                continue
            fixture_counter = failed_by_fixture.setdefault(fixture, Counter())
            action_counter = failed_by_action.setdefault(action, Counter())
            fixture_action_counter = failed_by_fixture_action.setdefault(
                f"{fixture}:{action}",
                Counter(),
            )
            for floor in failed:
                fixture_counter[floor] += 1
                action_counter[floor] += 1
                fixture_action_counter[floor] += 1
    top_failed = _top_counter(
        Counter(
            {
                floor: sum(counter.get(floor, 0) for counter in failed_by_fixture.values())
                for floor in {
                    key
                    for counter in failed_by_fixture.values()
                    for key in counter
                }
            }
        )
    )
    return {
        "policy": "m3_safe_archive_expansion_action_run_safety_vet_summary_v1",
        "action_run_count": total,
        "safe_action_run_count": safe,
        "unsafe_action_run_count": total - safe,
        "action_run_counts_by_fixture": dict(sorted(action_run_counts_by_fixture.items())),
        "action_run_counts_by_action": dict(sorted(action_run_counts_by_action.items())),
        "safe_action_run_counts_by_fixture": dict(sorted(safe_counts_by_fixture.items())),
        "safe_action_run_counts_by_action": dict(sorted(safe_counts_by_action.items())),
        "failed_floor_counts_by_fixture": _counter_mapping(failed_by_fixture),
        "failed_floor_counts_by_action": _counter_mapping(failed_by_action),
        "failed_floor_counts_by_fixture_action": _counter_mapping(
            failed_by_fixture_action
        ),
        "top_failed_floors": top_failed,
    }


def _safe_archive_expansion_excluded_diagnostics(
    excluded_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_fixture: dict[str, Counter[str]] = {}
    by_action: dict[str, Counter[str]] = {}
    by_fixture_action: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    for row in excluded_rows:
        fixture = str(row.get("fixture", "unknown"))
        action = str(row.get("label_action", "unselected"))
        reason = str(row.get("excluded_reason", "unknown"))
        reason_counts[reason] += 1
        by_fixture.setdefault(fixture, Counter())[reason] += 1
        by_action.setdefault(action, Counter())[reason] += 1
        by_fixture_action.setdefault(f"{fixture}:{action}", Counter())[reason] += 1
    return {
        "policy": "m3_safe_archive_expansion_excluded_row_summary_v1",
        "excluded_row_count": len(excluded_rows),
        "excluded_reason_counts": dict(sorted(reason_counts.items())),
        "excluded_reason_counts_by_fixture": _counter_mapping(by_fixture),
        "excluded_reason_counts_by_action": _counter_mapping(by_action),
        "excluded_reason_counts_by_fixture_action": _counter_mapping(
            by_fixture_action
        ),
    }


def _safe_archive_expansion_carrion_failure_diagnostics(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    branch_count = 0
    action_run_count = 0
    safe_action_run_count = 0
    failed_floor_counts: Counter[str] = Counter()
    failed_floor_counts_by_action: dict[str, Counter[str]] = {}
    unsupported_resolved = _metric_failure_summary(
        failed_floor="no_resolved_invalid_increase"
    )
    alive = _metric_failure_summary(failed_floor="no_alive_regression")
    births = _metric_failure_summary(failed_floor="no_birth_regression")
    replay = _metric_failure_summary(failed_floor="branch_replay_deterministic")
    heuristic = _metric_failure_summary(
        failed_floor="zero_heuristic_action_source_count"
    )
    unsupported_requested = _metric_failure_summary(
        failed_floor="no_unsupported_requested_actions"
    )
    for result in branch_results:
        if str(result.get("fixture")) != "carrion_only":
            continue
        branch_count += 1
        for run in _list_of_mappings(result.get("action_runs")):
            action_run_count += 1
            action = str(run.get("forced_action", "unknown"))
            vet = _safe_archive_expansion_label_vet(run)
            failed = _failed_floor_names(vet)
            if not failed:
                safe_action_run_count += 1
            action_counter = failed_floor_counts_by_action.setdefault(
                action,
                Counter(),
            )
            for floor in failed:
                failed_floor_counts[floor] += 1
                action_counter[floor] += 1
            _update_carrion_metric_summaries(
                run,
                failed=set(failed),
                unsupported_resolved=unsupported_resolved,
                alive=alive,
                births=births,
                replay=replay,
                heuristic=heuristic,
                unsupported_requested=unsupported_requested,
            )
    return {
        "policy": "m3_safe_archive_expansion_carrion_failure_summary_v1",
        "branch_result_count": branch_count,
        "action_run_count": action_run_count,
        "safe_action_run_count": safe_action_run_count,
        "unsafe_action_run_count": action_run_count - safe_action_run_count,
        "failed_floor_counts": dict(sorted(failed_floor_counts.items())),
        "failed_floor_counts_by_action": _counter_mapping(
            failed_floor_counts_by_action
        ),
        "top_failure_reasons": _top_counter(failed_floor_counts),
        "unsupported_resolved_action_count": unsupported_resolved,
        "alive_regression": alive,
        "birth_regression": births,
        "replay_failure": replay,
        "heuristic_action_source_count": heuristic,
        "unsupported_requested_action_count": unsupported_requested,
    }


def _update_carrion_metric_summaries(
    run: Mapping[str, object],
    *,
    failed: set[str],
    unsupported_resolved: dict[str, object],
    alive: dict[str, object],
    births: dict[str, object],
    replay: dict[str, object],
    heuristic: dict[str, object],
    unsupported_requested: dict[str, object],
) -> None:
    baseline = _mapping(run.get("deltas_vs_baseline"))
    _record_numeric_failure_summary(
        unsupported_resolved,
        value=baseline.get("unsupported_resolved_action_count"),
        failed="no_resolved_invalid_increase" in failed,
        positive_only=True,
    )
    _record_numeric_failure_summary(
        alive,
        value=baseline.get("alive_agents"),
        failed="no_alive_regression" in failed,
        negative_only=True,
    )
    _record_numeric_failure_summary(
        births,
        value=baseline.get("births"),
        failed="no_birth_regression" in failed,
        negative_only=True,
    )
    _record_numeric_failure_summary(
        unsupported_requested,
        value=run.get("unsupported_requested_action_count"),
        failed="no_unsupported_requested_actions" in failed,
        positive_only=True,
    )
    _record_boolean_failure_summary(
        replay,
        failed="branch_replay_deterministic" in failed,
        observed=_mapping(run.get("replay_verification")).get("verified"),
    )
    _record_numeric_failure_summary(
        heuristic,
        value=run.get("heuristic_action_source_count"),
        failed="zero_heuristic_action_source_count" in failed,
        positive_only=True,
    )


def _safe_archive_expansion_support_limitation_assessment(
    *,
    source_integrity: Mapping[str, object],
    safe_label_count: int,
    min_safe_label_count: int,
    branch_result_count: int,
    action_diagnostics: Mapping[str, object],
    carrion_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True:
        return {
            "primary": "source_integrity_limited",
            "rationale": "branch evidence source integrity did not pass",
        }
    if int(safe_label_count) >= int(min_safe_label_count):
        return {
            "primary": "support_ready",
            "rationale": "safe label support floor passed",
        }
    evidence_size_limited = int(branch_result_count) < int(min_safe_label_count)
    carrion_resolution = _int(
        _mapping(carrion_diagnostics.get("unsupported_resolved_action_count")).get(
            "failed_run_count"
        )
    ) + _int(
        _mapping(carrion_diagnostics.get("unsupported_requested_action_count")).get(
            "failed_run_count"
        )
    )
    carrion_objective = _int(
        _mapping(carrion_diagnostics.get("alive_regression")).get("failed_run_count")
    ) + _int(
        _mapping(carrion_diagnostics.get("birth_regression")).get("failed_run_count")
    )
    if carrion_resolution > 0 and carrion_resolution >= carrion_objective:
        primary = "action_resolution_limited"
        rationale = "carrion action runs most often fail action-resolution safety floors"
    elif carrion_objective > 0:
        primary = "objective_limited"
        rationale = "carrion action runs most often fail alive/birth objective floors"
    elif evidence_size_limited:
        primary = "data_limited"
        rationale = "branch evidence contains fewer branch points than the safe label floor"
    else:
        primary = "data_limited"
        rationale = "safe labels remain below floor after available evidence"
    return {
        "primary": primary,
        "rationale": rationale,
        "evidence_size_limited": evidence_size_limited,
        "branch_result_count": int(branch_result_count),
        "safe_label_count": int(safe_label_count),
        "min_safe_label_count": int(min_safe_label_count),
        "unsafe_action_run_count": _int(
            action_diagnostics.get("unsafe_action_run_count")
        ),
        "carrion_action_resolution_failure_count": carrion_resolution,
        "carrion_objective_failure_count": carrion_objective,
    }


def _safe_archive_expansion_support_floors(
    *,
    source_integrity: Mapping[str, object],
    dataset_scan: Mapping[str, object],
    safe_label_count: int,
    min_safe_label_count: int,
    dominant_label_share: float,
    heuristic_action_source_count: int,
) -> dict[str, object]:
    floors = [
        _floor(
            "source_integrity_passed",
            source_integrity.get("passed") is True,
            observed=source_integrity.get("failures"),
            required=[],
        ),
        _floor(
            "safe_label_count_gte_minimum",
            int(safe_label_count) >= int(min_safe_label_count),
            observed=safe_label_count,
            required=min_safe_label_count,
        ),
        _floor(
            "dominant_safe_label_action_share_lte_0_50",
            float(dominant_label_share) <= MAX_DOMINANT_SAFE_LABEL_ACTION_SHARE,
            observed=_round(dominant_label_share),
            required=MAX_DOMINANT_SAFE_LABEL_ACTION_SHARE,
        ),
        _floor(
            "zero_heuristic_action_source_count",
            int(heuristic_action_source_count) == 0,
            observed=heuristic_action_source_count,
            required=0,
        ),
        _floor(
            "no_strict_heldout_seed_training_leakage",
            _int(dataset_scan.get("strict_heldout_seed_trainable_leakage_count")) == 0,
            observed=dataset_scan.get("strict_heldout_seed_trainable_leakage_count"),
            required=0,
        ),
        _floor(
            "dataset_trainable_rows_leakage_safe",
            dataset_scan.get("passed") is True,
            observed=dataset_scan.get("failures"),
            required=[],
        ),
    ]
    first_failed = next((floor for floor in floors if floor["passed"] is not True), None)
    return {
        "policy": "m3_safe_archive_expansion_support_floors_v1",
        "passed": first_failed is None,
        "first_failed_floor": None if first_failed is None else first_failed["name"],
        "floors": floors,
    }


def _safe_archive_expansion_classification(
    *,
    source_integrity: Mapping[str, object],
    support_floors: Mapping[str, object],
) -> str:
    if "partial_branch_evidence" in set(_list(source_integrity.get("failures"))):
        return "m3_safe_archive_expansion_partial_branch_evidence_no_training"
    if source_integrity.get("passed") is not True:
        return "m3_safe_archive_expansion_source_integrity_failed"
    if support_floors.get("passed") is True:
        return "m3_safe_archive_expansion_support_ready_no_training_run"
    return "m3_safe_archive_expansion_blocked_no_training"


def _safety_vet_label(row: Mapping[str, object]) -> dict[str, object]:
    metadata = _mapping(row.get("metadata"))
    outcome = _mapping(metadata.get("outcome_evidence"))
    baseline = _mapping(outcome.get("deltas_vs_baseline"))
    floors = [
        _floor(
            "no_unsupported_requested_action_in_branch_evidence",
            _int(baseline.get("unsupported_requested_action_count")) == 0,
            observed=baseline.get("unsupported_requested_action_count"),
            required=0,
        ),
        _floor(
            "no_resolved_invalid_increase_in_branch_evidence",
            _int(baseline.get("unsupported_resolved_action_count")) <= 0,
            observed=baseline.get("unsupported_resolved_action_count"),
            required="<= 0",
        ),
        _floor(
            "no_alive_regression_in_branch_evidence",
            _int(baseline.get("alive_agents")) >= 0,
            observed=baseline.get("alive_agents"),
            required=">= 0",
        ),
        _floor(
            "no_birth_regression_in_branch_evidence",
            _int(baseline.get("births")) >= 0,
            observed=baseline.get("births"),
            required=">= 0",
        ),
    ]
    failed = [floor for floor in floors if floor["passed"] is not True]
    return {
        "policy": "v146_pretraining_label_safety_vet_v1",
        "passed": not failed,
        "failed_floor_count": len(failed),
        "floors": floors,
    }


def _v145_blacklist(v145_report: Mapping[str, object]) -> list[dict[str, object]]:
    route = _mapping(v145_report.get("route_decision"))
    blacklist = []
    for row in _list_of_mappings(route.get("label_blacklist")):
        blacklist.append(
            {
                "seed": _int(row.get("seed")),
                "agent_id": _int(row.get("agent_id")),
                "tick": _int(row.get("tick")),
                "residual_action": row.get("residual_action"),
                "branch_id": row.get("branch_id"),
                "label_source_row_index": row.get("label_source_row_index"),
            }
        )
    return blacklist


def _row_identity(
    *,
    index: int,
    row: Mapping[str, object],
) -> dict[str, object]:
    metadata = _mapping(row.get("metadata"))
    trainable = _mapping(row.get("trainable"))
    label = _mapping(trainable.get("label"))
    return {
        "row_index": int(index),
        "seed": _int(metadata.get("seed")),
        "fixture": metadata.get("fixture"),
        "branch_id": metadata.get("branch_id"),
        "branch_tick": _int(metadata.get("branch_tick")),
        "agent_id": _int(metadata.get("agent_id")),
        "label_action": label.get("action"),
    }


def _blacklisted(
    identity: Mapping[str, object],
    blacklist: Sequence[Mapping[str, object]],
    *,
    allow_label_source_row_index: bool = True,
) -> bool:
    for item in blacklist:
        if (
            allow_label_source_row_index
            and item.get("label_source_row_index") == identity.get("row_index")
        ):
            return True
        if (
            _int(item.get("seed")) == _int(identity.get("seed"))
            and _int(item.get("agent_id")) == _int(identity.get("agent_id"))
            and item.get("residual_action") == identity.get("label_action")
        ):
            return True
        if item.get("branch_id") and item.get("branch_id") == identity.get("branch_id"):
            return True
    return False


def _fixture_gate_from_runs(
    *,
    runs: Sequence[Mapping[str, object]],
    fixture_config: Mapping[str, object],
    policy_name: str,
) -> dict[str, object]:
    suite = {
        "policy": evaluate_cli.MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
        "suite": "basic",
        "fixture_names": ["carrion_only"],
        "evaluated_policy_key": "mind_v3",
        "evaluated_policy_name": policy_name,
        "seeds": list(STRICT_CARRION_FIXTURE_SEEDS),
        "ticks": STRICT_TICKS,
        "fixtures": [
            {
                "fixture": "carrion_only",
                "scenario_config": evaluate_cli._fixture_scenario_config(
                    "carrion_only"
                ),
                "comparison": {
                    "mind_v3": {
                        "runs": [dict(run) for run in runs],
                        "aggregate": _aggregate_runs(runs),
                    }
                },
            }
        ],
    }
    return evaluate_cli.mind_v3_fixture_gate_status(
        fixture_suite=suite,
        fixture_config=fixture_config,
    )


def _fixture_blocker_count(value: object) -> int:
    gate = _mapping(value)
    blockers = gate.get("blockers")
    if isinstance(blockers, list):
        return len(blockers)
    return _int(gate.get("blocker_count"))


def _first_failed_action_distribution(
    *,
    broad: Mapping[str, object] | None,
    carrion: Mapping[str, object] | None,
) -> dict[str, object]:
    distribution: dict[str, object] = {}
    for name, section in (("broad", broad), ("carrion", carrion)):
        aggregate = _mapping(_mapping(_mapping(section).get("candidate")).get("aggregate"))
        diagnostics = _mapping(aggregate.get("support_residual_diagnostics"))
        if aggregate:
            distribution[f"{name}_requested_action_counts"] = aggregate.get(
                "requested_action_counts"
            )
            distribution[f"{name}_dominant_requested_action_share"] = aggregate.get(
                "dominant_requested_action_share"
            )
            distribution[f"{name}_applied_override_action_counts"] = diagnostics.get(
                "applied_override_action_counts"
            )
    return distribution


def _infer_first_failing_seed(
    *,
    first_failed_floor: str,
    broad: Mapping[str, object] | None,
    carrion: Mapping[str, object] | None,
) -> int | None:
    if first_failed_floor == "broad_resolved_invalid_not_increased":
        for row in _list_of_mappings(_mapping(broad).get("per_seed_delta")):
            if _int(row.get("resolved_invalid_action_count_delta")) > 0:
                return _int(row.get("seed"))
    if first_failed_floor == "carrion_resolved_invalid_not_increased":
        for row in _list_of_mappings(_mapping(carrion).get("per_seed_delta")):
            if _int(row.get("resolved_invalid_action_count_delta")) > 0:
                return _int(row.get("seed"))
    if first_failed_floor == "carrion_alive_or_blocker_improved":
        rows = _list_of_mappings(_mapping(carrion).get("per_seed_delta"))
        return _int(rows[0].get("seed")) if rows else None
    return None


def _shadow_summary(shadow: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if shadow is None:
        return None
    aggregate = _mapping(shadow.get("aggregate"))
    diagnostics = _mapping(aggregate.get("support_residual_diagnostics"))
    return {
        "shadow_gate": shadow.get("shadow_gate"),
        "aggregate": {
            "unsupported_requested_action_count": aggregate.get(
                "unsupported_requested_action_count"
            ),
            "unsupported_proposed_action_count": diagnostics.get(
                "unsupported_proposed_action_count"
            ),
            "gate_accepted_override_count": diagnostics.get(
                "gate_accepted_override_count"
            ),
            "gate_accepted_override_action_counts": diagnostics.get(
                "gate_accepted_override_action_counts"
            ),
            "dominant_gate_accepted_override_action_share": diagnostics.get(
                "dominant_gate_accepted_override_action_share"
            ),
        },
    }


def _baseline_summary(baseline: Mapping[str, object]) -> dict[str, object]:
    broad = _mapping(_mapping(baseline.get("broad")).get("aggregate"))
    carrion = _mapping(_mapping(baseline.get("carrion_only")).get("aggregate"))
    return {
        "digest": stable_payload_digest(baseline),
        "broad": {
            "alive_agents_mean": broad.get("alive_agents_mean"),
            "births_mean": broad.get("births_mean"),
            "resolved_invalid_action_count": broad.get("resolved_invalid_action_count"),
            "dominant_requested_action_share": broad.get(
                "dominant_requested_action_share"
            ),
        },
        "carrion_only": {
            "alive_agents_mean": carrion.get("alive_agents_mean"),
            "births_mean": carrion.get("births_mean"),
            "resolved_invalid_action_count": carrion.get(
                "resolved_invalid_action_count"
            ),
            "dominant_requested_action_share": carrion.get(
                "dominant_requested_action_share"
            ),
        },
    }


def _load_campaign_config(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CandidateCampaignError("candidate campaign config must be a JSON object")
    return payload


def _load_optional_json(path: str | Path) -> dict[str, object] | None:
    resolved = Path(path)
    if not resolved.exists():
        return None
    return load_json_report(resolved)


def _load_candidate_artifact(path: str | Path) -> dict[str, object] | None:
    resolved = Path(path)
    if not resolved.exists():
        return None
    return load_support_gated_residual_artifact(resolved)


def _validate_mode(mode: str) -> str:
    if mode not in {"smoke", "full"}:
        raise CandidateCampaignError("mode must be smoke or full")
    return mode


def _candidate_family(candidate_id: str) -> str:
    if candidate_id == "linear_control":
        return "linear_control"
    if candidate_id == "v144_branch_intervention_residual":
        return "negative_control_residual"
    if candidate_id == "neural_offline":
        return "neural_offline"
    return "support_gated_residual"


def _support_artifact_contract() -> dict[str, bool]:
    return {
        "one_row_one_agent_local_decision": True,
        "requires_action_mask": True,
        "requires_policy_visible_features_only": True,
        "requires_linear_default_action": True,
        "requires_planner_outcome_tables": False,
        "requires_global_batch_assignment": False,
        "uses_heuristic_fallback": False,
        "uses_seed_id_as_runtime_feature": False,
        "uses_branch_id_as_runtime_feature": False,
        "uses_fixture_id_as_runtime_feature": False,
        "uses_logged_action_as_runtime_fallback": False,
        "uses_private_simulator_state": False,
    }


def _add_floor_blocker(
    blockers: list[dict[str, object]],
    *,
    name: str,
    passed: bool,
    observed: object,
    required: object,
    fixture: str | None = None,
) -> None:
    if passed:
        return
    blockers.append(
        {
            "name": name,
            "observed": observed,
            "required": required,
            "fixture": fixture,
        }
    )


def _floor(
    name: str,
    passed: bool,
    *,
    observed: object,
    required: object,
    fixture: str | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
        "fixture": fixture,
        "seed": seed,
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _limit_int_sequence(values: Sequence[int], *, limit: int) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if int(limit) <= 0:
        return normalized
    return normalized[: int(limit)]


def _failed_floor_names(vet: Mapping[str, object]) -> list[str]:
    return [
        str(floor.get("name"))
        for floor in _list_of_mappings(vet.get("floors"))
        if floor.get("passed") is not True
    ]


def _counter_mapping(counters: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        str(key): dict(sorted((str(name), int(count)) for name, count in counter.items()))
        for key, counter in sorted(counters.items())
    }


def _top_counter(counter: Counter[str], *, limit: int = 8) -> list[dict[str, object]]:
    return [
        {"reason": str(key), "count": int(count)}
        for key, count in sorted(
            counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[: int(limit)]
    ]


def _metric_failure_summary(*, failed_floor: str) -> dict[str, object]:
    return {
        "failed_floor": failed_floor,
        "run_count": 0,
        "failed_run_count": 0,
        "missing_or_nonfinite_count": 0,
        "positive_delta_count": 0,
        "negative_delta_count": 0,
        "observed_min": None,
        "observed_max": None,
    }


def _record_numeric_failure_summary(
    summary: dict[str, object],
    *,
    value: object,
    failed: bool,
    positive_only: bool = False,
    negative_only: bool = False,
) -> None:
    summary["run_count"] = _int(summary.get("run_count")) + 1
    if failed:
        summary["failed_run_count"] = _int(summary.get("failed_run_count")) + 1
    if not _is_finite_number(value):
        summary["missing_or_nonfinite_count"] = (
            _int(summary.get("missing_or_nonfinite_count")) + 1
        )
        return
    observed = _float(value)
    if observed > 0.0:
        summary["positive_delta_count"] = _int(summary.get("positive_delta_count")) + 1
    if observed < 0.0:
        summary["negative_delta_count"] = _int(summary.get("negative_delta_count")) + 1
    if positive_only and observed > 0.0:
        summary["positive_failure_count"] = _int(
            summary.get("positive_failure_count")
        ) + 1
    if negative_only and observed < 0.0:
        summary["negative_failure_count"] = _int(
            summary.get("negative_failure_count")
        ) + 1
    current_min = summary.get("observed_min")
    current_max = summary.get("observed_max")
    summary["observed_min"] = (
        observed
        if not _is_finite_number(current_min)
        else min(_float(current_min), observed)
    )
    summary["observed_max"] = (
        observed
        if not _is_finite_number(current_max)
        else max(_float(current_max), observed)
    )


def _record_boolean_failure_summary(
    summary: dict[str, object],
    *,
    failed: bool,
    observed: object,
) -> None:
    summary["run_count"] = _int(summary.get("run_count")) + 1
    if failed:
        summary["failed_run_count"] = _int(summary.get("failed_run_count")) + 1
    if observed is True:
        summary["true_count"] = _int(summary.get("true_count")) + 1
    elif observed is False:
        summary["false_count"] = _int(summary.get("false_count")) + 1
    else:
        summary["missing_or_non_bool_count"] = (
            _int(summary.get("missing_or_non_bool_count")) + 1
        )


def _flatten_trainable(
    value: object,
    *,
    prefix: str = "",
) -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, object]] = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_trainable(item, prefix=path))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            rows.extend(_flatten_trainable(item, prefix=path))
        return rows
    return [(prefix, value)]


def _dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = max(sorted(counts.items()), key=lambda item: (int(item[1]), item[0]))
    return {
        "key": key,
        "count": int(count),
        "share": _round(float(count) / float(total)),
    }


def _int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _bool_int(value: object) -> int:
    return 1 if value is True else 0
