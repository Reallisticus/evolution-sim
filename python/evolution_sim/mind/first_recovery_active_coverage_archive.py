from __future__ import annotations

import glob
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from evolution_sim.mind.evolution import load_mind_v3_founder_template
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    build_first_recovery_archive_rows,
    trainable_public_input_leakage,
)
from evolution_sim.mind.first_recovery_branch_oracle_audit import (
    FirstRecoveryBranchOracleAuditError,
    _branch_state_digest,
    _evaluate_branch_target,
    _materialize_branch_targets,
    _seed_from_path,
    _source_kind,
)
from evolution_sim.mind.first_recovery_rare_action_coverage_targeting import (
    RARE_ACTION_ADDITIONAL_NEEDED,
)
from evolution_sim.mind.first_recovery_rare_attack_coverage_collection import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V122_REPORT_PATH,
    EXPECTED_CURRENT_RARE_SUPPORT,
    MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION,
    TARGET_ACTIONS,
    build_first_recovery_rare_attack_coverage_collection,
)
from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V119_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V119_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V120_REPORT_PATH,
    _trainable_leakage,
)
from evolution_sim.mind.first_recovery_rare_action_coverage_targeting import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V121_REPORT_PATH,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_ACTIVE_COVERAGE_ARCHIVE_SCHEMA_VERSION = (
    "mind_v3_first_recovery_active_coverage_archive_v1"
)
MIND_V3_FIRST_RECOVERY_ACTIVE_COVERAGE_ARCHIVE_POLICY = (
    "diagnostics_only_first_recovery_v123_active_rare_attack_coverage_archive_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.json"
)
DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.jsonl.gz"
)
DEFAULT_V122_RERUN_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v122-over-v123-first-recovery-rare-attack-coverage-collection.json"
)
DEFAULT_V122_RERUN_CANDIDATES_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v122-over-v123-first-recovery-rare-attack-candidates.jsonl"
)
DEFAULT_ROLLOUT_CONTEXT_REPORT_PATH = Path(
    "output/mind/mind-v3-v5-rollout-context-search-80-120-diagnostic.json"
)
DEFAULT_TRAJECTORY_GLOB = (
    "output/mind/mind-v3-v5-carrion-only-120-trajectories/*mind-v3*.jsonl.gz"
)
DEFAULT_MAX_TRAJECTORIES = 6
DEFAULT_MAX_SOURCE_RECORDS_PER_ACTION = 4
DEFAULT_MAX_EVALUATED_BRANCHES = 12
MAX_EXAMPLES = 12

NO_AUTHORIZATION_FIELDS: tuple[str, ...] = (
    "v113_readiness_rerun_allowed",
    "downstream_shadow_scorer_allowed",
    "claim_causality",
)

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "diagnostics_only_no_runtime_promotion",
    "active_coverage_rare_attacks_found",
    "active_coverage_partial_rare_attack_found",
    "active_coverage_no_rare_attack_found",
    "active_coverage_source_integrity_failed",
    "readiness_rerun_blocked",
)


@dataclass(frozen=True, slots=True)
class FirstRecoveryActiveCoverageArchiveBuild:
    report: dict[str, object]
    archive_rows: tuple[dict[str, object], ...]
    v122_rerun_report: dict[str, object]
    v122_candidate_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _GenerationResult:
    archive_rows: tuple[dict[str, object], ...]
    branch_results: tuple[dict[str, object], ...]
    accepted_branch_ids: tuple[str, ...]
    search_budget_used: dict[str, object]
    source_record_counts: dict[str, int]
    rejection_counts: dict[str, dict[str, int]]
    rejection_examples: dict[str, dict[str, list[dict[str, object]]]]
    materialization_failures: tuple[dict[str, object], ...]


def build_first_recovery_active_coverage_archive(
    *,
    v119_report: Mapping[str, object] | None = None,
    v119_report_path: str | Path | None = DEFAULT_V119_REPORT_PATH,
    manifest_rows: Sequence[Mapping[str, object]] | None = None,
    manifest_path: str | Path | None = DEFAULT_V119_MANIFEST_PATH,
    v120_report: Mapping[str, object] | None = None,
    v120_report_path: str | Path | None = DEFAULT_V120_REPORT_PATH,
    v121_report: Mapping[str, object] | None = None,
    v121_report_path: str | Path | None = DEFAULT_V121_REPORT_PATH,
    rollout_context_report_path: str | Path | None = DEFAULT_ROLLOUT_CONTEXT_REPORT_PATH,
    trajectory_paths: Sequence[str | Path] = (),
    trajectory_glob_patterns: Sequence[str] = (DEFAULT_TRAJECTORY_GLOB,),
    max_trajectories: int = DEFAULT_MAX_TRAJECTORIES,
    max_source_records_per_action: int = DEFAULT_MAX_SOURCE_RECORDS_PER_ACTION,
    max_evaluated_branches: int = DEFAULT_MAX_EVALUATED_BRANCHES,
    verify_replay: bool = True,
) -> FirstRecoveryActiveCoverageArchiveBuild:
    prerequisite = build_first_recovery_rare_attack_coverage_collection(
        v119_report=v119_report,
        v119_report_path=v119_report_path,
        manifest_rows=manifest_rows,
        manifest_path=manifest_path,
        v120_report=v120_report,
        v120_report_path=v120_report_path,
        v121_report=v121_report,
        v121_report_path=v121_report_path,
        candidate_archive_report={
            "schema_version": MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION
        },
        candidate_archive_rows=[],
    )
    source_integrity = _source_integrity_from_v122_prerequisite(
        prerequisite.report,
        rollout_context_report_path=rollout_context_report_path,
        trajectory_paths=trajectory_paths,
        trajectory_glob_patterns=trajectory_glob_patterns,
    )
    search_config = _search_config(
        rollout_context_report_path=rollout_context_report_path,
        trajectory_paths=trajectory_paths,
        trajectory_glob_patterns=trajectory_glob_patterns,
        max_trajectories=max_trajectories,
        max_source_records_per_action=max_source_records_per_action,
        max_evaluated_branches=max_evaluated_branches,
        verify_replay=verify_replay,
    )
    generation = (
        _generate_active_archive_rows(
            search_config=search_config,
            v119_report=v119_report,
            v119_report_path=v119_report_path,
            manifest_rows=manifest_rows,
            manifest_path=manifest_path,
            v120_report=v120_report,
            v120_report_path=v120_report_path,
            v121_report=v121_report,
            v121_report_path=v121_report_path,
        )
        if source_integrity.get("passed") is True
        else _empty_generation()
    )
    v122_rerun = build_first_recovery_rare_attack_coverage_collection(
        v119_report=v119_report,
        v119_report_path=v119_report_path,
        manifest_rows=manifest_rows,
        manifest_path=manifest_path,
        v120_report=v120_report,
        v120_report_path=v120_report_path,
        v121_report=v121_report,
        v121_report_path=v121_report_path,
        candidate_archive_report={
            "schema_version": MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION
        },
        candidate_archive_rows=generation.archive_rows,
    )
    leakage = trainable_public_input_leakage(generation.archive_rows)
    v120_leakage = _trainable_leakage(generation.archive_rows)
    strict_seed_leakage_count = _strict_seed_leakage_count(generation.archive_rows)
    replay = _replay_verification(generation.branch_results)
    target_counts = {
        action: sum(
            1 for row in generation.archive_rows if row.get("candidate_action") == action
        )
        for action in TARGET_ACTIONS
    }
    found_counts = {
        action: int(
            dict(v122_rerun.report.get("found_candidate_counts", {})).get(action, 0)
        )
        for action in TARGET_ACTIONS
    }
    source_integrity = _final_source_integrity(
        source_integrity,
        leakage=leakage,
        v120_leakage=v120_leakage,
        strict_seed_leakage_count=strict_seed_leakage_count,
        replay=replay,
        v122_rerun_report=v122_rerun.report,
    )
    classification = _classification(source_integrity, found_counts)
    recommendation = _recommendation(classification, v122_rerun.report)
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
        "active_coverage_schema_version": (
            MIND_V3_FIRST_RECOVERY_ACTIVE_COVERAGE_ARCHIVE_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_ACTIVE_COVERAGE_ARCHIVE_POLICY,
        "contract": _contract(),
        "source_integrity": source_integrity,
        "search_config": search_config,
        "search_budget_used": generation.search_budget_used,
        "generated_branch_count": len(generation.accepted_branch_ids),
        "generated_archive_row_count": len(generation.archive_rows),
        "target_action_candidate_counts": target_counts,
        "accepted_by_v122_candidate_counts": found_counts,
        "rejection_counts": generation.rejection_counts,
        "rejection_examples": generation.rejection_examples,
        "replay_verified": replay["replay_verified"],
        "replay_verification": replay,
        "heuristic_action_source_count": replay["heuristic_action_source_count"],
        "trainable_leakage": {
            "branch_archive_leakage": leakage,
            "split_key_leak_count": v120_leakage["split_key_leak_count"],
            "forbidden_metadata_key_count": v120_leakage[
                "forbidden_metadata_key_count"
            ],
        },
        "strict_seed_leakage_count": strict_seed_leakage_count,
        "v122_rerun": {
            "schema_version": (
                MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION
            ),
            "classification": v122_rerun.report.get("classification"),
            "found_candidate_counts": found_counts,
            "would_clear_v120_rare_action_limitation_if_accepted": (
                _mapping(v122_rerun.report.get("recommendation")).get(
                    "would_clear_v120_rare_action_limitation_if_accepted",
                    False,
                )
            ),
            "report_path": str(DEFAULT_V122_RERUN_OUTPUT_PATH),
        },
        "classification": classification,
        "recommendation": recommendation,
        "archive_rows_path": str(DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH),
        "non_promoted": True,
    }
    return FirstRecoveryActiveCoverageArchiveBuild(
        report=report,
        archive_rows=generation.archive_rows,
        v122_rerun_report=v122_rerun.report,
        v122_candidate_rows=tuple(v122_rerun.candidate_rows),
    )


def write_first_recovery_active_coverage_archive_outputs(
    build: FirstRecoveryActiveCoverageArchiveBuild,
    *,
    output_path: str | Path,
    archive_rows_output_path: str | Path,
    v122_rerun_output_path: str | Path | None = DEFAULT_V122_RERUN_OUTPUT_PATH,
    v122_candidates_output_path: str | Path | None = (
        DEFAULT_V122_RERUN_CANDIDATES_OUTPUT_PATH
    ),
) -> None:
    rows_path = Path(archive_rows_output_path)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(rows_path, "wt", encoding="utf-8") as handle:
        for row in build.archive_rows:
            json.dump(row, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
    report = {
        **build.report,
        "archive_rows_path": str(rows_path),
        "branch_archive_summary": {
            "archive_row_count": len(build.archive_rows),
            "branch_result_count": build.report.get("generated_branch_count"),
            "action_run_count": len(build.archive_rows),
            "replay_verified": build.report.get("replay_verified"),
            "heuristic_action_source_count": build.report.get(
                "heuristic_action_source_count"
            ),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    if v122_rerun_output_path is not None:
        v122_path = Path(v122_rerun_output_path)
        v122_path.parent.mkdir(parents=True, exist_ok=True)
        v122_report = dict(build.v122_rerun_report)
        if v122_candidates_output_path is not None:
            candidate_path = Path(v122_candidates_output_path)
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            with candidate_path.open("w", encoding="utf-8") as handle:
                for row in build.v122_candidate_rows:
                    json.dump(row, handle, sort_keys=True, allow_nan=False)
                    handle.write("\n")
            v122_report["candidate_manifest"] = {
                **_mapping(v122_report.get("candidate_manifest")),
                "path": str(candidate_path),
            }
        with v122_path.open("w", encoding="utf-8") as handle:
            json.dump(v122_report, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "viewer_effect": "none",
        "objective_values_changed": False,
        "synthetic_labels_created": False,
        "training_executed": False,
        "readiness_rerun_executed": False,
        "shadow_scorer_implemented": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "claim_causality": False,
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_ACTIVE_COVERAGE_ARCHIVE_SCHEMA_VERSION
                ),
                "policy": MIND_V3_FIRST_RECOVERY_ACTIVE_COVERAGE_ARCHIVE_POLICY,
            }
        ),
    }


def _source_integrity_from_v122_prerequisite(
    v122_report: Mapping[str, object],
    *,
    rollout_context_report_path: str | Path | None,
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
) -> dict[str, object]:
    v122_source = _mapping(v122_report.get("source_integrity"))
    failures = list(str(item) for item in v122_source.get("failures", ()))
    if v122_source.get("passed") is not True:
        failures.append("v122_prerequisite_source_integrity_failed")
    for field in NO_AUTHORIZATION_FIELDS:
        if _mapping(v122_report.get("recommendation")).get(field) is not False:
            failures.append(f"v122_{field}_not_false")
    if rollout_context_report_path is None or not Path(rollout_context_report_path).exists():
        failures.append("rollout_context_report_missing")
    if not _resolve_trajectory_paths(trajectory_paths, trajectory_glob_patterns):
        failures.append("trajectory_inputs_missing")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v122_prerequisite_source_integrity": dict(v122_source),
        "current_rare_support": v122_source.get("current_rare_support"),
        "v121_valid_candidate_counts": v122_source.get("v121_valid_candidate_counts"),
    }


def _final_source_integrity(
    source_integrity: Mapping[str, object],
    *,
    leakage: Mapping[str, object],
    v120_leakage: Mapping[str, object],
    strict_seed_leakage_count: int,
    replay: Mapping[str, object],
    v122_rerun_report: Mapping[str, object],
) -> dict[str, object]:
    failures = list(str(item) for item in source_integrity.get("failures", ()))
    if replay.get("replay_verified") is not True:
        failures.append("replay_verification_failed")
    if int(replay.get("missing_replay_verification_count", 0)) != 0:
        failures.append("replay_verification_missing")
    if int(replay.get("replay_verification_skipped_count", 0)) != 0:
        failures.append("replay_verification_skipped")
    if int(replay.get("heuristic_action_source_count", 0)) != 0:
        failures.append("heuristic_action_source_nonzero")
    if leakage.get("leakage_detected") is True:
        failures.append("trainable_public_input_leakage")
    if int(v120_leakage.get("split_key_leak_count", 0)) != 0:
        failures.append("trainable_split_key_leakage")
    if int(v120_leakage.get("forbidden_metadata_key_count", 0)) != 0:
        failures.append("trainable_metadata_leakage")
    if strict_seed_leakage_count != 0:
        failures.append("strict_seed_leakage")
    if _mapping(v122_rerun_report.get("source_integrity")).get("passed") is not True:
        failures.append("v122_rerun_source_integrity_failed")
    return {
        **dict(source_integrity),
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v122_rerun_source_integrity": _mapping(
            v122_rerun_report.get("source_integrity")
        ),
    }


def _search_config(
    *,
    rollout_context_report_path: str | Path | None,
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
    max_trajectories: int,
    max_source_records_per_action: int,
    max_evaluated_branches: int,
    verify_replay: bool,
) -> dict[str, object]:
    paths = _resolve_trajectory_paths(trajectory_paths, trajectory_glob_patterns)
    return {
        "target_actions": list(TARGET_ACTIONS),
        "target_additional_needed": {
            action: RARE_ACTION_ADDITIONAL_NEEDED[action] for action in TARGET_ACTIONS
        },
        "current_rare_support": dict(EXPECTED_CURRENT_RARE_SUPPORT),
        "rollout_context_report_path": (
            str(rollout_context_report_path)
            if rollout_context_report_path is not None
            else None
        ),
        "trajectory_paths": [str(path) for path in paths[: max(0, max_trajectories)]],
        "trajectory_globs": [str(pattern) for pattern in trajectory_glob_patterns],
        "max_trajectories": max(0, int(max_trajectories)),
        "max_source_records_per_action": max(1, int(max_source_records_per_action)),
        "max_evaluated_branches": max(0, int(max_evaluated_branches)),
        "verify_replay": bool(verify_replay),
        "search_policy": (
            "deterministic_public_action_mask_attack_legal_records_then_branch_replay"
        ),
        "selection_authorized": False,
    }


def _generate_active_archive_rows(
    *,
    search_config: Mapping[str, object],
    v119_report: Mapping[str, object] | None,
    v119_report_path: str | Path | None,
    manifest_rows: Sequence[Mapping[str, object]] | None,
    manifest_path: str | Path | None,
    v120_report: Mapping[str, object] | None,
    v120_report_path: str | Path | None,
    v121_report: Mapping[str, object] | None,
    v121_report_path: str | Path | None,
) -> _GenerationResult:
    paths = [Path(path) for path in search_config.get("trajectory_paths", ())]
    rollout_context_report_path = search_config.get("rollout_context_report_path")
    if not isinstance(rollout_context_report_path, str):
        return _empty_generation()
    template = load_mind_v3_founder_template(rollout_context_report_path)
    source_records, path_metadata, source_counts = _select_source_records(
        paths,
        max_per_action=int(search_config.get("max_source_records_per_action", 1)),
    )
    accepted_rows: list[dict[str, object]] = []
    accepted_results: list[dict[str, object]] = []
    accepted_branch_ids: list[str] = []
    rejection_counts: dict[str, Counter[str]] = {
        action: Counter() for action in TARGET_ACTIONS
    }
    rejection_examples: dict[str, dict[str, list[dict[str, object]]]] = {
        action: defaultdict(list) for action in TARGET_ACTIONS
    }
    materialization_failures: list[dict[str, object]] = []
    evaluated = 0
    found = {action: 0 for action in TARGET_ACTIONS}
    max_evaluated = int(search_config.get("max_evaluated_branches", 0))
    for record in source_records:
        action = str(record.get("target_action"))
        if found.get(action, 0) >= RARE_ACTION_ADDITIONAL_NEEDED[action]:
            continue
        if evaluated >= max_evaluated:
            break
        evaluated += 1
        rows, results, branch_id, rejection, failures = _evaluate_source_record(
            record,
            path_metadata=path_metadata,
            founder_template=template,
            branch_index=evaluated - 1,
            verify_replay=bool(search_config.get("verify_replay", True)),
            search_config=search_config,
            v119_report=v119_report,
            v119_report_path=v119_report_path,
            manifest_rows=manifest_rows,
            manifest_path=manifest_path,
            v120_report=v120_report,
            v120_report_path=v120_report_path,
            v121_report=v121_report,
            v121_report_path=v121_report_path,
        )
        materialization_failures.extend(failures)
        if rejection is not None:
            rejection_counts[action][rejection] += 1
            _add_rejection_example(rejection_examples[action], rejection, record)
            continue
        accepted_rows.extend(rows)
        accepted_results.extend(results)
        accepted_branch_ids.append(branch_id)
        found[action] += 1
        if all(
            found[target] >= RARE_ACTION_ADDITIONAL_NEEDED[target]
            for target in TARGET_ACTIONS
        ):
            break
    return _GenerationResult(
        archive_rows=tuple(accepted_rows),
        branch_results=tuple(accepted_results),
        accepted_branch_ids=tuple(accepted_branch_ids),
        search_budget_used={
            "trajectory_count": len(paths),
            "source_record_counts": dict(source_counts),
            "selected_source_record_count": len(source_records),
            "evaluated_branch_count": evaluated,
            "max_evaluated_branches": max_evaluated,
            "accepted_branch_count": len(accepted_branch_ids),
            "budget_exhausted": evaluated >= max_evaluated
            and not all(
                found[action] >= RARE_ACTION_ADDITIONAL_NEEDED[action]
                for action in TARGET_ACTIONS
            ),
        },
        source_record_counts=dict(source_counts),
        rejection_counts={
            action: dict(sorted(counter.items()))
            for action, counter in rejection_counts.items()
        },
        rejection_examples={
            action: {reason: examples for reason, examples in per_action.items()}
            for action, per_action in rejection_examples.items()
        },
        materialization_failures=tuple(materialization_failures),
    )


def _select_source_records(
    paths: Sequence[Path],
    *,
    max_per_action: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], Counter[str]]:
    selected: dict[str, list[dict[str, object]]] = {action: [] for action in TARGET_ACTIONS}
    path_metadata: dict[str, dict[str, object]] = {}
    counts: Counter[str] = Counter()
    for path in paths:
        seed = _seed_from_path(str(path))
        record_index = 0
        with _open_trajectory(path) as handle:
            header = json.loads(next(handle))
            ticks = _mapping(header.get("config")).get("max_ticks")
            path_metadata[str(path)] = {"max_ticks": ticks}
            for line in handle:
                payload = json.loads(line)
                record = payload.get("record") if isinstance(payload, Mapping) else None
                if not isinstance(record, Mapping):
                    continue
                mask = _mapping(record.get("action_mask"))
                resolution_mask = _mapping(record.get("resolution_action_mask"))
                for action in TARGET_ACTIONS:
                    if mask.get(action) is True and resolution_mask.get(action) is True:
                        counts[action] += 1
                        if len(selected[action]) < max_per_action:
                            selected[action].append(
                                _source_record_from_trajectory(
                                    path=path,
                                    seed=seed,
                                    ticks=ticks,
                                    record_index=record_index,
                                    record=record,
                                    target_action=action,
                                )
                            )
                record_index += 1
    merged: list[dict[str, object]] = []
    for offset in range(max_per_action):
        for action in TARGET_ACTIONS:
            if offset < len(selected[action]):
                merged.append(selected[action][offset])
    return merged, path_metadata, counts


def _source_record_from_trajectory(
    *,
    path: Path,
    seed: int,
    ticks: object,
    record_index: int,
    record: Mapping[str, object],
    target_action: str,
) -> dict[str, object]:
    return {
        "path": str(path),
        "source_kind": _source_kind(str(path)),
        "seed": seed,
        "ticks": ticks,
        "recovery_tick": record.get("tick"),
        "recovery_record_index": record_index,
        "agent_id": record.get("agent_id"),
        "requested_action": record.get("requested_action"),
        "observation_digest": record.get("observation_digest"),
        "action_mask": record.get("action_mask"),
        "resolution_action_mask": record.get("resolution_action_mask"),
        "before": record.get("before"),
        "target_action": target_action,
    }


def _evaluate_source_record(
    record: Mapping[str, object],
    *,
    path_metadata: Mapping[str, Mapping[str, object]],
    founder_template: object,
    branch_index: int,
    verify_replay: bool,
    search_config: Mapping[str, object],
    v119_report: Mapping[str, object] | None,
    v119_report_path: str | Path | None,
    manifest_rows: Sequence[Mapping[str, object]] | None,
    manifest_path: str | Path | None,
    v120_report: Mapping[str, object] | None,
    v120_report_path: str | Path | None,
    v121_report: Mapping[str, object] | None,
    v121_report_path: str | Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str, str | None, list[dict[str, object]]]:
    points, failures = _materialize_branch_targets(
        [record],
        path_metadata=path_metadata,
        founder_template=founder_template,
    )
    action = str(record.get("target_action"))
    if failures or not points:
        return [], [], "", "materialization_failed", [dict(item) for item in failures]
    point = points[0]
    branch_id = _active_branch_id(point, action=action, branch_index=branch_index)
    point = replace(
        point,
        branch_id=branch_id,
        branch_state_digest=_branch_state_digest(
            point.world,
            branch_id=branch_id,
            branch_tick=point.branch_tick,
        ),
    )
    try:
        result = _evaluate_branch_target(point, verify_replay=verify_replay)
    except (FirstRecoveryBranchOracleAuditError, ValueError) as exc:
        return [], [], branch_id, type(exc).__name__, []
    rows = build_first_recovery_archive_rows(
        [result],
        branch_points_by_id={point.branch_id: point},
    )
    v122 = build_first_recovery_rare_attack_coverage_collection(
        v119_report=v119_report,
        v119_report_path=v119_report_path,
        manifest_rows=manifest_rows,
        manifest_path=manifest_path,
        v120_report=v120_report,
        v120_report_path=v120_report_path,
        v121_report=v121_report,
        v121_report_path=v121_report_path,
        candidate_archive_report={
            "schema_version": MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION
        },
        candidate_archive_rows=rows,
        target_actions=(action,),
        max_branches=1,
        max_candidates_per_action=1,
    )
    if _mapping(v122.report.get("found_candidate_counts")).get(action) == 1:
        return rows, [result], branch_id, None, []
    rejection = _first_rejection(v122.report, action)
    return [], [result], branch_id, rejection, []


def _active_branch_id(point: object, *, action: str, branch_index: int) -> str:
    return (
        f"first-recovery-active-coverage-seed-{int(point.seed)}-"
        f"target-{_safe_part(action)}-branch-{int(branch_index)}-"
        f"tick-{int(point.branch_tick)}-agent-{int(point.agent_id)}-"
        f"logged-{_safe_part(point.logged_action)}"
    )


def _first_rejection(report: Mapping[str, object], action: str) -> str:
    counts = _mapping(_mapping(report.get("rejection_counts")).get(action))
    if counts:
        return sorted(counts, key=lambda key: (-int(counts[key]), str(key)))[0]
    return "not_candidate"


def _empty_generation() -> _GenerationResult:
    return _GenerationResult(
        archive_rows=(),
        branch_results=(),
        accepted_branch_ids=(),
        search_budget_used={
            "trajectory_count": 0,
            "source_record_counts": {},
            "selected_source_record_count": 0,
            "evaluated_branch_count": 0,
            "max_evaluated_branches": 0,
            "accepted_branch_count": 0,
            "budget_exhausted": False,
        },
        source_record_counts={},
        rejection_counts={action: {} for action in TARGET_ACTIONS},
        rejection_examples={action: {} for action in TARGET_ACTIONS},
        materialization_failures=(),
    )


def _replay_verification(branch_results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    run_count = 0
    failed = 0
    missing = 0
    skipped = 0
    heuristic_count = 0
    for result in branch_results:
        for run in result.get("action_runs", ()):
            if not isinstance(run, Mapping):
                continue
            run_count += 1
            replay = _mapping(run.get("replay_verification"))
            if "replay_verification" not in run:
                missing += 1
            elif run.get("replay_verification") is None:
                skipped += 1
            elif replay.get("verified") is not True:
                failed += 1
            heuristic_count += int(run.get("heuristic_action_source_count", 0))
    return {
        "run_count": run_count,
        "replay_verified": run_count > 0 and failed == 0 and missing == 0 and skipped == 0,
        "replay_verification_failure_count": failed,
        "missing_replay_verification_count": missing,
        "replay_verification_skipped_count": skipped,
        "heuristic_action_source_count": heuristic_count,
    }


def _classification(
    source_integrity: Mapping[str, object],
    found_counts: Mapping[str, int],
) -> dict[str, object]:
    labels = ["diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
    if source_integrity.get("passed") is not True:
        primary = "active_coverage_source_integrity_failed"
    else:
        found = sum(
            1
            for action in TARGET_ACTIONS
            if int(found_counts.get(action, 0)) >= RARE_ACTION_ADDITIONAL_NEEDED[action]
        )
        if found == len(TARGET_ACTIONS):
            primary = "active_coverage_rare_attacks_found"
        elif found:
            primary = "active_coverage_partial_rare_attack_found"
        else:
            primary = "active_coverage_no_rare_attack_found"
    labels.insert(0, primary)
    return {
        "primary": primary,
        "labels": [label for label in labels if label in ALLOWED_CLASSIFICATIONS],
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(
    classification: Mapping[str, object],
    v122_report: Mapping[str, object],
) -> dict[str, object]:
    would_clear = bool(
        _mapping(v122_report.get("recommendation")).get(
            "would_clear_v120_rare_action_limitation_if_accepted",
            False,
        )
    )
    if classification.get("primary") == "active_coverage_rare_attacks_found":
        next_step = "review_diagnostics_only_v123_archive_before_any_contract_update"
    elif classification.get("primary") == "active_coverage_source_integrity_failed":
        next_step = "source_integrity_must_pass_before_active_coverage_use"
    else:
        next_step = "increase_bounded_active_coverage_budget_or_collect_new_diagnostics"
    return {
        "next_step": next_step,
        "would_clear_v120_rare_action_limitation_if_accepted": would_clear,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "viewer_change_recommended": False,
        "replay_golden_change_recommended": False,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "claim_causality": False,
    }


def _resolve_trajectory_paths(
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
) -> tuple[Path, ...]:
    paths = {Path(path) for path in trajectory_paths}
    for pattern in trajectory_glob_patterns:
        paths.update(Path(path) for path in glob.glob(str(pattern)))
    return tuple(sorted(paths, key=lambda path: str(path)))


def _add_rejection_example(
    examples: dict[str, list[dict[str, object]]],
    reason: str,
    record: Mapping[str, object],
) -> None:
    if len(examples[reason]) >= MAX_EXAMPLES:
        return
    examples[reason].append(
        {
            "target_action": record.get("target_action"),
            "source_path": record.get("path"),
            "seed": record.get("seed"),
            "tick": record.get("recovery_tick"),
            "record_index": record.get("recovery_record_index"),
            "agent_id": record.get("agent_id"),
            "logged_action": record.get("requested_action"),
        }
    )


def _strict_seed_leakage_count(rows: Sequence[Mapping[str, object]]) -> int:
    count = 0
    for row in rows:
        count += _key_token_count(row.get("trainable_public_input"), "seed")
    return count


def _key_token_count(value: object, token: str) -> int:
    if isinstance(value, Mapping):
        total = 0
        for key, item in value.items():
            total += int(token in str(key).lower().split("_"))
            total += _key_token_count(item, token)
        return total
    if isinstance(value, list):
        return sum(_key_token_count(item, token) for item in value)
    return 0


def _open_trajectory(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _safe_part(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
