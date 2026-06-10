from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind import evaluation_harness as evaluate_cli
from evolution_sim.env import SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.broad_regression_branch_intervention import (
    BroadRegressionBranchPoint,
    _evaluate_branch_point as _v143_evaluate_branch_point,
    _optional_string,
)
from evolution_sim.mind.candidate_campaign import (
    DEFAULT_MIN_SAFE_LABEL_COUNT,
    M3_SAFE_ARCHIVE_EXPANSION_DATASET_ROW_SCHEMA_VERSION,
    CandidateCampaignError,
    _all_valid_actions_evaluated,
    _blacklisted,
    _bool_action_mask,
    _floor,
    _int,
    _list,
    _list_of_mappings,
    _mapping,
    _missing_valid_action_detail,
    _safe_archive_expansion_branch_result_matches_point,
    _safe_archive_expansion_dataset_row,
    _safe_archive_expansion_label_vet,
    _safe_archive_expansion_reference_from_world,
    _v145_blacklist,
    load_safe_archive_expansion_branch_result_chunks,
    safe_archive_expansion_leakage_scan,
    write_safe_archive_expansion_dataset,
    write_safe_archive_expansion_branch_result_chunk,
)
from evolution_sim.mind.carrion_branch_explore import _branch_state_digest
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_SCHEMA_VERSION = (
    "m3_carrion_specific_archive_expansion_report_v1"
)
M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_specific_archive_expansion_v1"
)
M3_CARRION_SPECIFIC_ARCHIVE_BRANCH_POLICY = (
    "carrion_contact_hydration_stall_terminal_branch_selection_v1"
)
M3_CARRION_SPECIFIC_ARCHIVE_FEATURE_POLICY = (
    "public_observation_input_and_public_action_mask_v1"
)

TARGET_CARRION_SEEDS = (13, 19, 29, 37, 41, 43)
CARRION_BRANCH_REASONS = (
    "carrion_contact",
    "post_carrion_hydration_risk",
    "movement_stall",
    "terminal_extinction",
)
DEFAULT_TICKS = 120
DEFAULT_MAX_BRANCH_POINTS_PER_SEED = len(CARRION_BRANCH_REASONS)
DEFAULT_MAX_CANDIDATE_ACTIONS = 0
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v148-carrion-specific-archive-expansion-report.json"
)
DEFAULT_DATASET_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v148-carrion-specific-archive-expansion-dataset.jsonl"
)
DEFAULT_BRANCH_RESULT_CHUNK_DIR = Path(
    "output/mind/mind-v3-v148-carrion-specific-archive-expansion-chunks"
)
EXTRA_FORBIDDEN_TRAINABLE_TOKENS = (
    "private",
    "future",
    "outcome",
)


class CarrionSpecificArchiveExpansionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CarrionSpecificBranchPoint:
    point: BroadRegressionBranchPoint
    branch_reason: str
    reason_rank: int
    reason_evidence: dict[str, object]


def write_carrion_specific_archive_expansion_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_carrion_specific_archive_expansion_dataset(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> None:
    write_safe_archive_expansion_dataset(rows, output_path)


def load_carrion_specific_branch_results(path: str | Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, dict):
        value = payload.get("branch_results")
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    raise CarrionSpecificArchiveExpansionError(
        f"branch result input must be a JSON list or object with branch_results: {path}"
    )


def build_carrion_specific_archive_expansion_report(
    *,
    branch_results: Sequence[Mapping[str, object]],
    v145_report: Mapping[str, object] | None = None,
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
    min_safe_label_count: int = DEFAULT_MIN_SAFE_LABEL_COUNT,
    generation_status: Mapping[str, object] | None = None,
    generation_evidence: Mapping[str, object] | None = None,
    input_paths: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    resolved_branch_results = [dict(result) for result in branch_results]
    resolved_target_seeds = tuple(int(seed) for seed in target_seeds)
    status = dict(generation_status or _precomputed_generation_status(resolved_branch_results))
    evidence = dict(generation_evidence or {})
    report_ticks = _report_ticks(
        branch_results=resolved_branch_results,
        status=status,
        evidence=evidence,
    )
    shard_id = _report_shard_id(status=status, evidence=evidence)
    blacklist = _v145_blacklist(v145_report or {})
    rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    safe_action_counts: Counter[str] = Counter()
    safe_action_counts_by_seed: dict[int, Counter[str]] = {
        int(seed): Counter() for seed in resolved_target_seeds
    }

    for result_index, result in enumerate(resolved_branch_results):
        for action_run_index, run in enumerate(_list_of_mappings(result.get("action_runs"))):
            action = str(run.get("forced_action", ""))
            identity = _label_identity(
                result_index=result_index,
                action_run_index=action_run_index,
                result=result,
                run=run,
            )
            vet = _safe_archive_expansion_label_vet(run)
            if vet.get("passed") is not True:
                excluded_rows.append(
                    {
                        **identity,
                        "excluded_reason": "safety_vet_failed",
                        "safety_vet": vet,
                    }
                )
                continue
            if _blacklisted(identity, blacklist, allow_label_source_row_index=False):
                excluded_rows.append(
                    {
                        **identity,
                        "excluded_reason": "invalid_resolution_risk_blacklist",
                        "safety_vet": vet,
                    }
                )
                continue
            row = _safe_archive_expansion_dataset_row(
                row_index=len(rows),
                source_branch_result_index=result_index,
                result=result,
                selected_run=run,
                safety_vet=vet,
            )
            _normalize_trainable_policy(row)
            rows.append(row)
            safe_action_counts.update([action])
            seed = _int(result.get("seed"), default=-1)
            if seed in safe_action_counts_by_seed:
                safe_action_counts_by_seed[seed].update([action])

    dataset_scan = carrion_specific_archive_leakage_scan(
        rows,
        strict_heldout_seeds=resolved_target_seeds,
    )
    replay = _replay_verification_report(resolved_branch_results)
    action_coverage = _action_coverage_report(resolved_branch_results)
    heuristic_action_source_count = _heuristic_action_source_count(
        resolved_branch_results
    )
    per_seed_support = _per_seed_support(
        target_seeds=resolved_target_seeds,
        branch_results=resolved_branch_results,
        rows=rows,
    )
    branch_reason_support = _branch_reason_support(resolved_branch_results)
    dominant = _dominant_count_share(safe_action_counts)
    source_integrity = _source_integrity(
        status=status,
        branch_results=resolved_branch_results,
        replay=replay,
        action_coverage=action_coverage,
        dataset_scan=dataset_scan,
        heuristic_action_source_count=heuristic_action_source_count,
    )
    support_floors = _support_floors(
        source_integrity=source_integrity,
        safe_label_count=len(rows),
        min_safe_label_count=int(min_safe_label_count),
        dominant_safe_label_action_share=float(dominant["share"]),
        per_seed_support=per_seed_support,
        branch_reason_support=branch_reason_support,
    )
    classification = _classification(
        source_integrity=source_integrity,
        support_floors=support_floors,
        safe_label_count=len(rows),
        min_safe_label_count=int(min_safe_label_count),
        per_seed_support=per_seed_support,
    )
    contract = {
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "gate_relaxation": False,
        "trainer_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "branch_fixture": "carrion_only",
        "branch_seed_policy": "explicit_carrion_only_seeds_13_19_29_37_41_43",
        "branch_point_policy": M3_CARRION_SPECIFIC_ARCHIVE_BRANCH_POLICY,
        "candidate_action_policy": "all_currently_valid_public_action_mask_actions",
        "safe_label_policy": "all_safety_vetted_public_mask_action_runs",
        "blacklist_policy": "exclude_v145_invalid_resolution_risk_label_sources",
        "trainable_fields": [
            "public observation_input",
            "public action_mask",
            "label action",
        ],
        "excluded_trainable_fields": [
            "seed",
            "fixture",
            "branch id",
            "tick",
            "agent id",
            "digests",
            "paths",
            "private state",
            "future outcomes",
        ],
    }
    input_payload = _input_path_payload(input_paths)
    report = {
        "schema_version": M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_SCHEMA_VERSION,
        "policy": M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_POLICY,
        "diagnostics_only": True,
        "contract": contract,
        "inputs": {
            **input_payload,
            "target_fixture": "carrion_only",
            "target_carrion_seeds": [int(seed) for seed in resolved_target_seeds],
            "ticks": report_ticks,
            "min_safe_label_count": int(min_safe_label_count),
            "shard_id": shard_id,
        },
        "generation_status": status,
        "generation_evidence": evidence,
        "blacklist": {
            "policy": "v145_invalid_resolution_risk_blacklist_v1",
            "blacklist_count": len(blacklist),
        },
        "source_integrity": source_integrity,
        "safe_label_count": len(rows),
        "per_seed_support": per_seed_support,
        "branch_reason_support": branch_reason_support,
        "action_distribution": {
            "safe_label_action_counts": dict(sorted(safe_action_counts.items())),
            "safe_label_action_counts_by_seed": {
                str(seed): dict(sorted(counts.items()))
                for seed, counts in sorted(safe_action_counts_by_seed.items())
            },
            "dominant_safe_label_action": dominant["key"],
            "dominant_safe_label_action_count": dominant["count"],
            "dominant_safe_label_action_share": dominant["share"],
        },
        "replay_verification": replay,
        "action_coverage": action_coverage,
        "leakage_scan": dataset_scan,
        "dataset": {
            "row_schema_version": M3_SAFE_ARCHIVE_EXPANSION_DATASET_ROW_SCHEMA_VERSION,
            "feature_policy": M3_CARRION_SPECIFIC_ARCHIVE_FEATURE_POLICY,
            "safe_label_count": len(rows),
            "min_safe_label_count": int(min_safe_label_count),
            "safe_label_action_counts": dict(sorted(safe_action_counts.items())),
            "dominant_safe_label_action": dominant["key"],
            "dominant_safe_label_action_count": dominant["count"],
            "dominant_safe_label_action_share": dominant["share"],
            "leakage_scan": dataset_scan,
            "dataset_digest": stable_payload_digest(rows),
        },
        "excluded_row_count": len(excluded_rows),
        "excluded_rows": excluded_rows[:128],
        "support_floors": support_floors,
        "classification": {"primary": classification, "labels": [classification]},
        "authorization_block": {
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "runtime_policy_change_authorized": False,
        },
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "branch_results": resolved_branch_results,
        "branch_result_count": len(resolved_branch_results),
        "branch_evidence_digest": stable_payload_digest(resolved_branch_results),
        "non_promoted": True,
    }
    exact_payload = {
        "schema_version": report["schema_version"],
        "policy": report["policy"],
        "contract": contract,
        "target_carrion_seeds": [int(seed) for seed in resolved_target_seeds],
        "branch_evidence_digest": report["branch_evidence_digest"],
        "dataset_digest": report["dataset"]["dataset_digest"],
        "classification": report["classification"],
        "support_floors": report["support_floors"],
        "source_integrity": report["source_integrity"],
    }
    report["exact_digest"] = stable_payload_digest(exact_payload)
    report["provenance"] = {
        "contract_digest": stable_payload_digest(contract),
        "exact_digest_payload_policy": "stable_payload_digest_of_contract_sources_and_outcomes_v1",
        "exact_digest": report["exact_digest"],
    }
    return report, rows


def generate_carrion_specific_archive_branch_results(
    *,
    seeds: Sequence[int] = TARGET_CARRION_SEEDS,
    ticks: int = DEFAULT_TICKS,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    max_candidate_actions: int = DEFAULT_MAX_CANDIDATE_ACTIONS,
    shard_id: str | None = None,
    branch_index_include: Sequence[int] | None = None,
    verify_replay: bool = True,
    branch_result_chunk_dir: str | Path | None = None,
    resume_branch_results: bool = False,
    max_wall_seconds: float | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    started_at = time.monotonic()
    chunk_dir = Path(branch_result_chunk_dir) if branch_result_chunk_dir else None
    resumed = (
        load_safe_archive_expansion_branch_result_chunks(chunk_dir)
        if chunk_dir is not None and bool(resume_branch_results)
        else {}
    )
    selected_seeds = tuple(int(seed) for seed in seeds)
    selected_branch_indexes = _branch_index_include(branch_index_include)
    branch_results: list[dict[str, object]] = []
    seed_reports: list[dict[str, object]] = []
    generated_count = 0
    resumed_count = 0
    stop_reason: str | None = None
    for seed in selected_seeds:
        if _wall_budget_exhausted(started_at, max_wall_seconds):
            stop_reason = "max_wall_seconds_elapsed_before_seed"
            break
        points, reference, seed_report = materialize_carrion_specific_branch_points(
            seed=int(seed),
            ticks=int(ticks),
            max_branch_points=int(max_branch_points_per_seed),
        )
        points, seed_report = _filter_branch_points_by_index(
            points,
            seed_report=seed_report,
            branch_index_include=selected_branch_indexes,
        )
        seed_reports.append(seed_report)
        reference_runs = {int(seed): {"baseline": reference, "v142_override": reference}}
        for branch_point in points:
            if _wall_budget_exhausted(started_at, max_wall_seconds):
                stop_reason = "max_wall_seconds_elapsed_before_branch_point"
                break
            result, source = _evaluate_carrion_branch_point_checkpointed(
                branch_point,
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
            _emit_progress(
                progress_callback,
                {
                    "event": "branch_result",
                    "source": source,
                    "shard_id": shard_id,
                    "fixture": "carrion_only",
                    "seed": int(seed),
                    "branch_id": branch_point.point.branch_id,
                    "branch_point_index": int(branch_point.point.branch_index),
                    "branch_reason": branch_point.branch_reason,
                    "action_count": len(_list_of_mappings(result.get("action_runs"))),
                    "elapsed_seconds": _elapsed_seconds(started_at),
                },
            )
        if stop_reason is not None:
            break
    branch_point_count = sum(
        _int(report.get("branch_point_count")) for report in seed_reports
    )
    partial = stop_reason is not None or len(branch_results) < branch_point_count
    generation_status = {
        "policy": "m3_carrion_specific_archive_generation_status_v1",
        "state": "partial" if partial else "complete",
        "partial": bool(partial),
        "stop_reason": stop_reason,
        "elapsed_seconds": _elapsed_seconds(started_at),
        "max_wall_seconds": None if max_wall_seconds is None else float(max_wall_seconds),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in selected_seeds],
        "ticks": int(ticks),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "max_candidate_actions": (
            None if int(max_candidate_actions) == 0 else int(max_candidate_actions)
        ),
        "shard_id": shard_id,
        "branch_index_include": (
            None if selected_branch_indexes is None else [int(index) for index in selected_branch_indexes]
        ),
        "verify_replay": bool(verify_replay),
        "branch_point_count": int(branch_point_count),
        "branch_result_count": len(branch_results),
        "generated_branch_result_count": int(generated_count),
        "resumed_branch_result_count": int(resumed_count),
        "chunk_dir": None if chunk_dir is None else str(chunk_dir),
    }
    return {
        "policy": "m3_carrion_specific_archive_branch_generation_v1",
        "shard_id": shard_id,
        "branch_index_include": (
            None if selected_branch_indexes is None else [int(index) for index in selected_branch_indexes]
        ),
        "generation_status": generation_status,
        "seed_reports": seed_reports,
        "branch_results": branch_results,
        "branch_result_count": len(branch_results),
        "branch_evidence_digest": stable_payload_digest(branch_results),
    }


def merge_carrion_specific_archive_expansion_shards(
    *,
    shard_reports: Sequence[Mapping[str, object]],
    shard_report_paths: Sequence[str | Path] = (),
    shard_chunk_dirs: Sequence[str | Path] = (),
    v145_report: Mapping[str, object] | None = None,
    allow_partial_shard_evidence: bool = False,
    target_seeds: Sequence[int] | None = None,
    input_paths: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not shard_reports:
        raise CarrionSpecificArchiveExpansionError(
            "carrion-specific shard merge requires at least one shard report"
        )
    merged_by_branch_id: dict[str, dict[str, object]] = {}
    digests_by_branch_id: dict[str, str] = {}
    merge_identity: dict[str, object] | None = None
    merged_target_seed_set: set[int] = set()
    explicit_target_seed_set = (
        {int(seed) for seed in target_seeds} if target_seeds is not None else None
    )
    source_summaries: list[dict[str, object]] = []
    partial_sources: list[dict[str, object]] = []
    source_paths = [str(path) for path in shard_report_paths]

    for shard_index, report in enumerate(shard_reports):
        if report.get("schema_version") != M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_SCHEMA_VERSION:
            raise CarrionSpecificArchiveExpansionError(
                f"shard {shard_index} schema mismatch"
            )
        if report.get("policy") != M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_POLICY:
            raise CarrionSpecificArchiveExpansionError(
                f"shard {shard_index} policy mismatch"
            )
        identity = _carrion_shard_merge_identity(report)
        shard_target_seeds = {
            int(seed) for seed in _list(identity.get("target_carrion_seeds"))
        }
        if explicit_target_seed_set is not None and not shard_target_seeds.issubset(
            explicit_target_seed_set
        ):
            raise CarrionSpecificArchiveExpansionError(
                f"shard {shard_index} target seed outside requested merge seeds"
            )
        merged_target_seed_set.update(shard_target_seeds)
        comparable_identity = {
            key: value
            for key, value in identity.items()
            if key != "target_carrion_seeds"
        }
        if merge_identity is None:
            merge_identity = comparable_identity
        elif comparable_identity != merge_identity:
            raise CarrionSpecificArchiveExpansionError(
                f"shard {shard_index} target/tick/min-floor mismatch"
            )
        status = _mapping(report.get("generation_status"))
        source_integrity = _mapping(report.get("source_integrity"))
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
        status_is_partial = _status_partial(status)
        integrity_is_partial = "partial_branch_evidence" in set(integrity_failures)
        if source_integrity.get("passed") is not True:
            if non_partial_integrity_failures:
                raise CarrionSpecificArchiveExpansionError(
                    "shard "
                    f"{shard_index} source integrity failed: "
                    f"{non_partial_integrity_failures}"
                )
            if not (status_is_partial or integrity_is_partial):
                raise CarrionSpecificArchiveExpansionError(
                    f"shard {shard_index} source integrity failed without partial status"
                )
        source_name = _shard_source_name(report, shard_index=shard_index)
        if status_is_partial or integrity_is_partial:
            summary = {
                "source": source_name,
                "source_path": _optional_index(source_paths, shard_index),
                "shard_id": _shard_id_from_report(report),
                "state": status.get("state"),
                "stop_reason": status.get("stop_reason"),
                "source_integrity_failures": integrity_failures,
            }
            if not allow_partial_shard_evidence:
                raise CarrionSpecificArchiveExpansionError(
                    f"partial shard evidence requires explicit partial merge: {summary}"
                )
            partial_sources.append(summary)
        branch_results = _list_of_mappings(report.get("branch_results"))
        expected_digest = str(report.get("branch_evidence_digest", ""))
        actual_digest = stable_payload_digest(branch_results)
        if expected_digest and expected_digest != actual_digest:
            raise CarrionSpecificArchiveExpansionError(
                f"shard {shard_index} branch evidence digest mismatch"
            )
        for result in branch_results:
            _add_carrion_merged_branch_result(
                merged_by_branch_id=merged_by_branch_id,
                digests_by_branch_id=digests_by_branch_id,
                result=result,
                source=source_name,
            )
        source_summaries.append(
            {
                "source": source_name,
                "source_path": _optional_index(source_paths, shard_index),
                "shard_id": _shard_id_from_report(report),
                "branch_result_count": len(branch_results),
                "branch_evidence_digest": expected_digest or actual_digest,
                "partial": bool(status_is_partial or integrity_is_partial),
            }
        )

    for chunk_dir in shard_chunk_dirs:
        chunks = load_safe_archive_expansion_branch_result_chunks(chunk_dir)
        for result in chunks.values():
            branch_id = str(result.get("branch_id", ""))
            if branch_id not in merged_by_branch_id:
                raise CarrionSpecificArchiveExpansionError(
                    "chunk-dir branch result lacks matching shard report: "
                    f"{chunk_dir}:{branch_id}"
                )
            _add_carrion_merged_branch_result(
                merged_by_branch_id=merged_by_branch_id,
                digests_by_branch_id=digests_by_branch_id,
                result=result,
                source=f"chunk_dir:{chunk_dir}",
            )
        source_summaries.append(
            {
                "source": f"chunk_dir:{chunk_dir}",
                "source_path": str(chunk_dir),
                "branch_result_count": len(chunks),
                "partial": False,
            }
        )

    if not merged_by_branch_id:
        raise CarrionSpecificArchiveExpansionError(
            "carrion-specific shard merge has no branch results"
        )
    branch_results = sorted(
        merged_by_branch_id.values(),
        key=_carrion_branch_result_sort_key,
    )
    resolved_target_seeds = sorted(
        explicit_target_seed_set
        if explicit_target_seed_set is not None
        else merged_target_seed_set
    )
    min_safe_label_count = (
        _int(merge_identity.get("min_safe_label_count"))
        if merge_identity is not None
        else DEFAULT_MIN_SAFE_LABEL_COUNT
    )
    ticks = merge_identity.get("ticks") if merge_identity is not None else None
    generation_status = {
        "policy": "m3_carrion_specific_archive_merged_generation_status_v1",
        "state": "partial" if partial_sources else "complete",
        "partial": bool(partial_sources),
        "stop_reason": "partial_shard_evidence" if partial_sources else None,
        "source_count": len(source_summaries),
        "partial_sources": sorted(
            partial_sources,
            key=lambda item: str(item.get("source", "")),
        ),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in resolved_target_seeds],
        "ticks": ticks,
        "branch_result_count": len(branch_results),
        "branch_point_count": len(branch_results),
    }
    generation_evidence = {
        "policy": "m3_carrion_specific_archive_shard_merge_generation_evidence_v1",
        "merge_mode": True,
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "source_count": len(source_summaries),
        "branch_evidence_digest": stable_payload_digest(branch_results),
    }
    report, rows = build_carrion_specific_archive_expansion_report(
        branch_results=branch_results,
        v145_report=v145_report,
        target_seeds=[int(seed) for seed in resolved_target_seeds],
        min_safe_label_count=int(min_safe_label_count),
        generation_status=generation_status,
        generation_evidence=generation_evidence,
        input_paths=input_paths,
    )
    report["shard_merge"] = {
        "policy": "m3_carrion_specific_archive_shard_merge_v1",
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "sources": sorted(
            source_summaries,
            key=lambda item: (
                str(item.get("source", "")),
                str(item.get("source_path", "")),
                str(item.get("branch_evidence_digest", "")),
            ),
        ),
        "partial_sources": generation_status["partial_sources"],
        "duplicate_branch_id_policy": "same_digest_allowed_conflict_rejected",
    }
    return report, rows


def materialize_carrion_specific_branch_points(
    *,
    seed: int,
    ticks: int,
    max_branch_points: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
) -> tuple[list[CarrionSpecificBranchPoint], dict[str, object], dict[str, object]]:
    world = evaluate_cli._fixture_world(
        fixture_name="carrion_only",
        seed=int(seed),
        ticks=int(ticks),
        policy=MindV3EvolutionPolicy(seed=int(seed)),
    )
    from evolution_sim.mind.candidate_campaign import (
        _configure_branch_manual_summary_run,
    )

    _configure_branch_manual_summary_run(world)
    selected_reasons = CARRION_BRANCH_REASONS[: max(1, int(max_branch_points))]
    points_by_reason: dict[str, CarrionSpecificBranchPoint] = {}
    failures: list[dict[str, object]] = []
    carrion_contact_tick_by_agent: dict[int, int] = {}
    first_carrion_contact_tick: int | None = None
    used_records: set[tuple[int, int, int]] = set()
    record_index = 0
    ticks_executed = 0
    for tick in range(int(ticks)):
        world.tick = tick
        snapshot = deepcopy(world)
        world._run_tick()
        ticks_executed = tick + 1
        terminal_after_tick = not world.alive_agents()
        for record in list(world.tick_trajectory_records):
            agent_id = _int(record.get("agent_id"), default=-1)
            reasons = _branch_reasons_for_record(
                record=record,
                tick=int(tick),
                terminal_after_tick=terminal_after_tick,
                carrion_contact_tick_by_agent=carrion_contact_tick_by_agent,
                first_carrion_contact_tick=first_carrion_contact_tick,
            )
            for reason in selected_reasons:
                if reason not in reasons or reason in points_by_reason:
                    continue
                record_key = (int(tick), int(record_index), int(agent_id))
                if record_key in used_records:
                    continue
                point = _point_from_carrion_record(
                    fixture="carrion_only",
                    seed=int(seed),
                    ticks=int(ticks),
                    tick=int(tick),
                    record_index=int(record_index),
                    branch_index=selected_reasons.index(reason),
                    branch_reason=reason,
                    record=record,
                    snapshot=snapshot,
                )
                if isinstance(point, CarrionSpecificBranchPoint):
                    points_by_reason[reason] = point
                    used_records.add(record_key)
                else:
                    failures.append(point)
            if _record_is_carrion_contact(record):
                if agent_id >= 0:
                    carrion_contact_tick_by_agent.setdefault(agent_id, int(tick))
                if first_carrion_contact_tick is None:
                    first_carrion_contact_tick = int(tick)
            record_index += 1
        if all(reason in points_by_reason for reason in selected_reasons):
            break
        if not world.alive_agents():
            break
    while world.alive_agents() and ticks_executed < int(ticks):
        world.tick = ticks_executed
        world._run_tick()
        ticks_executed += 1
    ordered_points = [
        points_by_reason[reason] for reason in selected_reasons if reason in points_by_reason
    ]
    missing_reasons = [
        reason for reason in selected_reasons if reason not in points_by_reason
    ]
    reference = _safe_archive_expansion_reference_from_world(
        world,
        seed=int(seed),
        ticks=int(ticks),
        runtime="linear_mind_v3_carrion_specific_archive",
    )
    return (
        ordered_points,
        reference,
        {
            "policy": "m3_carrion_specific_archive_seed_materialization_v1",
            "fixture": "carrion_only",
            "seed": int(seed),
            "ticks_requested": int(ticks),
            "ticks_executed": int(ticks_executed),
            "branch_point_count": len(ordered_points),
            "branch_ids": [item.point.branch_id for item in ordered_points],
            "branch_reasons": [item.branch_reason for item in ordered_points],
            "required_branch_reasons": list(selected_reasons),
            "missing_branch_reasons": missing_reasons,
            "first_carrion_contact_tick": first_carrion_contact_tick,
            "failure_count": len(failures),
            "failures": failures[:12],
            "passed": not failures and not missing_reasons,
        },
    )


def carrion_specific_archive_leakage_scan(
    rows: Sequence[Mapping[str, object]],
    *,
    strict_heldout_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    shared = safe_archive_expansion_leakage_scan(
        rows,
        strict_heldout_seeds=strict_heldout_seeds,
    )
    extra_failures: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        trainable = _mapping(row.get("trainable"))
        for path, value in _flatten(trainable):
            lower_path = path.lower()
            if any(token in lower_path for token in EXTRA_FORBIDDEN_TRAINABLE_TOKENS):
                extra_failures.append(
                    {
                        "row_index": row_index,
                        "path": path,
                        "reason": "carrion_specific_forbidden_trainable_path_token",
                    }
                )
            if isinstance(value, str) and _looks_like_digest(value):
                extra_failures.append(
                    {
                        "row_index": row_index,
                        "path": path,
                        "reason": "carrion_specific_forbidden_digest_value",
                    }
                )
    failures = [
        *_list_of_mappings(shared.get("failures")),
        *extra_failures,
    ]
    return {
        **dict(shared),
        "policy": "m3_carrion_specific_archive_trainable_leakage_scan_v1",
        "passed": shared.get("passed") is True and not extra_failures,
        "shared_leakage_scan": shared,
        "extra_forbidden_tokens": list(EXTRA_FORBIDDEN_TRAINABLE_TOKENS),
        "extra_forbidden_failure_count": len(extra_failures),
        "failures": failures[:32],
    }


def _point_from_carrion_record(
    *,
    fixture: str,
    seed: int,
    ticks: int,
    tick: int,
    record_index: int,
    branch_index: int,
    branch_reason: str,
    record: Mapping[str, object],
    snapshot: SimulationWorld,
) -> CarrionSpecificBranchPoint | dict[str, object]:
    agent_id = _int(record.get("agent_id"), default=-1)
    if agent_id < 0:
        return {
            "seed": int(seed),
            "tick": int(tick),
            "record_index": int(record_index),
            "branch_reason": branch_reason,
            "reason": "missing_agent_id",
        }
    action_mask = _bool_action_mask(record.get("action_mask"))
    if not any(action_mask.values()):
        return {
            "seed": int(seed),
            "tick": int(tick),
            "record_index": int(record_index),
            "agent_id": int(agent_id),
            "branch_reason": branch_reason,
            "reason": "missing_public_action_mask",
        }
    requested = str(record.get("requested_action", ""))
    if requested not in ACTION_NAMES:
        return {
            "seed": int(seed),
            "tick": int(tick),
            "record_index": int(record_index),
            "agent_id": int(agent_id),
            "branch_reason": branch_reason,
            "reason": "missing_current_policy_requested_action",
        }
    observation_input = _mapping(record.get("observation_input"))
    if not observation_input:
        return {
            "seed": int(seed),
            "tick": int(tick),
            "record_index": int(record_index),
            "agent_id": int(agent_id),
            "branch_reason": branch_reason,
            "reason": "missing_public_observation_input",
        }
    slug = branch_reason.replace("_", "-")
    branch_id = (
        f"m3-carrion-specific-archive-seed-{seed}-{slug}-branch-"
        f"{branch_index}-tick-{tick}-agent-{agent_id}"
    )
    point = BroadRegressionBranchPoint(
        branch_id=branch_id,
        seed=int(seed),
        fixture=fixture,
        ticks=int(ticks),
        branch_tick=int(tick),
        record_index=int(record_index),
        branch_index=int(branch_index),
        agent_id=int(agent_id),
        baseline_action=requested,
        v142_requested_action=requested,
        v142_resolved_action=str(record.get("resolved_action", "")),
        action_mask=dict(action_mask),
        observation_input=dict(observation_input),
        observation_schema=_optional_string(record.get("observation_schema")),
        observation_digest=_optional_string(record.get("observation_digest")),
        source_trajectory_path=None,
        branch_state_digest=_branch_state_digest(
            snapshot,
            branch_id=branch_id,
            branch_tick=int(tick),
        ),
        world=deepcopy(snapshot),
    )
    return CarrionSpecificBranchPoint(
        point=point,
        branch_reason=branch_reason,
        reason_rank=int(branch_index),
        reason_evidence=_branch_reason_evidence(
            branch_reason=branch_reason,
            record=record,
            tick=int(tick),
        ),
    )


def _evaluate_carrion_branch_point_checkpointed(
    branch_point: CarrionSpecificBranchPoint,
    *,
    reference_runs: Mapping[int, Mapping[str, Mapping[str, object]]],
    max_candidate_actions: int,
    verify_replay: bool,
    chunk_dir: Path | None,
    resumed_branch_results: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], str]:
    point = branch_point.point
    context = _carrion_branch_context(branch_point)
    cached = resumed_branch_results.get(point.branch_id)
    if (
        cached is not None
        and _safe_archive_expansion_branch_result_matches_point(
            cached,
            point,
            max_candidate_actions=int(max_candidate_actions),
            verify_replay=bool(verify_replay),
        )
        and _mapping(cached.get("carrion_archive_context")) == context
    ):
        return dict(cached), "resumed"
    result = _v143_evaluate_branch_point(
        point,
        reference_runs=reference_runs,
        max_candidate_actions=int(max_candidate_actions),
        verify_replay=bool(verify_replay),
    )
    enriched = dict(result)
    enriched["carrion_archive_context"] = context
    if chunk_dir is not None:
        write_safe_archive_expansion_branch_result_chunk(enriched, chunk_dir)
    return enriched, "generated"


def _branch_reasons_for_record(
    *,
    record: Mapping[str, object],
    tick: int,
    terminal_after_tick: bool,
    carrion_contact_tick_by_agent: Mapping[int, int],
    first_carrion_contact_tick: int | None,
) -> set[str]:
    reasons: set[str] = set()
    agent_id = _int(record.get("agent_id"), default=-1)
    if _record_is_carrion_contact(record):
        reasons.add("carrion_contact")
    if _record_is_post_carrion_hydration_risk(
        record,
        tick=int(tick),
        agent_id=agent_id,
        carrion_contact_tick_by_agent=carrion_contact_tick_by_agent,
        first_carrion_contact_tick=first_carrion_contact_tick,
    ):
        reasons.add("post_carrion_hydration_risk")
    if _record_is_movement_stall(record):
        reasons.add("movement_stall")
    if terminal_after_tick and _record_is_terminal_death(record):
        reasons.add("terminal_extinction")
    return reasons


def _record_is_carrion_contact(record: Mapping[str, object]) -> bool:
    feeding = _mapping(_mapping(record.get("outcome")).get("feeding"))
    return bool(feeding.get("ate", False)) and str(feeding.get("food_source")) in {
        "carcass",
        "fresh_kill",
    }


def _record_is_post_carrion_hydration_risk(
    record: Mapping[str, object],
    *,
    tick: int,
    agent_id: int,
    carrion_contact_tick_by_agent: Mapping[int, int],
    first_carrion_contact_tick: int | None,
) -> bool:
    if first_carrion_contact_tick is None:
        return False
    same_agent_contact_tick = carrion_contact_tick_by_agent.get(int(agent_id))
    same_agent_after_contact = (
        same_agent_contact_tick is not None and int(tick) > int(same_agent_contact_tick)
    )
    global_post_contact = int(tick) > int(first_carrion_contact_tick)
    if not same_agent_after_contact and not global_post_contact:
        return False
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    hydration = min(
        _optional_ratio(before.get("hydration_ratio"), default=1.0),
        _optional_ratio(after.get("hydration_ratio"), default=1.0),
    )
    return hydration <= 0.55


def _record_is_movement_stall(record: Mapping[str, object]) -> bool:
    requested = str(record.get("requested_action", ""))
    resolved = str(record.get("resolved_action", ""))
    return requested.startswith("move_") and (
        record.get("moved") is False or resolved != requested
    )


def _record_is_terminal_death(record: Mapping[str, object]) -> bool:
    outcome = _mapping(record.get("outcome"))
    passive = _mapping(outcome.get("passive"))
    return bool(outcome.get("died", False)) or bool(
        passive.get("died_after_action", False)
    )


def _branch_reason_evidence(
    *,
    branch_reason: str,
    record: Mapping[str, object],
    tick: int,
) -> dict[str, object]:
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    outcome = _mapping(record.get("outcome"))
    feeding = _mapping(outcome.get("feeding"))
    passive = _mapping(outcome.get("passive"))
    return {
        "policy": M3_CARRION_SPECIFIC_ARCHIVE_BRANCH_POLICY,
        "branch_reason": branch_reason,
        "tick": int(tick),
        "requested_action": record.get("requested_action"),
        "resolved_action": record.get("resolved_action"),
        "moved": bool(record.get("moved", False)),
        "ate": bool(feeding.get("ate", False)),
        "food_source": feeding.get("food_source"),
        "hydration_ratio_before": _optional_ratio(before.get("hydration_ratio")),
        "hydration_ratio_after": _optional_ratio(after.get("hydration_ratio")),
        "energy_ratio_before": _optional_ratio(before.get("energy_ratio")),
        "energy_ratio_after": _optional_ratio(after.get("energy_ratio")),
        "death_cause": passive.get("death_cause"),
        "died_after_action": bool(passive.get("died_after_action", False)),
    }


def _carrion_branch_context(
    branch_point: CarrionSpecificBranchPoint,
) -> dict[str, object]:
    return {
        "policy": M3_CARRION_SPECIFIC_ARCHIVE_BRANCH_POLICY,
        "fixture": "carrion_only",
        "branch_reason": branch_point.branch_reason,
        "reason_rank": int(branch_point.reason_rank),
        "reason_evidence": dict(branch_point.reason_evidence),
    }


def _branch_index_include(
    branch_index_include: Sequence[int] | None,
) -> tuple[int, ...] | None:
    if branch_index_include is None:
        return None
    resolved = tuple(sorted({int(index) for index in branch_index_include}))
    invalid = [
        int(index)
        for index in resolved
        if int(index) < 0 or int(index) >= len(CARRION_BRANCH_REASONS)
    ]
    if invalid:
        raise CarrionSpecificArchiveExpansionError(
            f"invalid carrion branch indexes: {invalid}"
        )
    return resolved


def _filter_branch_points_by_index(
    points: Sequence[CarrionSpecificBranchPoint],
    *,
    seed_report: Mapping[str, object],
    branch_index_include: Sequence[int] | None,
) -> tuple[list[CarrionSpecificBranchPoint], dict[str, object]]:
    if branch_index_include is None:
        return list(points), dict(seed_report)
    include = {int(index) for index in branch_index_include}
    required_reasons = [CARRION_BRANCH_REASONS[index] for index in sorted(include)]
    filtered = [
        point for point in points if int(point.point.branch_index) in include
    ]
    branch_reasons = [point.branch_reason for point in filtered]
    failures = [
        dict(failure)
        for failure in _list_of_mappings(seed_report.get("failures"))
        if str(failure.get("branch_reason", "")) in set(required_reasons)
    ]
    missing_reasons = [
        reason for reason in required_reasons if reason not in set(branch_reasons)
    ]
    report = dict(seed_report)
    report.update(
        {
            "materialized_branch_point_count": _int(
                seed_report.get("branch_point_count")
            ),
            "branch_index_include": [int(index) for index in sorted(include)],
            "filtered_out_branch_point_count": max(
                0,
                _int(seed_report.get("branch_point_count")) - len(filtered),
            ),
            "branch_point_count": len(filtered),
            "branch_ids": [point.point.branch_id for point in filtered],
            "branch_reasons": branch_reasons,
            "required_branch_reasons": required_reasons,
            "missing_branch_reasons": missing_reasons,
            "failure_count": len(failures),
            "failures": failures[:12],
            "passed": not failures and not missing_reasons,
        }
    )
    return filtered, report


def _carrion_shard_merge_identity(report: Mapping[str, object]) -> dict[str, object]:
    inputs = _mapping(report.get("inputs"))
    status = _mapping(report.get("generation_status"))
    dataset = _mapping(report.get("dataset"))
    branch_results = _list_of_mappings(report.get("branch_results"))
    target_seeds = [
        int(seed)
        for seed in (
            _list(inputs.get("target_carrion_seeds"))
            or _list(status.get("target_carrion_seeds"))
            or _list(_mapping(report.get("per_seed_support")).get("target_seeds"))
        )
    ]
    ticks = (
        _optional_int(inputs.get("ticks"))
        if _optional_int(inputs.get("ticks")) is not None
        else _optional_int(status.get("ticks"))
    )
    if ticks is None:
        ticks = _common_result_ticks(branch_results)
    min_safe_label_count = (
        _optional_int(inputs.get("min_safe_label_count"))
        if _optional_int(inputs.get("min_safe_label_count")) is not None
        else _optional_int(dataset.get("min_safe_label_count"))
    )
    if inputs.get("target_fixture") != "carrion_only":
        raise CarrionSpecificArchiveExpansionError(
            "carrion-specific shard target fixture mismatch"
        )
    if not target_seeds:
        raise CarrionSpecificArchiveExpansionError(
            "carrion-specific shard missing target seed identity"
        )
    if ticks is None:
        raise CarrionSpecificArchiveExpansionError(
            "carrion-specific shard missing tick identity"
        )
    if min_safe_label_count is None:
        raise CarrionSpecificArchiveExpansionError(
            "carrion-specific shard missing min-floor identity"
        )
    return {
        "schema_version": report.get("schema_version"),
        "policy": report.get("policy"),
        "target_fixture": inputs.get("target_fixture"),
        "target_carrion_seeds": target_seeds,
        "ticks": int(ticks),
        "min_safe_label_count": int(min_safe_label_count),
    }


def _add_carrion_merged_branch_result(
    *,
    merged_by_branch_id: dict[str, dict[str, object]],
    digests_by_branch_id: dict[str, str],
    result: Mapping[str, object],
    source: str,
) -> None:
    branch_id = str(result.get("branch_id", ""))
    if not branch_id:
        raise CarrionSpecificArchiveExpansionError(
            f"merged carrion branch result missing branch_id: {source}"
        )
    if not _carrion_branch_result_has_replay_verification(result):
        raise CarrionSpecificArchiveExpansionError(
            "merged carrion branch result missing replay verification: "
            f"{source}:{branch_id}"
        )
    digest = stable_payload_digest(result)
    prior_digest = digests_by_branch_id.get(branch_id)
    if prior_digest is not None and prior_digest != digest:
        raise CarrionSpecificArchiveExpansionError(
            f"duplicate branch_id with different digest: {branch_id}"
        )
    merged_by_branch_id[branch_id] = dict(result)
    digests_by_branch_id[branch_id] = digest


def _carrion_branch_result_has_replay_verification(
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


def _carrion_branch_result_sort_key(
    result: Mapping[str, object],
) -> tuple[int, int, int, int, str]:
    fixture = str(result.get("fixture", ""))
    fixture_order = 0 if fixture == "carrion_only" else 1
    return (
        fixture_order,
        _int(result.get("seed")),
        _int(result.get("branch_index")),
        _int(result.get("branch_tick")),
        str(result.get("branch_id", "")),
    )


def _shard_source_name(
    report: Mapping[str, object],
    *,
    shard_index: int,
) -> str:
    shard_id = _shard_id_from_report(report)
    if shard_id is not None:
        return f"report:{shard_id}"
    branch_results = _list_of_mappings(report.get("branch_results"))
    digest = stable_payload_digest(branch_results)[:12]
    if digest:
        return f"report:{digest}"
    return f"report:{shard_index}"


def _shard_id_from_report(report: Mapping[str, object]) -> str | None:
    inputs = _mapping(report.get("inputs"))
    status = _mapping(report.get("generation_status"))
    evidence = _mapping(report.get("generation_evidence"))
    return _report_shard_id(status=status, evidence=evidence) or _optional_string(
        inputs.get("shard_id")
    )


def _optional_index(values: Sequence[str], index: int) -> str | None:
    if 0 <= int(index) < len(values):
        return values[int(index)]
    return None


def _label_identity(
    *,
    result_index: int,
    action_run_index: int,
    result: Mapping[str, object],
    run: Mapping[str, object],
) -> dict[str, object]:
    return {
        "row_index": int(result_index),
        "source_branch_result_index": int(result_index),
        "source_action_run_index": int(action_run_index),
        "seed": _int(result.get("seed")),
        "fixture": result.get("fixture"),
        "branch_id": result.get("branch_id"),
        "branch_tick": _int(result.get("branch_tick")),
        "agent_id": _int(result.get("agent_id")),
        "label_action": run.get("forced_action"),
    }


def _normalize_trainable_policy(row: dict[str, object]) -> None:
    trainable = _mapping(row.get("trainable"))
    if isinstance(trainable, dict):
        trainable["feature_policy"] = M3_CARRION_SPECIFIC_ARCHIVE_FEATURE_POLICY


def _source_integrity(
    *,
    status: Mapping[str, object],
    branch_results: Sequence[Mapping[str, object]],
    replay: Mapping[str, object],
    action_coverage: Mapping[str, object],
    dataset_scan: Mapping[str, object],
    heuristic_action_source_count: int,
) -> dict[str, object]:
    failures: list[str] = []
    if _status_partial(status):
        failures.append("partial_branch_evidence")
    if not branch_results:
        failures.append("no_carrion_branch_results")
    if replay.get("complete") is not True:
        failures.append("replay_verification_incomplete")
    if action_coverage.get("complete") is not True:
        failures.append("not_all_valid_candidate_actions_evaluated")
    if dataset_scan.get("passed") is not True:
        failures.append("trainable_leakage_detected")
    if int(heuristic_action_source_count) != 0:
        failures.append("heuristic_action_source_count_nonzero")
    return {
        "policy": "m3_carrion_specific_archive_source_integrity_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "generation_state": status.get("state"),
        "branch_result_count": len(branch_results),
        "replay_verification_complete": replay.get("complete") is True,
        "action_coverage_complete": action_coverage.get("complete") is True,
        "leakage_scan_passed": dataset_scan.get("passed") is True,
        "heuristic_action_source_count": int(heuristic_action_source_count),
    }


def _support_floors(
    *,
    source_integrity: Mapping[str, object],
    safe_label_count: int,
    min_safe_label_count: int,
    dominant_safe_label_action_share: float,
    per_seed_support: Mapping[str, object],
    branch_reason_support: Mapping[str, object],
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
            observed=int(safe_label_count),
            required=int(min_safe_label_count),
        ),
        _floor(
            "dominant_safe_label_action_share_lte_0_50",
            float(dominant_safe_label_action_share) <= 0.50,
            observed=_round(dominant_safe_label_action_share),
            required=0.50,
        ),
        _floor(
            "all_target_seeds_have_branch_results",
            not _list(per_seed_support.get("missing_branch_result_seeds")),
            observed=per_seed_support.get("missing_branch_result_seeds"),
            required=[],
            fixture="carrion_only",
        ),
        _floor(
            "all_target_seeds_have_safe_labels",
            not _list(per_seed_support.get("missing_safe_label_seeds")),
            observed=per_seed_support.get("missing_safe_label_seeds"),
            required=[],
            fixture="carrion_only",
        ),
        _floor(
            "all_branch_reason_targets_observed",
            not _list(branch_reason_support.get("missing_branch_reasons")),
            observed=branch_reason_support.get("missing_branch_reasons"),
            required=[],
            fixture="carrion_only",
        ),
    ]
    first_failed = next((floor for floor in floors if floor["passed"] is not True), None)
    return {
        "policy": "m3_carrion_specific_archive_support_floors_v1",
        "passed": first_failed is None,
        "first_failed_floor": None if first_failed is None else first_failed["name"],
        "floors": floors,
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    support_floors: Mapping[str, object],
    safe_label_count: int,
    min_safe_label_count: int,
    per_seed_support: Mapping[str, object],
) -> str:
    if "partial_branch_evidence" in set(_list(source_integrity.get("failures"))):
        return "carrion_specific_archive_partial_no_training"
    if source_integrity.get("passed") is not True:
        return "carrion_specific_archive_source_integrity_failed_no_training"
    if int(safe_label_count) < int(min_safe_label_count):
        return "archive_support_insufficient"
    if _list(per_seed_support.get("missing_safe_label_seeds")):
        return "carrion_specific_archive_missing_seed_support_no_training"
    if support_floors.get("passed") is True:
        return "carrion_specific_archive_support_ready_no_training"
    return "carrion_specific_archive_blocked_no_training"


def _per_seed_support(
    *,
    target_seeds: Sequence[int],
    branch_results: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    branch_counts = Counter(_int(result.get("seed")) for result in branch_results)
    safe_counts = Counter(
        _int(_mapping(row.get("metadata")).get("seed")) for row in rows
    )
    branch_reasons_by_seed: dict[int, Counter[str]] = {
        int(seed): Counter() for seed in target_seeds
    }
    safe_action_counts_by_seed: dict[int, Counter[str]] = {
        int(seed): Counter() for seed in target_seeds
    }
    for result in branch_results:
        seed = _int(result.get("seed"))
        if seed in branch_reasons_by_seed:
            context = _mapping(result.get("carrion_archive_context"))
            reason = str(context.get("branch_reason", "unknown"))
            branch_reasons_by_seed[seed].update([reason])
    for row in rows:
        metadata = _mapping(row.get("metadata"))
        trainable = _mapping(row.get("trainable"))
        label = _mapping(trainable.get("label"))
        seed = _int(metadata.get("seed"))
        action = str(label.get("action", ""))
        if seed in safe_action_counts_by_seed and action:
            safe_action_counts_by_seed[seed].update([action])
    per_seed = []
    for seed in target_seeds:
        resolved = int(seed)
        reasons = branch_reasons_by_seed[resolved]
        safe_actions = safe_action_counts_by_seed[resolved]
        per_seed.append(
            {
                "fixture": "carrion_only",
                "seed": resolved,
                "branch_result_count": int(branch_counts.get(resolved, 0)),
                "safe_label_count": int(safe_counts.get(resolved, 0)),
                "branch_reason_counts": dict(sorted(reasons.items())),
                "missing_branch_reasons": [
                    reason
                    for reason in CARRION_BRANCH_REASONS
                    if reasons.get(reason, 0) <= 0
                ],
                "safe_label_action_counts": dict(sorted(safe_actions.items())),
            }
        )
    return {
        "policy": "m3_carrion_specific_archive_per_seed_support_v1",
        "target_seeds": [int(seed) for seed in target_seeds],
        "per_seed": per_seed,
        "missing_branch_result_seeds": [
            int(seed) for seed in target_seeds if branch_counts.get(int(seed), 0) <= 0
        ],
        "missing_safe_label_seeds": [
            int(seed) for seed in target_seeds if safe_counts.get(int(seed), 0) <= 0
        ],
    }


def _branch_reason_support(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    by_seed_reason: Counter[str] = Counter()
    for result in branch_results:
        context = _mapping(result.get("carrion_archive_context"))
        reason = str(context.get("branch_reason", "unknown"))
        seed = _int(result.get("seed"))
        counts.update([reason])
        by_seed_reason.update([f"{seed}:{reason}"])
    return {
        "policy": "m3_carrion_specific_archive_branch_reason_support_v1",
        "branch_reason_counts": dict(sorted(counts.items())),
        "branch_reason_counts_by_seed": dict(sorted(by_seed_reason.items())),
        "missing_branch_reasons": [
            reason for reason in CARRION_BRANCH_REASONS if counts.get(reason, 0) <= 0
        ],
    }


def _replay_verification_report(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    action_run_count = 0
    verified_count = 0
    missing_count = 0
    failed_count = 0
    for result in branch_results:
        for run in _list_of_mappings(result.get("action_runs")):
            action_run_count += 1
            replay = run.get("replay_verification")
            if not isinstance(replay, Mapping):
                missing_count += 1
                continue
            if replay.get("verified") is True:
                verified_count += 1
            else:
                failed_count += 1
    return {
        "policy": "m3_carrion_specific_archive_replay_verification_v1",
        "complete": (
            action_run_count > 0
            and verified_count == action_run_count
            and missing_count == 0
            and failed_count == 0
        ),
        "action_run_count": int(action_run_count),
        "verified_count": int(verified_count),
        "missing_count": int(missing_count),
        "failed_count": int(failed_count),
    }


def _action_coverage_report(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    missing = []
    for index, result in enumerate(branch_results):
        if _all_valid_actions_evaluated(result):
            continue
        detail = _missing_valid_action_detail(index, result)
        if detail is not None:
            missing.append(detail)
    return {
        "policy": "m3_carrion_specific_archive_action_coverage_v1",
        "complete": not missing,
        "branch_result_count": len(branch_results),
        "missing_valid_action_branch_count": len(missing),
        "first_missing_valid_action": missing[0] if missing else None,
    }


def _heuristic_action_source_count(
    branch_results: Sequence[Mapping[str, object]],
) -> int:
    return sum(
        _int(run.get("heuristic_action_source_count"))
        for result in branch_results
        for run in _list_of_mappings(result.get("action_runs"))
    )


def _precomputed_generation_status(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "policy": "precomputed_m3_carrion_specific_archive_generation_status_v1",
        "state": "complete",
        "partial": False,
        "branch_point_count": len(branch_results),
        "branch_result_count": len(branch_results),
        "generated_branch_result_count": 0,
        "resumed_branch_result_count": 0,
    }


def _report_ticks(
    *,
    branch_results: Sequence[Mapping[str, object]],
    status: Mapping[str, object],
    evidence: Mapping[str, object],
) -> int | None:
    for value in (status.get("ticks"), evidence.get("ticks")):
        resolved = _optional_int(value)
        if resolved is not None:
            return int(resolved)
    return _common_result_ticks(branch_results)


def _common_result_ticks(
    branch_results: Sequence[Mapping[str, object]],
) -> int | None:
    ticks = {
        int(resolved)
        for result in branch_results
        if (resolved := _optional_int(result.get("ticks"))) is not None
    }
    if len(ticks) == 1:
        return next(iter(ticks))
    return None


def _report_shard_id(
    *,
    status: Mapping[str, object],
    evidence: Mapping[str, object],
) -> str | None:
    for source in (status, evidence):
        value = _optional_string(source.get("shard_id"))
        if value:
            return value
    return None


def _input_path_payload(
    input_paths: Mapping[str, object] | None,
) -> dict[str, object]:
    if input_paths is None:
        return {
            "v145_report": None,
            "branch_results": None,
            "merge_shard_reports": [],
            "merge_shard_chunk_dirs": [],
        }
    return {
        "v145_report": _input_path_value(input_paths.get("v145_report")),
        "branch_results": _input_path_value(input_paths.get("branch_results")),
        "merge_shard_reports": _input_path_sequence(
            input_paths.get("merge_shard_reports")
        ),
        "merge_shard_chunk_dirs": _input_path_sequence(
            input_paths.get("merge_shard_chunk_dirs")
        ),
    }


def _input_path_value(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _input_path_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, Sequence):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    if not counts:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), ACTION_NAMES.index(item[0]) if item[0] in ACTION_NAMES else 999, item[0]),
    )[0]
    return {
        "key": key,
        "count": int(count),
        "share": _round(float(count) / float(sum(counts.values()))),
    }


def _status_partial(status: Mapping[str, object]) -> bool:
    return status.get("partial") is True or status.get("state") == "partial"


def _wall_budget_exhausted(started_at: float, max_wall_seconds: float | None) -> bool:
    if max_wall_seconds is None:
        return False
    return (time.monotonic() - float(started_at)) >= max(0.0, float(max_wall_seconds))


def _elapsed_seconds(started_at: float) -> float:
    return round(max(0.0, time.monotonic() - float(started_at)), 3)


def _emit_progress(
    progress_callback: Callable[[Mapping[str, object]], None] | None,
    payload: Mapping[str, object],
) -> None:
    if progress_callback is not None:
        progress_callback(dict(payload))


def _optional_ratio(value: object, *, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(float(value), 6)
    return default


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(float(value)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def _round(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(float(value), 6)
    return 0.0


def _flatten(value: object, *, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, object]] = []
        for key, nested in value.items():
            path = str(key) if not prefix else f"{prefix}.{key}"
            rows.extend(_flatten(nested, prefix=path))
        return rows
    if isinstance(value, list):
        rows = []
        for index, nested in enumerate(value):
            path = f"{prefix}[{index}]"
            rows.extend(_flatten(nested, prefix=path))
        return rows
    return [(prefix, value)]


def _looks_like_digest(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value.lower())
