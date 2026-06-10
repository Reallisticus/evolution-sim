from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.cli import mind_v3_evaluate as evaluate_cli
from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.broad_regression_branch_intervention import (
    BroadRegressionBranchPoint,
    _configure_manual_summary_run,
    _evaluate_branch_point,
    _optional_string,
)
from evolution_sim.mind.candidate_campaign import (
    M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION,
    M3_SAFE_ARCHIVE_EXPANSION_REPORT_SCHEMA_VERSION,
    SAFE_ARCHIVE_TRAIN_EVAL_BROAD_SEEDS,
    SAFE_ARCHIVE_TRAIN_EVAL_CARRION_SEEDS,
    SAFE_ARCHIVE_TRAIN_EVAL_TICKS,
    CandidateCampaignError,
    _safe_archive_expansion_branch_result_matches_point,
    _safe_archive_expansion_label_vet,
    _safe_archive_expansion_reference_from_world,
    load_safe_archive_expansion_branch_result_chunks,
    load_safe_archive_expansion_dataset,
    safe_archive_expansion_leakage_scan,
    write_safe_archive_expansion_branch_result_chunk,
)
from evolution_sim.mind.carrion_branch_explore import _branch_state_digest
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.safe_archive_failure_autopsy import (
    DEFAULT_BP3_BRANCH_EVIDENCE_PATH,
    DEFAULT_BP3_DATASET_PATH,
    load_json_report,
)
from evolution_sim.mind.safe_archive_sequence_context_audit import (
    DEFAULT_BP3_SEQUENCE_CONTEXT_AUDIT_OUTPUT_PATH,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

M3_PUBLIC_SEQUENCE_CONTEXT_BRANCH_EVIDENCE_SCHEMA_VERSION = (
    "m3_public_sequence_context_branch_evidence_report_v1"
)
M3_PUBLIC_SEQUENCE_CONTEXT_BRANCH_EVIDENCE_POLICY = (
    "diagnostics_only_m3_public_sequence_context_branch_evidence_v1"
)
PUBLIC_SEQUENCE_CONTEXT_FEATURE_POLICY = (
    "current_public_observation_mask_with_prior_public_transition_summaries_v1"
)
DEFAULT_BP3_SAFE_ARCHIVE_REPORT_PATH = Path("output/mind/shards/bp3-merged-report.json")
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/shards/"
    "bp3-public-sequence-context-branch-evidence-report.json"
)
DEFAULT_CHUNK_DIR = Path(
    "output/mind/shards/bp3-public-sequence-context-branch-evidence-chunks"
)
DEFAULT_HISTORY_WINDOW = 3
DEFAULT_MIN_PRIOR_PUBLIC_STEPS = 1
DEFAULT_MAX_BRANCH_POINTS_PER_SEED = 1
DEFAULT_MAX_CANDIDATE_ACTIONS = 0

FORBIDDEN_SEQUENCE_TRAINABLE_TOKENS = (
    "seed",
    "fixture",
    "branch",
    "tick",
    "agent",
    "path",
    "digest",
    "private",
    "future",
    "provenance",
)


class PublicSequenceContextBranchEvidenceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PublicSequenceContextBranchPoint:
    point: BroadRegressionBranchPoint
    prior_public_transition_summaries: tuple[dict[str, object], ...]


def write_public_sequence_context_branch_evidence_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def build_public_sequence_context_branch_evidence_report(
    *,
    bp3_safe_archive_report: Mapping[str, object],
    bp3_dataset_rows: Sequence[Mapping[str, object]],
    bp3_branch_evidence: Mapping[str, object],
    branch_results: Sequence[Mapping[str, object]] | None = None,
    generation_status: Mapping[str, object] | None = None,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    max_candidate_actions: int = DEFAULT_MAX_CANDIDATE_ACTIONS,
    history_window: int = DEFAULT_HISTORY_WINDOW,
    min_prior_public_steps: int = DEFAULT_MIN_PRIOR_PUBLIC_STEPS,
    broad_seeds: Sequence[int] = SAFE_ARCHIVE_TRAIN_EVAL_BROAD_SEEDS,
    carrion_seeds: Sequence[int] = SAFE_ARCHIVE_TRAIN_EVAL_CARRION_SEEDS,
    ticks: int = SAFE_ARCHIVE_TRAIN_EVAL_TICKS,
    fixture_selection: Sequence[str] = ("broad", "carrion_only"),
    seed_include: Sequence[int] | None = None,
    branch_index_include: Sequence[int] | None = None,
    branch_evidence_chunk_dir: str | Path | None = None,
    resume_branch_evidence: bool = False,
    max_wall_seconds: float | None = None,
    shard_id: str | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    started_at = time.monotonic()
    branch_limit = max(1, int(max_branch_points_per_seed))
    candidate_limit = max(0, int(max_candidate_actions))
    resolved_history_window = max(1, int(history_window))
    resolved_min_prior = max(1, int(min_prior_public_steps))
    selected_indexes = tuple(
        int(index) for index in (branch_index_include or range(branch_limit))
    )
    chunk_dir = Path(branch_evidence_chunk_dir) if branch_evidence_chunk_dir else None
    resumed = (
        load_safe_archive_expansion_branch_result_chunks(chunk_dir)
        if chunk_dir is not None and bool(resume_branch_evidence)
        else {}
    )
    source_validation = validate_public_sequence_context_sources(
        bp3_safe_archive_report=bp3_safe_archive_report,
        bp3_dataset_rows=bp3_dataset_rows,
        bp3_branch_evidence=bp3_branch_evidence,
    )
    if branch_results is None:
        generated = _generate_branch_results(
            broad_seeds=tuple(int(seed) for seed in broad_seeds),
            carrion_seeds=tuple(int(seed) for seed in carrion_seeds),
            ticks=int(ticks),
            fixture_selection=fixture_selection,
            seed_include=seed_include,
            branch_limit=branch_limit,
            selected_branch_indexes=selected_indexes,
            candidate_limit=candidate_limit,
            history_window=resolved_history_window,
            min_prior_public_steps=resolved_min_prior,
            chunk_dir=chunk_dir,
            resumed_branch_results=resumed,
            started_at=started_at,
            max_wall_seconds=max_wall_seconds,
            progress_callback=progress_callback,
        )
        resolved_branch_results = generated["branch_results"]
        resolved_generation_status = generated["generation_status"]
        generation_evidence = generated["generation_evidence"]
    else:
        resolved_branch_results = [dict(item) for item in branch_results]
        resolved_generation_status = dict(generation_status or _complete_status(
            started_at=started_at,
            max_wall_seconds=max_wall_seconds,
            branch_point_count=len(resolved_branch_results),
            branch_result_count=len(resolved_branch_results),
            generated_count=0,
            resumed_count=0,
            chunk_dir=chunk_dir,
            fixtures=fixture_selection,
            stop_reason=None,
        ))
        generation_evidence = {
            "policy": "precomputed_public_sequence_context_branch_results_v1",
            "fixture_reports": [],
            "resumed_chunk_count": len(resumed),
        }
    return _finalize_report(
        bp3_safe_archive_report=bp3_safe_archive_report,
        bp3_dataset_rows=bp3_dataset_rows,
        bp3_branch_evidence=bp3_branch_evidence,
        source_validation=source_validation,
        branch_results=resolved_branch_results,
        generation_status=resolved_generation_status,
        generation_evidence=generation_evidence,
        max_branch_points_per_seed=branch_limit,
        max_candidate_actions=candidate_limit,
        history_window=resolved_history_window,
        min_prior_public_steps=resolved_min_prior,
        fixture_selection=fixture_selection,
        seed_include=seed_include,
        branch_index_include=selected_indexes,
        branch_evidence_chunk_dir=chunk_dir,
        resume_branch_evidence=resume_branch_evidence,
        max_wall_seconds=max_wall_seconds,
        shard_id=shard_id,
        input_paths=input_paths,
    )


def build_public_sequence_context_branch_evidence_report_from_paths(
    *,
    bp3_safe_archive_report_path: str | Path = DEFAULT_BP3_SAFE_ARCHIVE_REPORT_PATH,
    bp3_dataset_path: str | Path = DEFAULT_BP3_DATASET_PATH,
    bp3_branch_evidence_path: str | Path = DEFAULT_BP3_BRANCH_EVIDENCE_PATH,
    **kwargs: object,
) -> dict[str, object]:
    safe_report = load_json_report(bp3_safe_archive_report_path)
    dataset_rows = load_safe_archive_expansion_dataset(bp3_dataset_path)
    branch_evidence = load_json_report(bp3_branch_evidence_path)
    return build_public_sequence_context_branch_evidence_report(
        bp3_safe_archive_report=safe_report,
        bp3_dataset_rows=dataset_rows,
        bp3_branch_evidence=branch_evidence,
        input_paths={
            "bp3_safe_archive_report": bp3_safe_archive_report_path,
            "bp3_dataset": bp3_dataset_path,
            "bp3_branch_evidence": bp3_branch_evidence_path,
        },
        **kwargs,
    )


def merge_public_sequence_context_branch_evidence_reports(
    *,
    bp3_safe_archive_report: Mapping[str, object],
    bp3_dataset_rows: Sequence[Mapping[str, object]],
    bp3_branch_evidence: Mapping[str, object],
    shard_reports: Sequence[Mapping[str, object]],
    shard_chunk_dirs: Sequence[str | Path] = (),
    allow_partial_shard_evidence: bool = False,
    input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    merged: dict[str, dict[str, object]] = {}
    digests_by_branch_id: dict[str, str] = {}
    source_summaries: list[dict[str, object]] = []
    partial_sources: list[dict[str, object]] = []
    merge_identity: dict[str, object] | None = None
    for index, report in enumerate(shard_reports):
        if report.get("schema_version") != (
            M3_PUBLIC_SEQUENCE_CONTEXT_BRANCH_EVIDENCE_SCHEMA_VERSION
        ):
            raise PublicSequenceContextBranchEvidenceError(
                f"shard {index} schema mismatch"
            )
        if report.get("policy") != M3_PUBLIC_SEQUENCE_CONTEXT_BRANCH_EVIDENCE_POLICY:
            raise PublicSequenceContextBranchEvidenceError(
                f"shard {index} policy mismatch"
            )
        if report.get("diagnostics_only") is not True:
            raise PublicSequenceContextBranchEvidenceError(
                f"shard {index} missing diagnostics-only contract"
            )
        shard_auth_failures: list[str] = []
        _validate_authorization_flags(
            name=f"shard_{index}",
            payload=report,
            failures=shard_auth_failures,
            missing_is_drift=True,
        )
        if report.get("default_runtime_behavior_changed") is not False:
            shard_auth_failures.append(
                f"shard_{index}_default_runtime_behavior_changed_not_false"
            )
        if shard_auth_failures:
            raise PublicSequenceContextBranchEvidenceError(
                f"shard {index} authorization drift: {sorted(shard_auth_failures)}"
            )
        identity = _merge_identity(report)
        if merge_identity is None:
            merge_identity = identity
        elif identity != merge_identity:
            raise PublicSequenceContextBranchEvidenceError(
                f"shard {index} generation inputs mismatch"
            )
        source_integrity = _mapping(report.get("source_integrity"))
        status = _mapping(report.get("generation_status"))
        failures = [str(item) for item in _list(source_integrity.get("failures"))]
        non_partial_failures = sorted(
            failure for failure in failures if failure != "partial_branch_evidence"
        )
        partial = _status_partial(status) or "partial_branch_evidence" in failures
        if source_integrity.get("passed") is not True and non_partial_failures:
            raise PublicSequenceContextBranchEvidenceError(
                f"shard {index} source integrity failed: {non_partial_failures}"
            )
        if partial:
            summary = {
                "source": f"report:{index}",
                "state": status.get("state"),
                "stop_reason": status.get("stop_reason"),
                "source_integrity_failures": failures,
            }
            if not allow_partial_shard_evidence:
                raise PublicSequenceContextBranchEvidenceError(
                    f"partial shard evidence requires explicit partial merge: {summary}"
                )
            partial_sources.append(summary)
        branch_results = _list_of_mappings(report.get("branch_results"))
        expected_digest = report.get("branch_evidence_digest")
        observed_digest = stable_payload_digest(branch_results)
        if expected_digest != observed_digest:
            raise PublicSequenceContextBranchEvidenceError(
                f"shard {index} branch evidence digest mismatch"
            )
        for result in branch_results:
            _add_merged_branch_result(
                merged=merged,
                digests_by_branch_id=digests_by_branch_id,
                result=result,
                source=f"report:{index}",
            )
        source_summaries.append(
            {
                "source": f"report:{index}",
                "branch_result_count": len(branch_results),
                "branch_evidence_digest": expected_digest,
                "partial": partial,
                "shard_id": _mapping(report.get("inputs")).get("shard_id"),
            }
        )
    for chunk_dir in shard_chunk_dirs:
        chunks = load_safe_archive_expansion_branch_result_chunks(chunk_dir)
        for result in chunks.values():
            branch_id = str(result.get("branch_id", ""))
            if branch_id not in merged:
                raise PublicSequenceContextBranchEvidenceError(
                    "chunk branch result lacks matching shard report: "
                    f"{chunk_dir}:{branch_id}"
                )
            _add_merged_branch_result(
                merged=merged,
                digests_by_branch_id=digests_by_branch_id,
                result=result,
                source=f"chunk:{chunk_dir}",
            )
        source_summaries.append(
            {
                "source": f"chunk:{chunk_dir}",
                "branch_result_count": len(chunks),
                "partial": False,
            }
        )
    if not merged:
        raise PublicSequenceContextBranchEvidenceError(
            "public sequence context shard merge has no branch results"
        )
    branch_results = sorted(
        merged.values(),
        key=lambda item: (
            _fixture_order(str(item.get("fixture", ""))),
            _int(item.get("seed")),
            _int(item.get("branch_index")),
            str(item.get("branch_id", "")),
        ),
    )
    generation_status = {
        "policy": "m3_public_sequence_context_merged_generation_status_v1",
        "state": "partial" if partial_sources else "complete",
        "partial": bool(partial_sources),
        "stop_reason": "partial_shard_evidence" if partial_sources else None,
        "source_count": len(source_summaries),
        "partial_sources": partial_sources,
        "branch_result_count": len(branch_results),
    }
    report = build_public_sequence_context_branch_evidence_report(
        bp3_safe_archive_report=bp3_safe_archive_report,
        bp3_dataset_rows=bp3_dataset_rows,
        bp3_branch_evidence=bp3_branch_evidence,
        branch_results=branch_results,
        generation_status=generation_status,
        input_paths=input_paths,
    )
    report["shard_merge"] = {
        "policy": "m3_public_sequence_context_shard_merge_v1",
        "sources": source_summaries,
        "partial_sources": partial_sources,
        "duplicate_branch_id_policy": "same_digest_allowed_conflict_rejected",
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
    }
    return report


def validate_public_sequence_context_sources(
    *,
    bp3_safe_archive_report: Mapping[str, object],
    bp3_dataset_rows: Sequence[Mapping[str, object]],
    bp3_branch_evidence: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    report_dataset = _mapping(bp3_safe_archive_report.get("dataset"))
    report_source_integrity = _mapping(bp3_safe_archive_report.get("source_integrity"))
    report_classification = _mapping(bp3_safe_archive_report.get("classification"))
    branch_status = _mapping(bp3_branch_evidence.get("generation_status"))
    branch_source_integrity = _mapping(bp3_branch_evidence.get("source_integrity"))
    dataset_digest = stable_payload_digest(list(bp3_dataset_rows))
    branch_results = _list_of_mappings(bp3_branch_evidence.get("branch_results"))
    branch_digest = stable_payload_digest(branch_results)
    leakage_scan = safe_archive_expansion_leakage_scan(bp3_dataset_rows)
    if bp3_safe_archive_report.get("schema_version") != (
        M3_SAFE_ARCHIVE_EXPANSION_REPORT_SCHEMA_VERSION
    ):
        failures.append("bp3_safe_archive_report_schema_mismatch")
    if report_classification.get("primary") != (
        "m3_safe_archive_expansion_support_ready_no_training_run"
    ):
        failures.append("bp3_safe_archive_report_not_support_ready")
    if report_source_integrity.get("passed") is not True:
        failures.append("bp3_safe_archive_report_source_integrity_not_passed")
    if list(report_source_integrity.get("failures") or []) != []:
        failures.append("bp3_safe_archive_report_source_integrity_failures_present")
    if report_dataset.get("dataset_digest") != dataset_digest:
        failures.append("bp3_dataset_digest_mismatch")
    if not bp3_dataset_rows:
        failures.append("bp3_dataset_rows_missing")
    if leakage_scan.get("passed") is not True:
        failures.append("bp3_dataset_trainable_leakage_scan_failed")
    if bp3_branch_evidence.get("schema_version") != (
        M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION
    ):
        failures.append("bp3_branch_evidence_schema_mismatch")
    if branch_status.get("state") != "complete" or branch_status.get("partial") is True:
        failures.append("bp3_branch_evidence_not_complete")
    if branch_source_integrity.get("passed") is not True:
        failures.append("bp3_branch_evidence_source_integrity_not_passed")
    if list(branch_source_integrity.get("failures") or []) != []:
        failures.append("bp3_branch_evidence_source_integrity_failures_present")
    if bp3_branch_evidence.get("branch_evidence_digest") != branch_digest:
        failures.append("bp3_branch_evidence_digest_mismatch")
    for name, payload in (
        ("bp3_safe_archive_report", bp3_safe_archive_report),
        ("bp3_branch_evidence", bp3_branch_evidence),
    ):
        _validate_authorization_flags(name=name, payload=payload, failures=failures)
        _validate_authorization_flags(
            name=f"{name}_contract",
            payload=_mapping(payload.get("contract")),
            failures=failures,
            missing_is_drift=False,
        )
    return {
        "policy": "m3_public_sequence_context_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "bp3_safe_archive_classification": report_classification.get("primary"),
        "bp3_dataset_digest": dataset_digest,
        "bp3_branch_evidence_digest": branch_digest,
        "bp3_dataset_leakage_scan": leakage_scan,
        "bp3_branch_evidence_generation_state": branch_status.get("state"),
    }


def public_sequence_context_trainable_leakage_scan(
    trainable_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    finite_failures: list[dict[str, object]] = []
    for row_index, row in enumerate(trainable_rows):
        for path, value in _flatten(row):
            lower_path = path.lower()
            if any(token in lower_path for token in FORBIDDEN_SEQUENCE_TRAINABLE_TOKENS):
                failures.append(
                    {
                        "row_index": row_index,
                        "path": path,
                        "reason": "forbidden_trainable_path_token",
                    }
                )
            if isinstance(value, str) and _looks_like_digest(value):
                failures.append(
                    {
                        "row_index": row_index,
                        "path": path,
                        "reason": "forbidden_trainable_digest_value",
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
    all_failures = failures + finite_failures
    return {
        "policy": "m3_public_sequence_context_trainable_leakage_scan_v1",
        "passed": not all_failures,
        "row_count": len(trainable_rows),
        "forbidden_tokens": list(FORBIDDEN_SEQUENCE_TRAINABLE_TOKENS),
        "forbidden_failure_count": len(failures),
        "finite_failure_count": len(finite_failures),
        "failures": all_failures[:32],
    }


def public_sequence_context_for_branch_result_matches(
    result: Mapping[str, object],
    trainable_context: Mapping[str, object],
) -> bool:
    context = _mapping(_mapping(result.get("public_sequence_context")).get("trainable"))
    expected_digest = stable_payload_digest(trainable_context)
    return (
        context == dict(trainable_context)
        and _mapping(result.get("public_sequence_context")).get(
            "trainable_context_digest"
        )
        == expected_digest
    )


def _generate_branch_results(
    *,
    broad_seeds: Sequence[int],
    carrion_seeds: Sequence[int],
    ticks: int,
    fixture_selection: Sequence[str],
    seed_include: Sequence[int] | None,
    branch_limit: int,
    selected_branch_indexes: Sequence[int],
    candidate_limit: int,
    history_window: int,
    min_prior_public_steps: int,
    chunk_dir: Path | None,
    resumed_branch_results: Mapping[str, Mapping[str, object]],
    started_at: float,
    max_wall_seconds: float | None,
    progress_callback: Callable[[Mapping[str, object]], None] | None,
) -> dict[str, object]:
    selected_fixtures = _selected_fixtures(fixture_selection)
    seed_filter = None if seed_include is None else {int(seed) for seed in seed_include}
    fixture_reports = []
    branch_results: list[dict[str, object]] = []
    generated_count = 0
    resumed_count = 0
    stop_reason: str | None = None
    for fixture in selected_fixtures:
        seeds = broad_seeds if fixture == "broad" else carrion_seeds
        if seed_filter is not None:
            seeds = tuple(int(seed) for seed in seeds if int(seed) in seed_filter)
        fixture_result = _generate_fixture_branch_results(
            fixture=fixture,
            seeds=seeds,
            ticks=ticks,
            branch_limit=branch_limit,
            selected_branch_indexes=selected_branch_indexes,
            candidate_limit=candidate_limit,
            history_window=history_window,
            min_prior_public_steps=min_prior_public_steps,
            chunk_dir=chunk_dir,
            resumed_branch_results=resumed_branch_results,
            started_at=started_at,
            max_wall_seconds=max_wall_seconds,
            progress_callback=progress_callback,
        )
        fixture_reports.append(fixture_result["fixture_report"])
        branch_results.extend(fixture_result["branch_results"])
        generated_count += _int(fixture_result.get("generated_count"))
        resumed_count += _int(fixture_result.get("resumed_count"))
        if fixture_result.get("stop_reason") is not None:
            stop_reason = str(fixture_result.get("stop_reason"))
            break
    status = _complete_status(
        started_at=started_at,
        max_wall_seconds=max_wall_seconds,
        branch_point_count=sum(
            _int(_mapping(report.get("generation_status")).get("branch_point_count"))
            for report in fixture_reports
        ),
        branch_result_count=len(branch_results),
        generated_count=generated_count,
        resumed_count=resumed_count,
        chunk_dir=chunk_dir,
        fixtures=selected_fixtures,
        stop_reason=stop_reason,
    )
    return {
        "policy": "m3_public_sequence_context_generated_branch_results_v1",
        "branch_results": branch_results,
        "generation_status": status,
        "generation_evidence": {
            "policy": "m3_public_sequence_context_generation_evidence_v1",
            "fixture_reports": fixture_reports,
            "resumed_chunk_count": len(resumed_branch_results),
        },
    }


def _generate_fixture_branch_results(
    *,
    fixture: str,
    seeds: Sequence[int],
    ticks: int,
    branch_limit: int,
    selected_branch_indexes: Sequence[int],
    candidate_limit: int,
    history_window: int,
    min_prior_public_steps: int,
    chunk_dir: Path | None,
    resumed_branch_results: Mapping[str, Mapping[str, object]],
    started_at: float,
    max_wall_seconds: float | None,
    progress_callback: Callable[[Mapping[str, object]], None] | None,
) -> dict[str, object]:
    branch_results: list[dict[str, object]] = []
    seed_reports = []
    generated_count = 0
    resumed_count = 0
    branch_point_count = 0
    stop_reason: str | None = None
    for seed in seeds:
        if _wall_budget_exhausted(started_at, max_wall_seconds):
            stop_reason = "max_wall_seconds_elapsed_before_seed"
            break
        points, reference, seed_report = _materialize_sequence_context_points(
            fixture=fixture,
            seed=int(seed),
            ticks=int(ticks),
            branch_limit=int(branch_limit),
            history_window=int(history_window),
            min_prior_public_steps=int(min_prior_public_steps),
        )
        selected = [
            item
            for item in points
            if int(item.point.branch_index) in set(int(i) for i in selected_branch_indexes)
        ]
        branch_point_count += len(selected)
        seed_reports.append(seed_report)
        reference_runs = {int(seed): {"baseline": reference, "v142_override": reference}}
        for sequence_point in selected:
            if _wall_budget_exhausted(started_at, max_wall_seconds):
                stop_reason = "max_wall_seconds_elapsed_before_branch_point"
                break
            result, source = _evaluate_sequence_branch_point_checkpointed(
                sequence_point,
                reference_runs=reference_runs,
                candidate_limit=candidate_limit,
                chunk_dir=chunk_dir,
                resumed_branch_results=resumed_branch_results,
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
                    "fixture": fixture,
                    "seed": int(seed),
                    "branch_id": sequence_point.point.branch_id,
                    "branch_point_index": int(sequence_point.point.branch_index),
                    "action_count": len(_list_of_mappings(result.get("action_runs"))),
                    "elapsed_seconds": _elapsed_seconds(started_at),
                },
            )
        if stop_reason is not None:
            break
    fixture_status = _fixture_status(
        fixture=fixture,
        started_at=started_at,
        max_wall_seconds=max_wall_seconds,
        branch_point_count=branch_point_count,
        branch_result_count=len(branch_results),
        generated_count=generated_count,
        resumed_count=resumed_count,
        chunk_dir=chunk_dir,
        stop_reason=stop_reason,
    )
    return {
        "branch_results": branch_results,
        "generated_count": generated_count,
        "resumed_count": resumed_count,
        "stop_reason": stop_reason,
        "fixture_report": {
            "policy": "m3_public_sequence_context_fixture_generation_v1",
            "fixture": fixture,
            "seeds": [int(seed) for seed in seeds],
            "seed_reports": seed_reports,
            "generation_status": fixture_status,
            "branch_result_count": len(branch_results),
        },
    }


def _materialize_sequence_context_points(
    *,
    fixture: str,
    seed: int,
    ticks: int,
    branch_limit: int,
    history_window: int,
    min_prior_public_steps: int,
) -> tuple[list[PublicSequenceContextBranchPoint], dict[str, object], dict[str, object]]:
    if fixture == "carrion_only":
        world = evaluate_cli._fixture_world(
            fixture_name="carrion_only",
            seed=int(seed),
            ticks=int(ticks),
            policy=MindV3EvolutionPolicy(seed=int(seed)),
        )
    else:
        world = SimulationWorld(
            WorldConfig(seed=int(seed), max_ticks=int(ticks)),
            policy=MindV3EvolutionPolicy(seed=int(seed)),
        )
    _configure_manual_summary_run(world)
    points: list[PublicSequenceContextBranchPoint] = []
    history_by_agent: dict[int, list[dict[str, object]]] = {}
    failures: list[dict[str, object]] = []
    record_index = 0
    ticks_executed = 0
    for tick in range(int(ticks)):
        world.tick = tick
        snapshot = deepcopy(world)
        world._run_tick()
        ticks_executed = tick + 1
        for record in list(world.tick_trajectory_records):
            if len(points) < int(branch_limit):
                maybe_point = _point_from_record(
                    fixture=fixture,
                    seed=int(seed),
                    ticks=int(ticks),
                    tick=int(tick),
                    record_index=int(record_index),
                    branch_index=len(points),
                    record=record,
                    snapshot=snapshot,
                    prior_history=history_by_agent.get(
                        _int(record.get("agent_id"), default=-1),
                        [],
                    ),
                    history_window=int(history_window),
                    min_prior_public_steps=int(min_prior_public_steps),
                )
                if isinstance(maybe_point, PublicSequenceContextBranchPoint):
                    points.append(maybe_point)
                elif isinstance(maybe_point, Mapping):
                    failures.append(dict(maybe_point))
            _update_public_history_from_record(
                history_by_agent=history_by_agent,
                record=record,
                history_window=int(history_window),
            )
            record_index += 1
        if len(points) >= int(branch_limit):
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
        runtime=f"public_sequence_context_{fixture}_linear_mind_v3",
    )
    return (
        points,
        reference,
        {
            "fixture": fixture,
            "seed": int(seed),
            "ticks_requested": int(ticks),
            "ticks_executed": int(ticks_executed),
            "branch_point_count": len(points),
            "branch_ids": [item.point.branch_id for item in points],
            "history_window": int(history_window),
            "min_prior_public_steps": int(min_prior_public_steps),
            "failure_count": len(failures),
            "failures": failures[:12],
            "passed": bool(points) and not failures,
        },
    )


def _point_from_record(
    *,
    fixture: str,
    seed: int,
    ticks: int,
    tick: int,
    record_index: int,
    branch_index: int,
    record: Mapping[str, object],
    snapshot: SimulationWorld,
    prior_history: Sequence[Mapping[str, object]],
    history_window: int,
    min_prior_public_steps: int,
) -> PublicSequenceContextBranchPoint | dict[str, object] | None:
    agent_id = _int(record.get("agent_id"), default=-1)
    if agent_id < 0:
        return None
    prior = tuple(dict(item) for item in prior_history[-int(history_window):])
    if len(prior) < int(min_prior_public_steps):
        return None
    action_mask = _bool_action_mask(record.get("action_mask"))
    if not any(action_mask.values()):
        return None
    requested = str(record.get("requested_action", ""))
    if requested not in ACTION_NAMES:
        return {
            "fixture": fixture,
            "seed": int(seed),
            "tick": int(tick),
            "record_index": int(record_index),
            "agent_id": int(agent_id),
            "reason": "missing_current_policy_requested_action",
        }
    observation_input = _mapping(record.get("observation_input"))
    if not observation_input:
        return {
            "fixture": fixture,
            "seed": int(seed),
            "tick": int(tick),
            "record_index": int(record_index),
            "agent_id": int(agent_id),
            "reason": "missing_public_observation_input",
        }
    return _point_from_record_with_index(
        fixture=fixture,
        seed=seed,
        ticks=ticks,
        tick=tick,
        record_index=record_index,
        branch_index=branch_index,
        agent_id=agent_id,
        requested=requested,
        record=record,
        snapshot=snapshot,
        action_mask=action_mask,
        observation_input=observation_input,
        prior=prior,
    )


def _point_from_record_with_index(
    *,
    fixture: str,
    seed: int,
    ticks: int,
    tick: int,
    record_index: int,
    branch_index: int,
    agent_id: int,
    requested: str,
    record: Mapping[str, object],
    snapshot: SimulationWorld,
    action_mask: Mapping[str, bool],
    observation_input: Mapping[str, object],
    prior: Sequence[Mapping[str, object]],
) -> PublicSequenceContextBranchPoint:
    resolved_index = int(branch_index)
    if resolved_index < 0:
        resolved_index = 0
    slug = "carrion" if fixture == "carrion_only" else "broad"
    branch_id = (
        f"m3-public-sequence-context-{slug}-seed-{seed}-branch-"
        f"{resolved_index}-tick-{tick}-agent-{agent_id}"
    )
    point = BroadRegressionBranchPoint(
        branch_id=branch_id,
        seed=int(seed),
        fixture=fixture,
        ticks=int(ticks),
        branch_tick=int(tick),
        record_index=int(record_index),
        branch_index=int(resolved_index),
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
    return PublicSequenceContextBranchPoint(
        point=point,
        prior_public_transition_summaries=tuple(dict(item) for item in prior),
    )


def _evaluate_sequence_branch_point_checkpointed(
    sequence_point: PublicSequenceContextBranchPoint,
    *,
    reference_runs: Mapping[int, Mapping[str, Mapping[str, object]]],
    candidate_limit: int,
    chunk_dir: Path | None,
    resumed_branch_results: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], str]:
    point = sequence_point.point
    trainable_context = _trainable_context_for_sequence_point(sequence_point)
    cached = resumed_branch_results.get(point.branch_id)
    if (
        cached is not None
        and _safe_archive_expansion_branch_result_matches_point(
            cached,
            point,
            max_candidate_actions=int(candidate_limit),
            verify_replay=True,
        )
        and public_sequence_context_for_branch_result_matches(cached, trainable_context)
    ):
        return dict(cached), "resumed"
    result = _evaluate_branch_point(
        point,
        reference_runs=reference_runs,
        max_candidate_actions=int(candidate_limit),
        verify_replay=True,
    )
    enriched = _enrich_branch_result_with_sequence_context(
        result,
        sequence_point=sequence_point,
    )
    if chunk_dir is not None:
        write_safe_archive_expansion_branch_result_chunk(enriched, chunk_dir)
    return enriched, "generated"


def _enrich_branch_result_with_sequence_context(
    result: Mapping[str, object],
    *,
    sequence_point: PublicSequenceContextBranchPoint,
) -> dict[str, object]:
    trainable_context = _trainable_context_for_sequence_point(sequence_point)
    enriched = dict(result)
    enriched["public_sequence_context"] = {
        "feature_policy": PUBLIC_SEQUENCE_CONTEXT_FEATURE_POLICY,
        "policy_visible_public_inputs_only": True,
        "current_inputs": ["observation_input", "action_mask"],
        "prior_inputs": ["prior_finalized_public_transition_summaries"],
        "trainable": trainable_context,
        "history_step_count": len(sequence_point.prior_public_transition_summaries),
        "trainable_context_digest": stable_payload_digest(trainable_context),
    }
    return enriched


def _trainable_context_for_sequence_point(
    sequence_point: PublicSequenceContextBranchPoint,
) -> dict[str, object]:
    point = sequence_point.point
    return {
        "feature_policy": PUBLIC_SEQUENCE_CONTEXT_FEATURE_POLICY,
        "features": {
            "observation_input": dict(point.observation_input),
            "action_mask": dict(sorted(point.action_mask.items())),
            "prior_public_transition_summaries": [
                dict(item) for item in sequence_point.prior_public_transition_summaries
            ],
        },
    }


def _update_public_history_from_record(
    *,
    history_by_agent: dict[int, list[dict[str, object]]],
    record: Mapping[str, object],
    history_window: int,
) -> None:
    if record.get("action_source") == "passive":
        return
    agent_id = _int(record.get("agent_id"), default=-1)
    if agent_id < 0:
        return
    requested = record.get("requested_action")
    resolved = record.get("resolved_action")
    if not isinstance(requested, str) or not isinstance(resolved, str):
        return
    item = _public_transition_summary(record)
    history = history_by_agent.setdefault(agent_id, [])
    history.append(item)
    if len(history) > int(history_window):
        del history[0 : len(history) - int(history_window)]


def _public_transition_summary(record: Mapping[str, object]) -> dict[str, object]:
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    outcome = _mapping(record.get("outcome"))
    feeding = _mapping(outcome.get("feeding"))
    drinking = _mapping(outcome.get("drinking"))
    passive = _mapping(outcome.get("passive"))
    return {
        "requested_action": _optional_string(record.get("requested_action")),
        "resolved_action": _optional_string(record.get("resolved_action")),
        "action_valid": bool(record.get("action_valid", False)),
        "resolution_action_valid": bool(
            record.get("resolution_action_valid", False)
        ),
        "moved": bool(record.get("moved", False)),
        "x_delta": _int(after.get("x")) - _int(before.get("x")),
        "y_delta": _int(after.get("y")) - _int(before.get("y")),
        "energy_ratio_before": _optional_float(before.get("energy_ratio")),
        "energy_ratio_after": _optional_float(after.get("energy_ratio")),
        "energy_ratio_delta": _optional_delta(after, before, "energy_ratio"),
        "hydration_ratio_before": _optional_float(before.get("hydration_ratio")),
        "hydration_ratio_after": _optional_float(after.get("hydration_ratio")),
        "hydration_ratio_delta": _optional_delta(after, before, "hydration_ratio"),
        "health_ratio_before": _optional_float(before.get("health_ratio")),
        "health_ratio_after": _optional_float(after.get("health_ratio")),
        "health_ratio_delta": _optional_delta(after, before, "health_ratio"),
        "resource_gain": _optional_float(outcome.get("resource_gain")),
        "drank": bool(drinking.get("drank", False)),
        "ate": bool(feeding.get("ate", False)),
        "animal_resource_gain": str(feeding.get("food_source")) in {
            "carcass",
            "fresh_kill",
        },
        "died": bool(outcome.get("died", False)),
        "death_cause": _optional_string(passive.get("death_cause")),
        "died_after_action": bool(passive.get("died_after_action", False)),
    }


def _finalize_report(
    *,
    bp3_safe_archive_report: Mapping[str, object],
    bp3_dataset_rows: Sequence[Mapping[str, object]],
    bp3_branch_evidence: Mapping[str, object],
    source_validation: Mapping[str, object],
    branch_results: Sequence[Mapping[str, object]],
    generation_status: Mapping[str, object],
    generation_evidence: Mapping[str, object],
    max_branch_points_per_seed: int,
    max_candidate_actions: int,
    history_window: int,
    min_prior_public_steps: int,
    fixture_selection: Sequence[str],
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int],
    branch_evidence_chunk_dir: Path | None,
    resume_branch_evidence: bool,
    max_wall_seconds: float | None,
    shard_id: str | None,
    input_paths: Mapping[str, str | Path] | None,
) -> dict[str, object]:
    trainable_context_rows = _trainable_context_rows(branch_results)
    leakage_scan = public_sequence_context_trainable_leakage_scan(
        trainable_context_rows
    )
    aliasing = _bp3_aliasing_report(
        bp3_dataset_rows=bp3_dataset_rows,
        branch_results=branch_results,
    )
    replay = _replay_verification_report(branch_results)
    coverage = _coverage_report(branch_results)
    source_integrity = _source_integrity(
        source_validation=source_validation,
        leakage_scan=leakage_scan,
        aliasing=aliasing,
        replay=replay,
        coverage=coverage,
        generation_status=generation_status,
    )
    classification = _classification(
        source_integrity=source_integrity,
        generation_status=generation_status,
        aliasing=aliasing,
    )
    input_payload = _input_path_payload(input_paths)
    contract = {
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_policy_changed": False,
        "runtime_action_selection_changed": False,
        "gate_relaxation": False,
        "trainer_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "trainable_inputs": [
            "current observation_input",
            "current action_mask",
            "prior finalized public transition summaries",
        ],
        "excluded_trainable_inputs": [
            "seed",
            "fixture",
            "branch",
            "tick",
            "agent",
            "path",
            "digest",
            "private state",
            "future outcome",
        ],
    }
    return {
        "schema_version": M3_PUBLIC_SEQUENCE_CONTEXT_BRANCH_EVIDENCE_SCHEMA_VERSION,
        "policy": M3_PUBLIC_SEQUENCE_CONTEXT_BRANCH_EVIDENCE_POLICY,
        "diagnostics_only": True,
        "contract": contract,
        "provenance": {"contract_digest": stable_payload_digest(contract)},
        "inputs": {
            **input_payload,
            "shard_id": shard_id,
            "fixture_selection": list(_selected_fixtures(fixture_selection)),
            "seed_include": None
            if seed_include is None
            else [int(seed) for seed in seed_include],
            "branch_index_selection": [int(index) for index in branch_index_include],
            "max_branch_points_per_seed": int(max_branch_points_per_seed),
            "max_candidate_actions": (
                None if int(max_candidate_actions) == 0 else int(max_candidate_actions)
            ),
            "history_window": int(history_window),
            "min_prior_public_steps": int(min_prior_public_steps),
            "branch_evidence_chunk_dir": (
                None if branch_evidence_chunk_dir is None else str(branch_evidence_chunk_dir)
            ),
            "resume_branch_evidence": bool(resume_branch_evidence),
            "max_wall_seconds": (
                None if max_wall_seconds is None else float(max_wall_seconds)
            ),
            "bp3_safe_archive_report_digest": stable_payload_digest(
                bp3_safe_archive_report
            ),
            "bp3_dataset_digest": source_validation.get("bp3_dataset_digest"),
            "bp3_branch_evidence_digest": source_validation.get(
                "bp3_branch_evidence_digest"
            ),
        },
        "generation_status": dict(generation_status),
        "generation_evidence": dict(generation_evidence),
        "source_validation": dict(source_validation),
        "source_integrity": source_integrity,
        "coverage": coverage,
        "replay_verification": replay,
        "leakage_scan": leakage_scan,
        "bp3_action_only_aliasing": aliasing,
        "classification": {"primary": classification, "labels": [classification]},
        "recommendation": _recommendation(classification, aliasing),
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
        "branch_results": [dict(item) for item in branch_results],
        "branch_result_count": len(branch_results),
        "branch_evidence_digest": stable_payload_digest(
            [dict(item) for item in branch_results]
        ),
        "non_promoted": True,
    }


def _source_integrity(
    *,
    source_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    aliasing: Mapping[str, object],
    replay: Mapping[str, object],
    coverage: Mapping[str, object],
    generation_status: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if source_validation.get("passed") is not True:
        failures.append("bp3_source_validation_failed")
    if leakage_scan.get("passed") is not True:
        failures.append("public_sequence_context_trainable_leakage_failed")
    if replay.get("complete") is not True:
        failures.append("missing_or_failed_branch_replay_verification")
    if coverage.get("branch_result_count", 0) == 0:
        failures.append("no_public_sequence_context_branch_results")
    if coverage.get("all_branch_results_have_prior_context") is not True:
        failures.append("missing_prior_public_context")
    if coverage.get("all_valid_candidate_actions_evaluated") is not True:
        failures.append("not_all_valid_candidate_actions_evaluated")
    if _status_partial(generation_status):
        failures.append("partial_branch_evidence")
    return {
        "policy": "m3_public_sequence_context_source_integrity_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "source_validation_passed": source_validation.get("passed") is True,
        "leakage_scan_passed": leakage_scan.get("passed") is True,
        "replay_verification_complete": replay.get("complete") is True,
        "branch_result_count": coverage.get("branch_result_count"),
        "action_only_alias_count": aliasing.get("action_only_alias_count"),
        "sequence_separated_alias_count": aliasing.get(
            "sequence_separated_alias_count"
        ),
    }


def _bp3_aliasing_report(
    *,
    bp3_dataset_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    bp3_one_step: dict[str, list[dict[str, object]]] = {}
    bp3_sequence_keys: dict[str, set[str]] = {}
    for row_index, row in enumerate(bp3_dataset_rows):
        trainable = _mapping(row.get("trainable"))
        features = _mapping(trainable.get("features"))
        action = str(_mapping(trainable.get("label")).get("action", ""))
        if action not in ACTION_NAMES:
            continue
        one_key = _one_step_action_key(features, action)
        sequence_key = _sequence_action_key(features, action)
        bp3_one_step.setdefault(one_key, []).append(
            {
                "row_index": row_index,
                "action": action,
                "sequence_key": sequence_key,
            }
        )
        bp3_sequence_keys.setdefault(one_key, set()).add(sequence_key)
    alias_count = 0
    separated_count = 0
    safe_alias_count = 0
    safe_separated_count = 0
    context_available_alias_count = 0
    action_counts: Counter[str] = Counter()
    separated_action_counts: Counter[str] = Counter()
    samples: list[dict[str, object]] = []
    for result_index, result in enumerate(branch_results):
        context = _mapping(_mapping(result.get("public_sequence_context")).get("trainable"))
        features = _mapping(context.get("features"))
        prior = _list_of_mappings(features.get("prior_public_transition_summaries"))
        for run in _list_of_mappings(result.get("action_runs")):
            action = str(run.get("forced_action", ""))
            if action not in ACTION_NAMES:
                continue
            one_key = _one_step_action_key(features, action)
            matches = bp3_one_step.get(one_key, [])
            if not matches:
                continue
            alias_count += 1
            action_counts.update([action])
            sequence_key = _sequence_action_key(features, action)
            separated = sequence_key not in bp3_sequence_keys.get(one_key, set())
            if prior:
                context_available_alias_count += 1
            if separated:
                separated_count += 1
                separated_action_counts.update([action])
            safe = _safe_archive_expansion_label_vet(run).get("passed") is True
            if safe:
                safe_alias_count += 1
                if separated:
                    safe_separated_count += 1
            if len(samples) < 32:
                samples.append(
                    {
                        "source_branch_result_index": result_index,
                        "branch_id": result.get("branch_id"),
                        "fixture": result.get("fixture"),
                        "seed": result.get("seed"),
                        "branch_index": result.get("branch_index"),
                        "action": action,
                        "safe_action_run": safe,
                        "matched_bp3_row_count": len(matches),
                        "sequence_context_available": bool(prior),
                        "sequence_separated": separated,
                        "trainable_context_digest": _mapping(
                            result.get("public_sequence_context")
                        ).get("trainable_context_digest"),
                        "matched_bp3_rows_sample": matches[:5],
                    }
                )
    return {
        "policy": "m3_public_sequence_context_bp3_aliasing_v1",
        "bp3_dataset_row_count": len(bp3_dataset_rows),
        "bp3_one_step_action_key_count": len(bp3_one_step),
        "branch_action_evidence_row_count": sum(
            len(_list_of_mappings(result.get("action_runs")))
            for result in branch_results
        ),
        "action_only_alias_count": int(alias_count),
        "sequence_separated_alias_count": int(separated_count),
        "sequence_context_available_alias_count": int(context_available_alias_count),
        "safe_action_only_alias_count": int(safe_alias_count),
        "safe_sequence_separated_alias_count": int(safe_separated_count),
        "alias_action_counts": dict(sorted(action_counts.items())),
        "sequence_separated_action_counts": dict(sorted(separated_action_counts.items())),
        "separation_share": _safe_rate(separated_count, alias_count),
        "safe_separation_share": _safe_rate(safe_separated_count, safe_alias_count),
        "interpretation": (
            "counts compare bp3 one-step observation_input/action_mask/action "
            "aliases with the same rows after adding prior finalized public "
            "transition summaries"
        ),
        "samples": samples,
    }


def _one_step_action_key(features: Mapping[str, object], action: str) -> str:
    return stable_payload_digest(
        {
            "observation_input": _mapping(features.get("observation_input")),
            "action_mask": _bool_action_mask(features.get("action_mask")),
            "action": str(action),
        }
    )


def _sequence_action_key(features: Mapping[str, object], action: str) -> str:
    return stable_payload_digest(
        {
            "observation_input": _mapping(features.get("observation_input")),
            "action_mask": _bool_action_mask(features.get("action_mask")),
            "prior_public_transition_summaries": _list_of_mappings(
                features.get("prior_public_transition_summaries")
            ),
            "action": str(action),
        }
    )


def _coverage_report(branch_results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    branch_counts = Counter(str(result.get("fixture", "unknown")) for result in branch_results)
    by_fixture_seed = Counter(
        f"{result.get('fixture', 'unknown')}:{_int(result.get('seed'))}"
        for result in branch_results
    )
    missing_actions = []
    prior_missing = 0
    for index, result in enumerate(branch_results):
        if not _all_valid_actions_evaluated(result):
            missing_actions.append(_missing_valid_action_detail(index, result))
        context = _mapping(result.get("public_sequence_context"))
        if _int(context.get("history_step_count")) <= 0:
            prior_missing += 1
    return {
        "policy": "m3_public_sequence_context_coverage_v1",
        "branch_result_count": len(branch_results),
        "branch_point_counts": dict(sorted(branch_counts.items())),
        "branch_points_by_fixture_seed": dict(sorted(by_fixture_seed.items())),
        "all_branch_results_have_prior_context": prior_missing == 0,
        "missing_prior_context_count": prior_missing,
        "all_valid_candidate_actions_evaluated": not missing_actions,
        "missing_valid_action_count": len(missing_actions),
        "first_missing_valid_action": next(
            (item for item in missing_actions if item is not None),
            None,
        ),
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
        "policy": "m3_public_sequence_context_replay_verification_v1",
        "complete": (
            action_run_count > 0
            and missing_count == 0
            and failed_count == 0
            and verified_count == action_run_count
        ),
        "action_run_count": action_run_count,
        "verified_count": verified_count,
        "missing_count": missing_count,
        "failed_count": failed_count,
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    generation_status: Mapping[str, object],
    aliasing: Mapping[str, object],
) -> str:
    if _status_partial(generation_status):
        return "m3_public_sequence_context_branch_evidence_partial_no_training"
    if source_integrity.get("passed") is not True:
        return "m3_public_sequence_context_branch_evidence_source_integrity_failed"
    if _int(aliasing.get("action_only_alias_count")) <= 0:
        return "m3_public_sequence_context_branch_evidence_no_bp3_aliases_no_training"
    if _int(aliasing.get("sequence_separated_alias_count")) > 0:
        return "m3_public_sequence_context_branch_evidence_separates_bp3_aliases_no_training"
    return "m3_public_sequence_context_branch_evidence_no_sequence_separation_no_training"


def _recommendation(classification: str, aliasing: Mapping[str, object]) -> dict[str, object]:
    if classification.endswith("separates_bp3_aliases_no_training"):
        return {
            "primary": "build_sequence_context_archive_in_later_slice",
            "training_authorized": False,
            "rationale": (
                "diagnostics found bp3 one-step aliases that become distinct "
                "when prior finalized public transition summaries are included"
            ),
        }
    if _int(aliasing.get("action_only_alias_count")) <= 0:
        return {
            "primary": "collect_more_later_public_sequence_context_branch_points",
            "training_authorized": False,
            "rationale": "no bp3 action-only aliases were observed in this shard",
        }
    return {
        "primary": "stop_bp3_sequence_context_route_for_current_evidence",
        "training_authorized": False,
        "rationale": "available public sequence context did not separate observed aliases",
    }


def _trainable_context_rows(
    branch_results: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    rows = []
    for result in branch_results:
        context = _mapping(_mapping(result.get("public_sequence_context")).get("trainable"))
        if context:
            rows.append(context)
    return rows


def _all_valid_actions_evaluated(result: Mapping[str, object]) -> bool:
    valid = _valid_public_mask_actions(result)
    observed = {
        str(run.get("forced_action"))
        for run in _list_of_mappings(result.get("action_runs"))
        if str(run.get("forced_action")) in ACTION_NAMES
    }
    return bool(valid) and valid.issubset(observed)


def _missing_valid_action_detail(
    result_index: int,
    result: Mapping[str, object],
) -> dict[str, object] | None:
    valid = _valid_public_mask_actions(result)
    observed = {
        str(run.get("forced_action"))
        for run in _list_of_mappings(result.get("action_runs"))
        if str(run.get("forced_action")) in ACTION_NAMES
    }
    missing = sorted(valid - observed, key=_action_order)
    if not missing:
        return None
    return {
        "source_branch_result_index": int(result_index),
        "branch_id": result.get("branch_id"),
        "fixture": result.get("fixture"),
        "seed": result.get("seed"),
        "missing_action": missing[0],
        "missing_actions": missing,
    }


def _valid_public_mask_actions(result: Mapping[str, object]) -> set[str]:
    features = _mapping(result.get("public_features"))
    action_mask = _mapping(features.get("action_mask"))
    valid = {action for action in ACTION_NAMES if bool(action_mask.get(action, False))}
    if valid:
        return valid
    return {
        str(action)
        for action in _list(result.get("candidate_actions"))
        if str(action) in ACTION_NAMES
    }


def _complete_status(
    *,
    started_at: float,
    max_wall_seconds: float | None,
    branch_point_count: int,
    branch_result_count: int,
    generated_count: int,
    resumed_count: int,
    chunk_dir: Path | None,
    fixtures: Sequence[str],
    stop_reason: str | None,
) -> dict[str, object]:
    partial = stop_reason is not None or int(branch_result_count) < int(branch_point_count)
    return {
        "policy": "m3_public_sequence_context_generation_status_v1",
        "state": "partial" if partial else "complete",
        "partial": bool(partial),
        "stop_reason": stop_reason,
        "fixtures": list(_selected_fixtures(fixtures)),
        "elapsed_seconds": _elapsed_seconds(started_at),
        "max_wall_seconds": None if max_wall_seconds is None else float(max_wall_seconds),
        "branch_point_count": int(branch_point_count),
        "branch_result_count": int(branch_result_count),
        "generated_branch_result_count": int(generated_count),
        "resumed_branch_result_count": int(resumed_count),
        "chunk_dir": None if chunk_dir is None else str(chunk_dir),
    }


def _fixture_status(
    *,
    fixture: str,
    started_at: float,
    max_wall_seconds: float | None,
    branch_point_count: int,
    branch_result_count: int,
    generated_count: int,
    resumed_count: int,
    chunk_dir: Path | None,
    stop_reason: str | None,
) -> dict[str, object]:
    partial = stop_reason is not None or int(branch_result_count) < int(branch_point_count)
    return {
        "policy": "m3_public_sequence_context_fixture_generation_status_v1",
        "fixture": fixture,
        "state": "partial" if partial else "complete",
        "partial": bool(partial),
        "stop_reason": stop_reason,
        "elapsed_seconds": _elapsed_seconds(started_at),
        "max_wall_seconds": None if max_wall_seconds is None else float(max_wall_seconds),
        "branch_point_count": int(branch_point_count),
        "branch_result_count": int(branch_result_count),
        "generated_branch_result_count": int(generated_count),
        "resumed_branch_result_count": int(resumed_count),
        "chunk_dir": None if chunk_dir is None else str(chunk_dir),
    }


def _selected_fixtures(fixtures: Sequence[str]) -> tuple[str, ...]:
    selected: list[str] = []
    for fixture in fixtures or ("broad", "carrion_only"):
        value = str(fixture)
        if value == "both":
            for item in ("broad", "carrion_only"):
                if item not in selected:
                    selected.append(item)
            continue
        if value not in {"broad", "carrion_only"}:
            raise PublicSequenceContextBranchEvidenceError(
                f"unsupported fixture: {value}"
            )
        if value not in selected:
            selected.append(value)
    return tuple(selected or ["broad", "carrion_only"])


def _merge_identity(report: Mapping[str, object]) -> dict[str, object]:
    inputs = _mapping(report.get("inputs"))
    return {
        "schema_version": report.get("schema_version"),
        "policy": report.get("policy"),
        "max_branch_points_per_seed": inputs.get("max_branch_points_per_seed"),
        "max_candidate_actions": inputs.get("max_candidate_actions"),
        "history_window": inputs.get("history_window"),
        "min_prior_public_steps": inputs.get("min_prior_public_steps"),
        "bp3_safe_archive_report_digest": inputs.get(
            "bp3_safe_archive_report_digest"
        ),
        "bp3_dataset_digest": inputs.get("bp3_dataset_digest"),
        "bp3_branch_evidence_digest": inputs.get("bp3_branch_evidence_digest"),
    }


def _add_merged_branch_result(
    *,
    merged: dict[str, dict[str, object]],
    digests_by_branch_id: dict[str, str],
    result: Mapping[str, object],
    source: str,
) -> None:
    branch_id = str(result.get("branch_id", ""))
    if not branch_id:
        raise PublicSequenceContextBranchEvidenceError(
            f"merged branch result missing branch_id: {source}"
        )
    digest = stable_payload_digest(result)
    prior = digests_by_branch_id.get(branch_id)
    if prior is not None and prior != digest:
        raise PublicSequenceContextBranchEvidenceError(
            f"duplicate branch_id with different digest: {branch_id}"
        )
    merged[branch_id] = dict(result)
    digests_by_branch_id[branch_id] = digest


def _validate_authorization_flags(
    *,
    name: str,
    payload: Mapping[str, object],
    failures: list[str],
    missing_is_drift: bool = False,
) -> None:
    for key in (
        "training_authorized",
        "promotion_authorized",
        "runtime_promotion_allowed",
    ):
        value = payload.get(key)
        if value is True:
            failures.append(f"{name}_{key}_true")
        if missing_is_drift and value is None:
            failures.append(f"{name}_{key}_missing")


def _input_path_payload(input_paths: Mapping[str, str | Path] | None) -> dict[str, str]:
    defaults = {
        "bp3_safe_archive_report": DEFAULT_BP3_SAFE_ARCHIVE_REPORT_PATH,
        "bp3_dataset": DEFAULT_BP3_DATASET_PATH,
        "bp3_branch_evidence": DEFAULT_BP3_BRANCH_EVIDENCE_PATH,
        "prior_sequence_context_audit": DEFAULT_BP3_SEQUENCE_CONTEXT_AUDIT_OUTPUT_PATH,
    }
    values = dict(defaults)
    if input_paths is not None:
        for key, value in input_paths.items():
            if key in values:
                values[key] = Path(value)
    return {key: str(value) for key, value in values.items()}


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


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _bool_action_mask(value: object) -> dict[str, bool]:
    payload = _mapping(value)
    return {action: bool(payload.get(action, False)) for action in ACTION_NAMES}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _flatten(value: object, *, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, object]] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(child, prefix=path))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            rows.extend(_flatten(child, prefix=path))
        return rows
    return [(prefix, value)]


def _looks_like_digest(value: str) -> bool:
    lower = value.lower()
    return len(value) >= 32 and all(char in "0123456789abcdef" for char in lower)


def _int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 6)


def _optional_delta(
    after: Mapping[str, object],
    before: Mapping[str, object],
    field: str,
) -> float | None:
    after_value = _optional_float(after.get(field))
    before_value = _optional_float(before.get(field))
    if after_value is None or before_value is None:
        return None
    return round(after_value - before_value, 6)


def _action_order(action: str) -> int:
    return ACTION_NAMES.index(action) if action in ACTION_NAMES else len(ACTION_NAMES)


def _fixture_order(fixture: str) -> int:
    if fixture == "broad":
        return 0
    if fixture == "carrion_only":
        return 1
    return 2
