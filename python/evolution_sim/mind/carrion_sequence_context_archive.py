from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from evolution_sim.mind import evaluation_harness as evaluate_cli
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.carrion_archive_override_autopsy import (
    DEFAULT_V148_CARRION_ARCHIVE_DATASET_PATH,
    DEFAULT_V148_CARRION_ARCHIVE_REPORT_PATH,
    DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH,
    M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_POLICY,
    M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION,
    V149_FAILED_NON_PROMOTIONAL_CLASSIFICATION,
    V149_CARRION_TRAIN_EVAL_POLICY,
    V149_CARRION_TRAIN_EVAL_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_specific_archive_expansion import (
    CARRION_BRANCH_REASONS,
    DEFAULT_TICKS,
    M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_POLICY,
    M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_SCHEMA_VERSION,
    TARGET_CARRION_SEEDS,
)
from evolution_sim.mind.candidate_campaign import (
    _configure_branch_manual_summary_run,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION = (
    "m3_carrion_hydration_reproduction_sequence_context_archive_report_v1"
)
M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_POLICY = (
    "diagnostics_only_m3_carrion_hydration_reproduction_sequence_context_archive_v1"
)
M3_CARRION_SEQUENCE_CONTEXT_FEATURE_POLICY = (
    "public_observation_action_mask_with_prior_public_hydration_reproduction_sequence_context_v1"
)
M3_CARRION_SEQUENCE_CONTEXT_BRANCH_RESULT_CHUNK_SCHEMA_VERSION = (
    "m3_carrion_sequence_context_archive_branch_result_chunk_v1"
)
M3_CARRION_SEQUENCE_CONTEXT_BRANCH_RESULT_CHUNK_POLICY = (
    "diagnostics_only_m3_carrion_sequence_context_archive_branch_result_chunk_v1"
)

DEFAULT_V150_CARRION_OVERRIDE_AUTOPSY_PATH = Path(
    "output/mind/mind-v3-v150-carrion-archive-override-autopsy.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v151-carrion-sequence-context-archive.json"
)
DEFAULT_DATASET_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v151-carrion-sequence-context-archive-dataset.jsonl"
)
DEFAULT_BRANCH_RESULT_CHUNK_DIR = Path(
    "output/mind/mind-v3-v151-carrion-sequence-context-archive-chunks"
)
DEFAULT_HISTORY_WINDOW = 8
DEFAULT_MIN_PRIOR_PUBLIC_STEPS = 0
DEFAULT_MIN_SAFE_COMPARATOR_COUNT = 1
DEFAULT_HARMFUL_SUPPORT_SOURCES = (
    {
        "label_action": "move_north",
        "source_seed": 29,
        "source_branch_reason": "carrion_contact",
        "source_branch_id": (
            "m3-carrion-specific-archive-seed-29-carrion-contact-branch-0-tick-0-agent-9"
        ),
    },
    {
        "label_action": "move_east",
        "source_seed": 41,
        "source_branch_reason": "movement_stall",
        "source_branch_id": (
            "m3-carrion-specific-archive-seed-41-movement-stall-branch-2-tick-0-agent-10"
        ),
    },
    {
        "label_action": "move_south",
        "source_seed": 19,
        "source_branch_reason": "movement_stall",
        "source_branch_id": (
            "m3-carrion-specific-archive-seed-19-movement-stall-branch-2-tick-0-agent-12"
        ),
    },
)
FORBIDDEN_TRAINABLE_INPUT_TOKENS = (
    "seed",
    "fixture",
    "branch",
    "tick",
    "agent",
    "path",
    "digest",
    "private",
    "world",
    "future",
    "outcome",
    "label",
    "source",
    "trajectory",
    "record_index",
)


class CarrionSequenceContextArchiveError(ValueError):
    pass


def load_json_report(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CarrionSequenceContextArchiveError(
            f"JSON report must be an object: {path}"
        )
    return payload


def load_jsonl_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise CarrionSequenceContextArchiveError(
                    f"JSONL row {line_number} must be an object: {path}"
                )
            rows.append(payload)
    return rows


def write_carrion_sequence_context_archive_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_carrion_sequence_context_archive_dataset(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(dict(row), handle, sort_keys=True, allow_nan=False)
            handle.write("\n")


def write_carrion_sequence_context_branch_result_chunk(
    branch_result: Mapping[str, object],
    chunk_dir: str | Path,
) -> Path:
    branch_id = str(branch_result.get("branch_id", ""))
    if not branch_id:
        raise CarrionSequenceContextArchiveError(
            "cannot write v151 branch-result chunk without branch_id"
        )
    path = Path(chunk_dir)
    path.mkdir(parents=True, exist_ok=True)
    chunk_path = path / f"{_slug(branch_id)}.json"
    payload = {
        "schema_version": M3_CARRION_SEQUENCE_CONTEXT_BRANCH_RESULT_CHUNK_SCHEMA_VERSION,
        "policy": M3_CARRION_SEQUENCE_CONTEXT_BRANCH_RESULT_CHUNK_POLICY,
        "branch_result_digest": stable_payload_digest(branch_result),
        "branch_result": dict(branch_result),
    }
    with chunk_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return chunk_path


def load_carrion_sequence_context_branch_result_chunks(
    chunk_dir: str | Path,
) -> dict[str, dict[str, object]]:
    path = Path(chunk_dir)
    if not path.exists():
        return {}
    if not path.is_dir():
        raise CarrionSequenceContextArchiveError(
            f"v151 chunk path is not a directory: {path}"
        )
    chunks: dict[str, dict[str, object]] = {}
    digests_by_branch_id: dict[str, str] = {}
    for chunk_path in sorted(path.glob("*.json")):
        payload = load_json_report(chunk_path)
        if payload.get("schema_version") != (
            M3_CARRION_SEQUENCE_CONTEXT_BRANCH_RESULT_CHUNK_SCHEMA_VERSION
        ):
            raise CarrionSequenceContextArchiveError(
                f"v151 branch-result chunk schema mismatch: {chunk_path}"
            )
        if payload.get("policy") != M3_CARRION_SEQUENCE_CONTEXT_BRANCH_RESULT_CHUNK_POLICY:
            raise CarrionSequenceContextArchiveError(
                f"v151 branch-result chunk policy mismatch: {chunk_path}"
            )
        result = _mapping(payload.get("branch_result"))
        branch_id = str(result.get("branch_id", ""))
        if not branch_id:
            raise CarrionSequenceContextArchiveError(
                f"v151 branch-result chunk missing branch_id: {chunk_path}"
            )
        expected_digest = str(payload.get("branch_result_digest", ""))
        actual_digest = stable_payload_digest(result)
        if expected_digest and expected_digest != actual_digest:
            raise CarrionSequenceContextArchiveError(
                f"v151 branch-result chunk digest mismatch: {chunk_path}"
            )
        prior_digest = digests_by_branch_id.get(branch_id)
        if prior_digest is not None and prior_digest != actual_digest:
            raise CarrionSequenceContextArchiveError(
                f"conflicting v151 branch-result chunks for branch_id={branch_id}"
            )
        chunks[branch_id] = dict(result)
        digests_by_branch_id[branch_id] = actual_digest
    return dict(sorted(chunks.items()))


def build_carrion_sequence_context_archive_report_from_paths(
    *,
    archive_report_path: str | Path = DEFAULT_V148_CARRION_ARCHIVE_REPORT_PATH,
    archive_dataset_path: str | Path = DEFAULT_V148_CARRION_ARCHIVE_DATASET_PATH,
    train_eval_report_path: str | Path = DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH,
    autopsy_report_path: str | Path = DEFAULT_V150_CARRION_OVERRIDE_AUTOPSY_PATH,
    **kwargs: object,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    return build_carrion_sequence_context_archive_report(
        archive_report=load_json_report(archive_report_path),
        dataset_rows=load_jsonl_rows(archive_dataset_path),
        train_eval_report=load_json_report(train_eval_report_path),
        autopsy_report=load_json_report(autopsy_report_path),
        input_paths={
            "v148_carrion_archive_report": archive_report_path,
            "v148_carrion_archive_dataset": archive_dataset_path,
            "v149_carrion_train_eval_report": train_eval_report_path,
            "v150_carrion_override_autopsy": autopsy_report_path,
        },
        **kwargs,
    )


def build_carrion_sequence_context_archive_report(
    *,
    archive_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    train_eval_report: Mapping[str, object],
    autopsy_report: Mapping[str, object],
    branch_results: Sequence[Mapping[str, object]] | None = None,
    prior_context_by_branch_id: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    generation_status: Mapping[str, object] | None = None,
    generation_evidence: Mapping[str, object] | None = None,
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
    seed_include: Sequence[int] | None = None,
    branch_index_include: Sequence[int] | None = None,
    ticks: int = DEFAULT_TICKS,
    history_window: int = DEFAULT_HISTORY_WINDOW,
    min_prior_public_steps: int = DEFAULT_MIN_PRIOR_PUBLIC_STEPS,
    min_safe_comparator_count: int = DEFAULT_MIN_SAFE_COMPARATOR_COUNT,
    branch_result_chunk_dir: str | Path | None = None,
    resume_branch_results: bool = False,
    max_wall_seconds: float | None = None,
    shard_id: str | None = None,
    input_paths: Mapping[str, str | Path] | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    validation = validate_carrion_sequence_context_inputs(
        archive_report=archive_report,
        dataset_rows=dataset_rows,
        train_eval_report=train_eval_report,
        autopsy_report=autopsy_report,
    )
    harmful_sources = _harmful_sources_from_autopsy(autopsy_report)
    if branch_results is None:
        generated = generate_carrion_sequence_context_branch_results(
            archive_report=archive_report,
            dataset_rows=dataset_rows,
            harmful_sources=harmful_sources,
            prior_context_by_branch_id=prior_context_by_branch_id,
            target_seeds=target_seeds,
            seed_include=seed_include,
            branch_index_include=branch_index_include,
            ticks=int(ticks),
            history_window=int(history_window),
            min_prior_public_steps=int(min_prior_public_steps),
            branch_result_chunk_dir=branch_result_chunk_dir,
            resume_branch_results=bool(resume_branch_results),
            max_wall_seconds=max_wall_seconds,
            shard_id=shard_id,
            progress_callback=progress_callback,
        )
        resolved_branch_results = _list_of_mappings(generated.get("branch_results"))
        resolved_generation_status = _mapping(generated.get("generation_status"))
        resolved_generation_evidence = {
            key: value for key, value in dict(generated).items() if key != "branch_results"
        }
    else:
        resolved_branch_results = [dict(result) for result in branch_results]
        resolved_generation_status = dict(
            generation_status
            or _precomputed_generation_status(
                branch_results=resolved_branch_results,
                target_seeds=target_seeds,
                seed_include=seed_include,
                branch_index_include=branch_index_include,
                ticks=int(ticks),
                history_window=int(history_window),
                min_prior_public_steps=int(min_prior_public_steps),
                shard_id=shard_id,
            )
        )
        resolved_generation_evidence = dict(
            generation_evidence
            or {
                "policy": "m3_carrion_sequence_context_precomputed_evidence_v1",
                "shard_id": shard_id,
            }
        )
    return _finalize_report(
        archive_report=archive_report,
        dataset_rows=dataset_rows,
        train_eval_report=train_eval_report,
        autopsy_report=autopsy_report,
        source_validation=validation,
        harmful_sources=harmful_sources,
        branch_results=resolved_branch_results,
        generation_status=resolved_generation_status,
        generation_evidence=resolved_generation_evidence,
        target_seeds=target_seeds,
        seed_include=seed_include,
        branch_index_include=branch_index_include,
        ticks=int(ticks),
        history_window=int(history_window),
        min_prior_public_steps=int(min_prior_public_steps),
        min_safe_comparator_count=int(min_safe_comparator_count),
        branch_result_chunk_dir=branch_result_chunk_dir,
        resume_branch_results=bool(resume_branch_results),
        max_wall_seconds=max_wall_seconds,
        shard_id=shard_id,
        input_paths=input_paths,
    )


def generate_carrion_sequence_context_branch_results(
    *,
    archive_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    harmful_sources: Sequence[Mapping[str, object]],
    prior_context_by_branch_id: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
    seed_include: Sequence[int] | None = None,
    branch_index_include: Sequence[int] | None = None,
    ticks: int = DEFAULT_TICKS,
    history_window: int = DEFAULT_HISTORY_WINDOW,
    min_prior_public_steps: int = DEFAULT_MIN_PRIOR_PUBLIC_STEPS,
    branch_result_chunk_dir: str | Path | None = None,
    resume_branch_results: bool = False,
    max_wall_seconds: float | None = None,
    shard_id: str | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    started_at = time.monotonic()
    seed_filter = None if seed_include is None else {int(seed) for seed in seed_include}
    branch_filter = (
        None
        if branch_index_include is None
        else {int(index) for index in branch_index_include}
    )
    target_seed_set = {int(seed) for seed in target_seeds}
    archive_branch_results = _list_of_mappings(archive_report.get("branch_results"))
    selected_archive_results = [
        result
        for result in sorted(archive_branch_results, key=_archive_branch_result_sort_key)
        if _archive_result_selected(
            result,
            target_seed_set=target_seed_set,
            seed_filter=seed_filter,
            branch_filter=branch_filter,
        )
    ]
    rows_by_branch = _selected_dataset_rows_by_branch(
        dataset_rows=dataset_rows,
        archive_branch_results=selected_archive_results,
        harmful_sources=harmful_sources,
    )
    needed_results = [
        result
        for result in selected_archive_results
        if str(result.get("branch_id", "")) in rows_by_branch
    ]
    chunk_dir = Path(branch_result_chunk_dir) if branch_result_chunk_dir else None
    resumed = (
        load_carrion_sequence_context_branch_result_chunks(chunk_dir)
        if chunk_dir is not None and bool(resume_branch_results)
        else {}
    )
    if prior_context_by_branch_id is None:
        context_payload = materialize_carrion_prior_public_contexts(
            branch_results=needed_results,
            ticks=int(ticks),
            history_window=int(history_window),
            started_at=started_at,
            max_wall_seconds=max_wall_seconds,
        )
        prior_contexts = {
            str(key): _list_of_mappings(value)
            for key, value in _mapping(context_payload.get("contexts")).items()
        }
        context_evidence = dict(context_payload)
    else:
        prior_contexts = {
            str(key): [dict(item) for item in _list_of_mappings(value)]
            for key, value in prior_context_by_branch_id.items()
        }
        context_evidence = {
            "policy": "m3_carrion_sequence_context_supplied_prior_contexts_v1",
            "materialized": False,
            "context_count": len(prior_contexts),
            "failures": [],
            "partial": False,
        }
    branch_results: list[dict[str, object]] = []
    generated_count = 0
    resumed_count = 0
    missing_contexts = []
    stop_reason = None
    for result in needed_results:
        if _wall_budget_exhausted(started_at, max_wall_seconds):
            stop_reason = "max_wall_seconds_elapsed_before_sequence_branch"
            break
        branch_id = str(result.get("branch_id", ""))
        prior = prior_contexts.get(branch_id)
        if prior is None:
            missing_contexts.append(branch_id)
            continue
        source_rows = rows_by_branch.get(branch_id, [])
        cached = resumed.get(branch_id)
        if cached is not None and _v151_branch_result_matches(
            cached,
            archive_result=result,
            source_rows=source_rows,
            prior_context=prior,
            history_window=int(history_window),
            min_prior_public_steps=int(min_prior_public_steps),
        ):
            branch_results.append(dict(cached))
            resumed_count += 1
            source = "resumed"
        else:
            built = _sequence_branch_result(
                archive_result=result,
                dataset_rows=source_rows,
                harmful_sources=harmful_sources,
                prior_context=prior,
                history_window=int(history_window),
                min_prior_public_steps=int(min_prior_public_steps),
            )
            branch_results.append(built)
            generated_count += 1
            source = "generated"
            if chunk_dir is not None:
                write_carrion_sequence_context_branch_result_chunk(built, chunk_dir)
        _emit_progress(
            progress_callback,
            {
                "event": "sequence_branch_result",
                "source": source,
                "shard_id": shard_id,
                "fixture": result.get("fixture"),
                "seed": result.get("seed"),
                "branch_id": branch_id,
                "branch_index": result.get("branch_index"),
                "branch_reason": _branch_reason(result),
                "source_row_count": len(source_rows),
                "elapsed_seconds": _elapsed_seconds(started_at),
            },
        )
    partial = (
        stop_reason is not None
        or bool(missing_contexts)
        or bool(context_evidence.get("partial"))
        or len(branch_results) < len(needed_results)
    )
    generation_status = {
        "policy": "m3_carrion_sequence_context_generation_status_v1",
        "state": "partial" if partial else "complete",
        "partial": bool(partial),
        "stop_reason": (
            stop_reason
            or ("missing_prior_public_context" if missing_contexts else None)
            or (
                "partial_prior_context_materialization"
                if context_evidence.get("partial")
                else None
            )
        ),
        "elapsed_seconds": _elapsed_seconds(started_at),
        "max_wall_seconds": None if max_wall_seconds is None else float(max_wall_seconds),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in target_seeds],
        "seed_include": None if seed_filter is None else sorted(seed_filter),
        "branch_index_include": None if branch_filter is None else sorted(branch_filter),
        "ticks": int(ticks),
        "history_window": int(history_window),
        "min_prior_public_steps": int(min_prior_public_steps),
        "branch_result_count": len(branch_results),
        "selected_archive_branch_result_count": len(selected_archive_results),
        "needed_archive_branch_result_count": len(needed_results),
        "generated_branch_result_count": int(generated_count),
        "resumed_branch_result_count": int(resumed_count),
        "missing_prior_context_branch_count": len(missing_contexts),
        "shard_id": shard_id,
        "chunk_dir": None if chunk_dir is None else str(chunk_dir),
    }
    return {
        "policy": "m3_carrion_sequence_context_generated_branch_results_v1",
        "shard_id": shard_id,
        "branch_results": branch_results,
        "branch_result_count": len(branch_results),
        "branch_evidence_digest": stable_payload_digest(branch_results),
        "generation_status": generation_status,
        "generation_evidence": {
            "policy": "m3_carrion_sequence_context_generation_evidence_v1",
            "shard_id": shard_id,
            "target_harmful_sources": [dict(source) for source in harmful_sources],
            "harmful_actions": sorted(
                {str(source.get("label_action", "")) for source in harmful_sources}
            ),
            "prior_context_materialization": context_evidence,
            "resumed_chunk_count": len(resumed),
        },
    }


def materialize_carrion_prior_public_contexts(
    *,
    branch_results: Sequence[Mapping[str, object]],
    ticks: int,
    history_window: int,
    started_at: float | None = None,
    max_wall_seconds: float | None = None,
) -> dict[str, object]:
    started = time.monotonic() if started_at is None else float(started_at)
    needed: dict[int, dict[int, str]] = {}
    branch_identity_by_id: dict[str, dict[str, object]] = {}
    for result in branch_results:
        seed = _int(result.get("seed"), default=-1)
        record_index = _int(result.get("record_index"), default=-1)
        branch_id = str(result.get("branch_id", ""))
        if seed < 0 or record_index < 0 or not branch_id:
            continue
        needed.setdefault(seed, {})[record_index] = branch_id
        branch_identity_by_id[branch_id] = {
            "seed": seed,
            "record_index": record_index,
            "branch_tick": _int(result.get("branch_tick")),
            "agent_id": _int(result.get("agent_id")),
        }
    contexts: dict[str, list[dict[str, object]]] = {}
    seed_reports: list[dict[str, object]] = []
    stop_reason = None
    for seed in sorted(needed):
        if _wall_budget_exhausted(started, max_wall_seconds):
            stop_reason = "max_wall_seconds_elapsed_before_seed_context"
            break
        seed_report = _materialize_seed_prior_public_contexts(
            seed=seed,
            ticks=int(ticks),
            history_window=int(history_window),
            needed_record_indexes=needed[seed],
            contexts=contexts,
            started_at=started,
            max_wall_seconds=max_wall_seconds,
        )
        seed_reports.append(seed_report)
        if seed_report.get("stop_reason") is not None:
            stop_reason = str(seed_report.get("stop_reason"))
            break
    missing_branch_ids = sorted(set(branch_identity_by_id) - set(contexts))
    partial = bool(stop_reason) or bool(missing_branch_ids)
    return {
        "policy": "m3_carrion_sequence_context_prior_public_context_materialization_v1",
        "materialized": True,
        "partial": bool(partial),
        "stop_reason": stop_reason,
        "ticks": int(ticks),
        "history_window": int(history_window),
        "needed_branch_count": len(branch_identity_by_id),
        "context_count": len(contexts),
        "missing_branch_ids": missing_branch_ids[:32],
        "seed_reports": seed_reports,
        "contexts": contexts,
    }


def merge_carrion_sequence_context_archive_shards(
    *,
    archive_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    train_eval_report: Mapping[str, object],
    autopsy_report: Mapping[str, object],
    shard_reports: Sequence[Mapping[str, object]],
    shard_report_paths: Sequence[str | Path] = (),
    shard_chunk_dirs: Sequence[str | Path] = (),
    allow_partial_shard_evidence: bool = False,
    input_paths: Mapping[str, str | Path] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not shard_reports:
        raise CarrionSequenceContextArchiveError(
            "v151 carrion sequence-context shard merge requires reports"
        )
    validation = validate_carrion_sequence_context_inputs(
        archive_report=archive_report,
        dataset_rows=dataset_rows,
        train_eval_report=train_eval_report,
        autopsy_report=autopsy_report,
    )
    harmful_sources = _harmful_sources_from_autopsy(autopsy_report)
    merged_by_branch_id: dict[str, dict[str, object]] = {}
    digests_by_branch_id: dict[str, str] = {}
    source_summaries: list[dict[str, object]] = []
    partial_sources: list[dict[str, object]] = []
    merge_identity: dict[str, object] | None = None
    source_paths = [str(path) for path in shard_report_paths]
    for shard_index, report in enumerate(shard_reports):
        _validate_shard_contract(report, shard_index=shard_index)
        identity = _merge_identity(report)
        if merge_identity is None:
            merge_identity = identity
        elif identity != merge_identity:
            raise CarrionSequenceContextArchiveError(
                f"shard {shard_index} schema/policy/target/tick/min-floor mismatch"
            )
        status = _mapping(report.get("generation_status"))
        source_integrity = _mapping(report.get("source_integrity"))
        integrity_failures = [
            str(failure) for failure in _list(source_integrity.get("failures"))
        ]
        partial = _status_partial(status) or "partial_sequence_context_evidence" in set(
            integrity_failures
        )
        non_partial_failures = sorted(
            failure
            for failure in integrity_failures
            if failure != "partial_sequence_context_evidence"
        )
        if source_integrity.get("passed") is not True and non_partial_failures:
            raise CarrionSequenceContextArchiveError(
                f"shard {shard_index} source integrity failed: {non_partial_failures}"
            )
        source_name = _shard_source_name(report, shard_index=shard_index)
        if partial:
            summary = {
                "source": source_name,
                "source_path": _optional_index(source_paths, shard_index),
                "shard_id": _shard_id_from_report(report),
                "state": status.get("state"),
                "stop_reason": status.get("stop_reason"),
                "source_integrity_failures": integrity_failures,
            }
            if not allow_partial_shard_evidence:
                raise CarrionSequenceContextArchiveError(
                    f"partial shard evidence requires explicit partial merge: {summary}"
                )
            partial_sources.append(summary)
        branch_results = _list_of_mappings(report.get("branch_results"))
        expected_digest = str(report.get("branch_evidence_digest", ""))
        actual_digest = stable_payload_digest(branch_results)
        if expected_digest and expected_digest != actual_digest:
            raise CarrionSequenceContextArchiveError(
                f"shard {shard_index} branch evidence digest mismatch"
            )
        for result in branch_results:
            _add_merged_branch_result(
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
                "partial": bool(partial),
            }
        )
    for chunk_dir in shard_chunk_dirs:
        chunks = load_carrion_sequence_context_branch_result_chunks(chunk_dir)
        for result in chunks.values():
            branch_id = str(result.get("branch_id", ""))
            if branch_id not in merged_by_branch_id:
                raise CarrionSequenceContextArchiveError(
                    "chunk-dir branch result lacks matching shard report: "
                    f"{chunk_dir}:{branch_id}"
                )
            _add_merged_branch_result(
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
        raise CarrionSequenceContextArchiveError(
            "v151 carrion sequence-context shard merge has no branch results"
        )
    branch_results = sorted(
        merged_by_branch_id.values(),
        key=_v151_branch_result_sort_key,
    )
    merged_status = {
        "policy": "m3_carrion_sequence_context_merged_generation_status_v1",
        "state": "partial" if partial_sources else "complete",
        "partial": bool(partial_sources),
        "stop_reason": "partial_shard_evidence" if partial_sources else None,
        "source_count": len(source_summaries),
        "partial_sources": sorted(
            partial_sources,
            key=lambda item: str(item.get("source", "")),
        ),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": list(_list(_mapping(merge_identity).get("target_carrion_seeds"))),
        "ticks": _mapping(merge_identity).get("ticks"),
        "history_window": _mapping(merge_identity).get("history_window"),
        "min_prior_public_steps": _mapping(merge_identity).get(
            "min_prior_public_steps"
        ),
        "min_safe_comparator_count": _mapping(merge_identity).get(
            "min_safe_comparator_count"
        ),
        "branch_result_count": len(branch_results),
    }
    merged_evidence = {
        "policy": "m3_carrion_sequence_context_shard_merge_generation_evidence_v1",
        "merge_mode": True,
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "source_count": len(source_summaries),
        "branch_evidence_digest": stable_payload_digest(branch_results),
    }
    report, rows = _finalize_report(
        archive_report=archive_report,
        dataset_rows=dataset_rows,
        train_eval_report=train_eval_report,
        autopsy_report=autopsy_report,
        source_validation=validation,
        harmful_sources=harmful_sources,
        branch_results=branch_results,
        generation_status=merged_status,
        generation_evidence=merged_evidence,
        target_seeds=tuple(int(seed) for seed in _list(merged_status.get("target_carrion_seeds"))),
        seed_include=None,
        branch_index_include=None,
        ticks=_int(merged_status.get("ticks"), default=DEFAULT_TICKS),
        history_window=_int(merged_status.get("history_window"), default=DEFAULT_HISTORY_WINDOW),
        min_prior_public_steps=_int(
            merged_status.get("min_prior_public_steps"),
            default=DEFAULT_MIN_PRIOR_PUBLIC_STEPS,
        ),
        min_safe_comparator_count=_int(
            merged_status.get("min_safe_comparator_count"),
            default=DEFAULT_MIN_SAFE_COMPARATOR_COUNT,
        ),
        branch_result_chunk_dir=None,
        resume_branch_results=False,
        max_wall_seconds=None,
        shard_id=None,
        input_paths=input_paths,
    )
    report["shard_merge"] = {
        "policy": "m3_carrion_sequence_context_archive_shard_merge_v1",
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "sources": sorted(
            source_summaries,
            key=lambda item: (
                str(item.get("source", "")),
                str(item.get("source_path", "")),
                str(item.get("branch_evidence_digest", "")),
            ),
        ),
        "partial_sources": merged_status["partial_sources"],
        "duplicate_branch_id_policy": "same_digest_allowed_conflict_rejected",
    }
    return report, rows


def validate_carrion_sequence_context_inputs(
    *,
    archive_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    train_eval_report: Mapping[str, object],
    autopsy_report: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    archive_branch_results = _list_of_mappings(archive_report.get("branch_results"))
    dataset_digest = stable_payload_digest(list(dataset_rows))
    branch_evidence_digest = stable_payload_digest(archive_branch_results)
    train_eval_digest = stable_payload_digest(train_eval_report)
    autopsy_digest = stable_payload_digest(autopsy_report)
    archive_dataset = _mapping(archive_report.get("dataset"))
    archive_integrity = _mapping(archive_report.get("source_integrity"))
    archive_status = _mapping(archive_report.get("generation_status"))
    train_validation = _mapping(train_eval_report.get("validation"))
    train_acceptance = _mapping(train_eval_report.get("acceptance"))
    train_classification = _mapping(train_eval_report.get("classification")).get(
        "primary"
    )
    autopsy_inputs = _mapping(autopsy_report.get("inputs"))
    autopsy_validation = _mapping(autopsy_inputs.get("input_validation"))
    autopsy_failure_mode = _mapping(autopsy_report.get("failure_mode_classification"))

    if archive_report.get("schema_version") != (
        M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_SCHEMA_VERSION
    ):
        failures.append("v148_archive_schema_mismatch")
    if archive_report.get("policy") != M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_POLICY:
        failures.append("v148_archive_policy_mismatch")
    if archive_report.get("diagnostics_only") is not True:
        failures.append("v148_archive_diagnostics_only_not_true")
    if archive_integrity.get("passed") is not True:
        failures.append("v148_archive_source_integrity_not_passed")
    if _list(archive_integrity.get("failures")):
        failures.append("v148_archive_source_integrity_failures_present")
    if archive_status.get("state") != "complete" or archive_status.get("partial") is True:
        failures.append("v148_archive_generation_not_complete")
    if archive_dataset.get("dataset_digest") != dataset_digest:
        failures.append("v148_archive_dataset_digest_mismatch")
    if _int(archive_dataset.get("safe_label_count")) != len(dataset_rows):
        failures.append("v148_archive_safe_label_count_mismatch")
    if _int(archive_report.get("branch_result_count")) != len(archive_branch_results):
        failures.append("v148_archive_branch_result_count_mismatch")
    if archive_report.get("branch_evidence_digest") != branch_evidence_digest:
        failures.append("v148_archive_branch_evidence_digest_mismatch")
    if not dataset_rows:
        failures.append("v148_dataset_rows_missing")

    if train_eval_report.get("schema_version") != V149_CARRION_TRAIN_EVAL_SCHEMA_VERSION:
        failures.append("v149_train_eval_schema_mismatch")
    if train_eval_report.get("policy") != V149_CARRION_TRAIN_EVAL_POLICY:
        failures.append("v149_train_eval_policy_mismatch")
    if train_classification != V149_FAILED_NON_PROMOTIONAL_CLASSIFICATION:
        failures.append("v149_train_eval_not_failed_non_promotional")
    if train_validation.get("passed") is not True:
        failures.append("v149_train_eval_validation_not_passed")
    if _list(train_validation.get("failures")):
        failures.append("v149_train_eval_validation_failures_present")
    if train_acceptance.get("passed") is not False:
        failures.append("v149_train_eval_acceptance_not_failed")
    if not _list_of_mappings(train_acceptance.get("blockers")):
        failures.append("v149_train_eval_blockers_missing")
    if train_validation.get("dataset_digest") != dataset_digest:
        failures.append("v149_train_eval_dataset_digest_mismatch")
    if train_validation.get("branch_evidence_digest") != archive_report.get(
        "branch_evidence_digest"
    ):
        failures.append("v149_train_eval_branch_evidence_digest_mismatch")

    if autopsy_report.get("schema_version") != (
        M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION
    ):
        failures.append("v150_autopsy_schema_mismatch")
    if autopsy_report.get("policy") != M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_POLICY:
        failures.append("v150_autopsy_policy_mismatch")
    if autopsy_report.get("diagnostics_only") is not True:
        failures.append("v150_autopsy_diagnostics_only_not_true")
    if _mapping(autopsy_report.get("classification")).get("primary") != (
        "m3_carrion_archive_override_autopsy_complete_no_training"
    ):
        failures.append("v150_autopsy_not_complete")
    if autopsy_validation.get("passed") is not True:
        failures.append("v150_autopsy_input_validation_not_passed")
    if _list(autopsy_validation.get("failures")):
        failures.append("v150_autopsy_input_validation_failures_present")
    if autopsy_failure_mode.get("primary") != "missing_hydration_reproduction_context":
        failures.append("v150_autopsy_primary_blocker_mismatch")
    if autopsy_report.get("recommended_next_route") != (
        "build_carrion_hydration_reproduction_sequence_context_archive"
    ):
        failures.append("v150_autopsy_recommended_next_route_mismatch")
    if autopsy_inputs.get("dataset_digest") != dataset_digest:
        failures.append("v150_autopsy_dataset_digest_mismatch")
    if autopsy_inputs.get("branch_evidence_digest") != archive_report.get(
        "branch_evidence_digest"
    ):
        failures.append("v150_autopsy_branch_evidence_digest_mismatch")
    if autopsy_inputs.get("train_eval_report_digest") != train_eval_digest:
        failures.append("v150_autopsy_train_eval_digest_mismatch")

    harmful_sources = _harmful_sources_from_autopsy(autopsy_report, fail_closed=False)
    if len(harmful_sources) != len(DEFAULT_HARMFUL_SUPPORT_SOURCES):
        failures.append("v150_harmful_support_sources_missing")

    for name, payload in (
        ("v148_archive", archive_report),
        ("v149_train_eval", train_eval_report),
        ("v150_autopsy", autopsy_report),
    ):
        _validate_authorization_flags(name=name, payload=payload, failures=failures)
    if failures:
        raise CarrionSequenceContextArchiveError(
            "v151 carrion sequence-context input validation failed: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "policy": "m3_carrion_sequence_context_input_validation_v1",
        "passed": True,
        "failures": [],
        "v148_dataset_digest": dataset_digest,
        "v148_branch_evidence_digest": branch_evidence_digest,
        "v149_train_eval_report_digest": train_eval_digest,
        "v150_autopsy_report_digest": autopsy_digest,
        "harmful_support_source_count": len(harmful_sources),
    }


def carrion_sequence_context_trainable_leakage_scan(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    finite_failures: list[dict[str, object]] = []
    label_feature_failures: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        features = _mapping(_mapping(row.get("trainable")).get("features"))
        for path, value in _flatten(features):
            lower_path = path.lower()
            if any(token in lower_path for token in FORBIDDEN_TRAINABLE_INPUT_TOKENS):
                failures.append(
                    {
                        "row_index": int(row_index),
                        "path": f"features.{path}" if path else "features",
                        "reason": "forbidden_trainable_input_path_token",
                    }
                )
            if isinstance(value, str) and _looks_like_digest(value):
                failures.append(
                    {
                        "row_index": int(row_index),
                        "path": f"features.{path}" if path else "features",
                        "reason": "forbidden_trainable_input_digest_value",
                    }
                )
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    finite_failures.append(
                        {
                            "row_index": int(row_index),
                            "path": f"features.{path}" if path else "features",
                            "reason": "non_finite_numeric_value",
                        }
                    )
        if "label" in features or "labels" in features:
            label_feature_failures.append(
                {
                    "row_index": int(row_index),
                    "path": "features.label",
                    "reason": "label_present_in_trainable_input_features",
                }
            )
    all_failures = failures + finite_failures + label_feature_failures
    return {
        "policy": "m3_carrion_sequence_context_trainable_input_leakage_scan_v1",
        "passed": not all_failures,
        "row_count": len(rows),
        "forbidden_tokens": list(FORBIDDEN_TRAINABLE_INPUT_TOKENS),
        "forbidden_failure_count": len(failures),
        "finite_failure_count": len(finite_failures),
        "label_feature_failure_count": len(label_feature_failures),
        "failures": all_failures[:32],
    }


def _finalize_report(
    *,
    archive_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    train_eval_report: Mapping[str, object],
    autopsy_report: Mapping[str, object],
    source_validation: Mapping[str, object],
    harmful_sources: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    generation_status: Mapping[str, object],
    generation_evidence: Mapping[str, object],
    target_seeds: Sequence[int],
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    ticks: int,
    history_window: int,
    min_prior_public_steps: int,
    min_safe_comparator_count: int,
    branch_result_chunk_dir: str | Path | None,
    resume_branch_results: bool,
    max_wall_seconds: float | None,
    shard_id: str | None,
    input_paths: Mapping[str, str | Path] | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    resolved_branch_results = sorted(
        [dict(result) for result in branch_results],
        key=_v151_branch_result_sort_key,
    )
    rows = _dataset_rows_from_branch_results(resolved_branch_results)
    leakage_scan = carrion_sequence_context_trainable_leakage_scan(rows)
    replay = _source_replay_verification_report(rows)
    coverage = _coverage_report(
        rows=rows,
        branch_results=resolved_branch_results,
        harmful_sources=harmful_sources,
        target_seeds=target_seeds,
    )
    separation = _sequence_separation_report(rows=rows, harmful_sources=harmful_sources)
    action_distribution = _action_distribution(rows)
    source_integrity = _source_integrity(
        source_validation=source_validation,
        leakage_scan=leakage_scan,
        replay=replay,
        coverage=coverage,
        generation_status=generation_status,
        strict_harmful_source_coverage=(
            seed_include is None
            and branch_index_include is None
            and shard_id is None
            and generation_status.get("policy")
            != "m3_carrion_sequence_context_merged_generation_status_v1"
        )
        or generation_status.get("policy")
        == "m3_carrion_sequence_context_merged_generation_status_v1",
    )
    classification = _classification(
        source_integrity=source_integrity,
        generation_status=generation_status,
        coverage=coverage,
        separation=separation,
        min_safe_comparator_count=int(min_safe_comparator_count),
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
        "target_fixture": "carrion_only",
        "target_harmful_support_sources": [
            {
                "label_action": source.get("label_action"),
                "source_seed": source.get("source_seed"),
                "source_branch_reason": source.get("source_branch_reason"),
            }
            for source in harmful_sources
        ],
        "trainable_input_fields": [
            "public observation_input",
            "public action_mask",
            "prior finalized public sequence context",
        ],
        "trainable_target_fields": ["action label"],
        "excluded_trainable_input_fields": [
            "seed",
            "fixture",
            "branch id",
            "tick identity",
            "agent id",
            "path",
            "digest",
            "private world state",
            "future outcomes",
            "labels",
        ],
        "runtime_path": "none_added_diagnostics_report_only",
    }
    dataset_digest = stable_payload_digest(rows)
    branch_evidence_digest = stable_payload_digest(resolved_branch_results)
    inputs = {
        **_input_path_payload(input_paths),
        "shard_id": shard_id,
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in target_seeds],
        "seed_include": None if seed_include is None else [int(seed) for seed in seed_include],
        "branch_index_include": (
            None
            if branch_index_include is None
            else [int(index) for index in branch_index_include]
        ),
        "ticks": int(ticks),
        "history_window": int(history_window),
        "min_prior_public_steps": int(min_prior_public_steps),
        "min_safe_comparator_count": int(min_safe_comparator_count),
        "branch_result_chunk_dir": (
            None if branch_result_chunk_dir is None else str(branch_result_chunk_dir)
        ),
        "resume_branch_results": bool(resume_branch_results),
        "max_wall_seconds": None if max_wall_seconds is None else float(max_wall_seconds),
        "v148_dataset_digest": source_validation.get("v148_dataset_digest"),
        "v148_branch_evidence_digest": source_validation.get(
            "v148_branch_evidence_digest"
        ),
        "v149_train_eval_report_digest": source_validation.get(
            "v149_train_eval_report_digest"
        ),
        "v150_autopsy_report_digest": source_validation.get(
            "v150_autopsy_report_digest"
        ),
    }
    report = {
        "schema_version": M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION,
        "policy": M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_POLICY,
        "diagnostics_only": True,
        "contract": contract,
        "inputs": inputs,
        "generation_status": dict(generation_status),
        "generation_evidence": dict(generation_evidence),
        "source_validation": dict(source_validation),
        "source_integrity": source_integrity,
        "target_harmful_support_sources": [dict(source) for source in harmful_sources],
        "coverage": coverage,
        "sequence_context_separation": separation,
        "action_distribution": action_distribution,
        "replay_verification": replay,
        "leakage_scan": leakage_scan,
        "dataset": {
            "feature_policy": M3_CARRION_SEQUENCE_CONTEXT_FEATURE_POLICY,
            "row_count": len(rows),
            "dataset_digest": dataset_digest,
            "trainable_input_surface": (
                "public observation_input/action_mask plus prior finalized "
                "public sequence context; action label is target only"
            ),
            "harmful_source_row_count": coverage.get("harmful_source_row_count"),
            "safe_comparator_row_count": coverage.get("safe_comparator_row_count"),
        },
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
        "branch_evidence_digest": branch_evidence_digest,
        "non_promoted": True,
    }
    exact_payload = {
        "schema_version": report["schema_version"],
        "policy": report["policy"],
        "contract": contract,
        "inputs": {
            "target_carrion_seeds": inputs["target_carrion_seeds"],
            "ticks": inputs["ticks"],
            "history_window": inputs["history_window"],
            "min_prior_public_steps": inputs["min_prior_public_steps"],
            "min_safe_comparator_count": inputs["min_safe_comparator_count"],
            "v148_dataset_digest": inputs["v148_dataset_digest"],
            "v148_branch_evidence_digest": inputs["v148_branch_evidence_digest"],
            "v149_train_eval_report_digest": inputs["v149_train_eval_report_digest"],
            "v150_autopsy_report_digest": inputs["v150_autopsy_report_digest"],
        },
        "branch_evidence_digest": branch_evidence_digest,
        "dataset_digest": dataset_digest,
        "source_integrity": source_integrity,
        "coverage": coverage,
        "sequence_context_separation": separation,
        "classification": report["classification"],
    }
    report["exact_digest"] = stable_payload_digest(exact_payload)
    report["provenance"] = {
        "contract_digest": stable_payload_digest(contract),
        "exact_digest_payload_policy": (
            "stable_payload_digest_of_v151_contract_inputs_evidence_and_outcomes_v1"
        ),
        "exact_digest": report["exact_digest"],
    }
    return report, rows


def _sequence_branch_result(
    *,
    archive_result: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    harmful_sources: Sequence[Mapping[str, object]],
    prior_context: Sequence[Mapping[str, object]],
    history_window: int,
    min_prior_public_steps: int,
) -> dict[str, object]:
    branch_id = str(archive_result.get("branch_id", ""))
    prior = _finalized_prior_context(prior_context, history_window=int(history_window))
    source_rows = [
        _sequence_source_row(
            source_dataset_row_index=_int(_mapping(row.get("metadata")).get("row_index"), default=index),
            archive_result=archive_result,
            dataset_row=row,
            harmful_sources=harmful_sources,
            prior_context=prior,
            source_local_row_index=index,
        )
        for index, row in enumerate(sorted(dataset_rows, key=_dataset_row_sort_key))
    ]
    context_payload = {
        "feature_policy": M3_CARRION_SEQUENCE_CONTEXT_FEATURE_POLICY,
        "policy_visible_public_inputs_only": True,
        "prior_inputs": ["prior_finalized_public_sequence_context"],
        "history_window": int(history_window),
        "min_prior_public_steps": int(min_prior_public_steps),
        "history_step_count": len(prior),
        "hydration_reproduction_markers": _prior_context_markers(prior),
        "prior_context_digest": stable_payload_digest(prior),
    }
    return {
        "schema_version": "m3_carrion_sequence_context_branch_result_v1",
        "policy": M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_POLICY,
        "branch_id": branch_id,
        "fixture": archive_result.get("fixture"),
        "seed": archive_result.get("seed"),
        "ticks": archive_result.get("ticks"),
        "branch_index": archive_result.get("branch_index"),
        "branch_tick": archive_result.get("branch_tick"),
        "record_index": archive_result.get("record_index"),
        "agent_id": archive_result.get("agent_id"),
        "branch_reason": _branch_reason(archive_result),
        "carrion_archive_context": archive_result.get("carrion_archive_context"),
        "public_sequence_context": context_payload,
        "source_rows": source_rows,
        "source_row_count": len(source_rows),
        "harmful_source_row_count": sum(
            1 for row in source_rows if _mapping(row.get("metadata")).get("harmful_source") is True
        ),
        "safe_comparator_row_count": sum(
            1 for row in source_rows if _mapping(row.get("metadata")).get("harmful_source") is not True
        ),
        "diagnostics_only": True,
    }


def _sequence_source_row(
    *,
    source_dataset_row_index: int,
    archive_result: Mapping[str, object],
    dataset_row: Mapping[str, object],
    harmful_sources: Sequence[Mapping[str, object]],
    prior_context: Sequence[Mapping[str, object]],
    source_local_row_index: int,
) -> dict[str, object]:
    source_trainable = _mapping(dataset_row.get("trainable"))
    source_features = _mapping(source_trainable.get("features"))
    source_label = _mapping(source_trainable.get("label"))
    action = str(source_label.get("action", ""))
    action_mask = _bool_action_mask(source_features.get("action_mask"))
    harmful = _harmful_source_for(
        branch_id=str(archive_result.get("branch_id", "")),
        action=action,
        harmful_sources=harmful_sources,
    )
    trainable = {
        "feature_policy": M3_CARRION_SEQUENCE_CONTEXT_FEATURE_POLICY,
        "features": {
            "observation_input": source_features.get("observation_input"),
            "action_mask": action_mask,
            "prior_public_sequence_context": [dict(item) for item in prior_context],
        },
        "label": {
            "action": action,
            "label_policy": (
                "v148_safety_vetted_carrion_archive_action_label_reused_for_v151_diagnostics_v1"
            ),
        },
    }
    metadata = {
        "source_local_row_index": int(source_local_row_index),
        "source_dataset_row_index": int(source_dataset_row_index),
        "source_branch_result_index": _mapping(dataset_row.get("metadata")).get(
            "source_branch_result_index"
        ),
        "seed": archive_result.get("seed"),
        "fixture": archive_result.get("fixture"),
        "branch_id": archive_result.get("branch_id"),
        "branch_index": archive_result.get("branch_index"),
        "branch_tick": archive_result.get("branch_tick"),
        "record_index": archive_result.get("record_index"),
        "agent_id": archive_result.get("agent_id"),
        "branch_reason": _branch_reason(archive_result),
        "label_action": action,
        "harmful_source": harmful is not None,
        "harmful_source_evidence": None if harmful is None else dict(harmful),
        "prior_public_sequence_context_digest": stable_payload_digest(
            list(prior_context)
        ),
        "hydration_reproduction_markers": _prior_context_markers(prior_context),
        "source_safety_vet": _mapping(_mapping(dataset_row.get("metadata")).get("safety_vet")),
        "source_outcome_evidence": _mapping(
            _mapping(dataset_row.get("metadata")).get("outcome_evidence")
        ),
    }
    row = {
        "schema_version": "m3_carrion_sequence_context_archive_dataset_row_v1",
        "trainable": trainable,
        "metadata": metadata,
    }
    row["stable_source_row_digest"] = stable_payload_digest(
        {
            "trainable": trainable,
            "metadata_without_row_index": metadata,
        }
    )
    return row


def _dataset_rows_from_branch_results(
    branch_results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = [
        dict(row)
        for result in sorted(branch_results, key=_v151_branch_result_sort_key)
        for row in sorted(
            _list_of_mappings(result.get("source_rows")),
            key=_v151_source_row_sort_key,
        )
    ]
    normalized = []
    for row_index, row in enumerate(rows):
        resolved = dict(row)
        metadata = dict(_mapping(resolved.get("metadata")))
        metadata["row_index"] = int(row_index)
        resolved["metadata"] = metadata
        normalized.append(resolved)
    return normalized


def _materialize_seed_prior_public_contexts(
    *,
    seed: int,
    ticks: int,
    history_window: int,
    needed_record_indexes: Mapping[int, str],
    contexts: dict[str, list[dict[str, object]]],
    started_at: float,
    max_wall_seconds: float | None,
) -> dict[str, object]:
    world = evaluate_cli._fixture_world(
        fixture_name="carrion_only",
        seed=int(seed),
        ticks=int(ticks),
        policy=MindV3EvolutionPolicy(seed=int(seed)),
    )
    _configure_branch_manual_summary_run(world)
    history_by_agent: dict[int, list[dict[str, object]]] = {}
    record_index = 0
    ticks_executed = 0
    stop_reason = None
    materialized = 0
    for tick in range(int(ticks)):
        if _wall_budget_exhausted(started_at, max_wall_seconds):
            stop_reason = "max_wall_seconds_elapsed_during_seed_context"
            break
        world.tick = tick
        world._run_tick()
        ticks_executed = tick + 1
        for record in list(world.tick_trajectory_records):
            branch_id = needed_record_indexes.get(record_index)
            if branch_id is not None:
                agent_id = _int(record.get("agent_id"), default=-1)
                prior = history_by_agent.get(agent_id, [])
                contexts[branch_id] = _finalized_prior_context(
                    prior,
                    history_window=int(history_window),
                )
                materialized += 1
            _update_public_history_from_record(
                history_by_agent=history_by_agent,
                record=record,
                history_window=int(history_window),
            )
            record_index += 1
        if materialized >= len(needed_record_indexes):
            break
        if not world.alive_agents():
            break
    return {
        "policy": "m3_carrion_sequence_context_seed_prior_public_context_v1",
        "fixture": "carrion_only",
        "seed": int(seed),
        "ticks_requested": int(ticks),
        "ticks_executed": int(ticks_executed),
        "needed_context_count": len(needed_record_indexes),
        "materialized_context_count": int(materialized),
        "history_window": int(history_window),
        "stop_reason": stop_reason,
        "passed": stop_reason is None and materialized == len(needed_record_indexes),
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
    history = history_by_agent.setdefault(agent_id, [])
    history.append(_public_transition_summary(record))
    if len(history) > int(history_window):
        del history[0 : len(history) - int(history_window)]


def _public_transition_summary(record: Mapping[str, object]) -> dict[str, object]:
    outcome = _mapping(record.get("outcome"))
    feeding = _mapping(outcome.get("feeding"))
    drinking = _mapping(outcome.get("drinking"))
    movement = _mapping(outcome.get("movement"))
    passive = _mapping(outcome.get("passive"))
    food_kind = str(feeding.get("food_source", ""))
    gain = _optional_float(outcome.get("resource_gain"))
    return {
        "requested_action": _optional_action(record.get("requested_action")),
        "resolved_action": _optional_action(record.get("resolved_action")),
        "action_valid": bool(
            record.get("action_valid", outcome.get("observation_action_valid", False))
        ),
        "resolution_action_valid": bool(
            record.get(
                "resolution_action_valid",
                outcome.get("resolution_action_valid", False),
            )
        ),
        "moved": bool(movement.get("moved", record.get("moved", False))),
        "drank": bool(drinking.get("drank", False)),
        "ate": bool(feeding.get("ate", False)),
        "carcass_food": food_kind == "carcass",
        "fresh_kill_food": food_kind == "fresh_kill",
        "animal_food": food_kind in {"carcass", "fresh_kill"},
        "plant_food": food_kind == "plant",
        "gain_present": gain is not None and gain > 0.0,
        "reproduced": bool(outcome.get("reproduced", False)),
        "reproduction_ready_after": bool(
            outcome.get("reproduction_ready_after", False)
        ),
        "died_after_action": bool(passive.get("died_after_action", False)),
    }


def _finalized_prior_context(
    prior_context: Sequence[Mapping[str, object]],
    *,
    history_window: int,
) -> list[dict[str, object]]:
    prior = [dict(item) for item in prior_context[-int(history_window):]]
    size = len(prior)
    resolved = []
    for index, item in enumerate(prior):
        resolved.append(
            {
                "relative_step": int(index - size),
                **{
                    key: value
                    for key, value in sorted(item.items())
                    if key != "relative_step"
                },
            }
        )
    return resolved


def _coverage_report(
    *,
    rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    harmful_sources: Sequence[Mapping[str, object]],
    target_seeds: Sequence[int],
) -> dict[str, object]:
    harmful_rows = [row for row in rows if _is_harmful_row(row)]
    safe_rows = [row for row in rows if not _is_harmful_row(row)]
    observed_harmful_keys = {
        _harmful_key_from_row(row)
        for row in harmful_rows
        if _harmful_key_from_row(row) is not None
    }
    target_harmful_keys = {_harmful_key(source) for source in harmful_sources}
    missing_harmful = sorted(target_harmful_keys - observed_harmful_keys)
    rows_by_seed = Counter(_int(_mapping(row.get("metadata")).get("seed")) for row in rows)
    branch_by_seed = Counter(_int(result.get("seed")) for result in branch_results)
    per_seed = []
    for seed in target_seeds:
        resolved = int(seed)
        seed_rows = [
            row
            for row in rows
            if _int(_mapping(row.get("metadata")).get("seed"), default=-1) == resolved
        ]
        per_seed.append(
            {
                "fixture": "carrion_only",
                "seed": resolved,
                "branch_result_count": int(branch_by_seed.get(resolved, 0)),
                "sequence_row_count": int(rows_by_seed.get(resolved, 0)),
                "harmful_source_row_count": sum(1 for row in seed_rows if _is_harmful_row(row)),
                "safe_comparator_row_count": sum(
                    1 for row in seed_rows if not _is_harmful_row(row)
                ),
                "action_counts": dict(
                    sorted(
                        Counter(
                            str(_mapping(_mapping(row.get("trainable")).get("label")).get("action", ""))
                            for row in seed_rows
                        ).items()
                    )
                ),
            }
        )
    return {
        "policy": "m3_carrion_sequence_context_coverage_v1",
        "target_harmful_source_count": len(harmful_sources),
        "observed_harmful_source_count": len(observed_harmful_keys),
        "missing_harmful_sources": missing_harmful,
        "branch_result_count": len(branch_results),
        "sequence_row_count": len(rows),
        "harmful_source_row_count": len(harmful_rows),
        "safe_comparator_row_count": len(safe_rows),
        "target_seeds": [int(seed) for seed in target_seeds],
        "per_seed_support": per_seed,
        "missing_sequence_row_seeds": [
            int(seed) for seed in target_seeds if rows_by_seed.get(int(seed), 0) <= 0
        ],
        "missing_branch_result_seeds": [
            int(seed) for seed in target_seeds if branch_by_seed.get(int(seed), 0) <= 0
        ],
    }


def _sequence_separation_report(
    *,
    rows: Sequence[Mapping[str, object]],
    harmful_sources: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    safe_rows = [row for row in rows if not _is_harmful_row(row)]
    harmful_rows = [row for row in rows if _is_harmful_row(row)]
    safe_by_action: dict[str, list[Mapping[str, object]]] = {}
    for row in safe_rows:
        action = _row_action(row)
        safe_by_action.setdefault(action, []).append(row)
    harmful_reports = []
    separated_count = 0
    comparator_count_total = 0
    exact_one_step_comparator_count = 0
    for row in sorted(harmful_rows, key=_dataset_output_row_sort_key):
        action = _row_action(row)
        safe_matches = safe_by_action.get(action, [])
        sequence_key = _prior_sequence_action_key(row)
        same_sequence = [
            candidate
            for candidate in safe_matches
            if _prior_sequence_action_key(candidate) == sequence_key
        ]
        one_step_key = _one_step_action_key(row)
        exact_one_step_matches = [
            candidate for candidate in safe_matches if _one_step_action_key(candidate) == one_step_key
        ]
        separated = bool(safe_matches) and not same_sequence
        if separated:
            separated_count += 1
        comparator_count_total += len(safe_matches)
        exact_one_step_comparator_count += len(exact_one_step_matches)
        metadata = _mapping(row.get("metadata"))
        harmful_reports.append(
            {
                "source": _harmful_key_from_row(row),
                "label_action": action,
                "source_seed": metadata.get("seed"),
                "source_branch_reason": metadata.get("branch_reason"),
                "source_branch_id": metadata.get("branch_id"),
                "prior_step_count": _prior_step_count(row),
                "safe_same_action_comparator_count": len(safe_matches),
                "safe_same_sequence_comparator_count": len(same_sequence),
                "exact_one_step_comparator_count": len(exact_one_step_matches),
                "safe_nonempty_prior_comparator_count": sum(
                    1 for candidate in safe_matches if _prior_step_count(candidate) > 0
                ),
                "sequence_distinguishes_harmful_from_safe": separated,
                "harmful_prior_markers": metadata.get(
                    "hydration_reproduction_markers"
                ),
            }
        )
    action_reports = []
    for action in sorted({str(source.get("label_action", "")) for source in harmful_sources}):
        action_harmful = [row for row in harmful_rows if _row_action(row) == action]
        action_safe = safe_by_action.get(action, [])
        action_reports.append(
            {
                "label_action": action,
                "harmful_source_row_count": len(action_harmful),
                "safe_comparator_row_count": len(action_safe),
                "harmful_zero_prior_count": sum(
                    1 for row in action_harmful if _prior_step_count(row) == 0
                ),
                "safe_nonempty_prior_count": sum(
                    1 for row in action_safe if _prior_step_count(row) > 0
                ),
                "safe_hydration_or_reproduction_context_count": sum(
                    1
                    for row in action_safe
                    if _markers_have_hydration_or_reproduction(
                        _mapping(_mapping(row.get("metadata")).get("hydration_reproduction_markers"))
                    )
                ),
            }
        )
    return {
        "policy": "m3_carrion_sequence_context_harmful_safe_separation_v1",
        "harmful_source_row_count": len(harmful_rows),
        "safe_comparator_row_count": len(safe_rows),
        "safe_same_action_comparator_count": int(comparator_count_total),
        "exact_one_step_comparator_count": int(exact_one_step_comparator_count),
        "sequence_separated_harmful_source_count": int(separated_count),
        "all_harmful_sources_sequence_separated": (
            bool(harmful_rows) and separated_count == len(harmful_rows)
        ),
        "action_support": action_reports,
        "harmful_source_separation": harmful_reports,
        "interpretation": (
            "separation compares the harmful v150 support labels against v148 "
            "safe same-action carrion support rows after adding only prior "
            "finalized public sequence context"
        ),
    }


def _source_replay_verification_report(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    checked = 0
    verified = 0
    missing = 0
    failed = 0
    for row in rows:
        evidence = _mapping(
            _mapping(_mapping(row.get("metadata")).get("source_outcome_evidence")).get(
                "replay_verification"
            )
        )
        checked += 1
        if not evidence:
            missing += 1
        elif evidence.get("verified") is True:
            verified += 1
        else:
            failed += 1
    return {
        "policy": "m3_carrion_sequence_context_source_replay_verification_v1",
        "complete": checked > 0 and verified == checked and missing == 0 and failed == 0,
        "row_count": checked,
        "verified_count": verified,
        "missing_count": missing,
        "failed_count": failed,
    }


def _action_distribution(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    harmful = Counter(_row_action(row) for row in rows if _is_harmful_row(row))
    safe = Counter(_row_action(row) for row in rows if not _is_harmful_row(row))
    all_counts = Counter(_row_action(row) for row in rows)
    dominant_action = None
    dominant_count = 0
    if safe:
        dominant_action, dominant_count = max(
            sorted(safe.items()),
            key=lambda item: item[1],
        )
    return {
        "policy": "m3_carrion_sequence_context_action_distribution_v1",
        "row_action_counts": dict(sorted(all_counts.items())),
        "harmful_source_action_counts": dict(sorted(harmful.items())),
        "safe_comparator_action_counts": dict(sorted(safe.items())),
        "dominant_safe_comparator_action": dominant_action,
        "dominant_safe_comparator_action_count": int(dominant_count),
        "dominant_safe_comparator_action_share": _safe_rate(
            dominant_count,
            sum(safe.values()),
        ),
    }


def _source_integrity(
    *,
    source_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    replay: Mapping[str, object],
    coverage: Mapping[str, object],
    generation_status: Mapping[str, object],
    strict_harmful_source_coverage: bool,
) -> dict[str, object]:
    failures: list[str] = []
    if source_validation.get("passed") is not True:
        failures.append("input_validation_failed")
    if leakage_scan.get("passed") is not True:
        failures.append("trainable_input_leakage_detected")
    if replay.get("complete") is not True:
        failures.append("source_replay_verification_incomplete")
    if _status_partial(generation_status):
        failures.append("partial_sequence_context_evidence")
    if int(coverage.get("sequence_row_count", 0)) <= 0:
        failures.append("no_sequence_context_rows")
    if strict_harmful_source_coverage and _list(coverage.get("missing_harmful_sources")):
        failures.append("target_harmful_support_sources_missing")
    return {
        "policy": "m3_carrion_sequence_context_source_integrity_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "input_validation_passed": source_validation.get("passed") is True,
        "leakage_scan_passed": leakage_scan.get("passed") is True,
        "source_replay_verification_complete": replay.get("complete") is True,
        "generation_state": generation_status.get("state"),
        "sequence_row_count": coverage.get("sequence_row_count"),
        "harmful_source_row_count": coverage.get("harmful_source_row_count"),
        "safe_comparator_row_count": coverage.get("safe_comparator_row_count"),
        "strict_harmful_source_coverage": bool(strict_harmful_source_coverage),
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    generation_status: Mapping[str, object],
    coverage: Mapping[str, object],
    separation: Mapping[str, object],
    min_safe_comparator_count: int,
) -> str:
    if _status_partial(generation_status):
        return "m3_carrion_sequence_context_archive_partial_no_training"
    if source_integrity.get("passed") is not True:
        return "m3_carrion_sequence_context_archive_source_integrity_failed_no_training"
    if _list(coverage.get("missing_harmful_sources")):
        return "m3_carrion_sequence_context_archive_harmful_source_missing_no_training"
    if _int(coverage.get("safe_comparator_row_count")) < int(min_safe_comparator_count):
        return "m3_carrion_sequence_context_archive_support_insufficient_no_training"
    if separation.get("all_harmful_sources_sequence_separated") is True:
        return "m3_carrion_sequence_context_archive_separates_harmful_sources_no_training"
    if _int(separation.get("sequence_separated_harmful_source_count")) > 0:
        return "m3_carrion_sequence_context_archive_partial_sequence_separation_no_training"
    return "m3_carrion_sequence_context_archive_no_sequence_separation_no_training"


def _selected_dataset_rows_by_branch(
    *,
    dataset_rows: Sequence[Mapping[str, object]],
    archive_branch_results: Sequence[Mapping[str, object]],
    harmful_sources: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    selected_branch_ids = {str(result.get("branch_id", "")) for result in archive_branch_results}
    harmful_actions = {str(source.get("label_action", "")) for source in harmful_sources}
    rows: dict[str, list[Mapping[str, object]]] = {}
    for row in dataset_rows:
        metadata = _mapping(row.get("metadata"))
        branch_id = str(metadata.get("branch_id", ""))
        if branch_id not in selected_branch_ids:
            continue
        action = str(_mapping(_mapping(row.get("trainable")).get("label")).get("action", ""))
        if action not in harmful_actions:
            continue
        rows.setdefault(branch_id, []).append(row)
    return rows


def _harmful_sources_from_autopsy(
    autopsy_report: Mapping[str, object],
    *,
    fail_closed: bool = True,
) -> list[dict[str, object]]:
    observed = _list_of_mappings(autopsy_report.get("support_label_action_failure_map"))
    expected = {
        (
            str(source.get("label_action", "")),
            _int(source.get("source_seed"), default=-1),
            str(source.get("source_branch_reason", "")),
            str(source.get("source_branch_id", "")),
        )
        for source in DEFAULT_HARMFUL_SUPPORT_SOURCES
    }
    resolved = []
    for item in observed:
        key = (
            str(item.get("label_action", "")),
            _int(item.get("source_seed"), default=-1),
            str(item.get("source_branch_reason", "")),
            str(item.get("source_branch_id", "")),
        )
        if key in expected:
            resolved.append(dict(item))
    resolved = sorted(resolved, key=_harmful_source_sort_key)
    if fail_closed and len(resolved) != len(DEFAULT_HARMFUL_SUPPORT_SOURCES):
        raise CarrionSequenceContextArchiveError(
            "v150 autopsy missing required harmful support sources"
        )
    return resolved


def _harmful_source_for(
    *,
    branch_id: str,
    action: str,
    harmful_sources: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    for source in harmful_sources:
        if (
            str(source.get("source_branch_id", "")) == branch_id
            and str(source.get("label_action", "")) == action
        ):
            return source
    return None


def _harmful_key(source: Mapping[str, object]) -> str:
    return (
        f"{source.get('source_seed')}:{source.get('source_branch_reason')}:"
        f"{source.get('label_action')}:{source.get('source_branch_id')}"
    )


def _harmful_key_from_row(row: Mapping[str, object]) -> str | None:
    metadata = _mapping(row.get("metadata"))
    evidence = _mapping(metadata.get("harmful_source_evidence"))
    if evidence:
        return _harmful_key(evidence)
    return None


def _prior_context_markers(
    prior_context: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "prior_step_count": len(prior_context),
        "drank_count": sum(1 for item in prior_context if item.get("drank") is True),
        "ate_count": sum(1 for item in prior_context if item.get("ate") is True),
        "animal_food_count": sum(
            1 for item in prior_context if item.get("animal_food") is True
        ),
        "reproduced_count": sum(
            1 for item in prior_context if item.get("reproduced") is True
        ),
        "reproduction_ready_after_count": sum(
            1
            for item in prior_context
            if item.get("reproduction_ready_after") is True
        ),
        "movement_count": sum(1 for item in prior_context if item.get("moved") is True),
    }


def _markers_have_hydration_or_reproduction(markers: Mapping[str, object]) -> bool:
    return (
        _int(markers.get("drank_count")) > 0
        or _int(markers.get("reproduced_count")) > 0
        or _int(markers.get("reproduction_ready_after_count")) > 0
    )


def _precomputed_generation_status(
    *,
    branch_results: Sequence[Mapping[str, object]],
    target_seeds: Sequence[int],
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    ticks: int,
    history_window: int,
    min_prior_public_steps: int,
    shard_id: str | None,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_sequence_context_precomputed_generation_status_v1",
        "state": "complete",
        "partial": False,
        "stop_reason": None,
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in target_seeds],
        "seed_include": None if seed_include is None else [int(seed) for seed in seed_include],
        "branch_index_include": (
            None
            if branch_index_include is None
            else [int(index) for index in branch_index_include]
        ),
        "ticks": int(ticks),
        "history_window": int(history_window),
        "min_prior_public_steps": int(min_prior_public_steps),
        "branch_result_count": len(branch_results),
        "shard_id": shard_id,
    }


def _v151_branch_result_matches(
    cached: Mapping[str, object],
    *,
    archive_result: Mapping[str, object],
    source_rows: Sequence[Mapping[str, object]],
    prior_context: Sequence[Mapping[str, object]],
    history_window: int,
    min_prior_public_steps: int,
) -> bool:
    if str(cached.get("branch_id", "")) != str(archive_result.get("branch_id", "")):
        return False
    context = _mapping(cached.get("public_sequence_context"))
    if _int(context.get("history_window"), default=-1) != int(history_window):
        return False
    if _int(context.get("min_prior_public_steps"), default=-1) != int(min_prior_public_steps):
        return False
    if context.get("prior_context_digest") != stable_payload_digest(
        _finalized_prior_context(prior_context, history_window=int(history_window))
    ):
        return False
    cached_rows = _list_of_mappings(cached.get("source_rows"))
    expected_actions = sorted(
        str(_mapping(_mapping(row.get("trainable")).get("label")).get("action", ""))
        for row in source_rows
    )
    cached_actions = sorted(
        str(_mapping(_mapping(row.get("trainable")).get("label")).get("action", ""))
        for row in cached_rows
    )
    return expected_actions == cached_actions


def _validate_shard_contract(report: Mapping[str, object], *, shard_index: int) -> None:
    if report.get("schema_version") != M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION:
        raise CarrionSequenceContextArchiveError(f"shard {shard_index} schema mismatch")
    if report.get("policy") != M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_POLICY:
        raise CarrionSequenceContextArchiveError(f"shard {shard_index} policy mismatch")
    if report.get("diagnostics_only") is not True:
        raise CarrionSequenceContextArchiveError(
            f"shard {shard_index} missing diagnostics-only contract"
        )
    failures: list[str] = []
    _validate_authorization_flags(
        name=f"shard_{shard_index}",
        payload=report,
        failures=failures,
        missing_is_drift=True,
    )
    if report.get("default_runtime_behavior_changed") is not False:
        failures.append(f"shard_{shard_index}_default_runtime_behavior_changed_not_false")
    if failures:
        raise CarrionSequenceContextArchiveError(
            f"shard {shard_index} authorization drift: {sorted(failures)}"
        )


def _merge_identity(report: Mapping[str, object]) -> dict[str, object]:
    inputs = _mapping(report.get("inputs"))
    return {
        "schema_version": report.get("schema_version"),
        "policy": report.get("policy"),
        "target_fixture": inputs.get("target_fixture"),
        "target_carrion_seeds": _list(inputs.get("target_carrion_seeds")),
        "ticks": inputs.get("ticks"),
        "history_window": inputs.get("history_window"),
        "min_prior_public_steps": inputs.get("min_prior_public_steps"),
        "min_safe_comparator_count": inputs.get("min_safe_comparator_count"),
        "v148_dataset_digest": inputs.get("v148_dataset_digest"),
        "v148_branch_evidence_digest": inputs.get("v148_branch_evidence_digest"),
        "v149_train_eval_report_digest": inputs.get(
            "v149_train_eval_report_digest"
        ),
        "v150_autopsy_report_digest": inputs.get("v150_autopsy_report_digest"),
    }


def _add_merged_branch_result(
    *,
    merged_by_branch_id: dict[str, dict[str, object]],
    digests_by_branch_id: dict[str, str],
    result: Mapping[str, object],
    source: str,
) -> None:
    branch_id = str(result.get("branch_id", ""))
    if not branch_id:
        raise CarrionSequenceContextArchiveError(
            f"merged v151 branch result missing branch_id: {source}"
        )
    digest = stable_payload_digest(result)
    prior_digest = digests_by_branch_id.get(branch_id)
    if prior_digest is not None and prior_digest != digest:
        raise CarrionSequenceContextArchiveError(
            f"duplicate branch_id with different digest: {branch_id}"
        )
    merged_by_branch_id[branch_id] = dict(result)
    digests_by_branch_id[branch_id] = digest


def _archive_result_selected(
    result: Mapping[str, object],
    *,
    target_seed_set: set[int],
    seed_filter: set[int] | None,
    branch_filter: set[int] | None,
) -> bool:
    if str(result.get("fixture", "")) != "carrion_only":
        return False
    seed = _int(result.get("seed"), default=-1)
    if seed not in target_seed_set:
        return False
    if seed_filter is not None and seed not in seed_filter:
        return False
    branch_index = _int(result.get("branch_index"), default=-1)
    if branch_filter is not None and branch_index not in branch_filter:
        return False
    return True


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
        "default_runtime_behavior_changed",
        "runtime_action_selection_changed",
    ):
        value = payload.get(key)
        if value is True:
            failures.append(f"{name}_{key}_true")
        if missing_is_drift and value is None:
            failures.append(f"{name}_{key}_missing")
    if payload.get("diagnostics_only") is False:
        failures.append(f"{name}_diagnostics_only_false")


def _input_path_payload(input_paths: Mapping[str, str | Path] | None) -> dict[str, str]:
    defaults = {
        "v148_carrion_archive_report": DEFAULT_V148_CARRION_ARCHIVE_REPORT_PATH,
        "v148_carrion_archive_dataset": DEFAULT_V148_CARRION_ARCHIVE_DATASET_PATH,
        "v149_carrion_train_eval_report": DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH,
        "v150_carrion_override_autopsy": DEFAULT_V150_CARRION_OVERRIDE_AUTOPSY_PATH,
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


def _archive_branch_result_sort_key(result: Mapping[str, object]) -> tuple[int, int, int, str]:
    return (
        _int(result.get("seed")),
        _int(result.get("branch_index")),
        _int(result.get("branch_tick")),
        str(result.get("branch_id", "")),
    )


def _v151_branch_result_sort_key(result: Mapping[str, object]) -> tuple[int, int, int, str]:
    return _archive_branch_result_sort_key(result)


def _dataset_row_sort_key(row: Mapping[str, object]) -> tuple[int, int, str]:
    metadata = _mapping(row.get("metadata"))
    action = str(_mapping(_mapping(row.get("trainable")).get("label")).get("action", ""))
    return (
        _action_order(action),
        _int(metadata.get("row_index")),
        action,
    )


def _v151_source_row_sort_key(row: Mapping[str, object]) -> tuple[int, int, str]:
    metadata = _mapping(row.get("metadata"))
    action = _row_action(row)
    return (
        _action_order(action),
        _int(metadata.get("source_dataset_row_index")),
        str(row.get("stable_source_row_digest", "")),
    )


def _dataset_output_row_sort_key(row: Mapping[str, object]) -> tuple[int, int, int, str]:
    metadata = _mapping(row.get("metadata"))
    return (
        _int(metadata.get("seed")),
        _int(metadata.get("branch_index")),
        _action_order(_row_action(row)),
        str(metadata.get("branch_id", "")),
    )


def _harmful_source_sort_key(source: Mapping[str, object]) -> tuple[str, int, str]:
    return (
        str(source.get("label_action", "")),
        _int(source.get("source_seed"), default=-1),
        str(source.get("source_branch_id", "")),
    )


def _branch_reason(result: Mapping[str, object]) -> str:
    context = _mapping(result.get("carrion_archive_context"))
    return str(context.get("branch_reason", "unknown"))


def _is_harmful_row(row: Mapping[str, object]) -> bool:
    return _mapping(row.get("metadata")).get("harmful_source") is True


def _row_action(row: Mapping[str, object]) -> str:
    return str(_mapping(_mapping(row.get("trainable")).get("label")).get("action", ""))


def _prior_step_count(row: Mapping[str, object]) -> int:
    features = _mapping(_mapping(row.get("trainable")).get("features"))
    return len(_list_of_mappings(features.get("prior_public_sequence_context")))


def _one_step_action_key(row: Mapping[str, object]) -> str:
    trainable = _mapping(row.get("trainable"))
    features = _mapping(trainable.get("features"))
    return stable_payload_digest(
        {
            "observation_input": features.get("observation_input"),
            "action_mask": _bool_action_mask(features.get("action_mask")),
            "action": _row_action(row),
        }
    )


def _prior_sequence_action_key(row: Mapping[str, object]) -> str:
    trainable = _mapping(row.get("trainable"))
    features = _mapping(trainable.get("features"))
    return stable_payload_digest(
        {
            "prior_public_sequence_context": _list_of_mappings(
                features.get("prior_public_sequence_context")
            ),
            "action": _row_action(row),
        }
    )


def _shard_source_name(report: Mapping[str, object], *, shard_index: int) -> str:
    shard_id = _shard_id_from_report(report)
    if shard_id is not None:
        return f"report:{shard_id}"
    branch_results = _list_of_mappings(report.get("branch_results"))
    digest = stable_payload_digest(branch_results)[:12]
    return f"report:{digest or shard_index}"


def _shard_id_from_report(report: Mapping[str, object]) -> str | None:
    inputs = _mapping(report.get("inputs"))
    status = _mapping(report.get("generation_status"))
    evidence = _mapping(report.get("generation_evidence"))
    return (
        _optional_string(inputs.get("shard_id"))
        or _optional_string(status.get("shard_id"))
        or _optional_string(evidence.get("shard_id"))
    )


def _optional_index(values: Sequence[str], index: int) -> str | None:
    if 0 <= int(index) < len(values):
        return values[int(index)]
    return None


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _bool_action_mask(value: object) -> dict[str, bool]:
    payload = _mapping(value)
    return {action: bool(payload.get(action, False)) for action in ACTION_NAMES}


def _optional_action(value: object) -> str | None:
    if isinstance(value, str) and value in ACTION_NAMES:
        return value
    return None


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 6)


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


def _action_order(action: str) -> int:
    return ACTION_NAMES.index(action) if action in ACTION_NAMES else len(ACTION_NAMES)


def _slug(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("-")
    return "".join(allowed)[:180]
