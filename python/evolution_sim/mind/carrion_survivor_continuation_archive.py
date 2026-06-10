from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import math
import time
from pathlib import Path

from evolution_sim.mind import evaluation_harness as evaluate_cli
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.mind.broad_regression_branch_intervention import (
    BroadRegressionBranchPoint,
    _configure_manual_summary_run,
    _normal_mind_v3_delegate,
    _summarize_branch_world,
)
from evolution_sim.mind.candidate_campaign import (
    DEFAULT_MIN_SAFE_LABEL_COUNT,
    DEFAULT_V145_REPORT_PATH,
    _blacklisted,
    _bool_action_mask,
    _floor,
    _int,
    _list,
    _list_of_mappings,
    _mapping,
    _v145_blacklist,
    safe_archive_expansion_leakage_scan,
)
from evolution_sim.mind.carrion_archive_override_autopsy import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V150_AUTOPSY_PATH,
    M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_POLICY,
    M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_sequence_context_comparator_support_closeout import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V153_CLOSEOUT_PATH,
    M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY,
    M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_specific_archive_expansion import (
    CARRION_BRANCH_REASONS,
    DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    DEFAULT_TICKS,
    TARGET_CARRION_SEEDS,
    _carrion_branch_context,
    _filter_branch_points_by_index,
    _safe_archive_expansion_reference_from_world,
    materialize_carrion_specific_branch_points,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_archive_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_archive_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_DATASET_ROW_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_archive_dataset_row_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_BRANCH_RESULT_CHUNK_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_branch_result_chunk_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_BRANCH_RESULT_CHUNK_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_branch_result_chunk_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_POLICY = (
    "public_observation_action_mask_and_empty_public_prior_context_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v154-carrion-survivor-continuation-archive.json"
)
DEFAULT_DATASET_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v154-carrion-survivor-continuation-archive-dataset.jsonl"
)
DEFAULT_BRANCH_RESULT_CHUNK_DIR = Path(
    "output/mind/mind-v3-v154-carrion-survivor-continuation-chunks"
)
EXPECTED_V150_CLASSIFICATION = (
    "m3_carrion_archive_override_autopsy_complete_no_training"
)
EXPECTED_V153_CLASSIFICATION = (
    "m3_carrion_sequence_context_comparator_support_limited_closed_no_training"
)
MAX_DOMINANT_LABEL_ACTION_SHARE = 0.50
FORBIDDEN_TRAINABLE_FEATURE_TOKENS = (
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
    "provenance",
    "source",
)
DEFAULT_CONTINUATION_PLANS: tuple[dict[str, object], ...] = (
    {
        "continuation_index": 0,
        "name": "delegate_after_label",
        "tail_actions": (),
        "trainable_context": {"public_prior_context": []},
    },
    {
        "continuation_index": 1,
        "name": "drink_if_public_mask_allows",
        "tail_actions": ("drink",),
        "trainable_context": {"public_prior_context": []},
    },
    {
        "continuation_index": 2,
        "name": "eat_if_public_mask_allows",
        "tail_actions": ("eat",),
        "trainable_context": {"public_prior_context": []},
    },
    {
        "continuation_index": 3,
        "name": "stay_if_public_mask_allows",
        "tail_actions": ("stay",),
        "trainable_context": {"public_prior_context": []},
    },
)


class CarrionSurvivorContinuationArchiveError(ValueError):
    pass


class _ForcedShortContinuationThenMindV3Policy:
    policy_id = "mind_v3_v154_carrion_survivor_continuation_force"
    policy_version = (
        "mind_v3_v154_carrion_survivor_continuation_force_v1"
    )

    def __init__(
        self,
        *,
        target_agent_id: int,
        continuation_index: int,
        action_sequence: Sequence[str],
        delegate: object,
    ) -> None:
        self.target_agent_id = int(target_agent_id)
        self.continuation_index = int(continuation_index)
        self.action_sequence = tuple(str(action) for action in action_sequence)
        self.delegate = delegate
        self.next_action_index = 0
        self.forced_actions: list[dict[str, object]] = []
        self.skipped_actions: list[dict[str, object]] = []

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        metadata = observation.get("metadata")
        agent_id = metadata.get("agent_id") if isinstance(metadata, Mapping) else None
        if (
            agent_id is not None
            and int(agent_id) == self.target_agent_id
            and self.next_action_index < len(self.action_sequence)
        ):
            sequence_index = self.next_action_index
            action = self.action_sequence[sequence_index]
            self.next_action_index += 1
            if action in ACTION_NAMES and bool(action_mask.get(action, False)):
                self.forced_actions.append(
                    {
                        "sequence_index": int(sequence_index),
                        "action": action,
                        "public_mask_legal": True,
                    }
                )
                return ActionDecision(
                    requested_action=action,
                    source=(
                        "carrion_survivor_continuation_force:"
                        f"{self.continuation_index}:{sequence_index}:{action}"
                    ),
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    diagnostics={
                        "diagnostics_only": True,
                        "heuristic_free": True,
                        "target_agent_id": self.target_agent_id,
                        "continuation_index": self.continuation_index,
                        "sequence_index": int(sequence_index),
                        "forced_action": action,
                    },
                )
            self.skipped_actions.append(
                {
                    "sequence_index": int(sequence_index),
                    "action": action,
                    "public_mask_legal": False,
                }
            )
        decide = getattr(self.delegate, "decide", None)
        if not callable(decide):
            raise CarrionSurvivorContinuationArchiveError(
                "v154 branch delegate policy does not implement decide"
            )
        return decide(observation, action_mask)

    def observe_transition(self, record: dict[str, object]) -> dict[str, object] | None:
        observe = getattr(self.delegate, "observe_transition", None)
        if not callable(observe):
            return None
        feedback = dict(record)
        source = str(feedback.get("action_source", ""))
        if (
            _int(feedback.get("agent_id"), default=-1) == self.target_agent_id
            and source.startswith("carrion_survivor_continuation_force:")
        ):
            feedback["policy_id"] = getattr(self.delegate, "policy_id", None)
            feedback["policy_version"] = getattr(self.delegate, "policy_version", None)
        return observe(feedback)


def load_json_report(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CarrionSurvivorContinuationArchiveError(
            f"JSON report must be an object: {path}"
        )
    return payload


def write_carrion_survivor_continuation_archive_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_carrion_survivor_continuation_archive_dataset(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(dict(row), handle, sort_keys=True, allow_nan=False)
            handle.write("\n")


def build_carrion_survivor_continuation_archive_from_paths(
    *,
    autopsy_report_path: str | Path = DEFAULT_V150_AUTOPSY_PATH,
    v153_report_path: str | Path = DEFAULT_V153_CLOSEOUT_PATH,
    v145_report_path: str | Path = DEFAULT_V145_REPORT_PATH,
    allow_missing_v145_report: bool = False,
    **kwargs: object,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    return build_carrion_survivor_continuation_archive(
        autopsy_report=load_json_report(autopsy_report_path),
        v153_report=load_json_report(v153_report_path),
        v145_report=_load_optional_v145_report(
            v145_report_path,
            allow_missing=bool(allow_missing_v145_report),
        ),
        input_paths={
            "v150_autopsy_report": autopsy_report_path,
            "v153_closeout_report": v153_report_path,
            "v145_report": v145_report_path,
        },
        **kwargs,
    )


def build_carrion_survivor_continuation_archive(
    *,
    autopsy_report: Mapping[str, object],
    v153_report: Mapping[str, object],
    v145_report: Mapping[str, object] | None = None,
    continuation_branch_results: Sequence[Mapping[str, object]] | None = None,
    generation_status: Mapping[str, object] | None = None,
    generation_evidence: Mapping[str, object] | None = None,
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
    ticks: int = DEFAULT_TICKS,
    min_label_count: int = DEFAULT_MIN_SAFE_LABEL_COUNT,
    seed_include: Sequence[int] | None = None,
    branch_index_include: Sequence[int] | None = None,
    continuation_index_include: Sequence[int] | None = None,
    shard_id: str | None = None,
    branch_result_chunk_dir: str | Path | None = DEFAULT_BRANCH_RESULT_CHUNK_DIR,
    verify_replay: bool = True,
    max_wall_seconds: float | None = None,
    input_paths: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_validation = validate_carrion_survivor_continuation_inputs(
        autopsy_report=autopsy_report,
        v153_report=v153_report,
    )
    if continuation_branch_results is None:
        generated = generate_carrion_survivor_continuation_branch_results(
            seeds=seed_include or target_seeds,
            ticks=int(ticks),
            max_branch_points_per_seed=DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
            branch_index_include=branch_index_include,
            continuation_index_include=continuation_index_include,
            shard_id=shard_id,
            verify_replay=bool(verify_replay),
            branch_result_chunk_dir=branch_result_chunk_dir,
            max_wall_seconds=max_wall_seconds,
        )
        branch_results = _list_of_mappings(generated.get("branch_results"))
        status = _mapping(generated.get("generation_status"))
        evidence = {
            key: value for key, value in generated.items() if key != "branch_results"
        }
    else:
        branch_results = [dict(result) for result in continuation_branch_results]
        status = dict(
            generation_status
            or _precomputed_generation_status(
                branch_results=branch_results,
                target_seeds=seed_include or target_seeds,
                ticks=int(ticks),
                branch_index_include=branch_index_include,
                continuation_index_include=continuation_index_include,
                shard_id=shard_id,
            )
        )
        evidence = dict(
            generation_evidence
            or {
                "policy": "precomputed_m3_carrion_survivor_continuation_branch_results_v1",
                "shard_id": shard_id,
                "continuation_branch_result_count": len(branch_results),
                "continuation_branch_evidence_digest": stable_payload_digest(
                    branch_results
                ),
            }
        )
    return _finalize_report(
        autopsy_report=autopsy_report,
        v153_report=v153_report,
        v145_report=v145_report or {"route_decision": {"label_blacklist": []}},
        source_validation=source_validation,
        continuation_branch_results=branch_results,
        generation_status=status,
        generation_evidence=evidence,
        target_seeds=target_seeds,
        ticks=int(ticks),
        min_label_count=int(min_label_count),
        seed_include=seed_include,
        branch_index_include=branch_index_include,
        continuation_index_include=continuation_index_include,
        shard_id=shard_id,
        branch_result_chunk_dir=branch_result_chunk_dir,
        input_paths=input_paths,
    )


def generate_carrion_survivor_continuation_branch_results(
    *,
    seeds: Sequence[int] = TARGET_CARRION_SEEDS,
    ticks: int = DEFAULT_TICKS,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    branch_index_include: Sequence[int] | None = None,
    continuation_index_include: Sequence[int] | None = None,
    shard_id: str | None = None,
    verify_replay: bool = True,
    branch_result_chunk_dir: str | Path | None = DEFAULT_BRANCH_RESULT_CHUNK_DIR,
    max_wall_seconds: float | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    started_at = time.monotonic()
    selected_seeds = tuple(int(seed) for seed in seeds)
    selected_branch_indexes = _normalize_branch_indexes(branch_index_include)
    selected_continuation_indexes = _normalize_continuation_indexes(
        continuation_index_include
    )
    plans = _selected_continuation_plans(selected_continuation_indexes)
    chunk_dir = Path(branch_result_chunk_dir) if branch_result_chunk_dir else None
    branch_results: list[dict[str, object]] = []
    seed_reports: list[dict[str, object]] = []
    generated_count = 0
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
            result = _evaluate_survivor_continuation_branch_point(
                branch_point,
                reference_runs=reference_runs,
                continuation_plans=plans,
                verify_replay=bool(verify_replay),
            )
            branch_results.append(result)
            generated_count += 1
            if chunk_dir is not None:
                write_carrion_survivor_continuation_branch_result_chunk(
                    result,
                    chunk_dir,
                )
            _emit_progress(
                progress_callback,
                {
                    "event": "continuation_branch_result",
                    "shard_id": shard_id,
                    "fixture": "carrion_only",
                    "seed": int(seed),
                    "branch_id": branch_point.point.branch_id,
                    "branch_index": int(branch_point.point.branch_index),
                    "branch_reason": branch_point.branch_reason,
                    "continuation_run_count": len(
                        _list_of_mappings(result.get("continuation_runs"))
                    ),
                    "elapsed_seconds": _elapsed_seconds(started_at),
                },
            )
        if stop_reason is not None:
            break
    branch_point_count = sum(_int(report.get("branch_point_count")) for report in seed_reports)
    partial = stop_reason is not None or len(branch_results) < branch_point_count
    status = {
        "policy": "m3_carrion_survivor_continuation_generation_status_v1",
        "state": "partial" if partial else "complete",
        "partial": bool(partial),
        "stop_reason": stop_reason,
        "elapsed_seconds": _elapsed_seconds(started_at),
        "max_wall_seconds": None if max_wall_seconds is None else float(max_wall_seconds),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in selected_seeds],
        "ticks": int(ticks),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "branch_index_include": (
            None
            if selected_branch_indexes is None
            else [int(index) for index in selected_branch_indexes]
        ),
        "continuation_index_include": [int(index) for index in selected_continuation_indexes],
        "shard_id": shard_id,
        "verify_replay": bool(verify_replay),
        "branch_point_count": int(branch_point_count),
        "continuation_branch_result_count": len(branch_results),
        "generated_branch_result_count": int(generated_count),
        "continuation_run_count": sum(
            len(_list_of_mappings(result.get("continuation_runs")))
            for result in branch_results
        ),
        "chunk_dir": None if chunk_dir is None else str(chunk_dir),
    }
    return {
        "policy": "m3_carrion_survivor_continuation_branch_generation_v1",
        "shard_id": shard_id,
        "generation_status": status,
        "seed_reports": seed_reports,
        "continuation_plans": plans,
        "branch_results": branch_results,
        "continuation_branch_result_count": len(branch_results),
        "continuation_branch_evidence_digest": stable_payload_digest(branch_results),
    }


def merge_carrion_survivor_continuation_archive_shards(
    *,
    autopsy_report: Mapping[str, object],
    v153_report: Mapping[str, object],
    v145_report: Mapping[str, object] | None,
    shard_reports: Sequence[Mapping[str, object]],
    shard_report_paths: Sequence[str | Path] = (),
    shard_chunk_dirs: Sequence[str | Path] = (),
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
    allow_partial_shard_evidence: bool = False,
    input_paths: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not shard_reports:
        raise CarrionSurvivorContinuationArchiveError(
            "v154 survivor-continuation shard merge requires reports"
        )
    source_validation = validate_carrion_survivor_continuation_inputs(
        autopsy_report=autopsy_report,
        v153_report=v153_report,
    )
    merged_by_id: dict[str, dict[str, object]] = {}
    digests_by_id: dict[str, str] = {}
    source_summaries: list[dict[str, object]] = []
    partial_sources: list[dict[str, object]] = []
    merge_identity: dict[str, object] | None = None
    source_paths = [str(path) for path in shard_report_paths]
    observed_seed_set: set[int] = set()
    for shard_index, report in enumerate(shard_reports):
        _validate_shard_contract(report, shard_index=shard_index)
        identity = _merge_identity(report)
        comparable_identity = {
            key: value
            for key, value in identity.items()
            if key != "target_carrion_seeds"
        }
        if merge_identity is None:
            merge_identity = comparable_identity
        elif comparable_identity != merge_identity:
            raise CarrionSurvivorContinuationArchiveError(
                f"shard {shard_index} schema/policy/target/tick/min-floor mismatch"
            )
        shard_seed_set = {int(seed) for seed in _list(identity.get("target_carrion_seeds"))}
        if not shard_seed_set.issubset({int(seed) for seed in target_seeds}):
            raise CarrionSurvivorContinuationArchiveError(
                f"shard {shard_index} target seed outside requested merge seeds"
            )
        observed_seed_set.update(shard_seed_set)
        status = _mapping(report.get("generation_status"))
        source_integrity = _mapping(report.get("source_integrity"))
        failures = [str(failure) for failure in _list(source_integrity.get("failures"))]
        partial = _status_partial(status) or "partial_branch_evidence" in set(failures)
        non_partial_failures = sorted(
            failure for failure in failures if failure != "partial_branch_evidence"
        )
        if source_integrity.get("passed") is not True and non_partial_failures:
            raise CarrionSurvivorContinuationArchiveError(
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
                "source_integrity_failures": failures,
            }
            if not allow_partial_shard_evidence:
                raise CarrionSurvivorContinuationArchiveError(
                    "partial shard evidence requires explicit partial merge: "
                    f"{summary}"
                )
            partial_sources.append(summary)
        branch_results = _list_of_mappings(report.get("continuation_branch_results"))
        expected_digest = str(report.get("continuation_branch_evidence_digest", ""))
        actual_digest = stable_payload_digest(branch_results)
        if expected_digest and expected_digest != actual_digest:
            raise CarrionSurvivorContinuationArchiveError(
                f"shard {shard_index} continuation evidence digest mismatch"
            )
        for result in branch_results:
            _add_merged_branch_result(
                merged_by_id=merged_by_id,
                digests_by_id=digests_by_id,
                result=result,
                source=source_name,
            )
        source_summaries.append(
            {
                "source": source_name,
                "source_path": _optional_index(source_paths, shard_index),
                "shard_id": _shard_id_from_report(report),
                "continuation_branch_result_count": len(branch_results),
                "continuation_branch_evidence_digest": expected_digest or actual_digest,
                "partial": bool(partial),
            }
        )
    for chunk_dir in shard_chunk_dirs:
        chunks = load_carrion_survivor_continuation_branch_result_chunks(chunk_dir)
        for result in chunks.values():
            branch_id = str(result.get("branch_id", ""))
            if branch_id not in merged_by_id:
                raise CarrionSurvivorContinuationArchiveError(
                    "chunk-dir continuation result lacks matching shard report: "
                    f"{chunk_dir}:{branch_id}"
                )
            _add_merged_branch_result(
                merged_by_id=merged_by_id,
                digests_by_id=digests_by_id,
                result=result,
                source=f"chunk_dir:{chunk_dir}",
            )
        source_summaries.append(
            {
                "source": f"chunk_dir:{chunk_dir}",
                "source_path": str(chunk_dir),
                "continuation_branch_result_count": len(chunks),
                "partial": False,
            }
        )
    missing_seeds = sorted({int(seed) for seed in target_seeds} - observed_seed_set)
    if missing_seeds and not allow_partial_shard_evidence:
        raise CarrionSurvivorContinuationArchiveError(
            "partial shard evidence requires explicit partial merge: "
            f"missing_target_seeds={missing_seeds}"
        )
    if missing_seeds:
        partial_sources.append(
            {
                "source": "merged_union",
                "state": "partial",
                "stop_reason": "missing_target_seed_evidence",
                "missing_target_seeds": missing_seeds,
            }
        )
    branch_results = sorted(merged_by_id.values(), key=_branch_result_sort_key)
    identity = _mapping(merge_identity)
    status = {
        "policy": "m3_carrion_survivor_continuation_merged_generation_status_v1",
        "state": "partial" if partial_sources else "complete",
        "partial": bool(partial_sources),
        "stop_reason": "partial_shard_evidence" if partial_sources else None,
        "source_count": len(source_summaries),
        "partial_sources": sorted(partial_sources, key=lambda item: str(item.get("source", ""))),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in target_seeds],
        "ticks": identity.get("ticks"),
        "min_label_count": identity.get("min_label_count"),
        "continuation_index_include": identity.get("continuation_index_include"),
        "continuation_branch_result_count": len(branch_results),
        "continuation_run_count": sum(
            len(_list_of_mappings(result.get("continuation_runs")))
            for result in branch_results
        ),
    }
    evidence = {
        "policy": "m3_carrion_survivor_continuation_shard_merge_generation_evidence_v1",
        "merge_mode": True,
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "source_count": len(source_summaries),
        "continuation_branch_evidence_digest": stable_payload_digest(branch_results),
    }
    report, rows = _finalize_report(
        autopsy_report=autopsy_report,
        v153_report=v153_report,
        v145_report=v145_report or {"route_decision": {"label_blacklist": []}},
        source_validation=source_validation,
        continuation_branch_results=branch_results,
        generation_status=status,
        generation_evidence=evidence,
        target_seeds=target_seeds,
        ticks=_int(identity.get("ticks"), default=DEFAULT_TICKS),
        min_label_count=_int(identity.get("min_label_count"), default=DEFAULT_MIN_SAFE_LABEL_COUNT),
        seed_include=None,
        branch_index_include=None,
        continuation_index_include=[
            int(index) for index in _list(identity.get("continuation_index_include"))
        ],
        shard_id=None,
        branch_result_chunk_dir=None,
        input_paths=input_paths,
    )
    report["shard_merge"] = {
        "policy": "m3_carrion_survivor_continuation_shard_merge_v1",
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "sources": sorted(
            source_summaries,
            key=lambda item: (
                str(item.get("source", "")),
                str(item.get("source_path", "")),
                str(item.get("continuation_branch_evidence_digest", "")),
            ),
        ),
        "partial_sources": status["partial_sources"],
        "duplicate_branch_id_policy": "same_digest_allowed_conflict_rejected",
    }
    return report, rows


def write_carrion_survivor_continuation_branch_result_chunk(
    branch_result: Mapping[str, object],
    chunk_dir: str | Path,
) -> Path:
    branch_id = str(branch_result.get("branch_id", ""))
    if not branch_id:
        raise CarrionSurvivorContinuationArchiveError(
            "cannot write v154 continuation chunk without branch_id"
        )
    path = Path(chunk_dir)
    path.mkdir(parents=True, exist_ok=True)
    chunk_path = path / f"{_slug(branch_id)}.json"
    payload = {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_BRANCH_RESULT_CHUNK_SCHEMA_VERSION,
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_BRANCH_RESULT_CHUNK_POLICY,
        "branch_result_digest": stable_payload_digest(branch_result),
        "branch_result": dict(branch_result),
    }
    with chunk_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return chunk_path


def load_carrion_survivor_continuation_branch_result_chunks(
    chunk_dir: str | Path,
) -> dict[str, dict[str, object]]:
    path = Path(chunk_dir)
    if not path.exists():
        return {}
    if not path.is_dir():
        raise CarrionSurvivorContinuationArchiveError(
            f"v154 chunk path is not a directory: {path}"
        )
    chunks: dict[str, dict[str, object]] = {}
    digests_by_branch_id: dict[str, str] = {}
    for chunk_path in sorted(path.glob("*.json")):
        payload = load_json_report(chunk_path)
        if payload.get("schema_version") != (
            M3_CARRION_SURVIVOR_CONTINUATION_BRANCH_RESULT_CHUNK_SCHEMA_VERSION
        ):
            raise CarrionSurvivorContinuationArchiveError(
                f"v154 branch-result chunk schema mismatch: {chunk_path}"
            )
        if payload.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_BRANCH_RESULT_CHUNK_POLICY:
            raise CarrionSurvivorContinuationArchiveError(
                f"v154 branch-result chunk policy mismatch: {chunk_path}"
            )
        result = _mapping(payload.get("branch_result"))
        branch_id = str(result.get("branch_id", ""))
        if not branch_id:
            raise CarrionSurvivorContinuationArchiveError(
                f"v154 branch-result chunk missing branch_id: {chunk_path}"
            )
        expected = str(payload.get("branch_result_digest", ""))
        actual = stable_payload_digest(result)
        if expected and expected != actual:
            raise CarrionSurvivorContinuationArchiveError(
                f"v154 branch-result chunk digest mismatch: {chunk_path}"
            )
        prior = digests_by_branch_id.get(branch_id)
        if prior is not None and prior != actual:
            raise CarrionSurvivorContinuationArchiveError(
                f"conflicting v154 branch-result chunks for branch_id={branch_id}"
            )
        chunks[branch_id] = dict(result)
        digests_by_branch_id[branch_id] = actual
    return dict(sorted(chunks.items()))


def validate_carrion_survivor_continuation_inputs(
    *,
    autopsy_report: Mapping[str, object],
    v153_report: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    autopsy_classification = _mapping(autopsy_report.get("classification")).get("primary")
    v153_classification = _mapping(v153_report.get("classification")).get("primary")
    if autopsy_report.get("schema_version") != M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION:
        failures.append("v150_schema_mismatch")
    if autopsy_report.get("policy") != M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_POLICY:
        failures.append("v150_policy_mismatch")
    if autopsy_classification != EXPECTED_V150_CLASSIFICATION:
        failures.append("v150_unexpected_classification")
    if v153_report.get("schema_version") != (
        M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION
    ):
        failures.append("v153_schema_mismatch")
    if v153_report.get("policy") != M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY:
        failures.append("v153_policy_mismatch")
    if v153_classification != EXPECTED_V153_CLASSIFICATION:
        failures.append("v153_unexpected_classification")
    if _mapping(v153_report.get("source_integrity")).get("passed") is not True:
        failures.append("v153_source_integrity_not_passed")
    if _mapping(v153_report.get("recommendation")).get("route_closed") is not True:
        failures.append("v153_route_not_closed")
    for prefix, payload in (("v150", autopsy_report), ("v153", v153_report)):
        for key, expected in (
            ("diagnostics_only", True),
            ("training_authorized", False),
            ("promotion_authorized", False),
            ("runtime_promotion_allowed", False),
            ("default_runtime_behavior_changed", False),
            ("runtime_action_selection_changed", False),
        ):
            if payload.get(key) is not expected:
                failures.append(f"{prefix}_{key}_not_{str(expected).lower()}")
    if failures:
        raise CarrionSurvivorContinuationArchiveError(
            "v154 carrion survivor-continuation input validation failed: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "policy": "m3_carrion_survivor_continuation_input_validation_v1",
        "passed": True,
        "failures": [],
        "v150_exact_digest": autopsy_report.get("exact_digest"),
        "v150_report_digest": stable_payload_digest(autopsy_report),
        "v153_exact_digest": v153_report.get("exact_digest"),
        "v153_report_digest": stable_payload_digest(v153_report),
        "v153_classification": v153_classification,
    }


def carrion_survivor_continuation_leakage_scan(
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
        features = _mapping(trainable.get("features"))
        for path, value in _flatten(features):
            lower_path = path.lower()
            if any(token in lower_path for token in FORBIDDEN_TRAINABLE_FEATURE_TOKENS):
                extra_failures.append(
                    {
                        "row_index": int(row_index),
                        "path": f"trainable.features.{path}",
                        "reason": "forbidden_trainable_feature_path_token",
                    }
                )
            if isinstance(value, str) and _looks_like_digest(value):
                extra_failures.append(
                    {
                        "row_index": int(row_index),
                        "path": f"trainable.features.{path}",
                        "reason": "forbidden_trainable_feature_digest_value",
                    }
                )
    failures = [
        *_list_of_mappings(shared.get("failures")),
        *extra_failures,
    ]
    return {
        **dict(shared),
        "policy": "m3_carrion_survivor_continuation_trainable_leakage_scan_v1",
        "passed": shared.get("passed") is True and not extra_failures,
        "shared_leakage_scan": shared,
        "extra_forbidden_feature_tokens": list(FORBIDDEN_TRAINABLE_FEATURE_TOKENS),
        "extra_forbidden_failure_count": len(extra_failures),
        "failures": failures[:32],
    }


def _finalize_report(
    *,
    autopsy_report: Mapping[str, object],
    v153_report: Mapping[str, object],
    v145_report: Mapping[str, object],
    source_validation: Mapping[str, object],
    continuation_branch_results: Sequence[Mapping[str, object]],
    generation_status: Mapping[str, object],
    generation_evidence: Mapping[str, object],
    target_seeds: Sequence[int],
    ticks: int,
    min_label_count: int,
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    continuation_index_include: Sequence[int] | None,
    shard_id: str | None,
    branch_result_chunk_dir: str | Path | None,
    input_paths: Mapping[str, object] | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    branch_results = sorted(
        [dict(result) for result in continuation_branch_results],
        key=_branch_result_sort_key,
    )
    blacklist = _v145_blacklist(v145_report)
    rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    safe_action_counts: Counter[str] = Counter()
    safe_action_counts_by_seed: dict[int, Counter[str]] = {
        int(seed): Counter() for seed in target_seeds
    }
    blacklist_hits: list[dict[str, object]] = []
    for result_index, result in enumerate(branch_results):
        for run_index, run in enumerate(_list_of_mappings(result.get("continuation_runs"))):
            identity = _label_identity(
                result_index=result_index,
                run_index=run_index,
                result=result,
                run=run,
            )
            vet = _survivor_continuation_label_vet(run)
            if vet.get("passed") is not True:
                excluded_rows.append(
                    {
                        **identity,
                        "excluded_reason": "survivor_continuation_vet_failed",
                        "label_vet": vet,
                    }
                )
                continue
            if _blacklisted(identity, blacklist, allow_label_source_row_index=False):
                row = {
                    **identity,
                    "excluded_reason": "invalid_resolution_risk_blacklist",
                    "label_vet": vet,
                }
                excluded_rows.append(row)
                blacklist_hits.append(row)
                continue
            row = _dataset_row(
                row_index=len(rows),
                source_branch_result_index=result_index,
                source_continuation_run_index=run_index,
                result=result,
                selected_run=run,
                label_vet=vet,
            )
            rows.append(row)
            action = str(run.get("forced_action", ""))
            safe_action_counts.update([action])
            seed = _int(result.get("seed"), default=-1)
            if seed in safe_action_counts_by_seed:
                safe_action_counts_by_seed[seed].update([action])
    leakage_scan = carrion_survivor_continuation_leakage_scan(
        rows,
        strict_heldout_seeds=target_seeds,
    )
    replay = _replay_verification_report(branch_results)
    action_coverage = _action_coverage_report(branch_results)
    heuristic_action_source_count = _heuristic_action_source_count(branch_results)
    per_seed_support = _per_seed_support(
        target_seeds=target_seeds,
        branch_results=branch_results,
        rows=rows,
    )
    branch_reason_support = _branch_reason_support(branch_results)
    dominant = _dominant_count_share(safe_action_counts)
    source_integrity = _source_integrity(
        source_validation=source_validation,
        generation_status=generation_status,
        branch_results=branch_results,
        replay=replay,
        action_coverage=action_coverage,
        leakage_scan=leakage_scan,
        heuristic_action_source_count=heuristic_action_source_count,
        per_seed_support=per_seed_support,
        strict_full_scope=_strict_full_scope(
            seed_include=seed_include,
            branch_index_include=branch_index_include,
            continuation_index_include=continuation_index_include,
            shard_id=shard_id,
        ),
    )
    support_floors = _support_floors(
        source_integrity=source_integrity,
        label_count=len(rows),
        min_label_count=int(min_label_count),
        dominant_label_action_share=float(dominant["share"]),
        per_seed_support=per_seed_support,
        replay=replay,
        leakage_scan=leakage_scan,
    )
    classification = _classification(
        source_integrity=source_integrity,
        support_floors=support_floors,
        label_count=len(rows),
        min_label_count=int(min_label_count),
        per_seed_support=per_seed_support,
        dominant_label_action_share=float(dominant["share"]),
    )
    recommendation = _recommendation(classification)
    dataset_digest = stable_payload_digest(rows)
    branch_evidence_digest = stable_payload_digest(branch_results)
    contract = {
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "trainable_dataset_emitted": bool(rows),
        "branch_fixture": "carrion_only",
        "target_seed_policy": "explicit_carrion_only_seeds_13_19_29_37_41_43",
        "branch_point_policy": "carrion_contact_post_hydration_risk_movement_stall_terminal_extinction",
        "candidate_action_policy": "all_currently_valid_public_action_mask_actions",
        "continuation_policy": "short_public_mask_safe_go_explore_style_continuations",
        "label_policy": "replay_verified_survivor_continuation_outcome_improvement_v1",
        "blacklist_policy": "exclude_v145_invalid_resolution_risk_label_sources",
        "trainable_fields": [
            "public observation_input",
            "public action_mask",
            "empty public prior/continuation context",
            "label action",
        ],
        "excluded_trainable_fields": [
            "seed",
            "fixture",
            "branch id",
            "tick",
            "agent id",
            "path",
            "digest",
            "private world state",
            "future outcome",
            "provenance",
        ],
    }
    inputs = {
        **_input_path_payload(input_paths),
        "v150_exact_digest": source_validation.get("v150_exact_digest"),
        "v153_exact_digest": source_validation.get("v153_exact_digest"),
        "v153_classification": source_validation.get("v153_classification"),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in target_seeds],
        "ticks": int(ticks),
        "min_label_count": int(min_label_count),
        "branch_result_chunk_dir": (
            None if branch_result_chunk_dir is None else str(branch_result_chunk_dir)
        ),
        "shard": {
            "shard_id": shard_id,
            "seed_include": None
            if seed_include is None
            else [int(seed) for seed in seed_include],
            "branch_index_include": None
            if branch_index_include is None
            else [int(index) for index in branch_index_include],
            "continuation_index_include": None
            if continuation_index_include is None
            else [int(index) for index in continuation_index_include],
        },
    }
    report = {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION,
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY,
        "diagnostics_only": True,
        "contract": contract,
        "inputs": inputs,
        "generation_status": dict(generation_status),
        "generation_evidence": dict(generation_evidence),
        "source_validation": dict(source_validation),
        "source_integrity": source_integrity,
        "blacklist": {
            "policy": "v145_invalid_resolution_risk_blacklist_v1",
            "blacklist_count": len(blacklist),
            "blacklist_hit_count": len(blacklist_hits),
            "blacklist_hits": blacklist_hits[:64],
        },
        "label_count": len(rows),
        "per_seed_support": per_seed_support,
        "branch_reason_support": branch_reason_support,
        "action_distribution": {
            "label_action_counts": dict(sorted(safe_action_counts.items())),
            "label_action_counts_by_seed": {
                str(seed): dict(sorted(counts.items()))
                for seed, counts in sorted(safe_action_counts_by_seed.items())
            },
            "dominant_label_action": dominant["key"],
            "dominant_label_action_count": dominant["count"],
            "dominant_label_action_share": dominant["share"],
        },
        "replay_verification": replay,
        "action_coverage": action_coverage,
        "leakage_scan": leakage_scan,
        "dataset": {
            "row_schema_version": M3_CARRION_SURVIVOR_CONTINUATION_DATASET_ROW_SCHEMA_VERSION,
            "feature_policy": M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_POLICY,
            "row_count": len(rows),
            "dataset_digest": dataset_digest,
            "dominant_label_action_share": dominant["share"],
            "leakage_scan": leakage_scan,
        },
        "excluded_row_count": len(excluded_rows),
        "excluded_rows": excluded_rows[:128],
        "support_floors": support_floors,
        "classification": {"primary": classification, "labels": [classification]},
        "recommendation": recommendation,
        "continuation_branch_results": branch_results,
        "continuation_branch_result_count": len(branch_results),
        "continuation_branch_evidence_digest": branch_evidence_digest,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "non_promoted": True,
    }
    exact_payload = {
        "schema_version": report["schema_version"],
        "policy": report["policy"],
        "contract": contract,
        "inputs": {
            "v150_exact_digest": inputs["v150_exact_digest"],
            "v153_exact_digest": inputs["v153_exact_digest"],
            "v153_classification": inputs["v153_classification"],
            "target_carrion_seeds": inputs["target_carrion_seeds"],
            "ticks": inputs["ticks"],
            "min_label_count": inputs["min_label_count"],
        },
        "continuation_branch_evidence_digest": branch_evidence_digest,
        "dataset_digest": dataset_digest,
        "source_integrity": source_integrity,
        "support_floors": support_floors,
        "classification": report["classification"],
        "recommendation": recommendation,
    }
    report["exact_digest"] = stable_payload_digest(exact_payload)
    report["provenance"] = {
        "exact_digest_payload_policy": (
            "stable_payload_digest_of_v154_contract_sources_branch_evidence_dataset_and_closeout_v1"
        ),
        "contract_digest": stable_payload_digest(contract),
        "exact_digest": report["exact_digest"],
    }
    return report, rows


def _evaluate_survivor_continuation_branch_point(
    branch_point: object,
    *,
    reference_runs: Mapping[int, Mapping[str, Mapping[str, object]]],
    continuation_plans: Sequence[Mapping[str, object]],
    verify_replay: bool,
) -> dict[str, object]:
    point = branch_point.point
    valid_actions = _candidate_actions(point)
    refs = _mapping(reference_runs.get(point.seed))
    baseline_ref = _mapping(refs.get("baseline"))
    override_ref = _mapping(refs.get("v142_override"))
    runs: list[dict[str, object]] = []
    for action in valid_actions:
        for plan in continuation_plans:
            runs.append(
                _execute_continuation_branch(
                    point,
                    forced_action=action,
                    continuation_plan=plan,
                    baseline_ref=baseline_ref,
                    override_ref=override_ref,
                    verify_replay=bool(verify_replay),
                )
            )
    return {
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
        "candidate_actions": valid_actions,
        "candidate_action_count": len(valid_actions),
        "continuation_indexes": [
            int(plan.get("continuation_index", -1)) for plan in continuation_plans
        ],
        "continuation_count": len(continuation_plans),
        "branch_state_digest": point.branch_state_digest,
        "source_trajectory_path": point.source_trajectory_path,
        "public_features": {
            "observation_input": dict(point.observation_input),
            "action_mask": dict(sorted(point.action_mask.items())),
        },
        "carrion_archive_context": _carrion_branch_context(branch_point),
        "continuation_runs": sorted(runs, key=_continuation_run_sort_key),
        "diagnostics_only": True,
    }


def _execute_continuation_branch(
    point: BroadRegressionBranchPoint,
    *,
    forced_action: str,
    continuation_plan: Mapping[str, object],
    baseline_ref: Mapping[str, object],
    override_ref: Mapping[str, object],
    verify_replay: bool,
) -> dict[str, object]:
    run, digest = _execute_continuation_once(
        point,
        forced_action=forced_action,
        continuation_plan=continuation_plan,
        baseline_ref=baseline_ref,
        override_ref=override_ref,
    )
    verification = None
    if verify_replay:
        replay, replay_digest = _execute_continuation_once(
            point,
            forced_action=forced_action,
            continuation_plan=continuation_plan,
            baseline_ref=baseline_ref,
            override_ref=override_ref,
        )
        verification = {
            "verified": replay_digest == digest,
            "expected_digest": digest,
            "actual_digest": replay_digest,
            "replay_alive_agents": replay.get("alive_agents"),
            "replay_births": replay.get("births"),
            "replay_deaths": replay.get("deaths"),
        }
    run["replay_digest"] = digest
    run["replay_verification"] = verification
    run["continuation_run_id"] = _continuation_run_id(
        branch_id=point.branch_id,
        action=forced_action,
        continuation_index=_int(continuation_plan.get("continuation_index")),
    )
    return run


def _execute_continuation_once(
    point: BroadRegressionBranchPoint,
    *,
    forced_action: str,
    continuation_plan: Mapping[str, object],
    baseline_ref: Mapping[str, object],
    override_ref: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    world = deepcopy(point.world)
    delegate = _normal_mind_v3_delegate(world.policy)
    sequence = _action_sequence(
        forced_action=forced_action,
        continuation_plan=continuation_plan,
    )
    wrapper = _ForcedShortContinuationThenMindV3Policy(
        target_agent_id=point.agent_id,
        continuation_index=_int(continuation_plan.get("continuation_index")),
        action_sequence=sequence,
        delegate=delegate,
    )
    world.policy = wrapper
    _configure_manual_summary_run(world)
    for tick in range(point.branch_tick, point.ticks):
        world.tick = tick
        world._run_tick()
        if not world.alive_agents():
            break
    first_used = bool(
        wrapper.forced_actions
        and _int(wrapper.forced_actions[0].get("sequence_index"), default=-1) == 0
        and wrapper.forced_actions[0].get("action") == forced_action
    )
    run = _summarize_branch_world(
        world,
        point=point,
        forced_action=forced_action,
        forced_used=first_used,
        baseline_ref=baseline_ref,
        override_ref=override_ref,
    )
    run.update(
        {
            "continuation_index": _int(continuation_plan.get("continuation_index")),
            "continuation_name": continuation_plan.get("name"),
            "continuation_tail_actions": list(
                _object_sequence(continuation_plan.get("tail_actions"))
            ),
            "planned_action_sequence": list(sequence),
            "forced_action_sequence_used": list(wrapper.forced_actions),
            "skipped_continuation_actions": list(wrapper.skipped_actions),
            "forced_sequence_action_count": len(wrapper.forced_actions),
            "skipped_sequence_action_count": len(wrapper.skipped_actions),
            "continuation_forced_action_source_count": sum(
                int(count)
                for source, count in _mapping(run.get("action_source_counts")).items()
                if str(source).startswith("carrion_survivor_continuation_force:")
            ),
        }
    )
    run["outcome_improvement"] = _outcome_improvement(run)
    return run, stable_payload_digest(_continuation_run_digest_payload(run))


def _survivor_continuation_label_vet(
    run: Mapping[str, object],
) -> dict[str, object]:
    baseline = _mapping(run.get("deltas_vs_baseline"))
    replay = _mapping(run.get("replay_verification"))
    improvement = _mapping(run.get("outcome_improvement"))
    resolved_invalid_delta = baseline.get("unsupported_resolved_action_count")
    floors = [
        _floor(
            "forced_label_action_was_used",
            run.get("forced_action_used") is True,
            observed=run.get("forced_action_used"),
            required=True,
        ),
        _floor(
            "forced_label_action_supported_by_public_mask",
            run.get("forced_action_supported") is True,
            observed=run.get("forced_action_supported"),
            required=True,
        ),
        _floor(
            "continuation_replay_deterministic",
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
            "zero_heuristic_action_source_count",
            _int(run.get("heuristic_action_source_count")) == 0,
            observed=run.get("heuristic_action_source_count"),
            required=0,
        ),
        _floor(
            "replay_verified_outcome_improvement",
            improvement.get("improved") is True,
            observed=improvement,
            required=(
                "survival, hydration recovery, reproduction readiness, or fewer blockers"
            ),
        ),
    ]
    failed = [floor for floor in floors if floor.get("passed") is not True]
    return {
        "policy": "m3_carrion_survivor_continuation_label_vet_v1",
        "passed": not failed,
        "failed_floor_count": len(failed),
        "floors": floors,
        "outcome_improvement": improvement,
    }


def _outcome_improvement(run: Mapping[str, object]) -> dict[str, object]:
    baseline = _mapping(run.get("deltas_vs_baseline"))
    first = _mapping(run.get("first_action_outcome"))
    first_outcome = _mapping(first.get("outcome"))
    survival = (
        _int(baseline.get("alive_agents")) > 0
        or _int(baseline.get("target_alive")) > 0
        or _int(baseline.get("deaths")) < 0
    )
    hydration_delta = _optional_float(baseline.get("target_hydration_ratio"))
    target_hydration = _optional_float(run.get("target_hydration_ratio_at_end"))
    hydration_recovery = (
        (hydration_delta is not None and hydration_delta > 0.0)
        or (target_hydration is not None and target_hydration >= 0.55)
    )
    reproduction = (
        _int(baseline.get("births")) > 0
        or first_outcome.get("reproduced") is True
        or first_outcome.get("reproduction_ready_after") is True
    )
    fewer_blockers = (
        _int(baseline.get("unsupported_requested_action_count")) < 0
        or _int(baseline.get("unsupported_resolved_action_count")) < 0
        or _int(baseline.get("deaths")) < 0
    )
    labels = []
    if survival:
        labels.append("survival")
    if hydration_recovery:
        labels.append("hydration_recovery")
    if reproduction:
        labels.append("reproduction_readiness")
    if fewer_blockers:
        labels.append("fewer_blockers")
    return {
        "policy": "m3_carrion_survivor_continuation_outcome_improvement_v1",
        "improved": bool(labels),
        "labels": labels,
        "survival_improved": bool(survival),
        "hydration_recovery": bool(hydration_recovery),
        "reproduction_readiness": bool(reproduction),
        "fewer_blockers": bool(fewer_blockers),
        "no_resolved_invalid_increase": (
            _optional_float(baseline.get("unsupported_resolved_action_count")) is not None
            and _float(baseline.get("unsupported_resolved_action_count")) <= 0.0
        ),
    }


def _dataset_row(
    *,
    row_index: int,
    source_branch_result_index: int,
    source_continuation_run_index: int,
    result: Mapping[str, object],
    selected_run: Mapping[str, object],
    label_vet: Mapping[str, object],
) -> dict[str, object]:
    features = _mapping(result.get("public_features"))
    action = str(selected_run.get("forced_action", ""))
    return {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_DATASET_ROW_SCHEMA_VERSION,
        "trainable": {
            "feature_policy": M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_POLICY,
            "features": {
                "observation_input": features.get("observation_input"),
                "action_mask": features.get("action_mask"),
                "prior_public_context": [],
            },
            "label": {
                "action": action,
                "label_policy": (
                    "replay_verified_survivor_continuation_outcome_improvement_v1"
                ),
            },
        },
        "metadata": {
            "row_index": int(row_index),
            "source_branch_result_index": int(source_branch_result_index),
            "source_continuation_run_index": int(source_continuation_run_index),
            "seed": result.get("seed"),
            "fixture": result.get("fixture"),
            "branch_id": result.get("branch_id"),
            "branch_tick": result.get("branch_tick"),
            "record_index": result.get("record_index"),
            "agent_id": result.get("agent_id"),
            "continuation_index": selected_run.get("continuation_index"),
            "continuation_name": selected_run.get("continuation_name"),
            "replay_digest": selected_run.get("replay_digest"),
            "candidate_actions": result.get("candidate_actions"),
            "label_vet": label_vet,
            "outcome_evidence": {
                "deltas_vs_baseline": selected_run.get("deltas_vs_baseline"),
                "target_terminal": selected_run.get("target_terminal"),
                "outcome_improvement": selected_run.get("outcome_improvement"),
                "replay_verification": selected_run.get("replay_verification"),
            },
            "provenance": {
                "policy": M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY,
                "public_feature_digest": stable_payload_digest(features),
                "source_run_digest": stable_payload_digest(selected_run),
            },
        },
    }


def _source_integrity(
    *,
    source_validation: Mapping[str, object],
    generation_status: Mapping[str, object],
    branch_results: Sequence[Mapping[str, object]],
    replay: Mapping[str, object],
    action_coverage: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    heuristic_action_source_count: int,
    per_seed_support: Mapping[str, object],
    strict_full_scope: bool,
) -> dict[str, object]:
    failures: list[str] = []
    if source_validation.get("passed") is not True:
        failures.append("source_validation_failed")
    if _status_partial(generation_status):
        failures.append("partial_branch_evidence")
    if not branch_results:
        failures.append("no_continuation_branch_results")
    if replay.get("complete") is not True:
        failures.append("replay_verification_incomplete")
    if action_coverage.get("complete") is not True:
        failures.append("not_all_valid_action_continuations_evaluated")
    if leakage_scan.get("passed") is not True:
        failures.append("trainable_leakage_detected")
    if int(heuristic_action_source_count) != 0:
        failures.append("heuristic_action_source_count_nonzero")
    if strict_full_scope and _list(per_seed_support.get("missing_branch_result_seeds")):
        failures.append("missing_target_seed_branch_evidence")
    return {
        "policy": "m3_carrion_survivor_continuation_source_integrity_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "generation_state": generation_status.get("state"),
        "continuation_branch_result_count": len(branch_results),
        "replay_verification_complete": replay.get("complete") is True,
        "action_coverage_complete": action_coverage.get("complete") is True,
        "leakage_scan_passed": leakage_scan.get("passed") is True,
        "heuristic_action_source_count": int(heuristic_action_source_count),
    }


def _support_floors(
    *,
    source_integrity: Mapping[str, object],
    label_count: int,
    min_label_count: int,
    dominant_label_action_share: float,
    per_seed_support: Mapping[str, object],
    replay: Mapping[str, object],
    leakage_scan: Mapping[str, object],
) -> dict[str, object]:
    floors = [
        _floor(
            "source_integrity_passed",
            source_integrity.get("passed") is True,
            observed=source_integrity.get("failures"),
            required=[],
        ),
        _floor(
            "label_count_gte_minimum",
            int(label_count) >= int(min_label_count),
            observed=int(label_count),
            required=int(min_label_count),
        ),
        _floor(
            "dominant_label_action_share_lte_0_50",
            float(dominant_label_action_share) <= MAX_DOMINANT_LABEL_ACTION_SHARE,
            observed=_round(dominant_label_action_share),
            required=MAX_DOMINANT_LABEL_ACTION_SHARE,
        ),
        _floor(
            "all_target_seeds_have_branch_results",
            not _list(per_seed_support.get("missing_branch_result_seeds")),
            observed=per_seed_support.get("missing_branch_result_seeds"),
            required=[],
            fixture="carrion_only",
        ),
        _floor(
            "all_target_seeds_have_labels",
            not _list(per_seed_support.get("missing_label_seeds")),
            observed=per_seed_support.get("missing_label_seeds"),
            required=[],
            fixture="carrion_only",
        ),
        _floor(
            "replay_verification_complete",
            replay.get("complete") is True,
            observed=replay,
            required="all continuation action runs verified",
        ),
        _floor(
            "no_trainable_leakage",
            leakage_scan.get("passed") is True,
            observed=leakage_scan.get("failures"),
            required=[],
        ),
    ]
    failed = [floor for floor in floors if floor.get("passed") is not True]
    return {
        "policy": "m3_carrion_survivor_continuation_support_floors_v1",
        "passed": not failed,
        "first_failed_floor": None if not failed else failed[0]["name"],
        "floors": floors,
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    support_floors: Mapping[str, object],
    label_count: int,
    min_label_count: int,
    per_seed_support: Mapping[str, object],
    dominant_label_action_share: float,
) -> str:
    failures = set(_list(source_integrity.get("failures")))
    if "partial_branch_evidence" in failures:
        return "m3_carrion_survivor_continuation_archive_partial_no_training"
    if source_integrity.get("passed") is not True:
        return "m3_carrion_survivor_continuation_archive_source_integrity_failed_no_training"
    if int(label_count) < int(min_label_count) or _list(per_seed_support.get("missing_label_seeds")):
        return "m3_carrion_survivor_continuation_archive_support_limited_closed_no_training"
    if float(dominant_label_action_share) > MAX_DOMINANT_LABEL_ACTION_SHARE:
        return "m3_carrion_survivor_continuation_archive_label_distribution_collapsed_closed_no_training"
    if support_floors.get("passed") is True:
        return "m3_carrion_survivor_continuation_archive_support_ready_no_training"
    return "m3_carrion_survivor_continuation_archive_blocked_closed_no_training"


def _recommendation(classification: str) -> dict[str, object]:
    ready = (
        classification
        == "m3_carrion_survivor_continuation_archive_support_ready_no_training"
    )
    return {
        "policy": "m3_carrion_survivor_continuation_route_recommendation_v1",
        "route_ready_for_review": bool(ready),
        "route_closed": not ready,
        "recommended_next_route": (
            "review_replay_verified_survivor_continuation_archive_before_any_training_route"
            if ready
            else "support_limited_closed_no_training_return_to_recovery_archive_or_expand_go_explore_survivor_continuations"
        ),
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
    }


def _per_seed_support(
    *,
    target_seeds: Sequence[int],
    branch_results: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    branch_counts = Counter(_int(result.get("seed")) for result in branch_results)
    label_counts = Counter(
        _int(_mapping(row.get("metadata")).get("seed")) for row in rows
    )
    branch_reasons_by_seed: dict[int, Counter[str]] = {
        int(seed): Counter() for seed in target_seeds
    }
    label_actions_by_seed: dict[int, Counter[str]] = {
        int(seed): Counter() for seed in target_seeds
    }
    for result in branch_results:
        seed = _int(result.get("seed"))
        context = _mapping(result.get("carrion_archive_context"))
        reason = str(context.get("branch_reason", "unknown"))
        if seed in branch_reasons_by_seed:
            branch_reasons_by_seed[seed].update([reason])
    for row in rows:
        metadata = _mapping(row.get("metadata"))
        trainable = _mapping(row.get("trainable"))
        label = _mapping(trainable.get("label"))
        seed = _int(metadata.get("seed"))
        action = str(label.get("action", ""))
        if seed in label_actions_by_seed and action:
            label_actions_by_seed[seed].update([action])
    per_seed = []
    for seed in target_seeds:
        resolved = int(seed)
        per_seed.append(
            {
                "fixture": "carrion_only",
                "seed": resolved,
                "branch_result_count": int(branch_counts.get(resolved, 0)),
                "label_count": int(label_counts.get(resolved, 0)),
                "branch_reason_counts": dict(sorted(branch_reasons_by_seed[resolved].items())),
                "missing_branch_reasons": [
                    reason
                    for reason in CARRION_BRANCH_REASONS
                    if branch_reasons_by_seed[resolved].get(reason, 0) <= 0
                ],
                "label_action_counts": dict(sorted(label_actions_by_seed[resolved].items())),
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_per_seed_support_v1",
        "target_seeds": [int(seed) for seed in target_seeds],
        "per_seed": per_seed,
        "missing_branch_result_seeds": [
            int(seed)
            for seed in target_seeds
            if branch_counts.get(int(seed), 0) <= 0
        ],
        "missing_label_seeds": [
            int(seed)
            for seed in target_seeds
            if label_counts.get(int(seed), 0) <= 0
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
        "policy": "m3_carrion_survivor_continuation_branch_reason_support_v1",
        "branch_reason_counts": dict(sorted(counts.items())),
        "branch_reason_counts_by_seed": dict(sorted(by_seed_reason.items())),
        "missing_branch_reasons": [
            reason for reason in CARRION_BRANCH_REASONS if counts.get(reason, 0) <= 0
        ],
    }


def _replay_verification_report(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    run_count = 0
    verified_count = 0
    missing_count = 0
    failed_count = 0
    for result in branch_results:
        for run in _list_of_mappings(result.get("continuation_runs")):
            run_count += 1
            replay = run.get("replay_verification")
            if not isinstance(replay, Mapping):
                missing_count += 1
                continue
            if replay.get("verified") is True:
                verified_count += 1
            else:
                failed_count += 1
    return {
        "policy": "m3_carrion_survivor_continuation_replay_verification_v1",
        "complete": (
            run_count > 0
            and verified_count == run_count
            and missing_count == 0
            and failed_count == 0
        ),
        "continuation_run_count": int(run_count),
        "verified_count": int(verified_count),
        "missing_count": int(missing_count),
        "failed_count": int(failed_count),
    }


def _action_coverage_report(
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    missing: list[dict[str, object]] = []
    for result_index, result in enumerate(branch_results):
        valid_actions = set(_valid_public_mask_actions(result))
        indexes = {
            _int(index)
            for index in _list(result.get("continuation_indexes"))
            if _int(index, default=-1) >= 0
        }
        observed = {
            (str(run.get("forced_action")), _int(run.get("continuation_index")))
            for run in _list_of_mappings(result.get("continuation_runs"))
        }
        for action in sorted(valid_actions, key=_action_order):
            for index in sorted(indexes):
                if (action, index) not in observed:
                    missing.append(
                        {
                            "source_branch_result_index": int(result_index),
                            "branch_id": result.get("branch_id"),
                            "seed": _int(result.get("seed")),
                            "missing_action": action,
                            "missing_continuation_index": int(index),
                        }
                    )
    return {
        "policy": "m3_carrion_survivor_continuation_action_coverage_v1",
        "complete": not missing,
        "branch_result_count": len(branch_results),
        "missing_action_continuation_count": len(missing),
        "first_missing_action_continuation": missing[0] if missing else None,
    }


def _heuristic_action_source_count(
    branch_results: Sequence[Mapping[str, object]],
) -> int:
    return sum(
        _int(run.get("heuristic_action_source_count"))
        for result in branch_results
        for run in _list_of_mappings(result.get("continuation_runs"))
    )


def _label_identity(
    *,
    result_index: int,
    run_index: int,
    result: Mapping[str, object],
    run: Mapping[str, object],
) -> dict[str, object]:
    return {
        "row_index": int(result_index),
        "source_branch_result_index": int(result_index),
        "source_continuation_run_index": int(run_index),
        "seed": _int(result.get("seed")),
        "fixture": result.get("fixture"),
        "branch_id": result.get("branch_id"),
        "branch_tick": _int(result.get("branch_tick")),
        "agent_id": _int(result.get("agent_id")),
        "label_action": run.get("forced_action"),
        "continuation_index": run.get("continuation_index"),
    }


def _candidate_actions(point: BroadRegressionBranchPoint) -> list[str]:
    mask = _bool_action_mask(point.action_mask)
    return [action for action in ACTION_NAMES if bool(mask.get(action, False))]


def _valid_public_mask_actions(result: Mapping[str, object]) -> list[str]:
    features = _mapping(result.get("public_features"))
    mask = _mapping(features.get("action_mask"))
    valid = [action for action in ACTION_NAMES if bool(mask.get(action, False))]
    if valid:
        return valid
    return [
        str(action)
        for action in _list(result.get("candidate_actions"))
        if str(action) in ACTION_NAMES
    ]


def _strict_full_scope(
    *,
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    continuation_index_include: Sequence[int] | None,
    shard_id: str | None,
) -> bool:
    return (
        seed_include is None
        and branch_index_include is None
        and continuation_index_include is None
        and not shard_id
    )


def _object_sequence(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _action_sequence(
    *,
    forced_action: str,
    continuation_plan: Mapping[str, object],
) -> tuple[str, ...]:
    tail = []
    for action in _object_sequence(continuation_plan.get("tail_actions")):
        resolved = forced_action if str(action) == "__label__" else str(action)
        if resolved in ACTION_NAMES:
            tail.append(resolved)
    return (str(forced_action), *tail)


def _continuation_run_id(
    *,
    branch_id: str,
    action: str,
    continuation_index: int,
) -> str:
    return f"{branch_id}::action={action}::continuation={int(continuation_index)}"


def _continuation_run_digest_payload(run: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in sorted(run.items())
        if key not in {"replay_digest", "replay_verification"}
    }


def _continuation_run_sort_key(run: Mapping[str, object]) -> tuple[int, int, str]:
    action = str(run.get("forced_action", ""))
    return (
        _action_order(action),
        _int(run.get("continuation_index")),
        str(run.get("continuation_run_id", "")),
    )


def _branch_result_sort_key(result: Mapping[str, object]) -> tuple[int, int, int, int, str]:
    return (
        0 if result.get("fixture") == "carrion_only" else 1,
        _int(result.get("seed")),
        _int(result.get("branch_index")),
        _int(result.get("branch_tick")),
        str(result.get("branch_id", "")),
    )


def _add_merged_branch_result(
    *,
    merged_by_id: dict[str, dict[str, object]],
    digests_by_id: dict[str, str],
    result: Mapping[str, object],
    source: str,
) -> None:
    branch_id = str(result.get("branch_id", ""))
    if not branch_id:
        raise CarrionSurvivorContinuationArchiveError(
            f"merged v154 branch result missing branch_id: {source}"
        )
    digest = stable_payload_digest(result)
    prior = digests_by_id.get(branch_id)
    if prior is not None and prior != digest:
        raise CarrionSurvivorContinuationArchiveError(
            f"duplicate branch_id with different digest: {branch_id}"
        )
    merged_by_id[branch_id] = dict(result)
    digests_by_id[branch_id] = digest


def _validate_shard_contract(report: Mapping[str, object], *, shard_index: int) -> None:
    if report.get("schema_version") != M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION:
        raise CarrionSurvivorContinuationArchiveError(
            f"shard {shard_index} schema mismatch"
        )
    if report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY:
        raise CarrionSurvivorContinuationArchiveError(
            f"shard {shard_index} policy mismatch"
        )
    for key, expected in (
        ("diagnostics_only", True),
        ("training_authorized", False),
        ("promotion_authorized", False),
        ("runtime_promotion_allowed", False),
        ("default_runtime_behavior_changed", False),
        ("runtime_action_selection_changed", False),
    ):
        if report.get(key) is not expected:
            raise CarrionSurvivorContinuationArchiveError(
                f"shard {shard_index} {key} mismatch"
            )


def _merge_identity(report: Mapping[str, object]) -> dict[str, object]:
    inputs = _mapping(report.get("inputs"))
    shard = _mapping(inputs.get("shard"))
    status = _mapping(report.get("generation_status"))
    target_seeds = [
        int(seed)
        for seed in (
            _list(shard.get("seed_include"))
            or _list(status.get("target_carrion_seeds"))
            or _list(inputs.get("target_carrion_seeds"))
        )
    ]
    if inputs.get("target_fixture") != "carrion_only":
        raise CarrionSurvivorContinuationArchiveError(
            "v154 shard target fixture mismatch"
        )
    if not target_seeds:
        raise CarrionSurvivorContinuationArchiveError(
            "v154 shard missing target seed identity"
        )
    continuation_indexes = _list(status.get("continuation_index_include"))
    if not continuation_indexes:
        continuation_indexes = _list(
            _mapping(inputs.get("shard")).get("continuation_index_include")
        )
    return {
        "schema_version": report.get("schema_version"),
        "policy": report.get("policy"),
        "target_fixture": inputs.get("target_fixture"),
        "target_carrion_seeds": [int(seed) for seed in target_seeds],
        "ticks": _int(inputs.get("ticks") or status.get("ticks")),
        "min_label_count": _int(inputs.get("min_label_count")),
        "v150_exact_digest": inputs.get("v150_exact_digest"),
        "v153_exact_digest": inputs.get("v153_exact_digest"),
        "continuation_index_include": [int(index) for index in continuation_indexes],
    }


def _shard_source_name(report: Mapping[str, object], *, shard_index: int) -> str:
    shard_id = _shard_id_from_report(report)
    if shard_id:
        return f"report:{shard_id}"
    digest = str(report.get("continuation_branch_evidence_digest", ""))
    if digest:
        return f"report:{digest[:12]}"
    return f"report:{shard_index}"


def _shard_id_from_report(report: Mapping[str, object]) -> str | None:
    inputs = _mapping(report.get("inputs"))
    shard = _mapping(inputs.get("shard"))
    status = _mapping(report.get("generation_status"))
    evidence = _mapping(report.get("generation_evidence"))
    for value in (shard.get("shard_id"), status.get("shard_id"), evidence.get("shard_id")):
        if isinstance(value, str) and value:
            return value
    return None


def _precomputed_generation_status(
    *,
    branch_results: Sequence[Mapping[str, object]],
    target_seeds: Sequence[int],
    ticks: int,
    branch_index_include: Sequence[int] | None,
    continuation_index_include: Sequence[int] | None,
    shard_id: str | None,
) -> dict[str, object]:
    return {
        "policy": "precomputed_m3_carrion_survivor_continuation_generation_status_v1",
        "state": "complete",
        "partial": False,
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in target_seeds],
        "ticks": int(ticks),
        "branch_index_include": None
        if branch_index_include is None
        else [int(index) for index in branch_index_include],
        "continuation_index_include": _normalize_continuation_indexes(
            continuation_index_include
        ),
        "shard_id": shard_id,
        "continuation_branch_result_count": len(branch_results),
        "continuation_run_count": sum(
            len(_list_of_mappings(result.get("continuation_runs")))
            for result in branch_results
        ),
    }


def _input_path_payload(input_paths: Mapping[str, object] | None) -> dict[str, object]:
    if input_paths is None:
        return {
            "v150_autopsy_report": None,
            "v153_closeout_report": None,
            "v145_report": None,
            "merge_shard_reports": [],
            "merge_shard_chunk_dirs": [],
        }
    return {
        "v150_autopsy_report": _input_path_value(input_paths.get("v150_autopsy_report")),
        "v153_closeout_report": _input_path_value(input_paths.get("v153_closeout_report")),
        "v145_report": _input_path_value(input_paths.get("v145_report")),
        "merge_shard_reports": _input_path_sequence(input_paths.get("merge_shard_reports")),
        "merge_shard_chunk_dirs": _input_path_sequence(input_paths.get("merge_shard_chunk_dirs")),
    }


def _load_optional_v145_report(
    path: str | Path,
    *,
    allow_missing: bool,
) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.exists():
        if allow_missing:
            return {"route_decision": {"label_blacklist": []}}
        raise CarrionSurvivorContinuationArchiveError(f"missing v145 report: {path}")
    return load_json_report(resolved)


def _selected_continuation_plans(
    continuation_indexes: Sequence[int],
) -> list[dict[str, object]]:
    index_set = {int(index) for index in continuation_indexes}
    return [
        dict(plan)
        for plan in DEFAULT_CONTINUATION_PLANS
        if _int(plan.get("continuation_index"), default=-1) in index_set
    ]


def _normalize_continuation_indexes(
    continuation_index_include: Sequence[int] | None,
) -> list[int]:
    if continuation_index_include is None:
        return [
            _int(plan.get("continuation_index"))
            for plan in DEFAULT_CONTINUATION_PLANS
        ]
    resolved = sorted({int(index) for index in continuation_index_include})
    valid = {
        _int(plan.get("continuation_index"))
        for plan in DEFAULT_CONTINUATION_PLANS
    }
    invalid = [int(index) for index in resolved if int(index) not in valid]
    if invalid:
        raise CarrionSurvivorContinuationArchiveError(
            f"invalid continuation indexes: {invalid}"
        )
    return resolved


def _normalize_branch_indexes(
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
        raise CarrionSurvivorContinuationArchiveError(
            f"invalid carrion branch indexes: {invalid}"
        )
    return resolved


def _dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    if not counts:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = sorted(
        counts.items(),
        key=lambda item: (
            -int(item[1]),
            _action_order(str(item[0])),
            str(item[0]),
        ),
    )[0]
    return {
        "key": key,
        "count": int(count),
        "share": _round(float(count) / float(sum(counts.values()))),
    }


def _status_partial(status: Mapping[str, object]) -> bool:
    return status.get("partial") is True or status.get("state") == "partial"


def _action_order(action: str) -> int:
    return ACTION_NAMES.index(action) if action in ACTION_NAMES else len(ACTION_NAMES)


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


def _optional_index(values: Sequence[str], index: int) -> str | None:
    if 0 <= int(index) < len(values):
        return values[int(index)]
    return None


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


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _float(value: object, *, default: float = 0.0) -> float:
    resolved = _optional_float(value)
    return default if resolved is None else float(resolved)


def _is_finite_number(value: object) -> bool:
    return _optional_float(value) is not None


def _round(value: object) -> float:
    resolved = _optional_float(value)
    return 0.0 if resolved is None else round(float(resolved), 6)


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


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    chars = []
    for char in text:
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_", ".", ":"}:
            chars.append("-")
        else:
            chars.append("-")
    slug = "".join(chars).strip("-")
    return slug or "unknown"
