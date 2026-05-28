from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES, MOVEMENT_ACTIONS
from evolution_sim.mind.dataset import (
    TRAJECTORY_EPISODE_ID_FIELD,
    TrajectoryJsonlDataset,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.rollout_context import rollout_context_event_from_record

MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_SCHEMA_VERSION = (
    "mind_v3_transition_aligned_recovery_audit_v1"
)
MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_POLICY = (
    "diagnostics_only_transition_aligned_first_recovery_support_and_mask_drift_v1"
)

ALLOWED_CLASSIFICATION_LABELS: tuple[str, ...] = (
    "transition_aligned_first_recovery_constructible",
    "transition_aligned_first_recovery_not_constructible",
    "transition_aligned_first_recovery_inconclusive",
    "branch_oracle_first_recovery_overlap_present",
    "branch_oracle_first_recovery_overlap_sparse",
    "branch_oracle_first_recovery_overlap_absent",
    "supported_recovery_actions_observation_legal_resolution_legal",
    "supported_recovery_actions_observation_legal_resolution_invalid",
    "supported_recovery_actions_not_observation_legal",
    "movement_mask_drift_occupancy_race",
    "movement_mask_drift_blocked_route",
    "movement_mask_drift_depleted_resource",
    "movement_mask_drift_other_public_resolution_path",
    "movement_mask_drift_inconclusive",
    "seed29_birth_regression_movement_failure_after_carrion",
    "seed29_birth_regression_unsupported_resolution",
    "seed29_birth_regression_missed_drink_after_carrion",
    "seed29_birth_regression_delayed_reproduction_readiness",
    "seed29_birth_regression_inconclusive_public_evidence_gap",
    "heldout_support_sufficient_for_v108",
    "heldout_support_insufficient_for_v108",
    "missing_evidence_inconclusive",
    "no_material_blocker_detected",
)
CLASSIFICATION_LABEL_SET = frozenset(ALLOWED_CLASSIFICATION_LABELS)

DEFAULT_MIN_OVERLAP_PRESENT_SHARE = 0.25
DEFAULT_LOW_HYDRATION_AFTER_CARRION = 0.6
_SEED_PATTERN = re.compile(r"(?:seed[-_=]?|mind-v3-|heuristic-)(\d+)")


class TransitionAlignedRecoveryAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _LoadedReport:
    name: str
    path: str | None
    payload: Mapping[str, object] | None
    evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LoadedTrajectories:
    datasets: tuple[TrajectoryJsonlDataset, ...]
    evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LenientTrajectoryLoad:
    dataset: TrajectoryJsonlDataset
    malformed_record_count: int
    malformed_records: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _RecoveryConstruction:
    section: dict[str, object]
    rows: tuple[dict[str, object], ...]
    exact_index: Mapping[tuple[int, int, int, str], tuple[dict[str, object], ...]]
    state_index: Mapping[tuple[int, int, int], tuple[dict[str, object], ...]]
    digest_index: Mapping[tuple[str, str, int, int, str], tuple[dict[str, object], ...]]


@dataclass(frozen=True, slots=True)
class _BranchOverlap:
    section: dict[str, object]
    matched_pairs: tuple[dict[str, object], ...]


def load_transition_aligned_recovery_json(path: str | Path) -> dict[str, object]:
    with _open_input(Path(path)) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TransitionAlignedRecoveryAuditError(
            f"report must be a JSON object: {path}"
        )
    return payload


def write_transition_aligned_recovery_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def build_transition_aligned_recovery_audit_report(
    *,
    v106_report: Mapping[str, object] | None = None,
    v106_report_path: str | Path | None = None,
    recovery_action_target_audit: Mapping[str, object] | None = None,
    recovery_action_target_audit_path: str | Path | None = None,
    branch_oracle_audit: Mapping[str, object] | None = None,
    branch_oracle_audit_path: str | Path | None = None,
    rollout_context_report: Mapping[str, object] | None = None,
    rollout_context_report_path: str | Path | None = None,
    baseline_report: Mapping[str, object] | None = None,
    baseline_report_path: str | Path | None = None,
    trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None = None,
    trajectory_paths: Sequence[str | Path] = (),
    trajectory_glob_patterns: Sequence[str] = (),
    min_overlap_present_share: float = DEFAULT_MIN_OVERLAP_PRESENT_SHARE,
) -> dict[str, object]:
    if not 0.0 <= min_overlap_present_share <= 1.0:
        raise TransitionAlignedRecoveryAuditError(
            "min_overlap_present_share must be in [0.0, 1.0]"
        )
    loaded_reports = {
        "v106_report": _resolve_report("v106_report", v106_report, v106_report_path),
        "recovery_action_target_audit": _resolve_report(
            "recovery_action_target_audit",
            recovery_action_target_audit,
            recovery_action_target_audit_path,
        ),
        "branch_oracle_audit": _resolve_report(
            "branch_oracle_audit",
            branch_oracle_audit,
            branch_oracle_audit_path,
        ),
        "rollout_context_report": _resolve_report(
            "rollout_context_report",
            rollout_context_report,
            rollout_context_report_path,
        ),
        "baseline_report": _resolve_report(
            "baseline_report",
            baseline_report,
            baseline_report_path,
        ),
    }
    trajectories = _resolve_trajectories(
        trajectory_datasets=trajectory_datasets,
        trajectory_paths=trajectory_paths,
        trajectory_glob_patterns=trajectory_glob_patterns,
    )
    evidence = {
        "reports": {
            name: loaded.evidence for name, loaded in sorted(loaded_reports.items())
        },
        "trajectories": trajectories.evidence,
    }
    recovery = _transition_aligned_first_recovery(trajectories.datasets)
    overlap = _branch_oracle_overlap(
        loaded_reports["branch_oracle_audit"].payload,
        recovery,
        min_overlap_present_share=min_overlap_present_share,
    )
    action_support = _action_support_legality(overlap)
    mask_drift = _mask_drift_root_cause(recovery)
    seed29 = _seed29_birth_regression(
        rollout_context_report=loaded_reports["rollout_context_report"].payload,
        baseline_report=loaded_reports["baseline_report"].payload,
        recovery_rows=recovery.rows,
    )
    heldout_support = _heldout_support(
        transition_aligned_first_recovery=recovery.section,
        branch_oracle_overlap=overlap.section,
        action_support_legality=action_support,
    )
    sections = {
        "transition_aligned_first_recovery": recovery.section,
        "branch_oracle_overlap": overlap.section,
        "action_support_legality": action_support,
        "mask_drift_root_cause": mask_drift,
        "seed29_birth_regression": seed29,
        "heldout_support": heldout_support,
    }
    classification = _classification(evidence=evidence, sections=sections)
    research_recommendation = _research_recommendation(
        classification=classification,
        sections=sections,
    )
    contract = {
        "schema_version": MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_POLICY,
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "artifact_promotion_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "gate_effect": "none",
        "fixture_floor_effect": "none",
        "heuristic_action_selection_effect": "none",
        "action_mask_effect": "none",
        "cache_internal_effect": "none",
        "private_world_state_policy": "not_read",
        "fixture_identity_policy": "not_used",
        "animal_resource_gain_detector": "rollout_context_event_from_record",
        "branch_join_policy": (
            "exact_seed_tick_agent_action_then_state_then_available_digest_v1"
        ),
        "missing_evidence_policy": "classify_inconclusive_without_runtime_workaround_v1",
    }
    return {
        "schema_version": MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_POLICY,
        "contract": contract,
        "provenance": {"contract_digest": stable_payload_digest(contract)},
        "evidence": evidence,
        "transition_aligned_first_recovery": recovery.section,
        "branch_oracle_overlap": overlap.section,
        "action_support_legality": action_support,
        "mask_drift_root_cause": mask_drift,
        "seed29_birth_regression": seed29,
        "heldout_support": heldout_support,
        "classification": classification,
        "research_recommendation": research_recommendation,
        "non_promoted": True,
    }


def reconstruct_transition_aligned_first_recovery_rows(
    *,
    trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None = None,
    trajectory_paths: Sequence[str | Path] = (),
    trajectory_glob_patterns: Sequence[str] = (),
) -> dict[str, object]:
    """Reconstruct v107 first-recovery rows without changing v107 report shape."""
    trajectories = _resolve_trajectories(
        trajectory_datasets=trajectory_datasets,
        trajectory_paths=trajectory_paths,
        trajectory_glob_patterns=trajectory_glob_patterns,
    )
    recovery = _transition_aligned_first_recovery(trajectories.datasets)
    return {
        "evidence": trajectories.evidence,
        "section": recovery.section,
        "rows": [dict(row) for row in recovery.rows],
        "path_metadata": {
            str(dataset.path): {
                "seed": _dataset_seed(dataset),
                "max_ticks": _dataset_ticks(dataset),
                "record_count": len(dataset.records),
            }
            for dataset in trajectories.datasets
        },
    }


def _resolve_report(
    name: str,
    payload: Mapping[str, object] | None,
    path: str | Path | None,
) -> _LoadedReport:
    path_string = str(path) if path is not None else None
    if payload is not None:
        return _LoadedReport(
            name=name,
            path=path_string,
            payload=payload,
            evidence=_report_evidence(
                path=path_string,
                payload=payload,
                loaded=True,
                load_error=None,
                source="inline" if path is None else "inline_with_path",
            ),
        )
    if path is None:
        return _LoadedReport(
            name=name,
            path=None,
            payload=None,
            evidence=_report_evidence(
                path=None,
                payload=None,
                loaded=False,
                load_error="not_provided",
                source="missing",
            ),
        )
    try:
        loaded = load_transition_aligned_recovery_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _LoadedReport(
            name=name,
            path=path_string,
            payload=None,
            evidence=_report_evidence(
                path=path_string,
                payload=None,
                loaded=False,
                load_error=f"{type(exc).__name__}: {exc}",
                source="path",
            ),
        )
    return _LoadedReport(
        name=name,
        path=path_string,
        payload=loaded,
        evidence=_report_evidence(
            path=path_string,
            payload=loaded,
            loaded=True,
            load_error=None,
            source="path",
        ),
    )


def _report_evidence(
    *,
    path: str | None,
    payload: Mapping[str, object] | None,
    loaded: bool,
    load_error: str | None,
    source: str,
) -> dict[str, object]:
    return {
        "path": path,
        "source": source,
        "loaded": loaded,
        "load_error": load_error,
        "schema_version": payload.get("schema_version") if payload is not None else None,
        "digest": stable_payload_digest(payload) if payload is not None else None,
    }


def _resolve_trajectories(
    *,
    trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None,
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
) -> _LoadedTrajectories:
    if trajectory_datasets is not None:
        datasets = tuple(trajectory_datasets)
        return _LoadedTrajectories(
            datasets=datasets,
            evidence={
                "source": "inline",
                "requested_paths": [str(path) for path in trajectory_paths],
                "requested_globs": list(trajectory_glob_patterns),
                "loaded_path_count": len(datasets),
                "load_failure_count": 0,
                "load_failures": [],
                "malformed_record_count": 0,
                "malformed_records": [],
                "loaded_paths": [str(dataset.path) for dataset in datasets],
            },
        )
    datasets: list[TrajectoryJsonlDataset] = []
    failures: list[dict[str, object]] = []
    malformed_count = 0
    malformed_records: list[dict[str, object]] = []
    for path in trajectory_paths:
        try:
            loaded = _load_trajectory_jsonl_lenient(path)
            datasets.append(loaded.dataset)
            malformed_count += loaded.malformed_record_count
            malformed_records.extend(loaded.malformed_records)
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return _LoadedTrajectories(
        datasets=tuple(datasets),
        evidence={
            "source": "path",
            "requested_paths": [str(path) for path in trajectory_paths],
            "requested_globs": list(trajectory_glob_patterns),
            "loaded_path_count": len(datasets),
            "load_failure_count": len(failures),
            "load_failures": failures,
            "malformed_record_count": malformed_count,
            "malformed_records": malformed_records[:12],
            "loaded_paths": [str(dataset.path) for dataset in datasets],
        },
    )


def _load_trajectory_jsonl_lenient(path: str | Path) -> _LenientTrajectoryLoad:
    resolved_path = Path(path)
    payloads: list[dict[str, object]] = []
    with _open_input(resolved_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise TransitionAlignedRecoveryAuditError(
                    f"trajectory payload at line {line_number} must be a JSON object"
                )
            payloads.append(payload)
    if len(payloads) < 2:
        raise TransitionAlignedRecoveryAuditError(
            "trajectory JSONL must include header and footer"
        )
    records: list[dict[str, object]] = []
    malformed_records: list[dict[str, object]] = []
    for index, payload in enumerate(payloads[1:-1]):
        try:
            records.append(_trajectory_record_payload(payload, index))
        except TransitionAlignedRecoveryAuditError as exc:
            malformed_records.append(
                {
                    "path": str(resolved_path),
                    "record_index": index,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return _LenientTrajectoryLoad(
        dataset=TrajectoryJsonlDataset(
            path=resolved_path,
            header=dict(payloads[0]),
            records=tuple(records),
            footer=dict(payloads[-1]),
        ),
        malformed_record_count=len(malformed_records),
        malformed_records=tuple(malformed_records),
    )


def _trajectory_record_payload(
    payload: Mapping[str, object],
    index: int,
) -> dict[str, object]:
    record = payload.get("record")
    if isinstance(record, Mapping):
        return dict(record)
    if payload.get("type") == "record":
        return dict(payload)
    raise TransitionAlignedRecoveryAuditError(
        f"trajectory record {index} must be wrapped as a record payload"
    )


def _transition_aligned_first_recovery(
    datasets: Sequence[TrajectoryJsonlDataset],
) -> _RecoveryConstruction:
    rows: list[dict[str, object]] = []
    gain_count = 0
    missing_next = 0
    by_seed_gain_count: Counter[str] = Counter()
    by_seed_row_count: Counter[str] = Counter()
    for dataset_index, dataset in enumerate(datasets):
        seed = _dataset_seed(dataset)
        for episode_index, segment in enumerate(_episode_segments(dataset.records)):
            episode_id = _episode_id(
                dataset,
                dataset_index=dataset_index,
                episode_index=episode_index,
                segment=segment,
            )
            for position, record in segment:
                event = rollout_context_event_from_record(record)
                if not bool(event.get("animal_resource_gain")):
                    continue
                gain_count += 1
                by_seed_gain_count[str(seed)] += 1
                next_item = _first_same_agent_decision_after(
                    segment,
                    start_position=position,
                    agent_id=_int(event.get("agent_id"), default=-1),
                )
                if next_item is None:
                    missing_next += 1
                    continue
                next_position, next_record = next_item
                row = _recovery_row(
                    dataset=dataset,
                    seed=seed,
                    episode_id=episode_id,
                    gain_position=position,
                    gain_record=record,
                    gain_event=event,
                    recovery_position=next_position,
                    recovery_record=next_record,
                )
                rows.append(row)
                by_seed_row_count[str(seed)] += 1
    exact_index: dict[tuple[int, int, int, str], list[dict[str, object]]] = {}
    state_index: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    digest_index: dict[tuple[str, str, int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        seed = _int(row.get("seed"), default=-1)
        tick = _int(row.get("recovery_tick"), default=-1)
        agent_id = _int(row.get("agent_id"), default=-1)
        action = str(row.get("requested_action"))
        exact_index.setdefault((seed, tick, agent_id, action), []).append(row)
        state_index.setdefault((seed, tick, agent_id), []).append(row)
        for field in ("branch_state_digest", "observation_digest"):
            digest = row.get(field)
            if isinstance(digest, str) and digest:
                digest_index.setdefault((field, digest, seed, agent_id, action), []).append(
                    row
                )
    answer = _transition_constructibility_answer(
        trajectory_count=len(datasets),
        gain_count=gain_count,
        row_count=len(rows),
    )
    section = {
        "answer": answer,
        "allowed_answers": [
            "transition_aligned_first_recovery_constructible",
            "transition_aligned_first_recovery_not_constructible",
            "transition_aligned_first_recovery_inconclusive",
        ],
        "trajectory_count": len(datasets),
        "animal_resource_gain_record_count": gain_count,
        "constructible_first_recovery_row_count": len(rows),
        "missing_next_same_agent_decision_count": missing_next,
        "by_seed": {
            seed: {
                "animal_resource_gain_record_count": by_seed_gain_count.get(seed, 0),
                "constructible_first_recovery_row_count": by_seed_row_count.get(
                    seed,
                    0,
                ),
            }
            for seed in sorted(set(by_seed_gain_count) | set(by_seed_row_count))
        },
        "examples": [_row_example(row) for row in rows[:12]],
    }
    return _RecoveryConstruction(
        section=section,
        rows=tuple(rows),
        exact_index={key: tuple(value) for key, value in sorted(exact_index.items())},
        state_index={key: tuple(value) for key, value in sorted(state_index.items())},
        digest_index={key: tuple(value) for key, value in sorted(digest_index.items())},
    )


def _episode_segments(
    records: Sequence[dict[str, object]],
) -> tuple[tuple[tuple[int, dict[str, object]], ...], ...]:
    segments: list[tuple[tuple[int, dict[str, object]], ...]] = []
    current: list[tuple[int, dict[str, object]]] = []
    previous_tick: int | None = None
    previous_episode_id: str | None = None
    for index, record in enumerate(records):
        tick = _int(record.get("tick"), default=0)
        episode_id = _optional_string(record.get(TRAJECTORY_EPISODE_ID_FIELD))
        starts_new = False
        if current:
            if episode_id is not None or previous_episode_id is not None:
                starts_new = episode_id != previous_episode_id
            elif previous_tick is not None:
                starts_new = tick < previous_tick
        if starts_new:
            segments.append(tuple(current))
            current = []
        current.append((index, record))
        previous_tick = tick
        previous_episode_id = episode_id
    if current:
        segments.append(tuple(current))
    return tuple(segments)


def _episode_id(
    dataset: TrajectoryJsonlDataset,
    *,
    dataset_index: int,
    episode_index: int,
    segment: Sequence[tuple[int, Mapping[str, object]]],
) -> str:
    for _, record in segment:
        episode_id = _optional_string(record.get(TRAJECTORY_EPISODE_ID_FIELD))
        if episode_id is not None:
            return episode_id
    return f"{dataset_index}:{dataset.path}:episode={episode_index}"


def _first_same_agent_decision_after(
    segment: Sequence[tuple[int, dict[str, object]]],
    *,
    start_position: int,
    agent_id: int,
) -> tuple[int, dict[str, object]] | None:
    for position, record in segment:
        if position <= start_position:
            continue
        if _int(record.get("agent_id"), default=-1) != agent_id:
            continue
        if str(record.get("action_source", "")) == "passive":
            continue
        return position, record
    return None


def _recovery_row(
    *,
    dataset: TrajectoryJsonlDataset,
    seed: int,
    episode_id: str,
    gain_position: int,
    gain_record: Mapping[str, object],
    gain_event: Mapping[str, object],
    recovery_position: int,
    recovery_record: Mapping[str, object],
) -> dict[str, object]:
    requested = str(recovery_record.get("requested_action", "unknown"))
    action_mask = _bool_mapping(recovery_record.get("action_mask"))
    resolution_mask = _bool_mapping(recovery_record.get("resolution_action_mask"))
    observation_legal = bool(action_mask.get(requested, False))
    resolution_legal = bool(resolution_mask.get(requested, False))
    outcome = _mapping(recovery_record.get("outcome"))
    invalid_reason = recovery_record.get("invalid_reason")
    if invalid_reason is None:
        invalid_reason = outcome.get("invalid_reason")
    before = _state_excerpt(recovery_record.get("before"))
    after = _state_excerpt(recovery_record.get("after"))
    row = {
        "path": str(dataset.path),
        "seed": seed,
        "episode_id": episode_id,
        "agent_id": _int(recovery_record.get("agent_id"), default=-1),
        "gain_record_index": gain_position,
        "gain_tick": gain_event.get("tick"),
        "gain_requested_action": gain_event.get("requested_action"),
        "gain_resolved_action": gain_event.get("resolved_action"),
        "gain_resource_gain": gain_event.get("resource_gain"),
        "recovery_record_index": recovery_position,
        "recovery_tick": recovery_record.get("tick"),
        "ticks_after_gain": _delta_ticks(
            recovery_record.get("tick"),
            gain_record.get("tick"),
        ),
        "requested_action": requested,
        "resolved_action": str(recovery_record.get("resolved_action", "unknown")),
        "action_source": str(recovery_record.get("action_source", "unknown")),
        "action_valid": bool(recovery_record.get("action_valid", observation_legal)),
        "resolution_action_valid": bool(
            recovery_record.get("resolution_action_valid", resolution_legal)
        ),
        "requested_action_observation_legal": observation_legal,
        "requested_action_resolution_legal": resolution_legal,
        "invalid_reason": invalid_reason,
        "action_mask": action_mask,
        "resolution_action_mask": resolution_mask,
        "observation_digest": _optional_string(recovery_record.get("observation_digest")),
        "branch_state_digest": _optional_string(recovery_record.get("branch_state_digest")),
        "before": before,
        "after": after,
        "outcome": {
            "moved": bool(
                recovery_record.get("moved")
                if isinstance(recovery_record.get("moved"), bool)
                else _mapping(outcome.get("movement")).get("moved", False)
            ),
            "drank": bool(_mapping(outcome.get("drinking")).get("drank", False)),
            "ate": bool(_mapping(outcome.get("feeding")).get("ate", False)),
            "resource_gain": _number(outcome.get("resource_gain")),
            "reproduced": bool(outcome.get("reproduced", False)),
            "reproduction_ready_after": bool(
                outcome.get("reproduction_ready_after", False)
            ),
            "died": bool(outcome.get("died", False)),
        },
    }
    row["mask_drift_public_cause"] = _public_mask_drift_cause(
        action=requested,
        observation_legal=observation_legal,
        resolution_legal=resolution_legal,
    )
    return row


def _branch_oracle_overlap(
    branch_oracle_audit: Mapping[str, object] | None,
    recovery: _RecoveryConstruction,
    *,
    min_overlap_present_share: float,
) -> _BranchOverlap:
    if branch_oracle_audit is None:
        return _BranchOverlap(
            section={
                "answer": "missing_evidence_inconclusive",
                "allowed_answers": [
                    "branch_oracle_first_recovery_overlap_present",
                    "branch_oracle_first_recovery_overlap_sparse",
                    "branch_oracle_first_recovery_overlap_absent",
                    "missing_evidence_inconclusive",
                ],
                "loaded": False,
                "branch_result_count": 0,
                "matched_branch_result_count": 0,
                "diagnostics_only": True,
            },
            matched_pairs=(),
        )
    branch_results = [
        result
        for result in _list(branch_oracle_audit.get("branch_results"))
        if isinstance(result, Mapping)
    ]
    exact_count = 0
    state_count = 0
    digest_count = 0
    matched_count = 0
    matched_pairs: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    for result in branch_results:
        seed = _int(result.get("seed"), default=-1)
        branch_tick = _int(result.get("branch_tick"), default=-1)
        agent_id = _int(result.get("agent_id"), default=-1)
        logged_action = str(result.get("logged_action", ""))
        exact_rows = tuple(
            recovery.exact_index.get((seed, branch_tick, agent_id, logged_action), ())
        )
        state_rows = tuple(recovery.state_index.get((seed, branch_tick, agent_id), ()))
        digest_rows = _digest_matches(
            result,
            recovery.digest_index,
            seed=seed,
            agent_id=agent_id,
            logged_action=logged_action,
        )
        if exact_rows:
            exact_count += 1
        if state_rows:
            state_count += 1
        if digest_rows:
            digest_count += 1
        chosen_rows, match_policy = _chosen_branch_matches(
            exact_rows=exact_rows,
            state_rows=state_rows,
            digest_rows=digest_rows,
        )
        if chosen_rows:
            matched_count += 1
        for row in chosen_rows:
            pair = {
                "match_policy": match_policy,
                "branch": _branch_excerpt(result),
                "recovery_row": _row_example(row),
            }
            matched_pairs.append(pair)
        if len(examples) < 12 and chosen_rows:
            examples.append(
                {
                    "match_policy": match_policy,
                    "branch": _branch_excerpt(result),
                    "recovery_rows": [_row_example(row) for row in chosen_rows[:4]],
                }
            )
    matched_share = _share(matched_count, len(branch_results))
    if matched_count <= 0:
        answer = "branch_oracle_first_recovery_overlap_absent"
    elif matched_share is not None and matched_share >= min_overlap_present_share:
        answer = "branch_oracle_first_recovery_overlap_present"
    else:
        answer = "branch_oracle_first_recovery_overlap_sparse"
    return _BranchOverlap(
        section={
            "answer": answer,
            "allowed_answers": [
                "branch_oracle_first_recovery_overlap_present",
                "branch_oracle_first_recovery_overlap_sparse",
                "branch_oracle_first_recovery_overlap_absent",
                "missing_evidence_inconclusive",
            ],
            "loaded": True,
            "schema_version": branch_oracle_audit.get("schema_version"),
            "branch_result_count": len(branch_results),
            "matched_branch_result_count": matched_count,
            "matched_branch_result_share": matched_share,
            "exact_join_branch_result_count": exact_count,
            "state_join_branch_result_count": state_count,
            "digest_join_branch_result_count": digest_count,
            "min_overlap_present_share": min_overlap_present_share,
            "matched_pair_count": len(matched_pairs),
            "examples": examples,
            "diagnostics_only": True,
        },
        matched_pairs=tuple(matched_pairs),
    )


def _digest_matches(
    result: Mapping[str, object],
    digest_index: Mapping[tuple[str, str, int, int, str], tuple[dict[str, object], ...]],
    *,
    seed: int,
    agent_id: int,
    logged_action: str,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for field in ("branch_state_digest", "observation_digest"):
        digest = result.get(field)
        if not isinstance(digest, str) or not digest:
            continue
        for row in digest_index.get((field, digest, seed, agent_id, logged_action), ()):
            key = (str(row.get("path")), _int(row.get("recovery_record_index")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return tuple(rows)


def _chosen_branch_matches(
    *,
    exact_rows: Sequence[dict[str, object]],
    state_rows: Sequence[dict[str, object]],
    digest_rows: Sequence[dict[str, object]],
) -> tuple[tuple[dict[str, object], ...], str]:
    if exact_rows:
        return tuple(exact_rows), "exact_seed_tick_agent_action"
    if digest_rows:
        return tuple(digest_rows), "available_digest"
    if state_rows:
        return tuple(state_rows), "state_seed_tick_agent"
    return (), "none"


def _action_support_legality(overlap: _BranchOverlap) -> dict[str, object]:
    counts = Counter()
    examples: list[dict[str, object]] = []
    for pair in overlap.matched_pairs:
        branch = _mapping(pair.get("branch"))
        row = _mapping(pair.get("recovery_row"))
        action = str(branch.get("oracle_best_action") or row.get("requested_action"))
        action_mask = _bool_mapping(row.get("action_mask"))
        resolution_mask = _bool_mapping(row.get("resolution_action_mask"))
        observation_legal = bool(action_mask.get(action, False))
        resolution_legal = bool(resolution_mask.get(action, False))
        if observation_legal and resolution_legal:
            category = "supported_recovery_actions_observation_legal_resolution_legal"
        elif observation_legal:
            category = "supported_recovery_actions_observation_legal_resolution_invalid"
        else:
            category = "supported_recovery_actions_not_observation_legal"
        counts[category] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "match_policy": pair.get("match_policy"),
                    "seed": row.get("seed"),
                    "tick": row.get("recovery_tick"),
                    "agent_id": row.get("agent_id"),
                    "logged_action": branch.get("logged_action"),
                    "oracle_best_action": branch.get("oracle_best_action"),
                    "legality_category": category,
                    "observation_legal": observation_legal,
                    "resolution_legal": resolution_legal,
                }
            )
    if counts["supported_recovery_actions_observation_legal_resolution_invalid"] > 0:
        answer = "supported_recovery_actions_observation_legal_resolution_invalid"
    elif counts["supported_recovery_actions_observation_legal_resolution_legal"] > 0:
        answer = "supported_recovery_actions_observation_legal_resolution_legal"
    elif counts["supported_recovery_actions_not_observation_legal"] > 0:
        answer = "supported_recovery_actions_not_observation_legal"
    else:
        answer = "missing_evidence_inconclusive"
    return {
        "answer": answer,
        "allowed_answers": [
            "supported_recovery_actions_observation_legal_resolution_legal",
            "supported_recovery_actions_observation_legal_resolution_invalid",
            "supported_recovery_actions_not_observation_legal",
            "missing_evidence_inconclusive",
        ],
        "matched_pair_count": len(overlap.matched_pairs),
        "category_counts": _counter_to_ordered_dict(counts),
        "examples": examples,
        "diagnostics_only": True,
    }


def _mask_drift_root_cause(recovery: _RecoveryConstruction) -> dict[str, object]:
    counts = Counter()
    examples: list[dict[str, object]] = []
    for row in recovery.rows:
        cause = str(row.get("mask_drift_public_cause", "none"))
        if cause == "none":
            continue
        counts[cause] += 1
        if len(examples) < 12:
            examples.append(_row_example(row))
    if counts["movement_mask_drift_occupancy_race"] > 0:
        answer = "movement_mask_drift_occupancy_race"
    elif counts["movement_mask_drift_blocked_route"] > 0:
        answer = "movement_mask_drift_blocked_route"
    elif counts["movement_mask_drift_depleted_resource"] > 0:
        answer = "movement_mask_drift_depleted_resource"
    elif counts["movement_mask_drift_other_public_resolution_path"] > 0:
        answer = "movement_mask_drift_other_public_resolution_path"
    else:
        answer = "movement_mask_drift_inconclusive"
    return {
        "answer": answer,
        "allowed_answers": [
            "movement_mask_drift_occupancy_race",
            "movement_mask_drift_blocked_route",
            "movement_mask_drift_depleted_resource",
            "movement_mask_drift_other_public_resolution_path",
            "movement_mask_drift_inconclusive",
        ],
        "drift_row_count": sum(counts.values()),
        "category_counts": _counter_to_ordered_dict(counts),
        "examples": examples,
        "diagnostics_only": True,
    }


def _seed29_birth_regression(
    *,
    rollout_context_report: Mapping[str, object] | None,
    baseline_report: Mapping[str, object] | None,
    recovery_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidate = _seed_run(rollout_context_report, seed=29)
    baseline = _seed_run(baseline_report, seed=29)
    births_delta = _delta(candidate.get("births"), baseline.get("births"))
    unsupported_delta = _delta(
        candidate.get("unsupported_resolved_action_count"),
        baseline.get("unsupported_resolved_action_count"),
    )
    seed_rows = [row for row in recovery_rows if _int(row.get("seed")) == 29]
    movement_drift_count = sum(
        1
        for row in seed_rows
        if row.get("mask_drift_public_cause")
        in ("movement_mask_drift_occupancy_race", "movement_mask_drift_blocked_route")
    )
    unsupported_count = sum(
        1 for row in seed_rows if row.get("resolution_action_valid") is False
    )
    missed_drink_count = sum(1 for row in seed_rows if _missed_drink_after_carrion(row))
    delayed_repro_count = sum(
        1
        for row in seed_rows
        if _mapping(row.get("outcome")).get("reproduction_ready_after") is False
    )
    if births_delta is None or births_delta >= 0.0:
        answer = "seed29_birth_regression_inconclusive_public_evidence_gap"
    elif movement_drift_count > 0:
        answer = "seed29_birth_regression_movement_failure_after_carrion"
    elif (unsupported_delta is not None and unsupported_delta > 0.0) or unsupported_count > 0:
        answer = "seed29_birth_regression_unsupported_resolution"
    elif missed_drink_count > 0:
        answer = "seed29_birth_regression_missed_drink_after_carrion"
    elif delayed_repro_count > 0:
        answer = "seed29_birth_regression_delayed_reproduction_readiness"
    else:
        answer = "seed29_birth_regression_inconclusive_public_evidence_gap"
    return {
        "answer": answer,
        "allowed_answers": [
            "seed29_birth_regression_movement_failure_after_carrion",
            "seed29_birth_regression_unsupported_resolution",
            "seed29_birth_regression_missed_drink_after_carrion",
            "seed29_birth_regression_delayed_reproduction_readiness",
            "seed29_birth_regression_inconclusive_public_evidence_gap",
        ],
        "seed": 29,
        "births_delta": births_delta,
        "alive_agents_delta": _delta(
            candidate.get("alive_agents"),
            baseline.get("alive_agents"),
        ),
        "movement_event_rate_delta": _delta(
            candidate.get("movement_event_rate"),
            baseline.get("movement_event_rate"),
        ),
        "unsupported_resolved_action_delta": unsupported_delta,
        "transition_aligned_first_recovery_row_count": len(seed_rows),
        "movement_drift_after_carrion_count": movement_drift_count,
        "unsupported_resolution_after_carrion_count": unsupported_count,
        "missed_drink_after_carrion_count": missed_drink_count,
        "delayed_reproduction_readiness_public_row_count": delayed_repro_count,
        "examples": [_row_example(row) for row in seed_rows[:12]],
    }


def _heldout_support(
    *,
    transition_aligned_first_recovery: Mapping[str, object],
    branch_oracle_overlap: Mapping[str, object],
    action_support_legality: Mapping[str, object],
) -> dict[str, object]:
    constructible = (
        transition_aligned_first_recovery.get("answer")
        == "transition_aligned_first_recovery_constructible"
    )
    overlap_present = (
        branch_oracle_overlap.get("answer")
        == "branch_oracle_first_recovery_overlap_present"
    )
    legal_counts = _mapping(action_support_legality.get("category_counts"))
    legal_supported_count = _int(
        legal_counts.get("supported_recovery_actions_observation_legal_resolution_legal")
    )
    resolution_invalid_count = _int(
        legal_counts.get(
            "supported_recovery_actions_observation_legal_resolution_invalid"
        )
    )
    sufficient = (
        constructible
        and overlap_present
        and legal_supported_count > 0
        and resolution_invalid_count == 0
    )
    answer = (
        "heldout_support_sufficient_for_v108"
        if sufficient
        else "heldout_support_insufficient_for_v108"
    )
    return {
        "answer": answer,
        "allowed_answers": [
            "heldout_support_sufficient_for_v108",
            "heldout_support_insufficient_for_v108",
        ],
        "constructible": constructible,
        "overlap_present": overlap_present,
        "legal_supported_recovery_action_count": legal_supported_count,
        "resolution_invalid_supported_recovery_action_count": resolution_invalid_count,
        "diagnostics_only": True,
    }


def _classification(
    *,
    evidence: Mapping[str, object],
    sections: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    scores = {label: 0 for label in ALLOWED_CLASSIFICATION_LABELS}
    missing = _missing_evidence(evidence)
    answers = {
        name: str(section.get("answer"))
        for name, section in sorted(sections.items())
        if isinstance(section.get("answer"), str)
    }
    if missing:
        scores["missing_evidence_inconclusive"] = 100
        return {
            "primary": "missing_evidence_inconclusive",
            "labels": ["missing_evidence_inconclusive"],
            "category_scores": scores,
            "evidence": [],
            "diagnostic_answers": answers,
            "missing_evidence": missing,
            "diagnostics_only": True,
        }
    weights = {
        "transition_aligned_first_recovery": 30,
        "branch_oracle_overlap": 24,
        "action_support_legality": 20,
        "mask_drift_root_cause": 18,
        "seed29_birth_regression": 12,
        "heldout_support": 10,
    }
    for section_name, answer in answers.items():
        if answer in scores:
            scores[answer] += weights.get(section_name, 1)
    positive = [label for label, score in scores.items() if score > 0]
    if not positive:
        scores["no_material_blocker_detected"] = 1
        positive = ["no_material_blocker_detected"]
    primary = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
    labels = [label for label in positive if label in CLASSIFICATION_LABEL_SET]
    return {
        "primary": primary,
        "labels": labels,
        "category_scores": scores,
        "evidence": [answer for answer in answers.values() if answer in scores],
        "diagnostic_answers": answers,
        "missing_evidence": missing,
        "diagnostics_only": True,
    }


def _research_recommendation(
    *,
    classification: Mapping[str, object],
    sections: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    support = _mapping(sections.get("heldout_support"))
    sufficient = support.get("answer") == "heldout_support_sufficient_for_v108"
    return {
        "classification_primary": classification.get("primary"),
        "diagnostic_answers": {
            name: section.get("answer") for name, section in sorted(sections.items())
        },
        "recommendation": (
            "public_transition_alignment_supports_v108_design_probe"
            if sufficient
            else "do_not_promote; collect_or_align_more_first_recovery_support_before_v108"
        ),
        "v108_ready_from_public_support": sufficient,
        "promote_v107": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_relaxation_recommended": False,
    }


def _transition_constructibility_answer(
    *,
    trajectory_count: int,
    gain_count: int,
    row_count: int,
) -> str:
    if trajectory_count <= 0:
        return "transition_aligned_first_recovery_inconclusive"
    if row_count > 0:
        return "transition_aligned_first_recovery_constructible"
    if gain_count > 0:
        return "transition_aligned_first_recovery_inconclusive"
    return "transition_aligned_first_recovery_not_constructible"


def _public_mask_drift_cause(
    *,
    action: str,
    observation_legal: bool,
    resolution_legal: bool,
) -> str:
    if resolution_legal:
        return "none"
    if action in MOVEMENT_ACTIONS or action.startswith("move_"):
        if observation_legal:
            return "movement_mask_drift_occupancy_race"
        return "movement_mask_drift_blocked_route"
    if action in ("eat", "drink") and observation_legal:
        return "movement_mask_drift_depleted_resource"
    return "movement_mask_drift_other_public_resolution_path"


def _missed_drink_after_carrion(row: Mapping[str, object]) -> bool:
    before = _mapping(row.get("before"))
    hydration = _number_or_none(before.get("hydration_ratio"))
    if hydration is None or hydration >= DEFAULT_LOW_HYDRATION_AFTER_CARRION:
        return False
    if str(row.get("requested_action")) == "drink":
        return False
    return bool(_bool_mapping(row.get("action_mask")).get("drink", False))


def _missing_evidence(evidence: Mapping[str, object]) -> list[str]:
    missing: list[str] = []
    reports = _mapping(evidence.get("reports"))
    for name, payload in sorted(reports.items()):
        if not bool(_mapping(payload).get("loaded")):
            missing.append(str(name))
    trajectories = _mapping(evidence.get("trajectories"))
    if _int(trajectories.get("loaded_path_count")) <= 0:
        missing.append("trajectories")
    if _int(trajectories.get("load_failure_count")) > 0:
        missing.append("trajectory_load_failures")
    return missing


def _seed_run(
    report: Mapping[str, object] | None,
    *,
    seed: int,
) -> Mapping[str, object]:
    holdout = _mapping(_mapping(report or {}).get("holdout_evaluation"))
    for run in _list(holdout.get("runs")):
        if isinstance(run, Mapping) and _int(run.get("seed"), default=-1) == seed:
            return run
    return {}


def _branch_excerpt(result: Mapping[str, object]) -> dict[str, object]:
    payload = {
        "branch_id": result.get("branch_id"),
        "seed": result.get("seed"),
        "branch_tick": result.get("branch_tick"),
        "agent_id": result.get("agent_id"),
        "logged_action": result.get("logged_action"),
        "oracle_best_action": result.get("oracle_best_action"),
        "branch_state_digest": result.get("branch_state_digest"),
    }
    observation_digest = result.get("observation_digest")
    if isinstance(observation_digest, str) and observation_digest:
        payload["observation_digest"] = observation_digest
    return payload


def _row_example(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": row.get("path"),
        "seed": row.get("seed"),
        "episode_id": row.get("episode_id"),
        "agent_id": row.get("agent_id"),
        "gain_record_index": row.get("gain_record_index"),
        "gain_tick": row.get("gain_tick"),
        "recovery_record_index": row.get("recovery_record_index"),
        "recovery_tick": row.get("recovery_tick"),
        "requested_action": row.get("requested_action"),
        "resolved_action": row.get("resolved_action"),
        "action_valid": row.get("action_valid"),
        "resolution_action_valid": row.get("resolution_action_valid"),
        "requested_action_observation_legal": row.get(
            "requested_action_observation_legal"
        ),
        "requested_action_resolution_legal": row.get(
            "requested_action_resolution_legal"
        ),
        "invalid_reason": row.get("invalid_reason"),
        "observation_digest": row.get("observation_digest"),
        "branch_state_digest": row.get("branch_state_digest"),
        "mask_drift_public_cause": row.get("mask_drift_public_cause"),
        "before": row.get("before"),
        "after": row.get("after"),
        "outcome": row.get("outcome"),
        "action_mask": row.get("action_mask"),
        "resolution_action_mask": row.get("resolution_action_mask"),
    }


def _state_excerpt(value: object) -> dict[str, object]:
    state = _mapping(value)
    return {
        key: state.get(key)
        for key in (
            "x",
            "y",
            "alive",
            "energy_ratio",
            "hydration_ratio",
            "health_ratio",
            "age",
        )
        if key in state
    }


def _dataset_seed(dataset: TrajectoryJsonlDataset) -> int:
    header_config = _mapping(dataset.header.get("config"))
    footer_summary = _mapping(dataset.footer.get("summary"))
    for value in (header_config.get("seed"), footer_summary.get("seed")):
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    match = _SEED_PATTERN.search(str(dataset.path))
    if match is not None:
        return int(match.group(1))
    return -1


def _dataset_ticks(dataset: TrajectoryJsonlDataset) -> int:
    header_config = _mapping(dataset.header.get("config"))
    footer_summary = _mapping(dataset.footer.get("summary"))
    for value in (header_config.get("max_ticks"), footer_summary.get("ticks_executed")):
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return 0


def _delta(candidate: object, baseline: object) -> float | None:
    candidate_number = _number_or_none(candidate)
    baseline_number = _number_or_none(baseline)
    if candidate_number is None or baseline_number is None:
        return None
    return _round(candidate_number - baseline_number)


def _delta_ticks(current: object, previous: object) -> int | None:
    current_int = _int_or_none(current)
    previous_int = _int_or_none(previous)
    if current_int is None or previous_int is None:
        return None
    return current_int - previous_int


def _counter_to_ordered_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _bool_mapping(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): bool(item) for key, item in value.items()}


def _share(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _round(numerator / float(denominator))


def _number(value: object) -> float:
    number = _number_or_none(value)
    return number if number is not None else 0.0


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return int(value)


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _round(value: float) -> float:
    return round(float(value), 6)


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
