from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES, MOVEMENT_ACTIONS
from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
)
from evolution_sim.mind.first_recovery_candidate_ranker_capacity_audit import (
    _list_like,
    _ratio,
)
from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    _resolve_jsonl_rows,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _mapping,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_refreshed_surface_blocker_slice_audit import (
    DEFAULT_DETAIL_ROWS_OUTPUT_PATH as DEFAULT_V132_DETAIL_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V132_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recovery_context import (
    recovery_context_feature_contract,
    recovery_context_values,
    recovery_context_vector_fields,
)
from evolution_sim.mind.rollout_context import (
    RolloutContextState,
    rollout_context_feature_contract,
    rollout_context_vector_fields,
)

MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_public_rollout_history_context_audit_v1"
)
MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_ROW_SCHEMA_VERSION = (
    "mind_v3_first_recovery_public_rollout_history_context_row_v1"
)
MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_POLICY = (
    "diagnostics_only_first_recovery_v133_public_rollout_history_context_feasibility_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v133-first-recovery-public-rollout-history-context-audit.json"
)
DEFAULT_HISTORY_ROWS_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v133-first-recovery-public-rollout-history-context-rows.jsonl"
)

REQUIRED_V132_RECOMMENDATION = "add_public_rollout_history_context"
RECOMMENDATIONS: tuple[str, ...] = (
    "public_rollout_history_context_ready_for_refreshed_surface",
    "collect_missing_public_trajectory_history",
    "history_context_present_but_not_discriminative",
    "history_context_leakage_or_integrity_failed",
    "stop_ranker_path_insufficient_public_history_signal",
)
CLASSIFICATIONS: tuple[str, ...] = (
    "public_rollout_history_context_source_integrity_failed",
    "public_rollout_history_context_public_inputs_missing",
    "public_rollout_history_context_ready_for_refreshed_surface",
    "public_rollout_history_context_present_but_not_discriminative",
    "public_rollout_history_context_insufficient_signal",
)
FORBIDDEN_FEATURE_PATH_PARTS = {
    "archive",
    "archive_id",
    "archive_row_id",
    "branch",
    "branch_id",
    "digest",
    "fixture",
    "fold",
    "material_gain_label",
    "metadata",
    "objective",
    "oracle",
    "private",
    "private_world_state",
    "provenance",
    "repaired",
    "repaired_action",
    "replay",
    "seed",
    "source",
    "source_path",
    "split",
    "world",
}
FORBIDDEN_FEATURE_PATH_SUBSTRINGS = (
    "first_action_outcome",
    "post_decision",
    "replay_verification",
    "resolution_action_mask",
    "resolution_legal",
)


@dataclass(frozen=True, slots=True)
class FirstRecoveryPublicRolloutHistoryContextAuditBuild:
    report: dict[str, object]
    history_rows: tuple[dict[str, object], ...]


def build_first_recovery_public_rollout_history_context_audit(
    *,
    v132_report: Mapping[str, object] | None = None,
    v132_report_path: str | Path | None = DEFAULT_V132_REPORT_PATH,
    v132_detail_rows: Sequence[Mapping[str, object]] | None = None,
    v132_detail_rows_path: str | Path | None = DEFAULT_V132_DETAIL_ROWS_PATH,
    v124_manifest_rows: Sequence[Mapping[str, object]] | None = None,
    v124_manifest_path: str | Path | None = DEFAULT_V124_MANIFEST_PATH,
    trajectory_records_by_path: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> FirstRecoveryPublicRolloutHistoryContextAuditBuild:
    v132_payload, v132_evidence = _resolve_json_report(
        v132_report,
        v132_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_SCHEMA_VERSION
        ),
    )
    v132_rows, v132_rows_evidence = _resolve_jsonl_rows(
        v132_detail_rows,
        v132_detail_rows_path,
        row_kind="v132_blocker_slice_rows",
    )
    manifest_rows, manifest_evidence = _resolve_jsonl_rows(
        v124_manifest_rows,
        v124_manifest_path,
        row_kind="v124_manifest",
    )
    source_reports = {
        "v132_report": v132_evidence,
        "v132_detail_rows": v132_rows_evidence,
        "v124_manifest": manifest_evidence,
    }
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v132_report=v132_payload,
        v132_detail_rows=v132_rows,
        manifest_rows=manifest_rows,
    )
    blocker_branches = _blocker_branches(
        v132_detail_rows=v132_rows,
        manifest_rows=manifest_rows,
    )
    history_rows, trajectory_evidence = _history_rows(
        blocker_branches,
        trajectory_records_by_path=trajectory_records_by_path,
    )
    availability = _public_history_availability(
        rows=history_rows,
        trajectory_evidence=trajectory_evidence,
    )
    feature_surface = _feature_surface(history_rows)
    variance = _within_branch_history_variance(history_rows)
    contrast = _failed_vs_correct_history_contrast(history_rows)
    open_analysis = _open_fixture_collapse_analysis(history_rows, v132_payload or {})
    heldout_analysis = _heldout_failure_analysis(history_rows)
    leakage = _leakage_audit(history_rows)
    recommendation = _recommendation(
        source_integrity=source_integrity,
        availability=availability,
        leakage=leakage,
        open_analysis=open_analysis,
        contrast=contrast,
        feature_surface=feature_surface,
    )
    classification = _classification(
        source_integrity=source_integrity,
        availability=availability,
        leakage=leakage,
        recommendation=recommendation,
    )
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "v132_blocker_contract": _v132_blocker_contract(v132_payload or {}, history_rows),
        "public_history_availability": availability,
        "extracted_history_feature_surface": feature_surface,
        "within_branch_history_variance": variance,
        "failed_vs_correct_history_contrast": contrast,
        "open_fixture_collapse_analysis": open_analysis,
        "heldout_failure_analysis": heldout_analysis,
        "leakage_audit": leakage,
        "recommendation": recommendation,
        "classification": classification,
        "authorization_block": _authorization_block(),
        "history_rows_summary": {
            "row_count": len(history_rows),
            "default_output_path": str(DEFAULT_HISTORY_ROWS_OUTPUT_PATH),
            "runtime_loadable": False,
        },
        "non_promoted": True,
    }
    return FirstRecoveryPublicRolloutHistoryContextAuditBuild(
        report=report,
        history_rows=tuple(history_rows),
    )


def write_first_recovery_public_rollout_history_context_audit_report(
    build: FirstRecoveryPublicRolloutHistoryContextAuditBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_first_recovery_public_rollout_history_context_rows(
    build: FirstRecoveryPublicRolloutHistoryContextAuditBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in build.history_rows:
            json.dump(row, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "report_only_public_rollout_history_context_feasibility": True,
        "training_executed": False,
        "shadow_scorer_effect": "none",
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "runtime_policy_effect": "none",
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "foundation_effect": "none",
        "runtime_loadable_artifact_created": False,
        "claim_causality": False,
    }


def _source_integrity(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    v132_report: Mapping[str, object] | None,
    v132_detail_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")
    report = _mapping(v132_report or {})
    source = _mapping(report.get("source_integrity"))
    recommendation = _mapping(report.get("recommendation"))
    rows_summary = _mapping(report.get("detail_rows_summary"))
    if source.get("passed") is not True:
        failures.append("v132_source_integrity_not_passed")
    if recommendation.get("next_step") != REQUIRED_V132_RECOMMENDATION:
        failures.append("v132_recommendation_not_public_rollout_history_context")
    if _int(rows_summary.get("row_count")) != len(v132_detail_rows):
        failures.append("v132_detail_row_count_mismatch")
    max_ordinal = max(
        (_int(row.get("candidate_set_ordinal"), default=-1) for row in v132_detail_rows),
        default=-1,
    )
    if max_ordinal >= len(manifest_rows):
        failures.append("v132_candidate_set_ordinal_missing_from_v124_manifest")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v132_source_integrity_passed": source.get("passed"),
        "v132_recommendation": recommendation.get("next_step"),
        "required_v132_recommendation": REQUIRED_V132_RECOMMENDATION,
        "v132_detail_row_count": len(v132_detail_rows),
        "v124_manifest_row_count": len(manifest_rows),
        "v132_detail_rows_digest": stable_payload_digest(list(v132_detail_rows)),
    }


def _v132_blocker_contract(
    v132_report: Mapping[str, object],
    history_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    aggregate = _mapping(v132_report.get("blocker_class_aggregate"))
    fixture_failures = _mapping(aggregate.get("fixture_open_failures"))
    dominant = _mapping(fixture_failures.get("dominant_predicted_action"))
    fixture_collapse = [
        row for row in history_rows if row.get("blocker_class") == "fixture_open_action_collapse"
    ]
    failed_heldout = [
        row
        for row in history_rows
        if row.get("split") in {"validation", "test"}
        and row.get("is_failed_branch") is True
    ]
    overlap = [
        row
        for row in history_rows
        if row.get("fixture_group") == "open_mind_v3"
        and row.get("split") in {"validation", "test"}
    ]
    return {
        "required_recommendation": REQUIRED_V132_RECOMMENDATION,
        "v132_recommendation": _mapping(v132_report.get("recommendation")).get(
            "next_step"
        ),
        "v132_blocker_class_counts": dict(
            _mapping(aggregate.get("counts_by_blocker_class"))
        ),
        "open_fixture_dominant_predicted_action": dict(dominant),
        "fixture_open_action_collapse_row_count": len(fixture_collapse),
        "failed_heldout_row_count": len(failed_heldout),
        "open_heldout_overlap_row_count": len(overlap),
        "claim_causality": False,
    }


def _blocker_branches(
    *,
    v132_detail_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    sorted_manifest = tuple(sorted(manifest_rows, key=lambda row: str(row.get("branch_id"))))
    branches: list[dict[str, object]] = []
    for row in v132_detail_rows:
        ordinal = _int(row.get("candidate_set_ordinal"), default=-1)
        manifest = sorted_manifest[ordinal] if 0 <= ordinal < len(sorted_manifest) else {}
        branches.append(
            {
                "v132_row": dict(row),
                "manifest": dict(manifest),
                "candidate_set_ordinal": ordinal,
            }
        )
    return tuple(branches)


def _history_rows(
    blocker_branches: Sequence[Mapping[str, object]],
    *,
    trajectory_records_by_path: Mapping[str, Sequence[Mapping[str, object]]] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    cache: dict[str, tuple[tuple[dict[str, object], ...], dict[str, object]]] = {}
    for branch in blocker_branches:
        manifest = _mapping(branch.get("manifest"))
        detail = _mapping(branch.get("v132_row"))
        metadata = _mapping(manifest.get("non_trainable_audit_metadata"))
        source_path = _string(metadata.get("source_path"))
        if source_path not in cache:
            cache[source_path] = _resolve_trajectory_records(
                source_path,
                trajectory_records_by_path=trajectory_records_by_path,
            )
        records, evidence = cache[source_path]
        rows.append(
            _history_row(
                detail=detail,
                manifest=manifest,
                records=records,
                trajectory_evidence=evidence,
            )
        )
    evidence_by_path = {path: evidence for path, (_, evidence) in cache.items()}
    return rows, {
        "trajectory_input_count": len(evidence_by_path),
        "trajectory_inputs": evidence_by_path,
    }


def _resolve_trajectory_records(
    source_path: str | None,
    *,
    trajectory_records_by_path: Mapping[str, Sequence[Mapping[str, object]]] | None,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    if not source_path:
        return (), {"loaded": False, "path": None, "error": "missing_source_path"}
    if trajectory_records_by_path is not None and source_path in trajectory_records_by_path:
        rows = tuple(dict(row) for row in trajectory_records_by_path[source_path])
        return rows, {
            "loaded": True,
            "in_memory": True,
            "path": source_path,
            "record_count": len(rows),
        }
    path = Path(source_path)
    if not path.exists():
        return (), {
            "loaded": False,
            "path": source_path,
            "record_count": 0,
            "error": "trajectory_path_missing",
        }
    try:
        rows = _load_trajectory_records(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (), {
            "loaded": False,
            "path": source_path,
            "record_count": 0,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    return rows, {
        "loaded": True,
        "path": source_path,
        "record_count": len(rows),
        "file_sha256": _file_sha256(path),
    }


def _load_trajectory_records(path: Path) -> tuple[dict[str, object], ...]:
    payloads: list[dict[str, object]] = []
    with _open_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError("trajectory payload must be a JSON object")
            payloads.append(payload)
    records: list[dict[str, object]] = []
    for payload in payloads:
        if payload.get("type") != "record":
            continue
        record = payload.get("record")
        records.append(dict(record if isinstance(record, Mapping) else payload))
    return tuple(records)


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _history_row(
    *,
    detail: Mapping[str, object],
    manifest: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    trajectory_evidence: Mapping[str, object],
) -> dict[str, object]:
    metadata = _mapping(manifest.get("non_trainable_audit_metadata"))
    record_index = _int(metadata.get("record_index"), default=-1)
    agent_id = _int(metadata.get("agent_id"), default=-1)
    missing: list[str] = []
    current: Mapping[str, object] = {}
    if trajectory_evidence.get("loaded") is not True:
        missing.append("trajectory_records")
    elif not (0 <= record_index < len(records)):
        missing.append("trajectory.record_index")
    else:
        current = records[record_index]
    if current:
        if _string(manifest.get("selected_observation_digest")) and (
            current.get("observation_digest") != manifest.get("selected_observation_digest")
        ):
            missing.append("trajectory.selected_observation_digest_match")
        if not isinstance(current.get("action_mask"), Mapping):
            missing.append("trajectory.current.action_mask")
        if not isinstance(current.get("observation_input"), Mapping):
            missing.append("trajectory.current.observation_input")
    prior_records = [
        dict(record)
        for record in records[: max(record_index, 0)]
        if _int(record.get("agent_id"), default=-999999) == agent_id
    ]
    for index, prior in enumerate(prior_records):
        missing.extend(_missing_prior_public_fields(prior, index=index))
    features: dict[str, object] = {}
    public_input_paths: list[str] = []
    if not missing and current:
        features, public_input_paths = _history_features_for_record(
            current=current,
            prior_same_agent_records=prior_records,
        )
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_ROW_SCHEMA_VERSION,
        "candidate_set_ordinal": detail.get("candidate_set_ordinal"),
        "slice_memberships": _list_like(detail.get("slice_memberships")),
        "split": detail.get("split"),
        "fixture_group": detail.get("fixture_group"),
        "blocker_class": detail.get("blocker_class"),
        "is_failed_branch": detail.get("is_failed_branch"),
        "prediction_correct": detail.get("prediction_correct"),
        "repaired_action": detail.get("repaired_action"),
        "predicted_action": detail.get("predicted_action"),
        "public_history_available": not missing,
        "missing_public_history_fields": sorted(set(missing)),
        "prior_same_agent_record_count": len(prior_records),
        "history_features": features,
        "public_input_paths_used": public_input_paths,
        "non_feature_metadata": {
            "source_path_used_for_audit_join": bool(_string(metadata.get("source_path"))),
            "record_index_used_for_audit_join": record_index >= 0,
            "agent_id_used_for_audit_join": agent_id >= 0,
            "trajectory_loaded": trajectory_evidence.get("loaded") is True,
        },
        "diagnostics_only": True,
        "training_authorized": False,
        "runtime_policy_authorized": False,
        "claim_causality": False,
    }


def _missing_prior_public_fields(
    record: Mapping[str, object],
    *,
    index: int,
) -> list[str]:
    missing: list[str] = []
    for field in ("requested_action", "resolved_action", "moved", "action_mask"):
        if field not in record:
            missing.append(f"trajectory.prior[{index}].{field}")
    outcome = _mapping(record.get("outcome"))
    if not outcome:
        missing.append(f"trajectory.prior[{index}].outcome")
    else:
        for path in ("resource_gain", "feeding", "drinking"):
            if path not in outcome:
                missing.append(f"trajectory.prior[{index}].outcome.{path}")
    if not isinstance(record.get("observation_input"), Mapping):
        missing.append(f"trajectory.prior[{index}].observation_input")
    return missing


def _history_features_for_record(
    *,
    current: Mapping[str, object],
    prior_same_agent_records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[str]]:
    rollout_state = RolloutContextState()
    for prior in prior_same_agent_records:
        rollout_state.update_from_record(prior)
    rollout_values = rollout_state.values()
    features = {
        f"rollout_context.{field}": value
        for field, value in zip(rollout_context_vector_fields(), rollout_values)
    }
    action_mask = {
        str(action): bool(value)
        for action, value in _mapping(current.get("action_mask")).items()
    }
    previous_observation = (
        prior_same_agent_records[-1].get("observation_input")
        if prior_same_agent_records
        else None
    )
    recovery_values = recovery_context_values(
        rollout_context_snapshot=rollout_state.snapshot(),
        action_mask=action_mask,
        current_observation_input=_mapping(current.get("observation_input")),
        previous_observation_input=previous_observation,
    )
    features.update(
        {
            f"recovery_context.{field}": value
            for field, value in zip(recovery_context_vector_fields(), recovery_values)
        }
    )
    features.update(_legal_action_mask_history_features(prior_same_agent_records))
    features.update(
        {
            "public_history.prior_same_agent_record_count": len(prior_same_agent_records),
            "public_history.has_prior_same_agent_history": bool(
                prior_same_agent_records
            ),
            "current_public_action_mask.legal_action_count": sum(
                1 for action in ACTION_NAMES if bool(action_mask.get(action))
            ),
            "current_public_action_mask.eat_legal": bool(action_mask.get("eat")),
            "current_public_action_mask.drink_legal": bool(action_mask.get("drink")),
            "current_public_action_mask.movement_legal": any(
                bool(action_mask.get(action)) for action in MOVEMENT_ACTIONS
            ),
        }
    )
    return features, _public_input_paths()


def _legal_action_mask_history_features(
    prior_same_agent_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    recent = list(prior_same_agent_records[-3:])
    denominator = max(1, len(recent))
    legal_counts = [
        sum(1 for action in ACTION_NAMES if bool(_mapping(row.get("action_mask")).get(action)))
        for row in recent
    ]
    return {
        "legal_action_mask_history.recent_window_size": len(recent),
        "legal_action_mask_history.recent_legal_action_count_mean": _round(
            sum(legal_counts) / denominator
        ),
        "legal_action_mask_history.recent_eat_legal_rate": _round(
            sum(1 for row in recent if bool(_mapping(row.get("action_mask")).get("eat")))
            / denominator
        ),
        "legal_action_mask_history.recent_drink_legal_rate": _round(
            sum(1 for row in recent if bool(_mapping(row.get("action_mask")).get("drink")))
            / denominator
        ),
        "legal_action_mask_history.recent_movement_legal_rate": _round(
            sum(
                1
                for row in recent
                if any(
                    bool(_mapping(row.get("action_mask")).get(action))
                    for action in MOVEMENT_ACTIONS
                )
            )
            / denominator
        ),
    }


def _public_input_paths() -> list[str]:
    return [
        "current_record.action_mask",
        "current_record.observation_input",
        "prior_same_agent_record.action_mask",
        "prior_same_agent_record.observation_input",
        "prior_same_agent_record.requested_action",
        "prior_same_agent_record.resolved_action",
        "prior_same_agent_record.moved",
        "prior_same_agent_record.outcome.resource_gain",
        "prior_same_agent_record.outcome.feeding",
        "prior_same_agent_record.outcome.drinking",
    ]


def _public_history_availability(
    *,
    rows: Sequence[Mapping[str, object]],
    trajectory_evidence: Mapping[str, object],
) -> dict[str, object]:
    missing_counter: Counter[str] = Counter()
    for row in rows:
        missing_counter.update(str(item) for item in _list_like(row.get("missing_public_history_fields")))
    loaded_inputs = [
        evidence
        for evidence in _mapping(trajectory_evidence.get("trajectory_inputs")).values()
        if _mapping(evidence).get("loaded") is True
    ]
    complete_rows = [row for row in rows if row.get("public_history_available") is True]
    prior_rows = [
        row for row in complete_rows if _int(row.get("prior_same_agent_record_count")) > 0
    ]
    return {
        "branch_count": len(rows),
        "history_row_count": len(rows),
        "history_available_row_count": len(complete_rows),
        "history_missing_row_count": len(rows) - len(complete_rows),
        "prior_same_agent_history_available_row_count": len(prior_rows),
        "missing_public_history_fields": dict(sorted(missing_counter.items())),
        "missing_public_history_field_count": sum(missing_counter.values()),
        "trajectory_input_count": trajectory_evidence.get("trajectory_input_count"),
        "loaded_trajectory_input_count": len(loaded_inputs),
        "trajectory_load_failures": [
            dict(evidence)
            for evidence in _mapping(trajectory_evidence.get("trajectory_inputs")).values()
            if _mapping(evidence).get("loaded") is not True
        ][:8],
        "complete": len(rows) > 0 and len(complete_rows) == len(rows),
        "allowed_public_history_feature_families": [
            "recent_public_action_history",
            "recent_public_observation_derived_resource_target_neighborhood_trends",
            "recent_legal_action_mask_history",
            "prior_public_material_opportunity_indicators",
            "branch_local_public_trajectory_context_before_decision",
        ],
    }


def _feature_surface(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    feature_paths = sorted(
        {
            str(path)
            for row in rows
            for path in _mapping(row.get("history_features")).keys()
        }
    )
    family_counts = Counter(path.split(".", 1)[0] for path in feature_paths)
    return {
        "row_count": len(rows),
        "history_feature_path_count": len(feature_paths),
        "history_feature_paths": feature_paths,
        "history_feature_family_counts": dict(sorted(family_counts.items())),
        "trainable_public_feature_payload_key": "history_features",
        "non_feature_metadata_key": "non_feature_metadata",
        "non_feature_metadata_separated": True,
        "rollout_context_feature_contract": rollout_context_feature_contract(),
        "recovery_context_feature_contract": recovery_context_feature_contract(),
    }


def _within_branch_history_variance(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "scope_note": (
            "v133 emits one branch-local pre-decision public history vector per "
            "v132 detail branch; counts below measure variation across affected "
            "branches within each slice."
        ),
        "all_blocker_rows": _variance_summary(rows),
        "fixture_open_action_collapse_rows": _variance_summary(
            [
                row
                for row in rows
                if row.get("blocker_class") == "fixture_open_action_collapse"
            ]
        ),
        "failed_heldout_rows": _variance_summary(
            [
                row
                for row in rows
                if row.get("split") in {"validation", "test"}
                and row.get("is_failed_branch") is True
            ]
        ),
        "open_heldout_overlap_rows": _variance_summary(
            [
                row
                for row in rows
                if row.get("fixture_group") == "open_mind_v3"
                and row.get("split") in {"validation", "test"}
            ]
        ),
    }


def _variance_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    features_by_path = _features_by_path(rows)
    varying = sorted(
        path
        for path, values in features_by_path.items()
        if len({_value_key(value) for value in values}) > 1
    )
    return {
        "row_count": len(rows),
        "available_row_count": sum(1 for row in rows if row.get("public_history_available") is True),
        "feature_path_count": len(features_by_path),
        "varying_feature_path_count": len(varying),
        "constant_feature_path_count": len(features_by_path) - len(varying),
        "varying_feature_paths": varying[:24],
    }


def _failed_vs_correct_history_contrast(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return _contrast_summary(
        rows,
        failed_label="failed_v132_blocker_rows",
        correct_label="nonfailed_v132_detail_rows",
        failed_filter=lambda row: row.get("is_failed_branch") is True,
        correct_filter=lambda row: row.get("is_failed_branch") is not True,
    )


def _open_fixture_collapse_analysis(
    rows: Sequence[Mapping[str, object]],
    v132_report: Mapping[str, object],
) -> dict[str, object]:
    open_rows = [row for row in rows if row.get("fixture_group") == "open_mind_v3"]
    collapse_rows = [
        row for row in open_rows if row.get("blocker_class") == "fixture_open_action_collapse"
    ]
    aggregate = _mapping(v132_report.get("blocker_class_aggregate"))
    fixture_failures = _mapping(aggregate.get("fixture_open_failures"))
    dominant = _mapping(fixture_failures.get("dominant_predicted_action"))
    contrast = _contrast_summary(
        open_rows,
        failed_label="fixture_open_action_collapse_rows",
        correct_label="open_fixture_nonfailed_rows",
        failed_filter=lambda row: row.get("blocker_class") == "fixture_open_action_collapse",
        correct_filter=lambda row: row.get("is_failed_branch") is not True,
    )
    return {
        "open_fixture_row_count": len(open_rows),
        "fixture_open_action_collapse_row_count": len(collapse_rows),
        "collapse_history_available_row_count": sum(
            1 for row in collapse_rows if row.get("public_history_available") is True
        ),
        "dominant_predicted_action": dict(dominant),
        "collapse_variance": _variance_summary(collapse_rows),
        "collapse_vs_nonfailed_open_contrast": contrast,
        "claim_causality": False,
    }


def _heldout_failure_analysis(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    heldout_rows = [row for row in rows if row.get("split") in {"validation", "test"}]
    failed = [row for row in heldout_rows if row.get("is_failed_branch") is True]
    return {
        "heldout_row_count": len(heldout_rows),
        "heldout_failed_row_count": len(failed),
        "heldout_failure_class_counts": dict(
            sorted(Counter(str(row.get("blocker_class")) for row in failed).items())
        ),
        "failed_vs_correct_heldout_contrast": _contrast_summary(
            heldout_rows,
            failed_label="failed_heldout_rows",
            correct_label="correct_heldout_rows",
            failed_filter=lambda row: row.get("is_failed_branch") is True,
            correct_filter=lambda row: row.get("is_failed_branch") is not True,
        ),
        "claim_causality": False,
    }


def _contrast_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    failed_label: str,
    correct_label: str,
    failed_filter,
    correct_filter,
) -> dict[str, object]:
    failed = [row for row in rows if failed_filter(row) and row.get("public_history_available") is True]
    correct = [row for row in rows if correct_filter(row) and row.get("public_history_available") is True]
    failed_by_path = _features_by_path(failed)
    correct_by_path = _features_by_path(correct)
    paths = sorted(set(failed_by_path) | set(correct_by_path))
    contrasting: list[dict[str, object]] = []
    for path in paths:
        failed_values = failed_by_path.get(path, [])
        correct_values = correct_by_path.get(path, [])
        if not failed_values or not correct_values:
            contrasting.append(
                {
                    "path": path,
                    "reason": "missing_in_one_group",
                    "failed_unique_count": len({_value_key(value) for value in failed_values}),
                    "correct_unique_count": len({_value_key(value) for value in correct_values}),
                }
            )
            continue
        failed_keys = {_value_key(value) for value in failed_values}
        correct_keys = {_value_key(value) for value in correct_values}
        if failed_keys != correct_keys:
            contrasting.append(
                {
                    "path": path,
                    "reason": "value_set_differs",
                    "failed_unique_count": len(failed_keys),
                    "correct_unique_count": len(correct_keys),
                    "failed_mean": _numeric_mean(failed_values),
                    "correct_mean": _numeric_mean(correct_values),
                }
            )
    return {
        f"{failed_label}_count": len(failed),
        f"{correct_label}_count": len(correct),
        "contrasting_feature_path_count": len(contrasting),
        "contrasting_feature_paths": contrasting[:24],
        "claim_causality": False,
    }


def _features_by_path(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[object]]:
    values: dict[str, list[object]] = {}
    for row in rows:
        if row.get("public_history_available") is not True:
            continue
        for path, value in _mapping(row.get("history_features")).items():
            values.setdefault(str(path), []).append(value)
    return values


def _leakage_audit(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    paths = sorted(
        {
            str(path)
            for row in rows
            for path in _mapping(row.get("history_features")).keys()
        }
    )
    forbidden = [
        {"path": path, "matched_forbidden_parts": _forbidden_matches(path)}
        for path in paths
        if _forbidden_matches(path)
    ]
    return {
        "passed": not forbidden,
        "leakage_count": len(forbidden),
        "forbidden_feature_paths": forbidden[:24],
        "forbidden_feature_families": sorted(FORBIDDEN_FEATURE_PATH_PARTS),
        "current_record_outcome_used_as_feature": False,
        "repaired_label_used_as_trainable_feature": False,
        "source_or_seed_identity_used_as_trainable_feature": False,
        "claim_causality": False,
    }


def _recommendation(
    *,
    source_integrity: Mapping[str, object],
    availability: Mapping[str, object],
    leakage: Mapping[str, object],
    open_analysis: Mapping[str, object],
    contrast: Mapping[str, object],
    feature_surface: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True or leakage.get("passed") is not True:
        next_step = "history_context_leakage_or_integrity_failed"
    elif availability.get("complete") is not True:
        next_step = "collect_missing_public_trajectory_history"
    else:
        open_contrast = _mapping(open_analysis.get("collapse_vs_nonfailed_open_contrast"))
        open_variance = _mapping(open_analysis.get("collapse_variance"))
        if (
            _int(open_contrast.get("contrasting_feature_path_count")) > 0
            and _int(open_variance.get("varying_feature_path_count")) > 0
        ):
            next_step = "public_rollout_history_context_ready_for_refreshed_surface"
        elif _int(contrast.get("contrasting_feature_path_count")) > 0:
            next_step = "history_context_present_but_not_discriminative"
        elif _int(feature_surface.get("history_feature_path_count")) > 0:
            next_step = "stop_ranker_path_insufficient_public_history_signal"
        else:
            next_step = "collect_missing_public_trajectory_history"
    return {
        "next_step": next_step,
        "allowed_next_steps": list(RECOMMENDATIONS),
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "training_executed": False,
        "runtime_policy_change_recommended": False,
        "claim_causality": False,
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    availability: Mapping[str, object],
    leakage: Mapping[str, object],
    recommendation: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True or leakage.get("passed") is not True:
        primary = "public_rollout_history_context_source_integrity_failed"
    elif availability.get("complete") is not True:
        primary = "public_rollout_history_context_public_inputs_missing"
    elif recommendation.get("next_step") == (
        "public_rollout_history_context_ready_for_refreshed_surface"
    ):
        primary = "public_rollout_history_context_ready_for_refreshed_surface"
    elif recommendation.get("next_step") == "history_context_present_but_not_discriminative":
        primary = "public_rollout_history_context_present_but_not_discriminative"
    else:
        primary = "public_rollout_history_context_insufficient_signal"
    return {
        "primary": primary,
        "labels": [primary],
        "allowed_classifications": list(CLASSIFICATIONS),
    }


def _authorization_block() -> dict[str, object]:
    return {
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "training_authorized": False,
        "training_executed": False,
        "runtime_policy_change_authorized": False,
        "runtime_policy_change_recommended": False,
        "runtime_loadable_artifact_created": False,
        "readiness_rerun_executed": False,
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "foundation_effect": "none",
        "shadow_scorer_effect": "none",
        "claim_causality": False,
    }


def _forbidden_matches(path: str) -> list[str]:
    lowered = path.lower()
    tokens = set(_path_tokens(lowered))
    matches = sorted(FORBIDDEN_FEATURE_PATH_PARTS & tokens)
    for substring in FORBIDDEN_FEATURE_PATH_SUBSTRINGS:
        if substring in lowered and substring not in matches:
            matches.append(substring)
    return sorted(matches)


def _path_tokens(path: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for character in path:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    tokens.extend(path.replace("-", "_").replace(":", "_").split("_"))
    return [token for token in tokens if token]


def _value_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _numeric_mean(values: Sequence[object]) -> float | None:
    parsed = [
        float(value)
        for value in values
        if not isinstance(value, bool) and isinstance(value, (int, float))
    ]
    if not parsed:
        return None
    return _round(sum(parsed) / len(parsed))


def _int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return int(value)


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _round(value: float) -> float:
    return round(float(value), 6)
