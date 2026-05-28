from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V124_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V123_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    trainable_public_input_leakage,
)
from evolution_sim.mind.first_recovery_candidate_public_feature_surface import (
    DEFAULT_FEATURE_ROWS_OUTPUT_PATH as DEFAULT_V129_FEATURE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V129_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION,
    MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION,
    _action_semantics,
    _feature_family,
)
from evolution_sim.mind.first_recovery_candidate_ranker_capacity_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V128_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_CANDIDATE_RANKER_CAPACITY_AUDIT_SCHEMA_VERSION,
    _list_like,
)
from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V127_REPORT_PATH,
    DEFAULT_PREDICTIONS_OUTPUT_PATH as DEFAULT_V127_PREDICTIONS_PATH,
    DEFAULT_V115_ARCHIVE_ROWS_PATH,
    DEFAULT_V115_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_SCHEMA_VERSION,
    _candidate_groups,
    _candidate_set_audit,
    _resolve_jsonl_rows,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _decode_observation,
    _int,
    _mapping,
    _resolve_archive_rows,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    _trainable_leakage,
)
from evolution_sim.mind.first_recovery_shadow_ranker import (
    _load_join_records,
    _matching_record,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_observation_candidate_context_v1"
)
MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_POLICY = (
    "diagnostics_only_first_recovery_v130_observation_candidate_context_v1"
)
MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_ROW_SCHEMA_VERSION = (
    "mind_v3_first_recovery_observation_candidate_context_row_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v130-first-recovery-observation-candidate-context.json"
)
DEFAULT_CONTEXT_ROWS_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v130-first-recovery-observation-candidate-context-rows.jsonl"
)

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "observation_candidate_context_source_integrity_failed",
    "observation_candidate_context_public_fields_missing",
    "observation_candidate_context_ready_for_v129_surface_refresh",
)

FORBIDDEN_FEATURE_PATH_PARTS = frozenset(
    {
        "archive_id",
        "archive_row_id",
        "branch_id",
        "fixture",
        "fixture_identity",
        "fixture_name",
        "first_action_outcome",
        "material_gain",
        "material_gain_label",
        "objective",
        "oracle",
        "oracle_rank",
        "private",
        "provenance",
        "record_index",
        "recovery_vitals_deltas",
        "repaired_action",
        "repaired_archive_row_id",
        "replay_verification",
        "resolution_action_mask",
        "resolution_legal",
        "seed",
        "source",
        "source_path",
        "world",
    }
)
FORBIDDEN_FEATURE_PATH_SUBSTRINGS = (
    "archive_row_id",
    "first_action_outcome",
    "material_gain_label",
    "oracle_rank",
    "private_world_state",
    "recovery_vitals_deltas",
    "repaired_action",
    "repaired_archive_row_id",
    "replay_verification",
    "resolution_action_mask",
    "resolution_legal",
    "source_path",
)
REQUIRED_PUBLIC_OBSERVATION_FIELDS = (
    "trajectory_record.observation_input",
    "trajectory_record.action_mask",
    "decoded.patch",
    "decoded.navigation",
)
MAX_EXAMPLES = 16


@dataclass(frozen=True, slots=True)
class FirstRecoveryObservationCandidateContextBuild:
    report: dict[str, object]
    context_rows: tuple[dict[str, object], ...]


def build_first_recovery_observation_candidate_context(
    *,
    v129_report: Mapping[str, object] | None = None,
    v129_report_path: str | Path | None = DEFAULT_V129_REPORT_PATH,
    v129_feature_rows: Sequence[Mapping[str, object]] | None = None,
    v129_feature_rows_path: str | Path | None = DEFAULT_V129_FEATURE_ROWS_PATH,
    v128_report: Mapping[str, object] | None = None,
    v128_report_path: str | Path | None = DEFAULT_V128_REPORT_PATH,
    v127_report: Mapping[str, object] | None = None,
    v127_report_path: str | Path | None = DEFAULT_V127_REPORT_PATH,
    v127_prediction_rows: Sequence[Mapping[str, object]] | None = None,
    v127_predictions_path: str | Path | None = DEFAULT_V127_PREDICTIONS_PATH,
    v124_report: Mapping[str, object] | None = None,
    v124_report_path: str | Path | None = DEFAULT_V124_REPORT_PATH,
    v124_manifest_rows: Sequence[Mapping[str, object]] | None = None,
    v124_manifest_path: str | Path | None = DEFAULT_V124_MANIFEST_PATH,
    v115_report: Mapping[str, object] | None = None,
    v115_report_path: str | Path | None = DEFAULT_V115_REPORT_PATH,
    v115_archive_rows: Sequence[Mapping[str, object]] | None = None,
    v115_archive_rows_path: str | Path | None = DEFAULT_V115_ARCHIVE_ROWS_PATH,
    v123_report: Mapping[str, object] | None = None,
    v123_report_path: str | Path | None = DEFAULT_V123_REPORT_PATH,
    v123_archive_rows: Sequence[Mapping[str, object]] | None = None,
    v123_archive_rows_path: str | Path | None = DEFAULT_V123_ARCHIVE_ROWS_PATH,
    trajectory_records: Sequence[Mapping[str, object]] | None = None,
    trajectory_paths: Sequence[str | Path] | None = None,
) -> FirstRecoveryObservationCandidateContextBuild:
    v129_payload, v129_evidence = _resolve_json_report(
        v129_report,
        v129_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION,
    )
    v129_rows, v129_rows_evidence = _resolve_jsonl_rows(
        v129_feature_rows,
        v129_feature_rows_path,
        row_kind="v129_feature_rows",
    )
    v128_payload, v128_evidence = _resolve_json_report(
        v128_report,
        v128_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_CANDIDATE_RANKER_CAPACITY_AUDIT_SCHEMA_VERSION,
    )
    v127_payload, v127_evidence = _resolve_json_report(
        v127_report,
        v127_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_SCHEMA_VERSION,
    )
    v127_predictions, v127_predictions_evidence = _resolve_jsonl_rows(
        v127_prediction_rows,
        v127_predictions_path,
        row_kind="v127_predictions",
    )
    v124_payload, v124_evidence = _resolve_json_report(
        v124_report,
        v124_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION,
    )
    manifest_rows, manifest_evidence = _resolve_jsonl_rows(
        v124_manifest_rows,
        v124_manifest_path,
        row_kind="v124_manifest",
    )
    v115_payload, v115_evidence = _resolve_json_report(
        v115_report,
        v115_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    v115_rows, v115_rows_evidence = _resolve_archive_rows(
        v115_archive_rows,
        v115_archive_rows_path,
    )
    v123_payload, v123_evidence = _resolve_json_report(
        v123_report,
        v123_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    v123_rows, v123_rows_evidence = _resolve_archive_rows(
        v123_archive_rows,
        v123_archive_rows_path,
    )
    candidate_rows = tuple(v115_rows) + tuple(v123_rows)
    candidate_groups = _candidate_groups(candidate_rows)
    candidate_set_audit = _candidate_set_audit(
        manifest_rows=manifest_rows,
        candidate_groups=candidate_groups,
        candidate_rows=candidate_rows,
        v115_rows=v115_rows,
        v123_rows=v123_rows,
    )
    trajectory_join = _resolve_trajectory_records(
        candidate_rows=candidate_rows,
        trajectory_records=trajectory_records,
        trajectory_paths=trajectory_paths,
    )
    source_reports = {
        "v129_report": v129_evidence,
        "v129_feature_rows": v129_rows_evidence,
        "v128_report": v128_evidence,
        "v127_report": v127_evidence,
        "v127_predictions": v127_predictions_evidence,
        "v124_report": v124_evidence,
        "v124_manifest": manifest_evidence,
        "v115_report": v115_evidence,
        "v115_archive_rows": v115_rows_evidence,
        "v123_report": v123_evidence,
        "v123_archive_rows": v123_rows_evidence,
        "trajectory_inputs": trajectory_join["evidence"],
    }
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v129_report=v129_payload,
        v129_feature_rows=v129_rows,
        v128_report=v128_payload,
        v127_report=v127_payload,
        v127_predictions=v127_predictions,
        v124_report=v124_payload,
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
        candidate_set_audit=candidate_set_audit,
        trajectory_join=trajectory_join,
    )
    context_dataset, context_rows = _context_dataset(
        manifest_rows=manifest_rows,
        candidate_groups=candidate_groups,
        trajectory_join=trajectory_join,
    )
    allowlist = _allowlist_audit(context_dataset)
    forbidden = _forbidden_field_scan(context_dataset)
    availability = _candidate_context_availability(context_dataset)
    variance = _within_branch_variance(context_dataset)
    leakage = _leakage_audit(
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
        forbidden_scan=forbidden,
    )
    metric_gate = _metric_gate(
        source_integrity=source_integrity,
        allowlist=allowlist,
        forbidden=forbidden,
        availability=availability,
        variance=variance,
        leakage=leakage,
    )
    classification = _classification(source_integrity, metric_gate)
    authorization_block = _authorization_block()
    recommendation = _recommendation(classification)
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "public_input_allowlist": allowlist,
        "forbidden_field_scan": forbidden,
        "candidate_context_availability": availability,
        "within_branch_variance": variance,
        "leakage_audit": leakage,
        "metric_gate": metric_gate,
        "classification": classification,
        "authorization_block": authorization_block,
        "recommendation": recommendation,
        "context_rows_summary": {
            "row_count": len(context_rows),
            "context_rows_digest": stable_payload_digest(list(context_rows)),
            "default_output_path": str(DEFAULT_CONTEXT_ROWS_OUTPUT_PATH),
            "schema_version": MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_ROW_SCHEMA_VERSION,
            "runtime_loadable": False,
        },
        "non_promoted": True,
    }
    return FirstRecoveryObservationCandidateContextBuild(
        report=report,
        context_rows=tuple(context_rows),
    )


def write_first_recovery_observation_candidate_context_report(
    build: FirstRecoveryObservationCandidateContextBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_first_recovery_observation_candidate_context_rows(
    build: FirstRecoveryObservationCandidateContextBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in build.context_rows:
            json.dump(row, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_SCHEMA_VERSION,
        "diagnostics_only": True,
        "report_only_observation_candidate_context": True,
        "training_executed": False,
        "runtime_loadable_artifact_created": False,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "readiness_rerun_executed": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "foundation_effect": "none",
        "claim_causality": False,
    }


def _source_integrity(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    v129_report: Mapping[str, object] | None,
    v129_feature_rows: Sequence[Mapping[str, object]],
    v128_report: Mapping[str, object] | None,
    v127_report: Mapping[str, object] | None,
    v127_predictions: Sequence[Mapping[str, object]],
    v124_report: Mapping[str, object] | None,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    candidate_set_audit: Mapping[str, object],
    trajectory_join: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if name == "trajectory_inputs":
            if evidence.get("loaded") is not True:
                failures.append("missing_trajectory_inputs")
            continue
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")

    v129 = _mapping(v129_report or {})
    v128 = _mapping(v128_report or {})
    v127 = _mapping(v127_report or {})
    v124 = _mapping(v124_report or {})
    v129_summary = _mapping(v129.get("feature_surface_summary"))
    v129_source = _mapping(v129.get("source_integrity"))
    v129_forbidden = _mapping(v129.get("forbidden_feature_scan"))
    v128_source = _mapping(v128.get("source_integrity"))
    v127_source = _mapping(v127.get("source_integrity"))
    v124_source = _mapping(v124.get("source_integrity"))
    v124_contract = _mapping(v124.get("contract_checks"))
    v124_manifest = _mapping(v124.get("manifest"))
    manifest_digest = stable_payload_digest(list(manifest_rows))
    prediction_digest = stable_payload_digest(list(v127_predictions))

    if v129_source.get("passed") is not True:
        failures.append("v129_source_integrity_not_passed")
    if _int(v129_forbidden.get("forbidden_feature_path_count")) != 0:
        failures.append("v129_forbidden_features_nonzero")
    if _int(v129_summary.get("candidate_feature_row_count")) != len(v129_feature_rows):
        failures.append("v129_feature_row_count_mismatch")
    if len(v129_feature_rows) != len(candidate_rows):
        failures.append("v129_feature_rows_do_not_match_candidate_rows")
    if v128_source.get("passed") is not True:
        failures.append("v128_source_integrity_not_passed")
    if v127_source.get("passed") is not True:
        failures.append("v127_source_integrity_not_passed")
    if _mapping(v127.get("prediction_summary")).get("prediction_rows_digest") not in {
        None,
        prediction_digest,
    }:
        failures.append("v127_prediction_digest_mismatch")
    if v124_source.get("passed") is not True:
        failures.append("v124_source_integrity_not_passed")
    if v124_contract.get("passed") is not True:
        failures.append("v124_contract_checks_not_passed")
    if v124_manifest.get("manifest_digest") != manifest_digest:
        failures.append("v124_manifest_digest_mismatch")

    if _int(candidate_set_audit.get("branch_count")) != len(manifest_rows):
        failures.append("candidate_set_branch_count_mismatch")
    if _int(candidate_set_audit.get("total_candidate_rows")) != len(candidate_rows):
        failures.append("candidate_set_candidate_row_count_mismatch")
    if _int(candidate_set_audit.get("missing_candidate_set_branch_count")) != 0:
        failures.append("candidate_set_missing_branches")
    if _int(candidate_set_audit.get("unsupported_repaired_labels_count")) != 0:
        failures.append("candidate_set_unsupported_repaired_labels")

    join = _mapping(trajectory_join.get("join"))
    if _int(join.get("matched_candidate_row_count")) != len(candidate_rows):
        failures.append("trajectory_join_candidate_row_mismatch")
    if _int(join.get("missing_candidate_row_count")) != 0:
        failures.append("trajectory_join_missing_candidate_rows")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "manifest_row_count": len(manifest_rows),
        "manifest_digest": manifest_digest,
        "candidate_row_count": len(candidate_rows),
        "branch_count": _int(candidate_set_audit.get("branch_count")),
        "v129_source_integrity_passed": v129_source.get("passed"),
        "v129_feature_row_count": len(v129_feature_rows),
        "v129_forbidden_feature_path_count": _int(
            v129_forbidden.get("forbidden_feature_path_count")
        ),
        "v128_source_integrity_passed": v128_source.get("passed"),
        "v127_source_integrity_passed": v127_source.get("passed"),
        "v124_source_integrity_passed": v124_source.get("passed"),
        "candidate_set_audit": dict(candidate_set_audit),
        "trajectory_join": join,
        "missing_public_fields": _list_like(trajectory_join.get("missing_public_fields")),
    }


def _resolve_trajectory_records(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    trajectory_records: Sequence[Mapping[str, object]] | None,
    trajectory_paths: Sequence[str | Path] | None,
) -> dict[str, object]:
    required_paths = sorted(
        {
            str(_mapping(row.get("provenance")).get("source_path"))
            for row in candidate_rows
            if _mapping(row.get("provenance")).get("source_path")
        }
    )
    if trajectory_records is not None:
        record_by_key: dict[tuple[str, int], dict[str, object]] = {}
        for record in trajectory_records:
            row = dict(record)
            source_path = row.get("source_path")
            record_index = _int(row.get("record_index"), default=-1)
            if isinstance(source_path, str) and record_index >= 0:
                record_by_key[(source_path, record_index)] = row
        evidence = {
            "loaded": True,
            "in_memory": True,
            "loaded_path_count": len({key[0] for key in record_by_key}),
            "record_count": len(record_by_key),
            "required_source_path_count": len(required_paths),
            "required_source_paths": required_paths,
        }
    else:
        paths = tuple(Path(path) for path in (trajectory_paths or required_paths))
        loaded = _load_join_records(paths, ())
        evidence = dict(_mapping(loaded.get("evidence")))
        evidence["loaded"] = _int(evidence.get("loaded_path_count")) > 0
        evidence["required_source_path_count"] = len(required_paths)
        evidence["required_source_paths"] = required_paths
        record_by_key = {
            key: dict(value)
            for key, value in _mapping(loaded.get("record_by_key")).items()
            if isinstance(key, tuple) and len(key) == 2 and isinstance(value, Mapping)
        }
    matched = 0
    missing = 0
    missing_examples: list[dict[str, object]] = []
    missing_fields: Counter[str] = Counter()
    for row in candidate_rows:
        record = _matching_record(row, record_by_key)
        if record is None:
            missing += 1
            missing_fields["trajectory_record.join_by_source_path_record_index"] += 1
            if len(missing_examples) < MAX_EXAMPLES:
                missing_examples.append(_join_missing_example(row))
            continue
        matched += 1
        for field in _record_missing_public_fields(record):
            missing_fields[field] += 1
    return {
        "evidence": evidence,
        "record_by_key": record_by_key,
        "join": {
            "matched_candidate_row_count": matched,
            "missing_candidate_row_count": missing,
            "expected_candidate_row_count": len(candidate_rows),
            "missing_examples": missing_examples,
        },
        "missing_public_fields": [
            {"field": field, "candidate_row_count": count}
            for field, count in sorted(missing_fields.items())
        ],
    }


def _context_dataset(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
    trajectory_join: Mapping[str, object],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    record_by_key = _mapping(trajectory_join.get("record_by_key"))
    dataset: list[dict[str, object]] = []
    rows_out: list[dict[str, object]] = []
    branch_ordinals = {
        _branch_id(manifest): index
        for index, manifest in enumerate(
            sorted(manifest_rows, key=lambda row: str(row.get("branch_id")))
        )
        if _branch_id(manifest) is not None
    }
    for manifest in sorted(manifest_rows, key=lambda row: str(row.get("branch_id"))):
        branch_id = _branch_id(manifest)
        if branch_id is None:
            continue
        candidates = tuple(candidate_groups.get(branch_id, ()))
        for candidate_index, candidate in enumerate(candidates):
            record = _matching_record(candidate, record_by_key)
            extraction = _candidate_context_features(candidate, record)
            feature_row = {
                "schema_version": MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_ROW_SCHEMA_VERSION,
                "candidate_observation_context": extraction["features"],
                "extractor_source_paths": extraction["source_paths"],
                "missing_public_fields": extraction["missing_public_fields"],
                "non_feature_metadata": {
                    "candidate_set_ordinal": branch_ordinals.get(branch_id),
                    "candidate_ordinal_within_set": candidate_index,
                },
                "diagnostics_only": True,
                "training_authorized": False,
                "runtime_policy_authorized": False,
            }
            dataset.append(
                {
                    "branch_id": branch_id,
                    "candidate_action": candidate.get("candidate_action"),
                    "features": extraction["features"],
                    "source_paths": extraction["source_paths"],
                    "missing_public_fields": extraction["missing_public_fields"],
                }
            )
            rows_out.append(feature_row)
    return tuple(dataset), tuple(rows_out)


def _candidate_context_features(
    candidate: Mapping[str, object],
    record: Mapping[str, object] | None,
) -> dict[str, object]:
    action = str(candidate.get("candidate_action") or "")
    action_mask = _mapping(_mapping(candidate.get("trainable_public_input")).get("action_mask"))
    if record is not None and isinstance(record.get("action_mask"), Mapping):
        action_mask = _mapping(record.get("action_mask"))
    features: dict[str, object] = {
        "candidate_action": action,
        "candidate_action_observation_legal": action_mask.get(action) is True,
    }
    source_paths = {
        "candidate_row.candidate_action",
        "trajectory_record.action_mask",
    }
    missing = []
    for key, value in _action_semantics(action).items():
        features[f"candidate_action_{key}"] = value
    if record is None:
        missing.append("trajectory_record.join_by_source_path_record_index")
        return {
            "features": features,
            "source_paths": tuple(sorted(source_paths)),
            "missing_public_fields": tuple(missing),
        }
    decoded = _mapping(record.get("decoded_observation"))
    if not decoded:
        try:
            decoded = _decode_observation(_mapping(record.get("observation_input")))
        except (TypeError, ValueError):
            decoded = {}
    if not decoded:
        missing.append("trajectory_record.observation_input")
        return {
            "features": features,
            "source_paths": tuple(sorted(source_paths)),
            "missing_public_fields": tuple(missing),
        }
    source_paths.add("trajectory_record.observation_input")
    source_paths.add("decoded.patch")
    source_paths.add("decoded.navigation")
    patch = _mapping(decoded.get("patch"))
    navigation = _mapping(decoded.get("navigation"))
    direction = _direction_delta(action)
    if direction is not None:
        dx, dy = direction
        cell = _patch_cell(patch, dx, dy)
        if not cell:
            missing.append(f"decoded.patch[{dx},{dy}]")
        _add_neighbor_context(features, cell, prefix="candidate_target_context")
    else:
        _add_not_applicable_neighbor_context(features, prefix="candidate_target_context")
    center = _patch_cell(patch, 0, 0)
    if action == "eat":
        _add_resource_context(features, center, navigation, resource="eat")
    elif action == "drink":
        _add_resource_context(features, center, navigation, resource="drink")
    else:
        _add_not_applicable_resource_context(features)
    _add_neighborhood_summary(features, patch)
    return {
        "features": features,
        "source_paths": tuple(sorted(source_paths)),
        "missing_public_fields": tuple(sorted(set(missing))),
    }


def _add_neighbor_context(
    features: dict[str, object],
    cell: Mapping[str, object],
    *,
    prefix: str,
) -> None:
    features[f"{prefix}.in_bounds"] = _boolish(cell.get("in_bounds"))
    features[f"{prefix}.terrain_bucket"] = _signed_bucket(cell.get("terrain_code"))
    features[f"{prefix}.occupant_bucket"] = _signed_bucket(cell.get("occupant_code"))
    features[f"{prefix}.same_lineage_public"] = _boolish(cell.get("same_lineage"))
    features[f"{prefix}.food_bucket"] = _positive_bucket(cell.get("food"))
    features[f"{prefix}.vegetation_bucket"] = _positive_bucket(cell.get("vegetation"))
    features[f"{prefix}.fresh_kill_bucket"] = _positive_bucket(cell.get("fresh_kill_energy"))
    features[f"{prefix}.carcass_bucket"] = _positive_bucket(cell.get("carcass_energy"))
    features[f"{prefix}.prey_biomass_bucket"] = _positive_bucket(cell.get("prey_biomass"))
    features[f"{prefix}.carrion_signal_bucket"] = _positive_bucket(cell.get("carrion_signal"))
    features[f"{prefix}.predator_risk_bucket"] = _positive_bucket(cell.get("predator_risk"))
    features[f"{prefix}.hazard_bucket"] = _positive_bucket(cell.get("hazard_level"))
    features[f"{prefix}.water_access_bucket"] = _signed_bucket(
        cell.get("water_access_reason_code")
    )
    features[f"{prefix}.ecology_state_bucket"] = _signed_bucket(
        cell.get("ecology_state_code")
    )


def _add_not_applicable_neighbor_context(
    features: dict[str, object],
    *,
    prefix: str,
) -> None:
    for name in (
        "in_bounds",
        "terrain_bucket",
        "occupant_bucket",
        "same_lineage_public",
        "food_bucket",
        "vegetation_bucket",
        "fresh_kill_bucket",
        "carcass_bucket",
        "prey_biomass_bucket",
        "carrion_signal_bucket",
        "predator_risk_bucket",
        "hazard_bucket",
        "water_access_bucket",
        "ecology_state_bucket",
    ):
        features[f"{prefix}.{name}"] = "not_applicable"


def _add_resource_context(
    features: dict[str, object],
    center: Mapping[str, object],
    navigation: Mapping[str, object],
    *,
    resource: str,
) -> None:
    features["candidate_resource_context.kind"] = resource
    if resource == "eat":
        features["candidate_resource_context.center_food_bucket"] = _positive_bucket(center.get("food"))
        features["candidate_resource_context.center_vegetation_bucket"] = _positive_bucket(center.get("vegetation"))
        features["candidate_resource_context.center_fresh_kill_bucket"] = _positive_bucket(center.get("fresh_kill_energy"))
        features["candidate_resource_context.center_carcass_bucket"] = _positive_bucket(center.get("carcass_energy"))
        features["candidate_resource_context.center_prey_biomass_bucket"] = _positive_bucket(center.get("prey_biomass"))
        features["candidate_resource_context.nav_plant_strength_bucket"] = _nav_bucket(navigation, "plant", "strength")
        features["candidate_resource_context.nav_carrion_strength_bucket"] = _nav_bucket(navigation, "carrion", "strength")
        features["candidate_resource_context.nav_prey_strength_bucket"] = _nav_bucket(navigation, "prey", "strength")
        features["candidate_resource_context.nav_water_strength_bucket"] = "not_applicable"
    else:
        features["candidate_resource_context.center_food_bucket"] = "not_applicable"
        features["candidate_resource_context.center_vegetation_bucket"] = "not_applicable"
        features["candidate_resource_context.center_fresh_kill_bucket"] = "not_applicable"
        features["candidate_resource_context.center_carcass_bucket"] = "not_applicable"
        features["candidate_resource_context.center_prey_biomass_bucket"] = "not_applicable"
        features["candidate_resource_context.nav_plant_strength_bucket"] = "not_applicable"
        features["candidate_resource_context.nav_carrion_strength_bucket"] = "not_applicable"
        features["candidate_resource_context.nav_prey_strength_bucket"] = "not_applicable"
        features["candidate_resource_context.nav_water_strength_bucket"] = _nav_bucket(navigation, "water", "strength")
    features["candidate_resource_context.center_water_access_bucket"] = _signed_bucket(
        center.get("water_access_reason_code")
    )
    features["candidate_resource_context.center_terrain_bucket"] = _signed_bucket(
        center.get("terrain_code")
    )


def _add_not_applicable_resource_context(features: dict[str, object]) -> None:
    for name in (
        "kind",
        "center_food_bucket",
        "center_vegetation_bucket",
        "center_fresh_kill_bucket",
        "center_carcass_bucket",
        "center_prey_biomass_bucket",
        "center_water_access_bucket",
        "center_terrain_bucket",
        "nav_plant_strength_bucket",
        "nav_carrion_strength_bucket",
        "nav_prey_strength_bucket",
        "nav_water_strength_bucket",
    ):
        features[f"candidate_resource_context.{name}"] = "not_applicable"


def _add_neighborhood_summary(
    features: dict[str, object],
    patch: Mapping[object, object],
) -> None:
    cells = [_mapping(value) for value in patch.values()]
    features["candidate_neighborhood_context.in_bounds_count_bucket"] = _count_bucket(
        sum(1 for cell in cells if _boolish(cell.get("in_bounds")) is True)
    )
    features["candidate_neighborhood_context.food_cell_count_bucket"] = _count_bucket(
        sum(1 for cell in cells if _number(cell.get("food")) > 0.05)
    )
    features["candidate_neighborhood_context.carrion_cell_count_bucket"] = _count_bucket(
        sum(
            1
            for cell in cells
            if _number(cell.get("fresh_kill_energy")) > 0.05
            or _number(cell.get("carcass_energy")) > 0.05
        )
    )
    features["candidate_neighborhood_context.prey_cell_count_bucket"] = _count_bucket(
        sum(1 for cell in cells if _number(cell.get("prey_biomass")) > 0.05)
    )
    features["candidate_neighborhood_context.occupied_cell_count_bucket"] = _count_bucket(
        sum(1 for cell in cells if _number(cell.get("occupant_code")) > 0.1)
    )
    features["candidate_neighborhood_context.hazard_cell_count_bucket"] = _count_bucket(
        sum(1 for cell in cells if _number(cell.get("hazard_level")) > 0.05)
    )
    features["candidate_neighborhood_context.water_cell_count_bucket"] = _count_bucket(
        sum(
            1
            for cell in cells
            if _number(cell.get("water_access_reason_code")) > 0.1
            or _number(cell.get("terrain_code")) > 0.75
        )
    )


def _candidate_context_availability(
    dataset: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    action_counts: Counter[str] = Counter()
    available_counts: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    branch_ids = {str(row.get("branch_id")) for row in dataset}
    for row in dataset:
        action = str(row.get("candidate_action"))
        action_counts[action] += 1
        missing = _list_like(row.get("missing_public_fields"))
        if not missing:
            available_counts[action] += 1
        for field in missing:
            missing_fields[str(field)] += 1
    per_action = {
        action: {
            "candidate_row_count": action_counts[action],
            "available_context_row_count": available_counts[action],
            "availability_rate": _ratio(
                available_counts[action],
                action_counts[action],
            ),
        }
        for action in sorted(action_counts)
    }
    return {
        "branch_count": len(branch_ids),
        "candidate_row_count": len(dataset),
        "available_context_row_count": sum(available_counts.values()),
        "per_action_context_availability": per_action,
        "missing_public_observation_fields": [
            {"field": field, "candidate_row_count": count}
            for field, count in sorted(missing_fields.items())
        ],
        "required_public_observation_fields": list(REQUIRED_PUBLIC_OBSERVATION_FIELDS),
    }


def _within_branch_variance(
    dataset: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in dataset:
        grouped[str(row.get("branch_id"))].append(row)
    path_branch_counts: Counter[str] = Counter()
    path_examples: defaultdict[str, list[str]] = defaultdict(list)
    for branch_id, rows in sorted(grouped.items()):
        path_values: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            for path, value in _mapping(row.get("features")).items():
                path_values[path].add(json.dumps(value, sort_keys=True))
        for path, values in path_values.items():
            if len(values) <= 1:
                continue
            path_branch_counts[path] += 1
            if len(path_examples[path]) < 4:
                path_examples[path].append(branch_id)
    varying = {
        path: {
            "branch_count": count,
            "example_branches": path_examples[path],
            "field_family": _context_feature_family(path),
        }
        for path, count in sorted(path_branch_counts.items())
    }
    return {
        "branch_count": len(grouped),
        "action_identity_variance": _variance_family(varying, "action_identity"),
        "action_semantics_variance": _variance_family(varying, "action_semantics"),
        "action_mask_legality_variance": _variance_family(
            varying,
            "action_mask_legality",
        ),
        "target_resource_neighborhood_variance": _variance_family(
            varying,
            "target_resource_neighborhood",
        ),
        "varying_paths": varying,
        "varying_path_count": len(varying),
    }


def _allowlist_audit(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    feature_paths = _actual_feature_paths(dataset)
    source_paths = sorted(
        {
            str(path)
            for row in dataset
            for path in _list_like(row.get("source_paths"))
        }
    )
    denied_feature_paths = [
        path for path in feature_paths if not _allowed_feature_path(path)
    ]
    denied_source_paths = [
        path for path in source_paths if not _allowed_source_path(path)
    ]
    return {
        "passed": not denied_feature_paths and not denied_source_paths,
        "actual_feature_paths": feature_paths,
        "actual_extractor_source_paths": source_paths,
        "denied_feature_paths": denied_feature_paths,
        "denied_source_paths": denied_source_paths,
        "policy": "observation_time_public_observation_input_and_action_mask_only",
    }


def _forbidden_field_scan(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    paths = _actual_feature_paths(dataset)
    forbidden = [
        {"path": path, "matched_forbidden_parts": _forbidden_matches(path)}
        for path in paths
        if _forbidden_matches(path)
    ]
    return {
        "passed": not forbidden,
        "forbidden_feature_path_count": len(forbidden),
        "forbidden_feature_paths": forbidden[:MAX_EXAMPLES],
        "scanned_feature_path_count": len(paths),
        "forbidden_feature_families": sorted(FORBIDDEN_FEATURE_PATH_PARTS),
    }


def _leakage_audit(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    forbidden_scan: Mapping[str, object],
) -> dict[str, object]:
    manifest_leakage = _trainable_leakage(manifest_rows)
    candidate_leakage = trainable_public_input_leakage(candidate_rows)
    count = (
        _int(manifest_leakage.get("split_key_leak_count"))
        + _int(manifest_leakage.get("forbidden_metadata_key_count"))
        + _int(candidate_leakage.get("leak_count"))
        + _int(forbidden_scan.get("forbidden_feature_path_count"))
    )
    return {
        "passed": count == 0,
        "leakage_count": count,
        "manifest_trainable_leakage": manifest_leakage,
        "candidate_trainable_leakage": candidate_leakage,
        "forbidden_feature_scan": dict(forbidden_scan),
    }


def _metric_gate(
    *,
    source_integrity: Mapping[str, object],
    allowlist: Mapping[str, object],
    forbidden: Mapping[str, object],
    availability: Mapping[str, object],
    variance: Mapping[str, object],
    leakage: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    target_variance = _mapping(variance.get("target_resource_neighborhood_variance"))
    target_varying = _int(target_variance.get("varying_path_count"))
    if source_integrity.get("passed") is not True:
        failures.append("source_integrity_failed")
    if allowlist.get("passed") is not True:
        failures.append("public_input_allowlist_failed")
    if forbidden.get("passed") is not True:
        failures.append("forbidden_field_scan_failed")
    if leakage.get("passed") is not True:
        failures.append("leakage_failed")
    if _list_like(availability.get("missing_public_observation_fields")):
        failures.append("public_observation_fields_missing")
    if target_varying <= 0:
        failures.append("target_resource_neighborhood_context_does_not_vary")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "source_integrity_passed": source_integrity.get("passed"),
        "public_input_allowlist_passed": allowlist.get("passed"),
        "forbidden_field_scan_passed": forbidden.get("passed"),
        "leakage_passed": leakage.get("passed"),
        "missing_public_observation_field_count": len(
            _list_like(availability.get("missing_public_observation_fields"))
        ),
        "target_resource_neighborhood_varying_path_count": target_varying,
        "target_resource_neighborhood_variance_required": True,
    }


def _classification(
    source_integrity: Mapping[str, object],
    metric_gate: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True:
        primary = "observation_candidate_context_source_integrity_failed"
    elif metric_gate.get("passed") is True:
        primary = "observation_candidate_context_ready_for_v129_surface_refresh"
    else:
        primary = "observation_candidate_context_public_fields_missing"
    return {
        "primary": primary,
        "labels": [primary],
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    ready = (
        classification.get("primary")
        == "observation_candidate_context_ready_for_v129_surface_refresh"
    )
    return {
        "next_step": (
            "review_v130_rows_before_any_v129_surface_refresh"
            if ready
            else "add_missing_observation_time_public_fields_before_surface_refresh"
        ),
        "context_ready_for_v129_surface_refresh": ready,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "training_executed": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "model_artifact_created": False,
        "gate_change_recommended": False,
        "viewer_change_recommended": False,
        "replay_golden_change_recommended": False,
        "foundation_change_recommended": False,
        "claim_causality": False,
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
    }


def _record_missing_public_fields(record: Mapping[str, object]) -> list[str]:
    missing = []
    if not isinstance(record.get("observation_input"), Mapping) and not isinstance(
        record.get("decoded_observation"),
        Mapping,
    ):
        missing.append("trajectory_record.observation_input")
    if not isinstance(record.get("action_mask"), Mapping):
        missing.append("trajectory_record.action_mask")
    decoded = _mapping(record.get("decoded_observation"))
    if not decoded and isinstance(record.get("observation_input"), Mapping):
        try:
            decoded = _decode_observation(_mapping(record.get("observation_input")))
        except (TypeError, ValueError):
            missing.append("trajectory_record.observation_input.decode")
            return missing
    if decoded and not isinstance(decoded.get("patch"), Mapping):
        missing.append("decoded.patch")
    if decoded and not isinstance(decoded.get("navigation"), Mapping):
        missing.append("decoded.navigation")
    return missing


def _join_missing_example(row: Mapping[str, object]) -> dict[str, object]:
    provenance = _mapping(row.get("provenance"))
    return {
        "source_path": provenance.get("source_path"),
        "record_index": row.get("record_index"),
        "agent_id": provenance.get("agent_id"),
        "tick": row.get("tick"),
        "observation_digest": row.get("observation_digest"),
    }


def _direction_delta(action: str) -> tuple[int, int] | None:
    if action.endswith("_north"):
        return (0, -1)
    if action.endswith("_south"):
        return (0, 1)
    if action.endswith("_east"):
        return (1, 0)
    if action.endswith("_west"):
        return (-1, 0)
    return None


def _patch_cell(
    patch: Mapping[object, object],
    dx: int,
    dy: int,
) -> Mapping[str, object]:
    for key in ((dx, dy), f"{dx},{dy}", f"{dx}:{dy}"):
        value = patch.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _nav_bucket(
    navigation: Mapping[str, object],
    target: str,
    field: str,
) -> str:
    return _positive_bucket(_mapping(navigation.get(target)).get(field))


def _actual_feature_paths(dataset: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        {
            str(path)
            for row in dataset
            for path in _mapping(row.get("features")).keys()
        }
    )


def _allowed_feature_path(path: str) -> bool:
    return path == "candidate_action" or path.startswith(
        (
            "candidate_action_",
            "candidate_target_context.",
            "candidate_resource_context.",
            "candidate_neighborhood_context.",
        )
    )


def _allowed_source_path(path: str) -> bool:
    return path in {
        "candidate_row.candidate_action",
        "trajectory_record.action_mask",
        "trajectory_record.observation_input",
        "decoded.patch",
        "decoded.navigation",
    }


def _context_feature_family(path: str) -> str:
    if path == "candidate_action":
        return "action_identity"
    if path == "candidate_action_observation_legal":
        return "action_mask_legality"
    if path.startswith("candidate_action_"):
        return "action_semantics"
    if path.startswith(
        (
            "candidate_target_context.",
            "candidate_resource_context.",
            "candidate_neighborhood_context.",
        )
    ):
        return "target_resource_neighborhood"
    return _feature_family(path)


def _variance_family(
    varying: Mapping[str, object],
    family: str,
) -> dict[str, object]:
    paths = {
        path: payload
        for path, payload in sorted(varying.items())
        if _mapping(payload).get("field_family") == family
    }
    return {
        "field_family": family,
        "varying_paths": paths,
        "varying_path_count": len(paths),
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
            continue
        if current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    tokens.extend(path.replace("-", "_").split("_"))
    return [token for token in tokens if token]


def _branch_id(row: Mapping[str, object]) -> str | None:
    value = row.get("branch_id")
    return value if isinstance(value, str) and value else None


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _boolish(value: object) -> bool | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.5
    return "missing"


def _positive_bucket(value: object) -> str:
    number = _number(value)
    if number <= 0.0:
        return "none"
    if number < 0.25:
        return "low"
    if number < 0.6:
        return "mid"
    return "high"


def _signed_bucket(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "missing"
    number = float(value)
    if number < -0.5:
        return "negative"
    if number < 0.1:
        return "zero"
    if number < 0.5:
        return "low"
    return "high"


def _count_bucket(count: int) -> str:
    if count <= 0:
        return "zero"
    if count <= 2:
        return "low"
    if count <= 6:
        return "mid"
    return "high"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
