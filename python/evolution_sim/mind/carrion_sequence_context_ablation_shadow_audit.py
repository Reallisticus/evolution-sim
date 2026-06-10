from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path

from evolution_sim.mind.carrion_archive_override_autopsy import (
    DEFAULT_CARRION_SEEDS,
    DEFAULT_TICKS,
    DEFAULT_V149_CARRION_ARTIFACT_PATH,
    DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH,
)
from evolution_sim.mind.carrion_sequence_context_archive import (
    DEFAULT_DATASET_OUTPUT_PATH as DEFAULT_V151_DATASET_PATH,
    DEFAULT_HARMFUL_SUPPORT_SOURCES,
    DEFAULT_OUTPUT_PATH as DEFAULT_V151_REPORT_PATH,
    M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_POLICY,
    M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION,
    carrion_sequence_context_trainable_leakage_scan,
    load_json_report,
    load_jsonl_rows,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_sequence_context_ablation_shadow_audit_report_v1"
)
M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_sequence_context_ablation_shadow_audit_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v152-carrion-sequence-context-ablation-shadow-audit.json"
)
DEFAULT_SIMILAR_ACTION_MASK_THRESHOLD = 0.75
HYDRATION_REPRODUCTION_MARKER_FIELDS = (
    "drank_count",
    "ate_count",
    "animal_food_count",
    "reproduced_count",
    "reproduction_ready_after_count",
)


class CarrionSequenceContextAblationShadowAuditError(ValueError):
    pass


def write_carrion_sequence_context_ablation_shadow_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def build_carrion_sequence_context_ablation_shadow_audit_report_from_paths(
    *,
    v151_report_path: str | Path = DEFAULT_V151_REPORT_PATH,
    v151_dataset_path: str | Path = DEFAULT_V151_DATASET_PATH,
    shadow_artifact_path: str | Path = DEFAULT_V149_CARRION_ARTIFACT_PATH,
    shadow_train_eval_report_path: str | Path = (
        DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH
    ),
    **kwargs: object,
) -> dict[str, object]:
    return build_carrion_sequence_context_ablation_shadow_audit_report(
        v151_report=load_json_report(v151_report_path),
        v151_rows=load_jsonl_rows(v151_dataset_path),
        shadow_artifact_path=shadow_artifact_path,
        shadow_train_eval_report_path=shadow_train_eval_report_path,
        input_paths={
            "v151_report": v151_report_path,
            "v151_dataset": v151_dataset_path,
            "shadow_artifact": shadow_artifact_path,
            "shadow_train_eval_report": shadow_train_eval_report_path,
        },
        **kwargs,
    )


def build_carrion_sequence_context_ablation_shadow_audit_report(
    *,
    v151_report: Mapping[str, object],
    v151_rows: Sequence[Mapping[str, object]],
    audit_rows: Sequence[Mapping[str, object]] | None = None,
    generation_status: Mapping[str, object] | None = None,
    generation_evidence: Mapping[str, object] | None = None,
    seed_include: Sequence[int] | None = None,
    branch_index_include: Sequence[int] | None = None,
    row_index_include: Sequence[int] | None = None,
    row_index_start: int | None = None,
    row_index_count: int | None = None,
    shard_id: str | None = None,
    similar_action_mask_threshold: float = DEFAULT_SIMILAR_ACTION_MASK_THRESHOLD,
    shadow_live_probe: bool = False,
    shadow_artifact_path: str | Path | None = DEFAULT_V149_CARRION_ARTIFACT_PATH,
    shadow_train_eval_report_path: str | Path | None = (
        DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH
    ),
    shadow_probe_results: Mapping[str, object] | None = None,
    target_seeds: Sequence[int] = DEFAULT_CARRION_SEEDS,
    ticks: int = DEFAULT_TICKS,
    input_paths: Mapping[str, str | Path | None] | None = None,
) -> dict[str, object]:
    source_validation = validate_carrion_sequence_context_ablation_inputs(
        v151_report=v151_report,
        v151_rows=v151_rows,
    )
    if audit_rows is None:
        resolved_rows = _selected_audit_rows(
            rows=v151_rows,
            seed_include=seed_include,
            branch_index_include=branch_index_include,
            row_index_include=row_index_include,
            row_index_start=row_index_start,
            row_index_count=row_index_count,
        )
    else:
        resolved_rows = [dict(row) for row in audit_rows]
    if generation_status is None:
        generation_status = _generation_status(
            selected_rows=resolved_rows,
            total_v151_row_count=len(v151_rows),
            seed_include=seed_include,
            branch_index_include=branch_index_include,
            row_index_include=row_index_include,
            row_index_start=row_index_start,
            row_index_count=row_index_count,
            shard_id=shard_id,
        )
    if generation_evidence is None:
        generation_evidence = {
            "policy": "m3_carrion_sequence_context_ablation_shadow_audit_generation_evidence_v1",
            "shard_id": shard_id,
            "input_v151_exact_digest": v151_report.get("exact_digest"),
            "input_v151_dataset_digest": source_validation.get(
                "v151_dataset_digest"
            ),
        }
    return _finalize_report(
        v151_report=v151_report,
        v151_rows=v151_rows,
        audit_rows=resolved_rows,
        source_validation=source_validation,
        generation_status=generation_status,
        generation_evidence=generation_evidence,
        seed_include=seed_include,
        branch_index_include=branch_index_include,
        row_index_include=row_index_include,
        row_index_start=row_index_start,
        row_index_count=row_index_count,
        shard_id=shard_id,
        similar_action_mask_threshold=float(similar_action_mask_threshold),
        shadow_live_probe=bool(shadow_live_probe),
        shadow_artifact_path=shadow_artifact_path,
        shadow_train_eval_report_path=shadow_train_eval_report_path,
        shadow_probe_results=shadow_probe_results,
        target_seeds=target_seeds,
        ticks=int(ticks),
        input_paths=input_paths,
    )


def merge_carrion_sequence_context_ablation_shadow_audit_shards(
    *,
    v151_report: Mapping[str, object],
    v151_rows: Sequence[Mapping[str, object]],
    shard_reports: Sequence[Mapping[str, object]],
    shard_report_paths: Sequence[str | Path] = (),
    allow_partial_shard_evidence: bool = False,
    input_paths: Mapping[str, str | Path | None] | None = None,
) -> dict[str, object]:
    if not shard_reports:
        raise CarrionSequenceContextAblationShadowAuditError(
            "v152 carrion sequence-context ablation shard merge requires reports"
        )
    validation = validate_carrion_sequence_context_ablation_inputs(
        v151_report=v151_report,
        v151_rows=v151_rows,
    )
    merged_by_row_id: dict[str, dict[str, object]] = {}
    digests_by_row_id: dict[str, str] = {}
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
            raise CarrionSequenceContextAblationShadowAuditError(
                f"shard {shard_index} schema/policy/target/tick/min-floor mismatch"
            )
        status = _mapping(report.get("generation_status"))
        source_integrity = _mapping(report.get("source_integrity"))
        integrity_failures = [
            str(failure) for failure in _list(source_integrity.get("failures"))
        ]
        partial = _status_partial(status)
        if source_integrity.get("passed") is not True and not partial:
            raise CarrionSequenceContextAblationShadowAuditError(
                f"shard {shard_index} source integrity failed: {integrity_failures}"
            )
        source_name = _shard_source_name(report, shard_index=shard_index)
        if partial:
            summary = {
                "source": source_name,
                "source_path": _optional_index(source_paths, shard_index),
                "shard_id": _mapping(_mapping(report.get("inputs")).get("shard")).get(
                    "shard_id"
                ),
                "state": status.get("state"),
                "stop_reason": status.get("stop_reason"),
                "source_integrity_failures": integrity_failures,
            }
            if not allow_partial_shard_evidence:
                raise CarrionSequenceContextAblationShadowAuditError(
                    "partial shard evidence requires explicit partial merge: "
                    f"{summary}"
                )
            partial_sources.append(summary)
        rows = _list_of_mappings(report.get("audit_rows"))
        expected_digest = str(report.get("audit_row_digest", ""))
        actual_digest = stable_payload_digest(rows)
        if expected_digest and expected_digest != actual_digest:
            raise CarrionSequenceContextAblationShadowAuditError(
                f"shard {shard_index} audit row digest mismatch"
            )
        for row in rows:
            _add_merged_audit_row(
                merged_by_row_id=merged_by_row_id,
                digests_by_row_id=digests_by_row_id,
                row=row,
            )
        source_summaries.append(
            {
                "source": source_name,
                "source_path": _optional_index(source_paths, shard_index),
                "shard_id": _mapping(_mapping(report.get("inputs")).get("shard")).get(
                    "shard_id"
                ),
                "audit_row_count": len(rows),
                "audit_row_digest": expected_digest or actual_digest,
                "partial": bool(partial),
            }
        )
    expected_full_row_count = _int(validation.get("v151_row_count"))
    missing_full_rows = len(merged_by_row_id) < expected_full_row_count
    if missing_full_rows and not allow_partial_shard_evidence:
        raise CarrionSequenceContextAblationShadowAuditError(
            "partial shard evidence requires explicit partial merge: "
            f"merged_row_count={len(merged_by_row_id)} "
            f"expected_row_count={expected_full_row_count}"
        )
    if missing_full_rows:
        partial_sources.append(
            {
                "source": "merged_union",
                "state": "partial",
                "stop_reason": "merged_row_count_below_v151_dataset_row_count",
                "merged_row_count": len(merged_by_row_id),
                "expected_row_count": expected_full_row_count,
            }
        )
    rows = sorted(merged_by_row_id.values(), key=_audit_row_sort_key)
    merged_status = {
        "policy": "m3_carrion_sequence_context_ablation_shadow_audit_merged_generation_status_v1",
        "state": "partial" if partial_sources else "complete",
        "partial": bool(partial_sources),
        "stop_reason": "partial_shard_evidence" if partial_sources else None,
        "source_count": len(source_summaries),
        "partial_sources": sorted(
            partial_sources,
            key=lambda item: str(item.get("source", "")),
        ),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": list(
            _list(_mapping(merge_identity).get("target_carrion_seeds"))
        ),
        "ticks": _mapping(merge_identity).get("ticks"),
        "history_window": _mapping(merge_identity).get("history_window"),
        "min_prior_public_steps": _mapping(merge_identity).get(
            "min_prior_public_steps"
        ),
        "similar_action_mask_threshold": _mapping(merge_identity).get(
            "similar_action_mask_threshold"
        ),
        "audit_row_count": len(rows),
        "expected_v151_row_count": expected_full_row_count,
    }
    merged_evidence = {
        "policy": "m3_carrion_sequence_context_ablation_shadow_audit_shard_merge_generation_evidence_v1",
        "merge_mode": True,
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "source_count": len(source_summaries),
        "audit_row_digest": stable_payload_digest(rows),
    }
    report = build_carrion_sequence_context_ablation_shadow_audit_report(
        v151_report=v151_report,
        v151_rows=v151_rows,
        audit_rows=rows,
        generation_status=merged_status,
        generation_evidence=merged_evidence,
        similar_action_mask_threshold=float(
            _mapping(merge_identity).get(
                "similar_action_mask_threshold",
                DEFAULT_SIMILAR_ACTION_MASK_THRESHOLD,
            )
        ),
        shadow_live_probe=False,
        input_paths=input_paths,
    )
    report["shard_merge"] = {
        "policy": "m3_carrion_sequence_context_ablation_shadow_audit_shard_merge_v1",
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "sources": sorted(
            source_summaries,
            key=lambda item: (
                str(item.get("source", "")),
                str(item.get("source_path", "")),
                str(item.get("audit_row_digest", "")),
            ),
        ),
        "partial_sources": merged_status["partial_sources"],
        "duplicate_row_policy": "same_digest_allowed_conflict_rejected",
    }
    return report


def validate_carrion_sequence_context_ablation_inputs(
    *,
    v151_report: Mapping[str, object],
    v151_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    dataset_digest = stable_payload_digest(list(v151_rows))
    branch_results = _list_of_mappings(v151_report.get("branch_results"))
    branch_evidence_digest = stable_payload_digest(branch_results)
    dataset = _mapping(v151_report.get("dataset"))
    source_integrity = _mapping(v151_report.get("source_integrity"))
    generation_status = _mapping(v151_report.get("generation_status"))
    classification = _mapping(v151_report.get("classification"))
    leakage = _mapping(v151_report.get("leakage_scan"))
    replay = _mapping(v151_report.get("replay_verification"))

    if v151_report.get("schema_version") != (
        M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION
    ):
        failures.append("v151_schema_mismatch")
    if v151_report.get("policy") != M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_POLICY:
        failures.append("v151_policy_mismatch")
    if v151_report.get("diagnostics_only") is not True:
        failures.append("v151_diagnostics_only_not_true")
    if classification.get("primary") != (
        "m3_carrion_sequence_context_archive_separates_harmful_sources_no_training"
    ):
        failures.append("v151_classification_not_sequence_separated")
    if source_integrity.get("passed") is not True:
        failures.append("v151_source_integrity_not_passed")
    if _list(source_integrity.get("failures")):
        failures.append("v151_source_integrity_failures_present")
    if (
        generation_status.get("state") != "complete"
        or generation_status.get("partial") is True
    ):
        failures.append("v151_generation_not_complete")
    if leakage.get("passed") is not True:
        failures.append("v151_leakage_scan_not_passed")
    if replay.get("complete") is not True:
        failures.append("v151_replay_verification_not_complete")
    if dataset.get("dataset_digest") != dataset_digest:
        failures.append("v151_dataset_digest_mismatch")
    if _int(dataset.get("row_count")) != len(v151_rows):
        failures.append("v151_dataset_row_count_mismatch")
    if v151_report.get("branch_evidence_digest") != branch_evidence_digest:
        failures.append("v151_branch_evidence_digest_mismatch")
    if not v151_rows:
        failures.append("v151_dataset_rows_missing")
    for key, expected in (
        ("training_authorized", False),
        ("promotion_authorized", False),
        ("runtime_promotion_allowed", False),
        ("default_runtime_behavior_changed", False),
        ("runtime_action_selection_changed", False),
    ):
        if v151_report.get(key) is not expected:
            failures.append(f"v151_{key}_not_{str(expected).lower()}")
    observed_sources = {
        _harmful_key_from_row(row) for row in v151_rows if _is_harmful_row(row)
    }
    expected_sources = {_harmful_key(source) for source in DEFAULT_HARMFUL_SUPPORT_SOURCES}
    if expected_sources - observed_sources:
        failures.append("v151_harmful_support_sources_missing")
    if failures:
        raise CarrionSequenceContextAblationShadowAuditError(
            "v152 carrion sequence-context ablation input validation failed: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "policy": "m3_carrion_sequence_context_ablation_shadow_audit_input_validation_v1",
        "passed": True,
        "failures": [],
        "v151_report_digest": stable_payload_digest(v151_report),
        "v151_exact_digest": v151_report.get("exact_digest"),
        "v151_dataset_digest": dataset_digest,
        "v151_branch_evidence_digest": branch_evidence_digest,
        "v151_row_count": len(v151_rows),
    }


def _finalize_report(
    *,
    v151_report: Mapping[str, object],
    v151_rows: Sequence[Mapping[str, object]],
    audit_rows: Sequence[Mapping[str, object]],
    source_validation: Mapping[str, object],
    generation_status: Mapping[str, object],
    generation_evidence: Mapping[str, object],
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    row_index_include: Sequence[int] | None,
    row_index_start: int | None,
    row_index_count: int | None,
    shard_id: str | None,
    similar_action_mask_threshold: float,
    shadow_live_probe: bool,
    shadow_artifact_path: str | Path | None,
    shadow_train_eval_report_path: str | Path | None,
    shadow_probe_results: Mapping[str, object] | None,
    target_seeds: Sequence[int],
    ticks: int,
    input_paths: Mapping[str, str | Path | None] | None,
) -> dict[str, object]:
    rows = sorted([dict(row) for row in audit_rows], key=_audit_row_sort_key)
    leakage_scan = carrion_sequence_context_trainable_leakage_scan(rows)
    coverage = _coverage_report(
        rows=rows,
        target_seeds=target_seeds,
        strict_full_scope=_strict_full_scope(
            seed_include=seed_include,
            branch_index_include=branch_index_include,
            row_index_include=row_index_include,
            row_index_start=row_index_start,
            row_index_count=row_index_count,
            shard_id=shard_id,
            generation_status=generation_status,
        ),
    )
    comparator_availability = _comparator_availability_report(
        rows=rows,
        similar_action_mask_threshold=float(similar_action_mask_threshold),
    )
    ablations = _ablation_suite(rows)
    shadow_probe = _shadow_live_probe_report(
        enabled=bool(shadow_live_probe),
        rows=rows,
        artifact_path=shadow_artifact_path,
        train_eval_report_path=shadow_train_eval_report_path,
        target_seeds=target_seeds,
        ticks=int(ticks),
        supplied_result=shadow_probe_results,
    )
    source_integrity = _source_integrity(
        source_validation=source_validation,
        leakage_scan=leakage_scan,
        coverage=coverage,
        generation_status=generation_status,
    )
    classification = _classification(
        source_integrity=source_integrity,
        comparator_availability=comparator_availability,
        ablations=ablations,
    )
    audit_row_digest = stable_payload_digest(rows)
    contract = {
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "gate_relaxation": False,
        "trainer_effect": "none",
        "runtime_artifact_created": False,
        "runtime_path": "none_added_report_only",
        "trainable_dataset_emitted": False,
        "audited_trainable_surface": [
            "v151 public observation_input",
            "v151 public action_mask",
            "v151 prior finalized public sequence context",
            "action label as target only",
        ],
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
    }
    inputs = {
        **_input_path_payload(input_paths),
        "v151_report_digest": source_validation.get("v151_report_digest"),
        "v151_exact_digest": source_validation.get("v151_exact_digest"),
        "v151_dataset_digest": source_validation.get("v151_dataset_digest"),
        "v151_branch_evidence_digest": source_validation.get(
            "v151_branch_evidence_digest"
        ),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in target_seeds],
        "ticks": int(ticks),
        "history_window": _mapping(v151_report.get("inputs")).get("history_window"),
        "min_prior_public_steps": _mapping(v151_report.get("inputs")).get(
            "min_prior_public_steps"
        ),
        "similar_action_mask_threshold": float(similar_action_mask_threshold),
        "shard": {
            "shard_id": shard_id,
            "seed_include": None
            if seed_include is None
            else [int(seed) for seed in seed_include],
            "branch_index_include": None
            if branch_index_include is None
            else [int(index) for index in branch_index_include],
            "row_index_include": None
            if row_index_include is None
            else [int(index) for index in row_index_include],
            "row_index_start": row_index_start,
            "row_index_count": row_index_count,
        },
        "shadow_live_probe_requested": bool(shadow_live_probe),
    }
    report = {
        "schema_version": M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_SCHEMA_VERSION,
        "policy": M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_POLICY,
        "diagnostics_only": True,
        "contract": contract,
        "inputs": inputs,
        "generation_status": dict(generation_status),
        "generation_evidence": dict(generation_evidence),
        "source_validation": dict(source_validation),
        "source_integrity": source_integrity,
        "coverage": coverage,
        "comparator_availability": comparator_availability,
        "ablations": ablations,
        "shadow_live_probe": shadow_probe,
        "classification": {"primary": classification, "labels": [classification]},
        "audit_rows": rows,
        "audit_row_count": len(rows),
        "audit_row_digest": audit_row_digest,
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
            "v151_exact_digest": inputs["v151_exact_digest"],
            "v151_dataset_digest": inputs["v151_dataset_digest"],
            "v151_branch_evidence_digest": inputs["v151_branch_evidence_digest"],
            "target_carrion_seeds": inputs["target_carrion_seeds"],
            "ticks": inputs["ticks"],
            "history_window": inputs["history_window"],
            "min_prior_public_steps": inputs["min_prior_public_steps"],
            "similar_action_mask_threshold": inputs["similar_action_mask_threshold"],
        },
        "audit_row_digest": audit_row_digest,
        "source_integrity": source_integrity,
        "coverage": coverage,
        "comparator_availability": comparator_availability,
        "ablations": ablations,
        "shadow_live_probe": shadow_probe,
        "classification": report["classification"],
    }
    report["exact_digest"] = stable_payload_digest(exact_payload)
    report["provenance"] = {
        "contract_digest": stable_payload_digest(contract),
        "exact_digest_payload_policy": (
            "stable_payload_digest_of_v152_contract_inputs_comparators_ablations_and_shadow_probe_v1"
        ),
        "exact_digest": report["exact_digest"],
    }
    return report


def _selected_audit_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    row_index_include: Sequence[int] | None,
    row_index_start: int | None,
    row_index_count: int | None,
) -> list[dict[str, object]]:
    seed_filter = None if seed_include is None else {int(seed) for seed in seed_include}
    branch_filter = (
        None
        if branch_index_include is None
        else {int(index) for index in branch_index_include}
    )
    row_filter = (
        None if row_index_include is None else {int(index) for index in row_index_include}
    )
    if row_index_start is not None or row_index_count is not None:
        if row_index_start is None or row_index_count is None:
            raise CarrionSequenceContextAblationShadowAuditError(
                "--row-index-start and --row-index-count must be provided together"
            )
        if int(row_index_start) < 0 or int(row_index_count) < 0:
            raise CarrionSequenceContextAblationShadowAuditError(
                "row index start/count must be non-negative"
            )
        start = int(row_index_start)
        row_filter = set(row_filter or set())
        row_filter.update(range(start, start + int(row_index_count)))
    selected = []
    for fallback_index, row in enumerate(rows):
        metadata = _mapping(row.get("metadata"))
        row_index = _int(metadata.get("row_index"), default=fallback_index)
        seed = _int(metadata.get("seed"), default=-1)
        branch_index = _int(metadata.get("branch_index"), default=-1)
        if seed_filter is not None and seed not in seed_filter:
            continue
        if branch_filter is not None and branch_index not in branch_filter:
            continue
        if row_filter is not None and row_index not in row_filter:
            continue
        selected.append(dict(row))
    return sorted(selected, key=_audit_row_sort_key)


def _generation_status(
    *,
    selected_rows: Sequence[Mapping[str, object]],
    total_v151_row_count: int,
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    row_index_include: Sequence[int] | None,
    row_index_start: int | None,
    row_index_count: int | None,
    shard_id: str | None,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_sequence_context_ablation_shadow_audit_generation_status_v1",
        "state": "complete",
        "partial": False,
        "stop_reason": None,
        "target_fixture": "carrion_only",
        "selection_scope": "filtered_shard" if shard_id is not None else "full_or_filtered",
        "audit_row_count": len(selected_rows),
        "total_v151_row_count": int(total_v151_row_count),
        "seed_include": None
        if seed_include is None
        else [int(seed) for seed in seed_include],
        "branch_index_include": None
        if branch_index_include is None
        else [int(index) for index in branch_index_include],
        "row_index_include": None
        if row_index_include is None
        else [int(index) for index in row_index_include],
        "row_index_start": row_index_start,
        "row_index_count": row_index_count,
        "shard_id": shard_id,
    }


def _coverage_report(
    *,
    rows: Sequence[Mapping[str, object]],
    target_seeds: Sequence[int],
    strict_full_scope: bool,
) -> dict[str, object]:
    harmful_rows = [row for row in rows if _is_harmful_row(row)]
    safe_rows = [row for row in rows if not _is_harmful_row(row)]
    observed_harmful = {
        _harmful_key_from_row(row)
        for row in harmful_rows
        if _harmful_key_from_row(row) is not None
    }
    expected_harmful = {_harmful_key(source) for source in DEFAULT_HARMFUL_SUPPORT_SOURCES}
    rows_by_seed = Counter(
        _int(_mapping(row.get("metadata")).get("seed"), default=-1) for row in rows
    )
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
                "audit_row_count": len(seed_rows),
                "harmful_source_row_count": sum(
                    1 for row in seed_rows if _is_harmful_row(row)
                ),
                "safe_comparator_row_count": sum(
                    1 for row in seed_rows if not _is_harmful_row(row)
                ),
                "action_counts": dict(
                    sorted(Counter(_row_action(row) for row in seed_rows).items())
                ),
                "missing_is_blocker": bool(strict_full_scope and not seed_rows),
            }
        )
    return {
        "policy": "m3_carrion_sequence_context_ablation_shadow_audit_coverage_v1",
        "target_fixture": "carrion_only",
        "target_seeds": [int(seed) for seed in target_seeds],
        "strict_full_scope": bool(strict_full_scope),
        "audit_row_count": len(rows),
        "harmful_source_row_count": len(harmful_rows),
        "safe_comparator_row_count": len(safe_rows),
        "observed_harmful_source_count": len(observed_harmful),
        "target_harmful_source_count": len(expected_harmful),
        "missing_harmful_sources": sorted(expected_harmful - observed_harmful),
        "per_seed_support": per_seed,
        "missing_audit_row_seeds": [
            int(seed) for seed in target_seeds if rows_by_seed.get(int(seed), 0) <= 0
        ],
    }


def _comparator_availability_report(
    *,
    rows: Sequence[Mapping[str, object]],
    similar_action_mask_threshold: float,
) -> dict[str, object]:
    harmful_rows = sorted(
        [row for row in rows if _is_harmful_row(row)],
        key=_audit_row_sort_key,
    )
    safe_rows = [row for row in rows if not _is_harmful_row(row)]
    harmful_reports = []
    counts = Counter()
    for harmful in harmful_rows:
        action = _row_action(harmful)
        reason = _branch_reason(harmful)
        same_action = [row for row in safe_rows if _row_action(row) == action]
        exact_mask = [
            row
            for row in same_action
            if _action_mask_key(row) == _action_mask_key(harmful)
        ]
        similar_mask = [
            row
            for row in same_action
            if _action_mask_similarity(row, harmful) >= similar_action_mask_threshold
        ]
        same_reason = [row for row in same_action if _branch_reason(row) == reason]
        same_reason_exact = [
            row for row in same_reason if _action_mask_key(row) == _action_mask_key(harmful)
        ]
        exact_one_step = [
            row for row in same_action if _one_step_features_key(row) == _one_step_features_key(harmful)
        ]
        zero_prior_safe = [
            row for row in same_action if _prior_step_count(row) == 0
        ]
        nonzero_prior_harmful = [
            row
            for row in harmful_rows
            if row is not harmful
            and _row_action(row) == action
            and _prior_step_count(row) > 0
        ]
        near_exact = [
            row
            for row in same_action
            if (
                _action_mask_similarity(row, harmful) >= similar_action_mask_threshold
                and (
                    _branch_reason(row) == reason
                    or _one_step_observation_key(row) == _one_step_observation_key(harmful)
                )
            )
        ]
        report = {
            "source": _harmful_key_from_row(harmful),
            "label_action": action,
            "source_seed": _mapping(harmful.get("metadata")).get("seed"),
            "source_branch_reason": reason,
            "source_branch_id": _mapping(harmful.get("metadata")).get("branch_id"),
            "prior_step_count": _prior_step_count(harmful),
            "same_label_action_safe_comparator_count": len(same_action),
            "same_action_exact_mask_safe_comparator_count": len(exact_mask),
            "same_action_similar_mask_safe_comparator_count": len(similar_mask),
            "same_action_same_harmful_source_branch_reason_safe_comparator_count": len(
                same_reason
            ),
            "same_action_same_reason_exact_mask_safe_comparator_count": len(
                same_reason_exact
            ),
            "exact_one_step_safe_comparator_count": len(exact_one_step),
            "near_exact_safe_comparator_count": len(near_exact),
            "zero_prior_safe_same_action_comparator_count": len(zero_prior_safe),
            "nonzero_prior_harmful_same_action_comparator_count": len(
                nonzero_prior_harmful
            ),
            "exact_action_mask_available": bool(exact_mask),
            "similar_action_mask_available": bool(similar_mask),
            "same_harmful_source_branch_reason_available": bool(same_reason),
            "zero_prior_safe_comparator_available": bool(zero_prior_safe),
            "nonzero_prior_harmful_comparator_available": bool(nonzero_prior_harmful),
        }
        harmful_reports.append(report)
        if same_action:
            counts["same_label_action_available"] += 1
        if exact_mask:
            counts["same_action_exact_mask_available"] += 1
        if similar_mask:
            counts["same_action_similar_mask_available"] += 1
        if same_reason:
            counts["same_action_same_reason_available"] += 1
        if exact_one_step:
            counts["exact_one_step_available"] += 1
        if near_exact:
            counts["near_exact_available"] += 1
        if zero_prior_safe:
            counts["zero_prior_safe_available"] += 1
        if nonzero_prior_harmful:
            counts["nonzero_prior_harmful_available"] += 1
    return {
        "policy": "m3_carrion_sequence_context_ablation_shadow_audit_comparator_availability_v1",
        "similar_action_mask_threshold": float(similar_action_mask_threshold),
        "harmful_source_count": len(harmful_rows),
        "safe_comparator_count": len(safe_rows),
        "same_label_action_available_count": int(
            counts["same_label_action_available"]
        ),
        "same_action_exact_mask_available_count": int(
            counts["same_action_exact_mask_available"]
        ),
        "same_action_similar_mask_available_count": int(
            counts["same_action_similar_mask_available"]
        ),
        "same_action_same_harmful_source_branch_reason_available_count": int(
            counts["same_action_same_reason_available"]
        ),
        "exact_one_step_available_count": int(counts["exact_one_step_available"]),
        "near_exact_available_count": int(counts["near_exact_available"]),
        "zero_prior_safe_available_count": int(counts["zero_prior_safe_available"]),
        "nonzero_prior_harmful_available_count": int(
            counts["nonzero_prior_harmful_available"]
        ),
        "zero_prior_safe_same_action_comparator_total": sum(
            _int(item.get("zero_prior_safe_same_action_comparator_count"))
            for item in harmful_reports
        ),
        "nonzero_prior_harmful_same_action_comparator_total": sum(
            _int(item.get("nonzero_prior_harmful_same_action_comparator_count"))
            for item in harmful_reports
        ),
        "harmful_source_comparators": harmful_reports,
    }


def _ablation_suite(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    reports = {
        "one_step_only_features": _ablation_report(
            rows=rows,
            mode="one_step_only_features",
            key_fn=_one_step_features_key,
            trainable_surface=(
                "public current observation_input and public action_mask only"
            ),
        ),
        "prior_length_only_features": _ablation_report(
            rows=rows,
            mode="prior_length_only_features",
            key_fn=_prior_length_features_key,
            trainable_surface="prior finalized public sequence length only",
        ),
        "public_hydration_reproduction_marker_features": _ablation_report(
            rows=rows,
            mode="public_hydration_reproduction_marker_features",
            key_fn=_hydration_reproduction_marker_features_key,
            trainable_surface=(
                "public hydration/reproduction marker counts from prior sequence; "
                "prior length excluded"
            ),
        ),
        "full_public_prior_sequence_features": _ablation_report(
            rows=rows,
            mode="full_public_prior_sequence_features",
            key_fn=_full_prior_sequence_features_key,
            trainable_surface="full prior finalized public sequence context",
        ),
    }
    length = reports["prior_length_only_features"]
    markers = reports["public_hydration_reproduction_marker_features"]
    full = reports["full_public_prior_sequence_features"]
    coarse_prior_presence_explains_full = (
        full.get("all_harmful_sources_separated") is True
        and length.get("all_harmful_sources_separated") is True
        and markers.get("all_harmful_sources_separated") is not True
    )
    one_step_support_limited_explains_full = (
        full.get("all_harmful_sources_separated") is True
        and reports["one_step_only_features"].get("all_harmful_sources_separated")
        is True
    )
    strong_sequence_signal = (
        full.get("all_harmful_sources_separated") is True
        and markers.get("all_harmful_sources_separated") is True
        and length.get("all_harmful_sources_separated") is not True
        and reports["one_step_only_features"].get("all_harmful_sources_separated")
        is not True
    )
    return {
        "policy": "m3_carrion_sequence_context_ablation_shadow_audit_ablation_suite_v1",
        "modes": reports,
        "coarse_prior_presence_explains_full_sequence_separation": bool(
            coarse_prior_presence_explains_full
        ),
        "one_step_support_limited_explains_full_sequence_separation": bool(
            one_step_support_limited_explains_full
        ),
        "strong_hydration_reproduction_sequence_signal": bool(
            strong_sequence_signal
        ),
        "interpretation": (
            "Ablations compare each harmful support label only against safe "
            "comparators with the same label action. Prior-length-only "
            "separation or one-step-only separation without marker separation "
            "is classified as support-limited evidence, not strong sequence "
            "evidence."
        ),
    }


def _ablation_report(
    *,
    rows: Sequence[Mapping[str, object]],
    mode: str,
    key_fn: object,
    trainable_surface: str,
) -> dict[str, object]:
    harmful_rows = sorted(
        [row for row in rows if _is_harmful_row(row)],
        key=_audit_row_sort_key,
    )
    safe_rows = [row for row in rows if not _is_harmful_row(row)]
    separated = 0
    matched_safe_total = 0
    same_action_comparator_total = 0
    reports = []
    for harmful in harmful_rows:
        action = _row_action(harmful)
        safe_same_action = [row for row in safe_rows if _row_action(row) == action]
        harmful_key = key_fn(harmful)  # type: ignore[operator]
        matches = [
            row for row in safe_same_action if key_fn(row) == harmful_key  # type: ignore[operator]
        ]
        is_separated = bool(safe_same_action) and not matches
        if is_separated:
            separated += 1
        matched_safe_total += len(matches)
        same_action_comparator_total += len(safe_same_action)
        reports.append(
            {
                "source": _harmful_key_from_row(harmful),
                "label_action": action,
                "source_seed": _mapping(harmful.get("metadata")).get("seed"),
                "source_branch_reason": _branch_reason(harmful),
                "prior_step_count": _prior_step_count(harmful),
                "same_action_safe_comparator_count": len(safe_same_action),
                "same_ablation_key_safe_comparator_count": len(matches),
                "separates_harmful_from_same_action_safe": bool(is_separated),
                "ablation_key_digest": stable_payload_digest(harmful_key),
            }
        )
    return {
        "policy": "m3_carrion_sequence_context_ablation_shadow_audit_ablation_mode_v1",
        "mode": mode,
        "trainable_surface": trainable_surface,
        "harmful_source_count": len(harmful_rows),
        "same_action_safe_comparator_count": int(same_action_comparator_total),
        "same_ablation_key_safe_comparator_count": int(matched_safe_total),
        "separated_harmful_source_count": int(separated),
        "all_harmful_sources_separated": (
            bool(harmful_rows) and separated == len(harmful_rows)
        ),
        "harmful_source_results": reports,
    }


def _source_integrity(
    *,
    source_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    coverage: Mapping[str, object],
    generation_status: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if source_validation.get("passed") is not True:
        failures.append("input_validation_failed")
    if leakage_scan.get("passed") is not True:
        failures.append("trainable_input_leakage_detected")
    if _status_partial(generation_status):
        failures.append("partial_ablation_evidence")
    if int(coverage.get("audit_row_count", 0)) <= 0:
        failures.append("no_audit_rows")
    if coverage.get("strict_full_scope") is True:
        if _list(coverage.get("missing_harmful_sources")):
            failures.append("target_harmful_support_sources_missing")
        if _list(coverage.get("missing_audit_row_seeds")):
            failures.append("target_seed_support_missing")
    return {
        "policy": "m3_carrion_sequence_context_ablation_shadow_audit_source_integrity_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "input_validation_passed": source_validation.get("passed") is True,
        "leakage_scan_passed": leakage_scan.get("passed") is True,
        "generation_state": generation_status.get("state"),
        "audit_row_count": coverage.get("audit_row_count"),
        "strict_full_scope": coverage.get("strict_full_scope"),
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    comparator_availability: Mapping[str, object],
    ablations: Mapping[str, object],
) -> str:
    if source_integrity.get("passed") is not True:
        return "m3_carrion_sequence_context_ablation_source_integrity_failed_no_training"
    modes = _mapping(ablations.get("modes"))
    full = _mapping(modes.get("full_public_prior_sequence_features"))
    length = _mapping(modes.get("prior_length_only_features"))
    markers = _mapping(modes.get("public_hydration_reproduction_marker_features"))
    one_step = _mapping(modes.get("one_step_only_features"))
    harmful_count = _int(comparator_availability.get("harmful_source_count"))
    same_action_count = _int(
        comparator_availability.get("same_label_action_available_count")
    )
    if harmful_count <= 0 or same_action_count < harmful_count:
        return "m3_carrion_sequence_context_ablation_comparator_support_insufficient_no_training"
    if full.get("all_harmful_sources_separated") is not True:
        return "m3_carrion_sequence_context_ablation_no_sequence_evidence_no_training"
    if one_step.get("all_harmful_sources_separated") is True:
        return "m3_carrion_sequence_context_ablation_one_step_and_prior_presence_support_limited_no_training"
    if (
        markers.get("all_harmful_sources_separated") is True
        and length.get("all_harmful_sources_separated") is not True
    ):
        return "m3_carrion_sequence_context_ablation_strong_sequence_evidence_no_training"
    if (
        length.get("all_harmful_sources_separated") is True
        and markers.get("all_harmful_sources_separated") is not True
    ):
        return "m3_carrion_sequence_context_ablation_coarse_prior_presence_evidence_no_training"
    if (
        markers.get("all_harmful_sources_separated") is True
        and length.get("all_harmful_sources_separated") is True
    ):
        return "m3_carrion_sequence_context_ablation_ambiguous_sequence_and_prior_presence_evidence_no_training"
    return "m3_carrion_sequence_context_ablation_mixed_or_weak_sequence_evidence_no_training"


def _shadow_live_probe_report(
    *,
    enabled: bool,
    rows: Sequence[Mapping[str, object]],
    artifact_path: str | Path | None,
    train_eval_report_path: str | Path | None,
    target_seeds: Sequence[int],
    ticks: int,
    supplied_result: Mapping[str, object] | None,
) -> dict[str, object]:
    if supplied_result is not None:
        result = dict(supplied_result)
        result.setdefault("policy", "m3_carrion_sequence_context_shadow_live_probe_supplied_result_v1")
        result.setdefault("enabled", bool(enabled))
        result.setdefault("runtime_artifact_created", False)
        return result
    harmful_rows = [row for row in rows if _is_harmful_row(row)]
    signatures = _shadow_suppression_signatures(harmful_rows)
    if not enabled:
        return {
            "policy": "m3_carrion_sequence_context_shadow_live_probe_v1",
            "enabled": False,
            "status": "not_run",
            "not_run_reason": "shadow_live_probe_not_requested",
            "runtime_artifact_created": False,
            "suppression_scope": (
                "only the three v150 harmful support labels under their v151 "
                "public context signatures when explicitly requested"
            ),
            "suppression_signature_count": len(signatures),
            "suppression_signature_digest": stable_payload_digest(signatures),
        }
    if artifact_path is None or train_eval_report_path is None:
        raise CarrionSequenceContextAblationShadowAuditError(
            "shadow live probe requires artifact and train/eval report paths"
        )
    from evolution_sim.mind.carrion_archive_override_autopsy import (
        _rerun_carrion_case,
    )
    from evolution_sim.mind.support_gated_residual import (
        load_support_gated_residual_artifact,
    )

    artifact = load_support_gated_residual_artifact(artifact_path)
    train_eval = load_json_report(train_eval_report_path)
    shadow_artifact, suppression = _shadow_suppressed_artifact(
        artifact=artifact,
        signatures=signatures,
    )
    seed_reports = []
    for seed in target_seeds:
        baseline = _rerun_carrion_case(
            seed=int(seed),
            ticks=int(ticks),
            artifact=artifact,
            seed_delta={},
        )
        shadow = _rerun_carrion_case(
            seed=int(seed),
            ticks=int(ticks),
            artifact=shadow_artifact,
            seed_delta={},
        )
        baseline_summary = _mapping(baseline.get("summary"))
        shadow_summary = _mapping(shadow.get("summary"))
        seed_reports.append(
            {
                "fixture": "carrion_only",
                "seed": int(seed),
                "baseline_summary": dict(baseline_summary),
                "shadow_summary": dict(shadow_summary),
                "births_delta_vs_baseline": _int(shadow_summary.get("births"))
                - _int(baseline_summary.get("births")),
                "resolved_invalid_delta_vs_baseline": _int(
                    shadow_summary.get("resolved_invalid_action_count")
                )
                - _int(baseline_summary.get("resolved_invalid_action_count")),
                "alive_delta_vs_baseline": _int(shadow_summary.get("alive_agents"))
                - _int(baseline_summary.get("alive_agents")),
                "baseline_applied_override_count": baseline.get(
                    "applied_override_count"
                ),
                "shadow_applied_override_count": shadow.get("applied_override_count"),
            }
        )
    total_birth_delta = sum(
        _int(report.get("births_delta_vs_baseline")) for report in seed_reports
    )
    total_invalid_delta = sum(
        _int(report.get("resolved_invalid_delta_vs_baseline"))
        for report in seed_reports
    )
    return {
        "policy": "m3_carrion_sequence_context_shadow_live_probe_v1",
        "enabled": True,
        "status": "complete",
        "runtime_artifact_created": False,
        "suppression_scope": (
            "in-memory support-example suppression for the three v150 harmful "
            "support labels identified by v151 public context signatures"
        ),
        "suppression": suppression,
        "target_fixture": "carrion_only",
        "target_seeds": [int(seed) for seed in target_seeds],
        "ticks": int(ticks),
        "train_eval_report_digest": stable_payload_digest(train_eval),
        "artifact_digest": stable_payload_digest(artifact),
        "shadow_artifact_digest": stable_payload_digest(shadow_artifact),
        "seed_reports": seed_reports,
        "aggregate": {
            "births_delta_vs_baseline": int(total_birth_delta),
            "resolved_invalid_delta_vs_baseline": int(total_invalid_delta),
            "birth_blocker_improved": total_birth_delta > 0,
            "resolved_invalid_blocker_improved": total_invalid_delta < 0,
        },
    }


def _shadow_suppression_signatures(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    signatures = []
    for row in sorted(rows, key=_audit_row_sort_key):
        metadata = _mapping(row.get("metadata"))
        trainable = _mapping(row.get("trainable"))
        features = _mapping(trainable.get("features"))
        label = _mapping(trainable.get("label"))
        signature = {
            "label_action": label.get("action"),
            "source_dataset_row_index": metadata.get("source_dataset_row_index"),
            "source_seed": metadata.get("seed"),
            "source_branch_reason": metadata.get("branch_reason"),
            "public_context_signature_digest": stable_payload_digest(
                {
                    "observation_input": features.get("observation_input"),
                    "action_mask": _bool_action_mask(features.get("action_mask")),
                    "prior_public_sequence_context": _list_of_mappings(
                        features.get("prior_public_sequence_context")
                    ),
                    "label_action": label.get("action"),
                }
            ),
        }
        signatures.append(signature)
    return signatures


def _shadow_suppressed_artifact(
    *,
    artifact: Mapping[str, object],
    signatures: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    targets = {
        (
            _int(signature.get("source_dataset_row_index"), default=-1),
            str(signature.get("label_action", "")),
        )
        for signature in signatures
    }
    shadow = deepcopy(dict(artifact))
    kept = []
    removed = []
    for example in _list_of_mappings(artifact.get("support_examples")):
        key = (_int(example.get("example_index"), default=-1), str(example.get("action", "")))
        if key in targets:
            removed.append(dict(example))
        else:
            kept.append(dict(example))
    shadow["support_examples"] = kept
    shadow["training_row_count"] = len(kept)
    shadow["support_action_counts"] = dict(
        sorted(Counter(str(item.get("action", "")) for item in kept).items())
    )
    return shadow, {
        "policy": "m3_carrion_sequence_context_shadow_support_label_suppression_v1",
        "requested_signature_count": len(signatures),
        "requested_signature_digest": stable_payload_digest(list(signatures)),
        "removed_support_example_count": len(removed),
        "removed_support_examples": [
            {
                "example_index": item.get("example_index"),
                "action": item.get("action"),
                "mode": item.get("mode"),
            }
            for item in removed
        ],
        "kept_support_example_count": len(kept),
    }


def _validate_shard_contract(report: Mapping[str, object], *, shard_index: int) -> None:
    if report.get("schema_version") != (
        M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_SCHEMA_VERSION
    ):
        raise CarrionSequenceContextAblationShadowAuditError(
            f"shard {shard_index} schema mismatch"
        )
    if report.get("policy") != M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_POLICY:
        raise CarrionSequenceContextAblationShadowAuditError(
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
            raise CarrionSequenceContextAblationShadowAuditError(
                f"shard {shard_index} {key} mismatch"
            )


def _merge_identity(report: Mapping[str, object]) -> dict[str, object]:
    inputs = _mapping(report.get("inputs"))
    return {
        "schema_version": report.get("schema_version"),
        "policy": report.get("policy"),
        "v151_exact_digest": inputs.get("v151_exact_digest"),
        "v151_dataset_digest": inputs.get("v151_dataset_digest"),
        "v151_branch_evidence_digest": inputs.get("v151_branch_evidence_digest"),
        "target_carrion_seeds": list(_list(inputs.get("target_carrion_seeds"))),
        "ticks": inputs.get("ticks"),
        "history_window": inputs.get("history_window"),
        "min_prior_public_steps": inputs.get("min_prior_public_steps"),
        "similar_action_mask_threshold": inputs.get("similar_action_mask_threshold"),
        "shadow_live_probe_requested": inputs.get("shadow_live_probe_requested"),
    }


def _add_merged_audit_row(
    *,
    merged_by_row_id: dict[str, dict[str, object]],
    digests_by_row_id: dict[str, str],
    row: Mapping[str, object],
) -> None:
    row_id = _audit_row_id(row)
    digest = stable_payload_digest(row)
    prior = digests_by_row_id.get(row_id)
    if prior is not None:
        if prior != digest:
            raise CarrionSequenceContextAblationShadowAuditError(
                f"duplicate audit row with different digest: {row_id}"
            )
        return
    merged_by_row_id[row_id] = dict(row)
    digests_by_row_id[row_id] = digest


def _strict_full_scope(
    *,
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    row_index_include: Sequence[int] | None,
    row_index_start: int | None,
    row_index_count: int | None,
    shard_id: str | None,
    generation_status: Mapping[str, object],
) -> bool:
    if _status_partial(generation_status):
        return True
    if generation_status.get("policy") == (
        "m3_carrion_sequence_context_ablation_shadow_audit_merged_generation_status_v1"
    ):
        return True
    return (
        seed_include is None
        and branch_index_include is None
        and row_index_include is None
        and row_index_start is None
        and row_index_count is None
        and shard_id is None
    )


def _row_action(row: Mapping[str, object]) -> str:
    metadata = _mapping(row.get("metadata"))
    label = _mapping(_mapping(row.get("trainable")).get("label"))
    return str(metadata.get("label_action") or label.get("action") or "")


def _is_harmful_row(row: Mapping[str, object]) -> bool:
    return _mapping(row.get("metadata")).get("harmful_source") is True


def _branch_reason(row: Mapping[str, object]) -> str:
    return str(_mapping(row.get("metadata")).get("branch_reason", ""))


def _prior_context(row: Mapping[str, object]) -> list[dict[str, object]]:
    features = _mapping(_mapping(row.get("trainable")).get("features"))
    return _list_of_mappings(features.get("prior_public_sequence_context"))


def _prior_step_count(row: Mapping[str, object]) -> int:
    markers = _mapping(_mapping(row.get("metadata")).get("hydration_reproduction_markers"))
    if "prior_step_count" in markers:
        return _int(markers.get("prior_step_count"))
    return len(_prior_context(row))


def _one_step_features_key(row: Mapping[str, object]) -> dict[str, object]:
    features = _mapping(_mapping(row.get("trainable")).get("features"))
    return {
        "observation_input": features.get("observation_input"),
        "action_mask": _bool_action_mask(features.get("action_mask")),
    }


def _one_step_observation_key(row: Mapping[str, object]) -> object:
    features = _mapping(_mapping(row.get("trainable")).get("features"))
    return features.get("observation_input")


def _prior_length_features_key(row: Mapping[str, object]) -> dict[str, object]:
    return {"prior_step_count": _prior_step_count(row)}


def _hydration_reproduction_marker_features_key(row: Mapping[str, object]) -> dict[str, object]:
    markers = _mapping(_mapping(row.get("metadata")).get("hydration_reproduction_markers"))
    return {
        key: _int(markers.get(key))
        for key in HYDRATION_REPRODUCTION_MARKER_FIELDS
    }


def _full_prior_sequence_features_key(row: Mapping[str, object]) -> dict[str, object]:
    return {"prior_public_sequence_context": _prior_context(row)}


def _action_mask_key(row: Mapping[str, object]) -> tuple[tuple[str, bool], ...]:
    features = _mapping(_mapping(row.get("trainable")).get("features"))
    return tuple(sorted(_bool_action_mask(features.get("action_mask")).items()))


def _action_mask_similarity(
    row_a: Mapping[str, object],
    row_b: Mapping[str, object],
) -> float:
    mask_a = {action for action, allowed in _action_mask_key(row_a) if allowed}
    mask_b = {action for action, allowed in _action_mask_key(row_b) if allowed}
    union = mask_a | mask_b
    if not union:
        return 1.0
    return len(mask_a & mask_b) / len(union)


def _audit_row_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    metadata = _mapping(row.get("metadata"))
    return (
        _int(metadata.get("row_index"), default=10**9),
        _int(metadata.get("source_dataset_row_index"), default=10**9),
        str(metadata.get("branch_id", "")),
        _row_action(row),
        stable_payload_digest(row),
    )


def _audit_row_id(row: Mapping[str, object]) -> str:
    metadata = _mapping(row.get("metadata"))
    return ":".join(
        [
            str(metadata.get("source_dataset_row_index", "")),
            str(metadata.get("branch_id", "")),
            _row_action(row),
        ]
    )


def _harmful_key(source: Mapping[str, object]) -> str:
    return ":".join(
        [
            str(source.get("source_seed", "")),
            str(source.get("source_branch_reason", "")),
            str(source.get("label_action", "")),
            str(source.get("source_branch_id", "")),
        ]
    )


def _harmful_key_from_row(row: Mapping[str, object]) -> str | None:
    metadata = _mapping(row.get("metadata"))
    harmful = _mapping(metadata.get("harmful_source_evidence"))
    if harmful:
        return _harmful_key(harmful)
    if metadata.get("harmful_source") is not True:
        return None
    return _harmful_key(
        {
            "source_seed": metadata.get("seed"),
            "source_branch_reason": metadata.get("branch_reason"),
            "label_action": _row_action(row),
            "source_branch_id": metadata.get("branch_id"),
        }
    )


def _bool_action_mask(value: object) -> dict[str, bool]:
    return {
        str(key): bool(allowed)
        for key, allowed in sorted(_mapping(value).items(), key=lambda item: str(item[0]))
    }


def _input_path_payload(
    input_paths: Mapping[str, str | Path | None] | None,
) -> dict[str, object]:
    if input_paths is None:
        return {}
    return {
        "input_paths": {
            str(key): None if value is None else str(value)
            for key, value in sorted(input_paths.items())
        }
    }


def _shard_source_name(report: Mapping[str, object], *, shard_index: int) -> str:
    shard = _mapping(_mapping(report.get("inputs")).get("shard"))
    shard_id = shard.get("shard_id")
    if shard_id:
        return f"shard:{shard_id}"
    return f"shard_index:{shard_index}"


def _status_partial(status: Mapping[str, object]) -> bool:
    return status.get("partial") is True or str(status.get("state", "")) == "partial"


def _optional_index(values: Sequence[object], index: int) -> object | None:
    return values[index] if index < len(values) else None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _list_of_mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(default)
