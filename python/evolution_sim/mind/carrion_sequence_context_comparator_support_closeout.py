from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from evolution_sim.mind.carrion_archive_override_autopsy import DEFAULT_CARRION_SEEDS
from evolution_sim.mind.carrion_sequence_context_ablation_shadow_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V152_REPORT_PATH,
    DEFAULT_SIMILAR_ACTION_MASK_THRESHOLD,
    M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_POLICY,
    M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_SCHEMA_VERSION,
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

M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION = (
    "m3_carrion_sequence_context_comparator_support_closeout_report_v1"
)
M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY = (
    "diagnostics_only_m3_carrion_sequence_context_comparator_support_closeout_v1"
)
M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_EVIDENCE_CHUNK_SCHEMA_VERSION = (
    "m3_carrion_sequence_context_comparator_evidence_chunk_v1"
)
M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_EVIDENCE_CHUNK_POLICY = (
    "diagnostics_only_m3_carrion_sequence_context_comparator_evidence_chunk_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v153-carrion-sequence-context-comparator-support-closeout.json"
)
DEFAULT_COMPARATOR_CHUNK_DIR = Path(
    "output/mind/mind-v3-v153-carrion-sequence-context-comparator-support-chunks"
)
EXPECTED_V151_CLASSIFICATION = (
    "m3_carrion_sequence_context_archive_separates_harmful_sources_no_training"
)
EXPECTED_V152_CLASSIFICATIONS = frozenset(
    {
        "m3_carrion_sequence_context_ablation_one_step_and_prior_presence_support_limited_no_training",
        "m3_carrion_sequence_context_ablation_coarse_prior_presence_evidence_no_training",
    }
)


class CarrionSequenceContextComparatorSupportCloseoutError(ValueError):
    pass


def write_carrion_sequence_context_comparator_support_closeout_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def build_carrion_sequence_context_comparator_support_closeout_report_from_paths(
    *,
    v151_report_path: str | Path = DEFAULT_V151_REPORT_PATH,
    v151_dataset_path: str | Path = DEFAULT_V151_DATASET_PATH,
    v152_report_path: str | Path = DEFAULT_V152_REPORT_PATH,
    **kwargs: object,
) -> dict[str, object]:
    return build_carrion_sequence_context_comparator_support_closeout_report(
        v151_report=load_json_report(v151_report_path),
        v151_rows=load_jsonl_rows(v151_dataset_path),
        v152_report=load_json_report(v152_report_path),
        input_paths={
            "v151_report": v151_report_path,
            "v151_dataset": v151_dataset_path,
            "v152_report": v152_report_path,
        },
        **kwargs,
    )


def build_carrion_sequence_context_comparator_support_closeout_report(
    *,
    v151_report: Mapping[str, object],
    v151_rows: Sequence[Mapping[str, object]],
    v152_report: Mapping[str, object],
    comparator_evidence_rows: Sequence[Mapping[str, object]] | None = None,
    generation_status: Mapping[str, object] | None = None,
    generation_evidence: Mapping[str, object] | None = None,
    seed_include: Sequence[int] | None = None,
    branch_index_include: Sequence[int] | None = None,
    row_index_include: Sequence[int] | None = None,
    row_index_start: int | None = None,
    row_index_count: int | None = None,
    shard_id: str | None = None,
    similar_action_mask_threshold: float = DEFAULT_SIMILAR_ACTION_MASK_THRESHOLD,
    comparator_chunk_dir: str | Path | None = None,
    input_paths: Mapping[str, str | Path | None] | None = None,
) -> dict[str, object]:
    source_validation = validate_carrion_sequence_context_comparator_inputs(
        v151_report=v151_report,
        v151_rows=v151_rows,
        v152_report=v152_report,
    )
    harmful_sources = [dict(source) for source in DEFAULT_HARMFUL_SUPPORT_SOURCES]
    if comparator_evidence_rows is None:
        generated = generate_carrion_sequence_context_comparator_evidence(
            v151_rows=v151_rows,
            harmful_sources=harmful_sources,
            seed_include=seed_include,
            branch_index_include=branch_index_include,
            row_index_include=row_index_include,
            row_index_start=row_index_start,
            row_index_count=row_index_count,
            shard_id=shard_id,
            similar_action_mask_threshold=float(similar_action_mask_threshold),
            comparator_chunk_dir=comparator_chunk_dir,
        )
        rows = _list_of_mappings(generated.get("comparator_evidence_rows"))
        resolved_status = _mapping(generated.get("generation_status"))
        resolved_evidence = dict(generated)
        resolved_evidence.pop("comparator_evidence_rows", None)
    else:
        rows = [dict(row) for row in comparator_evidence_rows]
        resolved_status = dict(
            generation_status
            or _generation_status(
                selected_candidate_count=_selected_candidate_count(rows),
                selected_candidate_row_indexes=_selected_candidate_row_indexes(rows),
                total_v151_row_count=len(v151_rows),
                seed_include=seed_include,
                branch_index_include=branch_index_include,
                row_index_include=row_index_include,
                row_index_start=row_index_start,
                row_index_count=row_index_count,
                shard_id=shard_id,
            )
        )
        resolved_evidence = dict(
            generation_evidence
            or {
                "policy": "m3_carrion_sequence_context_comparator_precomputed_evidence_v1",
                "shard_id": shard_id,
            }
        )
    return _finalize_report(
        v151_report=v151_report,
        v151_rows=v151_rows,
        v152_report=v152_report,
        source_validation=source_validation,
        harmful_sources=harmful_sources,
        comparator_evidence_rows=rows,
        generation_status=resolved_status,
        generation_evidence=resolved_evidence,
        seed_include=seed_include,
        branch_index_include=branch_index_include,
        row_index_include=row_index_include,
        row_index_start=row_index_start,
        row_index_count=row_index_count,
        shard_id=shard_id,
        similar_action_mask_threshold=float(similar_action_mask_threshold),
        comparator_chunk_dir=comparator_chunk_dir,
        input_paths=input_paths,
    )


def generate_carrion_sequence_context_comparator_evidence(
    *,
    v151_rows: Sequence[Mapping[str, object]],
    harmful_sources: Sequence[Mapping[str, object]],
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    row_index_include: Sequence[int] | None,
    row_index_start: int | None,
    row_index_count: int | None,
    shard_id: str | None,
    similar_action_mask_threshold: float,
    comparator_chunk_dir: str | Path | None,
) -> dict[str, object]:
    harmful_rows = _target_harmful_rows(v151_rows, harmful_sources)
    candidate_rows = _selected_candidate_rows(
        rows=v151_rows,
        seed_include=seed_include,
        branch_index_include=branch_index_include,
        row_index_include=row_index_include,
        row_index_start=row_index_start,
        row_index_count=row_index_count,
    )
    evidence_rows: list[dict[str, object]] = []
    for harmful in harmful_rows:
        for candidate in candidate_rows:
            evidence_rows.append(
                _comparator_evidence_row(
                    harmful=harmful,
                    candidate=candidate,
                    similar_action_mask_threshold=float(similar_action_mask_threshold),
                )
            )
    evidence_rows = sorted(evidence_rows, key=_evidence_row_sort_key)
    if comparator_chunk_dir is not None:
        write_carrion_sequence_context_comparator_evidence_chunks(
            evidence_rows=evidence_rows,
            chunk_dir=comparator_chunk_dir,
        )
    status = _generation_status(
        selected_candidate_count=len(candidate_rows),
        selected_candidate_row_indexes=[
            _row_index(row, fallback=index) for index, row in enumerate(candidate_rows)
        ],
        total_v151_row_count=len(v151_rows),
        seed_include=seed_include,
        branch_index_include=branch_index_include,
        row_index_include=row_index_include,
        row_index_start=row_index_start,
        row_index_count=row_index_count,
        shard_id=shard_id,
    )
    return {
        "policy": "m3_carrion_sequence_context_comparator_evidence_generation_v1",
        "shard_id": shard_id,
        "comparator_evidence_rows": evidence_rows,
        "comparator_evidence_row_count": len(evidence_rows),
        "comparator_evidence_digest": stable_payload_digest(evidence_rows),
        "generation_status": status,
        "target_harmful_support_sources": [dict(source) for source in harmful_sources],
        "similar_action_mask_threshold": float(similar_action_mask_threshold),
        "comparator_chunk_dir": (
            None if comparator_chunk_dir is None else str(comparator_chunk_dir)
        ),
    }


def merge_carrion_sequence_context_comparator_support_closeout_shards(
    *,
    v151_report: Mapping[str, object],
    v151_rows: Sequence[Mapping[str, object]],
    v152_report: Mapping[str, object],
    shard_reports: Sequence[Mapping[str, object]],
    shard_report_paths: Sequence[str | Path] = (),
    shard_chunk_dirs: Sequence[str | Path] = (),
    allow_partial_shard_evidence: bool = False,
    input_paths: Mapping[str, str | Path | None] | None = None,
) -> dict[str, object]:
    if not shard_reports:
        raise CarrionSequenceContextComparatorSupportCloseoutError(
            "v153 carrion comparator support closeout shard merge requires reports"
        )
    source_validation = validate_carrion_sequence_context_comparator_inputs(
        v151_report=v151_report,
        v151_rows=v151_rows,
        v152_report=v152_report,
    )
    merged_by_id: dict[str, dict[str, object]] = {}
    digests_by_id: dict[str, str] = {}
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
            raise CarrionSequenceContextComparatorSupportCloseoutError(
                f"shard {shard_index} schema/policy/target/tick/min-floor mismatch"
            )
        status = _mapping(report.get("generation_status"))
        source_integrity = _mapping(report.get("source_integrity"))
        integrity_failures = [
            str(failure) for failure in _list(source_integrity.get("failures"))
        ]
        partial = _status_partial(status)
        if source_integrity.get("passed") is not True and not partial:
            raise CarrionSequenceContextComparatorSupportCloseoutError(
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
                raise CarrionSequenceContextComparatorSupportCloseoutError(
                    "partial shard evidence requires explicit partial merge: "
                    f"{summary}"
                )
            partial_sources.append(summary)
        evidence_rows = _list_of_mappings(report.get("comparator_evidence_rows"))
        expected_digest = str(report.get("comparator_evidence_digest", ""))
        actual_digest = stable_payload_digest(evidence_rows)
        if expected_digest and expected_digest != actual_digest:
            raise CarrionSequenceContextComparatorSupportCloseoutError(
                f"shard {shard_index} comparator evidence digest mismatch"
            )
        for row in evidence_rows:
            _add_merged_evidence_row(
                merged_by_id=merged_by_id,
                digests_by_id=digests_by_id,
                row=row,
            )
        source_summaries.append(
            {
                "source": source_name,
                "source_path": _optional_index(source_paths, shard_index),
                "shard_id": _mapping(_mapping(report.get("inputs")).get("shard")).get(
                    "shard_id"
                ),
                "comparator_evidence_row_count": len(evidence_rows),
                "comparator_evidence_digest": expected_digest or actual_digest,
                "partial": bool(partial),
            }
        )
    for chunk_dir in shard_chunk_dirs:
        chunks = load_carrion_sequence_context_comparator_evidence_chunks(chunk_dir)
        for row in chunks:
            evidence_id = str(row.get("evidence_id", ""))
            if evidence_id not in merged_by_id:
                raise CarrionSequenceContextComparatorSupportCloseoutError(
                    "chunk-dir comparator evidence lacks matching shard report: "
                    f"{chunk_dir}:{evidence_id}"
                )
            _add_merged_evidence_row(
                merged_by_id=merged_by_id,
                digests_by_id=digests_by_id,
                row=row,
            )
        source_summaries.append(
            {
                "source": f"chunk_dir:{chunk_dir}",
                "source_path": str(chunk_dir),
                "comparator_evidence_row_count": len(chunks),
                "partial": False,
            }
        )
    expected_full_count = len(DEFAULT_HARMFUL_SUPPORT_SOURCES) * len(v151_rows)
    if len(merged_by_id) < expected_full_count and not allow_partial_shard_evidence:
        raise CarrionSequenceContextComparatorSupportCloseoutError(
            "partial shard evidence requires explicit partial merge: "
            f"merged_evidence_count={len(merged_by_id)} "
            f"expected_evidence_count={expected_full_count}"
        )
    if len(merged_by_id) < expected_full_count:
        partial_sources.append(
            {
                "source": "merged_union",
                "state": "partial",
                "stop_reason": "merged_evidence_count_below_full_v151_cross_product",
                "merged_evidence_count": len(merged_by_id),
                "expected_evidence_count": expected_full_count,
            }
        )
    rows = sorted(merged_by_id.values(), key=_evidence_row_sort_key)
    identity = _mapping(merge_identity)
    merged_status = {
        "policy": "m3_carrion_sequence_context_comparator_support_closeout_merged_generation_status_v1",
        "state": "partial" if partial_sources else "complete",
        "partial": bool(partial_sources),
        "stop_reason": "partial_shard_evidence" if partial_sources else None,
        "source_count": len(source_summaries),
        "partial_sources": sorted(
            partial_sources,
            key=lambda item: str(item.get("source", "")),
        ),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": list(_list(identity.get("target_carrion_seeds"))),
        "similar_action_mask_threshold": identity.get("similar_action_mask_threshold"),
        "selected_candidate_count": len(_selected_candidate_row_indexes(rows)),
        "total_v151_row_count": int(source_validation.get("v151_row_count", 0)),
        "comparator_evidence_row_count": len(rows),
        "expected_full_comparator_evidence_row_count": expected_full_count,
    }
    merged_evidence = {
        "policy": "m3_carrion_sequence_context_comparator_support_closeout_shard_merge_generation_evidence_v1",
        "merge_mode": True,
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "source_count": len(source_summaries),
        "comparator_evidence_digest": stable_payload_digest(rows),
    }
    report = build_carrion_sequence_context_comparator_support_closeout_report(
        v151_report=v151_report,
        v151_rows=v151_rows,
        v152_report=v152_report,
        comparator_evidence_rows=rows,
        generation_status=merged_status,
        generation_evidence=merged_evidence,
        similar_action_mask_threshold=float(
            identity.get(
                "similar_action_mask_threshold",
                DEFAULT_SIMILAR_ACTION_MASK_THRESHOLD,
            )
        ),
        input_paths=input_paths,
    )
    report["shard_merge"] = {
        "policy": "m3_carrion_sequence_context_comparator_support_closeout_shard_merge_v1",
        "allow_partial_shard_evidence": bool(allow_partial_shard_evidence),
        "sources": sorted(
            source_summaries,
            key=lambda item: (
                str(item.get("source", "")),
                str(item.get("source_path", "")),
                str(item.get("comparator_evidence_digest", "")),
            ),
        ),
        "partial_sources": merged_status["partial_sources"],
        "duplicate_evidence_policy": "same_digest_allowed_conflict_rejected",
    }
    return report


def write_carrion_sequence_context_comparator_evidence_chunks(
    *,
    evidence_rows: Sequence[Mapping[str, object]],
    chunk_dir: str | Path,
) -> None:
    path = Path(chunk_dir)
    path.mkdir(parents=True, exist_ok=True)
    rows_by_source: dict[str, list[dict[str, object]]] = {}
    for row in evidence_rows:
        source = str(row.get("harmful_source_key", ""))
        rows_by_source.setdefault(source, []).append(dict(row))
    for source, rows in sorted(rows_by_source.items()):
        payload = {
            "schema_version": M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_EVIDENCE_CHUNK_SCHEMA_VERSION,
            "policy": M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_EVIDENCE_CHUNK_POLICY,
            "harmful_source_key": source,
            "comparator_evidence_rows": sorted(rows, key=_evidence_row_sort_key),
            "comparator_evidence_digest": stable_payload_digest(
                sorted(rows, key=_evidence_row_sort_key)
            ),
        }
        chunk_path = path / f"{_slug(source)}.json"
        with chunk_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")


def load_carrion_sequence_context_comparator_evidence_chunks(
    chunk_dir: str | Path,
) -> list[dict[str, object]]:
    path = Path(chunk_dir)
    if not path.exists():
        return []
    if not path.is_dir():
        raise CarrionSequenceContextComparatorSupportCloseoutError(
            f"v153 comparator chunk path is not a directory: {path}"
        )
    rows: list[dict[str, object]] = []
    for chunk_path in sorted(path.glob("*.json")):
        payload = load_json_report(chunk_path)
        if payload.get("schema_version") != (
            M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_EVIDENCE_CHUNK_SCHEMA_VERSION
        ):
            raise CarrionSequenceContextComparatorSupportCloseoutError(
                f"v153 comparator chunk schema mismatch: {chunk_path}"
            )
        if payload.get("policy") != (
            M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_EVIDENCE_CHUNK_POLICY
        ):
            raise CarrionSequenceContextComparatorSupportCloseoutError(
                f"v153 comparator chunk policy mismatch: {chunk_path}"
            )
        chunk_rows = _list_of_mappings(payload.get("comparator_evidence_rows"))
        expected = str(payload.get("comparator_evidence_digest", ""))
        actual = stable_payload_digest(chunk_rows)
        if expected and expected != actual:
            raise CarrionSequenceContextComparatorSupportCloseoutError(
                f"v153 comparator chunk digest mismatch: {chunk_path}"
            )
        rows.extend(chunk_rows)
    return sorted(rows, key=_evidence_row_sort_key)


def validate_carrion_sequence_context_comparator_inputs(
    *,
    v151_report: Mapping[str, object],
    v151_rows: Sequence[Mapping[str, object]],
    v152_report: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    v151_dataset_digest = stable_payload_digest(list(v151_rows))
    v151_branch_results = _list_of_mappings(v151_report.get("branch_results"))
    v151_branch_digest = stable_payload_digest(v151_branch_results)
    v151_dataset = _mapping(v151_report.get("dataset"))
    v151_source = _mapping(v151_report.get("source_integrity"))
    v151_generation = _mapping(v151_report.get("generation_status"))
    v151_classification = _mapping(v151_report.get("classification")).get("primary")
    v152_source = _mapping(v152_report.get("source_integrity"))
    v152_generation = _mapping(v152_report.get("generation_status"))
    v152_classification = _mapping(v152_report.get("classification")).get("primary")

    if v151_report.get("schema_version") != (
        M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION
    ):
        failures.append("v151_schema_mismatch")
    if v151_report.get("policy") != M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_POLICY:
        failures.append("v151_policy_mismatch")
    if v151_classification != EXPECTED_V151_CLASSIFICATION:
        failures.append("v151_unexpected_classification")
    if v151_source.get("passed") is not True or _list(v151_source.get("failures")):
        failures.append("v151_source_integrity_not_passed")
    if (
        v151_generation.get("state") != "complete"
        or v151_generation.get("partial") is True
    ):
        failures.append("v151_generation_not_complete")
    if v151_dataset.get("dataset_digest") != v151_dataset_digest:
        failures.append("v151_dataset_digest_mismatch")
    if _int(v151_dataset.get("row_count")) != len(v151_rows):
        failures.append("v151_row_count_mismatch")
    if v151_report.get("branch_evidence_digest") != v151_branch_digest:
        failures.append("v151_branch_evidence_digest_mismatch")

    if v152_report.get("schema_version") != (
        M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_SCHEMA_VERSION
    ):
        failures.append("v152_schema_mismatch")
    if v152_report.get("policy") != (
        M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_POLICY
    ):
        failures.append("v152_policy_mismatch")
    if v152_classification not in EXPECTED_V152_CLASSIFICATIONS:
        failures.append("v152_unexpected_classification")
    if v152_source.get("passed") is not True or _list(v152_source.get("failures")):
        failures.append("v152_source_integrity_not_passed")
    if (
        v152_generation.get("state") != "complete"
        or v152_generation.get("partial") is True
    ):
        failures.append("v152_generation_not_complete")
    if v152_report.get("audit_row_digest") != v151_dataset_digest:
        failures.append("v152_audit_row_digest_mismatch")
    if _int(v152_report.get("audit_row_count")) != len(v151_rows):
        failures.append("v152_audit_row_count_mismatch")

    for prefix, payload in (("v151", v151_report), ("v152", v152_report)):
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
    observed_sources = {
        _harmful_key_from_row(row) for row in v151_rows if _is_harmful_row(row)
    }
    expected_sources = {_harmful_key(source) for source in DEFAULT_HARMFUL_SUPPORT_SOURCES}
    if expected_sources - observed_sources:
        failures.append("target_harmful_support_sources_missing")
    if failures:
        raise CarrionSequenceContextComparatorSupportCloseoutError(
            "v153 carrion sequence-context comparator closeout input validation failed: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "policy": "m3_carrion_sequence_context_comparator_support_closeout_input_validation_v1",
        "passed": True,
        "failures": [],
        "v151_report_digest": stable_payload_digest(v151_report),
        "v151_exact_digest": v151_report.get("exact_digest"),
        "v151_dataset_digest": v151_dataset_digest,
        "v151_branch_evidence_digest": v151_branch_digest,
        "v151_row_count": len(v151_rows),
        "v152_report_digest": stable_payload_digest(v152_report),
        "v152_exact_digest": v152_report.get("exact_digest"),
        "v152_classification": v152_classification,
    }


def _finalize_report(
    *,
    v151_report: Mapping[str, object],
    v151_rows: Sequence[Mapping[str, object]],
    v152_report: Mapping[str, object],
    source_validation: Mapping[str, object],
    harmful_sources: Sequence[Mapping[str, object]],
    comparator_evidence_rows: Sequence[Mapping[str, object]],
    generation_status: Mapping[str, object],
    generation_evidence: Mapping[str, object],
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    row_index_include: Sequence[int] | None,
    row_index_start: int | None,
    row_index_count: int | None,
    shard_id: str | None,
    similar_action_mask_threshold: float,
    comparator_chunk_dir: str | Path | None,
    input_paths: Mapping[str, str | Path | None] | None,
) -> dict[str, object]:
    rows = sorted(
        [dict(row) for row in comparator_evidence_rows],
        key=_evidence_row_sort_key,
    )
    leakage_scan = carrion_sequence_context_trainable_leakage_scan(rows)
    support = _comparator_support_report(
        evidence_rows=rows,
        harmful_sources=harmful_sources,
    )
    coverage = _coverage_report(
        evidence_rows=rows,
        harmful_sources=harmful_sources,
        total_v151_row_count=len(v151_rows),
        target_seeds=DEFAULT_CARRION_SEEDS,
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
    source_integrity = _source_integrity(
        source_validation=source_validation,
        leakage_scan=leakage_scan,
        generation_status=generation_status,
        coverage=coverage,
    )
    classification = _classification(
        source_integrity=source_integrity,
        support=support,
    )
    recommendation = _recommendation(classification)
    evidence_digest = stable_payload_digest(rows)
    contract = {
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "trainable_dataset_emitted": False,
        "runtime_path": "none_added_report_only",
        "trainable_input_surface": (
            "v151 public observation_input, public action_mask, and prior "
            "finalized public sequence context only"
        ),
        "trainable_label_surface": "action label target only",
        "excluded_trainable_input_fields": [
            "seed",
            "fixture",
            "branch id",
            "tick identity",
            "agent id",
            "path",
            "digest",
            "private world state",
            "future rows",
            "future outcomes",
            "labels",
            "provenance",
        ],
    }
    inputs = {
        **_input_path_payload(input_paths),
        "v151_exact_digest": source_validation.get("v151_exact_digest"),
        "v151_dataset_digest": source_validation.get("v151_dataset_digest"),
        "v151_branch_evidence_digest": source_validation.get(
            "v151_branch_evidence_digest"
        ),
        "v152_exact_digest": source_validation.get("v152_exact_digest"),
        "v152_classification": source_validation.get("v152_classification"),
        "target_fixture": "carrion_only",
        "target_carrion_seeds": [int(seed) for seed in DEFAULT_CARRION_SEEDS],
        "target_harmful_support_sources": [
            {
                "label_action": source.get("label_action"),
                "source_seed": source.get("source_seed"),
                "source_branch_reason": source.get("source_branch_reason"),
            }
            for source in harmful_sources
        ],
        "similar_action_mask_threshold": float(similar_action_mask_threshold),
        "comparator_chunk_dir": (
            None if comparator_chunk_dir is None else str(comparator_chunk_dir)
        ),
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
    }
    report = {
        "schema_version": M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION,
        "policy": M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY,
        "diagnostics_only": True,
        "contract": contract,
        "inputs": inputs,
        "generation_status": dict(generation_status),
        "generation_evidence": dict(generation_evidence),
        "source_validation": dict(source_validation),
        "source_integrity": source_integrity,
        "coverage": coverage,
        "comparator_support": support,
        "leakage_scan": leakage_scan,
        "classification": {"primary": classification, "labels": [classification]},
        "recommendation": recommendation,
        "comparator_evidence_rows": rows,
        "comparator_evidence_row_count": len(rows),
        "comparator_evidence_digest": evidence_digest,
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
            "v152_exact_digest": inputs["v152_exact_digest"],
            "v152_classification": inputs["v152_classification"],
            "target_carrion_seeds": inputs["target_carrion_seeds"],
            "similar_action_mask_threshold": inputs["similar_action_mask_threshold"],
        },
        "comparator_evidence_digest": evidence_digest,
        "source_integrity": source_integrity,
        "coverage": coverage,
        "comparator_support": support,
        "classification": report["classification"],
        "recommendation": recommendation,
    }
    report["exact_digest"] = stable_payload_digest(exact_payload)
    report["provenance"] = {
        "contract_digest": stable_payload_digest(contract),
        "exact_digest_payload_policy": (
            "stable_payload_digest_of_v153_contract_inputs_comparator_evidence_and_closeout_v1"
        ),
        "exact_digest": report["exact_digest"],
    }
    return report


def _comparator_evidence_row(
    *,
    harmful: Mapping[str, object],
    candidate: Mapping[str, object],
    similar_action_mask_threshold: float,
) -> dict[str, object]:
    harmful_action = _row_action(harmful)
    candidate_action = _row_action(candidate)
    metadata = _mapping(candidate.get("metadata"))
    candidate_role = (
        "target_harmful_source"
        if _harmful_key_from_row(candidate) == _harmful_key_from_row(harmful)
        else ("harmful_comparator" if _is_harmful_row(candidate) else "safe_comparator")
    )
    same_label_action = candidate_action == harmful_action
    exact_mask = _action_mask_key(candidate) == _action_mask_key(harmful)
    mask_similarity = _action_mask_similarity(candidate, harmful)
    similar_mask = mask_similarity >= float(similar_action_mask_threshold)
    safe = candidate_role == "safe_comparator"
    harmful_comparator = candidate_role == "harmful_comparator"
    matches = {
        "same_label_action": bool(same_label_action),
        "same_label_action_exact_action_mask": bool(same_label_action and exact_mask),
        "same_label_action_similar_action_mask": bool(
            same_label_action and similar_mask
        ),
        "same_harmful_source_branch_reason": bool(
            same_label_action and _branch_reason(candidate) == _branch_reason(harmful)
        ),
        "exact_one_step_safe_comparator": bool(
            safe
            and same_label_action
            and _one_step_features_key(candidate) == _one_step_features_key(harmful)
        ),
        "zero_prior_safe_comparator": bool(
            safe and same_label_action and _prior_step_count(candidate) == 0
        ),
        "nonzero_prior_harmful_comparator": bool(
            harmful_comparator
            and same_label_action
            and _prior_step_count(candidate) > 0
        ),
    }
    trainable = {
        "feature_policy": (
            "public_observation_action_mask_prior_public_sequence_context_v153_comparator_support_v1"
        ),
        "features": {
            "observation_input": _features(candidate).get("observation_input"),
            "action_mask": _bool_action_mask(_features(candidate).get("action_mask")),
            "prior_public_sequence_context": _prior_context(candidate),
        },
        "label": {"action": candidate_action},
    }
    row_id_payload = {
        "harmful_source_key": _harmful_key_from_row(harmful),
        "candidate_row_id": _candidate_row_id(candidate),
        "candidate_action": candidate_action,
    }
    evidence_id = stable_payload_digest(row_id_payload)
    row = {
        "schema_version": "m3_carrion_sequence_context_comparator_evidence_row_v1",
        "policy": M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY,
        "evidence_id": evidence_id,
        "harmful_source_key": _harmful_key_from_row(harmful),
        "candidate_row_id": _candidate_row_id(candidate),
        "candidate_role": candidate_role,
        "label_action": candidate_action,
        "target_harmful_label_action": harmful_action,
        "matches": matches,
        "match_strength": _match_strength(matches),
        "action_mask_similarity": _round(mask_similarity),
        "candidate_prior_step_count": _prior_step_count(candidate),
        "target_prior_step_count": _prior_step_count(harmful),
        "trainable": trainable,
        "metadata": {
            "candidate_seed": metadata.get("seed"),
            "candidate_fixture": metadata.get("fixture"),
            "candidate_branch_id": metadata.get("branch_id"),
            "candidate_branch_index": metadata.get("branch_index"),
            "candidate_branch_reason": metadata.get("branch_reason"),
            "candidate_row_index": metadata.get("row_index"),
            "candidate_source_dataset_row_index": metadata.get(
                "source_dataset_row_index"
            ),
            "target_source_seed": _mapping(harmful.get("metadata")).get("seed"),
            "target_source_branch_id": _mapping(harmful.get("metadata")).get(
                "branch_id"
            ),
            "target_source_branch_reason": _branch_reason(harmful),
            "metadata_excluded_from_trainable_input": True,
        },
    }
    row["stable_evidence_digest"] = stable_payload_digest(
        {
            "evidence_id": evidence_id,
            "trainable": trainable,
            "matches": matches,
            "candidate_role": candidate_role,
        }
    )
    return row


def _comparator_support_report(
    *,
    evidence_rows: Sequence[Mapping[str, object]],
    harmful_sources: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    reports = []
    totals = Counter()
    for source in harmful_sources:
        key = _harmful_key(source)
        rows = [
            row
            for row in evidence_rows
            if str(row.get("harmful_source_key", "")) == key
        ]
        counts = _support_counts(rows)
        reports.append(
            {
                "harmful_source_key": key,
                "label_action": source.get("label_action"),
                "source_seed": source.get("source_seed"),
                "source_branch_reason": source.get("source_branch_reason"),
                "source_branch_id": source.get("source_branch_id"),
                **counts,
                "support_unavailable": (
                    counts["exact_one_step_safe_comparator_count"] == 0
                    and counts["zero_prior_safe_comparator_count"] == 0
                    and counts["nonzero_prior_harmful_comparator_count"] == 0
                ),
            }
        )
        for name, value in counts.items():
            totals[name] += int(value)
    all_strong_unavailable = all(
        report["support_unavailable"] is True for report in reports
    )
    return {
        "policy": "m3_carrion_sequence_context_comparator_support_summary_v1",
        "target_harmful_source_count": len(harmful_sources),
        "harmful_source_support": reports,
        "exact_one_step_safe_comparator_total": int(
            totals["exact_one_step_safe_comparator_count"]
        ),
        "zero_prior_safe_comparator_total": int(
            totals["zero_prior_safe_comparator_count"]
        ),
        "nonzero_prior_harmful_comparator_total": int(
            totals["nonzero_prior_harmful_comparator_count"]
        ),
        "same_label_action_safe_comparator_total": int(
            totals["same_label_action_safe_comparator_count"]
        ),
        "same_label_action_exact_mask_safe_comparator_total": int(
            totals["same_label_action_exact_mask_safe_comparator_count"]
        ),
        "same_label_action_similar_mask_safe_comparator_total": int(
            totals["same_label_action_similar_mask_safe_comparator_count"]
        ),
        "same_label_action_same_reason_safe_comparator_total": int(
            totals["same_label_action_same_reason_safe_comparator_count"]
        ),
        "all_strong_comparator_support_unavailable": bool(all_strong_unavailable),
        "strong_comparator_support_found_for_all_harmful_sources": all(
            report["support_unavailable"] is False for report in reports
        ),
    }


def _support_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    safe_same_action = [
        row
        for row in rows
        if row.get("candidate_role") == "safe_comparator"
        and _mapping(row.get("matches")).get("same_label_action") is True
    ]
    harmful_same_action = [
        row
        for row in rows
        if row.get("candidate_role") == "harmful_comparator"
        and _mapping(row.get("matches")).get("same_label_action") is True
    ]
    return {
        "candidate_evidence_row_count": len(rows),
        "same_label_action_safe_comparator_count": len(safe_same_action),
        "same_label_action_exact_mask_safe_comparator_count": sum(
            1
            for row in safe_same_action
            if _mapping(row.get("matches")).get("same_label_action_exact_action_mask")
            is True
        ),
        "same_label_action_similar_mask_safe_comparator_count": sum(
            1
            for row in safe_same_action
            if _mapping(row.get("matches")).get("same_label_action_similar_action_mask")
            is True
        ),
        "same_label_action_same_reason_safe_comparator_count": sum(
            1
            for row in safe_same_action
            if _mapping(row.get("matches")).get("same_harmful_source_branch_reason")
            is True
        ),
        "exact_one_step_safe_comparator_count": sum(
            1
            for row in safe_same_action
            if _mapping(row.get("matches")).get("exact_one_step_safe_comparator")
            is True
        ),
        "zero_prior_safe_comparator_count": sum(
            1
            for row in safe_same_action
            if _mapping(row.get("matches")).get("zero_prior_safe_comparator") is True
        ),
        "nonzero_prior_harmful_comparator_count": sum(
            1
            for row in harmful_same_action
            if _mapping(row.get("matches")).get("nonzero_prior_harmful_comparator")
            is True
        ),
    }


def _coverage_report(
    *,
    evidence_rows: Sequence[Mapping[str, object]],
    harmful_sources: Sequence[Mapping[str, object]],
    total_v151_row_count: int,
    target_seeds: Sequence[int],
    strict_full_scope: bool,
) -> dict[str, object]:
    candidate_row_indexes = _selected_candidate_row_indexes(evidence_rows)
    expected_full_count = len(harmful_sources) * int(total_v151_row_count)
    seeds = sorted(
        {
            _int(_mapping(row.get("metadata")).get("candidate_seed"), default=-1)
            for row in evidence_rows
            if _int(_mapping(row.get("metadata")).get("candidate_seed"), default=-1)
            >= 0
        }
    )
    return {
        "policy": "m3_carrion_sequence_context_comparator_support_coverage_v1",
        "target_fixture": "carrion_only",
        "target_seeds": [int(seed) for seed in target_seeds],
        "strict_full_scope": bool(strict_full_scope),
        "target_harmful_source_count": len(harmful_sources),
        "selected_candidate_row_count": len(candidate_row_indexes),
        "selected_candidate_row_indexes": candidate_row_indexes,
        "selected_candidate_seeds": seeds,
        "total_v151_row_count": int(total_v151_row_count),
        "comparator_evidence_row_count": len(evidence_rows),
        "expected_full_comparator_evidence_row_count": expected_full_count,
        "full_cross_product_covered": len(evidence_rows) == expected_full_count,
        "missing_candidate_row_count": max(
            0,
            int(total_v151_row_count) - len(candidate_row_indexes),
        ),
    }


def _source_integrity(
    *,
    source_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    generation_status: Mapping[str, object],
    coverage: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if source_validation.get("passed") is not True:
        failures.append("input_validation_failed")
    if leakage_scan.get("passed") is not True:
        failures.append("trainable_input_leakage_detected")
    if _status_partial(generation_status):
        failures.append("partial_comparator_evidence")
    if int(coverage.get("comparator_evidence_row_count", 0)) <= 0:
        failures.append("no_comparator_evidence_rows")
    if (
        coverage.get("strict_full_scope") is True
        and coverage.get("full_cross_product_covered") is not True
    ):
        failures.append("full_comparator_cross_product_missing")
    return {
        "policy": "m3_carrion_sequence_context_comparator_support_source_integrity_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "input_validation_passed": source_validation.get("passed") is True,
        "leakage_scan_passed": leakage_scan.get("passed") is True,
        "generation_state": generation_status.get("state"),
        "strict_full_scope": coverage.get("strict_full_scope"),
        "comparator_evidence_row_count": coverage.get(
            "comparator_evidence_row_count"
        ),
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    support: Mapping[str, object],
) -> str:
    if source_integrity.get("passed") is not True:
        return "m3_carrion_sequence_context_comparator_support_source_integrity_failed_no_training"
    if support.get("all_strong_comparator_support_unavailable") is True:
        return "m3_carrion_sequence_context_comparator_support_limited_closed_no_training"
    if support.get("strong_comparator_support_found_for_all_harmful_sources") is True:
        return "m3_carrion_sequence_context_comparator_support_expanded_no_training"
    return "m3_carrion_sequence_context_comparator_support_partial_no_training"


def _recommendation(classification: str) -> dict[str, object]:
    closed = classification == (
        "m3_carrion_sequence_context_comparator_support_limited_closed_no_training"
    )
    return {
        "policy": "m3_carrion_sequence_context_comparator_support_closeout_recommendation_v1",
        "route_closed": bool(closed),
        "recommended_next_route": (
            "return_to_carrion_recovery_archive_or_go_explore_survivor_continuation_evidence"
            if closed
            else "review_new_comparator_support_before_any_training_route"
        ),
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "rationale": (
            "v151/v152 sequence-context evidence lacks exact one-step, "
            "zero-prior safe, and nonzero-prior harmful comparators for the "
            "three harmful carrion support labels."
            if closed
            else "new strong public comparator support exists and needs a separate review."
        ),
    }


def _target_harmful_rows(
    rows: Sequence[Mapping[str, object]],
    harmful_sources: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_key = {
        _harmful_key_from_row(row): dict(row)
        for row in rows
        if _is_harmful_row(row) and _harmful_key_from_row(row) is not None
    }
    result = []
    for source in harmful_sources:
        key = _harmful_key(source)
        row = by_key.get(key)
        if row is None:
            raise CarrionSequenceContextComparatorSupportCloseoutError(
                f"missing target harmful v151 row: {key}"
            )
        result.append(row)
    return sorted(result, key=_candidate_row_sort_key)


def _selected_candidate_rows(
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
            raise CarrionSequenceContextComparatorSupportCloseoutError(
                "--row-index-start and --row-index-count must be provided together"
            )
        if int(row_index_start) < 0 or int(row_index_count) < 0:
            raise CarrionSequenceContextComparatorSupportCloseoutError(
                "row index start/count must be non-negative"
            )
        row_filter = set(row_filter or set())
        row_filter.update(
            range(int(row_index_start), int(row_index_start) + int(row_index_count))
        )
    selected = []
    for index, row in enumerate(rows):
        metadata = _mapping(row.get("metadata"))
        row_index = _row_index(row, fallback=index)
        seed = _int(metadata.get("seed"), default=-1)
        branch_index = _int(metadata.get("branch_index"), default=-1)
        if seed_filter is not None and seed not in seed_filter:
            continue
        if branch_filter is not None and branch_index not in branch_filter:
            continue
        if row_filter is not None and row_index not in row_filter:
            continue
        selected.append(dict(row))
    return sorted(selected, key=_candidate_row_sort_key)


def _generation_status(
    *,
    selected_candidate_count: int,
    selected_candidate_row_indexes: Sequence[int],
    total_v151_row_count: int,
    seed_include: Sequence[int] | None,
    branch_index_include: Sequence[int] | None,
    row_index_include: Sequence[int] | None,
    row_index_start: int | None,
    row_index_count: int | None,
    shard_id: str | None,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_sequence_context_comparator_support_closeout_generation_status_v1",
        "state": "complete",
        "partial": False,
        "stop_reason": None,
        "target_fixture": "carrion_only",
        "target_harmful_source_count": len(DEFAULT_HARMFUL_SUPPORT_SOURCES),
        "selected_candidate_count": int(selected_candidate_count),
        "selected_candidate_row_indexes": [int(index) for index in selected_candidate_row_indexes],
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


def _validate_shard_contract(report: Mapping[str, object], *, shard_index: int) -> None:
    if report.get("schema_version") != (
        M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION
    ):
        raise CarrionSequenceContextComparatorSupportCloseoutError(
            f"shard {shard_index} schema mismatch"
        )
    if report.get("policy") != (
        M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY
    ):
        raise CarrionSequenceContextComparatorSupportCloseoutError(
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
            raise CarrionSequenceContextComparatorSupportCloseoutError(
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
        "v152_exact_digest": inputs.get("v152_exact_digest"),
        "v152_classification": inputs.get("v152_classification"),
        "target_carrion_seeds": list(_list(inputs.get("target_carrion_seeds"))),
        "similar_action_mask_threshold": inputs.get("similar_action_mask_threshold"),
    }


def _add_merged_evidence_row(
    *,
    merged_by_id: dict[str, dict[str, object]],
    digests_by_id: dict[str, str],
    row: Mapping[str, object],
) -> None:
    evidence_id = str(row.get("evidence_id", ""))
    if not evidence_id:
        raise CarrionSequenceContextComparatorSupportCloseoutError(
            "cannot merge comparator evidence row without evidence_id"
        )
    digest = stable_payload_digest(row)
    prior = digests_by_id.get(evidence_id)
    if prior is not None:
        if prior != digest:
            raise CarrionSequenceContextComparatorSupportCloseoutError(
                f"duplicate comparator evidence with different digest: {evidence_id}"
            )
        return
    merged_by_id[evidence_id] = dict(row)
    digests_by_id[evidence_id] = digest


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
        "m3_carrion_sequence_context_comparator_support_closeout_merged_generation_status_v1"
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


def _selected_candidate_count(rows: Sequence[Mapping[str, object]]) -> int:
    return len(_selected_candidate_row_indexes(rows))


def _selected_candidate_row_indexes(rows: Sequence[Mapping[str, object]]) -> list[int]:
    indexes = {
        _int(_mapping(row.get("metadata")).get("candidate_row_index"), default=-1)
        for row in rows
    }
    return sorted(index for index in indexes if index >= 0)


def _candidate_row_id(row: Mapping[str, object]) -> str:
    metadata = _mapping(row.get("metadata"))
    return ":".join(
        [
            str(metadata.get("source_dataset_row_index", metadata.get("row_index", ""))),
            str(metadata.get("branch_id", "")),
            _row_action(row),
        ]
    )


def _row_action(row: Mapping[str, object]) -> str:
    metadata = _mapping(row.get("metadata"))
    label = _mapping(_mapping(row.get("trainable")).get("label"))
    return str(metadata.get("label_action") or label.get("action") or "")


def _is_harmful_row(row: Mapping[str, object]) -> bool:
    return _mapping(row.get("metadata")).get("harmful_source") is True


def _branch_reason(row: Mapping[str, object]) -> str:
    return str(_mapping(row.get("metadata")).get("branch_reason", ""))


def _features(row: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(_mapping(row.get("trainable")).get("features"))


def _prior_context(row: Mapping[str, object]) -> list[dict[str, object]]:
    return _list_of_mappings(_features(row).get("prior_public_sequence_context"))


def _prior_step_count(row: Mapping[str, object]) -> int:
    markers = _mapping(_mapping(row.get("metadata")).get("hydration_reproduction_markers"))
    if "prior_step_count" in markers:
        return _int(markers.get("prior_step_count"))
    return len(_prior_context(row))


def _one_step_features_key(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "observation_input": _features(row).get("observation_input"),
        "action_mask": _bool_action_mask(_features(row).get("action_mask")),
    }


def _action_mask_key(row: Mapping[str, object]) -> tuple[tuple[str, bool], ...]:
    return tuple(sorted(_bool_action_mask(_features(row).get("action_mask")).items()))


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


def _match_strength(matches: Mapping[str, object]) -> str:
    if matches.get("exact_one_step_safe_comparator") is True:
        return "exact_one_step_safe"
    if matches.get("zero_prior_safe_comparator") is True:
        return "zero_prior_safe"
    if matches.get("nonzero_prior_harmful_comparator") is True:
        return "nonzero_prior_harmful"
    if matches.get("same_label_action_exact_action_mask") is True:
        return "same_label_action_exact_mask"
    if matches.get("same_label_action_similar_action_mask") is True:
        return "same_label_action_similar_mask"
    if matches.get("same_label_action") is True:
        return "same_label_action"
    return "not_comparable"


def _candidate_row_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    metadata = _mapping(row.get("metadata"))
    return (
        _row_index(row),
        _int(metadata.get("source_dataset_row_index"), default=10**9),
        str(metadata.get("branch_id", "")),
        _row_action(row),
    )


def _evidence_row_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    metadata = _mapping(row.get("metadata"))
    return (
        str(row.get("harmful_source_key", "")),
        _int(metadata.get("candidate_row_index"), default=10**9),
        str(row.get("candidate_row_id", "")),
        str(row.get("evidence_id", "")),
    )


def _row_index(row: Mapping[str, object], fallback: int = 10**9) -> int:
    return _int(_mapping(row.get("metadata")).get("row_index"), default=fallback)


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


def _slug(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("-")
    return "".join(allowed).strip("-")[:180] or "chunk"


def _round(value: float) -> float:
    return round(float(value), 6)


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
