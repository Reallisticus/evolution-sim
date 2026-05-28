from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
)
from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_candidate_ranker_capacity_audit import (
    _list_like,
    _ratio,
)
from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    DEFAULT_V115_ARCHIVE_ROWS_PATH,
    _candidate_groups,
    _resolve_jsonl_rows,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _int,
    _mapping,
    _resolve_archive_rows,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_refreshed_candidate_public_feature_surface import (
    DEFAULT_FEATURE_ROWS_OUTPUT_PATH as DEFAULT_V131_FEATURE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V131_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION,
    MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION,
    V131_FORBIDDEN_FEATURE_PATH_PARTS,
    V131_FORBIDDEN_FEATURE_PATH_SUBSTRINGS,
    _feature_key,
    _fixture_group,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_refreshed_surface_blocker_slice_audit_v1"
)
MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_POLICY = (
    "diagnostics_only_first_recovery_v132_refreshed_surface_blocker_slice_audit_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v132-first-recovery-refreshed-surface-blocker-slice-audit.json"
)
DEFAULT_DETAIL_ROWS_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v132-first-recovery-refreshed-surface-blocker-slices.jsonl"
)

EXPECTED_V131_CLASSIFICATION = (
    "refreshed_candidate_public_feature_surface_blocked_by_signal"
)
EXPECTED_V131_FAILURES: tuple[str, str] = (
    "fixture_open_failed",
    "heldout_signal_not_above_action_only_baseline",
)
PRIMARY_PROBE = "refreshed_public_context_probe"

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "refreshed_surface_blocker_slice_audit_source_integrity_failed",
    "refreshed_surface_blocker_slice_audit_completed",
)
BLOCKER_CLASSES: tuple[str, ...] = (
    "train_support_hole",
    "feature_alias_collision",
    "action_prior_tie_or_fallback",
    "fixture_open_action_collapse",
    "material_candidate_not_selected",
    "insufficient_public_context_signal",
    "inconclusive",
)
RECOMMENDATIONS: tuple[str, ...] = (
    "collect_targeted_public_context_rows",
    "add_public_rollout_history_context",
    "repair_feature_key_collision_surface",
    "stop_ranker_path_insufficient_signal",
    "inconclusive_requires_manual_review",
)
V132_FORBIDDEN_FEATURE_PATH_PARTS = V131_FORBIDDEN_FEATURE_PATH_PARTS | {
    "audit",
    "digest",
    "fold",
    "metadata",
    "split",
}
V132_FORBIDDEN_FEATURE_PATH_SUBSTRINGS = tuple(
    sorted(
        set(V131_FORBIDDEN_FEATURE_PATH_SUBSTRINGS)
        | {
            "audit_metadata",
            "candidate_context_join_digest",
        }
    )
)


@dataclass(frozen=True, slots=True)
class FirstRecoveryRefreshedSurfaceBlockerSliceAuditBuild:
    report: dict[str, object]
    detail_rows: tuple[dict[str, object], ...]


def build_first_recovery_refreshed_surface_blocker_slice_audit(
    *,
    v131_report: Mapping[str, object] | None = None,
    v131_report_path: str | Path | None = DEFAULT_V131_REPORT_PATH,
    v131_feature_rows: Sequence[Mapping[str, object]] | None = None,
    v131_feature_rows_path: str | Path | None = DEFAULT_V131_FEATURE_ROWS_PATH,
    v124_manifest_rows: Sequence[Mapping[str, object]] | None = None,
    v124_manifest_path: str | Path | None = DEFAULT_V124_MANIFEST_PATH,
    v115_archive_rows: Sequence[Mapping[str, object]] | None = None,
    v115_archive_rows_path: str | Path | None = DEFAULT_V115_ARCHIVE_ROWS_PATH,
    v123_archive_rows: Sequence[Mapping[str, object]] | None = None,
    v123_archive_rows_path: str | Path | None = DEFAULT_V123_ARCHIVE_ROWS_PATH,
) -> FirstRecoveryRefreshedSurfaceBlockerSliceAuditBuild:
    v131_payload, v131_evidence = _resolve_json_report(
        v131_report,
        v131_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION,
    )
    v131_rows, v131_rows_evidence = _resolve_jsonl_rows(
        v131_feature_rows,
        v131_feature_rows_path,
        row_kind="v131_feature_rows",
    )
    manifest_rows, manifest_evidence = _resolve_jsonl_rows(
        v124_manifest_rows,
        v124_manifest_path,
        row_kind="v124_manifest",
    )
    v115_rows, v115_evidence = _resolve_archive_rows(
        v115_archive_rows,
        v115_archive_rows_path,
    )
    v123_rows, v123_evidence = _resolve_archive_rows(
        v123_archive_rows,
        v123_archive_rows_path,
    )
    candidate_rows = tuple(v115_rows) + tuple(v123_rows)
    candidate_groups = _candidate_groups(candidate_rows)
    row_alignment = _row_alignment(
        manifest_rows=manifest_rows,
        candidate_groups=candidate_groups,
        feature_rows=v131_rows,
    )
    feature_scan = _feature_row_forbidden_scan(v131_rows)
    source_reports = {
        "v131_report": v131_evidence,
        "v131_feature_rows": v131_rows_evidence,
        "v124_manifest": manifest_evidence,
        "v115_archive_rows": v115_evidence,
        "v123_archive_rows": v123_evidence,
    }
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v131_report=v131_payload,
        v131_feature_rows=v131_rows,
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
        row_alignment=row_alignment,
        feature_scan=feature_scan,
    )
    dataset = _dataset(
        manifest_rows=manifest_rows,
        candidate_groups=candidate_groups,
        row_alignment=row_alignment,
    )
    score_tables = _score_tables(dataset)
    branch_reports = _branch_reports(
        dataset=dataset,
        score_tables=score_tables,
    )
    blocker_rows = _blocker_rows(branch_reports)
    aggregate = _aggregate(blocker_rows)
    recommendation = _recommendation(aggregate)
    classification = _classification(source_integrity)
    detail_rows = tuple(
        _detail_row_for_output(row)
        for row in blocker_rows
    )
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "v131_blocker_contract": _v131_blocker_contract(v131_payload),
        "score_table_summary": _score_table_summary(score_tables),
        "slice_extraction": _slice_extraction_summary(blocker_rows),
        "blocker_class_aggregate": aggregate,
        "recommendation": recommendation,
        "classification": classification,
        "authorization_block": _authorization_block(),
        "detail_rows_summary": {
            "row_count": len(detail_rows),
            "default_output_path": str(DEFAULT_DETAIL_ROWS_OUTPUT_PATH),
            "runtime_loadable": False,
        },
        "non_promoted": True,
    }
    return FirstRecoveryRefreshedSurfaceBlockerSliceAuditBuild(
        report=report,
        detail_rows=detail_rows,
    )


def write_first_recovery_refreshed_surface_blocker_slice_audit_report(
    build: FirstRecoveryRefreshedSurfaceBlockerSliceAuditBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_first_recovery_refreshed_surface_blocker_slice_rows(
    build: FirstRecoveryRefreshedSurfaceBlockerSliceAuditBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in build.detail_rows:
            json.dump(row, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "report_only_blocker_slice_audit": True,
        "training_executed": False,
        "runtime_policy_effect": "none",
        "readiness_rerun_executed": False,
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
    v131_report: Mapping[str, object] | None,
    v131_feature_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    row_alignment: Mapping[str, object],
    feature_scan: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")
    report = _mapping(v131_report or {})
    classification = _mapping(report.get("classification")).get("primary")
    source = _mapping(report.get("source_integrity"))
    metric_gate = _mapping(report.get("metric_gate"))
    feature_summary = _mapping(report.get("feature_surface_summary"))
    forbidden = _mapping(report.get("forbidden_feature_scan"))
    leakage = _mapping(report.get("leakage_audit"))
    source_row_digests = _mapping(source.get("source_row_digests"))
    v129_digest = _mapping(source_row_digests.get("v129_feature_rows"))
    v130_digest = _mapping(source_row_digests.get("v130_context_rows"))
    metric_failures = sorted(str(item) for item in _list_like(metric_gate.get("failures")))
    expected_failures = sorted(EXPECTED_V131_FAILURES)
    if classification != EXPECTED_V131_CLASSIFICATION:
        failures.append("v131_classification_not_blocked_by_signal")
    if source.get("passed") is not True:
        failures.append("v131_source_integrity_not_passed")
    if v129_digest.get("matches_report_digest") is not True:
        failures.append("v131_v129_row_digest_not_enforced")
    if v130_digest.get("matches_report_digest") is not True:
        failures.append("v131_v130_row_digest_not_enforced")
    if metric_failures != expected_failures:
        failures.append("v131_required_failure_labels_mismatch")
    if _int(feature_summary.get("candidate_feature_row_count")) != len(v131_feature_rows):
        failures.append("v131_feature_row_count_mismatch")
    if _int(source.get("candidate_row_count")) != len(candidate_rows):
        failures.append("candidate_row_count_mismatch")
    if _int(source.get("manifest_row_count")) != len(manifest_rows):
        failures.append("manifest_row_count_mismatch")
    if _int(forbidden.get("forbidden_feature_path_count")) != 0:
        failures.append("v131_report_forbidden_paths_nonzero")
    if _int(leakage.get("leakage_count")) != 0:
        failures.append("v131_report_leakage_nonzero")
    if feature_scan.get("passed") is not True:
        failures.append("v131_feature_rows_forbidden_or_leaky")
    if row_alignment.get("passed") is not True:
        failures.append("v131_feature_row_alignment_failed")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v131_classification": classification,
        "v131_metric_failures": metric_failures,
        "expected_v131_metric_failures": expected_failures,
        "v131_source_integrity_passed": source.get("passed"),
        "v131_feature_row_count": len(v131_feature_rows),
        "manifest_row_count": len(manifest_rows),
        "candidate_row_count": len(candidate_rows),
        "v131_source_row_digests": dict(source_row_digests),
        "feature_row_digest": stable_payload_digest(list(v131_feature_rows)),
        "feature_row_forbidden_scan": dict(feature_scan),
        "feature_row_alignment": _alignment_summary(row_alignment),
    }


def _v131_blocker_contract(v131_report: Mapping[str, object]) -> dict[str, object]:
    comparison = _mapping(v131_report.get("probe_comparison"))
    fixture_open = _mapping(v131_report.get("fixture_open_evaluation"))
    open_group = _mapping(_mapping(fixture_open.get("groups")).get("open_mind_v3"))
    open_dominant = _mapping(open_group.get("dominant_predicted_action"))
    return {
        "required_classification": EXPECTED_V131_CLASSIFICATION,
        "required_metric_failures": list(EXPECTED_V131_FAILURES),
        "heldout_refreshed_accuracy": comparison.get("heldout_accuracy"),
        "heldout_action_only_baseline_accuracy": comparison.get(
            "heldout_action_only_accuracy"
        ),
        "heldout_action_order_baseline_accuracy": comparison.get(
            "heldout_action_order_accuracy"
        ),
        "fixture_open_open_mind_v3_dominant_action": open_dominant,
        "no_authorization_expected": True,
    }


def _row_alignment(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
    feature_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    branch_ordinals = _branch_ordinals(manifest_rows)
    candidates_by_ordinal: dict[tuple[int, int], Mapping[str, object]] = {}
    for branch_id, branch_ordinal in branch_ordinals.items():
        for candidate_ordinal, candidate in enumerate(candidate_groups.get(branch_id, ())):
            candidates_by_ordinal[(branch_ordinal, candidate_ordinal)] = candidate
    rows_by_ordinal: dict[tuple[int, int], Mapping[str, object]] = {}
    malformed = 0
    duplicates = 0
    schema_mismatch = 0
    action_mismatches: list[dict[str, object]] = []
    for row in feature_rows:
        if row.get("schema_version") != (
            MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION
        ):
            schema_mismatch += 1
        metadata = _mapping(row.get("non_feature_metadata"))
        set_ordinal = metadata.get("candidate_set_ordinal")
        candidate_ordinal = metadata.get("candidate_ordinal_within_set")
        if not isinstance(set_ordinal, int) or not isinstance(candidate_ordinal, int):
            malformed += 1
            continue
        ordinal = (set_ordinal, candidate_ordinal)
        if ordinal in rows_by_ordinal:
            duplicates += 1
            continue
        rows_by_ordinal[ordinal] = row
    missing = sorted(set(candidates_by_ordinal) - set(rows_by_ordinal))
    extra = sorted(set(rows_by_ordinal) - set(candidates_by_ordinal))
    if schema_mismatch:
        failures.append("v131_feature_row_schema_mismatch")
    if malformed:
        failures.append("v131_feature_row_ordinal_metadata_malformed")
    if duplicates:
        failures.append("v131_feature_row_duplicate_ordinals")
    if missing:
        failures.append("v131_feature_rows_missing_ordinals")
    if extra:
        failures.append("v131_feature_rows_extra_ordinals")
    for ordinal, candidate in sorted(candidates_by_ordinal.items()):
        row = rows_by_ordinal.get(ordinal)
        if row is None:
            continue
        candidate_action = str(candidate.get("candidate_action"))
        row_action = str(_mapping(row.get("candidate_public_features")).get("candidate_action"))
        if candidate_action != row_action:
            if len(action_mismatches) < 8:
                action_mismatches.append(
                    {
                        "ordinal": list(ordinal),
                        "candidate_action": candidate_action,
                        "feature_row_action": row_action,
                    }
                )
    if action_mismatches:
        failures.append("v131_feature_row_candidate_action_mismatch")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "branch_count": len(branch_ordinals),
        "candidate_ordinal_count": len(candidates_by_ordinal),
        "matched_feature_row_count": len(set(candidates_by_ordinal).intersection(rows_by_ordinal)),
        "missing_ordinals": [list(item) for item in missing[:8]],
        "extra_ordinals": [list(item) for item in extra[:8]],
        "action_mismatch_examples": action_mismatches,
        "candidate_rows_by_ordinal": candidates_by_ordinal,
        "feature_rows_by_ordinal": rows_by_ordinal,
    }


def _alignment_summary(alignment: Mapping[str, object]) -> dict[str, object]:
    return {
        "passed": alignment.get("passed"),
        "failures": _list_like(alignment.get("failures")),
        "branch_count": alignment.get("branch_count"),
        "candidate_ordinal_count": alignment.get("candidate_ordinal_count"),
        "matched_feature_row_count": alignment.get("matched_feature_row_count"),
        "missing_ordinals": _list_like(alignment.get("missing_ordinals")),
        "extra_ordinals": _list_like(alignment.get("extra_ordinals")),
        "action_mismatch_examples": _list_like(alignment.get("action_mismatch_examples")),
    }


def _feature_row_forbidden_scan(
    feature_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    feature_paths = sorted(
        {
            str(path)
            for row in feature_rows
            for path in _mapping(row.get("candidate_public_features")).keys()
        }
    )
    extractor_paths = sorted(
        {
            str(path)
            for row in feature_rows
            for path in _list_like(row.get("extractor_source_paths"))
        }
    )
    all_paths = sorted(set(feature_paths) | set(extractor_paths))
    forbidden = [
        {"path": path, "matched_forbidden_parts": _forbidden_matches(path)}
        for path in all_paths
        if _forbidden_matches(path)
    ]
    return {
        "passed": not forbidden,
        "forbidden_feature_path_count": len(forbidden),
        "forbidden_feature_paths": forbidden[:16],
        "scanned_path_count": len(all_paths),
        "forbidden_feature_families": sorted(V132_FORBIDDEN_FEATURE_PATH_PARTS),
    }


def _dataset(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
    row_alignment: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    branch_ordinals = _branch_ordinals(manifest_rows)
    feature_rows_by_ordinal = _mapping(row_alignment.get("feature_rows_by_ordinal"))
    rows: list[dict[str, object]] = []
    for manifest in sorted(manifest_rows, key=lambda row: str(row.get("branch_id"))):
        branch_id = _branch_id(manifest)
        if branch_id is None:
            continue
        branch_ordinal = branch_ordinals.get(branch_id)
        fixture_group = _fixture_group(manifest)
        repaired_action = str(manifest.get("repaired_action"))
        repaired_archive_id = manifest.get("repaired_archive_row_id")
        for candidate_ordinal, candidate in enumerate(candidate_groups.get(branch_id, ())):
            ordinal = (branch_ordinal, candidate_ordinal)
            feature_row = _mapping(feature_rows_by_ordinal.get(ordinal))
            features = _mapping(feature_row.get("candidate_public_features"))
            metadata = _mapping(feature_row.get("non_feature_metadata"))
            feature_key = _feature_key(features)
            rows.append(
                {
                    "branch_id": branch_id,
                    "candidate_set_ordinal": branch_ordinal,
                    "candidate_ordinal_within_set": candidate_ordinal,
                    "split": metadata.get("split"),
                    "fixture_group": fixture_group,
                    "candidate_action": features.get("candidate_action")
                    or candidate.get("candidate_action"),
                    "repaired_action": repaired_action,
                    "repaired_archive_row_id": repaired_archive_id,
                    "archive_row_id": candidate.get("archive_row_id"),
                    "is_positive": candidate.get("archive_row_id") == repaired_archive_id,
                    "material_gain_label": candidate.get("material_gain_label") is True,
                    "candidate_public_features": dict(features),
                    "feature_key": feature_key,
                    "legal": features.get("candidate_action_observation_legal") is True,
                }
            )
    return tuple(rows)


def _score_tables(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    train = [row for row in dataset if row.get("split") == "train"]
    return {
        "action": _rate_table(train, lambda row: (row.get("candidate_action"),)),
        "feature": _rate_table(train, lambda row: row.get("feature_key")),
        "train_branch_count": len({row.get("branch_id") for row in train}),
        "train_candidate_row_count": len(train),
    }


def _rate_table(rows: Sequence[Mapping[str, object]], key_fn) -> dict[object, dict[str, object]]:
    totals: Counter[object] = Counter()
    positives: Counter[object] = Counter()
    negatives: Counter[object] = Counter()
    for row in rows:
        key = key_fn(row)
        totals[key] += 1
        if row.get("is_positive") is True:
            positives[key] += 1
        else:
            negatives[key] += 1
    return {
        key: {
            "support_count": totals[key],
            "positive_count": positives[key],
            "negative_count": negatives[key],
            "score": _ratio(positives[key], totals[key]),
        }
        for key in totals
    }


def _branch_reports(
    *,
    dataset: Sequence[Mapping[str, object]],
    score_tables: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in dataset:
        grouped[str(row.get("branch_id"))].append(row)
    feature_table = _mapping(score_tables.get("feature"))
    action_table = _mapping(score_tables.get("action"))
    branch_reports: list[dict[str, object]] = []
    precomputed_candidates_by_branch: dict[str, list[dict[str, object]]] = {}
    for branch_id, candidates in sorted(grouped.items()):
        scored = [
            _candidate_score(row, feature_table=feature_table, action_table=action_table)
            for row in candidates
        ]
        legal = [row for row in scored if row["legal"] is True]
        selected = sorted(
            legal or scored,
            key=lambda row: (
                -_float(row.get("refreshed_feature_score")),
                str(row.get("candidate_action")),
                stable_payload_digest(_mapping(row.get("candidate_public_features"))),
            ),
        )[0]
        precomputed_candidates_by_branch[branch_id] = scored
        positive = next((row for row in scored if row.get("is_positive") is True), scored[0])
        branch_reports.append(
            {
                "branch_id": branch_id,
                "candidate_set_ordinal": positive.get("candidate_set_ordinal"),
                "split": positive.get("split"),
                "fixture_group": positive.get("fixture_group"),
                "repaired_action": positive.get("repaired_action"),
                "predicted_action": selected.get("candidate_action"),
                "prediction_correct": selected.get("is_positive") is True,
                "candidate_count": len(scored),
                "legal_candidate_count": sum(1 for row in scored if row["legal"] is True),
                "material_positive_candidate_available": any(
                    row.get("material_gain_label") is True for row in scored
                ),
                "selected_candidate_material_flag": selected.get("material_gain_label") is True,
                "selected_candidate_ordinal": selected.get("candidate_ordinal_within_set"),
                "selected_action_only_baseline_score": selected.get("action_only_baseline_score"),
                "selected_refreshed_feature_score": selected.get("refreshed_feature_score"),
                "selected_fallback_or_action_prior_used": selected.get(
                    "fallback_or_action_prior_used"
                ),
                "selected_feature_key_support_count": selected.get(
                    "feature_key_support_count"
                ),
                "selected_feature_key_positive_count": selected.get(
                    "feature_key_positive_count"
                ),
                "exact_competing_candidate_scores": scored,
            }
        )
    open_dominant = _open_fixture_dominant(branch_reports)
    finalized: list[dict[str, object]] = []
    for report in branch_reports:
        branch_id = str(report.get("branch_id"))
        scored = precomputed_candidates_by_branch.get(branch_id, [])
        classification = _classify_branch(report, scored, open_dominant)
        finalized.append(
            {
                **report,
                "slice_memberships": _slice_memberships(report),
                "blocker_class": classification,
                "is_failed_branch": _is_failed_branch(report, open_dominant),
            }
        )
    return tuple(finalized)


def _candidate_score(
    row: Mapping[str, object],
    *,
    feature_table: Mapping[object, object],
    action_table: Mapping[object, object],
) -> dict[str, object]:
    feature_key = row.get("feature_key")
    action_key = (row.get("candidate_action"),)
    feature_stats = _mapping(feature_table.get(feature_key))
    action_stats = _mapping(action_table.get(action_key))
    action_score = _float(action_stats.get("score"))
    feature_support = _int(feature_stats.get("support_count"))
    fallback = feature_support <= 0
    refreshed_score = action_score if fallback else _float(feature_stats.get("score"))
    return {
        "branch_id": row.get("branch_id"),
        "candidate_set_ordinal": row.get("candidate_set_ordinal"),
        "candidate_ordinal_within_set": row.get("candidate_ordinal_within_set"),
        "split": row.get("split"),
        "fixture_group": row.get("fixture_group"),
        "candidate_action": row.get("candidate_action"),
        "repaired_action": row.get("repaired_action"),
        "repaired_archive_row_id": row.get("repaired_archive_row_id"),
        "archive_row_id": row.get("archive_row_id"),
        "legal": row.get("legal") is True,
        "is_positive": row.get("is_positive") is True,
        "is_repaired_candidate": row.get("is_positive") is True,
        "material_gain_label": row.get("material_gain_label") is True,
        "action_only_baseline_score": action_score,
        "refreshed_feature_score": refreshed_score,
        "fallback_or_action_prior_used": fallback,
        "feature_key_support_count": feature_support,
        "feature_key_positive_count": _int(feature_stats.get("positive_count")),
        "feature_key_negative_count": _int(feature_stats.get("negative_count")),
        "feature_key_digest": stable_payload_digest(row.get("feature_key")),
        "candidate_public_features": row.get("candidate_public_features"),
    }


def _blocker_rows(
    branch_reports: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    return tuple(
        dict(report)
        for report in branch_reports
        if report.get("split") in {"validation", "test"}
        or report.get("fixture_group") == "open_mind_v3"
    )


def _slice_memberships(report: Mapping[str, object]) -> list[str]:
    memberships: list[str] = []
    if report.get("split") in {"validation", "test"}:
        memberships.append("heldout_validation_test")
    if report.get("fixture_group") == "open_mind_v3":
        memberships.append("fixture_open_mind_v3")
    return memberships


def _open_fixture_dominant(branch_reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts = Counter(
        str(row.get("predicted_action"))
        for row in branch_reports
        if row.get("fixture_group") == "open_mind_v3"
    )
    if not counts:
        return {"action": None, "count": 0, "share": 0.0, "total": 0, "failed": False}
    action, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    total = sum(counts.values())
    return {
        "action": action,
        "count": count,
        "share": _ratio(count, total),
        "total": total,
        "failed": _ratio(count, total) > 0.5,
    }


def _is_failed_branch(
    report: Mapping[str, object],
    open_dominant: Mapping[str, object],
) -> bool:
    if report.get("prediction_correct") is not True:
        return True
    return (
        report.get("fixture_group") == "open_mind_v3"
        and open_dominant.get("failed") is True
        and str(report.get("predicted_action")) == str(open_dominant.get("action"))
    )


def _classify_branch(
    report: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    open_dominant: Mapping[str, object],
) -> str:
    if (
        report.get("fixture_group") == "open_mind_v3"
        and open_dominant.get("failed") is True
        and str(report.get("predicted_action")) == str(open_dominant.get("action"))
    ):
        return "fixture_open_action_collapse"
    if report.get("prediction_correct") is True:
        return "inconclusive"
    selected = _selected_candidate(report, candidates)
    repaired = [row for row in candidates if row.get("is_repaired_candidate") is True]
    if any(
        row.get("feature_key_digest") == selected.get("feature_key_digest")
        for row in repaired
    ):
        return "feature_alias_collision"
    if _int(selected.get("feature_key_support_count")) <= 0:
        return "train_support_hole"
    if _top_score_tie_count(candidates) > 1 or selected.get("fallback_or_action_prior_used") is True:
        return "action_prior_tie_or_fallback"
    if (
        report.get("material_positive_candidate_available") is True
        and report.get("selected_candidate_material_flag") is not True
    ):
        return "material_candidate_not_selected"
    if repaired:
        best_repaired = max(_float(row.get("refreshed_feature_score")) for row in repaired)
        if _float(selected.get("refreshed_feature_score")) >= best_repaired:
            return "insufficient_public_context_signal"
    return "inconclusive"


def _selected_candidate(
    report: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    ordinal = report.get("selected_candidate_ordinal")
    for row in candidates:
        if row.get("candidate_ordinal_within_set") == ordinal:
            return row
    return candidates[0] if candidates else {}


def _top_score_tie_count(candidates: Sequence[Mapping[str, object]]) -> int:
    if not candidates:
        return 0
    top = max(_float(row.get("refreshed_feature_score")) for row in candidates)
    return sum(
        1
        for row in candidates
        if abs(_float(row.get("refreshed_feature_score")) - top) < 1e-12
    )


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    class_counts = Counter(str(row.get("blocker_class")) for row in rows)
    failed_class_counts = Counter(
        str(row.get("blocker_class"))
        for row in rows
        if row.get("is_failed_branch") is True
    )
    split_counts = Counter(str(row.get("split")) for row in rows)
    fixture_counts = Counter(str(row.get("fixture_group")) for row in rows)
    pair_counts = Counter(
        f"{row.get('repaired_action')}->{row.get('predicted_action')}"
        for row in rows
    )
    heldout_rows = [row for row in rows if row.get("split") in {"validation", "test"}]
    heldout_failures = [
        row for row in heldout_rows if row.get("prediction_correct") is not True
    ]
    open_rows = [row for row in rows if row.get("fixture_group") == "open_mind_v3"]
    open_dominant = _open_fixture_dominant(open_rows)
    fixture_failures = [
        row
        for row in open_rows
        if open_dominant.get("failed") is True
        and str(row.get("predicted_action")) == str(open_dominant.get("action"))
    ]
    explained_classes = set(BLOCKER_CLASSES) - {"inconclusive"}
    heldout_explained = [
        row for row in heldout_failures if row.get("blocker_class") in explained_classes
    ]
    fixture_explained = [
        row for row in fixture_failures if row.get("blocker_class") in explained_classes
    ]
    return {
        "counts_by_blocker_class": dict(sorted(class_counts.items())),
        "counts_by_failed_blocker_class": dict(sorted(failed_class_counts.items())),
        "counts_by_split": dict(sorted(split_counts.items())),
        "counts_by_fixture_group": dict(sorted(fixture_counts.items())),
        "counts_by_repaired_predicted_action_pair": dict(sorted(pair_counts.items())),
        "heldout_branch_count": len(heldout_rows),
        "fixture_open_branch_count": len(open_rows),
        "detail_row_count": len(rows),
        "heldout_failures": {
            "failed_branch_count": len(heldout_failures),
            "explained_count": len(heldout_explained),
            "unexplained_count": len(heldout_failures) - len(heldout_explained),
        },
        "fixture_open_failures": {
            "failed_branch_count": len(fixture_failures),
            "explained_count": len(fixture_explained),
            "unexplained_count": len(fixture_failures) - len(fixture_explained),
            "dominant_predicted_action": open_dominant,
        },
    }


def _recommendation(aggregate: Mapping[str, object]) -> dict[str, object]:
    counts = Counter({
        str(key): _int(value)
        for key, value in _mapping(aggregate.get("counts_by_blocker_class")).items()
    })
    failed_counts = Counter({
        str(key): _int(value)
        for key, value in _mapping(
            aggregate.get("counts_by_failed_blocker_class")
        ).items()
    })
    support_hole_failures = (
        failed_counts.get("train_support_hole", 0)
        + failed_counts.get("action_prior_tie_or_fallback", 0)
    )
    fixture_open_failures = failed_counts.get("fixture_open_action_collapse", 0)
    alias_failures = failed_counts.get("feature_alias_collision", 0)
    insufficient_failures = failed_counts.get("insufficient_public_context_signal", 0)
    if alias_failures >= max(1, support_hole_failures, fixture_open_failures):
        next_step = "repair_feature_key_collision_surface"
    elif fixture_open_failures >= max(1, support_hole_failures, insufficient_failures):
        next_step = "add_public_rollout_history_context"
    elif support_hole_failures > 0:
        next_step = "collect_targeted_public_context_rows"
    elif insufficient_failures > 0:
        next_step = "stop_ranker_path_insufficient_signal"
    elif counts.get("feature_alias_collision", 0) >= max(1, counts.get("train_support_hole", 0)):
        next_step = "repair_feature_key_collision_surface"
    elif counts.get("fixture_open_action_collapse", 0) > 0:
        next_step = "add_public_rollout_history_context"
    elif counts.get("train_support_hole", 0) > 0 or counts.get("action_prior_tie_or_fallback", 0) > 0:
        next_step = "collect_targeted_public_context_rows"
    elif counts.get("insufficient_public_context_signal", 0) > 0:
        next_step = "stop_ranker_path_insufficient_signal"
    else:
        next_step = "inconclusive_requires_manual_review"
    return {
        "next_step": next_step,
        "allowed_next_steps": list(RECOMMENDATIONS),
        "selection_policy": "largest_failed_blocker_class_before_nonfailed_support_notes",
        "failed_blocker_class_counts": dict(sorted(failed_counts.items())),
        "support_hole_failed_count": support_hole_failures,
        "fixture_open_action_collapse_failed_count": fixture_open_failures,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "training_executed": False,
        "runtime_policy_change_recommended": False,
        "claim_causality": False,
    }


def _classification(source_integrity: Mapping[str, object]) -> dict[str, object]:
    primary = (
        "refreshed_surface_blocker_slice_audit_completed"
        if source_integrity.get("passed") is True
        else "refreshed_surface_blocker_slice_audit_source_integrity_failed"
    )
    return {
        "primary": primary,
        "labels": [primary],
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
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


def _score_table_summary(score_tables: Mapping[str, object]) -> dict[str, object]:
    action = _mapping(score_tables.get("action"))
    feature = _mapping(score_tables.get("feature"))
    return {
        "train_branch_count": score_tables.get("train_branch_count"),
        "train_candidate_row_count": score_tables.get("train_candidate_row_count"),
        "action_key_count": len(action),
        "refreshed_feature_key_count": len(feature),
    }


def _slice_extraction_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    heldout = [row for row in rows if row.get("split") in {"validation", "test"}]
    fixture_open = [row for row in rows if row.get("fixture_group") == "open_mind_v3"]
    return {
        "heldout_validation_test_branch_count": len(heldout),
        "fixture_open_mind_v3_branch_count": len(fixture_open),
        "union_detail_row_count": len(rows),
        "expected_real_heldout_validation_test_branch_count": 18,
        "expected_real_fixture_open_mind_v3_branch_count": 10,
    }


def _detail_row_for_output(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": (
            "mind_v3_first_recovery_refreshed_surface_blocker_slice_row_v1"
        ),
        "candidate_set_ordinal": row.get("candidate_set_ordinal"),
        "slice_memberships": _list_like(row.get("slice_memberships")),
        "split": row.get("split"),
        "fixture_group": row.get("fixture_group"),
        "repaired_action": row.get("repaired_action"),
        "predicted_action": row.get("predicted_action"),
        "prediction_correct": row.get("prediction_correct"),
        "candidate_count": row.get("candidate_count"),
        "legal_candidate_count": row.get("legal_candidate_count"),
        "material_positive_candidate_available": row.get(
            "material_positive_candidate_available"
        ),
        "selected_candidate_material_flag": row.get("selected_candidate_material_flag"),
        "action_only_baseline_score": row.get("selected_action_only_baseline_score"),
        "refreshed_feature_score": row.get("selected_refreshed_feature_score"),
        "fallback_or_action_prior_used": row.get(
            "selected_fallback_or_action_prior_used"
        ),
        "feature_key_support_count_in_train": row.get(
            "selected_feature_key_support_count"
        ),
        "exact_competing_candidate_scores": [
            _candidate_score_for_output(candidate)
            for candidate in _list_like(row.get("exact_competing_candidate_scores"))
        ],
        "blocker_class": row.get("blocker_class"),
        "is_failed_branch": row.get("is_failed_branch"),
        "diagnostics_only": True,
        "training_authorized": False,
        "runtime_policy_authorized": False,
    }


def _candidate_score_for_output(candidate: Mapping[str, object]) -> dict[str, object]:
    return {
        "candidate_ordinal_within_set": candidate.get("candidate_ordinal_within_set"),
        "candidate_action": candidate.get("candidate_action"),
        "legal": candidate.get("legal"),
        "is_repaired_candidate": candidate.get("is_repaired_candidate"),
        "material_gain_label": candidate.get("material_gain_label"),
        "action_only_baseline_score": candidate.get("action_only_baseline_score"),
        "refreshed_feature_score": candidate.get("refreshed_feature_score"),
        "fallback_or_action_prior_used": candidate.get("fallback_or_action_prior_used"),
        "feature_key_support_count_in_train": candidate.get("feature_key_support_count"),
        "feature_key_positive_count_in_train": candidate.get("feature_key_positive_count"),
        "feature_key_negative_count_in_train": candidate.get("feature_key_negative_count"),
        "feature_key_digest": candidate.get("feature_key_digest"),
    }


def _branch_ordinals(manifest_rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        str(manifest.get("branch_id")): index
        for index, manifest in enumerate(
            sorted(manifest_rows, key=lambda row: str(row.get("branch_id")))
        )
        if _branch_id(manifest) is not None
    }


def _branch_id(row: Mapping[str, object]) -> str | None:
    value = row.get("branch_id")
    return value if isinstance(value, str) and value else None


def _forbidden_matches(path: str) -> list[str]:
    lowered = path.lower()
    tokens = set(_path_tokens(lowered))
    matches = sorted(V132_FORBIDDEN_FEATURE_PATH_PARTS & tokens)
    for substring in V132_FORBIDDEN_FEATURE_PATH_SUBSTRINGS:
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


def _float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)
