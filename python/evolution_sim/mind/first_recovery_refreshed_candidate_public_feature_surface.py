from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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
    FORBIDDEN_FEATURE_PATH_PARTS,
    FORBIDDEN_FEATURE_PATH_SUBSTRINGS,
    MAX_EXAMPLES,
    MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION,
    MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_candidate_ranker_capacity_audit import (
    ACTION_ONLY_BASELINE,
    ACTION_ORDER_BASELINE,
    DEFAULT_OUTPUT_PATH as DEFAULT_V128_REPORT_PATH,
    EXPECTED_CANDIDATE_ROW_COUNT,
    MIND_V3_FIRST_RECOVERY_CANDIDATE_RANKER_CAPACITY_AUDIT_SCHEMA_VERSION,
    _action_distribution,
    _fixture_open_evaluation,
    _heldout_accuracy,
    _list_like,
    _material_gain_recall,
    _metrics,
    _number,
    _positive_rate_table,
    _ratio,
    _score_lookup,
    _seed29_evaluation,
    _unsupported_action_audit,
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
from evolution_sim.mind.first_recovery_observation_candidate_context import (
    DEFAULT_CONTEXT_ROWS_OUTPUT_PATH as DEFAULT_V130_CONTEXT_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V130_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_ROW_SCHEMA_VERSION,
    MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _int,
    _mapping,
    _resolve_archive_rows,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    _stratified_three_way_assignments,
    _trainable_leakage,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION = (
    "mind_v3_first_recovery_refreshed_candidate_public_feature_surface_v1"
)
MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_POLICY = (
    "diagnostics_only_first_recovery_v131_refreshed_candidate_public_feature_surface_v1"
)
MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION = (
    "mind_v3_first_recovery_refreshed_candidate_public_feature_row_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v131-first-recovery-refreshed-candidate-public-feature-surface.json"
)
DEFAULT_FEATURE_ROWS_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v131-first-recovery-refreshed-candidate-public-feature-rows.jsonl"
)

PRIMARY_PROBE = "refreshed_public_context_probe"
ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "refreshed_candidate_public_feature_surface_source_integrity_failed",
    "refreshed_candidate_public_feature_surface_blocked_by_signal",
    "refreshed_candidate_public_feature_surface_ready_for_ranker_review",
)
V130_READY_CLASSIFICATION = "observation_candidate_context_ready_for_v129_surface_refresh"
V131_FORBIDDEN_FEATURE_PATH_PARTS: frozenset[str] = frozenset(
    set(FORBIDDEN_FEATURE_PATH_PARTS)
    | {
        "oracle",
        "private",
        "private_world_state",
        "world",
    }
)
V131_FORBIDDEN_FEATURE_PATH_SUBSTRINGS: tuple[str, ...] = tuple(
    sorted(
        set(FORBIDDEN_FEATURE_PATH_SUBSTRINGS)
        | {
            "oracle_score",
            "private_world_state",
            "world_state",
        }
    )
)
JOIN_PROOF_SCHEMA_VERSION = (
    "mind_v3_first_recovery_v131_candidate_context_join_proof_v1"
)


@dataclass(frozen=True, slots=True)
class FirstRecoveryRefreshedCandidatePublicFeatureSurfaceBuild:
    report: dict[str, object]
    feature_rows: tuple[dict[str, object], ...]


def build_first_recovery_refreshed_candidate_public_feature_surface(
    *,
    v130_report: Mapping[str, object] | None = None,
    v130_report_path: str | Path | None = DEFAULT_V130_REPORT_PATH,
    v130_context_rows: Sequence[Mapping[str, object]] | None = None,
    v130_context_rows_path: str | Path | None = DEFAULT_V130_CONTEXT_ROWS_PATH,
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
) -> FirstRecoveryRefreshedCandidatePublicFeatureSurfaceBuild:
    v130_payload, v130_evidence = _resolve_json_report(
        v130_report,
        v130_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_SCHEMA_VERSION,
    )
    v130_rows, v130_rows_evidence = _resolve_jsonl_rows(
        v130_context_rows,
        v130_context_rows_path,
        row_kind="v130_context_rows",
    )
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
    alignment = _ordinal_alignment(
        manifest_rows=manifest_rows,
        candidate_groups=candidate_groups,
        v129_feature_rows=v129_rows,
        v130_context_rows=v130_rows,
    )
    source_reports = {
        "v130_report": v130_evidence,
        "v130_context_rows": v130_rows_evidence,
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
    }
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v130_report=v130_payload,
        v130_context_rows=v130_rows,
        v129_report=v129_payload,
        v129_feature_rows=v129_rows,
        v128_report=v128_payload,
        v127_report=v127_payload,
        v127_predictions=v127_predictions,
        v124_report=v124_payload,
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
        candidate_set_audit=candidate_set_audit,
        alignment=alignment,
    )
    dataset, feature_rows = _refreshed_feature_dataset(
        manifest_rows=manifest_rows,
        candidate_groups=candidate_groups,
        alignment=alignment,
    )
    allowlist = _feature_allowlist(dataset)
    forbidden = _forbidden_feature_scan(dataset)
    leakage = _leakage_audit(
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
        forbidden_scan=forbidden,
    )
    variance = _within_branch_variance(dataset)
    probes = (
        _run_probes(dataset)
        if source_integrity["passed"]
        and allowlist["passed"]
        and forbidden["passed"]
        and leakage["passed"]
        else _empty_probe_reports()
    )
    probe_comparison = _probe_comparison(probes)
    primary = _mapping(probes.get(PRIMARY_PROBE))
    metric_gate = _metric_gate(
        source_integrity=source_integrity,
        allowlist=allowlist,
        forbidden=forbidden,
        leakage=leakage,
        variance=variance,
        probes=probes,
        probe_comparison=probe_comparison,
        primary=primary,
    )
    classification = _classification(
        source_integrity=source_integrity,
        allowlist=allowlist,
        forbidden=forbidden,
        leakage=leakage,
        metric_gate=metric_gate,
    )
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION
        ),
        "audit_policy": (
            MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_POLICY
        ),
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "feature_rules": _feature_rules(),
        "feature_allowlist": allowlist,
        "forbidden_feature_scan": forbidden,
        "leakage_audit": leakage,
        "within_branch_variance": variance,
        "feature_surface_summary": _feature_surface_summary(dataset, feature_rows),
        "probe_definitions": _probe_definitions(),
        "probe_reports": probes,
        "probe_comparison": probe_comparison,
        "heldout_evaluation": _heldout_evaluation(probes),
        "seed29_evaluation": primary.get("seed29_evaluation", {}),
        "fixture_open_evaluation": primary.get("fixture_open_evaluation", {}),
        "action_distribution": primary.get("action_distribution", {}),
        "unsupported_action_audit": primary.get("unsupported_action_audit", {}),
        "material_exact_match_recall": primary.get("material_gain_recall", {}),
        "heldout_material_exact_match_recall": (
            _heldout_material_exact_match_recall(primary)
        ),
        "metric_gate": metric_gate,
        "classification": classification,
        "authorization_block": _authorization_block(),
        "recommendation": _recommendation(classification),
        "non_promoted": True,
    }
    return FirstRecoveryRefreshedCandidatePublicFeatureSurfaceBuild(
        report=report,
        feature_rows=tuple(feature_rows),
    )


def write_first_recovery_refreshed_candidate_public_feature_surface_report(
    build: FirstRecoveryRefreshedCandidatePublicFeatureSurfaceBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_first_recovery_refreshed_candidate_public_feature_rows(
    build: FirstRecoveryRefreshedCandidatePublicFeatureSurfaceBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in build.feature_rows:
            json.dump(row, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "report_only_refreshed_candidate_public_feature_surface": True,
        "v130_observation_candidate_context_required": True,
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


def _feature_rules() -> dict[str, object]:
    return {
        "allowed_feature_sources": [
            "v129 candidate_action/action-derived public fields",
            "v130 observation-time target/resource/neighborhood candidate context",
        ],
        "v129_branch_level_context_excluded": True,
        "v130_context_trainable_public_only": True,
        "non_feature_metadata_separated": True,
        "forbidden_feature_path_parts": sorted(V131_FORBIDDEN_FEATURE_PATH_PARTS),
        "forbidden_feature_path_substrings": list(
            V131_FORBIDDEN_FEATURE_PATH_SUBSTRINGS
        ),
    }


def _source_integrity(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    v130_report: Mapping[str, object] | None,
    v130_context_rows: Sequence[Mapping[str, object]],
    v129_report: Mapping[str, object] | None,
    v129_feature_rows: Sequence[Mapping[str, object]],
    v128_report: Mapping[str, object] | None,
    v127_report: Mapping[str, object] | None,
    v127_predictions: Sequence[Mapping[str, object]],
    v124_report: Mapping[str, object] | None,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    candidate_set_audit: Mapping[str, object],
    alignment: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")

    v130 = _mapping(v130_report or {})
    v129 = _mapping(v129_report or {})
    v128 = _mapping(v128_report or {})
    v127 = _mapping(v127_report or {})
    v124 = _mapping(v124_report or {})
    v130_classification = _mapping(v130.get("classification")).get("primary")
    v130_source = _mapping(v130.get("source_integrity"))
    v130_summary = _mapping(v130.get("context_rows_summary"))
    v130_availability = _mapping(v130.get("candidate_context_availability"))
    v130_forbidden = _mapping(v130.get("forbidden_field_scan"))
    v130_variance = _mapping(v130.get("within_branch_variance"))
    v130_target_variance = _mapping(
        v130_variance.get("target_resource_neighborhood_variance")
    )
    v129_source = _mapping(v129.get("source_integrity"))
    v128_source = _mapping(v128.get("source_integrity"))
    v127_source = _mapping(v127.get("source_integrity"))
    v124_source = _mapping(v124.get("source_integrity"))
    v124_contract = _mapping(v124.get("contract_checks"))
    v124_manifest = _mapping(v124.get("manifest"))
    manifest_digest = stable_payload_digest(list(manifest_rows))
    prediction_digest = stable_payload_digest(list(v127_predictions))
    row_digests = _source_row_digests(
        v129_report=v129,
        v129_feature_rows=v129_feature_rows,
        v130_report=v130,
        v130_context_rows=v130_context_rows,
    )

    if v130_classification != V130_READY_CLASSIFICATION:
        failures.append("v130_classification_not_ready_for_surface_refresh")
    if v130_source.get("passed") is not True:
        failures.append("v130_source_integrity_not_passed")
    if _int(v130_summary.get("row_count")) != len(v130_context_rows):
        failures.append("v130_context_row_count_mismatch")
    if len(v130_context_rows) != EXPECTED_CANDIDATE_ROW_COUNT:
        failures.append("v130_context_rows_not_542")
    if _int(v130_availability.get("candidate_row_count")) != len(v130_context_rows):
        failures.append("v130_availability_row_count_mismatch")
    if _list_like(v130_availability.get("missing_public_observation_fields")):
        failures.append("v130_missing_public_observation_fields")
    if _int(v130_forbidden.get("forbidden_feature_path_count")) != 0:
        failures.append("v130_forbidden_feature_paths_nonzero")
    if _int(v130_target_variance.get("varying_path_count")) <= 0:
        failures.append("v130_target_resource_neighborhood_variance_missing")
    v130_digest = _mapping(row_digests.get("v130_context_rows"))
    if v130_digest.get("report_digest_present") is not True:
        failures.append("v130_context_rows_digest_missing")
    elif v130_digest.get("matches_report_digest") is not True:
        failures.append("v130_context_rows_digest_mismatch")

    v129_summary = _mapping(v129.get("feature_surface_summary"))
    if v129_source.get("passed") is not True:
        failures.append("v129_source_integrity_not_passed")
    if _int(v129_summary.get("candidate_feature_row_count")) != len(v129_feature_rows):
        failures.append("v129_feature_row_count_mismatch")
    if len(v129_feature_rows) != EXPECTED_CANDIDATE_ROW_COUNT:
        failures.append("v129_feature_rows_not_542")
    v129_digest = _mapping(row_digests.get("v129_feature_rows"))
    if v129_digest.get("report_digest_present") is not True:
        failures.append("v129_feature_rows_digest_missing")
    elif v129_digest.get("matches_report_digest") is not True:
        failures.append("v129_feature_rows_digest_mismatch")
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
    if alignment.get("passed") is not True:
        failures.append("candidate_context_row_alignment_failed")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "manifest_row_count": len(manifest_rows),
        "manifest_digest": manifest_digest,
        "candidate_row_count": len(candidate_rows),
        "v130_classification": v130_classification,
        "v130_source_integrity_passed": v130_source.get("passed"),
        "v130_context_row_count": len(v130_context_rows),
        "v130_target_resource_neighborhood_varying_path_count": _int(
            v130_target_variance.get("varying_path_count")
        ),
        "v130_missing_public_observation_fields": _list_like(
            v130_availability.get("missing_public_observation_fields")
        ),
        "v130_forbidden_feature_path_count": _int(
            v130_forbidden.get("forbidden_feature_path_count")
        ),
        "v129_source_integrity_passed": v129_source.get("passed"),
        "v128_source_integrity_passed": v128_source.get("passed"),
        "v127_source_integrity_passed": v127_source.get("passed"),
        "v124_source_integrity_passed": v124_source.get("passed"),
        "candidate_set_audit": dict(candidate_set_audit),
        "candidate_context_row_alignment": _alignment_summary(alignment),
        "source_row_digests": row_digests,
    }


def _source_row_digests(
    *,
    v129_report: Mapping[str, object],
    v129_feature_rows: Sequence[Mapping[str, object]],
    v130_report: Mapping[str, object],
    v130_context_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    v129_computed = stable_payload_digest(list(v129_feature_rows))
    v130_computed = stable_payload_digest(list(v130_context_rows))
    v129_declared = _declared_digest(
        v129_report,
        (
            ("feature_surface_summary", "feature_rows_digest"),
            ("feature_surface_summary", "candidate_feature_rows_digest"),
            ("source_integrity", "v129_feature_rows_digest"),
        ),
    )
    v130_declared = _declared_digest(
        v130_report,
        (
            ("context_rows_summary", "context_rows_digest"),
            ("context_rows_summary", "candidate_context_rows_digest"),
            ("source_integrity", "v130_context_rows_digest"),
        ),
    )
    return {
        "v129_feature_rows": _row_digest_report(
            computed_digest=v129_computed,
            report_digest=v129_declared,
        ),
        "v130_context_rows": _row_digest_report(
            computed_digest=v130_computed,
            report_digest=v130_declared,
        ),
    }


def _declared_digest(
    report: Mapping[str, object],
    paths: Sequence[tuple[str, str]],
) -> str | None:
    for section, field in paths:
        value = _mapping(report.get(section)).get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _row_digest_report(
    *,
    computed_digest: str,
    report_digest: str | None,
) -> dict[str, object]:
    return {
        "computed_digest": computed_digest,
        "report_digest": report_digest,
        "report_digest_present": report_digest is not None,
        "matches_report_digest": (
            None if report_digest is None else report_digest == computed_digest
        ),
    }


def _ordinal_alignment(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
    v129_feature_rows: Sequence[Mapping[str, object]],
    v130_context_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    branch_ordinals = {
        _branch_id(manifest): index
        for index, manifest in enumerate(
            sorted(manifest_rows, key=lambda row: str(row.get("branch_id")))
        )
        if _branch_id(manifest) is not None
    }
    candidates_by_ordinal: dict[tuple[int, int], Mapping[str, object]] = {}
    for branch_id, branch_ordinal in branch_ordinals.items():
        for candidate_ordinal, row in enumerate(candidate_groups.get(branch_id, ())):
            candidates_by_ordinal[(branch_ordinal, candidate_ordinal)] = row

    v129_by_ordinal, v129_failures = _rows_by_ordinal(
        v129_feature_rows,
        expected_schema=MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION,
        row_kind="v129_feature_rows",
    )
    v130_by_ordinal, v130_failures = _rows_by_ordinal(
        v130_context_rows,
        expected_schema=MIND_V3_FIRST_RECOVERY_OBSERVATION_CANDIDATE_CONTEXT_ROW_SCHEMA_VERSION,
        row_kind="v130_context_rows",
    )
    failures.extend(v129_failures)
    failures.extend(v130_failures)

    missing_v129 = sorted(set(candidates_by_ordinal) - set(v129_by_ordinal))
    extra_v129 = sorted(set(v129_by_ordinal) - set(candidates_by_ordinal))
    missing_v130 = sorted(set(candidates_by_ordinal) - set(v130_by_ordinal))
    extra_v130 = sorted(set(v130_by_ordinal) - set(candidates_by_ordinal))
    if missing_v129:
        failures.append("v129_missing_candidate_ordinals")
    if extra_v129:
        failures.append("v129_extra_candidate_ordinals")
    if missing_v130:
        failures.append("v130_missing_candidate_ordinals")
    if extra_v130:
        failures.append("v130_extra_candidate_ordinals")

    action_mismatches: list[dict[str, object]] = []
    join_digest_mismatches: list[dict[str, object]] = []
    expected_join_digests: dict[tuple[int, int], str] = {}
    observed_join_digest_count = 0
    for ordinal, candidate in sorted(candidates_by_ordinal.items()):
        v129_row = v129_by_ordinal.get(ordinal)
        v130_row = v130_by_ordinal.get(ordinal)
        if v129_row is None or v130_row is None:
            continue
        expected_join_digest = _candidate_context_join_digest(
            ordinal=ordinal,
            candidate=candidate,
            context_row=v130_row,
        )
        expected_join_digests[ordinal] = expected_join_digest
        observed_join_digest = _observed_context_join_digest(v130_row)
        if observed_join_digest is not None:
            observed_join_digest_count += 1
            if observed_join_digest != expected_join_digest:
                if len(join_digest_mismatches) < MAX_EXAMPLES:
                    join_digest_mismatches.append(
                        {
                            "ordinal": list(ordinal),
                            "candidate_action": candidate.get("candidate_action"),
                            "expected_join_digest": expected_join_digest,
                            "observed_join_digest": observed_join_digest,
                        }
                    )
        candidate_action = str(candidate.get("candidate_action"))
        v129_action = str(
            _mapping(v129_row.get("candidate_public_features")).get("candidate_action")
        )
        v130_action = str(
            _mapping(v130_row.get("candidate_observation_context")).get(
                "candidate_action"
            )
        )
        if candidate_action != v129_action or candidate_action != v130_action:
            if len(action_mismatches) < MAX_EXAMPLES:
                action_mismatches.append(
                    {
                        "ordinal": list(ordinal),
                        "candidate_action": candidate_action,
                        "v129_candidate_action": v129_action,
                        "v130_candidate_action": v130_action,
                    }
                )
    if action_mismatches:
        failures.append("candidate_action_alignment_mismatch")
    if join_digest_mismatches:
        failures.append("v130_candidate_context_join_digest_mismatch")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "branch_count": len(branch_ordinals),
        "candidate_ordinal_count": len(candidates_by_ordinal),
        "matched_v129_row_count": len(
            set(candidates_by_ordinal).intersection(v129_by_ordinal)
        ),
        "matched_v130_row_count": len(
            set(candidates_by_ordinal).intersection(v130_by_ordinal)
        ),
        "missing_v129_ordinals": [list(item) for item in missing_v129[:MAX_EXAMPLES]],
        "extra_v129_ordinals": [list(item) for item in extra_v129[:MAX_EXAMPLES]],
        "missing_v130_ordinals": [list(item) for item in missing_v130[:MAX_EXAMPLES]],
        "extra_v130_ordinals": [list(item) for item in extra_v130[:MAX_EXAMPLES]],
        "action_mismatch_examples": action_mismatches,
        "candidate_context_join_digest_schema_version": JOIN_PROOF_SCHEMA_VERSION,
        "v130_join_digest_evidence_present": observed_join_digest_count > 0,
        "v130_join_digest_evidence_count": observed_join_digest_count,
        "v130_join_digest_mismatch_examples": join_digest_mismatches,
        "candidate_rows_by_ordinal": candidates_by_ordinal,
        "v129_rows_by_ordinal": v129_by_ordinal,
        "v130_rows_by_ordinal": v130_by_ordinal,
        "expected_join_digests_by_ordinal": expected_join_digests,
    }


def _rows_by_ordinal(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_schema: str,
    row_kind: str,
) -> tuple[dict[tuple[int, int], Mapping[str, object]], list[str]]:
    failures: list[str] = []
    by_ordinal: dict[tuple[int, int], Mapping[str, object]] = {}
    duplicates = 0
    malformed = 0
    schema_mismatch = 0
    for row in rows:
        if row.get("schema_version") != expected_schema:
            schema_mismatch += 1
        meta = _mapping(row.get("non_feature_metadata"))
        set_ordinal = meta.get("candidate_set_ordinal")
        candidate_ordinal = meta.get("candidate_ordinal_within_set")
        if not isinstance(set_ordinal, int) or not isinstance(candidate_ordinal, int):
            malformed += 1
            continue
        key = (set_ordinal, candidate_ordinal)
        if key in by_ordinal:
            duplicates += 1
            continue
        by_ordinal[key] = row
    if schema_mismatch:
        failures.append(f"{row_kind}_schema_mismatch")
    if malformed:
        failures.append(f"{row_kind}_ordinal_metadata_malformed")
    if duplicates:
        failures.append(f"{row_kind}_duplicate_ordinals")
    return by_ordinal, failures


def _candidate_context_join_digest(
    *,
    ordinal: tuple[int, int],
    candidate: Mapping[str, object],
    context_row: Mapping[str, object] | None = None,
) -> str:
    provenance = _mapping(candidate.get("provenance"))
    context = (
        _mapping(context_row.get("candidate_observation_context"))
        if context_row is not None
        else {}
    )
    return stable_payload_digest(
        {
            "schema_version": JOIN_PROOF_SCHEMA_VERSION,
            "branch_id": provenance.get("branch_id"),
            "candidate_set_ordinal": ordinal[0],
            "candidate_ordinal_within_set": ordinal[1],
            "candidate_action": candidate.get("candidate_action"),
            "observation_digest": candidate.get("observation_digest"),
            "candidate_observation_context": context,
        }
    )


def _observed_context_join_digest(row: Mapping[str, object]) -> str | None:
    metadata = _mapping(row.get("non_feature_metadata"))
    for key in (
        "candidate_context_join_digest",
        "audit_join_digest",
        "v131_candidate_context_join_digest",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _alignment_summary(alignment: Mapping[str, object]) -> dict[str, object]:
    return {
        "passed": alignment.get("passed"),
        "failures": _list_like(alignment.get("failures")),
        "branch_count": alignment.get("branch_count"),
        "candidate_ordinal_count": alignment.get("candidate_ordinal_count"),
        "matched_v129_row_count": alignment.get("matched_v129_row_count"),
        "matched_v130_row_count": alignment.get("matched_v130_row_count"),
        "missing_v129_ordinals": _list_like(alignment.get("missing_v129_ordinals")),
        "missing_v130_ordinals": _list_like(alignment.get("missing_v130_ordinals")),
        "action_mismatch_examples": _list_like(
            alignment.get("action_mismatch_examples")
        ),
        "candidate_context_join_digest_schema_version": alignment.get(
            "candidate_context_join_digest_schema_version"
        ),
        "v130_join_digest_evidence_present": alignment.get(
            "v130_join_digest_evidence_present"
        ),
        "v130_join_digest_evidence_count": alignment.get(
            "v130_join_digest_evidence_count"
        ),
        "v130_join_digest_mismatch_examples": _list_like(
            alignment.get("v130_join_digest_mismatch_examples")
        ),
    }


def _refreshed_feature_dataset(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
    alignment: Mapping[str, object],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    assignments = _stratified_three_way_assignments(manifest_rows)
    branch_ordinals = {
        _branch_id(manifest): index
        for index, manifest in enumerate(
            sorted(manifest_rows, key=lambda row: str(row.get("branch_id")))
        )
        if _branch_id(manifest) is not None
    }
    v129_by_ordinal = _mapping(alignment.get("v129_rows_by_ordinal"))
    v130_by_ordinal = _mapping(alignment.get("v130_rows_by_ordinal"))
    join_digests_by_ordinal = _mapping(alignment.get("expected_join_digests_by_ordinal"))
    dataset: list[dict[str, object]] = []
    rows_out: list[dict[str, object]] = []
    for manifest in sorted(manifest_rows, key=lambda row: str(row.get("branch_id"))):
        branch_id = _branch_id(manifest)
        if branch_id is None:
            continue
        branch_ordinal = branch_ordinals.get(branch_id)
        repaired_action = str(manifest.get("repaired_action"))
        repaired_archive_id = manifest.get("repaired_archive_row_id")
        split = assignments.get(branch_id)
        seed = _mapping(manifest.get("non_trainable_audit_metadata")).get("seed")
        fixture_group = _fixture_group(manifest)
        for candidate_ordinal, candidate in enumerate(candidate_groups.get(branch_id, ())):
            ordinal = (branch_ordinal, candidate_ordinal)
            v129_row = _mapping(v129_by_ordinal.get(ordinal))
            v130_row = _mapping(v130_by_ordinal.get(ordinal))
            join_digest = join_digests_by_ordinal.get(ordinal)
            if not isinstance(join_digest, str):
                join_digest = _candidate_context_join_digest(
                    ordinal=ordinal,
                    candidate=candidate,
                )
            extraction = _refreshed_features(v129_row, v130_row)
            features = extraction["features"]
            public_action_mask = dict(
                _mapping(_mapping(candidate.get("trainable_public_input")).get("action_mask"))
            )
            dataset_row = {
                "branch_id": branch_id,
                "split": split,
                "seed": seed,
                "fixture_group": fixture_group,
                "repaired_action": repaired_action,
                "repaired_archive_row_id": repaired_archive_id,
                "archive_row_id": candidate.get("archive_row_id"),
                "candidate_action": features.get("candidate_action"),
                "is_positive": candidate.get("archive_row_id") == repaired_archive_id,
                "material_gain_label": candidate.get("material_gain_label"),
                "candidate_public_features": features,
                "extractor_source_paths": extraction["source_paths"],
                "public_action_mask": public_action_mask,
                "action_key": (features.get("candidate_action"),),
                "refreshed_feature_key": _feature_key(features),
            }
            dataset.append(dataset_row)
            rows_out.append(
                {
                    "schema_version": (
                        MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION
                    ),
                    "candidate_public_features": features,
                    "extractor_source_paths": extraction["source_paths"],
                    "non_feature_metadata": {
                        "candidate_set_ordinal": branch_ordinal,
                        "candidate_ordinal_within_set": candidate_ordinal,
                        "candidate_context_join_digest": join_digest,
                        "candidate_context_join_digest_schema_version": (
                            JOIN_PROOF_SCHEMA_VERSION
                        ),
                        "split_role": "evaluation_only_not_feature",
                        "split": split,
                    },
                    "diagnostics_only": True,
                    "training_authorized": False,
                    "runtime_policy_authorized": False,
                }
            )
    return tuple(dataset), tuple(rows_out)


def _refreshed_features(
    v129_row: Mapping[str, object],
    v130_row: Mapping[str, object],
) -> dict[str, object]:
    v129_features = _mapping(v129_row.get("candidate_public_features"))
    v130_context = _mapping(v130_row.get("candidate_observation_context"))
    features: dict[str, object] = {}
    source_paths: set[str] = set()
    for path, value in sorted(v129_features.items(), key=lambda item: str(item[0])):
        path_text = str(path)
        if path_text == "candidate_action" or path_text.startswith("candidate_action_"):
            features[path_text] = value
            source_paths.add(f"v129_feature_row.candidate_public_features.{path_text}")
    for path, value in sorted(v130_context.items(), key=lambda item: str(item[0])):
        path_text = str(path)
        if path_text.startswith(
            (
                "candidate_target_context.",
                "candidate_resource_context.",
                "candidate_neighborhood_context.",
            )
        ):
            features[path_text] = value
            source_paths.add(
                f"v130_context_row.candidate_observation_context.{path_text}"
            )
    if "candidate_action_observation_legal" not in features:
        features["candidate_action_observation_legal"] = v130_context.get(
            "candidate_action_observation_legal"
        )
        source_paths.add(
            "v130_context_row.candidate_observation_context."
            "candidate_action_observation_legal"
        )
    if "candidate_action" not in features:
        features["candidate_action"] = v130_context.get("candidate_action")
        source_paths.add("v130_context_row.candidate_observation_context.candidate_action")
    return {
        "features": features,
        "source_paths": tuple(sorted(source_paths)),
    }


def _feature_allowlist(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    feature_paths = _actual_feature_paths(dataset)
    source_paths = _actual_source_paths(dataset)
    denied_features = [
        path for path in feature_paths if not _allowed_feature_path(path)
    ]
    denied_sources = [
        path for path in source_paths if not _allowed_extractor_source_path(path)
    ]
    return {
        "passed": not denied_features and not denied_sources,
        "actual_feature_paths": feature_paths,
        "actual_extractor_source_paths": source_paths,
        "denied_feature_paths": denied_features,
        "denied_source_paths": denied_sources,
        "candidate_action_index_used": "candidate_action_index" in feature_paths,
        "audit_source": "actual_v131_refreshed_feature_rows",
    }


def _forbidden_feature_scan(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    paths = sorted(set(_actual_feature_paths(dataset)) | set(_actual_source_paths(dataset)))
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
        "forbidden_feature_families": sorted(V131_FORBIDDEN_FEATURE_PATH_PARTS),
    }


def _leakage_audit(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    forbidden_scan: Mapping[str, object],
) -> dict[str, object]:
    manifest_leakage = _trainable_leakage(manifest_rows)
    candidate_leakage = trainable_public_input_leakage(candidate_rows)
    leakage_count = (
        _int(manifest_leakage.get("split_key_leak_count"))
        + _int(manifest_leakage.get("forbidden_metadata_key_count"))
        + _int(candidate_leakage.get("leak_count"))
        + _int(forbidden_scan.get("forbidden_feature_path_count"))
    )
    return {
        "passed": leakage_count == 0,
        "leakage_count": leakage_count,
        "manifest_trainable_leakage": manifest_leakage,
        "candidate_trainable_leakage": candidate_leakage,
        "forbidden_feature_scan": dict(forbidden_scan),
    }


def _within_branch_variance(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in dataset:
        grouped[str(row.get("branch_id"))].append(row)
    path_branch_counts: Counter[str] = Counter()
    path_examples: defaultdict[str, list[str]] = defaultdict(list)
    for branch_id, rows in sorted(grouped.items()):
        path_values: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            for path, value in _mapping(row.get("candidate_public_features")).items():
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
            "field_family": _feature_family(path),
        }
        for path, count in sorted(path_branch_counts.items())
    }
    return {
        "branch_count": len(grouped),
        "varying_paths": varying,
        "varying_path_count": len(varying),
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
    }


def _variance_family(varying: Mapping[str, object], family: str) -> dict[str, object]:
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


def _feature_surface_summary(
    dataset: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "candidate_feature_row_count": len(rows),
        "feature_path_count": len(_actual_feature_paths(dataset)),
        "feature_paths": _actual_feature_paths(dataset),
        "example_candidate_public_features": [
            _mapping(row.get("candidate_public_features")) for row in rows[:3]
        ],
        "feature_rows_schema_version": (
            MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION
        ),
        "feature_rows_default_output_path": str(DEFAULT_FEATURE_ROWS_OUTPUT_PATH),
        "feature_rows_runtime_loadable": False,
    }


def _run_probes(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    train = [row for row in dataset if row.get("split") == "train"]
    action_prior_stats = _positive_rate_table(train, lambda row: row["action_key"])
    refreshed_stats = _positive_rate_table(
        train,
        lambda row: row["refreshed_feature_key"],
    )
    probe_rows = {
        ACTION_ORDER_BASELINE: _predict_by_branch(dataset, _score_action_order),
        ACTION_ONLY_BASELINE: _predict_by_branch(
            dataset,
            lambda row: _score_lookup(row, action_prior_stats, row["action_key"]),
        ),
        PRIMARY_PROBE: _predict_by_branch(
            dataset,
            lambda row: _score_lookup(
                row,
                refreshed_stats,
                row["refreshed_feature_key"],
                fallback=action_prior_stats.get(row["action_key"], 0.0),
            ),
        ),
    }
    return {name: _probe_report(name, rows) for name, rows in probe_rows.items()}


def _empty_probe_reports() -> dict[str, object]:
    return {
        name: _probe_report(name, ())
        for name in (ACTION_ORDER_BASELINE, ACTION_ONLY_BASELINE, PRIMARY_PROBE)
    }


def _probe_report(
    name: str,
    branch_predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "name": name,
        "prediction_count": len(branch_predictions),
        "overall_metrics": _metrics(branch_predictions),
        "per_split_metrics": {
            split: _metrics(
                [row for row in branch_predictions if row.get("split") == split]
            )
            for split in ("train", "validation", "test")
        },
        "seed29_evaluation": _seed29_evaluation(branch_predictions),
        "fixture_open_evaluation": _fixture_open_evaluation(branch_predictions),
        "action_distribution": _action_distribution(branch_predictions),
        "unsupported_action_audit": _unsupported_action_audit(branch_predictions),
        "material_gain_recall": _material_gain_recall(branch_predictions),
    }


def _predict_by_branch(
    dataset: Sequence[Mapping[str, object]],
    score_fn,
) -> tuple[dict[str, object], ...]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in dataset:
        grouped[str(row.get("branch_id"))].append(row)
    predictions: list[dict[str, object]] = []
    for branch_id, rows in sorted(grouped.items()):
        legal = [
            row
            for row in rows
            if _mapping(row.get("candidate_public_features")).get(
                "candidate_action_observation_legal"
            )
            is True
        ]
        selected = sorted(
            legal or rows,
            key=lambda row: (
                -score_fn(row),
                str(row.get("candidate_action")),
                stable_payload_digest(_mapping(row.get("candidate_public_features"))),
            ),
        )[0]
        positive = next((row for row in rows if row.get("is_positive") is True), rows[0])
        selected_features = _mapping(selected.get("candidate_public_features"))
        predictions.append(
            {
                "branch_id": branch_id,
                "split": positive.get("split"),
                "seed": positive.get("seed"),
                "fixture_group": positive.get("fixture_group"),
                "repaired_action": positive.get("repaired_action"),
                "predicted_action": selected.get("candidate_action"),
                "prediction_correct": selected.get("is_positive") is True,
                "unsupported_action": selected_features.get(
                    "candidate_action_observation_legal"
                )
                is not True,
                "repaired_label_material_gain_positive": (
                    positive.get("material_gain_label") is True
                ),
                "repaired_label_material_gain_exact_match_recalled": (
                    selected.get("is_positive") is True
                    and positive.get("material_gain_label") is True
                ),
                "candidate_set_has_material_gain_candidate": any(
                    row.get("material_gain_label") is True for row in rows
                ),
                "selected_material_gain_candidate": (
                    selected.get("material_gain_label") is True
                ),
            }
        )
    return tuple(predictions)


def _score_action_order(row: Mapping[str, object]) -> float:
    action_mask = _mapping(row.get("public_action_mask"))
    actions = sorted(str(action) for action, legal in action_mask.items() if legal is True)
    try:
        return 1.0 - (
            actions.index(str(row.get("candidate_action"))) / max(1, len(actions))
        )
    except ValueError:
        return 0.0


def _probe_definitions() -> dict[str, object]:
    return {
        ACTION_ORDER_BASELINE: (
            "Select observation-legal candidates by public candidate_action name order."
        ),
        ACTION_ONLY_BASELINE: (
            "Report-only train-split positive-rate table keyed only by candidate_action."
        ),
        PRIMARY_PROBE: (
            "Report-only train-split positive-rate table keyed by v129 action/"
            "action-semantics plus v130 observation-time target/resource/"
            "neighborhood context, with action-prior fallback."
        ),
    }


def _probe_comparison(probes: Mapping[str, object]) -> dict[str, object]:
    primary = _mapping(_mapping(probes.get(PRIMARY_PROBE)).get("overall_metrics"))
    action_only = _mapping(_mapping(probes.get(ACTION_ONLY_BASELINE)).get("overall_metrics"))
    action_order = _mapping(_mapping(probes.get(ACTION_ORDER_BASELINE)).get("overall_metrics"))
    heldout_primary = _heldout_accuracy(probes, PRIMARY_PROBE)
    heldout_action_only = _heldout_accuracy(probes, ACTION_ONLY_BASELINE)
    heldout_action_order = _heldout_accuracy(probes, ACTION_ORDER_BASELINE)
    return {
        "primary_probe": PRIMARY_PROBE,
        "refreshed_public_context_probe": probes.get(PRIMARY_PROBE, {}),
        "action_only_baseline": probes.get(ACTION_ONLY_BASELINE, {}),
        "action_order_baseline": probes.get(ACTION_ORDER_BASELINE, {}),
        "overall_accuracy_delta_vs_action_only": _number(primary.get("accuracy"), 0.0)
        - _number(action_only.get("accuracy"), 0.0),
        "overall_accuracy_delta_vs_action_order": _number(primary.get("accuracy"), 0.0)
        - _number(action_order.get("accuracy"), 0.0),
        "heldout_accuracy": heldout_primary,
        "heldout_action_only_accuracy": heldout_action_only,
        "heldout_action_order_accuracy": heldout_action_order,
        "heldout_delta_vs_action_only": heldout_primary - heldout_action_only,
        "heldout_delta_vs_action_order": heldout_primary - heldout_action_order,
        "heldout_signal_beats_action_only_baseline": heldout_primary > heldout_action_only,
        "heldout_signal_beats_action_order_baseline": heldout_primary > heldout_action_order,
    }


def _heldout_evaluation(probes: Mapping[str, object]) -> dict[str, object]:
    return {
        "probe": PRIMARY_PROBE,
        "heldout_accuracy": _heldout_accuracy(probes, PRIMARY_PROBE),
        "action_only_baseline_accuracy": _heldout_accuracy(probes, ACTION_ONLY_BASELINE),
        "action_order_baseline_accuracy": _heldout_accuracy(probes, ACTION_ORDER_BASELINE),
    }


def _heldout_material_exact_match_recall(
    primary_probe: Mapping[str, object],
) -> dict[str, object]:
    per_split = _mapping(primary_probe.get("per_split_metrics"))
    split_summaries = {
        split: _material_recall_summary(_mapping(per_split.get(split)))
        for split in ("train", "validation", "test")
    }
    heldout_positive = sum(
        _int(split_summaries[split].get("material_positive_count"))
        for split in ("validation", "test")
    )
    heldout_recalled = sum(
        _int(split_summaries[split].get("material_exact_match_recalled_count"))
        for split in ("validation", "test")
    )
    return {
        "metric_name": "heldout_repaired_label_material_gain_exact_match_recall",
        "gate_enforced": False,
        "reason": (
            "v131 readiness still gates on the pre-existing overall material "
            "exact-match recall floor; this section exposes heldout recall so "
            "train-heavy material performance cannot hide validation/test behavior."
        ),
        "heldout": {
            "material_positive_count": heldout_positive,
            "material_exact_match_recalled_count": heldout_recalled,
            "material_exact_match_recall": _ratio(heldout_recalled, heldout_positive),
        },
        "per_split": split_summaries,
    }


def _material_recall_summary(metrics: Mapping[str, object]) -> dict[str, object]:
    positive = _int(metrics.get("repaired_label_material_gain_positive_count"))
    recalled = _int(
        metrics.get("repaired_label_material_gain_exact_match_recalled_count")
    )
    return {
        "row_count": _int(metrics.get("row_count")),
        "material_positive_count": positive,
        "material_exact_match_recalled_count": recalled,
        "material_exact_match_recall": _ratio(recalled, positive),
    }


def _metric_gate(
    *,
    source_integrity: Mapping[str, object],
    allowlist: Mapping[str, object],
    forbidden: Mapping[str, object],
    leakage: Mapping[str, object],
    variance: Mapping[str, object],
    probes: Mapping[str, object],
    probe_comparison: Mapping[str, object],
    primary: Mapping[str, object],
) -> dict[str, object]:
    del probes
    failures: list[str] = []
    target_variance = _mapping(variance.get("target_resource_neighborhood_variance"))
    target_varying = _int(target_variance.get("varying_path_count"))
    action_distribution = _mapping(primary.get("action_distribution"))
    unsupported = _mapping(primary.get("unsupported_action_audit"))
    material = _mapping(primary.get("material_gain_recall"))
    if source_integrity.get("passed") is not True:
        failures.append("source_integrity_failed")
    if allowlist.get("passed") is not True:
        failures.append("feature_allowlist_failed")
    if forbidden.get("passed") is not True:
        failures.append("forbidden_feature_scan_failed")
    if leakage.get("passed") is not True:
        failures.append("leakage_failed")
    if target_varying <= 0:
        failures.append("target_resource_neighborhood_context_does_not_vary")
    if probe_comparison.get("heldout_signal_beats_action_only_baseline") is not True:
        failures.append("heldout_signal_not_above_action_only_baseline")
    if probe_comparison.get("heldout_signal_beats_action_order_baseline") is not True:
        failures.append("heldout_signal_not_above_action_order_baseline")
    if _mapping(primary.get("seed29_evaluation")).get("passed") is not True:
        failures.append("seed29_failed")
    if _mapping(primary.get("fixture_open_evaluation")).get("passed") is not True:
        failures.append("fixture_open_failed")
    if action_distribution.get("passed") is not True:
        failures.append("action_collapse_detected")
    if unsupported.get("passed") is not True:
        failures.append("unsupported_selection_detected")
    if material.get("passed") is not True:
        failures.append("material_exact_match_recall_below_floor")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "primary_probe": PRIMARY_PROBE,
        "source_integrity_passed": source_integrity.get("passed"),
        "feature_allowlist_passed": allowlist.get("passed"),
        "forbidden_feature_scan_passed": forbidden.get("passed"),
        "leakage_passed": leakage.get("passed"),
        "target_resource_neighborhood_varying_path_count": target_varying,
        "heldout_signal_beats_action_only_baseline": probe_comparison.get(
            "heldout_signal_beats_action_only_baseline"
        ),
        "heldout_signal_beats_action_order_baseline": probe_comparison.get(
            "heldout_signal_beats_action_order_baseline"
        ),
        "seed29_passed": _mapping(primary.get("seed29_evaluation")).get("passed"),
        "fixture_open_passed": _mapping(primary.get("fixture_open_evaluation")).get(
            "passed"
        ),
        "dominant_predicted_action_share_passed": action_distribution.get("passed"),
        "unsupported_action_rate_passed": unsupported.get("passed"),
        "material_exact_match_recall_passed": material.get("passed"),
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    allowlist: Mapping[str, object],
    forbidden: Mapping[str, object],
    leakage: Mapping[str, object],
    metric_gate: Mapping[str, object],
) -> dict[str, object]:
    if (
        source_integrity.get("passed") is not True
        or allowlist.get("passed") is not True
        or forbidden.get("passed") is not True
        or leakage.get("passed") is not True
    ):
        primary = "refreshed_candidate_public_feature_surface_source_integrity_failed"
    elif metric_gate.get("passed") is True:
        primary = "refreshed_candidate_public_feature_surface_ready_for_ranker_review"
    else:
        primary = "refreshed_candidate_public_feature_surface_blocked_by_signal"
    return {
        "primary": primary,
        "labels": [primary],
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    ready = (
        classification.get("primary")
        == "refreshed_candidate_public_feature_surface_ready_for_ranker_review"
    )
    return {
        "next_step": (
            "review_v131_refreshed_surface_before_any_downstream_design"
            if ready
            else "stop_before_shadow_scorer_and_review_signal_blockers"
        ),
        "refreshed_candidate_public_feature_surface_ready_for_ranker_review": ready,
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


def _actual_feature_paths(dataset: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        {
            str(path)
            for row in dataset
            for path in _mapping(row.get("candidate_public_features")).keys()
        }
    )


def _actual_source_paths(dataset: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        {
            str(path)
            for row in dataset
            for path in _list_like(row.get("extractor_source_paths"))
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


def _allowed_extractor_source_path(path: str) -> bool:
    return path.startswith(
        (
            "v129_feature_row.candidate_public_features.candidate_action",
            "v130_context_row.candidate_observation_context.candidate_target_context.",
            "v130_context_row.candidate_observation_context.candidate_resource_context.",
            "v130_context_row.candidate_observation_context.candidate_neighborhood_context.",
            "v130_context_row.candidate_observation_context.candidate_action",
        )
    )


def _forbidden_matches(path: str) -> list[str]:
    lowered = path.lower()
    tokens = set(_path_tokens(lowered))
    matches = sorted(V131_FORBIDDEN_FEATURE_PATH_PARTS & tokens)
    for substring in V131_FORBIDDEN_FEATURE_PATH_SUBSTRINGS:
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


def _feature_family(path: str) -> str:
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
    return "other_public_context"


def _feature_key(features: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(key), json.dumps(value, sort_keys=True))
        for key, value in sorted(features.items(), key=lambda item: str(item[0]))
    )


def _branch_id(row: Mapping[str, object]) -> str | None:
    value = row.get("branch_id")
    return value if isinstance(value, str) and value else None


def _fixture_group(manifest: Mapping[str, object]) -> str:
    meta = _mapping(manifest.get("non_trainable_audit_metadata"))
    source = str(meta.get("source") or meta.get("source_kind") or meta.get("source_path") or "")
    if "active-coverage" in source or source == "open_mind_v3":
        return "open_mind_v3"
    if "fixture_carrion_only" in source or "carrion_only" in source:
        return "fixture_carrion_only"
    return source or "unknown_source"
