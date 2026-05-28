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
    ACTION_ONLY_BASELINE,
    ACTION_ORDER_BASELINE,
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
    DEFAULT_V115_ARCHIVE_ROWS_PATH,
    _candidate_groups,
    _resolve_jsonl_rows,
)
from evolution_sim.mind.first_recovery_public_rollout_history_context_audit import (
    DEFAULT_HISTORY_ROWS_OUTPUT_PATH as DEFAULT_V133_HISTORY_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V133_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION,
    MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_ROW_SCHEMA_VERSION,
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
    _feature_key,
    _fixture_group,
)
from evolution_sim.mind.first_recovery_shadow_ranker import (
    DOMINANT_SELECTED_ACTION_SHARE_MAX,
    MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_PROBE_SCHEMA_VERSION = (
    "mind_v3_first_recovery_history_refreshed_surface_probe_v1"
)
MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_ROW_SCHEMA_VERSION = (
    "mind_v3_first_recovery_history_refreshed_surface_row_v1"
)
MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_PROBE_POLICY = (
    "diagnostics_only_first_recovery_v134_history_refreshed_surface_probe_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v134-first-recovery-history-refreshed-surface-probe.json"
)
DEFAULT_HISTORY_REFRESHED_ROWS_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v134-first-recovery-history-refreshed-surface-rows.jsonl"
)

PRIMARY_PROBE = "history_refreshed_public_context_probe"
REQUIRED_V133_RECOMMENDATION = (
    "public_rollout_history_context_ready_for_refreshed_surface"
)

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "history_refreshed_surface_source_integrity_failed",
    "history_refreshed_surface_blocked_by_signal",
    "history_refreshed_surface_ready_for_shadow_proposal",
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
class FirstRecoveryHistoryRefreshedSurfaceProbeBuild:
    report: dict[str, object]
    feature_rows: tuple[dict[str, object], ...]


def build_first_recovery_history_refreshed_surface_probe(
    *,
    v131_report: Mapping[str, object] | None = None,
    v131_report_path: str | Path | None = DEFAULT_V131_REPORT_PATH,
    v131_feature_rows: Sequence[Mapping[str, object]] | None = None,
    v131_feature_rows_path: str | Path | None = DEFAULT_V131_FEATURE_ROWS_PATH,
    v133_report: Mapping[str, object] | None = None,
    v133_report_path: str | Path | None = DEFAULT_V133_REPORT_PATH,
    v133_history_rows: Sequence[Mapping[str, object]] | None = None,
    v133_history_rows_path: str | Path | None = DEFAULT_V133_HISTORY_ROWS_PATH,
    v124_manifest_rows: Sequence[Mapping[str, object]] | None = None,
    v124_manifest_path: str | Path | None = DEFAULT_V124_MANIFEST_PATH,
    v115_archive_rows: Sequence[Mapping[str, object]] | None = None,
    v115_archive_rows_path: str | Path | None = DEFAULT_V115_ARCHIVE_ROWS_PATH,
    v123_archive_rows: Sequence[Mapping[str, object]] | None = None,
    v123_archive_rows_path: str | Path | None = DEFAULT_V123_ARCHIVE_ROWS_PATH,
) -> FirstRecoveryHistoryRefreshedSurfaceProbeBuild:
    v131_payload, v131_evidence = _resolve_json_report(
        v131_report,
        v131_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION
        ),
    )
    v131_rows, v131_rows_evidence = _resolve_jsonl_rows(
        v131_feature_rows,
        v131_feature_rows_path,
        row_kind="v131_feature_rows",
    )
    v133_payload, v133_evidence = _resolve_json_report(
        v133_report,
        v133_report_path,
        expected_schema=(
            MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION
        ),
    )
    v133_rows, v133_rows_evidence = _resolve_jsonl_rows(
        v133_history_rows,
        v133_history_rows_path,
        row_kind="v133_history_rows",
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
    alignment = _row_alignment(
        v131_feature_rows=v131_rows,
        v133_history_rows=v133_rows,
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
    )
    source_reports = {
        "v131_report": v131_evidence,
        "v131_feature_rows": v131_rows_evidence,
        "v133_report": v133_evidence,
        "v133_history_rows": v133_rows_evidence,
        "v124_manifest": manifest_evidence,
        "v115_archive_rows": v115_evidence,
        "v123_archive_rows": v123_evidence,
    }
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v131_report=v131_payload,
        v131_feature_rows=v131_rows,
        v133_report=v133_payload,
        v133_history_rows=v133_rows,
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
        alignment=alignment,
    )
    dataset, feature_rows = _history_refreshed_dataset(
        manifest_rows=manifest_rows,
        candidate_groups=_candidate_groups(candidate_rows),
        alignment=alignment,
    )
    allowlist = _feature_allowlist(dataset)
    forbidden = _forbidden_feature_scan(dataset)
    leakage = _leakage_audit(forbidden)
    variance = _within_branch_variance(dataset)
    probes = _run_probes(dataset)
    primary = _mapping(probes.get(PRIMARY_PROBE))
    probe_comparison = _probe_comparison(probes)
    metric_gate = _metric_gate(
        source_integrity=source_integrity,
        allowlist=allowlist,
        leakage=leakage,
        probe_comparison=probe_comparison,
        primary=primary,
    )
    classification = _classification(
        source_integrity=source_integrity,
        allowlist=allowlist,
        leakage=leakage,
        metric_gate=metric_gate,
    )
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_PROBE_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_PROBE_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "feature_surface_summary": _feature_surface_summary(dataset, feature_rows),
        "feature_allowlist": allowlist,
        "forbidden_feature_scan": forbidden,
        "leakage_audit": leakage,
        "within_branch_variance": variance,
        "probe_definitions": _probe_definitions(),
        "probe_reports": probes,
        "probe_comparison": probe_comparison,
        "heldout_evaluation": _heldout_evaluation(probes),
        "seed29_evaluation": primary.get("seed29_evaluation", {}),
        "fixture_open_evaluation": primary.get("fixture_open_evaluation", {}),
        "action_distribution": primary.get("action_distribution", {}),
        "unsupported_action_audit": primary.get("unsupported_action_audit", {}),
        "material_exact_match_recall": primary.get("material_gain_recall", {}),
        "metric_gate": metric_gate,
        "classification": classification,
        "recommendation": _recommendation(classification),
        "authorization_block": _authorization_block(),
        "non_promoted": True,
    }
    return FirstRecoveryHistoryRefreshedSurfaceProbeBuild(
        report=report,
        feature_rows=feature_rows,
    )


def write_first_recovery_history_refreshed_surface_probe_report(
    build: FirstRecoveryHistoryRefreshedSurfaceProbeBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_first_recovery_history_refreshed_surface_rows(
    build: FirstRecoveryHistoryRefreshedSurfaceProbeBuild,
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
        "report_only_history_refreshed_surface_probe": True,
        "training_executed": False,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "runtime_policy_effect": "none",
        "runtime_loadable_artifact_created": False,
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
    v133_report: Mapping[str, object] | None,
    v133_history_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    alignment: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")
    v131_source = _mapping(_mapping(v131_report or {}).get("source_integrity"))
    v131_summary = _mapping(_mapping(v131_report or {}).get("feature_surface_summary"))
    v133_source = _mapping(_mapping(v133_report or {}).get("source_integrity"))
    v133_recommendation = _mapping(_mapping(v133_report or {}).get("recommendation"))
    v133_leakage = _mapping(_mapping(v133_report or {}).get("leakage_audit"))
    v133_rows_summary = _mapping(_mapping(v133_report or {}).get("history_rows_summary"))
    if v131_source.get("passed") is not True:
        failures.append("v131_source_integrity_not_passed")
    if v133_source.get("passed") is not True:
        failures.append("v133_source_integrity_not_passed")
    if v133_recommendation.get("next_step") != REQUIRED_V133_RECOMMENDATION:
        failures.append("v133_not_ready_for_history_refreshed_surface")
    if _int(v133_leakage.get("leakage_count")) != 0:
        failures.append("v133_leakage_nonzero")
    if _int(v131_summary.get("candidate_feature_row_count")) != len(v131_feature_rows):
        failures.append("v131_feature_row_count_mismatch")
    if _int(v133_rows_summary.get("row_count")) != len(v133_history_rows):
        failures.append("v133_history_row_count_mismatch")
    if alignment.get("passed") is not True:
        failures.append("history_refreshed_row_alignment_failed")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v131_source_integrity_passed": v131_source.get("passed"),
        "v133_source_integrity_passed": v133_source.get("passed"),
        "v133_recommendation": v133_recommendation.get("next_step"),
        "required_v133_recommendation": REQUIRED_V133_RECOMMENDATION,
        "v133_leakage_count": _int(v133_leakage.get("leakage_count")),
        "v131_feature_row_count": len(v131_feature_rows),
        "v133_history_row_count": len(v133_history_rows),
        "manifest_row_count": len(manifest_rows),
        "candidate_row_count": len(candidate_rows),
        "v131_feature_rows_digest": stable_payload_digest(list(v131_feature_rows)),
        "v133_history_rows_digest": stable_payload_digest(list(v133_history_rows)),
        "row_alignment": _alignment_summary(alignment),
    }


def _row_alignment(
    *,
    v131_feature_rows: Sequence[Mapping[str, object]],
    v133_history_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    candidate_groups = _candidate_groups(candidate_rows)
    branch_ordinals = _branch_ordinals(manifest_rows)
    candidate_ordinals: set[tuple[int, int]] = set()
    for branch_id, branch_ordinal in branch_ordinals.items():
        for candidate_ordinal, _candidate in enumerate(candidate_groups.get(branch_id, ())):
            candidate_ordinals.add((branch_ordinal, candidate_ordinal))
    v131_by_ordinal: dict[tuple[int, int], Mapping[str, object]] = {}
    malformed_v131 = 0
    schema_mismatch_v131 = 0
    for row in v131_feature_rows:
        if row.get("schema_version") != (
            MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION
        ):
            schema_mismatch_v131 += 1
        metadata = _mapping(row.get("non_feature_metadata"))
        ordinal = (
            metadata.get("candidate_set_ordinal"),
            metadata.get("candidate_ordinal_within_set"),
        )
        if not isinstance(ordinal[0], int) or not isinstance(ordinal[1], int):
            malformed_v131 += 1
            continue
        v131_by_ordinal[(ordinal[0], ordinal[1])] = row
    v133_by_ordinal: dict[int, Mapping[str, object]] = {}
    malformed_v133 = 0
    schema_mismatch_v133 = 0
    unavailable_v133 = 0
    for row in v133_history_rows:
        if row.get("schema_version") != (
            MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_ROW_SCHEMA_VERSION
        ):
            schema_mismatch_v133 += 1
        ordinal = row.get("candidate_set_ordinal")
        if not isinstance(ordinal, int):
            malformed_v133 += 1
            continue
        if row.get("public_history_available") is not True:
            unavailable_v133 += 1
        v133_by_ordinal[ordinal] = row
    required_candidate_ordinals = {
        ordinal
        for ordinal in candidate_ordinals
        if ordinal[0] in v133_by_ordinal
    }
    missing_v131 = sorted(required_candidate_ordinals - set(v131_by_ordinal))
    extra_v133 = sorted(set(v133_by_ordinal) - set(branch_ordinals.values()))
    if schema_mismatch_v131:
        failures.append("v131_feature_row_schema_mismatch")
    if malformed_v131:
        failures.append("v131_feature_row_ordinal_malformed")
    if schema_mismatch_v133:
        failures.append("v133_history_row_schema_mismatch")
    if malformed_v133:
        failures.append("v133_history_row_ordinal_malformed")
    if unavailable_v133:
        failures.append("v133_history_rows_unavailable")
    if missing_v131:
        failures.append("v131_rows_missing_for_v133_history_branches")
    if extra_v133:
        failures.append("v133_history_rows_missing_from_manifest")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v131_rows_by_ordinal": v131_by_ordinal,
        "v133_history_rows_by_candidate_set_ordinal": v133_by_ordinal,
        "v133_candidate_set_ordinals": sorted(v133_by_ordinal),
        "v133_branch_count": len(v133_by_ordinal),
        "candidate_row_count_in_v133_scope": len(required_candidate_ordinals),
        "matched_v131_candidate_row_count": len(
            required_candidate_ordinals.intersection(v131_by_ordinal)
        ),
        "missing_v131_ordinals": [list(item) for item in missing_v131[:12]],
        "extra_v133_ordinals": extra_v133[:12],
    }


def _alignment_summary(alignment: Mapping[str, object]) -> dict[str, object]:
    return {
        "passed": alignment.get("passed"),
        "failures": _list_like(alignment.get("failures")),
        "v133_branch_count": alignment.get("v133_branch_count"),
        "candidate_row_count_in_v133_scope": alignment.get(
            "candidate_row_count_in_v133_scope"
        ),
        "matched_v131_candidate_row_count": alignment.get(
            "matched_v131_candidate_row_count"
        ),
        "missing_v131_ordinals": _list_like(alignment.get("missing_v131_ordinals")),
        "extra_v133_ordinals": _list_like(alignment.get("extra_v133_ordinals")),
    }


def _history_refreshed_dataset(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
    alignment: Mapping[str, object],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    branch_ordinals = _branch_ordinals(manifest_rows)
    v131_by_ordinal = _mapping(alignment.get("v131_rows_by_ordinal"))
    v133_by_ordinal = _mapping(
        alignment.get("v133_history_rows_by_candidate_set_ordinal")
    )
    dataset: list[dict[str, object]] = []
    rows_out: list[dict[str, object]] = []
    for manifest in sorted(manifest_rows, key=lambda row: str(row.get("branch_id"))):
        branch_id = _branch_id(manifest)
        if branch_id is None:
            continue
        branch_ordinal = branch_ordinals.get(branch_id)
        if branch_ordinal not in v133_by_ordinal:
            continue
        history_row = _mapping(v133_by_ordinal.get(branch_ordinal))
        history_features = _mapping(history_row.get("history_features"))
        repaired_action = str(manifest.get("repaired_action"))
        repaired_archive_id = manifest.get("repaired_archive_row_id")
        fixture_group = _fixture_group(manifest)
        seed = _mapping(manifest.get("non_trainable_audit_metadata")).get("seed")
        for candidate_ordinal, candidate in enumerate(candidate_groups.get(branch_id, ())):
            ordinal = (branch_ordinal, candidate_ordinal)
            v131_row = _mapping(v131_by_ordinal.get(ordinal))
            metadata = _mapping(v131_row.get("non_feature_metadata"))
            features = _combined_features(v131_row, history_features)
            source_paths = _combined_source_paths(v131_row, history_features)
            public_action_mask = dict(
                _mapping(_mapping(candidate.get("trainable_public_input")).get("action_mask"))
            )
            split = metadata.get("split")
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
                "extractor_source_paths": source_paths,
                "public_action_mask": public_action_mask,
                "action_key": (features.get("candidate_action"),),
                "history_refreshed_feature_key": _feature_key(features),
            }
            dataset.append(dataset_row)
            rows_out.append(
                {
                    "schema_version": (
                        MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_ROW_SCHEMA_VERSION
                    ),
                    "candidate_public_features": features,
                    "extractor_source_paths": source_paths,
                    "non_feature_metadata": {
                        "candidate_set_ordinal": branch_ordinal,
                        "candidate_ordinal_within_set": candidate_ordinal,
                        "split": split,
                        "history_context_available": True,
                        "diagnostic_scope": "v133_history_rows_only",
                    },
                    "diagnostics_only": True,
                    "training_authorized": False,
                    "runtime_policy_authorized": False,
                    "claim_causality": False,
                }
            )
    return tuple(dataset), tuple(rows_out)


def _combined_features(
    v131_row: Mapping[str, object],
    history_features: Mapping[str, object],
) -> dict[str, object]:
    features = dict(_mapping(v131_row.get("candidate_public_features")))
    for path, value in sorted(history_features.items(), key=lambda item: str(item[0])):
        features[str(path)] = value
    return features


def _combined_source_paths(
    v131_row: Mapping[str, object],
    history_features: Mapping[str, object],
) -> tuple[str, ...]:
    paths = {str(path) for path in _list_like(v131_row.get("extractor_source_paths"))}
    paths.update(
        f"v133_history_row.history_features.{path}"
        for path in sorted(str(path) for path in history_features)
    )
    return tuple(sorted(paths))


def _feature_allowlist(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    feature_paths = _actual_feature_paths(dataset)
    source_paths = _actual_source_paths(dataset)
    denied_features = [
        path for path in feature_paths if not _allowed_feature_path(path)
    ]
    denied_sources = [
        path for path in source_paths if not _allowed_source_path(path)
    ]
    return {
        "passed": not denied_features and not denied_sources,
        "actual_feature_paths": feature_paths,
        "actual_extractor_source_paths": source_paths,
        "denied_feature_paths": denied_features,
        "denied_source_paths": denied_sources,
        "trainable_public_feature_payload_key": "candidate_public_features",
        "non_feature_metadata_separated": True,
    }


def _allowed_feature_path(path: str) -> bool:
    return path == "candidate_action" or path.startswith(
        (
            "candidate_action_",
            "candidate_target_context.",
            "candidate_resource_context.",
            "candidate_neighborhood_context.",
            "rollout_context.",
            "recovery_context.",
            "legal_action_mask_history.",
            "public_history.",
            "current_public_action_mask.",
        )
    )


def _allowed_source_path(path: str) -> bool:
    return path.startswith(
        (
            "v129_feature_row.",
            "v130_context_row.",
            "v133_history_row.history_features.",
        )
    )


def _forbidden_feature_scan(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    paths = sorted(_actual_feature_paths(dataset))
    forbidden = [
        {"path": path, "matched_forbidden_parts": _forbidden_matches(path)}
        for path in paths
        if _forbidden_matches(path)
    ]
    return {
        "passed": not forbidden,
        "forbidden_feature_path_count": len(forbidden),
        "forbidden_feature_paths": forbidden[:24],
        "scanned_feature_path_count": len(paths),
        "forbidden_feature_families": sorted(FORBIDDEN_FEATURE_PATH_PARTS),
    }


def _leakage_audit(forbidden_scan: Mapping[str, object]) -> dict[str, object]:
    leakage_count = _int(forbidden_scan.get("forbidden_feature_path_count"))
    return {
        "passed": leakage_count == 0,
        "leakage_count": leakage_count,
        "forbidden_feature_scan": dict(forbidden_scan),
        "repaired_label_used_as_trainable_feature": False,
        "source_or_seed_identity_used_as_trainable_feature": False,
        "post_decision_outcome_used_as_trainable_feature": False,
        "claim_causality": False,
    }


def _within_branch_variance(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in dataset:
        grouped[str(row.get("branch_id"))].append(row)
    path_branch_counts: Counter[str] = Counter()
    for rows in grouped.values():
        path_values: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            for path, value in _mapping(row.get("candidate_public_features")).items():
                path_values[str(path)].add(json.dumps(value, sort_keys=True))
        for path, values in path_values.items():
            if len(values) > 1:
                path_branch_counts[path] += 1
    return {
        "branch_count": len(grouped),
        "varying_path_count": len(path_branch_counts),
        "varying_paths": dict(sorted(path_branch_counts.items())),
        "history_feature_varying_path_count": sum(
            1 for path in path_branch_counts if _history_path(path)
        ),
        "candidate_context_varying_path_count": sum(
            1
            for path in path_branch_counts
            if path.startswith(
                (
                    "candidate_target_context.",
                    "candidate_resource_context.",
                    "candidate_neighborhood_context.",
                )
            )
        ),
    }


def _run_probes(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    train = [row for row in dataset if row.get("split") == "train"]
    action_prior_stats = _positive_rate_table(train, lambda row: row["action_key"])
    history_stats = _positive_rate_table(
        train,
        lambda row: row["history_refreshed_feature_key"],
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
                history_stats,
                row["history_refreshed_feature_key"],
                fallback=action_prior_stats.get(row["action_key"], 0.0),
            ),
        ),
    }
    return {name: _probe_report(name, rows) for name, rows in probe_rows.items()}


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
            "Report-only train-split positive-rate table keyed by v131 refreshed "
            "candidate public features plus v133 branch-local public rollout/history "
            "context, with action-prior fallback."
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
        "history_refreshed_public_context_probe": probes.get(PRIMARY_PROBE, {}),
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


def _metric_gate(
    *,
    source_integrity: Mapping[str, object],
    allowlist: Mapping[str, object],
    leakage: Mapping[str, object],
    probe_comparison: Mapping[str, object],
    primary: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    fixture_open = _mapping(primary.get("fixture_open_evaluation"))
    open_group = _mapping(_mapping(fixture_open.get("groups")).get("open_mind_v3"))
    open_dominant = _mapping(open_group.get("dominant_predicted_action"))
    unsupported = _mapping(primary.get("unsupported_action_audit"))
    material = _mapping(primary.get("material_gain_recall"))
    if source_integrity.get("passed") is not True:
        failures.append("source_integrity_failed")
    if allowlist.get("passed") is not True:
        failures.append("feature_allowlist_failed")
    if leakage.get("passed") is not True:
        failures.append("leakage_failed")
    if probe_comparison.get("heldout_signal_beats_action_only_baseline") is not True:
        failures.append("heldout_signal_not_above_action_only_baseline")
    if _number(open_dominant.get("share"), 1.0) > DOMINANT_SELECTED_ACTION_SHARE_MAX:
        failures.append("fixture_open_dominant_action_share_failed")
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
        "leakage_passed": leakage.get("passed"),
        "heldout_signal_beats_action_only_baseline": probe_comparison.get(
            "heldout_signal_beats_action_only_baseline"
        ),
        "heldout_signal_beats_action_order_baseline": probe_comparison.get(
            "heldout_signal_beats_action_order_baseline"
        ),
        "fixture_open_dominant_predicted_action": dict(open_dominant),
        "fixture_open_dominant_action_share_passed": (
            _number(open_dominant.get("share"), 1.0)
            <= DOMINANT_SELECTED_ACTION_SHARE_MAX
        ),
        "unsupported_action_count": unsupported.get("unsupported_action_count"),
        "unsupported_action_rate": unsupported.get("unsupported_action_rate"),
        "material_exact_match_recall": material.get(
            "repaired_label_material_gain_exact_match_recall"
        ),
        "minimum_material_exact_match_recall": MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    allowlist: Mapping[str, object],
    leakage: Mapping[str, object],
    metric_gate: Mapping[str, object],
) -> dict[str, object]:
    if (
        source_integrity.get("passed") is not True
        or allowlist.get("passed") is not True
        or leakage.get("passed") is not True
    ):
        primary = "history_refreshed_surface_source_integrity_failed"
    elif metric_gate.get("passed") is True:
        primary = "history_refreshed_surface_ready_for_shadow_proposal"
    else:
        primary = "history_refreshed_surface_blocked_by_signal"
    return {
        "primary": primary,
        "labels": [primary],
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    ready = (
        classification.get("primary")
        == "history_refreshed_surface_ready_for_shadow_proposal"
    )
    return {
        "next_step": (
            "review_v134_history_refreshed_surface_before_shadow_proposal"
            if ready
            else "stop_before_shadow_scorer_and_review_history_signal_blockers"
        ),
        "history_refreshed_surface_ready_for_shadow_proposal": ready,
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
        "claim_causality": False,
    }


def _feature_surface_summary(
    dataset: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    paths = _actual_feature_paths(dataset)
    return {
        "candidate_feature_row_count": len(rows),
        "candidate_branch_scope": "v133_history_rows_only",
        "candidate_branch_count": len({row.get("branch_id") for row in dataset}),
        "feature_path_count": len(paths),
        "feature_paths": paths,
        "history_feature_path_count": len([path for path in paths if _history_path(path)]),
        "feature_rows_schema_version": (
            MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_ROW_SCHEMA_VERSION
        ),
        "feature_rows_default_output_path": str(DEFAULT_HISTORY_REFRESHED_ROWS_OUTPUT_PATH),
        "feature_rows_runtime_loadable": False,
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


def _history_path(path: str) -> bool:
    return path.startswith(
        (
            "rollout_context.",
            "recovery_context.",
            "legal_action_mask_history.",
            "public_history.",
            "current_public_action_mask.",
        )
    )


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
