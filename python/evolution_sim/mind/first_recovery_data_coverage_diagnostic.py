from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_branch_archive import (
    FORBIDDEN_TRAINABLE_KEYS as ARCHIVE_FORBIDDEN_TRAINABLE_KEYS,
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    trainable_public_input_leakage,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_SCHEMA_VERSION,
    SIGNAL_FAMILY_ORDER,
    _counter_to_dict,
    _int,
    _list,
    _mapping,
    _resolve_archive_rows,
    _resolve_json_report,
    _round,
    _share,
)
from evolution_sim.mind.first_recovery_shadow_ranker import (
    group_archive_rows_by_branch_target,
)
from evolution_sim.mind.first_recovery_state_action_readiness_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_STATE_ACTION_READINESS_AUDIT_PATH,
    DEFAULT_PUBLIC_SIGNAL_AUDIT_PATH,
    MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION = (
    "mind_v3_first_recovery_data_coverage_diagnostic_v1"
)
MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_POLICY = (
    "diagnostics_only_first_recovery_v109_selection_coverage_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v114-first-recovery-data-coverage-diagnostic.json"
)

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "diagnostics_only_no_runtime_promotion",
    "missing_evidence_inconclusive",
    "v109_selection_bias_detected",
    "v109_selection_representative",
    "expanded_archive_required",
    "expanded_archive_not_justified",
    "readiness_rerun_blocked",
    "readiness_rerun_allowed",
    "trainable_signal_leakage_detected",
)

FORBIDDEN_TRAINABLE_SIGNAL_TOKENS = frozenset(
    {
        *ARCHIVE_FORBIDDEN_TRAINABLE_KEYS,
        "branch",
        "fixture",
        "private",
        "private_world",
        "provenance",
        "source",
    }
)

BIAS_MIN_RECONSTRUCTED_KEY_COUNT = 2
BIAS_DISTRIBUTION_DELTA_THRESHOLD = 0.20
BIAS_RATE_MULTIPLIER = 2.0
MAX_EXAMPLES = 16


class FirstRecoveryDataCoverageDiagnosticError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FirstRecoveryDataCoverageDiagnosticBuild:
    report: dict[str, object]


def build_first_recovery_data_coverage_diagnostic(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    archive_rows: Sequence[Mapping[str, object]] | None = None,
    archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    public_signal_audit: Mapping[str, object] | None = None,
    public_signal_audit_path: str | Path | None = DEFAULT_PUBLIC_SIGNAL_AUDIT_PATH,
    state_action_readiness_audit: Mapping[str, object] | None = None,
    state_action_readiness_audit_path: str | Path | None = (
        DEFAULT_STATE_ACTION_READINESS_AUDIT_PATH
    ),
) -> FirstRecoveryDataCoverageDiagnosticBuild:
    contract = _contract()
    archive_payload, archive_evidence = _resolve_json_report(
        archive_report,
        archive_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    rows, rows_evidence = _resolve_archive_rows(archive_rows, archive_rows_path)
    public_payload, public_evidence = _resolve_json_report(
        public_signal_audit,
        public_signal_audit_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_SCHEMA_VERSION,
    )
    readiness_payload, readiness_evidence = _resolve_json_report(
        state_action_readiness_audit,
        state_action_readiness_audit_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_SCHEMA_VERSION,
    )
    groups = group_archive_rows_by_branch_target(rows)
    source_reports = _source_reports(
        archive_report=archive_payload,
        archive_evidence=archive_evidence,
        rows=rows,
        rows_evidence=rows_evidence,
        public_signal_audit=public_payload,
        public_evidence=public_evidence,
        state_action_readiness_audit=readiness_payload,
        readiness_evidence=readiness_evidence,
        group_count=len(groups),
    )
    reconstruction_coverage = _reconstruction_coverage(
        archive_report=archive_payload,
        archive_rows=rows,
        archive_group_count=len(groups),
    )
    selected_vs_reconstructed = _selected_vs_reconstructed_coverage(
        archive_report=archive_payload,
        groups=groups,
        reconstruction_coverage=reconstruction_coverage,
    )
    skipped_target_coverage = _skipped_target_coverage(
        archive_report=archive_payload,
        reconstruction_coverage=reconstruction_coverage,
    )
    alias_family_coverage = _v112_alias_family_coverage(
        public_signal_audit=public_payload,
        reconstruction_coverage=reconstruction_coverage,
    )
    blocker_review = _v113_blocker_review(
        state_action_readiness_audit=readiness_payload,
        selected_vs_reconstructed_coverage=selected_vs_reconstructed,
        v112_alias_family_coverage=alias_family_coverage,
    )
    leakage_guard = _leakage_guard(rows)
    missing_evidence = _missing_evidence(
        source_reports=source_reports,
        reconstruction_coverage=reconstruction_coverage,
        selected_vs_reconstructed_coverage=selected_vs_reconstructed,
    )
    expansion_budget = _expansion_budget(
        reconstruction_coverage=reconstruction_coverage,
        selected_vs_reconstructed_coverage=selected_vs_reconstructed,
        v113_blocker_review=blocker_review,
        missing_evidence=missing_evidence,
        leakage_guard=leakage_guard,
    )
    classification = _classification(
        missing_evidence=missing_evidence,
        selected_vs_reconstructed_coverage=selected_vs_reconstructed,
        v113_blocker_review=blocker_review,
        expansion_budget=expansion_budget,
        leakage_guard=leakage_guard,
    )
    recommendation = _recommendation(
        classification=classification,
        expansion_budget=expansion_budget,
        v113_blocker_review=blocker_review,
    )
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_POLICY,
        "contract": contract,
        "source_reports": source_reports,
        "reconstruction_coverage": reconstruction_coverage,
        "selected_vs_reconstructed_coverage": selected_vs_reconstructed,
        "skipped_target_coverage": skipped_target_coverage,
        "v112_alias_family_coverage": alias_family_coverage,
        "v113_blocker_review": blocker_review,
        "expansion_budget": expansion_budget,
        "leakage_guard": leakage_guard,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryDataCoverageDiagnosticBuild(report=report)


def write_first_recovery_data_coverage_diagnostic_report(
    build: FirstRecoveryDataCoverageDiagnosticBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def first_recovery_data_coverage_trainable_field_leakage(
    field_names: Sequence[str],
) -> tuple[str, ...]:
    leaks: set[str] = set()
    for name in field_names:
        normalized = str(name).lower()
        tokens = set(_field_tokens(normalized))
        if tokens & FORBIDDEN_TRAINABLE_SIGNAL_TOKENS:
            leaks.add(str(name))
            continue
        if "private_world" in normalized or "simulation_world" in normalized:
            leaks.add(str(name))
    return tuple(sorted(leaks))


def _contract() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "observation_field_change": False,
        "expanded_archive_replay_executed": False,
        "readiness_rerun_executed": False,
        "runtime_policy_implemented": False,
        "trained_artifact_emitted": False,
        "private_world_state_input": False,
        "private_world_state_serialized": False,
        "seed_identity_trainable_signal": False,
        "source_identity_trainable_signal": False,
        "fixture_identity_trainable_signal": False,
        "branch_identity_trainable_signal": False,
        "identity_fields_used_for_grouping_only": True,
        "selection_diagnostic_policy": (
            "compare_v109_reconstructed_first_recovery_support_to_selected_archive_support"
        ),
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION
                ),
                "audit_policy": MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_POLICY,
            }
        ),
    }


def _source_reports(
    *,
    archive_report: Mapping[str, object] | None,
    archive_evidence: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    rows_evidence: Mapping[str, object],
    public_signal_audit: Mapping[str, object] | None,
    public_evidence: Mapping[str, object],
    state_action_readiness_audit: Mapping[str, object] | None,
    readiness_evidence: Mapping[str, object],
    group_count: int,
) -> dict[str, object]:
    return {
        "archive_report": archive_evidence,
        "archive_rows": rows_evidence,
        "public_signal_audit": public_evidence,
        "state_action_readiness_audit": readiness_evidence,
        "archive_schema_version": _mapping(archive_report or {}).get("schema_version"),
        "archive_row_count": len(rows),
        "archive_branch_target_group_count": int(group_count),
        "public_signal_schema_version": _mapping(public_signal_audit or {}).get(
            "schema_version"
        ),
        "state_action_readiness_schema_version": _mapping(
            state_action_readiness_audit or {}
        ).get("schema_version"),
        "source_report_policy": "inputs_are_read_only_diagnostics_sources",
    }


def _reconstruction_coverage(
    *,
    archive_report: Mapping[str, object] | None,
    archive_rows: Sequence[Mapping[str, object]],
    archive_group_count: int,
) -> dict[str, object]:
    report = _mapping(archive_report or {})
    reconstruction = _mapping(report.get("row_reconstruction"))
    selection = _mapping(report.get("target_selection"))
    summary = _mapping(report.get("branch_archive_summary"))
    reconstructed_count = _first_positive_int(
        reconstruction.get("reconstructed_first_recovery_row_count"),
        reconstruction.get("expected_constructible_first_recovery_row_count"),
        selection.get("candidate_row_count"),
        summary.get("full_reconstruction_row_count"),
    )
    selected_count = _first_positive_int(
        selection.get("selected_target_count"),
        summary.get("selected_row_count"),
        archive_group_count,
    )
    skipped_count = _int(
        selection.get("skipped_target_count"),
        default=max(0, reconstructed_count - selected_count),
    )
    return {
        "coverage_status": "available" if reconstructed_count > 0 else "missing",
        "reconstructed_first_recovery_row_count": reconstructed_count,
        "expected_constructible_first_recovery_row_count": _int(
            reconstruction.get("expected_constructible_first_recovery_row_count")
        ),
        "expected_v108_reconstructed_first_recovery_row_count": _int(
            reconstruction.get("expected_v108_reconstructed_first_recovery_row_count")
        ),
        "row_count_matches_v107": reconstruction.get("row_count_matches_v107") is True,
        "row_count_matches_v108": reconstruction.get("row_count_matches_v108") is True,
        "alignment_complete": (
            reconstruction.get("row_count_matches_v107") is True
            and reconstruction.get("row_count_matches_v108") is True
        ),
        "selected_target_count": selected_count,
        "skipped_target_count": skipped_count,
        "selection_share": _share(selected_count, reconstructed_count),
        "archive_row_count": len(archive_rows),
        "archive_branch_target_group_count": archive_group_count,
        "archive_groups_match_selected_target_count": (
            archive_group_count == selected_count if archive_group_count else False
        ),
        "selection_policy": selection.get("selection_policy"),
        "selection_reason_counts": _int_counter(selection.get("selection_reason_counts")),
        "reconstructed_by_seed": _int_counter(reconstruction.get("by_seed")),
        "reconstructed_by_source": _int_counter(reconstruction.get("by_source")),
        "reconstructed_by_seed_source": _int_counter(reconstruction.get("by_seed_source")),
        "reconstructed_by_logged_action": _int_counter(
            reconstruction.get("by_logged_action")
        ),
        "selected_by_seed": _int_counter(selection.get("selected_by_seed")),
        "selected_by_source": _int_counter(selection.get("selected_by_source")),
        "selected_by_seed_source": _int_counter(selection.get("selected_by_seed_source")),
        "selected_by_logged_action": _int_counter(selection.get("selected_by_logged_action")),
        "selected_ticks_by_seed_source": {
            str(key): [int(item) for item in _list(value)]
            for key, value in sorted(
                _mapping(selection.get("selected_ticks_by_seed_source")).items(),
                key=lambda item: str(item[0]),
            )
        },
    }


def _selected_vs_reconstructed_coverage(
    *,
    archive_report: Mapping[str, object] | None,
    groups: Sequence[object],
    reconstruction_coverage: Mapping[str, object],
) -> dict[str, object]:
    selected_by_seed = _int_counter(reconstruction_coverage.get("selected_by_seed"))
    selected_by_source = _int_counter(reconstruction_coverage.get("selected_by_source"))
    selected_by_seed_source = _int_counter(
        reconstruction_coverage.get("selected_by_seed_source")
    )
    selected_by_action = _int_counter(
        reconstruction_coverage.get("selected_by_logged_action")
    )
    if not selected_by_seed or not selected_by_source or not selected_by_action:
        derived = _selected_counts_from_archive_groups(groups)
        selected_by_seed = selected_by_seed or derived["seed"]
        selected_by_source = selected_by_source or derived["source"]
        selected_by_seed_source = selected_by_seed_source or derived["seed_source"]
        selected_by_action = selected_by_action or derived["logged_action"]

    dimensions = {
        "source": _dimension_comparison(
            "source",
            _int_counter(reconstruction_coverage.get("reconstructed_by_source")),
            selected_by_source,
            min_reconstructed_key_count=1,
        ),
        "seed": _dimension_comparison(
            "seed",
            _int_counter(reconstruction_coverage.get("reconstructed_by_seed")),
            selected_by_seed,
            min_reconstructed_key_count=BIAS_MIN_RECONSTRUCTED_KEY_COUNT,
        ),
        "seed_source": _dimension_comparison(
            "seed_source",
            _int_counter(reconstruction_coverage.get("reconstructed_by_seed_source")),
            selected_by_seed_source,
            min_reconstructed_key_count=BIAS_MIN_RECONSTRUCTED_KEY_COUNT,
        ),
        "logged_action": _dimension_comparison(
            "logged_action",
            _int_counter(reconstruction_coverage.get("reconstructed_by_logged_action")),
            selected_by_action,
            min_reconstructed_key_count=BIAS_MIN_RECONSTRUCTED_KEY_COUNT,
        ),
    }
    target_selection = _mapping(_mapping(archive_report or {}).get("target_selection"))
    tick_coverage = _selected_tick_coverage(target_selection)
    biased_dimensions = [
        name for name, section in dimensions.items() if section.get("bias_detected")
    ]
    selected_count = _int(reconstruction_coverage.get("selected_target_count"))
    reconstructed_count = _int(
        reconstruction_coverage.get("reconstructed_first_recovery_row_count")
    )
    representative = (
        reconstructed_count > 0
        and selected_count > 0
        and not biased_dimensions
        and all(
            _int(section.get("missing_reconstructed_key_count")) == 0
            for section in dimensions.values()
        )
    )
    return {
        "comparison_policy": (
            "selected_target_distribution_vs_reconstructed_first_recovery_distribution"
        ),
        "reconstructed_target_count": reconstructed_count,
        "selected_target_count": selected_count,
        "selection_share": _share(selected_count, reconstructed_count),
        "dimensions": dimensions,
        "tick_coverage": tick_coverage,
        "bias_detected": bool(biased_dimensions),
        "biased_dimensions": biased_dimensions,
        "representative": representative,
        "largest_distribution_delta": _largest_dimension_delta(dimensions),
        "selection_bias_reason_count": sum(
            _int(section.get("bias_reason_count")) for section in dimensions.values()
        ),
        "insufficient_dimension_support": [
            name
            for name, section in dimensions.items()
            if _int(section.get("selected_key_count")) < _int(
                section.get("reconstructed_key_count")
            )
        ],
    }


def _skipped_target_coverage(
    *,
    archive_report: Mapping[str, object] | None,
    reconstruction_coverage: Mapping[str, object],
) -> dict[str, object]:
    selection = _mapping(_mapping(archive_report or {}).get("target_selection"))
    skipped = _list(selection.get("skipped_examples"))
    skipped_count = _int(reconstruction_coverage.get("skipped_target_count"))
    reconstructed_count = _int(
        reconstruction_coverage.get("reconstructed_first_recovery_row_count")
    )
    by_seed: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_action: Counter[str] = Counter()
    by_tick: Counter[str] = Counter()
    for item in skipped:
        row = _mapping(item)
        if row.get("seed") is not None:
            by_seed[str(row.get("seed"))] += 1
        if row.get("source_kind") is not None:
            by_source[str(row.get("source_kind"))] += 1
        if row.get("requested_action") is not None:
            by_action[str(row.get("requested_action"))] += 1
        if row.get("recovery_tick") is not None:
            by_tick[str(row.get("recovery_tick"))] += 1
    return {
        "skipped_target_count": skipped_count,
        "skipped_target_share": _share(skipped_count, reconstructed_count),
        "skip_reason_counts": _int_counter(selection.get("skip_reason_counts")),
        "skipped_examples_count": len(skipped),
        "skipped_examples_are_complete": len(skipped) == skipped_count,
        "skipped_examples_capped": len(skipped) < skipped_count,
        "reported_skipped_examples_by_seed": _counter_to_dict(by_seed),
        "reported_skipped_examples_by_source": _counter_to_dict(by_source),
        "reported_skipped_examples_by_logged_action": _counter_to_dict(by_action),
        "reported_skipped_examples_by_tick": _counter_to_dict(by_tick),
        "examples": [dict(_mapping(item)) for item in skipped[:MAX_EXAMPLES]],
        "full_skipped_distribution_available": len(skipped) == skipped_count,
    }


def _v112_alias_family_coverage(
    *,
    public_signal_audit: Mapping[str, object] | None,
    reconstruction_coverage: Mapping[str, object],
) -> dict[str, object]:
    audit = _mapping(public_signal_audit or {})
    families = _mapping(audit.get("signal_families"))
    selected_by_seed = _int_counter(reconstruction_coverage.get("selected_by_seed"))
    selected_by_source = _int_counter(reconstruction_coverage.get("selected_by_source"))
    reconstructed_by_seed = _int_counter(
        reconstruction_coverage.get("reconstructed_by_seed")
    )
    reconstructed_by_source = _int_counter(
        reconstruction_coverage.get("reconstructed_by_source")
    )
    selected_count = _int(reconstruction_coverage.get("selected_target_count"))
    reconstructed_count = _int(
        reconstruction_coverage.get("reconstructed_first_recovery_row_count")
    )
    family_reports: list[dict[str, object]] = []
    for family in _family_order(families):
        section = _mapping(families.get(family))
        per_seed = _int_counter(section.get("per_seed_support"))
        per_source = _int_counter(section.get("fixture_open_support"))
        seed_support_total = sum(per_seed.values())
        source_support_total = sum(per_source.values())
        reconstructed_seed_gap = _positive_counter_delta(reconstructed_by_seed, per_seed)
        reconstructed_source_gap = _positive_counter_delta(reconstructed_by_source, per_source)
        selected_seed_delta = _counter_abs_delta(selected_by_seed, per_seed)
        selected_source_delta = _counter_abs_delta(selected_by_source, per_source)
        family_reports.append(
            {
                "family": family,
                "alias_group_count": _int(section.get("alias_group_count")),
                "missing_public_fields": _list(section.get("missing_public_fields")),
                "per_seed_support": per_seed,
                "fixture_open_support": per_source,
                "seed_support_total": seed_support_total,
                "source_support_total": source_support_total,
                "support_total_matches_selected_target_count": (
                    seed_support_total == selected_count
                    or source_support_total == selected_count
                ),
                "support_matches_selected_distribution": (
                    selected_seed_delta == 0 and selected_source_delta == 0
                ),
                "reconstructed_seed_support_gap": reconstructed_seed_gap,
                "reconstructed_source_support_gap": reconstructed_source_gap,
                "reconstructed_support_gap_count": sum(
                    reconstructed_seed_gap.values()
                )
                + sum(reconstructed_source_gap.values()),
                "support_scope": (
                    "selected_archive_only"
                    if (
                        reconstructed_count > selected_count
                        and (
                            seed_support_total == selected_count
                            or source_support_total == selected_count
                        )
                    )
                    else "not_limited_to_selected_count"
                ),
            }
        )
    selected_only_count = sum(
        1 for item in family_reports if item.get("support_scope") == "selected_archive_only"
    )
    return {
        "v112_schema_version": audit.get("schema_version"),
        "v112_classification_primary": _mapping(audit.get("classification")).get("primary"),
        "family_count": len(family_reports),
        "families_with_alias_groups": [
            str(item["family"])
            for item in family_reports
            if _int(item.get("alias_group_count")) > 0
        ],
        "family_reports": family_reports,
        "alias_evidence_uses_selected_archive_only": selected_only_count > 0,
        "selected_archive_only_family_count": selected_only_count,
        "selected_target_count": selected_count,
        "reconstructed_target_count": reconstructed_count,
    }


def _v113_blocker_review(
    *,
    state_action_readiness_audit: Mapping[str, object] | None,
    selected_vs_reconstructed_coverage: Mapping[str, object],
    v112_alias_family_coverage: Mapping[str, object],
) -> dict[str, object]:
    audit = _mapping(state_action_readiness_audit or {})
    classification = _mapping(audit.get("classification"))
    recommendation = _mapping(audit.get("recommendation"))
    blockers = [str(item) for item in _list(recommendation.get("blockers"))]
    labels = [str(item) for item in _list(classification.get("labels"))]
    bias_detected = selected_vs_reconstructed_coverage.get("bias_detected") is True
    alias_selected_only = (
        v112_alias_family_coverage.get("alias_evidence_uses_selected_archive_only") is True
    )
    reviews = [
        _blocker_plausibility(
            blocker,
            bias_detected=bias_detected,
            alias_selected_only=alias_selected_only,
        )
        for blocker in blockers
    ]
    plausible_count = sum(1 for item in reviews if item.get("coverage_plausible"))
    lookup_design_blockers = [
        blocker
        for blocker in blockers
        if blocker
        in {
            "state_action_interaction_does_not_beat_action_only_heldout",
            "alias_resolution_not_stable_under_quantization",
            "shuffled_control_artifact_detected",
        }
    ]
    loaded = bool(audit)
    coverage_confound_plausible = bool(blockers) and plausible_count == len(blockers)
    return {
        "v113_schema_version": audit.get("schema_version"),
        "v113_classification_primary": classification.get("primary"),
        "v113_classification_labels": labels,
        "v113_recommendation_next_step": recommendation.get("next_step"),
        "v113_blockers": blockers,
        "blocker_reviews": reviews,
        "coverage_plausible_blocker_count": plausible_count,
        "blocker_count": len(blockers),
        "coverage_plausibly_confounds_v113_readiness": coverage_confound_plausible,
        "lookup_design_blockers": lookup_design_blockers,
        "coverage_cannot_be_separated_from_lookup_design_failure": (
            loaded and bool(lookup_design_blockers) and not bias_detected
        ),
        "readiness_rerun_on_current_v109_archive_allowed": (
            loaded and not blockers and not bias_detected
        ),
    }


def _expansion_budget(
    *,
    reconstruction_coverage: Mapping[str, object],
    selected_vs_reconstructed_coverage: Mapping[str, object],
    v113_blocker_review: Mapping[str, object],
    missing_evidence: Sequence[str],
    leakage_guard: Mapping[str, object],
) -> dict[str, object]:
    reconstructed_count = _int(
        reconstruction_coverage.get("reconstructed_first_recovery_row_count")
    )
    selected_count = _int(reconstruction_coverage.get("selected_target_count"))
    additional = max(0, reconstructed_count - selected_count)
    bias_detected = selected_vs_reconstructed_coverage.get("bias_detected") is True
    coverage_confound_plausible = (
        v113_blocker_review.get("coverage_plausibly_confounds_v113_readiness")
        is True
    )
    leakage = _int(leakage_guard.get("leakage_count")) > 0
    recommend_expansion = (
        not missing_evidence
        and not leakage
        and bias_detected
        and coverage_confound_plausible
    )
    return {
        "expanded_archive_replay_executed": False,
        "readiness_rerun_executed": False,
        "reconstructed_target_count": reconstructed_count,
        "selected_target_count": selected_count,
        "selected_target_share": _share(selected_count, reconstructed_count),
        "minimum_additional_targets_to_exhaustive_reconstructed_rows": additional,
        "additional_targets_by_source": _positive_counter_delta(
            _int_counter(reconstruction_coverage.get("reconstructed_by_source")),
            _int_counter(reconstruction_coverage.get("selected_by_source")),
        ),
        "additional_targets_by_seed": _positive_counter_delta(
            _int_counter(reconstruction_coverage.get("reconstructed_by_seed")),
            _int_counter(reconstruction_coverage.get("selected_by_seed")),
        ),
        "additional_targets_by_logged_action": _positive_counter_delta(
            _int_counter(reconstruction_coverage.get("reconstructed_by_logged_action")),
            _int_counter(reconstruction_coverage.get("selected_by_logged_action")),
        ),
        "recommend_expanded_archive": recommend_expansion,
        "expanded_archive_allowed_by_v114": recommend_expansion,
        "readiness_rerun_on_current_v109_archive_allowed": (
            v113_blocker_review.get("readiness_rerun_on_current_v109_archive_allowed")
            is True
            and not missing_evidence
            and not leakage
        ),
        "readiness_rerun_after_expanded_archive_allowed": recommend_expansion,
        "budget_policy": (
            "only recommend expanded replay when concrete selection bias plausibly "
            "confounds v113 readiness interpretation"
        ),
    }


def _leakage_guard(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    archive_leakage = trainable_public_input_leakage(rows)
    field_names = _trainable_public_input_field_paths(rows)
    field_leaks = first_recovery_data_coverage_trainable_field_leakage(field_names)
    example_paths = {
        str(item.get("path"))
        for item in _list(archive_leakage.get("examples"))
        if _mapping(item).get("path") is not None
    }
    leaks = sorted(set(field_leaks) | example_paths)
    return {
        "answer": "leakage_detected" if leaks else "leakage_free",
        "trainable_public_input_leakage": archive_leakage,
        "field_name_leakage_count": len(field_leaks),
        "leaking_trainable_field_names": list(field_leaks),
        "leakage_count": len(leaks),
        "leaking_trainable_signals": leaks,
        "forbidden_trainable_signal_tokens": sorted(FORBIDDEN_TRAINABLE_SIGNAL_TOKENS),
        "identity_fields_used_for_coverage_grouping_only": True,
    }


def _classification(
    *,
    missing_evidence: Sequence[str],
    selected_vs_reconstructed_coverage: Mapping[str, object],
    v113_blocker_review: Mapping[str, object],
    expansion_budget: Mapping[str, object],
    leakage_guard: Mapping[str, object],
) -> dict[str, object]:
    labels: list[str] = ["diagnostics_only_no_runtime_promotion"]
    if _int(leakage_guard.get("leakage_count")) > 0:
        labels.insert(0, "trainable_signal_leakage_detected")
        labels.append("readiness_rerun_blocked")
        primary = "trainable_signal_leakage_detected"
    elif missing_evidence:
        labels.insert(0, "missing_evidence_inconclusive")
        labels.append("readiness_rerun_blocked")
        primary = "missing_evidence_inconclusive"
    else:
        if selected_vs_reconstructed_coverage.get("bias_detected") is True:
            labels.append("v109_selection_bias_detected")
        else:
            labels.append("v109_selection_representative")
        if expansion_budget.get("recommend_expanded_archive") is True:
            labels.append("expanded_archive_required")
            labels.append("readiness_rerun_blocked")
            primary = "v109_selection_bias_detected"
        else:
            labels.append("expanded_archive_not_justified")
            labels.append(
                "readiness_rerun_allowed"
                if expansion_budget.get("readiness_rerun_on_current_v109_archive_allowed")
                is True
                else "readiness_rerun_blocked"
            )
            primary = (
                "v109_selection_bias_detected"
                if selected_vs_reconstructed_coverage.get("bias_detected") is True
                else "v109_selection_representative"
            )
    deduped = _dedupe_allowed(labels)
    return {
        "primary": primary,
        "labels": deduped,
        "missing_evidence": sorted(set(str(item) for item in missing_evidence)),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
        "v113_coverage_confound_plausible": v113_blocker_review.get(
            "coverage_plausibly_confounds_v113_readiness"
        )
        is True,
    }


def _recommendation(
    *,
    classification: Mapping[str, object],
    expansion_budget: Mapping[str, object],
    v113_blocker_review: Mapping[str, object],
) -> dict[str, object]:
    labels = set(str(item) for item in _list(classification.get("labels")))
    if "trainable_signal_leakage_detected" in labels:
        next_step = "stop_until_trainable_signal_leakage_is_removed"
        summary = (
            "v114 found identity or private-world trainable signal leakage; do not "
            "expand, train, rerun readiness, or patch observation contracts from this data."
        )
    elif "missing_evidence_inconclusive" in labels:
        next_step = "load_required_v109_v112_v113_inputs_before_deciding_archive_expansion"
        summary = "v114 could not decide coverage bias because required evidence is missing."
    elif "expanded_archive_required" in labels:
        next_step = "run_expanded_v109_archive_replay_before_any_v113_readiness_rerun"
        summary = (
            "v109 selected coverage is biased relative to reconstructed first-recovery "
            "support, and that bias plausibly confounds v113 readiness interpretation."
        )
    else:
        next_step = "stop_archive_expansion_move_to_non_lookup_ranker_diagnostic_or_close_line"
        summary = (
            "v109 selected coverage is representative enough for this diagnostic, or "
            "coverage does not plausibly confound the v113 readiness interpretation."
        )
    return {
        "next_step": next_step,
        "summary": summary,
        "expanded_archive_allowed": expansion_budget.get("expanded_archive_allowed_by_v114")
        is True,
        "readiness_rerun_on_current_v109_archive_allowed": expansion_budget.get(
            "readiness_rerun_on_current_v109_archive_allowed"
        )
        is True,
        "readiness_rerun_after_expanded_archive_allowed": expansion_budget.get(
            "readiness_rerun_after_expanded_archive_allowed"
        )
        is True,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "coverage_cannot_be_separated_from_lookup_design_failure": (
            v113_blocker_review.get("coverage_cannot_be_separated_from_lookup_design_failure")
            is True
        ),
    }


def _dimension_comparison(
    name: str,
    reconstructed: Mapping[str, int],
    selected: Mapping[str, int],
    *,
    min_reconstructed_key_count: int,
) -> dict[str, object]:
    reconstructed_counter = Counter(
        {str(key): int(value) for key, value in reconstructed.items()}
    )
    selected_counter = Counter({str(key): int(value) for key, value in selected.items()})
    reconstructed_total = sum(reconstructed_counter.values())
    selected_total = sum(selected_counter.values())
    global_rate = _share(selected_total, reconstructed_total)
    rows: list[dict[str, object]] = []
    bias_reasons: list[dict[str, object]] = []
    for key in sorted(set(reconstructed_counter) | set(selected_counter)):
        reconstructed_count = int(reconstructed_counter[key])
        selected_count = int(selected_counter[key])
        reconstructed_share = _share(reconstructed_count, reconstructed_total)
        selected_share = _share(selected_count, selected_total)
        selection_rate = _share(selected_count, reconstructed_count)
        delta = _round(selected_share - reconstructed_share)
        abs_delta = abs(delta)
        expected_selected = _round(reconstructed_share * float(selected_total))
        missing = reconstructed_count > 0 and selected_count == 0
        enough_support = reconstructed_count >= min_reconstructed_key_count
        underrepresented = (
            enough_support
            and reconstructed_count > 0
            and (
                missing
                or abs_delta >= BIAS_DISTRIBUTION_DELTA_THRESHOLD
                and selected_share < reconstructed_share
                or (
                    global_rate > 0.0
                    and selection_rate <= global_rate / BIAS_RATE_MULTIPLIER
                )
            )
        )
        overrepresented = (
            enough_support
            and reconstructed_count > 0
            and selected_count > 0
            and (
                abs_delta >= BIAS_DISTRIBUTION_DELTA_THRESHOLD
                and selected_share > reconstructed_share
                or (
                    global_rate > 0.0
                    and selection_rate >= min(1.0, global_rate * BIAS_RATE_MULTIPLIER)
                    and abs_delta >= BIAS_DISTRIBUTION_DELTA_THRESHOLD / 2.0
                )
            )
        )
        row = {
            "key": key,
            "reconstructed_count": reconstructed_count,
            "selected_count": selected_count,
            "reconstructed_share": reconstructed_share,
            "selected_share": selected_share,
            "selection_rate": selection_rate,
            "expected_selected_if_proportional": expected_selected,
            "selected_minus_reconstructed_share_delta": delta,
            "abs_share_delta": _round(abs_delta),
            "missing_in_selected": missing,
            "underrepresented": underrepresented,
            "overrepresented": overrepresented,
        }
        rows.append(row)
        if underrepresented or overrepresented:
            bias_reasons.append(row)
    missing_count = sum(1 for item in rows if item["missing_in_selected"])
    return {
        "dimension": name,
        "reconstructed_total": reconstructed_total,
        "selected_total": selected_total,
        "global_selection_rate": global_rate,
        "reconstructed_key_count": len(
            [key for key, value in reconstructed_counter.items() if value > 0]
        ),
        "selected_key_count": len(
            [key for key, value in selected_counter.items() if value > 0]
        ),
        "missing_reconstructed_key_count": missing_count,
        "bias_detected": bool(bias_reasons),
        "bias_reason_count": len(bias_reasons),
        "max_abs_share_delta": _round(
            max((float(item["abs_share_delta"]) for item in rows), default=0.0)
        ),
        "rows": rows,
        "bias_examples": bias_reasons[:MAX_EXAMPLES],
    }


def _selected_tick_coverage(target_selection: Mapping[str, object]) -> dict[str, object]:
    selected_targets = [_mapping(item) for item in _list(target_selection.get("selected_targets"))]
    selected_tick_counts = Counter(
        str(row.get("recovery_tick"))
        for row in selected_targets
        if row.get("recovery_tick") is not None
    )
    skipped_examples = [_mapping(item) for item in _list(target_selection.get("skipped_examples"))]
    skipped_tick_counts = Counter(
        str(row.get("recovery_tick"))
        for row in skipped_examples
        if row.get("recovery_tick") is not None
    )
    skipped_count = _int(target_selection.get("skipped_target_count"))
    return {
        "selected_tick_count": len(selected_tick_counts),
        "selected_target_ticks": _counter_to_dict(selected_tick_counts),
        "reported_skipped_example_ticks": _counter_to_dict(skipped_tick_counts),
        "skipped_tick_distribution_complete": len(skipped_examples) == skipped_count,
        "tick_bias_used_for_classification": False,
        "reason": (
            "v109 report does not serialize complete reconstructed tick distribution; "
            "tick support is reported but not used as primary bias evidence"
        ),
    }


def _blocker_plausibility(
    blocker: str,
    *,
    bias_detected: bool,
    alias_selected_only: bool,
) -> dict[str, object]:
    if blocker == "state_action_interaction_does_not_beat_action_only_heldout":
        plausible = bias_detected
        reason = "heldout lookup comparison can be distorted by selected archive support"
    elif blocker == "alias_resolution_not_stable_under_quantization":
        plausible = bias_detected or alias_selected_only
        reason = "alias evidence was measured on selected archive support"
    elif blocker == "shuffled_control_artifact_detected":
        plausible = bias_detected
        reason = "shuffled-control artifact is plausible when selected support is biased"
    else:
        plausible = False
        reason = "blocker is not a coverage-specific v113 blocker"
    return {
        "blocker": blocker,
        "coverage_plausible": plausible,
        "reason": reason,
    }


def _missing_evidence(
    *,
    source_reports: Mapping[str, object],
    reconstruction_coverage: Mapping[str, object],
    selected_vs_reconstructed_coverage: Mapping[str, object],
) -> list[str]:
    missing: list[str] = []
    for key in (
        "archive_report",
        "archive_rows",
        "public_signal_audit",
        "state_action_readiness_audit",
    ):
        evidence = _mapping(source_reports.get(key))
        if evidence.get("loaded") is not True:
            missing.append(key)
        elif evidence.get("schema_matches") is False and key != "archive_rows":
            missing.append(f"{key}_schema_mismatch")
    if reconstruction_coverage.get("coverage_status") != "available":
        missing.append("reconstruction_coverage")
    if _int(selected_vs_reconstructed_coverage.get("reconstructed_target_count")) <= 0:
        missing.append("selected_vs_reconstructed_coverage")
    return sorted(set(missing))


def _selected_counts_from_archive_groups(
    groups: Sequence[object],
) -> dict[str, Counter[str]]:
    by_seed: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_seed_source: Counter[str] = Counter()
    by_logged_action: Counter[str] = Counter()
    for group in groups:
        rows = tuple(getattr(group, "rows", ()))
        first = _mapping(rows[0]) if rows else {}
        provenance = _mapping(first.get("provenance"))
        seed = provenance.get("seed")
        source = provenance.get("source_kind")
        action = provenance.get("logged_action")
        if seed is not None:
            by_seed[str(seed)] += 1
        if source is not None:
            by_source[str(source)] += 1
        if seed is not None and source is not None:
            by_seed_source[f"{source}:{seed}"] += 1
        if action is not None:
            by_logged_action[str(action)] += 1
    return {
        "seed": by_seed,
        "source": by_source,
        "seed_source": by_seed_source,
        "logged_action": by_logged_action,
    }


def _trainable_public_input_field_paths(
    rows: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    paths: set[str] = set()
    for row in rows:
        trainable = _mapping(row.get("trainable_public_input"))
        paths.update(_nested_paths(trainable, prefix="trainable_public_input"))
    return tuple(sorted(paths))


def _nested_paths(value: object, *, prefix: str) -> tuple[str, ...]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            paths.append(path)
            paths.extend(_nested_paths(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_nested_paths(child, prefix=f"{prefix}[{index}]"))
    return tuple(paths)


def _field_tokens(value: str) -> tuple[str, ...]:
    normalized = re.sub(r"[^a-z0-9_]+", ".", value.lower())
    tokens: list[str] = []
    for chunk in normalized.split("."):
        if not chunk:
            continue
        tokens.append(chunk)
        tokens.extend(part for part in chunk.split("_") if part)
    return tuple(tokens)


def _family_order(families: Mapping[str, object]) -> tuple[str, ...]:
    ordered = [family for family in SIGNAL_FAMILY_ORDER if family in families]
    ordered.extend(sorted(str(family) for family in families if family not in ordered))
    return tuple(ordered)


def _largest_dimension_delta(dimensions: Mapping[str, object]) -> dict[str, object]:
    best = {"dimension": None, "max_abs_share_delta": 0.0}
    for name, section in dimensions.items():
        delta = float(_mapping(section).get("max_abs_share_delta") or 0.0)
        if delta > float(best["max_abs_share_delta"]):
            best = {"dimension": name, "max_abs_share_delta": _round(delta)}
    return best


def _positive_counter_delta(
    left: Mapping[str, int],
    right: Mapping[str, int],
) -> dict[str, int]:
    delta = {
        str(key): max(0, int(left.get(key, 0)) - int(right.get(key, 0)))
        for key in sorted(set(left) | set(right))
    }
    return {key: value for key, value in delta.items() if value > 0}


def _counter_abs_delta(left: Mapping[str, int], right: Mapping[str, int]) -> int:
    return sum(
        abs(int(left.get(key, 0)) - int(right.get(key, 0)))
        for key in set(left) | set(right)
    )


def _int_counter(value: object) -> dict[str, int]:
    mapping = _mapping(value)
    return {
        str(key): _int(item)
        for key, item in sorted(mapping.items(), key=lambda pair: str(pair[0]))
        if _int(item) > 0
    }


def _first_positive_int(*values: object) -> int:
    for value in values:
        parsed = _int(value)
        if parsed > 0:
            return parsed
    return 0


def _dedupe_allowed(labels: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    allowed = set(ALLOWED_CLASSIFICATIONS)
    for label in labels:
        if label not in allowed or label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return deduped
