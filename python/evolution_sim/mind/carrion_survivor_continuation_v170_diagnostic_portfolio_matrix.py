from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    _dominant_count_share,
    _float,
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    _safe_rate,
    write_json,
)
from evolution_sim.mind.carrion_sequence_context_comparator_support_closeout import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V153_REPORT_PATH,
    M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY,
    M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    DEFAULT_DATASET_OUTPUT_PATH as DEFAULT_V154_DATASET_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V154_REPORT_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V155_REPORT_PATH,
    EXPECTED_V154_SUPPORT_READY_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION,
    load_json_report,
    load_v154_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    _complete_bool_mask,
    _safe_action_set,
)
from evolution_sim.mind.carrion_survivor_continuation_v160_action_value_scorer import (
    _vector_distance,
)
from evolution_sim.mind.carrion_survivor_continuation_v165_preterminal_target_dataset_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v166_source_split_action_value_scorer import (
    DEFAULT_V165_DATASET_PATH,
    EXPECTED_V165_DATASET_DIGEST,
    source_split_groups,
)
from evolution_sim.mind.carrion_survivor_continuation_v167_source_split_failure_autopsy import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V167_REPORT_PATH,
    EXPECTED_V166_EXACT_DIGEST,
    M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v168_public_feature_contract_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V168_REPORT_PATH,
    EXPECTED_V167_EXACT_DIGEST,
    M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION,
    _feature_vectors_from_payloads,
    _json_round_trip_digest,
    candidate_feature_leakage_scan,
)
from evolution_sim.mind.carrion_survivor_continuation_v169_public_temporal_context_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V169_REPORT_PATH,
    EXPECTED_V168_EXACT_DIGEST,
    FEATURE_FAMILIES as V169_FEATURE_FAMILIES,
    M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_SCHEMA_VERSION,
    build_candidate_feature_families,
    load_public_temporal_contexts,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix_v1"
)
EXPECTED_V169_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v169_public_temporal_context_probe_"
    "public_temporal_context_probe_no_net_viable_projection_closed"
)
EXPECTED_V169_EXACT_DIGEST = (
    "dbc96a211b76447d7f97cc26c79d207ca6f697e050b99e6a1f09dd8f46efbcfd"
)
EXPECTED_V155_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_train_eval_"
    "pretraining_alias_prior_blocked_closed_no_training"
)
EXPECTED_V155_EXACT_DIGEST = (
    "ec74b0a8c275bad70fd4dbf212c3c124ed20d61118255798e712b800fca0c476"
)
EXPECTED_V154_EXACT_DIGEST = (
    "a93b3f9ab9a823105daaef30e59e7c2612da3f283ee5740b3e3ed3f7ef9f0d55"
)
EXPECTED_V153_EXACT_DIGEST = (
    "30134d4699e2043695aaa9b7bc388404971dcf8a59e4cc2bd802d28407a37c73"
)
V155_REFERENCE_EXACT_FEATURE_GROUP_COUNT = 11
V155_REFERENCE_CONFLICTING_GROUP_COUNT = 10
V155_REFERENCE_CONFLICTING_ROW_COUNT = 119
TOP_N_VALUES = (1, 3, 5)
REPLAY_EXPANSION_TARGET_SEEDS = (13, 19, 29, 41, 5, 37)
BASELINE_ZERO_HIT_SEEDS = (13, 19, 29, 41)
REGRESSION_NEW_ZERO_SEED = 5
LOW_SUPPORT_EDGE_SEED = 37
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v170-carrion-survivor-continuation-diagnostic-portfolio-matrix.json"
)
DEFAULT_SHARD_PLAN_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v170-carrion-survivor-continuation-replay-expansion-shards.jsonl"
)
FORBIDDEN_SEQUENCE_PAYLOAD_TOKENS = (
    "seed",
    "fixture",
    "branch",
    "tick",
    "agent",
    "path",
    "digest",
    "provenance",
    "private",
    "future",
    "outcome",
    "label",
)


class CarrionSurvivorContinuationV170DiagnosticPortfolioMatrixError(ValueError):
    pass


def run_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix(
    *,
    v169_report_path: str | Path = DEFAULT_V169_REPORT_PATH,
    v168_report_path: str | Path = DEFAULT_V168_REPORT_PATH,
    v167_report_path: str | Path = DEFAULT_V167_REPORT_PATH,
    v165_dataset_path: str | Path = DEFAULT_V165_DATASET_PATH,
    v155_report_path: str | Path = DEFAULT_V155_REPORT_PATH,
    v154_report_path: str | Path = DEFAULT_V154_REPORT_PATH,
    v154_dataset_path: str | Path = DEFAULT_V154_DATASET_PATH,
    v153_report_path: str | Path = DEFAULT_V153_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    shard_plan_output_path: str | Path | None = DEFAULT_SHARD_PLAN_OUTPUT_PATH,
    expected_v169_exact_digest: str | None = EXPECTED_V169_EXACT_DIGEST,
    expected_v169_classification: str = EXPECTED_V169_CLASSIFICATION,
    expected_v168_exact_digest: str | None = EXPECTED_V168_EXACT_DIGEST,
    expected_v167_exact_digest: str | None = EXPECTED_V167_EXACT_DIGEST,
    expected_v165_dataset_digest: str | None = EXPECTED_V165_DATASET_DIGEST,
    expected_v155_exact_digest: str | None = EXPECTED_V155_EXACT_DIGEST,
    expected_v154_exact_digest: str | None = EXPECTED_V154_EXACT_DIGEST,
    expected_v153_exact_digest: str | None = EXPECTED_V153_EXACT_DIGEST,
) -> dict[str, object]:
    v169_report = load_json_report(v169_report_path)
    v168_report = load_json_report(v168_report_path)
    v167_report = load_json_report(v167_report_path)
    v165_rows = load_v154_dataset(v165_dataset_path)
    v155_report = load_json_report(v155_report_path)
    v154_report = load_json_report(v154_report_path)
    v154_rows = load_v154_dataset(v154_dataset_path)
    v153_report = load_json_report(v153_report_path)

    source_validation = validate_v170_sources(
        v169_report=v169_report,
        v168_report=v168_report,
        v167_report=v167_report,
        v165_rows=v165_rows,
        v155_report=v155_report,
        v154_report=v154_report,
        v154_rows=v154_rows,
        v153_report=v153_report,
        expected_v169_exact_digest=expected_v169_exact_digest,
        expected_v169_classification=expected_v169_classification,
        expected_v168_exact_digest=expected_v168_exact_digest,
        expected_v167_exact_digest=expected_v167_exact_digest,
        expected_v165_dataset_digest=expected_v165_dataset_digest,
        expected_v155_exact_digest=expected_v155_exact_digest,
        expected_v154_exact_digest=expected_v154_exact_digest,
        expected_v153_exact_digest=expected_v153_exact_digest,
    )
    set_lane = build_set_valued_ranking_diagnostic(
        rows=v165_rows,
        v167_report=v167_report,
        v169_report=v169_report,
    )
    sequence_lane = build_sequence_memory_alias_diagnostic(
        v154_rows=v154_rows,
        v154_report=v154_report,
        v155_report=v155_report,
    )
    shard_plan = build_replay_expansion_shard_plan(
        rows=v165_rows,
        v167_report=v167_report,
        v169_report=v169_report,
    )
    if shard_plan_output_path is not None:
        _write_jsonl(shard_plan_output_path, _list_of_mappings(shard_plan.get("shards")))
    shard_plan_output = _shard_plan_output_summary(
        shard_plan=shard_plan,
        shard_plan_output_path=shard_plan_output_path,
    )
    classification = _classification(
        source_validation=source_validation,
        set_lane=set_lane,
        sequence_lane=sequence_lane,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V170_DIAGNOSTIC_PORTFOLIO_MATRIX_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v169_report": str(v169_report_path),
            "v168_report": str(v168_report_path),
            "v167_report": str(v167_report_path),
            "v165_dataset": str(v165_dataset_path),
            "v155_report": str(v155_report_path),
            "v154_report": str(v154_report_path),
            "v154_dataset": str(v154_dataset_path),
            "v153_report": str(v153_report_path),
            "output": str(output_path),
            "shard_plan_output": (
                str(shard_plan_output_path) if shard_plan_output_path is not None else None
            ),
            "top_n_values": list(TOP_N_VALUES),
            "replay_expansion_target_seeds": list(REPLAY_EXPANSION_TARGET_SEEDS),
        },
        "source_validation": source_validation,
        "portfolio_lanes": {
            "set_valued_ranking": set_lane,
            "sequence_memory_alias": sequence_lane,
            "replay_expansion_shard_planner": shard_plan,
        },
        "negative_control_summary": _negative_control_summary(
            set_lane=set_lane,
            sequence_lane=sequence_lane,
        ),
        "shard_plan_output": shard_plan_output,
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "dataset_digests": {
            "v165_dataset": stable_payload_digest(v165_rows),
            "v154_dataset": stable_payload_digest(v154_rows),
        },
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "replay_viewer_schema_changed": False,
        "training_ran": False,
        "shadow_eval_ran": False,
        "live_ab_allowed": False,
        "k_tuning_ran": False,
        "threshold_tuning_ran": False,
        "promotion_authorized": False,
        "staging_authorized": False,
        "commit_authorized": False,
        "reset_authorized": False,
        "clean_authorized": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v170_sources(
    *,
    v169_report: Mapping[str, object],
    v168_report: Mapping[str, object],
    v167_report: Mapping[str, object],
    v165_rows: Sequence[Mapping[str, object]],
    v155_report: Mapping[str, object],
    v154_report: Mapping[str, object],
    v154_rows: Sequence[Mapping[str, object]],
    v153_report: Mapping[str, object],
    expected_v169_exact_digest: str | None = EXPECTED_V169_EXACT_DIGEST,
    expected_v169_classification: str = EXPECTED_V169_CLASSIFICATION,
    expected_v168_exact_digest: str | None = EXPECTED_V168_EXACT_DIGEST,
    expected_v167_exact_digest: str | None = EXPECTED_V167_EXACT_DIGEST,
    expected_v165_dataset_digest: str | None = EXPECTED_V165_DATASET_DIGEST,
    expected_v155_exact_digest: str | None = EXPECTED_V155_EXACT_DIGEST,
    expected_v154_exact_digest: str | None = EXPECTED_V154_EXACT_DIGEST,
    expected_v153_exact_digest: str | None = EXPECTED_V153_EXACT_DIGEST,
) -> dict[str, object]:
    failures: list[str] = []
    v165_digest = stable_payload_digest(v165_rows)
    v154_dataset_digest = stable_payload_digest(v154_rows)
    v169_classification = str(
        _mapping(v169_report.get("classification")).get("primary") or ""
    )
    v155_classification = str(
        _mapping(v155_report.get("classification")).get("primary") or ""
    )
    if (
        v169_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_SCHEMA_VERSION
    ):
        failures.append("v169_schema_version_mismatch")
    if v169_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_POLICY:
        failures.append("v169_policy_mismatch")
    if v169_classification != expected_v169_classification:
        failures.append("v169_unexpected_classification")
    v169_exact = exact_digest_validation_report(v169_report)
    if v169_exact.get("passed") is not True:
        failures.append("v169_exact_digest_mismatch")
    observed_v169_digest = str(v169_report.get("exact_digest") or "")
    if expected_v169_exact_digest and observed_v169_digest != expected_v169_exact_digest:
        failures.append("v169_unexpected_exact_digest")

    if (
        v168_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION
    ):
        failures.append("v168_schema_version_mismatch")
    if v168_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_POLICY:
        failures.append("v168_policy_mismatch")
    v168_exact = exact_digest_validation_report(v168_report)
    observed_v168_digest = str(v168_report.get("exact_digest") or "")
    if v168_exact.get("passed") is not True:
        failures.append("v168_exact_digest_mismatch")
    if expected_v168_exact_digest and observed_v168_digest != expected_v168_exact_digest:
        failures.append("v168_unexpected_exact_digest")

    if (
        v167_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION
    ):
        failures.append("v167_schema_version_mismatch")
    if v167_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_POLICY:
        failures.append("v167_policy_mismatch")
    v167_exact = exact_digest_validation_report(v167_report)
    observed_v167_digest = str(v167_report.get("exact_digest") or "")
    if v167_exact.get("passed") is not True:
        failures.append("v167_exact_digest_mismatch")
    if expected_v167_exact_digest and observed_v167_digest != expected_v167_exact_digest:
        failures.append("v167_unexpected_exact_digest")
    if expected_v165_dataset_digest and v165_digest != expected_v165_dataset_digest:
        failures.append("v165_dataset_digest_mismatch")
    for label, report in (
        ("v169", v169_report),
        ("v168", v168_report),
        ("v167", v167_report),
    ):
        if report.get("dataset_digest") != v165_digest:
            failures.append(f"{label}_reported_v165_dataset_digest_mismatch")

    if (
        v155_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION
    ):
        failures.append("v155_schema_version_mismatch")
    if v155_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY:
        failures.append("v155_policy_mismatch")
    if v155_classification != EXPECTED_V155_CLASSIFICATION:
        failures.append("v155_unexpected_classification")
    v155_exact = exact_digest_validation_report(v155_report)
    observed_v155_digest = str(v155_report.get("exact_digest") or "")
    if v155_exact.get("passed") is not True:
        failures.append("v155_exact_digest_mismatch")
    if expected_v155_exact_digest and observed_v155_digest != expected_v155_exact_digest:
        failures.append("v155_unexpected_exact_digest")
    v155_source = _mapping(v155_report.get("source_validation"))
    if v155_source.get("dataset_digest") != v154_dataset_digest:
        failures.append("v155_source_dataset_digest_mismatch")
    v155_alias_reference = _v155_reference_alias_counts(v155_report)
    if v155_alias_reference.get("matches_required_reference") is not True:
        failures.append("v155_reference_alias_counts_mismatch")

    if (
        v154_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION
    ):
        failures.append("v154_schema_version_mismatch")
    if v154_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY:
        failures.append("v154_policy_mismatch")
    v154_classification = str(
        _mapping(v154_report.get("classification")).get("primary") or ""
    )
    if v154_classification != EXPECTED_V154_SUPPORT_READY_CLASSIFICATION:
        failures.append("v154_unexpected_classification")
    observed_v154_digest = str(v154_report.get("exact_digest") or "")
    if expected_v154_exact_digest and observed_v154_digest != expected_v154_exact_digest:
        failures.append("v154_unexpected_exact_digest")
    if _mapping(v154_report.get("dataset")).get("dataset_digest") != v154_dataset_digest:
        failures.append("v154_report_dataset_digest_mismatch")

    if (
        v153_report.get("schema_version")
        != M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION
    ):
        failures.append("v153_schema_version_mismatch")
    if v153_report.get("policy") != M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY:
        failures.append("v153_policy_mismatch")
    observed_v153_digest = str(v153_report.get("exact_digest") or "")
    if expected_v153_exact_digest and observed_v153_digest != expected_v153_exact_digest:
        failures.append("v153_unexpected_exact_digest")

    lifecycle = _source_lifecycle_scan(
        reports={
            "v169": v169_report,
            "v168": v168_report,
            "v167": v167_report,
            "v155": v155_report,
            "v154": v154_report,
        }
    )
    if lifecycle.get("passed") is not True:
        failures.append("source_lifecycle_authorization_mismatch")
    return {
        "policy": "m3_carrion_survivor_continuation_v170_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v169_classification": expected_v169_classification,
        "observed_v169_classification": v169_classification,
        "expected_v169_exact_digest": expected_v169_exact_digest,
        "observed_v169_exact_digest": observed_v169_digest,
        "v169_exact_digest_validation": v169_exact,
        "expected_v168_exact_digest": expected_v168_exact_digest,
        "observed_v168_exact_digest": observed_v168_digest,
        "v168_exact_digest_validation": v168_exact,
        "expected_v167_exact_digest": expected_v167_exact_digest,
        "observed_v167_exact_digest": observed_v167_digest,
        "v167_exact_digest_validation": v167_exact,
        "expected_v165_dataset_digest": expected_v165_dataset_digest,
        "observed_v165_dataset_digest": v165_digest,
        "expected_v155_exact_digest": expected_v155_exact_digest,
        "observed_v155_exact_digest": observed_v155_digest,
        "v155_exact_digest_validation": v155_exact,
        "expected_v154_exact_digest": expected_v154_exact_digest,
        "observed_v154_exact_digest": observed_v154_digest,
        "v154_legacy_exact_digest_note": (
            "v154 exact_digest is validated by embedded digest; older report "
            "style is not required to recompute without exact_digest"
        ),
        "expected_v153_exact_digest": expected_v153_exact_digest,
        "observed_v153_exact_digest": observed_v153_digest,
        "v153_legacy_exact_digest_note": (
            "v153 exact_digest is validated by embedded digest; older report "
            "style is not required to recompute without exact_digest"
        ),
        "v154_dataset_digest": v154_dataset_digest,
        "v155_reference_alias_counts": v155_alias_reference,
        "source_lifecycle_authorization": lifecycle,
        "v165_row_count": len(v165_rows),
        "v154_row_count": len(v154_rows),
    }


def build_set_valued_ranking_diagnostic(
    *,
    rows: Sequence[Mapping[str, object]],
    v167_report: Mapping[str, object],
    v169_report: Mapping[str, object],
) -> dict[str, object]:
    temporal = load_public_temporal_contexts(rows)
    families = build_candidate_feature_families(
        rows=rows,
        temporal_contexts_by_row=_mapping(temporal.get("contexts_by_row")),
    )
    baseline_zero = _baseline_zero_hit_seeds(v169_report)
    reports: list[dict[str, object]] = []
    for family_name in V169_FEATURE_FAMILIES:
        family = _mapping(families.get(family_name))
        payloads = _list_of_mappings(family.get("payloads"))
        leakage = candidate_feature_leakage_scan(payloads)
        vectors, feature_keys, normalization = _feature_vectors_from_payloads(payloads)
        for top_n in TOP_N_VALUES:
            entry = _evaluate_set_valued_family(
                rows=rows,
                vectors=vectors,
                feature_keys=feature_keys,
                family_name=family_name,
                top_n=top_n,
                randomized_safe_sets=False,
            )
            control = _evaluate_set_valued_family(
                rows=rows,
                vectors=vectors,
                feature_keys=feature_keys,
                family_name=family_name,
                top_n=top_n,
                randomized_safe_sets=True,
            )
            entry["feature_payload_digest"] = stable_payload_digest(payloads)
            entry["feature_key_count"] = len(feature_keys)
            entry["normalization_digest"] = stable_payload_digest(normalization)
            entry["leakage_scan"] = leakage
            entry["baseline_comparison"] = _zero_hit_baseline_comparison(
                baseline_zero=baseline_zero,
                zero_hit_seeds=entry.get("zero_safe_hit_preterminal_source_seeds"),
            )
            entry["negative_control"] = _set_lane_negative_control(
                actual=entry,
                control=control,
                baseline_zero=baseline_zero,
            )
            reports.append(entry)
    best = _best_set_valued_entry(reports=reports, baseline_zero=baseline_zero)
    leakage_passed = all(
        _mapping(report.get("leakage_scan")).get("passed") is True
        for report in reports
    )
    suspicious_entries = [
        report
        for report in reports
        if _mapping(report.get("negative_control")).get("passed") is not True
    ]
    best_control = _mapping(best.get("negative_control"))
    controls_passed = (
        best_control.get("passed") is True
        if best
        else not suspicious_entries
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v170_set_valued_ranking_diagnostic_v1",
        "diagnostics_only": True,
        "uses_v165_rows": True,
        "uses_v167_failure_evidence": bool(v167_report),
        "uses_v169_feature_families": True,
        "top_n_values": list(TOP_N_VALUES),
        "no_runtime_top_n_selected": True,
        "k_tuning_ran": False,
        "runtime_action_selection_changed": False,
        "temporal_context_load": temporal.get("summary"),
        "baseline_zero_safe_hit_preterminal_source_seeds": baseline_zero,
        "matrix": reports,
        "best_matrix_entry": best,
        "zero_hit_coverage_improved": bool(best.get("zero_hit_coverage_improved")),
        "best_entry_sets_are_broad": bool(best.get("sets_are_broad")),
        "leakage_scan_passed": leakage_passed,
        "negative_controls_passed": controls_passed,
        "negative_control_suspicious_entry_count": len(suspicious_entries),
        "lane_valid": leakage_passed and controls_passed,
    }


def build_sequence_memory_alias_diagnostic(
    *,
    v154_rows: Sequence[Mapping[str, object]],
    v154_report: Mapping[str, object],
    v155_report: Mapping[str, object],
) -> dict[str, object]:
    reference = _v155_reference_alias_counts(v155_report)
    candidate_payloads = []
    for policy_id, builder in (
        ("current_public_observation_action_mask", _sequence_features_current),
        (
            "current_public_observation_action_mask_plus_existing_legal_prior_sequence_summary",
            _sequence_features_with_prior_summary,
        ),
    ):
        feature_rows = []
        for index, row in enumerate(v154_rows):
            features = builder(row)
            feature_rows.append(
                {
                    "row_index": index,
                    "features": features,
                    "label_action": _label_action(row),
                }
            )
        leakage = sequence_feature_leakage_scan(
            [_mapping(item.get("features")) for item in feature_rows]
        )
        alias = _sequence_alias_report(feature_rows)
        control_alias = _sequence_alias_report(_shuffled_label_rows(feature_rows))
        comparison = _sequence_alias_reference_comparison(
            alias=alias,
            reference=reference,
        )
        candidate_payloads.append(
            {
                "policy_id": policy_id,
                "row_count": len(feature_rows),
                "feature_payload_digest": stable_payload_digest(
                    [_mapping(item.get("features")) for item in feature_rows]
                ),
                "alias_report": alias,
                "reference_comparison": comparison,
                "leakage_scan": leakage,
                "negative_control": _sequence_negative_control(
                    alias=alias,
                    control_alias=control_alias,
                    comparison=comparison,
                ),
                "training_authorized": False,
                "runtime_action_selection_changed": False,
            }
        )
    best = _best_sequence_alias_entry(candidate_payloads)
    leakage_passed = all(
        _mapping(report.get("leakage_scan")).get("passed") is True
        for report in candidate_payloads
    )
    controls_passed = all(
        _mapping(report.get("negative_control")).get("passed") is True
        for report in candidate_payloads
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v170_sequence_memory_alias_diagnostic_v1",
        "diagnostics_only": True,
        "uses_v154_archive_dataset": True,
        "uses_v155_alias_reference": True,
        "v154_report_exact_digest": v154_report.get("exact_digest"),
        "v155_reference_alias_counts": reference,
        "reference_v155_conflicting_exact_groups": V155_REFERENCE_CONFLICTING_GROUP_COUNT,
        "reference_v155_conflicting_rows": V155_REFERENCE_CONFLICTING_ROW_COUNT,
        "reference_v155_exact_groups": V155_REFERENCE_EXACT_FEATURE_GROUP_COUNT,
        "candidate_feature_policies": candidate_payloads,
        "best_candidate": best,
        "material_alias_conflict_reduction": bool(
            best.get("material_alias_conflict_reduction")
        ),
        "leakage_scan_passed": leakage_passed,
        "negative_controls_passed": controls_passed,
        "lane_valid": leakage_passed and controls_passed,
        "training_authorized": False,
        "runtime_action_selection_changed": False,
    }


def build_replay_expansion_shard_plan(
    *,
    rows: Sequence[Mapping[str, object]],
    v167_report: Mapping[str, object],
    v169_report: Mapping[str, object],
) -> dict[str, object]:
    row_by_index = {index: row for index, row in enumerate(rows)}
    failed_by_seed: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for failure in _list_of_mappings(v167_report.get("preterminal_row_autopsy")):
        if failure.get("safe_hit") is True:
            continue
        failed_by_seed[_int(failure.get("source_seed"))].append(failure)
    rows_by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for index, row in row_by_index.items():
        if (
            row.get("schema_version")
            != M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ):
            continue
        metadata = _mapping(row.get("metadata"))
        rows_by_seed[_int(metadata.get("seed"))].append(
            {
                "row_index": index,
                "branch_tick": _int(metadata.get("branch_tick")),
                "safe_action_set": _safe_action_set(row),
                "source_group": f"v165_source_seed:{_int(metadata.get('seed'))}",
            }
        )
    v169_best = _mapping(v169_report.get("best_candidate"))
    shards = []
    for seed in REPLAY_EXPANSION_TARGET_SEEDS:
        seed_rows = rows_by_seed.get(int(seed), [])
        failed_rows = failed_by_seed.get(int(seed), [])
        if failed_rows:
            window_rows = [
                _source_window_row(failure, row_by_index=row_by_index)
                for failure in failed_rows
            ]
        else:
            window_rows = [
                {
                    "row_index": row["row_index"],
                    "source_seed": int(seed),
                    "source_tick": row["branch_tick"],
                    "safe_action_set": row["safe_action_set"],
                }
                for row in seed_rows
            ]
        windows = _branch_windows_from_rows(window_rows)
        shard_id = f"v171-carrion-survivor-continuation-expansion-seed-{int(seed):03d}"
        shards.append(
            {
                "schema_version": "m3_carrion_survivor_continuation_v170_replay_expansion_shard_plan_row_v1",
                "policy": "diagnostics_only_m3_carrion_survivor_continuation_v170_replay_expansion_shard_planner_v1",
                "shard_id": shard_id,
                "seed": int(seed),
                "rationale": _shard_seed_rationale(seed, v169_best=v169_best),
                "source_row_count": len(seed_rows),
                "failed_row_count": len(failed_rows),
                "proposed_branch_windows": windows,
                "expected_command_shape": _expected_v171_command_shape(
                    seed=int(seed),
                    shard_id=shard_id,
                    windows=windows,
                ),
                "plan_only_not_evidence": True,
                "long_branch_replay_ran": False,
                "training_authorized": False,
                "runtime_action_selection_changed": False,
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v170_replay_expansion_shard_planner_v1",
        "plan_only_not_evidence": True,
        "long_branch_replay_ran": False,
        "target_seeds": list(REPLAY_EXPANSION_TARGET_SEEDS),
        "baseline_zero_hit_seeds": list(BASELINE_ZERO_HIT_SEEDS),
        "regression_new_zero_seed": REGRESSION_NEW_ZERO_SEED,
        "low_support_edge_seed": LOW_SUPPORT_EDGE_SEED,
        "shard_count": len(shards),
        "shards": shards,
        "merge_policy": {
            "policy": "m3_carrion_survivor_continuation_v170_shard_merge_policy_v1",
            "merge_requires_all_target_seed_shards": True,
            "merge_order": "sort_by_seed_then_shard_id",
            "duplicate_source_row_policy": "same_payload_digest_allowed_conflict_rejected",
            "partial_shard_behavior": "fail_closed_unless_explicit_v171_allow_partial_flag",
            "partial_shards_are_not_evidence": True,
            "training_authorized_after_merge": False,
            "runtime_integration_authorized_after_merge": False,
        },
    }


def sequence_feature_leakage_scan(
    feature_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for row_index, payload in enumerate(feature_payloads):
        _scan_sequence_payload(
            value=payload,
            row_index=row_index,
            path=("features",),
            failures=failures,
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v170_sequence_payload_leakage_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
        "forbidden_tokens": list(FORBIDDEN_SEQUENCE_PAYLOAD_TOKENS),
        "payload_count": len(feature_payloads),
    }


def _evaluate_set_valued_family(
    *,
    rows: Sequence[Mapping[str, object]],
    vectors: Sequence[Mapping[str, float]],
    feature_keys: Sequence[str],
    family_name: str,
    top_n: int,
    randomized_safe_sets: bool,
) -> dict[str, object]:
    groups = source_split_groups(rows)
    group_by_row = {int(group["row_index"]): group for group in groups}
    safe_hits = 0
    unsupported_rows = 0
    unsupported_actions_total = 0
    full_action_set_count = 0
    widths: list[int] = []
    per_seed: dict[int, dict[str, int]] = defaultdict(lambda: {"rows": 0, "safe": 0})
    first_public_hits = 0
    best_fixed_hits = 0
    global_top_n_hits = 0
    full_public_hits = 0
    examples = []
    for row_index, row in enumerate(rows):
        group_key = str(group_by_row[row_index]["group_key"])
        train_indexes = [
            index
            for index in range(len(rows))
            if str(group_by_row[index]["group_key"]) != group_key
        ]
        ranked = _ranked_neighbor_indexes(
            row_index=row_index,
            train_indexes=train_indexes,
            vectors=vectors,
            feature_keys=feature_keys,
        )
        candidate = _candidate_action_set_from_ranked(
            row_index=row_index,
            row=row,
            rows=rows,
            ranked=ranked,
            train_indexes=train_indexes,
            top_n=top_n,
            randomized_safe_sets=randomized_safe_sets,
        )
        candidate_actions = set(_list_of_strings(candidate.get("candidate_action_set")))
        safe_set = set(_safe_action_set(row))
        public_mask = _complete_bool_mask(row.get("public_action_mask"))
        legal_actions = _legal_actions(public_mask)
        safe_hit = bool(safe_set & candidate_actions)
        safe_hits += int(safe_hit)
        unsupported = _list_of_strings(candidate.get("unsupported_actions"))
        unsupported_rows += int(bool(unsupported))
        unsupported_actions_total += len(unsupported)
        width = len(candidate_actions)
        widths.append(width)
        full_action_set_count += int(candidate_actions == set(legal_actions) and bool(legal_actions))
        first_public_hits += int(_first_public_set(public_mask) & safe_set != set())
        best_fixed_hits += int(
            _best_fixed_action_set(rows=rows, train_indexes=train_indexes, public_mask=public_mask)
            & safe_set
            != set()
        )
        global_top_n_hits += int(
            _global_top_n_safe_action_set(
                rows=rows,
                train_indexes=train_indexes,
                public_mask=public_mask,
                top_n=top_n,
            )
            & safe_set
            != set()
        )
        full_public_hits += int(set(legal_actions) & safe_set != set())
        if (
            row.get("schema_version")
            == M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ):
            seed = _int(group_by_row[row_index].get("source_seed"))
            per_seed[seed]["rows"] += 1
            per_seed[seed]["safe"] += int(safe_hit)
        if len(examples) < 16:
            examples.append(
                {
                    "row_index": row_index,
                    "source_seed": group_by_row[row_index].get("source_seed"),
                    "top_n": int(top_n),
                    "neighbor_row_indexes": candidate.get("neighbor_row_indexes"),
                    "candidate_action_set": sorted(candidate_actions, key=_action_order),
                    "safe_action_set": sorted(safe_set, key=_action_order),
                    "set_safe_hit": safe_hit,
                    "unsupported_actions": unsupported,
                }
            )
    row_count = len(rows)
    set_safe_rate = _safe_rate(safe_hits, row_count)
    best_narrow_trivial = max(
        _safe_rate(first_public_hits, row_count),
        _safe_rate(best_fixed_hits, row_count),
        _safe_rate(global_top_n_hits, row_count),
    )
    zero_seeds = sorted(
        seed
        for seed, payload in per_seed.items()
        if payload["rows"] and payload["safe"] == 0
    )
    average_width = _round(sum(widths) / len(widths)) if widths else 0.0
    return {
        "policy": "m3_carrion_survivor_continuation_v170_source_split_top_n_set_eval_v1",
        "feature_family": family_name,
        "top_n": int(top_n),
        "randomized_safe_set_control": bool(randomized_safe_sets),
        "row_count": row_count,
        "set_safe_hit_count": safe_hits,
        "set_safe_hit_rate": set_safe_rate,
        "average_set_width": average_width,
        "max_set_width": max(widths, default=0),
        "full_action_set_count": full_action_set_count,
        "full_action_set_share": _safe_rate(full_action_set_count, row_count),
        "unsupported_prediction_row_count": unsupported_rows,
        "unsupported_action_count": unsupported_actions_total,
        "zero_safe_hit_preterminal_source_seeds": zero_seeds,
        "per_source_seed_set_safe_hit_rates": {
            str(seed): {
                "row_count": payload["rows"],
                "set_safe_hit_count": payload["safe"],
                "set_safe_hit_rate": _safe_rate(payload["safe"], payload["rows"]),
            }
            for seed, payload in sorted(per_seed.items())
        },
        "trivial_set_baselines": {
            "first_public_singleton_hit_rate": _safe_rate(first_public_hits, row_count),
            "best_fixed_singleton_hit_rate": _safe_rate(best_fixed_hits, row_count),
            "global_top_n_safe_action_set_hit_rate": _safe_rate(
                global_top_n_hits,
                row_count,
            ),
            "full_public_action_set_hit_rate": _safe_rate(full_public_hits, row_count),
            "best_narrow_trivial_set_hit_rate": best_narrow_trivial,
        },
        "set_safe_hit_margin_over_best_narrow_trivial": _round(
            set_safe_rate - best_narrow_trivial
        ),
        "set_safe_hit_margin_over_full_public_trivial": _round(
            set_safe_rate - _safe_rate(full_public_hits, row_count)
        ),
        "sets_are_broad": average_width > 2.5 or full_action_set_count > 0,
        "examples": examples,
    }


def _candidate_action_set_from_ranked(
    *,
    row_index: int,
    row: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    ranked: Sequence[int],
    train_indexes: Sequence[int],
    top_n: int,
    randomized_safe_sets: bool,
) -> dict[str, object]:
    public_mask = _complete_bool_mask(row.get("public_action_mask"))
    raw_actions: set[str] = set()
    source_indexes = list(ranked[:top_n])
    if randomized_safe_sets and train_indexes:
        source_indexes = [
            train_indexes[(row_index + offset + int(top_n)) % len(train_indexes)]
            for offset, _index in enumerate(source_indexes)
        ]
    for index in source_indexes:
        raw_actions.update(_safe_action_set(rows[index]))
    unsupported = sorted(
        [action for action in raw_actions if public_mask.get(action) is not True],
        key=_action_order,
    )
    candidate_actions = sorted(
        [action for action in raw_actions if public_mask.get(action) is True],
        key=_action_order,
    )
    return {
        "neighbor_row_indexes": list(ranked[:top_n]),
        "safe_set_source_row_indexes": source_indexes,
        "raw_neighbor_safe_action_set": sorted(raw_actions, key=_action_order),
        "candidate_action_set": candidate_actions,
        "unsupported_actions": unsupported,
    }


def _ranked_neighbor_indexes(
    *,
    row_index: int,
    train_indexes: Sequence[int],
    vectors: Sequence[Mapping[str, float]],
    feature_keys: Sequence[str],
) -> list[int]:
    return [
        index
        for _distance, index in sorted(
            (
                (
                    _vector_distance(vectors[row_index], vectors[index], feature_keys),
                    index,
                )
                for index in train_indexes
            ),
            key=lambda item: (item[0], item[1]),
        )
    ]


def _set_lane_negative_control(
    *,
    actual: Mapping[str, object],
    control: Mapping[str, object],
    baseline_zero: Sequence[int],
) -> dict[str, object]:
    actual_zero = set(_list_of_ints(actual.get("zero_safe_hit_preterminal_source_seeds")))
    control_zero = set(_list_of_ints(control.get("zero_safe_hit_preterminal_source_seeds")))
    baseline = set(int(seed) for seed in baseline_zero)
    actual_improves = len(actual_zero) < len(baseline) and not (actual_zero - baseline)
    suspicious = (
        actual_improves
        and _float(control.get("set_safe_hit_rate")) >= _float(actual.get("set_safe_hit_rate"))
        and len(control_zero) <= len(actual_zero)
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v170_set_lane_randomized_safe_set_control_v1",
        "passed": not suspicious,
        "suspicious_control": suspicious,
        "control_set_safe_hit_rate": control.get("set_safe_hit_rate"),
        "actual_set_safe_hit_rate": actual.get("set_safe_hit_rate"),
        "control_zero_safe_hit_preterminal_source_seeds": sorted(control_zero),
        "actual_zero_safe_hit_preterminal_source_seeds": sorted(actual_zero),
        "control_average_set_width": control.get("average_set_width"),
        "actual_average_set_width": actual.get("average_set_width"),
    }


def _sequence_features_current(row: Mapping[str, object]) -> dict[str, object]:
    features = _mapping(_mapping(row.get("trainable")).get("features"))
    return {
        "observation_input": deepcopy(_mapping(features.get("observation_input"))),
        "action_mask": _complete_bool_mask(features.get("action_mask")),
    }


def _sequence_features_with_prior_summary(row: Mapping[str, object]) -> dict[str, object]:
    features = _mapping(_mapping(row.get("trainable")).get("features"))
    prior = _list_of_mappings(features.get("prior_public_context"))
    return {
        **_sequence_features_current(row),
        "legal_prior_public_sequence_summary": {
            "available": bool(prior),
            "count_capped": min(len(prior), 5),
            "items": [_sanitize_prior_public_summary(item) for item in prior[:5]],
        },
    }


def _sanitize_prior_public_summary(item: Mapping[str, object]) -> dict[str, object]:
    sanitized = {}
    for key, value in sorted(item.items()):
        key_text = str(key)
        if _contains_forbidden_token(key_text, FORBIDDEN_SEQUENCE_PAYLOAD_TOKENS):
            continue
        if key_text == "action_mask":
            sanitized[key_text] = _complete_bool_mask(value)
        elif isinstance(value, Mapping):
            nested = _sanitize_prior_public_summary(value)
            if nested:
                sanitized[key_text] = nested
        elif isinstance(value, list):
            sanitized[key_text] = [
                _sanitize_prior_public_summary(entry)
                for entry in _list_of_mappings(value)
            ]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key_text] = value
    return sanitized


def _sequence_alias_report(
    feature_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in feature_rows:
        groups[stable_payload_digest(_mapping(row.get("features")))].append(row)
    conflicts = []
    for digest, items in sorted(groups.items()):
        counts = Counter(str(item.get("label_action") or "") for item in items)
        if len(counts) <= 1:
            continue
        conflicts.append(
            {
                "feature_digest": digest,
                "row_count": len(items),
                "label_action_counts": dict(sorted(counts.items())),
                "row_indexes_sample": [
                    _int(item.get("row_index")) for item in items[:12]
                ],
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v170_sequence_alias_report_v1",
        "exact_feature_group_count": len(groups),
        "conflicting_group_count": len(conflicts),
        "conflicting_row_count": sum(int(item["row_count"]) for item in conflicts),
        "max_rows_per_feature_group": max((len(items) for items in groups.values()), default=0),
        "max_label_action_count_per_feature_group": max(
            (
                len(Counter(str(item.get("label_action") or "") for item in items))
                for items in groups.values()
            ),
            default=0,
        ),
        "conflicting_examples": conflicts[:8],
    }


def _sequence_alias_reference_comparison(
    *,
    alias: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict[str, object]:
    reference_rows = _int(reference.get("conflicting_exact_feature_row_count"))
    reference_groups = _int(reference.get("conflicting_exact_feature_group_count"))
    conflict_rows = _int(alias.get("conflicting_row_count"))
    conflict_groups = _int(alias.get("conflicting_group_count"))
    material_threshold = reference_rows // 2
    return {
        "policy": "m3_carrion_survivor_continuation_v170_sequence_alias_reference_comparison_v1",
        "reference_exact_feature_group_count": reference.get("exact_feature_group_count"),
        "candidate_exact_feature_group_count": alias.get("exact_feature_group_count"),
        "reference_conflicting_group_count": reference_groups,
        "candidate_conflicting_group_count": conflict_groups,
        "reference_conflicting_row_count": reference_rows,
        "candidate_conflicting_row_count": conflict_rows,
        "conflicting_group_count_delta_vs_v155": conflict_groups - reference_groups,
        "conflicting_row_count_delta_vs_v155": conflict_rows - reference_rows,
        "conflicting_row_reduction_vs_v155": reference_rows - conflict_rows,
        "material_conflicting_row_threshold": material_threshold,
        "material_alias_conflict_reduction": (
            conflict_rows <= material_threshold and conflict_groups < reference_groups
        ),
    }


def _sequence_negative_control(
    *,
    alias: Mapping[str, object],
    control_alias: Mapping[str, object],
    comparison: Mapping[str, object],
) -> dict[str, object]:
    material = comparison.get("material_alias_conflict_reduction") is True
    suspicious = (
        material
        and _int(control_alias.get("conflicting_row_count")) <= _int(alias.get("conflicting_row_count"))
        and _int(control_alias.get("conflicting_group_count")) <= _int(alias.get("conflicting_group_count"))
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v170_sequence_shuffled_label_control_v1",
        "passed": not suspicious,
        "suspicious_control": suspicious,
        "control_conflicting_group_count": control_alias.get("conflicting_group_count"),
        "control_conflicting_row_count": control_alias.get("conflicting_row_count"),
        "actual_conflicting_group_count": alias.get("conflicting_group_count"),
        "actual_conflicting_row_count": alias.get("conflicting_row_count"),
    }


def _shuffled_label_rows(
    feature_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    labels = [str(row.get("label_action") or "") for row in feature_rows]
    if not labels:
        return []
    offset = 7 % len(labels)
    return [
        {
            **dict(row),
            "label_action": labels[(index + offset) % len(labels)],
        }
        for index, row in enumerate(feature_rows)
    ]


def _scan_sequence_payload(
    *,
    value: object,
    row_index: int,
    path: tuple[str, ...],
    failures: list[dict[str, object]],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current_path = (*path, key_text)
            if _contains_forbidden_token(key_text, FORBIDDEN_SEQUENCE_PAYLOAD_TOKENS):
                failures.append(
                    {
                        "row_index": row_index,
                        "path": ".".join(current_path),
                        "reason": "forbidden_key_token",
                        "token": _matching_forbidden_token(
                            key_text,
                            FORBIDDEN_SEQUENCE_PAYLOAD_TOKENS,
                        ),
                    }
                )
            _scan_sequence_payload(
                value=item,
                row_index=row_index,
                path=current_path,
                failures=failures,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_sequence_payload(
                value=item,
                row_index=row_index,
                path=(*path, str(index)),
                failures=failures,
            )
    elif isinstance(value, str) and _contains_forbidden_token(
        value,
        FORBIDDEN_SEQUENCE_PAYLOAD_TOKENS,
    ):
        failures.append(
            {
                "row_index": row_index,
                "path": ".".join(path),
                "reason": "forbidden_string_token",
                "token": _matching_forbidden_token(
                    value,
                    FORBIDDEN_SEQUENCE_PAYLOAD_TOKENS,
                ),
            }
        )


def _source_window_row(
    failure: Mapping[str, object],
    *,
    row_by_index: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    row_index = _int(failure.get("row_index"))
    row = _mapping(row_by_index.get(row_index))
    metadata = _mapping(row.get("metadata"))
    return {
        "row_index": row_index,
        "source_seed": _int(failure.get("source_seed")),
        "source_tick": _int(metadata.get("branch_tick")),
        "safe_action_set": _safe_action_set(row),
        "predicted_action": failure.get("predicted_action"),
        "failure_mode": failure.get("failure_mode"),
    }


def _branch_windows_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    radius: int = 5,
) -> list[dict[str, object]]:
    windows = []
    seen = set()
    for row in sorted(rows, key=lambda item: (_int(item.get("source_tick")), _int(item.get("row_index")))):
        tick = _int(row.get("source_tick"))
        if tick < 0:
            continue
        key = (tick, _int(row.get("row_index")))
        if key in seen:
            continue
        seen.add(key)
        windows.append(
            {
                "source_tick": tick,
                "start_tick": max(0, tick - int(radius)),
                "end_tick": tick + int(radius),
                "source_row_index": row.get("row_index"),
                "safe_action_set": row.get("safe_action_set"),
                "window_radius": int(radius),
                "failed_row_source": bool(row.get("failure_mode")),
            }
        )
    return windows


def _expected_v171_command_shape(
    *,
    seed: int,
    shard_id: str,
    windows: Sequence[Mapping[str, object]],
) -> list[str]:
    window_args: list[str] = []
    for window in windows:
        window_args.extend(
            [
                "--branch-window",
                f"{_int(window.get('start_tick'))}:{_int(window.get('end_tick'))}",
            ]
        )
    return [
        "npm",
        "run",
        "sim:mind:v3:carrion-survivor-continuation-v171-replay-expansion",
        "--",
        "--shard-id",
        shard_id,
        "--seed-include",
        str(int(seed)),
        *window_args,
        "--output",
        f"output/mind/shards/{shard_id}.json",
        "--dataset-output",
        f"output/mind/shards/{shard_id}.jsonl",
        "--fail-on-partial-shard",
    ]


def _shard_seed_rationale(
    seed: int,
    *,
    v169_best: Mapping[str, object],
) -> list[str]:
    rationale = []
    if int(seed) in BASELINE_ZERO_HIT_SEEDS:
        rationale.append("baseline_v167_v169_zero_hit_seed")
    if int(seed) == REGRESSION_NEW_ZERO_SEED:
        rationale.append("v169_best_candidate_new_zero_or_regression_seed")
    if int(seed) == LOW_SUPPORT_EDGE_SEED:
        rationale.append("low_support_edge_seed_preserve_nonzero_coverage")
    if int(seed) in _list_of_ints(v169_best.get("new_zero_hit_seeds_introduced")):
        rationale.append("observed_v169_new_zero_hit_seed")
    return rationale


def _classification(
    *,
    source_validation: Mapping[str, object],
    set_lane: Mapping[str, object],
    sequence_lane: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix_"
    if source_validation.get("passed") is not True:
        return prefix + "closed_invalid"
    if set_lane.get("lane_valid") is not True or sequence_lane.get("lane_valid") is not True:
        return prefix + "closed_invalid"
    if sequence_lane.get("material_alias_conflict_reduction") is True:
        return prefix + "sequence_memory_alias_reduction_ready_for_dataset_design_no_training"
    if set_lane.get("zero_hit_coverage_improved") is True:
        return prefix + "set_valued_ranking_support_partial_no_runtime"
    return prefix + "portfolio_recommends_parallel_archive_source_expansion_shards_no_training"


def _route_recommendation(classification: str) -> dict[str, object]:
    sequence_ready = classification.endswith(
        "sequence_memory_alias_reduction_ready_for_dataset_design_no_training"
    )
    set_partial = classification.endswith("set_valued_ranking_support_partial_no_runtime")
    invalid = classification.endswith("closed_invalid")
    return {
        "policy": "m3_carrion_survivor_continuation_v170_route_recommendation_v1",
        "recommended_next_route": (
            "repair_source_or_control_failure"
            if invalid
            else "sequence_memory_dataset_design_no_training"
            if sequence_ready
            else "parallel_archive_source_expansion_shards"
            if set_partial
            else "parallel_archive_source_expansion_shards"
        ),
        "parallel_archive_source_expansion_shards_recommended": (
            not invalid and not sequence_ready
        ),
        "sequence_memory_dataset_design_recommended": sequence_ready,
        "set_valued_runtime_recommended": False,
        "training_recommended": False,
        "shadow_eval_recommended": False,
        "live_ab_allowed": False,
        "k_tuning_recommended": False,
        "threshold_tuning_recommended": False,
        "runtime_policy_integration_allowed": False,
        "promotion_authorized": False,
        "staging_authorized": False,
        "runtime_action_selection_changed": False,
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "portfolio_matrix_only": True,
        "replay_expansion_shard_plan_only": True,
        "plan_artifact_is_not_evidence": True,
        "training_ran": False,
        "training_authorized": False,
        "shadow_eval_ran": False,
        "live_ab_allowed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "replay_viewer_schema_changed": False,
        "k_tuning_ran": False,
        "threshold_tuning_ran": False,
        "promotion_authorized": False,
        "staging_authorized": False,
        "commit_authorized": False,
        "reset_authorized": False,
        "clean_authorized": False,
        "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
        "uses_private_world_state_as_trainable_input": False,
        "uses_future_outcome_as_trainable_input": False,
        "uses_label_as_trainable_input": False,
    }


def _source_lifecycle_scan(
    *,
    reports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    failures = []
    false_fields = (
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "runtime_observation_schema_changed",
        "runtime_policy_changed",
        "replay_viewer_schema_changed",
        "threshold_tuning_ran",
        "k_tuning_ran",
        "training_ran",
        "shadow_eval_ran",
        "live_ab_allowed",
        "promotion_authorized",
        "training_authorized",
        "runtime_promotion_allowed",
        "default_runtime_behavior_changed",
    )
    for label, report in sorted(reports.items()):
        if report.get("diagnostics_only") is not True:
            failures.append(
                {"report": label, "field": "diagnostics_only", "observed": report.get("diagnostics_only")}
            )
        for field in false_fields:
            if field in report and report.get(field) is not False:
                failures.append(
                    {"report": label, "field": field, "observed": report.get(field)}
                )
    return {
        "policy": "m3_carrion_survivor_continuation_v170_source_lifecycle_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _v155_reference_alias_counts(v155_report: Mapping[str, object]) -> dict[str, object]:
    audit = _mapping(
        _mapping(v155_report.get("pretraining_review")).get(
            "nearest_neighbor_alias_collision_audit"
        )
    )
    exact_groups = _int(audit.get("exact_feature_group_count"))
    conflict_groups = _int(audit.get("conflicting_exact_feature_group_count"))
    conflict_rows = _int(audit.get("conflicting_exact_feature_row_count"))
    return {
        "policy": "m3_carrion_survivor_continuation_v170_v155_reference_alias_counts_v1",
        "exact_feature_group_count": exact_groups,
        "conflicting_exact_feature_group_count": conflict_groups,
        "conflicting_exact_feature_row_count": conflict_rows,
        "expected_exact_feature_group_count": V155_REFERENCE_EXACT_FEATURE_GROUP_COUNT,
        "expected_conflicting_exact_feature_group_count": (
            V155_REFERENCE_CONFLICTING_GROUP_COUNT
        ),
        "expected_conflicting_exact_feature_row_count": V155_REFERENCE_CONFLICTING_ROW_COUNT,
        "matches_required_reference": (
            exact_groups == V155_REFERENCE_EXACT_FEATURE_GROUP_COUNT
            and conflict_groups == V155_REFERENCE_CONFLICTING_GROUP_COUNT
            and conflict_rows == V155_REFERENCE_CONFLICTING_ROW_COUNT
        ),
    }


def _best_set_valued_entry(
    *,
    reports: Sequence[Mapping[str, object]],
    baseline_zero: Sequence[int],
) -> dict[str, object]:
    if not reports:
        return {}
    baseline = set(int(seed) for seed in baseline_zero)
    decorated = []
    for report in reports:
        zero = set(_list_of_ints(report.get("zero_safe_hit_preterminal_source_seeds")))
        comparison = _mapping(report.get("baseline_comparison"))
        improved = len(zero) < len(baseline) and not (zero - baseline)
        decorated.append(
            (
                not improved,
                bool(report.get("sets_are_broad")) is True,
                len(zero),
                -_float(report.get("set_safe_hit_rate")),
                _float(report.get("average_set_width")),
                str(report.get("feature_family")),
                _int(report.get("top_n")),
                report,
                comparison,
            )
        )
    *_, best, comparison = sorted(decorated, key=lambda item: item[:7])[0]
    zero = _list_of_ints(best.get("zero_safe_hit_preterminal_source_seeds"))
    improved = len(set(zero)) < len(baseline) and not (set(zero) - baseline)
    return {
        "feature_family": best.get("feature_family"),
        "top_n": best.get("top_n"),
        "set_safe_hit_rate": best.get("set_safe_hit_rate"),
        "set_safe_hit_margin_over_best_narrow_trivial": best.get(
            "set_safe_hit_margin_over_best_narrow_trivial"
        ),
        "average_set_width": best.get("average_set_width"),
        "full_action_set_share": best.get("full_action_set_share"),
        "unsupported_prediction_row_count": best.get("unsupported_prediction_row_count"),
        "zero_safe_hit_preterminal_source_seeds": zero,
        "zero_hit_coverage_improved": improved,
        "sets_are_broad": bool(best.get("sets_are_broad")),
        "baseline_comparison": comparison,
        "negative_control": best.get("negative_control"),
        "no_runtime_top_n_selected": True,
    }


def _best_sequence_alias_entry(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not reports:
        return {}
    best = sorted(
        reports,
        key=lambda report: (
            not _mapping(report.get("reference_comparison")).get(
                "material_alias_conflict_reduction"
            )
            is True,
            _int(_mapping(report.get("alias_report")).get("conflicting_row_count")),
            _int(_mapping(report.get("alias_report")).get("conflicting_group_count")),
            str(report.get("policy_id")),
        ),
    )[0]
    alias = _mapping(best.get("alias_report"))
    comparison = _mapping(best.get("reference_comparison"))
    return {
        "policy_id": best.get("policy_id"),
        "exact_feature_group_count": alias.get("exact_feature_group_count"),
        "conflicting_group_count": alias.get("conflicting_group_count"),
        "conflicting_row_count": alias.get("conflicting_row_count"),
        "material_alias_conflict_reduction": comparison.get(
            "material_alias_conflict_reduction"
        ),
        "conflicting_row_reduction_vs_v155": comparison.get(
            "conflicting_row_reduction_vs_v155"
        ),
        "leakage_scan": best.get("leakage_scan"),
        "negative_control": best.get("negative_control"),
    }


def _zero_hit_baseline_comparison(
    *,
    baseline_zero: Sequence[int],
    zero_hit_seeds: object,
) -> dict[str, object]:
    baseline = set(int(seed) for seed in baseline_zero)
    zero = set(_list_of_ints(zero_hit_seeds))
    return {
        "baseline_zero_safe_hit_preterminal_source_seeds": sorted(baseline),
        "candidate_zero_safe_hit_preterminal_source_seeds": sorted(zero),
        "zero_safe_hit_seed_count_delta_vs_baseline": len(zero) - len(baseline),
        "zero_safe_hit_seeds_removed_vs_baseline": sorted(baseline - zero),
        "new_zero_hit_seeds_introduced": sorted(zero - baseline),
        "zero_hit_coverage_improved_without_new_zero_seeds": (
            len(zero) < len(baseline) and not (zero - baseline)
        ),
    }


def _baseline_zero_hit_seeds(v169_report: Mapping[str, object]) -> list[int]:
    baseline = _mapping(v169_report.get("v168_baseline_family_recomputed"))
    seeds = _list_of_ints(baseline.get("zero_safe_hit_preterminal_source_seeds"))
    if seeds:
        return seeds
    return list(BASELINE_ZERO_HIT_SEEDS)


def _negative_control_summary(
    *,
    set_lane: Mapping[str, object],
    sequence_lane: Mapping[str, object],
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v170_negative_control_summary_v1",
        "set_valued_ranking_controls_passed": set_lane.get("negative_controls_passed"),
        "sequence_memory_alias_controls_passed": sequence_lane.get(
            "negative_controls_passed"
        ),
        "all_controls_passed": (
            set_lane.get("negative_controls_passed") is True
            and sequence_lane.get("negative_controls_passed") is True
        ),
    }


def _shard_plan_output_summary(
    *,
    shard_plan: Mapping[str, object],
    shard_plan_output_path: str | Path | None,
) -> dict[str, object]:
    shards = _list_of_mappings(shard_plan.get("shards"))
    return {
        "policy": "m3_carrion_survivor_continuation_v170_shard_plan_output_v1",
        "path": str(shard_plan_output_path) if shard_plan_output_path is not None else None,
        "written": shard_plan_output_path is not None,
        "jsonl_row_count": len(shards) if shard_plan_output_path is not None else 0,
        "shard_plan_digest": stable_payload_digest(shards),
    }


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(dict(row), handle, sort_keys=True, allow_nan=False)
            handle.write("\n")


def _legal_actions(public_mask: Mapping[str, bool]) -> list[str]:
    return [action for action in ACTION_NAMES if public_mask.get(action) is True]


def _first_public_set(public_mask: Mapping[str, bool]) -> set[str]:
    for action in ACTION_NAMES:
        if public_mask.get(action) is True:
            return {action}
    return set()


def _best_fixed_action_set(
    *,
    rows: Sequence[Mapping[str, object]],
    train_indexes: Sequence[int],
    public_mask: Mapping[str, bool],
) -> set[str]:
    counts: Counter[str] = Counter()
    for index in train_indexes:
        counts.update(
            action
            for action in _safe_action_set(rows[index])
            if public_mask.get(action) is True
        )
    action = _majority_action(counts)
    return {action} if action else set()


def _global_top_n_safe_action_set(
    *,
    rows: Sequence[Mapping[str, object]],
    train_indexes: Sequence[int],
    public_mask: Mapping[str, bool],
    top_n: int,
) -> set[str]:
    counts: Counter[str] = Counter()
    for index in train_indexes:
        counts.update(
            action
            for action in _safe_action_set(rows[index])
            if public_mask.get(action) is True
        )
    return {
        action
        for action, _count in sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), _action_order(str(item[0]))),
        )[: int(top_n)]
    }


def _majority_action(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), _action_order(str(item[0]))),
    )[0][0]


def _label_action(row: Mapping[str, object]) -> str:
    return str(_mapping(_mapping(row.get("trainable")).get("label")).get("action", ""))


def _list_of_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _list_of_ints(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    return sorted({_int(item) for item in value})


def _contains_forbidden_token(value: str, tokens: Sequence[str]) -> bool:
    lower = value.lower()
    return any(token in lower for token in tokens)


def _matching_forbidden_token(value: str, tokens: Sequence[str]) -> str:
    lower = value.lower()
    for token in tokens:
        if token in lower:
            return token
    return ""
