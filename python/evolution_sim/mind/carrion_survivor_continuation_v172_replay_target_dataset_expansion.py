from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    _dominant_count_share,
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    DEFAULT_DATASET_OUTPUT_PATH as DEFAULT_V171_DATASET_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V171_REPORT_PATH,
    FORBIDDEN_TRAINABLE_PAYLOAD_TOKENS,
    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_DATASET_FEATURE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_ROW_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION,
    load_jsonl_dataset,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v172_replay_target_dataset_expansion_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v172_replay_target_dataset_expansion_row_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v172_replay_target_dataset_expansion_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY = (
    "public_observation_action_mask_v172_replay_target_dataset_v1"
)
EXPECTED_V171_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v171_replay_expansion_"
    "replay_expansion_support_ready_for_v172_target_dataset_expansion_no_training"
)
EXPECTED_V171_EXACT_DIGEST = (
    "80d46c17927cdf33ebb7ea23d2e5d24d1bcf7d560ac856ae790f8d456a67904e"
)
EXPECTED_V171_DATASET_DIGEST = (
    "4050d8c0642175baa99d9ecd0df17eabab7ee6538329d8be6b3f286628b60532"
)
EXPECTED_V171_REPLAY_VERIFIED_COUNT = 4730
EXPECTED_V171_DATASET_ROW_COUNT = 910
EXPECTED_V171_AVERAGE_REPLAY_SAFE_SET_WIDTH = 1.763736
V170_BEST_SET_WIDTH = 3.391304
SUPPORT_PROVENANCE_SEEDS = (5, 13, 19, 29, 37, 41)
DEFAULT_MAX_DOMINANT_SUPPORT_ACTION_SHARE = 0.75
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v172-carrion-survivor-continuation-replay-target-dataset-expansion.json"
)
DEFAULT_TARGET_DATASET_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v172-carrion-survivor-continuation-replay-target-dataset-expansion.jsonl"
)


class CarrionSurvivorContinuationV172ReplayTargetDatasetExpansionError(
    ValueError
):
    pass


def run_carrion_survivor_continuation_v172_replay_target_dataset_expansion(
    *,
    v171_report_path: str | Path = DEFAULT_V171_REPORT_PATH,
    v171_dataset_path: str | Path = DEFAULT_V171_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    target_dataset_output_path: str | Path = DEFAULT_TARGET_DATASET_OUTPUT_PATH,
    expected_v171_exact_digest: str | None = EXPECTED_V171_EXACT_DIGEST,
    expected_v171_dataset_digest: str | None = EXPECTED_V171_DATASET_DIGEST,
    expected_v171_classification: str = EXPECTED_V171_CLASSIFICATION,
    expected_v171_replay_verified_count: int = EXPECTED_V171_REPLAY_VERIFIED_COUNT,
    expected_v171_dataset_row_count: int = EXPECTED_V171_DATASET_ROW_COUNT,
    expected_v171_average_replay_safe_set_width: float = (
        EXPECTED_V171_AVERAGE_REPLAY_SAFE_SET_WIDTH
    ),
    v170_best_set_width: float = V170_BEST_SET_WIDTH,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
    max_dominant_support_action_share: float = (
        DEFAULT_MAX_DOMINANT_SUPPORT_ACTION_SHARE
    ),
) -> dict[str, object]:
    v171_report = load_json_report(v171_report_path)
    v171_rows = load_jsonl_dataset(v171_dataset_path)
    source_validation = validate_v172_sources(
        v171_report=v171_report,
        v171_rows=v171_rows,
        expected_v171_exact_digest=expected_v171_exact_digest,
        expected_v171_dataset_digest=expected_v171_dataset_digest,
        expected_v171_classification=expected_v171_classification,
        expected_v171_replay_verified_count=expected_v171_replay_verified_count,
        expected_v171_dataset_row_count=expected_v171_dataset_row_count,
        expected_v171_average_replay_safe_set_width=(
            expected_v171_average_replay_safe_set_width
        ),
        v170_best_set_width=v170_best_set_width,
    )
    target_rows: list[dict[str, object]] = []
    row_build_validation = _skipped_row_build_validation("source_validation_failed")
    if source_validation.get("passed") is True:
        target_rows, row_build_validation = build_v172_target_rows(
            v171_rows,
            v171_report_exact_digest=str(v171_report.get("exact_digest") or ""),
            v171_dataset_digest=stable_payload_digest(v171_rows),
            support_provenance_seeds=support_provenance_seeds,
        )
    leakage_scan = trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in target_rows]
    )
    row_schema_validation = validate_v172_target_rows(
        target_rows,
        support_provenance_seeds=support_provenance_seeds,
    )
    target_summary = target_dataset_summary(
        target_rows,
        v170_best_set_width=v170_best_set_width,
        max_dominant_support_action_share=max_dominant_support_action_share,
        support_provenance_seeds=support_provenance_seeds,
    )
    source_split_plan = source_split_evaluation_plan_for_v173(
        support_provenance_seeds=support_provenance_seeds,
    )
    classification = _classification(
        source_validation=source_validation,
        row_build_validation=row_build_validation,
        leakage_scan=leakage_scan,
        row_schema_validation=row_schema_validation,
        target_summary=target_summary,
    )
    dataset_digest = stable_payload_digest(target_rows)
    _write_jsonl(target_dataset_output_path, target_rows)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v171_report": str(v171_report_path),
            "v171_dataset": str(v171_dataset_path),
            "expected_v171_classification": expected_v171_classification,
            "expected_v171_exact_digest": expected_v171_exact_digest,
            "expected_v171_dataset_digest": expected_v171_dataset_digest,
            "expected_v171_replay_verified_count": int(
                expected_v171_replay_verified_count
            ),
            "expected_v171_dataset_row_count": int(
                expected_v171_dataset_row_count
            ),
            "expected_v171_average_replay_safe_set_width": _round(
                expected_v171_average_replay_safe_set_width
            ),
            "v170_best_set_width": _round(v170_best_set_width),
            "target_dataset_output": str(target_dataset_output_path),
            "support_provenance_seeds": [
                int(seed) for seed in support_provenance_seeds
            ],
            "max_dominant_support_action_share": _round(
                max_dominant_support_action_share
            ),
        },
        "source_validation": source_validation,
        "row_build_validation": row_build_validation,
        "leakage_scan": leakage_scan,
        "row_schema_validation": row_schema_validation,
        "dataset": {
            "path": str(target_dataset_output_path),
            "row_count": len(target_rows),
            "row_schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
            ),
            "feature_policy_id": (
                M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY
            ),
            "source_v171_row_schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_ROW_SCHEMA_VERSION
            ),
            "source_v171_feature_policy_id": (
                M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_DATASET_FEATURE_POLICY
            ),
            "dataset_digest": dataset_digest,
        },
        "target_dataset_summary": target_summary,
        "action_support_distribution": target_summary.get(
            "action_support_distribution"
        ),
        "safe_set_width_distribution": target_summary.get(
            "safe_set_width_distribution"
        ),
        "unique_action_vs_tied_action_row_counts": target_summary.get(
            "unique_action_vs_tied_action_row_counts"
        ),
        "per_seed_support_summary": target_summary.get("per_seed_support_summary"),
        "source_split_evaluation_plan_for_v173": source_split_plan,
        "promotion_heldout_warning": promotion_heldout_warning(
            support_provenance_seeds=support_provenance_seeds,
        ),
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v172_sources(
    *,
    v171_report: Mapping[str, object],
    v171_rows: Sequence[Mapping[str, object]],
    expected_v171_exact_digest: str | None = EXPECTED_V171_EXACT_DIGEST,
    expected_v171_dataset_digest: str | None = EXPECTED_V171_DATASET_DIGEST,
    expected_v171_classification: str = EXPECTED_V171_CLASSIFICATION,
    expected_v171_replay_verified_count: int = EXPECTED_V171_REPLAY_VERIFIED_COUNT,
    expected_v171_dataset_row_count: int = EXPECTED_V171_DATASET_ROW_COUNT,
    expected_v171_average_replay_safe_set_width: float = (
        EXPECTED_V171_AVERAGE_REPLAY_SAFE_SET_WIDTH
    ),
    v170_best_set_width: float = V170_BEST_SET_WIDTH,
) -> dict[str, object]:
    failures: list[str] = []
    observed_classification = str(
        _mapping(v171_report.get("classification")).get("primary") or ""
    )
    if (
        v171_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_SCHEMA_VERSION
    ):
        failures.append("v171_schema_version_mismatch")
    if (
        v171_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_POLICY
    ):
        failures.append("v171_policy_mismatch")
    if observed_classification != expected_v171_classification:
        failures.append("v171_unexpected_classification")
    exact = exact_digest_validation_report(v171_report)
    observed_exact = str(v171_report.get("exact_digest") or "")
    if exact.get("passed") is not True:
        failures.append("v171_exact_digest_mismatch")
    if expected_v171_exact_digest and observed_exact != expected_v171_exact_digest:
        failures.append("v171_unexpected_exact_digest")
    dataset_digest = stable_payload_digest(v171_rows)
    reported_dataset = _mapping(v171_report.get("dataset"))
    reported_dataset_digest = str(reported_dataset.get("dataset_digest") or "")
    if expected_v171_dataset_digest and dataset_digest != expected_v171_dataset_digest:
        failures.append("v171_unexpected_dataset_digest")
    if (
        expected_v171_dataset_digest
        and reported_dataset_digest != expected_v171_dataset_digest
    ):
        failures.append("v171_reported_unexpected_dataset_digest")
    if reported_dataset_digest != dataset_digest:
        failures.append("v171_dataset_digest_mismatch")
    if len(v171_rows) != int(expected_v171_dataset_row_count):
        failures.append("v171_dataset_row_count_mismatch")
    if _int(reported_dataset.get("row_count")) != len(v171_rows):
        failures.append("v171_reported_dataset_row_count_mismatch")
    metrics = _mapping(v171_report.get("metrics"))
    replay_verified_count = _int(metrics.get("replay_verified_run_count"))
    candidate_run_count = _int(metrics.get("candidate_run_count"))
    if replay_verified_count != int(expected_v171_replay_verified_count):
        failures.append("v171_replay_verified_count_mismatch")
    if candidate_run_count != int(expected_v171_replay_verified_count):
        failures.append("v171_candidate_run_count_mismatch")
    if metrics.get("all_replays_verified") is not True:
        failures.append("v171_replays_not_all_verified")
    if metrics.get("ready_support_criteria_passed") is not True:
        failures.append("v171_ready_support_criteria_not_passed")
    if _round(metrics.get("average_replay_safe_set_width")) != _round(
        expected_v171_average_replay_safe_set_width
    ):
        failures.append("v171_average_replay_safe_set_width_mismatch")
    observed_v170_width = _round(metrics.get("v170_best_set_average_width"))
    if observed_v170_width != _round(v170_best_set_width):
        failures.append("v171_v170_width_mismatch")
    if (
        _round(metrics.get("average_replay_safe_set_width"))
        >= _round(v170_best_set_width)
    ):
        failures.append("v171_safe_set_width_not_improved_vs_v170")
    partial = _mapping(v171_report.get("partial_shard_status"))
    if partial.get("partial") is not False:
        failures.append("v171_partial_shard_status_not_complete")
    route = _mapping(v171_report.get("route_recommendation"))
    if route.get("v172_target_dataset_expansion_recommended") is not True:
        failures.append("v171_v172_route_not_recommended")
    leakage = _mapping(v171_report.get("leakage_scan"))
    if leakage.get("passed") is not True:
        failures.append("v171_leakage_scan_failed")
    lifecycle = _v171_lifecycle_validation(v171_report)
    if lifecycle.get("passed") is not True:
        failures.append("v171_lifecycle_not_diagnostics_only")
    row_source_scan = validate_v171_source_rows(v171_rows)
    if row_source_scan.get("passed") is not True:
        failures.append("v171_dataset_rows_invalid")
    return {
        "policy": "m3_carrion_survivor_continuation_v172_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v171_classification": expected_v171_classification,
        "observed_v171_classification": observed_classification,
        "expected_v171_exact_digest": expected_v171_exact_digest,
        "observed_v171_exact_digest": observed_exact,
        "v171_exact_digest_validation": exact,
        "expected_v171_dataset_digest": expected_v171_dataset_digest,
        "observed_v171_dataset_digest": dataset_digest,
        "v171_reported_dataset_digest": reported_dataset_digest,
        "expected_v171_replay_verified_count": int(
            expected_v171_replay_verified_count
        ),
        "observed_v171_replay_verified_count": replay_verified_count,
        "observed_v171_candidate_run_count": candidate_run_count,
        "expected_v171_dataset_row_count": int(expected_v171_dataset_row_count),
        "observed_v171_dataset_row_count": len(v171_rows),
        "expected_v171_average_replay_safe_set_width": _round(
            expected_v171_average_replay_safe_set_width
        ),
        "observed_v171_average_replay_safe_set_width": _round(
            metrics.get("average_replay_safe_set_width")
        ),
        "v170_best_set_width": _round(v170_best_set_width),
        "width_improved_vs_v170": (
            _round(metrics.get("average_replay_safe_set_width"))
            < _round(v170_best_set_width)
        ),
        "v171_leakage_scan_passed": leakage.get("passed"),
        "v171_partial_status": partial,
        "v171_lifecycle_validation": lifecycle,
        "v171_source_row_validation": row_source_scan,
    }


def validate_v171_source_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    schema_counts: Counter[str] = Counter()
    feature_policy_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        schema = str(row.get("schema_version") or "")
        feature_policy = str(row.get("feature_policy_id") or "")
        schema_counts.update([schema])
        feature_policy_counts.update([feature_policy])
        if (
            schema
            != M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_ROW_SCHEMA_VERSION
        ):
            _append_row_failure(failures, row_index, "v171_row_schema_mismatch")
        if (
            feature_policy
            != M3_CARRION_SURVIVOR_CONTINUATION_V171_REPLAY_EXPANSION_DATASET_FEATURE_POLICY
        ):
            _append_row_failure(failures, row_index, "v171_feature_policy_mismatch")
        features = _mapping(row.get("trainable_public_features"))
        if set(features) != {"public_observation", "action_mask"}:
            _append_row_failure(
                failures,
                row_index,
                "v171_trainable_feature_key_set_mismatch",
            )
        if not _mapping(features.get("public_observation")):
            _append_row_failure(
                failures,
                row_index,
                "v171_public_observation_missing",
            )
        if _complete_action_mask(_mapping(features.get("action_mask"))) != (
            _complete_action_mask(_mapping(row.get("public_action_mask")))
        ):
            _append_row_failure(
                failures,
                row_index,
                "v171_feature_mask_public_mask_mismatch",
            )
        if not _ordered_action_set(row.get("safe_action_set")):
            _append_row_failure(failures, row_index, "v171_safe_action_set_empty")
    return {
        "policy": "m3_carrion_survivor_continuation_v172_v171_source_row_validation_v1",
        "passed": not failures and bool(rows),
        "failure_count": len(failures),
        "failures": failures[:96],
        "row_count": len(rows),
        "schema_counts": dict(sorted(schema_counts.items())),
        "feature_policy_counts": dict(sorted(feature_policy_counts.items())),
    }


def build_v172_target_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    v171_report_exact_digest: str,
    v171_dataset_digest: str,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    target_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    support_seeds = {int(seed) for seed in support_provenance_seeds}
    for row_index, row in enumerate(rows):
        safe_set = _ordered_action_set(row.get("safe_action_set"))
        targets = _normalized_action_value_targets(
            row,
            safe_action_set=safe_set,
        )
        metadata = _mapping(row.get("metadata"))
        seed = _int(metadata.get("seed"), default=-1)
        if seed not in support_seeds:
            _append_row_failure(
                failures,
                row_index,
                "source_seed_not_in_support_provenance_seeds",
                seed=seed,
            )
        classification = (
            "unique_replay_verified_best_action"
            if len(safe_set) == 1
            else "tied_replay_verified_best_action_set"
            if len(safe_set) > 1
            else "unresolved_replay_target"
        )
        target_row: dict[str, object] = {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
            ),
            "row_origin": "v172_from_v171_replay_verified_target_row",
            "feature_policy_id": (
                M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY
            ),
            "trainable_public_features": deepcopy(
                dict(_mapping(row.get("trainable_public_features")))
            ),
            "public_action_mask": _complete_action_mask(
                _mapping(row.get("public_action_mask"))
            ),
            "action_value_targets": targets,
            "target_best_outcome_action_set": safe_set,
            "safe_action_set": safe_set,
            "target_outcome_summary": _target_outcome_summary(
                targets,
                safe_action_set=safe_set,
            ),
            "target_classification": classification,
            "metadata": _v172_metadata(
                source_metadata=metadata,
                source_row=row,
                seed=seed,
                v171_report_exact_digest=v171_report_exact_digest,
                v171_dataset_digest=v171_dataset_digest,
            ),
        }
        if len(safe_set) == 1:
            target_row["robust_winner_action"] = safe_set[0]
        target_rows.append(target_row)
    return (
        target_rows,
        {
            "policy": "m3_carrion_survivor_continuation_v172_row_build_validation_v1",
            "passed": not failures and len(target_rows) == len(rows),
            "failure_count": len(failures),
            "failures": failures[:96],
            "source_v171_row_count": len(rows),
            "target_row_count": len(target_rows),
        },
    )


def validate_v172_target_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    schema_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()
    source_seed_counts: Counter[int] = Counter()
    support_seed_set = {int(seed) for seed in support_provenance_seeds}
    for row_index, row in enumerate(rows):
        schema = str(row.get("schema_version") or "")
        classification = str(row.get("target_classification") or "")
        schema_counts.update([schema])
        classification_counts.update([classification])
        if (
            schema
            != M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ):
            _append_row_failure(failures, row_index, "row_schema_version_mismatch")
        if (
            str(row.get("feature_policy_id") or "")
            != M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY
        ):
            _append_row_failure(failures, row_index, "feature_policy_mismatch")
        features = _mapping(row.get("trainable_public_features"))
        if set(features) != {"public_observation", "action_mask"}:
            _append_row_failure(
                failures,
                row_index,
                "trainable_feature_key_set_mismatch",
            )
        if not _mapping(features.get("public_observation")):
            _append_row_failure(failures, row_index, "public_observation_missing")
        mask = _complete_action_mask(_mapping(row.get("public_action_mask")))
        feature_mask = _complete_action_mask(_mapping(features.get("action_mask")))
        if mask != feature_mask:
            _append_row_failure(failures, row_index, "feature_action_mask_mismatch")
        safe_set = _ordered_action_set(row.get("safe_action_set"))
        best_set = _ordered_action_set(row.get("target_best_outcome_action_set"))
        if not safe_set:
            _append_row_failure(failures, row_index, "safe_action_set_empty")
        if safe_set != best_set:
            _append_row_failure(
                failures,
                row_index,
                "target_best_outcome_action_set_mismatch",
            )
        for action in safe_set:
            if mask.get(action) is not True:
                _append_row_failure(
                    failures,
                    row_index,
                    "safe_action_not_public_mask_supported",
                    action=action,
                )
        targets = _list_of_mappings(row.get("action_value_targets"))
        targets_by_action = {
            str(target.get("action")): target
            for target in targets
            if str(target.get("action")) in ACTION_NAMES
        }
        if len(targets) != len(ACTION_NAMES):
            _append_row_failure(failures, row_index, "target_count_mismatch")
        if set(targets_by_action) != set(ACTION_NAMES):
            _append_row_failure(failures, row_index, "target_action_set_mismatch")
        if [str(target.get("action") or "") for target in targets] != list(
            ACTION_NAMES
        ):
            _append_row_failure(failures, row_index, "target_action_order_mismatch")
        for action in ACTION_NAMES:
            target = _mapping(targets_by_action.get(action))
            public_mask = target.get("public_mask") is True
            target_available = target.get("target_available") is True
            replay_verified = target.get("replay_verified") is True
            safe_target = target.get("safe_target") is True
            if public_mask != (mask.get(action) is True):
                _append_row_failure(
                    failures,
                    row_index,
                    "target_public_mask_mismatch",
                    action=action,
                )
            if target_available and public_mask is not True:
                _append_row_failure(
                    failures,
                    row_index,
                    "available_target_not_public_mask_supported",
                    action=action,
                )
            if target_available and replay_verified is not True:
                _append_row_failure(
                    failures,
                    row_index,
                    "available_target_not_replay_verified",
                    action=action,
                )
            if safe_target != (action in safe_set):
                _append_row_failure(
                    failures,
                    row_index,
                    "safe_target_mismatch",
                    action=action,
                )
            if safe_target and target_available is not True:
                _append_row_failure(
                    failures,
                    row_index,
                    "safe_target_unavailable",
                    action=action,
                )
            if target_available:
                value = target.get("value_target")
                if not _finite_number(value):
                    _append_row_failure(
                        failures,
                        row_index,
                        "value_target_not_finite",
                        action=action,
                    )
            elif target.get("value_target") is not None:
                _append_row_failure(
                    failures,
                    row_index,
                    "value_target_present_for_unavailable_target",
                    action=action,
                )
        robust_winner = str(row.get("robust_winner_action") or "")
        if classification == "unique_replay_verified_best_action":
            if len(safe_set) != 1:
                _append_row_failure(
                    failures,
                    row_index,
                    "unique_classification_safe_set_not_single",
                )
            if robust_winner not in safe_set:
                _append_row_failure(
                    failures,
                    row_index,
                    "unique_winner_missing_from_safe_set",
                )
        elif classification == "tied_replay_verified_best_action_set":
            if len(safe_set) < 2:
                _append_row_failure(
                    failures,
                    row_index,
                    "tied_classification_safe_set_too_small",
                )
            if robust_winner:
                _append_row_failure(
                    failures,
                    row_index,
                    "tied_row_has_robust_winner_action",
                )
        else:
            _append_row_failure(failures, row_index, "unknown_target_classification")
        metadata = _mapping(row.get("metadata"))
        seed = _int(metadata.get("seed"), default=-1)
        if seed > 0:
            source_seed_counts.update([seed])
        if seed not in support_seed_set:
            _append_row_failure(
                failures,
                row_index,
                "source_seed_not_support_provenance",
                seed=seed,
            )
        if (
            metadata.get("source_seed_is_support_provenance_not_future_promotion_holdout")
            is not True
        ):
            _append_row_failure(
                failures,
                row_index,
                "source_seed_not_marked_support_provenance",
            )
        for flag in (
            "source_identity_used_as_trainable_input",
            "runtime_requested_or_resolved_action_used_as_trainable_input",
            "future_outcome_used_as_trainable_input",
            "target_safe_action_used_as_trainable_input",
            "private_state_used_as_trainable_input",
        ):
            if metadata.get(flag) is not False:
                _append_row_failure(
                    failures,
                    row_index,
                    "metadata_trainable_input_flag_not_false",
                    field=flag,
                )
    return {
        "policy": "m3_carrion_survivor_continuation_v172_row_schema_validation_v1",
        "passed": not failures and bool(rows),
        "failure_count": len(failures),
        "failures": failures[:128],
        "row_count": len(rows),
        "schema_counts": dict(sorted(schema_counts.items())),
        "classification_counts": dict(sorted(classification_counts.items())),
        "support_provenance_seed_row_counts": {
            str(seed): int(source_seed_counts.get(int(seed), 0))
            for seed in sorted(support_seed_set)
            if int(source_seed_counts.get(int(seed), 0)) > 0
        },
    }


def trainable_payload_leakage_scan(
    feature_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for row_index, payload in enumerate(feature_payloads):
        _scan_trainable_payload(
            value=payload,
            row_index=row_index,
            path=("trainable_public_features",),
            failures=failures,
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v172_trainable_payload_leakage_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:96],
        "forbidden_tokens": list(FORBIDDEN_TRAINABLE_PAYLOAD_TOKENS),
        "payload_count": len(feature_payloads),
    }


def target_dataset_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    v170_best_set_width: float = V170_BEST_SET_WIDTH,
    max_dominant_support_action_share: float = DEFAULT_MAX_DOMINANT_SUPPORT_ACTION_SHARE,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> dict[str, object]:
    support_counts: Counter[str] = Counter()
    target_available_counts: Counter[str] = Counter()
    width_counts: Counter[int] = Counter()
    unique_rows = 0
    tied_rows = 0
    unresolved_rows = 0
    per_seed: dict[int, dict[str, object]] = defaultdict(
        lambda: {
            "row_count": 0,
            "support_counts": Counter(),
            "width_counts": Counter(),
            "unique_action_row_count": 0,
            "tied_action_row_count": 0,
            "unresolved_row_count": 0,
        }
    )
    for row in rows:
        safe_set = _ordered_action_set(row.get("safe_action_set"))
        width = len(safe_set)
        width_counts.update([width])
        support_counts.update(safe_set)
        unique_rows += int(width == 1)
        tied_rows += int(width > 1)
        unresolved_rows += int(width == 0)
        for target in _list_of_mappings(row.get("action_value_targets")):
            action = str(target.get("action") or "")
            if action in ACTION_NAMES and target.get("target_available") is True:
                target_available_counts.update([action])
        seed = _int(_mapping(row.get("metadata")).get("seed"), default=-1)
        seed_payload = per_seed[seed]
        seed_payload["row_count"] = _int(seed_payload["row_count"]) + 1
        counter = seed_payload["support_counts"]
        if isinstance(counter, Counter):
            counter.update(safe_set)
        widths = seed_payload["width_counts"]
        if isinstance(widths, Counter):
            widths.update([width])
        seed_payload["unique_action_row_count"] = _int(
            seed_payload["unique_action_row_count"]
        ) + int(width == 1)
        seed_payload["tied_action_row_count"] = _int(
            seed_payload["tied_action_row_count"]
        ) + int(width > 1)
        seed_payload["unresolved_row_count"] = _int(
            seed_payload["unresolved_row_count"]
        ) + int(width == 0)
    support_counts = _ordered_counter(support_counts)
    target_available_counts = _ordered_counter(target_available_counts)
    total_support = sum(support_counts.values())
    dominant = _dominant_count_share(Counter(support_counts))
    widths = [
        width
        for width, count in width_counts.items()
        for _ in range(int(count))
        if width > 0
    ]
    average_width = _round(sum(widths) / len(widths)) if widths else 0.0
    width_improved = bool(widths) and average_width < _round(v170_best_set_width)
    noncollapsed = (
        len([action for action, count in support_counts.items() if count > 0]) >= 2
        and float(dominant.get("share") or 0.0)
        <= float(max_dominant_support_action_share)
    )
    seed_summary = _per_seed_summary(per_seed, support_provenance_seeds)
    return {
        "policy": "m3_carrion_survivor_continuation_v172_target_dataset_summary_v1",
        "row_count": len(rows),
        "action_support_distribution": {
            "per_action_support_counts": dict(support_counts),
            "total_support_labels": int(total_support),
            "distinct_supported_action_count": len(support_counts),
            "dominant_support_action": dominant.get("key"),
            "dominant_support_action_count": dominant.get("count"),
            "dominant_support_action_share": dominant.get("share"),
            "max_dominant_support_action_share": _round(
                max_dominant_support_action_share
            ),
            "action_distribution_noncollapsed": noncollapsed,
        },
        "target_available_action_distribution": {
            "per_action_target_available_counts": dict(target_available_counts),
        },
        "safe_set_width_distribution": {
            "width_counts": {
                str(width): int(count)
                for width, count in sorted(width_counts.items())
            },
            "average_safe_set_width": average_width,
            "v170_best_set_width": _round(v170_best_set_width),
            "width_improved_vs_v170": width_improved,
        },
        "unique_action_vs_tied_action_row_counts": {
            "unique_action_row_count": int(unique_rows),
            "tied_action_row_count": int(tied_rows),
            "unresolved_row_count": int(unresolved_rows),
        },
        "per_seed_support_summary": seed_summary,
        "support_ready": (
            bool(rows)
            and width_improved
            and noncollapsed
            and unresolved_rows == 0
            and _all_support_provenance_seeds_present(
                seed_summary,
                support_provenance_seeds,
            )
        ),
    }


def source_split_evaluation_plan_for_v173(
    *,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> dict[str, object]:
    seeds = sorted({int(seed) for seed in support_provenance_seeds})
    return {
        "policy": "m3_carrion_survivor_continuation_v172_v173_source_split_plan_v1",
        "recommended_next_route": "v173_diagnostics_only_source_split_scorer_no_training",
        "source_split_evaluation_required": True,
        "leave_one_support_seed_out_required": True,
        "source_group_key": "metadata.seed",
        "support_provenance_seeds": seeds,
        "disallowed_future_promotion_heldout_seed_reuse": seeds,
        "new_promotion_heldout_broad_seeds_required": True,
        "target_leaky_or_exact-top_baselines_may_be_reported_as_non_runtime_upper_bounds": True,
        "training_authorized": False,
        "scorer_retraining_authorized": False,
        "shadow_eval_authorized": False,
        "runtime_integration_authorized": False,
        "promotion_authorized": False,
    }


def promotion_heldout_warning(
    *,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> dict[str, object]:
    seeds = sorted({int(seed) for seed in support_provenance_seeds})
    seed_text = ",".join(str(seed) for seed in seeds)
    return {
        "policy": "m3_carrion_survivor_continuation_v172_promotion_heldout_warning_v1",
        "warning": (
            f"Seeds {seed_text} are support-provenance seeds for this "
            "dataset and are not valid future promotion heldout seeds for any "
            "scorer trained or selected on it."
        ),
        "support_provenance_seeds": seeds,
        "future_promotion_heldout_seed_reuse_allowed": False,
        "new_heldout_seed_matrix_required": True,
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    row_build_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    target_summary: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v172_replay_target_dataset_expansion_"
    if (
        source_validation.get("passed") is not True
        or row_build_validation.get("passed") is not True
        or leakage_scan.get("passed") is not True
        or row_schema_validation.get("passed") is not True
    ):
        return prefix + "closed_invalid"
    if target_summary.get("support_ready") is True:
        return (
            prefix
            + "replay_target_dataset_support_ready_for_v173_source_split_scorer_no_training"
        )
    return prefix + "replay_target_dataset_support_limited_no_training"


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification.endswith(
        "replay_target_dataset_support_ready_for_v173_source_split_scorer_no_training"
    )
    invalid = classification.endswith("closed_invalid")
    return {
        "policy": "m3_carrion_survivor_continuation_v172_route_recommendation_v1",
        "recommended_next_route": (
            "repair_source_leakage_or_row_schema_before_dataset_use"
            if invalid
            else "v173_diagnostics_only_source_split_scorer_no_training"
            if ready
            else "expand_replay_target_support_before_scorer_work"
        ),
        "v173_source_split_scorer_recommended": ready,
        "training_authorized": False,
        "scorer_retraining_authorized": False,
        "runtime_integration_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
    }


def _v172_metadata(
    *,
    source_metadata: Mapping[str, object],
    source_row: Mapping[str, object],
    seed: int,
    v171_report_exact_digest: str,
    v171_dataset_digest: str,
) -> dict[str, object]:
    return {
        "metadata_schema_version": (
            "m3_carrion_survivor_continuation_v172_replay_target_metadata_v1"
        ),
        "source": "v171_replay_expansion_dataset",
        "source_v171_report_exact_digest": v171_report_exact_digest,
        "source_v171_dataset_digest": v171_dataset_digest,
        "source_v171_row_schema_version": source_row.get("schema_version"),
        "source_v171_feature_policy_id": source_row.get("feature_policy_id"),
        "seed": seed,
        "branch_id": source_metadata.get("branch_id"),
        "branch_tick": source_metadata.get("branch_tick"),
        "agent_id": source_metadata.get("agent_id"),
        "source_path": source_metadata.get("source_path"),
        "line_number": source_metadata.get("line_number"),
        "source_record_digest": source_metadata.get("source_record_digest"),
        "materialized_record_digest": source_metadata.get("materialized_record_digest"),
        "branch_state_digest": source_metadata.get("branch_state_digest"),
        "replay_digests_by_action": dict(
            _mapping(source_metadata.get("replay_digests_by_action"))
        ),
        "source_identity_used_for_exact_materialization_only": True,
        "source_seed_is_support_provenance_not_future_promotion_holdout": True,
        "source_identity_used_as_trainable_input": False,
        "runtime_requested_or_resolved_action_used_as_trainable_input": False,
        "future_outcome_used_as_trainable_input": False,
        "target_safe_action_used_as_trainable_input": False,
        "private_state_used_as_trainable_input": False,
    }


def _normalized_action_value_targets(
    row: Mapping[str, object],
    *,
    safe_action_set: Sequence[str],
) -> list[dict[str, object]]:
    source_targets = {
        str(target.get("action")): target
        for target in _list_of_mappings(row.get("action_value_targets"))
        if str(target.get("action")) in ACTION_NAMES
    }
    safe = set(_ordered_action_set(safe_action_set))
    targets = []
    for action in ACTION_NAMES:
        source = _mapping(source_targets.get(action))
        target_available = source.get("target_available") is True
        targets.append(
            {
                "action": action,
                "public_mask": source.get("public_mask") is True,
                "target_available": target_available,
                "replay_verified": source.get("replay_verified") is True,
                "safe_target": action in safe,
                "best_outcome_action": action in safe,
                "replay_outcome_summary": (
                    list(source.get("replay_outcome_key") or [])
                    if target_available
                    else None
                ),
                "score_target": source.get("score_target")
                if target_available
                else None,
                "value_target": source.get("value_target")
                if target_available
                else None,
            }
        )
    return targets


def _target_outcome_summary(
    targets: Sequence[Mapping[str, object]],
    *,
    safe_action_set: Sequence[str],
) -> dict[str, object]:
    available = [
        str(target.get("action"))
        for target in targets
        if target.get("target_available") is True
    ]
    replay_verified = [
        str(target.get("action"))
        for target in targets
        if target.get("replay_verified") is True
    ]
    return {
        "best_outcome_action_set": _ordered_action_set(safe_action_set),
        "best_outcome_action_count": len(_ordered_action_set(safe_action_set)),
        "target_available_actions": _ordered_action_set(available),
        "target_available_action_count": len(available),
        "replay_verified_actions": _ordered_action_set(replay_verified),
        "replay_verified_action_count": len(replay_verified),
        "replay_outcome_summaries_by_action": {
            str(target.get("action")): target.get("replay_outcome_summary")
            for target in targets
            if target.get("target_available") is True
        },
    }


def _per_seed_summary(
    payloads: Mapping[int, Mapping[str, object]],
    support_provenance_seeds: Sequence[int],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for seed in sorted({int(seed) for seed in support_provenance_seeds}):
        payload = _mapping(payloads.get(seed))
        support = payload.get("support_counts")
        widths = payload.get("width_counts")
        support_counts = support if isinstance(support, Counter) else Counter()
        width_counts = widths if isinstance(widths, Counter) else Counter()
        width_values = [
            width
            for width, count in width_counts.items()
            for _ in range(int(count))
            if width > 0
        ]
        result[str(seed)] = {
            "support_provenance_seed": True,
            "future_promotion_heldout_seed": False,
            "row_count": _int(payload.get("row_count")),
            "action_support_counts": dict(_ordered_counter(support_counts)),
            "safe_set_width_counts": {
                str(width): int(count)
                for width, count in sorted(width_counts.items())
                if int(count) > 0
            },
            "average_safe_set_width": (
                _round(sum(width_values) / len(width_values))
                if width_values
                else 0.0
            ),
            "unique_action_row_count": _int(
                payload.get("unique_action_row_count")
            ),
            "tied_action_row_count": _int(payload.get("tied_action_row_count")),
            "unresolved_row_count": _int(payload.get("unresolved_row_count")),
        }
    return result


def _all_support_provenance_seeds_present(
    per_seed_summary: Mapping[str, Mapping[str, object]],
    support_provenance_seeds: Sequence[int],
) -> bool:
    return all(
        _int(_mapping(per_seed_summary.get(str(int(seed)))).get("row_count")) > 0
        for seed in support_provenance_seeds
    )


def _scan_trainable_payload(
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
            token = _matching_forbidden_token(key_text)
            if token:
                failures.append(
                    {
                        "row_index": row_index,
                        "path": ".".join(current_path),
                        "reason": "forbidden_key_token",
                        "token": token,
                    }
                )
            _scan_trainable_payload(
                value=item,
                row_index=row_index,
                path=current_path,
                failures=failures,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_trainable_payload(
                value=item,
                row_index=row_index,
                path=(*path, str(index)),
                failures=failures,
            )


def _matching_forbidden_token(text: str) -> str | None:
    lowered = text.lower()
    for token in FORBIDDEN_TRAINABLE_PAYLOAD_TOKENS:
        if token in lowered:
            return token
    return None


def _v171_lifecycle_validation(report: Mapping[str, object]) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    if report.get("diagnostics_only") is not True:
        failures.append(
            {
                "field": "diagnostics_only",
                "observed": report.get("diagnostics_only"),
            }
        )
    for field in (
        "training_ran",
        "scorer_retraining_ran",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "runtime_observation_schema_changed",
        "runtime_policy_changed",
        "shadow_eval_ran",
        "live_ab_ran",
        "live_ab_allowed",
        "k_tuning_ran",
        "threshold_tuning_ran",
        "replay_viewer_schema_changed",
        "promotion_authorized",
        "staging_authorized",
        "commit_authorized",
        "reset_authorized",
        "clean_authorized",
    ):
        if field in report and report.get(field) is not False:
            failures.append({"field": field, "observed": report.get(field)})
    return {
        "policy": "m3_carrion_survivor_continuation_v172_v171_lifecycle_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "target_dataset_expansion_only": True,
        "input_rows_must_come_from_v171_merged_jsonl": True,
        "validates_v171_exact_digest_before_build": True,
        "validates_v171_dataset_digest_before_build": True,
        "trainable_payload_public_observation_action_mask_only": True,
        "target_fields_may_contain_replay_best_outcome_action_set": True,
        "target_fields_may_contain_replay_outcome_summary": True,
        "metadata_is_non_trainable_provenance_only": True,
        "support_provenance_seeds_are_future_promotion_heldout_seeds": False,
        "training_allowed": False,
        "scorer_retraining_allowed": False,
        "runtime_artifact_allowed": False,
        "runtime_action_change_allowed": False,
        "shadow_or_live_eval_allowed": False,
        "k_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "replay_viewer_schema_change_allowed": False,
        "promotion_allowed": False,
        "staging_allowed": False,
        "commit_allowed": False,
        "reset_allowed": False,
        "clean_allowed": False,
    }


def _lifecycle_flags() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "dataset_created": True,
        "training_ran": False,
        "training_authorized": False,
        "scorer_retraining_ran": False,
        "scorer_retraining_authorized": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": False,
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "k_tuning_ran": False,
        "threshold_tuning_ran": False,
        "replay_viewer_schema_changed": False,
        "promotion_authorized": False,
        "staging_authorized": False,
        "commit_authorized": False,
        "reset_authorized": False,
        "clean_authorized": False,
        "non_promoted": True,
    }


def _skipped_row_build_validation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v172_row_build_validation_v1",
        "passed": False,
        "failure_count": 1,
        "failures": [{"reason": reason}],
        "source_v171_row_count": 0,
        "target_row_count": 0,
    }


def _complete_action_mask(mask: Mapping[str, object]) -> dict[str, bool]:
    return {action: bool(mask.get(action, False)) for action in ACTION_NAMES}


def _ordered_action_set(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(
        {str(action) for action in value if str(action) in ACTION_NAMES},
        key=_action_order,
    )


def _ordered_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        action: int(counter.get(action, 0))
        for action in ACTION_NAMES
        if int(counter.get(action, 0)) > 0
    }


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _append_row_failure(
    failures: list[dict[str, object]],
    row_index: int,
    reason: str,
    **extra: object,
) -> None:
    failures.append({"row_index": row_index, "reason": reason, **extra})


def _json_round_trip_digest(payload: Mapping[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(without_digest, sort_keys=True, allow_nan=False))
    )


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
        for row in rows
    ]
    output.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
