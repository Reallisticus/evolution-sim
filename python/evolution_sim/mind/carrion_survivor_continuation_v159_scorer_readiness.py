from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    _dominant_count_share,
    _float,
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
    hard_trainable_feature_leakage_scan,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
    DEFAULT_OUTPUT_PATH as DEFAULT_V158_REPORT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V158_DATASET_PATH,
    DEFAULT_V154_DATASET_PATH,
    DEFAULT_V154_REPORT_PATH,
    DEFAULT_V155_REPORT_PATH,
    DEFAULT_V156_REPORT_PATH,
    DEFAULT_V157_REPORT_PATH,
    EXPECTED_V157_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_SCHEMA_VERSION,
    exact_digest_validation_report,
    lifecycle_authorization_scan,
    validate_action_value_target_dataset_sources,
)
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    TARGET_CARRION_SEEDS,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
    load_v154_dataset,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V159_SCORER_READINESS_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_action_value_scorer_readiness_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V159_SCORER_READINESS_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_action_value_scorer_readiness_v1"
)
EXPECTED_V158_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_public_state_action_value_"
    "target_dataset_support_ready_no_training"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v159-carrion-survivor-continuation-scorer-readiness-report.json"
)
DEFAULT_MAX_DOMINANT_PROPOSED_ACTION_SHARE = 0.50
DEFAULT_MIN_EXACT_SHADOW_SAFE_HIT_SHARE = 0.95
DEFAULT_MIN_EXACT_SHADOW_MINUS_BEST_TRIVIAL_COVERAGE_SHARE = 0.05
DEFAULT_MIN_NEAR_EXACT_MASK_COMPARATOR_COVERAGE_SHARE = 0.50
DEFAULT_MIN_NEAR_EXACT_MASK_SHADOW_SAFE_HIT_SHARE = 0.50


class CarrionSurvivorContinuationV159ScorerReadinessError(ValueError):
    pass


def run_carrion_survivor_continuation_v159_scorer_readiness(
    *,
    v154_report_path: str | Path = DEFAULT_V154_REPORT_PATH,
    v154_dataset_path: str | Path = DEFAULT_V154_DATASET_PATH,
    v155_report_path: str | Path = DEFAULT_V155_REPORT_PATH,
    v156_report_path: str | Path = DEFAULT_V156_REPORT_PATH,
    v157_report_path: str | Path = DEFAULT_V157_REPORT_PATH,
    v158_report_path: str | Path = DEFAULT_V158_REPORT_PATH,
    v158_dataset_path: str | Path = DEFAULT_V158_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    max_dominant_safe_action_share: float = DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
    max_dominant_proposed_action_share: float = (
        DEFAULT_MAX_DOMINANT_PROPOSED_ACTION_SHARE
    ),
    min_exact_shadow_safe_hit_share: float = DEFAULT_MIN_EXACT_SHADOW_SAFE_HIT_SHARE,
    min_exact_shadow_minus_best_trivial_coverage_share: float = (
        DEFAULT_MIN_EXACT_SHADOW_MINUS_BEST_TRIVIAL_COVERAGE_SHARE
    ),
    min_near_exact_mask_comparator_coverage_share: float = (
        DEFAULT_MIN_NEAR_EXACT_MASK_COMPARATOR_COVERAGE_SHARE
    ),
    min_near_exact_mask_shadow_safe_hit_share: float = (
        DEFAULT_MIN_NEAR_EXACT_MASK_SHADOW_SAFE_HIT_SHARE
    ),
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    v154_report = load_json_report(v154_report_path)
    v155_report = load_json_report(v155_report_path)
    v156_report = load_json_report(v156_report_path)
    v157_report = load_json_report(v157_report_path)
    v158_report = load_json_report(v158_report_path)
    v154_dataset_rows = load_v154_dataset(v154_dataset_path)
    v158_rows = load_v154_dataset(v158_dataset_path)
    branch_results = _list_of_mappings(
        v154_report.get("continuation_branch_results")
    )

    source_validation = validate_v159_sources(
        v154_report=v154_report,
        v155_report=v155_report,
        v156_report=v156_report,
        v157_report=v157_report,
        v158_report=v158_report,
        v154_dataset_rows=v154_dataset_rows,
        v158_rows=v158_rows,
        branch_results=branch_results,
        target_seeds=target_seeds,
    )
    row_contract = validate_v158_target_rows(v158_rows)
    leakage_scan = v159_target_dataset_leakage_scan(v158_rows)
    action_support = action_support_diagnostics(
        v158_rows,
        max_dominant_safe_action_share=max_dominant_safe_action_share,
        max_dominant_proposed_action_share=max_dominant_proposed_action_share,
    )
    ranking = action_ranking_diagnostics(v158_rows)
    shadow_coverage = shadow_coverage_diagnostics(
        v158_rows,
        min_exact_shadow_safe_hit_share=min_exact_shadow_safe_hit_share,
        min_exact_shadow_minus_best_trivial_coverage_share=(
            min_exact_shadow_minus_best_trivial_coverage_share
        ),
        min_near_exact_mask_comparator_coverage_share=(
            min_near_exact_mask_comparator_coverage_share
        ),
        min_near_exact_mask_shadow_safe_hit_share=(
            min_near_exact_mask_shadow_safe_hit_share
        ),
    )
    classification = _top_level_classification(
        source_validation=source_validation,
        row_contract=row_contract,
        leakage_scan=leakage_scan,
        action_support=action_support,
        shadow_coverage=shadow_coverage,
    )
    route = _route_recommendation(classification)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V159_SCORER_READINESS_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V159_SCORER_READINESS_POLICY,
        "contract": {
            "diagnostics_only": True,
            "artifact_creation_allowed": False,
            "model_artifact_creation_allowed": False,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
            "shadow_live_ab_allowed": False,
            "live_runtime_override_allowed": False,
            "future_training_requires_separate_opt_in_command": True,
            "future_training_diagnostic_recommended_by_route_only": bool(
                route.get("future_opt_in_training_diagnostic_recommended")
            ),
            "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_future_outcome_as_trainable_input": False,
            "uses_branch_reason_as_trainable_input": False,
        },
        "inputs": {
            "v154_report": str(v154_report_path),
            "v154_dataset": str(v154_dataset_path),
            "v155_report": str(v155_report_path),
            "v156_report": str(v156_report_path),
            "v157_report": str(v157_report_path),
            "v158_report": str(v158_report_path),
            "v158_dataset": str(v158_dataset_path),
            "target_seeds": [int(seed) for seed in target_seeds],
            "max_dominant_safe_action_share": _round(
                max_dominant_safe_action_share
            ),
            "max_dominant_proposed_action_share": _round(
                max_dominant_proposed_action_share
            ),
            "min_exact_shadow_safe_hit_share": _round(
                min_exact_shadow_safe_hit_share
            ),
            "min_exact_shadow_minus_best_trivial_coverage_share": _round(
                min_exact_shadow_minus_best_trivial_coverage_share
            ),
            "min_near_exact_mask_comparator_coverage_share": _round(
                min_near_exact_mask_comparator_coverage_share
            ),
            "min_near_exact_mask_shadow_safe_hit_share": _round(
                min_near_exact_mask_shadow_safe_hit_share
            ),
        },
        "source_validation": source_validation,
        "row_contract_validation": row_contract,
        "leakage_scan": leakage_scan,
        "action_support": action_support,
        "action_ranking": ranking,
        "shadow_coverage": shadow_coverage,
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": route,
        "dataset_digest": stable_payload_digest(v158_rows),
        "artifact_created": False,
        "model_artifact_created": False,
        "training_ran": False,
        "shadow_live_ab_ran": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = stable_payload_digest(report)
    write_json(output_path, report)
    return report


def validate_v159_sources(
    *,
    v154_report: Mapping[str, object],
    v155_report: Mapping[str, object],
    v156_report: Mapping[str, object],
    v157_report: Mapping[str, object],
    v158_report: Mapping[str, object],
    v154_dataset_rows: Sequence[Mapping[str, object]],
    v158_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    target_seeds: Sequence[int] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    failures: list[str] = []
    v154_to_v157 = validate_action_value_target_dataset_sources(
        v154_report=v154_report,
        v155_report=v155_report,
        v156_report=v156_report,
        v157_report=v157_report,
        dataset_rows=v154_dataset_rows,
        branch_results=branch_results,
        target_seeds=target_seeds,
    )
    if v154_to_v157.get("passed") is not True:
        failures.append("v154_v155_v156_v157_source_validation_failed")
    if (
        v158_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_SCHEMA_VERSION
    ):
        failures.append("v158_schema_version_mismatch")
    if (
        v158_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_POLICY
    ):
        failures.append("v158_policy_mismatch")
    observed_v158_classification = _mapping(v158_report.get("classification")).get(
        "primary"
    )
    if observed_v158_classification != EXPECTED_V158_CLASSIFICATION:
        failures.append("v158_unexpected_classification")
    if _mapping(v158_report.get("source_validation")).get("passed") is not True:
        failures.append("v158_source_validation_failed")
    if _mapping(v158_report.get("target_build_validation")).get("passed") is not True:
        failures.append("v158_target_build_validation_failed")
    if _mapping(v158_report.get("leakage_scan")).get("passed") is not True:
        failures.append("v158_leakage_scan_failed")
    lifecycle = lifecycle_authorization_scan(
        {
            "v154": v154_report,
            "v155": v155_report,
            "v156": v156_report,
            "v157": v157_report,
            "v158": v158_report,
        }
    )
    if lifecycle.get("passed") is not True:
        failures.append("lifecycle_authorization_fields_not_false")
    exact_digest_validation = exact_digest_validation_report(v158_report)
    if exact_digest_validation.get("passed") is not True:
        failures.append("v158_exact_digest_mismatch")
    v158_dataset_digest = stable_payload_digest(v158_rows)
    v158_dataset_report = _mapping(v158_report.get("dataset"))
    if v158_dataset_report.get("dataset_digest") != v158_dataset_digest:
        failures.append("v158_dataset_digest_mismatch")
    if _int(v158_dataset_report.get("action_value_target_row_count")) != len(
        v158_rows
    ):
        failures.append("v158_dataset_row_count_mismatch")
    if (
        v158_dataset_report.get("row_schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION
    ):
        failures.append("v158_row_schema_version_mismatch")
    return {
        "policy": "m3_carrion_survivor_continuation_v159_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v157_classification": EXPECTED_V157_CLASSIFICATION,
        "expected_v158_classification": EXPECTED_V158_CLASSIFICATION,
        "observed_v158_classification": observed_v158_classification,
        "v154_to_v157_validation": v154_to_v157,
        "lifecycle_authorization": lifecycle,
        "v158_exact_digest_validation": exact_digest_validation,
        "v154_dataset_digest": stable_payload_digest(v154_dataset_rows),
        "v158_dataset_digest": v158_dataset_digest,
        "v158_reported_dataset_digest": v158_dataset_report.get("dataset_digest"),
        "v158_action_value_target_row_count": len(v158_rows),
        "v158_exact_digest": v158_report.get("exact_digest"),
    }


def validate_v158_target_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    complete_target_rows = 0
    public_action_count = 0
    unavailable_public_target_count = 0
    unavailable_target_count = 0
    safe_target_count = 0
    for row_index, row in enumerate(rows):
        if (
            row.get("schema_version")
            != M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION
        ):
            _append_failure(failures, row_index, "row_schema_version_mismatch")
        features = _mapping(row.get("trainable_public_features"))
        if not features:
            _append_failure(failures, row_index, "trainable_public_features_missing")
        mask = _mapping(row.get("public_action_mask"))
        mask_failures = _validate_public_action_mask(mask)
        for reason in mask_failures:
            _append_failure(failures, row_index, reason)
        feature_mask = _mapping(features.get("action_mask"))
        if feature_mask and _complete_bool_mask(feature_mask) != _complete_bool_mask(mask):
            _append_failure(failures, row_index, "feature_action_mask_mismatch")
        safe_set = _safe_action_set(row)
        if not safe_set:
            _append_failure(failures, row_index, "safe_action_set_empty")
        for action in safe_set:
            if bool(mask.get(action, False)) is not True:
                _append_failure(
                    failures,
                    row_index,
                    "safe_action_not_public_mask_supported",
                    action=action,
                )
        target_list = _list_of_mappings(row.get("action_value_targets"))
        targets_by_action = _targets_by_action(row)
        if len(target_list) != len(ACTION_NAMES):
            _append_failure(failures, row_index, "action_value_target_count_mismatch")
        if set(targets_by_action) != set(ACTION_NAMES):
            _append_failure(failures, row_index, "action_value_target_action_set_mismatch")
        if [str(target.get("action", "")) for target in target_list] != list(
            ACTION_NAMES
        ):
            _append_failure(failures, row_index, "action_value_target_order_mismatch")
        if set(targets_by_action) == set(ACTION_NAMES):
            complete_target_rows += 1
        for action in ACTION_NAMES:
            public = bool(mask.get(action, False))
            public_action_count += int(public)
            target = _mapping(targets_by_action.get(action))
            target_available = target.get("target_available") is True
            safe_target = target.get("safe_target") is True
            safe_target_count += int(safe_target)
            unavailable_target_count += int(not target_available)
            if public and not target_available:
                unavailable_public_target_count += 1
                _append_failure(
                    failures,
                    row_index,
                    "public_action_missing_value_target",
                    action=action,
                )
            if target_available and not public:
                _append_failure(
                    failures,
                    row_index,
                    "unmasked_action_has_value_target",
                    action=action,
                )
            if safe_target != (action in safe_set):
                _append_failure(
                    failures,
                    row_index,
                    "safe_target_mismatch",
                    action=action,
                )
            if safe_target and not target_available:
                _append_failure(
                    failures,
                    row_index,
                    "safe_target_unavailable",
                    action=action,
                )
            _validate_target_numeric_fields(
                failures=failures,
                row_index=row_index,
                action=action,
                target=target,
                target_available=target_available,
            )
        classification = str(row.get("target_classification", ""))
        robust_winner = str(row.get("robust_winner_action") or "")
        if classification == "unique_robust_winner":
            if robust_winner not in safe_set:
                _append_failure(
                    failures,
                    row_index,
                    "unique_winner_missing_from_safe_action_set",
                    action=robust_winner,
                )
            if len(safe_set) != 1:
                _append_failure(failures, row_index, "unique_winner_safe_set_not_single")
        elif robust_winner:
            _append_failure(failures, row_index, "unexpected_robust_winner_action")
        if classification == "multi_action_safe_set" and len(safe_set) < 2:
            _append_failure(failures, row_index, "multi_action_safe_set_too_small")
    total_target_slots = len(rows) * len(ACTION_NAMES)
    return {
        "policy": "m3_carrion_survivor_continuation_v159_row_contract_validation_v1",
        "passed": not failures and bool(rows),
        "failure_count": len(failures),
        "failures": failures[:64],
        "row_count": len(rows),
        "complete_action_value_target_row_count": complete_target_rows,
        "public_action_target_count": public_action_count,
        "safe_target_count": safe_target_count,
        "unavailable_target_count": unavailable_target_count,
        "unavailable_target_share": _safe_rate(unavailable_target_count, total_target_slots),
        "unavailable_public_target_count": unavailable_public_target_count,
    }


def v159_target_dataset_leakage_scan(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    trainable_payloads = [
        {
            "trainable_public_features": deepcopy(
                _mapping(row.get("trainable_public_features"))
            ),
            "public_action_mask": deepcopy(_mapping(row.get("public_action_mask"))),
        }
        for row in rows
    ]
    scan = hard_trainable_feature_leakage_scan(trainable_payloads)
    return {
        "policy": "m3_carrion_survivor_continuation_v159_leakage_scan_v1",
        "passed": scan.get("passed") is True,
        "failure_count": scan.get("failure_count"),
        "failures": scan.get("failures"),
        "trainable_feature_row_count": len(trainable_payloads),
        "hard_trainable_feature_leakage_scan": scan,
    }


def action_support_diagnostics(
    rows: Sequence[Mapping[str, object]],
    *,
    max_dominant_safe_action_share: float = DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
    max_dominant_proposed_action_share: float = (
        DEFAULT_MAX_DOMINANT_PROPOSED_ACTION_SHARE
    ),
) -> dict[str, object]:
    safe_counts: Counter[str] = Counter()
    safe_set_sizes: Counter[int] = Counter()
    proposed_counts: Counter[str] = Counter()
    proposed_safe_hits = 0
    proposed_available_rows = 0
    for row in rows:
        safe_set = _safe_action_set(row)
        safe_set_sizes[len(safe_set)] += 1
        safe_counts.update(safe_set)
        proposed = _top_value_target(row)
        if proposed:
            proposed_available_rows += 1
            action = str(proposed.get("action", ""))
            proposed_counts.update([action])
            if action in safe_set:
                proposed_safe_hits += 1
    safe_counts = Counter(
        {action: int(safe_counts.get(action, 0)) for action in ACTION_NAMES}
    )
    proposed_counts = Counter(
        {action: int(proposed_counts.get(action, 0)) for action in ACTION_NAMES}
    )
    safe_dominant = _dominant_count_share(_positive_counter(safe_counts))
    proposed_dominant = _dominant_count_share(_positive_counter(proposed_counts))
    total_safe = sum(int(count) for count in safe_counts.values())
    action_support_non_collapsed = (
        total_safe > 0
        and len([action for action, count in safe_counts.items() if count > 0]) >= 2
        and _float(safe_dominant.get("share")) <= float(max_dominant_safe_action_share)
    )
    proposed_action_non_collapsed = (
        proposed_available_rows > 0
        and len([action for action, count in proposed_counts.items() if count > 0])
        >= 2
        and _float(proposed_dominant.get("share"))
        <= float(max_dominant_proposed_action_share)
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v159_action_support_diagnostics_v1",
        "row_count": len(rows),
        "safe_action_total": int(total_safe),
        "safe_set_row_count": sum(1 for row in rows if _safe_action_set(row)),
        "safe_set_size_counts": {
            str(size): int(count) for size, count in sorted(safe_set_sizes.items())
        },
        "per_action_safe_support_counts": dict(
            sorted(_positive_counter(safe_counts).items())
        ),
        "dominant_safe_action": safe_dominant.get("key"),
        "dominant_safe_action_count": safe_dominant.get("count"),
        "dominant_safe_action_share": safe_dominant.get("share"),
        "max_dominant_safe_action_share": _round(max_dominant_safe_action_share),
        "action_support_non_collapsed": action_support_non_collapsed,
        "proposed_action_counts": dict(
            sorted(_positive_counter(proposed_counts).items())
        ),
        "proposed_action_row_count": proposed_available_rows,
        "proposed_safe_hit_count": proposed_safe_hits,
        "proposed_safe_hit_share": _safe_rate(proposed_safe_hits, len(rows)),
        "dominant_proposed_action": proposed_dominant.get("key"),
        "dominant_proposed_action_count": proposed_dominant.get("count"),
        "dominant_proposed_action_share": proposed_dominant.get("share"),
        "max_dominant_proposed_action_share": _round(
            max_dominant_proposed_action_share
        ),
        "proposed_action_non_collapsed": proposed_action_non_collapsed,
    }


def action_ranking_diagnostics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    preference_counts: Counter[str] = Counter()
    safe_over_unsafe_counts: Counter[str] = Counter()
    unavailable_counts: Counter[str] = Counter()
    available_counts: Counter[str] = Counter()
    pair_count = 0
    tie_count = 0
    for row in rows:
        targets = _available_targets(row)
        for target in _list_of_mappings(row.get("action_value_targets")):
            action = str(target.get("action", ""))
            if action not in ACTION_NAMES:
                continue
            if target.get("target_available") is True:
                available_counts.update([action])
            else:
                unavailable_counts.update([action])
        for left_index, left in enumerate(targets):
            for right in targets[left_index + 1 :]:
                left_action = str(left.get("action", ""))
                right_action = str(right.get("action", ""))
                left_value = _float(left.get("value_target"))
                right_value = _float(right.get("value_target"))
                pair_count += 1
                if left_value > right_value:
                    preference_counts.update([f"{left_action}>{right_action}"])
                    if left.get("safe_target") is True and right.get("safe_target") is not True:
                        safe_over_unsafe_counts.update([f"{left_action}>{right_action}"])
                elif right_value > left_value:
                    preference_counts.update([f"{right_action}>{left_action}"])
                    if right.get("safe_target") is True and left.get("safe_target") is not True:
                        safe_over_unsafe_counts.update([f"{right_action}>{left_action}"])
                else:
                    tie_count += 1
    total_target_slots = len(rows) * len(ACTION_NAMES)
    unavailable_total = sum(int(count) for count in unavailable_counts.values())
    return {
        "policy": "m3_carrion_survivor_continuation_v159_action_ranking_diagnostics_v1",
        "pairwise_action_preference_count": int(pair_count - tie_count),
        "pairwise_action_tie_count": int(tie_count),
        "pairwise_action_total_count": int(pair_count),
        "pairwise_action_preference_counts": dict(
            sorted(preference_counts.items())
        ),
        "safe_over_unsafe_preference_counts": dict(
            sorted(safe_over_unsafe_counts.items())
        ),
        "per_action_value_target_available_counts": dict(
            sorted(_positive_counter(available_counts).items())
        ),
        "per_action_value_target_unavailable_counts": dict(
            sorted(_positive_counter(unavailable_counts).items())
        ),
        "unavailable_target_count": int(unavailable_total),
        "unavailable_target_share": _safe_rate(unavailable_total, total_target_slots),
    }


def shadow_coverage_diagnostics(
    rows: Sequence[Mapping[str, object]],
    *,
    min_exact_shadow_safe_hit_share: float = DEFAULT_MIN_EXACT_SHADOW_SAFE_HIT_SHARE,
    min_exact_shadow_minus_best_trivial_coverage_share: float = (
        DEFAULT_MIN_EXACT_SHADOW_MINUS_BEST_TRIVIAL_COVERAGE_SHARE
    ),
    min_near_exact_mask_comparator_coverage_share: float = (
        DEFAULT_MIN_NEAR_EXACT_MASK_COMPARATOR_COVERAGE_SHARE
    ),
    min_near_exact_mask_shadow_safe_hit_share: float = (
        DEFAULT_MIN_NEAR_EXACT_MASK_SHADOW_SAFE_HIT_SHARE
    ),
) -> dict[str, object]:
    exact_feature_groups: dict[str, list[int]] = defaultdict(list)
    mask_groups: dict[str, list[int]] = defaultdict(list)
    fixed_action_safe_counts: Counter[str] = Counter()
    action_order_first_safe_hits = 0
    exact_shadow_safe_hits = 0
    for index, row in enumerate(rows):
        exact_feature_groups[_exact_feature_digest(row)].append(index)
        mask_groups[stable_payload_digest(_complete_bool_mask(row.get("public_action_mask")))].append(
            index
        )
        safe_set = set(_safe_action_set(row))
        fixed_action_safe_counts.update(safe_set)
        first_public = _first_public_action(row)
        if first_public in safe_set:
            action_order_first_safe_hits += 1
        proposed = _top_value_target(row)
        if proposed and str(proposed.get("action", "")) in safe_set:
            exact_shadow_safe_hits += 1
    best_fixed = _dominant_count_share(_positive_counter(fixed_action_safe_counts))
    best_fixed_share = _safe_rate(_int(best_fixed.get("count")), len(rows))
    exact_shadow_share = _safe_rate(exact_shadow_safe_hits, len(rows))
    exact_minus_best_trivial = _round(exact_shadow_share - best_fixed_share)

    near_covered = 0
    near_hits = 0
    near_predictions: Counter[str] = Counter()
    for index, row in enumerate(rows):
        group = mask_groups.get(
            stable_payload_digest(_complete_bool_mask(row.get("public_action_mask"))),
            [],
        )
        others = [other for other in group if other != index]
        if not others:
            continue
        near_covered += 1
        prediction = _near_exact_mask_prediction([rows[other] for other in others])
        if prediction:
            near_predictions.update([prediction])
            if prediction in _safe_action_set(row):
                near_hits += 1
    near_coverage_share = _safe_rate(near_covered, len(rows))
    near_hit_share_all_rows = _safe_rate(near_hits, len(rows))
    near_hit_share_covered_rows = _safe_rate(near_hits, near_covered)
    exact_useful = (
        exact_shadow_share >= float(min_exact_shadow_safe_hit_share)
        and exact_minus_best_trivial
        >= float(min_exact_shadow_minus_best_trivial_coverage_share)
    )
    near_useful = (
        near_coverage_share >= float(min_near_exact_mask_comparator_coverage_share)
        and near_hit_share_all_rows
        >= float(min_near_exact_mask_shadow_safe_hit_share)
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v159_shadow_coverage_diagnostics_v1",
        "exact_public_feature_group_count": len(exact_feature_groups),
        "exact_public_feature_duplicate_group_count": sum(
            1 for indexes in exact_feature_groups.values() if len(indexes) > 1
        ),
        "exact_shadow_safe_hit_count": exact_shadow_safe_hits,
        "exact_shadow_safe_hit_share": exact_shadow_share,
        "min_exact_shadow_safe_hit_share": _round(min_exact_shadow_safe_hit_share),
        "best_fixed_action": best_fixed.get("key"),
        "best_fixed_action_safe_coverage_count": best_fixed.get("count"),
        "best_fixed_action_safe_coverage_share": best_fixed_share,
        "action_order_first_public_safe_hit_count": action_order_first_safe_hits,
        "action_order_first_public_safe_hit_share": _safe_rate(
            action_order_first_safe_hits,
            len(rows),
        ),
        "exact_shadow_minus_best_trivial_coverage_share": exact_minus_best_trivial,
        "min_exact_shadow_minus_best_trivial_coverage_share": _round(
            min_exact_shadow_minus_best_trivial_coverage_share
        ),
        "exact_shadow_coverage_useful": exact_useful,
        "near_exact_mask_group_count": len(mask_groups),
        "near_exact_mask_comparator_row_count": near_covered,
        "near_exact_mask_comparator_coverage_share": near_coverage_share,
        "near_exact_mask_shadow_safe_hit_count": near_hits,
        "near_exact_mask_shadow_safe_hit_share_all_rows": near_hit_share_all_rows,
        "near_exact_mask_shadow_safe_hit_share_covered_rows": near_hit_share_covered_rows,
        "near_exact_mask_prediction_counts": dict(sorted(near_predictions.items())),
        "min_near_exact_mask_comparator_coverage_share": _round(
            min_near_exact_mask_comparator_coverage_share
        ),
        "min_near_exact_mask_shadow_safe_hit_share": _round(
            min_near_exact_mask_shadow_safe_hit_share
        ),
        "near_exact_mask_shadow_coverage_useful": near_useful,
        "shadow_coverage_useful": exact_useful and near_useful,
    }


def _top_level_classification(
    *,
    source_validation: Mapping[str, object],
    row_contract: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    action_support: Mapping[str, object],
    shadow_coverage: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v159_scorer_readiness_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if row_contract.get("passed") is not True:
        return prefix + "row_contract_invalid_closed_no_training"
    if leakage_scan.get("passed") is not True:
        return prefix + "leakage_failed_closed_no_training"
    if action_support.get("action_support_non_collapsed") is not True:
        return prefix + "action_support_collapsed_closed_no_training"
    if action_support.get("proposed_action_non_collapsed") is not True:
        return prefix + "proposed_action_collapsed_closed_no_training"
    if shadow_coverage.get("shadow_coverage_useful") is not True:
        return prefix + "shadow_coverage_insufficient_closed_no_training"
    return prefix + "recommends_separate_opt_in_training_diagnostic_no_training"


def _route_recommendation(classification: str) -> dict[str, object]:
    recommended = classification.endswith(
        "recommends_separate_opt_in_training_diagnostic_no_training"
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v159_route_recommendation_v1",
        "future_opt_in_training_diagnostic_recommended": recommended,
        "recommended_next_route": (
            "separate_opt_in_action_value_scorer_training_diagnostic"
            if recommended
            else "keep_v158_target_dataset_closed_until_readiness_failure_is_resolved"
        ),
        "training_authorized": False,
        "this_command_trained_model": False,
        "future_training_requires_separate_command_and_contract": True,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "runtime_action_selection_changed": False,
    }


def _available_targets(row: Mapping[str, object]) -> list[dict[str, object]]:
    targets = []
    for target in _list_of_mappings(row.get("action_value_targets")):
        if (
            target.get("target_available") is True
            and target.get("public_mask") is True
            and _finite_number(target.get("value_target"))
        ):
            targets.append(dict(target))
    return targets


def _top_value_target(row: Mapping[str, object]) -> dict[str, object]:
    targets = _available_targets(row)
    if not targets:
        return {}
    return sorted(
        targets,
        key=lambda target: (
            -_float(target.get("value_target")),
            target.get("safe_target") is not True,
            target.get("robust_safe_action") is not True,
            _action_order(str(target.get("action", ""))),
        ),
    )[0]


def _near_exact_mask_prediction(rows: Sequence[Mapping[str, object]]) -> str:
    value_sums: dict[str, float] = defaultdict(float)
    value_counts: Counter[str] = Counter()
    safe_counts: Counter[str] = Counter()
    robust_counts: Counter[str] = Counter()
    for row in rows:
        safe_counts.update(_safe_action_set(row))
        for target in _available_targets(row):
            action = str(target.get("action", ""))
            value_sums[action] += _float(target.get("value_target"))
            value_counts.update([action])
            if target.get("robust_safe_action") is True:
                robust_counts.update([action])
    if not value_counts:
        return ""
    actions = sorted(
        value_counts,
        key=lambda action: (
            -_safe_rate(value_sums[action], value_counts[action]),
            -int(safe_counts.get(action, 0)),
            -int(robust_counts.get(action, 0)),
            _action_order(action),
        ),
    )
    return actions[0]


def _targets_by_action(row: Mapping[str, object]) -> dict[str, dict[str, object]]:
    targets: dict[str, dict[str, object]] = {}
    for target in _list_of_mappings(row.get("action_value_targets")):
        action = str(target.get("action", ""))
        if action in targets:
            targets[f"{action}#duplicate"] = dict(target)
        else:
            targets[action] = dict(target)
    return targets


def _validate_public_action_mask(mask: Mapping[str, object]) -> list[str]:
    failures = []
    if set(mask) != set(ACTION_NAMES):
        failures.append("public_action_mask_action_set_mismatch")
    for action in ACTION_NAMES:
        if not isinstance(mask.get(action), bool):
            failures.append("public_action_mask_non_bool_value")
            break
    return failures


def _validate_target_numeric_fields(
    *,
    failures: list[dict[str, object]],
    row_index: int,
    action: str,
    target: Mapping[str, object],
    target_available: bool,
) -> None:
    for field in ("score_target", "value_target"):
        value = target.get(field)
        if target_available:
            if not _finite_number(value):
                _append_failure(
                    failures,
                    row_index,
                    f"{field}_not_finite_for_available_target",
                    action=action,
                )
        elif value is not None:
            _append_failure(
                failures,
                row_index,
                f"{field}_present_for_unavailable_target",
                action=action,
            )


def _safe_action_set(row: Mapping[str, object]) -> list[str]:
    value = row.get("safe_action_set")
    if not isinstance(value, list):
        return []
    return sorted(
        {str(action) for action in value if str(action) in ACTION_NAMES},
        key=_action_order,
    )


def _complete_bool_mask(mask: object) -> dict[str, bool]:
    payload = _mapping(mask)
    return {action: bool(payload.get(action, False)) for action in ACTION_NAMES}


def _exact_feature_digest(row: Mapping[str, object]) -> str:
    return stable_payload_digest(
        {
            "trainable_public_features": _mapping(
                row.get("trainable_public_features")
            ),
            "public_action_mask": _complete_bool_mask(row.get("public_action_mask")),
        }
    )


def _first_public_action(row: Mapping[str, object]) -> str:
    mask = _mapping(row.get("public_action_mask"))
    for action in ACTION_NAMES:
        if mask.get(action) is True:
            return action
    return ""


def _append_failure(
    failures: list[dict[str, object]],
    row_index: int,
    reason: str,
    *,
    action: str | None = None,
) -> None:
    payload: dict[str, object] = {"row_index": int(row_index), "reason": reason}
    if action is not None:
        payload["action"] = action
    failures.append(payload)


def _positive_counter(counter: Counter[str]) -> Counter[str]:
    return Counter(
        {
            key: int(value)
            for key, value in counter.items()
            if int(value) > 0 and str(key)
        }
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _safe_rate(numerator: float | int, denominator: float | int) -> float:
    denominator_float = float(denominator)
    if denominator_float <= 0.0:
        return 0.0
    return _round(float(numerator) / denominator_float)
