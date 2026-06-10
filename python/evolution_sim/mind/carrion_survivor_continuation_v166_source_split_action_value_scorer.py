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
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION,
    exact_digest_validation_report,
    target_dataset_leakage_scan,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
    load_v154_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    _complete_bool_mask,
    _first_public_action,
    _safe_action_set,
    _top_value_target,
    _validate_public_action_mask,
)
from evolution_sim.mind.carrion_survivor_continuation_v160_action_value_scorer import (
    K_NEIGHBORS,
    _artifact_action_value_targets,
    _best_fixed_safe_action,
    _feature_vectors,
    _finite_number,
    _predict_1nn_action,
    _safe_rate,
    _sparse_vector,
)
from evolution_sim.mind.carrion_survivor_continuation_v163_tied_set_branch_target_expansion import (
    STRICT_BROAD_SEEDS,
)
from evolution_sim.mind.carrion_survivor_continuation_v165_preterminal_target_dataset_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V165_REPORT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V165_DATASET_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_source_split_action_value_scorer_training_diagnostic_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_source_split_action_value_scorer_diagnostic_artifact_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_source_split_action_value_scorer_training_v1"
)
EXPECTED_V165_CLASSIFICATION = "expanded_target_dataset_support_ready_no_training"
EXPECTED_V165_EXACT_DIGEST = (
    "74b5d4bde49fb594fabb2ac7982ab872f9aa154c9b470a296f57a921f3b5134b"
)
EXPECTED_V165_DATASET_DIGEST = (
    "e022efbf5eb88b688d052dc3532626be5bd1484109d2369829f34293fa1fe091"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v166-carrion-survivor-continuation-source-split-action-value-scorer-training.json"
)
DEFAULT_ARTIFACT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v166-carrion-survivor-continuation-source-split-action-value-scorer-artifact.json"
)
DEFAULT_MAX_DOMINANT_PREDICTED_ACTION_SHARE = 0.50
DEFAULT_MIN_SAFE_HIT_MARGIN_OVER_BEST_TRIVIAL = 0.05


class CarrionSurvivorContinuationV166SourceSplitActionValueScorerError(ValueError):
    pass


def run_carrion_survivor_continuation_v166_source_split_action_value_scorer(
    *,
    v165_report_path: str | Path = DEFAULT_V165_REPORT_PATH,
    v165_dataset_path: str | Path = DEFAULT_V165_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    artifact_output_path: str | Path = DEFAULT_ARTIFACT_OUTPUT_PATH,
    expected_v165_exact_digest: str | None = EXPECTED_V165_EXACT_DIGEST,
    expected_v165_dataset_digest: str | None = EXPECTED_V165_DATASET_DIGEST,
    source_seed_diagnostic_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    max_dominant_predicted_action_share: float = (
        DEFAULT_MAX_DOMINANT_PREDICTED_ACTION_SHARE
    ),
    min_safe_hit_margin_over_best_trivial: float = (
        DEFAULT_MIN_SAFE_HIT_MARGIN_OVER_BEST_TRIVIAL
    ),
) -> dict[str, object]:
    v165_report = load_json_report(v165_report_path)
    rows = load_v154_dataset(v165_dataset_path)
    source_validation = validate_v166_sources(
        v165_report,
        rows=rows,
        expected_v165_exact_digest=expected_v165_exact_digest,
        expected_v165_dataset_digest=expected_v165_dataset_digest,
        source_seed_diagnostic_seeds=source_seed_diagnostic_seeds,
    )
    row_contract = validate_v166_mixed_target_rows(
        rows,
        source_seed_diagnostic_seeds=source_seed_diagnostic_seeds,
    )
    leakage_scan = target_dataset_leakage_scan(rows)
    artifact = build_diagnostic_scorer_artifact(
        rows=rows,
        source_validation=source_validation,
        row_contract=row_contract,
        leakage_scan=leakage_scan,
    )
    write_json(artifact_output_path, artifact)
    source_split = source_split_diagnostics(
        rows=rows,
        source_seed_diagnostic_seeds=source_seed_diagnostic_seeds,
        max_dominant_predicted_action_share=max_dominant_predicted_action_share,
        min_safe_hit_margin_over_best_trivial=(
            min_safe_hit_margin_over_best_trivial
        ),
    )
    classification = _top_level_classification(
        source_validation=source_validation,
        row_contract=row_contract,
        leakage_scan=leakage_scan,
        source_split=source_split,
    )
    artifact_digest = str(artifact.get("exact_digest") or "")
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_REPORT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_POLICY,
        "contract": {
            "diagnostics_only": True,
            "serialized_scorer_artifact_created": True,
            "runtime_artifact_created": False,
            "runtime_policy_integration_allowed": False,
            "training_scope": "diagnostics_only_source_split_public_action_value_scorer",
            "source_split_evaluation_required": True,
            "uses_only_public_trainable_features": True,
            "uses_public_action_masks": True,
            "uses_safe_action_set_targets": True,
            "uses_action_value_targets": True,
            "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
            "uses_runtime_actions_as_trainable_input": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_future_outcome_as_trainable_input": False,
            "uses_branch_reason_as_trainable_input": False,
            "training_authorized_for_runtime": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "shadow_live_ab_allowed": False,
            "live_runtime_override_allowed": False,
            "threshold_tuning_recommended": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
        },
        "inputs": {
            "v165_report": str(v165_report_path),
            "v165_dataset": str(v165_dataset_path),
            "artifact_output": str(artifact_output_path),
            "expected_v165_exact_digest": expected_v165_exact_digest,
            "expected_v165_dataset_digest": expected_v165_dataset_digest,
            "source_seed_diagnostic_seeds": [
                int(seed) for seed in source_seed_diagnostic_seeds
            ],
            "k_neighbors": K_NEIGHBORS,
            "max_dominant_predicted_action_share": _round(
                max_dominant_predicted_action_share
            ),
            "min_safe_hit_margin_over_best_trivial": _round(
                min_safe_hit_margin_over_best_trivial
            ),
        },
        "source_validation": source_validation,
        "row_contract_validation": row_contract,
        "leakage_scan": leakage_scan,
        "artifact": {
            "path": str(artifact_output_path),
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION
            ),
            "scorer_type": "public_feature_mask_1nn_action_value_scorer_v1",
            "diagnostics_only": True,
            "runtime_artifact": False,
            "artifact_digest": artifact_digest,
        },
        "source_split_diagnostics": source_split,
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "dataset_digest": stable_payload_digest(rows),
        "artifact_created": True,
        "runtime_artifact_created": False,
        "diagnostic_training_ran": True,
        "training_ran": True,
        "shadow_live_ab_ran": False,
        "training_authorized_for_runtime": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "threshold_tuning_recommended": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v166_sources(
    v165_report: Mapping[str, object],
    *,
    rows: Sequence[Mapping[str, object]],
    expected_v165_exact_digest: str | None = EXPECTED_V165_EXACT_DIGEST,
    expected_v165_dataset_digest: str | None = EXPECTED_V165_DATASET_DIGEST,
    source_seed_diagnostic_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
) -> dict[str, object]:
    failures: list[str] = []
    if (
        v165_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_SCHEMA_VERSION
    ):
        failures.append("v165_schema_version_mismatch")
    if (
        v165_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_POLICY
    ):
        failures.append("v165_policy_mismatch")
    observed_classification = _mapping(v165_report.get("classification")).get(
        "primary"
    )
    if observed_classification != EXPECTED_V165_CLASSIFICATION:
        failures.append("v165_unexpected_classification")
    exact_digest_validation = exact_digest_validation_report(v165_report)
    if exact_digest_validation.get("passed") is not True:
        failures.append("v165_exact_digest_mismatch")
    observed_exact = str(v165_report.get("exact_digest") or "")
    if expected_v165_exact_digest and observed_exact != expected_v165_exact_digest:
        failures.append("v165_unexpected_exact_digest")
    dataset_digest = stable_payload_digest(rows)
    reported_dataset_digest = _mapping(v165_report.get("dataset")).get(
        "dataset_digest"
    )
    if expected_v165_dataset_digest and dataset_digest != expected_v165_dataset_digest:
        failures.append("v165_unexpected_dataset_digest")
    if expected_v165_dataset_digest and reported_dataset_digest != expected_v165_dataset_digest:
        failures.append("v165_reported_unexpected_dataset_digest")
    if reported_dataset_digest != dataset_digest:
        failures.append("v165_dataset_digest_mismatch")
    if _mapping(v165_report.get("leakage_scan")).get("passed") is not True:
        failures.append("v165_leakage_scan_failed")
    future_policy = _v165_future_evaluation_policy_validation(
        v165_report,
        source_seed_diagnostic_seeds=source_seed_diagnostic_seeds,
    )
    if future_policy.get("passed") is not True:
        failures.append("v165_future_evaluation_policy_mismatch")
    lifecycle = _v165_lifecycle_runtime_authorization_scan(v165_report)
    if lifecycle.get("passed") is not True:
        failures.append("v165_lifecycle_runtime_authorization_not_false")
    return {
        "policy": "m3_carrion_survivor_continuation_v166_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v165_classification": EXPECTED_V165_CLASSIFICATION,
        "observed_v165_classification": observed_classification,
        "expected_v165_exact_digest": expected_v165_exact_digest,
        "observed_v165_exact_digest": observed_exact,
        "v165_exact_digest_validation": exact_digest_validation,
        "expected_v165_dataset_digest": expected_v165_dataset_digest,
        "dataset_digest": dataset_digest,
        "v165_reported_dataset_digest": reported_dataset_digest,
        "v165_leakage_scan_passed": _mapping(v165_report.get("leakage_scan")).get(
            "passed"
        ),
        "future_evaluation_policy_validation": future_policy,
        "lifecycle_runtime_authorization": lifecycle,
        "row_count": len(rows),
    }


def validate_v166_mixed_target_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    source_seed_diagnostic_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    schema_counts: Counter[str] = Counter()
    source_seed_counts: Counter[int] = Counter()
    base_fallback_rows = 0
    safe_target_count = 0
    target_available_count = 0
    trainable_payloads = trainable_feature_payloads(rows)
    for row_index, row in enumerate(rows):
        schema = str(row.get("schema_version") or "")
        schema_counts.update([schema])
        if schema not in _allowed_row_schemas():
            _append_row_failure(failures, row_index, "row_schema_version_mismatch")
        features = _mapping(row.get("trainable_public_features"))
        if not features:
            _append_row_failure(failures, row_index, "trainable_public_features_missing")
        mask = _mapping(row.get("public_action_mask"))
        for reason in _validate_public_action_mask(mask):
            _append_row_failure(failures, row_index, reason)
        feature_mask = _mapping(features.get("action_mask"))
        if feature_mask and _complete_bool_mask(feature_mask) != _complete_bool_mask(mask):
            _append_row_failure(failures, row_index, "feature_action_mask_mismatch")
        safe_set = _safe_action_set(row)
        if not safe_set:
            _append_row_failure(failures, row_index, "safe_action_set_empty")
        for action in safe_set:
            if bool(mask.get(action, False)) is not True:
                _append_row_failure(
                    failures,
                    row_index,
                    "safe_action_not_public_mask_supported",
                    action=action,
                )
        targets = _list_of_mappings(row.get("action_value_targets"))
        targets_by_action = _targets_by_action(row)
        if len(targets) != len(ACTION_NAMES):
            _append_row_failure(failures, row_index, "action_value_target_count_mismatch")
        if set(targets_by_action) != set(ACTION_NAMES):
            _append_row_failure(
                failures,
                row_index,
                "action_value_target_action_set_mismatch",
            )
        if [str(target.get("action", "")) for target in targets] != list(ACTION_NAMES):
            _append_row_failure(failures, row_index, "action_value_target_order_mismatch")
        for action in ACTION_NAMES:
            target = _mapping(targets_by_action.get(action))
            target_public = target.get("public_mask") is True
            row_public = bool(mask.get(action, False))
            if target and target_public != row_public:
                _append_row_failure(
                    failures,
                    row_index,
                    "target_public_mask_mismatch",
                    action=action,
                )
            available = target.get("target_available") is True
            safe_target = target.get("safe_target") is True
            target_available_count += int(available)
            safe_target_count += int(safe_target)
            if available and not target_public:
                _append_row_failure(
                    failures,
                    row_index,
                    "available_target_not_public_mask_supported",
                    action=action,
                )
            if safe_target != (action in safe_set):
                _append_row_failure(
                    failures,
                    row_index,
                    "safe_target_mismatch",
                    action=action,
                )
            if safe_target and not available:
                _append_row_failure(
                    failures,
                    row_index,
                    "safe_target_unavailable",
                    action=action,
                )
            value = target.get("value_target")
            if available:
                if not _finite_number(value):
                    _append_row_failure(
                        failures,
                        row_index,
                        "value_target_not_finite_for_available_target",
                        action=action,
                    )
            elif value is not None:
                _append_row_failure(
                    failures,
                    row_index,
                    "value_target_present_for_unavailable_target",
                    action=action,
                )
        classification = str(row.get("target_classification", ""))
        robust_winner = str(row.get("robust_winner_action") or "")
        if classification == "unique_robust_winner":
            if robust_winner not in safe_set:
                _append_row_failure(
                    failures,
                    row_index,
                    "unique_winner_missing_from_safe_action_set",
                    action=robust_winner,
                )
            if len(safe_set) != 1:
                _append_row_failure(failures, row_index, "unique_winner_safe_set_not_single")
        elif robust_winner:
            _append_row_failure(failures, row_index, "unexpected_robust_winner_action")
        if classification == "multi_action_safe_set" and len(safe_set) < 2:
            _append_row_failure(failures, row_index, "multi_action_safe_set_too_small")
        if schema == M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION:
            metadata = _mapping(row.get("metadata"))
            source_seed = _int(metadata.get("seed"))
            if not metadata:
                _append_row_failure(failures, row_index, "v165_metadata_missing")
            if source_seed <= 0:
                _append_row_failure(failures, row_index, "v165_source_seed_missing")
            else:
                source_seed_counts.update([source_seed])
            if metadata.get("source_seed_is_support_provenance_not_future_promotion_holdout") is not True:
                _append_row_failure(
                    failures,
                    row_index,
                    "v165_source_seed_not_marked_support_provenance",
                )
            if metadata.get("runtime_requested_action_used_as_scorer_input") is not False:
                _append_row_failure(
                    failures,
                    row_index,
                    "v165_runtime_action_trainable_input_flag_not_false",
                )
            if metadata.get("future_outcomes_used_as_trainable_input") is not False:
                _append_row_failure(
                    failures,
                    row_index,
                    "v165_future_outcome_trainable_input_flag_not_false",
                )
        else:
            base_fallback_rows += 1
    trainable_payload_digest = stable_payload_digest(trainable_payloads)
    expected_seeds = sorted({int(seed) for seed in source_seed_diagnostic_seeds})
    return {
        "policy": "m3_carrion_survivor_continuation_v166_mixed_row_contract_validation_v1",
        "passed": not failures and bool(rows),
        "failure_count": len(failures),
        "failures": failures[:96],
        "row_count": len(rows),
        "schema_counts": dict(sorted(schema_counts.items())),
        "base_row_fallback_count": base_fallback_rows,
        "preterminal_source_seed_row_counts": {
            str(seed): int(source_seed_counts.get(seed, 0))
            for seed in expected_seeds
            if int(source_seed_counts.get(seed, 0)) > 0
        },
        "preterminal_source_seed_count": len(source_seed_counts),
        "safe_target_count": safe_target_count,
        "target_available_count": target_available_count,
        "trainable_feature_payload_count": len(trainable_payloads),
        "trainable_feature_payload_digest": trainable_payload_digest,
    }


def trainable_feature_payloads(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "trainable_public_features": deepcopy(
                _mapping(row.get("trainable_public_features"))
            ),
            "public_action_mask": deepcopy(_complete_bool_mask(row.get("public_action_mask"))),
        }
        for row in rows
    ]


def build_diagnostic_scorer_artifact(
    *,
    rows: Sequence[Mapping[str, object]],
    source_validation: Mapping[str, object],
    row_contract: Mapping[str, object],
    leakage_scan: Mapping[str, object],
) -> dict[str, object]:
    vectors, feature_keys, normalization = _feature_vectors(rows)
    training_rows = []
    for row_index, row in enumerate(rows):
        training_rows.append(
            {
                "row_index": row_index,
                "row_schema_version": row.get("schema_version"),
                "feature_vector": _sparse_vector(vectors[row_index]),
                "public_action_mask": _complete_bool_mask(row.get("public_action_mask")),
                "safe_action_set": _safe_action_set(row),
                "action_value_targets": _artifact_action_value_targets(row),
            }
        )
    artifact = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_ARTIFACT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V166_SOURCE_SPLIT_ACTION_VALUE_SCORER_POLICY,
        "scorer_type": "public_feature_mask_1nn_action_value_scorer_v1",
        "contract": {
            "diagnostics_only": True,
            "runtime_artifact": False,
            "runtime_policy_integration_allowed": False,
            "training_scope": "diagnostics_only_source_split_public_features_action_value_targets",
            "k_neighbors": K_NEIGHBORS,
            "uses_only_public_trainable_features": True,
            "uses_public_action_masks": True,
            "uses_safe_action_set_targets": True,
            "uses_action_value_targets": True,
            "uses_source_group_as_trainable_input": False,
            "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
            "uses_runtime_actions_as_trainable_input": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_future_outcome_as_trainable_input": False,
            "runtime_action_selection_changed": False,
        },
        "source": {
            "dataset_digest": stable_payload_digest(rows),
            "source_validation_passed": source_validation.get("passed") is True,
            "row_contract_validation_passed": row_contract.get("passed") is True,
            "leakage_scan_passed": leakage_scan.get("passed") is True,
            "row_count": len(rows),
        },
        "feature_keys": feature_keys,
        "normalization": normalization,
        "training_rows": training_rows,
    }
    artifact["exact_digest"] = _json_round_trip_digest(artifact)
    return artifact


def source_split_groups(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups = []
    fallback_index = 0
    for row_index, row in enumerate(rows):
        schema = str(row.get("schema_version") or "")
        if (
            schema
            == M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ):
            metadata = _mapping(row.get("metadata"))
            source_seed = _int(metadata.get("seed"))
            groups.append(
                {
                    "row_index": row_index,
                    "group_key": f"v165_source_seed:{source_seed}",
                    "group_type": "v165_preterminal_source_seed",
                    "source_seed": source_seed,
                    "fallback_group": False,
                }
            )
        else:
            groups.append(
                {
                    "row_index": row_index,
                    "group_key": f"v158_base_row_fallback:{fallback_index}",
                    "group_type": "v158_base_row_fallback",
                    "source_seed": None,
                    "fallback_group": True,
                }
            )
            fallback_index += 1
    return groups


def source_split_diagnostics(
    *,
    rows: Sequence[Mapping[str, object]],
    source_seed_diagnostic_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    max_dominant_predicted_action_share: float = (
        DEFAULT_MAX_DOMINANT_PREDICTED_ACTION_SHARE
    ),
    min_safe_hit_margin_over_best_trivial: float = (
        DEFAULT_MIN_SAFE_HIT_MARGIN_OVER_BEST_TRIVIAL
    ),
) -> dict[str, object]:
    vectors, feature_keys, normalization = _feature_vectors(rows)
    groups = source_split_groups(rows)
    group_by_row = {int(group["row_index"]): group for group in groups}
    group_row_indexes: dict[str, list[int]] = defaultdict(list)
    for group in groups:
        group_row_indexes[str(group["group_key"])].append(int(group["row_index"]))

    predictions: list[dict[str, object]] = []
    group_predictions: dict[str, list[dict[str, object]]] = defaultdict(list)
    prediction_counts: Counter[str] = Counter()
    safe_hit_count = 0
    unsupported_prediction_count = 0
    no_prediction_count = 0
    best_fixed_hit_count = 0
    first_public_hit_count = 0
    exact_top_hit_count = 0

    for row_index, row in enumerate(rows):
        group = group_by_row[row_index]
        group_key = str(group["group_key"])
        excluded = list(group_row_indexes[group_key])
        train_indexes = [
            index
            for index in range(len(rows))
            if str(group_by_row[index]["group_key"]) != group_key
        ]
        scorer_action, source, neighbor_indexes = _predict_1nn_action(
            rows=rows,
            vectors=vectors,
            feature_keys=feature_keys,
            normalization=normalization,
            row_index=row_index,
            train_indexes=train_indexes,
        )
        safe_set = set(_safe_action_set(row))
        public_mask = _complete_bool_mask(row.get("public_action_mask"))
        safe_hit = bool(scorer_action and scorer_action in safe_set)
        unsupported = bool(scorer_action and public_mask.get(scorer_action) is not True)
        if scorer_action:
            prediction_counts.update([scorer_action])
            safe_hit_count += int(safe_hit)
            unsupported_prediction_count += int(unsupported)
        else:
            no_prediction_count += 1
        best_fixed = _best_fixed_safe_action(rows, train_indexes, public_mask)
        first_public = _first_public_action(row)
        best_fixed_hit = best_fixed in safe_set
        first_public_hit = first_public in safe_set
        best_fixed_hit_count += int(best_fixed_hit)
        first_public_hit_count += int(first_public_hit)
        exact_top = str(_top_value_target(row).get("action", ""))
        exact_top_hit = exact_top in safe_set
        exact_top_hit_count += int(exact_top_hit)
        prediction = {
            "row_index": row_index,
            "source_group_key": group_key,
            "source_group_type": group["group_type"],
            "source_seed": group.get("source_seed"),
            "excluded_row_indexes": excluded,
            "train_row_count": len(train_indexes),
            "predicted_action": scorer_action,
            "prediction_source": source,
            "nearest_neighbor_row_indexes": neighbor_indexes,
            "safe_hit": safe_hit,
            "unsupported_prediction": unsupported,
            "safe_action_set": sorted(safe_set, key=_action_order),
            "best_fixed_safe_action_baseline": best_fixed,
            "best_fixed_safe_action_baseline_hit": best_fixed_hit,
            "first_public_action_baseline": first_public,
            "first_public_action_baseline_hit": first_public_hit,
            "target_leaky_exact_top_upper_bound_action": exact_top,
            "target_leaky_exact_top_upper_bound_safe_hit": exact_top_hit,
        }
        predictions.append(prediction)
        group_predictions[group_key].append(prediction)

    row_count = len(rows)
    dominant = _dominant_count_share(_positive_counter(prediction_counts))
    safe_hit_rate = _safe_rate(safe_hit_count, row_count)
    best_fixed_rate = _safe_rate(best_fixed_hit_count, row_count)
    first_public_rate = _safe_rate(first_public_hit_count, row_count)
    best_trivial_rate = max(best_fixed_rate, first_public_rate)
    margin = _round(safe_hit_rate - best_trivial_rate)
    unsupported_floor_passed = (
        unsupported_prediction_count == 0 and no_prediction_count == 0
    )
    dominant_share = _float(dominant.get("share"))
    dominance_floor_passed = (
        bool(prediction_counts)
        and dominant_share <= float(max_dominant_predicted_action_share)
    )
    margin_floor_passed = (
        margin >= float(min_safe_hit_margin_over_best_trivial)
    )
    per_group = _per_source_group_metrics(group_predictions)
    per_seed_rates = _per_source_seed_safe_hit_rates(
        per_group,
        source_seed_diagnostic_seeds=source_seed_diagnostic_seeds,
    )
    first_failing_seed = _first_zero_safe_hit_preterminal_seed(per_group)
    preterminal_source_seed_floor_passed = first_failing_seed is None
    base_fallback = _base_fallback_summary(per_group)
    return {
        "policy": "m3_carrion_survivor_continuation_v166_source_split_diagnostics_v1",
        "scorer_type": "public_feature_mask_1nn_action_value_scorer_v1",
        "k_neighbors": K_NEIGHBORS,
        "row_count": row_count,
        "group_count": len(per_group),
        "source_grouping": _source_grouping_summary(groups),
        "base_row_fallback": base_fallback,
        "prediction_count": sum(prediction_counts.values()),
        "no_prediction_count": no_prediction_count,
        "unsupported_prediction_count": unsupported_prediction_count,
        "unsupported_prediction_floor_passed": unsupported_floor_passed,
        "predicted_action_counts": dict(sorted(prediction_counts.items())),
        "dominant_predicted_action": dominant.get("key"),
        "dominant_predicted_action_count": dominant.get("count"),
        "dominant_predicted_action_share": dominant.get("share"),
        "max_dominant_predicted_action_share": _round(
            max_dominant_predicted_action_share
        ),
        "dominant_prediction_floor_passed": dominance_floor_passed,
        "safe_hit_count": safe_hit_count,
        "safe_hit_rate": safe_hit_rate,
        "best_fixed_safe_action_baseline_hit_count": best_fixed_hit_count,
        "best_fixed_safe_action_baseline_hit_rate": best_fixed_rate,
        "first_public_action_baseline_hit_count": first_public_hit_count,
        "first_public_action_baseline_hit_rate": first_public_rate,
        "best_trivial_baseline_hit_rate": best_trivial_rate,
        "safe_hit_margin_over_best_trivial": margin,
        "min_safe_hit_margin_over_best_trivial": _round(
            min_safe_hit_margin_over_best_trivial
        ),
        "safe_hit_margin_floor_passed": margin_floor_passed,
        "preterminal_source_seed_floor_passed": preterminal_source_seed_floor_passed,
        "first_zero_safe_hit_preterminal_source_seed": first_failing_seed,
        "per_source_seed_safe_hit_rates": per_seed_rates,
        "per_source_group_metrics": per_group,
        "target_leaky_exact_top_upper_bound": {
            "non_runtime": True,
            "target_leaky": True,
            "safe_hit_count": exact_top_hit_count,
            "safe_hit_rate": _safe_rate(exact_top_hit_count, row_count),
        },
        "floor_passed": (
            unsupported_floor_passed
            and dominance_floor_passed
            and margin_floor_passed
            and preterminal_source_seed_floor_passed
        ),
        "predictions": predictions,
    }


def _top_level_classification(
    *,
    source_validation: Mapping[str, object],
    row_contract: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    source_split: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v166_source_split_action_value_scorer_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_shadow"
    if row_contract.get("passed") is not True:
        return prefix + "row_contract_invalid_closed_no_shadow"
    if leakage_scan.get("passed") is not True:
        return prefix + "leakage_failed_closed_no_shadow"
    if source_split.get("unsupported_prediction_floor_passed") is not True:
        return prefix + "unsupported_predictions_closed_no_shadow"
    if source_split.get("dominant_prediction_floor_passed") is not True:
        return prefix + "predicted_action_dominance_closed_no_shadow"
    if source_split.get("safe_hit_margin_floor_passed") is not True:
        return prefix + "source_split_generalization_failed_closed_archive_source_expansion"
    if source_split.get("preterminal_source_seed_floor_passed") is not True:
        return prefix + "preterminal_source_seed_zero_safe_hits_closed_source_expansion"
    return prefix + "diagnostic_scorer_ready_for_future_diagnostics_only_shadow_eval"


def _route_recommendation(classification: str) -> dict[str, object]:
    shadow_ready = classification.endswith(
        "diagnostic_scorer_ready_for_future_diagnostics_only_shadow_eval"
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v166_route_recommendation_v1",
        "future_diagnostics_only_shadow_evaluation_recommended": shadow_ready,
        "recommended_next_route": (
            "separate_opt_in_diagnostics_only_shadow_scorer_evaluation_without_runtime_override"
            if shadow_ready
            else "expand_carrion_survivor_continuation_source_support_before_shadow_eval"
        ),
        "threshold_tuning_recommended": False,
        "live_ab_allowed": False,
        "runtime_policy_integration_allowed": False,
        "runtime_override_path_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "runtime_action_selection_changed": False,
    }


def _v165_future_evaluation_policy_validation(
    report: Mapping[str, object],
    *,
    source_seed_diagnostic_seeds: Sequence[int],
) -> dict[str, object]:
    policy = _mapping(report.get("future_evaluation_policy"))
    expected_seeds = sorted({int(seed) for seed in source_seed_diagnostic_seeds})
    support_seeds = sorted(_int(seed) for seed in _list_like(policy.get("support_provenance_seeds")))
    disallowed = sorted(
        _int(seed)
        for seed in _list_like(policy.get("disallowed_future_promotion_heldout_seed_reuse"))
    )
    failures = []
    if policy.get("leave_source_seed_out_diagnostics_required") is not True:
        failures.append("leave_source_seed_out_not_required")
    if policy.get("new_held_out_broad_seeds_required") is not True:
        failures.append("new_held_out_broad_seeds_not_required")
    if (
        policy.get("v164_strict_broad_seeds_become_support_provenance_seeds_not_heldout")
        is not True
    ):
        failures.append("strict_broad_seeds_not_marked_support_provenance_only")
    if (
        policy.get("v164_support_provenance_seed_exclusion_required_for_future_promotion")
        is not True
    ):
        failures.append("support_provenance_seed_exclusion_not_required")
    if support_seeds != expected_seeds:
        failures.append("support_provenance_seed_set_mismatch")
    if disallowed != expected_seeds:
        failures.append("disallowed_future_promotion_seed_set_mismatch")
    if policy.get("promotion_authorized") is not False:
        failures.append("future_policy_promotion_authorized_not_false")
    if policy.get("live_ab_allowed") is not False:
        failures.append("future_policy_live_ab_allowed_not_false")
    return {
        "policy": "m3_carrion_survivor_continuation_v166_future_evaluation_policy_validation_v1",
        "passed": not failures,
        "failures": failures,
        "expected_support_provenance_seeds": expected_seeds,
        "observed_support_provenance_seeds": support_seeds,
        "observed_disallowed_future_promotion_seeds": disallowed,
    }


def _v165_lifecycle_runtime_authorization_scan(
    report: Mapping[str, object],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    _scan_false_fields(
        failures,
        source="v165",
        payload=report,
        fields=(
            "training_authorized",
            "training_authorized_for_runtime",
            "artifact_created",
            "model_artifact_creation_allowed_for_runtime",
            "runtime_artifact_created",
            "runtime_policy_integration_allowed",
            "runtime_policy_change_authorized",
            "live_ab_allowed",
            "live_ab_ran",
            "shadow_live_ab_allowed",
            "live_runtime_override_allowed",
            "runtime_override_path_allowed",
            "runtime_override_path_created",
            "promotion_authorized",
            "runtime_promotion_allowed",
            "threshold_tuning_recommended",
            "runtime_action_selection_changed",
        ),
    )
    _scan_false_fields(
        failures,
        source="v165.contract",
        payload=_mapping(report.get("contract")),
        fields=(
            "training_authorized",
            "training_authorized_for_runtime",
            "training_ran",
            "model_artifact_creation_allowed_for_runtime",
            "runtime_artifact_created",
            "runtime_policy_integration_allowed",
            "runtime_policy_change_authorized",
            "live_ab_allowed",
            "shadow_live_ab_allowed",
            "live_runtime_override_allowed",
            "runtime_override_path_allowed",
            "runtime_override_path_created",
            "promotion_authorized",
            "runtime_promotion_allowed",
            "threshold_tuning_recommended",
            "runtime_action_selection_changed",
        ),
    )
    _scan_false_fields(
        failures,
        source="v165.route_recommendation",
        payload=_mapping(report.get("route_recommendation")),
        fields=(
            "training_authorized",
            "training_authorized_for_runtime",
            "live_ab_allowed",
            "shadow_live_ab_allowed",
            "live_runtime_override_allowed",
            "runtime_override_path_allowed",
            "promotion_authorized",
            "runtime_policy_integration_allowed",
            "runtime_policy_change_authorized",
            "runtime_action_selection_changed",
            "threshold_tuning_recommended",
        ),
    )
    _scan_false_fields(
        failures,
        source="v165.future_evaluation_policy",
        payload=_mapping(report.get("future_evaluation_policy")),
        fields=("promotion_authorized", "live_ab_allowed"),
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v166_lifecycle_runtime_authorization_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def _scan_false_fields(
    failures: list[dict[str, object]],
    *,
    source: str,
    payload: Mapping[str, object],
    fields: Sequence[str],
) -> None:
    for field in fields:
        if field in payload and payload.get(field) is not False:
            failures.append(
                {
                    "source": source,
                    "field": field,
                    "observed": payload.get(field),
                    "expected": False,
                }
            )


def _per_source_group_metrics(
    group_predictions: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    metrics = []
    for group_key in sorted(group_predictions):
        predictions = list(group_predictions[group_key])
        counts = Counter(
            str(prediction.get("predicted_action"))
            for prediction in predictions
            if str(prediction.get("predicted_action") or "")
        )
        dominant = _dominant_count_share(_positive_counter(counts))
        safe_hits = sum(1 for prediction in predictions if prediction.get("safe_hit") is True)
        unsupported = sum(
            1 for prediction in predictions if prediction.get("unsupported_prediction") is True
        )
        no_prediction = sum(
            1 for prediction in predictions if not prediction.get("predicted_action")
        )
        best_fixed_hits = sum(
            1
            for prediction in predictions
            if prediction.get("best_fixed_safe_action_baseline_hit") is True
        )
        first_public_hits = sum(
            1
            for prediction in predictions
            if prediction.get("first_public_action_baseline_hit") is True
        )
        exact_hits = sum(
            1
            for prediction in predictions
            if prediction.get("target_leaky_exact_top_upper_bound_safe_hit") is True
        )
        row_count = len(predictions)
        best_trivial = max(
            _safe_rate(best_fixed_hits, row_count),
            _safe_rate(first_public_hits, row_count),
        )
        first = predictions[0] if predictions else {}
        safe_rate = _safe_rate(safe_hits, row_count)
        metrics.append(
            {
                "source_group_key": group_key,
                "source_group_type": first.get("source_group_type"),
                "source_seed": first.get("source_seed"),
                "row_count": row_count,
                "prediction_count": sum(counts.values()),
                "no_prediction_count": no_prediction,
                "unsupported_prediction_count": unsupported,
                "predicted_action_counts": dict(sorted(counts.items())),
                "dominant_predicted_action": dominant.get("key"),
                "dominant_predicted_action_share": dominant.get("share"),
                "safe_hit_count": safe_hits,
                "safe_hit_rate": safe_rate,
                "best_trivial_baseline_hit_rate": best_trivial,
                "safe_hit_margin_over_best_trivial": _round(safe_rate - best_trivial),
                "target_leaky_exact_top_upper_bound": {
                    "non_runtime": True,
                    "target_leaky": True,
                    "safe_hit_count": exact_hits,
                    "safe_hit_rate": _safe_rate(exact_hits, row_count),
                },
                "row_indexes": [prediction.get("row_index") for prediction in predictions],
                "excluded_row_indexes": predictions[0].get("excluded_row_indexes")
                if predictions
                else [],
            }
        )
    return metrics


def _per_source_seed_safe_hit_rates(
    per_group: Sequence[Mapping[str, object]],
    *,
    source_seed_diagnostic_seeds: Sequence[int],
) -> dict[str, dict[str, object]]:
    by_seed: dict[int, dict[str, int]] = {
        int(seed): {"row_count": 0, "safe_hit_count": 0}
        for seed in source_seed_diagnostic_seeds
    }
    for group in per_group:
        if group.get("source_group_type") != "v165_preterminal_source_seed":
            continue
        seed = _int(group.get("source_seed"))
        payload = by_seed.setdefault(seed, {"row_count": 0, "safe_hit_count": 0})
        payload["row_count"] += _int(group.get("row_count"))
        payload["safe_hit_count"] += _int(group.get("safe_hit_count"))
    return {
        str(seed): {
            "present": payload["row_count"] > 0,
            "row_count": payload["row_count"],
            "safe_hit_count": payload["safe_hit_count"],
            "safe_hit_rate": _safe_rate(payload["safe_hit_count"], payload["row_count"]),
        }
        for seed, payload in sorted(by_seed.items())
    }


def _first_zero_safe_hit_preterminal_seed(
    per_group: Sequence[Mapping[str, object]],
) -> int | None:
    seeds = sorted(
        {
            _int(group.get("source_seed"))
            for group in per_group
            if group.get("source_group_type") == "v165_preterminal_source_seed"
        }
    )
    for seed in seeds:
        matching = [
            group
            for group in per_group
            if group.get("source_group_type") == "v165_preterminal_source_seed"
            and _int(group.get("source_seed")) == seed
        ]
        if matching and sum(_int(group.get("safe_hit_count")) for group in matching) == 0:
            return seed
    return None


def _base_fallback_summary(
    per_group: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    fallback = [
        group
        for group in per_group
        if group.get("source_group_type") == "v158_base_row_fallback"
    ]
    row_count = sum(_int(group.get("row_count")) for group in fallback)
    safe_hits = sum(_int(group.get("safe_hit_count")) for group in fallback)
    return {
        "policy": "m3_carrion_survivor_continuation_v166_base_row_fallback_summary_v1",
        "fallback_group_count": len(fallback),
        "row_count": row_count,
        "safe_hit_count": safe_hits,
        "safe_hit_rate": _safe_rate(safe_hits, row_count),
        "reported_separately": True,
        "fallback_reason": "v158_base_rows_have_no_source_seed_provenance",
    }


def _source_grouping_summary(
    groups: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    group_counts: Counter[str] = Counter(str(group.get("group_key")) for group in groups)
    type_counts: Counter[str] = Counter(str(group.get("group_type")) for group in groups)
    seed_counts: Counter[int] = Counter(
        _int(group.get("source_seed"))
        for group in groups
        if group.get("group_type") == "v165_preterminal_source_seed"
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v166_source_grouping_v1",
        "row_count": len(groups),
        "group_count": len(group_counts),
        "group_type_row_counts": dict(sorted(type_counts.items())),
        "preterminal_source_seed_row_counts": {
            str(seed): int(count) for seed, count in sorted(seed_counts.items())
        },
        "base_row_fallback_group_count": sum(
            1
            for group in groups
            if group.get("group_type") == "v158_base_row_fallback"
        ),
    }


def _targets_by_action(row: Mapping[str, object]) -> dict[str, dict[str, object]]:
    targets: dict[str, dict[str, object]] = {}
    for target in _list_of_mappings(row.get("action_value_targets")):
        action = str(target.get("action", ""))
        if action in targets:
            targets[f"{action}#duplicate"] = dict(target)
        else:
            targets[action] = dict(target)
    return targets


def _allowed_row_schemas() -> set[str]:
    return {
        M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION,
        M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
    }


def _append_row_failure(
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


def _list_like(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _json_round_trip_digest(payload: Mapping[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    json_payload = json.loads(
        json.dumps(without_digest, sort_keys=True, allow_nan=False)
    )
    return stable_payload_digest(json_payload)
