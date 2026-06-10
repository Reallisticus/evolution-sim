from __future__ import annotations

import base64
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import struct
import zlib

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
    FORBIDDEN_TRAINABLE_PAYLOAD_TOKENS,
    load_jsonl_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V172_REPORT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V172_DATASET_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY,
    SUPPORT_PROVENANCE_SEEDS,
    trainable_payload_leakage_scan,
    validate_v172_target_rows,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v173_source_split_scorer_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v173_source_split_scorer_v1"
)
EXPECTED_V172_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v172_replay_target_dataset_expansion_"
    "replay_target_dataset_support_ready_for_v173_source_split_scorer_no_training"
)
EXPECTED_V172_EXACT_DIGEST = (
    "e731a2899cf84149e0ed2df6b11c525ce1614768c4ecd2e020a9557e973d9404"
)
EXPECTED_V172_DATASET_DIGEST = (
    "206f0b8f11db854da59bbece74590d34b64034ad7c1b3bf5b0cbca34321d122d"
)
EXPECTED_V172_ROW_COUNT = 910
EXPECTED_V172_UNIQUE_ROW_COUNT = 742
EXPECTED_V172_TIED_ROW_COUNT = 168
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v173-carrion-survivor-continuation-source-split-scorer.json"
)
DEFAULT_MIN_MARGIN_OVER_BEST_TRIVIAL = 0.05
DEFAULT_MAX_DOMINANT_SINGLETON_ACTION_SHARE = 0.75
PUBLIC_OBSERVATION_SAMPLE_COUNT = 64


class CarrionSurvivorContinuationV173SourceSplitScorerError(ValueError):
    pass


def run_carrion_survivor_continuation_v173_source_split_scorer(
    *,
    v172_report_path: str | Path = DEFAULT_V172_REPORT_PATH,
    v172_dataset_path: str | Path = DEFAULT_V172_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v172_exact_digest: str | None = EXPECTED_V172_EXACT_DIGEST,
    expected_v172_dataset_digest: str | None = EXPECTED_V172_DATASET_DIGEST,
    expected_v172_row_count: int = EXPECTED_V172_ROW_COUNT,
    expected_v172_unique_row_count: int = EXPECTED_V172_UNIQUE_ROW_COUNT,
    expected_v172_tied_row_count: int = EXPECTED_V172_TIED_ROW_COUNT,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
    min_margin_over_best_trivial: float = DEFAULT_MIN_MARGIN_OVER_BEST_TRIVIAL,
    max_dominant_singleton_action_share: float = (
        DEFAULT_MAX_DOMINANT_SINGLETON_ACTION_SHARE
    ),
) -> dict[str, object]:
    v172_report = load_json_report(v172_report_path)
    rows = load_jsonl_dataset(v172_dataset_path)
    source_validation = validate_v173_sources(
        v172_report=v172_report,
        rows=rows,
        expected_v172_exact_digest=expected_v172_exact_digest,
        expected_v172_dataset_digest=expected_v172_dataset_digest,
        expected_v172_row_count=expected_v172_row_count,
        expected_v172_unique_row_count=expected_v172_unique_row_count,
        expected_v172_tied_row_count=expected_v172_tied_row_count,
        support_provenance_seeds=support_provenance_seeds,
    )
    leakage_scan = trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in rows]
    )
    row_schema_validation = validate_v172_target_rows(
        rows,
        support_provenance_seeds=support_provenance_seeds,
    )
    split_evaluation = _skipped_split_evaluation("source_validation_failed")
    if (
        source_validation.get("passed") is True
        and leakage_scan.get("passed") is True
        and row_schema_validation.get("passed") is True
    ):
        split_evaluation = source_split_scorer_evaluation(
            rows,
            support_provenance_seeds=support_provenance_seeds,
            min_margin_over_best_trivial=min_margin_over_best_trivial,
            max_dominant_singleton_action_share=max_dominant_singleton_action_share,
        )
    classification = _classification(
        source_validation=source_validation,
        leakage_scan=leakage_scan,
        row_schema_validation=row_schema_validation,
        split_evaluation=split_evaluation,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V173_SOURCE_SPLIT_SCORER_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v172_report": str(v172_report_path),
            "v172_dataset": str(v172_dataset_path),
            "expected_v172_exact_digest": expected_v172_exact_digest,
            "expected_v172_dataset_digest": expected_v172_dataset_digest,
            "expected_v172_row_count": int(expected_v172_row_count),
            "expected_v172_unique_row_count": int(expected_v172_unique_row_count),
            "expected_v172_tied_row_count": int(expected_v172_tied_row_count),
            "support_provenance_seeds": [
                int(seed) for seed in support_provenance_seeds
            ],
            "source_split_group_key": "metadata.seed",
            "public_observation_sample_count": PUBLIC_OBSERVATION_SAMPLE_COUNT,
            "min_margin_over_best_trivial": _round(min_margin_over_best_trivial),
            "max_dominant_singleton_action_share": _round(
                max_dominant_singleton_action_share
            ),
        },
        "source_validation": source_validation,
        "leakage_scan": leakage_scan,
        "row_schema_validation": row_schema_validation,
        "source_split_evaluation": split_evaluation,
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "dataset_digest": stable_payload_digest(rows),
        "support_provenance_seed_policy": support_provenance_seed_policy(
            support_provenance_seeds=support_provenance_seeds,
        ),
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v173_sources(
    *,
    v172_report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v172_exact_digest: str | None = EXPECTED_V172_EXACT_DIGEST,
    expected_v172_dataset_digest: str | None = EXPECTED_V172_DATASET_DIGEST,
    expected_v172_row_count: int = EXPECTED_V172_ROW_COUNT,
    expected_v172_unique_row_count: int = EXPECTED_V172_UNIQUE_ROW_COUNT,
    expected_v172_tied_row_count: int = EXPECTED_V172_TIED_ROW_COUNT,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> dict[str, object]:
    failures: list[str] = []
    observed_classification = str(
        _mapping(v172_report.get("classification")).get("primary") or ""
    )
    if (
        v172_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_SCHEMA_VERSION
    ):
        failures.append("v172_schema_version_mismatch")
    if (
        v172_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_POLICY
    ):
        failures.append("v172_policy_mismatch")
    if observed_classification != EXPECTED_V172_CLASSIFICATION:
        failures.append("v172_unexpected_classification")
    exact = exact_digest_validation_report(v172_report)
    observed_exact = str(v172_report.get("exact_digest") or "")
    if exact.get("passed") is not True:
        failures.append("v172_exact_digest_mismatch")
    if expected_v172_exact_digest and observed_exact != expected_v172_exact_digest:
        failures.append("v172_unexpected_exact_digest")
    dataset_digest = stable_payload_digest(rows)
    reported_dataset = _mapping(v172_report.get("dataset"))
    reported_dataset_digest = str(reported_dataset.get("dataset_digest") or "")
    if expected_v172_dataset_digest and dataset_digest != expected_v172_dataset_digest:
        failures.append("v172_unexpected_dataset_digest")
    if (
        expected_v172_dataset_digest
        and reported_dataset_digest != expected_v172_dataset_digest
    ):
        failures.append("v172_reported_unexpected_dataset_digest")
    if reported_dataset_digest != dataset_digest:
        failures.append("v172_dataset_digest_mismatch")
    if len(rows) != int(expected_v172_row_count):
        failures.append("v172_row_count_mismatch")
    if _int(reported_dataset.get("row_count")) != len(rows):
        failures.append("v172_reported_row_count_mismatch")
    if (
        reported_dataset.get("row_schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
    ):
        failures.append("v172_reported_row_schema_mismatch")
    if (
        reported_dataset.get("feature_policy_id")
        != M3_CARRION_SURVIVOR_CONTINUATION_V172_REPLAY_TARGET_DATASET_FEATURE_POLICY
    ):
        failures.append("v172_reported_feature_policy_mismatch")
    if _mapping(v172_report.get("leakage_scan")).get("passed") is not True:
        failures.append("v172_leakage_scan_not_passed")
    if _mapping(v172_report.get("row_schema_validation")).get("passed") is not True:
        failures.append("v172_row_schema_not_passed")
    counts = _mapping(v172_report.get("unique_action_vs_tied_action_row_counts"))
    if _int(counts.get("unique_action_row_count")) != int(
        expected_v172_unique_row_count
    ):
        failures.append("v172_unique_row_count_mismatch")
    if _int(counts.get("tied_action_row_count")) != int(
        expected_v172_tied_row_count
    ):
        failures.append("v172_tied_row_count_mismatch")
    route = _mapping(v172_report.get("route_recommendation"))
    if (
        route.get("recommended_next_route")
        != "v173_diagnostics_only_source_split_scorer_no_training"
    ):
        failures.append("v172_route_not_v173_source_split")
    if route.get("v173_source_split_scorer_recommended") is not True:
        failures.append("v172_v173_recommendation_missing")
    plan = _mapping(v172_report.get("source_split_evaluation_plan_for_v173"))
    plan_validation = _v172_source_split_plan_validation(
        plan,
        support_provenance_seeds=support_provenance_seeds,
    )
    if plan_validation.get("passed") is not True:
        failures.append("v172_source_split_plan_invalid")
    provenance_validation = _source_provenance_seed_validation(
        rows,
        support_provenance_seeds=support_provenance_seeds,
    )
    if provenance_validation.get("passed") is not True:
        failures.append("v172_support_provenance_seed_rows_invalid")
    lifecycle = _v172_lifecycle_validation(v172_report)
    if lifecycle.get("passed") is not True:
        failures.append("v172_lifecycle_not_diagnostics_only")
    return {
        "policy": "m3_carrion_survivor_continuation_v173_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v172_classification": EXPECTED_V172_CLASSIFICATION,
        "observed_v172_classification": observed_classification,
        "expected_v172_exact_digest": expected_v172_exact_digest,
        "observed_v172_exact_digest": observed_exact,
        "v172_exact_digest_validation": exact,
        "expected_v172_dataset_digest": expected_v172_dataset_digest,
        "observed_v172_dataset_digest": dataset_digest,
        "v172_reported_dataset_digest": reported_dataset_digest,
        "expected_v172_row_count": int(expected_v172_row_count),
        "expected_v172_unique_row_count": int(expected_v172_unique_row_count),
        "expected_v172_tied_row_count": int(expected_v172_tied_row_count),
        "observed_v172_row_count": len(rows),
        "v172_reported_row_count": _int(reported_dataset.get("row_count")),
        "v172_report_leakage_scan_passed": _mapping(
            v172_report.get("leakage_scan")
        ).get("passed"),
        "v172_report_row_schema_passed": _mapping(
            v172_report.get("row_schema_validation")
        ).get("passed"),
        "v172_route_recommendation": route,
        "v172_source_split_plan_validation": plan_validation,
        "support_provenance_seed_validation": provenance_validation,
        "v172_lifecycle_validation": lifecycle,
    }


def source_split_scorer_evaluation(
    rows: Sequence[Mapping[str, object]],
    *,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
    min_margin_over_best_trivial: float = DEFAULT_MIN_MARGIN_OVER_BEST_TRIVIAL,
    max_dominant_singleton_action_share: float = (
        DEFAULT_MAX_DOMINANT_SINGLETON_ACTION_SHARE
    ),
) -> dict[str, object]:
    seeds = sorted({int(seed) for seed in support_provenance_seeds})
    row_seeds = [_source_seed(row) for row in rows]
    indexes_by_seed: dict[int, list[int]] = defaultdict(list)
    for index, seed in enumerate(row_seeds):
        indexes_by_seed[seed].append(index)
    vectors = _normalized_feature_vectors(rows)
    shuffled_safe_sets = _rotated_safe_sets(rows)

    predictions: list[dict[str, object]] = []
    for seed in seeds:
        heldout_indexes = list(indexes_by_seed.get(seed, []))
        train_indexes = [
            index for index, row_seed in enumerate(row_seeds) if int(row_seed) != seed
        ]
        for row_index in heldout_indexes:
            row = rows[row_index]
            public_mask = _complete_action_mask(
                _mapping(row.get("public_action_mask"))
            )
            safe_set = _safe_action_set(row)
            singleton_action, scores = _singleton_public_feature_prediction(
                rows=rows,
                row_index=row_index,
                train_indexes=train_indexes,
                vectors=vectors,
                public_mask=public_mask,
            )
            set_actions = _set_valued_candidate_actions(scores)
            action_frequency = _best_action_frequency_prediction(
                rows=rows,
                train_indexes=train_indexes,
                public_mask=public_mask,
            )
            mask_only = _mask_only_frequency_prediction(
                rows=rows,
                train_indexes=train_indexes,
                public_mask=public_mask,
            )
            first_public = _first_public_action(public_mask)
            shuffled_action, _ = _singleton_public_feature_prediction(
                rows=rows,
                row_index=row_index,
                train_indexes=train_indexes,
                vectors=vectors,
                public_mask=public_mask,
                safe_sets_by_row=shuffled_safe_sets,
            )
            public_actions = _public_actions(public_mask)
            row_type = "unique" if len(safe_set) == 1 else "tied"
            prediction = {
                "row_index": row_index,
                "left_out_seed": seed,
                "source_split_group_key": f"metadata.seed:{seed}",
                "train_row_count": len(train_indexes),
                "heldout_seed_row_count": len(heldout_indexes),
                "row_type": row_type,
                "safe_action_set": safe_set,
                "singleton_predicted_action": singleton_action,
                "singleton_safe_hit": singleton_action in safe_set,
                "singleton_unsupported": bool(
                    singleton_action and public_mask.get(singleton_action) is not True
                ),
                "singleton_no_prediction": singleton_action is None,
                "set_valued_predicted_actions": set_actions,
                "set_valued_safe_hit": bool(set(set_actions) & set(safe_set)),
                "set_valued_unsupported": any(
                    public_mask.get(action) is not True for action in set_actions
                ),
                "set_valued_no_prediction": not set_actions,
                "set_valued_candidate_width": len(set_actions),
                "set_valued_full_public_mask_set": set(set_actions)
                == set(public_actions),
                "public_action_count": len(public_actions),
                "action_frequency_baseline_action": action_frequency,
                "action_frequency_baseline_hit": action_frequency in safe_set,
                "mask_only_baseline_action": mask_only,
                "mask_only_baseline_hit": mask_only in safe_set,
                "first_public_action_baseline_action": first_public,
                "first_public_action_baseline_hit": first_public in safe_set,
                "shuffled_target_control_action": shuffled_action,
                "shuffled_target_control_hit": shuffled_action in safe_set,
            }
            predictions.append(prediction)

    singleton = _singleton_metrics(
        predictions,
        min_margin_over_best_trivial=min_margin_over_best_trivial,
        max_dominant_singleton_action_share=max_dominant_singleton_action_share,
    )
    set_valued = _set_valued_metrics(
        predictions,
        min_margin_over_best_trivial=min_margin_over_best_trivial,
    )
    baselines = _baseline_metrics(predictions)
    per_seed = _per_left_out_seed_metrics(predictions, seeds)
    first_failing_seed = _first_failing_seed(per_seed)
    failed_examples = _failed_row_examples(predictions)
    singleton_ready = (
        singleton.get("unsupported_prediction_count") == 0
        and singleton.get("no_prediction_count") == 0
        and singleton.get("dominant_prediction_floor_passed") is True
        and singleton.get("safe_hit_margin_floor_passed") is True
        and singleton.get("every_left_out_seed_has_nonzero_safe_hit_support") is True
    )
    set_valued_working = (
        set_valued.get("unsupported_prediction_count") == 0
        and set_valued.get("no_prediction_count") == 0
        and set_valued.get("set_hit_margin_floor_passed") is True
        and set_valued.get("every_left_out_seed_has_nonzero_set_hit_support") is True
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v173_source_split_evaluation_v1",
        "source_split_group_key": "metadata.seed",
        "leave_one_support_seed_out": True,
        "support_provenance_seeds": seeds,
        "row_count": len(rows),
        "prediction_count": len(predictions),
        "unique_row_count": sum(
            1 for prediction in predictions if prediction.get("row_type") == "unique"
        ),
        "tied_row_count": sum(
            1 for prediction in predictions if prediction.get("row_type") == "tied"
        ),
        "scorers_evaluated": [
            "singleton_public_feature_similarity_ranker",
            "set_valued_positive_support_ranker",
            "action_frequency_baseline",
            "mask_only_frequency_baseline",
            "first_public_action_baseline",
            "shuffled_target_negative_control",
        ],
        "trainable_input_policy": {
            "uses_only_trainable_public_features": True,
            "uses_public_action_mask": True,
            "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
            "uses_outcome_target_or_runtime_action_as_trainable_input": False,
            "feature_normalization_uses_public_features_only": True,
        },
        "singleton_scorer": singleton,
        "set_valued_ranker": set_valued,
        "trivial_and_negative_controls": baselines,
        "per_left_out_seed": per_seed,
        "first_failing_seed": first_failing_seed,
        "failed_row_examples": failed_examples,
        "singleton_ready_for_shadow_eval": singleton_ready,
        "set_valued_partial_support": set_valued_working and not singleton_ready,
        "floor_passed": singleton_ready,
        "predictions_sample": predictions[:24],
    }


def support_provenance_seed_policy(
    *,
    support_provenance_seeds: Sequence[int] = SUPPORT_PROVENANCE_SEEDS,
) -> dict[str, object]:
    seeds = sorted({int(seed) for seed in support_provenance_seeds})
    return {
        "policy": "m3_carrion_survivor_continuation_v173_support_provenance_seed_policy_v1",
        "support_provenance_seeds": seeds,
        "support_provenance_seeds_are_future_promotion_heldout": False,
        "future_promotion_heldout_seed_reuse_allowed": False,
        "new_promotion_heldout_broad_seeds_required": True,
    }


def _singleton_public_feature_prediction(
    *,
    rows: Sequence[Mapping[str, object]],
    row_index: int,
    train_indexes: Sequence[int],
    vectors: Sequence[Sequence[float]],
    public_mask: Mapping[str, bool],
    safe_sets_by_row: Mapping[int, Sequence[str]] | None = None,
) -> tuple[str | None, dict[str, float]]:
    scores: dict[str, float] = {
        action: 0.0 for action in ACTION_NAMES if public_mask.get(action) is True
    }
    if not scores:
        return None, {}
    target_vector = vectors[row_index]
    for train_index in train_indexes:
        distance = _squared_distance(target_vector, vectors[train_index])
        weight = 1.0 / (1.0 + distance)
        safe_set = (
            _ordered_action_set(safe_sets_by_row.get(train_index))
            if safe_sets_by_row is not None
            else _safe_action_set(rows[train_index])
        )
        for action in safe_set:
            if action in scores:
                scores[action] += weight
    positive = {action: score for action, score in scores.items() if score > 0.0}
    if not positive:
        return None, scores
    predicted = sorted(
        positive,
        key=lambda action: (-positive[action], _action_order(action)),
    )[0]
    return predicted, {action: _round(score) for action, score in scores.items()}


def _set_valued_candidate_actions(scores: Mapping[str, float]) -> list[str]:
    return sorted(
        [action for action, score in scores.items() if float(score) > 0.0],
        key=_action_order,
    )


def _best_action_frequency_prediction(
    *,
    rows: Sequence[Mapping[str, object]],
    train_indexes: Sequence[int],
    public_mask: Mapping[str, bool],
) -> str | None:
    counts: Counter[str] = Counter()
    for index in train_indexes:
        counts.update(_safe_action_set(rows[index]))
    return _best_counted_public_action(counts, public_mask)


def _mask_only_frequency_prediction(
    *,
    rows: Sequence[Mapping[str, object]],
    train_indexes: Sequence[int],
    public_mask: Mapping[str, bool],
) -> str | None:
    target_signature = _mask_signature(public_mask)
    counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    for index in train_indexes:
        safe = _safe_action_set(rows[index])
        fallback_counts.update(safe)
        mask = _complete_action_mask(_mapping(rows[index].get("public_action_mask")))
        if _mask_signature(mask) == target_signature:
            counts.update(safe)
    return _best_counted_public_action(counts or fallback_counts, public_mask)


def _best_counted_public_action(
    counts: Counter[str],
    public_mask: Mapping[str, bool],
) -> str | None:
    candidates = [
        action
        for action in ACTION_NAMES
        if public_mask.get(action) is True and int(counts.get(action, 0)) > 0
    ]
    if not candidates:
        return _first_public_action(public_mask)
    return sorted(
        candidates,
        key=lambda action: (-int(counts.get(action, 0)), _action_order(action)),
    )[0]


def _singleton_metrics(
    predictions: Sequence[Mapping[str, object]],
    *,
    min_margin_over_best_trivial: float,
    max_dominant_singleton_action_share: float,
) -> dict[str, object]:
    row_count = len(predictions)
    predicted_actions = Counter(
        str(prediction.get("singleton_predicted_action"))
        for prediction in predictions
        if prediction.get("singleton_predicted_action")
    )
    safe_hits = sum(prediction.get("singleton_safe_hit") is True for prediction in predictions)
    unsupported = sum(
        prediction.get("singleton_unsupported") is True for prediction in predictions
    )
    no_prediction = sum(
        prediction.get("singleton_no_prediction") is True for prediction in predictions
    )
    unique = [
        prediction for prediction in predictions if prediction.get("row_type") == "unique"
    ]
    tied = [
        prediction for prediction in predictions if prediction.get("row_type") == "tied"
    ]
    robust_hits = sum(
        prediction.get("singleton_safe_hit") is True for prediction in unique
    )
    tied_hits = sum(
        prediction.get("singleton_safe_hit") is True for prediction in tied
    )
    baselines = _baseline_metrics(predictions)
    best_trivial = _round(baselines.get("best_trivial_singleton_hit_rate"))
    safe_hit_rate = _rate(safe_hits, row_count)
    margin = _round(safe_hit_rate - best_trivial)
    dominant = _dominant_count_share(predicted_actions)
    dominant_share = float(dominant.get("share") or 0.0)
    per_seed = _per_left_out_seed_metrics(
        predictions,
        sorted({int(prediction.get("left_out_seed")) for prediction in predictions}),
    )
    every_seed_positive = all(
        _int(_mapping(payload).get("singleton_safe_hit_count")) > 0
        for payload in per_seed.values()
    )
    return {
        "scorer_type": "singleton_public_feature_similarity_ranker",
        "row_count": row_count,
        "prediction_count": sum(predicted_actions.values()),
        "safe_hit_count": int(safe_hits),
        "safe_hit_rate": safe_hit_rate,
        "safe_hit_margin_over_best_trivial": margin,
        "best_trivial_singleton_hit_rate": best_trivial,
        "min_margin_over_best_trivial": _round(min_margin_over_best_trivial),
        "safe_hit_margin_floor_passed": margin >= float(min_margin_over_best_trivial),
        "robust_winner_unique_row_count": len(unique),
        "robust_winner_hit_count": int(robust_hits),
        "robust_winner_hit_rate": _rate(robust_hits, len(unique)),
        "tied_row_count": len(tied),
        "tied_row_singleton_safe_hit_count": int(tied_hits),
        "tied_row_singleton_safe_hit_rate": _rate(tied_hits, len(tied)),
        "unsupported_prediction_count": int(unsupported),
        "no_prediction_count": int(no_prediction),
        "predicted_action_counts": dict(_ordered_counter(predicted_actions)),
        "dominant_singleton_predicted_action": dominant.get("key"),
        "dominant_singleton_predicted_action_count": dominant.get("count"),
        "dominant_singleton_predicted_action_share": dominant.get("share"),
        "max_dominant_singleton_action_share": _round(
            max_dominant_singleton_action_share
        ),
        "dominant_prediction_floor_passed": (
            bool(predicted_actions)
            and dominant_share <= float(max_dominant_singleton_action_share)
        ),
        "every_left_out_seed_has_nonzero_safe_hit_support": every_seed_positive,
    }


def _set_valued_metrics(
    predictions: Sequence[Mapping[str, object]],
    *,
    min_margin_over_best_trivial: float,
) -> dict[str, object]:
    row_count = len(predictions)
    set_hits = sum(
        prediction.get("set_valued_safe_hit") is True for prediction in predictions
    )
    unsupported = sum(
        prediction.get("set_valued_unsupported") is True for prediction in predictions
    )
    no_prediction = sum(
        prediction.get("set_valued_no_prediction") is True for prediction in predictions
    )
    widths = [
        _int(prediction.get("set_valued_candidate_width"))
        for prediction in predictions
    ]
    full_public = sum(
        prediction.get("set_valued_full_public_mask_set") is True
        for prediction in predictions
    )
    tied = [
        prediction for prediction in predictions if prediction.get("row_type") == "tied"
    ]
    tied_hits = sum(
        prediction.get("set_valued_safe_hit") is True for prediction in tied
    )
    per_seed = _per_left_out_seed_metrics(
        predictions,
        sorted({int(prediction.get("left_out_seed")) for prediction in predictions}),
    )
    every_seed_positive = all(
        _int(_mapping(payload).get("set_valued_safe_hit_count")) > 0
        for payload in per_seed.values()
    )
    best_trivial = _round(_baseline_metrics(predictions).get("best_trivial_singleton_hit_rate"))
    set_hit_rate = _rate(set_hits, row_count)
    margin = _round(set_hit_rate - best_trivial)
    return {
        "scorer_type": "set_valued_positive_support_ranker",
        "row_count": row_count,
        "set_hit_count": int(set_hits),
        "set_hit_rate": set_hit_rate,
        "set_hit_margin_over_best_trivial_singleton": margin,
        "min_margin_over_best_trivial": _round(min_margin_over_best_trivial),
        "set_hit_margin_floor_passed": margin >= float(min_margin_over_best_trivial),
        "tied_row_count": len(tied),
        "tied_row_set_hit_count": int(tied_hits),
        "tied_row_set_hit_rate": _rate(tied_hits, len(tied)),
        "unsupported_prediction_count": int(unsupported),
        "no_prediction_count": int(no_prediction),
        "average_candidate_set_width": (
            _round(sum(widths) / len(widths)) if widths else 0.0
        ),
        "candidate_set_width_counts": {
            str(width): int(count)
            for width, count in sorted(Counter(widths).items())
        },
        "full_set_share": _rate(full_public, row_count),
        "full_public_mask_set_count": int(full_public),
        "every_left_out_seed_has_nonzero_set_hit_support": every_seed_positive,
    }


def _baseline_metrics(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    row_count = len(predictions)
    action_frequency_hits = sum(
        prediction.get("action_frequency_baseline_hit") is True
        for prediction in predictions
    )
    mask_only_hits = sum(
        prediction.get("mask_only_baseline_hit") is True for prediction in predictions
    )
    first_public_hits = sum(
        prediction.get("first_public_action_baseline_hit") is True
        for prediction in predictions
    )
    shuffled_hits = sum(
        prediction.get("shuffled_target_control_hit") is True
        for prediction in predictions
    )
    action_frequency_rate = _rate(action_frequency_hits, row_count)
    mask_only_rate = _rate(mask_only_hits, row_count)
    first_public_rate = _rate(first_public_hits, row_count)
    best_trivial = max(action_frequency_rate, mask_only_rate, first_public_rate)
    return {
        "policy": "m3_carrion_survivor_continuation_v173_baselines_v1",
        "action_frequency_baseline_hit_count": int(action_frequency_hits),
        "action_frequency_baseline_hit_rate": action_frequency_rate,
        "mask_only_frequency_baseline_hit_count": int(mask_only_hits),
        "mask_only_frequency_baseline_hit_rate": mask_only_rate,
        "first_public_action_baseline_hit_count": int(first_public_hits),
        "first_public_action_baseline_hit_rate": first_public_rate,
        "best_trivial_singleton_hit_rate": best_trivial,
        "shuffled_target_negative_control_hit_count": int(shuffled_hits),
        "shuffled_target_negative_control_hit_rate": _rate(shuffled_hits, row_count),
    }


def _per_left_out_seed_metrics(
    predictions: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
) -> dict[str, dict[str, object]]:
    by_seed: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for prediction in predictions:
        by_seed[_int(prediction.get("left_out_seed"))].append(prediction)
    result = {}
    for seed in sorted({int(seed) for seed in seeds}):
        rows = by_seed.get(seed, [])
        unique = [row for row in rows if row.get("row_type") == "unique"]
        tied = [row for row in rows if row.get("row_type") == "tied"]
        singleton_hits = sum(row.get("singleton_safe_hit") is True for row in rows)
        set_hits = sum(row.get("set_valued_safe_hit") is True for row in rows)
        result[str(seed)] = {
            "support_provenance_seed": True,
            "future_promotion_heldout_seed": False,
            "row_count": len(rows),
            "unique_row_count": len(unique),
            "tied_row_count": len(tied),
            "singleton_safe_hit_count": int(singleton_hits),
            "singleton_safe_hit_rate": _rate(singleton_hits, len(rows)),
            "robust_winner_hit_count": int(
                sum(row.get("singleton_safe_hit") is True for row in unique)
            ),
            "robust_winner_hit_rate": _rate(
                sum(row.get("singleton_safe_hit") is True for row in unique),
                len(unique),
            ),
            "set_valued_safe_hit_count": int(set_hits),
            "set_valued_safe_hit_rate": _rate(set_hits, len(rows)),
            "tied_row_set_hit_count": int(
                sum(row.get("set_valued_safe_hit") is True for row in tied)
            ),
            "tied_row_set_hit_rate": _rate(
                sum(row.get("set_valued_safe_hit") is True for row in tied),
                len(tied),
            ),
            "unsupported_prediction_count": int(
                sum(row.get("singleton_unsupported") is True for row in rows)
            ),
            "no_prediction_count": int(
                sum(row.get("singleton_no_prediction") is True for row in rows)
            ),
            "average_candidate_set_width": (
                _round(
                    sum(_int(row.get("set_valued_candidate_width")) for row in rows)
                    / len(rows)
                )
                if rows
                else 0.0
            ),
        }
    return result


def _classification(
    *,
    source_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    split_evaluation: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v173_source_split_scorer_"
    if (
        source_validation.get("passed") is not True
        or leakage_scan.get("passed") is not True
        or row_schema_validation.get("passed") is not True
    ):
        return prefix + "closed_invalid_no_shadow"
    if split_evaluation.get("floor_passed") is True:
        return prefix + "source_split_scorer_ready_for_v174_shadow_eval_no_runtime"
    if split_evaluation.get("set_valued_partial_support") is True:
        return prefix + "source_split_set_valued_partial_no_shadow"
    return prefix + "source_split_generalization_failed_closed_no_shadow"


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification.endswith(
        "source_split_scorer_ready_for_v174_shadow_eval_no_runtime"
    )
    partial = classification.endswith("source_split_set_valued_partial_no_shadow")
    return {
        "policy": "m3_carrion_survivor_continuation_v173_route_recommendation_v1",
        "recommended_next_route": (
            "v174_diagnostics_only_shadow_eval_no_runtime"
            if ready
            else "narrow_set_valued_support_or_expand_public_context_before_shadow"
            if partial
            else "close_v173_source_split_scorer_without_shadow"
        ),
        "v174_shadow_eval_recommended": ready,
        "set_valued_followup_recommended": partial,
        "training_authorized": False,
        "scorer_retraining_authorized": False,
        "runtime_integration_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
    }


def _v172_source_split_plan_validation(
    plan: Mapping[str, object],
    *,
    support_provenance_seeds: Sequence[int],
) -> dict[str, object]:
    expected = sorted({int(seed) for seed in support_provenance_seeds})
    observed = sorted(_int(seed) for seed in _list_like(plan.get("support_provenance_seeds")))
    disallowed = sorted(
        _int(seed)
        for seed in _list_like(plan.get("disallowed_future_promotion_heldout_seed_reuse"))
    )
    failures = []
    if plan.get("recommended_next_route") != "v173_diagnostics_only_source_split_scorer_no_training":
        failures.append("recommended_next_route_mismatch")
    if plan.get("source_split_evaluation_required") is not True:
        failures.append("source_split_not_required")
    if plan.get("leave_one_support_seed_out_required") is not True:
        failures.append("leave_one_support_seed_out_not_required")
    if plan.get("source_group_key") != "metadata.seed":
        failures.append("source_group_key_mismatch")
    if observed != expected:
        failures.append("support_provenance_seed_mismatch")
    if disallowed != expected:
        failures.append("disallowed_promotion_heldout_seed_mismatch")
    for field in (
        "training_authorized",
        "scorer_retraining_authorized",
        "shadow_eval_authorized",
        "runtime_integration_authorized",
        "promotion_authorized",
    ):
        if plan.get(field) is not False:
            failures.append(f"{field}_not_false")
    return {
        "policy": "m3_carrion_survivor_continuation_v173_v172_plan_validation_v1",
        "passed": not failures,
        "failures": failures,
        "expected_support_provenance_seeds": expected,
        "observed_support_provenance_seeds": observed,
        "observed_disallowed_future_promotion_heldout_seeds": disallowed,
    }


def _source_provenance_seed_validation(
    rows: Sequence[Mapping[str, object]],
    *,
    support_provenance_seeds: Sequence[int],
) -> dict[str, object]:
    expected = sorted({int(seed) for seed in support_provenance_seeds})
    counts = Counter(_source_seed(row) for row in rows)
    failures = []
    observed = sorted(seed for seed in counts if seed > 0)
    if observed != expected:
        failures.append("source_seed_set_mismatch")
    for row_index, row in enumerate(rows):
        metadata = _mapping(row.get("metadata"))
        if (
            metadata.get("source_seed_is_support_provenance_not_future_promotion_holdout")
            is not True
        ):
            failures.append(f"row_{row_index}_source_seed_not_support_provenance")
            break
    return {
        "policy": "m3_carrion_survivor_continuation_v173_support_seed_validation_v1",
        "passed": not failures,
        "failures": failures,
        "expected_support_provenance_seeds": expected,
        "observed_source_seeds": observed,
        "row_counts_by_seed": {
            str(seed): int(counts.get(seed, 0)) for seed in expected
        },
    }


def _v172_lifecycle_validation(report: Mapping[str, object]) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    if report.get("diagnostics_only") is not True:
        failures.append(
            {"field": "diagnostics_only", "observed": report.get("diagnostics_only")}
        )
    for field in (
        "training_ran",
        "training_authorized",
        "scorer_retraining_ran",
        "scorer_retraining_authorized",
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
        "policy": "m3_carrion_survivor_continuation_v173_v172_lifecycle_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "source_split_scorer_evaluation_only": True,
        "leave_one_support_seed_out_by_metadata_seed": True,
        "uses_only_trainable_public_features_and_public_action_mask": True,
        "support_provenance_seeds_are_future_promotion_heldout_seeds": False,
        "training_allowed": False,
        "scorer_retraining_allowed": False,
        "runtime_artifact_allowed": False,
        "diagnostic_artifact_created": False,
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
        "artifact_created": False,
        "diagnostic_artifact_created": False,
        "runtime_artifact_created": False,
        "training_ran": False,
        "training_authorized": False,
        "scorer_retraining_ran": False,
        "scorer_retraining_authorized": False,
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


def _normalized_feature_vectors(
    rows: Sequence[Mapping[str, object]],
) -> list[list[float]]:
    raw_vectors = [_public_feature_vector(row) for row in rows]
    width = max((len(vector) for vector in raw_vectors), default=0)
    padded = [vector + [0.0] * (width - len(vector)) for vector in raw_vectors]
    if not padded:
        return []
    mins = [min(vector[index] for vector in padded) for index in range(width)]
    maxs = [max(vector[index] for vector in padded) for index in range(width)]
    normalized: list[list[float]] = []
    for vector in padded:
        current = []
        for index, value in enumerate(vector):
            span = maxs[index] - mins[index]
            current.append(0.0 if span == 0.0 else (value - mins[index]) / span)
        normalized.append(current)
    return normalized


def _public_feature_vector(row: Mapping[str, object]) -> list[float]:
    features = _mapping(row.get("trainable_public_features"))
    observation = _mapping(features.get("public_observation"))
    values = _public_observation_values(observation)
    mask = _complete_action_mask(_mapping(features.get("action_mask")))
    values.extend(1.0 if mask.get(action) is True else 0.0 for action in ACTION_NAMES)
    return values


def _public_observation_values(observation: Mapping[str, object]) -> list[float]:
    if (
        observation.get("storage_encoding") == "zlib_base64_little_endian_int16"
        and isinstance(observation.get("data"), str)
    ):
        decoded = _decode_zlib_int16(str(observation.get("data") or ""))
        if decoded:
            samples = _even_samples(decoded, PUBLIC_OBSERVATION_SAMPLE_COUNT)
            values = [float(value) / 32768.0 for value in samples]
            values.extend(
                [
                    len(decoded) / 1000.0,
                    float(sum(decoded)) / (32768.0 * len(decoded)),
                    float(min(decoded)) / 32768.0,
                    float(max(decoded)) / 32768.0,
                ]
            )
            return values
    values: list[float] = []
    _flatten_public_value(observation, values, path=())
    return values[: PUBLIC_OBSERVATION_SAMPLE_COUNT + 4]


def _decode_zlib_int16(data: str) -> list[int]:
    try:
        raw = zlib.decompress(base64.b64decode(data.encode("ascii")))
    except (OSError, ValueError, zlib.error):
        return []
    count = len(raw) // 2
    if count <= 0:
        return []
    return list(struct.unpack("<" + "h" * count, raw[: count * 2]))


def _even_samples(values: Sequence[int], count: int) -> list[int]:
    if len(values) <= count:
        return [int(value) for value in values]
    if count <= 1:
        return [int(values[0])]
    return [
        int(values[round(index * (len(values) - 1) / (count - 1))])
        for index in range(count)
    ]


def _flatten_public_value(
    value: object,
    values: list[float],
    *,
    path: tuple[str, ...],
) -> None:
    if isinstance(value, bool):
        values.append(1.0 if value else 0.0)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            values.append(float(value))
    elif isinstance(value, str):
        digest = stable_payload_digest({"path": path, "value": value})
        values.append(int(digest[:8], 16) / 0xFFFFFFFF)
    elif isinstance(value, Mapping):
        for key in sorted(value):
            _flatten_public_value(value[key], values, path=(*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value[:PUBLIC_OBSERVATION_SAMPLE_COUNT]):
            _flatten_public_value(item, values, path=(*path, str(index)))


def _rotated_safe_sets(
    rows: Sequence[Mapping[str, object]],
) -> dict[int, list[str]]:
    if not rows:
        return {}
    safe_sets = [_safe_action_set(row) for row in rows]
    return {
        index: list(safe_sets[(index + 1) % len(safe_sets)])
        for index in range(len(rows))
    }


def _failed_row_examples(
    predictions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    examples = []
    for prediction in predictions:
        if (
            prediction.get("singleton_safe_hit") is True
            and prediction.get("singleton_unsupported") is not True
            and prediction.get("singleton_no_prediction") is not True
        ):
            continue
        examples.append(
            {
                "row_index": prediction.get("row_index"),
                "left_out_seed": prediction.get("left_out_seed"),
                "row_type": prediction.get("row_type"),
                "safe_action_set": prediction.get("safe_action_set"),
                "singleton_predicted_action": prediction.get(
                    "singleton_predicted_action"
                ),
                "set_valued_predicted_actions": prediction.get(
                    "set_valued_predicted_actions"
                ),
                "action_frequency_baseline_action": prediction.get(
                    "action_frequency_baseline_action"
                ),
                "mask_only_baseline_action": prediction.get(
                    "mask_only_baseline_action"
                ),
            }
        )
        if len(examples) >= 24:
            break
    return examples


def _first_failing_seed(
    per_seed: Mapping[str, Mapping[str, object]],
) -> int | None:
    for seed in sorted((_int(seed) for seed in per_seed), key=int):
        payload = _mapping(per_seed.get(str(seed)))
        if _int(payload.get("singleton_safe_hit_count")) == 0:
            return int(seed)
    return None


def _skipped_split_evaluation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v173_source_split_evaluation_v1",
        "skipped": True,
        "reason": reason,
        "floor_passed": False,
        "set_valued_partial_support": False,
    }


def _source_seed(row: Mapping[str, object]) -> int:
    return _int(_mapping(row.get("metadata")).get("seed"), default=-1)


def _safe_action_set(row: Mapping[str, object]) -> list[str]:
    return _ordered_action_set(row.get("safe_action_set"))


def _public_actions(public_mask: Mapping[str, bool]) -> list[str]:
    return [action for action in ACTION_NAMES if public_mask.get(action) is True]


def _first_public_action(public_mask: Mapping[str, bool]) -> str | None:
    for action in ACTION_NAMES:
        if public_mask.get(action) is True:
            return action
    return None


def _mask_signature(public_mask: Mapping[str, bool]) -> tuple[str, ...]:
    return tuple(action for action in ACTION_NAMES if public_mask.get(action) is True)


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


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def _rate(count: int, total: int) -> float:
    return _round(float(count) / float(total)) if total else 0.0


def _list_like(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _json_round_trip_digest(payload: Mapping[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(without_digest, sort_keys=True, allow_nan=False))
    )
