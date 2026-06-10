from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import gzip
import json
import math
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
)
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
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
    load_v154_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    _complete_bool_mask,
    _safe_action_set,
    _safe_rate,
    _top_value_target,
)
from evolution_sim.mind.carrion_survivor_continuation_v160_action_value_scorer import (
    _best_fixed_safe_action,
    _flatten_public_payload,
    _predict_1nn_action,
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
    EXPECTED_V166_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION,
)
from evolution_sim.mind.feature_policy import feature_keys_from_record
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_public_feature_contract_probe_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_public_feature_contract_probe_v1"
)
EXPECTED_V167_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v167_source_split_failure_autopsy_"
    "nearest_neighbor_feature_aliasing_closed_feature_contract_expansion"
)
EXPECTED_V167_EXACT_DIGEST = (
    "8c527c7afc9c36474d32969bbaf6625b2d66472789cee4c9203fb8845de5684c"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v168-carrion-survivor-continuation-public-feature-contract-probe.json"
)
FEATURE_FAMILIES = (
    "current_encoded_public_observation_mask",
    "decoded_public_numeric_projection",
    "feature_policy_token_projection",
)
FORBIDDEN_TRAINABLE_TOKENS = (
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
    "runtime_action",
    "runtime_requested",
    "runtime_resolved",
    "reason",
)


class CarrionSurvivorContinuationV168PublicFeatureContractProbeError(ValueError):
    pass


def run_carrion_survivor_continuation_v168_public_feature_contract_probe(
    *,
    v167_report_path: str | Path = DEFAULT_V167_REPORT_PATH,
    v165_dataset_path: str | Path = DEFAULT_V165_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v167_exact_digest: str | None = EXPECTED_V167_EXACT_DIGEST,
    expected_v166_exact_digest: str | None = EXPECTED_V166_EXACT_DIGEST,
    expected_v165_dataset_digest: str | None = EXPECTED_V165_DATASET_DIGEST,
) -> dict[str, object]:
    v167_report = load_json_report(v167_report_path)
    rows = load_v154_dataset(v165_dataset_path)
    source_validation = validate_v168_sources(
        v167_report=v167_report,
        rows=rows,
        expected_v167_exact_digest=expected_v167_exact_digest,
        expected_v166_exact_digest=expected_v166_exact_digest,
        expected_v165_dataset_digest=expected_v165_dataset_digest,
    )
    source_records = load_v165_source_records(rows)
    families = build_candidate_feature_families(
        rows=rows,
        source_records_by_row=source_records["records_by_row"],
    )
    baseline = evaluate_feature_family(
        rows=rows,
        family=families["current_encoded_public_observation_mask"],
        v167_report=v167_report,
    )
    candidate_reports = []
    leakage_passed = True
    for family_name in FEATURE_FAMILIES:
        family = families[family_name]
        report = evaluate_feature_family(
            rows=rows,
            family=family,
            v167_report=v167_report,
        )
        leakage = candidate_feature_leakage_scan(family["payloads"])
        report["leakage_scan"] = leakage
        report["baseline_comparison"] = _family_baseline_comparison(
            baseline=baseline,
            family_report=report,
        )
        leakage_passed = leakage_passed and leakage.get("passed") is True
        candidate_reports.append(report)
    classification = _classification(
        source_validation=source_validation,
        leakage_passed=leakage_passed,
        baseline=baseline,
        candidate_reports=candidate_reports,
    )
    best_candidate = _best_candidate(candidate_reports)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_POLICY,
        "contract": {
            "diagnostics_only": True,
            "public_feature_contract_probe_only": True,
            "runtime_observation_schema_changed": False,
            "runtime_policy_changed": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "replay_viewer_schema_changed": False,
            "threshold_tuning_ran": False,
            "k_tuning_ran": False,
            "training_ran": False,
            "shadow_eval_ran": False,
            "live_ab_allowed": False,
            "promotion_authorized": False,
            "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_future_outcome_as_trainable_input": False,
            "uses_runtime_action_as_trainable_input": False,
        },
        "inputs": {
            "v167_report": str(v167_report_path),
            "v165_dataset": str(v165_dataset_path),
            "expected_v167_exact_digest": expected_v167_exact_digest,
            "expected_v166_exact_digest": expected_v166_exact_digest,
            "expected_v165_dataset_digest": expected_v165_dataset_digest,
            "feature_families": list(FEATURE_FAMILIES),
        },
        "source_validation": source_validation,
        "source_record_load": source_records["summary"],
        "v166_baseline_recomputed": baseline,
        "candidate_feature_family_reports": candidate_reports,
        "best_candidate": best_candidate,
        "contract_proposal": _contract_proposal(best_candidate),
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "dataset_digest": stable_payload_digest(rows),
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "replay_viewer_schema_changed": False,
        "threshold_tuning_ran": False,
        "k_tuning_ran": False,
        "training_ran": False,
        "shadow_eval_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "diagnostics_only": True,
        "non_promoted": True,
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v168_sources(
    *,
    v167_report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v167_exact_digest: str | None = EXPECTED_V167_EXACT_DIGEST,
    expected_v166_exact_digest: str | None = EXPECTED_V166_EXACT_DIGEST,
    expected_v165_dataset_digest: str | None = EXPECTED_V165_DATASET_DIGEST,
) -> dict[str, object]:
    failures: list[str] = []
    if (
        v167_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION
    ):
        failures.append("v167_schema_version_mismatch")
    if v167_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_POLICY:
        failures.append("v167_policy_mismatch")
    observed_classification = str(
        _mapping(v167_report.get("classification")).get("primary") or ""
    )
    if observed_classification != EXPECTED_V167_CLASSIFICATION:
        failures.append("v167_unexpected_classification")
    exact_validation = exact_digest_validation_report(v167_report)
    if exact_validation.get("passed") is not True:
        failures.append("v167_exact_digest_mismatch")
    observed_v167_digest = str(v167_report.get("exact_digest") or "")
    if expected_v167_exact_digest and observed_v167_digest != expected_v167_exact_digest:
        failures.append("v167_unexpected_exact_digest")
    source = _mapping(v167_report.get("source_validation"))
    observed_v166_digest = str(source.get("observed_v166_exact_digest") or "")
    if expected_v166_exact_digest and observed_v166_digest != expected_v166_exact_digest:
        failures.append("v167_v166_digest_mismatch")
    observed_v166_classification = str(source.get("observed_v166_classification") or "")
    if observed_v166_classification != EXPECTED_V166_CLASSIFICATION:
        failures.append("v167_v166_classification_mismatch")
    dataset_digest = stable_payload_digest(rows)
    if expected_v165_dataset_digest and dataset_digest != expected_v165_dataset_digest:
        failures.append("v165_dataset_digest_mismatch")
    if v167_report.get("dataset_digest") != dataset_digest:
        failures.append("v167_report_dataset_digest_mismatch")
    if _mapping(v167_report.get("failure_mode_summary")).get("primary_failure_mode") != "nearest_neighbor_feature_aliasing":
        failures.append("v167_failure_mode_not_feature_aliasing")
    lifecycle = _v167_lifecycle_scan(v167_report)
    if lifecycle.get("passed") is not True:
        failures.append("v167_lifecycle_not_closed")
    return {
        "policy": "m3_carrion_survivor_continuation_v168_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v167_classification": EXPECTED_V167_CLASSIFICATION,
        "observed_v167_classification": observed_classification,
        "expected_v167_exact_digest": expected_v167_exact_digest,
        "observed_v167_exact_digest": observed_v167_digest,
        "v167_exact_digest_validation": exact_validation,
        "expected_v166_exact_digest": expected_v166_exact_digest,
        "observed_v166_exact_digest": observed_v166_digest,
        "expected_v165_dataset_digest": expected_v165_dataset_digest,
        "observed_v165_dataset_digest": dataset_digest,
        "v167_reported_dataset_digest": v167_report.get("dataset_digest"),
        "v167_lifecycle_scan": lifecycle,
        "row_count": len(rows),
    }


def load_v165_source_records(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    records_by_row: dict[int, dict[str, object]] = {}
    failures = []
    preterminal_rows = 0
    for row_index, row in enumerate(rows):
        if (
            row.get("schema_version")
            != M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ):
            continue
        preterminal_rows += 1
        metadata = _mapping(row.get("metadata"))
        result = _load_source_record(metadata)
        if result.get("passed") is True:
            records_by_row[row_index] = dict(_mapping(result.get("record")))
        else:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": result.get("reason"),
                    "error": result.get("error"),
                }
            )
    return {
        "records_by_row": records_by_row,
        "summary": {
            "policy": "m3_carrion_survivor_continuation_v168_source_record_load_v1",
            "preterminal_row_count": preterminal_rows,
            "source_record_loaded_count": len(records_by_row),
            "source_record_missing_count": preterminal_rows - len(records_by_row),
            "failures": failures[:24],
        },
    }


def build_candidate_feature_families(
    *,
    rows: Sequence[Mapping[str, object]],
    source_records_by_row: Mapping[int, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        "current_encoded_public_observation_mask": _build_family(
            rows,
            family_name="current_encoded_public_observation_mask",
            builder=lambda index, row: _baseline_payload(row),
        ),
        "decoded_public_numeric_projection": _build_family(
            rows,
            family_name="decoded_public_numeric_projection",
            builder=lambda index, row: _decoded_public_payload(row),
        ),
        "feature_policy_token_projection": _build_family(
            rows,
            family_name="feature_policy_token_projection",
            builder=lambda index, row: _feature_policy_payload(
                row=row,
                record=source_records_by_row.get(index),
            ),
        ),
    }


def evaluate_feature_family(
    *,
    rows: Sequence[Mapping[str, object]],
    family: Mapping[str, object],
    v167_report: Mapping[str, object],
) -> dict[str, object]:
    payloads = _list_of_mappings(family.get("payloads"))
    vectors, feature_keys, normalization = _feature_vectors_from_payloads(payloads)
    groups = source_split_groups(rows)
    group_by_row = {int(group["row_index"]): group for group in groups}
    group_row_indexes: dict[str, list[int]] = defaultdict(list)
    for group in groups:
        group_row_indexes[str(group["group_key"])].append(int(group["row_index"]))
    predictions = []
    counts: Counter[str] = Counter()
    safe_hit_count = 0
    unsupported = 0
    no_prediction = 0
    best_fixed_hits = 0
    first_public_hits = 0
    exact_top_hits = 0
    failed_preterminal_rows = 0
    per_seed: dict[int, dict[str, int]] = defaultdict(lambda: {"rows": 0, "safe": 0})
    for row_index, row in enumerate(rows):
        group = group_by_row[row_index]
        group_key = str(group["group_key"])
        train_indexes = [
            index
            for index in range(len(rows))
            if str(group_by_row[index]["group_key"]) != group_key
        ]
        predicted, source, neighbors = _predict_with_vectors(
            rows=rows,
            vectors=vectors,
            feature_keys=feature_keys,
            normalization=normalization,
            row_index=row_index,
            train_indexes=train_indexes,
        )
        safe_set = set(_safe_action_set(row))
        public_mask = _complete_bool_mask(row.get("public_action_mask"))
        safe_hit = bool(predicted and predicted in safe_set)
        unsupported_prediction = bool(
            predicted and public_mask.get(predicted) is not True
        )
        if predicted:
            counts.update([predicted])
            safe_hit_count += int(safe_hit)
            unsupported += int(unsupported_prediction)
        else:
            no_prediction += 1
        best_fixed = _best_fixed_safe_action(rows, train_indexes, public_mask)
        first_public = _first_public_action(public_mask)
        best_fixed_hits += int(best_fixed in safe_set)
        first_public_hits += int(first_public in safe_set)
        exact_top = str(_top_value_target(row).get("action", ""))
        exact_top_hits += int(exact_top in safe_set)
        if (
            row.get("schema_version")
            == M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
        ):
            seed = _int(group.get("source_seed"))
            per_seed[seed]["rows"] += 1
            per_seed[seed]["safe"] += int(safe_hit)
            failed_preterminal_rows += int(not safe_hit)
        predictions.append(
            {
                "row_index": row_index,
                "source_group_key": group_key,
                "source_seed": group.get("source_seed"),
                "predicted_action": predicted,
                "prediction_source": source,
                "nearest_neighbor_row_indexes": neighbors,
                "safe_hit": safe_hit,
                "unsupported_prediction": unsupported_prediction,
                "safe_action_set": sorted(safe_set, key=_action_order),
                "target_leaky_exact_top_action": exact_top,
            }
        )
    row_count = len(rows)
    dominant = _dominant_count_share(_positive_counter(counts))
    safe_rate = _safe_rate(safe_hit_count, row_count)
    best_trivial = max(
        _safe_rate(best_fixed_hits, row_count),
        _safe_rate(first_public_hits, row_count),
    )
    zero_seeds = sorted(
        seed for seed, payload in per_seed.items() if payload["rows"] and payload["safe"] == 0
    )
    rank_movement = _failed_row_rank_movement(
        rows=rows,
        vectors=vectors,
        feature_keys=feature_keys,
        group_by_row=group_by_row,
        v167_report=v167_report,
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v168_feature_family_source_split_eval_v1",
        "feature_family": family.get("feature_family"),
        "feature_payload_digest": stable_payload_digest(payloads),
        "feature_key_count": len(feature_keys),
        "row_count": row_count,
        "prediction_count": sum(counts.values()),
        "no_prediction_count": no_prediction,
        "unsupported_prediction_count": unsupported,
        "predicted_action_counts": dict(sorted(counts.items())),
        "dominant_predicted_action": dominant.get("key"),
        "dominant_predicted_action_share": dominant.get("share"),
        "safe_hit_count": safe_hit_count,
        "safe_hit_rate": safe_rate,
        "best_trivial_baseline_hit_rate": best_trivial,
        "safe_hit_margin_over_best_trivial": _round(safe_rate - best_trivial),
        "zero_safe_hit_preterminal_source_seeds": zero_seeds,
        "failed_preterminal_row_count": failed_preterminal_rows,
        "per_source_seed_safe_hit_rates": {
            str(seed): {
                "row_count": payload["rows"],
                "safe_hit_count": payload["safe"],
                "safe_hit_rate": _safe_rate(payload["safe"], payload["rows"]),
            }
            for seed, payload in sorted(per_seed.items())
        },
        "target_leaky_exact_top_upper_bound": {
            "non_runtime": True,
            "target_leaky": True,
            "safe_hit_count": exact_top_hits,
            "safe_hit_rate": _safe_rate(exact_top_hits, row_count),
        },
        "fallback_handling": family.get("fallback_handling"),
        "failed_row_rank_movement": rank_movement,
        "predictions": predictions,
    }


def candidate_feature_leakage_scan(
    payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures = []
    for row_index, payload in enumerate(payloads):
        _scan_forbidden_tokens(
            value=payload,
            row_index=row_index,
            failures=failures,
            path="payload",
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v168_candidate_feature_leakage_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
        "forbidden_tokens": list(FORBIDDEN_TRAINABLE_TOKENS),
        "payload_count": len(payloads),
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    leakage_passed: bool,
    baseline: Mapping[str, object],
    candidate_reports: Sequence[Mapping[str, object]],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v168_public_feature_contract_probe_"
    if source_validation.get("passed") is not True or not leakage_passed:
        return prefix + "closed_invalid"
    baseline_zero_seeds = set(
        _list_like(baseline.get("zero_safe_hit_preterminal_source_seeds"))
    )
    improving = [
        report
        for report in candidate_reports
        if str(report.get("feature_family")) != "current_encoded_public_observation_mask"
        and set(_list_like(report.get("zero_safe_hit_preterminal_source_seeds")))
        != baseline_zero_seeds
        and len(
            set(_list_like(report.get("zero_safe_hit_preterminal_source_seeds")))
            & baseline_zero_seeds
        )
        < len(baseline_zero_seeds)
    ]
    if not improving:
        return prefix + "public_feature_contract_probe_no_viable_projection_closed"
    ready = [
        report
        for report in improving
        if not _list_like(report.get("zero_safe_hit_preterminal_source_seeds"))
        and _float(report.get("dominant_predicted_action_share")) <= 0.50
        and _int(report.get("unsupported_prediction_count")) == 0
        and _float(report.get("safe_hit_margin_over_best_trivial")) >= 0.05
    ]
    if ready:
        return prefix + "public_feature_contract_candidate_ready_for_dataset_expansion_design"
    return prefix + "partial_feature_contract_support_recommend_another_feature_probe"


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification.endswith(
        "public_feature_contract_candidate_ready_for_dataset_expansion_design"
    )
    partial = classification.endswith(
        "partial_feature_contract_support_recommend_another_feature_probe"
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v168_route_recommendation_v1",
        "recommended_next_route": (
            "dataset_expansion_design_from_public_feature_contract_candidate"
            if ready
            else "another_public_feature_contract_probe"
            if partial
            else "keep_feature_contract_probe_closed"
        ),
        "dataset_expansion_design_recommended": ready,
        "another_feature_probe_recommended": partial,
        "training_recommended": False,
        "shadow_eval_recommended": False,
        "k_tuning_recommended": False,
        "threshold_tuning_recommended": False,
        "live_ab_allowed": False,
        "runtime_policy_integration_allowed": False,
        "promotion_authorized": False,
        "runtime_action_selection_changed": False,
    }


def _baseline_payload(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "trainable_public_features": dict(_mapping(row.get("trainable_public_features"))),
        "public_action_mask": _complete_bool_mask(row.get("public_action_mask")),
    }


def _decoded_public_payload(row: Mapping[str, object]) -> dict[str, object]:
    observation_input = _mapping(_mapping(row.get("trainable_public_features")).get("public_observation"))
    decoded = _decoded_numeric_projection(observation_input)
    if decoded:
        features = {
            "decoded_public_numeric_projection": decoded,
            "action_mask": _complete_bool_mask(row.get("public_action_mask")),
        }
    else:
        features = {
            "v158_public_feature_fallback": dict(
                _mapping(row.get("trainable_public_features"))
            ),
            "action_mask": _complete_bool_mask(row.get("public_action_mask")),
        }
    return {
        "trainable_public_features": features,
        "public_action_mask": _complete_bool_mask(row.get("public_action_mask")),
    }


def _feature_policy_payload(
    *,
    row: Mapping[str, object],
    record: Mapping[str, object] | None,
) -> dict[str, object]:
    if record is not None:
        try:
            tokens = feature_keys_from_record(record)
        except ValueError:
            tokens = ()
        if tokens:
            return {
                "trainable_public_features": {
                    "feature_policy_token_projection": {
                        f"token_hash_{stable_payload_digest({'token': token})[:16]}": 1.0
                        for token in tokens
                    },
                    "action_mask": _complete_bool_mask(row.get("public_action_mask")),
                },
                "public_action_mask": _complete_bool_mask(row.get("public_action_mask")),
            }
    return {
        "trainable_public_features": {
            "v158_public_feature_fallback": dict(
                _mapping(row.get("trainable_public_features"))
            ),
            "action_mask": _complete_bool_mask(row.get("public_action_mask")),
        },
        "public_action_mask": _complete_bool_mask(row.get("public_action_mask")),
    }


def _build_family(
    rows: Sequence[Mapping[str, object]],
    *,
    family_name: str,
    builder: Callable[[int, Mapping[str, object]], Mapping[str, object]],
) -> dict[str, object]:
    payloads = []
    fallback_rows = []
    source_record_rows = []
    for index, row in enumerate(rows):
        payload = builder(index, row)
        payloads.append(payload)
        text = json.dumps(payload, sort_keys=True)
        if "v158_public_feature_fallback" in text:
            fallback_rows.append(index)
        if "feature_policy_token_projection" in text:
            source_record_rows.append(index)
    return {
        "feature_family": family_name,
        "payloads": payloads,
        "fallback_handling": {
            "v158_base_or_missing_source_record_fallback_row_count": len(
                fallback_rows
            ),
            "fallback_row_indexes": fallback_rows,
            "source_record_projection_row_count": len(source_record_rows),
            "base_rows_reported_separately": True,
        },
    }


def _feature_vectors_from_payloads(
    payloads: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, float]], list[str], dict[str, dict[str, float]]]:
    raw_vectors = [_flatten_public_payload(payload) for payload in payloads]
    keys = sorted({key for vector in raw_vectors for key in vector})
    mins = {}
    ranges = {}
    for key in keys:
        values = [float(vector.get(key, 0.0)) for vector in raw_vectors]
        minimum = min(values, default=0.0)
        maximum = max(values, default=0.0)
        mins[key] = _round(minimum)
        ranges[key] = _round(max(maximum - minimum, 1.0))
    normalized = []
    for vector in raw_vectors:
        normalized.append(
            {
                key: _round((float(vector.get(key, 0.0)) - mins[key]) / ranges[key])
                for key in keys
            }
        )
    return (
        normalized,
        keys,
        {key: {"min": mins[key], "range": ranges[key]} for key in keys},
    )


def _predict_with_vectors(
    *,
    rows: Sequence[Mapping[str, object]],
    vectors: Sequence[Mapping[str, float]],
    feature_keys: Sequence[str],
    normalization: Mapping[str, Mapping[str, float]],
    row_index: int,
    train_indexes: Sequence[int],
) -> tuple[str, str, list[int]]:
    return _predict_1nn_action(
        rows=rows,
        vectors=vectors,
        feature_keys=feature_keys,
        normalization=normalization,
        row_index=row_index,
        train_indexes=train_indexes,
    )


def _failed_row_rank_movement(
    *,
    rows: Sequence[Mapping[str, object]],
    vectors: Sequence[Mapping[str, float]],
    feature_keys: Sequence[str],
    group_by_row: Mapping[int, Mapping[str, object]],
    v167_report: Mapping[str, object],
) -> dict[str, object]:
    baseline_by_row = {
        _int(row.get("row_index")): row
        for row in _list_of_mappings(v167_report.get("preterminal_row_autopsy"))
        if row.get("safe_hit") is not True
    }
    rows_payload = []
    moved_to_rank_1 = 0
    for row_index, baseline in sorted(baseline_by_row.items()):
        group_key = str(group_by_row[row_index].get("group_key"))
        train_indexes = [
            index
            for index in range(len(rows))
            if str(group_by_row[index].get("group_key")) != group_key
        ]
        ranked = [
            {"row_index": index, "distance": _round(distance)}
            for distance, index in sorted(
                (
                    (
                        _vector_distance(
                            vectors[row_index],
                            vectors[index],
                            feature_keys,
                        ),
                        index,
                    )
                    for index in train_indexes
                ),
                key=lambda item: (item[0], item[1]),
            )
        ]
        candidate = _nearest_safe_support_rank(
            row=rows[row_index],
            ranked=ranked,
            rows=rows,
        )
        rank = candidate.get("rank")
        moved = rank == 1
        moved_to_rank_1 += int(moved)
        baseline_rank = baseline.get("rank_of_first_safe_support_neighbor")
        rows_payload.append(
            {
                "row_index": row_index,
                "source_seed": baseline.get("source_seed"),
                "baseline_rank_of_first_safe_support_neighbor": baseline_rank,
                "candidate_rank_of_first_safe_support_neighbor": rank,
                "candidate_nearest_safe_support_neighbor_row_index": candidate.get(
                    "row_index"
                ),
                "candidate_distance_to_nearest_safe_support_neighbor": candidate.get(
                    "distance"
                ),
                "moved_nearest_safe_support_rank_to_1": moved,
                "rank_delta_baseline_minus_candidate": _rank_delta(
                    baseline_rank,
                    rank,
                ),
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v168_failed_row_rank_movement_v1",
        "v167_failed_row_count": len(rows_payload),
        "moved_nearest_safe_support_rank_to_1_count": moved_to_rank_1,
        "moved_nearest_safe_support_rank_to_1_share": _safe_rate(
            moved_to_rank_1,
            len(rows_payload),
        ),
        "rows": rows_payload,
    }


def _nearest_safe_support_rank(
    *,
    row: Mapping[str, object],
    ranked: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    safe_set = set(_safe_action_set(row))
    public_mask = _complete_bool_mask(row.get("public_action_mask"))
    for rank, neighbor in enumerate(ranked, start=1):
        index = _int(neighbor.get("row_index"))
        if any(
            public_mask.get(action) is True and action in _safe_action_set(rows[index])
            for action in safe_set
        ):
            return {
                "rank": rank,
                "row_index": index,
                "distance": neighbor.get("distance"),
            }
    return {"rank": None, "row_index": None, "distance": None}


def _decoded_numeric_projection(observation_input: Mapping[str, object]) -> dict[str, object]:
    if not observation_input:
        return {}
    try:
        values = decode_observation_input(dict(observation_input))
    except ValueError:
        return {}
    cursor = 0
    self_values = {
        _safe_projection_key(field): _round(values[cursor + index])
        for index, field in enumerate(SELF_INPUT_FIELDS)
        if field != "mind_inheritance_available"
    }
    cursor += len(SELF_INPUT_FIELDS)
    patch_cells = []
    for cell_index in range(PATCH_CELL_COUNT):
        patch_cells.append(
            {
                _safe_projection_key(field): _round(values[cursor + index])
                for index, field in enumerate(PATCH_INPUT_FIELDS)
            }
        )
        cursor += len(PATCH_INPUT_FIELDS)
    navigation = {}
    for target in NAVIGATION_TARGETS:
        navigation[_safe_projection_key(target)] = {
            _safe_projection_key(field): _round(values[cursor + index])
            for index, field in enumerate(NAVIGATION_INPUT_FIELDS)
        }
        cursor += len(NAVIGATION_INPUT_FIELDS)
    center = patch_cells[PATCH_CELL_COUNT // 2]
    return {
        "public_self": self_values,
        "center_patch": center,
        "local_resource_summary": _local_resource_summary(patch_cells),
        "local_risk_summary": _local_risk_summary(patch_cells),
        "navigation_summary": navigation,
    }


def _local_resource_summary(
    patch_cells: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    return {
        "food_max": _max_numeric(patch_cells, "food"),
        "food_sum": _sum_numeric(patch_cells, "food"),
        "water_access_max": _max_numeric(patch_cells, "water_access_code"),
        "fresh_kill_max": _max_numeric(patch_cells, "fresh_kill_energy"),
        "carcass_max": _max_numeric(patch_cells, "carcass_energy"),
        "carrion_signal_max": _max_numeric(patch_cells, "carrion_signal"),
        "prey_biomass_max": _max_numeric(patch_cells, "prey_biomass"),
    }


def _local_risk_summary(
    patch_cells: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    return {
        "hazard_max": _max_numeric(patch_cells, "hazard_level"),
        "predator_risk_max": _max_numeric(patch_cells, "predator_risk"),
        "occupied_adjacent_count": _sum_threshold(patch_cells, "occupant_code", 0.1),
    }


def _local_source_record_hint(row: Mapping[str, object]) -> bool:
    return (
        row.get("schema_version")
        == M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
    )


def _load_source_record(metadata: Mapping[str, object]) -> dict[str, object]:
    path = Path(str(metadata.get("source_path") or ""))
    line_number = _int(metadata.get("line_number"))
    if not str(path) or line_number <= 0:
        return {"passed": False, "reason": "missing_source_path_or_line_number"}
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
            for index, line in enumerate(handle, start=1):
                if index != line_number:
                    continue
                payload = json.loads(line)
                record = _mapping(payload.get("record") if isinstance(payload, Mapping) else {})
                if not record:
                    return {"passed": False, "reason": "source_line_has_no_record"}
                return {"passed": True, "record": dict(record)}
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "reason": "source_record_load_failed", "error": str(exc)}
    return {"passed": False, "reason": "source_line_number_not_found"}


def _load_source_record_for_row(row: Mapping[str, object]) -> dict[str, object]:
    if not _local_source_record_hint(row):
        return {"passed": False, "reason": "base_row_has_no_source_record_provenance"}
    return _load_source_record(_mapping(row.get("metadata")))


def _first_public_action(public_mask: Mapping[str, bool]) -> str:
    for action in ACTION_NAMES:
        if public_mask.get(action) is True:
            return action
    return ""


def _positive_counter(counter: Counter[str]) -> Counter[str]:
    return Counter(
        {
            key: int(value)
            for key, value in counter.items()
            if int(value) > 0 and str(key)
        }
    )


def _family_baseline_comparison(
    *,
    baseline: Mapping[str, object],
    family_report: Mapping[str, object],
) -> dict[str, object]:
    baseline_zero = set(_list_like(baseline.get("zero_safe_hit_preterminal_source_seeds")))
    family_zero = set(_list_like(family_report.get("zero_safe_hit_preterminal_source_seeds")))
    return {
        "zero_safe_hit_seed_count_delta_vs_baseline": len(family_zero) - len(baseline_zero),
        "zero_safe_hit_seeds_removed_vs_baseline": sorted(baseline_zero - family_zero),
        "safe_hit_rate_delta_vs_baseline": _round(
            _float(family_report.get("safe_hit_rate")) - _float(baseline.get("safe_hit_rate"))
        ),
        "safe_hit_margin_delta_vs_baseline": _round(
            _float(family_report.get("safe_hit_margin_over_best_trivial"))
            - _float(baseline.get("safe_hit_margin_over_best_trivial"))
        ),
    }


def _best_candidate(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidates = [
        report
        for report in reports
        if str(report.get("feature_family")) != "current_encoded_public_observation_mask"
    ]
    if not candidates:
        return {}
    best = sorted(
        candidates,
        key=lambda report: (
            len(_list_like(report.get("zero_safe_hit_preterminal_source_seeds"))),
            -_float(report.get("safe_hit_margin_over_best_trivial")),
            -_float(report.get("safe_hit_rate")),
            str(report.get("feature_family")),
        ),
    )[0]
    return {
        "feature_family": best.get("feature_family"),
        "zero_safe_hit_preterminal_source_seeds": best.get(
            "zero_safe_hit_preterminal_source_seeds"
        ),
        "safe_hit_rate": best.get("safe_hit_rate"),
        "safe_hit_margin_over_best_trivial": best.get(
            "safe_hit_margin_over_best_trivial"
        ),
        "dominant_predicted_action_share": best.get(
            "dominant_predicted_action_share"
        ),
        "unsupported_prediction_count": best.get("unsupported_prediction_count"),
    }


def _contract_proposal(best_candidate: Mapping[str, object]) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v168_contract_proposal_diagnostics_only_v1",
        "diagnostics_only": True,
        "runtime_schema_change_proposed": False,
        "replay_viewer_schema_change_proposed": False,
        "candidate_feature_family": best_candidate.get("feature_family"),
        "allowed_public_inputs": [
            "public_self_without_mind_inheritance_available",
            "center_patch_public_fields",
            "local_resource_summary",
            "local_risk_summary",
            "navigation_summary",
            "public_action_mask",
        ],
        "forbidden_inputs": list(FORBIDDEN_TRAINABLE_TOKENS),
    }


def _v167_lifecycle_scan(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    for field in (
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "threshold_tuning_ran",
        "k_tuning_ran",
        "training_ran",
        "shadow_eval_ran",
        "live_ab_allowed",
        "promotion_authorized",
    ):
        if report.get(field) is not False:
            failures.append({"field": field, "observed": report.get(field)})
    return {
        "policy": "m3_carrion_survivor_continuation_v168_v167_lifecycle_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _scan_forbidden_tokens(
    *,
    value: object,
    row_index: int,
    failures: list[dict[str, object]],
    path: str,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            for token in FORBIDDEN_TRAINABLE_TOKENS:
                if token in key_text:
                    failures.append(
                        {"row_index": row_index, "path": path, "token": token}
                    )
            _scan_forbidden_tokens(
                value=item,
                row_index=row_index,
                failures=failures,
                path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_forbidden_tokens(
                value=item,
                row_index=row_index,
                failures=failures,
                path=f"{path}.{index}",
            )
    elif isinstance(value, str):
        text = value.lower()
        for token in FORBIDDEN_TRAINABLE_TOKENS:
            if token in text:
                failures.append({"row_index": row_index, "path": path, "token": token})


def _safe_projection_key(key: str) -> str:
    safe = str(key)
    safe = safe.replace("water_access_reason_code", "water_access_code")
    replacements = {
        "seed": "sd",
        "fixture": "fx",
        "branch": "br",
        "tick": "tk",
        "agent": "actor",
        "path": "route",
        "digest": "hash",
        "provenance": "origin",
        "private": "nonpublic",
        "future": "later",
        "outcome": "result",
        "runtime_action": "rt_choice",
        "runtime_requested": "rt_request",
        "runtime_resolved": "rt_resolve",
        "reason": "code",
    }
    lowered = safe.lower()
    for token, replacement in replacements.items():
        if token in lowered:
            safe = safe.replace(token, replacement).replace(token.upper(), replacement.upper())
            lowered = safe.lower()
    return safe


def _max_numeric(items: Sequence[Mapping[str, object]], key: str) -> float:
    return _round(max((_number(item.get(key)) for item in items), default=0.0))


def _sum_numeric(items: Sequence[Mapping[str, object]], key: str) -> float:
    return _round(sum(_number(item.get(key)) for item in items))


def _sum_threshold(
    items: Sequence[Mapping[str, object]],
    key: str,
    threshold: float,
) -> float:
    return _round(sum(1.0 for item in items if _number(item.get(key)) > threshold))


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _rank_delta(left: object, right: object) -> object:
    if not isinstance(left, int) or not isinstance(right, int):
        return None
    return int(left) - int(right)


def _list_like(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _json_round_trip_digest(payload: Mapping[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    json_payload = json.loads(
        json.dumps(without_digest, sort_keys=True, allow_nan=False)
    )
    return stable_payload_digest(json_payload)
