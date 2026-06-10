from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    build_linear_baseline_cache,
    build_safe_archive_diagnostic_artifact,
    safe_archive_expansion_leakage_scan,
    _baseline_summary,
    _comparison_against_baseline,
    _dominant_count_share,
    _float,
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    _run_policy,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    DEFAULT_DATASET_OUTPUT_PATH as DEFAULT_V154_DATASET_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V154_REPORT_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_DATASET_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_POLICY,
    TARGET_CARRION_SEEDS,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.support_gated_residual import (
    load_support_gated_residual_artifact,
    score_support_gated_residual_artifact,
)
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)

M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_archive_train_eval_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_archive_train_eval_v1"
)
EXPECTED_V154_SUPPORT_READY_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_archive_support_ready_no_training"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v155-carrion-survivor-continuation-train-eval.json"
)
DEFAULT_ARTIFACT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v155-carrion-survivor-continuation-support-gated-artifact.json"
)
DEFAULT_MIN_LABEL_COUNT = 100
MAX_DOMINANT_LABEL_ACTION_SHARE = 0.50
STRICT_BROAD_SEEDS = (5, 13, 19, 29, 37, 41)
STRICT_CARRION_SEEDS = TARGET_CARRION_SEEDS
STRICT_LIVE_AB_TICKS = 120
DEFAULT_MIN_NN_OVER_TRIVIAL_MARGIN = 0.05
DEFAULT_MAX_TRIVIAL_BASELINE_ACCURACY = 0.80
V155_CANDIDATE_ID = "carrion_survivor_continuation_support_gated_residual"
V155_SUPPORT_MODE = "v155_carrion_survivor_continuation_support"
V155_TRAINING_SOURCE_LABEL = "v154_carrion_survivor_continuation_archive"


class CarrionSurvivorContinuationTrainEvalError(ValueError):
    pass


def load_json_report(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CarrionSurvivorContinuationTrainEvalError(
            f"JSON report must be an object: {path}"
        )
    return payload


def load_v154_dataset(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise CarrionSurvivorContinuationTrainEvalError(
                f"v154 dataset row {line_number} must be a JSON object"
            )
        rows.append(payload)
    return rows


def run_carrion_survivor_continuation_train_eval(
    *,
    v154_report_path: str | Path = DEFAULT_V154_REPORT_PATH,
    v154_dataset_path: str | Path = DEFAULT_V154_DATASET_PATH,
    artifact_output_path: str | Path = DEFAULT_ARTIFACT_OUTPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    min_label_count: int = DEFAULT_MIN_LABEL_COUNT,
    max_dominant_label_action_share: float = MAX_DOMINANT_LABEL_ACTION_SHARE,
    min_nn_over_trivial_margin: float = DEFAULT_MIN_NN_OVER_TRIVIAL_MARGIN,
    max_trivial_baseline_accuracy: float = DEFAULT_MAX_TRIVIAL_BASELINE_ACCURACY,
    expected_dataset_digest: str | None = None,
    run_evaluation: bool = True,
    run_live_ab: bool = True,
    broad_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    carrion_seeds: Sequence[int] = STRICT_CARRION_SEEDS,
    ticks: int = STRICT_LIVE_AB_TICKS,
) -> dict[str, object]:
    v154_report = load_json_report(v154_report_path)
    dataset_rows = load_v154_dataset(v154_dataset_path)
    source_validation = validate_v154_train_eval_inputs(
        v154_report=v154_report,
        dataset_rows=dataset_rows,
        min_label_count=min_label_count,
        max_dominant_label_action_share=max_dominant_label_action_share,
        expected_dataset_digest=expected_dataset_digest,
        target_seeds=carrion_seeds,
    )
    branch_results = _list_of_mappings(
        v154_report.get("continuation_branch_results")
    )
    pretraining_review = (
        build_pretraining_review(
            dataset_rows=dataset_rows,
            branch_results=branch_results,
            target_seeds=carrion_seeds,
            min_nn_over_trivial_margin=min_nn_over_trivial_margin,
            max_trivial_baseline_accuracy=max_trivial_baseline_accuracy,
        )
        if source_validation.get("passed") is True
        else _skipped_pretraining_review("source_validation_failed")
    )
    artifact: dict[str, object]
    runtime_artifact: dict[str, object] | None = None
    training: dict[str, object]
    evaluation: dict[str, object]
    acceptance: dict[str, object]
    artifact_output = Path(artifact_output_path)
    if source_validation.get("passed") is not True:
        artifact = _artifact_status(
            artifact_output,
            created=False,
            reason="source_validation_failed",
        )
        training = _skipped_training("source_validation_failed")
        evaluation = _skipped_evaluation("source_validation_failed")
        acceptance = _closed_acceptance(
            reason="source_validation_failed",
            validation=source_validation,
            pretraining_review=pretraining_review,
        )
    elif pretraining_review.get("passed") is not True:
        artifact = _artifact_status(
            artifact_output,
            created=False,
            reason="pretraining_review_failed",
        )
        training = _skipped_training("pretraining_review_failed")
        evaluation = _skipped_evaluation("pretraining_review_failed")
        acceptance = _closed_acceptance(
            reason="pretraining_review_failed",
            validation=source_validation,
            pretraining_review=pretraining_review,
        )
    else:
        artifact_payload = build_safe_archive_diagnostic_artifact(
            safe_archive_dataset_rows=dataset_rows,
            candidate_id=V155_CANDIDATE_ID,
            support_mode=V155_SUPPORT_MODE,
            training_source_label=V155_TRAINING_SOURCE_LABEL,
        )
        write_json(artifact_output, artifact_payload)
        runtime_artifact = load_support_gated_residual_artifact(artifact_output)
        artifact_digest = stable_payload_digest(runtime_artifact)
        artifact = {
            "created": True,
            "path": str(artifact_output),
            "digest": artifact_digest,
            "schema_version": runtime_artifact.get("schema_version"),
            "policy": runtime_artifact.get("policy"),
            "training_row_count": runtime_artifact.get("training_row_count"),
            "support_action_counts": runtime_artifact.get("support_action_counts"),
            "diagnostics_only": True,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
        }
        training = build_training_summary(
            artifact=runtime_artifact,
            dataset_rows=dataset_rows,
            branch_results=branch_results,
        )
        if run_evaluation:
            evaluation = run_shadow_then_live_evaluation(
                artifact=runtime_artifact,
                output_path=output_path,
                run_live_ab=run_live_ab,
                broad_seeds=broad_seeds,
                carrion_seeds=carrion_seeds,
                ticks=ticks,
            )
            acceptance = build_top_level_acceptance(
                validation=source_validation,
                pretraining_review=pretraining_review,
                evaluation=evaluation,
            )
        else:
            evaluation = _skipped_evaluation("run_evaluation_false")
            acceptance = _closed_acceptance(
                reason="evaluation_not_run",
                validation=source_validation,
                pretraining_review=pretraining_review,
            )
    report = {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_SCHEMA_VERSION,
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_TRAIN_EVAL_POLICY,
        "contract": {
            "diagnostics_only": True,
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
            "default_runtime_load": False,
            "runtime_artifact_promoted": False,
            "live_ab_enabled_by_default": False,
            "uses_private_world_state_as_trainable_input": False,
            "uses_seed_fixture_branch_path_digest_provenance_as_trainable_input": False,
        },
        "inputs": {
            "v154_report": str(v154_report_path),
            "v154_dataset": str(v154_dataset_path),
            "artifact_output": str(artifact_output_path),
            "min_label_count": int(min_label_count),
            "max_dominant_label_action_share": _round(
                max_dominant_label_action_share
            ),
            "min_nn_over_trivial_margin": _round(min_nn_over_trivial_margin),
            "max_trivial_baseline_accuracy": _round(max_trivial_baseline_accuracy),
            "expected_dataset_digest": expected_dataset_digest,
            "run_evaluation": bool(run_evaluation),
            "run_live_ab": bool(run_live_ab),
            "broad_seeds": [int(seed) for seed in broad_seeds],
            "carrion_seeds": [int(seed) for seed in carrion_seeds],
            "ticks": int(ticks),
        },
        "source_validation": source_validation,
        "pretraining_review": pretraining_review,
        "artifact": artifact,
        "training": training,
        "evaluation": evaluation,
        "acceptance": acceptance,
        "classification": _classification(
            source_validation=source_validation,
            pretraining_review=pretraining_review,
            artifact=artifact,
            evaluation=evaluation,
            acceptance=acceptance,
        ),
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


def validate_v154_train_eval_inputs(
    *,
    v154_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    min_label_count: int = DEFAULT_MIN_LABEL_COUNT,
    max_dominant_label_action_share: float = MAX_DOMINANT_LABEL_ACTION_SHARE,
    expected_dataset_digest: str | None = None,
    target_seeds: Sequence[int] = STRICT_CARRION_SEEDS,
) -> dict[str, object]:
    failures: list[str] = []
    classification = _mapping(v154_report.get("classification")).get("primary")
    source_integrity = _mapping(v154_report.get("source_integrity"))
    support_floors = _mapping(v154_report.get("support_floors"))
    dataset = _mapping(v154_report.get("dataset"))
    generation_status = _mapping(v154_report.get("generation_status"))
    report_leakage = _mapping(dataset.get("leakage_scan"))
    dataset_digest = stable_payload_digest(dataset_rows)
    action_counts = Counter(_label_action(row) for row in dataset_rows)
    action_counts.pop("", None)
    dominant = _dominant_count_share(action_counts)
    seed_counts = Counter(_int(_mapping(row.get("metadata")).get("seed")) for row in dataset_rows)
    missing_seeds = [
        int(seed) for seed in target_seeds if seed_counts.get(int(seed), 0) <= 0
    ]
    row_schema_failures = [
        index
        for index, row in enumerate(dataset_rows)
        if row.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_DATASET_ROW_SCHEMA_VERSION
    ]
    feature_policy_failures = [
        index
        for index, row in enumerate(dataset_rows)
        if _mapping(row.get("trainable")).get("feature_policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_FEATURE_POLICY
    ]
    leakage_scan = safe_archive_expansion_leakage_scan(
        dataset_rows,
        strict_heldout_seeds=target_seeds,
    )
    if (
        v154_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION
    ):
        failures.append("v154_schema_version_mismatch")
    if v154_report.get("policy") != M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_POLICY:
        failures.append("v154_policy_mismatch")
    if classification != EXPECTED_V154_SUPPORT_READY_CLASSIFICATION:
        failures.append("v154_not_support_ready")
    if v154_report.get("diagnostics_only") is not True:
        failures.append("v154_diagnostics_only_not_true")
    if v154_report.get("training_authorized") is not False:
        failures.append("v154_training_authorized_not_false")
    if v154_report.get("promotion_authorized") is not False:
        failures.append("v154_promotion_authorized_not_false")
    if v154_report.get("runtime_promotion_allowed") is not False:
        failures.append("v154_runtime_promotion_allowed_not_false")
    if v154_report.get("default_runtime_behavior_changed") is not False:
        failures.append("v154_default_runtime_behavior_changed_not_false")
    if v154_report.get("runtime_action_selection_changed") is not False:
        failures.append("v154_runtime_action_selection_changed_not_false")
    if source_integrity.get("passed") is not True:
        failures.append("v154_source_integrity_failed")
    if list(source_integrity.get("failures") or []):
        failures.append("v154_source_integrity_failures_present")
    if source_integrity.get("replay_verification_complete") is not True:
        failures.append("v154_replay_verification_incomplete")
    if source_integrity.get("leakage_scan_passed") is not True:
        failures.append("v154_source_leakage_scan_failed")
    if _int(source_integrity.get("heuristic_action_source_count")) != 0:
        failures.append("v154_heuristic_action_source_count_nonzero")
    if support_floors.get("passed") is not True:
        failures.append("v154_support_floors_failed")
    if generation_status.get("state") != "complete":
        failures.append("v154_generation_not_complete")
    if generation_status.get("partial") is True:
        failures.append("v154_generation_partial")
    if int(dataset.get("row_count", -1)) != len(dataset_rows):
        failures.append("v154_dataset_row_count_mismatch")
    if int(dataset.get("row_count", 0)) < int(min_label_count):
        failures.append("v154_report_label_count_below_floor")
    if len(dataset_rows) < int(min_label_count):
        failures.append("v154_dataset_label_count_below_floor")
    if dataset.get("dataset_digest") != dataset_digest:
        failures.append("v154_dataset_digest_mismatch")
    if expected_dataset_digest and dataset_digest != str(expected_dataset_digest):
        failures.append("v154_dataset_digest_unexpected")
    if _float(dataset.get("dominant_label_action_share")) > float(
        max_dominant_label_action_share
    ):
        failures.append("v154_dominant_label_action_share_gt_floor")
    if _float(dominant.get("share")) > float(max_dominant_label_action_share):
        failures.append("v154_dataset_dominant_label_action_share_gt_floor")
    if report_leakage.get("passed") is not True:
        failures.append("v154_report_leakage_scan_failed")
    if leakage_scan.get("passed") is not True:
        failures.append("v154_trainable_leakage_scan_failed")
    if missing_seeds:
        failures.append("v154_missing_target_seed_labels")
    if row_schema_failures:
        failures.append("v154_dataset_row_schema_mismatch")
    if feature_policy_failures:
        failures.append("v154_dataset_feature_policy_mismatch")
    return {
        "policy": "m3_carrion_survivor_continuation_train_eval_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_classification": EXPECTED_V154_SUPPORT_READY_CLASSIFICATION,
        "observed_classification": classification,
        "dataset_digest": dataset_digest,
        "report_dataset_digest": dataset.get("dataset_digest"),
        "expected_dataset_digest": expected_dataset_digest,
        "label_count": len(dataset_rows),
        "min_label_count": int(min_label_count),
        "action_counts": dict(sorted(action_counts.items())),
        "dominant_label_action": dominant.get("key"),
        "dominant_label_action_share": dominant.get("share"),
        "max_dominant_label_action_share": _round(
            max_dominant_label_action_share
        ),
        "per_seed_support": [
            {"fixture": "carrion_only", "seed": int(seed), "label_count": seed_counts.get(int(seed), 0)}
            for seed in target_seeds
        ],
        "missing_target_seed_labels": missing_seeds,
        "source_integrity": {
            "passed": source_integrity.get("passed"),
            "replay_verification_complete": source_integrity.get(
                "replay_verification_complete"
            ),
            "leakage_scan_passed": source_integrity.get("leakage_scan_passed"),
            "heuristic_action_source_count": source_integrity.get(
                "heuristic_action_source_count"
            ),
        },
        "support_floors_passed": support_floors.get("passed"),
        "generation_state": generation_status.get("state"),
        "generation_partial": generation_status.get("partial"),
        "trainable_leakage_scan": leakage_scan,
    }


def build_pretraining_review(
    *,
    dataset_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    target_seeds: Sequence[int] = STRICT_CARRION_SEEDS,
    min_nn_over_trivial_margin: float = DEFAULT_MIN_NN_OVER_TRIVIAL_MARGIN,
    max_trivial_baseline_accuracy: float = DEFAULT_MAX_TRIVIAL_BASELINE_ACCURACY,
) -> dict[str, object]:
    alias_audit = build_alias_collision_audit(dataset_rows)
    action_only = build_action_only_baseline(dataset_rows, target_seeds=target_seeds)
    mask_only = build_mask_only_baseline(dataset_rows, target_seeds=target_seeds)
    loo = build_leave_one_seed_out_label_generalization(
        dataset_rows=dataset_rows,
        branch_results=branch_results,
        target_seeds=target_seeds,
    )
    trivial_best = max(
        _float(action_only.get("accuracy")),
        _float(mask_only.get("accuracy")),
    )
    nn_margin = _round(_float(loo.get("accuracy")) - trivial_best)
    blockers: list[dict[str, object]] = []
    if alias_audit.get("conflicting_exact_feature_group_count", 0):
        blockers.append(
            _blocker(
                "exact_public_feature_alias_conflicts",
                observed=alias_audit.get("conflicting_exact_feature_group_count"),
                required=0,
            )
        )
    if alias_audit.get("conflicting_exact_feature_row_count", 0):
        blockers.append(
            _blocker(
                "exact_public_feature_conflicting_rows",
                observed=alias_audit.get("conflicting_exact_feature_row_count"),
                required=0,
            )
        )
    if loo.get("all_target_seeds_evaluated") is not True:
        blockers.append(
            _blocker(
                "leave_one_seed_out_missing_target_seed",
                observed=loo.get("missing_target_seeds"),
                required=[],
            )
        )
    if nn_margin < float(min_nn_over_trivial_margin):
        blockers.append(
            _blocker(
                "nearest_neighbor_not_better_than_trivial_priors",
                observed=nn_margin,
                required=f">= {float(min_nn_over_trivial_margin):.6f}",
            )
        )
    if _float(action_only.get("accuracy")) >= float(max_trivial_baseline_accuracy):
        blockers.append(
            _blocker(
                "action_only_baseline_too_predictive",
                observed=action_only.get("accuracy"),
                required=f"< {float(max_trivial_baseline_accuracy):.6f}",
            )
        )
    if _float(mask_only.get("accuracy")) >= float(max_trivial_baseline_accuracy):
        blockers.append(
            _blocker(
                "mask_only_baseline_too_predictive",
                observed=mask_only.get("accuracy"),
                required=f"< {float(max_trivial_baseline_accuracy):.6f}",
            )
        )
    return {
        "policy": "m3_carrion_survivor_continuation_pretraining_review_v1",
        "passed": not blockers,
        "blockers": blockers,
        "fail_closed": bool(blockers),
        "leave_one_seed_out_label_generalization": loo,
        "action_only_baseline": action_only,
        "mask_only_baseline": mask_only,
        "nearest_neighbor_alias_collision_audit": alias_audit,
        "nearest_neighbor_minus_best_trivial_accuracy": nn_margin,
        "min_nn_over_trivial_margin": _round(min_nn_over_trivial_margin),
        "max_trivial_baseline_accuracy": _round(max_trivial_baseline_accuracy),
        "trivial_memorization_assessment": (
            "public_feature_alias_or_trivial_prior_blocked"
            if blockers
            else "non_trivial_public_feature_signal_available"
        ),
    }


def build_alias_collision_audit(
    dataset_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    feature_groups: dict[str, list[tuple[int, Mapping[str, object]]]] = defaultdict(list)
    mask_groups: dict[str, list[tuple[int, Mapping[str, object]]]] = defaultdict(list)
    for index, row in enumerate(dataset_rows):
        trainable = _mapping(row.get("trainable"))
        features = _mapping(trainable.get("features"))
        feature_groups[stable_payload_digest(features)].append((index, row))
        mask_groups[stable_payload_digest(_mapping(features.get("action_mask")))].append(
            (index, row)
        )
    exact_conflicts = _conflicting_groups(feature_groups)
    mask_conflicts = _conflicting_groups(mask_groups)
    duplicate_feature_row_count = sum(
        max(len(items) - 1, 0) for items in feature_groups.values()
    )
    duplicate_mask_row_count = sum(
        max(len(items) - 1, 0) for items in mask_groups.values()
    )
    return {
        "policy": "m3_carrion_survivor_continuation_nn_alias_collision_audit_v1",
        "row_count": len(dataset_rows),
        "exact_feature_group_count": len(feature_groups),
        "duplicate_exact_feature_row_count": duplicate_feature_row_count,
        "conflicting_exact_feature_group_count": len(exact_conflicts),
        "conflicting_exact_feature_row_count": sum(
            int(item["row_count"]) for item in exact_conflicts
        ),
        "mask_signature_group_count": len(mask_groups),
        "duplicate_mask_signature_row_count": duplicate_mask_row_count,
        "conflicting_mask_signature_group_count": len(mask_conflicts),
        "conflicting_mask_signature_row_count": sum(
            int(item["row_count"]) for item in mask_conflicts
        ),
        "max_rows_per_exact_feature": max(
            (len(items) for items in feature_groups.values()),
            default=0,
        ),
        "max_label_action_count_per_exact_feature": max(
            (len(_label_counts(items)) for items in feature_groups.values()),
            default=0,
        ),
        "conflicting_exact_feature_examples": exact_conflicts[:8],
        "conflicting_mask_signature_examples": mask_conflicts[:8],
    }


def build_action_only_baseline(
    dataset_rows: Sequence[Mapping[str, object]],
    *,
    target_seeds: Sequence[int] = STRICT_CARRION_SEEDS,
) -> dict[str, object]:
    return _leave_one_seed_out_prior_baseline(
        dataset_rows,
        target_seeds=target_seeds,
        grouping_key=None,
        policy="m3_carrion_survivor_continuation_action_only_baseline_v1",
    )


def build_mask_only_baseline(
    dataset_rows: Sequence[Mapping[str, object]],
    *,
    target_seeds: Sequence[int] = STRICT_CARRION_SEEDS,
) -> dict[str, object]:
    return _leave_one_seed_out_prior_baseline(
        dataset_rows,
        target_seeds=target_seeds,
        grouping_key=_row_mask_digest,
        policy="m3_carrion_survivor_continuation_mask_only_baseline_v1",
    )


def build_leave_one_seed_out_label_generalization(
    *,
    dataset_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
    target_seeds: Sequence[int] = STRICT_CARRION_SEEDS,
) -> dict[str, object]:
    branch_baseline = {
        str(result.get("branch_id", "")): str(result.get("baseline_action", "stay"))
        for result in branch_results
    }
    per_seed = []
    correct_total = 0
    predicted_total = 0
    missing_target_seeds = []
    for seed in target_seeds:
        heldout = [
            row for row in dataset_rows if _row_seed(row) == int(seed)
        ]
        train = [
            row for row in dataset_rows if _row_seed(row) != int(seed)
        ]
        if not heldout:
            missing_target_seeds.append(int(seed))
            per_seed.append(
                {
                    "seed": int(seed),
                    "train_row_count": len(train),
                    "heldout_row_count": 0,
                    "predicted_count": 0,
                    "correct_count": 0,
                    "accuracy": 0.0,
                    "missing": True,
                }
            )
            continue
        if not train:
            per_seed.append(
                {
                    "seed": int(seed),
                    "train_row_count": 0,
                    "heldout_row_count": len(heldout),
                    "predicted_count": 0,
                    "correct_count": 0,
                    "accuracy": 0.0,
                    "missing": False,
                    "blocked_reason": "no_training_rows_after_holdout",
                }
            )
            predicted_total += len(heldout)
            continue
        artifact = build_safe_archive_diagnostic_artifact(
            safe_archive_dataset_rows=train,
            candidate_id=f"{V155_CANDIDATE_ID}_loo_seed_{int(seed)}",
            support_mode=V155_SUPPORT_MODE,
            training_source_label=V155_TRAINING_SOURCE_LABEL,
        )
        selected_counts: Counter[str] = Counter()
        correct = 0
        sample_rows = []
        for index, row in enumerate(heldout):
            trainable = _mapping(row.get("trainable"))
            features = _mapping(trainable.get("features"))
            label = _label_action(row)
            branch_id = str(_mapping(row.get("metadata")).get("branch_id", ""))
            scored = score_support_gated_residual_artifact(
                artifact=artifact,
                observation_input=_mapping(features.get("observation_input")),
                action_mask=_mapping(features.get("action_mask")),
                public_history_trace=_list_of_mappings(
                    features.get("prior_public_context")
                ),
                linear_action=branch_baseline.get(branch_id, "stay"),
            )
            selected = str(scored.get("selected_action") or "")
            selected_counts.update([selected])
            if selected == label:
                correct += 1
            if index < 8:
                sample_rows.append(
                    {
                        "row_index": _int(_mapping(row.get("metadata")).get("row_index")),
                        "label_action": label,
                        "selected_action": selected,
                        "nearest_support_distance": scored.get(
                            "nearest_support_distance"
                        ),
                        "score_margin": scored.get("score_margin"),
                    }
                )
        predicted_total += len(heldout)
        correct_total += correct
        per_seed.append(
            {
                "seed": int(seed),
                "train_row_count": len(train),
                "heldout_row_count": len(heldout),
                "predicted_count": len(heldout),
                "correct_count": correct,
                "accuracy": _safe_rate(correct, len(heldout)),
                "selected_action_counts": dict(sorted(selected_counts.items())),
                "sample_predictions": sample_rows,
                "missing": False,
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_leave_one_seed_out_nn_v1",
        "all_target_seeds_evaluated": not missing_target_seeds,
        "missing_target_seeds": missing_target_seeds,
        "target_seeds": [int(seed) for seed in target_seeds],
        "row_count": len(dataset_rows),
        "predicted_count": predicted_total,
        "correct_count": correct_total,
        "accuracy": _safe_rate(correct_total, predicted_total),
        "per_seed": per_seed,
    }


def build_training_summary(
    *,
    artifact: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    branch_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    branch_baseline = {
        str(result.get("branch_id", "")): str(result.get("baseline_action", "stay"))
        for result in branch_results
    }
    rows = []
    correct = 0
    support_gate_passed_count = 0
    override_allowed_count = 0
    for index, row in enumerate(dataset_rows):
        trainable = _mapping(row.get("trainable"))
        features = _mapping(trainable.get("features"))
        label = _label_action(row)
        branch_id = str(_mapping(row.get("metadata")).get("branch_id", ""))
        scored = score_support_gated_residual_artifact(
            artifact=artifact,
            observation_input=_mapping(features.get("observation_input")),
            action_mask=_mapping(features.get("action_mask")),
            public_history_trace=_list_of_mappings(features.get("prior_public_context")),
            linear_action=branch_baseline.get(branch_id, "stay"),
        )
        selected = str(scored.get("selected_action") or "")
        if selected == label:
            correct += 1
        if scored.get("support_gate_passed") is True:
            support_gate_passed_count += 1
        if scored.get("override_allowed") is True:
            override_allowed_count += 1
        if index < 96:
            rows.append(
                {
                    "row_index": index,
                    "label_action": label,
                    "selected_action": selected,
                    "selected_action_correct": selected == label,
                    "support_gate_passed": scored.get("support_gate_passed"),
                    "override_allowed": scored.get("override_allowed"),
                    "nearest_support_distance": scored.get("nearest_support_distance"),
                    "score_margin": scored.get("score_margin"),
                }
            )
    row_count = len(dataset_rows)
    return {
        "policy": "m3_carrion_survivor_continuation_support_gated_training_summary_v1",
        "diagnostic_training_ran": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "row_count": row_count,
        "selected_action_accuracy_count": correct,
        "selected_action_accuracy": _safe_rate(correct, row_count),
        "support_gate_passed_count": support_gate_passed_count,
        "override_allowed_count": override_allowed_count,
        "accuracy_is_success_criterion": False,
        "rows": rows,
    }


def run_shadow_then_live_evaluation(
    *,
    artifact: Mapping[str, object],
    output_path: str | Path,
    run_live_ab: bool = True,
    broad_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    carrion_seeds: Sequence[int] = STRICT_CARRION_SEEDS,
    ticks: int = STRICT_LIVE_AB_TICKS,
) -> dict[str, object]:
    run_dir = Path(output_path).with_suffix("")
    baseline = build_linear_baseline_cache(
        broad_seeds=broad_seeds,
        ticks=ticks,
        fixture_seeds=carrion_seeds,
        fixture_ticks=ticks,
        keep_trajectories=False,
        output_dir=run_dir / "linear_control",
    )
    shadow = _run_runtime_mode_evaluation(
        artifact=artifact,
        runtime_mode="shadow",
        baseline=baseline,
        broad_seeds=broad_seeds,
        carrion_seeds=carrion_seeds,
        ticks=ticks,
    )
    shadow_gate = build_shadow_gate(shadow)
    if shadow_gate.get("passed") is True and run_live_ab:
        live = _run_runtime_mode_evaluation(
            artifact=artifact,
            runtime_mode="live",
            baseline=baseline,
            broad_seeds=broad_seeds,
            carrion_seeds=carrion_seeds,
            ticks=ticks,
        )
        live_gate = build_live_ab_gate(live)
    elif shadow_gate.get("passed") is not True:
        live = {"skipped": True, "reason": "shadow_gate_failed"}
        live_gate = _gate_not_run("live_ab_not_run_shadow_gate_failed")
    else:
        live = {"skipped": True, "reason": "run_live_ab_false"}
        live_gate = _gate_not_run("live_ab_not_run")
    return {
        "policy": "m3_carrion_survivor_continuation_shadow_then_live_eval_v1",
        "baseline": _baseline_summary(baseline),
        "shadow": shadow,
        "shadow_gate": shadow_gate,
        "live_ab": live,
        "live_ab_gate": live_gate,
    }


def build_shadow_gate(evaluation: Mapping[str, object]) -> dict[str, object]:
    metrics = _runtime_metrics(evaluation, use_gate_accepted=True)
    blockers = _common_runtime_blockers(metrics, prefix="shadow")
    return {
        "policy": "m3_carrion_survivor_continuation_shadow_gate_v1",
        "passed": not blockers,
        "blockers": blockers,
        "metrics": metrics,
    }


def build_live_ab_gate(evaluation: Mapping[str, object]) -> dict[str, object]:
    metrics = _runtime_metrics(evaluation, use_gate_accepted=False)
    blockers = _common_runtime_blockers(metrics, prefix="live")
    broad = _mapping(evaluation.get("broad"))
    carrion = _mapping(evaluation.get("carrion_only"))
    for row in _list_of_mappings(broad.get("per_seed_delta")):
        seed = _int(row.get("seed"))
        if _int(row.get("alive_delta")) < 0:
            blockers.append(
                _blocker(
                    "broad_seed_alive_regression",
                    observed={"seed": seed, "delta": row.get("alive_delta")},
                    required=">= 0",
                )
            )
        if _int(row.get("births_delta")) < 0:
            blockers.append(
                _blocker(
                    "broad_seed_birth_regression",
                    observed={"seed": seed, "delta": row.get("births_delta")},
                    required=">= 0",
                )
            )
    carrion_delta = _mapping(carrion.get("aggregate_delta"))
    baseline_gate = _mapping(carrion.get("baseline_fixture_gate"))
    candidate_gate = _mapping(carrion.get("candidate_fixture_gate"))
    baseline_blockers = len(_list_of_mappings(baseline_gate.get("blockers")))
    candidate_blockers = len(_list_of_mappings(candidate_gate.get("blockers")))
    terminal_alive_improved = _float(carrion_delta.get("alive_agents_mean")) > 0.0
    blocker_reduced = candidate_blockers < baseline_blockers
    if not terminal_alive_improved and not blocker_reduced:
        blockers.append(
            _blocker(
                "carrion_no_blocker_reduction_or_terminal_alive_improvement",
                observed={
                    "carrion_alive_delta": carrion_delta.get("alive_agents_mean"),
                    "baseline_blocker_count": baseline_blockers,
                    "candidate_blocker_count": candidate_blockers,
                },
                required="alive_delta > 0 or candidate_blockers < baseline_blockers",
            )
        )
    return {
        "policy": "m3_carrion_survivor_continuation_live_ab_strict_gate_v1",
        "passed": not blockers,
        "blockers": blockers,
        "metrics": {
            **metrics,
            "broad_per_seed_delta": broad.get("per_seed_delta"),
            "carrion_per_seed_delta": carrion.get("per_seed_delta"),
            "carrion_alive_delta": carrion_delta.get("alive_agents_mean"),
            "carrion_births_delta": carrion_delta.get("births_mean"),
            "carrion_baseline_blocker_count": baseline_blockers,
            "carrion_candidate_blocker_count": candidate_blockers,
            "carrion_blocker_reduced": blocker_reduced,
            "carrion_terminal_alive_improved": terminal_alive_improved,
        },
    }


def build_top_level_acceptance(
    *,
    validation: Mapping[str, object],
    pretraining_review: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    if validation.get("passed") is not True:
        blockers.append(
            _blocker("source_validation_failed", observed=validation.get("failures"), required=[])
        )
    if pretraining_review.get("passed") is not True:
        blockers.append(
            _blocker(
                "pretraining_review_failed",
                observed=pretraining_review.get("blockers"),
                required=[],
            )
        )
    shadow_gate = _mapping(evaluation.get("shadow_gate"))
    live_gate = _mapping(evaluation.get("live_ab_gate"))
    if shadow_gate.get("passed") is not True:
        blockers.append(
            _blocker("shadow_gate_failed", observed=shadow_gate.get("blockers"), required=[])
        )
    if live_gate.get("passed") is not True:
        blockers.append(
            _blocker("live_ab_gate_failed", observed=live_gate.get("blockers"), required=[])
        )
    first = blockers[0] if blockers else {}
    return {
        "policy": "m3_carrion_survivor_continuation_train_eval_acceptance_v1",
        "passed": not blockers,
        "blockers": blockers,
        "first_blocker": first.get("reason"),
        "metrics": {
            "shadow": _mapping(shadow_gate.get("metrics")),
            "live_ab": _mapping(live_gate.get("metrics")),
        },
    }


def _run_runtime_mode_evaluation(
    *,
    artifact: Mapping[str, object],
    runtime_mode: str,
    baseline: Mapping[str, object],
    broad_seeds: Sequence[int],
    carrion_seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    broad_runs = [
        _run_policy(
            seed=seed,
            ticks=ticks,
            artifact=artifact,
            runtime_mode=runtime_mode,
            fixture=None,
            trajectory_path=None,
        )
        for seed in broad_seeds
    ]
    carrion_runs = [
        _run_policy(
            seed=seed,
            ticks=ticks,
            artifact=artifact,
            runtime_mode=runtime_mode,
            fixture="carrion_only",
            trajectory_path=None,
        )
        for seed in carrion_seeds
    ]
    return {
        "runtime_mode": runtime_mode,
        "broad": _comparison_against_baseline(
            fixture="broad",
            baseline_runs=_list_of_mappings(_mapping(baseline.get("broad")).get("runs")),
            candidate_runs=broad_runs,
        ),
        "carrion_only": _comparison_against_baseline(
            fixture="carrion_only",
            baseline_runs=_list_of_mappings(
                _mapping(baseline.get("carrion_only")).get("runs")
            ),
            candidate_runs=carrion_runs,
        ),
    }


def _runtime_metrics(
    evaluation: Mapping[str, object],
    *,
    use_gate_accepted: bool,
) -> dict[str, object]:
    broad_candidate = _mapping(
        _mapping(_mapping(evaluation.get("broad")).get("candidate")).get("aggregate")
    )
    carrion_candidate = _mapping(
        _mapping(_mapping(evaluation.get("carrion_only")).get("candidate")).get(
            "aggregate"
        )
    )
    broad_diag = _mapping(broad_candidate.get("support_residual_diagnostics"))
    carrion_diag = _mapping(carrion_candidate.get("support_residual_diagnostics"))
    requested_counts = _sum_counter(
        broad_candidate.get("requested_action_counts"),
        carrion_candidate.get("requested_action_counts"),
    )
    diagnostic_action_key = (
        "gate_accepted_override_action_counts"
        if use_gate_accepted
        else "applied_override_action_counts"
    )
    diagnostic_counts = _sum_counter(
        broad_diag.get(diagnostic_action_key),
        carrion_diag.get(diagnostic_action_key),
    )
    requested_dominant = _dominant_count_share(requested_counts)
    diagnostic_dominant = _dominant_count_share(diagnostic_counts)
    dominant_requested_share = max(
        _float(broad_candidate.get("dominant_requested_action_share")),
        _float(carrion_candidate.get("dominant_requested_action_share")),
        _float(requested_dominant.get("share")),
    )
    return {
        "heuristic_action_source_count": _int(
            broad_candidate.get("heuristic_action_source_count")
        )
        + _int(carrion_candidate.get("heuristic_action_source_count")),
        "unsupported_requested_action_count": _int(
            broad_candidate.get("unsupported_requested_action_count")
        )
        + _int(carrion_candidate.get("unsupported_requested_action_count")),
        "unsupported_proposed_action_count": _int(
            broad_diag.get("unsupported_proposed_action_count")
        )
        + _int(carrion_diag.get("unsupported_proposed_action_count")),
        "dominant_requested_action": requested_dominant.get("key"),
        "dominant_requested_action_share": _round(dominant_requested_share),
        "requested_action_counts": dict(sorted(requested_counts.items())),
        "diagnostic_action_counts": dict(sorted(diagnostic_counts.items())),
        "dominant_diagnostic_action": diagnostic_dominant.get("key"),
        "dominant_diagnostic_action_share": diagnostic_dominant.get("share"),
        "broad_support_residual_diagnostics": broad_diag,
        "carrion_support_residual_diagnostics": carrion_diag,
    }


def _common_runtime_blockers(
    metrics: Mapping[str, object],
    *,
    prefix: str,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if _int(metrics.get("heuristic_action_source_count")) != 0:
        blockers.append(
            _blocker(
                f"{prefix}_heuristic_action_source_count_nonzero",
                observed=metrics.get("heuristic_action_source_count"),
                required=0,
            )
        )
    if _int(metrics.get("unsupported_requested_action_count")) != 0:
        blockers.append(
            _blocker(
                f"{prefix}_unsupported_requested_action_count_nonzero",
                observed=metrics.get("unsupported_requested_action_count"),
                required=0,
            )
        )
    if _int(metrics.get("unsupported_proposed_action_count")) != 0:
        blockers.append(
            _blocker(
                f"{prefix}_unsupported_proposed_action_count_nonzero",
                observed=metrics.get("unsupported_proposed_action_count"),
                required=0,
            )
        )
    if _float(metrics.get("dominant_requested_action_share")) > 0.50:
        blockers.append(
            _blocker(
                f"{prefix}_dominant_requested_action_share_gt_0_50",
                observed=metrics.get("dominant_requested_action_share"),
                required="<= 0.50",
            )
        )
    return blockers


def _leave_one_seed_out_prior_baseline(
    dataset_rows: Sequence[Mapping[str, object]],
    *,
    target_seeds: Sequence[int],
    grouping_key,
    policy: str,
) -> dict[str, object]:
    per_seed = []
    correct_total = 0
    predicted_total = 0
    missing_target_seeds = []
    for seed in target_seeds:
        heldout = [row for row in dataset_rows if _row_seed(row) == int(seed)]
        train = [row for row in dataset_rows if _row_seed(row) != int(seed)]
        if not heldout:
            missing_target_seeds.append(int(seed))
            per_seed.append(
                {
                    "seed": int(seed),
                    "heldout_row_count": 0,
                    "correct_count": 0,
                    "accuracy": 0.0,
                    "missing": True,
                }
            )
            continue
        global_counts = Counter(_label_action(row) for row in train)
        fallback = _majority_action(global_counts)
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        if grouping_key is not None:
            for row in train:
                grouped[str(grouping_key(row))].update([_label_action(row)])
        correct = 0
        predictions: Counter[str] = Counter()
        for row in heldout:
            if grouping_key is None:
                predicted = fallback
            else:
                predicted = _majority_action(grouped.get(str(grouping_key(row)), Counter()))
                if not predicted:
                    predicted = fallback
            predictions.update([predicted])
            if predicted == _label_action(row):
                correct += 1
        correct_total += correct
        predicted_total += len(heldout)
        per_seed.append(
            {
                "seed": int(seed),
                "train_row_count": len(train),
                "heldout_row_count": len(heldout),
                "correct_count": correct,
                "accuracy": _safe_rate(correct, len(heldout)),
                "prediction_counts": dict(sorted(predictions.items())),
                "missing": False,
            }
        )
    return {
        "policy": policy,
        "target_seeds": [int(seed) for seed in target_seeds],
        "missing_target_seeds": missing_target_seeds,
        "all_target_seeds_evaluated": not missing_target_seeds,
        "row_count": len(dataset_rows),
        "predicted_count": predicted_total,
        "correct_count": correct_total,
        "accuracy": _safe_rate(correct_total, predicted_total),
        "per_seed": per_seed,
    }


def _conflicting_groups(
    groups: Mapping[str, Sequence[tuple[int, Mapping[str, object]]]],
) -> list[dict[str, object]]:
    conflicts = []
    for digest, items in sorted(groups.items()):
        counts = _label_counts(items)
        if len(counts) <= 1:
            continue
        seeds = sorted({_row_seed(row) for _, row in items})
        branch_ids = sorted(
            {
                str(_mapping(row.get("metadata")).get("branch_id", ""))
                for _, row in items
            }
        )
        conflicts.append(
            {
                "digest": digest,
                "row_count": len(items),
                "label_action_counts": dict(sorted(counts.items())),
                "seeds": seeds,
                "branch_ids_sample": branch_ids[:8],
                "row_indexes_sample": [
                    _int(_mapping(row.get("metadata")).get("row_index"))
                    for _, row in items[:12]
                ],
            }
        )
    return conflicts


def _label_counts(
    items: Sequence[tuple[int, Mapping[str, object]]],
) -> Counter[str]:
    return Counter(_label_action(row) for _, row in items)


def _row_mask_digest(row: Mapping[str, object]) -> str:
    trainable = _mapping(row.get("trainable"))
    features = _mapping(trainable.get("features"))
    return stable_payload_digest(_mapping(features.get("action_mask")))


def _row_seed(row: Mapping[str, object]) -> int:
    return _int(_mapping(row.get("metadata")).get("seed"))


def _label_action(row: Mapping[str, object]) -> str:
    return str(_mapping(_mapping(row.get("trainable")).get("label")).get("action", ""))


def _majority_action(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return sorted(
        counts.items(),
        key=lambda item: (int(item[1]), _action_sort_key(str(item[0]))),
        reverse=True,
    )[0][0]


def _action_sort_key(action: str) -> int:
    try:
        return len(ACTION_NAMES) - ACTION_NAMES.index(action)
    except ValueError:
        return 0


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return _round(float(numerator) / float(denominator))


def _sum_counter(*values: object) -> Counter[str]:
    counter: Counter[str] = Counter()
    for value in values:
        for key, count in _mapping(value).items():
            counter.update({str(key): _int(count)})
    return counter


def _blocker(reason: str, *, observed: object, required: object) -> dict[str, object]:
    return {"reason": reason, "observed": observed, "required": required}


def _skipped_pretraining_review(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_pretraining_review_v1",
        "passed": False,
        "skipped": True,
        "reason": reason,
        "blockers": [_blocker(reason, observed="skipped", required="source validation")],
        "fail_closed": True,
    }


def _artifact_status(path: Path, *, created: bool, reason: str) -> dict[str, object]:
    return {
        "created": bool(created),
        "path": str(path),
        "skipped": not created,
        "reason": reason,
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
    }


def _skipped_training(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_support_gated_training_summary_v1",
        "diagnostic_training_ran": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "skipped": True,
        "reason": reason,
    }


def _skipped_evaluation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_shadow_then_live_eval_v1",
        "skipped": True,
        "reason": reason,
        "shadow": {"skipped": True, "reason": reason},
        "shadow_gate": _gate_not_run("shadow_not_run"),
        "live_ab": {"skipped": True, "reason": reason},
        "live_ab_gate": _gate_not_run("live_ab_not_run"),
    }


def _gate_not_run(reason: str) -> dict[str, object]:
    return {
        "passed": False,
        "skipped": True,
        "reason": reason,
        "blockers": [_blocker(reason, observed="skipped", required="gate run")],
        "metrics": {},
    }


def _closed_acceptance(
    *,
    reason: str,
    validation: Mapping[str, object],
    pretraining_review: Mapping[str, object],
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_train_eval_acceptance_v1",
        "passed": False,
        "blockers": [
            {
                "reason": reason,
                "observed": {
                    "source_validation_passed": validation.get("passed"),
                    "pretraining_review_passed": pretraining_review.get("passed"),
                },
                "required": "source validation and pretraining review pass",
            }
        ],
        "first_blocker": reason,
        "metrics": {},
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    pretraining_review: Mapping[str, object],
    artifact: Mapping[str, object],
    evaluation: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True:
        primary = (
            "m3_carrion_survivor_continuation_train_eval_"
            "source_invalid_closed_no_training"
        )
    elif pretraining_review.get("passed") is not True:
        primary = (
            "m3_carrion_survivor_continuation_train_eval_"
            "pretraining_alias_prior_blocked_closed_no_training"
        )
    elif artifact.get("created") is not True:
        primary = (
            "m3_carrion_survivor_continuation_train_eval_"
            "artifact_not_created_closed_no_training"
        )
    elif _mapping(evaluation.get("shadow_gate")).get("passed") is not True:
        primary = (
            "m3_carrion_survivor_continuation_train_eval_"
            "shadow_failed_no_live_ab"
        )
    elif _mapping(evaluation.get("live_ab_gate")).get("passed") is not True:
        primary = (
            "m3_carrion_survivor_continuation_train_eval_"
            "live_ab_failed_non_promotional"
        )
    elif acceptance.get("passed") is True:
        primary = (
            "m3_carrion_survivor_continuation_train_eval_"
            "strict_live_ab_passed_non_promotional"
        )
    else:
        primary = (
            "m3_carrion_survivor_continuation_train_eval_"
            "closed_non_promotional"
        )
    return {"primary": primary, "labels": [primary]}


def nearest_neighbor_feature_vector_for_row(
    row: Mapping[str, object],
    action: str,
) -> tuple[float, ...]:
    trainable = _mapping(row.get("trainable"))
    features = _mapping(trainable.get("features"))
    runtime_row = planner_distilled_runtime_row(
        observation_input=_mapping(features.get("observation_input")),
        action_mask=_mapping(features.get("action_mask")),
        public_history_trace=_list_of_mappings(features.get("prior_public_context")),
    )
    return candidate_feature_vector(runtime_row, action)
