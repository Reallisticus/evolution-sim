from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from evolution_sim.cli import mind_v3_evaluate as evaluate_cli
from evolution_sim.cli import mind_v3_transition_value_live_ab as v142_live_ab
from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.broad_regression_branch_intervention import (
    MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION,
    MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION,
    _dataset_leakage_scan as v143_dataset_leakage_scan,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.support_gated_residual import (
    MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
    MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
    V103_ACTION_PRIOR_BALANCE_PENALTY,
    _action_option_mode,
    aggregate_support_gated_residual_runtime_diagnostics,
    load_support_gated_residual_artifact,
    score_support_gated_residual_artifact,
    support_gated_residual_runtime_diagnostics,
    validate_support_gated_residual_artifact,
)
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_REPORT_SCHEMA_VERSION = (
    "mind_v3_v144_branch_intervention_residual_live_ab_v1"
)
MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_TRAINING_POLICY = (
    "v144_branch_intervention_public_observation_mask_label_residual_training_v1"
)
MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_SHADOW_POLICY = (
    "v144_branch_intervention_residual_strict_shadow_gate_v1"
)
MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_ACCEPTANCE_POLICY = (
    "v144_branch_intervention_residual_strict_live_ab_acceptance_v1"
)
DEFAULT_V143_REPORT_PATH = Path(
    "output/mind/mind-v3-v143-broad-regression-branch-intervention-report.json"
)
DEFAULT_V143_DATASET_PATH = Path(
    "output/mind/mind-v3-v143-broad-regression-branch-intervention-dataset.jsonl"
)
DEFAULT_ARTIFACT_PATH = Path(
    "output/mind/mind-v3-v144-branch-intervention-residual-artifact.json"
)
DEFAULT_REPORT_PATH = Path(
    "output/mind/mind-v3-v144-branch-intervention-residual-live-ab.json"
)
STRICT_BROAD_SEEDS = v142_live_ab.STRICT_BROAD_SEEDS
STRICT_CARRION_FIXTURE_SEEDS = v142_live_ab.STRICT_CARRION_FIXTURE_SEEDS
STRICT_TICKS = v142_live_ab.STRICT_TICKS
REQUIRED_V143_REGRESSION_SEEDS = (5, 13, 19, 29, 37)
V144_MAX_DOMINANT_ACTION_SHARE = 0.50


class BranchInterventionResidualError(ValueError):
    pass


def load_json_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BranchInterventionResidualError(
            f"failed to read JSON report: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BranchInterventionResidualError(
            f"invalid JSON report {resolved}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BranchInterventionResidualError(
            f"JSON report must be an object: {resolved}"
        )
    return payload


def load_v143_branch_intervention_dataset(
    path: str | Path,
) -> list[dict[str, object]]:
    resolved = Path(path)
    rows: list[dict[str, object]] = []
    try:
        with _open_input(resolved) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BranchInterventionResidualError(
                        f"invalid JSONL row {line_number} in {resolved}: {exc.msg}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise BranchInterventionResidualError(
                        f"JSONL row {line_number} in {resolved} must be an object"
                    )
                rows.append(payload)
    except OSError as exc:
        raise BranchInterventionResidualError(
            f"failed to read v143 dataset: {resolved}"
        ) from exc
    return rows


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_branch_intervention_residual_artifact(
    *,
    v143_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    v143_report_path: str | Path | None = None,
    v143_dataset_path: str | Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    source_integrity = validate_v143_branch_intervention_source(
        v143_report=v143_report,
        dataset_rows=dataset_rows,
        v143_report_path=v143_report_path,
        v143_dataset_path=v143_dataset_path,
    )
    if source_integrity["passed"] is not True:
        raise BranchInterventionResidualError(
            "v143 branch-intervention source integrity failed: "
            f"{source_integrity['failures']}"
        )
    support_examples = _support_examples_from_dataset(dataset_rows)
    action_counts = Counter(str(example["action"]) for example in support_examples)
    contract = {
        "one_row_one_agent_local_decision": True,
        "requires_action_mask": True,
        "requires_policy_visible_features_only": True,
        "requires_linear_default_action": True,
        "requires_planner_outcome_tables": False,
        "requires_global_batch_assignment": False,
        "uses_heuristic_fallback": False,
        "uses_seed_id_as_runtime_feature": False,
        "uses_branch_id_as_runtime_feature": False,
        "uses_fixture_id_as_runtime_feature": False,
        "uses_logged_action_as_runtime_fallback": False,
        "uses_private_simulator_state": False,
    }
    artifact = {
        "schema_version": (
            MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION
        ),
        "policy": MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
        "default_action_policy": "linear_mind_v3",
        "runtime_ready": True,
        "promotion_ready": False,
        "runtime_promotion_allowed": False,
        "training_policy": MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_TRAINING_POLICY,
        "v143_report_digest": stable_payload_digest(v143_report),
        "v143_dataset_digest": stable_payload_digest(list(dataset_rows)),
        "scorer_rule": "action_prior_balanced_nearest_support_v1",
        "action_prior_penalty_scale": V103_ACTION_PRIOR_BALANCE_PENALTY,
        "support_gate": {
            "policy": "v144_exact_public_observation_mask_support_gate_v1",
            "threshold_source": (
                "exact_match_to_v143_public_observation_mask_label_rows_v1"
            ),
            "legal_action_required": True,
            "action_support_required": True,
            "distance_threshold_inclusive": True,
            "margin_threshold_inclusive": True,
            "nearest_support_distance_threshold": 0.0,
            "residual_score_margin_threshold": 0.0,
        },
        "inference_contract": contract,
        "training_row_count": len(support_examples),
        "support_action_counts": dict(sorted(action_counts.items())),
        "teacher_action_counts": dict(sorted(action_counts.items())),
        "support_examples": support_examples,
        "threshold_diagnostics": {
            "policy": "v144_exact_support_thresholds_v1",
            "nearest_support_distance_threshold": 0.0,
            "residual_score_margin_threshold": 0.0,
            "calibration_row_count": len(support_examples),
            "calibration_action_counts": dict(sorted(action_counts.items())),
            "threshold_tuning_after_live_failure": False,
        },
    }
    validate_support_gated_residual_artifact(artifact)
    roundtrip = artifact_roundtrip_score_check(
        artifact=artifact,
        dataset_rows=dataset_rows,
    )
    training = {
        "policy": MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_TRAINING_POLICY,
        "source_integrity": source_integrity,
        "artifact_roundtrip": roundtrip,
        "training_row_count": len(dataset_rows),
        "support_example_count": len(support_examples),
        "support_action_counts": dict(sorted(action_counts.items())),
        "dominant_support_action": _dominant_count_share(action_counts)["key"],
        "dominant_support_action_share": _dominant_count_share(action_counts)[
            "share"
        ],
        "uses_public_observation_input": True,
        "uses_public_action_mask": True,
        "uses_label_action": True,
        "uses_non_trainable_metadata_as_features": False,
        "runtime_promotion_allowed": False,
    }
    return artifact, training


def validate_v143_branch_intervention_source(
    *,
    v143_report: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
    v143_report_path: str | Path | None = None,
    v143_dataset_path: str | Path | None = None,
) -> dict[str, object]:
    failures: list[str] = []
    row_failures: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    support_seed_values: set[int] = set()
    if v143_report.get("schema_version") != MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION:
        failures.append("v143_report_schema_mismatch")
    classification = _mapping(v143_report.get("classification"))
    if classification.get("primary") != (
        "broad_regression_branch_intervention_supported_for_v144_training"
    ):
        failures.append("v143_classification_not_supported_for_v144")
    if _mapping(v143_report.get("source_precheck")).get("passed") is not True:
        failures.append("v143_source_precheck_not_passed")
    if _mapping(v143_report.get("source_integrity")).get("passed") is not True:
        failures.append("v143_source_integrity_not_passed")
    acceptance = _mapping(v143_report.get("acceptance"))
    if acceptance.get("passed") is not True:
        failures.append("v143_acceptance_not_passed")
    if _float(acceptance.get("dominant_label_action_share")) > V144_MAX_DOMINANT_ACTION_SHARE:
        failures.append("v143_label_dominant_share_above_cap")

    report_dataset = _mapping(v143_report.get("dataset"))
    computed_digest = stable_payload_digest(list(dataset_rows))
    if report_dataset.get("dataset_digest") != computed_digest:
        failures.append("v143_dataset_digest_mismatch")
    if _int(report_dataset.get("row_count")) != len(dataset_rows):
        failures.append("v143_dataset_row_count_mismatch")
    leakage_scan = v143_dataset_leakage_scan(dataset_rows)
    if leakage_scan.get("passed") is not True:
        failures.append("v143_dataset_leakage_scan_failed")

    for index, row in enumerate(dataset_rows):
        row_failure_reasons = _dataset_row_failure_reasons(row)
        trainable = _mapping(row.get("trainable"))
        label = _mapping(trainable.get("label"))
        action = str(label.get("action", ""))
        if action in ACTION_NAMES:
            label_counts.update([action])
        metadata = _mapping(row.get("metadata"))
        seed_value = _optional_int(metadata.get("seed"))
        if seed_value is not None:
            support_seed_values.add(seed_value)
        if row_failure_reasons:
            row_failures.append(
                {
                    "row_index": index,
                    "failures": row_failure_reasons,
                    "label_action": action,
                    "metadata_seed": seed_value,
                }
            )
    if row_failures:
        failures.append("v143_dataset_row_contract_failed")

    required = set(REQUIRED_V143_REGRESSION_SEEDS)
    support_by_seed = {
        str(seed): bool(seed in support_seed_values)
        for seed in REQUIRED_V143_REGRESSION_SEEDS
    }
    missing_support = sorted(required - support_seed_values)
    if missing_support:
        failures.append("v143_regression_seed_support_missing")
    reported_support = _mapping(acceptance.get("support_by_seed"))
    for seed in REQUIRED_V143_REGRESSION_SEEDS:
        if reported_support.get(str(seed)) is not True:
            failures.append("v143_acceptance_support_by_seed_mismatch")
            break

    dominant = _dominant_count_share(label_counts)
    if dominant["share"] > V144_MAX_DOMINANT_ACTION_SHARE:
        failures.append("v143_dataset_label_dominant_share_above_cap")
    seed_41_status = _seed_41_non_regression_status(
        v143_report=v143_report,
        v143_report_path=v143_report_path,
    )
    if seed_41_status["passed"] is not True:
        failures.append("v143_seed_41_non_regression_not_confirmed")

    return {
        "policy": "v144_v143_branch_intervention_source_integrity_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v143_report_path": str(v143_report_path) if v143_report_path else None,
        "v143_dataset_path": str(v143_dataset_path) if v143_dataset_path else None,
        "v143_report_file_sha256": (
            _file_sha256(Path(v143_report_path)) if v143_report_path else None
        ),
        "v143_dataset_file_sha256": (
            _file_sha256(Path(v143_dataset_path)) if v143_dataset_path else None
        ),
        "v143_dataset_digest": computed_digest,
        "v143_report_dataset_digest": report_dataset.get("dataset_digest"),
        "dataset_row_count": len(dataset_rows),
        "dataset_leakage_scan": leakage_scan,
        "row_failure_count": len(row_failures),
        "row_failures": row_failures[:16],
        "label_action_counts": dict(sorted(label_counts.items())),
        "dominant_label_action": dominant["key"],
        "dominant_label_action_count": dominant["count"],
        "dominant_label_action_share": dominant["share"],
        "required_regression_seeds": list(REQUIRED_V143_REGRESSION_SEEDS),
        "support_by_regression_seed": support_by_seed,
        "missing_regression_seed_support": missing_support,
        "seed_41_non_regression": seed_41_status,
    }


def artifact_roundtrip_score_check(
    *,
    artifact: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    serialized = json.dumps(dict(artifact), sort_keys=True, allow_nan=False)
    loaded_payload = json.loads(serialized)
    loaded = load_support_gated_residual_artifact(loaded_payload)
    pre_scores = _score_dataset_rows(artifact, dataset_rows)
    post_scores = _score_dataset_rows(loaded, dataset_rows)
    mismatches = []
    for index, (pre, post) in enumerate(zip(pre_scores, post_scores)):
        if pre != post:
            mismatches.append({"row_index": index, "pre": pre, "post": post})
    label_match_count = sum(1 for score in pre_scores if score.get("label_match"))
    return {
        "policy": "v144_artifact_json_roundtrip_score_equivalence_v1",
        "loaded_artifact_scores_match_pre_serialization": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:8],
        "score_row_count": len(pre_scores),
        "label_match_count": int(label_match_count),
        "all_training_rows_select_label_action": label_match_count == len(pre_scores),
        "serialized_json_length": len(serialized),
    }


def build_branch_intervention_residual_live_ab_report(
    *,
    artifact: Mapping[str, object],
    training_report: Mapping[str, object],
    broad_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    ticks: int = STRICT_TICKS,
    fixture_seeds: Sequence[int] = STRICT_CARRION_FIXTURE_SEEDS,
    fixture_ticks: int = STRICT_TICKS,
    artifact_output: str | Path | None = DEFAULT_ARTIFACT_PATH,
) -> dict[str, object]:
    validate_support_gated_residual_artifact(artifact)
    broad_seed_values = tuple(int(seed) for seed in broad_seeds)
    fixture_seed_values = tuple(int(seed) for seed in fixture_seeds)
    shadow = build_shadow_strict_broad_report(
        runtime_artifact=artifact,
        seeds=broad_seed_values,
        ticks=int(ticks),
    )
    broad = None
    carrion = None
    if shadow["shadow_gate"]["passed"] is True:
        broad = build_live_broad_ab_report(
            runtime_artifact=artifact,
            seeds=broad_seed_values,
            ticks=int(ticks),
        )
        carrion = build_live_carrion_fixture_ab_report(
            runtime_artifact=artifact,
            seeds=fixture_seed_values,
            ticks=int(fixture_ticks),
        )
    acceptance = _acceptance(
        training_report=training_report,
        shadow=shadow,
        broad=broad,
        carrion=carrion,
        broad_seeds=broad_seed_values,
        ticks=int(ticks),
        fixture_seeds=fixture_seed_values,
        fixture_ticks=int(fixture_ticks),
    )
    primary = (
        "branch_intervention_residual_live_ab_passed_strict_matrix"
        if acceptance["passed"]
        else "branch_intervention_residual_blocked_non_promotable"
    )
    return {
        "schema_version": (
            MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_REPORT_SCHEMA_VERSION
        ),
        "policy": MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
        "training_policy": MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_TRAINING_POLICY,
        "contract": {
            "default_runtime_behavior_changed": False,
            "explicit_opt_in_required": True,
            "runtime_action_selection_changed_when_enabled": True,
            "trainer_effect": "none",
            "gate_effect": "none",
            "viewer_effect": "none",
            "replay_golden_effect": "none",
            "trajectory_schema_changed": False,
            "runtime_promotion_authorized": False,
            "training_promotion_authorized": False,
            "heuristic_fallback_added": False,
            "threshold_tuning_after_live_failure": False,
        },
        "artifact_output": str(artifact_output) if artifact_output else None,
        "artifact_digest": stable_payload_digest(artifact),
        "artifact_schema_version": artifact.get("schema_version"),
        "training": dict(training_report),
        "matrix": {
            "broad_seeds": list(broad_seed_values),
            "ticks": int(ticks),
            "fixture": "carrion_only",
            "fixture_seeds": list(fixture_seed_values),
            "fixture_ticks": int(fixture_ticks),
        },
        "shadow_strict_broad": shadow,
        "live_strict_broad": broad,
        "carrion_only": carrion,
        "acceptance": acceptance,
        "classification": {"primary": primary, "labels": [primary]},
        "non_default_runtime": True,
        "non_promoted": True,
    }


def build_shadow_strict_broad_report(
    *,
    runtime_artifact: Mapping[str, object],
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    runs = [
        _run_broad_policy(
            seed=seed,
            ticks=ticks,
            runtime_artifact=runtime_artifact,
            runtime_mode="shadow",
        )
        for seed in seeds
    ]
    aggregate = _aggregate_runs(runs)
    gate = _shadow_gate(
        aggregate=aggregate,
        diagnostics=_mapping(aggregate.get("support_residual_diagnostics")),
    )
    return {
        "schema_version": "mind_v3_v144_branch_intervention_residual_shadow_v1",
        "policy": MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_SHADOW_POLICY,
        "runtime_policy": MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
        "runtime_mode": "shadow",
        "strict_live_promotion_executed": False,
        "runtime_promotion_allowed": False,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "runs": runs,
        "aggregate": aggregate,
        "shadow_gate": gate,
        "passed": bool(gate["passed"]),
    }


def build_live_broad_ab_report(
    *,
    runtime_artifact: Mapping[str, object],
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    linear_runs = [
        _run_broad_policy(
            seed=seed,
            ticks=ticks,
            runtime_artifact=None,
        )
        for seed in seeds
    ]
    residual_runs = [
        _run_broad_policy(
            seed=seed,
            ticks=ticks,
            runtime_artifact=runtime_artifact,
            runtime_mode="live",
        )
        for seed in seeds
    ]
    return {
        "schema_version": "mind_v3_v144_branch_intervention_residual_broad_ab_v1",
        "fixture": "broad",
        "runtime_mode": "live",
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        **_comparison_section(
            linear_runs=linear_runs,
            residual_runs=residual_runs,
        ),
    }


def build_live_carrion_fixture_ab_report(
    *,
    runtime_artifact: Mapping[str, object],
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    fixture_config = evaluate_cli.mind_v3_fixture_gate_config(
        suite="basic",
        seeds=[int(seed) for seed in seeds],
        ticks=int(ticks),
        min_alive=1.0,
        min_births=0.0,
        min_mixed_stable_births=0.0,
        min_energy_viability=0.0,
        min_hydration_viability=0.0,
        min_health_viability=0.0,
        min_matched_diet_viability=0.0,
        min_biologically_ready=0.0,
    )
    linear_suite = _run_carrion_fixture_suite(
        seeds=seeds,
        ticks=ticks,
        runtime_artifact=None,
        runtime_mode="live",
        policy_name="linear_mind_v3",
    )
    residual_suite = _run_carrion_fixture_suite(
        seeds=seeds,
        ticks=ticks,
        runtime_artifact=runtime_artifact,
        runtime_mode="live",
        policy_name="v144_branch_intervention_residual",
    )
    linear_gate = evaluate_cli.mind_v3_fixture_gate_status(
        fixture_suite=linear_suite,
        fixture_config=fixture_config,
    )
    residual_gate = evaluate_cli.mind_v3_fixture_gate_status(
        fixture_suite=residual_suite,
        fixture_config=fixture_config,
    )
    linear_runs = _fixture_runs(linear_suite)
    residual_runs = _fixture_runs(residual_suite)
    return {
        "schema_version": "mind_v3_v144_branch_intervention_residual_carrion_ab_v1",
        "fixture": "carrion_only",
        "runtime_mode": "live",
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "fixture_config": fixture_config,
        "linear_fixture_suite": linear_suite,
        "residual_fixture_suite": residual_suite,
        "linear_fixture_gate": linear_gate,
        "residual_fixture_gate": residual_gate,
        **_comparison_section(
            linear_runs=linear_runs,
            residual_runs=residual_runs,
        ),
    }


def _support_examples_from_dataset(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    examples = []
    for index, row in enumerate(rows):
        trainable = _mapping(row.get("trainable"))
        features = _mapping(trainable.get("features"))
        label = _mapping(trainable.get("label"))
        action = str(label.get("action", ""))
        observation_input = _mapping(features.get("observation_input"))
        action_mask = _mapping(features.get("action_mask"))
        runtime_row = planner_distilled_runtime_row(
            observation_input=observation_input,
            action_mask=action_mask,
            public_history_trace=[],
        )
        vector = candidate_feature_vector(runtime_row, action)
        if action not in ACTION_NAMES or not vector:
            raise BranchInterventionResidualError(
                f"row {index} cannot produce a support vector for action {action!r}"
            )
        examples.append(
            {
                "example_index": index,
                "action": action,
                "mode": _action_option_mode(action),
                "feature_vector": [_round(value) for value in vector],
                "weight": 1.0,
            }
        )
    if not examples:
        raise BranchInterventionResidualError("v144 training dataset is empty")
    return examples


def _dataset_row_failure_reasons(row: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if row.get("schema_version") != (
        MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION
    ):
        failures.append("schema_version_mismatch")
    trainable = _mapping(row.get("trainable"))
    features = _mapping(trainable.get("features"))
    label = _mapping(trainable.get("label"))
    observation_input = _mapping(features.get("observation_input"))
    action_mask = _mapping(features.get("action_mask"))
    action = str(label.get("action", ""))
    if not observation_input:
        failures.append("missing_observation_input")
    if not action_mask:
        failures.append("missing_action_mask")
    if action not in ACTION_NAMES:
        failures.append("label_action_unsupported")
    elif action_mask.get(action) is not True:
        failures.append("label_action_not_legal_under_mask")
    if trainable.get("feature_policy") != (
        "public_observation_input_and_public_action_mask_v1"
    ):
        failures.append("feature_policy_mismatch")
    if observation_input and action_mask and action in ACTION_NAMES:
        runtime_row = planner_distilled_runtime_row(
            observation_input=observation_input,
            action_mask=action_mask,
            public_history_trace=[],
        )
        if not candidate_feature_vector(runtime_row, action):
            failures.append("label_feature_vector_empty")
    return failures


def _score_dataset_rows(
    artifact: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    scores = []
    for row in rows:
        trainable = _mapping(row.get("trainable"))
        features = _mapping(trainable.get("features"))
        label = str(_mapping(trainable.get("label")).get("action", ""))
        scored = score_support_gated_residual_artifact(
            artifact=artifact,
            observation_input=_mapping(features.get("observation_input")),
            action_mask=_mapping(features.get("action_mask")),
            public_history_trace=[],
            linear_action=label,
        )
        selected = scored.get("selected_action")
        scores.append(
            {
                "label_action": label,
                "selected_action": selected,
                "label_match": selected == label,
                "selected_score": scored.get("selected_score"),
                "nearest_support_distance": scored.get("nearest_support_distance"),
                "score_margin": scored.get("score_margin"),
                "support_gate_passed": scored.get("support_gate_passed"),
                "candidate_scores_top": scored.get("candidate_scores_top"),
            }
        )
    return scores


def _seed_41_non_regression_status(
    *,
    v143_report: Mapping[str, object],
    v143_report_path: str | Path | None,
) -> dict[str, object]:
    matrix = _mapping(v143_report.get("matrix"))
    broad_seeds = tuple(_int(seed) for seed in _list(matrix.get("broad_seeds")))
    regression_seeds = tuple(
        _int(seed)
        for seed in _list(
            matrix.get("regression_seeds")
            or _mapping(v143_report.get("acceptance")).get("regression_seeds")
        )
    )
    status: dict[str, object] = {
        "policy": "v144_seed_41_v143_non_regression_status_v1",
        "seed": 41,
        "strict_broad_seed_present": 41 in broad_seeds,
        "listed_as_regression_seed": 41 in regression_seeds,
        "classification": (
            "non_regression_in_v143"
            if 41 in broad_seeds and 41 not in regression_seeds
            else "missing_or_regression"
        ),
        "missing_support_is_expected": 41 in broad_seeds and 41 not in regression_seeds,
        "passed": 41 in broad_seeds and 41 not in regression_seeds,
    }
    live_path = _v142_live_report_path(v143_report, v143_report_path)
    if live_path is None or not live_path.exists():
        status["v142_live_report_checked"] = False
        status["v142_live_report_path"] = str(live_path) if live_path else None
        return status
    try:
        live = load_json_report(live_path)
    except BranchInterventionResidualError:
        status["v142_live_report_checked"] = False
        status["v142_live_report_path"] = str(live_path)
        return status
    delta = _broad_seed_delta(live, seed=41)
    status["v142_live_report_checked"] = True
    status["v142_live_report_path"] = str(live_path)
    status["v142_alive_delta"] = delta.get("alive_delta")
    status["v142_births_delta"] = delta.get("births_delta")
    non_regression = (
        _int(delta.get("alive_delta")) >= 0 and _int(delta.get("births_delta")) >= 0
    )
    status["v142_live_non_regression"] = non_regression
    status["passed"] = bool(status["passed"]) and non_regression
    if not status["passed"]:
        status["classification"] = "seed_41_regression_or_missing_in_v142"
    return status


def _v142_live_report_path(
    v143_report: Mapping[str, object],
    v143_report_path: str | Path | None,
) -> Path | None:
    raw = _mapping(v143_report.get("inputs")).get("v142_live_report")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if path.is_absolute() or v143_report_path is None:
        return path
    return Path(v143_report_path).resolve().parents[2] / path


def _broad_seed_delta(
    report: Mapping[str, object],
    *,
    seed: int,
) -> Mapping[str, object]:
    for item in _list_of_mappings(_mapping(report.get("broad")).get("per_seed_delta")):
        if _int(item.get("seed")) == int(seed):
            return item
    return {}


def _run_broad_policy(
    *,
    seed: int,
    ticks: int,
    runtime_artifact: Mapping[str, object] | None,
    runtime_mode: str = "live",
) -> dict[str, object]:
    policy = _policy(
        seed=seed,
        runtime_artifact=runtime_artifact,
        runtime_mode=runtime_mode,
    )
    world = SimulationWorld(
        WorldConfig(seed=int(seed), max_ticks=int(ticks)),
        policy=policy,
    )
    return _run_world_with_support(
        world=world,
        seed=int(seed),
        ticks=int(ticks),
    )


def _run_carrion_fixture_suite(
    *,
    seeds: Sequence[int],
    ticks: int,
    runtime_artifact: Mapping[str, object] | None,
    runtime_mode: str,
    policy_name: str,
) -> dict[str, object]:
    runs = []
    for seed in seeds:
        policy = _policy(
            seed=seed,
            runtime_artifact=runtime_artifact,
            runtime_mode=runtime_mode,
        )
        world = evaluate_cli._fixture_world(
            fixture_name="carrion_only",
            seed=int(seed),
            ticks=int(ticks),
            policy=policy,
        )
        run = _run_world_with_support(
            world=world,
            seed=int(seed),
            ticks=int(ticks),
        )
        run["fixture"] = "carrion_only"
        run["evaluation_context"] = "controlled_ecology_fixture"
        runs.append(run)
    aggregate = _aggregate_runs(runs)
    return {
        "policy": evaluate_cli.MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
        "suite": "basic",
        "fixture_names": ["carrion_only"],
        "evaluated_policy_key": "mind_v3",
        "evaluated_policy_name": policy_name,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "fixtures": [
            {
                "fixture": "carrion_only",
                "scenario_config": evaluate_cli._fixture_scenario_config(
                    "carrion_only"
                ),
                "comparison": {
                    "mind_v3": {
                        "runs": runs,
                        "aggregate": aggregate,
                    }
                },
            }
        ],
    }


def _policy(
    *,
    seed: int,
    runtime_artifact: Mapping[str, object] | None,
    runtime_mode: str,
) -> MindV3EvolutionPolicy:
    if runtime_artifact is None:
        return MindV3EvolutionPolicy(seed=int(seed))
    return MindV3EvolutionPolicy(
        seed=int(seed),
        support_residual_artifact=runtime_artifact,
        support_residual_runtime_mode=runtime_mode,
    )


def _run_world_with_support(
    *,
    world: SimulationWorld,
    seed: int,
    ticks: int,
) -> dict[str, object]:
    run = evaluate_cli._run_world(
        world=world,
        seed=int(seed),
        ticks=int(ticks),
    )
    diagnostics = support_gated_residual_runtime_diagnostics(
        world.policy_decision_diagnostics_records,
    )
    run["support_residual_diagnostics"] = diagnostics
    run["unsupported_proposed_action_count"] = int(
        diagnostics.get("unsupported_proposed_action_count", 0)
    )
    run["resolved_invalid_action_count"] = int(
        run.get("unsupported_resolved_action_count", 0)
    )
    return run


def _aggregate_runs(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    aggregate = evaluate_cli._aggregate_runs([dict(run) for run in runs])
    diagnostics = aggregate_support_gated_residual_runtime_diagnostics(runs)
    aggregate["support_residual_diagnostics"] = diagnostics
    aggregate["unsupported_proposed_action_count"] = int(
        diagnostics.get("unsupported_proposed_action_count", 0)
    )
    aggregate["resolved_invalid_action_count"] = int(
        aggregate.get("unsupported_resolved_action_count", 0)
    )
    return aggregate


def _comparison_section(
    *,
    linear_runs: Sequence[Mapping[str, object]],
    residual_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    linear_aggregate = _aggregate_runs(linear_runs)
    residual_aggregate = _aggregate_runs(residual_runs)
    return {
        "linear": {
            "runs": [dict(run) for run in linear_runs],
            "aggregate": linear_aggregate,
        },
        "residual": {
            "runs": [dict(run) for run in residual_runs],
            "aggregate": residual_aggregate,
        },
        "aggregate_delta": _aggregate_delta(
            linear=linear_aggregate,
            residual=residual_aggregate,
        ),
        "per_seed_delta": _per_seed_delta(
            linear_runs=linear_runs,
            residual_runs=residual_runs,
        ),
    }


def _aggregate_delta(
    *,
    linear: Mapping[str, object],
    residual: Mapping[str, object],
) -> dict[str, object]:
    residual_diag = _mapping(residual.get("support_residual_diagnostics"))
    return {
        "alive_agents_mean": _round(
            _float(residual.get("alive_agents_mean"))
            - _float(linear.get("alive_agents_mean"))
        ),
        "births_mean": _round(
            _float(residual.get("births_mean")) - _float(linear.get("births_mean"))
        ),
        "deaths_mean": _round(
            _float(residual.get("deaths_mean")) - _float(linear.get("deaths_mean"))
        ),
        "unsupported_requested_action_count_delta": int(
            _int(residual.get("unsupported_requested_action_count"))
            - _int(linear.get("unsupported_requested_action_count"))
        ),
        "resolved_invalid_action_count_delta": int(
            _int(residual.get("resolved_invalid_action_count"))
            - _int(linear.get("resolved_invalid_action_count"))
        ),
        "heuristic_action_source_count_delta": int(
            _int(residual.get("heuristic_action_source_count"))
            - _int(linear.get("heuristic_action_source_count"))
        ),
        "dominant_requested_action_share_delta": _round(
            _float(residual.get("dominant_requested_action_share"))
            - _float(linear.get("dominant_requested_action_share"))
        ),
        "applied_override_count": _int(residual_diag.get("applied_override_count")),
        "applied_override_share": _float(residual_diag.get("applied_override_share")),
        "applied_override_action_counts": residual_diag.get(
            "applied_override_action_counts"
        ),
    }


def _per_seed_delta(
    *,
    linear_runs: Sequence[Mapping[str, object]],
    residual_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    linear_by_seed = {_int(run.get("seed")): run for run in linear_runs}
    rows = []
    for residual in sorted(residual_runs, key=lambda item: _int(item.get("seed"))):
        seed = _int(residual.get("seed"))
        linear = _mapping(linear_by_seed.get(seed))
        diagnostics = _mapping(residual.get("support_residual_diagnostics"))
        rows.append(
            {
                "seed": seed,
                "linear_alive_agents": _int(linear.get("alive_agents")),
                "residual_alive_agents": _int(residual.get("alive_agents")),
                "alive_delta": _int(residual.get("alive_agents"))
                - _int(linear.get("alive_agents")),
                "linear_births": _int(linear.get("births")),
                "residual_births": _int(residual.get("births")),
                "births_delta": _int(residual.get("births"))
                - _int(linear.get("births")),
                "unsupported_requested_action_count_delta": (
                    _int(residual.get("unsupported_requested_action_count"))
                    - _int(linear.get("unsupported_requested_action_count"))
                ),
                "resolved_invalid_action_count_delta": (
                    _int(residual.get("resolved_invalid_action_count"))
                    - _int(linear.get("resolved_invalid_action_count"))
                ),
                "unsupported_proposed_action_count": _int(
                    diagnostics.get("unsupported_proposed_action_count")
                ),
                "applied_override_count": _int(
                    diagnostics.get("applied_override_count")
                ),
                "applied_override_action_counts": diagnostics.get(
                    "applied_override_action_counts"
                ),
                "dominant_applied_override_action": diagnostics.get(
                    "dominant_applied_override_action"
                ),
                "dominant_applied_override_action_share": diagnostics.get(
                    "dominant_applied_override_action_share"
                ),
            }
        )
    return rows


def _shadow_gate(
    *,
    aggregate: Mapping[str, object],
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []

    def block(reason: str, field: str, observed: object, required: object, comparator: str) -> None:
        blockers.append(
            {
                "reason": reason,
                "field": field,
                "observed": observed,
                "required": required,
                "comparator": comparator,
            }
        )

    if _int(aggregate.get("unsupported_requested_action_count")) != 0:
        block(
            "unsupported_requested_action",
            "unsupported_requested_action_count",
            aggregate.get("unsupported_requested_action_count"),
            0,
            "eq",
        )
    if _int(diagnostics.get("unsupported_proposed_action_count")) != 0:
        block(
            "unsupported_proposed_action",
            "unsupported_proposed_action_count",
            diagnostics.get("unsupported_proposed_action_count"),
            0,
            "eq",
        )
    if _int(diagnostics.get("gate_accepted_override_count")) <= 0:
        block(
            "no_gate_accepted_shadow_overrides",
            "gate_accepted_override_count",
            diagnostics.get("gate_accepted_override_count"),
            0,
            "gt",
        )
    if (
        _float(diagnostics.get("dominant_gate_accepted_override_action_share"))
        > V144_MAX_DOMINANT_ACTION_SHARE
    ):
        block(
            "dominant_shadow_override_action_share_above_cap",
            "dominant_gate_accepted_override_action_share",
            diagnostics.get("dominant_gate_accepted_override_action_share"),
            V144_MAX_DOMINANT_ACTION_SHARE,
            "le",
        )
    return {
        "policy": "v144_shadow_strict_broad_gate_v1",
        "passed": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "runtime_promotion_allowed": False,
    }


def _acceptance(
    *,
    training_report: Mapping[str, object],
    shadow: Mapping[str, object],
    broad: Mapping[str, object] | None,
    carrion: Mapping[str, object] | None,
    broad_seeds: Sequence[int],
    ticks: int,
    fixture_seeds: Sequence[int],
    fixture_ticks: int,
) -> dict[str, object]:
    floors: list[dict[str, object]] = []
    training_source = _mapping(training_report.get("source_integrity"))
    roundtrip = _mapping(training_report.get("artifact_roundtrip"))
    shadow_gate = _mapping(shadow.get("shadow_gate"))
    floors.extend(
        [
            _floor(
                "v143_source_integrity_passed",
                training_source.get("passed") is True,
                observed=training_source.get("failures"),
                required=[],
            ),
            _floor(
                "artifact_roundtrip_score_equivalent",
                roundtrip.get("loaded_artifact_scores_match_pre_serialization")
                is True,
                observed=roundtrip.get("mismatch_count"),
                required=0,
            ),
            _floor(
                "shadow_strict_broad_seed_set",
                tuple(int(seed) for seed in broad_seeds) == STRICT_BROAD_SEEDS,
                observed=list(broad_seeds),
                required=list(STRICT_BROAD_SEEDS),
                fixture="broad",
            ),
            _floor(
                "shadow_strict_broad_ticks",
                int(ticks) == STRICT_TICKS,
                observed=int(ticks),
                required=STRICT_TICKS,
                fixture="broad",
            ),
            _floor(
                "shadow_strict_broad_gate_passed",
                shadow_gate.get("passed") is True,
                observed=shadow_gate.get("blockers"),
                required=[],
                fixture="broad",
            ),
        ]
    )
    if broad is None or carrion is None:
        first_failed = next((floor for floor in floors if floor["passed"] is not True), None)
        return _acceptance_payload(
            floors=floors,
            first_failed=first_failed,
            broad=broad,
            carrion=carrion,
        )

    broad_residual = _mapping(_mapping(broad.get("residual")).get("aggregate"))
    carrion_residual = _mapping(_mapping(carrion.get("residual")).get("aggregate"))
    broad_linear = _mapping(_mapping(broad.get("linear")).get("aggregate"))
    carrion_linear = _mapping(_mapping(carrion.get("linear")).get("aggregate"))
    broad_diag = _mapping(broad_residual.get("support_residual_diagnostics"))
    carrion_diag = _mapping(carrion_residual.get("support_residual_diagnostics"))
    broad_per_seed = _list_of_mappings(broad.get("per_seed_delta"))
    carrion_per_seed = _list_of_mappings(carrion.get("per_seed_delta"))
    linear_gate = _mapping(carrion.get("linear_fixture_gate"))
    residual_gate = _mapping(carrion.get("residual_fixture_gate"))
    linear_blockers = _list_of_mappings(linear_gate.get("blockers"))
    residual_blockers = _list_of_mappings(residual_gate.get("blockers"))
    carrion_delta = _mapping(carrion.get("aggregate_delta"))
    total_heuristic = _int(broad_residual.get("heuristic_action_source_count")) + _int(
        carrion_residual.get("heuristic_action_source_count")
    )
    total_unsupported_requested = _int(
        broad_residual.get("unsupported_requested_action_count")
    ) + _int(carrion_residual.get("unsupported_requested_action_count"))
    total_unsupported_proposed = _int(
        broad_diag.get("unsupported_proposed_action_count")
    ) + _int(carrion_diag.get("unsupported_proposed_action_count"))
    total_applied = _int(broad_diag.get("applied_override_count")) + _int(
        carrion_diag.get("applied_override_count")
    )
    max_requested_share = max(
        _float(broad_residual.get("dominant_requested_action_share")),
        _float(carrion_residual.get("dominant_requested_action_share")),
    )
    max_applied_share = max(
        _float(broad_diag.get("dominant_applied_override_action_share")),
        _float(carrion_diag.get("dominant_applied_override_action_share")),
    )
    floors.extend(
        [
            _floor(
                "strict_broad_seed_set",
                tuple(int(seed) for seed in broad_seeds) == STRICT_BROAD_SEEDS,
                observed=list(broad_seeds),
                required=list(STRICT_BROAD_SEEDS),
                fixture="broad",
            ),
            _floor(
                "strict_broad_ticks",
                int(ticks) == STRICT_TICKS,
                observed=int(ticks),
                required=STRICT_TICKS,
                fixture="broad",
            ),
            _floor(
                "strict_carrion_fixture_seed_set",
                tuple(int(seed) for seed in fixture_seeds)
                == STRICT_CARRION_FIXTURE_SEEDS,
                observed=list(fixture_seeds),
                required=list(STRICT_CARRION_FIXTURE_SEEDS),
                fixture="carrion_only",
            ),
            _floor(
                "strict_carrion_fixture_ticks",
                int(fixture_ticks) == STRICT_TICKS,
                observed=int(fixture_ticks),
                required=STRICT_TICKS,
                fixture="carrion_only",
            ),
            _floor(
                "zero_heuristic_action_source_count",
                total_heuristic == 0,
                observed=total_heuristic,
                required=0,
            ),
            _floor(
                "zero_unsupported_requested_action_count",
                total_unsupported_requested == 0,
                observed=total_unsupported_requested,
                required=0,
            ),
            _floor(
                "zero_unsupported_proposed_action_count",
                total_unsupported_proposed == 0,
                observed=total_unsupported_proposed,
                required=0,
            ),
            _floor(
                "broad_resolved_invalid_not_increased",
                _int(broad_residual.get("resolved_invalid_action_count"))
                <= _int(broad_linear.get("resolved_invalid_action_count")),
                observed={
                    "linear": broad_linear.get("resolved_invalid_action_count"),
                    "residual": broad_residual.get("resolved_invalid_action_count"),
                },
                required="residual <= linear",
                fixture="broad",
            ),
            _floor(
                "carrion_resolved_invalid_not_increased",
                _int(carrion_residual.get("resolved_invalid_action_count"))
                <= _int(carrion_linear.get("resolved_invalid_action_count")),
                observed={
                    "linear": carrion_linear.get("resolved_invalid_action_count"),
                    "residual": carrion_residual.get("resolved_invalid_action_count"),
                },
                required="residual <= linear",
                fixture="carrion_only",
            ),
            _floor(
                "dominant_requested_action_share_lte_0_50",
                max_requested_share <= V144_MAX_DOMINANT_ACTION_SHARE,
                observed=max_requested_share,
                required=V144_MAX_DOMINANT_ACTION_SHARE,
            ),
            _floor(
                "dominant_applied_residual_action_share_lte_0_50",
                max_applied_share <= V144_MAX_DOMINANT_ACTION_SHARE,
                observed=max_applied_share,
                required=V144_MAX_DOMINANT_ACTION_SHARE,
            ),
            _floor(
                "applied_override_count_nonzero",
                total_applied > 0,
                observed=total_applied,
                required=">0",
            ),
        ]
    )
    for item in broad_per_seed:
        seed = _int(item.get("seed"))
        floors.append(
            _floor(
                f"broad_seed_{seed}_alive_no_regression",
                _int(item.get("alive_delta")) >= 0,
                observed=item.get("alive_delta"),
                required=0,
                seed=seed,
                fixture="broad",
            )
        )
        floors.append(
            _floor(
                f"broad_seed_{seed}_births_no_regression",
                _int(item.get("births_delta")) >= 0,
                observed=item.get("births_delta"),
                required=0,
                seed=seed,
                fixture="broad",
            )
        )
        floors.append(
            _floor(
                f"broad_seed_{seed}_resolved_invalid_not_increased",
                _int(item.get("resolved_invalid_action_count_delta")) <= 0,
                observed=item.get("resolved_invalid_action_count_delta"),
                required="<=0",
                seed=seed,
                fixture="broad",
            )
        )
    for item in carrion_per_seed:
        seed = _int(item.get("seed"))
        floors.append(
            _floor(
                f"carrion_seed_{seed}_resolved_invalid_not_increased",
                _int(item.get("resolved_invalid_action_count_delta")) <= 0,
                observed=item.get("resolved_invalid_action_count_delta"),
                required="<=0",
                seed=seed,
                fixture="carrion_only",
            )
        )
    floors.append(
        _floor(
            "carrion_alive_improves_or_blocker_count_reduces",
            _float(carrion_delta.get("alive_agents_mean")) > 0.0
            or len(residual_blockers) < len(linear_blockers),
            observed={
                "alive_delta": carrion_delta.get("alive_agents_mean"),
                "linear_blocker_count": len(linear_blockers),
                "residual_blocker_count": len(residual_blockers),
            },
            required=(
                "carrion alive mean delta > 0 or residual blocker count "
                "< linear blocker count"
            ),
            fixture="carrion_only",
        )
    )
    first_failed = next((floor for floor in floors if floor["passed"] is not True), None)
    return _acceptance_payload(
        floors=floors,
        first_failed=first_failed,
        broad=broad,
        carrion=carrion,
        total_applied=total_applied,
    )


def _acceptance_payload(
    *,
    floors: Sequence[Mapping[str, object]],
    first_failed: Mapping[str, object] | None,
    broad: Mapping[str, object] | None,
    carrion: Mapping[str, object] | None,
    total_applied: int = 0,
) -> dict[str, object]:
    first_seed_failure = next(
        (
            floor
            for floor in floors
            if floor.get("passed") is not True and "seed" in floor
        ),
        None,
    )
    first_fixture_failure = next(
        (
            floor
            for floor in floors
            if floor.get("passed") is not True and "fixture" in floor
        ),
        None,
    )
    return {
        "policy": MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_ACCEPTANCE_POLICY,
        "passed": first_failed is None,
        "first_failed_floor": None if first_failed is None else first_failed.get("name"),
        "first_failing_seed": (
            None if first_seed_failure is None else first_seed_failure.get("seed")
        ),
        "first_failing_fixture": (
            None
            if first_fixture_failure is None
            else first_fixture_failure.get("fixture")
        ),
        "first_failed_action_distribution": (
            None
            if first_failed is None
            else _first_failed_action_distribution(broad=broad, carrion=carrion)
        ),
        "applied_override_count": total_applied,
        "floor_count": len(floors),
        "floors": [dict(floor) for floor in floors],
        "runtime_promotion_allowed": False,
        "training_promotion_allowed": False,
    }


def _first_failed_action_distribution(
    *,
    broad: Mapping[str, object] | None,
    carrion: Mapping[str, object] | None,
) -> dict[str, object]:
    broad_residual = _mapping(_mapping((broad or {}).get("residual")).get("aggregate"))
    carrion_residual = _mapping(
        _mapping((carrion or {}).get("residual")).get("aggregate")
    )
    broad_diag = _mapping(broad_residual.get("support_residual_diagnostics"))
    carrion_diag = _mapping(carrion_residual.get("support_residual_diagnostics"))
    return {
        "broad_requested_action_counts": broad_residual.get(
            "requested_action_counts"
        ),
        "carrion_requested_action_counts": carrion_residual.get(
            "requested_action_counts"
        ),
        "broad_dominant_requested_action_share": broad_residual.get(
            "dominant_requested_action_share"
        ),
        "carrion_dominant_requested_action_share": carrion_residual.get(
            "dominant_requested_action_share"
        ),
        "broad_applied_override_action_counts": broad_diag.get(
            "applied_override_action_counts"
        ),
        "carrion_applied_override_action_counts": carrion_diag.get(
            "applied_override_action_counts"
        ),
        "broad_abstention_reason_counts": broad_diag.get(
            "abstention_reason_counts"
        ),
        "carrion_abstention_reason_counts": carrion_diag.get(
            "abstention_reason_counts"
        ),
    }


def _floor(
    name: str,
    passed: bool,
    *,
    observed: object,
    required: object,
    seed: int | None = None,
    fixture: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if fixture is not None:
        payload["fixture"] = fixture
    return payload


def _fixture_runs(fixture_suite: Mapping[str, object]) -> list[Mapping[str, object]]:
    fixture = evaluate_cli._fixture_report_by_name(fixture_suite, "carrion_only")
    evaluated_key = str(fixture_suite.get("evaluated_policy_key", "mind_v3"))
    return _list_of_mappings(
        _mapping(_mapping(fixture.get("comparison")).get(evaluated_key)).get("runs")
    )


def _dominant_count_share(counter: Mapping[str, int] | Counter[str]) -> dict[str, object]:
    total = sum(int(value) for value in counter.values())
    if total <= 0:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = max(
        ((str(key), int(value)) for key, value in counter.items()),
        key=lambda item: (item[1], item[0]),
    )
    return {
        "key": key,
        "count": int(count),
        "share": _round(float(count) / float(total)),
    }


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_input(path: Path) -> TextIO:
    return path.open("rt", encoding="utf-8")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return None


def _int(value: object) -> int:
    parsed = _optional_int(value)
    return 0 if parsed is None else parsed


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _round(value: float) -> float:
    return round(float(value), 6)
