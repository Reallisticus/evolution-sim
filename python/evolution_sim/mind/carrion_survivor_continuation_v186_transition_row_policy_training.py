from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v178_transition_row_dataset_audit as v178,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v180_transition_row_policy_training as v180,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v185_v183_target_resolution_repair as v185,
)
from evolution_sim.mind.candidate_campaign import _mapping, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    load_jsonl_dataset,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.transition_value_scorer import (
    MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
    load_transition_value_scorer_artifact,
)

M3_CARRION_SURVIVOR_CONTINUATION_V186_TRANSITION_ROW_POLICY_TRAINING_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v186_transition_row_policy_training_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V186_TRANSITION_ROW_POLICY_TRAINING_POLICY = (
    "opt_in_m3_carrion_survivor_continuation_v186_transition_row_policy_training_slice_2_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V186_ARTIFACT_POLICY = (
    "opt_in_m3_carrion_survivor_continuation_v186_public_transition_row_policy_v1"
)
V186_CANDIDATE_KEY = "v186_transition_row_policy_slice_2"

DEFAULT_AUTHORIZATION_REPORT_PATH = v185.DEFAULT_REPAIRED_DATASET_AUDIT_OUTPUT_PATH
DEFAULT_TRANSITION_DATASET_PATH = v185.DEFAULT_REPAIRED_TRANSITION_DATASET_OUTPUT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v186-carrion-survivor-continuation-transition-row-policy-training.json"
)
DEFAULT_ARTIFACT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v186-carrion-survivor-continuation-transition-row-policy-artifact.json"
)

EXPECTED_V185_REPAIRED_AUDIT_EXACT_DIGEST = (
    "abd8c06733373b441c337187191cb04d3755968335b97f2fbcb350f550db8a50"
)
EXPECTED_V185_REPAIRED_TRANSITION_DATASET_DIGEST = (
    "532817eb68cebf34cfb27f8abbbd86631cb142e03127661286d88d5154320f51"
)
EXPECTED_V185_REPAIRED_AUDIT_CLASSIFICATION = v185.V185_AUDIT_AUTHORIZED_CLASSIFICATION
EXPECTED_V185_SOURCE_PRODUCER = v178.V185_SOURCE_PRODUCER
EXPECTED_V186_TRAINING_ROUTE = v178.V186_SLICE_2_TRAINING_ROUTE

TRAINING_SLICE_INDEX = 2
PREVIOUS_TRAINING_SLICES_CONSUMED = 1
CAMPAIGN_SLICE_CAP = 10

DEFAULT_BROAD_SEEDS = v180.DEFAULT_BROAD_SEEDS
DEFAULT_CARRION_FIXTURE_SEEDS = v180.DEFAULT_CARRION_FIXTURE_SEEDS
DEFAULT_TICKS = v180.DEFAULT_TICKS
DEFAULT_FEATURE_KEY_LIMIT = v180.DEFAULT_FEATURE_KEY_LIMIT
DEFAULT_UNOBSERVED_ACTION_UTILITY = v180.DEFAULT_UNOBSERVED_ACTION_UTILITY
MAX_DOMINANT_REQUESTED_ACTION_SHARE = v180.MAX_DOMINANT_REQUESTED_ACTION_SHARE


def run_carrion_survivor_continuation_v186_transition_row_policy_training(
    *,
    authorization_report_path: str | Path = DEFAULT_AUTHORIZATION_REPORT_PATH,
    transition_dataset_path: str | Path = DEFAULT_TRANSITION_DATASET_PATH,
    artifact_output_path: str | Path = DEFAULT_ARTIFACT_OUTPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_authorization_report_exact_digest: str = (
        EXPECTED_V185_REPAIRED_AUDIT_EXACT_DIGEST
    ),
    expected_dataset_digest: str = EXPECTED_V185_REPAIRED_TRANSITION_DATASET_DIGEST,
    expected_authorization_classification: str = (
        EXPECTED_V185_REPAIRED_AUDIT_CLASSIFICATION
    ),
    expected_source_producer: str = EXPECTED_V185_SOURCE_PRODUCER,
    expected_training_route: str = EXPECTED_V186_TRAINING_ROUTE,
    feature_key_limit: int = DEFAULT_FEATURE_KEY_LIMIT,
    unobserved_action_utility: float = DEFAULT_UNOBSERVED_ACTION_UTILITY,
    run_evaluation: bool = True,
    broad_seeds: Sequence[int] = DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = DEFAULT_TICKS,
) -> dict[str, object]:
    authorization_report = load_json_report(authorization_report_path)
    rows = [dict(row) for row in load_jsonl_dataset(transition_dataset_path)]
    authorization_validation = validate_v186_transition_row_training_authorization_report(
        authorization_report,
        expected_exact_digest=expected_authorization_report_exact_digest,
        expected_dataset_digest=expected_dataset_digest,
        expected_classification=expected_authorization_classification,
        expected_source_producer=expected_source_producer,
        expected_training_route=expected_training_route,
    )
    source_validation = _source_validation(
        rows=rows,
        authorization_validation=authorization_validation,
        expected_dataset_digest=expected_dataset_digest,
    )
    artifact: dict[str, object]
    training: dict[str, object]
    evaluation: dict[str, object]
    acceptance: dict[str, object]
    if source_validation.get("passed") is not True:
        artifact = _artifact_status(
            artifact_output_path,
            created=False,
            reason="source_validation_failed",
        )
        training = _skipped_training("source_validation_failed")
        evaluation = _skipped_evaluation("source_validation_failed")
        acceptance = _closed_acceptance("source_validation_failed")
    else:
        artifact_payload, training = build_v186_transition_row_policy_artifact(
            rows,
            dataset_digest=expected_dataset_digest,
            authorization_report_exact_digest=expected_authorization_report_exact_digest,
            authorization_route=expected_training_route,
            source_producer=expected_source_producer,
            feature_key_limit=feature_key_limit,
            unobserved_action_utility=unobserved_action_utility,
        )
        write_json(artifact_output_path, artifact_payload)
        artifact_digest = stable_payload_digest(artifact_payload)
        artifact = _artifact_status(
            artifact_output_path,
            created=True,
            reason=None,
            payload=artifact_payload,
            digest=artifact_digest,
        )
        if run_evaluation:
            evaluation = run_v186_shadow_evaluation(
                artifact=artifact_payload,
                broad_seeds=broad_seeds,
                carrion_fixture_seeds=carrion_fixture_seeds,
                ticks=ticks,
            )
            acceptance = _acceptance(evaluation)
        else:
            evaluation = _skipped_evaluation("run_evaluation_false")
            acceptance = _closed_acceptance("evaluation_not_run")
    classification = _classification(
        source_validation=source_validation,
        training=training,
        evaluation=evaluation,
        acceptance=acceptance,
    )
    slice_consumed = training.get("ran") is True and artifact.get("created") is True
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V186_TRANSITION_ROW_POLICY_TRAINING_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V186_TRANSITION_ROW_POLICY_TRAINING_POLICY,
        "contract": _contract(
            expected_authorization_report_exact_digest=(
                expected_authorization_report_exact_digest
            ),
            expected_dataset_digest=expected_dataset_digest,
            expected_authorization_classification=(
                expected_authorization_classification
            ),
            expected_source_producer=expected_source_producer,
            expected_training_route=expected_training_route,
        ),
        "inputs": {
            "authorization_report": str(authorization_report_path),
            "transition_dataset": str(transition_dataset_path),
            "artifact_output": str(artifact_output_path),
            "expected_authorization_report_exact_digest": (
                expected_authorization_report_exact_digest
            ),
            "expected_dataset_digest": expected_dataset_digest,
            "expected_authorization_classification": (
                expected_authorization_classification
            ),
            "expected_source_producer": expected_source_producer,
            "expected_training_route": expected_training_route,
            "feature_key_limit": int(feature_key_limit),
            "unobserved_action_utility": v180._round(unobserved_action_utility),
            "run_evaluation": bool(run_evaluation),
            "broad_seeds": [int(seed) for seed in broad_seeds],
            "carrion_fixture_seeds": [int(seed) for seed in carrion_fixture_seeds],
            "ticks": int(ticks),
        },
        "authorization_report_validation": authorization_validation,
        "source_validation": source_validation,
        "dataset": {
            "path": str(transition_dataset_path),
            "row_count": len(rows),
            "dataset_digest": stable_payload_digest(rows),
            "source_producer": expected_source_producer,
        },
        "artifact": artifact,
        "training": training,
        "evaluation": evaluation,
        "acceptance": acceptance,
        "training_slice_budget": {
            "campaign": "carrion_transition_row_policy",
            "slice_cap": CAMPAIGN_SLICE_CAP,
            "previous_slices_consumed": PREVIOUS_TRAINING_SLICES_CONSUMED,
            "this_slice_index": TRAINING_SLICE_INDEX,
            "this_slice_consumed": slice_consumed,
            "current_slices_consumed": (
                TRAINING_SLICE_INDEX
                if slice_consumed
                else PREVIOUS_TRAINING_SLICES_CONSUMED
            ),
        },
        "classification": {"primary": classification, "labels": [classification]},
        **_lifecycle_flags(
            source_authorized=source_validation.get("passed") is True,
            training_ran=training.get("ran") is True,
            training_artifact_created=artifact.get("created") is True,
            shadow_eval_ran=evaluation.get("ran") is True,
            slice_2_training_consumed=slice_consumed,
        ),
    }
    report["exact_digest"] = _digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v186_transition_row_training_authorization_report(
    report: Mapping[str, object],
    *,
    expected_exact_digest: str,
    expected_dataset_digest: str,
    expected_classification: str,
    expected_source_producer: str = EXPECTED_V185_SOURCE_PRODUCER,
    expected_training_route: str = EXPECTED_V186_TRAINING_ROUTE,
) -> dict[str, object]:
    base_validation = v178.validate_v178_transition_row_training_authorization_report(
        report,
        expected_exact_digest=expected_exact_digest,
        expected_dataset_digest=expected_dataset_digest,
        expected_classification=expected_classification,
        expected_source_producer=expected_source_producer,
    )
    exact_validation = exact_digest_validation_report(report)
    authorization = _mapping(report.get("training_authorization"))
    route = _mapping(report.get("route_recommendation"))
    contract = _mapping(report.get("contract"))
    dataset = _mapping(report.get("dataset"))
    source = _mapping(report.get("source_validation"))
    classification = _mapping(report.get("classification"))
    checks = {
        "base_authorization_report_validation_passed": (
            base_validation.get("passed") is True
        ),
        "schema_version_matches_v185_repaired_audit": (
            report.get("schema_version")
            == v185.M3_CARRION_SURVIVOR_CONTINUATION_V185_REPAIRED_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION
        ),
        "policy_matches_v185_repaired_audit": (
            report.get("policy")
            == v185.M3_CARRION_SURVIVOR_CONTINUATION_V185_REPAIRED_TRANSITION_ROW_DATASET_AUDIT_POLICY
        ),
        "exact_digest_valid": exact_validation.get("passed") is True,
        "expected_exact_digest_matches": (
            str(report.get("exact_digest") or "") == str(expected_exact_digest)
        ),
        "classification_matches_v185_authorized": (
            str(classification.get("primary") or "") == str(expected_classification)
        ),
        "dataset_digest_matches_expected": (
            str(dataset.get("dataset_digest") or "") == str(expected_dataset_digest)
        ),
        "source_validation_passed": source.get("passed") is True,
        "source_producer_matches_v185_repair": (
            str(source.get("source_producer") or "") == str(expected_source_producer)
        ),
        "training_authorization_authorized": (
            authorization.get("authorized") is True
        ),
        "training_authorization_slice_authorized": (
            authorization.get("next_same_lane_opt_in_training_slice_authorized")
            is True
        ),
        "training_authorization_failures_empty": (
            list(authorization.get("failures") or []) == []
        ),
        "route_matches_v186_slice_2_opt_in": (
            str(route.get("recommended_next_route") or "")
            == str(expected_training_route)
        ),
        "route_slice_2_opt_in_authorized": (
            route.get("slice_2_opt_in_training_route_authorized") is True
        ),
        "route_slice_2_training_authorized": (
            route.get("slice_2_training_authorized") is True
        ),
        "route_first_slice_not_authorized": (
            route.get("first_opt_in_training_slice_authorized") is False
        ),
        "contract_slice_2_route_authorized": (
            contract.get("slice_2_opt_in_training_route_authorized") is True
        ),
        "contract_first_slice_not_authorized": (
            contract.get("first_opt_in_training_slice_authorized") is False
        ),
        "v185_audit_training_did_not_run": report.get("training_ran") is False,
        "v185_audit_training_artifact_not_created": (
            report.get("training_artifact_created") is False
        ),
        "v185_audit_slice_2_not_consumed": (
            report.get("slice_2_training_consumed") is False
        ),
        "v185_audit_runtime_artifact_not_created": (
            report.get("runtime_artifact_created") is False
        ),
        "v185_audit_runtime_action_selection_unchanged": (
            report.get("runtime_action_selection_changed") is False
        ),
        "v185_audit_promotion_not_authorized": (
            report.get("promotion_authorized") is False
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": (
            "m3_carrion_survivor_continuation_v186_transition_row_training_"
            "authorization_report_validation_v1"
        ),
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_exact_digest": expected_exact_digest,
        "observed_exact_digest": report.get("exact_digest"),
        "exact_digest_validation": exact_validation,
        "v178_style_authorization_report_validation": base_validation,
        "expected_classification": expected_classification,
        "observed_classification": classification.get("primary"),
        "expected_dataset_digest": expected_dataset_digest,
        "observed_dataset_digest": dataset.get("dataset_digest"),
        "expected_source_producer": expected_source_producer,
        "observed_source_producer": source.get("source_producer"),
        "expected_training_route": expected_training_route,
        "observed_training_route": route.get("recommended_next_route"),
        "training_authorization": dict(authorization),
        "route_recommendation": dict(route),
        "contract": dict(contract),
    }


def build_v186_transition_row_policy_artifact(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset_digest: str,
    authorization_report_exact_digest: str,
    authorization_route: str,
    source_producer: str,
    feature_key_limit: int = DEFAULT_FEATURE_KEY_LIMIT,
    unobserved_action_utility: float = DEFAULT_UNOBSERVED_ACTION_UTILITY,
) -> tuple[dict[str, object], dict[str, object]]:
    artifact, training = v180.build_v180_transition_row_policy_artifact(
        rows,
        dataset_digest=dataset_digest,
        authorization_report_exact_digest=authorization_report_exact_digest,
        feature_key_limit=feature_key_limit,
        unobserved_action_utility=unobserved_action_utility,
    )
    artifact = dict(artifact)
    artifact["artifact_policy"] = M3_CARRION_SURVIVOR_CONTINUATION_V186_ARTIFACT_POLICY
    artifact["built_from"] = dict(_mapping(artifact.get("built_from")))
    artifact["built_from"].update(
        {
            "source": "v185_repaired_transition_rows",
            "source_producer": source_producer,
            "dataset_digest": dataset_digest,
            "authorization_report_exact_digest": authorization_report_exact_digest,
            "authorization_route": authorization_route,
            "training_slice_index": TRAINING_SLICE_INDEX,
            "previous_training_slices_consumed": (
                PREVIOUS_TRAINING_SLICES_CONSUMED
            ),
            "campaign_slice_cap": CAMPAIGN_SLICE_CAP,
            "first_opt_in_training_slice": False,
            "slice_2_opt_in_training_slice": True,
            "runtime_action_selection_authorized": False,
            "promotion_authorized": False,
        }
    )
    tables = dict(_mapping(artifact.get("utility_tables")))
    tables["policy"] = "v186_mean_v185_repaired_transition_utility_by_public_feature_key_and_action_v1"
    artifact["utility_tables"] = tables
    loaded = load_transition_value_scorer_artifact(artifact)
    training = dict(training)
    training.update(
        {
            "policy": (
                "m3_carrion_survivor_continuation_v186_training_summary_v1"
            ),
            "training_slice_index": TRAINING_SLICE_INDEX,
            "previous_training_slices_consumed": (
                PREVIOUS_TRAINING_SLICES_CONSUMED
            ),
            "campaign_slice_cap": CAMPAIGN_SLICE_CAP,
            "source_producer": source_producer,
            "authorization_route": authorization_route,
            "first_opt_in_training_slice": False,
            "slice_2_opt_in_training_slice": True,
            "artifact_load_check": {
                "policy": "v186_transition_value_artifact_load_check_v1",
                "loaded": loaded.artifact == artifact,
                "schema_version": loaded.artifact.get("schema_version"),
                "artifact_policy": loaded.artifact.get("artifact_policy"),
            },
        }
    )
    return artifact, training


def run_v186_shadow_evaluation(
    *,
    artifact: Mapping[str, object],
    broad_seeds: Sequence[int] = DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = DEFAULT_TICKS,
) -> dict[str, object]:
    evaluation = dict(
        v180.run_v180_shadow_evaluation(
            artifact=artifact,
            broad_seeds=broad_seeds,
            carrion_fixture_seeds=carrion_fixture_seeds,
            ticks=ticks,
        )
    )
    evaluation.update(
        {
            "policy": (
                "m3_carrion_survivor_continuation_v186_shadow_evaluation_v1"
            ),
            "candidate_key": V186_CANDIDATE_KEY,
            "mode": "offline_shadow_controlled_eval",
            "runtime_integration": False,
            "promotion_evidence": False,
        }
    )
    return evaluation


def _source_validation(
    *,
    rows: Sequence[Mapping[str, object]],
    authorization_validation: Mapping[str, object],
    expected_dataset_digest: str,
) -> dict[str, object]:
    dataset_digest = stable_payload_digest([dict(row) for row in rows])
    checks = {
        "authorization_report_validation_passed": (
            authorization_validation.get("passed") is True
        ),
        "dataset_digest_matches_expected": dataset_digest == expected_dataset_digest,
        "dataset_digest_matches_authorization_report": (
            dataset_digest
            == str(authorization_validation.get("observed_dataset_digest") or "")
        ),
        "dataset_non_empty": bool(rows),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v186_source_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "dataset_digest": dataset_digest,
        "expected_dataset_digest": expected_dataset_digest,
        "row_count": len(rows),
        "required_source_producer": EXPECTED_V185_SOURCE_PRODUCER,
        "required_training_route": EXPECTED_V186_TRAINING_ROUTE,
    }


def _acceptance(evaluation: Mapping[str, object]) -> dict[str, object]:
    acceptance = dict(v180._acceptance(evaluation))
    acceptance.update(
        {
            "policy": "m3_carrion_survivor_continuation_v186_acceptance_v1",
            "training_slice_index": TRAINING_SLICE_INDEX,
            "max_dominant_requested_action_share": (
                MAX_DOMINANT_REQUESTED_ACTION_SHARE
            ),
            "shadow_acceptance_only": True,
            "runtime_integration_authorized": False,
            "promotion_evidence": False,
        }
    )
    return acceptance


def _classification(
    *,
    source_validation: Mapping[str, object],
    training: Mapping[str, object],
    evaluation: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v186_transition_row_policy_training_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if training.get("passed") is not True:
        return prefix + "training_artifact_invalid_closed_no_promotion"
    if evaluation.get("ran") is not True:
        return prefix + "slice_2_trained_shadow_eval_not_run_no_promotion"
    if acceptance.get("passed") is True:
        return prefix + "slice_2_shadow_acceptance_passed_no_promotion"
    return prefix + "slice_2_shadow_acceptance_failed_no_promotion"


def _contract(
    *,
    expected_authorization_report_exact_digest: str,
    expected_dataset_digest: str,
    expected_authorization_classification: str,
    expected_source_producer: str,
    expected_training_route: str,
) -> dict[str, object]:
    return {
        "explicit_opt_in_training_slice": True,
        "training_slice_index": TRAINING_SLICE_INDEX,
        "previous_training_slices_consumed": PREVIOUS_TRAINING_SLICES_CONSUMED,
        "campaign_slice_cap": CAMPAIGN_SLICE_CAP,
        "training_requires_v185_repaired_audit_authorization_report": True,
        "training_requires_v185_repaired_dataset_digest": True,
        "training_requires_v186_slice_2_route": True,
        "first_opt_in_training_slice": False,
        "input_dataset_digest_pinned": expected_dataset_digest,
        "authorization_report_exact_digest_pinned": (
            expected_authorization_report_exact_digest
        ),
        "authorization_classification_required": expected_authorization_classification,
        "source_producer_required": expected_source_producer,
        "training_route_required": expected_training_route,
        "runtime_integration_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "evaluation_scope": "offline_shadow_controlled_eval_only",
        "promotion_evidence": False,
    }


def _lifecycle_flags(
    *,
    source_authorized: bool,
    training_ran: bool,
    training_artifact_created: bool,
    shadow_eval_ran: bool,
    slice_2_training_consumed: bool,
) -> dict[str, object]:
    return {
        "training_authorized": bool(source_authorized),
        "training_ran": bool(training_ran),
        "training_artifact_created": bool(training_artifact_created),
        "fit_ran": bool(training_ran),
        "slice_2_training_consumed": bool(slice_2_training_consumed),
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": bool(shadow_eval_ran),
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_ran": False,
        "replay_viewer_schema_changed": False,
        "non_promoted": True,
    }


def _artifact_status(
    output_path: str | Path,
    *,
    created: bool,
    reason: str | None,
    payload: Mapping[str, object] | None = None,
    digest: str | None = None,
) -> dict[str, object]:
    status = {
        "created": bool(created),
        "path": str(output_path),
        "digest": digest,
        "schema_version": None,
        "artifact_policy": None,
        "reason": reason,
        "training_slice_index": TRAINING_SLICE_INDEX,
        "explicit_opt_in_required": True,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
    }
    if payload is not None:
        status.update(
            {
                "schema_version": payload.get("schema_version"),
                "artifact_policy": payload.get("artifact_policy"),
                "model_id": payload.get("model_id"),
                "training_row_count": _mapping(payload.get("built_from")).get(
                    "training_row_count"
                ),
                "feature_key_count": len(
                    _mapping(
                        _mapping(payload.get("utility_tables")).get(
                            "feature_action_utility"
                        )
                    )
                ),
            }
        )
    return status


def _skipped_training(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v186_training_summary_v1",
        "ran": False,
        "passed": False,
        "reason": reason,
        "training_slice_index": TRAINING_SLICE_INDEX,
    }


def _skipped_evaluation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v186_shadow_evaluation_v1",
        "ran": False,
        "reason": reason,
    }


def _closed_acceptance(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v186_acceptance_v1",
        "passed": False,
        "blocker_count": 1,
        "blockers": [v180._blocker(reason, observed=False, required=True)],
        "training_slice_index": TRAINING_SLICE_INDEX,
        "promotion_evidence": False,
    }


def _digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)
