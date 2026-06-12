from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v178_transition_row_dataset_audit as v178,
)
from evolution_sim.mind.candidate_campaign import _mapping, write_json
from evolution_sim.mind.carrion_survivor_continuation_v183_exact_transition_support_expansion import (
    DEFAULT_EXPANDED_TRANSITION_DATASET_OUTPUT_PATH as DEFAULT_V183_TRANSITION_DATASET_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V183_REPORT_PATH,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V184_V183_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V184_V183_TRANSITION_ROW_DATASET_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit_v1"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v184-carrion-survivor-continuation-v183-transition-row-dataset-audit.json"
)
V184_AUTHORIZED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit_"
    "valid_support_ready_slice_2_training_route_authorized"
)
V184_SOURCE_INVALID_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit_"
    "source_invalid_closed_no_training"
)
V184_DATASET_CONTRACT_INVALID_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit_"
    "dataset_contract_invalid_closed_no_training"
)
V184_SUPPORT_LIMITED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit_"
    "valid_support_limited_closed_no_training"
)
V184_RECHECK_REQUIRED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit_"
    "valid_support_ready_default_threshold_recheck_required_no_training"
)
V184_DIGEST_PINS_REQUIRED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit_"
    "valid_support_ready_digest_pins_required_no_training"
)


def run_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit(
    *,
    v183_report_path: str | Path = DEFAULT_V183_REPORT_PATH,
    transition_dataset_path: str | Path = DEFAULT_V183_TRANSITION_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v183_report_exact_digest: str | None = None,
    expected_dataset_digest: str | None = None,
    min_row_count: int = v178.DEFAULT_MIN_ROW_COUNT,
    min_seed_count: int = v178.DEFAULT_MIN_SEED_COUNT,
    min_branch_count: int = v178.DEFAULT_MIN_BRANCH_COUNT,
    min_forced_action_count: int = v178.DEFAULT_MIN_FORCED_ACTION_COUNT,
) -> dict[str, object]:
    expected_source_digest = (
        expected_v183_report_exact_digest or v178.EXPECTED_V183_REPORT_EXACT_DIGEST
    )
    expected_row_digest = expected_dataset_digest or v178.EXPECTED_V183_DATASET_DIGEST
    audit = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
        transition_dataset_path=transition_dataset_path,
        v177_report_path=v183_report_path,
        output_path=output_path,
        expected_v177_report_exact_digest=expected_source_digest,
        expected_dataset_digest=expected_row_digest,
        min_row_count=min_row_count,
        min_seed_count=min_seed_count,
        min_branch_count=min_branch_count,
        min_forced_action_count=min_forced_action_count,
    )
    report = _v184_report(
        audit,
        v183_report_path=v183_report_path,
        transition_dataset_path=transition_dataset_path,
        expected_v183_report_exact_digest=expected_source_digest,
        expected_dataset_digest=expected_row_digest,
    )
    write_json(output_path, report)
    return report


def _v184_report(
    audit: Mapping[str, object],
    *,
    v183_report_path: str | Path,
    transition_dataset_path: str | Path,
    expected_v183_report_exact_digest: str,
    expected_dataset_digest: str,
) -> dict[str, object]:
    report = dict(audit)
    source = dict(_mapping(report.get("source_validation")))
    authorization = dict(_mapping(report.get("training_authorization")))
    source_producer = str(source.get("source_producer") or "")
    if source_producer != v178.V183_SOURCE_PRODUCER:
        source["passed"] = False
        failures = [str(item) for item in source.get("failures") or []]
        failures.append("v184_expected_v183_source_producer")
        source["failures"] = sorted(set(failures))
        source["failure_count"] = len(source["failures"])
        authorization["authorized"] = False
        authorization["training_authorized"] = False
        authorization["transition_row_training_authorized"] = False
        authorization["next_same_lane_opt_in_training_slice_authorized"] = False
        auth_failures = [str(item) for item in authorization.get("failures") or []]
        auth_failures.append("v184_expected_v183_source_producer")
        authorization["failures"] = sorted(set(auth_failures))
        authorization["failure_count"] = len(authorization["failures"])
    report["source_validation"] = source
    report["training_authorization"] = authorization
    route = _v184_route_recommendation(
        _mapping(audit.get("route_recommendation")),
        source_validation=source,
        training_authorization=authorization,
    )
    classification = _v184_classification(
        v178_classification=str(
            _mapping(audit.get("classification")).get("primary") or ""
        ),
        route=route,
        source_validation=source,
    )
    report["schema_version"] = (
        M3_CARRION_SURVIVOR_CONTINUATION_V184_V183_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION
    )
    report["policy"] = (
        M3_CARRION_SURVIVOR_CONTINUATION_V184_V183_TRANSITION_ROW_DATASET_AUDIT_POLICY
    )
    report["contract"] = _v184_contract(route)
    inputs = dict(_mapping(report.get("inputs")))
    inputs.update(
        {
            "source_report": str(v183_report_path),
            "v183_report": str(v183_report_path),
            "transition_dataset": str(transition_dataset_path),
            "expected_source_report_exact_digest": expected_v183_report_exact_digest,
            "expected_v183_report_exact_digest": expected_v183_report_exact_digest,
            "expected_dataset_digest": expected_dataset_digest,
            "canonical_v183_report_exact_digest": (
                v178.EXPECTED_V183_REPORT_EXACT_DIGEST
            ),
            "canonical_v183_dataset_digest": v178.EXPECTED_V183_DATASET_DIGEST,
        }
    )
    report["inputs"] = inputs
    report["classification"] = {
        "primary": classification,
        "labels": [classification],
        "v178_style_audit_classification": _mapping(audit.get("classification")).get(
            "primary"
        ),
    }
    report["route_recommendation"] = route
    report["v178_style_audit_summary"] = {
        "schema_version": audit.get("schema_version"),
        "policy": audit.get("policy"),
        "classification": _mapping(audit.get("classification")).get("primary"),
        "exact_digest": audit.get("exact_digest"),
        "source_producer": source.get("source_producer"),
    }
    report["training_authorized"] = False
    report["training_artifact_created"] = False
    report["slice_2_training_consumed"] = False
    report["runtime_artifact_created"] = False
    report["runtime_action_selection_changed"] = False
    report["promotion_authorized"] = False
    report["exact_digest"] = _json_round_trip_digest(report)
    return report


def _v184_route_recommendation(
    v178_route: Mapping[str, object],
    *,
    source_validation: Mapping[str, object],
    training_authorization: Mapping[str, object],
) -> dict[str, object]:
    source_is_v183 = source_validation.get("source_producer") == v178.V183_SOURCE_PRODUCER
    authorized = (
        source_is_v183
        and source_validation.get("passed") is True
        and training_authorization.get("authorized") is True
        and v178_route.get("transition_row_training_authorized") is True
    )
    route = dict(v178_route)
    if authorized:
        next_route = v178.V185_SLICE_2_TRAINING_ROUTE
    elif not source_is_v183:
        next_route = "rerun_v184_with_canonical_v183_source_report_and_dataset"
    else:
        next_route = str(v178_route.get("recommended_next_route") or "closed")
    route.update(
        {
            "policy": (
                "m3_carrion_survivor_continuation_v184_route_recommendation_v1"
            ),
            "recommended_next_route": next_route,
            "source_producer": source_validation.get("source_producer"),
            "transition_row_training_authorized": authorized,
            "training_authorized": authorized,
            "next_same_lane_opt_in_training_slice_authorized": authorized,
            "first_opt_in_training_slice_authorized": False,
            "slice_2_opt_in_training_route_authorized": authorized,
            "slice_2_training_authorized": authorized,
            "training_authorization_scope": (
                "next_same_lane_opt_in_slice_2_training" if authorized else "closed"
            ),
            "runtime_integration_authorized": False,
            "shadow_or_live_eval_authorized": False,
            "promotion_authorized": False,
        }
    )
    return route


def _v184_contract(route: Mapping[str, object]) -> dict[str, object]:
    authorized = route.get("slice_2_opt_in_training_route_authorized") is True
    return {
        "diagnostics_only": True,
        "training_allowed": False,
        "training_authorized_for_this_command": False,
        "next_same_lane_opt_in_training_slice_authorized": authorized,
        "first_opt_in_training_slice_authorized": False,
        "slice_2_opt_in_training_route_authorized": authorized,
        "slice_2_training_consumed": False,
        "training_route_if_explicitly_requested": (
            v178.V185_SLICE_2_TRAINING_ROUTE if authorized else None
        ),
        "fit_allowed": False,
        "runtime_artifact_allowed": False,
        "runtime_action_change_allowed": False,
        "shadow_or_live_eval_allowed": False,
        "promotion_allowed": False,
        "gate_relaxation_allowed": False,
        "source_report_must_be_canonical_v183_exact_transition_support_expansion": True,
        "canonical_v183_report_exact_digest": v178.EXPECTED_V183_REPORT_EXACT_DIGEST,
        "canonical_v183_dataset_digest": v178.EXPECTED_V183_DATASET_DIGEST,
    }


def _v184_classification(
    *,
    v178_classification: str,
    route: Mapping[str, object],
    source_validation: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True:
        return V184_SOURCE_INVALID_CLASSIFICATION
    if v178_classification.endswith("dataset_contract_invalid_closed_no_training"):
        return V184_DATASET_CONTRACT_INVALID_CLASSIFICATION
    if v178_classification.endswith("valid_support_limited_expand_before_training"):
        return V184_SUPPORT_LIMITED_CLASSIFICATION
    if v178_classification.endswith(
        "valid_support_ready_default_threshold_recheck_required_no_training"
    ):
        return V184_RECHECK_REQUIRED_CLASSIFICATION
    if v178_classification.endswith("valid_support_ready_digest_pins_required_no_training"):
        return V184_DIGEST_PINS_REQUIRED_CLASSIFICATION
    if route.get("slice_2_opt_in_training_route_authorized") is True:
        return V184_AUTHORIZED_CLASSIFICATION
    return V184_SOURCE_INVALID_CLASSIFICATION


def _json_round_trip_digest(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    )
