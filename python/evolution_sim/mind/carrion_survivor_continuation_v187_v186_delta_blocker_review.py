from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v186_transition_row_policy_training as v186,
)
from evolution_sim.mind.candidate_campaign import (
    _float,
    _int,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V187_V186_DELTA_BLOCKER_REVIEW_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v187_v186_delta_blocker_review_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V187_V186_DELTA_BLOCKER_REVIEW_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v187_v186_delta_blocker_review_v1"
)

DEFAULT_V186_REPORT_PATH = v186.DEFAULT_OUTPUT_PATH
DEFAULT_V186_ARTIFACT_PATH = v186.DEFAULT_ARTIFACT_OUTPUT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v187-carrion-survivor-continuation-v186-delta-blocker-review.json"
)

EXPECTED_V186_REPORT_EXACT_DIGEST = (
    "f4f404b88093f00bc8e7d655ff6f1937c4e4786acdca6118275decb463193075"
)
EXPECTED_V186_ARTIFACT_DIGEST = (
    "729997cd3a3672dd3ceabf08ccb4a9b5a7ce67f0621ecdb73ff4216dd921b6ce"
)
EXPECTED_V186_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v186_transition_row_policy_training_"
    "slice_2_shadow_acceptance_failed_no_promotion"
)
EXPECTED_V186_BACKUP = (
    "gdrive:evolution-sim-backups/archives/"
    "20260612T151102Z-v186-transition-row-policy-training-slice-2.tar.zst"
)
EXPECTED_V186_BACKUP_SHA256 = (
    "9fba7700ec13b23f70f31b0269022c19b74bb7ed2967d2d70a9add05919eb31a"
)

EXPECTED_EMPTY_BROAD_ALIVE_BIRTH_REGRESSIONS = True
EXPECTED_DOMINANT_REQUESTED_ACTION_SHARE = 0.4179
EXPECTED_HEURISTIC_ACTION_SOURCE_COUNT = 0
EXPECTED_CARRION_ONLY_TERMINAL_SURVIVORS = 0
EXPECTED_CARRION_FIXTURE_BIRTHS_MEAN = 2.8333

RECOMMENDED_NEXT_ROUTE = (
    "v188_terminal_carrion_survival_support_generation_or_feasibility_proof_no_training"
)
STOP_ROUTE = "stop"


def run_carrion_survivor_continuation_v187_v186_delta_blocker_review(
    *,
    v186_report_path: str | Path = DEFAULT_V186_REPORT_PATH,
    v186_artifact_path: str | Path = DEFAULT_V186_ARTIFACT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v186_report_exact_digest: str = EXPECTED_V186_REPORT_EXACT_DIGEST,
    expected_v186_artifact_digest: str = EXPECTED_V186_ARTIFACT_DIGEST,
    expected_v186_classification: str = EXPECTED_V186_CLASSIFICATION,
) -> dict[str, object]:
    v186_report = load_json_report(v186_report_path)
    v186_artifact = load_json_report(v186_artifact_path)
    source_validation = validate_v187_v186_source_pins(
        v186_report,
        v186_artifact,
        expected_v186_report_exact_digest=expected_v186_report_exact_digest,
        expected_v186_artifact_digest=expected_v186_artifact_digest,
        expected_v186_classification=expected_v186_classification,
    )
    v186_delta = extract_v186_delta(v186_report)
    delta_validation = validate_v186_delta(v186_delta)
    inherited_prior_findings = _inherited_prior_findings()
    route_decision = _route_decision(
        source_validation=source_validation,
        delta_validation=delta_validation,
        v186_delta=v186_delta,
    )
    classification = _classification(
        source_validation=source_validation,
        delta_validation=delta_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V187_V186_DELTA_BLOCKER_REVIEW_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V187_V186_DELTA_BLOCKER_REVIEW_POLICY,
        "contract": _contract(
            expected_v186_report_exact_digest=expected_v186_report_exact_digest,
            expected_v186_artifact_digest=expected_v186_artifact_digest,
            expected_v186_classification=expected_v186_classification,
        ),
        "inputs": {
            "v186_report": str(v186_report_path),
            "v186_artifact": str(v186_artifact_path),
            "expected_v186_report_exact_digest": expected_v186_report_exact_digest,
            "expected_v186_artifact_digest": expected_v186_artifact_digest,
            "expected_v186_classification": expected_v186_classification,
            "v186_backup": EXPECTED_V186_BACKUP,
            "v186_backup_sha256": EXPECTED_V186_BACKUP_SHA256,
        },
        "source_validation": source_validation,
        "v186_delta": v186_delta,
        "delta_validation": delta_validation,
        "inherited_prior_findings": inherited_prior_findings,
        "blocker_review": _blocker_review(
            source_validation=source_validation,
            delta_validation=delta_validation,
            v186_delta=v186_delta,
        ),
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "v186_delta_only",
                "inherited_prior_findings_not_rediscovered",
                "no_slice_3_training",
                "no_runtime_integration",
                "no_promotion",
            ],
        },
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v187_v186_source_pins(
    v186_report: Mapping[str, object],
    v186_artifact: Mapping[str, object],
    *,
    expected_v186_report_exact_digest: str = EXPECTED_V186_REPORT_EXACT_DIGEST,
    expected_v186_artifact_digest: str = EXPECTED_V186_ARTIFACT_DIGEST,
    expected_v186_classification: str = EXPECTED_V186_CLASSIFICATION,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v186_report)
    report_artifact = _mapping(v186_report.get("artifact"))
    report_contract = _mapping(v186_report.get("contract"))
    report_training = _mapping(v186_report.get("training"))
    report_budget = _mapping(v186_report.get("training_slice_budget"))
    report_inputs = _mapping(v186_report.get("inputs"))
    report_source = _mapping(v186_report.get("source_validation"))
    report_classification = _mapping(v186_report.get("classification"))
    artifact_built_from = _mapping(v186_artifact.get("built_from"))
    artifact_digest = stable_payload_digest(v186_artifact)
    checks = {
        "v186_report_schema_matches": (
            v186_report.get("schema_version")
            == v186.M3_CARRION_SURVIVOR_CONTINUATION_V186_TRANSITION_ROW_POLICY_TRAINING_SCHEMA_VERSION
        ),
        "v186_report_policy_matches": (
            v186_report.get("policy")
            == v186.M3_CARRION_SURVIVOR_CONTINUATION_V186_TRANSITION_ROW_POLICY_TRAINING_POLICY
        ),
        "v186_report_exact_digest_valid": exact_validation.get("passed") is True,
        "v186_report_exact_digest_matches_expected": (
            str(v186_report.get("exact_digest") or "")
            == str(expected_v186_report_exact_digest)
        ),
        "v186_report_classification_matches_expected": (
            str(report_classification.get("primary") or "")
            == str(expected_v186_classification)
        ),
        "v186_report_training_ran": v186_report.get("training_ran") is True,
        "v186_report_training_artifact_created": (
            v186_report.get("training_artifact_created") is True
        ),
        "v186_report_slice_2_consumed": (
            v186_report.get("slice_2_training_consumed") is True
        ),
        "v186_report_training_slice_index_2": (
            _int(report_training.get("training_slice_index")) == 2
        ),
        "v186_report_first_slice_not_training": (
            report_training.get("first_opt_in_training_slice") is False
        ),
        "v186_report_slice_2_opt_in_training": (
            report_training.get("slice_2_opt_in_training_slice") is True
        ),
        "v186_report_training_route_matches": (
            str(report_training.get("authorization_route") or "")
            == v186.EXPECTED_V186_TRAINING_ROUTE
        ),
        "v186_report_source_producer_matches": (
            str(report_training.get("source_producer") or "")
            == v186.EXPECTED_V185_SOURCE_PRODUCER
        ),
        "v186_report_current_budget_is_2": (
            _int(report_budget.get("current_slices_consumed")) == 2
        ),
        "v186_report_artifact_digest_matches_expected": (
            str(report_artifact.get("digest") or "")
            == str(expected_v186_artifact_digest)
        ),
        "v186_artifact_digest_matches_expected": (
            artifact_digest == str(expected_v186_artifact_digest)
        ),
        "v186_report_artifact_digest_matches_artifact_file": (
            str(report_artifact.get("digest") or "") == artifact_digest
        ),
        "v186_artifact_policy_matches": (
            v186_artifact.get("artifact_policy")
            == v186.M3_CARRION_SURVIVOR_CONTINUATION_V186_ARTIFACT_POLICY
        ),
        "v186_artifact_training_slice_index_2": (
            _int(artifact_built_from.get("training_slice_index")) == 2
        ),
        "v186_artifact_first_slice_false": (
            artifact_built_from.get("first_opt_in_training_slice") is False
        ),
        "v186_artifact_slice_2_true": (
            artifact_built_from.get("slice_2_opt_in_training_slice") is True
        ),
        "v186_artifact_authorization_route_matches": (
            str(artifact_built_from.get("authorization_route") or "")
            == v186.EXPECTED_V186_TRAINING_ROUTE
        ),
        "v186_artifact_source_producer_matches": (
            str(artifact_built_from.get("source_producer") or "")
            == v186.EXPECTED_V185_SOURCE_PRODUCER
        ),
        "v186_report_source_validation_passed": (
            report_source.get("passed") is True
        ),
        "v186_report_input_route_matches": (
            str(report_inputs.get("expected_training_route") or "")
            == v186.EXPECTED_V186_TRAINING_ROUTE
        ),
        "v186_report_contract_route_matches": (
            str(report_contract.get("training_route_required") or "")
            == v186.EXPECTED_V186_TRAINING_ROUTE
        ),
        "v186_report_runtime_artifact_not_created": (
            v186_report.get("runtime_artifact_created") is False
        ),
        "v186_report_runtime_action_selection_unchanged": (
            v186_report.get("runtime_action_selection_changed") is False
        ),
        "v186_report_promotion_not_authorized": (
            v186_report.get("promotion_authorized") is False
        ),
        "v186_report_gate_relaxation_not_ran": (
            v186_report.get("gate_relaxation_ran") is False
        ),
        "v186_report_contract_gate_relaxation_not_allowed": (
            report_contract.get("gate_relaxation_allowed") is False
        ),
        "v186_artifact_runtime_action_selection_not_authorized": (
            v186_artifact.get("runtime_action_selection_authorized") is False
            and artifact_built_from.get("runtime_action_selection_authorized") is False
        ),
        "v186_artifact_promotion_not_authorized": (
            v186_artifact.get("promotion_authorized") is False
            and artifact_built_from.get("promotion_authorized") is False
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v187_v186_source_pin_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v186_report_exact_digest": expected_v186_report_exact_digest,
        "observed_v186_report_exact_digest": v186_report.get("exact_digest"),
        "v186_report_exact_digest_validation": exact_validation,
        "expected_v186_artifact_digest": expected_v186_artifact_digest,
        "observed_v186_artifact_digest": artifact_digest,
        "reported_v186_artifact_digest": report_artifact.get("digest"),
        "expected_v186_classification": expected_v186_classification,
        "observed_v186_classification": report_classification.get("primary"),
        "expected_training_route": v186.EXPECTED_V186_TRAINING_ROUTE,
        "observed_training_route": report_training.get("authorization_route"),
        "expected_source_producer": v186.EXPECTED_V185_SOURCE_PRODUCER,
        "observed_source_producer": report_training.get("source_producer"),
    }


def extract_v186_delta(v186_report: Mapping[str, object]) -> dict[str, object]:
    acceptance = _mapping(v186_report.get("acceptance"))
    controlled = _mapping(acceptance.get("controlled_fixture"))
    regressions = [
        dict(regression)
        for regression in acceptance.get("per_seed_alive_birth_regressions", [])
        if isinstance(regression, Mapping)
    ]
    terminal_survivors = _int(controlled.get("total_terminal_alive_agents"))
    dominant_share = _round(_float(acceptance.get("dominant_requested_action_share")))
    heuristic_count = _int(acceptance.get("heuristic_action_source_count"))
    births_mean = _round(_float(controlled.get("births_mean")))
    return {
        "policy": "m3_carrion_survivor_continuation_v187_v186_delta_extract_v1",
        "source": "pinned_v186_report_acceptance",
        "delta_scope": "narrow_v186_delta_against_inherited_prior_findings",
        "broad_alive_birth_regressions": regressions,
        "broad_alive_birth_regressions_empty": not regressions,
        "dominant_requested_action_share": dominant_share,
        "dominant_requested_action_share_lte_0_50": dominant_share <= 0.50,
        "heuristic_action_source_count": heuristic_count,
        "heuristic_action_source_count_zero": heuristic_count == 0,
        "carrion_only_terminal_survivors": terminal_survivors,
        "carrion_only_terminal_survivors_still_zero": terminal_survivors == 0,
        "carrion_fixture_births_mean": births_mean,
        "carrion_fixture_births_mean_positive": births_mean > 0.0,
        "acceptance_passed": acceptance.get("passed") is True,
        "acceptance_blockers": [
            dict(blocker)
            for blocker in acceptance.get("blockers", [])
            if isinstance(blocker, Mapping)
        ],
        "new_v186_evidence": {
            "broad_alive_birth_regressions_now_empty": not regressions,
            "dominant_requested_action_share_controlled": dominant_share <= 0.50,
            "heuristic_action_source_count_remains_zero": heuristic_count == 0,
            "carrion_only_terminal_survivors_remain_zero": terminal_survivors == 0,
            "carrion_fixture_births_without_terminal_survival": (
                births_mean > 0.0 and terminal_survivors == 0
            ),
        },
        "old_conclusions_marked_inherited_not_rediscovered": True,
    }


def validate_v186_delta(v186_delta: Mapping[str, object]) -> dict[str, object]:
    checks = {
        "broad_alive_birth_regressions_empty": (
            v186_delta.get("broad_alive_birth_regressions_empty")
            is EXPECTED_EMPTY_BROAD_ALIVE_BIRTH_REGRESSIONS
        ),
        "dominant_requested_action_share_matches_expected": _float_matches(
            v186_delta.get("dominant_requested_action_share"),
            EXPECTED_DOMINANT_REQUESTED_ACTION_SHARE,
        ),
        "dominant_requested_action_share_lte_0_50": (
            v186_delta.get("dominant_requested_action_share_lte_0_50") is True
        ),
        "heuristic_action_source_count_matches_expected": (
            _int(v186_delta.get("heuristic_action_source_count"))
            == EXPECTED_HEURISTIC_ACTION_SOURCE_COUNT
        ),
        "heuristic_action_source_count_zero": (
            v186_delta.get("heuristic_action_source_count_zero") is True
        ),
        "carrion_only_terminal_survivors_matches_expected": (
            _int(v186_delta.get("carrion_only_terminal_survivors"))
            == EXPECTED_CARRION_ONLY_TERMINAL_SURVIVORS
        ),
        "carrion_only_terminal_survivors_still_zero": (
            v186_delta.get("carrion_only_terminal_survivors_still_zero") is True
        ),
        "carrion_fixture_births_mean_matches_expected": _float_matches(
            v186_delta.get("carrion_fixture_births_mean"),
            EXPECTED_CARRION_FIXTURE_BIRTHS_MEAN,
        ),
        "carrion_fixture_births_mean_positive": (
            v186_delta.get("carrion_fixture_births_mean_positive") is True
        ),
        "acceptance_still_failed": v186_delta.get("acceptance_passed") is False,
        "old_conclusions_not_rediscovered": (
            v186_delta.get("old_conclusions_marked_inherited_not_rediscovered")
            is True
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v187_v186_delta_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected": {
            "broad_alive_birth_regressions_empty": (
                EXPECTED_EMPTY_BROAD_ALIVE_BIRTH_REGRESSIONS
            ),
            "dominant_requested_action_share": EXPECTED_DOMINANT_REQUESTED_ACTION_SHARE,
            "heuristic_action_source_count": EXPECTED_HEURISTIC_ACTION_SOURCE_COUNT,
            "carrion_only_terminal_survivors": EXPECTED_CARRION_ONLY_TERMINAL_SURVIVORS,
            "carrion_fixture_births_mean": EXPECTED_CARRION_FIXTURE_BIRTHS_MEAN,
        },
    }


def _route_decision(
    *,
    source_validation: Mapping[str, object],
    delta_validation: Mapping[str, object],
    v186_delta: Mapping[str, object],
) -> dict[str, object]:
    source_passed = source_validation.get("passed") is True
    delta_passed = delta_validation.get("passed") is True
    selected_route = RECOMMENDED_NEXT_ROUTE if source_passed and delta_passed else STOP_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v187_route_decision_v1",
        "selected_route": selected_route,
        "recommended_next_route": selected_route,
        "exactly_one_next_route_recommended": True,
        "classification": (
            "terminal_survival_support_generation_before_slice_3"
            if selected_route == RECOMMENDED_NEXT_ROUTE
            else "closed_due_to_invalid_source_or_unexpected_delta"
        ),
        "rationale": _route_rationale(
            source_validation=source_validation,
            delta_validation=delta_validation,
            v186_delta=v186_delta,
        ),
        "alternatives": [
            {
                "route": RECOMMENDED_NEXT_ROUTE,
                "selected": selected_route == RECOMMENDED_NEXT_ROUTE,
                "reason": (
                    "v186 removed the broad alive/birth regression blocker and "
                    "kept action-source gates controlled, but still produced zero "
                    "carrion_only@120 terminal survivors. The narrow next question "
                    "is whether positive terminal-survival support can be generated "
                    "or proven infeasible before spending slice 3."
                ),
            },
            {
                "route": "v188_objective_utility_redesign_before_slice_3_no_training",
                "selected": False,
                "reason": (
                    "v90/v91 objective mismatch remains inherited context, but v186 "
                    "does not add a new objective-specific delta after broad "
                    "regressions disappeared and births stayed positive."
                ),
            },
            {
                "route": "v188_architecture_branch_no_training",
                "selected": False,
                "reason": (
                    "v186 adds no architecture-specific evidence; terminal survival "
                    "support feasibility is the narrower blocker."
                ),
            },
            {
                "route": STOP_ROUTE,
                "selected": selected_route == STOP_ROUTE,
                "reason": (
                    "Selected only if v186 pins fail or the expected v186 delta is "
                    "not present."
                ),
            },
        ],
        "slice_3_training_allowed": False,
        "support_generation_ran": False,
        "support_generation_requires_separate_explicit_task": True,
        "runtime_integration_allowed": False,
        "promotion_authorized": False,
    }


def _route_rationale(
    *,
    source_validation: Mapping[str, object],
    delta_validation: Mapping[str, object],
    v186_delta: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True:
        return "v186 report/artifact pins did not validate; stop without routing."
    if delta_validation.get("passed") is not True:
        return "v186 delta did not match the pinned blocker-review contract; stop."
    if (
        v186_delta.get("broad_alive_birth_regressions_empty") is True
        and v186_delta.get("dominant_requested_action_share_lte_0_50") is True
        and v186_delta.get("heuristic_action_source_count_zero") is True
        and v186_delta.get("carrion_only_terminal_survivors_still_zero") is True
    ):
        return (
            "The new v186 delta removes broad regression and action-collapse "
            "blockers while preserving the terminal-survival failure. Route to a "
            "no-training terminal-survival support generation or feasibility proof "
            "before considering slice 3."
        )
    return "No selected route; delta conditions were incomplete."


def _blocker_review(
    *,
    source_validation: Mapping[str, object],
    delta_validation: Mapping[str, object],
    v186_delta: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    if source_validation.get("passed") is not True:
        blockers.append(
            {
                "reason": "v186_source_pins_invalid",
                "observed": source_validation.get("failures"),
                "required": "pinned v186 report and artifact digests plus contracts validate",
            }
        )
    if delta_validation.get("passed") is not True:
        blockers.append(
            {
                "reason": "v186_delta_unexpected",
                "observed": delta_validation.get("failures"),
                "required": "pinned v186 acceptance delta matches v187 contract",
            }
        )
    if source_validation.get("passed") is True and delta_validation.get("passed") is True:
        blockers.append(
            {
                "reason": "terminal_survival_support_missing_before_slice_3",
                "observed": {
                    "carrion_only_terminal_survivors": v186_delta.get(
                        "carrion_only_terminal_survivors"
                    ),
                    "carrion_fixture_births_mean": v186_delta.get(
                        "carrion_fixture_births_mean"
                    ),
                    "broad_alive_birth_regressions_empty": v186_delta.get(
                        "broad_alive_birth_regressions_empty"
                    ),
                    "dominant_requested_action_share": v186_delta.get(
                        "dominant_requested_action_share"
                    ),
                    "heuristic_action_source_count": v186_delta.get(
                        "heuristic_action_source_count"
                    ),
                },
                "required": "positive terminal survival support or infeasibility proof before slice 3",
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v187_blocker_review_v1",
        "scope": "delta_only_not_generic_carrion_autopsy",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "old_conclusions_inherited_not_rediscovered": True,
        "v180_rerun": False,
        "v186_rerun": False,
        "support_expansion_ran": False,
    }


def _inherited_prior_findings() -> list[dict[str, object]]:
    return [
        {
            "finding_id": "v31",
            "summary": "post-carrion low-gain eat / movement-to-energy-death loops",
            "status": "inherited_not_rediscovered",
            "rerun": False,
            "used_as": "historical_context_boundary",
        },
        {
            "finding_id": "v36",
            "summary": "missing positive carrion_only@120 terminal survival support",
            "status": "inherited_not_rediscovered",
            "rerun": False,
            "used_as": "historical_context_boundary",
        },
        {
            "finding_id": "v90_v91",
            "summary": "objective mismatch at frontier states",
            "status": "inherited_not_rediscovered",
            "rerun": False,
            "used_as": "historical_context_boundary",
        },
        {
            "finding_id": "v181_v182",
            "summary": (
                "v180 sparse/imputed support, broad eat overrides, and carrion no "
                "complete-support coverage"
            ),
            "status": "inherited_not_rediscovered",
            "rerun": False,
            "used_as": "historical_context_boundary",
        },
    ]


def _classification(
    *,
    source_validation: Mapping[str, object],
    delta_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v187_v186_delta_blocker_review_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if delta_validation.get("passed") is not True:
        return prefix + "unexpected_delta_closed_no_training"
    if route_decision.get("selected_route") == RECOMMENDED_NEXT_ROUTE:
        return prefix + "terminal_survival_support_generation_before_slice_3_no_training"
    return prefix + "closed_no_training"


def _contract(
    *,
    expected_v186_report_exact_digest: str,
    expected_v186_artifact_digest: str,
    expected_v186_classification: str,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "delta_only": True,
        "generic_carrion_autopsy_rerun_allowed": False,
        "v180_rerun_allowed": False,
        "v186_rerun_allowed": False,
        "support_expansion_ran": False,
        "support_expansion_requires_separate_explicit_task": True,
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "input_v186_report_exact_digest_pinned": expected_v186_report_exact_digest,
        "input_v186_artifact_digest_pinned": expected_v186_artifact_digest,
        "input_v186_classification_required": expected_v186_classification,
        "expected_new_v186_delta": {
            "broad_alive_birth_regressions_empty": (
                EXPECTED_EMPTY_BROAD_ALIVE_BIRTH_REGRESSIONS
            ),
            "dominant_requested_action_share": EXPECTED_DOMINANT_REQUESTED_ACTION_SHARE,
            "heuristic_action_source_count": EXPECTED_HEURISTIC_ACTION_SOURCE_COUNT,
            "carrion_only_terminal_survivors": EXPECTED_CARRION_ONLY_TERMINAL_SURVIVORS,
            "carrion_fixture_births_mean": EXPECTED_CARRION_FIXTURE_BIRTHS_MEAN,
        },
        "recommended_route_if_facts_hold": RECOMMENDED_NEXT_ROUTE,
        "promotion_evidence": False,
    }


def _lifecycle_flags() -> dict[str, object]:
    return {
        "training_ran": False,
        "training_artifact_created": False,
        "fit_ran": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": False,
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "gate_relaxation_ran": False,
        "support_expansion_ran": False,
        "generic_carrion_autopsy_rerun": False,
        "v180_rerun": False,
        "v186_rerun": False,
        "non_promoted": True,
    }


def _float_matches(observed: object, expected: float) -> bool:
    return abs(_float(observed) - float(expected)) <= 1e-6


def _digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)
