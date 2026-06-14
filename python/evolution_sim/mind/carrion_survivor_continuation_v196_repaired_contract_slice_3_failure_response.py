from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.mind import (
    carrion_survivor_continuation_v186_transition_row_policy_training as v186,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training as v195,
)
from evolution_sim.mind.candidate_campaign import _float, _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.evaluation_harness import (
    _fixture_world,
    _mind_v3_policy,
    _run_world,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.transition_value_scorer import (
    load_transition_value_scorer_artifact,
    transition_value_source_key_category,
)

M3_CARRION_SURVIVOR_CONTINUATION_V196_REPAIRED_CONTRACT_SLICE_3_FAILURE_RESPONSE_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V196_REPAIRED_CONTRACT_SLICE_3_FAILURE_RESPONSE_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response_v1"
)

DEFAULT_V195_REPORT_PATH = v195.DEFAULT_OUTPUT_PATH
DEFAULT_V195_ARTIFACT_PATH = v195.DEFAULT_ARTIFACT_OUTPUT_PATH
DEFAULT_V186_REPORT_PATH = v186.DEFAULT_OUTPUT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v196-carrion-survivor-continuation-repaired-contract-slice-3-failure-response.json"
)
DEFAULT_BACKUP_DOC_PATHS = (
    Path("docs/mind-v3-autonomous-evolution.md"),
    Path("AGENTS.md"),
)

EXPECTED_V195_REPORT_EXACT_DIGEST = (
    "0d69540a3817b7b7c10f19b440bc63d9651b26a5c9d4d199f55b48253e2edcde"
)
EXPECTED_V195_ARTIFACT_DIGEST = (
    "9cc3df4b5d87af5dd4dfb0e79f7282debbcc9f98ba7393f42a45d360e46c4470"
)
EXPECTED_V194_REPORT_EXACT_DIGEST = v195.EXPECTED_V194_REPORT_EXACT_DIGEST
EXPECTED_V194_COMPACT_DATASET_DIGEST = v195.EXPECTED_V194_COMPACT_DATASET_DIGEST
EXPECTED_V195_ROUTE = v195.FAILURE_ROUTE
EXPECTED_V195_AUTHORIZATION_ROUTE = v195.EXPECTED_V194_ROUTE
EXPECTED_V195_BACKUP = (
    "gdrive:evolution-sim-backups/archives/"
    "20260613T185630Z-v195-repaired-contract-terminal-survival-support-training-slice-3.tar.zst"
)
EXPECTED_V195_BACKUP_SHA256 = (
    "f4d3785619e3937949654cd586d1896fdc20d89fa3d9f3f14d191748ca776049"
)
EXPECTED_V195_RCLONE_DIFFERENCES = 0
EXPECTED_V195_RCLONE_MATCHING_FILES = 1

EXPECTED_CARRION_TERMINAL_SURVIVORS = 0
EXPECTED_DOMINANT_REQUESTED_ACTION_SHARE = 0.9422
EXPECTED_HEURISTIC_ACTION_SOURCE_COUNT = 0
EXPECTED_TRAINING_DOMINANT_ACTION = "stay"
EXPECTED_TRAINING_DOMINANT_ACTION_SHARE = 0.207558
EXPECTED_BROAD_REGRESSIONS = {
    5: {"alive_agents_delta": -6, "births_delta": -5},
    13: {"alive_agents_delta": -6, "births_delta": -3},
    19: {"alive_agents_delta": -10, "births_delta": -10},
    29: {"alive_agents_delta": -9, "births_delta": -5},
    37: {"alive_agents_delta": -10, "births_delta": -8},
    41: {"alive_agents_delta": -7, "births_delta": -6},
}

COVERAGE_OR_ABSTENTION_REPAIR_ROUTE = (
    "v197_slice_4_design_requires_coverage_or_abstention_repair_no_training"
)
DATASET_COVERAGE_DESIGN_ROUTE = "v197_repaired_contract_dataset_coverage_design_no_training"
INSTRUMENTATION_ROUTE = "v197_slice_3_failure_instrumentation_no_training"
STOP_ROUTE = "stop_v196_source_pins_invalid_no_training"

PRIMARY_COVERAGE_COLLAPSE_MECHANISM = (
    "artifact_low_specificity_feature_coverage_collapse_to_eat_with_miss_abstention"
)
PRIMARY_DATASET_INSUFFICIENCY_MECHANISM = (
    "repaired_contract_support_rows_insufficient_for_runtime_distribution"
)
PRIMARY_INCONCLUSIVE_MECHANISM = "slice_3_failure_mechanism_inconclusive"


def run_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response(
    *,
    v195_report_path: str | Path = DEFAULT_V195_REPORT_PATH,
    v195_artifact_path: str | Path = DEFAULT_V195_ARTIFACT_PATH,
    v186_report_path: str | Path = DEFAULT_V186_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v195_report_exact_digest: str = EXPECTED_V195_REPORT_EXACT_DIGEST,
    expected_v195_artifact_digest: str = EXPECTED_V195_ARTIFACT_DIGEST,
    expected_v194_report_exact_digest: str = EXPECTED_V194_REPORT_EXACT_DIGEST,
    expected_v194_dataset_digest: str = EXPECTED_V194_COMPACT_DATASET_DIGEST,
    expected_v195_route: str = EXPECTED_V195_ROUTE,
    backup_doc_paths: Sequence[str | Path] = DEFAULT_BACKUP_DOC_PATHS,
    backup_metadata_override: Mapping[str, object] | None = None,
    run_diagnostic_replay: bool = True,
    lookup_diagnostics_override: Mapping[str, object] | None = None,
    broad_seeds: Sequence[int] = v195.DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = v195.DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = v195.DEFAULT_TICKS,
) -> dict[str, object]:
    v195_report = load_json_report(v195_report_path)
    v195_artifact = load_json_report(v195_artifact_path)
    v186_report = _load_optional_json_report(v186_report_path)
    source_validation = validate_v196_source_pins(
        v195_report=v195_report,
        v195_artifact=v195_artifact,
        expected_v195_report_exact_digest=expected_v195_report_exact_digest,
        expected_v195_artifact_digest=expected_v195_artifact_digest,
        expected_v194_report_exact_digest=expected_v194_report_exact_digest,
        expected_v194_dataset_digest=expected_v194_dataset_digest,
        expected_v195_route=expected_v195_route,
        backup_doc_paths=backup_doc_paths,
        backup_metadata_override=backup_metadata_override,
    )
    v195_failure_facts = extract_v195_failure_facts(v195_report)
    failure_fact_validation = validate_v195_failure_facts(v195_failure_facts)
    action_distribution_comparison = compare_training_and_shadow_action_distribution(
        v195_report
    )
    artifact_inspection = inspect_v195_artifact(v195_artifact)
    v186_comparison = compare_v186_to_v195(
        v195_report=v195_report,
        v186_report=v186_report,
    )
    if lookup_diagnostics_override is not None:
        lookup_diagnostics = dict(lookup_diagnostics_override)
    elif source_validation.get("passed") is True and run_diagnostic_replay:
        lookup_diagnostics = run_v195_lookup_coverage_diagnostic_replay(
            artifact=v195_artifact,
            broad_seeds=broad_seeds,
            carrion_fixture_seeds=carrion_fixture_seeds,
            ticks=ticks,
        )
    else:
        lookup_diagnostics = skipped_lookup_diagnostics(
            "source_validation_failed"
            if source_validation.get("passed") is not True
            else "diagnostic_replay_skipped"
        )
    mechanism = classify_v196_failure_mechanism(
        source_validation=source_validation,
        failure_fact_validation=failure_fact_validation,
        action_distribution_comparison=action_distribution_comparison,
        artifact_inspection=artifact_inspection,
        lookup_diagnostics=lookup_diagnostics,
        v186_comparison=v186_comparison,
    )
    route_decision = route_decision_for_v196(
        source_validation=source_validation,
        failure_fact_validation=failure_fact_validation,
        mechanism=mechanism,
    )
    classification = classification_for_v196(
        source_validation=source_validation,
        failure_fact_validation=failure_fact_validation,
        mechanism=mechanism,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V196_REPAIRED_CONTRACT_SLICE_3_FAILURE_RESPONSE_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V196_REPAIRED_CONTRACT_SLICE_3_FAILURE_RESPONSE_POLICY
        ),
        "contract": contract(
            expected_v195_report_exact_digest=expected_v195_report_exact_digest,
            expected_v195_artifact_digest=expected_v195_artifact_digest,
            expected_v194_report_exact_digest=expected_v194_report_exact_digest,
            expected_v194_dataset_digest=expected_v194_dataset_digest,
            expected_v195_route=expected_v195_route,
        ),
        "inputs": {
            "v195_report": str(v195_report_path),
            "v195_artifact": str(v195_artifact_path),
            "v186_report": str(v186_report_path),
            "expected_v195_report_exact_digest": expected_v195_report_exact_digest,
            "expected_v195_artifact_digest": expected_v195_artifact_digest,
            "expected_v194_report_exact_digest": expected_v194_report_exact_digest,
            "expected_v194_dataset_digest": expected_v194_dataset_digest,
            "expected_v195_route": expected_v195_route,
            "run_diagnostic_replay": bool(run_diagnostic_replay),
            "broad_seeds": [int(seed) for seed in broad_seeds],
            "carrion_fixture_seeds": [int(seed) for seed in carrion_fixture_seeds],
            "ticks": int(ticks),
        },
        "source_pin_validation": source_validation,
        "artifact_digest_validation": artifact_digest_validation(
            v195_report=v195_report,
            v195_artifact=v195_artifact,
            expected_v195_artifact_digest=expected_v195_artifact_digest,
        ),
        "v195_failure_facts": v195_failure_facts,
        "failure_fact_validation": failure_fact_validation,
        "action_distribution_comparison": action_distribution_comparison,
        "artifact_inspection": artifact_inspection,
        "lookup_coverage_diagnostics": lookup_diagnostics,
        "v186_comparison": v186_comparison,
        "mechanism_analysis": mechanism,
        "per_seed_regression_summary": v195_failure_facts[
            "broad_alive_birth_regressions"
        ],
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "v195_failure_response",
                "no_training",
                "no_slice_4_consumption",
                "no_runtime_integration",
                "no_gate_relaxation",
                "no_promotion",
            ],
        },
        **lifecycle_flags(
            diagnostic_replay_ran=lookup_diagnostics.get("ran") is True,
        ),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v196_source_pins(
    *,
    v195_report: Mapping[str, object],
    v195_artifact: Mapping[str, object],
    expected_v195_report_exact_digest: str,
    expected_v195_artifact_digest: str,
    expected_v194_report_exact_digest: str,
    expected_v194_dataset_digest: str,
    expected_v195_route: str,
    backup_doc_paths: Sequence[str | Path],
    backup_metadata_override: Mapping[str, object] | None,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v195_report)
    artifact_digest = stable_payload_digest(v195_artifact)
    report_artifact = _mapping(v195_report.get("artifact"))
    auth = _mapping(v195_report.get("authorization_report_validation"))
    dataset = _mapping(v195_report.get("dataset"))
    training = _mapping(v195_report.get("training"))
    budget = _mapping(v195_report.get("training_slice_budget"))
    route = _mapping(v195_report.get("route_decision"))
    built_from = _mapping(v195_artifact.get("built_from"))
    backup = (
        dict(backup_metadata_override)
        if backup_metadata_override is not None
        else backup_metadata_audit(backup_doc_paths)
    )
    checks = {
        "v195_report_schema_matches": (
            v195_report.get("schema_version")
            == v195.M3_CARRION_SURVIVOR_CONTINUATION_V195_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_TRAINING_SCHEMA_VERSION
        ),
        "v195_report_policy_matches": (
            v195_report.get("policy")
            == v195.M3_CARRION_SURVIVOR_CONTINUATION_V195_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_TRAINING_POLICY
        ),
        "v195_report_exact_digest_valid": exact_validation.get("passed") is True,
        "v195_report_exact_digest_matches_expected": (
            v195_report.get("exact_digest") == expected_v195_report_exact_digest
        ),
        "v195_report_artifact_digest_matches_expected": (
            report_artifact.get("digest") == expected_v195_artifact_digest
        ),
        "v195_artifact_digest_matches_expected": (
            artifact_digest == expected_v195_artifact_digest
        ),
        "v195_report_artifact_digest_matches_file": (
            report_artifact.get("digest") == artifact_digest
        ),
        "v195_artifact_policy_matches": (
            v195_artifact.get("artifact_policy")
            == v195.M3_CARRION_SURVIVOR_CONTINUATION_V195_ARTIFACT_POLICY
        ),
        "v195_artifact_runtime_action_selection_not_authorized": (
            v195_artifact.get("runtime_action_selection_authorized") is False
            and built_from.get("runtime_action_selection_authorized") is False
        ),
        "v195_artifact_promotion_not_authorized": (
            v195_artifact.get("promotion_authorized") is False
            and built_from.get("promotion_authorized") is False
        ),
        "v195_artifact_slice_3_built_from": (
            _int(built_from.get("training_slice_index")) == 3
            and built_from.get("slice_3_opt_in_training_slice") is True
        ),
        "v195_artifact_dataset_digest_matches_expected": (
            built_from.get("dataset_digest") == expected_v194_dataset_digest
        ),
        "v195_artifact_authorization_report_digest_matches_expected": (
            built_from.get("authorization_report_exact_digest")
            == expected_v194_report_exact_digest
        ),
        "v195_artifact_authorization_route_matches_expected": (
            built_from.get("authorization_route") == EXPECTED_V195_AUTHORIZATION_ROUTE
        ),
        "v194_report_digest_pin_matches": (
            auth.get("observed_exact_digest") == expected_v194_report_exact_digest
        ),
        "v194_dataset_digest_pin_matches": (
            auth.get("observed_dataset_digest") == expected_v194_dataset_digest
            and dataset.get("dataset_digest") == expected_v194_dataset_digest
        ),
        "v194_training_route_pin_matches": (
            auth.get("observed_training_route") == EXPECTED_V195_AUTHORIZATION_ROUTE
        ),
        "v195_training_ran": v195_report.get("training_ran") is True,
        "v195_training_artifact_created": (
            v195_report.get("training_artifact_created") is True
        ),
        "v195_slice_3_consumed": (
            v195_report.get("slice_3_training_consumed") is True
        ),
        "v195_training_slice_index_3": (
            _int(training.get("training_slice_index")) == 3
        ),
        "v195_current_slice_budget_3": (
            _int(budget.get("current_slices_consumed")) == 3
        ),
        "v195_route_matches_required_failure_response": (
            route.get("recommended_next_route") == expected_v195_route
        ),
        "v195_shadow_eval_ran": v195_report.get("shadow_eval_ran") is True,
        "v195_runtime_artifact_not_created": (
            v195_report.get("runtime_artifact_created") is False
        ),
        "v195_runtime_action_selection_unchanged": (
            v195_report.get("runtime_action_selection_changed") is False
        ),
        "v195_promotion_not_authorized": (
            v195_report.get("promotion_authorized") is False
        ),
        "v195_gate_relaxation_not_allowed": (
            v195_report.get("gate_relaxation_allowed") is False
        ),
        "v195_backup_metadata_validated": backup.get("passed") is True,
    }
    try:
        loaded = load_transition_value_scorer_artifact(v195_artifact)
        load_check = {"loaded": loaded.artifact == dict(v195_artifact)}
    except (TypeError, ValueError) as exc:
        load_check = {"loaded": False, "error": str(exc)}
    checks["v195_artifact_loads_as_transition_value_scorer"] = (
        load_check.get("loaded") is True
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v196_source_pin_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v195_report_exact_digest": expected_v195_report_exact_digest,
        "observed_v195_report_exact_digest": v195_report.get("exact_digest"),
        "v195_report_exact_digest_validation": exact_validation,
        "expected_v195_artifact_digest": expected_v195_artifact_digest,
        "observed_v195_artifact_digest": artifact_digest,
        "reported_v195_artifact_digest": report_artifact.get("digest"),
        "expected_v194_report_exact_digest": expected_v194_report_exact_digest,
        "observed_v194_report_exact_digest": auth.get("observed_exact_digest"),
        "expected_v194_dataset_digest": expected_v194_dataset_digest,
        "observed_v194_dataset_digest": auth.get("observed_dataset_digest"),
        "expected_v195_route": expected_v195_route,
        "observed_v195_route": route.get("recommended_next_route"),
        "v195_backup_metadata": backup,
        "artifact_load_check": load_check,
        "checks": checks,
    }


def artifact_digest_validation(
    *,
    v195_report: Mapping[str, object],
    v195_artifact: Mapping[str, object],
    expected_v195_artifact_digest: str,
) -> dict[str, object]:
    digest = stable_payload_digest(v195_artifact)
    reported = _mapping(v195_report.get("artifact")).get("digest")
    checks = {
        "artifact_digest_matches_expected": digest == expected_v195_artifact_digest,
        "artifact_digest_matches_v195_report": digest == reported,
        "artifact_runtime_action_selection_not_authorized": (
            v195_artifact.get("runtime_action_selection_authorized") is False
        ),
        "artifact_promotion_not_authorized": (
            v195_artifact.get("promotion_authorized") is False
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v196_artifact_digest_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_artifact_digest": expected_v195_artifact_digest,
        "observed_artifact_digest": digest,
        "reported_artifact_digest": reported,
        "checks": checks,
    }


def extract_v195_failure_facts(v195_report: Mapping[str, object]) -> dict[str, object]:
    acceptance = _mapping(v195_report.get("acceptance"))
    controlled = _mapping(acceptance.get("controlled_fixture"))
    training = _mapping(v195_report.get("training"))
    regressions = []
    for item in acceptance.get("per_seed_alive_birth_regressions", []):
        if isinstance(item, Mapping) and item.get("suite") == "broad":
            regressions.append(
                {
                    "seed": _int(item.get("seed")),
                    "alive_agents_delta": _int(item.get("alive_agents_delta")),
                    "births_delta": _int(item.get("births_delta")),
                    "baseline_alive_agents": _int(item.get("baseline_alive_agents")),
                    "candidate_alive_agents": _int(item.get("candidate_alive_agents")),
                    "baseline_births": _int(item.get("baseline_births")),
                    "candidate_births": _int(item.get("candidate_births")),
                }
            )
    regressions.sort(key=lambda item: int(item["seed"]))
    return {
        "policy": "m3_carrion_survivor_continuation_v196_v195_failure_fact_extract_v1",
        "source": "pinned_v195_report_acceptance_and_training",
        "v195_training_slice_consumed": (
            v195_report.get("slice_3_training_consumed") is True
        ),
        "campaign_slice_budget_current": _int(
            _mapping(v195_report.get("training_slice_budget")).get(
                "current_slices_consumed"
            )
        ),
        "acceptance_passed": acceptance.get("passed") is True,
        "carrion_terminal_survivors": _int(
            controlled.get("total_terminal_alive_agents")
        ),
        "dominant_requested_action_share": _round(
            _float(acceptance.get("dominant_requested_action_share"))
        ),
        "heuristic_action_source_count": _int(
            acceptance.get("heuristic_action_source_count")
        ),
        "broad_alive_birth_regressions": regressions,
        "training_dominant_action": training.get("dominant_training_action"),
        "training_dominant_action_share": _round(
            _float(training.get("dominant_training_action_share"))
        ),
        "training_action_support_counts": _int_counter(
            training.get("action_support_counts")
        ),
    }


def validate_v195_failure_facts(facts: Mapping[str, object]) -> dict[str, object]:
    expected_regressions = [
        {"seed": seed, **values}
        for seed, values in sorted(EXPECTED_BROAD_REGRESSIONS.items())
    ]
    observed_regressions = [
        {
            "seed": _int(item.get("seed")),
            "alive_agents_delta": _int(item.get("alive_agents_delta")),
            "births_delta": _int(item.get("births_delta")),
        }
        for item in facts.get("broad_alive_birth_regressions", [])
        if isinstance(item, Mapping)
    ]
    checks = {
        "v195_consumed_slice_3": facts.get("v195_training_slice_consumed") is True,
        "campaign_budget_current_3": (
            _int(facts.get("campaign_slice_budget_current")) == 3
        ),
        "acceptance_failed": facts.get("acceptance_passed") is False,
        "carrion_terminal_survivors_zero": (
            _int(facts.get("carrion_terminal_survivors"))
            == EXPECTED_CARRION_TERMINAL_SURVIVORS
        ),
        "dominant_requested_action_share_matches_expected": _float_matches(
            facts.get("dominant_requested_action_share"),
            EXPECTED_DOMINANT_REQUESTED_ACTION_SHARE,
        ),
        "heuristic_action_source_count_zero": (
            _int(facts.get("heuristic_action_source_count"))
            == EXPECTED_HEURISTIC_ACTION_SOURCE_COUNT
        ),
        "broad_regressions_match_expected": (
            observed_regressions == expected_regressions
        ),
        "training_dominant_action_matches_expected": (
            facts.get("training_dominant_action") == EXPECTED_TRAINING_DOMINANT_ACTION
        ),
        "training_dominant_share_matches_expected": _float_matches(
            facts.get("training_dominant_action_share"),
            EXPECTED_TRAINING_DOMINANT_ACTION_SHARE,
            digits=6,
        ),
        "training_share_not_label_imbalance_explanation_alone": (
            _float(facts.get("training_dominant_action_share")) < 0.50
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v196_v195_failure_fact_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected": {
            "carrion_terminal_survivors": EXPECTED_CARRION_TERMINAL_SURVIVORS,
            "dominant_requested_action_share": EXPECTED_DOMINANT_REQUESTED_ACTION_SHARE,
            "heuristic_action_source_count": EXPECTED_HEURISTIC_ACTION_SOURCE_COUNT,
            "broad_regressions": expected_regressions,
            "training_dominant_action": EXPECTED_TRAINING_DOMINANT_ACTION,
            "training_dominant_action_share": EXPECTED_TRAINING_DOMINANT_ACTION_SHARE,
        },
        "observed_broad_regressions": observed_regressions,
        "checks": checks,
    }


def compare_training_and_shadow_action_distribution(
    v195_report: Mapping[str, object],
) -> dict[str, object]:
    training = _mapping(v195_report.get("training"))
    evaluation = _mapping(v195_report.get("evaluation"))
    broad_candidate = _mapping(_mapping(_mapping(evaluation.get("broad")).get("candidate")).get("aggregate"))
    fixture_candidate = _mapping(
        _mapping(
            _mapping(evaluation.get("controlled_fixture")).get("candidate")
        ).get("aggregate")
    )
    training_counts = _int_counter(training.get("action_support_counts"))
    broad_counts = _int_counter(broad_candidate.get("requested_action_counts"))
    fixture_counts = _int_counter(fixture_candidate.get("requested_action_counts"))
    combined_shadow = Counter(broad_counts)
    combined_shadow.update(fixture_counts)
    training_dominant = _dominant(training_counts)
    broad_dominant = _dominant(broad_counts)
    fixture_dominant = _dominant(fixture_counts)
    combined_dominant = _dominant(combined_shadow)
    return {
        "policy": "m3_carrion_survivor_continuation_v196_training_vs_shadow_action_distribution_v1",
        "training": {
            "action_counts": dict(sorted(training_counts.items())),
            "dominant_action": training_dominant["action"],
            "dominant_action_share": training_dominant["share"],
            "row_count": sum(training_counts.values()),
        },
        "shadow_broad": {
            "action_counts": dict(sorted(broad_counts.items())),
            "dominant_action": broad_dominant["action"],
            "dominant_action_share": broad_dominant["share"],
            "decision_count": sum(broad_counts.values()),
        },
        "shadow_carrion_only": {
            "action_counts": dict(sorted(fixture_counts.items())),
            "dominant_action": fixture_dominant["action"],
            "dominant_action_share": fixture_dominant["share"],
            "decision_count": sum(fixture_counts.values()),
        },
        "shadow_combined": {
            "action_counts": dict(sorted(combined_shadow.items())),
            "dominant_action": combined_dominant["action"],
            "dominant_action_share": combined_dominant["share"],
            "decision_count": sum(combined_shadow.values()),
        },
        "label_imbalance_alone_ruled_out": (
            training_dominant["share"] < 0.50
            and fixture_dominant["share"] > 0.90
            and fixture_dominant["action"] != training_dominant["action"]
        ),
        "training_to_fixture_dominant_share_delta": _round(
            fixture_dominant["share"] - training_dominant["share"]
        ),
    }


def inspect_v195_artifact(artifact: Mapping[str, object]) -> dict[str, object]:
    table = _mapping(_mapping(artifact.get("utility_tables")).get("feature_action_utility"))
    key_categories: Counter[str] = Counter()
    best_action_counts: Counter[str] = Counter()
    best_action_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    action_count_per_key: Counter[int] = Counter()
    for key, value in table.items():
        category = _source_key_category("feature_action_utility", str(key))
        key_categories.update([category])
        action_stats = _mapping(value)
        action_count_per_key.update([len(action_stats)])
        best_action = _best_action_for_stats(action_stats)
        if best_action:
            best_action_counts.update([best_action])
            best_action_by_category[category].update([best_action])
    dominant_best = _dominant(best_action_counts)
    return {
        "policy": "m3_carrion_survivor_continuation_v196_v195_artifact_inspection_v1",
        "artifact_digest": stable_payload_digest(artifact),
        "feature_key_count": len(table),
        "feature_key_category_counts": dict(sorted(key_categories.items())),
        "action_count_per_feature_key": {
            str(key): int(value) for key, value in sorted(action_count_per_key.items())
        },
        "best_utility_action_counts": dict(sorted(best_action_counts.items())),
        "dominant_best_utility_action": dominant_best["action"],
        "dominant_best_utility_action_share": dominant_best["share"],
        "best_utility_action_counts_by_feature_key_category": {
            category: dict(sorted(counter.items()))
            for category, counter in sorted(best_action_by_category.items())
        },
        "global_feature_key_present": "global" in table,
        "global_feature_key_excluded": _mapping(artifact.get("built_from")).get(
            "global_feature_key_excluded"
        )
        is True,
    }


def run_v195_lookup_coverage_diagnostic_replay(
    *,
    artifact: Mapping[str, object],
    broad_seeds: Sequence[int],
    carrion_fixture_seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    scorer = load_transition_value_scorer_artifact(artifact)
    broad = _run_lookup_scope(
        scope="broad",
        scorer=scorer,
        seeds=broad_seeds,
        ticks=ticks,
    )
    carrion = _run_lookup_scope(
        scope="carrion_only",
        scorer=scorer,
        seeds=carrion_fixture_seeds,
        ticks=ticks,
    )
    combined = combine_lookup_scopes(broad, carrion)
    return {
        "policy": "m3_carrion_survivor_continuation_v196_lookup_coverage_diagnostic_replay_v1",
        "ran": True,
        "diagnostics_only": True,
        "training_rerun": False,
        "slice_4_training_consumed": False,
        "broad": broad,
        "carrion_only": carrion,
        "combined": combined,
    }


def _run_lookup_scope(
    *,
    scope: str,
    scorer: object,
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    stats = _empty_lookup_stats()
    per_seed: list[dict[str, object]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        policy = _mind_v3_policy(
            seed=seed,
            founder_template=None,
            transition_value_scorer=scorer,
            transition_value_action_override=True,
            transition_value_action_override_source_integrity_passed=True,
        )
        if scope == "broad":
            world = SimulationWorld(
                WorldConfig(seed=seed, max_ticks=int(ticks)),
                policy=policy,
            )
        elif scope == "carrion_only":
            world = _fixture_world(
                fixture_name="carrion_only",
                seed=seed,
                ticks=int(ticks),
                policy=policy,
            )
        else:
            raise ValueError(f"unsupported lookup diagnostic scope: {scope}")
        run = _run_world(
            world=world,
            seed=seed,
            ticks=int(ticks),
            trajectory_output_path=None,
            trajectory_split_id=f"v196_{scope}",
        )
        seed_stats = _lookup_stats_from_decisions(world.policy_decision_diagnostics_records)
        _merge_lookup_stats(stats, seed_stats)
        per_seed.append(
            {
                "seed": seed,
                "alive_agents": run.get("alive_agents"),
                "births": run.get("births"),
                "requested_action_counts": run.get("requested_action_counts"),
                "dominant_requested_action": run.get("dominant_requested_action"),
                "dominant_requested_action_share": run.get(
                    "dominant_requested_action_share"
                ),
                "lookup": _finalize_lookup_stats(seed_stats),
            }
        )
    aggregate = _finalize_lookup_stats(stats)
    return {
        "policy": "m3_carrion_survivor_continuation_v196_lookup_scope_diagnostics_v1",
        "scope": scope,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "per_seed": per_seed,
        **aggregate,
    }


def _lookup_stats_from_decisions(
    decision_diagnostics: Sequence[Mapping[str, object] | None],
) -> dict[str, object]:
    stats = _empty_lookup_stats()
    for diagnostic in decision_diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        transition = _mapping(diagnostic.get("transition_value_scorer"))
        if not transition:
            continue
        _update_lookup_stats(stats, transition)
    return stats


def combine_lookup_scopes(*scopes: Mapping[str, object]) -> dict[str, object]:
    stats = _empty_lookup_stats()
    for scope in scopes:
        _merge_lookup_payload(stats, scope)
    return {
        "policy": "m3_carrion_survivor_continuation_v196_combined_lookup_diagnostics_v1",
        **_finalize_lookup_stats(stats),
    }


def _empty_lookup_stats() -> dict[str, object]:
    return {
        "decision_count": 0,
        "supported_score_count": 0,
        "observed_support_floor_satisfied_count": 0,
        "clear_best_count": 0,
        "override_applied_count": 0,
        "runtime_action_selection_changed_count": 0,
        "missing_supported_score_count": 0,
        "low_observed_support_count": 0,
        "no_prediction_count": 0,
        "imputed_valid_action_decision_count": 0,
        "score_source_counts": Counter(),
        "source_key_category_counts": Counter(),
        "predicted_action_counts": Counter(),
        "applied_override_action_counts": Counter(),
        "final_requested_action_counts": Counter(),
        "original_mind_v3_requested_action_counts": Counter(),
        "override_rejected_reason_counts": Counter(),
        "final_requested_action_counts_by_rejected_reason": defaultdict(Counter),
        "final_requested_action_counts_by_source_key_category": defaultdict(Counter),
    }


def _update_lookup_stats(stats: dict[str, object], transition: Mapping[str, object]) -> None:
    stats["decision_count"] = int(stats["decision_count"]) + 1
    score_source = str(transition.get("score_source") or "")
    source_key_category = _source_key_category(score_source, transition.get("source_key"))
    _counter(stats, "score_source_counts").update([score_source or "unknown"])
    _counter(stats, "source_key_category_counts").update([source_key_category])
    final_requested = str(transition.get("final_requested_action") or "")
    original_requested = str(transition.get("original_mind_v3_requested_action") or "")
    predicted_action = str(transition.get("predicted_action") or "")
    if final_requested:
        _counter(stats, "final_requested_action_counts").update([final_requested])
        _nested_counter(
            stats,
            "final_requested_action_counts_by_source_key_category",
            source_key_category,
        ).update([final_requested])
    if original_requested:
        _counter(stats, "original_mind_v3_requested_action_counts").update(
            [original_requested]
        )
    if predicted_action:
        _counter(stats, "predicted_action_counts").update([predicted_action])
    else:
        stats["no_prediction_count"] = int(stats["no_prediction_count"]) + 1
    if transition.get("supported_scores_for_all_valid_actions") is True:
        stats["supported_score_count"] = int(stats["supported_score_count"]) + 1
    else:
        stats["missing_supported_score_count"] = (
            int(stats["missing_supported_score_count"]) + 1
        )
    if (
        transition.get("observed_support_floor_satisfied_for_all_valid_actions")
        is True
    ):
        stats["observed_support_floor_satisfied_count"] = (
            int(stats["observed_support_floor_satisfied_count"]) + 1
        )
    if transition.get("clear_best_valid_action") is True:
        stats["clear_best_count"] = int(stats["clear_best_count"]) + 1
    if transition.get("has_imputed_valid_action_score") is True:
        stats["imputed_valid_action_decision_count"] = (
            int(stats["imputed_valid_action_decision_count"]) + 1
        )
    low_support = _int(transition.get("low_observed_support_valid_action_score_count"))
    if low_support > 0:
        stats["low_observed_support_count"] = (
            int(stats["low_observed_support_count"]) + 1
        )
    if transition.get("override_applied") is True:
        stats["override_applied_count"] = int(stats["override_applied_count"]) + 1
        if predicted_action:
            _counter(stats, "applied_override_action_counts").update([predicted_action])
    else:
        reason = str(transition.get("override_rejected_reason") or "none")
        _counter(stats, "override_rejected_reason_counts").update([reason])
        if final_requested:
            _nested_counter(
                stats,
                "final_requested_action_counts_by_rejected_reason",
                reason,
            ).update([final_requested])
    if transition.get("runtime_action_selection_changed") is True:
        stats["runtime_action_selection_changed_count"] = (
            int(stats["runtime_action_selection_changed_count"]) + 1
        )


def _merge_lookup_stats(target: dict[str, object], source: Mapping[str, object]) -> None:
    for key in (
        "decision_count",
        "supported_score_count",
        "observed_support_floor_satisfied_count",
        "clear_best_count",
        "override_applied_count",
        "runtime_action_selection_changed_count",
        "missing_supported_score_count",
        "low_observed_support_count",
        "no_prediction_count",
        "imputed_valid_action_decision_count",
    ):
        target[key] = int(target[key]) + int(source.get(key, 0))
    for key in (
        "score_source_counts",
        "source_key_category_counts",
        "predicted_action_counts",
        "applied_override_action_counts",
        "final_requested_action_counts",
        "original_mind_v3_requested_action_counts",
        "override_rejected_reason_counts",
    ):
        _counter(target, key).update(_int_counter(source.get(key)))
    for key in (
        "final_requested_action_counts_by_rejected_reason",
        "final_requested_action_counts_by_source_key_category",
    ):
        nested = source.get(key)
        if isinstance(nested, Mapping):
            for nested_key, counts in nested.items():
                _nested_counter(target, key, str(nested_key)).update(
                    _int_counter(counts)
                )


def _merge_lookup_payload(target: dict[str, object], source: Mapping[str, object]) -> None:
    payload = {
        "decision_count": source.get("decision_count"),
        "supported_score_count": source.get("supported_score_count"),
        "observed_support_floor_satisfied_count": source.get(
            "observed_support_floor_satisfied_count"
        ),
        "clear_best_count": source.get("clear_best_count"),
        "override_applied_count": source.get("override_applied_count"),
        "runtime_action_selection_changed_count": source.get(
            "runtime_action_selection_changed_count"
        ),
        "missing_supported_score_count": source.get("missing_supported_score_count"),
        "low_observed_support_count": source.get("low_observed_support_count"),
        "no_prediction_count": source.get("no_prediction_count"),
        "imputed_valid_action_decision_count": source.get(
            "imputed_valid_action_decision_count"
        ),
        "score_source_counts": source.get("score_source_counts"),
        "source_key_category_counts": source.get("source_key_category_counts"),
        "predicted_action_counts": source.get("predicted_action_counts"),
        "applied_override_action_counts": source.get("applied_override_action_counts"),
        "final_requested_action_counts": source.get("final_requested_action_counts"),
        "original_mind_v3_requested_action_counts": source.get(
            "original_mind_v3_requested_action_counts"
        ),
        "override_rejected_reason_counts": source.get(
            "override_rejected_reason_counts"
        ),
        "final_requested_action_counts_by_rejected_reason": source.get(
            "final_requested_action_counts_by_rejected_reason"
        ),
        "final_requested_action_counts_by_source_key_category": source.get(
            "final_requested_action_counts_by_source_key_category"
        ),
    }
    _merge_lookup_stats(target, payload)


def _finalize_lookup_stats(stats: Mapping[str, object]) -> dict[str, object]:
    decision_count = _int(stats.get("decision_count"))
    supported_count = _int(stats.get("supported_score_count"))
    floor_count = _int(stats.get("observed_support_floor_satisfied_count"))
    applied_count = _int(stats.get("override_applied_count"))
    changed_count = _int(stats.get("runtime_action_selection_changed_count"))
    missing_count = _int(stats.get("missing_supported_score_count"))
    low_support_count = _int(stats.get("low_observed_support_count"))
    source_category_counts = _int_counter(stats.get("source_key_category_counts"))
    applied_counts = _int_counter(stats.get("applied_override_action_counts"))
    final_counts = _int_counter(stats.get("final_requested_action_counts"))
    predicted_counts = _int_counter(stats.get("predicted_action_counts"))
    missing_final_counts = _int_counter(
        _mapping(stats.get("final_requested_action_counts_by_rejected_reason")).get(
            "missing_supported_scores_for_valid_actions"
        )
    )
    low_specificity_count = sum(
        int(source_category_counts.get(category, 0))
        for category in (
            "mask_only_hit",
            "coarse_feature_hit",
            "self_feature_hit",
            "self_coarse_feature_hit",
        )
    )
    exact_hit_count = int(source_category_counts.get("exact_context_feature_hit", 0))
    miss_dominant = _dominant(missing_final_counts)
    applied_dominant = _dominant(applied_counts)
    final_dominant = _dominant(final_counts)
    predicted_dominant = _dominant(predicted_counts)
    return {
        "decision_count": decision_count,
        "supported_score_count": supported_count,
        "supported_score_share": _share(supported_count, decision_count),
        "observed_support_floor_satisfied_count": floor_count,
        "observed_support_floor_satisfied_share": _share(floor_count, decision_count),
        "clear_best_count": _int(stats.get("clear_best_count")),
        "clear_best_share": _share(_int(stats.get("clear_best_count")), decision_count),
        "override_applied_count": applied_count,
        "override_applied_share": _share(applied_count, decision_count),
        "runtime_action_selection_changed_count": changed_count,
        "runtime_action_selection_changed_share": _share(changed_count, decision_count),
        "missing_supported_score_count": missing_count,
        "missing_supported_score_share": _share(missing_count, decision_count),
        "low_observed_support_count": low_support_count,
        "low_observed_support_share": _share(low_support_count, decision_count),
        "no_prediction_count": _int(stats.get("no_prediction_count")),
        "imputed_valid_action_decision_count": _int(
            stats.get("imputed_valid_action_decision_count")
        ),
        "exact_feature_hit_count": exact_hit_count,
        "exact_feature_hit_share": _share(exact_hit_count, decision_count),
        "low_specificity_feature_hit_count": low_specificity_count,
        "low_specificity_feature_hit_share": _share(
            low_specificity_count,
            decision_count,
        ),
        "mask_only_feature_hit_count": int(source_category_counts.get("mask_only_hit", 0)),
        "mask_only_feature_hit_share": _share(
            int(source_category_counts.get("mask_only_hit", 0)),
            decision_count,
        ),
        "score_source_counts": dict(sorted(_int_counter(stats.get("score_source_counts")).items())),
        "source_key_category_counts": dict(sorted(source_category_counts.items())),
        "predicted_action_counts": dict(sorted(predicted_counts.items())),
        "dominant_predicted_action": predicted_dominant["action"],
        "dominant_predicted_action_share": predicted_dominant["share"],
        "applied_override_action_counts": dict(sorted(applied_counts.items())),
        "dominant_applied_override_action": applied_dominant["action"],
        "dominant_applied_override_action_share": applied_dominant["share"],
        "final_requested_action_counts": dict(sorted(final_counts.items())),
        "dominant_final_requested_action": final_dominant["action"],
        "dominant_final_requested_action_share": final_dominant["share"],
        "original_mind_v3_requested_action_counts": dict(
            sorted(_int_counter(stats.get("original_mind_v3_requested_action_counts")).items())
        ),
        "override_rejected_reason_counts": dict(
            sorted(_int_counter(stats.get("override_rejected_reason_counts")).items())
        ),
        "final_requested_action_counts_by_rejected_reason": {
            key: dict(sorted(_int_counter(value).items()))
            for key, value in sorted(
                _mapping(stats.get("final_requested_action_counts_by_rejected_reason")).items()
            )
        },
        "final_requested_action_counts_by_source_key_category": {
            key: dict(sorted(_int_counter(value).items()))
            for key, value in sorted(
                _mapping(stats.get("final_requested_action_counts_by_source_key_category")).items()
            )
        },
        "missing_state_final_requested_action_counts": dict(
            sorted(missing_final_counts.items())
        ),
        "missing_state_dominant_final_requested_action": miss_dominant["action"],
        "missing_state_dominant_final_requested_action_share": miss_dominant["share"],
        "missing_states_defaulted_to_stay": (
            miss_dominant["action"] == "stay" and miss_dominant["share"] > 0.50
        ),
    }


def skipped_lookup_diagnostics(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v196_lookup_coverage_diagnostic_replay_v1",
        "ran": False,
        "reason": reason,
        "training_rerun": False,
        "slice_4_training_consumed": False,
    }


def compare_v186_to_v195(
    *,
    v195_report: Mapping[str, object],
    v186_report: Mapping[str, object] | None,
) -> dict[str, object]:
    if v186_report is None:
        return {
            "policy": "m3_carrion_survivor_continuation_v196_v186_comparison_v1",
            "available": False,
            "reason": "v186_report_missing",
        }
    v186_exact = exact_digest_validation_report(v186_report)
    v186_acceptance = _mapping(v186_report.get("acceptance"))
    v195_acceptance = _mapping(v195_report.get("acceptance"))
    v186_regressions = [
        item
        for item in v186_acceptance.get("per_seed_alive_birth_regressions", [])
        if isinstance(item, Mapping) and item.get("suite") == "broad"
    ]
    v195_regressions = [
        item
        for item in v195_acceptance.get("per_seed_alive_birth_regressions", [])
        if isinstance(item, Mapping) and item.get("suite") == "broad"
    ]
    v186_share = _round(_float(v186_acceptance.get("dominant_requested_action_share")))
    v195_share = _round(_float(v195_acceptance.get("dominant_requested_action_share")))
    return {
        "policy": "m3_carrion_survivor_continuation_v196_v186_comparison_v1",
        "available": True,
        "v186_report_exact_digest_valid": v186_exact.get("passed") is True,
        "v186_report_exact_digest": v186_report.get("exact_digest"),
        "v186_dominant_requested_action_share": v186_share,
        "v195_dominant_requested_action_share": v195_share,
        "dominant_requested_action_share_delta_v195_minus_v186": _round(
            v195_share - v186_share
        ),
        "v186_broad_regression_count": len(v186_regressions),
        "v195_broad_regression_count": len(v195_regressions),
        "broad_regression_count_delta_v195_minus_v186": (
            len(v195_regressions) - len(v186_regressions)
        ),
        "v186_carrion_terminal_survivors": _int(
            _mapping(v186_acceptance.get("controlled_fixture")).get(
                "total_terminal_alive_agents"
            )
        ),
        "v195_carrion_terminal_survivors": _int(
            _mapping(v195_acceptance.get("controlled_fixture")).get(
                "total_terminal_alive_agents"
            )
        ),
        "v186_had_zero_survivors_without_v195_action_collapse": (
            _int(
                _mapping(v186_acceptance.get("controlled_fixture")).get(
                    "total_terminal_alive_agents"
                )
            )
            == 0
            and v186_share <= 0.50
            and not v186_regressions
        ),
        "v195_failure_profile_is_new_action_collapse_plus_broad_regression": (
            v195_share > 0.90 and bool(v195_regressions)
        ),
    }


def classify_v196_failure_mechanism(
    *,
    source_validation: Mapping[str, object],
    failure_fact_validation: Mapping[str, object],
    action_distribution_comparison: Mapping[str, object],
    artifact_inspection: Mapping[str, object],
    lookup_diagnostics: Mapping[str, object],
    v186_comparison: Mapping[str, object],
) -> dict[str, object]:
    combined = _mapping(lookup_diagnostics.get("combined"))
    carrion = _mapping(lookup_diagnostics.get("carrion_only"))
    broad = _mapping(lookup_diagnostics.get("broad"))
    ran = lookup_diagnostics.get("ran") is True
    source_ok = (
        source_validation.get("passed") is True
        and failure_fact_validation.get("passed") is True
    )
    applied_eat_collapse = (
        combined.get("dominant_applied_override_action") == "eat"
        and _float(combined.get("dominant_applied_override_action_share")) > 0.85
        and _float(combined.get("override_applied_share")) > 0.50
    )
    low_specificity_dominates = (
        _float(combined.get("low_specificity_feature_hit_share")) > 0.70
        and _float(carrion.get("mask_only_feature_hit_share")) > 0.90
    )
    missing_abstains = (
        combined.get("missing_states_defaulted_to_stay") is False
        and _int(combined.get("missing_supported_score_count")) > 0
    )
    label_imbalance_ruled_out = (
        action_distribution_comparison.get("label_imbalance_alone_ruled_out") is True
    )
    v186_delta_supports_new_mechanism = (
        v186_comparison.get("v186_had_zero_survivors_without_v195_action_collapse")
        is True
        and v186_comparison.get(
            "v195_failure_profile_is_new_action_collapse_plus_broad_regression"
        )
        is True
    )
    if (
        source_ok
        and ran
        and applied_eat_collapse
        and low_specificity_dominates
        and missing_abstains
    ):
        primary = PRIMARY_COVERAGE_COLLAPSE_MECHANISM
        confirmed = True
        route_class = "artifact_fallback_or_coverage_collapse"
    elif source_ok and ran and label_imbalance_ruled_out:
        primary = PRIMARY_DATASET_INSUFFICIENCY_MECHANISM
        confirmed = True
        route_class = "dataset_insufficiency"
    else:
        primary = PRIMARY_INCONCLUSIVE_MECHANISM
        confirmed = False
        route_class = "inconclusive"
    return {
        "policy": "m3_carrion_survivor_continuation_v196_failure_mechanism_analysis_v1",
        "primary_mechanism": primary,
        "confirmed": confirmed,
        "route_class": route_class,
        "summary": _mechanism_summary(primary),
        "evidence": {
            "lookup_diagnostic_replay_ran": ran,
            "label_imbalance_alone_ruled_out": label_imbalance_ruled_out,
            "training_dominant_action": _mapping(
                action_distribution_comparison.get("training")
            ).get("dominant_action"),
            "training_dominant_action_share": _mapping(
                action_distribution_comparison.get("training")
            ).get("dominant_action_share"),
            "shadow_carrion_dominant_action": _mapping(
                action_distribution_comparison.get("shadow_carrion_only")
            ).get("dominant_action"),
            "shadow_carrion_dominant_action_share": _mapping(
                action_distribution_comparison.get("shadow_carrion_only")
            ).get("dominant_action_share"),
            "combined_override_applied_share": combined.get("override_applied_share"),
            "combined_dominant_applied_override_action": combined.get(
                "dominant_applied_override_action"
            ),
            "combined_dominant_applied_override_action_share": combined.get(
                "dominant_applied_override_action_share"
            ),
            "combined_low_specificity_feature_hit_share": combined.get(
                "low_specificity_feature_hit_share"
            ),
            "broad_mask_only_feature_hit_share": broad.get(
                "mask_only_feature_hit_share"
            ),
            "carrion_mask_only_feature_hit_share": carrion.get(
                "mask_only_feature_hit_share"
            ),
            "combined_exact_feature_hit_share": combined.get("exact_feature_hit_share"),
            "missing_states_defaulted_to_stay": combined.get(
                "missing_states_defaulted_to_stay"
            ),
            "missing_state_dominant_final_requested_action": combined.get(
                "missing_state_dominant_final_requested_action"
            ),
            "v186_delta_supports_new_mechanism": v186_delta_supports_new_mechanism,
            "artifact_global_feature_key_present": artifact_inspection.get(
                "global_feature_key_present"
            ),
            "artifact_feature_key_count": artifact_inspection.get("feature_key_count"),
        },
        "ruled_out": {
            "label_imbalance_alone": label_imbalance_ruled_out,
            "default_action_on_miss_to_stay": missing_abstains,
            "imputed_score_collapse": (
                _int(combined.get("imputed_valid_action_decision_count")) == 0
            ),
            "v186_same_failure_profile": v186_delta_supports_new_mechanism,
        },
        "contributing_mechanisms": [
            "high_specificity_feature_key_mismatch_backed_off_to_mask_only_keys",
            "missing_broad_regularization_or_support",
            "repaired_contract_support_rows_insufficient_for_runtime_distribution",
        ]
        if primary == PRIMARY_COVERAGE_COLLAPSE_MECHANISM
        else [],
        "no_training": True,
        "slice_4_training_allowed": False,
    }


def route_decision_for_v196(
    *,
    source_validation: Mapping[str, object],
    failure_fact_validation: Mapping[str, object],
    mechanism: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True or failure_fact_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif mechanism.get("route_class") == "artifact_fallback_or_coverage_collapse":
        route = COVERAGE_OR_ABSTENTION_REPAIR_ROUTE
    elif mechanism.get("route_class") == "dataset_insufficiency":
        route = DATASET_COVERAGE_DESIGN_ROUTE
    else:
        route = INSTRUMENTATION_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v196_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "mechanism_route_class": mechanism.get("route_class"),
        "direct_slice_4_training_allowed": False,
        "slice_4_training_consumed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "gate_relaxation_allowed": False,
        "promotion_authorized": False,
        "rationale": route_rationale(route, mechanism),
    }


def classification_for_v196(
    *,
    source_validation: Mapping[str, object],
    failure_fact_validation: Mapping[str, object],
    mechanism: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = (
        "m3_carrion_survivor_continuation_v196_repaired_contract_slice_3_"
        "failure_response_"
    )
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if failure_fact_validation.get("passed") is not True:
        return prefix + "unexpected_v195_failure_facts_closed_no_training"
    if route_decision.get("recommended_next_route") == COVERAGE_OR_ABSTENTION_REPAIR_ROUTE:
        return prefix + "artifact_coverage_collapse_routes_to_coverage_or_abstention_repair_no_training"
    if route_decision.get("recommended_next_route") == DATASET_COVERAGE_DESIGN_ROUTE:
        return prefix + "dataset_insufficiency_routes_to_coverage_design_no_training"
    if mechanism.get("primary_mechanism") == PRIMARY_INCONCLUSIVE_MECHANISM:
        return prefix + "mechanism_inconclusive_routes_to_instrumentation_no_training"
    return prefix + "closed_no_training"


def contract(
    *,
    expected_v195_report_exact_digest: str,
    expected_v195_artifact_digest: str,
    expected_v194_report_exact_digest: str,
    expected_v194_dataset_digest: str,
    expected_v195_route: str,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_allowed": False,
        "training_rerun_allowed": False,
        "slice_4_training_consumption_allowed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "gate_relaxation_allowed": False,
        "promotion_authorized": False,
        "requires_v195_report_digest": expected_v195_report_exact_digest,
        "requires_v195_artifact_digest": expected_v195_artifact_digest,
        "requires_v194_report_digest": expected_v194_report_exact_digest,
        "requires_v194_dataset_digest": expected_v194_dataset_digest,
        "requires_v195_route": expected_v195_route,
        "analysis_scope": [
            "v195 source pin validation",
            "v195 artifact lookup coverage",
            "training label distribution versus shadow action distribution",
            "v186 comparison",
            "no-training route recommendation",
        ],
    }


def lifecycle_flags(*, diagnostic_replay_ran: bool) -> dict[str, object]:
    return {
        "training_ran": False,
        "training_artifact_created": False,
        "fit_ran": False,
        "v195_training_rerun": False,
        "slice_3_training_consumed": False,
        "slice_4_training_consumed": False,
        "slice_4_training_started": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_policy_changed": False,
        "runtime_integration_ran": False,
        "shadow_eval_ran": False,
        "diagnostic_replay_ran": bool(diagnostic_replay_ran),
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "gate_relaxation_ran": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "non_promoted": True,
    }


def backup_metadata_audit(paths: Sequence[str | Path]) -> dict[str, object]:
    readable: list[str] = []
    missing: list[str] = []
    text_parts: list[str] = []
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            missing.append(str(path))
            continue
        readable.append(str(path))
        text_parts.append(path.read_text(encoding="utf-8"))
    text = "\n".join(text_parts)
    checks = {
        "backup_path_recorded": EXPECTED_V195_BACKUP in text,
        "archive_sha256_recorded": EXPECTED_V195_BACKUP_SHA256 in text,
        "rclone_check_recorded": "rclone check" in text,
        "rclone_zero_differences_recorded": (
            f"{EXPECTED_V195_RCLONE_DIFFERENCES}` differences" in text
            or f"{EXPECTED_V195_RCLONE_DIFFERENCES} differences" in text
        ),
        "rclone_matching_file_count_recorded": (
            f"{EXPECTED_V195_RCLONE_MATCHING_FILES}` matching file" in text
            or f"{EXPECTED_V195_RCLONE_MATCHING_FILES} matching file" in text
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v196_v195_backup_metadata_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "readable_paths": readable,
        "missing_paths": missing,
        "expected_v195_backup": EXPECTED_V195_BACKUP,
        "expected_v195_backup_sha256": EXPECTED_V195_BACKUP_SHA256,
        "expected_rclone_differences": EXPECTED_V195_RCLONE_DIFFERENCES,
        "expected_rclone_matching_files": EXPECTED_V195_RCLONE_MATCHING_FILES,
        "checks": checks,
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def route_rationale(route: str, mechanism: Mapping[str, object]) -> str:
    if route == COVERAGE_OR_ABSTENTION_REPAIR_ROUTE:
        return (
            "The v195 artifact did not default misses to stay. It frequently "
            "found low-specificity supported feature scores, applied overrides, "
            "and selected eat at a high share. The next step is a no-training "
            "design repair that requires enough feature coverage or abstention "
            "before any slice-4 training."
        )
    if route == DATASET_COVERAGE_DESIGN_ROUTE:
        return (
            "The pinned failure is consistent with repaired-contract support rows "
            "being insufficient for the runtime distribution. Route to dataset "
            "coverage design without training."
        )
    if route == INSTRUMENTATION_ROUTE:
        return (
            "The available diagnostics do not isolate a concrete mechanism. Add "
            "instrumentation before any further training."
        )
    return "Source pins or required v195 failure facts did not validate; stop."


def _mechanism_summary(primary: str) -> str:
    if primary == PRIMARY_COVERAGE_COLLAPSE_MECHANISM:
        return (
            "v195 collapsed because runtime states mostly backed off to "
            "low-specificity supported feature keys, especially mask-only keys, "
            "where clear-best supported scores produced many eat overrides. "
            "Misses abstained to the original Mind v3 action and did not default "
            "to stay."
        )
    if primary == PRIMARY_DATASET_INSUFFICIENCY_MECHANISM:
        return (
            "The action collapse cannot be explained by label imbalance alone and "
            "the repaired-contract support dataset is insufficient for the shadow "
            "runtime distribution."
        )
    return "The v195 failure mechanism was not isolated by the current diagnostics."


def _source_key_category(score_source: str, source_key: object) -> str:
    return transition_value_source_key_category(score_source, source_key)


def _best_action_for_stats(action_stats: Mapping[str, object]) -> str | None:
    best: tuple[str, float] | None = None
    for action, stats in action_stats.items():
        value = _float(_mapping(stats).get("utility_mean"))
        if best is None or value > best[1]:
            best = (str(action), value)
    return best[0] if best is not None else None


def _dominant(counts: Mapping[str, int] | Counter[str]) -> dict[str, object]:
    counter = Counter({str(key): int(value) for key, value in dict(counts).items()})
    total = sum(counter.values())
    if total <= 0:
        return {"action": None, "count": 0, "share": 0.0}
    action, count = max(counter.items(), key=lambda item: (item[1], item[0]))
    return {"action": action, "count": int(count), "share": _share(count, total)}


def _share(count: int | float, total: int | float) -> float:
    total_value = float(total)
    if total_value <= 0.0:
        return 0.0
    return _round(float(count) / total_value)


def _int_counter(value: object) -> Counter[str]:
    return Counter({str(key): int(raw) for key, raw in _mapping(value).items()})


def _counter(stats: dict[str, object], key: str) -> Counter[str]:
    value = stats.get(key)
    if not isinstance(value, Counter):
        value = Counter()
        stats[key] = value
    return value


def _nested_counter(
    stats: dict[str, object],
    key: str,
    nested_key: str,
) -> Counter[str]:
    value = stats.get(key)
    if not isinstance(value, defaultdict):
        value = defaultdict(Counter)
        stats[key] = value
    nested = value[nested_key]
    if not isinstance(nested, Counter):
        nested = Counter()
        value[nested_key] = nested
    return nested


def _float_matches(value: object, expected: float, *, digits: int = 4) -> bool:
    return round(_float(value), digits) == round(float(expected), digits)


def _load_optional_json_report(path_value: str | Path) -> dict[str, object] | None:
    path = Path(path_value)
    if not path.exists():
        return None
    return load_json_report(path)
