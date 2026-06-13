from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind import (
    carrion_survivor_continuation_v180_transition_row_policy_training as v180,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit as v194,
)
from evolution_sim.mind.candidate_campaign import _float, _int, _mapping, _round, write_json
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
from evolution_sim.mind.rollout_context import RolloutContextState
from evolution_sim.mind.transition_value_scorer import (
    MIND_V3_TRANSITION_VALUE_MODEL_ID,
    MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
    load_transition_value_scorer_artifact,
    transition_value_feature_keys,
)

M3_CARRION_SURVIVOR_CONTINUATION_V195_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_TRAINING_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V195_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_TRAINING_POLICY = (
    "opt_in_m3_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training_slice_3_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V195_ARTIFACT_POLICY = (
    "opt_in_m3_carrion_survivor_continuation_v195_public_repaired_contract_terminal_survival_support_policy_v1"
)
V195_CANDIDATE_KEY = "v195_repaired_contract_terminal_survival_support_slice_3"

DEFAULT_AUTHORIZATION_REPORT_PATH = v194.DEFAULT_OUTPUT_PATH
DEFAULT_COMPACT_SUPPORT_DATASET_PATH = v194.DEFAULT_COMPACT_SUPPORT_DATASET_OUTPUT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v195-carrion-survivor-continuation-repaired-contract-terminal-survival-support-training.json"
)
DEFAULT_ARTIFACT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v195-carrion-survivor-continuation-repaired-contract-terminal-survival-support-policy-artifact.json"
)
DEFAULT_BACKUP_DOC_PATHS = (
    Path("docs/mind-v3-autonomous-evolution.md"),
    Path("AGENTS.md"),
)

EXPECTED_V194_REPORT_EXACT_DIGEST = (
    "ccd02e7bc4862f8d8797fa21efb3d29fc2034196d2a54a6d42032062e2caaf2e"
)
EXPECTED_V194_COMPACT_DATASET_DIGEST = (
    "dd0061fae106fbe9c101c426357b3f78c9b5c4ed5989011baaf9a3cc514c2ffc"
)
EXPECTED_V193_REPORT_EXACT_DIGEST = v194.EXPECTED_V193_REPORT_EXACT_DIGEST
EXPECTED_V192_REPORT_EXACT_DIGEST = v194.EXPECTED_V192_REPORT_EXACT_DIGEST
EXPECTED_V194_ROUTE = v194.SUCCESS_ROUTE
EXPECTED_V194_COMPACT_DATASET_ROW_COUNT = 3599
EXPECTED_UNSUPPORTED_REQUESTED_ACTION_COUNT = 0
EXPECTED_SELECTED_SAME_TICK_OCCUPANCY_DRIFT_COUNT = 77
EXPECTED_UNEXPECTED_RESOLUTION_INVALID_COUNT = 0
EXPECTED_V194_BACKUP = (
    "gdrive:evolution-sim-backups/archives/"
    "20260613T175848Z-v194-repaired-contract-terminal-survival-support-dataset-audit.tar.zst"
)
EXPECTED_V194_BACKUP_SHA256 = (
    "461295919b0ed81c4d7dc2a4a7ffcb4ceab82c5e4779256205419180129d748e"
)
EXPECTED_V194_RCLONE_DIFFERENCES = 0
EXPECTED_V194_RCLONE_MATCHING_FILES = 1

TRAINING_SLICE_INDEX = 3
PREVIOUS_TRAINING_SLICES_CONSUMED = 2
CAMPAIGN_SLICE_CAP = 10

DEFAULT_BROAD_SEEDS = v180.DEFAULT_BROAD_SEEDS
DEFAULT_CARRION_FIXTURE_SEEDS = v180.DEFAULT_CARRION_FIXTURE_SEEDS
DEFAULT_TICKS = v180.DEFAULT_TICKS
DEFAULT_FEATURE_KEY_LIMIT = 8
MAX_DOMINANT_REQUESTED_ACTION_SHARE = v180.MAX_DOMINANT_REQUESTED_ACTION_SHARE

SUCCESS_ROUTE = "v196_runtime_integration_readiness_audit_no_runtime_change"
FAILURE_ROUTE = "v196_repaired_contract_slice_3_failure_response_no_training"

FALSE_V194_LIFECYCLE_FLAGS = (
    "training_ran",
    "training_artifact_created",
    "slice_3_training_consumed",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "promotion_authorized",
    "gate_relaxation_allowed",
    "support_generation_ran",
    "support_expansion_ran",
)


def run_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training(
    *,
    authorization_report_path: str | Path = DEFAULT_AUTHORIZATION_REPORT_PATH,
    compact_support_dataset_path: str | Path = DEFAULT_COMPACT_SUPPORT_DATASET_PATH,
    artifact_output_path: str | Path = DEFAULT_ARTIFACT_OUTPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_authorization_report_exact_digest: str = EXPECTED_V194_REPORT_EXACT_DIGEST,
    expected_dataset_digest: str = EXPECTED_V194_COMPACT_DATASET_DIGEST,
    expected_training_route: str = EXPECTED_V194_ROUTE,
    expected_v193_report_exact_digest: str = EXPECTED_V193_REPORT_EXACT_DIGEST,
    expected_v192_report_exact_digest: str = EXPECTED_V192_REPORT_EXACT_DIGEST,
    expected_dataset_row_count: int = EXPECTED_V194_COMPACT_DATASET_ROW_COUNT,
    expected_unsupported_requested_action_count: int = EXPECTED_UNSUPPORTED_REQUESTED_ACTION_COUNT,
    expected_selected_same_tick_occupancy_drift_count: int = EXPECTED_SELECTED_SAME_TICK_OCCUPANCY_DRIFT_COUNT,
    expected_unexpected_resolution_invalid_count: int = EXPECTED_UNEXPECTED_RESOLUTION_INVALID_COUNT,
    backup_doc_paths: Sequence[str | Path] = DEFAULT_BACKUP_DOC_PATHS,
    backup_metadata_override: Mapping[str, object] | None = None,
    feature_key_limit: int = DEFAULT_FEATURE_KEY_LIMIT,
    run_evaluation: bool = True,
    broad_seeds: Sequence[int] = DEFAULT_BROAD_SEEDS,
    carrion_fixture_seeds: Sequence[int] = DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = DEFAULT_TICKS,
) -> dict[str, object]:
    authorization_report = load_json_report(authorization_report_path)
    rows = [dict(row) for row in load_jsonl_dataset(compact_support_dataset_path)]
    authorization_validation = validate_v195_training_authorization_report(
        authorization_report,
        expected_exact_digest=expected_authorization_report_exact_digest,
        expected_dataset_digest=expected_dataset_digest,
        expected_training_route=expected_training_route,
        expected_v193_report_exact_digest=expected_v193_report_exact_digest,
        expected_v192_report_exact_digest=expected_v192_report_exact_digest,
        expected_dataset_row_count=expected_dataset_row_count,
        expected_unsupported_requested_action_count=(
            expected_unsupported_requested_action_count
        ),
        expected_selected_same_tick_occupancy_drift_count=(
            expected_selected_same_tick_occupancy_drift_count
        ),
        expected_unexpected_resolution_invalid_count=(
            expected_unexpected_resolution_invalid_count
        ),
        backup_doc_paths=backup_doc_paths,
        backup_metadata_override=backup_metadata_override,
    )
    source_validation = source_validation_for_v195_training(
        rows=rows,
        authorization_report=authorization_report,
        authorization_validation=authorization_validation,
        compact_support_dataset_path=compact_support_dataset_path,
        expected_dataset_digest=expected_dataset_digest,
        expected_v193_report_exact_digest=expected_v193_report_exact_digest,
        expected_dataset_row_count=expected_dataset_row_count,
    )
    if source_validation.get("passed") is True:
        artifact_payload, training = build_v195_repaired_contract_support_policy_artifact(
            rows,
            dataset_digest=expected_dataset_digest,
            authorization_report_exact_digest=(
                expected_authorization_report_exact_digest
            ),
            authorization_route=expected_training_route,
            feature_key_limit=feature_key_limit,
        )
        write_json(artifact_output_path, artifact_payload)
        artifact_digest = stable_payload_digest(artifact_payload)
        artifact = artifact_status(
            artifact_output_path,
            created=True,
            reason=None,
            payload=artifact_payload,
            digest=artifact_digest,
        )
        if run_evaluation:
            evaluation = run_v195_shadow_evaluation(
                artifact=artifact_payload,
                broad_seeds=broad_seeds,
                carrion_fixture_seeds=carrion_fixture_seeds,
                ticks=ticks,
            )
            acceptance = acceptance_for_v195(evaluation)
        else:
            evaluation = skipped_evaluation("run_evaluation_false")
            acceptance = closed_acceptance("evaluation_not_run")
    else:
        artifact = artifact_status(
            artifact_output_path,
            created=False,
            reason="source_validation_failed",
        )
        training = skipped_training("source_validation_failed")
        evaluation = skipped_evaluation("source_validation_failed")
        acceptance = closed_acceptance("source_validation_failed")
    slice_consumed = training.get("ran") is True and artifact.get("created") is True
    route_decision = route_decision_for_v195(
        source_validation=source_validation,
        training=training,
        artifact=artifact,
        evaluation=evaluation,
        acceptance=acceptance,
    )
    classification = classification_for_v195(
        source_validation=source_validation,
        training=training,
        evaluation=evaluation,
        acceptance=acceptance,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V195_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_TRAINING_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V195_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_TRAINING_POLICY
        ),
        "contract": contract(
            expected_authorization_report_exact_digest=(
                expected_authorization_report_exact_digest
            ),
            expected_dataset_digest=expected_dataset_digest,
            expected_training_route=expected_training_route,
            expected_v193_report_exact_digest=expected_v193_report_exact_digest,
            expected_v192_report_exact_digest=expected_v192_report_exact_digest,
            expected_dataset_row_count=expected_dataset_row_count,
        ),
        "inputs": {
            "authorization_report": str(authorization_report_path),
            "compact_support_dataset": str(compact_support_dataset_path),
            "artifact_output": str(artifact_output_path),
            "expected_authorization_report_exact_digest": (
                expected_authorization_report_exact_digest
            ),
            "expected_dataset_digest": expected_dataset_digest,
            "expected_training_route": expected_training_route,
            "expected_v193_report_exact_digest": expected_v193_report_exact_digest,
            "expected_v192_report_exact_digest": expected_v192_report_exact_digest,
            "expected_dataset_row_count": int(expected_dataset_row_count),
            "feature_key_limit": int(feature_key_limit),
            "run_evaluation": bool(run_evaluation),
            "broad_seeds": [int(seed) for seed in broad_seeds],
            "carrion_fixture_seeds": [int(seed) for seed in carrion_fixture_seeds],
            "ticks": int(ticks),
        },
        "authorization_report_validation": authorization_validation,
        "source_validation": source_validation,
        "dataset": {
            "path": str(compact_support_dataset_path),
            "row_count": len(rows),
            "dataset_digest": stable_payload_digest(rows),
            "source_producer": (
                "v194_repaired_contract_terminal_survival_support_dataset_audit"
            ),
        },
        "artifact": artifact,
        "training": training,
        "evaluation": evaluation,
        "acceptance": acceptance,
        "route_decision": route_decision,
        "training_slice_budget": {
            "campaign": "carrion_repaired_contract_terminal_survival_support",
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
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "explicit_opt_in_training_slice_3",
                "shadow_eval_only",
                "no_runtime_integration",
                "no_promotion",
            ],
        },
        **lifecycle_flags(
            source_authorized=source_validation.get("passed") is True,
            training_ran=training.get("ran") is True,
            training_artifact_created=artifact.get("created") is True,
            shadow_eval_ran=evaluation.get("ran") is True,
            slice_3_training_consumed=slice_consumed,
        ),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v195_training_authorization_report(
    report: Mapping[str, object],
    *,
    expected_exact_digest: str,
    expected_dataset_digest: str,
    expected_training_route: str,
    expected_v193_report_exact_digest: str,
    expected_v192_report_exact_digest: str,
    expected_dataset_row_count: int,
    expected_unsupported_requested_action_count: int,
    expected_selected_same_tick_occupancy_drift_count: int,
    expected_unexpected_resolution_invalid_count: int,
    backup_doc_paths: Sequence[str | Path],
    backup_metadata_override: Mapping[str, object] | None,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(report)
    source = _mapping(report.get("source_validation"))
    route = _mapping(report.get("route_decision"))
    dataset = _mapping(report.get("compact_support_dataset"))
    dataset_audit = _mapping(report.get("dataset_audit"))
    selected = _mapping(report.get("selected_support_facts_audit"))
    classification = _mapping(report.get("classification"))
    backup = (
        dict(backup_metadata_override)
        if backup_metadata_override is not None
        else backup_metadata_audit(backup_doc_paths)
    )
    checks = {
        "schema_version_matches_v194_dataset_audit": (
            report.get("schema_version")
            == v194.M3_CARRION_SURVIVOR_CONTINUATION_V194_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_SCHEMA_VERSION
        ),
        "policy_matches_v194_dataset_audit": (
            report.get("policy")
            == v194.M3_CARRION_SURVIVOR_CONTINUATION_V194_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_POLICY
        ),
        "exact_digest_valid": exact_validation.get("passed") is True,
        "expected_exact_digest_matches": report.get("exact_digest") == expected_exact_digest,
        "classification_matches_v194_success": (
            classification.get("primary")
            == (
                "m3_carrion_survivor_continuation_v194_repaired_contract_terminal_"
                "survival_support_dataset_audit_support_dataset_ready_for_future_"
                "explicit_slice_3_training"
            )
        ),
        "source_validation_passed": source.get("passed") is True,
        "v193_digest_pin_matches_expected": (
            source.get("observed_v193_report_exact_digest")
            == expected_v193_report_exact_digest
        ),
        "v192_digest_pin_matches_expected": (
            source.get("observed_v192_report_exact_digest")
            == expected_v192_report_exact_digest
        ),
        "v194_backup_metadata_validated": backup.get("passed") is True,
        "dataset_created": dataset.get("created") is True,
        "dataset_digest_matches_expected": (
            dataset.get("dataset_digest") == expected_dataset_digest
        ),
        "dataset_row_count_matches_expected": (
            _int(dataset.get("row_count")) == int(expected_dataset_row_count)
        ),
        "dataset_audit_passed": dataset_audit.get("passed") is True,
        "dataset_audit_ready_for_slice_3": (
            dataset_audit.get("dataset_ready_for_future_slice_3_training") is True
        ),
        "dataset_audit_digest_matches_expected": (
            dataset_audit.get("dataset_digest") == expected_dataset_digest
        ),
        "dataset_audit_row_count_matches_expected": (
            _int(dataset_audit.get("row_count")) == int(expected_dataset_row_count)
        ),
        "route_matches_v195_slice_3_opt_in": (
            route.get("recommended_next_route") == expected_training_route
        ),
        "route_authorizes_future_slice_3": (
            route.get("future_explicit_slice_3_training_route_authorized") is True
        ),
        "unsupported_requested_actions_zero": (
            _int(selected.get("unsupported_requested_action_count"))
            == int(expected_unsupported_requested_action_count)
        ),
        "selected_same_tick_occupancy_drift_matches_expected": (
            _int(selected.get("selected_expected_same_tick_occupancy_drift_count"))
            == int(expected_selected_same_tick_occupancy_drift_count)
        ),
        "unexpected_resolution_invalid_zero": (
            _int(selected.get("unexpected_resolution_invalid_count"))
            == int(expected_unexpected_resolution_invalid_count)
        ),
    }
    repaired_resolution = _mapping(dataset_audit.get("repaired_resolution_check"))
    checks.update(
        {
            "dataset_repaired_resolution_unsupported_requested_zero": (
                _int(repaired_resolution.get("unsupported_requested_action_count"))
                == int(expected_unsupported_requested_action_count)
            ),
            "dataset_repaired_resolution_selected_drift_matches_expected": (
                _int(
                    repaired_resolution.get(
                        "expected_same_tick_occupancy_drift_count"
                    )
                )
                == int(expected_selected_same_tick_occupancy_drift_count)
            ),
            "dataset_repaired_resolution_unexpected_invalid_zero": (
                _int(repaired_resolution.get("unexpected_resolution_invalid_count"))
                == int(expected_unexpected_resolution_invalid_count)
            ),
            "dataset_expected_drift_not_successful_move": (
                _int(
                    repaired_resolution.get(
                        "expected_same_tick_occupancy_drift_counted_as_successful_move_count"
                    )
                )
                == 0
            ),
        }
    )
    for flag in FALSE_V194_LIFECYCLE_FLAGS:
        checks[f"v194_{flag}_closed"] = report.get(flag) is False
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v195_v194_training_authorization_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_exact_digest": expected_exact_digest,
        "observed_exact_digest": report.get("exact_digest"),
        "exact_digest_validation": exact_validation,
        "expected_dataset_digest": expected_dataset_digest,
        "observed_dataset_digest": dataset.get("dataset_digest"),
        "expected_training_route": expected_training_route,
        "observed_training_route": route.get("recommended_next_route"),
        "expected_v193_report_exact_digest": expected_v193_report_exact_digest,
        "observed_v193_report_exact_digest": source.get(
            "observed_v193_report_exact_digest"
        ),
        "expected_v192_report_exact_digest": expected_v192_report_exact_digest,
        "observed_v192_report_exact_digest": source.get(
            "observed_v192_report_exact_digest"
        ),
        "expected_dataset_row_count": int(expected_dataset_row_count),
        "observed_dataset_row_count": dataset.get("row_count"),
        "v194_backup_metadata": backup,
        "checks": checks,
    }


def source_validation_for_v195_training(
    *,
    rows: Sequence[Mapping[str, object]],
    authorization_report: Mapping[str, object],
    authorization_validation: Mapping[str, object],
    compact_support_dataset_path: str | Path,
    expected_dataset_digest: str,
    expected_v193_report_exact_digest: str,
    expected_dataset_row_count: int,
) -> dict[str, object]:
    dataset_digest = stable_payload_digest([dict(row) for row in rows])
    selected = _mapping(authorization_report.get("selected_support_facts_audit"))
    dataset_write = {
        "created": True,
        "path": str(compact_support_dataset_path),
        "row_count": len(rows),
        "dataset_digest": dataset_digest,
        "gitignored_output_mind_path": str(compact_support_dataset_path).startswith(
            "output/mind/"
        ),
    }
    dataset_audit = v194.compact_support_dataset_audit(
        rows,
        dataset_write=dataset_write,
        selected_support=selected,
        expected_v193_report_exact_digest=expected_v193_report_exact_digest,
    )
    checks = {
        "authorization_report_validation_passed": (
            authorization_validation.get("passed") is True
        ),
        "dataset_digest_matches_expected": dataset_digest == expected_dataset_digest,
        "dataset_digest_matches_authorization_report": (
            dataset_digest
            == str(authorization_validation.get("observed_dataset_digest") or "")
        ),
        "dataset_row_count_matches_expected": (
            len(rows) == int(expected_dataset_row_count)
        ),
        "dataset_non_empty": bool(rows),
        "dataset_audit_recomputed_passed": dataset_audit.get("passed") is True,
        "dataset_leakage_check_passed": (
            _mapping(dataset_audit.get("leakage_check")).get("passed") is True
        ),
        "dataset_action_mask_check_passed": (
            _mapping(dataset_audit.get("action_mask_check")).get("passed") is True
        ),
        "dataset_repaired_resolution_check_passed": (
            _mapping(dataset_audit.get("repaired_resolution_check")).get("passed")
            is True
        ),
        "dataset_observation_check_passed": (
            _mapping(dataset_audit.get("observation_check")).get("passed") is True
        ),
        "dataset_target_check_passed": (
            _mapping(dataset_audit.get("target_check")).get("passed") is True
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v195_source_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "dataset_digest": dataset_digest,
        "expected_dataset_digest": expected_dataset_digest,
        "row_count": len(rows),
        "expected_dataset_row_count": int(expected_dataset_row_count),
        "dataset_audit": dataset_audit,
        "checks": checks,
    }


def build_v195_repaired_contract_support_policy_artifact(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset_digest: str,
    authorization_report_exact_digest: str,
    authorization_route: str,
    feature_key_limit: int = DEFAULT_FEATURE_KEY_LIMIT,
) -> tuple[dict[str, object], dict[str, object]]:
    key_limit = max(1, min(int(feature_key_limit), 8))
    buckets: defaultdict[str, defaultdict[str, dict[str, object]]] = defaultdict(
        lambda: defaultdict(
            lambda: {"count": 0, "utility_sum": 0.0, "components": defaultdict(float)}
        )
    )
    row_failures: list[dict[str, object]] = []
    row_action_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        try:
            features = _mapping(row.get("trainable_public_features"))
            observation = _mapping(features.get("current_public_observation"))
            action_mask = _mapping(features.get("current_public_action_mask"))
            target = _mapping(row.get("supervised_target"))
            action = str(target.get("requested_action") or "")
            if action not in ACTION_NAMES:
                raise ValueError("requested_action_not_in_action_names")
            if action_mask.get(action) is not True:
                raise ValueError("requested_action_not_in_current_public_action_mask")
            if target.get("requested_action_valid_in_current_public_action_mask") is not True:
                raise ValueError("requested_action_valid_flag_false")
            if target.get("terminal_survival_support") is not True:
                raise ValueError("terminal_survival_support_false")
            utility = compact_support_row_utility(row)
            feature_keys = transition_value_feature_keys(
                observation_input=observation,
                valid_action_mask=action_mask,
                state=RolloutContextState(),
            )[:key_limit]
        except (TypeError, ValueError) as exc:
            row_failures.append(
                {
                    "row_index": row_index,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        row_action_counts.update([action])
        for feature_key in feature_keys:
            _add_utility(buckets[str(feature_key)][action], utility)
    feature_action_utility = {
        feature_key: {
            action: _finalize_action_stats(bucket)
            for action, bucket in sorted(action_bucket.items())
            if action in ACTION_NAMES
        }
        for feature_key, action_bucket in sorted(buckets.items())
    }
    artifact = {
        "schema_version": MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
        "artifact_policy": M3_CARRION_SURVIVOR_CONTINUATION_V195_ARTIFACT_POLICY,
        "model_id": MIND_V3_TRANSITION_VALUE_MODEL_ID,
        "diagnostics_only": False,
        "explicit_opt_in_required": True,
        "runtime_action_selection_authorized": False,
        "runtime_policy_change_requires_explicit_flag": True,
        "promotion_authorized": False,
        "built_from": {
            "source": (
                "v194_repaired_contract_terminal_survival_support_compact_dataset"
            ),
            "training_row_count": len(rows),
            "accepted_training_row_count": sum(row_action_counts.values()),
            "dataset_digest": dataset_digest,
            "authorization_report_exact_digest": authorization_report_exact_digest,
            "authorization_route": authorization_route,
            "feature_key_limit": key_limit,
            "global_feature_key_excluded": True,
            "training_slice_index": TRAINING_SLICE_INDEX,
            "previous_training_slices_consumed": (
                PREVIOUS_TRAINING_SLICES_CONSUMED
            ),
            "campaign_slice_cap": CAMPAIGN_SLICE_CAP,
            "slice_3_opt_in_training_slice": True,
            "runtime_action_selection_authorized": False,
            "promotion_authorized": False,
        },
        "utility_policy": {
            "target": (
                "repaired_contract_terminal_survival_support_requested_action_utility"
            ),
            "weights": compact_support_utility_weights(),
            "estimates_action_utility_not_action_frequency": True,
            "expected_same_tick_occupancy_drift_counted_separately": True,
            "expected_same_tick_occupancy_drift_counted_as_successful_move": False,
        },
        "feature_policy": {
            "current_decision_inputs": [
                "public observation_input",
                "public action_mask",
            ],
            "prior_history_inputs": [
                "same-agent finalized public trajectory rows before current decision"
            ],
            "excluded_score_features": [
                "seed identity",
                "fixture identity",
                "private world state",
                "future rows at decision time",
                "heuristic recommendations",
                "source path",
                "provenance",
                "terminal outcome labels at decision time",
            ],
        },
        "utility_tables": {
            "policy": (
                "v195_mean_repaired_contract_terminal_survival_support_utility_by_public_feature_key_and_action_v1"
            ),
            "feature_action_utility": feature_action_utility,
        },
        "action_support_counts": _counter_dict(row_action_counts),
        "unsupported_action_rejection": {
            "policy": "runtime_scores_only_currently_valid_actions_v1",
            "invalid_actions_are_never_selected": True,
        },
    }
    loaded = load_transition_value_scorer_artifact(artifact)
    artifact_leakage_scan = v180._artifact_feature_leakage_scan(artifact)
    dominant = v180._dominant_share(row_action_counts)
    training = {
        "policy": (
            "m3_carrion_survivor_continuation_v195_training_summary_v1"
        ),
        "ran": True,
        "passed": (
            not row_failures
            and artifact_leakage_scan.get("passed") is True
            and loaded.artifact == artifact
        ),
        "training_slice_index": TRAINING_SLICE_INDEX,
        "previous_training_slices_consumed": PREVIOUS_TRAINING_SLICES_CONSUMED,
        "campaign_slice_cap": CAMPAIGN_SLICE_CAP,
        "training_row_count": len(rows),
        "accepted_training_row_count": sum(row_action_counts.values()),
        "row_failure_count": len(row_failures),
        "row_failures": row_failures[:24],
        "feature_key_count": len(feature_action_utility),
        "feature_key_limit": key_limit,
        "global_feature_key_excluded": True,
        "action_support_counts": _counter_dict(row_action_counts),
        "dominant_training_action": dominant["action"],
        "dominant_training_action_share": dominant["share"],
        "artifact_feature_leakage_scan": artifact_leakage_scan,
        "artifact_load_check": {
            "policy": "v195_transition_value_artifact_load_check_v1",
            "loaded": loaded.artifact == artifact,
            "schema_version": loaded.artifact.get("schema_version"),
            "artifact_policy": loaded.artifact.get("artifact_policy"),
        },
    }
    return artifact, training


def compact_support_row_utility(row: Mapping[str, object]) -> dict[str, object]:
    terminal = _mapping(row.get("terminal_support_summary"))
    target = _mapping(row.get("supervised_target"))
    resolution = _mapping(row.get("repaired_resolution"))
    components = {
        "terminal_survival_support": (
            1.0 if target.get("terminal_survival_support") is True else 0.0
        ),
        "alive_agents": _float(terminal.get("alive_agents")),
        "births": _float(terminal.get("births")),
        "requested_action_valid": (
            1.0
            if target.get("requested_action_valid_in_current_public_action_mask")
            is True
            else 0.0
        ),
        "resolution_valid": (
            1.0 if resolution.get("resolution_action_valid") is True else 0.0
        ),
        "expected_same_tick_occupancy_drift": (
            1.0
            if resolution.get(
                "expected_same_tick_occupancy_drift_counted_separately"
            )
            is True
            else 0.0
        ),
    }
    weights = compact_support_utility_weights()
    total = sum(
        float(components[name]) * float(weights[name])
        for name in sorted(weights)
    )
    return {"utility": _round(total), "components": components}


def compact_support_utility_weights() -> dict[str, float]:
    return {
        "terminal_survival_support": 5.0,
        "alive_agents": 0.05,
        "births": 0.08,
        "requested_action_valid": 0.5,
        "resolution_valid": 0.1,
        "expected_same_tick_occupancy_drift": -0.05,
    }


def run_v195_shadow_evaluation(
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
                "m3_carrion_survivor_continuation_v195_shadow_evaluation_v1"
            ),
            "candidate_key": V195_CANDIDATE_KEY,
            "mode": "offline_shadow_controlled_eval",
            "runtime_integration": False,
            "runtime_action_selection_changed": False,
            "promotion_evidence": False,
        }
    )
    return evaluation


def acceptance_for_v195(evaluation: Mapping[str, object]) -> dict[str, object]:
    acceptance = dict(v180._acceptance(evaluation))
    acceptance.update(
        {
            "policy": "m3_carrion_survivor_continuation_v195_acceptance_v1",
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


def route_decision_for_v195(
    *,
    source_validation: Mapping[str, object],
    training: Mapping[str, object],
    artifact: Mapping[str, object],
    evaluation: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True:
        route = repair_route_for("source_validation_failed")
        blockers = list(source_validation.get("failures") or ["source_validation_failed"])
    elif training.get("passed") is not True or artifact.get("created") is not True:
        route = repair_route_for("training_artifact_invalid")
        blockers = ["training_artifact_invalid"]
    elif evaluation.get("ran") is not True:
        route = repair_route_for("shadow_evaluation_not_run")
        blockers = ["shadow_evaluation_not_run"]
    elif acceptance.get("passed") is True:
        route = SUCCESS_ROUTE
        blockers = []
    else:
        route = FAILURE_ROUTE
        blockers = [str(item.get("name") or item) for item in acceptance.get("blockers", [])]
    return {
        "policy": "m3_carrion_survivor_continuation_v195_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "shadow_acceptance_passed": acceptance.get("passed") is True,
        "shadow_acceptance_failed": (
            evaluation.get("ran") is True and acceptance.get("passed") is not True
        ),
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
    }


def repair_route_for(blocker: str) -> str:
    return (
        "v196_repaired_contract_slice_3_"
        f"{str(blocker).lower()}_repair_no_training"
    )


def classification_for_v195(
    *,
    source_validation: Mapping[str, object],
    training: Mapping[str, object],
    evaluation: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> str:
    prefix = (
        "m3_carrion_survivor_continuation_v195_repaired_contract_terminal_"
        "survival_support_training_"
    )
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if training.get("passed") is not True:
        return prefix + "slice_3_training_artifact_invalid_no_runtime_change"
    if evaluation.get("ran") is not True:
        return prefix + "slice_3_trained_shadow_eval_not_run_no_runtime_change"
    if acceptance.get("passed") is True:
        return prefix + "slice_3_shadow_acceptance_passed_routes_to_runtime_readiness_audit"
    return prefix + "slice_3_shadow_acceptance_failed_routes_to_failure_response"


def contract(
    *,
    expected_authorization_report_exact_digest: str,
    expected_dataset_digest: str,
    expected_training_route: str,
    expected_v193_report_exact_digest: str,
    expected_v192_report_exact_digest: str,
    expected_dataset_row_count: int,
) -> dict[str, object]:
    return {
        "explicit_opt_in_training_slice": True,
        "training_slice_index": TRAINING_SLICE_INDEX,
        "previous_training_slices_consumed": PREVIOUS_TRAINING_SLICES_CONSUMED,
        "campaign_slice_cap": CAMPAIGN_SLICE_CAP,
        "training_requires_v194_dataset_audit_report": True,
        "training_requires_v194_compact_dataset_digest": True,
        "training_requires_v195_slice_3_route": True,
        "authorization_report_exact_digest_pinned": (
            expected_authorization_report_exact_digest
        ),
        "input_dataset_digest_pinned": expected_dataset_digest,
        "training_route_required": expected_training_route,
        "v193_report_exact_digest_pinned": expected_v193_report_exact_digest,
        "v192_report_exact_digest_pinned": expected_v192_report_exact_digest,
        "expected_dataset_row_count": int(expected_dataset_row_count),
        "runtime_integration_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "evaluation_scope": "offline_shadow_controlled_eval_only",
        "promotion_evidence": False,
    }


def lifecycle_flags(
    *,
    source_authorized: bool,
    training_ran: bool,
    training_artifact_created: bool,
    shadow_eval_ran: bool,
    slice_3_training_consumed: bool,
) -> dict[str, object]:
    return {
        "training_authorized": bool(source_authorized),
        "training_ran": bool(training_ran),
        "training_artifact_created": bool(training_artifact_created),
        "fit_ran": bool(training_ran),
        "slice_3_training_consumed": bool(slice_3_training_consumed),
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": bool(shadow_eval_ran),
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "gate_relaxation_ran": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "v180_rerun": False,
        "v186_rerun": False,
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
        "backup_path_recorded": EXPECTED_V194_BACKUP in text,
        "archive_sha256_recorded": EXPECTED_V194_BACKUP_SHA256 in text,
        "rclone_check_recorded": "rclone check" in text,
        "rclone_zero_differences_recorded": (
            f"{EXPECTED_V194_RCLONE_DIFFERENCES}` differences" in text
            or f"{EXPECTED_V194_RCLONE_DIFFERENCES} differences" in text
        ),
        "rclone_matching_file_count_recorded": (
            f"{EXPECTED_V194_RCLONE_MATCHING_FILES}` matching file" in text
            or f"{EXPECTED_V194_RCLONE_MATCHING_FILES} matching file" in text
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v195_v194_backup_metadata_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "readable_paths": readable,
        "missing_paths": missing,
        "expected_v194_backup": EXPECTED_V194_BACKUP,
        "expected_v194_backup_sha256": EXPECTED_V194_BACKUP_SHA256,
        "expected_rclone_differences": EXPECTED_V194_RCLONE_DIFFERENCES,
        "expected_rclone_matching_files": EXPECTED_V194_RCLONE_MATCHING_FILES,
        "checks": checks,
    }


def artifact_status(
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
        table = _mapping(_mapping(payload.get("utility_tables")).get("feature_action_utility"))
        status.update(
            {
                "schema_version": payload.get("schema_version"),
                "artifact_policy": payload.get("artifact_policy"),
                "model_id": payload.get("model_id"),
                "training_row_count": _mapping(payload.get("built_from")).get(
                    "training_row_count"
                ),
                "feature_key_count": len(table),
            }
        )
    return status


def skipped_training(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v195_training_summary_v1",
        "ran": False,
        "passed": False,
        "reason": reason,
        "training_slice_index": TRAINING_SLICE_INDEX,
    }


def skipped_evaluation(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v195_shadow_evaluation_v1",
        "ran": False,
        "reason": reason,
    }


def closed_acceptance(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v195_acceptance_v1",
        "passed": False,
        "blocker_count": 1,
        "blockers": [v180._blocker(reason, observed=False, required=True)],
        "training_slice_index": TRAINING_SLICE_INDEX,
        "promotion_evidence": False,
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def _add_utility(bucket: dict[str, object], utility: Mapping[str, object]) -> None:
    bucket["count"] = int(bucket.get("count", 0)) + 1
    bucket["utility_sum"] = float(bucket.get("utility_sum", 0.0)) + _float(
        utility.get("utility")
    )
    components = bucket.get("components")
    if not isinstance(components, defaultdict):
        components = defaultdict(float)
        bucket["components"] = components
    for name, value in _mapping(utility.get("components")).items():
        components[str(name)] += _float(value)


def _finalize_action_stats(bucket: Mapping[str, object]) -> dict[str, object]:
    count = _int(bucket.get("count"))
    components = _mapping(bucket.get("components"))
    return {
        "count": count,
        "utility_mean": _round(_float(bucket.get("utility_sum")) / float(count)),
        "component_means": {
            name: _round(_float(value) / float(count))
            for name, value in sorted(components.items())
        },
    }


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {action: int(counter[action]) for action in ACTION_NAMES if counter[action]}
