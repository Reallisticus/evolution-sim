from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import gzip
import json
from pathlib import Path

from evolution_sim.mind.candidate_campaign import _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion as v190,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v191_legal_support_repair_architecture_review as v191,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V192_ACTION_RESOLUTION_CONTRACT_REPAIR_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v192_action_resolution_contract_repair_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V192_ACTION_RESOLUTION_CONTRACT_REPAIR_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v192_action_resolution_contract_repair_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v192-carrion-survivor-continuation-action-resolution-contract-repair.json"
)
DEFAULT_V191_REPORT_PATH = v191.DEFAULT_OUTPUT_PATH
DEFAULT_V190_REPORT_PATH = v190.DEFAULT_OUTPUT_PATH
EXPECTED_V191_REPORT_EXACT_DIGEST = (
    "eef890ac70140028f9d407d0b24fca40827947e75153ba5e9563b027a4233ce5"
)
EXPECTED_V190_REPORT_EXACT_DIGEST = v191.EXPECTED_V190_REPORT_EXACT_DIGEST
REQUIRED_V191_ROUTE = v191.ACTION_RESOLUTION_CONTRACT_REPAIR_ROUTE
EXPECTED_UNSUPPORTED_REQUESTED_ACTION_COUNT = 0
EXPECTED_UNSUPPORTED_RESOLVED_ACTION_COUNT = 860
EXPECTED_UNSUPPORTED_RESOLVED_COUNTS_BY_SEED = {
    "13": 127,
    "19": 149,
    "29": 194,
    "37": 117,
    "41": 133,
    "43": 140,
}
EXPECTED_UNSUPPORTED_RESOLVED_COUNTS_BY_ACTION = {
    "move_east": 295,
    "move_north": 236,
    "move_south": 206,
    "move_west": 123,
}
EXPECTED_UNSUPPORTED_RESOLVED_COUNTS_BY_REASON = {
    "not_in_resolution_action_mask": 860,
}
EXPECTED_UNSUPPORTED_RESOLVED_COUNTS_BY_RESOLVED_ACTION = {"stay": 860}
EXPECTED_ROOT_CAUSE = "action_mask_timing_mismatch_same_tick_movement_occupancy_race"
TARGET_SEEDS = (13, 19, 29, 37, 41, 43)

NEXT_ROUTE_AFTER_REPAIR = (
    "v193_fresh_targeted_legal_support_expansion_after_action_resolution_contract_repair_no_training"
)
NEXT_ROUTE_ARCHITECTURE_REVIEW = "v193_action_resolution_architecture_review_no_training"
STOP_ROUTE = "stop"

MOVEMENT_DELTAS = {
    "move_north": (0, -1),
    "move_south": (0, 1),
    "move_east": (1, 0),
    "move_west": (-1, 0),
}

FALSE_LIFECYCLE_FLAGS = (
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


def run_carrion_survivor_continuation_v192_action_resolution_contract_repair(
    *,
    v191_report_path: str | Path = DEFAULT_V191_REPORT_PATH,
    v190_report_path: str | Path = DEFAULT_V190_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v191_report_exact_digest: str = EXPECTED_V191_REPORT_EXACT_DIGEST,
    expected_v190_report_exact_digest: str = EXPECTED_V190_REPORT_EXACT_DIGEST,
    required_v191_route: str = REQUIRED_V191_ROUTE,
    expected_unsupported_requested_action_count: int = (
        EXPECTED_UNSUPPORTED_REQUESTED_ACTION_COUNT
    ),
    expected_unsupported_resolved_action_count: int = (
        EXPECTED_UNSUPPORTED_RESOLVED_ACTION_COUNT
    ),
    expected_counts_by_seed: Mapping[str, int] = (
        EXPECTED_UNSUPPORTED_RESOLVED_COUNTS_BY_SEED
    ),
    expected_counts_by_action: Mapping[str, int] = (
        EXPECTED_UNSUPPORTED_RESOLVED_COUNTS_BY_ACTION
    ),
    expected_counts_by_reason: Mapping[str, int] = (
        EXPECTED_UNSUPPORTED_RESOLVED_COUNTS_BY_REASON
    ),
    target_seeds: Sequence[int] = TARGET_SEEDS,
    v191_report_override: Mapping[str, object] | None = None,
    v190_report_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    v191_report = (
        dict(v191_report_override)
        if v191_report_override is not None
        else load_json_report(v191_report_path)
    )
    v190_report = (
        dict(v190_report_override)
        if v190_report_override is not None
        else load_json_report(v190_report_path)
    )
    source_validation = validate_v192_sources(
        v191_report,
        v190_report,
        expected_v191_report_exact_digest=expected_v191_report_exact_digest,
        expected_v190_report_exact_digest=expected_v190_report_exact_digest,
        required_v191_route=required_v191_route,
        expected_unsupported_requested_action_count=(
            expected_unsupported_requested_action_count
        ),
        expected_unsupported_resolved_action_count=(
            expected_unsupported_resolved_action_count
        ),
        expected_counts_by_seed=expected_counts_by_seed,
        expected_counts_by_action=expected_counts_by_action,
        expected_counts_by_reason=expected_counts_by_reason,
        target_seeds=target_seeds,
    )
    movement_audit = movement_target_blocker_audit(
        v190_report,
        expected_unsupported_resolved_action_count=(
            expected_unsupported_resolved_action_count
        ),
    )
    repair = repair_scope_decision_audit(
        source_validation=source_validation,
        movement_audit=movement_audit,
    )
    route = route_decision_audit(
        source_validation=source_validation,
        movement_audit=movement_audit,
        repair=repair,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V192_ACTION_RESOLUTION_CONTRACT_REPAIR_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V192_ACTION_RESOLUTION_CONTRACT_REPAIR_POLICY,
        "contract": contract(
            expected_v191_report_exact_digest=expected_v191_report_exact_digest,
            required_v191_route=required_v191_route,
        ),
        "inputs": {
            "v191_report": str(v191_report_path),
            "v190_report": str(v190_report_path),
            "expected_v191_report_exact_digest": expected_v191_report_exact_digest,
            "expected_v190_report_exact_digest": expected_v190_report_exact_digest,
            "required_v191_route": required_v191_route,
            "v191_backup": (
                "gdrive:evolution-sim-backups/archives/"
                "20260613T122723Z-v191-legal-support-repair-architecture-review.tar.zst"
            ),
            "v191_backup_sha256": (
                "bf0184dc2caa6be860723a4b0d6daa35b43264560117dc0794b45c8ce45fdff3"
            ),
            "output": str(output_path),
        },
        "source_validation": source_validation,
        "movement_target_blocker_audit": movement_audit,
        "repair_scope_decision": repair,
        "route_decision": route,
        "classification": {
            "primary": classification_for(
                source_validation=source_validation,
                route_decision=route,
            ),
            "labels": [
                "diagnostics_only",
                "action_resolution_contract_repair",
                "support_evidence_contract_repair",
                "no_runtime_semantics_change",
                "no_slice_3_training",
                "no_promotion",
            ],
        },
        **lifecycle_flags(),
    }
    report["classification"]["labels"].insert(0, report["classification"]["primary"])
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v192_sources(
    v191_report: Mapping[str, object],
    v190_report: Mapping[str, object],
    *,
    expected_v191_report_exact_digest: str,
    expected_v190_report_exact_digest: str,
    required_v191_route: str,
    expected_unsupported_requested_action_count: int,
    expected_unsupported_resolved_action_count: int,
    expected_counts_by_seed: Mapping[str, int],
    expected_counts_by_action: Mapping[str, int],
    expected_counts_by_reason: Mapping[str, int],
    target_seeds: Sequence[int],
) -> dict[str, object]:
    v191_digest = exact_digest_validation_report(v191_report)
    v190_digest = exact_digest_validation_report(v190_report)
    v191_source = _mapping(v191_report.get("source_validation"))
    v191_classification = _mapping(
        v191_report.get("unsupported_resolved_action_classification")
    )
    v191_root = _mapping(v191_classification.get("root_cause_assessment"))
    v191_route = _mapping(v191_report.get("route_decision"))
    v190_search = _mapping(v190_report.get("expansion_search"))
    expected_seed_counts = {str(key): int(value) for key, value in expected_counts_by_seed.items()}
    expected_action_counts = {
        str(key): int(value) for key, value in expected_counts_by_action.items()
    }
    expected_reason_counts = {
        str(key): int(value) for key, value in expected_counts_by_reason.items()
    }
    checks = {
        "v191_exact_digest_valid": v191_digest.get("passed") is True,
        "v191_exact_digest_matches_expected": (
            v191_report.get("exact_digest") == expected_v191_report_exact_digest
        ),
        "v191_schema_matches": (
            v191_report.get("schema_version")
            == v191.M3_CARRION_SURVIVOR_CONTINUATION_V191_LEGAL_SUPPORT_REPAIR_ARCHITECTURE_REVIEW_SCHEMA_VERSION
        ),
        "v191_policy_matches": (
            v191_report.get("policy")
            == v191.M3_CARRION_SURVIVOR_CONTINUATION_V191_LEGAL_SUPPORT_REPAIR_ARCHITECTURE_REVIEW_POLICY
        ),
        "v191_source_validation_passed": v191_source.get("passed") is True,
        "v191_route_matches_required": (
            v191_route.get("recommended_next_route") == required_v191_route
        ),
        "v191_exactly_one_route": (
            v191_route.get("exactly_one_next_route_recommended") is True
        ),
        "v191_root_cause_matches_expected": (
            v191_root.get("primary") == EXPECTED_ROOT_CAUSE
        ),
        "v191_unsupported_requested_matches_expected": (
            _int(v191_classification.get("v190_reported_unsupported_requested_action_count"))
            == int(expected_unsupported_requested_action_count)
        ),
        "v191_unsupported_resolved_matches_expected": (
            _int(v191_classification.get("v190_reported_unsupported_resolved_action_count"))
            == int(expected_unsupported_resolved_action_count)
            and _int(v191_classification.get("observed_unsupported_resolved_action_count"))
            == int(expected_unsupported_resolved_action_count)
        ),
        "v191_counts_by_seed_match_expected": (
            _int_dict(v191_classification.get("counts_by_seed")) == expected_seed_counts
        ),
        "v191_counts_by_action_match_expected": (
            _int_dict(v191_classification.get("counts_by_requested_action"))
            == expected_action_counts
        ),
        "v191_counts_by_reason_match_expected": (
            _int_dict(v191_classification.get("counts_by_legality_reason"))
            == expected_reason_counts
        ),
        "v191_counts_by_resolved_action_match_expected": (
            _int_dict(v191_classification.get("counts_by_resolved_action"))
            == EXPECTED_UNSUPPORTED_RESOLVED_COUNTS_BY_RESOLVED_ACTION
            if int(expected_unsupported_resolved_action_count)
            == EXPECTED_UNSUPPORTED_RESOLVED_ACTION_COUNT
            else True
        ),
        "v191_all_resolved_to_stay": (
            v191_classification.get("all_invalid_resolution_records_resolve_to_stay")
            is True
        ),
        "v191_all_observation_valid_resolution_invalid_movement": (
            v191_classification.get("all_invalid_resolution_records_are_observation_valid")
            is True
            and v191_classification.get(
                "all_invalid_resolution_records_flip_observation_true_to_resolution_false"
            )
            is True
            and v191_classification.get(
                "all_invalid_resolution_requested_actions_are_movement"
            )
            is True
        ),
        "v190_exact_digest_valid": v190_digest.get("passed") is True,
        "v190_exact_digest_matches_v191_pin": (
            v190_report.get("exact_digest")
            == v191_source.get("observed_v190_report_exact_digest")
            == expected_v190_report_exact_digest
        ),
        "v190_unsupported_requested_matches_expected": (
            _int(v190_search.get("unsupported_requested_action_count"))
            == int(expected_unsupported_requested_action_count)
        ),
        "v190_unsupported_resolved_matches_expected": (
            _int(v190_search.get("unsupported_resolved_action_count"))
            == int(expected_unsupported_resolved_action_count)
        ),
        "target_seeds_match_expected": (
            [int(seed) for seed in target_seeds] == list(TARGET_SEEDS)
            if tuple(target_seeds) == TARGET_SEEDS
            else True
        ),
    }
    for flag in FALSE_LIFECYCLE_FLAGS:
        checks[f"v191_{flag}_closed"] = v191_report.get(flag) is False
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v192_source_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v191_report_exact_digest": expected_v191_report_exact_digest,
        "observed_v191_report_exact_digest": v191_report.get("exact_digest"),
        "required_v191_route": required_v191_route,
        "observed_v191_route": v191_route.get("recommended_next_route"),
        "expected_v190_report_exact_digest": expected_v190_report_exact_digest,
        "observed_v190_report_exact_digest": v190_report.get("exact_digest"),
        "checks": checks,
        "v191_exact_digest_validation": v191_digest,
        "v190_exact_digest_validation": v190_digest,
    }


def movement_target_blocker_audit(
    v190_report: Mapping[str, object],
    *,
    expected_unsupported_resolved_action_count: int,
) -> dict[str, object]:
    branch_report = _mapping(_mapping(v190_report.get("expansion_search")).get("branch_report"))
    branch_runs = _mappings(branch_report.get("branch_runs"))
    events: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    missing_paths: list[str] = []
    read_errors: list[dict[str, str]] = []
    for run in branch_runs:
        path = Path(str(run.get("trajectory_path") or ""))
        run_events: list[dict[str, object]] = []
        if not path.exists():
            missing_paths.append(str(path))
        else:
            try:
                header, records = _trajectory_header_and_records(path)
                config = _mapping(header.get("config"))
                for record in records:
                    if record.get("resolution_action_valid") is not False:
                        continue
                    event = movement_target_blocker_event(run, record, config=config)
                    run_events.append(event)
                    events.append(event)
            except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
                read_errors.append({"path": str(path), "error": str(exc)})
        run_summaries.append(
            {
                "seed": _int(run.get("seed")),
                "branch_id": run.get("branch_id"),
                "continuation_script": run.get("continuation_script"),
                "trajectory_path": str(path),
                "reported_unsupported_resolved_action_count": _int(
                    run.get("unsupported_resolved_action_count")
                ),
                "observed_unsupported_resolved_action_count": len(run_events),
                "reported_unsupported_requested_action_count": _int(
                    run.get("unsupported_requested_action_count")
                ),
                "event_count_matches_report": (
                    len(run_events)
                    == _int(run.get("unsupported_resolved_action_count"))
                ),
            }
        )
    classification_counts = _count_by(events, "blocker_classification")
    all_expected = bool(events) and all(
        event.get("blocker_classification")
        == "resolution_invalid_same_tick_occupancy_race"
        for event in events
    )
    counts_match = len(events) == int(expected_unsupported_resolved_action_count)
    run_counts_match = all(
        summary.get("event_count_matches_report") is True for summary in run_summaries
    )
    sufficient = (
        all_expected
        and counts_match
        and run_counts_match
        and not missing_paths
        and not read_errors
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v192_movement_target_blocker_audit_v1",
        "diagnostics_only": True,
        "repair_contract_event_count": len(events),
        "expected_unsupported_resolved_action_count": int(
            expected_unsupported_resolved_action_count
        ),
        "event_count_matches_expected": counts_match,
        "run_event_counts_match_v190_run_summaries": run_counts_match,
        "trajectory_path_missing_count": len(missing_paths),
        "trajectory_path_missing": missing_paths,
        "trajectory_read_error_count": len(read_errors),
        "trajectory_read_errors": read_errors,
        "blocker_classification_counts": classification_counts,
        "counts_by_seed": _count_by(events, "seed"),
        "counts_by_requested_action": _count_by(events, "requested_action"),
        "counts_by_legality_reason": _count_by(events, "legality_reason"),
        "counts_by_target_in_bounds": _count_by(events, "target_in_bounds"),
        "all_resolution_invalid_events_have_movement_target_detail": all(
            event.get("movement_target_detail_available") is True for event in events
        ),
        "all_resolution_invalid_events_are_same_tick_occupancy_races": all_expected,
        "bounds_blocker_count": classification_counts.get("bounds_blocker", 0),
        "water_blocker_count": classification_counts.get("water_blocker", 0),
        "hazard_blocker_count": classification_counts.get("hazard_blocker", 0),
        "depleted_resource_blocker_count": classification_counts.get(
            "depleted_resource_blocker",
            0,
        ),
        "stale_or_illegal_script_mask_count": classification_counts.get(
            "stale_or_illegal_script_mask",
            0,
        ),
        "contract_repair_sufficient": sufficient,
        "support_evidence_contract_interpretation": (
            "Observation-valid movement that resolves invalid solely because a "
            "fresh same-tick resolution mask rejects an otherwise in-bounds movement "
            "target is expected deterministic action-order drift, not an illegal "
            "requested action. It must be reported separately and still must not "
            "be counted as a runtime move."
        ),
        "trajectory_schema_changed": False,
        "runtime_action_resolution_changed": False,
        "movement_target_blocker_events": events,
        "run_summaries": run_summaries,
    }


def movement_target_blocker_event(
    run: Mapping[str, object],
    record: Mapping[str, object],
    *,
    config: Mapping[str, object],
) -> dict[str, object]:
    requested = str(record.get("requested_action") or "")
    action_mask = _mapping(record.get("action_mask"))
    resolution_mask = _mapping(record.get("resolution_action_mask"))
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    outcome = _mapping(record.get("outcome"))
    dx, dy = MOVEMENT_DELTAS.get(requested, (0, 0))
    movement_action = requested in MOVEMENT_DELTAS
    from_x = _int(before.get("x"))
    from_y = _int(before.get("y"))
    target_x = from_x + dx if movement_action else None
    target_y = from_y + dy if movement_action else None
    width = _int(config.get("width"))
    height = _int(config.get("height"))
    target_in_bounds = (
        target_x is not None
        and target_y is not None
        and width > 0
        and height > 0
        and 0 <= target_x < width
        and 0 <= target_y < height
    )
    observation_valid = (
        record.get("action_valid") is True and action_mask.get(requested) is True
    )
    resolution_valid = (
        record.get("resolution_action_valid") is True
        and resolution_mask.get(requested) is True
    )
    resolved_action = str(record.get("resolved_action") or "")
    legality_reason = str(outcome.get("invalid_reason") or "unknown")
    blocker = classify_resolution_invalid_blocker(
        requested_action=requested,
        movement_action=movement_action,
        target_in_bounds=target_in_bounds,
        observation_valid=observation_valid,
        resolution_valid=resolution_valid,
        resolution_mask_allows=resolution_mask.get(requested) is True,
        resolved_action=resolved_action,
        legality_reason=legality_reason,
    )
    return {
        "seed": _int(run.get("seed")),
        "branch_id": str(run.get("branch_id") or ""),
        "continuation_script": str(run.get("continuation_script") or ""),
        "trajectory_path": str(run.get("trajectory_path") or ""),
        "tick": _int(record.get("tick")),
        "agent_id": _int(record.get("agent_id")),
        "requested_action": requested,
        "resolved_action": resolved_action,
        "legality_reason": legality_reason,
        "requested_action_valid_at_observation_time": observation_valid,
        "requested_action_valid_at_resolution_time": resolution_valid,
        "observation_mask_allows_requested_action": action_mask.get(requested) is True,
        "resolution_mask_allows_requested_action": resolution_mask.get(requested) is True,
        "movement_target_detail_available": movement_action and target_x is not None,
        "from_x": from_x if movement_action else None,
        "from_y": from_y if movement_action else None,
        "target_x": target_x,
        "target_y": target_y,
        "dx": dx if movement_action else None,
        "dy": dy if movement_action else None,
        "target_in_bounds": target_in_bounds,
        "after_x": _int(after.get("x")) if movement_action else None,
        "after_y": _int(after.get("y")) if movement_action else None,
        "agent_moved": record.get("moved") is True,
        "resolution_invalid_due_to_same_tick_occupancy_race": (
            blocker == "resolution_invalid_same_tick_occupancy_race"
        ),
        "blocker_classification": blocker,
        "blocker_detail": blocker_detail(
            blocker=blocker,
            movement_action=movement_action,
            observation_valid=observation_valid,
            target_in_bounds=target_in_bounds,
        ),
    }


def classify_resolution_invalid_blocker(
    *,
    requested_action: str,
    movement_action: bool,
    target_in_bounds: bool,
    observation_valid: bool,
    resolution_valid: bool,
    resolution_mask_allows: bool,
    resolved_action: str,
    legality_reason: str,
) -> str:
    if resolution_valid or resolution_mask_allows:
        return "not_resolution_invalid"
    if not movement_action:
        if requested_action in ("eat", "drink") and observation_valid:
            return "depleted_resource_blocker"
        return "non_movement_resolution_invalid"
    if not target_in_bounds:
        return "bounds_blocker"
    if not observation_valid:
        return "stale_or_illegal_script_mask"
    if resolved_action != "stay" or legality_reason != "not_in_resolution_action_mask":
        return "unexpected_resolution_path"
    return "resolution_invalid_same_tick_occupancy_race"


def blocker_detail(
    *,
    blocker: str,
    movement_action: bool,
    observation_valid: bool,
    target_in_bounds: bool,
) -> dict[str, object]:
    same_tick = blocker == "resolution_invalid_same_tick_occupancy_race"
    return {
        "bounds_blocker": blocker == "bounds_blocker",
        "water_blocker": False,
        "hazard_blocker": False,
        "depleted_resource_blocker": blocker == "depleted_resource_blocker",
        "stale_or_illegal_script_mask": blocker == "stale_or_illegal_script_mask",
        "same_tick_occupancy_race": same_tick,
        "water_excluded_by_observation_movement_mask": (
            same_tick and movement_action and observation_valid
        ),
        "bounds_excluded_by_target_coordinate": same_tick and target_in_bounds,
        "hazard_excluded_by_runtime_movement_mask_contract": same_tick,
        "depleted_resource_excluded_by_action_family": same_tick and movement_action,
        "runtime_target_occupant_id_serialized": False,
        "runtime_target_terrain_serialized": False,
        "evidence_note": (
            "v190 trajectories do not serialize live target occupant or terrain. "
            "v192 distinguishes blockers by action family, before-state target "
            "coordinates, observation-mask legality, resolution-mask legality, and "
            "the runtime movement mask contract: movement availability depends on "
            "bounds, non-water terrain, and empty occupancy, while hazards do not "
            "disable movement."
        ),
    }


def repair_scope_decision_audit(
    *,
    source_validation: Mapping[str, object],
    movement_audit: Mapping[str, object],
) -> dict[str, object]:
    sufficient = (
        source_validation.get("passed") is True
        and movement_audit.get("contract_repair_sufficient") is True
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v192_repair_scope_decision_v1",
        "diagnostics_only": True,
        "selected_repair_type": "support_evidence_contract_repair",
        "selected_repair_type_is_exactly_one": True,
        "contract_repair_sufficient": sufficient,
        "support_evidence_contract_repair": {
            "selected": True,
            "definition": (
                "Future support audits may separate deterministic same-tick "
                "occupancy drift from illegal requested actions when the request was "
                "valid at observation time, invalid at resolution time, movement-only, "
                "in bounds, resolved to stay, and explicitly classified by this audit."
            ),
            "count_as_runtime_move": False,
            "count_as_illegal_requested_action": False,
            "still_requires_terminal_survival_and_action_share_cap": True,
        },
        "trajectory_diagnostics_serialization_repair": {
            "selected": False,
            "reason": (
                "v192 serializes derived movement target/blocker audit fields in the "
                "diagnostic report. It does not change the runtime trajectory schema "
                "or replay payload."
            ),
        },
        "narrow_simulator_replay_contract_repair": {
            "selected": False,
            "reason": (
                "Accepting stale movement would alter deterministic same-tick action "
                "ordering. Existing runtime tests intentionally record this as "
                "observation-valid but resolution-invalid."
            ),
        },
    }


def route_decision_audit(
    *,
    source_validation: Mapping[str, object],
    movement_audit: Mapping[str, object],
    repair: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[str] = []
    if source_validation.get("passed") is not True:
        blockers.append("v191_source_validation_failed")
    if movement_audit.get("event_count_matches_expected") is not True:
        blockers.append("movement_event_count_mismatch")
    if movement_audit.get("all_resolution_invalid_events_are_same_tick_occupancy_races") is not True:
        blockers.append("non_occupancy_resolution_invalid_event_present")
    if movement_audit.get("trajectory_path_missing_count") != 0:
        blockers.append("trajectory_path_missing")
    if movement_audit.get("trajectory_read_error_count") != 0:
        blockers.append("trajectory_read_error")
    if source_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif repair.get("contract_repair_sufficient") is True and not blockers:
        route = NEXT_ROUTE_AFTER_REPAIR
    else:
        route = NEXT_ROUTE_ARCHITECTURE_REVIEW
    return {
        "policy": "m3_carrion_survivor_continuation_v192_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "slice_3_training_authorized": False,
        "slice_3_training_allowed_for_this_command": False,
        "slice_3_training_consumed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "primary_blocker": blockers[0] if blockers else "contract_repair_complete",
        "rationale": route_rationale(route),
    }


def route_rationale(route: str) -> str:
    if route == STOP_ROUTE:
        return "Pinned v191 source evidence did not validate; stop without v192 claims."
    if route == NEXT_ROUTE_AFTER_REPAIR:
        return (
            "All pinned unsupported resolved actions are observation-valid movement "
            "requests that become resolution-invalid only through deterministic "
            "same-tick occupancy drift. The support-evidence contract repair is "
            "sufficient; rerun targeted legal support expansion with this repaired "
            "contract before any slice-3 training."
        )
    return (
        "At least one unsupported resolved event is not safely explained by the "
        "same-tick occupancy contract; route to an action-resolution architecture "
        "review with no training."
    )


def classification_for(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = (
        "m3_carrion_survivor_continuation_v192_action_resolution_contract_repair_"
    )
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if route_decision.get("recommended_next_route") == NEXT_ROUTE_AFTER_REPAIR:
        return prefix + "support_evidence_contract_repaired_routes_to_fresh_support_no_training"
    return prefix + "architecture_review_route_no_training"


def contract(
    *,
    expected_v191_report_exact_digest: str,
    required_v191_route: str,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "selected_repair_type": "support_evidence_contract_repair",
        "runtime_action_resolution_changed": False,
        "trajectory_schema_changed": False,
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "input_v191_report_exact_digest_pinned": expected_v191_report_exact_digest,
        "required_v191_route": required_v191_route,
        "allowed_next_routes": [
            NEXT_ROUTE_AFTER_REPAIR,
            NEXT_ROUTE_ARCHITECTURE_REVIEW,
            STOP_ROUTE,
        ],
        "forbidden_routes": [
            "slice_3_training",
            "runtime_integration",
            "runtime_action_selection_change",
            "promotion",
            "gate_relaxation",
            "blind_support_expansion",
        ],
    }


def lifecycle_flags() -> dict[str, object]:
    return {
        "training_ran": False,
        "training_artifact_created": False,
        "fit_ran": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "trajectory_schema_changed": False,
        "simulator_replay_semantics_changed": False,
        "shadow_eval_ran": False,
        "live_ab_ran": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "gate_relaxation_ran": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "v190_rerun": False,
        "v191_rerun": False,
        "non_promoted": True,
        "diagnostics_only": True,
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def _trajectory_header_and_records(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    header: dict[str, object] = {}
    records: list[dict[str, object]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            if row.get("type") == "header":
                header = dict(row)
                continue
            if row.get("type") != "record":
                continue
            record = row.get("record")
            if isinstance(record, Mapping):
                records.append(dict(record))
    return header, records


def _mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _count_by(events: Sequence[Mapping[str, object]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter(str(event.get(key)) for event in events)
    return dict(sorted(counts.items()))


def _int_dict(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _int(raw) for key, raw in sorted(value.items())}
