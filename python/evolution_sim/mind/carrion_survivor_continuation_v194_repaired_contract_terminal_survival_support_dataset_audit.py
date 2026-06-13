from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import gzip
import json
import re
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v192_action_resolution_contract_repair as v192,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair as v193,
)
from evolution_sim.mind.candidate_campaign import _float, _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V194_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V194_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V194_COMPACT_SUPPORT_ROW_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v194_compact_terminal_survival_support_row_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v194-carrion-survivor-continuation-repaired-contract-terminal-survival-support-dataset-audit.json"
)
DEFAULT_COMPACT_SUPPORT_DATASET_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v194-carrion-survivor-continuation-repaired-contract-terminal-survival-support-compact-dataset.jsonl"
)
DEFAULT_V193_REPORT_PATH = v193.DEFAULT_OUTPUT_PATH
DEFAULT_BACKUP_DOC_PATHS = (
    Path("docs/mind-v3-autonomous-evolution.md"),
    Path("AGENTS.md"),
)

EXPECTED_V193_REPORT_EXACT_DIGEST = (
    "006a1bf33a4c092a9ae584603cddb434b868f21493cde8f710c9d816c66c1a21"
)
EXPECTED_V192_REPORT_EXACT_DIGEST = v193.EXPECTED_V192_REPORT_EXACT_DIGEST
EXPECTED_V191_REPORT_EXACT_DIGEST = v193.EXPECTED_V191_REPORT_EXACT_DIGEST
EXPECTED_V190_REPORT_EXACT_DIGEST = v193.EXPECTED_V190_REPORT_EXACT_DIGEST
REQUIRED_V193_ROUTE = v193.SUCCESS_ROUTE
EXPECTED_V193_BACKUP = (
    "gdrive:evolution-sim-backups/archives/"
    "20260613T165955Z-v193-fresh-targeted-legal-support-expansion-after-contract-repair.tar.zst"
)
EXPECTED_V193_BACKUP_SHA256 = (
    "ceaeb7c2027551393adb1d1374d7f5fd5621aadc3374275992a28b002da0b964"
)
EXPECTED_V193_RCLONE_DIFFERENCES = 0
EXPECTED_V193_RCLONE_MATCHING_FILES = 1

SUCCESS_ROUTE = (
    "v195_repaired_contract_terminal_survival_support_training_slice_3_opt_in"
)

DEFAULT_EXPECTED_AGGREGATE_SUPPORT = {
    "clean_support_seed_count": 6,
    "target_seed_count": 6,
    "dominant_requested_action": "stay",
    "dominant_requested_action_share": 0.207558,
    "aggregate_unsupported_requested_action_count": 0,
    "aggregate_expected_same_tick_occupancy_drift_count": 860,
    "aggregate_unexpected_resolution_invalid_count": 0,
}
DEFAULT_EXPECTED_SELECTED_SUPPORT_BY_SEED = {
    "13": {
        "alive_agents": 4,
        "births": 12,
        "dominant_requested_action": "eat",
        "dominant_requested_action_share": 0.2078,
        "expected_same_tick_occupancy_drift_count": 9,
    },
    "19": {
        "alive_agents": 2,
        "births": 10,
        "dominant_requested_action": "move_north",
        "dominant_requested_action_share": 0.2430,
        "expected_same_tick_occupancy_drift_count": 14,
    },
    "29": {
        "alive_agents": 3,
        "births": 12,
        "dominant_requested_action": "stay",
        "dominant_requested_action_share": 0.2400,
        "expected_same_tick_occupancy_drift_count": 23,
    },
    "37": {
        "alive_agents": 3,
        "births": 10,
        "dominant_requested_action": "stay",
        "dominant_requested_action_share": 0.2655,
        "expected_same_tick_occupancy_drift_count": 7,
    },
    "41": {
        "alive_agents": 2,
        "births": 13,
        "dominant_requested_action": "stay",
        "dominant_requested_action_share": 0.2123,
        "expected_same_tick_occupancy_drift_count": 10,
    },
    "43": {
        "alive_agents": 5,
        "births": 12,
        "dominant_requested_action": "stay",
        "dominant_requested_action_share": 0.2047,
        "expected_same_tick_occupancy_drift_count": 14,
    },
}

FALSE_V193_LIFECYCLE_FLAGS = (
    "training_ran",
    "training_artifact_created",
    "slice_3_training_consumed",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "promotion_authorized",
    "gate_relaxation_allowed",
)


def run_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit(
    *,
    v193_report_path: str | Path = DEFAULT_V193_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    compact_support_dataset_output_path: str
    | Path = DEFAULT_COMPACT_SUPPORT_DATASET_OUTPUT_PATH,
    expected_v193_report_exact_digest: str = EXPECTED_V193_REPORT_EXACT_DIGEST,
    expected_v192_report_exact_digest: str = EXPECTED_V192_REPORT_EXACT_DIGEST,
    expected_v191_report_exact_digest: str = EXPECTED_V191_REPORT_EXACT_DIGEST,
    expected_v190_report_exact_digest: str = EXPECTED_V190_REPORT_EXACT_DIGEST,
    required_v193_route: str = REQUIRED_V193_ROUTE,
    expected_selected_support_by_seed: Mapping[str, Mapping[str, object]] = (
        DEFAULT_EXPECTED_SELECTED_SUPPORT_BY_SEED
    ),
    expected_aggregate_support: Mapping[str, object] = (
        DEFAULT_EXPECTED_AGGREGATE_SUPPORT
    ),
    backup_doc_paths: Sequence[str | Path] = DEFAULT_BACKUP_DOC_PATHS,
    backup_metadata_override: Mapping[str, object] | None = None,
    create_compact_support_dataset: bool = True,
    v193_report_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    v193_report = (
        dict(v193_report_override)
        if v193_report_override is not None
        else load_json_report(v193_report_path)
    )
    source_validation = validate_v194_v193_source(
        v193_report,
        expected_v193_report_exact_digest=expected_v193_report_exact_digest,
        expected_v192_report_exact_digest=expected_v192_report_exact_digest,
        expected_v191_report_exact_digest=expected_v191_report_exact_digest,
        expected_v190_report_exact_digest=expected_v190_report_exact_digest,
        required_v193_route=required_v193_route,
        expected_aggregate_support=expected_aggregate_support,
        backup_doc_paths=backup_doc_paths,
        backup_metadata_override=backup_metadata_override,
    )
    selected_support = selected_support_facts_audit(
        v193_report,
        expected_selected_support_by_seed=expected_selected_support_by_seed,
        expected_aggregate_support=expected_aggregate_support,
    )
    dataset_rows: list[dict[str, object]] = []
    dataset_write = skipped_dataset_write("compact_support_dataset_disabled")
    if (
        create_compact_support_dataset
        and source_validation.get("passed") is True
        and selected_support.get("passed") is True
    ):
        dataset_rows = build_compact_support_dataset_rows(
            selected_support.get("selected_run_audits", []),
            source_report_path=v193_report_path,
            source_report_digest=expected_v193_report_exact_digest,
        )
        _write_jsonl(compact_support_dataset_output_path, dataset_rows)
        dataset_write = {
            "policy": "m3_carrion_survivor_continuation_v194_compact_support_dataset_write_v1",
            "created": True,
            "path": str(compact_support_dataset_output_path),
            "row_count": len(dataset_rows),
            "dataset_digest": stable_payload_digest(dataset_rows),
            "gitignored_output_mind_path": str(compact_support_dataset_output_path).startswith(
                "output/mind/"
            ),
        }
    elif create_compact_support_dataset:
        dataset_write = skipped_dataset_write("source_or_selected_support_audit_failed")

    dataset_audit = compact_support_dataset_audit(
        dataset_rows,
        dataset_write=dataset_write,
        selected_support=selected_support,
        expected_v193_report_exact_digest=expected_v193_report_exact_digest,
    )
    route_decision = route_decision_audit(
        source_validation=source_validation,
        selected_support=selected_support,
        dataset_audit=dataset_audit,
    )
    classification = classification_for(
        source_validation=source_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V194_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V194_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_POLICY
        ),
        "contract": contract(
            expected_v193_report_exact_digest=expected_v193_report_exact_digest,
            expected_v192_report_exact_digest=expected_v192_report_exact_digest,
            expected_v191_report_exact_digest=expected_v191_report_exact_digest,
            expected_v190_report_exact_digest=expected_v190_report_exact_digest,
            required_v193_route=required_v193_route,
        ),
        "inputs": {
            "v193_report": str(v193_report_path),
            "expected_v193_report_exact_digest": expected_v193_report_exact_digest,
            "expected_v192_report_exact_digest": expected_v192_report_exact_digest,
            "expected_v191_report_exact_digest": expected_v191_report_exact_digest,
            "expected_v190_report_exact_digest": expected_v190_report_exact_digest,
            "required_v193_route": required_v193_route,
            "v193_backup": EXPECTED_V193_BACKUP,
            "v193_backup_sha256": EXPECTED_V193_BACKUP_SHA256,
            "output": str(output_path),
            "compact_support_dataset_output": str(compact_support_dataset_output_path),
        },
        "source_validation": source_validation,
        "selected_support_facts_audit": selected_support,
        "compact_support_dataset": dataset_write,
        "dataset_audit": dataset_audit,
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "repaired_contract_terminal_survival_support_dataset_audit",
                "compact_support_dataset",
                "no_training",
                "no_slice_3_consumption",
                "no_runtime_integration",
                "no_promotion",
            ],
        },
        **lifecycle_flags(dataset_created=dataset_write.get("created") is True),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v194_v193_source(
    v193_report: Mapping[str, object],
    *,
    expected_v193_report_exact_digest: str,
    expected_v192_report_exact_digest: str,
    expected_v191_report_exact_digest: str,
    expected_v190_report_exact_digest: str,
    required_v193_route: str,
    expected_aggregate_support: Mapping[str, object],
    backup_doc_paths: Sequence[str | Path],
    backup_metadata_override: Mapping[str, object] | None,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v193_report)
    route = _mapping(v193_report.get("route_decision"))
    source = _mapping(v193_report.get("source_validation"))
    support = _mapping(v193_report.get("repaired_contract_support_audit"))
    backup = (
        dict(backup_metadata_override)
        if backup_metadata_override is not None
        else backup_metadata_audit(backup_doc_paths)
    )
    checks = {
        "v193_schema_matches": (
            v193_report.get("schema_version")
            == v193.M3_CARRION_SURVIVOR_CONTINUATION_V193_FRESH_TARGETED_LEGAL_SUPPORT_EXPANSION_AFTER_CONTRACT_REPAIR_SCHEMA_VERSION
        ),
        "v193_policy_matches": (
            v193_report.get("policy")
            == v193.M3_CARRION_SURVIVOR_CONTINUATION_V193_FRESH_TARGETED_LEGAL_SUPPORT_EXPANSION_AFTER_CONTRACT_REPAIR_POLICY
        ),
        "v193_exact_digest_valid": exact_validation.get("passed") is True,
        "v193_exact_digest_matches_expected": (
            v193_report.get("exact_digest") == expected_v193_report_exact_digest
        ),
        "v193_route_matches_required": (
            route.get("recommended_next_route") == required_v193_route
        ),
        "v193_source_validation_passed": source.get("passed") is True,
        "v193_v192_digest_pin_matches_expected": (
            source.get("observed_v192_report_exact_digest")
            == expected_v192_report_exact_digest
        ),
        "v193_v191_digest_pin_matches_expected": (
            source.get("observed_v191_report_exact_digest")
            == expected_v191_report_exact_digest
        ),
        "v193_v190_digest_pin_matches_expected": (
            source.get("observed_v190_report_exact_digest")
            == expected_v190_report_exact_digest
        ),
        "v193_backup_metadata_validated": backup.get("passed") is True,
        "v193_repaired_contract_support_passed": support.get("passed") is True,
        "v193_support_coverage_is_6_of_6": (
            _int(support.get("clean_support_seed_count"))
            == _int(expected_aggregate_support.get("clean_support_seed_count"))
            == _int(support.get("target_seed_count"))
            == _int(expected_aggregate_support.get("target_seed_count"))
        ),
        "v193_unsupported_requested_actions_zero": (
            _int(support.get("aggregate_unsupported_requested_action_count"))
            == _int(
                expected_aggregate_support.get(
                    "aggregate_unsupported_requested_action_count"
                )
            )
        ),
        "v193_expected_same_tick_occupancy_drift_matches_expected": (
            _int(support.get("aggregate_expected_same_tick_occupancy_drift_count"))
            == _int(
                expected_aggregate_support.get(
                    "aggregate_expected_same_tick_occupancy_drift_count"
                )
            )
        ),
        "v193_unexpected_resolution_invalid_zero": (
            _int(support.get("aggregate_unexpected_resolution_invalid_count"))
            == _int(
                expected_aggregate_support.get(
                    "aggregate_unexpected_resolution_invalid_count"
                )
            )
        ),
        "v193_expected_drift_not_counted_as_successful_move": (
            support.get("aggregate_expected_occupancy_drift_counted_as_successful_move")
            is False
        ),
        "v193_aggregate_dominant_action_matches_expected": (
            support.get("dominant_requested_action")
            == expected_aggregate_support.get("dominant_requested_action")
        ),
        "v193_aggregate_dominant_action_share_matches_expected": _float_matches(
            support.get("dominant_requested_action_share"),
            expected_aggregate_support.get("dominant_requested_action_share"),
        ),
        "v193_support_expansion_ran_as_prior_diagnostic": (
            v193_report.get("support_expansion_ran") is True
        ),
        "v193_support_generation_ran_as_prior_diagnostic": (
            v193_report.get("support_generation_ran") is True
        ),
    }
    for flag in FALSE_V193_LIFECYCLE_FLAGS:
        checks[f"v193_{flag}_closed"] = v193_report.get(flag) is False
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v194_v193_source_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v193_report_exact_digest": expected_v193_report_exact_digest,
        "observed_v193_report_exact_digest": v193_report.get("exact_digest"),
        "expected_v192_report_exact_digest": expected_v192_report_exact_digest,
        "observed_v192_report_exact_digest": source.get(
            "observed_v192_report_exact_digest"
        ),
        "expected_v191_report_exact_digest": expected_v191_report_exact_digest,
        "observed_v191_report_exact_digest": source.get(
            "observed_v191_report_exact_digest"
        ),
        "expected_v190_report_exact_digest": expected_v190_report_exact_digest,
        "observed_v190_report_exact_digest": source.get(
            "observed_v190_report_exact_digest"
        ),
        "required_v193_route": required_v193_route,
        "observed_v193_route": route.get("recommended_next_route"),
        "v193_exact_digest_validation": exact_validation,
        "v193_backup_metadata": backup,
        "checks": checks,
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
        "backup_path_recorded": EXPECTED_V193_BACKUP in text,
        "archive_sha256_recorded": EXPECTED_V193_BACKUP_SHA256 in text,
        "rclone_check_recorded": "rclone check" in text,
        "rclone_zero_differences_recorded": (
            f"{EXPECTED_V193_RCLONE_DIFFERENCES}` differences" in text
            or f"{EXPECTED_V193_RCLONE_DIFFERENCES} differences" in text
        ),
        "rclone_matching_file_count_recorded": (
            f"{EXPECTED_V193_RCLONE_MATCHING_FILES}` matching file" in text
            or f"{EXPECTED_V193_RCLONE_MATCHING_FILES} matching file" in text
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v194_v193_backup_metadata_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "readable_paths": readable,
        "missing_paths": missing,
        "expected_v193_backup": EXPECTED_V193_BACKUP,
        "expected_v193_backup_sha256": EXPECTED_V193_BACKUP_SHA256,
        "expected_rclone_differences": EXPECTED_V193_RCLONE_DIFFERENCES,
        "expected_rclone_matching_files": EXPECTED_V193_RCLONE_MATCHING_FILES,
        "checks": checks,
    }


def selected_support_facts_audit(
    v193_report: Mapping[str, object],
    *,
    expected_selected_support_by_seed: Mapping[str, Mapping[str, object]],
    expected_aggregate_support: Mapping[str, object],
) -> dict[str, object]:
    support = _mapping(v193_report.get("repaired_contract_support_audit"))
    per_seed = _mapping(support.get("per_seed"))
    run_audits = _mappings(support.get("run_audits"))
    selected: list[dict[str, object]] = []
    per_seed_audits: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for seed in sorted(expected_selected_support_by_seed, key=lambda item: int(item)):
        expected = dict(expected_selected_support_by_seed[seed])
        seed_summary = _mapping(per_seed.get(seed))
        best = _mapping(seed_summary.get("best_repaired_contract_attempt"))
        full_run = _matching_run_audit(best, run_audits)
        audit = selected_seed_support_audit(seed, full_run, expected)
        per_seed_audits[seed] = audit
        if audit.get("passed") is not True:
            failures.append(f"seed_{seed}_selected_support_fact_mismatch")
        if audit.get("selected_run_audit"):
            selected.append(dict(_mapping(audit.get("selected_run_audit"))))

    aggregate = selected_support_aggregate_audit(
        support,
        selected,
        expected_aggregate_support=expected_aggregate_support,
    )
    if aggregate.get("passed") is not True:
        failures.append("aggregate_selected_support_mismatch")
    return {
        "policy": "m3_carrion_survivor_continuation_v194_selected_support_facts_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "selected_seed_count": len(selected),
        "expected_selected_seed_count": len(expected_selected_support_by_seed),
        "selected_run_audits": selected,
        "per_seed": per_seed_audits,
        "aggregate": aggregate,
        "unsupported_requested_action_count": aggregate.get(
            "aggregate_unsupported_requested_action_count"
        ),
        "expected_same_tick_occupancy_drift_count": aggregate.get(
            "aggregate_expected_same_tick_occupancy_drift_count"
        ),
        "selected_expected_same_tick_occupancy_drift_count": aggregate.get(
            "selected_expected_same_tick_occupancy_drift_count"
        ),
        "unexpected_resolution_invalid_count": aggregate.get(
            "aggregate_unexpected_resolution_invalid_count"
        ),
        "dominant_requested_action": aggregate.get("dominant_requested_action"),
        "dominant_requested_action_share": aggregate.get(
            "dominant_requested_action_share"
        ),
    }


def selected_seed_support_audit(
    seed: str,
    run: Mapping[str, object],
    expected: Mapping[str, object],
) -> dict[str, object]:
    path = Path(str(run.get("trajectory_path") or ""))
    trajectory = v193.repaired_contract_trajectory_audit(path, run)
    schema = selected_trajectory_schema_audit(path, run)
    checks = {
        "selected_run_present": bool(run),
        "counts_as_repaired_contract_support": (
            run.get("counts_as_repaired_contract_support") is True
        ),
        "alive_agents_matches_expected": (
            _int(run.get("alive_agents")) == _int(expected.get("alive_agents"))
        ),
        "births_matches_expected": (
            _int(run.get("births")) == _int(expected.get("births"))
        ),
        "dominant_requested_action_matches_expected": (
            run.get("dominant_requested_action")
            == expected.get("dominant_requested_action")
        ),
        "dominant_requested_action_share_matches_expected": _float_matches(
            run.get("dominant_requested_action_share"),
            expected.get("dominant_requested_action_share"),
        ),
        "expected_same_tick_occupancy_drift_matches_expected": (
            _int(run.get("expected_same_tick_occupancy_drift_count"))
            == _int(expected.get("expected_same_tick_occupancy_drift_count"))
        ),
        "unsupported_requested_actions_zero": (
            _int(run.get("unsupported_requested_action_count")) == 0
            and _int(trajectory.get("unsupported_requested_count")) == 0
        ),
        "unexpected_resolution_invalid_zero": (
            _int(run.get("unexpected_resolution_invalid_count")) == 0
            and _int(trajectory.get("unexpected_resolution_invalid_count")) == 0
        ),
        "expected_drift_counted_separately": (
            _int(trajectory.get("expected_occupancy_drift_moved_count")) == 0
        ),
        "path_exists": path.exists(),
        "trajectory_readable": trajectory.get("readable") is True,
        "replay_verified": run.get("replay_verified") is True,
        "terminal_facts_match_manifest": (
            trajectory.get("terminal_facts_match_manifest") is True
        ),
        "record_count_matches_footer": (
            trajectory.get("record_count_matches_footer") is True
        ),
        "action_mask_legality_passed_under_repaired_contract": (
            trajectory.get("action_mask_legality_passed_under_repaired_contract")
            is True
        ),
        "trajectory_invalid_counts_match_repaired_contract": (
            trajectory.get("trajectory_invalid_counts_match_repaired_contract")
            is True
        ),
        "observation_schema_audit_passed": schema.get("observation_passed") is True,
        "target_audit_passed": schema.get("target_passed") is True,
        "trainable_leakage_scan_passed": (
            trajectory.get("trainable_leakage_scan_passed") is True
            and schema.get("trainable_leakage_passed") is True
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v194_selected_seed_support_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "seed": _int(seed),
        "expected": dict(expected),
        "observed": {
            "alive_agents": _int(run.get("alive_agents")),
            "births": _int(run.get("births")),
            "dominant_requested_action": run.get("dominant_requested_action"),
            "dominant_requested_action_share": run.get(
                "dominant_requested_action_share"
            ),
            "expected_same_tick_occupancy_drift_count": _int(
                run.get("expected_same_tick_occupancy_drift_count")
            ),
            "unsupported_requested_action_count": _int(
                run.get("unsupported_requested_action_count")
            ),
            "unexpected_resolution_invalid_count": _int(
                run.get("unexpected_resolution_invalid_count")
            ),
            "trajectory_path": str(path),
        },
        "selected_run_audit": dict(run),
        "trajectory_audit": trajectory,
        "trajectory_schema_audit": schema,
        "checks": checks,
    }


def selected_trajectory_schema_audit(
    path: Path,
    run: Mapping[str, object],
) -> dict[str, object]:
    try:
        rows = _read_jsonl(path)
    except (OSError, EOFError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "policy": "m3_carrion_survivor_continuation_v194_selected_trajectory_schema_audit_v1",
            "readable": False,
            "error": str(exc),
            "observation_passed": False,
            "target_passed": False,
            "trainable_leakage_passed": False,
            "failures": ["trajectory_unreadable"],
        }
    records = [
        _mapping(row.get("record"))
        for row in rows
        if row.get("type") == "record" and isinstance(row.get("record"), Mapping)
    ]
    observation_failures: list[dict[str, object]] = []
    target_failures: list[dict[str, object]] = []
    leakage_failures: list[dict[str, object]] = []
    for record_index, record in enumerate(records):
        observation = _mapping(record.get("observation_input"))
        action_mask = _mapping(record.get("action_mask"))
        requested = str(record.get("requested_action") or "")
        if observation.get("schema_version") != "mind_observation_v3":
            observation_failures.append(
                {"record_index": record_index, "reason": "observation_schema_mismatch"}
            )
        if not observation.get("data") or not isinstance(
            observation.get("shape"), Sequence
        ):
            observation_failures.append(
                {"record_index": record_index, "reason": "observation_payload_incomplete"}
            )
        if not requested:
            target_failures.append(
                {"record_index": record_index, "reason": "requested_action_missing"}
            )
        if requested and action_mask.get(requested) is not True:
            target_failures.append(
                {
                    "record_index": record_index,
                    "reason": "requested_action_not_in_current_public_action_mask",
                    "requested_action": requested,
                }
            )
        candidate_payload = {
            "current_public_observation": observation,
            "current_public_action_mask": action_mask,
        }
        v193.scan_trainable_key_leakage(
            candidate_payload,
            failures=leakage_failures,
            path=("candidate_trainable_payload",),
            record_index=record_index,
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v194_selected_trajectory_schema_audit_v1",
        "readable": True,
        "record_count": len(records),
        "observation_passed": not observation_failures,
        "observation_failure_count": len(observation_failures),
        "observation_failures": observation_failures[:32],
        "target_passed": not target_failures,
        "target_failure_count": len(target_failures),
        "target_failures": target_failures[:32],
        "trainable_leakage_passed": not leakage_failures,
        "trainable_leakage_failure_count": len(leakage_failures),
        "trainable_leakage_failures": leakage_failures[:32],
        "terminal_alive_agents": _int(run.get("alive_agents")),
        "terminal_births": _int(run.get("births")),
        "terminal_survival_positive": _int(run.get("alive_agents")) > 0,
    }


def selected_support_aggregate_audit(
    support: Mapping[str, object],
    selected_runs: Sequence[Mapping[str, object]],
    *,
    expected_aggregate_support: Mapping[str, object],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    for run in selected_runs:
        counts.update(
            {
                str(action): _int(count)
                for action, count in _mapping(run.get("requested_action_counts")).items()
            }
        )
    selected_dominant = _dominant_count_share(counts)
    checks = {
        "clean_support_seed_count_matches_expected": (
            _int(support.get("clean_support_seed_count"))
            == _int(expected_aggregate_support.get("clean_support_seed_count"))
        ),
        "target_seed_count_matches_expected": (
            _int(support.get("target_seed_count"))
            == _int(expected_aggregate_support.get("target_seed_count"))
        ),
        "aggregate_unsupported_requested_action_count_matches_expected": (
            _int(support.get("aggregate_unsupported_requested_action_count"))
            == _int(
                expected_aggregate_support.get(
                    "aggregate_unsupported_requested_action_count"
                )
            )
        ),
        "aggregate_expected_same_tick_occupancy_drift_count_matches_expected": (
            _int(support.get("aggregate_expected_same_tick_occupancy_drift_count"))
            == _int(
                expected_aggregate_support.get(
                    "aggregate_expected_same_tick_occupancy_drift_count"
                )
            )
        ),
        "aggregate_unexpected_resolution_invalid_count_matches_expected": (
            _int(support.get("aggregate_unexpected_resolution_invalid_count"))
            == _int(
                expected_aggregate_support.get(
                    "aggregate_unexpected_resolution_invalid_count"
                )
            )
        ),
        "dominant_requested_action_matches_expected": (
            support.get("dominant_requested_action")
            == expected_aggregate_support.get("dominant_requested_action")
        ),
        "dominant_requested_action_share_matches_expected": _float_matches(
            support.get("dominant_requested_action_share"),
            expected_aggregate_support.get("dominant_requested_action_share"),
        ),
        "selected_requested_action_counts_match_report_dominant": (
            selected_dominant.get("key") == support.get("dominant_requested_action")
            and _float_matches(
                selected_dominant.get("share"),
                support.get("dominant_requested_action_share"),
            )
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v194_selected_support_aggregate_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "clean_support_seed_count": _int(support.get("clean_support_seed_count")),
        "target_seed_count": _int(support.get("target_seed_count")),
        "aggregate_unsupported_requested_action_count": _int(
            support.get("aggregate_unsupported_requested_action_count")
        ),
        "aggregate_expected_same_tick_occupancy_drift_count": _int(
            support.get("aggregate_expected_same_tick_occupancy_drift_count")
        ),
        "selected_expected_same_tick_occupancy_drift_count": sum(
            _int(run.get("expected_same_tick_occupancy_drift_count"))
            for run in selected_runs
        ),
        "aggregate_unexpected_resolution_invalid_count": _int(
            support.get("aggregate_unexpected_resolution_invalid_count")
        ),
        "dominant_requested_action": support.get("dominant_requested_action"),
        "dominant_requested_action_share": support.get(
            "dominant_requested_action_share"
        ),
        "selected_requested_action_counts": dict(sorted(counts.items())),
        "selected_recomputed_dominant_requested_action": selected_dominant.get("key"),
        "selected_recomputed_dominant_requested_action_share": selected_dominant.get(
            "share"
        ),
        "checks": checks,
    }


def build_compact_support_dataset_rows(
    selected_runs: object,
    *,
    source_report_path: str | Path,
    source_report_digest: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in _mappings(selected_runs):
        path = Path(str(run.get("trajectory_path") or ""))
        trajectory_rows = _read_jsonl(path)
        header = next(
            (row for row in trajectory_rows if row.get("type") == "header"),
            {},
        )
        config = _mapping(_mapping(header).get("config"))
        records = [
            _mapping(row.get("record"))
            for row in trajectory_rows
            if row.get("type") == "record" and isinstance(row.get("record"), Mapping)
        ]
        for record_index, record in enumerate(records):
            rows.append(
                compact_support_dataset_row(
                    run=run,
                    record=record,
                    record_index=record_index,
                    config=config,
                    source_report_path=source_report_path,
                    source_report_digest=source_report_digest,
                )
            )
    return rows


def compact_support_dataset_row(
    *,
    run: Mapping[str, object],
    record: Mapping[str, object],
    record_index: int,
    config: Mapping[str, object],
    source_report_path: str | Path,
    source_report_digest: str,
) -> dict[str, object]:
    requested = str(record.get("requested_action") or "")
    resolution = repaired_resolution_payload(run=run, record=record, config=config)
    return {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_V194_COMPACT_SUPPORT_ROW_SCHEMA_VERSION,
        "row_origin": (
            "v194_repaired_contract_terminal_survival_support_dataset_audit"
        ),
        "source": {
            "source_v193_report": str(source_report_path),
            "source_v193_report_exact_digest": source_report_digest,
            "trajectory_path": str(run.get("trajectory_path") or ""),
            "trajectory_record_index": int(record_index),
            "source_record_digest": stable_payload_digest(record),
        },
        "metadata": {
            "seed": _int(run.get("seed")),
            "branch_id": str(run.get("branch_id") or ""),
            "continuation_script": str(run.get("continuation_script") or ""),
            "trajectory_path": str(run.get("trajectory_path") or ""),
            "source_identity_used_as_trainable_input": False,
            "fixture_identity_used_as_trainable_input": False,
            "private_world_state_used_as_trainable_input": False,
            "current_or_future_outcome_used_as_trainable_input": False,
            "diagnostic_target_used_as_trainable_input": False,
        },
        "trainable_public_features": {
            "current_public_observation": record.get("observation_input"),
            "current_public_action_mask": record.get("action_mask"),
        },
        "supervised_target": {
            "requested_action": requested,
            "requested_action_valid_in_current_public_action_mask": (
                _mapping(record.get("action_mask")).get(requested) is True
                and record.get("action_valid") is True
            ),
            "terminal_survival_support": _int(run.get("alive_agents")) > 0,
        },
        "repaired_resolution": resolution,
        "terminal_support_summary": {
            "alive_agents": _int(run.get("alive_agents")),
            "births": _int(run.get("births")),
            "deaths": _int(run.get("deaths")),
            "dominant_requested_action": run.get("dominant_requested_action"),
            "dominant_requested_action_share": run.get(
                "dominant_requested_action_share"
            ),
        },
    }


def repaired_resolution_payload(
    *,
    run: Mapping[str, object],
    record: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    requested = str(record.get("requested_action") or "")
    resolved = str(record.get("resolved_action") or requested)
    movement_action = requested in v192.MOVEMENT_DELTAS
    if record.get("resolution_action_valid") is False:
        event = v192.movement_target_blocker_event(run, record, config=config)
        classification = str(event.get("blocker_classification") or "")
        expected_drift = classification == "resolution_invalid_same_tick_occupancy_race"
        return {
            "requested_action": requested,
            "resolved_action": resolved,
            "resolution_action_valid": False,
            "classification": classification,
            "expected_same_tick_occupancy_drift_counted_separately": expected_drift,
            "expected_same_tick_occupancy_drift_counted_as_successful_move": False,
            "counts_as_successful_movement": False,
            "movement_action": movement_action,
            "movement_target_blocker_event": event,
        }
    return {
        "requested_action": requested,
        "resolved_action": resolved,
        "resolution_action_valid": True,
        "classification": "valid_resolution",
        "expected_same_tick_occupancy_drift_counted_separately": False,
        "expected_same_tick_occupancy_drift_counted_as_successful_move": False,
        "counts_as_successful_movement": (
            movement_action and record.get("moved") is True
        ),
        "movement_action": movement_action,
        "movement_target_blocker_event": None,
    }


def compact_support_dataset_audit(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset_write: Mapping[str, object],
    selected_support: Mapping[str, object],
    expected_v193_report_exact_digest: str,
) -> dict[str, object]:
    if dataset_write.get("created") is not True:
        return {
            "policy": "m3_carrion_survivor_continuation_v194_compact_support_dataset_audit_v1",
            "passed": False,
            "skipped": True,
            "reason": dataset_write.get("reason"),
            "dataset_created": False,
            "dataset_ready_for_future_slice_3_training": False,
        }
    schema = compact_dataset_schema_audit(rows)
    leakage = compact_dataset_leakage_audit(rows)
    action_mask = compact_dataset_action_mask_audit(rows)
    repaired_resolution = compact_dataset_repaired_resolution_audit(
        rows,
        expected_selected_drift_count=_int(
            selected_support.get("selected_expected_same_tick_occupancy_drift_count")
        ),
    )
    observation = compact_dataset_observation_audit(rows)
    target = compact_dataset_target_audit(rows)
    source = compact_dataset_source_audit(
        rows,
        expected_v193_report_exact_digest=expected_v193_report_exact_digest,
    )
    checks = {
        "source_check_passed": source.get("passed") is True,
        "schema_check_passed": schema.get("passed") is True,
        "leakage_check_passed": leakage.get("passed") is True,
        "action_mask_check_passed": action_mask.get("passed") is True,
        "repaired_resolution_check_passed": repaired_resolution.get("passed") is True,
        "observation_check_passed": observation.get("passed") is True,
        "target_check_passed": target.get("passed") is True,
        "dataset_digest_matches_rows": (
            dataset_write.get("dataset_digest") == stable_payload_digest(rows)
        ),
        "dataset_row_count_positive": len(rows) > 0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v194_compact_support_dataset_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "dataset_created": True,
        "dataset_path": dataset_write.get("path"),
        "dataset_digest": dataset_write.get("dataset_digest"),
        "row_count": len(rows),
        "selected_support_seed_count": selected_support.get("selected_seed_count"),
        "dataset_ready_for_future_slice_3_training": not failures,
        "source_check": source,
        "schema_check": schema,
        "leakage_check": leakage,
        "action_mask_check": action_mask,
        "repaired_resolution_check": repaired_resolution,
        "observation_check": observation,
        "target_check": target,
        "checks": checks,
    }


def compact_dataset_source_audit(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_v193_report_exact_digest: str,
) -> dict[str, object]:
    failures = [
        index
        for index, row in enumerate(rows)
        if _mapping(row.get("source")).get("source_v193_report_exact_digest")
        != expected_v193_report_exact_digest
    ]
    return {
        "policy": "m3_carrion_survivor_continuation_v194_dataset_source_check_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failure_indices": failures[:32],
        "expected_v193_report_exact_digest": expected_v193_report_exact_digest,
        "unique_trajectory_count": len(
            {str(_mapping(row.get("source")).get("trajectory_path") or "") for row in rows}
        ),
    }


def compact_dataset_schema_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    required = (
        "schema_version",
        "row_origin",
        "source",
        "metadata",
        "trainable_public_features",
        "supervised_target",
        "repaired_resolution",
        "terminal_support_summary",
    )
    for index, row in enumerate(rows):
        if row.get("schema_version") != (
            M3_CARRION_SURVIVOR_CONTINUATION_V194_COMPACT_SUPPORT_ROW_SCHEMA_VERSION
        ):
            failures.append({"row_index": index, "reason": "schema_version_mismatch"})
        for key in required:
            if key not in row:
                failures.append({"row_index": index, "reason": f"missing_{key}"})
    return {
        "policy": "m3_carrion_survivor_continuation_v194_dataset_schema_check_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:32],
        "row_count": len(rows),
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_V194_COMPACT_SUPPORT_ROW_SCHEMA_VERSION,
    }


def compact_dataset_leakage_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    allowed_trainable_keys = {"current_public_observation", "current_public_action_mask"}
    for row_index, row in enumerate(rows):
        trainable = _mapping(row.get("trainable_public_features"))
        extra_keys = sorted(set(str(key) for key in trainable) - allowed_trainable_keys)
        if extra_keys:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "unexpected_trainable_public_feature_key",
                    "keys": extra_keys,
                }
            )
        v193.scan_trainable_key_leakage(
            trainable,
            failures=failures,
            path=("dataset_row", str(row_index), "trainable_public_features"),
            record_index=row_index,
        )
        metadata = _mapping(row.get("metadata"))
        for key in (
            "source_identity_used_as_trainable_input",
            "fixture_identity_used_as_trainable_input",
            "private_world_state_used_as_trainable_input",
            "current_or_future_outcome_used_as_trainable_input",
            "diagnostic_target_used_as_trainable_input",
        ):
            if metadata.get(key) is not False:
                failures.append(
                    {
                        "row_index": row_index,
                        "reason": "metadata_leakage_flag_not_false",
                        "key": key,
                    }
                )
    return {
        "policy": "m3_carrion_survivor_continuation_v194_dataset_leakage_check_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
        "future_trainable_payload_policy": (
            "current_public_observation_and_current_public_action_mask_only"
        ),
        "fixture_identity_or_private_world_state_in_trainable_payload": bool(failures),
    }


def compact_dataset_action_mask_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        trainable = _mapping(row.get("trainable_public_features"))
        mask = _mapping(trainable.get("current_public_action_mask"))
        target = _mapping(row.get("supervised_target"))
        requested = str(target.get("requested_action") or "")
        if not mask:
            failures.append({"row_index": index, "reason": "action_mask_missing"})
        if requested and mask.get(requested) is not True:
            failures.append(
                {
                    "row_index": index,
                    "reason": "requested_action_not_valid_in_current_public_mask",
                    "requested_action": requested,
                }
            )
        if target.get("requested_action_valid_in_current_public_action_mask") is not True:
            failures.append(
                {
                    "row_index": index,
                    "reason": "target_requested_action_valid_flag_false",
                }
            )
    return {
        "policy": "m3_carrion_survivor_continuation_v194_dataset_action_mask_check_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def compact_dataset_repaired_resolution_audit(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_selected_drift_count: int,
) -> dict[str, object]:
    classification_counts: Counter[str] = Counter()
    expected_drift_as_move_count = 0
    unexpected_resolution_invalid_count = 0
    unsupported_requested_count = 0
    for row in rows:
        target = _mapping(row.get("supervised_target"))
        if target.get("requested_action_valid_in_current_public_action_mask") is not True:
            unsupported_requested_count += 1
        resolution = _mapping(row.get("repaired_resolution"))
        classification = str(resolution.get("classification") or "")
        classification_counts.update([classification])
        if classification == "resolution_invalid_same_tick_occupancy_race":
            if resolution.get("counts_as_successful_movement") is True:
                expected_drift_as_move_count += 1
            if (
                resolution.get(
                    "expected_same_tick_occupancy_drift_counted_separately"
                )
                is not True
            ):
                unexpected_resolution_invalid_count += 1
        elif resolution.get("resolution_action_valid") is False:
            unexpected_resolution_invalid_count += 1
    expected_drift = _int(
        classification_counts.get("resolution_invalid_same_tick_occupancy_race")
    )
    checks = {
        "unsupported_requested_actions_zero": unsupported_requested_count == 0,
        "unexpected_resolution_invalid_count_zero": (
            unexpected_resolution_invalid_count == 0
        ),
        "expected_same_tick_occupancy_drift_count_matches_selected_support": (
            expected_drift == int(expected_selected_drift_count)
        ),
        "expected_same_tick_occupancy_drift_counted_separately": (
            expected_drift >= 0 and unexpected_resolution_invalid_count == 0
        ),
        "expected_same_tick_occupancy_drift_never_successful_move": (
            expected_drift_as_move_count == 0
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v194_dataset_repaired_resolution_check_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "unsupported_requested_action_count": unsupported_requested_count,
        "expected_same_tick_occupancy_drift_count": expected_drift,
        "expected_same_tick_occupancy_drift_counted_as_successful_move_count": (
            expected_drift_as_move_count
        ),
        "unexpected_resolution_invalid_count": unexpected_resolution_invalid_count,
        "classification_counts": dict(sorted(classification_counts.items())),
        "checks": checks,
    }


def compact_dataset_observation_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        trainable = _mapping(row.get("trainable_public_features"))
        observation = _mapping(trainable.get("current_public_observation"))
        if observation.get("schema_version") != "mind_observation_v3":
            failures.append(
                {"row_index": index, "reason": "observation_schema_mismatch"}
            )
        if not observation.get("data"):
            failures.append({"row_index": index, "reason": "observation_data_missing"})
        if not isinstance(observation.get("shape"), Sequence):
            failures.append({"row_index": index, "reason": "observation_shape_missing"})
    return {
        "policy": "m3_carrion_survivor_continuation_v194_dataset_observation_check_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def compact_dataset_target_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        target = _mapping(row.get("supervised_target"))
        terminal = _mapping(row.get("terminal_support_summary"))
        if not target.get("requested_action"):
            failures.append({"row_index": index, "reason": "requested_action_missing"})
        if target.get("terminal_survival_support") is not True:
            failures.append(
                {"row_index": index, "reason": "terminal_survival_support_false"}
            )
        if _int(terminal.get("alive_agents")) <= 0:
            failures.append(
                {"row_index": index, "reason": "terminal_alive_not_positive"}
            )
    return {
        "policy": "m3_carrion_survivor_continuation_v194_dataset_target_check_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def route_decision_audit(
    *,
    source_validation: Mapping[str, object],
    selected_support: Mapping[str, object],
    dataset_audit: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[str] = []
    if source_validation.get("passed") is not True:
        blockers.append("v193_source_validation_failed")
    if selected_support.get("passed") is not True:
        blockers.append("selected_support_facts_audit_failed")
    if dataset_audit.get("passed") is not True:
        blockers.append("compact_support_dataset_audit_failed")
    route = SUCCESS_ROUTE if not blockers else repair_route_for(blockers[0])
    return {
        "policy": "m3_carrion_survivor_continuation_v194_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "future_explicit_slice_3_training_route_recommended": not blockers,
        "future_explicit_slice_3_training_route_authorized": not blockers,
        "slice_3_training_allowed_for_this_command": False,
        "slice_3_training_consumed": False,
        "training_ran": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "primary_blocker": blockers[0] if blockers else None,
        "dataset_digest": dataset_audit.get("dataset_digest"),
        "dataset_path": dataset_audit.get("dataset_path"),
        "clean_support_seed_count": selected_support.get("selected_seed_count"),
        "dominant_requested_action": selected_support.get("dominant_requested_action"),
        "dominant_requested_action_share": selected_support.get(
            "dominant_requested_action_share"
        ),
        "rationale": route_rationale(route=route, blockers=blockers),
    }


def repair_route_for(blocker: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", str(blocker).lower()).strip("_")
    return (
        "v195_repaired_contract_terminal_survival_support_dataset_repair_"
        f"{safe}_no_training"
    )


def route_rationale(*, route: str, blockers: Sequence[str]) -> str:
    if route == SUCCESS_ROUTE:
        return (
            "The repaired-contract terminal-survival support dataset passes source, "
            "schema, leakage, action-mask, repaired-resolution, observation, and "
            "target checks. Recommend exactly one future explicit opt-in slice-3 "
            "training route; this command did not train."
        )
    return (
        "At least one repaired-contract support dataset audit check failed. "
        f"Route to no-training repair for first blocker: {blockers[0]}."
    )


def classification_for(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = (
        "m3_carrion_survivor_continuation_v194_repaired_contract_terminal_"
        "survival_support_dataset_audit_"
    )
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_routes_to_no_training_repair"
    if route_decision.get("recommended_next_route") == SUCCESS_ROUTE:
        return prefix + "support_dataset_ready_for_future_explicit_slice_3_training"
    return prefix + "blocked_routes_to_no_training_repair"


def contract(
    *,
    expected_v193_report_exact_digest: str,
    expected_v192_report_exact_digest: str,
    expected_v191_report_exact_digest: str,
    expected_v190_report_exact_digest: str,
    required_v193_route: str,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "input_v193_report_exact_digest_pinned": expected_v193_report_exact_digest,
        "input_v192_report_exact_digest_pinned": expected_v192_report_exact_digest,
        "input_v191_report_exact_digest_pinned": expected_v191_report_exact_digest,
        "input_v190_report_exact_digest_pinned": expected_v190_report_exact_digest,
        "required_v193_route": required_v193_route,
        "expected_v193_backup": EXPECTED_V193_BACKUP,
        "expected_v193_backup_sha256": EXPECTED_V193_BACKUP_SHA256,
        "success_route": SUCCESS_ROUTE,
        "repaired_support_evidence_contract": {
            "unsupported_requested_actions_block_support": True,
            "expected_same_tick_occupancy_drift_counted_separately": True,
            "expected_same_tick_occupancy_drift_counted_as_successful_move": False,
            "unexpected_resolution_invalid_blocks_support": True,
            "future_trainable_payload_uses_only_public_observation_and_action_mask": True,
            "fixture_identity_private_world_state_not_trainable": True,
        },
    }


def lifecycle_flags(*, dataset_created: bool) -> dict[str, object]:
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
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "compact_support_dataset_created": bool(dataset_created),
        "generic_carrion_autopsy_rerun": False,
        "v180_rerun": False,
        "v186_rerun": False,
        "v188_rerun": False,
        "v189_rerun": False,
        "v190_rerun": False,
        "v191_rerun": False,
        "v192_rerun": False,
        "v193_rerun": False,
        "non_promoted": True,
        "diagnostics_only": True,
    }


def skipped_dataset_write(reason: str) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v194_compact_support_dataset_write_v1",
        "created": False,
        "reason": reason,
        "path": None,
        "row_count": 0,
        "dataset_digest": None,
        "gitignored_output_mind_path": True,
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def _matching_run_audit(
    best: Mapping[str, object],
    run_audits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    for run in run_audits:
        if (
            _int(run.get("seed")) == _int(best.get("seed"))
            and str(run.get("branch_id") or "") == str(best.get("branch_id") or "")
            and str(run.get("continuation_script") or "")
            == str(best.get("continuation_script") or "")
            and str(run.get("trajectory_path") or "")
            == str(best.get("trajectory_path") or "")
        ):
            return dict(run)
    return dict(best)


def _mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _float_matches(observed: object, expected: object, *, tolerance: float = 1e-6) -> bool:
    return abs(_float(observed) - _float(expected)) <= float(tolerance)


def _dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = max(counts.items(), key=lambda item: (int(item[1]), str(item[0])))
    return {"key": key, "count": int(count), "share": _round(count / total)}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
