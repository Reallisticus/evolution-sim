from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v188_terminal_carrion_survival_support as v188,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit as v189,
)
from evolution_sim.mind.candidate_campaign import _float, _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_branch_explore import (
    DEFAULT_CARRION_BRANCH_BASE_SCRIPT,
    MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
    build_carrion_branch_explore_report,
)
from evolution_sim.mind.carrion_counterfactual import DEFAULT_COUNTERFACTUAL_SCRIPTS
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V190_TARGETED_LEGAL_TERMINAL_SURVIVAL_SUPPORT_EXPANSION_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V190_TARGETED_LEGAL_TERMINAL_SURVIVAL_SUPPORT_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v190-carrion-survivor-continuation-targeted-legal-terminal-survival-support-expansion.json"
)
DEFAULT_SUPPORT_TRAJECTORY_DIR = Path(
    "output/mind/v190-targeted-legal-terminal-survival-support-trajectories"
)
DEFAULT_V189_REPORT_PATH = v189.DEFAULT_OUTPUT_PATH

EXPECTED_V189_REPORT_EXACT_DIGEST = (
    "2505049a4a81f17ca5fb79f004c9d4ea7f43b1169f9e461e6a722df04e7d369e"
)
EXPECTED_V189_REQUIRED_ROUTE = v189.TARGETED_SUPPORT_EXPANSION_ROUTE
EXPECTED_V189_BACKUP = (
    "gdrive:evolution-sim-backups/archives/"
    "20260612T175835Z-v189-terminal-survival-support-dataset-audit.tar.zst"
)
EXPECTED_V189_BACKUP_SHA256 = (
    "c9a347d1504e2222473f3db2d49f2f0c5afe21c61ff877730c1fefe12045ff92"
)

EXPECTED_V189_BLOCKERS = (
    "legal_terminal_support_target_seed_coverage_insufficient",
    "selected_support_dominant_requested_action_share_above_cap",
    "aggregate_attempted_continuations_rejected_as_support",
)
EXPECTED_V189_AGGREGATE_UNSUPPORTED_RESOLVED_ACTIONS = 285
EXPECTED_V189_LEGAL_POSITIVE_TARGET_SEED_COUNT = 1
EXPECTED_V189_TARGET_SEED_COUNT = 6
EXPECTED_V189_SELECTED_SUPPORT_DOMINANT_REQUESTED_ACTION_SHARE = 0.539244

DEFAULT_CARRION_FIXTURE_SEEDS = v188.DEFAULT_CARRION_FIXTURE_SEEDS
DEFAULT_TICKS = 120
DEFAULT_BASE_SCRIPT = DEFAULT_CARRION_BRANCH_BASE_SCRIPT
DEFAULT_CONTINUATION_SCRIPTS = DEFAULT_COUNTERFACTUAL_SCRIPTS
DEFAULT_MAX_BRANCH_POINTS_PER_SEED = 3
DEFAULT_MIN_BRANCH_TICK = 0
DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE = 0.50
DEFAULT_METADATA_DOC_PATHS = (
    Path("AGENTS.md"),
    Path("docs/mind-v3-autonomous-evolution.md"),
    Path("docs/repository-audit-2026-06-10-remediation.md"),
)

SUCCESS_ROUTE = "v191_terminal_survival_support_dataset_audit_before_slice_3_training"
BLOCKED_ROUTE = "v191_targeted_legal_support_repair_or_architecture_review_no_training"
STOP_ROUTE = "stop"

FALSE_SOURCE_LIFECYCLE_FLAGS = (
    "training_ran",
    "training_artifact_created",
    "slice_3_training_consumed",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "promotion_authorized",
    "gate_relaxation_allowed",
    "support_expansion_ran",
)


def run_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion(
    *,
    v189_report_path: str | Path = DEFAULT_V189_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    support_trajectory_dir: str | Path = DEFAULT_SUPPORT_TRAJECTORY_DIR,
    expected_v189_report_exact_digest: str = EXPECTED_V189_REPORT_EXACT_DIGEST,
    required_v189_route: str = EXPECTED_V189_REQUIRED_ROUTE,
    seeds: Sequence[int] = DEFAULT_CARRION_FIXTURE_SEEDS,
    ticks: int = DEFAULT_TICKS,
    base_script: str = DEFAULT_BASE_SCRIPT,
    continuation_scripts: Sequence[str] = DEFAULT_CONTINUATION_SCRIPTS,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    min_branch_tick: int = DEFAULT_MIN_BRANCH_TICK,
    max_dominant_requested_action_share: float = (
        DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE
    ),
    verify_replay: bool = True,
    v189_report_override: Mapping[str, object] | None = None,
    branch_report_override: Mapping[str, object] | None = None,
    metadata_doc_paths: Sequence[str | Path] = DEFAULT_METADATA_DOC_PATHS,
) -> dict[str, object]:
    v189_report = (
        dict(v189_report_override)
        if v189_report_override is not None
        else load_json_report(v189_report_path)
    )
    source_validation = validate_v190_v189_source(
        v189_report,
        expected_v189_report_exact_digest=expected_v189_report_exact_digest,
        required_v189_route=required_v189_route,
        metadata_doc_paths=metadata_doc_paths,
    )
    mined_evidence = mine_v189_evidence(
        v189_report,
        max_dominant_requested_action_share=max_dominant_requested_action_share,
    )
    target_manifest = build_target_manifest(
        mined_evidence,
        fallback_target_seeds=seeds,
    )
    search_budget = build_search_budget(
        target_manifest=target_manifest,
        ticks=ticks,
        base_script=base_script,
        continuation_scripts=continuation_scripts,
        max_branch_points_per_seed=max_branch_points_per_seed,
        min_branch_tick=min_branch_tick,
        verify_replay=verify_replay,
        support_trajectory_dir=support_trajectory_dir,
    )

    if source_validation.get("passed") is not True:
        branch_report = None
        expansion_search = skipped_expansion_search(
            search_budget,
            reason="source_validation_failed",
        )
    elif branch_report_override is not None:
        branch_report = dict(branch_report_override)
        expansion_search = summarize_expansion_search(branch_report, search_budget)
    else:
        branch_report = build_carrion_branch_explore_report(
            seeds=tuple(_int(seed) for seed in target_manifest["targeted_expansion_seeds"]),
            ticks=int(ticks),
            fixture_name="carrion_only",
            base_script=str(base_script),
            continuation_scripts=tuple(str(script) for script in continuation_scripts),
            max_branch_points_per_seed=int(max_branch_points_per_seed),
            min_branch_tick=int(min_branch_tick),
            trajectory_output_dir=Path(support_trajectory_dir),
            verify_replay=bool(verify_replay),
        )
        expansion_search = summarize_expansion_search(branch_report, search_budget)

    support_audit = legal_support_audit(
        branch_report=branch_report,
        target_manifest=target_manifest,
        max_dominant_requested_action_share=max_dominant_requested_action_share,
        replay_verification_required=verify_replay,
    )
    route_decision = route_decision_audit(
        source_validation=source_validation,
        mined_evidence=mined_evidence,
        support_audit=support_audit,
    )
    classification = classification_for(
        source_validation=source_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V190_TARGETED_LEGAL_TERMINAL_SURVIVAL_SUPPORT_EXPANSION_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V190_TARGETED_LEGAL_TERMINAL_SURVIVAL_SUPPORT_EXPANSION_POLICY
        ),
        "contract": contract(
            expected_v189_report_exact_digest=expected_v189_report_exact_digest,
            required_v189_route=required_v189_route,
            max_dominant_requested_action_share=max_dominant_requested_action_share,
        ),
        "inputs": {
            "v189_report": str(v189_report_path),
            "expected_v189_report_exact_digest": expected_v189_report_exact_digest,
            "required_v189_route_path": "route_decision.recommended_next_route",
            "required_v189_route": required_v189_route,
            "v189_backup": EXPECTED_V189_BACKUP,
            "v189_backup_sha256": EXPECTED_V189_BACKUP_SHA256,
            "output": str(output_path),
            "support_trajectory_dir": str(support_trajectory_dir),
        },
        "source_validation": source_validation,
        "mined_v188_v189_evidence": mined_evidence,
        "target_manifest": target_manifest,
        "search_budget": search_budget,
        "expansion_search": expansion_search,
        "legal_support_audit": support_audit,
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "targeted_legal_terminal_survival_support_expansion",
                "no_slice_3_training",
                "no_runtime_integration",
                "no_promotion",
            ],
        },
        **lifecycle_flags(support_expansion_ran=expansion_search.get("ran") is True),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v190_v189_source(
    v189_report: Mapping[str, object],
    *,
    expected_v189_report_exact_digest: str = EXPECTED_V189_REPORT_EXACT_DIGEST,
    required_v189_route: str = EXPECTED_V189_REQUIRED_ROUTE,
    metadata_doc_paths: Sequence[str | Path] = DEFAULT_METADATA_DOC_PATHS,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(v189_report)
    route = _mapping(v189_report.get("route_decision"))
    coverage = _mapping(v189_report.get("support_coverage_audit"))
    aggregate = _mapping(v189_report.get("aggregate_attempted_continuation_audit"))
    selected = _mapping(v189_report.get("selected_support_trajectory_audit"))
    dedupe = _mapping(v189_report.get("historical_dedupe"))
    backup = backup_metadata_audit(metadata_doc_paths)
    observed_blockers = tuple(str(item) for item in route.get("blockers", []))
    checks = {
        "v189_schema_matches": (
            v189_report.get("schema_version")
            == v189.M3_CARRION_SURVIVOR_CONTINUATION_V189_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_SCHEMA_VERSION
        ),
        "v189_policy_matches": (
            v189_report.get("policy")
            == v189.M3_CARRION_SURVIVOR_CONTINUATION_V189_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_POLICY
        ),
        "v189_exact_digest_valid": exact_validation.get("passed") is True,
        "v189_exact_digest_matches_expected": (
            str(v189_report.get("exact_digest") or "")
            == str(expected_v189_report_exact_digest)
        ),
        "v189_route_matches_required": (
            str(route.get("recommended_next_route") or "") == str(required_v189_route)
        ),
        "v189_blockers_match_expected": observed_blockers == EXPECTED_V189_BLOCKERS,
        "v189_legal_support_coverage_matches_expected": (
            _int(coverage.get("legal_positive_target_seed_count"))
            == EXPECTED_V189_LEGAL_POSITIVE_TARGET_SEED_COUNT
            and _int(coverage.get("target_seed_count"))
            == EXPECTED_V189_TARGET_SEED_COUNT
        ),
        "v189_selected_support_dominant_share_matches_expected": (
            _float(selected.get("dominant_requested_action_share"))
            == EXPECTED_V189_SELECTED_SUPPORT_DOMINANT_REQUESTED_ACTION_SHARE
        ),
        "v189_aggregate_unsupported_resolved_matches_expected": (
            _int(aggregate.get("unsupported_resolved_action_count"))
            == EXPECTED_V189_AGGREGATE_UNSUPPORTED_RESOLVED_ACTIONS
        ),
        "v189_aggregate_attempts_rejected_as_support": (
            aggregate.get("aggregate_attempted_continuations_rejected_as_support")
            is True
        ),
        "v189_historical_dedupe_passed": dedupe.get("passed") is True,
        "v189_old_lanes_not_rerun": (
            dedupe.get("rerun_old_feasibility_counterfactual_iql_or_scorer_loops")
            is False
        ),
        "v189_backup_metadata_recorded": backup.get("passed") is True,
        **{
            f"v189_{flag}_closed": v189_report.get(flag) is False
            for flag in FALSE_SOURCE_LIFECYCLE_FLAGS
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v190_v189_source_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v189_report_exact_digest": expected_v189_report_exact_digest,
        "observed_v189_report_exact_digest": v189_report.get("exact_digest"),
        "v189_exact_digest_validation": exact_validation,
        "required_v189_route": required_v189_route,
        "observed_v189_route": route.get("recommended_next_route"),
        "observed_v189_blockers": list(observed_blockers),
        "expected_v189_blockers": list(EXPECTED_V189_BLOCKERS),
        "v189_backup_metadata_audit": backup,
    }


def backup_metadata_audit(
    metadata_doc_paths: Sequence[str | Path] = DEFAULT_METADATA_DOC_PATHS,
) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for raw_path in metadata_doc_paths:
        path = Path(raw_path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            findings.append(
                {
                    "path": str(path),
                    "readable": False,
                    "error": str(exc),
                    "contains_backup": False,
                    "contains_archive_sha256": False,
                }
            )
            continue
        findings.append(
            {
                "path": str(path),
                "readable": True,
                "contains_backup": EXPECTED_V189_BACKUP in text,
                "contains_archive_sha256": EXPECTED_V189_BACKUP_SHA256 in text,
            }
        )
    matching_paths = [
        item["path"]
        for item in findings
        if item.get("contains_backup") is True
        and item.get("contains_archive_sha256") is True
    ]
    return {
        "policy": "m3_carrion_survivor_continuation_v190_v189_backup_metadata_audit_v1",
        "passed": bool(matching_paths),
        "expected_v189_backup": EXPECTED_V189_BACKUP,
        "expected_v189_backup_sha256": EXPECTED_V189_BACKUP_SHA256,
        "matching_doc_paths": matching_paths,
        "checked_doc_paths": findings,
    }


def mine_v189_evidence(
    v189_report: Mapping[str, object],
    *,
    max_dominant_requested_action_share: float,
) -> dict[str, object]:
    coverage = _mapping(v189_report.get("support_coverage_audit"))
    aggregate = _mapping(v189_report.get("aggregate_attempted_continuation_audit"))
    selected = _mapping(v189_report.get("selected_support_trajectory_audit"))
    selected_runs = _mappings(selected.get("run_audits"))
    selected_support = selected_runs[0] if selected_runs else {}
    selected_seed = _int(selected_support.get("seed"), default=13)
    selected_share = _float(selected.get("dominant_requested_action_share"))
    terminal_by_seed = {
        str(seed): _int(count)
        for seed, count in _mapping(coverage.get("terminal_survivors_by_seed")).items()
    }
    attempted_by_seed = {
        str(seed): _int(count)
        for seed, count in _mapping(
            aggregate.get("attempted_terminal_survivors_by_seed")
            or coverage.get("attempted_terminal_survivors_by_seed")
        ).items()
    }
    missing = sorted(int(seed) for seed, count in terminal_by_seed.items() if count <= 0)
    diversity_repair = (
        [selected_seed]
        if selected_support and selected_share > max_dominant_requested_action_share
        else []
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v190_v188_v189_evidence_mining_v1",
        "selected_v188_support": {
            "seed": selected_seed if selected_support else None,
            "branch_id": selected_support.get("branch_id"),
            "continuation_script": selected_support.get("continuation_script"),
            "terminal_alive_agents": selected_support.get("alive_agents"),
            "births": selected_support.get("births"),
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 0,
            "dominant_requested_action": selected.get("dominant_requested_action"),
            "dominant_requested_action_share": selected_share,
            "max_dominant_requested_action_share": _round(
                max_dominant_requested_action_share
            ),
            "counts_as_v190_clean_support": (
                bool(selected_support)
                and selected_share <= max_dominant_requested_action_share
            ),
            "v190_rejection_reason": (
                "dominant_requested_action_share_above_cap"
                if selected_support
                and selected_share > max_dominant_requested_action_share
                else None
            ),
        },
        "terminal_survivors_by_seed": terminal_by_seed,
        "attempted_terminal_survivors_by_seed": attempted_by_seed,
        "missing_legal_support_seeds": missing,
        "action_diversity_repair_seeds": diversity_repair,
        "aggregate_attempted_continuations": {
            "attempted_positive_seed_count": aggregate.get(
                "attempted_positive_seed_count"
            ),
            "unsupported_requested_action_count": aggregate.get(
                "unsupported_requested_action_count"
            ),
            "unsupported_resolved_action_count": aggregate.get(
                "unsupported_resolved_action_count"
            ),
            "rejected_as_support": aggregate.get(
                "aggregate_attempted_continuations_rejected_as_support"
            )
            is True,
            "diagnostic_leads_only": True,
        },
        "historical_dedupe_preserved": True,
    }


def build_target_manifest(
    mined_evidence: Mapping[str, object],
    *,
    fallback_target_seeds: Sequence[int],
) -> dict[str, object]:
    terminal = _mapping(mined_evidence.get("terminal_survivors_by_seed"))
    target_seeds = (
        sorted(int(seed) for seed in terminal)
        if terminal
        else sorted(int(seed) for seed in fallback_target_seeds)
    )
    missing = sorted(_int(seed) for seed in mined_evidence.get("missing_legal_support_seeds", []))
    diversity = sorted(
        _int(seed) for seed in mined_evidence.get("action_diversity_repair_seeds", [])
    )
    targeted = sorted(set(missing) | set(diversity))
    return {
        "policy": "m3_carrion_survivor_continuation_v190_target_manifest_v1",
        "fixture": "carrion_only",
        "horizon_ticks": DEFAULT_TICKS,
        "target_seeds": target_seeds,
        "target_seed_count": len(target_seeds),
        "missing_legal_support_seeds": missing,
        "action_diversity_repair_seeds": diversity,
        "targeted_expansion_seeds": targeted,
        "targeted_expansion_seed_count": len(targeted),
        "requires_clean_support_for_all_target_seeds": True,
        "requires_seed_13_action_diversity_repair": 13 in diversity,
        "aggregate_attempted_v188_trajectories_are_leads_only": True,
        "training_or_runtime_change_allowed": False,
    }


def build_search_budget(
    *,
    target_manifest: Mapping[str, object],
    ticks: int,
    base_script: str,
    continuation_scripts: Sequence[str],
    max_branch_points_per_seed: int,
    min_branch_tick: int,
    verify_replay: bool,
    support_trajectory_dir: str | Path,
) -> dict[str, object]:
    seeds = [_int(seed) for seed in target_manifest.get("targeted_expansion_seeds", [])]
    scripts = [str(script) for script in continuation_scripts]
    return {
        "policy": "m3_carrion_survivor_continuation_v190_search_budget_v1",
        "bounded": True,
        "open_ended_sweep": False,
        "fixture": "carrion_only",
        "horizon_ticks": int(ticks),
        "targeted_expansion_seeds": seeds,
        "targeted_expansion_seed_count": len(seeds),
        "base_branch_policy": str(base_script),
        "continuation_scripts": scripts,
        "script_count": len(scripts),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "min_branch_tick": int(min_branch_tick),
        "max_branch_points": int(max_branch_points_per_seed) * len(seeds),
        "max_branch_continuation_runs": (
            int(max_branch_points_per_seed) * len(seeds) * len(scripts)
        ),
        "branch_engine_schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_engine_policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "verify_replay": bool(verify_replay),
        "support_trajectory_dir": str(support_trajectory_dir),
        "uses_existing_exact_branch_replay_utility": True,
        "training_ran": False,
        "runtime_action_selection_changed": False,
    }


def skipped_expansion_search(
    search_budget: Mapping[str, object],
    *,
    reason: str,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v190_expansion_search_v1",
        "ran": False,
        "reason": reason,
        "search_budget": dict(search_budget),
        "branch_report_digest": None,
        "branch_report": None,
    }


def summarize_expansion_search(
    branch_report: Mapping[str, object],
    search_budget: Mapping[str, object],
) -> dict[str, object]:
    aggregate = _mapping(branch_report.get("aggregate"))
    return {
        "policy": "m3_carrion_survivor_continuation_v190_expansion_search_v1",
        "ran": True,
        "search_budget": dict(search_budget),
        "branch_report_digest": stable_payload_digest(branch_report),
        "branch_report_schema_version": branch_report.get("schema_version"),
        "branch_report_policy": branch_report.get("branch_policy"),
        "branch_point_count": _int(aggregate.get("branch_point_count")),
        "branch_run_count": _int(aggregate.get("branch_run_count")),
        "successful_branch_run_count": _int(
            aggregate.get("successful_branch_run_count")
        ),
        "positive_seed_count": _int(aggregate.get("positive_seed_count")),
        "replay_verified": aggregate.get("replay_verified") is True,
        "unsupported_requested_action_count": _int(
            aggregate.get("unsupported_requested_action_count")
        ),
        "unsupported_resolved_action_count": _int(
            aggregate.get("unsupported_resolved_action_count")
        ),
        "attempted_terminal_survivors_by_seed": _mapping(
            aggregate.get("terminal_survivor_count_by_seed")
        ),
        "trajectory_paths": list(aggregate.get("trajectory_paths") or []),
        "branch_report": dict(branch_report),
    }


def legal_support_audit(
    *,
    branch_report: Mapping[str, object] | None,
    target_manifest: Mapping[str, object],
    max_dominant_requested_action_share: float,
    replay_verification_required: bool,
) -> dict[str, object]:
    target_seeds = [_int(seed) for seed in target_manifest.get("target_seeds", [])]
    if branch_report is None:
        return {
            "policy": "m3_carrion_survivor_continuation_v190_legal_support_audit_v1",
            "passed": False,
            "reason": "expansion_search_not_run",
            "target_seed_count": len(target_seeds),
            "clean_legal_support_seed_count": 0,
            "clean_terminal_survivors_by_seed": {
                str(seed): 0 for seed in target_seeds
            },
            "support_runs": [],
            "support_trajectory_manifest": [],
            "per_seed": {
                str(seed): {"clean_support_run_count": 0, "blockers": ["not_searched"]}
                for seed in target_seeds
            },
            "blockers": ["expansion_search_not_run"],
        }

    branch_runs = _mappings(branch_report.get("branch_runs"))
    run_audits = [
        support_run_legality_audit(
            run,
            max_dominant_requested_action_share=max_dominant_requested_action_share,
            replay_verification_required=replay_verification_required,
        )
        for run in branch_runs
    ]
    clean_runs = [
        support_run_payload(_mappings(branch_report.get("branch_runs"))[index], audit)
        for index, audit in enumerate(run_audits)
        if audit.get("counts_as_clean_support") is True
    ]
    clean_by_seed: dict[str, int] = {str(seed): 0 for seed in target_seeds}
    terminal_alive_by_seed: dict[str, int] = {str(seed): 0 for seed in target_seeds}
    for run in clean_runs:
        seed_key = str(_int(run.get("seed")))
        clean_by_seed[seed_key] = clean_by_seed.get(seed_key, 0) + 1
        terminal_alive_by_seed[seed_key] = terminal_alive_by_seed.get(seed_key, 0) + _int(
            run.get("alive_agents")
        )
    per_seed = {
        str(seed): per_seed_support_summary(
            seed=seed,
            run_audits=run_audits,
            max_dominant_requested_action_share=max_dominant_requested_action_share,
        )
        for seed in target_seeds
    }
    missing = [
        seed for seed in target_seeds if _int(clean_by_seed.get(str(seed))) <= 0
    ]
    requested_counts: Counter[str] = Counter()
    for run in clean_runs:
        requested_counts.update(
            {
                str(action): _int(count)
                for action, count in _mapping(run.get("requested_action_counts")).items()
            }
        )
    dominant = dominant_count_share(requested_counts)
    blockers: list[str] = []
    if missing:
        blockers.append("not_all_target_seeds_have_clean_legal_support")
    if _float(dominant.get("share")) > max_dominant_requested_action_share:
        blockers.append("selected_support_set_dominant_requested_action_share_above_cap")
    return {
        "policy": "m3_carrion_survivor_continuation_v190_legal_support_audit_v1",
        "passed": not blockers,
        "blockers": blockers,
        "target_seed_count": len(target_seeds),
        "clean_legal_support_seed_count": sum(
            1 for count in clean_by_seed.values() if int(count) > 0
        ),
        "clean_terminal_survivors_by_seed": terminal_alive_by_seed,
        "clean_support_run_count_by_seed": clean_by_seed,
        "missing_clean_support_seeds": missing,
        "support_run_count": len(clean_runs),
        "support_runs": clean_runs,
        "support_trajectory_manifest": [
            {
                "seed": run.get("seed"),
                "branch_id": run.get("branch_id"),
                "continuation_script": run.get("continuation_script"),
                "path": run.get("trajectory_path"),
                "terminal_alive_agents": run.get("alive_agents"),
                "births": run.get("births"),
                "dominant_requested_action_share": run.get(
                    "dominant_requested_action_share"
                ),
            }
            for run in clean_runs
        ],
        "per_seed": per_seed,
        "run_audit_count": len(run_audits),
        "run_audits": run_audits,
        "dominant_requested_action": dominant.get("key"),
        "dominant_requested_action_count": dominant.get("count"),
        "dominant_requested_action_share": dominant.get("share"),
        "max_dominant_requested_action_share": _round(
            max_dominant_requested_action_share
        ),
        "future_trainable_payload_policy": (
            "current_public_observation_and_current_public_action_mask_only"
        ),
        "trainable_dataset_created": False,
    }


def support_run_legality_audit(
    run: Mapping[str, object],
    *,
    max_dominant_requested_action_share: float,
    replay_verification_required: bool,
) -> dict[str, object]:
    replay = _mapping(run.get("replay_verification"))
    replay_verified = (
        replay.get("verified") is True or run.get("replay_verified") is True
    )
    path = Path(str(run.get("trajectory_path") or ""))
    checks = {
        "terminal_alive_positive": _int(run.get("alive_agents")) > 0,
        "unsupported_requested_action_count_zero": (
            _int(run.get("unsupported_requested_action_count")) == 0
        ),
        "unsupported_resolved_action_count_zero": (
            _int(run.get("unsupported_resolved_action_count")) == 0
        ),
        "heuristic_action_source_count_zero": (
            _int(run.get("heuristic_action_source_count")) == 0
        ),
        "dominant_requested_action_share_within_cap": (
            _float(run.get("dominant_requested_action_share"))
            <= max_dominant_requested_action_share
        ),
        "deterministic_replay_verified": (
            replay_verified if replay_verification_required else True
        ),
        "trajectory_path_present": bool(str(run.get("trajectory_path") or "")),
    }
    precheck_failures = [name for name, passed in checks.items() if not passed]
    trajectory_audit: dict[str, object] | None = None
    if not precheck_failures:
        trajectory_audit = v189._audit_trajectory_path(path, run)
        checks.update(
            {
                "path_backed": trajectory_audit.get("readable") is True,
                "terminal_facts_match_manifest": (
                    trajectory_audit.get("terminal_facts_match_manifest") is True
                ),
                "action_mask_legality_passed": (
                    trajectory_audit.get("action_mask_legality_passed") is True
                ),
                "trajectory_invalid_counts_zero": (
                    trajectory_audit.get("trajectory_invalid_counts_zero") is True
                ),
                "trainable_leakage_scan_passed": (
                    trajectory_audit.get("trainable_leakage_scan_passed") is True
                ),
                "trajectory_dominant_requested_action_share_within_cap": (
                    _float(
                        trajectory_audit.get(
                            "dominant_requested_action_share_from_trajectory"
                        )
                    )
                    <= max_dominant_requested_action_share
                ),
            }
        )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v190_support_run_legality_audit_v1",
        "counts_as_clean_support": not failures,
        "failures": failures,
        "precheck_failures": precheck_failures,
        "seed": _int(run.get("seed")),
        "branch_id": run.get("branch_id"),
        "continuation_script": run.get("continuation_script"),
        "alive_agents": _int(run.get("alive_agents")),
        "births": _int(run.get("births")),
        "unsupported_requested_action_count": _int(
            run.get("unsupported_requested_action_count")
        ),
        "unsupported_resolved_action_count": _int(
            run.get("unsupported_resolved_action_count")
        ),
        "heuristic_action_source_count": _int(run.get("heuristic_action_source_count")),
        "dominant_requested_action": run.get("dominant_requested_action"),
        "dominant_requested_action_share": run.get(
            "dominant_requested_action_share"
        ),
        "replay_verified": replay_verified,
        "trajectory_path": str(path),
        "trajectory_audit": trajectory_audit,
        **checks,
    }


def support_run_payload(
    run: Mapping[str, object],
    audit: Mapping[str, object],
) -> dict[str, object]:
    trajectory = _mapping(audit.get("trajectory_audit"))
    return {
        "seed": _int(run.get("seed")),
        "fixture": run.get("fixture"),
        "branch_id": run.get("branch_id"),
        "branch_tick": _int(run.get("branch_tick")),
        "base_script": run.get("base_script"),
        "continuation_script": run.get("continuation_script"),
        "branch_state_digest": run.get("branch_state_digest"),
        "alive_agents": _int(run.get("alive_agents")),
        "births": _int(run.get("births")),
        "deaths": _int(run.get("deaths")),
        "trajectory_path": run.get("trajectory_path"),
        "replay_digest": run.get("replay_digest"),
        "replay_verified": audit.get("replay_verified") is True,
        "unsupported_requested_action_count": _int(
            run.get("unsupported_requested_action_count")
        ),
        "unsupported_resolved_action_count": _int(
            run.get("unsupported_resolved_action_count")
        ),
        "heuristic_action_source_count": _int(run.get("heuristic_action_source_count")),
        "dominant_requested_action": run.get("dominant_requested_action"),
        "dominant_requested_action_share": run.get(
            "dominant_requested_action_share"
        ),
        "requested_action_counts": trajectory.get("requested_action_counts", {}),
        "trainable_leakage_scan_passed": (
            trajectory.get("trainable_leakage_scan_passed") is True
        ),
    }


def per_seed_support_summary(
    *,
    seed: int,
    run_audits: Sequence[Mapping[str, object]],
    max_dominant_requested_action_share: float,
) -> dict[str, object]:
    seed_audits = [audit for audit in run_audits if _int(audit.get("seed")) == int(seed)]
    clean = [audit for audit in seed_audits if audit.get("counts_as_clean_support")]
    terminal = [audit for audit in seed_audits if _int(audit.get("alive_agents")) > 0]
    zero_unsupported = [
        audit
        for audit in seed_audits
        if _int(audit.get("unsupported_requested_action_count")) == 0
        and _int(audit.get("unsupported_resolved_action_count")) == 0
    ]
    within_share = [
        audit
        for audit in seed_audits
        if _float(audit.get("dominant_requested_action_share"))
        <= max_dominant_requested_action_share
    ]
    blockers: list[str] = []
    if not clean:
        blockers.append("no_clean_legal_support_run")
    if terminal and not clean:
        blockers.append("terminal_survivors_exist_only_as_rejected_attempts")
    return {
        "seed": int(seed),
        "branch_run_count": len(seed_audits),
        "terminal_survivor_attempt_count": len(terminal),
        "zero_unsupported_attempt_count": len(zero_unsupported),
        "within_action_share_cap_attempt_count": len(within_share),
        "clean_support_run_count": len(clean),
        "blockers": blockers,
        "best_attempt": best_attempt(seed_audits),
    }


def best_attempt(audits: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    if not audits:
        return None

    def key(audit: Mapping[str, object]) -> tuple[int, int, int, float, str]:
        return (
            _int(audit.get("alive_agents")),
            -_int(audit.get("unsupported_resolved_action_count")),
            -_int(audit.get("unsupported_requested_action_count")),
            -_float(audit.get("dominant_requested_action_share")),
            str(audit.get("branch_id") or ""),
        )

    selected = max(audits, key=key)
    return {
        "branch_id": selected.get("branch_id"),
        "continuation_script": selected.get("continuation_script"),
        "alive_agents": selected.get("alive_agents"),
        "births": selected.get("births"),
        "unsupported_requested_action_count": selected.get(
            "unsupported_requested_action_count"
        ),
        "unsupported_resolved_action_count": selected.get(
            "unsupported_resolved_action_count"
        ),
        "dominant_requested_action": selected.get("dominant_requested_action"),
        "dominant_requested_action_share": selected.get(
            "dominant_requested_action_share"
        ),
        "counts_as_clean_support": selected.get("counts_as_clean_support"),
        "failures": selected.get("failures"),
    }


def route_decision_audit(
    *,
    source_validation: Mapping[str, object],
    mined_evidence: Mapping[str, object],
    support_audit: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[str] = []
    if source_validation.get("passed") is not True:
        blockers.append("v189_source_validation_failed")
    if support_audit.get("passed") is not True:
        blockers.extend(str(item) for item in support_audit.get("blockers", []))
    if source_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif not blockers:
        route = SUCCESS_ROUTE
    else:
        route = BLOCKED_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v190_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "slice_3_training_authorized": False,
        "slice_3_training_allowed_for_this_command": False,
        "slice_3_training_consumed": False,
        "fresh_v191_dataset_audit_required_before_slice_3_training": (
            route == SUCCESS_ROUTE
        ),
        "runtime_integration_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "primary_blocker": blockers[0] if blockers else None,
        "clean_legal_support_seed_count": support_audit.get(
            "clean_legal_support_seed_count"
        ),
        "target_seed_count": support_audit.get("target_seed_count"),
        "dominant_requested_action": support_audit.get("dominant_requested_action"),
        "dominant_requested_action_share": support_audit.get(
            "dominant_requested_action_share"
        ),
        "v188_aggregate_attempts_rejected_as_support": (
            _mapping(mined_evidence.get("aggregate_attempted_continuations")).get(
                "rejected_as_support"
            )
            is True
        ),
        "rationale": route_rationale(route=route, blockers=blockers),
    }


def route_rationale(*, route: str, blockers: Sequence[str]) -> str:
    if route == STOP_ROUTE:
        return "Pinned v189 source evidence did not validate; stop without new work."
    if route == SUCCESS_ROUTE:
        return (
            "Targeted expansion produced clean legal terminal-survival support for "
            "all target seeds under the action-share cap. Route to a fresh v191 "
            "dataset audit before any slice-3 training."
        )
    return (
        "Targeted expansion did not produce clean legal terminal-survival support "
        "for all target seeds under the action-share cap. Slice 3 remains blocked; "
        "route to narrower legal-support repair or architecture review."
    )


def classification_for(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = (
        "m3_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_"
        "support_expansion_"
    )
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if route_decision.get("recommended_next_route") == SUCCESS_ROUTE:
        return prefix + "all_target_seed_support_found_routes_to_v191_audit_no_training"
    return prefix + "partial_or_empty_support_routes_to_repair_no_training"


def contract(
    *,
    expected_v189_report_exact_digest: str,
    required_v189_route: str,
    max_dominant_requested_action_share: float,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "targeted_support_expansion_only": True,
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "input_v189_report_exact_digest_pinned": expected_v189_report_exact_digest,
        "required_v189_route": required_v189_route,
        "expected_v189_backup": EXPECTED_V189_BACKUP,
        "expected_v189_backup_sha256": EXPECTED_V189_BACKUP_SHA256,
        "max_dominant_requested_action_share": _round(
            max_dominant_requested_action_share
        ),
        "success_route": SUCCESS_ROUTE,
        "blocked_route": BLOCKED_ROUTE,
        "forbidden_routes": [
            "slice_3_training",
            "v180_rerun",
            "v186_rerun",
            "runtime_integration",
            "promotion",
            "gate_relaxation",
        ],
    }


def lifecycle_flags(*, support_expansion_ran: bool) -> dict[str, object]:
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
        "support_generation_ran": bool(support_expansion_ran),
        "support_expansion_ran": bool(support_expansion_ran),
        "generic_carrion_autopsy_rerun": False,
        "v180_rerun": False,
        "v186_rerun": False,
        "v188_rerun": False,
        "v189_rerun": False,
        "non_promoted": True,
        "diagnostics_only": True,
    }


def dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return {"key": key, "count": int(count), "share": _round(count / total)}


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def _mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]
